#pragma once
// ---------------------------------------------------------------------------
// measured_bxdf.h -- Real-world measured BRDF (port of pbrt-v4 MeasuredBxDF)
//
// Mirrors pbrt-v4 MeasuredBxDF / MeasuredBxDFData (bxdfs.h + bxdfs.cpp,
// Apache-2.0).  Uses bilinear-warp importance sampling over tabulated BRDF
// data from the RGL measured BRDF database (Dupuy & Jakob 2018).
//
// What this provides
// ==================
//   MeasuredBRDFData       -- tabulated warp tables (ndf, sigma, vndf,
//                             luminance, spectra) built from raw float arrays
//   MeasuredBxDF<T>        -- f(), sample_f(), pdf() using the 5D spectral
//                             interpolant and 2D warp chain
//
// pbrt-v4 alignment
// =================
//   MeasuredBxDFData maps directly to pbrt-v4 MeasuredBxDFData:
//     ndf      -- PiecewiseLinear2D<0>  (NDF over theta_m space)
//     sigma    -- PiecewiseLinear2D<0>  (projected area, for normalization)
//     vndf     -- PiecewiseLinear2D<2>  (VNDF warp conditioned on phi_i, theta_i)
//     luminance-- PiecewiseLinear2D<2>  (luminance warp for spectral sampling)
//     spectra  -- PiecewiseLinear2D<3>  (spectral residual over (phi_i, theta_i, lambda))
//
//   MeasuredBxDF::f         mirrors pbrt-v4 MeasuredBxDF::f        (bxdfs.cpp line 999)
//   MeasuredBxDF::sample_f  mirrors pbrt-v4 MeasuredBxDF::Sample_f (bxdfs.cpp line 1036)
//   MeasuredBxDF::pdf       mirrors pbrt-v4 MeasuredBxDF::PDF       (bxdfs.cpp line 1087)
//
// Parameterization (unchanged from pbrt-v4):
//   theta -> u :  u    = sqrt(theta * 2/pi)     (nonlinear to concentrate near grazing)
//   u -> theta  :  theta = u^2 * pi/2
//   phi   -> u :  u    = phi / (2*pi) + 0.5
//   u -> phi    :  phi   = (2*u - 1) * pi
//
// Design rules (same as bxdfs.h / volume_scattering.h)
// =====================================================
//   - Header-only, no virtual functions, no heap allocation
//   - MeasuredBRDFData holds the tables by value (uses std::vector-backed PL2D)
//   - MeasuredBxDF holds a const pointer to MeasuredBRDFData + per-sample wavelengths
//   - f(), sample_f(), pdf() work in the local shading frame (z = normal)
//   - RGB spectral model: wavelengths[] stores {0=R, 1=G, 2=B} query values
//
// References
// ==========
//   pbrt-v4 src/pbrt/bxdfs.h + bxdfs.cpp  (Apache-2.0)
//   Dupuy & Jakob, "An Adaptive Parameterization for Efficient Material
//     Acquisition and Rendering", SIGGRAPH Asia 2018
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#  if defined(__CUDACC__)
#    define CPU_GPU __host__ __device__ __forceinline__
#  else
#    define CPU_GPU
#  endif
#endif

#include <cmath>
#include <algorithm>
#include <vector>

#include "piecewise_linear_2d.h"   // PiecewiseLinear2D<N>, PLSample

// ---------------------------------------------------------------------------
// Local geometry helpers (mirrors pbrt-v4 util/vecmath.h / scattering.h)
// They are defined here so measured_bxdf.h is self-contained without
// pulling in the full bxdfs.h header.
// ---------------------------------------------------------------------------
namespace mbxdf_detail {

static constexpr float kPi     = 3.14159265358979323846f;
static constexpr float kInvPi  = 1.0f / kPi;
static constexpr float kInv2Pi = 1.0f / (2.0f * kPi);

// CosTheta: z-component of unit vector in local frame (z = normal)
inline float CosTheta(float x, float y, float z) { return z; }
inline float AbsCosTheta(float x, float y, float z) { return std::fabs(z); }

// Spherical -> Cartesian
inline void SphericalDir(float sinTheta, float cosTheta, float phi,
						 float& wx, float& wy, float& wz) {
	wx = sinTheta * std::cos(phi);
	wy = sinTheta * std::sin(phi);
	wz = cosTheta;
}

// Cartesian -> spherical
inline float SphericalTheta(float x, float y, float z) {
	return std::acos(std::max(-1.0f, std::min(1.0f, z)));
}
inline float SphericalPhi(float x, float y) {
	return std::atan2(y, x);  // range [-pi, pi]
}

// Reflect wo across half-vector: wi = 2*(dot(wo,wm))*wm - wo
inline void ReflectAcross(float wox, float woy, float woz,
						   float wmx, float wmy, float wmz,
						   float& wix, float& wiy, float& wiz) {
	float d = 2.0f * (wox*wmx + woy*wmy + woz*wmz);
	wix = d*wmx - wox;
	wiy = d*wmy - woy;
	wiz = d*wmz - woz;
}

// SameHemisphere: both directions on same side of z=0 plane
inline bool SameHemisphere(float wox, float woy, float woz,
							 float wix, float wiy, float wiz) {
	return woz * wiz > 0.0f;
}

// Normalize 3D vector; returns false if degenerate
inline bool Normalize3(float& x, float& y, float& z) {
	float len2 = x*x + y*y + z*z;
	if (len2 == 0.0f) return false;
	float inv = 1.0f / std::sqrt(len2);
	x *= inv; y *= inv; z *= inv;
	return true;
}

// Parameterization (pbrt-v4 MeasuredBxDF private methods)
inline float theta2u(float theta) { return std::sqrt(theta * (2.0f * kInvPi)); }
inline float phi2u(float phi)     { return phi * kInv2Pi + 0.5f; }
inline float u2theta(float u)     { return u * u * (kPi * 0.5f); }
inline float u2phi(float u)       { return (2.0f * u - 1.0f) * kPi; }

} // namespace mbxdf_detail

// ---------------------------------------------------------------------------
// MeasuredBRDFData
//
// Holds all interpolation tables for one measured BRDF material.
// Mirrors pbrt-v4 MeasuredBxDFData (bxdfs.cpp lines 860-887).
//
// Fields
// ------
//   ndf       -- marginal NDF table, shape [ntheta_m, nphi_m],
//                 2D warping distribution over the unit square of (u_theta, u_phi)
//   sigma     -- projected area lookup, shape [ntheta_o, nphi_o]
//   vndf      -- VNDF warp table, conditioned on (phi_i, theta_i),
//                 shape [ntheta_m, nphi_m, nphi_i, ntheta_i]
//   luminance -- per-wavelength luminance warp for spectral sampling,
//                 conditioned on (phi_i, theta_i)
//   spectra   -- spectral residual, conditioned on (phi_i, theta_i, lambda)
//   wavelengths -- query wavelengths for the spectra table (e.g., {R, G, B} nm)
//   isotropic -- true if the BRDF is isotropic (phi dimension collapsed)
// ---------------------------------------------------------------------------
struct MeasuredBRDFData {
	PiecewiseLinear2D<0> ndf;        // NDF: P(u_wm) -- used for normalization
	PiecewiseLinear2D<0> sigma;      // projected area: sigma(u_wo)
	PiecewiseLinear2D<2> vndf;       // VNDF warp: Sample/Invert(u_wm | phi_i, theta_i)
	PiecewiseLinear2D<2> luminance;  // luminance warp (phi_i, theta_i)
	PiecewiseLinear2D<3> spectra;    // spectral residual (phi_i, theta_i, lambda)
	std::vector<float>   wavelengths; // query wavelengths for spectra axis
	bool                 isotropic = true;

	// -----------------------------------------------------------------------
	// Build from raw data arrays.  All arrays are row-major (y-major).
	//
	// ndf_data    [ntheta_m * nphi_m] -- normalized NDF values
	// sigma_data  [ntheta_o * nphi_o] -- projected area values
	// vndf_data   [nphi_i * ntheta_i * ntheta_m * nphi_m]
	// lum_data    [nphi_i * ntheta_i * ntheta_m * nphi_m]
	// spectra_data[nphi_i * ntheta_i * nwl * ntheta_m * nphi_m]
	// phi_i_vals, theta_i_vals, wl_vals -- parameter axis sample points
	// -----------------------------------------------------------------------
	void Build(// NDF table
			   const float* ndf_data,    int ndf_nx,    int ndf_ny,
			   // sigma table
			   const float* sigma_data,  int sigma_nx,  int sigma_ny,
			   // VNDF table (conditioned on phi_i, theta_i)
			   const float* vndf_data,   int vndf_nx,   int vndf_ny,
			   int nphi_i, const float* phi_i_vals,
			   int ntheta_i, const float* theta_i_vals,
			   // luminance table (same conditioning as vndf)
			   const float* lum_data,
			   // spectra table (conditioned on phi_i, theta_i, lambda)
			   const float* spectra_data,
			   int nwl, const float* wl_vals,
			   bool is_isotropic = true)
	{
		isotropic = is_isotropic;

		// NDF and sigma have no parameter conditioning
		ndf   = PiecewiseLinear2D<0>(ndf_data,   ndf_nx,   ndf_ny,   true, false);
		sigma = PiecewiseLinear2D<0>(sigma_data,  sigma_nx, sigma_ny, true, false);

		// VNDF and luminance: conditioned on (phi_i, theta_i)
		const int    vndf_res[2]    = { nphi_i, ntheta_i };
		const float* vndf_pvals[2]  = { phi_i_vals, theta_i_vals };
		vndf      = PiecewiseLinear2D<2>(vndf_data, vndf_nx, vndf_ny,
										 vndf_res, vndf_pvals, true, true);
		luminance = PiecewiseLinear2D<2>(lum_data,  vndf_nx, vndf_ny,
										 vndf_res, vndf_pvals, true, true);

		// spectra: conditioned on (phi_i, theta_i, lambda)
		const int    spec_res[3]    = { nphi_i, ntheta_i, nwl };
		const float* spec_pvals[3]  = { phi_i_vals, theta_i_vals, wl_vals };
		spectra = PiecewiseLinear2D<3>(spectra_data, vndf_nx, vndf_ny,
									   spec_res, spec_pvals, false, false);

		wavelengths.assign(wl_vals, wl_vals + nwl);
	}
};

// ---------------------------------------------------------------------------
// MeasuredBxDF<T>
//
// Evaluates a measured BRDF for given (wo, wi) directions in the local shading
// frame (z = shading normal).  Carries per-sample spectral query wavelengths.
//
// Mirrors pbrt-v4 MeasuredBxDF (bxdfs.h + bxdfs.cpp).
//
// Template parameter T: float or double (double used in tests/CPU path).
//   Internal PL2D tables are always float; T is used for the in/out directions
//   and returned BRDF values.
//
// Usage
// -----
//   MeasuredBxDF<float> bxdf(&data, lambda_r, lambda_g, lambda_b);
//   // directions in local shading frame (z = normal):
//   float fr, fg, fb;
//   bxdf.f(wo_x, wo_y, wo_z,  wi_x, wi_y, wi_z,  fr, fg, fb);
//   float pdf = bxdf.pdf(wo_x,..,  wi_x,..);
//   float swi_x,..,swi_z, spdf;
//   bool ok = bxdf.sample_f(wo_x,.., u0, u1, swi_x,..,swi_z, fr, fg, fb, spdf);
// ---------------------------------------------------------------------------
template <typename T>
struct MeasuredBxDF {

	const MeasuredBRDFData* data = nullptr;
	// Per-wavelength query values for the 3-channel spectral interpolant.
	// Index 0 = R, 1 = G, 2 = B.  Store as float for PL2D compatibility.
	float lambda[3] = { 0.0f, 0.0f, 0.0f };

	CPU_GPU MeasuredBxDF() = default;

	CPU_GPU MeasuredBxDF(const MeasuredBRDFData* d,
						 float lambda_r, float lambda_g, float lambda_b)
		: data(d) {
		lambda[0] = lambda_r;
		lambda[1] = lambda_g;
		lambda[2] = lambda_b;
	}

	// ------------------------------------------------------------------
	// f(wo, wi) -- BRDF evaluation
	//
	// Mirrors pbrt-v4 MeasuredBxDF::f (bxdfs.cpp lines 999-1033).
	//
	// Algorithm:
	//   1. Reject if not same hemisphere (return 0)
	//   2. Flip if below z=0 (two-sided)
	//   3. Compute half-vector wm = normalize(wo + wi)
	//   4. Map wo and wm to unit-square coords (u_wo, u_wm) using
	//      theta2u / phi2u parameterization
	//   5. For isotropic BRDF: use relative phi: phi_m - phi_o
	//   6. Invert VNDF warp at u_wm given (phi_o, theta_o) -> uniform sample ui
	//   7. Evaluate 5D spectral interpolant at (ui, phi_o, theta_o, lambda[c])
	//   8. Return spectra * ndf(u_wm) / (4 * sigma(u_wo) * cosTheta(wi))
	// ------------------------------------------------------------------
	CPU_GPU void f(T wox, T woy, T woz,
				   T wix, T wiy, T wiz,
				   T& fr, T& fg, T& fb) const
	{
		fr = fg = fb = T(0);
		if (!data) return;

		// Same hemisphere check (z-components must have same sign)
		if (woz * wiz <= T(0)) return;

		// Flip to upper hemisphere for two-sided support
		bool flip = (woz < T(0));
		if (flip) { wox=-wox; woy=-woy; woz=-woz; wix=-wix; wiy=-wiy; wiz=-wiz; }

		// Half-direction vector wm = normalize(wi + wo)
		float wmx = float(wix + wox);
		float wmy = float(wiy + woy);
		float wmz = float(wiz + woz);
		if (!mbxdf_detail::Normalize3(wmx, wmy, wmz)) return;

		// Spherical coordinates of wo and wm
		float theta_o = mbxdf_detail::SphericalTheta(float(wox), float(woy), float(woz));
		float phi_o   = mbxdf_detail::SphericalPhi(float(wox), float(woy));
		float theta_m = mbxdf_detail::SphericalTheta(wmx, wmy, wmz);
		float phi_m   = mbxdf_detail::SphericalPhi(wmx, wmy);

		// Unit-square coordinates
		float u_wo_x = mbxdf_detail::theta2u(theta_o);
		float u_wo_y = mbxdf_detail::phi2u(phi_o);
		float u_wm_x = mbxdf_detail::theta2u(theta_m);
		// Isotropic: use relative phi; anisotropic: use absolute phi_m
		float u_wm_y = mbxdf_detail::phi2u(data->isotropic ? (phi_m - phi_o) : phi_m);
		// Wrap u_wm_y to [0,1)
		u_wm_y = u_wm_y - std::floor(u_wm_y);

		// Invert VNDF warp: (u_wm | phi_o, theta_o) -> uniform sample ui
		PLSample ui = data->vndf.Invert(u_wm_x, u_wm_y, phi_o, theta_o);

		// Evaluate spectral interpolant for each channel
		float val[3];
		for (int c = 0; c < 3; ++c)
			val[c] = std::max(0.0f, data->spectra.Eval(ui.px, ui.py,
														phi_o, theta_o, lambda[c]));

		// Normalization: ndf(u_wm) / (4 * sigma(u_wo) * |cos(theta_wi)|)
		float ndf_val   = data->ndf.Eval(u_wm_x, u_wm_y);
		float sigma_val = data->sigma.Eval(u_wo_x, u_wo_y);
		float cos_wi    = mbxdf_detail::AbsCosTheta(float(wix), float(wiy), float(wiz));
		float denom = 4.0f * sigma_val * cos_wi;
		if (denom == 0.0f) return;

		float scale = ndf_val / denom;
		fr = T(val[0] * scale);
		fg = T(val[1] * scale);
		fb = T(val[2] * scale);
	}

	// ------------------------------------------------------------------
	// sample_f -- importance-sample the measured BRDF
	//
	// Mirrors pbrt-v4 MeasuredBxDF::Sample_f (bxdfs.cpp lines 1036-1085).
	//
	// Returns false if the sample is invalid (transmitted or degenerate).
	// On success: wi_x/y/z is the sampled incident direction, fr/fg/fb is
	// the BRDF weight (f * |cos(theta_i)| / pdf), and pdf_out is the sampling PDF.
	//
	// Algorithm:
	//   1. Flip wo to upper hemisphere if needed
	//   2. Compute theta_o, phi_o
	//   3. Warp u through luminance: (u0,u1) -> lum warp -> u' + lum_pdf
	//   4. Warp u' through VNDF: (u0',u1') -> (u_wm, vndf_pdf)
	//   5. Map u_wm -> (phi_m, theta_m) -> wm direction
	//   6. wi = Reflect(wo, wm); reject if wi below horizon
	//   7. Evaluate spectra at (u_wm, phi_o, theta_o, lambda)
	//   8. Scale by ndf/sigma; compute full PDF; flip wi back if needed
	// ------------------------------------------------------------------
	CPU_GPU bool sample_f(T wox, T woy, T woz,
						  float u0, float u1,
						  T& wix, T& wiy, T& wiz,
						  T& fr,  T& fg,  T& fb,
						  T& pdf_out) const
	{
		fr = fg = fb = T(0);
		pdf_out = T(0);
		if (!data) return false;

		bool flip = (woz < T(0));
		if (flip) { wox=-wox; woy=-woy; woz=-woz; }
		if (woz <= T(0)) return false;

		float theta_o = mbxdf_detail::SphericalTheta(float(wox), float(woy), float(woz));
		float phi_o   = mbxdf_detail::SphericalPhi(float(wox), float(woy));

		// Step 1: warp through luminance distribution
		PLSample ls = data->luminance.Sample(u0, u1, phi_o, theta_o);
		u0 = ls.px; u1 = ls.py;
		float lum_pdf = ls.pdf;
		if (lum_pdf == 0.0f) return false;

		// Step 2: warp through VNDF
		PLSample vs = data->vndf.Sample(u0, u1, phi_o, theta_o);
		float u_wm_x = vs.px, u_wm_y = vs.py;
		float vndf_pdf = vs.pdf;
		if (vndf_pdf == 0.0f) return false;

		// Step 3: map to half-vector direction
		float phi_m   = mbxdf_detail::u2phi(u_wm_y);
		float theta_m = mbxdf_detail::u2theta(u_wm_x);
		if (data->isotropic) phi_m += phi_o;
		float sinTheta_m = std::sin(theta_m);
		float cosTheta_m = std::cos(theta_m);
		float wmx, wmy, wmz;
		mbxdf_detail::SphericalDir(sinTheta_m, cosTheta_m, phi_m, wmx, wmy, wmz);

		// Step 4: compute wi = Reflect(wo, wm)
		//   wi = 2*dot(wo,wm)*wm - wo
		float dot_wo_wm_s = float(wox)*wmx + float(woy)*wmy + float(woz)*wmz;
		float wi_x = 2.0f * dot_wo_wm_s * wmx - float(wox);
		float wi_y = 2.0f * dot_wo_wm_s * wmy - float(woy);
		float wi_z = 2.0f * dot_wo_wm_s * wmz - float(woz);
		if (wi_z <= 0.0f) return false;

		// Step 5: evaluate spectral interpolant at (u_wm, phi_o, theta_o, lambda)
		float val[3];
		for (int c = 0; c < 3; ++c)
			val[c] = std::max(0.0f, data->spectra.Eval(u_wm_x, u_wm_y,
														phi_o, theta_o, lambda[c]));

		// Step 6: BRDF normalization
		float u_wo_x    = mbxdf_detail::theta2u(theta_o);
		float u_wo_y    = mbxdf_detail::phi2u(phi_o);
		float ndf_val   = data->ndf.Eval(u_wm_x, u_wm_y);
		float sigma_val = data->sigma.Eval(u_wo_x, u_wo_y);
		float abs_cos_wi = std::fabs(wi_z);
		float denom = 4.0f * sigma_val * abs_cos_wi;
		if (denom == 0.0f) return false;
		float scale = ndf_val / denom;

		// Step 7: Jacobian of half-vector parameterization
		//   |d wm / d wi| = 1/(4 * dot(wo,wm))
		//   PL2D jacobian = 2*pi^2 * u_wm_x * sinTheta_m  (from parameterization)
		//   (pbrt-v4: 4 * dot(wo,wm) * max(2*pi^2 * u_wm_x * sinTheta_m, 1e-6))
		float jacobian  = 4.0f * dot_wo_wm_s *
						  std::max(2.0f * mbxdf_detail::kPi * mbxdf_detail::kPi
								   * u_wm_x * sinTheta_m, 1e-6f);
		if (jacobian == 0.0f) return false;

		float final_pdf = vndf_pdf * lum_pdf / jacobian;
		if (final_pdf == 0.0f) return false;

		wix = T(wi_x); wiy = T(wi_y); wiz = T(wi_z);
		if (flip) { wix=-wix; wiy=-wiy; wiz=-wiz; }

		fr  = T(val[0] * scale);
		fg  = T(val[1] * scale);
		fb  = T(val[2] * scale);
		pdf_out = T(final_pdf);
		return true;
	}

	// ------------------------------------------------------------------
	// pdf(wo, wi) -- sampling PDF for the BRDF
	//
	// Mirrors pbrt-v4 MeasuredBxDF::PDF (bxdfs.cpp lines 1087-1120).
	//
	// = (vndf.Invert(u_wm | phi_o, theta_o).pdf * luminance.Eval(u | phi_o, theta_o))
	//   / jacobian
	// where jacobian = 4 * dot(wo,wm) * max(2*pi^2 * u_wm_x * sinTheta_m, 1e-6)
	// ------------------------------------------------------------------
	CPU_GPU T pdf(T wox, T woy, T woz,
				  T wix, T wiy, T wiz) const
	{
		if (!data) return T(0);
		if (woz * wiz <= T(0)) return T(0);
		if (woz < T(0)) { wox=-wox; woy=-woy; woz=-woz; wix=-wix; wiy=-wiy; wiz=-wiz; }

		// Half-vector
		float wmx = float(wix + wox);
		float wmy = float(wiy + woy);
		float wmz = float(wiz + woz);
		if (!mbxdf_detail::Normalize3(wmx, wmy, wmz)) return T(0);

		float theta_o = mbxdf_detail::SphericalTheta(float(wox), float(woy), float(woz));
		float phi_o   = mbxdf_detail::SphericalPhi(float(wox), float(woy));
		float theta_m = mbxdf_detail::SphericalTheta(wmx, wmy, wmz);
		float phi_m   = mbxdf_detail::SphericalPhi(wmx, wmy);

		float u_wm_x = mbxdf_detail::theta2u(theta_m);
		float u_wm_y = mbxdf_detail::phi2u(data->isotropic ? (phi_m - phi_o) : phi_m);
		u_wm_y = u_wm_y - std::floor(u_wm_y);

		// Invert VNDF warp -> recover the uniform sample ui and vndf pdf
		PLSample ui = data->vndf.Invert(u_wm_x, u_wm_y, phi_o, theta_o);

		// Luminance density at ui
		float lum = data->luminance.Eval(ui.px, ui.py, phi_o, theta_o);

		// Jacobian
		float sinTheta_m = std::sqrt(std::max(0.0f, 1.0f - wmz*wmz));
		float dot_wo_wm  = float(wox)*wmx + float(woy)*wmy + float(woz)*wmz;
		float jacobian   = 4.0f * dot_wo_wm *
						   std::max(2.0f * mbxdf_detail::kPi * mbxdf_detail::kPi
									* u_wm_x * sinTheta_m, 1e-6f);
		if (jacobian == 0.0f) return T(0);

		return T(ui.pdf * lum / jacobian);
	}
};
