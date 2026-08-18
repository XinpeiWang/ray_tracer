#pragma once
// ---------------------------------------------------------------------------
// volume_scattering.h -- Shared CPU/GPU volumetric scattering math
//
// Mirrors pbrt-v4 HGPhaseFunction (media.h), HenyeyGreenstein and
// SampleHenyeyGreenstein (util/scattering.h, util/sampling.cpp),
// and HomogeneousMedium (media.h).
//
// Design rules (same as bxdfs.h):
//   - Plain template structs, CPU_GPU tagged
//   - No virtual functions, no heap allocation
//   - T = double on CPU, float on GPU
//
// Components:
//   HenyeyGreensteinPhaseFunction<T>  -- anisotropic phase function (g in [-1,1])
//   HomogeneousMediumData<T>          -- sigma_a, sigma_s, g, transmittance, delta-tracking
// ---------------------------------------------------------------------------

#include "cpu_gpu.h"
#include "scalar_math.h"

#if defined(__CUDACC__)
#   include <math_functions.h>
#else
#   include <cmath>
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace vol_detail {

template<typename T>
CPU_GPU T safe_sqrt(T x) {
	return SafeSqrt(x);
}

template<typename T>
CPU_GPU T clamp(T x, T lo, T hi) {
	return x < lo ? lo : (x > hi ? hi : x);
}

template<typename T>
CPU_GPU T sqr(T x) { return x * x; }

// Build an orthonormal frame from a unit vector z (pbrt-v4 Frame::FromZ)
template<typename T>
CPU_GPU void frame_from_z(T zx, T zy, T zz,
						   T& xx, T& xy, T& xz,
						   T& yx, T& yy, T& yz) {
	// Pixar technique (Duff et al. 2017)
	T sign = (zz >= T(0)) ? T(1) : T(-1);
	T a = T(-1) / (sign + zz);
	T b = zx * zy * a;
	xx = T(1) + sign * sqr(zx) * a;
	xy = sign * b;
	xz = -sign * zx;
	yx = b;
	yy = sign + sqr(zy) * a;
	yz = -zy;
}

} // namespace vol_detail

// ---------------------------------------------------------------------------
// 1. HenyeyGreensteinPhaseFunction<T>
//
// pbrt-v4 reference: HGPhaseFunction (media.h), HenyeyGreenstein and
// SampleHenyeyGreenstein (util/scattering.h, util/sampling.cpp).
//
// p(cosTheta, g) = (1/(4pi)) * (1 - g^2) / (1 + g^2 + 2g*cosTheta)^(3/2)
//
// g =  0  : isotropic (recovers old constant_medium behavior)
// g >  0  : forward scattering
// g <  0  : back scattering
// ---------------------------------------------------------------------------
template<typename T>
struct HenyeyGreensteinPhaseFunction {
	T g;  // asymmetry parameter, clamped to (-0.99, 0.99) on eval/sample

	CPU_GPU HenyeyGreensteinPhaseFunction() : g(T(0)) {}
	CPU_GPU explicit HenyeyGreensteinPhaseFunction(T g_) : g(g_) {}

	// Phase function value p(wo, wi)
	// pbrt-v4: HenyeyGreenstein(Dot(wo,wi), g)
	CPU_GPU T p(T cos_theta) const {
		T gc = vol_detail::clamp(g, T(-0.99), T(0.99));
		const T inv4pi = T(1) / (T(4) * T(3.14159265358979323846));
		T denom = T(1) + vol_detail::sqr(gc) + T(2) * gc * cos_theta;
		return inv4pi * (T(1) - vol_detail::sqr(gc)) /
			   (denom * vol_detail::safe_sqrt(denom));
	}

	// Sample a new direction wi given incoming wo (world space unit vectors)
	// Returns: sampled wi (world space), writes out the PDF value.
	// pbrt-v4: SampleHenyeyGreenstein (util/sampling.cpp lines 348-375)
	//
	// Inputs:
	//   wo_x/y/z  -- outgoing direction (from surface toward camera), unit vector
	//   u1, u2    -- uniform random numbers in [0,1)
	//   pdf_out   -- receives p(wo, wi) for the sampled direction
	//   wi_x/y/z  -- receives sampled incoming direction
	CPU_GPU void sample(T wo_x, T wo_y, T wo_z,
						T u1, T u2,
						T& wi_x, T& wi_y, T& wi_z,
						T& pdf_out) const {
		T gc = vol_detail::clamp(g, T(-0.99), T(0.99));

		// Compute cosTheta for the HG sample (pbrt-v4 SampleHenyeyGreenstein)
		T cos_theta;
		if (std::abs(gc) < T(1e-3)) {
			cos_theta = T(1) - T(2) * u1;
		} else {
			T sq = (T(1) - vol_detail::sqr(gc)) / (T(1) + gc - T(2) * gc * u1);
			cos_theta = -T(1) / (T(2) * gc) * (T(1) + vol_detail::sqr(gc) - vol_detail::sqr(sq));
		}

		T sin_theta = vol_detail::safe_sqrt(T(1) - vol_detail::sqr(cos_theta));
		T phi = T(2) * T(3.14159265358979323846) * u2;

#if defined(__CUDACC__)
		T sin_phi = sinf(phi), cos_phi = cosf(phi);
#else
		T sin_phi = std::sin(phi), cos_phi = std::cos(phi);
#endif

		// Local frame from wo (pbrt-v4 Frame::FromZ(wo))
		T xx, xy, xz, yx, yy, yz;
		vol_detail::frame_from_z(wo_x, wo_y, wo_z, xx, xy, xz, yx, yy, yz);

		// SphericalDirection in local frame then transform to world
		// Local: (sin*cos_phi, sin*sin_phi, cos_theta)  -- pbrt-v4 SphericalDirection
		T lx = sin_theta * cos_phi;
		T ly = sin_theta * sin_phi;
		T lz = cos_theta;

		// wi = lx * x_axis + ly * y_axis + lz * wo
		wi_x = lx * xx + ly * yx + lz * wo_x;
		wi_y = lx * xy + ly * yy + lz * wo_y;
		wi_z = lx * xz + ly * yz + lz * wo_z;

		pdf_out = p(cos_theta);
	}

	// PDF for a given (wo, wi) pair
	CPU_GPU T pdf(T wo_x, T wo_y, T wo_z,
				  T wi_x, T wi_y, T wi_z) const {
		T cos_theta = wo_x*wi_x + wo_y*wi_y + wo_z*wi_z;
		return p(cos_theta);
	}
};

// ---------------------------------------------------------------------------
// 2. HomogeneousMediumData<T>
//
// pbrt-v4 reference: HomogeneousMedium (media.h)
// Parameters:
//   sigma_a  -- absorption coefficient (RGB or scalar)
//   sigma_s  -- scattering coefficient (RGB or scalar)
//   g        -- HG asymmetry parameter
//
// Key operations:
//   sigma_t()  -- extinction = sigma_a + sigma_s
//   albedo()   -- single-scattering albedo = sigma_s / sigma_t
//   transmittance(t) -- Beer-Lambert: exp(-sigma_t * t)
//   sample_free_path(u) -- delta-tracking free-path length: -ln(u)/sigma_t
//
// RGB variant: uses per-channel extinction; for free-path sampling we use the
// average sigma_t as the majorant (matches pbrt-v4 HomogeneousMajorantIterator
// approach: sigma_maj = sigma_a + sigma_s, single majorant for all channels).
// ---------------------------------------------------------------------------
template<typename T>
struct HomogeneousMediumData {
	T sigma_ar, sigma_ag, sigma_ab;  // absorption (RGB)
	T sigma_sr, sigma_sg, sigma_sb;  // scattering (RGB)
	T g;                             // HG asymmetry

	CPU_GPU HomogeneousMediumData() : sigma_ar(0), sigma_ag(0), sigma_ab(0),
									   sigma_sr(0), sigma_sg(0), sigma_sb(0),
									   g(0) {}

	// Scalar constructor: uniform RGB coefficients
	CPU_GPU HomogeneousMediumData(T sa, T ss, T g_)
		: sigma_ar(sa), sigma_ag(sa), sigma_ab(sa),
		  sigma_sr(ss), sigma_sg(ss), sigma_sb(ss),
		  g(g_) {}

	// RGB constructor
	CPU_GPU HomogeneousMediumData(T sar, T sag, T sab,
								   T ssr, T ssg, T ssb, T g_)
		: sigma_ar(sar), sigma_ag(sag), sigma_ab(sab),
		  sigma_sr(ssr), sigma_sg(ssg), sigma_sb(ssb),
		  g(g_) {}

	// Extinction per channel
	CPU_GPU T sigma_tr() const { return sigma_ar + sigma_sr; }
	CPU_GPU T sigma_tg() const { return sigma_ag + sigma_sg; }
	CPU_GPU T sigma_tb() const { return sigma_ab + sigma_sb; }

	// Average extinction (used as majorant for delta tracking)
	CPU_GPU T sigma_t_avg() const {
		return (sigma_tr() + sigma_tg() + sigma_tb()) / T(3);
	}

	// Beer-Lambert transmittance: Tr(t) = exp(-sigma_t * t), per channel
	// pbrt-v4: T(t) = exp(-sigma_t * t)
	CPU_GPU void transmittance(T t, T& Tr_r, T& Tr_g, T& Tr_b) const {
#if defined(__CUDACC__)
		Tr_r = expf(-sigma_tr() * t);
		Tr_g = expf(-sigma_tg() * t);
		Tr_b = expf(-sigma_tb() * t);
#else
		Tr_r = std::exp(-sigma_tr() * t);
		Tr_g = std::exp(-sigma_tg() * t);
		Tr_b = std::exp(-sigma_tb() * t);
#endif
	}

	// Sample a free-path distance using delta tracking with average sigma_t as majorant.
	// pbrt-v4: t = -log(1-u) / sigma_maj  (equivalent to -log(u)/sigma_t for u uniform)
	// Returns the sampled distance; caller checks against medium extent.
	CPU_GPU T sample_free_path(T u) const {
		T st = sigma_t_avg();
		if (st <= T(0)) return T(1e30);
#if defined(__CUDACC__)
		return -logf(T(1) - u) / st;
#else
		return -std::log(T(1) - u) / st;
#endif
	}

	// Weight for scattering event at distance t:
	// w = sigma_s / sigma_t  (per channel, the single-scattering albedo)
	// pbrt-v4: pathWeight = sigma_s / sigma_t at the scattering point
	CPU_GPU void scatter_weight(T& wr, T& wg, T& wb) const {
		T denom_r = sigma_tr(), denom_g = sigma_tg(), denom_b = sigma_tb();
		wr = (denom_r > T(0)) ? sigma_sr / denom_r : T(0);
		wg = (denom_g > T(0)) ? sigma_sg / denom_g : T(0);
		wb = (denom_b > T(0)) ? sigma_sb / denom_b : T(0);
	}

	// Transmittance weight for a ray that exits without scattering at distance d:
	// w = Tr(d) / p_exit  where p_exit = Tr(d) (for homogeneous medium)
	// => ratio = 1.0  (cancels out, standard ratio-tracking result)
	// pbrt-v4: pathWeight for null interaction = 1 after ratio tracking
	CPU_GPU void exit_weight(T /*d*/, T& wr, T& wg, T& wb) const {
		wr = wg = wb = T(1);
	}

	HenyeyGreensteinPhaseFunction<T> phase_function() const {
		return HenyeyGreensteinPhaseFunction<T>(g);
	}
};
