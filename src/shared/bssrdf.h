#pragma once
// ---------------------------------------------------------------------------
// bssrdf.h -- Tabulated BSSRDF (subsurface scattering) -- pbrt-v4 port
//
// Mirrors pbrt-v4 bssrdf.h/bssrdf.cpp (BSSRDFTable, ComputeBeamDiffusionBSSRDF,
// TabulatedBSSRDF) adapted to use our double-precision CPU helpers and the
// shared Catmull-Rom/sampling utilities already present in sampling.h.
//
// Key simplifications relative to pbrt-v4:
//   - Single spectral channel (monochromatic) so SampledSpectrum<N> becomes
//     a plain array passed as a pointer.
//   - No parallel-for: serial loop in ComputeBeamDiffusionBSSRDF (fine for
//     offline use and test coverage).
//   - No Allocator: uses std::vector for the table arrays.
//   - TabulatedBSSRDF exposes Sr(), SampleSr(), PDF_Sr() per channel.
//   - SubsurfaceFromDiffuse() maps (rhoEff, mfp) -> (sigma_a, sigma_s).
//
// Design rules (same as volume_scattering.h / bxdfs.h):
//   - Plain structs/classes, CPU_GPU tagged where sensible (table builder is CPU-only)
//   - No virtual functions, no heap-allocated polymorphism
//
// References:
//   pbrt-v4 src/pbrt/bssrdf.h  (Apache-2.0)
//   pbrt-v4 src/pbrt/bssrdf.cpp (Apache-2.0)
//   Donner & Jensen, "Light Diffusion in Multi-Layered Translucent Materials", 2005
//   Grosjean, "A high accuracy approximation for the transport equation", 1956
// ---------------------------------------------------------------------------

#include "cpu_gpu.h"
#include "scalar_math.h"

#include "sampling.h"   // catmull_rom_weights, integrate_catmull_rom, SampleCatmullRom2D, InvertCatmullRom
#include "fresnel.h"    // FresnelMoment1, FrDielectric

#include <cmath>
#include <vector>
#include <algorithm>

// ===========================================================================
// Helpers shared with pbrt-v4 internals
// ===========================================================================

namespace bssrdf_detail {

static const double kPi    = 3.14159265358979323846;
static const double kInv4Pi = 1.0 / (4.0 * kPi);

inline double fast_exp(double x) { return std::exp(x); }
inline double safe_sqrt(double x) { return SafeSqrt(x); }
inline double sqr(double x) { return x * x; }
inline double pow3(double x) { return x * x * x; }

// FrDielectric: compute Fresnel reflectance for a real dielectric interface.
// cosTheta_i may be negative (entering from below). Clamps to [0,1].
// pbrt-v4 util/scattering.h FrDielectric
inline double fr_dielectric(double cos_theta_i, double eta) {
	cos_theta_i = std::max(-1.0, std::min(1.0, cos_theta_i));
	if (cos_theta_i < 0.0) { eta = 1.0 / eta; cos_theta_i = -cos_theta_i; }
	double sin2_theta_t = (1.0 - cos_theta_i * cos_theta_i) / (eta * eta);
	if (sin2_theta_t >= 1.0) return 1.0;
	double cos_theta_t = safe_sqrt(1.0 - sin2_theta_t);
	double r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
	double r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
	return (r_parl * r_parl + r_perp * r_perp) * 0.5;
}

// FresnelMoment2 -- second Fresnel moment integral (pbrt-v4 scattering.cpp)
// Polynomial fit in two branches.
inline double fresnel_moment2(double eta) {
	double eta2 = eta*eta, eta3 = eta2*eta, eta4 = eta3*eta, eta5 = eta4*eta;
	if (eta < 1.0)
		return 0.27614 - 0.87350*eta + 1.12077*eta2 - 0.65095*eta3 + 0.07883*eta4 + 0.04942*eta5;
	else
		return -9.23372 + 22.2272*eta - 20.9292*eta2 + 10.2291*eta3 - 2.54396*eta4 + 0.254913*eta5;
}

// Henyey-Greenstein phase function evaluation (cosTheta = dot(wo, wi))
// pbrt-v4 HenyeyGreenstein
inline double henyey_greenstein(double cos_theta, double g) {
	double denom = 1.0 + g*g + 2.0*g*cos_theta;
	return (1.0 / (4.0 * kPi)) * (1.0 - g*g) / (denom * safe_sqrt(denom));
}

// SampleExponential: invert CDF of exponential: -log(1-u)/sigma_t
// for stratified sample (i+0.5)/n, avoiding log(0)
inline double sample_exponential(double u, double sigma_t) {
	return -std::log(std::max(1e-300, 1.0 - u)) / sigma_t;
}

// InvertCatmullRom -- find x such that f(x) = u, where f is a Catmull-Rom spline.
// pbrt-v4: InvertCatmullRom (util/math.cpp).
//
// u is a VALUE in the range of f (not a normalized fraction).
// Returns nodes[0] if u <= f[0], nodes[n-1] if u >= f[n-1].
// Inverts using Newton-bisection within the identified segment.
//
// Used by SubsurfaceFromDiffuse to invert the rhoEff(rho) curve.
inline double invert_catmull_rom(const double* nodes, const double* f, int n, double u) {
	// Boundary clamp (pbrt-v4: stop when u is out of bounds)
	if (!(u > f[0]))   return nodes[0];
	if (!(u < f[n-1])) return nodes[n-1];

	// Find interval i such that f[i] <= u < f[i+1]
	int i = 0;
	for (int k = 0; k < n - 1; ++k) { if (f[k] <= u) i = k; else break; }

	// Look up x_i, x_{i+1} and function values of spline segment i
	double x0 = nodes[i], x1 = nodes[i + 1];
	double f0 = f[i],     f1 = f[i + 1];
	double width = x1 - x0;

	// Approximate derivatives using finite differences (pbrt-v4 exact match)
	double d0 = (i > 0)     ? width * (f1 - f[i-1]) / (x1 - nodes[i-1]) : (f1 - f0);
	double d1 = (i+2 < n)   ? width * (f[i+2] - f0) / (nodes[i+2] - x0) : (f1 - f0);

	// Newton-bisection: solve Fhat(t) - u = 0 where Fhat is the cubic Hermite
	auto eval = [&](double t) -> std::pair<double, double> {
		double t2 = t*t, t3 = t2*t;
		// Fhat: cubic Hermite evaluated at t (pbrt-v4 formula)
		double Fhat = (2*t3 - 3*t2 + 1)*f0 + (-2*t3 + 3*t2)*f1
					+ (t3 - 2*t2 + t)*d0   + (t3 - t2)*d1;
		// fhat: derivative dFhat/dt
		double fhat = (6*t2 - 6*t)*f0 + (-6*t2 + 6*t)*f1
					+ (3*t2 - 4*t + 1)*d0 + (3*t2 - 2*t)*d1;
		return {Fhat - u, fhat};
	};
	double t = catmullrom_newton_bisection(0.0, 1.0, eval);
	return x0 + t * width;
}

} // namespace bssrdf_detail

// ===========================================================================
// BeamDiffusionSS -- single-scattering contribution to BSSRDF profile
// pbrt-v4: BeamDiffusionSS (bssrdf.cpp)
// Integrates over depth t the product of:
//   - exponential free path exp(-sigma_t*(d+tCrit))
//   - HG phase function for the turn angle
//   - Fresnel transmission at the exit interface
// Result: the single-scattering contribution to the radial profile R(r).
// ===========================================================================
inline double BeamDiffusionSS(double sigma_s, double sigma_a, double g, double eta, double r) {
	using namespace bssrdf_detail;
	double sigma_t = sigma_a + sigma_s;
	double rho     = sigma_s / sigma_t;
	double t_crit  = r * safe_sqrt(eta*eta - 1.0);

	const int n = 100;
	double Ess = 0.0;
	for (int i = 0; i < n; ++i) {
		double ti = t_crit + sample_exponential((i + 0.5) / n, sigma_t);
		double d  = std::sqrt(sqr(r) + sqr(ti));
		double cos_theta_o = ti / d;
		Ess += rho * fast_exp(-sigma_t * (d + t_crit)) / sqr(d)
			 * henyey_greenstein(cos_theta_o, g)
			 * (1.0 - fr_dielectric(-cos_theta_o, eta))
			 * std::abs(cos_theta_o);
	}
	return Ess / n;
}

// ===========================================================================
// BeamDiffusionMS -- multiple-scattering (dipole) contribution to BSSRDF profile
// pbrt-v4: BeamDiffusionMS (bssrdf.cpp)
// Uses the dipole diffusion approximation (Donner & Jensen 2005, Grosjean 1956).
// ===========================================================================
inline double BeamDiffusionMS(double sigma_s, double sigma_a, double g, double eta, double r) {
	using namespace bssrdf_detail;
	const int n = 100;
	double Ed = 0.0;

	double sigmap_s = sigma_s * (1.0 - g);
	double sigmap_t = sigma_a + sigmap_s;
	double rhop     = sigmap_s / sigmap_t;
	double D_g      = (2.0*sigma_a + sigmap_s) / (3.0 * sigmap_t * sigmap_t);
	double sigma_tr = safe_sqrt(sigma_a / D_g);

	double fm1 = FresnelMoment1(eta);
	double fm2 = fresnel_moment2(eta);
	double ze  = -2.0 * D_g * (1.0 + 3.0*fm2) / (1.0 - 2.0*fm1);

	double c_phi = 0.25 * (1.0 - 2.0*fm1);
	double c_E   = 0.5  * (1.0 - 3.0*fm2);

	for (int i = 0; i < n; ++i) {
		double zr  = sample_exponential((i + 0.5) / n, sigmap_t);
		double zv  = -zr + 2.0 * ze;
		double dr  = std::sqrt(sqr(r) + sqr(zr));
		double dv  = std::sqrt(sqr(r) + sqr(zv));

		double phi_D = kInv4Pi / D_g * (fast_exp(-sigma_tr*dr)/dr - fast_exp(-sigma_tr*dv)/dv);
		double EDn   = kInv4Pi * (zr*(1.0+sigma_tr*dr)*fast_exp(-sigma_tr*dr)/pow3(dr)
								- zv*(1.0+sigma_tr*dv)*fast_exp(-sigma_tr*dv)/pow3(dv));
		double E     = phi_D * c_phi + EDn * c_E;
		double kappa = 1.0 - fast_exp(-2.0 * sigmap_t * (dr + zr));
		Ed += kappa * rhop * rhop * E;
	}
	return Ed / n;
}

// ===========================================================================
// BSSRDFTable -- tabulated BSSRDF profile data
//
// Stores a 2D table of R(rho, r_optical) = 2*pi*r*(SS+MS) values together
// with the effective albedo rhoEff and the marginal CDF for importance sampling.
//
// pbrt-v4: struct BSSRDFTable (bssrdf.h/bssrdf.cpp)
// ===========================================================================
struct BSSRDFTable {
	int n_rho;     ///< Number of albedo (rho) samples
	int n_radius;  ///< Number of radius samples

	std::vector<double> rho_samples;    ///< Albedo axis nodes, size n_rho
	std::vector<double> radius_samples; ///< Radius axis nodes, size n_radius
	std::vector<double> profile;        ///< profile[i*n_radius+j] = R(rho_i, r_j)
	std::vector<double> rho_eff;        ///< Effective albedo per rho row
	std::vector<double> profile_cdf;    ///< CDF along radius axis, same layout as profile

	BSSRDFTable() = default;
	BSSRDFTable(int nRho, int nRadius)
		: n_rho(nRho), n_radius(nRadius),
		  rho_samples(nRho, 0.0), radius_samples(nRadius, 0.0),
		  profile(nRho * nRadius, 0.0), rho_eff(nRho, 0.0),
		  profile_cdf(nRho * nRadius, 0.0) {}

	double eval_profile(int rho_idx, int radius_idx) const {
		return profile[rho_idx * n_radius + radius_idx];
	}
};

// ===========================================================================
// ComputeBeamDiffusionBSSRDF -- fill a BSSRDFTable
//
// Sets up the rho and radius grids, then evaluates the diffusion profile
// (SS+MS) at each (rho, r) pair and builds the CDF for importance sampling.
//
// pbrt-v4: ComputeBeamDiffusionBSSRDF (bssrdf.cpp)
// ===========================================================================
inline void ComputeBeamDiffusionBSSRDF(double g, double eta, BSSRDFTable* t) {
	using namespace bssrdf_detail;
	int nr = t->n_radius, nrho = t->n_rho;

	// Radius grid: geometric progression starting at 0, 2.5e-3, *1.2 each step
	t->radius_samples[0] = 0.0;
	t->radius_samples[1] = 2.5e-3;
	for (int i = 2; i < nr; ++i)
		t->radius_samples[i] = t->radius_samples[i - 1] * 1.2;

	// Rho grid: exponentially-spaced albedo values in (0,1)
	for (int i = 0; i < nrho; ++i)
		t->rho_samples[i] = (1.0 - fast_exp(-8.0 * i / (nrho - 1.0))) / (1.0 - fast_exp(-8.0));

	for (int i = 0; i < nrho; ++i) {
		double rho = t->rho_samples[i];
		for (int j = 0; j < nr; ++j) {
			double r = t->radius_samples[j];
			// Profile = 2*pi*r * (SS + MS)
			t->profile[i * nr + j] = 2.0 * kPi * r
				* (BeamDiffusionSS(rho, 1.0 - rho, g, eta, r)
				 + BeamDiffusionMS(rho, 1.0 - rho, g, eta, r));
		}
		// Integrate profile row to build CDF and get effective albedo rhoEff
		t->rho_eff[i] = integrate_catmull_rom(
			t->radius_samples.data(),
			&t->profile[i * nr],
			nr,
			&t->profile_cdf[i * nr]);
	}
}

// ===========================================================================
// SubsurfaceFromDiffuse -- convert (rhoEff, mfp) to (sigma_a, sigma_s)
//
// Inverts the tabulated rhoEff curve to find albedo rho, then converts the
// mean free path to absorption and scattering coefficients.
//
// pbrt-v4: SubsurfaceFromDiffuse (bssrdf.h)
// ===========================================================================
inline void SubsurfaceFromDiffuse(const BSSRDFTable& t,
								   double rho_eff, double mfp,
								   double& sigma_a, double& sigma_s) {
	using namespace bssrdf_detail;
	double rho = invert_catmull_rom(t.rho_samples.data(), t.rho_eff.data(),
									 t.n_rho, rho_eff);
	sigma_s = rho / mfp;
	sigma_a = (1.0 - rho) / mfp;
}

// ===========================================================================
// TabulatedBSSRDF -- evaluate / sample the tabulated diffusion profile
//
// Per-channel Sr(r), SampleSr(u), and PDF_Sr(r) using bilinear Catmull-Rom
// interpolation over the prebuilt BSSRDFTable.
//
// Usage:
//   TabulatedBSSRDF bssrdf(sigma_a, sigma_s, &table, n_channels);
//   double r_sampled = bssrdf.sample_sr(0, u);     // sample radius for channel 0
//   double sr        = bssrdf.sr(0, r_sampled);    // profile value
//   double pdf       = bssrdf.pdf_sr(0, r_sampled);
//
// pbrt-v4: TabulatedBSSRDF (bssrdf.h)
// ===========================================================================
class TabulatedBSSRDF {
  public:
	// sigma_a and sigma_s are arrays of n_ch channels each.
	TabulatedBSSRDF() = default;
	TabulatedBSSRDF(const double* sigma_a, const double* sigma_s,
					const BSSRDFTable* table, int n_ch)
		: table_(table), n_ch_(n_ch) {
		sigma_t_.resize(n_ch);
		rho_.resize(n_ch);
		for (int i = 0; i < n_ch; ++i) {
			sigma_t_[i] = sigma_a[i] + sigma_s[i];
			rho_[i]     = (sigma_t_[i] > 0.0) ? sigma_s[i] / sigma_t_[i] : 0.0;
		}
	}

	// Sr(ch, r) -- profile value at radius r for channel ch.
	// pbrt-v4: TabulatedBSSRDF::Sr (bssrdf.h)
	double sr(int ch, double r) const {
		double r_opt = r * sigma_t_[ch];
		int rho_off, rad_off;
		double rho_w[4], rad_w[4];
		if (!catmull_rom_weights(table_->rho_samples.data(), table_->n_rho,
								  rho_[ch], &rho_off, rho_w))
			return 0.0;
		if (!catmull_rom_weights(table_->radius_samples.data(), table_->n_radius,
								  r_opt, &rad_off, rad_w))
			return 0.0;
		double val = 0.0;
		for (int j = 0; j < 4; ++j)
			for (int k = 0; k < 4; ++k) {
				double w = rho_w[j] * rad_w[k];
				if (w != 0.0)
					val += w * table_->eval_profile(rho_off + j, rad_off + k);
			}
		if (r_opt != 0.0) val /= (2.0 * bssrdf_detail::kPi * r_opt);
		// Convert from optical radius space back to rendering space
		return std::max(0.0, val) * bssrdf_detail::sqr(sigma_t_[ch]);
	}

	// SampleSr(ch, u) -- importance-sample the profile for channel ch.
	// Returns the sampled radius in rendering units, or -1 if sigma_t==0.
	// pbrt-v4: TabulatedBSSRDF::SampleSr (bssrdf.h)
	double sample_sr(int ch, double u) const {
		if (sigma_t_[ch] == 0.0) return -1.0;
		double r_opt = SampleCatmullRom2D(
			table_->rho_samples.data(),
			table_->radius_samples.data(),
			table_->profile.data(),
			table_->profile_cdf.data(),
			table_->n_rho, table_->n_radius,
			rho_[ch], u);
		return r_opt / sigma_t_[ch];
	}

	// PDF_Sr(ch, r) -- probability density of the profile sample at radius r.
	// pbrt-v4: TabulatedBSSRDF::PDF_Sr (bssrdf.h)
	double pdf_sr(int ch, double r) const {
		double r_opt = r * sigma_t_[ch];
		int rho_off, rad_off;
		double rho_w[4], rad_w[4];
		if (!catmull_rom_weights(table_->rho_samples.data(), table_->n_rho,
								  rho_[ch], &rho_off, rho_w))
			return 0.0;
		if (!catmull_rom_weights(table_->radius_samples.data(), table_->n_radius,
								  r_opt, &rad_off, rad_w))
			return 0.0;
		double val = 0.0, rho_eff_interp = 0.0;
		for (int j = 0; j < 4; ++j) {
			if (rho_w[j] == 0.0) continue;
			rho_eff_interp += table_->rho_eff[rho_off + j] * rho_w[j];
			for (int k = 0; k < 4; ++k)
				if (rad_w[k] != 0.0)
					val += table_->eval_profile(rho_off + j, rad_off + k) * rho_w[j] * rad_w[k];
		}
		if (r_opt != 0.0) val /= (2.0 * bssrdf_detail::kPi * r_opt);
		if (rho_eff_interp == 0.0) return 0.0;
		return std::max(0.0, val * bssrdf_detail::sqr(sigma_t_[ch]) / rho_eff_interp);
	}

	int num_channels() const { return n_ch_; }

  private:
	const BSSRDFTable*   table_  = nullptr;
	int                  n_ch_   = 0;
	std::vector<double>  sigma_t_;
	std::vector<double>  rho_;
};
