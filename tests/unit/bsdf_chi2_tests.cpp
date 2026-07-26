// bsdf_chi2_tests.cpp
// pbrt-v4-style chi-squared BSDF sampling tests + white-furnace energy
// conservation tests for our shared BxDF implementations.
//
// pbrt-v4 alignment
// -----------------
// pbrt-v4's bsdfs_test.cpp uses three complementary testing strategies:
//
//   1. Chi-squared test (FrequencyTable + IntegrateFrequencyTable + Chi2Test)
//      Sample directions from the BxDF, bin into a theta/phi histogram,
//      integrate the PDF analytically per cell, run chi-squared with cell
//      pooling and Sidak multiple-test correction.  Catches wrong sampling
//      distributions that range/monotone checks miss.
//
//   2. White-furnace energy conservation (TestEnergyConservation)
//      For each of several outgoing directions wo, MC-integrate
//      f(wo,wi)*|cosTheta_wi| / pdf(wi) over the hemisphere.
//      A physical BxDF must return <= 1 (<=1.01 with numerical slack).
//
// Chi-squared significance level : 0.01  (Sidak-corrected for CHI2_RUNS=5)
// Sample count per chi2 test     : 50 000
// Histogram                      : 20 theta x 40 phi = 800 cells
// White-furnace samples          : 8 192 per wo direction, 5 wo directions
//
// BxDFs tested
// ------------
//   DiffuseBxDF<double>                -- Lambertian (chi2 + white-furnace)
//   RoughMetalBxDF<double> alpha=0.5   -- GGX conductor (chi2 + white-furnace)
//   RoughMetalBxDF<double> alpha=0.1   -- GGX conductor low-roughness (white-furnace)
//   NormalizedFresnelBxDF<double>      -- Fresnel-weighted diffuse (chi2 + white-furnace)
//
// All BxDFs are evaluated in their local (shading) frame: normal = (0,0,1).

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include "../../src/shared/bxdfs.h"
#include "../../src/shared/microfacet.h"
#include "../../src/shared/fresnel.h"

// ---------------------------------------------------------------------------
// Constants & low-discrepancy helpers
// ---------------------------------------------------------------------------
static constexpr double kPi  = 3.14159265358979323846;
static constexpr double k2Pi = 6.28318530717958647692;

// Radical inverse base-2 (Van der Corput)
static double ri2(uint32_t n) {
	n = (n << 16) | (n >> 16);
	n = ((n & 0x55555555u) << 1) | ((n & 0xAAAAAAAAu) >> 1);
	n = ((n & 0x33333333u) << 2) | ((n & 0xCCCCCCCCu) >> 2);
	n = ((n & 0x0F0F0F0Fu) << 4) | ((n & 0xF0F0F0F0u) >> 4);
	n = ((n & 0x00FF00FFu) << 8) | ((n & 0xFF00FF00u) >> 8);
	return static_cast<double>(n) / 4294967296.0;
}
static double ri3(uint32_t n) {
	double r=0,f=1./3.; for(;n;n/=3){r+=(n%3)*f;f/=3.;} return r;
}
static double ri5(uint32_t n) {
	double r=0,f=0.2; for(;n;n/=5){r+=(n%5)*f;f/=5.;} return r;
}

// Cosine-weighted hemisphere sample (local frame, n=(0,0,1))
static void sample_cos_hemi(double u1, double u2,
							 double& x, double& y, double& z) {
	double phi = k2Pi * u1, r = std::sqrt(u2);
	x = r * std::cos(phi); y = r * std::sin(phi);
	z = std::sqrt(std::max(0.0, 1.0 - u2));
}

// ---------------------------------------------------------------------------
// Regularised lower incomplete gamma (Cephes) -- for chi-squared CDF
// Identical to pbrt-v4 RLGamma.
// ---------------------------------------------------------------------------
static double rl_gamma(double a, double x) {
	if (x == 0) return 0;
	const double eps = 1e-15, big = 4503599627370496.0, bigInv = 2.22044604925031e-16;
	double ax = a * std::log(x) - x - std::lgamma(a);
	if (ax < -709.78) return a < x ? 1.0 : 0.0;
	if (x <= 1 || x <= a) {
		double r = a, c = 1, ans = 1;
		do { r++; c *= x/r; ans += c; } while (c/ans > eps);
		return std::exp(ax) * ans / a;
	}
	int cnt = 0; double y = 1-a, z = x+y+1, p3=1, q3=x, p2=x+1, q2=z*x;
	double ans = p2/q2, err;
	do {
		++cnt; y++; z += 2; double yc = y*cnt;
		double p = p2*z - p3*yc, q = q2*z - q3*yc;
		if (q) { double na = p/q; err = std::fabs((ans-na)/na); ans = na; } else err = 1;
		p3=p2; p2=p; q3=q2; q2=q;
		if (std::fabs(p) > big) { p3*=bigInv; p2*=bigInv; q3*=bigInv; q2*=bigInv; }
	} while (err > eps);
	return 1.0 - std::exp(ax) * ans;
}
static double chi2_cdf(double x, int dof) {
	if (dof < 1 || x < 0) return 0;
	if (dof == 2) return 1.0 - std::exp(-0.5*x);
	return rl_gamma(0.5*dof, 0.5*x);
}

// ---------------------------------------------------------------------------
// chi2_test -- adapted from pbrt-v4 Chi2Test()
//
// observed[i] : MC sample count in cell i
// expected[i] : pdf_integral * total_samples for cell i
// Pools cells below min_expected frequency; applies Sidak correction.
// Returns {pass, message}.
// ---------------------------------------------------------------------------
static std::pair<bool,std::string> chi2_test(
	const std::vector<double>& obs, const std::vector<double>& exp_freq,
	int N, double min_exp = 5.0, double slevel = 0.01, int num_tests = 1)
{
	int n = (int)obs.size();
	std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
	std::sort(idx.begin(), idx.end(),
			  [&](int a, int b){ return exp_freq[a] < exp_freq[b]; });

	double chsq = 0, pool_o = 0, pool_e = 0; int dof = 0;
	for (int ii : idx) {
		if (exp_freq[ii] == 0) {
			if (obs[ii] > N * 1e-5)
				return {false, "Samples in zero-probability cell"};
		} else if (exp_freq[ii] < min_exp || (pool_e > 0 && pool_e < min_exp)) {
			pool_o += obs[ii]; pool_e += exp_freq[ii];
		} else {
			double d = obs[ii] - exp_freq[ii];
			chsq += d*d / exp_freq[ii]; ++dof;
		}
	}
	if (pool_e > 0) { double d = pool_o - pool_e; chsq += d*d/pool_e; ++dof; }
	dof -= 1;
	if (dof <= 0) return {false, "Too few degrees of freedom"};
	double pval  = 1.0 - chi2_cdf(chsq, dof);
	double alpha = 1.0 - std::pow(1.0 - slevel, 1.0 / num_tests);
	if (pval < alpha || !std::isfinite(pval))
		return {false, "Rejected: p=" + std::to_string(pval) +
					   " chi2=" + std::to_string(chsq) +
					   " dof=" + std::to_string(dof)};
	return {true, "passed"};
}

// ---------------------------------------------------------------------------
// run_chi2_bxdf -- generic chi-squared driver
//
// sample_fn(u1, u2) -> (x,y,z, valid)    -- sample a direction in local frame
// pdf_fn(x, y, z)   -> double            -- evaluate the sampling PDF
//
// Local frame: surface normal = (0,0,1).  Directions with z<=0 are ignored.
// ---------------------------------------------------------------------------
struct S3 { double x, y, z; bool valid; };

static void run_chi2_bxdf(
	const std::string& name,
	std::function<S3(double, double)> sample_fn,
	std::function<double(double, double, double)> pdf_fn,
	int N = 50000, int T_RES = 20, int P_RES = 40, int CHI2_RUNS = 5)
{
	int cells = T_RES * P_RES;
	std::vector<double> obs(cells, 0), exp_f(cells, 0);

	// --- MC sampling pass ---
	int valid = 0;
	for (int i = 0; i < N; ++i) {
		auto s = sample_fn(ri2(i), ri3(i));
		if (!s.valid || s.z <= 0) continue;
		++valid;
		double theta = std::acos(std::max(-1.0, std::min(1.0, s.z)));
		double phi   = std::atan2(s.y, s.x); if (phi < 0) phi += k2Pi;
		int tb = std::min(T_RES-1, (int)(theta / kPi * T_RES));
		int pb = std::min(P_RES-1, (int)(phi   / k2Pi * P_RES));
		obs[tb * P_RES + pb] += 1.0;
	}

	// --- Analytical PDF integration per cell (8x8 quadrature) ---
	double dT = kPi / T_RES, dP = k2Pi / P_RES;
	const int Q = 8;
	for (int ti = 0; ti < T_RES; ++ti) {
		for (int pi = 0; pi < P_RES; ++pi) {
			double val = 0;
			for (int qi = 0; qi < Q; ++qi) for (int qj = 0; qj < Q; ++qj) {
				double theta = (ti + (qi + 0.5) / Q) * dT;
				double phi   = (pi + (qj + 0.5) / Q) * dP;
				double st = std::sin(theta);
				double x = st * std::cos(phi), y = st * std::sin(phi), z = std::cos(theta);
				double p = pdf_fn(x, y, z);
				val += p * st;           // solid-angle Jacobian
			}
			exp_f[ti * P_RES + pi] = val * (dT * dP) / (Q * Q) * N;
		}
	}

	auto [pass, msg] = chi2_test(obs, exp_f, valid, 5.0, 0.01, CHI2_RUNS);
	EXPECT_TRUE(pass) << "[" << name << "] " << msg;
}

// ---------------------------------------------------------------------------
// run_white_furnace -- generic white-furnace energy conservation driver
//
// For each of n_wo outgoing directions, MC-integrate the IS estimator
//   E[f(wo,wi) * cosTheta_wi / pdf(wi)]
// using sample_fn which returns BxDFSampleResult whose r/g/b already encode
// the IS weight f * cosTheta / pdf (standard throughput convention).
// Expected value <= 1 for any physical BxDF.
// ---------------------------------------------------------------------------
static void run_white_furnace(
	const std::string& name,
	std::function<BxDFSampleResult<double>(double, double)> sample_fn,
	// weight_fn(result) -> IS weight to accumulate.  Default: max(r,g,b).
	std::function<double(const BxDFSampleResult<double>&)> weight_fn =
		[](const BxDFSampleResult<double>& s){ return std::max({s.r, s.g, s.b}); },
	int n_wo = 5, int n_samp = 8192)
{
	for (int wi = 0; wi < n_wo; ++wi) {
		// Fixed outgoing directions in the upper hemisphere
		double wox, woy, woz;
		sample_cos_hemi(ri2(wi * 7919u + 1), ri3(wi * 7919u + 1), wox, woy, woz);

		double Lo = 0; int cnt = 0;
		for (int j = 0; j < n_samp; ++j) {
			auto s = sample_fn(ri2(j + 1), ri3(j + 1));
			if (!s.valid) continue;
			Lo += weight_fn(s);
			++cnt;
		}
		if (cnt == 0) continue;
		Lo /= cnt;
		EXPECT_LE(Lo, 1.05)   // 5% slack for finite-sample noise
			<< "[" << name << "] white-furnace Lo=" << Lo
			<< " (wo=(" << wox << "," << woy << "," << woz << "))";
	}
}

// ===========================================================================
// DiffuseBxDF (Lambertian)
// ===========================================================================

TEST(BxDFChi2, Lambertian) {
	// In local frame, n=(0,0,1).  DiffuseBxDF.sample() takes world-space n
	// but we pass local-frame n directly (equivalent when n=(0,0,1)).
	DiffuseBxDF<double> bxdf{1.0, 1.0, 1.0};

	run_chi2_bxdf(
		"DiffuseBxDF",
		[&](double u1, double u2) -> S3 {
			// sample in local frame: n=(0,0,1) so world = local
			auto s = bxdf.sample(0, 0, 1, u1, u2);
			return {s.wo_x, s.wo_y, s.wo_z, s.valid};
		},
		[&](double x, double y, double z) -> double {
			return bxdf.scattering_pdf(0, 0, 1, x, y, z);
		}
	);
}

TEST(BxDFWhiteFurnace, LambertianUnitAlbedo) {
	DiffuseBxDF<double> bxdf{1.0, 1.0, 1.0};

	run_white_furnace(
		"DiffuseBxDF albedo=1",
		[&](double u1, double u2) {
			return bxdf.sample(0, 0, 1, u1, u2);
		}
	);
}

TEST(BxDFWhiteFurnace, LambertianDarkAlbedo) {
	DiffuseBxDF<double> bxdf{0.2, 0.4, 0.6};

	run_white_furnace(
		"DiffuseBxDF albedo=0.2/0.4/0.6",
		[&](double u1, double u2) {
			return bxdf.sample(0, 0, 1, u1, u2);
		}
	);
}

// ===========================================================================
// RoughMetalBxDF (GGX VNDF)
//
// sample_local(wi_x,wi_y,wi_z, u1,u2) -- local frame, wi=(0,0,1) for normal incidence
// PDF(wo) = D_visible(wi, wm) / (4 * dot(wi, wm))
//         where wm = normalize(wi + wo)
// ===========================================================================

// Compute GGX VNDF PDF for wo given wi (both in local frame, z-up)
static double ggx_pdf(double alpha,
					  double wix, double wiy, double wiz,
					  double wox, double woy, double woz) {
	if (wiz <= 0 || woz <= 0) return 0;
	// half-vector
	double hmx = wix + wox, hmy = wiy + woy, hmz = wiz + woz;
	double hlen = std::sqrt(hmx*hmx + hmy*hmy + hmz*hmz);
	if (hlen < 1e-8) return 0;
	hmx /= hlen; hmy /= hlen; hmz /= hlen;
	if (hmz <= 0) return 0;
	TrowbridgeReitz<double> dist(alpha, alpha);
	double dot_wi_wm = wix*hmx + wiy*hmy + wiz*hmz;
	if (dot_wi_wm <= 0) return 0;
	// PDF(wo) = D_visible(wi, wm) / (4 * |dot(wi,wm)|)
	double dvndf = dist.D_visible(wix, wiy, wiz, hmx, hmy, hmz);
	return dvndf / (4.0 * dot_wi_wm);
}

TEST(BxDFChi2, RoughMetalAlpha05) {
	RoughMetalBxDF<double> bxdf{1.0, 1.0, 1.0, 0.5};
	// Normal incidence: wi=(0,0,1) (local frame, pointing away from surface)
	const double wix = 0, wiy = 0, wiz = 1;

	run_chi2_bxdf(
		"RoughMetalBxDF alpha=0.5",
		[&](double u1, double u2) -> S3 {
			auto s = bxdf.sample_local(wix, wiy, wiz, u1, u2);
			return {s.wo_x, s.wo_y, s.wo_z, s.valid && s.wo_z > 0};
		},
		[&](double x, double y, double z) -> double {
			if (z <= 0) return 0;
			return ggx_pdf(0.5, wix, wiy, wiz, x, y, z);
		}
	);
}

TEST(BxDFChi2, RoughMetalAlpha02) {
	RoughMetalBxDF<double> bxdf{1.0, 1.0, 1.0, 0.2};
	const double wix = 0, wiy = 0, wiz = 1;

	run_chi2_bxdf(
		"RoughMetalBxDF alpha=0.2",
		[&](double u1, double u2) -> S3 {
			auto s = bxdf.sample_local(wix, wiy, wiz, u1, u2);
			return {s.wo_x, s.wo_y, s.wo_z, s.valid && s.wo_z > 0};
		},
		[&](double x, double y, double z) -> double {
			if (z <= 0) return 0;
			return ggx_pdf(0.2, wix, wiy, wiz, x, y, z);
		}
	);
}

TEST(BxDFWhiteFurnace, RoughMetalAlpha05) {
	RoughMetalBxDF<double> bxdf{1.0, 1.0, 1.0, 0.5};
	const double wix = 0, wiy = 0, wiz = 1;

	run_white_furnace(
		"RoughMetalBxDF alpha=0.5",
		[&](double u1, double u2) {
			return bxdf.sample_local(wix, wiy, wiz, u1, u2);
		}
	);
}

TEST(BxDFWhiteFurnace, RoughMetalAlpha01) {
	RoughMetalBxDF<double> bxdf{1.0, 1.0, 1.0, 0.1};
	const double wix = 0, wiy = 0, wiz = 1;

	run_white_furnace(
		"RoughMetalBxDF alpha=0.1",
		[&](double u1, double u2) {
			return bxdf.sample_local(wix, wiy, wiz, u1, u2);
		}
	);
}

// ===========================================================================
// NormalizedFresnelBxDF (Fresnel-weighted diffuse)
// Has both sample() and scattering_pdf() -- full chi2 + white-furnace
// ===========================================================================

// FresnelMoment1: int_0^1 Fr(cos, eta) d(cos) -- Lagarde & de Rousiers 2014 approx
static double fresnel_moment1(double eta) {
	// Two-sided approximation from pbrt-v4 util/scattering.h
	if (eta < 1) {
		return 0.45966 - 1.73965*eta + 3.37668*eta*eta - 3.04942*eta*eta*eta
			 + 1.49747*eta*eta*eta*eta - 0.32228*eta*eta*eta*eta*eta;
	}
	return -4.61686 + 11.1136*eta - 10.4646*eta*eta + 5.11455*eta*eta*eta
		  - 1.27198*eta*eta*eta*eta + 0.12746*eta*eta*eta*eta*eta;
}

TEST(BxDFChi2, NormalizedFresnel) {
	// NormalizedFresnelBxDF::sample() uses cosine-hemisphere sampling.
	// The chi-squared test must compare the *sampling distribution* against
	// its PDF -- which is cos(theta)/pi, NOT scattering_pdf().
	// (scattering_pdf() returns the BSDF density, not the sampling density;
	//  they differ by a (1-Fr)/c factor.)
	// This is identical to the Lambertian case sampling-distribution-wise.
	double eta = 1.5;
	double c   = 1.0 - 2.0 * fresnel_moment1(1.0 / eta);
	c = std::max(c, 1e-4);
	NormalizedFresnelBxDF<double> bxdf{eta, c};

	run_chi2_bxdf(
		"NormalizedFresnelBxDF eta=1.5 (cosine-hemisphere sampling)",
		[&](double u1, double u2) -> S3 {
			auto s = bxdf.sample(0, 0, 1, u1, u2);
			return {s.wo_x, s.wo_y, s.wo_z, s.valid};
		},
		[&](double x, double y, double z) -> double {
			// Sampling PDF = cosine hemisphere = cos(theta) / pi
			double cos_theta = z;  // n=(0,0,1)
			return (cos_theta > 0) ? cos_theta / kPi : 0.0;
		}
	);
}

TEST(BxDFWhiteFurnace, NormalizedFresnel) {
	double eta = 1.5;
	double c   = 1.0 - 2.0 * fresnel_moment1(1.0 / eta);
	c = std::max(c, 1e-4);
	NormalizedFresnelBxDF<double> bxdf{eta, c};

	run_white_furnace(
		"NormalizedFresnelBxDF eta=1.5",
		[&](double u1, double u2) {
			return bxdf.sample(0, 0, 1, u1, u2);
		}
	);
}
