// microfacet_tests.cpp
// Statistical validation of TrowbridgeReitz<T> in src/shared/microfacet.h.
//
// Methodology mirrors pbrt-v4's chi-squared BSDF tests (bsdf_test.cpp):
//   Draw N samples from Sample_wm(), bin them, compare to analytic PDF()
//   via Pearson chi-squared statistic and verify the null hypothesis holds.
//
// Tests:
//   1.  D_IntegratesTo1_Isotropic       -- NDF integrates to 1 (hemisphere MC)
//   2.  D_IntegratesTo1_Anisotropic     -- NDF integrates to 1, alpha_x != alpha_y
//   3.  G1_Reciprocal                   -- G(wo,wi) = 1/(1+L(wo)+L(wi))
//   4.  ChiSquared_IsotropicLow         -- VNDF sampling matches PDF, alpha=0.1
//   5.  ChiSquared_IsotropicMid         -- VNDF sampling matches PDF, alpha=0.5
//   6.  ChiSquared_IsotropicHigh        -- VNDF sampling matches PDF, alpha=0.9
//   7.  ChiSquared_Anisotropic          -- VNDF sampling matches PDF, ax!=ay
//   8.  ChiSquared_GrazingWo            -- VNDF sampling near-grazing wo
//   9.  PDF_NonNegative                 -- PDF(wo,wm) >= 0 for random pairs
//  10.  PDF_Normalization               -- integral of PDF over hemisphere ~1
//  11.  Regularize_ClampsLowAlpha       -- alphas below 0.3 are doubled then clamped
//  12.  Regularize_LeavesHighAlpha      -- alphas >= 0.3 are unchanged
//  13.  RoughnessToAlpha_Sqrt           -- RoughnessToAlpha(r) == sqrt(r)
//  14.  EffectivelySmooth_Threshold     -- true only when max(ax,ay) < 1e-3
//  15.  SampleWm_NormalizedOutput       -- Sample_wm output is unit length
//  16.  SampleWm_UpperHemisphere        -- Sample_wm wz > 0 for wo in upper hemi

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <array>
#include <numeric>
#include <random>

#include "../../src/shared/microfacet.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static constexpr double kPi = 3.14159265358979323846;

// Simple LCG-based 2D sample sequence (deterministic, reproducible).
struct SamplerLCG {
	uint64_t state;
	explicit SamplerLCG(uint64_t seed = 12345678ULL) : state(seed) {}
	double next() {
		state = state * 6364136223846793005ULL + 1442695040888963407ULL;
		return (state >> 11) * (1.0 / (1ULL << 53));
	}
};

// Normalize a 3-vector in place; returns false if degenerate.
static bool normalize3(double &x, double &y, double &z) {
	double len = std::sqrt(x*x + y*y + z*z);
	if (len < 1e-12) return false;
	x /= len; y /= len; z /= len;
	return true;
}

// Sample a cosine-weighted hemisphere direction (for random wo).
static void sample_cos_hemi(double u1, double u2,
							 double &wx, double &wy, double &wz) {
	double phi   = 2.0 * kPi * u1;
	double cos_t = std::sqrt(u2);       // cosine-weighted: cos^2 distribution
	double sin_t = std::sqrt(1.0 - u2);
	wx = sin_t * std::cos(phi);
	wy = sin_t * std::sin(phi);
	wz = cos_t;
	// ensure upper hemisphere
	if (wz < 1e-3) wz = 1e-3;
	normalize3(wx, wy, wz);
}

// -------------------------------------------------------------------------
// Chi-squared test helper.
//
// Strategy (matches pbrt-v4 bsdf_test approach):
//   - Discretise the upper hemisphere into N_THETA x N_PHI equal solid-angle
//     cells using equal-area latitude-longitude parameterisation.
//   - Draw N_SAMPLES from Sample_wm and bin them.
//   - Compute expected counts from analytic PDF integrated over each cell.
//   - Evaluate Pearson chi-squared statistic; return p-value via chi2 CDF.
//
// For a correct implementation the statistic should follow chi2(df) with
// df = (N_THETA*N_PHI - 1).  We assert chi2_stat < critical_value_0.001
// (one in a thousand false-positive rate per test).
// -------------------------------------------------------------------------

// Incomplete gamma function (regularised) via series expansion -- used for
// chi-squared p-value.  Accurate enough for our df sizes.
static double igam_series(double a, double x) {
	if (x <= 0.0) return 0.0;
	double sum = 1.0 / a;
	double term = sum;
	for (int k = 1; k < 300; ++k) {
		term *= x / (a + k);
		sum  += term;
		if (std::fabs(term) < 1e-12 * std::fabs(sum)) break;
	}
	return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

// Regularised upper incomplete gamma Q(a, x) = 1 - P(a, x).
static double chi2_pvalue(double chi2_stat, int df) {
	if (!std::isfinite(chi2_stat) || chi2_stat <= 0.0) return 1.0;
	double a = 0.5 * df, x = 0.5 * chi2_stat;
	// For x >> a the p-value is effectively 0 (series overflows); guard here.
	if (x > a + 50.0 * std::sqrt(a) + 1000.0) return 0.0;
	return 1.0 - igam_series(a, x);
}

struct ChiSquaredResult { double stat; double pvalue; int df; };

// Run the chi-squared test for a given TrowbridgeReitz<double> and outgoing
// direction (wox, woy, woz).
static ChiSquaredResult run_chi2(
	const TrowbridgeReitz<double>& dist,
	double wox, double woy, double woz,
	int n_theta = 16, int n_phi = 32,
	int n_samples = 200000,
	uint64_t seed = 42ULL)
{
	const int N_CELLS = n_theta * n_phi;

	// -- Step 1: compute analytic expected counts per cell --
	// Each cell covers solid angle dOmega = sin(theta)*dTheta*dPhi.
	// We use midpoint rule over a fine sub-grid.
	const int SUB = 8;
	std::vector<double> expected(N_CELLS, 0.0);

	for (int it = 0; it < n_theta; ++it) {
		for (int ip = 0; ip < n_phi; ++ip) {
			double integral = 0.0;
			for (int st = 0; st < SUB; ++st) {
				for (int sp = 0; sp < SUB; ++sp) {
					double theta = kPi * 0.5 * ((it + (st + 0.5) / SUB) / n_theta);
					double phi   = 2.0 * kPi * ((ip + (sp + 0.5) / SUB) / n_phi);
					double wm_x = std::sin(theta) * std::cos(phi);
					double wm_y = std::sin(theta) * std::sin(phi);
					double wm_z = std::cos(theta);
					// VNDF is only nonzero where dot(wo, wm) > 0
					double dot_wo_wm = wox*wm_x + woy*wm_y + woz*wm_z;
					if (dot_wo_wm < 0.0) continue;
					double pdf  = dist.PDF(wox, woy, woz, wm_x, wm_y, wm_z);
					// dOmega for sub-cell = sin(theta)*dTheta*dPhi
					double dtheta = (kPi * 0.5) / (n_theta * SUB);
					double dphi   = (2.0 * kPi)  / (n_phi   * SUB);
					integral += pdf * std::sin(theta) * dtheta * dphi;
				}
			}
			expected[it * n_phi + ip] = integral * n_samples;
		}
	}

	// -- Step 2: draw samples and bin --
	std::vector<int> observed(N_CELLS, 0);
	SamplerLCG rng(seed);
	for (int i = 0; i < n_samples; ++i) {
		double u1 = rng.next(), u2 = rng.next();
		double wmx, wmy, wmz;
		dist.Sample_wm(wox, woy, woz, u1, u2, wmx, wmy, wmz);
		// Clamp to upper hemisphere (wm should already be there)
		if (wmz <= 0.0) continue;
		// Skip samples outside the VNDF domain (dot(wo,wm) < 0)
		if (wox*wmx + woy*wmy + woz*wmz < 0.0) continue;
		// Convert to spherical
		double theta = std::acos(std::min(1.0, wmz));
		double phi   = std::atan2(wmy, wmx);
		if (phi < 0.0) phi += 2.0 * kPi;
		int it = std::min((int)(theta / (kPi * 0.5) * n_theta), n_theta - 1);
		int ip = std::min((int)(phi   / (2.0 * kPi) * n_phi),   n_phi   - 1);
		++observed[it * n_phi + ip];
	}

	// -- Step 3: Pearson chi-squared statistic (pool tiny cells) --
	double chi2_stat = 0.0;
	int effective_df = 0;
	double obs_pool  = 0.0;
	double exp_pool  = 0.0;

	for (int c = 0; c < N_CELLS; ++c) {
		if (expected[c] < 5.0) {
			// pool small cells
			obs_pool += observed[c];
			exp_pool += expected[c];
			if (exp_pool >= 5.0) {
				double d = obs_pool - exp_pool;
				chi2_stat += d * d / exp_pool;
				++effective_df;
				obs_pool = exp_pool = 0.0;
			}
		} else {
			double d = observed[c] - expected[c];
			chi2_stat += d * d / expected[c];
			++effective_df;
		}
	}
	if (exp_pool > 0.0) {
		double d = obs_pool - exp_pool;
		chi2_stat += d * d / exp_pool;
		++effective_df;
	}
	--effective_df; // subtract 1 for normalisation constraint

	double pvalue = chi2_pvalue(chi2_stat, std::max(1, effective_df));
	return {chi2_stat, pvalue, effective_df};
}

// ---------------------------------------------------------------------------
// Test 1: D() integrates to 1 over upper hemisphere (Monte Carlo, isotropic)
// ---------------------------------------------------------------------------
TEST(MicrofacetTest, D_IntegratesTo1_Isotropic) {
	TrowbridgeReitz<double> dist(0.4, 0.4);
	SamplerLCG rng(1);
	const int N = 500000;
	double sum = 0.0;
	for (int i = 0; i < N; ++i) {
		// Uniform solid angle over hemisphere: cos_t ~ Uniform[0,1] => p = 1/(2*pi)
		double u = rng.next(), v = rng.next();
		double cos_t = 1.0 - u;       // linear: uniform solid angle
		double sin2_t = std::max(0.0, 1.0 - cos_t * cos_t);
		double sin_t  = std::sqrt(sin2_t);
		double phi    = 2.0 * kPi * v;
		double wx = sin_t * std::cos(phi);
		double wy = sin_t * std::sin(phi);
		double wz = cos_t;
		if (wz <= 0.0) continue;
		// Estimator: D(wm)*cos_t / p(wm) = D(wm)*cos_t * 2*pi
		// Integral should equal 1 (pbrt-v4 §9.6.1)
		sum += dist.D(wx, wy, wz) * wz * (2.0 * kPi);
	}
	double integral = sum / N;
	EXPECT_NEAR(integral, 1.0, 0.01)
		<< "D() should integrate to 1 over upper hemisphere (isotropic alpha=0.4)";
}

// ---------------------------------------------------------------------------
// Test 2: D() integrates to 1 (anisotropic)
// ---------------------------------------------------------------------------
TEST(MicrofacetTest, D_IntegratesTo1_Anisotropic) {
	TrowbridgeReitz<double> dist(0.2, 0.7);
	SamplerLCG rng(2);
	const int N = 500000;
	double sum = 0.0;
	for (int i = 0; i < N; ++i) {
		double u = rng.next(), v = rng.next();
		double cos_t = 1.0 - u;
		double sin2_t = std::max(0.0, 1.0 - cos_t * cos_t);
		double sin_t  = std::sqrt(sin2_t);
		double phi    = 2.0 * kPi * v;
		double wx = sin_t * std::cos(phi);
		double wy = sin_t * std::sin(phi);
		double wz = cos_t;
		if (wz <= 0.0) continue;
		sum += dist.D(wx, wy, wz) * wz * (2.0 * kPi);
	}
	EXPECT_NEAR(sum / N, 1.0, 0.015)
		<< "D() should integrate to 1 (anisotropic alpha_x=0.2 alpha_y=0.7)";
}

// ---------------------------------------------------------------------------
// Test 3: G(wo,wi) = 1/(1 + Lambda(wo) + Lambda(wi))  (identity check)
// ---------------------------------------------------------------------------
TEST(MicrofacetTest, G_MatchesLambda) {
	TrowbridgeReitz<double> dist(0.3, 0.5);
	SamplerLCG rng(3);
	for (int i = 0; i < 1000; ++i) {
		double wox, woy, woz, wix, wiy, wiz;
		sample_cos_hemi(rng.next(), rng.next(), wox, woy, woz);
		sample_cos_hemi(rng.next(), rng.next(), wix, wiy, wiz);
		double G_direct = dist.G(wox, woy, woz, wix, wiy, wiz);
		double G_lambda = 1.0 / (1.0 + dist.Lambda(wox, woy, woz)
									  + dist.Lambda(wix, wiy, wiz));
		EXPECT_NEAR(G_direct, G_lambda, 1e-12)
			<< "G != 1/(1+L(wo)+L(wi)) at sample " << i;
	}
}

// ---------------------------------------------------------------------------
// Tests 4-8: chi-squared sampling validation
// ---------------------------------------------------------------------------

// Helper: assert chi-squared p-value > significance threshold.
// We use alpha=0.001 (one-in-thousand false-positive rate per test).
static void assert_chi2_pass(const ChiSquaredResult& r, const char* label) {
	EXPECT_GT(r.pvalue, 0.001)
		<< label << ": chi2=" << r.stat
		<< " df=" << r.df << " p=" << r.pvalue
		<< " (p < 0.001 suggests Sample_wm doesn't match PDF)";
}

TEST(MicrofacetTest, ChiSquared_IsotropicLow) {
	TrowbridgeReitz<double> dist(0.1, 0.1);
	auto r = run_chi2(dist, 0.0, 0.0, 1.0);   // wo = straight up
	assert_chi2_pass(r, "isotropic alpha=0.1, wo=z");
}

TEST(MicrofacetTest, ChiSquared_IsotropicMid) {
	TrowbridgeReitz<double> dist(0.5, 0.5);
	// Oblique outgoing direction
	double wox = 0.5, woy = 0.3, woz = std::sqrt(1.0 - 0.5*0.5 - 0.3*0.3);
	auto r = run_chi2(dist, wox, woy, woz);
	assert_chi2_pass(r, "isotropic alpha=0.5, oblique wo");
}

TEST(MicrofacetTest, ChiSquared_IsotropicHigh) {
	TrowbridgeReitz<double> dist(0.9, 0.9);
	double wox = 0.3, woy = 0.0, woz = std::sqrt(1.0 - 0.09);
	auto r = run_chi2(dist, wox, woy, woz);
	assert_chi2_pass(r, "isotropic alpha=0.9, oblique wo");
}

TEST(MicrofacetTest, ChiSquared_Anisotropic) {
	TrowbridgeReitz<double> dist(0.2, 0.6);
	double wox = 0.4, woy = 0.2, woz = std::sqrt(1.0 - 0.16 - 0.04);
	auto r = run_chi2(dist, wox, woy, woz);
	assert_chi2_pass(r, "anisotropic ax=0.2 ay=0.6");
}

TEST(MicrofacetTest, ChiSquared_GrazingWo) {
	TrowbridgeReitz<double> dist(0.4, 0.4);
	// Steep oblique outgoing direction: theta_wo ~ 72 deg (woz ~ 0.309)
	// This exercises the large-angle region of the VNDF where visibility
	// culling (dot(wo,wm)>=0) removes roughly half the hemisphere.
	double wox = std::sqrt(1.0 - 0.309*0.309), woy = 0.0, woz = 0.309;
	auto r = run_chi2(dist, wox, woy, woz);
	assert_chi2_pass(r, "steep oblique wo (theta~72 deg)");
}

// ---------------------------------------------------------------------------
// Test 9: PDF(wo, wm) >= 0 for all directions
// ---------------------------------------------------------------------------
TEST(MicrofacetTest, PDF_NonNegative) {
	TrowbridgeReitz<double> dist(0.35, 0.55);
	SamplerLCG rng(9);
	for (int i = 0; i < 10000; ++i) {
		double wox, woy, woz, wmx, wmy, wmz;
		sample_cos_hemi(rng.next(), rng.next(), wox, woy, woz);
		sample_cos_hemi(rng.next(), rng.next(), wmx, wmy, wmz);
		double pdf = dist.PDF(wox, woy, woz, wmx, wmy, wmz);
		EXPECT_GE(pdf, 0.0) << "PDF < 0 at sample " << i;
	}
}

// ---------------------------------------------------------------------------
// Test 10: integral of PDF(wo, wm) over upper hemisphere ~ 1
// Mirrors pbrt-v4 BSDF test: the VNDF integrates to 1 by construction.
// ---------------------------------------------------------------------------
TEST(MicrofacetTest, PDF_Normalization) {
	TrowbridgeReitz<double> dist(0.4, 0.4);
	// wo = (0.5, 0, sqrt(0.75))
	double wox = 0.5, woy = 0.0, woz = std::sqrt(0.75);

	SamplerLCG rng(10);
	const int N = 300000;
	double sum = 0.0;
	for (int i = 0; i < N; ++i) {
		// Uniform solid angle: cos_t linear in [0,1] => p(wm) = 1/(2*pi)
		double u = rng.next(), v = rng.next();
		double cos_t = 1.0 - u;
		double sin2_t = std::max(0.0, 1.0 - cos_t * cos_t);
		double sin_t  = std::sqrt(sin2_t);
		double phi    = 2.0 * kPi * v;
		double wmx = sin_t * std::cos(phi);
		double wmy = sin_t * std::sin(phi);
		double wmz = cos_t;
		if (wmz <= 0.0) continue;
		double pdf = dist.PDF(wox, woy, woz, wmx, wmy, wmz);
		// Estimator: pdf / p(wm) = pdf * 2*pi
		sum += pdf * (2.0 * kPi);
	}
	double integral = sum / N;
	// The VNDF PDF integrates to 1 over the upper hemisphere.
	EXPECT_NEAR(integral, 1.0, 0.02)
		<< "PDF normalization: integral should be 1";
}

// ---------------------------------------------------------------------------
// Test 11: Regularize clamps low alphas into [0.1, 0.3]
// pbrt-v4: Clamp(2*alpha, 0.1, 0.3)
// ---------------------------------------------------------------------------
TEST(MicrofacetTest, Regularize_ClampsLowAlpha) {
	// Very low alpha: 2*0.05 = 0.10, clamped to [0.1, 0.3] -> 0.10
	{
		TrowbridgeReitz<double> d(0.05, 0.05);
		d.Regularize();
		EXPECT_NEAR(d.alpha_x, 0.10, 1e-12);
		EXPECT_NEAR(d.alpha_y, 0.10, 1e-12);
	}
	// Mid-low alpha: 2*0.1 = 0.20, in [0.1,0.3] -> 0.20
	{
		TrowbridgeReitz<double> d(0.10, 0.10);
		d.Regularize();
		EXPECT_NEAR(d.alpha_x, 0.20, 1e-12);
		EXPECT_NEAR(d.alpha_y, 0.20, 1e-12);
	}
	// Upper-low alpha: 2*0.14 = 0.28, in [0.1,0.3] -> 0.28
	{
		TrowbridgeReitz<double> d(0.14, 0.20);
		d.Regularize();
		EXPECT_NEAR(d.alpha_x, 0.28, 1e-12);
		EXPECT_NEAR(d.alpha_y, 0.30, 1e-12);  // 2*0.20=0.40 > 0.3 -> 0.30
	}
}

// ---------------------------------------------------------------------------
// Test 12: Regularize leaves high alphas (>= 0.3) unchanged
// ---------------------------------------------------------------------------
TEST(MicrofacetTest, Regularize_LeavesHighAlpha) {
	TrowbridgeReitz<double> d(0.5, 0.8);
	d.Regularize();
	EXPECT_DOUBLE_EQ(d.alpha_x, 0.5);
	EXPECT_DOUBLE_EQ(d.alpha_y, 0.8);
}

// ---------------------------------------------------------------------------
// Test 13: RoughnessToAlpha(r) == sqrt(r)
// Matches pbrt-v4: "return std::sqrt(roughness)"
// ---------------------------------------------------------------------------
TEST(MicrofacetTest, RoughnessToAlpha_Sqrt) {
	for (double r : {0.0, 0.01, 0.1, 0.25, 0.5, 1.0}) {
		double expected = std::sqrt(r);
		double actual   = TrowbridgeReitz<double>::RoughnessToAlpha(r);
		EXPECT_DOUBLE_EQ(actual, expected) << "r=" << r;
	}
}

// ---------------------------------------------------------------------------
// Test 14: EffectivelySmooth threshold = max(ax,ay) < 1e-3
// ---------------------------------------------------------------------------
TEST(MicrofacetTest, EffectivelySmooth_Threshold) {
	EXPECT_TRUE( TrowbridgeReitz<double>(0.0005, 0.0005).EffectivelySmooth());
	EXPECT_TRUE( TrowbridgeReitz<double>(0.0001, 0.0009).EffectivelySmooth());
	EXPECT_FALSE(TrowbridgeReitz<double>(0.001,  0.001 ).EffectivelySmooth());
	EXPECT_FALSE(TrowbridgeReitz<double>(0.002,  0.001 ).EffectivelySmooth());
	EXPECT_FALSE(TrowbridgeReitz<double>(0.5,    0.5   ).EffectivelySmooth());
}

// ---------------------------------------------------------------------------
// Test 15: Sample_wm output is unit length
// ---------------------------------------------------------------------------
TEST(MicrofacetTest, SampleWm_NormalizedOutput) {
	TrowbridgeReitz<double> dist(0.4, 0.6);
	SamplerLCG rng(15);
	for (int i = 0; i < 5000; ++i) {
		double wox, woy, woz;
		sample_cos_hemi(rng.next(), rng.next(), wox, woy, woz);
		double wmx, wmy, wmz;
		dist.Sample_wm(wox, woy, woz, rng.next(), rng.next(), wmx, wmy, wmz);
		double len = std::sqrt(wmx*wmx + wmy*wmy + wmz*wmz);
		EXPECT_NEAR(len, 1.0, 1e-9) << "Sample_wm not unit at sample " << i;
	}
}

// ---------------------------------------------------------------------------
// Test 16: Sample_wm returns wm in upper hemisphere (wz > 0) for wo in upper
// ---------------------------------------------------------------------------
TEST(MicrofacetTest, SampleWm_UpperHemisphere) {
	TrowbridgeReitz<double> dist(0.5, 0.5);
	SamplerLCG rng(16);
	int violations = 0;
	for (int i = 0; i < 10000; ++i) {
		double wox, woy, woz;
		sample_cos_hemi(rng.next(), rng.next(), wox, woy, woz);
		double wmx, wmy, wmz;
		dist.Sample_wm(wox, woy, woz, rng.next(), rng.next(), wmx, wmy, wmz);
		if (wmz <= 0.0) ++violations;
	}
	EXPECT_EQ(violations, 0)
		<< "Sample_wm produced " << violations << " samples with wz <= 0";
}
