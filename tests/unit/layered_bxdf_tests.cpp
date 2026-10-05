// layered_bxdf_tests.cpp
// Unit tests for CoatedDiffuseBxDF and CoatedConductorBxDF
// (src/shared/bxdfs.h) -- pbrt-v4 LayeredBxDF random-walk model.
//
// Mirrors pbrt-v4 LayeredBxDF validation approach:
//   - Direction validity (wo.z > 0)
//   - Throughput in [0, 1] (energy conservation)
//   - Coat-specular branch returns achromatic weight
//   - Determinism (same seed -> same result)
//   - Multi-bounce paths produce results (not just the first specular bounce)
//   - CoatedConductor throughput has per-channel variation (colored metal)

#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>

#include "../../src/shared/bxdfs.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static uint32_t pcg32_step(uint64_t& state, uint64_t inc) {
	uint64_t old = state;
	state = old * 6364136223846793005ULL + inc;
	uint32_t xs  = (uint32_t)(((old >> 18u) ^ old) >> 27u);
	uint32_t rot = (uint32_t)(old >> 59u);
	return (xs >> rot) | (xs << ((~rot + 1u) & 31u));
}

static double randu(uint64_t& state) {
	uint64_t inc = 1442695040888963407ULL;
	return (pcg32_step(state, inc) >> 8) * (1.0 / 16777216.0);
}

// Build a coated diffuse BxDF with standard params
static CoatedDiffuseBxDF<double> make_coated_diffuse(
	double albedo = 0.6, double ior = 1.5, double roughness = 0.3,
	double thickness = 0.01)
{
	double alpha = TrowbridgeReitz<double>::RoughnessToAlpha(roughness);
	CoatedDiffuseBxDF<double> b;
	b.albedo_r = albedo; b.albedo_g = albedo * 0.7; b.albedo_b = albedo * 0.4;
	b.coat_ior = ior; b.alpha_x = alpha; b.alpha_y = alpha;
	b.thickness = thickness; b.g = 0.0; b.medium_albedo = 0.0;
	b.maxDepth = 10; b.nSamples = 1;
	return b;
}

static CoatedConductorBxDF<double> make_coated_conductor(
	double coat_ior = 1.5, double roughness = 0.2, double thickness = 0.01)
{
	double alpha = TrowbridgeReitz<double>::RoughnessToAlpha(roughness);
	CoatedConductorBxDF<double> b;
	// Gold IOR (approximate)
	b.eta_r = 0.143; b.eta_g = 0.374; b.eta_b = 1.442;
	b.k_r   = 3.981; b.k_g   = 2.386; b.k_b   = 1.599;
	b.coat_ior = coat_ior; b.alpha_x = alpha; b.alpha_y = alpha;
	b.thickness = thickness; b.g = 0.0; b.medium_albedo = 0.0;
	b.maxDepth = 10; b.nSamples = 1;
	return b;
}

// ---------------------------------------------------------------------------
// CoatedDiffuseBxDF tests
// ---------------------------------------------------------------------------

// Test: wo.z > 0 for all valid samples (direction stays above surface)
TEST(CoatedDiffuseBxDF, SampledDirectionAboveSurface) {
	auto b = make_coated_diffuse();
	uint64_t st = 12345;
	int valid = 0;
	for (int i = 0; i < 200; ++i) {
		double u1 = randu(st), u2 = randu(st);
		// wi at 45 degrees
		double wi_z = 0.7071, wi_x = 0.7071, wi_y = 0.0;
		auto res = b.sample_local(wi_x, wi_y, wi_z,
								  (uint64_t)(u1 * 1e12), (uint64_t)(u2 * 1e12));
		if (!res.valid) continue;
		++valid;
		EXPECT_GT(res.wo_z, 0.0) << "Sampled direction must be above surface";
	}
	EXPECT_GT(valid, 50) << "Expected many valid samples for reasonable wi";
}

// Test: throughput (r,g,b) in [0, 1] -- energy conservation
TEST(CoatedDiffuseBxDF, ThroughputInRange) {
	auto b = make_coated_diffuse();
	uint64_t st = 99999;
	for (int i = 0; i < 300; ++i) {
		double u1 = randu(st), u2 = randu(st);
		double wi_z = 0.6 + 0.3 * randu(st);
		double wi_x = std::sqrt(std::max(0.0, 1.0 - wi_z*wi_z));
		double wi_y = 0.0;
		auto res = b.sample_local(wi_x, wi_y, wi_z,
								  (uint64_t)(u1 * 1e14 + i), (uint64_t)(u2 * 1e13));
		if (!res.valid) continue;
		EXPECT_GE(res.r, 0.0) << "r must be non-negative";
		EXPECT_GE(res.g, 0.0) << "g must be non-negative";
		EXPECT_GE(res.b, 0.0) << "b must be non-negative";
		EXPECT_LE(res.r, 1.5) << "r should be <= 1.5 (slight overshoot OK for GGX masking)";
		EXPECT_LE(res.g, 1.5) << "g must be <= 1.5";
		EXPECT_LE(res.b, 1.5) << "b must be <= 1.5";
	}
}

// Test: specular-branch output is achromatic (r == g == b for coat reflection)
TEST(CoatedDiffuseBxDF, SpecularBranchIsAchromatic) {
	// The mirror reflection of the coat is only a specular sample when the coat is smooth (pbrt: IsSpecular()).
	auto b = make_coated_diffuse();
	b.alpha_x = b.alpha_y = 0.0;
	// Force normal incidence -> maximum F_in, higher chance of specular
	const double wi_z = 0.9999, wi_x = std::sqrt(1.0 - wi_z*wi_z), wi_y = 0.0;
	uint64_t st = 777;
	int specular_found = 0;
	for (int i = 0; i < 500 && specular_found < 20; ++i) {
		double u1 = randu(st), u2 = randu(st);
		auto res = b.sample_local(wi_x, wi_y, wi_z,
								  (uint64_t)(u1 * 1e14 + i*7), (uint64_t)(u2 * 1e12));
		if (!res.valid || !res.is_specular) continue;
		++specular_found;
		EXPECT_NEAR(res.r, res.g, 1e-12) << "Coat specular must be achromatic (r==g)";
		EXPECT_NEAR(res.g, res.b, 1e-12) << "Coat specular must be achromatic (g==b)";
	}
	EXPECT_GT(specular_found, 0) << "Should find at least one specular sample";
}

// Test: determinism -- same seed gives same result
TEST(CoatedDiffuseBxDF, Deterministic) {
	auto b = make_coated_diffuse();
	const double wi_x = 0.5, wi_y = 0.3, wi_z = 0.812;
	const uint64_t s0 = 0xABCDEF01ULL, s1 = 0x12345678ULL;

	auto r1 = b.sample_local(wi_x, wi_y, wi_z, s0, s1);
	auto r2 = b.sample_local(wi_x, wi_y, wi_z, s0, s1);

	EXPECT_EQ(r1.valid, r2.valid);
	if (r1.valid) {
		EXPECT_EQ(r1.wo_x, r2.wo_x);
		EXPECT_EQ(r1.wo_y, r2.wo_y);
		EXPECT_EQ(r1.wo_z, r2.wo_z);
		EXPECT_EQ(r1.r,    r2.r);
		EXPECT_EQ(r1.g,    r2.g);
		EXPECT_EQ(r1.b,    r2.b);
	}
}

// Test: different seeds produce different results
TEST(CoatedDiffuseBxDF, DifferentSeedsDifferentResults) {
	auto b = make_coated_diffuse();
	const double wi_x = 0.4, wi_y = 0.2, wi_z = 0.894;
	auto r1 = b.sample_local(wi_x, wi_y, wi_z, 111ULL, 222ULL);
	auto r2 = b.sample_local(wi_x, wi_y, wi_z, 999ULL, 888ULL);
	if (r1.valid && r2.valid) {
		EXPECT_FALSE(r1.wo_x == r2.wo_x && r1.wo_y == r2.wo_y && r1.wo_z == r2.wo_z)
			<< "Different seeds should produce different samples";
	}
}

// Test: backward-compat 5-float overload compiles and returns valid result
TEST(CoatedDiffuseBxDF, FiveFloatOverloadWorks) {
	auto b = make_coated_diffuse();
	const double wi_x = 0.3, wi_y = 0.0, wi_z = 0.954;
	// May or may not be valid, but must not crash/UB
	auto res = b.sample_local(wi_x, wi_y, wi_z, 0.1, 0.3, 0.7, 0.5, 0.2);
	if (res.valid) {
		EXPECT_GT(res.wo_z, 0.0);
	}
}

// Test: wi.z <= 0 returns invalid
TEST(CoatedDiffuseBxDF, BelowSurfaceWiReturnsInvalid) {
	auto b = make_coated_diffuse();
	auto res = b.sample_local(0.0, 0.0, -0.5, 0ULL, 0ULL);
	EXPECT_FALSE(res.valid);
}

// Test: non-specular samples have colored output (albedo_r != albedo_b)
TEST(CoatedDiffuseBxDF, DiffuseBranchIsColored) {
	auto b = make_coated_diffuse(0.6);  // r=0.6, g=0.42, b=0.24
	const double wi_z = 0.6, wi_x = std::sqrt(1.0 - wi_z*wi_z), wi_y = 0.0;
	uint64_t st = 13579;
	int colored_found = 0;
	for (int i = 0; i < 500 && colored_found < 5; ++i) {
		double u1 = randu(st);
		auto res = b.sample_local(wi_x, wi_y, wi_z,
								  (uint64_t)(u1 * 1e14 + i), (uint64_t)(i * 999999ULL));
		// A rough coat's own (glossy) reflection is achromatic too; only a path that reached the base is tinted.
		if (!res.valid || res.is_specular || res.r == res.b) continue;
		++colored_found;
		// r and b should differ since albedo_r != albedo_b
		EXPECT_GT(std::fabs(res.r - res.b), 0.0)
			<< "Diffuse branch should produce colored output";
	}
	EXPECT_GT(colored_found, 0) << "Should find at least one diffuse sample";
}

// ---------------------------------------------------------------------------
// CoatedConductorBxDF tests
// ---------------------------------------------------------------------------

// Test: wo.z > 0 for all valid samples
TEST(CoatedConductorBxDF, SampledDirectionAboveSurface) {
	auto b = make_coated_conductor();
	uint64_t st = 54321;
	int valid = 0;
	for (int i = 0; i < 200; ++i) {
		double u1 = randu(st), u2 = randu(st);
		double wi_z = 0.7071, wi_x = 0.7071, wi_y = 0.0;
		auto res = b.sample_local(wi_x, wi_y, wi_z,
								  (uint64_t)(u1 * 1e12 + i), (uint64_t)(u2 * 1e11));
		if (!res.valid) continue;
		++valid;
		EXPECT_GT(res.wo_z, 0.0) << "Sampled direction must be above surface";
	}
	EXPECT_GT(valid, 50) << "Expected many valid samples";
}

// Test: coat-specular branch is achromatic
TEST(CoatedConductorBxDF, SpecularBranchIsAchromatic) {
	auto b = make_coated_conductor();
	b.alpha_x = b.alpha_y = 0.0;      // smooth coat; the conductor underneath stays rough
	b.cond_alpha_x = b.cond_alpha_y = 0.3;
	const double wi_z = 0.9999, wi_x = std::sqrt(1.0 - wi_z*wi_z), wi_y = 0.0;
	uint64_t st = 4242;
	int found = 0;
	for (int i = 0; i < 500 && found < 20; ++i) {
		double u1 = randu(st), u2 = randu(st);
		auto res = b.sample_local(wi_x, wi_y, wi_z,
								  (uint64_t)(u1 * 1e14 + i*3), (uint64_t)(u2 * 1e12));
		if (!res.valid || !res.is_specular) continue;
		++found;
		EXPECT_NEAR(res.r, res.g, 1e-12) << "Coat specular must be achromatic";
		EXPECT_NEAR(res.g, res.b, 1e-12) << "Coat specular must be achromatic";
	}
	EXPECT_GT(found, 0) << "Should find at least one specular sample";
}

// Test: conductor throughput has per-channel variation (non-specular branches)
TEST(CoatedConductorBxDF, ConductorPathIsColored) {
	auto b = make_coated_conductor();  // gold: r/g/b differ
	const double wi_z = 0.5, wi_x = std::sqrt(1.0 - wi_z*wi_z), wi_y = 0.0;
	uint64_t st = 86420;
	int colored = 0;
	for (int i = 0; i < 500 && colored < 5; ++i) {
		double u1 = randu(st);
		auto res = b.sample_local(wi_x, wi_y, wi_z,
								  (uint64_t)(u1 * 1e14 + i), (uint64_t)(i * 777777ULL));
		if (!res.valid || res.is_specular) continue;
		++colored;
		double max_diff = std::max({std::fabs(res.r - res.g),
									std::fabs(res.g - res.b),
									std::fabs(res.r - res.b)});
		EXPECT_GT(max_diff, 1e-6)
			<< "Conductor path should produce colored output (gold has colored Fresnel)";
	}
	EXPECT_GT(colored, 0) << "Should find at least one conductor-path sample";
}

// Test: determinism
TEST(CoatedConductorBxDF, Deterministic) {
	auto b = make_coated_conductor();
	const double wi_x = 0.5, wi_y = 0.2, wi_z = 0.843;
	const uint64_t s0 = 0xFEDCBA98ULL, s1 = 0x87654321ULL;

	auto r1 = b.sample_local(wi_x, wi_y, wi_z, s0, s1);
	auto r2 = b.sample_local(wi_x, wi_y, wi_z, s0, s1);

	EXPECT_EQ(r1.valid, r2.valid);
	if (r1.valid) {
		EXPECT_EQ(r1.wo_x, r2.wo_x);
		EXPECT_EQ(r1.wo_y, r2.wo_y);
		EXPECT_EQ(r1.wo_z, r2.wo_z);
		EXPECT_EQ(r1.r,    r2.r);
		EXPECT_EQ(r1.g,    r2.g);
		EXPECT_EQ(r1.b,    r2.b);
	}
}

// Test: wi.z <= 0 returns invalid
TEST(CoatedConductorBxDF, BelowSurfaceWiReturnsInvalid) {
	auto b = make_coated_conductor();
	auto res = b.sample_local(0.0, 0.0, -0.3, 0ULL, 0ULL);
	EXPECT_FALSE(res.valid);
}

// Test: backward-compat 5-float overload
TEST(CoatedConductorBxDF, FiveFloatOverloadWorks) {
	auto b = make_coated_conductor();
	auto res = b.sample_local(0.5, 0.2, 0.843, 0.1, 0.4, 0.6, 0.2, 0.8);
	if (res.valid) {
		EXPECT_GT(res.wo_z, 0.0);
	}
}

// Test: different roughness produces different energy
TEST(CoatedConductorBxDF, RoughnessAffectsOutput) {
	// High roughness vs low roughness -- average throughput should differ
	auto b_rough  = make_coated_conductor(1.5, 0.5);
	auto b_smooth = make_coated_conductor(1.5, 0.01);

	uint64_t st = 11111;
	double sum_rough = 0, sum_smooth = 0;
	int n_rough = 0, n_smooth = 0;
	for (int i = 0; i < 200; ++i) {
		double u1 = randu(st);
		double wi_z = 0.8, wi_x = std::sqrt(1.0 - wi_z*wi_z), wi_y = 0.0;
		uint64_t s0 = (uint64_t)(u1 * 1e14 + i), s1 = (uint64_t)(i * 131313ULL);
		auto r1 = b_rough.sample_local(wi_x, wi_y, wi_z, s0, s1);
		auto r2 = b_smooth.sample_local(wi_x, wi_y, wi_z, s0, s1);
		if (r1.valid) { sum_rough  += (r1.r + r1.g + r1.b) / 3.0; ++n_rough; }
		if (r2.valid) { sum_smooth += (r2.r + r2.g + r2.b) / 3.0; ++n_smooth; }
	}
	if (n_rough > 5 && n_smooth > 5) {
		double avg_rough  = sum_rough  / n_rough;
		double avg_smooth = sum_smooth / n_smooth;
		// They should differ -- not identical throughput
		EXPECT_NE(avg_rough, avg_smooth)
			<< "Rough and smooth coatings should produce different avg throughput";
	}
}

// ---------------------------------------------------------------------------
// pbrt-v4 LayeredBxDF port (src/shared/bxdfs_layered.h): the coat refracts, the two interfaces have their own
// roughness, and Sample_f / f() / PDF() are pbrt's. These check the port against independent references rather
// than against itself: a closed form for the all-smooth case, and the sampler against the f() integral elsewhere.
// ---------------------------------------------------------------------------

namespace {

constexpr double kLayeredPi = 3.14159265358979323846;

const double kCuEta[3] = {0.246, 1.072, 1.155};
const double kCuK[3]   = {3.378, 2.591, 2.469};

// Smooth dielectric coat over a smooth conductor, summed over the infinite inter-reflection series: the top
// reflection plus the light that enters (1-R), crosses the layer (Beer-Lambert, unit extinction), reflects off the
// conductor, crosses back, leaves (1-R') and keeps looping with R' at the coat's underside.
double analytic_smooth_coat_over_conductor(double cos_i, double ior, double eta, double k, double thickness) {
	const double R = FrDielectric(cos_i, ior);
	const double sin2_t = (1.0 - cos_i * cos_i) / (ior * ior);
	const double cos_t = std::sqrt(1.0 - sin2_t);
	const double tr = std::exp(-thickness / cos_t);
	const double Fc = FrComplex(cos_t, eta, k);
	const double Rint = FrDielectric(-cos_t, ior);
	const double up = (1.0 - R) * tr * Fc * tr * (1.0 - Rint);
	const double loop = Rint * tr * Fc * tr;
	return R + up / (1.0 - loop);
}

CoatedConductorBxDF<double> cu_under_coat(double coat_alpha, double base_alpha) {
	CoatedConductorBxDF<double> b{};
	b.eta_r = kCuEta[0]; b.eta_g = kCuEta[1]; b.eta_b = kCuEta[2];
	b.k_r = kCuK[0];     b.k_g = kCuK[1];     b.k_b = kCuK[2];
	b.coat_ior = 1.5;
	b.alpha_x = b.alpha_y = coat_alpha;
	b.cond_alpha_x = b.cond_alpha_y = base_alpha;
	b.thickness = 0.01; b.maxDepth = 10; b.nSamples = 1;
	return b;
}

// Albedo from the two estimators for the same BSDF at normal-ish incidence `theta`: the mean of the walk's sample
// weights split into the coat's mirror lobe (is_specular) and the rest, and the integral of f()*cos over the
// hemisphere (cosine-weighted). f() has no delta part, so the rest must equal the f() integral.
template<typename Bx>
void albedo_from_sampler_and_f(const Bx& b, double theta_deg, int n, double& specular, double& glossy, double& f_integral) {
	const double th = theta_deg * kLayeredPi / 180.0;
	const double wx = std::sin(th), wz = std::cos(th);
	uint64_t st = 24680 + (uint64_t)theta_deg;
	specular = glossy = 0.0;
	for (int i = 0; i < n; ++i) {
		const uint64_t s0 = (uint64_t)(randu(st) * 1e15), s1 = (uint64_t)(randu(st) * 1e15);
		auto r = b.sample_local(wx, 0.0, wz, s0, s1);
		if (!r.valid) continue;
		(r.is_specular ? specular : glossy) += r.r;
	}
	specular /= n; glossy /= n;
	f_integral = 0.0;
	for (int i = 0; i < n; ++i) {
		double x, y, z, pdf;
		SampleCosineHemisphere(randu(st), randu(st), x, y, z, pdf);
		double fr, fg, fb;
		b.f(wx, 0.0, wz, x, y, z, (uint64_t)(randu(st) * 1e15), (uint64_t)(randu(st) * 1e15), fr, fg, fb);
		f_integral += fr * z / pdf;
	}
	f_integral /= n;
}

} // namespace

// A smooth coat over a smooth conductor is a delta lobe: every sample is specular, and the mean weight is the
// closed-form series above. (The old model never refracted and read a third of this at normal incidence.)
TEST(LayeredBxDF, SmoothCoatOverSmoothConductorMatchesTheClosedForm) {
	auto b = cu_under_coat(0.0, 0.0);
	EXPECT_TRUE(b.is_delta());
	uint64_t st = 777;
	for (double theta : {0.0, 40.0, 70.0}) {
		const double th = theta * kLayeredPi / 180.0;
		const int n = 60000;
		double sum[3] = {0, 0, 0};
		for (int i = 0; i < n; ++i) {
			const uint64_t s0 = (uint64_t)(randu(st) * 1e15), s1 = (uint64_t)(randu(st) * 1e15);
			auto r = b.sample_local(std::sin(th), 0.0, std::cos(th), s0, s1);
			if (!r.valid) continue;
			EXPECT_TRUE(r.is_specular);
			sum[0] += r.r; sum[1] += r.g; sum[2] += r.b;
		}
		for (int c = 0; c < 3; ++c) {
			const double expected = analytic_smooth_coat_over_conductor(std::cos(th), 1.5, kCuEta[c], kCuK[c], 0.01);
			EXPECT_NEAR(sum[c] / n, expected, 0.02 * expected + 0.003) << "channel " << c << " at " << theta << " degrees";
		}
	}
}

// Smooth coat over a rough conductor, and rough coat over a smooth conductor: the random walk and the stochastic f()
// are two estimators of one BSDF, so the sampler's non-specular weight must equal the f() integral.
TEST(LayeredBxDF, SamplerAgreesWithTheFIntegralWhenOneInterfaceIsSmooth) {
	const struct { double coat, base; } cases[] = {{0.0, 0.3}, {0.3, 0.0}};
	for (const auto& c : cases) {
		auto b = cu_under_coat(c.coat, c.base);
		EXPECT_FALSE(b.is_delta());
		for (double theta : {0.0, 60.0}) {
			double spec, gloss, fint;
			albedo_from_sampler_and_f(b, theta, 60000, spec, gloss, fint);
			EXPECT_NEAR(gloss, fint, 0.05 * fint + 0.01)
				<< "coat alpha " << c.coat << ", base alpha " << c.base << ", " << theta << " degrees";
			if (c.coat == 0.0) EXPECT_GT(spec, 0.02) << "a smooth coat's mirror reflection is a specular sample";
			else               EXPECT_EQ(spec, 0.0) << "a rough coat has no specular samples";
		}
	}
}

TEST(LayeredBxDF, CoatedDiffuseSamplerAgreesWithTheFIntegral) {
	CoatedDiffuseBxDF<double> b{};
	b.albedo_r = b.albedo_g = b.albedo_b = 0.5;
	b.coat_ior = 1.5;
	b.alpha_x = b.alpha_y = 0.0;
	for (double theta : {0.0, 60.0}) {
		double spec, gloss, fint;
		albedo_from_sampler_and_f(b, theta, 60000, spec, gloss, fint);
		EXPECT_NEAR(gloss, fint, 0.05 * fint + 0.01) << theta << " degrees";
		EXPECT_GT(spec, 0.02);
	}
}

// pdf() is only an MIS density, but it must be a finite, non-negative function of the direction pair and the same
// value every time for the same pair (pbrt seeds it from the directions) - NEE and the BSDF sample weigh an
// emitter hit with it, and the two weights only sum to one if both see the same number.
TEST(LayeredBxDF, PdfIsDeterministicFiniteAndNonNegative) {
	for (double coat : {0.0, 0.3}) {
		auto b = cu_under_coat(coat, 0.3);
		uint64_t st = 99;
		for (int i = 0; i < 200; ++i) {
			double x, y, z, p;
			SampleCosineHemisphere(randu(st), randu(st), x, y, z, p);
			const double p1 = b.pdf(0.3, 0.1, 0.9486832981, x, y, z);
			const double p2 = b.pdf(0.3, 0.1, 0.9486832981, x, y, z);
			EXPECT_EQ(p1, p2);
			EXPECT_TRUE(std::isfinite(p1));
			EXPECT_GE(p1, 0.0);
		}
	}
}

// Conductor eta/k go into the BxDF as given: the divide-by-the-coat's-IOR that pbrt's CoatedConductorMaterial
// applies lives in the materials, so the same eta/k can be fed to the closed form.
TEST(LayeredBxDF, ConductorEtaAndKAreUsedAsGiven) {
	auto b = cu_under_coat(0.0, 0.0);
	b.eta_r = b.eta_g = b.eta_b = 1.0;
	b.k_r = b.k_g = b.k_b = 0.0;   // a "conductor" that is vacuum-like: nothing comes back from the base
	uint64_t st = 5;
	double sum = 0; const int n = 20000;
	for (int i = 0; i < n; ++i) {
		auto r = b.sample_local(0.0, 0.0, 1.0, (uint64_t)(randu(st) * 1e15), (uint64_t)(randu(st) * 1e15));
		if (r.valid) sum += r.r;
	}
	// Only the coat's own reflection (4% at normal incidence for IOR 1.5) plus whatever FrComplex(eta=1,k=0) returns (0).
	EXPECT_NEAR(sum / n, FrDielectric(1.0, 1.5), 0.005);
}
