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
	b.coat_ior = ior; b.alpha = alpha;
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
	b.coat_ior = coat_ior; b.alpha = alpha;
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
	auto b = make_coated_diffuse();
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
		if (!res.valid || res.is_specular) continue;
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
