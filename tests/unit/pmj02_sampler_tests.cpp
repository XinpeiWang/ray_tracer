// pmj02_sampler_tests.cpp
// Unit tests for src/shared/pmj02_sampler.h
//
// pbrt-v4 reference: src/pbrt/samplers.h (PMJ02BNSampler)
//                    src/pbrt/util/pmj02tables.h
//
// Tests:
//   Table
//   1.  Table_SampleInRange          : GetPMJ02BNSample always in [0,1)
//   2.  Table_SetIndependence        : different sets produce different samples
//   3.  Table_SampleIndexWraps       : index wraps mod 65536
//   4.  Table_SetIndexWraps          : set index wraps mod 5
//
//   Sampler
//   5.  Sampler_Get1D_InRange        : get_1d() always in [0,1)
//   6.  Sampler_Get2D_InRange        : get_2d() both components in [0,1)
//   7.  Sampler_GetPixel2D_InRange   : get_pixel_2d() in [0,1)
//   8.  Sampler_DimensionAdvances    : consecutive get_1d() differ
//   9.  Sampler_PixelsDiffer         : adjacent pixels yield different samples
//   10. Sampler_Reproducible         : same pixel+index repeats exactly
//   11. Sampler_1D_LowVariance       : mean of 64 samples close to 0.5
//   12. Sampler_2D_LowVariance       : 2D mean close to (0.5, 0.5)
//   13. Sampler_Pixel2D_LowVariance  : pixel2d mean close to (0.5, 0.5)
//   14. Sampler_1D_Stratified        : 64 samples have small max gap
//   15. Sampler_PowerOf4_Warning     : power-of-4 spp accepted without error
//   16. Sampler_NonPow4_Accepted     : non-power-of-4 spp still constructs OK
//   17. Sampler_SeedsDiffer          : different seeds yield different samples

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>

#include "../../src/shared/pmj02_sampler.h"

// Helper to access the internal table lookup
static std::pair<float,float> sample_table(int set, int idx) {
	return pmj02_detail::get_pmj02bn_sample(set, idx);
}

// ---------------------------------------------------------------------------
// 1. Table_SampleInRange
// ---------------------------------------------------------------------------
TEST(PMJ02Table, Table_SampleInRange) {
	for (int s = 0; s < 5; ++s)
		for (int i = 0; i < 65536; i += 256) {
			auto [u, v] = sample_table(s, i);
			EXPECT_GE(u, 0.0f) << "set=" << s << " i=" << i;
			EXPECT_LT(u, 1.0f) << "set=" << s << " i=" << i;
			EXPECT_GE(v, 0.0f) << "set=" << s << " i=" << i;
			EXPECT_LT(v, 1.0f) << "set=" << s << " i=" << i;
		}
}

// ---------------------------------------------------------------------------
// 2. Table_SetIndependence
// ---------------------------------------------------------------------------
TEST(PMJ02Table, Table_SetIndependence) {
	int same = 0;
	for (int i = 0; i < 256; ++i) {
		auto [u0, v0] = sample_table(0, i);
		auto [u1, v1] = sample_table(1, i);
		if (u0 == u1 && v0 == v1) ++same;
	}
	EXPECT_LT(same, 5) << "Sets should be mostly independent";
}

// ---------------------------------------------------------------------------
// 3. Table_SampleIndexWraps
// ---------------------------------------------------------------------------
TEST(PMJ02Table, Table_SampleIndexWraps) {
	for (int i = 0; i < 16; ++i) {
		auto [u0, v0] = sample_table(0, i);
		auto [u1, v1] = sample_table(0, i + 65536);
		EXPECT_FLOAT_EQ(u0, u1);
		EXPECT_FLOAT_EQ(v0, v1);
	}
}

// ---------------------------------------------------------------------------
// 4. Table_SetIndexWraps
// ---------------------------------------------------------------------------
TEST(PMJ02Table, Table_SetIndexWraps) {
	for (int i = 0; i < 32; ++i) {
		auto [u0, v0] = sample_table(0, i);
		auto [u1, v1] = sample_table(5, i);   // 5 % 5 == 0
		EXPECT_FLOAT_EQ(u0, u1);
		EXPECT_FLOAT_EQ(v0, v1);
	}
}

// ---------------------------------------------------------------------------
// 5. Sampler_Get1D_InRange
// ---------------------------------------------------------------------------
TEST(PMJ02BNSamplerTest, Sampler_Get1D_InRange) {
	PMJ02BNSampler s(16);
	for (int px = 0; px < 4; ++px)
		for (int py = 0; py < 4; ++py)
			for (int i = 0; i < 16; ++i) {
				s.start_pixel_sample(px, py, i);
				float u = s.get_1d();
				EXPECT_GE(u, 0.0f); EXPECT_LT(u, 1.0f);
			}
}

// ---------------------------------------------------------------------------
// 6. Sampler_Get2D_InRange
// ---------------------------------------------------------------------------
TEST(PMJ02BNSamplerTest, Sampler_Get2D_InRange) {
	PMJ02BNSampler s(16);
	s.start_pixel_sample(3, 7, 0);
	for (int i = 0; i < 20; ++i) {
		auto [u, v] = s.get_2d();
		EXPECT_GE(u, 0.0f); EXPECT_LT(u, 1.0f);
		EXPECT_GE(v, 0.0f); EXPECT_LT(v, 1.0f);
	}
}

// ---------------------------------------------------------------------------
// 7. Sampler_GetPixel2D_InRange
// ---------------------------------------------------------------------------
TEST(PMJ02BNSamplerTest, Sampler_GetPixel2D_InRange) {
	PMJ02BNSampler s(16);
	for (int px = 0; px < 8; ++px)
		for (int py = 0; py < 8; ++py)
			for (int i = 0; i < 16; ++i) {
				s.start_pixel_sample(px, py, i);
				auto [u, v] = s.get_pixel_2d();
				EXPECT_GE(u, 0.0f); EXPECT_LT(u, 1.0f);
				EXPECT_GE(v, 0.0f); EXPECT_LT(v, 1.0f);
			}
}

// ---------------------------------------------------------------------------
// 8. Sampler_DimensionAdvances
// ---------------------------------------------------------------------------
TEST(PMJ02BNSamplerTest, Sampler_DimensionAdvances) {
	PMJ02BNSampler s(16);
	s.start_pixel_sample(0, 0, 0);
	float a = s.get_1d(), b = s.get_1d(), c = s.get_1d();
	EXPECT_TRUE(a != b || b != c) << "All dimensions should not be identical";
}

// ---------------------------------------------------------------------------
// 9. Sampler_PixelsDiffer
// ---------------------------------------------------------------------------
TEST(PMJ02BNSamplerTest, Sampler_PixelsDiffer) {
	PMJ02BNSampler s(16);
	int same = 0;
	for (int i = 0; i < 16; ++i) {
		s.start_pixel_sample(0, 0, i); float a = s.get_1d();
		s.start_pixel_sample(1, 0, i); float b = s.get_1d();
		if (a == b) ++same;
	}
	EXPECT_LT(same, 4) << "Adjacent pixels should mostly differ";
}

// ---------------------------------------------------------------------------
// 10. Sampler_Reproducible
// ---------------------------------------------------------------------------
TEST(PMJ02BNSamplerTest, Sampler_Reproducible) {
	PMJ02BNSampler s(16);
	s.start_pixel_sample(5, 3, 7);
	float a0 = s.get_1d(); auto [a1, a2] = s.get_2d();

	s.start_pixel_sample(5, 3, 7);
	float b0 = s.get_1d(); auto [b1, b2] = s.get_2d();

	EXPECT_FLOAT_EQ(a0, b0);
	EXPECT_FLOAT_EQ(a1, b1);
	EXPECT_FLOAT_EQ(a2, b2);
}

// ---------------------------------------------------------------------------
// 11. Sampler_1D_LowVariance
// ---------------------------------------------------------------------------
TEST(PMJ02BNSamplerTest, Sampler_1D_LowVariance) {
	PMJ02BNSampler s(64);
	double sum = 0.0;
	for (int i = 0; i < 64; ++i) {
		s.start_pixel_sample(10, 20, i);
		sum += s.get_1d();
	}
	EXPECT_NEAR(sum / 64.0, 0.5, 0.05);
}

// ---------------------------------------------------------------------------
// 12. Sampler_2D_LowVariance
// ---------------------------------------------------------------------------
TEST(PMJ02BNSamplerTest, Sampler_2D_LowVariance) {
	PMJ02BNSampler s(64);
	double su = 0.0, sv = 0.0;
	for (int i = 0; i < 64; ++i) {
		s.start_pixel_sample(3, 5, i);
		auto [u, v] = s.get_2d();
		su += u; sv += v;
	}
	EXPECT_NEAR(su / 64.0, 0.5, 0.05);
	EXPECT_NEAR(sv / 64.0, 0.5, 0.05);
}

// ---------------------------------------------------------------------------
// 13. Sampler_Pixel2D_LowVariance
// ---------------------------------------------------------------------------
TEST(PMJ02BNSamplerTest, Sampler_Pixel2D_LowVariance) {
	PMJ02BNSampler s(64);
	double su = 0.0, sv = 0.0;
	for (int i = 0; i < 64; ++i) {
		s.start_pixel_sample(7, 11, i);
		auto [u, v] = s.get_pixel_2d();
		su += u; sv += v;
	}
	EXPECT_NEAR(su / 64.0, 0.5, 0.06);
	EXPECT_NEAR(sv / 64.0, 0.5, 0.06);
}

// ---------------------------------------------------------------------------
// 14. Sampler_1D_Stratified
// ---------------------------------------------------------------------------
TEST(PMJ02BNSamplerTest, Sampler_1D_Stratified) {
	const int N = 64;
	PMJ02BNSampler s(N);
	std::vector<float> vals(N);
	for (int i = 0; i < N; ++i) {
		s.start_pixel_sample(0, 0, i);
		vals[i] = s.get_1d();
	}
	std::sort(vals.begin(), vals.end());
	float max_gap = 0.0f;
	for (int i = 1; i < N; ++i)
		max_gap = std::max(max_gap, vals[i] - vals[i - 1]);
	// For N=64 pure random, expected max gap ~0.17; stratified should be tighter
	EXPECT_LT(max_gap, 0.12f) << "Max gap " << max_gap << " suggests poor stratification";
}

// ---------------------------------------------------------------------------
// 15. Sampler_PowerOf4_Accepted
// ---------------------------------------------------------------------------
TEST(PMJ02BNSamplerTest, Sampler_PowerOf4_Accepted) {
	for (int spp : {1, 4, 16, 64, 256, 1024}) {
		PMJ02BNSampler s(spp);
		EXPECT_EQ(s.samples_per_pixel(), spp);
		EXPECT_TRUE(s.is_power_of4_spp());
		s.start_pixel_sample(0, 0, 0);
		float u = s.get_1d();
		EXPECT_GE(u, 0.0f); EXPECT_LT(u, 1.0f);
	}
}

// ---------------------------------------------------------------------------
// 16. Sampler_NonPow4_Accepted
// ---------------------------------------------------------------------------
TEST(PMJ02BNSamplerTest, Sampler_NonPow4_Accepted) {
	// Non-power-of-4 spp constructs without crash; quality is degraded but valid
	PMJ02BNSampler s(10);
	s.start_pixel_sample(0, 0, 0);
	float u = s.get_1d();
	EXPECT_GE(u, 0.0f); EXPECT_LT(u, 1.0f);
	auto [a, b] = s.get_2d();
	EXPECT_GE(a, 0.0f); EXPECT_LT(a, 1.0f);
}

// ---------------------------------------------------------------------------
// 17. Sampler_SeedsDiffer
// ---------------------------------------------------------------------------
TEST(PMJ02BNSamplerTest, Sampler_SeedsDiffer) {
	PMJ02BNSampler s0(16, 0);
	PMJ02BNSampler s1(16, 42);
	int same = 0;
	for (int i = 0; i < 16; ++i) {
		s0.start_pixel_sample(3, 7, i); float a = s0.get_1d();
		s1.start_pixel_sample(3, 7, i); float b = s1.get_1d();
		if (a == b) ++same;
	}
	EXPECT_LT(same, 4) << "Different seeds should produce different samples";
}
