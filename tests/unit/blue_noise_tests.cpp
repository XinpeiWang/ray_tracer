// blue_noise_tests.cpp
// Unit tests for src/shared/bluenoise.h and src/shared/bluenoise_sampler.h
//
// pbrt-v4 reference: src/pbrt/util/bluenoise.h, src/pbrt/samplers.h
//
// Tests:
//   1.  TableLookup_InRange           : blue_noise() always in [0, 1)
//   2.  TableLookup_Tiling            : 128-pixel tile wraps correctly
//   3.  TableLookup_NegativeCoords    : negative pixel coords wrap correctly
//   4.  TableLookup_TableIndexWraps   : tableIndex wraps mod 48
//   5.  TableLookup_IndependentTables : two different tables differ
//   6.  TableLookup_NotConstant       : values vary across a row
//   7.  Sampler_Get1D_InRange         : get_1d() in [0, 1)
//   8.  Sampler_Get2D_InRange         : get_2d() both components in [0, 1)
//   9.  Sampler_DimensionAdvances     : consecutive calls advance dimension
//  10.  Sampler_PixelOffsetsDiffer    : different pixels yield different samples
//  11.  Sampler_SampleIndexProgresses : different sample indices differ
//  12.  Sampler_LowVariance_1D        : mean of 256 samples close to 0.5
//  13.  Sampler_LowVariance_2D        : 2D mean close to (0.5, 0.5)
//  14.  Sampler_Stratified_1D         : 64 samples span [0,1) evenly
//  15.  Sampler_Reset_Reproducible    : same pixel+index yields same value

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <set>

#include "../../src/shared/bluenoise.h"
#include "../../src/shared/bluenoise_sampler.h"

// ---------------------------------------------------------------------------
// 1. TableLookup_InRange
// ---------------------------------------------------------------------------
TEST(BlueNoiseTable, TableLookup_InRange) {
	for (int t = 0; t < 48; ++t)
		for (int x = 0; x < 128; x += 8)
			for (int y = 0; y < 128; y += 8) {
				float v = blue_noise(t, x, y);
				EXPECT_GE(v, 0.0f) << "t=" << t << " x=" << x << " y=" << y;
				EXPECT_LT(v, 1.0f) << "t=" << t << " x=" << x << " y=" << y;
			}
}

// ---------------------------------------------------------------------------
// 2. TableLookup_Tiling
// ---------------------------------------------------------------------------
TEST(BlueNoiseTable, TableLookup_Tiling) {
	for (int t = 0; t < 48; t += 7)
		for (int x = 0; x < 128; x += 16)
			for (int y = 0; y < 128; y += 16) {
				EXPECT_FLOAT_EQ(blue_noise(t, x, y), blue_noise(t, x + 128, y))
					<< "x tiling failed at t=" << t;
				EXPECT_FLOAT_EQ(blue_noise(t, x, y), blue_noise(t, x, y + 128))
					<< "y tiling failed at t=" << t;
				EXPECT_FLOAT_EQ(blue_noise(t, x, y), blue_noise(t, x + 256, y + 256))
					<< "double tiling failed at t=" << t;
			}
}

// ---------------------------------------------------------------------------
// 3. TableLookup_NegativeCoords
// ---------------------------------------------------------------------------
TEST(BlueNoiseTable, TableLookup_NegativeCoords) {
	// blue_noise(-1, -1) should wrap to (127, 127)
	float expected = blue_noise(0, 127, 127);
	float actual   = blue_noise(0, -1, -1);
	EXPECT_FLOAT_EQ(expected, actual);
}

// ---------------------------------------------------------------------------
// 4. TableLookup_TableIndexWraps
// ---------------------------------------------------------------------------
TEST(BlueNoiseTable, TableLookup_TableIndexWraps) {
	for (int x = 0; x < 16; ++x)
		for (int y = 0; y < 16; ++y)
			EXPECT_FLOAT_EQ(blue_noise(0, x, y), blue_noise(48, x, y));
}

// ---------------------------------------------------------------------------
// 5. TableLookup_IndependentTables
// ---------------------------------------------------------------------------
TEST(BlueNoiseTable, TableLookup_IndependentTables) {
	int diff = 0;
	for (int x = 0; x < 128; x += 4)
		for (int y = 0; y < 128; y += 4)
			if (blue_noise(0, x, y) != blue_noise(1, x, y)) ++diff;
	// Virtually all texels should differ between two independent tables
	EXPECT_GT(diff, 900);
}

// ---------------------------------------------------------------------------
// 6. TableLookup_NotConstant
// ---------------------------------------------------------------------------
TEST(BlueNoiseTable, TableLookup_NotConstant) {
	float v0 = blue_noise(0, 0, 0);
	int   diff = 0;
	for (int x = 0; x < 128; ++x)
		if (blue_noise(0, x, 0) != v0) ++diff;
	EXPECT_GT(diff, 100) << "Row 0 should not be constant";
}

// ---------------------------------------------------------------------------
// 7. Sampler_Get1D_InRange
// ---------------------------------------------------------------------------
TEST(BlueNoiseSampler, Sampler_Get1D_InRange) {
	BlueNoiseSampler s(64);
	for (int px = 0; px < 8; ++px)
		for (int py = 0; py < 8; ++py)
			for (int i = 0; i < 64; ++i) {
				s.start_pixel_sample(px, py, i);
				float u = s.get_1d();
				EXPECT_GE(u, 0.0f);
				EXPECT_LT(u, 1.0f);
			}
}

// ---------------------------------------------------------------------------
// 8. Sampler_Get2D_InRange
// ---------------------------------------------------------------------------
TEST(BlueNoiseSampler, Sampler_Get2D_InRange) {
	BlueNoiseSampler s(64);
	s.start_pixel_sample(5, 7, 0);
	for (int i = 0; i < 64; ++i) {
		auto [u, v] = s.get_2d();
		EXPECT_GE(u, 0.0f); EXPECT_LT(u, 1.0f);
		EXPECT_GE(v, 0.0f); EXPECT_LT(v, 1.0f);
	}
}

// ---------------------------------------------------------------------------
// 9. Sampler_DimensionAdvances
// ---------------------------------------------------------------------------
TEST(BlueNoiseSampler, Sampler_DimensionAdvances) {
	BlueNoiseSampler s(4);
	s.start_pixel_sample(0, 0, 0);
	float a = s.get_1d();
	float b = s.get_1d();
	float c = s.get_1d();
	// Three consecutive dimensions should (almost certainly) differ
	EXPECT_TRUE(a != b || b != c) << "All dimensions returned the same value";
}

// ---------------------------------------------------------------------------
// 10. Sampler_PixelOffsetsDiffer
// ---------------------------------------------------------------------------
TEST(BlueNoiseSampler, Sampler_PixelOffsetsDiffer) {
	BlueNoiseSampler s(64);
	int same = 0;
	for (int i = 0; i < 32; ++i) {
		s.start_pixel_sample(0, 0, i); float a = s.get_1d();
		s.start_pixel_sample(1, 0, i); float b = s.get_1d();
		if (a == b) ++same;
	}
	EXPECT_LT(same, 4) << "Adjacent pixels should mostly differ";
}

// ---------------------------------------------------------------------------
// 11. Sampler_SampleIndexProgresses
// ---------------------------------------------------------------------------
TEST(BlueNoiseSampler, Sampler_SampleIndexProgresses) {
	BlueNoiseSampler s(64);
	std::set<float> vals;
	for (int i = 0; i < 64; ++i) {
		s.start_pixel_sample(3, 7, i);
		vals.insert(s.get_1d());
	}
	// 64 samples should produce many distinct values
	EXPECT_GT((int)vals.size(), 32);
}

// ---------------------------------------------------------------------------
// 12. Sampler_LowVariance_1D
// ---------------------------------------------------------------------------
TEST(BlueNoiseSampler, Sampler_LowVariance_1D) {
	BlueNoiseSampler s(256);
	double sum = 0.0;
	for (int i = 0; i < 256; ++i) {
		s.start_pixel_sample(10, 20, i);
		sum += s.get_1d();
	}
	double mean = sum / 256.0;
	EXPECT_NEAR(mean, 0.5, 0.05) << "Mean of 256 BN samples should be near 0.5";
}

// ---------------------------------------------------------------------------
// 13. Sampler_LowVariance_2D
// ---------------------------------------------------------------------------
TEST(BlueNoiseSampler, Sampler_LowVariance_2D) {
	BlueNoiseSampler s(256);
	double su = 0.0, sv = 0.0;
	for (int i = 0; i < 256; ++i) {
		s.start_pixel_sample(3, 5, i);
		auto [u, v] = s.get_2d();
		su += u; sv += v;
	}
	EXPECT_NEAR(su / 256.0, 0.5, 0.05);
	EXPECT_NEAR(sv / 256.0, 0.5, 0.05);
}

// ---------------------------------------------------------------------------
// 14. Sampler_Stratified_1D
// ---------------------------------------------------------------------------
TEST(BlueNoiseSampler, Sampler_Stratified_1D) {
	// 64 samples from one pixel should cover [0,1) without huge gaps
	const int N = 64;
	BlueNoiseSampler s(N);
	std::vector<float> vals(N);
	for (int i = 0; i < N; ++i) {
		s.start_pixel_sample(0, 0, i);
		vals[i] = s.get_1d();
	}
	std::sort(vals.begin(), vals.end());
	// Max gap between sorted samples should be smaller than for pure random
	float max_gap = 0.0f;
	for (int i = 1; i < N; ++i)
		max_gap = std::max(max_gap, vals[i] - vals[i-1]);
	// For N=64 pure random, expected max gap ≈ 0.17; LD sequences are tighter
	EXPECT_LT(max_gap, 0.15f) << "Max gap " << max_gap << " suggests poor stratification";
}

// ---------------------------------------------------------------------------
// 15. Sampler_Reset_Reproducible
// ---------------------------------------------------------------------------
TEST(BlueNoiseSampler, Sampler_Reset_Reproducible) {
	BlueNoiseSampler s(16);
	s.start_pixel_sample(7, 3, 5);
	float a0 = s.get_1d(), a1 = s.get_1d();

	s.start_pixel_sample(7, 3, 5);
	float b0 = s.get_1d(), b1 = s.get_1d();

	EXPECT_FLOAT_EQ(a0, b0);
	EXPECT_FLOAT_EQ(a1, b1);
}
