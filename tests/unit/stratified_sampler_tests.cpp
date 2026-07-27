// stratified_sampler_tests.cpp
// Unit tests for StratifiedSampler (src/shared/stratified_sampler.h)
// Mirrors pbrt-v4 StratifiedSampler validation approach.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#include "../../src/shared/stratified_sampler.h"

// ---------------------------------------------------------------------------
// Test: samples_per_pixel() == nx * ny
// ---------------------------------------------------------------------------
TEST(StratifiedSampler, SamplesPerPixel) {
	StratifiedSampler s(4, 4, true, 0);
	EXPECT_EQ(s.samples_per_pixel(), 16);

	StratifiedSampler s2(2, 8, true, 0);
	EXPECT_EQ(s2.samples_per_pixel(), 16);

	StratifiedSampler s3(1, 1, false, 0);
	EXPECT_EQ(s3.samples_per_pixel(), 1);
}

// ---------------------------------------------------------------------------
// Test: get_1d() values all in [0, 1)
// ---------------------------------------------------------------------------
TEST(StratifiedSampler, Get1DInRange) {
	StratifiedSampler s(4, 4, true, 42);
	for (int i = 0; i < s.samples_per_pixel(); ++i) {
		s.start_pixel_sample(0, 0, i, 0);
		double v = s.get_1d();
		EXPECT_GE(v, 0.0);
		EXPECT_LT(v, 1.0);
	}
}

// ---------------------------------------------------------------------------
// Test: get_2d() values all in [0,1)^2
// ---------------------------------------------------------------------------
TEST(StratifiedSampler, Get2DInRange) {
	StratifiedSampler s(4, 4, true, 0);
	for (int i = 0; i < s.samples_per_pixel(); ++i) {
		s.start_pixel_sample(3, 7, i, 0);
		auto [x, y] = s.get_2d();
		EXPECT_GE(x, 0.0); EXPECT_LT(x, 1.0);
		EXPECT_GE(y, 0.0); EXPECT_LT(y, 1.0);
	}
}

// ---------------------------------------------------------------------------
// Test: get_1d() stratification — each of the n strata is hit exactly once
//   per pixel (the defining property of stratified sampling)
// ---------------------------------------------------------------------------
TEST(StratifiedSampler, Get1DCoversAllStrata) {
	const int spp = 16;
	StratifiedSampler s(spp, 1, /*jitter=*/false, 0);

	std::vector<bool> stratum_hit(spp, false);
	for (int i = 0; i < spp; ++i) {
		s.start_pixel_sample(5, 3, i, 0);
		double v = s.get_1d();
		int stratum = (int)(v * spp);
		ASSERT_GE(stratum, 0);
		ASSERT_LT(stratum, spp);
		EXPECT_FALSE(stratum_hit[stratum])
			<< "Stratum " << stratum << " hit more than once";
		stratum_hit[stratum] = true;
	}
	for (int i = 0; i < spp; ++i)
		EXPECT_TRUE(stratum_hit[i]) << "Stratum " << i << " never hit";
}

// ---------------------------------------------------------------------------
// Test: get_2d() stratification — each 2D stratum hit exactly once (no jitter)
// ---------------------------------------------------------------------------
TEST(StratifiedSampler, Get2DCoversAllStrata) {
	const int nx = 4, ny = 4;
	StratifiedSampler s(nx, ny, /*jitter=*/false, 0);

	std::vector<bool> hit(nx * ny, false);
	for (int i = 0; i < nx * ny; ++i) {
		s.start_pixel_sample(2, 9, i, 0);
		auto [x, y] = s.get_2d();
		int sx = (int)(x * nx), sy = (int)(y * ny);
		ASSERT_GE(sx, 0); ASSERT_LT(sx, nx);
		ASSERT_GE(sy, 0); ASSERT_LT(sy, ny);
		int idx = sy * nx + sx;
		EXPECT_FALSE(hit[idx])
			<< "2D stratum (" << sx << "," << sy << ") hit more than once";
		hit[idx] = true;
	}
	for (int i = 0; i < nx * ny; ++i)
		EXPECT_TRUE(hit[i]) << "2D stratum " << i << " never hit";
}

// ---------------------------------------------------------------------------
// Test: jitter=false always returns stratum centers (0.5/n in each stratum)
// ---------------------------------------------------------------------------
TEST(StratifiedSampler, NoJitterReturnsCenters) {
	const int spp = 8;
	StratifiedSampler s(spp, 1, /*jitter=*/false, 0);

	for (int i = 0; i < spp; ++i) {
		s.start_pixel_sample(0, 0, i, 0);
		double v = s.get_1d();
		// v = (stratum + 0.5) / spp, so v * spp - floor(v * spp) == 0.5
		double frac = v * spp - std::floor(v * spp);
		EXPECT_NEAR(frac, 0.5, 1e-10)
			<< "Sample " << i << " not at stratum center: v=" << v;
	}
}

// ---------------------------------------------------------------------------
// Test: determinism — same pixel/sample/seed always gives same result
// ---------------------------------------------------------------------------
TEST(StratifiedSampler, Deterministic) {
	StratifiedSampler s1(4, 4, true, 7);
	StratifiedSampler s2(4, 4, true, 7);

	for (int i = 0; i < 16; ++i) {
		s1.start_pixel_sample(3, 5, i, 0);
		s2.start_pixel_sample(3, 5, i, 0);
		double a = s1.get_1d();
		double b = s2.get_1d();
		EXPECT_EQ(a, b) << "Sample " << i << " not deterministic";
	}
}

// ---------------------------------------------------------------------------
// Test: different pixels produce decorrelated samples (not all identical)
// ---------------------------------------------------------------------------
TEST(StratifiedSampler, DifferentPixelsDifferentSamples) {
	StratifiedSampler s(4, 4, true, 0);

	s.start_pixel_sample(0, 0, 0, 0);
	double v00 = s.get_1d();

	s.start_pixel_sample(1, 0, 0, 0);
	double v10 = s.get_1d();

	s.start_pixel_sample(0, 1, 0, 0);
	double v01 = s.get_1d();

	// Very unlikely all three are equal if decorrelation works
	EXPECT_FALSE(v00 == v10 && v10 == v01)
		<< "Samples from different pixels should differ";
}

// ---------------------------------------------------------------------------
// Test: different seeds produce different strata ordering
// ---------------------------------------------------------------------------
TEST(StratifiedSampler, DifferentSeedsDifferentOrder) {
	StratifiedSampler s0(8, 1, false, 0);
	StratifiedSampler s1(8, 1, false, 42);

	std::vector<double> vals0, vals1;
	for (int i = 0; i < 8; ++i) {
		s0.start_pixel_sample(0, 0, i, 0); vals0.push_back(s0.get_1d());
		s1.start_pixel_sample(0, 0, i, 0); vals1.push_back(s1.get_1d());
	}
	EXPECT_NE(vals0, vals1) << "Different seeds should produce different orderings";
}

// ---------------------------------------------------------------------------
// Test: multiple dimensions are independent (get_1d called twice per sample)
// ---------------------------------------------------------------------------
TEST(StratifiedSampler, MultipleDimensionsIndependent) {
	const int spp = 16;
	StratifiedSampler s(spp, 1, false, 0);

	// Collect (dim0, dim1) pairs and check both are stratified
	std::vector<bool> hit0(spp, false), hit1(spp, false);
	for (int i = 0; i < spp; ++i) {
		s.start_pixel_sample(0, 0, i, 0);
		double v0 = s.get_1d();
		double v1 = s.get_1d();
		int s0 = (int)(v0 * spp), s1 = (int)(v1 * spp);
		ASSERT_GE(s0, 0); ASSERT_LT(s0, spp);
		ASSERT_GE(s1, 0); ASSERT_LT(s1, spp);
		hit0[s0] = true;
		hit1[s1] = true;
	}
	for (int i = 0; i < spp; ++i) {
		EXPECT_TRUE(hit0[i]) << "Dim0 stratum " << i << " never hit";
		EXPECT_TRUE(hit1[i]) << "Dim1 stratum " << i << " never hit";
	}
}

// ---------------------------------------------------------------------------
// Test: get_pixel_2d() same as get_2d() for stratified sampler
// ---------------------------------------------------------------------------
TEST(StratifiedSampler, GetPixel2DSameAsGet2D) {
	StratifiedSampler s1(4, 4, true, 3);
	StratifiedSampler s2(4, 4, true, 3);

	for (int i = 0; i < 16; ++i) {
		s1.start_pixel_sample(1, 2, i, 0);
		s2.start_pixel_sample(1, 2, i, 0);
		auto [x1, y1] = s1.get_pixel_2d();
		auto [x2, y2] = s2.get_2d();
		EXPECT_EQ(x1, x2);
		EXPECT_EQ(y1, y2);
	}
}

// ---------------------------------------------------------------------------
// Test: variance of get_1d is lower than random (stratified reduces variance)
//   For n=16 strata, variance of stratified should be < variance of uniform.
//   We measure empirical variance over pixels and compare.
// ---------------------------------------------------------------------------
TEST(StratifiedSampler, StratifiedReducesVariance) {
	const int spp = 64;
	StratifiedSampler s(spp, 1, true, 0);

	// Collect one get_1d() per sample, averaged over a pixel
	// True E[X] = 0.5 for uniform [0,1)
	// Stratified should give lower variance than iid uniform
	double sum_sq_dev = 0.0;
	const int n_pixels = 50;

	for (int p = 0; p < n_pixels; ++p) {
		double pixel_mean = 0.0;
		for (int i = 0; i < spp; ++i) {
			s.start_pixel_sample(p, 0, i, 0);
			pixel_mean += s.get_1d();
		}
		pixel_mean /= spp;
		double dev = pixel_mean - 0.5;
		sum_sq_dev += dev * dev;
	}
	double pixel_var = sum_sq_dev / n_pixels;

	// Theoretical variance of mean of n stratified [0,1) samples = 1/(12*n^2)
	// Theoretical variance of mean of n iid uniform = 1/(12*n)
	// Stratified should be significantly smaller.
	double iid_var = 1.0 / (12.0 * spp);
	EXPECT_LT(pixel_var, iid_var)
		<< "Stratified pixel mean variance=" << pixel_var
		<< " should be less than iid variance=" << iid_var;
}
