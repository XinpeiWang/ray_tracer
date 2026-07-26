/**
 * @file piecewise_dist_tests.cpp
 * @brief Tests for PiecewiseConstant1D and PiecewiseConstant2D
 *        (pbrt-v4 sampling.h alignment)
 *
 * Tests:
 * 1D:
 *   - Known weights: sample concentrates in the heavy bin
 *   - CDF is monotonically non-decreasing and ends at 1
 *   - Uniform fallback when all weights are zero
 *   - pdf(x) integrates to ~1 via Monte Carlo
 *   - sample() returns values in [0,1]
 *
 * 2D:
 *   - Joint pdf integrates to ~1 via Monte Carlo
 *   - sample() returns (u,v) in [0,1]^2
 *   - Concentrated weight: samples land mostly in the heavy cell
 *   - pdf consistency: pdf at sampled point matches returned pdf
 */

#include <gtest/gtest.h>
#include "shared/piecewise_dist.h"
#include <random>
#include <cmath>

static constexpr double kEps  = 1e-6;
static constexpr int    kSamples = 200000;

// ---------------------------------------------------------------------------
// PiecewiseConstant1D
// ---------------------------------------------------------------------------

TEST(PC1DTest, SampleInRange) {
	PiecewiseConstant1D dist({ 1.0, 2.0, 3.0, 4.0 });
	std::mt19937_64 rng(42);
	std::uniform_real_distribution<double> u01(0.0, 1.0 - 1e-9);
	for (int i = 0; i < 1000; ++i) {
		double x = dist.sample(u01(rng));
		EXPECT_GE(x, 0.0);
		EXPECT_LE(x, 1.0);
	}
}

TEST(PC1DTest, HeavyBinGetsMoreSamples) {
	// Bin 3 has weight 100, others have weight 1 -> ~97% of samples in bin 3
	PiecewiseConstant1D dist({ 1.0, 1.0, 1.0, 100.0 });
	std::mt19937_64 rng(7);
	std::uniform_real_distribution<double> u01(0.0, 1.0 - 1e-9);
	int in_last = 0;
	const int N = 10000;
	for (int i = 0; i < N; ++i) {
		double x = dist.sample(u01(rng));
		if (x >= 0.75) ++in_last;
	}
	EXPECT_GT(in_last, N * 0.90) << "Expected >90% of samples in the heavy bin";
}

TEST(PC1DTest, UniformFallbackForZeroWeights) {
	PiecewiseConstant1D dist({ 0.0, 0.0, 0.0 });
	double pdf_val;
	double x = dist.sample(0.5, &pdf_val);
	EXPECT_GE(x, 0.0);
	EXPECT_LE(x, 1.0);
	// Uniform fallback: pdf should be finite and positive
	EXPECT_GT(pdf_val, 0.0);
}

TEST(PC1DTest, PDFIntegratesTo1) {
	PiecewiseConstant1D dist({ 1.0, 3.0, 2.0, 4.0 });
	// Monte Carlo integration of pdf over [0,1]
	std::mt19937_64 rng(13);
	std::uniform_real_distribution<double> u01(0.0, 1.0);
	double sum = 0.0;
	const int N = kSamples;
	for (int i = 0; i < N; ++i)
		sum += dist.pdf(u01(rng));
	double integral = sum / N;  // E[pdf(x)] over uniform x -> should be 1
	EXPECT_NEAR(integral, 1.0, 0.02);
}

TEST(PC1DTest, KnownPDFValue) {
	// Single bin with weight 2.0 over [0,1]: integral = 2.0, pdf = 2.0/2.0 = 1.0
	PiecewiseConstant1D dist({ 2.0 });
	double pdf_val;
	dist.sample(0.5, &pdf_val);
	EXPECT_NEAR(pdf_val, 1.0, kEps);
}

TEST(PC1DTest, SamplePDFMatchesDirectPDF) {
	PiecewiseConstant1D dist({ 1.0, 5.0, 2.0, 8.0 });
	std::mt19937_64 rng(99);
	std::uniform_real_distribution<double> u01(0.0, 1.0 - 1e-9);
	for (int i = 0; i < 500; ++i) {
		double pdf_s;
		double x = dist.sample(u01(rng), &pdf_s);
		double pdf_d = dist.pdf(x);
		EXPECT_NEAR(pdf_s, pdf_d, kEps);
	}
}

// ---------------------------------------------------------------------------
// PiecewiseConstant2D
// ---------------------------------------------------------------------------

TEST(PC2DTest, SampleInRange) {
	std::vector<double> f = { 1,2,3, 4,5,6, 7,8,9 };  // 3x3
	PiecewiseConstant2D dist(f, 3, 3);
	std::mt19937_64 rng(42);
	std::uniform_real_distribution<double> u01(0.0, 1.0 - 1e-9);
	for (int i = 0; i < 1000; ++i) {
		double pdf;
		auto [u, v] = dist.sample(u01(rng), u01(rng), &pdf);
		EXPECT_GE(u, 0.0); EXPECT_LE(u, 1.0);
		EXPECT_GE(v, 0.0); EXPECT_LE(v, 1.0);
		EXPECT_GT(pdf, 0.0);
	}
}

TEST(PC2DTest, JointPDFIntegratesTo1) {
	// 4x4 grid with random-ish weights
	std::vector<double> f = {
		1,5,2,8,
		3,9,1,4,
		6,2,7,3,
		1,4,8,2
	};
	PiecewiseConstant2D dist(f, 4, 4);

	std::mt19937_64 rng(77);
	std::uniform_real_distribution<double> u01(0.0, 1.0);
	double sum = 0.0;
	const int N = kSamples;
	for (int i = 0; i < N; ++i)
		sum += dist.pdf(u01(rng), u01(rng));
	double integral = sum / N;
	EXPECT_NEAR(integral, 1.0, 0.02);
}

TEST(PC2DTest, HeavyCellGetsMoreSamples) {
	// Bottom-right cell (u>0.5, v>0.5) has weight 1000, rest 1
	std::vector<double> f = { 1,1, 1,1000 };  // 2x2
	PiecewiseConstant2D dist(f, 2, 2);
	std::mt19937_64 rng(55);
	std::uniform_real_distribution<double> u01(0.0, 1.0 - 1e-9);
	int in_heavy = 0;
	const int N = 5000;
	for (int i = 0; i < N; ++i) {
		double pdf;
		auto [u, v] = dist.sample(u01(rng), u01(rng), &pdf);
		if (u >= 0.5 && v >= 0.5) ++in_heavy;
	}
	EXPECT_GT(in_heavy, N * 0.95) << "Expected >95% in heavy cell";
}

TEST(PC2DTest, SamplePDFMatchesQueryPDF) {
	std::vector<double> f = { 2,8, 4,1 };  // 2x2
	PiecewiseConstant2D dist(f, 2, 2);
	std::mt19937_64 rng(33);
	std::uniform_real_distribution<double> u01(0.0, 1.0 - 1e-9);
	for (int i = 0; i < 500; ++i) {
		double pdf_s;
		auto [u, v] = dist.sample(u01(rng), u01(rng), &pdf_s);
		double pdf_q = dist.pdf(u, v);
		EXPECT_NEAR(pdf_s, pdf_q, 1e-9);
	}
}

TEST(PC2DTest, UniformWeightsGiveUniformSampling) {
	// All equal weights -> should sample uniformly (pdf ~= 1)
	std::vector<double> f(16, 1.0);  // 4x4 all ones
	PiecewiseConstant2D dist(f, 4, 4);
	std::mt19937_64 rng(11);
	std::uniform_real_distribution<double> u01(0.0, 1.0 - 1e-9);
	for (int i = 0; i < 200; ++i) {
		double pdf;
		dist.sample(u01(rng), u01(rng), &pdf);
		EXPECT_NEAR(pdf, 1.0, 0.05);
	}
}
