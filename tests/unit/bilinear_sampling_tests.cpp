// bilinear_sampling_tests.cpp
// Validation for bilinear importance sampling and SampleUniformTriangle
// -- pbrt-v4 util/sampling.h ports
//
// Tests:
//   1.  LinearPDF: non-negative, integrates to 1
//   2.  SampleLinear: output in [0,1), uniform weights -> uniform samples
//   3.  SampleLinear/InvertLinearSample are inverses
//   4.  BilinearPDF: non-negative, integrates to 1 (Monte Carlo)
//   5.  BilinearPDF: zero outside [0,1]^2
//   6.  BilinearPDF: uniform weights -> constant 1
//   7.  SampleBilinear: output in [0,1)^2
//   8.  SampleBilinear/InvertBilinearSample are inverses
//   9.  SampleBilinear uniform weights -> approximate uniform distribution
//  10.  SampleUniformTriangle: barycentrics sum to 1, all non-negative
//  11.  SampleUniformTriangle: covers triangle uniformly (area test)
//  12.  SampleUniformTriangle: corner cases (u0=0, u1=0)

#include <gtest/gtest.h>
#include "../../src/shared/sampling.h"
#include <vector>
#include <cmath>
#include <numeric>

static const double kPi = 3.14159265358979323846;

// ---- 1. LinearPDF non-negative and integrates to 1 ------------------------
TEST(BilinearSamplingTest, LinearPDFIntegratesTo1) {
	// Numerical integration with 1000 points
	const int N = 1000;
	double a = 0.3, b = 1.7;
	double sum = 0.0;
	for (int i = 0; i < N; ++i) {
		double x = (i + 0.5) / N;
		double pdf = LinearPDF(x, a, b);
		EXPECT_GE(pdf, 0.0);
		sum += pdf / N;
	}
	EXPECT_NEAR(sum, 1.0, 0.01);
}

// ---- 2. SampleLinear in [0,1) with uniform weights -> uniform samples ------
TEST(BilinearSamplingTest, SampleLinearRangeUniform) {
	for (int i = 0; i <= 100; ++i) {
		double u = i / 100.0;
		double x = SampleLinear(u, 1.0, 1.0);
		EXPECT_GE(x, 0.0);
		EXPECT_LT(x, 1.0);
	}
	// With equal weights, sample should be close to u (linear distribution is uniform)
	EXPECT_NEAR(SampleLinear(0.5, 1.0, 1.0), 0.5, 0.01);
	EXPECT_NEAR(SampleLinear(0.25, 1.0, 1.0), 0.25, 0.01);
}

// ---- 3. SampleLinear and InvertLinearSample are inverses -------------------
TEST(BilinearSamplingTest, SampleLinearInvertRoundtrip) {
	double a = 0.5, b = 2.0;
	for (int i = 1; i < 20; ++i) {
		double u = i / 20.0;
		double x = SampleLinear(u, a, b);
		double u2 = InvertLinearSample(x, a, b);
		EXPECT_NEAR(u, u2, 1e-6) << "i=" << i;
	}
}

// ---- 4. BilinearPDF integrates to 1 (Monte Carlo) -------------------------
TEST(BilinearSamplingTest, BilinearPDFIntegratesTo1) {
	const int N = 200;
	double w0=1.0, w1=2.0, w2=0.5, w3=3.0;
	double sum = 0.0;
	double dx = 1.0 / N, dy = 1.0 / N;
	for (int i = 0; i < N; ++i)
		for (int j = 0; j < N; ++j) {
			double px = (i + 0.5) / N, py = (j + 0.5) / N;
			sum += BilinearPDF(px, py, w0, w1, w2, w3) * dx * dy;
		}
	EXPECT_NEAR(sum, 1.0, 0.02);
}

// ---- 5. BilinearPDF zero outside [0,1]^2 ----------------------------------
TEST(BilinearSamplingTest, BilinearPDFZeroOutside) {
	EXPECT_EQ(BilinearPDF(-0.1, 0.5, 1.0, 1.0, 1.0, 1.0), 0.0);
	EXPECT_EQ(BilinearPDF(1.1,  0.5, 1.0, 1.0, 1.0, 1.0), 0.0);
	EXPECT_EQ(BilinearPDF(0.5, -0.1, 1.0, 1.0, 1.0, 1.0), 0.0);
	EXPECT_EQ(BilinearPDF(0.5,  1.1, 1.0, 1.0, 1.0, 1.0), 0.0);
}

// ---- 6. BilinearPDF uniform weights -> constant 1 -------------------------
TEST(BilinearSamplingTest, BilinearPDFUniformWeightsConstant1) {
	for (int i = 1; i < 10; ++i)
		for (int j = 1; j < 10; ++j) {
			double px = i / 10.0, py = j / 10.0;
			EXPECT_NEAR(BilinearPDF(px, py, 1.0, 1.0, 1.0, 1.0), 1.0, 1e-10);
		}
}

// ---- 7. SampleBilinear output in [0,1)^2 ----------------------------------
TEST(BilinearSamplingTest, SampleBilinearRange) {
	double w0=1.0, w1=3.0, w2=0.5, w3=2.0;
	for (int i = 0; i < 10; ++i)
		for (int j = 0; j < 10; ++j) {
			double u0 = (i + 0.5) / 10.0, u1 = (j + 0.5) / 10.0;
			double px, py;
			SampleBilinear(u0, u1, w0, w1, w2, w3, px, py);
			EXPECT_GE(px, 0.0); EXPECT_LT(px, 1.0);
			EXPECT_GE(py, 0.0); EXPECT_LT(py, 1.0);
		}
}

// ---- 8. SampleBilinear/InvertBilinearSample are inverses ------------------
TEST(BilinearSamplingTest, SampleBilinearInvertRoundtrip) {
	double w0=0.5, w1=2.0, w2=1.5, w3=0.8;
	for (int i = 1; i < 8; ++i)
		for (int j = 1; j < 8; ++j) {
			double u0 = i / 8.0, u1 = j / 8.0;
			double px, py;
			SampleBilinear(u0, u1, w0, w1, w2, w3, px, py);
			double ru0, ru1;
			InvertBilinearSample(px, py, w0, w1, w2, w3, ru0, ru1);
			EXPECT_NEAR(u0, ru0, 1e-5) << "i=" << i << " j=" << j;
			EXPECT_NEAR(u1, ru1, 1e-5) << "i=" << i << " j=" << j;
		}
}

// ---- 9. SampleBilinear uniform weights -> approximate uniform distribution -
TEST(BilinearSamplingTest, SampleBilinearUniformIsUniform) {
	// With equal weights, samples should be uniformly distributed in [0,1)^2
	const int N = 10;
	// Check that x and y are close to the uniform sample values
	for (int i = 1; i < N; ++i)
		for (int j = 1; j < N; ++j) {
			double u0 = i / (double)N, u1 = j / (double)N;
			double px, py;
			SampleBilinear(u0, u1, 1.0, 1.0, 1.0, 1.0, px, py);
			EXPECT_NEAR(px, u0, 0.01) << "i=" << i << " j=" << j;
			EXPECT_NEAR(py, u1, 0.01) << "i=" << i << " j=" << j;
		}
}

// ---- 10. SampleUniformTriangle: barycentrics sum to 1, all >= 0 -----------
TEST(BilinearSamplingTest, SampleUniformTriangleBarycentricsValid) {
	for (int i = 0; i <= 10; ++i)
		for (int j = 0; j <= 10 - i; ++j) {
			double u0 = i / 10.0, u1 = j / 10.0;
			double b0, b1, b2;
			SampleUniformTriangle(u0, u1, b0, b1, b2);
			EXPECT_GE(b0, -1e-10) << "u0=" << u0 << " u1=" << u1;
			EXPECT_GE(b1, -1e-10) << "u0=" << u0 << " u1=" << u1;
			EXPECT_GE(b2, -1e-10) << "u0=" << u0 << " u1=" << u1;
			EXPECT_NEAR(b0 + b1 + b2, 1.0, 1e-10) << "u0=" << u0 << " u1=" << u1;
		}
}

// ---- 11. SampleUniformTriangle: covers triangle uniformly -----------------
TEST(BilinearSamplingTest, SampleUniformTriangleUniformCoverage) {
	// Collect N samples and check that b0, b1, b2 are each ~ uniformly in [0,1/2]
	// (each barycentric has marginal distribution linearly decreasing from 0 to 1)
	const int N = 400;
	double sum_b0 = 0, sum_b1 = 0, sum_b2 = 0;
	for (int i = 0; i < N; ++i)
		for (int j = 0; j < N; ++j) {
			double u0 = (i + 0.5) / N, u1 = (j + 0.5) / N;
			double b0, b1, b2;
			SampleUniformTriangle(u0, u1, b0, b1, b2);
			sum_b0 += b0; sum_b1 += b1; sum_b2 += b2;
		}
	int total = N * N;
	// E[b0] = E[b1] = E[b2] = 1/3 for uniform triangle sampling
	EXPECT_NEAR(sum_b0 / total, 1.0 / 3.0, 0.02);
	EXPECT_NEAR(sum_b1 / total, 1.0 / 3.0, 0.02);
	EXPECT_NEAR(sum_b2 / total, 1.0 / 3.0, 0.02);
}

// ---- 12. SampleUniformTriangle: corner cases ------------------------------
TEST(BilinearSamplingTest, SampleUniformTriangleCornerCases) {
	double b0, b1, b2;
	// (0,0) -> b0=0, b1=0, b2=1
	SampleUniformTriangle(0.0, 0.0, b0, b1, b2);
	EXPECT_NEAR(b2, 1.0, 1e-10);
	EXPECT_NEAR(b0 + b1 + b2, 1.0, 1e-10);
	// (1,0) -> b0=1, b1=0, b2=0
	SampleUniformTriangle(1.0, 0.0, b0, b1, b2);
	EXPECT_NEAR(b0, 1.0, 1e-10);
	EXPECT_NEAR(b0 + b1 + b2, 1.0, 1e-10);
}
