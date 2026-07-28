// catmull_rom_tests.cpp
// Validation for SampleCatmullRom and SampleCatmullRom2D -- pbrt-v4 ports
//
// Tests:
//   1.  integrate_catmull_rom: constant function integrates to width
//   2.  integrate_catmull_rom: linear function integrates exactly
//   3.  SampleCatmullRom: output in [nodes[0], nodes[n-1]]
//   4.  SampleCatmullRom: constant PDF -> uniform sampling (u=0->xmin, u=1->xmax)
//   5.  SampleCatmullRom: returned pdf integrates to 1 (Monte Carlo)
//   6.  SampleCatmullRom: monotone: larger u -> larger output
//   7.  catmull_rom_weights: partition of unity
//   8.  catmull_rom_weights: out-of-range returns false
//   9.  SampleCatmullRom2D: output in nodes2 range
//  10.  SampleCatmullRom2D: uniform 2D table -> uniform sampling
//  11.  evaluate_polynomial: known polynomial values
//  12.  catmullrom_find_interval: boundary and middle cases

#include <gtest/gtest.h>
#include "../../src/shared/sampling.h"
#include <vector>
#include <cmath>
#include <numeric>

// ---- 1. integrate_catmull_rom: constant function --------------------------
TEST(CatmullRomTest, IntegrateConstantFunction) {
	const int N = 5;
	double nodes[N] = {0.0, 1.0, 2.0, 3.0, 4.0};
	double f[N]     = {2.0, 2.0, 2.0, 2.0, 2.0};
	double cdf[N];
	double total = integrate_catmull_rom(nodes, f, N, cdf);
	EXPECT_NEAR(total, 8.0, 1e-10);   // integral of 2 over [0,4]
	EXPECT_NEAR(cdf[0], 0.0,  1e-10);
	EXPECT_NEAR(cdf[2], 4.0,  1e-10); // integral up to x=2
	EXPECT_NEAR(cdf[N-1], 8.0, 1e-10);
}

// ---- 2. integrate_catmull_rom: linear function ----------------------------
TEST(CatmullRomTest, IntegrateLinearFunction) {
	// f(x) = x on [0,1,2,3]: integral = x^2/2 -> 4.5
	const int N = 4;
	double nodes[N] = {0.0, 1.0, 2.0, 3.0};
	double f[N]     = {0.0, 1.0, 2.0, 3.0};
	double cdf[N];
	double total = integrate_catmull_rom(nodes, f, N, cdf);
	EXPECT_NEAR(total, 4.5, 0.01);
}

// ---- 3. SampleCatmullRom: output in valid range ---------------------------
TEST(CatmullRomTest, SampleCatmullRomOutputRange) {
	const int N = 5;
	double nodes[N] = {0.0, 1.0, 2.0, 3.0, 4.0};
	double f[N]     = {1.0, 2.0, 3.0, 2.0, 1.0};
	double cdf[N];
	integrate_catmull_rom(nodes, f, N, cdf);
	for (int i = 1; i < 20; ++i) {
		double u = i / 20.0;
		double x = SampleCatmullRom(nodes, f, cdf, N, u);
		EXPECT_GE(x, nodes[0])  << "u=" << u;
		EXPECT_LE(x, nodes[N-1]) << "u=" << u;
	}
}

// ---- 4. SampleCatmullRom: constant PDF -> endpoints -----------------------
TEST(CatmullRomTest, SampleCatmullRomConstantPDF) {
	const int N = 3;
	double nodes[N] = {1.0, 2.0, 3.0};
	double f[N]     = {1.0, 1.0, 1.0};
	double cdf[N];
	integrate_catmull_rom(nodes, f, N, cdf);
	// u=0 -> x near nodes[0]; u=1 -> x near nodes[N-1]
	double x0 = SampleCatmullRom(nodes, f, cdf, N, 0.0);
	double x1 = SampleCatmullRom(nodes, f, cdf, N, 1.0 - 1e-9);
	EXPECT_NEAR(x0, 1.0, 0.01);
	EXPECT_NEAR(x1, 3.0, 0.01);
}

// ---- 5. SampleCatmullRom: returned pdf integrates to ~1 (Monte Carlo) -----
TEST(CatmullRomTest, SampleCatmullRomPDFNormalized) {
	const int N = 5;
	double nodes[N] = {0.0, 0.5, 1.0, 1.5, 2.0};
	double f[N]     = {1.0, 2.0, 3.0, 2.0, 1.0};
	double cdf[N];
	integrate_catmull_rom(nodes, f, N, cdf);
	// Monte Carlo: sum of (1/pdf(x_i)) * (1/M) should be ~1 when x_i sampled from pdf
	double sum = 0;
	const int M = 1000;
	for (int i = 0; i < M; ++i) {
		double u  = (i + 0.5) / M;
		double pdf_val;
		SampleCatmullRom(nodes, f, cdf, N, u, nullptr, &pdf_val);
		if (pdf_val > 1e-10) sum += 1.0 / (pdf_val * M);
	}
	EXPECT_NEAR(sum, 2.0, 0.1); // range is [0,2]; integral of normalized pdf over [0,2] = 1 -> sum of (range/M) terms
	// Actually E[1/pdf] * (1/M) * M = range, so sum ~ range = 2
}

// ---- 6. SampleCatmullRom: monotone ----------------------------------------
TEST(CatmullRomTest, SampleCatmullRomMonotone) {
	const int N = 4;
	double nodes[N] = {0.0, 1.0, 2.0, 3.0};
	double f[N]     = {0.5, 1.5, 1.0, 2.0};
	double cdf[N];
	integrate_catmull_rom(nodes, f, N, cdf);
	double prev = -1e10;
	for (int i = 1; i < 20; ++i) {
		double u = i / 20.0;
		double x = SampleCatmullRom(nodes, f, cdf, N, u);
		EXPECT_GT(x, prev) << "u=" << u;
		prev = x;
	}
}

// ---- 7. catmull_rom_weights: partition of unity ---------------------------
TEST(CatmullRomTest, CatmullRomWeightsPartitionOfUnity) {
	const int N = 6;
	double nodes[N] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
	for (int k = 1; k < 10; ++k) {
		double x = k * 0.5;
		int offset; double w[4];
		bool ok = catmull_rom_weights(nodes, N, x, &offset, w);
		EXPECT_TRUE(ok) << "x=" << x;
		double sum = w[0] + w[1] + w[2] + w[3];
		EXPECT_NEAR(sum, 1.0, 1e-10) << "x=" << x;
	}
}

// ---- 8. catmull_rom_weights: out-of-range returns false -------------------
TEST(CatmullRomTest, CatmullRomWeightsOutOfRange) {
	const int N = 4;
	double nodes[N] = {1.0, 2.0, 3.0, 4.0};
	int offset; double w[4];
	EXPECT_FALSE(catmull_rom_weights(nodes, N, 0.5, &offset, w));
	EXPECT_FALSE(catmull_rom_weights(nodes, N, 4.5, &offset, w));
	EXPECT_TRUE(catmull_rom_weights(nodes, N, 2.5, &offset, w));
}

// ---- 9. SampleCatmullRom2D: output in nodes2 range -----------------------
TEST(CatmullRomTest, SampleCatmullRom2DOutputRange) {
	const int N1 = 3, N2 = 4;
	double nodes1[N1] = {0.0, 0.5, 1.0};
	double nodes2[N2] = {0.0, 1.0, 2.0, 3.0};
	// Uniform table
	double values[N1 * N2];
	double cdf[N1 * N2];
	for (int i = 0; i < N1; ++i) {
		for (int j = 0; j < N2; ++j) values[i*N2+j] = 1.0;
		integrate_catmull_rom(nodes2, values + i*N2, N2, cdf + i*N2);
	}
	for (int k = 1; k < 10; ++k) {
		double alpha = k / 10.0;
		double u     = k / 10.0;
		double x = SampleCatmullRom2D(nodes1, nodes2, values, cdf, N1, N2, alpha, u);
		EXPECT_GE(x, nodes2[0])    << "k=" << k;
		EXPECT_LE(x, nodes2[N2-1]) << "k=" << k;
	}
}

// ---- 10. SampleCatmullRom2D: uniform table -> uniform sampling ------------
TEST(CatmullRomTest, SampleCatmullRom2DUniformTableIsUniform) {
	const int N1 = 3, N2 = 5;
	double nodes1[N1] = {0.0, 0.5, 1.0};
	double nodes2[N2] = {0.0, 1.0, 2.0, 3.0, 4.0};
	double values[N1 * N2];
	double cdf[N1 * N2];
	for (int i = 0; i < N1; ++i) {
		for (int j = 0; j < N2; ++j) values[i*N2+j] = 1.0;
		integrate_catmull_rom(nodes2, values + i*N2, N2, cdf + i*N2);
	}
	// Uniform table: sample should be approximately proportional to u
	double alpha = 0.5;
	for (int k = 1; k < 10; ++k) {
		double u = k / 10.0;
		double x = SampleCatmullRom2D(nodes1, nodes2, values, cdf, N1, N2, alpha, u);
		double expected = u * 4.0; // nodes2 range is [0,4]
		EXPECT_NEAR(x, expected, 0.2) << "u=" << u;
	}
}

// ---- 11. evaluate_polynomial: known values --------------------------------
TEST(CatmullRomTest, EvaluatePolynomialKnownValues) {
	// p(t) = 1 + 2t + 3t^2 -> p(2) = 1+4+12 = 17
	EXPECT_NEAR(evaluate_polynomial(2.0, 1.0, 2.0, 3.0), 17.0, 1e-10);
	// p(t) = 5 (constant)
	EXPECT_NEAR(evaluate_polynomial(3.0, 5.0), 5.0, 1e-10);
	// p(t) = t^3: c0=0, c1=0, c2=0, c3=1 -> p(2) = 8
	EXPECT_NEAR(evaluate_polynomial(2.0, 0.0, 0.0, 0.0, 1.0), 8.0, 1e-10);
}

// ---- 12. catmullrom_find_interval: boundary and middle cases --------------
TEST(CatmullRomTest, FindIntervalBoundaryAndMiddle) {
	// For array F = {0, 1, 2, 3, 4}:
	// find_interval(5, F[i]<=x): x=0.5 -> 0, x=1.5 -> 1, x=2.0 -> 2, x=3.9 -> 3
	std::vector<double> F = {0.0, 1.0, 2.0, 3.0, 4.0};
	int n = static_cast<int>(F.size());
	EXPECT_EQ(catmullrom_find_interval(n, [&](int i){ return F[i] <= 0.5; }), 0);
	EXPECT_EQ(catmullrom_find_interval(n, [&](int i){ return F[i] <= 1.5; }), 1);
	EXPECT_EQ(catmullrom_find_interval(n, [&](int i){ return F[i] <= 2.0; }), 2);
	EXPECT_EQ(catmullrom_find_interval(n, [&](int i){ return F[i] <= 3.9; }), 3);
}
