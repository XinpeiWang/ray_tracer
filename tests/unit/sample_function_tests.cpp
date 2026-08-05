// sample_function_tests.cpp
// Validation for Sample1DFunction and Sample2DFunction ported from pbrt-v4.
//
// Methodology:
//   For each function we verify the key contract: every cell value must be
//   >= the true maximum of |f| inside that cell (conservative upper bound).
//   We also check that the tabulation is non-negative and finite.
//
// Tests:
//   1.  Sample1D_SinFunction            -- sin(x) on [0, pi]: known analytic max per cell
//   2.  Sample1D_ConstantFunction        -- f(x)=C: all cells must equal C
//   3.  Sample1D_LinearFunction          -- f(x)=x on [0,1]: cell max = right endpoint
//   4.  Sample1D_AccuracyCheck           -- relative accuracy vs true cell max
//   5.  Sample1D_OutputNonNegative       -- no negative or NaN values in output
//   6.  Sample1D_StepFunction            -- f=0 for x<0.35, f=1 for x>=0.35
//   7.  Sample2D_ConstantFunction        -- f(x,y)=C: all cells = C
//   8.  Sample2D_ProductFunction         -- f(x,y)=sin(x)*sin(y): accuracy bound
//   9.  Sample2D_RowMajorLayout          -- verify values[v*nu+u] layout
//   10. Sample2D_OutputNonNegative       -- no negative or NaN values in 2D output
//   11. Sample1D_PbrtCanonical_LinearP1  -- pbrt-v4 test: f(x)=1+x on [-1,3], 65536 cells
//   12. Sample2D_PbrtCanonical_XsqY      -- pbrt-v4 test: f(x,y)=x^2*y on [0,1]^2, 4x2

#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <vector>
#include <limits>

#include "../../src/shared/sampling_patched.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Brute-force true max of |f| on [a, b] using many sub-samples.
static double true_max_1d(std::function<double(double)> f, double a, double b, int N = 4096) {
	double mx = 0.0;
	for (int i = 0; i <= N; ++i) {
		double x = a + (b - a) * (double(i) / N);
		mx = std::max(mx, std::abs(f(x)));
	}
	return mx;
}

// Brute-force true max of |f| on [x0,x1] x [y0,y1].
static double true_max_2d(std::function<double(double, double)> f,
						  double x0, double y0, double x1, double y1, int N = 128) {
	double mx = 0.0;
	for (int j = 0; j <= N; ++j)
		for (int i = 0; i <= N; ++i) {
			double x = x0 + (x1 - x0) * (double(i) / N);
			double y = y0 + (y1 - y0) * (double(j) / N);
			mx = std::max(mx, std::abs(f(x, y)));
		}
	return mx;
}

// ===========================================================================
// 1D Tests
// ===========================================================================

// Test 1: sin(x) on [0, pi] — known analytic shape, monotonically rising
// then falling. Each cell's tabulated value must >= true cell max.
TEST(SampleFunctionTest, Sample1D_SinFunction) {
	const int nSteps   = 32;
	const int nSamples = 32;
	const double xMin  = 0.0;
	const double xMax  = M_PI;
	auto f = [](double x) { return std::sin(x); };

	auto values = Sample1DFunction(f, nSteps, nSamples, xMin, xMax);
	ASSERT_EQ((int)values.size(), nSteps);

	for (int i = 0; i < nSteps; ++i) {
		double a    = xMin + (xMax - xMin) * (i     / double(nSteps));
		double b    = xMin + (xMax - xMin) * ((i+1) / double(nSteps));
		double tmax = true_max_1d(f, a, b);
		EXPECT_GE(values[i], tmax - 1e-10)
			<< "cell " << i << ": tabulated=" << values[i] << " true_max=" << tmax;
	}
}

// Test 2: constant function — every cell must equal C exactly.
TEST(SampleFunctionTest, Sample1D_ConstantFunction) {
	const double C = 3.14159;
	auto f = [C](double) { return C; };
	auto values = Sample1DFunction(f, 16, 16, -1.0, 1.0);
	for (int i = 0; i < 16; ++i)
		EXPECT_NEAR(values[i], C, 1e-12) << "cell " << i;
}

// Test 3: f(x) = x on [0, 1] — cell max = right endpoint of cell.
TEST(SampleFunctionTest, Sample1D_LinearFunction) {
	const int nSteps = 20;
	auto f = [](double x) { return x; };
	auto values = Sample1DFunction(f, nSteps, 64, 0.0, 1.0);
	ASSERT_EQ((int)values.size(), nSteps);
	for (int i = 0; i < nSteps; ++i) {
		double cell_right = (i + 1.0) / nSteps;
		EXPECT_GE(values[i], cell_right - 1e-10) << "cell " << i;
	}
}

// Test 4: accuracy test for Sample1DFunction.
// pbrt-v4's tabulation is a best-effort max approximation \u2014 NOT a strict
// conservative upper bound. The correct contract is that the tabulated value
// is within a small relative error of the true cell maximum.
// We verify that for sin(3x)*cos(2x) with 128 samples/cell the tabulated
// value is within 1% of the true cell max for non-trivial cells.
TEST(SampleFunctionTest, Sample1D_AccuracyCheck) {
	auto f = [](double x) {
		return std::sin(3.0 * x) * std::cos(2.0 * x);
	};
	const int nSteps   = 32;
	const int nSamples = 128;
	auto values = Sample1DFunction(f, nSteps, nSamples, 0.0, 2.0 * M_PI);
	for (int i = 0; i < nSteps; ++i) {
		double a    = 2.0 * M_PI * (i     / double(nSteps));
		double b    = 2.0 * M_PI * ((i+1) / double(nSteps));
		double tmax = true_max_1d(f, a, b, 8192);
		// Skip near-zero cells where relative error is meaningless.
		if (tmax < 1e-4) continue;
		double rel_err = std::abs(values[i] - tmax) / tmax;
		EXPECT_LT(rel_err, 0.01)
			<< "cell " << i << " rel_err=" << rel_err
			<< " tab=" << values[i] << " true=" << tmax;
		EXPECT_GE(values[i], 0.0) << "cell " << i << " is negative";
	}
}

// Test 5: all output values non-negative and finite.
TEST(SampleFunctionTest, Sample1D_OutputNonNegative) {
	auto f = [](double x) { return std::cos(x * 7.3) - 0.5; };
	auto values = Sample1DFunction(f, 40, 16, -5.0, 5.0);
	for (int i = 0; i < (int)values.size(); ++i) {
		EXPECT_GE(values[i], 0.0) << "cell " << i << " is negative";
		EXPECT_TRUE(std::isfinite(values[i])) << "cell " << i << " is not finite";
	}
}

// Test 6: step function — cells strictly below the step have max=0,
// cells strictly above have max=1. The step is at x=0.35; with 10 cells
// over [0,1] each cell is 0.1 wide: cells 0-2 are entirely in [0,0.3] (f=0),
// cells 4-9 are entirely in [0.4,1.0] (f=1). Cell 3 straddles the boundary
// and is intentionally skipped.
TEST(SampleFunctionTest, Sample1D_StepFunction) {
	const double step = 0.35;
	auto f = [step](double x) { return x >= step ? 1.0 : 0.0; };
	const int nSteps = 10;  // cells of width 0.1
	auto values = Sample1DFunction(f, nSteps, 32, 0.0, 1.0);
	// Cells 0..2: [0.0,0.1], [0.1,0.2], [0.2,0.3] -- entirely below step
	for (int i = 0; i < 3; ++i)
		EXPECT_NEAR(values[i], 0.0, 1e-12) << "left cell " << i;
	// Cells 4..9: [0.4,0.5]..[0.9,1.0] -- entirely above step
	for (int i = 4; i < nSteps; ++i)
		EXPECT_NEAR(values[i], 1.0, 1e-12) << "right cell " << i;
}

// ===========================================================================
// 2D Tests
// ===========================================================================

// Test 7: constant 2D function.
TEST(SampleFunctionTest, Sample2D_ConstantFunction) {
	const double C = 2.718;
	auto f = [C](double, double) { return C; };
	auto values = Sample2DFunction(f, 8, 8, 16, 0.0, 0.0, 1.0, 1.0);
	ASSERT_EQ((int)values.size(), 64);
	for (int i = 0; i < 64; ++i)
		EXPECT_NEAR(values[i], C, 1e-12) << "cell " << i;
}

// Test 8: f(x,y) = sin(x)*sin(y) on [0,pi]^2 — conservative bound per cell.
TEST(SampleFunctionTest, Sample2D_ProductFunction) {
	const int nu = 16, nv = 16;
	auto f = [](double x, double y) { return std::sin(x) * std::sin(y); };
	auto values = Sample2DFunction(f, nu, nv, 64, 0.0, 0.0, M_PI, M_PI);
	ASSERT_EQ((int)values.size(), nu * nv);
	for (int v = 0; v < nv; ++v) {
		for (int u = 0; u < nu; ++u) {
			double x0 = M_PI * (u     / double(nu));
			double x1 = M_PI * ((u+1) / double(nu));
			double y0 = M_PI * (v     / double(nv));
			double y1 = M_PI * ((v+1) / double(nv));
			double tmax = true_max_2d(f, x0, y0, x1, y1, 64);
			EXPECT_GE(values[v * nu + u] + 1e-9, tmax)
				<< "cell (" << u << "," << v << "): tab=" << values[v*nu+u]
				<< " true=" << tmax;
		}
	}
}

// Test 9: row-major layout — values[v*nu + u] addressed correctly.
TEST(SampleFunctionTest, Sample2D_RowMajorLayout) {
	// f(x,y) = x + 10*y so each cell has a unique value based on position.
	const int nu = 4, nv = 4;
	auto f = [](double x, double y) { return x + 10.0 * y; };
	auto values = Sample2DFunction(f, nu, nv, 32, 0.0, 0.0, 1.0, 1.0);
	// The max in cell (u,v) is at the top-right corner: x1 + 10*y1.
	for (int v = 0; v < nv; ++v) {
		for (int u = 0; u < nu; ++u) {
			double x1 = (u + 1.0) / nu;
			double y1 = (v + 1.0) / nv;
			double expected_max = x1 + 10.0 * y1;
			EXPECT_GE(values[v * nu + u], expected_max - 1e-9)
				<< "cell (" << u << "," << v << ") layout check";
		}
	}
}

// Test 10: all 2D output values non-negative and finite.
TEST(SampleFunctionTest, Sample2D_OutputNonNegative) {
	auto f = [](double x, double y) { return std::sin(x * 5.0) * std::cos(y * 3.7) - 0.2; };
	auto values = Sample2DFunction(f, 12, 12, 32, 0.0, 0.0, M_PI, M_PI);
	for (int i = 0; i < (int)values.size(); ++i) {
		EXPECT_GE(values[i], 0.0) << "cell " << i << " is negative";
		EXPECT_TRUE(std::isfinite(values[i])) << "cell " << i << " is not finite";
	}
}

// ===========================================================================
// Tests 11-12: pbrt-v4 canonical alignment
// These mirror the exact test cases from pbrt-v4/src/pbrt/util/sampling_test.cpp
// to ensure our port matches pbrt-v4's expected behaviour exactly.
// ===========================================================================

// Test 11: mirrors pbrt-v4 PiecewiseConstant1D.FromFuncLInfinity
// f(x) = 1 + x on [-1, 3]: linear, so each cell's max = right endpoint value.
// With 65536 cells and 4 sub-samples the right-endpoint is always sampled
// (delta=1.0 when j=nSamples), so each cell value equals f(cell_right) exactly.
TEST(SampleFunctionTest, Sample1D_PbrtCanonical_LinearP1) {
	auto f = [](double x) { return 1.0 + x; };
	const int nSteps   = 65536;
	const int nSamples = 4;
	const double xMin  = -1.0, xMax = 3.0;
	auto values = Sample1DFunction(f, nSteps, nSamples, xMin, xMax);
	ASSERT_EQ((int)values.size(), nSteps);
	for (int i = 0; i < nSteps; ++i) {
		// Right endpoint of cell i: f is monotone increasing, so max = f(right).
		double x_right = xMin + (xMax - xMin) * ((i + 1.0) / nSteps);
		double expected = 1.0 + x_right;
		EXPECT_NEAR(values[i], expected, 1e-9)
			<< "cell " << i << ": got " << values[i] << " expected " << expected;
	}
}

// Test 12: mirrors pbrt-v4 PiecewiseConstant2D.FromFuncLInfinity
// f(x,y) = x^2 * y on [0,1]^2, nu=4, nv=2, nSamples=1.
// With nSamples=1 the only interior sample is Halton(0,0)=0, Halton(1,0)=0,
// i.e. the cell origin. The mandatory corners add (0,1),(1,0),(1,1).
// For a monotone-increasing (in both x and y) function the max is always at
// the top-right corner (x1, y1) of each cell.
// Expected values match pbrt-v4's own test: Sqr(u_right) * v_top.
TEST(SampleFunctionTest, Sample2D_PbrtCanonical_XsqY) {
	auto f = [](double x, double y) { return x * x * y; };
	const int nu = 4, nv = 2;
	auto values = Sample2DFunction(f, nu, nv, 1, 0.0, 0.0, 1.0, 1.0);
	ASSERT_EQ((int)values.size(), nu * nv);
	// Exact expected values from pbrt-v4 sampling_test.cpp:
	// { (0.25)^2*0.5, (0.5)^2*0.5, (0.75)^2*0.5, 1^2*0.5,
	//   (0.25)^2*1,   (0.5)^2*1,   (0.75)^2*1,   1^2*1   }
	const double exact[8] = {
		0.25*0.25*0.5, 0.5*0.5*0.5, 0.75*0.75*0.5, 1.0*1.0*0.5,
		0.25*0.25*1.0, 0.5*0.5*1.0, 0.75*0.75*1.0, 1.0*1.0*1.0
	};
	for (int v = 0; v < nv; ++v)
		for (int u = 0; u < nu; ++u)
			EXPECT_NEAR(values[v * nu + u], exact[v * nu + u], 1e-9)
				<< "cell (" << u << "," << v << ")";
}
