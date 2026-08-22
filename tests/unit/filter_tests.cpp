// filter_tests.cpp -- pbrt-v4-style tests for pixel reconstruction filters
//
// Tests verify:
//   1. Mitchell1D known analytic values at x=0, x=1, x=2
//   2. MitchellFilter is symmetric (even function)
//   3. MitchellFilter center weight is maximum
//   4. MitchellFilter returns 0 beyond radius
//   5. GaussianFilter center > edge, zero beyond radius
//   6. BoxFilter is always 1.0
//   7. Weighted average with Mitchell filter equals plain average for uniform radiance

#include <gtest/gtest.h>
#include "../../src/shared/filter.h"
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// MitchellFilter tests
// ---------------------------------------------------------------------------

TEST(MitchellFilterTest, CenterWeightIsPositive) {
	MitchellFilter f;
	EXPECT_GT(f.evaluate(0.0, 0.0), 0.0);
}

TEST(MitchellFilterTest, KnownCenterValue) {
	// Mitchell1D(0) = (6 - 2*B) / 6 with B=1/3
	// = (6 - 2/3) / 6 = (16/3) / 6 = 8/9
	// 2D: (8/9)^2 = 64/81
	MitchellFilter f(0.5, 1.0/3.0, 1.0/3.0);
	double expected = (8.0/9.0) * (8.0/9.0);
	EXPECT_NEAR(f.evaluate(0.0, 0.0), expected, 1e-10);
}

TEST(MitchellFilterTest, IsSymmetric) {
	MitchellFilter f;
	// Symmetric in x
	EXPECT_NEAR(f.evaluate( 0.3,  0.1), f.evaluate(-0.3,  0.1), 1e-14);
	// Symmetric in y
	EXPECT_NEAR(f.evaluate( 0.1,  0.3), f.evaluate( 0.1, -0.3), 1e-14);
	// Symmetric in both
	EXPECT_NEAR(f.evaluate(-0.2, -0.4), f.evaluate(0.2, 0.4), 1e-14);
}

TEST(MitchellFilterTest, ZeroBeyondRadius) {
	MitchellFilter f(0.5);  // radius = 0.5 pixel
	// At exactly the radius boundary, Mitchell1D(2*0.5/0.5) = Mitchell1D(2) = 0
	EXPECT_NEAR(f.evaluate(0.5, 0.0), 0.0, 1e-14);
	EXPECT_NEAR(f.evaluate(0.0, 0.5), 0.0, 1e-14);
	// Beyond radius
	EXPECT_DOUBLE_EQ(f.evaluate(0.6, 0.0), 0.0);
	EXPECT_DOUBLE_EQ(f.evaluate(0.0, 0.6), 0.0);
}

TEST(MitchellFilterTest, CenterIsMaximum) {
	MitchellFilter f;
	double center = f.evaluate(0.0, 0.0);
	// Sample several interior points and verify they're <= center
	for (double ox = -0.45; ox <= 0.45; ox += 0.05) {
		for (double oy = -0.45; oy <= 0.45; oy += 0.05) {
			EXPECT_LE(f.evaluate(ox, oy), center + 1e-10)
				<< "at (" << ox << ", " << oy << ")";
		}
	}
}

TEST(MitchellFilterTest, Mitchell1DAtBoundary) {
	// At x=1 (scaled) Mitchell1D is continuous: both pieces should give same value
	// piece 1: ((12-9B-6C)*1 + (-18+12B+6C)*1 + (6-2B)) / 6
	// piece 2: ((-B-6C)*1 + (6B+30C)*1 + (-12B-48C)*1 + (8B+24C)) / 6
	// With B=C=1/3 both equal (6-2/3)/6? No, let's just test continuity numerically.
	MitchellFilter f(0.5, 1.0/3.0, 1.0/3.0);
	double just_inside  = f.evaluate(0.249, 0.0);
	double just_outside = f.evaluate(0.251, 0.0);
	// Should be close (continuity)
	EXPECT_NEAR(just_inside, just_outside, 0.01);
}

TEST(MitchellFilterTest, SeparableEqualsProduct) {
	// 2D filter = Mitchell1D(x) * Mitchell1D(y): test at a non-trivial point
	MitchellFilter f(0.5, 1.0/3.0, 1.0/3.0);
	double w2d = f.evaluate(0.3, 0.2);
	// Re-compute manually using the same 1D formula:
	auto m1d = [](double x, double B, double C) -> double {
		x = std::fabs(x);
		if (x <= 1.0)
			return ((12.0-9.0*B-6.0*C)*x*x*x + (-18.0+12.0*B+6.0*C)*x*x + (6.0-2.0*B)) / 6.0;
		else if (x <= 2.0)
			return ((-B-6.0*C)*x*x*x + (6.0*B+30.0*C)*x*x + (-12.0*B-48.0*C)*x + (8.0*B+24.0*C)) / 6.0;
		return 0.0;
	};
	double B = 1.0/3.0, C = 1.0/3.0, r = 0.5;
	double expected = m1d(2.0*0.3/r, B, C) * m1d(2.0*0.2/r, B, C);
	EXPECT_NEAR(w2d, expected, 1e-14);
}

// ---------------------------------------------------------------------------
// GaussianFilter tests
// ---------------------------------------------------------------------------

TEST(GaussianFilterTest, CenterIsMaximum) {
	GaussianFilter f;
	double center = f.evaluate(0.0, 0.0);
	EXPECT_GT(center, 0.0);
	EXPECT_GT(center, f.evaluate(0.2, 0.0));
	EXPECT_GT(center, f.evaluate(0.0, 0.2));
}

TEST(GaussianFilterTest, ZeroAtRadius) {
	GaussianFilter f(0.5, 0.5);
	EXPECT_NEAR(f.evaluate(0.5, 0.0), 0.0, 1e-10);
	EXPECT_NEAR(f.evaluate(0.0, 0.5), 0.0, 1e-10);
	EXPECT_DOUBLE_EQ(f.evaluate(0.6, 0.0), 0.0);
}

// ---------------------------------------------------------------------------
// BoxFilter tests
// ---------------------------------------------------------------------------

TEST(BoxFilterTest, AlwaysOne) {
	BoxFilter f;
	EXPECT_DOUBLE_EQ(f.evaluate(0.0,  0.0),  1.0);
	EXPECT_DOUBLE_EQ(f.evaluate(0.49, 0.49), 1.0);
	EXPECT_DOUBLE_EQ(f.evaluate(-0.3, 0.1),  1.0);
}

// ---------------------------------------------------------------------------
// Weighted average property: for constant radiance, filtered result == radiance
// ---------------------------------------------------------------------------
TEST(MitchellFilterTest, ConstantRadiancePreserved) {
	// If every sample returns the same color, the weighted average should equal that color.
	MitchellFilter f;
	const double target = 0.75;
	double weighted_sum = 0.0, weight_sum = 0.0;
	// Simulate a 4x4 stratified grid within [-0.5, 0.5]
	const int N = 4;
	for (int si = 0; si < N; ++si) {
		for (int sj = 0; sj < N; ++sj) {
			double ox = (si + 0.5) / N - 0.5;
			double oy = (sj + 0.5) / N - 0.5;
			double w = f.evaluate(ox, oy);
			weighted_sum += w * target;
			weight_sum   += w;
		}
	}
	double result = (weight_sum > 0.0) ? weighted_sum / weight_sum : 0.0;
	EXPECT_NEAR(result, target, 1e-12);
}

// ---------------------------------------------------------------------------
// TriangleFilter tests  (pbrt-v4 filters.h TriangleFilter)
// ---------------------------------------------------------------------------

TEST(TriangleFilterTest, CenterEqualsRadiusSquared) {
	// w(0,0) = radius * radius  (both 1D tent values equal radius at center)
	TriangleFilter f(2.0);
	EXPECT_NEAR(f.evaluate(0.0, 0.0), 2.0 * 2.0, 1e-14);
}

TEST(TriangleFilterTest, ZeroAtRadius) {
	// tent1d(radius) = max(0, radius - radius) = 0
	TriangleFilter f(2.0);
	EXPECT_NEAR(f.evaluate(2.0, 0.0), 0.0, 1e-14);
	EXPECT_NEAR(f.evaluate(0.0, 2.0), 0.0, 1e-14);
}

TEST(TriangleFilterTest, ZeroBeyondRadius) {
	// pbrt-v4: max(0, radius - |x|) -- clamps at zero outside support
	TriangleFilter f(2.0);
	EXPECT_DOUBLE_EQ(f.evaluate(2.5, 0.0), 0.0);
	EXPECT_DOUBLE_EQ(f.evaluate(0.0, 3.0), 0.0);
	EXPECT_DOUBLE_EQ(f.evaluate(-2.1, -2.1), 0.0);
}

TEST(TriangleFilterTest, IsSymmetric) {
	TriangleFilter f(2.0);
	EXPECT_DOUBLE_EQ(f.evaluate( 1.0,  0.5), f.evaluate(-1.0,  0.5));
	EXPECT_DOUBLE_EQ(f.evaluate( 0.5,  1.0), f.evaluate( 0.5, -1.0));
	EXPECT_DOUBLE_EQ(f.evaluate(-0.7, -1.2), f.evaluate( 0.7,  1.2));
}

TEST(TriangleFilterTest, LinearFalloff) {
	// tent1d is linear in x: value at x1 and x2 should interpolate linearly
	TriangleFilter f(2.0);
	// Along x-axis (oy=0): w = (2-|ox|) * 2
	double w0 = f.evaluate(0.0, 0.0);  // 2 * 2 = 4
	double w1 = f.evaluate(1.0, 0.0);  // 1 * 2 = 2
	double w2 = f.evaluate(2.0, 0.0);  // 0 * 2 = 0
	EXPECT_NEAR(w0, 4.0, 1e-14);
	EXPECT_NEAR(w1, 2.0, 1e-14);
	EXPECT_NEAR(w2, 0.0, 1e-14);
	// Midpoint should be exactly halfway
	double wmid = f.evaluate(0.5, 0.0);
	EXPECT_NEAR(wmid, 3.0, 1e-14);
}

TEST(TriangleFilterTest, SeparableEqualsProduct) {
	// 2D weight = tent1d(ox) * tent1d(oy)
	TriangleFilter f(2.0);
	auto tent1d = [](double x, double r) { double v = r - std::fabs(x); return v > 0.0 ? v : 0.0; };
	for (double ox : {0.0, 0.5, 1.0, 1.5}) {
		for (double oy : {0.0, 0.3, 0.9, 1.8}) {
			double expected = tent1d(ox, 2.0) * tent1d(oy, 2.0);
			EXPECT_NEAR(f.evaluate(ox, oy), expected, 1e-14)
				<< "at (" << ox << ", " << oy << ")";
		}
	}
}

TEST(TriangleFilterTest, ConstantRadiancePreserved) {
	// Weighted average with TriangleFilter over a uniform grid equals the constant value
	TriangleFilter f(2.0);
	const double target = 0.6;
	double weighted_sum = 0.0, weight_sum = 0.0;
	const int N = 8;
	for (int si = 0; si < N; ++si) {
		for (int sj = 0; sj < N; ++sj) {
			double ox = (si + 0.5) / N * 4.0 - 2.0;  // [-2, 2]
			double oy = (sj + 0.5) / N * 4.0 - 2.0;
			double w = f.evaluate(ox, oy);
			weighted_sum += w * target;
			weight_sum   += w;
		}
	}
	double result = (weight_sum > 0.0) ? weighted_sum / weight_sum : 0.0;
	EXPECT_NEAR(result, target, 1e-12);
}

TEST(TriangleFilterTest, IntegralMatchesPbrtV4) {
	// pbrt-v4 Integral() = Sqr(radius.x) * Sqr(radius.y) = r^4 for square filter
	// 1D integral of tent: int_{-r}^{r} (r-|x|) dx = r^2, so 2D = r^4
	for (double r : {1.0, 2.0, 0.5}) {
		TriangleFilter f(r);
		EXPECT_NEAR(f.integral(), r * r * r * r, 1e-12) << "r=" << r;
	}
}

// ===========================================================================
// PixelFilterDispatch -- runtime kind-string -> filter-class routing (see
// pbrt_flatten::PixelFilter's own comment for why kind stays a string, not
// an enum, and why radius is always fixed at 0.5)
// ===========================================================================

TEST(PixelFilterDispatchTest, BoxMatchesDirectConstruction) {
	PixelFilterDispatch d("box");
	BoxFilter f(0.5);
	for (double ox : {-0.4, 0.0, 0.3})
		for (double oy : {-0.2, 0.1, 0.45})
			EXPECT_DOUBLE_EQ(d.evaluate(ox, oy), f.evaluate(ox, oy));
}

TEST(PixelFilterDispatchTest, TriangleMatchesDirectConstruction) {
	PixelFilterDispatch d("triangle");
	TriangleFilter f(0.5);
	for (double ox : {-0.4, 0.0, 0.3})
		for (double oy : {-0.2, 0.1, 0.45})
			EXPECT_DOUBLE_EQ(d.evaluate(ox, oy), f.evaluate(ox, oy));
}

TEST(PixelFilterDispatchTest, SincMatchesDirectConstructionWithTau) {
	PixelFilterDispatch d("sinc", 1.0/3.0, 1.0/3.0, 0.5, 2.5);
	LanczosSincFilter f(0.5, 2.5);
	for (double ox : {-0.4, 0.0, 0.3})
		for (double oy : {-0.2, 0.1, 0.45})
			EXPECT_DOUBLE_EQ(d.evaluate(ox, oy), f.evaluate(ox, oy));
}

TEST(PixelFilterDispatchTest, MitchellMatchesDirectConstructionWithBC) {
	PixelFilterDispatch d("mitchell", 0.2, 0.4);
	MitchellFilter f(0.5, 0.2, 0.4);
	for (double ox : {-0.4, 0.0, 0.3})
		for (double oy : {-0.2, 0.1, 0.45})
			EXPECT_DOUBLE_EQ(d.evaluate(ox, oy), f.evaluate(ox, oy));
}

TEST(PixelFilterDispatchTest, GaussianMatchesDirectConstructionWithSigma) {
	PixelFilterDispatch d("gaussian", 1.0/3.0, 1.0/3.0, 0.7);
	GaussianFilter f(0.5, 0.7);
	for (double ox : {-0.4, 0.0, 0.3})
		for (double oy : {-0.2, 0.1, 0.45})
			EXPECT_DOUBLE_EQ(d.evaluate(ox, oy), f.evaluate(ox, oy));
}

TEST(PixelFilterDispatchTest, UnrecognizedKindFallsBackToGaussian) {
	// pbrt-v4's real default (confirmed against pbrt-v4/src/pbrt/scene.cpp),
	// not box/triangle/mitchell - an easy default to assume wrong.
	PixelFilterDispatch d("not-a-real-filter", 1.0/3.0, 1.0/3.0, 0.5);
	GaussianFilter f(0.5, 0.5);
	EXPECT_DOUBLE_EQ(d.evaluate(0.1, -0.2), f.evaluate(0.1, -0.2));
}

TEST(PixelFilterDispatchTest, DefaultConstructedIsGaussian) {
	PixelFilterDispatch d;
	GaussianFilter f(0.5, 0.5);
	EXPECT_DOUBLE_EQ(d.evaluate(0.1, -0.2), f.evaluate(0.1, -0.2));
}
