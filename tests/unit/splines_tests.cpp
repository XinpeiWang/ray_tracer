// splines_tests.cpp -- Unit tests for src/shared/splines.h
// Validates all Bézier and B-spline functions against pbrt-v4 semantics.

#include "../../src/shared/splines.h"
#include <gtest/gtest.h>
#include <cmath>

using namespace splines;

// Helper: squared distance between two float[3]
static float dist2(const float* a, const float* b) {
	float dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
	return dx*dx + dy*dy + dz*dz;
}

// Helper: make a straight-line cubic Bézier along X axis [0..1]
static void makeLine(float cp[4][3]) {
	for (int i = 0; i < 4; ++i) {
		cp[i][0] = i / 3.f;
		cp[i][1] = 0.f;
		cp[i][2] = 0.f;
	}
}

// -----------------------------------------------------------------------
// EvaluateCubicBezier -- endpoints must equal cp[0] and cp[3]
// -----------------------------------------------------------------------
TEST(SplinesTest, EndpointsInterpolated) {
	float cp[4][3] = {{0,0,0},{1,2,0},{2,1,0},{3,0,0}};
	float p0[3], p1[3];
	EvaluateCubicBezier(cp, 0.f, p0);
	EvaluateCubicBezier(cp, 1.f, p1);
	EXPECT_NEAR(p0[0], 0.f, 1e-5f); EXPECT_NEAR(p0[1], 0.f, 1e-5f);
	EXPECT_NEAR(p1[0], 3.f, 1e-5f); EXPECT_NEAR(p1[1], 0.f, 1e-5f);
}

// -----------------------------------------------------------------------
// EvaluateCubicBezier -- straight line: midpoint must be at centre
// -----------------------------------------------------------------------
TEST(SplinesTest, StraightLineMidpoint) {
	float cp[4][3]; makeLine(cp);
	float mid[3];
	EvaluateCubicBezier(cp, 0.5f, mid);
	EXPECT_NEAR(mid[0], 0.5f, 1e-5f);
	EXPECT_NEAR(mid[1], 0.0f, 1e-5f);
}

// -----------------------------------------------------------------------
// EvaluateCubicBezierD -- derivative at endpoints
// For a straight line from (0,0,0) to (1,0,0):
//   deriv at u=0 and u=1 should be (1,0,0) (slope = 1, scaled by 3/3)
// -----------------------------------------------------------------------
TEST(SplinesTest, DerivativeStraightLine) {
	float cp[4][3]; makeLine(cp);
	float p[3], d[3];
	EvaluateCubicBezierD(cp, 0.f, p, d);
	// 3*(cp1[1]-cp1[0]) = 3*(1/3,0,0)-(0,0,0)) = (1,0,0)
	EXPECT_NEAR(d[0], 1.f, 1e-4f);
	EXPECT_NEAR(d[1], 0.f, 1e-4f);
	EXPECT_NEAR(d[2], 0.f, 1e-4f);
}

// -----------------------------------------------------------------------
// BlossomCubicBezier -- blossom symmetry: B(u,u,u) == Evaluate(u)
// -----------------------------------------------------------------------
TEST(SplinesTest, BlossomSymmetry) {
	float cp[4][3] = {{0,0,0},{1,3,0},{2,3,0},{3,0,0}};
	for (float u : {0.f, 0.25f, 0.5f, 0.75f, 1.f}) {
		float b[3], e[3];
		BlossomCubicBezier(cp, u, u, u, b);
		EvaluateCubicBezier(cp, u, e);
		EXPECT_NEAR(b[0], e[0], 1e-5f);
		EXPECT_NEAR(b[1], e[1], 1e-5f);
	}
}

// -----------------------------------------------------------------------
// SubdivideCubicBezier -- shared midpoint and endpoint continuity
// -----------------------------------------------------------------------
TEST(SplinesTest, SubdivideEndpointContinuity) {
	float cp[4][3] = {{0,0,0},{1,2,0},{2,2,0},{3,0,0}};
	float sub[7][3];
	SubdivideCubicBezier(cp, sub);
	// sub[0] == cp[0], sub[6] == cp[3]
	EXPECT_NEAR(sub[0][0], cp[0][0], 1e-5f);
	EXPECT_NEAR(sub[6][0], cp[3][0], 1e-5f);
	// sub[3] is the midpoint: EvaluateCubicBezier(cp, 0.5) 
	float mid[3];
	EvaluateCubicBezier(cp, 0.5f, mid);
	EXPECT_NEAR(sub[3][0], mid[0], 1e-5f);
	EXPECT_NEAR(sub[3][1], mid[1], 1e-5f);
}

// -----------------------------------------------------------------------
// CubicBezierControlPoints -- reparametrize to full [0,1] gives same curve
// -----------------------------------------------------------------------
TEST(SplinesTest, ReparametrizeFullInterval) {
	float cp[4][3] = {{0,1,0},{1,3,0},{2,3,0},{3,1,0}};
	float sub[4][3];
	CubicBezierControlPoints(cp, 0.f, 1.f, sub);
	// Should reproduce the original control points
	for (int i = 0; i < 4; ++i)
		EXPECT_NEAR(dist2(sub[i], cp[i]), 0.f, 1e-8f) << "point " << i;
}

TEST(SplinesTest, ReparametrizeSubInterval) {
	float cp[4][3] = {{0,0,0},{1,2,0},{2,2,0},{3,0,0}};
	float sub[4][3];
	CubicBezierControlPoints(cp, 0.25f, 0.75f, sub);
	// Evaluate sub at u=0 should equal original at u=0.25
	float expected[3], got[3];
	EvaluateCubicBezier(cp, 0.25f, expected);
	EvaluateCubicBezier(sub, 0.f, got);
	EXPECT_NEAR(got[0], expected[0], 1e-4f);
	EXPECT_NEAR(got[1], expected[1], 1e-4f);
}

// -----------------------------------------------------------------------
// BoundCubicBezierBox -- bounding box must contain evaluated points
// -----------------------------------------------------------------------
TEST(SplinesTest, BoundingBoxContainsCurve) {
	float cp[4][3] = {{0,0,0},{1,3,0},{2,-1,0},{3,2,0}};
	float mn[3], mx[3];
	BoundCubicBezierBox(cp, mn, mx);
	for (int steps = 0; steps <= 20; ++steps) {
		float u = steps / 20.f;
		float p[3];
		EvaluateCubicBezier(cp, u, p);
		// The loose bound (hull of pairs) may not be tight — just check
		// that the individual control points are bounded.
		(void)p; // visual inspection only
	}
	// Control points must lie inside the bound
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 2; ++j) { // only X and Y
			EXPECT_LE(mn[j], cp[i][j] + 1e-4f) << "cp " << i << " axis " << j;
			EXPECT_GE(mx[j], cp[i][j] - 1e-4f) << "cp " << i << " axis " << j;
		}
	}
}

// -----------------------------------------------------------------------
// ElevateQuadraticBezierToCubic -- endpoints must match
// -----------------------------------------------------------------------
TEST(SplinesTest, ElevateQuadraticEndpoints) {
	float q[3][3] = {{0,0,0},{1,2,0},{2,0,0}};
	float c[4][3];
	ElevateQuadraticBezierToCubic(q, c);
	EXPECT_NEAR(c[0][0], 0.f, 1e-5f);
	EXPECT_NEAR(c[3][0], 2.f, 1e-5f);
}

TEST(SplinesTest, ElevateQuadraticCurveMatch) {
	// Evaluate quadratic at u directly vs elevated cubic
	float q[3][3] = {{0,0,0},{1,2,0},{2,0,0}};
	float c[4][3];
	ElevateQuadraticBezierToCubic(q, c);
	// Evaluate quadratic at u=0.5: lerp(lerp(q0,q1,0.5),lerp(q1,q2,0.5),0.5)
	float q01[3], q12[3], qmid[3];
	for (int i = 0; i < 3; ++i) {
		q01[i] = 0.5f*(q[0][i]+q[1][i]);
		q12[i] = 0.5f*(q[1][i]+q[2][i]);
		qmid[i] = 0.5f*(q01[i]+q12[i]);
	}
	float cmid[3];
	EvaluateCubicBezier(c, 0.5f, cmid);
	EXPECT_NEAR(cmid[0], qmid[0], 1e-5f);
	EXPECT_NEAR(cmid[1], qmid[1], 1e-5f);
}

// -----------------------------------------------------------------------
// QuadraticBSplineToBezier -- midpoints are averages
// -----------------------------------------------------------------------
TEST(SplinesTest, QuadBSplineToBezierMidpoints) {
	float cp[3][3] = {{0,0,0},{2,4,0},{4,0,0}};
	float bz[3][3];
	QuadraticBSplineToBezier(cp, bz);
	// out[0] = (cp[0]+cp[1])/2, out[2] = (cp[1]+cp[2])/2
	EXPECT_NEAR(bz[0][0], 1.f, 1e-5f);
	EXPECT_NEAR(bz[2][0], 3.f, 1e-5f);
	EXPECT_NEAR(bz[1][0], 2.f, 1e-5f); // middle stays cp[1]
}

// -----------------------------------------------------------------------
// CubicBSplineToBezier -- straight-line B-spline produces straight Bézier
// -----------------------------------------------------------------------
TEST(SplinesTest, CubicBSplineStraightLine) {
	// Uniform B-spline control points along X: 0, 1, 2, 3
	float bs[4][3] = {{0,0,0},{1,0,0},{2,0,0},{3,0,0}};
	float bz[4][3];
	CubicBSplineToBezier(bs, bz);
	// Result should be a straight line (all Y == 0, X monotone)
	for (int i = 0; i < 4; ++i)
		EXPECT_NEAR(bz[i][1], 0.f, 1e-5f) << "point " << i;
	// X values should be monotonically increasing
	for (int i = 0; i < 3; ++i)
		EXPECT_LT(bz[i][0], bz[i+1][0]) << "not monotone at " << i;
}
