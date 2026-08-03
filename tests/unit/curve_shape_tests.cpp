// curve_shape_tests.cpp
// Unit tests for CurveShape<T> in src/shared/shapes.h
//
// Mirrors the pbrt-v4 Curve primitive tests and validates:
//   - area() approximation is positive and reasonable
//   - Flat curve: ray hits centre, misses with offset, shadow-ray early-out
//   - Cylinder curve: hit normal perpendicular to dpdu
//   - Ribbon curve: ribbon orientation respected
//   - t_min / t_max clipping enforced
//   - Tapered width: wide-end hit, narrow-end miss

#include <gtest/gtest.h>
#include <cmath>
#include "../../src/shared/shapes.h"

using CS = CurveShape<double>;
using CH = CS::CurveHit;

// Straight horizontal cubic Bezier segment along X axis, y=z=0, in [0,1].
// Control points: (0,0,0), (1/3,0,0), (2/3,0,0), (1,0,0)
static CS make_flat_straight()
{
	double cx[4] = {0.0, 1.0/3.0, 2.0/3.0, 1.0};
	double cy[4] = {0.0, 0.0, 0.0, 0.0};
	double cz[4] = {0.0, 0.0, 0.0, 0.0};
	return CS::make(cx, cy, cz, 0.0, 1.0, 0.1, 0.1, CurveType::Flat);
}

// Straight horizontal cubic Bezier segment along X axis, tapered.
static CS make_flat_tapered()
{
	double cx[4] = {0.0, 1.0/3.0, 2.0/3.0, 1.0};
	double cy[4] = {0.0, 0.0, 0.0, 0.0};
	double cz[4] = {0.0, 0.0, 0.0, 0.0};
	return CS::make(cx, cy, cz, 0.0, 1.0, 0.2, 0.02, CurveType::Flat);
}

// Straight horizontal Cylinder curve.
static CS make_cylinder_straight()
{
	double cx[4] = {0.0, 1.0/3.0, 2.0/3.0, 1.0};
	double cy[4] = {0.0, 0.0, 0.0, 0.0};
	double cz[4] = {0.0, 0.0, 0.0, 0.0};
	return CS::make(cx, cy, cz, 0.0, 1.0, 0.1, 0.1, CurveType::Cylinder);
}

// ===========================================================================
// area()
// ===========================================================================

TEST(CurveShape, AreaPositive) {
	CS c = make_flat_straight();
	EXPECT_GT(c.area(), 0.0);
}

TEST(CurveShape, AreaApproximate) {
	// Straight curve of length 1.0 with constant width 0.1
	// area ≈ length * avgWidth = 1.0 * 0.1 = 0.1
	CS c = make_flat_straight();
	EXPECT_NEAR(c.area(), 0.1, 0.01);
}

TEST(CurveShape, AreaTapered) {
	// Straight tapered curve, avgWidth = (0.2+0.02)/2 = 0.11
	CS c = make_flat_tapered();
	EXPECT_NEAR(c.area(), 0.11, 0.015);
}

// ===========================================================================
// Flat curve: intersection
// ===========================================================================

TEST(CurveShape, FlatHitCentre) {
	// Ray shoots from above the midpoint of a horizontal curve
	// Ray origin: (0.5, 0.0, 1.0), direction: (0, 0, -1)
	CS c = make_flat_straight();
	CH hit;
	bool got = c.intersect(0.5, 0.0, 1.0,  0.0, 0.0, -1.0,  1e-4, 10.0, &hit);
	EXPECT_TRUE(got);
	EXPECT_GT(hit.t, 0.0);
	EXPECT_LT(hit.t, 10.0);
	// u parameter should be near the midpoint of the curve
	EXPECT_NEAR(hit.u, 0.5, 0.15);
}

TEST(CurveShape, FlatMissOffset) {
	// Ray shoots from far to the side (y-offset >> half-width=0.05)
	CS c = make_flat_straight();
	bool got = c.intersect(0.5, 0.5, 1.0,  0.0, 0.0, -1.0,  1e-4, 10.0);
	EXPECT_FALSE(got);
}

TEST(CurveShape, FlatMissBehindRay) {
	// Curve is at z=0, ray origin at z=-2 pointing away (-z direction)
	CS c = make_flat_straight();
	bool got = c.intersect(0.5, 0.0, -2.0,  0.0, 0.0, -1.0,  1e-4, 10.0);
	EXPECT_FALSE(got);
}

TEST(CurveShape, FlatMissTMaxClip) {
	// t_max = 0.5, but the curve is at z=0 from origin at z=1 => t~1.0
	CS c = make_flat_straight();
	bool got = c.intersect(0.5, 0.0, 1.0,  0.0, 0.0, -1.0,  1e-4, 0.5);
	EXPECT_FALSE(got);
}

TEST(CurveShape, FlatShadowRayNoHitInfo) {
	// Passing nullptr for out should still return true on hit (shadow test)
	CS c = make_flat_straight();
	bool got = c.intersect(0.5, 0.0, 1.0,  0.0, 0.0, -1.0,  1e-4, 10.0, nullptr);
	EXPECT_TRUE(got);
}

TEST(CurveShape, FlatHitUVRange) {
	// u and v must be in [uMin, uMax] and [0, 1] respectively
	CS c = make_flat_straight();
	CH hit;
	bool got = c.intersect(0.5, 0.0, 1.0,  0.0, 0.0, -1.0,  1e-4, 10.0, &hit);
	ASSERT_TRUE(got);
	EXPECT_GE(hit.u, 0.0);
	EXPECT_LE(hit.u, 1.0);
	EXPECT_GE(hit.v, 0.0);
	EXPECT_LE(hit.v, 1.0);
}

// ===========================================================================
// Tapered curve: width test
// ===========================================================================

TEST(CurveShape, TaperedWideEndHit) {
	// Ray at x=0.0 (wide end, half-width=0.1), offset y=0.08 -- should hit
	CS c = make_flat_tapered();
	CH hit;
	bool got = c.intersect(0.05, 0.08, 1.0,  0.0, 0.0, -1.0,  1e-4, 10.0, &hit);
	EXPECT_TRUE(got);
}

TEST(CurveShape, TaperedNarrowEndMiss) {
	// Ray at x=1.0 (narrow end, half-width=0.01), offset y=0.05 -- should miss
	CS c = make_flat_tapered();
	bool got = c.intersect(0.95, 0.05, 1.0,  0.0, 0.0, -1.0,  1e-4, 10.0);
	EXPECT_FALSE(got);
}

// ===========================================================================
// Cylinder curve: normal roughly perpendicular to dpdu
// ===========================================================================

TEST(CurveShape, CylinderNormalPerpendicularToDpdu) {
	CS c = make_cylinder_straight();
	CH hit;
	bool got = c.intersect(0.5, 0.0, 1.0,  0.0, 0.0, -1.0,  1e-4, 10.0, &hit);
	ASSERT_TRUE(got);
	// dpdu should point roughly along X, normal roughly along Y or Z
	double dot = hit.dpdu_x * hit.nx + hit.dpdu_y * hit.ny + hit.dpdu_z * hit.nz;
	EXPECT_NEAR(dot, 0.0, 0.1);
}

// ===========================================================================
// Ribbon curve
// ===========================================================================

TEST(CurveShape, RibbonHit) {
	double cx[4] = {0.0, 1.0/3.0, 2.0/3.0, 1.0};
	double cy[4] = {0.0, 0.0, 0.0, 0.0};
	double cz[4] = {0.0, 0.0, 0.0, 0.0};
	// Normals pointing -Z: ribbon faces the incoming -Z ray, maximising projected width.
	// pbrt-v4: hitWidth *= AbsDot(nHit, ray.d) / rayLength; with (0,0,-1)·(0,0,-1)=1.
	CS c = CS::make_ribbon(cx, cy, cz, 0.0, 1.0, 0.1, 0.1,
						   0.0, 0.0, -1.0,   // n0
						   0.0, 0.0, -1.0);  // n1
	CH hit;
	bool got = c.intersect(0.5, 0.0, 1.0,  0.0, 0.0, -1.0,  1e-4, 10.0, &hit);
	EXPECT_TRUE(got);
}

// ===========================================================================
// Sub-interval curve (uMin != 0 or uMax != 1)
// ===========================================================================

TEST(CurveShape, SubIntervalHitNearEnd) {
	// Curve covers only [0.4, 0.6]; ray targets u=0.5 of original strand
	double cx[4] = {0.0, 1.0/3.0, 2.0/3.0, 1.0};
	double cy[4] = {0.0, 0.0, 0.0, 0.0};
	double cz[4] = {0.0, 0.0, 0.0, 0.0};
	CS c = CS::make(cx, cy, cz, 0.4, 0.6, 0.1, 0.1, CurveType::Flat);
	CH hit;
	bool got = c.intersect(0.5, 0.0, 1.0,  0.0, 0.0, -1.0,  1e-4, 10.0, &hit);
	EXPECT_TRUE(got);
	EXPECT_NEAR(hit.u, 0.5, 0.1);
}

TEST(CurveShape, SubIntervalMissOutsideRange) {
	// Curve covers [0.7, 1.0]; ray targets u=0.5 (outside sub-interval's world range)
	double cx[4] = {0.0, 1.0/3.0, 2.0/3.0, 1.0};
	double cy[4] = {0.0, 0.0, 0.0, 0.0};
	double cz[4] = {0.0, 0.0, 0.0, 0.0};
	CS c = CS::make(cx, cy, cz, 0.7, 1.0, 0.1, 0.1, CurveType::Flat);
	// Sub-interval spans x in ~[0.7, 1.0]; ray at x=0.3 should miss
	bool got = c.intersect(0.3, 0.0, 1.0,  0.0, 0.0, -1.0,  1e-4, 10.0);
	EXPECT_FALSE(got);
}
