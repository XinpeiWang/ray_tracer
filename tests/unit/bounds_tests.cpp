// ---------------------------------------------------------------------------
// bounds_tests.cpp
// Unit tests for Bounds3<T> from src/shared/grid_medium.h
//
// Mirrors pbrt-v4 Bounds3f tests (src/pbrt/util/vecmath_test.cpp):
//   - DefaultEmpty: default-constructed box has inverted extents
//   - Union: merging two boxes gives tight enclosing box
//   - PointDistance: distance from point to box surface
//   - RayHit/RayMiss: intersect_ray correctly identifies hits and misses
//   - Diagonal: axis extents are correct
//   - Offset: normalized fractional position inside box
// ---------------------------------------------------------------------------
#include <gtest/gtest.h>
#include "../../src/shared/grid_medium.h"

#include <cmath>
#include <algorithm>

// Merge two Bounds3 instances (Union, pbrt-v4 style).
static Bounds3<float> bounds_union(const Bounds3<float>& a, const Bounds3<float>& b) {
	Bounds3<float> r;
	for (int i = 0; i < 3; ++i) {
		r.pMin[i] = std::min(a.pMin[i], b.pMin[i]);
		r.pMax[i] = std::max(a.pMax[i], b.pMax[i]);
	}
	return r;
}

// Distance from a point to the surface of a Bounds3 (0 if inside).
static float point_distance(const Bounds3<float>& b, const float p[3]) {
	float dx = std::max(0.0f, std::max(b.pMin[0] - p[0], p[0] - b.pMax[0]));
	float dy = std::max(0.0f, std::max(b.pMin[1] - p[1], p[1] - b.pMax[1]));
	float dz = std::max(0.0f, std::max(b.pMin[2] - p[2], p[2] - b.pMax[2]));
	return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(Bounds3, DefaultHasInvertedExtents) {
	Bounds3<float> b;
	EXPECT_GT(b.pMin[0], b.pMax[0]) << "Default box should have inverted x extents (empty)";
	EXPECT_GT(b.pMin[1], b.pMax[1]) << "Default box should have inverted y extents (empty)";
	EXPECT_GT(b.pMin[2], b.pMax[2]) << "Default box should have inverted z extents (empty)";
}

TEST(Bounds3, ExplicitConstruction) {
	Bounds3<float> b(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
	EXPECT_FLOAT_EQ(b.pMin[0], 1.0f);
	EXPECT_FLOAT_EQ(b.pMin[1], 2.0f);
	EXPECT_FLOAT_EQ(b.pMin[2], 3.0f);
	EXPECT_FLOAT_EQ(b.pMax[0], 4.0f);
	EXPECT_FLOAT_EQ(b.pMax[1], 5.0f);
	EXPECT_FLOAT_EQ(b.pMax[2], 6.0f);
}

// ---------------------------------------------------------------------------
// Diagonal
// ---------------------------------------------------------------------------

TEST(Bounds3, DiagonalIsExtent) {
	Bounds3<float> b(1.0f, 2.0f, 3.0f, 4.0f, 7.0f, 9.0f);
	EXPECT_FLOAT_EQ(b.diagonal(0), 3.0f);
	EXPECT_FLOAT_EQ(b.diagonal(1), 5.0f);
	EXPECT_FLOAT_EQ(b.diagonal(2), 6.0f);
}

TEST(Bounds3, DiagonalZeroForDegenerate) {
	Bounds3<float> b(1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f);
	EXPECT_FLOAT_EQ(b.diagonal(0), 0.0f);
	EXPECT_FLOAT_EQ(b.diagonal(1), 0.0f);
	EXPECT_FLOAT_EQ(b.diagonal(2), 0.0f);
}

// ---------------------------------------------------------------------------
// Offset (normalized fractional position)
// ---------------------------------------------------------------------------

TEST(Bounds3, OffsetAtMinIsZero) {
	Bounds3<float> b(0.0f, 0.0f, 0.0f, 4.0f, 6.0f, 8.0f);
	EXPECT_FLOAT_EQ(b.offset(0.0f, 0), 0.0f);
	EXPECT_FLOAT_EQ(b.offset(0.0f, 1), 0.0f);
	EXPECT_FLOAT_EQ(b.offset(0.0f, 2), 0.0f);
}

TEST(Bounds3, OffsetAtMaxIsOne) {
	Bounds3<float> b(0.0f, 0.0f, 0.0f, 4.0f, 6.0f, 8.0f);
	EXPECT_FLOAT_EQ(b.offset(4.0f, 0), 1.0f);
	EXPECT_FLOAT_EQ(b.offset(6.0f, 1), 1.0f);
	EXPECT_FLOAT_EQ(b.offset(8.0f, 2), 1.0f);
}

TEST(Bounds3, OffsetAtMidpointIsHalf) {
	Bounds3<float> b(0.0f, 0.0f, 0.0f, 4.0f, 6.0f, 8.0f);
	EXPECT_FLOAT_EQ(b.offset(2.0f, 0), 0.5f);
	EXPECT_FLOAT_EQ(b.offset(3.0f, 1), 0.5f);
	EXPECT_FLOAT_EQ(b.offset(4.0f, 2), 0.5f);
}

// ---------------------------------------------------------------------------
// Union (pbrt-v4: Union(Bounds3f, Bounds3f))
// ---------------------------------------------------------------------------

TEST(Bounds3, UnionTwoSeparateBoxes) {
	Bounds3<float> a(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
	Bounds3<float> b(2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f);
	Bounds3<float> u = bounds_union(a, b);
	EXPECT_FLOAT_EQ(u.pMin[0], 0.0f);
	EXPECT_FLOAT_EQ(u.pMin[1], 0.0f);
	EXPECT_FLOAT_EQ(u.pMin[2], 0.0f);
	EXPECT_FLOAT_EQ(u.pMax[0], 3.0f);
	EXPECT_FLOAT_EQ(u.pMax[1], 3.0f);
	EXPECT_FLOAT_EQ(u.pMax[2], 3.0f);
}

TEST(Bounds3, UnionWithSelfIsUnchanged) {
	Bounds3<float> a(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
	Bounds3<float> u = bounds_union(a, a);
	for (int i = 0; i < 3; ++i) {
		EXPECT_FLOAT_EQ(u.pMin[i], a.pMin[i]);
		EXPECT_FLOAT_EQ(u.pMax[i], a.pMax[i]);
	}
}

TEST(Bounds3, UnionOverlappingBoxes) {
	Bounds3<float> a(0.0f, 0.0f, 0.0f, 3.0f, 3.0f, 3.0f);
	Bounds3<float> b(1.0f, 1.0f, 1.0f, 5.0f, 5.0f, 5.0f);
	Bounds3<float> u = bounds_union(a, b);
	EXPECT_FLOAT_EQ(u.pMin[0], 0.0f);
	EXPECT_FLOAT_EQ(u.pMax[0], 5.0f);
}

// ---------------------------------------------------------------------------
// PointDistance (pbrt-v4: DistanceSquared/Distance(Point3f, Bounds3f))
// ---------------------------------------------------------------------------

TEST(Bounds3, PointInsideHasZeroDistance) {
	Bounds3<float> b(0.0f, 0.0f, 0.0f, 4.0f, 4.0f, 4.0f);
	const float p[3] = {2.0f, 2.0f, 2.0f};
	EXPECT_FLOAT_EQ(point_distance(b, p), 0.0f);
}

TEST(Bounds3, PointOnFaceHasZeroDistance) {
	Bounds3<float> b(0.0f, 0.0f, 0.0f, 4.0f, 4.0f, 4.0f);
	const float p[3] = {0.0f, 2.0f, 2.0f};
	EXPECT_FLOAT_EQ(point_distance(b, p), 0.0f);
}

TEST(Bounds3, PointOutsideDistanceAlongAxis) {
	Bounds3<float> b(0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 2.0f);
	const float p[3] = {5.0f, 1.0f, 1.0f};
	EXPECT_FLOAT_EQ(point_distance(b, p), 3.0f);
}

TEST(Bounds3, PointOutsideCornerDistance) {
	// Point at (4,4,0) from box [0..2, 0..2, 0..2]
	// nearest corner is (2,2,0): distance = sqrt(4+4) = 2*sqrt(2)
	Bounds3<float> b(0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 2.0f);
	const float p[3] = {4.0f, 4.0f, 1.0f};
	float expected = std::sqrt(4.0f + 4.0f + 0.0f);
	EXPECT_NEAR(point_distance(b, p), expected, 1e-5f);
}

// ---------------------------------------------------------------------------
// intersect_ray (pbrt-v4: Bounds3::IntersectP)
// ---------------------------------------------------------------------------

TEST(Bounds3, RayHitsUnitBox) {
	Bounds3<float> b(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
	const float o[3] = {-1.0f, 0.5f, 0.5f};
	const float d[3] = { 1.0f, 0.0f, 0.0f};
	float tMin, tMax;
	EXPECT_TRUE(b.intersect_ray(o, d, 100.0f, &tMin, &tMax));
	EXPECT_NEAR(tMin, 1.0f, 1e-5f);
	EXPECT_NEAR(tMax, 2.0f, 1e-5f);
}

TEST(Bounds3, RayMissesBoxParallel) {
	Bounds3<float> b(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
	const float o[3] = {-1.0f, 2.0f, 0.5f};   // y=2 outside [0..1]
	const float d[3] = { 1.0f, 0.0f, 0.0f};
	float tMin, tMax;
	EXPECT_FALSE(b.intersect_ray(o, d, 100.0f, &tMin, &tMax));
}

TEST(Bounds3, RayMissesTooShort) {
	Bounds3<float> b(5.0f, 0.0f, 0.0f, 6.0f, 1.0f, 1.0f);
	const float o[3] = {0.0f, 0.5f, 0.5f};
	const float d[3] = {1.0f, 0.0f, 0.0f};
	float tMin, tMax;
	// Box is at x=[5..6] but tMax=3 cuts it short
	EXPECT_FALSE(b.intersect_ray(o, d, 3.0f, &tMin, &tMax));
}

TEST(Bounds3, RayFromInsideHitsBothSides) {
	Bounds3<float> b(0.0f, 0.0f, 0.0f, 4.0f, 4.0f, 4.0f);
	const float o[3] = {2.0f, 2.0f, 2.0f};  // inside
	const float d[3] = {1.0f, 0.0f, 0.0f};
	float tMin, tMax;
	EXPECT_TRUE(b.intersect_ray(o, d, 100.0f, &tMin, &tMax));
	EXPECT_NEAR(tMax, 2.0f, 1e-5f); // exits at x=4 which is 2 units away
}
