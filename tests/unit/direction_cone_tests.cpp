// direction_cone_tests.cpp
// Unit tests for src/shared/direction_cone.h
//
// Groups:
//   1. Construction / IsEmpty / EntireSphere
//   2. Inside()
//   3. BoundSubtendedDirections()
//   4. Union()
//   5. ClosestVectorInCone()

#include <gtest/gtest.h>
#include <cmath>
#include "../../src/shared/direction_cone.h"

static constexpr float kPi = 3.14159265358979323846f;

// Helper: angle between two unit directions in degrees
static float angleDeg(float ax, float ay, float az, float bx, float by, float bz) {
	float d = ax*bx + ay*by + az*bz;
	d = d < -1.f ? -1.f : (d > 1.f ? 1.f : d);
	return std::acos(d) * (180.f / kPi);
}

// ---------------------------------------------------------------------------
// 1. Construction / IsEmpty / EntireSphere
// ---------------------------------------------------------------------------

TEST(DirectionCone, DefaultIsEmpty) {
	DirectionCone c;
	EXPECT_TRUE(c.IsEmpty());
}

TEST(DirectionCone, SingleDirectionNotEmpty) {
	DirectionCone c(0.f, 0.f, 1.f);
	EXPECT_FALSE(c.IsEmpty());
	EXPECT_FLOAT_EQ(c.cosTheta, 1.f);
}

TEST(DirectionCone, EntireSphereNotEmpty) {
	DirectionCone c = DirectionCone::EntireSphere();
	EXPECT_FALSE(c.IsEmpty());
	EXPECT_FLOAT_EQ(c.cosTheta, -1.f);
}

TEST(DirectionCone, AxisIsNormalized) {
	DirectionCone c(3.f, 4.f, 0.f, 0.5f);
	float len = std::sqrt(c.wx*c.wx + c.wy*c.wy + c.wz*c.wz);
	EXPECT_NEAR(len, 1.f, 1e-5f);
}

// ---------------------------------------------------------------------------
// 2. Inside()
// ---------------------------------------------------------------------------

TEST(DirectionCone, InsideAlongAxis) {
	// Exact axis direction must be inside
	DirectionCone c(0.f, 0.f, 1.f, 0.5f);
	EXPECT_TRUE(Inside(c, 0.f, 0.f, 1.f));
}

TEST(DirectionCone, InsideEmptyCone) {
	DirectionCone c;
	EXPECT_FALSE(Inside(c, 0.f, 0.f, 1.f));
}

TEST(DirectionCone, InsideEntireSphere) {
	DirectionCone c = DirectionCone::EntireSphere();
	EXPECT_TRUE(Inside(c, 1.f, 0.f, 0.f));
	EXPECT_TRUE(Inside(c, 0.f, -1.f, 0.f));
	EXPECT_TRUE(Inside(c, 0.f, 0.f, -1.f));
}

TEST(DirectionCone, InsideBoundary) {
	// cos(45 deg) cone around +Z; +X is at 90 deg -- outside
	float cos45 = std::cos(kPi / 4.f);
	DirectionCone c(0.f, 0.f, 1.f, cos45);
	EXPECT_FALSE(Inside(c, 1.f, 0.f, 0.f));
	// Direction at 44 deg from +Z should be inside
	float v44 = std::cos(44.f * kPi / 180.f);
	float h44 = std::sin(44.f * kPi / 180.f);
	EXPECT_TRUE(Inside(c, h44, 0.f, v44));
}

// ---------------------------------------------------------------------------
// 3. BoundSubtendedDirections()
// ---------------------------------------------------------------------------

TEST(DirectionCone, BoundSubtended_AxisAligned) {
	// Unit box [0,1]^3, viewed from far above along +Z (p at z=100)
	// Axis points from p=(0.5,0.5,100) toward box center=(0.5,0.5,0.5): wz < 0
	DirectionCone c = BoundSubtendedDirections(0,0,0, 1,1,1, 0.5f, 0.5f, 100.f);
	EXPECT_FALSE(c.IsEmpty());
	EXPECT_LT(c.wz, 0.f);  // axis points downward toward the box
}

TEST(DirectionCone, BoundSubtended_AxisPointsToBox) {
	// p far above, box centered at origin
	DirectionCone c = BoundSubtendedDirections(-1,-1,-1, 1,1,1, 0.f, 0.f, 100.f);
	EXPECT_FALSE(c.IsEmpty());
	// Axis should point downward (-Z direction) from p toward origin
	EXPECT_LT(c.wz, 0.f);
}

TEST(DirectionCone, BoundSubtended_InsideBox_EntireSphere) {
	// Point inside the bounding sphere -> EntireSphere
	DirectionCone c = BoundSubtendedDirections(0,0,0, 2,2,2, 1.f, 1.f, 1.f);
	EXPECT_FLOAT_EQ(c.cosTheta, -1.f);
}

TEST(DirectionCone, BoundSubtended_CoversBoxCorners) {
	// Verify all 8 corners of the box are Inside the returned cone
	DirectionCone c = BoundSubtendedDirections(0,0,0, 1,1,1, 0.f, 0.f, 10.f);
	float corners[8][3] = {
		{0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}
	};
	for (auto& corner : corners) {
		float dx = corner[0] - 0.f, dy = corner[1] - 0.f, dz = corner[2] - 10.f;
		EXPECT_TRUE(Inside(c, dx, dy, dz)) << "corner=(" << corner[0] << "," << corner[1] << "," << corner[2] << ")";
	}
}

// ---------------------------------------------------------------------------
// 4. Union()
// ---------------------------------------------------------------------------

TEST(DirectionCone, Union_EmptyWithCone) {
	DirectionCone empty;
	DirectionCone c(1.f, 0.f, 0.f, 0.7f);
	DirectionCone u = Union(empty, c);
	EXPECT_FALSE(u.IsEmpty());
	EXPECT_FLOAT_EQ(u.cosTheta, 0.7f);
}

TEST(DirectionCone, Union_TwoIdentical) {
	DirectionCone c(0.f, 0.f, 1.f, 0.5f);
	DirectionCone u = Union(c, c);
	EXPECT_FALSE(u.IsEmpty());
	EXPECT_NEAR(u.cosTheta, 0.5f, 1e-4f);
}

TEST(DirectionCone, Union_OneContainsOther) {
	// Wide cone (30 deg) contains narrow cone (10 deg) along same axis
	DirectionCone wide(0.f, 0.f, 1.f, std::cos(30.f * kPi / 180.f));
	DirectionCone narrow(0.f, 0.f, 1.f, std::cos(10.f * kPi / 180.f));
	DirectionCone u = Union(wide, narrow);
	// Result should equal the wide cone
	EXPECT_NEAR(u.cosTheta, wide.cosTheta, 1e-4f);
}

TEST(DirectionCone, Union_OppositeDirections_EntireSphere) {
	// Two single-direction cones pointing in opposite directions
	DirectionCone a(0.f, 0.f,  1.f);  // +Z, half-angle=0
	DirectionCone b(0.f, 0.f, -1.f);  // -Z, half-angle=0
	DirectionCone u = Union(a, b);
	EXPECT_FLOAT_EQ(u.cosTheta, -1.f);  // EntireSphere
}

TEST(DirectionCone, Union_ContainsBothAxes) {
	// Union must contain both original axes
	DirectionCone a(1.f, 0.f, 0.f, std::cos(20.f * kPi / 180.f));
	DirectionCone b(0.f, 1.f, 0.f, std::cos(20.f * kPi / 180.f));
	DirectionCone u = Union(a, b);
	EXPECT_TRUE(Inside(u, 1.f, 0.f, 0.f));
	EXPECT_TRUE(Inside(u, 0.f, 1.f, 0.f));
}

// ---------------------------------------------------------------------------
// 5. ClosestVectorInCone()
// ---------------------------------------------------------------------------

TEST(DirectionCone, ClosestVector_InsideReturnsInput) {
	// Direction that is already inside the cone should be returned unchanged
	DirectionCone c(0.f, 0.f, 1.f, std::cos(45.f * kPi / 180.f));
	float ox, oy, oz;
	ClosestVectorInCone(c, 0.f, 0.f, 1.f, ox, oy, oz);
	EXPECT_NEAR(ox, 0.f, 1e-5f);
	EXPECT_NEAR(oy, 0.f, 1e-5f);
	EXPECT_NEAR(oz, 1.f, 1e-5f);
}

TEST(DirectionCone, ClosestVector_OutsideOnBoundary) {
	// Direction 90 deg away from +Z cone axis (half-angle = 45 deg)
	// Closest should be on the boundary, angle from axis == 45 deg
	float halfAngle = 45.f * kPi / 180.f;
	DirectionCone c(0.f, 0.f, 1.f, std::cos(halfAngle));
	float ox, oy, oz;
	ClosestVectorInCone(c, 1.f, 0.f, 0.f, ox, oy, oz);
	// The result must be inside the cone
	EXPECT_TRUE(Inside(c, ox, oy, oz));
	// And the angle from axis should be ~ halfAngle
	float ang = angleDeg(c.wx, c.wy, c.wz, ox, oy, oz);
	EXPECT_NEAR(ang, 45.f, 0.5f);
}

TEST(DirectionCone, ClosestVector_IsUnitLength) {
	DirectionCone c(0.f, 0.f, 1.f, 0.5f);
	float ox, oy, oz;
	ClosestVectorInCone(c, 1.f, 0.f, 0.f, ox, oy, oz);
	float len = std::sqrt(ox*ox + oy*oy + oz*oz);
	EXPECT_NEAR(len, 1.f, 1e-5f);
}
