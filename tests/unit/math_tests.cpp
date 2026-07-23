/**
 * @file math_tests.cpp
 * @brief Unit tests for vector and color math operations
 * 
 * Tests the core mathematical functions:
 * - Vector operations (dot, cross, normalize, length)
 * - Color operations (addition, multiplication, clamping)
 * - Interval clamping and utility functions
 */

#include <gtest/gtest.h>
#include "rtweekend.h"
#include "vec3.h"
#include "color.h"
#include "interval.h"
#include <cmath>

// ============================================================================
// Vector Construction Tests
// ============================================================================

TEST(Vec3Test, DefaultConstruction) {
	vec3 v;
	EXPECT_DOUBLE_EQ(v.x(), 0.0);
	EXPECT_DOUBLE_EQ(v.y(), 0.0);
	EXPECT_DOUBLE_EQ(v.z(), 0.0);
}

TEST(Vec3Test, ParameterizedConstruction) {
	vec3 v(1.0, 2.0, 3.0);
	EXPECT_DOUBLE_EQ(v.x(), 1.0);
	EXPECT_DOUBLE_EQ(v.y(), 2.0);
	EXPECT_DOUBLE_EQ(v.z(), 3.0);
}

TEST(Vec3Test, ArrayAccess) {
	vec3 v(10.0, 20.0, 30.0);
	EXPECT_DOUBLE_EQ(v[0], 10.0);
	EXPECT_DOUBLE_EQ(v[1], 20.0);
	EXPECT_DOUBLE_EQ(v[2], 30.0);
}

// ============================================================================
// Vector Arithmetic Tests
// ============================================================================

TEST(Vec3Test, Addition) {
	vec3 a(1.0, 2.0, 3.0);
	vec3 b(4.0, 5.0, 6.0);
	vec3 c = a + b;

	EXPECT_DOUBLE_EQ(c.x(), 5.0);
	EXPECT_DOUBLE_EQ(c.y(), 7.0);
	EXPECT_DOUBLE_EQ(c.z(), 9.0);
}

TEST(Vec3Test, Subtraction) {
	vec3 a(10.0, 8.0, 6.0);
	vec3 b(1.0, 2.0, 3.0);
	vec3 c = a - b;

	EXPECT_DOUBLE_EQ(c.x(), 9.0);
	EXPECT_DOUBLE_EQ(c.y(), 6.0);
	EXPECT_DOUBLE_EQ(c.z(), 3.0);
}

TEST(Vec3Test, ScalarMultiplication) {
	vec3 v(1.0, 2.0, 3.0);
	vec3 scaled = v * 2.0;

	EXPECT_DOUBLE_EQ(scaled.x(), 2.0);
	EXPECT_DOUBLE_EQ(scaled.y(), 4.0);
	EXPECT_DOUBLE_EQ(scaled.z(), 6.0);
}

TEST(Vec3Test, ScalarDivision) {
	vec3 v(10.0, 20.0, 30.0);
	vec3 divided = v / 10.0;

	EXPECT_DOUBLE_EQ(divided.x(), 1.0);
	EXPECT_DOUBLE_EQ(divided.y(), 2.0);
	EXPECT_DOUBLE_EQ(divided.z(), 3.0);
}

TEST(Vec3Test, ComponentWiseMultiplication) {
	vec3 a(2.0, 3.0, 4.0);
	vec3 b(5.0, 6.0, 7.0);
	vec3 c = a * b;

	EXPECT_DOUBLE_EQ(c.x(), 10.0);
	EXPECT_DOUBLE_EQ(c.y(), 18.0);
	EXPECT_DOUBLE_EQ(c.z(), 28.0);
}

// ============================================================================
// Vector Length Tests
// ============================================================================

TEST(Vec3Test, Length) {
	vec3 v(3.0, 4.0, 0.0);
	EXPECT_DOUBLE_EQ(v.length(), 5.0); // 3-4-5 triangle
}

TEST(Vec3Test, LengthSquared) {
	vec3 v(3.0, 4.0, 0.0);
	EXPECT_DOUBLE_EQ(v.length_squared(), 25.0);
}

TEST(Vec3Test, UnitVector) {
	vec3 v(3.0, 4.0, 0.0);
	vec3 u = unit_vector(v);

	EXPECT_DOUBLE_EQ(u.x(), 0.6);
	EXPECT_DOUBLE_EQ(u.y(), 0.8);
	EXPECT_DOUBLE_EQ(u.z(), 0.0);

	// Unit vector should have length 1
	EXPECT_NEAR(u.length(), 1.0, 1e-10);
}

// ============================================================================
// Dot and Cross Product Tests
// ============================================================================

TEST(Vec3Test, DotProduct) {
	vec3 a(1.0, 0.0, 0.0);
	vec3 b(0.0, 1.0, 0.0);

	EXPECT_DOUBLE_EQ(dot(a, b), 0.0); // Perpendicular vectors

	vec3 c(1.0, 2.0, 3.0);
	vec3 d(4.0, 5.0, 6.0);

	EXPECT_DOUBLE_EQ(dot(c, d), 32.0); // 1*4 + 2*5 + 3*6
}

TEST(Vec3Test, CrossProduct) {
	vec3 x(1.0, 0.0, 0.0);
	vec3 y(0.0, 1.0, 0.0);
	vec3 z = cross(x, y);

	EXPECT_DOUBLE_EQ(z.x(), 0.0);
	EXPECT_DOUBLE_EQ(z.y(), 0.0);
	EXPECT_DOUBLE_EQ(z.z(), 1.0);
}

TEST(Vec3Test, CrossProductAntiCommutative) {
	vec3 a(1.0, 2.0, 3.0);
	vec3 b(4.0, 5.0, 6.0);

	vec3 ab = cross(a, b);
	vec3 ba = cross(b, a);

	EXPECT_DOUBLE_EQ(ab.x(), -ba.x());
	EXPECT_DOUBLE_EQ(ab.y(), -ba.y());
	EXPECT_DOUBLE_EQ(ab.z(), -ba.z());
}

// ============================================================================
// Color Tests
// ============================================================================

TEST(ColorTest, RGBConstruction) {
	color c(0.5, 0.75, 1.0);
	EXPECT_DOUBLE_EQ(c.x(), 0.5);
	EXPECT_DOUBLE_EQ(c.y(), 0.75);
	EXPECT_DOUBLE_EQ(c.z(), 1.0);
}

TEST(ColorTest, ColorAddition) {
	color a(0.2, 0.3, 0.4);
	color b(0.5, 0.5, 0.5);
	color c = a + b;

	EXPECT_DOUBLE_EQ(c.x(), 0.7);
	EXPECT_DOUBLE_EQ(c.y(), 0.8);
	EXPECT_DOUBLE_EQ(c.z(), 0.9);
}

TEST(ColorTest, ColorMultiplication) {
	color a(0.5, 0.75, 1.0);
	color b(0.8, 0.8, 0.5);
	color c = a * b;

	EXPECT_DOUBLE_EQ(c.x(), 0.4);
	EXPECT_DOUBLE_EQ(c.y(), 0.6);
	EXPECT_DOUBLE_EQ(c.z(), 0.5);
}

TEST(ColorTest, ScalarMultiplication) {
	color c(0.2, 0.4, 0.6);
	color scaled = c * 2.0;

	EXPECT_DOUBLE_EQ(scaled.x(), 0.4);
	EXPECT_DOUBLE_EQ(scaled.y(), 0.8);
	EXPECT_DOUBLE_EQ(scaled.z(), 1.2);
}

// ============================================================================
// Interval Tests
// ============================================================================

TEST(IntervalTest, DefaultConstruction) {
	interval i;
	EXPECT_DOUBLE_EQ(i.min, +infinity);
	EXPECT_DOUBLE_EQ(i.max, -infinity);
}

TEST(IntervalTest, ParameterizedConstruction) {
	interval i(0.0, 1.0);
	EXPECT_DOUBLE_EQ(i.min, 0.0);
	EXPECT_DOUBLE_EQ(i.max, 1.0);
}

TEST(IntervalTest, Contains) {
	interval i(0.0, 1.0);

	EXPECT_FALSE(i.contains(-0.1));
	EXPECT_TRUE(i.contains(0.0));
	EXPECT_TRUE(i.contains(0.5));
	EXPECT_TRUE(i.contains(1.0));
	EXPECT_FALSE(i.contains(1.1));
}

TEST(IntervalTest, Surrounds) {
	interval i(0.0, 1.0);

	EXPECT_FALSE(i.surrounds(-0.1));
	EXPECT_FALSE(i.surrounds(0.0));  // Boundary not surrounded
	EXPECT_TRUE(i.surrounds(0.5));
	EXPECT_FALSE(i.surrounds(1.0));  // Boundary not surrounded
	EXPECT_FALSE(i.surrounds(1.1));
}

TEST(IntervalTest, Clamp) {
	interval i(0.0, 1.0);

	EXPECT_DOUBLE_EQ(i.clamp(-0.5), 0.0);
	EXPECT_DOUBLE_EQ(i.clamp(0.0), 0.0);
	EXPECT_DOUBLE_EQ(i.clamp(0.5), 0.5);
	EXPECT_DOUBLE_EQ(i.clamp(1.0), 1.0);
	EXPECT_DOUBLE_EQ(i.clamp(1.5), 1.0);
}

TEST(IntervalTest, Size) {
	interval i(2.0, 5.0);
	EXPECT_DOUBLE_EQ(i.size(), 3.0);
}

TEST(IntervalTest, Expand) {
	interval i(1.0, 3.0);
	interval expanded = i.expand(1.0);

	EXPECT_DOUBLE_EQ(expanded.min, 0.5);
	EXPECT_DOUBLE_EQ(expanded.max, 3.5);
}

// ============================================================================
// Special Case Tests
// ============================================================================

TEST(Vec3Test, ZeroVector) {
	vec3 zero;
	EXPECT_DOUBLE_EQ(zero.length(), 0.0);
	EXPECT_DOUBLE_EQ(zero.length_squared(), 0.0);
}

TEST(Vec3Test, NegativeComponents) {
	vec3 v(-1.0, -2.0, -3.0);
	EXPECT_DOUBLE_EQ(v.x(), -1.0);
	EXPECT_DOUBLE_EQ(v.y(), -2.0);
	EXPECT_DOUBLE_EQ(v.z(), -3.0);
}

TEST(Vec3Test, LargeValues) {
	vec3 v(1e10, 1e10, 1e10);
	EXPECT_GT(v.length(), 1e10);
	EXPECT_LT(v.length(), 2e10);
}

TEST(ColorTest, BlackColor) {
	color black(0.0, 0.0, 0.0);
	EXPECT_DOUBLE_EQ(black.x(), 0.0);
	EXPECT_DOUBLE_EQ(black.y(), 0.0);
	EXPECT_DOUBLE_EQ(black.z(), 0.0);
}

TEST(ColorTest, WhiteColor) {
	color white(1.0, 1.0, 1.0);
	EXPECT_DOUBLE_EQ(white.x(), 1.0);
	EXPECT_DOUBLE_EQ(white.y(), 1.0);
	EXPECT_DOUBLE_EQ(white.z(), 1.0);
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST(UtilityTest, DegreesToRadians) {
	EXPECT_NEAR(degrees_to_radians(0.0), 0.0, 1e-10);
	EXPECT_NEAR(degrees_to_radians(90.0), pi / 2.0, 1e-10);
	EXPECT_NEAR(degrees_to_radians(180.0), pi, 1e-10);
	EXPECT_NEAR(degrees_to_radians(360.0), 2.0 * pi, 1e-10);
}

TEST(UtilityTest, RandomDouble) {
	// Test that random_double returns values in [0, 1)
	for (int i = 0; i < 100; ++i) {
		double r = random_double();
		EXPECT_GE(r, 0.0);
		EXPECT_LT(r, 1.0);
	}
}

TEST(UtilityTest, RandomDoubleRange) {
	// Test that random_double(min, max) returns values in [min, max)
	double min = 5.0;
	double max = 10.0;

	for (int i = 0; i < 100; ++i) {
		double r = random_double(min, max);
		EXPECT_GE(r, min);
		EXPECT_LT(r, max);
	}
}
