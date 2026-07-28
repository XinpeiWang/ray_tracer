// color_utils_tests.cpp -- Unit tests for src/shared/color_utils.h
// Validates LinearToSRGB, SRGBToLinear, SRGB8ToLinear, LinearToSRGB8,
// XYZ class, and WhiteBalance against pbrt-v4 semantics.

#include "../../src/shared/color_utils.h"
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>

// -----------------------------------------------------------------------
// LinearToSRGB -- known values
// -----------------------------------------------------------------------
TEST(ColorUtilsTest, LinearToSRGBZero) {
	EXPECT_NEAR(LinearToSRGB(0.f), 0.f, 1e-6f);
}

TEST(ColorUtilsTest, LinearToSRGBLinearSegment) {
	// For x <= 0.0031308, result = 12.92 * x
	float x = 0.001f;
	EXPECT_NEAR(LinearToSRGB(x), 12.92f * x, 1e-5f);
}

TEST(ColorUtilsTest, LinearToSRGBOne) {
	// Linear 1.0 should map to sRGB 1.0
	EXPECT_NEAR(LinearToSRGB(1.f), 1.f, 1e-4f);
}

TEST(ColorUtilsTest, LinearToSRGBMidGray) {
	// sRGB mid-gray is ~0.2140 linear -> ~0.5 sRGB (approximate)
	float encoded = LinearToSRGB(0.2140f);
	EXPECT_GT(encoded, 0.45f);
	EXPECT_LT(encoded, 0.55f);
}

TEST(ColorUtilsTest, LinearToSRGBMonotone) {
	float prev = LinearToSRGB(0.f);
	for (int i = 1; i <= 100; ++i) {
		float cur = LinearToSRGB(i / 100.f);
		EXPECT_GE(cur, prev) << "not monotone at i=" << i;
		prev = cur;
	}
}

// -----------------------------------------------------------------------
// SRGBToLinear -- inverse of LinearToSRGB
// -----------------------------------------------------------------------
TEST(ColorUtilsTest, SRGBToLinearZero) {
	EXPECT_NEAR(SRGBToLinear(0.f), 0.f, 1e-6f);
}

TEST(ColorUtilsTest, SRGBToLinearOne) {
	EXPECT_NEAR(SRGBToLinear(1.f), 1.f, 1e-4f);
}

TEST(ColorUtilsTest, SRGBLinearRoundTrip) {
	for (float x : {0.001f, 0.01f, 0.1f, 0.5f, 0.9f, 0.99f}) {
		float roundtrip = SRGBToLinear(LinearToSRGB(x));
		EXPECT_NEAR(roundtrip, x, 1e-4f) << "roundtrip failed at x=" << x;
	}
}

// -----------------------------------------------------------------------
// LinearToSRGB8 -- quantization
// -----------------------------------------------------------------------
TEST(ColorUtilsTest, LinearToSRGB8Zero) {
	EXPECT_EQ(LinearToSRGB8(0.f), 0);
}

TEST(ColorUtilsTest, LinearToSRGB8One) {
	EXPECT_EQ(LinearToSRGB8(1.f), 255);
}

TEST(ColorUtilsTest, LinearToSRGB8Midpoint) {
	// 0.2140 linear -> ~128 (roughly mid-gray)
	uint8_t v = LinearToSRGB8(0.2140f);
	EXPECT_GE(v, 115);
	EXPECT_LE(v, 140);
}

TEST(ColorUtilsTest, LinearToSRGB8Monotone) {
	uint8_t prev = LinearToSRGB8(0.f);
	for (int i = 1; i <= 100; ++i) {
		uint8_t cur = LinearToSRGB8(i / 100.f);
		EXPECT_GE(cur, prev) << "not monotone at i=" << i;
		prev = cur;
	}
}

// -----------------------------------------------------------------------
// SRGB8ToLinear -- LUT correctness
// -----------------------------------------------------------------------
TEST(ColorUtilsTest, SRGB8ToLinearZero) {
	EXPECT_NEAR(SRGB8ToLinear(0), 0.f, 1e-6f);
}

TEST(ColorUtilsTest, SRGB8ToLinear255) {
	EXPECT_NEAR(SRGB8ToLinear(255), 1.f, 1e-4f);
}

TEST(ColorUtilsTest, SRGB8ToLinearRoundTrip) {
	for (int i = 0; i < 256; ++i) {
		float linear = SRGB8ToLinear(static_cast<uint8_t>(i));
		uint8_t back = LinearToSRGB8(linear);
		// Quantisation can shift by at most 1 step
		EXPECT_LE(std::abs(static_cast<int>(back) - i), 1) << "at i=" << i;
	}
}

// -----------------------------------------------------------------------
// XYZ -- construction and operators
// -----------------------------------------------------------------------
TEST(ColorUtilsTest, XYZDefaultZero) {
	XYZ xyz;
	EXPECT_EQ(xyz.X, 0.f);
	EXPECT_EQ(xyz.Y, 0.f);
	EXPECT_EQ(xyz.Z, 0.f);
}

TEST(ColorUtilsTest, XYZArithmetic) {
	XYZ a(1.f, 2.f, 3.f), b(0.5f, 0.5f, 0.5f);
	XYZ sum = a + b;
	EXPECT_NEAR(sum.X, 1.5f, 1e-6f);
	EXPECT_NEAR(sum.Y, 2.5f, 1e-6f);
	EXPECT_NEAR(sum.Z, 3.5f, 1e-6f);
}

TEST(ColorUtilsTest, XYZScalarMultiply) {
	XYZ a(1.f, 2.f, 4.f);
	XYZ b = a * 3.f;
	EXPECT_NEAR(b.X, 3.f, 1e-6f);
	EXPECT_NEAR(b.Y, 6.f, 1e-6f);
	EXPECT_NEAR(b.Z, 12.f, 1e-6f);
}

TEST(ColorUtilsTest, XYZFromxyY) {
	// D65 white: xy = (0.3127, 0.3290), Y = 1
	XYZ d65 = XYZ::FromxyY(0.3127f, 0.3290f, 1.f);
	EXPECT_NEAR(d65.Y, 1.f, 1e-5f);
	// Verify chromaticity round-trip
	float cx = d65.x(), cy = d65.y();
	EXPECT_NEAR(cx, 0.3127f, 1e-4f);
	EXPECT_NEAR(cy, 0.3290f, 1e-4f);
}

TEST(ColorUtilsTest, XYZLerp) {
	XYZ a(0.f, 0.f, 0.f), b(2.f, 4.f, 6.f);
	XYZ mid = Lerp(0.5f, a, b);
	EXPECT_NEAR(mid.X, 1.f, 1e-5f);
	EXPECT_NEAR(mid.Y, 2.f, 1e-5f);
	EXPECT_NEAR(mid.Z, 3.f, 1e-5f);
}

// -----------------------------------------------------------------------
// WhiteBalance -- identity when src == dst
// -----------------------------------------------------------------------
TEST(ColorUtilsTest, WhiteBalanceIdentity) {
	// D65 -> D65 should give the identity matrix
	float d65x = 0.3127f, d65y = 0.3290f;
	SquareMatrix<3> M = WhiteBalance(d65x, d65y, d65x, d65y);
	// Diagonal should be ~1, off-diagonal ~0
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j) {
			float expected = (i == j) ? 1.f : 0.f;
			EXPECT_NEAR(M[i][j], expected, 1e-3f) << "M[" << i << "][" << j << "]";
		}
}

TEST(ColorUtilsTest, WhiteBalanceD65toD50) {
	// D65 xy = (0.3127, 0.3290), D50 xy = (0.3457, 0.3585)
	SquareMatrix<3> M = WhiteBalance(0.3127f, 0.3290f, 0.3457f, 0.3585f);
	// Applying M to D65 XYZ should produce approximately D50 XYZ.
	XYZ d65 = XYZ::FromxyY(0.3127f, 0.3290f, 1.f);
	XYZ adapted = M * d65;
	XYZ d50 = XYZ::FromxyY(0.3457f, 0.3585f, 1.f);
	EXPECT_NEAR(adapted.x(), d50.x(), 0.01f);
	EXPECT_NEAR(adapted.y(), d50.y(), 0.01f);
}
