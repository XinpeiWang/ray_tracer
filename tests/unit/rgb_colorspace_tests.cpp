// rgb_colorspace_tests.cpp -- Unit tests for src/shared/rgb_colorspace.h
// Validates RGBColorSpace against pbrt-v4 util/colorspace.h semantics.

#include "../../src/shared/rgb_colorspace.h"
#include <gtest/gtest.h>
#include <cmath>

// -----------------------------------------------------------------------
// Helper: apply 3x3 matrix to RGB triplet
// -----------------------------------------------------------------------
static void ApplyMatrix(const SquareMatrix<3>& M,
						float r, float g, float b,
						float& or_, float& og, float& ob) {
	or_ = (float)(M[0][0]*r + M[0][1]*g + M[0][2]*b);
	og  = (float)(M[1][0]*r + M[1][1]*g + M[1][2]*b);
	ob  = (float)(M[2][0]*r + M[2][1]*g + M[2][2]*b);
}

// -----------------------------------------------------------------------
// RGBColorSpace::sRGB -- construction and basic properties
// -----------------------------------------------------------------------

TEST(RGBColorSpaceTest, sRGBPrimariesMatchSpec) {
	// IEC 61966-2-1: r(0.64,0.33) g(0.30,0.60) b(0.15,0.06) w(0.3127,0.3290)
	const auto& cs = RGBColorSpace::sRGB();
	EXPECT_NEAR(cs.rx, 0.64f, 1e-5f);
	EXPECT_NEAR(cs.ry, 0.33f, 1e-5f);
	EXPECT_NEAR(cs.gx, 0.30f, 1e-5f);
	EXPECT_NEAR(cs.gy, 0.60f, 1e-5f);
	EXPECT_NEAR(cs.bx, 0.15f, 1e-5f);
	EXPECT_NEAR(cs.by, 0.06f, 1e-5f);
	EXPECT_NEAR(cs.wx, 0.3127f, 1e-5f);
	EXPECT_NEAR(cs.wy, 0.3290f, 1e-5f);
}

TEST(RGBColorSpaceTest, sRGBRoundtripRGBtoXYZtoRGB) {
	// RGB -> XYZ -> RGB should be identity
	const auto& cs = RGBColorSpace::sRGB();
	float r0 = 0.8f, g0 = 0.4f, b0 = 0.2f;
	XYZ xyz = cs.ToXYZ(r0, g0, b0);
	float r1, g1, b1;
	cs.FromXYZ(xyz.X, xyz.Y, xyz.Z, r1, g1, b1);
	EXPECT_NEAR(r1, r0, 1e-4f);
	EXPECT_NEAR(g1, g0, 1e-4f);
	EXPECT_NEAR(b1, b0, 1e-4f);
}

TEST(RGBColorSpaceTest, sRGBWhitePointMapsToEqualLuminance) {
	// Equal-energy white (1,1,1) should map to XYZ with X≈Y≈Z is not
	// required; but the D65 white point should have Y=1 in XYZ.
	const auto& cs = RGBColorSpace::sRGB();
	// The white point xy=(0.3127,0.3290) with Y=1 gives a known XYZ.
	// Applying RGBFromXYZ to that XYZ should give approximately (1,1,1).
	XYZ w = XYZ::FromxyY(0.3127f, 0.3290f, 1.f);
	float r, g, b;
	cs.FromXYZ(w.X, w.Y, w.Z, r, g, b);
	EXPECT_NEAR(r, 1.f, 0.01f);
	EXPECT_NEAR(g, 1.f, 0.01f);
	EXPECT_NEAR(b, 1.f, 0.01f);
}

TEST(RGBColorSpaceTest, sRGBXYZFromRGBRGBFromXYZAreInverses) {
	// XYZFromRGB * RGBFromXYZ should be close to identity
	const auto& cs = RGBColorSpace::sRGB();
	SquareMatrix<3> prod = cs.XYZFromRGB * cs.RGBFromXYZ;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j) {
			double expected = (i == j) ? 1.0 : 0.0;
			EXPECT_NEAR(prod[i][j], expected, 1e-5) << "M[" << i << "][" << j << "]";
		}
}

TEST(RGBColorSpaceTest, sRGBLuminanceVectorSumsToOne) {
	// For a calibrated color space the luminance weights should sum to 1
	// (they equal the Y row of XYZFromRGB, and Y of white = 1).
	const auto& cs = RGBColorSpace::sRGB();
	float lr, lg, lb;
	cs.LuminanceVector(lr, lg, lb);
	EXPECT_NEAR(lr + lg + lb, 1.f, 1e-4f);
	// Standard sRGB luminance weights: ~0.2126, 0.7152, 0.0722
	EXPECT_NEAR(lr, 0.2126f, 0.005f);
	EXPECT_NEAR(lg, 0.7152f, 0.005f);
	EXPECT_NEAR(lb, 0.0722f, 0.005f);
}

// -----------------------------------------------------------------------
// DCI-P3
// -----------------------------------------------------------------------

TEST(RGBColorSpaceTest, DCIP3RoundtripRGBtoXYZtoRGB) {
	const auto& cs = RGBColorSpace::DCI_P3();
	float r0 = 0.5f, g0 = 0.7f, b0 = 0.3f;
	XYZ xyz = cs.ToXYZ(r0, g0, b0);
	float r1, g1, b1;
	cs.FromXYZ(xyz.X, xyz.Y, xyz.Z, r1, g1, b1);
	EXPECT_NEAR(r1, r0, 1e-4f);
	EXPECT_NEAR(g1, g0, 1e-4f);
	EXPECT_NEAR(b1, b0, 1e-4f);
}

TEST(RGBColorSpaceTest, DCIP3MatricesAreInverses) {
	const auto& cs = RGBColorSpace::DCI_P3();
	SquareMatrix<3> prod = cs.XYZFromRGB * cs.RGBFromXYZ;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			EXPECT_NEAR(prod[i][j], (i==j)?1.0:0.0, 1e-5)
				<< "DCI-P3 M[" << i << "][" << j << "]";
}

// -----------------------------------------------------------------------
// Rec. 2020
// -----------------------------------------------------------------------

TEST(RGBColorSpaceTest, Rec2020RoundtripRGBtoXYZtoRGB) {
	const auto& cs = RGBColorSpace::Rec2020();
	float r0 = 0.3f, g0 = 0.6f, b0 = 0.9f;
	XYZ xyz = cs.ToXYZ(r0, g0, b0);
	float r1, g1, b1;
	cs.FromXYZ(xyz.X, xyz.Y, xyz.Z, r1, g1, b1);
	EXPECT_NEAR(r1, r0, 1e-4f);
	EXPECT_NEAR(g1, g0, 1e-4f);
	EXPECT_NEAR(b1, b0, 1e-4f);
}

TEST(RGBColorSpaceTest, Rec2020LuminanceWeightsSumToOne) {
	const auto& cs = RGBColorSpace::Rec2020();
	float lr, lg, lb;
	cs.LuminanceVector(lr, lg, lb);
	EXPECT_NEAR(lr + lg + lb, 1.f, 1e-4f);
	// BT.2020 luminance: ~0.2627, 0.6780, 0.0593
	EXPECT_NEAR(lr, 0.2627f, 0.005f);
	EXPECT_NEAR(lg, 0.6780f, 0.005f);
	EXPECT_NEAR(lb, 0.0593f, 0.005f);
}

// -----------------------------------------------------------------------
// ACES2065-1
// -----------------------------------------------------------------------

TEST(RGBColorSpaceTest, ACES2065RoundtripRGBtoXYZtoRGB) {
	const auto& cs = RGBColorSpace::ACES2065_1();
	float r0 = 0.4f, g0 = 0.5f, b0 = 0.6f;
	XYZ xyz = cs.ToXYZ(r0, g0, b0);
	float r1, g1, b1;
	cs.FromXYZ(xyz.X, xyz.Y, xyz.Z, r1, g1, b1);
	EXPECT_NEAR(r1, r0, 1e-4f);
	EXPECT_NEAR(g1, g0, 1e-4f);
	EXPECT_NEAR(b1, b0, 1e-4f);
}

TEST(RGBColorSpaceTest, ACES2065MatricesAreInverses) {
	const auto& cs = RGBColorSpace::ACES2065_1();
	SquareMatrix<3> prod = cs.XYZFromRGB * cs.RGBFromXYZ;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			EXPECT_NEAR(prod[i][j], (i==j)?1.0:0.0, 1e-5)
				<< "ACES M[" << i << "][" << j << "]";
}

// -----------------------------------------------------------------------
// ConvertRGBColorSpace
// -----------------------------------------------------------------------

TEST(RGBColorSpaceTest, ConvertSameSpaceIsIdentity) {
	// pbrt-v4: ConvertRGBColorSpace returns identity when from == to
	const auto& cs = RGBColorSpace::sRGB();
	SquareMatrix<3> M = ConvertRGBColorSpace(cs, cs);
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			EXPECT_NEAR(M[i][j], (i==j)?1.0:0.0, 1e-10)
				<< "Identity M[" << i << "][" << j << "]";
}

TEST(RGBColorSpaceTest, ConvertSRGBtoRec2020RoundtripIsIdentity) {
	// sRGB -> Rec2020 -> sRGB should yield identity matrix product
	SquareMatrix<3> fwd = ConvertRGBColorSpace(RGBColorSpace::sRGB(),  RGBColorSpace::Rec2020());
	SquareMatrix<3> bwd = ConvertRGBColorSpace(RGBColorSpace::Rec2020(), RGBColorSpace::sRGB());
	SquareMatrix<3> prod = bwd * fwd;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			EXPECT_NEAR(prod[i][j], (i==j)?1.0:0.0, 1e-5)
				<< "Roundtrip M[" << i << "][" << j << "]";
}

TEST(RGBColorSpaceTest, ConvertSRGBtoRec2020PreservesWhite) {
	// Converting D65 white (1,1,1) in sRGB to Rec2020 should stay (1,1,1)
	// since both use D65 illuminant.
	SquareMatrix<3> M = ConvertRGBColorSpace(RGBColorSpace::sRGB(), RGBColorSpace::Rec2020());
	float r, g, b;
	ApplyMatrix(M, 1.f, 1.f, 1.f, r, g, b);
	EXPECT_NEAR(r, 1.f, 0.01f);
	EXPECT_NEAR(g, 1.f, 0.01f);
	EXPECT_NEAR(b, 1.f, 0.01f);
}

TEST(RGBColorSpaceTest, EqualityOperator) {
	EXPECT_TRUE(RGBColorSpace::sRGB()   == RGBColorSpace::sRGB());
	EXPECT_TRUE(RGBColorSpace::Rec2020() == RGBColorSpace::Rec2020());
	EXPECT_FALSE(RGBColorSpace::sRGB()  == RGBColorSpace::Rec2020());
	EXPECT_FALSE(RGBColorSpace::DCI_P3() == RGBColorSpace::ACES2065_1());
}
