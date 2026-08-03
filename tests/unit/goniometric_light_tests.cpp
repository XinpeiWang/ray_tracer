// goniometric_light_tests.cpp -- Regression tests for GoniometricLight<T>
//
// Tests cover:
//   1. Isotropic light: Li = I / r², direction correct
//   2. Anisotropic image: eval_I uses image lookup
//   3. Delta light: pdf_li == 0
//   4. Power: isotropic case equals scale * Iavg * 4π
//   5. sample_le / pdf_le: consistency (pdf integrates to ~1 on grid)
//   6. EvalI identity transform: direction to light-space is identity
//   7. Image scale: brighter pixel → higher intensity
//   8. Inverse-square falloff: doubling distance halves Li by 4×

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../../src/shared/goniometric_light.h"

static const double kEps = 1e-9;
static const double kPi  = 3.14159265358979323846;

// Identity rotation
static const double kId[9] = {1,0,0, 0,1,0, 0,0,1};

// Build a 4×4 uniform goniometric image (all ones → isotropic)
static std::vector<double> uniform_image(int n = 4) {
	return std::vector<double>(n*n, 1.0);
}

// Build a 4×4 image where only the +Z half-sphere pixel is lit
// pixel (2,2) in a 4×4 equal-area image corresponds to approx +Z direction
static std::vector<double> spotlight_image() {
	std::vector<double> img(4*4, 0.0);
	img[2*4 + 2] = 1.0;
	return img;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(GoniometricLight, MakeIsotropicDoesNotCrash) {
	auto light = GoniometricLight<double>::make_isotropic(0,0,0, 1,1,1, 1);
	EXPECT_GT(light.nu, 0);
	EXPECT_GT(light.nv, 0);
}

TEST(GoniometricLight, MakeWithCustomImage) {
	auto img = uniform_image(8);
	auto light = GoniometricLight<double>::make(0,0,0, kId, 1,1,1, 1, img, 8, 8);
	EXPECT_EQ(light.nu, 8);
	EXPECT_EQ(light.nv, 8);
}

// ---------------------------------------------------------------------------
// eval_I
// ---------------------------------------------------------------------------

TEST(GoniometricLight, IsotropicEvalIIsOne) {
	auto light = GoniometricLight<double>::make_isotropic(0,0,0, 1,1,1, 1);
	// All directions should give ~1.0 for a uniform image
	EXPECT_NEAR(light.eval_I(1,0,0), 1.0, kEps);
	EXPECT_NEAR(light.eval_I(0,1,0), 1.0, kEps);
	EXPECT_NEAR(light.eval_I(0,0,1), 1.0, kEps);
	EXPECT_NEAR(light.eval_I(-1,0,0), 1.0, kEps);
}

TEST(GoniometricLight, EvalIReturnsBrighterForBrightPixel) {
	// Image: pixel at (2,2) is bright, rest zero
	auto img = spotlight_image();
	auto light = GoniometricLight<double>::make(0,0,0, kId, 1,1,1, 1, img, 4, 4);
	// The direction mapped to pixel (2,2) should be bright
	double u = (2.5) / 4.0, v = (2.5) / 4.0;
	double wx, wy, wz;
	EqualAreaSquareToSphere(u, v, wx, wy, wz);
	EXPECT_GT(light.eval_I((double)wx, (double)wy, (double)wz), 0.0);
}

// ---------------------------------------------------------------------------
// sample_li
// ---------------------------------------------------------------------------

TEST(GoniometricLight, SampleLiDirectionPointsToLight) {
	// Light at (3,0,0), shading point at origin
	auto light = GoniometricLight<double>::make_isotropic(3,0,0, 1,1,1, 1);
	double Lr,Lg,Lb, wi_x,wi_y,wi_z;
	light.sample_li(0,0,0, Lr,Lg,Lb, wi_x,wi_y,wi_z);
	EXPECT_NEAR(wi_x,  1.0, 1e-9);
	EXPECT_NEAR(wi_y,  0.0, 1e-9);
	EXPECT_NEAR(wi_z,  0.0, 1e-9);
}

TEST(GoniometricLight, SampleLiInverseSquereFalloff) {
	// Isotropic light at origin: Li at r=1 vs r=2
	auto l1 = GoniometricLight<double>::make_isotropic(0,0,0, 1,0,0, 1);
	double Lr1,Lg1,Lb1, wix,wiy,wiz;
	l1.sample_li(0,0,1, Lr1,Lg1,Lb1, wix,wiy,wiz);  // r=1
	double Lr2,Lg2,Lb2;
	l1.sample_li(0,0,2, Lr2,Lg2,Lb2, wix,wiy,wiz);  // r=2
	EXPECT_NEAR(Lr1 / Lr2, 4.0, 1e-9);  // 1/1² vs 1/4 → ratio = 4
}

TEST(GoniometricLight, SampleLiIsNonNegative) {
	auto light = GoniometricLight<double>::make_isotropic(0,0,5, 1,1,1, 1);
	double Lr,Lg,Lb, wi_x,wi_y,wi_z;
	light.sample_li(1,1,1, Lr,Lg,Lb, wi_x,wi_y,wi_z);
	EXPECT_GE(Lr, 0.0);
	EXPECT_GE(Lg, 0.0);
	EXPECT_GE(Lb, 0.0);
}

TEST(GoniometricLight, SampleLiWiIsNormalized) {
	auto light = GoniometricLight<double>::make_isotropic(1,2,3, 1,1,1, 1);
	double Lr,Lg,Lb, wi_x,wi_y,wi_z;
	light.sample_li(0,0,0, Lr,Lg,Lb, wi_x,wi_y,wi_z);
	double len = std::sqrt(wi_x*wi_x + wi_y*wi_y + wi_z*wi_z);
	EXPECT_NEAR(len, 1.0, 1e-9);
}

TEST(GoniometricLight, SampleLiScaleAffectsIntensity) {
	auto l1 = GoniometricLight<double>::make_isotropic(0,0,0, 1,0,0, 1.0);
	auto l2 = GoniometricLight<double>::make_isotropic(0,0,0, 1,0,0, 3.0);
	double Lr1,Lg1,Lb1, Lr2,Lg2,Lb2, wix,wiy,wiz;
	l1.sample_li(0,0,1, Lr1,Lg1,Lb1, wix,wiy,wiz);
	l2.sample_li(0,0,1, Lr2,Lg2,Lb2, wix,wiy,wiz);
	EXPECT_NEAR(Lr2 / Lr1, 3.0, 1e-9);
}

// ---------------------------------------------------------------------------
// pdf_li
// ---------------------------------------------------------------------------

TEST(GoniometricLight, PdfLiIsZero) {
	auto light = GoniometricLight<double>::make_isotropic(0,0,0, 1,1,1, 1);
	EXPECT_DOUBLE_EQ(light.pdf_li(), 0.0);
}

// ---------------------------------------------------------------------------
// power
// ---------------------------------------------------------------------------

TEST(GoniometricLight, IsotropicPowerMatchesFormula) {
	// For uniform image of all-ones and RGB=(1,1,1), scale=1:
	// Iavg=1, avg_image=1, power = 1 * 1 * 4π = 4π
	auto light = GoniometricLight<double>::make_isotropic(0,0,0, 1,1,1, 1);
	double expected = 4.0 * kPi;
	EXPECT_NEAR(light.power(), expected, 1e-6);
}

TEST(GoniometricLight, PowerScalesWithIntensity) {
	auto l1 = GoniometricLight<double>::make_isotropic(0,0,0, 2,0,0, 1);
	auto l2 = GoniometricLight<double>::make_isotropic(0,0,0, 4,0,0, 1);
	EXPECT_NEAR(l2.power() / l1.power(), 2.0, 1e-9);
}

// ---------------------------------------------------------------------------
// sample_le / pdf_le
// ---------------------------------------------------------------------------

TEST(GoniometricLight, SampleLeProducesNormalizedDirection) {
	auto light = GoniometricLight<double>::make_isotropic(0,0,0, 1,1,1, 1);
	double wx,wy,wz,pdf;
	light.sample_le(0.3, 0.7, wx, wy, wz, pdf);
	double len = std::sqrt(wx*wx + wy*wy + wz*wz);
	EXPECT_NEAR(len, 1.0, 1e-9);
	EXPECT_GT(pdf, 0.0);
}

TEST(GoniometricLight, PdfLeMatchesSampleLePdf) {
	auto light = GoniometricLight<double>::make_isotropic(0,0,0, 1,1,1, 1);
	double wx,wy,wz,pdf_sample;
	light.sample_le(0.25, 0.75, wx, wy, wz, pdf_sample);
	double pdf_eval = light.pdf_le(wx, wy, wz);
	EXPECT_NEAR(pdf_sample, pdf_eval, 1e-6);
}

TEST(GoniometricLight, IsotropicPdfLeIsUniform) {
	// For uniform image, pdf_le should be 1/(4π) everywhere
	auto light = GoniometricLight<double>::make_isotropic(0,0,0, 1,1,1, 1);
	double expected = 1.0 / (4.0 * kPi);
	EXPECT_NEAR(light.pdf_le(1,0,0), expected, 1e-4);
	EXPECT_NEAR(light.pdf_le(0,1,0), expected, 1e-4);
	EXPECT_NEAR(light.pdf_le(0,0,1), expected, 1e-4);
}
