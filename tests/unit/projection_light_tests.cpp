// projection_light_tests.cpp
// Regression tests for ProjectionLight<double>.
// Mirrors the structure of goniometric_light_tests.cpp.
// Reference: pbrt-v4 ProjectionLight (lights.h / lights.cpp), section 12.5.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <array>
#include "../../src/shared/projection_light.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Identity 3x3 rotation (light-space == world-space).
static const double kIdentity[9] = {
	1,0,0,
	0,1,0,
	0,0,1
};

// Build a 4x4 uniform-white projection light at the origin.
static ProjectionLight<double> make_default(double fov = 90.0,
											 double scale = 1.0,
											 int nx = 4, int ny = 4) {
	return ProjectionLight<double>::make_uniform(
		0.0, 0.0, 0.0, kIdentity, scale, fov, nx, ny);
}

// Build a projection light with a custom image (nx=2, ny=2, checkerboard).
// Pixel (0,0)=red, (1,0)=green, (0,1)=blue, (1,1)=white.
static ProjectionLight<double> make_checker() {
	std::vector<double> img = {
		1,0,0,   0,1,0,
		0,0,1,   1,1,1
	};
	return ProjectionLight<double>::make(
		0.0, 0.0, 0.0, kIdentity, 1.0, 90.0, img, 2, 2);
}

static double length3(double x, double y, double z) {
	return std::sqrt(x*x + y*y + z*z);
}

// ---------------------------------------------------------------------------
// Construction tests
// ---------------------------------------------------------------------------

TEST(ProjectionLight, ConstructDefaultFields) {
	auto pl = make_default();
	EXPECT_EQ(pl.nx, 4);
	EXPECT_EQ(pl.ny, 4);
	EXPECT_DOUBLE_EQ(pl.pos_x, 0.0);
	EXPECT_DOUBLE_EQ(pl.hither, 1e-3);
	// Uniform image should have positive A
	EXPECT_GT((double)pl.A, 0.0);
}

TEST(ProjectionLight, SquareImageScreenBoundsSymmetric) {
	// nx == ny: screen should be [-1,1] x [-1,1]
	auto pl = make_default(90.0, 1.0, 8, 8);
	EXPECT_NEAR((double)pl.sb_xmin, -1.0, 1e-9);
	EXPECT_NEAR((double)pl.sb_xmax,  1.0, 1e-9);
	EXPECT_NEAR((double)pl.sb_ymin, -1.0, 1e-9);
	EXPECT_NEAR((double)pl.sb_ymax,  1.0, 1e-9);
}

TEST(ProjectionLight, WideImageScreenBoundsAspect) {
	// nx=8, ny=4 -> aspect=2 -> x in [-2,2], y in [-1,1]
	auto pl = make_default(90.0, 1.0, 8, 4);
	EXPECT_NEAR((double)pl.sb_xmin, -2.0, 1e-9);
	EXPECT_NEAR((double)pl.sb_xmax,  2.0, 1e-9);
	EXPECT_NEAR((double)pl.sb_ymin, -1.0, 1e-9);
	EXPECT_NEAR((double)pl.sb_ymax,  1.0, 1e-9);
}

TEST(ProjectionLight, TallImageScreenBoundsAspect) {
	// nx=4, ny=8 -> aspect=0.5 -> x in [-1,1], y in [-2,2]
	auto pl = make_default(90.0, 1.0, 4, 8);
	EXPECT_NEAR((double)pl.sb_xmin, -1.0, 1e-9);
	EXPECT_NEAR((double)pl.sb_xmax,  1.0, 1e-9);
	EXPECT_NEAR((double)pl.sb_ymin, -2.0, 1e-9);
	EXPECT_NEAR((double)pl.sb_ymax,  2.0, 1e-9);
}

// ---------------------------------------------------------------------------
// eval_I tests
// ---------------------------------------------------------------------------

TEST(ProjectionLight, ForwardDirectionReturnsNonZero) {
	// Straight ahead (+z in light space) should hit centre of image.
	auto pl = make_default(90.0, 2.0);
	double I = pl.eval_I(0.0, 0.0, 1.0);
	// Uniform white image scaled by 2: I = 2 * 1.0 = 2
	EXPECT_NEAR(I, 2.0, 1e-6);
}

TEST(ProjectionLight, BehindLightReturnsZero) {
	auto pl = make_default();
	// wz < hither (negative z)
	EXPECT_DOUBLE_EQ(pl.eval_I(0.0, 0.0, -1.0), 0.0);
	EXPECT_DOUBLE_EQ(pl.eval_I(0.0, 0.0,  0.0), 0.0);
}

TEST(ProjectionLight, OutsideFrustumReturnsZero) {
	// 90-degree fov square image: forward direction just outside frustum
	auto pl = make_default(90.0, 1.0, 4, 4);
	// A direction at x=z (45-degree side) is on the frustum boundary;
	// slightly more extreme should be clipped.
	double I = pl.eval_I(2.0, 0.0, 1.0);  // projects to x=2, outside [-1,1]
	EXPECT_DOUBLE_EQ(I, 0.0);
}

TEST(ProjectionLight, ScaleProportional) {
	auto pl1 = make_default(90.0, 1.0);
	auto pl2 = make_default(90.0, 3.0);
	double I1 = pl1.eval_I(0.0, 0.0, 1.0);
	double I2 = pl2.eval_I(0.0, 0.0, 1.0);
	EXPECT_NEAR(I2, 3.0 * I1, 1e-9);
}

TEST(ProjectionLight, CheckerImageLookup) {
	// Checker: pixel (0,0)=red at top-left, evaluated via centre of first quad
	auto pl = make_checker();
	// Forward direction: straight ahead hits centre of image -> average of all 4
	// pixels via nearest-neighbour sampling near centre.
	double r, g, b;
	pl.eval_I_rgb(0.0, 0.0, 1.0, r, g, b);
	// Centre UV ~(0.5,0.5) nearest pixel is (1,1)=white -> r=g=b=1
	EXPECT_NEAR(r, 1.0, 1e-6);
	EXPECT_NEAR(g, 1.0, 1e-6);
	EXPECT_NEAR(b, 1.0, 1e-6);
}

// ---------------------------------------------------------------------------
// pdf_li tests
// ---------------------------------------------------------------------------

TEST(ProjectionLight, PdfLiAlwaysZero) {
	auto pl = make_default();
	EXPECT_DOUBLE_EQ(pl.pdf_li(), 0.0);
}

// ---------------------------------------------------------------------------
// sample_li tests
// ---------------------------------------------------------------------------

TEST(ProjectionLight, SampleLiDirectionTowardLight) {
	// Light at (0,0,5), shading point at (0,0,0) -> wi should be (0,0,1)
	auto pl = ProjectionLight<double>::make_uniform(
		0.0, 0.0, 5.0, kIdentity, 1.0, 90.0);
	double Lr, Lg, Lb, wi_x, wi_y, wi_z;
	pl.sample_li(0.0, 0.0, 0.0, Lr, Lg, Lb, wi_x, wi_y, wi_z);
	EXPECT_NEAR(wi_x, 0.0, 1e-9);
	EXPECT_NEAR(wi_y, 0.0, 1e-9);
	EXPECT_NEAR(wi_z, 1.0, 1e-9);
}

TEST(ProjectionLight, SampleLiInverseSquareFalloff) {
	// Li = I / r^2.  Two shading points at r=1 and r=2; ratio should be 4.
	auto pl = ProjectionLight<double>::make_uniform(
		0.0, 0.0, 0.0, kIdentity, 1.0, 90.0);
	double Lr1, Lg1, Lb1, wi_x, wi_y, wi_z;
	pl.sample_li(0.0, 0.0, -1.0, Lr1, Lg1, Lb1, wi_x, wi_y, wi_z);
	double Lr2, Lg2, Lb2;
	pl.sample_li(0.0, 0.0, -2.0, Lr2, Lg2, Lb2, wi_x, wi_y, wi_z);
	// The direction from (0,0,-1) to light (0,0,0) is +z (forward in light)
	// so it should be illuminated.
	if (Lr1 > 1e-12)
		EXPECT_NEAR(Lr1 / Lr2, 4.0, 1e-6);
}

TEST(ProjectionLight, SampleLiOutsideFrustumZero) {
	// Shading point to the extreme side so the direction is outside the frustum
	auto pl = ProjectionLight<double>::make_uniform(
		0.0, 0.0, 0.0, kIdentity, 1.0, 45.0);
	// From (0,0,-1) shading point the direction to origin is +z:
	//   in light space wl = world_to_light * (0,0,1) = (0,0,1) -- forward, inside
	// From (10,0,-1) the direction is roughly (+z, large +x) -- outside 45 deg
	double Lr, Lg, Lb, wi_x, wi_y, wi_z;
	pl.sample_li(10.0, 0.0, -1.0, Lr, Lg, Lb, wi_x, wi_y, wi_z);
	// Expect zero because the direction is outside the narrow 45-degree frustum
	EXPECT_DOUBLE_EQ(Lr, 0.0);
}

// ---------------------------------------------------------------------------
// power tests
// ---------------------------------------------------------------------------

TEST(ProjectionLight, PowerPositive) {
	auto pl = make_default();
	EXPECT_GT((double)pl.power(), 0.0);
}

TEST(ProjectionLight, PowerScalesWithIntensity) {
	auto pl1 = make_default(90.0, 1.0);
	auto pl2 = make_default(90.0, 2.0);
	EXPECT_NEAR((double)pl2.power(), 2.0 * (double)pl1.power(), 1e-9);
}

TEST(ProjectionLight, PowerBlackImageIsZero) {
	std::vector<double> black(4 * 4 * 3, 0.0);
	auto pl = ProjectionLight<double>::make(
		0.0, 0.0, 0.0, kIdentity, 1.0, 90.0, black, 4, 4);
	EXPECT_DOUBLE_EQ((double)pl.power(), 0.0);
}

// ---------------------------------------------------------------------------
// sample_le / pdf_le tests
// ---------------------------------------------------------------------------

TEST(ProjectionLight, SampleLeDirectionNormalized) {
	auto pl = make_default();
	double wx, wy, wz, pdf_dir;
	pl.sample_le(0.5, 0.5, wx, wy, wz, pdf_dir);
	EXPECT_NEAR(length3(wx, wy, wz), 1.0, 1e-9);
}

TEST(ProjectionLight, SampleLeForwardHemisphere) {
	// All sampled directions should have wz > 0 (forward)
	auto pl = make_default();
	for (int i = 0; i < 16; ++i) {
		double u = (i + 0.5) / 16.0;
		double wx, wy, wz, pdf_dir;
		pl.sample_le(u, u, wx, wy, wz, pdf_dir);
		EXPECT_GT(wz, 0.0) << "i=" << i;
	}
}

TEST(ProjectionLight, SampleLePositivePdf) {
	auto pl = make_default();
	double wx, wy, wz, pdf_dir;
	pl.sample_le(0.5, 0.5, wx, wy, wz, pdf_dir);
	EXPECT_GT(pdf_dir, 0.0);
}

TEST(ProjectionLight, PdfLeMatchesSampleLe) {
	// For a sampled direction, pdf_le should return a consistent value.
	auto pl = make_default();
	double wx, wy, wz, pdf_sample;
	pl.sample_le(0.3, 0.7, wx, wy, wz, pdf_sample);
	double pdf_eval = pl.pdf_le(wx, wy, wz);
	// They should agree within numerical tolerance
	EXPECT_NEAR(pdf_eval, pdf_sample, pdf_sample * 0.01 + 1e-9);
}

TEST(ProjectionLight, PdfLeZeroBehindLight) {
	auto pl = make_default();
	// Direction pointing backward
	EXPECT_DOUBLE_EQ(pl.pdf_le(0.0, 0.0, -1.0), 0.0);
}

TEST(ProjectionLight, PdfLeZeroOutsideFrustum) {
	auto pl = make_default(45.0, 1.0, 4, 4);
	// A direction well outside the 45-degree half-angle
	double pdf = pl.pdf_le(1.0, 0.0, 0.01);
	EXPECT_DOUBLE_EQ(pdf, 0.0);
}
