// portal_image_infinite_light_tests.cpp
// Unit tests for PortalImageInfiniteLightData<T>
// Mirrors pbrt-v4 PortalImageInfiniteLight semantics (lights.cpp).
#include <gtest/gtest.h>
#include "../../src/shared/portal_image_infinite_light.h"
#include <cmath>
#include <vector>
#include <array>

static constexpr double PI = 3.14159265358979323846;

// Helper: build a uniform 1x1 (trivial) or NxN equal-area HDR image with constant colour.
static std::vector<float> make_uniform_image(int w, int h, float r, float g, float b) {
	std::vector<float> img(w * h * 3);
	for (int i = 0; i < w * h; ++i) {
		img[i*3  ] = r;
		img[i*3+1] = g;
		img[i*3+2] = b;
	}
	return img;
}

// Portal facing +Z (frame z-axis = +Z, shading points are in the -Z half-space).
// Corner ordering ensures p03 = portal[3]-portal[0] = right = (1,0,0)
// and p01 = portal[1]-portal[0] = up = (0,1,0),
// so Frame::FromXY(p03, p01).z = Cross((1,0,0),(0,1,0)) = (0,0,1) pointing toward origin.
using Vec3d = pil_detail::Vec3<double>;
static std::array<Vec3d,4> make_z_portal(double z = 10.0, double hw = 1.0) {
	return {{
		{-hw, -hw, z},   // portal[0]
		{-hw,  hw, z},   // portal[1]  p01 = up  = (0,1,0)
		{ hw,  hw, z},   // portal[2]  diagonal
		{ hw, -hw, z}    // portal[3]  p03 = right = (1,0,0)
	}};
}

// ---------------------------------------------------------------------------
// Basic construction
// ---------------------------------------------------------------------------
TEST(PortalImageInfiniteLight, ConstructsWithoutCrash) {
	auto img = make_uniform_image(4, 4, 1.f, 1.f, 1.f);
	auto portal = make_z_portal();
	PortalImageInfiniteLightData<double> light(img.data(), 4, 4, 1.0, portal);
	EXPECT_EQ(light.width(), 4);
	EXPECT_EQ(light.height(), 4);
}

TEST(PortalImageInfiniteLight, ScaleAccessor) {
	auto img = make_uniform_image(4, 4, 1.f, 0.f, 0.f);
	auto portal = make_z_portal();
	PortalImageInfiniteLightData<double> light(img.data(), 4, 4, 2.5, portal);
	EXPECT_DOUBLE_EQ(light.scale(), 2.5);
}

// ---------------------------------------------------------------------------
// eval_Le: direction through portal returns positive; direction away returns 0
// ---------------------------------------------------------------------------
TEST(PortalImageInfiniteLight, Le_ThroughPortalIsPositive) {
	auto img = make_uniform_image(8, 8, 1.f, 1.f, 1.f);
	auto portal = make_z_portal(10.0, 1.0);
	PortalImageInfiniteLightData<double> light(img.data(), 8, 8, 1.0, portal);
	// Ray origin at origin, direction straight through the portal (+Z)
	double val = light.eval_Le(0, 0, 0,  0, 0, 1);
	EXPECT_GT(val, 0.0);
}

TEST(PortalImageInfiniteLight, Le_AwayFromPortalIsZero) {
	auto img = make_uniform_image(8, 8, 1.f, 1.f, 1.f);
	auto portal = make_z_portal(10.0, 1.0);
	PortalImageInfiniteLightData<double> light(img.data(), 8, 8, 1.0, portal);
	// Direction pointing away from portal (-Z)
	double val = light.eval_Le(0, 0, 0,  0, 0, -1);
	EXPECT_DOUBLE_EQ(val, 0.0);
}

// ---------------------------------------------------------------------------
// eval_Le_rgb: RGB result proportional to input colours
// ---------------------------------------------------------------------------
TEST(PortalImageInfiniteLight, Le_rgb_ProportionalToInputColour) {
	// Red-only image
	auto img = make_uniform_image(4, 4, 2.f, 0.f, 0.f);
	auto portal = make_z_portal(10.0, 1.0);
	PortalImageInfiniteLightData<double> light(img.data(), 4, 4, 1.0, portal);
	double r, g, b;
	light.eval_Le_rgb(0, 0, 0,  0, 0, 1,  r, g, b);
	// Red channel should dominate, g and b near 0
	EXPECT_GT(r, 0.0);
	EXPECT_NEAR(g, 0.0, 1e-6);
	EXPECT_NEAR(b, 0.0, 1e-6);
}

// ---------------------------------------------------------------------------
// sample_li returns valid direction pointing through the portal
// ---------------------------------------------------------------------------
TEST(PortalImageInfiniteLight, SampleLi_ReturnsTrueForVisiblePortal) {
	auto img = make_uniform_image(8, 8, 1.f, 1.f, 1.f);
	auto portal = make_z_portal(10.0, 1.0);
	PortalImageInfiniteLightData<double> light(img.data(), 8, 8, 1.0, portal);
	double wx, wy, wz, pdf;
	bool ok = light.sample_li(0.5, 0.5,  0.0, 0.0, 0.0,  wx, wy, wz, pdf);
	EXPECT_TRUE(ok);
	EXPECT_GT(pdf, 0.0);
	// Sampled direction should have positive Z (through portal)
	EXPECT_GT(wz, 0.0);
	// Direction should be normalised
	double len = std::sqrt(wx*wx + wy*wy + wz*wz);
	EXPECT_NEAR(len, 1.0, 1e-6);
}

TEST(PortalImageInfiniteLight, SampleLi_ReturnsFalseFromBehindPortal) {
	auto img = make_uniform_image(4, 4, 1.f, 1.f, 1.f);
	// Portal in +Z hemisphere, origin at z=20 (behind the portal from shading point of view)
	auto portal = make_z_portal(10.0, 0.01);  // tiny portal, shading point at z=20 is behind
	PortalImageInfiniteLightData<double> light(img.data(), 4, 4, 1.0, portal);
	double wx, wy, wz, pdf;
	// Shading point behind the portal (z=20 > z=10) - portal behind the point
	bool ok = light.sample_li(0.5, 0.5, 0.0, 0.0, 20.0, wx, wy, wz, pdf);
	// ImageBounds will fail since the portal is behind the shading point
	EXPECT_FALSE(ok);
}

// ---------------------------------------------------------------------------
// pdf_li is consistent with sample_li (same direction -> positive PDF)
// ---------------------------------------------------------------------------
TEST(PortalImageInfiniteLight, PdfLi_PositiveForSampledDirection) {
	auto img = make_uniform_image(8, 8, 1.f, 1.f, 1.f);
	auto portal = make_z_portal(10.0, 1.0);
	PortalImageInfiniteLightData<double> light(img.data(), 8, 8, 1.0, portal);
	double wx, wy, wz, sample_pdf;
	bool ok = light.sample_li(0.3, 0.7,  0.0, 0.0, 0.0, wx, wy, wz, sample_pdf);
	ASSERT_TRUE(ok);
	double pdf = light.pdf_li(0.0, 0.0, 0.0,  wx, wy, wz);
	EXPECT_GT(pdf, 0.0);
	// Both PDFs should be in reasonable agreement (within 50% since nearest-vs-bilinear)
	EXPECT_NEAR(pdf, sample_pdf, 0.5 * sample_pdf + 1e-12);
}

// ---------------------------------------------------------------------------
// pdf_li returns 0 for direction outside portal angular window
// ---------------------------------------------------------------------------
TEST(PortalImageInfiniteLight, PdfLi_ZeroForDirectionAwayFromPortal) {
	auto img = make_uniform_image(8, 8, 1.f, 1.f, 1.f);
	auto portal = make_z_portal(10.0, 1.0);
	PortalImageInfiniteLightData<double> light(img.data(), 8, 8, 1.0, portal);
	// Direction pointing -Z (away from the portal)
	double pdf = light.pdf_li(0.0, 0.0, 0.0,  0.0, 0.0, -1.0);
	EXPECT_DOUBLE_EQ(pdf, 0.0);
}

// ---------------------------------------------------------------------------
// scale factor scales Le proportionally
// ---------------------------------------------------------------------------
TEST(PortalImageInfiniteLight, ScaleFactorAffectsLe) {
	auto img = make_uniform_image(4, 4, 1.f, 1.f, 1.f);
	auto portal = make_z_portal(10.0, 1.0);
	PortalImageInfiniteLightData<double> light1(img.data(), 4, 4, 1.0, portal);
	PortalImageInfiniteLightData<double> light2(img.data(), 4, 4, 3.0, portal);
	double v1 = light1.eval_Le(0,0,0, 0,0,1);
	double v2 = light2.eval_Le(0,0,0, 0,0,1);
	EXPECT_GT(v1, 0.0);
	EXPECT_NEAR(v2, 3.0 * v1, 1e-10);
}

// ---------------------------------------------------------------------------
// power() is positive for non-black image
// ---------------------------------------------------------------------------
TEST(PortalImageInfiniteLight, PowerIsPositive) {
	auto img = make_uniform_image(4, 4, 1.f, 1.f, 1.f);
	auto portal = make_z_portal(10.0, 1.0);
	PortalImageInfiniteLightData<double> light(img.data(), 4, 4, 1.0, portal);
	EXPECT_GT(light.power(), 0.0);
}

TEST(PortalImageInfiniteLight, PowerIsZeroForBlackImage) {
	auto img = make_uniform_image(4, 4, 0.f, 0.f, 0.f);
	auto portal = make_z_portal(10.0, 1.0);
	PortalImageInfiniteLightData<double> light(img.data(), 4, 4, 1.0, portal);
	EXPECT_DOUBLE_EQ(light.power(), 0.0);
}

// ---------------------------------------------------------------------------
// Multiple samples from sample_li all produce directions in the +Z hemisphere
// (consistent with a portal facing +Z)
// ---------------------------------------------------------------------------
TEST(PortalImageInfiniteLight, MultiSample_AllDirectionsThroughPortal) {
	auto img = make_uniform_image(8, 8, 1.f, 1.f, 1.f);
	auto portal = make_z_portal(10.0, 1.0);
	PortalImageInfiniteLightData<double> light(img.data(), 8, 8, 1.0, portal);
	// Test a grid of sample points
	for (int i = 1; i <= 4; ++i) {
		for (int j = 1; j <= 4; ++j) {
			double ru = i / 5.0, rv = j / 5.0;
			double wx, wy, wz, pdf;
			bool ok = light.sample_li(ru, rv, 0.0, 0.0, 0.0, wx, wy, wz, pdf);
			if (ok) {
				EXPECT_GT(pdf, 0.0);
				EXPECT_GT(wz, 0.0) << "ru=" << ru << " rv=" << rv;
			}
		}
	}
}
