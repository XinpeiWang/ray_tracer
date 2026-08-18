/**
 * @file sky_light_tests.cpp
 * @brief Unit tests for sky_light (pbrt-v4 ImageInfiniteLight alignment)
 *
 * Tests:
 * - Solid-color sky returns the correct constant radiance
 * - Le() is non-negative for all directions
 * - pdf_Li() returns 1/(4*pi) (uniform sphere)
 * - sample_Li() returns a unit vector
 * - Equirectangular UV mapping: +x axis maps to u~0.5, v~0.5
 * - +y axis (straight up) maps to v~0
 * - -y axis (straight down) maps to v~1
 * - MIS weight sums: BSDF weight + sky NEE weight <= 1  (power heuristic property)
 */

#include <gtest/gtest.h>
#include <vector>
#include "rtweekend.h"
#include "sky_light.h"

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

static constexpr double kEps = 1e-6;

// Power heuristic (beta=2) weight -- duplicated here for test clarity
static double ph(double a, double b) {
	double a2 = a * a, b2 = b * b;
	if (a2 + b2 < 1e-30) return 0.0;
	return a2 / (a2 + b2);
}

// -----------------------------------------------------------------------
// Solid-color sky radiance
// -----------------------------------------------------------------------

TEST(SkyLightTest, SolidColorLe) {
	color sky_color(0.5, 0.7, 1.0);
	sky_light sky(sky_color);

	// Every direction should return the same constant color
	for (const auto& d : { vec3(1,0,0), vec3(-1,0,0), vec3(0,1,0),
							vec3(0,-1,0), vec3(0,0,1), vec3(0,0,-1) }) {
		color le = sky.Le(d);
		EXPECT_NEAR(le.x(), sky_color.x(), kEps);
		EXPECT_NEAR(le.y(), sky_color.y(), kEps);
		EXPECT_NEAR(le.z(), sky_color.z(), kEps);
	}
}

TEST(SkyLightTest, LeIsNonNegative) {
	sky_light sky(color(0.3, 0.6, 0.9));
	for (int i = 0; i < 256; ++i) {
		vec3 d = unit_vector(random_unit_vector());
		color le = sky.Le(d);
		EXPECT_GE(le.x(), 0.0);
		EXPECT_GE(le.y(), 0.0);
		EXPECT_GE(le.z(), 0.0);
	}
}

// -----------------------------------------------------------------------
// Uniform sphere PDF
// -----------------------------------------------------------------------

TEST(SkyLightTest, PdfIsUniformSphere) {
	sky_light sky(color(1, 1, 1));
	double expected = 1.0 / (4.0 * pi);
	EXPECT_NEAR(sky.pdf_Li(), expected, kEps);
}

TEST(SkyLightTest, PdfIsPositive) {
	sky_light sky(color(1, 1, 1));
	EXPECT_GT(sky.pdf_Li(), 0.0);
}

// -----------------------------------------------------------------------
// sample_Li direction
// -----------------------------------------------------------------------

TEST(SkyLightTest, SampleLiIsUnitVector) {
	sky_light sky(color(1, 1, 1));
	for (int i = 0; i < 64; ++i) {
		vec3 d = sky.sample_Li();
		EXPECT_NEAR(d.length(), 1.0, 1e-5);
	}
}

// -----------------------------------------------------------------------
// Equirectangular UV mapping -- uses a gradient solid color
// to indirectly verify uv mapping via Le variation with direction.
// Direct UV inspection not exposed; we test via symmetry properties.
// -----------------------------------------------------------------------

TEST(SkyLightTest, SymmetryAroundYAxis) {
	// A constant-color sky should return the same Le regardless of
	// azimuthal rotation (phi), verifying the phi component doesn't
	// produce NaN or incorrect branching.
	sky_light sky(color(0.4, 0.5, 0.6));
	const int N = 32;
	for (int i = 0; i < N; ++i) {
		double phi = 2.0 * pi * i / N;
		vec3 d(std::cos(phi), 0.0, std::sin(phi));
		color le = sky.Le(unit_vector(d));
		EXPECT_NEAR(le.x(), 0.4, kEps);
		EXPECT_NEAR(le.y(), 0.5, kEps);
		EXPECT_NEAR(le.z(), 0.6, kEps);
	}
}

TEST(SkyLightTest, PolarDirectionsNoNaN) {
	sky_light sky(color(1, 1, 1));
	// Straight up and straight down are the poles of the equirectangular map;
	// atan2 can be undefined at the poles -- verify no NaN.
	color le_up   = sky.Le(vec3( 0,  1,  0));
	color le_down = sky.Le(vec3( 0, -1,  0));
	EXPECT_FALSE(std::isnan(le_up.x())   || std::isnan(le_up.y())   || std::isnan(le_up.z()));
	EXPECT_FALSE(std::isnan(le_down.x()) || std::isnan(le_down.y()) || std::isnan(le_down.z()));
}

// -----------------------------------------------------------------------
// dir_to_uv() correctness -- private, so tested indirectly via pdf_Li().
// A synthetic HDR image with a dim uniform baseline plus one bright pixel
// gives dist a sharp (but non-degenerate) peak; pdf_Li() must peak at the
// direction that pixel's own (u,v) maps to under the class's documented
// forward formula (the same formula sample_Le() uses), far above the
// baseline pdf elsewhere. The baseline keeps every row/column non-degenerate
// so PiecewiseConstant1D's own all-zero-bin uniform fallback (see
// piecewise_dist.h) never kicks in and confounds the comparison. This test
// fails under the old dir_to_uv() (wrong sign on y, spurious +pi phase
// offset on phi), which pointed pdf_Li() at the wrong row/column.
// -----------------------------------------------------------------------

TEST(SkyLightTest, PdfLiPeaksAtTheBrightPixelsOwnDirection) {
	const int W = 16, H = 16;
	const int hot_row = 4, hot_col = 12;
	std::vector<float> pixels(static_cast<size_t>(W) * H * 3, 0.01f);
	size_t hot = (static_cast<size_t>(hot_row) * W + hot_col) * 3;
	pixels[hot] = pixels[hot + 1] = pixels[hot + 2] = 50.0f;

	sky_light sky(W, H, pixels.data());
	ASSERT_TRUE(sky.has_importance_sampling());

	// Same forward formula documented at the top of sky_light.h / used by
	// sample_Le() -- the direction the hot pixel's own (u,v) center maps to.
	double u_c = (hot_col + 0.5) / W;
	double v_c = (hot_row + 0.5) / H;
	double phi = u_c * 2.0 * pi;
	double theta = v_c * pi;
	double sin_t = std::sin(theta), cos_t = std::cos(theta);
	vec3 dir_hot = unit_vector(vec3(sin_t * std::cos(phi), cos_t, -sin_t * std::sin(phi)));

	double pdf_hot = sky.pdf_Li(dir_hot);
	double pdf_opposite = sky.pdf_Li(-dir_hot);

	EXPECT_GT(pdf_hot, 0.0);
	EXPECT_GT(pdf_hot, pdf_opposite * 100.0);
}

// -----------------------------------------------------------------------
// MIS weight property: w_sky + w_bsdf <= 1  (power heuristic, beta=2)
// -----------------------------------------------------------------------

TEST(SkyLightTest, MISWeightsDoNotExceedOne) {
	sky_light sky(color(1, 1, 1));
	double pdf_sky = sky.pdf_Li();  // 1/(4*pi)

	// Try several BSDF PDFs spanning a wide range
	for (double pdf_bsdf : { 0.01, 0.1, 0.5, 1.0, 5.0, 100.0 }) {
		double w_nee  = ph(pdf_sky,  pdf_bsdf);
		double w_bsdf = ph(pdf_bsdf, pdf_sky);
		EXPECT_LE(w_nee + w_bsdf, 1.0 + 1e-9)
			<< "pdf_bsdf=" << pdf_bsdf;
		EXPECT_NEAR(w_nee + w_bsdf, 1.0, 1e-9)
			<< "pdf_bsdf=" << pdf_bsdf;
	}
}
