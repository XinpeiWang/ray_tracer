/**
 * @file spectral_rgb_parity_tests.cpp
 * @brief Verifies --spectral (camera.h's ray_color_spectral()) is
 * RGB-equivalent on the non-dispersive materials it supports.
 *
 * This is a same-integrator comparison (CPU default RGB path vs CPU
 * --spectral path, same scene, same spp), not a cross-backend one like
 * material_cpu_gpu_parity_tests.cpp - both sides run the identical
 * integrator/BSDF code, so on a non-dispersive scene they should converge
 * to the same expected radiance and differ only in noise pattern (RGB math
 * vs a 4-hero-wavelength Monte Carlo estimate). A large, consistent gap
 * here means a real wiring bug (e.g. a missing D65 illuminant
 * normalization - see cie_data.h's GetNormalizedD65Illuminant() comment
 * for the exact real bug class this would reproduce), not something to
 * tolerance-widen away.
 *
 * Tolerance calibration: kRelTolerance below was set from real measured
 * data, not guessed - a manual run of the four positive scenes at this
 * file's resolution/spp measured per-channel gaps under 1% on every
 * channel (see the --spectral wiring commit's own render-verification),
 * so 12% gives real margin while still catching a materially wrong uplift/
 * reconstruction bug.
 */

#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>

#include "scene_registry.h"
#include "../ppm_test_utils.h"

extern "C" {
	#include "cpu_interface.h"
}

// ============================================================================
// Local, static-scoped RGB-average helper - same shape as
// cpu_gpu_comparison_tests.cpp's avg_channels and material_cpu_gpu_parity_
// tests.cpp's mp_avg_channels, but `static` (internal linkage) so this file
// can use the same short name without a duplicate-symbol link error against
// those other translation units in the same test binary - see
// material_cpu_gpu_parity_tests.cpp's own comment on this exact naming
// convention. The PPM loader itself is NOT re-derived here - ppm_test_utils.h
// (PPMImage/load_ppm) already exists as a shared, header-only (inline, so
// collision-free) implementation hoisted for exactly this reuse.
// ============================================================================

namespace {

struct SRRGBAverage { float r, g, b; };

static SRRGBAverage sr_avg_channels(const PPMImage& img) {
	SRRGBAverage out{0, 0, 0};
	int n = img.width * img.height;
	if (n == 0) return out;
	for (int i = 0; i < n; ++i) {
		out.r += img.pixels[i * 3 + 0];
		out.g += img.pixels[i * 3 + 1];
		out.b += img.pixels[i * 3 + 2];
	}
	out.r /= n; out.g /= n; out.b /= n;
	return out;
}

constexpr float kRelTolerance = 0.12f;
// Below this brightness, both values are close enough to black that a
// relative-difference comparison is meaningless (dividing by near-zero
// blows the ratio up on pure noise) - same reasoning/threshold as
// material_cpu_gpu_parity_tests.cpp's own kMinComparableValue.
constexpr float kMinComparableValue = 0.004f;

static void check_relative_parity(const char* sceneName, const char* label,
                                   float rgbVal, float specVal) {
	if (rgbVal < kMinComparableValue && specVal < kMinComparableValue) return;
	float maxV = std::max(rgbVal, specVal);
	float relDiff = std::abs(rgbVal - specVal) / maxV;
	EXPECT_LT(relDiff, kRelTolerance)
		<< sceneName << " " << label << ": RGB=" << rgbVal << " vs spectral=" << specVal
		<< " -- relative difference " << (relDiff * 100.0f) << "% exceeds "
		<< (kRelTolerance * 100.0f) << "% tolerance. --spectral should be RGB-equivalent "
		   "(same expected color, different noise only) on this non-dispersive scene.";
}

constexpr int kWidth  = 60;
constexpr int kHeight = 60;
constexpr int kDepth  = 8;
constexpr int kSpp    = 200;

static PPMImage render_once(const SceneDescriptor& s, bool spectral) {
	const std::string fn = "specparity_" + s.id + (spectral ? "_spec.ppm" : "_rgb.ppm");
	std::remove(fn.c_str());
	cpu_render_main(kWidth, kHeight, kSpp, kDepth, fn.c_str(), s.id.c_str(),
	                 s.camera.lookfrom_x, s.camera.lookfrom_y, s.camera.lookfrom_z,
	                 0, 1.0, nullptr, spectral);
	PPMImage img = load_ppm(fn.c_str());
	std::remove(fn.c_str());
	return img;
}

} // namespace

// ============================================================================
// Positive: RGB vs --spectral should be statistically equivalent
//
// A1 (Cornell Box: lambertian/dielectric/diffuse_light), A2 (Bouncing
// Spheres: lambertian/metal/dielectric, also exercises a bvh_node tree -
// see spectral_scan_hittable()'s own comment in cpu_interface.cpp), B3
// (Cornell Rough Glass: adds rough_dielectric), B4 (Cornell Conductor:
// adds conductor) - together cover all 5 whitelisted BSDF materials plus
// diffuse_light.
// ============================================================================

class SpectralRgbParityTest : public ::testing::TestWithParam<const char*> {};

TEST_P(SpectralRgbParityTest, MatchesDefaultRGBWithinTolerance) {
	const SceneDescriptor* s = find_scene(GetParam());
	ASSERT_NE(s, nullptr);

	PPMImage rgbImg  = render_once(*s, /*spectral=*/false);
	PPMImage specImg = render_once(*s, /*spectral=*/true);

	ASSERT_TRUE(rgbImg.valid)  << s->name << ": default-RGB render failed to produce a valid PPM";
	ASSERT_TRUE(specImg.valid) << s->name << ": --spectral render failed to produce a valid PPM";

	SRRGBAverage rgbC  = sr_avg_channels(rgbImg);
	SRRGBAverage specC = sr_avg_channels(specImg);

	check_relative_parity(s->name, "R channel", rgbC.r, specC.r);
	check_relative_parity(s->name, "G channel", rgbC.g, specC.g);
	check_relative_parity(s->name, "B channel", rgbC.b, specC.b);
}

INSTANTIATE_TEST_SUITE_P(
	SupportedMaterials, SpectralRgbParityTest,
	::testing::Values("A1", "A2", "B3", "B4"),
	[](const ::testing::TestParamInfo<const char*>& info) { return std::string(info.param); });

// ============================================================================
// Negative: a material outside --spectral's whitelist must fail closed
// (a named, non-zero error and no output image), never silently render
// with the wrong color model - see cpu_interface.cpp's
// spectral_scan_hittable() for the actual guard this exercises.
// ============================================================================

TEST(SpectralRgbParityTest, UnsupportedMaterialFailsClosed) {
	// B1 (RoughMetalSpheres) uses rough_metal, which is not in --spectral's
	// {lambertian, metal, dielectric, rough_dielectric, conductor,
	// diffuse_light} whitelist.
	const SceneDescriptor* s = find_scene("B1");
	ASSERT_NE(s, nullptr);

	const std::string fn = "specparity_B1_reject.ppm";
	std::remove(fn.c_str());
	int rc = cpu_render_main(kWidth, kHeight, 8, kDepth, fn.c_str(), s->id.c_str(),
	                          s->camera.lookfrom_x, s->camera.lookfrom_y, s->camera.lookfrom_z,
	                          0, 1.0, nullptr, /*spectral=*/true);

	EXPECT_NE(rc, 0) << "B1 uses rough_metal, not in --spectral's material whitelist - "
	                     "expected a non-zero error, not a silently-wrong render.";
	std::ifstream f(fn);
	EXPECT_FALSE(f.good()) << "B1 should not have produced an output image under --spectral.";
	std::remove(fn.c_str());
}
