// gpu_recursive_dispersion_test.cpp
// Real end-to-end verification that GPU-recursive (--gpu, no --wavefront)
// actually disperses light through a dispersive dielectric material -
// closes a coverage gap a code-review pass found: this backend's new
// dispersion support (shade_material()'s Dielectric case, gpu/optix/
// optix_device_helpers.h) had zero automated coverage before this test,
// unlike CPU's own CauchyEta()/dielectric::scatter_dispersive() (tests/unit/
// fresnel_tests.cpp, tests/unit/scatter_record_transmission_tests.cpp) -
// those only exercise code this feature reuses unchanged, not the new
// per-path stochastic-channel-selection mechanism or the payload plumbing
// that carries it across bounces.
//
// The actual bug this backend's dispersion fixed: EVERY pixel in the
// refracted/dispersed region used the same flat, undispersed mat.ior, so a
// white light through the prism stayed a uniform, achromatic (R==G==B)
// grey/white strip - no chromatic fan at all. This test locks in the fix by
// rendering B23 (Glass Prism Dispersion, src/TheRestOfYourLife/
// scene_registry.h) and asserting at least one pixel shows real hue
// separation (max(R,G,B) - min(R,G,B) meaningfully above zero) - a property
// the pre-fix flat-IOR code could never produce, since B23's only light
// source is achromatic white and its walls/background are neutral
// grey/white (see build_prism_dispersion()'s own comment, src/
// TheRestOfYourLife/scenes_materials.h) - the ONLY source of real per-
// channel divergence anywhere in this scene is the dispersive refraction
// itself.
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdio>
#include <string>
#include "../ppm_test_utils.h"

extern "C" {
	#include "optix_interface.h"
}

namespace {

class GpuRecursiveDispersionTest : public ::testing::Test {
  protected:
	void SetUp() override {
		if (!optix_is_available()) {
			GTEST_SKIP() << "OptiX not available on this system";
		}
	}
	void TearDown() override {
		for (const auto& f : outputFiles_) std::remove(f.c_str());
	}
	std::vector<std::string> outputFiles_;
};

} // namespace

TEST_F(GpuRecursiveDispersionTest, GlassPrismShowsRealChromaticSeparation) {
	const char* path = "gpu_recursive_dispersion_test_b23.ppm";
	outputFiles_.push_back(path);

	// force_camera_override=1: B23 has its own registered camera
	// (kPrismCamera, scene_registry.h) - cam_x/y/z below are ignored,
	// matching every other B-series scene test in this codebase.
	int result = optix_render_main(
		/*image_width=*/200, /*image_height=*/100,
		/*samples_per_pixel=*/32, /*max_depth=*/8,
		path, /*scene_id=*/"B23",
		0.0, 0.0, 0.0, /*force_camera_override=*/1);
	ASSERT_EQ(result, 0) << "optix_render_main failed on scene B23 (--gpu recursive)";

	PPMImage img = load_ppm(path);
	ASSERT_TRUE(img.valid) << "Failed to load rendered PPM";
	ASSERT_EQ(img.width, 200);
	ASSERT_EQ(img.height, 100);

	float maxChannelSpread = 0.0f;
	for (int i = 0; i + 2 < static_cast<int>(img.pixels.size()); i += 3) {
		float r = img.pixels[i], g = img.pixels[i + 1], b = img.pixels[i + 2];
		float spread = std::max({r, g, b}) - std::min({r, g, b});
		maxChannelSpread = std::max(maxChannelSpread, spread);
	}

	// A flat/undispersed render of this scene (every backend before this
	// round's own fix, and GPU-recursive specifically before the dispersion
	// commit) never exceeds a few percent here - noise/tonemapping jitter
	// only, no real hue. Real dispersion produces a strong, unmistakable
	// red/orange/yellow fan (see this file's own header comment) - 0.15 is
	// comfortably above noise floor and comfortably below the real effect's
	// actual magnitude (empirically closer to 0.5-0.8 at the fan's core).
	EXPECT_GT(maxChannelSpread, 0.15f)
		<< "No pixel shows real R/G/B divergence - GPU-recursive's dispersive "
		<< "Dielectric may have regressed back to flat/undispersed refraction "
		<< "(max channel spread found: " << maxChannelSpread << ")";
}
