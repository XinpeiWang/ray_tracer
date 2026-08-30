/**
 * @file bdpt_first_render_test.cpp
 * @brief First real end-to-end verification of BDPTSceneAdapter, mirroring
 * tests/integration/sppm_first_slice_test.cpp's own structure and rationale
 * (see that file's own doc comment) for the --bdpt CLI integration.
 *
 * Targets scene A1 (Cornell Box) directly via its existing, unmodified
 * build_cornell_box() builder -- bypassing cpu_interface/CLI entirely at
 * this stage, matching sppm_first_slice_test.cpp's own "cheapest possible
 * real-scene test before investing in CLI/registry polish" precedent.
 * Small parameters throughout (this is about correctness, not
 * quality/convergence).
 *
 * Uses bdpt_adapter.h's shared bdpt_render_with_adapter()/bdpt_write_ppm()
 * helpers (also used by cpu_renderer/cpu_interface_bdpt.cpp's --bdpt CLI
 * path) rather than duplicating the render loop here.
 */

#include <gtest/gtest.h>
#include "rtweekend.h"
#include "hittable_list.h"
#include "camera.h"
#include "scenes_book.h"   // build_cornell_box()
#include "color.h"
#include "bdpt_adapter.h"

#include <cmath>
#include <filesystem>

namespace {

camera make_cornell_camera(int width) {
	camera cam;
	cam.aspect_ratio = 1.0;
	cam.image_width = width;
	cam.vup = vec3(0, 1, 0);
	cam.vfov = 40;
	cam.background = color(0, 0, 0);
	cam.lookfrom = point3(278, 278, -800);
	cam.lookat = point3(278, 278, 278);
	cam.focus_dist = 800.0;   // matches book Cornell box's own lookfrom/lookat distance -- see camera.h's viewport formula
	cam.initialize();
	return cam;
}

} // namespace

TEST(BdptFirstRender, CornellBoxProducesFiniteNonNegativeImage) {
	hittable_list world = build_cornell_box();
	camera cam = make_cornell_camera(32);

	BDPTSceneAdapter adapter(world, cam);

	std::vector<double> out_rgb;
	bdpt_render_with_adapter(adapter, cam.image_width, cam.image_height,
	                          /*spp=*/4, /*maxDepth=*/5, out_rgb);

	ASSERT_EQ((int)out_rgb.size(), cam.image_width * cam.image_height * 3);

	int nonzero_count = 0;
	for (int i = 0; i < (int)out_rgb.size(); ++i) {
		double v = out_rgb[i];
		EXPECT_TRUE(std::isfinite(v)) << "index " << i;
		EXPECT_GE(v, 0.0) << "index " << i;
		if (v > 0.0) ++nonzero_count;
	}
	// The interior of a lit Cornell box should not render as entirely black.
	EXPECT_GT(nonzero_count, 0);

	// Write a PPM for manual visual inspection (not itself asserted on).
	std::filesystem::path out_dir = std::filesystem::temp_directory_path() / "ray_tracer_bdpt_test";
	std::filesystem::create_directories(out_dir);
	bdpt_write_ppm((out_dir / "cornell_box_bdpt.ppm").string(),
	               cam.image_width, cam.image_height, out_rgb);
}

TEST(BdptFirstRender, CenterPixelRegionReceivesLight) {
	// A cheaper, more targeted check than the full-image test above: sample
	// just a handful of pixels known to be inside the lit box interior
	// (roughly the image center) and confirm they're not all zero -- catches
	// a totally broken light-transport pipeline (e.g. SampleLight/
	// SampleLightLe always returning false, or Unoccluded always false)
	// faster than scanning the whole image.
	hittable_list world = build_cornell_box();
	camera cam = make_cornell_camera(24);

	BDPTSceneAdapter adapter(world, cam);

	std::vector<double> out_rgb;
	bdpt_render_with_adapter(adapter, cam.image_width, cam.image_height,
	                          /*spp=*/6, /*maxDepth=*/5, out_rgb);

	double sum = 0.0;
	int cx = cam.image_width / 2, cy = cam.image_height / 2;
	for (int dy = -3; dy <= 3; ++dy) {
		for (int dx = -3; dx <= 3; ++dx) {
			int x = cx + dx, y = cy + dy;
			if (x < 0 || x >= cam.image_width || y < 0 || y >= cam.image_height) continue;
			int idx = (y * cam.image_width + x) * 3;
			sum += out_rgb[idx] + out_rgb[idx + 1] + out_rgb[idx + 2];
		}
	}
	EXPECT_GT(sum, 0.0) << "center region of a lit Cornell box rendered as pure black";
}

// BDPTSceneAdapter's SampleLightLe()/emitter-power weighting deliberately
// mirrors SPPMSceneAdapter's (see bdpt_adapter.h's own file comment) --
// this test locks in that a scene whose ONLY light source is that single
// area light still lights the box's walls via indirect diffuse bounces
// (the light subpath side of BDPT, not just the s=0/camera-sees-light-
// directly strategy), which is what would break first if SampleLightLe()
// or the light subpath's own BSDF sampling were silently broken.
TEST(BdptFirstRender, WallsReceiveIndirectLight) {
	hittable_list world = build_cornell_box();
	camera cam = make_cornell_camera(24);

	BDPTSceneAdapter adapter(world, cam);

	std::vector<double> out_rgb;
	bdpt_render_with_adapter(adapter, cam.image_width, cam.image_height,
	                          /*spp=*/8, /*maxDepth=*/6, out_rgb);

	// A column of pixels near the left edge of frame views one of the box's
	// side walls (not the ceiling light itself) - should not be pure black
	// even though no light directly hits the camera's line of sight there;
	// any illumination visible must have arrived via an indirect bounce.
	double left_sum = 0.0;
	int x = 2;
	for (int y = 4; y < cam.image_height - 4; ++y) {
		int idx = (y * cam.image_width + x) * 3;
		left_sum += out_rgb[idx] + out_rgb[idx + 1] + out_rgb[idx + 2];
	}
	EXPECT_GT(left_sum, 0.0) << "Cornell box side wall received no indirect light under BDPT";
}

// Film "cropwindow"/"pixelbounds" end-to-end - added by a code-review pass
// that found the whole crop-on-BDPT/MLT/RandomWalk/AO/SimplePath/
// SimpleVolPath/LightPath/SPPM feature had zero automated coverage across
// its entire multi-commit history (only camera::initialize()'s own
// crop_x0/x1/y0/y1 resolution ARITHMETIC was tested, never an integrator's
// actual CONSUMPTION of it). Restricts to the right half of the frame only
// (opposite of this session's own manual scratch-scene verification, purely
// so this test's "left column receives light" assertion above and this
// one's own "left half is black" assertion can't accidentally both pass on
// a crop that silently did nothing). Also exercises BDPT's own t==1
// light-tracing splat path (SplatFilm), not just the per-pixel t>=2 loop -
// see bdpt_render_with_adapter()'s own comment on why that path needed its
// own, separate fix during this same review round.
TEST(BdptFirstRender, CropRestrictsRenderToRightHalf) {
	hittable_list world = build_cornell_box();
	camera cam = make_cornell_camera(32);

	BDPTSceneAdapter adapter(world, cam);

	std::vector<double> out_rgb;
	const int cropX0 = cam.image_width / 2;
	bdpt_render_with_adapter(adapter, cam.image_width, cam.image_height,
	                          /*spp=*/4, /*maxDepth=*/5, out_rgb,
	                          cropX0, cam.image_width, 0, cam.image_height);

	ASSERT_EQ((int)out_rgb.size(), cam.image_width * cam.image_height * 3);

	double left_sum = 0.0, right_sum = 0.0;
	for (int y = 0; y < cam.image_height; ++y) {
		for (int x = 0; x < cam.image_width; ++x) {
			int idx = (y * cam.image_width + x) * 3;
			double px_sum = out_rgb[idx] + out_rgb[idx + 1] + out_rgb[idx + 2];
			if (x < cropX0) left_sum += px_sum; else right_sum += px_sum;
		}
	}
	EXPECT_EQ(left_sum, 0.0) << "pixels outside the crop rect were not left black";
	EXPECT_GT(right_sum, 0.0) << "pixels inside the crop rect received no light at all";
}
