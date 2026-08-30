/**
 * @file mlt_first_render_test.cpp
 * @brief First real end-to-end verification of mlt_render_with_adapter()
 * (src/TheRestOfYourLife/bdpt_adapter.h), mirroring
 * tests/integration/bdpt_first_render_test.cpp's own structure (which in
 * turn mirrors tests/integration/sppm_first_slice_test.cpp) for the --mlt
 * CLI integration.
 *
 * Targets scene A1 (Cornell Box) via build_cornell_box(), same as
 * bdpt_first_render_test.cpp. Small parameters throughout.
 *
 * mlt_render_with_adapter() is depth-stratified (see its own doc comment
 * in bdpt_adapter.h for why a naive single-shared-alias-table design
 * measurably starved every depth except 0 on this exact scene) -- these
 * tests use maxDepth small enough (3) that stratification's "at least one
 * chain per depth" guarantee is cheap to satisfy even on a low-core test
 * machine, while still exercising more than one depth.
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
	cam.focus_dist = 800.0;
	cam.initialize();
	return cam;
}

} // namespace

TEST(MltFirstRender, CornellBoxProducesFiniteNonNegativeImage) {
	hittable_list world = build_cornell_box();
	camera cam = make_cornell_camera(24);

	BDPTSceneAdapter adapter(world, cam);

	std::vector<double> out_rgb;
	mlt_render_with_adapter(adapter, cam.image_width, cam.image_height,
	                         /*nBootstrap=*/500, /*nMutations=*/60000, /*maxDepth=*/3,
	                         /*sigma=*/0.01, /*largeStepProb=*/0.3, out_rgb);

	ASSERT_EQ((int)out_rgb.size(), cam.image_width * cam.image_height * 3);

	int nonzero_count = 0;
	for (int i = 0; i < (int)out_rgb.size(); ++i) {
		double v = out_rgb[i];
		EXPECT_TRUE(std::isfinite(v)) << "index " << i;
		EXPECT_GE(v, 0.0) << "index " << i;
		if (v > 0.0) ++nonzero_count;
	}
	// A near-completely-black image (a handful of nonzero pixels out of
	// hundreds) is exactly the failure mode this test would have caught
	// before mlt_render_with_adapter()'s depth-stratification and
	// density-reconstruction fixes -- see that function's own doc comment.
	// Require a meaningfully large fraction of the image lit, not just
	// "at least one nonzero pixel", to actually pin that regression.
	double nonzero_frac = double(nonzero_count) / double(out_rgb.size());
	EXPECT_GT(nonzero_frac, 0.10) << "MLT image is nearly all black (" << nonzero_count
	                               << "/" << out_rgb.size() << " nonzero channels)";

	std::filesystem::path out_dir = std::filesystem::temp_directory_path() / "ray_tracer_mlt_test";
	std::filesystem::create_directories(out_dir);
	bdpt_write_ppm((out_dir / "cornell_box_mlt.ppm").string(),
	               cam.image_width, cam.image_height, out_rgb);
}

TEST(MltFirstRender, CenterPixelRegionReceivesLight) {
	hittable_list world = build_cornell_box();
	camera cam = make_cornell_camera(20);

	BDPTSceneAdapter adapter(world, cam);

	std::vector<double> out_rgb;
	mlt_render_with_adapter(adapter, cam.image_width, cam.image_height,
	                         /*nBootstrap=*/500, /*nMutations=*/60000, /*maxDepth=*/3,
	                         /*sigma=*/0.01, /*largeStepProb=*/0.3, out_rgb);

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
	EXPECT_GT(sum, 0.0) << "center region of a lit Cornell box rendered as pure black under MLT";
}

// Bootstrap phase alone (mlt_render_with_adapter()'s shared
// MLTBootstrap() call, and the per-depth b_depth[] derivation from it)
// should already produce a positive total normalization constant for a
// scene with a real, reachable light source -- a fast, targeted check of
// just that first stage rather than a full render.
TEST(MltFirstRender, BootstrapFindsNonzeroWeight) {
	hittable_list world = build_cornell_box();
	camera cam = make_cornell_camera(16);

	BDPTSceneAdapter adapter(world, cam);
	std::vector<double> bootstrapWeights;
	double b = MLTBootstrap<double>(/*nBootstrap=*/500, /*maxDepth=*/3,
	                                 /*sigma=*/0.01, /*largeStepProb=*/0.3,
	                                 adapter, bootstrapWeights);

	EXPECT_GT(b, 0.0) << "MLTBootstrap found no light-carrying paths at all in a lit Cornell box";
	EXPECT_TRUE(std::isfinite(b));
}

// Film "cropwindow"/"pixelbounds" end-to-end - see bdpt_first_render_test.cpp's
// identical test for why this coverage was added. MLT has no per-pixel loop
// at all (its own Markov chain visits essentially arbitrary (px,py) via
// mutation, not pixel iteration - see mlt_render_with_adapter()'s own splat
// lambda comment), so this specifically exercises the inline crop check on
// that lambda, a mechanism neither of BDPT's/SPPM's own crop tests reach.
TEST(MltFirstRender, CropRestrictsRenderToRightHalf) {
	hittable_list world = build_cornell_box();
	camera cam = make_cornell_camera(24);

	BDPTSceneAdapter adapter(world, cam);

	std::vector<double> out_rgb;
	const int cropX0 = cam.image_width / 2;
	mlt_render_with_adapter(adapter, cam.image_width, cam.image_height,
	                         /*nBootstrap=*/500, /*nMutations=*/60000, /*maxDepth=*/3,
	                         /*sigma=*/0.01, /*largeStepProb=*/0.3, out_rgb,
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
	EXPECT_EQ(left_sum, 0.0) << "pixels outside the crop rect were not left black under MLT";
	EXPECT_GT(right_sum, 0.0) << "pixels inside the crop rect received no light at all under MLT";
}
