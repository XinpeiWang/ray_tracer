/**
 * @file sppm_first_slice_test.cpp
 * @brief First real end-to-end verification of SPPMSceneAdapter (Phase 4 of
 * the SPPM integration plan, see C:\Users\xinpe\.claude\plans\cached-wobbling-ritchie.md).
 *
 * Targets scene 11 (CornellRoughGlass) directly via its existing,
 * unmodified builder functions -- bypassing cpu_interface/CLI entirely at
 * this stage, matching the plan's "cheapest possible real-scene test before
 * investing in CLI/registry polish". Small parameters are used throughout
 * (this phase is about correctness, not quality/convergence).
 *
 * Uses sppm_adapter.h's shared sppm_render_with_adapter()/sppm_write_ppm()
 * helpers (also used by cpu_interface.cpp's --sppm CLI path) rather than
 * duplicating the iteration loop here -- see sppm_render_with_adapter()'s
 * own doc comment for why it needs to call SPPMSceneAdapter::BeginIteration()
 * once per iteration (a detail sppm.h's own generic SPPMRender() can't know
 * about).
 */

#include <gtest/gtest.h>
#include "rtweekend.h"
#include "hittable_list.h"
#include "camera.h"
#include "scenes_materials.h"   // build_cornell_rough_glass()
#include "cornell_box_scene.h"  // build_cornell_box_lights()
#include "scenes_advanced.h"    // build_point_light_cornell()/build_point_light_punct()
#include "power_light_sampler.h"
#include "color.h"
#include "sppm_adapter.h"

#include <cmath>
#include <filesystem>

TEST(SppmFirstSlice, CornellRoughGlassProducesFiniteNonNegativeImage) {
	hittable_list world = build_cornell_rough_glass();
	hittable_list lights_raw = build_cornell_box_lights();
	power_light_list lights(lights_raw);

	camera cam;
	cam.aspect_ratio = 1.0;
	cam.image_width = 64;
	cam.samples_per_pixel = 1;   // unused by SPPM -- camera is only used for get_ray()
	cam.max_depth = 5;           // unused by SPPM -- camera is only used for get_ray()
	cam.vup = vec3(0, 1, 0);
	cam.vfov = 40;
	cam.background = color(0, 0, 0);
	cam.lookfrom = point3(278, 278, -800);
	cam.lookat = point3(278, 278, 278);
	cam.initialize();

	SPPMSceneAdapter adapter(world, lights, cam);

	std::vector<double> out_rgb;
	sppm_render_with_adapter(adapter, cam.image_width, cam.image_height,
	         /*nIterations=*/20, /*nPhotons=*/2000, /*maxDepth=*/5,
	         /*initialRadius=*/10.0, out_rgb);

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
	std::filesystem::path out_dir = std::filesystem::temp_directory_path() / "ray_tracer_sppm_test";
	std::filesystem::create_directories(out_dir);
	sppm_write_ppm((out_dir / "cornell_rough_glass_sppm.ppm").string(),
	          cam.image_width, cam.image_height, out_rgb);
}

TEST(SppmFirstSlice, CenterPixelRegionReceivesLight) {
	// A cheaper, more targeted check than the full-image test above: sample
	// just a handful of pixels known to be inside the lit box interior
	// (roughly the image center, where the rough-glass sphere and back
	// wall are) and confirm they're not all zero -- catches a totally
	// broken light-transport pipeline (e.g. DirectLight/SampleLightLe
	// always returning zero) faster than scanning the whole image.
	hittable_list world = build_cornell_rough_glass();
	hittable_list lights_raw = build_cornell_box_lights();
	power_light_list lights(lights_raw);

	camera cam;
	cam.aspect_ratio = 1.0;
	cam.image_width = 32;
	cam.samples_per_pixel = 1;
	cam.max_depth = 5;
	cam.vup = vec3(0, 1, 0);
	cam.vfov = 40;
	cam.background = color(0, 0, 0);
	cam.lookfrom = point3(278, 278, -800);
	cam.lookat = point3(278, 278, 278);
	cam.initialize();

	SPPMSceneAdapter adapter(world, lights, cam);

	std::vector<double> out_rgb;
	sppm_render_with_adapter(adapter, cam.image_width, cam.image_height,
	         /*nIterations=*/15, /*nPhotons=*/1500, /*maxDepth=*/5,
	         /*initialRadius=*/10.0, out_rgb);

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

// ============================================================================
// Punctual (delta) light support (SPPM Phase 6b)
// ============================================================================
// Scene 27 (Point Light Cornell) has NO area lights at all - build_lights()
// is no_lights (see scene_registry.h) and all illumination comes from
// build_point_light_punct()'s single point light via cam.punct_lights. An
// earlier version of SPPMSceneAdapter::DirectLight() only queried
// nee_lights_ (empty for this scene) and never reached the punctual-light
// block at all, silently rendering scenes like this one pure black under
// --sppm. This test targets that exact scenario directly, matching the
// pattern camera.h's own path tracer already handles correctly.
TEST(SppmFirstSlice, PointLightOnlySceneIsNotPureBlack) {
	hittable_list world = build_point_light_cornell();
	hittable_list lights_raw;   // no_lights: genuinely empty, matching the real registry entry
	power_light_list lights(lights_raw);

	camera cam;
	cam.aspect_ratio = 1.0;
	cam.image_width = 32;
	cam.samples_per_pixel = 1;
	cam.max_depth = 5;
	cam.vup = vec3(0, 1, 0);
	cam.vfov = 40;
	cam.background = color(0, 0, 0);
	cam.lookfrom = point3(278, 278, -800);
	cam.lookat = point3(278, 278, 278);
	cam.punct_lights = build_point_light_punct();   // the fix under test
	cam.initialize();

	SPPMSceneAdapter adapter(world, lights, cam);

	std::vector<double> out_rgb;
	sppm_render_with_adapter(adapter, cam.image_width, cam.image_height,
	         /*nIterations=*/10, /*nPhotons=*/500, /*maxDepth=*/5,
	         /*initialRadius=*/10.0, out_rgb);

	ASSERT_EQ((int)out_rgb.size(), cam.image_width * cam.image_height * 3);
	int nonzero_count = 0;
	for (double v : out_rgb) {
		EXPECT_TRUE(std::isfinite(v));
		EXPECT_GE(v, 0.0);
		if (v > 0.0) ++nonzero_count;
	}
	EXPECT_GT(nonzero_count, 0) << "scene with only a punctual light rendered pure black under --sppm";
}

// ============================================================================
// Sky (infinite) light support
// ============================================================================
// Scene 24 (HDRI Sky) has NO area lights and NO punctual lights - build_lights()
// is no_lights and build_punct is nullptr (see scene_registry.h); all
// illumination comes from build_hdri_sky()'s procedural gradient sky. This
// exercises BOTH halves of sky support: sppm_camera_pass_with_sky()'s
// sky-on-miss handling (for the scene's own metal/glass spheres, which
// escape to the sky via specular bounces) and DirectLight()'s sky NEE
// block (for the ground plane and the diffuse orange sphere, which need
// direct skylight to not render pure black).
TEST(SppmFirstSlice, SkyOnlySceneIsNotPureBlack) {
	hittable_list world = build_hdri_sky_world();
	hittable_list lights_raw;   // no_lights: genuinely empty, matching the real registry entry
	power_light_list lights(lights_raw);

	camera cam;
	cam.aspect_ratio = 1.0;
	cam.image_width = 32;
	cam.samples_per_pixel = 1;
	cam.max_depth = 5;
	cam.vup = vec3(0, 1, 0);
	cam.vfov = 30;
	cam.background = color(0, 0, 0);
	cam.lookfrom = point3(0, 2, 10);
	cam.lookat = point3(0, 1, 0);
	cam.sky = build_hdri_sky();   // the fix under test
	cam.initialize();

	SPPMSceneAdapter adapter(world, lights, cam);

	std::vector<double> out_rgb;
	sppm_render_with_adapter(adapter, cam.image_width, cam.image_height,
	         /*nIterations=*/10, /*nPhotons=*/500, /*maxDepth=*/5,
	         /*initialRadius=*/10.0, out_rgb);

	ASSERT_EQ((int)out_rgb.size(), cam.image_width * cam.image_height * 3);
	int nonzero_count = 0;
	for (double v : out_rgb) {
		EXPECT_TRUE(std::isfinite(v));
		EXPECT_GE(v, 0.0);
		if (v > 0.0) ++nonzero_count;
	}
	EXPECT_GT(nonzero_count, 0) << "scene with only a sky light rendered pure black under --sppm";
}

// Targeted check that ground-plane pixels specifically (a diffuse surface,
// not one of the scene's specular spheres) receive skylight - this is the
// half of sky support that DirectLight()'s NEE block provides and the
// sky-on-miss handling alone would NOT cover (a diffuse hit never reaches
// the miss branch). Samples the bottom row of the image, which the chosen
// camera framing keeps below the spheres' horizon.
TEST(SppmFirstSlice, GroundPlaneReceivesSkylightViaNEE) {
	hittable_list world = build_hdri_sky_world();
	hittable_list lights_raw;
	power_light_list lights(lights_raw);

	camera cam;
	cam.aspect_ratio = 1.0;
	cam.image_width = 32;
	cam.samples_per_pixel = 1;
	cam.max_depth = 5;
	cam.vup = vec3(0, 1, 0);
	cam.vfov = 30;
	cam.background = color(0, 0, 0);
	cam.lookfrom = point3(0, 2, 10);
	cam.lookat = point3(0, 1, 0);
	cam.sky = build_hdri_sky();
	cam.initialize();

	SPPMSceneAdapter adapter(world, lights, cam);

	std::vector<double> out_rgb;
	sppm_render_with_adapter(adapter, cam.image_width, cam.image_height,
	         /*nIterations=*/15, /*nPhotons=*/500, /*maxDepth=*/5,
	         /*initialRadius=*/10.0, out_rgb);

	double bottom_row_sum = 0.0;
	int y = cam.image_height - 1;
	for (int x = 0; x < cam.image_width; ++x) {
		int idx = (y * cam.image_width + x) * 3;
		bottom_row_sum += out_rgb[idx] + out_rgb[idx + 1] + out_rgb[idx + 2];
	}
	EXPECT_GT(bottom_row_sum, 0.0) << "ground plane (diffuse surface) received no skylight";
}

// ============================================================================
// mix_material support
// ============================================================================
// A Cornell box with a mix_material sphere (half lambertian, half mirror
// metal) - end-to-end confirmation that SPPMSceneAdapter::Intersect()'s
// per-hit resolution (sppm_resolve_material(), unit-tested in isolation in
// sppm_adapter_bsdf_tests.cpp) holds up through a real render: the delta
// draws must continue as specular bounces without ever reaching
// DirectLight(), and the non-delta draws must record visible points and
// receive direct lighting normally, all against the SAME sphere/bsdf_id
// across many stochastic per-hit realizations spanning both the camera and
// photon passes.
TEST(SppmFirstSlice, MixMaterialSphereRendersFiniteNonNegativeImage) {
	hittable_list world;
	add_cornell_walls_and_main_light(world);
	auto mat_a = make_shared<lambertian>(color(0.6, 0.2, 0.2));
	auto mat_b = make_shared<metal>(color(0.9, 0.9, 0.9), 0.0);
	auto mix = make_shared<mix_material>(mat_a, mat_b, 0.5);
	world.add(make_shared<sphere>(point3(190, 90, 190), 90, mix));

	hittable_list lights_raw = build_cornell_box_lights();
	power_light_list lights(lights_raw);

	camera cam;
	cam.aspect_ratio = 1.0;
	cam.image_width = 32;
	cam.samples_per_pixel = 1;
	cam.max_depth = 8;
	cam.vup = vec3(0, 1, 0);
	cam.vfov = 40;
	cam.background = color(0, 0, 0);
	cam.lookfrom = point3(278, 278, -800);
	cam.lookat = point3(278, 278, 278);
	cam.initialize();

	SPPMSceneAdapter adapter(world, lights, cam);

	std::vector<double> out_rgb;
	sppm_render_with_adapter(adapter, cam.image_width, cam.image_height,
	         /*nIterations=*/20, /*nPhotons=*/2000, /*maxDepth=*/8,
	         /*initialRadius=*/10.0, out_rgb);

	ASSERT_EQ((int)out_rgb.size(), cam.image_width * cam.image_height * 3);
	int nonzero_count = 0;
	for (double v : out_rgb) {
		EXPECT_TRUE(std::isfinite(v));
		EXPECT_GE(v, 0.0);
		if (v > 0.0) ++nonzero_count;
	}
	EXPECT_GT(nonzero_count, 0) << "mix_material sphere rendered pure black under --sppm";
}

// ============================================================================
// Film "cropwindow"/"pixelbounds"
// ============================================================================
// End-to-end coverage - see tests/integration/bdpt_first_render_test.cpp's
// identical test for why this was added by a code-review pass. Exercises
// sppm_camera_pass_with_sky()'s own crop skip, the third and final distinct
// crop-gating mechanism this feature uses (per-pixel loop skip, same family
// as BDPT/RandomWalk/AO/SimplePath/SimpleVolPath, but on SPPM's own
// iteration-based accumulation - a pixel that's never touched by ANY
// iteration's camera pass must still reconstruct as pure black, not NaN/
// garbage, via SPPMFinalImage()'s own radius-based denominator).
TEST(SppmFirstSlice, CropRestrictsRenderToRightHalf) {
	hittable_list world = build_cornell_rough_glass();
	hittable_list lights_raw = build_cornell_box_lights();
	power_light_list lights(lights_raw);

	camera cam;
	cam.aspect_ratio = 1.0;
	cam.image_width = 32;
	cam.samples_per_pixel = 1;
	cam.max_depth = 5;
	cam.vup = vec3(0, 1, 0);
	cam.vfov = 40;
	cam.background = color(0, 0, 0);
	cam.lookfrom = point3(278, 278, -800);
	cam.lookat = point3(278, 278, 278);
	cam.initialize();

	SPPMSceneAdapter adapter(world, lights, cam);

	std::vector<double> out_rgb;
	const int cropX0 = cam.image_width / 2;
	sppm_render_with_adapter(adapter, cam.image_width, cam.image_height,
	         /*nIterations=*/20, /*nPhotons=*/2000, /*maxDepth=*/5,
	         /*initialRadius=*/10.0, out_rgb,
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
	EXPECT_EQ(left_sum, 0.0) << "pixels outside the crop rect were not left black under SPPM";
	EXPECT_GT(right_sum, 0.0) << "pixels inside the crop rect received no light at all under SPPM";
}
