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
 * Runs its own thin SPPM iteration loop (camera pass -> hash grid build ->
 * photon pass -> radius update, repeated) rather than calling sppm.h's
 * SPPMRender() directly, because SPPMSceneAdapter::BeginIteration() must be
 * called once per iteration before that iteration's camera pass (to clear
 * the shading-context table -- see sppm_adapter.h's BeginIteration() doc
 * comment for why), and sppm.h's own SPPMRender() has no hook for that
 * (it doesn't know about this adapter's internal bookkeeping). This loop
 * is otherwise identical to SPPMRender()'s body.
 */

#include <gtest/gtest.h>
#include "rtweekend.h"
#include "hittable_list.h"
#include "camera.h"
#include "scenes_materials.h"   // build_cornell_rough_glass()
#include "cornell_box_scene.h"  // build_cornell_box_lights()
#include "power_light_sampler.h"
#include "color.h"
#include "sppm_adapter.h"
#include "../../src/shared/sppm.h"

#include <cmath>
#include <fstream>
#include <filesystem>

namespace {

void run_sppm(const SPPMSceneAdapter& scene, int width, int height,
              int nIterations, int nPhotons, int maxDepth, double initialRadius,
              std::vector<double>& out_rgb) {
	int nPixels = width * height;
	std::vector<SPPMPixel<double>> pixels(nPixels);
	for (int i = 0; i < nPixels; ++i) {
		pixels[i].radius = initialRadius;
		pixels[i].px = i % width;
		pixels[i].py = i / width;
	}

	int64_t totalPhotonPaths = 0;
	for (int iter = 0; iter < nIterations; ++iter) {
		scene.BeginIteration();
		SPPMCameraPass(pixels, width, height, maxDepth, scene);

		sppm_detail::HashGrid<double> hashGrid;
		hashGrid.Build(pixels);

		SPPMPhotonPass(pixels, hashGrid, nPhotons, maxDepth, scene);
		totalPhotonPaths += nPhotons;

		SPPMUpdateRadius(pixels);
	}

	SPPMFinalImage(pixels, nIterations, totalPhotonPaths, out_rgb);
}

// Writes a P3 PPM for manual visual inspection. Not asserted on beyond
// "the file was created" -- this is a debugging aid, not a correctness
// check (see the programmatic asserts in the TEST body for that).
void write_ppm(const std::string& path, int width, int height, const std::vector<double>& rgb) {
	std::ofstream out(path);
	out << "P3\n" << width << ' ' << height << "\n255\n";
	for (int i = 0; i < width * height; ++i) {
		color c(rgb[i * 3 + 0], rgb[i * 3 + 1], rgb[i * 3 + 2]);
		write_color(out, c);
	}
}

} // namespace

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
	run_sppm(adapter, cam.image_width, cam.image_height,
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
	write_ppm((out_dir / "cornell_rough_glass_sppm.ppm").string(),
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
	run_sppm(adapter, cam.image_width, cam.image_height,
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
