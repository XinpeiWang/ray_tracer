/**
 * @file skip_pdf_material_brightness_tests.cpp
 * @brief Regression coverage for skip_pdf ("specular") materials rendered
 * through the real path tracer (camera::ray_color()), not just isolated
 * BSDF math.
 *
 * Guards against the class of bug fixed in commit b8d355d: scatter_record::
 * eta/is_transmission were left uninitialized by every skip_pdf material
 * except dielectric, which corrupted camera.h's Russian-roulette
 * bookkeeping (eta_scale) and killed most multi-bounce rough_dielectric
 * paths before they reached a wall or light -- scene 11's frosted-glass
 * sphere rendered near-solid-black regardless of sample count. Nothing in
 * the suite previously rendered these materials through ray_color() and
 * checked brightness -- sppm_adapter_bsdf_tests.cpp only covers the
 * isolated BSDF bridge math used by the SPPM integrator, a completely
 * separate code path from camera.h's own scatter()/ray_color() loop.
 *
 * Strategy: fire rays directly from the camera position at each scene's
 * sphere center (point3(190,90,190), shared by the Cornell "Medium"
 * material-showcase scenes 10-13) and average camera::ray_color()'s output
 * over many samples -- the materials' own internal random_double() calls
 * (BSDF lobe selection, roughness sampling, NEE light sampling) already
 * provide the Monte Carlo variance, so no extra ray jitter is needed.
 * Thresholds are set comfortably above what the bug produced (empirically
 * indistinguishable from 0, confirmed via debug instrumentation described
 * in the fix commit) and comfortably below a fully-converged render.
 */

#include <gtest/gtest.h>
#include "rtweekend.h"
#include "hittable_list.h"
#include "camera.h"
#include "scenes_materials.h"   // build_cornell_rough_glass/rough_metal/conductor/coated_diffuse
#include "cornell_box_scene.h"  // build_cornell_box_lights
#include "power_light_sampler.h"
#include "color.h"

namespace {

// Cornell "Medium" material-showcase scenes (10-13) all share this camera
// config -- see scene_registry.h's entries for scenes 10-13.
camera make_showcase_camera() {
	camera cam;
	cam.aspect_ratio = 1.0;
	cam.image_width = 60;
	cam.samples_per_pixel = 1;
	cam.max_depth = 12;
	cam.vup = vec3(0, 1, 0);
	cam.vfov = 40;
	cam.background = color(0, 0, 0);
	cam.lookfrom = point3(278, 278, -800);
	cam.lookat = point3(278, 278, 278);
	cam.initialize();
	return cam;
}

// Normalized screen-space window covering the sphere shared by scenes
// 10-13 (center (190,90,190), radius 90) under make_showcase_camera()'s
// framing -- confirmed empirically while diagnosing the b8d355d bug (see
// this file's top comment). A per-pixel average over this whole window
// -- not a single ray aimed dead-center at the sphere -- is deliberate:
// a ray fired straight at a near-mirror material's front-center point
// reflects almost straight back out through the camera's own (wall-less)
// side of the box, which is degenerate for THAT material regardless of
// any bug. Sweeping across the window instead samples many different
// points/normals on the sphere, including the specular highlight facing
// the ceiling light, giving a physically meaningful brightness signal
// for both diffuse-ish and near-mirror materials alike.
constexpr double kWinX0 = 0.35, kWinX1 = 0.70;
constexpr double kWinY0 = 0.55, kWinY1 = 0.90;

// Average ray_color() over every pixel in the sphere's screen-space
// window, spp samples per pixel. Returns mean Rec.709 luminance.
//
// spp=1 (the original value here) turned out to make this test genuinely
// flaky for coated_diffuse specifically: its LayeredBxDF does an internal
// stochastic random walk (reflect off the coat / transmit-bounce-transmit
// out / absorb), high enough per-sample variance that a single sample per
// pixel occasionally averaged just under the threshold across the whole
// window (observed failing mean: 0.0294, threshold 0.03) even though nothing
// was actually wrong -- confirmed by rerunning this exact test standalone
// and seeing it flip between pass and fail. 4 samples/pixel quarters that
// variance for a very small runtime cost (still well under a second).
double average_sphere_brightness(const hittable_list& world, const hittable& lights,
                                  const camera& cam, int spp = 4) {
	int x0 = static_cast<int>(kWinX0 * cam.image_width);
	int x1 = static_cast<int>(kWinX1 * cam.image_width);
	int y0 = static_cast<int>(kWinY0 * cam.image_height);
	int y1 = static_cast<int>(kWinY1 * cam.image_height);

	double sum = 0.0;
	int count = 0;
	for (int j = y0; j < y1; ++j) {
		for (int i = x0; i < x1; ++i) {
			ray r = cam.get_ray(i, j, 0, 0, vec3(0.0, 0.0, 0.0));
			for (int s = 0; s < spp; ++s) {
				SobolSampler ps(s * (cam.image_width * cam.image_height) + j * cam.image_width + i, i, j);
				color c = cam.ray_color(r, cam.max_depth, world, lights, ps);
				sum += 0.2126 * c.x() + 0.7152 * c.y() + 0.0722 * c.z();
				++count;
			}
		}
	}
	return sum / count;
}

} // namespace

TEST(SkipPdfMaterialBrightness, RoughDielectricSphereIsNotNearBlack) {
	hittable_list world = build_cornell_rough_glass();
	hittable_list lights_raw = build_cornell_box_lights();
	power_light_list lights(lights_raw);
	camera cam = make_showcase_camera();

	double mean_luminance = average_sphere_brightness(world, lights, cam);

	EXPECT_GT(mean_luminance, 0.03)
		<< "rough_dielectric sphere rendered too dark (mean luminance "
		<< mean_luminance << ") -- possible regression of the "
		   "scatter_record::eta/is_transmission uninitialized-field bug "
		   "(commit b8d355d)";
}

TEST(SkipPdfMaterialBrightness, RoughMetalSphereIsNotNearBlack) {
	hittable_list world = build_cornell_rough_metal();
	hittable_list lights_raw = build_cornell_box_lights();
	power_light_list lights(lights_raw);
	camera cam = make_showcase_camera();

	double mean_luminance = average_sphere_brightness(world, lights, cam);

	EXPECT_GT(mean_luminance, 0.03)
		<< "rough_metal sphere rendered too dark (mean luminance " << mean_luminance << ")";
}

TEST(SkipPdfMaterialBrightness, ConductorSphereIsNotNearBlack) {
	hittable_list world = build_cornell_conductor();
	hittable_list lights_raw = build_cornell_box_lights();
	power_light_list lights(lights_raw);
	camera cam = make_showcase_camera();

	double mean_luminance = average_sphere_brightness(world, lights, cam);

	EXPECT_GT(mean_luminance, 0.03)
		<< "conductor sphere rendered too dark (mean luminance " << mean_luminance << ")";
}

TEST(SkipPdfMaterialBrightness, CoatedDiffuseSphereIsNotNearBlack) {
	hittable_list world = build_cornell_coated_diffuse();
	hittable_list lights_raw = build_cornell_box_lights();
	power_light_list lights(lights_raw);
	camera cam = make_showcase_camera();

	double mean_luminance = average_sphere_brightness(world, lights, cam);

	EXPECT_GT(mean_luminance, 0.03)
		<< "coated_diffuse sphere rendered too dark (mean luminance " << mean_luminance << ")";
}
