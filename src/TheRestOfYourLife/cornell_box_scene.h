#ifndef CORNELL_BOX_SCENE_H
#define CORNELL_BOX_SCENE_H

//==============================================================================================
// Standalone Cornell Box Scene Builder (test-only)
// build_cornell_box_scene() below is used only by tests/unit/scene_tests.cpp
// as a lightweight, self-contained scene to exercise geometry/material code
// without going through the scene registry. It is NOT the Cornell Box scene
// the actual renderer builds for scene id "A1" - that's
// scenes_book.h::build_cornell_box(), reached via
// scene_registry.h/scene_builder.cpp (GPU), so this file has no bearing on
// CPU/GPU output parity.
//==============================================================================================

#include "hittable_list.h"
#include "quad.h"
#include "sphere.h"
#include "material.h"
#include "power_light_sampler.h"
#include "bvh_light_sampler.h"
#include "../shared/cornell_box_data.h"

// Build the standard Cornell box scene with a glass sphere and rotated box
inline hittable_list build_cornell_box_scene() {
	hittable_list world;

	// Materials
	auto red   = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light       = make_shared<diffuse_light>(color(15, 15, 15));  // bright white light
	auto light_warm  = make_shared<diffuse_light>(color(4, 2, 1));     // dim warm accent light

	// Cornell box walls
	world.add(make_shared<quad>(point3(555,0,0), vec3(0,0,555), vec3(0,555,0), green));  // right (green)
	world.add(make_shared<quad>(point3(0,0,555), vec3(0,0,-555), vec3(0,555,0), red));   // left (red)
	world.add(make_shared<quad>(point3(0,555,0), vec3(555,0,0), vec3(0,0,555), white));  // ceiling (white)
	world.add(make_shared<quad>(point3(0,0,555), vec3(555,0,0), vec3(0,0,-555), white)); // floor (white)
	world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), white)); // back (white)

	// Main ceiling light (bright)
	world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light));
	// Secondary accent light on the right wall (dim warm)
	world.add(make_shared<quad>(point3(554,100,200), vec3(0,0,150), vec3(0,200,0), light_warm));

	// Rotated box
	shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), white);
	box1 = make_shared<rotate_y>(box1, 15);
	box1 = make_shared<translate>(box1, vec3(265,0,295));
	world.add(box1);

	// Glass sphere
	auto glass = make_shared<dielectric>(1.5);
	world.add(make_shared<sphere>(point3(190,90,190), 90, glass));

	return world;
}

// Build the light sources list for importance sampling (power-weighted)
// Returns hittable_list; wrap in power_light_list at the call site for weighted sampling.
inline hittable_list build_cornell_box_lights() {
	hittable_list lights;
	auto empty_material = shared_ptr<material>();

	// Main ceiling light
	lights.add(
		make_shared<quad>(point3(343,554,332), vec3(-130,0,0), vec3(0,0,-105), empty_material));
	// Glass sphere (acts as a secondary sampled target)
	lights.add(make_shared<sphere>(point3(190, 90, 190), 90, empty_material));

	return lights;
}

// Light list for build_bilinear_patch_scene() (scene F1): same ceiling
// light rectangle as build_cornell_box_lights() (matching geometry: the
// scene's own quad Q=(213,554,227), u=(130,0,0), v=(0,0,105) is the same
// rectangle from the opposite corner), but scene F1 has no glass sphere -
// its "second object" is two bilinear patches, not a sphere - so
// build_cornell_box_lights()'s sphere entry aimed roughly half of every
// NEE sample at a phantom target with no real geometry there, same class
// of bug already fixed for build_cornell_smoke_lights() below.
inline hittable_list build_bilinear_patch_lights() {
	hittable_list lights;
	auto empty_material = shared_ptr<material>();
	lights.add(
		make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), empty_material));
	return lights;
}

// Light list for build_cornell_smoke() (scene 7): that scene's ceiling light
// is its own, much larger rectangle (world quad Q=(113,554,127),
// u=(330,0,0), v=(0,0,305), emission (7,7,7) - see scenes_book.h) rather
// than the standard Cornell-box light build_cornell_box_lights() assumes,
// and scene 7 has neither a glass sphere nor the accent wall light - so
// reusing build_cornell_box_lights() aimed importance sampling at a
// mis-sized light rect plus a phantom sphere target with no real geometry
// there. This gives scene 7 its own correctly-sized single-light list.
inline hittable_list build_cornell_smoke_lights() {
	hittable_list lights;
	auto empty_material = shared_ptr<material>();
	lights.add(
		make_shared<quad>(point3(443,554,432), vec3(-330,0,0), vec3(0,0,-305), empty_material));
	return lights;
}

// Light list for build_cornell_thin_glass() (scene 14): same ceiling light
// rectangle as scene 0, but scene 14 has neither a glass sphere nor the
// accent wall light, so build_cornell_box_lights()'s sphere entry aimed
// samples at a phantom target with no real geometry there.
inline hittable_list build_cornell_thin_glass_lights() {
	hittable_list lights;
	auto empty_material = shared_ptr<material>();
	lights.add(
		make_shared<quad>(point3(343,554,332), vec3(-130,0,0), vec3(0,0,-105), empty_material));
	return lights;
}

// Light list for build_homogeneous_medium_scene() (scene 30): same ceiling
// light rectangle as scene 0 (213,554,227)/(130,0,0)/(0,0,105), but scene
// 30 has neither a glass sphere nor the accent wall light (just fog), so
// build_cornell_box_lights()'s sphere entry aimed roughly half of every
// NEE sample at a phantom target with no real geometry there - a real,
// meaningful contributor to this scene's noise on top of the medium
// boundary fix (see build_homogeneous_medium_scene's comment).
inline hittable_list build_homogeneous_medium_lights() {
	hittable_list lights;
	auto empty_material = shared_ptr<material>();
	lights.add(
		make_shared<quad>(point3(343,554,332), vec3(-130,0,0), vec3(0,0,-105), empty_material));
	return lights;
}

// Education (I8): same Cornell box shell (walls/rotated box/glass sphere,
// via cornell_box_data.h - shared with the real A1/GPU builder, not the
// standalone build_cornell_box_scene() above) as A1, but with the single
// ceiling light replaced by FIVE small quad lights of deliberately
// lopsided power - roughly 1 : 2 : 6 : 15 : 80 - spread around the
// ceiling (one at the original A1 light's spot, four more in the
// otherwise-unlit corners). --lightsampler uniform picks among all five
// with equal 1/5 probability regardless of how much each actually
// contributes, so at low spp it wastes most of its NEE samples on the
// four dim lights while the one genuinely dominant light (the bright
// corner one) is undersampled and noisy; power/bvh weight the selection
// toward that dominant light instead, converging faster on the identical
// scene. CPU only - see --lightsampler's own help text.
inline hittable_list build_light_sampler_comparison() {
	using namespace cornell_box_data;
	hittable_list world;

	// The 5 walls only (kQuads[5] is A1's own single ceiling light -
	// skipped here, replaced by the five below).
	for (int i = 0; i < 5; ++i) {
		const QuadSpec& q = kQuads[i];
		world.add(make_shared<quad>(
			point3(q.Q.x, q.Q.y, q.Q.z),
			vec3(q.u.x, q.u.y, q.u.z),
			vec3(q.v.x, q.v.y, q.v.z),
			make_shared<lambertian>(color(q.color.r, q.color.g, q.color.b))));
	}

	shared_ptr<hittable> box1 = box(
		point3(kBox.corner_min.x, kBox.corner_min.y, kBox.corner_min.z),
		point3(kBox.corner_max.x, kBox.corner_max.y, kBox.corner_max.z),
		make_shared<lambertian>(color(kBox.color.r, kBox.color.g, kBox.color.b)));
	box1 = make_shared<rotate_y>(box1, kBox.rotate_y_degrees);
	box1 = make_shared<translate>(box1, vec3(kBox.translate.x, kBox.translate.y, kBox.translate.z));
	world.add(box1);

	world.add(make_shared<sphere>(
		point3(kGlassSphere.center.x, kGlassSphere.center.y, kGlassSphere.center.z),
		kGlassSphere.radius, make_shared<dielectric>(kGlassSphere.glass_ior)));

	// Five lights, ~1:2:6:15:80 power ratio (quad area is uniform at
	// 40x40, so this ratio is also each one's emission scale directly).
	world.add(make_shared<quad>(point3(30,554,30), vec3(40,0,0), vec3(0,0,40),
		make_shared<diffuse_light>(color(1,1,1))));
	world.add(make_shared<quad>(point3(485,554,30), vec3(40,0,0), vec3(0,0,40),
		make_shared<diffuse_light>(color(2,2,2))));
	world.add(make_shared<quad>(point3(30,554,485), vec3(40,0,0), vec3(0,0,40),
		make_shared<diffuse_light>(color(6,6,6))));
	world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105),
		make_shared<diffuse_light>(color(15,15,15))));
	world.add(make_shared<quad>(point3(485,554,485), vec3(40,0,0), vec3(0,0,40),
		make_shared<diffuse_light>(color(80,80,80))));

	return world;
}

// Light-sampling target list for build_light_sampler_comparison() - the
// five ceiling lights above (as empty-material NEE targets, matching
// build_cornell_box_lights()'s own convention) plus the glass sphere.
inline hittable_list build_light_sampler_comparison_lights() {
	hittable_list lights;
	auto empty_material = shared_ptr<material>();
	lights.add(make_shared<quad>(point3(30,554,30), vec3(40,0,0), vec3(0,0,40), empty_material));
	lights.add(make_shared<quad>(point3(485,554,30), vec3(40,0,0), vec3(0,0,40), empty_material));
	lights.add(make_shared<quad>(point3(30,554,485), vec3(40,0,0), vec3(0,0,40), empty_material));
	lights.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), empty_material));
	lights.add(make_shared<quad>(point3(485,554,485), vec3(40,0,0), vec3(0,0,40), empty_material));
	lights.add(make_shared<sphere>(
		point3(cornell_box_data::kGlassSphere.center.x, cornell_box_data::kGlassSphere.center.y,
			   cornell_box_data::kGlassSphere.center.z),
		cornell_box_data::kGlassSphere.radius, empty_material));
	return lights;
}

// Build a light sampler with power weights computed from geometry + emission.
// Mirrors pbrt-v4 PowerLightSampler: phi = light.Phi() = area * Le_avg * pi
// For a quad area light:  phi = |u x v| * luminance(emission) * pi
// For a sphere geometry target (glass, no emission): phi = pi*r^2 * 1 (geometry weight)
//
// Templated on the sampler type so the same weights populate either
// power_light_list (flat alias-table selection) or bvh_light_sampler
// (bounding-cone BVH selection) - both expose the identical
// add(shared_ptr<hittable>, double phi) interface, so only the return type
// differs between build_cornell_box_power_lights() and
// build_cornell_box_bvh_lights() below.
template <class Sampler>
inline Sampler build_cornell_box_lights_weighted() {
	Sampler lights;
	auto empty_material = shared_ptr<material>();

	// Helper: quad area = |u x v|, luminance from emission color
	auto quad_phi = [](vec3 u, vec3 v, color emission) -> double {
		double area = cross(u, v).length();
		double lum = 0.2126 * emission.x() + 0.7152 * emission.y() + 0.0722 * emission.z();
		return area * lum * pi;  // pbrt-v4: phi = area * Le * pi
	};
	auto sphere_phi = [](double radius) -> double {
		return pi * radius * radius;  // geometry-only target (no emission)
	};

	// Main ceiling light: emission (15,15,15), area = 130*105
	lights.add(
		make_shared<quad>(point3(343,554,332), vec3(-130,0,0), vec3(0,0,-105), empty_material),
		quad_phi(vec3(-130,0,0), vec3(0,0,-105), color(15,15,15)));

	// Glass sphere: geometry sampling target, no emission
	lights.add(
		make_shared<sphere>(point3(190, 90, 190), 90, empty_material),
		sphere_phi(90));

	// Accent wall light: emission (4,2,1), area = 150*200
	lights.add(
		make_shared<quad>(point3(554,100,200), vec3(0,0,150), vec3(0,200,0), empty_material),
		quad_phi(vec3(0,0,150), vec3(0,200,0), color(4,2,1)));

	return lights;
}

inline power_light_list build_cornell_box_power_lights() {
	return build_cornell_box_lights_weighted<power_light_list>();
}

inline bvh_light_sampler build_cornell_box_bvh_lights() {
	return build_cornell_box_lights_weighted<bvh_light_sampler>();
}

#endif
