#ifndef CORNELL_BOX_SCENE_H
#define CORNELL_BOX_SCENE_H

//==============================================================================================
// Shared Cornell Box Scene Definition
// This file provides a centralized scene builder to ensure CPU and GPU renderers
// produce identical output.
//==============================================================================================

#include "hittable_list.h"
#include "quad.h"
#include "sphere.h"
#include "material.h"
#include "power_light_sampler.h"

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

// Build a power_light_list with power weights computed from geometry + emission.
// Mirrors pbrt-v4 PowerLightSampler: phi = light.Phi() = area * Le_avg * pi
// For a quad area light:  phi = |u x v| * luminance(emission) * pi
// For a sphere geometry target (glass, no emission): phi = pi*r^2 * 1 (geometry weight)
inline power_light_list build_cornell_box_power_lights() {
	power_light_list lights;
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

#endif
