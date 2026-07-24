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
	// Accent wall light
	lights.add(
		make_shared<quad>(point3(554,100,200), vec3(0,0,150), vec3(0,200,0), empty_material));

	return lights;
}

// Build a power_light_list with correct power weights for the Cornell box lights.
// power = area * luminance(emission)
//   - Main ceiling quad: 130*105 area * luminance(15,15,15) = 13650 * 15 = 204750
//   - Glass sphere:      pi*90^2 area * 0 emission => use 1 (geometry sampling target)
//   - Accent wall quad:  150*200 area * luminance(4,2,1) = 30000 * 2.8 = 84000  (approx)
inline power_light_list build_cornell_box_power_lights() {
	power_light_list lights;
	auto empty_material = shared_ptr<material>();

	lights.add(
		make_shared<quad>(point3(343,554,332), vec3(-130,0,0), vec3(0,0,-105), empty_material),
		204750.0);   // bright ceiling light
	lights.add(
		make_shared<sphere>(point3(190, 90, 190), 90, empty_material),
		1000.0);     // glass sphere (geometry target, low power)
	lights.add(
		make_shared<quad>(point3(554,100,200), vec3(0,0,150), vec3(0,200,0), empty_material),
		84000.0);    // dim warm accent light

	return lights;
}

#endif
