#ifndef SCENES_H
#define SCENES_H

/**
 * @file scenes.h
 * @brief Centralized scene library with all available ray tracing scenes
 * 
 * This file provides a collection of pre-built scenes from "Ray Tracing in One Weekend" series.
 * Each scene is self-contained and returns a configured hittable_list ready for rendering.
 */

#include "hittable_list.h"
#include "sphere.h"
#include "quad.h"
#include "material.h"
#include "texture.h"
#include "bvh.h"
#include "constant_medium.h"
#include "hair_material.h"
#include "principled_material.h"
#include "normal_map_materials.h"
#include "../shared/bilinear_patch.h"

//==============================================================================================
// Scene Builder Functions
//==============================================================================================

/**
 * Build Cornell box scene with glass sphere and rotated box
 */
inline hittable_list build_cornell_box() {
	hittable_list world;

	// Materials
	auto red   = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light       = make_shared<diffuse_light>(color(15, 15, 15));  // bright white light
	auto light_warm  = make_shared<diffuse_light>(color(4, 2, 1));     // dim warm accent light

	// Cornell box walls (matching original cornell_box_scene.h)
	world.add(make_shared<quad>(point3(555,0,0), vec3(0,0,555), vec3(0,555,0), green));  // right (green)
	world.add(make_shared<quad>(point3(0,0,555), vec3(0,0,-555), vec3(0,555,0), red));   // left (red)
	world.add(make_shared<quad>(point3(0,555,0), vec3(555,0,0), vec3(0,0,555), white));  // ceiling (white)
	world.add(make_shared<quad>(point3(0,0,555), vec3(555,0,0), vec3(0,0,-555), white)); // floor (white)
	world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), white)); // back (white)

	// Main ceiling light (bright)
	world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light));
	// Secondary accent light on the right wall (dim warm)
	world.add(make_shared<quad>(point3(554,100,200), vec3(0,0,150), vec3(0,200,0), light_warm));

	// Rotated box (white diffuse, not metal)
	shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), white);
	box1 = make_shared<rotate_y>(box1, 15);
	box1 = make_shared<translate>(box1, vec3(265,0,295));
	world.add(box1);

	// Glass sphere
	auto glass = make_shared<dielectric>(1.5);
	world.add(make_shared<sphere>(point3(190,90,190), 90, glass));

	return world;
}

/**
 * Build bouncing spheres scene (Ray Tracing in One Weekend final scene)
 */
inline hittable_list build_bouncing_spheres() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.32, color(.2, .3, .1), color(.9, .9, .9));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			auto choose_mat = random_double();
			point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

			if ((center - point3(4, 0.2, 0)).length() > 0.9) {
				shared_ptr<material> sphere_material;

				if (choose_mat < 0.8) {
					// diffuse
					auto albedo = color::random() * color::random();
					sphere_material = make_shared<lambertian>(albedo);
					auto center2 = center + vec3(0, random_double(0,.5), 0);
					world.add(make_shared<sphere>(center, center2, 0.2, sphere_material));
				} else if (choose_mat < 0.95) {
					// metal
					auto albedo = color::random(0.5, 1);
					auto fuzz = random_double(0, 0.5);
					sphere_material = make_shared<metal>(albedo, fuzz);
					world.add(make_shared<sphere>(center, 0.2, sphere_material));
				} else {
					// glass
					sphere_material = make_shared<dielectric>(1.5);
					world.add(make_shared<sphere>(center, 0.2, sphere_material));
				}
			}
		}
	}

	auto material1 = make_shared<dielectric>(1.5);
	world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

	auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
	world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

	auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
	world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

	// Re-enable BVH for performance
	world = hittable_list(make_shared<bvh_node>(world));

	return world;
}

/**
 * Build checkered spheres scene
 */
inline hittable_list build_checkered_spheres() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.32, color(.2, .3, .1), color(.9, .9, .9));

	world.add(make_shared<sphere>(point3(0,-10, 0), 10, make_shared<lambertian>(checker)));
	world.add(make_shared<sphere>(point3(0, 10, 0), 10, make_shared<lambertian>(checker)));

	return world;
}

/**
 * Build earth globe scene (requires earthmap.jpg)
 */
inline hittable_list build_earth() {
	auto earth_texture = make_shared<image_texture>("earthmap.jpg");
	auto earth_surface = make_shared<lambertian>(earth_texture);
	auto globe = make_shared<sphere>(point3(0,0,0), 2, earth_surface);

	return hittable_list(globe);
}

/**
 * Build Perlin noise spheres scene
 */
inline hittable_list build_perlin_spheres() {
	hittable_list world;

	auto pertext = make_shared<noise_texture>(4);
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(pertext)));
	world.add(make_shared<sphere>(point3(0,2,0), 2, make_shared<lambertian>(pertext)));

	return world;
}

/**
 * Build colored quads scene
 */
inline hittable_list build_quads() {
	hittable_list world;

	// Materials
	auto left_red     = make_shared<lambertian>(color(1.0, 0.2, 0.2));
	auto back_green   = make_shared<lambertian>(color(0.2, 1.0, 0.2));
	auto right_blue   = make_shared<lambertian>(color(0.2, 0.2, 1.0));
	auto upper_orange = make_shared<lambertian>(color(1.0, 0.5, 0.0));
	auto lower_teal   = make_shared<lambertian>(color(0.2, 0.8, 0.8));

	// Quads
	world.add(make_shared<quad>(point3(-3,-2, 5), vec3(0, 0,-4), vec3(0, 4, 0), left_red));
	world.add(make_shared<quad>(point3(-2,-2, 0), vec3(4, 0, 0), vec3(0, 4, 0), back_green));
	world.add(make_shared<quad>(point3( 3,-2, 1), vec3(0, 0, 4), vec3(0, 4, 0), right_blue));
	world.add(make_shared<quad>(point3(-2, 3, 1), vec3(4, 0, 0), vec3(0, 0, 4), upper_orange));
	world.add(make_shared<quad>(point3(-2,-3, 5), vec3(4, 0, 0), vec3(0, 0,-4), lower_teal));

	return world;
}

/**
 * Build simple light scene with Perlin spheres
 */
inline hittable_list build_simple_light() {
	hittable_list world;

	auto pertext = make_shared<noise_texture>(4);
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(pertext)));
	world.add(make_shared<sphere>(point3(0,2,0), 2, make_shared<lambertian>(pertext)));

	auto difflight = make_shared<diffuse_light>(color(4,4,4));
	world.add(make_shared<sphere>(point3(0,7,0), 2, difflight));
	world.add(make_shared<quad>(point3(3,1,-2), vec3(2,0,0), vec3(0,2,0), difflight));

	return world;
}

/**
 * Build Cornell box with smoke/fog
 */
inline hittable_list build_cornell_smoke() {
	hittable_list world;

	auto red   = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light = make_shared<diffuse_light>(color(7, 7, 7));

	world.add(make_shared<quad>(point3(555,0,0), vec3(0,555,0), vec3(0,0,555), green));
	world.add(make_shared<quad>(point3(0,0,0), vec3(0,555,0), vec3(0,0,555), red));
	world.add(make_shared<quad>(point3(113,554,127), vec3(330,0,0), vec3(0,0,305), light));
	world.add(make_shared<quad>(point3(0,555,0), vec3(555,0,0), vec3(0,0,555), white));
	world.add(make_shared<quad>(point3(0,0,0), vec3(555,0,0), vec3(0,0,555), white));
	world.add(make_shared<quad>(point3(0,0,555), vec3(555,0,0), vec3(0,555,0), white));

	shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), white);
	box1 = make_shared<rotate_y>(box1, 15);
	box1 = make_shared<translate>(box1, vec3(265,0,295));

	shared_ptr<hittable> box2 = box(point3(0,0,0), point3(165,165,165), white);
	box2 = make_shared<rotate_y>(box2, -18);
	box2 = make_shared<translate>(box2, vec3(130,0,65));

	world.add(make_shared<constant_medium>(box1, 0.01, color(0,0,0)));
	world.add(make_shared<constant_medium>(box2, 0.01, color(1,1,1)));

	return world;
}

/**
 * Build final complex scene (very computationally expensive!)
 */
inline hittable_list build_final_scene() {
	hittable_list boxes1;
	auto ground = make_shared<lambertian>(color(0.48, 0.83, 0.53));

	int boxes_per_side = 20;
	for (int i = 0; i < boxes_per_side; i++) {
		for (int j = 0; j < boxes_per_side; j++) {
			auto w = 100.0;
			auto x0 = -1000.0 + i*w;
			auto z0 = -1000.0 + j*w;
			auto y0 = 0.0;
			auto x1 = x0 + w;
			auto y1 = random_double(1,101);
			auto z1 = z0 + w;

			boxes1.add(box(point3(x0,y0,z0), point3(x1,y1,z1), ground));
		}
	}

	hittable_list world;
	world.add(make_shared<bvh_node>(boxes1));

	auto light = make_shared<diffuse_light>(color(7, 7, 7));
	world.add(make_shared<quad>(point3(123,554,147), vec3(300,0,0), vec3(0,0,265), light));

	auto center1 = point3(400, 400, 200);
	auto center2 = center1 + vec3(30,0,0);
	auto sphere_material = make_shared<lambertian>(color(0.7, 0.3, 0.1));
	world.add(make_shared<sphere>(center1, center2, 50, sphere_material));

	world.add(make_shared<sphere>(point3(260, 150, 45), 50, make_shared<dielectric>(1.5)));
	world.add(make_shared<sphere>(point3(0, 150, 145), 50, make_shared<metal>(color(0.8, 0.8, 0.9), 1.0)));

	auto boundary = make_shared<sphere>(point3(360,150,145), 70, make_shared<dielectric>(1.5));
	world.add(boundary);
	world.add(make_shared<constant_medium>(boundary, 0.2, color(0.2, 0.4, 0.9)));
	boundary = make_shared<sphere>(point3(0,0,0), 5000, make_shared<dielectric>(1.5));
	world.add(make_shared<constant_medium>(boundary, .0001, color(1,1,1)));

	auto emat = make_shared<lambertian>(make_shared<image_texture>("earthmap.jpg"));
	world.add(make_shared<sphere>(point3(400,200,400), 100, emat));
	auto pertext = make_shared<noise_texture>(0.2);
	world.add(make_shared<sphere>(point3(220,280,300), 80, make_shared<lambertian>(pertext)));

	hittable_list boxes2;
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	int ns = 1000;
	for (int j = 0; j < ns; j++) {
		boxes2.add(make_shared<sphere>(point3::random(0,165), 10, white));
	}

	world.add(make_shared<translate>(
		make_shared<rotate_y>(make_shared<bvh_node>(boxes2), 15),
		vec3(-100,270,395)
	));

	return world;
}

/**
 * Light list for the final scene -- the single area light quad used for NEE.
 */
inline hittable_list build_final_scene_lights() {
	hittable_list lights;
	auto empty_mat = std::shared_ptr<material>();
	// Same geometry as the light quad in build_final_scene()
	lights.add(make_shared<quad>(point3(123,554,147), vec3(300,0,0), vec3(0,0,265), empty_mat));
	return lights;
}

/**
 * Rough Metal Spheres -- GGX roughness progression showcase
 * Five spheres in a row, roughness 0.05 -> 0.2 -> 0.4 -> 0.6 -> 0.8
 * Lit from a large area light above; sky background for ambient fill.
 */
inline hittable_list build_rough_metal_spheres() {
	hittable_list world;

	// Ground plane
	auto ground = make_shared<lambertian>(color(0.2, 0.2, 0.2));
	world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground));

	// Five rough-metal spheres with increasing roughness
	const double roughnesses[] = { 0.05, 0.2, 0.4, 0.6, 0.8 };
	// Gold-ish tint
	auto albedo = color(0.95, 0.85, 0.55);
	for (int i = 0; i < 5; i++) {
		double x = (i - 2) * 2.5;
		world.add(make_shared<sphere>(
			point3(x, 1.0, 0), 1.0,
			make_shared<rough_metal>(albedo, roughnesses[i])
		));
	}

	// Large area light above
	auto light = make_shared<diffuse_light>(color(6, 6, 6));
	world.add(make_shared<quad>(point3(-6, 6, -4), vec3(12, 0, 0), vec3(0, 0, 8), light));

	return world;
}

/**
 * Cornell box with rough metal objects (GGX microfacet showcase)
 * Replaces the white diffuse box with rough aluminum and the glass sphere
 * with a rough gold sphere -- directly shows the GGX BRDF in a familiar scene.
 */
inline hittable_list build_cornell_rough_metal() {
	hittable_list world;

	auto red   = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light = make_shared<diffuse_light>(color(15, 15, 15));

	// Cornell box walls
	world.add(make_shared<quad>(point3(555,0,0),   vec3(0,0,555),  vec3(0,555,0), green));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(0,0,-555), vec3(0,555,0), red));
	world.add(make_shared<quad>(point3(0,555,0),   vec3(555,0,0),  vec3(0,0,555), white));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(555,0,0),  vec3(0,0,-555), white));
	world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), white));

	// Ceiling light
	world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light));

	// Rough aluminum box (roughness 0.15 -- brushed metal look)
	auto alum = make_shared<rough_metal>(color(0.8, 0.85, 0.88), 0.15);
	shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), alum);
	box1 = make_shared<rotate_y>(box1, 15);
	box1 = make_shared<translate>(box1, vec3(265,0,295));
	world.add(box1);

	// Rough gold sphere (roughness 0.3 -- warm brushed gold)
	auto gold = make_shared<rough_metal>(color(0.95, 0.78, 0.28), 0.3);
	world.add(make_shared<sphere>(point3(190, 90, 190), 90, gold));

	return world;
}

/*
 * build_cornell_rough_glass -- scene 11
 * Cornell box with a GGX rough-dielectric sphere (pbrt-v4 RoughDielectricBxDF).
 * Roughness 0.2 gives a "frosted glass" look while still showing refraction.
 */
inline hittable_list build_cornell_rough_glass() {
	hittable_list world;

	auto red   = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light = make_shared<diffuse_light>(color(15, 15, 15));

	// Cornell box walls
	world.add(make_shared<quad>(point3(555,0,0),   vec3(0,0,555),  vec3(0,555,0), green));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(0,0,-555), vec3(0,555,0), red));
	world.add(make_shared<quad>(point3(0,555,0),   vec3(555,0,0),  vec3(0,0,555), white));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(555,0,0),  vec3(0,0,-555), white));
	world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), white));

	// Ceiling light
	world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light));

	// White diffuse box (same as original Cornell box, right side)
	auto box_mat = make_shared<lambertian>(color(.73, .73, .73));
	shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), box_mat);
	box1 = make_shared<rotate_y>(box1, 15);
	box1 = make_shared<translate>(box1, vec3(265,0,295));
	world.add(box1);

	// Rough glass sphere (roughness 0.2 -- frosted glass)
	auto rough_glass = make_shared<rough_dielectric>(1.5, 0.2);
	world.add(make_shared<sphere>(point3(190, 90, 190), 90, rough_glass));

	return world;
}

/*
 * build_cornell_conductor -- scene 12
 * Cornell box with a polished gold sphere and a polished aluminium box.
 * Uses the conductor material (GGX VNDF + per-channel complex Fresnel,
 * mirroring pbrt-v4 ConductorBxDF) instead of the simpler rough_metal.
 */
inline hittable_list build_cornell_conductor() {
	hittable_list world;

	auto red   = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light = make_shared<diffuse_light>(color(15, 15, 15));

	// Cornell box walls
	world.add(make_shared<quad>(point3(555,0,0),   vec3(0,0,555),  vec3(0,555,0), green));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(0,0,-555), vec3(0,555,0), red));
	world.add(make_shared<quad>(point3(0,555,0),   vec3(555,0,0),  vec3(0,0,555), white));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(555,0,0),  vec3(0,0,-555), white));
	world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), white));

	// Ceiling light
	world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light));

	// Polished gold sphere (conductor, roughness 0.1)
	auto gold = make_shared<conductor>(kConductorAu, 0.1);
	world.add(make_shared<sphere>(point3(190, 90, 190), 90, gold));

	// Polished aluminium box (conductor, roughness 0.05)
	auto alum = make_shared<conductor>(kConductorAl, 0.05);
	shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), alum);
	box1 = make_shared<rotate_y>(box1, 15);
	box1 = make_shared<translate>(box1, vec3(265,0,295));
	world.add(box1);

	return world;
}

/*
 * build_cornell_coated_diffuse -- scene 13
 * Cornell box with a coated-diffuse sphere and a coated-diffuse box.
 * Uses the coated_diffuse material (rough dielectric coat + Lambertian base,
 * mirroring pbrt-v4 CoatedDiffuseBxDF) to show the interplay between
 * specular coat reflection and diffuse-coloured transmission.
 */
inline hittable_list build_cornell_coated_diffuse() {
	hittable_list world;

	auto red   = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light = make_shared<diffuse_light>(color(15, 15, 15));

	// Cornell box walls
	world.add(make_shared<quad>(point3(555,0,0),   vec3(0,0,555),  vec3(0,555,0), green));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(0,0,-555), vec3(0,555,0), red));
	world.add(make_shared<quad>(point3(0,555,0),   vec3(555,0,0),  vec3(0,0,555), white));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(555,0,0),  vec3(0,0,-555), white));
	world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), white));

	// Ceiling light
	world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light));

	// Blue coated-diffuse sphere (IOR 1.5, roughness 0.1)
	auto coated_blue = make_shared<coated_diffuse>(color(0.2, 0.3, 0.9), 1.5, 0.1);
	world.add(make_shared<sphere>(point3(190, 90, 190), 90, coated_blue));

	// Red coated-diffuse box (IOR 1.5, roughness 0.2 -- slightly rougher coat)
	auto coated_red = make_shared<coated_diffuse>(color(0.8, 0.1, 0.1), 1.5, 0.2);
	shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), coated_red);
	box1 = make_shared<rotate_y>(box1, 15);
	box1 = make_shared<translate>(box1, vec3(265,0,295));
	world.add(box1);

	return world;
}

/*
 * build_cornell_thin_glass -- scene 14
 * Cornell box with a vertical thin-glass panel in the centre of the box,
 * demonstrating pbrt-v4 ThinDielectricBxDF: zero-thickness glass slab
 * with analytic multi-bounce Fresnel (R_eff = R + T^2*R/(1-R^2)).
 * The panel splits the box -- light refracts straight through (no bending)
 * and reflects specularly, producing subtle caustic-like interplay.
 */
inline hittable_list build_cornell_thin_glass() {
	hittable_list world;

	auto red   = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light = make_shared<diffuse_light>(color(15, 15, 15));

	// Cornell box walls
	world.add(make_shared<quad>(point3(555,0,0),   vec3(0,0,555),  vec3(0,555,0), green));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(0,0,-555), vec3(0,555,0), red));
	world.add(make_shared<quad>(point3(0,555,0),   vec3(555,0,0),  vec3(0,0,555), white));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(555,0,0),  vec3(0,0,-555), white));
	world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), white));

	// Ceiling light
	world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light));

	// White diffuse box (right side)
	auto box_mat = make_shared<lambertian>(color(.73, .73, .73));
	shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), box_mat);
	box1 = make_shared<rotate_y>(box1, 15);
	box1 = make_shared<translate>(box1, vec3(265,0,295));
	world.add(box1);

	// Thin-glass panel (IOR 1.5) -- vertical slab spanning box interior
	auto panel = make_shared<thin_dielectric>(1.5);
	world.add(make_shared<quad>(point3(100, 0, 200), vec3(0, 555, 0), vec3(355, 0, 0), panel));

	return world;
}

/*
 * build_cornell_coated_conductor -- scene 15
 * Cornell box with a lacquered-gold sphere and a lacquered-copper box,
 * demonstrating pbrt-v4 CoatedConductorBxDF: rough dielectric coat over a
 * GGX conductor base with complex Fresnel (FrComplex per RGB channel).
 * The coat adds an achromatic gloss layer over the spectral metal tint.
 */
inline hittable_list build_cornell_coated_conductor() {
	hittable_list world;

	auto red   = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light = make_shared<diffuse_light>(color(15, 15, 15));

	// Cornell box walls
	world.add(make_shared<quad>(point3(555,0,0),   vec3(0,0,555),  vec3(0,555,0), green));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(0,0,-555), vec3(0,555,0), red));
	world.add(make_shared<quad>(point3(0,555,0),   vec3(555,0,0),  vec3(0,0,555), white));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(555,0,0),  vec3(0,0,-555), white));
	world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), white));

	// Ceiling light
	world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light));

	// Lacquered-gold sphere (Au conductor, IOR-1.5 coat, roughness 0.1)
	auto gold_lacquer = make_shared<coated_conductor>(kConductorAu, 1.5, 0.1);
	world.add(make_shared<sphere>(point3(190, 90, 190), 90, gold_lacquer));

	// Lacquered-copper box (Cu conductor, IOR-1.5 coat, roughness 0.2)
	auto copper_lacquer = make_shared<coated_conductor>(kConductorCu, 1.5, 0.2);
	shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), copper_lacquer);
	box1 = make_shared<rotate_y>(box1, 15);
	box1 = make_shared<translate>(box1, vec3(265,0,295));
	world.add(box1);

	return world;
}

/**
 * build_cornell_wax_slab -- scene 16
 * Cornell box with a translucent wax slab and a diffuse box,
 * demonstrating pbrt-v4 DiffuseTransmissionBxDF: wax-like material
 * that scatters light both in the same hemisphere (diffuse reflection)
 * and the opposite hemisphere (diffuse transmission / subsurface approx).
 */
inline hittable_list build_cornell_wax_slab() {
	hittable_list world;

	auto red   = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light = make_shared<diffuse_light>(color(15, 15, 15));

	// Cornell box walls
	world.add(make_shared<quad>(point3(555,0,0),   vec3(0,0,555),  vec3(0,555,0), green));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(0,0,-555), vec3(0,555,0), red));
	world.add(make_shared<quad>(point3(0,555,0),   vec3(555,0,0),  vec3(0,0,555), white));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(555,0,0),  vec3(0,0,-555), white));
	world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), white));

	// Ceiling light
	world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light));

	// Wax sphere (left): warm ivory wax color -- more transmittance than reflectance
	// R (reflectance) = warm ivory, T (transmittance) = warm amber
	auto wax = make_shared<diffuse_transmission>(
		color(0.6, 0.5, 0.3),   // R: reflected diffuse color
		color(0.8, 0.6, 0.3));  // T: transmitted diffuse color
	world.add(make_shared<sphere>(point3(190, 90, 190), 90, wax));

	// White diffuse box (right)
	shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), white);
	box1 = make_shared<rotate_y>(box1, 15);
	box1 = make_shared<translate>(box1, vec3(265,0,295));
	world.add(box1);

	return world;
}

/**
 * build_cornell_crystal -- scene 17
 * Cornell box with a crystal sphere demonstrating pbrt-v4 NormalizedFresnelBxDF:
 * Fresnel-weighted diffuse reflection -- light exits more at grazing angles
 * (low Fresnel reflection = high BSDF value at grazing).
 * IOR 1.5 (typical glass/crystal).
 */
inline hittable_list build_cornell_crystal() {
	hittable_list world;

	auto red   = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light = make_shared<diffuse_light>(color(15, 15, 15));

	// Cornell box walls
	world.add(make_shared<quad>(point3(555,0,0),   vec3(0,0,555),  vec3(0,555,0), green));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(0,0,-555), vec3(0,555,0), red));
	world.add(make_shared<quad>(point3(0,555,0),   vec3(555,0,0),  vec3(0,0,555), white));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(555,0,0),  vec3(0,0,-555), white));
	world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), white));

	// Ceiling light
	world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light));

	// Crystal sphere (left): NormalizedFresnelBxDF with IOR 1.5
	auto crystal = make_shared<normalized_fresnel>(1.5);
	world.add(make_shared<sphere>(point3(190, 90, 190), 90, crystal));

	// White diffuse box (right)
	shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), white);
	box1 = make_shared<rotate_y>(box1, 15);
	box1 = make_shared<translate>(box1, vec3(265,0,295));
	world.add(box1);

	return world;
}

//==============================================================================================
// Scene 18: Principled Showcase
// A row of 7 spheres demonstrating the principled BSDF parameter space:
//   matte diffuse -> plastic -> semi-metallic -> fully metallic -> clearcoated metal
//==============================================================================================
inline hittable_list build_principled_showcase() {
	hittable_list world;

	// Ground plane (subtle dark checker)
	auto checker = make_shared<checker_texture>(0.5, color(0.1, 0.1, 0.12), color(0.2, 0.2, 0.22));
	world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, make_shared<lambertian>(checker)));

	// Row of 7 spheres: metallic 0->1, roughness varies, clearcoat on last two
	//  0: pure matte diffuse (red)
	world.add(make_shared<sphere>(point3(-3, 1, 0), 1.0,
		make_shared<principled>(color(0.8, 0.1, 0.1), 0.0, 0.9, 1.5, 0.0)));
	//  1: plastic, low roughness (blue)
	world.add(make_shared<sphere>(point3(-2, 1, 0), 1.0,
		make_shared<principled>(color(0.1, 0.2, 0.8), 0.0, 0.2, 1.5, 0.0)));
	//  2: plastic, clearcoated (green)
	world.add(make_shared<sphere>(point3(-1, 1, 0), 1.0,
		make_shared<principled>(color(0.1, 0.7, 0.2), 0.0, 0.3, 1.5, 1.0, 0.05)));
	//  3: semi-metallic (gold-tinted)
	world.add(make_shared<sphere>(point3(0, 1, 0), 1.0,
		make_shared<principled>(color(0.9, 0.7, 0.2), 0.5, 0.3, 1.5, 0.0)));
	//  4: near-metallic, rough (copper-ish)
	world.add(make_shared<sphere>(point3(1, 1, 0), 1.0,
		make_shared<principled>(color(0.8, 0.45, 0.2), 0.8, 0.4, 1.5, 0.0)));
	//  5: fully metallic, smooth (silver)
	world.add(make_shared<sphere>(point3(2, 1, 0), 1.0,
		make_shared<principled>(color(0.9, 0.9, 0.9), 1.0, 0.05, 1.5, 0.0)));
	//  6: fully metallic, clearcoated (lacquered gold)
	world.add(make_shared<sphere>(point3(3, 1, 0), 1.0,
		make_shared<principled>(color(0.9, 0.7, 0.1), 1.0, 0.1, 1.5, 1.0, 0.08)));

	return world;
}

//==============================================================================================
// Scene 19: Hair Fibers
// A cluster of spheres using the hair_material to mimic fur/fiber appearance.
// Each sphere uses slightly different hair parameters (color, roughness) to show variety.
//==============================================================================================
inline hittable_list build_hair_fibers() {
	hittable_list world;

	// Dark floor
	world.add(make_shared<sphere>(point3(0, -1000, 0), 1000,
		make_shared<lambertian>(color(0.05, 0.05, 0.06))));

	// Dark brown hair (default sigma_a)
	world.add(make_shared<sphere>(point3(-2.5, 1, 0), 1.0,
		make_shared<hair_material>(0.06, 0.10, 0.20, 0.25, 0.25, 2.0)));
	// Blonde hair (low absorption, warm tint)
	world.add(make_shared<sphere>(point3(-0.8, 1, 0.3), 1.0,
		make_shared<hair_material>(0.01, 0.015, 0.03, 0.30, 0.30, 2.0)));
	// Auburn hair (strong red absorption pattern)
	world.add(make_shared<sphere>(point3(0.9, 1, -0.3), 1.0,
		make_shared<hair_material>(0.02, 0.08, 0.18, 0.20, 0.20, 3.0)));
	// White/silver fur (very low absorption, rough)
	world.add(make_shared<sphere>(point3(2.5, 1, 0), 1.0,
		make_shared<hair_material>(0.001, 0.001, 0.002, 0.45, 0.45, 1.0)));
	// Fine black fur (very high absorption)
	world.add(make_shared<sphere>(point3(0, 1, 1.8), 1.0,
		make_shared<hair_material>(0.50, 0.55, 0.60, 0.15, 0.15, 2.0)));

	return world;
}

//==============================================================================================
// Scene 20: Normal Mapped Cornell Box
// Cornell box where the back wall has a bump-mapped wavy displacement and the
// sphere has a procedural normal-map perturbation, showing pbrt-v4 NormalMap/BumpMap.
//==============================================================================================
inline hittable_list build_normal_mapped_cornell() {
	hittable_list world;

	auto red   = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light = make_shared<diffuse_light>(color(15, 15, 15));

	// Cornell box walls (plain)
	world.add(make_shared<quad>(point3(555,0,0),   vec3(0,0,555),  vec3(0,555,0), green));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(0,0,-555), vec3(0,555,0), red));
	world.add(make_shared<quad>(point3(0,555,0),   vec3(555,0,0),  vec3(0,0,555), white));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(555,0,0),  vec3(0,0,-555), white));
	world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light));

	// Back wall: bump-mapped wavy surface (uses marble Perlin noise as displacement)
	auto marble_tex = make_shared<noise_texture>(4.0);
	auto white_base = make_shared<lambertian>(color(.73, .73, .73));
	auto bumped_back = make_shared<bump_map_material>(marble_tex, white_base, 0.08, 0.002);
	world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), bumped_back));

	// Floor: plain white
	world.add(make_shared<quad>(point3(0,0,555), vec3(555,0,0), vec3(0,0,-555), white));

	// Sphere: normal-map perturbed lambertian (checker pattern as normal source)
	// The checker drives subtle low-frequency normal variation over the sphere surface
	auto norm_tex = make_shared<checker_texture>(8.0, color(0.5, 0.5, 1.0), color(0.8, 0.8, 1.0));
	auto blue_base = make_shared<lambertian>(color(0.2, 0.3, 0.8));
	auto normal_sphere_mat = make_shared<normal_map_material>(norm_tex, blue_base);
	world.add(make_shared<sphere>(point3(190, 90, 190), 90, normal_sphere_mat));

	// Rotated box: bump-mapped white
	auto bumped_box = make_shared<bump_map_material>(marble_tex, white_base, 0.05, 0.002);
	shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), bumped_box);
	box1 = make_shared<rotate_y>(box1, 15);
	box1 = make_shared<translate>(box1, vec3(265,0,295));
	world.add(box1);

	return world;
}

//==============================================================================================
// Scene 21: Subsurface Slab
// Cornell box with layered constant_medium objects that approximate subsurface
// scattering: a translucent milk-white slab and a jade-green sphere.
// True BSSRDF would require a separate path-length sampling loop;
// here we use constant_medium (homogeneous participating media) to achieve
// a similar translucent glow that shows light scattering inside the object.
//==============================================================================================
inline hittable_list build_subsurface_slab() {
	hittable_list world;

	auto red   = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light = make_shared<diffuse_light>(color(12, 12, 12));

	// Cornell box walls
	world.add(make_shared<quad>(point3(555,0,0),   vec3(0,0,555),  vec3(0,555,0), green));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(0,0,-555), vec3(0,555,0), red));
	world.add(make_shared<quad>(point3(0,555,0),   vec3(555,0,0),  vec3(0,0,555), white));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(555,0,0),  vec3(0,0,-555), white));
	world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), white));
	world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light));

	// Translucent wax slab: opaque glass boundary + dense scattering interior
	// Outer shell is dielectric (lets light in/out), interior is constant_medium (milky scattering)
	auto glass = make_shared<dielectric>(1.4);
	shared_ptr<hittable> slab = box(point3(0,0,0), point3(200,300,160), glass);
	slab = make_shared<translate>(slab, vec3(270,0,230));
	world.add(slab);
	// Volumetric fill inside the slab (milky white, moderate density)
	shared_ptr<hittable> slab_vol = box(point3(0,0,0), point3(200,300,160), glass);
	slab_vol = make_shared<translate>(slab_vol, vec3(270,0,230));
	world.add(make_shared<constant_medium>(slab_vol, 0.04, color(0.98, 0.96, 0.90)));

	// Jade sphere: glass shell + green scattering interior
	auto jade_glass = make_shared<dielectric>(1.5);
	world.add(make_shared<sphere>(point3(160, 90, 160), 90, jade_glass));
	world.add(make_shared<constant_medium>(
		make_shared<sphere>(point3(160, 90, 160), 90, jade_glass),
		0.06, color(0.1, 0.5, 0.2)));

	return world;
}

//==============================================================================================
// Scene 22: Depth of Field
// An open-air scene with spheres at varying depths, rendered with defocus blur.
// The camera is focused at distance 3.5 with a wide aperture to exaggerate DOF.
// Camera setup (defocus_angle, focus_dist) is applied in cpu_interface via CameraConfig.
//==============================================================================================
inline hittable_list build_depth_of_field() {
	hittable_list world;

	// Checker ground
	auto checker = make_shared<checker_texture>(0.5, color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9));
	world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, make_shared<lambertian>(checker)));

	// Background sphere (out of focus far)
	world.add(make_shared<sphere>(point3(-4, 1, -3), 1.0,
		make_shared<lambertian>(color(0.4, 0.2, 0.1))));
	// Near sphere (out of focus near)
	world.add(make_shared<sphere>(point3(-1.5, 0.5, 1.5), 0.5,
		make_shared<lambertian>(color(0.7, 0.3, 0.3))));
	// In-focus center sphere (glass)
	world.add(make_shared<sphere>(point3(0, 1, 0), 1.0,
		make_shared<dielectric>(1.5)));
	// In-focus metal sphere
	world.add(make_shared<sphere>(point3(2, 1, 0), 1.0,
		make_shared<metal>(color(0.7, 0.6, 0.5), 0.05)));
	// Far sphere (out of focus)
	world.add(make_shared<sphere>(point3(4, 1, -2), 1.0,
		make_shared<lambertian>(color(0.1, 0.2, 0.6))));
	// Small near spheres
	for (int i = -3; i <= 3; ++i) {
		world.add(make_shared<sphere>(point3(i * 1.2, 0.2, 2.5 + i * 0.3), 0.2,
			make_shared<lambertian>(color(random_double(0.3,0.9), random_double(0.3,0.9), random_double(0.3,0.9)))));
	}

	return world;
}

//==============================================================================================
// Scene 23: Bilinear Patch
// Cornell box containing a curved bilinear patch saddle surface demonstrating
// pbrt-v4 BilinearPatch intersection (non-planar quadrilateral).
// We wrap BilinearPatchShape<double> in a custom hittable adapter.
//==============================================================================================

// Inline hittable wrapper for BilinearPatchShape<double>
class bilinear_patch_hittable : public hittable {
public:
	bilinear_patch_hittable(
		point3 p00, point3 p10, point3 p01, point3 p11,
		shared_ptr<material> mat)
		: mat(mat)
	{
		shape.p00x = p00.x(); shape.p00y = p00.y(); shape.p00z = p00.z();
		shape.p10x = p10.x(); shape.p10y = p10.y(); shape.p10z = p10.z();
		shape.p01x = p01.x(); shape.p01y = p01.y(); shape.p01z = p01.z();
		shape.p11x = p11.x(); shape.p11y = p11.y(); shape.p11z = p11.z();

		// Compute bounding box from the 4 corners (conservative)
		double xmin = std::min({p00.x(),p10.x(),p01.x(),p11.x()});
		double xmax = std::max({p00.x(),p10.x(),p01.x(),p11.x()});
		double ymin = std::min({p00.y(),p10.y(),p01.y(),p11.y()});
		double ymax = std::max({p00.y(),p10.y(),p01.y(),p11.y()});
		double zmin = std::min({p00.z(),p10.z(),p01.z(),p11.z()});
		double zmax = std::max({p00.z(),p10.z(),p01.z(),p11.z()});
		bbox = aabb(interval(xmin-0.01,xmax+0.01), interval(ymin-0.01,ymax+0.01), interval(zmin-0.01,zmax+0.01));
	}

	bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
		double ro[3] = { r.origin().x(), r.origin().y(), r.origin().z() };
		double rd[3] = { r.direction().x(), r.direction().y(), r.direction().z() };
		auto h = shape.intersect(ro[0],ro[1],ro[2], rd[0],rd[1],rd[2],
			ray_t.min, ray_t.max);
		if (!h) return false;
		rec.t = h->t;
		rec.p = r.at(rec.t);
		vec3 outward_normal(h->nx, h->ny, h->nz);
		rec.set_face_normal(r, outward_normal);
		rec.u = h->u; rec.v = h->v;
		rec.mat = mat;
		// dpdu: approximate tangent along u direction
		double px,py,pz, dx,dy,dz, ex,ey,ez;
		float fro[3]={(float)ro[0],(float)ro[1],(float)ro[2]};
		float frd[3]={(float)rd[0],(float)rd[1],(float)rd[2]};
		float sq00[3]={(float)shape.p00x,(float)shape.p00y,(float)shape.p00z};
		float sq10[3]={(float)shape.p10x,(float)shape.p10y,(float)shape.p10z};
		float sq01[3]={(float)shape.p01x,(float)shape.p01y,(float)shape.p01z};
		float sq11[3]={(float)shape.p11x,(float)shape.p11y,(float)shape.p11z};
		float out_p[3], out_dpdu[3], out_dpdv[3];
		blp_point(sq00,sq10,sq01,sq11,(float)h->u,(float)h->v, out_p,out_dpdu,out_dpdv);
		rec.dpdu = vec3(out_dpdu[0], out_dpdu[1], out_dpdu[2]);
		return true;
	}

	aabb bounding_box() const override { return bbox; }

private:
	BilinearPatchShape<double> shape;
	shared_ptr<material> mat;
	aabb bbox;
};

inline hittable_list build_bilinear_patch_scene() {
	hittable_list world;

	auto red   = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light = make_shared<diffuse_light>(color(15, 15, 15));

	// Cornell box walls
	world.add(make_shared<quad>(point3(555,0,0),   vec3(0,0,555),  vec3(0,555,0), green));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(0,0,-555), vec3(0,555,0), red));
	world.add(make_shared<quad>(point3(0,555,0),   vec3(555,0,0),  vec3(0,0,555), white));
	world.add(make_shared<quad>(point3(0,0,555),   vec3(555,0,0),  vec3(0,0,-555), white));
	world.add(make_shared<quad>(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), white));
	world.add(make_shared<quad>(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light));

	// Bilinear patch saddle surface: four corners with different Y heights
	// p00=bottom-left, p10=bottom-right, p01=top-left, p11=top-right
	// Saddle: p00 and p11 high, p10 and p01 low (classic hyperbolic paraboloid)
	auto patch_mat = make_shared<metal>(color(0.8, 0.7, 0.3), 0.05);
	world.add(make_shared<bilinear_patch_hittable>(
		point3(150,  80, 200),   // p00 (u=0,v=0) -- higher
		point3(400,  50, 200),   // p10 (u=1,v=0) -- lower
		point3(150,  50, 400),   // p01 (u=0,v=1) -- lower
		point3(400,  80, 400),   // p11 (u=1,v=1) -- higher
		patch_mat
	));

	// Second patch: curved ramp (linear in u, curved in v)
	auto blue_mat = make_shared<metal>(color(0.2, 0.4, 0.8), 0.1);
	world.add(make_shared<bilinear_patch_hittable>(
		point3(200, 200, 220),   // p00
		point3(370, 200, 220),   // p10
		point3(150, 380, 420),   // p01
		point3(420, 320, 420),   // p11
		blue_mat
	));

	return world;
}

#endif // SCENES_H
