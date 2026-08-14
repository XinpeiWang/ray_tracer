#pragma once
// scenes_book.h -- Book series scenes (IDs 0-9)
// Included by scenes.h umbrella header.

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
#include "../shared/cornell_box_data.h"

//==============================================================================================
// Scene Builder Functions
//==============================================================================================

/**
 * Adds the 5 standard Cornell-box walls (red/white/green/white/white) and the
 * main ceiling light - all of cornell_box_data::kQuads - to `world`. Shared
 * by every "Cornell family" scene (10-17 in scenes_materials.h) that keeps
 * the standard box shell but swaps in different sphere/box materials, so
 * those scenes stop each hand-typing the same 5 walls + light quad.
 */
inline void add_cornell_walls_and_main_light(hittable_list& world) {
	using namespace cornell_box_data;
	for (int i = 0; i < 6; ++i) {
		const QuadSpec& q = kQuads[i];
		shared_ptr<material> mat = q.is_light
			? shared_ptr<material>(make_shared<diffuse_light>(color(q.color.r, q.color.g, q.color.b)))
			: shared_ptr<material>(make_shared<lambertian>(color(q.color.r, q.color.g, q.color.b)));
		world.add(make_shared<quad>(
			point3(q.Q.x, q.Q.y, q.Q.z),
			vec3(q.u.x, q.u.y, q.u.z),
			vec3(q.v.x, q.v.y, q.v.z),
			mat));
	}
}

/**
 * Build Cornell box scene with glass sphere and rotated box.
 * Geometry/material data comes from src/shared/cornell_box_data.h, shared
 * with the GPU builder (gpu/optix/scene_builder.cpp::build_cornell_box())
 * so the two can't silently drift apart - see that header's comment.
 */
inline hittable_list build_cornell_box() {
	using namespace cornell_box_data;
	hittable_list world;

	for (const auto& q : kQuads) {
		shared_ptr<material> mat = q.is_light
			? shared_ptr<material>(make_shared<diffuse_light>(color(q.color.r, q.color.g, q.color.b)))
			: shared_ptr<material>(make_shared<lambertian>(color(q.color.r, q.color.g, q.color.b)));
		world.add(make_shared<quad>(
			point3(q.Q.x, q.Q.y, q.Q.z),
			vec3(q.u.x, q.u.y, q.u.z),
			vec3(q.v.x, q.v.y, q.v.z),
			mat));
	}

	// Rotated box (white diffuse, not metal)
	auto box_mat = make_shared<lambertian>(color(kBox.color.r, kBox.color.g, kBox.color.b));
	shared_ptr<hittable> box1 = box(
		point3(kBox.corner_min.x, kBox.corner_min.y, kBox.corner_min.z),
		point3(kBox.corner_max.x, kBox.corner_max.y, kBox.corner_max.z),
		box_mat);
	box1 = make_shared<rotate_y>(box1, kBox.rotate_y_degrees);
	box1 = make_shared<translate>(box1, vec3(kBox.translate.x, kBox.translate.y, kBox.translate.z));
	world.add(box1);

	// Glass sphere
	auto glass = make_shared<dielectric>(kGlassSphere.glass_ior);
	world.add(make_shared<sphere>(
		point3(kGlassSphere.center.x, kGlassSphere.center.y, kGlassSphere.center.z),
		kGlassSphere.radius, glass));

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

	// Small accent spheres resting on the visible cap of the lower "planet"
	// (radius 10, centered (0,-10,0), so its near-camera pole sits right
	// around y=0) - scale/depth reference and material variety for what was
	// otherwise just 2 bare checker spheres with nothing to catch light or
	// frame against.
	world.add(make_shared<sphere>(point3(1.6, 0.5, 2.2), 0.9,
		make_shared<lambertian>(color(0.55, 0.15, 0.10))));
	world.add(make_shared<sphere>(point3(-1.4, 0.45, 1.6), 0.7,
		make_shared<metal>(color(0.8, 0.75, 0.6), 0.05)));
	world.add(make_shared<sphere>(point3(0.1, 0.15, 3.0), 0.6,
		make_shared<dielectric>(1.5)));

	return world;
}

/**
 * Build earth globe scene (requires earthmap.jpg)
 */
inline hittable_list build_earth() {
	hittable_list world;

	auto earth_texture = make_shared<image_texture>("earthmap.jpg");
	auto earth_surface = make_shared<lambertian>(earth_texture);
	world.add(make_shared<sphere>(point3(0,0,0), 2, earth_surface));

	// Small grey "moon" for scale/context - the globe used to float alone
	// with nothing to read its size against.
	world.add(make_shared<sphere>(point3(2.0, 1.3, 0.5), 0.35,
		make_shared<lambertian>(color(0.6, 0.6, 0.62))));

	// Dim cool rim light behind the globe (far side from camera, offset to
	// one side so it reads as a crescent highlight rather than a flat
	// silhouette wash) - see build_earth_lights() for its NEE-sampled twin.
	auto rim = make_shared<diffuse_light>(color(0.9, 1.0, 1.3));
	world.add(make_shared<quad>(point3(-4.0, -2.5, -6.0), vec3(3.0,0,0), vec3(0,5.0,0), rim));

	return world;
}

/**
 * Light list for build_earth() - the rim-light quad, for NEE importance
 * sampling. Replaces sky_dummy_lights() now that the scene has a real light.
 */
inline hittable_list build_earth_lights() {
	hittable_list lights;
	auto empty_mat = std::shared_ptr<material>();
	lights.add(make_shared<quad>(point3(-4.0, -2.5, -6.0), vec3(3.0,0,0), vec3(0,5.0,0), empty_mat));
	return lights;
}

/**
 * Build Perlin noise spheres scene
 */
inline hittable_list build_perlin_spheres() {
	hittable_list world;

	auto pertext = make_shared<noise_texture>(4);
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(pertext)));
	world.add(make_shared<sphere>(point3(0,2,0), 2, make_shared<lambertian>(pertext)));

	// Two smaller marble companion spheres (different noise scale for
	// variety) grouped near the main sphere - was previously just 2 bare
	// spheres lit only by flat sky ambient with no directed light at all.
	auto pertext2 = make_shared<noise_texture>(8);
	world.add(make_shared<sphere>(point3(2.2, 0.8, 1.0), 0.8, make_shared<lambertian>(pertext2)));
	world.add(make_shared<sphere>(point3(-1.8, 0.6, -1.2), 0.6, make_shared<lambertian>(pertext2)));

	// Warm key light from upper-left - see build_perlin_spheres_lights().
	auto key = make_shared<diffuse_light>(color(8, 6, 3));
	world.add(make_shared<quad>(point3(-4,6,-3), vec3(4,0,0), vec3(0,0,4), key));

	return world;
}

/**
 * Light list for build_perlin_spheres() - the key-light quad, for NEE
 * importance sampling. Replaces sky_dummy_lights() now that the scene has a
 * real light.
 */
inline hittable_list build_perlin_spheres_lights() {
	hittable_list lights;
	auto empty_mat = std::shared_ptr<material>();
	lights.add(make_shared<quad>(point3(-4,6,-3), vec3(4,0,0), vec3(0,0,4), empty_mat));
	return lights;
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

	// A real light floating in the room, facing the camera - previously
	// this scene had NO registered lights at all, so every quad read as a
	// flat, orientation-independent color swatch lit only by the ambient
	// background. See build_quads_lights() for its NEE-sampled twin.
	auto lamp = make_shared<diffuse_light>(color(7, 7, 6.5));
	world.add(make_shared<quad>(point3(-1,0.5,3), vec3(2,0,0), vec3(0,1,0), lamp));

	return world;
}

/**
 * Light list for build_quads() - the floating lamp quad, for NEE importance
 * sampling. Replaces sky_dummy_lights() now that the scene has a real
 * light.
 */
inline hittable_list build_quads_lights() {
	hittable_list lights;
	auto empty_mat = std::shared_ptr<material>();
	lights.add(make_shared<quad>(point3(-1,0.5,3), vec3(2,0,0), vec3(0,1,0), empty_mat));
	return lights;
}

/**
 * Build simple light scene with Perlin spheres
 */
inline hittable_list build_simple_light() {
	hittable_list world;

	auto pertext = make_shared<noise_texture>(4);
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(pertext)));
	world.add(make_shared<sphere>(point3(0,2,0), 2, make_shared<lambertian>(pertext)));

	// Warm sphere light above, cool quad light to the side - previously
	// both were the same flat white (4,4,4), placed symmetrically, so
	// there was no color/temperature contrast to read as two distinct
	// lights rather than one doubled-up source.
	auto warm_light = make_shared<diffuse_light>(color(6,3,1));
	world.add(make_shared<sphere>(point3(0,7,0), 2, warm_light));
	auto cool_light = make_shared<diffuse_light>(color(2,3,6));
	world.add(make_shared<quad>(point3(3.5,1,-3), vec3(2,0,0), vec3(0,2,0), cool_light));

	return world;
}

/**
 * Build Cornell box with smoke/fog
 */
inline hittable_list build_cornell_smoke() {
	using namespace cornell_box_data;
	hittable_list world;

	// The 5 standard walls (green/red/ceiling/floor/back) - shares
	// cornell_box_data::kQuads[0..4] with GPU's build_cornell_smoke_gpu().
	// This scene's own light is a different size/color than kQuads[5], so
	// it's added separately below rather than looping through index 5.
	for (int i = 0; i < 5; ++i) {
		const QuadSpec& q = kQuads[i];
		auto mat = make_shared<lambertian>(color(q.color.r, q.color.g, q.color.b));
		world.add(make_shared<quad>(
			point3(q.Q.x, q.Q.y, q.Q.z),
			vec3(q.u.x, q.u.y, q.u.z),
			vec3(q.v.x, q.v.y, q.v.z),
			mat));
	}

	auto white = make_shared<lambertian>(color(.73, .73, .73));  // for the boxes below
	auto light = make_shared<diffuse_light>(color(7, 7, 7));
	world.add(make_shared<quad>(point3(113,554,127), vec3(330,0,0), vec3(0,0,305), light));

	shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), white);
	box1 = make_shared<rotate_y>(box1, 15);
	box1 = make_shared<translate>(box1, vec3(265,0,295));

	shared_ptr<hittable> box2 = box(point3(0,0,0), point3(165,165,165), white);
	box2 = make_shared<rotate_y>(box2, -18);
	box2 = make_shared<translate>(box2, vec3(130,0,65));

	// Tinted fog instead of monochrome black/white - gives the two smoke
	// boxes real color interest as light scatters through them.
	world.add(make_shared<constant_medium>(box1, 0.01, color(0.05, 0.07, 0.12)));
	world.add(make_shared<constant_medium>(box2, 0.01, color(1.0, 0.85, 0.6)));

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
