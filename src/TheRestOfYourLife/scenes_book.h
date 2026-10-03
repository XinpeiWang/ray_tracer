#pragma once
// scenes_book.h -- Book series scenes (category A, "Basics" - A1 through A9)
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

// build_perlin_spheres()/build_perlin_spheres_lights() deleted - A5 migrated to
// pbrt-backed, see pbrt_scenes/perlin-spheres.pbrt and scene_registry_data.h's own
// entry (that file's header has the one deliberate noise-texture substitution).
// No other consumer (gpu/metal/'s own Perlin scene is an independent hand-ported
// Metal implementation that never called either function).

// build_final_scene()/build_final_scene_lights() deleted - A9 migrated to
// pbrt-backed, see pbrt_scenes/final-scene.pbrt and scene_registry_data.h's
// own entry (that file's own header comment has the full derivation,
// including the one deliberate noise-texture approximation). No other
// consumer (gpu/metal/metal_poc_scenes_a.mm's own scene is an independent
// hand-ported Metal implementation that never called either function).
