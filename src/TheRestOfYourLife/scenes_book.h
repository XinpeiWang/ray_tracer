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

// build_bouncing_spheres() deleted - A2 migrated to pbrt-backed, see
// pbrt_scenes/bouncing-spheres.pbrt and scene_registry_data.h's own entry
// (that file's own header comment has the full derivation of its grid
// layout, reused from GPU's own deterministic std::mt19937(42) sequence).
// No other consumer (gpu/metal/metal_poc_scenes_a.mm's own scene is an
// independent hand-ported Metal implementation that never called this
// function).

// build_checkered_spheres() deleted - A3 migrated to pbrt-backed, see
// pbrt_scenes/checkered-spheres.pbrt and scene_registry_data.h's own entry.
// No other consumer.

// build_earth()/build_earth_lights() deleted - A4 migrated to pbrt-backed,
// see pbrt_scenes/earth-globe.pbrt and scene_registry_data.h's own entry.
// Neither had any other consumer.

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

// build_quads()/build_quads_lights() (former A6 Colored Quads native CPU
// builder) deleted - A6 migrated to pbrt-backed, see pbrt_scenes/
// colored-quads.pbrt and scene_registry_data.h's A6 entry. NOT the Metal
// backend's own buildColoredQuads() (gpu/metal/metal_poc_scenes_a.mm) -
// that's a fully independent, hardcoded dispatch unaffected by this C++
// registry, deliberately out of this migration's scope.

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

// build_cornell_smoke() deleted - A8 migrated to pbrt-backed, see
// pbrt_scenes/cornell-smoke.pbrt and scene_registry_data.h's own entry. No
// other consumer.

// build_final_scene()/build_final_scene_lights() deleted - A9 migrated to
// pbrt-backed, see pbrt_scenes/final-scene.pbrt and scene_registry_data.h's
// own entry (that file's own header comment has the full derivation,
// including the one deliberate noise-texture approximation). No other
// consumer (gpu/metal/metal_poc_scenes_a.mm's own scene is an independent
// hand-ported Metal implementation that never called either function).
