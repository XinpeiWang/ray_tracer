#pragma once
// scenes_materials.h -- PBR material showcase scenes (category B, "Materials")
// Included by scenes.h umbrella header.

#include "hittable_list.h"
#include "sphere.h"
#include "quad.h"
#include "triangle.h"
#include "material.h"
#include "bvh.h"
#include "scenes_book.h"  // add_cornell_walls_and_main_light()
#include "punctual_light_objects.h"  // distant_light_obj, via punctual_light_list::add_distant()

// build_rough_metal_spheres() deleted - B1 migrated to pbrt-backed, see
// pbrt_scenes/rough-metal-spheres.pbrt and scene_registry_data.h's own entry. No
// other consumer. (build_cornell_rough_metal() below stays, like its siblings
// build_cornell_rough_glass/conductor/coated_diffuse: B2 migrated to pbrt-backed,
// but tests/integration/skip_pdf_material_brightness_tests.cpp still renders that
// native world through camera::ray_color() as a regression fixture.)


/**
 * Cornell box with rough metal objects (GGX microfacet showcase)
 * Replaces the white diffuse box with rough aluminum and the glass sphere
 * with a rough gold sphere -- directly shows the GGX BRDF in a familiar scene.
 */
inline hittable_list build_cornell_rough_metal() {
	hittable_list world;
	add_cornell_walls_and_main_light(world);

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
	add_cornell_walls_and_main_light(world);

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
	add_cornell_walls_and_main_light(world);

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
	add_cornell_walls_and_main_light(world);

	// Blue coated-diffuse sphere (IOR 1.5, roughness 0.1)
	auto coated_blue = make_shared<coated_diffuse>(color(0.2, 0.3, 0.9), 1.5, 0.1);
	world.add(make_shared<sphere>(point3(190, 90, 190), 90, coated_blue));

	// Orange/terracotta coated-diffuse box (IOR 1.5, roughness 0.2 -- slightly
	// rougher coat). Was near-identical red to the wall behind it and didn't
	// read as a distinct object; shifted hue only, same coat properties.
	auto coated_red = make_shared<coated_diffuse>(color(0.75, 0.35, 0.1), 1.5, 0.2);
	shared_ptr<hittable> box1 = box(point3(0,0,0), point3(165,330,165), coated_red);
	box1 = make_shared<rotate_y>(box1, 15);
	box1 = make_shared<translate>(box1, vec3(265,0,295));
	world.add(box1);

	return world;
}

// build_cornell_thin_glass() (former scene 14 / B6), build_cornell_coated_
// conductor() (former scene 15 / B7), and build_cornell_wax_slab() (former
// scene 16 / B8) all deleted - B6/B7/B8 migrated to pbrt-backed, see
// pbrt_scenes/cornell-thin-glass.pbrt/cornell-coated-conductor.pbrt/
// cornell-wax-slab.pbrt and scene_registry_data.h's own entries. None of
// these three had any other consumer (unlike build_cornell_box/
// build_cornell_conductor/build_cornell_coated_diffuse/build_cornell_
// rough_glass, which stay).

// build_cornell_crystal() deleted - B9 migrated to pbrt-backed, see
// pbrt_scenes/cornell-crystal.pbrt and scene_registry_data.h's own entry
// (that file's own header comment documents the new, non-standard Material
// "normalizedfresnel" kind added to the pbrt loader for this migration).
// No other consumer.

// build_prism_dispersion()/build_prism_dispersion_geometry()/build_prism_dispersion_punct()
// deleted - B23 migrated to pbrt-backed earlier and I2 (their last consumer, a CPU-only
// Education scene) has now too: both read pbrt_scenes/prism-dispersion.pbrt. No other
// consumer.

