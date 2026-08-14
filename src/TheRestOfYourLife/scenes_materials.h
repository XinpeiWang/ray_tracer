#pragma once
// scenes_materials.h -- PBR material showcase scenes (scenes 10-17)
// Included by scenes.h umbrella header.

#include "hittable_list.h"
#include "sphere.h"
#include "quad.h"
#include "material.h"
#include "bvh.h"
#include "scenes_book.h"  // add_cornell_walls_and_main_light()

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

	// Thin-glass panel (IOR 1.5) -- angled ~62 degrees off the camera's
	// straight-on view axis so it's actually visible. Fresnel reflectance
	// for IOR 1.5 only rises steeply near grazing incidence (~4% at 0 deg,
	// ~9% at 60 deg, ~35% at 80 deg) - facing the camera dead-on (0 deg, as
	// this panel used to) or even a mild 28-degree tilt (still <5%) both
	// made it imperceptible; 62 degrees was tuned by rendering until the
	// sheen actually reads while the panel is still wide enough on screen
	// to not foreshorten into an unreadable sliver. Built centered at the
	// local origin so rotate_y (which pivots around world/local (0,0,0))
	// rotates the panel in place, then translated to its position in the box.
	auto panel = make_shared<thin_dielectric>(1.5);
	shared_ptr<hittable> panel_quad = make_shared<quad>(
		point3(-177.5, -277.5, 0), vec3(0, 555, 0), vec3(355, 0, 0), panel);
	panel_quad = make_shared<rotate_y>(panel_quad, 62);
	panel_quad = make_shared<translate>(panel_quad, vec3(277.5, 277.5, 200));
	world.add(panel_quad);

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
	add_cornell_walls_and_main_light(world);

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
	add_cornell_walls_and_main_light(world);
	auto white = make_shared<lambertian>(color(.73, .73, .73));  // for the box below, same albedo as the walls

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
	add_cornell_walls_and_main_light(world);
	auto white = make_shared<lambertian>(color(.73, .73, .73));  // for the box below, same albedo as the walls

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

