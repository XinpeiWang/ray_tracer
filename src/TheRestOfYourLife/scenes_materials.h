#pragma once
// scenes_materials.h -- PBR material showcase scenes (scenes 10-17)
// Included by scenes.h umbrella header.

#include "hittable_list.h"
#include "sphere.h"
#include "quad.h"
#include "triangle.h"
#include "material.h"
#include "bvh.h"
#include "scenes_book.h"  // add_cornell_walls_and_main_light()
#include "punctual_light_objects.h"  // distant_light_obj, via punctual_light_list::add_distant()

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

/**
 * build_prism_dispersion -- scene B23
 * A literal glass prism (crown glass, eta_d=1.52, Abbe=59, via dielectric's
 * new dispersive constructor - material_simple.h) lit by a near-horizontal
 * parallel white light, splitting into a visible chromatic fan on a catcher
 * screen. --spectral only: under the default flat-RGB path this is just an
 * ordinary (non-dispersive-looking) glass wedge, since dispersion requires
 * per-wavelength tracking (camera.h's ray_color_spectral()).
 *
 * Deliberately NOT a Cornell box - colored walls would tint/muddy a pure-
 * white dispersion effect. A minimal standalone scene instead: black
 * background, the prism, and one white catcher screen.
 *
 * Light travels primarily along world +Z (entering the prism, exiting
 * toward the screen) - deliberately matching every other scene in this
 * registry's own "camera looks along +Z" convention (e.g. Cornell box's
 * lookfrom=(278,278,-800)/lookat=(278,278,278)) rather than an arbitrary
 * axis, so this scene's own camera framing behaves as predictably as any
 * other scene here instead of fighting an unfamiliar viewing angle.
 *
 * Geometry: a real triangular prism, 3 rectangular quad sides + 2 triangle
 * end caps sharing one small triangle_mesh_data (2 faces), all under one
 * dispersive dielectric - same construction pattern as
 * scenes_advanced.h's build_triangle_mesh_scene() icosahedron. Cross-
 * section triangle (in the Y-Z plane, apex up) A(y=0,z=0) B(y=0,z=140)
 * C(y=121,z=70) (near-equilateral, side ~140), extruded along +X by 150 (the
 * prism's own "length", not otherwise significant). End-capping properly
 * (not leaving the prism "tube" open) matters for correctness: the
 * renderer's entering/exiting-medium bookkeeping needs a watertight
 * boundary, or a ray could exit through an uncapped end mid-medium with
 * the wrong material state. Side-quad and end-cap triangle winding below
 * is hand-derived (verified against each face's own outward direction from
 * the solid's centroid) so each face's geometric normal (quad.h:
 * cross(u,v) / triangle.h: cross(p1-p0,p2-p0)) points OUTWARD.
 */
inline hittable_list build_prism_dispersion() {
	hittable_list world;

	const point3 A(0, 0, 0), B(0, 0, 140), C(0, 121, 70);
	const vec3 depth(150, 0, 0);

	auto glass = make_shared<dielectric>(1.52, 59.0, /*dispersive_tag=*/true);

	// 3 rectangular sides (outward-normal winding - see this function's own
	// comment). Note u/v order is (depth, edge) here, not (edge, depth) -
	// verified per-face against the solid's centroid, needed because this
	// cross-section's apex-up orientation is now in Y-Z (not X-Y).
	world.add(make_shared<quad>(A, depth, B - A, glass));               // base (z=0..140 side, y=0)
	world.add(make_shared<quad>(B, depth, C - B, glass));               // exit slant (toward +z)
	world.add(make_shared<quad>(C, depth, A - C, glass));               // entry slant (toward -z)

	// 2 triangular end caps, sharing one small mesh.
	auto mesh_data = make_shared<triangle_mesh_data>();
	mesh_data->positions = { A, B, C, A + depth, B + depth, C + depth };
	// x=0 cap: forward winding (A,B,C) -> cross(B-A,C-A) points -X (outward).
	// x=150 cap: reversed winding (A',C',B') -> points +X (outward).
	mesh_data->indices = { 0, 1, 2,   3, 5, 4 };
	world.add(make_shared<triangle>(mesh_data, 0, glass));
	world.add(make_shared<triangle>(mesh_data, 1, glass));

	// Catcher screen: large white diffuse wall on the far (+Z) side.
	auto screen_mat = make_shared<lambertian>(color(0.9, 0.9, 0.9));
	world.add(make_shared<quad>(point3(-300, -300, 600), vec3(600, 0, 0), vec3(0, 700, 0), screen_mat));

	return world;
}

inline std::shared_ptr<punctual_light_list> build_prism_dispersion_punct() {
	auto pl = std::make_shared<punctual_light_list>();
	// add_distant()'s dir is the direction TOWARD the light source (verified
	// empirically - confusingly, NOT "the direction light travels", despite
	// that field's own doc comment in punctual_light_objects.h suggesting
	// otherwise). The light source is toward -Z (and slightly +Y, i.e.
	// "up"), so its rays travel toward +Z and slightly downward (-Y) into
	// the prism's entry slant - a slight downward tilt so the dispersed
	// fan lands comfortably within the screen's extent.
	pl->add_distant(vec3(0.0, 0.06, -1.0), color(1.0, 1.0, 1.0), 1000.0, 3.0);
	return pl;
}

