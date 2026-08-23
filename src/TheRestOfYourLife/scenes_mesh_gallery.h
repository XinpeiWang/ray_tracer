#pragma once
// scenes_mesh_gallery.h -- imported third-party mesh scenes (Stanford models,
// Crytek Sponza, Amazon Lumberyard Bistro, and the rest of the external-asset
// gallery, scenes 38-81). Split out of scenes_advanced.h, which #includes
// this file at its own end; see that file's own header comment for the full
// picture (hand-authored technique-showcase scenes vs. this imported
// gallery). Every scene here follows the same rigid template: load one or
// more external mesh files via mesh.h's triangle_mesh, wrap in a
// checkered-ground + light setup, and (for outdoor scenes) a matching
// build_X_sky() helper.

#include "hittable_list.h"
#include "sphere.h"
#include "material.h"
#include "sky_light.h"
#include "mesh.h"
#include <memory>

// ============================================================================
// Scene 38: Stanford Bunny
// The classic Stanford 3D Scanning Repository bunny (35,947 vertices, 69,451
// triangles - the standard "bun_zipper" reconstruction, downloaded as
// models/stanford-bunny.obj) rendered in polished bronze, sitting on a
// checkered ground under an overhead area light. Unlike scene 37's
// procedural icosahedron, this exercises mesh.h's load_obj() end-to-end
// against a real, large, external asset - the file has positions only (no
// vn/vt), so triangle::hit() falls back to flat per-face geometric normals
// here too, same as scene 37. That ruled out a dielectric material here: a
// glass surface refracts per-facet with no smooth normal interpolation to
// hide it, so light scatters incoherently across adjacent faces instead of
// converging into a clear "see-through" image - it renders as a matte,
// frosted-looking blob rather than glass (confirmed by an earlier render of
// this exact scene). Metal reflects the same per-facet normals but doesn't
// need coherence to look right, so the facets read as an intentional
// low-poly-statue style instead of a rendering artifact - same reasoning as
// scene 37's own metal icosahedron.
// scale/offset below were computed from the raw OBJ's own bounding box
// (x:[-0.09469,0.06101] y:[0.03299,0.18732] z:[-0.06187,0.05880], i.e. a
// real-world ~15cm scan) to sit the model on the ground plane (y=0) centered
// on the origin at a ~3-unit height, matching scene 37's icosahedron scale.
// ============================================================================
inline hittable_list build_stanford_bunny() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto bronze = make_shared<metal>(color(0.71, 0.43, 0.20), 0.15);
	world.add(std::make_shared<triangle_mesh>(
		"stanford-bunny.obj", bronze,
		/*scale=*/19.4, point3(0.3267, -0.6398, 0.0298)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 39: Stanford Armadillo
// The Stanford 3D Scanning Repository armadillo (49,990 vertices, 99,976
// triangles, downloaded as models/armadillo.obj from the same common-3d-
// test-models mirror as scene 38's bunny) rendered in gunmetal, sitting on
// the same checkered-ground + overhead-area-light setup as scene 38.
// Same "positions only, no vn/vt" situation as scenes 37/38 (confirmed via
// grep - zero `vn` lines in the source file), so triangle::hit() falls back
// to flat per-face geometric normals here too, and metal is used for the
// same reason as those two scenes: it reflects the per-facet normals
// coherently (reads as an intentional low-poly-statue style) where a
// dielectric would scatter light incoherently across adjacent facets and
// look like frosted glass instead of clear glass.
// scale/offset below were computed from the raw OBJ's own bounding box
// (x:[-63.497,63.517] y:[-54.220,97.087] z:[-57.715,57.696], i.e. a
// real-world ~150mm scan) to sit the model on the ground plane (y=0)
// centered on the origin at a ~3-unit height, matching scenes 37/38's own
// mesh scale convention.
// ============================================================================
inline hittable_list build_stanford_armadillo() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto gunmetal = make_shared<metal>(color(0.55, 0.56, 0.58), 0.08);
	world.add(std::make_shared<triangle_mesh>(
		"armadillo.obj", gunmetal,
		/*scale=*/0.0198, point3(0.0, 1.0736, 0.0)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 40: Stanford Happy Buddha
// The Stanford 3D Scanning Repository happy buddha (49,251 vertices, 98,601
// triangles, downloaded as models/happy-buddha.obj from the same common-3d-
// test-models mirror as scenes 38/39) rendered in polished gold, sitting on
// the same checkered-ground + overhead-area-light setup as those two scenes.
// Same "positions only, no vn/vt" situation (confirmed via grep - zero `vn`
// lines in the source file), so triangle::hit() falls back to flat per-face
// geometric normals here too, and metal is used for the same reason as
// scenes 37/38/39: it reflects the per-facet normals coherently (reads as
// an intentional low-poly-statue style) where a dielectric would scatter
// light incoherently across adjacent facets and look like frosted glass
// instead of clear glass.
// scale/offset below were computed from the raw OBJ's own bounding box
// (x:[-0.04610,0.03522] y:[0.04976,0.24779] z:[-0.04739,0.03400], i.e. a
// real-world ~20cm scan) to sit the model on the ground plane (y=0)
// centered on the origin at a ~3-unit height, matching scenes 37/38/39's
// own mesh scale convention.
// ============================================================================
inline hittable_list build_stanford_happy_buddha() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto gold = make_shared<metal>(color(0.83, 0.69, 0.22), 0.05);
	world.add(std::make_shared<triangle_mesh>(
		"happy-buddha.obj", gold,
		/*scale=*/15.1496, point3(0.0824, -0.7539, 0.1015)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 41: Stanford Lucy
// The Stanford 3D Scanning Repository Lucy (a standing angel figure,
// 49,987 vertices, 99,970 triangles - this mirror's decimated version; the
// original scan is ~28M triangles), downloaded as models/lucy.obj from the
// same common-3d-test-models mirror as scenes 38-40, rendered in bright
// silver, sitting on the same checkered-ground + overhead-area-light setup
// as those three scenes.
// UNLIKE those three, this mirror's lucy.obj ships Z-up (its raw bounding
// box has z-range 1597, dwarfing its y-range of 534 and x-range of 930 - a
// tall standing figure should have its LARGEST extent along the up axis,
// which was Z here, not Y like every other mesh scene in this codebase
// assumes). Since mesh.h's triangle_mesh only supports a scale+translate
// transform (see its constructor - no rotation parameter), the fix was
// applied once, directly to the downloaded file, before this scene was
// wired up: every vertex was rotated -90 degrees about the X axis
// (new_y=old_z, new_z=-old_y) so the file committed to this repo is
// already Y-up, matching every other mesh scene's convention - no special-
// casing needed anywhere in the loading/rendering path.
// Same "positions only, no vn/vt" situation as scenes 37-40 (confirmed via
// grep - zero `vn` lines in the source file).
// scale/offset below were computed from the (already-rotated) OBJ's own
// bounding box (x:[225.9,1156.0] y:[-605.9,991.3] z:[-145.3,388.4]) to sit
// the model on the ground plane (y=0) centered on the origin at a ~3-unit
// height, matching scenes 37-40's own mesh scale convention.
// ============================================================================
inline hittable_list build_stanford_lucy() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto silver = make_shared<metal>(color(0.85, 0.85, 0.88), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"lucy.obj", silver,
		/*scale=*/0.0018783, point3(-1.2978, 1.1380, -0.2283)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 42: Stanford XYZRGB Dragon
// Same external-mesh convention as scenes 37-41 (checkered ground, metal
// material, overhead area light). Source file's own bounding box
// (x:[-98.6,103.1] y:[-62.75,49.29] z:[-57.25,76.97]) confirmed Y-up out of
// the box (unlike Lucy) - no rotation needed.
// scale/offset computed from that bounding box to sit the model on the
// ground plane (y=0) centered on the origin at a ~3-unit height, matching
// scenes 37-41's own mesh scale convention.
// ============================================================================
inline hittable_list build_stanford_dragon() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto silver = make_shared<metal>(color(0.85, 0.85, 0.88), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"xyzrgb_dragon.obj", silver,
		/*scale=*/0.0267772, point3(-0.0600, 1.6803, -0.2640)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 43: Utah Teapot
// Same external-mesh convention as scenes 37-42 (checkered ground, metal
// material, overhead area light) - but the classic CG test model instead
// of a statue. Source file's own bounding box (x:[-3.0,6.434] y:[0.0,3.15]
// z:[-2.0,2.0]) confirmed Y-up and already sitting on y=0 - no rotation
// needed, and the y offset is 0 (the mesh already touches the ground plane
// at its minimum, no lift required).
// scale/offset computed from that bounding box to normalize the model to
// the same ~3-unit height and origin-centered footprint used by every
// other mesh scene.
// ============================================================================
inline hittable_list build_utah_teapot() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto silver = make_shared<metal>(color(0.85, 0.85, 0.88), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"teapot.obj", silver,
		/*scale=*/0.952381, point3(-1.6352, 0.0, 0.0)));

	// Area light - raised to y=20 instead of the usual y=8: this scene's
	// camera is pulled back and raised well above the other mesh scenes'
	// default (see this function's registry comment) specifically to fit
	// the teapot's wide spout+handle, and at that raised angle the
	// standard y=8 light sphere sat inside the camera's FOV and rendered
	// as a blown-out white disc in frame (an x-shift alone wasn't enough
	// margin - confirmed by render). Moving it further overhead pushes it
	// well outside the frustum for any reasonably-angled camera while
	// still lighting the scene from roughly straight above. Brightness
	// scaled up ~8x (6,6,6 -> 48,48,48) to compensate for inverse-square
	// falloff over the much greater light-to-subject distance (roughly
	// 2.7x further than the usual y=8 placement).
	world.add(make_shared<sphere>(point3(0, 20, 0), 2, make_shared<diffuse_light>(color(48,48,48))));
	return world;
}

// ============================================================================
// Scene 44: Spot the Cow (Keenan Crane)
// Same external-mesh convention as scenes 37-43 (checkered ground, metal
// material, overhead area light). Axis orientation unverified before first
// render (unlike scenes 37-43, whose bounding boxes already made Y-up vs
// Z-up unambiguous) - scale/offset below assume Y-up like every prior mesh
// scene; if the first CPU render shows the cow lying on its side, this
// comment and the offset need updating to rotate it, matching how Lucy
// (scene 41) was handled.
// scale/offset computed from the raw OBJ's own bounding box (x:[-0.472,
// 0.472] y:[-0.737,0.954] z:[-0.669,1.049]) to normalize the model to the
// same ~3-unit standing height used by every other mesh scene.
// ============================================================================
inline hittable_list build_spot_cow() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto silver = make_shared<metal>(color(0.85, 0.85, 0.88), 0.1);
	// The raw mesh's forward-facing direction points away from the camera
	// (confirmed by render - the shot showed hindquarters/tail, not the
	// face) - mesh.h's triangle_mesh has no rotation parameter, so the mesh
	// is wrapped in the same rotate_y() hittable the CSG box scenes already
	// use. rotate_y pivots about the WORLD origin, not the object's own
	// center, so the mesh is built with NO offset (raw*scale only, still
	// centered on its own local origin), rotated 180deg about Y there, and
	// only THEN translated to the original target offset - building at the
	// final offset first and rotating afterward would have pivoted around
	// the wrong point and shifted the cow's world position by 2x its own
	// (nonzero) z-offset.
	shared_ptr<hittable> cow = std::make_shared<triangle_mesh>(
		"spot.obj", silver, /*scale=*/1.7747);
	cow = std::make_shared<rotate_y>(cow, 180);
	world.add(std::make_shared<translate>(cow, vec3(0.0, 1.3076, -0.3373)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 45: Suzanne (Blender's monkey-head mascot)
// Same external-mesh convention as scenes 37-44 (checkered ground, metal
// material, overhead area light). Source faces are mostly quads (468 of
// 500), fan-triangulated the same way every other mesh scene's loader
// already handles n-gons - no special casing needed. Axis orientation
// unverified before first render, same situation as Spot (scene 44) - scale/
// offset below assume Y-up; if the first CPU render shows it on its side,
// this needs updating to rotate it, matching how Lucy (scene 41) was
// handled.
// scale/offset computed from the raw OBJ's own bounding box (x:[-3.861,
// -1.127] y:[0.267,2.236] z:[3.252,4.955] - not origin-centered in the
// source file, unlike every prior mesh scene, but the offset below
// re-centers it the same way regardless) to normalize the model to the
// same ~3-unit standing height used by every other mesh scene.
// ============================================================================
inline hittable_list build_suzanne() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto silver = make_shared<metal>(color(0.85, 0.85, 0.88), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"suzanne.obj", silver,
		/*scale=*/1.52381, point3(3.8005, -0.4073, -6.2536)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}
// ============================================================================
// Scene 46: Nefertiti Bust
// Same external-mesh convention as scenes 37-45 (checkered ground, metal
// material, overhead area light). Source file's own bounding box (x:[-119.3,
// 119.4] y:[-181.3,181.3] z:[-247.3,247.3]) had its LARGEST extent on z, not
// y - unlike every prior mesh scene (whose bounding box either already made
// Y-up unambiguous, or turned out Y-up anyway after a first CPU render) -
// which for a bust's naturally-tall proportions meant Z-up, the same
// situation Lucy (scene 41) hit. Fixed the same way: every vertex was
// rotated -90 degrees about the X axis (new_y=old_z, new_z=-old_y) directly
// in the downloaded file before it was committed to this repo, so no
// rotation is needed here - just the usual scale+translate.
// scale/offset computed from the (already-rotated) OBJ's own bounding box
// (x:[-119.3,119.4] y:[-247.3,247.3] z:[-181.2,181.3]) to sit the model on
// the ground plane (y=0) centered on the origin at the same ~3-unit
// standing height used by every other mesh scene.
// ============================================================================
inline hittable_list build_nefertiti() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto silver = make_shared<metal>(color(0.85, 0.85, 0.88), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"nefertiti.obj", silver,
		/*scale=*/0.0060654, point3(-0.0001, 1.4998, -0.0002)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 47: Horse
// Same external-mesh convention as scenes 37-46 (checkered ground, metal
// material, overhead area light). Axis orientation unverified before first
// render - scale/offset below assume Y-up like most prior mesh scenes; if
// the first CPU render shows it on its side, this needs updating to rotate
// it, matching how Lucy (scene 41) and Nefertiti (scene 46) were handled.
// scale/offset computed from the raw OBJ's own bounding box (x:[-0.042,
// 0.042] y:[-0.0917,0.0917] z:[-0.0764,0.0764]) to normalize the model to
// the same ~3-unit standing height used by every other mesh scene.
// ============================================================================
inline hittable_list build_horse() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto silver = make_shared<metal>(color(0.85, 0.85, 0.88), 0.1);
	// Same wrong-facing-direction problem as build_spot_cow() (confirmed by
	// render - shows the back of the arched neck, not the face) and the
	// same fix: rotate_y() wrapper, since mesh.h's triangle_mesh has no
	// rotation parameter of its own.
	shared_ptr<hittable> horse = std::make_shared<triangle_mesh>(
		"horse.obj", silver,
		/*scale=*/16.36295, point3(0.0, 1.5, 0.0));
	world.add(std::make_shared<rotate_y>(horse, 180));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 48: Cheburashka
// Same external-mesh convention as scenes 37-47 (checkered ground, metal
// material, overhead area light). Axis orientation unverified before first
// render - scale/offset below assume Y-up like most prior mesh scenes; if
// the first CPU render shows it on its side, this needs updating to rotate
// it, matching how Lucy (scene 41) and Nefertiti (scene 46) were handled.
// scale/offset computed from the raw OBJ's own bounding box (x:[0.05,0.95]
// y:[0.07923,0.92077] z:[0.338318,0.661682]) to normalize the model to the
// same ~3-unit standing height used by every other mesh scene.
// ============================================================================
inline hittable_list build_cheburashka() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto silver = make_shared<metal>(color(0.85, 0.85, 0.88), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"cheburashka.obj", silver,
		/*scale=*/3.5648929, point3(-1.7824465, -0.2824465, -1.7824465)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 49: Trophy Room
// First scene to place multiple external meshes together in one composition
// (every prior mesh scene, 37-48, is a single statue alone on the checkered
// ground). Four already-downloaded meshes - bunny, teapot, Suzanne, and Spot
// the Cow - reused at a shared, smaller ~1.6-unit display height and lined
// up along a shelf, each in a different one of the polished-metal tones
// already proven safe on scanned meshes elsewhere in this file (bronze:
// scene 38, silver/chrome: scene 43, gold: scene 40, gunmetal: scene 39) -
// deliberately staying off dielectric here: an earlier attempt to render the
// XYZRGB Dragon (scene 42's mesh) in glass produced persistent salt-and-
// pepper noise that did not converge even at 3000spp (vs the usual ~150-300
// for these scenes), most likely from inconsistent triangle winding in that
// scan confusing the inside/outside test refraction depends on - flagged
// separately for investigation, not something to build a showcase scene on
// top of yet.
// Each mesh's scale/offset below is its own solo scene's scale/offset
// (see scenes 38/43/44/45's own comments for each raw bounding box) scaled
// down by 1.6/3.0 to shrink it from that solo scene's ~3-unit standing
// height to ~1.6 units, plus a per-item x-shift to line the four up along
// the shelf; the y/z centering falls out of that same scaling automatically
// since each solo scale/offset pair was already tuned to center its mesh at
// x=0/z=0 sitting on y=0 - scaling both scale and offset by the same
// fraction preserves that centering at the smaller size, and the x-shift is
// simply added on top after.
// ============================================================================
inline hittable_list build_trophy_room() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto bronze   = make_shared<metal>(color(0.71, 0.43, 0.20), 0.15);
	auto chrome   = make_shared<metal>(color(0.85, 0.85, 0.88), 0.10);
	auto gold     = make_shared<metal>(color(0.83, 0.69, 0.22), 0.05);
	auto gunmetal = make_shared<metal>(color(0.55, 0.56, 0.58), 0.08);

	// Bunny (bronze) - solo scale/offset from scene 38: 19.4, (0.3267,-0.6398,0.0298)
	world.add(std::make_shared<triangle_mesh>(
		"stanford-bunny.obj", bronze,
		/*scale=*/10.3467, point3(-3.32576, -0.34123, 0.01589)));

	// Teapot (chrome) - solo scale/offset from scene 43: 0.952381, (-1.6352,0,0)
	world.add(std::make_shared<triangle_mesh>(
		"teapot.obj", chrome,
		/*scale=*/0.50794, point3(-2.07211, 0.0, 0.0)));

	// Suzanne (gold) - solo scale/offset from scene 45: 1.52381, (3.8005,-0.4073,-6.2536).
	// y lowered by 0.14 from the pure-scaled value (-0.21723 -> -0.35723):
	// Suzanne is a disembodied head with no neck/pedestal, so grounding its
	// chin at y=0 like every other mesh puts its eyes (~65% up the model,
	// same proportions as the solo scene) around y=1.04 - well above this
	// scene's shared shelf camera's lookat (y=0.9, chosen for the other
	// three grounded meshes). Solo scene 45 fixes the equivalent mismatch
	// by raising ITS camera instead (see that scene's registry comment),
	// but this composite scene has one camera for all four meshes, so the
	// only available lever here is lowering Suzanne herself closer to the
	// shared aim height.
	world.add(std::make_shared<triangle_mesh>(
		"suzanne.obj", gold,
		/*scale=*/0.81270, point3(3.22694, -0.35723, -3.33526)));

	// Spot the Cow (gunmetal) - solo scale/offset from scene 44: 1.7747, (0,1.3076,-0.3373).
	// Same wrong-facing-direction fix as solo build_spot_cow(), and the same
	// build-at-origin/rotate/translate composition (see that function's
	// comment for why: rotate_y pivots around the world origin, and this
	// piece's offset also carries the shelf x-shift on top of centering, so
	// rotating a mesh already built at its final offset would land it on
	// the wrong side of the shelf).
	shared_ptr<hittable> trophy_cow = std::make_shared<triangle_mesh>(
		"spot.obj", gunmetal, /*scale=*/0.94651);
	trophy_cow = std::make_shared<rotate_y>(trophy_cow, 180);
	world.add(std::make_shared<translate>(trophy_cow, vec3(3.5, 0.69739, -0.17989)));

	// Area light - standard radius-2/(0,8,0) placement, same as every other
	// mesh scene (a wider light was tried first, but became visible as a
	// blown-out disc in-frame with this scene's wider camera FOV).
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 50: Glass Dragon
// Same mesh/normalization as scene 42 (Stanford XYZRGB Dragon) - identical
// scale/offset - but `dielectric` glass instead of metal. This is the
// scene that originally motivated building --sppm in the first place: a
// large, deeply concave mesh refracting light needs dozens of internal
// bounces to resolve its caustics, which is chaotically sensitive to ray
// direction under unidirectional path tracing (both the regular CPU/GPU
// path tracer AND pbrt-v4's own default PathIntegrator hit this same
// wall - see the investigation this scene is named after). That's not a
// bug: MSE(100spp,10000spp) was shown to be no smaller than
// MSE(3000spp,10000spp) - the "noise" is the genuinely converged answer
// for that integrator, not shrinking variance.
//
// IMPORTANT, discovered while adding this scene: --sppm does NOT clean up
// the dragon's own surface either. sppm_adapter.h's camera pass only
// records a "visible point" (the thing photon density estimation actually
// improves) at a non-delta hit (is_delta_bsdf == false); the dragon is
// 100% dielectric (delta), so the camera never records a visible point ON
// it - only wherever its specular chain eventually lands on a non-delta
// surface (here, the checkered floor, or nowhere if the chain escapes to
// background). That means the dragon's own directly-visible glass surface
// is rendered by the exact same per-pixel specular-chain resampling as the
// plain path tracer (averaged only across nIterations camera passes via
// Ld/nIterations) - SPPM's photon-density machinery contributes nothing to
// IT specifically. Verified empirically: 80 iter x 60k photons vs 300 iter
// x 200k photons (4.8M vs 60M total photons) produced visually identical
// dragon-surface noise. What --sppm DOES add here is a genuine floor
// caustic under/around the dragon (photon density landing on the diffuse
// checker floor) that the regular path tracer's NEE-only floor shading
// can't resolve - a real, visible difference, just not a "clean glass
// dragon." A fully clean render of the glass surface itself would need
// bidirectional path tracing or MLT, neither of which exist in this
// codebase. GPU SPPM is Phase-1-scoped to scene 11 only as of this
// writing, so even the floor-caustic benefit needs CPU `--sppm`
// specifically for this scene.
// ============================================================================
inline hittable_list build_glass_dragon() {
	hittable_list world;

	// Ground
	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto glass = make_shared<dielectric>(1.5);
	world.add(std::make_shared<triangle_mesh>(
		"xyzrgb_dragon.obj", glass,
		/*scale=*/0.0267772, point3(-0.0600, 1.6803, -0.2640)));

	// Area light
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 51: Beast
// Fantasy creature bust from common-3d-test-models. Scale/offset from raw
// OBJ bounding box (Y-up, standing on y=0), same convention as scenes
// 37-50.
// ============================================================================
inline hittable_list build_beast() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto bronze = make_shared<metal>(color(0.71, 0.43, 0.20), 0.15);
	world.add(std::make_shared<triangle_mesh>(
		"beast.obj", bronze,
		/*scale=*/0.0118956, point3(0.0000, 0.0103, -0.3401)));

	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 52: VW Beetle
// Classic CAD-style Volkswagen Beetle test mesh. Notably elongated along Z
// after normalization (~3.6 wide/tall x ~8.8 deep) - same situation as
// scene 43's Utah Teapot, camera pulled back accordingly in the registry.
// ============================================================================
inline hittable_list build_beetle() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto chrome = make_shared<metal>(color(0.85, 0.85, 0.88), 0.08);
	world.add(std::make_shared<triangle_mesh>(
		"beetle.obj", chrome,
		/*scale=*/9.9009901, point3(0.3614, -3.0297, -1.9010)));

	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 54: Bimba
// Smooth abstract bust/statue (AIM@SHAPE repository test model).
// ============================================================================
inline hittable_list build_bimba() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto gold = make_shared<metal>(color(0.83, 0.69, 0.22), 0.05);
	world.add(std::make_shared<triangle_mesh>(
		"bimba.obj", gold,
		/*scale=*/9.2592593, point3(0.3333, 2.2963, 10.7454)));

	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 55: Cow
// Classic Viewpoint/Alias Cow test model (distinct from scene 44's Spot
// the Cow - a different, higher-poly untextured cow mesh).
// ============================================================================
inline hittable_list build_cow() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto brass = make_shared<metal>(color(0.80, 0.65, 0.28), 0.15);
	world.add(std::make_shared<triangle_mesh>(
		"cow.obj", brass,
		/*scale=*/0.4689698, point3(-0.3639, 1.7056, 0.0000)));

	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 56: Fandisk
// Classic CAD/mechanical-engineering test model with sharp creases, used
// in countless edge-preserving-smoothing papers.
// ============================================================================
inline hittable_list build_fandisk() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto gunmetal = make_shared<metal>(color(0.55, 0.56, 0.58), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"fandisk.obj", gunmetal,
		/*scale=*/0.5719733, point3(-1.3807, -7.2097, 0.7664)));

	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 57: Homer
// Homer Simpson bust - a fun, recognizable geometry-processing test model.
// ============================================================================
inline hittable_list build_homer() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto gold = make_shared<metal>(color(0.85, 0.70, 0.25), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"homer.obj", gold,
		/*scale=*/3.5671819, point3(-1.7818, -0.5565, -1.7568)));

	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 58: Igea
// Classical Italian bust (Igea, Roman goddess of health) - high-resolution
// scan, common geometry-processing test model.
// ============================================================================
inline hittable_list build_igea() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto silver = make_shared<metal>(color(0.85, 0.85, 0.88), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"igea.obj", silver,
		/*scale=*/30.0000000, point3(0.0000, 1.5000, 0.0000)));

	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 59: Max Planck
// Scanned bust of physicist Max Planck - a well-known geometry-processing
// test model.
// ============================================================================
inline hittable_list build_max_planck() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto aged_bronze = make_shared<metal>(color(0.65, 0.45, 0.30), 0.2);
	world.add(std::make_shared<triangle_mesh>(
		"max-planck.obj", aged_bronze,
		/*scale=*/0.0092252, point3(-0.2823, 1.6670, -0.7592)));

	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 60: Ogre
// Fantasy ogre head - stylized, chunky geometry, a fun creature scene.
// ============================================================================
inline hittable_list build_ogre() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto olive_metal = make_shared<metal>(color(0.45, 0.50, 0.35), 0.2);
	world.add(std::make_shared<triangle_mesh>(
		"ogre.obj", olive_metal,
		/*scale=*/0.0936388, point3(0.0000, 0.0007, 0.0557)));

	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 62: Crytek Sponza
// First "whole environment" mesh scene, not a single statue -- a complete
// architectural interior (262K triangles), the classic global-illumination
// benchmark scene. Departs from the scenes-37-61 convention in three ways:
//   1. No separate checkered-ground sphere -- the mesh's own floor is part
//      of the geometry.
//   2. Per-face materials come from the companion sponza.mtl via
//      load_obj_mtl(), including real map_Kd image textures (sampled via
//      the mesh's own UVs, models/sponza_textures/) where a material has
//      one, falling back to a flat Kd color and finally to a flat
//      sandstone lambertian for any face with no usemtl/unknown material/
//      missing texture.
//   3. Lit by an open-sky sky_light (like scene 24's build_hdri_sky)
//      instead of a single overhead point/area light -- Sponza's own
//      architecture is an open colonnade meant to be lit by daylight
//      filtering through the arches, not a single bulb.
//
// Scale/offset: kept at the raw OBJ's native scale (no normalization to a
// ~3-unit figure like the statue scenes -- this is a building, not an
// object on a table) with only a re-centering offset: raw bbox
// x=[-1920.95,1799.91] y=[-126.44,1429.43] z=[-1182.81,1105.43], offset by
// (60.52, 126.44, 38.69) to sit the (near-)floor at y=0 and center the plan
// on (x,z)=(0,0).
//
// Camera: unlike every other mesh scene here, a bbox-derived "just back up
// and look at the center" camera does NOT work for this one -- Sponza's
// footprint is mostly enclosed (side aisles under a solid roof), with only
// a narrow open-air central nave (roughly world x in [-1300,1300], z in
// [-235,235], no roof) letting the sky_light actually reach the interior.
// A handful of blind camera guesses all landed in solid geometry or fully
// enclosed side rooms and rendered pure black (no sky reachable within
// max_depth bounces, hence no light at all under sky-only illumination).
// Found the real position by a standalone Moller-Trumbore ray-triangle
// probe script (not committed -- one-off diagnostic) run directly against
// the raw OBJ data: cast rays straight down across an (x,z) grid to map
// roof-vs-open-sky, then cast horizontal rays from candidate points to
// confirm clear sightlines before ever spending a render cycle on them.
// The registry's camera ({-800,300,0} -> {800,300,0}) sits in the middle
// of that verified-open nave, looking down its ~2600-unit length -- the
// classic "columns receding down the corridor" Sponza shot.
// ============================================================================
// Memoized (magic-static, built at most once regardless of how many times
// build_sponza()/build_sponza_lights() are each called): this mesh is 262K
// triangles plus real texture decode, far too expensive to build twice per
// render just so build_world() and build_lights() can each get their own
// copy. Mirrors the same "shared build state behind separate world/lights
// accessors" pattern build_instanced_spheres_descriptor() (scene F3) already
// uses for its .pbrt BuildResult.
inline std::shared_ptr<triangle_mesh_mtl> sponza_mesh() {
	static const auto mesh = std::make_shared<triangle_mesh_mtl>(
		"sponza.obj", make_shared<lambertian>(color(0.80, 0.74, 0.62)),
		/*scale=*/1.0, point3(60.52, 126.44, 38.69),
		/*smooth_normals=*/false, "sponza_textures");
	return mesh;
}

inline hittable_list build_sponza() {
	hittable_list world;
	world.add(sponza_mesh());
	return world;
}

// Empty as of this writing -- sponza.mtl's Ke is zero for all 25 materials
// -- but wired for real so any future Sponza-like asset with genuine
// emissive materials (e.g. lamps) lights correctly with no further changes.
inline hittable_list build_sponza_lights() {
	return sponza_mesh()->lights();
}

inline std::shared_ptr<sky_light> build_sponza_sky() {
	// Brightened from the original (0.65,0.78,0.95): Sponza's interior is a
	// mostly-enclosed corridor reached only through a narrow roof opening
	// (see build_sponza()'s own comment), so even a same-magnitude sky as
	// scenes 63/64 leaves the corridor severely light-starved by geometric
	// occlusion alone - confirmed by comparison render, not guesswork.
	return std::make_shared<sky_light>(color(1.3, 1.56, 1.9));
}

// ============================================================================
// Scene 61: Rocker Arm
// Mechanical engine-part test model, elongated along Z after
// normalization (similar to the Beetle scenes).
// ============================================================================
inline hittable_list build_rocker_arm() {
	hittable_list world;

	auto checker = make_shared<checker_texture>(0.8, color(0.15,0.15,0.15), color(0.85,0.85,0.85));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	auto gunmetal = make_shared<metal>(color(0.55, 0.56, 0.58), 0.1);
	world.add(std::make_shared<triangle_mesh>(
		"rocker-arm.obj", gunmetal,
		/*scale=*/5.8365759, point3(0.0000, 1.5000, 0.0000)));

	// Area light - unlike scene 43's Utah Teapot, moving/brightening this
	// light did NOT change the bright patch on the now-visible boss tops
	// (tried y=20 + 8x brightness, re-rendered, same shape/extent) - this
	// confirmed it's a legitimate blown-out mirror-like specular highlight
	// off a flat, low-roughness surface that the camera fix revealed, not
	// the light sphere itself being in frame, so the standard light
	// placement stays unchanged here.
	world.add(make_shared<sphere>(point3(0, 8, 0), 2, make_shared<diffuse_light>(color(6,6,6))));
	return world;
}

// ============================================================================
// Scene 63: Amazon Lumberyard Bistro (Exterior)
// Second "whole environment" scene (see build_sponza()'s own comment for
// the shared design rationale: no separate ground, real per-face .mtl
// materials/textures via load_obj_mtl(), open-sky lighting instead of a
// point/area light). 2.84M triangles - a full
// outdoor street block (multiple buildings, a plaza/street network), not
// one enclosed building like Sponza.
//
// Scale/offset: native scale, only re-centered/floor-aligned: raw bbox
// x=[-3903.64,6956.37] y=[-472.62,2720.77] z=[-5496.06,6030.09], offset by
// (-1526.37, 472.62, -267.01).
//
// Camera: same problem as Sponza (a naive "back up and look at the
// center" framing risks landing inside a building or a fully-enclosed
// courtyard) but a DIFFERENT shape here - this is an open street block,
// not one solid building, so open ground-level space is actually common,
// not rare. Still verified with the same standalone ray-triangle probe
// script (not committed) before spending a render cycle: cast rays
// straight down across a coarse (x,z) grid to find street-level height
// (~496 world units, most points hit either roughly that height or a
// rooftop ~1400-2400 up, with several grid points missing geometry
// entirely - this scene has real open plazas), then horizontal rays from
// several street-level candidates to find one with a long, unobstructed
// sightline. (1500,700,2000) looking toward +X had a verified-clear
// 3000-unit sightline - a "walking down the street between buildings"
// shot, this scene's version of Sponza's "looking down the nave."
// ============================================================================
// Memoized for the same reason as sponza_mesh() above (2.84M triangles plus
// texture decode).
inline std::shared_ptr<triangle_mesh_mtl> bistro_exterior_mesh() {
	static const auto mesh = std::make_shared<triangle_mesh_mtl>(
		"bistro_exterior.obj", make_shared<lambertian>(color(0.75, 0.62, 0.50)),
		/*scale=*/1.0, point3(-1526.37, 472.62, -267.01),
		/*smooth_normals=*/false, "bistro_textures");
	return mesh;
}

inline hittable_list build_bistro_exterior() {
	hittable_list world;
	world.add(bistro_exterior_mesh());
	return world;
}

// Empty as of this writing -- exterior.mtl's Ke is zero across all 91
// materials, including one literally named "Spotlight_Emissive" -- but
// wired for real, same rationale as build_sponza_lights() above.
inline hittable_list build_bistro_exterior_lights() {
	return bistro_exterior_mesh()->lights();
}

inline std::shared_ptr<sky_light> build_bistro_exterior_sky() {
	return std::make_shared<sky_light>(color(0.55, 0.72, 0.95));
}

// ============================================================================
// Scene 64: Rungholt
// Third "whole environment" scene (see build_sponza()'s own comment for
// the shared design rationale). A giant blocky Minecraft-style town
// (6.7M triangles) rather than a real-world building/street. Uses
// load_obj_mtl() like Sponza/Bistro for real per-block Kd colors (grass,
// stone, wood, thatch, ...) instead of one flat tone -- still no textures
// (map_Kd), but since the geometry itself is inherently voxel-blocky, flat
// per-material colors alone already read as "textured" far more than they
// do on Sponza/Bistro's smoothly-curved surfaces.
//
// This mesh EXPOSED A REAL BUG in this codebase's shared OBJ loader
// (src/TheRestOfYourLife/mesh.h's load_obj()): rungholt.obj uses negative
// ("relative") face-vertex indices throughout, a valid part of the OBJ
// spec that the loader never handled -- every negative index silently
// resolved to a nonsense value and got dropped, which would have loaded
// this mesh with most of its geometry missing and no error at all. Fixed
// in load_obj() itself (now resolves negative indices against the
// vertex/uv/normal count parsed so far, per spec) rather than worked
// around here -- see tests/unit/obj_negative_indices_tests.cpp for the
// regression coverage.
//
// Scale/offset: none needed -- the raw OBJ is already centered near the
// origin with its floor already at y=0 (raw bbox x=[-327,328]
// y=[0,69] z=[-275,275]), unlike every other mesh scene in this file.
//
// Camera: unlike Sponza/Bistro, no street-level ray-probing was needed --
// Rungholt's low, sprawling footprint (69 units tall vs. 655x550 wide/deep)
// makes a simple elevated 3/4 overview safe by construction (nothing to
// get lost inside), verified with one quick ray-triangle probe (same
// standalone script used for Sponza/Bistro, not committed): camera at
// (400,300,400) looking toward the town center does hit the town's
// rooftops (~639 units away), not empty space -- a "whole village from
// above" establishing shot.
// ============================================================================
// Memoized for the same reason as sponza_mesh() above (6.7M triangles).
inline std::shared_ptr<triangle_mesh_mtl> rungholt_mesh() {
	static const auto mesh = std::make_shared<triangle_mesh_mtl>(
		"rungholt.obj", make_shared<lambertian>(color(0.62, 0.48, 0.34)),
		/*scale=*/1.0, point3(0.0, 0.0, 0.0));
	return mesh;
}

inline hittable_list build_rungholt() {
	hittable_list world;
	world.add(rungholt_mesh());
	return world;
}

// Empty -- rungholt.mtl has no Ke lines at all (not even zero-valued ones)
// -- but wired for real, same rationale as build_sponza_lights() above.
inline hittable_list build_rungholt_lights() {
	return rungholt_mesh()->lights();
}

inline std::shared_ptr<sky_light> build_rungholt_sky() {
	return std::make_shared<sky_light>(color(0.55, 0.72, 0.95));
}

// ============================================================================
// Scene 73: Fireplace Room
// Fourth "whole environment" mesh scene, same design rationale as scenes
// 62-64 (see build_sponza()'s own comment) -- real per-face .mtl materials
// and image textures (wood floor/furniture, framed pictures, foliage)
// loaded from models/fireplace_room_textures/, lit by an open sky, loaded
// from an external .obj file (requires models/fireplace_room.obj). Unlike
// 62-64, a small, human-scale FURNISHED INTERIOR (a living room with a
// fireplace) rather than a building-scale environment -- 22 materials,
// including real map_d alpha-cutout foliage (a potted-plant leaf material)
// and an illum-7 glass material for the windows/fireplace screen, so this
// exercises Phase 2/4's specular-dispatch and alpha-cutout work on a
// second, structurally different asset from Bistro's.
//
// Source: McGuire Computer Graphics Archive (casual-effects.com/data),
// CC BY 3.0, originally modeled for the "grey and white room" scene.
//
// Scale/offset: kept at the raw OBJ's native scale (already human-scale,
// no normalization needed) with only a re-centering offset: raw bbox
// x=[-0.513,5.123] y=[-0.003,2.879] z=[-3.635,0.599], offset by
// (-2.305, 0.003, 1.518) to sit the floor at y=0 and center the room's
// footprint on (x,z)=(0,0).
//
// Camera: an enclosed room, not an open colonnade/street (Sponza/Bistro's
// problem) or a low sprawling town (Rungholt's) -- placed at human eye
// height (1.6) just inside the room looking across it toward the fireplace
// wall, verified by CPU render (this room is small/simple enough that a
// straightforward "corner, eye height, look at center" placement didn't
// need the standalone ray-probe script the larger scenes required).
// ============================================================================
inline std::shared_ptr<triangle_mesh_mtl> fireplace_room_mesh() {
	static const auto mesh = std::make_shared<triangle_mesh_mtl>(
		"fireplace_room.obj", make_shared<lambertian>(color(0.55, 0.45, 0.35)),
		/*scale=*/1.0, point3(-2.305, 0.003, 1.518),
		/*smooth_normals=*/false, "fireplace_room_textures");
	return mesh;
}

inline hittable_list build_fireplace_room() {
	hittable_list world;
	world.add(fireplace_room_mesh());
	return world;
}

inline hittable_list build_fireplace_room_lights() {
	return fireplace_room_mesh()->lights();
}

inline std::shared_ptr<sky_light> build_fireplace_room_sky() {
	return std::make_shared<sky_light>(color(0.6, 0.75, 0.95));
}

// ============================================================================
// Scene 74: San Miguel
// Fifth "whole environment" mesh scene, same design rationale as scenes
// 62-64/73 (see build_sponza()'s own comment) -- real per-face .mtl
// materials and image textures (tile, wood, fabric, foliage) loaded from
// models/san_miguel_textures/, lit by an open sky, loaded from an external
// .obj file (requires models/san_miguel.obj). A dense Mexican hacienda
// courtyard/villa - the classic "hero" benchmark scene, comparable
// prestige/complexity to Bistro (9.9M triangles, heavy foliage). Real
// map_bump normal maps (roof beams, tree bark, water) and map_d
// alpha-cutout foliage throughout, plus a genuine illum-7 glass material
// (Ns 1000) -- exercises Phases 2-4 on a third, much larger asset.
//
// Source: McGuire Computer Graphics Archive (casual-effects.com/data),
// CC BY 3.0, modeled by Guillermo M. Leal Llaguno based on a real hacienda
// in San Miguel de Allende, Mexico; this 2017 revision by Morgan McGuire,
// Guedis Cardenas, and Michael Mara (Williams College) and Nicholas Hull
// (NVIDIA).
//
// Scale/offset: kept at the raw OBJ's native scale (a building-scale
// environment like Sponza/Bistro, not normalized) with only a re-centering
// offset: raw bbox x=[-22.274,46.774] y=[-0.463,14.600]
// z=[-12.042,14.937], offset by (-12.25, 0.463, -1.4475) to sit the
// (near-)floor at y=0 and center the plan on (x,z)=(0,0).
//
// Camera: a walled courtyard complex, not a single open colonnade
// (Sponza) -- found by direct CPU-render iteration rather than a
// standalone probe script (each render already costs ~60s regardless of
// resolution/SPP here, since building the BVH for 9.9M triangles dominates
// - a separate probe script would cost as much as just rendering), landing
// on a courtyard-level vantage point looking down a colonnade toward a
// fountain, arches, and potted foliage - the same "receding corridor" shot
// language as Sponza/Bistro's own cameras.
// ============================================================================
inline std::shared_ptr<triangle_mesh_mtl> san_miguel_mesh() {
	static const auto mesh = std::make_shared<triangle_mesh_mtl>(
		"san_miguel.obj", make_shared<lambertian>(color(0.75, 0.65, 0.55)),
		/*scale=*/1.0, point3(-12.25, 0.463, -1.4475),
		/*smooth_normals=*/false, "san_miguel_textures");
	return mesh;
}

inline hittable_list build_san_miguel() {
	hittable_list world;
	world.add(san_miguel_mesh());
	return world;
}

inline hittable_list build_san_miguel_lights() {
	return san_miguel_mesh()->lights();
}

// Brightened from a first-pass (0.6,0.75,0.95): like Sponza's own sky (see
// build_sponza_sky()'s comment), San Miguel's courtyard is mostly reached
// through covered colonnades/arches rather than direct open sky, leaving a
// same-magnitude sky severely light-starved by geometric occlusion alone -
// confirmed by an underexposed first-pass render, not guesswork.
inline std::shared_ptr<sky_light> build_san_miguel_sky() {
	return std::make_shared<sky_light>(color(1.4, 1.68, 2.0));
}

// ============================================================================
// Scene 75: Sibenik Cathedral
// Sixth "whole environment" mesh scene. A Gothic cathedral interior in
// Sibenik, Croatia - tall vaulted nave, stone columns, a rose window, and
// several colored stained-glass windows, with real per-face .mtl materials,
// image textures (kamen.png, KAMEN-stup.png, mramor6x6.png), and real
// map_Bump normal maps (kamen-bump.png, mramor6x6-bump.png) loaded from
// models/sibenik_cathedral_textures/. The stained-glass materials
// (staklo/staklo_zeleno/staklo_plavo/staklo_zuto, illum 6/4, Kd 0,0,0) are
// not mapped to dielectric by this codebase's illum dispatch (only illum 2/3
// -> metal and illum 7 -> dielectric are handled), so the colored glass
// renders as plain black window cutouts rather than tinted transparent
// glass - light still reaches the interior through the cathedral's open
// doorway/arches, confirmed by render, not guesswork.
//
// Source: McGuire Computer Graphics Archive (casual-effects.com/data),
// CC BY-NC 3.0 (non-commercial), originally modeled by Marko Dabrovic, with
// mesh holes corrected by Kenzie Lamar (Vicarious Visions) and high-
// resolution textures/bump maps painted by Morgan McGuire.
//
// Scale/offset: raw OBJ units, no rescale. Raw bbox x=[-20.14,20.14]
// y=[-15.31,15.30] z=[-8.50,8.50] - already centered on (x,z)=(0,0) by the
// original model, offset by (0, 15.3123, 0) to sit the floor at y=0.
//
// Camera: found by direct CPU-render iteration, landing on a nave-level
// vantage point looking down the long axis toward the far apse, columns
// receding on both sides.
// ============================================================================
inline std::shared_ptr<triangle_mesh_mtl> sibenik_cathedral_mesh() {
	static const auto mesh = std::make_shared<triangle_mesh_mtl>(
		"sibenik_cathedral.obj", make_shared<lambertian>(color(0.72, 0.71, 0.65)),
		/*scale=*/1.0, point3(0.0, 15.3123, 0.0),
		/*smooth_normals=*/false, "sibenik_cathedral_textures");
	return mesh;
}

inline hittable_list build_sibenik_cathedral() {
	hittable_list world;
	world.add(sibenik_cathedral_mesh());
	return world;
}

inline hittable_list build_sibenik_cathedral_lights() {
	return sibenik_cathedral_mesh()->lights();
}

// Brightened well past Sponza/San Miguel's own modest 1.4-2.3x boosts (see
// build_sponza_sky()'s comment) - confirmed by iterating from 1x through
// 10x, not guesswork. Even at 10x, most of the nave away from the window
// openings stays near-black: Sibenik's real window apertures are small
// relative to its stone-walled volume (unlike Sponza's open colonnade), so
// this is a physically plausible "shafts of light in an otherwise dim stone
// interior" result, not a bug - the same reason real cathedral photography
// looks like this. Settled on a middle value that properly exposes the
// window/column areas without blowing them out, rather than chasing an
// evenly-lit look this geometry doesn't physically support without adding
// a light source that isn't in the real asset.
inline std::shared_ptr<sky_light> build_sibenik_cathedral_sky() {
	return std::make_shared<sky_light>(color(4.5, 4.8, 5.2));
}

// ============================================================================
// Scene 76: Breakfast Room
// Seventh "whole environment" mesh scene. A cozy Blender-sourced dining/
// breakfast interior with glassware, table settings, and marble/tile
// textures. Every one of breakfast_room.mtl's 15 materials is tagged
// "illum 4" (this exporter's blanket default, not a per-material glass
// signal the way Bistro's real illum-7 materials were - confirmed by every
// material using it regardless of Kd/Ks, including plain paint and rubber),
// so illum 4 is deliberately left unmapped to dielectric here; only its
// map_Kd textures (picture3.jpg, tiles.png) exercise real coverage.
//
// Source: McGuire Computer Graphics Archive (casual-effects.com/data),
// CC BY 3.0.
//
// Scale/offset: raw OBJ units, no rescale. Raw bbox x=[-6.35,5.26]
// y=[-1.42,7.90] z=[-4.54,9.89], offset by (0.54, 1.42, -2.67) to sit the
// floor at y=0 and center the (x,z) plan on the origin.
//
// Camera: found by direct CPU-render iteration, landing on a seated-eye-
// height vantage point at the table looking across the room toward the
// window and artwork.
// ============================================================================
inline std::shared_ptr<triangle_mesh_mtl> breakfast_room_mesh() {
	static const auto mesh = std::make_shared<triangle_mesh_mtl>(
		"breakfast_room.obj", make_shared<lambertian>(color(0.6, 0.55, 0.5)),
		/*scale=*/1.0, point3(0.54, 1.42, -2.67),
		/*smooth_normals=*/false, "breakfast_room_textures");
	return mesh;
}

inline hittable_list build_breakfast_room() {
	hittable_list world;
	world.add(breakfast_room_mesh());
	return world;
}

inline hittable_list build_breakfast_room_lights() {
	return breakfast_room_mesh()->lights();
}

// Brightened from a first-pass (0.65,0.75,0.9), but only modestly: unlike
// Sibenik/Gallery's small apertures, Breakfast Room's window is a large
// glass wall panel with direct sky access, so a same-magnitude boost to
// their level (3.5-4.2) blew this scene out to solid white - confirmed by
// render, dialed back to the value below.
inline std::shared_ptr<sky_light> build_breakfast_room_sky() {
	return std::make_shared<sky_light>(color(1.1, 1.2, 1.35));
}

// ============================================================================
// Scene 77: Salle de Bain (bathroom)
// Eighth "whole environment" mesh scene. A tiled bathroom with a "Mirror"
// material (Kd 0,0,0, Ks 0.99, illum 3) and a "Light" material with a real
// Ke 10,10,10 - the second OBJ/.mtl asset (after Fireplace Room) to register
// a genuine Ke-emissive triangle as an NEE light, and the first to exercise
// illum 3 (extended alongside illum 2 into the glossy-metal dispatch branch
// specifically for this Mirror material - see mesh.h/scene_builder.cpp's
// specular-dispatch comment).
//
// Source: McGuire Computer Graphics Archive (casual-effects.com/data),
// CC BY 3.0, by Nacimus Prime (Blend Swap), ported by Benedikt Bitterli.
//
// Scale/offset: raw OBJ units, no rescale. Raw bbox x=[-17.05,16.90]
// y=[0.03,33.60] z=[-23.26,22.49], offset by (0.08, -0.03, 0.39) to sit the
// floor at y=0 and center the (x,z) plan on the origin.
//
// Camera: found by direct CPU-render iteration, landing on a vantage point
// framing the tub, mirror, and tiled wall.
// ============================================================================
inline std::shared_ptr<triangle_mesh_mtl> salle_de_bain_mesh() {
	static const auto mesh = std::make_shared<triangle_mesh_mtl>(
		"salle_de_bain.obj", make_shared<lambertian>(color(0.85, 0.85, 0.85)),
		/*scale=*/1.0, point3(0.08, -0.03, 0.39),
		/*smooth_normals=*/false, "salle_de_bain_textures");
	return mesh;
}

inline hittable_list build_salle_de_bain() {
	hittable_list world;
	world.add(salle_de_bain_mesh());
	return world;
}

inline hittable_list build_salle_de_bain_lights() {
	return salle_de_bain_mesh()->lights();
}

inline std::shared_ptr<sky_light> build_salle_de_bain_sky() {
	return std::make_shared<sky_light>(color(0.4, 0.45, 0.5));
}

// ============================================================================
// Scene 78: Gallery
// Ninth "whole environment" mesh scene. The Hallwyl Museum picture gallery
// in Stockholm - an ornate room of framed paintings, chandeliers, and
// parquet floor, all a single material sharing one large gallery.jpg texture
// referenced as both "map_Kd -bm 0.7 gallery.jpg" and
// "map_Ke -bm 0.3 gallery.jpg". The leading "-bm <value>" option before the
// filename was previously unhandled by parse_mtl_textures()/the GPU
// equivalent (both assumed no leading options, since Sponza/Bistro/Rungholt/
// Fireplace Room/San Miguel never had any) - fixed alongside this scene so
// the texture actually loads instead of silently falling back to flat Kd.
// map_Ke (an emissive *texture*, as opposed to the scalar Ke this codebase's
// Phase 1 already handles) remains unsupported - the gallery's real painted-
// canvas glow isn't reproduced, only its diffuse appearance under the scene's
// sky light.
//
// Source: McGuire Computer Graphics Archive (casual-effects.com/data),
// CC BY-SA 4.0.
//
// Scale/offset: raw OBJ units, no rescale. Raw bbox x=[-6.20,5.00]
// y=[0.06,6.26] z=[-14.04,11.38], offset by (0.60, -0.06, 1.33) to sit the
// floor at y=0 and center the (x,z) plan on the origin.
//
// Camera: found by direct CPU-render iteration, landing on a vantage point
// down the gallery's central axis framing the hung paintings on both walls.
// ============================================================================
inline std::shared_ptr<triangle_mesh_mtl> gallery_mesh() {
	static const auto mesh = std::make_shared<triangle_mesh_mtl>(
		"gallery.obj", make_shared<lambertian>(color(0.6, 0.55, 0.45)),
		/*scale=*/1.0, point3(0.60, -0.06, 1.33),
		/*smooth_normals=*/false, "gallery_textures");
	return mesh;
}

inline hittable_list build_gallery() {
	hittable_list world;
	world.add(gallery_mesh());
	return world;
}

inline hittable_list build_gallery_lights() {
	return gallery_mesh()->lights();
}

// Brightened from a first-pass (0.6,0.65,0.75) for the same reason as
// Breakfast Room/Sibenik above - the gallery is reached through a limited
// skylight/doorway rather than direct open sky.
inline std::shared_ptr<sky_light> build_gallery_sky() {
	return std::make_shared<sky_light>(color(3.0, 3.2, 3.6));
}

// ============================================================================
// Scene 79: Lost Empire
// Tenth "whole environment" mesh scene, and the first sourced from a
// Minecraft world export rather than a photogrammetry/CAD scan - a large
// half-buried ancient city (temple platforms, staircases, a lava chamber)
// exported from the "Vokselia" Minecraft world via Mineways. Blocky
// low-poly geometry, but at a genuinely large scale (165 units deep) that
// suits a long video flythrough far better than the single-room scenes
// above. Uses per-face .mtl materials and an image texture loaded from
// models/lost_empire_textures/ (requires models/lost_empire.obj).
//
// Source: McGuire Computer Graphics Archive (casual-effects.com/data),
// CC BY 3.0.
//
// Scale/offset: raw OBJ units, no rescale (same convention as Gallery
// above). Raw bbox x=[-37.10,38.12] y=[0,49.00] z=[-82.00,83.12], offset by
// (-0.51, 0, -0.56) to center the (x,z) plan on the origin - y is already
// floored at 0.
//
// Camera: found by direct CPU-render iteration, framing the temple's front
// staircase from ground level outside the entrance.
// ============================================================================
inline std::shared_ptr<triangle_mesh_mtl> lost_empire_mesh() {
	static const auto mesh = std::make_shared<triangle_mesh_mtl>(
		"lost_empire.obj", make_shared<lambertian>(color(0.6, 0.6, 0.6)),
		/*scale=*/1.0, point3(-0.51, 0.0, -0.56),
		/*smooth_normals=*/false, "lost_empire_textures");
	return mesh;
}

inline hittable_list build_lost_empire() {
	hittable_list world;
	world.add(lost_empire_mesh());
	return world;
}

inline hittable_list build_lost_empire_lights() {
	return lost_empire_mesh()->lights();
}

inline std::shared_ptr<sky_light> build_lost_empire_sky() {
	return std::make_shared<sky_light>(color(0.5, 0.6, 0.8));
}

// ============================================================================
// Scene 80: Vokselia Spawn
// Eleventh "whole environment" mesh scene - a small floating voxel island,
// also exported from Minecraft via Mineways (the same "Vokselia" world as
// Lost Empire above, from its spawn point rather than the buried city).
// Much smaller and flatter than every other environment scene here (under
// 4 units across), which makes it a good deliberately-different pace change
// for a video flythrough - a slow orbit around a small floating world
// rather than a long corridor traversal. Uses per-face .mtl materials and
// a single image texture loaded from models/vokselia_spawn_textures/
// (requires models/vokselia_spawn.obj).
//
// Source: McGuire Computer Graphics Archive (casual-effects.com/data),
// CC BY 3.0.
//
// Scale/offset: raw OBJ units, no rescale. Raw bbox x=[-1.93,1.93]
// y=[0,0.66] z=[-1.92,1.92] is already centered on (x,z) with y floored at
// 0, so no offset is needed.
//
// Camera: found by direct CPU-render iteration, pulled back far enough to
// frame the whole floating island against open sky.
// ============================================================================
inline std::shared_ptr<triangle_mesh_mtl> vokselia_spawn_mesh() {
	static const auto mesh = std::make_shared<triangle_mesh_mtl>(
		"vokselia_spawn.obj", make_shared<lambertian>(color(0.6, 0.6, 0.6)),
		/*scale=*/1.0, point3(0.0, 0.0, 0.0),
		/*smooth_normals=*/false, "vokselia_spawn_textures");
	return mesh;
}

inline hittable_list build_vokselia_spawn() {
	hittable_list world;
	world.add(vokselia_spawn_mesh());
	return world;
}

inline hittable_list build_vokselia_spawn_lights() {
	return vokselia_spawn_mesh()->lights();
}

inline std::shared_ptr<sky_light> build_vokselia_spawn_sky() {
	return std::make_shared<sky_light>(color(0.5, 0.6, 0.8));
}

// ============================================================================
// Scene 81: Power Plant
// Twelfth "whole environment" mesh scene, and by far the largest yet: a
// complete model of an actual coal-fired power plant (12.76M triangles,
// 5.98M vertices), originally released by UNC as 1,185 PLY files and merged
// into a single OBJ/.mtl pair by Morgan McGuire and Guedis Cardenas. Flat
// per-face .mtl colors only (66 materials, zero image textures), so this
// exercises pure geometric scale rather than any texturing path - already
// proven at comparable scale by San Miguel (H5: 9.93M triangles from a
// 1.14GB OBJ), which is what made attempting this scene practical rather
// than reckless.
//
// Source: McGuire Computer Graphics Archive (casual-effects.com/data),
// non-commercial use only (UNC), requires models/powerplant.obj.
//
// Scale/offset: the ONLY "whole environment" scene so far that needs a real
// rescale rather than the usual raw-units-as-is convention. The raw OBJ's
// own units span x=[-205000,406335] y=[0,249000] z=[-25646.6,160098] - a
// ~600,000-unit range, evidently the original CAD/engineering drawing's
// native units rather than anything scene-scale. scale=0.0004 (1/2500)
// brings that down to a ~245x100x74 unit structure, comparable in size to
// Lost Empire's own 165-unit video-flythrough scale. offset is applied
// AFTER scale (see triangle_mesh_mtl's positions.push_back(x*scale+offset.x,
// ...)), so it must already be in scaled units: offset=(-center_x*scale, 0,
// -center_z*scale) centers the plant on (x,z) with y floored at 0, exactly
// like every other whole-environment scene's own centering, just computed
// in the scaled frame instead of the raw one.
//
// Camera: found by direct CPU-render iteration, pulled back to frame the
// whole industrial complex.
// ============================================================================
inline std::shared_ptr<triangle_mesh_mtl> power_plant_mesh() {
	static const auto mesh = std::make_shared<triangle_mesh_mtl>(
		"powerplant.obj", make_shared<lambertian>(color(0.6, 0.6, 0.6)),
		/*scale=*/0.0004, point3(-40.267, 0.0, -26.8903),
		/*smooth_normals=*/false);
	return mesh;
}

inline hittable_list build_power_plant() {
	hittable_list world;
	world.add(power_plant_mesh());
	return world;
}

inline hittable_list build_power_plant_lights() {
	return power_plant_mesh()->lights();
}

inline std::shared_ptr<sky_light> build_power_plant_sky() {
	return std::make_shared<sky_light>(color(0.5, 0.6, 0.8));
}
