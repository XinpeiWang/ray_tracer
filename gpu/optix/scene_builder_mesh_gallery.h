#pragma once
// scene_builder_mesh_gallery.h -- imported third-party environment scenes for
// the GPU-recursive backend (Crytek Sponza, Amazon Lumberyard Bistro, and the
// rest of the .mtl-textured external-asset gallery). Split out of
// scene_builder.cpp, which #includes this file directly into its own
// translation unit (NOT compiled separately - every function here is
// `static`, relying on internal linkage within scene_builder.cpp's own TU,
// and calls scene_builder.cpp's own anonymous-namespace helpers like
// add_lambertian()/load_obj_triangles_mtl_gpu()). Mirrors
// src/TheRestOfYourLife/scenes_mesh_gallery.h's identical CPU-side split.
//
// The single-/multi-mesh statue scenes that used to open this file (scenes
// 38-61: Stanford scans, teapot, Spot, Suzanne, ... Rocker Arm) migrated to
// pbrt-backed scenes (pbrt_scenes/mesh-*.pbrt) and were deleted; what remains
// are the 12 environment scenes whose per-face .mtl materials and map_Kd
// textures the pbrt loader does not import yet.

/// @brief Scene 62: Crytek Sponza. Matches CPU build_sponza() exactly: no
/// separate ground sphere (the mesh's own floor is part of the geometry),
/// real per-face textures sampled via the mesh's own UVs from the companion
/// sponza.mtl's map_Kd images (models/sponza_textures/, falling back to a
/// flat sandstone lambertian for any face with no usemtl/unknown material/
/// missing texture), no explicit light source object -- lit purely via the
/// flat-color "sky" set on GpuCameraParams::backgroundColor at the case-62
/// dispatch site below (GPU has no sky_light/infinite-light object the way
/// the CPU registry's build_sky field does -- background color IS the GPU
/// sky, same convention as scene 24's HDRI Sky).
static void build_sponza_gpu(SceneData& scene) {
	const int mat_stone = add_lambertian(scene, make_float3(0.80f, 0.74f, 0.62f));
	load_obj_triangles_mtl_gpu(scene, "sponza.obj", mat_stone,
		/*scale=*/1.0f, make_float3(60.52f, 126.44f, 38.69f), "sponza_textures");
}

/// @brief Scene 63: Amazon Lumberyard Bistro (Exterior). Matches CPU
/// build_bistro_exterior() exactly. See build_sponza_gpu()'s own comment
/// for the shared design rationale (real per-face textures via map_Kd, sky
/// via backgroundColor). 2.84M triangles -- the largest mesh in this
/// codebase.
static void build_bistro_exterior_gpu(SceneData& scene) {
	const int mat_plaster = add_lambertian(scene, make_float3(0.75f, 0.62f, 0.50f));
	load_obj_triangles_mtl_gpu(scene, "bistro_exterior.obj", mat_plaster,
		/*scale=*/1.0f, make_float3(-1526.37f, 472.62f, -267.01f), "bistro_textures");
}

/// @brief Scene 64: Rungholt. Matches CPU build_rungholt() exactly. See
/// build_sponza_gpu()'s comment for the shared design rationale, and CPU
/// build_rungholt()'s own comment for the real negative-face-index OBJ
/// loader bug this mesh exposed (fixed in load_obj_triangles_gpu()'s
/// underlying parser the same way as the CPU loader -- see that function;
/// load_obj_triangles_mtl_gpu() shares the same fix).
static void build_rungholt_gpu(SceneData& scene) {
	const int mat_wood = add_lambertian(scene, make_float3(0.62f, 0.48f, 0.34f));
	load_obj_triangles_mtl_gpu(scene, "rungholt.obj", mat_wood,
		/*scale=*/1.0f, make_float3(0.0f, 0.0f, 0.0f));
}

/// @brief Scene 73: Fireplace Room. Matches CPU build_fireplace_room()
/// exactly. See build_sponza_gpu()'s own comment for the shared design
/// rationale (real per-face textures via map_Kd, sky via backgroundColor).
/// A small furnished interior rather than a building-scale environment.
static void build_fireplace_room_gpu(SceneData& scene) {
	const int mat_wood = add_lambertian(scene, make_float3(0.55f, 0.45f, 0.35f));
	load_obj_triangles_mtl_gpu(scene, "fireplace_room.obj", mat_wood,
		/*scale=*/1.0f, make_float3(-2.305f, 0.003f, 1.518f), "fireplace_room_textures");
}

/// @brief Scene 74: San Miguel. Matches CPU build_san_miguel() exactly.
/// See build_sponza_gpu()'s own comment for the shared design rationale.
/// 9.9M triangles -- the largest mesh in this codebase (Bistro was
/// previously the largest at 2.84M).
static void build_san_miguel_gpu(SceneData& scene) {
	const int mat_adobe = add_lambertian(scene, make_float3(0.75f, 0.65f, 0.55f));
	load_obj_triangles_mtl_gpu(scene, "san_miguel.obj", mat_adobe,
		/*scale=*/1.0f, make_float3(-12.25f, 0.463f, -1.4475f), "san_miguel_textures");
}

/// @brief Scene 75: Sibenik Cathedral. Matches CPU build_sibenik_cathedral()
/// exactly. See build_sponza_gpu()'s own comment for the shared design
/// rationale.
static void build_sibenik_cathedral_gpu(SceneData& scene) {
	const int mat_stone = add_lambertian(scene, make_float3(0.72f, 0.71f, 0.65f));
	load_obj_triangles_mtl_gpu(scene, "sibenik_cathedral.obj", mat_stone,
		/*scale=*/1.0f, make_float3(0.0f, 15.3123f, 0.0f), "sibenik_cathedral_textures");
}

/// @brief Scene 76: Breakfast Room. Matches CPU build_breakfast_room()
/// exactly. See build_sponza_gpu()'s own comment for the shared design
/// rationale.
static void build_breakfast_room_gpu(SceneData& scene) {
	const int mat_room = add_lambertian(scene, make_float3(0.6f, 0.55f, 0.5f));
	load_obj_triangles_mtl_gpu(scene, "breakfast_room.obj", mat_room,
		/*scale=*/1.0f, make_float3(0.54f, 1.42f, -2.67f), "breakfast_room_textures");
}

/// @brief Scene 77: Salle de Bain. Matches CPU build_salle_de_bain()
/// exactly. See build_sponza_gpu()'s own comment for the shared design
/// rationale. Real Ke on the "Light" material -- second OBJ/.mtl asset
/// (after Fireplace Room) to register a genuine Ke-emissive triangle as a
/// GpuLightKind::Triangle light, exercising the wavefront NEE fix a second
/// time.
static void build_salle_de_bain_gpu(SceneData& scene) {
	const int mat_room = add_lambertian(scene, make_float3(0.85f, 0.85f, 0.85f));
	load_obj_triangles_mtl_gpu(scene, "salle_de_bain.obj", mat_room,
		/*scale=*/1.0f, make_float3(0.08f, -0.03f, 0.39f), "salle_de_bain_textures");
}

/// @brief Scene 78: Gallery. Matches CPU build_gallery() exactly. See
/// build_sponza_gpu()'s own comment for the shared design rationale.
static void build_gallery_gpu(SceneData& scene) {
	const int mat_room = add_lambertian(scene, make_float3(0.6f, 0.55f, 0.45f));
	load_obj_triangles_mtl_gpu(scene, "gallery.obj", mat_room,
		/*scale=*/1.0f, make_float3(0.60f, -0.06f, 1.33f), "gallery_textures");
}

/// @brief Scene 79: Lost Empire. Matches CPU build_lost_empire() exactly. See
/// build_sponza_gpu()'s own comment for the shared design rationale.
static void build_lost_empire_gpu(SceneData& scene) {
	const int mat_room = add_lambertian(scene, make_float3(0.6f, 0.6f, 0.6f));
	load_obj_triangles_mtl_gpu(scene, "lost_empire.obj", mat_room,
		/*scale=*/1.0f, make_float3(-0.51f, 0.0f, -0.56f), "lost_empire_textures");
}

/// @brief Scene 80: Vokselia Spawn. Matches CPU build_vokselia_spawn()
/// exactly. See build_sponza_gpu()'s own comment for the shared design
/// rationale.
static void build_vokselia_spawn_gpu(SceneData& scene) {
	const int mat_room = add_lambertian(scene, make_float3(0.6f, 0.6f, 0.6f));
	load_obj_triangles_mtl_gpu(scene, "vokselia_spawn.obj", mat_room,
		/*scale=*/1.0f, make_float3(0.0f, 0.0f, 0.0f), "vokselia_spawn_textures");
}

/// @brief Scene 81: Power Plant. Matches CPU build_power_plant() exactly. See
/// build_sponza_gpu()'s own comment for the shared design rationale. No
/// textureDir - this .mtl has flat per-face colors only, zero image
/// textures (see build_power_plant()'s own comment).
static void build_power_plant_gpu(SceneData& scene) {
	const int mat_room = add_lambertian(scene, make_float3(0.6f, 0.6f, 0.6f));
	load_obj_triangles_mtl_gpu(scene, "powerplant.obj", mat_room,
		/*scale=*/0.0004f, make_float3(-40.267f, 0.0f, -26.8903f));
}
