#pragma once
// scene_builder_mesh_gallery.h -- imported third-party mesh scenes for the
// GPU-recursive backend (Stanford models, Crytek Sponza, Amazon Lumberyard
// Bistro, and the rest of the external-asset gallery). Split out of
// scene_builder.cpp, which #includes this file directly into its own
// translation unit (NOT compiled separately - every function here is
// `static`, relying on internal linkage within scene_builder.cpp's own TU,
// and calls scene_builder.cpp's own anonymous-namespace helpers like
// add_lambertian()/load_obj_triangles_gpu()) at the point Stanford-onward
// scenes used to live; see that file's own header comment for the full
// picture. Mirrors src/TheRestOfYourLife/scenes_mesh_gallery.h's identical
// CPU-side split (hand-authored technique-showcase scenes vs. this imported
// gallery) - scene 8 (build_final_scene_gpu, the book's Next Week finale) is
// a hand-authored procedural scene that sits in this same numeric range but
// deliberately stayed in scene_builder.cpp itself, not moved here.

/// @brief Scene 38: Stanford Bunny. Matches CPU build_stanford_bunny()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly: same checkered ground
/// (now a real GPU checker texture, see add_checker_texture_gpu - unlike
/// scene 37's flat-gray ground approximation, added before this codebase
/// had any GPU texture support at all), same bronze metal material, same
/// scale/offset/light placement. The bunny geometry itself is loaded via
/// load_obj_triangles_gpu() from the same models/stanford-bunny.obj CPU
/// loads - 69,451 real triangles, not a procedural stand-in like scene 37's
/// icosahedron, so this is the actual GPU exercise of load_obj_triangles_gpu
/// against a large external asset.
static void build_stanford_bunny_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	// Bunny mesh, in polished bronze - matches CPU's exact scale/offset
	// (both computed from the raw OBJ's own bounding box, see CPU's
	// build_stanford_bunny() comment for the numbers).
	const int mat_bunny = add_metal(scene, make_float3(0.71f, 0.43f, 0.20f), 0.15f);
	load_obj_triangles_gpu(scene, "stanford-bunny.obj", mat_bunny,
		/*scale=*/19.4f, make_float3(0.3267f, -0.6398f, 0.0298f));

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 39: Stanford Armadillo. Matches CPU build_stanford_armadillo()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly: same checkered ground,
/// same gunmetal metal material, same scale/offset/light placement. The
/// armadillo geometry itself is loaded via load_obj_triangles_gpu() from
/// the same models/armadillo.obj CPU loads - 99,976 real triangles, same
/// "positions only, no vn/vt" situation as scene 38's bunny (confirmed via
/// grep - zero `vn` lines in the source file).
static void build_stanford_armadillo_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	// Armadillo mesh, in gunmetal - matches CPU's exact scale/offset (both
	// computed from the raw OBJ's own bounding box, see CPU's
	// build_stanford_armadillo() comment for the numbers).
	const int mat_armadillo = add_metal(scene, make_float3(0.55f, 0.56f, 0.58f), 0.08f);
	load_obj_triangles_gpu(scene, "armadillo.obj", mat_armadillo,
		/*scale=*/0.0198f, make_float3(0.0f, 1.0736f, 0.0f));

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 40: Stanford Happy Buddha. Matches CPU
/// build_stanford_happy_buddha() (src/TheRestOfYourLife/scenes_advanced.h)
/// exactly: same checkered ground, same polished-gold metal material, same
/// scale/offset/light placement. The buddha geometry itself is loaded via
/// load_obj_triangles_gpu() from the same models/happy-buddha.obj CPU
/// loads - 98,601 real triangles, same "positions only, no vn/vt" situation
/// as scenes 38/39 (confirmed via grep - zero `vn` lines in the source
/// file).
static void build_stanford_happy_buddha_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	// Buddha mesh, in polished gold - matches CPU's exact scale/offset (both
	// computed from the raw OBJ's own bounding box, see CPU's
	// build_stanford_happy_buddha() comment for the numbers).
	const int mat_buddha = add_metal(scene, make_float3(0.83f, 0.69f, 0.22f), 0.05f);
	load_obj_triangles_gpu(scene, "happy-buddha.obj", mat_buddha,
		/*scale=*/15.1496f, make_float3(0.0824f, -0.7539f, 0.1015f));

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 41: Stanford Lucy. Matches CPU build_stanford_lucy()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly: same checkered
/// ground, same bright-silver metal material, same scale/offset/light
/// placement. The Lucy geometry itself is loaded via
/// load_obj_triangles_gpu() from the same models/lucy.obj CPU loads -
/// 99,970 real triangles, same "positions only, no vn/vt" situation as
/// scenes 37-40 (confirmed via grep - zero `vn` lines in the source
/// file). That file was rotated to Y-up once, directly, before being
/// committed to this repo - see CPU's build_stanford_lucy() comment for
/// why (this mirror's lucy.obj ships Z-up) - so no rotation is needed
/// here either, same scale+translate-only transform as every other mesh
/// scene.
static void build_stanford_lucy_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	// Lucy mesh, in bright silver - matches CPU's exact scale/offset (both
	// computed from the raw OBJ's own bounding box, see CPU's
	// build_stanford_lucy() comment for the numbers).
	const int mat_lucy = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.1f);
	load_obj_triangles_gpu(scene, "lucy.obj", mat_lucy,
		/*scale=*/0.0018783f, make_float3(-1.2978f, 1.1380f, -0.2283f));

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 42: Stanford XYZRGB Dragon. Matches CPU build_stanford_dragon()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly: same checkered ground,
/// same bright-silver metal material, same scale/offset/light placement.
/// The dragon geometry itself is loaded via load_obj_triangles_gpu() from
/// the same models/xyzrgb_dragon.obj CPU loads - 249,882 real triangles,
/// same "positions only, no vn/vt" situation as scenes 37-41. Unlike Lucy,
/// this source file is already Y-up (confirmed via its own bounding box),
/// so no pre-rotation was needed - same scale+translate-only transform as
/// every other mesh scene.
static void build_stanford_dragon_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	// Dragon mesh, in bright silver - matches CPU's exact scale/offset (both
	// computed from the raw OBJ's own bounding box, see CPU's
	// build_stanford_dragon() comment for the numbers).
	const int mat_dragon = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.1f);
	load_obj_triangles_gpu(scene, "xyzrgb_dragon.obj", mat_dragon,
		/*scale=*/0.0267772f, make_float3(-0.0600f, 1.6803f, -0.2640f));

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 50: Glass Dragon. Matches CPU build_glass_dragon()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly: same mesh/scale/
/// offset/light placement as scene 42's dragon, but MaterialType::Dielectric
/// (IOR 1.5, matches CPU's dielectric(1.5)) instead of Metal. See CPU
/// build_glass_dragon()'s own comment for why neither the regular path
/// tracer NOR --sppm render this scene's dragon surface itself cleanly
/// (a genuinely hard case, not a bug) - this GPU builder only targets the
/// regular GPU path tracer; GPU SPPM is Phase-1-scoped to scene 11 only.
static void build_glass_dragon_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	// Dragon mesh, in clear glass - matches CPU's exact scale/offset (same
	// numbers as scene 42's metal dragon, see build_stanford_dragon_gpu's
	// comment).
	const int mat_dragon = add_dielectric(scene, 1.5f);
	load_obj_triangles_gpu(scene, "xyzrgb_dragon.obj", mat_dragon,
		/*scale=*/0.0267772f, make_float3(-0.0600f, 1.6803f, -0.2640f));

	// Area light
	const int mat_light2 = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light2{}; light2.center = make_float3(0.0f, 8.0f, 0.0f); light2.radius = 2.0f; light2.materialIdx = mat_light2;
	scene.spheres.push_back(light2);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 51: Beast. Matches CPU build_beast() exactly.
static void build_beast_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	const int mat_beast = add_metal(scene, make_float3(0.71f, 0.43f, 0.20f), 0.15f);
	load_obj_triangles_gpu(scene, "beast.obj", mat_beast,
		/*scale=*/0.0118956f, make_float3(0.0000f, 0.0103f, -0.3401f));

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 52: VW Beetle. Matches CPU build_beetle() exactly.
static void build_beetle_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	const int mat_beetle = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.08f);
	load_obj_triangles_gpu(scene, "beetle.obj", mat_beetle,
		/*scale=*/9.9009901f, make_float3(0.3614f, -3.0297f, -1.9010f));

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 54: Bimba. Matches CPU build_bimba() exactly.
static void build_bimba_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	const int mat_bimba = add_metal(scene, make_float3(0.83f, 0.69f, 0.22f), 0.05f);
	load_obj_triangles_gpu(scene, "bimba.obj", mat_bimba,
		/*scale=*/9.2592593f, make_float3(0.3333f, 2.2963f, 10.7454f));

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 55: Cow. Matches CPU build_cow() exactly.
static void build_cow_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	const int mat_cow = add_metal(scene, make_float3(0.80f, 0.65f, 0.28f), 0.15f);
	load_obj_triangles_gpu(scene, "cow.obj", mat_cow,
		/*scale=*/0.4689698f, make_float3(-0.3639f, 1.7056f, 0.0000f));

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 56: Fandisk. Matches CPU build_fandisk() exactly.
static void build_fandisk_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	const int mat_fandisk = add_metal(scene, make_float3(0.55f, 0.56f, 0.58f), 0.1f);
	load_obj_triangles_gpu(scene, "fandisk.obj", mat_fandisk,
		/*scale=*/0.5719733f, make_float3(-1.3807f, -7.2097f, 0.7664f));

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 57: Homer. Matches CPU build_homer() exactly.
static void build_homer_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	const int mat_homer = add_metal(scene, make_float3(0.85f, 0.70f, 0.25f), 0.1f);
	load_obj_triangles_gpu(scene, "homer.obj", mat_homer,
		/*scale=*/3.5671819f, make_float3(-1.7818f, -0.5565f, -1.7568f));

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 58: Igea. Matches CPU build_igea() exactly (unrotated mesh --
/// this scan's face points up rather than forward, compensated by the
/// registry's camera position, not a mesh transform -- see build_igea()'s
/// own comment).
static void build_igea_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	const int mat_igea = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.1f);
	load_obj_triangles_gpu(scene, "igea.obj", mat_igea,
		/*scale=*/30.0000000f, make_float3(0.0000f, 1.5000f, 0.0000f));

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 59: Max Planck. Matches CPU build_max_planck() exactly.
static void build_max_planck_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	const int mat_planck = add_metal(scene, make_float3(0.65f, 0.45f, 0.30f), 0.2f);
	load_obj_triangles_gpu(scene, "max-planck.obj", mat_planck,
		/*scale=*/0.0092252f, make_float3(-0.2823f, 1.6670f, -0.7592f));

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 60: Ogre. Matches CPU build_ogre() exactly.
static void build_ogre_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	const int mat_ogre = add_metal(scene, make_float3(0.45f, 0.50f, 0.35f), 0.2f);
	load_obj_triangles_gpu(scene, "ogre.obj", mat_ogre,
		/*scale=*/0.0936388f, make_float3(0.0000f, 0.0007f, 0.0557f));

	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 61: Rocker Arm. Matches CPU build_rocker_arm() exactly.
static void build_rocker_arm_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	const int mat_rocker = add_metal(scene, make_float3(0.55f, 0.56f, 0.58f), 0.1f);
	load_obj_triangles_gpu(scene, "rocker-arm.obj", mat_rocker,
		/*scale=*/5.8365759f, make_float3(0.0000f, 1.5000f, 0.0000f));

	// Area light - matches CPU build_rocker_arm(): unlike scene 43's Utah
	// Teapot, repositioning/brightening this light didn't change the bright
	// patch on the now-visible boss tops, confirming it's a legitimate
	// specular highlight off a flat surface, not the light itself in
	// frame - standard placement stays unchanged.
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 43: Utah Teapot. Matches CPU build_utah_teapot()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly: same checkered
/// ground, same bright-silver metal material, same scale/offset/light
/// placement. The teapot geometry itself is loaded via
/// load_obj_triangles_gpu() from the same models/teapot.obj CPU loads -
/// 6,320 real triangles, same "positions only, no vn/vt" situation as
/// scenes 37-42. Already Y-up and already sitting on y=0 (confirmed via
/// its own bounding box), so no rotation and no y-offset needed.
static void build_utah_teapot_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	// Teapot mesh, in bright silver - matches CPU's exact scale/offset (both
	// computed from the raw OBJ's own bounding box, see CPU's
	// build_utah_teapot() comment for the numbers).
	const int mat_teapot = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.1f);
	load_obj_triangles_gpu(scene, "teapot.obj", mat_teapot,
		/*scale=*/0.952381f, make_float3(-1.6352f, 0.0f, 0.0f));

	// Area light - raised to y=20, brightness scaled ~8x (matches CPU
	// build_utah_teapot() - see that function's comment: this scene's
	// raised/pulled-back camera brought the standard y=8 light into frame
	// as a blown-out disc; an x-only shift wasn't enough margin).
	const int mat_light = add_diffuse_light(scene, make_float3(48.0f, 48.0f, 48.0f));
	SphereData light{}; light.center = make_float3(0.0f, 20.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 44: Spot the Cow (Keenan Crane). Matches CPU build_spot_cow()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly: same checkered
/// ground, same bright-silver metal material, same scale/offset/light
/// placement. The cow geometry itself is loaded via load_obj_triangles_gpu()
/// from the same models/spot.obj CPU loads - 5,856 real triangles. Source
/// faces use "position/uv" indices (no vn), same "position index only"
/// situation load_obj_triangles_gpu() already handles for scenes 37-44's
/// other meshes. Confirmed Y-up (no rotation) via CPU's first render.
static void build_spot_cow_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	// Cow mesh, in bright silver - matches CPU's exact scale/offset (both
	// computed from the raw OBJ's own bounding box, see CPU's
	// build_spot_cow() comment for the numbers).
	const int mat_cow = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.1f);
	// flip_xz=true: matches CPU's rotate_y(mesh, 180) wrapper in
	// build_spot_cow() - the raw mesh faces away from the camera.
	load_obj_triangles_gpu(scene, "spot.obj", mat_cow,
		/*scale=*/1.7747f, make_float3(0.0f, 1.3076f, -0.3373f), /*flip_xz=*/true);

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 45: Suzanne (Blender's monkey-head mascot). Matches CPU
/// build_suzanne() (src/TheRestOfYourLife/scenes_advanced.h) exactly: same
/// checkered ground, same bright-silver metal material, same scale/offset/
/// light placement. The mesh itself is loaded via load_obj_triangles_gpu()
/// from the same models/suzanne.obj CPU loads - mostly quad faces (468 of
/// 500), fan-triangulated by the loader into 968 real triangles, same
/// mechanism already exercised by every other mesh scene's occasional
/// n-gon. Confirmed Y-up (no rotation) via CPU's first render.
static void build_suzanne_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	// Suzanne mesh, in bright silver - matches CPU's exact scale/offset
	// (both computed from the raw OBJ's own bounding box, see CPU's
	// build_suzanne() comment for the numbers).
	const int mat_suzanne = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.1f);
	load_obj_triangles_gpu(scene, "suzanne.obj", mat_suzanne,
		/*scale=*/1.52381f, make_float3(3.8005f, -0.4073f, -6.2536f));

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 46: Nefertiti Bust. Matches CPU build_nefertiti()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly: same checkered
/// ground, same bright-silver metal material, same scale/offset/light
/// placement. The bust geometry itself is loaded via load_obj_triangles_gpu()
/// from the same models/nefertiti.obj CPU loads - 99,938 real triangles,
/// no vn/vt data (flat-shaded, same as most other mesh scenes). That file
/// was rotated to Y-up once, directly, before being committed to this repo
/// - see CPU's build_nefertiti() comment for why (this mirror's
/// nefertiti.obj ships Z-up, same situation Lucy hit) - so no rotation is
/// needed here either, same scale+translate-only transform as every other
/// mesh scene.
static void build_nefertiti_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	// Nefertiti mesh, in bright silver - matches CPU's exact scale/offset
	// (both computed from the raw OBJ's own bounding box, see CPU's
	// build_nefertiti() comment for the numbers).
	const int mat_nefertiti = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.1f);
	load_obj_triangles_gpu(scene, "nefertiti.obj", mat_nefertiti,
		/*scale=*/0.0060654f, make_float3(-0.0001f, 1.4998f, -0.0002f));

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 47: Horse (classic geometry-processing test model - a
/// chess-knight-style head+neck bust, not a full body). Matches CPU
/// build_horse() (src/TheRestOfYourLife/scenes_advanced.h) exactly: same
/// checkered ground, same bright-silver metal material, same scale/offset/
/// light placement. The mesh itself is loaded via load_obj_triangles_gpu()
/// from the same models/horse.obj CPU loads - 96,966 real triangles, no
/// vn/vt data (flat-shaded). Confirmed Y-up (no rotation) via CPU's first
/// render.
static void build_horse_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	// Horse mesh, in bright silver - matches CPU's exact scale/offset (both
	// computed from the raw OBJ's own bounding box, see CPU's
	// build_horse() comment for the numbers).
	const int mat_horse = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.1f);
	// flip_xz=true: matches CPU's rotate_y(mesh, 180) wrapper in
	// build_horse() - the raw mesh faces away from the camera.
	load_obj_triangles_gpu(scene, "horse.obj", mat_horse,
		/*scale=*/16.36295f, make_float3(0.0f, 1.5f, 0.0f), /*flip_xz=*/true);

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 48: Cheburashka (beloved cartoon-character bust from Keenan
/// Crane's geometry-processing course). Matches CPU build_cheburashka()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly: same checkered
/// ground, same bright-silver metal material, same scale/offset/light
/// placement. The mesh itself is loaded via load_obj_triangles_gpu() from
/// the same models/cheburashka.obj CPU loads - 13,334 real triangles, no
/// vn/vt data (flat-shaded). Confirmed Y-up (no rotation) via CPU's first
/// render.
static void build_cheburashka_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	// Cheburashka mesh, in bright silver - matches CPU's exact scale/offset
	// (both computed from the raw OBJ's own bounding box, see CPU's
	// build_cheburashka() comment for the numbers).
	const int mat_cheb = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.1f);
	load_obj_triangles_gpu(scene, "cheburashka.obj", mat_cheb,
		/*scale=*/3.5648929f, make_float3(-1.7824465f, -0.2824465f, -1.7824465f));

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}

/// @brief Scene 49: Trophy Room. Matches CPU build_trophy_room()
/// (src/TheRestOfYourLife/scenes_advanced.h) exactly: same four meshes
/// (bunny/teapot/Suzanne/Spot the Cow) at the same shrunk-and-shifted
/// scale/offset, same four metal tones (bronze/chrome/gold/gunmetal), same
/// checkered ground and light. First scene to place multiple external
/// meshes in one composition - see CPU's comment for why this one uses
/// four opaque metals rather than glass (an earlier dielectric-on-mesh
/// attempt produced non-converging noise, flagged separately).
static void build_trophy_room_gpu(SceneData& scene) {
	// Ground
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	const int checkerTexIdx = add_checker_texture_gpu(scene, 0.8f,
		make_float3(0.15f, 0.15f, 0.15f), make_float3(0.85f, 0.85f, 0.85f));
	add_lambertian(scene, make_float3(1.0f, 1.0f, 1.0f), checkerTexIdx);
	SphereData ground = make_ground_sphere_1000(mat_ground);
	scene.spheres.push_back(ground);

	// Bunny (bronze) - matches CPU's shrunk scale/offset exactly.
	const int mat_bronze = add_metal(scene, make_float3(0.71f, 0.43f, 0.20f), 0.15f);
	load_obj_triangles_gpu(scene, "stanford-bunny.obj", mat_bronze,
		/*scale=*/10.3467f, make_float3(-3.32576f, -0.34123f, 0.01589f));

	// Teapot (chrome)
	const int mat_chrome = add_metal(scene, make_float3(0.85f, 0.85f, 0.88f), 0.10f);
	load_obj_triangles_gpu(scene, "teapot.obj", mat_chrome,
		/*scale=*/0.50794f, make_float3(-2.07211f, 0.0f, 0.0f));

	// Suzanne (gold). y lowered by 0.14 from the pure-scaled value, matching
	// CPU's build_trophy_room() - see that function's comment for why
	// (Suzanne is a headless-body-free mesh whose grounded chin puts its
	// eyes above this scene's shared shelf camera aim height).
	const int mat_gold = add_metal(scene, make_float3(0.83f, 0.69f, 0.22f), 0.05f);
	load_obj_triangles_gpu(scene, "suzanne.obj", mat_gold,
		/*scale=*/0.81270f, make_float3(3.22694f, -0.35723f, -3.33526f));

	// Spot the Cow (gunmetal). flip_xz=true: matches CPU's rotate_y(180)
	// fix in build_trophy_room() - the raw mesh faces away from the camera.
	// The flip in load_obj_triangles_gpu() is applied to raw*scale BEFORE
	// offset is added (same effective order as CPU's build-then-rotate-
	// then-translate composition), so this is correct even with the
	// nonzero shelf x-shift baked into this offset.
	const int mat_gunmetal = add_metal(scene, make_float3(0.55f, 0.56f, 0.58f), 0.08f);
	load_obj_triangles_gpu(scene, "spot.obj", mat_gunmetal,
		/*scale=*/0.94651f, make_float3(3.5f, 0.69739f, -0.17989f), /*flip_xz=*/true);

	// Area light
	const int mat_light = add_diffuse_light(scene, make_float3(6.0f, 6.0f, 6.0f));
	SphereData light{}; light.center = make_float3(0.0f, 8.0f, 0.0f); light.radius = 2.0f; light.materialIdx = mat_light;
	scene.spheres.push_back(light);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.lightKinds.push_back(GpuLightKind::Sphere);
}
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
