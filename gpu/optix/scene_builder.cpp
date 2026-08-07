// Scene Builder Implementation
// Converts shared scene definitions to OptiX geometry

#include "scene_builder.h"
#include "optix_math_helpers.h"
#include "../../src/shared/scene_descriptor.h"
#include <cmath>
#include <cassert>
#include <iostream>

#include "../../src/shared/conductor_data.h"

namespace {
	// Constants for Cornell Box dimensions
	constexpr float kBoxSize = 555.0f;
	constexpr float kLightIntensity = 15.0f;
	constexpr float kGlassIOR = 1.5f;

	// Helper to safely cast vector size to int with bounds checking
	inline int safe_cast_to_int(size_t value) {
		assert(value <= static_cast<size_t>(INT_MAX) && "Material index overflow");
		return static_cast<int>(value);
	}

	// Helper to rotate a point around Y axis
	inline float3 rotate_y(const float3& p, float angle_degrees) {
		const float radians = angle_degrees * (3.14159265358979323846f / 180.0f);
		const float cos_theta = std::cos(radians);
		const float sin_theta = std::sin(radians);
		return make_float3(
			cos_theta * p.x + sin_theta * p.z,
			p.y,
			-sin_theta * p.x + cos_theta * p.z
		);
	}

	// Helper to translate a point
	inline float3 translate(const float3& p, const float3& offset) {
		return make_float3(p.x + offset.x, p.y + offset.y, p.z + offset.z);
	}

	// Helper to check if a material is emissive
	inline bool is_emissive(const SceneData& scene, int material_idx) {
		if (material_idx < 0 || material_idx >= static_cast<int>(scene.materials.size()))
			return false;
		const auto& mat = scene.materials[material_idx];
		return mat.type == MaterialType::DiffuseLight;
	}

	// Helper to add a quad with optional rotation and translation
	inline void add_transformed_quad(
		SceneData& scene,
		const float3& Q,
		const float3& u,
		const float3& v,
		int material_idx,
		float rotation_y_degrees = 0.0f,
		const float3& translation = make_float3(0, 0, 0))
	{
		// Apply rotation first, then translation (matching CPU transform order)
		float3 Q_transformed = Q;
		float3 u_transformed = u;
		float3 v_transformed = v;

		if (rotation_y_degrees != 0.0f) {
			Q_transformed = rotate_y(Q, rotation_y_degrees);
			u_transformed = rotate_y(u, rotation_y_degrees);
			v_transformed = rotate_y(v, rotation_y_degrees);
		}

		Q_transformed = translate(Q_transformed, translation);

		// Build the quad
		QuadData quad{};
		quad.Q = Q_transformed;
		quad.u = u_transformed;
		quad.v = v_transformed;
		const float3 quad_cross = cross(quad.u, quad.v);
		quad.w = quad_cross;
		quad.normal = normalize(quad_cross);
		quad.D = dot(quad.normal, quad.Q);
		quad.materialIdx = material_idx;
		scene.quads.push_back(quad);

		// Track if this quad is a light
		if (is_emissive(scene, material_idx)) {
			scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
			scene.isLightSphere.push_back(false); // false = quad
		}
	}

	// Helper to add a box (6 quads) with rotation and translation
	// Matches CPU box() function from src/TheRestOfYourLife/quad.h
	inline void add_box(
		SceneData& scene,
		const float3& corner_a,
		const float3& corner_b,
		int material_idx,
		float rotation_y_degrees = 0.0f,
		const float3& translation = make_float3(0, 0, 0))
	{
		// Construct min and max corners
		const float3 min_corner = make_float3(
			fminf(corner_a.x, corner_b.x),
			fminf(corner_a.y, corner_b.y),
			fminf(corner_a.z, corner_b.z)
		);
		const float3 max_corner = make_float3(
			fmaxf(corner_a.x, corner_b.x),
			fmaxf(corner_a.y, corner_b.y),
			fmaxf(corner_a.z, corner_b.z)
		);

		const float3 dx = make_float3(max_corner.x - min_corner.x, 0, 0);
		const float3 dy = make_float3(0, max_corner.y - min_corner.y, 0);
		const float3 dz = make_float3(0, 0, max_corner.z - min_corner.z);

		// Six faces matching CPU box() in quad.h:
		// Front face (min.x, min.y, max.z)
		add_transformed_quad(scene, make_float3(min_corner.x, min_corner.y, max_corner.z), dx, dy, material_idx, rotation_y_degrees, translation);
		// Right face (max.x, min.y, max.z)
		add_transformed_quad(scene, make_float3(max_corner.x, min_corner.y, max_corner.z), make_float3(-dz.z, 0, 0), dy, material_idx, rotation_y_degrees, translation);
		// Back face (max.x, min.y, min.z)
		add_transformed_quad(scene, make_float3(max_corner.x, min_corner.y, min_corner.z), make_float3(-dx.x, 0, 0), dy, material_idx, rotation_y_degrees, translation);
		// Left face (min.x, min.y, min.z)
		add_transformed_quad(scene, make_float3(min_corner.x, min_corner.y, min_corner.z), dz, dy, material_idx, rotation_y_degrees, translation);
		// Top face (min.x, max.y, max.z)
		add_transformed_quad(scene, make_float3(min_corner.x, max_corner.y, max_corner.z), dx, make_float3(0, 0, -dz.z), material_idx, rotation_y_degrees, translation);
		// Bottom face (min.x, min.y, min.z)
		add_transformed_quad(scene, make_float3(min_corner.x, min_corner.y, min_corner.z), dx, dz, material_idx, rotation_y_degrees, translation);
	}
}

/// @brief Build the Cornell Box scene with box primitive
/// @param scene Output scene data container
static void build_cornell_box(SceneData& scene) {
	// Materials
	const int mat_red = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({
		MaterialType::Lambertian,
		make_float3(0.65f, 0.05f, 0.05f),  // albedo - red diffuse
		0.0f, 0.0f,  // fuzz, ior (unused for lambertian)
		make_float3(0.0f, 0.0f, 0.0f)  // emission
	});

	const int mat_white = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({
		MaterialType::Lambertian,
		make_float3(0.73f, 0.73f, 0.73f),  // albedo - white diffuse
		0.0f, 0.0f,
		make_float3(0.0f, 0.0f, 0.0f)
	});

	const int mat_green = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({
		MaterialType::Lambertian,
		make_float3(0.12f, 0.45f, 0.15f),  // albedo - green diffuse
		0.0f, 0.0f,
		make_float3(0.0f, 0.0f, 0.0f)
	});

	const int mat_light = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({
		MaterialType::DiffuseLight,
		make_float3(0.0f, 0.0f, 0.0f),  // albedo (unused for emissive)
		0.0f, 0.0f,
		make_float3(kLightIntensity, kLightIntensity, kLightIntensity)  // emission
	});

	const int mat_glass = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({
		MaterialType::Dielectric,
		make_float3(1.0f, 1.0f, 1.0f),  // albedo (unused for dielectric)
		0.0f,
		kGlassIOR,  // index of refraction
		make_float3(0.0f, 0.0f, 0.0f)
	});

	// Cornell Box quads (walls, floor, ceiling)
	// Green wall (right - +X face at x=555)
	QuadData wall_right{};
	wall_right.Q = make_float3(kBoxSize, 0.0f, 0.0f);
	wall_right.u = make_float3(0.0f, 0.0f, kBoxSize);
	wall_right.v = make_float3(0.0f, kBoxSize, 0.0f);
	const float3 wall_right_cross = cross(wall_right.u, wall_right.v);
	wall_right.w = wall_right_cross;
	wall_right.normal = normalize(wall_right_cross);
	wall_right.D = dot(wall_right.normal, wall_right.Q);
	wall_right.materialIdx = mat_green;
	scene.quads.push_back(wall_right);
	if (is_emissive(scene, mat_green)) {
		scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
		scene.isLightSphere.push_back(false);
	}

	// Red wall (left - origin face at x=0)
	QuadData wall_left{};
	wall_left.Q = make_float3(0.0f, 0.0f, kBoxSize);
	wall_left.u = make_float3(0.0f, 0.0f, -kBoxSize);
	wall_left.v = make_float3(0.0f, kBoxSize, 0.0f);
	const float3 wall_left_cross = cross(wall_left.u, wall_left.v);
	wall_left.w = wall_left_cross;
	wall_left.normal = normalize(wall_left_cross);
	wall_left.D = dot(wall_left.normal, wall_left.Q);
	wall_left.materialIdx = mat_red;
	scene.quads.push_back(wall_left);
	if (is_emissive(scene, mat_red)) {
		scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
		scene.isLightSphere.push_back(false);
	}

	// Light (matching CPU: Q=(213,554,227), u=(130,0,0), v=(0,0,105))
	QuadData light_quad{};
	light_quad.Q = make_float3(213.0f, 554.0f, 227.0f);
	light_quad.u = make_float3(130.0f, 0.0f, 0.0f);
	light_quad.v = make_float3(0.0f, 0.0f, 105.0f);
	const float3 light_cross = cross(light_quad.u, light_quad.v);
	light_quad.w = light_cross;
	light_quad.normal = normalize(light_cross);
	light_quad.D = dot(light_quad.normal, light_quad.Q);
	light_quad.materialIdx = mat_light;
	scene.quads.push_back(light_quad);

	// Track this light for MIS
	scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
	scene.isLightSphere.push_back(false); // false = quad

	// White ceiling (+Y face at y=555)
	QuadData ceiling{};
	ceiling.Q = make_float3(0.0f, kBoxSize, 0.0f);
	ceiling.u = make_float3(kBoxSize, 0.0f, 0.0f);
	ceiling.v = make_float3(0.0f, 0.0f, kBoxSize);
	const float3 ceiling_cross = cross(ceiling.u, ceiling.v);
	ceiling.w = ceiling_cross;
	ceiling.normal = normalize(ceiling_cross);
	ceiling.D = dot(ceiling.normal, ceiling.Q);
	ceiling.materialIdx = mat_white;
	scene.quads.push_back(ceiling);
	if (is_emissive(scene, mat_white)) {
		scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
		scene.isLightSphere.push_back(false);
	}

	// White floor (origin, XZ plane at y=0)
	QuadData floor{};
	floor.Q = make_float3(0.0f, 0.0f, kBoxSize);
	floor.u = make_float3(kBoxSize, 0.0f, 0.0f);
	floor.v = make_float3(0.0f, 0.0f, -kBoxSize);
	const float3 floor_cross = cross(floor.u, floor.v);
	floor.w = floor_cross;
	floor.normal = normalize(floor_cross);
	floor.D = dot(floor.normal, floor.Q);
	floor.materialIdx = mat_white;
	scene.quads.push_back(floor);
	if (is_emissive(scene, mat_white)) {
		scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
		scene.isLightSphere.push_back(false);
	}

	// White back wall (+Z face at z=555)
	QuadData back_wall{};
	back_wall.Q = make_float3(kBoxSize, 0.0f, kBoxSize);
	back_wall.u = make_float3(-kBoxSize, 0.0f, 0.0f);
	back_wall.v = make_float3(0.0f, kBoxSize, 0.0f);
	const float3 back_cross = cross(back_wall.u, back_wall.v);
	back_wall.w = back_cross;
	back_wall.normal = normalize(back_cross);
	back_wall.D = dot(back_wall.normal, back_wall.Q);
	back_wall.materialIdx = mat_white;
	scene.quads.push_back(back_wall);
	if (is_emissive(scene, mat_white)) {
		scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
		scene.isLightSphere.push_back(false);
	}

	// Glass sphere (left)
	SphereData glass_sphere{};
	glass_sphere.center = make_float3(190.0f, 90.0f, 190.0f);
	glass_sphere.radius = 90.0f;
	glass_sphere.materialIdx = mat_glass;
	scene.spheres.push_back(glass_sphere);

	// Track if this sphere is a light (glass is not)
	if (is_emissive(scene, mat_glass)) {
		scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
		scene.isLightSphere.push_back(true); // true = sphere
	}

	// White rotated box (right) - matching CPU scene definition
	// box(point3(0,0,0), point3(165,330,165), white) rotated 15° then translated (265,0,295)
	add_box(scene,
		make_float3(0.0f, 0.0f, 0.0f),
		make_float3(165.0f, 330.0f, 165.0f),
		mat_white,
		15.0f,  // rotation angle in degrees
		make_float3(265.0f, 0.0f, 295.0f));  // translation offset
}

/// @brief Build Rough Metal Spheres scene (scene 9)
/// Matches CPU build_rough_metal_spheres(): large ground sphere, 5 rough-metal
/// spheres with roughness 0.05..0.8, and a large quad area light above.
static void build_rough_metal_spheres(SceneData& scene) {
    // Ground (large dark-grey Lambertian sphere)
    const int mat_ground = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.2f, 0.2f, 0.2f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    // Area light quad material
    constexpr float kRMSLightIntensity = 6.0f;
    const int mat_light = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f,
                                 make_float3(kRMSLightIntensity, kRMSLightIntensity, kRMSLightIntensity) });

    // Five rough-metal sphere materials: roughness 0.05, 0.2, 0.4, 0.6, 0.8
    const float roughnesses[5] = { 0.05f, 0.2f, 0.4f, 0.6f, 0.8f };
    int mat_metal[5];
    for (int i = 0; i < 5; ++i) {
        mat_metal[i] = safe_cast_to_int(scene.materials.size());
        scene.materials.push_back({ MaterialType::Metal, make_float3(0.95f, 0.85f, 0.55f),
                                     roughnesses[i], 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
    }

    // Ground sphere: center (0,-1000,0), radius 1000
    SphereData ground{};
    ground.center = make_float3(0.0f, -1000.0f, 0.0f);
    ground.radius = 1000.0f;
    ground.materialIdx = mat_ground;
    scene.spheres.push_back(ground);

    // Five metal spheres at x = (i-2)*2.5, y=1, z=0, radius=1
    for (int i = 0; i < 5; ++i) {
        SphereData s{};
        s.center = make_float3((i - 2) * 2.5f, 1.0f, 0.0f);
        s.radius = 1.0f;
        s.materialIdx = mat_metal[i];
        scene.spheres.push_back(s);
    }

    // Area light quad: Q=(-6,6,-4), u=(12,0,0), v=(0,0,8)
    QuadData lq{};
    lq.Q = make_float3(-6.0f, 6.0f, -4.0f);
    lq.u = make_float3(12.0f, 0.0f, 0.0f);
    lq.v = make_float3(0.0f, 0.0f, 8.0f);
    const float3 lc = cross(lq.u, lq.v);
    lq.w = lc;
    lq.normal = normalize(lc);
    lq.D = dot(lq.normal, lq.Q);
    lq.materialIdx = mat_light;
    scene.quads.push_back(lq);
    scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
    scene.isLightSphere.push_back(false);
}

/// @brief Build Cornell Rough Metal scene (scene 10)
/// Matches CPU build_cornell_rough_metal(): same walls/light, rough aluminum box + rough gold sphere
static void build_cornell_rough_metal(SceneData& scene) {
    // Materials
    const int mat_red = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.65f, 0.05f, 0.05f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_white = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_green = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.12f, 0.45f, 0.15f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_light = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f, make_float3(kLightIntensity, kLightIntensity, kLightIntensity) });

    // Rough aluminum box material (roughness 0.15)
    const int mat_alum = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Metal, make_float3(0.8f, 0.85f, 0.88f), 0.15f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    // Rough gold sphere material (roughness 0.3)
    const int mat_gold = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Metal, make_float3(0.95f, 0.78f, 0.28f), 0.3f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    // Cornell Box walls (same geometry as build_cornell_box)
    // Green wall (right, +X)
    { QuadData q{}; q.Q = make_float3(kBoxSize,0,0); q.u = make_float3(0,0,kBoxSize); q.v = make_float3(0,kBoxSize,0); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_green; scene.quads.push_back(q); }
    // Red wall (left, -X)
    { QuadData q{}; q.Q = make_float3(0,0,kBoxSize); q.u = make_float3(0,0,-kBoxSize); q.v = make_float3(0,kBoxSize,0); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_red; scene.quads.push_back(q); }
    // Light
    { QuadData q{}; q.Q = make_float3(213,554,227); q.u = make_float3(130,0,0); q.v = make_float3(0,0,105); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_light; scene.quads.push_back(q);
      scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1); scene.isLightSphere.push_back(false); }
    // White ceiling
    { QuadData q{}; q.Q = make_float3(0,kBoxSize,0); q.u = make_float3(kBoxSize,0,0); q.v = make_float3(0,0,kBoxSize); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }
    // White floor
    { QuadData q{}; q.Q = make_float3(0,0,kBoxSize); q.u = make_float3(kBoxSize,0,0); q.v = make_float3(0,0,-kBoxSize); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }
    // White back wall
    { QuadData q{}; q.Q = make_float3(kBoxSize,0,kBoxSize); q.u = make_float3(-kBoxSize,0,0); q.v = make_float3(0,kBoxSize,0); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }

    // Rough gold sphere (center 190, 90, 190), radius 90
    SphereData sphere{};
    sphere.center = make_float3(190.0f, 90.0f, 190.0f);
    sphere.radius = 90.0f;
    sphere.materialIdx = mat_gold;
    scene.spheres.push_back(sphere);

    // Rough aluminum box: box(0,0,0 -> 165,330,165), rotated 15 deg, translated (265,0,295)
    add_box(scene,
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(165.0f, 330.0f, 165.0f),
        mat_alum,
        15.0f,
        make_float3(265.0f, 0.0f, 295.0f));
}

/// @brief Build Cornell Conductor scene (scene 12)
/// Matches CPU build_cornell_conductor(): Cornell box with a gold sphere and
/// aluminium box using GGX VNDF + complex Fresnel (pbrt-v4 ConductorBxDF).
static void build_cornell_conductor(SceneData& scene) {
    const int mat_red   = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.65f, 0.05f, 0.05f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_white = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_green = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.12f, 0.45f, 0.15f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_light = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f,
                                 make_float3(kLightIntensity, kLightIntensity, kLightIntensity) });

    // Gold sphere material (conductor, roughness 0.1 -- polished gold)
    const int mat_gold = safe_cast_to_int(scene.materials.size());
    {
        MaterialData md{};
        md.type  = MaterialType::Conductor;
        md.fuzz  = 0.1f;   // roughness; alpha = sqrt(0.1)
        md.eta_c = make_float3(kConductorAu.eta_r, kConductorAu.eta_g, kConductorAu.eta_b);
        md.k_c   = make_float3(kConductorAu.k_r,   kConductorAu.k_g,   kConductorAu.k_b);
        scene.materials.push_back(md);
    }

    // Aluminium box material (conductor, roughness 0.05 -- polished aluminium)
    const int mat_alum = safe_cast_to_int(scene.materials.size());
    {
        MaterialData md{};
        md.type  = MaterialType::Conductor;
        md.fuzz  = 0.05f;
        md.eta_c = make_float3(kConductorAl.eta_r, kConductorAl.eta_g, kConductorAl.eta_b);
        md.k_c   = make_float3(kConductorAl.k_r,   kConductorAl.k_g,   kConductorAl.k_b);
        scene.materials.push_back(md);
    }

    // Cornell Box walls
    { QuadData q{}; q.Q=make_float3(kBoxSize,0,0); q.u=make_float3(0,0,kBoxSize); q.v=make_float3(0,kBoxSize,0); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_green; scene.quads.push_back(q); }
    { QuadData q{}; q.Q=make_float3(0,0,kBoxSize); q.u=make_float3(0,0,-kBoxSize); q.v=make_float3(0,kBoxSize,0); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_red; scene.quads.push_back(q); }
    { QuadData q{}; q.Q=make_float3(213,554,227); q.u=make_float3(130,0,0); q.v=make_float3(0,0,105); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_light; scene.quads.push_back(q);
      scene.lightIndices.push_back(static_cast<int>(scene.quads.size())-1); scene.isLightSphere.push_back(false); }
    { QuadData q{}; q.Q=make_float3(0,kBoxSize,0); q.u=make_float3(kBoxSize,0,0); q.v=make_float3(0,0,kBoxSize); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }
    { QuadData q{}; q.Q=make_float3(0,0,kBoxSize); q.u=make_float3(kBoxSize,0,0); q.v=make_float3(0,0,-kBoxSize); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }
    { QuadData q{}; q.Q=make_float3(kBoxSize,0,kBoxSize); q.u=make_float3(-kBoxSize,0,0); q.v=make_float3(0,kBoxSize,0); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }

    // Gold sphere
    SphereData sphere{};
    sphere.center = make_float3(190.0f, 90.0f, 190.0f);
    sphere.radius = 90.0f;
    sphere.materialIdx = mat_gold;
    scene.spheres.push_back(sphere);

    // Polished aluminium box
    add_box(scene,
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(165.0f, 330.0f, 165.0f),
        mat_alum,
        15.0f,
        make_float3(265.0f, 0.0f, 295.0f));
}

/// @brief Build Cornell Coated Diffuse scene (scene 13)
/// Matches CPU build_cornell_coated_diffuse(): Cornell box with a blue coated sphere
/// and a red coated box (rough dielectric coat over Lambertian base, pbrt-v4 CoatedDiffuseBxDF)
static void build_cornell_coated_diffuse(SceneData& scene) {
    const int mat_red   = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.65f, 0.05f, 0.05f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_white = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_green = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.12f, 0.45f, 0.15f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_light = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f,
                                 make_float3(kLightIntensity, kLightIntensity, kLightIntensity) });

    // Blue coated-diffuse sphere (IOR 1.5, roughness 0.1)
    const int mat_coated_blue = safe_cast_to_int(scene.materials.size());
    {
        MaterialData md{};
        md.type   = MaterialType::CoatedDiffuse;
        md.albedo = make_float3(0.2f, 0.3f, 0.9f);  // diffuse base colour
        md.fuzz   = 0.1f;                            // coat roughness (RoughnessToAlpha done in shader)
        md.ior    = 1.5f;                            // coat IOR (glass-like)
        scene.materials.push_back(md);
    }

    // Red coated-diffuse box (IOR 1.5, roughness 0.2)
    const int mat_coated_red = safe_cast_to_int(scene.materials.size());
    {
        MaterialData md{};
        md.type   = MaterialType::CoatedDiffuse;
        md.albedo = make_float3(0.8f, 0.1f, 0.1f);
        md.fuzz   = 0.2f;
        md.ior    = 1.5f;
        scene.materials.push_back(md);
    }

    // Cornell Box walls
    { QuadData q{}; q.Q = make_float3(kBoxSize,0,0); q.u = make_float3(0,0,kBoxSize); q.v = make_float3(0,kBoxSize,0); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_green; scene.quads.push_back(q); }
    { QuadData q{}; q.Q = make_float3(0,0,kBoxSize); q.u = make_float3(0,0,-kBoxSize); q.v = make_float3(0,kBoxSize,0); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_red; scene.quads.push_back(q); }
    { QuadData q{}; q.Q = make_float3(213,554,227); q.u = make_float3(130,0,0); q.v = make_float3(0,0,105); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_light; scene.quads.push_back(q);
      scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1); scene.isLightSphere.push_back(false); }
    { QuadData q{}; q.Q = make_float3(0,kBoxSize,0); q.u = make_float3(kBoxSize,0,0); q.v = make_float3(0,0,kBoxSize); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }
    { QuadData q{}; q.Q = make_float3(0,0,kBoxSize); q.u = make_float3(kBoxSize,0,0); q.v = make_float3(0,0,-kBoxSize); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }
    { QuadData q{}; q.Q = make_float3(kBoxSize,0,kBoxSize); q.u = make_float3(-kBoxSize,0,0); q.v = make_float3(0,kBoxSize,0); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }

    // Blue coated-diffuse sphere
    SphereData sph{};
    sph.center    = make_float3(190.0f, 90.0f, 190.0f);
    sph.radius    = 90.0f;
    sph.materialIdx = mat_coated_blue;
    scene.spheres.push_back(sph);

    // Red coated-diffuse box
    add_box(scene,
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(165.0f, 330.0f, 165.0f),
        mat_coated_red,
        15.0f,
        make_float3(265.0f, 0.0f, 295.0f));
}

/// @brief Build Cornell Thin Glass scene (scene 14)
/// Matches CPU build_cornell_thin_glass(): Cornell box with a vertical thin-glass panel
/// using pbrt-v4 ThinDielectricBxDF (analytic multi-bounce Fresnel, no bending).
static void build_cornell_thin_glass(SceneData& scene) {
    const int mat_red   = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.65f, 0.05f, 0.05f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_white = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_green = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.12f, 0.45f, 0.15f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_light = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f,
                                 make_float3(kLightIntensity, kLightIntensity, kLightIntensity) });

    // White diffuse box
    const int mat_box = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    // Thin-glass panel (IOR 1.5)
    const int mat_panel = safe_cast_to_int(scene.materials.size());
    {
        MaterialData md{};
        md.type = MaterialType::ThinDielectric;
        md.ior  = 1.5f;
        scene.materials.push_back(md);
    }

    // Cornell Box walls
    { QuadData q{}; q.Q = make_float3(kBoxSize,0,0); q.u = make_float3(0,0,kBoxSize); q.v = make_float3(0,kBoxSize,0); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_green; scene.quads.push_back(q); }
    { QuadData q{}; q.Q = make_float3(0,0,kBoxSize); q.u = make_float3(0,0,-kBoxSize); q.v = make_float3(0,kBoxSize,0); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_red; scene.quads.push_back(q); }
    { QuadData q{}; q.Q = make_float3(213,554,227); q.u = make_float3(130,0,0); q.v = make_float3(0,0,105); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_light; scene.quads.push_back(q);
      scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1); scene.isLightSphere.push_back(false); }
    { QuadData q{}; q.Q = make_float3(0,kBoxSize,0); q.u = make_float3(kBoxSize,0,0); q.v = make_float3(0,0,kBoxSize); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }
    { QuadData q{}; q.Q = make_float3(0,0,kBoxSize); q.u = make_float3(kBoxSize,0,0); q.v = make_float3(0,0,-kBoxSize); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }
    { QuadData q{}; q.Q = make_float3(kBoxSize,0,kBoxSize); q.u = make_float3(-kBoxSize,0,0); q.v = make_float3(0,kBoxSize,0); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }

    // White diffuse box (right side)
    add_box(scene,
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(165.0f, 330.0f, 165.0f),
        mat_box,
        15.0f,
        make_float3(265.0f, 0.0f, 295.0f));

    // Thin-glass panel: vertical slab spanning box interior
    // Q=(100,0,200), u=(0,555,0), v=(355,0,0)  -- horizontal quad lying in xz plane rotated
    {
        QuadData q{};
        q.Q = make_float3(100.0f, 0.0f, 200.0f);
        q.u = make_float3(0.0f, 555.0f, 0.0f);
        q.v = make_float3(355.0f, 0.0f, 0.0f);
        const float3 c = cross(q.u, q.v);
        q.w      = c;
        q.normal = normalize(c);
        q.D      = dot(q.normal, q.Q);
        q.materialIdx = mat_panel;
        scene.quads.push_back(q);
    }
}

/// @brief Build Cornell Coated Conductor scene (scene 15)
/// Matches CPU build_cornell_coated_conductor(): Cornell box with a lacquered-gold sphere
/// and a lacquered-copper box using pbrt-v4 CoatedConductorBxDF.
/// coat: IOR=1.5, roughness=0.1/0.2; conductor: Au sphere, Cu box.
static void build_cornell_coated_conductor(SceneData& scene) {
    const int mat_red   = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.65f, 0.05f, 0.05f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_white = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_green = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.12f, 0.45f, 0.15f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_light = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f,
                                 make_float3(kLightIntensity, kLightIntensity, kLightIntensity) });

    // Lacquered-gold sphere (Au conductor, IOR-1.5 coat, roughness 0.1)
    const int mat_gold_lacquer = safe_cast_to_int(scene.materials.size());
    {
        MaterialData md{};
        md.type  = MaterialType::CoatedConductor;
        md.fuzz  = 0.1f;   // coat roughness
        md.ior   = 1.5f;   // coat IOR
        md.eta_c = make_float3(kConductorAu.eta_r, kConductorAu.eta_g, kConductorAu.eta_b);
        md.k_c   = make_float3(kConductorAu.k_r,   kConductorAu.k_g,   kConductorAu.k_b);
        scene.materials.push_back(md);
    }

    // Lacquered-copper box (Cu conductor, IOR-1.5 coat, roughness 0.2)
    const int mat_copper_lacquer = safe_cast_to_int(scene.materials.size());
    {
        MaterialData md{};
        md.type  = MaterialType::CoatedConductor;
        md.fuzz  = 0.2f;
        md.ior   = 1.5f;
        md.eta_c = make_float3(kConductorCu.eta_r, kConductorCu.eta_g, kConductorCu.eta_b);
        md.k_c   = make_float3(kConductorCu.k_r,   kConductorCu.k_g,   kConductorCu.k_b);
        scene.materials.push_back(md);
    }

    // Cornell Box walls
    { QuadData q{}; q.Q=make_float3(kBoxSize,0,0); q.u=make_float3(0,0,kBoxSize); q.v=make_float3(0,kBoxSize,0); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_green; scene.quads.push_back(q); }
    { QuadData q{}; q.Q=make_float3(0,0,kBoxSize); q.u=make_float3(0,0,-kBoxSize); q.v=make_float3(0,kBoxSize,0); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_red; scene.quads.push_back(q); }
    { QuadData q{}; q.Q=make_float3(213,554,227); q.u=make_float3(130,0,0); q.v=make_float3(0,0,105); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_light; scene.quads.push_back(q);
      scene.lightIndices.push_back(static_cast<int>(scene.quads.size())-1); scene.isLightSphere.push_back(false); }
    { QuadData q{}; q.Q=make_float3(0,kBoxSize,0); q.u=make_float3(kBoxSize,0,0); q.v=make_float3(0,0,kBoxSize); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }
    { QuadData q{}; q.Q=make_float3(0,0,kBoxSize); q.u=make_float3(kBoxSize,0,0); q.v=make_float3(0,0,-kBoxSize); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }
    { QuadData q{}; q.Q=make_float3(kBoxSize,0,kBoxSize); q.u=make_float3(-kBoxSize,0,0); q.v=make_float3(0,kBoxSize,0); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }

    // Lacquered-gold sphere
    SphereData sphere{};
    sphere.center = make_float3(190.0f, 90.0f, 190.0f);
    sphere.radius = 90.0f;
    sphere.materialIdx = mat_gold_lacquer;
    scene.spheres.push_back(sphere);

    // Lacquered-copper box
    add_box(scene,
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(165.0f, 330.0f, 165.0f),
        mat_copper_lacquer,
        15.0f,
        make_float3(265.0f, 0.0f, 295.0f));
}

/// Matches CPU build_cornell_wax_slab(): Cornell box with a wax sphere (DiffuseTransmissionBxDF).
/// albedo = reflectance R, emission field = transmittance T.
static void build_cornell_wax_slab(SceneData& scene) {
    const int mat_red   = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.65f, 0.05f, 0.05f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_white = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_green = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.12f, 0.45f, 0.15f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_light = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f,
                                 make_float3(kLightIntensity, kLightIntensity, kLightIntensity) });

    // Wax sphere: albedo = R (reflectance), emission = T (transmittance)
    const int mat_wax = safe_cast_to_int(scene.materials.size());
    {
        MaterialData md{};
        md.type    = MaterialType::DiffuseTransmission;
        md.albedo  = make_float3(0.6f, 0.5f, 0.3f);   // R: reflected diffuse color
        md.emission = make_float3(0.8f, 0.6f, 0.3f);  // T: transmitted diffuse color
        md.fuzz    = 0.0f;
        md.ior     = 0.0f;
        scene.materials.push_back(md);
    }

    // Cornell Box walls
    { QuadData q{}; q.Q=make_float3(kBoxSize,0,0); q.u=make_float3(0,0,kBoxSize); q.v=make_float3(0,kBoxSize,0); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_green; scene.quads.push_back(q); }
    { QuadData q{}; q.Q=make_float3(0,0,kBoxSize); q.u=make_float3(0,0,-kBoxSize); q.v=make_float3(0,kBoxSize,0); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_red; scene.quads.push_back(q); }
    { QuadData q{}; q.Q=make_float3(213,554,227); q.u=make_float3(130,0,0); q.v=make_float3(0,0,105); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_light; scene.quads.push_back(q);
      scene.lightIndices.push_back(static_cast<int>(scene.quads.size())-1); scene.isLightSphere.push_back(false); }
    { QuadData q{}; q.Q=make_float3(0,kBoxSize,0); q.u=make_float3(kBoxSize,0,0); q.v=make_float3(0,0,kBoxSize); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }
    { QuadData q{}; q.Q=make_float3(0,0,kBoxSize); q.u=make_float3(kBoxSize,0,0); q.v=make_float3(0,0,-kBoxSize); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }
    { QuadData q{}; q.Q=make_float3(kBoxSize,0,kBoxSize); q.u=make_float3(-kBoxSize,0,0); q.v=make_float3(0,kBoxSize,0); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }

    // Wax sphere (left)
    SphereData wax_sphere{};
    wax_sphere.center    = make_float3(190.0f, 90.0f, 190.0f);
    wax_sphere.radius    = 90.0f;
    wax_sphere.materialIdx = mat_wax;
    scene.spheres.push_back(wax_sphere);

    // White diffuse box (right)
    add_box(scene,
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(165.0f, 330.0f, 165.0f),
        mat_white,
        15.0f,
        make_float3(265.0f, 0.0f, 295.0f));
}

/// @brief Build Cornell Crystal scene (scene 17)
/// Matches CPU build_cornell_crystal(): Cornell box with NormalizedFresnelBxDF crystal sphere.
/// ior stored in mat.ior; normalization constant c computed on GPU via FresnelMoment1.
static void build_cornell_crystal(SceneData& scene) {
    const int mat_red   = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.65f, 0.05f, 0.05f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_white = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_green = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.12f, 0.45f, 0.15f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_light = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f,
                                 make_float3(kLightIntensity, kLightIntensity, kLightIntensity) });

    // Crystal sphere: NormalizedFresnelBxDF, IOR 1.5 (glass/crystal)
    const int mat_crystal = safe_cast_to_int(scene.materials.size());
    {
        MaterialData md{};
        md.type    = MaterialType::NormalizedFresnel;
        md.albedo  = make_float3(1.0f, 1.0f, 1.0f);  // unused -- weight computed from Fresnel
        md.ior     = 1.5f;
        md.fuzz    = 0.0f;
        md.emission = make_float3(0.0f, 0.0f, 0.0f);
        scene.materials.push_back(md);
    }

    // Cornell Box walls
    { QuadData q{}; q.Q=make_float3(kBoxSize,0,0); q.u=make_float3(0,0,kBoxSize); q.v=make_float3(0,kBoxSize,0); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_green; scene.quads.push_back(q); }
    { QuadData q{}; q.Q=make_float3(0,0,kBoxSize); q.u=make_float3(0,0,-kBoxSize); q.v=make_float3(0,kBoxSize,0); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_red; scene.quads.push_back(q); }
    { QuadData q{}; q.Q=make_float3(213,554,227); q.u=make_float3(130,0,0); q.v=make_float3(0,0,105); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_light; scene.quads.push_back(q);
      scene.lightIndices.push_back(static_cast<int>(scene.quads.size())-1); scene.isLightSphere.push_back(false); }
    { QuadData q{}; q.Q=make_float3(0,kBoxSize,0); q.u=make_float3(kBoxSize,0,0); q.v=make_float3(0,0,kBoxSize); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }
    { QuadData q{}; q.Q=make_float3(0,0,kBoxSize); q.u=make_float3(kBoxSize,0,0); q.v=make_float3(0,0,-kBoxSize); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }
    { QuadData q{}; q.Q=make_float3(kBoxSize,0,kBoxSize); q.u=make_float3(-kBoxSize,0,0); q.v=make_float3(0,kBoxSize,0); const float3 c=cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }

    // Crystal sphere (left)
    SphereData crystal_sphere{};
    crystal_sphere.center    = make_float3(190.0f, 90.0f, 190.0f);
    crystal_sphere.radius    = 90.0f;
    crystal_sphere.materialIdx = mat_crystal;
    scene.spheres.push_back(crystal_sphere);

    // White diffuse box (right)
    add_box(scene,
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(165.0f, 330.0f, 165.0f),
        mat_white,
        15.0f,
        make_float3(265.0f, 0.0f, 295.0f));
}

/// @brief Build Cornell Rough Glass scene (scene 11)
/// Matches CPU build_cornell_rough_glass(): same walls/light, diffuse box + rough-glass sphere
static void build_cornell_rough_glass(SceneData& scene) {
    const int mat_red   = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.65f, 0.05f, 0.05f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_white = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_green = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.12f, 0.45f, 0.15f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

    const int mat_light = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f, make_float3(kLightIntensity, kLightIntensity, kLightIntensity) });

    // Rough glass sphere (roughness 0.2, IOR 1.5 -- frosted glass)
    const int mat_rough_glass = safe_cast_to_int(scene.materials.size());
    scene.materials.push_back({ MaterialType::RoughDielectric, make_float3(1.0f, 1.0f, 1.0f), 0.2f, kGlassIOR, make_float3(0.0f, 0.0f, 0.0f) });

    // Cornell Box walls (same geometry as build_cornell_box)
    { QuadData q{}; q.Q = make_float3(kBoxSize,0,0); q.u = make_float3(0,0,kBoxSize); q.v = make_float3(0,kBoxSize,0); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_green; scene.quads.push_back(q); }
    { QuadData q{}; q.Q = make_float3(0,0,kBoxSize); q.u = make_float3(0,0,-kBoxSize); q.v = make_float3(0,kBoxSize,0); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_red; scene.quads.push_back(q); }
    { QuadData q{}; q.Q = make_float3(213,554,227); q.u = make_float3(130,0,0); q.v = make_float3(0,0,105); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_light; scene.quads.push_back(q);
      scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1); scene.isLightSphere.push_back(false); }
    { QuadData q{}; q.Q = make_float3(0,kBoxSize,0); q.u = make_float3(kBoxSize,0,0); q.v = make_float3(0,0,kBoxSize); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }
    { QuadData q{}; q.Q = make_float3(0,0,kBoxSize); q.u = make_float3(kBoxSize,0,0); q.v = make_float3(0,0,-kBoxSize); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }
    { QuadData q{}; q.Q = make_float3(kBoxSize,0,kBoxSize); q.u = make_float3(-kBoxSize,0,0); q.v = make_float3(0,kBoxSize,0); const float3 c = cross(q.u,q.v); q.w=c; q.normal=normalize(c); q.D=dot(q.normal,q.Q); q.materialIdx=mat_white; scene.quads.push_back(q); }

    // White diffuse box (right)
    add_box(scene,
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(165.0f, 330.0f, 165.0f),
        mat_white,
        15.0f,
        make_float3(265.0f, 0.0f, 295.0f));

    // Rough glass sphere (left, same position as original glass sphere)
    SphereData rg_sphere{};
    rg_sphere.center    = make_float3(190.0f, 90.0f, 190.0f);
    rg_sphere.radius    = 90.0f;
    rg_sphere.materialIdx = mat_rough_glass;
    scene.spheres.push_back(rg_sphere);
}

/**
 * Build checkered spheres scene (scene 2)
 * Two spheres with different albedos
 */
void build_checkered_spheres(SceneData& scene) {
	// Bottom sphere (darker)
	const int mat1 = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({
		MaterialType::Lambertian,
		make_float3(0.2f, 0.3f, 0.1f),  // albedo
		0.0f, 0.0f,
		make_float3(0.0f, 0.0f, 0.0f)  // no emission
	});

	SphereData sphere1{};
	sphere1.center = make_float3(0.0f, -10.0f, 0.0f);
	sphere1.radius = 10.0f;
	sphere1.materialIdx = mat1;
	scene.spheres.push_back(sphere1);
	if (is_emissive(scene, mat1)) {
		scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
		scene.isLightSphere.push_back(true);
	}

	// Top sphere (lighter)
	const int mat2 = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({
		MaterialType::Lambertian,
		make_float3(0.9f, 0.9f, 0.9f),  // albedo
		0.0f, 0.0f,
		make_float3(0.0f, 0.0f, 0.0f)  // no emission
	});

	SphereData sphere2{};
	sphere2.center = make_float3(0.0f, 10.0f, 0.0f);
	sphere2.radius = 10.0f;
	sphere2.materialIdx = mat2;
	scene.spheres.push_back(sphere2);
	if (is_emissive(scene, mat2)) {
		scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
		scene.isLightSphere.push_back(true);
	}
}

/**
 * Build colored quads scene (scene 5)
 * Five colored quads arranged in 3D space
 */
void build_quads_scene(SceneData& scene) {
	// Helper lambda to add a quad with its material
	auto add_quad = [&scene](float Qx, float Qy, float Qz,
							  float ux, float uy, float uz,
							  float vx, float vy, float vz,
							  float r, float g, float b) {
		// Add material
		const int mat_idx = safe_cast_to_int(scene.materials.size());
		scene.materials.push_back({
			MaterialType::Lambertian,
			make_float3(r, g, b),
			0.0f, 0.0f,
			make_float3(0.0f, 0.0f, 0.0f)
		});

		// Add quad
		QuadData quad{};
		quad.Q = make_float3(Qx, Qy, Qz);
		quad.u = make_float3(ux, uy, uz);
		quad.v = make_float3(vx, vy, vz);
		const float3 cross_prod = cross(quad.u, quad.v);
		quad.normal = normalize(cross_prod);
		quad.D = dot(quad.normal, quad.Q);
		quad.materialIdx = mat_idx;
		scene.quads.push_back(quad);

		// Track if emissive
		if (is_emissive(scene, mat_idx)) {
			scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
			scene.isLightSphere.push_back(false);
		}
	};

	// Left red quad
	add_quad(-3.0f, -2.0f, 5.0f, 0.0f, 0.0f, -4.0f, 0.0f, 4.0f, 0.0f, 1.0f, 0.2f, 0.2f);

	// Back green quad
	add_quad(-2.0f, -2.0f, 0.0f, 4.0f, 0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.2f, 1.0f, 0.2f);

	// Right blue quad
	add_quad(3.0f, -2.0f, 1.0f, 0.0f, 0.0f, 4.0f, 0.0f, 4.0f, 0.0f, 0.2f, 0.2f, 1.0f);

	// Upper orange quad
	add_quad(-2.0f, 3.0f, 1.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f, 1.0f, 0.5f, 0.0f);

	// Lower teal quad
	add_quad(-2.0f, -3.0f, 5.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, -4.0f, 0.2f, 0.8f, 0.8f);
}

/// @brief Cornell box walls (no light quad) + two spheres, for scenes lit by
/// a punctual (point/spot/distant) light instead of an emissive quad.
/// Matches CPU src/TheRestOfYourLife/scenes_advanced.h cornell_walls_no_light()
/// exactly: same 5 walls, same two sphere positions/materials.
static void build_punctual_light_walls(SceneData& scene) {
	const int mat_red = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.65f, 0.05f, 0.05f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

	const int mat_white = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

	const int mat_green = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.12f, 0.45f, 0.15f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

	// Sphere materials: white lambertian + blue-tinted fuzzy metal (matches CPU)
	const int mat_white_sphere = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

	const int mat_metal_sphere = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Metal, make_float3(0.8f, 0.8f, 0.9f), 0.1f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

	// Walls (green/red/ceiling/floor/back - no front, no light quad)
	add_transformed_quad(scene, make_float3(kBoxSize, 0, 0), make_float3(0, 0, kBoxSize), make_float3(0, kBoxSize, 0), mat_green);
	add_transformed_quad(scene, make_float3(0, 0, kBoxSize), make_float3(0, 0, -kBoxSize), make_float3(0, kBoxSize, 0), mat_red);
	add_transformed_quad(scene, make_float3(0, kBoxSize, 0), make_float3(kBoxSize, 0, 0), make_float3(0, 0, kBoxSize), mat_white);   // ceiling
	add_transformed_quad(scene, make_float3(0, 0, kBoxSize), make_float3(kBoxSize, 0, 0), make_float3(0, 0, -kBoxSize), mat_white);  // floor
	add_transformed_quad(scene, make_float3(kBoxSize, 0, kBoxSize), make_float3(-kBoxSize, 0, 0), make_float3(0, kBoxSize, 0), mat_white); // back

	// Two spheres (matches CPU cornell_walls_no_light exactly)
	{ SphereData s{}; s.center = make_float3(190.0f, 90.0f, 190.0f); s.radius = 90.0f; s.materialIdx = mat_white_sphere; scene.spheres.push_back(s); }
	{ SphereData s{}; s.center = make_float3(370.0f, 120.0f, 380.0f); s.radius = 120.0f; s.materialIdx = mat_metal_sphere; scene.spheres.push_back(s); }
}

/// @brief Scene 25: Spotlight Cornell. Matches CPU build_spotlight_punct().
static void build_spotlight_cornell_gpu(SceneData& scene) {
	build_punctual_light_walls(scene);

	constexpr float kPi = 3.14159265358979323846f;
	auto deg2rad = [](float d) { return d * kPi / 180.0f; };

	PunctualLightGPU light{};
	light.kind = PunctualLightKind::Spot;
	light.spot.pos_x = 278.0f; light.spot.pos_y = 548.0f; light.spot.pos_z = 278.0f;
	light.spot.dir_x = 0.0f;   light.spot.dir_y = -1.0f;  light.spot.dir_z = 0.0f;  // already unit
	light.spot.ir = 1.0f; light.spot.ig = 0.95f; light.spot.ib = 0.85f;
	light.spot.scale = 600000.0f;
	light.spot.cos_falloff_start = cosf(deg2rad(15.0f));
	light.spot.cos_falloff_end   = cosf(deg2rad(30.0f));
	scene.punctualLights.push_back(light);
}

/// @brief Scene 26: Distant Light Cornell. Matches CPU build_distant_light_punct().
static void build_distant_light_cornell_gpu(SceneData& scene) {
	build_punctual_light_walls(scene);

	float3 dir = normalize(make_float3(-0.4f, -1.0f, -0.2f));

	PunctualLightGPU light{};
	light.kind = PunctualLightKind::Distant;
	light.distant.dir_x = dir.x; light.distant.dir_y = dir.y; light.distant.dir_z = dir.z;
	light.distant.ir = 1.0f; light.distant.ig = 0.98f; light.distant.ib = 0.92f;
	light.distant.scale = 800000.0f;
	light.distant.scene_radius = 1000.0f;
	scene.punctualLights.push_back(light);
}

/// @brief Scene 27: Point Light Cornell. Matches CPU build_point_light_punct().
static void build_point_light_cornell_gpu(SceneData& scene) {
	build_punctual_light_walls(scene);

	PunctualLightGPU light{};
	light.kind = PunctualLightKind::Point;
	light.point.pos_x = 278.0f; light.point.pos_y = 540.0f; light.point.pos_z = 278.0f;
	light.point.ir = 1.0f; light.point.ig = 0.98f; light.point.ib = 0.90f;
	light.point.scale = 5000000.0f;
	scene.punctualLights.push_back(light);
}

/// @brief Scene 28: Goniometric Light Cornell. Matches CPU build_goniometric_punct().
static void build_goniometric_cornell_gpu(SceneData& scene) {
	build_punctual_light_walls(scene);

	PunctualLightGPU light{};
	light.kind = PunctualLightKind::Goniometric;
	GoniometricLightGPU& g = light.gonio;
	g.pos_x = 278.0f; g.pos_y = 520.0f; g.pos_z = 278.0f;
	// Identity rotation (matches CPU's id[9] = {1,0,0, 0,1,0, 0,0,1})
	g.world_to_light[0] = 1.0f; g.world_to_light[1] = 0.0f; g.world_to_light[2] = 0.0f;
	g.world_to_light[3] = 0.0f; g.world_to_light[4] = 1.0f; g.world_to_light[5] = 0.0f;
	g.world_to_light[6] = 0.0f; g.world_to_light[7] = 0.0f; g.world_to_light[8] = 1.0f;
	g.ir = 1.0f; g.ig = 0.9f; g.ib = 0.7f;
	g.scale = 4000000.0f;
	// Same synthetic profile as CPU build_goniometric_punct(): bright toward
	// the bottom hemisphere (v > NV/2), dim toward the top.
	g.nu = 16; g.nv = 8;
	for (int v = 0; v < g.nv; ++v) {
		float t = (float)v / (float)g.nv;
		for (int u = 0; u < g.nu; ++u)
			g.image[v * g.nu + u] = 0.2f + 0.8f * t;
	}
	scene.punctualLights.push_back(light);
}

/// @brief Scene 29: Projection Light Cornell. Matches CPU build_projection_punct().
static void build_projection_cornell_gpu(SceneData& scene) {
	build_punctual_light_walls(scene);

	PunctualLightGPU light{};
	light.kind = PunctualLightKind::Projection;
	ProjectionLightGPU& pr = light.proj;
	pr.pos_x = 278.0f; pr.pos_y = 278.0f; pr.pos_z = -50.0f;
	// Identity rotation (matches CPU's wtl[9] = {1,0,0, 0,1,0, 0,0,1})
	pr.world_to_light[0] = 1.0f; pr.world_to_light[1] = 0.0f; pr.world_to_light[2] = 0.0f;
	pr.world_to_light[3] = 0.0f; pr.world_to_light[4] = 1.0f; pr.world_to_light[5] = 0.0f;
	pr.world_to_light[6] = 0.0f; pr.world_to_light[7] = 0.0f; pr.world_to_light[8] = 1.0f;
	pr.scale = 1000000.0f;
	pr.hither = 1e-3f;
	pr.nx = 8; pr.ny = 8;
	constexpr float kPi = 3.14159265358979323846f;
	const float fov_deg = 40.0f;
	// Screen bounds (mirrors ProjectionLight<T>::make, cameras.h aspect logic)
	const float aspect = (float)pr.nx / (float)pr.ny;
	if (aspect >= 1.0f) {
		pr.sb_xmin = -aspect; pr.sb_xmax = aspect;
		pr.sb_ymin = -1.0f;   pr.sb_ymax = 1.0f;
	} else {
		pr.sb_xmin = -1.0f;         pr.sb_xmax = 1.0f;
		pr.sb_ymin = -1.0f/aspect;  pr.sb_ymax = 1.0f/aspect;
	}
	// screenFromLight reduces to a single scalar - see ProjectionLightGPU's
	// comment in optix_types.h for why the full 4x4 matrix isn't needed.
	pr.inv_tan = 1.0f / tanf((kPi / 180.0f) * fov_deg / 2.0f);
	// Same 8x8 checkerboard slide as CPU build_projection_punct().
	for (int y = 0; y < pr.ny; ++y) {
		for (int x = 0; x < pr.nx; ++x) {
			float v = ((x + y) % 2 == 0) ? 1.0f : 0.05f;
			int idx = (y * pr.nx + x) * 3;
			pr.image_rgb[idx] = v; pr.image_rgb[idx + 1] = v; pr.image_rgb[idx + 2] = v;
		}
	}
	scene.punctualLights.push_back(light);
}

// Shared overhead area light used by scenes 22/32 below. GPU has no
// background/miss-color model (unlike CPU, which lights these scenes purely
// via a flat sky background color - see build_scene()'s camera-only comment
// for scenes 22/32) - without this, either GPU scene would render fully
// black despite the camera itself working correctly.
static void add_overhead_area_light(SceneData& scene, float3 center, float halfSize, float intensity) {
	const int mat_light = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f,
								 make_float3(intensity, intensity, intensity) });
	QuadData lq{};
	lq.Q = make_float3(center.x - halfSize, center.y, center.z - halfSize);
	lq.u = make_float3(2.0f * halfSize, 0.0f, 0.0f);
	lq.v = make_float3(0.0f, 0.0f, 2.0f * halfSize);
	const float3 lc = cross(lq.u, lq.v);
	lq.w = lc;
	lq.normal = normalize(lc);
	lq.D = dot(lq.normal, lq.Q);
	lq.materialIdx = mat_light;
	scene.quads.push_back(lq);
	scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
	scene.isLightSphere.push_back(false);
}

/// @brief Scene 22: Depth of Field. Matches CPU build_depth_of_field() in
/// spirit (ground + a row of spheres spanning near/far of the focus plane to
/// show defocus blur) - simplified to solid-color materials since GPU has no
/// checker/procedural texture support.
static void build_depth_of_field_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.5f, 0.5f, 0.5f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	SphereData ground{};
	// Modest flat ground (not the usual radius-1000 "planet" sphere used
	// elsewhere in this file) - at this camera's close distance/narrow fov,
	// a radius-1000 ground's curvature toward the horizon caught the
	// overhead light at a bad grazing angle and blew out most of the frame.
	ground.center = make_float3(0.0f, -50.0f, 0.0f);
	ground.radius = 50.0f;
	ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	// Five spheres spanning z = -4..+4 around the lookat point (z=0, matching
	// CPU's focus_dist=9 from lookfrom z=9), alternating material, so the
	// defocus blur visibly increases toward the near/far ends.
	const float3 colors[5] = {
		make_float3(0.8f, 0.2f, 0.2f), make_float3(0.2f, 0.8f, 0.2f), make_float3(0.9f, 0.9f, 0.9f),
		make_float3(0.2f, 0.2f, 0.8f), make_float3(0.8f, 0.8f, 0.2f)
	};
	const MaterialType kinds[5] = {
		MaterialType::Lambertian, MaterialType::Metal, MaterialType::Dielectric,
		MaterialType::Metal, MaterialType::Lambertian
	};
	for (int i = 0; i < 5; ++i) {
		const int mat = safe_cast_to_int(scene.materials.size());
		MaterialData m{};
		m.type = kinds[i];
		m.albedo = colors[i];
		m.fuzz = (kinds[i] == MaterialType::Metal) ? 0.05f : 0.0f;
		m.ior = (kinds[i] == MaterialType::Dielectric) ? 1.5f : 0.0f;
		scene.materials.push_back(m);
		SphereData s{};
		// Smaller radius than a first attempt at this scene used: at only 5
		// world units from the camera (nearest sphere, z=+4) with a narrow
		// 20-degree vfov, radius-1 spheres subtended more than the whole
		// frame and blew out to a solid color filling the image.
		s.center = make_float3(0.0f, 0.5f, (i - 2) * 2.0f);
		s.radius = 0.5f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}
	add_overhead_area_light(scene, make_float3(0.0f, 15.0f, 0.0f), 8.0f, 1.0f);
}

/// @brief Scene 32: Orthographic Camera. Matches CPU build_ortho_camera_scene()
/// in spirit (ground + a row of colored lambertian spheres) - simplified to
/// solid-color materials, same reasoning as scene 22 above.
static void build_ortho_camera_scene_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.5f, 0.5f, 0.5f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	SphereData ground{};
	ground.center = make_float3(0.0f, -1000.0f, 0.0f);
	ground.radius = 1000.0f;
	ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const float3 colors[5] = {
		make_float3(0.8f, 0.2f, 0.2f), make_float3(0.8f, 0.6f, 0.2f), make_float3(0.2f, 0.8f, 0.3f),
		make_float3(0.2f, 0.4f, 0.9f), make_float3(0.7f, 0.2f, 0.8f)
	};
	for (int i = 0; i < 5; ++i) {
		const int mat = safe_cast_to_int(scene.materials.size());
		scene.materials.push_back({ MaterialType::Lambertian, colors[i], 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
		SphereData s{};
		s.center = make_float3((i - 2) * 2.5f, 1.0f, 0.0f);
		s.radius = 1.0f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}
	add_overhead_area_light(scene, make_float3(0.0f, 15.0f, 0.0f), 8.0f, 4.0f);
}

/// @brief Scene 33: Spherical Camera. Matches CPU build_spherical_camera_scene()
/// in spirit (ground + a ring of colored spheres + one emissive sphere) -
/// self-illuminating, needs no extra light unlike scenes 22/32 above.
static void build_spherical_camera_scene_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.5f, 0.5f, 0.5f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	SphereData ground{};
	// CPU's SphericalCamera uses no camera_to_world (identity), so the
	// camera sits exactly at world origin - keep the ground surface below
	// that (top at y=-2) rather than tangent to it, matching how the row of
	// spheres/light below are all placed comfortably above the camera.
	ground.center = make_float3(0.0f, -1002.0f, 0.0f);
	ground.radius = 1000.0f;
	ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	constexpr float kPi = 3.14159265358979323846f;
	constexpr int kRingCount = 8;
	for (int i = 0; i < kRingCount; ++i) {
		float ang = (2.0f * kPi * i) / (float)kRingCount;
		float hue = (float)i / (float)kRingCount;
		float3 col = make_float3(0.5f + 0.5f * cosf(2.0f * kPi * hue),
								   0.5f + 0.5f * cosf(2.0f * kPi * (hue + 0.33f)),
								   0.5f + 0.5f * cosf(2.0f * kPi * (hue + 0.67f)));
		const int mat = safe_cast_to_int(scene.materials.size());
		scene.materials.push_back({ MaterialType::Lambertian, col, 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
		SphereData s{};
		s.center = make_float3(4.0f * cosf(ang), 1.0f, 4.0f * sinf(ang));
		s.radius = 0.8f;
		s.materialIdx = mat;
		scene.spheres.push_back(s);
	}

	// Central emissive sphere, matches CPU's central diffuse_light sphere.
	const int mat_light = safe_cast_to_int(scene.materials.size());
	constexpr float kSphericalLightIntensity = 8.0f;
	scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f,
								 make_float3(kSphericalLightIntensity, kSphericalLightIntensity, kSphericalLightIntensity) });
	SphereData lightSphere{};
	lightSphere.center = make_float3(0.0f, 3.0f, 0.0f);
	lightSphere.radius = 1.0f;
	lightSphere.materialIdx = mat_light;
	scene.spheres.push_back(lightSphere);
	scene.lightIndices.push_back(static_cast<int>(scene.spheres.size()) - 1);
	scene.isLightSphere.push_back(true);
}

/// @brief Scene 24: HDRI Sky. Matches CPU build_hdri_sky_world() (ground +
/// three spheres showcasing lambertian/metal/dielectric under sky light).
/// The CPU "HDRI" is actually a flat-color sky_light in practice (see
/// GpuCameraParams::backgroundColor's comment) - the caller sets that
/// separately, this only builds the ground+spheres geometry.
static void build_hdri_sky_world_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.4f, 0.4f, 0.4f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	SphereData ground{};
	ground.center = make_float3(0.0f, -1000.0f, 0.0f);
	ground.radius = 1000.0f;
	ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const int mat_lambertian = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.7f, 0.3f, 0.2f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	SphereData s1{};
	s1.center = make_float3(-3.0f, 1.0f, 0.0f);
	s1.radius = 1.0f;
	s1.materialIdx = mat_lambertian;
	scene.spheres.push_back(s1);

	const int mat_metal = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Metal, make_float3(0.8f, 0.8f, 0.9f), 0.05f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	SphereData s2{};
	s2.center = make_float3(0.0f, 1.0f, 0.0f);
	s2.radius = 1.0f;
	s2.materialIdx = mat_metal;
	scene.spheres.push_back(s2);

	const int mat_glass = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Dielectric, make_float3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f, make_float3(0.0f, 0.0f, 0.0f) });
	SphereData s3{};
	s3.center = make_float3(3.0f, 1.0f, 0.0f);
	s3.radius = 1.0f;
	s3.materialIdx = mat_glass;
	scene.spheres.push_back(s3);
}

/// @brief Scene 35: Portal Infinite Light. Matches CPU build_portal_light_scene()
/// (5-wall room, no front wall - "portal" for the sky to enter - + one metal
/// sphere). Uses the same wall-quad layout as build_punctual_light_walls,
/// duplicated here rather than shared since materials/sphere content differ.
static void build_portal_light_scene_gpu(SceneData& scene) {
	const int mat_red = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.65f, 0.05f, 0.05f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	const int mat_white = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	const int mat_green = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.12f, 0.45f, 0.15f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	const int mat_metal_sphere = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Metal, make_float3(0.8f, 0.8f, 0.9f), 0.05f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });

	add_transformed_quad(scene, make_float3(kBoxSize, 0, 0), make_float3(0, 0, kBoxSize), make_float3(0, kBoxSize, 0), mat_green);
	add_transformed_quad(scene, make_float3(0, 0, kBoxSize), make_float3(0, 0, -kBoxSize), make_float3(0, kBoxSize, 0), mat_red);
	add_transformed_quad(scene, make_float3(0, kBoxSize, 0), make_float3(kBoxSize, 0, 0), make_float3(0, 0, kBoxSize), mat_white);   // ceiling
	add_transformed_quad(scene, make_float3(0, 0, kBoxSize), make_float3(kBoxSize, 0, 0), make_float3(0, 0, -kBoxSize), mat_white);  // floor
	add_transformed_quad(scene, make_float3(kBoxSize, 0, kBoxSize), make_float3(-kBoxSize, 0, 0), make_float3(0, kBoxSize, 0), mat_white); // back

	SphereData s{};
	s.center = make_float3(190.0f, 100.0f, 190.0f);
	s.radius = 100.0f;
	s.materialIdx = mat_metal_sphere;
	scene.spheres.push_back(s);
}

/// @brief Scene 7: Cornell Smoke. Matches CPU build_cornell_smoke() (full
/// Cornell box + two constant_medium boxes, density 0.01, black/white).
/// GPU approximates the two rotated boxes as spheres (MaterialType::Medium
/// is sphere-only - see its comment in optix_types.h) positioned at roughly
/// the same locations, since a box boundary would need a second, more
/// involved AABB-slab intersection path not worth the complexity here.
static void build_cornell_smoke_gpu(SceneData& scene) {
	const int mat_red = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.65f, 0.05f, 0.05f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	const int mat_white = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	const int mat_green = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.12f, 0.45f, 0.15f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	const int mat_light = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f, make_float3(7.0f, 7.0f, 7.0f) });

	add_transformed_quad(scene, make_float3(kBoxSize, 0, 0), make_float3(0, 0, kBoxSize), make_float3(0, kBoxSize, 0), mat_green);
	add_transformed_quad(scene, make_float3(0, 0, kBoxSize), make_float3(0, 0, -kBoxSize), make_float3(0, kBoxSize, 0), mat_red);
	add_transformed_quad(scene, make_float3(0, kBoxSize, 0), make_float3(kBoxSize, 0, 0), make_float3(0, 0, kBoxSize), mat_white);   // ceiling
	add_transformed_quad(scene, make_float3(0, 0, kBoxSize), make_float3(kBoxSize, 0, 0), make_float3(0, 0, -kBoxSize), mat_white);  // floor
	add_transformed_quad(scene, make_float3(kBoxSize, 0, kBoxSize), make_float3(-kBoxSize, 0, 0), make_float3(0, kBoxSize, 0), mat_white); // back
	{
		QuadData lq{};
		lq.Q = make_float3(113.0f, 554.0f, 127.0f);
		lq.u = make_float3(330.0f, 0.0f, 0.0f);
		lq.v = make_float3(0.0f, 0.0f, 305.0f);
		const float3 lc = cross(lq.u, lq.v);
		lq.w = lc;
		lq.normal = normalize(lc);
		lq.D = dot(lq.normal, lq.Q);
		lq.materialIdx = mat_light;
		scene.quads.push_back(lq);
		scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
		scene.isLightSphere.push_back(false);
	}

	// Two medium spheres approximating CPU's two rotated boxes (centered
	// roughly where box1 [265,0,295]+165/2 and box2 [130,0,65]+82.5 sit).
	const int mat_medium_dark = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Medium, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.01f, make_float3(0.0f, 0.0f, 0.0f) });
	const int mat_medium_white = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Medium, make_float3(1.0f, 1.0f, 1.0f), 0.0f, 0.01f, make_float3(0.0f, 0.0f, 0.0f) });

	SphereData m1{}; m1.center = make_float3(347.0f, 165.0f, 377.0f); m1.radius = 115.0f; m1.materialIdx = mat_medium_dark;
	scene.spheres.push_back(m1);
	SphereData m2{}; m2.center = make_float3(212.0f, 82.0f, 147.0f); m2.radius = 82.0f; m2.materialIdx = mat_medium_white;
	scene.spheres.push_back(m2);
}

/// @brief Scene 30: Homogeneous Medium. Matches CPU
/// build_homogeneous_medium_scene() exactly (full Cornell box + a single
/// medium sphere at the box's center, radius 400, density 0.005, HG g=0.3).
static void build_homogeneous_medium_scene_gpu(SceneData& scene) {
	const int mat_red = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.65f, 0.05f, 0.05f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	const int mat_white = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.73f, 0.73f, 0.73f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	const int mat_green = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.12f, 0.45f, 0.15f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	const int mat_light = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::DiffuseLight, make_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f, make_float3(15.0f, 15.0f, 15.0f) });

	add_transformed_quad(scene, make_float3(kBoxSize, 0, 0), make_float3(0, 0, kBoxSize), make_float3(0, kBoxSize, 0), mat_green);
	add_transformed_quad(scene, make_float3(0, 0, kBoxSize), make_float3(0, 0, -kBoxSize), make_float3(0, kBoxSize, 0), mat_red);
	add_transformed_quad(scene, make_float3(0, kBoxSize, 0), make_float3(kBoxSize, 0, 0), make_float3(0, 0, kBoxSize), mat_white);   // ceiling
	add_transformed_quad(scene, make_float3(0, 0, kBoxSize), make_float3(kBoxSize, 0, 0), make_float3(0, 0, -kBoxSize), mat_white);  // floor
	add_transformed_quad(scene, make_float3(kBoxSize, 0, kBoxSize), make_float3(-kBoxSize, 0, 0), make_float3(0, kBoxSize, 0), mat_white); // back
	{
		QuadData lq{};
		lq.Q = make_float3(213.0f, 554.0f, 227.0f);
		lq.u = make_float3(130.0f, 0.0f, 0.0f);
		lq.v = make_float3(0.0f, 0.0f, 105.0f);
		const float3 lc = cross(lq.u, lq.v);
		lq.w = lc;
		lq.normal = normalize(lc);
		lq.D = dot(lq.normal, lq.Q);
		lq.materialIdx = mat_light;
		scene.quads.push_back(lq);
		scene.lightIndices.push_back(static_cast<int>(scene.quads.size()) - 1);
		scene.isLightSphere.push_back(false);
	}

	const int mat_medium = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Medium, make_float3(0.8f, 0.9f, 1.0f), 0.3f, 0.005f, make_float3(0.0f, 0.0f, 0.0f) });
	SphereData fog{};
	fog.center = make_float3(277.5f, 277.5f, 277.5f);
	fog.radius = 400.0f;
	fog.materialIdx = mat_medium;
	scene.spheres.push_back(fog);
}

/// @brief Scene 31: Cloud Medium. Matches CPU build_cloud_medium_scene()
/// (ground + one medium sphere, density 0.8, HG g=0.05) - the CPU's Perlin-
/// noise density texture is dead code there too (constructed but never
/// actually used by the constant_medium call, which passes a constant
/// density), so this is a faithful, not simplified, port.
static void build_cloud_medium_scene_gpu(SceneData& scene) {
	const int mat_ground = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Lambertian, make_float3(0.4f, 0.5f, 0.3f), 0.0f, 0.0f, make_float3(0.0f, 0.0f, 0.0f) });
	SphereData ground{};
	ground.center = make_float3(0.0f, -1000.0f, 0.0f);
	ground.radius = 1000.0f;
	ground.materialIdx = mat_ground;
	scene.spheres.push_back(ground);

	const int mat_medium = safe_cast_to_int(scene.materials.size());
	scene.materials.push_back({ MaterialType::Medium, make_float3(1.0f, 1.0f, 1.0f), 0.05f, 0.8f, make_float3(0.0f, 0.0f, 0.0f) });
	SphereData cloud{};
	cloud.center = make_float3(0.0f, 3.0f, 0.0f);
	cloud.radius = 3.0f;
	cloud.materialIdx = mat_medium;
	scene.spheres.push_back(cloud);
}

/// @brief Build a scene and configure the camera
/// @param scene_id Scene identifier (0 = Cornell Box)
/// @param image_width Output image width in pixels
/// @param image_height Output image height in pixels
/// @param scene Output scene data to populate
/// @param camera_params Output camera parameters array [origin(3), lower_left(3), horizontal(3), vertical(3)]
/// @return true if scene was built successfully, false for unknown scene_id
bool build_scene(
	const int scene_id,
	const int image_width,
	const int image_height,
	SceneData& scene,
	float* camera_params,
	const double cam_x,
	const double cam_y,
	const double cam_z,
	GpuCameraParams* out_camera_extra
) {
	if (camera_params == nullptr) {
		return false;  // Invalid camera parameter buffer
	}

	// Clear previous scene data
	scene.spheres.clear();
	scene.quads.clear();
	scene.materials.clear();

	// Build requested scene
	switch (scene_id) {
		case 0:  // Cornell Box
			build_cornell_box(scene);

			// Configure camera for Cornell Box
			{
				constexpr float kPi = 3.14159265358979323846f;

				// Camera parameters (use provided position)
				const float3 lookfrom = make_float3(
					static_cast<float>(cam_x),
					static_cast<float>(cam_y),
					static_cast<float>(cam_z)
				);
				const float3 lookat = make_float3(278.0f, 278.0f, 278.0f);  // Center of box
				const float3 vup = make_float3(0.0f, 1.0f, 0.0f);
				constexpr float vfov = 40.0f;  // Vertical field of view in degrees
				const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);

				// Calculate viewport dimensions
				const float theta = vfov * kPi / 180.0f;
				const float h = tanf(theta / 2.0f);
				const float viewport_height = 2.0f * h;
				const float viewport_width = aspect * viewport_height;

				// Calculate camera basis vectors
				const float3 view_direction = make_float3(
					lookfrom.x - lookat.x,
					lookfrom.y - lookat.y,
					lookfrom.z - lookat.z
				);
				const float3 w = normalize(view_direction);
				const float3 u = normalize(cross(vup, w));
				const float3 v = cross(w, u);

				// Calculate viewport vectors
				const float3 horizontal = make_float3(
					viewport_width * u.x,
					viewport_width * u.y,
					viewport_width * u.z
				);
				const float3 vertical = make_float3(
					viewport_height * v.x,
					viewport_height * v.y,
					viewport_height * v.z
				);

				// Calculate lower-left corner of viewport
				const float3 lower_left_corner = make_float3(
					lookfrom.x - horizontal.x / 2.0f - vertical.x / 2.0f - w.x,
					lookfrom.y - horizontal.y / 2.0f - vertical.y / 2.0f - w.y,
					lookfrom.z - horizontal.z / 2.0f - vertical.z / 2.0f - w.z
				);

				// Helper to pack float3 into array
				auto pack_float3 = [](float* dest, int offset, const float3& v) {
					dest[offset]     = v.x;
					dest[offset + 1] = v.y;
					dest[offset + 2] = v.z;
				};

				// Pack camera parameters: [origin(3), lower_left(3), horizontal(3), vertical(3)]
				pack_float3(camera_params, 0, lookfrom);
						pack_float3(camera_params, 3, lower_left_corner);
						pack_float3(camera_params, 6, horizontal);
						pack_float3(camera_params, 9, vertical);
					}
					break;

				case 2:  // Checkered Spheres
					build_checkered_spheres(scene);

					// Configure camera
					{
						constexpr float kPi = 3.14159265358979323846f;
						const float3 lookfrom = make_float3(static_cast<float>(cam_x), static_cast<float>(cam_y), static_cast<float>(cam_z));
						const float3 lookat = make_float3(0.0f, 0.0f, 0.0f);
						const float3 vup = make_float3(0.0f, 1.0f, 0.0f);
						constexpr float vfov = 20.0f;
						const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);

						const float theta = vfov * kPi / 180.0f;
						const float h = tanf(theta / 2.0f);
						const float viewport_height = 2.0f * h;
						const float viewport_width = aspect * viewport_height;

						const float3 view_direction = make_float3(lookfrom.x - lookat.x, lookfrom.y - lookat.y, lookfrom.z - lookat.z);
						const float3 w = normalize(view_direction);
						const float3 u = normalize(cross(vup, w));
						const float3 v = cross(w, u);

						const float3 horizontal = make_float3(viewport_width * u.x, viewport_width * u.y, viewport_width * u.z);
						const float3 vertical = make_float3(viewport_height * v.x, viewport_height * v.y, viewport_height * v.z);
						const float3 lower_left_corner = make_float3(
							lookfrom.x - horizontal.x / 2.0f - vertical.x / 2.0f - w.x,
							lookfrom.y - horizontal.y / 2.0f - vertical.y / 2.0f - w.y,
							lookfrom.z - horizontal.z / 2.0f - vertical.z / 2.0f - w.z
						);

						auto pack_float3 = [](float* dest, int offset, const float3& v) {
							dest[offset] = v.x; dest[offset + 1] = v.y; dest[offset + 2] = v.z;
						};

						pack_float3(camera_params, 0, lookfrom);
						pack_float3(camera_params, 3, lower_left_corner);
						pack_float3(camera_params, 6, horizontal);
						pack_float3(camera_params, 9, vertical);
					}
					break;

				case 5:  // Colored Quads
						build_quads_scene(scene);

						// Configure camera
					{
						constexpr float kPi = 3.14159265358979323846f;
						const float3 lookfrom = make_float3(static_cast<float>(cam_x), static_cast<float>(cam_y), static_cast<float>(cam_z));
						const float3 lookat = make_float3(0.0f, 0.0f, 0.0f);
						const float3 vup = make_float3(0.0f, 1.0f, 0.0f);
						constexpr float vfov = 80.0f;  // Wide angle for quads
						const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);

						const float theta = vfov * kPi / 180.0f;
						const float h = tanf(theta / 2.0f);
						const float viewport_height = 2.0f * h;
						const float viewport_width = aspect * viewport_height;

						const float3 view_direction = make_float3(lookfrom.x - lookat.x, lookfrom.y - lookat.y, lookfrom.z - lookat.z);
						const float3 w = normalize(view_direction);
						const float3 u = normalize(cross(vup, w));
						const float3 v = cross(w, u);

						const float3 horizontal = make_float3(viewport_width * u.x, viewport_width * u.y, viewport_width * u.z);
						const float3 vertical = make_float3(viewport_height * v.x, viewport_height * v.y, viewport_height * v.z);
						const float3 lower_left_corner = make_float3(
							lookfrom.x - horizontal.x / 2.0f - vertical.x / 2.0f - w.x,
							lookfrom.y - horizontal.y / 2.0f - vertical.y / 2.0f - w.y,
							lookfrom.z - horizontal.z / 2.0f - vertical.z / 2.0f - w.z
						);

						auto pack_float3 = [](float* dest, int offset, const float3& v) {
							dest[offset] = v.x; dest[offset + 1] = v.y; dest[offset + 2] = v.z;
						};

										pack_float3(camera_params, 0, lookfrom);
											pack_float3(camera_params, 3, lower_left_corner);
											pack_float3(camera_params, 6, horizontal);
											pack_float3(camera_params, 9, vertical);
										}
										break;

									case 9: {  // Rough Metal Spheres (GGX)
										build_rough_metal_spheres(scene);

										// Camera: vfov=35, lookfrom=(cam_x,cam_y,cam_z), lookat=(0,1,0)
										constexpr float kPi9 = 3.14159265358979323846f;
										const float3 lookfrom9 = make_float3(static_cast<float>(cam_x), static_cast<float>(cam_y), static_cast<float>(cam_z));
										const float3 lookat9   = make_float3(0.0f, 1.0f, 0.0f);
										const float3 vup9      = make_float3(0.0f, 1.0f, 0.0f);
										constexpr float vfov9  = 35.0f;
										const float aspect9    = static_cast<float>(image_width) / static_cast<float>(image_height);
										const float h9         = tanf((vfov9 * kPi9 / 180.0f) / 2.0f);
										const float vh9        = 2.0f * h9;
										const float vw9        = aspect9 * vh9;
										const float3 wd9 = normalize(make_float3(lookfrom9.x - lookat9.x, lookfrom9.y - lookat9.y, lookfrom9.z - lookat9.z));
										const float3 u9  = normalize(cross(vup9, wd9));
										const float3 v9  = cross(wd9, u9);
										const float3 horiz9 = make_float3(vw9*u9.x, vw9*u9.y, vw9*u9.z);
										const float3 vert9  = make_float3(vh9*v9.x, vh9*v9.y, vh9*v9.z);
										const float3 llc9   = make_float3(
											lookfrom9.x - horiz9.x/2.0f - vert9.x/2.0f - wd9.x,
											lookfrom9.y - horiz9.y/2.0f - vert9.y/2.0f - wd9.y,
											lookfrom9.z - horiz9.z/2.0f - vert9.z/2.0f - wd9.z);
										camera_params[0]=lookfrom9.x; camera_params[1]=lookfrom9.y; camera_params[2]=lookfrom9.z;
										camera_params[3]=llc9.x;      camera_params[4]=llc9.y;      camera_params[5]=llc9.z;
										camera_params[6]=horiz9.x;    camera_params[7]=horiz9.y;    camera_params[8]=horiz9.z;
										camera_params[9]=vert9.x;     camera_params[10]=vert9.y;    camera_params[11]=vert9.z;
										break;
									}

									case 10:  // Cornell Rough Metal (GGX)
								build_cornell_rough_metal(scene);

								// Same camera as Cornell Box (lookat center of box)
								// (camera block below handles this)
								// fallthrough intentional -- camera identical to case 11
								goto cornell_box_camera;

							case 11:  // Cornell Rough Glass (GGX)
											build_cornell_rough_glass(scene);
											cornell_box_camera:

											case 12:  // Cornell Conductor (GGX + complex Fresnel, pbrt-v4 ConductorBxDF)
												if (scene_id == 12) build_cornell_conductor(scene);
												// fallthrough

												case 13:  // Cornell Coated Diffuse (pbrt-v4 CoatedDiffuseBxDF)
												if (scene_id == 13) build_cornell_coated_diffuse(scene);
												// fallthrough

													case 14:  // Cornell Thin Glass (pbrt-v4 ThinDielectricBxDF)
													if (scene_id == 14) build_cornell_thin_glass(scene);
													// fallthrough

														case 15:  // Cornell Coated Conductor (pbrt-v4 CoatedConductorBxDF)
														if (scene_id == 15) build_cornell_coated_conductor(scene);
														// fallthrough

														case 16:  // Cornell Wax Slab (pbrt-v4 DiffuseTransmissionBxDF)
														if (scene_id == 16) build_cornell_wax_slab(scene);
														// fallthrough

														case 17:  // Cornell Crystal (pbrt-v4 NormalizedFresnelBxDF)
														if (scene_id == 17) build_cornell_crystal(scene);
														// fallthrough

														case 25:  // Spotlight Cornell (pbrt-v4 SpotLight)
														if (scene_id == 25) build_spotlight_cornell_gpu(scene);
														// fallthrough

														case 26:  // Distant Light Cornell (pbrt-v4 DistantLight)
														if (scene_id == 26) build_distant_light_cornell_gpu(scene);
														// fallthrough

														case 27:  // Point Light Cornell (pbrt-v4 PointLight)
														if (scene_id == 27) build_point_light_cornell_gpu(scene);
														// fallthrough

														case 28:  // Goniometric Light Cornell (pbrt-v4 GoniometricLight)
														if (scene_id == 28) build_goniometric_cornell_gpu(scene);
														// fallthrough

														case 29:  // Projection Light Cornell (pbrt-v4 ProjectionLight)
														if (scene_id == 29) build_projection_cornell_gpu(scene);
														// fallthrough

														case 35:  // Portal Infinite Light (pbrt-v4 PortalImageInfiniteLight)
														if (scene_id == 35) {
															build_portal_light_scene_gpu(scene);
															if (out_camera_extra) out_camera_extra->backgroundColor = make_float3(1.0f, 1.2f, 1.5f);
														}
														// fallthrough

														case 7:  // Cornell Smoke (constant_medium)
														if (scene_id == 7) build_cornell_smoke_gpu(scene);
														// fallthrough

														case 30:  // Homogeneous Medium (constant_medium, HG g=0.3)
														if (scene_id == 30) build_homogeneous_medium_scene_gpu(scene);
													{
									constexpr float kPi = 3.14159265358979323846f;
									const float3 lookfrom = make_float3(static_cast<float>(cam_x), static_cast<float>(cam_y), static_cast<float>(cam_z));
									const float3 lookat = make_float3(278.0f, 278.0f, 278.0f);
									const float3 vup = make_float3(0.0f, 1.0f, 0.0f);
									constexpr float vfov = 40.0f;
									const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);

									const float theta = vfov * kPi / 180.0f;
									const float h = tanf(theta / 2.0f);
									const float viewport_height = 2.0f * h;
									const float viewport_width = aspect * viewport_height;

									const float3 view_direction = make_float3(lookfrom.x - lookat.x, lookfrom.y - lookat.y, lookfrom.z - lookat.z);
									const float3 w = normalize(view_direction);
									const float3 u = normalize(cross(vup, w));
									const float3 v = cross(w, u);

									const float3 horizontal = make_float3(viewport_width * u.x, viewport_width * u.y, viewport_width * u.z);
									const float3 vertical   = make_float3(viewport_height * v.x, viewport_height * v.y, viewport_height * v.z);
									const float3 lower_left_corner = make_float3(
										lookfrom.x - horizontal.x / 2.0f - vertical.x / 2.0f - w.x,
										lookfrom.y - horizontal.y / 2.0f - vertical.y / 2.0f - w.y,
										lookfrom.z - horizontal.z / 2.0f - vertical.z / 2.0f - w.z
									);

									auto pack_float3 = [](float* dest, int offset, const float3& v) {
										dest[offset] = v.x; dest[offset + 1] = v.y; dest[offset + 2] = v.z;
									};

									pack_float3(camera_params, 0, lookfrom);
									pack_float3(camera_params, 3, lower_left_corner);
									pack_float3(camera_params, 6, horizontal);
									pack_float3(camera_params, 9, vertical);
								}
								break;

							case 22: {  // Depth of Field (thin-lens perspective camera)
								build_depth_of_field_gpu(scene);
								constexpr float kPi = 3.14159265358979323846f;
								const float3 lookfrom = make_float3(0.0f, 2.0f, 9.0f);
								const float3 lookat   = make_float3(0.0f, 1.0f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								constexpr float vfov = 20.0f;            // matches CPU CameraConfig row for scene 22
								constexpr float defocus_angle = 10.0f;   // ditto
								constexpr float focus_dist    = 9.0f;    // ditto
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);

								const float theta = vfov * kPi / 180.0f;
								const float h = tanf(theta / 2.0f);
								const float viewport_height = 2.0f * h * focus_dist;
								const float viewport_width  = aspect * viewport_height;

								const float3 w = normalize(make_float3(lookfrom.x - lookat.x, lookfrom.y - lookat.y, lookfrom.z - lookat.z));
								const float3 u = normalize(cross(vup, w));
								const float3 v = cross(w, u);

								const float3 horizontal = make_float3(viewport_width * u.x, viewport_width * u.y, viewport_width * u.z);
								const float3 vertical   = make_float3(viewport_height * v.x, viewport_height * v.y, viewport_height * v.z);
								const float3 lower_left_corner = make_float3(
									lookfrom.x - horizontal.x / 2.0f - vertical.x / 2.0f - focus_dist * w.x,
									lookfrom.y - horizontal.y / 2.0f - vertical.y / 2.0f - focus_dist * w.y,
									lookfrom.z - horizontal.z / 2.0f - vertical.z / 2.0f - focus_dist * w.z
								);

								auto pack_float3 = [](float* dest, int offset, const float3& vv) {
									dest[offset] = vv.x; dest[offset + 1] = vv.y; dest[offset + 2] = vv.z;
								};
								pack_float3(camera_params, 0, lookfrom);
								pack_float3(camera_params, 3, lower_left_corner);
								pack_float3(camera_params, 6, horizontal);
								pack_float3(camera_params, 9, vertical);

								if (out_camera_extra) {
									out_camera_extra->kind = CameraKind::Perspective;
									out_camera_extra->origin = lookfrom;
									out_camera_extra->lower_left_corner = lower_left_corner;
									out_camera_extra->horizontal = horizontal;
									out_camera_extra->vertical = vertical;
									// pbrt-v4/book-style thin-lens disk basis, scaled by focus_dist and
									// half the defocus cone angle - matches src/TheRestOfYourLife/
									// camera.h's defocus_disk_u/v exactly.
									const float defocus_radius = focus_dist * tanf((defocus_angle * kPi / 180.0f) / 2.0f);
									out_camera_extra->defocus_disk_u = make_float3(u.x * defocus_radius, u.y * defocus_radius, u.z * defocus_radius);
									out_camera_extra->defocus_disk_v = make_float3(v.x * defocus_radius, v.y * defocus_radius, v.z * defocus_radius);
								}
								break;
							}

							case 32: {  // Orthographic Camera (parallel projection)
								build_ortho_camera_scene_gpu(scene);
								const float3 lookfrom = make_float3(0.0f, 3.0f, 12.0f);
								const float3 lookat   = make_float3(0.0f, 1.0f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);

								// compute_screen_window-equivalent aspect-correct default, then
								// scaled x8 - matches CPU's setup_camera lambda for scene 32.
								float xmin, xmax, ymin, ymax;
								if (aspect >= 1.0f) { xmin = -aspect; xmax = aspect; ymin = -1.0f; ymax = 1.0f; }
								else                { xmin = -1.0f; xmax = 1.0f; ymin = -1.0f / aspect; ymax = 1.0f / aspect; }
								constexpr float kScreenScale = 8.0f;
								xmin *= kScreenScale; xmax *= kScreenScale; ymin *= kScreenScale; ymax *= kScreenScale;

								const float3 w = normalize(make_float3(lookfrom.x - lookat.x, lookfrom.y - lookat.y, lookfrom.z - lookat.z));
								const float3 u = normalize(cross(vup, w));
								const float3 v = cross(w, u);

								const float3 horizontal = make_float3((xmax - xmin) * u.x, (xmax - xmin) * u.y, (xmax - xmin) * u.z);
								const float3 vertical   = make_float3((ymax - ymin) * v.x, (ymax - ymin) * v.y, (ymax - ymin) * v.z);
								const float3 lower_left_corner = make_float3(
									lookfrom.x + xmin * u.x + ymin * v.x,
									lookfrom.y + xmin * u.y + ymin * v.y,
									lookfrom.z + xmin * u.z + ymin * v.z
								);

								auto pack_float3 = [](float* dest, int offset, const float3& vv) {
									dest[offset] = vv.x; dest[offset + 1] = vv.y; dest[offset + 2] = vv.z;
								};
								pack_float3(camera_params, 0, lookfrom);
								pack_float3(camera_params, 3, lower_left_corner);
								pack_float3(camera_params, 6, horizontal);
								pack_float3(camera_params, 9, vertical);

								if (out_camera_extra) {
									out_camera_extra->kind = CameraKind::Orthographic;
									out_camera_extra->lower_left_corner = lower_left_corner;
									out_camera_extra->horizontal = horizontal;
									out_camera_extra->vertical = vertical;
									out_camera_extra->w = make_float3(-w.x, -w.y, -w.z);  // forward = negated look-from/look-at "backward" w
								}
								break;
							}

							case 33: {  // Spherical (equirectangular) Camera
								build_spherical_camera_scene_gpu(scene);
								// Matches CPU: SphericalCamera constructed with no camera_to_world
								// arg, defaulting to identity - camera at world origin, world-axis
								// basis (see build_spherical_camera_scene_gpu's comment on why the
								// ground is placed below y=0 to accommodate this).
								auto pack_float3 = [](float* dest, int offset, const float3& vv) {
									dest[offset] = vv.x; dest[offset + 1] = vv.y; dest[offset + 2] = vv.z;
								};
								const float3 zero = make_float3(0.0f, 0.0f, 0.0f);
								pack_float3(camera_params, 0, zero);
								pack_float3(camera_params, 3, zero);
								pack_float3(camera_params, 6, zero);
								pack_float3(camera_params, 9, zero);

								if (out_camera_extra) {
									out_camera_extra->kind = CameraKind::Spherical;
									out_camera_extra->origin = zero;
									out_camera_extra->su = make_float3(1.0f, 0.0f, 0.0f);
									out_camera_extra->sv = make_float3(0.0f, 1.0f, 0.0f);
									out_camera_extra->sw = make_float3(0.0f, 0.0f, 1.0f);
								}
								break;
							}

							case 24: {  // HDRI Sky (flat-color background - see backgroundColor's comment)
								build_hdri_sky_world_gpu(scene);
								constexpr float kPi = 3.14159265358979323846f;
								const float3 lookfrom = make_float3(0.0f, 2.0f, 10.0f);
								const float3 lookat   = make_float3(0.0f, 1.0f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								constexpr float vfov = 30.0f;  // matches CPU CameraConfig row for scene 24
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);

								const float theta = vfov * kPi / 180.0f;
								const float h = tanf(theta / 2.0f);
								const float viewport_height = 2.0f * h;
								const float viewport_width  = aspect * viewport_height;

								const float3 w = normalize(make_float3(lookfrom.x - lookat.x, lookfrom.y - lookat.y, lookfrom.z - lookat.z));
								const float3 u = normalize(cross(vup, w));
								const float3 v = cross(w, u);

								const float3 horizontal = make_float3(viewport_width * u.x, viewport_width * u.y, viewport_width * u.z);
								const float3 vertical   = make_float3(viewport_height * v.x, viewport_height * v.y, viewport_height * v.z);
								const float3 lower_left_corner = make_float3(
									lookfrom.x - horizontal.x / 2.0f - vertical.x / 2.0f - w.x,
									lookfrom.y - horizontal.y / 2.0f - vertical.y / 2.0f - w.y,
									lookfrom.z - horizontal.z / 2.0f - vertical.z / 2.0f - w.z
								);

								auto pack_float3 = [](float* dest, int offset, const float3& vv) {
									dest[offset] = vv.x; dest[offset + 1] = vv.y; dest[offset + 2] = vv.z;
								};
								pack_float3(camera_params, 0, lookfrom);
								pack_float3(camera_params, 3, lower_left_corner);
								pack_float3(camera_params, 6, horizontal);
								pack_float3(camera_params, 9, vertical);

								if (out_camera_extra) {
									// Matches CPU build_hdri_sky()'s solid-color sky_light(0.3,0.6,1.0)
									// (see GpuCameraParams::backgroundColor's comment for why this is a
									// flat color, not an importance-sampled environment map).
									out_camera_extra->backgroundColor = make_float3(0.3f, 0.6f, 1.0f);
								}
								break;
							}

							case 31: {  // Cloud Medium (constant_medium, HG g=0.05)
								build_cloud_medium_scene_gpu(scene);
								constexpr float kPi = 3.14159265358979323846f;
								const float3 lookfrom = make_float3(0.0f, 5.0f, 20.0f);
								const float3 lookat   = make_float3(0.0f, 2.0f, 0.0f);
								const float3 vup       = make_float3(0.0f, 1.0f, 0.0f);
								constexpr float vfov = 20.0f;  // matches CPU CameraConfig row for scene 31
								const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);

								const float theta = vfov * kPi / 180.0f;
								const float h = tanf(theta / 2.0f);
								const float viewport_height = 2.0f * h;
								const float viewport_width  = aspect * viewport_height;

								const float3 w = normalize(make_float3(lookfrom.x - lookat.x, lookfrom.y - lookat.y, lookfrom.z - lookat.z));
								const float3 u = normalize(cross(vup, w));
								const float3 v = cross(w, u);

								const float3 horizontal = make_float3(viewport_width * u.x, viewport_width * u.y, viewport_width * u.z);
								const float3 vertical   = make_float3(viewport_height * v.x, viewport_height * v.y, viewport_height * v.z);
								const float3 lower_left_corner = make_float3(
									lookfrom.x - horizontal.x / 2.0f - vertical.x / 2.0f - w.x,
									lookfrom.y - horizontal.y / 2.0f - vertical.y / 2.0f - w.y,
									lookfrom.z - horizontal.z / 2.0f - vertical.z / 2.0f - w.z
								);

								auto pack_float3 = [](float* dest, int offset, const float3& vv) {
									dest[offset] = vv.x; dest[offset + 1] = vv.y; dest[offset + 2] = vv.z;
								};
								pack_float3(camera_params, 0, lookfrom);
								pack_float3(camera_params, 3, lower_left_corner);
								pack_float3(camera_params, 6, horizontal);
								pack_float3(camera_params, 9, vertical);
								if (out_camera_extra) {
									// Matches CPU CameraConfig bg for scene 31 - this is the scene's
									// ONLY light source (no emissive geometry), so a missing/black
									// background here means zero illumination anywhere in the image.
									out_camera_extra->backgroundColor = make_float3(0.5f, 0.7f, 1.0f);
								}
								break;
							}

							default: {
									const SceneDesc* desc = find_scene_desc(scene_id);
									if (desc) {
										std::cerr << "[OptiX] Scene '" << desc->name
											  << "' (id=" << scene_id << ") is not implemented for GPU rendering.\n"
											  << "[OptiX] Use CPU renderer for this scene.\n";
									} else {
										std::cerr << "[OptiX] Unknown scene id=" << scene_id << ".\n";
									}
									return false;
								}
							}

							return true;
						}

