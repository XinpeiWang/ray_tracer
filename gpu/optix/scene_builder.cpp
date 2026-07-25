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
	const double cam_z
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

