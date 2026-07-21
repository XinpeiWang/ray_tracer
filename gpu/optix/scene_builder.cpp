// Scene Builder Implementation
// Converts shared scene definitions to OptiX geometry

#include "scene_builder.h"
#include "optix_math_helpers.h"
#include <cmath>
#include <cassert>
#include <iostream>

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

	// Glass sphere (left)
	SphereData glass_sphere{};
	glass_sphere.center = make_float3(190.0f, 90.0f, 190.0f);
	glass_sphere.radius = 90.0f;
	glass_sphere.materialIdx = mat_glass;
	scene.spheres.push_back(glass_sphere);

	// White rotated box (right) - matching CPU scene definition
	// box(point3(0,0,0), point3(165,330,165), white) rotated 15° then translated (265,0,295)
	add_box(scene,
		make_float3(0.0f, 0.0f, 0.0f),
		make_float3(165.0f, 330.0f, 165.0f),
		mat_white,
		15.0f,  // rotation angle in degrees
		make_float3(265.0f, 0.0f, 295.0f));  // translation offset
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

		default:
			return false;  // Unknown scene ID
	}

	return true;
}

