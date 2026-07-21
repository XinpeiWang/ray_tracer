// Scene Builder Implementation
// Converts shared scene definitions to OptiX geometry

#include "scene_builder.h"
#include "optix_math_helpers.h"
#include <cmath>

// Build Cornell Box scene
static void build_cornell_box(SceneData& scene) {
	// Materials
	int mat_red = scene.materials.size();
	scene.materials.push_back({
		MaterialType::Lambertian,
		make_float3(0.65f, 0.05f, 0.05f),  // albedo
		0.0f, 0.0f,  // fuzz, ior (unused)
		make_float3(0.0f, 0.0f, 0.0f)  // emission
	});

	int mat_white = scene.materials.size();
	scene.materials.push_back({
		MaterialType::Lambertian,
		make_float3(0.73f, 0.73f, 0.73f),
		0.0f, 0.0f,
		make_float3(0.0f, 0.0f, 0.0f)
	});

	int mat_green = scene.materials.size();
	scene.materials.push_back({
		MaterialType::Lambertian,
		make_float3(0.12f, 0.45f, 0.15f),
		0.0f, 0.0f,
		make_float3(0.0f, 0.0f, 0.0f)
	});

	int mat_light = scene.materials.size();
	scene.materials.push_back({
		MaterialType::DiffuseLight,
		make_float3(0.0f, 0.0f, 0.0f),  // albedo (unused)
		0.0f, 0.0f,
		make_float3(15.0f, 15.0f, 15.0f)  // emission
	});

	int mat_glass = scene.materials.size();
	scene.materials.push_back({
		MaterialType::Dielectric,
		make_float3(1.0f, 1.0f, 1.0f),  // albedo (unused for dielectric)
		0.0f,
		1.5f,  // ior
		make_float3(0.0f, 0.0f, 0.0f)
	});

	// Cornell Box quads (walls, floor, ceiling)
	// Green wall (left)
	QuadData wall_left;
	wall_left.Q = make_float3(555.0f, 0.0f, 0.0f);
	wall_left.u = make_float3(0.0f, 555.0f, 0.0f);
	wall_left.v = make_float3(0.0f, 0.0f, 555.0f);
	wall_left.w = cross(wall_left.u, wall_left.v);
	wall_left.normal = normalize(cross(wall_left.u, wall_left.v));
	wall_left.D = dot(wall_left.normal, wall_left.Q);
	wall_left.materialIdx = mat_green;
	scene.quads.push_back(wall_left);

	// Red wall (right)
	QuadData wall_right;
	wall_right.Q = make_float3(0.0f, 0.0f, 0.0f);
	wall_right.u = make_float3(0.0f, 555.0f, 0.0f);
	wall_right.v = make_float3(0.0f, 0.0f, 555.0f);
	wall_right.w = cross(wall_right.u, wall_right.v);
	wall_right.normal = normalize(cross(wall_right.u, wall_right.v));
	wall_right.D = dot(wall_right.normal, wall_right.Q);
	wall_right.materialIdx = mat_red;
	scene.quads.push_back(wall_right);

	// Light (ceiling)
	QuadData light_quad;
	light_quad.Q = make_float3(343.0f, 554.0f, 332.0f);
	light_quad.u = make_float3(-130.0f, 0.0f, 0.0f);
	light_quad.v = make_float3(0.0f, 0.0f, -105.0f);
	light_quad.w = cross(light_quad.u, light_quad.v);
	light_quad.normal = normalize(cross(light_quad.u, light_quad.v));
	light_quad.D = dot(light_quad.normal, light_quad.Q);
	light_quad.materialIdx = mat_light;
	scene.quads.push_back(light_quad);

	// White ceiling
	QuadData ceiling;
	ceiling.Q = make_float3(0.0f, 555.0f, 0.0f);
	ceiling.u = make_float3(555.0f, 0.0f, 0.0f);
	ceiling.v = make_float3(0.0f, 0.0f, 555.0f);
	ceiling.w = cross(ceiling.u, ceiling.v);
	ceiling.normal = normalize(cross(ceiling.u, ceiling.v));
	ceiling.D = dot(ceiling.normal, ceiling.Q);
	ceiling.materialIdx = mat_white;
	scene.quads.push_back(ceiling);

	// White floor
	QuadData floor;
	floor.Q = make_float3(0.0f, 0.0f, 0.0f);
	floor.u = make_float3(555.0f, 0.0f, 0.0f);
	floor.v = make_float3(0.0f, 0.0f, 555.0f);
	floor.w = cross(floor.u, floor.v);
	floor.normal = normalize(cross(floor.u, floor.v));
	floor.D = dot(floor.normal, floor.Q);
	floor.materialIdx = mat_white;
	scene.quads.push_back(floor);

	// White back wall
	QuadData back_wall;
	back_wall.Q = make_float3(0.0f, 0.0f, 555.0f);
	back_wall.u = make_float3(555.0f, 0.0f, 0.0f);
	back_wall.v = make_float3(0.0f, 555.0f, 0.0f);
	back_wall.w = cross(back_wall.u, back_wall.v);
	back_wall.normal = normalize(cross(back_wall.u, back_wall.v));
	back_wall.D = dot(back_wall.normal, back_wall.Q);
	back_wall.materialIdx = mat_white;
	scene.quads.push_back(back_wall);

	// Glass sphere
	SphereData glass_sphere;
	glass_sphere.center = make_float3(190.0f, 90.0f, 190.0f);
	glass_sphere.radius = 90.0f;
	glass_sphere.materialIdx = mat_glass;
	scene.spheres.push_back(glass_sphere);

	// Aluminum box (represented as white sphere for simplicity - TODO: add box support)
	SphereData box_sphere;
	box_sphere.center = make_float3(350.0f, 82.5f, 351.0f);
	box_sphere.radius = 82.5f;
	box_sphere.materialIdx = mat_white;
	scene.spheres.push_back(box_sphere);
}

bool build_scene(
	int scene_id,
	int image_width,
	int image_height,
	SceneData& scene,
	float* camera_params
) {
	// Clear scene
	scene.spheres.clear();
	scene.quads.clear();
	scene.materials.clear();

	// Build requested scene
	switch (scene_id) {
		case 0:  // Cornell Box
			build_cornell_box(scene);

			// Camera for Cornell Box
			{
				float3 lookfrom = make_float3(278.0f, 278.0f, -800.0f);
				float3 lookat = make_float3(278.0f, 278.0f, 278.0f);  // Look at center of box
				float3 vup = make_float3(0.0f, 1.0f, 0.0f);
				float vfov = 40.0f;
				float aspect = float(image_width) / float(image_height);

				// Camera calculation
				float theta = vfov * 3.14159265f / 180.0f;
				float h = tanf(theta / 2.0f);
				float viewport_height = 2.0f * h;
				float viewport_width = aspect * viewport_height;

				float3 w = normalize(make_float3(
					lookfrom.x - lookat.x,
					lookfrom.y - lookat.y,
					lookfrom.z - lookat.z
				));
				float3 u = normalize(cross(vup, w));
				float3 v = cross(w, u);

				float3 horizontal = make_float3(
					viewport_width * u.x,
					viewport_width * u.y,
					viewport_width * u.z
				);
				float3 vertical = make_float3(
					viewport_height * v.x,
					viewport_height * v.y,
					viewport_height * v.z
				);
				float3 lower_left_corner = make_float3(
					lookfrom.x - horizontal.x / 2.0f - vertical.x / 2.0f - w.x,
					lookfrom.y - horizontal.y / 2.0f - vertical.y / 2.0f - w.y,
					lookfrom.z - horizontal.z / 2.0f - vertical.z / 2.0f - w.z
				);

				// Pack camera parameters
				camera_params[0] = lookfrom.x;
				camera_params[1] = lookfrom.y;
				camera_params[2] = lookfrom.z;
				camera_params[3] = lower_left_corner.x;
				camera_params[4] = lower_left_corner.y;
				camera_params[5] = lower_left_corner.z;
				camera_params[6] = horizontal.x;
				camera_params[7] = horizontal.y;
				camera_params[8] = horizontal.z;
				camera_params[9] = vertical.x;
				camera_params[10] = vertical.y;
				camera_params[11] = vertical.z;
			}
			break;

		default:
			return false;  // Unknown scene
	}

	return true;
}

