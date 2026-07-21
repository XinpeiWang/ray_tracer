// Scene Builder for OptiX
// Converts shared scene definitions to OptiX geometry

#pragma once

#include "optix_types.h"
#include <vector>

// Scene container
struct SceneData {
	std::vector<SphereData> spheres;
	std::vector<QuadData> quads;
	std::vector<MaterialData> materials;
};

// Build a scene and return geometry + camera
// camera_params: [origin(3), lower_left(3), horizontal(3), vertical(3)]
bool build_scene(
	int scene_id,
	int image_width,
	int image_height,
	SceneData& scene,
	float* camera_params
);
