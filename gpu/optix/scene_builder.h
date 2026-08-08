// Scene Builder for OptiX
// Converts shared scene definitions to OptiX geometry

#pragma once

#include "optix_types.h"
#include <vector>

// Scene container
struct SceneData {
	std::vector<SphereData> spheres;
	std::vector<QuadData> quads;
	std::vector<BilinearPatchData> bilinearPatches;
	std::vector<TriangleData> triangles;
	std::vector<MaterialData> materials;

	// RealisticCamera (CameraKind::Realistic) host-precomputed tables - see
	// optix_types.h's GpuLensElement/GpuExitPupilBounds. Empty for every
	// other scene.
	std::vector<GpuLensElement> lensElements;
	std::vector<GpuExitPupilBounds> exitPupilBounds;

	// Light tracking for MIS
	std::vector<int> lightIndices;      // Indices into sphere/quad arrays
	std::vector<bool> isLightSphere;    // True if sphere, false if quad

	// Punctual (delta) lights: point/spot/distant. Separate from the area
	// lights above - not geometry, evaluated deterministically every hit.
	std::vector<PunctualLightGPU> punctualLights;
};

// Build a scene and return geometry + camera
// camera_params: [origin(3), lower_left(3), horizontal(3), vertical(3)]
// out_camera_extra: when non-null, scenes using a non-default camera model
// (currently 22 DepthOfField, 32 OrthographicCamera, 33 SphericalCamera, 36
// RealisticCamera) fill
// it with their CameraKind + type-specific fields (see optix_types.h's
// GpuCameraParams). Scenes that don't need it leave *out_camera_extra
// untouched - callers should zero-init before calling and treat an
// unmodified (still-zeroed, kind=Perspective) result as "use camera_params
// as a plain perspective camera", matching every scene's prior behavior.
// force_camera_override: when true, scenes that would otherwise ignore
// cam_x/y/z and use their own fixed lookfrom (currently scenes 1 and 2 -
// see their case blocks) honor cam_x/y/z instead. Set by main.cpp's
// video-mode frame loop, which must animate the camera every frame
// regardless of a scene's single-image default.
bool build_scene(
	int scene_id,
	int image_width,
	int image_height,
	SceneData& scene,
	float* camera_params,
	double cam_x = 278.0,
	double cam_y = 278.0,
	double cam_z = -800.0,  // Far back view
	GpuCameraParams* out_camera_extra = nullptr,
	bool force_camera_override = false
);
