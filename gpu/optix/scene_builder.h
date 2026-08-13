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

	// ---- object instancing ------------------------------------------------
	// Geometry that exists once and is placed many times. Kept in a SEPARATE
	// array from `triangles` on purpose: these are in their definition's
	// OBJECT space, and appending them to the world-space list would put every
	// instanced object at the origin for any consumer that has not learned
	// about groups yet. Separation makes ignoring them safe rather than wrong.
	//
	// The renderer builds one GAS per group over its sub-range, and the base
	// index is what the device adds to optixGetPrimitiveIndex() to recover the
	// global primitive - see LaunchParams::instancePrimBase.
	//
	// Triangles and spheres are tracked separately because OptiX cannot put
	// them in one acceleration structure (native triangles vs custom AABB
	// primitives), so a group holding both becomes TWO GASes and a single
	// placement of it becomes two instance-hierarchy entries.
	struct InstanceGroupGPU {
		int triangleBase = 0;    // first triangle in `instanceTriangles`
		int triangleCount = 0;
		int sphereBase = 0;      // first sphere in `instanceSpheres`
		int sphereCount = 0;
	};
	struct InstancePlacementGPU {
		int group = -1;
		float transform[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};  // OptiX 3x4 row-major
	};
	std::vector<TriangleData> instanceTriangles;   // object space
	std::vector<SphereData> instanceSpheres;       // object space
	std::vector<InstanceGroupGPU> instanceGroups;
	std::vector<InstancePlacementGPU> instancePlacements;

	// Textures referenced by MaterialData::textureIdx (see optix_types.h's
	// TextureData). texturePixels is one shared flat 8-bit RGB buffer every
	// Image-kind texture's pixelOffset indexes into. Empty for every scene
	// that doesn't use MaterialData::textureIdx.
	std::vector<TextureData> textures;
	std::vector<unsigned char> texturePixels;

	// RealisticCamera (CameraKind::Realistic) host-precomputed tables - see
	// optix_types.h's GpuLensElement/GpuExitPupilBounds. Empty for every
	// other scene.
	std::vector<GpuLensElement> lensElements;
	std::vector<GpuExitPupilBounds> exitPupilBounds;

	// Light tracking for MIS
	std::vector<int> lightIndices;      // Indices into sphere/quad arrays
	std::vector<GpuLightKind> lightKinds;   // how to sample lightIndices[i]

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
