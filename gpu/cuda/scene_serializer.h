#pragma once

#include "cuda_scene.h"

// C-style interface for scene serialization (avoids C++ complexity in CUDA compiler)
#ifdef __cplusplus
extern "C" {
#endif

// Serialize scene and return pointers to device-allocated arrays
// Caller must free returned pointers
void serialize_scene_arrays(
	int image_width, int image_height, int samples_per_pixel, int max_depth,
	SpherePOD** out_spheres, int* out_num_spheres,
	QuadPOD** out_quads, int* out_num_quads,
	MaterialPOD** out_materials, int* out_num_materials,
	CameraPOD* out_camera
);

void free_scene_arrays(SpherePOD* spheres, QuadPOD* quads, MaterialPOD* materials);

#ifdef __cplusplus
}
#endif
