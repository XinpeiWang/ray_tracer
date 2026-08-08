// OptiX renderer C API
// Matches the signature of gpu/cuda/gpu_interface.h for drop-in replacement

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Check if GPU/OptiX rendering is available
bool optix_is_available();

// Main OptiX rendering entry point
// Supports multiple scenes via scene_id parameter
int optix_render_main(
	int image_width,
	int image_height,
	int samples_per_pixel,
	int max_depth,
	const char* output_path,
	int scene_id,
	double cam_x,
	double cam_y,
	double cam_z
);

// Returns the number of emissive light primitives (quads+spheres) that
// gpu/optix/scene_builder.cpp's build_scene() would upload for scene_id, or
// -1 if the scene fails to build. build_scene() is pure host-side C++ (no
// OptiX/CUDA device calls) - this works without a GPU present, unlike
// optix_render_main(). Used by CPU/GPU scene-parity tests (see
// tests/integration/cpu_gpu_comparison_tests.cpp) to catch cases where a
// scene's CPU and GPU builders have drifted out of sync on light count.
int gpu_scene_light_count(int scene_id, int image_width, int image_height);

#ifdef __cplusplus
}
#endif
