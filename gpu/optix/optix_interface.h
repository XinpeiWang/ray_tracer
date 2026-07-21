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

#ifdef __cplusplus
}
#endif
