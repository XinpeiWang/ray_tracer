#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Check if a CUDA-capable GPU is available.
// Returns 1 if GPU is available, 0 otherwise.
int gpu_is_available();

// Render an image on the GPU and write a PPM to out_path.
// Returns 0 on success, non-zero on failure.
int gpu_render_main(int image_width, int image_height, int samples_per_pixel, int max_depth, const char* out_path);

#ifdef __cplusplus
}
#endif
