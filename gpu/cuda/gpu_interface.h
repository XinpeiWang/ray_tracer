// ============================================================================
// GPU Renderer C Interface (CUDA)
// ============================================================================
// This header defines the C-linkage interface for the CUDA GPU ray tracer.
// The GPU renderer uses:
//   - CUDA kernels for massively parallel ray tracing
//   - Naive path tracing (no importance sampling)
//   - Scene data serialized to GPU-friendly POD arrays
//   - The shared Cornell box scene definition
//
// Performance Note:
//   The GPU uses naive path tracing while CPU uses importance sampling (PDFs).
//   GPU requires ~100x more samples to match CPU quality at low sample counts.
//   Example: CPU at 10 spp ≈ GPU at 1000 spp
//
// Camera System:
//   - Camera position (lookfrom) is configurable via cam_x, cam_y, cam_z
//   - Camera target (lookat) is fixed at Cornell box center: (278, 278, 278)
//   - View up vector (vup) is fixed at (0, 1, 0)
//   - Vertical field of view (vfov) is fixed at 40 degrees
//
// This interface allows the launcher (main.cpp) to call the GPU renderer
// as an in-process function with C linkage (no name mangling).
// ============================================================================

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Check if a CUDA-capable GPU is available at runtime.
 * 
 * This function queries the CUDA driver to detect GPUs with compute capability
 * required for the ray tracer.
 * 
 * @return 1 if a CUDA-capable GPU is available, 0 otherwise
 */
int gpu_is_available();

/**
 * Render the Cornell box scene using CUDA GPU path tracing.
 * 
 * The Cornell box scene is predefined in src/TheRestOfYourLife/cornell_box_scene.h
 * and serialized to GPU-friendly POD arrays via scene_serializer.cpp.
 * 
 * Rendering approach:
 *   - Naive path tracing (no importance sampling)
 *   - Requires more samples than CPU for equivalent quality
 *   - Massively parallel execution on GPU
 * 
 * Camera configuration:
 *   - lookfrom: (cam_x, cam_y, cam_z) - user-specified position
 *   - lookat:   (278, 278, 278) - fixed at Cornell box center
 *   - vup:      (0, 1, 0) - fixed up direction
 *   - vfov:     40 degrees - fixed field of view
 * 
 * @param image_width       Image width in pixels
 * @param image_height      Image height in pixels
 * @param samples_per_pixel Samples per pixel (GPU needs ~100x more than CPU)
 * @param max_depth         Maximum ray bounce depth
 * @param out_path          Output PPM file path (e.g., "C:/path/to/image.ppm")
 * @param cam_x             Camera position X coordinate (default: 278)
 * @param cam_y             Camera position Y coordinate (default: 278)
 * @param cam_z             Camera position Z coordinate (default: -800)
 * @return 0 on success, non-zero on failure
 */
int gpu_render_main(int image_width, int image_height, int samples_per_pixel, int max_depth, const char* out_path,
                    double cam_x, double cam_y, double cam_z);

#ifdef __cplusplus
}
#endif
