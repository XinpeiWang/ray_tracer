// ============================================================================
// CPU Renderer C Interface
// ============================================================================
// This header defines the C-linkage interface for the CPU-based ray tracer.
// The CPU renderer uses:
//   - Multithreaded C++ path tracing
//   - Importance sampling (PDF-based lighting)
//   - The shared Cornell box scene definition
//
// Camera System:
//   - Camera position (lookfrom) is configurable via cam_x, cam_y, cam_z
//   - Camera target (lookat) is fixed at Cornell box center: (278, 278, 278)
//   - View up vector (vup) is fixed at (0, 1, 0)
//   - Vertical field of view (vfov) is fixed at 40 degrees
//
// This interface allows the launcher (main.cpp) to call the CPU renderer
// as an in-process library function with C linkage (no name mangling).
// ============================================================================

#ifndef CPU_INTERFACE_H
#define CPU_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Render the Cornell box scene using CPU path tracing with importance sampling.
 * 
 * The Cornell box scene is predefined in src/TheRestOfYourLife/cornell_box_scene.h
 * and includes:
 *   - Cornell box walls (red, green, white)
 *   - Ceiling light source
 *   - Glass sphere
 *   - Rotated box
 * 
 * Camera configuration:
 *   - lookfrom: (cam_x, cam_y, cam_z) - user-specified position
 *   - lookat:   (278, 278, 278) - fixed at Cornell box center
 *   - vup:      (0, 1, 0) - fixed up direction
 *   - vfov:     40 degrees - fixed field of view
 * 
 * @param width         Image width in pixels (height = width for square aspect)
 * @param height        Image height in pixels
 * @param spp           Samples per pixel (higher = less noise, slower render)
 * @param max_depth     Maximum ray bounce depth (higher = more realistic lighting)
 * @param output_path   Output PPM file path (e.g., "C:/path/to/image.ppm")
 * @param cam_x         Camera position X coordinate (default: 278)
 * @param cam_y         Camera position Y coordinate (default: 278)
 * @param cam_z         Camera position Z coordinate (default: -800)
 * @return 0 on success, non-zero on failure
 */
int cpu_render_main(int width, int height, int spp, int max_depth, const char* output_path, 
                    double cam_x, double cam_y, double cam_z);

#ifdef __cplusplus
}
#endif

#endif // CPU_INTERFACE_H
