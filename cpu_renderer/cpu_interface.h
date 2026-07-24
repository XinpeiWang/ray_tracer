/// @file cpu_interface.h
/// @brief CPU Renderer C Interface
/// @details This header defines the C-linkage interface for the CPU-based ray tracer.
/// The CPU renderer uses:
///   - Multithreaded C++ path tracing
///   - Importance sampling (PDF-based lighting)
///   - The shared Cornell box scene definition
///
/// Camera System:
///   - Camera position (lookfrom) is configurable via cam_x, cam_y, cam_z
///   - Camera target (lookat) is fixed at Cornell box center: (278, 278, 278)
///   - View up vector (vup) is fixed at (0, 1, 0)
///   - Vertical field of view (vfov) is fixed at 40 degrees
///
/// This interface allows the launcher (main.cpp) to call the CPU renderer
/// as an in-process library function with C linkage (no name mangling).

#ifndef CPU_INTERFACE_H
#define CPU_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Render the Cornell box scene using CPU path tracing with importance sampling
/// @details The Cornell box scene is predefined in src/TheRestOfYourLife/cornell_box_scene.h
/// and includes:
///   - Cornell box walls (red, green, white)
///   - Ceiling light source
///   - Glass sphere
///   - Rotated box
/// 
/// Camera configuration:
///   - lookfrom: (cam_x, cam_y, cam_z) - user-specified position
///   - lookat:   (278, 278, 278) - fixed at Cornell box center
///   - vup:      (0, 1, 0) - fixed up direction
///   - vfov:     40 degrees - fixed field of view
/// 
/// @param width         Image width in pixels (height = width for square aspect)
/// @param height        Image height in pixels
/// @param spp           Samples per pixel (higher = less noise, slower render)
/// @param max_depth     Maximum ray bounce depth (higher = more realistic lighting)
/// @param output_path   Output PPM file path (e.g., "C:/path/to/image.ppm")
/// @param scene_id      Scene selector (0=Cornell Box, 1=Bouncing Spheres, etc.)
/// @param cam_x         Camera position X coordinate (default: 278)
/// @param cam_y         Camera position Y coordinate (default: 278)
/// @param cam_z         Camera position Z coordinate (default: -800)
/// @return 0 on success, non-zero error code on failure
int cpu_render_main(
    int width,
    int height,
    int spp,
    int max_depth,
    const char* output_path,
    int scene_id,
    double cam_x,
    double cam_y,
    double cam_z
);

/// Scene metadata C API -- lets the GUI query the registry without C++ headers
/// @return total number of registered scenes
int cpu_scene_count();

/// @param index  position in registry (0..cpu_scene_count()-1)
/// @return scene id, or -1 if index out of range
int cpu_scene_id(int index);

/// @param index  position in registry
/// @return scene name string, or "" if out of range
const char* cpu_scene_name(int index);

/// @param index  position in registry
/// @return short description string
const char* cpu_scene_description(int index);

/// @param index  position in registry
/// @return performance hint: "Fast", "Medium", "Slow", "Very Slow"
const char* cpu_scene_performance(int index);

/// @param index  position in registry
/// @return recommended samples-per-pixel
int cpu_scene_recommended_spp(int index);

/// @param index  position in registry
/// @return 1 if scene requires external files (earthmap.jpg etc), else 0
int cpu_scene_requires_files(int index);

/// @param index  position in registry
/// @return 1 if GPU compatible, else 0
int cpu_scene_gpu_compatible(int index);

#ifdef __cplusplus
}
#endif

#endif // CPU_INTERFACE_H
