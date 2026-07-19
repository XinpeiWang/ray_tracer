// ============================================================================
// GPU Scene Serialization Interface
// ============================================================================
// This header defines the interface for converting the Cornell box scene
// from C++ object-oriented structures to GPU-friendly POD (Plain Old Data) arrays.
//
// Purpose:
//   - CUDA kernels cannot easily work with C++ virtual functions & complex objects
//   - Scene data (spheres, quads, materials, camera) must be "flattened" into arrays
//   - This serializer builds the scene using shared Cornell box definition and
//     converts it to POD structs that can be copied to GPU device memory
//
// Camera Handling:
//   - Takes camera position parameters (cam_x, cam_y, cam_z) as input
//   - Sets lookfrom to the specified position
//   - Sets lookat to Cornell box center (278, 278, 278) - fixed
//   - Converts final camera to CameraPOD for GPU consumption
//
// Memory Management:
//   - serialize_scene_arrays() allocates device memory for POD arrays
//   - Caller must call free_scene_arrays() to deallocate GPU memory
// ============================================================================

#pragma once

#include "cuda_scene.h" // POD struct definitions: SpherePOD, QuadPOD, MaterialPOD, CameraPOD

// C-style interface for scene serialization (avoids C++ complexity in CUDA compiler)
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Serialize the Cornell box scene to GPU-friendly POD arrays.
 * 
 * This function:
 *   1. Builds the Cornell box scene using shared scene definition
 *   2. Configures camera with specified position (lookfrom = cam_x,y,z)
 *   3. Flattens all scene data (spheres, quads, materials, camera) to POD structs
 *   4. Allocates device memory and copies POD arrays to GPU
 * 
 * Camera configuration:
 *   - lookfrom: (cam_x, cam_y, cam_z) - user-specified position
 *   - lookat:   (278, 278, 278) - fixed at Cornell box center
 *   - vup:      (0, 1, 0) - fixed up direction
 *   - vfov:     40 degrees - fixed field of view
 * 
 * Memory: Allocates device memory for POD arrays. Caller must free via free_scene_arrays().
 * 
 * @param image_width           Image width in pixels
 * @param image_height          Image height in pixels
 * @param samples_per_pixel     Samples per pixel
 * @param max_depth             Maximum ray bounce depth
 * @param out_spheres           [OUT] Device pointer to sphere POD array
 * @param out_num_spheres       [OUT] Number of spheres in array
 * @param out_quads             [OUT] Device pointer to quad POD array
 * @param out_num_quads         [OUT] Number of quads in array
 * @param out_materials         [OUT] Device pointer to material POD array
 * @param out_num_materials     [OUT] Number of materials in array
 * @param out_camera            [OUT] Camera POD struct (host memory)
 * @param cam_x                 Camera position X coordinate
 * @param cam_y                 Camera position Y coordinate
 * @param cam_z                 Camera position Z coordinate
 */
void serialize_scene_arrays(
	int image_width, int image_height, int samples_per_pixel, int max_depth,
	SpherePOD** out_spheres, int* out_num_spheres,
	QuadPOD** out_quads, int* out_num_quads,
	MaterialPOD** out_materials, int* out_num_materials,
	CameraPOD* out_camera,
	double cam_x, double cam_y, double cam_z
);

/**
 * Free GPU device memory allocated by serialize_scene_arrays().
 * 
 * @param spheres    Device pointer to sphere array (from serialize_scene_arrays)
 * @param quads      Device pointer to quad array (from serialize_scene_arrays)
 * @param materials  Device pointer to material array (from serialize_scene_arrays)
 */
void free_scene_arrays(SpherePOD* spheres, QuadPOD* quads, MaterialPOD* materials);

#ifdef __cplusplus
}
#endif
