#pragma once

#include "optix_types.h"
#include <optix.h>
#include <vector>
#include <memory>

namespace optix_renderer {

/**
 * @brief Rendering mode selection
 */
enum class PathTracingMode {
	RECURSIVE,   // Traditional recursive path tracing (current implementation)
	WAVEFRONT    // Queue-based wavefront path tracing (future)
};

/**
 * @brief Abstract strategy interface for path tracing implementations
 * 
 * This interface allows different path tracing algorithms to be plugged into
 * the OptixRenderer without changing the infrastructure code. Implementations
 * can use recursive tracing, wavefront/queue-based tracing, or other approaches.
 * 
 * The strategy is responsible for:
 * - Creating OptiX program groups (raygen, closesthit, etc.)
 * - Building the shader binding table (SBT)
 * - Executing the path tracing algorithm
 * 
 * Scene data (GAS, materials, lights) remains managed by OptixRenderer.
 */
class PathTracingStrategy {
public:
	virtual ~PathTracingStrategy() = default;

	/**
	 * @brief Initialize the path tracing strategy
	 * 
	 * @param context OptiX context (already initialized by OptixRenderer)
	 * @param module OptiX module containing device programs (PTX loaded)
	 * @param stream CUDA stream for asynchronous operations
	 * @return true if initialization succeeded, false otherwise
	 */
	virtual bool initialize(
		OptixDeviceContext context,
		OptixModule module,
		CUstream stream
	) = 0;

	/**
	 * @brief Create OptiX program groups for this strategy
	 * 
	 * Different strategies may need different program groups:
	 * - Recursive: raygen, closesthit (sphere/quad), miss, shadow programs
	 * - Wavefront: simplified programs + CUDA kernels
	 * 
	 * @return true if program groups created successfully
	 */
	virtual bool createProgramGroups() = 0;

	/**
	 * @brief Link the OptiX pipeline for this strategy
	 * 
	 * @param maxTraceDepth Maximum ray recursion depth
	 * @return true if pipeline linked successfully
	 */
	virtual bool linkPipeline(unsigned int maxTraceDepth) = 0;

	/**
	 * @brief Build the Shader Binding Table (SBT)
	 *
	 * @param numSpheres Number of sphere primitives in scene
	 * @param numQuads Number of quad primitives in scene
	 * @param numBilinearPatches Number of bilinear patch primitives in scene
	 * @param numTriangles Number of triangle primitives in scene
	 * @return true if SBT built successfully
	 */
	virtual bool buildSBT(unsigned int numSpheres, unsigned int numQuads, unsigned int numBilinearPatches = 0, unsigned int numTriangles = 0) = 0;

	/**
	 * @brief Execute path tracing for the given parameters
	 * 
	 * @param width Image width in pixels
	 * @param height Image height in pixels
	 * @param samples_per_pixel Number of samples per pixel
	 * @param max_depth Maximum path depth (bounces)
	 * @param camera_origin Camera origin (world space)
	 * @param camera_lower_left Lower-left corner of viewport
	 * @param camera_horizontal Horizontal viewport vector
	 * @param camera_vertical Vertical viewport vector
	 * @param framebuffer Output framebuffer (device pointer, width*height*3 floats)
	 * @param gas_handle Geometry acceleration structure handle
	 * @param d_materials Device pointer to material array
	 * @param d_spheres Device pointer to sphere array
	 * @param d_quads Device pointer to quad array
	 * @param d_light_indices Device pointer to light index array
	 * @param d_lightKinds Device GpuLightKind array, one per light
	 * @param d_alias_table Device pointer to power-weighted alias table
	 * @param num_materials Number of materials
	 * @param num_spheres Number of spheres
	 * @param num_quads Number of quads
	 * @param num_lights Number of lights
	 * @param d_punctual_lights Device pointer to PunctualLightGPU array (point/spot/distant delta lights)
	 * @param num_punctual_lights Number of punctual lights
	 * @param d_bilinear_patches Device pointer to BilinearPatchData array (curved ruled surfaces, never lights)
	 * @param num_bilinear_patches Number of bilinear patches
	 * @param d_triangles Device pointer to TriangleData array (flat-shaded, never lights)
	 * @param num_triangles Number of triangles
	 * @return true if rendering succeeded
	 */
	virtual bool render(
		int width,
		int height,
		int samples_per_pixel,
		int max_depth,
		const GpuCameraParams& camera,
		float* framebuffer,
		OptixTraversableHandle gas_handle,
		CUdeviceptr d_materials,
		CUdeviceptr d_spheres,
		CUdeviceptr d_quads,
		CUdeviceptr d_light_indices,
		CUdeviceptr d_lightKinds,
		CUdeviceptr d_alias_table,
		unsigned int num_materials,
		unsigned int num_spheres,
		unsigned int num_quads,
		unsigned int num_lights,
		CUdeviceptr d_punctual_lights = 0,
		unsigned int num_punctual_lights = 0,
		CUdeviceptr d_bilinear_patches = 0,
		unsigned int num_bilinear_patches = 0,
		CUdeviceptr d_triangles = 0,
		unsigned int num_triangles = 0
	) = 0;

	/**
	 * @brief Clean up strategy resources
	 * 
	 * Called before destruction or when switching strategies.
	 */
	virtual void cleanup() = 0;

	/**
	 * @brief Get the rendering mode of this strategy
	 */
	virtual PathTracingMode getMode() const = 0;

	/**
	 * @brief Get a human-readable name for this strategy
	 */
	virtual const char* getName() const = 0;

protected:
	// Shared context references (not owned by strategy)
	OptixDeviceContext context_ = nullptr;
	OptixModule module_ = nullptr;
	CUstream stream_ = 0;

	// Strategy-specific pipeline
	OptixPipeline pipeline_ = nullptr;
	OptixShaderBindingTable sbt_ = {};
};

/**
 * @brief Factory function to create path tracing strategy
 * 
 * @param mode Desired rendering mode
 * @param enableFallback If true, fall back to RECURSIVE on failure
 * @return Unique pointer to strategy, or nullptr if creation failed
 */
std::unique_ptr<PathTracingStrategy> createPathTracingStrategy(
	PathTracingMode mode,
	bool enableFallback = true
);

/**
 * @brief Convert mode enum to string
 */
const char* pathTracingModeToString(PathTracingMode mode);

} // namespace optix_renderer
