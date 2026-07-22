#pragma once

#include "path_tracing_strategy.h"
#include "optix_types.h"
#include <optix.h>
#include <cuda_runtime.h>

namespace optix_renderer {

/**
 * @brief Recursive path tracing strategy (current implementation)
 * 
 * This strategy uses the traditional recursive ray tracing approach where
 * the raygen program traces complete paths recursively, calling closesthit
 * programs which may spawn new rays (scatter + shadow rays).
 * 
 * This is the current working implementation, wrapped in the strategy interface.
 */
class RecursivePathTracer : public PathTracingStrategy {
public:
	RecursivePathTracer();
	~RecursivePathTracer() override;

	// PathTracingStrategy interface implementation
	bool initialize(
		OptixDeviceContext context,
		OptixModule module,
		CUstream stream
	) override;

	bool createProgramGroups() override;
	bool linkPipeline(unsigned int maxTraceDepth) override;
	bool buildSBT(unsigned int numSpheres, unsigned int numQuads) override;

	bool render(
		int width,
		int height,
		int samples_per_pixel,
		int max_depth,
		const float* camera_origin,
		const float* camera_lower_left,
		const float* camera_horizontal,
		const float* camera_vertical,
		float* framebuffer,
		OptixTraversableHandle gas_handle,
		CUdeviceptr d_materials,
		CUdeviceptr d_spheres,
		CUdeviceptr d_quads,
		CUdeviceptr d_light_indices,
		CUdeviceptr d_is_light_sphere,
		unsigned int num_materials,
		unsigned int num_spheres,
		unsigned int num_quads,
		unsigned int num_lights
	) override;

	void cleanup() override;

	PathTracingMode getMode() const override { return PathTracingMode::RECURSIVE; }
	const char* getName() const override { return "RecursivePathTracer"; }

private:
	// Program groups specific to recursive tracing
	OptixProgramGroup raygenPG_ = nullptr;
	OptixProgramGroup missPG_ = nullptr;
	OptixProgramGroup shadowMissPG_ = nullptr;
	OptixProgramGroup hitgroupSpherePG_ = nullptr;
	OptixProgramGroup hitgroupQuadPG_ = nullptr;
	OptixProgramGroup shadowHitgroupSpherePG_ = nullptr;
	OptixProgramGroup shadowHitgroupQuadPG_ = nullptr;

	// Pipeline compile options
	OptixPipelineCompileOptions pipelineCompileOptions_ = {};

	// SBT records (device memory)
	CUdeviceptr d_raygenRecord_ = 0;
	CUdeviceptr d_missRecord_ = 0;
	CUdeviceptr d_hitgroupRecords_ = 0;
	size_t numHitRecords_ = 0;

	// Device launch parameters
	CUdeviceptr d_launchParams_ = 0;

	// Helper: destroy program groups
	void destroyProgramGroups();

	// Helper: destroy SBT
	void destroySBT();
};

} // namespace optix_renderer
