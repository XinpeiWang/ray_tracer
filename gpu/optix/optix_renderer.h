// OptiX Renderer - Main host-side implementation
// Handles OptiX context, pipeline, acceleration structure, and rendering

#pragma once

#include "optix_types.h"
#include <optix_stubs.h>
#include <vector>
#include <string>

class OptiXRenderer {
public:
	OptiXRenderer();
	~OptiXRenderer();

	// Initialize OptiX context and pipeline
	bool initialize();

	// Build scene acceleration structure from geometry
	bool buildScene(
		const std::vector<SphereData>& spheres,
		const std::vector<QuadData>& quads,
		const std::vector<MaterialData>& materials
	);

	// Render a frame
	bool render(
		unsigned int width,
		unsigned int height,
		unsigned int samplesPerPixel,
		unsigned int maxDepth,
		const float* cameraOrigin,      // float[3]
		const float* cameraLowerLeft,   // float[3]
		const float* cameraHorizontal,  // float[3]
		const float* cameraVertical,    // float[3]
		float* outputFramebuffer        // float[width * height * 3]
	);

	// Check if OptiX is available (driver, SDK)
	static bool isAvailable();

private:
	// OptiX context
	OptixDeviceContext context_ = nullptr;
	CUcontext cudaContext_ = nullptr;
	CUstream stream_ = nullptr;

	// Pipeline
	OptixPipeline pipeline_ = nullptr;
	OptixModule module_ = nullptr;
	OptixPipelineCompileOptions pipelineCompileOptions_ = {};

	// Programs
	OptixProgramGroup raygenPG_ = nullptr;
	OptixProgramGroup missPG_ = nullptr;
	OptixProgramGroup hitgroupSpherePG_ = nullptr;
	OptixProgramGroup hitgroupQuadPG_ = nullptr;

	// Shader Binding Table
	OptixShaderBindingTable sbt_ = {};
	CUdeviceptr d_raygenRecord_ = 0;
	CUdeviceptr d_missRecord_ = 0;
	CUdeviceptr d_hitgroupRecords_ = 0;
	size_t numHitRecords_ = 0;

	// Acceleration structure
	OptixTraversableHandle gasHandle_ = 0;
	CUdeviceptr d_gas_ = 0;

	// Scene data on device
	CUdeviceptr d_materials_ = 0;
	unsigned int numMaterials_ = 0;
	CUdeviceptr d_spheres_ = 0;
	unsigned int numSpheres_ = 0;
	CUdeviceptr d_quads_ = 0;
	unsigned int numQuads_ = 0;

	// Launch params
	CUdeviceptr d_launchParams_ = 0;

	// Helper methods
	bool createContext();
	bool createModule();
	bool createProgramGroups();
	bool linkPipeline();
	bool buildSBT(const std::vector<SphereData>& spheres, const std::vector<QuadData>& quads);
	void cleanup();

	// PTX loading
	std::string loadPTX(const char* filename);
};
