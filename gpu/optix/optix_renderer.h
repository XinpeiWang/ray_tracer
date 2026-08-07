/// @file optix_renderer.h
/// @brief OptiX Renderer - Main host-side implementation
/// @details Handles OptiX context, pipeline, acceleration structure, and rendering.
///          This class manages GPU resources and orchestrates the path tracing pipeline.

#pragma once

#include "optix_types.h"
#include <optix_stubs.h>
#include <vector>
#include <string>
#include <cstddef>
#include <memory>

// Forward declaration — avoid including wavefront header in all TUs
namespace optix_renderer { class WavefrontPathTracer; }

/// @class OptiXRenderer
/// @brief Main OptiX path tracer implementation.
/// @details Manages OptiX resources including context, pipeline, acceleration structures,
///          and rendering state. Uses RAII for resource management.
class OptiXRenderer {
public:
	/// @brief Construct a new OptiXRenderer
	OptiXRenderer();

	/// @brief Destroy the OptiXRenderer and release all GPU resources
	~OptiXRenderer();

	// Disable copy (GPU resources cannot be copied)
	OptiXRenderer(const OptiXRenderer&) = delete;
	OptiXRenderer& operator=(const OptiXRenderer&) = delete;

	// Allow move (for transferring ownership)
	OptiXRenderer(OptiXRenderer&&) noexcept = default;
	OptiXRenderer& operator=(OptiXRenderer&&) noexcept = default;

	/// @brief Initialize OptiX context and rendering pipeline
	/// @return true if initialization succeeded, false otherwise
	bool initialize();

	/// @brief Build scene acceleration structure from geometry
	/// @param spheres Vector of sphere geometry data
	/// @param quads Vector of quad geometry data
	/// @param materials Vector of material data
	/// @param lightIndices Vector of light primitive indices
	/// @param isLightSphere Vector of flags (true=sphere, false=quad)
	/// @param punctualLights Vector of point/spot/distant delta lights (separate
	///        from the area-light arrays above; evaluated deterministically,
	///        not selected via the alias table)
	/// @param bilinearPatches Vector of bilinear patch geometry data (curved
	///        ruled surfaces, never used as lights - see optix_types.h's
	///        BilinearPatchData)
	/// @return true if scene was built successfully, false otherwise
	bool buildScene(
		const std::vector<SphereData>& spheres,
		const std::vector<QuadData>& quads,
		const std::vector<MaterialData>& materials,
		const std::vector<int>& lightIndices,
		const std::vector<bool>& isLightSphere,
		const std::vector<PunctualLightGPU>& punctualLights = {},
		const std::vector<BilinearPatchData>& bilinearPatches = {}
	);

	/// @brief Render a frame using path tracing
	/// @param width Image width in pixels
	/// @param height Image height in pixels
	/// @param samplesPerPixel Number of samples per pixel for anti-aliasing
	/// @param maxDepth Maximum ray bounce depth
	/// @param camera Camera parameters (viewport + optional DOF/orthographic/spherical fields)
	/// @param outputFramebuffer Output RGB framebuffer (float[width * height * 3])
	/// @return true if rendering succeeded, false otherwise
	bool render(
		unsigned int width,
		unsigned int height,
		unsigned int samplesPerPixel,
		unsigned int maxDepth,
		const GpuCameraParams& camera,
		float* outputFramebuffer
	);

	/// @brief Check if OptiX is available on this system
	/// @details Verifies driver support and SDK availability
	/// @return true if OptiX can be used, false otherwise
	static bool isAvailable() noexcept;

	/// @brief Enable or disable wavefront GPU path tracing mode.
	/// @details When enabled, render() uses the WavefrontPathTracer (queue-based,
	///          pbrt-v4-style) instead of the default recursive path tracer.
	///          Call before render(); safe to toggle between frames.
	/// @param enable true = wavefront mode, false = recursive mode (default)
	/// @param ptxPath Optional explicit path to wavefront_programs.ptx.
	///                If empty, looks in the same directory as optix_programs.ptx.
	void enableWavefront(bool enable, const std::string& ptxPath = "");

private:
	// -------------------------------------------------------------------
	// OptiX Core Resources
	// -------------------------------------------------------------------
	OptixDeviceContext context_ = nullptr;  ///< OptiX device context
	CUcontext cudaContext_ = nullptr;       ///< CUDA context
	CUstream stream_ = nullptr;             ///< CUDA stream for async operations

	// -------------------------------------------------------------------
	// Wavefront path tracer (optional mode)
	// -------------------------------------------------------------------
	std::unique_ptr<optix_renderer::WavefrontPathTracer> wavefrontTracer_;
	bool useWavefront_ = false;  ///< If true, render() delegates to wavefrontTracer_

	// -------------------------------------------------------------------
	// Pipeline and Shaders
	// -------------------------------------------------------------------
	OptixPipeline pipeline_ = nullptr;                      ///< Compiled pipeline
	OptixModule module_ = nullptr;                          ///< PTX module
	OptixPipelineCompileOptions pipelineCompileOptions_{};  ///< Pipeline compilation options

	// Program groups
	OptixProgramGroup raygenPG_ = nullptr;        ///< Ray generation program
	OptixProgramGroup missPG_ = nullptr;          ///< Miss program (radiance)
	OptixProgramGroup shadowMissPG_ = nullptr;    ///< Shadow miss program
	OptixProgramGroup hitgroupSpherePG_ = nullptr;///< Sphere hit group (radiance)
	OptixProgramGroup hitgroupQuadPG_ = nullptr;  ///< Quad hit group (radiance)
	OptixProgramGroup hitgroupBilinearPatchPG_ = nullptr; ///< Bilinear patch hit group (radiance)
	OptixProgramGroup shadowHitgroupSpherePG_ = nullptr; ///< Sphere shadow hit group
	OptixProgramGroup shadowHitgroupQuadPG_ = nullptr;   ///< Quad shadow hit group
	OptixProgramGroup shadowHitgroupBilinearPatchPG_ = nullptr; ///< Bilinear patch shadow hit group

	// -------------------------------------------------------------------
	// Shader Binding Table (SBT)
	// -------------------------------------------------------------------
	OptixShaderBindingTable sbt_{};       ///< Shader binding table
	CUdeviceptr d_raygenRecord_ = 0;      ///< Device raygen record
	CUdeviceptr d_missRecord_ = 0;        ///< Device miss record  
	CUdeviceptr d_hitgroupRecords_ = 0;   ///< Device hit group records
	size_t numHitRecords_ = 0;            ///< Number of hit records

	// -------------------------------------------------------------------
	// Acceleration Structure (GAS)
	// -------------------------------------------------------------------
	OptixTraversableHandle gasHandle_ = 0; ///< Acceleration structure handle
	CUdeviceptr d_gas_ = 0;                ///< Device memory for GAS

	// -------------------------------------------------------------------
	// Scene Geometry and Materials (Device Memory)
	// -------------------------------------------------------------------
	CUdeviceptr d_materials_ = 0;     ///< Device material array
	unsigned int numMaterials_ = 0;   ///< Number of materials
	CUdeviceptr d_spheres_ = 0;       ///< Device sphere array
	unsigned int numSpheres_ = 0;     ///< Number of spheres
	CUdeviceptr d_quads_ = 0;         ///< Device quad array
	unsigned int numQuads_ = 0;       ///< Number of quads
	CUdeviceptr d_bilinearPatches_ = 0; ///< Device bilinear patch array
	unsigned int numBilinearPatches_ = 0; ///< Number of bilinear patches

	// Light sampling support for MIS
	CUdeviceptr d_lightIndices_ = 0;  ///< Device light primitive indices
	CUdeviceptr d_isLightSphere_ = 0; ///< Device light type flags (sphere/quad)
	CUdeviceptr d_aliasTable_ = 0;    ///< Device alias table for power-weighted light sampling
	unsigned int numLights_ = 0;      ///< Number of emissive lights

	// Punctual (delta) lights: point/spot/distant. Separate from the area
	// lights above - evaluated deterministically, not via the alias table.
	CUdeviceptr d_punctualLights_ = 0;
	unsigned int numPunctualLights_ = 0;

	// -------------------------------------------------------------------
	// Launch Parameters
	// -------------------------------------------------------------------
	CUdeviceptr d_launchParams_ = 0;  ///< Device launch parameters

	// -------------------------------------------------------------------
	// Private Helper Methods
	// -------------------------------------------------------------------

	/// @brief Create OptiX device context and CUDA resources
	bool createContext();

	/// @brief Load PTX and create OptiX module
	bool createModule();

	/// @brief Create program groups for raygen, miss, and hit programs
	bool createProgramGroups();

	/// @brief Link program groups into pipeline
	bool linkPipeline();

	/// @brief Build Shader Binding Table from geometry
	bool buildSBT(
		const std::vector<SphereData>& spheres,
		const std::vector<QuadData>& quads,
		const std::vector<BilinearPatchData>& bilinearPatches
	);

	/// @brief Release all GPU resources
	void cleanup() noexcept;

	/// @brief Load PTX shader code from file
	/// @param filename Path to PTX file
	/// @return PTX source code as string
	std::string loadPTX(const char* filename) const;
};
