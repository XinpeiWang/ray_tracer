/// @file optix_renderer.h
/// @brief OptiX Renderer - Main host-side implementation
/// @details Handles OptiX context, pipeline, acceleration structure, and rendering.
///          This class manages GPU resources and orchestrates the path tracing pipeline.

#pragma once

#include "optix_types.h"
// For SceneData's instancing structs. No cycle: scene_builder.h pulls only
// optix_types.h and <vector>.
#include "scene_builder.h"
#include <optix_stubs.h>
#include <vector>
#include <string>
#include <cstddef>
#include <memory>

// Forward declaration — avoid including wavefront header in all TUs
namespace optix_renderer { class WavefrontPathTracer; }
namespace optix_renderer { class SPPMPathTracer; }

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
	/// @param lightKinds How to sample each entry of lightIndices (see GpuLightKind)
	/// @param punctualLights Vector of point/spot/distant delta lights (separate
	///        from the area-light arrays above; evaluated deterministically,
	///        not selected via the alias table)
	/// @param bilinearPatches Vector of bilinear patch geometry data (curved
	///        ruled surfaces, never used as lights - see optix_types.h's
	///        BilinearPatchData)
	/// @param triangles Vector of flat-shaded triangle geometry data (never
	///        used as lights - see optix_types.h's TriangleData)
	/// @param lensElements Host-precomputed RealisticCamera lens table (see
	///        optix_types.h's GpuLensElement); empty unless the scene uses
	///        CameraKind::Realistic.
	/// @param exitPupilBounds Host-precomputed RealisticCamera exit-pupil
	///        bounds table (see optix_types.h's GpuExitPupilBounds); empty
	///        unless the scene uses CameraKind::Realistic.
	/// @return true if scene was built successfully, false otherwise
	bool buildScene(
		const std::vector<SphereData>& spheres,
		const std::vector<QuadData>& quads,
		const std::vector<MaterialData>& materials,
		const std::vector<int>& lightIndices,
		const std::vector<GpuLightKind>& lightKinds,
		const std::vector<PunctualLightGPU>& punctualLights = {},
		const std::vector<BilinearPatchData>& bilinearPatches = {},
		const std::vector<TriangleData>& triangles = {},
		const std::vector<GpuLensElement>& lensElements = {},
		const std::vector<GpuExitPupilBounds>& exitPupilBounds = {},
		const std::vector<TextureData>& textures = {},
		const std::vector<unsigned char>& texturePixels = {},
		const std::vector<CloudMedium<float>>& cloudMediums = {},
		const std::vector<GpuRgbGridMedium>& rgbGridMediums = {},
		const std::vector<float>& rgbGridData = {}
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

	/// @brief Whether the OptiX device context was created with
	///        OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL (see createContext()'s
	///        own comment for what that buys and costs). Read once via the
	///        RAY_TRACER_OPTIX_VALIDATION env var at context-creation time -
	///        the validation mode is fixed for the context's whole lifetime,
	///        it cannot be toggled per-render the way wavefront mode can.
	bool validationEnabled() const { return validationEnabled_; }

	/// @brief Messages the context log callback received at level <= 3
	///        (fatal/error/warning; see OptixLogCallback's own doc comment
	///        for the level scale) since the last clearLoggedIssues() call.
	///        With validation mode on, this is where a bug like an
	///        out-of-bounds SBT index shows up - OptiX reports it here
	///        deterministically even on a run that would otherwise render a
	///        plausible-looking image and never crash.
	const std::vector<std::string>& loggedIssues() const { return loggedIssues_; }
	void clearLoggedIssues() { loggedIssues_.clear(); }

	// contextLogCallback() needs to reach loggedIssues_ from a free function
	// (it's the OptixLogCallback function-pointer type, not a member).
	friend void contextLogCallback(unsigned int, const char*, const char*, void*);

	/// @brief Sub-phase 1b verification only: runs ONE SPPM camera pass
	///        (visible-point recording + NEE) and writes the resulting Ld
	///        to outputFramebuffer. Proves the SPPMPathTracer module/
	///        program-group/pipeline/SBT machinery AND the camera-pass math
	///        (specular-chain resampling, NEE via the alias table, shadow
	///        rays) work against the real uploaded scene (buildScene() must
	///        already have been called) before the real multi-iteration
	///        loop exists. Not the final SPPM entry point -- see sub-phase
	///        1e's renderSPPM().
	/// @param ptxPath Optional explicit path to sppm_programs.ptx.
	bool renderSPPMTrivial(unsigned int width, unsigned int height,
	                        const GpuCameraParams& camera, float* outputFramebuffer,
	                        unsigned int maxDepth = 8, const std::string& ptxPath = "");

	/// @brief Sub-phase 1d: the real multi-iteration GPU SPPM render (camera
	///        pass -> hash grid -> photon pass -> radius contraction,
	///        repeated nIterations times, then final-image reconstruction).
	///        See SPPMPathTracer::render()'s own comment for the full loop.
	/// @param ptxPath Optional explicit path to sppm_programs.ptx.
	bool renderSPPM(unsigned int width, unsigned int height,
	                 int nIterations, int nPhotons, unsigned int maxDepth, float initialRadius,
	                 const GpuCameraParams& camera, float* outputFramebuffer,
	                 const std::string& ptxPath = "");

private:
	// Lazily creates + initializes sppmTracer_ (module/program groups/
	// pipeline/SBT), shared by renderSPPMTrivial() and renderSPPM() so the
	// init sequence only exists once.
	bool ensureSPPMTracer(const std::string& ptxPath);

	// -------------------------------------------------------------------
	// OptiX Core Resources
	// -------------------------------------------------------------------
	OptixDeviceContext context_ = nullptr;  ///< OptiX device context
	bool validationEnabled_ = false;        ///< See validationEnabled()
	std::vector<std::string> loggedIssues_; ///< See loggedIssues()
	CUcontext cudaContext_ = nullptr;       ///< CUDA primary context (see cuDevice_'s own comment)
	CUstream stream_ = nullptr;             ///< CUDA stream for async operations
	// The device cudaContext_ was retained against (createContext()'s own
	// cuDeviceGet() call, kDefaultCudaDevice). Stored so cleanup() can
	// release the primary context correctly: cuDevicePrimaryCtxRetain()'d
	// contexts must be given back via cuDevicePrimaryCtxRelease(device),
	// not cuCtxDestroy(context) -- the driver rejects the latter for a
	// primary-context handle (confirmed via compute-sanitizer:
	// CUDA_ERROR_INVALID_CONTEXT, "Cannot destroy primary context" -- see
	// cleanup()'s own comment for the fuller story).
	CUdevice cuDevice_ = 0;

	// -------------------------------------------------------------------
	// Wavefront path tracer (optional mode)
	// -------------------------------------------------------------------
	std::unique_ptr<optix_renderer::WavefrontPathTracer> wavefrontTracer_;
	bool useWavefront_ = false;  ///< If true, render() delegates to wavefrontTracer_

	// -------------------------------------------------------------------
	// SPPM path tracer (Phase 1 GPU port, see renderSPPMTrivial())
	// -------------------------------------------------------------------
	std::unique_ptr<optix_renderer::SPPMPathTracer> sppmTracer_;

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
	OptixProgramGroup hitgroupTrianglePG_ = nullptr;      ///< Triangle hit group (radiance)
	OptixProgramGroup shadowHitgroupSpherePG_ = nullptr; ///< Sphere shadow hit group
	OptixProgramGroup shadowHitgroupQuadPG_ = nullptr;   ///< Quad shadow hit group
	OptixProgramGroup shadowHitgroupBilinearPatchPG_ = nullptr; ///< Bilinear patch shadow hit group
	OptixProgramGroup shadowHitgroupTrianglePG_ = nullptr;      ///< Triangle shadow hit group

	// -------------------------------------------------------------------
	// Shader Binding Table (SBT)
	// -------------------------------------------------------------------
	OptixShaderBindingTable sbt_{};       ///< Shader binding table
	CUdeviceptr d_raygenRecord_ = 0;      ///< Device raygen record
	CUdeviceptr d_missRecord_ = 0;        ///< Device miss record  
	CUdeviceptr d_hitgroupRecords_ = 0;   ///< Device hit group records
	size_t numHitRecords_ = 0;            ///< Number of hit records

	// -------------------------------------------------------------------
	// Acceleration Structure
	// -------------------------------------------------------------------
	// OptiX forbids mixing OPTIX_BUILD_INPUT_TYPE_TRIANGLES and
	// OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES build inputs in one GAS, so
	// triangles (native OptiX geometry) and spheres/quads/bilinear-patches
	// (custom AABB primitives) live in two separate GASes, combined under
	// one top-level IAS - gasHandle_/d_gas_ is that IAS (single-level
	// instancing), the traversable actually used at render time.
	OptixTraversableHandle gasHandle_ = 0;       ///< Top-level IAS handle (used by params.traversable)
	CUdeviceptr d_gas_ = 0;                      ///< Device memory for the IAS
	OptixTraversableHandle gasCustomHandle_ = 0; ///< Child GAS: spheres/quads/bilinear-patches
	CUdeviceptr d_gasCustom_ = 0;                ///< Device memory for the custom-primitive GAS
	OptixTraversableHandle gasTriHandle_ = 0;    ///< Child GAS: triangles (native OptiX geometry)
	CUdeviceptr d_gasTri_ = 0;                   ///< Device memory for the triangle GAS
	bool sceneHasMotion_ = false;          ///< True if the uploaded scene has >=1 moving sphere (see buildScene())

	// -------------------------------------------------------------------
	// Scene Geometry and Materials (Device Memory)
	// -------------------------------------------------------------------
	CUdeviceptr d_materials_ = 0;     ///< Device material array
	unsigned int numMaterials_ = 0;   ///< Number of materials
	CUdeviceptr d_textures_ = 0;      ///< Device texture metadata array (TextureData)
	unsigned int numTextures_ = 0;    ///< Number of textures
	CUdeviceptr d_texturePixels_ = 0; ///< Device shared flat texture pixel buffer
	CUdeviceptr d_spheres_ = 0;       ///< Device sphere array
	unsigned int numSpheres_ = 0;     ///< Number of spheres
	CUdeviceptr d_quads_ = 0;         ///< Device quad array
	unsigned int numQuads_ = 0;       ///< Number of quads
	CUdeviceptr d_bilinearPatches_ = 0; ///< Device bilinear patch array
	unsigned int numBilinearPatches_ = 0; ///< Number of bilinear patches
	CUdeviceptr d_triangles_ = 0;      ///< Device triangle array
	unsigned int numTriangles_ = 0;    ///< Number of triangles

	// ---- object instancing -------------------------------------------------
	// Supplied by setInstanceData() rather than as buildScene() parameters:
	// threading three more arguments through would churn its declaration, its
	// definition, and both call sites in optix_interface.cpp, all to carry
	// data only one kind of scene uses. A scene that never calls it instances
	// nothing, which is what every built-in scene wants and is exactly what
	// it did before this existed.
	std::vector<TriangleData> instanceTriangles_;              ///< object space
	std::vector<SphereData> instanceSpheres_;                  ///< object space
	std::vector<SceneData::InstanceGroupGPU> instanceGroups_;
	std::vector<SceneData::InstancePlacementGPU> instancePlacements_;
	// A group holding both triangles and spheres needs one GAS of each, since
	// OptiX will not mix native triangles with custom AABB primitives - so
	// these are parallel arrays over groups, either of which may hold a null
	// handle for a group that has no geometry of that kind.
	std::vector<OptixTraversableHandle> gasGroupTriHandles_;
	std::vector<CUdeviceptr> d_gasGroupTri_;
	std::vector<OptixTraversableHandle> gasGroupSphereHandles_;
	std::vector<CUdeviceptr> d_gasGroupSphere_;
	CUdeviceptr d_instanceBase_ = 0;    ///< LaunchParams::instancePrimBase table
	unsigned int sceneTriangleCount_ = 0;  ///< triangles before the instanced ones
	unsigned int sceneSphereCount_ = 0;    ///< spheres before the instanced ones

  public:
	/// Geometry stored once and placed many times. Call before buildScene();
	/// passing empty vectors (or not calling it) disables instancing for the
	/// next scene built.
	void setInstanceData(const std::vector<TriangleData>& triangles,
						 const std::vector<SphereData>& spheres,
						 const std::vector<SceneData::InstanceGroupGPU>& groups,
						 const std::vector<SceneData::InstancePlacementGPU>& placements) {
		instanceTriangles_ = triangles;
		instanceSpheres_ = spheres;
		instanceGroups_ = groups;
		instancePlacements_ = placements;
	}

  private:
	CUdeviceptr d_lensElements_ = 0;      ///< Device RealisticCamera lens table
	unsigned int numLensElements_ = 0;    ///< Number of lens elements
	CUdeviceptr d_exitPupilBounds_ = 0;    ///< Device RealisticCamera exit-pupil bounds table
	unsigned int numExitPupilBounds_ = 0;  ///< Number of exit-pupil bounds slabs
	CUdeviceptr d_cloudMediums_ = 0;       ///< Device CloudMedium<float> table (MaterialType::CloudMedium)
	unsigned int numCloudMediums_ = 0;     ///< Number of cloud media
	CUdeviceptr d_rgbGridMediums_ = 0;     ///< Device GpuRgbGridMedium table (MaterialType::RgbGridMedium)
	unsigned int numRgbGridMediums_ = 0;   ///< Number of RGB grid media
	CUdeviceptr d_rgbGridData_ = 0;        ///< Device flat voxel data for all RGB grid media
	unsigned int rgbGridDataCount_ = 0;    ///< Number of floats in d_rgbGridData_

	// Light sampling support for MIS
	CUdeviceptr d_lightIndices_ = 0;  ///< Device light primitive indices
	CUdeviceptr d_lightKinds_ = 0; ///< Device GpuLightKind array, one per light
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
	/// @param haveInstancedTriangles,haveInstancedSpheres Append a dedicated
	///        hit-record pair for instanced geometry of that type. Instanced
	///        GASes cannot reuse the scene's own records positionally - a
	///        child GAS's records are addressed as sbtOffset + build-input
	///        index, and the scene's custom-primitive region is packed with no
	///        gaps for absent types - so they get their own pair at the end,
	///        which buildScene() points their instances at.
	bool buildSBT(
		const std::vector<SphereData>& spheres,
		const std::vector<QuadData>& quads,
		const std::vector<BilinearPatchData>& bilinearPatches,
		const std::vector<TriangleData>& triangles,
		bool haveInstancedTriangles,
		bool haveInstancedSpheres
	);

	/// @brief Release all GPU resources
	void cleanup() noexcept;

	/// @brief Load PTX shader code from file
	/// @param filename Path to PTX file
	/// @return PTX source code as string
	std::string loadPTX(const char* filename) const;
};
