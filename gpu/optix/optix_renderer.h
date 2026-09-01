/// @file optix_renderer.h
/// @brief OptiX Renderer - Main host-side implementation
/// @details Handles OptiX context, pipeline, acceleration structure, and rendering.
///          This class manages GPU resources and orchestrates the path tracing pipeline.

#pragma once

#include "optix_types.h"
// For SceneData's instancing structs. No cycle: scene_builder.h pulls only
// optix_types.h and <vector>.
#include "scene_builder.h"
// For OptixDiagnostics (getDiagnostics()'s out-param) - a plain-POD extern
// "C" struct with no CUDA/OptiX types in it, safe to pull in here; no cycle
// (optix_interface.h doesn't include this header).
#include "optix_interface.h"
#include "optix_denoiser.h"  // DenoiserResources - shared with WavefrontPathTracer
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
		// Disk/Cylinder - supported on both the recursive (Phase 4b) and
		// wavefront (Phase 4c) backends; see DiskData/CylinderData's own
		// comment in optix_types.h and gasDiskCylinderHandle_'s comment
		// above for why these get their own child GAS rather than joining
		// spheres/quads/bilinearPatches in the shared one.
		const std::vector<DiskData>& disks = {},
		const std::vector<CylinderData>& cylinders = {},
		const std::vector<GpuLensElement>& lensElements = {},
		const std::vector<GpuExitPupilBounds>& exitPupilBounds = {},
		const std::vector<TextureData>& textures = {},
		const std::vector<unsigned char>& texturePixels = {},
		const std::vector<CloudMedium<float>>& cloudMediums = {},
		const std::vector<GpuRgbGridMedium>& rgbGridMediums = {},
		const std::vector<float>& rgbGridData = {},
		const std::vector<GpuGridMedium>& gridMediums = {},
		const std::vector<float>& gridData = {},
		const std::vector<GpuBssrdfTable>& bssrdfTables = {},
		const std::vector<float>& bssrdfRhoSamples = {},
		const std::vector<float>& bssrdfRadiusSamples = {},
		const std::vector<float>& bssrdfProfile = {},
		const std::vector<float>& bssrdfProfileCdf = {},
		const std::vector<GpuMeasuredTable>& measuredTables = {},
		const std::vector<float>& measuredParamValues = {},
		const std::vector<float>& measuredData = {},
		const std::vector<float>& measuredMcdf = {},
		const std::vector<float>& measuredCcdf = {},
		// Real importance-sampled HDR sky distribution (LightSource
		// "infinite" with an image - see optix_types.h's GpuSkyDistribution
		// comment and SceneData's own matching fields in scene_builder.h).
		// skyHeight stays 0 (its default) for every scene with a constant-
		// colour sky or no infinite light at all.
		const std::vector<float>& skyImagePixels = {},
		const std::vector<float>& skyMarginalCdf = {},
		const std::vector<float>& skyMarginalFunc = {},
		float skyMarginalFuncInt = 0.0f,
		const std::vector<float>& skyConditionalCdf = {},
		const std::vector<float>& skyConditionalFunc = {},
		const std::vector<float>& skyConditionalFuncInt = {},
		int skyWidth = 0, int skyHeight = 0, float skyScale = 1.0f,
		// pbrt-v4 "portal" (windowed) infinite light - see GpuPortalLight's
		// own comment (optix_types.h) and SceneData's own matching fields
		// (scene_builder.h). portalHeight stays 0 (its default) for every
		// scene without a real portal[4] quad - mutually exclusive with the
		// sky fields just above (matches CPU).
		const std::vector<float>& portalRectifiedImage = {},
		const std::vector<float>& portalDistFunc = {},
		const std::vector<double>& portalSatSum = {},
		int portalWidth = 0, int portalHeight = 0, float portalScale = 1.0f,
		float3 portalFrameX = {}, float3 portalFrameY = {}, float3 portalFrameZ = {},
		float3 portalP0 = {}, float3 portalP2 = {}
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

	/// @brief Full GPU/CUDA/OptiX capability probe for --diagnose.
	/// @details Same probe sequence as isAvailable(), but populates `out`
	/// with device name/versions/VRAM instead of discarding them, and
	/// records which step failed in out.failure_reason instead of only
	/// logging to stderr. See optix_interface.h's OptixDiagnostics comment.
	/// @return out.available (true if the full probe succeeded)
	static bool getDiagnostics(OptixDiagnostics& out) noexcept;

	/// @brief Enable or disable wavefront GPU path tracing mode.
	/// @details When enabled, render() uses the WavefrontPathTracer (queue-based,
	///          pbrt-v4-style) instead of the default recursive path tracer.
	///          Call before render(); safe to toggle between frames.
	/// @param enable true = wavefront mode, false = recursive mode (default)
	/// @param ptxPath Optional explicit path to wavefront_programs.ptx.
	///                If empty, looks in the same directory as optix_programs.ptx.
	void enableWavefront(bool enable, const std::string& ptxPath = "");

	/// @brief Enable or disable the OptiX AI (post-process) denoiser.
	/// @details When enabled, render() runs the built-in OptiX denoiser model
	///          on the accumulated framebuffer, on-device, right after the
	///          main path-tracing launch completes and before the result is
	///          copied back to host memory. Recursive backend only (see
	///          render()'s own call site) - wavefront mode ignores this flag,
	///          same scope-reduction pattern as enableWavefront() being
	///          recursive-only in reverse. Guided by albedo + world-space
	///          normal AOV buffers: render() allocates d_albedo/d_normal (via
	///          ensureAovBuffers(), persisted across calls like the denoiser's
	///          own state) and every closest-hit/miss program packs them into
	///          6 extra payload registers (p16-p21, see optix_types.h's
	///          PathTracingPayload::albedo/normal comment), accumulated at
	///          depth==0 in raygen with zero atomics needed (same single-
	///          thread-per-pixel pattern the existing pixel_color
	///          accumulation already uses). Both the packing
	///          (pack_aov_payload(), optix_device_helpers.h) and the
	///          accumulation are gated on params.albedoBuffer being non-null,
	///          so this costs nothing extra when denoising is off.
	///          denoise() itself derives OptixDenoiserOptions::guideAlbedo/
	///          guideNormal from whether its own d_albedo/d_normal arguments
	///          are non-null (not a hardcoded 1) - render() is the only
	///          caller and always passes both together or neither, but that
	///          pairing is this caller's contract, not something denoise()
	///          enforces on its own.
	/// @param enable true = denoise every render() call, false = off (default)
	void enableDenoise(bool enable) { denoiseEnabled_ = enable; }

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

	/// @brief Read back the persisted albedo/normal AOV buffers (see their
	///        own comment, in the private section below) to host memory -
	///        used for EXR AOV export (see optix_render_main()'s own AOV-
	///        export branch), a second consumer of the exact same buffers
	///        the denoiser guide layer already fills. Only valid after
	///        ensureAovBuffers() has run at this resolution (currently only
	///        denoise()/render() with denoising enabled does so - see that
	///        method's own comment); returns false (leaving the output
	///        vectors untouched) rather than reading a null or differently-
	///        sized buffer if that hasn't happened.
	/// @param albedoOut,normalOut Resized to width*height*3 floats
	///        (interleaved RGB, row-major - same layout write_exr_image()
	///        expects) on success.
	bool readAovBuffers(unsigned int width, unsigned int height,
		std::vector<float>& albedoOut, std::vector<float>& normalOut) const;

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
	// OptiX AI Denoiser (optional post-process, recursive backend)
	// -------------------------------------------------------------------
	bool denoiseEnabled_ = false;  ///< See enableDenoise()
	// Persisted across render() calls rather than created/destroyed fresh
	// each time - see denoise()'s own comment. Shared with
	// WavefrontPathTracer's identical member (optix_denoiser.h) - each
	// backend owns its own instance, keyed to its own context_/stream_.
	DenoiserResources denoiserResources_;

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
	OptixProgramGroup hitgroupDiskPG_ = nullptr;          ///< Disk hit group (radiance)
	OptixProgramGroup hitgroupCylinderPG_ = nullptr;      ///< Cylinder hit group (radiance)
	OptixProgramGroup hitgroupTrianglePG_ = nullptr;      ///< Triangle hit group (radiance)
	OptixProgramGroup shadowHitgroupSpherePG_ = nullptr; ///< Sphere shadow hit group
	OptixProgramGroup shadowHitgroupQuadPG_ = nullptr;   ///< Quad shadow hit group
	OptixProgramGroup shadowHitgroupBilinearPatchPG_ = nullptr; ///< Bilinear patch shadow hit group
	OptixProgramGroup shadowHitgroupDiskPG_ = nullptr;          ///< Disk shadow hit group
	OptixProgramGroup shadowHitgroupCylinderPG_ = nullptr;      ///< Cylinder shadow hit group
	OptixProgramGroup shadowHitgroupTrianglePG_ = nullptr;      ///< Triangle shadow hit group

	// RAY_TYPE_PROBE program groups (recursive backend only, Phase 1 BSSRDF
	// - see optix_types.h's RAY_TYPE_PROBE comment and optix_probe_hit.h).
	OptixProgramGroup probeMissPG_ = nullptr;                    ///< Probe miss program (no-op)
	OptixProgramGroup probeHitgroupSpherePG_ = nullptr;          ///< Sphere probe hit group
	OptixProgramGroup probeHitgroupQuadPG_ = nullptr;            ///< Quad probe hit group
	OptixProgramGroup probeHitgroupBilinearPatchPG_ = nullptr;   ///< Bilinear patch probe hit group
	OptixProgramGroup probeHitgroupDiskPG_ = nullptr;            ///< Disk probe hit group
	OptixProgramGroup probeHitgroupCylinderPG_ = nullptr;        ///< Cylinder probe hit group
	OptixProgramGroup probeHitgroupTrianglePG_ = nullptr;        ///< Triangle probe hit group

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
	// Disk/Cylinder get their OWN child GAS + IAS instance, appended AFTER
	// every other instance (see buildScene()'s own comment at the disk/
	// cylinder instance site) - deliberately NOT folded into gasCustomHandle_
	// alongside sphere/quad/bilinear-patch, so their presence can never shift
	// those types' build_input_index/SBT offsets. That matters because the
	// wavefront backend (Phase 4c) traces against this SAME shared
	// traversable with its OWN, separately-built SBT - both backends'
	// buildSBT() need every other type's baked instance.sbtOffset to stay
	// exactly where it was, whether or not the scene has any disks/cylinders
	// at all. It also matters for disk/cylinder's OWN offset: wavefront's
	// buildSBT() pads every type-group to RAY_TYPE_COUNT identical records
	// (see that function's pushTriple comment) specifically so its own
	// cumulative record count matches this backend's stride-RAY_TYPE_COUNT
	// baked offsets exactly, for any number of preceding type-groups - had
	// disk/cylinder shared gasCustomHandle_ instead of appending a trailing
	// instance, their build_input_index would depend on which OTHER custom
	// types the scene has, defeating that padding's whole purpose.
	OptixTraversableHandle gasDiskCylinderHandle_ = 0;
	CUdeviceptr d_gasDiskCylinder_ = 0;
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
	CUdeviceptr d_disks_ = 0;         ///< Device disk array
	unsigned int numDisks_ = 0;       ///< Number of disks
	CUdeviceptr d_cylinders_ = 0;     ///< Device cylinder array
	unsigned int numCylinders_ = 0;   ///< Number of cylinders
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
	CUdeviceptr d_gridMediums_ = 0;        ///< Device GpuGridMedium table (MaterialType::GridMedium)
	unsigned int numGridMediums_ = 0;      ///< Number of single-channel grid media
	CUdeviceptr d_gridData_ = 0;           ///< Device flat voxel data for all single-channel grid media
	unsigned int gridDataCount_ = 0;       ///< Number of floats in d_gridData_

	// Tabulated BSSRDF tables (MaterialType::Subsurface, recursive backend
	// only, Phase 1 - see optix_types.h's GpuBssrdfTable comment).
	CUdeviceptr d_bssrdfTables_ = 0;
	unsigned int numBssrdfTables_ = 0;
	CUdeviceptr d_bssrdfRhoSamples_ = 0;
	CUdeviceptr d_bssrdfRadiusSamples_ = 0;
	CUdeviceptr d_bssrdfProfile_ = 0;
	CUdeviceptr d_bssrdfProfileCdf_ = 0;

	// Real tabulated measured-BRDF tables (MaterialType::Measured, both GPU
	// backends - see optix_types.h's GpuMeasuredTable comment).
	CUdeviceptr d_measuredTables_ = 0;
	unsigned int numMeasuredTables_ = 0;
	CUdeviceptr d_measuredParamValues_ = 0;
	CUdeviceptr d_measuredData_ = 0;
	CUdeviceptr d_measuredMcdf_ = 0;
	CUdeviceptr d_measuredCcdf_ = 0;

	// Real importance-sampled HDR sky distribution (LightSource "infinite"
	// with an image - see optix_types.h's GpuSkyDistribution comment). Both
	// GPU backends: recursive reads these via params.camera.skyDist (patched
	// in fresh inside render(), same lifecycle as CameraKind::Realistic's
	// lensElements/exitPupilBounds just above), wavefront receives the same
	// GpuSkyDistribution by value through the `camera` parameter its own
	// render() already takes - see WavefrontPathTracer::render()'s call
	// sites. skyHeight_ <= 0 (the default) means "no image sky in this
	// scene", matching GpuSkyDistribution::height's own sentinel.
	CUdeviceptr d_skyImagePixels_ = 0;
	CUdeviceptr d_skyMarginalCdf_ = 0;
	CUdeviceptr d_skyMarginalFunc_ = 0;
	CUdeviceptr d_skyConditionalCdf_ = 0;
	CUdeviceptr d_skyConditionalFunc_ = 0;
	CUdeviceptr d_skyConditionalFuncInt_ = 0;
	int skyWidth_ = 0, skyHeight_ = 0;
	float skyScale_ = 1.0f;
	float skyMarginalFuncInt_ = 0.0f;

	// pbrt-v4 "portal" (windowed) infinite light - see GpuPortalLight's own
	// comment (optix_types.h). Same lifecycle/dual-backend delivery as the
	// sky distribution just above; mutually exclusive with it (matches CPU).
	// portalHeight_ <= 0 (the default) means "no portal light in this scene".
	CUdeviceptr d_portalRectifiedImage_ = 0;
	CUdeviceptr d_portalDistFunc_ = 0;
	CUdeviceptr d_portalSatSum_ = 0;
	int portalWidth_ = 0, portalHeight_ = 0;
	float portalScale_ = 1.0f;
	float3 portalFrameX_ = {}, portalFrameY_ = {}, portalFrameZ_ = {};
	float3 portalP0_ = {}, portalP2_ = {};

	// Light sampling support for MIS
	CUdeviceptr d_lightIndices_ = 0;  ///< Device light primitive indices
	CUdeviceptr d_lightKinds_ = 0; ///< Device GpuLightKind array, one per light
	CUdeviceptr d_aliasTable_ = 0;    ///< Device alias table for power-weighted light sampling
	unsigned int numLights_ = 0;      ///< Number of emissive lights

	// pbrt-v4 bounding-cone light BVH (GpuCameraParams::cameraMediumSigmaT's
	// sibling gap, per docs/FEATURE_INVENTORY.md's own "no light BVH on GPU"
	// entry) - CPU default is bvh_light_sampler.h; GPU (recursive backend
	// only, this round) reuses the existing, previously-dead-code
	// BVHLightSampler2 (src/shared/bvh_light_sampler2.h) to build the tree
	// host-side, then uploads the flat node/bit-trail arrays here for
	// gpu_light_bvh_sample()/gpu_light_bvh_pmf() (optix_device_helpers.h) to
	// traverse device-side. lightBvhNodeCount_==0 (default) means "no light
	// BVH built" - every NEE call site falls back to the alias table exactly
	// as before this feature existed, same zero-init-safe convention as
	// every other optional GPU feature in this codebase.
	CUdeviceptr d_lightBvhNodes_ = 0;      ///< Device LightBVHNode array
	CUdeviceptr d_lightBvhBitTrail_ = 0;   ///< Device per-light bit-trail (numLights_ entries)
	int lightBvhNodeCount_ = 0;
	float lightBvhAllBMinX_ = 0, lightBvhAllBMinY_ = 0, lightBvhAllBMinZ_ = 0;
	float lightBvhAllBMaxX_ = 0, lightBvhAllBMaxY_ = 0, lightBvhAllBMaxZ_ = 0;

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
		bool haveInstancedSpheres,
		const std::vector<DiskData>& disks = {},
		const std::vector<CylinderData>& cylinders = {}
	);

	/// @brief Release all GPU resources
	void cleanup() noexcept;

	/// @brief Run the OptiX AI denoiser on an in-device float3 buffer, in
	///        place. Thin wrapper over the shared runDenoiser() (see
	///        optix_denoiser.h for the full "why" behind every step),
	///        passing this backend's own denoiserResources_/context_/
	///        stream_.
	/// @param d_buffer Device float3 RGB buffer, width*height, already
	///        accumulated/averaged (same layout as LaunchParams::framebuffer).
	/// @param d_albedo Optional device float3 albedo guide-layer buffer,
	///        same width*height layout as d_buffer. Pass 0 to denoise
	///        beauty-only (no albedo guide layer).
	/// @param d_normal Optional device float3 world-space normal guide-layer
	///        buffer, same width*height layout as d_buffer. Pass 0 to
	///        denoise without a normal guide layer.
	/// @return true on success; false (with a logged reason) if any OptiX/CUDA
	///         call in the sequence fails - render() treats this as
	///         non-fatal, since a failed denoise leaves the buffer's already-
	///         valid noisy render intact.
	bool denoise(CUdeviceptr d_buffer, unsigned int width, unsigned int height,
		CUdeviceptr d_albedo = 0, CUdeviceptr d_normal = 0);

	/// @brief Free the persisted denoiser and its device buffers - thin
	///        wrapper over destroyDenoiserResources(denoiserResources_).
	///        Safe to call when nothing is allocated. Called from
	///        cleanup() and from denoise() itself when the requested
	///        resolution no longer matches denoiserResources_'s own.
	void destroyDenoiser() noexcept;

	/// @brief (Re)allocate denoiserResources_'s AOV buffers for the given
	///        resolution - thin wrapper over
	///        ensureAovBuffers(denoiserResources_, ...).
	void ensureAovBuffers(unsigned int width, unsigned int height);

	/// @brief Free denoiserResources_'s AOV buffers - thin wrapper over
	///        destroyAovBuffers(denoiserResources_). Called from cleanup().
	void destroyAovBuffers() noexcept;

	/// @brief Load PTX shader code from file
	/// @param filename Path to PTX file
	/// @return PTX source code as string
	std::string loadPTX(const char* filename) const;
};
