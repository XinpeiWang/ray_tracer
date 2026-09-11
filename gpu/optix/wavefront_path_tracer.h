// wavefront_path_tracer.h
// WavefrontPathTracer -- pbrt-v4-style queue-based GPU path tracer.
#pragma once

#include "path_tracing_strategy.h"
#include "wavefront_types.h"
#include "optix_types.h"
#include "optix_denoiser.h"  // DenoiserResources - shared with OptiXRenderer
#include "svgf_tuning_params.h"
#include <optix.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <string>
#include <vector>

namespace optix_renderer {

class WavefrontPathTracer : public PathTracingStrategy {
public:
    WavefrontPathTracer();
    ~WavefrontPathTracer() override;
    bool initialize(OptixDeviceContext context, OptixModule module, cudaStream_t stream) override;
    bool createProgramGroups() override;
    bool linkPipeline(unsigned int maxTraceDepth) override;
    bool buildSBT(unsigned int numSpheres, unsigned int numQuads, unsigned int numBilinearPatches = 0, unsigned int numTriangles = 0,
                  unsigned int numDisks = 0, unsigned int numCylinders = 0) override;
    bool render(int width, int height, int samples_per_pixel, int max_depth,
        const GpuCameraParams& camera,
        float* framebuffer, OptixTraversableHandle gas_handle,
        CUdeviceptr d_materials, CUdeviceptr d_spheres, CUdeviceptr d_quads,
        CUdeviceptr d_light_indices, CUdeviceptr d_lightKinds,
        CUdeviceptr d_alias_table, unsigned int num_materials,
        unsigned int num_spheres, unsigned int num_quads,
        unsigned int num_lights,
        CUdeviceptr d_punctual_lights = 0,
        unsigned int num_punctual_lights = 0,
        CUdeviceptr d_bilinear_patches = 0,
        unsigned int num_bilinear_patches = 0,
        CUdeviceptr d_triangles = 0,
        unsigned int num_triangles = 0,
        CUdeviceptr d_disks = 0,
        unsigned int num_disks = 0,
        CUdeviceptr d_cylinders = 0,
        unsigned int num_cylinders = 0) override;
    void cleanup() override;
    PathTracingMode getMode() const override { return PathTracingMode::WAVEFRONT; }
    const char* getName() const override { return "WavefrontPathTracer"; }
    void setPTXPath(const std::string& path) { ptxPath_ = path; }

    /// Per-instance primitive base table (LaunchParams::instancePrimBase's
    /// twin - see optix_types.h). A setter rather than another render()
    /// parameter because render() is a virtual override shared with
    /// RecursivePathTracer, and widening it would churn an interface for data
    /// only this backend reads. 0 = no instancing, which is every built-in
    /// scene.
    void setInstancePrimBase(CUdeviceptr p) { d_instancePrimBase_ = p; }

    /// Texture metadata + shared pixel buffer (OptiXRenderer's own
    /// d_textures_/d_texturePixels_, already uploaded once at buildScene()
    /// time for the recursive path). Same setter-not-render()-parameter
    /// pattern as setInstancePrimBase() above, for the same reason: render()
    /// is a virtual override shared with RecursivePathTracer, and widening
    /// it would churn an interface for data only this backend reads this way
    /// (the recursive path gets it through LaunchParams instead). 0/0 (the
    /// default) is a valid "no textures in this scene" state - MaterialType::
    /// Lambertian/NormalMappedLambertian hits with textureIdx < 0 never
    /// dereference either pointer.
    void setTextures(CUdeviceptr d_textures, CUdeviceptr d_texturePixels) {
        d_textures_ = d_textures;
        d_texturePixels_ = d_texturePixels;
    }

    /// Heterogeneous cloud media (MaterialType::CloudMedium), OptiXRenderer's
    /// own d_cloudMediums_/numCloudMediums_, already uploaded once at
    /// buildScene() time for the recursive path. Same setter-not-render()-
    /// parameter pattern as setInstancePrimBase()/setTextures() above, for the
    /// same reason: render() is a virtual override shared with
    /// RecursivePathTracer. 0/0 (the default) is a valid "no cloud media in
    /// this scene" state.
    void setCloudMediums(CUdeviceptr d_cloudMediums, unsigned int numCloudMediums) {
        d_cloudMediums_ = d_cloudMediums;
        numCloudMediums_ = numCloudMediums;
    }

    /// Heterogeneous RGB grid media (MaterialType::RgbGridMedium) - same
    /// setter-not-render()-parameter pattern as setCloudMediums() above, for
    /// the same reason. 0/0/0/0 (the default) is a valid "no RGB grid media
    /// in this scene" state.
    void setRgbGridMediums(CUdeviceptr d_rgbGridMediums, unsigned int numRgbGridMediums,
                            CUdeviceptr d_rgbGridData, unsigned int rgbGridDataCount) {
        d_rgbGridMediums_ = d_rgbGridMediums;
        numRgbGridMediums_ = numRgbGridMediums;
        d_rgbGridData_ = d_rgbGridData;
        rgbGridDataCount_ = rgbGridDataCount;
    }

    /// Heterogeneous single-channel grid media (MaterialType::GridMedium) -
    /// same setter-not-render()-parameter pattern as setRgbGridMediums()
    /// above, for the same reason. 0/0/0/0 (the default) is a valid "no
    /// grid media in this scene" state.
    void setGridMediums(CUdeviceptr d_gridMediums, unsigned int numGridMediums,
                         CUdeviceptr d_gridData, unsigned int gridDataCount) {
        d_gridMediums_ = d_gridMediums;
        numGridMediums_ = numGridMediums;
        d_gridData_ = d_gridData;
        gridDataCount_ = gridDataCount;
    }

    /// Tabulated BSSRDF profile tables (MaterialType::Subsurface) - same
    /// setter-not-render()-parameter pattern as setCloudMediums()/
    /// setRgbGridMediums() above, for the same reason. These are the SAME
    /// device buffers OptiXRenderer already builds/uploads once for the
    /// recursive backend (see optix_types.h's GpuBssrdfTable comment and
    /// pbrt_gpu_builder.h's getOrBuildBssrdfTable() - already backend-
    /// agnostic, untouched by this wiring). 0/0/... (the default) is a
    /// valid "no Subsurface materials in this scene" state.
    void setBssrdfTables(CUdeviceptr d_bssrdfTables, unsigned int numBssrdfTables,
                          CUdeviceptr d_bssrdfRhoSamples, CUdeviceptr d_bssrdfRadiusSamples,
                          CUdeviceptr d_bssrdfProfile, CUdeviceptr d_bssrdfProfileCdf) {
        d_bssrdfTables_ = d_bssrdfTables;
        numBssrdfTables_ = numBssrdfTables;
        d_bssrdfRhoSamples_ = d_bssrdfRhoSamples;
        d_bssrdfRadiusSamples_ = d_bssrdfRadiusSamples;
        d_bssrdfProfile_ = d_bssrdfProfile;
        d_bssrdfProfileCdf_ = d_bssrdfProfileCdf;
    }

    /// Real tabulated measured-BRDF tables (MaterialType::Measured) - same
    /// setter-not-render()-parameter pattern as setCloudMediums()/
    /// setRgbGridMediums()/setBssrdfTables() above, for the same reason.
    /// These are the SAME device buffers OptiXRenderer already builds/
    /// uploads once for the recursive backend (see optix_types.h's
    /// GpuMeasuredTable comment and pbrt_gpu_builder.h's
    /// getOrBuildMeasuredTable() - already backend-agnostic). 0/0/... (the
    /// default) is a valid "no Measured materials in this scene" state.
    void setMeasuredTables(CUdeviceptr d_measuredTables, unsigned int numMeasuredTables,
                            CUdeviceptr d_measuredParamValues, CUdeviceptr d_measuredData,
                            CUdeviceptr d_measuredMcdf, CUdeviceptr d_measuredCcdf) {
        d_measuredTables_ = d_measuredTables;
        numMeasuredTables_ = numMeasuredTables;
        d_measuredParamValues_ = d_measuredParamValues;
        d_measuredData_ = d_measuredData;
        d_measuredMcdf_ = d_measuredMcdf;
        d_measuredCcdf_ = d_measuredCcdf;
    }

    /// Whether the scene has instanced geometry of each kind. buildSBT() must
    /// append the same dedicated hit-record pairs, in the same order, that
    /// OptiXRenderer::buildSBT() does - the IAS instances carry sbtOffsets
    /// computed against that layout, and both SBTs are indexed by them, so a
    /// layout that disagrees sends a hit to the wrong program. Call before
    /// buildSBT().
    void setInstancedGeometryFlags(bool haveTriangles, bool haveSpheres) {
        haveInstancedTriangles_ = haveTriangles;
        haveInstancedSpheres_ = haveSpheres;
    }

    /// Whether this render() call should also run the OptiX AI denoiser
    /// (mirrors OptiXRenderer::enableDenoise() - see that method's own
    /// comment). Same setter-not-render()-parameter pattern as
    /// setInstancePrimBase()/setTextures()/etc above, for the same reason.
    /// Set per render() call by OptiXRenderer::render() right before
    /// delegating here, so a scene/mode switch never leaves a stale value
    /// wired in. false (the default) is every scene's prior behavior -
    /// --denoise previously had no effect under --wavefront at all.
    void setDenoiseEnabled(bool enabled) { denoiseEnabled_ = enabled; }

    /// Mirrors OptiXRenderer::setDenoiseBlend() - see that method's own
    /// comment. Same setter-not-render()-parameter pattern, same reasoning.
    void setDenoiseBlend(float blend) { denoiseBlend_ = blend; }

    /// Whether render() should also fill a per-pixel world-space primary-hit
    /// buffer (see d_worldPos_'s own comment) for Live Preview's temporal
    /// reprojection (qt_gui/realtime_preview_session.cpp, camera_math.h's
    /// projectToScreen()). Same setter-not-render()-parameter pattern as
    /// setDenoiseEnabled() above, for the same reason - only the realtime
    /// path ever sets this true; batch/video rendering never does, so it
    /// never pays for the extra allocation/write.
    void setWorldPosOutputEnabled(bool enabled) { worldPosOutputEnabled_ = enabled; }

    /// Reads back the persisted world-position buffer (see
    /// setWorldPosOutputEnabled()'s own comment) to host memory - same
    /// "separate consumer of a buffer render() already populated" shape as
    /// OptiXRenderer::readAovBuffers(). Only valid after a render() call with
    /// setWorldPosOutputEnabled(true) already in effect, at this exact
    /// resolution; returns false (leaving `out` untouched) otherwise.
    /// @param out Resized to width*height*4 floats (xyz + validity, row-
    ///        major) on success.
    bool readWorldPosBuffer(unsigned int width, unsigned int height, std::vector<float>& out) const;

    /// Whether render() should route primary-hit (depth==0) NEE through
    /// ReSTIR DI's weighted resampling (gpu/optix/wavefront_restir_helpers.h)
    /// instead of the classic single alias-table draw - see
    /// wf_finish_material_scatter's own restirReservoirs parameter comment.
    /// Same setter-not-render()-parameter pattern as setWorldPosOutputEnabled()
    /// above and for the same reason: only Live Preview ever sets this true,
    /// so batch/offline rendering (MaterialCpuGpuParityTest and friends) never
    /// pays for the extra allocation and keeps today's exact single-draw NEE
    /// statistics, unchanged.
    void setRestirEnabled(bool enabled) { restirEnabled_ = enabled; }

    /// Whether render() should also resample one-bounce indirect lighting via
    /// ReSTIR GI (gpu/optix/wavefront_restir_gi_math.h) - see
    /// wf_finish_material_scatter's own giOriginContext/giCandidateRadianceOut
    /// parameter comments. Independent from setRestirEnabled() (DI) above -
    /// either can be on/off on its own. Same "only Live Preview ever sets
    /// this true" opt-in shape, so batch/offline rendering never pays for the
    /// extra buffers or changes its own statistics.
    void setRestirGiEnabled(bool enabled) { restirGiEnabled_ = enabled; }

    /// Whether render() should run the SVGF spatiotemporal denoiser
    /// (gpu/optix/wavefront_svgf_math.h, wavefront_kernels_svgf.cu) on this
    /// frame's raw radiance before returning it - see that file's own
    /// header comment. Independent from DI/GI above (a Live Preview user
    /// can denoise with or without either resampling technique). Same
    /// "only Live Preview ever sets this true" opt-in shape - batch/offline
    /// rendering never pays for the extra buffers or changes its own
    /// statistics.
    void setSvgfEnabled(bool enabled) { svgfEnabled_ = enabled; }
    // SVGF's advanced tuning constants (gpu/optix/svgf_tuning_params.h) -
    // read directly from svgfTuning_ at every wf_launch_svgf_*() call site
    // inside launchSvgf() (this class's own .cpp), replacing what used to be
    // kSvgf* file-scope constexpr literals in wavefront_kernels_svgf.cu.
    void setSvgfTuning(const SvgfTuningParams& tuning) { svgfTuning_ = tuning; }

    /// Clears every technique's temporal history flag together - DI's
    /// restirHistoryValid_, GI's restirGiHistoryValid_, and SVGF's own
    /// svgfHistoryValid_ - one call for all three, since a scene switch/
    /// hard-reset invalidates all of them for the same reason at the same
    /// time, and no caller has ever needed to invalidate only one. Call
    /// whenever a scene switch/upload happens (a previous scene's light/
    /// indirect/temporally-integrated samples must never be reused into a
    /// new scene) - the actual history buffers are left as-is (they get
    /// fully overwritten on the next render() call anyway, same "no need to
    /// eagerly clear" reasoning as every other GPU buffer here) and are
    /// resized/reallocated normally if the resolution changed.
    void invalidateRestirHistory() { restirHistoryValid_ = false; restirGiHistoryValid_ = false; svgfHistoryValid_ = false; }

private:
    // Resize-on-resolution-change helper for a per-pixel GPU buffer: frees
    // and reallocates only when `capacity` doesn't already match `numPixels`,
    // exactly the pattern every resize-on-demand buffer in this class already
    // followed by hand (d_worldPos_, d_reservoirs_, d_reservoirsHistory_,
    // d_restirNormal_, and originally each of the 4 new GI buffers too - see
    // ReSTIR GI's own buffer-allocation block in render() for where this
    // replaced 4 near-identical copy-pasted blocks). Returns true iff a
    // reallocation actually happened (so a caller that needs to memset the
    // fresh allocation - e.g. once, for a buffer that's otherwise cleared per
    // SAMPLE rather than per render() call - knows whether one just occurred).
    // Left as a template (not reused by the older DI buffers above, which
    // predate it and are already independently tested) rather than retrofit
    // every existing call site in the same pass as ReSTIR GI's own fix.
    template <typename T>
    bool reallocateDeviceBufferIfNeeded(CUdeviceptr& ptr, int& capacity, int numPixels) {
        if (capacity == numPixels) return false;
        if (ptr) { cudaFree(reinterpret_cast<void*>(ptr)); ptr = 0; }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), static_cast<size_t>(numPixels) * sizeof(T)));
        capacity = numPixels;
        return true;
    }
    // Matching teardown for the "GI disabled" branch - frees unconditionally
    // (a no-op if already null) and resets capacity so the next enable sees
    // capacity != numPixels and reallocates fresh.
    void freeDeviceBuffer(CUdeviceptr& ptr, int& capacity) {
        if (ptr) { cudaFree(reinterpret_cast<void*>(ptr)); ptr = 0; }
        capacity = 0;
    }

    bool loadModule();
    void destroyProgramGroups();
    void destroySBT();
    bool allocateQueues(int numPixels);
    void freeQueues();
    void launchGenerateCameraRays(int width, int height, int sampleIdx,
        const GpuCameraParams& camera, float* d_weightBuffer, bool checkerboardActive);
    // Builds this call's ReSTIR temporal-reuse context from d_reservoirsHistory_/
    // d_worldPosHistory_/d_restirNormal_/prevRestirCamera_/restirHistoryValid_/
    // restirImageWidth_/restirImageHeight_ - shared by all 3 launchEvaluateMaterials*()
    // methods instead of each rebuilding it. Returns a default (historyValid=false)
    // context when !restirEnabled_, the same safe-no-op shape restirReservoirs
    // being null already has.
    GpuRestirTemporalContext buildRestirTemporalContext() const;
    // ReSTIR DI spatial reuse - see wavefront_kernels_restir.cu's own header
    // comment. Called once per render() call (after the whole sampleIdx
    // loop), reading d_reservoirs_/d_restirNormal_/d_worldPos_ (this call's
    // own) and writing into d_reservoirsHistory_ - which becomes the NEXT
    // call's temporal-reuse source.
    void launchRestirSpatialReuse(
        const SphereData* d_spheres, const QuadData* d_quads, const TriangleData* d_triangles,
        const BilinearPatchData* d_bilinearPatches, const DiskData* d_disks, const CylinderData* d_cylinders,
        const MaterialData* d_materials);
    // ReSTIR GI finalize - see wavefront_kernels_restir.cu's own
    // restir_gi_finalize header comment. Runs once per SAMPLE (called from
    // inside the per-depth loop, right after depth==1's own shadow-ray
    // accumulation - see render()'s own call site), unlike DI's spatial
    // reuse which runs once per whole render() call.
    void launchGiFinalize(const MaterialData* d_materials, float3* d_framebuffer, float maxComponentValue, const float* d_weightBuffer);
    // ReSTIR GI spatial reuse - see wavefront_kernels_restir.cu's own
    // restir_gi_spatial_reuse header comment. Called once per render() call,
    // same timing as launchRestirSpatialReuse() (DI's own).
    void launchGiSpatialReuse();
    // SVGF (wavefront_kernels_svgf.cu) - runs the full temporal-integrate +
    // A-trous filter sequence in place on d_framebuffer, once per render()
    // call, after launchNormalizeFramebuffer (this frame's raw radiance
    // must already be normalized/final before SVGF treats it as "this
    // frame's noisy 1-spp sample" - see wavefront_kernels_svgf.cu's own
    // header comment).
    void launchSvgf(float3* d_framebuffer, const float3* d_albedoAov, float3 cameraOrigin, const float* d_weightBuffer);
    void launchEvaluateMaterials(int numHits, int maxDepth, bool regularize, float maxComponentValue,
        const SphereData* d_spheres, unsigned int numSpheres,
        const QuadData* d_quads, unsigned int numQuads,
        const TriangleData* d_triangles, unsigned int numTriangles,
        const BilinearPatchData* d_bilinearPatches, unsigned int numBilinearPatches,
        const DiskData* d_disks, unsigned int numDisks,
        const CylinderData* d_cylinders, unsigned int numCylinders,
        const MaterialData* d_materials, unsigned int numMaterials,
        const int* d_lightIndices, const GpuLightKind* d_lightKinds,
        const GpuAliasEntry* d_aliasTable, unsigned int numLights,
        const PunctualLightGPU* d_punctualLights, unsigned int numPunctualLights,
        float3* d_framebuffer, float3 skyColor, float shadowRayEpsilon, GpuSkyDistribution skyDist, GpuPortalLight portalLight);
    // Twin of launchEvaluateMaterials() above, scoped to simpleHitQueue's
    // Lambertian/Metal hits (see wavefront_types.h's WavefrontQueues::
    // simpleHitQueue and wavefront_kernels.cu's evaluate_materials_simple()).
    // Fewer scene-data params - no cloud/RGB-grid medium or measured-BRDF
    // tables, since none of those material types can reach this queue.
    void launchEvaluateMaterialsSimple(int numHits, int maxDepth,
        const SphereData* d_spheres, unsigned int numSpheres,
        const QuadData* d_quads, unsigned int numQuads,
        const TriangleData* d_triangles, unsigned int numTriangles,
        const BilinearPatchData* d_bilinearPatches, unsigned int numBilinearPatches,
        const DiskData* d_disks, unsigned int numDisks,
        const CylinderData* d_cylinders, unsigned int numCylinders,
        const MaterialData* d_materials, unsigned int numMaterials,
        const int* d_lightIndices, const GpuLightKind* d_lightKinds,
        const GpuAliasEntry* d_aliasTable, unsigned int numLights,
        const PunctualLightGPU* d_punctualLights, unsigned int numPunctualLights,
        float3* d_framebuffer, float3 skyColor, float shadowRayEpsilon, GpuSkyDistribution skyDist, GpuPortalLight portalLight, float maxComponentValue);
    // Twin of launchEvaluateMaterialsSimple() above, scoped to
    // dielectricHitQueue's Dielectric/RoughDielectric hits (see
    // wavefront_types.h's WavefrontQueues::dielectricHitQueue and
    // wavefront_kernels.cu's evaluate_materials_dielectric()). No texture or
    // scene-medium/measured-BRDF params at all - neither material type ever
    // reads them.
    void launchEvaluateMaterialsDielectric(int numHits, int maxDepth, bool regularize, float maxComponentValue,
        const SphereData* d_spheres, unsigned int numSpheres,
        const QuadData* d_quads, unsigned int numQuads,
        const TriangleData* d_triangles, unsigned int numTriangles,
        const BilinearPatchData* d_bilinearPatches, unsigned int numBilinearPatches,
        const DiskData* d_disks, unsigned int numDisks,
        const CylinderData* d_cylinders, unsigned int numCylinders,
        const MaterialData* d_materials, unsigned int numMaterials,
        const int* d_lightIndices, const GpuLightKind* d_lightKinds,
        const GpuAliasEntry* d_aliasTable, unsigned int numLights,
        const PunctualLightGPU* d_punctualLights, unsigned int numPunctualLights,
        float3* d_framebuffer, float3 skyColor, float shadowRayEpsilon, GpuSkyDistribution skyDist, GpuPortalLight portalLight);
    // Neither this nor launchAccumulateMiss() below take the AOV guide
    // buffers as launchX-style params - like textures (see setTextures()'s
    // comment), they travel via denoiserResources_ member state instead,
    // read directly inside each kernel-launch method's own body.
    void launchAccumulateMiss(int numMiss, float3* d_framebuffer, float3 backgroundColor, GpuSkyDistribution skyDist, GpuPortalLight portalLight, float maxComponentValue);
    void launchAccumulateShadow(int numShadow, const bool* d_occluded, float3* d_framebuffer, float maxComponentValue);
    void launchResolveBssrdfExit(int numExit,
        const MaterialData* d_materials, unsigned int numMaterials,
        const SphereData* d_spheres, unsigned int numSpheres,
        const QuadData* d_quads, unsigned int numQuads,
        const TriangleData* d_triangles, unsigned int numTriangles,
        const BilinearPatchData* d_bilinearPatches, unsigned int numBilinearPatches,
        const DiskData* d_disks, unsigned int numDisks,
        const CylinderData* d_cylinders, unsigned int numCylinders,
        const int* d_lightIndices, const GpuLightKind* d_lightKinds,
        const GpuAliasEntry* d_aliasTable, unsigned int numLights,
        const PunctualLightGPU* d_punctualLights, unsigned int numPunctualLights,
        float3* d_framebuffer, float3 skyColor, float shadowRayEpsilon, GpuSkyDistribution skyDist, GpuPortalLight portalLight, float maxComponentValue);
    void launchNormalizeFramebuffer(unsigned int numPixels, const float* d_weightBuffer, float3* d_framebuffer);
    // Denoiser guide-layer AOV normalization (--denoise only) - plain mean
    // over samplesPerPixel, not filter-weighted like launchNormalizeFramebuffer()
    // above. See normalize_aov_buffers()'s own comment (wavefront_kernels.cu).
    void launchNormalizeAovBuffers(unsigned int numPixels, float3* d_albedoBuffer,
        float3* d_normalBuffer, unsigned int samplesPerPixel);
    int  readQueueSize(int* d_counter);
    void resetQueueCounter(int* d_counter);

    // OptiX AI denoiser (--denoise, this backend's own instance) - thin
    // wrappers over the shared runDenoiser()/destroyDenoiserResources()/
    // ensureAovBuffers()/destroyAovBuffers() free functions (optix_denoiser.h),
    // also used by OptiXRenderer (optix_renderer_render.cpp). No shared base
    // class or back-reference to the owning OptiXRenderer needed:
    // WavefrontPathTracer already has its own context_/stream_
    // (PathTracingStrategy's protected members), and the free functions take
    // those as explicit parameters against this class's own
    // denoiserResources_ instance.
    bool denoise(CUdeviceptr d_buffer, unsigned int width, unsigned int height,
        CUdeviceptr d_albedo, CUdeviceptr d_normal);
    void destroyDenoiser() noexcept;
    void ensureAovBuffers(unsigned int width, unsigned int height);
    void destroyAovBuffers() noexcept;

    OptixModule wfModule_ = nullptr;
    OptixPipelineCompileOptions pipelineCompileOptions_ = {};
    OptixProgramGroup raygenIntersectPG_ = nullptr;
    OptixProgramGroup missRadiancePG_    = nullptr;
    OptixProgramGroup hitSpherePG_       = nullptr;
    OptixProgramGroup hitQuadPG_         = nullptr;
    OptixProgramGroup hitBilinearPatchPG_ = nullptr;
    // Disk/Cylinder (Phase 4c) - see buildSBT()'s own comment for why these
    // land at the very end of intersectSBT_/shadowSBT_/probeSBT_'s hit-record
    // arrays, after triangle and the instanced-geometry pairs, mirroring
    // OptiXRenderer::buildScene()'s diskCylinderSbtOffset placement exactly.
    OptixProgramGroup hitDiskPG_         = nullptr;
    OptixProgramGroup hitCylinderPG_     = nullptr;
    OptixProgramGroup hitTrianglePG_     = nullptr;
    OptixProgramGroup raygenShadowPG_        = nullptr;
    OptixProgramGroup missShadowPG_          = nullptr;
    OptixProgramGroup anyhitShadowSpherePG_  = nullptr;
    OptixProgramGroup anyhitShadowQuadPG_    = nullptr;
    OptixProgramGroup anyhitShadowBilinearPatchPG_ = nullptr;
    OptixProgramGroup anyhitShadowDiskPG_    = nullptr;
    OptixProgramGroup anyhitShadowCylinderPG_ = nullptr;
    OptixProgramGroup anyhitShadowTrianglePG_ = nullptr;
    // BSSRDF probe walk (MaterialType::Subsurface, Phase 2) - linked into
    // the SAME intersectPipeline_/wfModule_ as the intersect programs above
    // (see wavefront_probe.h's own header comment for why: no new OptiX
    // module/pipeline is needed, just more program groups in the same
    // pipeline, selected via their own dedicated probeSBT_ at launch time).
    OptixProgramGroup raygenProbePG_             = nullptr;
    OptixProgramGroup missProbePG_               = nullptr;
    OptixProgramGroup hitProbeSpherePG_          = nullptr;
    OptixProgramGroup hitProbeQuadPG_            = nullptr;
    OptixProgramGroup hitProbeBilinearPatchPG_   = nullptr;
    OptixProgramGroup hitProbeDiskPG_            = nullptr;
    OptixProgramGroup hitProbeCylinderPG_        = nullptr;
    OptixProgramGroup hitProbeTrianglePG_        = nullptr;
    OptixProgramGroup exceptionPG_ = nullptr;  ///< CUDA-718 fix -- see initialize()'s exceptionFlags comment
    OptixPipeline intersectPipeline_ = nullptr;
    OptixPipeline shadowPipeline_    = nullptr;
    OptixShaderBindingTable intersectSBT_ = {};
    OptixShaderBindingTable shadowSBT_    = {};
    // Probe SBT - same intersectPipeline_, own raygen/hit records (mirrors
    // intersectSBT_'s own per-present-type/stride-RAY_TYPE_COUNT hit-record
    // layout exactly, just pointing at the probe hit groups instead of the
    // radiance ones - see buildSBT()'s own pushTriple comment).
    OptixShaderBindingTable probeSBT_     = {};
    CUdeviceptr d_intersectRaygenRecord_ = 0;
    CUdeviceptr d_intersectMissRecord_   = 0;
    CUdeviceptr d_intersectHitRecords_   = 0;
    CUdeviceptr d_shadowRaygenRecord_    = 0;
    CUdeviceptr d_shadowMissRecord_      = 0;
    CUdeviceptr d_shadowHitRecords_      = 0;
    CUdeviceptr d_probeRaygenRecord_     = 0;
    CUdeviceptr d_probeMissRecord_       = 0;
    CUdeviceptr d_probeHitRecords_       = 0;
    CUdeviceptr d_probeExceptionRecord_  = 0;
    CUdeviceptr d_intersectExceptionRecord_ = 0;  ///< see exceptionPG_
    CUdeviceptr d_shadowExceptionRecord_    = 0;
    CUdeviceptr d_wfLaunchParams_   = 0;
    CUdeviceptr d_rayItems_         = 0;
    CUdeviceptr d_nextRayItems_     = 0;
    CUdeviceptr d_hitItems_         = 0;
    CUdeviceptr d_simpleHitItems_   = 0;   ///< see WavefrontQueues::simpleHitQueue
    CUdeviceptr d_dielectricHitItems_ = 0; ///< see WavefrontQueues::dielectricHitQueue
    CUdeviceptr d_missItems_        = 0;
    CUdeviceptr d_shadowItems_      = 0;
    CUdeviceptr d_occluded_         = 0;
    CUdeviceptr d_probeItems_       = 0;   ///< BssrdfProbeWorkItem queue
    CUdeviceptr d_exitItems_        = 0;   ///< BssrdfExitWorkItem queue
    CUdeviceptr d_rayCounter_       = 0;
    CUdeviceptr d_nextRayCounter_   = 0;
    CUdeviceptr d_hitCounter_       = 0;
    CUdeviceptr d_simpleHitCounter_ = 0;   ///< see d_simpleHitItems_
    CUdeviceptr d_dielectricHitCounter_ = 0; ///< see d_dielectricHitItems_
    CUdeviceptr d_missCounter_      = 0;
    CUdeviceptr d_shadowCounter_    = 0;
    CUdeviceptr d_probeCounter_     = 0;
    CUdeviceptr d_exitCounter_      = 0;

    // Own stream for launchEvaluateMaterialsSimple()'s kernel, separate from
    // the base class's stream_ (externally owned by OptiXRenderer, shared
    // across every backend strategy - see PathTracingStrategy::stream_).
    // hitQueue and simpleHitQueue are disjoint (routed at push time in
    // wavefront_programs.cu) and both evaluate-materials kernels only ever
    // write into shared queues via atomicAdd-based WorkQueue::push(), so
    // running them on separate streams lets the GPU actually overlap them
    // instead of serializing two kernels that have no real dependency on
    // each other. Owned and destroyed by this class (unlike stream_).
    cudaStream_t simpleMaterialStream_ = nullptr;
    // Own stream for launchEvaluateMaterialsDielectric()'s kernel, same
    // overlap reasoning as simpleMaterialStream_ above - hitQueue/
    // simpleHitQueue/dielectricHitQueue are mutually disjoint at push time.
    cudaStream_t dielectricMaterialStream_ = nullptr;
    int          queueCapacity_ = 0;
    CUdeviceptr  d_bssrdfTables_ = 0;         ///< see setBssrdfTables()
    unsigned int numBssrdfTables_ = 0;
    CUdeviceptr  d_bssrdfRhoSamples_ = 0;
    CUdeviceptr  d_bssrdfRadiusSamples_ = 0;
    CUdeviceptr  d_bssrdfProfile_ = 0;
    CUdeviceptr  d_bssrdfProfileCdf_ = 0;

    CUdeviceptr  d_measuredTables_ = 0;       ///< see setMeasuredTables()
    unsigned int numMeasuredTables_ = 0;
    CUdeviceptr  d_measuredParamValues_ = 0;
    CUdeviceptr  d_measuredData_ = 0;
    CUdeviceptr  d_measuredMcdf_ = 0;
    CUdeviceptr  d_measuredCcdf_ = 0;
    std::string  ptxPath_;
    CUdeviceptr  d_instancePrimBase_ = 0;   ///< see setInstancePrimBase()
    CUdeviceptr  d_textures_ = 0;           ///< see setTextures()
    CUdeviceptr  d_texturePixels_ = 0;
    CUdeviceptr  d_cloudMediums_ = 0;       ///< see setCloudMediums()
    unsigned int numCloudMediums_ = 0;
    CUdeviceptr  d_rgbGridMediums_ = 0;     ///< see setRgbGridMediums()
    unsigned int numRgbGridMediums_ = 0;
    CUdeviceptr  d_rgbGridData_ = 0;
    unsigned int rgbGridDataCount_ = 0;
    CUdeviceptr  d_gridMediums_ = 0;        ///< see setGridMediums()
    unsigned int numGridMediums_ = 0;
    CUdeviceptr  d_gridData_ = 0;
    unsigned int gridDataCount_ = 0;
    bool         haveInstancedTriangles_ = false;  ///< see setInstancedGeometryFlags()
    bool         haveInstancedSpheres_ = false;
    unsigned int numSpheres_  = 0;
    unsigned int numQuads_    = 0;
    unsigned int numBilinearPatches_ = 0;
    unsigned int numTriangles_ = 0;
    unsigned int numDisks_ = 0;
    unsigned int numCylinders_ = 0;
    unsigned int frameNumber_ = 0;

    // Framebuffer accumulator + per-pixel filter-weight buffer (render()'s
    // own d_fb/d_weight) - persisted across calls and only reallocated when
    // numPixels changes, same resolution-keyed skip-reallocation pattern
    // allocateQueues() already uses for the (much larger) per-bounce work
    // queues above. Matters for a tight repeated-call loop (e.g. a live
    // preview re-rendering every frame at fixed resolution): avoids a
    // cudaMalloc/cudaFree pair on every single call for buffers that are
    // the same size every time anyway. Zeroed at the start of every render()
    // call regardless (these accumulate per-call, not across calls).
    CUdeviceptr  d_fb_ = 0;
    CUdeviceptr  d_weight_ = 0;
    int          fbCapacity_ = 0;

    // --denoise support (see setDenoiseEnabled()/denoise()) - own instance
    // of the shared DenoiserResources (optix_denoiser.h), persisted across
    // render() calls (recreated only on a resolution change) for the same
    // reason OptiXRenderer's own instance is: avoids re-running denoiser
    // create/setup/3-allocations on every one of video mode's hundreds of
    // same-resolution per-frame render() calls.
    bool             denoiseEnabled_ = false;
    float            denoiseBlend_ = 0.0f;  ///< See setDenoiseBlend()
    DenoiserResources denoiserResources_;

    // Live Preview reprojection guide buffer - see setWorldPosOutputEnabled()'s
    // own comment. Same resolution-keyed allocate-once/only-realloc-on-change
    // lifecycle as d_fb_/fbCapacity_ above, but a SEPARATE capacity tracker
    // and SEPARATE (conditional-on-worldPosOutputEnabled_) allocation, since
    // unlike d_fb_/d_weight_ this is opt-in - batch/video rendering never
    // sets worldPosOutputEnabled_, so it never pays for this allocation.
    // xyz = world-space primary-hit point, w = validity (1.0 = real hit,
    // 0.0 = miss or never-written) - see wf_write_world_pos()'s own comment
    // (wavefront_device_helpers.h).
    CUdeviceptr      d_worldPos_ = 0;
    int              worldPosCapacity_ = 0;
    bool             worldPosOutputEnabled_ = false;

    // ReSTIR DI (Live Preview only) current-frame reservoir buffer - see
    // setRestirEnabled()'s own comment. Same resolution-keyed allocate-once/
    // only-realloc-on-change lifecycle as d_worldPos_ above, and likewise a
    // SEPARATE capacity tracker and SEPARATE (conditional-on-restirEnabled_)
    // allocation - batch/video rendering never sets restirEnabled_, so it
    // never pays for this allocation either. Scratch: every render() call's
    // sampleIdx loop fully overwrites every entry it reaches (evaluate_
    // materials*'s own restir block writes restirReservoirs[pixelIndex]
    // unconditionally for every depth==0 non-specular hit).
    CUdeviceptr      d_reservoirs_ = 0;
    int              reservoirsCapacity_ = 0;
    bool             restirEnabled_ = false;

    // ReSTIR temporal reuse's cross-call history - see GpuRestirTemporalContext's
    // own comment (optix_types.h) for the read-then-overwrite lifecycle within
    // one render() call: d_reservoirsHistory_/d_worldPosHistory_ are READ at
    // the start of this call's sampleIdx loop (as "last call's result"), then
    // OVERWRITTEN with THIS call's own final reservoirs/world-pos at the end
    // of render() - safe without double-buffering because both uses are
    // strictly ordered by stream_'s own launch order, never concurrent.
    // d_restirNormal_ is this call's own per-pixel shading normal (plain
    // overwrite at depth==0, mirrors d_worldPos_) - kept for the follow-on
    // spatial-reuse pass's own neighbor-rejection test, not read across calls.
    // prevRestirCamera_/restirHistoryValid_ persist in host memory (not a GPU
    // buffer) - restirHistoryValid_ starts false and is only ever set true at
    // the end of a successful restirEnabled_ render() call, so the very first
    // call after enabling ReSTIR (or after invalidateRestirHistory(), e.g. on
    // a scene change) correctly skips temporal reuse instead of reading an
    // empty/stale history buffer.
    CUdeviceptr        d_reservoirsHistory_ = 0;
    int                reservoirsHistoryCapacity_ = 0;
    CUdeviceptr        d_worldPosHistory_ = 0;
    int                worldPosHistoryCapacity_ = 0;
    CUdeviceptr        d_restirNormal_ = 0;
    int                restirNormalCapacity_ = 0;
    GpuReprojectBasis  prevRestirCamera_{};
    bool               restirHistoryValid_ = false;
    // The exact width/height the CURRENT content of d_reservoirsHistory_/
    // d_worldPosHistory_ was laid out under - checked (not just numPixels =
    // width*height) at the top of every render() call, because a resolution
    // change whose width*height product happens to equal the previous
    // call's (e.g. 1024x768 -> 768x1024) would otherwise leave
    // reservoirsHistoryCapacity_/worldPosHistoryCapacity_ matching and
    // restirHistoryValid_ untouched, while temporal reuse computes
    // `py*imageWidth+px` against the NEW width over buffer content still
    // indexed under the OLD width - a row-major misindex reading unrelated
    // pixels' data. Set alongside restirHistoryValid_=true at the end of a
    // successful restirEnabled_ render() call.
    int                restirHistoryWidth_ = 0;
    int                restirHistoryHeight_ = 0;
    // Set at the top of every render() call - read by launchEvaluateMaterials*()
    // to build this call's GpuRestirTemporalContext (imageWidth/imageHeight),
    // which those private methods otherwise have no width/height parameter to
    // derive from (they read every other ReSTIR buffer directly off `this`
    // the same way, e.g. d_worldPos_ above).
    int                restirImageWidth_ = 0;
    int                restirImageHeight_ = 0;

    // ReSTIR GI (Live Preview only, gpu/optix/wavefront_restir_gi_math.h) -
    // own buffers, parallel to DI's own above but for a genuinely different
    // payload (GpuGiSample vs GpuLightSample) - reuses d_worldPos_/
    // d_worldPosHistory_/prevRestirCamera_/restirImageWidth_/
    // restirImageHeight_ as-is for reprojection (those only ever depend on
    // x0's own position/the camera, never on which ReSTIR technique is
    // consuming them - see the render()-time allocation gate comment on
    // d_worldPos_ for why its own gate now includes restirGiEnabled_ too).
    bool               restirGiEnabled_ = false;

    // Per-pixel stash of x0's shading context (GpuGiOriginContext, written at
    // depth==0, consumed by the GI finalize pass once depth 1 resolves - see
    // that struct's own comment, wavefront_types.h, for why this exists at
    // all). Pure per-frame scratch, same "harmless zeroed-out state, memset
    // every call" reasoning as d_reservoirs_ above.
    CUdeviceptr        d_giOriginContext_ = 0;
    int                giOriginContextCapacity_ = 0;

    // Per-pixel GI candidate (GpuGiSample) - x1Point/x1Normal/x0Point/pdfAtX0
    // written synchronously by wf_finish_material_scatter's own depth==1
    // handling, `.radiance` filled in afterward/asynchronously by
    // accumulate_shadow once occlusion resolves (see that parameter's own
    // comment, wavefront_device_helpers.h/wavefront_kernels_accumulate.cu).
    // Same pure-scratch/memset-every-call reasoning as d_giOriginContext_.
    CUdeviceptr        d_giCandidateOut_ = 0;
    int                giCandidateOutCapacity_ = 0;

    // GI's own current-frame/history reservoir buffers - same resolution-
    // keyed allocate-once/only-realloc-on-change lifecycle, and the same
    // read-then-overwrite-at-end-of-call cross-call relationship, as DI's
    // own d_reservoirs_/d_reservoirsHistory_ (that struct's own header
    // comment already explains the shape; not repeated here).
    CUdeviceptr        d_giReservoirs_ = 0;
    int                giReservoirsCapacity_ = 0;
    CUdeviceptr        d_giReservoirsHistory_ = 0;
    int                giReservoirsHistoryCapacity_ = 0;
    // Mirrors restirHistoryValid_ exactly, but for GI's own history buffer -
    // kept as a SEPARATE flag (not folded into restirHistoryValid_) because
    // restirEnabled_ (DI) and restirGiEnabled_ (GI) are independently
    // toggleable; invalidateRestirHistory() clears both together since a
    // scene switch/hard reset invalidates both for the same reason at the
    // same time (see that method's own comment).
    bool               restirGiHistoryValid_ = false;

    // SVGF (Live Preview only, gpu/optix/wavefront_svgf_math.h) - own
    // buffers, independent of DI/GI above (a user can enable SVGF with or
    // without either resampling technique). d_svgfCurrent_/d_svgfHistory_
    // hold GpuSvgfState (color/moments/historyLength), double-buffered and
    // read-then-overwrite-at-end-of-call exactly like d_giReservoirs_/
    // d_giReservoirsHistory_ above - reused reasoning, not repeated.
    bool               svgfEnabled_ = false;
    SvgfTuningParams   svgfTuning_;             // see setSvgfTuning()
    CUdeviceptr        d_svgfCurrent_ = 0;
    int                svgfCurrentCapacity_ = 0;
    CUdeviceptr        d_svgfHistory_ = 0;
    int                svgfHistoryCapacity_ = 0;
    bool               svgfHistoryValid_ = false;
    // A-trous ping-pong scratch (float3 color + float variance per pixel,
    // packed as float4 - xyz=color, w=variance) - purely transient WITHIN
    // one render() call's own filter pass sequence, never read across
    // frames, unlike every other buffer on this class. Two buffers so each
    // pass can read the previous pass's output while writing its own,
    // without a read/write race within one kernel launch (the same reason
    // DI's own spatial-reuse pass writes to a SEPARATE output buffer rather
    // than mutating its input in place).
    CUdeviceptr        d_svgfPingPong_[2] = {0, 0};
    // One capacity slot per element (not a single shared int) so each can go
    // through reallocateDeviceBufferIfNeeded<float4>()/freeDeviceBuffer()
    // independently - a shared capacity would make the second element's own
    // call see capacity already == numPixels (set by the first element's
    // call) and skip its own allocation entirely.
    int                svgfPingPongCapacity_[2] = {0, 0};
};

} // namespace optix_renderer
