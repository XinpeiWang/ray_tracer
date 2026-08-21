// wavefront_path_tracer.h
// WavefrontPathTracer -- pbrt-v4-style queue-based GPU path tracer.
#pragma once

#include "path_tracing_strategy.h"
#include "wavefront_types.h"
#include "optix_types.h"
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

private:
    bool loadModule();
    void destroyProgramGroups();
    void destroySBT();
    bool allocateQueues(int numPixels);
    void freeQueues();
    void launchGenerateCameraRays(int width, int height, int sampleIdx,
        const GpuCameraParams& camera);
    void launchEvaluateMaterials(int numHits, int maxDepth,
        const SphereData* d_spheres, unsigned int numSpheres,
        const QuadData* d_quads, unsigned int numQuads,
        const TriangleData* d_triangles, unsigned int numTriangles,
        const BilinearPatchData* d_bilinearPatches, unsigned int numBilinearPatches,
        const MaterialData* d_materials, unsigned int numMaterials,
        const int* d_lightIndices, const GpuLightKind* d_lightKinds,
        const GpuAliasEntry* d_aliasTable, unsigned int numLights,
        const PunctualLightGPU* d_punctualLights, unsigned int numPunctualLights,
        float3* d_framebuffer, float3 skyColor, float shadowRayEpsilon, GpuSkyDistribution skyDist);
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
        const MaterialData* d_materials, unsigned int numMaterials,
        const int* d_lightIndices, const GpuLightKind* d_lightKinds,
        const GpuAliasEntry* d_aliasTable, unsigned int numLights,
        const PunctualLightGPU* d_punctualLights, unsigned int numPunctualLights,
        float3* d_framebuffer, float3 skyColor, float shadowRayEpsilon, GpuSkyDistribution skyDist);
    // Twin of launchEvaluateMaterialsSimple() above, scoped to
    // dielectricHitQueue's Dielectric/RoughDielectric hits (see
    // wavefront_types.h's WavefrontQueues::dielectricHitQueue and
    // wavefront_kernels.cu's evaluate_materials_dielectric()). No texture or
    // scene-medium/measured-BRDF params at all - neither material type ever
    // reads them.
    void launchEvaluateMaterialsDielectric(int numHits, int maxDepth,
        const SphereData* d_spheres, unsigned int numSpheres,
        const QuadData* d_quads, unsigned int numQuads,
        const TriangleData* d_triangles, unsigned int numTriangles,
        const BilinearPatchData* d_bilinearPatches, unsigned int numBilinearPatches,
        const MaterialData* d_materials, unsigned int numMaterials,
        const int* d_lightIndices, const GpuLightKind* d_lightKinds,
        const GpuAliasEntry* d_aliasTable, unsigned int numLights,
        const PunctualLightGPU* d_punctualLights, unsigned int numPunctualLights,
        float3* d_framebuffer, float3 skyColor, float shadowRayEpsilon, GpuSkyDistribution skyDist);
    // Not a launchX-style param above deliberately - see setTextures()'s
    // comment for why textures travel via member state instead.
    void launchAccumulateMiss(int numMiss, float3* d_framebuffer, float3 backgroundColor, GpuSkyDistribution skyDist);
    void launchAccumulateShadow(int numShadow, const bool* d_occluded, float3* d_framebuffer);
    void launchResolveBssrdfExit(int numExit,
        const MaterialData* d_materials, unsigned int numMaterials,
        const SphereData* d_spheres, unsigned int numSpheres,
        const QuadData* d_quads, unsigned int numQuads,
        const TriangleData* d_triangles, unsigned int numTriangles,
        const BilinearPatchData* d_bilinearPatches, unsigned int numBilinearPatches,
        const int* d_lightIndices, const GpuLightKind* d_lightKinds,
        const GpuAliasEntry* d_aliasTable, unsigned int numLights,
        const PunctualLightGPU* d_punctualLights, unsigned int numPunctualLights,
        float3* d_framebuffer, float3 skyColor, float shadowRayEpsilon, GpuSkyDistribution skyDist);
    void launchNormalizeFramebuffer(unsigned int numPixels, float invSPP, float3* d_framebuffer);
    int  readQueueSize(int* d_counter);
    void resetQueueCounter(int* d_counter);

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
    bool         haveInstancedTriangles_ = false;  ///< see setInstancedGeometryFlags()
    bool         haveInstancedSpheres_ = false;
    unsigned int numSpheres_  = 0;
    unsigned int numQuads_    = 0;
    unsigned int numBilinearPatches_ = 0;
    unsigned int numTriangles_ = 0;
    unsigned int numDisks_ = 0;
    unsigned int numCylinders_ = 0;
    unsigned int frameNumber_ = 0;
};

} // namespace optix_renderer
