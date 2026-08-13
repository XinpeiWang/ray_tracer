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
    bool buildSBT(unsigned int numSpheres, unsigned int numQuads, unsigned int numBilinearPatches = 0, unsigned int numTriangles = 0) override;
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
        unsigned int num_triangles = 0) override;
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
        const MaterialData* d_materials, unsigned int numMaterials,
        const int* d_lightIndices, const GpuLightKind* d_lightKinds,
        const GpuAliasEntry* d_aliasTable, unsigned int numLights,
        const PunctualLightGPU* d_punctualLights, unsigned int numPunctualLights,
        float3* d_framebuffer);
    // Not a launchX-style param above deliberately - see setTextures()'s
    // comment for why textures travel via member state instead.
    void launchAccumulateMiss(int numMiss, float3* d_framebuffer, float3 backgroundColor);
    void launchAccumulateShadow(int numShadow, const bool* d_occluded, float3* d_framebuffer);
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
    OptixProgramGroup hitTrianglePG_     = nullptr;
    OptixProgramGroup raygenShadowPG_        = nullptr;
    OptixProgramGroup missShadowPG_          = nullptr;
    OptixProgramGroup anyhitShadowSpherePG_  = nullptr;
    OptixProgramGroup anyhitShadowQuadPG_    = nullptr;
    OptixProgramGroup anyhitShadowBilinearPatchPG_ = nullptr;
    OptixProgramGroup anyhitShadowTrianglePG_ = nullptr;
    OptixProgramGroup exceptionPG_ = nullptr;  ///< CUDA-718 fix -- see initialize()'s exceptionFlags comment
    OptixPipeline intersectPipeline_ = nullptr;
    OptixPipeline shadowPipeline_    = nullptr;
    OptixShaderBindingTable intersectSBT_ = {};
    OptixShaderBindingTable shadowSBT_    = {};
    CUdeviceptr d_intersectRaygenRecord_ = 0;
    CUdeviceptr d_intersectMissRecord_   = 0;
    CUdeviceptr d_intersectHitRecords_   = 0;
    CUdeviceptr d_shadowRaygenRecord_    = 0;
    CUdeviceptr d_shadowMissRecord_      = 0;
    CUdeviceptr d_shadowHitRecords_      = 0;
    CUdeviceptr d_intersectExceptionRecord_ = 0;  ///< see exceptionPG_
    CUdeviceptr d_shadowExceptionRecord_    = 0;
    CUdeviceptr d_wfLaunchParams_   = 0;
    CUdeviceptr d_rayItems_         = 0;
    CUdeviceptr d_nextRayItems_     = 0;
    CUdeviceptr d_hitItems_         = 0;
    CUdeviceptr d_missItems_        = 0;
    CUdeviceptr d_shadowItems_      = 0;
    CUdeviceptr d_occluded_         = 0;
    CUdeviceptr d_rayCounter_       = 0;
    CUdeviceptr d_nextRayCounter_   = 0;
    CUdeviceptr d_hitCounter_       = 0;
    CUdeviceptr d_missCounter_      = 0;
    CUdeviceptr d_shadowCounter_    = 0;
    int          queueCapacity_ = 0;
    std::string  ptxPath_;
    CUdeviceptr  d_instancePrimBase_ = 0;   ///< see setInstancePrimBase()
    CUdeviceptr  d_textures_ = 0;           ///< see setTextures()
    CUdeviceptr  d_texturePixels_ = 0;
    bool         haveInstancedTriangles_ = false;  ///< see setInstancedGeometryFlags()
    bool         haveInstancedSpheres_ = false;
    unsigned int numSpheres_  = 0;
    unsigned int numQuads_    = 0;
    unsigned int numBilinearPatches_ = 0;
    unsigned int numTriangles_ = 0;
    unsigned int frameNumber_ = 0;
};

} // namespace optix_renderer
