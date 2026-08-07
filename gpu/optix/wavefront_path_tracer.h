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
    bool buildSBT(unsigned int numSpheres, unsigned int numQuads) override;
    bool render(int width, int height, int samples_per_pixel, int max_depth,
        const GpuCameraParams& camera,
        float* framebuffer, OptixTraversableHandle gas_handle,
        CUdeviceptr d_materials, CUdeviceptr d_spheres, CUdeviceptr d_quads,
        CUdeviceptr d_light_indices, CUdeviceptr d_is_light_sphere,
        CUdeviceptr d_alias_table, unsigned int num_materials,
        unsigned int num_spheres, unsigned int num_quads,
        unsigned int num_lights,
        CUdeviceptr d_punctual_lights = 0,
        unsigned int num_punctual_lights = 0) override;
    void cleanup() override;
    PathTracingMode getMode() const override { return PathTracingMode::WAVEFRONT; }
    const char* getName() const override { return "WavefrontPathTracer"; }
    void setPTXPath(const std::string& path) { ptxPath_ = path; }

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
        const int* d_lightIndices, const bool* d_isLightSphere,
        const GpuAliasEntry* d_aliasTable, unsigned int numLights,
        const PunctualLightGPU* d_punctualLights, unsigned int numPunctualLights,
        float3* d_framebuffer);
    void launchAccumulateMiss(int numMiss, float3* d_framebuffer);
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
    OptixProgramGroup raygenShadowPG_        = nullptr;
    OptixProgramGroup missShadowPG_          = nullptr;
    OptixProgramGroup anyhitShadowSpherePG_  = nullptr;
    OptixProgramGroup anyhitShadowQuadPG_    = nullptr;
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
    unsigned int numSpheres_  = 0;
    unsigned int numQuads_    = 0;
    unsigned int frameNumber_ = 0;
};

} // namespace optix_renderer
