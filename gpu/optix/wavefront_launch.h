#pragma once

/// @file wavefront_launch.h
/// @brief Shared extern "C" declarations for wavefront_launch.cu's plain-C
///        launcher wrappers (the wf_launch_*/wf_upload_*/wf_reset_queue_counter
///        functions), included by both wavefront_launch.cu (whose definitions
///        this then type-checks against) and wavefront_path_tracer.cpp (the
///        sole caller). Previously each function's signature was hand-typed
///        twice, once per file - extern "C" linkage means these are NOT
///        type-checked across translation units, so a signature edit in one
///        copy without the other would compile cleanly and misbehave (or
///        crash) at runtime instead of failing to link. One shared
///        declaration removes that risk entirely.

#include "wavefront_types.h"
#include "optix_types.h"
#include <cuda_runtime.h>

extern "C" void wf_launch_generate_camera_rays(
	WorkQueue<RayWorkItem> rq,
	int width, int height, int sampleIdx,
	GpuCameraParams camera,
	unsigned int frameNumber,
	float* d_weightBuffer,
	cudaStream_t stream);

extern "C" void wf_launch_evaluate_materials(
	WorkQueue<HitWorkItem>       hq,
	int                          numHits,
	WorkQueue<RayWorkItem>       nextRayQueue,
	WorkQueue<ShadowRayWorkItem> shadowQueue,
	WorkQueue<BssrdfProbeWorkItem> bssrdfProbeQueue,
	float3*                      d_framebuffer,
	const SphereData*            d_spheres,   unsigned int numSpheres,
	const QuadData*              d_quads,     unsigned int numQuads,
	const TriangleData*          d_triangles, unsigned int numTriangles,
	const BilinearPatchData*     d_bilinearPatches, unsigned int numBilinearPatches,
	const DiskData*              d_disks,     unsigned int numDisks,
	const CylinderData*          d_cylinders, unsigned int numCylinders,
	const MaterialData*          d_materials, unsigned int numMaterials,
	const int*                   d_lightIndices,
	const GpuLightKind*          d_lightKinds,
	const GpuAliasEntry*         d_aliasTable,
	unsigned int                 numLights,
	const PunctualLightGPU*      d_punctualLights,
	unsigned int                 numPunctualLights,
	const TextureData*           d_textures,
	const unsigned char*         d_texturePixels,
	int                          maxDepth,
	const CloudMedium<float>*    d_cloudMediums, unsigned int numCloudMediums,
	const GpuRgbGridMedium*      d_rgbGridMediums,
	const float*                 d_rgbGridData,
	const GpuGridMedium*         d_gridMediums,
	const float*                 d_gridData,
	const GpuMeasuredTable*      d_measuredTables, unsigned int numMeasuredTables,
	const float*                 d_measuredParamValues,
	const float*                 d_measuredData,
	const float*                 d_measuredMcdf,
	const float*                 d_measuredCcdf,
	float3                       skyColor,
	float                        shadowRayEpsilon,
	GpuSkyDistribution           skyDist,
	bool                         regularize,
	float3*                      d_albedoBuffer,
	float3*                      d_normalBuffer,
	cudaStream_t                 stream);

extern "C" void wf_launch_evaluate_materials_simple(
	WorkQueue<HitWorkItem>       hq,
	int                          numHits,
	WorkQueue<RayWorkItem>       nextRayQueue,
	WorkQueue<ShadowRayWorkItem> shadowQueue,
	float3*                      d_framebuffer,
	const SphereData*            d_spheres,   unsigned int numSpheres,
	const QuadData*              d_quads,     unsigned int numQuads,
	const TriangleData*          d_triangles, unsigned int numTriangles,
	const BilinearPatchData*     d_bilinearPatches, unsigned int numBilinearPatches,
	const DiskData*              d_disks,     unsigned int numDisks,
	const CylinderData*          d_cylinders, unsigned int numCylinders,
	const MaterialData*          d_materials, unsigned int numMaterials,
	const int*                   d_lightIndices,
	const GpuLightKind*          d_lightKinds,
	const GpuAliasEntry*         d_aliasTable,
	unsigned int                 numLights,
	const PunctualLightGPU*      d_punctualLights,
	unsigned int                 numPunctualLights,
	const TextureData*           d_textures,
	const unsigned char*         d_texturePixels,
	int                          maxDepth,
	float3                       skyColor,
	float                        shadowRayEpsilon,
	GpuSkyDistribution           skyDist,
	float3*                      d_albedoBuffer,
	float3*                      d_normalBuffer,
	cudaStream_t                 stream);

extern "C" void wf_launch_evaluate_materials_dielectric(
	WorkQueue<HitWorkItem>       hq,
	int                          numHits,
	WorkQueue<RayWorkItem>       nextRayQueue,
	WorkQueue<ShadowRayWorkItem> shadowQueue,
	float3*                      d_framebuffer,
	const SphereData*            d_spheres,   unsigned int numSpheres,
	const QuadData*              d_quads,     unsigned int numQuads,
	const TriangleData*          d_triangles, unsigned int numTriangles,
	const BilinearPatchData*     d_bilinearPatches, unsigned int numBilinearPatches,
	const DiskData*              d_disks,     unsigned int numDisks,
	const CylinderData*          d_cylinders, unsigned int numCylinders,
	const MaterialData*          d_materials, unsigned int numMaterials,
	const int*                   d_lightIndices,
	const GpuLightKind*          d_lightKinds,
	const GpuAliasEntry*         d_aliasTable,
	unsigned int                 numLights,
	const PunctualLightGPU*      d_punctualLights,
	unsigned int                 numPunctualLights,
	const TextureData*           d_textures,
	const unsigned char*         d_texturePixels,
	int                          maxDepth,
	float3                       skyColor,
	float                        shadowRayEpsilon,
	GpuSkyDistribution           skyDist,
	bool                         regularize,
	float3*                      d_albedoBuffer,
	float3*                      d_normalBuffer,
	cudaStream_t                 stream);

extern "C" void wf_launch_accumulate_miss(
	WorkQueue<MissWorkItem> mq, int numMiss,
	float3* d_framebuffer, float3 backgroundColor, GpuSkyDistribution skyDist,
	float3* d_albedoBuffer, float3* d_normalBuffer, cudaStream_t stream);

extern "C" void wf_launch_accumulate_shadow(
	WorkQueue<ShadowRayWorkItem> sq, int numShadow,
	const bool* d_occluded, float3* d_framebuffer, cudaStream_t stream);

extern "C" void wf_launch_resolve_bssrdf_exit(
	WorkQueue<BssrdfExitWorkItem> eq,
	int                          numExit,
	WorkQueue<RayWorkItem>       nextRayQueue,
	WorkQueue<ShadowRayWorkItem> shadowQueue,
	float3*                      d_framebuffer,
	const SphereData*            d_spheres,   unsigned int numSpheres,
	const QuadData*              d_quads,     unsigned int numQuads,
	const TriangleData*          d_triangles, unsigned int numTriangles,
	const BilinearPatchData*     d_bilinearPatches, unsigned int numBilinearPatches,
	const DiskData*              d_disks,     unsigned int numDisks,
	const CylinderData*          d_cylinders, unsigned int numCylinders,
	const MaterialData*          d_materials, unsigned int numMaterials,
	const int*                   d_lightIndices,
	const GpuLightKind*          d_lightKinds,
	const GpuAliasEntry*         d_aliasTable,
	unsigned int                 numLights,
	const PunctualLightGPU*      d_punctualLights,
	unsigned int                 numPunctualLights,
	const TextureData*           d_textures,
	const unsigned char*         d_texturePixels,
	float3                       skyColor,
	float                        shadowRayEpsilon,
	GpuSkyDistribution           skyDist,
	cudaStream_t                 stream);

extern "C" void wf_launch_normalize_framebuffer(
	unsigned int numPixels, const float* d_weightBuffer, float3* d_framebuffer, cudaStream_t stream);

extern "C" void wf_launch_normalize_aov_buffers(
	unsigned int numPixels, float3* d_albedoBuffer, float3* d_normalBuffer,
	unsigned int samplesPerPixel, cudaStream_t stream);

extern "C" void wf_reset_queue_counter(int* d_counter, cudaStream_t stream);

extern "C" void wf_upload_cie_tables(
	const float* h_cie_x, const float* h_cie_y, const float* h_cie_z, int n_samples);

// h_d65: caller-normalised D65 SPD, resampled onto the same 360-830nm/1nm
// grid as d_cie_x/y/z (see wf_upload_srgb_table's own uses in
// wavefront_path_tracer.cpp for how the resulting table is consumed).
extern "C" void wf_upload_d65_table(const float* h_d65, int n_samples);

extern "C" void wf_upload_srgb_table(
	const float* h_zNodes,    // [64]
	const float* h_coeffs,    // [3*64*64*64*3]
	int n_res);
