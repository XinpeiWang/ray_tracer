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
#include "probe_grid_types.h"
#include "wavefront_guiding.h"
#include "wavefront_nrc_types.h"
#include "wavefront_upscale_types.h"
#include <cuda_runtime.h>

extern "C" void wf_launch_generate_camera_rays(
	WorkQueue<RayWorkItem> rq,
	int width, int height, int sampleIdx,
	GpuCameraParams camera,
	unsigned int frameNumber,
	float* d_weightBuffer,
	bool checkerboardActive,
	// Adaptive sampling - see generate_camera_rays's own activePixelMask
	// parameter comment (wavefront_kernels_camera.cu).
	const unsigned char* activePixelMask,
	// Live Preview's temporal upscale feature - see generate_camera_rays's
	// own temporalJitterEnabled parameter comment (wavefront_kernels_camera.cu).
	bool temporalJitterEnabled,
	unsigned int temporalJitterIndex,
	int temporalUpscaleFactor,
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
	GpuPortalLight               portalLight,
	bool                         regularize,
	float                        maxComponentValue,
	float3*                      d_albedoBuffer,
	float3*                      d_normalBuffer,
	float4*                      d_worldPosBuffer,
	// ReSTIR DI (Live Preview only) current-frame reservoir buffer, indexed
	// by pixelIndex - see wf_finish_material_scatter's own restirReservoirs
	// parameter comment (wavefront_device_helpers.h). nullptr for batch/
	// offline rendering, exactly like d_worldPosBuffer's own opt-in pattern.
	GpuReservoir*                d_restirReservoirs,
	// ReSTIR temporal reuse's history/reprojection context - see
	// wf_finish_material_scatter's own restirCtx parameter comment. Default-
	// constructed (historyValid=false) is a safe no-op, same as
	// d_restirReservoirs being null.
	GpuRestirTemporalContext     restirCtx,
	// ReSTIR GI (Live Preview only) - see wf_finish_material_scatter's own
	// giOriginContext/giCandidateOut parameter comments. nullptr for batch/
	// offline rendering, same opt-in pattern as d_restirReservoirs above.
	GpuGiOriginContext*          d_giOriginContext,
	GpuGiSample*                 d_giCandidateOut,
	// See wf_light_bvh_sample_index()'s own comment
	// (wavefront_restir_helpers.h). nodeCount<=0 means "no light BVH built".
	WfLightBvhContext            lightBvh,
	// Real-time path guiding (Live Preview only) - see evaluate_materials's
	// own guidingHistograms parameter comment (wavefront_kernels_materials.cu).
	// d_guidingHistograms==nullptr keeps every Conductor/RoughMetal case on
	// its existing, unmodified GGX/VNDF sampling.
	GpuProbeGridMeta             guidingGridMeta,
	const GpuGuidingHistogram*   d_guidingHistograms,
	// Same probe array wf_query_probe_grid() already leak-guards the SH-L1
	// diffuse cache with - see evaluate_materials's own guidingProbes
	// parameter comment.
	const GpuProbe*              d_guidingProbes,
	// Neural Radiance Cache (Live Preview only) - see evaluate_materials's
	// own nrcWeights parameter comment (wavefront_kernels_materials.cu).
	// nrcWeights==nullptr is a complete no-op.
	const float*                 nrcWeights,
	int                          nrcTrainingSteps,
	float3                       nrcAabbMin,
	float3                       nrcAabbExtent,
	// ReSTIR for volumetric/participating media (Live Preview only) - see
	// wf_finish_material_scatter's own restirVolumeReservoirs/restirVolumeCtx/
	// volumeMatIdxOut/volumeEntryPointOut parameter comments. nullptr for
	// batch/offline rendering, same opt-in pattern as d_restirReservoirs
	// above. Only this launcher's own kernel (evaluate_materials) ever
	// reaches a phase-scatter vertex - see this project's own plan.
	GpuVolumeReservoir*          d_restirVolumeReservoirs,
	GpuVolumeRestirTemporalContext restirVolumeCtx,
	int*                         d_volumeMatIdxOut,
	float4*                      d_volumePhaseWoGOut,
	float4*                      d_volumeEntryPointOut,
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
	GpuPortalLight               portalLight,
	float                        maxComponentValue,
	float3*                      d_albedoBuffer,
	float3*                      d_normalBuffer,
	float4*                      d_worldPosBuffer,
	// ReSTIR DI (Live Preview only) current-frame reservoir buffer, indexed
	// by pixelIndex - see wf_finish_material_scatter's own restirReservoirs
	// parameter comment (wavefront_device_helpers.h). nullptr for batch/
	// offline rendering, exactly like d_worldPosBuffer's own opt-in pattern.
	GpuReservoir*                d_restirReservoirs,
	// ReSTIR temporal reuse's history/reprojection context - see
	// wf_finish_material_scatter's own restirCtx parameter comment. Default-
	// constructed (historyValid=false) is a safe no-op, same as
	// d_restirReservoirs being null.
	GpuRestirTemporalContext     restirCtx,
	// ReSTIR GI (Live Preview only) - see wf_finish_material_scatter's own
	// giOriginContext/giCandidateOut parameter comments. nullptr for batch/
	// offline rendering, same opt-in pattern as d_restirReservoirs above.
	GpuGiOriginContext*          d_giOriginContext,
	GpuGiSample*                 d_giCandidateOut,
	// See wf_light_bvh_sample_index()'s own comment
	// (wavefront_restir_helpers.h). nodeCount<=0 means "no light BVH built".
	WfLightBvhContext            lightBvh,
	// World-space irradiance probe cache (Live Preview only) - see
	// wf_finish_material_scatter's own probeGridMeta/probeGrid parameter
	// comments (wavefront_device_helpers.h). d_probeGrid==nullptr (default,
	// every non-Live-Preview call site) is a complete no-op.
	GpuProbeGridMeta             probeGridMeta,
	const GpuProbe*              d_probeGrid,
	// Neural Radiance Cache (Live Preview only) - see evaluate_materials_
	// simple's own nrcWeights parameter comment (wavefront_kernels_
	// materials_simple.cu). nrcWeights==nullptr is a complete no-op.
	const float*                 nrcWeights,
	int                          nrcTrainingSteps,
	float3                       nrcAabbMin,
	float3                       nrcAabbExtent,
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
	GpuPortalLight               portalLight,
	bool                         regularize,
	float                        maxComponentValue,
	float3*                      d_albedoBuffer,
	float3*                      d_normalBuffer,
	float4*                      d_worldPosBuffer,
	// ReSTIR DI (Live Preview only) current-frame reservoir buffer, indexed
	// by pixelIndex - see wf_finish_material_scatter's own restirReservoirs
	// parameter comment (wavefront_device_helpers.h). nullptr for batch/
	// offline rendering, exactly like d_worldPosBuffer's own opt-in pattern.
	GpuReservoir*                d_restirReservoirs,
	// ReSTIR temporal reuse's history/reprojection context - see
	// wf_finish_material_scatter's own restirCtx parameter comment. Default-
	// constructed (historyValid=false) is a safe no-op, same as
	// d_restirReservoirs being null.
	GpuRestirTemporalContext     restirCtx,
	// ReSTIR GI (Live Preview only) - see wf_finish_material_scatter's own
	// giOriginContext/giCandidateOut parameter comments. nullptr for batch/
	// offline rendering, same opt-in pattern as d_restirReservoirs above.
	GpuGiOriginContext*          d_giOriginContext,
	GpuGiSample*                 d_giCandidateOut,
	// See wf_light_bvh_sample_index()'s own comment (wavefront_restir_
	// helpers.h). nodeCount<=0 means "no light BVH built" - previously
	// omitted entirely from this kernel, silently keeping Dielectric/
	// RoughDielectric NEE on the alias table only.
	WfLightBvhContext            lightBvh,
	cudaStream_t                 stream);

// ReSTIR DI spatial reuse - see wavefront_kernels_restir.cu's own header
// comment for why this is a separate, pixel-indexed kernel rather than
// folded into wf_finish_material_scatter like temporal reuse is.
extern "C" void wf_launch_restir_spatial_reuse(
	const GpuReservoir* d_currentReservoirs,
	const float3*       d_currentNormals,
	const float4*       d_currentWorldPos,
	GpuReservoir*       d_outputReservoirs,
	int width, int height,
	unsigned int frameSeed,
	// Every primIdx a stored GpuLightSample carries was already validated
	// against these SAME arrays at candidate-generation time (evaluate_
	// materials*) - no separate count needed here, just the pointers, unlike
	// the alias-table-drawing launches above which draw a FRESH index.
	const SphereData*   d_spheres,
	const QuadData*     d_quads,
	const TriangleData* d_triangles,
	const BilinearPatchData* d_bilinearPatches,
	const DiskData*     d_disks,
	const CylinderData* d_cylinders,
	const MaterialData* d_materials,
	const TextureData*  d_textures,
	const unsigned char* d_texturePixels,
	cudaStream_t stream);

// See wavefront_kernels_restir.cu's own restir_clear_reservoirs comment.
extern "C" void wf_launch_restir_clear_reservoirs(
	GpuReservoir* d_reservoirs, int width, int height,
	unsigned int frameNumber, bool checkerboardActive, cudaStream_t stream);

// ReSTIR for volumetric/participating media's own spatial reuse - see
// wavefront_kernels_restir.cu's own restir_volume_spatial_reuse comment and
// this project's own plan. Unlike restir_clear_reservoirs above, there is no
// wf_launch_restir_clear_volume_reservoirs: d_currentReservoirs/
// d_currentMatIdx are deliberately NEVER unconditionally cleared every
// render() call - see WavefrontPathTracer::render()'s own volume-reservoir
// allocation comment for why holding them across a medium's "no scatter this
// frame" pass-through sub-case is the whole point of this feature.
extern "C" void wf_launch_restir_volume_spatial_reuse(
	const GpuVolumeReservoir* d_currentReservoirs,
	const int*                d_currentMatIdx,
	const float4*             d_currentPhaseWoG,
	// This pixel's own held-over medium entry point (xyz) + mean free path
	// (w), packed into one float4 (d_volumeEntryPoint_) - NOT d_worldPos_/
	// d_currentWorldPos, which is refreshed every frame for EVERY depth==0
	// hit regardless of medium status; using it here would pair this frame's
	// fresh (possibly unrelated) hit point with the other 2 sticky buffers'
	// held-over values - see wf_finish_material_scatter's own
	// volumeEntryPointOut parameter comment (wavefront_device_helpers.h).
	const float4*             d_currentEntryPoint,
	GpuVolumeReservoir*       d_outputReservoirs,
	int width, int height,
	unsigned int frameSeed,
	const SphereData*   d_spheres,
	const QuadData*     d_quads,
	const TriangleData* d_triangles,
	const BilinearPatchData* d_bilinearPatches,
	const DiskData*     d_disks,
	const CylinderData* d_cylinders,
	const MaterialData* d_materials,
	const TextureData*  d_textures,
	const unsigned char* d_texturePixels,
	cudaStream_t stream);

// ReSTIR GI (see wavefront_kernels_restir.cu's own restir_gi_finalize/
// restir_gi_spatial_reuse header comments). Finalize runs once per SAMPLE
// (not once per render() call like DI's spatial reuse) - see that kernel's
// own comment for why.
extern "C" void wf_launch_restir_gi_finalize(
	const GpuGiOriginContext* d_originContext,
	const GpuGiSample*        d_candidateIn,
	const GpuGiReservoir*     d_history,
	const float4*             d_worldPosHistory,
	const float4*             d_currentWorldPos,
	const float*              d_weightBuffer,
	GpuReprojectBasis         prevCamera,
	bool                      historyValid,
	int width, int height,
	unsigned int frameSeed,
	const MaterialData*       d_materials,
	float3*                   d_framebuffer,
	float                     maxComponentValue,
	GpuGiReservoir*           d_outputReservoirs,
	cudaStream_t stream);

extern "C" void wf_launch_restir_gi_spatial_reuse(
	const GpuGiReservoir*     d_currentReservoirs,
	const GpuGiOriginContext* d_originContext,
	GpuGiReservoir*           d_outputReservoirs,
	int width, int height,
	unsigned int frameSeed,
	cudaStream_t stream);

// World-space irradiance probe cache (Live Preview only, gpu/optix/
// probe_grid_types.h and wavefront_kernels_restir.cu's own probe_cache_shade/
// probe_cache_accumulate comments) - the plain-CUDA half of the update pass,
// run once per render() call after __raygen__wf_probe_cache's own OptiX
// intersection launch (WavefrontPathTracer::launchProbeCacheUpdate()).
extern "C" void wf_launch_probe_cache_shade(
	WorkQueue<ProbeCacheHitWorkItem> hq,
	int                          numProbeCacheHits,
	const MaterialData*          d_materials,
	const SphereData*            d_spheres,
	const QuadData*              d_quads,
	const TriangleData*          d_triangles,
	const BilinearPatchData*     d_bilinearPatches,
	const DiskData*              d_disks,
	const CylinderData*          d_cylinders,
	const TextureData*           d_textures,
	const unsigned char*         d_texturePixels,
	const int*                   d_lightIndices,
	const GpuLightKind*          d_lightKinds,
	const GpuAliasEntry*         d_aliasTable,
	unsigned int                 numLights,
	WfLightBvhContext            lightBvh,
	float3                       backgroundColor,
	float                        shadowRayEpsilon,
	float3*                      d_probeCacheRadianceOut,
	float*                       d_probeCacheHitDistOut,
	WorkQueue<ShadowRayWorkItem> shadowQueue,
	cudaStream_t                 stream);

extern "C" void wf_launch_probe_cache_accumulate(
	const float3*    d_probeCacheRadianceOut,
	const float*     d_probeCacheHitDistOut,
	const float3*    d_probeCacheDirections,
	int              numBatchSlots,
	int              probeUpdateCursor,
	GpuProbeGridMeta gridMeta,
	GpuProbe*        d_probes,
	// Real-time path guiding (Live Preview only) - see probe_cache_accumulate's
	// own guidingHistograms parameter comment (wavefront_kernels_restir.cu).
	// nullptr (guiding disabled) keeps this a complete no-op.
	GpuGuidingHistogram* d_guidingHistograms,
	cudaStream_t     stream);

// SVGF (Live Preview only, gpu/optix/wavefront_svgf_math.h and
// wavefront_kernels_svgf.cu) - see that file's own header comment for the
// full 4-kernel pipeline these wrap.
extern "C" void wf_launch_svgf_temporal_integrate(
	const float3*        d_currentRadiance,
	const float4*        d_currentWorldPos,
	const GpuSvgfState*  d_history,
	const float4*        d_worldPosHistory,
	const float*         d_weightBuffer,
	GpuReprojectBasis    prevCamera,
	bool                 historyValid,
	int width, int height,
	float temporalAlpha, float maxHistoryLength,
	GpuSvgfState*        d_outputCurrent,
	cudaStream_t stream);

// Checkerboard temporal upsampling's frame-clear (svgf_checkerboard_clear_
// frame, wavefront_kernels_svgf.cu) - replaces the plain memset render()
// used to run on albedoAov/normalAov/worldPos, see that kernel's own comment.
extern "C" void wf_launch_svgf_checkerboard_clear_frame(
	float3* d_albedo, float3* d_normal, float4* d_worldPos,
	int width, int height, unsigned int frameNumber, bool checkerboardActive,
	cudaStream_t stream);

extern "C" void wf_launch_svgf_prepare_for_filter(
	const GpuSvgfState* d_current,
	const float3*       d_albedo,
	const float4*       d_currentWorldPos,
	int width, int height,
	float varianceBootstrapFrames, int varianceBootstrapRadius, float minAlbedo,
	float4*             d_outPingPong0,
	cudaStream_t stream);

extern "C" void wf_launch_svgf_atrous_pass(
	const float4* d_input,
	const float4* d_currentWorldPos,
	const float3* d_normals,
	float3        cameraOrigin,
	int width, int height,
	int stepSize,
	float sigmaNormal, float sigmaDepth, float sigmaLuminance, int atrousRadius,
	float4*       d_output,
	cudaStream_t stream);

extern "C" void wf_launch_svgf_finalize(
	const float4* d_filtered,
	const float3* d_albedo,
	int numPixels,
	float minAlbedo,
	float3*       d_framebuffer,
	cudaStream_t stream);

extern "C" void wf_launch_accumulate_miss(
	WorkQueue<MissWorkItem> mq, int numMiss,
	float3* d_framebuffer, float3 backgroundColor, GpuSkyDistribution skyDist,
	GpuPortalLight portalLight, float maxComponentValue,
	float3* d_albedoBuffer, float3* d_normalBuffer, float4* d_worldPosBuffer, cudaStream_t stream);

extern "C" void wf_launch_accumulate_shadow(
	WorkQueue<ShadowRayWorkItem> sq, int numShadow,
	const float* d_transmittance, float3* d_framebuffer, float maxComponentValue, cudaStream_t stream,
	// ReSTIR GI (Live Preview only) - see accumulate_shadow's own comment
	// (wavefront_kernels_accumulate.cu). nullptr (every existing call site)
	// keeps every shadow ray going to d_framebuffer exactly as before.
	GpuGiSample* d_giCandidateOut = nullptr,
	// Probe cache (Live Preview only) - see accumulate_shadow's own
	// probeCacheRadianceOut parameter comment. nullptr (every non-probe-cache
	// call site) keeps every shadow ray going to d_framebuffer as before.
	float3* d_probeCacheRadianceOut = nullptr,
	// Neural Radiance Cache training pipeline (Live Preview only) - see
	// accumulate_shadow's own nrcTrainingRecords parameter comment. nullptr
	// (every non-NRC call site) keeps every shadow ray going to
	// d_framebuffer as before.
	NrcTrainingRecord* d_nrcTrainingRecords = nullptr);

// Neural Radiance Cache training pipeline (Live Preview only, gpu/optix/
// wavefront_kernels_nrc.cu) - see that file's own header comment for the
// full Phase A/B design these wrap.
extern "C" void wf_launch_nrc_generate_training_rays(
	WorkQueue<RayWorkItem> rq, int numPaths, GpuCameraParams camera, unsigned int frameNumber,
	cudaStream_t stream);

extern "C" void wf_launch_nrc_training_shade_simple(
	WorkQueue<HitWorkItem> simpleHitQueue, int numHits, int depth, int maxDepth,
	const MaterialData* d_materials,
	const SphereData* d_spheres, const QuadData* d_quads, const TriangleData* d_triangles,
	const BilinearPatchData* d_bilinearPatches, const DiskData* d_disks, const CylinderData* d_cylinders,
	const TextureData* d_textures, const unsigned char* d_texturePixels,
	const int* d_lightIndices, const GpuLightKind* d_lightKinds, const GpuAliasEntry* d_aliasTable, unsigned int numLights,
	WfLightBvhContext lightBvh, float shadowRayEpsilon,
	NrcTrainingRecord* d_records, int* d_validRecordCounter,
	WorkQueue<RayWorkItem> nextRayQueue, WorkQueue<ShadowRayWorkItem> shadowQueue,
	cudaStream_t stream);

extern "C" void wf_launch_nrc_training_shade_full(
	WorkQueue<HitWorkItem> hitQueue, int numHits, int depth, int maxDepth,
	const MaterialData* d_materials,
	const SphereData* d_spheres, const QuadData* d_quads, const TriangleData* d_triangles,
	const BilinearPatchData* d_bilinearPatches, const DiskData* d_disks, const CylinderData* d_cylinders,
	const TextureData* d_textures, const unsigned char* d_texturePixels,
	const int* d_lightIndices, const GpuLightKind* d_lightKinds, const GpuAliasEntry* d_aliasTable, unsigned int numLights,
	WfLightBvhContext lightBvh, float shadowRayEpsilon,
	NrcTrainingRecord* d_records, int* d_validRecordCounter,
	WorkQueue<RayWorkItem> nextRayQueue, WorkQueue<ShadowRayWorkItem> shadowQueue,
	cudaStream_t stream);

extern "C" void wf_launch_nrc_bootstrap_and_train(
	const NrcTrainingRecord* d_records, int numRecords,
	const float* d_weights, float* d_gradAccum,
	float3 aabbMin, float3 aabbExtent,
	cudaStream_t stream);

extern "C" void wf_launch_nrc_apply_gradients(
	float* d_weights, float* d_adamM, float* d_adamV, float* d_gradAccum,
	int numContributingRecords, int stepCount,
	cudaStream_t stream);

extern "C" void wf_launch_nrc_reset_weights(
	float* d_weights, float* d_adamM, float* d_adamV, unsigned int seed,
	cudaStream_t stream);

// Neural temporal upscale (Live Preview only, gpu/optix/wavefront_kernels_upscale.cu) -
// see this project's own plan. Mirrors the NRC launcher family just above
// in shape (train/apply-gradients/reset-weights), plus its own motion-
// vector and inference launchers.
extern "C" void wf_launch_upscale_compute_motion_vectors(
	const float4* d_worldPos, float2* d_outMotionVectors,
	int width, int height, GpuReprojectBasis prevCamera,
	cudaStream_t stream);

extern "C" void wf_launch_upscale_train(
	const float3* d_lowResFramebuffer, int width, int height,
	const UpscaleForwardCache* d_forwardCache, int upscaleFactor, unsigned int dueCx, unsigned int dueCy,
	const float* d_weights, float* d_gradAccum, int* d_validRecordCounter,
	unsigned int frameSeed, int numSamples,
	cudaStream_t stream);

extern "C" void wf_launch_upscale_apply_gradients(
	float* d_weights, float* d_adamM, float* d_adamV, float* d_gradAccum,
	int numContributingRecords, int stepCount,
	cudaStream_t stream);

extern "C" void wf_launch_upscale_reset_weights(
	float* d_weights, float* d_adamM, float* d_adamV, unsigned int seed,
	cudaStream_t stream);

extern "C" void wf_launch_upscale_infer(
	const float3* d_lowResFramebuffer, const float4* d_worldPos, const float4* d_worldPosHistory,
	const float4* d_history, const float2* d_motionVectors,
	int width, int height, int upscaleFactor,
	unsigned int dueCx, unsigned int dueCy,
	GpuReprojectBasis prevCamera, bool historyValid, float3 cameraOrigin,
	const float* d_weights,
	float3* d_outColor, float4* d_outHistory, UpscaleForwardCache* d_outCache,
	cudaStream_t stream);

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
	GpuPortalLight               portalLight,
	float                        maxComponentValue,
	// ReSTIR GI (Live Preview only) - see wf_finish_material_scatter's own
	// giOriginContext/giCandidateOut parameter comments. A BSSRDF exit can
	// land at depth==1 for a pixel whose primary hit (x0) was GI-eligible
	// (see BssrdfExitWorkItem's own depth-propagation comment) - threading
	// these through lets that exit's own NEE contribute to the GI candidate
	// exactly like any other depth==1 material would, instead of silently
	// contributing nothing. nullptr (every batch/offline call site) keeps
	// this a complete no-op, same opt-in pattern as every other ReSTIR
	// parameter here.
	GpuGiOriginContext*          d_giOriginContext,
	GpuGiSample*                 d_giCandidateOut,
	// See wf_light_bvh_sample_index()'s own comment (wavefront_restir_
	// helpers.h). nodeCount<=0 means "no light BVH built" - previously
	// omitted entirely from this kernel, silently keeping a BSSRDF exit's
	// own NEE on the alias table only.
	WfLightBvhContext            lightBvh,
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
