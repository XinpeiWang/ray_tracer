// wavefront_path_tracer_launch.cpp - WavefrontPathTracer: the per-frame helpers render() calls - queues, kernel launches (materials, ReSTIR, GI, probe cache,
// NRC, neural upscale, SVGF, accumulation) and the denoiser. A pure split of wavefront_path_tracer.cpp; nothing here changed.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "wavefront_path_tracer.h"
#include "optix_types.h"
#include "../../src/data/cie_data.h"
#include "rgb_to_spectrum_table.h"
#include <optix_stack_size.h>
#include "optix_module_parallel.h"
#include <cuda.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <random>  // World-space irradiance probe cache - see launchProbeCacheUpdate()'s own comment

// Declarations for wavefront_launch.cu's C wrappers (no <<<>>> in .cpp) -
// shared with wavefront_launch.cu itself via wavefront_launch.h, rather than
// hand-typed separately in each file (extern "C" linkage isn't type-checked
// across translation units, so two independently-typed copies could drift
// out of sync silently).
#include "wavefront_launch.h"
#include "wavefront_temporal_upscale_math.h"

namespace optix_renderer {

// ============================================================================
// allocateQueues
// ============================================================================

bool WavefrontPathTracer::allocateQueues(int numPixels, int numPunctualLights) {
	// Shadow queue needs more headroom than every other per-bounce queue
	// here: up to ONE shadow ray per pixel for a stochastically-picked area
	// light, ONE more for the sky/infinite light, and one MORE PER
	// PUNCTUAL/DELTA LIGHT (wf_push_nee_shadow_ray's own call sites in
	// wavefront_device_helpers.h - the punctual-light loop pushes one
	// unconditionally per light, no stochastic selection, since delta
	// lights have zero probability of being hit by BSDF sampling). A scene
	// with N simultaneous punctual lights can therefore need up to (2+N)
	// shadow-ray slots for a single bounce's worth of hits, not 1 - sizing
	// this queue the same numPixels-per-bounce way every OTHER queue here
	// is (ray/hit/miss/probe/exit, each genuinely bounded at one live item
	// per pixel) silently overflowed WorkQueue::push() (returns -1 on a full
	// queue, dropping that shadow ray's contribution entirely, not
	// corrupting memory - see WorkQueue::push()'s own comment) for any scene
	// with 2+ simultaneous punctual lights, systematically under-lighting
	// it. Confirmed empirically: pbrt_scenes/punctual-lights.pbrt (5
	// punctual lights, scene C8) rendered 40-46% too dark on GPU-wavefront
	// specifically (CPU and GPU-recursive, which push each NEE shadow ray
	// through OptiX's own `optixTrace()` immediately rather than queuing it
	// for a later kernel pass, have no equivalent capacity to overflow).
	const int maxShadowRaysPerHit = 2 + std::max(0, numPunctualLights);
	const int shadowCapacity = numPixels * maxShadowRaysPerHit;
	if (queueCapacity_ == numPixels && shadowQueueCapacity_ == shadowCapacity) return true;  // already allocated

	freeQueues();
	queueCapacity_ = numPixels;
	shadowQueueCapacity_ = shadowCapacity;
	size_t rayItemSz    = numPixels * sizeof(RayWorkItem);
	size_t hitItemSz    = numPixels * sizeof(HitWorkItem);
	size_t missItemSz   = numPixels * sizeof(MissWorkItem);
	size_t shadowItemSz = static_cast<size_t>(shadowCapacity) * sizeof(ShadowRayWorkItem);
	// d_transmittance_ is indexed by the SAME shadow-queue slot index as
	// d_shadowItems_ (written by the shadow-ray-tracing kernel, read back by
	// accumulate_shadow - see that kernel's own "Must also guard against
	// shadowQueue.capacity" comment, wavefront_kernels_accumulate.cu), so it
	// needs the same shadowCapacity sizing, not numPixels - sizing it to
	// numPixels here was the one spot this fix originally missed, confirmed
	// by a real out-of-bounds-read artifact (colorful noise in the upper
	// portion of the frame, near scene C8's own overhead lights) once the
	// shadow queue itself could legitimately hold more than numPixels items.
	size_t transmittanceSz = static_cast<size_t>(shadowCapacity) * sizeof(float);
	// Worst case every hit this bounce is a Subsurface transmission -
	// same numPixels capacity class as every other per-bounce queue.
	size_t probeItemSz  = numPixels * sizeof(BssrdfProbeWorkItem);
	size_t exitItemSz   = numPixels * sizeof(BssrdfExitWorkItem);
	size_t counterSz    = sizeof(int);

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rayItems_),     rayItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nextRayItems_), rayItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitItems_),     hitItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_simpleHitItems_), hitItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dielectricHitItems_), hitItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_missItems_),    missItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_shadowItems_),  shadowItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_transmittance_), transmittanceSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probeItems_),   probeItemSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_exitItems_),    exitItemSz));

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rayCounter_),     counterSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nextRayCounter_), counterSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitCounter_),     counterSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_simpleHitCounter_), counterSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dielectricHitCounter_), counterSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_missCounter_),    counterSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_shadowCounter_),  counterSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probeCounter_),   counterSz));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_exitCounter_),    counterSz));

	return true;
}

void WavefrontPathTracer::freeQueues() {
	auto freeDev = [](CUdeviceptr& p) {
		if (p) { cudaFree(reinterpret_cast<void*>(p)); p = 0; }
	};
	freeDev(d_rayItems_);      freeDev(d_nextRayItems_);
	freeDev(d_hitItems_);      freeDev(d_simpleHitItems_);
	freeDev(d_dielectricHitItems_);
	freeDev(d_missItems_);
	freeDev(d_shadowItems_);   freeDev(d_transmittance_);
	freeDev(d_probeItems_);    freeDev(d_exitItems_);
	freeDev(d_rayCounter_);    freeDev(d_nextRayCounter_);
	freeDev(d_hitCounter_);    freeDev(d_simpleHitCounter_);
	freeDev(d_dielectricHitCounter_);
	freeDev(d_missCounter_);
	freeDev(d_shadowCounter_);
	freeDev(d_probeCounter_);  freeDev(d_exitCounter_);
	queueCapacity_ = 0;
	shadowQueueCapacity_ = 0;
}

// ============================================================================
// Helper: read queue counter (device -> host)
// ============================================================================

int WavefrontPathTracer::readQueueSize(int* d_counter) {
	int val = 0;
	CUDA_CHECK(cudaMemcpyAsync(&val, d_counter, sizeof(int), cudaMemcpyDeviceToHost, stream_));
	CUDA_CHECK(cudaStreamSynchronize(stream_));
	return val;
}

void WavefrontPathTracer::resetQueueCounter(int* d_counter) {
	wf_reset_queue_counter(d_counter, stream_);
}

// ============================================================================
// CUDA kernel launchers
// ============================================================================

void WavefrontPathTracer::launchGenerateCameraRays(
	int width, int height, int sampleIdx,
	const GpuCameraParams& camera, float* d_weightBuffer, bool checkerboardActive,
	const unsigned char* d_activePixelMask)
{
	WorkQueue<RayWorkItem> rq;
	rq.items    = reinterpret_cast<RayWorkItem*>(d_rayItems_);
	rq.counter  = reinterpret_cast<int*>(d_rayCounter_);
	rq.capacity = queueCapacity_;
	// Live Preview's temporal upscale feature - see setTemporalUpscaleJitter()'s
	// own comment. temporalJitterBaseIndex_ + sampleIdx gives each sample
	// within a single render() call (this function is called once per
	// sampleIdx in the sample loop below) its own distinct place in the
	// deterministic jitter sequence, the same way frameNumber_ already
	// varies seed's own random stream call to call.
	const unsigned int temporalJitterIndex = temporalJitterBaseIndex_ + (unsigned int)sampleIdx;
	wf_launch_generate_camera_rays(rq, width, height, sampleIdx, camera, frameNumber_, d_weightBuffer, checkerboardActive,
		d_activePixelMask, temporalUpscaleJitterEnabled_, temporalJitterIndex, temporalUpscaleFactor_, stream_);
}

GpuRestirTemporalContext WavefrontPathTracer::buildRestirTemporalContext() const {
	GpuRestirTemporalContext ctx;
	if (!restirEnabled_) return ctx;  // default: historyValid=false, a safe no-op
	ctx.history = reinterpret_cast<const GpuReservoir*>(d_reservoirsHistory_);
	ctx.worldPosHistory = reinterpret_cast<const float4*>(d_worldPosHistory_);
	ctx.normalOut = reinterpret_cast<float3*>(d_restirNormal_);
	ctx.prevCamera = prevRestirCamera_;
	ctx.historyValid = restirHistoryValid_;
	ctx.imageWidth = restirImageWidth_;
	ctx.imageHeight = restirImageHeight_;
	return ctx;
}

// Whether a medium scatter point's reservoir is carried across frames (temporal combine) and between pixels (the spatial pass that feeds it).
// OFF: a scatter point is redrawn along the ray every frame, so the previous frame's reservoir at the reprojected pixel belongs to a different
// point of the volume - its contribution weight W is for another target function, which temporal reuse then applies at this one. Measured on
// E1 (a fog sphere nearly filling the Cornell box, Live Preview, 300 frames): that reuse read the frame 3.4% dark with ReSTIR DI and 5% with
// DI+GI in every colour channel (a grey fog the same), and 0.6% / -0.5% with it off; the noise of a single frame is the same either way
// (E1 465% vs 437% of the mean raw at 1 spp, E3 and A8 identical), so it bought nothing. The within-frame resampling at the scatter point
// (kRestirCandidateCount candidates) is unchanged.
static constexpr bool kVolumeRestirHistoryReuse = false;

GpuVolumeRestirTemporalContext WavefrontPathTracer::buildVolumeRestirTemporalContext() const {
	GpuVolumeRestirTemporalContext ctx;
	if (!restirEnabled_ || !kVolumeRestirHistoryReuse) return ctx;  // default: historyValid=false, a safe no-op
	ctx.history = reinterpret_cast<const GpuVolumeReservoir*>(d_volumeReservoirsHistory_);
	ctx.worldPosHistory = reinterpret_cast<const float4*>(d_worldPosHistory_);
	ctx.prevCamera = prevRestirCamera_;
	ctx.historyValid = restirHistoryValid_;
	ctx.imageWidth = restirImageWidth_;
	ctx.imageHeight = restirImageHeight_;
	return ctx;
}

WfLightBvhContext WavefrontPathTracer::buildLightBvhContext() const {
	WfLightBvhContext ctx;
	ctx.nodes = reinterpret_cast<const LightBVHNode*>(d_lightBvhNodes_);
	ctx.bitTrail = reinterpret_cast<const unsigned int*>(d_lightBvhBitTrail_);
	ctx.nodeCount = lightBvhNodeCount_;
	ctx.allBMinX = lightBvhAllBMinX_; ctx.allBMinY = lightBvhAllBMinY_; ctx.allBMinZ = lightBvhAllBMinZ_;
	ctx.allBMaxX = lightBvhAllBMaxX_; ctx.allBMaxY = lightBvhAllBMaxY_; ctx.allBMaxZ = lightBvhAllBMaxZ_;
	return ctx;
}

void WavefrontPathTracer::launchEvaluateMaterials(
	int numHits, int maxDepth, bool regularize, float maxComponentValue,
	const SphereData*    d_spheres,   unsigned int numSpheres,
	const QuadData*      d_quads,     unsigned int numQuads,
	const TriangleData*  d_triangles, unsigned int numTriangles,
	const BilinearPatchData* d_bilinearPatches, unsigned int numBilinearPatches,
	const DiskData*      d_disks,      unsigned int numDisks,
	const CylinderData*  d_cylinders,  unsigned int numCylinders,
	const MaterialData*  d_materials, unsigned int numMaterials,
	const int*           d_lightIndices, const GpuLightKind* d_lightKinds,
	const GpuAliasEntry* d_aliasTable,  unsigned int numLights,
	const PunctualLightGPU* d_punctualLights, unsigned int numPunctualLights,
	float3*              d_framebuffer, float3 skyColor, float shadowRayEpsilon,
	GpuSkyDistribution skyDist, GpuPortalLight portalLight)
{
	if (numHits == 0) return;

	WorkQueue<HitWorkItem> hq;
	hq.items    = reinterpret_cast<HitWorkItem*>(d_hitItems_);
	hq.counter  = reinterpret_cast<int*>(d_hitCounter_);
	hq.capacity = queueCapacity_;

	WorkQueue<RayWorkItem> nq;
	nq.items    = reinterpret_cast<RayWorkItem*>(d_nextRayItems_);
	nq.counter  = reinterpret_cast<int*>(d_nextRayCounter_);
	nq.capacity = queueCapacity_;

	WorkQueue<ShadowRayWorkItem> sq;
	sq.items    = reinterpret_cast<ShadowRayWorkItem*>(d_shadowItems_);
	sq.counter  = reinterpret_cast<int*>(d_shadowCounter_);
	sq.capacity = shadowQueueCapacity_;

	// BSSRDF probe-request queue (MaterialType::Subsurface, Phase 2) - filled
	// by evaluate_materials's own Subsurface case instead of scattering
	// inline (see wavefront_types.h's BssrdfProbeWorkItem comment).
	WorkQueue<BssrdfProbeWorkItem> pq;
	pq.items    = reinterpret_cast<BssrdfProbeWorkItem*>(d_probeItems_);
	pq.counter  = reinterpret_cast<int*>(d_probeCounter_);
	pq.capacity = queueCapacity_;

	wf_launch_evaluate_materials(hq, numHits, nq, sq, pq, d_framebuffer,
		d_spheres, numSpheres,
		d_quads, numQuads,
		d_triangles, numTriangles,
		d_bilinearPatches, numBilinearPatches,
		d_disks, numDisks,
		d_cylinders, numCylinders,
		d_materials, numMaterials,
		d_lightIndices, d_lightKinds, d_aliasTable, numLights,
		d_punctualLights, numPunctualLights,
		reinterpret_cast<const TextureData*>(d_textures_),
		reinterpret_cast<const unsigned char*>(d_texturePixels_),
		maxDepth,
		reinterpret_cast<const CloudMedium<float>*>(d_cloudMediums_), numCloudMediums_,
		reinterpret_cast<const GpuRgbGridMedium*>(d_rgbGridMediums_),
		reinterpret_cast<const float*>(d_rgbGridData_),
		reinterpret_cast<const GpuGridMedium*>(d_gridMediums_),
		reinterpret_cast<const float*>(d_gridData_),
		reinterpret_cast<const GpuMeasuredTable*>(d_measuredTables_), numMeasuredTables_,
		reinterpret_cast<const float*>(d_measuredParamValues_),
		reinterpret_cast<const float*>(d_measuredData_),
		reinterpret_cast<const float*>(d_measuredMcdf_),
		reinterpret_cast<const float*>(d_measuredCcdf_),
		skyColor, shadowRayEpsilon, skyDist, portalLight, regularize, maxComponentValue,
		reinterpret_cast<float3*>(denoiserResources_.albedoAov),
		reinterpret_cast<float3*>(denoiserResources_.normalAov),
		reinterpret_cast<float4*>(d_worldPos_),
		reinterpret_cast<GpuReservoir*>(d_reservoirs_),
		buildRestirTemporalContext(),
		reinterpret_cast<GpuGiOriginContext*>(d_giOriginContext_),
		reinterpret_cast<GpuGiSample*>(d_giCandidateOut_),
		buildLightBvhContext(),
		probeGridMeta_,
		// nullptr (not d_guidingHistograms_) when disabled - evaluate_materials's
		// own Conductor/RoughMetal guiding gates on this pointer alone, same
		// "null disables the whole feature" shape as launchEvaluateMaterialsSimple's
		// own probeGrid forwarding. Also nullptr whenever the probe cache
		// itself is off, since path guiding hard-depends on it - see
		// guidingActive()'s own comment for why that's checked via one shared
		// helper rather than re-derived here, since OptiXRenderer always
		// builds/uploads the histogram array regardless of either flag
		// (cheap, matches d_probeGrid_'s own "always built, usage gated
		// separately" precedent).
		guidingActive() ? reinterpret_cast<const GpuGuidingHistogram*>(d_guidingHistograms_) : nullptr,
		// Same probe array the SH-L1 diffuse cache already leak-guards
		// against (probe_grid_types.h's wf_query_probe_grid()) - guiding's
		// own nearest-probe lookup reuses each probe's meanDist/meanDistSq
		// to reject a "nearest by index" probe that's actually occluded
		// (e.g. on the far side of a thin wall), the same failure mode the
		// SH-L1 path already guards against. nullptr whenever guiding itself
		// is inactive (guidingActive() above), matching every other guiding
		// pointer's "null disables" shape.
		guidingActive() ? reinterpret_cast<const GpuProbe*>(d_probeGrid_) : nullptr,
		// Neural Radiance Cache (Live Preview only) - see wf_finish_material_
		// scatter's own nrcWeights parameter comment. nrcEnabled_/d_nrcWeights_
		// gate this exactly like every other "null disables the feature"
		// pointer above; RoughMetal (the only material type reaching THIS
		// kernel that NRC supports) needs a real value here, unlike
		// probeGridMeta/probeGrid which stay on their own nullptr/default for
		// this kernel (Lambertian never reaches it).
		nrcEnabled_ ? reinterpret_cast<const float*>(d_nrcWeights_) : nullptr,
		nrcTrainingSteps_, nrcAabbMin_, nrcAabbExtent_,
		// ReSTIR for volumetric/participating media (Live Preview only) - see
		// wf_finish_material_scatter's own restirVolumeReservoirs/
		// restirVolumeCtx/volumeMatIdxOut/volumeEntryPointOut parameter
		// comments. Same restirEnabled_ gate as d_reservoirs_/
		// buildRestirTemporalContext() above - no separate UI toggle for
		// this feature (see this project's own plan).
		reinterpret_cast<GpuVolumeReservoir*>(d_volumeReservoirs_),
		buildVolumeRestirTemporalContext(),
		reinterpret_cast<int*>(d_volumeMatIdx_),
		reinterpret_cast<float4*>(d_volumePhaseWoG_),
		reinterpret_cast<float4*>(d_volumeEntryPoint_),
		stream_);
}

// Twin of launchEvaluateMaterials() above, scoped to simpleHitQueue's
// Lambertian/Metal hits - see wavefront_kernels.cu's evaluate_materials_simple()
// and wavefront_types.h's WavefrontQueues::simpleHitQueue comment.
void WavefrontPathTracer::launchEvaluateMaterialsSimple(
	int numHits, int maxDepth,
	const SphereData*    d_spheres,   unsigned int numSpheres,
	const QuadData*      d_quads,     unsigned int numQuads,
	const TriangleData*  d_triangles, unsigned int numTriangles,
	const BilinearPatchData* d_bilinearPatches, unsigned int numBilinearPatches,
	const DiskData*      d_disks,      unsigned int numDisks,
	const CylinderData*  d_cylinders,  unsigned int numCylinders,
	const MaterialData*  d_materials, unsigned int numMaterials,
	const int*           d_lightIndices, const GpuLightKind* d_lightKinds,
	const GpuAliasEntry* d_aliasTable,  unsigned int numLights,
	const PunctualLightGPU* d_punctualLights, unsigned int numPunctualLights,
	float3*              d_framebuffer, float3 skyColor, float shadowRayEpsilon,
	GpuSkyDistribution skyDist, GpuPortalLight portalLight, float maxComponentValue)
{
	if (numHits == 0) return;

	WorkQueue<HitWorkItem> hq;
	hq.items    = reinterpret_cast<HitWorkItem*>(d_simpleHitItems_);
	hq.counter  = reinterpret_cast<int*>(d_simpleHitCounter_);
	hq.capacity = queueCapacity_;

	WorkQueue<RayWorkItem> nq;
	nq.items    = reinterpret_cast<RayWorkItem*>(d_nextRayItems_);
	nq.counter  = reinterpret_cast<int*>(d_nextRayCounter_);
	nq.capacity = queueCapacity_;

	WorkQueue<ShadowRayWorkItem> sq;
	sq.items    = reinterpret_cast<ShadowRayWorkItem*>(d_shadowItems_);
	sq.counter  = reinterpret_cast<int*>(d_shadowCounter_);
	sq.capacity = shadowQueueCapacity_;

	wf_launch_evaluate_materials_simple(hq, numHits, nq, sq, d_framebuffer,
		d_spheres, numSpheres,
		d_quads, numQuads,
		d_triangles, numTriangles,
		d_bilinearPatches, numBilinearPatches,
		d_disks, numDisks,
		d_cylinders, numCylinders,
		d_materials, numMaterials,
		d_lightIndices, d_lightKinds, d_aliasTable, numLights,
		d_punctualLights, numPunctualLights,
		reinterpret_cast<const TextureData*>(d_textures_),
		reinterpret_cast<const unsigned char*>(d_texturePixels_),
		maxDepth,
		skyColor, shadowRayEpsilon, skyDist, portalLight, maxComponentValue,
		reinterpret_cast<float3*>(denoiserResources_.albedoAov),
		reinterpret_cast<float3*>(denoiserResources_.normalAov),
		reinterpret_cast<float4*>(d_worldPos_),
		reinterpret_cast<GpuReservoir*>(d_reservoirs_),
		buildRestirTemporalContext(),
		reinterpret_cast<GpuGiOriginContext*>(d_giOriginContext_),
		reinterpret_cast<GpuGiSample*>(d_giCandidateOut_),
		buildLightBvhContext(),
		probeGridMeta_,
		// nullptr (not d_probeGrid_) when disabled - wf_finish_material_
		// scatter's own probe-cache lookup block gates on this pointer alone
		// (probeGrid != nullptr), same "null disables the whole feature"
		// shape as restirReservoirs/giOriginContext - see setProbeCacheEnabled()'s
		// own comment for why the grid can be built/uploaded yet still unused.
		probeCacheEnabled_ ? reinterpret_cast<const GpuProbe*>(d_probeGrid_) : nullptr,
		// Neural Radiance Cache (Live Preview only) - see wf_finish_material_
		// scatter's own nrcWeights parameter comment. This is the only
		// evaluate_materials* kernel that ever sees a Lambertian hit, one of
		// the two material types NRC supports.
		nrcEnabled_ ? reinterpret_cast<const float*>(d_nrcWeights_) : nullptr,
		nrcTrainingSteps_, nrcAabbMin_, nrcAabbExtent_,
		simpleMaterialStream_);
}

// Twin of launchEvaluateMaterialsSimple() above, scoped to dielectricHitQueue's
// Dielectric/RoughDielectric hits - see wavefront_kernels.cu's
// evaluate_materials_dielectric() and wavefront_types.h's WavefrontQueues::
// dielectricHitQueue comment. No texture params - neither material type reads
// mat.textureIdx.
void WavefrontPathTracer::launchEvaluateMaterialsDielectric(
	int numHits, int maxDepth, bool regularize, float maxComponentValue,
	const SphereData*    d_spheres,   unsigned int numSpheres,
	const QuadData*      d_quads,     unsigned int numQuads,
	const TriangleData*  d_triangles, unsigned int numTriangles,
	const BilinearPatchData* d_bilinearPatches, unsigned int numBilinearPatches,
	const DiskData*      d_disks,      unsigned int numDisks,
	const CylinderData*  d_cylinders,  unsigned int numCylinders,
	const MaterialData*  d_materials, unsigned int numMaterials,
	const int*           d_lightIndices, const GpuLightKind* d_lightKinds,
	const GpuAliasEntry* d_aliasTable,  unsigned int numLights,
	const PunctualLightGPU* d_punctualLights, unsigned int numPunctualLights,
	float3*              d_framebuffer, float3 skyColor, float shadowRayEpsilon,
	GpuSkyDistribution skyDist, GpuPortalLight portalLight)
{
	if (numHits == 0) return;

	WorkQueue<HitWorkItem> hq;
	hq.items    = reinterpret_cast<HitWorkItem*>(d_dielectricHitItems_);
	hq.counter  = reinterpret_cast<int*>(d_dielectricHitCounter_);
	hq.capacity = queueCapacity_;

	WorkQueue<RayWorkItem> nq;
	nq.items    = reinterpret_cast<RayWorkItem*>(d_nextRayItems_);
	nq.counter  = reinterpret_cast<int*>(d_nextRayCounter_);
	nq.capacity = queueCapacity_;

	WorkQueue<ShadowRayWorkItem> sq;
	sq.items    = reinterpret_cast<ShadowRayWorkItem*>(d_shadowItems_);
	sq.counter  = reinterpret_cast<int*>(d_shadowCounter_);
	sq.capacity = shadowQueueCapacity_;

	wf_launch_evaluate_materials_dielectric(hq, numHits, nq, sq, d_framebuffer,
		d_spheres, numSpheres,
		d_quads, numQuads,
		d_triangles, numTriangles,
		d_bilinearPatches, numBilinearPatches,
		d_disks, numDisks,
		d_cylinders, numCylinders,
		d_materials, numMaterials,
		d_lightIndices, d_lightKinds, d_aliasTable, numLights,
		d_punctualLights, numPunctualLights,
		reinterpret_cast<const TextureData*>(d_textures_),
		reinterpret_cast<const unsigned char*>(d_texturePixels_),
		maxDepth,
		skyColor, shadowRayEpsilon, skyDist, portalLight, regularize, maxComponentValue,
		reinterpret_cast<float3*>(denoiserResources_.albedoAov),
		reinterpret_cast<float3*>(denoiserResources_.normalAov),
		reinterpret_cast<float4*>(d_worldPos_),
		reinterpret_cast<GpuReservoir*>(d_reservoirs_),
		buildRestirTemporalContext(),
		reinterpret_cast<GpuGiOriginContext*>(d_giOriginContext_),
		reinterpret_cast<GpuGiSample*>(d_giCandidateOut_),
		buildLightBvhContext(),
		dielectricMaterialStream_);
}

void WavefrontPathTracer::launchRestirSpatialReuse(
		const SphereData* d_spheres, const QuadData* d_quads, const TriangleData* d_triangles,
		const BilinearPatchData* d_bilinearPatches, const DiskData* d_disks, const CylinderData* d_cylinders,
		const MaterialData* d_materials) {
	if (!restirEnabled_) return;
	wf_launch_restir_spatial_reuse(
		reinterpret_cast<const GpuReservoir*>(d_reservoirs_),
		reinterpret_cast<const float3*>(d_restirNormal_),
		reinterpret_cast<const float4*>(d_worldPos_),
		reinterpret_cast<GpuReservoir*>(d_reservoirsHistory_),
		restirImageWidth_, restirImageHeight_,
		frameNumber_,
		d_spheres, d_quads, d_triangles, d_bilinearPatches, d_disks, d_cylinders, d_materials,
		reinterpret_cast<const TextureData*>(d_textures_),
		reinterpret_cast<const unsigned char*>(d_texturePixels_),
		stream_);
}

void WavefrontPathTracer::launchRestirVolumeSpatialReuse(
		const SphereData* d_spheres, const QuadData* d_quads, const TriangleData* d_triangles,
		const BilinearPatchData* d_bilinearPatches, const DiskData* d_disks, const CylinderData* d_cylinders,
		const MaterialData* d_materials) {
	// Its only consumer is the temporal combine above (it writes the history that combine reads) - see kVolumeRestirHistoryReuse.
	if (!restirEnabled_ || !kVolumeRestirHistoryReuse) return;
	wf_launch_restir_volume_spatial_reuse(
		reinterpret_cast<const GpuVolumeReservoir*>(d_volumeReservoirs_),
		reinterpret_cast<const int*>(d_volumeMatIdx_),
		reinterpret_cast<const float4*>(d_volumePhaseWoG_),
		reinterpret_cast<const float4*>(d_volumeEntryPoint_),
		reinterpret_cast<GpuVolumeReservoir*>(d_volumeReservoirsHistory_),
		restirImageWidth_, restirImageHeight_,
		frameNumber_,
		d_spheres, d_quads, d_triangles, d_bilinearPatches, d_disks, d_cylinders, d_materials,
		reinterpret_cast<const TextureData*>(d_textures_),
		reinterpret_cast<const unsigned char*>(d_texturePixels_),
		stream_);
}

void WavefrontPathTracer::launchGiFinalize(const MaterialData* d_materials, float3* d_framebuffer, float maxComponentValue, const float* d_weightBuffer) {
	if (!restirGiEnabled_) return;
	wf_launch_restir_gi_finalize(
		reinterpret_cast<const GpuGiOriginContext*>(d_giOriginContext_),
		reinterpret_cast<const GpuGiSample*>(d_giCandidateOut_),
		reinterpret_cast<const GpuGiReservoir*>(d_giReservoirsHistory_),
		reinterpret_cast<const float4*>(d_worldPosHistory_),
		reinterpret_cast<const float4*>(d_worldPos_),
		d_weightBuffer,
		prevRestirCamera_,
		restirGiHistoryValid_,
		restirImageWidth_, restirImageHeight_,
		frameNumber_,
		d_materials, d_framebuffer, maxComponentValue,
		reinterpret_cast<GpuGiReservoir*>(d_giReservoirs_),
		stream_);
}

void WavefrontPathTracer::launchGiSpatialReuse() {
	if (!restirGiEnabled_) return;
	wf_launch_restir_gi_spatial_reuse(
		reinterpret_cast<const GpuGiReservoir*>(d_giReservoirs_),
		reinterpret_cast<const GpuGiOriginContext*>(d_giOriginContext_),
		reinterpret_cast<GpuGiReservoir*>(d_giReservoirsHistory_),
		restirImageWidth_, restirImageHeight_,
		frameNumber_,
		stream_);
}

void WavefrontPathTracer::launchProbeCacheUpdate(const WavefrontLaunchParams& lp, float3 backgroundColor, float shadowRayEpsilon) {
	if (!probeCacheEnabled_ || probeGridMeta_.totalProbes <= 0 || d_probeGrid_ == 0) return;
	const int batchSize = std::min(kProbesPerFrame_, probeGridMeta_.totalProbes);
	if (batchSize <= 0) return;

	// kProbesPerFrame_ is a fixed compile-time constant (unlike every other
	// per-pixel queue in this class) - lazily allocate once, guarded by
	// d_probeCacheRayItems_ alone (see these members' own comments,
	// wavefront_path_tracer.h).
	if (!d_probeCacheRayItems_) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probeCacheRayItems_),    kProbesPerFrame_ * sizeof(ProbeCacheRayWorkItem)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probeCacheRayCounter_),  sizeof(int)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probeCacheHitItems_),    kProbesPerFrame_ * sizeof(ProbeCacheHitWorkItem)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probeCacheHitCounter_),  sizeof(int)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probeCacheRadianceOut_), kProbesPerFrame_ * sizeof(float3)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probeCacheHitDistOut_),  kProbesPerFrame_ * sizeof(float)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probeCacheDirections_),  kProbesPerFrame_ * sizeof(float3)));
	}

	// ------------------------------------------------------------------
	// Build this frame's round-robin batch host-side: probeUpdateCursor_..
	// +batchSize (mod totalProbes) - see ProbeCacheRayWorkItem::batchSlot's
	// own comment (wavefront_types.h) for why batchSlot is a batch-local
	// index, not the real probe index.
	// ------------------------------------------------------------------
	// Reused across calls (never reallocated once big enough) rather than a
	// fresh std::vector construction every render() call - see these
	// members' own comment (wavefront_path_tracer.h).
	if (probeCacheHostItems_.size() < static_cast<size_t>(batchSize)) {
		probeCacheHostItems_.resize(batchSize);
		probeCacheHostDirections_.resize(batchSize);
	}
	std::vector<ProbeCacheRayWorkItem>& hostItems = probeCacheHostItems_;
	std::vector<float3>& hostDirections = probeCacheHostDirections_;
	// Deterministic per-(frame,cursor) seed - real randomness isn't needed
	// here (this is a coarse, temporally-EMA'd cache, not a converged
	// reference render), just decorrelation between probes/frames so
	// repeated updates of the same probe eventually cover its whole sphere.
	std::mt19937 rng(0x9E3779B9u ^ static_cast<unsigned int>(frameNumber_) ^ static_cast<unsigned int>(probeUpdateCursor_));
	std::uniform_real_distribution<float> uni(0.0f, 1.0f);
	const int dimsXY = probeGridMeta_.dims.x * probeGridMeta_.dims.y;
	for (int i = 0; i < batchSize; ++i) {
		const int probeIdx = (probeUpdateCursor_ + i) % probeGridMeta_.totalProbes;
		const int pz = (dimsXY > 0) ? probeIdx / dimsXY : 0;
		const int rem = (dimsXY > 0) ? probeIdx % dimsXY : 0;
		const int py = (probeGridMeta_.dims.x > 0) ? rem / probeGridMeta_.dims.x : 0;
		const int px = (probeGridMeta_.dims.x > 0) ? rem % probeGridMeta_.dims.x : 0;
		const int3 coord = {px, py, pz};

		// Rejection-sample a uniform point in the unit ball then normalize -
		// same algorithm as wf_rand_unit()'s own device-side rejection loop
		// (wavefront_device_helpers.h), just built host-side here because a
		// probe ray's own direction must exist BEFORE the OptiX raygen ever
		// runs (see ProbeCacheRayWorkItem::direction's own comment).
		float3 dir = make_float3(0.0f, 0.0f, 1.0f);
		for (;;) {
			const float px3 = 2.0f * uni(rng) - 1.0f;
			const float py3 = 2.0f * uni(rng) - 1.0f;
			const float pz3 = 2.0f * uni(rng) - 1.0f;
			const float l = px3 * px3 + py3 * py3 + pz3 * pz3;
			if (l > 1e-8f && l < 1.0f) {
				const float invLen = 1.0f / sqrtf(l);
				dir = make_float3(px3 * invLen, py3 * invLen, pz3 * invLen);
				break;
			}
		}

		ProbeCacheRayWorkItem item;
		item.batchSlot = i;
		item.origin    = probeGridMeta_.probeWorldPos(coord);
		item.direction = dir;
		item.seed      = rng();
		hostItems[i]      = item;
		hostDirections[i] = dir;
	}

	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_probeCacheRayItems_), hostItems.data(),
							   static_cast<size_t>(batchSize) * sizeof(ProbeCacheRayWorkItem),
							   cudaMemcpyHostToDevice, stream_));
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_probeCacheDirections_), hostDirections.data(),
							   static_cast<size_t>(batchSize) * sizeof(float3),
							   cudaMemcpyHostToDevice, stream_));
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_probeCacheRayCounter_), &batchSize,
							   sizeof(int), cudaMemcpyHostToDevice, stream_));
	CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_probeCacheHitCounter_), 0, sizeof(int), stream_));
	CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_probeCacheRadianceOut_), 0,
							   static_cast<size_t>(batchSize) * sizeof(float3), stream_));

	// ------------------------------------------------------------------
	// Stage 1: OptiX intersection launch - __raygen__wf_probe_cache via
	// probeCacheSBT_ (wavefront_probe_cache.h's own header comment).
	// ------------------------------------------------------------------
	WavefrontLaunchParams probeLp = lp;
	probeLp.probeCacheRayQueue.items    = reinterpret_cast<ProbeCacheRayWorkItem*>(d_probeCacheRayItems_);
	probeLp.probeCacheRayQueue.counter  = reinterpret_cast<int*>(d_probeCacheRayCounter_);
	probeLp.probeCacheRayQueue.capacity = kProbesPerFrame_;
	probeLp.probeCacheHitQueue.items    = reinterpret_cast<ProbeCacheHitWorkItem*>(d_probeCacheHitItems_);
	probeLp.probeCacheHitQueue.counter  = reinterpret_cast<int*>(d_probeCacheHitCounter_);
	probeLp.probeCacheHitQueue.capacity = kProbesPerFrame_;

	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_wfLaunchParams_), &probeLp,
							   sizeof(WavefrontLaunchParams), cudaMemcpyHostToDevice, stream_));
	OPTIX_CHECK(optixLaunch(
		intersectPipeline_, stream_,
		d_wfLaunchParams_, sizeof(WavefrontLaunchParams),
		&probeCacheSBT_,
		static_cast<unsigned int>(batchSize), 1, 1));
	// No cudaStreamSynchronize here - readQueueSize() below issues its own
	// cudaMemcpyAsync on this SAME stream_, so stream ordering alone already
	// guarantees it sees this launch's finished results; an explicit sync
	// first would just be a second, redundant host-blocking wait.
	const int numHits = readQueueSize(reinterpret_cast<int*>(d_probeCacheHitCounter_));
	if (numHits <= 0) {
		probeUpdateCursor_ = (probeUpdateCursor_ + batchSize) % probeGridMeta_.totalProbes;
		return;
	}

	// ------------------------------------------------------------------
	// Stage 2: probe_cache_shade - one-light NEE draw, pushes any resulting
	// shadow ray into the ORDINARY shadow queue/pipeline/SBT. Safe to reuse
	// here (not a fresh dedicated pipeline) because this whole method runs
	// strictly after render()'s own per-bounce loop has fully drained those
	// buffers for this render() call - see this method's own declaration
	// comment (wavefront_path_tracer.h).
	// ------------------------------------------------------------------
	WorkQueue<ProbeCacheHitWorkItem> hq;
	hq.items    = reinterpret_cast<ProbeCacheHitWorkItem*>(d_probeCacheHitItems_);
	hq.counter  = reinterpret_cast<int*>(d_probeCacheHitCounter_);
	hq.capacity = kProbesPerFrame_;

	CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_shadowCounter_), 0, sizeof(int), stream_));

	WorkQueue<ShadowRayWorkItem> sq;
	sq.items    = reinterpret_cast<ShadowRayWorkItem*>(d_shadowItems_);
	sq.counter  = reinterpret_cast<int*>(d_shadowCounter_);
	sq.capacity = shadowQueueCapacity_;

	wf_launch_probe_cache_shade(
		hq, numHits,
		lp.materials, lp.spheres, lp.quads, lp.triangles, lp.bilinearPatches, lp.disks, lp.cylinders,
		lp.textures, lp.texturePixels,
		lp.lightIndices, lp.lightKinds, lp.aliasTable, lp.numLights,
		buildLightBvhContext(),
		backgroundColor, shadowRayEpsilon,
		reinterpret_cast<float3*>(d_probeCacheRadianceOut_),
		reinterpret_cast<float*>(d_probeCacheHitDistOut_),
		sq, stream_);
	// Same reasoning as Stage 1's own comment above - readQueueSize() below
	// syncs this same stream_ itself, so no separate sync is needed here.
	const int numShadow = readQueueSize(reinterpret_cast<int*>(d_shadowCounter_));
	if (numShadow > 0) {
		// --------------------------------------------------------------
		// Stage 3: OptiX shadow launch - real occlusion test, exactly the
		// same shadowPipeline_/shadowSBT_ every other NEE draw uses.
		// --------------------------------------------------------------
		WavefrontLaunchParams shadowLp = lp;
		shadowLp.shadowQueue = sq;
		// Temporarily point framebuffer to the transmittance float array -
		// same convention render()'s own per-bounce shadow launch uses (see
		// that call site's own comment).
		shadowLp.framebuffer = reinterpret_cast<float3*>(d_transmittance_);

		CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_wfLaunchParams_), &shadowLp,
								   sizeof(WavefrontLaunchParams), cudaMemcpyHostToDevice, stream_));
		OPTIX_CHECK(optixLaunch(
			shadowPipeline_, stream_,
			d_wfLaunchParams_, sizeof(WavefrontLaunchParams),
			&shadowSBT_,
			static_cast<unsigned int>(numShadow), 1, 1));
		// No sync here (or after the accumulate launch just below) - both
		// this shadow optixLaunch and wf_launch_accumulate_shadow run on the
		// same stream_, and so does wf_launch_probe_cache_accumulate further
		// down, so stream ordering alone already sequences all three
		// correctly; nothing in between needs a host-visible result. Same
		// "no correctness benefit, only a CPU stall" reasoning as render()'s
		// own per-bounce shadow-launch call site.
		//
		// Stage 4: redirect resolved Ld into d_probeCacheRadianceOut_ (every
		// item here has isProbeCacheRay=true - see probe_cache_shade's own
		// comment - so d_framebuffer/maxComponentValue are never touched).
		wf_launch_accumulate_shadow(sq, numShadow,
									 reinterpret_cast<const float*>(d_transmittance_),
									 /*d_framebuffer=*/nullptr, /*maxComponentValue=*/0.0f, stream_,
									 /*d_giCandidateOut=*/nullptr,
									 reinterpret_cast<float3*>(d_probeCacheRadianceOut_));
	}

	// ------------------------------------------------------------------
	// EMA-blend this frame's resolved samples into the persistent GpuProbe
	// array (probe_cache_accumulate, wavefront_kernels_restir.cu).
	// ------------------------------------------------------------------
	wf_launch_probe_cache_accumulate(
		reinterpret_cast<const float3*>(d_probeCacheRadianceOut_),
		reinterpret_cast<const float*>(d_probeCacheHitDistOut_),
		reinterpret_cast<const float3*>(d_probeCacheDirections_),
		batchSize, probeUpdateCursor_, probeGridMeta_,
		reinterpret_cast<GpuProbe*>(d_probeGrid_),
		// nullptr when path guiding is off (or the probe cache itself is off
		// - see guidingActive()'s own comment) - probe_cache_accumulate's own
		// guidingHistograms parameter comment (wavefront_kernels_restir.cu).
		// This function is itself only ever reached with probeCacheEnabled_
		// true (see its own early-return above), so guidingActive() here is
		// equivalent to the bare pathGuidingEnabled_ this used to read - but
		// spelled the same way as launchEvaluateMaterials()'s check so the
		// invariant has one enforcement point instead of two, and stays
		// correct even if this function's early-return is ever restructured.
		guidingActive() ? reinterpret_cast<GpuGuidingHistogram*>(d_guidingHistograms_) : nullptr,
		stream_);
	// No trailing sync - the only remaining work in this function is the
	// host-only probeUpdateCursor_ update just below (no GPU dependency),
	// and render()'s own final framebuffer readback (a plain, stream-less
	// cudaMemcpy right after this method returns) already synchronizes with
	// every stream in the context under CUDA's legacy-default-stream
	// semantics - an explicit sync here would only add one more redundant
	// host-blocking wait on top of that.

	probeUpdateCursor_ = (probeUpdateCursor_ + batchSize) % probeGridMeta_.totalProbes;
}

void WavefrontPathTracer::launchNrcTrainingUpdate(const WavefrontLaunchParams& lp, GpuCameraParams camera, float shadowRayEpsilon) {
	if (!nrcEnabled_) return;

	// Lazy allocation (guarded by d_nrcWeights_ alone) - see this method's
	// own declaration comment (wavefront_path_tracer.h) for the lifecycle
	// rationale and the documented no-reset-on-scene-change simplification.
	// Wrapped in try/catch: CUDA_CHECK throws std::runtime_error on failure
	// (optix_types.h), and rt_realtime_render_frame()'s own top-level catch
	// (optix_interface.cpp) swallows that per-frame rather than tearing down
	// this persistent object - without the rollback below, a transient
	// failure partway through (e.g. low VRAM while allocating the largest
	// buffer, d_nrcTrainingRecords_) would leave d_nrcWeights_ non-null
	// (guard already satisfied) but one or more LATER buffers still null,
	// permanently skipping this whole block on every subsequent frame and
	// leaving NRC broken for the rest of the session with no retry. Freeing
	// and re-nulling everything on any exception here keeps the guard
	// itself the single source of truth: non-null d_nrcWeights_ means ALL
	// SIX buffers are valid, or none are.
	if (!d_nrcWeights_) {
		try {
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nrcWeights_), kNrcNumWeights * sizeof(float)));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nrcAdamM_), kNrcNumWeights * sizeof(float)));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nrcAdamV_), kNrcNumWeights * sizeof(float)));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nrcGradAccum_), kNrcNumWeights * sizeof(float)));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nrcTrainingRecords_), kNrcTrainingRecordCapacity * sizeof(NrcTrainingRecord)));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nrcValidRecordCounter_), sizeof(int)));
			CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_nrcGradAccum_), 0, kNrcNumWeights * sizeof(float), stream_));
			wf_launch_nrc_reset_weights(
				reinterpret_cast<float*>(d_nrcWeights_), reinterpret_cast<float*>(d_nrcAdamM_), reinterpret_cast<float*>(d_nrcAdamV_),
				/*seed=*/0x9E3779B9u, stream_);
		} catch (...) {
			auto freeDev = [](CUdeviceptr& p) { if (p) { cudaFree(reinterpret_cast<void*>(p)); p = 0; } };
			freeDev(d_nrcWeights_); freeDev(d_nrcAdamM_); freeDev(d_nrcAdamV_);
			freeDev(d_nrcGradAccum_); freeDev(d_nrcTrainingRecords_); freeDev(d_nrcValidRecordCounter_);
			throw;
		}
	}

	// Scene AABB reused directly from probeGridMeta_ - see nrcAabbMin_/
	// nrcAabbExtent_'s own member comment (wavefront_path_tracer.h) for why
	// NRC doesn't derive its own separate scene AABB. Re-derived every call
	// (cheap) rather than cached once, matching probeGridMeta_ itself being
	// re-forwarded every render() call elsewhere in this class.
	nrcAabbMin_ = probeGridMeta_.gridMin;
	nrcAabbExtent_ = make_float3(
		probeGridMeta_.cellSize.x * (float)probeGridMeta_.dims.x,
		probeGridMeta_.cellSize.y * (float)probeGridMeta_.dims.y,
		probeGridMeta_.cellSize.z * (float)probeGridMeta_.dims.z);

	CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_nrcTrainingRecords_), 0,
							   kNrcTrainingRecordCapacity * sizeof(NrcTrainingRecord), stream_));
	CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_nrcValidRecordCounter_), 0, sizeof(int), stream_));

	// ------------------------------------------------------------------
	// Phase A: trace kNrcTrainingPathsPerFrame fresh camera rays depth-by-
	// depth, reusing the MAIN per-bounce loop's own rayQueue/nextRayQueue/
	// hitQueue/simpleHitQueue/dielectricHitQueue/missQueue/shadowQueue/
	// shadowPipeline_/intersectPipeline_/d_transmittance_ - safe because
	// this whole method runs strictly after render()'s own per-sample loop
	// has fully drained them for this render() call (same reasoning as
	// launchProbeCacheUpdate()'s own reuse of the shadow queue/pipeline).
	// dielectricHitQueue/missQueue are cleared each depth but never read -
	// a training ray that hits a Dielectric/RoughDielectric surface or
	// escapes the scene simply terminates (no record), see
	// wavefront_kernels_nrc.cu's own header comment for the v1 material
	// scope.
	// ------------------------------------------------------------------
	resetQueueCounter(reinterpret_cast<int*>(d_rayCounter_));
	// dielectricHitQueue/missQueue are never READ during training (see this
	// method's own comment above - a training ray reaching either just
	// terminates, no record) so, unlike hitCounter/simpleHitCounter/
	// nextRayCounter/shadowCounter below, their own counters don't need
	// resetting every depth - only once here, before the loop. The OptiX
	// intersect launch still WRITES into them each depth (any training ray
	// that hits a Dielectric/RoughDielectric surface or escapes the scene
	// pushes into one), but the total across every depth this whole method
	// ever runs (at most kNrcTrainingPathsPerFrame=4096 rays each) is far
	// below queueCapacity_ (width*height), so letting the counts accumulate
	// unreset across the up-to-5 depth iterations can never overflow either
	// buffer.
	resetQueueCounter(reinterpret_cast<int*>(d_dielectricHitCounter_));
	resetQueueCounter(reinterpret_cast<int*>(d_missCounter_));
	WorkQueue<RayWorkItem> initialRayQueue;
	initialRayQueue.items = reinterpret_cast<RayWorkItem*>(d_rayItems_);
	initialRayQueue.counter = reinterpret_cast<int*>(d_rayCounter_);
	initialRayQueue.capacity = queueCapacity_;
	wf_launch_nrc_generate_training_rays(
		initialRayQueue, kNrcTrainingPathsPerFrame, camera, frameNumber_ ^ 0xBEEF0000u, stream_);

	WavefrontLaunchParams trainLp = lp;
	int numActive = kNrcTrainingPathsPerFrame;
	for (int depth = 0; depth < kNrcTrainingMaxBounces; ++depth) {
		if (numActive <= 0) break;

		resetQueueCounter(reinterpret_cast<int*>(d_hitCounter_));
		resetQueueCounter(reinterpret_cast<int*>(d_simpleHitCounter_));
		resetQueueCounter(reinterpret_cast<int*>(d_nextRayCounter_));
		resetQueueCounter(reinterpret_cast<int*>(d_shadowCounter_));

		trainLp.rayQueue.items = reinterpret_cast<RayWorkItem*>(d_rayItems_);
		trainLp.rayQueue.counter = reinterpret_cast<int*>(d_rayCounter_);
		trainLp.rayQueue.capacity = queueCapacity_;
		trainLp.hitQueue.items = reinterpret_cast<HitWorkItem*>(d_hitItems_);
		trainLp.hitQueue.counter = reinterpret_cast<int*>(d_hitCounter_);
		trainLp.hitQueue.capacity = queueCapacity_;
		trainLp.simpleHitQueue.items = reinterpret_cast<HitWorkItem*>(d_simpleHitItems_);
		trainLp.simpleHitQueue.counter = reinterpret_cast<int*>(d_simpleHitCounter_);
		trainLp.simpleHitQueue.capacity = queueCapacity_;
		trainLp.dielectricHitQueue.items = reinterpret_cast<HitWorkItem*>(d_dielectricHitItems_);
		trainLp.dielectricHitQueue.counter = reinterpret_cast<int*>(d_dielectricHitCounter_);
		trainLp.dielectricHitQueue.capacity = queueCapacity_;
		trainLp.missQueue.items = reinterpret_cast<MissWorkItem*>(d_missItems_);
		trainLp.missQueue.counter = reinterpret_cast<int*>(d_missCounter_);
		trainLp.missQueue.capacity = queueCapacity_;

		CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_wfLaunchParams_), &trainLp,
								   sizeof(WavefrontLaunchParams), cudaMemcpyHostToDevice, stream_));
		OPTIX_CHECK(optixLaunch(intersectPipeline_, stream_, d_wfLaunchParams_, sizeof(WavefrontLaunchParams),
								 &intersectSBT_, (unsigned int)numActive, 1, 1));

		const int numSimpleHits = readQueueSize(reinterpret_cast<int*>(d_simpleHitCounter_));
		const int numFullHits = readQueueSize(reinterpret_cast<int*>(d_hitCounter_));

		WorkQueue<RayWorkItem> nq;
		nq.items = reinterpret_cast<RayWorkItem*>(d_nextRayItems_);
		nq.counter = reinterpret_cast<int*>(d_nextRayCounter_);
		nq.capacity = queueCapacity_;
		WorkQueue<ShadowRayWorkItem> sq;
		sq.items = reinterpret_cast<ShadowRayWorkItem*>(d_shadowItems_);
		sq.counter = reinterpret_cast<int*>(d_shadowCounter_);
		sq.capacity = shadowQueueCapacity_;

		if (numSimpleHits > 0) {
			WorkQueue<HitWorkItem> shq;
			shq.items = reinterpret_cast<HitWorkItem*>(d_simpleHitItems_);
			shq.counter = reinterpret_cast<int*>(d_simpleHitCounter_);
			shq.capacity = queueCapacity_;
			wf_launch_nrc_training_shade_simple(
				shq, numSimpleHits, depth, kNrcTrainingMaxBounces,
				lp.materials, lp.spheres, lp.quads, lp.triangles, lp.bilinearPatches, lp.disks, lp.cylinders,
				lp.textures, lp.texturePixels,
				lp.lightIndices, lp.lightKinds, lp.aliasTable, lp.numLights,
				buildLightBvhContext(), shadowRayEpsilon,
				reinterpret_cast<NrcTrainingRecord*>(d_nrcTrainingRecords_),
				reinterpret_cast<int*>(d_nrcValidRecordCounter_),
				nq, sq, stream_);
		}
		if (numFullHits > 0) {
			WorkQueue<HitWorkItem> fhq;
			fhq.items = reinterpret_cast<HitWorkItem*>(d_hitItems_);
			fhq.counter = reinterpret_cast<int*>(d_hitCounter_);
			fhq.capacity = queueCapacity_;
			wf_launch_nrc_training_shade_full(
				fhq, numFullHits, depth, kNrcTrainingMaxBounces,
				lp.materials, lp.spheres, lp.quads, lp.triangles, lp.bilinearPatches, lp.disks, lp.cylinders,
				lp.textures, lp.texturePixels,
				lp.lightIndices, lp.lightKinds, lp.aliasTable, lp.numLights,
				buildLightBvhContext(), shadowRayEpsilon,
				reinterpret_cast<NrcTrainingRecord*>(d_nrcTrainingRecords_),
				reinterpret_cast<int*>(d_nrcValidRecordCounter_),
				nq, sq, stream_);
		}

		const int numShadow = readQueueSize(reinterpret_cast<int*>(d_shadowCounter_));
		if (numShadow > 0) {
			WavefrontLaunchParams shadowLp = lp;
			shadowLp.shadowQueue = sq;
			shadowLp.framebuffer = reinterpret_cast<float3*>(d_transmittance_);
			CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_wfLaunchParams_), &shadowLp,
									   sizeof(WavefrontLaunchParams), cudaMemcpyHostToDevice, stream_));
			OPTIX_CHECK(optixLaunch(shadowPipeline_, stream_, d_wfLaunchParams_, sizeof(WavefrontLaunchParams),
									 &shadowSBT_, (unsigned int)numShadow, 1, 1));
			wf_launch_accumulate_shadow(sq, numShadow,
										 reinterpret_cast<const float*>(d_transmittance_),
										 /*d_framebuffer=*/nullptr, /*maxComponentValue=*/0.0f, stream_,
										 /*d_giCandidateOut=*/nullptr, /*d_probeCacheRadianceOut=*/nullptr,
										 reinterpret_cast<NrcTrainingRecord*>(d_nrcTrainingRecords_));
		}

		numActive = readQueueSize(reinterpret_cast<int*>(d_nextRayCounter_));
		std::swap(d_rayItems_, d_nextRayItems_);
		std::swap(d_rayCounter_, d_nextRayCounter_);
	}

	// ------------------------------------------------------------------
	// Phase B + apply step - see wavefront_kernels_nrc.cu's own header
	// comment for why Phase B is a single flat launch with no per-depth
	// ordering/sync required.
	// ------------------------------------------------------------------
	wf_launch_nrc_bootstrap_and_train(
		reinterpret_cast<const NrcTrainingRecord*>(d_nrcTrainingRecords_), kNrcTrainingRecordCapacity,
		reinterpret_cast<const float*>(d_nrcWeights_), reinterpret_cast<float*>(d_nrcGradAccum_),
		nrcAabbMin_, nrcAabbExtent_, stream_);

	const int numValidRecords = readQueueSize(reinterpret_cast<int*>(d_nrcValidRecordCounter_));
	++nrcTrainingSteps_;
	wf_launch_nrc_apply_gradients(
		reinterpret_cast<float*>(d_nrcWeights_), reinterpret_cast<float*>(d_nrcAdamM_), reinterpret_cast<float*>(d_nrcAdamV_),
		reinterpret_cast<float*>(d_nrcGradAccum_), numValidRecords, nrcTrainingSteps_, stream_);
}

// Neural temporal upscale (Live Preview only) - see wavefront_kernels_
// upscale.cu's own header comment for the full 3-kernel-family design.
// Called from render() BEFORE the world-pos-history/prevRestirCamera_
// overwrite (see this method's own declaration comment, wavefront_path_
// tracer.h) - every reprojection this method does needs those to still
// hold the PREVIOUS frame's values.
void WavefrontPathTracer::launchNeuralUpscaleUpdate(const float3* d_lowResFramebuffer, int width, int height, GpuCameraParams camera) {
	if (!neuralUpscaleEnabled_ || !temporalUpscaleJitterEnabled_) return;

	const int numPixels = width * height;
	const int upscaleFactor = temporalUpscaleFactor_;
	// wf_temporal_upscale_subcell() only supports factor 2 or 4 (anything
	// else falls back to its own 2x table but can still return a cx/cy of
	// 1) - the GUI never offers factor 1 with jitter enabled (that
	// combination means "Off"), but a direct rt_realtime_render_frame()
	// caller could pass one, and upscale_train/upscale_infer's
	// px*upscaleFactor+dueCx indexing would then read one high-res row/
	// column past the end of d_upscaleForwardCache_/d_upscaleHistory_.
	if (upscaleFactor != 2 && upscaleFactor != 4) return;
	const int numHighResPixels = numPixels * upscaleFactor * upscaleFactor;
	if (numPixels <= 0 || numHighResPixels <= 0) return;

	// Lazy allocation of the fixed-size weight/Adam/grad-accum/counter set -
	// same "guarded by one pointer being null, free-and-renull-all-on-
	// exception" lifecycle as d_nrcWeights_'s own block above (see that
	// block's own comment for the full rationale - a partial-allocation
	// failure must not permanently wedge this feature either).
	if (!d_upscaleWeights_) {
		try {
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_upscaleWeights_), kUpscaleNumWeights * sizeof(float)));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_upscaleAdamM_), kUpscaleNumWeights * sizeof(float)));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_upscaleAdamV_), kUpscaleNumWeights * sizeof(float)));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_upscaleGradAccum_), kUpscaleNumWeights * sizeof(float)));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_upscaleValidRecordCounter_), sizeof(int)));
			CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_upscaleGradAccum_), 0, kUpscaleNumWeights * sizeof(float), stream_));
			wf_launch_upscale_reset_weights(
				reinterpret_cast<float*>(d_upscaleWeights_), reinterpret_cast<float*>(d_upscaleAdamM_), reinterpret_cast<float*>(d_upscaleAdamV_),
				/*seed=*/0x51ED270Bu, stream_);
		} catch (...) {
			auto freeDev = [](CUdeviceptr& p) { if (p) { cudaFree(reinterpret_cast<void*>(p)); p = 0; } };
			freeDev(d_upscaleWeights_); freeDev(d_upscaleAdamM_); freeDev(d_upscaleAdamV_);
			freeDev(d_upscaleGradAccum_); freeDev(d_upscaleValidRecordCounter_);
			throw;
		}
	}

	// Resolution/upscale-factor-keyed buffers. d_upscaleForwardCache_/
	// d_upscaleHistory_ are zeroed ONLY on a fresh allocation (a genuinely
	// new/resized buffer's content is undefined otherwise) - both structs'
	// own all-zero state already IS their correct "never written" sentinel
	// (UpscaleForwardCache::flags==0, GpuVolumeReservoir-style; history age
	// w==0 reads as "just written this frame", the same safe-if-slightly-
	// optimistic default worldPos's own w==0/1 validity flag convention
	// uses elsewhere) - see wavefront_path_tracer.h's own member comment.
	reallocateDeviceBufferIfNeeded<float2>(d_upscaleMotionVectors_, upscaleMotionVectorsCapacity_, numPixels);
	if (reallocateDeviceBufferIfNeeded<float3>(d_upscaleOutput_, upscaleOutputCapacity_, numHighResPixels)) {
		CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_upscaleOutput_), 0, numHighResPixels * sizeof(float3), stream_));
	}
	// d_upscaleCurrent_ is pure per-frame scratch (upscale_infer writes
	// every cell unconditionally each call) - no memset needed on its own
	// resize, only d_upscaleHistory_ (the buffer actually READ) needs one.
	reallocateDeviceBufferIfNeeded<float4>(d_upscaleCurrent_, upscaleCurrentCapacity_, numHighResPixels);
	if (reallocateDeviceBufferIfNeeded<float4>(d_upscaleHistory_, upscaleHistoryCapacity_, numHighResPixels)) {
		CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_upscaleHistory_), 0, numHighResPixels * sizeof(float4), stream_));
		neuralUpscaleHistoryValid_ = false;  // stale/undefined content at the new size
	}
	if (reallocateDeviceBufferIfNeeded<UpscaleForwardCache>(d_upscaleForwardCache_, upscaleForwardCacheCapacity_, numHighResPixels)) {
		CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_upscaleForwardCache_), 0, numHighResPixels * sizeof(UpscaleForwardCache), stream_));
	}

	// This frame's own "which sub-cell is due" offset - see
	// neuralUpscaleWriteCounter_'s own member comment (wavefront_path_
	// tracer.h) for why this is a dedicated counter, not temporalJitterBaseIndex_.
	int dueCxInt = 0, dueCyInt = 0;
	wf_temporal_upscale_subcell(neuralUpscaleWriteCounter_, upscaleFactor, dueCxInt, dueCyInt);
	const unsigned int dueCx = (unsigned int)dueCxInt;
	const unsigned int dueCy = (unsigned int)dueCyInt;

	wf_launch_upscale_compute_motion_vectors(
		reinterpret_cast<const float4*>(d_worldPos_), reinterpret_cast<float2*>(d_upscaleMotionVectors_),
		width, height, prevRestirCamera_, stream_);

	// Train BEFORE infer - see wavefront_kernels_upscale.cu's own header
	// comment for why this ordering is load-bearing: upscale_train needs
	// d_upscaleForwardCache_'s contents from LAST frame's own upscale_infer
	// call, not this one's.
	//
	// Gated on neuralUpscaleHistoryValid_: d_upscaleForwardCache_ is
	// deliberately never cleared mid-session (see its own member comment),
	// so on a scene switch at an UNCHANGED resolution its cells keep the
	// PREVIOUS scene's cached predictions with kUpscaleCacheFlagValid still
	// set - upscale_train has no way to tell those apart from genuinely
	// current ones, and would otherwise train the network for up to one
	// full sub-cell period on (old-scene prediction, new-scene target)
	// pairs. invalidateRestirHistory() already resets
	// neuralUpscaleHistoryValid_ to false on exactly that transition (and
	// it's also false before the very first upscale_infer call ever runs),
	// so skipping training while it's false is both a correctness fix and,
	// as a side effect, avoids paying for the training+apply-gradients
	// dispatch on a frame where every sample would fail its own
	// per-cell validity check anyway.
	if (neuralUpscaleHistoryValid_) {
		CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_upscaleValidRecordCounter_), 0, sizeof(int), stream_));
		wf_launch_upscale_train(
			d_lowResFramebuffer, width, height,
			reinterpret_cast<const UpscaleForwardCache*>(d_upscaleForwardCache_), upscaleFactor, dueCx, dueCy,
			reinterpret_cast<const float*>(d_upscaleWeights_), reinterpret_cast<float*>(d_upscaleGradAccum_),
			reinterpret_cast<int*>(d_upscaleValidRecordCounter_),
			frameNumber_ ^ 0xACE1D00Du, kUpscaleTrainingRecordsPerFrame, stream_);
		const int numValidUpscaleRecords = readQueueSize(reinterpret_cast<int*>(d_upscaleValidRecordCounter_));
		++upscaleTrainingSteps_;
		wf_launch_upscale_apply_gradients(
			reinterpret_cast<float*>(d_upscaleWeights_), reinterpret_cast<float*>(d_upscaleAdamM_), reinterpret_cast<float*>(d_upscaleAdamV_),
			reinterpret_cast<float*>(d_upscaleGradAccum_), numValidUpscaleRecords, upscaleTrainingSteps_, stream_);
	}

	// Reads d_upscaleHistory_ (last call's own result) while writing
	// d_upscaleCurrent_ (this call's own result) - deliberately TWO
	// buffers, not one read-and-written in place - see d_upscaleCurrent_'s
	// own member comment (wavefront_path_tracer.h) for why aliasing them
	// would be a cross-thread race.
	wf_launch_upscale_infer(
		d_lowResFramebuffer, reinterpret_cast<const float4*>(d_worldPos_), reinterpret_cast<const float4*>(d_worldPosHistory_),
		reinterpret_cast<const float4*>(d_upscaleHistory_), reinterpret_cast<const float2*>(d_upscaleMotionVectors_),
		width, height, upscaleFactor,
		dueCx, dueCy,
		prevRestirCamera_, neuralUpscaleHistoryValid_, camera.origin,
		reinterpret_cast<const float*>(d_upscaleWeights_),
		reinterpret_cast<float3*>(d_upscaleOutput_), reinterpret_cast<float4*>(d_upscaleCurrent_), reinterpret_cast<UpscaleForwardCache*>(d_upscaleForwardCache_),
		stream_);

	// End-of-call history update - exactly d_worldPosHistory_'s own
	// read-then-overwrite-at-end-of-call pattern.
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_upscaleHistory_), reinterpret_cast<void*>(d_upscaleCurrent_),
							   (size_t)numHighResPixels * sizeof(float4), cudaMemcpyDeviceToDevice, stream_));

	neuralUpscaleHistoryValid_ = true;
	++neuralUpscaleWriteCounter_;
}

void WavefrontPathTracer::launchSvgf(float3* d_framebuffer, const float3* d_albedoAov, float3 cameraOrigin, const float* d_weightBuffer) {
	if (!svgfEnabled_) return;
	const int width = restirImageWidth_;
	const int height = restirImageHeight_;
	const int numPixels = width * height;
	if (numPixels <= 0) return;

	// All 4 kernels below launch on the SAME stream_, and every intermediate
	// buffer (d_svgfCurrent_, d_svgfPingPong_[]) is consumed only by the next
	// kernel on that same stream - never read back on the host in between -
	// so stream order alone already sequences them correctly, the same
	// "no internal sync needed" convention this file's own sibling
	// launchGiFinalize()/launchRestirSpatialReuse() already rely on. The
	// single sync the caller (render()) already issues right after
	// launchSvgf() returns is what actually matters (error-surfacing plus
	// ordering against the world-pos-history overwrite that follows it).

	// Kernel 1: temporal integrate - see wavefront_kernels_svgf.cu's own
	// header comment for the full 4-kernel pipeline. Reuses
	// prevRestirCamera_/d_worldPos_/d_worldPosHistory_ - the SAME
	// technique-agnostic camera-basis/world-pos bookkeeping DI/GI's own
	// temporal reuse already shares (this project's own SVGF plan).
	wf_launch_svgf_temporal_integrate(
		d_framebuffer,
		reinterpret_cast<const float4*>(d_worldPos_),
		reinterpret_cast<const GpuSvgfState*>(d_svgfHistory_),
		reinterpret_cast<const float4*>(d_worldPosHistory_),
		d_weightBuffer,
		prevRestirCamera_,
		svgfHistoryValid_,
		width, height,
		svgfTuning_.temporalAlpha, svgfTuning_.maxHistoryLength,
		reinterpret_cast<GpuSvgfState*>(d_svgfCurrent_),
		stream_);

	// Kernel 2: demodulate by albedo + prepare (bootstrap-filtered where
	// history is still short) variance for the A-trous sequence below.
	// Passes d_worldPos_ too, so the variance bootstrap can skip
	// miss/background neighbors instead of mixing their luminance
	// statistics into a silhouette-adjacent hit pixel's own variance.
	wf_launch_svgf_prepare_for_filter(
		reinterpret_cast<const GpuSvgfState*>(d_svgfCurrent_),
		d_albedoAov,
		reinterpret_cast<const float4*>(d_worldPos_),
		width, height,
		svgfTuning_.varianceBootstrapFrames, svgfTuning_.varianceBootstrapRadius, svgfTuning_.minAlbedo,
		reinterpret_cast<float4*>(d_svgfPingPong_[0]),
		stream_);

	// Kernel 3: A-trous wavelet filter, run svgfTuning_.atrousPasses times
	// with doubling step sizes (1,2,4,...), ping-ponging between the two
	// scratch buffers. The pass COUNT/step-size schedule lives here (host
	// side) - the kernel itself is stateless per pass (just takes whatever
	// stepSize this call gives it), so there is no cross-file constant to
	// keep in sync, unlike this file's own kRestirTemporalMaxM-style
	// constants.
	int src = 0, dst = 1;
	int stepSize = 1;
	for (int pass = 0; pass < svgfTuning_.atrousPasses; ++pass) {
		wf_launch_svgf_atrous_pass(
			reinterpret_cast<const float4*>(d_svgfPingPong_[src]),
			reinterpret_cast<const float4*>(d_worldPos_),
			reinterpret_cast<const float3*>(denoiserResources_.normalAov),
			cameraOrigin,
			width, height,
			stepSize,
			svgfTuning_.sigmaNormal, svgfTuning_.sigmaDepth, svgfTuning_.sigmaLuminance, svgfTuning_.atrousRadius,
			reinterpret_cast<float4*>(d_svgfPingPong_[dst]),
			stream_);
		std::swap(src, dst);
		stepSize *= 2;
	}

	// Kernel 4: re-multiply by albedo, write the final result into the real
	// framebuffer. `src` (not `dst`) holds the LAST pass's own output after
	// the swap above - unless atrousPasses==0, in which case src is still 0
	// and d_svgfPingPong_[0] correctly holds svgf_prepare_for_filter's own
	// (unfiltered) output.
	wf_launch_svgf_finalize(
		reinterpret_cast<const float4*>(d_svgfPingPong_[src]),
		d_albedoAov,
		numPixels,
		svgfTuning_.minAlbedo,
		d_framebuffer,
		stream_);

	// End-of-call history update - this frame's own temporally-integrated
	// state (d_svgfCurrent_, NOT the spatially-filtered d_svgfPingPong_
	// result - see GpuSvgfState::color's own comment for why the two must
	// stay separate) becomes next frame's history, mirroring DI/GI's own
	// identical end-of-render() copy.
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_svgfHistory_),
							   reinterpret_cast<void*>(d_svgfCurrent_),
							   static_cast<size_t>(numPixels) * sizeof(GpuSvgfState),
							   cudaMemcpyDeviceToDevice, stream_));
	svgfHistoryValid_ = true;
}

void WavefrontPathTracer::launchAccumulateMiss(int numMiss, float3* d_framebuffer, float3 backgroundColor,
												GpuSkyDistribution skyDist, GpuPortalLight portalLight, float maxComponentValue) {
	if (numMiss == 0) return;

	WorkQueue<MissWorkItem> mq;
	mq.items    = reinterpret_cast<MissWorkItem*>(d_missItems_);
	mq.counter  = reinterpret_cast<int*>(d_missCounter_);
	mq.capacity = queueCapacity_;

	wf_launch_accumulate_miss(mq, numMiss, d_framebuffer, backgroundColor, skyDist, portalLight, maxComponentValue,
		reinterpret_cast<float3*>(denoiserResources_.albedoAov),
		reinterpret_cast<float3*>(denoiserResources_.normalAov),
		reinterpret_cast<float4*>(d_worldPos_),
		stream_);
}

void WavefrontPathTracer::launchAccumulateShadow(
	int numShadow, const float* d_transmittance, float3* d_framebuffer, float maxComponentValue)
{
	if (numShadow == 0) return;

	WorkQueue<ShadowRayWorkItem> sq;
	sq.items    = reinterpret_cast<ShadowRayWorkItem*>(d_shadowItems_);
	sq.counter  = reinterpret_cast<int*>(d_shadowCounter_);
	sq.capacity = shadowQueueCapacity_;

	wf_launch_accumulate_shadow(sq, numShadow, d_transmittance, d_framebuffer, maxComponentValue, stream_,
								 reinterpret_cast<GpuGiSample*>(d_giCandidateOut_));
}

void WavefrontPathTracer::launchResolveBssrdfExit(
	int numExit,
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
	float3* d_framebuffer, float3 skyColor, float shadowRayEpsilon,
	GpuSkyDistribution skyDist, GpuPortalLight portalLight, float maxComponentValue)
{
	if (numExit == 0) return;

	WorkQueue<BssrdfExitWorkItem> eq;
	eq.items    = reinterpret_cast<BssrdfExitWorkItem*>(d_exitItems_);
	eq.counter  = reinterpret_cast<int*>(d_exitCounter_);
	eq.capacity = queueCapacity_;

	WorkQueue<RayWorkItem> nq;
	nq.items    = reinterpret_cast<RayWorkItem*>(d_nextRayItems_);
	nq.counter  = reinterpret_cast<int*>(d_nextRayCounter_);
	nq.capacity = queueCapacity_;

	WorkQueue<ShadowRayWorkItem> sq;
	sq.items    = reinterpret_cast<ShadowRayWorkItem*>(d_shadowItems_);
	sq.counter  = reinterpret_cast<int*>(d_shadowCounter_);
	sq.capacity = shadowQueueCapacity_;

	wf_launch_resolve_bssrdf_exit(eq, numExit, nq, sq, d_framebuffer,
		d_spheres, numSpheres,
		d_quads, numQuads,
		d_triangles, numTriangles,
		d_bilinearPatches, numBilinearPatches,
		d_disks, numDisks,
		d_cylinders, numCylinders,
		d_materials, numMaterials,
		d_lightIndices, d_lightKinds, d_aliasTable, numLights,
		d_punctualLights, numPunctualLights,
		reinterpret_cast<const TextureData*>(d_textures_),
		reinterpret_cast<const unsigned char*>(d_texturePixels_),
		skyColor, shadowRayEpsilon, skyDist, portalLight, maxComponentValue,
		reinterpret_cast<GpuGiOriginContext*>(d_giOriginContext_),
		reinterpret_cast<GpuGiSample*>(d_giCandidateOut_),
		buildLightBvhContext(),
		stream_);
}

void WavefrontPathTracer::launchNormalizeFramebuffer(
	unsigned int numPixels, const float* d_weightBuffer, float3* d_framebuffer)
{
	wf_launch_normalize_framebuffer(numPixels, d_weightBuffer, d_framebuffer, stream_);
}

void WavefrontPathTracer::launchNormalizeAovBuffers(
	unsigned int numPixels, float3* d_albedoBuffer, float3* d_normalBuffer, unsigned int samplesPerPixel)
{
	wf_launch_normalize_aov_buffers(numPixels, d_albedoBuffer, d_normalBuffer, samplesPerPixel, stream_);
}

// ============================================================================
// --denoise support - own copy of OptiXRenderer's denoiser/AOV-buffer
// lifecycle (optix_renderer_render.cpp) - see setDenoiseEnabled()'s own
// comment (wavefront_path_tracer.h) for why this isn't shared instead.
// ============================================================================

void WavefrontPathTracer::ensureAovBuffers(unsigned int width, unsigned int height) {
	::ensureAovBuffers(denoiserResources_, width, height);
}

void WavefrontPathTracer::destroyAovBuffers() noexcept {
	::destroyAovBuffers(denoiserResources_);
}

void WavefrontPathTracer::destroyDenoiser() noexcept {
	destroyDenoiserResources(denoiserResources_);
}

// See optix_denoiser.h's runDenoiser() for the full "why" behind every
// step - this backend just supplies its own denoiserResources_/context_/
// stream_, same as OptiXRenderer::denoise() (optix_renderer_render.cpp).
bool WavefrontPathTracer::denoise(CUdeviceptr d_buffer, unsigned int width, unsigned int height,
	CUdeviceptr d_albedo, CUdeviceptr d_normal) {
	return runDenoiser(denoiserResources_, context_, stream_, d_buffer, width, height,
		d_albedo, d_normal, denoiseBlend_, "[Wavefront]");
}


}  // namespace optix_renderer
