// wavefront_kernels_nrc.cu
// Neural Radiance Cache (NRC) training pipeline kernels (Live Preview only).
// See this project's own plan for the full design.
//
// Mirrors the world-space irradiance probe cache's own established shape
// (wavefront_probe_cache.h/wavefront_kernels_restir.cu's probe_cache_shade/
// probe_cache_accumulate): a dedicated, resolution-independent, fixed-size
// side pipeline run once per render() call, AFTER the main per-sample/
// per-depth loop has fully finished (so it's safe to reuse the main
// pipeline's own rayQueue/nextRayQueue/hitQueue/simpleHitQueue/
// dielectricHitQueue/missQueue/shadowQueue/transmittance buffers - see
// WavefrontPathTracer::launchNrcTrainingUpdate()'s own comment,
// wavefront_path_tracer.cpp, for why this is safe).
//
// Unlike the probe cache (a single OptiX raygen + one CUDA shading kernel),
// NRC's training paths need to BOUNCE (to gather multi-vertex training
// records for the bootstrap target) - this file's kernels are plain CUDA
// (`extern "C" __global__`, NOT OptiX raygen), reusing intersectPipeline_/
// intersectSBT_ (the SAME pipeline the main per-bounce loop already uses)
// for the actual intersection tracing, orchestrated host-side exactly like
// the main per-bounce loop's own depth-by-depth structure.
//
// v1 material scope: training paths only visit/continue through Lambertian
// and RoughMetal vertices (see nrc_training_shade_simple/_full below) -
// any other material type a training ray hits simply terminates that path
// (no record written for the unsupported vertex, and see
// nrc_bootstrap_and_train's own comment for how a missing "next" record is
// handled at training time). Conductor (complex Fresnel) is deferred to a
// future version.
//
// Two-launch-family design per render() call:
//   Phase A (this file's nrc_generate_training_rays +
//   nrc_training_shade_simple/_full): traced depth-by-depth, forward in
//   depth, writing one NrcTrainingRecord per visited Lambertian/RoughMetal
//   vertex into a flat, depth-major array (WavefrontPathTracer::
//   d_nrcTrainingRecords_, capacity kNrcTrainingRecordCapacity).
//   Phase B (this file's nrc_bootstrap_and_train): ONE single flat launch
//   over every record, run after Phase A fully completes. Unlike an
//   earlier version of this design (see this project's own plan), Phase B
//   does NOT need to run in reverse depth order or between per-depth syncs:
//   each record's bootstrap target reads ONLY (a) its own Phase-A-written
//   fields and (b) its successor's Phase-A-written INPUT FEATURES (never
//   the successor's own computed target or gradient) against this frame's
//   FIXED, not-yet-updated weights - there is no cross-record computation
//   dependency within Phase B at all, only within Phase A's own bounce
//   tracing (depth d+1 can only be traced once depth d's own scattered
//   direction is known).
//   Apply step (nrc_apply_gradients): one final tiny kernel, run after
//   Phase B, that averages the accumulated gradient and applies Adam - the
//   ONLY kernel that ever writes the live weight buffer.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "wavefront_device_helpers.h"
#include "wavefront_nrc_types.h"
#include "wavefront_nrc_encoding.h"
#include "wavefront_nrc_mlp.h"

// ---------------------------------------------------------------------------
// Shared NEE + record-write helper - used by both nrc_training_shade_simple
// (Lambertian) and nrc_training_shade_full (RoughMetal) below. Writes
// `record`'s own emission (folded in directly, no occlusion needed - mirrors
// probe_cache_shade's identical emission handling), draws one NEE light
// sample, evaluates the CALLER-SUPPLIED `brdfFn(toLight)` (the material's own
// f(wo,wi) at the sampled light direction - see this function's own
// `brdfFn` parameter comment), and - if a valid, non-degenerate sample was
// drawn - pushes a real spectral shadow ray tagged isNrcTrainingRay=true so
// accumulate_shadow (wavefront_kernels_accumulate.cu) redirects its resolved
// Ld into record->directRadiance once occlusion is resolved. Does NOT set
// position/outgoingDir/albedo/roughness/flags/nextRecordIndex - the caller
// does that itself before/after calling this, since those differ between
// the Lambertian/RoughMetal cases.
//
// `brdfFn` is a per-material BSDF evaluation callback taking the sampled
// light direction (world space) and returning f(wo,wi) as a plain RGB
// (NOT yet multiplied by cosTheta - that's applied uniformly below). An
// earlier version of this function hardcoded a Lambertian albedo/pi formula
// unconditionally, including for RoughMetal training vertices - a real,
// confirmed bug (RoughMetal's real BRDF is GGX-based, not diffuse), fixed
// by making the BRDF evaluation the caller's own responsibility, matching
// how wf_finish_material_scatter's own real NEE block (wavefront_device_
// helpers.h) dispatches to a per-material evalGlossyF/albedo-only formula
// rather than assuming one BRDF shape for every material type.
// ---------------------------------------------------------------------------
template <typename BrdfFn>
__device__ __forceinline__ void wf_nrc_do_nee(
	float3 hitPoint, float3 normal, unsigned int& seed, float time,
	int recordIdx, float3 emission,
	const SphereData* spheres, const QuadData* quads, const TriangleData* triangles,
	const BilinearPatchData* bilinearPatches, const DiskData* disks, const CylinderData* cylinders,
	const MaterialData* materials, const int* lightIndices, const GpuLightKind* lightKinds,
	const GpuAliasEntry* aliasTable, unsigned int numLights,
	const TextureData* textures, const unsigned char* texturePixels,
	WfLightBvhContext lightBvh, float shadowRayEpsilon,
	NrcTrainingRecord* records,
	WorkQueue<ShadowRayWorkItem>& shadowQueue,
	BrdfFn brdfFn)
{
	NrcTrainingRecord& record = records[recordIdx];
	record.directRadiance.x += emission.x;
	record.directRadiance.y += emission.y;
	record.directRadiance.z += emission.z;

	GpuLightSample cand;
	float3 toLight = make_float3(0.0f, 0.0f, 0.0f);
	float  maxDist = 0.0f, lightPdf = 0.0f;
	float3 rawEmission = make_float3(0.0f, 0.0f, 0.0f);
	if (!wf_generate_restir_candidate(hitPoint, seed, time,
			spheres, quads, triangles, bilinearPatches, disks, cylinders,
			materials, lightIndices, lightKinds, aliasTable, numLights,
			textures, texturePixels, cand, toLight, maxDist, lightPdf, rawEmission,
			lightBvh)) {
		return;
	}
	const float cosTheta = dot(normal, toLight);
	if (!(cosTheta > 0.0f) || !(lightPdf > 1e-6f)) return;

	const float3 brdfVal = brdfFn(toLight);
	if (brdfVal.x <= 0.0f && brdfVal.y <= 0.0f && brdfVal.z <= 0.0f) return;

	using SS  = SampledSpectrum<kWFNWavelengths>;
	using SWL = SampledWavelengths<kWFNWavelengths>;
	const SWL swl = SWL::SampleVisible(wf_rand(seed));

	const SS lightSpec = wf_lift_rgb_to_spectrum(rawEmission, swl, /*isIlluminant=*/true);
	const SS brdfSpec  = wf_lift_rgb_to_spectrum(brdfVal, swl, /*isIlluminant=*/false);
	const SS Ld = (cosTheta / lightPdf) * brdfSpec * lightSpec;
	if (!(bool)Ld) return;

	ShadowRayWorkItem sr;
	sr.origin    = hitPoint + shadowRayEpsilon * normal + shadowRayEpsilon * normalize(toLight);
	sr.direction = toLight;
	sr.tMax      = maxDist - 0.002f;
	for (int i = 0; i < kWFNWavelengths; ++i) {
		sr.Ld[i] = Ld[i];
		sr.wavelengths[i] = swl.lambda[i];
		sr.wavelength_pdfs[i] = swl.pdf[i];
	}
	sr.pixelIndex       = recordIdx;
	sr.time             = time;
	sr.isGiCandidate    = false;
	sr.isProbeCacheRay  = false;
	sr.isNrcTrainingRay = true;
	sr.seed             = seed;
	shadowQueue.push(sr);
}

// ---------------------------------------------------------------------------
// nrc_generate_training_rays -- one thread per training path
// (kNrcTrainingPathsPerFrame threads). Seeds a fresh camera ray at a
// uniformly-random continuous (u,v) over the WHOLE image (not tied to any
// real pixel - training paths never write to the displayed framebuffer, so
// there's no reconstruction-filter/pixel-center precision to preserve, only
// "sample the visible scene's own radiance distribution" the way the paper
// itself seeds its own training paths from camera pixels). Reuses
// wf_generate_primary_ray() (wavefront_device_helpers.h) - the exact same
// per-camera-kind ray formula generate_camera_rays (wavefront_kernels_
// camera.cu) already uses for real pixels.
// ---------------------------------------------------------------------------
extern "C" __global__ void nrc_generate_training_rays(
	WorkQueue<RayWorkItem> rayQueue,
	int numPaths,
	GpuCameraParams camera,
	unsigned int frameNumber)
{
	const int pathSlot = blockIdx.x * blockDim.x + threadIdx.x;
	if (pathSlot >= numPaths) return;

	unsigned int seed = wf_pcg(wf_pcg((unsigned int)pathSlot ^ 0xA5A5A5A5u) ^ frameNumber);
	const float u = wf_rand(seed);
	const float v = wf_rand(seed);

	RayWorkItem item;
	float weight;
	wf_generate_primary_ray(camera, u, v, seed, item.origin, item.direction, weight);
	for (int i = 0; i < kWFNWavelengths; ++i) {
		item.throughput[i] = 1.0f;
		item.radiance[i] = 0.0f;
		item.wavelengths[i] = 0.0f;
		item.wavelength_pdfs[i] = 0.0f;
	}
	item.time = camera.motionBlurEnabled ? wf_rand(seed) : 0.0f;
	item.seed = seed;
	// pathSlot, NOT a screen pixel - carried forward into HitWorkItem by the
	// shared closest-hit program's own wf_carry_ray_state(), read back by
	// nrc_training_shade_simple/_full below to address this path's own
	// per-depth record slot (depth*kNrcTrainingPathsPerFrame + pathSlot).
	item.pixelIndex = pathSlot;
	item.depth = 0;
	item.specular_bounce = 1;
	item.any_nonspecular = 0;
	item.etaScale = 1.0f;
	item.filterWeight = 1.0f;
	item.brdf_pdf = 0.0f;
	item.tMin = 0.001f;
	item.tMax = 1e30f;
	rayQueue.push(item);
}

// ---------------------------------------------------------------------------
// nrc_training_shade_simple -- processes simpleHitQueue (Lambertian/Metal
// hits - see WavefrontQueues::simpleHitQueue's own routing comment,
// wavefront_raygen.h). Lambertian gets a full record + cosine-weighted
// hemisphere continuation; Metal (a perfect mirror, no diffuse/glossy
// reflectance an NRC record could usefully describe) simply terminates
// this training path with no record written, matching the probe cache's
// own Lambertian-only scope for the analogous case.
// ---------------------------------------------------------------------------
extern "C" __global__ void nrc_training_shade_simple(
	WorkQueue<HitWorkItem> simpleHitQueue,
	int numHits,
	int depth, int maxDepth,
	const MaterialData* materials,
	const SphereData* spheres, const QuadData* quads, const TriangleData* triangles,
	const BilinearPatchData* bilinearPatches, const DiskData* disks, const CylinderData* cylinders,
	const TextureData* textures, const unsigned char* texturePixels,
	const int* lightIndices, const GpuLightKind* lightKinds, const GpuAliasEntry* aliasTable, unsigned int numLights,
	WfLightBvhContext lightBvh, float shadowRayEpsilon,
	NrcTrainingRecord* records,
	int* validRecordCounter,
	WorkQueue<RayWorkItem> nextRayQueue,
	WorkQueue<ShadowRayWorkItem> shadowQueue)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numHits || idx >= simpleHitQueue.capacity) return;

	const HitWorkItem& h = simpleHitQueue.items[idx];
	const MaterialData& mat = materials[h.materialIdx];
	if (mat.type != MaterialType::Lambertian) return;  // Metal: terminate, no record - see this kernel's own header comment

	const int recordIdx = depth * kNrcTrainingPathsPerFrame + h.pixelIndex;
	NrcTrainingRecord& record = records[recordIdx];
	atomicAdd(validRecordCounter, 1);
	record.position = h.hitPoint;
	record.normal = h.normal;
	record.roughness = kNrcLambertianRoughnessSentinel;
	record.albedo = mat.albedo;
	record.flags = kNrcRecordFlagValid;
	record.nextRecordIndex = (depth + 1 < maxDepth) ? (depth + 1) * kNrcTrainingPathsPerFrame + h.pixelIndex : -1;

	unsigned int seed = h.seed;
	const float3 lambertianAlbedo = mat.albedo;
	wf_nrc_do_nee(h.hitPoint, h.normal, seed, h.time, recordIdx, mat.emission,
				  spheres, quads, triangles, bilinearPatches, disks, cylinders,
				  materials, lightIndices, lightKinds, aliasTable, numLights,
				  textures, texturePixels, lightBvh, shadowRayEpsilon, records, shadowQueue,
				  [lambertianAlbedo](float3 /*toLight*/) -> float3 {
					  const float invPi = 1.0f / 3.14159265f;
					  return make_float3(lambertianAlbedo.x * invPi, lambertianAlbedo.y * invPi, lambertianAlbedo.z * invPi);
				  });

	if (record.nextRecordIndex < 0) return;  // last allowed depth - no continuation traced

	// Cosine-weighted hemisphere sample via the sphere-offset trick (Malley's
	// method) - the SAME one-line formula every real Lambertian-scatter call
	// site in this codebase actually uses (e.g. wavefront_kernels_materials.cu/
	// wavefront_kernels_materials_simple.cu's own Lambertian cases), not the
	// disk-sample+tangent-frame reconstruction an earlier version of this
	// function used instead (a different, needlessly more expensive way to
	// draw the same distribution - its own comment claiming it "matches
	// every other call site" was incorrect). throughputToNext == albedo
	// exactly for this sampling scheme (brdf=albedo/pi, pdf=cosTheta/pi,
	// brdf*cosTheta/pdf == albedo - the same identity wf_finish_material_
	// scatter's own Lambertian case relies on).
	float3 scatterDir = normalize(h.normal + wf_rand_unit(seed));
	if (wf_near_zero(scatterDir)) scatterDir = h.normal;
	record.outgoingDir = scatterDir;
	record.throughputToNext = mat.albedo;

	RayWorkItem next;
	next.origin = h.hitPoint + 0.001f * h.normal;
	next.direction = scatterDir;
	for (int i = 0; i < kWFNWavelengths; ++i) {
		next.throughput[i] = 1.0f; next.radiance[i] = 0.0f;
		next.wavelengths[i] = 0.0f; next.wavelength_pdfs[i] = 0.0f;
	}
	next.seed = seed;
	next.pixelIndex = h.pixelIndex;
	next.depth = depth + 1;
	next.specular_bounce = 0;
	next.any_nonspecular = 1;
	next.etaScale = 1.0f;
	next.filterWeight = 1.0f;
	next.brdf_pdf = 0.0f;
	next.tMin = 0.001f;
	next.tMax = 1e30f;
	next.time = h.time;
	nextRayQueue.push(next);
}

// ---------------------------------------------------------------------------
// nrc_training_shade_full -- processes hitQueue (every material type NOT
// routed to simpleHitQueue/dielectricHitQueue - see this file's own header
// comment for the v1 material scope). Only RoughMetal gets a record +
// continuation; every other material type terminates this training path
// with no record, exactly like Metal does in nrc_training_shade_simple.
// ---------------------------------------------------------------------------
extern "C" __global__ void nrc_training_shade_full(
	WorkQueue<HitWorkItem> hitQueue,
	int numHits,
	int depth, int maxDepth,
	const MaterialData* materials,
	const SphereData* spheres, const QuadData* quads, const TriangleData* triangles,
	const BilinearPatchData* bilinearPatches, const DiskData* disks, const CylinderData* cylinders,
	const TextureData* textures, const unsigned char* texturePixels,
	const int* lightIndices, const GpuLightKind* lightKinds, const GpuAliasEntry* aliasTable, unsigned int numLights,
	WfLightBvhContext lightBvh, float shadowRayEpsilon,
	NrcTrainingRecord* records,
	int* validRecordCounter,
	WorkQueue<RayWorkItem> nextRayQueue,
	WorkQueue<ShadowRayWorkItem> shadowQueue)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numHits || idx >= hitQueue.capacity) return;

	const HitWorkItem& h = hitQueue.items[idx];
	const MaterialData& mat = materials[h.materialIdx];
	if (mat.type != MaterialType::RoughMetal) return;  // every other type: terminate, no record

	const float alpha = wf_glossy_alpha(mat, /*do_regularize=*/false);
	float3 n = h.normal;
	float3 up = (fabsf(n.x) > 0.9f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
	float3 tangent = normalize(cross(up, n));
	float3 bitangent = cross(n, tangent);
	float3 wiWorld = -normalize(h.rayDir);
	const float wi_x = dot(wiWorld, tangent), wi_y = dot(wiWorld, bitangent), wi_z = dot(wiWorld, n);
	if (wi_z <= 0.0f) return;  // grazing/backfacing - terminate, no record (same guard as the main render path's own RoughMetal case)

	const int recordIdx = depth * kNrcTrainingPathsPerFrame + h.pixelIndex;
	NrcTrainingRecord& record = records[recordIdx];
	atomicAdd(validRecordCounter, 1);
	record.position = h.hitPoint;
	record.normal = n;
	record.roughness = alpha;
	record.albedo = mat.albedo;
	record.flags = kNrcRecordFlagValid;
	record.nextRecordIndex = (depth + 1 < maxDepth) ? (depth + 1) * kNrcTrainingPathsPerFrame + h.pixelIndex : -1;

	unsigned int seed = h.seed;
	// GGX microfacet reflection BRDF, f(wo,wi) = D(wm)*G(wo,wi)/(4*|cosTheta_o|*|cosTheta_i|),
	// times RoughMetal's own flat albedo tint (no complex Fresnel - see
	// MaterialType::RoughMetal's own comment, wavefront_kernels_materials.cu).
	// An earlier version of this function passed wf_nrc_do_nee a plain
	// Lambertian albedo/pi formula unconditionally - a real, confirmed bug
	// (see wf_nrc_do_nee's own header comment): RoughMetal's real BRDF is
	// this GGX formula, not diffuse, and using the wrong one here corrupted
	// every RoughMetal training record's own direct-lighting target.
	const TrowbridgeReitz<float> dist(alpha, alpha);
	const float3 metalAlbedo = mat.albedo;
	wf_nrc_do_nee(h.hitPoint, n, seed, h.time, recordIdx, mat.emission,
				  spheres, quads, triangles, bilinearPatches, disks, cylinders,
				  materials, lightIndices, lightKinds, aliasTable, numLights,
				  textures, texturePixels, lightBvh, shadowRayEpsilon, records, shadowQueue,
				  [dist, tangent, bitangent, n, wi_x, wi_y, wi_z, metalAlbedo](float3 toLight) -> float3 {
					  const float lx = dot(toLight, tangent), ly = dot(toLight, bitangent), lz = dot(toLight, n);
					  if (lz <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
					  float wmx = wi_x + lx, wmy = wi_y + ly, wmz = wi_z + lz;
					  const float wmLen = sqrtf(wmx * wmx + wmy * wmy + wmz * wmz);
					  if (wmLen < 1e-8f) return make_float3(0.0f, 0.0f, 0.0f);
					  wmx /= wmLen; wmy /= wmLen; wmz /= wmLen;
					  const float denom = 4.0f * fabsf(wi_z) * fabsf(lz);
					  if (denom < 1e-8f) return make_float3(0.0f, 0.0f, 0.0f);
					  const float brdfScalar = dist.D(wmx, wmy, wmz) * dist.G(wi_x, wi_y, wi_z, lx, ly, lz) / denom;
					  return make_float3(metalAlbedo.x * brdfScalar, metalAlbedo.y * brdfScalar, metalAlbedo.z * brdfScalar);
				  });

	// GGX VNDF sample - plain (non-guided) TrowbridgeReitz sampling, the
	// exact same math the main render path's own RoughMetal case uses
	// (guidingHistograms=nullptr degenerates wf_sample_guided_glossy to
	// plain, unmodified GGX/VNDF sampling - see that function's own header
	// comment, wavefront_device_helpers.h).
	WfGuidedGlossySample sample = wf_sample_guided_glossy(
		dist, wi_x, wi_y, wi_z, tangent, bitangent, n, alpha, alpha,
		h.hitPoint, GpuProbeGridMeta{}, /*guidingHistograms=*/nullptr, /*guidingProbes=*/nullptr, seed);
	if (!sample.scattered) {
		// This record's own nextRecordIndex was already set to a real slot
		// above (before sampling was known to succeed) - reset it to -1 so
		// nrc_bootstrap_and_train correctly sees "no continuation" instead
		// of a dangling index into a slot nothing will ever write this
		// frame (nrc_bootstrap_and_train's own hasValidNext check already
		// treats an unwritten slot as invalid via its Valid-flag check, so
		// this reset is defense-in-depth, not a currently-reachable defect).
		record.nextRecordIndex = -1;
		return;
	}
	if (record.nextRecordIndex < 0) return;

	const float3 scatterDir = normalize(sample.wo_x * tangent + sample.wo_y * bitangent + sample.wo_z * n);
	record.outgoingDir = scatterDir;
	record.throughputToNext = make_float3(mat.albedo.x * sample.weight, mat.albedo.y * sample.weight, mat.albedo.z * sample.weight);

	RayWorkItem next;
	next.origin = h.hitPoint + 0.001f * n;
	next.direction = scatterDir;
	for (int i = 0; i < kWFNWavelengths; ++i) {
		next.throughput[i] = 1.0f; next.radiance[i] = 0.0f;
		next.wavelengths[i] = 0.0f; next.wavelength_pdfs[i] = 0.0f;
	}
	next.seed = seed;
	next.pixelIndex = h.pixelIndex;
	next.depth = depth + 1;
	next.specular_bounce = dist.EffectivelySmooth() ? 1 : 0;
	next.any_nonspecular = dist.EffectivelySmooth() ? 0 : 1;
	next.etaScale = 1.0f;
	next.filterWeight = 1.0f;
	next.brdf_pdf = sample.pdf;
	next.tMin = 0.001f;
	next.tMax = 1e30f;
	next.time = h.time;
	nextRayQueue.push(next);
}

// ---------------------------------------------------------------------------
// nrc_bootstrap_and_train -- Phase B, ONE flat launch over every record (see
// this file's own header comment for why no per-depth ordering/sync is
// needed). One thread per record. Skips any record without
// kNrcRecordFlagValid set (never visited by Phase A this frame - the exact
// same "0 means untouched" convention GpuProbe::numRaysEverTraced==0 uses).
// ---------------------------------------------------------------------------
// Queried at vertex i's own (position, outgoingDir, normal, roughness,
// albedo), the network must predict ONLY the radiance arriving at i via
// CONTINUING the path beyond i (>=1 more bounce) - NOT vertex i's own
// direct-lighting term. This is what makes the render-path substitution
// (wf_finish_material_scatter, wavefront_device_helpers.h) additive with -
// not double-counting against - the real NEE that function ALREADY performs
// for the current hit before the NRC query ever runs: real NEE supplies the
// zero-bounce direct term, NRC supplies everything from one bounce onward,
// exactly mirroring how the world-space probe cache's own SH-L1 value is
// genuinely one-bounce-indirect-only (built from a SEPARATE secondary ray
// traced from the probe's own position - see probe_grid_types.h's own
// header comment) rather than the queried point's own total radiance.
//
// An earlier version of this kernel trained record i's own target as
// `directRadiance_i + throughputToNext_i * network(i+1)` - i.e. vertex i's
// OWN total outgoing radiance, direct term included. That is what the REAL
// paper's network represents, but substituting it at the SAME vertex whose
// own real NEE had already run (this codebase's own integration shape,
// unlike a reference implementation that skips real NEE whenever the cache
// is about to fire) silently double-counted that vertex's own direct light.
// Fixed here at the training-target level instead of restructuring the
// render path's own NEE/RR/substitution ordering: record i's target is
// computed from vertex i+1's OWN direct term plus vertex i+1's own
// bootstrapped continuation (querying the network at vertex i+2), i.e. a
// two-hop lookup ahead of i rather than one.
extern "C" __global__ void nrc_bootstrap_and_train(
	const NrcTrainingRecord* records,
	int numRecords,
	const float* weights,
	float* gradAccum,
	float3 aabbMin, float3 aabbExtent)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numRecords) return;
	const NrcTrainingRecord& record = records[i];
	if ((record.flags & kNrcRecordFlagValid) == 0) return;

	// No valid vertex i+1 - no continuation exists beyond i (dead end,
	// escaped the scene, or the last allowed training-path depth). Target
	// is a plain 0: "querying the cache here should predict no further
	// contribution", a reasonable approximation for a vertex this training
	// pass never saw continue further (see this project's own plan's
	// discussion of the finite training-depth limitation).
	float3 target = make_float3(0.0f, 0.0f, 0.0f);
	const bool hasValidNext = record.nextRecordIndex >= 0 &&
		(records[record.nextRecordIndex].flags & kNrcRecordFlagValid) != 0;
	if (hasValidNext) {
		const NrcTrainingRecord& next = records[record.nextRecordIndex];
		float3 nextTotal = next.directRadiance;
		const bool hasValidNextNext = next.nextRecordIndex >= 0 &&
			(records[next.nextRecordIndex].flags & kNrcRecordFlagValid) != 0;
		if (hasValidNextNext) {
			const NrcTrainingRecord& nextNext = records[next.nextRecordIndex];
			float nextNextFeatures[kNrcInputDim];
			wf_nrc_encode_features(nextNext.position, nextNext.outgoingDir, nextNext.normal,
									nextNext.roughness, nextNext.albedo,
									aabbMin, aabbExtent, nextNextFeatures);
			NrcForwardCache nextNextCache;
			wf_nrc_forward(weights, nextNextFeatures, nextNextCache);
			nextTotal.x += next.throughputToNext.x * nextNextCache.out[0];
			nextTotal.y += next.throughputToNext.y * nextNextCache.out[1];
			nextTotal.z += next.throughputToNext.z * nextNextCache.out[2];
		}
		target.x = record.throughputToNext.x * nextTotal.x;
		target.y = record.throughputToNext.y * nextTotal.y;
		target.z = record.throughputToNext.z * nextTotal.z;
	}

	float features[kNrcInputDim];
	wf_nrc_encode_features(record.position, record.outgoingDir, record.normal, record.roughness, record.albedo,
							aabbMin, aabbExtent, features);
	NrcForwardCache cache;
	wf_nrc_forward(weights, features, cache);

	const float targetArr[kNrcOutputDim] = {target.x, target.y, target.z};
	float dLossDOut[kNrcOutputDim];
	wf_nrc_loss_gradient(cache.out, targetArr, dLossDOut);
	wf_nrc_backward(weights, cache, dLossDOut, gradAccum);
}

// ---------------------------------------------------------------------------
// nrc_apply_gradients -- one thread per weight (kNrcNumWeights threads),
// the ONLY kernel that ever writes the live weight buffer - see this
// project's own plan's "gradient accumulation + single serial apply step"
// design. Averages gradAccum by the number of records that actually
// contributed this frame, applies Adam, then zeroes gradAccum for next
// frame's accumulation.
// ---------------------------------------------------------------------------
extern "C" __global__ void nrc_apply_gradients(
	float* weights, float* adamM, float* adamV, float* gradAccum,
	int numContributingRecords, int stepCount)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= kNrcNumWeights) return;
	if (numContributingRecords > 0) {
		const float grad = gradAccum[i] / (float)numContributingRecords;
		wf_nrc_adam_step(weights[i], adamM[i], adamV[i], grad, stepCount);
	}
	gradAccum[i] = 0.0f;
}

// ---------------------------------------------------------------------------
// nrc_reset_weights -- re-randomizes weights to small values and zeroes
// Adam's moment buffers, called once on scene load (see
// OptiXRenderer::buildProbeGrid()'s own call site) - both buffers, not just
// weights, since stale momentum from a different scene's gradient
// statistics would miscalibrate Adam's variance normalization even with
// freshly-randomized weights (see this project's own plan's Risks section).
// Small uniform range (matches a standard small-MLP initialization scale -
// large enough to break symmetry between units, small enough that the
// softplus output starts near its linear-ish region rather than saturating
// immediately).
// ---------------------------------------------------------------------------
extern "C" __global__ void nrc_reset_weights(float* weights, float* adamM, float* adamV, unsigned int seed)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= kNrcNumWeights) return;
	unsigned int s = wf_pcg(wf_pcg((unsigned int)i) ^ seed);
	weights[i] = (wf_rand(s) - 0.5f) * 0.2f;
	adamM[i] = 0.0f;
	adamV[i] = 0.0f;
}
