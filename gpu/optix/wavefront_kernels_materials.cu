// wavefront_kernels_materials.cu
// CUDA compute kernel: evaluate_materials, the full-switch material-shading
// kernel for the wavefront GPU path tracer (Kernel 2 of the original
// wavefront_kernels.cu, split out so an edit to this kernel no longer
// forces recompiling every other kernel family - see
// wavefront_device_helpers.h's own header comment for why the two helpers
// this split needed (wf_finish_material_scatter, wf_lift_rgb_to_spectrum)
// live there instead of in a 7th split file).

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "wavefront_device_helpers.h"

// ============================================================================
// Kernel 2 — evaluate_materials
//   Processes HitWorkItems.  For each hit:
//     - Evaluates the BSDF (same logic as closesthit in optix_programs.cu)
//     - Pushes a ShadowRayWorkItem for direct-light NEE
//     - Pushes a RayWorkItem into nextRayQueue for the next bounce
//     - Emissive surfaces write directly to the framebuffer
// ============================================================================
extern "C" __global__ void evaluate_materials(
	WorkQueue<HitWorkItem>       hitQueue,
	int                          numHits,
	WorkQueue<RayWorkItem>       nextRayQueue,
	WorkQueue<ShadowRayWorkItem> shadowQueue,
	// BSSRDF probe-request queue (MaterialType::Subsurface, Phase 2) - see
	// this file's own Subsurface case and BssrdfProbeWorkItem's comment
	// (wavefront_types.h).
	WorkQueue<BssrdfProbeWorkItem> bssrdfProbeQueue,
	float3*                      framebuffer,
	// Scene data
	const SphereData*   spheres,
	const QuadData*     quads,
	const TriangleData* triangles,
	const BilinearPatchData* bilinearPatches,
	const DiskData*     disks,
	const CylinderData* cylinders,
	const MaterialData* materials,
	const int*          lightIndices,
	const GpuLightKind* lightKinds,
	const GpuAliasEntry* aliasTable,
	unsigned int numLights,
	const PunctualLightGPU* punctualLights,
	unsigned int numPunctualLights,
	const TextureData*  textures,
	const unsigned char* texturePixels,
	int maxDepth,
	const CloudMedium<float>* cloudMediums,
	unsigned int numCloudMediums,
	const GpuRgbGridMedium* rgbGridMediums,
	const float* rgbGridData,
	const GpuGridMedium* gridMediums,
	const float* gridData,
	// Real tabulated measured-BRDF tables (MaterialType::Measured) - see
	// optix_types.h's GpuMeasuredTable comment and wavefront_measured_bxdf.h.
	// Plain extra kernel parameters, same shape as cloudMediums/
	// rgbGridMediums above (this material needs no OptiX trace, so it does
	// not go through the wf_params/lp raygen-launch-params path the BSSRDF
	// probe walk uses).
	const GpuMeasuredTable* measuredTables,
	unsigned int numMeasuredTables,
	const float* measuredParamValues,
	const float* measuredData,
	const float* measuredMcdf,
	const float* measuredCcdf,
	float3 skyColor,
	float shadowRayEpsilon,
	// Real importance-sampled HDR sky (LightSource "infinite" with an image) -
	// see optix_types.h's GpuSkyDistribution comment. Passed by value, same
	// established pattern as GpuCameraParams itself already crossing this
	// exact kernel-launch boundary (generate_camera_rays). height<=0 (the
	// default when the scene has no image sky) makes every sky-NEE/miss call
	// site below fall back to skyColor + uniform-sphere sampling, unchanged.
	GpuSkyDistribution skyDist,
	GpuPortalLight portalLight,
	// Integrator "bool regularize" (defaults false, matching pbrt-v4's own
	// real default) - a single scalar pulled out of GpuCameraParams and
	// passed independently, same established pattern as shadowRayEpsilon
	// above (this kernel doesn't otherwise need the whole struct). A
	// per-launch constant, unlike h.any_nonspecular (per-path history) -
	// AND'd together at each glossy-alpha call site below (see
	// wf_glossy_alpha()'s own comment for why the gate lives at the call
	// sites rather than inside that shared helper).
	bool regularize,
	// "float maxcomponentvalue" firefly clamp - see
	// GpuCameraParams::maxComponentValue's own comment (optix_types.h) and
	// this kernel's own addToFramebuffer's comment below. Same "single
	// scalar pulled out of GpuCameraParams" pattern as regularize above.
	float maxComponentValue,
	// Denoiser guide-layer AOVs (null unless --denoise was requested - same
	// convention as the recursive backend's LaunchParams::albedoBuffer/
	// normalBuffer, optix_types.h). Accumulated (atomicAdd, then divided by
	// samplesPerPixel - see launchNormalizeAovBuffers()) across every
	// sample's own primary-ray hit, mirroring optix_raygen.h's albedo_sum/
	// normal_sum - that backend can average in a plain per-thread local
	// since one thread owns a whole pixel's samples sequentially; this
	// backend processes one sample per kernel launch, so accumulation has
	// to persist across launches instead.
	float3* albedoBuffer,
	float3* normalBuffer,
	float4* worldPosBuffer,
	// ReSTIR DI (Live Preview only) - see wf_finish_material_scatter's own
	// restirReservoirs parameter comment. nullptr for batch/offline rendering.
	GpuReservoir* restirReservoirs,
	// See wf_finish_material_scatter's own restirCtx parameter comment.
	GpuRestirTemporalContext restirCtx,
	// ReSTIR GI (Live Preview only) - see wf_finish_material_scatter's own
	// giOriginContext/giCandidateOut parameter comments. nullptr for batch/
	// offline rendering.
	GpuGiOriginContext* giOriginContext,
	GpuGiSample* giCandidateOut,
	// See wf_light_bvh_sample_index()'s own comment
	// (wavefront_restir_helpers.h). lightBvh.nodeCount<=0 (the default) means
	// "no light BVH built" - forwarded to wf_finish_material_scatter() below
	// unchanged, which itself falls straight through to the alias table for
	// that case.
	WfLightBvhContext lightBvh = {},
	// Real-time path guiding (Live Preview only, gpu/optix/wavefront_guiding.h)
	// - see this project's own plan. Consulted by the Conductor/RoughMetal
	// cases below (v1 scope) for their own scatter-direction mixture pdf,
	// AND forwarded into wf_finish_material_scatter() so its NEE/MIS pdf
	// (evalGlossyF's own Conductor/RoughMetal branches) uses the SAME
	// mixture rather than the plain BSDF pdf alone - unlike probeGridMeta/
	// probeGrid there (which stay on their nullptr/default - Lambertian
	// never reaches this kernel, see WavefrontQueues::simpleHitQueue's own
	// comment). guidingHistograms==nullptr (the default, every non-Live-
	// Preview call site, or Live Preview with the feature toggled off) is a
	// complete no-op - every glossy case below falls through to its own
	// existing, unmodified GGX/VNDF sampling unchanged.
	GpuProbeGridMeta guidingGridMeta = {},
	const GpuGuidingHistogram* guidingHistograms = nullptr,
	// Same probe array wf_query_probe_grid() (probe_grid_types.h) already
	// leak-guards the SH-L1 diffuse cache against, reused here so
	// wf_guiding_nearest_probe() can reject a "nearest by grid index" probe
	// that's actually occluded from hit_point (e.g. on the far side of a
	// thin wall) instead of returning it unconditionally. nullptr whenever
	// guidingHistograms is (guidingActive() in wavefront_path_tracer.cpp
	// gates both together) - wf_guiding_nearest_probe() treats a null probes
	// pointer as "skip the leak test", never as a crash.
	const GpuProbe* guidingProbes = nullptr,
	// Neural Radiance Cache (Live Preview only) - see wf_finish_material_
	// scatter's own nrcWeights parameter comment. This IS a real call site
	// (unlike probeGridMeta/probeGrid just above, which stay on their
	// nullptr/default here): RoughMetal is routed to THIS kernel (hitQueue,
	// not simpleHitQueue - see WavefrontQueues::simpleHitQueue's own routing
	// comment) and is one of the two material types NRC supports.
	// nrcWeights==nullptr (every non-Live-Preview call site, or Live
	// Preview with the feature toggled off) is a complete no-op.
	const float* nrcWeights = nullptr,
	int nrcTrainingSteps = 0,
	float3 nrcAabbMin = make_float3(0.0f, 0.0f, 0.0f),
	float3 nrcAabbExtent = make_float3(0.0f, 0.0f, 0.0f),
	// ReSTIR for volumetric/participating media (Live Preview only) - see
	// wf_finish_material_scatter's own restirVolumeReservoirs/restirVolumeCtx/
	// volumeMatIdxOut/volumeEntryPointOut parameter comments. nullptr for
	// batch/offline rendering, same shape as restirReservoirs above. This IS
	// a real call site (all 5 medium MaterialTypes are routed to THIS kernel,
	// not simpleHitQueue).
	GpuVolumeReservoir* restirVolumeReservoirs = nullptr,
	GpuVolumeRestirTemporalContext restirVolumeCtx = {},
	int* volumeMatIdxOut = nullptr,
	float4* volumePhaseWoGOut = nullptr,
	float4* volumeEntryPointOut = nullptr
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numHits) return;

	const HitWorkItem& h = hitQueue.items[idx];
	const MaterialData& mat = materials[h.materialIdx];

	// Denoiser guide-layer AOV - primary-ray hits only (depth==0). See
	// wf_accumulate_aov()'s own comment (above gpu_in_crop()) for why.
	if (albedoBuffer && h.depth == 0) {
		wf_accumulate_aov(albedoBuffer, normalBuffer, h.pixelIndex, mat.albedo, h.normal);
	}
	// Live Preview reprojection guide buffer - see wf_write_world_pos()'s
	// own comment for why this is a plain overwrite, not an atomicAdd.
	if (worldPosBuffer && h.depth == 0) {
		wf_write_world_pos(worldPosBuffer, h.pixelIndex, h.hitPoint);
	}

	// Hoisted once and reused by every glossy-alpha call site below - both
	// operands are fixed for the rest of this thread's execution (regularize
	// is a per-launch kernel parameter, h is a const reference never
	// reassigned after this point), so recomputing the AND at each site would
	// be pure duplication, not a different value.
	const bool do_regularize = regularize && (bool)h.any_nonspecular;

	// Mirrors optix_intersection_disk_cylinder.h's own trap: the pbrt loader
	// never assigns these material types to a disk/cylinder (see
	// pbrt_gpu_builder.h's disk/cylinder loop), so this is unreachable
	// today, but kept loud rather than silently wrong if that ever changes -
	// unlike the recursive backend, this backend has no shape-specific
	// handling for any of these on disk/cylinder geometry (geomType 4/5).
	// Only this general queue needs the check: simpleHitQueue/
	// dielectricHitQueue only ever receive Lambertian/Metal and
	// Dielectric/RoughDielectric hits respectively, never these types.
	//
	// Hair and (cylinder-only) Medium/DielectricMedium are real, reachable
	// combinations this single shared queue already knows how to shade
	// (Hair via the real MaterialType::Hair case below, keyed only on
	// mat.type; Medium/DielectricMedium via h.t/h.mediumTFar/h.frontFace,
	// recomputed by __closesthit__wf_cylinder - see pbrt_gpu_builder.h's
	// cylinder loop comment) - trapping any of these here would be a
	// genuine regression from real support to "falls back to Lambertian"/
	// an abort, not the unreachable-defensive-code every other trapped type
	// here still is. wf_material_supported_on_disk_cylinder_geom() (this
	// file, above) is the one place these per-shape exemptions live, so
	// this condition itself doesn't grow a new `&& !flagN` term each time
	// another (shape, material) combination gains real support.
	if ((h.geomType == 4 || h.geomType == 5) &&
		!wf_material_supported_on_disk_cylinder_geom(mat.type, h.geomType)) {
		printf("[WF-DISK-CYL-SHADE] MaterialType %d is not supported on disk/cylinder geometry (geomType=%d)\n",
			   (int)mat.type, h.geomType);
		__trap();
	}

	float3 normal    = h.normal;
	float3 hit_point = h.hitPoint;
	unsigned int seed = h.seed;
	// <= 0 (zero-init default) means "use the standard 0.01f" - mirrors
	// optix_device_helpers.h's trace_shadow_ray() exactly, see
	// GpuCameraParams::shadowRayEpsilon's own comment.
	const float shadow_eps = (shadowRayEpsilon > 0.0f) ? shadowRayEpsilon : 0.01f;

	// Reconstruct spectral path state from work item
	using SS  = SampledSpectrum<kWFNWavelengths>;
	using SWL = SampledWavelengths<kWFNWavelengths>;
	SS throughput(h.throughput);
	SS radiance(h.radiance);
	SWL swl;
	for (int i = 0; i < kWFNWavelengths; ++i) {
		swl.lambda[i] = h.wavelengths[i];
		swl.pdf[i]    = h.wavelength_pdfs[i];
	}

	// Helper: convert SampledSpectrum to sRGB and atomicAdd to framebuffer
	auto addToFramebuffer = [&](int pixIdx, const SS& L) {
		auto xyz = SampledSpectrumToXYZ(L, swl, d_cie_x, d_cie_y, d_cie_z,
										kDevCIEMin, kDevCIENSamples);
		float r, g, b;
		wf_xyz_to_linear_rgb(xyz.x, xyz.y, xyz.z, r, g, b);
		// "float maxcomponentvalue" firefly clamp - see
		// GpuCameraParams::maxComponentValue's own comment (optix_types.h)
		// for why this is a per-CONTRIBUTION clamp, not a true
		// per-sample-total clamp CPU/the recursive backend apply.
		const float m = fmaxf(r, fmaxf(g, b));
		if (maxComponentValue > 0.0f && m > maxComponentValue) {
			const float s = maxComponentValue / m;
			r *= s; g *= s; b *= s;
		}
		atomicAdd(&framebuffer[pixIdx].x, r);
		atomicAdd(&framebuffer[pixIdx].y, g);
		atomicAdd(&framebuffer[pixIdx].z, b);
	};

	// -------------------------------------------------------------------------
	// Emissive: add emission term, path terminates (no scatter).
	// pbrt-v4 alignment: the camera ray and a specular bounce add the full emission (no NEE partner);
	// any other arrival adds it weighted by the MIS complement of the NEE shadow ray taken at the
	// previous vertex, so the two strategies together count the lamp exactly once.
	// -------------------------------------------------------------------------
	if (mat.type == MaterialType::DiffuseLight) {
		// A BSDF-sampled arrival (not specular, not the camera ray) is MIS-weighted against the NEE sample that
		// the previous vertex took at this same point - see wf_emitter_hit_mis_weight().
		float emitMis = 1.0f;
		if (!h.specular_bounce && h.depth > 0)
			emitMis = wf_emitter_hit_mis_weight(h, spheres, quads, triangles, bilinearPatches, disks, cylinders,
												lightIndices, lightKinds, aliasTable, numLights, lightBvh);
		if (emitMis > 0.0f) {
			// mat.twoSided (pbrt AreaLightSource "diffuse" "bool twosided") lets
			// a light emit from both faces instead of gating on frontFace - see
			// CPU's diffuse_light::is_two_sided()/the recursive backend's
			// material_emission() for the matching gate. h.frontFace, NOT
			// dot(-h.rayDir, normal): every __closesthit__wf_* program already
			// flips `normal` to face the incoming ray before packing it into
			// the payload (matching CPU's own set_face_normal() convention),
			// so that dot product is always positive by construction and can
			// never discriminate front from back - a pre-existing bug (predates
			// this twoSided feature entirely) that silently made every
			// wavefront-rendered area light two-sided regardless of what the
			// scene asked for, only surfaced now by testing twosided=false
			// end-to-end.
			if (mat.twoSided || h.frontFace) {
				// A real map_Ke texture (Gallery's painted-canvas glow) or a
				// pbrt AreaLightSource "filename" image is sampled at the hit
				// UV instead of the flat mat.emission - see add_diffuse_light()'s
				// textureIdx comment and the recursive backend's identical
				// material_emission() change. emissionScale is a no-op 1.0
				// multiply unless the light was built with a "scale" param.
				float3 le = (mat.textureIdx >= 0)
					? wf_sample_texture(textures, texturePixels, mat.textureIdx, h.uv_u, h.uv_v, hit_point)
					: mat.emission;
				if (mat.textureIdx >= 0) {
					le.x *= mat.emissionScale;
					le.y *= mat.emissionScale;
					le.z *= mat.emissionScale;
				}
				// Guarded (not unconditional): a zero/negative `le` must leave
				// `radiance` completely untouched, not just add a zero spectrum -
				// wf_lift_rgb_to_spectrum's own internal sc<=0 early-out makes the
				// uplift itself a no-op either way, but `throughput * SS(0.f)` is
				// NOT provably `SS(0.f)` if `throughput` were ever NaN/Inf (IEEE-754
				// 0*NaN=NaN), so skip the multiply entirely rather than rely on
				// multiplying by zero to cancel it out.
				if (le.x > 0.0f || le.y > 0.0f || le.z > 0.0f)
					radiance = radiance + (throughput * emitMis) * wf_lift_rgb_to_spectrum(le, swl, /*isIlluminant=*/true);
			}
		}
		addToFramebuffer(h.pixelIndex, radiance * h.filterWeight);
		return; // path ends at emissive surface
	}

	if (h.depth >= maxDepth) {
		// Max depth reached — just accumulate what we have.
		addToFramebuffer(h.pixelIndex, radiance * h.filterWeight);
		return;
	}

	// -------------------------------------------------------------------------
	// Scatter — evaluate BSDF (mirrors optix_programs.cu closesthit logic)
	// -------------------------------------------------------------------------
	float3 scattered_dir    = make_float3(0, 0, 0);
	SS     attenuation(0.f);   // spectral BSDF * cos / pdf
	bool   scattered        = false;
	bool   is_specular      = false;
	// True only for MaterialType::Interface - handled specially right
	// after this switch, bypassing wf_finish_material_scatter() (NEE/RR/
	// depth+specular_bounce+brdf_pdf recompute) entirely, since nothing
	// actually scattered here. See MaterialType::Interface's own comment
	// (optix_types.h).
	bool   is_medium_boundary = false;
	float  brdf_pdf_override = -1.0f;
	// Set by MaterialType::Medium/CloudMedium/RgbGridMedium/GridMedium (on a
	// genuine scatter, not the no-interaction pass-through sub-case) and by
	// DielectricMedium's own medium-interior phase-scatter sub-case (see
	// each case below) - passed through to wf_finish_material_scatter's NEE
	// block for real phase-function NEE+MIS. Unused (harmless zero) for
	// every other material type.
	float3 phaseWo = make_float3(0.0f, 0.0f, 0.0f);
	// wf_finish_material_scatter's `nfEta` slot - mat.ior by default
	// (correct for MaterialType::NormalizedFresnel); MaterialType::
	// RoughDielectric's glossy branch overrides this to its own precomputed
	// eta (front_face ? 1/ior : ior), which is NOT the same value for a
	// back-face hit - see that case's own comment.
	float matEta = mat.ior;
	// pbrt-v4 etaScale term for THIS event only - see RayWorkItem::
	// etaScale's own comment. 1.0f (a no-op) unless a case below sets it on
	// a genuine transmission; combined with h.etaScale at this function's
	// tail call to wf_finish_material_scatter. The Subsurface case's own
	// TRANSMISSION sub-case doesn't use this at all (it returns early via
	// bssrdfProbeQueue.push instead of reaching that tail) - it sets
	// probeItem.etaScale directly.
	float eventEta = 1.0f;
	float  phaseG  = 0.0f;
	// ReSTIR for volumetric/participating media (Live Preview only) - see
	// wf_finish_material_scatter's own mediumMeanFreePath parameter comment.
	// Set alongside phaseG by whichever of the 5 medium cases below computes
	// its own sigma_t/sigma_maj for free-flight distance sampling (that same
	// local value is exactly what wf_restir_volume_mean_free_path needs) -
	// set unconditionally there, even on the no-interaction pass-through
	// sub-case, since it's simply unused (isPhase never true) in that case.
	float mediumMeanFreePath = 0.0f;
	// Set inside whichever glossy case (Conductor/RoughMetal/RoughDielectric/
	// CoatedDiffuse/CoatedConductor) this hit takes, via wf_glossy_alpha() -
	// passed to wf_finish_material_scatter's glossyAlpha parameter below so
	// NEE reuses the exact same regularized alpha the switch case's own
	// BSDF-sampling step just used, instead of re-deriving it a second time.
	// Left at 0 (harmless - see that parameter's own comment) for every
	// non-glossy case.
	// The measured-table arrays, for the Measured case's direct-light evaluation inside wf_finish_material_scatter().
	const WfMeasuredTables measuredBundle{ measuredTables, numMeasuredTables, measuredParamValues, measuredData, measuredMcdf, measuredCcdf };
	float glossyAlphaForNEE = 0.0f;
	float glossyAlphaVForNEE = 0.0f;
	// CoatedConductor's base-conductor alphas (the two above are the coat's); -1 = not a coated conductor.
	float glossyCondAlphaForNEE = -1.0f;
	float glossyCondAlphaVForNEE = -1.0f;
	// wf_finish_material_scatter's own matType argument - mat.type by
	// default (every case's real material type, unchanged). MaterialType::
	// DielectricMedium's own fused ROUGH entry/exit sub-case (see that case
	// below) overrides this to MaterialType::RoughDielectric for just that
	// one tail call: wf_finish_material_scatter's isPhase/glossy_isType
	// dispatch (wavefront_device_helpers.h) is driven entirely by this
	// argument, not by re-reading mat.type itself, and unconditionally
	// treats every DielectricMedium event as a medium-interior phase
	// scatter - there is no fused "rough dielectric surface" case in that
	// dispatch to route into instead. Passing RoughDielectric here (while
	// h.materialIdx/mat still correctly point at the real DielectricMedium
	// MaterialData for every index-based field lookup - ior, glossyAlpha,
	// etc., all identical between the two types - see DielectricMedium's
	// own case for which fields alias which) reuses that EXISTING, already-
	// correct, already-tested glossy code path verbatim: zero edits to the
	// shared function, correct is_specular/MIS/next-bounce behavior, and
	// ReSTIR DI/GI/probe-cache/NRC-training all keep working for this
	// combination - chosen over hand-duplicating the NEE block inline after
	// finding that the naive duplicate needs to force is_specular=true to
	// suppress the shared function's own (wrong, phase-based) NEE attempt,
	// which also mislabels the next bounce as specular for its own later
	// MIS decision, a real bias a true duplicate would need to additionally
	// take over next-ray-setup/RR/throughput logic to avoid - this
	// substitution avoids that trade-off entirely instead.
	MaterialType effectiveMatType = mat.type;

	// Helper: uplift RGB albedo to SampledSpectrum via device sigmoid polynomial
	auto albedoSpectrum = [&](float3 rgb) -> SS {
		float c0, c1, c2;
		float r = rgb.x < 0.f ? 0.f : (rgb.x > 1.f ? 1.f : rgb.x);
		float g = rgb.y < 0.f ? 0.f : (rgb.y > 1.f ? 1.f : rgb.y);
		float b = rgb.z < 0.f ? 0.f : (rgb.z > 1.f ? 1.f : rgb.z);
		dev_srgb_to_coeffs(r, g, b, c0, c1, c2);
		RGBSigmoidPolynomial poly(c0, c1, c2);
		SS s(0.f);
		for (int i = 0; i < kWFNWavelengths; ++i)
			s[i] = poly(swl.lambda[i]);
		return s;
	};
	auto emissionSpectrum = [&](float3 /*rgb*/) -> SS { return SS(0.f); };  // placeholder, liftEmission used inline
	(void)emissionSpectrum;

	// Uplift an unbounded-positive RGB weight (can exceed 1, unlike a plain
	// reflectance) to spectral: same normalize-by-2*max/uplift/rescale
	// technique as the NEE block's liftEmission lambda below - needed for
	// MaterialType::Hair, whose HairBxDF::sample() result already divides by
	// the sample pdf (a proper importance-sampling weight, not a [0,1]
	// albedo), so naively clamping it into albedoSpectrum's range would bias
	// bright/peaky fiber lobes toward black.
	auto unboundedSpectrum = [&](float3 rgb) -> SS {
		// Clamp negative components before normalizing - dev_srgb_to_coeffs
		// expects each channel in [0,1] once divided by sc; a slightly
		// negative component (numerically possible at extreme absorption or
		// near-grazing angles in HairBxDF's Marschner model, the only caller
		// of this lambda) would otherwise reach it unclamped, unlike
		// albedoSpectrum above which always clamps first.
		float rx = rgb.x < 0.f ? 0.f : rgb.x;
		float ry = rgb.y < 0.f ? 0.f : rgb.y;
		float rz = rgb.z < 0.f ? 0.f : rgb.z;
		float m = rx > ry ? (rx > rz ? rx : rz) : (ry > rz ? ry : rz);
		float sc = 2.f * m;
		if (sc <= 0.f) return SS(0.f);
		float c0, c1, c2;
		dev_srgb_to_coeffs(rx/sc, ry/sc, rz/sc, c0, c1, c2);
		RGBSigmoidPolynomial poly(c0, c1, c2);
		SS s(0.f);
		for (int i = 0; i < kWFNWavelengths; ++i)
			s[i] = sc * poly(swl.lambda[i]);
		return s;
	};

	switch (mat.type) {
	case MaterialType::Lambertian: {
		scattered_dir = normalize(normal + wf_rand_unit(seed));
		if (wf_near_zero(scattered_dir)) scattered_dir = normal;
		const float3 lambertianColor = (mat.textureIdx >= 0)
			? wf_sample_texture(textures, texturePixels, mat.textureIdx, h.uv_u, h.uv_v, hit_point) * mat.emissionScale
			: mat.albedo;
		attenuation = albedoSpectrum(lambertianColor);
		scattered   = true;
		is_specular = false;
		break;
	}
	case MaterialType::Subsurface: {
		// Real tabulated BSSRDF (wavefront backend, Phase 2 - Phase 1
		// shipped this for the recursive backend only, see optix_device_
		// helpers.h's shade_material() Subsurface case, which this mirrors).
		//
		// Entry interface: the exact same smooth dielectric sample as
		// MaterialType::Dielectric, via the already-shared wf_dielectric_
		// scatter() helper (also used by DielectricMedium's own entry
		// surface above) - matches pbrt-v4's SubsurfaceMaterial::GetBxDF /
		// this codebase's CPU `class subsurface` (material_pbrt.h), and the
		// recursive backend's identical dielectric_scatter() call.
		//
		// front_face is hardcoded true: by this whole design's construction
		// (mirroring pbrt-v4's own SubsurfaceMaterial and the recursive
		// backend's Phase 1), a Subsurface material's entry interface is
		// only ever reached from a genuine outside hit - the probe walk's
		// own exit point resumes as a MaterialType::NormalizedFresnel bounce
		// instead of a second Subsurface entry (see resolve_bssrdf_exit()
		// below), so this switch case is never reached "from inside".
		// HitWorkItem::frontFace itself isn't usable here regardless - it's
		// only populated for sphere hits (__closesthit__wf_sphere), and the
		// SSS dragon scenes this targets are triangle meshes.
		float3 sdir = wf_dielectric_scatter(h.rayDir, normal, /*front_face=*/true, mat.ior, seed);
		bool is_transmission = dot(sdir, normal) < 0.0f;
		if (!is_transmission) {
			// Specular reflection off the entry interface - identical to
			// Dielectric's own reflection case; falls through to the shared
			// tail below as an ordinary specular bounce (no NEE, no probe
			// walk needed).
			scattered_dir = sdir;
			attenuation   = SS(1.f);
			scattered     = true;
			is_specular   = true;
			break;
		}

		// Transmission: hand off to the dedicated BSSRDF probe-walk stage
		// (wavefront_probe.h's __raygen__wf_probe + this file's own
		// resolve_bssrdf_exit(), wired into WavefrontPathTracer::render()'s
		// per-bounce loop) instead of scattering inline - the probe walk
		// needs real optixTrace calls, unavailable from this plain CUDA
		// kernel. Carries everything needed to resume the path once (or if)
		// an exit point is found - see BssrdfProbeWorkItem's own comment.
		// This thread's own work ends here: nothing is pushed to
		// nextRayQueue/shadowQueue/framebuffer now, that happens later (this
		// bounce, before the shadow pass - see the per-bounce loop) via
		// resolve_bssrdf_exit() once the probe walk resolves.
		{
			BssrdfProbeWorkItem probeItem;
			probeItem.p0     = hit_point;
			probeItem.axis0  = normal;
			probeItem.matIdx = h.materialIdx;
			probeItem.seed   = seed;
			// pbrt-v4 etaScale, entry interface only - front_face is
			// hardcoded true here (see this case's own comment), so
			// eta = 1/mat.ior. See BssrdfProbeWorkItem::etaScale's own
			// comment for why this needs to survive the probe walk.
			{
				const float entryEta = 1.0f / mat.ior;
				probeItem.etaScale = h.etaScale * entryEta * entryEta;
			}
			// Pure reconstruction weight, carried unchanged through the probe
			// walk - see RayWorkItem::filterWeight's own comment.
			probeItem.filterWeight = h.filterWeight;
			for (int i = 0; i < kWFNWavelengths; ++i) {
				probeItem.throughput[i]      = throughput[i];
				probeItem.radiance[i]        = radiance[i];
				probeItem.wavelengths[i]     = swl.lambda[i];
				probeItem.wavelength_pdfs[i] = swl.pdf[i];
			}
			probeItem.pixelIndex = h.pixelIndex;
			probeItem.depth      = h.depth;
			probeItem.time       = h.time;
			bssrdfProbeQueue.push(probeItem);
		}
		return;
	}
	case MaterialType::Metal: {
		float3 reflected = wf_reflect(normalize(h.rayDir), normal);
		scattered_dir    = normalize(reflected + mat.fuzz * wf_rand_unit(seed));
		attenuation      = albedoSpectrum(mat.albedo);
		scattered        = (dot(scattered_dir, normal) > 0.0f);
		is_specular      = true;
		break;
	}
	case MaterialType::Measured: {
		// Real tabulated measured BRDF (pbrt-v4 MeasuredBxDF): one VNDF-importance-sampled sample continues the path with the weight
		// f * |cos| / pdf, and the vertex takes direct-light samples (area, sky, punctual) with MIS through wf_finish_material_scatter()'s
		// evalGlossyF, which evaluates the BxDF's own f() and pdf() (wf_gpu_measured_f_pdf). The weight needs unboundedSpectrum, not
		// albedoSpectrum: it is a throughput weight, not a [0,1] albedo - same convention as MaterialType::Hair/Principled above.
		// (It used to be a specular-style bounce with no light sampling, unbiased but very noisy under a small light, which never lit
		// the material from a point/spot/distant light at all.)
		float3 sdir, atten;
		float mpdf;
		const bool m_ok = wf_sample_measured_material(h.rayDir, normal, mat,
				measuredTables, numMeasuredTables,
				measuredParamValues, measuredData, measuredMcdf, measuredCcdf,
				seed, sdir, atten, mpdf);
		// A rejected sample (grazing wo, zero pdf, a reflection below the horizon) must not skip this vertex's NEE - pbrt takes the
		// direct-light sample whether or not the sample that follows succeeds. It carries zero weight and the path ends after NEE.
		scattered_dir = m_ok ? sdir : normal;
		attenuation   = m_ok ? unboundedSpectrum(atten) : SS(0.f);
		scattered     = true;
		is_specular   = false;
		brdf_pdf_override = m_ok ? mpdf : -1.0f;
		phaseWo = -normalize(h.rayDir);   // the outgoing direction evalGlossyF evaluates f() and pdf() against
		break;
	}
	case MaterialType::Interface: {
		// Real pass-through - exact same direction as the incoming ray, no
		// Fresnel/refraction math at all. `scattered`/`attenuation`/
		// `scattered_dir` are still set (mirroring every other case) so the
		// `if (!scattered)` absorbed-path check right after this switch
		// doesn't misclassify this as absorption - but is_medium_boundary
		// (checked BEFORE that absorbed check) routes this to its own
		// direct RayWorkItem push instead of falling into
		// wf_finish_material_scatter().
		scattered_dir      = h.rayDir;
		attenuation         = SS(1.f);
		scattered           = true;
		is_medium_boundary  = true;
		break;
	}
	case MaterialType::Dielectric: {
		// h.frontFace, NOT dot(h.rayDir, normal) - see
		// evaluate_materials_dielectric()'s identical Dielectric case (this
		// file) for why the dot-product re-derivation is always wrong
		// (always negative by construction against the pre-flipped
		// `normal`). This case is unreachable in practice (see the comment
		// below), but kept bug-for-bug consistent with the live case rather
		// than left as a misleading template for anyone extending this
		// switch.
		float eta = h.frontFace ? (1.0f / mat.ior) : mat.ior;
		float3 unit_dir = normalize(h.rayDir);
		float  cos_t = fminf(dot(-unit_dir, normal), 1.0f);
		float  sin_t = sqrtf(1.0f - cos_t * cos_t);
		bool   cannot_refract = eta * sin_t > 1.0f;
		// exact Fresnel reflectance (pbrt's FrDielectric, eta_t/eta_i = 1/eta) - this used to be Schlick's approximation fed the incident cosine, which under-reflects for a ray leaving the glass (the inside-to-outside Fresnel rises much faster than the outside-in one) and so sent ~2x too much light through two refractions: a smooth glass sphere read 105% of a pbrt path-level reference on the wavefront backend, 100-101% on the CPU and recursive backends, which already used FrDielectric
		float fresnel_R = FrDielectric(cos_t, 1.0f / eta);
		bool is_transmission;
		if (cannot_refract || fresnel_R > wf_rand(seed)) {
			scattered_dir = wf_reflect(unit_dir, normal);
			is_transmission = false;
		} else {
			scattered_dir = wf_refract(unit_dir, normal, eta);
			is_transmission = true;
		}
		// Tf tints transmission only, matching real colored glass - see
		// add_dielectric()'s transmission_filter comment (recursive
		// backend's optix_device_helpers.h identical case has the same
		// reasoning).
		attenuation = is_transmission ? albedoSpectrum(mat.transmission_filter) : SS(1.f);
		scattered   = true;
		is_specular = true;
		break;
	}
	// MaterialType::RoughDielectric has no case here (unlike the other 4
	// glossy types below) - wavefront_programs.cu's __raygen__wf_intersect
	// routes every Dielectric/RoughDielectric hit into dielectricHitQueue
	// (consumed only by evaluate_materials_dielectric(), which has the real,
	// live copy of this case), so hitQueue - and therefore this kernel - can
	// never actually see one. A RoughDielectric case here would be dead code
	// that looks live; if that routing ever changes, the `default:` trap
	// below will catch the drift loudly instead of silently running stale
	// unmaintained logic.
	case MaterialType::Conductor: {
		float c_alpha_x = wf_glossy_alpha(mat, do_regularize);
		float c_alpha_y = wf_glossy_alpha_v(mat, do_regularize);
		glossyAlphaForNEE = c_alpha_x;
		glossyAlphaVForNEE = c_alpha_y;
		float3 cn = normal;
		float3 ctan, cbitan;
		BuildDpduTangentFrame(cn.x, cn.y, cn.z, h.objDpdu.x, h.objDpdu.y, h.objDpdu.z,
		                       ctan.x, ctan.y, ctan.z, cbitan.x, cbitan.y, cbitan.z);
		float3 cwi = -normalize(h.rayDir);
		float cwi_x = dot(cwi, ctan), cwi_y = dot(cwi, cbitan), cwi_z = dot(cwi, cn);
		if (cwi_z <= 0.0f) { scattered = false; break; }
		TrowbridgeReitz<float> c_dist(c_alpha_x, c_alpha_y);

		// Real-time path guiding + GGX/VNDF scatter-direction sampling
		// (shared with MaterialType::RoughMetal below) - see
		// wf_sample_guided_glossy's own header comment (wavefront_device_
		// helpers.h) for the full mixture-pdf rationale.
		WfGuidedGlossySample c_sample = wf_sample_guided_glossy(
			c_dist, cwi_x, cwi_y, cwi_z, ctan, cbitan, cn, c_alpha_x, c_alpha_y,
			hit_point, guidingGridMeta, guidingHistograms, guidingProbes, seed);
		// A rejected continuation sample must not skip this vertex's NEE - pbrt takes the direct-light sample at every vertex whether or
		// not the sample that follows succeeds. It carries zero weight and the path ends after wf_finish_material_scatter()'s NEE; the
		// effectively-smooth lobe has no NEE and keeps the old early exit.
		const bool c_rej = !c_sample.scattered;
		if (c_rej && c_dist.EffectivelySmooth()) { scattered = false; break; }
		float3 c_F = c_rej ? make_float3(0.0f, 0.0f, 0.0f)
		                   : FrConductorRGB(c_sample.wm_dot_wi, mat.eta_c.x, mat.eta_c.y, mat.eta_c.z, mat.k_c.x, mat.k_c.y, mat.k_c.z);
		scattered_dir = c_rej ? cn : normalize(c_sample.wo_x*ctan + c_sample.wo_y*cbitan + c_sample.wo_z*cn);
		scattered   = true;

		// Real NEE/MIS for glossy (non-EffectivelySmooth) conductors, via
		// wf_finish_material_scatter's shared evalGlossyF (see its own
		// comment). brdf_pdf_override uses c_sample.pdf (the GGX-reflection
		// VNDF pdf, or path guiding's own mixture pdf, at the sampled
		// direction) - Conductor is reflection-only to begin with, so this
		// is the exact (not proxy) pdf, unlike CoatedDiffuse/CoatedConductor
		// below. phaseWo carries cwi (already `-ray_dir`, matching
		// evalGlossyF's `wi_world` convention).
		if (!c_dist.EffectivelySmooth()) {
			is_specular = false;
			brdf_pdf_override = c_rej ? -1.0f : c_sample.pdf;
			phaseWo = cwi;
		} else {
			is_specular = true;
		}
		// Use average Fresnel weight as scalar (conductor is specular, color from albedo)
		attenuation = c_rej ? SS(0.f) : albedoSpectrum(make_float3(c_F.x * c_sample.weight, c_F.y * c_sample.weight, c_F.z * c_sample.weight));
		break;
	}
	case MaterialType::RoughMetal: {
		// GGX VNDF + flat-tint reflectance, no complex Fresnel (pbrt-v4/RTOW
		// rough_metal) - same shape as MaterialType::Conductor above, minus
		// FrConductorRGB (RoughMetalBxDF has no real Fresnel model). NOT the
		// same material as MaterialType::Metal (fuzz-perturbed mirror, a
		// different model - CPU's plain `metal` class).
		float rm_alpha = wf_glossy_alpha(mat, do_regularize);
		glossyAlphaForNEE = rm_alpha;
		float3 rmn = normal;
		float3 rmup = (fabsf(rmn.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
		float3 rmtan   = normalize(cross(rmup, rmn));
		float3 rmbitan = cross(rmn, rmtan);
		float3 rmwi = -normalize(h.rayDir);
		float rmwi_x = dot(rmwi, rmtan), rmwi_y = dot(rmwi, rmbitan), rmwi_z = dot(rmwi, rmn);
		if (rmwi_z <= 0.0f) { scattered = false; break; }
		TrowbridgeReitz<float> rm_dist(rm_alpha, rm_alpha);

		// Real-time path guiding + GGX/VNDF scatter-direction sampling - see
		// MaterialType::Conductor's identical-shape block above; RoughMetal
		// has no Fresnel color of its own (uses mat.albedo directly below),
		// everything else is the shared wf_sample_guided_glossy().
		WfGuidedGlossySample rm_sample = wf_sample_guided_glossy(
			rm_dist, rmwi_x, rmwi_y, rmwi_z, rmtan, rmbitan, rmn, rm_alpha, rm_alpha,
			hit_point, guidingGridMeta, guidingHistograms, guidingProbes, seed);
		// A rejected continuation sample must not skip this vertex's NEE - pbrt takes the direct-light sample at every vertex whether or
		// not the sample that follows succeeds. It carries zero weight and the path ends after wf_finish_material_scatter()'s NEE; the
		// effectively-smooth lobe has no NEE and keeps the old early exit.
		const bool rm_rej = !rm_sample.scattered;
		if (rm_rej && rm_dist.EffectivelySmooth()) { scattered = false; break; }
		scattered_dir = rm_rej ? rmn : normalize(rm_sample.wo_x*rmtan + rm_sample.wo_y*rmbitan + rm_sample.wo_z*rmn);
		scattered   = true;

		// Real NEE/MIS for glossy (non-EffectivelySmooth) rough metal, via
		// wf_finish_material_scatter's shared evalGlossyF - see
		// MaterialType::Conductor's identical-shape block above.
		if (!rm_dist.EffectivelySmooth()) {
			is_specular = false;
			brdf_pdf_override = rm_rej ? -1.0f : rm_sample.pdf;
			phaseWo = rmwi;
		} else {
			is_specular = true;
		}
		attenuation = rm_rej ? SS(0.f) : albedoSpectrum(make_float3(mat.albedo.x * rm_sample.weight, mat.albedo.y * rm_sample.weight, mat.albedo.z * rm_sample.weight));
		break;
	}
	case MaterialType::CoatedDiffuse: {
		// Dielectric coat over a Lambertian base (pbrt-v4 CoatedDiffuseBxDF) - see wf_layered_scatter().
		// "reflectance" bound to a real Texture (pbrt's own ganesha/barcelona-pavilion "texture reflectance" - see
		// pbrt_flatten::Material::textureFilename's own comment) instead of mat.albedo's flat colour when
		// textureIdx>=0 - same pattern the Lambertian case above already uses, scaled by mat.emissionScale
		// (reused here for CoatedDiffuse's own "scale"-wrapped-imagemap case - see that field's own comment in
		// optix_types.h). wf_finish_material_scatter's own evalGlossyF (its CoatedDiffuse branch) does the identical
		// lookup for its NEE f() evaluation too, via the uv_u/uv_v this function threads through to it.
		const float3 cd_albedo = (mat.textureIdx >= 0)
			? wf_sample_texture(textures, texturePixels, mat.textureIdx, h.uv_u, h.uv_v, hit_point) * mat.emissionScale
			: mat.albedo;
		float cd_alpha_x = wf_glossy_alpha(mat, do_regularize);
		float cd_alpha_y = wf_glossy_alpha_v(mat, do_regularize);
		glossyAlphaForNEE = cd_alpha_x;
		glossyAlphaVForNEE = cd_alpha_y;
		CoatedDiffuseBxDF<float> cd_bxdf{ cd_albedo.x, cd_albedo.y, cd_albedo.z, mat.ior, cd_alpha_x, cd_alpha_y };
		const WfLayeredScatter cd_ls = wf_layered_scatter(cd_bxdf, normal, h.rayDir, h.objDpdu, seed);
		if (!cd_ls.scattered) { scattered = false; break; }
		scattered_dir = cd_ls.dir;
		attenuation   = cd_ls.rejected ? SS(0.f)
		              : (fmaxf(cd_ls.weight.x, fmaxf(cd_ls.weight.y, cd_ls.weight.z)) > 1.0f ? unboundedSpectrum(cd_ls.weight) : albedoSpectrum(cd_ls.weight));
		scattered     = true;
		if (cd_ls.nee) {
			is_specular       = false;
			brdf_pdf_override = cd_ls.rejected ? -1.0f : cd_ls.misPdf;
			phaseWo           = -normalize(h.rayDir);
		} else {
			is_specular = true;
		}
		break;
	}
	case MaterialType::ThinDielectric: {
		// Was: a hand-rolled Fr/(Fr+T^2) weighting (T=1-Fr) - a REAL, pre-
		// existing bug found while adding DielectricMedium's own thin-fused
		// variant (see wf_thin_dielectric_scatter()'s own comment,
		// wavefront_device_helpers.h): that formula does not match pbrt-v4's
		// ThinDielectricBxDF, and diverges substantially from it (e.g.
		// R=0.1: old gives ~0.110, correct gives ~0.182) - both the
		// authoritative shared src/shared/bxdfs_simple.h::ThinDielectricBxDF
		// (used by CPU, material_pbrt.h's thin_dielectric) and the recursive
		// GPU backend's own MaterialType::ThinDielectric case (optix_device_
		// helpers.h) already used the correct R_eff = R + T^2*R/(1-R^2)
		// multi-bounce geometric series; only this wavefront case had its
		// own independently-wrong formula. Now shares the same corrected
		// helper DielectricMedium's thin-fused case uses just below.
		scattered_dir = wf_thin_dielectric_scatter(h.rayDir, normal, mat.ior, seed);
		attenuation = SS(1.f);
		scattered   = true;
		is_specular = true;
		break;
	}
	case MaterialType::CoatedConductor: {
		// Dielectric coat over a GGX conductor (pbrt-v4 CoatedConductorBxDF) - see wf_layered_scatter(). mat.fuzz/
		// mat.roughnessV are the COAT's roughness; the conductor has its own (mat.condRoughness/condRoughnessV, negative
		// = the coat's). As in pbrt's CoatedConductorMaterial::GetBxDF the conductor's complex IOR is relative to the
		// coat: eta and k are both divided by the coat's IOR.
		float cc_alpha_x = wf_glossy_alpha(mat, do_regularize);
		float cc_alpha_y = wf_glossy_alpha_v(mat, do_regularize);
		float cc_cond_x, cc_cond_y;
		{
			float coat_ax_raw = mat.remapRoughness ? sqrtf(mat.fuzz) : mat.fuzz;
			float coat_ay_raw = ResolveAnisotropicAlphaV(mat.roughnessV, mat.fuzz, mat.remapRoughness);
			ResolveCoatedConductorBaseAlpha(mat.condRoughness, mat.condRoughnessV, mat.remapRoughness,
			                                coat_ax_raw, coat_ay_raw, cc_cond_x, cc_cond_y);
			if (do_regularize) { cc_cond_x = RegularizeAlpha(cc_cond_x); cc_cond_y = RegularizeAlpha(cc_cond_y); }
		}
		glossyAlphaForNEE = cc_alpha_x;
		glossyAlphaVForNEE = cc_alpha_y;
		glossyCondAlphaForNEE = cc_cond_x;
		glossyCondAlphaVForNEE = cc_cond_y;
		const float cc_inv_ior = 1.0f / mat.ior;
		CoatedConductorBxDF<float> cc_bxdf{ mat.eta_c.x * cc_inv_ior, mat.eta_c.y * cc_inv_ior, mat.eta_c.z * cc_inv_ior,
		                                    mat.k_c.x * cc_inv_ior, mat.k_c.y * cc_inv_ior, mat.k_c.z * cc_inv_ior,
		                                    mat.ior, cc_alpha_x, cc_alpha_y, mat.layerThickness, 0.0f, 0.0f, 10, 1,
		                                    cc_cond_x, cc_cond_y };
		const WfLayeredScatter cc_ls = wf_layered_scatter(cc_bxdf, normal, h.rayDir, h.objDpdu, seed);
		if (!cc_ls.scattered) { scattered = false; break; }
		scattered_dir = cc_ls.dir;
		attenuation   = cc_ls.rejected ? SS(0.f)
		              : (fmaxf(cc_ls.weight.x, fmaxf(cc_ls.weight.y, cc_ls.weight.z)) > 1.0f ? unboundedSpectrum(cc_ls.weight) : albedoSpectrum(cc_ls.weight));
		scattered     = true;
		if (cc_ls.nee) {
			is_specular       = false;
			brdf_pdf_override = cc_ls.rejected ? -1.0f : cc_ls.misPdf;
			phaseWo           = -normalize(h.rayDir);
		} else {
			is_specular = true;
		}
		break;
	}
	case MaterialType::DiffuseTransmission: {
		// pbrt-v4 DiffuseTransmissionBxDF - matches optix_intersection_sphere.h's
		// recursive-path handling exactly: albedo = reflectance R (same
		// hemisphere), emission = transmittance T (reused field, not
		// derived as 1-R), max-component probability weighting.
		//
		// is_specular = true (this material's BSDF is genuinely non-specular,
		// but the flag here means "skip the caller's generic NEE block", see
		// below): recursive's own shade_material() case for this material has
		// no explicit light-sampling code at all, so setting is_specular=false
		// here (matching a naive reading of "diffuse = non-specular") made
		// this branch the ONLY one of the two backends' NEE blocks that ran
		// for DiffuseTransmission - and it ran with the caller's hardcoded
		// Lambertian-shaped 1/pi white BRDF default, which is wrong on two
		// counts: it ignores mat.albedo's actual color entirely, and it can't
		// tell the reflective lobe (BRDF = R/pi) from the transmissive one
		// (BRDF = T/pi, and only for a light on the FAR side of the surface,
		// which the block's `dot(to_light, normal) > 0` gate would wrongly
		// reject anyway). CPU's own diffuse_transmission material (see
		// material_pbrt.h) DOES support real two-hemisphere NEE via its own
		// cosine_pdf(±normal)/skip_pdf=false machinery - a real BRDF-aware fix
		// here would port that, not just disable NEE - but until that lands on
		// both GPU backends, disabling it here removes the wrong, colour-blind
		// contribution and matches what recursive already (implicitly) ships:
		// zero explicit NEE, all illumination via the BSDF-sampled bounce.
		// Real per-point value when texture-bound (barcelona-pavilion's
		// foliage - see pbrt_flatten::Material::textureFilename/
		// transmittanceTextureFilename's own comments), else the flat
		// mat.albedo/mat.emission fallback - mirrors optix_device_helpers.h's
		// identical recursive-backend case.
		float3 R = (mat.textureIdx >= 0)
			? wf_sample_texture(textures, texturePixels, mat.textureIdx, h.uv_u, h.uv_v, hit_point) * mat.emissionScale
			: mat.albedo;
		float3 T_col = (mat.transmittanceTextureIdx >= 0)
			? wf_sample_texture(textures, texturePixels, mat.transmittanceTextureIdx, h.uv_u, h.uv_v, hit_point) * mat.transmittanceScale
			: mat.emission;
		float pr = fmaxf(R.x, fmaxf(R.y, R.z));
		float pt = fmaxf(T_col.x, fmaxf(T_col.y, T_col.z));
		if (pr + pt <= 0.0f) { scattered = false; break; }
		// Path weight f * cos / pdf = R / p_lobe for a cosine lobe chosen with probability p_lobe (pr / (pr + pt) or pt / (pr + pt)), as pbrt-v4's
		// DiffuseTransmissionBxDF::Sample_f; the weight used to be R (or T) alone, a factor p_lobe too small (see optix_device_helpers.h).
		const float p_refl = pr / (pr + pt);
		if (wf_rand(seed) < p_refl) {
			scattered_dir = normalize(normal + wf_rand_unit(seed));
			if (wf_near_zero(scattered_dir)) scattered_dir = normal;
			attenuation = albedoSpectrum(R / p_refl);
		} else {
			float3 neg_n = -normal;
			scattered_dir = normalize(neg_n + wf_rand_unit(seed));
			if (wf_near_zero(scattered_dir)) scattered_dir = neg_n;
			attenuation = albedoSpectrum(T_col / (1.0f - p_refl));
		}
		scattered   = true;
		is_specular = true;
		break;
	}
	case MaterialType::NormalizedFresnel: {
		float weight;
		wf_sample_normalized_fresnel(mat.ior, normal, seed, scattered_dir, weight, brdf_pdf_override);
		attenuation = SS(weight);
		scattered   = true;
		is_specular = false;
		break;
	}
	case MaterialType::Medium: {
		// A scatter in pbrt's camera medium (h.geomType == kWfGeomCameraMedium): there is no boundary geometry here -
		// __raygen__wf_trace already sampled the free flight against the nearest surface and found a collision at
		// h.hitPoint, so go straight to the phase-function scatter (the same hand-off to wf_finish_material_scatter's
		// phase NEE that the in-geometry branch below uses).
		if (h.geomType == kWfGeomCameraMedium) {
			const float3 unit_dir = normalize(h.rayDir);
			mediumMeanFreePath = wf_restir_volume_mean_free_path(mat.ior);
			hit_point     = h.hitPoint;
			scattered_dir = wf_sample_phase_scatter(unit_dir, mat.fuzz, seed, phaseWo, phaseG, brdf_pdf_override);
			// A camera medium with per-wavelength extinction: the collision weights are a function of the hit distance (h.t), drawn in __raygen__wf_trace.
			const bool cmChroma = wf_medium_is_chromatic(mat);
			SS cmW(1.f), cmE(0.f);
			if (cmChroma) {
				float cw[kWFNWavelengths], ce[kWFNWavelengths];
				if (wf_chroma_collision_at(mat, swl.lambda, h.t, cw, ce))
					for (int i = 0; i < kWFNWavelengths; ++i) { cmW[i] = cw[i]; cmE[i] = ce[i]; }
			}
			attenuation   = cmChroma ? cmW : unboundedSpectrum(mat.albedo);   // tint * sigma_s/sigma_t; the tint can exceed 1, so unbounded
			is_specular   = false;
			if (cmChroma) {
				if (mat.chromaLe.x > 0.0f || mat.chromaLe.y > 0.0f || mat.chromaLe.z > 0.0f)
					radiance = radiance + throughput * (wf_lift_rgb_to_spectrum(mat.chromaLe, swl, /*isIlluminant=*/true) * cmE);
			} else if (mat.medium_emission.x > 0.0f || mat.medium_emission.y > 0.0f || mat.medium_emission.z > 0.0f)
				radiance = radiance + throughput * wf_lift_rgb_to_spectrum(mat.medium_emission, swl, /*isIlluminant=*/true);
			scattered = true;
			break;
		}
		// Homogeneous participating medium - mirrors optix_intersection_sphere.h's
		// closesthit Medium case. Geometry (entry/exit roots) was already
		// recomputed in __closesthit__wf_sphere and handed over via h.t (near)
		// and h.mediumTFar (far); here we sample the free-path distance and
		// either scatter inside via the HG phase function or pass straight
		// through to the exit point.
		float t_near = h.t;
		float t_far  = h.mediumTFar;
		float dist_inside = fmaxf(0.0f, t_far - t_near);
		float sigma_t = mat.ior;
		mediumMeanFreePath = wf_restir_volume_mean_free_path(sigma_t);
		float free_path = (sigma_t > 1e-8f) ? (-logf(fmaxf(1e-8f, 1.0f - wf_rand(seed))) / sigma_t) : 1e30f;

		// Per-channel extinction (see MaterialData::chromaSigmaA): the free flight is drawn per hero wavelength with the balance-heuristic
		// estimator (wf_chroma_event); chroma_w is the collision's path weight, or the pass-through weight when the ray gets through.
		const bool chroma = wf_medium_is_chromatic(mat);
		bool chroma_collided = false;
		SS chroma_w(1.f), chroma_e(0.f);
		if (chroma) {
			const float cu_channel = wf_rand(seed);
			const float cu_dist = wf_rand(seed);
			const WfChromaEvent cev = wf_chroma_event(mat, swl.lambda, dist_inside, cu_channel, cu_dist);
			chroma_collided = cev.collided;
			free_path = cev.collided ? cev.t : 1e30f;
			for (int i = 0; i < kWFNWavelengths; ++i) { chroma_w[i] = cev.w[i]; chroma_e[i] = cev.e[i]; }
		}
		float3 unit_dir = normalize(h.rayDir);
		if (chroma ? chroma_collided : (free_path < dist_inside)) {
			float medium_t = t_near + free_path;
			hit_point     = h.rayOrigin + medium_t * unit_dir;
			// Real NEE+MIS at the phase-function scatter event, matching
			// CPU's hg_phase_material (skip_pdf=false) - see
			// optix_intersection_sphere.h's identical fix for the full
			// root-cause derivation. This backend defers the actual light
			// sampling/shadow-ray tracing to wf_finish_material_scatter()
			// (below) rather than tracing inline - wf_sample_phase_scatter()
			// handing off phaseWo/phaseG/brdf_pdf_override and flipping
			// is_specular is all this case needs to do (see DielectricMedium's
			// own interior sub-case, same file, for the identical mechanism
			// already wired up).
			scattered_dir = wf_sample_phase_scatter(unit_dir, mat.fuzz, seed, phaseWo, phaseG, brdf_pdf_override);
			// Medium albedo is sigma_s/luminance(sigma_s) per channel (pbrt_gpu_builder.h),
			// which EXCEEDS 1 for any channel brighter than the luminance - E11's
			// sigma_s=(1.2,0.8,0.4) gives R=1.40. albedoSpectrum() clamps each channel
			// to [0,1] (it is a reflectance uplift), so the wavefront backend silently
			// capped that channel and rendered ~12-15% darker than both CPU and
			// GPU-recursive (which multiply the raw RGB). unboundedSpectrum() is the
			// right uplift for an unbounded weight. A grey medium (albedo<=1) was
			// never affected, which is how this was isolated.
			attenuation   = chroma ? chroma_w : unboundedSpectrum(mat.albedo);
			is_specular   = false;
			// MakeNamedMedium's own "rgb Le"/"float Lescale" (pbrt-v4) - see
			// MaterialData::medium_emission's own comment (optix_types.h) for
			// the sigma_a/sigma_t weighting already baked in at build time.
			// Added unconditionally with respect to MIS (no specular_bounce/
			// depth==0 gate, unlike DiffuseLight's own direct-hit emission
			// above) - this medium is never a member of the light list and so
			// can never be NEE-sampled, matching CPU's own unconditional
			// hg_phase_material::emitted() call. throughput here is
			// deliberately the PRE-collision value (this event's own
			// attenuation is folded in later, at wf_finish_material_scatter's
			// new_throughput), matching CPU's beta timing at a medium-emission
			// vertex exactly.
			//
			// The `if` below is a SEPARATE, purely-performance zero-check (not
			// the MIS gate the comment above describes) - it skips the
			// dev_srgb_to_coeffs/per-wavelength-loop spectral uplift entirely
			// for the common case of a plain fog Medium with no "Le" at all.
			if (chroma) {
					if (mat.chromaLe.x > 0.0f || mat.chromaLe.y > 0.0f || mat.chromaLe.z > 0.0f)
						radiance = radiance + throughput * (wf_lift_rgb_to_spectrum(mat.chromaLe, swl, /*isIlluminant=*/true) * chroma_e);
				} else if (mat.medium_emission.x > 0.0f || mat.medium_emission.y > 0.0f || mat.medium_emission.z > 0.0f)
				radiance = radiance + throughput * wf_lift_rgb_to_spectrum(mat.medium_emission, swl, /*isIlluminant=*/true);
		} else {
			hit_point     = h.rayOrigin + t_far * unit_dir;
			scattered_dir = unit_dir;  // straight through, no interaction
			attenuation   = chroma ? chroma_w : SS(1.f);
			is_specular   = true;  // no interaction - a free/non-scattering pass-through
			// A free crossing, like MaterialType::Interface: flag it so the last real vertex's MIS state (prev BSDF pdf,
			// specular flag) survives it instead of being reset as if a specular bounce had happened.
			is_medium_boundary = true;
		}
		scattered   = true;
		break;
	}
	case MaterialType::CloudMedium: {
		// Heterogeneous, procedural Perlin-noise cloud - mirrors
		// optix_intersection_sphere.h's closesthit CloudMedium case exactly
		// (see that comment for the full reasoning). Unlike Medium above,
		// this does NOT use h.t/h.mediumTFar (the sphere's own near/far
		// roots, which __closesthit__wf_sphere only recomputes for
		// MaterialType::Medium/DielectricMedium - see that function's
		// needsNearFar) - CloudMedium's own world-space AABB, tested fresh
		// against the true ray via CloudMedium<float>::sample_ray(), is what
		// actually bounds it; the trigger sphere's own geometry is
		// irrelevant beyond having gotten the ray into this branch at all.
		const CloudMedium<float>& cloud = cloudMediums[(int)mat.cloud_medium_extra.cloudMediumIdx];
		float3 unit_dir = normalize(h.rayDir);
		float ray_o3[3] = { h.rayOrigin.x, h.rayOrigin.y, h.rayOrigin.z };
		float ray_d3[3] = { unit_dir.x, unit_dir.y, unit_dir.z };

		auto maj_it = cloud.sample_ray(ray_o3, ray_d3, 1e30f);
		float segMin, segMax, sigma_maj;
		bool has_seg = maj_it.next(segMin, segMax, sigma_maj);
		if (has_seg) mediumMeanFreePath = wf_restir_volume_mean_free_path(sigma_maj);

		bool did_scatter = false;
		float medium_t = 0.0f;
		if (has_seg && sigma_maj > 0.0f) {
			if (segMin < 0.0f) segMin = 0.0f;
			float tt = segMin;
			// Cap matches optix_intersection_sphere.h's CloudMedium branch.
			for (int iter = 0; iter < 128 && !did_scatter; ++iter) {
				float dt = -logf(fmaxf(1e-8f, 1.0f - wf_rand(seed))) / sigma_maj;
				tt += dt;
				if (tt >= segMax) break;
				float3 p = h.rayOrigin + tt * unit_dir;
				float mx, my, mz;
				cloud.world_to_medium_pt(p.x, p.y, p.z, mx, my, mz);
				float d = gpu_cloud_density(cloud, mx, my, mz);
				float sigma_s_local = d * cloud.sigma_s;
				if (wf_rand(seed) < sigma_s_local / sigma_maj) {
					did_scatter = true;
					medium_t      = tt;
					// Real NEE+MIS at the phase-function scatter event - see
					// MaterialType::Medium's identical fix above.
					scattered_dir = wf_sample_phase_scatter(unit_dir, mat.fuzz, seed, phaseWo, phaseG, brdf_pdf_override);
					attenuation   = albedoSpectrum(mat.albedo);
				}
			}
			if (!did_scatter) medium_t = segMax;
		} else {
			medium_t = h.t;  // missed the medium's own AABB - pass straight through
		}
		hit_point = h.rayOrigin + medium_t * unit_dir;
		if (did_scatter) {
			is_specular = false;
		} else {
			scattered_dir = unit_dir;
			attenuation   = SS(1.f);
			is_specular   = true;  // no interaction - a free/non-scattering pass-through
			// A free crossing, like MaterialType::Interface: flag it so the last real vertex's MIS state (prev BSDF pdf,
			// specular flag) survives it instead of being reset as if a specular bounce had happened.
			is_medium_boundary = true;
		}
		scattered   = true;
		break;
	}

	case MaterialType::RgbGridMedium: {
		// Heterogeneous per-voxel R/G/B scattering grid - mirrors
		// optix_intersection_sphere.h's closesthit RgbGridMedium case exactly
		// (see that comment for the full reasoning, including why this uses
		// one single GLOBAL majorant rather than CPU's real per-voxel DDA
		// majorant grid).
		const GpuRgbGridMedium& grid = rgbGridMediums[(int)mat.rgb_grid_medium_extra.rgbGridMediumIdx];
		mediumMeanFreePath = wf_restir_volume_mean_free_path(grid.sigma_maj);
		float3 unit_dir = normalize(h.rayDir);

		float mox = grid.mat[0]*h.rayOrigin.x + grid.mat[1]*h.rayOrigin.y + grid.mat[2]*h.rayOrigin.z + grid.translate[0];
		float moy = grid.mat[3]*h.rayOrigin.x + grid.mat[4]*h.rayOrigin.y + grid.mat[5]*h.rayOrigin.z + grid.translate[1];
		float moz = grid.mat[6]*h.rayOrigin.x + grid.mat[7]*h.rayOrigin.y + grid.mat[8]*h.rayOrigin.z + grid.translate[2];
		float mdx = grid.mat[0]*unit_dir.x + grid.mat[1]*unit_dir.y + grid.mat[2]*unit_dir.z;
		float mdy = grid.mat[3]*unit_dir.x + grid.mat[4]*unit_dir.y + grid.mat[5]*unit_dir.z;
		float mdz = grid.mat[6]*unit_dir.x + grid.mat[7]*unit_dir.y + grid.mat[8]*unit_dir.z;

		float segMin = 0.0f, segMax = 1e30f;
		bool has_seg = true;
		{
			float invd, s0, s1;
			invd = (mdx != 0.0f) ? 1.0f/mdx : 1e30f;
			s0 = (0.0f - mox)*invd; s1 = (1.0f - mox)*invd;
			if (s0 > s1) { float tmp = s0; s0 = s1; s1 = tmp; }
			segMin = fmaxf(segMin, s0); segMax = fminf(segMax, s1);
			if (segMin > segMax) has_seg = false;

			invd = (mdy != 0.0f) ? 1.0f/mdy : 1e30f;
			s0 = (0.0f - moy)*invd; s1 = (1.0f - moy)*invd;
			if (s0 > s1) { float tmp = s0; s0 = s1; s1 = tmp; }
			segMin = fmaxf(segMin, s0); segMax = fminf(segMax, s1);
			if (segMin > segMax) has_seg = false;

			invd = (mdz != 0.0f) ? 1.0f/mdz : 1e30f;
			s0 = (0.0f - moz)*invd; s1 = (1.0f - moz)*invd;
			if (s0 > s1) { float tmp = s0; s0 = s1; s1 = tmp; }
			segMin = fmaxf(segMin, s0); segMax = fminf(segMax, s1);
			if (segMin > segMax) has_seg = false;
		}

		bool did_scatter = false;
		float medium_t = 0.0f;
		// Running product of the null-collision weights: the path weight of a ray that gets through without a real collision.
		SS grid_w(1.f);
		if (has_seg && grid.sigma_maj > 0.0f) {
			if (segMin < 0.0f) segMin = 0.0f;
			float tt = segMin;
			const int voxelCount = grid.nx * grid.ny * grid.nz;
			const float* sRData = rgbGridData + grid.dataOffset;
			const float* sGData = sRData + voxelCount;
			const float* sBData = sGData + voxelCount;
			const bool hasSa = grid.saDataOffset >= 0;
			const float* aRData = hasSa ? rgbGridData + grid.saDataOffset : nullptr;
			const float* aGData = hasSa ? aRData + voxelCount : nullptr;
			const float* aBData = hasSa ? aGData + voxelCount : nullptr;
			for (int iter = 0; iter < 128 && !did_scatter; ++iter) {
				float dt = -logf(fmaxf(1e-8f, 1.0f - wf_rand(seed))) / grid.sigma_maj;
				tt += dt;
				if (tt >= segMax) break;
				float px = mox + tt*mdx, py = moy + tt*mdy, pz = moz + tt*mdz;
				// Per-voxel RGB sigma_s / sigma_a at the tentative point (an absent sigma_a grid is pbrt's default 1), spread over the hero
				// wavelengths with wf_rgb_wavelength_basis, then one step of per-wavelength spectral tracking (heterogeneous_tracking_step).
				const float ssRgb[3] = { gpu_rgb_grid_trilinear(sRData, grid.nx, grid.ny, grid.nz, px, py, pz) * grid.sigma_scale,
				                         gpu_rgb_grid_trilinear(sGData, grid.nx, grid.ny, grid.nz, px, py, pz) * grid.sigma_scale,
				                         gpu_rgb_grid_trilinear(sBData, grid.nx, grid.ny, grid.nz, px, py, pz) * grid.sigma_scale };
				const float saRgb[3] = { (hasSa ? gpu_rgb_grid_trilinear(aRData, grid.nx, grid.ny, grid.nz, px, py, pz) : 1.0f) * grid.sigma_scale,
				                         (hasSa ? gpu_rgb_grid_trilinear(aGData, grid.nx, grid.ny, grid.nz, px, py, pz) : 1.0f) * grid.sigma_scale,
				                         (hasSa ? gpu_rgb_grid_trilinear(aBData, grid.nx, grid.ny, grid.nz, px, py, pz) : 1.0f) * grid.sigma_scale };
				float ssN[kWFNWavelengths], saN[kWFNWavelengths], wN[kWFNWavelengths], cwN[kWFNWavelengths], ceN[kWFNWavelengths];
				wf_rgb_to_wavelengths(ssRgb, swl.lambda, ssN);
				wf_rgb_to_wavelengths(saRgb, swl.lambda, saN);
				for (int i = 0; i < kWFNWavelengths; ++i) wN[i] = grid_w[i];
				const float u_real = wf_rand(seed);
				if (heterogeneous_tracking_step<kWFNWavelengths, float>(saN, ssN, grid.sigma_maj, u_real, wN, cwN, ceN)) {
					did_scatter = true;
					medium_t      = tt;
					// Real NEE+MIS at the phase-function scatter event - see MaterialType::Medium's identical fix above.
					scattered_dir = wf_sample_phase_scatter(unit_dir, grid.phase_g, seed, phaseWo, phaseG, brdf_pdf_override);
					for (int i = 0; i < kWFNWavelengths; ++i) attenuation[i] = cwN[i];
					// Real per-voxel "rgb Le" at this collision, weighted by the event's absorption share as on the CPU
					// (Le_c * w_c * sigma_a_c / mean(sigma_t)); added directly to radiance (this backend's per-case convention, see
					// MaterialType::Medium above) since wavefront's NEE for this hit happens in a later kernel with no grid data.
					if (grid.leDataOffset >= 0) {
						const float* leRData = rgbGridData + grid.leDataOffset;
						const float* leGData = leRData + voxelCount;
						const float* leBData = leGData + voxelCount;
						float ler = gpu_rgb_grid_trilinear(leRData, grid.nx, grid.ny, grid.nz, px, py, pz);
						float leg = gpu_rgb_grid_trilinear(leGData, grid.nx, grid.ny, grid.nz, px, py, pz);
						float leb = gpu_rgb_grid_trilinear(leBData, grid.nx, grid.ny, grid.nz, px, py, pz);
						float3 selfEmission = make_float3(ler, leg, leb) * grid.Le_scale;
						if (selfEmission.x > 0.0f || selfEmission.y > 0.0f || selfEmission.z > 0.0f) {
							SS ceSpec(0.f);
							for (int i = 0; i < kWFNWavelengths; ++i) ceSpec[i] = ceN[i];
							radiance = radiance + throughput * (wf_lift_rgb_to_spectrum(selfEmission, swl, /*isIlluminant=*/true) * ceSpec);
						}
					}
				} else {
					for (int i = 0; i < kWFNWavelengths; ++i) grid_w[i] = wN[i];
				}
			}
			if (!did_scatter) medium_t = segMax;
		} else {
			medium_t = h.t;
		}
		hit_point = h.rayOrigin + medium_t * unit_dir;
		if (did_scatter) {
			is_specular = false;
		} else {
			scattered_dir = unit_dir;
			attenuation   = grid_w;   // pass weight of the null collisions
			is_specular   = true;  // no interaction - a free/non-scattering pass-through
			// A free crossing, like MaterialType::Interface: flag it so the last real vertex's MIS state (prev BSDF pdf,
			// specular flag) survives it instead of being reset as if a specular bounce had happened.
			is_medium_boundary = true;
		}
		scattered   = true;
		break;
	}
	case MaterialType::GridMedium: {
		// Heterogeneous single-channel scalar density grid - single-channel
		// twin of the RgbGridMedium case just above (see that one's own
		// comment; same single-GLOBAL-majorant simplification), and mirrors
		// optix_intersection_sphere.h's closesthit GridMedium case exactly.
		const GpuGridMedium& grid = gridMediums[(int)mat.grid_medium_extra.gridMediumIdx];
		mediumMeanFreePath = wf_restir_volume_mean_free_path(grid.sigma_maj);
		float3 unit_dir = normalize(h.rayDir);

		float mox = grid.mat[0]*h.rayOrigin.x + grid.mat[1]*h.rayOrigin.y + grid.mat[2]*h.rayOrigin.z + grid.translate[0];
		float moy = grid.mat[3]*h.rayOrigin.x + grid.mat[4]*h.rayOrigin.y + grid.mat[5]*h.rayOrigin.z + grid.translate[1];
		float moz = grid.mat[6]*h.rayOrigin.x + grid.mat[7]*h.rayOrigin.y + grid.mat[8]*h.rayOrigin.z + grid.translate[2];
		float mdx = grid.mat[0]*unit_dir.x + grid.mat[1]*unit_dir.y + grid.mat[2]*unit_dir.z;
		float mdy = grid.mat[3]*unit_dir.x + grid.mat[4]*unit_dir.y + grid.mat[5]*unit_dir.z;
		float mdz = grid.mat[6]*unit_dir.x + grid.mat[7]*unit_dir.y + grid.mat[8]*unit_dir.z;

		float segMin = 0.0f, segMax = 1e30f;
		bool has_seg = true;
		{
			float invd, s0, s1;
			invd = (mdx != 0.0f) ? 1.0f/mdx : 1e30f;
			s0 = (0.0f - mox)*invd; s1 = (1.0f - mox)*invd;
			if (s0 > s1) { float tmp = s0; s0 = s1; s1 = tmp; }
			segMin = fmaxf(segMin, s0); segMax = fminf(segMax, s1);
			if (segMin > segMax) has_seg = false;

			invd = (mdy != 0.0f) ? 1.0f/mdy : 1e30f;
			s0 = (0.0f - moy)*invd; s1 = (1.0f - moy)*invd;
			if (s0 > s1) { float tmp = s0; s0 = s1; s1 = tmp; }
			segMin = fmaxf(segMin, s0); segMax = fminf(segMax, s1);
			if (segMin > segMax) has_seg = false;

			invd = (mdz != 0.0f) ? 1.0f/mdz : 1e30f;
			s0 = (0.0f - moz)*invd; s1 = (1.0f - moz)*invd;
			if (s0 > s1) { float tmp = s0; s0 = s1; s1 = tmp; }
			segMin = fmaxf(segMin, s0); segMax = fminf(segMax, s1);
			if (segMin > segMax) has_seg = false;
		}

		bool did_scatter = false;
		float medium_t = 0.0f;
		if (has_seg && grid.sigma_maj > 0.0f) {
			if (segMin < 0.0f) segMin = 0.0f;
			float tt = segMin;
			const float* dData = gridData + grid.dataOffset;
			for (int iter = 0; iter < 128 && !did_scatter; ++iter) {
				float dt = -logf(fmaxf(1e-8f, 1.0f - wf_rand(seed))) / grid.sigma_maj;
				tt += dt;
				if (tt >= segMax) break;
				float px = mox + tt*mdx, py = moy + tt*mdy, pz = moz + tt*mdz;
				float d = gpu_rgb_grid_trilinear(dData, grid.nx, grid.ny, grid.nz, px, py, pz);
				float sigma_t_local = d * grid.sigma_scale;
				if (wf_rand(seed) < sigma_t_local / grid.sigma_maj) {
					did_scatter = true;
					medium_t      = tt;
					// Real NEE+MIS at the phase-function scatter event - see
					// MaterialType::Medium's identical fix above.
					scattered_dir = wf_sample_phase_scatter(unit_dir, grid.phase_g, seed, phaseWo, phaseG, brdf_pdf_override);
					attenuation   = albedoSpectrum(mat.albedo);  // no per-voxel colour - see grid_medium_hittable.h's own comment
				}
			}
			if (!did_scatter) medium_t = segMax;
		} else {
			medium_t = h.t;
		}
		hit_point = h.rayOrigin + medium_t * unit_dir;
		if (did_scatter) {
			is_specular = false;
		} else {
			scattered_dir = unit_dir;
			attenuation   = SS(1.f);
			is_specular   = true;  // no interaction - a free/non-scattering pass-through
			// A free crossing, like MaterialType::Interface: flag it so the last real vertex's MIS state (prev BSDF pdf,
			// specular flag) survives it instead of being reset as if a specular bounce had happened.
			is_medium_boundary = true;
		}
		scattered   = true;
		break;
	}
	case MaterialType::Hair: {
		// Marschner/Chiang fiber scattering - see wf_sample_hair_material's
		// comment above. Matches hair_material.h's skip_pdf=true: no NEE/MIS
		// (is_specular=true), and the result needs unboundedSpectrum (not
		// albedoSpectrum) since it's already divided by the sample pdf.
		//
		// Fiber tangent: a bilinear patch (geomType==2, the tessellated-curve
		// GPU path - see pbrt_gpu_builder.h/curve_tessellate.h) carries its
		// own genuine dpdu in h.objDpdu (see that field's own comment,
		// __closesthit__wf_bilinear_patch) - the real fiber axis, unlike
		// `normal` (perpendicular to the tube). Every other geomType falls
		// back to the shading-normal proxy, matching hair_material.h's own
		// default (tangent_is_dpdu=false) exactly. h.objDpdu is stored
		// UNNORMALIZED (see __closesthit__wf_bilinear_patch's own comment) -
		// normalized here, the only reader, with the same degenerate-dpdu
		// fallback hair_material.h's CPU-side fiber_tangent() uses.
		const bool hasCurveTangent = (h.geomType == 2) && (dot(h.objDpdu, h.objDpdu) > 1e-12f);
		const float3 hairTangent = hasCurveTangent ? normalize(h.objDpdu) : normal;
		float3 sdir, atten;
		if (wf_sample_hair_material(h.rayDir, hairTangent, mat, seed, sdir, atten)) {
			scattered_dir = sdir;
			attenuation   = unboundedSpectrum(atten);
			scattered     = true;
		} else {
			scattered = false;
		}
		is_specular = true;
		break;
	}
	case MaterialType::DielectricMedium: {
		// Combined dielectric surface + internal medium - mirrors
		// optix_intersection_sphere.h's closesthit case exactly. On entry
		// (h.frontFace) the direct dielectric surface always wins the bounce
		// (the medium's sampled hit distance can never be closer than the
		// entry surface), so just refract/reflect normally. On the exit
		// surface (frontFace false), __closesthit__wf_sphere already
		// recomputed the near/far roots into h.t/h.mediumTFar (same
		// mechanism as MaterialType::Medium above) - sample a free path
		// through that interior segment and either scatter via the HG phase
		// function or fall through to a normal exit refraction/reflection
		// at the far surface.
		//
		// surfaceKind: which BSDF model this fused material's surface uses
		// instead of always smooth Dielectric refraction - see pbrt_gpu_
		// builder.h's mediumMaterialIndex() for how a shape's own Material
		// "dielectric" (rough) / "thindielectric" (thin) sets dielectric_
		// medium_extra.surfaceKind (0=smooth/default, 1=thin, 2=rough), and
		// optix_intersection_sphere.h's identical DielectricMedium branch
		// for the full comment on all three. Thin needs no etaScale
		// adjustment (no actual refraction/depth change).
		const float rdm_kind = mat.dielectric_medium_extra.surfaceKind;
		const bool is_rough = rdm_kind >= 1.5f;
		const bool is_thin  = !is_rough && rdm_kind >= 0.5f;

		// Real GGX microfacet rough-dielectric scatter, shared by the entry
		// and exit surface sub-cases below (both need the identical math,
		// just with their own front_face). Mirrors evaluate_materials_
		// dielectric.cu's own MaterialType::RoughDielectric case exactly
		// (same shared CPU_GPU TrowbridgeReitz<float>/RoughDielectricBxDF<float>
		// templates), reading roughness from dielectric_medium_extra.
		// roughness instead of mat.fuzz - that union slot is already taken
		// by `g` (the medium's own Henyey-Greenstein asymmetry) for this
		// fused type, see optix_types.h's own comment.
		//
		// Returns false only in the rare grazing-angle degenerate case
		// (mirrors evaluate_materials_dielectric.cu's own `scattered=false;
		// break;` for the identical situation). On success, also sets
		// effectiveMatType to MaterialType::RoughDielectric when the glossy
		// (non-EffectivelySmooth) lobe fires, so the shared tail call below
		// reuses wf_finish_material_scatter's EXISTING, already-correct
		// RoughDielectric glossy-NEE path instead of its own DielectricMedium
		// handling (which unconditionally assumes a medium-interior phase
		// event) - see effectiveMatType's own declaration comment (above,
		// this function) for the full rationale on why this substitution was
		// chosen over a hand-duplicated inline NEE block.
		auto roughDielectricMediumScatter = [&](bool rd_front_face) -> bool {
			float rd_alpha_x, rd_alpha_y;
			if (mat.textureIdx >= 0) {
				const float rd_rough = wf_sample_texture(textures, texturePixels, mat.textureIdx, h.uv_u, h.uv_v, hit_point).x;
				const float rd_a = mat.remapRoughness ? sqrtf(rd_rough) : rd_rough;
				rd_alpha_x = rd_alpha_y = do_regularize ? RegularizeAlpha(rd_a) : rd_a;
			} else {
				const float rd_flat = mat.dielectric_medium_extra.roughness;
				float rd_a  = mat.remapRoughness ? sqrtf(rd_flat) : rd_flat;
				float rd_av = ResolveAnisotropicAlphaV(mat.roughnessV, rd_flat, mat.remapRoughness);
				rd_alpha_x = do_regularize ? RegularizeAlpha(rd_a)  : rd_a;
				rd_alpha_y = do_regularize ? RegularizeAlpha(rd_av) : rd_av;
			}
			glossyAlphaForNEE  = rd_alpha_x;
			glossyAlphaVForNEE = rd_alpha_y;
			const float rd_ri = rd_front_face ? (1.0f / mat.ior) : mat.ior;

			float3 n = normal;
			float3 tan_v, bitan;
			BuildDpduTangentFrame(n.x, n.y, n.z, h.objDpdu.x, h.objDpdu.y, h.objDpdu.z,
								  tan_v.x, tan_v.y, tan_v.z, bitan.x, bitan.y, bitan.z);
			float3 wi_w = normalize(-h.rayDir);
			float wi_x = dot(wi_w, tan_v), wi_y = dot(wi_w, bitan), wi_z = dot(wi_w, n);
			if (wi_z < 0.0f) { wi_z = -wi_z; wi_x = -wi_x; wi_y = -wi_y; }

			TrowbridgeReitz<float> rd_dist(rd_alpha_x, rd_alpha_y);
			float wm_x, wm_y, wm_z;
			rd_dist.Sample_wm(wi_x, wi_y, wi_z, wf_rand(seed), wf_rand(seed), wm_x, wm_y, wm_z);
			const float rd_dot = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
			const float Fr = FrDielectric(rd_dot, 1.0f / rd_ri);

			// A rejected continuation sample must not skip this vertex's NEE (see the RoughDielectric case in
			// wavefront_kernels_materials_dielectric.cu): it carries zero weight and the path ends after the NEE.
			bool rd_rej = false;
			float wo_x, wo_y, wo_z;
			if (wf_rand(seed) < Fr) {
				wo_x = 2.0f*rd_dot*wm_x - wi_x;
				wo_y = 2.0f*rd_dot*wm_y - wi_y;
				wo_z = 2.0f*rd_dot*wm_z - wi_z;
				if (wo_z <= 0.0f) rd_rej = true;
				else scattered_dir = normalize(wo_x*tan_v + wo_y*bitan + wo_z*n);
			} else {
				float3 wm_world = wm_x*tan_v + wm_y*bitan + wm_z*n;
				const float eta_ratio = rd_front_face ? (1.0f / mat.ior) : mat.ior;
				float3 refracted = wf_refract(normalize(h.rayDir), wm_world, eta_ratio);
				wo_x = dot(refracted, tan_v); wo_y = dot(refracted, bitan); wo_z = dot(refracted, n);
				// pbrt-v4 rejects a refracted sample that ends up on the same side as wo (DielectricBxDF::Sample_f: SameHemisphere(wo, wi)), which a strongly tilted
				// microfacet produces at grazing incidence. Keeping it added ~7% to the albedo leaving glass at 75 degrees.
				if (wo_z >= 0.0f) rd_rej = true;
				else {
					scattered_dir = refracted;
					eventEta = eta_ratio;
				}
			}
			if (rd_rej && rd_dist.EffectivelySmooth()) return false;   // a smooth lobe has no NEE to keep
			if (rd_rej) {
				attenuation   = SS(0.f);
				scattered_dir = normal;
			} else {
				const float wo_z_abs = fabsf(wo_z);
				const float G2 = rd_dist.G(wi_x, wi_y, wi_z, wo_x, wo_y, wo_z_abs);
				const float G1_wi = rd_dist.G1(wi_x, wi_y, wi_z);
				const float w = (G1_wi > 1e-8f) ? (G2 / G1_wi) : 0.0f;
				attenuation = SS(w);
			}
			if (!rd_dist.EffectivelySmooth()) {
				is_specular = false;
				matEta = rd_ri;
				effectiveMatType = MaterialType::RoughDielectric;
				RoughDielectricBxDF<float> rd_bxdf{ mat.ior, rd_alpha_x, rd_alpha_y };
				brdf_pdf_override = rd_rej ? -1.0f : rd_bxdf.pdf(wi_x, wi_y, wi_z, rd_ri, wo_x, wo_y, wo_z);
				phaseWo = wi_w;
			} else {
				is_specular = true;
			}
			return true;
		};

		if (h.frontFace) {
			if (is_rough) {
				if (!roughDielectricMediumScatter(true)) { scattered = false; break; }
			} else {
				attenuation   = SS(1.f);
				if (is_thin) {
					scattered_dir = wf_thin_dielectric_scatter(h.rayDir, normal, mat.ior, seed);
				} else {
					scattered_dir = wf_dielectric_scatter(h.rayDir, normal, true, mat.ior, seed);
					// pbrt-v4 etaScale (entry surface) - see MaterialType::
					// Dielectric's identical eta computation above.
					if (dot(scattered_dir, normal) < 0.0f) eventEta = 1.0f / mat.ior;
				}
				is_specular   = true;  // genuinely specular (Dirac-delta) dielectric bounce
			}
		} else {
			float t_near = h.t;
			float t_far  = h.mediumTFar;
			float dist_inside = fmaxf(0.0f, t_far - t_near);
			float sigma_t = mat.eta_c.x;  // dielectric_medium_extra.sigma_t
			mediumMeanFreePath = wf_restir_volume_mean_free_path(sigma_t);
			float free_path = (sigma_t > 1e-8f) ? (-logf(fmaxf(1e-8f, 1.0f - wf_rand(seed))) / sigma_t) : 1e30f;

			// Per-channel extinction (see MaterialData::chromaSigmaA): the free flight is drawn per hero wavelength with the balance-heuristic
			// estimator (wf_chroma_event); chroma_w is the collision's path weight, or the pass-through weight when the ray gets through.
			const bool chroma = wf_medium_is_chromatic(mat);
			bool chroma_collided = false;
			SS chroma_w(1.f), chroma_e(0.f);
			if (chroma) {
				const float cu_channel = wf_rand(seed);
				const float cu_dist = wf_rand(seed);
				const WfChromaEvent cev = wf_chroma_event(mat, swl.lambda, dist_inside, cu_channel, cu_dist);
				chroma_collided = cev.collided;
				free_path = cev.collided ? cev.t : 1e30f;
				for (int i = 0; i < kWFNWavelengths; ++i) { chroma_w[i] = cev.w[i]; chroma_e[i] = cev.e[i]; }
			}
			float3 unit_dir = normalize(h.rayDir);
			if (chroma ? chroma_collided : (free_path < dist_inside)) {
				float medium_t = t_near + free_path;
				hit_point     = h.rayOrigin + medium_t * unit_dir;
				// Real NEE+MIS at the phase-function scatter event, matching
				// CPU's hg_phase_material (skip_pdf=false) - see
				// optix_intersection_sphere.h's identical fix for the full
				// root-cause derivation (this closes B13/SubsurfaceSlab's
				// ~32-38% CPU-brighter gap). Unlike the recursive backend,
				// this path's shadow rays are queued (wf_finish_material_
				// scatter, below) rather than traced inline, so all this case
				// needs to do is hand off phaseWo/phaseG/brdf_pdf_override
				// (via wf_sample_phase_scatter()) and flip is_specular.
				scattered_dir = wf_sample_phase_scatter(unit_dir, mat.fuzz, seed, phaseWo, phaseG, brdf_pdf_override);
				// Unbounded uplift - see MaterialType::Medium's identical site above.
				attenuation   = chroma ? chroma_w : unboundedSpectrum(mat.albedo);
				is_specular = false;
			} else if (is_rough) {
				hit_point = h.rayOrigin + t_far * unit_dir;
					if (chroma) throughput = throughput * chroma_w;   // crossed the interior: the exit surface's direct light carries the pass weight
				if (!roughDielectricMediumScatter(false)) { scattered = false; break; }
			} else {
				hit_point     = h.rayOrigin + t_far * unit_dir;
					if (chroma) throughput = throughput * chroma_w;   // crossed the interior: the exit surface's direct light carries the pass weight
				attenuation   = SS(1.f);
				if (is_thin) {
					scattered_dir = wf_thin_dielectric_scatter(h.rayDir, normal, mat.ior, seed);
				} else {
					scattered_dir = wf_dielectric_scatter(h.rayDir, normal, false, mat.ior, seed);
					// pbrt-v4 etaScale (exit surface, front_face is false here) -
					// see MaterialType::Dielectric's identical eta computation
					// above.
					if (dot(scattered_dir, normal) < 0.0f) eventEta = mat.ior;
				}
				is_specular   = true;  // genuinely specular exit refraction/reflection
			}
		}
		scattered   = true;
		break;
	}
	case MaterialType::NormalMappedLambertian: {
		// Lambertian with a perturbed shading normal from a tangent-space
		// RGB normal-map texture - mirrors optix_intersection_sphere.h's
		// (spheres) / optix_intersection_triangle.h's (triangles) closesthit
		// cases. h.objDpdu now carries a real, ready-to-use (world-space)
		// dpdu for EVERY geomType (sphere/quad/triangle/disk/cylinder all
		// populate it at intersection time - see each __closesthit__wf_*'s
		// own comment; bilinear patch's is left unnormalized, matching Hair's
		// own reader) - previously this was only true for triangle, and
		// every other geometry derived an approximate cross(world_up,
		// raw_normal) tangent instead, which is also what the 4 anisotropy-
		// capable material kinds below now rely on this same field for.
		const float3 dpdu = h.objDpdu;

		// Decode the tangent-space normal from the map texture: 2*RGB-1,
		// normalize (fallback (0,0,1) i.e. "no perturbation" if degenerate) -
		// matches CPU's normal_map_material::apply() exactly.
		const float3 packed = wf_sample_texture(textures, texturePixels, mat.textureIdx, h.uv_u, h.uv_v, hit_point);
		float ns_x = 2.0f * packed.x - 1.0f;
		float ns_y = 2.0f * packed.y - 1.0f;
		float ns_z = 2.0f * packed.z - 1.0f;
		const float ns_len = sqrtf(ns_x * ns_x + ns_y * ns_y + ns_z * ns_z);
		if (ns_len > 1e-8f) { ns_x /= ns_len; ns_y /= ns_len; ns_z /= ns_len; }
		else                { ns_x = 0.0f; ns_y = 0.0f; ns_z = 1.0f; }

		float out_nx, out_ny, out_nz;
		apply_normal_map(ns_x, ns_y, ns_z, normal.x, normal.y, normal.z,
			dpdu.x, dpdu.y, dpdu.z, out_nx, out_ny, out_nz);
		normal = make_float3(out_nx, out_ny, out_nz);  // perturbed shading normal;
		                                                // NEE below reads this same
		                                                // outer-scope `normal`.

		// Shade as plain Lambertian (mat.albedo) using the perturbed normal -
		// reuses the Lambertian case's logic rather than duplicating it.
		// textureIdx means "normal map" for this material type, not "albedo
		// texture" (matches recursive - CPU never combines the two).
		scattered_dir = normalize(normal + wf_rand_unit(seed));
		if (wf_near_zero(scattered_dir)) scattered_dir = normal;
		attenuation = albedoSpectrum(mat.albedo);
		scattered   = true;
		is_specular = false;
		break;
	}
	case MaterialType::Principled: {
		// Disney/pbrt-v4-style multi-lobe BSDF - see wf_sample_principled_material's
		// comment above. Matches principled_material.h's skip_pdf=true: no
		// NEE/MIS (is_specular=true), and the result needs unboundedSpectrum
		// (not albedoSpectrum) since it's already divided by the sample pdf -
		// same convention as MaterialType::Hair above.
		float3 sdir, atten;
		if (wf_sample_principled_material(h.rayDir, normal, mat, seed, sdir, atten)) {
			scattered_dir = sdir;
			attenuation   = unboundedSpectrum(atten);
			scattered     = true;
		} else {
			scattered = false;
		}
		is_specular = true;
		break;
	}
	// Every MaterialType except DiffuseLight (handled via the early-exit
	// above, before this switch) has a real case above - this default: is
	// only ever reached by a genuinely new MaterialType nobody wired into
	// this backend yet, exactly the class of gap this project's own history
	// has had to find and fix reactively more than once (see e.g.
	// "Wavefront gaps" in prior commits). The obvious fix - drop the
	// catch-all and let the compiler's missing-enum-case-in-switch warning
	// catch it at compile time - does not actually work here: nvcc's
	// device-code frontend (confirmed empirically against this project's
	// CUDA 13.2 toolchain, including every --diag-warn flag it exposes)
	// does not implement that diagnostic the way MSVC's C4062 or clang's
	// -Wswitch do, so a missing case compiles silently clean either way.
	// Trap loudly instead of absorbing the ray, so the gap surfaces the
	// moment a real render exercises it. Not gated behind NDEBUG - nvcc's
	// own invocation for this file never defines it either way (see
	// build_optix.targets), so the guard is simply always on.
	default: {
		printf("[EVAL-MATERIALS] unhandled MaterialType %d\n", (int)mat.type);
		__trap();
	}
	}

	if (is_medium_boundary) {
		// True pass-through (MaterialType::Interface) - nothing actually
		// scattered here, so this bypasses wf_finish_material_scatter()
		// entirely (no NEE, no RR, no depth/specular_bounce/brdf_pdf
		// recompute) and pushes the next RayWorkItem directly, preserving
		// h's own depth/specular_bounce/any_nonspecular/etaScale/brdf_pdf
		// exactly as they arrived - the NEXT real hit's MIS state should
		// reflect whatever the last REAL vertex was, not this crossing.
		// Matches CPU camera.h's/the recursive backend's own is_medium_
		// boundary branch exactly. No separate crossing-count safety cap is
		// needed here (unlike those two, which loop internally per-ray):
		// the outer host-side bounce loop (wavefront_path_tracer.cpp) is
		// already a fixed `for (depth = 0; depth < max_depth + 1 + kMaxMediumBoundaryCrossings; ++depth)`
		// iteration count, independent of any individual ray's own depth
		// field, so a degenerate scene can't hang this backend either way.
		RayWorkItem next;
		next.origin    = hit_point + 0.001f * scattered_dir;
		next.direction = normalize(scattered_dir);
		next.seed      = seed;
		wf_carry_ray_state(next, h);  // pixelIndex/depth/specular_bounce/any_nonspecular/etaScale/filterWeight/brdf_pdf, unchanged
		next.tMin      = 0.001f;
		next.tMax      = 1e30f;
		for (int i = 0; i < kWFNWavelengths; ++i) {
			next.throughput[i]      = throughput[i] * attenuation[i];
			next.radiance[i]        = radiance[i];
			next.wavelengths[i]     = swl.lambda[i];
			next.wavelength_pdfs[i] = swl.pdf[i];
		}
		nextRayQueue.push(next);
		return;
	}

	if (!scattered) {
		// Path absorbed — accumulate what we have.
		addToFramebuffer(h.pixelIndex, radiance * h.filterWeight);
		return;
	}

	// eventEta was set above only on a genuine transmission (DielectricMedium's
	// entry/exit surfaces) - see this function's own eventEta local comment.
	wf_finish_material_scatter(effectiveMatType, matEta, h.materialIdx, (bool)h.any_nonspecular, glossyAlphaForNEE, glossyAlphaVForNEE, h.etaScale * eventEta * eventEta, h.filterWeight, maxComponentValue, normal, hit_point, h.objDpdu, seed,
		throughput, radiance, swl, attenuation, scattered_dir, is_specular, brdf_pdf_override,
		phaseWo, phaseG,
		h.pixelIndex, h.depth,
		spheres, quads, triangles, bilinearPatches, disks, cylinders, materials,
		lightIndices, lightKinds, aliasTable, numLights,
		punctualLights, numPunctualLights,
		skyColor, shadow_eps, skyDist, portalLight,
		shadowQueue, nextRayQueue, framebuffer,
		textures, texturePixels, h.uv_u, h.uv_v, h.time, restirReservoirs, restirCtx,
		giOriginContext, giCandidateOut,
		lightBvh,
		// probeGridMeta/probeGrid (SH-L1 diffuse cache) intentionally left on
		// their nullptr/default just above - Lambertian never reaches this
		// kernel (see this file's own guidingHistograms parameter comment).
		// guidingGridMeta/guidingHistograms/guidingProbes DO apply here
		// (Conductor/RoughMetal's own NEE/MIS pdf, via evalGlossyF) - same
		// pointers this kernel's own Conductor/RoughMetal switch-arms above
		// already consult for their scatter-direction pdf, now also reaching
		// wf_finish_material_scatter() so its NEE MIS weight uses the same
		// mixture pdf rather than the plain BSDF pdf alone.
		GpuProbeGridMeta{}, nullptr,
		guidingGridMeta, guidingHistograms, guidingProbes,
		nrcWeights, nrcTrainingSteps, nrcAabbMin, nrcAabbExtent,
		// mediumEntryPoint is h.hitPoint (the medium's own stable boundary
		// intersection, written into worldPosBuffer above) - NOT hit_point,
		// which every one of the 5 medium switch-cases above has already
		// reassigned to the stochastic interior scatter point by this point.
		// See mediumEntryPoint's own parameter comment (wavefront_device_
		// helpers.h) for why only the entry point is stable enough to
		// reproject frame-to-frame.
		h.hitPoint, restirVolumeReservoirs, restirVolumeCtx, volumeMatIdxOut,
		mediumMeanFreePath, volumePhaseWoGOut, volumeEntryPointOut,
		glossyCondAlphaForNEE, glossyCondAlphaVForNEE,
		&measuredBundle);
}

