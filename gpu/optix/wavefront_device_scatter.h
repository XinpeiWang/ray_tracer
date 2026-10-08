#pragma once
// wavefront_device_scatter.h -- part 3 of 3 of wavefront_device_helpers.h (included by it, in order; not meant to be included on its own).

// ============================================================================
// Material-scatter finish + RGB->spectral uplift (moved from
// wavefront_kernels.cu when that file was split into per-kernel-family .cu
// files - see build_optix.targets' own comment on why a NEW .cu file needs
// build-system wiring while a header does not. wf_finish_material_scatter is
// shared by evaluate_materials/evaluate_materials_simple/
// evaluate_materials_dielectric/resolve_bssrdf_exit, now living in 4
// different .cu files - keeping it here (as __device__ __forceinline__, the
// same one-definition-rule-safe pattern already used by every other
// function in this header) means each of those files gets its own inlined
// copy with zero cross-TU device-linking required for this specific
// function, exactly like everything else here already works.
// ============================================================================

// Forward declaration - real definition (and its own full doc comment) is
// further down this same file. wf_finish_material_scatter's own probe-cache
// lookup block needs to call this before that point, so it needs a
// declaration in scope here; the definition below is unchanged either way.
// SampledSpectrum<N>/SampledWavelengths<N> are already fully defined by this
// point (sampled_spectrum.h, included at this file's own top) - only the
// FUNCTION itself needs forward-declaring.
// No default arg here (that stays on the real definition below) - a default
// argument can only be specified once per scope in one translation unit.
__device__ __forceinline__ SampledSpectrum<kWFNWavelengths> wf_lift_rgb_to_spectrum(
	float3 rgb, const SampledWavelengths<kWFNWavelengths>& swl, bool isIlluminant);

__device__ __forceinline__ void wf_finish_material_scatter(
	MaterialType matType, float nfEta, int matIdx,
	// True once any PRIOR bounce along this path was non-specular (see
	// RayWorkItem::any_nonspecular's own comment, wavefront_types.h) - folded
	// (OR'd with !is_specular) into the pushed next-bounce RayWorkItem's own
	// any_nonspecular at this function's tail, regardless of matType. Does
	// NOT drive glossy_alpha's regularization directly (see glossyAlpha's
	// own comment below) - kept only for this tail-propagation role.
	bool do_regularize,
	// Pre-computed via wf_glossy_alpha() by whichever glossy switch-arm this
	// call follows (Conductor/RoughDielectric/RoughMetal/CoatedDiffuse/
	// CoatedConductor - see that function's own comment), already folding in
	// do_regularize's widening. Passed in rather than re-derived from
	// materials[matIdx].fuzz here, so the NEE alpha (glossy_alpha below) is
	// always bit-identical to the alpha the caller's own BSDF-sampling step
	// just used - re-deriving independently would risk the two silently
	// drifting apart if either formula is ever edited without the other.
	// Meaningless/unread for every non-glossy matType (glossy_isType below
	// gates its only use) - callers that never reach a glossy case (e.g.
	// evaluate_materials_simple's Lambertian/Metal, resolve_bssrdf_exit's
	// NormalizedFresnel) just pass 0.0f.
	float glossyAlpha,
	// Second (v/bitangent-axis) GGX alpha, mirroring glossyAlpha above -
	// see wf_glossy_alpha_v()'s own comment for which 4 material kinds set
	// this to a real per-call value (Conductor/RoughDielectric/
	// CoatedDiffuse/CoatedConductor) vs. leave it at the caller's own
	// declared-but-never-set 0.0f (RoughMetal, and every non-glossy
	// matType) - <=0 here means "isotropic, use glossyAlpha for both axes"
	// (see MaterialData::roughnessV's own sentinel convention), handled
	// below rather than requiring every caller to redundantly pass
	// glossyAlpha twice.
	float glossyAlphaV,
	// pbrt-v4 etaScale for the Russian Roulette test below - already fully
	// updated for THIS event (incoming RayWorkItem/HitWorkItem::etaScale
	// times this bounce's own eta^2, or unchanged if this event wasn't a
	// transmission) by whichever caller computed it - see RayWorkItem::
	// etaScale's own comment (wavefront_types.h) and each call site's own
	// eventEta local. Written unchanged into the pushed next RayWorkItem's
	// own etaScale at this function's tail.
	float etaScale,
	// Pixel reconstruction filter weight for this path's own sample - see
	// RayWorkItem::filterWeight's own comment. Multiplied into every
	// radiance contribution this function adds to the framebuffer (NEE
	// shadow rays, the RR-kill/hit-light flush) and written unchanged into
	// the pushed next RayWorkItem's own filterWeight.
	float filterWeight,
	// "float maxcomponentvalue" firefly clamp - see addToFramebuffer's own
	// comment below and GpuCameraParams::maxComponentValue's own comment
	// (optix_types.h). 1e9f (unbounded) when not requested.
	float maxComponentValue,
	const float3& normal, const float3& hit_point,
	// World-space surface tangent (dp/du) at this shading point - real
	// per-shape value from the caller (see HitWorkItem::objDpdu's own
	// comment), used ONLY by evalGlossyF's UV-aligned frame construction
	// below for the 4 anisotropy-capable material kinds (RoughMetal and
	// every non-glossy matType ignore it, keeping the arbitrary frame -
	// matches optix_device_helpers.h's identical RoughMetal exclusion).
	const float3& dpdu,
	unsigned int& seed,
	const SampledSpectrum<kWFNWavelengths>& throughput,
	const SampledSpectrum<kWFNWavelengths>& radiance,
	const SampledWavelengths<kWFNWavelengths>& swl,
	const SampledSpectrum<kWFNWavelengths>& attenuation,
	const float3& scattered_dir, bool is_specular, float brdf_pdf_override,
	// Only meaningful when isPhase (below) is true and is_specular == false -
	// a genuine medium-interior phase-function scatter event, reached by
	// MaterialType::Medium/CloudMedium/RgbGridMedium/GridMedium (always) or
	// DielectricMedium's own interior sub-case only (its other two
	// sub-cases, the entry/exit dielectric-surface refractions, are
	// genuinely specular and never clear the `if (!is_specular)` gate
	// below). phaseWo is the direction back toward where the ray came from
	// (-incoming ray dir, matching CPU hg_phase_pdf's own `wo`); phaseG is
	// the medium's HG asymmetry (mat.fuzz, or grid.phase_g for the two grid
	// medium types). ALSO reused (same "-incoming ray dir" meaning) as
	// `wi_world` by the Conductor/RoughDielectric/CoatedDiffuse/CoatedConductor
	// glossy branches above to rebuild their local shading frame - harmless/
	// unused for every other material type.
	const float3& phaseWo, float phaseG,
	int pixelIndex, int depth,
	const SphereData* spheres, const QuadData* quads,
	const TriangleData* triangles, const BilinearPatchData* bilinearPatches,
	const DiskData* disks, const CylinderData* cylinders,
	const MaterialData* materials,
	const int* lightIndices, const GpuLightKind* lightKinds,
	const GpuAliasEntry* aliasTable, unsigned int numLights,
	const PunctualLightGPU* punctualLights, unsigned int numPunctualLights,
	float3 skyColor, float shadow_eps, const GpuSkyDistribution& skyDist,
	const GpuPortalLight& portalLight,
	WorkQueue<ShadowRayWorkItem>& shadowQueue,
	WorkQueue<RayWorkItem>& nextRayQueue,
	float3* framebuffer,
	// Needed for a textured (pbrt AreaLightSource "filename") NEE target -
	// see this function's own Triangle-light NEE branch below - AND for a
	// texture-bound CoatedDiffuse's NEE/MIS f() (see uv_u/uv_v below).
	// Every caller already has these in scope (evaluate_materials/
	// evaluate_materials_simple as kernel params, evaluate_materials_dielectric/
	// resolve_bssrdf_exit newly threaded through for this same reason - see
	// wavefront_launch.cu/wavefront_path_tracer.cpp).
	const TextureData* textures, const unsigned char* texturePixels,
	// The CURRENT hit's own UV (HitWorkItem::uv_u/uv_v) - lets evalGlossyF's
	// CoatedDiffuse branch sample the material's real per-point reflectance
	// texture for NEE/MIS, matching the scatter path's own lookup instead of
	// falling back to fm.albedo. Every caller already holds `h.uv_u`/`h.uv_v`
	// in scope (same HitWorkItem the caller's own scatter-path texture
	// lookup already reads, e.g. evaluate_materials()'s Lambertian/
	// CoatedDiffuse cases) - callers whose own material switch can never
	// reach CoatedDiffuse (evaluate_materials_simple: Lambertian/Metal only;
	// evaluate_materials_dielectric/resolve_bssrdf_exit: dielectric-only
	// matTypes) just pass their own h.uv_u/h.uv_v too since it's free and
	// harmless - evalGlossyF's CoatedDiffuse branch is simply never reached
	// from those call sites.
	float uv_u, float uv_v,
	// Object (per-primitive sphere) motion blur shutter time - see
	// RayWorkItem::time's own comment (wavefront_types.h). Carried unchanged
	// from the incoming hit into every ShadowRayWorkItem/RayWorkItem this
	// function pushes below, exactly like filterWeight/etaScale above.
	float time,
	// ReSTIR DI (Live Preview only) current-frame reservoir buffer, indexed
	// by pixelIndex - non-null only when the caller's kernel launch was
	// built with real-time preview's reservoir buffer allocated (see
	// WavefrontPathTracer::setRestirEnabled()). Defaulted to nullptr so the
	// existing batch/offline call sites (evaluate_materials/_simple/
	// _dielectric) need no edit to keep their current single-draw NEE
	// behavior exactly as before - only the real-time-preview launch path
	// ever passes a real pointer. depth>0 (indirect-bounce NEE) always takes
	// the single-draw path too, even with a non-null buffer - ReSTIR DI is a
	// primary-hit-only technique (see this function's own RIS block below).
	// wavefront_kernels_bssrdf.cu's resolve_bssrdf_exit is ALSO left on this
	// nullptr default, but NOT because a BSSRDF exit can't have depth==0 -
	// it demonstrably can (a camera ray hitting a Subsurface surface keeps
	// depth==0 unchanged through the probe-request/probe-exit round trip,
	// see BssrdfProbeWorkItem/BssrdfExitWorkItem's own depth fields). It's
	// safe purely because useRestir below gates on `restirReservoirs !=
	// nullptr`, not depth alone - resolve_bssrdf_exit simply never threads a
	// real reservoir buffer through. If ReSTIR is ever extended to BSSRDF
	// exits, do NOT assume depth is guaranteed nonzero there.
	GpuReservoir* restirReservoirs = nullptr,
	// ReSTIR temporal reuse's previous-frame history + reprojection basis -
	// see GpuRestirTemporalContext's own comment (optix_types.h). Default-
	// constructed (historyValid=false) is a safe no-op for every existing
	// call site - wf_restir_temporal_combine() returns immediately without
	// touching `res` when historyValid is false, exactly like restirReservoirs
	// being null skips the whole ReSTIR block above it.
	const GpuRestirTemporalContext& restirCtx = GpuRestirTemporalContext{},
	// ReSTIR GI (Live Preview only, gpu/optix/wavefront_restir_gi_math.h) -
	// ONE per-pixel buffer serving two depth-dependent roles, both gated on
	// this pointer being non-null (nullptr is a complete no-op, same "every
	// existing call site needs no edit" shape as restirReservoirs above):
	//   depth==0 (WRITE): the primary hit x0's own shading context (position,
	//   normal, the BSDF-sampled x0->x1 direction/pdf, throughput, this
	//   path's hero wavelengths) is stashed here at pixelIndex, for the GI
	//   finalize pass (wavefront_kernels_restir.cu) to consume once depth 1
	//   resolves - see GpuGiOriginContext's own comment (wavefront_types.h)
	//   for why this stash exists at all (DI never needed one).
	//   depth==1 (READ): giOriginContext[pixelIndex].valid() gates whether
	//   this hit's own NEE/emission should populate giCandidateOut below
	//   instead of leaving it untouched - false whenever depth 0 wasn't
	//   GI-eligible (non-Lambertian, specular, or GI disabled that frame),
	//   leaving this pixel's depth-1 contribution completely unaffected.
	GpuGiOriginContext* giOriginContext = nullptr,
	// ReSTIR GI's per-pixel candidate buffer - written ONLY for a depth==1
	// hit whose giOriginContext[pixelIndex] is valid (see above). Unlike
	// giOriginContext (whose x1-side fields aren't known until depth 1's own
	// intersection), THIS hit's x1Point/x1Normal/x0Point/pdfAtX0 are all
	// known synchronously right here - written once, immediately, when
	// giCandidateEligible first becomes true (below); only `.radiance`
	// (caching Lo(x1 -> x0): x1's own NEE, via ShadowRayWorkItem::
	// isGiCandidate, resolved later and asynchronously by accumulate_shadow)
	// fills in afterward via atomicAdd, since occlusion isn't known yet at
	// this point. `.radiance` is RGB, not spectral - see GpuGiSample::
	// radiance's own comment (wavefront_types.h) for why a cached spectral
	// value would be meaningless once reused by a different pixel/frame's
	// own independently-sampled hero wavelengths. MVP scope: when x1 is
	// reached by hitting an emitter directly (the `radiance` flush below,
	// unrelated variable name collision with GpuGiSample::radiance - this is
	// the PATH's accumulated emission-hit radiance, not the GI sample's own
	// field), `.radiance` is deliberately left untouched for that
	// contribution (real framebuffer still gets it as before) rather than
	// caching it - see this project's own ReSTIR GI plan for why (a glowing
	// x1 is treated as "no GI candidate this frame" rather than
	// approximated, avoiding a whole extra class of edge cases for a rare
	// path).
	GpuGiSample* giCandidateOut = nullptr,
	// See wf_light_bvh_sample_index()'s own comment
	// (wavefront_restir_helpers.h). lightBvh.nodeCount<=0 (the default) means
	// "no light BVH built" - forwarded to wf_generate_restir_candidate()
	// below unchanged, which itself falls straight through to the alias
	// table for that case, so every existing call site needs no edit.
	WfLightBvhContext lightBvh = {},
	// World-space irradiance probe cache (Live Preview only, gpu/optix/
	// probe_grid_types.h) - see this project's own plan. probeGrid==nullptr
	// (the default, every non-Live-Preview call site) is a complete no-op,
	// same "null pointer disables the whole feature" shape as
	// restirReservoirs/giOriginContext above. When non-null, a depth>=
	// kProbeCacheMinDepth Lambertian scatter queries the cache instead of
	// tracing another bounce - see this function's own lookup block, right
	// before the "Bounce: push next ray" section below.
	const GpuProbeGridMeta& probeGridMeta = GpuProbeGridMeta{},
	const GpuProbe* probeGrid = nullptr,
	// Real-time path guiding (Live Preview only, gpu/optix/wavefront_guiding.h)
	// - lets evalGlossyF's Conductor/RoughMetal branches (below) report the
	// SAME mixture pdf evaluate_materials()'s own scatter path already uses
	// for the BSDF-sampled continuation direction, evaluated instead at the
	// NEE-queried light direction - see evalGlossyF's own header comment for
	// why MIS needs the pdf at THIS direction, not the continuation one, and
	// this project's own plan for the mixture-pdf rationale. nullptr (the
	// default, every non-Live-Preview call site, or Live Preview with the
	// feature toggled off) keeps evalGlossyF on the plain BSDF pdf it always
	// used before path guiding existed - a complete no-op, same shape as
	// probeGrid just above.
	GpuProbeGridMeta guidingGridMeta = {},
	const GpuGuidingHistogram* guidingHistograms = nullptr,
	const GpuProbe* guidingProbes = nullptr,
	// Neural Radiance Cache (Live Preview only, gpu/optix/wavefront_nrc_*.h) -
	// see this project's own plan. nrcWeights==nullptr (the default, every
	// non-Live-Preview call site, or Live Preview with the feature toggled
	// off) is a complete no-op, same "null pointer disables the whole
	// feature" shape as probeGrid/guidingHistograms above. When non-null, a
	// depth>=kNrcMinDepth Lambertian/RoughMetal scatter may stochastically
	// terminate and substitute the network's own predicted radiance instead
	// of tracing another bounce - see this function's own lookup block,
	// right before the probe-cache lookup below (tried FIRST: NRC supports
	// both material types the probe cache is gated to, plus RoughMetal,
	// which the probe cache's SH-L1 cache cannot represent at all).
	// nrcTrainingSteps gates a cold/untrained network out exactly like the
	// probe cache's own numRaysEverTraced==0 check (WavefrontPathTracer's
	// own persistent training-step counter, incremented once per
	// launchNrcTrainingUpdate() call). nrcAabbMin/nrcAabbExtent are the
	// scene's own world-space bounds, reused directly from probeGridMeta's
	// gridMin/dims*cellSize (see launchNrcTrainingUpdate()'s own comment for
	// why NRC doesn't compute its own separate scene AABB).
	const float* nrcWeights = nullptr,
	int nrcTrainingSteps = 0,
	float3 nrcAabbMin = make_float3(0.0f, 0.0f, 0.0f),
	float3 nrcAabbExtent = make_float3(0.0f, 0.0f, 0.0f),
	// ReSTIR for volumetric/participating-media phase-scatter vertices (Live
	// Preview only) - see this project's own plan. Mirrors restirReservoirs/
	// restirCtx above but keyed by mediumEntryPoint (the medium's own
	// boundary/entry point, HitWorkItem::hitPoint) rather than hit_point -
	// by the time isPhase is true, hit_point has already been reassigned to
	// the stochastic INTERIOR scatter point re-drawn fresh every frame, which
	// is NOT stable enough to reproject frame-to-frame; the entry point is a
	// deterministic ray/geometry intersection exactly as stable as a surface
	// hit. nullptr (the default, every non-Live-Preview call site, or Live
	// Preview with the feature toggled off) is a complete no-op, same "null
	// pointer disables the whole feature" shape as restirReservoirs above.
	// Only ever consulted when isPhase is true (see this function's own
	// isPhase-gated RIS block below) - harmless/unused otherwise.
	const float3& mediumEntryPoint = make_float3(0.0f, 0.0f, 0.0f),
	GpuVolumeReservoir* restirVolumeReservoirs = nullptr,
	const GpuVolumeRestirTemporalContext& restirVolumeCtx = GpuVolumeRestirTemporalContext{},
	// This frame's own medium identity (materials[] index), written
	// unconditionally whenever isPhase is true so the separate, later
	// restir_volume_spatial_reuse kernel launch (wavefront_kernels_restir.cu)
	// - which has no live per-thread material context of its own by that
	// point - can still recover "which medium is this pixel's reservoir even
	// for" at spatial-reuse time. nullptr is a complete no-op, same shape as
	// restirVolumeReservoirs above.
	int* volumeMatIdxOut = nullptr,
	// This medium's own mean free path (wf_restir_volume_mean_free_path,
	// 1/sigma_t for Medium/DielectricMedium's homogeneous case, 1/sigma_maj
	// for CloudMedium/RgbGridMedium/GridMedium's heterogeneous case) - the
	// caller (wavefront_kernels_materials.cu) already computes sigma_t/
	// sigma_maj locally for its own free-flight distance sampling in each of
	// the 5 medium switch-cases, so this is passed in rather than re-derived
	// here from materials[matIdx] alone, which lacks the heterogeneous
	// types' own CloudMediumData/GpuRgbGridMedium/GpuGridMedium sigma_maj
	// (an index away, not a flat field) - see this project's own plan for
	// why the spatial-reuse kernel needs this value at all (its own neighbor
	// distance-validity gate, wf_restir_volume_spatial_valid). Meaningless/
	// unread when isPhase is false.
	float mediumMeanFreePath = 0.0f,
	// This frame's own phaseWo (xyz)/phaseG (w), packed into one float4 the
	// same "position/data + extra scalar" packing convention worldPosBuffer
	// uses - the volumetric analog of d_restirNormal_'s role for DI's own
	// spatial reuse (wf_restir_target_proxy_phase needs phaseWo/phaseG to
	// re-evaluate a reused candidate's target function fresh at THIS pixel,
	// exactly like wf_restir_target_proxy needs `normal`), since the separate,
	// later restir_volume_spatial_reuse kernel launch has no live per-thread
	// phaseWo/phaseG of its own to read. Written unconditionally whenever
	// isPhase is true, same shape as volumeMatIdxOut above. nullptr is a
	// complete no-op.
	float4* volumePhaseWoGOut = nullptr,
	// This frame's own mediumEntryPoint (xyz) + mediumMeanFreePath (w),
	// packed into one float4 - same packing convention as volumePhaseWoGOut
	// above, folding what used to be two separate buffers (a float4 entry
	// point and a plain float mean-free-path) into one, since both are
	// always written together and read together by restir_volume_spatial_
	// reuse's own neighbor-validity gate. Written alongside volumeMatIdxOut/
	// volumePhaseWoGOut above so the separate, later restir_volume_spatial_
	// reuse kernel launch reads an entry point that is ALWAYS from the same
	// frame/call as those other two sticky fields - critically, NOT the
	// shared worldPosBuffer, which is overwritten every frame for every
	// depth==0 hit (including a frame where THIS pixel is no longer a phase
	// vertex at all). Reading worldPosBuffer there instead would pair this
	// frame's fresh (and possibly totally unrelated) hit point with the
	// other sticky fields' stale, held-over values whenever a pixel's
	// depth==0 hit type changes between phase-scatter frames - see this
	// project's own plan for the "isPhase never clears these buffers"
	// design and why staying self-consistent across all sticky fields is
	// what actually fixes that. nullptr is a complete no-op, same shape as
	// volumeMatIdxOut/volumePhaseWoGOut above.
	float4* volumeEntryPointOut = nullptr,
	// MaterialType::CoatedConductor only: the BASE conductor's (already regularized) GGX alphas - glossyAlpha/
	// glossyAlphaV above are the coat's. Negative (the default, every other call site) means "derive them from
	// materials[matIdx]", unregularized.
	float glossyCondAlpha = -1.0f,
	float glossyCondAlphaV = -1.0f,
	// MaterialType::Measured only: the measured-table arrays evalGlossyF below evaluates f() and pdf() from. nullptr for every other
	// material (and any call site that never reaches a Measured hit).
	const WfMeasuredTables* measured = nullptr)
{
	using SS = SampledSpectrum<kWFNWavelengths>;

	// See phaseWo/phaseG's own parameter comment above - matType alone
	// unambiguously identifies a medium-interior phase-scatter event here:
	// DielectricMedium's other two (specular) sub-cases never reach this NEE
	// block at all (is_specular stays true for those), and Medium/
	// CloudMedium/RgbGridMedium/GridMedium's own "no interaction, straight
	// pass-through" sub-case is likewise still is_specular=true - only a
	// genuine scatter event for any of these 5 material types clears the
	// caller's `if (!is_specular)` gate and reaches here.
	const bool isPhase = (matType == MaterialType::DielectricMedium ||
						   matType == MaterialType::Medium ||
						   matType == MaterialType::CloudMedium ||
						   matType == MaterialType::RgbGridMedium ||
						   matType == MaterialType::GridMedium);

	auto addToFramebuffer = [&](int pixIdx, const SS& L) {
		auto xyz = SampledSpectrumToXYZ(L, swl, d_cie_x, d_cie_y, d_cie_z,
										kDevCIEMin, kDevCIENSamples);
		float r, g, b;
		wf_xyz_to_linear_rgb(xyz.x, xyz.y, xyz.z, r, g, b);
		// "float maxcomponentvalue" firefly clamp - see
		// GpuCameraParams::maxComponentValue's own comment (optix_types.h)
		// for why this is a per-CONTRIBUTION clamp here, not the true
		// per-sample-total clamp CPU/the recursive backend apply: this
		// atomicAdd is one of several independent partial contributions to
		// the same sample's eventual total, with no single point on this
		// backend where the whole sample's radiance is ever known as one
		// value to clamp.
		const float m = fmaxf(r, fmaxf(g, b));
		if (maxComponentValue > 0.0f && m > maxComponentValue) {
			const float s = maxComponentValue / m;
			r *= s; g *= s; b *= s;
		}
		atomicAdd(&framebuffer[pixIdx].x, r);
		atomicAdd(&framebuffer[pixIdx].y, g);
		atomicAdd(&framebuffer[pixIdx].z, b);
	};

	// ReSTIR GI (see this function's own giOriginContext/giCandidateOut
	// parameter comments) - true only for a depth==1 hit whose ORIGINATING
	// primary hit (x0) was GI-eligible this frame. Computed once, used by
	// every NEE block below to tag its own ShadowRayWorkItem's isGiCandidate
	// so accumulate_shadow (wavefront_kernels_accumulate.cu) routes that
	// ray's Ld into giCandidateOut's own `.radiance` instead of the real
	// framebuffer, once occlusion resolves (this function itself never
	// learns whether a shadow ray it pushes is occluded - that's
	// accumulate_shadow's own job, a separate kernel launch that runs after
	// this one). depth==0's own NEE (DI's ReSTIR path or the classic
	// single-draw path) is never affected, since this is false for every
	// depth other than 1.
	//
	// !is_specular is required here (not just implied by "no shadow ray gets
	// pushed for a specular hit anyway"): without it, a specular x1 still
	// registers a real GpuGiSample with radiance permanently (0,0,0) (the
	// `if (!is_specular)` NEE block below never runs for it, so `.radiance`
	// is never filled in) - restir_reservoir_add unconditionally does
	// `r.M += 1` for this zero-weight candidate regardless of its weight, so
	// once temporal reuse folds in real history the combined M is inflated
	// by this phantom candidate while weightSum is untouched, systematically
	// biasing W - and therefore this pixel's whole GI contribution - low
	// every frame a specular x1 occurs. Gating eligibility on !is_specular
	// up front means a specular x1 is treated exactly like the already-
	// documented "no GI candidate this frame" cases (a glowing x1, GI
	// disabled) instead of silently diluting the reservoir.
	const bool giCandidateEligible = (depth == 1 && !is_specular && giOriginContext != nullptr &&
									   giCandidateOut != nullptr &&
									   giOriginContext[pixelIndex].valid());
	// The candidate's geometry-side fields ARE all known synchronously right
	// here (unlike its `.radiance`, filled in later/asynchronously - see
	// giCandidateOut's own parameter comment) - written once, immediately,
	// so every NEE block below (and accumulate_shadow afterward) has a
	// fully-formed GpuGiSample to accumulate `.radiance` into regardless of
	// which block(s) actually fire for this particular hit.
	if (giCandidateEligible) {
		GpuGiSample& cand = giCandidateOut[pixelIndex];
		cand.x1Point = hit_point;
		cand.x1Normal = normal;
		cand.x0Point = giOriginContext[pixelIndex].x0Point;
		cand.pdfAtX0 = giOriginContext[pixelIndex].pdfAtX0;
		cand.radiance = make_float3(0.0f, 0.0f, 0.0f);
	}

	// Real per-direction, per-channel f() for the 4 glossy (non-
	// EffectivelySmooth) BxDF-templated materials, via the same CPU_GPU
	// structs already verified on CPU (#222) and the recursive backend
	// (#229) - reconstructs the local shading frame from `normal` alone
	// (deterministic, matches every per-material frame-construction formula
	// in evaluate_materials()'s own switch exactly) and `phaseWo` as wi_world
	// (see that parameter's own comment). Returns false - meaning "no
	// contribution, not an error" - for a matType this function doesn't
	// handle, or when the queried direction falls in a hemisphere the
	// material can't reach from wi (grazing/back-facing wi, or wo below the
	// coat/dielectric's own reflection hemisphere - see RoughDielectric's
	// wo_z<=0 check and this function's own header comment on why
	// transmission-side NEE is out of scope here).
	//
	// Also returns outPdf: the BSDF pdf AT THIS QUERIED DIRECTION (not the
	// separately-tracked brdf_pdf_override, which is the pdf at the BSDF-
	// sampled CONTINUATION direction computed once in evaluate_materials() -
	// a different quantity). MIS needs the pdf at the direction actually
	// being weighted; mirrors optix_device_helpers.h's NEE blocks, which
	// compute a fresh {c,rm,rd,cd,cc}_bxdf.pdf(wi,...,ll/sk...) per light/sky
	// sample rather than reusing the continuation-direction pdf. For
	// Conductor/RoughMetal this is the BxDF's own real pdf() (a thin wrapper
	// over ggx_vndf_reflection_pdf) - blended with real-time path guiding's
	// own mixture pdf (glossy_pGuide/glossy_guideProbeIdx below) whenever
	// guiding is active for this hit, so this stays the pdf of whichever
	// technique is ACTUALLY being used to sample the continuation direction,
	// exactly matching brdf_pdf_override's own mixture in evaluate_materials()'s
	// Conductor/RoughMetal cases - MIS is only correct when both directions'
	// pdfs come from the same distribution the renderer is really sampling
	// from; RoughDielectric uses its own real pdf(); CoatedDiffuse/
	// CoatedConductor have no closed-form pdf for their unbounded-depth
	// random walk, so - matching optix_device_helpers.h's documented choice
	// exactly - they reuse the coat's top-surface GGX VNDF pdf as a cheap
	// shape-matched proxy (any valid pdf keeps MIS unbiased; this only
	// affects variance, not correctness).
	// Shared per-shading-point setup for evalGlossyF below, computed ONCE
	// instead of on each of the (up to) 3 calls it gets per shading point
	// (area-light NEE, sky NEE, punctual-light NEE) via wf_setup_glossy_context()
	// - normal/phaseWo/matType/matIdx are all invariant across those calls
	// within a single wf_finish_material_scatter() invocation, so
	// re-deriving the local frame, wi, and material lookup on every call
	// redid identical work up to 3x for no reason. glossy_valid folds both
	// of evalGlossyF's old early-outs (wrong matType, wi_z<=0) into one
	// check the lambda itself no longer needs to redo. Unpacked into the
	// same local names evalGlossyF below already captures by reference, so
	// that lambda's own body needs no change from this extraction.
	const GlossyCtx glossyCtx = wf_setup_glossy_context(matType, normal, dpdu, phaseWo,
		glossyAlpha, glossyAlphaV, guidingGridMeta, guidingHistograms, guidingProbes, hit_point);
	// Recomputed here (not carried on GlossyCtx) since it's also read much
	// further down, outside evalGlossyF's own scope, in the 3x-duplicated
	// matType switch this same function still has inline - a cheap, pure
	// boolean re-check, not worth widening the struct's contract for.
	const bool glossy_isType = (matType == MaterialType::Conductor || matType == MaterialType::RoughDielectric ||
		matType == MaterialType::CoatedDiffuse || matType == MaterialType::CoatedConductor ||
		matType == MaterialType::RoughMetal || matType == MaterialType::Measured);
	const bool glossy_valid = glossyCtx.valid;
	// Materials whose BSDF takes light from both sides of the surface: the NEE sites below use |cos| and aim the shadow ray's origin to the light's side.
	const bool twoSidedNee = (matType == MaterialType::RoughDielectric || matType == MaterialType::DiffuseTransmission);
	const float3 glossy_tan = glossyCtx.tan;
	const float3 glossy_bit = glossyCtx.bit;
	const float glossy_wi_x = glossyCtx.wi_x;
	const float glossy_wi_y = glossyCtx.wi_y;
	const float glossy_wi_z = glossyCtx.wi_z;
	const float glossy_alpha = glossyCtx.alpha;
	const float glossy_alpha_v = glossyCtx.alpha_v;
	const float glossy_pGuide = glossyCtx.pGuide;
	const int glossy_guideProbeIdx = glossyCtx.guideProbeIdx;

	auto evalGlossyF = [&](const float3& queryDir, float3& outF, float& outPdf) -> bool {
		if (!glossy_valid) return false;
		float wo_x = dot(queryDir, glossy_tan), wo_y = dot(queryDir, glossy_bit), wo_z = dot(queryDir, normal);
		// RoughDielectric reaches both hemispheres (wo_z<0 = transmission,
		// "seen through the glass" - RoughDielectricBxDF::f()/pdf() already
		// handle either sign, matching optix_device_helpers.h's identical
		// `llz != 0.0f` gate); every other glossy type here is reflection-
		// only, unchanged.
		if (matType == MaterialType::RoughDielectric) {
			if (wo_z == 0.0f) return false;
		} else if (wo_z <= 0.0f) {
			return false;
		}
		const MaterialData& fm = materials[matIdx];
		float fr = 0.0f, fg = 0.0f, fb = 0.0f;
		if (matType == MaterialType::Conductor) {
			ConductorBxDF<float> bx{ fm.eta_c.x, fm.eta_c.y, fm.eta_c.z, fm.k_c.x, fm.k_c.y, fm.k_c.z, glossy_alpha, glossy_alpha_v };
			bx.f(glossy_wi_x, glossy_wi_y, glossy_wi_z, wo_x, wo_y, wo_z, fr, fg, fb);
			outPdf = bx.pdf(glossy_wi_x, glossy_wi_y, glossy_wi_z, wo_x, wo_y, wo_z);
			// Blend to the SAME mixture pdf the scatter path uses for its own
			// BSDF-sampled continuation direction (evaluate_materials()'s own
			// Conductor case), evaluated here at queryDir instead - MIS must
			// weight against the pdf of the technique actually being used to
			// sample this material's continuation, which is this mixture
			// whenever glossy_pGuide>0, not the plain BSDF pdf alone.
			if (glossy_pGuide > 0.0f) {
				const float pdf_guide = wf_guided_pdf(guidingHistograms[glossy_guideProbeIdx], queryDir);
				outPdf = glossy_pGuide * pdf_guide + (1.0f - glossy_pGuide) * outPdf;
			}
		} else if (matType == MaterialType::RoughMetal) {
			RoughMetalBxDF<float> bx{ fm.albedo.x, fm.albedo.y, fm.albedo.z, glossy_alpha, glossy_alpha };
			bx.f(glossy_wi_x, glossy_wi_y, glossy_wi_z, wo_x, wo_y, wo_z, fr, fg, fb);
			outPdf = bx.pdf(glossy_wi_x, glossy_wi_y, glossy_wi_z, wo_x, wo_y, wo_z);
			// See MaterialType::Conductor's identical-shape blend just above.
			if (glossy_pGuide > 0.0f) {
				const float pdf_guide = wf_guided_pdf(guidingHistograms[glossy_guideProbeIdx], queryDir);
				outPdf = glossy_pGuide * pdf_guide + (1.0f - glossy_pGuide) * outPdf;
			}
		} else if (matType == MaterialType::RoughDielectric) {
			RoughDielectricBxDF<float> bx{ fm.ior, glossy_alpha, glossy_alpha_v };
			float v = bx.f(glossy_wi_x, glossy_wi_y, glossy_wi_z, nfEta, wo_x, wo_y, wo_z);
			fr = fg = fb = v;
			outPdf = bx.pdf(glossy_wi_x, glossy_wi_y, glossy_wi_z, nfEta, wo_x, wo_y, wo_z);
		} else if (matType == MaterialType::CoatedDiffuse) {
			// Real per-point reflectance when texture-bound (uv_u/uv_v is
			// the CURRENT hit's own UV, threaded in for exactly this - see
			// this function's own uv_u/uv_v parameter comment), matching
			// evaluate_materials()'s own scatter-path lookup for the same
			// material exactly; fm.albedo (flat) otherwise.
			const float3 coatedAlbedo = (fm.textureIdx >= 0)
				? wf_sample_texture(textures, texturePixels, fm.textureIdx, uv_u, uv_v, hit_point) * fm.emissionScale
				: fm.albedo;
			CoatedDiffuseBxDF<float> bx{ coatedAlbedo.x, coatedAlbedo.y, coatedAlbedo.z, fm.ior, glossy_alpha, glossy_alpha_v };
			uint64_t s0, s1; wf_random_seed64_pair(seed, s0, s1);
			bx.f(glossy_wi_x, glossy_wi_y, glossy_wi_z, wo_x, wo_y, wo_z, s0, s1, fr, fg, fb);
			// pbrt's LayeredBxDF::PDF(): the same density the scatter step reports for the sampled continuation
			// (brdf_pdf_override), here at the queried light direction - MIS weighs both with it.
			outPdf = bx.pdf(glossy_wi_x, glossy_wi_y, glossy_wi_z, wo_x, wo_y, wo_z);
		} else if (matType == MaterialType::Measured) {
			// Real tabulated measured BRDF (pbrt-v4 MeasuredBxDF): f() and pdf() in the local frame the sampling side uses
			// (ShadingFrame::from_normal, see wf_sample_measured_material), not the UV-aligned glossy frame above.
			if (measured == nullptr || fm.textureIdx < 0 || (unsigned int)fm.textureIdx >= measured->numTables) return false;
			const GpuMeasuredTable& mtab = measured->tables[fm.textureIdx];
			ShadingFrame<float> mframe = ShadingFrame<float>::from_normal(normal.x, normal.y, normal.z);
			float mwox, mwoy, mwoz, mwix, mwiy, mwiz;
			mframe.to_local(phaseWo.x, phaseWo.y, phaseWo.z, mwox, mwoy, mwoz);
			mframe.to_local(queryDir.x, queryDir.y, queryDir.z, mwix, mwiy, mwiz);
			const float mlambda[3] = { 612.0f, 549.0f, 465.0f };   // the fixed query wavelengths, as the sampling side uses
			if (!wf_gpu_measured_f_pdf(mtab, mtab.isotropic != 0, measured->paramValues, measured->data, measured->mcdf, measured->ccdf,
			                           mwox, mwoy, mwoz, mwix, mwiy, mwiz, mlambda, fr, fg, fb, outPdf))
				return false;
		} else {
			float cond_ax = glossyCondAlpha, cond_ay = glossyCondAlphaV;
			if (cond_ax < 0.0f)
				ResolveCoatedConductorBaseAlpha(fm.condRoughness, fm.condRoughnessV, fm.remapRoughness,
				                                glossy_alpha, glossy_alpha_v, cond_ax, cond_ay);
			const float inv_ior = 1.0f / fm.ior;
			CoatedConductorBxDF<float> bx{ fm.eta_c.x * inv_ior, fm.eta_c.y * inv_ior, fm.eta_c.z * inv_ior,
			                               fm.k_c.x * inv_ior, fm.k_c.y * inv_ior, fm.k_c.z * inv_ior,
			                               fm.ior, glossy_alpha, glossy_alpha_v, fm.layerThickness, 0.0f, 0.0f, 10, 1,
			                               cond_ax, cond_ay };
			uint64_t s0, s1; wf_random_seed64_pair(seed, s0, s1);
			bx.f(glossy_wi_x, glossy_wi_y, glossy_wi_z, wo_x, wo_y, wo_z, s0, s1, fr, fg, fb);
			outPdf = bx.pdf(glossy_wi_x, glossy_wi_y, glossy_wi_z, wo_x, wo_y, wo_z);
		}
		outF = make_float3(fr, fg, fb);
		return true;
	};

	// Uplift an unbounded-positive RGB f() value (can exceed 1, unlike a
	// plain reflectance) to spectral - same technique as evaluate_materials()'s
	// own unboundedSpectrum lambda, deliberately WITHOUT the D65 illuminant
	// factor liftEmission below applies (that factor belongs to light sources,
	// not BRDF values).
	auto liftUnboundedRGB = [&](float3 rgb) -> SS {
		float m = rgb.x > rgb.y ? (rgb.x > rgb.z ? rgb.x : rgb.z) : (rgb.y > rgb.z ? rgb.y : rgb.z);
		float sc = 2.f * m;
		if (sc <= 0.f) return SS(0.f);
		float c0, c1, c2;
		dev_srgb_to_coeffs(rgb.x / sc, rgb.y / sc, rgb.z / sc, c0, c1, c2);
		RGBSigmoidPolynomial poly(c0, c1, c2);
		SS s(0.f);
		for (int i = 0; i < kWFNWavelengths; ++i)
			s[i] = sc * poly(swl.lambda[i]);
		return s;
	};

	// Factors the matType-based BSDF value/color/pdf computation that used
	// to be hand-duplicated identically across all three NEE light-sampling
	// blocks below (area-light draw, sky/portal, punctual) - a code-health
	// pass, see this project's own plan. A local lambda (not a free
	// function) because it calls evalGlossyF above, itself a local lambda
	// capturing per-shading-point state by reference - a real function
	// can't reach a caller's own local lambda, so this waits for
	// evalGlossyF's own eventual conversion to a real function (a separate,
	// later step) before it could become one too. outGlossyPdf is always
	// computed, even for the punctual-light call site below, which doesn't
	// use it for MIS (delta lights have none) - matches that call site's
	// own prior behavior of computing and discarding an identical unused
	// pdf, not a new cost.
	auto wf_local_light_bsdf = [&](float cos_l, const float3& lightDir,
									float& outBsdfVal, SS& outBsdfColor, float& outGlossyPdf) {
		outBsdfVal = 1.0f / 3.14159265f; // Lambertian default
		// See this function's own attenuation-reuse rationale where this
		// lambda's 3 call sites used to inline this same chain: `attenuation`
		// already equals albedoSpectrum(mat.albedo) for Lambertian and the
		// phase-scatter case, direction-independent, safe to reuse directly.
		outBsdfColor = SS(1.f);
		outGlossyPdf = 0.0f;
		// NormalMappedLambertian shades as a plain Lambertian with the perturbed normal (its sampling case in
		// wavefront_kernels_materials.cu sets attenuation = albedoSpectrum(mat.albedo)), so its NEE needs the same
		// albedo. Without it the colour stayed at the white default and a normal-mapped blue sphere lit by a lamp
		// rendered grey (B12: 164-188% of the CPU over the sphere).
		if (matType == MaterialType::Lambertian || matType == MaterialType::NormalMappedLambertian) {
			outBsdfColor = attenuation;
		} else if (matType == MaterialType::NormalizedFresnel) {
			float inv_eta = 1.0f / nfEta;
			float nf_c = 1.0f - 2.0f * FresnelMoment1(inv_eta);
			if (nf_c <= 0.0f) nf_c = 1e-6f;
			float fr_l = FrDielectric(cos_l, nfEta);
			outBsdfVal = (1.0f - fr_l) / (nf_c * 3.14159265f);
		} else if (matType == MaterialType::DiffuseTransmission) {
			// pbrt-v4 DiffuseTransmissionBxDF: R/pi for a light on wo's side of the surface, T/pi through it; the BSDF picks between the two cosine lobes
			// with probabilities pr/(pr+pt) and pt/(pr+pt), so the density at lightDir is that lobe's probability times |cos|/pi (cos_l is |cos| here).
			const MaterialData& dm = materials[matIdx];
			const float3 dtR = (dm.textureIdx >= 0)
				? wf_sample_texture(textures, texturePixels, dm.textureIdx, uv_u, uv_v, hit_point) * dm.emissionScale
				: dm.albedo;
			const float3 dtT = (dm.transmittanceTextureIdx >= 0)
				? wf_sample_texture(textures, texturePixels, dm.transmittanceTextureIdx, uv_u, uv_v, hit_point) * dm.transmittanceScale
				: dm.emission;
			const float dtPr = fmaxf(dtR.x, fmaxf(dtR.y, dtR.z)), dtPt = fmaxf(dtT.x, fmaxf(dtT.y, dtT.z));
			const bool reflSide = dot(lightDir, normal) > 0.0f;
			outBsdfVal = 1.0f / 3.14159265f;
			outBsdfColor = liftUnboundedRGB(reflSide ? dtR : dtT);
			outGlossyPdf = (dtPr + dtPt > 0.0f) ? ((reflSide ? dtPr : dtPt) / (dtPr + dtPt)) * cos_l / 3.14159265f : 0.0f;
		} else if (isPhase) {
			outBsdfVal = wf_hg_phase_value(dot(phaseWo, lightDir), phaseG);
			outBsdfColor = attenuation;
		} else {
			// Conductor/RoughDielectric/CoatedDiffuse/CoatedConductor
			// (glossy) - see evalGlossyF's own comment. bsdf_val=1 folds the
			// whole per-channel f() into bsdf_color instead of splitting it
			// into a scalar shape * color like the Lambertian/
			// NormalizedFresnel cases above (those have an achromatic BRDF
			// shape; this one doesn't). evalGlossyF itself returns false
			// both for "not one of my 4 types" (in which case leave
			// outBsdfVal/outBsdfColor at their Lambertian-shaped defaults,
			// unchanged from before this else - matches e.g.
			// NormalMappedLambertian's existing behavior) and "wrong
			// hemisphere for one of my 4 types" (this light sits behind the
			// glass' reflection side) - only the second case should zero
			// the contribution, so explicitly re-check matType here rather
			// than trusting evalGlossyF's return value alone.
			float3 fRgb;
			if (evalGlossyF(lightDir, fRgb, outGlossyPdf)) {
				outBsdfVal = 1.0f;
				outBsdfColor = liftUnboundedRGB(fRgb);
			} else if (glossy_isType) {
				outBsdfVal = 0.0f;
			}
		}
	};

	// -------------------------------------------------------------------------
	// NEE: direct-light shadow ray (non-specular materials only)
	// -------------------------------------------------------------------------
	if (!is_specular) {
	// The ReSTIR-or-classic direct-light draw - see wf_nee_pick_light's own
	// header comment for the full ReSTIR-vs-classic rationale and side-
	// effect inventory. haveSample/to_light/max_dist/light_pdf/nee_norm/
	// light_emission_spec are filled by wf_nee_pick_light exactly as they
	// used to be filled by this same code inline, then consumed identically
	// by the shared BSDF-evaluation/shadow-ray code that follows.
	NeeLightSample neeSample = wf_nee_pick_light(hit_point, seed, time,
		spheres, quads, triangles, bilinearPatches, disks, cylinders,
		materials, lightIndices, lightKinds, aliasTable, numLights,
		textures, texturePixels, lightBvh, swl,
		pixelIndex, depth, matIdx, isPhase,
		normal, phaseWo, phaseG,
		restirReservoirs, restirCtx,
		mediumEntryPoint, mediumMeanFreePath,
		restirVolumeReservoirs, restirVolumeCtx,
		volumeMatIdxOut, volumePhaseWoGOut, volumeEntryPointOut);
	bool   haveSample = neeSample.haveSample;
	float3 to_light = neeSample.to_light;
	float  max_dist = neeSample.max_dist;
	float  light_pdf = neeSample.light_pdf;
	float  nee_norm = neeSample.nee_norm;
	SS light_emission_spec = neeSample.light_emission_spec;

	// RoughDielectric NEE can reach lights on EITHER side of the
	// interface (reflection when raw_cos>0, transmission/"seen through
	// the glass" when raw_cos<0) - see evalGlossyF's own comment and
	// optix_device_helpers.h's identical two-sided RoughDielectric NEE,
	// which this now matches. Every other material stays reflection-
	// only (raw_cos>0 required), same as before. Shared by both the ReSTIR
	// and classic paths above - haveSample/nee_norm/to_light/light_pdf are
	// filled identically by either one by this point.
	{
		float raw_cos = dot(to_light, normal);
		if (haveSample && nee_norm > 0.0f && (isPhase || twoSidedNee || raw_cos > 0.0f)) {
			// A phase function has no hemisphere/cosine restriction (it's
			// normalized over the full sphere, unlike a surface BRDF) - the
			// `cos_l` slot is set to 1 so the shared `bsdf_val * cos_l`
			// formula below reduces to the phase value alone. RoughDielectric
			// uses the absolute cosine (matches optix_device_helpers.h's
			// fabsf(llz)): the sign only selects reflection vs. transmission,
			// already handled inside evalGlossyF/RoughDielectricBxDF, not a
			// zero-below-the-hemisphere cutoff like the other glossy types.
			float cos_l = isPhase ? 1.0f
				: (twoSidedNee) ? fabsf(raw_cos)
				: fmaxf(raw_cos, 0.0f);
			float bsdf_val, glossyPdf;
			SS bsdf_color;
			wf_local_light_bsdf(cos_l, to_light, bsdf_val, bsdf_color, glossyPdf);
			// glossyPdf (the BSDF pdf evaluated AT to_light, from evalGlossyF
			// above) takes priority for the 5 glossy types - brdf_pdf_override
			// is the pdf at the unrelated BSDF-sampled continuation direction
			// and must not be reused here (see evalGlossyF's own header
			// comment). glossyPdf is 0 (falls through to the non-glossy
			// branches) for every other material type, matching prior
			// behavior exactly.
			// A phase vertex's pdf at the light direction is the phase value itself
			// (cos_l == 1); brdf_pdf_override is the pdf at the CONTINUATION
			// direction, which only equals it for an isotropic phase (g == 0).
			float brdf_pdf_l = (glossyPdf > 0.0f) ? glossyPdf
				: (!isPhase && brdf_pdf_override > 0.0f) ? brdf_pdf_override : (bsdf_val * cos_l);
			float mis_w = wf_mis(light_pdf, brdf_pdf_l);

			// Spectral direct-light contribution. nee_norm is 1/light_pdf
			// classically, or the ReSTIR reservoir's own unbiased contribution
			// weight W (see this function's own useRestir block comment) -
			// either way it already IS the correct radiance normalization, so
			// this multiplies rather than dividing by light_pdf.
			SS Ld = (mis_w * bsdf_val * cos_l * nee_norm) * throughput * bsdf_color * light_emission_spec;

			// 0.01, not the original 0.001: a scene with many densely-packed
			// custom primitives (spheres) reproducibly crashed the shadow
			// OptiX launch with an illegal memory access - the 0.001 offset
			// left the shadow ray's origin too close to its own emitting
			// surface (and, at high primitive density, to a neighboring
			// primitive's surface too) for this driver's any-hit traversal
			// over custom AABBs to handle; confirmed via bisection on
			// synthetic sphere-field pbrt scenes (compute-sanitizer memcheck/
			// initcheck found nothing, ruling out a plain buffer overrun).
			// Offsetting along BOTH the shading normal AND the shadow ray's
			// own direction, not just the normal alone. The normal-only
			// offset is load-bearing on its own (see above): a pure
			// direction-only offset - matching optix_device_helpers.h's
			// trace_shadow_ray() - was tried first and broke
			// NormalMappedLambertian (scene 20 rendered 99% black), since a
			// shading normal bent away from the true surface no longer
			// guarantees "away from this primitive" the way the true
			// geometric normal does, letting the ray re-enter its own
			// (unperturbed) sphere at a grazing angle. But normal-only,
			// even at a much larger shadow_eps, turned out far weaker than
			// direction-only at escaping self-intersection on dense,
			// closely-packed real geometry: Sibenik Cathedral's stone
			// tracery false-occluded most sky-NEE shadow rays at
			// shadow_eps=0.01, and bumping shadow_eps up to 2.0 with a
			// normal-only offset barely helped (GPU stayed under 50% of
			// CPU's brightness at matched settings) - adding the direction
			// component back in (while keeping the normal component, so
			// NormalMappedLambertian stays fixed) closed the gap to ~73%
			// at shadow_eps=0.5, confirmed by direct experiment. 2.0 with
			// the combined offset overshot badly (GPU 1.5x *brighter* than
			// CPU - real light leaks from skipping past legitimate
			// occluders), which is why this is a per-scene-tunable value
			// (GpuCameraParams::shadowRayEpsilon), not a blanket increase.
			// isPhase (medium-interior scatter point): skip the normal-based
			// offset entirely - `normal` here is the sphere's own entry-surface
			// normal, not a meaningful direction at this interior point, and
			// direction-only is exactly what optix_device_helpers.h's own
			// trace_shadow_ray() already uses for this same phase-scatter NEE
			// on the recursive backend (see that call site's comment).
			// The normal-offset nudges toward whichever side to_light is
			// actually on (copysignf(shadow_eps, raw_cos), not always
			// +shadow_eps): for every material except RoughDielectric's new
			// transmission-side case raw_cos>0 always holds here (the gate
			// above still requires it), so this is exactly +shadow_eps as
			// before - only RoughDielectric's transmission side (raw_cos<0)
			// newly nudges the origin to the far side of the surface instead
			// of back into the same hemisphere the ray isn't going toward.
			// giCandidateEligible redirects this ray's Ld into the GI candidate
			// buffer instead of the real framebuffer, once accumulate_shadow
			// resolves occlusion, whenever this is a depth==1 hit whose
			// originating x0 was GI-eligible; a no-op (false) for every other
			// call (depth==0's own NEE, or GI disabled/ineligible this frame).
			wf_push_nee_shadow_ray(shadowQueue, hit_point, normal, isPhase, shadow_eps,
				raw_cos, to_light, max_dist - 0.002f, Ld, filterWeight, swl,
				pixelIndex, time, giCandidateEligible, wf_pcg(seed));
		}
	}
	// -------------------------------------------------------------------------
	// NEE: sky (infinite) light. Mirrors CPU's camera.h Strategy A-2 and the
	// recursive backend's own sky-NEE block (optix_device_helpers.h) - see
	// that comment for the full story (GPU used to only pick up sky light via
	// a lucky BSDF-sampled escape, no NEE, which under-lit small-aperture
	// interiors like Sibenik Cathedral). skyColor is the same constant color
	// the miss path already uses (camera.backgroundColor - see
	// optix_miss.h/accumulate_miss's own use of it), defaulting to black for
	// every scene without a sky, which makes this a free no-op there.
	// -------------------------------------------------------------------------
	if (!is_specular && (skyColor.x > 0.0f || skyColor.y > 0.0f || skyColor.z > 0.0f)) {
		// wf_sample_sky_nee() (wavefront_sky_light.h) dispatches to real HDR
		// importance sampling when the scene's infinite light carries an
		// image (skyDist.height > 0), falling back to this exact uniform-
		// sphere sample + flat skyColor otherwise - see optix_device_
		// helpers.h's own Lambertian sky-NEE block for the recursive
		// backend's identical dispatch, and optix_sky_light.h for the full
		// algorithm comment.
		float3 sky_dir, sky_Le_val; float pdf_sky;
		wf_sample_sky_nee(skyDist, portalLight, seed, skyColor, hit_point, sky_dir, pdf_sky, sky_Le_val);
		// See the area-light block above for the RoughDielectric two-sided
		// rationale (matches optix_device_helpers.h's sky-NEE block).
		float  raw_cos = dot(sky_dir, normal);
		float  cos_l   = isPhase ? 1.0f
			: (twoSidedNee) ? fabsf(raw_cos)
			: raw_cos;
		// pdf_sky > 0.0f is REQUIRED here, not just an optimization: a
		// portal-light NEE sample (wf_sample_sky_nee() -> gpu_portal_sample_Li())
		// returns pdf_sky == 0.0f with sky_Le_val == (0,0,0) whenever the
		// portal window subtends zero area from hit_point (a routine outcome
		// for most points not facing the window, not a rare edge case) -
		// without this guard, `isPhase`/RoughDielectric (which bypass the
		// cos_l gate entirely) unconditionally divide by pdf_sky below and
		// produce NaN (0.0f/0.0f). Mirrors CPU's own
		// `if (portal->sample_li(...) && pdf_portal > 0.0)` guard
		// (src/TheRestOfYourLife/camera.h) and this file's own
		// medium_phase_nee_mis()-equivalent pattern elsewhere.
		if (pdf_sky > 0.0f && (isPhase || twoSidedNee || cos_l > 0.0f)) {
			float bsdf_val, glossyPdf;
			SS bsdf_color;
			wf_local_light_bsdf(cos_l, sky_dir, bsdf_val, bsdf_color, glossyPdf);
			// See the area-light block above: glossyPdf (pdf at sky_dir) takes
			// priority over brdf_pdf_override (pdf at the unrelated
			// continuation direction) for the 5 glossy types.
			float brdf_pdf_sky = (glossyPdf > 0.0f) ? glossyPdf
				: (!isPhase && brdf_pdf_override > 0.0f) ? brdf_pdf_override : (bsdf_val * cos_l);
			float mis_w = wf_mis(pdf_sky, brdf_pdf_sky);

			// Uplift RGB sky_Le_val (the real per-direction radiance for an
			// image sky, or the flat skyColor otherwise - see
			// wf_sample_sky_nee()) to spectrum (same pattern as liftEmission
			// above, including the D65 illuminant factor -- see that
			// lambda's own comment).
			float m = sky_Le_val.x > sky_Le_val.y ? (sky_Le_val.x > sky_Le_val.z ? sky_Le_val.x : sky_Le_val.z)
			                                       : (sky_Le_val.y > sky_Le_val.z ? sky_Le_val.y : sky_Le_val.z);
			float sc = 2.f * m;
			SS sky_spec(0.f);
			if (sc > 0.f) {
				float c0, c1, c2;
				dev_srgb_to_coeffs(sky_Le_val.x / sc, sky_Le_val.y / sc, sky_Le_val.z / sc, c0, c1, c2);
				RGBSigmoidPolynomial poly(c0, c1, c2);
				for (int i = 0; i < kWFNWavelengths; ++i)
					sky_spec[i] = sc * poly(swl.lambda[i]) * dev_sample_d65(swl.lambda[i]);
			}

			SS Ld = (mis_w * bsdf_val * cos_l / pdf_sky) * throughput * bsdf_color * sky_spec;

			// See wf_push_nee_shadow_ray's own comment for why this is a
			// combined normal+direction offset (not the normal alone), why
			// isPhase skips the normal term entirely, why the normal term
			// uses copysignf(shadow_eps, raw_cos) rather than always
			// +shadow_eps, and why tMax is the unadjusted 1e30f sentinel here
			// (unlike the area/punctual sites' max_dist/t_max - 0.002f).
			wf_push_nee_shadow_ray(shadowQueue, hit_point, normal, isPhase, shadow_eps,
				raw_cos, sky_dir, 1e30f, Ld, filterWeight, swl,
				pixelIndex, time, giCandidateEligible, wf_pcg(seed));
		}
	}

	// -------------------------------------------------------------------------
	// NEE: punctual (point/spot/distant) delta lights. Unlike the area-light
	// block above (one stochastic pick via alias table), every contributing
	// punctual light gets its own shadow ray - matches recursive path
	// (optix_device_helpers.h add_punctual_lights_lambertian) and the CPU
	// reference (camera.h punct_lights loop): pdf=1 by construction, so no
	// MIS weight or pdf division, just beta * BRDF * cos_theta * Li per light.
	// -------------------------------------------------------------------------
	for (unsigned int pli = 0; pli < numPunctualLights; ++pli) {
		float3 wi, Li; float t_max;
		if (!wf_eval_punctual_light(punctualLights[pli], hit_point, wi, Li, t_max)) continue;
		// See the area-light block above for the RoughDielectric two-sided
		// rationale (matches optix_device_helpers.h's punctual-light NEE,
		// which only excludes exact grazing (plz==0), not either sign).
		// isPhase (medium-interior scatter point, see the area-light block's
		// own identical comment): `normal` is meaningless here, so a phase
		// function - defined over the full sphere, no hemisphere restriction
		// - must skip this cosine-based cull entirely, the same way the
		// area-light block above already does. This was missing here even
		// though the area/sky blocks in this same function already handle
		// it - previously the only material this loop's own isPhase-shaped
		// gap could reach was DielectricMedium's interior sub-case; fixing
		// it here at the same time real NEE was added to Medium/CloudMedium/
		// RgbGridMedium/GridMedium's own phase-scatter cases, since those
		// newly reach this exact gap too.
		float raw_cos = dot(wi, normal);
		if (!isPhase && !twoSidedNee && raw_cos <= 0.0f) continue;
		if (!isPhase && twoSidedNee && raw_cos == 0.0f) continue;
		float cos_l = isPhase ? 1.0f
			: (twoSidedNee) ? fabsf(raw_cos) : raw_cos;

		// No MIS weight for punctual (delta) lights, same as every other
		// material here - the pdf-at-wi output isn't needed here, unlike the
		// area/sky NEE blocks above, but wf_local_light_bsdf always computes
		// it anyway (matches this call site's own prior behavior of
		// computing and discarding an identical unused pdf).
		float bsdf_val, unusedPdf;
		SS bsdf_color;
		wf_local_light_bsdf(cos_l, wi, bsdf_val, bsdf_color, unusedPdf);

		// Uplift RGB Li to spectrum (same pattern as liftEmission above,
		// including the D65 illuminant factor -- see that lambda's own
		// comment)
		float m = Li.x > Li.y ? (Li.x > Li.z ? Li.x : Li.z) : (Li.y > Li.z ? Li.y : Li.z);
		float sc = 2.f * m;
		SS Li_spec(0.f);
		if (sc > 0.f) {
			float c0, c1, c2;
			dev_srgb_to_coeffs(Li.x / sc, Li.y / sc, Li.z / sc, c0, c1, c2);
			RGBSigmoidPolynomial poly(c0, c1, c2);
			for (int i = 0; i < kWFNWavelengths; ++i)
				Li_spec[i] = sc * poly(swl.lambda[i]) * dev_sample_d65(swl.lambda[i]);
		}

		SS Ld = (bsdf_val * cos_l) * throughput * bsdf_color * Li_spec;

		// See wf_push_nee_shadow_ray's own comment for why this is a combined
		// normal+direction offset, not the normal alone, why the normal term
		// uses copysignf(shadow_eps, raw_cos) rather than always +shadow_eps
		// (RoughDielectric's transmission side), why isPhase skips the normal
		// term entirely (no meaningful normal at a medium-interior scatter
		// point), and why the seed mixes in `pli`: unlike the area/sky sites
		// (each fires at most once per call), this loop can push several
		// shadow rays per call, one per punctual light, with no
		// wf_rand(seed)-consuming call between iterations to decorrelate them
		// - mixing in pli keeps two lights on the same hit from getting
		// identical ratio-tracking noise if they both cross the same medium.
		wf_push_nee_shadow_ray(shadowQueue, hit_point, normal, isPhase, shadow_eps,
			raw_cos, wi, t_max - 0.002f, Ld, filterWeight, swl,
			pixelIndex, time, giCandidateEligible, wf_pcg(seed ^ (pli * 0x9E3779B9u)));
	}

	// Flush accumulated prior-bounce radiance (once, regardless of whether
	// any area/punctual lights actually contributed above). Deliberately NOT
	// redirected for ReSTIR GI even when giCandidateEligible - see
	// giCandidateOut's own parameter comment (MVP scope: a glowing x1 is
	// treated as "no GI candidate this frame", not approximated).
	if ((bool)radiance) addToFramebuffer(pixelIndex, radiance * filterWeight);
	} // if (!is_specular) - NEE (area + punctual lights)

	// -------------------------------------------------------------------------
	// Bounce: push next ray
	// -------------------------------------------------------------------------
	SS new_throughput = throughput * attenuation;
	// The CPU's per-path throughput ceiling (kMaxPathThroughput, optix_math_helpers.h) - see its comment.
	for (int i = 0; i < kWFNWavelengths; ++i)
		if (new_throughput[i] > kMaxPathThroughput) new_throughput[i] = kMaxPathThroughput;
	// A BSDF sample that was rejected after NEE still ran (rough glass: a reflection below the horizon, a refraction on the
	// wrong side) carries zero weight - there is nothing to continue. Specular vertices skip NEE, so they flush the radiance
	// carried so far here, as the Russian roulette termination below does.
	if (!(new_throughput.MaxComponentValue() > 0.0f)) {
		if (is_specular) addToFramebuffer(pixelIndex, radiance * filterWeight);
		return;
	}

	// Russian roulette (pbrt-v4 PathIntegrator formula - matches CPU's
	// camera.h and the recursive backend's optix_raygen.h exactly):
	// rrBeta = beta * etaScale; q = max(0, 1 - MaxComponent(rrBeta));
	// terminate if rand < q, else reweight beta itself by 1/(1-q). This
	// file used to compute p = MaxComponent(beta) directly and terminate on
	// rand >= p - an older (pbrt-v3-style) scheme that isn't wrong on its
	// own, but disagreed with the OTHER two backends in this codebase, and
	// started at depth>=3 rather than the depth>1 both of those use (so the
	// wavefront backend also spent one extra bounce before RR could kick
	// in). etaScale (pbrt-v4's per-refraction correction that avoids
	// killing transmission-heavy paths too aggressively) now matches both
	// other backends too - see this parameter's own comment.
	if (depth > 1) {
		float rr_max = (new_throughput * etaScale).MaxComponentValue();
		if (rr_max < 1.0f) {
			float q = fmaxf(0.0f, 1.0f - rr_max);
			if (wf_rand(seed) < q) {
				if (is_specular) addToFramebuffer(pixelIndex, radiance * filterWeight);
				return;
			}
			new_throughput = new_throughput / (1.0f - q);
		}
	}

	// -------------------------------------------------------------------------
	// Neural Radiance Cache lookup (Live Preview only) - see this project's
	// own plan. Tried BEFORE the probe cache below: unlike that cache's
	// view-independent, Lambertian-only SH-L1 grid, NRC is directionally
	// aware and supports RoughMetal too, so it's strictly more capable
	// wherever both are eligible. Gates:
	//  - depth>=kNrcMinDepth (matches the probe cache's own threshold).
	//  - !is_specular (same as probe cache/path guiding).
	//  - Lambertian or RoughMetal only - the ONLY two material types the
	//    training pipeline (wavefront_kernels_nrc.cu) ever visits/learns
	//    from; querying the network for a material type it was never
	//    trained on would just return an unreliable extrapolation with no
	//    warm-up/confidence gate protecting it (that gate only covers "the
	//    whole network is cold", not "this specific input is off the
	//    training manifold").
	//  - nrcTrainingSteps>=kNrcWarmupSteps - a cold/untrained network's
	//    output is near-random noise; rejecting it here is the exact same
	//    "0 means never updated, skip it" convention GpuProbe::
	//    numRaysEverTraced==0 already uses for the probe cache, applied
	//    globally to the whole network rather than per-cell.
	//  - a STOCHASTIC termination roll (not a hard cutoff): pTerm rises
	//    linearly with depth past kNrcMinDepth, capped at
	//    kNrcMaxTerminationProb (<1.0) so the renderer stays asymptotically
	//    unbiased in the limit and a real bounce is always still possible.
	//    Drawn from the SAME wf_rand(seed) sequence Russian roulette above
	//    already advances - a path RR just killed never reaches this check.
	// On acceptance: reject (fall through to a real bounce, exactly like a
	// miss) if the network's own output has any non-finite or negative
	// component - a hard backstop independent of whatever gradient
	// clipping the training side applies, since this is a hand-written,
	// from-scratch-trained model with no external correctness guarantee.
	//
	// IMPORTANT: the network is trained (nrc_bootstrap_and_train,
	// wavefront_kernels_nrc.cu) to predict ONLY the radiance arriving at
	// the queried vertex via CONTINUING the path beyond it (>=1 more
	// bounce) - it deliberately does NOT include this same vertex's own
	// direct-lighting term, which the classic/ReSTIR NEE block earlier in
	// THIS function has already added to the framebuffer for the current
	// hit. Substituting the network's output here is therefore additive
	// with that real NEE, not a replacement for it - mirroring exactly how
	// the world-space probe cache below is genuinely one-bounce-indirect-
	// only (see its own header comment). An earlier version of this
	// feature trained the network on the queried vertex's own TOTAL
	// radiance (direct term included) and substituted it here anyway,
	// silently double-counting that vertex's own direct light every time
	// this block fired - a real, confirmed bug fixed at the training-
	// target level (nrc_bootstrap_and_train), not by restructuring this
	// function's own NEE/RR/substitution ordering.
	if (nrcWeights != nullptr && !is_specular && depth >= kNrcMinDepth &&
		(matType == MaterialType::Lambertian || matType == MaterialType::RoughMetal) &&
		nrcTrainingSteps >= kNrcWarmupSteps) {
		const float pTerm = fminf((float)(depth - kNrcMinDepth) * kNrcTerminationRamp, kNrcMaxTerminationProb);
		if (wf_rand(seed) < pTerm) {
			// glossyAlpha, NOT a fresh wf_glossy_alpha(materials[matIdx],
			// do_regularize) call - do_regularize alone (without the
			// caller's own `regularize` render-option flag folded in, see
			// this function's own do_regularize parameter comment above)
			// is NOT the same value the caller's real BSDF sample/NEE used
			// for this material's alpha. glossyAlpha is that exact,
			// already-computed value (0 for Lambertian, meaningless but
			// harmless), keeping the query's own roughness feature
			// bit-identical to what the real render path used - and to
			// what the training pipeline computes (wavefront_kernels_nrc.cu
			// always passes do_regularize=false, which would otherwise
			// silently diverge from a regularized live query whenever
			// do_regularize alone is true but the caller's real alpha
			// wasn't regularized).
			const float roughness = (matType == MaterialType::Lambertian)
				? kNrcLambertianRoughnessSentinel
				: glossyAlpha;
			float nrcFeatures[kNrcInputDim];
			wf_nrc_encode_features(hit_point, scattered_dir, normal, roughness, materials[matIdx].albedo,
									nrcAabbMin, nrcAabbExtent, nrcFeatures);
			NrcForwardCache nrcCache;
			wf_nrc_forward(nrcWeights, nrcFeatures, nrcCache);
			const bool valid = isfinite(nrcCache.out[0]) && isfinite(nrcCache.out[1]) && isfinite(nrcCache.out[2]) &&
								nrcCache.out[0] >= 0.0f && nrcCache.out[1] >= 0.0f && nrcCache.out[2] >= 0.0f;
			if (valid) {
				const float3 L_hat = make_float3(nrcCache.out[0], nrcCache.out[1], nrcCache.out[2]);
				const SS nrcSpec = wf_lift_rgb_to_spectrum(L_hat, swl, /*isIlluminant=*/true);
				addToFramebuffer(pixelIndex, new_throughput * nrcSpec * filterWeight);
				return;
			}
		}
	}

	// -------------------------------------------------------------------------
	// World-space irradiance probe cache lookup (Live Preview only) - see
	// this project's own plan. Fills exactly the gap ReSTIR GI's own
	// Lambertian-x0, depth 0->1-only MVP leaves open: every bounce beyond
	// depth 1 today traces on with classic NEE at full cost forever, with no
	// caching at all. A cache HIT here terminates the path immediately
	// instead of pushing another bounce - a MISS (the common case for an
	// uninitialized or newly-scaled scene) falls through to the ordinary
	// bounce below completely unaffected. Gated on Lambertian only, same
	// "BSDF eval trapped in a capturing lambda, can't be re-evaluated outside
	// this function's own call frame" reason ReSTIR GI's own x0 restriction
	// has (wavefront_kernels_restir.cu's own "MVP scope" comment) - every
	// other material type falls through unaffected, same as a cache miss.
	// depth>=2 (kProbeCacheMinDepth): depth 0->1 is ReSTIR GI's own territory
	// already; this cache only ever replaces bounces GI itself doesn't reach.
	if (probeGrid != nullptr && !is_specular && depth >= /*kProbeCacheMinDepth*/ 2 &&
		matType == MaterialType::Lambertian) {
		const float3 cacheIrradiance = wf_query_probe_grid(probeGridMeta, probeGrid, hit_point, normal);
		if (cacheIrradiance.x > 0.0f || cacheIrradiance.y > 0.0f || cacheIrradiance.z > 0.0f) {
			// Uses new_throughput (== throughput * attenuation, already
			// reweighted by the Russian-roulette survival boost just above)
			// rather than the pre-bounce `throughput` - this contribution
			// terminates the path exactly like a real next bounce would, so
			// it must carry the SAME BSDF-color-times-RR-boost factor a real
			// next bounce's own throughput would, not the un-boosted,
			// pre-attenuation value. attenuation == albedoSpectrum(mat.albedo)
			// for Lambertian is already folded into new_throughput here -
			// multiplying by it a second time (as an earlier version of this
			// block did) would double-count the surface color. cacheIrradiance
			// is treated as an illuminant (like mat.emission/background
			// radiance elsewhere in this codebase) since it already IS a
			// fully-resolved incoming-radiance estimate by the time it left
			// the probe cache, not a reflectance being lit.
			const SS cacheSpec = wf_lift_rgb_to_spectrum(cacheIrradiance, swl, /*isIlluminant=*/true);
			addToFramebuffer(pixelIndex, new_throughput * cacheSpec * (1.0f / 3.14159265f) * filterWeight);
			return;
		}
	}

	RayWorkItem next;
	next.origin     = hit_point + 0.001f * scattered_dir;
	next.direction  = normalize(scattered_dir);
	next.seed            = seed;
	next.pixelIndex      = pixelIndex;
	next.depth           = depth + 1;
	next.specular_bounce = is_specular ? 1 : 0;
	// See RayWorkItem::any_nonspecular's own comment - never cleared, only
	// ever set once this bounce (or an earlier one) was non-specular.
	next.any_nonspecular = (do_regularize || !is_specular) ? 1 : 0;
	// Already fully updated for this event - see this function's own
	// etaScale parameter comment.
	next.etaScale = etaScale;
	// Pure reconstruction weight, carried unchanged - see
	// RayWorkItem::filterWeight's own comment.
	next.filterWeight = filterWeight;
	// BRDF PDF of this new direction, for MIS if this ray escapes the scene
	// on its next bounce (see RayWorkItem::brdf_pdf's own comment) - mirrors
	// optix_intersection_sphere.h's brdf_pdf_out exactly (0 for specular,
	// else brdf_pdf_override if the material set one, else the cosine-
	// weighted hemisphere pdf every non-specular case here samples from).
	next.scatterOrigin = hit_point;   // see RayWorkItem::scatterOrigin
	next.brdf_pdf = is_specular ? 0.0f
		: (brdf_pdf_override > 0.0f ? brdf_pdf_override
			: fmaxf(dot(next.direction, normal), 0.0f) / 3.14159265f);

	// ReSTIR GI (see this function's own giOriginContext parameter comment) -
	// stash x0's shading context for the GI finalize pass to consume once
	// depth 1 resolves. MVP scope: Lambertian x0 ONLY, not merely !is_specular
	// - the GI finalize pass (wavefront_kernels_restir.cu) needs to evaluate
	// x0's REAL BSDF value in an arbitrary (possibly reservoir-reused)
	// direction, and every OTHER non-specular material's own BSDF evaluation
	// here (NormalizedFresnel, or the 4 glossy evalGlossyF-driven types) is a
	// capturing lambda entangled with a lot of this function's own local
	// state (dpdu tangent frame, glossyAlpha/glossyAlphaV, phaseWo...) that
	// cannot be evaluated again later, outside this call, without a much
	// larger refactor - deferred rather than attempted here. Lambertian's own
	// BSDF is trivial (albedo/pi, direction-independent), which is exactly
	// why it's the one case worth supporting first: it is also the classic,
	// highest-value GI showcase (diffuse-to-diffuse color bleeding). A
	// degenerate/grazing next.brdf_pdf (0.0f despite matType==Lambertian -
	// rare, but possible) naturally marks this pixel GI-ineligible via
	// GpuGiOriginContext::valid()'s own `pdfAtX0 > 0.0f` check, with no
	// special-casing needed here.
	if (giOriginContext != nullptr && depth == 0 && matType == MaterialType::Lambertian) {
		GpuGiOriginContext& ctx = giOriginContext[pixelIndex];
		ctx.x0Point = hit_point;
		ctx.x0Normal = normal;
		ctx.dirX0ToX1 = next.direction;
		ctx.pdfAtX0 = next.brdf_pdf;
		ctx.materialIdx = matIdx;
		for (int i = 0; i < kWFNWavelengths; ++i) {
			ctx.throughputX0[i] = throughput[i];
			ctx.wavelengths[i] = swl.lambda[i];
			ctx.wavelength_pdfs[i] = swl.pdf[i];
		}
	}

	next.tMin       = 0.001f;
	next.tMax       = 1e30f;
	// Fixed for the whole path once sampled at the camera - see RayWorkItem::
	// time's own comment.
	next.time       = time;
	for (int i = 0; i < kWFNWavelengths; ++i) {
		next.throughput[i]      = new_throughput[i];
		next.radiance[i]        = is_specular ? radiance[i] : 0.0f;
		next.wavelengths[i]     = swl.lambda[i];
		next.wavelength_pdfs[i] = swl.pdf[i];
	}
	nextRayQueue.push(next);
}

// Uplift a flat RGB color to a spectral sample at the given hero
// wavelengths. Shared across every call site in the split kernel files that
// needs the same uplift: evaluate_materials's own DiffuseLight direct-hit
// emission and MaterialType::Medium "rgb Le" cases
// (wavefront_kernels_materials.cu), plus resolve_bssrdf_exit
// (wavefront_kernels_bssrdf.cu) and accumulate_miss
// (wavefront_kernels_accumulate.cu). NOT the same thing as the `liftEmission`
// lambda inside wf_finish_material_scatter just above - lambdas can't cross
// function boundaries, so that one stays a separate, function-local copy of
// the same math.
//
// isIlluminant: true for a light-source colour (DiffuseLight/medium Le
// above, background/sky radiance in accumulate_miss), which needs the D65
// illuminant factor -- see liftEmission's own comment above for why. false
// (default) for an already-computed reflectance-like weight (e.g.
// resolve_bssrdf_exit's Sp spatial term), which must NOT get an extra
// illuminant multiply -- it's not a colour being lit, it's already a
// throughput.
__device__ __forceinline__ SampledSpectrum<kWFNWavelengths> wf_lift_rgb_to_spectrum(
	float3 rgb, const SampledWavelengths<kWFNWavelengths>& swl, bool isIlluminant = false
) {
	using SS = SampledSpectrum<kWFNWavelengths>;
	float m = rgb.x > rgb.y ? (rgb.x > rgb.z ? rgb.x : rgb.z) : (rgb.y > rgb.z ? rgb.y : rgb.z);
	float sc = 2.f * m;
	if (sc <= 0.f) return SS(0.f);
	float c0, c1, c2;
	dev_srgb_to_coeffs(rgb.x / sc, rgb.y / sc, rgb.z / sc, c0, c1, c2);
	RGBSigmoidPolynomial poly(c0, c1, c2);
	SS s(0.f);
	for (int i = 0; i < kWFNWavelengths; ++i) {
		s[i] = sc * poly(swl.lambda[i]);
		if (isIlluminant) s[i] *= dev_sample_d65(swl.lambda[i]);
	}
	return s;
}
