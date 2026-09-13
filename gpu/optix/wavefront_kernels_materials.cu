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
	// - see this project's own plan. Consulted ONLY by the Conductor/
	// RoughMetal cases below (v1 scope), BEFORE wf_finish_material_scatter is
	// even called, unlike probeGridMeta/probeGrid there (which stay on their
	// nullptr/default - Lambertian never reaches this kernel, see
	// WavefrontQueues::simpleHitQueue's own comment). guidingHistograms==
	// nullptr (the default, every non-Live-Preview call site, or Live
	// Preview with the feature toggled off) is a complete no-op - every
	// glossy case below falls through to its own existing, unmodified
	// GGX/VNDF sampling unchanged.
	GpuProbeGridMeta guidingGridMeta = {},
	const GpuGuidingHistogram* guidingHistograms = nullptr
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
	// Hair and (cylinder-only) Medium are real, reachable combinations this
	// single shared queue already knows how to shade (Hair via the real
	// MaterialType::Hair case below, keyed only on mat.type; Medium via
	// h.t/h.mediumTFar, recomputed by __closesthit__wf_cylinder - see
	// pbrt_gpu_builder.h's cylinder loop comment) - trapping either here
	// would be a genuine regression from real support to "falls back to
	// Lambertian"/an abort, not the unreachable-defensive-code every other
	// trapped type here still is. wf_material_supported_on_disk_cylinder_
	// geom() (this file, above) is the one place these per-shape exemptions
	// live, so this condition itself doesn't grow a new `&& !flagN` term
	// each time another (shape, material) combination gains real support.
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
	// pbrt-v4 alignment: only add emissive if depth==0 or specular_bounce==1
	// to avoid double-counting with NEE shadow rays at non-specular bounces.
	// -------------------------------------------------------------------------
	if (mat.type == MaterialType::DiffuseLight) {
		if (h.specular_bounce || h.depth == 0) {
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
					radiance = radiance + throughput * wf_lift_rgb_to_spectrum(le, swl, /*isIlluminant=*/true);
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
	// Set inside whichever glossy case (Conductor/RoughMetal/RoughDielectric/
	// CoatedDiffuse/CoatedConductor) this hit takes, via wf_glossy_alpha() -
	// passed to wf_finish_material_scatter's glossyAlpha parameter below so
	// NEE reuses the exact same regularized alpha the switch case's own
	// BSDF-sampling step just used, instead of re-deriving it a second time.
	// Left at 0 (harmless - see that parameter's own comment) for every
	// non-glossy case.
	float glossyAlphaForNEE = 0.0f;
	float glossyAlphaVForNEE = 0.0f;

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
		// Real tabulated measured-BRDF - see wf_sample_measured_material's
		// comment above. Matches material_pbrt.h `class measured`'s
		// skip_pdf=true: no NEE/MIS (is_specular=true), and the result needs
		// unboundedSpectrum (not albedoSpectrum) since sample_f()'s returned
		// fr/fg/fb is already the material's full throughput weight, not a
		// [0,1] albedo - same convention as MaterialType::Hair/Principled
		// above (their BxDF::sample() results are likewise already-weighted,
		// potentially-unbounded values).
		float3 sdir, atten;
		if (wf_sample_measured_material(h.rayDir, normal, mat,
				measuredTables, numMeasuredTables,
				measuredParamValues, measuredData, measuredMcdf, measuredCcdf,
				seed, sdir, atten)) {
			scattered_dir = sdir;
			attenuation   = unboundedSpectrum(atten);
			scattered     = true;
		} else {
			scattered = false;
		}
		is_specular = true;
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
		float  r0 = (1.0f - mat.ior) / (1.0f + mat.ior);
		r0 = r0 * r0;
		float schlick = r0 + (1.0f - r0) * powf(1.0f - cos_t, 5.0f);
		bool is_transmission;
		if (cannot_refract || schlick > wf_rand(seed)) {
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

		// Real-time path guiding (Live Preview only, gpu/optix/
		// wavefront_guiding.h) - only for non-EffectivelySmooth (genuinely
		// glossy, not near-mirror) surfaces, matching the existing NEE/MIS
		// gate below exactly (an EffectivelySmooth conductor stays specular
		// either way, guiding or not).
		float c_pGuide = 0.0f;
		int c_probeIdx = -1;
		if (guidingHistograms != nullptr && !c_dist.EffectivelySmooth()) {
			c_probeIdx = wf_guiding_nearest_probe(guidingGridMeta, hit_point);
			if (c_probeIdx >= 0) {
				c_pGuide = wf_guiding_probability(guidingHistograms[c_probeIdx].numSamplesEverAdded);
			}
		}

		float cwm_x, cwm_y, cwm_z, cwo_x, cwo_y, cwo_z;
		if (c_pGuide > 0.0f && wf_rand(seed) < c_pGuide) {
			// Guided branch: draw wo from the nearest probe's own directional
			// histogram instead of Sample_wm(), then reconstruct wm - see
			// wavefront_guiding.h's own header comment for the full
			// mixture-pdf rationale.
			const float3 c_wo_world = wf_guided_sample_direction(
				guidingHistograms[c_probeIdx], wf_rand(seed), wf_rand(seed), wf_rand(seed));
			cwo_x = dot(c_wo_world, ctan); cwo_y = dot(c_wo_world, cbitan); cwo_z = dot(c_wo_world, cn);
			if (cwo_z <= 0.0f) { scattered = false; break; }
			const float wmx = cwi_x + cwo_x, wmy = cwi_y + cwo_y, wmz = cwi_z + cwo_z;
			const float wmlen = sqrtf(wmx*wmx + wmy*wmy + wmz*wmz);
			if (wmlen < 1e-8f) { scattered = false; break; }
			cwm_x = wmx / wmlen; cwm_y = wmy / wmlen; cwm_z = wmz / wmlen;
		} else {
			c_dist.Sample_wm(cwi_x, cwi_y, cwi_z, wf_rand(seed), wf_rand(seed), cwm_x, cwm_y, cwm_z);
			const float c_dot0 = cwi_x*cwm_x + cwi_y*cwm_y + cwi_z*cwm_z;
			cwo_x = 2.0f*c_dot0*cwm_x - cwi_x;
			cwo_y = 2.0f*c_dot0*cwm_y - cwi_y;
			cwo_z = 2.0f*c_dot0*cwm_z - cwi_z;
			if (cwo_z <= 0.0f) { scattered = false; break; }
		}
		float c_dot = cwi_x*cwm_x + cwi_y*cwm_y + cwi_z*cwm_z;
		float c_G1_wi  = c_dist.G1(cwi_x, cwi_y, cwi_z);
		float c_G_wowi = c_dist.G(cwo_x, cwo_y, cwo_z, cwi_x, cwi_y, cwi_z);
		// f(wi,wo)*cos(wo)/pdf_bsdf(wo) == G(wo,wi)/G1(wi) for a VNDF-sampled
		// microfacet BRDF, at ANY wo paired with wm=normalize(wi+wo) - not
		// only ones Sample_wm() itself produced (already relied on by
		// evalGlossyF's own NEE evaluation above). This is what lets guiding
		// substitute the pdf below transparently, with no separate f()
		// evaluation in either branch.
		float c_weight = (c_G1_wi > 1e-8f) ? c_G_wowi / c_G1_wi : 0.0f;
		float3 c_F = FrConductorRGB(c_dot, mat.eta_c.x, mat.eta_c.y, mat.eta_c.z, mat.k_c.x, mat.k_c.y, mat.k_c.z);
		scattered_dir = normalize(cwo_x*ctan + cwo_y*cbitan + cwo_z*cn);
		scattered   = true;

		// Real NEE/MIS for glossy (non-EffectivelySmooth) conductors, via
		// wf_finish_material_scatter's shared evalGlossyF (see its own
		// comment). brdf_pdf_override uses the GGX-reflection VNDF pdf
		// (ggx_vndf_reflection_pdf, src/shared/bxdfs_conductor.h) at the
		// sampled direction - Conductor is reflection-only to begin with,
		// so this is the exact (not proxy) pdf, unlike CoatedDiffuse/
		// CoatedConductor below. phaseWo carries cwi (already `-ray_dir`,
		// matching evalGlossyF's `wi_world` convention).
		if (!c_dist.EffectivelySmooth()) {
			is_specular = false;
			const float c_pdf_bsdf = ggx_vndf_reflection_pdf(cwi_x, cwi_y, cwi_z, cwo_x, cwo_y, cwo_z, c_alpha_x, c_alpha_y);
			float c_pdf_final = c_pdf_bsdf;
			// Rescale f*cos/pdf_bsdf (already in c_weight) to f*cos/pdf_mixture -
			// the only new arithmetic path guiding needs here (see
			// wavefront_guiding.h's own header comment). A no-op when
			// c_pGuide==0 (guiding off, or this probe's histogram still
			// unpopulated).
			if (c_pGuide > 0.0f && c_probeIdx >= 0) {
				const float c_pdf_guide = wf_guided_pdf(guidingHistograms[c_probeIdx], scattered_dir);
				const float c_pdf_mixture = c_pGuide * c_pdf_guide + (1.0f - c_pGuide) * c_pdf_bsdf;
				if (c_pdf_mixture > 1e-8f) {
					c_weight *= c_pdf_bsdf / c_pdf_mixture;
					c_pdf_final = c_pdf_mixture;
				}
			}
			brdf_pdf_override = c_pdf_final;
			phaseWo = cwi;
		} else {
			is_specular = true;
		}
		// Use average Fresnel weight as scalar (conductor is specular, color from albedo)
		attenuation = albedoSpectrum(make_float3(c_F.x * c_weight, c_F.y * c_weight, c_F.z * c_weight));
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

		// Real-time path guiding (Live Preview only) - see
		// MaterialType::Conductor's identical-shape block above for the full
		// rationale; RoughMetal has no Fresnel color of its own, everything
		// else about the guiding gate/branch is identical.
		float rm_pGuide = 0.0f;
		int rm_probeIdx = -1;
		if (guidingHistograms != nullptr && !rm_dist.EffectivelySmooth()) {
			rm_probeIdx = wf_guiding_nearest_probe(guidingGridMeta, hit_point);
			if (rm_probeIdx >= 0) {
				rm_pGuide = wf_guiding_probability(guidingHistograms[rm_probeIdx].numSamplesEverAdded);
			}
		}

		float rmwm_x, rmwm_y, rmwm_z, rmwo_x, rmwo_y, rmwo_z;
		if (rm_pGuide > 0.0f && wf_rand(seed) < rm_pGuide) {
			const float3 rm_wo_world = wf_guided_sample_direction(
				guidingHistograms[rm_probeIdx], wf_rand(seed), wf_rand(seed), wf_rand(seed));
			rmwo_x = dot(rm_wo_world, rmtan); rmwo_y = dot(rm_wo_world, rmbitan); rmwo_z = dot(rm_wo_world, rmn);
			if (rmwo_z <= 0.0f) { scattered = false; break; }
			const float wmx = rmwi_x + rmwo_x, wmy = rmwi_y + rmwo_y, wmz = rmwi_z + rmwo_z;
			const float wmlen = sqrtf(wmx*wmx + wmy*wmy + wmz*wmz);
			if (wmlen < 1e-8f) { scattered = false; break; }
			rmwm_x = wmx / wmlen; rmwm_y = wmy / wmlen; rmwm_z = wmz / wmlen;
		} else {
			rm_dist.Sample_wm(rmwi_x, rmwi_y, rmwi_z, wf_rand(seed), wf_rand(seed), rmwm_x, rmwm_y, rmwm_z);
			const float rm_dot0 = rmwi_x*rmwm_x + rmwi_y*rmwm_y + rmwi_z*rmwm_z;
			rmwo_x = 2.0f*rm_dot0*rmwm_x - rmwi_x;
			rmwo_y = 2.0f*rm_dot0*rmwm_y - rmwi_y;
			rmwo_z = 2.0f*rm_dot0*rmwm_z - rmwi_z;
			if (rmwo_z <= 0.0f) { scattered = false; break; }
		}
		float rm_G1_wi  = rm_dist.G1(rmwi_x, rmwi_y, rmwi_z);
		float rm_G_wowi = rm_dist.G(rmwo_x, rmwo_y, rmwo_z, rmwi_x, rmwi_y, rmwi_z);
		float rm_weight = (rm_G1_wi > 1e-8f) ? rm_G_wowi / rm_G1_wi : 0.0f;
		scattered_dir = normalize(rmwo_x*rmtan + rmwo_y*rmbitan + rmwo_z*rmn);
		scattered   = true;

		// Real NEE/MIS for glossy (non-EffectivelySmooth) rough metal, via
		// wf_finish_material_scatter's shared evalGlossyF - see
		// MaterialType::Conductor's identical-shape block above.
		if (!rm_dist.EffectivelySmooth()) {
			is_specular = false;
			const float rm_pdf_bsdf = ggx_vndf_reflection_pdf(rmwi_x, rmwi_y, rmwi_z, rmwo_x, rmwo_y, rmwo_z, rm_alpha, rm_alpha);
			float rm_pdf_final = rm_pdf_bsdf;
			if (rm_pGuide > 0.0f && rm_probeIdx >= 0) {
				const float rm_pdf_guide = wf_guided_pdf(guidingHistograms[rm_probeIdx], scattered_dir);
				const float rm_pdf_mixture = rm_pGuide * rm_pdf_guide + (1.0f - rm_pGuide) * rm_pdf_bsdf;
				if (rm_pdf_mixture > 1e-8f) {
					rm_weight *= rm_pdf_bsdf / rm_pdf_mixture;
					rm_pdf_final = rm_pdf_mixture;
				}
			}
			brdf_pdf_override = rm_pdf_final;
			phaseWo = rmwi;
		} else {
			is_specular = true;
		}
		attenuation = albedoSpectrum(make_float3(mat.albedo.x * rm_weight, mat.albedo.y * rm_weight, mat.albedo.z * rm_weight));
		break;
	}
	case MaterialType::CoatedDiffuse: {
		// "reflectance" bound to a real Texture (pbrt's own ganesha/
		// barcelona-pavilion "texture reflectance" - see
		// pbrt_flatten::Material::textureFilename's own comment) instead of
		// mat.albedo's flat colour when textureIdx>=0 - same pattern the
		// Lambertian case above already uses, scaled by mat.emissionScale
		// (reused here for CoatedDiffuse's own "scale"-wrapped-imagemap
		// case - see that field's own comment in optix_types.h).
		// wf_finish_material_scatter's own evalGlossyF (its CoatedDiffuse
		// branch) does the identical lookup for its NEE/MIS f() evaluation
		// too, via the uv_u/uv_v this function now threads through to it -
		// see that function's own uv_u/uv_v parameter comment.
		const float3 cd_albedo = (mat.textureIdx >= 0)
			? wf_sample_texture(textures, texturePixels, mat.textureIdx, h.uv_u, h.uv_v, hit_point) * mat.emissionScale
			: mat.albedo;
		float cd_alpha_x = wf_glossy_alpha(mat, do_regularize);
		float cd_alpha_y = wf_glossy_alpha_v(mat, do_regularize);
		glossyAlphaForNEE = cd_alpha_x;
		glossyAlphaVForNEE = cd_alpha_y;
		float3 cdn = normal;
		float3 cdtan, cdbitan;
		BuildDpduTangentFrame(cdn.x, cdn.y, cdn.z, h.objDpdu.x, h.objDpdu.y, h.objDpdu.z,
		                       cdtan.x, cdtan.y, cdtan.z, cdbitan.x, cdbitan.y, cdbitan.z);
		float3 cdwi = -normalize(h.rayDir);
		float cdwi_x = dot(cdwi, cdtan), cdwi_y = dot(cdwi, cdbitan), cdwi_z = dot(cdwi, cdn);
		// Grazing/back-facing incoming ray: no valid local frame to sample
		// against - matches optix_device_helpers.h's shade_material()
		// CoatedDiffuse case (`if (cdwi_z <= 0.0f) { scattered = false; }`),
		// missing here let a grazing-angle ray fall through to Sample_wm()
		// with a degenerate/negative-z local direction instead of
		// terminating the path like the recursive backend does.
		if (cdwi_z <= 0.0f) { scattered = false; break; }
		TrowbridgeReitz<float> cd_dist(cd_alpha_x, cd_alpha_y);
		float cdwm_x, cdwm_y, cdwm_z;
		cd_dist.Sample_wm(cdwi_x, cdwi_y, cdwi_z, wf_rand(seed), wf_rand(seed), cdwm_x, cdwm_y, cdwm_z);
		float cd_dot = cdwi_x*cdwm_x + cdwi_y*cdwm_y + cdwi_z*cdwm_z;
		float Fr = FrDielectric(fmaxf(cd_dot, 0.0f), mat.ior);
		if (wf_rand(seed) < Fr) {
			float cdwo_x = 2.0f*cd_dot*cdwm_x - cdwi_x;
			float cdwo_y = 2.0f*cd_dot*cdwm_y - cdwi_y;
			float cdwo_z = 2.0f*cd_dot*cdwm_z - cdwi_z;
			scattered_dir = normalize(cdwo_x*cdtan + cdwo_y*cdbitan + cdwo_z*cdn);
			attenuation   = SS(1.f);
		} else {
			// Multi-bounce escape through the coat, matching
			// optix_device_helpers.h's shade_material() CoatedDiffuse case
			// (see its own comment for why this replaced a single Lambertian
			// bounce + one exit attempt: giving up immediately on a failed
			// exit test discards that sample's energy entirely, rendering
			// far too dark, rather than retrying like the real layered
			// material's random walk does). is_specular is decided once,
			// uniformly, after this if/else (see below) rather than per-
			// branch now that glossy coats get real NEE via CoatedDiffuseBxDF
			// ::f() - pbrt-v4's LayeredBxDF has no closed-form f(wo,wi)
			// either, but the shared BxDF template's own stochastic estimator
			// (src/shared/bxdfs_layered.h, already verified on CPU for #228
			// and the recursive backend for #229) gives us one anyway.
			constexpr int kMaxCoatBounces = 8;
			float3 beta = make_float3(1.0f - Fr, 1.0f - Fr, 1.0f - Fr);
			float3 diff_dir = cdn;
			bool escaped = false;
			for (int cb = 0; cb < kMaxCoatBounces; ++cb) {
				diff_dir = cdn + wf_rand_unit(seed);
				if (wf_near_zero(diff_dir)) diff_dir = cdn;
				diff_dir = normalize(diff_dir);
				beta.x *= cd_albedo.x; beta.y *= cd_albedo.y; beta.z *= cd_albedo.z;

				float dw_x = dot(diff_dir, cdtan), dw_y = dot(diff_dir, cdbitan), dw_z = dot(diff_dir, cdn);
				float dwm_x, dwm_y, dwm_z;
				cd_dist.Sample_wm(dw_x, dw_y, dw_z, wf_rand(seed), wf_rand(seed), dwm_x, dwm_y, dwm_z);
				float cos_out = dw_x*dwm_x + dw_y*dwm_y + dw_z*dwm_z;
				float F_out = FrDielectric(cos_out, 1.0f / mat.ior);
				if (wf_rand(seed) < F_out) continue;  // TIR: bounce again
				beta.x *= (1.0f - F_out); beta.y *= (1.0f - F_out); beta.z *= (1.0f - F_out);
				escaped = true;
				break;
			}
			if (!escaped) { scattered = false; break; }
			attenuation   = albedoSpectrum(beta);
			scattered_dir = diff_dir;
		}
		scattered = (dot(scattered_dir, normal) > 0.0f);

		// Real NEE/MIS for glossy (non-EffectivelySmooth) coats, via
		// wf_finish_material_scatter's shared evalGlossyF (see its own
		// comment and MaterialType::Conductor's identical-shape block
		// above). brdf_pdf_override uses the coat's own top-surface GGX-
		// reflection VNDF pdf as a proxy for the walk's true (unknown, no
		// closed form) density - matches CPU's coated_diffuse::scatter()
		// (material_pbrt.h), which uses the same ggx_reflection_pdf proxy
		// for its own srec.pdf_ptr. phaseWo carries cdwi (already
		// `-ray_dir`, matching evalGlossyF's `wi_world` convention).
		if (scattered && !cd_dist.EffectivelySmooth()) {
			is_specular = false;
			float swo_x = dot(scattered_dir, cdtan), swo_y = dot(scattered_dir, cdbitan), swo_z = dot(scattered_dir, cdn);
			brdf_pdf_override = (swo_z > 0.0f)
				? ggx_vndf_reflection_pdf(cdwi_x, cdwi_y, cdwi_z, swo_x, swo_y, swo_z, cd_alpha_x, cd_alpha_y)
				: 0.0f;
			phaseWo = cdwi;
		} else {
			is_specular = true;
		}
		break;
	}
	case MaterialType::ThinDielectric: {
		float3 V = -normalize(h.rayDir);
		float cos_i = fabsf(dot(V, normal));
		float Fr = FrDielectric(cos_i, mat.ior);
		// Account for double transmission (glass slab): T^2
		float T2 = (1.0f - Fr) * (1.0f - Fr);
		if (wf_rand(seed) < Fr / (Fr + T2)) {
			scattered_dir = wf_reflect(-V, normal);
		} else {
			scattered_dir = normalize(h.rayDir); // straight through
		}
		attenuation = SS(1.f);
		scattered   = true;
		is_specular = true;
		break;
	}
	case MaterialType::CoatedConductor: {
		// Rough dielectric coat over GGX conductor (pbrt-v4 CoatedConductorBxDF).
		// Matches optix_intersection_sphere.h's recursive-path handling: path B
		// (transmit into the coat, bounce off the conductor, exit back through
		// the coat) must weight the conductor's Fresnel by T_in*T_out (how
		// much light actually gets through the dielectric coat both ways) -
		// the previous version here resampled a microfacet from the ORIGINAL
		// viewing direction and skipped both the refraction-into-the-coat
		// step and the T_in/T_out weighting entirely, making the conductor
		// visible at full strength as if the coat weren't there.
		float cc_alpha_x = wf_glossy_alpha(mat, do_regularize);
		float cc_alpha_y = wf_glossy_alpha_v(mat, do_regularize);
		glossyAlphaForNEE = cc_alpha_x;
		glossyAlphaVForNEE = cc_alpha_y;
		float3 ccn = normal;
		float3 cctan, ccbitan;
		BuildDpduTangentFrame(ccn.x, ccn.y, ccn.z, h.objDpdu.x, h.objDpdu.y, h.objDpdu.z,
		                       cctan.x, cctan.y, cctan.z, ccbitan.x, ccbitan.y, ccbitan.z);
		float3 ccwi = -normalize(h.rayDir);
		float ccwi_x = dot(ccwi, cctan), ccwi_y = dot(ccwi, ccbitan), ccwi_z = dot(ccwi, ccn);
		if (ccwi_z <= 0.0f) { scattered = false; break; }
		TrowbridgeReitz<float> cc_dist(cc_alpha_x, cc_alpha_y);
		float ccwm_x, ccwm_y, ccwm_z;
		cc_dist.Sample_wm(ccwi_x, ccwi_y, ccwi_z, wf_rand(seed), wf_rand(seed), ccwm_x, ccwm_y, ccwm_z);
		float cc_dot = ccwi_x*ccwm_x + ccwi_y*ccwm_y + ccwi_z*ccwm_z;
		float F_in = FrDielectric(fmaxf(cc_dot, 0.0f), mat.ior);
		float3 ccwo;
		if (wf_rand(seed) < F_in) {
			// Path A: coat specular reflection
			float wo_x = 2.0f*cc_dot*ccwm_x - ccwi_x;
			float wo_y = 2.0f*cc_dot*ccwm_y - ccwi_y;
			float wo_z = 2.0f*cc_dot*ccwm_z - ccwi_z;
			if (wo_z <= 0.0f) { scattered = false; break; }
			float G1 = cc_dist.G1(ccwi_x, ccwi_y, ccwi_z);
			float G  = cc_dist.G(wo_x, wo_y, wo_z, ccwi_x, ccwi_y, ccwi_z);
			float w  = (G1 > 1e-8f) ? G / G1 : 0.0f;
			float fv = F_in * w;
			attenuation = SS(fv);
			ccwo = make_float3(wo_x, wo_y, wo_z);
		} else {
			// Path B: transmit into layer -> conductor bounce -> exit coat
			float w_x = 2.0f*cc_dot*ccwm_x - ccwi_x;
			float w_y = 2.0f*cc_dot*ccwm_y - ccwi_y;
			float w_z = 2.0f*cc_dot*ccwm_z - ccwi_z;
			if (w_z > 0.0f) w_z = -w_z;   // ensure pointing downward into layer
			if (w_z == 0.0f) { scattered = false; break; }

			// Flip to conductor frame: "incoming from above" (fw_z > 0)
			float fw_x = -w_x, fw_y = -w_y, fw_z = -w_z;

			float bwm_x, bwm_y, bwm_z;
			cc_dist.Sample_wm(fw_x, fw_y, fw_z, wf_rand(seed), wf_rand(seed), bwm_x, bwm_y, bwm_z);
			float cos_c = fw_x*bwm_x + fw_y*bwm_y + fw_z*bwm_z;
			if (cos_c <= 0.0f) { scattered = false; break; }

			float rwo_x = 2.0f*cos_c*bwm_x - fw_x;
			float rwo_y = 2.0f*cos_c*bwm_y - fw_y;
			float rwo_z = 2.0f*cos_c*bwm_z - fw_z;
			if (rwo_z <= 0.0f) { scattered = false; break; }

			float G1_c = cc_dist.G1(fw_x, fw_y, fw_z);
			float G_c  = cc_dist.G(rwo_x, rwo_y, rwo_z, fw_x, fw_y, fw_z);
			float wt_c = (G1_c > 1e-8f) ? G_c / G1_c : 0.0f;

			float3 c_F  = FrConductorRGB(cos_c, mat.eta_c.x, mat.eta_c.y, mat.eta_c.z, mat.k_c.x, mat.k_c.y, mat.k_c.z);

			float F_out = FrDielectric(rwo_z, 1.0f / mat.ior);  // inside -> outside
			float T_out = 1.0f - F_out;
			float T_in  = 1.0f - F_in;

			attenuation = albedoSpectrum(make_float3(
				c_F.x * wt_c * T_in * T_out,
				c_F.y * wt_c * T_in * T_out,
				c_F.z * wt_c * T_in * T_out));
			ccwo = make_float3(rwo_x, rwo_y, rwo_z);
		}
		scattered_dir = normalize(ccwo.x*cctan + ccwo.y*ccbitan + ccwo.z*ccn);
		scattered   = (dot(scattered_dir, normal) > 0.0f);

		// Real NEE/MIS for glossy (non-EffectivelySmooth) coats - see
		// MaterialType::CoatedDiffuse's identical-shape block above for the
		// full rationale (shared CoatedConductorBxDF<float>::f() stochastic
		// estimator, GGX top-surface VNDF pdf as the MIS proxy).
		if (scattered && !cc_dist.EffectivelySmooth()) {
			is_specular = false;
			float swo_x = dot(scattered_dir, cctan), swo_y = dot(scattered_dir, ccbitan), swo_z = dot(scattered_dir, ccn);
			brdf_pdf_override = (swo_z > 0.0f)
				? ggx_vndf_reflection_pdf(ccwi_x, ccwi_y, ccwi_z, swo_x, swo_y, swo_z, cc_alpha_x, cc_alpha_y)
				: 0.0f;
			phaseWo = ccwi;
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
		if (wf_rand(seed) < pr / (pr + pt)) {
			scattered_dir = normalize(normal + wf_rand_unit(seed));
			if (wf_near_zero(scattered_dir)) scattered_dir = normal;
			attenuation = albedoSpectrum(R);
		} else {
			float3 neg_n = -normal;
			scattered_dir = normalize(neg_n + wf_rand_unit(seed));
			if (wf_near_zero(scattered_dir)) scattered_dir = neg_n;
			attenuation = albedoSpectrum(T_col);
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
		float free_path = (sigma_t > 1e-8f) ? (-logf(fmaxf(1e-8f, 1.0f - wf_rand(seed))) / sigma_t) : 1e30f;
		float3 unit_dir = normalize(h.rayDir);
		if (free_path < dist_inside) {
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
			attenuation   = albedoSpectrum(mat.albedo);
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
			if (mat.medium_emission.x > 0.0f || mat.medium_emission.y > 0.0f || mat.medium_emission.z > 0.0f)
				radiance = radiance + throughput * wf_lift_rgb_to_spectrum(mat.medium_emission, swl, /*isIlluminant=*/true);
		} else {
			hit_point     = h.rayOrigin + t_far * unit_dir;
			scattered_dir = unit_dir;  // straight through, no interaction
			attenuation   = SS(1.f);
			is_specular   = true;  // no interaction - a free/non-scattering pass-through
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
			const int voxelCount = grid.nx * grid.ny * grid.nz;
			const float* rData = rgbGridData + grid.dataOffset;
			const float* gData = rData + voxelCount;
			const float* bData = gData + voxelCount;
			for (int iter = 0; iter < 128 && !did_scatter; ++iter) {
				float dt = -logf(fmaxf(1e-8f, 1.0f - wf_rand(seed))) / grid.sigma_maj;
				tt += dt;
				if (tt >= segMax) break;
				float px = mox + tt*mdx, py = moy + tt*mdy, pz = moz + tt*mdz;
				float dr = gpu_rgb_grid_trilinear(rData, grid.nx, grid.ny, grid.nz, px, py, pz);
				float dg = gpu_rgb_grid_trilinear(gData, grid.nx, grid.ny, grid.nz, px, py, pz);
				float db = gpu_rgb_grid_trilinear(bData, grid.nx, grid.ny, grid.nz, px, py, pz);
				float sr = dr * grid.sigma_scale, sg = dg * grid.sigma_scale, sb = db * grid.sigma_scale;
				float sigma_t_local = fmaxf(sr, fmaxf(sg, sb));
				if (wf_rand(seed) < sigma_t_local / grid.sigma_maj) {
					did_scatter = true;
					medium_t      = tt;
					// Real NEE+MIS at the phase-function scatter event - see
					// MaterialType::Medium's identical fix above.
					scattered_dir = wf_sample_phase_scatter(unit_dir, grid.phase_g, seed, phaseWo, phaseG, brdf_pdf_override);
					float maxc = fmaxf(sr, fmaxf(sg, fmaxf(sb, 1e-6f)));
					attenuation = albedoSpectrum(make_float3(sr/maxc, sg/maxc, sb/maxc));
					// Real per-voxel "rgb Le" - emits the FULL Le on every
					// accepted collision (weight 1), not the sigma_a/sigma_t-
					// weighted fraction CPU's RGBGridMediumData::sample_point()
					// uses, unlike MaterialType::Medium's own build-time-baked
					// mat.medium_emission above (an exact weighted constant,
					// since GPU has no sigma_a grid here to weight by at all -
					// see GpuRgbGridMedium::leDataOffset's own comment,
					// optix_types.h, for the full "why" and its brightness-
					// vs-CPU consequence). Added directly to radiance here
					// (this backend's own per-case convention, see
					// MaterialType::Medium's identical `radiance = radiance +
					// throughput * ...` above) rather than threaded through to
					// a later NEE step, since wavefront's NEE/shadow-ray pass
					// for this hit queue happens in a separate kernel
					// (wf_finish_material_scatter) that has no per-voxel grid
					// data to re-sample this point from.
					if (grid.leDataOffset >= 0) {
						const float* leRData = rgbGridData + grid.leDataOffset;
						const float* leGData = leRData + voxelCount;
						const float* leBData = leGData + voxelCount;
						float ler = gpu_rgb_grid_trilinear(leRData, grid.nx, grid.ny, grid.nz, px, py, pz);
						float leg = gpu_rgb_grid_trilinear(leGData, grid.nx, grid.ny, grid.nz, px, py, pz);
						float leb = gpu_rgb_grid_trilinear(leBData, grid.nx, grid.ny, grid.nz, px, py, pz);
						float3 selfEmission = make_float3(ler, leg, leb) * grid.Le_scale;
						if (selfEmission.x > 0.0f || selfEmission.y > 0.0f || selfEmission.z > 0.0f)
							radiance = radiance + throughput * wf_lift_rgb_to_spectrum(selfEmission, swl, /*isIlluminant=*/true);
					}
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
		if (h.frontFace) {
			attenuation   = SS(1.f);
			scattered_dir = wf_dielectric_scatter(h.rayDir, normal, true, mat.ior, seed);
			is_specular   = true;  // genuinely specular (Dirac-delta) dielectric bounce
			// pbrt-v4 etaScale (entry surface) - see MaterialType::
			// Dielectric's identical eta computation above.
			if (dot(scattered_dir, normal) < 0.0f) eventEta = 1.0f / mat.ior;
		} else {
			float t_near = h.t;
			float t_far  = h.mediumTFar;
			float dist_inside = fmaxf(0.0f, t_far - t_near);
			float sigma_t = mat.eta_c.x;  // dielectric_medium_extra.sigma_t
			float free_path = (sigma_t > 1e-8f) ? (-logf(fmaxf(1e-8f, 1.0f - wf_rand(seed))) / sigma_t) : 1e30f;
			float3 unit_dir = normalize(h.rayDir);
			if (free_path < dist_inside) {
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
				attenuation   = albedoSpectrum(mat.albedo);
				is_specular = false;
			} else {
				hit_point     = h.rayOrigin + t_far * unit_dir;
				attenuation   = SS(1.f);
				scattered_dir = wf_dielectric_scatter(h.rayDir, normal, false, mat.ior, seed);
				is_specular   = true;  // genuinely specular exit refraction/reflection
				// pbrt-v4 etaScale (exit surface, front_face is false here) -
				// see MaterialType::Dielectric's identical eta computation
				// above.
				if (dot(scattered_dir, normal) < 0.0f) eventEta = mat.ior;
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
		// already a fixed `for (depth = 0; depth < max_depth; ++depth)`
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
	wf_finish_material_scatter(mat.type, matEta, h.materialIdx, (bool)h.any_nonspecular, glossyAlphaForNEE, glossyAlphaVForNEE, h.etaScale * eventEta * eventEta, h.filterWeight, maxComponentValue, normal, hit_point, h.objDpdu, seed,
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
		lightBvh);
}

