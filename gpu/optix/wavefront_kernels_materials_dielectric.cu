// wavefront_kernels_materials_dielectric.cu
// CUDA compute kernel: evaluate_materials_dielectric, the Dielectric/
// RoughDielectric-only twin of evaluate_materials (see this file's own
// header comment on why it exists as a separate kernel). Split out of the
// original wavefront_kernels.cu for the same independent-recompilation
// reason as wavefront_kernels_materials.cu - see
// wavefront_device_helpers.h's own header comment for where the two shared
// helpers now live.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "wavefront_device_helpers.h"

// ============================================================================
// evaluate_materials_dielectric
//
// Third twin of evaluate_materials(), scoped to MaterialType::Dielectric and
// MaterialType::RoughDielectric - the two material types routed into
// dielectricHitQueue at push time (see wavefront_programs.cu's
// __raygen__wf_intersect), for the same register-pressure reason as
// evaluate_materials_simple() above (see its own comment). The Dielectric/
// RoughDielectric case bodies below are copied verbatim from
// evaluate_materials()'s switch, not reimplemented. Same trap-on-unexpected-
// type guard as evaluate_materials_simple(), for the same reason (the push-
// site routing condition living in a different file than this kernel).
//
// No textures/texturePixels params - neither Dielectric nor RoughDielectric
// ever reads mat.textureIdx (grepped: no call site exists), so they're
// dropped from this kernel's signature entirely rather than carried as dead
// parameters, unlike evaluate_materials_simple() which does need them for
// Lambertian's textured-albedo case.
// ============================================================================
extern "C" __global__ void evaluate_materials_dielectric(
	WorkQueue<HitWorkItem>       hitQueue,
	int                          numHits,
	WorkQueue<RayWorkItem>       nextRayQueue,
	WorkQueue<ShadowRayWorkItem> shadowQueue,
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
	// Needed only for a textured (pbrt AreaLightSource "filename") NEE
	// target - see wf_finish_material_scatter()'s own comment.
	const TextureData* textures,
	const unsigned char* texturePixels,
	int maxDepth,
	float3 skyColor,
	float shadowRayEpsilon,
	GpuSkyDistribution skyDist,
	GpuPortalLight portalLight,
	// Integrator "bool regularize" - see evaluate_materials()'s own
	// identical parameter comment above.
	bool regularize,
	// "float maxcomponentvalue" firefly clamp - see
	// GpuCameraParams::maxComponentValue's own comment (optix_types.h).
	float maxComponentValue,
	// Denoiser guide-layer AOVs - see evaluate_materials()'s own comment.
	float3* albedoBuffer,
	float3* normalBuffer,
	float4* worldPosBuffer,
	// ReSTIR DI (Live Preview only) - see wf_finish_material_scatter's own
	// restirReservoirs parameter comment. nullptr for batch/offline rendering.
	GpuReservoir* restirReservoirs,
	// See wf_finish_material_scatter's own restirCtx parameter comment.
	GpuRestirTemporalContext restirCtx,
	// ReSTIR GI (Live Preview only) - see wf_finish_material_scatter's own
	// giOriginContext/giCandidateOut parameter comments. GI's MVP-scope
	// Lambertian-only restriction applies to x0 (the ORIGINATING primary
	// hit), not to whatever x1 turns out to be - a Lambertian x0's
	// continuation ray can perfectly well land on a dielectric/RoughDielectric
	// surface at depth 1, and THIS kernel is what processes that hit, so it
	// needs the same depth==1 redirect wiring every other evaluate_materials*
	// variant has (a specular x1 here simply never reaches the `!is_specular`
	// NEE gate, so it naturally contributes nothing - not a special case to
	// handle here).
	GpuGiOriginContext* giOriginContext,
	GpuGiSample* giCandidateOut,
	// See wf_light_bvh_sample_index()'s own comment (wavefront_restir_
	// helpers.h). lightBvh.nodeCount<=0 (the default) means "no light BVH
	// built" - forwarded to wf_finish_material_scatter() below unchanged,
	// which itself falls straight through to the alias table for that case.
	// Previously always defaulted here (this kernel was never wired up when
	// the light BVH first shipped for Lambertian/Metal), silently keeping
	// Dielectric/RoughDielectric NEE on the alias table even in scenes where
	// every other material type already uses the spatial+power selection.
	WfLightBvhContext lightBvh = {}
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numHits) return;

	const HitWorkItem& h = hitQueue.items[idx];
	const MaterialData& mat = materials[h.materialIdx];

	if (albedoBuffer && h.depth == 0) {
		wf_accumulate_aov(albedoBuffer, normalBuffer, h.pixelIndex, mat.albedo, h.normal);
	}
	if (worldPosBuffer && h.depth == 0) {
		wf_write_world_pos(worldPosBuffer, h.pixelIndex, h.hitPoint);
	}

	// See evaluate_materials()'s own identical hoist just above its `h` load -
	// same reasoning: both operands are fixed for the rest of this thread.
	const bool do_regularize = regularize && (bool)h.any_nonspecular;

	float3 normal    = h.normal;
	float3 hit_point = h.hitPoint;
	unsigned int seed = h.seed;
	const float shadow_eps = (shadowRayEpsilon > 0.0f) ? shadowRayEpsilon : 0.01f;

	using SS  = SampledSpectrum<kWFNWavelengths>;
	using SWL = SampledWavelengths<kWFNWavelengths>;
	SS throughput(h.throughput);
	SS radiance(h.radiance);
	SWL swl;
	for (int i = 0; i < kWFNWavelengths; ++i) {
		swl.lambda[i] = h.wavelengths[i];
		swl.pdf[i]    = h.wavelength_pdfs[i];
	}

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

	if (h.depth >= maxDepth) {
		addToFramebuffer(h.pixelIndex, radiance * h.filterWeight);
		return;
	}

	float3 scattered_dir    = make_float3(0, 0, 0);
	SS     attenuation(0.f);
	bool   scattered        = false;
	bool   is_specular      = false;
	float  brdf_pdf_override = -1.0f;
	float3 phaseWo = make_float3(0.0f, 0.0f, 0.0f);
	float matEta = mat.ior;
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
	// pbrt-v4 etaScale term for THIS event only - see RayWorkItem::
	// etaScale's own comment. 1.0f (a no-op) unless a case below sets it on
	// a genuine transmission; combined with h.etaScale at the call site.
	float eventEta = 1.0f;

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

	// Dielectric/RoughDielectric only: this hit's material ior, unless
	// MaterialData::dispersive_extra marks it dispersive (cauchy_A > 0), in
	// which case the flat .ior is replaced by the Cauchy-formula ior at this
	// path's hero wavelength (swl.lambda[0]) - CauchyEta() is already
	// CPU_GPU-tagged (src/shared/fresnel.h), no device-side port needed,
	// same formula CPU's already-shipped dielectric/rough_dielectric
	// dispersion uses. See MaterialData::dispersive_extra's own comment
	// (optix_types.h) for the sentinel convention.
	const bool matIsDispersive = mat.dispersive_extra.cauchy_A > 0.0f;
	const float dispersiveIor = matIsDispersive
		? CauchyEta(swl.lambda[0], mat.dispersive_extra.cauchy_A, mat.dispersive_extra.cauchy_B)
		: mat.ior;

	switch (mat.type) {
	case MaterialType::Dielectric: {
		// h.frontFace, NOT dot(h.rayDir, normal) - `normal` here is already
		// flipped to face the incoming ray (see __closesthit__wf_quad/
		// __closesthit__wf_sphere's own "Flip to face the ray" comment), so
		// dot(h.rayDir, normal) is always negative BY CONSTRUCTION regardless
		// of which side was actually hit - re-deriving front/back from it
		// silently always took the "entering" branch, using 1/ior even when
		// exiting the glass. h.frontFace is the pre-flip boolean this needs,
		// same fix the RoughDielectric case below already gets right via its
		// own rd_front_face. Bug found while verifying B23 (Glass Prism
		// Dispersion) on wavefront: harmless-looking for a single-bounce
		// sphere (A1), but compounds badly over the prism's many internal
		// TIR bounces, making the whole solid render far too transparent.
		bool front_face = h.frontFace != 0;
		float eta = front_face ? (1.0f / dispersiveIor) : dispersiveIor;
		float3 unit_dir = normalize(h.rayDir);
		float  cos_t = fminf(dot(-unit_dir, normal), 1.0f);
		float  sin_t = sqrtf(1.0f - cos_t * cos_t);
		bool   cannot_refract = eta * sin_t > 1.0f;
		float  r0 = (1.0f - dispersiveIor) / (1.0f + dispersiveIor);
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
		attenuation = is_transmission ? albedoSpectrum(mat.transmission_filter) : SS(1.f);
		scattered   = true;
		is_specular = true;
		// pbrt-v4 etaScale - see this function's own eventEta local comment.
		if (is_transmission) eventEta = eta;
		// Collapse the hero-wavelength set to lambda[0] on the one event
		// that's actually wavelength-dependent for smooth Dielectric (the
		// refraction itself - the Fresnel-weighted reflect/refract COIN FLIP
		// above already used the dispersive eta, but a reflected ray leaves
		// this surface with no further per-wavelength distinction to carry).
		// Matches CPU dielectric's own TerminateSecondary() gate exactly
		// (camera.h: `if (srec.is_transmission && disp) swl.TerminateSecondary();`).
		if (matIsDispersive && is_transmission) swl.TerminateSecondary();
		break;
	}
	case MaterialType::RoughDielectric: {
		// mat.textureIdx>=0 means "roughness" was texture-bound (pbrt-v4
		// "texture roughness" on a Dielectric - see MaterialData::
		// textureIdx's own comment) - sample the image's red/x channel as
		// the scalar isotropic roughness at THIS hit instead of the flat
		// mat.fuzz, matching optix_device_helpers.h's identical recursive-
		// backend branch and CPU's rough_dielectric::true_alpha()
		// (material_pbrt.h). do_regularize still applies via
		// RegularizeAlpha, same as the non-textured wf_glossy_alpha() path.
		float rd_alpha_x, rd_alpha_y;
		if (mat.textureIdx >= 0) {
			const float rd_rough = wf_sample_texture(textures, texturePixels, mat.textureIdx, h.uv_u, h.uv_v, hit_point).x;
			const float rd_a = mat.remapRoughness ? sqrtf(rd_rough) : rd_rough;
			rd_alpha_x = rd_alpha_y = do_regularize
				? RegularizeAlpha(rd_a) : rd_a;
		} else {
			rd_alpha_x = wf_glossy_alpha(mat, do_regularize);
			rd_alpha_y = wf_glossy_alpha_v(mat, do_regularize);
		}
		glossyAlphaForNEE = rd_alpha_x;
		glossyAlphaVForNEE = rd_alpha_y;
		bool rd_front_face = h.frontFace != 0;
		float rd_ri = rd_front_face ? (1.0f / dispersiveIor) : dispersiveIor;
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
		float rd_dot = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
		float Fr = FrDielectric(rd_dot, 1.0f / rd_ri);
		float wo_x, wo_y, wo_z;
		if (wf_rand(seed) < Fr) {
			// Reflect
			wo_x = 2.0f*rd_dot*wm_x - wi_x;
			wo_y = 2.0f*rd_dot*wm_y - wi_y;
			wo_z = 2.0f*rd_dot*wm_z - wi_z;
			if (wo_z <= 0.0f) { scattered = false; break; }
			scattered_dir = normalize(wo_x*tan_v + wo_y*bitan + wo_z*n);
		} else {
			// Refract
			float3 wm_world = wm_x*tan_v + wm_y*bitan + wm_z*n;
			float eta = rd_front_face ? (1.0f / dispersiveIor) : dispersiveIor;
			float3 refracted = wf_refract(normalize(h.rayDir), wm_world, eta);
			wo_x = dot(refracted, tan_v); wo_y = dot(refracted, bitan); wo_z = dot(refracted, n);
			scattered_dir = refracted;
			// pbrt-v4 etaScale - see this function's own eventEta local comment.
			eventEta = eta;
		}
		{
			float wo_z_abs = fabsf(wo_z);
			float G2 = rd_dist.G(wi_x, wi_y, wi_z, wo_x, wo_y, wo_z_abs);
			float G1_wi = rd_dist.G1(wi_x, wi_y, wi_z);
			float w = (G1_wi > 1e-8f) ? (G2 / G1_wi) : 0.0f;
			attenuation = SS(w);
		}
		scattered   = true;
		// Unlike smooth Dielectric above, BOTH lobes here are Fresnel/eta-
		// dependent (RoughDielectricBxDF's reflection lobe is not eta-
		// independent the way a perfect mirror's would be), so termination
		// fires on every dispersive hit regardless of which lobe ends up
		// sampled - matches CPU rough_dielectric's own already-shipped rule
		// exactly (material_pbrt.h's scatter(), same reasoning). Deliberately
		// placed AFTER the wo_z<=0 rejection break above (not right after
		// rd_ri is computed) - CPU only ever calls TerminateSecondary() once
		// a scatter event is confirmed (camera.h: `if (!scattered) break;`
		// precedes it), and this function's own `if (!scattered) { ...
		// addToFramebuffer(...); return; }` tail converts already-accumulated
		// `radiance` (weighted under the OLD, un-terminated pdfs from earlier
		// bounces) to XYZ using whatever `swl` state is current - collapsing
		// it before a possible rejection would bias that flush's per-channel
		// weights (one channel ~4x too high, three dropped) for a code path
		// that never actually reaches a dispersive event at all.
		if (matIsDispersive) swl.TerminateSecondary();

		if (!rd_dist.EffectivelySmooth()) {
			is_specular = false;
			matEta = rd_ri;
			// rd_bxdf's own .ior member is dead for .pdf() (eta is the
			// explicit rd_ri runtime parameter below, matching CPU
			// rough_dielectric's identical "the BxDF's stored ior field is
			// never read by f()/pdf()/sample_local()" pattern) - set it to
			// dispersiveIor anyway so nothing here still displays the flat,
			// non-dispersive value.
			RoughDielectricBxDF<float> rd_bxdf{ dispersiveIor, rd_alpha_x, rd_alpha_y };
			brdf_pdf_override = rd_bxdf.pdf(wi_x, wi_y, wi_z, rd_ri, wo_x, wo_y, wo_z);
			phaseWo = wi_w;
		} else {
			is_specular = true;
		}
		break;
	}
	default:
		// Should be unreachable - dielectricHitQueue is only ever populated
		// with Dielectric/RoughDielectric hits (see this kernel's own header
		// comment). Trap loudly rather than silently mis-shading whatever
		// this actually is, matching evaluate_materials_simple()'s guard.
		printf("[EVAL-MATERIALS-DIELECTRIC] unexpected MaterialType %d in dielectricHitQueue\n", (int)mat.type);
		__trap();
	}

	if (!scattered) {
		addToFramebuffer(h.pixelIndex, radiance * h.filterWeight);
		return;
	}

	// eventEta was set above only on a genuine transmission (Dielectric or
	// RoughDielectric's refract branch) - see this function's own eventEta
	// local comment.
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
