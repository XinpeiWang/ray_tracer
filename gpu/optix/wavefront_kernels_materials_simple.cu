// wavefront_kernels_materials_simple.cu
// CUDA compute kernel: evaluate_materials_simple, the Lambertian/Metal-only
// twin of evaluate_materials (see this file's own header comment on why it
// exists as a separate kernel). Split out of the original
// wavefront_kernels.cu for the same independent-recompilation reason as
// wavefront_kernels_materials.cu - see wavefront_device_helpers.h's own
// header comment for where the two shared helpers now live.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "wavefront_device_helpers.h"

// ============================================================================
// evaluate_materials_simple
//
// Twin of evaluate_materials() above, scoped to just MaterialType::Lambertian
// and MaterialType::Metal - the two cheap material types routed into
// simpleHitQueue at push time (see wavefront_programs.cu's
// __raygen__wf_intersect). Everything shared with evaluate_materials() below
// (the spectral-state preamble, the maxDepth cutoff, the Lambertian/Metal
// case bodies themselves, the wf_finish_material_scatter tail call) is copied
// verbatim, not reimplemented - this kernel exists ONLY to replace the big
// switch with a 2-way if/else, so the compiler doesn't have to reserve
// registers for the switch's expensive arms (CoatedDiffuse/CoatedConductor/
// Principled/RoughDielectric/...) for threads that will never take them,
// improving occupancy for this launch. No DiffuseLight early-exit is needed
// here - the routing at push time guarantees only Lambertian/Metal hits ever
// reach this queue. That routing lives in a different file
// (wavefront_programs.cu's __raygen__wf_intersect) than this kernel, though,
// so nothing here structurally prevents the two from drifting apart (e.g. a
// future edit widening the push-site condition without updating this
// kernel) - explicitly checking MaterialType::Metal rather than assuming
// "must be Metal" in an unconditional else, with the same trap-on-default
// this file's evaluate_materials() switch uses, catches that drift loudly
// instead of silently mis-shading whatever slipped through as Metal.
//
// Fewer scene-data parameters than evaluate_materials(): no bssrdfProbeQueue
// (Subsurface-only), no cloud/RGB-grid medium buffers or measured-BRDF
// tables (none of those material types can reach this queue).
// ============================================================================
extern "C" __global__ void evaluate_materials_simple(
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
	const TextureData*  textures,
	const unsigned char* texturePixels,
	int maxDepth,
	float3 skyColor,
	float shadowRayEpsilon,
	GpuSkyDistribution skyDist,
	GpuPortalLight portalLight,
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
	// giOriginContext/giCandidateOut parameter comments. nullptr for batch/
	// offline rendering. This is the kernel Lambertian hits are actually
	// routed through (WavefrontQueues::simpleHitQueue's own comment) - the
	// ONLY material type ReSTIR GI's MVP scope supports (see
	// wf_finish_material_scatter's own giOriginContext-stash comment,
	// wavefront_device_helpers.h) - so this call site matters most.
	GpuGiOriginContext* giOriginContext,
	GpuGiSample* giCandidateOut,
	// See wf_light_bvh_sample_index()'s own comment
	// (wavefront_restir_helpers.h). lightBvh.nodeCount<=0 (the default) means
	// "no light BVH built".
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
		// Max depth reached — just accumulate what we have.
		addToFramebuffer(h.pixelIndex, radiance * h.filterWeight);
		return;
	}

	float3 scattered_dir    = make_float3(0, 0, 0);
	SS     attenuation(0.f);
	bool   scattered        = false;
	bool   is_specular      = false;
	float  brdf_pdf_override = -1.0f;
	// Never set by Lambertian/Metal - passed through to wf_finish_material_
	// scatter as harmless zero, matching evaluate_materials()'s own default
	// for every material type that doesn't use them.
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

	if (mat.type == MaterialType::Lambertian) {
		scattered_dir = normalize(normal + wf_rand_unit(seed));
		if (wf_near_zero(scattered_dir)) scattered_dir = normal;
		const float3 lambertianColor = (mat.textureIdx >= 0)
			? wf_sample_texture(textures, texturePixels, mat.textureIdx, h.uv_u, h.uv_v, hit_point) * mat.emissionScale
			: mat.albedo;
		attenuation = albedoSpectrum(lambertianColor);
		scattered   = true;
		is_specular = false;
	} else if (mat.type == MaterialType::Metal) {
		float3 reflected = wf_reflect(normalize(h.rayDir), normal);
		scattered_dir    = normalize(reflected + mat.fuzz * wf_rand_unit(seed));
		attenuation      = albedoSpectrum(mat.albedo);
		scattered        = (dot(scattered_dir, normal) > 0.0f);
		is_specular      = true;
	} else {
		// Should be unreachable - simpleHitQueue is only ever populated with
		// Lambertian/Metal hits (see this kernel's own header comment). Trap
		// loudly rather than silently mis-shading whatever this actually is
		// as Metal, matching evaluate_materials()'s own default: guard.
		printf("[EVAL-MATERIALS-SIMPLE] unexpected MaterialType %d in simpleHitQueue\n", (int)mat.type);
		__trap();
	}

	if (!scattered) {
		addToFramebuffer(h.pixelIndex, radiance * h.filterWeight);
		return;
	}

	// Lambertian/Metal never transmit - h.etaScale passes through unchanged
	// (see RayWorkItem::etaScale's own comment).
	wf_finish_material_scatter(mat.type, matEta, h.materialIdx, (bool)h.any_nonspecular, glossyAlphaForNEE, glossyAlphaVForNEE, h.etaScale, h.filterWeight, maxComponentValue, normal, hit_point, h.objDpdu, seed,
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
