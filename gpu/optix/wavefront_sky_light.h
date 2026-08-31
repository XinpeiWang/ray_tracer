// wavefront_sky_light.h -- Wavefront-backend wrapper around the shared
// HDR-sky math in gpu_sky_light_shared.h and gpu_portal_light_shared.h (see
// each file's own header comment - this used to be an independent, hand-
// duplicated copy of the sky-only math under a wf_ prefix). Included by
// wavefront_kernels.cu, after wf_rand()/wf_rand_unit() are defined
// (wf_sample_sky_nee() calls wf_rand()).
//
// Unlike optix_sky_light.h's own wrappers, these take explicit
// `const GpuSkyDistribution&`/`const GpuPortalLight&` parameters rather than
// reading a `params` global - evaluate_materials/accumulate_miss are plain
// CUDA kernels, not OptiX programs, so every array/struct they need arrives
// as an explicit parameter (same reasoning as wavefront_measured_bxdf.h's
// own functions taking explicit table/array parameters instead of a
// launch-params global).

// ---------------------------------------------------------------------------
// High-level wrappers used by wf_finish_material_scatter()'s sky-NEE block
// and accumulate_miss(). See gpu_sky_light_shared.h's/gpu_portal_light_
// shared.h's own comments for the underlying math and optix_sky_light.h's
// own identical dispatch/priority-order comment (portal, then image sky,
// then flat-colour) - the only differences here are `sky`/`portal` arriving
// as explicit parameters instead of reads off params.camera, and randoms
// via this backend's own wf_rand()/wf_rand_unit() instead of
// random_float()/random_unit_vector(). `shadingPoint`: see optix_sky_light.h's
// own comment - a portal light's visible window depends on it.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wf_sample_sky_nee(const GpuSkyDistribution& sky, const GpuPortalLight& portal,
													 unsigned int& seed, const float3& flatColor,
													 const float3& shadingPoint,
													 float3& dir_out, float& pdf_out, float3& Le_out) {
	if (portal.height > 0 && portal.width > 0) {
		const float ru = wf_rand(seed), rv = wf_rand(seed);
		if (gpu_portal_sample_Li(portal, shadingPoint, ru, rv, dir_out, pdf_out, Le_out)) {
			return;
		}
		// Portal window subtends zero area from here - see optix_sky_light.h's
		// sample_sky_nee() own comment for why this is a real "no
		// contribution" outcome, not a fallback-to-flat-colour case.
		pdf_out = 0.0f;
		Le_out = make_float3(0.0f, 0.0f, 0.0f);
		dir_out = make_float3(0.0f, 1.0f, 0.0f);
		return;
	}
	if (sky.height > 0 && sky.width > 0) {
		const float ru = wf_rand(seed), rv = wf_rand(seed);
		if (gpu_sky_sample_Li(sky, ru, rv, dir_out, pdf_out)) {
			Le_out = gpu_sky_Le(sky, dir_out);
			return;
		}
	}
	dir_out = wf_rand_unit(seed);
	pdf_out = 1.0f / (4.0f * 3.14159265f);
	Le_out = flatColor;
}

__device__ __forceinline__ float wf_sky_pdf_for_mis(const GpuSkyDistribution& sky, const GpuPortalLight& portal,
													  const float3& dir, const float3& shadingPoint) {
	if (portal.height > 0 && portal.width > 0) return gpu_portal_pdf_Li(portal, shadingPoint, dir);
	if (sky.height > 0 && sky.width > 0) return gpu_sky_pdf_Li(sky, dir);
	return 1.0f / (4.0f * 3.14159265f);
}

__device__ __forceinline__ float3 wf_sky_radiance(const GpuSkyDistribution& sky, const GpuPortalLight& portal,
													 const float3& dir, const float3& flatColor,
													 const float3& shadingPoint) {
	if (portal.height > 0 && portal.width > 0) return gpu_portal_Le(portal, shadingPoint, dir);
	if (sky.height > 0 && sky.width > 0) return gpu_sky_Le(sky, dir);
	return flatColor;
}
