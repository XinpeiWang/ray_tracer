// wavefront_sky_light.h -- Wavefront-backend wrapper around the shared
// HDR-sky math in gpu_sky_light_shared.h (see that file's own header
// comment - this used to be an independent, hand-duplicated copy of that
// same math under a wf_ prefix). Included by wavefront_kernels.cu, after
// wf_rand()/wf_rand_unit() are defined (wf_sample_sky_nee() calls wf_rand()).
//
// Unlike optix_sky_light.h's own wrappers, these take an explicit
// `const GpuSkyDistribution&` parameter rather than reading a `params`
// global - evaluate_materials/accumulate_miss are plain CUDA kernels, not
// OptiX programs, so every array/struct they need arrives as an explicit
// parameter (same reasoning as wavefront_measured_bxdf.h's own functions
// taking explicit table/array parameters instead of a launch-params global).

// ---------------------------------------------------------------------------
// High-level wrappers used by wf_finish_material_scatter()'s sky-NEE block
// and accumulate_miss(). See gpu_sky_light_shared.h's own comments for the
// underlying math - identical dispatch to optix_sky_light.h's own
// equivalents, just taking `d` as an explicit parameter instead of reading
// params.camera.skyDist, and generating randoms via this backend's own
// wf_rand()/wf_rand_unit() instead of random_float()/random_unit_vector().
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wf_sample_sky_nee(const GpuSkyDistribution& d, unsigned int& seed,
													 const float3& flatColor,
													 float3& dir_out, float& pdf_out, float3& Le_out) {
	if (d.height > 0 && d.width > 0) {
		const float ru = wf_rand(seed), rv = wf_rand(seed);
		if (gpu_sky_sample_Li(d, ru, rv, dir_out, pdf_out)) {
			Le_out = gpu_sky_Le(d, dir_out);
			return;
		}
	}
	dir_out = wf_rand_unit(seed);
	pdf_out = 1.0f / (4.0f * 3.14159265f);
	Le_out = flatColor;
}

__device__ __forceinline__ float wf_sky_pdf_for_mis(const GpuSkyDistribution& d, const float3& dir) {
	if (d.height > 0 && d.width > 0) return gpu_sky_pdf_Li(d, dir);
	return 1.0f / (4.0f * 3.14159265f);
}

__device__ __forceinline__ float3 wf_sky_radiance(const GpuSkyDistribution& d, const float3& dir,
													 const float3& flatColor) {
	if (d.height > 0 && d.width > 0) return gpu_sky_Le(d, dir);
	return flatColor;
}
