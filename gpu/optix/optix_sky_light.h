// optix_sky_light.h -- Recursive-backend wrapper around the shared HDR-sky
// math in gpu_sky_light_shared.h (see that file's own header comment - this
// used to be an independent, hand-duplicated copy of that same math).
// Included by optix_device_helpers.h, AFTER the `extern "C" { __constant__
// LaunchParams params; }` declaration and after random_float()/
// random_unit_vector() are defined - the functions below read
// params.camera.skyDist directly and call random_float() for their own
// samples, same inclusion-order requirement gpu_sky_light_shared.h's own
// functions (included just above this file) never needed, since none of
// them touch `params` or this backend's RNG.

// ---------------------------------------------------------------------------
// High-level wrappers used by shade_material()'s NEE blocks and
// __miss__ms() (optix_miss.h). Dispatch between the real image-based
// machinery in gpu_sky_light_shared.h (params.camera.skyDist.height > 0)
// and the existing uniform-sphere + flat-colour path (unchanged), exactly
// mirroring sky_light's own dual-mode behavior (has_dist true/false).
// ---------------------------------------------------------------------------

// One sky NEE sample: direction, its solid-angle pdf, and the radiance
// arriving from it (Le at that exact direction for an image sky, or the flat
// `flatColor` otherwise).
__device__ __forceinline__ void sample_sky_nee(unsigned int& seed, const float3& flatColor,
												  float3& dir_out, float& pdf_out, float3& Le_out) {
	const GpuSkyDistribution& d = params.camera.skyDist;
	if (d.height > 0 && d.width > 0) {
		const float ru = random_float(seed), rv = random_float(seed);
		if (gpu_sky_sample_Li(d, ru, rv, dir_out, pdf_out)) {
			Le_out = gpu_sky_Le(d, dir_out);
			return;
		}
	}
	const float kPi = 3.14159265358979323846f;
	dir_out = random_unit_vector(seed);
	pdf_out = 1.0f / (4.0f * kPi);
	Le_out = flatColor;
}

// Solid-angle pdf of the sky strategy at a SPECIFIC direction (for MIS
// against a BSDF-sampled escape - __miss__ms()). Uniform-sphere constant when
// no image was uploaded, matching the existing behavior exactly.
__device__ __forceinline__ float sky_pdf_for_mis(const float3& dir) {
	const GpuSkyDistribution& d = params.camera.skyDist;
	if (d.height > 0 && d.width > 0) return gpu_sky_pdf_Li(d, dir);
	const float kPi = 3.14159265358979323846f;
	return 1.0f / (4.0f * kPi);
}

// Radiance arriving from a specific (escaped-ray) direction - real per-
// direction Le() for an image sky, else the flat `flatColor`.
__device__ __forceinline__ float3 sky_radiance(const float3& dir, const float3& flatColor) {
	const GpuSkyDistribution& d = params.camera.skyDist;
	if (d.height > 0 && d.width > 0) return gpu_sky_Le(d, dir);
	return flatColor;
}
