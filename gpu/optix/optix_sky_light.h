// optix_sky_light.h -- Recursive-backend wrapper around the shared HDR-sky
// math in gpu_sky_light_shared.h and gpu_portal_light_shared.h (see each
// file's own header comment - this used to be an independent, hand-
// duplicated copy of the sky-only math). Included by optix_device_helpers.h,
// AFTER the `extern "C" { __constant__ LaunchParams params; }` declaration
// and after random_float()/random_unit_vector() are defined - the functions
// below read params.camera.skyDist/portalLight directly and call
// random_float() for their own samples, same inclusion-order requirement
// gpu_sky_light_shared.h's/gpu_portal_light_shared.h's own functions
// (included just above this file) never needed, since none of them touch
// `params` or this backend's RNG.

// ---------------------------------------------------------------------------
// High-level wrappers used by shade_material()'s NEE blocks and
// __miss__ms() (optix_miss.h). Dispatch between three modes, in priority
// order: a real portal light (params.camera.portalLight.height > 0), a real
// image sky (params.camera.skyDist.height > 0), or the existing uniform-
// sphere + flat-colour path - the first two are mutually exclusive (matches
// CPU: a scene's one infinite light is either a portal or a plain image/
// flat-colour sky, never both - GpuPortalLight's own comment), so checking
// portal first and returning is always correct, never a missed sky check.
// `shadingPoint`: the point emission/NEE is being evaluated FROM - a portal
// light's visible window depends on it (GpuPortalLight's own comment); the
// image-sky and flat-colour paths ignore it entirely, same as before this
// parameter existed.
// ---------------------------------------------------------------------------

// One sky/portal NEE sample: direction, its solid-angle pdf, and the
// radiance arriving from it.
__device__ __forceinline__ void sample_sky_nee(unsigned int& seed, const float3& flatColor,
												  const float3& shadingPoint,
												  float3& dir_out, float& pdf_out, float3& Le_out) {
	const GpuPortalLight& p = params.camera.portalLight;
	if (p.height > 0 && p.width > 0) {
		const float ru = random_float(seed), rv = random_float(seed);
		if (gpu_portal_sample_Li(p, shadingPoint, ru, rv, dir_out, pdf_out, Le_out)) {
			return;
		}
		// Portal window subtends zero area from here (e.g. shading point
		// behind the portal plane) - no light arrives from this strategy at
		// all, unlike the image-sky/flat-colour paths below, which always
		// have SOME direction to offer. Matches CPU's sample_li() returning
		// false with no further fallback (camera.h's own portal NEE block).
		pdf_out = 0.0f;
		Le_out = make_float3(0.0f, 0.0f, 0.0f);
		dir_out = make_float3(0.0f, 1.0f, 0.0f);
		return;
	}
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

// Solid-angle pdf of the sky/portal strategy at a SPECIFIC direction (for
// MIS against a BSDF-sampled escape - __miss__ms()). Uniform-sphere constant
// when neither a portal nor an image sky was uploaded, matching the
// existing behavior exactly.
__device__ __forceinline__ float sky_pdf_for_mis(const float3& dir, const float3& shadingPoint) {
	const GpuPortalLight& p = params.camera.portalLight;
	if (p.height > 0 && p.width > 0) return gpu_portal_pdf_Li(p, shadingPoint, dir);
	const GpuSkyDistribution& d = params.camera.skyDist;
	if (d.height > 0 && d.width > 0) return gpu_sky_pdf_Li(d, dir);
	const float kPi = 3.14159265358979323846f;
	return 1.0f / (4.0f * kPi);
}

// Radiance arriving from a specific (escaped-ray) direction, as seen from
// shadingPoint - real per-direction Le() for a portal or image sky, else
// the flat `flatColor`.
__device__ __forceinline__ float3 sky_radiance(const float3& dir, const float3& flatColor, const float3& shadingPoint) {
	const GpuPortalLight& p = params.camera.portalLight;
	if (p.height > 0 && p.width > 0) return gpu_portal_Le(p, shadingPoint, dir);
	const GpuSkyDistribution& d = params.camera.skyDist;
	if (d.height > 0 && d.width > 0) return gpu_sky_Le(d, dir);
	return flatColor;
}
