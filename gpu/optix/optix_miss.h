// optix_miss.h -- Miss programs
// Included by optix_programs.cu

extern "C" __global__ void __miss__ms() {
	// Flat constant-color background (params.camera.backgroundColor) for a
	// scene with no real HDR sky image, matching src/TheRestOfYourLife/
	// sky_light.h's constant-color constructor - OR, when the scene's
	// infinite light carries a real image (params.camera.skyDist.height > 0,
	// see optix_types.h's GpuSkyDistribution comment), the REAL per-
	// direction radiance at this exact escaped ray's own direction
	// (sky_radiance(), optix_sky_light.h), mirroring sky_light::Le() -
	// instead of the flat mean-brightness wash this backend used to add
	// unconditionally here. `flatColor` still gates "does this scene have a
	// sky at all" below (nonzero for both an image sky and a constant-color
	// one, zero only when there is truly no infinite light), matching prior
	// behavior exactly for every scene without one.
	const float3& flatColor = params.camera.backgroundColor;
	const float3 rayDir = optixGetWorldRayDirection();
	const float3 color = sky_radiance(rayDir, flatColor);

	// Unpack attenuation from payload
	float3 attenuation = make_float3(
		__uint_as_float(optixGetPayload_0()),
		__uint_as_float(optixGetPayload_1()),
		__uint_as_float(optixGetPayload_2())
	);
	unsigned int seed = optixGetPayload_9();

	// MIS weight against the sky-NEE strategy (shade_material()'s own
	// sky-NEE blocks - see their comment), mirroring CPU's camera.h escaped-
	// ray handling exactly: a camera ray or specular bounce (prev_brdf_pdf
	// == 0, optix_raygen.h's own convention) gets the sky's full,
	// unweighted radiance (no NEE strategy could have sampled this ray in
	// the first place); any other bounce discounts it by the balance
	// heuristic against the sky strategy's own pdf at this direction
	// (sky_pdf_for_mis() - the real importance-sampled pdf for an image sky,
	// or the uniform-sphere 1/4pi constant otherwise), so a genuinely-
	// escaped ray doesn't double-count against the NEE sample already taken
	// at the previous hit. Before this, every miss added the FULL background
	// color unconditionally, which - paired with adding sky-NEE - would have
	// double-counted the sky's contribution on every non-specular escape.
	const float prev_brdf_pdf = __uint_as_float(optixGetPayload_12());
	float3 emission = color;
	if (prev_brdf_pdf > 0.0f && (flatColor.x > 0.0f || flatColor.y > 0.0f || flatColor.z > 0.0f)) {
		const float pdf_sky = sky_pdf_for_mis(rayDir);
		float w_b = mis_power_heuristic(prev_brdf_pdf, pdf_sky);
		emission = w_b * color;
	}

	optixSetPayload_3(__float_as_uint(emission.x));
	optixSetPayload_4(__float_as_uint(emission.y));
	optixSetPayload_5(__float_as_uint(emission.z));
	optixSetPayload_9(seed);
	optixSetPayload_10(0);  // absorbed (terminate path with no emission)
	optixSetPayload_12(0);  // no brdf_pdf (path terminated)
}

//==============================================================================
// Shadow Miss Program
//==============================================================================

extern "C" __global__ void __miss__shadow() {
	// Shadow ray missed all geometry - path is clear (not occluded)
	optixSetPayload_0(0);  // occluded = false
}

//==============================================================================
// Ray Generation Program (will implement with path tracing loop)
//==============================================================================

