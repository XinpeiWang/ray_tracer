// optix_miss.h -- Miss programs
// Included by optix_programs.cu

extern "C" __global__ void __miss__ms() {
	// Flat constant-color background (params.backgroundColor), matching
	// src/TheRestOfYourLife/sky_light.h's constant-color constructor - the
	// only sky_light mode any scene actually uses (scenes_advanced.h's
	// build_hdri_sky()/build_portal_sky() both return a solid color; the
	// "HDRI"/importance-sampled-image machinery in image_infinite_light.h
	// is unused dead code on the CPU side too, never wired to a scene).
	// Defaults to black (0,0,0) for every scene that doesn't set it,
	// preserving prior behavior exactly.
	const float3 color = params.camera.backgroundColor;

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
	// heuristic against the sky's uniform-sphere pdf (1/4pi - the only sky
	// mode any GPU scene builder sets), so a genuinely-escaped ray doesn't
	// double-count against the NEE sample already taken at the previous
	// hit. Before this, every miss added the FULL background color
	// unconditionally, which - paired with adding sky-NEE - would have
	// double-counted the sky's contribution on every non-specular escape.
	const float prev_brdf_pdf = __uint_as_float(optixGetPayload_12());
	float3 emission = color;
	if (prev_brdf_pdf > 0.0f && (color.x > 0.0f || color.y > 0.0f || color.z > 0.0f)) {
		constexpr float pdf_sky = 1.0f / (4.0f * 3.14159265358979323846f);
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

