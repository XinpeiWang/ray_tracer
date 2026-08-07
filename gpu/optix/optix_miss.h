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

	float3 emission = color;

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

