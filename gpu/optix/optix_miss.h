// optix_miss.h -- Miss programs
// Included by optix_programs.cu

extern "C" __global__ void __miss__ms() {
	// Cornell Box uses BLACK background (no sky light)
	const float3 color = make_float3(0.0f, 0.0f, 0.0f);

	// Unpack attenuation from payload
	float3 attenuation = make_float3(
		__uint_as_float(optixGetPayload_0()),
		__uint_as_float(optixGetPayload_1()),
		__uint_as_float(optixGetPayload_2())
	);
	unsigned int seed = optixGetPayload_9();

	// Black background - no emission
	float3 emission = make_float3(0.0f, 0.0f, 0.0f);

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

