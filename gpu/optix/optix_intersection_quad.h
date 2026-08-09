// optix_intersection_quad.h -- Quad intersection + closest-hit programs
// Included by optix_programs.cu

extern "C" __global__ void __intersection__quad() {
	// Get primitive index from OptiX (this is relative to the build input)
	// Since we have 2 build inputs (spheres first, quads second), we need the quad index
	const unsigned int primIdx = optixGetPrimitiveIndex();

	// Fetch quad data from device array
	const QuadData& quad = params.quads[primIdx];

	// Get ray parameters
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float ray_tmin = optixGetRayTmin();
	const float ray_tmax = optixGetRayTmax();

	// Quad intersection (plane intersection + bounds check)
	const float denom = dot(quad.normal, ray_dir);

	// Parallel to plane?
	if (fabsf(denom) < 1e-8f) return;

	// Compute t
	const float t = (quad.D - dot(quad.normal, ray_orig)) / denom;
	if (t < ray_tmin || t > ray_tmax) return;

	// Compute hit point in plane
	const float3 intersection = ray_orig + t * ray_dir;
	const float3 planar_vec = intersection - quad.Q;

	// Check if hit is within quad bounds using barycentric coordinates
	const float w_dot_w = dot(quad.w, quad.w);
	const float alpha = dot(quad.w, cross(planar_vec, quad.v)) / w_dot_w;
	const float beta = dot(quad.w, cross(quad.u, planar_vec)) / w_dot_w;

	if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f)
		return;  // Outside quad

	// Report intersection
	optixReportIntersection(
		t,                      // t value
		0,                      // hit kind
		__float_as_int(quad.normal.x),  // attribute 0 (normal)
		__float_as_int(quad.normal.y),  // attribute 1
		__float_as_int(quad.normal.z),  // attribute 2
		0                                // attribute 3 (unused, but required for consistency)
	);
}

//==============================================================================
// Quad Closest Hit Program
//==============================================================================

extern "C" __global__ void __closesthit__quad() {
	// Get primitive index
	const unsigned int primIdx = optixGetPrimitiveIndex();

	// Fetch quad data from device array
	const QuadData& quad = params.quads[primIdx];
	const int matIdx = quad.materialIdx;
	const MaterialData& mat = params.materials[matIdx];

	// Reconstruct normal from attributes
	const float3 normal = make_float3(
		__int_as_float(optixGetAttribute_0()),
		__int_as_float(optixGetAttribute_1()),
		__int_as_float(optixGetAttribute_2())
	);

	// Get hit point
	const float t = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t * ray_dir;

	// Determine front face
	const bool front_face = dot(ray_dir, normal) < 0.0f;
	const float3 final_normal = front_face ? normal : -normal;

	// Unpack payload from registers
	float3 attenuation_in = make_float3(
		__uint_as_float(optixGetPayload_0()),
		__uint_as_float(optixGetPayload_1()),
		__uint_as_float(optixGetPayload_2())
	);
	unsigned int seed = optixGetPayload_9();

	// Get emission from material (all materials can emit, most have emission=0)
	float3 emission = mat.emission;

	// Material scattering (same as sphere)
	float3 attenuation;
	float3 scattered_dir;
	bool scattered = false;
	bool is_specular = false;  // pbrt-v4 specularBounce: MIS is skipped for specular events
	float brdf_pdf_override = -1.0f;  // if >= 0, overrides cosine_pdf in payload packing

	shade_material(mat, final_normal, ray_dir, hit_point, front_face, 0.0f, 0.0f, seed,
		attenuation, scattered_dir, scattered, is_specular, brdf_pdf_override, emission);
			// Pack updated payload back into registers
			// p0-p2: surface attenuation (BRDF albedo - raygen multiplies with throughput)
	// p3-p5: emission from this surface hit
	// p6-p8: scatter direction (if scattered)
	// p9: updated seed
	// p10: scattered flag (0=absorbed, 1=scattered, 2=hit_light)
	// p11: hit distance 't'
	// p12: brdf_pdf of scattered dir (flag==1) OR light NEE pdf of incoming dir (flag==2) for MIS

	// Always set emission (for all material types - non-lights have emission=0)
	optixSetPayload_3(__float_as_uint(emission.x));
	optixSetPayload_4(__float_as_uint(emission.y));
	optixSetPayload_5(__float_as_uint(emission.z));
	optixSetPayload_9(seed);

	if (scattered) {
		// Return surface attenuation ONLY (raygen will multiply with throughput)
		float t_hit = optixGetRayTmax();  // Hit distance
		// p12: BRDF PDF for MIS on the next bounce.
		// Specular (Metal/Dielectric): 0 = no MIS (pbrt-v4 specularBounce pattern).
		// Lambertian: cosine_pdf of the sampled direction.
		float brdf_pdf_out = is_specular ? 0.0f
						  : (brdf_pdf_override >= 0.0f ? brdf_pdf_override : cosine_pdf(scattered_dir, final_normal));

		optixSetPayload_0(__float_as_uint(attenuation.x));
		optixSetPayload_1(__float_as_uint(attenuation.y));
		optixSetPayload_2(__float_as_uint(attenuation.z));
		optixSetPayload_6(__float_as_uint(scattered_dir.x));  // Scatter direction
		optixSetPayload_7(__float_as_uint(scattered_dir.y));
		optixSetPayload_8(__float_as_uint(scattered_dir.z));
		optixSetPayload_10(1);  // scattered
		optixSetPayload_11(__float_as_uint(t_hit));
		optixSetPayload_12(__float_as_uint(brdf_pdf_out));
	} else if (mat.type == MaterialType::DiffuseLight) {
		// p12: NEE PDF for the incoming ray direction reaching this quad light.
		float light_pdf_for_incoming = 0.0f;
		if (params.aliasTable && params.numLights > 0) {
			int prim_idx = (int)primIdx;
			float sel_pdf = 0.0f;
			for (unsigned int li = 0; li < params.numLights; ++li) {
				if (params.lightIndices[li] == prim_idx && !params.isLightSphere[li]) {
					sel_pdf = params.aliasTable[li].pdf;
					break;
				}
			}
			// Solid-angle PDF for quad: dist^2 / (cos * area)
			float3 to_light = hit_point - ray_orig;
			float dist_sq = dot(to_light, to_light);
			float area = length(quad.w);  // |u x v|
			float cos_theta = fabsf(dot(normalize(ray_dir), quad.normal));
			if (cos_theta > 1e-6f && area > 1e-6f && dist_sq > 1e-6f)
				light_pdf_for_incoming = sel_pdf * dist_sq / (cos_theta * area);
		}
		optixSetPayload_10(2);  // hit_light
		optixSetPayload_12(__float_as_uint(light_pdf_for_incoming));
	} else {
		optixSetPayload_10(0);  // absorbed
		optixSetPayload_12(0);
	}
}

//==============================================================================
// Shadow Any-Hit Programs
//==============================================================================

// Shadow any-hit for spheres
// For opaque geometry, any hit means occlusion - terminate immediately
