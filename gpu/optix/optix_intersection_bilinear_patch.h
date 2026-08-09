// optix_intersection_bilinear_patch.h -- Bilinear patch intersection + closest-hit programs
// Included by optix_programs.cu
//
// A genuinely curved (non-planar in general) ruled surface through 4 corners
// p00,p10,p01,p11 - NOT a flat quad. Ray intersection ports
// src/shared/bilinear_patch.h's blp_intersect() (Ramsey et al. 2004 / pbrt-v4
// IntersectBilinearPatch): solve a quadratic in u for where the ray crosses
// the patch's u-isolines, then for each valid root find the closest-point-
// between-two-lines (ray vs. u-isoline) to get v and t.

extern "C" __global__ void __intersection__bilinear_patch() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const BilinearPatchData& patch = params.bilinearPatches[primIdx];

	const float3 ro = optixGetWorldRayOrigin();
	const float3 rd = optixGetWorldRayDirection();
	const float ray_tmin = optixGetRayTmin();
	const float ray_tmax = optixGetRayTmax();

	const float3 p00 = patch.p00;
	const float3 p10 = patch.p10;
	const float3 p01 = patch.p01;
	const float3 p11 = patch.p11;

	// Quadratic coefficients for the u iso-line distance (blp_intersect)
	const float3 e0 = p10 - p00;
	const float3 e1 = p01 - p11;
	const float3 e2 = p00 - ro;
	const float3 e3 = p10 - ro;

	const float a = dot(cross(e0, e1), rd);
	const float c = dot(cross(e2, rd), p01 - p00);
	const float b = dot(cross(e3, rd), p11 - p10) - (a + c);

	float u1, u2;
	if (a == 0.0f) {
		if (b == 0.0f) return;  // no solution
		u1 = u2 = -c / b;
	} else {
		const double disc = (double)b * (double)b - 4.0 * (double)a * (double)c;
		if (disc < 0.0) return;
		const double sqrt_disc = sqrt(disc);
		const double q = (b < 0.0f) ? -0.5 * ((double)b - sqrt_disc) : -0.5 * ((double)b + sqrt_disc);
		u1 = (float)(q / (double)a);
		u2 = (q != 0.0) ? (float)((double)c / q) : 0.0f;
		if (u1 > u2) { const float tmp = u1; u1 = u2; u2 = tmp; }
	}

	// eps = gamma(30) * (max|ro| + max|rd| + max|p00|+max|p10|+max|p01|+max|p11|)
	const float eps = 4e-6f *
		(fmaxf(fmaxf(fabsf(ro.x), fabsf(ro.y)), fabsf(ro.z)) +
		 fmaxf(fmaxf(fabsf(rd.x), fabsf(rd.y)), fabsf(rd.z)) +
		 fmaxf(fmaxf(fabsf(p00.x), fabsf(p00.y)), fabsf(p00.z)) +
		 fmaxf(fmaxf(fabsf(p10.x), fabsf(p10.y)), fabsf(p10.z)) +
		 fmaxf(fmaxf(fabsf(p01.x), fabsf(p01.y)), fabsf(p01.z)) +
		 fmaxf(fmaxf(fabsf(p11.x), fabsf(p11.y)), fabsf(p11.z)));

	float t_hit = ray_tmax;
	float u_hit = -1.0f, v_hit = -1.0f;
	bool found = false;

	for (int i = 0; i < 2; ++i) {
		const float u_cand = (i == 0) ? u1 : u2;
		if (i == 1 && u2 == u1) continue;
		if (u_cand < 0.0f || u_cand > 1.0f) continue;

		const float3 uo   = lerp(p00, p10, u_cand);
		const float3 p11u = lerp(p01, p11, u_cand);
		const float3 ud   = p11u - uo;

		const float3 delta = uo - ro;
		const float3 perp  = cross(rd, ud);
		const float p2 = dot(perp, perp);
		if (p2 < 1e-20f) continue;

		// det3(a,b,c) == dot(a, cross(b,c)) (scalar triple product)
		const float v_num = dot(delta, cross(rd, perp));
		const float t_num = dot(delta, cross(ud, perp));

		const float t_cand = t_num / p2;
		const float v_cand = v_num / p2;

		if (t_num > p2 * eps && v_cand >= 0.0f && v_cand <= 1.0f &&
			t_cand < t_hit && t_cand >= ray_tmin && t_cand <= ray_tmax) {
			t_hit = t_cand;
			u_hit = u_cand;
			v_hit = v_cand;
			found = true;
		}
	}

	if (!found) return;

	optixReportIntersection(
		t_hit, 0,
		__float_as_int(u_hit),  // attribute 0
		__float_as_int(v_hit),  // attribute 1
		0, 0                    // attributes 2-3 unused
	);
}

//==============================================================================
// Bilinear Patch Closest Hit Program
//==============================================================================

extern "C" __global__ void __closesthit__bilinear_patch() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const BilinearPatchData& patch = params.bilinearPatches[primIdx];
	const int matIdx = patch.materialIdx;
	const MaterialData& mat = params.materials[matIdx];

	const float u = __int_as_float(optixGetAttribute_0());
	const float v = __int_as_float(optixGetAttribute_1());

	const float t = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t * ray_dir;

	// dpdu/dpdv at (u,v) - see bilinear_patch.h's blp_point
	const float3 pu0 = lerp(patch.p00, patch.p01, v);
	const float3 pu1 = lerp(patch.p10, patch.p11, v);
	const float3 dpdu = pu1 - pu0;
	const float3 dpdv = lerp(patch.p01, patch.p11, u) - lerp(patch.p00, patch.p10, u);

	const float3 geom_normal = normalize(cross(dpdu, dpdv));
	const bool front_face = dot(ray_dir, geom_normal) < 0.0f;
	const float3 final_normal = front_face ? geom_normal : -geom_normal;

	// Unpack payload from registers
	float3 attenuation_in = make_float3(
		__uint_as_float(optixGetPayload_0()),
		__uint_as_float(optixGetPayload_1()),
		__uint_as_float(optixGetPayload_2())
	);
	unsigned int seed = optixGetPayload_9();

	// Get emission from material (all materials can emit, most have emission=0)
	float3 emission = mat.emission;

	// Material scattering (same as sphere/quad)
	float3 attenuation;
	float3 scattered_dir;
	bool scattered = false;
	bool is_specular = false;  // pbrt-v4 specularBounce: MIS is skipped for specular events
	float brdf_pdf_override = -1.0f;  // if >= 0, overrides cosine_pdf in payload packing

	shade_material(mat, final_normal, ray_dir, hit_point, front_face, seed,
		attenuation, scattered_dir, scattered, is_specular, brdf_pdf_override, emission);

	// Pack updated payload back into registers (see optix_intersection_quad.h
	// for the p0-p12 layout - identical here)
	optixSetPayload_3(__float_as_uint(emission.x));
	optixSetPayload_4(__float_as_uint(emission.y));
	optixSetPayload_5(__float_as_uint(emission.z));
	optixSetPayload_9(seed);

	if (scattered) {
		float t_hit = optixGetRayTmax();
		float brdf_pdf_out = is_specular ? 0.0f
						  : (brdf_pdf_override >= 0.0f ? brdf_pdf_override : cosine_pdf(scattered_dir, final_normal));

		optixSetPayload_0(__float_as_uint(attenuation.x));
		optixSetPayload_1(__float_as_uint(attenuation.y));
		optixSetPayload_2(__float_as_uint(attenuation.z));
		optixSetPayload_6(__float_as_uint(scattered_dir.x));
		optixSetPayload_7(__float_as_uint(scattered_dir.y));
		optixSetPayload_8(__float_as_uint(scattered_dir.z));
		optixSetPayload_10(1);  // scattered
		optixSetPayload_11(__float_as_uint(t_hit));
		optixSetPayload_12(__float_as_uint(brdf_pdf_out));
	} else if (mat.type == MaterialType::DiffuseLight) {
		// See the DiffuseLight case comment above - not reachable in practice
		// (no scene registers a bilinear patch as a light), pdf=0 is safe.
		optixSetPayload_10(2);  // hit_light
		optixSetPayload_12(0);
	} else {
		optixSetPayload_10(0);  // absorbed
		optixSetPayload_12(0);
	}
}
