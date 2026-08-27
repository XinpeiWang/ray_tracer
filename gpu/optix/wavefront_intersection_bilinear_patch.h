// wavefront_intersection_bilinear_patch.h -- Bilinear patch intersection + closest-hit programs
// Included by wavefront_programs.cu, after wavefront_common.h.

// ============================================================================
// __intersection__wf_bilinear_patch / __closesthit__wf_bilinear_patch
//
// Wavefront-native duplicate of optix_intersection_bilinear_patch.h's
// __intersection__bilinear_patch (same quadratic-solve math, wf_params
// instead of the recursive path's params global) - same cross-module
// payload-type mismatch reason as __intersection__wf_sphere/wf_quad above
// for why this isn't just reused directly from the recursive module.
// closesthit only fills geometry (mirrors wf_sphere/wf_quad) - material
// shading happens uniformly in evaluate_materials, keyed off mat.type, not
// per-geometry-type.
// ============================================================================

extern "C" __global__ void __intersection__wf_bilinear_patch() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const BilinearPatchData& patch = wf_params.bilinearPatches[primIdx];

	const float3 ro = optixGetWorldRayOrigin();
	const float3 rd = optixGetWorldRayDirection();
	const float ray_tmin = optixGetRayTmin();
	const float ray_tmax = optixGetRayTmax();

	const float3 p00 = patch.p00;
	const float3 p10 = patch.p10;
	const float3 p01 = patch.p01;
	const float3 p11 = patch.p11;

	const float3 e0 = p10 - p00;
	const float3 e1 = p01 - p11;
	const float3 e2 = p00 - ro;
	const float3 e3 = p10 - ro;

	const float a = dot(cross(e0, e1), rd);
	const float c = dot(cross(e2, rd), p01 - p00);
	const float b = dot(cross(e3, rd), p11 - p10) - (a + c);

	float u1, u2;
	if (a == 0.0f) {
		if (b == 0.0f) return;
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
		__float_as_int(u_hit),
		__float_as_int(v_hit),
		0, 0);
}

extern "C" __global__ void __closesthit__wf_bilinear_patch() {
	WfHitPayload* payload = (WfHitPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	const int primIdx = optixGetPrimitiveIndex();
	const BilinearPatchData& patch = wf_params.bilinearPatches[primIdx];

	const float u = __int_as_float(optixGetAttribute_0());
	const float v = __int_as_float(optixGetAttribute_1());

	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float  t_hit    = optixGetRayTmax();

	float3 hit_point = ray_orig + t_hit * ray_dir;

	const float3 pu0 = lerp(patch.p00, patch.p01, v);
	const float3 pu1 = lerp(patch.p10, patch.p11, v);
	const float3 dpdu = pu1 - pu0;
	const float3 dpdv = lerp(patch.p01, patch.p11, u) - lerp(patch.p00, patch.p10, u);

	float3 geom_normal = normalize(cross(dpdu, dpdv));
	bool front_face = dot(ray_dir, geom_normal) < 0.0f;
	float3 normal = front_face ? geom_normal : -geom_normal;

	payload->hitPoint    = hit_point;
	payload->normal      = normal;
	payload->t           = t_hit;
	payload->materialIdx = patch.materialIdx;
	payload->geomType    = 2;
	payload->hit         = true;
	payload->mediumTFar  = 0.0f;
	// See __closesthit__wf_quad's comment on frontFace - same missing
	// initialization here.
	payload->frontFace   = front_face ? 1 : 0;
	// Real UV: (u,v) are the patch's own bilinear parametric coordinates,
	// already recovered from the intersection attributes above - matches
	// the recursive backend's __closesthit__bilinear_patch, which passes
	// this same (u,v) pair to material_emission()/shade_material() instead
	// of always reading texel (0,0) (was previously zeroed here, a real gap
	// between the two backends since the intersection program already
	// computed the real values).
	payload->uv_u        = u;
	payload->uv_v        = v;
	// See WfHitPayload::objDpdu's own comment - the bilinear-patch use,
	// only meaningful for MaterialType::Hair. Stored UNNORMALIZED - the
	// normalize() (sqrt + 3 divides) is deferred to evaluate_materials()'s
	// own Hair case, the only reader, so every non-Hair bilinear-patch hit
	// (most of them - see that field's own comment) doesn't pay for it.
	payload->objDpdu     = dpdu;
}

