// wavefront_intersection_quad.h -- Quad intersection + closest-hit programs
// Included by wavefront_programs.cu, after wavefront_common.h.

extern "C" __global__ void __intersection__wf_quad() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const QuadData& quad = wf_params.quads[primIdx];

	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float  ray_tmin = optixGetRayTmin();
	const float  ray_tmax = optixGetRayTmax();

	const float denom = dot(quad.normal, ray_dir);
	if (fabsf(denom) < 1e-8f) return;

	const float t = (quad.D - dot(quad.normal, ray_orig)) / denom;
	if (t < ray_tmin || t > ray_tmax) return;

	const float3 intersection = ray_orig + t * ray_dir;
	const float3 planar_vec = intersection - quad.Q;

	const float w_dot_w = dot(quad.w, quad.w);
	const float alpha = dot(quad.w, cross(planar_vec, quad.v)) / w_dot_w;
	const float beta  = dot(quad.w, cross(quad.u, planar_vec)) / w_dot_w;
	if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f) return;

	optixReportIntersection(
		t, 0,
		__float_as_int(quad.normal.x),
		__float_as_int(quad.normal.y),
		__float_as_int(quad.normal.z),
		0);
}

// ============================================================================
// __closesthit__wf_quad
// ============================================================================
extern "C" __global__ void __closesthit__wf_quad() {
	WfHitPayload* payload = (WfHitPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	const int quadIdx = optixGetPrimitiveIndex();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float  t_hit    = optixGetRayTmax();

	float3 hit_point = ray_orig + t_hit * ray_dir;

	const QuadData& q = wf_params.quads[quadIdx];
	bool front_face = dot(ray_dir, q.normal) < 0.0f;
	float3 normal = front_face ? q.normal : -q.normal;

	payload->hitPoint    = hit_point;
	payload->normal      = normal;
	payload->t           = t_hit;
	payload->materialIdx = q.materialIdx;
	payload->geomType    = 1;
	payload->hit         = true;
	payload->mediumTFar  = 0.0f;
	// frontFace was left uninitialized here (only __closesthit__wf_sphere set
	// it) - RoughDielectric/DielectricMedium in wavefront_kernels.cu read
	// h.frontFace unconditionally regardless of which geometry type hit,
	// so a quad-hit ray fed those cases garbage stack data instead of the
	// real hit side.
	payload->frontFace   = front_face ? 1 : 0;
	// Real UV (alpha,beta - same planar-decomposition formula the recursive
	// backend's __closesthit__quad recomputes from hit_point, and matching
	// CPU quad.h's own rec.u=alpha,rec.v=beta convention) - was previously
	// always (0,0) here, a real gap vs. the recursive backend.
	{
		const float3 planar_vec = hit_point - q.Q;
		const float w_dot_w = dot(q.w, q.w);
		payload->uv_u = dot(q.w, cross(planar_vec, q.v)) / w_dot_w;
		payload->uv_v = dot(q.w, cross(q.u, planar_vec)) / w_dot_w;
	}
}

