// optix_probe_hit.h -- RAY_TYPE_PROBE closest-hit + miss programs (recursive
// backend only, Phase 1 BSSRDF). Included by optix_programs.cu, AFTER
// optix_device_helpers.h (needs `params` and trace_probe_ray()'s
// ProbeHit type).
//
// One closest-hit program per geometry type (sphere/quad/bilinear-patch/
// triangle), each a deliberately MINIMAL twin of that type's own
// __closesthit__<type>() in optix_intersection_<type>.h: they compute the
// SAME hit position + front-face-corrected shading normal + material index,
// but skip shade_material()/NEE/scattering entirely and instead write those
// three values into the small dedicated probe payload (see optix_types.h's
// RAY_TYPE_PROBE comment for why this needs its own hit groups rather than
// reusing the radiance ones). bssrdf_probe_walk() (optix_device_helpers.h)
// is the sole caller, via trace_probe_ray().
//
// Deliberately skips alpha-cutout (__anyhit__triangle's map_d handling): the
// probe walk only ever runs against a Subsurface material's own mesh in
// every scene this phase targets (the SSS dragon scenes have no alpha-
// cutout materials at all), and a probe ray passing near unrelated alpha-
// cutout geometry elsewhere in a general scene would at worst report one
// extra non-matching-material candidate that bssrdf_probe_walk() already
// discards (see its own reservoir-sampling loop) - a documented, low-risk
// simplification, not a correctness gap for this phase's target scenes.
//
// Probe payload (7 registers, distinct from the radiance ray's 13 - see
// RAY_TYPE_PROBE's own comment for why they don't share one convention):
//   p0        : material index (int, bit-reinterpreted; -1 = no hit / miss)
//   p1,p2,p3  : world-space hit position (float bits)
//   p4,p5,p6  : world-space front-face-corrected shading normal (float bits)

extern "C" __global__ void __closesthit__probe_sphere() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const int instBase = params.instancePrimBase
		? params.instancePrimBase[optixGetInstanceId()] : -1;
	const unsigned int sphBase = (instBase >= 0) ? (unsigned int)instBase : 0u;
	const SphereData& sphere = params.spheres[sphBase + primIdx];

	const float3 sphere_center = make_float3(
		__int_as_float(optixGetAttribute_0()),
		__int_as_float(optixGetAttribute_1()),
		__int_as_float(optixGetAttribute_2())
	);
	const float sphere_radius = __int_as_float(optixGetAttribute_3());

	const float t = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t * ray_dir;

	// See GpuMediumShapeKind's comment in optix_types.h / optix_intersection_
	// sphere.h's own __closesthit__sphere for why box bounds are re-read
	// directly rather than via attributes. No scene currently combines a
	// Subsurface probe walk with a box-shaped medium boundary (Subsurface
	// never applies to spheres in this loader at all), so this branch is
	// unreachable today, but kept consistent with the radiance closest-hit
	// above rather than left silently wrong.
	const bool is_box = (sphere.shapeKind == GpuMediumShapeKind::Box);

	float3 obj_hit = hit_point;
	if (instBase >= 0) obj_hit = optixTransformPointFromWorldToObjectSpace(hit_point);
	const float3 obj_normal = is_box
		? box_face_normal(obj_hit, sphere.boxMin, sphere.boxMax)
		: (obj_hit - sphere_center) / sphere_radius;
	float3 outward_normal = obj_normal;
	if (instBase >= 0) outward_normal = normalize(optixTransformNormalFromObjectToWorldSpace(obj_normal));
	const bool front_face = dot(ray_dir, outward_normal) < 0.0f;
	const float3 normal = front_face ? outward_normal : -outward_normal;

	optixSetPayload_0((unsigned int)sphere.materialIdx);
	optixSetPayload_1(__float_as_uint(hit_point.x));
	optixSetPayload_2(__float_as_uint(hit_point.y));
	optixSetPayload_3(__float_as_uint(hit_point.z));
	optixSetPayload_4(__float_as_uint(normal.x));
	optixSetPayload_5(__float_as_uint(normal.y));
	optixSetPayload_6(__float_as_uint(normal.z));
}

extern "C" __global__ void __closesthit__probe_quad() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const QuadData& quad = params.quads[primIdx];

	const float3 normal_raw = make_float3(
		__int_as_float(optixGetAttribute_0()),
		__int_as_float(optixGetAttribute_1()),
		__int_as_float(optixGetAttribute_2())
	);

	const float t = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t * ray_dir;

	const bool front_face = dot(ray_dir, normal_raw) < 0.0f;
	const float3 normal = front_face ? normal_raw : -normal_raw;

	optixSetPayload_0((unsigned int)quad.materialIdx);
	optixSetPayload_1(__float_as_uint(hit_point.x));
	optixSetPayload_2(__float_as_uint(hit_point.y));
	optixSetPayload_3(__float_as_uint(hit_point.z));
	optixSetPayload_4(__float_as_uint(normal.x));
	optixSetPayload_5(__float_as_uint(normal.y));
	optixSetPayload_6(__float_as_uint(normal.z));
}

extern "C" __global__ void __closesthit__probe_bilinear_patch() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const BilinearPatchData& patch = params.bilinearPatches[primIdx];

	const float u = __int_as_float(optixGetAttribute_0());
	const float v = __int_as_float(optixGetAttribute_1());

	const float t = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t * ray_dir;

	const float3 pu0 = lerp(patch.p00, patch.p01, v);
	const float3 pu1 = lerp(patch.p10, patch.p11, v);
	const float3 dpdu = pu1 - pu0;
	const float3 dpdv = lerp(patch.p01, patch.p11, u) - lerp(patch.p00, patch.p10, u);
	const float3 geom_normal = normalize(cross(dpdu, dpdv));
	const bool front_face = dot(ray_dir, geom_normal) < 0.0f;
	const float3 normal = front_face ? geom_normal : -geom_normal;

	optixSetPayload_0((unsigned int)patch.materialIdx);
	optixSetPayload_1(__float_as_uint(hit_point.x));
	optixSetPayload_2(__float_as_uint(hit_point.y));
	optixSetPayload_3(__float_as_uint(hit_point.z));
	optixSetPayload_4(__float_as_uint(normal.x));
	optixSetPayload_5(__float_as_uint(normal.y));
	optixSetPayload_6(__float_as_uint(normal.z));
}

extern "C" __global__ void __closesthit__probe_triangle() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const int instBase = params.instancePrimBase
		? params.instancePrimBase[optixGetInstanceId()] : -1;
	const unsigned int triBase = (instBase >= 0) ? (unsigned int)instBase : 0u;
	const TriangleData& tri = params.triangles[triBase + primIdx];

	const float t = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t * ray_dir;

	float3 shading_normal;
	if (tri.hasNormals) {
		const float2 bary = optixGetTriangleBarycentrics();
		const float b1 = bary.x, b2 = bary.y, b0 = 1.0f - b1 - b2;
		shading_normal = normalize(b0 * tri.n0 + b1 * tri.n1 + b2 * tri.n2);
	} else {
		shading_normal = normalize(cross(tri.p1 - tri.p0, tri.p2 - tri.p0));
	}
	if (instBase >= 0) shading_normal = normalize(optixTransformNormalFromObjectToWorldSpace(shading_normal));

	const bool front_face = dot(ray_dir, shading_normal) < 0.0f;
	const float3 normal = front_face ? shading_normal : -shading_normal;

	optixSetPayload_0((unsigned int)tri.materialIdx);
	optixSetPayload_1(__float_as_uint(hit_point.x));
	optixSetPayload_2(__float_as_uint(hit_point.y));
	optixSetPayload_3(__float_as_uint(hit_point.z));
	optixSetPayload_4(__float_as_uint(normal.x));
	optixSetPayload_5(__float_as_uint(normal.y));
	optixSetPayload_6(__float_as_uint(normal.z));
}

// No-op: leaves the payload exactly as trace_probe_ray() initialised it
// (materialIdx = -1, the "no hit" sentinel bssrdf_probe_walk() checks for).
extern "C" __global__ void __miss__probe() {}
