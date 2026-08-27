// wavefront_intersection_triangle.h -- Triangle intersection + any-hit (alpha cutout) + closest-hit programs
// Included by wavefront_programs.cu, after wavefront_common.h.

// ============================================================================
// __intersection__wf_triangle / __closesthit__wf_triangle
//
// Wavefront-native duplicate of optix_intersection_triangle.h's
// __intersection__triangle (same watertight Woop/Moller-Trumbore math,
// wf_params instead of the recursive path's params global) - same cross-
// module payload-type mismatch reason as wf_sphere/wf_quad/wf_bilinear_patch
// above. Flat shading only (no per-vertex normals - see TriangleData).
// ============================================================================

extern "C" __global__ void __intersection__wf_triangle() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const TriangleData& tri = wf_params.triangles[wf_prim_base(wf_instance_base()) + primIdx];

	// Object space, for the same reason as __intersection__wf_sphere above -
	// identical to world space on the non-instanced path, since those
	// instances carry an identity transform.
	const float3 ro = optixGetObjectRayOrigin();
	const float3 rd = optixGetObjectRayDirection();
	const float ray_tmin = optixGetRayTmin();
	const float ray_tmax = optixGetRayTmax();

	float3 p0t = tri.p0 - ro;
	float3 p1t = tri.p1 - ro;
	float3 p2t = tri.p2 - ro;

	const float ax = fabsf(rd.x), ay = fabsf(rd.y), az = fabsf(rd.z);
	int kz, kx, ky;
	if (ax >= ay && ax >= az)      { kz = 0; kx = 1; ky = 2; }
	else if (ay >= ax && ay >= az) { kz = 1; kx = 2; ky = 0; }
	else                            { kz = 2; kx = 0; ky = 1; }

	auto comp = [](const float3& v, int k) -> float {
		return k == 0 ? v.x : (k == 1 ? v.y : v.z);
	};
	const float Dx = comp(rd, kx), Dy = comp(rd, ky), Dz = comp(rd, kz);
	float p0x = comp(p0t, kx), p0y = comp(p0t, ky), p0z = comp(p0t, kz);
	float p1x = comp(p1t, kx), p1y = comp(p1t, ky), p1z = comp(p1t, kz);
	float p2x = comp(p2t, kx), p2y = comp(p2t, ky), p2z = comp(p2t, kz);

	const float Sx = -Dx / Dz, Sy = -Dy / Dz, Sz = 1.0f / Dz;
	p0x += Sx * p0z; p0y += Sy * p0z;
	p1x += Sx * p1z; p1y += Sy * p1z;
	p2x += Sx * p2z; p2y += Sy * p2z;

	const float e0 = p1x*p2y - p1y*p2x;
	const float e1 = p2x*p0y - p2y*p0x;
	const float e2 = p0x*p1y - p0y*p1x;

	if ((e0 < 0.0f || e1 < 0.0f || e2 < 0.0f) && (e0 > 0.0f || e1 > 0.0f || e2 > 0.0f))
		return;
	const float det = e0 + e1 + e2;
	if (det == 0.0f) return;

	p0z *= Sz; p1z *= Sz; p2z *= Sz;
	const float tScaled = e0*p0z + e1*p1z + e2*p2z;
	if (det < 0.0f && (tScaled >= 0.0f || tScaled < ray_tmax * det)) return;
	if (det > 0.0f && (tScaled <= 0.0f || tScaled > ray_tmax * det)) return;

	const float invDet = 1.0f / det;
	const float t = tScaled * invDet;
	if (t < ray_tmin || t > ray_tmax) return;

	// Barycentric weights for p0/p1/p2, same convention optixGetTriangleBarycentrics()
	// uses on the recursive (native-triangle) path: b1 weights p1, b2 weights
	// p2, b0=1-b1-b2 weights p0 (see optix_intersection_triangle.h's comment).
	// e0/e1/e2 above are already proportional to the point-vs-opposite-edge
	// signed areas (this IS a Möller-Trumbore/Woop-style barycentric test),
	// so dividing by their sum (det) gives the weights directly - no extra
	// work, just exposing numbers this program already computed. These two
	// attribute slots were unused (hardcoded 0) before; not a payload/ABI
	// change, since attribute count is fixed by pipelineCompileOptions
	// regardless of how many of the 4 a given intersection program fills in.
	const float b1 = e1 * invDet;
	const float b2 = e2 * invDet;
	optixReportIntersection(t, 0, __float_as_int(b1), __float_as_int(b2), 0, 0);
}

// Alpha-cutout for radiance rays (MaterialData::alphaMaskTexIdx) - runs
// BEFORE closest-hit is decided, which is exactly when a transparent pixel
// needs to be skipped so it can't win as "the" hit in the first place.
// Registered on the same hit group as __intersection__wf_triangle/
// __closesthit__wf_triangle above (wavefront_path_tracer.cpp's triHitDesc
// now sets moduleAH too), not a new program group. Uses
// optixGetAttribute_0/1() (this hit group's own custom-intersection
// barycentrics), NOT optixGetTriangleBarycentrics() - that's only valid
// for OptiX's built-in triangle intersection, which the recursive path
// uses but this wavefront-native one does not (see the header comment
// above). A no-op for the overwhelming majority of triangles, whose
// material has no alpha mask.
extern "C" __global__ void __anyhit__wf_triangle() {
	const int instBase = wf_instance_base();
	const int primIdx = (int)(wf_prim_base(instBase) + optixGetPrimitiveIndex());
	const TriangleData& tri = wf_params.triangles[primIdx];
	// Cheap common-case gate BEFORE paying for hit_point/Mix-resolution -
	// see __anyhit__triangle's identical gate (optix_intersection_triangle.h)
	// for the full reasoning (this program runs per candidate intersection,
	// not just the closest hit, and the overwhelming majority of triangles
	// are neither Mix nor alpha-masked).
	const MaterialData& raw_mat = wf_params.materials[tri.materialIdx];
	if (raw_mat.type != MaterialType::Mix && raw_mat.alphaMaskTexIdx < 0) return;

	// Mix must resolve before alphaMaskTexIdx is read - a Mix wrapper's own
	// alphaMaskTexIdx is always -1 (never authored directly on it), so
	// checking it unresolved would silently skip a sub-material's real alpha
	// cutout. See __anyhit__triangle's identical fix (optix_intersection_
	// triangle.h) and MaterialType::Mix's own comment (optix_types.h).
	const float wf_t = optixGetRayTmax();
	const float3 wf_hit_point = optixGetWorldRayOrigin() + wf_t * optixGetWorldRayDirection();
	int matIdx = tri.materialIdx;
	const MaterialData mat = wf_resolve_mix_material(wf_params.materials[matIdx], matIdx, wf_hit_point, matIdx);
	if (mat.alphaMaskTexIdx < 0) return;

	// Real UV when authored, else the same barycentric fallback CPU's own
	// triangle.h uses (rec.u=b1,rec.v=b2) - see __closesthit__wf_triangle's
	// own comment for why a fixed (0,0) here was a real bug, not just a
	// cosmetic gap.
	const float b1 = __int_as_float(optixGetAttribute_0());
	const float b2 = __int_as_float(optixGetAttribute_1());
	const float b0 = 1.0f - b1 - b2;
	float uv_u = b1, uv_v = b2;
	if (tri.hasUVs) {
		uv_u = b0 * tri.uv0.x + b1 * tri.uv1.x + b2 * tri.uv2.x;
		uv_v = b0 * tri.uv0.y + b1 * tri.uv1.y + b2 * tri.uv2.y;
	}
	if (!wf_passes_alpha_cutout(mat.alphaMaskTexIdx, uv_u, uv_v))
		optixIgnoreIntersection();
}

extern "C" __global__ void __closesthit__wf_triangle() {
	WfHitPayload* payload = (WfHitPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	const int instBase = wf_instance_base();
	const int primIdx = (int)(wf_prim_base(instBase) + optixGetPrimitiveIndex());
	const TriangleData& tri = wf_params.triangles[primIdx];

	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float  t_hit    = optixGetRayTmax();

	float3 hit_point = ray_orig + t_hit * ray_dir;

	// The vertices are in the definition's object space for an instanced
	// triangle, so the normal derived from them has to be carried to world
	// space before it can be compared against the world ray. Skipped entirely
	// when instBase < 0.
	float3 geom_normal = normalize(cross(tri.p1 - tri.p0, tri.p2 - tri.p0));
	if (instBase >= 0)
		geom_normal = normalize(optixTransformNormalFromObjectToWorldSpace(geom_normal));
	bool front_face = dot(ray_dir, geom_normal) < 0.0f;
	float3 normal = front_face ? geom_normal : -geom_normal;

	// Barycentric-interpolated UV, matching optix_intersection_triangle.h's
	// closesthit exactly - tri.hasUVs gates it the same way tri.hasNormals
	// gates smooth-normal interpolation there (this program stays flat-shaded
	// regardless, see this function's own header comment; UV is independent
	// of that and is the one piece MaterialType::Lambertian's textureIdx
	// actually needs). Falls back to the barycentric weights themselves
	// (matching CPU triangle.h's own rec.u=b1,rec.v=b2) rather than a fixed
	// (0,0) when the mesh has no real UV - a fixed value here sampled the
	// exact same texel for every point on the whole mesh, a real "solid
	// black on GPU" bug for an untextured-UV mesh with a real texture bound
	// (docs/PBRT_SUPPORT.md tracked this).
	const float b1 = __int_as_float(optixGetAttribute_0());
	const float b2 = __int_as_float(optixGetAttribute_1());
	const float b0 = 1.0f - b1 - b2;
	float uv_u = b1, uv_v = b2;
	if (tri.hasUVs) {
		uv_u = b0 * tri.uv0.x + b1 * tri.uv1.x + b2 * tri.uv2.x;
		uv_v = b0 * tri.uv0.y + b1 * tri.uv1.y + b2 * tri.uv2.y;
	}

	// Real UV-derived tangent (dpdu), for MaterialType::NormalMappedLambertian
	// - stashed in objDpdu (see HitWorkItem::objDpdu's own comment) since
	// WfHitPayload/HitWorkItem carry no per-vertex position/UV data of their
	// own to recompute this from later. Solves the standard 2x2 system
	// relating edge vectors to UV deltas (pbrt-v4 Triangle::Intersect),
	// mirroring optix_intersection_triangle.h's recursive-path equivalent
	// exactly. Computed unconditionally (cheap, pure math) rather than
	// gated on the hit material actually being normal-mapped.
	float3 tri_dpdu;
	{
		const float3 e1 = tri.p1 - tri.p0;
		if (tri.hasUVs) {
			const float3 e2 = tri.p2 - tri.p0;
			const float du1 = tri.uv1.x - tri.uv0.x, dv1 = tri.uv1.y - tri.uv0.y;
			const float du2 = tri.uv2.x - tri.uv0.x, dv2 = tri.uv2.y - tri.uv0.y;
			const float det = du1 * dv2 - dv1 * du2;
			if (fabsf(det) > 1e-12f) {
				const float invDet = 1.0f / det;
				tri_dpdu = (dv2 * invDet) * e1 - (dv1 * invDet) * e2;
				const float dpdu_len = length(tri_dpdu);
				tri_dpdu = (dpdu_len > 1e-8f) ? (tri_dpdu / dpdu_len) : normalize(e1);
			} else {
				tri_dpdu = normalize(e1);
			}
		} else {
			tri_dpdu = normalize(e1);
		}
		if (instBase >= 0) tri_dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(tri_dpdu));
	}

	payload->hitPoint    = hit_point;
	payload->normal      = normal;
	payload->t           = t_hit;
	payload->materialIdx = tri.materialIdx;
	payload->geomType    = 3;
	payload->hit         = true;
	payload->mediumTFar  = 0.0f;
	// See __closesthit__wf_quad's comment on frontFace - same missing
	// initialization here. RoughDielectric OBJ meshes (e.g. glass triangles)
	// are the case this actually affects.
	payload->frontFace   = front_face ? 1 : 0;
	payload->objDpdu     = tri_dpdu;
	payload->uv_u        = uv_u;
	payload->uv_v        = uv_v;
}

