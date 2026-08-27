// wavefront_intersection_disk_cylinder.h -- Disk + cylinder intersection + closest-hit programs
// Included by wavefront_programs.cu, after wavefront_common.h.

// ============================================================================
// __intersection__wf_disk / __intersection__wf_cylinder,
// __closesthit__wf_disk / __closesthit__wf_cylinder (Phase 4c)
//
// Wavefront-native duplicates of the recursive backend's
// optix_intersection_disk_cylinder.h (see that file's own header comment for
// the full design rationale: both shapes carry their own object<->world
// transform - DiskData::o2w/w2o, CylinderData::o2w/w2o - applied BY HAND
// here rather than through OptiX's per-GAS-instance transform, since
// neither shape is instanced). Duplicated rather than shared because
// wavefront_programs.cu and optix_programs.cu are two separate OptiX
// modules (see this file's own top-of-file header comment on why every
// other wf_ intersection program here is likewise a native copy, not
// cross-module reuse).
//
// Unlike the recursive backend's __closesthit__disk/__closesthit__cylinder,
// these don't call shade_material() - they only fill WfHitPayload, exactly
// like __closesthit__wf_quad above; evaluate_materials() (wavefront_
// kernels.cu) shades the hit later from the queued HitWorkItem.
//
// wf_dc_apply_point()/wf_dc_apply_vector()/wf_dc_apply_normal_from_w2o()
// (the object<->world transform helpers both shapes need) are defined
// earlier in this file, right after packPointer()/unpackPointer() - before
// the "#include wavefront_probe.h" line, since that file's own probe-disk/
// probe-cylinder closest-hit programs need them too.
// ============================================================================

extern "C" __global__ void __intersection__wf_disk() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const DiskData& disk = wf_params.disks[primIdx];

	const float3 ray_orig_w = optixGetWorldRayOrigin();
	const float3 ray_dir_w  = optixGetWorldRayDirection();
	const float ray_tmin = optixGetRayTmin();
	const float ray_tmax = optixGetRayTmax();

	const float3 ro = wf_dc_apply_point(disk.w2o, ray_orig_w);
	const float3 rd = wf_dc_apply_vector(disk.w2o, ray_dir_w);

	if (rd.z == 0.0f) return;
	const float t = (disk.height - ro.z) / rd.z;
	if (t < ray_tmin || t > ray_tmax) return;

	const float hx = ro.x + t * rd.x;
	const float hy = ro.y + t * rd.y;
	const float dist2 = hx * hx + hy * hy;
	if (dist2 > disk.radius * disk.radius || dist2 < disk.innerRadius * disk.innerRadius) return;

	float phi = atan2f(hy, hx);
	if (phi < 0.0f) phi += 6.283185307179586f;
	if (phi > disk.phiMax) return;

	optixReportIntersection(t, 0, 0, 0, 0, 0);
}

extern "C" __global__ void __closesthit__wf_disk() {
	WfHitPayload* payload = (WfHitPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	const unsigned int primIdx = optixGetPrimitiveIndex();
	const DiskData& disk = wf_params.disks[primIdx];

	const float t_hit = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t_hit * ray_dir;

	// A disk is flat: its object-space normal is the constant +Z everywhere
	// on its surface, unlike Cylinder's (see __closesthit__wf_cylinder),
	// which varies with hit position.
	const float3 obj_normal = make_float3(0.0f, 0.0f, 1.0f);
	float3 outward_normal = normalize(wf_dc_apply_normal_from_w2o(disk.w2o, obj_normal));
	const bool front_face = dot(ray_dir, outward_normal) < 0.0f;
	const float3 normal = front_face ? outward_normal : -outward_normal;

	payload->hitPoint    = hit_point;
	payload->normal      = normal;
	payload->t           = t_hit;
	payload->materialIdx = disk.materialIdx;
	payload->geomType    = 4;
	payload->hit         = true;
	payload->mediumTFar  = 0.0f;
	payload->frontFace   = front_face ? 1 : 0;
	// Real UV (phi/phiMax, radial fraction) - same formula the recursive
	// backend's __closesthit__disk recomputes from hit_point, matching CPU
	// DiskShape<T>::intersect's own u=phi/phi_max, v=1-(dist-inner_r)/
	// (outer_r-inner_r) convention (shapes.h:577-582). Was previously always
	// (0,0) here, a real gap vs. the recursive backend.
	{
		const float3 obj_hit_uv = wf_dc_apply_point(disk.w2o, hit_point);
		float uv_phi = atan2f(obj_hit_uv.y, obj_hit_uv.x);
		if (uv_phi < 0.0f) uv_phi += 6.283185307179586f;
		const float uv_dist = sqrtf(obj_hit_uv.x * obj_hit_uv.x + obj_hit_uv.y * obj_hit_uv.y);
		payload->uv_u = uv_phi / disk.phiMax;
		payload->uv_v = (disk.radius > disk.innerRadius)
			? 1.0f - (uv_dist - disk.innerRadius) / (disk.radius - disk.innerRadius)
			: 0.0f;
		// Real analytic dpdu, matching CPU's disk_cylinder_hittable.h and
		// the recursive backend's identical derivation (optix_intersection_
		// disk_cylinder.h) - object-space dpdu = phiMax*(-hy,hx,0), carried
		// to world space via o2w as a genuine tangent VECTOR (wf_dc_apply_
		// vector, NOT the inverse-transpose normal transform outward_normal
		// above uses). Matches quad/sphere's identical use of this "dual-
		// purpose carrier" field.
		payload->objNormal = wf_dc_apply_vector(disk.o2w,
			make_float3(-disk.phiMax * obj_hit_uv.y, disk.phiMax * obj_hit_uv.x, 0.0f));
	}
}

// One candidate root: in range, inside the object-space Z clip, and inside
// the phi sweep. A free device function rather than a lambda, matching this
// file's own established style (no CUDA extended-lambda dependency).
__device__ __forceinline__ bool wf_dc_cylinder_check(const CylinderData& cyl, const float3& ro, const float3& rd,
													   float t, float ray_tmin, float ray_tmax) {
	if (t < ray_tmin || t > ray_tmax) return false;
	const float hz = ro.z + t * rd.z;
	if (hz < cyl.zMin || hz > cyl.zMax) return false;
	const float hx = ro.x + t * rd.x;
	const float hy = ro.y + t * rd.y;
	float phi = atan2f(hy, hx);
	if (phi < 0.0f) phi += 6.283185307179586f;
	return phi <= cyl.phiMax;
}

extern "C" __global__ void __intersection__wf_cylinder() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const CylinderData& cyl = wf_params.cylinders[primIdx];

	const float3 ray_orig_w = optixGetWorldRayOrigin();
	const float3 ray_dir_w  = optixGetWorldRayDirection();
	const float ray_tmin = optixGetRayTmin();
	const float ray_tmax = optixGetRayTmax();

	const float3 ro = wf_dc_apply_point(cyl.w2o, ray_orig_w);
	const float3 rd = wf_dc_apply_vector(cyl.w2o, ray_dir_w);

	// Stable quadratic solve in double precision - see optix_intersection_
	// disk_cylinder.h's own __intersection__cylinder for why (avoids
	// catastrophic cancellation for a thin cylinder seen nearly edge-on) -
	// factored into wf_dc_solve_tube_quadratic (this file), shared with the
	// Medium closest-hit case below.
	float t0, t1;
	if (!wf_dc_solve_tube_quadratic(ro, rd, cyl.radius, t0, t1)) return;
	if (t0 > ray_tmax || t1 < ray_tmin) return;

	float t = t0;
	if (!wf_dc_cylinder_check(cyl, ro, rd, t0, ray_tmin, ray_tmax)) {
		t = t1;
		if (!wf_dc_cylinder_check(cyl, ro, rd, t1, ray_tmin, ray_tmax)) return;
	}

	optixReportIntersection(t, 0, 0, 0, 0, 0);
}

extern "C" __global__ void __closesthit__wf_cylinder() {
	WfHitPayload* payload = (WfHitPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	const unsigned int primIdx = optixGetPrimitiveIndex();
	const CylinderData& cyl = wf_params.cylinders[primIdx];

	const float t_hit = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t_hit * ray_dir;

	// Cylinder's normal varies with hit position (radial, from the axis) -
	// recover the object-space hit point to compute it, same technique as
	// the recursive backend's __closesthit__cylinder.
	const float3 obj_hit = wf_dc_apply_point(cyl.w2o, hit_point);
	const float obj_hit_len = sqrtf(obj_hit.x * obj_hit.x + obj_hit.y * obj_hit.y);
	const float3 obj_normal = (obj_hit_len > 1e-8f)
		? make_float3(obj_hit.x / obj_hit_len, obj_hit.y / obj_hit_len, 0.0f)
		: make_float3(1.0f, 0.0f, 0.0f);
	float3 outward_normal = normalize(wf_dc_apply_normal_from_w2o(cyl.w2o, obj_normal));
	const bool front_face = dot(ray_dir, outward_normal) < 0.0f;
	const float3 normal = front_face ? outward_normal : -outward_normal;

	payload->hitPoint    = hit_point;
	payload->normal      = normal;
	payload->t           = t_hit;
	payload->materialIdx = cyl.materialIdx;
	payload->geomType    = 5;
	payload->hit         = true;
	payload->mediumTFar  = 0.0f;
	payload->frontFace   = front_face ? 1 : 0;
	// Real UV (phi/phiMax, z-fraction) - same formula the recursive backend's
	// __closesthit__cylinder recomputes from obj_hit, matching CPU
	// CylinderShape<T>'s own u=phi/phi_max, v=(hz-z_min)/(z_max-z_min)
	// convention (shapes.h:760). Was previously always (0,0) here, a real
	// gap vs. the recursive backend.
	{
		float uv_phi = atan2f(obj_hit.y, obj_hit.x);
		if (uv_phi < 0.0f) uv_phi += 6.283185307179586f;
		payload->uv_u = uv_phi / cyl.phiMax;
		payload->uv_v = (cyl.zMax > cyl.zMin)
			? (obj_hit.z - cyl.zMin) / (cyl.zMax - cyl.zMin)
			: 0.0f;
	}
	// Real analytic dpdu, matching disk's identical phi-tangent form (the
	// axis doesn't affect the azimuthal tangent direction) and the
	// recursive backend's identical derivation (optix_intersection_
	// disk_cylinder.h).
	payload->objNormal = wf_dc_apply_vector(cyl.o2w,
		make_float3(-cyl.phiMax * obj_hit.y, cyl.phiMax * obj_hit.x, 0.0f));

	// MaterialType::Medium: override with the entry (near) / exit (far)
	// roots, matching __closesthit__wf_sphere's own needsNearFar block and
	// the recursive backend's identical __closesthit__cylinder Medium case
	// (optix_intersection_disk_cylinder.h - see that comment for the tube-
	// quadric-clipped-to-a-z-slab derivation and its phi-sweep scope limit).
	// Object space, since CylinderData::zMin/zMax are object-space.
	const MaterialData& cyl_mat = wf_params.materials[cyl.materialIdx];
	if (cyl_mat.type == MaterialType::Medium) {
		const float3 ro = wf_dc_apply_point(cyl.w2o, ray_orig);
		const float3 rd = wf_dc_apply_vector(cyl.w2o, ray_dir);
		// Ray parallel to the axis handled directly here, NOT via
		// wf_dc_solve_tube_quadratic() below - see the recursive backend's
		// identical __closesthit__cylinder Medium case (optix_intersection_
		// disk_cylinder.h) for why: that function's a==0 case answers a
		// SURFACE-crossing question (always no), the wrong question for
		// this VOLUME/interval test.
		float tube_t0 = -1e30f, tube_t1 = 1e30f;
		bool hasTube;
		if (rd.x == 0.0f && rd.y == 0.0f) {
			hasTube = (double)ro.x * ro.x + (double)ro.y * ro.y <= (double)cyl.radius * (double)cyl.radius;
		} else {
			hasTube = wf_dc_solve_tube_quadratic(ro, rd, cyl.radius, tube_t0, tube_t1);
		}

		float z_t0 = -1e30f, z_t1 = 1e30f;
		bool hasZSlab = true;
		if (rd.z == 0.0f) {
			hasZSlab = (ro.z >= cyl.zMin && ro.z <= cyl.zMax);
		} else {
			float za = (cyl.zMin - ro.z) / rd.z;
			float zb = (cyl.zMax - ro.z) / rd.z;
			z_t0 = fminf(za, zb);
			z_t1 = fmaxf(za, zb);
		}

		float t_near = fmaxf(0.0f, fmaxf(tube_t0, z_t0));
		float t_far  = fminf(tube_t1, z_t1);
		if (!hasTube || !hasZSlab || t_far < t_near) { t_near = 0.0f; t_far = 0.0f; }

		float3 unit_dir = normalize(ray_dir);
		payload->t          = t_near;
		payload->hitPoint   = ray_orig + t_near * unit_dir;
		payload->mediumTFar = t_far;
	}
}

