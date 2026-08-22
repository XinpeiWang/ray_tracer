// optix_intersection_disk_cylinder.h -- Disk/Cylinder intersection +
// closest-hit programs. Included by optix_programs.cu.
//
// Both shapes carry their own object<->world transform (DiskData::o2w/w2o,
// CylinderData::o2w/w2o - see optix_types.h's own comment on why, mirroring
// src/TheRestOfYourLife/disk_cylinder_hittable.h's CPU technique) rather than
// being baked to world space or routed through an OptiX instance transform.
// The intersection programs below carry the RAY into object space by hand
// (dc_apply_point/dc_apply_vector) and solve there; the closest-hit programs
// need no reported attributes at all to recover the hit - unlike sphere/quad,
// which report the data their closest-hit needs because it's cheaper than
// recomputing it, here the world-space hit point (ray_orig + t*ray_dir, using
// the ORIGINAL world ray, not the object-space one) is already exactly
// recoverable from optixGetRayTmax() alone, and re-deriving the object-space
// point from it via dc_apply_point(w2o, hit_point) costs less than widening
// the attribute budget - a genuine algebraic identity, not an approximation:
// since intersection solved t against ro=w2o*world_orig and rd=w2o*world_dir
// (both linear maps, no renormalisation), o2w*(ro+t*rd) == world_orig +
// t*world_dir exactly.

// Row-major 3x4 affine transform, dropping the implicit [0,0,0,1] bottom row
// - same convention as SceneData::InstancePlacementGPU::transform.
__device__ __forceinline__ float3 dc_apply_point(const float m[12], const float3& p) {
	return make_float3(
		m[0] * p.x + m[1] * p.y + m[2]  * p.z + m[3],
		m[4] * p.x + m[5] * p.y + m[6]  * p.z + m[7],
		m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]);
}

// A direction: the translation column is deliberately not applied. NOT
// renormalised on purpose (see this file's header comment and disk_cylinder_
// hittable.h's own apply_vector) - that's what lets t survive unmodified
// between object and world space.
__device__ __forceinline__ float3 dc_apply_vector(const float m[12], const float3& v) {
	return make_float3(
		m[0] * v.x + m[1] * v.y + m[2]  * v.z,
		m[4] * v.x + m[5] * v.y + m[6]  * v.z,
		m[8] * v.x + m[9] * v.y + m[10] * v.z);
}

// Object -> world for a normal is the TRANSPOSE of world -> object (same
// reasoning as disk_cylinder_hittable.h's apply_normal), so this reads w2o
// column-wise rather than taking o2w row-wise.
__device__ __forceinline__ float3 dc_apply_normal_from_w2o(const float w2o[12], const float3& n) {
	return make_float3(
		w2o[0] * n.x + w2o[4] * n.y + w2o[8]  * n.z,
		w2o[1] * n.x + w2o[5] * n.y + w2o[9]  * n.z,
		w2o[2] * n.x + w2o[6] * n.y + w2o[10] * n.z);
}

//==============================================================================
// Disk
//==============================================================================

extern "C" __global__ void __intersection__disk() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const DiskData& disk = params.disks[primIdx];

	const float3 ray_orig_w = optixGetWorldRayOrigin();
	const float3 ray_dir_w  = optixGetWorldRayDirection();
	const float ray_tmin = optixGetRayTmin();
	const float ray_tmax = optixGetRayTmax();

	const float3 ro = dc_apply_point(disk.w2o, ray_orig_w);
	const float3 rd = dc_apply_vector(disk.w2o, ray_dir_w);

	if (rd.z == 0.0f) return;  // Ray parallel to the disk's plane
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

extern "C" __global__ void __closesthit__disk() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const DiskData& disk = params.disks[primIdx];
	const int matIdx = disk.materialIdx;
	const MaterialData& mat = params.materials[matIdx];

	const float t = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t * ray_dir;

	// A disk is flat: its object-space normal is the constant +Z everywhere
	// on its surface, unlike Cylinder's (see __closesthit__cylinder), which
	// varies with hit position.
	const float3 obj_normal = make_float3(0.0f, 0.0f, 1.0f);
	float3 outward_normal = normalize(dc_apply_normal_from_w2o(disk.w2o, obj_normal));
	const bool front_face = dot(ray_dir, outward_normal) < 0.0f;
	const float3 normal = front_face ? outward_normal : -outward_normal;

	unsigned int seed = optixGetPayload_9();
	float3 emission = material_emission(mat, front_face);

	float3 attenuation;
	float3 scattered_dir;
	bool scattered = false;
	bool is_specular = false;
	float brdf_pdf_override = -1.0f;
	bool bssrdf_exit = false;
	float3 bssrdf_exit_pos = make_float3(0.0f, 0.0f, 0.0f);

	// Same scope boundary as quad's own closest-hit: Medium/DielectricMedium/
	// CloudMedium/RgbGridMedium/Hair/Principled/NormalMappedLambertian need
	// shape-specific handling this file doesn't implement (the pbrt loader
	// never assigns Medium/DielectricMedium to a disk/cylinder in the first
	// place - see pbrt_gpu_builder.h's disk/cylinder loop - so this trap is
	// unreachable today, but kept loud rather than silently wrong if that
	// ever changes).
	if (material_requires_sphere_only_handling(mat.type) ||
		mat.type == MaterialType::NormalMappedLambertian) {
		printf("[DISK-SHADE] MaterialType %d is not supported on disk geometry\n", (int)mat.type);
		__trap();
	}

	float out_eta;
	shade_material(mat, matIdx, normal, ray_dir, hit_point, front_face, 0.0f, 0.0f, seed,
		attenuation, scattered_dir, scattered, is_specular, brdf_pdf_override, emission,
		bssrdf_exit, bssrdf_exit_pos, out_eta);

	optixSetPayload_3(__float_as_uint(emission.x));
	optixSetPayload_4(__float_as_uint(emission.y));
	optixSetPayload_5(__float_as_uint(emission.z));
	optixSetPayload_9(seed);

	{
		float3 albedoAov = scattered ? attenuation : mat.albedo;
		pack_aov_payload(albedoAov, normal);
	}
	optixSetPayload_22(__float_as_uint(out_eta));  // pbrt-v4 etaScale - see PathTracingPayload::eta

	if (scattered) {
		float t_hit = optixGetRayTmax();
		float brdf_pdf_out = is_specular ? 0.0f
						  : (brdf_pdf_override >= 0.0f ? brdf_pdf_override : cosine_pdf(scattered_dir, normal));

		optixSetPayload_0(__float_as_uint(attenuation.x));
		optixSetPayload_1(__float_as_uint(attenuation.y));
		optixSetPayload_2(__float_as_uint(attenuation.z));
		optixSetPayload_6(__float_as_uint(scattered_dir.x));
		optixSetPayload_7(__float_as_uint(scattered_dir.y));
		optixSetPayload_8(__float_as_uint(scattered_dir.z));
		optixSetPayload_10(bssrdf_exit ? 3 : 1);
		optixSetPayload_11(__float_as_uint(t_hit));
		optixSetPayload_12(__float_as_uint(brdf_pdf_out));
		if (bssrdf_exit) {
			optixSetPayload_13(__float_as_uint(bssrdf_exit_pos.x));
			optixSetPayload_14(__float_as_uint(bssrdf_exit_pos.y));
			optixSetPayload_15(__float_as_uint(bssrdf_exit_pos.z));
		}
	} else if (mat.type == MaterialType::DiffuseLight) {
		// Disk/Cylinder aren't registered in the NEE light list yet (see
		// optix_types.h's DiskData/CylinderData comment and PBRT_SUPPORT.md) -
		// light_pdf_for_incoming stays 0, same "hit but not aimed-at" tier as
		// any emissive shape this loader doesn't add to the alias table.
		optixSetPayload_10(2);
		optixSetPayload_12(0);
	} else {
		optixSetPayload_10(0);
		optixSetPayload_12(0);
	}
}

//==============================================================================
// Cylinder
//==============================================================================

// One candidate root: in range, inside the object-space Z clip, and inside
// the phi sweep. Mirrors CylinderShape<T>::intersect's check() lambda
// (src/shared/shapes.h) exactly, ported to float/CUDA math rather than
// instantiating that template device-side - see this project's own precedent
// for why (optix_intersection_sphere.h's gpu_cloud_density comment).
__device__ __forceinline__ bool dc_cylinder_check(const CylinderData& cyl, const float3& ro, const float3& rd,
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

extern "C" __global__ void __intersection__cylinder() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const CylinderData& cyl = params.cylinders[primIdx];

	const float3 ray_orig_w = optixGetWorldRayOrigin();
	const float3 ray_dir_w  = optixGetWorldRayDirection();
	const float ray_tmin = optixGetRayTmin();
	const float ray_tmax = optixGetRayTmax();

	const float3 ro = dc_apply_point(cyl.w2o, ray_orig_w);
	const float3 rd = dc_apply_vector(cyl.w2o, ray_dir_w);

	// Stable quadratic solve in double precision (matches pbrt-v4's own
	// Cylinder::BasicIntersect and this project's CPU CylinderShape<T> -
	// avoids the catastrophic cancellation a naive b*b-4ac has for a thin
	// cylinder seen nearly edge-on).
	const double da = (double)rd.x, db = (double)rd.y;
	const double oa = (double)ro.x, ob = (double)ro.y;
	const double a = da * da + db * db;
	const double b = 2.0 * (oa * da + ob * db);
	const double c = oa * oa + ob * ob - (double)cyl.radius * (double)cyl.radius;
	if (a == 0.0) return;  // Ray parallel to the cylinder's axis
	const double f = b / (2.0 * a);
	const double vx = oa - f * da, vy = ob - f * db;
	const double len_v = sqrt(vx * vx + vy * vy);
	const double discrim = 4.0 * a * ((double)cyl.radius + len_v) * ((double)cyl.radius - len_v);
	if (discrim < 0.0) return;
	const double sqrt_disc = sqrt(discrim);
	const double q = (b < 0.0) ? -0.5 * (b - sqrt_disc) : -0.5 * (b + sqrt_disc);
	float t0 = (float)(q / a), t1 = (float)(c / q);
	if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
	if (t0 > ray_tmax || t1 < ray_tmin) return;

	float t = t0;
	if (!dc_cylinder_check(cyl, ro, rd, t0, ray_tmin, ray_tmax)) {
		t = t1;
		if (!dc_cylinder_check(cyl, ro, rd, t1, ray_tmin, ray_tmax)) return;
	}

	optixReportIntersection(t, 0, 0, 0, 0, 0);
}

extern "C" __global__ void __closesthit__cylinder() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const CylinderData& cyl = params.cylinders[primIdx];
	const int matIdx = cyl.materialIdx;
	const MaterialData& mat = params.materials[matIdx];

	const float t = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t * ray_dir;

	// Cylinder's normal varies with hit position (radial, from the axis) -
	// unlike disk's constant one - so the object-space hit point has to be
	// recovered first (see this file's header comment for why re-deriving it
	// from the world hit point is exact, not approximate).
	const float3 obj_hit = dc_apply_point(cyl.w2o, hit_point);
	const float obj_hit_len = sqrtf(obj_hit.x * obj_hit.x + obj_hit.y * obj_hit.y);
	const float3 obj_normal = (obj_hit_len > 1e-8f)
		? make_float3(obj_hit.x / obj_hit_len, obj_hit.y / obj_hit_len, 0.0f)
		: make_float3(1.0f, 0.0f, 0.0f);
	float3 outward_normal = normalize(dc_apply_normal_from_w2o(cyl.w2o, obj_normal));
	const bool front_face = dot(ray_dir, outward_normal) < 0.0f;
	const float3 normal = front_face ? outward_normal : -outward_normal;

	unsigned int seed = optixGetPayload_9();
	float3 emission = material_emission(mat, front_face);

	float3 attenuation;
	float3 scattered_dir;
	bool scattered = false;
	bool is_specular = false;
	float brdf_pdf_override = -1.0f;
	bool bssrdf_exit = false;
	float3 bssrdf_exit_pos = make_float3(0.0f, 0.0f, 0.0f);

	if (material_requires_sphere_only_handling(mat.type) ||
		mat.type == MaterialType::NormalMappedLambertian) {
		printf("[CYLINDER-SHADE] MaterialType %d is not supported on cylinder geometry\n", (int)mat.type);
		__trap();
	}

	float out_eta;
	shade_material(mat, matIdx, normal, ray_dir, hit_point, front_face, 0.0f, 0.0f, seed,
		attenuation, scattered_dir, scattered, is_specular, brdf_pdf_override, emission,
		bssrdf_exit, bssrdf_exit_pos, out_eta);

	optixSetPayload_3(__float_as_uint(emission.x));
	optixSetPayload_4(__float_as_uint(emission.y));
	optixSetPayload_5(__float_as_uint(emission.z));
	optixSetPayload_9(seed);

	{
		float3 albedoAov = scattered ? attenuation : mat.albedo;
		pack_aov_payload(albedoAov, normal);
	}
	optixSetPayload_22(__float_as_uint(out_eta));  // pbrt-v4 etaScale - see PathTracingPayload::eta

	if (scattered) {
		float t_hit = optixGetRayTmax();
		float brdf_pdf_out = is_specular ? 0.0f
						  : (brdf_pdf_override >= 0.0f ? brdf_pdf_override : cosine_pdf(scattered_dir, normal));

		optixSetPayload_0(__float_as_uint(attenuation.x));
		optixSetPayload_1(__float_as_uint(attenuation.y));
		optixSetPayload_2(__float_as_uint(attenuation.z));
		optixSetPayload_6(__float_as_uint(scattered_dir.x));
		optixSetPayload_7(__float_as_uint(scattered_dir.y));
		optixSetPayload_8(__float_as_uint(scattered_dir.z));
		optixSetPayload_10(bssrdf_exit ? 3 : 1);
		optixSetPayload_11(__float_as_uint(t_hit));
		optixSetPayload_12(__float_as_uint(brdf_pdf_out));
		if (bssrdf_exit) {
			optixSetPayload_13(__float_as_uint(bssrdf_exit_pos.x));
			optixSetPayload_14(__float_as_uint(bssrdf_exit_pos.y));
			optixSetPayload_15(__float_as_uint(bssrdf_exit_pos.z));
		}
	} else if (mat.type == MaterialType::DiffuseLight) {
		optixSetPayload_10(2);
		optixSetPayload_12(0);
	} else {
		optixSetPayload_10(0);
		optixSetPayload_12(0);
	}
}
