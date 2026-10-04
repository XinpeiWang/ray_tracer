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
//
// dc_apply_point/dc_apply_vector/dc_apply_normal_from_w2o/dc_cylinder_check
// now live in optix_disk_cylinder_helpers.h (included earlier by optix_
// programs.cu, before optix_device_helpers.h - see that file's own header
// comment for why: sample_disk_light()/sample_cylinder_light() need them
// too, and optix_device_helpers.h is included before this file).

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

	const float t = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t * ray_dir;

	// Resolved to a real, non-Mix material before any mat.type branch below -
	// see MaterialType::Mix's own comment (optix_types.h).
	int matIdx = disk.materialIdx;
	const MaterialData mat = resolve_mix_material(params.materials[matIdx], matIdx, hit_point, matIdx);

	// A disk is flat: its object-space normal is the constant +Z everywhere
	// on its surface, unlike Cylinder's (see __closesthit__cylinder), which
	// varies with hit position.
	const float3 obj_normal = make_float3(0.0f, 0.0f, 1.0f);
	float3 outward_normal = normalize(dc_apply_normal_from_w2o(disk.w2o, obj_normal));
	const bool front_face = dot(ray_dir, outward_normal) < 0.0f;
	const float3 normal = front_face ? outward_normal : -outward_normal;

	unsigned int seed = optixGetPayload_9();

	// Real UV (phi/phiMax, radial fraction - CPU DiskShape<T>::intersect's own
	// u=phi/phi_max, v=1-(dist-inner_r)/(outer_r-inner_r) convention, shapes.h
	// :577-582) for a pbrt AreaLightSource "filename" disk light to sample its
	// image correctly instead of always reading texel (0,0) - recomputed here
	// from hit_point rather than a widened attribute budget, same "cheaper to
	// recompute than widen attributes" reasoning __intersection__disk's own
	// phi computation follows (this file, above) and __closesthit__quad's
	// alpha/beta UV follows (optix_intersection_quad.h).
	const float3 obj_hit_uv = dc_apply_point(disk.w2o, hit_point);
	float uv_phi = atan2f(obj_hit_uv.y, obj_hit_uv.x);
	if (uv_phi < 0.0f) uv_phi += 6.283185307179586f;
	const float uv_dist = sqrtf(obj_hit_uv.x * obj_hit_uv.x + obj_hit_uv.y * obj_hit_uv.y);
	const float uv_u = uv_phi / disk.phiMax;
	const float uv_v = (disk.radius > disk.innerRadius)
		? 1.0f - (uv_dist - disk.innerRadius) / (disk.radius - disk.innerRadius)
		: 0.0f;

	// Real analytic dpdu, matching CPU's disk_cylinder_hittable.h exactly:
	// object-space dpdu = phiMax * (-hy, hx, 0) (the azimuthal tangent at
	// this hit's object-space position, already computed above as
	// obj_hit_uv), carried to world space via the o2w matrix as a genuine
	// tangent VECTOR (dc_apply_vector, NOT the inverse-transpose normal
	// transform dc_apply_normal_from_w2o uses above). Gated on
	// material_needs_dpdu() - only NormalMappedLambertian and the 4
	// anisotropic material kinds ever read this.
	const float3 disk_dpdu = material_needs_dpdu(mat.type)
		? dc_apply_vector(disk.o2w,
			make_float3(-disk.phiMax * obj_hit_uv.y, disk.phiMax * obj_hit_uv.x, 0.0f))
		: make_float3(0.0f, 0.0f, 0.0f);

	float3 emission = material_emission(mat, front_face, uv_u, uv_v, hit_point);

	float3 attenuation;
	float3 scattered_dir;
	bool scattered = false;
	bool is_specular = false;
	bool is_medium_boundary = false;  // MaterialType::Interface - see optix_types.h
	float brdf_pdf_override = -1.0f;
	bool bssrdf_exit = false;
	float3 bssrdf_exit_pos = make_float3(0.0f, 0.0f, 0.0f);

	// Real Hair support on disk geometry, using the shading normal as the
	// fiber-tangent proxy - same simplification as sphere's own Hair branch.
	// Unlike Medium/DielectricMedium (which the pbrt loader genuinely never
	// assigns to a disk - MediumInterface stays sphere-only, see
	// pbrt_gpu_builder.h's disk/cylinder loop comment), Material "hair" +
	// Shape "disk" is an ordinary, reachable pbrt combination once Round 7
	// wired "hair" into the loader for real - trapping it here would have
	// been a genuine regression from "falls back to Lambertian", not the
	// unreachable-defensive-code every other trapped type here still is.
	float out_eta = 1.0f;
	// See PathTracingPayload's own p24/rgbChannel comment (optix_renderer_init.cpp)
	// and shade_material()'s inout_rgb_channel parameter comment - read
	// unconditionally so a channel already chosen by an earlier bounce
	// still survives unchanged through this one.
	unsigned int rgbChannel = optixGetPayload_24();
	if (mat.type == MaterialType::Hair) {
		scattered   = sample_hair_material(ray_dir, normal, mat, seed, scattered_dir, attenuation);
		is_specular = true;
	} else {
		// Same scope boundary as quad's own closest-hit: Medium/DielectricMedium/
		// CloudMedium/RgbGridMedium/Principled/NormalMappedLambertian need
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

		// Integrator "bool regularize" - see current_do_regularize()'s own
		// comment (optix_device_helpers.h).
		const bool do_regularize = current_do_regularize();
		shade_material(mat, matIdx, normal, ray_dir, hit_point, front_face, uv_u, uv_v, disk_dpdu, do_regularize, seed,
			attenuation, scattered_dir, scattered, is_specular, is_medium_boundary, brdf_pdf_override, emission,
			bssrdf_exit, bssrdf_exit_pos, out_eta, rgbChannel);
	}

	optixSetPayload_3(__float_as_uint(emission.x));
	optixSetPayload_4(__float_as_uint(emission.y));
	optixSetPayload_5(__float_as_uint(emission.z));
	optixSetPayload_9(seed);

	{
		float3 albedoAov = scattered ? attenuation : mat.albedo;
		pack_aov_payload(albedoAov, normal);
	}
	optixSetPayload_22(__float_as_uint(out_eta));  // pbrt-v4 etaScale - see PathTracingPayload::eta
	optixSetPayload_24(rgbChannel);  // dispersion channel - see shade_material()'s inout_rgb_channel comment

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
		optixSetPayload_10(pack_scatter_flag(bssrdf_exit, is_medium_boundary, is_specular));  // scattered (see pack_scatter_flag's own comment)
		optixSetPayload_11(__float_as_uint(t_hit));
		optixSetPayload_12(__float_as_uint(brdf_pdf_out));
		if (bssrdf_exit) {
			optixSetPayload_13(__float_as_uint(bssrdf_exit_pos.x));
			optixSetPayload_14(__float_as_uint(bssrdf_exit_pos.y));
			optixSetPayload_15(__float_as_uint(bssrdf_exit_pos.z));
		}
	} else if (mat.type == MaterialType::DiffuseLight) {
		// Real NEE-aware MIS weight now (mirrors __closesthit__quad's
		// identical alias-table-scan pattern) - a disk area light is
		// registered in the NEE list whenever pbrt_gpu_builder.h's disk loop
		// gave it one (d.areaLight >= 0), so a ray that happened to hit it via
		// a BSDF bounce (not NEE) can correctly weight against the NEE
		// strategy via mis_power_heuristic(), instead of always treating this
		// as an un-aimable light (light_pdf_for_incoming == 0).
		float light_pdf_for_incoming = 0.0f;
		const float3 mis_o = mis_origin_from_payload();  // last real vertex, not the (possibly moved) ray origin
		if (params.aliasTable && params.numLights > 0) {
			const int prim_idx = (int)primIdx;
			float sel_pdf = 0.0f;
			for (unsigned int li = 0; li < params.numLights; ++li) {
				if (params.lightIndices[li] == prim_idx && params.lightKinds[li] == GpuLightKind::Disk) {
					// See optix_intersection_sphere.h's identical block for
					// why this checks the light BVH first.
					sel_pdf = (params.lightBvhNodeCount > 0)
						? gpu_light_bvh_pmf(mis_o.x, mis_o.y, mis_o.z, 0.f, 0.f, 0.f, (int)li)
						: params.aliasTable[li].pdf;
					break;
				}
			}
			if (sel_pdf > 0.0f) {
				const float geom_pdf = dc_pdf_disk(disk, mis_o, normalize(ray_dir));
				light_pdf_for_incoming = sel_pdf * geom_pdf;
			}
		}
		optixSetPayload_10(2);
		optixSetPayload_11(__float_as_uint(optixGetRayTmax()));  // camera-medium clip distance (see optix_raygen.h's own call site)
		optixSetPayload_12(__float_as_uint(light_pdf_for_incoming));
	} else {
		optixSetPayload_10(0);
		optixSetPayload_11(__float_as_uint(optixGetRayTmax()));  // camera-medium clip distance (see optix_raygen.h's own call site)
		optixSetPayload_12(0);
	}
}

//==============================================================================
// Cylinder
//==============================================================================

// dc_cylinder_check now lives in optix_disk_cylinder_helpers.h - see this
// file's own top-of-file comment.

// Hit-kind sentinel optixReportIntersection()'s 2nd argument carries from
// __intersection__cylinder to __closesthit__cylinder - every OTHER hit
// (hitKind 0) crossed the actual lateral wall, where the closest-hit
// program's own radial obj_normal computation is geometrically valid; a
// kOpenEndVolumeHit hit instead entered this cylinder's attached Medium/
// DielectricMedium purely through an OPEN END (pbrt cylinders have no end
// caps - see dc_cylinder_medium_near_far_world's own comment below), so
// that same radial-normal computation is evaluated at a point INSIDE the
// cross-section, not ON it - geometrically meaningless, and (confirmed by
// direct device-side printf debugging, not guessed) wrong often enough to
// flip `front_face` to true, routing a DielectricMedium hit into the
// ENTRY-surface refraction branch instead of the interior-medium branch -
// silently skipping the fog scattering entirely and letting the ray pass
// through to whatever's behind (background black), exactly matching the
// CPU/GPU parity sweep's "100% relative difference, GPU shows nothing"
// finding for scene E6. __closesthit__cylinder checks this hit kind and
// forces front_face=false for a kOpenEndVolumeHit, which is sufficient and
// correct: see this file's own DielectricMedium `else` (not front_face)
// branch, which already independently RECOMPUTES its own entry/exit
// interval from the current ray on every call - it does not depend on
// which surface (or non-surface) the intersection program found, so
// routing an open-end entry there is exactly right, no further change
// needed downstream.
constexpr unsigned int kOpenEndVolumeHit = 1u;

// dc_cylinder_medium_near_far_world() - tube-quadric-clipped-to-a-z-slab
// entry(near)/exit(far) interval for a Medium/DielectricMedium-attached
// cylinder, object-space math identical to __closesthit__cylinder()'s own
// local `cylinderMediumNearFar` lambda below (see that lambda's own comment
// for the full derivation and its "no partial phi sweep" scope limit) - a
// deliberate, standalone DUPLICATE rather than a shared refactor, so
// __intersection__cylinder() below can call it without touching that
// lambda's own already-working, already-tested closest-hit code (matches
// this file's own established "intersection vs closest-hit duplication is
// deliberate, not an oversight" precedent - see this file's header comment).
// Takes WORLD-space ray_orig/ray_dir (transforms to object space itself),
// matching the wavefront backend's own equivalent, standalone
// wf_medium_cylinder_near_far() (wavefront_intersection_disk_cylinder.h).
__device__ __forceinline__ void dc_cylinder_medium_near_far_world(
		const float3& ray_orig, const float3& ray_dir, const CylinderData& cyl,
		float& t_near, float& t_far) {
	const float3 ro = dc_apply_point(cyl.w2o, ray_orig);
	const float3 rd = dc_apply_vector(cyl.w2o, ray_dir);

	float tube_t0 = -1e30f, tube_t1 = 1e30f;
	bool hasTube;
	if (rd.x == 0.0f && rd.y == 0.0f) {
		hasTube = (double)ro.x * ro.x + (double)ro.y * ro.y <= (double)cyl.radius * (double)cyl.radius;
	} else {
		hasTube = dc_solve_tube_quadratic(ro, rd, cyl.radius, tube_t0, tube_t1);
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

	t_near = fmaxf(0.0f, fmaxf(tube_t0, z_t0));
	t_far  = fminf(tube_t1, z_t1);
	if (!hasTube || !hasZSlab || t_far < t_near) { t_near = 0.0f; t_far = 0.0f; }
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
	// cylinder seen nearly edge-on) - factored into dc_solve_tube_quadratic
	// (optix_disk_cylinder_helpers.h), shared with dc_pdf_cylinder and the
	// Medium closest-hit case below (see that function's own comment for
	// why a==0 - ray parallel to the axis - returns false here specifically,
	// unlike the Medium case's own different a==0 handling).
	float t0, t1;
	if (dc_solve_tube_quadratic(ro, rd, cyl.radius, t0, t1) && !(t0 > ray_tmax || t1 < ray_tmin)) {
		float t = t0;
		if (dc_cylinder_check(cyl, ro, rd, t0, ray_tmin, ray_tmax)) {
			optixReportIntersection(t, 0, 0, 0, 0, 0);
			return;
		}
		t = t1;
		if (dc_cylinder_check(cyl, ro, rd, t1, ray_tmin, ray_tmax)) {
			optixReportIntersection(t, 0, 0, 0, 0, 0);
			return;
		}
	}

	// Lateral-wall surface test above found no crossing within [zMin,zMax] -
	// but pbrt cylinders have no end caps, so a ray can still legitimately
	// pass through this cylinder's attached Medium/DielectricMedium volume
	// purely via its OPEN ends (grazing in/out through the top or bottom
	// rather than crossing the tube wall), which the surface-only test above
	// has no way to see - a real bug a CPU/GPU parity sweep found (tests/
	// integration/material_cpu_gpu_parity_tests.cpp's E6 finding: total,
	// 100%-relative-difference CPU-vs-GPU disagreement confined to exactly
	// the screen blocks where sightlines graze a medium cylinder's open
	// ends). Fall back to the tube-x-zslab volume interval and report an
	// intersection at its near root if it's non-empty and overlaps this
	// ray's valid range, tagged with kOpenEndVolumeHit so __closesthit__
	// cylinder knows NOT to trust its own radial-normal/front_face
	// computation for this hit (see that constant's own comment for why,
	// and for why no further downstream change is needed). CPU's own
	// equivalent (src/TheRestOfYourLife/disk_cylinder_hittable.h's
	// `volume_bounds()`) already handles exactly this case via a dedicated
	// volume-bounds path instead of its own surface hit() test.
	const MaterialData& cyl_mat = params.materials[cyl.materialIdx];
	const bool isMediumAttached = (cyl_mat.type == MaterialType::Medium) ||
		(cyl_mat.type == MaterialType::DielectricMedium);
	if (!isMediumAttached) return;

	float t_near, t_far;
	dc_cylinder_medium_near_far_world(ray_orig_w, ray_dir_w, cyl, t_near, t_far);
	if (t_far <= t_near) return;                        // empty interval
	if (t_near > ray_tmax || t_far < ray_tmin) return;   // doesn't overlap the ray's valid range
	optixReportIntersection(fmaxf(t_near, ray_tmin), kOpenEndVolumeHit, 0, 0, 0, 0);
}

extern "C" __global__ void __closesthit__cylinder() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const CylinderData& cyl = params.cylinders[primIdx];

	const float t = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t * ray_dir;

	// Resolved to a real, non-Mix material before any mat.type branch below -
	// see MaterialType::Mix's own comment (optix_types.h).
	int matIdx = cyl.materialIdx;
	const MaterialData mat = resolve_mix_material(params.materials[matIdx], matIdx, hit_point, matIdx);

	// Cylinder's normal varies with hit position (radial, from the axis) -
	// unlike disk's constant one - so the object-space hit point has to be
	// recovered first (see this file's header comment for why re-deriving it
	// from the world hit point is exact, not approximate). This radial
	// computation is only geometrically valid for a hitKind-0 hit (a real
	// lateral-wall crossing) - see kOpenEndVolumeHit's own comment above for
	// why a hitKind-1 (open-end volume entry) hit point isn't ON this
	// surface at all, making obj_normal/front_face meaningless for it; the
	// override just below is what actually matters for that case, not this
	// computation itself.
	const float3 obj_hit = dc_apply_point(cyl.w2o, hit_point);
	const float obj_hit_len = sqrtf(obj_hit.x * obj_hit.x + obj_hit.y * obj_hit.y);
	const float3 obj_normal = (obj_hit_len > 1e-8f)
		? make_float3(obj_hit.x / obj_hit_len, obj_hit.y / obj_hit_len, 0.0f)
		: make_float3(1.0f, 0.0f, 0.0f);
	float3 outward_normal = normalize(dc_apply_normal_from_w2o(cyl.w2o, obj_normal));
	// A kOpenEndVolumeHit is, by construction, already inside this
	// cylinder's attached medium (entered via an open end, not a real
	// surface) - force front_face=false so MaterialType::DielectricMedium's
	// dispatch below takes its "already inside, recompute near/far, sample
	// the medium" branch rather than its "fresh surface entry, refract/
	// reflect, skip the medium entirely" branch (the latter is what this
	// hit kind exists to avoid - see kOpenEndVolumeHit's own comment for
	// the full bug this fixes). MaterialType::Medium's own branch below
	// doesn't condition on front_face at all, so it's unaffected either way.
	const bool front_face = (optixGetHitKind() == kOpenEndVolumeHit)
		? false
		: (dot(ray_dir, outward_normal) < 0.0f);
	const float3 normal = front_face ? outward_normal : -outward_normal;

	unsigned int seed = optixGetPayload_9();

	// Real UV (phi/phiMax, z-fraction - CPU CylinderShape<T>'s own
	// u=phi/phi_max, v=(hz-z_min)/(z_max-z_min) convention, shapes.h:760)
	// for a pbrt AreaLightSource "filename" cylinder light to sample its
	// image correctly instead of always reading texel (0,0) - recomputed
	// here from obj_hit (already computed above for the normal) rather than
	// a widened attribute budget, same reasoning __closesthit__disk's phi/
	// radial-fraction UV follows (this file, above).
	float uv_phi = atan2f(obj_hit.y, obj_hit.x);
	if (uv_phi < 0.0f) uv_phi += 6.283185307179586f;
	const float uv_u = uv_phi / cyl.phiMax;
	const float uv_v = (cyl.zMax > cyl.zMin)
		? (obj_hit.z - cyl.zMin) / (cyl.zMax - cyl.zMin)
		: 0.0f;

	// Real analytic dpdu, matching CPU's disk_cylinder_hittable.h exactly -
	// same phi-tangent form as disk (the axis (z) doesn't affect the
	// azimuthal tangent direction), carried to world space via o2w as a
	// genuine tangent vector. Gated on material_needs_dpdu() - same
	// reasoning as disk's own identical gate above.
	const float3 cyl_dpdu = material_needs_dpdu(mat.type)
		? dc_apply_vector(cyl.o2w,
			make_float3(-cyl.phiMax * obj_hit.y, cyl.phiMax * obj_hit.x, 0.0f))
		: make_float3(0.0f, 0.0f, 0.0f);

	float3 emission = material_emission(mat, front_face, uv_u, uv_v, hit_point);

	float3 attenuation;
	float3 scattered_dir;
	bool scattered = false;
	bool is_specular = false;
	bool is_medium_boundary = false;  // MaterialType::Interface - see optix_types.h
	float brdf_pdf_override = -1.0f;
	bool bssrdf_exit = false;
	float3 bssrdf_exit_pos = make_float3(0.0f, 0.0f, 0.0f);
	// MaterialType::Medium: t_hit below must use medium_t_hit, not
	// optixGetRayTmax() (which is just the cylinder's own entry root) - see
	// that branch's own comment, mirrors optix_intersection_sphere.h's
	// identical is_medium/medium_t_hit pair exactly.
	bool is_medium = false;
	float medium_t_hit = 0.0f;

	// Real Hair support on cylinder geometry - see __closesthit__disk's
	// identical branch (this file, above) for why this moved out of the trap.
	float out_eta = 1.0f;
	// See PathTracingPayload's own p24/rgbChannel comment (optix_renderer_init.cpp)
	// and shade_material()'s inout_rgb_channel parameter comment - read
	// unconditionally so a channel already chosen by an earlier bounce
	// still survives unchanged through this one, regardless of which
	// branch below (Hair/Medium/shade_material) actually runs.
	unsigned int rgbChannel = optixGetPayload_24();
	// Integrator "bool regularize" - see current_do_regularize()'s own
	// comment (optix_device_helpers.h). Hoisted here (rather than declared
	// just before the final shade_material() call site below, as it used
	// to be) so MaterialType::DielectricMedium's own rough-fused sub-case,
	// further up this if-chain, can read it too - matches __closesthit__
	// sphere's identical early hoist (optix_intersection_sphere.h).
	const bool do_regularize = current_do_regularize();
	// Shared cylinder entry(near)/exit(far) chord against the tube-quadric +
	// z-slab bound - factored out so both the homogeneous Medium case below
	// AND DielectricMedium's own exit-surface sub-case (further below) use
	// the exact same object-space math instead of two independently-
	// drifting copies. `ro`/`rd` are object-space (dc_apply_point/vector of
	// the WORLD ray through cyl.w2o), NOT normalised - matches this file's
	// own "t needs no rescaling between object/world space" convention.
	// Deliberately does NOT account for a partial phi sweep (phiMax < 2*pi)
	// - a "pie slice" cross-section makes the entry/exit computation
	// genuinely harder (the volume is no longer a simple slab-clipped
	// tube), and MediumInterface on a phi-clipped cylinder is a rare enough
	// combination that this scope limitation is documented (docs/
	// PBRT_SUPPORT.md) rather than handled; a mismatch there under-
	// estimates dist_inside rather than crashing or overestimating (the
	// tube/z-slab bound is still a superset of the real phi-clipped
	// volume), so this degrades gracefully.
	const auto cylinderMediumNearFar = [&](const float3 &ro, const float3 &rd, float &t_near, float &t_far) {
		// Ray parallel to the axis (rd.x==rd.y==0) is handled here directly,
		// NOT via dc_solve_tube_quadratic() below - that function's a==0
		// case answers "is there a discrete surface crossing" (always no),
		// which is the wrong question for a VOLUME/interval test: a ray
		// parallel to and inside the axis is entirely inside the medium for
		// its whole length, a real non-degenerate answer dc_solve_tube_
		// quadratic deliberately doesn't provide (see its own comment).
		float tube_t0 = -1e30f, tube_t1 = 1e30f;
		bool hasTube;
		if (rd.x == 0.0f && rd.y == 0.0f) {
			hasTube = (double)ro.x * ro.x + (double)ro.y * ro.y <= (double)cyl.radius * (double)cyl.radius;
		} else {
			hasTube = dc_solve_tube_quadratic(ro, rd, cyl.radius, tube_t0, tube_t1);
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

		t_near = fmaxf(0.0f, fmaxf(tube_t0, z_t0));
		t_far  = fminf(tube_t1, z_t1);
		if (!hasTube || !hasZSlab || t_far < t_near) { t_near = 0.0f; t_far = 0.0f; }
	};
	if (mat.type == MaterialType::Hair) {
		scattered   = sample_hair_material(ray_dir, normal, mat, seed, scattered_dir, attenuation);
		is_specular = true;
	} else if (mat.type == MaterialType::Medium) {
		// Homogeneous participating medium - see MaterialType::Medium's
		// comment in optix_types.h and optix_intersection_sphere.h's
		// identical closesthit case. Unlike sphere (a single closed
		// quadric), a finite cylinder is a tube quadric CLIPPED to a z-slab
		// (see cylinderMediumNearFar() above for the full derivation);
		// entry/exit is computed in OBJECT space (matching __intersection__
		// cylinder's own ray transform) since CylinderData::zMin/zMax are
		// object-space.
		const float3 ro = dc_apply_point(cyl.w2o, ray_orig);
		const float3 rd = dc_apply_vector(cyl.w2o, ray_dir);  // NOT normalised - see file header comment
		float t_near, t_far;
		cylinderMediumNearFar(ro, rd, t_near, t_far);
		float dist_inside = fmaxf(0.0f, t_far - t_near);

		float sigma_t = mat.ior;
		float free_path = (sigma_t > 1e-8f) ? (-logf(fmaxf(1e-8f, 1.0f - random_float(seed))) / sigma_t) : 1e30f;
		float3 unit_dir = normalize(ray_dir);
		if (free_path < dist_inside) {
			medium_t_hit = t_near + free_path;
			float3 wo = -unit_dir;
			scattered_dir = sample_henyey_greenstein(wo, mat.fuzz, seed);
			attenuation = mat.albedo;
			// Real NEE+MIS at the phase-function scatter event, plus
			// MakeNamedMedium's own "rgb Le" self-emission (folded into
			// medium_phase_nee_mis()'s own return value - see that function's
			// own comment, optix_device_helpers.h, and MaterialData::
			// medium_emission's own comment, optix_types.h, for the
			// sigma_a/sigma_t weighting already baked in at build time).
			// medium_t_hit is parametrized against the RAW (not renormalized)
			// ray_dir - matches this file's own "t needs no rescaling between
			// object/world space" convention (dc_apply_vector isn't
			// renormalized, see file header comment) - so the world-space
			// point must scale by ray_dir here too, NOT unit_dir.
			float3 medium_point = ray_orig + medium_t_hit * ray_dir;
			emission = emission + medium_phase_nee_mis(
				medium_point, wo, mat.fuzz, attenuation, scattered_dir, seed, brdf_pdf_override, mat.medium_emission, optixGetRayTime());
			is_specular = false;
		} else {
			medium_t_hit = t_far;
			scattered_dir = unit_dir;
			attenuation = make_float3(1.0f, 1.0f, 1.0f);
			is_specular = true;  // no interaction - a free/non-scattering pass-through
			// A free crossing, like MaterialType::Interface: flag it so the last real vertex's MIS state (prev BSDF pdf,
			// specular flag) survives it instead of being reset as if a specular bounce had happened.
			is_medium_boundary = true;
		}
		scattered   = true;
		is_medium   = true;
	} else if (mat.type == MaterialType::DielectricMedium) {
		// Combined dielectric surface + internal medium on a cylinder - real
		// GPU support, mirroring optix_intersection_sphere.h's own
		// DielectricMedium branch and this file's Medium case just above
		// exactly. On entry (front_face) the direct dielectric surface
		// always wins the bounce (matches CPU's two-hittable trick: the
		// medium's sampled hit distance can never be closer than the entry
		// surface), so just refract/reflect normally. On the following
		// bounce, now travelling inside toward this cylinder's exit surface
		// (front_face false), recompute the remaining tube/z-slab chord via
		// cylinderMediumNearFar() to get the distance to the exit, sample a
		// free path, and either scatter via the HG phase function or fall
		// through to a normal exit refraction/reflection (or thin coin-
		// flip, or rough glass with real glossy NEE) at the far surface.
		// surfaceKind/is_rough/is_thin/rdm_scatter_ok: see sphere's own
		// identical DielectricMedium branch (optix_intersection_sphere.h)
		// for the full comment on all four.
		const float rdm_kind = mat.dielectric_medium_extra.surfaceKind;
		const bool is_rough = rdm_kind >= 1.5f;
		const bool is_thin  = !is_rough && rdm_kind >= 0.5f;
		bool rdm_scatter_ok = true;
		if (front_face) {
			if (is_rough) {
				rdm_scatter_ok = rough_dielectric_scatter_and_nee(
					mat, mat.dielectric_medium_extra.roughness, /*allowDispersion=*/false,
					normal, ray_dir, hit_point, front_face, uv_u, uv_v, cyl_dpdu,
					do_regularize, seed, rgbChannel,
					attenuation, scattered_dir, is_specular, brdf_pdf_override, emission, out_eta);
			} else {
				attenuation = make_float3(1.0f, 1.0f, 1.0f);
				if (is_thin) {
					scattered_dir = thin_dielectric_scatter(ray_dir, normal, mat.ior, seed);
				} else {
					scattered_dir = dielectric_scatter(ray_dir, normal, front_face, mat.ior, seed);
					// pbrt-v4 etaScale (entry surface) - see MaterialType::
					// Dielectric's identical eta computation (optix_device_helpers.h).
					if (dot(scattered_dir, normal) < 0.0f) out_eta = front_face ? (1.0f / mat.ior) : mat.ior;
				}
				is_specular = true;
			}
		} else {
			const float3 ro = dc_apply_point(cyl.w2o, ray_orig);
			const float3 rd = dc_apply_vector(cyl.w2o, ray_dir);  // NOT normalised - see file header comment
			float t_near, t_far;
			cylinderMediumNearFar(ro, rd, t_near, t_far);
			float dist_inside = fmaxf(0.0f, t_far - t_near);

			float sigma_t = mat.dielectric_medium_extra.sigma_t;
			float free_path = (sigma_t > 1e-8f) ? (-logf(fmaxf(1e-8f, 1.0f - random_float(seed))) / sigma_t) : 1e30f;
			float3 unit_dir = normalize(ray_dir);

			if (free_path < dist_inside) {
				medium_t_hit = t_near + free_path;
				// medium_t_hit/medium_point parametrized against the RAW
				// ray_dir, not unit_dir - see the Medium branch's own
				// identical comment just above.
				float3 medium_point = ray_orig + medium_t_hit * ray_dir;
				float g = mat.g;  // Medium/DielectricMedium: HG asymmetry
				float3 wo = -unit_dir;
				scattered_dir = sample_henyey_greenstein(wo, g, seed);
				attenuation = mat.medium_albedo;
				is_medium = true;
				// Real NEE+MIS at the phase-function scatter event - see
				// medium_phase_nee_mis()'s own comment (optix_device_
				// helpers_lighting.h). The two OTHER DielectricMedium sub-
				// cases (the entry/exit dielectric-surface refractions just
				// above/below) are genuinely specular and correctly stay
				// is_specular=true - only this interior phase-function
				// event is smooth/continuous like a diffuse BRDF and
				// benefits from NEE the same way, matching sphere's
				// identical DielectricMedium branch.
				emission = emission + medium_phase_nee_mis(
					medium_point, wo, g, attenuation, scattered_dir, seed, brdf_pdf_override, mat.medium_emission, optixGetRayTime());
				is_specular = false;
			} else if (is_rough) {
				rdm_scatter_ok = rough_dielectric_scatter_and_nee(
					mat, mat.dielectric_medium_extra.roughness, /*allowDispersion=*/false,
					normal, ray_dir, hit_point, front_face, uv_u, uv_v, cyl_dpdu,
					do_regularize, seed, rgbChannel,
					attenuation, scattered_dir, is_specular, brdf_pdf_override, emission, out_eta);
			} else {
				attenuation = make_float3(1.0f, 1.0f, 1.0f);
				if (is_thin) {
					scattered_dir = thin_dielectric_scatter(ray_dir, normal, mat.ior, seed);
				} else {
					scattered_dir = dielectric_scatter(ray_dir, normal, front_face, mat.ior, seed);
					// pbrt-v4 etaScale (exit surface, front_face is false here) -
					// see MaterialType::Dielectric's identical eta computation.
					if (dot(scattered_dir, normal) < 0.0f) out_eta = front_face ? (1.0f / mat.ior) : mat.ior;
				}
				is_specular = true;
			}
		}
		scattered = rdm_scatter_ok;
	} else {
		if (material_requires_sphere_only_handling(mat.type) ||
			mat.type == MaterialType::NormalMappedLambertian) {
			printf("[CYLINDER-SHADE] MaterialType %d is not supported on cylinder geometry\n", (int)mat.type);
			__trap();
		}

		// do_regularize already hoisted above (see its own comment).
		shade_material(mat, matIdx, normal, ray_dir, hit_point, front_face, uv_u, uv_v, cyl_dpdu, do_regularize, seed,
			attenuation, scattered_dir, scattered, is_specular, is_medium_boundary, brdf_pdf_override, emission,
			bssrdf_exit, bssrdf_exit_pos, out_eta, rgbChannel);
	}

	optixSetPayload_3(__float_as_uint(emission.x));
	optixSetPayload_4(__float_as_uint(emission.y));
	optixSetPayload_5(__float_as_uint(emission.z));
	optixSetPayload_9(seed);

	{
		float3 albedoAov = scattered ? attenuation : mat.albedo;
		pack_aov_payload(albedoAov, normal);
	}
	optixSetPayload_22(__float_as_uint(out_eta));  // pbrt-v4 etaScale - see PathTracingPayload::eta
	optixSetPayload_24(rgbChannel);  // dispersion channel - see shade_material()'s inout_rgb_channel comment

	if (scattered) {
		// Medium: the scatter/exit point is not the cylinder's entry root
		// (optixGetRayTmax()) - use the distance computed in the Medium case
		// above instead, same as optix_intersection_sphere.h's own
		// is_medium/medium_t_hit handling.
		float t_hit = is_medium ? medium_t_hit : optixGetRayTmax();
		float brdf_pdf_out = is_specular ? 0.0f
						  : (brdf_pdf_override >= 0.0f ? brdf_pdf_override : cosine_pdf(scattered_dir, normal));

		optixSetPayload_0(__float_as_uint(attenuation.x));
		optixSetPayload_1(__float_as_uint(attenuation.y));
		optixSetPayload_2(__float_as_uint(attenuation.z));
		optixSetPayload_6(__float_as_uint(scattered_dir.x));
		optixSetPayload_7(__float_as_uint(scattered_dir.y));
		optixSetPayload_8(__float_as_uint(scattered_dir.z));
		optixSetPayload_10(pack_scatter_flag(bssrdf_exit, is_medium_boundary, is_specular));  // scattered (see pack_scatter_flag's own comment)
		optixSetPayload_11(__float_as_uint(t_hit));
		optixSetPayload_12(__float_as_uint(brdf_pdf_out));
		if (bssrdf_exit) {
			optixSetPayload_13(__float_as_uint(bssrdf_exit_pos.x));
			optixSetPayload_14(__float_as_uint(bssrdf_exit_pos.y));
			optixSetPayload_15(__float_as_uint(bssrdf_exit_pos.z));
		}
	} else if (mat.type == MaterialType::DiffuseLight) {
		// See __closesthit__disk's identical pattern (this file, above).
		float light_pdf_for_incoming = 0.0f;
		const float3 mis_o = mis_origin_from_payload();  // last real vertex, not the (possibly moved) ray origin
		if (params.aliasTable && params.numLights > 0) {
			const int prim_idx = (int)primIdx;
			float sel_pdf = 0.0f;
			for (unsigned int li = 0; li < params.numLights; ++li) {
				if (params.lightIndices[li] == prim_idx && params.lightKinds[li] == GpuLightKind::Cylinder) {
					sel_pdf = (params.lightBvhNodeCount > 0)
						? gpu_light_bvh_pmf(mis_o.x, mis_o.y, mis_o.z, 0.f, 0.f, 0.f, (int)li)
						: params.aliasTable[li].pdf;
					break;
				}
			}
			if (sel_pdf > 0.0f) {
				const float geom_pdf = dc_pdf_cylinder(cyl, mis_o, normalize(ray_dir));
				light_pdf_for_incoming = sel_pdf * geom_pdf;
			}
		}
		optixSetPayload_10(2);
		optixSetPayload_11(__float_as_uint(optixGetRayTmax()));  // camera-medium clip distance (see optix_raygen.h's own call site)
		optixSetPayload_12(__float_as_uint(light_pdf_for_incoming));
	} else {
		optixSetPayload_10(0);
		optixSetPayload_11(__float_as_uint(optixGetRayTmax()));  // camera-medium clip distance (see optix_raygen.h's own call site)
		optixSetPayload_12(0);
	}
}
