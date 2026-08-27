// wavefront_intersection_sphere.h -- Sphere intersection + closest-hit programs
// Included by wavefront_programs.cu, after wavefront_common.h.

// ============================================================================
// __intersection__wf_sphere / __intersection__wf_quad
//
// Wavefront-native duplicates of optix_intersection_{sphere,quad}.h's
// __intersection__sphere/__intersection__quad (same math, wf_params instead
// of the recursive path's params global). createProgramGroups() originally
// reused those programs directly from the recursive path's module (custom
// AABB primitives need *some* intersection program, and wavefront_programs.cu
// never defined its own) - but OptiX rejects combining intersection/
// closest-hit programs from two separately-compiled modules with different
// pipelineCompileOptions.numPayloadValues (12 registers for the recursive
// path's PathTracingPayload vs 2 here) into the same hit group: "could not
// be resolved to a common payloadType". Compiling our own copies here, under
// this module's own pipeline options, sidesteps the mismatch entirely -
// neither program actually touches payload registers (attributes only), so
// there's no behavioral difference, just no more cross-module payload
// conflict.
// ============================================================================

// Wavefront-native copy of optix_intersection_sphere.h's box_slab_intersect()/
// box_face_normal() - same reason wf_sphere/wf_quad/wf_triangle above are
// duplicated rather than shared (separately-compiled OptiX module, see this
// file's own header comment). See GpuMediumShapeKind's comment in
// optix_types.h for what shapeKind==Box means.
__device__ __forceinline__ bool wf_box_slab_intersect(
		const float3& ray_orig, const float3& ray_dir,
		const float3& bmin, const float3& bmax,
		float& t_near_out, float& t_far_out) {
	float t0 = -1e30f, t1 = 1e30f;
	float invd, tn, tf;

	invd = (ray_dir.x != 0.0f) ? (1.0f / ray_dir.x) : 1e30f;
	tn = (bmin.x - ray_orig.x) * invd; tf = (bmax.x - ray_orig.x) * invd;
	if (tn > tf) { float tmp = tn; tn = tf; tf = tmp; }
	t0 = fmaxf(t0, tn); t1 = fminf(t1, tf);

	invd = (ray_dir.y != 0.0f) ? (1.0f / ray_dir.y) : 1e30f;
	tn = (bmin.y - ray_orig.y) * invd; tf = (bmax.y - ray_orig.y) * invd;
	if (tn > tf) { float tmp = tn; tn = tf; tf = tmp; }
	t0 = fmaxf(t0, tn); t1 = fminf(t1, tf);

	invd = (ray_dir.z != 0.0f) ? (1.0f / ray_dir.z) : 1e30f;
	tn = (bmin.z - ray_orig.z) * invd; tf = (bmax.z - ray_orig.z) * invd;
	if (tn > tf) { float tmp = tn; tn = tf; tf = tmp; }
	t0 = fmaxf(t0, tn); t1 = fminf(t1, tf);

	t_near_out = t0;
	t_far_out = t1;
	return t0 <= t1;
}

__device__ __forceinline__ float3 wf_box_face_normal(const float3& p, const float3& bmin, const float3& bmax) {
	float best = fabsf(p.x - bmin.x);
	float3 n = make_float3(-1.0f, 0.0f, 0.0f);
	float d;
	d = fabsf(p.x - bmax.x); if (d < best) { best = d; n = make_float3(1.0f, 0.0f, 0.0f); }
	d = fabsf(p.y - bmin.y); if (d < best) { best = d; n = make_float3(0.0f, -1.0f, 0.0f); }
	d = fabsf(p.y - bmax.y); if (d < best) { best = d; n = make_float3(0.0f, 1.0f, 0.0f); }
	d = fabsf(p.z - bmin.z); if (d < best) { best = d; n = make_float3(0.0f, 0.0f, -1.0f); }
	d = fabsf(p.z - bmax.z); if (d < best) { best = d; n = make_float3(0.0f, 0.0f, 1.0f); }
	return n;
}

extern "C" __global__ void __intersection__wf_sphere() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const SphereData& sphere = wf_params.spheres[wf_prim_base(wf_instance_base()) + primIdx];

	// OBJECT space - the sphere's centre/radius live in whatever space its GAS
	// was built in. Not a branch for instanced geometry: the scene's own GAS
	// instances carry an identity transform, for which OptiX's object ray IS
	// the world ray, so the non-instanced path is unchanged by construction.
	// t needs no rescaling either, since OptiX transforms the direction
	// without renormalising it.
	const float3 ray_orig = optixGetObjectRayOrigin();
	const float3 ray_dir  = optixGetObjectRayDirection();
	const float  ray_tmin = optixGetRayTmin();
	const float  ray_tmax = optixGetRayTmax();

	if (sphere.shapeKind == GpuMediumShapeKind::Box) {
		// See optix_intersection_sphere.h's own box branch - identical dual-
		// root selection, attributes unused (box bounds are static, re-read
		// directly by __closesthit__wf_sphere below instead).
		float t_near, t_far;
		if (!wf_box_slab_intersect(ray_orig, ray_dir, sphere.boxMin, sphere.boxMax, t_near, t_far))
			return;
		float root = t_near;
		if (root < ray_tmin || root > ray_tmax) {
			root = t_far;
			if (root < ray_tmin || root > ray_tmax)
				return;
		}
		optixReportIntersection(root, 0, 0, 0, 0, 0);
		return;
	}

	const float3 oc = ray_orig - sphere.center;
	const float a = dot(ray_dir, ray_dir);
	const float half_b = dot(oc, ray_dir);
	const float c = dot(oc, oc) - sphere.radius * sphere.radius;
	const float discriminant = half_b * half_b - a * c;
	if (discriminant < 0.0f) return;

	const float sqrtd = sqrtf(discriminant);
	float root = (-half_b - sqrtd) / a;
	if (root < ray_tmin || root > ray_tmax) {
		root = (-half_b + sqrtd) / a;
		if (root < ray_tmin || root > ray_tmax)
			return;
	}

	optixReportIntersection(
		root, 0,
		__float_as_int(sphere.center.x),
		__float_as_int(sphere.center.y),
		__float_as_int(sphere.center.z),
		__float_as_int(sphere.radius));
}

// ============================================================================
// __closesthit__wf_sphere
// ============================================================================
extern "C" __global__ void __closesthit__wf_sphere() {
	WfHitPayload* payload = (WfHitPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	// Custom intersection stores sphere index in attribute 0.
	const int instBase  = wf_instance_base();
	const int sphereIdx = (int)(wf_prim_base(instBase) + optixGetPrimitiveIndex());

	// Reconstruct hit point and normal from ray + t. Position and view
	// direction stay in WORLD space - they feed lighting, which is world-space
	// throughout - while the normal is derived in the sphere's own space and
	// brought back, so a placement's non-uniform scale yields the ellipsoid
	// normal rather than a resized sphere's.
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float  t_hit    = optixGetRayTmax();

	float3 hit_point = ray_orig + t_hit * ray_dir;

	// Sphere normal. optixGetObjectRay*() is illegal in a closest-hit program,
	// so the hit point is transformed instead of the ray rebuilt - same reason
	// optix_intersection_sphere.h does it this way. Both transforms are
	// skipped outright when instBase < 0, leaving that path bit-for-bit what
	// it computed before instancing existed.
	const SphereData& sph = wf_params.spheres[sphereIdx];
	// See GpuMediumShapeKind's comment in optix_types.h. Box bounds are
	// static, so re-read directly from `sph` rather than needing any
	// motion-interpolated attribute.
	const bool is_box = (sph.shapeKind == GpuMediumShapeKind::Box);
	float3 obj_hit = hit_point;
	if (instBase >= 0) obj_hit = optixTransformPointFromWorldToObjectSpace(hit_point);
	// Kept OBJECT-space (never transformed) for UV purposes, matching
	// optix_intersection_sphere.h's own obj_normal/outward_normal split - a
	// texture stays pinned to the geometry as a placement rotates it, which a
	// world-space UV wouldn't. outward_normal below is the (possibly
	// world-transformed) copy used for shading/dpdu.
	const float3 obj_normal = is_box
		? wf_box_face_normal(obj_hit, sph.boxMin, sph.boxMax)
		: normalize(obj_hit - sph.center);
	float3 outward_normal = obj_normal;
	if (instBase >= 0)
		outward_normal = normalize(optixTransformNormalFromObjectToWorldSpace(outward_normal));
	// Flip to face the ray
	bool front_face = dot(ray_dir, outward_normal) < 0.0f;
	float3 normal = front_face ? outward_normal : -outward_normal;

	// Sphere UV - matches CPU's get_sphere_uv() / optix_intersection_sphere.h
	// exactly (uses the raw, un-flipped obj_normal - UV is a property of the
	// surface point, independent of which side the ray hit from).
	const float sphere_theta = acosf(-obj_normal.y);
	const float sphere_phi = atan2f(-obj_normal.z, obj_normal.x) + 3.14159265358979323846f;

	// Real analytic dpdu (tangent), matching CPU's sphere.h and
	// optix_intersection_sphere.h's identical derivation exactly (theta=
	// pi*v, phi=2*pi*u; differentiate w.r.t. phi/theta, chain-rule through
	// u/v) - computed in OBJECT space from obj_normal/sphere_theta/
	// sphere_phi (same convention the UV above uses), then carried to
	// world space as a genuine tangent VECTOR transform
	// (optixTransformVectorFromObjectToWorldSpace, NOT the inverse-
	// transpose normal transform outward_normal above uses). Degenerate at
	// the poles (sin_theta -> 0, a real parametric singularity, not a bug):
	// falls back to cross(world_up, obj_normal) rescaled to the same
	// vanishing magnitude, exactly as CPU does. This REPLACES the previous
	// approximate cross(world_up, objDpdu) tangent evaluate_materials()
	// used to derive from this field for NormalMappedLambertian - the field
	// now carries the real thing directly, matching triangle/bilinear-patch's
	// own convention below, and is also the tangent the 4 anisotropy-
	// capable material kinds need for a UV-aligned shading frame.
	float3 sphere_dpdu;
	{
		const float sin_theta = sinf(sphere_theta);
		const float sin_phi = sinf(sphere_phi), cos_phi = cosf(sphere_phi);
		sphere_dpdu = make_float3(sin_theta * sin_phi, 0.0f, sin_theta * cos_phi)
			* (sph.radius * 2.0f * 3.14159265358979323846f);
		if (dot(sphere_dpdu, sphere_dpdu) < 1e-14f) {
			const float3 world_up = make_float3(0.0f, 1.0f, 0.0f);
			const float3 tangent = cross(world_up, obj_normal);
			const float tlen = length(tangent);
			const float3 dir = (tlen > 1e-6f) ? (tangent / tlen) : make_float3(1.0f, 0.0f, 0.0f);
			sphere_dpdu = dir * (sph.radius * 2.0f * 3.14159265358979323846f * sin_theta);
		}
	}
	if (instBase >= 0) sphere_dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(sphere_dpdu));

	payload->hitPoint    = hit_point;
	payload->normal      = normal;
	payload->t           = t_hit;
	payload->materialIdx = sph.materialIdx;
	payload->geomType    = 0;
	payload->hit         = true;
	payload->mediumTFar  = 0.0f;
	payload->frontFace   = front_face ? 1 : 0;
	payload->objDpdu     = sphere_dpdu;
	payload->uv_u        = sphere_phi / (2.0f * 3.14159265358979323846f);
	payload->uv_v        = sphere_theta / 3.14159265358979323846f;

	// MaterialType::Medium (always) and MaterialType::DielectricMedium (exit
	// surface only, front_face false - its entry surface just refracts/
	// reflects normally, no re-intersection needed there): override with the
	// entry (near) / exit (far) roots recomputed relative to the CURRENT ray
	// origin, matching optix_intersection_sphere.h's __closesthit__sphere.
	// optixGetRayTmax() alone isn't enough - it reports whichever single root
	// the intersection program found valid, which is the FAR root when this
	// ray already starts inside the sphere (e.g. continuing after a prior
	// in-medium scatter). Recomputing both roots here handles that re-entry
	// case the same way the recursive path does.
	const MaterialData& sph_mat = wf_params.materials[sph.materialIdx];
	const bool needsNearFar = (sph_mat.type == MaterialType::Medium) ||
		(sph_mat.type == MaterialType::DielectricMedium && !front_face);
	if (needsNearFar) {
		float3 unit_dir = normalize(ray_dir);
		float t_near, t_far;
		if (is_box) {
			float bn, bf;
			wf_box_slab_intersect(ray_orig, unit_dir, sph.boxMin, sph.boxMax, bn, bf);
			t_near = fmaxf(0.0f, bn);
			t_far  = bf;
		} else {
			float3 oc2 = ray_orig - sph.center;
			float half_b2 = dot(oc2, unit_dir);
			float c2 = dot(oc2, oc2) - sph.radius * sph.radius;
			float disc2 = fmaxf(0.0f, half_b2 * half_b2 - c2);
			float sq2 = sqrtf(disc2);
			t_near = fmaxf(0.0f, -half_b2 - sq2);
			t_far  = -half_b2 + sq2;
		}
		payload->t          = t_near;
		payload->hitPoint   = ray_orig + t_near * unit_dir;
		payload->mediumTFar = t_far;
	}
}

