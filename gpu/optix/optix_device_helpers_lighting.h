// optix_device_helpers_lighting.h -- light sampling, NEE, and medium/shadow
// __device__ helpers for the GPU-recursive backend's closest-hit programs.
//
// Split out of optix_device_helpers.h, which #includes this file directly
// at the point this content used to live (relies on that file's own
// `params`/utility functions declared above the split point, and on
// optix_types.h/fresnel.h/etc already included there - not meant to be
// included standalone). shade_material() and texture sampling stayed in
// optix_device_helpers.h itself - see that file's own header comment.

// Sample a random point on a sphere light
//
// out_u/out_v/out_normal: the sampled surface point's UV (same theta/phi
// convention as optix_intersection_sphere.h's direct-hit formula: theta=
// acos(-p.y), phi=atan2(-p.z,p.x)+pi, u=phi/(2pi), v=theta/pi, evaluated on
// the local unit-sphere point) and world-space outward normal - recovered
// by re-intersecting the sampled cone direction against the sphere (the
// near root), since cone sampling itself only produces a direction, not a
// point. Needed so a pbrt AreaLightSource "filename" sphere light can be
// sampled with a real UV instead of always reading texel (0,0), and so
// mat.twoSided can be checked against the true surface orientation - see
// sample_area_light_by_kind()'s own comment on both.
__device__ __forceinline__ float3 sample_sphere_light(
	const SphereData& sphere,
	const float3& origin,
	unsigned int& seed,
	float& pdf,
	float& out_u,
	float& out_v,
	float3& out_normal,
	float ray_time
) {
	// Object (per-primitive sphere) motion blur: interpolate to this ray's
	// time, same lerp/no-op-when-static reasoning as optix_intersection_
	// sphere.h's own __intersection__sphere - a moving emissive sphere's NEE
	// sample must target the SAME interpolated position the caller's own
	// shadow ray (traced at this same time) will test occlusion against, or
	// the two disagree on where the light actually is. `ray_time` is passed
	// in explicitly (NOT read via optixGetRayTime() here) because this
	// helper is called both from closest-hit-invoked shading chains (where
	// optixGetRayTime() is legal) AND from the recursive backend's raygen
	// loop's own camera-medium NEE (optix_raygen.h - see sample_camera_
	// medium()'s own comment), where there is no "current ray" for
	// optixGetRayTime() to read - every caller passes optixGetRayTime() or
	// its own already-known ray time, matching CPU's own convention of
	// explicit time-threading rather than an implicit "current ray" global.
	const float3 center = make_float3(
		sphere.center.x + ray_time * (sphere.center1.x - sphere.center.x),
		sphere.center.y + ray_time * (sphere.center1.y - sphere.center.y),
		sphere.center.z + ray_time * (sphere.center1.z - sphere.center.z));

	// Direction from origin to sphere center
	float3 to_center = center - origin;
	float dist_sq = dot(to_center, to_center);

	// Avoid division by zero
	if (dist_sq < 1e-6f) {
		pdf = 0.0f;
		out_u = out_v = 0.0f;
		out_normal = make_float3(0.0f, 0.0f, 1.0f);
		return make_float3(0.0f, 0.0f, 1.0f);
	}

	// Compute solid angle PDF
	float cos_theta_max = sqrtf(1.0f - sphere.radius * sphere.radius / dist_sq);
	float solid_angle = 2.0f * 3.14159265358979323846f * (1.0f - cos_theta_max);
	pdf = 1.0f / solid_angle;

	// Build ONB around direction to sphere
	float3 w = normalize(to_center);
	float3 a = (fabsf(w.x) > 0.9f) ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
	float3 v = normalize(cross(w, a));
	float3 u = cross(w, v);

	// Sample direction within cone
	float z = 1.0f + random_float(seed) * (cos_theta_max - 1.0f);
	float phi = 2.0f * 3.14159265358979323846f * random_float(seed);
	float r = sqrtf(1.0f - z * z);

	float3 direction = normalize(r * cosf(phi) * u + r * sinf(phi) * v + z * w);

	// Recover the sampled surface point (near root of ray-sphere) to get its
	// UV/normal - an algebraic identity given `direction` was drawn to hit
	// the sphere, not an approximation.
	float3 oc = origin - center;
	float b = dot(oc, direction);
	float c = dot(oc, oc) - sphere.radius * sphere.radius;
	float disc = fmaxf(0.0f, b * b - c);
	float t_near = -b - sqrtf(disc);
	float3 point = origin + t_near * direction;
	float3 local = (point - center) / sphere.radius;
	out_normal = local;

	// UV convention must match whatever the DIRECT-hit closest-hit program
	// uses for this same shapeKind, or a "filename"-textured light samples a
	// different texel via NEE than a camera ray hitting it directly (a real,
	// visible mismatch, not just noise) - a ClippedSphere's direct hit uses
	// pbrt-v4's Z-pole convention (__closesthit__sphere's own comment), not
	// this function's plain-sphere Y-pole one, since zMin/zMax/phiMax are
	// themselves Z-pole-defined. `local`/`sphere.center`/`sphere.radius`
	// above stay the full-sphere-cone approximation (this function's own
	// established, accepted geometric/pdf simplification - see
	// SphereData::center's comment) - only the UV derivation below is
	// shapeKind-aware, by transforming the sampled point through the real
	// object-space affine purely to get a real, direct-hit-consistent (u,v).
	if (sphere.shapeKind == GpuMediumShapeKind::ClippedSphere) {
		const float3 objPt = dc_apply_point(sphere.w2o, point);
		const float rl = sphere.radiusLocal;
		const float cosTheta = fminf(1.0f, fmaxf(-1.0f, (rl > 0.0f) ? (objPt.z / rl) : 0.0f));
		const float theta = acosf(cosTheta);
		float phi = atan2f(objPt.y, objPt.x);
		if (phi < 0.0f) phi += 2.0f * 3.14159265358979323846f;
		// thetaZMin/thetaZMax are host-precomputed (SphereData's own comment).
		out_u = (sphere.phiMax > 1e-8f) ? (phi / sphere.phiMax) : 0.0f;
		out_v = (sphere.thetaZMax > sphere.thetaZMin)
			? (theta - sphere.thetaZMin) / (sphere.thetaZMax - sphere.thetaZMin) : 0.0f;
	} else {
		const float sphere_theta = acosf(fmaxf(-1.0f, fminf(1.0f, -local.y)));
		const float sphere_phi = atan2f(-local.z, local.x) + 3.14159265358979323846f;
		out_u = sphere_phi / (2.0f * 3.14159265358979323846f);
		out_v = sphere_theta / 3.14159265358979323846f;
	}

	return direction;
}

// Sample a random point on a quad light
//
// out_u/out_v: the sampled point's own (a,b) parametrization, which IS the
// alpha/beta UV __closesthit__quad recomputes for a direct hit (same
// planar-decomposition convention) - no extra work, just exposing the
// (a,b) this function already draws.
__device__ __forceinline__ float3 sample_quad_light(
	const QuadData& quad,
	const float3& origin,
	unsigned int& seed,
	float& pdf,
	float& out_dist,
	float& out_u,
	float& out_v
) {
	// Random point on quad surface
	float a = random_float(seed);
	float b = random_float(seed);
	float3 point = quad.Q + a * quad.u + b * quad.v;
	out_u = a;
	out_v = b;

	// Direction to sampled point
	float3 to_light = point - origin;
	float dist_sq = dot(to_light, to_light);
	out_dist = sqrtf(dist_sq);
	float3 direction = to_light / out_dist;

	// Area-based PDF converted to solid angle
	float area = length(quad.w);  // w = u x v, so |w| = area
	float cosine = fabsf(dot(direction, quad.normal));

	if (cosine < 1e-6f || area < 1e-6f) {
		pdf = 0.0f;
		return direction;
	}

	pdf = dist_sq / (cosine * area);
	return direction;
}

// Sample a random point on a triangle light.
//
// Same shape of answer as sample_quad_light above - a direction plus a
// solid-angle pdf - and deliberately the same structure, because the only
// real differences are how a uniform point is drawn and that the area is half
// a parallelogram's.
//
// Most pbrt area lights never reach here: they arrive as triangle PAIRS that
// pbrt_quadify.h rejoins into one parallelogram, which is both cheaper to
// sample and what the quad path already did well. This is for the ones that
// will not merge - an odd triangle, a fan, anything non-parallelogram - which
// before this existed were emitted as geometry that glows when hit but that
// next-event estimation could not aim at.
__device__ __forceinline__ float3 sample_triangle_light(
	const TriangleData& tri,
	const float3& origin,
	unsigned int& seed,
	float& pdf,
	float& out_dist,
	float& out_u,
	float& out_v,
	float3& out_normal
) {
	// Uniform point on the triangle by folding the unit square onto it: draw
	// (a,b) in [0,1]^2 and reflect the half that falls outside a+b<=1 back
	// through the diagonal. Area-preserving, so the point stays uniform, and
	// branch-cheap compared with the sqrt form.
	float a = random_float(seed);
	float b = random_float(seed);
	if (a + b > 1.0f) { a = 1.0f - a; b = 1.0f - b; }

	const float3 e1 = tri.p1 - tri.p0;
	const float3 e2 = tri.p2 - tri.p0;
	const float3 point = tri.p0 + a * e1 + b * e2;

	// Same b0/b1/b2 = (1-a-b, a, b) convention __anyhit__triangle/
	// __closesthit__triangle use via optixGetTriangleBarycentrics() (see
	// optix_intersection_triangle.h's own comment) - a weights p1, b weights
	// p2, so this needs no reordering to land on the same UV a direct hit
	// on this exact sampled point would compute.
	out_u = out_v = 0.0f;
	if (tri.hasUVs) {
		const float b0 = 1.0f - a - b;
		out_u = b0 * tri.uv0.x + a * tri.uv1.x + b * tri.uv2.x;
		out_v = b0 * tri.uv0.y + a * tri.uv1.y + b * tri.uv2.y;
	}

	float3 to_light = point - origin;
	float dist_sq = dot(to_light, to_light);
	out_dist = sqrtf(dist_sq);
	if (out_dist < 1e-6f) { pdf = 0.0f; out_normal = make_float3(0.0f, 0.0f, 1.0f); return make_float3(0.0f, 0.0f, 1.0f); }
	float3 direction = to_light / out_dist;

	// GEOMETRIC normal, not the interpolated shading normal: this converts an
	// area measure to a solid-angle one, which is a property of the surface
	// the sample was actually drawn on.
	const float3 n_unnorm = cross(e1, e2);
	const float twice_area = length(n_unnorm);
	const float area = 0.5f * twice_area;      // half the parallelogram
	if (twice_area < 1e-12f) { pdf = 0.0f; out_normal = direction; return direction; }
	const float3 normal = n_unnorm / twice_area;
	out_normal = normal;

	const float cosine = fabsf(dot(direction, normal));
	if (cosine < 1e-6f || area < 1e-12f) {
		pdf = 0.0f;
		return direction;
	}

	pdf = dist_sq / (cosine * area);
	return direction;
}

// Same shape of answer as sample_triangle_light above - a direction plus a
// solid-angle pdf - for Shape "bilinearmesh" area lights (GpuLightKind::
// BilinearPatch). blp_sample (src/shared/bilinear_patch.h, CPU_GPU-tagged,
// shared with the CPU builder's own NEE hooks - see
// bilinear_patch_hittable::random() in scenes_advanced.h) draws a uniform-
// area point and returns an AREA-domain pdf; the area-to-solid-angle
// Jacobian conversion below is the same one sample_triangle_light applies,
// using blp_sample's own returned normal rather than recomputing one.
__device__ __forceinline__ float3 sample_bilinear_patch_light(
	const BilinearPatchData& bp,
	const float3& origin,
	unsigned int& seed,
	float& pdf,
	float& out_dist,
	float& out_u,
	float& out_v,
	float3& out_normal
) {
	const float p00[3] = {bp.p00.x, bp.p00.y, bp.p00.z};
	const float p10[3] = {bp.p10.x, bp.p10.y, bp.p10.z};
	const float p01[3] = {bp.p01.x, bp.p01.y, bp.p01.z};
	const float p11[3] = {bp.p11.x, bp.p11.y, bp.p11.z};
	const float u2[2] = {random_float(seed), random_float(seed)};
	float outP[3], outN[3], areaPdf = 0.0f, su = 0.0f, sv = 0.0f;
	blp_sample(p00, p10, p01, p11, u2, outP, outN, &areaPdf, &su, &sv);
	out_u = su;
	out_v = sv;

	const float3 point = make_float3(outP[0], outP[1], outP[2]);
	const float3 normal = make_float3(outN[0], outN[1], outN[2]);
	out_normal = normal;
	float3 to_light = point - origin;
	float dist_sq = dot(to_light, to_light);
	out_dist = sqrtf(dist_sq);
	if (out_dist < 1e-6f || areaPdf <= 0.0f) { pdf = 0.0f; return make_float3(0.0f, 0.0f, 1.0f); }
	float3 direction = to_light / out_dist;

	const float cosine = fabsf(dot(direction, normal));
	if (cosine < 1e-6f) { pdf = 0.0f; return direction; }

	pdf = areaPdf * dist_sq / cosine;
	return direction;
}

// Same shape of answer as sample_quad_light/sample_bilinear_patch_light
// above, for Shape "disk"/"cylinder" area lights (GpuLightKind::Disk/
// Cylinder). dc_sample_disk (optix_disk_cylinder_helpers.h, included
// earlier by optix_programs.cu specifically so it's available here - see
// that header's own comment) draws a uniform-area point in WORLD space and
// returns an AREA-domain pdf; the area-to-solid-angle Jacobian conversion
// below is the same one every other *_light sampler in this file applies.
// out_u/out_v: recomputed from the sampled world point via the same
// object-space phi/radial-fraction formula __closesthit__disk uses for a
// direct hit (this file's own "recompute from the point" convention).
__device__ __forceinline__ float3 sample_disk_light(
	const DiskData& disk,
	const float3& origin,
	unsigned int& seed,
	float& pdf,
	float& out_dist,
	float& out_u,
	float& out_v,
	float3& out_normal
) {
	float3 point, normal; float area_pdf;
	dc_sample_disk(disk, random_float(seed), random_float(seed), point, normal, area_pdf);
	out_normal = normal;
	{
		const float3 obj_pt = dc_apply_point(disk.w2o, point);
		float uv_phi = atan2f(obj_pt.y, obj_pt.x);
		if (uv_phi < 0.0f) uv_phi += 6.283185307179586f;
		const float uv_dist = sqrtf(obj_pt.x * obj_pt.x + obj_pt.y * obj_pt.y);
		out_u = uv_phi / disk.phiMax;
		out_v = (disk.radius > disk.innerRadius)
			? 1.0f - (uv_dist - disk.innerRadius) / (disk.radius - disk.innerRadius)
			: 0.0f;
	}
	float3 to_light = point - origin;
	float dist_sq = dot(to_light, to_light);
	out_dist = sqrtf(dist_sq);
	if (out_dist < 1e-6f || area_pdf <= 0.0f) { pdf = 0.0f; return make_float3(0.0f, 0.0f, 1.0f); }
	float3 direction = to_light / out_dist;

	const float cosine = fabsf(dot(direction, normal));
	if (cosine < 1e-6f) { pdf = 0.0f; return direction; }

	pdf = area_pdf * dist_sq / cosine;
	return direction;
}

// out_u/out_v: recomputed from the sampled world point via the same
// object-space phi/z-fraction formula __closesthit__cylinder uses for a
// direct hit.
__device__ __forceinline__ float3 sample_cylinder_light(
	const CylinderData& cyl,
	const float3& origin,
	unsigned int& seed,
	float& pdf,
	float& out_dist,
	float& out_u,
	float& out_v,
	float3& out_normal
) {
	float3 point, normal; float area_pdf;
	dc_sample_cylinder(cyl, random_float(seed), random_float(seed), point, normal, area_pdf);
	out_normal = normal;
	{
		const float3 obj_pt = dc_apply_point(cyl.w2o, point);
		float uv_phi = atan2f(obj_pt.y, obj_pt.x);
		if (uv_phi < 0.0f) uv_phi += 6.283185307179586f;
		out_u = uv_phi / cyl.phiMax;
		out_v = (cyl.zMax > cyl.zMin) ? (obj_pt.z - cyl.zMin) / (cyl.zMax - cyl.zMin) : 0.0f;
	}
	float3 to_light = point - origin;
	float dist_sq = dot(to_light, to_light);
	out_dist = sqrtf(dist_sq);
	if (out_dist < 1e-6f || area_pdf <= 0.0f) { pdf = 0.0f; return make_float3(0.0f, 0.0f, 1.0f); }
	float3 direction = to_light / out_dist;

	const float cosine = fabsf(dot(direction, normal));
	if (cosine < 1e-6f) { pdf = 0.0f; return direction; }

	pdf = area_pdf * dist_sq / cosine;
	return direction;
}

// NEE texture lookup for a "filename"/map_Ke area light, shared by every
// non-Triangle case in sample_area_light_by_kind() below. Deliberately the
// same hand-inlined Image-kind logic as Triangle's own copy (this file, a
// few lines below) rather than a call to the shared sample_texture() helper
// - see Triangle's own comment for why that call specifically crashes from
// inside this function; since __forceinline__ guarantees this collapses
// into the caller exactly like a hand-copy would, one shared copy here is
// the same generated code as five separate ones, just not five separately-
// maintained ones.
__device__ __forceinline__ float3 nee_light_texture_emission(const MaterialData& lm, float lu, float lv) {
	if (lm.textureIdx < 0) return lm.emission;
	const TextureData& dtex = params.textures[lm.textureIdx];
	if (dtex.kind == TextureKind::Image && dtex.width > 0 && dtex.height > 0) {
		// Wrap-mode-aware, matching sample_texture()'s own sampleImage lambda
		// (this file) - currently a no-op in practice (an emission texture is
		// always built with the default Clamp wrap, see getOrBuildPbrtImage
		// Texture()'s own call sites, pbrt_gpu_builder.h), but kept in sync so
		// this copy doesn't silently diverge if wrap support is ever extended
		// to AreaLightSource "filename" textures too.
		const float uw = fminf(fmaxf(lu, -1024.0f), 1024.0f);
		const float vw = fminf(fmaxf(1.0f - lv, -1024.0f), 1024.0f);
		int ti = static_cast<int>(floor((double)uw * dtex.width));
		int tj = static_cast<int>(floor((double)vw * dtex.height));
		switch (dtex.wrapMode) {
		case GpuWrapMode::Repeat:
			ti = ((ti % dtex.width) + dtex.width) % dtex.width;
			tj = ((tj % dtex.height) + dtex.height) % dtex.height;
			break;
		case GpuWrapMode::Black:
			if (ti < 0 || ti >= dtex.width || tj < 0 || tj >= dtex.height) return make_float3(0.0f, 0.0f, 0.0f);
			break;
		case GpuWrapMode::Clamp:
			ti = min(max(ti, 0), dtex.width - 1);
			tj = min(max(tj, 0), dtex.height - 1);
			break;
		}
		const unsigned char* px = params.texturePixels + dtex.pixelOffset + (tj * dtex.width + ti) * 3;
		constexpr float kCS = 1.0f / 255.0f;
		return make_float3(px[0]*kCS*lm.emissionScale, px[1]*kCS*lm.emissionScale, px[2]*kCS*lm.emissionScale);
	}
	return make_float3(0.0f, 1.0f, 1.0f);
}

// Zeroes `emission` when the light material is one-sided (mat.twoSided ==
// false) and the sampled surface point's outward normal faces AWAY from the
// receiver - the NEE counterpart of material_emission()'s own front_face
// gate on a direct hit (this file, above). `direction` points from the
// receiver toward the light (same sense optix_intersection_*.h's `ray_dir`
// has at a direct hit), so front-facing is dot(direction, light_normal) < 0,
// matching front_face's own convention exactly. Was previously never
// checked for ANY light kind here (including Triangle) - every one-sided
// area light was silently treated as two-sided by NEE, contributing light
// from its back face that a direct BSDF-sampled ray hitting the same face
// would correctly have shown as unlit.
__device__ __forceinline__ void nee_gate_one_sided(
	const MaterialData& lm, const float3& direction, const float3& light_normal, float3& emission
) {
	if (!lm.twoSided && dot(direction, light_normal) >= 0.0f)
		emission = make_float3(0.0f, 0.0f, 0.0f);
}

// Sample whichever kind of area light `light_idx` names, and report the
// emitter's radiance alongside it.
//
// One helper rather than the same three-way branch written out at each of the
// next-event sites below - they had already drifted into two slightly
// different spellings of the two-way version, and adding a third kind to each
// by hand is exactly how a backend ends up sampling a light one way and
// weighting it another.
__device__ __forceinline__ float3 sample_area_light_by_kind(
	int light_idx,
	const float3& origin,
	unsigned int& seed,
	float& geom_pdf,
	float& max_dist,
	float3& emission,
	float ray_time
) {
	const int prim_idx = params.lightIndices[light_idx];
	switch (params.lightKinds[light_idx]) {
	case GpuLightKind::Sphere: {
		const SphereData& s = params.spheres[prim_idx];
		float su, sv; float3 snormal;
		const float3 dir = sample_sphere_light(s, origin, seed, geom_pdf, su, sv, snormal, ray_time);
		// Distance to the CENTRE, matching what this path has always used to
		// bound the shadow ray for a sphere light. Interpolated the same way
		// sample_sphere_light() itself just did (see that function's own
		// comment) - a moving sphere light's shadow-ray bound has to agree
		// with the position its own NEE sample was actually taken against.
		// ray_time is the SAME parameter just passed to sample_sphere_light()
		// above, not a fresh optixGetRayTime() read - see this function's own
		// parameter comment for why an implicit "current ray" read isn't
		// always legal here.
		{
			const float3 center = make_float3(
				s.center.x + ray_time * (s.center1.x - s.center.x),
				s.center.y + ray_time * (s.center1.y - s.center.y),
				s.center.z + ray_time * (s.center1.z - s.center.z));
			max_dist = length(center - origin);
		}
		const MaterialData& sm = params.materials[s.materialIdx];
		emission = nee_light_texture_emission(sm, su, sv);
		nee_gate_one_sided(sm, dir, snormal, emission);
		return dir;
	}
	case GpuLightKind::Triangle: {
		// The scene's own triangle array - an instanced triangle is never
		// emissive (flatten() bakes emitters per placement into world space),
		// so no per-instance base offset applies here.
		const TriangleData& t = params.triangles[prim_idx];
		float lu, lv; float3 tnormal;
		const float3 dir = sample_triangle_light(t, origin, seed, geom_pdf, max_dist, lu, lv, tnormal);
		const MaterialData& lm = params.materials[t.materialIdx];
		// A pbrt AreaLightSource "filename" triangle light needs the real
		// sampled UV to look up its image, matching material_emission()'s
		// direct-hit texture lookup (this is the NEE counterpart of it) -
		// everything else (map_Ke-textured triangles, flat-color lights)
		// keeps reading mat.emission raw exactly as before.
		//
		// Deliberately NOT a call to the shared sample_texture() helper
		// (used successfully by material_emission() and Lambertian albedo
		// lookups elsewhere in this same file): calling it from THIS call
		// site specifically produced a reproducible CUDA 700 illegal-
		// memory-access as soon as a scene actually exercised NEE against a
		// textured light (a plain direct-hit-only scene never crashed) -
		// confirmed by bisection: reverting to this identical hand-inlined
		// copy of sample_texture()'s own Image-kind logic made the crash
		// disappear with no other change. Root cause not established
		// (suspected codegen/inlining-depth interaction specific to this
		// call site inside the recursive backend's single-module mega-
		// kernel - sample_area_light_by_kind() is itself already inlined
		// into every one of shade_material()'s many NEE call sites), so
		// this stays a hand-inlined duplicate rather than a call, matching
		// this codebase's own established fallback for GPU codegen
		// surprises (see the recursive-backend member-call stall memory/
		// this file's own precedent for hand-duplicating rather than
		// calling in a hazardous context).
		if (lm.textureIdx >= 0) {
			const TextureData& dtex = params.textures[lm.textureIdx];
			if (dtex.kind == TextureKind::Image && dtex.width > 0 && dtex.height > 0) {
				// Wrap-mode-aware, matching sample_texture()'s own sampleImage
				// lambda and nee_light_texture_emission()'s identical fix
				// above (this file) - see that one's own comment for why this
				// is currently a no-op (emission textures always resolve to
				// Clamp) kept in sync for the future rather than a live bug.
				const float uw = fminf(fmaxf(lu, -1024.0f), 1024.0f);
				const float vw = fminf(fmaxf(1.0f - lv, -1024.0f), 1024.0f);
				int ti = static_cast<int>(floor((double)uw * dtex.width));
				int tj = static_cast<int>(floor((double)vw * dtex.height));
				bool blackWrap = false;
				switch (dtex.wrapMode) {
				case GpuWrapMode::Repeat:
					ti = ((ti % dtex.width) + dtex.width) % dtex.width;
					tj = ((tj % dtex.height) + dtex.height) % dtex.height;
					break;
				case GpuWrapMode::Black:
					blackWrap = (ti < 0 || ti >= dtex.width || tj < 0 || tj >= dtex.height);
					break;
				case GpuWrapMode::Clamp:
					ti = min(max(ti, 0), dtex.width - 1);
					tj = min(max(tj, 0), dtex.height - 1);
					break;
				}
				if (blackWrap) {
					emission = make_float3(0.0f, 0.0f, 0.0f);
				} else {
					const unsigned char* px = params.texturePixels + dtex.pixelOffset + (tj * dtex.width + ti) * 3;
					constexpr float kCS = 1.0f / 255.0f;
					emission = make_float3(px[0]*kCS*lm.emissionScale, px[1]*kCS*lm.emissionScale, px[2]*kCS*lm.emissionScale);
				}
			} else {
				emission = make_float3(0.0f, 1.0f, 1.0f);
			}
		} else {
			emission = lm.emission;
		}
		nee_gate_one_sided(lm, dir, tnormal, emission);
		return dir;
	}
	case GpuLightKind::BilinearPatch: {
		const BilinearPatchData& bp = params.bilinearPatches[prim_idx];
		float bu, bv; float3 bnormal;
		const float3 dir = sample_bilinear_patch_light(bp, origin, seed, geom_pdf, max_dist, bu, bv, bnormal);
		const MaterialData& bm = params.materials[bp.materialIdx];
		emission = nee_light_texture_emission(bm, bu, bv);
		nee_gate_one_sided(bm, dir, bnormal, emission);
		return dir;
	}
	case GpuLightKind::Disk: {
		const DiskData& d = params.disks[prim_idx];
		float du, dv; float3 dnormal;
		const float3 dir = sample_disk_light(d, origin, seed, geom_pdf, max_dist, du, dv, dnormal);
		const MaterialData& dm = params.materials[d.materialIdx];
		emission = nee_light_texture_emission(dm, du, dv);
		nee_gate_one_sided(dm, dir, dnormal, emission);
		return dir;
	}
	case GpuLightKind::Cylinder: {
		const CylinderData& c = params.cylinders[prim_idx];
		float cu, cv; float3 cnormal;
		const float3 dir = sample_cylinder_light(c, origin, seed, geom_pdf, max_dist, cu, cv, cnormal);
		const MaterialData& cm = params.materials[c.materialIdx];
		emission = nee_light_texture_emission(cm, cu, cv);
		nee_gate_one_sided(cm, dir, cnormal, emission);
		return dir;
	}
	case GpuLightKind::Quad:
	default: {
		const QuadData& q = params.quads[prim_idx];
		float qu, qv;
		const float3 dir = sample_quad_light(q, origin, seed, geom_pdf, max_dist, qu, qv);
		const MaterialData& qm = params.materials[q.materialIdx];
		emission = nee_light_texture_emission(qm, qu, qv);
		nee_gate_one_sided(qm, dir, q.normal, emission);
		return dir;
	}
	}
}

// pbrt-v4 bounding-cone light BVH (GpuCameraParams::cameraMediumSigmaT's
// sibling gap - see OptiXRenderer::d_lightBvhNodes_'s own comment,
// optix_renderer.h, for the host build/upload). Device-side mirror of
// BVHLightSampler2::Sample()/PMF() (src/shared/bvh_light_sampler2.h) - that
// class's own query methods aren't called directly here because they
// operate on a std::vector-backed `nodes_`/std::unordered_map-backed
// `lightToBitTrail_` (host-only containers); these two functions instead
// walk the flat `params.lightBvhNodes`/`params.lightBvhBitTrail` device
// arrays that class's own build already produced, using the exact same
// LightBVHNode::lightBounds.Importance() (CompactLightBounds, light_bvh_
// node.h) stochastic-descent algorithm.
//
// FORMERLY a KNOWN UNRESOLVED BUG - reading params.lightBvhNodes[nodeIndex]/
// params.lightBvhBitTrail[lightIndex] during traversal reproducibly
// triggered a CUDA 700 "illegal memory access" on GPU-recursive for at
// least one real multi-light scene (pbrt_scenes/triangle-fan-light.pbrt, 5
// lights, 9-node tree), even though the tree itself was directly verified
// well-formed (host- and device-side dumps of every node's isLeaf/
// childOrLightIndex matched exactly, and sizeof(LightBVHNode)/sizeof(
// CompactLightBounds) matched exactly between the MSVC host build and the
// NVCC device build - ruling out a struct-layout/ABI mismatch). A later
// diagnosis session instrumented every array read in both functions below
// with a bounds check that records what actually went out of range instead
// of crashing (a CAS-guarded single coherent snapshot, since concurrent
// GPU threads racing to write a shared debug buffer produce self-
// contradictory "torn" values otherwise) - across repeated full
// reproductions of the exact crashing scene, the actual out-of-range
// index/value seen varied between runs and was itself sometimes NOT out of
// range by the numbers alone (consistent with a genuine compiler/codegen-
// level bug in this exact NVCC/OptiX toolchain - this file's own established
// class of prior bug, see gpu_cloud_density()'s dnoise() history - rather
// than a logic error in the tree data or the traversal code as written).
// The exact root cause was still NOT established, but the bounds checks
// themselves reliably and reproducibly eliminate the crash: every one of
// the checks below returns a safe "treat as no light BVH data available
// here" fallback instead of dereferencing an address that may not
// (depending on whatever miscompiles) actually hold what the index
// arithmetic says it should. `LaunchParams::lightBvhNodeCount` is now set
// for real (OptiXRenderer::render(), optix_renderer_render.cpp) - the
// guards below are what makes that safe to do. If a bounds check below
// ever actually fires on real hardware, that specific NEE decision falls
// back to zero light-BVH contribution for that one sample (a rare,
// self-correcting bias - MIS/the alias-table-based selection elsewhere in
// the same frame still lights the scene) rather than crashing the render.
//
// Bounds+monotonicity guards - both gpu_light_bvh_sample_index() and
// gpu_light_bvh_pmf() below hand-duplicate the same two checks at their own
// 2 call sites each (4 total), rather than sharing them via a helper
// function - matching this file's own established "hand-duplicate at the
// call site" convention for this exact recursive mega-kernel (see the
// "shared-function-call codegen/inlining issue" ruled-out theory above,
// and GpuLightBvhSample's own by-value-return precedent below, both from
// the ORIGINAL crash diagnosis). NOTE: while fixing this, `optixModuleCreate`
// was observed taking ~2 minutes regardless of whether these checks were
// shared or duplicated, or even present at all - traced to something
// unrelated to this specific change (the byte-identical, already-shipped
// baseline showed the same delay) rather than caused by this fix; kept the
// duplicated form anyway since it costs nothing extra and matches this
// file's own established convention. See this file's own header comment
// above for the crash-diagnosis history these checks exist to guard
// against. A code-review pass on the first version
// of this fix found it caught out-of-range indices but not two other real
// gaps, both fixed at each of the 4 sites below: (1) the interior-node
// child check now rejects not just an out-of-range c1Index but also one
// that doesn't strictly follow the current node - the forward-progress
// invariant a well-formed flattened BVH always guarantees (see
// BVHLightSampler2::buildBVH()'s own `nodeIndex + 1 == i0` assertion,
// src/shared/bvh_light_sampler2.h: the right child always comes strictly
// after the ENTIRE left subtree, which itself occupies at least node
// nodeIndex+1, so a genuine right-child index can never be <= nodeIndex+1).
// Catching this - not just range - is what prevents a corrupted-but-in-
// range index (this file's own comment above notes the observed
// corruption "was itself sometimes NOT out of range by the numbers alone")
// from turning either traversal's `while(true)` loop into an infinite loop
// (a GPU hang / driver TDR) instead of the safe fallback a pure range
// check alone does not guarantee. (2) gpu_light_bvh_sample_index()'s own
// leaf branch now bounds-checks `childOrLightIndex` before returning it -
// a LIGHT index (into `params.lightIndices`/`lightKinds`), a completely
// different range than node indices, that a caller dereferences those
// arrays with unchecked otherwise.
//
// KNOWN UNRESOLVED BUG (found while authoring pbrt_scenes/gpu-light-bvh-
// many-lights.pbrt, the first scene with a real multi-level tree - 12
// lights, 23 nodes; every scene this feature had been verified against
// before that had either a single trivial leaf-only tree or a shallow
// 9-node one): on a tree this size, the monotonicity guard below was
// observed - via device-side printf tracing - REJECTING a demonstrably
// valid, in-range, monotonic c1Index at the tree's own root, producing a
// fully black render (NEE silently returning "no light" on every call)
// despite the host-side BVHLightSampler2::Sample() (the exact same
// algorithm, same built tree, called from scene_builder.cpp) handling the
// identical query correctly. Multiple independent mitigations were tried
// and NONE resolved it: (1) reading `params.lightBvhNodes[nodeIndex]` into
// a by-value `LightBVHNode` local instead of a `const LightBVHNode&`
// reference, (2) splitting the combined `a || b` guard condition into two
// separate sequential `if` statements, (3) marking both functions
// `__noinline__` instead of `__forceinline__` (the fix pattern that
// resolved the unrelated "member-call stall" precedent elsewhere in this
// codebase). A printf placed immediately before the failing `if`, printing
// its own pre-computed operands, showed values that do not satisfy the
// condition being taken (e.g. `c1Index=18 nodeCount=23` failing a
// `c1Index >= nodeCount` check) - not explainable by the guard logic
// itself under any of the three tried structures. This looks like a
// genuine NVCC/OptiX codegen bug specific to this exact recursive mega-
// kernel under real interior-node depth, in the same toolchain-fragility
// family as GpuLightBvhSample's own by-value-return fix below and the
// "member-call stall" precedent - but unlike those, NOT resolved here.
// Re-verified the SAME build still renders a previously-working 5-light/
// 9-node scene (triangle-fan-light.pbrt) correctly, so this is not a
// blanket regression - it is specific to deeper/larger trees, exact
// trigger unknown. Left as `__forceinline__`/reference/combined-guard (the
// last VERIFIED-correct form) rather than keeping an unverified "fix" that
// demonstrably did not fix the reproducer. Anyone picking this up next
// should reach for compute-sanitizer or Nsight Compute rather than printf -
// see this file's own earlier crash-diagnosis history for the CAS-guarded-
// debug-buffer technique that worked for a different symptom.

// Return type for gpu_light_bvh_sample_index() - a plain by-value struct,
// deliberately NOT a `float&`/`int&` reference-output parameter. This
// codebase's own memory of a prior GPU recursive-backend miscompile
// (CloudMedium::compute_density()'s dnoise() helper, see gpu_cloud_density()'s
// own history) found reference-output device functions unreliable in this
// exact NVCC/OptiX toolchain when NOT force-inlined - by-value struct
// returns sidestep that class of bug entirely, matching that fix's own
// "called by value, no reference/pointer output params" guidance.
struct GpuLightBvhSample {
	int lightIndex;  // -1 = no light BVH built, or zero importance everywhere
	float pmf;
};

// gpu_light_bvh_sample_index: returns the selected light's index (or -1 if
// no light BVH was built for this scene, or every light's importance at
// this point is zero) and its selection PMF - a drop-in replacement for the
// alias table's `selection_pdf` at every call site below, since
// `light_pdf = pmf * geom_pdf` is the same formula either way.
__device__ __forceinline__ GpuLightBvhSample gpu_light_bvh_sample_index(
	float px, float py, float pz, float nx, float ny, float nz, float u)
{
	if (params.lightBvhNodeCount <= 0) return GpuLightBvhSample{-1, 0.f};
	int nodeIndex = 0;
	float pmf = 1.f;
	u = fminf(u, 1.f - 1e-7f);
	while (true) {
		// Bounds guard - see this file's own header comment above.
		if (nodeIndex < 0 || nodeIndex >= params.lightBvhNodeCount) {
			return GpuLightBvhSample{-1, 0.f};
		}
		const LightBVHNode& node = params.lightBvhNodes[nodeIndex];
		if (!node.isLeaf) {
			const int c1Index = (int)node.childOrLightIndex;
			// Range AND forward-progress guard - see this file's own
			// header comment above for why `c1Index <= nodeIndex + 1` is
			// rejected too, not just an out-of-range one.
			if (c1Index <= nodeIndex + 1 || c1Index >= params.lightBvhNodeCount) {
				return GpuLightBvhSample{-1, 0.f};
			}
			const LightBVHNode& c0 = params.lightBvhNodes[nodeIndex + 1];
			const LightBVHNode& c1 = params.lightBvhNodes[node.childOrLightIndex];
			float ci0 = c0.lightBounds.Importance(px,py,pz, nx,ny,nz,
				params.lightBvhAllBMinX,params.lightBvhAllBMinY,params.lightBvhAllBMinZ,
				params.lightBvhAllBMaxX,params.lightBvhAllBMaxY,params.lightBvhAllBMaxZ);
			float ci1 = c1.lightBounds.Importance(px,py,pz, nx,ny,nz,
				params.lightBvhAllBMinX,params.lightBvhAllBMinY,params.lightBvhAllBMinZ,
				params.lightBvhAllBMaxX,params.lightBvhAllBMaxY,params.lightBvhAllBMaxZ);
			if (ci0 == 0.f && ci1 == 0.f) return GpuLightBvhSample{-1, 0.f};
			float sum = ci0 + ci1;
			float nodePMF; int child;
			if (u < ci0 / sum) { child = 0; nodePMF = ci0 / sum; u = u / nodePMF; }
			else { child = 1; nodePMF = ci1 / sum; u = (u - ci0/sum) / nodePMF; }
			u = fminf(u, 1.f - 1e-7f);
			pmf *= nodePMF;
			nodeIndex = (child == 0) ? (nodeIndex + 1) : (int)node.childOrLightIndex;
		} else {
			// Leaf's own bounds guard - childOrLightIndex here is a LIGHT
			// index (into params.lightIndices/lightKinds), a completely
			// different range than node indices - see this file's own
			// header comment above.
			if (node.childOrLightIndex >= (unsigned int)params.numLights) {
				return GpuLightBvhSample{-1, 0.f};
			}
			return GpuLightBvhSample{(int)node.childOrLightIndex, pmf};
		}
	}
}

// gpu_light_bvh_pmf: replays the bit-trail for `lightIndex` to recompute its
// selection PMF at THIS shading point (position-dependent, unlike the alias
// table's fixed pdf) - needed at a BSDF-sampled light hit to MIS-weight
// against whatever NEE would have picked from the point the BSDF sample was
// actually taken (see this function's callers in the 6 shape closest-hit
// files' own DiffuseLight branches). Returns 0 if no light BVH was built, or
// if every ancestor's combined importance was zero (can't happen for a real
// bit-trail from a light actually in the tree, but matches PMF()'s own
// defensive return).
__device__ __forceinline__ float gpu_light_bvh_pmf(
	float px, float py, float pz, float nx, float ny, float nz, int lightIndex)
{
	if (params.lightBvhNodeCount <= 0) return 0.f;
	// Bounds guards below - see this file's own header comment above
	// (gpu_light_bvh_sample_index's) for why these exist and why a safe
	// fallback rather than an assert/crash is the right response here.
	if (lightIndex < 0 || (unsigned int)lightIndex >= params.numLights) {
		return 0.f;
	}
	uint32_t bitTrail = params.lightBvhBitTrail[lightIndex];
	float pmf = 1.f;
	int nodeIndex = 0;
	while (true) {
		// Bounds guard - see this file's own header comment above.
		if (nodeIndex < 0 || nodeIndex >= params.lightBvhNodeCount) {
			return 0.f;
		}
		const LightBVHNode& node = params.lightBvhNodes[nodeIndex];
		if (node.isLeaf) return pmf;
		const int c1Index = (int)node.childOrLightIndex;
		// Range AND forward-progress guard - see this file's own header
		// comment above for why `c1Index <= nodeIndex + 1` is rejected
		// too, not just an out-of-range one.
		if (c1Index <= nodeIndex + 1 || c1Index >= params.lightBvhNodeCount) {
			return 0.f;
		}
		const LightBVHNode& c0 = params.lightBvhNodes[nodeIndex + 1];
		const LightBVHNode& c1 = params.lightBvhNodes[node.childOrLightIndex];
		float ci0 = c0.lightBounds.Importance(px,py,pz, nx,ny,nz,
			params.lightBvhAllBMinX,params.lightBvhAllBMinY,params.lightBvhAllBMinZ,
			params.lightBvhAllBMaxX,params.lightBvhAllBMaxY,params.lightBvhAllBMaxZ);
		float ci1 = c1.lightBounds.Importance(px,py,pz, nx,ny,nz,
			params.lightBvhAllBMinX,params.lightBvhAllBMinY,params.lightBvhAllBMinZ,
			params.lightBvhAllBMaxX,params.lightBvhAllBMaxY,params.lightBvhAllBMaxZ);
		float sum = ci0 + ci1;
		if (sum == 0.f) return 0.f;
		int branch = (int)(bitTrail & 1u);
		pmf *= (branch == 0 ? ci0 : ci1) / sum;
		nodeIndex = (branch == 0) ? (nodeIndex + 1) : (int)node.childOrLightIndex;
		bitTrail >>= 1;
	}
}

// Selects one area light (light BVH when the scene built one - see
// gpu_light_bvh_sample_index()'s own comment - else the power-weighted
// alias table, falling back further to uniform selection when neither was
// built), samples a direction toward it via sample_area_light_by_kind(),
// and returns the combined selection*geometric PDF. Every NEE call site in
// shade_material() used to spell this exact selection dance out by hand -
// one helper keeps them from drifting into slightly different selection
// logic per material.
// Returns false (light_pdf left untouched) when there are no lights, or
// when the sampled direction's combined pdf is too small to divide by -
// callers should skip their light contribution in that case, exactly as
// they did before this was factored out.
__device__ __forceinline__ bool sample_nee_light(
	const float3& origin,
	unsigned int& seed,
	float3& to_light,
	float3& sampled_light_emission,
	float& max_dist,
	float& light_pdf,
	float ray_time
) {
	if (params.numLights <= 0) return false;

	int light_idx;
	float selection_pdf;
	// Light BVH first (real spatial+power selection, position-dependent -
	// see gpu_light_bvh_sample_index()'s own comment); n=0 (no surface-
	// normal weighting at the SELECTION stage) matches this function's own
	// pre-existing signature, which never threaded a normal through from
	// any of its 7 call sites - sample_area_light_by_kind() below still
	// gets the real geometric pdf regardless. Falls back to the alias
	// table/uniform selection exactly as before this feature existed for
	// any scene that didn't build a light BVH (every backend/mode other
	// than GPU-recursive, this round).
	if (params.lightBvhNodeCount > 0) {
		GpuLightBvhSample s = gpu_light_bvh_sample_index(
			origin.x, origin.y, origin.z, 0.f, 0.f, 0.f, random_float(seed));
		if (s.lightIndex < 0) return false;
		light_idx = s.lightIndex;
		selection_pdf = s.pmf;
	} else if (params.aliasTable) {
		int slot = int(random_float(seed) * float(params.numLights));
		if (slot >= int(params.numLights)) slot = int(params.numLights) - 1;
		const GpuAliasEntry& entry = params.aliasTable[slot];
		light_idx = (random_float(seed) < entry.q) ? slot : entry.alias;
		selection_pdf = params.aliasTable[light_idx].pdf;
	} else {
		light_idx = int(random_float(seed) * float(params.numLights));
		if (light_idx >= int(params.numLights)) light_idx = int(params.numLights) - 1;
		selection_pdf = 1.0f / float(params.numLights);
	}

	float geom_pdf = 0.0f;
	to_light = sample_area_light_by_kind(
		light_idx, origin, seed, geom_pdf, max_dist, sampled_light_emission, ray_time);

	light_pdf = selection_pdf * geom_pdf;
	return light_pdf > 1e-6f;
}

// Evaluate quad light PDF for a given direction
__device__ __forceinline__ float quad_light_pdf(
	const QuadData& quad,
	const float3& origin,
	const float3& direction
) {
	// Intersect ray with quad plane
	float denom = dot(direction, quad.normal);
	if (fabsf(denom) < 1e-6f) return 0.0f;

	float t = (quad.D - dot(quad.normal, origin)) / denom;
	if (t < 0.001f) return 0.0f;

	// Check if hit point is inside quad
	float3 hit_point = origin + t * direction;
	float3 p = hit_point - quad.Q;

	// Solve for (alpha, beta) such that p = alpha*u + beta*v
	float3 n = quad.w;  // u x v
	float n_len_sq = dot(n, n);
	if (n_len_sq < 1e-6f) return 0.0f;

	float alpha = dot(cross(p, quad.v), n) / n_len_sq;
	float beta = dot(cross(quad.u, p), n) / n_len_sq;

	if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f) {
		return 0.0f;  // Outside quad
	}

	// Compute PDF
	float dist_sq = t * t * dot(direction, direction);
	float cosine = fabsf(dot(direction, quad.normal));
	float area = sqrtf(n_len_sq);

	return dist_sq / (cosine * area);
}

// Evaluate sphere light PDF for a given direction
__device__ __forceinline__ float sphere_light_pdf(
	const SphereData& sphere,
	const float3& origin,
	const float3& direction
) {
	// Check if direction intersects sphere (simplified - just use solid angle)
	float3 to_center = sphere.center - origin;
	float dist_sq = dot(to_center, to_center);

	if (dist_sq < 1e-6f) return 0.0f;

	float cos_theta_max = sqrtf(1.0f - sphere.radius * sphere.radius / dist_sq);
	float solid_angle = 2.0f * 3.14159265358979323846f * (1.0f - cos_theta_max);

	return 1.0f / solid_angle;
}

// Trace a shadow ray to test visibility
// Returns true if path to light is unoccluded (false if occluded)
__device__ __forceinline__ bool trace_shadow_ray(
	const float3& origin,
	const float3& direction,
	float max_distance
) {
	// Pack shadow payload (single bool: occluded)
	unsigned int occluded = 1;  // Default to occluded (will be set to 0 if miss)

	// Nudge the origin along the ray's own travel direction before tracing,
	// mirroring optix_raygen.h's scatter_origin = hit_point + 0.01f *
	// normalize(scatterDir) for continuation rays. Shadow rays had no such
	// offset - only tmin=0.001 - which is fine for the flat quads/spheres
	// every scene used to test GPU shadow rays, where the shading normal IS
	// the geometric normal and self-intersection can't happen at a grazing
	// angle. It falls apart for a smooth-shaded triangle mesh (per-vertex
	// interpolated normals from e.g. a loopsubdiv or OBJ "vn" import): the
	// shading normal at a point routinely diverges from its triangle's own
	// flat facet, most sharply at high-curvature areas (joints, haunches),
	// so a shadow ray toward a light can leave at a near-tangent angle to
	// the ACTUAL facet and immediately self-intersect it or a neighboring
	// facet sharing that vertex - tmin alone doesn't stop that, since it
	// only cuts off a distance along the ray, not a lateral margin. Found
	// via killeroo-simple.pbrt (a real pbrt-v4 scene: loop-subdivided,
	// coateddiffuse-shaded, lit by one small hard area light) rendering
	// almost solid black on GPU with sparse white flecks - CPU rendered it
	// correctly - while every scene already covered by the test suite
	// (quads/spheres, or flat/architectural triangle meshes where shading
	// and geometric normals nearly coincide) never exercised this path
	// clearly enough to surface it.
	// <= 0 (zero-init default) means "use the standard 0.01f" - see
	// GpuCameraParams::shadowRayEpsilon's own comment for why some scenes
	// need a larger, explicitly-set override.
	const float shadow_eps = (params.camera.shadowRayEpsilon > 0.0f) ? params.camera.shadowRayEpsilon : 0.01f;
	const float3 shadow_origin = origin + shadow_eps * normalize(direction);

	// --stats: null unless --stats was requested - see optix_types.h's
	// LaunchParams::statsShadowRays own comment. This helper is recursive-
	// backend only (wavefront_kernels.cu never calls trace_shadow_ray(),
	// confirmed by grep - it has its own separate shadow-ray path already
	// counted by WavefrontRenderStats), so no double-counting risk from the
	// two backends sharing this header.
	if (params.statsShadowRays) atomicAdd(params.statsShadowRays, 1ull);

	// Trace shadow ray with occlusion testing
	optixTrace(
		params.traversable,           // Acceleration structure
		shadow_origin,                 // Ray origin
		direction,                     // Ray direction
		0.001f,                        // tmin (avoid self-intersection)
		max_distance,                  // tmax
		0.0f,                          // rayTime
		OptixVisibilityMask(255),      // Visibility mask
		OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,  // Flags
		RAY_TYPE_SHADOW,               // SBT offset (shadow ray type)
		RAY_TYPE_COUNT,                // SBT stride (number of ray types)
		RAY_TYPE_SHADOW,               // Miss SBT index
		occluded                       // Payload (single unsigned int)
	);

	// Return true if NOT occluded (path is clear)
	return (occluded == 0);
}

// Real NEE+MIS at a Henyey-Greenstein phase-function scatter event inside a
// participating medium - the volumetric counterpart of a diffuse/glossy
// BSDF's own NEE block, reused by every medium-interior scatter case in this
// module (MaterialType::Medium/CloudMedium/RgbGridMedium/GridMedium, and
// MaterialType::DielectricMedium's own interior sub-case) rather than
// duplicating this ~40-line block once per medium type. Matches CPU's
// hg_phase_material (src/TheRestOfYourLife/constant_medium.h, skip_pdf=false,
// routed through hg_phase_pdf) exactly - see the DielectricMedium branch this
// was originally written for (optix_intersection_sphere.h) for the full
// root-cause derivation of why this matters (B13/SubsurfaceSlab's ~32-38%
// CPU-brighter gap without it): without NEE, the only way a phase-scattered
// ray picks up light is a lucky HG-sampled random walk eventually escaping
// the medium AND hitting a light before the path's depth budget runs out -
// unbiased in the limit, but nowhere near converged at any real sample count
// for a dense/room-filling medium.
//
// `medium_point`: the world-space scatter location (NOT a surface point - no
// meaningful geometric normal exists here, so callers must not offset it
// along one; trace_shadow_ray()'s own shadow_eps offset is already applied
// along the ray direction instead, which is safe for an interior point too).
// `wo`: direction back toward where the ray came from (-incoming ray
// direction, matching CPU hg_phase_pdf's own `wo` convention - get this
// backwards and an anisotropic medium's forward/back-scatter bias inverts).
// `g`: the medium's Henyey-Greenstein asymmetry parameter. `attenuation`:
// the medium's own scattering albedo at this event (already resolved by the
// caller - single RGB for Medium/CloudMedium/GridMedium, per-voxel RGB for
// RgbGridMedium). `scattered_dir`: the already phase-function-sampled
// continuation direction (used only to compute the outgoing brdf_pdf_override
// for the NEXT bounce's own MIS, not to sample here). `self_emission`: the
// medium's own directly-visible "rgb Le" (MakeNamedMedium's own emission,
// MaterialData::medium_emission's own comment in optix_types.h) - folded
// into this function's return value rather than added separately at each
// call site, so a call site can never add real NEE without also picking up
// whatever self-emission the material carries (previously these were two
// independent lines at each call site, an easy one to forget for a future
// medium type). Every caller not yet wired for emission (CloudMedium/
// RgbGridMedium/GridMedium/DielectricMedium's own interior sub-case) passes
// mat.medium_emission too - safe because that field is documented to be
// zero for every material kind that doesn't explicitly set it, so this is a
// pure no-op there today and "just works" the moment any of those types
// gains real Le support. Returns the direct lighting contribution PLUS this
// self-emission, to add to this event's emission; also writes
// brdf_pdf_override (the phase value at the sampled continuation direction),
// which every caller must pass through to shade_material()-style payload
// packing instead of the surface cosine_pdf default.
__device__ __forceinline__ float3 medium_phase_nee_mis(
	const float3& medium_point, const float3& wo, float g,
	const float3& attenuation, const float3& scattered_dir,
	unsigned int& seed, float& brdf_pdf_override, const float3& self_emission,
	float ray_time)
{
	float3 medium_emission = self_emission;
	if (params.numLights > 0) {
		int light_idx;
		float selection_pdf;
		// Same light-BVH-first, alias-table-fallback selection as
		// sample_nee_light()'s own identical block - see that function's
		// own comment on both the ordering and the n=0 simplification.
		// Consistency with sample_nee_light() here matters beyond style: a
		// later BSDF-sampled bounce that hits a light MIS-weights against
		// gpu_light_bvh_pmf() unconditionally now (see the 6 closest-hit
		// files' own DiffuseLight branches) - if THIS selection stayed
		// alias-table-only while surface NEE switched to the light BVH, a
		// path through a medium scatter would MIS-weight against the wrong
		// selection strategy's pdf, a real bias, not just an inconsistency.
		bool have_light = true;
		if (params.lightBvhNodeCount > 0) {
			GpuLightBvhSample s = gpu_light_bvh_sample_index(
				medium_point.x, medium_point.y, medium_point.z, 0.f, 0.f, 0.f,
				random_float(seed));
			have_light = (s.lightIndex >= 0);
			light_idx = s.lightIndex;
			selection_pdf = s.pmf;
		} else if (params.aliasTable) {
			int slot = int(random_float(seed) * float(params.numLights));
			if (slot >= int(params.numLights)) slot = int(params.numLights) - 1;
			const GpuAliasEntry& entry = params.aliasTable[slot];
			light_idx = (random_float(seed) < entry.q) ? slot : entry.alias;
			selection_pdf = params.aliasTable[light_idx].pdf;
		} else {
			light_idx = int(random_float(seed) * float(params.numLights));
			if (light_idx >= int(params.numLights)) light_idx = int(params.numLights) - 1;
			selection_pdf = 1.0f / float(params.numLights);
		}
		if (have_light) {
			float geom_pdf = 0.0f, max_dist = 0.0f;
			float3 sampled_light_emission = make_float3(0.0f, 0.0f, 0.0f);
			float3 to_light = sample_area_light_by_kind(
				light_idx, medium_point, seed, geom_pdf, max_dist, sampled_light_emission, ray_time);
			float light_pdf = selection_pdf * geom_pdf;
			if (light_pdf > 1e-6f) {
				float phase_val = hg_phase_value(dot(wo, to_light), g);
				if (trace_shadow_ray(medium_point, to_light, max_dist)) {
					float mis_weight = mis_power_heuristic(light_pdf, phase_val);
					medium_emission = medium_emission +
						(mis_weight * phase_val / light_pdf) * attenuation * sampled_light_emission;
				}
			}
		}
	}
	{
		const float3& skyColor = params.camera.backgroundColor;
		if (skyColor.x > 0.0f || skyColor.y > 0.0f || skyColor.z > 0.0f) {
			float3 sky_dir, sky_Le_val; float pdf_sky;
			sample_sky_nee(seed, skyColor, medium_point, sky_dir, pdf_sky, sky_Le_val);
			if (pdf_sky > 0.0f) {
				float phase_val_sky = hg_phase_value(dot(wo, sky_dir), g);
				if (trace_shadow_ray(medium_point, sky_dir, 1e30f)) {
					float mis_weight = mis_power_heuristic(pdf_sky, phase_val_sky);
					medium_emission = medium_emission +
						(mis_weight * phase_val_sky / pdf_sky) * attenuation * sky_Le_val;
				}
			}
		}
	}
	brdf_pdf_override = hg_phase_value(dot(wo, scattered_dir), g);
	return medium_emission;
}

// pbrt-v4's own "camera medium" (unbounded ambient fog the camera itself
// starts inside) - GPU counterpart of CPU's ambient_medium::sample_scatter()/
// transmittance_over() (src/TheRestOfYourLife/constant_medium.h) exactly:
// the same single-inversion Beer-Lambert free-path sample against a scalar
// sigma_t (a homogeneous medium's extinction has no spatial variation, so
// this needs no delta-tracking/majorant rejection loop the way CloudMedium/
// GridMedium's real heterogeneous sampling does). `surface_t` is the
// ALREADY-KNOWN distance to the nearest real hit (or a huge sentinel for an
// escaped ray) - see optix_raygen.h's own call site for why this runs AFTER
// the primary optixTrace rather than as one more scene-BVH entry (the real
// reason, per ambient_medium's own comment: a plain hittable has no side-
// channel for a non-winning child to still attenuate the path's
// throughput - not a traversal-order problem).
//
// Returns true on a scatter event within [0, surface_t]: writes
// `out_scatter_point`/`out_scatter_dir` (phase-function-sampled)/
// `out_brdf_pdf_override` (for the NEXT bounce's own MIS)/`out_emission`
// (NEE + self-emission at the event, via medium_phase_nee_mis - reused, not
// reimplemented) - `out_transmittance` is left untouched. Returns false
// otherwise (including "no camera medium" and "no real hit AND zero
// extinction" - both degenerate to a no-op multiply): leaves the scatter
// outputs untouched and writes `out_transmittance` (a single scalar, not
// float3 - see GpuCameraParams::cameraMediumSigmaT's own comment for why a
// scalar sigma_t is correct here, matching CPU exactly) to multiply into
// throughput. sigma_t>0 is already guaranteed on every path that reaches the
// expf() call, so `expf(-sigma_t * surface_t)` never hits CPU's documented
// exp(-0*infinity)=NaN edge case (guarded by the early return just below).
// `ray_time` is passed through to medium_phase_nee_mis()'s own NEE sampling
// (sample_area_light_by_kind()/sample_sphere_light(), for a moving sphere
// light's interpolated position) rather than read via optixGetRayTime()
// anywhere in this call chain - this function is called from optix_raygen.h's
// RAYGEN loop, where optixGetRayTime() is illegal (no "current ray" context
// exists there the way it does inside a closest-hit program) - confirmed by
// a real OptiX module-compile failure ("Illegal call to optixGetRayTime in
// function __raygen__rg") the first time this call chain read it implicitly.
// Beer-Lambert transmittance of the camera medium (GpuCameraParams::
// cameraMediumSigmaT) over a shadow ray's own real distance to its target -
// closes the gap this feature's own scope comment used to document ("no
// shadow-ray/NEE attenuation through it yet ... a light behind the fog
// isn't dimmed by it on the way to a shadow-ray target"), matching CPU's
// identical fix (camera.h's own camera_medium_trans lambda, ray_color()).
// The camera medium is untraced geometry (see sample_camera_medium()'s own
// comment for why it isn't a hittable/BVH entry), so trace_shadow_ray()'s
// optixTrace() call never sees it - every NEE call site in
// optix_device_helpers.h has to apply this separately, once per light
// strategy, exactly mirroring each one's own already-computed shadow-ray
// distance argument.
//
// `max_distance` mirrors the SAME value already passed as trace_shadow_ray()'s
// own third argument at every call site, including the `1e30f` sentinel
// every sky-NEE block uses for "infinite distance" - treated here as an
// exact 0 transmittance for any positive sigma_t (nothing is visible
// arbitrarily far through an unbounded absorbing/scattering medium),
// matching CPU's transmittance_over(infinity) exactly, rather than letting
// expf(-sigma_t * 1e30f) do it implicitly (correct in IEEE754 for any real
// sigma_t here, but explicit is clearer and matches CPU's own dedicated
// infinity branch).
__device__ __forceinline__ float camera_medium_shadow_trans(float max_distance) {
	const float sigma_t = params.camera.cameraMediumSigmaT;
	if (sigma_t <= 0.0f) return 1.0f;
	if (max_distance >= 1e29f) return 0.0f;
	return expf(-sigma_t * max_distance);
}

__device__ __forceinline__ bool sample_camera_medium(
	const float3& ray_orig, const float3& ray_dir, float surface_t, unsigned int& seed,
	float3& out_scatter_point, float3& out_scatter_dir, float& out_brdf_pdf_override,
	float3& out_emission, float& out_transmittance, float ray_time)
{
	const float sigma_t = params.camera.cameraMediumSigmaT;
	if (sigma_t <= 0.0f || surface_t <= 0.0f) { out_transmittance = 1.0f; return false; }

	const float free_path = -logf(fmaxf(1e-8f, 1.0f - random_float(seed))) / sigma_t;
	if (free_path >= surface_t) {
		out_transmittance = expf(-sigma_t * surface_t);
		return false;
	}

	out_scatter_point = ray_orig + free_path * ray_dir;
	const float3 wo = -ray_dir;
	out_scatter_dir = sample_henyey_greenstein(wo, params.camera.cameraMediumG, seed);
	out_emission = medium_phase_nee_mis(out_scatter_point, wo, params.camera.cameraMediumG,
		params.camera.cameraMediumAlbedo, out_scatter_dir, seed, out_brdf_pdf_override,
		params.camera.cameraMediumEmission, ray_time);
	return true;
}

// Result of one RAY_TYPE_PROBE trace - see trace_probe_ray() below and
// optix_probe_hit.h's own payload-layout comment.
struct ProbeHit {
	bool   found;
	float3 position;
	float3 normal;
	int    materialIdx;
};

// Trace a single closest-hit probe ray (RAY_TYPE_PROBE), used by
// bssrdf_probe_walk() below to step along a BSSRDF probe segment. Modeled
// directly on trace_shadow_ray() above - the same already-proven pattern of
// a sequential, non-nested optixTrace() call issued from within a hit
// program - but targets the dedicated probe hit groups (optix_probe_hit.h)
// instead of the shadow ones, and asks for the actual closest hit (no
// OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT/DISABLE_CLOSESTHIT) since the probe
// walk needs real hit geometry, not just an occlusion bit.
//
// No self-intersection epsilon nudge along the ray direction (unlike
// trace_shadow_ray()): the caller (bssrdf_probe_walk()) already advances
// `base` past each hit by t+1e-4 in world units before the next call, which
// serves the same purpose for a ray that is walked in fixed steps rather
// than re-fired from a fresh surface point each time.
__device__ __forceinline__ ProbeHit trace_probe_ray(
	const float3& origin,
	const float3& direction,
	float max_distance
) {
	// Sentinel "no hit" state - __miss__probe() is a true no-op, so these
	// values pass through unchanged on a miss (see optix_probe_hit.h).
	unsigned int p0 = (unsigned int)(-1);
	unsigned int p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0;

	optixTrace(
		params.traversable,
		origin,
		direction,
		1e-5f,                     // tmin
		max_distance,              // tmax
		0.0f,                      // rayTime
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_NONE,       // real closest-hit, not occlusion-only
		RAY_TYPE_PROBE,            // SBT offset
		RAY_TYPE_COUNT,            // SBT stride
		RAY_TYPE_PROBE,            // miss SBT index
		p0, p1, p2, p3, p4, p5, p6
	);

	ProbeHit hit;
	hit.materialIdx = (int)p0;
	hit.found = (hit.materialIdx >= 0);
	hit.position = make_float3(__uint_as_float(p1), __uint_as_float(p2), __uint_as_float(p3));
	hit.normal   = make_float3(__uint_as_float(p4), __uint_as_float(p5), __uint_as_float(p6));
	return hit;
}

// Evaluate one punctual (point/spot/distant) light at shading point p:
// direction toward the light (wi), incident radiance (Li), and shadow-ray
// max distance (t_max). Returns false if the light contributes nothing here
// (e.g. outside a spot's cone) so the caller can skip the shadow ray.
// Equal-area sphere->square mapping, used by the goniometric light's image
// lookup. Direct copy of src/shared/sampling_extra.h's EqualAreaSphereToSquare
// (CPU_GPU-tagged there too, but that header pulls in ~1400 lines of mostly
// CPU-only sampling code not needed here - duplicated locally instead,
// matching this file's existing pattern of small self-contained device
// helpers rather than pulling in unrelated shared headers).
__device__ __forceinline__ void dev_equal_area_sphere_to_square(
	double wx, double wy, double wz, double& u, double& v
) {
	double x = fabs(wx), y = fabs(wy), z = fabs(wz);
	double r = sqrt(fmax(0.0, 1.0 - z));
	double a = (x > y) ? x : y;
	double b = (x > y) ? y : x;
	b = (a == 0.0) ? 0.0 : b / a;

	const double t1 =  0.406758566246788489601959989e-5;
	const double t2 =  0.636226545274016134946890922156;
	const double t3 =  0.61572017898280213493197203466e-2;
	const double t4 = -0.247333733281268944196501420480;
	const double t5 =  0.881770664775316294736387951347e-1;
	const double t6 =  0.419038818029165735901852432784e-1;
	const double t7 = -0.251390972343483509333252996350e-1;
	double phi = t1 + b*(t2 + b*(t3 + b*(t4 + b*(t5 + b*(t6 + b*t7)))));
	if (x < y) phi = 1.0 - phi;

	double vv = phi * r;
	double uu = r - vv;
	if (wz < 0.0) { double tmp = uu; uu = 1.0 - vv; vv = 1.0 - tmp; }
	uu = copysign(uu, wx);
	vv = copysign(vv, wy);
	u = 0.5*(uu + 1.0);
	v = 0.5*(vv + 1.0);
}

// Forward direction of the mapping above: (u,v) in [0,1]^2 -> unit sphere
// direction, equal-area. Direct copy of src/shared/sampling_sphere.h's
// EqualAreaSquareToSphere (same "small self-contained device helper" reason
// as dev_equal_area_sphere_to_square just above - avoids pulling in that
// header's ~1400 lines of mostly CPU-only sampling code).
__device__ __forceinline__ void dev_equal_area_square_to_sphere(
	double u, double v, double& wx, double& wy, double& wz
) {
	double uu = 2.0*u - 1.0, vv = 2.0*v - 1.0;
	double up = fabs(uu), vp = fabs(vv);
	double signed_dist = 1.0 - (up + vp);
	double d = fabs(signed_dist);
	double r = 1.0 - d;
	double phi = (r == 0.0 ? 1.0 : (vp - up) / r + 1.0) * (3.14159265358979323846 / 4.0);
	wz = copysign(1.0 - r*r, signed_dist);
	double cos_phi = cos(phi);
	double sin_phi = sin(phi);
	double xy_r = r * sqrt(fmax(0.0, 2.0 - r*r));
	wx = copysign(cos_phi * xy_r, uu);
	wy = copysign(sin_phi * xy_r, vv);
}

// Mirror (u,v) outside [0,1]^2 back onto the equal-area square. Direct copy
// of src/shared/sampling_sphere.h's WrapEqualAreaSquare.
__device__ __forceinline__ void dev_wrap_equal_area_square(double& u, double& v) {
	if (u < 0.0) { u = -u; v = 1.0 - v; }
	else if (u > 1.0) { u = 2.0 - u; v = 1.0 - v; }
	if (v < 0.0) { u = 1.0 - u; v = -v; }
	else if (v > 1.0) { u = 1.0 - u; v = 2.0 - v; }
}

// Mirrors src/TheRestOfYourLife/punctual_light_objects.h's PunctualLiSample /
// sample_direct() on the CPU - pdf is always 1 for these delta lights, so
// callers add the contribution directly with no MIS weight or pdf division
// (see camera.h's punct_lights NEE block for the reference formula).
__device__ __forceinline__ bool eval_punctual_light(
	const PunctualLightGPU& light,
	const float3& p,
	float3& wi,
	float3& Li,
	float& t_max
) {
	float Lr = 0.0f, Lg = 0.0f, Lb = 0.0f, wx = 0.0f, wy = 0.0f, wz = 0.0f;
	switch (light.kind) {
		case PunctualLightKind::Point: {
			light.point.sample_wi(p.x, p.y, p.z, wx, wy, wz);
			light.point.eval_Li(p.x, p.y, p.z, Lr, Lg, Lb);
			float dx = light.point.pos_x - p.x, dy = light.point.pos_y - p.y, dz = light.point.pos_z - p.z;
			t_max = sqrtf(dx * dx + dy * dy + dz * dz);
			break;
		}
		case PunctualLightKind::Spot: {
			light.spot.sample_wi(p.x, p.y, p.z, wx, wy, wz);
			light.spot.eval_Li(p.x, p.y, p.z, Lr, Lg, Lb);
			float dx = light.spot.pos_x - p.x, dy = light.spot.pos_y - p.y, dz = light.spot.pos_z - p.z;
			t_max = sqrtf(dx * dx + dy * dy + dz * dz);
			break;
		}
		case PunctualLightKind::Distant: {
			light.distant.sample_wi(wx, wy, wz);
			light.distant.eval_Li(Lr, Lg, Lb);
			t_max = 1e30f;  // no finite geometric distance for a directional light
			break;
		}
		case PunctualLightKind::Goniometric: {
			const GoniometricLightGPU& g = light.gonio;
			float dx = g.pos_x - p.x, dy = g.pos_y - p.y, dz = g.pos_z - p.z;
			float r2 = dx*dx + dy*dy + dz*dz;
			if (r2 < 1e-20f) return false;
			t_max = sqrtf(r2);
			float inv_r = 1.0f / t_max;
			wx = dx * inv_r; wy = dy * inv_r; wz = dz * inv_r;
			// Direction from light toward shading point, rotated into light
			// space (mirrors GoniometricLight::eval_I(-wi) called from sample_li).
			float lx = g.world_to_light[0]*(-wx) + g.world_to_light[1]*(-wy) + g.world_to_light[2]*(-wz);
			float ly = g.world_to_light[3]*(-wx) + g.world_to_light[4]*(-wy) + g.world_to_light[5]*(-wz);
			float lz = g.world_to_light[6]*(-wx) + g.world_to_light[7]*(-wy) + g.world_to_light[8]*(-wz);
			double u, v;
			dev_equal_area_sphere_to_square((double)lx, (double)ly, (double)lz, u, v);
			int iu = (int)(u * g.nu); iu = iu < 0 ? 0 : (iu >= g.nu ? g.nu - 1 : iu);
			int iv = (int)(v * g.nv); iv = iv < 0 ? 0 : (iv >= g.nv ? g.nv - 1 : iv);
			float gonio = g.image[iv * g.nu + iu];
			float weight = g.scale * gonio / r2;
			Lr = g.ir * weight; Lg = g.ig * weight; Lb = g.ib * weight;
			break;
		}
		case PunctualLightKind::Projection: {
			const ProjectionLightGPU& pr = light.proj;
			float dx = pr.pos_x - p.x, dy = pr.pos_y - p.y, dz = pr.pos_z - p.z;
			float r2 = dx*dx + dy*dy + dz*dz;
			if (r2 < 1e-20f) return false;
			t_max = sqrtf(r2);
			float inv_r = 1.0f / t_max;
			wx = dx * inv_r; wy = dy * inv_r; wz = dz * inv_r;
			// Direction from light toward shading point, rotated into light space.
			float lx = pr.world_to_light[0]*(-wx) + pr.world_to_light[1]*(-wy) + pr.world_to_light[2]*(-wz);
			float ly = pr.world_to_light[3]*(-wx) + pr.world_to_light[4]*(-wy) + pr.world_to_light[5]*(-wz);
			float lz = pr.world_to_light[6]*(-wx) + pr.world_to_light[7]*(-wy) + pr.world_to_light[8]*(-wz);
			if (lz < pr.hither) return false;
			// screenFromLight reduces to this for make_perspective()'s matrix
			// shape (see ProjectionLightGPU's comment in optix_types.h).
			float sx = pr.inv_tan * lx / lz;
			float sy = pr.inv_tan * ly / lz;
			if (sx < pr.sb_xmin || sx > pr.sb_xmax || sy < pr.sb_ymin || sy > pr.sb_ymax) return false;
			float u = (sx - pr.sb_xmin) / (pr.sb_xmax - pr.sb_xmin);
			float v = (sy - pr.sb_ymin) / (pr.sb_ymax - pr.sb_ymin);
			int iu = (int)(u * pr.nx); iu = iu < 0 ? 0 : (iu >= pr.nx ? pr.nx - 1 : iu);
			int iv = (int)(v * pr.ny); iv = iv < 0 ? 0 : (iv >= pr.ny ? pr.ny - 1 : iv);
			int idx = (iv * pr.nx + iu) * 3;
			float rC = fmaxf(0.0f, pr.image_rgb[idx + 0]);
			float gC = fmaxf(0.0f, pr.image_rgb[idx + 1]);
			float bC = fmaxf(0.0f, pr.image_rgb[idx + 2]);
			float inv_r2 = 1.0f / r2;
			Lr = pr.scale * rC * inv_r2; Lg = pr.scale * gC * inv_r2; Lb = pr.scale * bC * inv_r2;
			break;
		}
		default:
			return false;
	}
	if (Lr <= 0.0f && Lg <= 0.0f && Lb <= 0.0f) return false;
	wi = make_float3(wx, wy, wz);
	Li = make_float3(Lr, Lg, Lb);
	return true;
}

// Add every punctual light's direct contribution at a Lambertian hit to
// `emission` (accumulated in-place). `normal` must be the shading normal
// (front-facing). Call from the same place area-light NEE happens.
__device__ __forceinline__ void add_punctual_lights_lambertian(
	const float3& hit_point,
	const float3& normal,
	const float3& albedo,
	float3& emission
) {
	for (unsigned int i = 0; i < params.numPunctualLights; ++i) {
		float3 wi, Li; float t_max;
		if (!eval_punctual_light(params.punctualLights[i], hit_point, wi, Li, t_max)) continue;
		float cos_theta = dot(wi, normal);
		if (cos_theta <= 0.0f) continue;
		if (trace_shadow_ray(hit_point, wi, t_max)) {
			float3 brdf = albedo / 3.14159265358979323846f;
			emission = emission + brdf * Li * cos_theta * camera_medium_shadow_trans(t_max);
		}
	}
}

// De Casteljau cubic Bezier evaluation at t in [0,1] over 4 RGB control
// points - matches marble_texture::cubic_bezier4 (texture.h) exactly.
__device__ __forceinline__ float3 marble_cubic_bezier4(
		const float3& p0, const float3& p1, const float3& p2, const float3& p3, float t) {
	const float s = 1.0f - t;
	const float w0 = s*s*s, w1 = 3.0f*s*s*t, w2 = 3.0f*s*t*t, w3 = t*t*t;
	return make_float3(
		w0*p0.x + w1*p1.x + w2*p2.x + w3*p3.x,
		w0*p0.y + w1*p1.y + w2*p2.y + w3*p3.y,
		w0*p0.z + w1*p1.z + w2*p2.z + w3*p3.z);
}

// Matches windy_texture::value() (texture.h) exactly: two FBm calls (a
// coarse "wind strength", a finer "wave height") combined multiplicatively,
// remapped to greyscale [0,1] the same way fbm_texture is.
__device__ __forceinline__ float3 sample_windy_texture(const float3& p) {
	const float windStrength = fbm_simple<float>(0.1f*p.x, 0.1f*p.y, 0.1f*p.z, 0.5f, 3);
	const float waveHeight   = fbm_simple<float>(p.x, p.y, p.z, 0.5f, 6);
	const float v = fabsf(windStrength) * waveHeight;
	float t = 0.5f + 0.5f * v;
	t = fminf(fmaxf(t, 0.0f), 1.0f);
	return make_float3(t, t, t);
}

// Matches wrinkled_texture::value() (texture.h) exactly: raw Turbulence
// (already non-negative), plain-clamped to [0,1] - no sign-remap needed.
__device__ __forceinline__ float3 sample_wrinkled_texture(const TextureData& tex, const float3& p) {
	const float v = turbulence_simple<float>(p.x, p.y, p.z, tex.omega, tex.octaves);
	const float t = fminf(fmaxf(v, 0.0f), 1.0f);
	return make_float3(t, t, t);
}

// Matches dots_texture::is_inside_dot() (texture.h) exactly - a per-UV-cell
// polka-dot presence/jitter test via perlin_noise at a fixed z=0.5 (pbrt-v4's
// own Noise(x,y) 2-arg overload). Hard binary edge, no antialiasing, matching
// real pbrt-v4's InsidePolkaDot() exactly.
__device__ __forceinline__ bool is_inside_dot(float s, float t) {
	const float sCell = floorf(s + 0.5f);
	const float tCell = floorf(t + 0.5f);
	if (perlin_noise<float>(sCell + 0.5f, tCell + 0.5f, 0.5f) <= 0.0f) return false;
	constexpr float radius = 0.35f;
	constexpr float maxShift = 0.5f - radius;
	const float sCenter = sCell + maxShift * perlin_noise<float>(sCell + 1.5f, tCell + 2.8f, 0.5f);
	const float tCenter = tCell + maxShift * perlin_noise<float>(sCell + 4.5f, tCell + 9.8f, 0.5f);
	const float ds = s - sCenter, dt = t - tCenter;
	return ds*ds + dt*dt < radius*radius;
}

// Matches bilerp_texture::value() (texture.h) exactly: plain bilinear blend
// of 4 corner colours by (u,v). color1/color2 carry v00/v01; the other two
// corners are packed into uScale/vScale/omega and marbleScale/
// marbleVariation/mixAmount (see TextureKind::Bilerp's own comment,
// optix_types.h).
__device__ __forceinline__ float3 sample_bilerp_texture(const TextureData& tex, float u, float v) {
	const float3& v00 = tex.color1; const float3& v01 = tex.color2;
	const float3 v10 = make_float3(tex.uScale, tex.vScale, tex.omega);
	const float3 v11 = make_float3(tex.marbleScale, tex.marbleVariation, tex.mixAmount);
	const float a = (1.0f-u)*(1.0f-v), b = u*(1.0f-v), c = (1.0f-u)*v, d = u*v;
	return make_float3(
		a*v00.x + b*v10.x + c*v01.x + d*v11.x,
		a*v00.y + b*v10.y + c*v01.y + d*v11.y,
		a*v00.z + b*v10.z + c*v01.z + d*v11.z);
}

// Matches marble_texture::value() (texture.h) exactly: FBm-perturbed sine
// wave mapped through the same 9-knot pbrt-v4 marble colour spline. A
// standalone function (not inlined into sample_texture's own if/else chain)
// since resolve_bssrdf_exit-style callers never need it and the 9-knot
// table is sizeable to duplicate inline twice (recursive backend here,
// wavefront's own copy in wavefront_kernels.cu per this file's established
// no-shared-device-helpers-across-backends convention).
__device__ __forceinline__ float3 sample_marble_texture(const TextureData& tex, const float3& p) {
	const float px = p.x * tex.marbleScale, py = p.y * tex.marbleScale, pz = p.z * tex.marbleScale;
	const float fbm_val = fbm_simple<float>(px, py, pz, tex.omega, tex.octaves);
	const float marble = py + tex.marbleVariation * fbm_val;
	float t = 0.5f + 0.5f * sinf(marble);

	constexpr int kN = 9;
	const float3 knots[kN] = {
		make_float3(.58f,.58f,.60f), make_float3(.58f,.58f,.60f), make_float3(.58f,.58f,.60f),
		make_float3(.50f,.50f,.50f), make_float3(.60f,.59f,.58f), make_float3(.58f,.58f,.60f),
		make_float3(.58f,.58f,.60f), make_float3(.20f,.20f,.33f), make_float3(.58f,.58f,.60f)
	};
	constexpr int nSeg = kN - 3;
	int first = static_cast<int>(t * nSeg);
	if (first >= nSeg) first = nSeg - 1;
	const float lt = t * nSeg - first;

	float3 rgb = marble_cubic_bezier4(knots[first], knots[first+1], knots[first+2], knots[first+3], lt);
	return make_float3(fminf(rgb.x * 1.5f, 1.0f), fminf(rgb.y * 1.5f, 1.0f), fminf(rgb.z * 1.5f, 1.0f));
}

