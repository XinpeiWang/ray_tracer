// optix_anyhit_shadow.h -- Shadow any-hit programs
// Included by optix_programs.cu

extern "C" __global__ void __anyhit__shadow_sphere() {
	// Get primitive and material
	const unsigned int primIdx = optixGetPrimitiveIndex();
	// See optix_intersection_sphere.h: a primitive index is local to its GAS,
	// and the table entry is SIGNED (-1 = not instanced) - casting -1 straight
	// to unsigned would wrap to UINT_MAX instead of falling back to base 0.
	const int instBase = params.instancePrimBase
		? params.instancePrimBase[optixGetInstanceId()] : -1;
	const unsigned int sphBase = (instBase >= 0) ? (unsigned int)instBase : 0u;
	const SphereData& sphere = params.spheres[sphBase + primIdx];
	// Resolved to a real, non-Mix material before any mat.type check below -
	// see MaterialType::Mix's own comment (optix_types.h). Matches CPU's
	// mix_material::is_shadow_transmissive(), which delegates to the same
	// deterministic (hashed-hit-point) sub-material pick as scatter().
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = sphere.materialIdx;
	const MaterialData mat = resolve_mix_material(params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	// IMPORTANT: When hitting light source, set NOT occluded and terminate
	// This allows the shadow ray to "see" the light
	if (mat.type == MaterialType::DiffuseLight) {
		optixSetPayload_0(0);  // NOT occluded - light is visible
		optixTerminateRay();
		return;
	}

	// Homogeneous Medium/DielectricMedium: real Beer-Lambert attenuation of
	// the running shadow-ray transmittance (payload 1), NOT unconditional
	// pass-through - matches CPU's hg_phase_material::shadow_transmittance_
	// impl (constant_medium.h) and GPU-wavefront's identical branch
	// (__anyhit__wf_shadow_sphere, wavefront_anyhit_shadow.h) exactly, using
	// the same analytic near/far chord math. This replaced an earlier
	// unconditional optixIgnoreIntersection() (below, still used for
	// CloudMedium/RgbGridMedium/GridMedium, and for the dielectric-family
	// surface types) that treated these two types as perfectly, losslessly
	// transparent to every shadow ray - confirmed as the actual cause of a
	// real CPU/GPU-recursive/GPU-wavefront 3-way brightness mismatch on B13
	// (Subsurface Slab, MaterialCpuGpuParityTest.
	// BrightnessAndChannelsConsistentAcrossBackends/Scene21_Subsurface_Slab):
	// CPU and GPU-wavefront both already dim a shadow ray by the medium's
	// own extinction; GPU-recursive alone let 100% of a light's contribution
	// straight through the wax slab/jade sphere's own fill, every time
	// medium_phase_nee_mis()'s interior phase-scatter NEE (optix_device_
	// helpers_lighting.h) sampled a shadow ray back out through this same
	// boundary toward a light - measured ~36-42% too bright vs. both other
	// backends (which agreed with each other) before this fix.
	if (mat.type == MaterialType::Medium || mat.type == MaterialType::DielectricMedium) {
		const float3 ray_orig = optixGetWorldRayOrigin();
		const float3 ray_dir = optixGetWorldRayDirection();  // unit length by construction (trace_shadow_ray's own callers)
		const bool is_box = (sphere.shapeKind == GpuMediumShapeKind::Box);
		float t_near, t_far;
		if (is_box) {
			float bn, bf;
			box_slab_intersect(ray_orig, ray_dir, sphere.boxMin, sphere.boxMax, bn, bf);
			t_near = fmaxf(0.0f, bn);
			t_far = bf;
		} else {
			// Object-space center/radius via the intersection program's own
			// reported attributes (time-interpolated for a moving sphere) -
			// see __closesthit__sphere's identical read just above, in this
			// same translation unit's own optix_intersection_sphere.h.
			const float3 sphere_center = make_float3(
				__int_as_float(optixGetAttribute_0()),
				__int_as_float(optixGetAttribute_1()),
				__int_as_float(optixGetAttribute_2()));
			const float sphere_radius = __int_as_float(optixGetAttribute_3());
			const float3 oc = ray_orig - sphere_center;
			const float half_b = dot(oc, ray_dir);
			const float c = dot(oc, oc) - sphere_radius * sphere_radius;
			const float disc = fmaxf(0.0f, half_b * half_b - c);
			const float sq = sqrtf(disc);
			t_near = fmaxf(0.0f, -half_b - sq);
			t_far = -half_b + sq;
		}
		const float sigma_t = (mat.type == MaterialType::Medium)
			? mat.sigma_t : mat.dielectric_medium_extra.sigma_t;
		// shadow_t (this function's own optixGetRayTmax(), read above) is the
		// shadow ray's ORIGINAL max_distance (the light's own distance) for
		// every candidate along an all-ignoring traversal like this one - the
		// same quantity wavefront's WfShadowPayload::tMax caches explicitly.
		const float segFar = fminf(t_far, shadow_t);
		const float segLen = fmaxf(0.0f, segFar - fmaxf(0.0f, t_near));
		float transmittance = __uint_as_float(optixGetPayload_1());
		transmittance *= expf(-sigma_t * segLen);
		optixSetPayload_1(__float_as_uint(transmittance));
		if (transmittance <= 0.0f) {
			optixSetPayload_0(1);  // fully attenuated - treat as occluded
			optixTerminateRay();
			return;
		}
		optixIgnoreIntersection();  // continue traversal (attenuated, not occluding)
		return;
	}

	// Transmissive materials let light through -- ignore them in shadow rays.
	// CloudMedium's/RgbGridMedium's/GridMedium's trigger spheres get the
	// same unconditional pass-through as the dielectric-family surface types
	// below: their heterogeneous ratio-tracking (wavefront_anyhit_shadow.h's
	// own CloudMedium/RgbGridMedium/GridMedium branches) hasn't been ported
	// to this shadow-ray path yet, so - like Medium/DielectricMedium used to
	// be, just above - they're treated as non-occluding (light passes
	// straight through) rather than wrongly blocking NEE entirely. Without
	// this, every shadow ray toward a light on the far side of one of these
	// bounding spheres would be wrongly treated as fully occluded.
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission ||
		mat.type == MaterialType::CloudMedium ||
		mat.type == MaterialType::RgbGridMedium ||
		mat.type == MaterialType::GridMedium ||
		mat.type == MaterialType::Interface) {
		optixIgnoreIntersection();  // continue traversal (not an occluder)
		return;
	}

	// For opaque materials, treat as occluder
	optixSetPayload_0(1);  // occluded = true
	optixTerminateRay();   // Stop traversal (found occlusion)
}

// Shadow any-hit for quads
// For opaque geometry, any hit means occlusion - terminate immediately
extern "C" __global__ void __anyhit__shadow_quad() {
	// Get primitive and material
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const QuadData& quad = params.quads[primIdx];
	// See __anyhit__shadow_sphere's own comment for why Mix must resolve here.
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = quad.materialIdx;
	const MaterialData mat = resolve_mix_material(params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	// IMPORTANT: When hitting a light source, set NOT occluded and terminate
	// This allows the shadow ray to "see" the light
	if (mat.type == MaterialType::DiffuseLight) {
		optixSetPayload_0(0);  // NOT occluded - light is visible
		optixTerminateRay();
		return;
	}

	// Transmissive materials let light through -- ignore them in shadow rays
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission ||
		mat.type == MaterialType::Interface) {
		optixIgnoreIntersection();  // continue traversal (not an occluder)
		return;
	}

	// For opaque materials, treat as occluder
	optixSetPayload_0(1);  // occluded = true
	optixTerminateRay();   // Stop traversal (found occlusion)
}

// Shadow any-hit for bilinear patches
extern "C" __global__ void __anyhit__shadow_bilinear_patch() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const BilinearPatchData& patch = params.bilinearPatches[primIdx];
	// See __anyhit__shadow_sphere's own comment for why Mix must resolve here.
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = patch.materialIdx;
	const MaterialData mat = resolve_mix_material(params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	if (mat.type == MaterialType::DiffuseLight) {
		optixSetPayload_0(0);
		optixTerminateRay();
		return;
	}

	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission ||
		mat.type == MaterialType::Interface) {
		optixIgnoreIntersection();
		return;
	}

	optixSetPayload_0(1);  // occluded = true
	optixTerminateRay();
}

// Shadow any-hit for disks. Same simple (DiffuseLight not-occluding,
// dielectric-family ignored, everything else opaque) list as quad/bilinear
// patch above - the pbrt loader never assigns a Medium/DielectricMedium
// material to a disk (see pbrt_gpu_builder.h's disk loop), so no CloudMedium/
// RgbGridMedium/Medium/DielectricMedium entries are needed here.
extern "C" __global__ void __anyhit__shadow_disk() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const DiskData& disk = params.disks[primIdx];
	// See __anyhit__shadow_sphere's own comment for why Mix must resolve here.
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = disk.materialIdx;
	const MaterialData mat = resolve_mix_material(params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	if (mat.type == MaterialType::DiffuseLight) {
		optixSetPayload_0(0);
		optixTerminateRay();
		return;
	}

	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission ||
		mat.type == MaterialType::Interface) {
		optixIgnoreIntersection();
		return;
	}

	optixSetPayload_0(1);  // occluded = true
	optixTerminateRay();
}

// Shadow any-hit for cylinders - same list as disk above.
extern "C" __global__ void __anyhit__shadow_cylinder() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const CylinderData& cyl = params.cylinders[primIdx];
	// See __anyhit__shadow_sphere's own comment for why Mix must resolve here.
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = cyl.materialIdx;
	const MaterialData mat = resolve_mix_material(params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	if (mat.type == MaterialType::DiffuseLight) {
		optixSetPayload_0(0);
		optixTerminateRay();
		return;
	}

	// MaterialType::Medium belongs here for the same reason as
	// __anyhit__shadow_sphere's own Medium entry: full Beer-Lambert shadow-
	// ray transmittance through a fog cylinder isn't implemented, so it's
	// treated as non-occluding (light passes straight through) rather than
	// wrongly blocking NEE entirely. Real, reachable now that pbrt_gpu_
	// builder.h's cylinder loop resolves MediumInterface via
	// mediumMaterialIndex() - omitting it here was a real bug (a shadow ray
	// toward a light on the far side of a fog cylinder was wrongly reported
	// fully occluded), same bug class __anyhit__shadow_sphere's own comment
	// already documents for that shape.
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission ||
		mat.type == MaterialType::Medium ||
		mat.type == MaterialType::Interface) {
		optixIgnoreIntersection();
		return;
	}

	optixSetPayload_0(1);  // occluded = true
	optixTerminateRay();
}

// Shadow any-hit for triangles
extern "C" __global__ void __anyhit__shadow_triangle() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	// See optix_intersection_triangle.h: a primitive index is local to its GAS,
	// and the table entry is SIGNED (-1 = not instanced) - casting -1 straight
	// to unsigned would wrap to UINT_MAX instead of falling back to base 0.
	const int instBase = params.instancePrimBase
		? params.instancePrimBase[optixGetInstanceId()] : -1;
	const unsigned int triBase = (instBase >= 0) ? (unsigned int)instBase : 0u;
	const TriangleData& tri = params.triangles[triBase + primIdx];
	// See __anyhit__shadow_sphere's own comment for why Mix must resolve here
	// (before even the alpha-cutout check just below - a Mix wrapper's own
	// alphaMaskTexIdx is always -1, so checking it unresolved would silently
	// skip a sub-material's real alpha cutout).
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = tri.materialIdx;
	const MaterialData mat = resolve_mix_material(params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	// Alpha-cutout: a transparent pixel of a leaf/foliage material casts no
	// shadow there, same as any other "not actually here" any-hit case
	// below - checked first since it applies regardless of the material's
	// type otherwise (an alpha-masked material here is always ordinary
	// lambertian, never a light or dielectric, but there's no reason to
	// assume that will always hold). No-op (skips straight to the existing
	// checks) for the overwhelming majority of triangles, whose material
	// has no alpha mask.
	if (mat.alphaMaskTexIdx >= 0) {
		float uv_u = 0.0f, uv_v = 0.0f;
		if (tri.hasUVs) {
			const float2 bary = optixGetTriangleBarycentrics();
			const float b1 = bary.x, b2 = bary.y, b0 = 1.0f - b1 - b2;
			uv_u = b0 * tri.uv0.x + b1 * tri.uv1.x + b2 * tri.uv2.x;
			uv_v = b0 * tri.uv0.y + b1 * tri.uv1.y + b2 * tri.uv2.y;
		}
		if (!passes_alpha_cutout(mat.alphaMaskTexIdx, uv_u, uv_v, shadow_hit_point)) {
			optixIgnoreIntersection();
			return;
		}
	}

	if (mat.type == MaterialType::DiffuseLight) {
		optixSetPayload_0(0);
		optixTerminateRay();
		return;
	}

	// MaterialType::Subsurface belongs in this list: its entry interface IS a
	// plain dielectric surface (see MaterialType::Subsurface's own comment,
	// shade_material() in optix_device_helpers.h - "the exact same smooth
	// DielectricBxDF sample as MaterialType::Dielectric"), and CPU's
	// equivalent `class subsurface::is_shadow_transmissive()`
	// (material_pbrt.h) returns true with a comment explicitly stating it
	// matches this list - Subsurface's own omission here was a real bug
	// (the CPU code was written assuming GPU parity that didn't actually
	// exist): any NEE shadow ray crossing a Subsurface-shaded mesh (skin,
	// wax, marble - Subsurface is triangle-only in this backend, which is
	// why only this triangle any-hit needs it, not the sphere one above)
	// was wrongly reported fully occluded instead of passing through.
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission ||
		mat.type == MaterialType::Subsurface ||
		mat.type == MaterialType::Interface) {
		optixIgnoreIntersection();
		return;
	}

	optixSetPayload_0(1);  // occluded = true
	optixTerminateRay();
}

//==============================================================================
// Miss Program
//==============================================================================

