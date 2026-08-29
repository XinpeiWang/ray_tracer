// wavefront_anyhit_shadow.h -- Shadow-ray any-hit programs, one per geometry type
// Included by wavefront_programs.cu, after wavefront_common.h.

// __anyhit__wf_shadow_*: one per geometry type (sphere/quad/bilinear_patch/
// triangle), mirroring optix_anyhit_shadow.h's __anyhit__shadow_* exactly -
// see that file's own comments for why DiffuseLight and the transmissive
// materials need special handling, not just "any hit = occluded":
//
//   - MaterialType::DiffuseLight is NOT an occluder - a shadow ray sampled
//     toward a light is EXPECTED to end inside/on that light's own surface,
//     so treating the hit as occlusion would make every area light shadow
//     itself. This one previously used a single generic __anyhit__wf_shadow
//     with no material lookup at all (unconditional occluded=true regardless
//     of what was hit), so any shadow ray whose sampled direction actually
//     reached the light within its tMax read as occluded - which for a
//     sphere light is any point with a large-enough cosine to the light
//     (i.e. exactly the well-lit directions NEE most needs), and for a quad
//     light happened to not manifest because the tMax/origin-offset epsilon
//     interaction (see wavefront_kernels.cu's shadow-ray-setup comment)
//     pushed the effective light-plane distance the other way for a
//     camera-facing quad normal. Confirmed via scene B14 (Measured BRDF
//     showroom, sphere light only): floor and sphere-tops facing the light
//     directly went black under --wavefront while the recursive path
//     rendered them correctly lit.
//   - Dielectric/RoughDielectric/ThinDielectric/DiffuseTransmission (and
//     Medium/CloudMedium/RgbGridMedium/GridMedium/DielectricMedium,
//     sphere-only - see optix_anyhit_shadow.h's own comment) let light
//     through rather than blocking NEE outright.
extern "C" __global__ void __anyhit__wf_shadow_sphere() {
	const int instBase = wf_instance_base();
	const SphereData& sph = wf_params.spheres[wf_prim_base(instBase) + optixGetPrimitiveIndex()];
	// Resolved to a real, non-Mix material before any mat.type check below -
	// see MaterialType::Mix's own comment (optix_types.h). Matches CPU's
	// mix_material::is_shadow_transmissive(), which delegates to the same
	// deterministic (hashed-hit-point) sub-material pick as scatter().
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = sph.materialIdx;
	const MaterialData mat = wf_resolve_mix_material(wf_params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	WfShadowPayload* sp = (WfShadowPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	if (mat.type == MaterialType::DiffuseLight) {
		sp->occluded = false;
		optixTerminateRay();
		return;
	}
	// MaterialType::DielectricMedium belongs here for the same reason as
	// Medium/CloudMedium/RgbGridMedium/GridMedium just above - see
	// optix_anyhit_shadow.h's __anyhit__shadow_sphere for the full comment
	// (that omission was a real bug, found while adding real NEE to this
	// material's medium-interior phase-scatter case in wavefront_kernels.cu /
	// optix_intersection_sphere.h). GridMedium itself had the identical bug
	// here too - missing from this list at the same time DielectricMedium
	// was added, caught by a follow-up review pass.
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission ||
		mat.type == MaterialType::Medium ||
		mat.type == MaterialType::CloudMedium ||
		mat.type == MaterialType::RgbGridMedium ||
		mat.type == MaterialType::GridMedium ||
		mat.type == MaterialType::DielectricMedium) {
		optixIgnoreIntersection();
		return;
	}
	sp->occluded = true;
	optixTerminateRay();
}

extern "C" __global__ void __anyhit__wf_shadow_quad() {
	const QuadData& quad = wf_params.quads[optixGetPrimitiveIndex()];
	// See __anyhit__wf_shadow_sphere's own comment for why Mix must resolve here.
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = quad.materialIdx;
	const MaterialData mat = wf_resolve_mix_material(wf_params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	WfShadowPayload* sp = (WfShadowPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	if (mat.type == MaterialType::DiffuseLight) {
		sp->occluded = false;
		optixTerminateRay();
		return;
	}
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission) {
		optixIgnoreIntersection();
		return;
	}
	sp->occluded = true;
	optixTerminateRay();
}

extern "C" __global__ void __anyhit__wf_shadow_bilinear_patch() {
	const BilinearPatchData& patch = wf_params.bilinearPatches[optixGetPrimitiveIndex()];
	// See __anyhit__wf_shadow_sphere's own comment for why Mix must resolve here.
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = patch.materialIdx;
	const MaterialData mat = wf_resolve_mix_material(wf_params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	WfShadowPayload* sp = (WfShadowPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	if (mat.type == MaterialType::DiffuseLight) {
		sp->occluded = false;
		optixTerminateRay();
		return;
	}
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission) {
		optixIgnoreIntersection();
		return;
	}
	sp->occluded = true;
	optixTerminateRay();
}

extern "C" __global__ void __anyhit__wf_shadow_triangle() {
	const int instBase = wf_instance_base();
	const TriangleData& tri = wf_params.triangles[wf_prim_base(instBase) + optixGetPrimitiveIndex()];
	// Mix must resolve before even the alpha-cutout check just below - a Mix
	// wrapper's own alphaMaskTexIdx is always -1, so checking it unresolved
	// would silently skip a sub-material's real alpha cutout. See
	// __anyhit__wf_shadow_sphere's own comment.
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = tri.materialIdx;
	const MaterialData mat = wf_resolve_mix_material(wf_params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	// Alpha-cutout: a transparent pixel of a leaf/foliage material casts no
	// shadow there - see __anyhit__wf_triangle's own comment for why
	// optixGetAttribute_0/1() (this hit group's own custom-intersection
	// barycentrics), not optixGetTriangleBarycentrics(). No-op for the
	// overwhelming majority of triangles, whose material has no alpha mask.
	if (mat.alphaMaskTexIdx >= 0) {
		// Barycentric UV fallback - see __anyhit__wf_triangle's own comment.
		const float b1 = __int_as_float(optixGetAttribute_0());
		const float b2 = __int_as_float(optixGetAttribute_1());
		const float b0 = 1.0f - b1 - b2;
		float uv_u = b1, uv_v = b2;
		if (tri.hasUVs) {
			uv_u = b0 * tri.uv0.x + b1 * tri.uv1.x + b2 * tri.uv2.x;
			uv_v = b0 * tri.uv0.y + b1 * tri.uv1.y + b2 * tri.uv2.y;
		}
		if (!wf_passes_alpha_cutout(mat.alphaMaskTexIdx, uv_u, uv_v)) {
			optixIgnoreIntersection();
			return;
		}
	}

	WfShadowPayload* sp = (WfShadowPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	if (mat.type == MaterialType::DiffuseLight) {
		sp->occluded = false;
		optixTerminateRay();
		return;
	}
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission) {
		optixIgnoreIntersection();
		return;
	}
	sp->occluded = true;
	optixTerminateRay();
}

// Disk/Cylinder (Phase 4c) - same simple DiffuseLight/dielectric-family/
// opaque list as quad/bilinear-patch above, matching the recursive backend's
// __anyhit__shadow_disk/__anyhit__shadow_cylinder (optix_anyhit_shadow.h)
// exactly - the pbrt loader never assigns Medium/DielectricMedium to a
// disk/cylinder (see pbrt_gpu_builder.h's disk/cylinder loop), so no
// CloudMedium/RgbGridMedium/Medium/DielectricMedium entries are needed here.
extern "C" __global__ void __anyhit__wf_shadow_disk() {
	const DiskData& disk = wf_params.disks[optixGetPrimitiveIndex()];
	// See __anyhit__wf_shadow_sphere's own comment for why Mix must resolve here.
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = disk.materialIdx;
	const MaterialData mat = wf_resolve_mix_material(wf_params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	WfShadowPayload* sp = (WfShadowPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	if (mat.type == MaterialType::DiffuseLight) {
		sp->occluded = false;
		optixTerminateRay();
		return;
	}
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission) {
		optixIgnoreIntersection();
		return;
	}
	sp->occluded = true;
	optixTerminateRay();
}

extern "C" __global__ void __anyhit__wf_shadow_cylinder() {
	const CylinderData& cyl = wf_params.cylinders[optixGetPrimitiveIndex()];
	// See __anyhit__wf_shadow_sphere's own comment for why Mix must resolve here.
	const float shadow_t = optixGetRayTmax();
	const float3 shadow_hit_point = optixGetWorldRayOrigin() + shadow_t * optixGetWorldRayDirection();
	int matIdx = cyl.materialIdx;
	const MaterialData mat = wf_resolve_mix_material(wf_params.materials[matIdx], matIdx, shadow_hit_point, matIdx);

	WfShadowPayload* sp = (WfShadowPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	if (mat.type == MaterialType::DiffuseLight) {
		sp->occluded = false;
		optixTerminateRay();
		return;
	}
	// MaterialType::Medium: see __anyhit__shadow_cylinder's identical fix
	// (optix_anyhit_shadow.h) - a shadow ray through a fog cylinder should
	// pass through, not be reported as a hard occluder.
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission ||
		mat.type == MaterialType::Medium) {
		optixIgnoreIntersection();
		return;
	}
	sp->occluded = true;
	optixTerminateRay();
}

