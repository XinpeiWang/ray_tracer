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
	const MaterialData& mat = params.materials[sphere.materialIdx];

	// IMPORTANT: When hitting light source, set NOT occluded and terminate
	// This allows the shadow ray to "see" the light
	if (mat.type == MaterialType::DiffuseLight) {
		optixSetPayload_0(0);  // NOT occluded - light is visible
		optixTerminateRay();
		return;
	}

	// Transmissive materials let light through -- ignore them in shadow rays.
	// Medium (participating media) is included here too: full Beer-Lambert
	// shadow-ray transmittance isn't implemented, so it's treated as
	// non-occluding (light passes straight through) rather than wrongly
	// blocking NEE entirely - a reasonable simplification matching how
	// this file already approximates other volumetric-adjacent cases.
	// CloudMedium's and RgbGridMedium's trigger spheres get the same
	// treatment for the same reason - without this, every shadow ray toward
	// a light on the far side of one of these bounding spheres would be
	// wrongly treated as fully occluded, rather than just passing through
	// unattenuated.
	//
	// MaterialType::DielectricMedium belongs in this list for the identical
	// reason, and its omission was a real bug (found while adding real NEE
	// to that material's medium-interior phase-scatter case - see
	// optix_intersection_sphere.h's DielectricMedium comment): CPU's
	// equivalent two-hittable construction (an outer `class dielectric`
	// shell wrapping a `constant_medium` fill, src/TheRestOfYourLife/
	// scenes_advanced.h build_subsurface_slab()) is non-occluding for
	// shadow rays on BOTH layers (dielectric::is_shadow_transmissive() and
	// hg_phase_material::is_shadow_transmissive(), material_simple.h /
	// constant_medium.h, both return true) - so CPU shadow rays pass
	// straight through the wax slab/jade sphere entirely. Without
	// DielectricMedium here, GPU's shadow any-hit instead treated that same
	// sphere/box as a fully opaque occluder, so EVERY shadow ray whose path
	// crossed the slab/jade sphere - not just the new phase-scatter NEE's
	// own shadow rays, but any other surface's NEE shadow ray that happened
	// to graze it too - was wrongly reported occluded. This alone made the
	// phase-scatter NEE fix above measure as a no-op (every one of its
	// shadow rays died right here, at the medium's own boundary, before
	// ever reaching the light).
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission ||
		mat.type == MaterialType::Medium ||
		mat.type == MaterialType::CloudMedium ||
		mat.type == MaterialType::RgbGridMedium ||
		mat.type == MaterialType::DielectricMedium) {
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
	const MaterialData& mat = params.materials[quad.materialIdx];

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
		mat.type == MaterialType::DiffuseTransmission) {
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
	const MaterialData& mat = params.materials[patch.materialIdx];

	if (mat.type == MaterialType::DiffuseLight) {
		optixSetPayload_0(0);
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
	const MaterialData& mat = params.materials[tri.materialIdx];

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
		const float t = optixGetRayTmax();
		const float3 hit_point = optixGetWorldRayOrigin() + t * optixGetWorldRayDirection();
		if (!passes_alpha_cutout(mat.alphaMaskTexIdx, uv_u, uv_v, hit_point)) {
			optixIgnoreIntersection();
			return;
		}
	}

	if (mat.type == MaterialType::DiffuseLight) {
		optixSetPayload_0(0);
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

	optixSetPayload_0(1);  // occluded = true
	optixTerminateRay();
}

//==============================================================================
// Miss Program
//==============================================================================

