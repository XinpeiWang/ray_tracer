// optix_anyhit_shadow.h -- Shadow any-hit programs
// Included by optix_programs.cu

extern "C" __global__ void __anyhit__shadow_sphere() {
	// Get primitive and material
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const SphereData& sphere = params.spheres[primIdx];
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
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission ||
		mat.type == MaterialType::Medium) {
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
	const TriangleData& tri = params.triangles[primIdx];
	const MaterialData& mat = params.materials[tri.materialIdx];

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

