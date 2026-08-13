// optix_intersection_triangle.h -- Triangle closest-hit program
// Included by optix_programs.cu
//
// Intersection itself uses OptiX's built-in hardware triangle test
// (OPTIX_BUILD_INPUT_TYPE_TRIANGLES, see optix_renderer.cpp's
// buildAccelerationStructure) rather than a custom intersection program -
// matches pbrt-v4's own GPU renderer, which likewise gives triangles
// OptiX's native path (passing nullptr for the IS program) while keeping
// quadrics/bilinear-patches as custom AABB primitives with hand-written
// intersection routines, exactly as this codebase still does for spheres/
// quads/bilinear-patches. A prior version of this file carried a from-
// scratch watertight Woop/Moller-Trumbore intersection routine for
// triangles too; it was replaced after an extensive investigation traced a
// GPU-only "meshes cast no shadow" bug to that custom path (root cause
// unresolved - the built-in path sidesteps it entirely, matching the
// battle-tested approach every other OptiX renderer uses for triangles).
//
// Shading normal: interpolated from TriangleData::n0/n1/n2 via the
// built-in intersection's barycentric attributes when the source mesh had
// "vn" data (tri.hasNormals - see scene_builder.cpp's load_obj_triangles_gpu
// and optix_types.h's TriangleData), else the flat geometric normal
// cross(e1,e2). optixGetTriangleBarycentrics() returns (u,v) weighting
// p1/p2 with p0's weight (1-u-v) - the same b0/b1/b2 convention CPU's
// src/TheRestOfYourLife/triangle.h uses for its own p0/p1/p2-ordered
// interpolation, so no reordering is needed to match it.

//==============================================================================
// Triangle Closest Hit Program
//==============================================================================

extern "C" __global__ void __closesthit__triangle() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	// A primitive index is local to its GAS, so an instanced definition's
	// triangle 0 and the scene's triangle 0 are different triangles. The base
	// table maps this instance back to its slice of the global array; a scene
	// with no instancing has no table and lands on 0, exactly as before.
	const unsigned int triBase = params.instanceTriBase
		? (unsigned int)params.instanceTriBase[optixGetInstanceId()] : 0u;
	const TriangleData& tri = params.triangles[triBase + primIdx];
	const MaterialData& mat = params.materials[tri.materialIdx];

	const float t = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t * ray_dir;

	float3 shading_normal;
	float uv_u = 0.0f, uv_v = 0.0f;
	if (tri.hasNormals || tri.hasUVs) {
		const float2 bary = optixGetTriangleBarycentrics();
		const float b1 = bary.x, b2 = bary.y, b0 = 1.0f - b1 - b2;
		if (tri.hasNormals) {
			shading_normal = normalize(b0 * tri.n0 + b1 * tri.n1 + b2 * tri.n2);
		} else {
			shading_normal = normalize(cross(tri.p1 - tri.p0, tri.p2 - tri.p0));
		}
		if (tri.hasUVs) {
			uv_u = b0 * tri.uv0.x + b1 * tri.uv1.x + b2 * tri.uv2.x;
			uv_v = b0 * tri.uv0.y + b1 * tri.uv1.y + b2 * tri.uv2.y;
		}
	} else {
		shading_normal = normalize(cross(tri.p1 - tri.p0, tri.p2 - tri.p0));
	}
	// NO object-to-world normal transform here.
	//
	// db7a609 added optixTransformNormalFromObjectToWorldSpace() at this
	// point, ahead of object instancing, reasoning that a non-instanced GAS
	// has an identity transform so the call would be a no-op. That reasoning
	// was untested against a scene containing any triangles (scene 0, the
	// regression check used, has zero) and it was wrong: with it in place
	// every loaded pbrt scene rendered pure black on GPU. Confirmed by
	// isolated bisection - removing only this line, output cleared and PTX
	// recompiled before each measurement, brings the render back.
	//
	// When instancing is implemented for real, this belongs here again but
	// gated on the geometry actually being instanced (e.g. only when
	// params.instanceTriBase is non-null AND resolves to a real instance for
	// this primitive) - and it must be re-verified against a scene that
	// actually contains triangles, not scene 0.

	const bool front_face = dot(ray_dir, shading_normal) < 0.0f;
	const float3 final_normal = front_face ? shading_normal : -shading_normal;

	// Unpack payload from registers
	float3 attenuation_in = make_float3(
		__uint_as_float(optixGetPayload_0()),
		__uint_as_float(optixGetPayload_1()),
		__uint_as_float(optixGetPayload_2())
	);
	unsigned int seed = optixGetPayload_9();

	float3 emission = mat.emission;

	float3 attenuation;
	float3 scattered_dir;
	bool scattered = false;
	bool is_specular = false;
	float brdf_pdf_override = -1.0f;

	shade_material(mat, final_normal, ray_dir, hit_point, front_face, uv_u, uv_v, seed,
		attenuation, scattered_dir, scattered, is_specular, brdf_pdf_override, emission);

	optixSetPayload_3(__float_as_uint(emission.x));
	optixSetPayload_4(__float_as_uint(emission.y));
	optixSetPayload_5(__float_as_uint(emission.z));
	optixSetPayload_9(seed);

	if (scattered) {
		float t_hit = optixGetRayTmax();
		float brdf_pdf_out = is_specular ? 0.0f
						  : (brdf_pdf_override >= 0.0f ? brdf_pdf_override : cosine_pdf(scattered_dir, final_normal));

		optixSetPayload_0(__float_as_uint(attenuation.x));
		optixSetPayload_1(__float_as_uint(attenuation.y));
		optixSetPayload_2(__float_as_uint(attenuation.z));
		optixSetPayload_6(__float_as_uint(scattered_dir.x));
		optixSetPayload_7(__float_as_uint(scattered_dir.y));
		optixSetPayload_8(__float_as_uint(scattered_dir.z));
		optixSetPayload_10(1);
		optixSetPayload_11(__float_as_uint(t_hit));
		optixSetPayload_12(__float_as_uint(brdf_pdf_out));
	} else if (mat.type == MaterialType::DiffuseLight) {
		// Triangles are never registered as lights by scene_builder.cpp (no
		// scene emits light from a triangle mesh), so this is unreachable in
		// practice - pdf=0 is safe, matching bilinear patch's same note.
		optixSetPayload_10(2);
		optixSetPayload_12(0);
	} else {
		optixSetPayload_10(0);
		optixSetPayload_12(0);
	}
}
