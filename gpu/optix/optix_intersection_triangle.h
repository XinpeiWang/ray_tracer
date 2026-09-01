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
//
// Alpha-cutout (MaterialData::alphaMaskTexIdx, see __anyhit__triangle
// below): unlike the from-scratch intersection routine mentioned above,
// this any-hit program is a standard, low-risk OptiX hook that doesn't
// replace the built-in intersection test at all - it only conditionally
// rejects an already-computed hit via optixIgnoreIntersection(), so it
// isn't implicated by that prior instability.

//==============================================================================
// Triangle Any-Hit Program (radiance rays only - alpha-cutout)
//==============================================================================
// Registered on the SAME hit group as the closest-hit program below
// (optix_renderer.cpp's triangleHitDesc now sets both moduleCH and
// moduleAH), not a new program group - OptiX runs any-hit for every
// candidate intersection along a ray BEFORE closest-hit is decided, which
// is exactly when a transparent pixel needs to be skipped so it can't win
// as "the" hit in the first place; closest-hit runs too late for that. A
// no-op (returns immediately) for the overwhelming majority of triangles,
// whose material has no alpha mask.
extern "C" __global__ void __anyhit__triangle() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const int instBase = params.instancePrimBase
		? params.instancePrimBase[optixGetInstanceId()] : -1;
	const unsigned int triBase = (instBase >= 0) ? (unsigned int)instBase : 0u;
	const TriangleData& tri = params.triangles[triBase + primIdx];
	// Cheap common-case gate BEFORE paying for hit_point/Mix-resolution: a
	// triangle whose raw (unresolved) material is neither Mix nor
	// alpha-masked can never need either, so skip both entirely - this
	// program runs for every candidate BVH intersection along every ray,
	// not just the eventual closest hit, and the overwhelming majority of
	// triangles are neither. A Mix wrapper's own alphaMaskTexIdx is always
	// -1 (never authored directly on it), so Mix must still fall through
	// here even though its own alphaMaskTexIdx looks like "no mask" -
	// resolution might reveal a sub-material that has one.
	const MaterialData& raw_mat = params.materials[tri.materialIdx];
	if (raw_mat.type != MaterialType::Mix && raw_mat.alphaMaskTexIdx < 0) return;

	const float t = optixGetRayTmax();
	const float3 hit_point = optixGetWorldRayOrigin() + t * optixGetWorldRayDirection();
	// Mix must resolve before alphaMaskTexIdx is read - see MaterialType::
	// Mix's own comment (optix_types.h).
	int matIdx = tri.materialIdx;
	const MaterialData mat = resolve_mix_material(params.materials[matIdx], matIdx, hit_point, matIdx);
	if (mat.alphaMaskTexIdx < 0) return;

	// Real UV when authored, else the same barycentric fallback CPU's own
	// triangle.h uses (rec.u=b1,rec.v=b2) - a degenerate always-(0,0) UV
	// here would sample the exact same alpha-mask texel for the whole
	// mesh regardless of hit position, same bug class as untextured
	// shading (see __closesthit__triangle's own comment below).
	const float2 bary = optixGetTriangleBarycentrics();
	const float b1 = bary.x, b2 = bary.y, b0 = 1.0f - b1 - b2;
	float uv_u = b1, uv_v = b2;
	if (tri.hasUVs) {
		uv_u = b0 * tri.uv0.x + b1 * tri.uv1.x + b2 * tri.uv2.x;
		uv_v = b0 * tri.uv0.y + b1 * tri.uv1.y + b2 * tri.uv2.y;
	}
	if (!passes_alpha_cutout(mat.alphaMaskTexIdx, uv_u, uv_v, hit_point))
		optixIgnoreIntersection();
}

//==============================================================================
// Triangle Closest Hit Program
//==============================================================================

extern "C" __global__ void __closesthit__triangle() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	// A primitive index is local to its GAS, so an instanced definition's
	// triangle 0 and the scene's triangle 0 are different triangles. The base
	// table maps this instance back to its slice of the global array; a scene
	// with no instancing has no table and lands on base 0, exactly as before.
	//
	// The table entry is SIGNED and -1 is a real sentinel, not "unused": the
	// scene's own two instances (custom-prim id 0, triangle id 1) get -1
	// because their triangles already sit in world space - only a placement
	// of an instance DEFINITION gets a real (>= 0) base. instBase is read
	// again below to gate the normal transform on the same condition.
	const int instBase = params.instancePrimBase
		? params.instancePrimBase[optixGetInstanceId()] : -1;
	const unsigned int triBase = (instBase >= 0) ? (unsigned int)instBase : 0u;
	const TriangleData& tri = params.triangles[triBase + primIdx];

	const float t = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t * ray_dir;

	// Resolved to a real, non-Mix material before any mat.type branch below -
	// see MaterialType::Mix's own comment (optix_types.h).
	int matIdx = tri.materialIdx;
	const MaterialData mat = resolve_mix_material(params.materials[matIdx], matIdx, hit_point, matIdx);

	float3 shading_normal;
	// Real UV when authored, else the same barycentric fallback CPU's own
	// triangle.h uses (rec.u=b1,rec.v=b2 - see that file's own "barycentric
	// fallback" comment) - this used to stay at a fixed (0,0) for every
	// point on every UV-less mesh, which for a textured material samples
	// the exact same texel across the whole surface regardless of hit
	// position (the "solid black on GPU" bug docs/PBRT_SUPPORT.md tracked -
	// texel (0,0) commonly falls on a transparent/black image border).
	const float2 bary = optixGetTriangleBarycentrics();
	const float b1 = bary.x, b2 = bary.y, b0 = 1.0f - b1 - b2;
	float uv_u = b1, uv_v = b2;
	if (tri.hasNormals || tri.hasUVs) {
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
	// Object-to-world normal transform, but ONLY for genuinely instanced
	// geometry (instBase >= 0 - see its comment above).
	//
	// db7a609 first added this unconditionally, reasoning a non-instanced GAS
	// has an identity transform so the call is a no-op there. That was true
	// for the transform itself, but the bug was elsewhere: a stacked, unrelated
	// regression (2e8c79b) made every loaded pbrt scene render black regardless,
	// which made the unconditional version LOOK responsible when it was
	// reverted alongside the real fix. With that other bug now fixed and a
	// scene actually rendering, gating this call on instBase costs nothing on
	// the identity path (skipped entirely rather than executed as a no-op) and
	// is what makes instanced normals correct once instances exist.
	if (instBase >= 0) {
		shading_normal = normalize(optixTransformNormalFromObjectToWorldSpace(shading_normal));
	}

	const bool front_face = dot(ray_dir, shading_normal) < 0.0f;
	const float3 final_normal = front_face ? shading_normal : -shading_normal;

	// Unpack payload from registers
	float3 attenuation_in = make_float3(
		__uint_as_float(optixGetPayload_0()),
		__uint_as_float(optixGetPayload_1()),
		__uint_as_float(optixGetPayload_2())
	);
	unsigned int seed = optixGetPayload_9();

	// Get emission from material - see material_emission()'s own comment
	// (optix_device_helpers.h) for why this goes through an accessor rather
	// than reading mat.emission raw. Triangles are the only geometry type
	// that currently ever has a real map_Ke texture (e.g. Gallery's
	// painted-canvas glow - see add_diffuse_light()'s textureIdx comment),
	// hence the only caller that passes uv_u/uv_v/hit_point through.
	float3 emission = material_emission(mat, front_face, uv_u, uv_v, hit_point);

	float3 attenuation;
	float3 scattered_dir;
	bool scattered = false;
	bool is_specular = false;
	bool is_medium_boundary = false;  // MaterialType::Interface - see optix_types.h
	float brdf_pdf_override = -1.0f;
	bool bssrdf_exit = false;
	float3 bssrdf_exit_pos = make_float3(0.0f, 0.0f, 0.0f);

	// NormalMappedLambertian: mirrors optix_intersection_sphere.h's own
	// handling (decode the tangent-space normal, perturb, then shade as
	// plain Lambertian with the perturbed normal) but with a REAL tangent
	// instead of a world-up cross-product approximation - triangles carry
	// actual UVs, so the standard pbrt-v4 Triangle::Intersect approach
	// (solve the 2x2 system relating edge vectors to UV deltas) applies
	// directly, unlike a sphere's arbitrary-at-the-poles tangent frame.
	// Was previously unhandled here entirely (fell through shade_material()'s
	// switch to its default: scattered=false, i.e. every normal-mapped
	// triangle silently absorbed all light) - this is what actually wires
	// the material type into triangle shading for the first time.
	//
	// Also needed by shade_material()'s 4 anisotropy-capable material kinds
	// for their own UV-aligned shading frame (see BuildDpduTangentFrame's
	// own comment, microfacet.h) - gated on material_needs_dpdu() so the
	// 2x2 solve + normalize is skipped entirely for every other material
	// kind (Lambertian, Metal, Dielectric, DiffuseLight, etc. - the
	// overwhelming common case), which never reads it.
	float3 tri_dpdu = make_float3(0.0f, 0.0f, 0.0f);
	if (material_needs_dpdu(mat.type)) {
		const float3 e1 = tri.p1 - tri.p0;
		const float3 e2 = tri.p2 - tri.p0;
		const float du1 = tri.uv1.x - tri.uv0.x, dv1 = tri.uv1.y - tri.uv0.y;
		const float du2 = tri.uv2.x - tri.uv0.x, dv2 = tri.uv2.y - tri.uv0.y;
		const float det = du1 * dv2 - dv1 * du2;
		if (fabsf(det) > 1e-12f) {
			const float invDet = 1.0f / det;
			tri_dpdu = (dv2 * invDet) * e1 - (dv1 * invDet) * e2;
			const float dpdu_len = length(tri_dpdu);
			tri_dpdu = (dpdu_len > 1e-8f) ? (tri_dpdu / dpdu_len) : normalize(e1);
		} else {
			tri_dpdu = normalize(e1);
		}
		if (instBase >= 0) tri_dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(tri_dpdu));
	}

	float3 shade_normal = final_normal;
	MaterialData shade_mat = mat;
	if (mat.type == MaterialType::NormalMappedLambertian) {
		const float3& dpdu = tri_dpdu;
		const float3 packed = sample_texture(mat.textureIdx, uv_u, uv_v, hit_point);
		float ns_x = 2.0f * packed.x - 1.0f;
		float ns_y = 2.0f * packed.y - 1.0f;
		float ns_z = 2.0f * packed.z - 1.0f;
		const float ns_len = sqrtf(ns_x * ns_x + ns_y * ns_y + ns_z * ns_z);
		if (ns_len > 1e-8f) { ns_x /= ns_len; ns_y /= ns_len; ns_z /= ns_len; }
		else                { ns_x = 0.0f; ns_y = 0.0f; ns_z = 1.0f; }

		float out_nx, out_ny, out_nz;
		apply_normal_map(ns_x, ns_y, ns_z, final_normal.x, final_normal.y, final_normal.z,
			dpdu.x, dpdu.y, dpdu.z, out_nx, out_ny, out_nz);
		shade_normal = make_float3(out_nx, out_ny, out_nz);

		// Reuses shade_material()'s existing Lambertian NEE/MIS logic
		// verbatim rather than duplicating it; textureIdx is cleared since
		// it means "normal map" for this material type, not "albedo
		// texture" the way Lambertian itself reads it - matches the sphere
		// path's identical effective-material substitution.
		shade_mat.type = MaterialType::Lambertian;
		shade_mat.textureIdx = -1;
	} else if (mat.type == MaterialType::Hair) {
		// Real Hair support on triangle geometry, using the shading normal as
		// the fiber-tangent proxy - same simplification as sphere's own Hair
		// branch (optix_intersection_sphere.h) for any non-curve shape (real
		// curve geometry never produces triangles - see curve_tessellate.h,
		// bilinear patches only). Was previously part of the sphere-only trap
		// below - safe while Material "hair" could never actually reach a
		// triangle's material index (pbrt_flatten.h mapped it to
		// MaterialKind::Unsupported), but once Round 7 wired it up for real,
		// an ordinary Shape "trianglemesh" with Material "hair" trapped the
		// whole render instead of falling back the way every other
		// unsupported combination does.
		scattered   = sample_hair_material(ray_dir, final_normal, mat, seed, scattered_dir, attenuation);
		is_specular = true;
	} else if (material_requires_sphere_only_handling(mat.type)) {
		// Medium/DielectricMedium/CloudMedium/RgbGridMedium/Principled need
		// sphere-specific handling this file doesn't implement (see
		// material_requires_sphere_only_handling()'s comment - Hair no longer
		// belongs to this trapped set, see the branch above). Trap with a
		// specific message rather than falling through to shade_material()'s
		// own generic "unhandled MaterialType" default.
		printf("[TRIANGLE-SHADE] MaterialType %d is not supported on triangle geometry\n", (int)mat.type);
		__trap();
	}

	float out_eta = 1.0f;
	// See PathTracingPayload's own p24/rgbChannel comment (optix_renderer_init.cpp)
	// and shade_material()'s inout_rgb_channel parameter comment - read
	// unconditionally (even when shade_material() isn't called below, e.g.
	// Hair) so a channel already chosen by an earlier bounce still survives
	// unchanged through this one.
	unsigned int rgbChannel = optixGetPayload_24();
	if (mat.type != MaterialType::Hair) {
		// Integrator "bool regularize" - see current_do_regularize()'s own
		// comment (optix_device_helpers.h).
		const bool do_regularize = current_do_regularize();
		shade_material(shade_mat, matIdx, shade_normal, ray_dir, hit_point, front_face, uv_u, uv_v, tri_dpdu, do_regularize, seed,
			attenuation, scattered_dir, scattered, is_specular, is_medium_boundary, brdf_pdf_override, emission,
			bssrdf_exit, bssrdf_exit_pos, out_eta, rgbChannel);
	}

	optixSetPayload_3(__float_as_uint(emission.x));
	optixSetPayload_4(__float_as_uint(emission.y));
	optixSetPayload_5(__float_as_uint(emission.z));
	optixSetPayload_9(seed);

	// p16-p21: denoiser guide-layer AOVs (recursive backend only) - see
	// pack_aov_payload()'s own comment (optix_device_helpers.h) for why
	// this is unconditional (every branch below, not just `scattered`).
	// `attenuation` is only guaranteed written when `scattered` is true
	// (it's an uninitialized local otherwise) - mat.albedo is the safe
	// always-valid fallback for the DiffuseLight/absorbed branches.
	// `shade_normal` itself is always valid here regardless of branch.
	{
		float3 albedoAov = scattered ? attenuation : mat.albedo;
		pack_aov_payload(albedoAov, shade_normal);
	}
	optixSetPayload_22(__float_as_uint(out_eta));  // pbrt-v4 etaScale - see PathTracingPayload::eta
	optixSetPayload_24(rgbChannel);  // dispersion channel - see shade_material()'s inout_rgb_channel comment

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
		// flag 3 (not 1): MaterialType::Subsurface's probe walk found an
		// exit point off to the side of this ray - the next ray must
		// originate there, not at hit_point (see optix_raygen.h's flag==3
		// handling and shade_material()'s own out_bssrdf_exit comment).
		// flag 4: MaterialType::Interface - see optix_raygen.h's flag==4 branch.
		optixSetPayload_10(pack_scatter_flag(bssrdf_exit, is_medium_boundary, is_specular));  // scattered (see pack_scatter_flag's own comment)
		optixSetPayload_11(__float_as_uint(t_hit));
		optixSetPayload_12(__float_as_uint(brdf_pdf_out));
		if (bssrdf_exit) {
			optixSetPayload_13(__float_as_uint(bssrdf_exit_pos.x));
			optixSetPayload_14(__float_as_uint(bssrdf_exit_pos.y));
			optixSetPayload_15(__float_as_uint(bssrdf_exit_pos.z));
		}
	} else if (mat.type == MaterialType::DiffuseLight) {
		// NEE pdf for the incoming direction reaching this triangle light, so
		// MIS in raygen can weight a BSDF-sampled path that happened to land
		// here against the light-sampled estimate of the same thing. Mirrors
		// the quad program's block exactly, differing only in halving the
		// parallelogram's area - see sample_triangle_light().
		//
		// This used to be an unconditional pdf=0 on the grounds that no scene
		// emits from a triangle. A pbrt file whose area light will not fold
		// into a parallelogram now does, and leaving the pdf at zero would
		// hand MIS a weight of 1 for the BSDF strategy - double-counting the
		// light rather than merely losing a little quality.
		float light_pdf_for_incoming = 0.0f;
		if (params.aliasTable && params.numLights > 0) {
			// Global index: lightIndices addresses the scene's own triangle
			// array. An instanced triangle is never emissive, so its global
			// index simply never matches, which is the correct answer.
			const int prim_global = (int)(triBase + primIdx);
			float sel_pdf = 0.0f;
			for (unsigned int li = 0; li < params.numLights; ++li) {
				if (params.lightIndices[li] == prim_global &&
					params.lightKinds[li] == GpuLightKind::Triangle) {
					sel_pdf = params.aliasTable[li].pdf;
					break;
				}
			}
			if (sel_pdf > 0.0f) {
				const float3 e1 = tri.p1 - tri.p0;
				const float3 e2 = tri.p2 - tri.p0;
				const float3 n_unnorm = cross(e1, e2);
				const float twice_area = length(n_unnorm);
				const float area = 0.5f * twice_area;
				const float dist_sq = t * t * dot(ray_dir, ray_dir);
				const float cosine = (twice_area > 1e-12f)
					? fabsf(dot(ray_dir, n_unnorm / twice_area)) : 0.0f;
				if (cosine > 1e-6f && area > 1e-12f)
					light_pdf_for_incoming = sel_pdf * dist_sq / (cosine * area);
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
