// optix_device_helpers.h -- Shared device utilities for OptiX programs
// Included by optix_programs.cu

// OptiX Device Programs
// Ray generation, intersection, closest-hit, and miss programs

#include <optix.h>
#include "optix_types.h"
#include "optix_math_helpers.h"
#include "../../src/shared/fresnel.h"    // Shared exact Fresnel (CPU+GPU)
#include "../../src/shared/microfacet.h" // GGX TrowbridgeReitz (CPU+GPU)
#include "../../src/shared/bxdfs.h"      // HairBxDF<T> (CPU+GPU) - see MaterialType::Hair
#include "../../src/shared/noise.h"      // Perlin turbulence (CPU+GPU) - see sample_texture()
#include "../../src/shared/normal_map.h" // apply_normal_map (CPU+GPU) - see MaterialType::NormalMappedLambertian
#include "../../src/shared/bilinear_patch.h" // blp_sample/blp_pdf_wi (CPU+GPU) - see GpuLightKind::BilinearPatch
#include "../../src/shared/shading_frame.h"  // ShadingFrame<T> (CPU+GPU) - see MaterialType::Measured

// Launch parameters (constant across all threads)
extern "C" { __constant__ LaunchParams params; }

// Device-side tabulated-BSSRDF evaluation (MaterialType::Subsurface,
// recursive backend only) - needs `params` above for the flat table arrays,
// so this include must stay below it.
#include "optix_bssrdf.h"

//==============================================================================
// Utility functions
//==============================================================================

// Random number generator (PCG)
__device__ __forceinline__ unsigned int pcg_hash(unsigned int seed) {
	unsigned int state = seed * 747796405u + 2891336453u;
	unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

__device__ __forceinline__ float random_float(unsigned int& seed) {
	seed = pcg_hash(seed);
	return float(seed) / 4294967296.0f;
}

__device__ __forceinline__ float3 random_float3(unsigned int& seed) {
	return make_float3(random_float(seed), random_float(seed), random_float(seed));
}

__device__ __forceinline__ float3 random_in_unit_sphere(unsigned int& seed) {
	while (true) {
		float3 p = 2.0f * random_float3(seed) - make_float3(1.0f, 1.0f, 1.0f);
		if (dot(p, p) < 1.0f) return p;
	}
}

__device__ __forceinline__ float3 random_unit_vector(unsigned int& seed) {
	return normalize(random_in_unit_sphere(seed));
}

// Rejection-sampled point in the unit disk (z=0). Mirrors src/TheRestOfYourLife/
// vec3.h's random_in_unit_disk() - used by the thin-lens depth-of-field camera.
__device__ __forceinline__ float3 random_in_unit_disk(unsigned int& seed) {
	while (true) {
		float3 p = make_float3(2.0f * random_float(seed) - 1.0f, 2.0f * random_float(seed) - 1.0f, 0.0f);
		if (dot(p, p) < 1.0f) return p;
	}
}

// Henyey-Greenstein phase function importance sampling. `wo` is the
// direction of travel (incoming ray direction, forward); returns a new
// travel direction sampled relative to it - g>0 biases toward continuing
// forward (near wo), g<0 biases backward, g=0 is isotropic. Mirrors
// src/shared/volume_scattering.h's HenyeyGreensteinPhaseFunction (itself a
// port of pbrt-v4's HGPhaseFunction::Sample_p).
__device__ __forceinline__ float3 sample_henyey_greenstein(const float3& wo, float g, unsigned int& seed) {
	float u1 = random_float(seed), u2 = random_float(seed);
	float cos_theta;
	if (fabsf(g) < 1e-3f) {
		cos_theta = 1.0f - 2.0f * u1;
	} else {
		float sqr = (1.0f - g * g) / (1.0f + g - 2.0f * g * u1);
		cos_theta = -(1.0f + g * g - sqr * sqr) / (2.0f * g);
	}
	float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
	float phi = 2.0f * 3.14159265358979323846f * u2;
	float3 t1 = (fabsf(wo.x) > 0.9f) ? normalize(cross(make_float3(0, 1, 0), wo))
									  : normalize(cross(make_float3(1, 0, 0), wo));
	float3 t2 = cross(wo, t1);
	return normalize(sin_theta * cosf(phi) * t1 + sin_theta * sinf(phi) * t2 + cos_theta * wo);
}

// Henyey-Greenstein phase function VALUE p(cos_theta, g) - the same quantity
// as both the phase "BSDF" f and its own pdf (a properly normalized phase
// function is its own perfect importance sampler, exactly like
// src/TheRestOfYourLife/constant_medium.h's hg_phase_pdf::value() ==
// hg_phase_material::scattering_pdf(), both of which just call
// HenyeyGreensteinPhaseFunction<double>::p()). Hand-duplicated as a free
// function rather than calling src/shared/volume_scattering.h's
// HenyeyGreensteinPhaseFunction<T>::p() member function directly, matching
// sample_henyey_greenstein()'s own reason just above (this backend's
// recursive mega-kernel has previously stalled on CPU_GPU-tagged struct
// member-function calls - see this project's own notes on that issue; a
// hand-duplicated free function sidesteps it entirely, the same fix already
// applied there).
//
// Used by MaterialType::DielectricMedium's medium-interior phase-scatter
// case (optix_intersection_sphere.h) to do real NEE+MIS at that scatter
// event - see that call site's own comment for why (closing B13's
// CPU-vs-GPU brightness gap).
__device__ __forceinline__ float hg_phase_value(float cos_theta, float g) {
	float gc = fminf(0.99f, fmaxf(-0.99f, g));
	const float inv4pi = 1.0f / (4.0f * 3.14159265358979323846f);
	float denom = 1.0f + gc * gc + 2.0f * gc * cos_theta;
	return inv4pi * (1.0f - gc * gc) / (denom * sqrtf(fmaxf(1e-12f, denom)));
}

// MaterialType::Hair: Marschner/Chiang fiber scattering (src/shared/
// bxdfs_hair.h's HairBxDF<T>), using the shading normal as a fiber-tangent
// proxy - matches src/TheRestOfYourLife/hair_material.h::scatter() exactly
// (same "no literal fiber geometry" simplification, see MaterialType::Hair's
// comment in optix_types.h). Returns false if the sample should be rejected
// (mirrors hair_material.h's `if (!res.valid) return false;`).
__device__ __forceinline__ bool sample_hair_material(
	const float3& ray_dir, const float3& normal, const MaterialData& mat,
	unsigned int& seed, float3& scattered_dir, float3& attenuation)
{
	HairBxDF<float> bxdf(
		random_float(seed) * 2.0f - 1.0f,  // h in [-1,1], sampled per-scatter like the CPU
		mat.ior,                            // eta
		mat.albedo.x, mat.albedo.y, mat.albedo.z,  // sigma_a (RGB absorption)
		mat.fuzz,                           // beta_m
		mat.eta_c.x,                        // beta_n
		mat.eta_c.y);                       // alpha_deg

	float3 unit_dir = normalize(ray_dir);
	float u1 = random_float(seed), u2 = random_float(seed);
	float u3 = random_float(seed), u4 = random_float(seed);

	auto res = bxdf.sample(
		normal.x, normal.y, normal.z,
		unit_dir.x, unit_dir.y, unit_dir.z,
		u1, u2, u3, u4);

	if (!res.valid) return false;

	scattered_dir = make_float3(res.wo_x, res.wo_y, res.wo_z);
	attenuation   = make_float3(res.r, res.g, res.b);
	return true;
}

// MaterialType::Principled: Disney/pbrt-v4-style multi-lobe BSDF (src/shared/
// bxdfs_principled.h's PrincipledBxDF<T>) - matches src/TheRestOfYourLife/
// principled_material.h::scatter() exactly (same "instantiate the shared
// CPU_GPU BxDF struct directly" pattern as sample_hair_material() above).
// Returns false if the sample should be rejected (mirrors principled's
// `if (!res.valid) return false;`). Field reuse: albedo=base color, ior=ior,
// fuzz=roughness, eta_c.x=metallic, eta_c.y=clearcoat, eta_c.z=clearcoat_rough
// - see MaterialType::Principled's comment in optix_types.h.
__device__ __forceinline__ bool sample_principled_material(
	const float3& ray_dir, const float3& normal, const MaterialData& mat,
	unsigned int& seed, float3& scattered_dir, float3& attenuation)
{
	PrincipledBxDF<float> bxdf{
		mat.albedo.x, mat.albedo.y, mat.albedo.z,
		mat.eta_c.x,   // metallic
		mat.fuzz,      // roughness
		mat.ior,
		mat.eta_c.y,   // clearcoat
		mat.eta_c.z }; // clearcoat_rough

	float3 unit_dir = normalize(ray_dir);
	float u1 = random_float(seed), u2 = random_float(seed), u3 = random_float(seed);

	auto res = bxdf.sample(
		normal.x, normal.y, normal.z,
		unit_dir.x, unit_dir.y, unit_dir.z,
		u1, u2, u3);

	if (!res.valid) return false;

	scattered_dir = make_float3(res.wo_x, res.wo_y, res.wo_z);
	attenuation   = make_float3(res.r, res.g, res.b);
	return true;
}

// Device-side real tabulated-measured-BRDF evaluation (MaterialType::
// Measured, both GPU backends - this is the recursive backend's copy) -
// needs `params` (declared above) for the flat table arrays and
// random_float() (defined above) for sample_measured_material()'s own
// per-sample randoms, so this include must stay below both.
#include "optix_measured_bxdf.h"

// Device-side real importance-sampled HDR sky (LightSource "infinite" with
// an image) - needs `params` and random_float()/random_unit_vector() (all
// defined above), so this include must stay below them too. See that file's
// own header comment.
#include "optix_sky_light.h"

__device__ __forceinline__ float3 random_on_hemisphere(const float3& normal, unsigned int& seed) {
	float3 on_unit_sphere = random_unit_vector(seed);
	if (dot(on_unit_sphere, normal) > 0.0f)
		return on_unit_sphere;
	else
		return -on_unit_sphere;
}

__device__ __forceinline__ bool near_zero(const float3& v) {
	const float s = 1e-8f;
	return (fabsf(v.x) < s) && (fabsf(v.y) < s) && (fabsf(v.z) < s);
}

// reflect/refract from shared CPU/GPU header (pbrt-v4 pattern)
#include "../../src/shared/math_utils.h"
__device__ __forceinline__ float3 reflect(const float3& v, const float3& n) { return cpu_gpu_reflect(v, n); }
__device__ __forceinline__ float3 refract(const float3& uv, const float3& n, float e) { return cpu_gpu_refract<float3,float>(uv, n, e); }

// Schlick's approximation removed — FrDielectric<float> from shared/fresnel.h is used instead.

// Smooth dielectric reflect-or-refract, shared by MaterialType::Dielectric
// (shade_material's case below) and MaterialType::DielectricMedium
// (optix_intersection_sphere.h's inline handling, which needs this same
// surface interaction at both the entry AND exit surface, alongside its own
// medium free-path sampling that doesn't fit shade_material's generic
// per-material switch). `front_face` selects the eta ratio direction; `ior`
// is the material's index of refraction on the denser side.
__device__ __forceinline__ float3 dielectric_scatter(const float3& ray_dir, const float3& normal,
		bool front_face, float ior, unsigned int& seed) {
	float ri = front_face ? (1.0f / ior) : ior;
	float3 unit_direction = normalize(ray_dir);
	float cos_theta = fminf(dot(-unit_direction, normal), 1.0f);
	float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

	bool cannot_refract = ri * sin_theta > 1.0f;

	// FrDielectric expects eta_t/eta_i; ri = eta_i/eta_t, so pass 1/ri
	if (cannot_refract || FrDielectric(cos_theta, 1.0f / ri) > random_float(seed)) {
		return reflect(unit_direction, normal);
	} else {
		return refract(unit_direction, normal, ri);
	}
}

//==============================================================================
// Multiple Importance Sampling (MIS) Helpers
//==============================================================================

// MIS power heuristic (beta=2) -- delegates to shared PowerHeuristic (pbrt-v4 pattern)
__device__ __forceinline__ float mis_power_heuristic(float pdf_a, float pdf_b) {
	return PowerHeuristic(pdf_a, pdf_b);
}

// Cosine-weighted hemisphere sampling PDF
__device__ __forceinline__ float cosine_pdf(const float3& direction, const float3& normal) {
	float cosine = dot(normalize(direction), normal);
	return fmaxf(0.0f, cosine / 3.14159265358979323846f);
}

// Sample a random point on a sphere light
__device__ __forceinline__ float3 sample_sphere_light(
	const SphereData& sphere,
	const float3& origin,
	unsigned int& seed,
	float& pdf
) {
	// Direction from origin to sphere center
	float3 to_center = sphere.center - origin;
	float dist_sq = dot(to_center, to_center);

	// Avoid division by zero
	if (dist_sq < 1e-6f) {
		pdf = 0.0f;
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

	float3 direction = r * cosf(phi) * u + r * sinf(phi) * v + z * w;
	return normalize(direction);
}

// Sample a random point on a quad light
__device__ __forceinline__ float3 sample_quad_light(
	const QuadData& quad,
	const float3& origin,
	unsigned int& seed,
	float& pdf,
	float& out_dist
) {
	// Random point on quad surface
	float a = random_float(seed);
	float b = random_float(seed);
	float3 point = quad.Q + a * quad.u + b * quad.v;

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
	float& out_dist
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

	float3 to_light = point - origin;
	float dist_sq = dot(to_light, to_light);
	out_dist = sqrtf(dist_sq);
	if (out_dist < 1e-6f) { pdf = 0.0f; return make_float3(0.0f, 0.0f, 1.0f); }
	float3 direction = to_light / out_dist;

	// GEOMETRIC normal, not the interpolated shading normal: this converts an
	// area measure to a solid-angle one, which is a property of the surface
	// the sample was actually drawn on.
	const float3 n_unnorm = cross(e1, e2);
	const float twice_area = length(n_unnorm);
	const float area = 0.5f * twice_area;      // half the parallelogram
	if (twice_area < 1e-12f) { pdf = 0.0f; return direction; }
	const float3 normal = n_unnorm / twice_area;

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
	float& out_dist
) {
	const float p00[3] = {bp.p00.x, bp.p00.y, bp.p00.z};
	const float p10[3] = {bp.p10.x, bp.p10.y, bp.p10.z};
	const float p01[3] = {bp.p01.x, bp.p01.y, bp.p01.z};
	const float p11[3] = {bp.p11.x, bp.p11.y, bp.p11.z};
	const float u2[2] = {random_float(seed), random_float(seed)};
	float outP[3], outN[3], areaPdf = 0.0f;
	blp_sample(p00, p10, p01, p11, u2, outP, outN, &areaPdf);

	const float3 point = make_float3(outP[0], outP[1], outP[2]);
	const float3 normal = make_float3(outN[0], outN[1], outN[2]);
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
	float3& emission
) {
	const int prim_idx = params.lightIndices[light_idx];
	switch (params.lightKinds[light_idx]) {
	case GpuLightKind::Sphere: {
		const SphereData& s = params.spheres[prim_idx];
		const float3 dir = sample_sphere_light(s, origin, seed, geom_pdf);
		// Distance to the CENTRE, matching what this path has always used to
		// bound the shadow ray for a sphere light.
		max_dist = length(s.center - origin);
		emission = params.materials[s.materialIdx].emission;
		return dir;
	}
	case GpuLightKind::Triangle: {
		// The scene's own triangle array - an instanced triangle is never
		// emissive (flatten() bakes emitters per placement into world space),
		// so no per-instance base offset applies here.
		const TriangleData& t = params.triangles[prim_idx];
		const float3 dir = sample_triangle_light(t, origin, seed, geom_pdf, max_dist);
		emission = params.materials[t.materialIdx].emission;
		return dir;
	}
	case GpuLightKind::BilinearPatch: {
		const BilinearPatchData& bp = params.bilinearPatches[prim_idx];
		const float3 dir = sample_bilinear_patch_light(bp, origin, seed, geom_pdf, max_dist);
		emission = params.materials[bp.materialIdx].emission;
		return dir;
	}
	case GpuLightKind::Quad:
	default: {
		const QuadData& q = params.quads[prim_idx];
		const float3 dir = sample_quad_light(q, origin, seed, geom_pdf, max_dist);
		emission = params.materials[q.materialIdx].emission;
		return dir;
	}
	}
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
// lookup. Direct copy of src/shared/sampling_patched.h's EqualAreaSphereToSquare
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
			emission = emission + brdf * Li * cos_theta;
		}
	}
}

// Samples a texture by index (see TextureData in optix_types.h), matching
// CPU's texture::value(u, v, p) dispatch (src/TheRestOfYourLife/texture.h)
// for the two kinds ported so far - only called when
// MaterialData::textureIdx >= 0 (Lambertian only for now, see
// shade_material() below). u/v are used for Image; p (world-space hit
// point) is used for Noise - CPU's own base-class interface takes all
// three regardless of which the concrete texture subclass actually needs,
// same here.
__device__ __forceinline__ float3 sample_texture(int textureIdx, float u, float v, const float3& p) {
	const TextureData& tex = params.textures[textureIdx];
	if (tex.kind == TextureKind::Image) {
		// Matches image_texture::value() (texture.h:74-88) exactly: clamp
		// uv to [0,1], flip v (stored image rows are top-to-bottom, v=0 is
		// the bottom of the [0,1] texture-coordinate convention), clamp the
		// resulting integer pixel index to the image bounds (rtw_image::
		// pixel_data()'s own clamp), nearest-neighbor, 8-bit -> [0,1] float.
		// A failed image load (width/height <= 0) matches CPU's own solid-
		// cyan debugging fallback (texture.h:76) exactly.
		if (tex.width <= 0 || tex.height <= 0) return make_float3(0.0f, 1.0f, 1.0f);
		const float uc = fminf(fmaxf(u, 0.0f), 1.0f);
		const float vc = 1.0f - fminf(fmaxf(v, 0.0f), 1.0f);
		const int i = min(static_cast<int>(uc * tex.width), tex.width - 1);
		const int j = min(static_cast<int>(vc * tex.height), tex.height - 1);
		const unsigned char* px = params.texturePixels + tex.pixelOffset + (j * tex.width + i) * 3;
		constexpr float kColorScale = 1.0f / 255.0f;
		return make_float3(px[0] * kColorScale, px[1] * kColorScale, px[2] * kColorScale);
	} else if (tex.kind == TextureKind::Checker) {
		// Matches checker_texture::value() (texture.h:53-61) exactly: floor
		// each world-space coordinate scaled by 1/scale, sum the three
		// integers, and pick a color by parity - equality-to-zero on `%2`
		// is sign-agnostic, so this is correct for negative coordinates too
		// (just like the CPU version, which relies on the same C++ rule).
		const int xi = static_cast<int>(floorf(tex.noiseScale * p.x));
		const int yi = static_cast<int>(floorf(tex.noiseScale * p.y));
		const int zi = static_cast<int>(floorf(tex.noiseScale * p.z));
		const bool is_even = ((xi + yi + zi) % 2) == 0;
		return is_even ? tex.color1 : tex.color2;
	} else {
		// Matches noise_texture::value() (texture.h:127-129) exactly:
		// color(.5,.5,.5) * (1 + sin(scale*p.z + 10*turb(p,7))), where
		// turb(p,7) is perlin::turb's own default (depth=7, omega=0.5,
		// perlin.h:37) delegating to turbulence_simple<T> (noise.h:252).
		const float turb = turbulence_simple<float>(p.x, p.y, p.z, 0.5f, 7);
		const float s = 0.5f * (1.0f + sinf(tex.noiseScale * p.z + 10.0f * turb));
		return make_float3(s, s, s);
	}
}

// Alpha-cutout test (OBJ/.mtl map_d): returns true if the hit should be
// kept, false if it should be treated as fully transparent - a caller at an
// any-hit call site (radiance or shadow) should then call
// optixIgnoreIntersection() so the ray continues past this point as if the
// geometry weren't there. A no-op (always true) for alphaMaskTexIdx < 0,
// i.e. the overwhelming majority of materials. Samples the mask's red
// channel only against a fixed threshold - real alpha-cutout masks are
// strongly bimodal (opaque/transparent), so a fixed threshold is the
// standard, simple choice, matching CPU's own kAlphaCutoutThreshold
// (triangle.h).
__device__ __forceinline__ bool passes_alpha_cutout(int alphaMaskTexIdx, float u, float v, const float3& p) {
	if (alphaMaskTexIdx < 0) return true;
	constexpr float kAlphaCutoutThreshold = 0.5f;
	return sample_texture(alphaMaskTexIdx, u, v, p).x >= kAlphaCutoutThreshold;
}

// pbrt-v4 NormalizedFresnelBxDF (MaterialType::NormalizedFresnel's own
// logic, factored out to a standalone function so MaterialType::
// Subsurface's probe-walk exit point (bssrdf_probe_walk() below /
// shade_material()'s own Subsurface case) can shade with it too WITHOUT
// calling shade_material() itself a second time - OptiX's module compiler
// statically rejects any self-recursive call graph outright ("COMPILE
// ERROR: Malformed input... Found call graph recursion involving
// shade_material") even when the actual runtime call depth is providably
// bounded to one extra level (a different mat.type reaching a different
// switch case) - the module simply fails to compile at OptiX pipeline
// creation time, not a soft warning. This function is called from BOTH
// shade_material()'s own NormalizedFresnel case (unchanged behavior) and
// its Subsurface case (the new caller), so there is exactly one
// implementation of pbrt-v4's Sw exit BSDF, just no longer reached via a
// recursive shade_material() call.
//
// Always scatters (NormalizedFresnel has no failure/absorption case), so
// there is no out_scattered - unlike shade_material() itself.
__device__ __forceinline__ void shade_normalized_fresnel(
	float eta,
	const float3& normal,
	const float3& hit_point,
	unsigned int& seed,
	float3& out_attenuation,
	float3& out_scattered_dir,
	float& out_brdf_pdf_override,
	float3& emission)
{
	float nf_eta = eta;
	float inv_eta = 1.0f / nf_eta;
	float nf_c = 1.0f - 2.0f * FresnelMoment1(inv_eta);
	if (nf_c <= 0.0f) nf_c = 1e-6f;

	float3 scattered_dir = normalize(normal + random_unit_vector(seed));
	if (near_zero(scattered_dir)) scattered_dir = normal;

	float cos_wi = fmaxf(dot(scattered_dir, normal), 1e-6f);
	float fr     = FrDielectric(cos_wi, nf_eta);
	float weight = (1.0f - fr) / nf_c;
	float3 attenuation = make_float3(weight, weight, weight);
	// p12: correct BSDF PDF for MIS on next bounce: (1-Fr)*cos/(c*pi)
	float brdf_pdf_override = (1.0f - fr) * cos_wi / (nf_c * 3.14159265358979323846f);

	if (params.numLights > 0) {
		int light_idx;
		float selection_pdf;
		if (params.aliasTable) {
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
		float max_dist = 0.0f;
		float3 light_emission = make_float3(0.0f, 0.0f, 0.0f);
		float3 to_light = sample_area_light_by_kind(
			light_idx, hit_point, seed, geom_pdf, max_dist, light_emission);

		float light_pdf = selection_pdf * geom_pdf;
		if (light_pdf > 1e-6f) {
			bool visible = trace_shadow_ray(hit_point, to_light, max_dist);
			if (visible) {
				float cos_to_light = fmaxf(dot(to_light, normal), 0.0f);
				if (cos_to_light > 0.0f) {
					float fr_l  = FrDielectric(cos_to_light, nf_eta);
					float brdf_val = (1.0f - fr_l) / (nf_c * 3.14159265358979323846f);
					float brdf_pdf_l = brdf_val * cos_to_light;
					float mis_weight = mis_power_heuristic(light_pdf, brdf_pdf_l);

					float3 direct_light = mis_weight * brdf_val * light_emission * cos_to_light / light_pdf;
					emission = emission + direct_light;
				}
			}
		}
	}

	{
		const float3& skyColor = params.camera.backgroundColor;
		if (skyColor.x > 0.0f || skyColor.y > 0.0f || skyColor.z > 0.0f) {
			// sample_sky_nee() (optix_sky_light.h) - see shade_material()'s
			// own Lambertian sky-NEE block for the full comment.
			float3 sky_dir, sky_Le_val; float pdf_sky;
			sample_sky_nee(seed, skyColor, sky_dir, pdf_sky, sky_Le_val);
			float  cos_sky = dot(sky_dir, normal);
			if (cos_sky > 0.0f) {
				if (trace_shadow_ray(hit_point, sky_dir, 1e30f)) {
					float fr_sky       = FrDielectric(cos_sky, nf_eta);
					float brdf_val_sky = (1.0f - fr_sky) / (nf_c * 3.14159265358979323846f);
					float brdf_pdf_sky = brdf_val_sky * cos_sky;
					float mis_weight    = mis_power_heuristic(pdf_sky, brdf_pdf_sky);
					emission = emission + mis_weight * brdf_val_sky * sky_Le_val * cos_sky / pdf_sky;
				}
			}
		}
	}

	out_attenuation = attenuation;
	out_scattered_dir = scattered_dir;
	out_brdf_pdf_override = brdf_pdf_override;
}

// MaterialType::Subsurface's probe/exit-point search - the GPU port of
// src/TheRestOfYourLife/camera.h::sample_bssrdf_exit(), replicating its
// exact 3-axis MIS algorithm (see that function's own extensive comment for
// the full derivation): pick an RGB channel uniformly for importance-
// sampling the radius only, sample r/r_max from that channel's profile,
// pick one of 3 orthonormal probe axes (shading normal, weight 0.5; two
// tangents, weight 0.25 each), build a probe segment (disc-sampled point +-
// half_len along the chosen axis), walk the segment via repeated
// trace_probe_ray() calls collecting same-material candidates via
// unweighted reservoir sampling (Algorithm R - pbrt-v4's GPU
// __raygen__randomHit does the identical bounded loop of sequential
// optixTrace() calls, see this codebase's own research notes on that
// function), then compute Sp (evaluated at the full 3D entry-to-exit
// distance) and the combined (one-sample MIS, balance heuristic) pdf summed
// over all 3 axes exactly as camera.h does.
//
// `matIdx` is this material's own index into params.materials[] (shade_
// material() doesn't otherwise know it - see its own new parameter) - the
// GPU equivalent of camera.h's `rec.mat.get()` pointer-identity check
// against a probe candidate's own material.
//
// bounded to kMaxProbeSteps segment-walk iterations (matches pbrt-v4's GPU
// __raygen__randomHit's own 100-iteration cap) so a pathological segment
// (e.g. many thin, closely-stacked same-material faces) cannot turn into an
// unbounded device loop.
__device__ __forceinline__ bool bssrdf_probe_walk(
	const MaterialData& mat, int matIdx,
	const float3& p0, const float3& axis0,
	unsigned int& seed,
	float3& out_exit_pos, float3& out_exit_normal,
	float3& out_Sp, float& out_pdf, float& out_sample_prob)
{
	constexpr int kMaxProbeSteps = 100;
	const float kPi = 3.14159265358979323846f;

	const GpuBssrdfTable& table = params.bssrdfTables[mat.textureIdx];
	const float sigma_a[3] = { mat.bssrdf_sigma_a.x, mat.bssrdf_sigma_a.y, mat.bssrdf_sigma_a.z };
	const float sigma_s[3] = { mat.bssrdf_sigma_s.x, mat.bssrdf_sigma_s.y, mat.bssrdf_sigma_s.z };
	float sigma_t[3], rho[3];
	for (int c = 0; c < 3; ++c) {
		sigma_t[c] = sigma_a[c] + sigma_s[c];
		rho[c] = (sigma_t[c] > 0.0f) ? (sigma_s[c] / sigma_t[c]) : 0.0f;
	}

	const float3 axis = normalize(axis0);
	const float3 t1 = (fabsf(axis.x) > 0.9f) ? normalize(cross(make_float3(0, 1, 0), axis))
											  : normalize(cross(make_float3(1, 0, 0), axis));
	const float3 t2 = cross(axis, t1);

	const int channel = min(2, (int)(random_float(seed) * 3.0f));
	const float r = gpu_bssrdf_sample_sr(table, sigma_t[channel], rho[channel], random_float(seed));
	if (r < 0.0f) return false;
	const float r_max = gpu_bssrdf_sample_sr(table, sigma_t[channel], rho[channel], 0.999f);
	if (r_max <= 0.0f || r >= r_max) return false;

	const float phi = 2.0f * kPi * random_float(seed);
	const float half_len = sqrtf(fmaxf(0.0f, r_max * r_max - r * r));

	const float u_axis = random_float(seed);
	float3 probe_axis, basis_a, basis_b;
	if (u_axis < 0.5f)       { probe_axis = axis; basis_a = t1;   basis_b = t2; }
	else if (u_axis < 0.75f) { probe_axis = t1;   basis_a = t2;   basis_b = axis; }
	else                     { probe_axis = t2;   basis_a = axis; basis_b = t1; }

	const float3 p_target = p0 + r * (cosf(phi) * basis_a + sinf(phi) * basis_b);
	const float3 p_start  = p_target - half_len * probe_axis;
	const float3 p_end    = p_target + half_len * probe_axis;

	float3 seg_dir = p_end - p_start;
	float seg_len = length(seg_dir);
	if (seg_len < 1e-10f) return false;
	seg_dir = seg_dir / seg_len;

	float3 chosen_pos = make_float3(0.0f, 0.0f, 0.0f);
	float3 chosen_normal = make_float3(0.0f, 0.0f, 0.0f);
	int candidate_count = 0;
	float3 base = p_start;
	float remaining = seg_len;
	for (int iter = 0; iter < kMaxProbeSteps && remaining > 1e-6f; ++iter) {
		ProbeHit hit = trace_probe_ray(base, seg_dir, remaining);
		if (!hit.found) break;
		const float t_local = length(hit.position - base);
		if (hit.materialIdx == matIdx) {
			++candidate_count;
			if (random_float(seed) < 1.0f / (float)candidate_count) {
				chosen_pos = hit.position;
				chosen_normal = hit.normal;
			}
		}
		const float step = t_local + 1e-4f;
		base = base + step * seg_dir;
		remaining -= step;
	}
	if (candidate_count == 0) return false;

	const float sample_prob = 1.0f / (float)candidate_count;

	const float3 d = chosen_pos - p0;
	const float dist = length(d);
	const float3 Sp = make_float3(
		gpu_bssrdf_sr(table, sigma_t[0], rho[0], dist),
		gpu_bssrdf_sr(table, sigma_t[1], rho[1], dist),
		gpu_bssrdf_sr(table, sigma_t[2], rho[2], dist));

	const float3 exit_n = normalize(chosen_normal);
	const float d_n  = dot(d, axis);
	const float d_t1 = dot(d, t1);
	const float d_t2 = dot(d, t2);
	const float r_proj_axis = sqrtf(fmaxf(0.0f, d_t1 * d_t1 + d_t2 * d_t2));
	const float r_proj_t1   = sqrtf(fmaxf(0.0f, d_t2 * d_t2 + d_n  * d_n));
	const float r_proj_t2   = sqrtf(fmaxf(0.0f, d_n  * d_n  + d_t1 * d_t1));
	const float cos_axis = fabsf(dot(exit_n, axis));
	const float cos_t1   = fabsf(dot(exit_n, t1));
	const float cos_t2   = fabsf(dot(exit_n, t2));
	constexpr float kAxisProb = 0.5f, kTangentProb = 0.25f;
	float pdf = 0.0f;
	for (int c = 0; c < 3; ++c) {
		pdf += kAxisProb   * gpu_bssrdf_pdf_sr(table, sigma_t[c], rho[c], r_proj_axis) * cos_axis
			 + kTangentProb * gpu_bssrdf_pdf_sr(table, sigma_t[c], rho[c], r_proj_t1)   * cos_t1
			 + kTangentProb * gpu_bssrdf_pdf_sr(table, sigma_t[c], rho[c], r_proj_t2)   * cos_t2;
	}
	pdf /= 3.0f;
	if (pdf <= 0.0f) return false;

	out_exit_pos = chosen_pos;
	out_exit_normal = exit_n;
	out_Sp = Sp;
	out_pdf = pdf;
	out_sample_prob = sample_prob;
	return true;
}

// Evaluates material scattering for every MaterialType except Medium and
// Hair, which are sphere-only and stay in optix_intersection_sphere.h's own
// closest-hit program (they need shape-specific re-intersection/geometry
// data - a medium's exit distance, a hair fiber's curve parameterization -
// that no other primitive type has). Identical scattering/NEE/MIS logic
// regardless of which primitive was actually hit; only `normal`,
// `hit_point`, `front_face`, and `uv` differ per caller, and those are
// passed in rather than recomputed here. Leaves out_scattered false (every
// other output untouched) for DiffuseLight/default, exactly matching each
// call site's prior inline default: case - `emission` is the sole in/out
// parameter (NEE contributions from Lambertian/CoatedDiffuse/
// NormalizedFresnel get added directly into whatever the caller already
// initialized it to, i.e. mat.emission). `uv` is only meaningful for
// Lambertian materials with textureIdx >= 0 (scene 8's Earth/noise
// spheres) - every other caller/material can pass (0,0).
// `matIdx` (this material's own index into params.materials[]) is only used
// by MaterialType::Subsurface's probe walk (identifying "same material"
// candidates - see bssrdf_probe_walk()'s own comment); every other case
// ignores it. `out_bssrdf_exit`/`out_bssrdf_exit_pos` are only ever set true
// by that same case, signalling to the caller (each geometry type's own
// closest-hit program) that the NEXT ray's origin must be this explicit
// world-space position - found via an off-ray probe walk - rather than the
// usual `hit_point + t_hit*ray_dir` reconstruction every other material
// (including the Medium family's own t-along-the-original-ray override)
// relies on. See optix_raygen.h's flag==3 handling.
__device__ __forceinline__ void shade_material(
	const MaterialData& mat,
	int matIdx,
	const float3& normal,
	const float3& ray_dir,
	const float3& hit_point,
	bool front_face,
	float uv_u, float uv_v,
	unsigned int& seed,
	float3& out_attenuation,
	float3& out_scattered_dir,
	bool& out_scattered,
	bool& out_is_specular,
	float& out_brdf_pdf_override,
	float3& emission,
	bool& out_bssrdf_exit,
	float3& out_bssrdf_exit_pos
) {
	float3 attenuation;
	float3 scattered_dir;
	bool scattered = false;
	bool is_specular = false;  // pbrt-v4 specularBounce: MIS is skipped for specular events
	float brdf_pdf_override = -1.0f;  // if >= 0, overrides cosine_pdf in payload packing
	bool bssrdf_exit = false;
	float3 bssrdf_exit_pos = make_float3(0.0f, 0.0f, 0.0f);

	switch (mat.type) {
		case MaterialType::Lambertian: {
			// Multiple Importance Sampling (MIS) for diffuse surfaces

			// Sample BRDF (cosine-weighted hemisphere) for indirect lighting
			scattered_dir = normal + random_unit_vector(seed);
			if (near_zero(scattered_dir)) {
				scattered_dir = normal;
			}
			scattered_dir = normalize(scattered_dir);
			attenuation = (mat.textureIdx >= 0) ? sample_texture(mat.textureIdx, uv_u, uv_v, hit_point) : mat.albedo;
			scattered = true;

			// Add direct lighting via explicit light sampling (Next Event Estimation)
			if (params.numLights > 0) {
				// Power-weighted light selection via alias table (pbrt-v4 PowerLightSampler pattern)
				int light_idx;
				float selection_pdf;
				if (params.aliasTable) {
					// O(1) alias-table sample: pick slot, accept or redirect
					int slot = int(random_float(seed) * float(params.numLights));
					if (slot >= int(params.numLights)) slot = int(params.numLights) - 1;
					const GpuAliasEntry& entry = params.aliasTable[slot];
					light_idx = (random_float(seed) < entry.q) ? slot : entry.alias;
					selection_pdf = params.aliasTable[light_idx].pdf;
				} else {
					// Fallback: uniform selection
					light_idx = int(random_float(seed) * float(params.numLights));
					if (light_idx >= int(params.numLights)) light_idx = int(params.numLights) - 1;
					selection_pdf = 1.0f / float(params.numLights);
				}

				// Sample direction toward light
				float geom_pdf = 0.0f;  // PDF of the sampled direction (geometry term)
				float max_dist = 0.0f;
				float3 sampled_light_emission = make_float3(0.0f, 0.0f, 0.0f);
				float3 to_light = sample_area_light_by_kind(
					light_idx, hit_point, seed, geom_pdf, max_dist, sampled_light_emission);

				// Combined PDF = selection_pdf * geometric_pdf
				float light_pdf = selection_pdf * geom_pdf;

				if (light_pdf > 1e-6f) {
					// Check if light is visible (shadow ray)
					bool visible = trace_shadow_ray(hit_point, to_light, max_dist);

					if (visible) {
						// Evaluate BRDF PDF for this direction
						float brdf_pdf = cosine_pdf(to_light, normal);

						// MIS weight using power heuristic
						float mis_weight = mis_power_heuristic(light_pdf, brdf_pdf);

						// Emission of the light that was actually sampled -
						// looked up by the same helper that chose it, so the
						// two can't disagree about which primitive it was.
						const float3 light_emission = sampled_light_emission;

						// L = BRDF * emission * cos(theta) * MIS_weight / pdf
						// Uses `attenuation` (already texture-sampled above if
						// mat.textureIdx >= 0), NOT mat.albedo directly - for a
						// textured Lambertian (e.g. scenes 38-40's checkered
						// ground), mat.albedo is just a (1,1,1) white
						// placeholder (see line 611's own comment), so using it
						// here made every NEE-lit textured surface's direct
						// lighting term use full white reflectance regardless
						// of its actual (possibly much darker) texture color -
						// confirmed via a real, sample-count-independent ~1.5x
						// CPU/GPU brightness mismatch on scene 40's checkered
						// ground (a hard bias, not noise: identical ratio at
						// 100 spp and 2000 spp) that CPU didn't have (its own
						// NEE path already samples the checker texture, see
						// camera.h's `srec.attenuation`). Indirect/BSDF-sampled
						// bounces were never affected - they already read
						// `attenuation`, not mat.albedo.
						float cos_theta = fmaxf(0.0f, dot(to_light, normal));
						float3 brdf = attenuation / 3.14159265358979323846f;  // Lambertian BRDF
						float3 direct_light = mis_weight * brdf * light_emission * cos_theta / light_pdf;

						// Add to emission (raygen will apply throughput)
						emission = emission + direct_light;
					}
				}
			}

			// Direct lighting from punctual (point/spot/distant) lights -
			// deterministic delta lights, evaluated separately from the
			// area-light alias table above (see optix_device_helpers.h).
			// Same fix as the NEE term above: pass the already texture-
			// sampled `attenuation`, not the (1,1,1)-placeholder mat.albedo,
			// so a textured Lambertian under a punctual light isn't
			// incorrectly lit as if fully white.
			add_punctual_lights_lambertian(hit_point, normal, attenuation, emission);

			// Direct lighting from the sky (infinite light) - mirrors CPU's
			// camera.h Strategy A-2 (NEE toward the sky, MIS-weighted against
			// the BSDF-sampled bounce), which the GPU miss program
			// (optix_miss.h) never had: it only adds the background color when
			// a bounced ray happens to escape the scene entirely, with no
			// importance sampling toward it - fine for open scenes, but
			// starves interiors with small apertures (confirmed: Sibenik
			// Cathedral rendered 2.85x darker than CPU at matched settings
			// before this fix). sample_sky_nee() (optix_sky_light.h)
			// dispatches to real HDR importance sampling when the scene's
			// infinite light carries an image (params.camera.skyDist.height >
			// 0), falling back to this exact uniform-sphere sample otherwise -
			// same shape as sky_light::sample_Le()'s own two modes. Skipped
			// outright when backgroundColor is black (every scene without a
			// sky) - free, since NEE toward a zero-radiance light contributes
			// nothing.
			{
				const float3& skyColor = params.camera.backgroundColor;
				if (skyColor.x > 0.0f || skyColor.y > 0.0f || skyColor.z > 0.0f) {
					float3 sky_dir, sky_Le_val; float pdf_sky;
					sample_sky_nee(seed, skyColor, sky_dir, pdf_sky, sky_Le_val);
					float  cos_sky = dot(sky_dir, normal);
					if (cos_sky > 0.0f) {
						if (trace_shadow_ray(hit_point, sky_dir, 1e30f)) {
							float brdf_pdf_sky = cosine_pdf(sky_dir, normal);
							float mis_weight    = mis_power_heuristic(pdf_sky, brdf_pdf_sky);
							float3 brdf = attenuation / 3.14159265358979323846f;
							emission = emission + mis_weight * brdf * sky_Le_val * cos_sky / pdf_sky;
						}
					}
				}
			}

			break;
		}

		case MaterialType::Metal: {
			float3 reflected = reflect(normalize(ray_dir), normal);
			scattered_dir = normalize(reflected) + mat.fuzz * random_in_unit_sphere(seed);
			// Clamp to hemisphere: if fuzz pushes below surface, use pure reflection
			if (dot(scattered_dir, normal) <= 0.0f)
				scattered_dir = reflected;
			attenuation = mat.albedo;
			scattered = true;
			is_specular = true;  // specular bounce: next hit adds full emission, no MIS
			break;
		}

		case MaterialType::Measured: {
			// Real tabulated measured-BRDF (pbrt-v4 Material "measured") -
			// a single VNDF-importance-sampled BxDF sample per hit, same
			// *shape* as Metal above (sample once, get a direction + full
			// throughput weight, continue as a non-NEE specular-style
			// bounce) - see sample_measured_material() (optix_measured_
			// bxdf.h) and MaterialType::Measured's own comment in
			// optix_types.h. Falls back to absorption (scattered=false,
			// matching CPU's `if (!ok) return false;`) on any of
			// sample_f()'s own rejection paths (grazing wo, zero pdf, a
			// reflected half-vector below the horizon) or an invalid table
			// index.
			float3 sdir, atten;
			if (sample_measured_material(ray_dir, normal, mat, seed, sdir, atten)) {
				scattered_dir = sdir;
				attenuation   = atten;
				scattered     = true;
			} else {
				scattered = false;
			}
			is_specular = true;
			break;
		}

		case MaterialType::Dielectric: {
			scattered_dir = dielectric_scatter(ray_dir, normal, front_face, mat.ior, seed);
			// `normal` here always satisfies dot(ray_dir, normal) < 0 (flipped
			// to face the incoming ray - see the front_face/final_normal
			// convention at each closesthit's call site). A reflected
			// direction bounces back to the ray_dir side (dot > 0); a
			// refracted direction continues through to the far side
			// (dot < 0, same sign as ray_dir's own). Tf tints transmission
			// only, matching real colored glass (a mirror-like reflection
			// off the surface doesn't pick up the pane's tint) - see
			// add_dielectric()'s transmission_filter comment.
			bool is_transmission = dot(scattered_dir, normal) < 0.0f;
			attenuation = is_transmission ? mat.transmission_filter : make_float3(1.0f, 1.0f, 1.0f);
			scattered = true;
			is_specular = true;  // specular bounce: next hit adds full emission, no MIS
			break;
		}

		case MaterialType::Subsurface: {
			// Real tabulated BSSRDF (recursive backend only, Phase 1 - see
			// this MaterialType's own comment in optix_types.h and
			// bssrdf_probe_walk()'s comment above for the full algorithm).
			// Entry interface: the exact same smooth DielectricBxDF sample
			// as MaterialType::Dielectric above (matches pbrt-v4's own
			// SubsurfaceMaterial::GetBxDF / this codebase's CPU `class
			// subsurface`, material_pbrt.h - a plain DielectricBxDF(eta),
			// no roughness, no Tf tint).
			//
			// `emission` on entry is whatever the caller initialized it to
			// (mat.emission, per shade_material()'s own general convention)
			// - but Subsurface has no real emission field: that union slot
			// is bssrdf_sigma_a here (see optix_types.h's MaterialType::
			// Subsurface comment), so the caller's mat.emission read is
			// actually sigma_a's bytes reinterpreted as a color. Reset to a
			// clean zero before anything below might add to it.
			emission = make_float3(0.0f, 0.0f, 0.0f);
			scattered_dir = dielectric_scatter(ray_dir, normal, front_face, mat.ior, seed);
			bool is_transmission = dot(scattered_dir, normal) < 0.0f;
			if (!is_transmission) {
				// Specular reflection off the entry interface - identical
				// to Dielectric's own reflection case.
				attenuation = make_float3(1.0f, 1.0f, 1.0f);
				scattered   = true;
				is_specular = true;
				break;
			}

			// Transmission: attempt the probe/exit-point search. On
			// failure the path terminates with zero contribution, matching
			// camera.h's `if (!sample_bssrdf_exit(...)) break;`.
			float3 exit_pos, exit_normal, Sp;
			float pdf, sample_prob;
			if (!bssrdf_probe_walk(mat, matIdx, hit_point, normal, seed,
									exit_pos, exit_normal, Sp, pdf, sample_prob)) {
				scattered = false;
				break;
			}
			const float3 sss_weight = Sp / (sample_prob * pdf);

			// Shade the exit point as pbrt-v4's Sw (NormalizedFresnel(eta))
			// via shade_normalized_fresnel() - the same standalone function
			// MaterialType::NormalizedFresnel's own case calls, NOT a
			// nested shade_material() call: OptiX's module compiler
			// statically rejects any self-recursive call graph (see that
			// function's own comment for the exact compile error), even
			// though the actual call depth here would have been provably
			// bounded to one extra level. This still exactly mirrors
			// camera.h's own hand-off design (rec.mat swapped to
			// ss->get_exit_bsdf(), then that material's own scatter()
			// called once) - just with the shared logic factored into its
			// own function instead of reached by re-entering the switch.
			// shade_normalized_fresnel() always scatters (no failure case).
			float3 exit_atten, exit_dir;
			float exit_pdf_override;
			shade_normalized_fresnel(mat.ior, exit_normal, exit_pos, seed,
				exit_atten, exit_dir, exit_pdf_override, emission);

			// emission was just accumulated (in place) by that call's
			// own NEE terms, evaluated at exit_pos with the OLD (pre-this-
			// bounce) throughput - but those terms didn't know about
			// sss_weight, so scale the just-added contribution by it,
			// leaving whatever emission already held (mat.emission, always
			// zero for Subsurface) untouched. Mirrors camera.h's own
			// ordering: sample_bssrdf_exit() multiplies `beta` by
			// entry_attenuation*Sp*inv BEFORE the exit bounce's own NEE
			// (inside ray_color's main loop) ever runs, so that NEE already
			// sees the weighted throughput - here, where entry+exit shading
			// happen in one call, the weighting has to be applied after
			// the fact instead.
			emission = sss_weight * emission;

			attenuation    = sss_weight * exit_atten;
			scattered_dir  = exit_dir;
			scattered      = true;
			is_specular    = false;  // NormalizedFresnel participates in MIS
			brdf_pdf_override = exit_pdf_override;

			// The next ray must originate at exit_pos (found off to the
			// side via the probe walk), not along this hit's own ray - see
			// shade_material()'s own comment on out_bssrdf_exit/
			// out_bssrdf_exit_pos and optix_raygen.h's flag==3 handling.
			bssrdf_exit     = true;
			bssrdf_exit_pos = exit_pos;
			break;
		}

		case MaterialType::ThinDielectric: {
			// Zero-thickness glass slab (pbrt-v4 ThinDielectricBxDF) -- sphere version
			// Transmission goes straight through (no bending); reflection is specular.
			// Multiple internal bounces folded analytically: R_eff = R + T^2*R/(1-R^2)
			float3 unit_direction = normalize(ray_dir);
			float cos_theta = fabsf(dot(unit_direction, normal));
			float R = FrDielectric(cos_theta, mat.ior);
			if (R < 1.0f) {
				float T = 1.0f - R;
				R += T * T * R / (1.0f - R * R);
			}
			attenuation = make_float3(1.0f, 1.0f, 1.0f);
			if (random_float(seed) < R) {
				scattered_dir = reflect(unit_direction, normal);
			} else {
				scattered_dir = unit_direction;  // straight through
			}
			scattered   = true;
			is_specular = true;
			break;
		}

		case MaterialType::CoatedConductor: {
			// Rough dielectric coat over GGX conductor (pbrt-v4 CoatedConductorBxDF) -- sphere version
			// coat: ior=mat.ior, roughness=mat.fuzz; conductor: eta_c, k_c per RGB
			float cc_alpha = sqrtf(mat.fuzz);
			float3 cc_n   = normal;
			float3 cc_up  = (fabsf(cc_n.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
			float3 cc_tan  = normalize(cross(cc_up, cc_n));
			float3 cc_bit  = cross(cc_n, cc_tan);

			float3 cc_wi_w = normalize(-ray_dir);
			float cc_wi_x = dot(cc_wi_w, cc_tan);
			float cc_wi_y = dot(cc_wi_w, cc_bit);
			float cc_wi_z = dot(cc_wi_w, cc_n);
			if (cc_wi_z <= 0.0f) { scattered = false; break; }

			TrowbridgeReitz<float> cc_dist(cc_alpha, cc_alpha);

			// Coat top interface: GGX VNDF + FrDielectric
			float cwm_x, cwm_y, cwm_z;
			cc_dist.Sample_wm(cc_wi_x, cc_wi_y, cc_wi_z,
							  random_float(seed), random_float(seed),
							  cwm_x, cwm_y, cwm_z);
			float cc_cos_i = cc_wi_x*cwm_x + cc_wi_y*cwm_y + cc_wi_z*cwm_z;
			float F_in     = FrDielectric(cc_cos_i, mat.ior);

			float3 cc_wo;
			if (random_float(seed) < F_in) {
				// Path A: coat specular reflection
				float wo_x = 2.0f*cc_cos_i*cwm_x - cc_wi_x;
				float wo_y = 2.0f*cc_cos_i*cwm_y - cc_wi_y;
				float wo_z = 2.0f*cc_cos_i*cwm_z - cc_wi_z;
				if (wo_z <= 0.0f) { scattered = false; break; }
				float G1 = cc_dist.G1(cc_wi_x, cc_wi_y, cc_wi_z);
				float G  = cc_dist.G(wo_x, wo_y, wo_z, cc_wi_x, cc_wi_y, cc_wi_z);
				float w  = (G1 > 1e-8f) ? G / G1 : 0.0f;
				float fv = F_in * w;
				attenuation = make_float3(fv, fv, fv);
				cc_wo = make_float3(wo_x, wo_y, wo_z);
			} else {
					// Path B: transmit into layer -> conductor bounce -> exit coat
					// Step 1: transmitted direction inside layer.
					// The coat microfacet cwm was already sampled above; the reflected direction
					// off cwm is (2*cc_cos_i*cwm - cc_wi). Inside the slab the ray goes downward,
					// so we flip the z component.  (Matches CPU CoatedConductorBxDF step 1.)
					float w_x = 2.0f*cc_cos_i*cwm_x - cc_wi_x;
					float w_y = 2.0f*cc_cos_i*cwm_y - cc_wi_y;
					float w_z = 2.0f*cc_cos_i*cwm_z - cc_wi_z;
					if (w_z > 0.0f) w_z = -w_z;   // ensure pointing downward into layer
					if (w_z == 0.0f) { scattered = false; break; }

					// Step 2: flip to conductor frame -- "incoming from above" (fw_z > 0)
					float fw_x = -w_x, fw_y = -w_y, fw_z = -w_z;

					// Sample conductor microfacet from the correct transmitted direction
					float bwm_x, bwm_y, bwm_z;
					cc_dist.Sample_wm(fw_x, fw_y, fw_z,
									  random_float(seed), random_float(seed),
									  bwm_x, bwm_y, bwm_z);
					float cos_c = fw_x*bwm_x + fw_y*bwm_y + fw_z*bwm_z;
					if (cos_c <= 0.0f) { scattered = false; break; }

					// Conductor reflection direction (upward, rwo_z > 0 = exits toward coat top)
					float rwo_x = 2.0f*cos_c*bwm_x - fw_x;
					float rwo_y = 2.0f*cos_c*bwm_y - fw_y;
					float rwo_z = 2.0f*cos_c*bwm_z - fw_z;
					if (rwo_z <= 0.0f) { scattered = false; break; }

					float G1_c = cc_dist.G1(fw_x, fw_y, fw_z);
					float G_c  = cc_dist.G(rwo_x, rwo_y, rwo_z, fw_x, fw_y, fw_z);
					float wt_c = (G1_c > 1e-8f) ? G_c / G1_c : 0.0f;

					float F_r = FrComplex(cos_c, mat.eta_c.x, mat.k_c.x) * wt_c;
					float F_g = FrComplex(cos_c, mat.eta_c.y, mat.k_c.y) * wt_c;
					float F_b = FrComplex(cos_c, mat.eta_c.z, mat.k_c.z) * wt_c;

					// Step 3: exit through coat top � Fresnel at exit angle (rwo_z = cos of exit)
					float F_out = FrDielectric(rwo_z, 1.0f / mat.ior);  // inside->outside
					float T_out = 1.0f - F_out;
					float T_in  = 1.0f - F_in;

					attenuation = make_float3(F_r * T_in * T_out,
											 F_g * T_in * T_out,
											 F_b * T_in * T_out);
					cc_wo = make_float3(rwo_x, rwo_y, rwo_z);
				}
			scattered_dir = normalize(cc_wo.x*cc_tan + cc_wo.y*cc_bit + cc_wo.z*cc_n);
			scattered   = true;
			is_specular = true;
			break;
		}

		case MaterialType::RoughDielectric: {
			// GGX microfacet BSDF (pbrt-v4 RoughDielectricBxDF)
			// fuzz field stores GGX roughness; ior = index of refraction
			float rd_alpha = mat.fuzz;
			rd_alpha = sqrtf(rd_alpha);  // RoughnessToAlpha: alpha = sqrt(roughness)
			float rd_ri    = front_face ? (1.0f / mat.ior) : mat.ior;

			// Local shading frame (n = +Z)
			float3 n = normal;
			float3 up_v = (fabsf(n.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
			float3 tan  = normalize(cross(up_v, n));
			float3 bitan = cross(n, tan);

			float3 wi_w = normalize(-ray_dir);
			float wi_x = dot(wi_w, tan), wi_y = dot(wi_w, bitan), wi_z = dot(wi_w, n);
			if (wi_z < 0.0f) { wi_z=-wi_z; wi_x=-wi_x; wi_y=-wi_y; }

			TrowbridgeReitz<float> rd_dist(rd_alpha, rd_alpha);
			float wm_x, wm_y, wm_z;
			rd_dist.Sample_wm(wi_x, wi_y, wi_z,
							  random_float(seed), random_float(seed),
							  wm_x, wm_y, wm_z);

			float cos_i = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
			float F = FrDielectric(cos_i, 1.0f / rd_ri);

			float3 wo_local;
			if (random_float(seed) < F) {
				// Reflect
				float wo_x = 2.0f*cos_i*wm_x - wi_x;
				float wo_y = 2.0f*cos_i*wm_y - wi_y;
				float wo_z = 2.0f*cos_i*wm_z - wi_z;
				if (wo_z <= 0.0f) { scattered = false; break; }
				wo_local = make_float3(wo_x, wo_y, wo_z);
			} else {
				// Refract
				if (wm_z < 0.0f) { wm_x=-wm_x; wm_y=-wm_y; wm_z=-wm_z; }
				float sin2_t = rd_ri*rd_ri * (1.0f - cos_i*cos_i);
				if (sin2_t >= 1.0f) {
					// TIR: reflect
					float wo_x = 2.0f*cos_i*wm_x - wi_x;
					float wo_y = 2.0f*cos_i*wm_y - wi_y;
					float wo_z = 2.0f*cos_i*wm_z - wi_z;
					if (wo_z <= 0.0f) { scattered = false; break; }
					wo_local = make_float3(wo_x, wo_y, wo_z);
				} else {
					// Transmitted direction (pbrt-v4 Refract in local frame):
					//   wo = -eta*wi + (eta*dot(wi,wm) - cos_t)*wm
					// wo_z < 0: ray crosses through the surface boundary
					float cos_t = sqrtf(1.0f - sin2_t);
					float wo_x = rd_ri*(-wi_x) + (rd_ri*cos_i - cos_t)*wm_x;
					float wo_y = rd_ri*(-wi_y) + (rd_ri*cos_i - cos_t)*wm_y;
					float wo_z = -(rd_ri*wi_z  - (rd_ri*cos_i - cos_t)*wm_z);
					wo_local = make_float3(wo_x, wo_y, wo_z);
				}
			}
			scattered_dir = normalize(wo_local.x*tan + wo_local.y*bitan + wo_local.z*n);
			// pbrt-v4 RoughDielectricBxDF: BSDF weight with VNDF sampling = G(wo,wi)/G1(wi)
			// (the D, cos, and pdf terms cancel; only shadow-masking ratio remains)
			{
				float wo_x = wo_local.x, wo_y = wo_local.y, wo_z = fabsf(wo_local.z);
				float G2 = rd_dist.G(wi_x, wi_y, wi_z, wo_x, wo_y, wo_z);
				float G1_wi = rd_dist.G1(wi_x, wi_y, wi_z);
				float w = (G1_wi > 1e-8f) ? (G2 / G1_wi) : 0.0f;
				attenuation = make_float3(w, w, w);
			}
			scattered     = true;
			is_specular   = true;
			break;
		}

		case MaterialType::Conductor: {
			// GGX VNDF + complex Fresnel (pbrt-v4 ConductorBxDF) -- sphere version
			float c_alpha = sqrtf(mat.fuzz);
			float3 cn = normal;
			float3 cup = (fabsf(cn.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
			float3 ctan   = normalize(cross(cup, cn));
			float3 cbitan = cross(cn, ctan);
			float3 cwi = normalize(-ray_dir);
			float cwi_x = dot(cwi, ctan), cwi_y = dot(cwi, cbitan), cwi_z = dot(cwi, cn);
			if (cwi_z <= 0.0f) { scattered = false; break; }
			TrowbridgeReitz<float> c_dist(c_alpha, c_alpha);
			float cwm_x, cwm_y, cwm_z;
			c_dist.Sample_wm(cwi_x, cwi_y, cwi_z, random_float(seed), random_float(seed), cwm_x, cwm_y, cwm_z);
			float c_dot = cwi_x*cwm_x + cwi_y*cwm_y + cwi_z*cwm_z;
			float cwo_x = 2.0f*c_dot*cwm_x - cwi_x;
			float cwo_y = 2.0f*c_dot*cwm_y - cwi_y;
			float cwo_z = 2.0f*c_dot*cwm_z - cwi_z;
			if (cwo_z <= 0.0f) { scattered = false; break; }
			float c_G1_wi  = c_dist.G1(cwi_x, cwi_y, cwi_z);
			float c_G_wowi = c_dist.G(cwo_x, cwo_y, cwo_z, cwi_x, cwi_y, cwi_z);
			float c_weight = (c_G1_wi > 1e-8f) ? c_G_wowi / c_G1_wi : 0.0f;
			float3 c_F = FrConductorRGB(c_dot, mat.eta_c.x, mat.eta_c.y, mat.eta_c.z, mat.k_c.x, mat.k_c.y, mat.k_c.z);
			attenuation = make_float3(c_F.x * c_weight, c_F.y * c_weight, c_F.z * c_weight);
			scattered_dir = normalize(cwo_x*ctan + cwo_y*cbitan + cwo_z*cn);
			scattered     = true;
			is_specular   = true;
			break;
		}

		case MaterialType::CoatedDiffuse: {
			// Rough dielectric coat over Lambertian base (pbrt-v4 CoatedDiffuseBxDF) -- sphere version
			float cd_alpha = sqrtf(mat.fuzz);
			float3 cdn  = normal;
			float3 cdup = (fabsf(cdn.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
			float3 cdtan = normalize(cross(cdup, cdn));
			float3 cdbit = cross(cdn, cdtan);
			float3 cdwi  = normalize(-ray_dir);
			float cdwi_x = dot(cdwi, cdtan), cdwi_y = dot(cdwi, cdbit), cdwi_z = dot(cdwi, cdn);
			if (cdwi_z <= 0.0f) { scattered = false; break; }
			TrowbridgeReitz<float> cd_dist(cd_alpha, cd_alpha);
			float cdwm_x, cdwm_y, cdwm_z;
			cd_dist.Sample_wm(cdwi_x, cdwi_y, cdwi_z, random_float(seed), random_float(seed), cdwm_x, cdwm_y, cdwm_z);
			float cd_cosi = cdwi_x*cdwm_x + cdwi_y*cdwm_y + cdwi_z*cdwm_z;
			float F_in = FrDielectric(cd_cosi, mat.ior);
			if (random_float(seed) < F_in) {
				float cwo_x2 = 2.0f*cd_cosi*cdwm_x - cdwi_x;
				float cwo_y2 = 2.0f*cd_cosi*cdwm_y - cdwi_y;
				float cwo_z2 = 2.0f*cd_cosi*cdwm_z - cdwi_z;
				if (cwo_z2 <= 0.0f) { scattered = false; break; }
				float G1 = cd_dist.G1(cdwi_x, cdwi_y, cdwi_z);
				float G  = cd_dist.G(cwo_x2, cwo_y2, cwo_z2, cdwi_x, cdwi_y, cdwi_z);
				float wt = (G1 > 1e-8f) ? G / G1 : 0.0f;
				float fw = F_in * wt;
				attenuation   = make_float3(fw, fw, fw);
				scattered_dir = normalize(cwo_x2*cdtan + cwo_y2*cdbit + cwo_z2*cdn);
				scattered     = true;
				is_specular   = true;
			} else {
				// Multi-bounce escape through the coat, mirroring the actual
				// random walk in src/shared/bxdfs_layered.h's
				// CoatedDiffuseBxDF::sample_local (medium scattering omitted
				// - coateddiffuse never sets a scattering medium). A single
				// Lambertian bounce followed by one exit attempt is what the
				// earlier version of this branch did (with or without the
				// cd_c normalization above) and it stayed far too dark: most
				// individual cosine-sampled directions fail their exit test,
				// and giving up immediately on failure discards that
				// sample's energy entirely rather than retrying, which is
				// what the real layered material does - a failed exit
				// attempt scatters back into the diffuse base for another
				// Lambertian bounce and another try, compounding by albedo
				// each time, until one succeeds or the bounce budget runs
				// out. No explicit light sampling here, matching how CPU's
				// coated_diffuse material treats this whole material as
				// skip_pdf (material_pbrt.h): pbrt-v4's LayeredBxDF only
				// supports sampling a direction, never evaluating f(wo,wi)
				// at an arbitrary chosen one, so there is no light-sampling
				// direction to aim a shadow ray at - illumination comes
				// purely from where the bounces the walk below happens to
				// land, same as CPU.
				// Each exit attempt below re-samples a GGX microfacet normal
				// relative to the CURRENT internal direction (cd_dist.Sample_wm
				// again, exactly like the entry test above) rather than testing
				// Fresnel at the macro normal directly - matching
				// bxdfs_layered.h's own exit test. For a rough coat the two
				// disagree noticeably: the macro-normal angle can be steep even
				// when the GGX distribution still offers plenty of near-normal
				// microfacets to escape through, so testing only the macro
				// angle over-rejects and compounds too many extra albedo
				// multiplies from the retries that didn't need to happen.
				constexpr int kMaxCoatBounces = 8;
				float3 beta = make_float3(1.0f - F_in, 1.0f - F_in, 1.0f - F_in);
				float3 diff_dir = cdn;
				bool escaped = false;
				for (int cb = 0; cb < kMaxCoatBounces; ++cb) {
					diff_dir = cdn + random_unit_vector(seed);
					if (near_zero(diff_dir)) diff_dir = cdn;
					diff_dir = normalize(diff_dir);
					beta.x *= mat.albedo.x; beta.y *= mat.albedo.y; beta.z *= mat.albedo.z;

					float dw_x = dot(diff_dir, cdtan), dw_y = dot(diff_dir, cdbit), dw_z = dot(diff_dir, cdn);
					float dwm_x, dwm_y, dwm_z;
					cd_dist.Sample_wm(dw_x, dw_y, dw_z, random_float(seed), random_float(seed), dwm_x, dwm_y, dwm_z);
					float cos_out = dw_x*dwm_x + dw_y*dwm_y + dw_z*dwm_z;
					float F_out   = FrDielectric(cos_out, 1.0f / mat.ior);
					if (random_float(seed) < F_out) continue;  // TIR: bounce again
					beta.x *= (1.0f - F_out); beta.y *= (1.0f - F_out); beta.z *= (1.0f - F_out);
					escaped = true;
					break;
				}
				if (!escaped) { scattered = false; break; }
				attenuation   = beta;
				scattered_dir = diff_dir;
				scattered     = true;
				is_specular   = true;
			}
			break;
		}

		case MaterialType::DiffuseTransmission: {
			// pbrt-v4 DiffuseTransmissionBxDF -- sphere version
			// albedo = reflectance R (same hemisphere), emission = transmittance T (reused field)
			float3 R = mat.albedo;
			float3 T_col = mat.emission;  // transmittance packed into emission field
			float pr = fmaxf(R.x, fmaxf(R.y, R.z));
			float pt = fmaxf(T_col.x, fmaxf(T_col.y, T_col.z));
			if (pr + pt <= 0.0f) { scattered = false; break; }

			if (random_float(seed) < pr / (pr + pt)) {
				// Diffuse reflection: cosine-weighted same hemisphere
				scattered_dir = normalize(normal + random_unit_vector(seed));
				if (near_zero(scattered_dir)) scattered_dir = normal;
				attenuation   = R;
			} else {
				// Diffuse transmission: cosine-weighted opposite hemisphere
				float3 neg_n  = -normal;
				scattered_dir = normalize(neg_n + random_unit_vector(seed));
				if (near_zero(scattered_dir)) scattered_dir = neg_n;
				attenuation   = T_col;
			}
			scattered   = true;
			is_specular = false;
			break;
		}

		case MaterialType::NormalizedFresnel: {
			// pbrt-v4 NormalizedFresnelBxDF - see shade_normalized_fresnel()
			// above (factored out so MaterialType::Subsurface's exit point
			// can shade with the exact same code without a self-recursive
			// call into shade_material() itself, which OptiX's module
			// compiler statically rejects - see that function's own
			// comment). Diffuse BSDF: participates in MIS (is_specular=false).
			float3 nf_atten, nf_dir;
			float nf_pdf_override;
			shade_normalized_fresnel(mat.ior, normal, hit_point, seed,
				nf_atten, nf_dir, nf_pdf_override, emission);
			attenuation = nf_atten;
			scattered_dir = nf_dir;
			brdf_pdf_override = nf_pdf_override;
			scattered = true;
			is_specular = false;
			break;
		}
		case MaterialType::DiffuseLight: {
			// Emissive material - no scattering
			scattered = false;
			break;
		}

		default: {
			// Unknown material - absorb
			scattered = false;
			break;
		}
	}

	out_attenuation = attenuation;
	out_scattered_dir = scattered_dir;
	out_scattered = scattered;
	out_is_specular = is_specular;
	out_brdf_pdf_override = brdf_pdf_override;
	out_bssrdf_exit = bssrdf_exit;
	out_bssrdf_exit_pos = bssrdf_exit_pos;
}

// Realistic multi-element lens camera (pbrt-v4 RealisticCamera, src/shared/
// cameras.h). Host-side precompute (focus-adjusted lens table + exit-pupil
// bounds table) happens once in scene_builder.cpp by directly constructing a
// RealisticCamera<float> and reading its accessors - this device function
// only ports the per-ray hot path: film-plane mapping, exit-pupil sampling,
// and the per-element Snell's-law trace (Woop et al. watertight lens
// intersection, mirrors cameras.h's trace_lenses_from_film/sample_exit_pupil/
// generate_ray exactly, including the eta_i/eta_t Refract convention fix).
// u,v are in [0,1]^2 with v already flipped by the raygen caller (matching
// the other CameraKinds' "lower-left-origin" viewport convention) - unlike
// those, RealisticCamera's film coordinates increase top-to-bottom matching
// raw raster row order (see raster_to_film's comment in cameras.h), so v is
// un-flipped back here first.
// Returns false (weight left at 0) if the ray is fully vignetted.
__device__ __forceinline__ bool sample_realistic_camera_ray(
	const GpuCameraParams& cam, float u, float v, unsigned int& seed,
	float3& out_origin, float3& out_direction, float& out_weight
) {
	out_weight = 0.0f;
	if (cam.numLensElements <= 0 || cam.numExitPupilBounds <= 0) return false;

	float v_raw = 1.0f - v;  // undo raygen's lower-left-origin flip
	float pfx = -((2.0f*u - 1.0f) * cam.film_half_x);  // pbrt-v4 negates x
	float pfy =   (2.0f*v_raw - 1.0f) * cam.film_half_y;

	// sample_exit_pupil
	float rFilm = sqrtf(pfx*pfx + pfy*pfy);
	float film_diag = 2.0f * sqrtf(cam.film_half_x*cam.film_half_x + cam.film_half_y*cam.film_half_y);
	int sz = cam.numExitPupilBounds;
	int rIndex = (int)(rFilm / (film_diag*0.5f) * (float)sz);
	if (rIndex >= sz) rIndex = sz - 1;
	if (rIndex < 0) rIndex = 0;
	const GpuExitPupilBounds& b = cam.exitPupilBounds[rIndex];
	if (b.degenerate != 0) return false;

	float area = (b.xMax - b.xMin) * (b.yMax - b.yMin);
	if (area <= 0.0f) return false;
	float ppdf = 1.0f / area;

	float u0 = random_float(seed), u1 = random_float(seed);
	float lx = b.xMin + u0*(b.xMax - b.xMin);
	float ly = b.yMin + u1*(b.yMax - b.yMin);

	float sinTheta = (rFilm > 0.0f) ? pfy/rFilm : 0.0f;
	float cosTheta0 = (rFilm > 0.0f) ? pfx/rFilm : 1.0f;
	float ppx = cosTheta0*lx - sinTheta*ly;
	float ppy = sinTheta*lx + cosTheta0*ly;
	float ppz = cam.lens_rear_z;

	float rdx = ppx - pfx, rdy = ppy - pfy, rdz = ppz;
	float rLen = sqrtf(rdx*rdx + rdy*rdy + rdz*rdz);

	// trace_lenses_from_film: camera space (film z=0, +z toward scene) ->
	// lens space (z flipped): loz=-oz, ldz=-dz.
	float lox = pfx, loy = pfy, loz = 0.0f;
	float ldx = rdx, ldy = rdy, ldz = -rdz;
	float elementZ = 0.0f;

	for (int i = cam.numLensElements - 1; i >= 0; --i) {
		const GpuLensElement& el = cam.lensElements[i];
		elementZ -= el.thickness;
		bool isStop = (el.curvatureRadius == 0.0f);
		float t, nx = 0.0f, ny = 0.0f, nz = 0.0f;

		if (isStop) {
			if (ldz == 0.0f) return false;
			t = (elementZ - loz) / ldz;
			if (t < 0.0f) return false;
		} else {
			// intersect_spherical
			float zCenter = elementZ + el.curvatureRadius;
			float cox = lox, coy = loy, coz = loz - zCenter;
			float A = ldx*ldx + ldy*ldy + ldz*ldz;
			float B = 2.0f*(ldx*cox + ldy*coy + ldz*coz);
			float C = cox*cox + coy*coy + coz*coz - el.curvatureRadius*el.curvatureRadius;
			float disc = B*B - 4.0f*A*C;
			if (disc < 0.0f) return false;
			float sq = sqrtf(disc);
			float q = (B < 0.0f) ? -0.5f*(B - sq) : -0.5f*(B + sq);
			float t0 = q / A;
			float t1 = C / q;
			if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
			bool useCloserT = (ldz > 0.0f) != (el.curvatureRadius < 0.0f);
			t = useCloserT ? fminf(t0, t1) : fmaxf(t0, t1);
			if (t < 0.0f) return false;
			float hx0 = lox+t*ldx, hy0 = loy+t*ldy, hz0 = loz+t*ldz;
			nx = hx0; ny = hy0; nz = hz0 - zCenter;
			float nlen = sqrtf(nx*nx+ny*ny+nz*nz);
			if (nlen == 0.0f) return false;
			nx/=nlen; ny/=nlen; nz/=nlen;
			if (ldx*nx+ldy*ny+ldz*nz > 0.0f) { nx=-nx; ny=-ny; nz=-nz; }
		}

		float hx = lox+t*ldx, hy = loy+t*ldy, hz = loz+t*ldz;
		if (hx*hx + hy*hy > el.apertureRadius*el.apertureRadius) return false;
		lox = hx; loy = hy; loz = hz;

		if (!isStop) {
			float eta_i = (el.eta == 0.0f) ? 1.0f : el.eta;
			float eta_t = (i > 0 && cam.lensElements[i-1].eta != 0.0f) ? cam.lensElements[i-1].eta : 1.0f;
			float len = sqrtf(ldx*ldx+ldy*ldy+ldz*ldz);
			float dxn = ldx/len, dyn = ldy/len, dzn = ldz/len;
			float eta = eta_i/eta_t;  // matches cameras.h Refract's eta=eta_i/eta_t contract
			float cosI = -(dxn*nx+dyn*ny+dzn*nz);
			float sin2T = eta*eta * fmaxf(0.0f, 1.0f - cosI*cosI);
			if (sin2T >= 1.0f) return false;
			float cosT = sqrtf(1.0f - sin2T);
			ldx = eta*dxn + (eta*cosI - cosT)*nx;
			ldy = eta*dyn + (eta*cosI - cosT)*ny;
			ldz = eta*dzn + (eta*cosI - cosT)*nz;
		}
	}

	float lensOutOx = lox, lensOutOy = loy, lensOutOz = -loz;
	float lensOutDx = ldx, lensOutDy = ldy, lensOutDz = -ldz;

	float cosThetaW = (rLen > 0.0f) ? fabsf(rdz/rLen) : 0.0f;
	float lrz = cam.lens_rear_z;
	if (lrz <= 0.0f) return false;
	float w = (cosThetaW*cosThetaW*cosThetaW*cosThetaW) / (ppdf * lrz * lrz);

	// Camera space -> world space via su(right)/sv(up)/sw(forward)/origin.
	out_origin = cam.origin + lensOutOx*cam.su + lensOutOy*cam.sv + lensOutOz*cam.sw;
	out_direction = normalize(lensOutDx*cam.su + lensOutDy*cam.sv + lensOutDz*cam.sw);
	out_weight = w;
	return true;
}

// Generate a primary camera ray for raster coordinates (u,v) in [0,1]^2,
// dispatching on params.camera.kind. Mirrors the CPU camera models in
// src/shared/cameras.h (OrthographicCamera/SphericalCamera/RealisticCamera::
// generate_ray) and src/TheRestOfYourLife/camera.h's book-style
// defocus_angle/focus_dist thin-lens DOF, which the Perspective case folds
// in via camera.defocus_disk_u/v (both zero = DOF disabled, matching how
// scene_builder.cpp always zero-initializes GpuCameraParams). `weight`
// applies to Realistic only (cos^4(theta)/pdf vignetting term, 1.0 for every
// other CameraKind - fold into the caller's initial path throughput).
__device__ __forceinline__ void generate_primary_ray(
	float u, float v, unsigned int& seed, float3& origin, float3& direction, float& weight
) {
	const GpuCameraParams& cam = params.camera;
	weight = 1.0f;
	switch (cam.kind) {
		case CameraKind::Orthographic: {
			origin = cam.lower_left_corner + u * cam.horizontal + v * cam.vertical;
			direction = cam.w;
			break;
		}
		case CameraKind::Spherical: {
			// pbrt-v4 SphericalCamera::GenerateRay (EquiRectangular mapping):
			// theta in [0,pi], phi in [0,2pi], then swap(dir.y, dir.z) - see
			// src/shared/cameras.h for the reference this mirrors.
			float theta = 3.14159265358979323846f * v;
			float phi   = 2.0f * 3.14159265358979323846f * u;
			float sin_t = sinf(theta), cos_t = cosf(theta);
			float lx = sin_t * cosf(phi);
			float ly = cos_t;             // swapped with lz below (pbrt-v4 convention)
			float lz = sin_t * sinf(phi); // swapped with ly above
			origin = cam.origin;
			direction = normalize(lx * cam.su + ly * cam.sv + lz * cam.sw);
			break;
		}
		case CameraKind::Realistic: {
			if (!sample_realistic_camera_ray(cam, u, v, seed, origin, direction, weight)) {
				origin = cam.origin;
				direction = cam.sw;  // arbitrary valid direction; weight=0 zeroes its contribution
				weight = 0.0f;
			}
			break;
		}
		default: { // Perspective, optionally thin-lens DOF
			float3 pixel_sample = cam.lower_left_corner + u * cam.horizontal + v * cam.vertical;
			bool hasDOF = (cam.defocus_disk_u.x != 0.0f || cam.defocus_disk_u.y != 0.0f || cam.defocus_disk_u.z != 0.0f ||
						   cam.defocus_disk_v.x != 0.0f || cam.defocus_disk_v.y != 0.0f || cam.defocus_disk_v.z != 0.0f);
			if (hasDOF) {
				float3 p = random_in_unit_disk(seed);
				origin = cam.origin + p.x * cam.defocus_disk_u + p.y * cam.defocus_disk_v;
			} else {
				origin = cam.origin;
			}
			direction = normalize(pixel_sample - origin);
			break;
		}
	}
}

//==============================================================================
// Sphere Intersection Program
//==============================================================================

