// sppm_programs.cu
// OptiX device programs for the GPU SPPM (Stochastic Progressive Photon
// Mapping) integrator.
//
// Own OptiX module (own pipeline/payload layout) -- same reason
// wavefront_programs.cu has its own sphere/quad intersection copies
// instead of reusing optix_programs.cu's: OptiX rejects combining
// hit-group programs from two modules compiled with different
// pipelineCompileOptions.numPayloadValues into one hitgroup ("could not
// be resolved to a common payloadType"). See wavefront_programs.cu's own
// comment for the fuller story -- this is a third copy of the same
// sphere/quad intersection math for the same reason, under a third module.
//
// __raygen__sppm_camera_pass (sub-phase 1b): one thread per pixel, direct
// port of src/shared/sppm.h's SPPMCameraPass()/sppm_adapter.h's
// sppm_camera_pass_with_sky() (minus the sky branch -- no sky/infinite-light
// support yet, see optix_interface.cpp's scene-capability check). Follows a
// specular chain via delta-BSDF resampling (RoughDielectricBxDF-style GGX,
// plus Metal/Dielectric/Conductor added in a later generalization pass -- see
// sppm_is_delta_material()/sppm_sample_delta_material() below for the full,
// current delta set and why it stops there) until it hits a non-delta
// surface -- Lambertian always, or a glossy (non-smooth) Conductor/
// RoughMetal instance as of a later fix (see sppm_is_delta_material()'s own
// comment on why that classification is now per-instance, not per-type) --
// records that as the pixel's visible point, and computes one NEE sample
// toward the scene's area light(s) via the same power-weighted alias table
// the regular path tracers already use. Unlike
// the regular path tracers' NEE, there's no MIS weight here: SPPM's camera
// pass calls this exactly once per pixel with no competing BSDF-sampled
// continuation in the same estimate (matches sppm_adapter.h's own
// DirectLight() -- see its comment for the fuller reasoning), so no
// balance/power heuristic is needed.
//
// pixels[i].Ld is written straight to the framebuffer; the real
// multi-iteration loop (hash grid, photon pass, radius contraction) lives in
// SPPMPathTracer::render() (sppm_path_tracer.cpp), which also computes the
// hash-grid's world-space bounds dynamically from each iteration's actual
// visible points -- NOT a fixed/hardcoded box, so that part of the pipeline
// was already scene-agnostic before this generalization pass; the real
// scene-B3-only restriction this file's material dispatch used to impose is
// what's documented at each MaterialType branch below.

#include <optix.h>
#include "sppm_types.h"
#include "optix_types.h"
#include "optix_math_helpers.h"
#include "math_utils.h"   // cpu_gpu_reflect/cpu_gpu_refract
#include "fresnel.h"       // FrDielectric
#include "microfacet.h"    // TrowbridgeReitz
#include "bxdfs_conductor.h" // ConductorBxDF/RoughMetalBxDF - real glossy f()/effectively_smooth()

extern "C" { __constant__ SPPMLaunchParams sppm_params; }

// ---- pointer packing (same technique as wavefront_programs.cu) ----
static __device__ __forceinline__ void sppmPackPointer(void* ptr, unsigned int& p0, unsigned int& p1) {
	const unsigned long long up = (unsigned long long)ptr;
	p0 = (unsigned int)(up >> 32);
	p1 = (unsigned int)(up & 0xFFFFFFFFull);
}
static __device__ __forceinline__ void* sppmUnpackPointer(unsigned int p0, unsigned int p1) {
	return (void*)((unsigned long long)p0 << 32 | (unsigned long long)p1);
}

// Payload for the radiance ray (primary + specular-chain continuation).
struct SPPMHitPayload {
	float3 hitPoint;
	float3 normal;
	int    materialIdx;
	bool   hit;
};

// ---- RNG: PCG hash, identical algorithm to wavefront_kernels.cu's wf_pcg/
// wf_rand (duplicated with an sppm_ prefix rather than shared -- matches
// this GPU codebase's established pattern throughout of not sharing
// device helpers across the recursive/wavefront/SPPM backends). ----
__device__ __forceinline__ unsigned int sppm_pcg(unsigned int seed) {
	unsigned int state = seed * 747796405u + 2891336453u;
	unsigned int word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}
__device__ __forceinline__ float sppm_rand(unsigned int& seed) {
	seed = sppm_pcg(seed);
	return float(seed) / 4294967296.0f;
}
// Uniform point on the unit sphere via rejection sampling -- direct copy of
// wavefront_kernels.cu's wf_rand_unit (identical algorithm, sppm_ prefix,
// same "don't share device helpers across backends" convention as this
// file's other sppm_-prefixed ports). Used by the photon pass's
// Lambertian-bounce sampling below: normalize(normal + sppm_rand_unit(seed))
// is a cosine-weighted hemisphere sample (same identity wavefront_kernels.cu's
// own Lambertian case already relies on), so f_val*cosI/pdf collapses to
// exactly `albedo` with no separate cosine-pdf bookkeeping needed.
__device__ __forceinline__ float3 sppm_rand_unit(unsigned int& seed) {
	while (true) {
		float3 p = make_float3(2.0f*sppm_rand(seed)-1.0f, 2.0f*sppm_rand(seed)-1.0f, 2.0f*sppm_rand(seed)-1.0f);
		float l = dot(p, p);
		if (l > 1e-8f && l < 1.0f) return p / sqrtf(l);
	}
}

// ============================================================================
// __intersection__sppm_sphere / __intersection__sppm_quad
// __closesthit__sppm_sphere / __closesthit__sppm_quad
// __miss__sppm_radiance
//
// Direct copies of wavefront_programs.cu's radiance-ray sphere/quad
// programs (identical math) -- see this file's top comment for why this
// module needs its own copy rather than sharing wavefront_programs.cu's.
// ============================================================================

extern "C" __global__ void __intersection__sppm_sphere() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const SphereData& sphere = sppm_params.spheres[primIdx];

	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float  ray_tmin = optixGetRayTmin();
	const float  ray_tmax = optixGetRayTmax();

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

extern "C" __global__ void __intersection__sppm_quad() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const QuadData& quad = sppm_params.quads[primIdx];

	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float  ray_tmin = optixGetRayTmin();
	const float  ray_tmax = optixGetRayTmax();

	const float denom = dot(quad.normal, ray_dir);
	if (fabsf(denom) < 1e-8f) return;

	const float t = (quad.D - dot(quad.normal, ray_orig)) / denom;
	if (t < ray_tmin || t > ray_tmax) return;

	const float3 intersection = ray_orig + t * ray_dir;
	const float3 planar_vec = intersection - quad.Q;

	const float w_dot_w = dot(quad.w, quad.w);
	const float alpha = dot(quad.w, cross(planar_vec, quad.v)) / w_dot_w;
	const float beta  = dot(quad.w, cross(quad.u, planar_vec)) / w_dot_w;
	if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f) return;

	optixReportIntersection(
		t, 0,
		__float_as_int(quad.normal.x),
		__float_as_int(quad.normal.y),
		__float_as_int(quad.normal.z),
		0);
}

extern "C" __global__ void __closesthit__sppm_sphere() {
	SPPMHitPayload* payload = (SPPMHitPayload*)sppmUnpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	const int sphereIdx = optixGetPrimitiveIndex();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float  t_hit    = optixGetRayTmax();
	float3 hit_point = ray_orig + t_hit * ray_dir;

	const SphereData& sph = sppm_params.spheres[sphereIdx];
	float3 outward_normal = normalize(hit_point - sph.center);
	bool front_face = dot(ray_dir, outward_normal) < 0.0f;
	float3 normal = front_face ? outward_normal : -outward_normal;

	payload->hitPoint    = hit_point;
	payload->normal      = normal;
	payload->materialIdx = sph.materialIdx;
	payload->hit         = true;
}

extern "C" __global__ void __closesthit__sppm_quad() {
	SPPMHitPayload* payload = (SPPMHitPayload*)sppmUnpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	const int quadIdx = optixGetPrimitiveIndex();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float  t_hit    = optixGetRayTmax();
	float3 hit_point = ray_orig + t_hit * ray_dir;

	const QuadData& q = sppm_params.quads[quadIdx];
	bool front_face = dot(ray_dir, q.normal) < 0.0f;
	float3 normal = front_face ? q.normal : -q.normal;

	payload->hitPoint    = hit_point;
	payload->normal      = normal;
	payload->materialIdx = q.materialIdx;
	payload->hit         = true;
}

extern "C" __global__ void __miss__sppm_radiance() {
	// nothing -- hit remains false
}

// ============================================================================
// Shadow ray: single-scalar payload (matches optix_device_helpers.h's
// trace_shadow_ray exactly -- occluded=1 default, any opaque hit sets/keeps
// it, a light or transmissive hit clears it). Separate any-hit programs per
// geometry type, mirroring optix_anyhit_shadow.h's __anyhit__shadow_sphere/
// __anyhit__shadow_quad (same occlusion semantics: a diffuse-light hit lets
// the shadow ray "see" the light; RoughDielectric -- Phase 1's only
// transmissive material -- is treated as non-occluding, same simplification
// the regular path tracers already make, not something new to this port).
// ============================================================================

// Transmissive materials that let a shadow ray pass through rather than
// blocking NEE outright -- matches the established convention already used
// by both other GPU backends (optix_anyhit_shadow.h's __anyhit__shadow_*,
// wavefront_programs.cu's __anyhit__wf_shadow_*, see the latter's own
// comment for the full "why", including the B14 regression that motivated
// treating DiffuseLight specially too). Generalized here from "RoughDielectric
// only" to also cover MaterialType::Dielectric (A1 Cornell Box's smooth glass
// sphere, and any other scene using a plain dielectric) now that GPU SPPM's
// camera/photon passes handle it (see sppm_is_delta_material() below) --
// Metal/Conductor are deliberately NOT added here: both are opaque reflective
// materials (no transmission lobe at all), so the pre-existing "anything else
// blocks the shadow ray" fallthrough is already the physically correct
// behavior for them, same as it already is for Lambertian.
static __device__ __forceinline__ bool sppm_is_transmissive_material(MaterialType t) {
	return t == MaterialType::RoughDielectric || t == MaterialType::Dielectric ||
	       t == MaterialType::DiffuseTransmission;
}

extern "C" __global__ void __anyhit__sppm_shadow_sphere() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const SphereData& sphere = sppm_params.spheres[primIdx];
	const MaterialData& mat = sppm_params.materials[sphere.materialIdx];

	if (mat.type == MaterialType::DiffuseLight) {
		optixSetPayload_0(0);
		optixTerminateRay();
		return;
	}
	if (sppm_is_transmissive_material(mat.type)) {
		optixIgnoreIntersection();
		return;
	}
	optixSetPayload_0(1);
	optixTerminateRay();
}

extern "C" __global__ void __anyhit__sppm_shadow_quad() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const QuadData& quad = sppm_params.quads[primIdx];
	const MaterialData& mat = sppm_params.materials[quad.materialIdx];

	if (mat.type == MaterialType::DiffuseLight) {
		optixSetPayload_0(0);
		optixTerminateRay();
		return;
	}
	if (sppm_is_transmissive_material(mat.type)) {
		optixIgnoreIntersection();
		return;
	}
	optixSetPayload_0(1);
	optixTerminateRay();
}

extern "C" __global__ void __miss__sppm_shadow() {
	optixSetPayload_0(0);  // unoccluded
}

// Returns true if the path from origin toward direction (up to max_distance)
// is unoccluded. Same call shape as optix_device_helpers.h's
// trace_shadow_ray(): SBT offset/stride encode [radiance, shadow] pairs per
// present geometry type (stride=2), matching sppm_path_tracer.cpp's SBT
// layout.
static __device__ __forceinline__ bool sppm_trace_shadow_ray(
	const float3& origin, const float3& direction, float max_distance) {
	unsigned int occluded = 1;
	optixTrace(
		sppm_params.traversable,
		origin, direction,
		0.001f, max_distance,
		0.0f,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
		1,  // SBT offset (shadow ray type)
		2,  // SBT stride (2 ray types: radiance, shadow)
		1,  // miss SBT index (shadow miss)
		occluded
	);
	return occluded == 0;
}

// Sample a point on a quad light -- direct copy of wavefront_kernels.cu's
// wf_sample_quad_light (identical math, sppm_ prefix -- see this file's
// top comment on why device helpers aren't shared across backends here).
static __device__ float3 sppm_sample_quad_light(const QuadData& q, const float3& hit,
                                                  unsigned int& seed, float& geom_pdf, float& maxDist) {
	float s = sppm_rand(seed), t = sppm_rand(seed);
	float3 p = q.Q + s * q.u + t * q.v;
	float3 dir = p - hit;
	maxDist    = length(dir);
	dir        = normalize(dir);
	float area  = length(cross(q.u, q.v));
	float cos_l = fabsf(dot(q.normal, -dir));
	geom_pdf = (cos_l > 1e-6f && maxDist > 1e-6f) ? (maxDist * maxDist) / (cos_l * area) : 0.0f;
	return dir;
}

// Sample a point on a sphere light -- direct copy of wavefront_kernels.cu's
// wf_sample_sphere_light (identical math, sppm_ prefix).
static __device__ float3 sppm_sample_sphere_light(const SphereData& sph, const float3& hit,
                                                    unsigned int& seed, float& pdf, float& maxDist) {
	float3 to_c = sph.center - hit;
	float dist  = length(to_c);
	float r     = sph.radius;
	float3 dir;
	if (dist <= r) {
		pdf = 1.0f / (4.0f * 3.14159265f * r * r);
		// Uniform sphere-surface direction via rejection sampling (no
		// existing wf_rand_unit-equivalent in this file yet -- Phase 1's
		// only light in its only target scene, scene 11, is a quad, so this
		// branch is unexercised in practice but kept for completeness/
		// future scenes with a sphere light).
		float3 p;
		do {
			p = make_float3(2.0f*sppm_rand(seed)-1.0f, 2.0f*sppm_rand(seed)-1.0f, 2.0f*sppm_rand(seed)-1.0f);
		} while (dot(p, p) > 1.0f || dot(p, p) < 1e-8f);
		dir = normalize(p);
	} else {
		float cos_max = sqrtf(fmaxf(0.0f, 1.0f - (r * r) / (dist * dist)));
		float phi     = 2.0f * 3.14159265f * sppm_rand(seed);
		float cos_t   = 1.0f - sppm_rand(seed) * (1.0f - cos_max);
		float sin_t   = sqrtf(fmaxf(0.0f, 1.0f - cos_t * cos_t));
		float3 w      = normalize(to_c);
		float3 u, v;
		if (fabsf(w.x) > 0.9f) u = normalize(cross(make_float3(0,1,0), w));
		else                    u = normalize(cross(make_float3(1,0,0), w));
		v = cross(w, u);
		dir = normalize(sin_t * cosf(phi) * u + sin_t * sinf(phi) * v + cos_t * w);
		float solid = 2.0f * 3.14159265f * (1.0f - cos_max);
		pdf = (solid > 1e-10f) ? 1.0f / solid : 1.0f;
	}
	float3 oc = hit - sph.center;
	float b = dot(oc, dir);
	float c = dot(oc, oc) - r * r;
	float disc = fmaxf(0.0f, b * b - c);
	float sq = sqrtf(disc);
	float tNear = -b - sq;
	maxDist = (tNear > 1e-6f) ? tNear : fmaxf(0.0f, -b + sq);
	return dir;
}

// GGX VNDF microfacet resample for a RoughDielectric hit -- shared by both
// raygens below (camera pass and photon pass, sub-phase 1d). Same-file
// sharing only; this codebase's "no cross-backend device helper sharing"
// convention (see this file's top comment) is about not reaching across
// wavefront/recursive/SPPM module boundaries, not about duplicating logic
// within one already-cohesive module. Returns false (and leaves out_dir
// unset) on the wi_z<=0 grazing-incidence edge case -- caller must treat
// that as a terminated path, matching the camera pass's original inline
// `break` here before this was extracted.
static __device__ __forceinline__ bool sppm_sample_rough_dielectric(
	const float3& dir_in, const float3& n, const MaterialData& mat,
	unsigned int& seed, float3& out_dir) {
	float alpha = sqrtf(mat.fuzz);
	float3 up_v = (fabsf(n.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
	float3 tan_v = normalize(cross(up_v, n));
	float3 bitan = cross(n, tan_v);
	float3 wi_w = -normalize(dir_in);
	float wi_x = dot(wi_w, tan_v), wi_y = dot(wi_w, bitan), wi_z = dot(wi_w, n);
	if (wi_z <= 0.0f) return false;

	TrowbridgeReitz<float> rd_dist(alpha, alpha);
	float wm_x, wm_y, wm_z;
	rd_dist.Sample_wm(wi_x, wi_y, wi_z, sppm_rand(seed), sppm_rand(seed), wm_x, wm_y, wm_z);
	float rd_dot = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
	float Fr = FrDielectric(rd_dot, mat.ior);

	if (sppm_rand(seed) < Fr) {
		float wo_x = 2.0f*rd_dot*wm_x - wi_x;
		float wo_y = 2.0f*rd_dot*wm_y - wi_y;
		float wo_z = 2.0f*rd_dot*wm_z - wi_z;
		out_dir = normalize(wo_x*tan_v + wo_y*bitan + wo_z*n);
	} else {
		float3 wm_world = wm_x*tan_v + wm_y*bitan + wm_z*n;
		float eta = dot(dir_in, n) < 0.0f ? (1.0f / mat.ior) : mat.ior;
		out_dir = cpu_gpu_refract<float3, float>(normalize(dir_in), wm_world, eta);
	}
	return true;
}

// True for the MaterialTypes/instances GPU SPPM's camera/photon passes
// treat as a delta (specular) BSDF -- i.e. resampled via BSDFSampleF-style
// importance sampling alone, with no visible point recorded and no
// NEE/photon-deposit evaluation at that hit. Matches CPU's
// material::is_delta_bsdf() (src/TheRestOfYourLife/material_base.h)
// restricted to the subset this GPU port actually implements a sampler
// for: CPU's own delta set also includes coated_diffuse, thin_dielectric,
// and coated_conductor -- those remain genuinely unsupported here (see
// optix_interface.cpp's scene-capability check, which rejects any scene
// using a MaterialType not covered by this function or by the Lambertian
// fallback below) rather than silently mis-scattered through that
// fallback.
//
// Conductor/RoughMetal are NOT a fixed answer per MaterialType the way
// Metal/Dielectric/DiffuseTransmission are: like their CPU material
// counterparts (rough_metal::is_delta_bsdf()/conductor::is_delta_bsdf(),
// material_pbrt.h), they branch on the instance's ACTUAL roughness via
// BxDF::effectively_smooth() (bxdfs_conductor.h) -- this used to be a
// static per-type answer here too, which silently starved every glossy
// (non-smooth) Conductor/RoughMetal instance of NEE and photon deposit
// exactly like the CPU bug this mirrors and was fixed alongside.
static __device__ __forceinline__ bool sppm_is_delta_material(const MaterialData& mat) {
	if (mat.type == MaterialType::Conductor) {
		float alpha = sqrtf(mat.fuzz);   // same derivation sppm_sample_delta_material's Conductor case already uses
		return ConductorBxDF<float>{ mat.eta_c.x, mat.eta_c.y, mat.eta_c.z,
		                              mat.k_c.x, mat.k_c.y, mat.k_c.z, alpha, alpha }.effectively_smooth();
	}
	if (mat.type == MaterialType::RoughMetal) {
		float alpha = sqrtf(mat.fuzz);   // same derivation sppm_sample_delta_material's RoughMetal case already uses
		return RoughMetalBxDF<float>{ mat.albedo.x, mat.albedo.y, mat.albedo.z, alpha, alpha }.effectively_smooth();
	}
	return mat.type == MaterialType::RoughDielectric || mat.type == MaterialType::Metal ||
	       mat.type == MaterialType::Dielectric || mat.type == MaterialType::DiffuseTransmission;
}

// f(wo,wi) for a non-delta material at a captured visible point, mirroring
// CPU's sppm_bsdf_f() (src/TheRestOfYourLife/bsdf_bridge.h): `wo` is the
// FIXED/view direction (the captured vp_wo, or the direction back toward
// the previous photon vertex), `wi` is the QUERIED direction (toward the
// light for camera-pass NEE, or the reversed current photon direction for
// photon-pass deposit) -- both callers below supply these in that order.
// Lambertian is Phase 1's original, only case; Conductor/RoughMetal are new
// as of this fix, now reachable here because sppm_is_delta_material() above
// can classify a glossy instance of either as non-delta.
static __device__ __forceinline__ float3 sppm_bsdf_f(
	const MaterialData& mat, const float3& wo, const float3& wi, const float3& n) {
	if (mat.type == MaterialType::Conductor || mat.type == MaterialType::RoughMetal) {
		float3 up_v  = (fabsf(n.x) > 0.9f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
		float3 tan_v = normalize(cross(up_v, n));
		float3 bitan = cross(n, tan_v);
		// BxDF struct convention in this codebase (see rough_metal::
		// scattering_pdf(), material_pbrt.h): "wi" params take the FIXED/
		// view direction, "wo" params take the newly QUERIED direction --
		// inverted from the usual pbrt wo=view/wi=light naming, so `wo`
		// (this function's fixed direction) maps to the BxDF's wi_x/y/z
		// below, not its wo_x/y/z.
		float bxdf_wi_x = dot(wo, tan_v), bxdf_wi_y = dot(wo, bitan), bxdf_wi_z = dot(wo, n);
		float bxdf_wo_x = dot(wi, tan_v), bxdf_wo_y = dot(wi, bitan), bxdf_wo_z = dot(wi, n);
		float alpha = sqrtf(mat.fuzz);
		float fr, fg, fb;
		if (mat.type == MaterialType::Conductor) {
			ConductorBxDF<float> bx{ mat.eta_c.x, mat.eta_c.y, mat.eta_c.z,
			                          mat.k_c.x, mat.k_c.y, mat.k_c.z, alpha, alpha };
			bx.f(bxdf_wi_x, bxdf_wi_y, bxdf_wi_z, bxdf_wo_x, bxdf_wo_y, bxdf_wo_z, fr, fg, fb);
		} else {
			RoughMetalBxDF<float> bx{ mat.albedo.x, mat.albedo.y, mat.albedo.z, alpha, alpha };
			bx.f(bxdf_wi_x, bxdf_wi_y, bxdf_wi_z, bxdf_wo_x, bxdf_wo_y, bxdf_wo_z, fr, fg, fb);
		}
		return make_float3(fr, fg, fb);
	}
	// Lambertian fast path (Phase 1's original, only case before this fix).
	const float inv_pi = 1.0f / 3.14159265358979323846f;
	return mat.albedo * inv_pi;
}

// Importance-samples a new direction + beta multiplier at a Metal/Dielectric/
// Conductor hit (RoughDielectric keeps its own pre-existing
// sppm_sample_rough_dielectric() above, called separately by both raygens --
// left untouched rather than folded in here, so scene B3's already-verified
// output can't be perturbed by this generalization). Direct, unmodified-math
// ports of wavefront_kernels.cu's own MaterialType::Metal/Dielectric/
// Conductor cases -- see that file's per-case comments for the full
// derivation of each (Schlick reflectance for Dielectric, GGX VNDF + complex
// Fresnel with the G2/G1 shadow-masking weight for Conductor). Shared by both
// the camera pass and the photon pass raygens below: CPU's SPPMCameraPass/
// SPPMPhotonPass treat every delta material identically (resample, multiply
// beta, continue the walk), so one shared sampler is correct for both rather
// than two independent copies.
static __device__ __forceinline__ bool sppm_sample_delta_material(
	const float3& dir_in, const float3& n, const MaterialData& mat,
	unsigned int& seed, float3& out_dir, float3& out_atten) {
	switch (mat.type) {
	case MaterialType::Metal: {
		float3 reflected = cpu_gpu_reflect(normalize(dir_in), n);
		float3 d = normalize(reflected + mat.fuzz * sppm_rand_unit(seed));
		if (dot(d, n) <= 0.0f) return false;  // scattered below the surface -- absorbed
		out_dir   = d;
		out_atten = mat.albedo;
		return true;
	}
	case MaterialType::Dielectric: {
		float  eta      = dot(dir_in, n) < 0.0f ? (1.0f / mat.ior) : mat.ior;
		float3 unit_dir = normalize(dir_in);
		float  cos_t    = fminf(dot(-unit_dir, n), 1.0f);
		float  sin_t    = sqrtf(fmaxf(0.0f, 1.0f - cos_t * cos_t));
		bool   cannot_refract = eta * sin_t > 1.0f;
		float  r0 = (1.0f - mat.ior) / (1.0f + mat.ior);
		r0 = r0 * r0;
		float schlick = r0 + (1.0f - r0) * powf(1.0f - cos_t, 5.0f);
		bool is_transmission;
		if (cannot_refract || schlick > sppm_rand(seed)) {
			out_dir = cpu_gpu_reflect(unit_dir, n);
			is_transmission = false;
		} else {
			out_dir = cpu_gpu_refract<float3, float>(unit_dir, n, eta);
			is_transmission = true;
		}
		// Tf tints transmission only, matching real colored glass -- see
		// add_dielectric()'s transmission_filter comment (identical
		// convention on both other backends' Dielectric cases).
		out_atten = is_transmission ? mat.transmission_filter : make_float3(1.0f, 1.0f, 1.0f);
		return true;
	}
	case MaterialType::Conductor: {
		float alpha  = sqrtf(mat.fuzz);
		float3 up_v  = (fabsf(n.x) > 0.9f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
		float3 tan_v = normalize(cross(up_v, n));
		float3 bitan = cross(n, tan_v);
		float3 wi_w  = normalize(-dir_in);
		float wi_x = dot(wi_w, tan_v), wi_y = dot(wi_w, bitan), wi_z = dot(wi_w, n);
		if (wi_z <= 0.0f) return false;
		TrowbridgeReitz<float> dist(alpha, alpha);
		float wm_x, wm_y, wm_z;
		dist.Sample_wm(wi_x, wi_y, wi_z, sppm_rand(seed), sppm_rand(seed), wm_x, wm_y, wm_z);
		float c_dot = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
		float wo_x = 2.0f*c_dot*wm_x - wi_x;
		float wo_y = 2.0f*c_dot*wm_y - wi_y;
		float wo_z = 2.0f*c_dot*wm_z - wi_z;
		if (wo_z <= 0.0f) return false;
		float G1_wi  = dist.G1(wi_x, wi_y, wi_z);
		float G_wowi = dist.G(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
		float weight = (G1_wi > 1e-8f) ? G_wowi / G1_wi : 0.0f;
		float3 F = FrConductorRGB(c_dot, mat.eta_c.x, mat.eta_c.y, mat.eta_c.z, mat.k_c.x, mat.k_c.y, mat.k_c.z);
		out_atten = make_float3(F.x * weight, F.y * weight, F.z * weight);
		out_dir   = normalize(wo_x*tan_v + wo_y*bitan + wo_z*n);
		return true;
	}
	case MaterialType::RoughMetal: {
		// GGX VNDF + flat-tint reflectance, no complex Fresnel (pbrt-v4/RTOW
		// rough_metal) - same shape as MaterialType::Conductor above, minus
		// FrConductorRGB (RoughMetalBxDF has no real Fresnel model). Direct,
		// unmodified-math port of wavefront_kernels.cu's own
		// MaterialType::RoughMetal case, matching this function's own header
		// comment for Metal/Dielectric/Conductor.
		float alpha  = sqrtf(mat.fuzz);
		float3 up_v  = (fabsf(n.x) > 0.9f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
		float3 tan_v = normalize(cross(up_v, n));
		float3 bitan = cross(n, tan_v);
		float3 wi_w  = normalize(-dir_in);
		float wi_x = dot(wi_w, tan_v), wi_y = dot(wi_w, bitan), wi_z = dot(wi_w, n);
		if (wi_z <= 0.0f) return false;
		TrowbridgeReitz<float> dist(alpha, alpha);
		float wm_x, wm_y, wm_z;
		dist.Sample_wm(wi_x, wi_y, wi_z, sppm_rand(seed), sppm_rand(seed), wm_x, wm_y, wm_z);
		float c_dot = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
		float wo_x = 2.0f*c_dot*wm_x - wi_x;
		float wo_y = 2.0f*c_dot*wm_y - wi_y;
		float wo_z = 2.0f*c_dot*wm_z - wi_z;
		if (wo_z <= 0.0f) return false;
		float G1_wi  = dist.G1(wi_x, wi_y, wi_z);
		float G_wowi = dist.G(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
		float weight = (G1_wi > 1e-8f) ? G_wowi / G1_wi : 0.0f;
		out_atten = make_float3(mat.albedo.x * weight, mat.albedo.y * weight, mat.albedo.z * weight);
		out_dir   = normalize(wo_x*tan_v + wo_y*bitan + wo_z*n);
		return true;
	}
	case MaterialType::DiffuseTransmission: {
		// pbrt-v4 DiffuseTransmissionBxDF -- direct, unmodified-math port of
		// wavefront_kernels.cu's own MaterialType::DiffuseTransmission case.
		// .albedo/.emission are genuine union aliases for R (reflectance) /
		// T (transmittance) for this MaterialType (see optix_types.h's
		// MaterialData), not real emission -- same field-reuse convention
		// wavefront already relies on. No NEE (matches wavefront's own
		// "zero explicit NEE, all illumination via the BSDF-sampled bounce"),
		// which is exactly this function's delta-material shape. Texture-
		// bound R/T deliberately omitted: scene_builder.cpp's
		// add_diffuse_transmission() never sets a texture index, so no
		// current GPU SPPM-eligible scene needs it.
		float3 R = mat.albedo;
		float3 T = mat.emission;
		float pr = fmaxf(R.x, fmaxf(R.y, R.z));
		float pt = fmaxf(T.x, fmaxf(T.y, T.z));
		if (pr + pt <= 0.0f) return false;
		if (sppm_rand(seed) < pr / (pr + pt)) {
			out_dir   = normalize(n + sppm_rand_unit(seed));
			out_atten = R;
		} else {
			out_dir   = normalize(-n + sppm_rand_unit(seed));
			out_atten = T;
		}
		return true;
	}
	default:
		return false;  // RoughDielectric (own sampler) / non-delta -- not this function's job
	}
}

// Uniform point on a quad light's surface + its (fixed) front-facing normal
// -- direct analog of CPU quad::sample_area() (src/TheRestOfYourLife/
// quad.h), used for photon EMISSION (sub-phase 1d), as opposed to
// sppm_sample_quad_light's NEE-toward-a-reference-point sampling above.
// out_pdf_pos is the area-measure density alone (1/area); the caller
// combines it with the light-selection pmf separately, matching CPU
// SampleLightLe's own `pdf_pos = as.pdf_pos * emitter_alias_.pmf(idx)`.
static __device__ float3 sppm_sample_quad_area(const QuadData& q, unsigned int& seed,
                                                  float3& out_n, float& out_pdf_pos) {
	float s = sppm_rand(seed), t = sppm_rand(seed);
	float3 p = q.Q + s * q.u + t * q.v;
	out_n = q.normal;
	float area = length(cross(q.u, q.v));
	out_pdf_pos = (area > 1e-12f) ? (1.0f / area) : 0.0f;
	return p;
}

// Uniform point on a sphere light's surface -- rejection-sampled, same
// approach as sppm_sample_sphere_light's own dist<=r branch above.
static __device__ float3 sppm_sample_sphere_area(const SphereData& sph, unsigned int& seed,
                                                    float3& out_n, float& out_pdf_pos) {
	float3 p;
	do {
		p = make_float3(2.0f*sppm_rand(seed)-1.0f, 2.0f*sppm_rand(seed)-1.0f, 2.0f*sppm_rand(seed)-1.0f);
	} while (dot(p, p) > 1.0f || dot(p, p) < 1e-8f);
	out_n = normalize(p);
	out_pdf_pos = 1.0f / (4.0f * 3.14159265f * sph.radius * sph.radius);
	return sph.center + sph.radius * out_n;
}

// ============================================================================
// __raygen__sppm_camera_pass (sub-phase 1b) -- see file header comment.
// ============================================================================
extern "C" __global__ void __raygen__sppm_camera_pass() {
	const uint3 idx = optixGetLaunchIndex();
	const unsigned int width  = sppm_params.width;
	const unsigned int height = sppm_params.height;
	if (idx.x >= width || idx.y >= height) return;
	const unsigned int pixelIdx = idx.y * width + idx.x;

	unsigned int seed = sppm_pcg(sppm_pcg(pixelIdx) ^ 0x9E3779B9u);

	SPPMPixelGPU& pixel = sppm_params.pixels[pixelIdx];
	// pixel.Ld is NOT reset here -- it accumulates across every iteration of
	// the outer SPPM loop and is divided by nIterations in
	// sppm_final_image_kernel (sppm_kernels.cu), mirroring src/shared/
	// sppm.h's own SPPMCameraPass comment on this exact point. The pixel
	// buffer is zero-initialized ONCE by the host before the first
	// iteration (SPPMPathTracer::render()/renderTrivial()) so this still
	// starts from zero on iteration 0.
	pixel.vp_valid = false;

	// Primary ray through the pixel center (no jitter/DOF -- matches
	// sub-phase 1a's convention; antialiasing isn't the point of Phase 1).
	// Y is flipped (height-1-idx.y, not idx.y) because image row 0 is the
	// TOP of the output image but GpuCameraParams::lower_left_corner-based
	// viewport math has v=0 at the BOTTOM -- same flip optix_raygen.h's own
	// __raygen__rg applies (its comment: "Flip Y"); missing it here first
	// showed up as the ceiling light rendering at the bottom of the image.
	float s = (float(idx.x) + 0.5f) / float(width);
	float t = (float(height - 1 - idx.y) + 0.5f) / float(height);
	float3 org = sppm_params.camera.origin;
	float3 dir = normalize(
		sppm_params.camera.lower_left_corner
		+ s * sppm_params.camera.horizontal
		+ t * sppm_params.camera.vertical
		- org);

	float3 beta = make_float3(1.0f, 1.0f, 1.0f);

	for (unsigned int depth = 0; depth < sppm_params.maxDepth; ++depth) {
		SPPMHitPayload payload;
		payload.hit = false;

		unsigned int p0, p1;
		sppmPackPointer(&payload, p0, p1);

		optixTrace(
			sppm_params.traversable,
			org, dir,
			0.001f, 1e30f,
			0.0f,
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_NONE,
			0,  // SBT offset (radiance ray type)
			2,  // SBT stride (2 ray types: radiance, shadow)
			0,  // miss SBT index (radiance miss)
			p0, p1
		);

		if (!payload.hit) break;  // no sky in Phase 1 -- ray escapes, contributes nothing

		const MaterialData& mat = sppm_params.materials[payload.materialIdx];

		if (mat.type == MaterialType::DiffuseLight) {
			pixel.Ld = pixel.Ld + beta * mat.emission;
			break;
		}

		if (mat.type == MaterialType::RoughDielectric) {
			// GGX VNDF microfacet sample -- left byte-for-byte as it was
			// before this function grew Metal/Dielectric/Conductor support,
			// so scene B3's already-verified output can't be perturbed by
			// that generalization (see sppm_sample_delta_material()'s own
			// comment on why RoughDielectric keeps its own separate sampler
			// rather than being folded into that shared helper). attenuation
			// is always mat.albedo (white for scene 11's glass sphere, see
			// scene_builder.cpp's build_cornell_rough_glass): VNDF sampling
			// is self-normalizing for dielectrics, matching the CPU
			// RoughDielectricBxDF's own "attenuation = white" convention
			// (src/shared/bxdfs_conductor.h).
			float3 new_dir;
			if (!sppm_sample_rough_dielectric(dir, payload.normal, mat, seed, new_dir)) break;

			beta = beta * mat.albedo;
			org = payload.hitPoint;
			dir = new_dir;
			continue;
		}

		if (sppm_is_delta_material(mat)) {
			// Generalization beyond Phase 1's original RoughDielectric-only
			// delta set (see sppm_is_delta_material()/sppm_sample_delta_
			// material()'s own comments) -- covers B1/B2's rough metal
			// spheres (MaterialType::RoughMetal since the Metal/rough_metal
			// model-mismatch fix routed them off the plain Metal fuzz-mirror
			// approximation), A1's smooth glass sphere, B4's polished-
			// conductor gold/aluminium, and B8's translucent wax sphere
			// (DiffuseTransmission), none of which recorded a (physically
			// wrong) Lambertian visible point before this change. A glossy
			// (non-smooth) Conductor/RoughMetal instance now falls through
			// to the visible-point path below instead of landing here - see
			// sppm_is_delta_material()'s own comment.
			float3 new_dir, atten;
			if (!sppm_sample_delta_material(dir, payload.normal, mat, seed, new_dir, atten)) break;

			beta = beta * atten;
			org = payload.hitPoint;
			dir = new_dir;
			continue;
		}

		// Non-delta: record the visible point and compute one NEE sample
		// toward the scene's area light(s), then stop -- matches
		// SPPMCameraPass()'s own "record visible point, compute Ld, break"
		// structure exactly. Originally Lambertian-only (Phase 1); a glossy
		// Conductor/RoughMetal instance can reach here too as of this fix -
		// sppm_bsdf_f() below dispatches per mat.type instead of assuming
		// Lambertian.
		pixel.vp_p           = payload.hitPoint;
		pixel.vp_wo          = -normalize(dir);
		pixel.vp_n           = payload.normal;
		pixel.vp_beta        = beta;
		pixel.vp_materialIdx = payload.materialIdx;
		pixel.vp_valid       = true;

		if (sppm_params.numLights > 0 && sppm_params.aliasTable) {
			int slot = int(sppm_rand(seed) * float(sppm_params.numLights));
			if (slot >= (int)sppm_params.numLights) slot = (int)sppm_params.numLights - 1;
			const GpuAliasEntry& entry = sppm_params.aliasTable[slot];
			int light_idx = (sppm_rand(seed) < entry.q) ? slot : entry.alias;
			float selection_pdf = sppm_params.aliasTable[light_idx].pdf;

			int  prim_idx        = sppm_params.lightIndices[light_idx];
			bool is_sphere_light = sppm_params.lightKinds[light_idx] == GpuLightKind::Sphere;

			float geom_pdf = 0.0f, max_dist = 0.0f;
			float3 to_light;
			float3 light_emission;
			if (is_sphere_light) {
				const SphereData& sph = sppm_params.spheres[prim_idx];
				to_light = sppm_sample_sphere_light(sph, pixel.vp_p, seed, geom_pdf, max_dist);
				light_emission = sppm_params.materials[sph.materialIdx].emission;
			} else {
				const QuadData& q = sppm_params.quads[prim_idx];
				to_light = sppm_sample_quad_light(q, pixel.vp_p, seed, geom_pdf, max_dist);
				light_emission = sppm_params.materials[q.materialIdx].emission;
			}

			float light_pdf = selection_pdf * geom_pdf;
			float cos_l = dot(to_light, pixel.vp_n);
			if (light_pdf > 1e-6f && cos_l > 0.0f) {
				// f(wo,wi) via the shared helper above (Lambertian's
				// albedo/pi, or real GGX for a glossy Conductor/RoughMetal
				// visible point). No MIS weight -- see this file's top
				// comment for why SPPM's single-sample NEE doesn't need
				// one, unlike the regular path tracers' NEE.
				float3 bsdf_val = sppm_bsdf_f(mat, pixel.vp_wo, to_light, pixel.vp_n);
				float3 Ld_contrib = beta * bsdf_val * cos_l * light_emission / light_pdf;

				if (sppm_trace_shadow_ray(pixel.vp_p + 0.001f * pixel.vp_n, to_light, max_dist - 0.002f)) {
					pixel.Ld = pixel.Ld + Ld_contrib;
				}
			}
		}
		break;
	}

	sppm_params.framebuffer[pixelIdx] = pixel.Ld;
}

// ============================================================================
// __raygen__sppm_photon_pass (sub-phase 1d): one thread per photon, launched
// as (nPhotons, 1, 1). Direct port of src/shared/sppm.h's SPPMPhotonPass()/
// sppm_adapter.h's sppm_photon_pass_mt(): sample a light emission point +
// direction (SampleLightLe equivalent, see sppm_sample_quad_area/
// sppm_sample_sphere_area above), trace up to maxDepth bounces, and at each
// non-delta hit past depth 0 look up the hash grid (built just before this
// launch, see SPPMPathTracer::render()) and atomically deposit flux into
// nearby visible points' Phi/m. Continues the walk via Russian roulette,
// matching sppm.h's pbrt-v4 formula exactly.
// ============================================================================
extern "C" __global__ void __raygen__sppm_photon_pass() {
	const unsigned int photonIdx = optixGetLaunchIndex().x;
	if (photonIdx >= sppm_params.nPhotons) return;
	if (sppm_params.numLights == 0 || !sppm_params.aliasTable) return;

	unsigned int seed = sppm_pcg(sppm_pcg(photonIdx ^ sppm_params.photonSeedBase) ^ 0x2545F491u);

	// ---- Sample a light + emission point/direction (SampleLightLe) ----
	int slot = int(sppm_rand(seed) * float(sppm_params.numLights));
	if (slot >= (int)sppm_params.numLights) slot = (int)sppm_params.numLights - 1;
	const GpuAliasEntry& entry = sppm_params.aliasTable[slot];
	int light_idx = (sppm_rand(seed) < entry.q) ? slot : entry.alias;
	float selection_pdf = sppm_params.aliasTable[light_idx].pdf;

	int  prim_idx        = sppm_params.lightIndices[light_idx];
	bool is_sphere_light  = sppm_params.lightKinds[light_idx] == GpuLightKind::Sphere;

	float3 p, n_light;
	float area_pdf = 0.0f;
	float3 Le;
	if (is_sphere_light) {
		const SphereData& sph = sppm_params.spheres[prim_idx];
		p = sppm_sample_sphere_area(sph, seed, n_light, area_pdf);
		Le = sppm_params.materials[sph.materialIdx].emission;
	} else {
		const QuadData& q = sppm_params.quads[prim_idx];
		p = sppm_sample_quad_area(q, seed, n_light, area_pdf);
		Le = sppm_params.materials[q.materialIdx].emission;
	}
	float pdf_pos = area_pdf * selection_pdf;
	if (pdf_pos <= 1e-9f) return;

	// Cosine-weighted direction leaving the light into its outward
	// hemisphere -- normalize(n + rand_unit()) identity (see sppm_rand_unit's
	// own comment): same technique this file's camera pass and
	// wavefront_kernels.cu's own Lambertian case already rely on.
	float3 dir = normalize(n_light + sppm_rand_unit(seed));
	float cos_theta = dot(dir, n_light);
	if (cos_theta <= 0.0f) return; // degenerate onb edge case, matches CPU's own early-out
	const float kPi = 3.14159265358979323846f;
	float pdf_dir = cos_theta / kPi;
	if (pdf_dir <= 1e-9f) return;

	// beta = Le * cos_theta / (pdf_pos * pdf_dir) -- matches CPU's
	// `les.L * les.abs_cos_theta / (les.pdf_pos * les.pdf_dir)` exactly.
	float3 beta = Le * (cos_theta / (pdf_pos * pdf_dir));

	float3 org = p + 0.001f * n_light;
	float3 dir_cur = dir;

	for (unsigned int depth = 0; depth < sppm_params.maxDepth; ++depth) {
		SPPMHitPayload payload;
		payload.hit = false;
		unsigned int p0, p1;
		sppmPackPointer(&payload, p0, p1);
		optixTrace(
			sppm_params.traversable,
			org, dir_cur,
			0.001f, 1e30f,
			0.0f,
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_NONE,
			0,  // SBT offset (radiance ray type)
			2,  // SBT stride (2 ray types: radiance, shadow)
			0,  // miss SBT index (radiance miss)
			p0, p1);
		if (!payload.hit) break;

		const MaterialData& mat = sppm_params.materials[payload.materialIdx];

		// Deposit flux at nearby visible points -- depth>0 mirrors CPU (the
		// first bounce is left for the camera pass's own NEE, avoiding
		// double-counting direct illumination). Non-delta == "not one of
		// sppm_is_delta_material()'s types/instances" (generalized from the
		// original Phase 1 "not RoughDielectric" check), matching CPU's
		// `!hit.is_delta_bsdf`. Uses the VISIBLE POINT's own recorded
		// material (vp_materialIdx), not the photon's current hit material
		// -- same as CPU's BSDFf(px.vp_bsdf_id, ...) call. A visible point
		// could be Lambertian OR a glossy Conductor/RoughMetal as of this
		// fix (see sppm_is_delta_material()'s own comment) - sppm_bsdf_f()
		// below dispatches per the visible point's own mat.type instead of
		// assuming Lambertian.
		if (depth > 0 && !sppm_is_delta_material(mat) && mat.type != MaterialType::DiffuseLight) {
			int bucket = sppm_hash_bucket(sppm_params.hashGrid, payload.hitPoint.x, payload.hitPoint.y, payload.hitPoint.z);
			int slotIdx = sppm_params.cellHead[bucket];
			while (slotIdx != -1) {
				int pixelIdx = sppm_params.nodePixel[slotIdx];
				SPPMPixelGPU& vp = sppm_params.pixels[pixelIdx];
				if (vp.vp_valid) {
					float3 d = vp.vp_p - payload.hitPoint;
					if (dot(d, d) <= vp.radius * vp.radius) {
						float3 wi = -dir_cur;
						float cos_wi = dot(wi, vp.vp_n);
						if (cos_wi > 0.0f) {
							const MaterialData& vpMat = sppm_params.materials[vp.vp_materialIdx];
							float3 f = sppm_bsdf_f(vpMat, vp.vp_wo, wi, vp.vp_n);
							float3 phi_add = beta * vp.vp_beta * f;
							atomicAdd(&vp.Phi.x, phi_add.x);
							atomicAdd(&vp.Phi.y, phi_add.y);
							atomicAdd(&vp.Phi.z, phi_add.z);
							atomicAdd(&vp.m, 1);
						}
					}
				}
				slotIdx = sppm_params.nodeNext[slotIdx];
			}
		}

		// Continue the photon random walk.
		float3 beta_new;
		float3 new_dir;
		if (mat.type == MaterialType::RoughDielectric) {
			if (!sppm_sample_rough_dielectric(dir_cur, payload.normal, mat, seed, new_dir)) break;
			beta_new = beta * mat.albedo;
		} else if (mat.type == MaterialType::Metal || mat.type == MaterialType::Dielectric ||
		           mat.type == MaterialType::Conductor || mat.type == MaterialType::RoughMetal ||
		           mat.type == MaterialType::DiffuseTransmission) {
			// Generalization beyond Phase 1's original RoughDielectric-only
			// delta set -- see sppm_sample_delta_material()'s own comment.
			// Dispatched unconditionally by TYPE here (not gated by
			// sppm_is_delta_material()'s smooth/glossy classification, unlike
			// the deposit check above): sppm_sample_delta_material()'s own
			// VNDF-sampling identity makes its Conductor/RoughMetal cases the
			// mathematically correct importance-sampled continuation for
			// EITHER roughness regime, so a glossy hit (which may also have
			// just deposited above) still continues the walk through here
			// exactly like a smooth one always did.
			float3 atten;
			if (!sppm_sample_delta_material(dir_cur, payload.normal, mat, seed, new_dir, atten)) break;
			beta_new = beta * atten;
		} else if (mat.type == MaterialType::DiffuseLight) {
			break; // photon paths don't continue off emitters
		} else {
			// Lambertian (Phase 1's only remaining material): cosine-weighted
			// bounce, f*cosI/pdf collapses to exactly `albedo` (see
			// sppm_rand_unit's own comment) -- same technique
			// wavefront_kernels.cu's Lambertian case already relies on.
			new_dir = normalize(payload.normal + sppm_rand_unit(seed));
			beta_new = beta * mat.albedo;
		}

		// Russian roulette (mirrors SPPMPhotonPass's pbrt-v4 formula exactly:
		// q = max(0, 1 - betaNewMax/betaMax)).
		float betaMax    = fmaxf(beta.x, fmaxf(beta.y, beta.z));
		float betaNewMax = fmaxf(beta_new.x, fmaxf(beta_new.y, beta_new.z));
		float q_rr = (betaMax > 0.0f) ? fmaxf(0.0f, 1.0f - betaNewMax / betaMax) : 0.0f;
		if (sppm_rand(seed) < q_rr) break;
		float invSurv = 1.0f / (1.0f - q_rr);
		beta = beta_new * invSurv;

		org     = payload.hitPoint;
		dir_cur = new_dir;
	}
}
