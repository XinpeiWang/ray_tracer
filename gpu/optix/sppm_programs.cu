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
// sppm_camera_pass_with_sky() (minus the sky branch -- Phase 1 has no sky
// support). Follows a specular chain via RoughDielectricBxDF-style GGX
// resampling (ported from wavefront_kernels.cu's MaterialType::RoughDielectric
// case, the only delta material Phase 1 supports) until it hits a
// non-delta (Lambertian, Phase 1's only non-delta type) surface, records
// that as the pixel's visible point, and computes one NEE sample toward
// the scene's area light(s) via the same power-weighted alias table the
// regular path tracers already use. Unlike the regular path tracers' NEE,
// there's no MIS weight here: SPPM's camera pass calls this exactly once
// per pixel with no competing BSDF-sampled continuation in the same
// estimate (matches sppm_adapter.h's own DirectLight() -- see its comment
// for the fuller reasoning), so no balance/power heuristic is needed.
//
// This is still a single-iteration verification step (see the plan,
// C:\Users\xinpe\.claude\plans\cached-wobbling-ritchie.md, sub-phase 1b):
// pixels[i].Ld is written straight to the framebuffer for visual
// inspection. The real multi-iteration loop (hash grid, photon pass,
// radius contraction) is sub-phases 1c/1d.

#include <optix.h>
#include "sppm_types.h"
#include "optix_types.h"
#include "optix_math_helpers.h"
#include "math_utils.h"   // cpu_gpu_reflect/cpu_gpu_refract
#include "fresnel.h"       // FrDielectric
#include "microfacet.h"    // TrowbridgeReitz

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

extern "C" __global__ void __anyhit__sppm_shadow_sphere() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const SphereData& sphere = sppm_params.spheres[primIdx];
	const MaterialData& mat = sppm_params.materials[sphere.materialIdx];

	if (mat.type == MaterialType::DiffuseLight) {
		optixSetPayload_0(0);
		optixTerminateRay();
		return;
	}
	if (mat.type == MaterialType::RoughDielectric) {
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
	if (mat.type == MaterialType::RoughDielectric) {
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
	pixel.Ld = make_float3(0.0f, 0.0f, 0.0f);
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
			// GGX VNDF microfacet sample -- ported from wavefront_kernels.cu's
			// MaterialType::RoughDielectric case (identical math), Phase 1's
			// only delta material. attenuation is always mat.albedo (white
			// for scene 11's glass sphere, see scene_builder.cpp's
			// build_cornell_rough_glass): VNDF sampling is self-normalizing
			// for dielectrics, matching the CPU RoughDielectricBxDF's own
			// "attenuation = white" convention (src/shared/bxdfs_conductor.h).
			float alpha = sqrtf(mat.fuzz);
			float3 n = payload.normal;
			float3 up_v = (fabsf(n.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
			float3 tan_v = normalize(cross(up_v, n));
			float3 bitan = cross(n, tan_v);
			float3 wi_w = -normalize(dir);
			float wi_x = dot(wi_w, tan_v), wi_y = dot(wi_w, bitan), wi_z = dot(wi_w, n);
			if (wi_z <= 0.0f) break;

			TrowbridgeReitz<float> rd_dist(alpha, alpha);
			float wm_x, wm_y, wm_z;
			rd_dist.Sample_wm(wi_x, wi_y, wi_z, sppm_rand(seed), sppm_rand(seed), wm_x, wm_y, wm_z);
			float rd_dot = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
			float Fr = FrDielectric(rd_dot, mat.ior);

			float3 new_dir;
			if (sppm_rand(seed) < Fr) {
				float wo_x = 2.0f*rd_dot*wm_x - wi_x;
				float wo_y = 2.0f*rd_dot*wm_y - wi_y;
				float wo_z = 2.0f*rd_dot*wm_z - wi_z;
				new_dir = normalize(wo_x*tan_v + wo_y*bitan + wo_z*n);
			} else {
				float3 wm_world = wm_x*tan_v + wm_y*bitan + wm_z*n;
				float eta = dot(dir, payload.normal) < 0.0f ? (1.0f / mat.ior) : mat.ior;
				new_dir = cpu_gpu_refract<float3, float>(normalize(dir), wm_world, eta);
			}

			beta = beta * mat.albedo;
			org = payload.hitPoint;
			dir = new_dir;
			continue;
		}

		// Non-delta (Lambertian, Phase 1's only non-delta material): record
		// the visible point and compute one NEE sample toward the scene's
		// area light(s), then stop -- matches SPPMCameraPass()'s own
		// "record visible point, compute Ld, break" structure exactly.
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
			bool is_sphere_light = sppm_params.isLightSphere[light_idx];

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
				// Lambertian f(wo,wi) = albedo/pi (this codebase's
				// scattering_pdf() convention, matching sppm_bsdf_f's own
				// lambertian branch on CPU). No MIS weight -- see this
				// file's top comment for why SPPM's single-sample NEE
				// doesn't need one, unlike the regular path tracers' NEE.
				const float inv_pi = 1.0f / 3.14159265358979323846f;
				float3 bsdf_val = mat.albedo * inv_pi;
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
