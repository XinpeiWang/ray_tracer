// wavefront_programs.cu
// OptiX device programs for the wavefront GPU path tracer.
//
// Unlike optix_programs.cu (recursive), these programs do NOT shade or scatter.
// They only drive the intersection phase of the wavefront loop:
//
//   __raygen__wf_intersect  - Dispatches one optixTrace per RayWorkItem.
//                             One thread per ray in the queue (launch dim = numRays).
//   __closesthit__wf_sphere / __closesthit__wf_quad
//                           - Append a HitWorkItem to hitQueue.
//   __miss__wf_radiance     - Append a MissWorkItem to missQueue.
//
// Shadow phase (separate optixLaunch):
//   __raygen__wf_shadow     - Dispatches one shadow-test per ShadowRayWorkItem.
//   __anyhit__wf_shadow     - Terminates immediately (any hit = occluded).
//   __miss__wf_shadow       - Writes occluded[rayIndex] = false.
//
// The host (wavefront_path_tracer.cpp) drives two launches per bounce:
//   1. optixLaunch(intersectPipeline, numRays)   -> fills hitQueue + missQueue
//   2. optixLaunch(shadowPipeline,   numShadow)  -> fills occluded[] array

#include <optix.h>
#include "wavefront_types.h"
#include "optix_types.h"
#include "optix_math_helpers.h"

// Wavefront launch params live in constant memory.
extern "C" { __constant__ WavefrontLaunchParams wf_params; }

// Per-shadow-ray occlusion output (device pointer passed via launch params extension).
// We reuse a float3* slot in WavefrontLaunchParams — see wavefront_path_tracer.cpp
// which passes d_occluded via the misuse of framebuffer during the shadow pass.
// Actually we pass occluded as a separate bool* stored in wf_params.framebuffer cast.
// For clarity we just read it from the framebuffer pointer (see shadow launch setup).

// ============================================================================
// Utility: encode/decode 64-bit pointers as two 32-bit payload registers
// ============================================================================
static __device__ __forceinline__ void packPointer(void* ptr, unsigned int& p0, unsigned int& p1) {
	const unsigned long long up = (unsigned long long)ptr;
	p0 = (unsigned int)(up >> 32);
	p1 = (unsigned int)(up & 0xFFFFFFFFull);
}
static __device__ __forceinline__ void* unpackPointer(unsigned int p0, unsigned int p1) {
	return (void*)((unsigned long long)p0 << 32 | (unsigned long long)p1);
}

// ============================================================================
// Payload structs (passed by pointer via p0/p1)
// ============================================================================

struct WfHitPayload {
	float3 hitPoint;
	float3 normal;
	float  t;
	int    materialIdx;
	int    geomType;   // 0 = sphere, 1 = quad
	bool   hit;
};

struct WfShadowPayload {
	bool occluded;
};

// ============================================================================
// __raygen__wf_intersect
//   Launch dimensions: (numRaysInQueue, 1, 1).
//   Each thread handles one RayWorkItem from rayQueue.
// ============================================================================
extern "C" __global__ void __raygen__wf_intersect() {
	const unsigned int rayIdx = optixGetLaunchIndex().x;
	// Bounds check (launch size is numRays, set by host)
	const WorkQueue<RayWorkItem>& rq = wf_params.rayQueue;
	if ((int)rayIdx >= *rq.counter) return;

	const RayWorkItem& ray = rq.items[rayIdx];

	WfHitPayload payload;
	payload.hit = false;

	unsigned int p0, p1;
	packPointer(&payload, p0, p1);

	optixTrace(
		wf_params.traversable,
		ray.origin,
		ray.direction,
		ray.tMin,
		ray.tMax,
		0.0f,                             // rayTime
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_NONE,
		0,                                // SBT offset (radiance)
		2,                                // SBT stride (sphere=0, quad=1)
		0,                                // miss SBT index (radiance miss)
		p0, p1
	);

	if (payload.hit) {
		HitWorkItem h;
		h.hitPoint    = payload.hitPoint;
		h.normal      = payload.normal;
		h.t           = payload.t;
		h.materialIdx = payload.materialIdx;
		h.geomType    = payload.geomType;
		h.rayOrigin   = ray.origin;
		h.rayDir      = ray.direction;
		for (int i = 0; i < kWFNWavelengths; ++i) {
			h.throughput[i]      = ray.throughput[i];
			h.radiance[i]        = ray.radiance[i];
			h.wavelengths[i]     = ray.wavelengths[i];
			h.wavelength_pdfs[i] = ray.wavelength_pdfs[i];
		}
		h.seed        = ray.seed;
		h.pixelIndex  = ray.pixelIndex;
		h.depth       = ray.depth;
		h.specular_bounce = ray.specular_bounce;
		wf_params.hitQueue.push(h);
	} else {
		MissWorkItem m;
		for (int i = 0; i < kWFNWavelengths; ++i) {
			m.throughput[i]      = ray.throughput[i];
			m.wavelengths[i]     = ray.wavelengths[i];
			m.wavelength_pdfs[i] = ray.wavelength_pdfs[i];
		}
		m.rayDir      = ray.direction;
		m.pixelIndex  = ray.pixelIndex;
		wf_params.missQueue.push(m);
	}
}

// ============================================================================
// __intersection__wf_sphere / __intersection__wf_quad
//
// Wavefront-native duplicates of optix_intersection_{sphere,quad}.h's
// __intersection__sphere/__intersection__quad (same math, wf_params instead
// of the recursive path's params global). createProgramGroups() originally
// reused those programs directly from the recursive path's module (custom
// AABB primitives need *some* intersection program, and wavefront_programs.cu
// never defined its own) - but OptiX rejects combining intersection/
// closest-hit programs from two separately-compiled modules with different
// pipelineCompileOptions.numPayloadValues (12 registers for the recursive
// path's PathTracingPayload vs 2 here) into the same hit group: "could not
// be resolved to a common payloadType". Compiling our own copies here, under
// this module's own pipeline options, sidesteps the mismatch entirely -
// neither program actually touches payload registers (attributes only), so
// there's no behavioral difference, just no more cross-module payload
// conflict.
// ============================================================================

extern "C" __global__ void __intersection__wf_sphere() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const SphereData& sphere = wf_params.spheres[primIdx];

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

extern "C" __global__ void __intersection__wf_quad() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const QuadData& quad = wf_params.quads[primIdx];

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

// ============================================================================
// __closesthit__wf_sphere
// ============================================================================
extern "C" __global__ void __closesthit__wf_sphere() {
	WfHitPayload* payload = (WfHitPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	// Custom intersection stores sphere index in attribute 0.
	const int sphereIdx = optixGetPrimitiveIndex();

	// Reconstruct hit point and normal from ray + t.
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float  t_hit    = optixGetRayTmax();

	float3 hit_point = ray_orig + t_hit * ray_dir;

	// Sphere normal
	const SphereData& sph = wf_params.spheres[sphereIdx];
	float3 outward_normal = normalize(hit_point - sph.center);
	// Flip to face the ray
	bool front_face = dot(ray_dir, outward_normal) < 0.0f;
	float3 normal = front_face ? outward_normal : -outward_normal;

	payload->hitPoint    = hit_point;
	payload->normal      = normal;
	payload->t           = t_hit;
	payload->materialIdx = sph.materialIdx;
	payload->geomType    = 0;
	payload->hit         = true;
}

// ============================================================================
// __closesthit__wf_quad
// ============================================================================
extern "C" __global__ void __closesthit__wf_quad() {
	WfHitPayload* payload = (WfHitPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	const int quadIdx = optixGetPrimitiveIndex();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float  t_hit    = optixGetRayTmax();

	float3 hit_point = ray_orig + t_hit * ray_dir;

	const QuadData& q = wf_params.quads[quadIdx];
	bool front_face = dot(ray_dir, q.normal) < 0.0f;
	float3 normal = front_face ? q.normal : -q.normal;

	payload->hitPoint    = hit_point;
	payload->normal      = normal;
	payload->t           = t_hit;
	payload->materialIdx = q.materialIdx;
	payload->geomType    = 1;
	payload->hit         = true;
}

// ============================================================================
// __miss__wf_radiance
//   Ray escaped — payload.hit stays false; raygen will add to missQueue.
// ============================================================================
extern "C" __global__ void __miss__wf_radiance() {
	// nothing — hit remains false
}

// ============================================================================
// Shadow pass programs
// ============================================================================

// Shadow launch params: reuse wf_params but a separate optixLaunch with a
// shadow pipeline.  We store occluded[] as a bool* in wf_params.framebuffer
// during the shadow pass launch (the host casts it; the bool array has exactly
// numShadow entries allocated separately).

// __raygen__wf_shadow: one thread per shadow ray.
extern "C" __global__ void __raygen__wf_shadow() {
	const unsigned int idx = optixGetLaunchIndex().x;
	const WorkQueue<ShadowRayWorkItem>& sq = wf_params.shadowQueue;
	if ((int)idx >= *sq.counter) return;

	const ShadowRayWorkItem& s = sq.items[idx];

	WfShadowPayload sp;
	sp.occluded = false;

	unsigned int p0, p1;
	packPointer(&sp, p0, p1);

	optixTrace(
		wf_params.traversable,
		s.origin,
		s.direction,
		0.001f,
		s.tMax,
		0.0f,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
		1,                  // SBT offset (shadow)
		2,                  // SBT stride
		1,                  // miss SBT index (shadow miss)
		p0, p1
	);

	// Write result into the bool array (reused from framebuffer pointer during shadow pass)
	bool* occluded = (bool*)wf_params.framebuffer;
	occluded[idx]  = sp.occluded;
}

// __anyhit__wf_shadow: terminate immediately on first hit.
extern "C" __global__ void __anyhit__wf_shadow() {
	WfShadowPayload* sp = (WfShadowPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());
	sp->occluded = true;
	optixTerminateRay();
}

// __miss__wf_shadow: ray reached tMax without hitting anything — not occluded.
extern "C" __global__ void __miss__wf_shadow() {
	// occluded stays false (its default)
}
