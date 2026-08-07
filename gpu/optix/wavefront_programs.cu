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
	float  mediumTFar; // MaterialType::Medium only - see HitWorkItem::mediumTFar
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
		h.mediumTFar  = payload.mediumTFar;
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
// __intersection__wf_bilinear_patch / __closesthit__wf_bilinear_patch
//
// Wavefront-native duplicate of optix_intersection_bilinear_patch.h's
// __intersection__bilinear_patch (same quadratic-solve math, wf_params
// instead of the recursive path's params global) - same cross-module
// payload-type mismatch reason as __intersection__wf_sphere/wf_quad above
// for why this isn't just reused directly from the recursive module.
// closesthit only fills geometry (mirrors wf_sphere/wf_quad) - material
// shading happens uniformly in evaluate_materials, keyed off mat.type, not
// per-geometry-type.
// ============================================================================

extern "C" __global__ void __intersection__wf_bilinear_patch() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const BilinearPatchData& patch = wf_params.bilinearPatches[primIdx];

	const float3 ro = optixGetWorldRayOrigin();
	const float3 rd = optixGetWorldRayDirection();
	const float ray_tmin = optixGetRayTmin();
	const float ray_tmax = optixGetRayTmax();

	const float3 p00 = patch.p00;
	const float3 p10 = patch.p10;
	const float3 p01 = patch.p01;
	const float3 p11 = patch.p11;

	const float3 e0 = p10 - p00;
	const float3 e1 = p01 - p11;
	const float3 e2 = p00 - ro;
	const float3 e3 = p10 - ro;

	const float a = dot(cross(e0, e1), rd);
	const float c = dot(cross(e2, rd), p01 - p00);
	const float b = dot(cross(e3, rd), p11 - p10) - (a + c);

	float u1, u2;
	if (a == 0.0f) {
		if (b == 0.0f) return;
		u1 = u2 = -c / b;
	} else {
		const double disc = (double)b * (double)b - 4.0 * (double)a * (double)c;
		if (disc < 0.0) return;
		const double sqrt_disc = sqrt(disc);
		const double q = (b < 0.0f) ? -0.5 * ((double)b - sqrt_disc) : -0.5 * ((double)b + sqrt_disc);
		u1 = (float)(q / (double)a);
		u2 = (q != 0.0) ? (float)((double)c / q) : 0.0f;
		if (u1 > u2) { const float tmp = u1; u1 = u2; u2 = tmp; }
	}

	const float eps = 4e-6f *
		(fmaxf(fmaxf(fabsf(ro.x), fabsf(ro.y)), fabsf(ro.z)) +
		 fmaxf(fmaxf(fabsf(rd.x), fabsf(rd.y)), fabsf(rd.z)) +
		 fmaxf(fmaxf(fabsf(p00.x), fabsf(p00.y)), fabsf(p00.z)) +
		 fmaxf(fmaxf(fabsf(p10.x), fabsf(p10.y)), fabsf(p10.z)) +
		 fmaxf(fmaxf(fabsf(p01.x), fabsf(p01.y)), fabsf(p01.z)) +
		 fmaxf(fmaxf(fabsf(p11.x), fabsf(p11.y)), fabsf(p11.z)));

	float t_hit = ray_tmax;
	float u_hit = -1.0f, v_hit = -1.0f;
	bool found = false;

	for (int i = 0; i < 2; ++i) {
		const float u_cand = (i == 0) ? u1 : u2;
		if (i == 1 && u2 == u1) continue;
		if (u_cand < 0.0f || u_cand > 1.0f) continue;

		const float3 uo   = lerp(p00, p10, u_cand);
		const float3 p11u = lerp(p01, p11, u_cand);
		const float3 ud   = p11u - uo;

		const float3 delta = uo - ro;
		const float3 perp  = cross(rd, ud);
		const float p2 = dot(perp, perp);
		if (p2 < 1e-20f) continue;

		const float v_num = dot(delta, cross(rd, perp));
		const float t_num = dot(delta, cross(ud, perp));

		const float t_cand = t_num / p2;
		const float v_cand = v_num / p2;

		if (t_num > p2 * eps && v_cand >= 0.0f && v_cand <= 1.0f &&
			t_cand < t_hit && t_cand >= ray_tmin && t_cand <= ray_tmax) {
			t_hit = t_cand;
			u_hit = u_cand;
			v_hit = v_cand;
			found = true;
		}
	}

	if (!found) return;

	optixReportIntersection(
		t_hit, 0,
		__float_as_int(u_hit),
		__float_as_int(v_hit),
		0, 0);
}

extern "C" __global__ void __closesthit__wf_bilinear_patch() {
	WfHitPayload* payload = (WfHitPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	const int primIdx = optixGetPrimitiveIndex();
	const BilinearPatchData& patch = wf_params.bilinearPatches[primIdx];

	const float u = __int_as_float(optixGetAttribute_0());
	const float v = __int_as_float(optixGetAttribute_1());

	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float  t_hit    = optixGetRayTmax();

	float3 hit_point = ray_orig + t_hit * ray_dir;

	const float3 pu0 = lerp(patch.p00, patch.p01, v);
	const float3 pu1 = lerp(patch.p10, patch.p11, v);
	const float3 dpdu = pu1 - pu0;
	const float3 dpdv = lerp(patch.p01, patch.p11, u) - lerp(patch.p00, patch.p10, u);

	float3 geom_normal = normalize(cross(dpdu, dpdv));
	bool front_face = dot(ray_dir, geom_normal) < 0.0f;
	float3 normal = front_face ? geom_normal : -geom_normal;

	payload->hitPoint    = hit_point;
	payload->normal      = normal;
	payload->t           = t_hit;
	payload->materialIdx = patch.materialIdx;
	payload->geomType    = 2;
	payload->hit         = true;
	payload->mediumTFar  = 0.0f;
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
	payload->mediumTFar  = 0.0f;

	// MaterialType::Medium: override with the entry (near) / exit (far) roots
	// recomputed relative to the CURRENT ray origin, matching optix_intersection_
	// sphere.h's __closesthit__sphere Medium case. optixGetRayTmax() alone isn't
	// enough - it reports whichever single root the intersection program found
	// valid, which is the FAR root when this ray already starts inside the
	// sphere (e.g. continuing after a prior in-medium scatter). Recomputing both
	// roots here handles that re-entry case the same way the recursive path does.
	const MaterialData& sph_mat = wf_params.materials[sph.materialIdx];
	if (sph_mat.type == MaterialType::Medium) {
		float3 unit_dir = normalize(ray_dir);
		float3 oc2 = ray_orig - sph.center;
		float half_b2 = dot(oc2, unit_dir);
		float c2 = dot(oc2, oc2) - sph.radius * sph.radius;
		float disc2 = fmaxf(0.0f, half_b2 * half_b2 - c2);
		float sq2 = sqrtf(disc2);
		float t_near = fmaxf(0.0f, -half_b2 - sq2);
		float t_far  = -half_b2 + sq2;
		payload->t          = t_near;
		payload->hitPoint   = ray_orig + t_near * unit_dir;
		payload->mediumTFar = t_far;
	}
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
