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
// Phase 1a: __raygen__sppm_camera_pass here is a TRIVIAL placeholder --
// one thread per pixel, traces a primary ray, writes white on hit / black
// on miss directly to the framebuffer. No SPPM logic yet. The point of
// this sub-phase is proving the pipeline/SBT/PTX-JIT machinery works
// against the real uploaded scene (scene 11) before any camera-pass/
// photon-pass math is written -- see the plan
// (C:\Users\xinpe\.claude\plans\cached-wobbling-ritchie.md)'s sub-phase
// 1a description. Real camera-pass logic (visible points, NEE) replaces
// this raygen body in sub-phase 1b.

#include <optix.h>
#include "sppm_types.h"
#include "optix_types.h"
#include "optix_math_helpers.h"

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

// Payload for the (currently trivial) camera-pass raygen. Extended in
// sub-phase 1b with whatever fields the real camera-pass bounce loop needs
// (throughput, depth-continuation state, etc.).
struct SPPMHitPayload {
	float3 hitPoint;
	float3 normal;
	int    materialIdx;
	bool   hit;
};

// ============================================================================
// __intersection__sppm_sphere / __intersection__sppm_quad
//
// Direct copies of wavefront_programs.cu's __intersection__wf_sphere/
// __intersection__wf_quad (identical math), including passing 4 dummy
// attribute values via optixReportIntersection even though neither
// closesthit program below reads them back via optixGetAttribute_N() --
// matching this module's own numAttributeValues=4 pipeline setting (see
// sppm_path_tracer.cpp), mirroring wavefront/recursive's own established
// pattern rather than risking an attribute-count mismatch with an
// untested reduced-attribute path.
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

// ============================================================================
// __closesthit__sppm_sphere / __closesthit__sppm_quad
// ============================================================================

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

// ============================================================================
// __miss__sppm_radiance -- ray escaped, payload.hit stays false
// ============================================================================
extern "C" __global__ void __miss__sppm_radiance() {
	// nothing -- hit remains false
}

// ============================================================================
// __raygen__sppm_camera_pass (Phase 1a placeholder -- see file header)
// ============================================================================
extern "C" __global__ void __raygen__sppm_camera_pass() {
	const uint3 idx = optixGetLaunchIndex();
	const unsigned int width  = sppm_params.width;
	const unsigned int height = sppm_params.height;
	if (idx.x >= width || idx.y >= height) return;
	const unsigned int pixelIdx = idx.y * width + idx.x;

	// Simple pinhole primary ray through the pixel center (no jitter/DOF --
	// Phase 1a doesn't need antialiasing, just "does tracing against the
	// real uploaded scene work at all"), matching GpuCameraParams'
	// lower_left_corner/horizontal/vertical viewport convention used
	// elsewhere (e.g. optix_device_helpers.h's generate_primary_ray()).
	float s = (float(idx.x) + 0.5f) / float(width);
	float t = (float(idx.y) + 0.5f) / float(height);
	float3 dir = sppm_params.camera.lower_left_corner
	           + s * sppm_params.camera.horizontal
	           + t * sppm_params.camera.vertical
	           - sppm_params.camera.origin;

	SPPMHitPayload payload;
	payload.hit = false;

	unsigned int p0, p1;
	sppmPackPointer(&payload, p0, p1);

	optixTrace(
		sppm_params.traversable,
		sppm_params.camera.origin,
		normalize(dir),
		0.001f,
		1e30f,
		0.0f,                             // rayTime
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_NONE,
		0,                                 // SBT offset (single ray type in Phase 1a)
		1,                                 // SBT stride (single ray type)
		0,                                 // miss SBT index
		p0, p1
	);

	sppm_params.framebuffer[pixelIdx] = payload.hit
		? make_float3(1.0f, 1.0f, 1.0f)
		: make_float3(0.0f, 0.0f, 0.0f);
}
