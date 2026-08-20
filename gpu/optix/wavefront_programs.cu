// wavefront_programs.cu
// OptiX device programs for the wavefront GPU path tracer.
//
// Unlike optix_programs.cu (recursive), these programs do NOT shade or scatter.
// They only drive the intersection phase of the wavefront loop:
//
//   __raygen__wf_intersect  - Dispatches one optixTrace per RayWorkItem.
//                             One thread per ray in the queue (launch dim = numRays).
//   __closesthit__wf_sphere / __closesthit__wf_quad
//                           - Fill the hit payload; __raygen__wf_intersect
//                             then builds a HitWorkItem and routes it to
//                             hitQueue or simpleHitQueue (Lambertian/Metal)
//                             based on material type.
//   __miss__wf_radiance     - Append a MissWorkItem to missQueue.
//
// Shadow phase (separate optixLaunch):
//   __raygen__wf_shadow          - Dispatches one shadow-test per ShadowRayWorkItem.
//   __anyhit__wf_shadow_sphere / _quad / _bilinear_patch / _triangle
//                                 - One per geometry type (mirrors
//                                   optix_anyhit_shadow.h): DiffuseLight is
//                                   NOT an occluder, transmissive materials
//                                   are ignored, everything else terminates
//                                   the ray with occluded=true.
//   __miss__wf_shadow            - Writes occluded[rayIndex] = false.
//
// The host (wavefront_path_tracer.cpp) drives two launches per bounce:
//   1. optixLaunch(intersectPipeline, numRays)   -> fills hitQueue/simpleHitQueue + missQueue
//   2. optixLaunch(shadowPipeline,   numShadow)  -> fills occluded[] array

#include <optix.h>
#include "wavefront_types.h"
#include "optix_types.h"
#include "optix_math_helpers.h"

// Wavefront launch params live in constant memory.
extern "C" { __constant__ WavefrontLaunchParams wf_params; }

// Object instancing, wavefront's copy of what optix_intersection_triangle.h
// and optix_intersection_sphere.h do on the recursive path.
//
// A primitive index restarts at 0 inside every GAS, so an instance
// definition's triangle 0 and the scene's triangle 0 are different triangles.
// This recovers the global index. The entry is SIGNED and -1 is a real
// sentinel meaning "already world space" (the scene's own two instances),
// which is also what gates the object-to-world normal transform below - so
// callers keep the raw value too, not just the clamped base.
__device__ __forceinline__ int wf_instance_base() {
	return wf_params.instancePrimBase
		? wf_params.instancePrimBase[optixGetInstanceId()] : -1;
}
__device__ __forceinline__ unsigned int wf_prim_base(int instBase) {
	return (instBase >= 0) ? (unsigned int)instBase : 0u;
}

// Alpha-cutout test (OBJ/.mtl map_d), wavefront's own copy of
// optix_device_helpers.h's passes_alpha_cutout()/sample_texture() -
// duplicated rather than shared because wavefront_programs.cu and the
// recursive path's optix_programs.cu are compiled as two separate OptiX
// modules (see this file's own header comment on why wf_sphere/wf_quad/
// wf_triangle intersection programs are likewise wavefront-native copies,
// not cross-module reuse). Only handles TextureKind::Image (alpha masks
// are always a loaded map_d file, never Checker/Noise), so this is
// deliberately narrower than the full sample_texture() - matches its
// Image-kind branch exactly. Returns true (keep the hit) for
// alphaMaskTexIdx < 0, i.e. the overwhelming majority of triangles.
__device__ __forceinline__ bool wf_passes_alpha_cutout(int alphaMaskTexIdx, float u, float v) {
	if (alphaMaskTexIdx < 0) return true;
	const TextureData& tex = wf_params.textures[alphaMaskTexIdx];
	if (tex.width <= 0 || tex.height <= 0) return true;
	const float uc = fminf(fmaxf(u, 0.0f), 1.0f);
	const float vc = 1.0f - fminf(fmaxf(v, 0.0f), 1.0f);
	const int i = min(static_cast<int>(uc * tex.width), tex.width - 1);
	const int j = min(static_cast<int>(vc * tex.height), tex.height - 1);
	const unsigned char* px = wf_params.texturePixels + tex.pixelOffset + (j * tex.width + i) * 3;
	constexpr float kAlphaCutoutThreshold = 0.5f;
	return (px[0] * (1.0f / 255.0f)) >= kAlphaCutoutThreshold;
}

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
// Disk/Cylinder object<->world transform helpers (Phase 4c) - wavefront-
// native duplicates of optix_intersection_disk_cylinder.h's identically-
// named recursive-backend functions (dc_apply_point/dc_apply_vector/
// dc_apply_normal_from_w2o), applying DiskData::o2w/w2o and CylinderData::
// o2w/w2o (row-major 3x4 affine, implicit [0,0,0,1] bottom row) by hand.
// Declared here (before wavefront_probe.h's own #include below) because
// that file's __closesthit__wf_probe_disk/probe_cylinder need them too, not
// just this file's own __intersection__wf_disk/wf_cylinder further down.
// ============================================================================
__device__ __forceinline__ float3 wf_dc_apply_point(const float m[12], const float3& p) {
	return make_float3(
		m[0] * p.x + m[1] * p.y + m[2]  * p.z + m[3],
		m[4] * p.x + m[5] * p.y + m[6]  * p.z + m[7],
		m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]);
}

__device__ __forceinline__ float3 wf_dc_apply_vector(const float m[12], const float3& v) {
	return make_float3(
		m[0] * v.x + m[1] * v.y + m[2]  * v.z,
		m[4] * v.x + m[5] * v.y + m[6]  * v.z,
		m[8] * v.x + m[9] * v.y + m[10] * v.z);
}

__device__ __forceinline__ float3 wf_dc_apply_normal_from_w2o(const float w2o[12], const float3& n) {
	return make_float3(
		w2o[0] * n.x + w2o[4] * n.y + w2o[8]  * n.z,
		w2o[1] * n.x + w2o[5] * n.y + w2o[9]  * n.z,
		w2o[2] * n.x + w2o[6] * n.y + w2o[10] * n.z);
}

// ============================================================================
// Payload structs (passed by pointer via p0/p1)
// ============================================================================

struct WfHitPayload {
	float3 hitPoint;
	float3 normal;
	float  t;
	int    materialIdx;
	int    geomType;   // 0 = sphere, 1 = quad, 4 = disk, 5 = cylinder
	bool   hit;
	float  mediumTFar; // MaterialType::Medium/DielectricMedium only - see HitWorkItem::mediumTFar
	int    frontFace;  // see HitWorkItem::frontFace
	float3 objNormal;  // sphere-only - see HitWorkItem::objNormal
	float  uv_u, uv_v; // see HitWorkItem::uv_u/uv_v
};

struct WfShadowPayload {
	bool occluded;
};

// BSSRDF probe walk (MaterialType::Subsurface, wavefront backend Phase 2) -
// needs wf_instance_base()/wf_prim_base()/packPointer()/unpackPointer()
// above, so this include has to stay below them. See wavefront_probe.h's
// own header comment for the full design.
#include "wavefront_probe.h"

// ============================================================================
// __raygen__wf_intersect
//   Launch dimensions: (numRaysInQueue, 1, 1).
//   Each thread handles one RayWorkItem from rayQueue.
// ============================================================================
extern "C" __global__ void __raygen__wf_intersect() {
	const unsigned int rayIdx = optixGetLaunchIndex().x;
	// Bounds check (launch size is numRays, set by host). Must also guard
	// against rq.capacity, not just the live *rq.counter: WorkQueue::push()
	// (wavefront_types.h) increments the counter unconditionally even once
	// the backing array is full - it only skips the actual item write past
	// capacity (returns -1, "should not happen if capacity >= numPixels").
	// A queue that legitimately overflows (see __raygen__wf_shadow's own
	// version of this comment for a confirmed real case) would otherwise
	// leave *rq.counter reporting more live entries than were ever written,
	// so trusting it alone here reads rq.items[] past its cudaMalloc'd end -
	// the exact "illegal memory access" this guard prevents.
	const WorkQueue<RayWorkItem>& rq = wf_params.rayQueue;
	if ((int)rayIdx >= *rq.counter || (int)rayIdx >= rq.capacity) return;

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
		h.frontFace   = payload.frontFace;
		h.objNormal   = payload.objNormal;
		h.uv_u        = payload.uv_u;
		h.uv_v        = payload.uv_v;
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
		// Route cheap materials (no texture/layered-BxDF work) into their
		// own queue so evaluate_materials_simple() can process them without
		// the big switch's register pressure - see WavefrontQueues::
		// simpleHitQueue's comment (wavefront_types.h).
		const MaterialType mt = wf_params.materials[h.materialIdx].type;
		if (mt == MaterialType::Lambertian || mt == MaterialType::Metal) {
			wf_params.simpleHitQueue.push(h);
		} else if (mt == MaterialType::Dielectric || mt == MaterialType::RoughDielectric) {
			wf_params.dielectricHitQueue.push(h);
		} else {
			wf_params.hitQueue.push(h);
		}
	} else {
		MissWorkItem m;
		for (int i = 0; i < kWFNWavelengths; ++i) {
			m.throughput[i]      = ray.throughput[i];
			m.radiance[i]        = ray.radiance[i];
			m.wavelengths[i]     = ray.wavelengths[i];
			m.wavelength_pdfs[i] = ray.wavelength_pdfs[i];
		}
		m.rayDir      = ray.direction;
		m.pixelIndex  = ray.pixelIndex;
		m.brdf_pdf    = ray.brdf_pdf;
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

// Wavefront-native copy of optix_intersection_sphere.h's box_slab_intersect()/
// box_face_normal() - same reason wf_sphere/wf_quad/wf_triangle above are
// duplicated rather than shared (separately-compiled OptiX module, see this
// file's own header comment). See GpuMediumShapeKind's comment in
// optix_types.h for what shapeKind==Box means.
__device__ __forceinline__ bool wf_box_slab_intersect(
		const float3& ray_orig, const float3& ray_dir,
		const float3& bmin, const float3& bmax,
		float& t_near_out, float& t_far_out) {
	float t0 = -1e30f, t1 = 1e30f;
	float invd, tn, tf;

	invd = (ray_dir.x != 0.0f) ? (1.0f / ray_dir.x) : 1e30f;
	tn = (bmin.x - ray_orig.x) * invd; tf = (bmax.x - ray_orig.x) * invd;
	if (tn > tf) { float tmp = tn; tn = tf; tf = tmp; }
	t0 = fmaxf(t0, tn); t1 = fminf(t1, tf);

	invd = (ray_dir.y != 0.0f) ? (1.0f / ray_dir.y) : 1e30f;
	tn = (bmin.y - ray_orig.y) * invd; tf = (bmax.y - ray_orig.y) * invd;
	if (tn > tf) { float tmp = tn; tn = tf; tf = tmp; }
	t0 = fmaxf(t0, tn); t1 = fminf(t1, tf);

	invd = (ray_dir.z != 0.0f) ? (1.0f / ray_dir.z) : 1e30f;
	tn = (bmin.z - ray_orig.z) * invd; tf = (bmax.z - ray_orig.z) * invd;
	if (tn > tf) { float tmp = tn; tn = tf; tf = tmp; }
	t0 = fmaxf(t0, tn); t1 = fminf(t1, tf);

	t_near_out = t0;
	t_far_out = t1;
	return t0 <= t1;
}

__device__ __forceinline__ float3 wf_box_face_normal(const float3& p, const float3& bmin, const float3& bmax) {
	float best = fabsf(p.x - bmin.x);
	float3 n = make_float3(-1.0f, 0.0f, 0.0f);
	float d;
	d = fabsf(p.x - bmax.x); if (d < best) { best = d; n = make_float3(1.0f, 0.0f, 0.0f); }
	d = fabsf(p.y - bmin.y); if (d < best) { best = d; n = make_float3(0.0f, -1.0f, 0.0f); }
	d = fabsf(p.y - bmax.y); if (d < best) { best = d; n = make_float3(0.0f, 1.0f, 0.0f); }
	d = fabsf(p.z - bmin.z); if (d < best) { best = d; n = make_float3(0.0f, 0.0f, -1.0f); }
	d = fabsf(p.z - bmax.z); if (d < best) { best = d; n = make_float3(0.0f, 0.0f, 1.0f); }
	return n;
}

extern "C" __global__ void __intersection__wf_sphere() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const SphereData& sphere = wf_params.spheres[wf_prim_base(wf_instance_base()) + primIdx];

	// OBJECT space - the sphere's centre/radius live in whatever space its GAS
	// was built in. Not a branch for instanced geometry: the scene's own GAS
	// instances carry an identity transform, for which OptiX's object ray IS
	// the world ray, so the non-instanced path is unchanged by construction.
	// t needs no rescaling either, since OptiX transforms the direction
	// without renormalising it.
	const float3 ray_orig = optixGetObjectRayOrigin();
	const float3 ray_dir  = optixGetObjectRayDirection();
	const float  ray_tmin = optixGetRayTmin();
	const float  ray_tmax = optixGetRayTmax();

	if (sphere.shapeKind == GpuMediumShapeKind::Box) {
		// See optix_intersection_sphere.h's own box branch - identical dual-
		// root selection, attributes unused (box bounds are static, re-read
		// directly by __closesthit__wf_sphere below instead).
		float t_near, t_far;
		if (!wf_box_slab_intersect(ray_orig, ray_dir, sphere.boxMin, sphere.boxMax, t_near, t_far))
			return;
		float root = t_near;
		if (root < ray_tmin || root > ray_tmax) {
			root = t_far;
			if (root < ray_tmin || root > ray_tmax)
				return;
		}
		optixReportIntersection(root, 0, 0, 0, 0, 0);
		return;
	}

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
	// See __closesthit__wf_quad's comment on frontFace - same missing
	// initialization here.
	payload->frontFace   = front_face ? 1 : 0;
	// No texturing support for bilinear patches on either backend yet - see
	// __closesthit__wf_quad's own comment on the same convention.
	payload->uv_u        = 0.0f;
	payload->uv_v        = 0.0f;
}

// ============================================================================
// __intersection__wf_triangle / __closesthit__wf_triangle
//
// Wavefront-native duplicate of optix_intersection_triangle.h's
// __intersection__triangle (same watertight Woop/Moller-Trumbore math,
// wf_params instead of the recursive path's params global) - same cross-
// module payload-type mismatch reason as wf_sphere/wf_quad/wf_bilinear_patch
// above. Flat shading only (no per-vertex normals - see TriangleData).
// ============================================================================

extern "C" __global__ void __intersection__wf_triangle() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const TriangleData& tri = wf_params.triangles[wf_prim_base(wf_instance_base()) + primIdx];

	// Object space, for the same reason as __intersection__wf_sphere above -
	// identical to world space on the non-instanced path, since those
	// instances carry an identity transform.
	const float3 ro = optixGetObjectRayOrigin();
	const float3 rd = optixGetObjectRayDirection();
	const float ray_tmin = optixGetRayTmin();
	const float ray_tmax = optixGetRayTmax();

	float3 p0t = tri.p0 - ro;
	float3 p1t = tri.p1 - ro;
	float3 p2t = tri.p2 - ro;

	const float ax = fabsf(rd.x), ay = fabsf(rd.y), az = fabsf(rd.z);
	int kz, kx, ky;
	if (ax >= ay && ax >= az)      { kz = 0; kx = 1; ky = 2; }
	else if (ay >= ax && ay >= az) { kz = 1; kx = 2; ky = 0; }
	else                            { kz = 2; kx = 0; ky = 1; }

	auto comp = [](const float3& v, int k) -> float {
		return k == 0 ? v.x : (k == 1 ? v.y : v.z);
	};
	const float Dx = comp(rd, kx), Dy = comp(rd, ky), Dz = comp(rd, kz);
	float p0x = comp(p0t, kx), p0y = comp(p0t, ky), p0z = comp(p0t, kz);
	float p1x = comp(p1t, kx), p1y = comp(p1t, ky), p1z = comp(p1t, kz);
	float p2x = comp(p2t, kx), p2y = comp(p2t, ky), p2z = comp(p2t, kz);

	const float Sx = -Dx / Dz, Sy = -Dy / Dz, Sz = 1.0f / Dz;
	p0x += Sx * p0z; p0y += Sy * p0z;
	p1x += Sx * p1z; p1y += Sy * p1z;
	p2x += Sx * p2z; p2y += Sy * p2z;

	const float e0 = p1x*p2y - p1y*p2x;
	const float e1 = p2x*p0y - p2y*p0x;
	const float e2 = p0x*p1y - p0y*p1x;

	if ((e0 < 0.0f || e1 < 0.0f || e2 < 0.0f) && (e0 > 0.0f || e1 > 0.0f || e2 > 0.0f))
		return;
	const float det = e0 + e1 + e2;
	if (det == 0.0f) return;

	p0z *= Sz; p1z *= Sz; p2z *= Sz;
	const float tScaled = e0*p0z + e1*p1z + e2*p2z;
	if (det < 0.0f && (tScaled >= 0.0f || tScaled < ray_tmax * det)) return;
	if (det > 0.0f && (tScaled <= 0.0f || tScaled > ray_tmax * det)) return;

	const float invDet = 1.0f / det;
	const float t = tScaled * invDet;
	if (t < ray_tmin || t > ray_tmax) return;

	// Barycentric weights for p0/p1/p2, same convention optixGetTriangleBarycentrics()
	// uses on the recursive (native-triangle) path: b1 weights p1, b2 weights
	// p2, b0=1-b1-b2 weights p0 (see optix_intersection_triangle.h's comment).
	// e0/e1/e2 above are already proportional to the point-vs-opposite-edge
	// signed areas (this IS a Möller-Trumbore/Woop-style barycentric test),
	// so dividing by their sum (det) gives the weights directly - no extra
	// work, just exposing numbers this program already computed. These two
	// attribute slots were unused (hardcoded 0) before; not a payload/ABI
	// change, since attribute count is fixed by pipelineCompileOptions
	// regardless of how many of the 4 a given intersection program fills in.
	const float b1 = e1 * invDet;
	const float b2 = e2 * invDet;
	optixReportIntersection(t, 0, __float_as_int(b1), __float_as_int(b2), 0, 0);
}

// Alpha-cutout for radiance rays (MaterialData::alphaMaskTexIdx) - runs
// BEFORE closest-hit is decided, which is exactly when a transparent pixel
// needs to be skipped so it can't win as "the" hit in the first place.
// Registered on the same hit group as __intersection__wf_triangle/
// __closesthit__wf_triangle above (wavefront_path_tracer.cpp's triHitDesc
// now sets moduleAH too), not a new program group. Uses
// optixGetAttribute_0/1() (this hit group's own custom-intersection
// barycentrics), NOT optixGetTriangleBarycentrics() - that's only valid
// for OptiX's built-in triangle intersection, which the recursive path
// uses but this wavefront-native one does not (see the header comment
// above). A no-op for the overwhelming majority of triangles, whose
// material has no alpha mask.
extern "C" __global__ void __anyhit__wf_triangle() {
	const int instBase = wf_instance_base();
	const int primIdx = (int)(wf_prim_base(instBase) + optixGetPrimitiveIndex());
	const TriangleData& tri = wf_params.triangles[primIdx];
	const MaterialData& mat = wf_params.materials[tri.materialIdx];
	if (mat.alphaMaskTexIdx < 0) return;

	float uv_u = 0.0f, uv_v = 0.0f;
	if (tri.hasUVs) {
		const float b1 = __int_as_float(optixGetAttribute_0());
		const float b2 = __int_as_float(optixGetAttribute_1());
		const float b0 = 1.0f - b1 - b2;
		uv_u = b0 * tri.uv0.x + b1 * tri.uv1.x + b2 * tri.uv2.x;
		uv_v = b0 * tri.uv0.y + b1 * tri.uv1.y + b2 * tri.uv2.y;
	}
	if (!wf_passes_alpha_cutout(mat.alphaMaskTexIdx, uv_u, uv_v))
		optixIgnoreIntersection();
}

extern "C" __global__ void __closesthit__wf_triangle() {
	WfHitPayload* payload = (WfHitPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	const int instBase = wf_instance_base();
	const int primIdx = (int)(wf_prim_base(instBase) + optixGetPrimitiveIndex());
	const TriangleData& tri = wf_params.triangles[primIdx];

	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float  t_hit    = optixGetRayTmax();

	float3 hit_point = ray_orig + t_hit * ray_dir;

	// The vertices are in the definition's object space for an instanced
	// triangle, so the normal derived from them has to be carried to world
	// space before it can be compared against the world ray. Skipped entirely
	// when instBase < 0.
	float3 geom_normal = normalize(cross(tri.p1 - tri.p0, tri.p2 - tri.p0));
	if (instBase >= 0)
		geom_normal = normalize(optixTransformNormalFromObjectToWorldSpace(geom_normal));
	bool front_face = dot(ray_dir, geom_normal) < 0.0f;
	float3 normal = front_face ? geom_normal : -geom_normal;

	// Barycentric-interpolated UV, matching optix_intersection_triangle.h's
	// closesthit exactly - tri.hasUVs gates it the same way tri.hasNormals
	// gates smooth-normal interpolation there (this program stays flat-shaded
	// regardless, see this function's own header comment; UV is independent
	// of that and is the one piece MaterialType::Lambertian's textureIdx
	// actually needs).
	float uv_u = 0.0f, uv_v = 0.0f;
	if (tri.hasUVs) {
		const float b1 = __int_as_float(optixGetAttribute_0());
		const float b2 = __int_as_float(optixGetAttribute_1());
		const float b0 = 1.0f - b1 - b2;
		uv_u = b0 * tri.uv0.x + b1 * tri.uv1.x + b2 * tri.uv2.x;
		uv_v = b0 * tri.uv0.y + b1 * tri.uv1.y + b2 * tri.uv2.y;
	}

	// Real UV-derived tangent (dpdu), for MaterialType::NormalMappedLambertian
	// - stashed in objNormal (sphere-only field otherwise left zero for
	// triangle hits, see HitWorkItem::objNormal's own comment) since
	// WfHitPayload/HitWorkItem carry no per-vertex position/UV data of their
	// own to recompute this from later. Solves the standard 2x2 system
	// relating edge vectors to UV deltas (pbrt-v4 Triangle::Intersect),
	// mirroring optix_intersection_triangle.h's recursive-path equivalent
	// exactly. Computed unconditionally (cheap, pure math) rather than
	// gated on the hit material actually being normal-mapped.
	float3 tri_dpdu;
	{
		const float3 e1 = tri.p1 - tri.p0;
		if (tri.hasUVs) {
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
		} else {
			tri_dpdu = normalize(e1);
		}
		if (instBase >= 0) tri_dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(tri_dpdu));
	}

	payload->hitPoint    = hit_point;
	payload->normal      = normal;
	payload->t           = t_hit;
	payload->materialIdx = tri.materialIdx;
	payload->geomType    = 3;
	payload->hit         = true;
	payload->mediumTFar  = 0.0f;
	// See __closesthit__wf_quad's comment on frontFace - same missing
	// initialization here. RoughDielectric OBJ meshes (e.g. glass triangles)
	// are the case this actually affects.
	payload->frontFace   = front_face ? 1 : 0;
	payload->objNormal   = tri_dpdu;
	payload->uv_u        = uv_u;
	payload->uv_v        = uv_v;
}

// ============================================================================
// __closesthit__wf_sphere
// ============================================================================
extern "C" __global__ void __closesthit__wf_sphere() {
	WfHitPayload* payload = (WfHitPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	// Custom intersection stores sphere index in attribute 0.
	const int instBase  = wf_instance_base();
	const int sphereIdx = (int)(wf_prim_base(instBase) + optixGetPrimitiveIndex());

	// Reconstruct hit point and normal from ray + t. Position and view
	// direction stay in WORLD space - they feed lighting, which is world-space
	// throughout - while the normal is derived in the sphere's own space and
	// brought back, so a placement's non-uniform scale yields the ellipsoid
	// normal rather than a resized sphere's.
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float  t_hit    = optixGetRayTmax();

	float3 hit_point = ray_orig + t_hit * ray_dir;

	// Sphere normal. optixGetObjectRay*() is illegal in a closest-hit program,
	// so the hit point is transformed instead of the ray rebuilt - same reason
	// optix_intersection_sphere.h does it this way. Both transforms are
	// skipped outright when instBase < 0, leaving that path bit-for-bit what
	// it computed before instancing existed.
	const SphereData& sph = wf_params.spheres[sphereIdx];
	// See GpuMediumShapeKind's comment in optix_types.h. Box bounds are
	// static, so re-read directly from `sph` rather than needing any
	// motion-interpolated attribute.
	const bool is_box = (sph.shapeKind == GpuMediumShapeKind::Box);
	float3 obj_hit = hit_point;
	if (instBase >= 0) obj_hit = optixTransformPointFromWorldToObjectSpace(hit_point);
	// Kept OBJECT-space (never transformed) for UV purposes, matching
	// optix_intersection_sphere.h's own obj_normal/outward_normal split - a
	// texture stays pinned to the geometry as a placement rotates it, which a
	// world-space UV wouldn't. outward_normal below is the (possibly
	// world-transformed) copy used for shading/dpdu.
	const float3 obj_normal = is_box
		? wf_box_face_normal(obj_hit, sph.boxMin, sph.boxMax)
		: normalize(obj_hit - sph.center);
	float3 outward_normal = obj_normal;
	if (instBase >= 0)
		outward_normal = normalize(optixTransformNormalFromObjectToWorldSpace(outward_normal));
	// Flip to face the ray
	bool front_face = dot(ray_dir, outward_normal) < 0.0f;
	float3 normal = front_face ? outward_normal : -outward_normal;

	// Sphere UV - matches CPU's get_sphere_uv() / optix_intersection_sphere.h
	// exactly (uses the raw, un-flipped obj_normal - UV is a property of the
	// surface point, independent of which side the ray hit from).
	const float sphere_theta = acosf(-obj_normal.y);
	const float sphere_phi = atan2f(-obj_normal.z, obj_normal.x) + 3.14159265358979323846f;

	payload->hitPoint    = hit_point;
	payload->normal      = normal;
	payload->t           = t_hit;
	payload->materialIdx = sph.materialIdx;
	payload->geomType    = 0;
	payload->hit         = true;
	payload->mediumTFar  = 0.0f;
	payload->frontFace   = front_face ? 1 : 0;
	payload->objNormal   = outward_normal;
	payload->uv_u        = sphere_phi / (2.0f * 3.14159265358979323846f);
	payload->uv_v        = sphere_theta / 3.14159265358979323846f;

	// MaterialType::Medium (always) and MaterialType::DielectricMedium (exit
	// surface only, front_face false - its entry surface just refracts/
	// reflects normally, no re-intersection needed there): override with the
	// entry (near) / exit (far) roots recomputed relative to the CURRENT ray
	// origin, matching optix_intersection_sphere.h's __closesthit__sphere.
	// optixGetRayTmax() alone isn't enough - it reports whichever single root
	// the intersection program found valid, which is the FAR root when this
	// ray already starts inside the sphere (e.g. continuing after a prior
	// in-medium scatter). Recomputing both roots here handles that re-entry
	// case the same way the recursive path does.
	const MaterialData& sph_mat = wf_params.materials[sph.materialIdx];
	const bool needsNearFar = (sph_mat.type == MaterialType::Medium) ||
		(sph_mat.type == MaterialType::DielectricMedium && !front_face);
	if (needsNearFar) {
		float3 unit_dir = normalize(ray_dir);
		float t_near, t_far;
		if (is_box) {
			float bn, bf;
			wf_box_slab_intersect(ray_orig, unit_dir, sph.boxMin, sph.boxMax, bn, bf);
			t_near = fmaxf(0.0f, bn);
			t_far  = bf;
		} else {
			float3 oc2 = ray_orig - sph.center;
			float half_b2 = dot(oc2, unit_dir);
			float c2 = dot(oc2, oc2) - sph.radius * sph.radius;
			float disc2 = fmaxf(0.0f, half_b2 * half_b2 - c2);
			float sq2 = sqrtf(disc2);
			t_near = fmaxf(0.0f, -half_b2 - sq2);
			t_far  = -half_b2 + sq2;
		}
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
	payload->mediumTFar  = 0.0f;
	// frontFace was left uninitialized here (only __closesthit__wf_sphere set
	// it) - RoughDielectric/DielectricMedium in wavefront_kernels.cu read
	// h.frontFace unconditionally regardless of which geometry type hit,
	// so a quad-hit ray fed those cases garbage stack data instead of the
	// real hit side.
	payload->frontFace   = front_face ? 1 : 0;
	// No per-quad UV exists on this codebase's GPU side (QuadData carries no
	// uv0/1/2 the way TriangleData does) - matches the recursive path, which
	// never textures a quad either (its shade_material() callers pass (0,0)
	// for every quad hit).
	payload->uv_u        = 0.0f;
	payload->uv_v        = 0.0f;
}

// ============================================================================
// __intersection__wf_disk / __intersection__wf_cylinder,
// __closesthit__wf_disk / __closesthit__wf_cylinder (Phase 4c)
//
// Wavefront-native duplicates of the recursive backend's
// optix_intersection_disk_cylinder.h (see that file's own header comment for
// the full design rationale: both shapes carry their own object<->world
// transform - DiskData::o2w/w2o, CylinderData::o2w/w2o - applied BY HAND
// here rather than through OptiX's per-GAS-instance transform, since
// neither shape is instanced). Duplicated rather than shared because
// wavefront_programs.cu and optix_programs.cu are two separate OptiX
// modules (see this file's own top-of-file header comment on why every
// other wf_ intersection program here is likewise a native copy, not
// cross-module reuse).
//
// Unlike the recursive backend's __closesthit__disk/__closesthit__cylinder,
// these don't call shade_material() - they only fill WfHitPayload, exactly
// like __closesthit__wf_quad above; evaluate_materials() (wavefront_
// kernels.cu) shades the hit later from the queued HitWorkItem.
//
// wf_dc_apply_point()/wf_dc_apply_vector()/wf_dc_apply_normal_from_w2o()
// (the object<->world transform helpers both shapes need) are defined
// earlier in this file, right after packPointer()/unpackPointer() - before
// the "#include wavefront_probe.h" line, since that file's own probe-disk/
// probe-cylinder closest-hit programs need them too.
// ============================================================================

extern "C" __global__ void __intersection__wf_disk() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const DiskData& disk = wf_params.disks[primIdx];

	const float3 ray_orig_w = optixGetWorldRayOrigin();
	const float3 ray_dir_w  = optixGetWorldRayDirection();
	const float ray_tmin = optixGetRayTmin();
	const float ray_tmax = optixGetRayTmax();

	const float3 ro = wf_dc_apply_point(disk.w2o, ray_orig_w);
	const float3 rd = wf_dc_apply_vector(disk.w2o, ray_dir_w);

	if (rd.z == 0.0f) return;
	const float t = (disk.height - ro.z) / rd.z;
	if (t < ray_tmin || t > ray_tmax) return;

	const float hx = ro.x + t * rd.x;
	const float hy = ro.y + t * rd.y;
	const float dist2 = hx * hx + hy * hy;
	if (dist2 > disk.radius * disk.radius || dist2 < disk.innerRadius * disk.innerRadius) return;

	float phi = atan2f(hy, hx);
	if (phi < 0.0f) phi += 6.283185307179586f;
	if (phi > disk.phiMax) return;

	optixReportIntersection(t, 0, 0, 0, 0, 0);
}

extern "C" __global__ void __closesthit__wf_disk() {
	WfHitPayload* payload = (WfHitPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	const unsigned int primIdx = optixGetPrimitiveIndex();
	const DiskData& disk = wf_params.disks[primIdx];

	const float t_hit = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t_hit * ray_dir;

	// A disk is flat: its object-space normal is the constant +Z everywhere
	// on its surface, unlike Cylinder's (see __closesthit__wf_cylinder),
	// which varies with hit position.
	const float3 obj_normal = make_float3(0.0f, 0.0f, 1.0f);
	float3 outward_normal = normalize(wf_dc_apply_normal_from_w2o(disk.w2o, obj_normal));
	const bool front_face = dot(ray_dir, outward_normal) < 0.0f;
	const float3 normal = front_face ? outward_normal : -outward_normal;

	payload->hitPoint    = hit_point;
	payload->normal      = normal;
	payload->t           = t_hit;
	payload->materialIdx = disk.materialIdx;
	payload->geomType    = 4;
	payload->hit         = true;
	payload->mediumTFar  = 0.0f;
	payload->frontFace   = front_face ? 1 : 0;
	payload->uv_u        = 0.0f;
	payload->uv_v        = 0.0f;
}

// One candidate root: in range, inside the object-space Z clip, and inside
// the phi sweep. A free device function rather than a lambda, matching this
// file's own established style (no CUDA extended-lambda dependency).
__device__ __forceinline__ bool wf_dc_cylinder_check(const CylinderData& cyl, const float3& ro, const float3& rd,
													   float t, float ray_tmin, float ray_tmax) {
	if (t < ray_tmin || t > ray_tmax) return false;
	const float hz = ro.z + t * rd.z;
	if (hz < cyl.zMin || hz > cyl.zMax) return false;
	const float hx = ro.x + t * rd.x;
	const float hy = ro.y + t * rd.y;
	float phi = atan2f(hy, hx);
	if (phi < 0.0f) phi += 6.283185307179586f;
	return phi <= cyl.phiMax;
}

extern "C" __global__ void __intersection__wf_cylinder() {
	const unsigned int primIdx = optixGetPrimitiveIndex();
	const CylinderData& cyl = wf_params.cylinders[primIdx];

	const float3 ray_orig_w = optixGetWorldRayOrigin();
	const float3 ray_dir_w  = optixGetWorldRayDirection();
	const float ray_tmin = optixGetRayTmin();
	const float ray_tmax = optixGetRayTmax();

	const float3 ro = wf_dc_apply_point(cyl.w2o, ray_orig_w);
	const float3 rd = wf_dc_apply_vector(cyl.w2o, ray_dir_w);

	// Stable quadratic solve in double precision - see optix_intersection_
	// disk_cylinder.h's own __intersection__cylinder for why (avoids
	// catastrophic cancellation for a thin cylinder seen nearly edge-on).
	const double da = (double)rd.x, db = (double)rd.y;
	const double oa = (double)ro.x, ob = (double)ro.y;
	const double a = da * da + db * db;
	const double b = 2.0 * (oa * da + ob * db);
	const double c = oa * oa + ob * ob - (double)cyl.radius * (double)cyl.radius;
	if (a == 0.0) return;
	const double f = b / (2.0 * a);
	const double vx = oa - f * da, vy = ob - f * db;
	const double len_v = sqrt(vx * vx + vy * vy);
	const double discrim = 4.0 * a * ((double)cyl.radius + len_v) * ((double)cyl.radius - len_v);
	if (discrim < 0.0) return;
	const double sqrt_disc = sqrt(discrim);
	const double q = (b < 0.0) ? -0.5 * (b - sqrt_disc) : -0.5 * (b + sqrt_disc);
	float t0 = (float)(q / a), t1 = (float)(c / q);
	if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
	if (t0 > ray_tmax || t1 < ray_tmin) return;

	float t = t0;
	if (!wf_dc_cylinder_check(cyl, ro, rd, t0, ray_tmin, ray_tmax)) {
		t = t1;
		if (!wf_dc_cylinder_check(cyl, ro, rd, t1, ray_tmin, ray_tmax)) return;
	}

	optixReportIntersection(t, 0, 0, 0, 0, 0);
}

extern "C" __global__ void __closesthit__wf_cylinder() {
	WfHitPayload* payload = (WfHitPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	const unsigned int primIdx = optixGetPrimitiveIndex();
	const CylinderData& cyl = wf_params.cylinders[primIdx];

	const float t_hit = optixGetRayTmax();
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir  = optixGetWorldRayDirection();
	const float3 hit_point = ray_orig + t_hit * ray_dir;

	// Cylinder's normal varies with hit position (radial, from the axis) -
	// recover the object-space hit point to compute it, same technique as
	// the recursive backend's __closesthit__cylinder.
	const float3 obj_hit = wf_dc_apply_point(cyl.w2o, hit_point);
	const float obj_hit_len = sqrtf(obj_hit.x * obj_hit.x + obj_hit.y * obj_hit.y);
	const float3 obj_normal = (obj_hit_len > 1e-8f)
		? make_float3(obj_hit.x / obj_hit_len, obj_hit.y / obj_hit_len, 0.0f)
		: make_float3(1.0f, 0.0f, 0.0f);
	float3 outward_normal = normalize(wf_dc_apply_normal_from_w2o(cyl.w2o, obj_normal));
	const bool front_face = dot(ray_dir, outward_normal) < 0.0f;
	const float3 normal = front_face ? outward_normal : -outward_normal;

	payload->hitPoint    = hit_point;
	payload->normal      = normal;
	payload->t           = t_hit;
	payload->materialIdx = cyl.materialIdx;
	payload->geomType    = 5;
	payload->hit         = true;
	payload->mediumTFar  = 0.0f;
	payload->frontFace   = front_face ? 1 : 0;
	payload->uv_u        = 0.0f;
	payload->uv_v        = 0.0f;
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
	// Must also guard against sq.capacity, not just the live *sq.counter:
	// WorkQueue::push() (wavefront_types.h) increments the counter
	// unconditionally even once the backing array (d_shadowItems_, sized to
	// queueCapacity_ = width*height) is full - it only skips the actual item
	// write past capacity (returns -1). The shadow queue is genuinely NOT
	// bounded by 1 push per hit: evaluate_materials's wf_finish_material_
	// scatter() (wavefront_kernels.cu) can push once for area-light NEE,
	// once more for sky NEE, and once per punctual light for the SAME hit,
	// so a scene combining several of those (confirmed: scene B2/"Cornell
	// Rough Metal", which sets a non-zero GpuCameraParams::backgroundColor -
	// see scene_builder.cpp case 10 - so its own sky-NEE branch fires
	// alongside its area light) can legitimately push more items in one
	// bounce than queueCapacity_ holds. Before this fix, *sq.counter (e.g.
	// 3934) exceeding sq.capacity (e.g. 3600) meant this raygen launched
	// with more threads than the buffer has slots for, and every idx in
	// [capacity, counter) read sq.items[idx]/occluded[idx] past their
	// cudaMalloc'd end - confirmed via compute-sanitizer as an out-of-bounds
	// __global__ read landing just past d_shadowItems_'s allocation. That
	// stray read (and the matching one in accumulate_shadow, wavefront_
	// kernels.cu) is what corrupted the CUDA/OptiX context: once enough
	// prior scene switches left the right garbage in the allocator's
	// adjacent memory, the read landed on unmapped/foreign memory instead of
	// harmless padding, raising CUDA error 700 for the rest of the process.
	// Excess pushes beyond capacity are still silently dropped (same
	// WorkQueue::push() contract as before this fix) - this only stops the
	// out-of-bounds READ of those dropped slots, it does not change which
	// shadow rays get traced.
	if ((int)idx >= *sq.counter || (int)idx >= sq.capacity) return;

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
		0,                  // miss SBT index -- shadowSBT_ has its OWN dedicated
							// missRecordBase with exactly ONE record (see
							// buildSBT()'s shadowSBT_.missRecordCount = 1), unlike
							// a combined [radiance, shadow] miss array; index 1 was
							// out-of-bounds. Silently "worked" on small scenes by
							// reading adjacent heap bytes past the 1-record cudaMalloc
							// (undefined behavior, not correctness) until confirmed via
							// OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL, which flags it
							// as MISS_SBT_OUT_OF_BOUNDS on every scene, and turns it into
							// a hard device fault whose exact address depends on
							// allocation layout -- which is what made scene 8's larger
							// memory footprint crash outright instead of "working".
		p0, p1
	);

	// Write result into the bool array (reused from framebuffer pointer during shadow pass)
	bool* occluded = (bool*)wf_params.framebuffer;
	occluded[idx]  = sp.occluded;
}

// __anyhit__wf_shadow_*: one per geometry type (sphere/quad/bilinear_patch/
// triangle), mirroring optix_anyhit_shadow.h's __anyhit__shadow_* exactly -
// see that file's own comments for why DiffuseLight and the transmissive
// materials need special handling, not just "any hit = occluded":
//
//   - MaterialType::DiffuseLight is NOT an occluder - a shadow ray sampled
//     toward a light is EXPECTED to end inside/on that light's own surface,
//     so treating the hit as occlusion would make every area light shadow
//     itself. This one previously used a single generic __anyhit__wf_shadow
//     with no material lookup at all (unconditional occluded=true regardless
//     of what was hit), so any shadow ray whose sampled direction actually
//     reached the light within its tMax read as occluded - which for a
//     sphere light is any point with a large-enough cosine to the light
//     (i.e. exactly the well-lit directions NEE most needs), and for a quad
//     light happened to not manifest because the tMax/origin-offset epsilon
//     interaction (see wavefront_kernels.cu's shadow-ray-setup comment)
//     pushed the effective light-plane distance the other way for a
//     camera-facing quad normal. Confirmed via scene B14 (Measured BRDF
//     showroom, sphere light only): floor and sphere-tops facing the light
//     directly went black under --wavefront while the recursive path
//     rendered them correctly lit.
//   - Dielectric/RoughDielectric/ThinDielectric/DiffuseTransmission (and
//     Medium/CloudMedium, sphere-only - see optix_anyhit_shadow.h's own
//     comment) let light through rather than blocking NEE outright.
extern "C" __global__ void __anyhit__wf_shadow_sphere() {
	const int instBase = wf_instance_base();
	const SphereData& sph = wf_params.spheres[wf_prim_base(instBase) + optixGetPrimitiveIndex()];
	const MaterialData& mat = wf_params.materials[sph.materialIdx];

	WfShadowPayload* sp = (WfShadowPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	if (mat.type == MaterialType::DiffuseLight) {
		sp->occluded = false;
		optixTerminateRay();
		return;
	}
	// MaterialType::DielectricMedium belongs here for the same reason as
	// Medium/CloudMedium/RgbGridMedium just above - see optix_anyhit_shadow.h's
	// __anyhit__shadow_sphere for the full comment (that omission was a real
	// bug, found while adding real NEE to this material's medium-interior
	// phase-scatter case in wavefront_kernels.cu / optix_intersection_sphere.h).
	if (mat.type == MaterialType::Dielectric ||
		mat.type == MaterialType::RoughDielectric ||
		mat.type == MaterialType::ThinDielectric ||
		mat.type == MaterialType::DiffuseTransmission ||
		mat.type == MaterialType::Medium ||
		mat.type == MaterialType::CloudMedium ||
		mat.type == MaterialType::RgbGridMedium ||
		mat.type == MaterialType::DielectricMedium) {
		optixIgnoreIntersection();
		return;
	}
	sp->occluded = true;
	optixTerminateRay();
}

extern "C" __global__ void __anyhit__wf_shadow_quad() {
	const QuadData& quad = wf_params.quads[optixGetPrimitiveIndex()];
	const MaterialData& mat = wf_params.materials[quad.materialIdx];

	WfShadowPayload* sp = (WfShadowPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	if (mat.type == MaterialType::DiffuseLight) {
		sp->occluded = false;
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
	sp->occluded = true;
	optixTerminateRay();
}

extern "C" __global__ void __anyhit__wf_shadow_bilinear_patch() {
	const BilinearPatchData& patch = wf_params.bilinearPatches[optixGetPrimitiveIndex()];
	const MaterialData& mat = wf_params.materials[patch.materialIdx];

	WfShadowPayload* sp = (WfShadowPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	if (mat.type == MaterialType::DiffuseLight) {
		sp->occluded = false;
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
	sp->occluded = true;
	optixTerminateRay();
}

extern "C" __global__ void __anyhit__wf_shadow_triangle() {
	const int instBase = wf_instance_base();
	const TriangleData& tri = wf_params.triangles[wf_prim_base(instBase) + optixGetPrimitiveIndex()];
	const MaterialData& mat = wf_params.materials[tri.materialIdx];

	// Alpha-cutout: a transparent pixel of a leaf/foliage material casts no
	// shadow there - see __anyhit__wf_triangle's own comment for why
	// optixGetAttribute_0/1() (this hit group's own custom-intersection
	// barycentrics), not optixGetTriangleBarycentrics(). No-op for the
	// overwhelming majority of triangles, whose material has no alpha mask.
	if (mat.alphaMaskTexIdx >= 0) {
		float uv_u = 0.0f, uv_v = 0.0f;
		if (tri.hasUVs) {
			const float b1 = __int_as_float(optixGetAttribute_0());
			const float b2 = __int_as_float(optixGetAttribute_1());
			const float b0 = 1.0f - b1 - b2;
			uv_u = b0 * tri.uv0.x + b1 * tri.uv1.x + b2 * tri.uv2.x;
			uv_v = b0 * tri.uv0.y + b1 * tri.uv1.y + b2 * tri.uv2.y;
		}
		if (!wf_passes_alpha_cutout(mat.alphaMaskTexIdx, uv_u, uv_v)) {
			optixIgnoreIntersection();
			return;
		}
	}

	WfShadowPayload* sp = (WfShadowPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	if (mat.type == MaterialType::DiffuseLight) {
		sp->occluded = false;
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
	sp->occluded = true;
	optixTerminateRay();
}

// Disk/Cylinder (Phase 4c) - same simple DiffuseLight/dielectric-family/
// opaque list as quad/bilinear-patch above, matching the recursive backend's
// __anyhit__shadow_disk/__anyhit__shadow_cylinder (optix_anyhit_shadow.h)
// exactly - the pbrt loader never assigns Medium/DielectricMedium to a
// disk/cylinder (see pbrt_gpu_builder.h's disk/cylinder loop), so no
// CloudMedium/RgbGridMedium/Medium/DielectricMedium entries are needed here.
extern "C" __global__ void __anyhit__wf_shadow_disk() {
	const DiskData& disk = wf_params.disks[optixGetPrimitiveIndex()];
	const MaterialData& mat = wf_params.materials[disk.materialIdx];

	WfShadowPayload* sp = (WfShadowPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	if (mat.type == MaterialType::DiffuseLight) {
		sp->occluded = false;
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
	sp->occluded = true;
	optixTerminateRay();
}

extern "C" __global__ void __anyhit__wf_shadow_cylinder() {
	const CylinderData& cyl = wf_params.cylinders[optixGetPrimitiveIndex()];
	const MaterialData& mat = wf_params.materials[cyl.materialIdx];

	WfShadowPayload* sp = (WfShadowPayload*)unpackPointer(
		optixGetPayload_0(), optixGetPayload_1());

	if (mat.type == MaterialType::DiffuseLight) {
		sp->occluded = false;
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
	sp->occluded = true;
	optixTerminateRay();
}

// __miss__wf_shadow: ray reached tMax without hitting anything — not occluded.
extern "C" __global__ void __miss__wf_shadow() {
	// occluded stays false (its default)
}

// Exception program -- see wavefront_path_tracer.cpp's exceptionFlags
// comment (initialize()) for the full story: registering this program and
// enabling STACK_OVERFLOW/TRACE_DEPTH is the actual fix for a CUDA 718
// "invalid program counter" crash that used to kill the device context
// after a specific sequence of earlier GPU tests. It has never been
// observed to fire in a passing run -- kept registered as a real safety
// net (prints what OptiX thinks went wrong instead of a bare post-hoc
// cudaStreamSynchronize failure) rather than diagnostic scaffolding.
extern "C" __global__ void __exception__wf_report() {
	const int code = optixGetExceptionCode();
	const uint3 idx = optixGetLaunchIndex();
	printf("[WF-EXCEPTION] code=%d launchIdx=(%u,%u,%u)\n", code, idx.x, idx.y, idx.z);
	if (code == OPTIX_EXCEPTION_CODE_STACK_OVERFLOW) {
		printf("[WF-EXCEPTION]   -> STACK_OVERFLOW\n");
	} else if (code == OPTIX_EXCEPTION_CODE_TRACE_DEPTH_EXCEEDED) {
		printf("[WF-EXCEPTION]   -> TRACE_DEPTH_EXCEEDED\n");
	}
}
