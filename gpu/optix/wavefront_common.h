// wavefront_common.h -- Shared wf_params/payload structs/device helpers for
// the wavefront GPU backend. Included FIRST by wavefront_programs.cu, before
// every other wavefront_*.h - see this file's own wf_params/WfHitPayload/
// WfShadowPayload/wf_dc_* declarations, which the other split files below
// all depend on.
//
// OptiX device programs for the wavefront GPU path tracer, split (like the
// recursive backend's optix_programs.cu) into wavefront_common.h (this file)
// plus wavefront_raygen.h/wavefront_intersection_*.h/wavefront_anyhit_
// shadow.h/wavefront_miss.h - wavefront_programs.cu itself is just the
// #include aggregator, mirroring optix_programs.cu's own shape.
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

// Wavefront-native duplicate of optix_device_helpers.h's mix_branch_hash01()/
// resolve_mix_material() (recursive backend) - see this codebase's own
// established "no shared device helpers between the two backends" convention
// (wf_material_requires_sphere_only_handling, wf_passes_alpha_cutout above,
// etc. are all likewise wavefront-native copies). Deterministic [0,1) hash
// from a world-space point, matching CPU's branch_hash01() (src/
// TheRestOfYourLife/material_pbrt.h) exactly - NOT a fresh random draw, so a
// radiance hit and its later shadow ray agree on which sub-material a Mix
// resolved to.
__device__ __forceinline__ float wf_mix_branch_hash01(const float3& p) {
	float h = sinf(p.x * 127.1f + p.y * 311.7f + p.z * 74.7f) * 43758.5453f;
	return h - floorf(h);
}

// Resolves a (possibly Mix) MaterialData to a real, non-Mix MaterialData,
// looping (not recursing) while the result is itself another Mix - see
// MaterialType::Mix's own comment (optix_types.h) and resolve_mix_material()'s
// identical recursive-backend twin (optix_device_helpers.h) for the full
// rationale. `outMatIdx` receives the resolved index into wf_params.materials.
__device__ __forceinline__ MaterialData wf_resolve_mix_material(MaterialData mat, int matIdx,
																 const float3& hit_point, int& outMatIdx) {
	constexpr int kMaxMixDepth = 8;
	for (int depth = 0; mat.type == MaterialType::Mix && depth < kMaxMixDepth; ++depth) {
		const float w = mat.mix_extra.mixWeight;
		const float h = wf_mix_branch_hash01(hit_point);
		const int subIdx = (h >= w) ? static_cast<int>(mat.mix_extra.mixMaterialAIdx)
									 : static_cast<int>(mat.mix_extra.mixMaterialBIdx);
		matIdx = subIdx;
		mat = wf_params.materials[subIdx];
	}
	outMatIdx = matIdx;
	return mat;
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

// Wavefront-native duplicate of optix_disk_cylinder_helpers.h's identically
// named recursive-backend function (same cross-module reason every other
// wf_dc_* helper in this file is duplicated, not shared via #include) -
// factored out of __intersection__wf_cylinder/__closesthit__wf_cylinder's
// Medium branch, which used to each carry their own copy of this exact
// double-precision tube-quadric solve. See the recursive twin's own
// comment for why ray-parallel-to-axis (a==0) returns false here
// specifically (a SURFACE-crossing answer), and why a VOLUME/interval
// caller (the Medium branch) must special-case that input itself rather
// than getting an answer from this function.
__device__ __forceinline__ bool wf_dc_solve_tube_quadratic(const float3& ro, const float3& rd, float radius,
															 float& t0, float& t1) {
	const double da = (double)rd.x, db = (double)rd.y;
	const double oa = (double)ro.x, ob = (double)ro.y;
	const double a = da * da + db * db;
	if (a == 0.0) return false;
	const double b = 2.0 * (oa * da + ob * db);
	const double c = oa * oa + ob * ob - (double)radius * (double)radius;
	const double f = b / (2.0 * a);
	const double vx = oa - f * da, vy = ob - f * db;
	const double len_v = sqrt(vx * vx + vy * vy);
	const double discrim = 4.0 * a * ((double)radius + len_v) * ((double)radius - len_v);
	if (discrim < 0.0) return false;
	const double sqrt_disc = sqrt(discrim);
	const double q = (b < 0.0) ? -0.5 * (b - sqrt_disc) : -0.5 * (b + sqrt_disc);
	t0 = (float)(q / a);
	t1 = (float)(c / q);
	if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
	return true;
}

// ============================================================================
// Payload structs (passed by pointer via p0/p1)
// ============================================================================

struct WfHitPayload {
	float3 hitPoint;
	float3 normal;
	float  t;
	int    materialIdx;
	int    geomType;   // 0 = sphere, 1 = quad, 2 = bilinear patch, 4 = disk, 5 = cylinder
	bool   hit;
	float  mediumTFar; // MaterialType::Medium/DielectricMedium only - see HitWorkItem::mediumTFar
	int    frontFace;  // see HitWorkItem::frontFace
	// Dual-purpose carrier - see HitWorkItem::objNormal's own comment for the
	// sphere/triangle uses this mirrors. Bilinear patch (geomType==2) is a
	// third use: the patch's own dpdu ("along the tube's length" for a
	// tessellated curve - curve_tessellate.h's own Quad corner convention),
	// UNNORMALIZED (evaluate_materials()'s Hair case, the only reader,
	// normalizes it) - the genuine fiber tangent this primitive has, unlike
	// its own `normal` (perpendicular to the tube) - see hair_material.h's
	// tangent_is_dpdu comment for the identical CPU-side reasoning.
	float3 objNormal;
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
