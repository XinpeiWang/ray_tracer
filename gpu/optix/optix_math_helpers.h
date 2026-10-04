// OptiX Math Helpers
// Simple vector math functions for CUDA/OptiX host code
// Compatible with CUDA vector types (float3, float4, etc.)

#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <cuda_runtime.h>
#include <vector_functions.h>
#include <cmath>

// ============================================================================
// Float3 Helper Functions
// ============================================================================
// Note: make_float3 is provided by CUDA's vector_functions.h

inline __host__ __device__ float3 operator+(const float3& a, const float3& b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator-(const float3& a, const float3& b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// Unary negation
inline __host__ __device__ float3 operator-(const float3& a) {
	return make_float3(-a.x, -a.y, -a.z);
}

inline __host__ __device__ float3 operator*(const float3& a, float s) {
	return make_float3(a.x * s, a.y * s, a.z * s);
}

inline __host__ __device__ float3 operator*(float s, const float3& a) {
	return make_float3(a.x * s, a.y * s, a.z * s);
}

// Component-wise multiplication
inline __host__ __device__ float3 operator*(const float3& a, const float3& b) {
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ float3 operator/(const float3& a, float s) {
	float inv = 1.0f / s;
	return make_float3(a.x * inv, a.y * inv, a.z * inv);
}

inline __host__ __device__ float dot(const float3& a, const float3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 cross(const float3& a, const float3& b) {
	return make_float3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}

inline __host__ __device__ float length(const float3& v) {
	return sqrtf(dot(v, v));
}

inline __host__ __device__ float3 normalize(const float3& v) {
	float len = length(v);
	if (len > 1e-6f) {
		return v / len;
	}
	return make_float3(0.0f, 0.0f, 0.0f);
}

// ============================================================================
// Float4 Helper Functions
// ============================================================================
// Note: make_float4 is provided by CUDA's vector_functions.h

inline __host__ __device__ float4 operator+(const float4& a, const float4& b) {
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __host__ __device__ float4 operator-(const float4& a, const float4& b) {
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline __host__ __device__ float4 operator*(const float4& a, float s) {
	return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

inline __host__ __device__ float4 operator/(const float4& a, float s) {
	float inv = 1.0f / s;
	return make_float4(a.x * inv, a.y * inv, a.z * inv, a.w * inv);
}

// ============================================================================
// Utility Functions
// ============================================================================

inline __host__ __device__ float clamp(float x, float min_val, float max_val) {
	return fmaxf(min_val, fminf(max_val, x));
}

inline __host__ __device__ float3 clamp(const float3& v, float min_val, float max_val) {
	return make_float3(
		clamp(v.x, min_val, max_val),
		clamp(v.y, min_val, max_val),
		clamp(v.z, min_val, max_val)
	);
}

inline __host__ __device__ float lerp(float a, float b, float t) {
	return a + t * (b - a);
}

inline __host__ __device__ float3 lerp(const float3& a, const float3& b, float t) {
	return a + (b - a) * t;
}

// Barycentric-interpolated vertex normal, falling back to the facet's geometric normal when the
// interpolation cancels to (near) zero - opposite vertex normals at a mid-edge point, or a
// degenerate authored normal. pbrt-v4 does the same (Triangle::InteractionFromIntersection,
// shapes.h:947); a bare normalize() here would turn that into a NaN shading normal.
inline __host__ __device__ float3 interpolate_shading_normal(
		const float3& n0, const float3& n1, const float3& n2,
		float b0, float b1, float b2, const float3& geometric_normal) {
	const float3 n = b0 * n0 + b1 * n1 + b2 * n2;
	const float len2 = n.x * n.x + n.y * n.y + n.z * n.z;
	return len2 > 1e-20f ? n * (1.0f / sqrtf(len2)) : geometric_normal;
}

// A shadow ray whose origin was nudged off the shading point (by a normal and/or direction epsilon) must
// have its length measured from the NEW origin: callers compute tMax as the distance from the original
// point to the light, so keeping it unchanged made every finite-light shadow ray overshoot the light by
// the nudge (~0.01 world units). With a light flush against a ceiling, a wall or a ledge - a lamp
// fixture, a window, a strip light - the overshoot ended inside that surface behind the light and the
// light sample was reported blocked. A closed 5-unit room lit by a quad 0.005 below its ceiling rendered
// at 22% (recursive) / 37% (wavefront) of the CPU's brightness; with the length corrected they agree.
// pbrt-v4 measures the shadow ray from its offset origin the same way (SpawnRayTo, ray.h:103-107).
// An unbounded sentinel (the sky's 1e30) is passed through untouched.
inline __host__ __device__ float shadow_tmax_after_shift(const float3& original, const float3& shifted,
														 const float3& unit_dir, float tmax) {
	if (tmax >= 1e29f) return tmax;
	const float d = (shifted.x - original.x) * unit_dir.x + (shifted.y - original.y) * unit_dir.y +
					(shifted.z - original.z) * unit_dir.z;
	return fmaxf(tmax - d, 0.0f);
}

// Aim a shifted shadow ray at the point `original + unit_dir * tmax` instead of keeping the old direction
// (pbrt's SpawnRayTo does the same). When the shift has a sideways part - the wavefront backend nudges along the
// surface normal too - a ray keeping its direction crosses a finite light's plane earlier than the nominal
// distance at grazing angles, and with emitters acting as occluders it is blocked by the light it targets.
// `tmax` must already stop short of the target. Unbounded rays (sky, tmax >= 1e29) keep their direction.
inline __host__ __device__ void shadow_ray_toward(const float3& original, const float3& shifted, const float3& unit_dir,
												 float tmax, float3& out_dir, float& out_tmax) {
	out_dir  = unit_dir;
	out_tmax = shadow_tmax_after_shift(original, shifted, unit_dir, tmax);
	if (tmax >= 1e29f) return;
	const float3 to_target = original + unit_dir * tmax - shifted;
	const float len = sqrtf(to_target.x * to_target.x + to_target.y * to_target.y + to_target.z * to_target.z);
	if (len > 1e-6f) { out_dir = to_target / len; out_tmax = len; }
}

// Ceiling on a path's running throughput, the CPU's kMaxPathThroughput (camera.h, 50 in both ray_color() and the
// spectral integrator): Russian roulette only acts on a throughput below 1, so nothing else bounds a BSDF whose
// per-bounce sample weight is large and compounds over bounces - HairBxDF's mean weight is ~4, so three bounces
// off the inside of a closed hair shape already reach the cap. Without the same cap here a hair sphere under a
// dim sky rendered 2.1x brighter on both GPU backends than on the CPU from the second hair bounce on.
constexpr float kMaxPathThroughput = 50.0f;
inline __host__ __device__ float3 clamp_path_throughput(const float3& t) {
	return make_float3(fminf(t.x, kMaxPathThroughput), fminf(t.y, kMaxPathThroughput), fminf(t.z, kMaxPathThroughput));
}
