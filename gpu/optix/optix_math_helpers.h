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

inline __host__ __device__ float lengthSquared(const float3& v) {
	return dot(v, v);
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
