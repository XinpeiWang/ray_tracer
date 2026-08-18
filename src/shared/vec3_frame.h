#pragma once
// ---------------------------------------------------------------------------
// vec3_frame.h -- Canonical Vec3<T> / Frame<T> (CPU/GPU shared)
//
// sampling_helpers.h (sampling_detail namespace) and
// portal_image_infinite_light.h (pil_detail namespace) each independently
// defined an equivalent 3-component vector and orthonormal frame, under two
// different calling conventions:
//   - sampling_detail: member methods, snake_case (v.dot(w), v.cross(w),
//     v.normalized(), frame.to_local(v), frame.from_local(v))
//   - pil_detail: free functions, mixed case (dot(v,w), cross(v,w),
//     normalize(v), frame.ToLocal(v), frame.FromLocal(v))
// Both spellings are kept below as thin forwards over one real
// implementation, so neither file's call sites needed to change.
//
// NOTE: Frame's "build from two axes" factory is deliberately NOT unified --
// from_xy() and FromXY() are not the same algorithm. from_xy() (mirrors
// sampling_detail's original) trusts its inputs as already orthonormal and
// only derives the third axis via cross+normalize. FromXY() (mirrors
// pil_detail's original, and pbrt-v4's own Frame::FromXY) performs full
// Gram-Schmidt orthogonalization, tolerating non-orthonormal inputs. Merging
// these would silently change behavior for whichever caller expected the
// other algorithm, so both are kept, each documented at its own call site.
// ---------------------------------------------------------------------------

#include "cpu_gpu.h"
#include <cmath>

template<typename T>
struct Vec3 {
	T x, y, z;
	CPU_GPU Vec3() : x(0), y(0), z(0) {}
	CPU_GPU Vec3(T x, T y, T z) : x(x), y(y), z(z) {}
	CPU_GPU Vec3 operator+(const Vec3& b) const { return {x+b.x, y+b.y, z+b.z}; }
	CPU_GPU Vec3 operator-(const Vec3& b) const { return {x-b.x, y-b.y, z-b.z}; }
	CPU_GPU Vec3 operator*(T s)           const { return {x*s,   y*s,   z*s};   }
	CPU_GPU T    dot(const Vec3& b)       const { return x*b.x + y*b.y + z*b.z; }
	CPU_GPU T    length_squared()         const { return dot(*this); }
	CPU_GPU T    length()                 const { return std::sqrt(length_squared()); }
	CPU_GPU Vec3 normalized()             const {
		T l = length();
		return l > T(0) ? Vec3(x/l, y/l, z/l) : Vec3(T(0),T(0),T(0));
	}
	CPU_GPU Vec3 cross(const Vec3& b) const {
		return { y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x };
	}
};

// Free-function spellings (pil_detail's original call-site style).
template<typename T> CPU_GPU T dot(Vec3<T> a, Vec3<T> b) { return a.dot(b); }
template<typename T> CPU_GPU Vec3<T> cross(Vec3<T> a, Vec3<T> b) { return a.cross(b); }
template<typename T> CPU_GPU Vec3<T> normalize(Vec3<T> v) { return v.normalized(); }
template<typename T> CPU_GPU Vec3<T> sub(Vec3<T> a, Vec3<T> b) { return a - b; }
template<typename T> CPU_GPU T length(Vec3<T> v) { return v.length(); }

// AngleBetween two unit vectors -- numerically stable across all angles.
// Mirrors pbrt-v4 AngleBetween (vecmath.h line 972):
//   dot < 0: Pi - 2*asin(|a+b|/2)   (stable near pi)
//   dot >= 0: 2*asin(|b-a|/2)        (stable near 0)
template<typename T>
CPU_GPU T angle_between(Vec3<T> a, Vec3<T> b) {
	if (a.dot(b) < T(0)) {
		Vec3<T> s = a + b;
		T half_len = std::sqrt(s.length_squared()) * T(0.5);
		half_len = half_len > T(1) ? T(1) : half_len;  // SafeASin clamp
		return T(3.14159265358979323846) - T(2) * std::asin(half_len);
	} else {
		Vec3<T> d = b - a;
		T half_len = std::sqrt(d.length_squared()) * T(0.5);
		half_len = half_len > T(1) ? T(1) : half_len;
		return T(2) * std::asin(half_len);
	}
}

// Orthonormal frame with x, y, z axes (matches pbrt-v4 Frame).
template<typename T>
struct Frame {
	Vec3<T> x, y, z;

	// Build from two axes already assumed orthonormal (sampling_detail's
	// original algorithm) -- only the third axis is derived.
	CPU_GPU static Frame from_xy(Vec3<T> ex_n, Vec3<T> ey_n) {
		Frame f;
		f.x = ex_n;
		f.y = ey_n;
		f.z = ex_n.cross(ey_n).normalized();
		return f;
	}

	// Build from two axes via full Gram-Schmidt (pil_detail's original
	// algorithm, matches pbrt-v4 Frame::FromXY) -- tolerates non-orthonormal
	// inputs. NOT equivalent to from_xy() above; see this file's header.
	static Frame FromXY(Vec3<T> vx, Vec3<T> vy) {
		Vec3<T> nx = normalize(vx);
		Vec3<T> nz = normalize(cross(nx, vy));
		Vec3<T> ny = cross(nz, nx);
		return { nx, ny, nz };
	}

	// Project world vector into local frame.
	CPU_GPU Vec3<T> to_local(Vec3<T> v) const {
		return Vec3<T>(v.dot(x), v.dot(y), v.dot(z));
	}
	CPU_GPU Vec3<T> ToLocal(Vec3<T> v) const { return to_local(v); }

	// Reconstruct world vector from local coords.
	CPU_GPU Vec3<T> from_local(Vec3<T> v) const {
		return x*v.x + y*v.y + z*v.z;
	}
	CPU_GPU Vec3<T> FromLocal(Vec3<T> v) const { return from_local(v); }
};
