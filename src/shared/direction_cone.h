// direction_cone.h
// Bounding cone for a set of unit directions, ported from pbrt-v4.
//
// A DirectionCone represents all unit vectors within angle theta of an axis w,
// i.e. { v : dot(v, w) >= cos(theta) }.  cosTheta = cos(theta) is stored.
// Special values:
//   cosTheta = +Infinity  => empty cone  (IsEmpty() == true)
//   cosTheta = -1         => entire sphere (EntireSphere())
//
// API (mirrors pbrt-v4 util/vecmath.h + util/vecmath.cpp):
//   DirectionCone(wx,wy,wz, cosTheta)  -- construct from axis + cosTheta
//   DirectionCone(wx,wy,wz)            -- single direction (cosTheta = 1)
//   IsEmpty()                          -- true when cone is invalid/empty
//   EntireSphere()                     -- static factory for the full sphere
//   Inside(cone, vx,vy,vz)            -- test whether direction is inside
//   BoundSubtendedDirections(bbox, px,py,pz) -- cone bounding a box from p
//   Union(a, b)                        -- smallest cone containing both
//   ClosestVectorInCone(cone, wpx,wpy,wpz, ox,oy,oz) -- nearest cone direction
//
// Dependencies: scalar_math.h  (SafeACos, SafeSqrt, Sqr, Degrees, kPi)
//
// References: pbrt-v4 src/pbrt/util/vecmath.h, util/vecmath.cpp (Apache-2.0)

#pragma once
#include <cmath>
#include <limits>
#include "scalar_math.h"

#ifndef CPU_GPU
#  ifdef __CUDACC__
#    define CPU_GPU __host__ __device__
#  else
#    define CPU_GPU
#  endif
#endif

#if defined(_MSC_VER)
#  pragma warning(push)
#endif

// ---------------------------------------------------------------------------
// Internal float3 helpers (local to this header; avoids ODR from sampling.h)
// ---------------------------------------------------------------------------
namespace dc_detail {

struct F3 { float x, y, z; };

CPU_GPU F3  add(F3 a, F3 b)       { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
CPU_GPU F3  sub(F3 a, F3 b)       { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
CPU_GPU F3  scale(F3 a, float s)  { return {a.x*s, a.y*s, a.z*s}; }
CPU_GPU float dot(F3 a, F3 b)     { return a.x*b.x + a.y*b.y + a.z*b.z; }
CPU_GPU float lenSq(F3 a)         { return dot(a, a); }
CPU_GPU float len(F3 a)           { return std::sqrt(lenSq(a)); }
CPU_GPU F3  normalize(F3 a)       { float l = len(a); return l > 0.f ? scale(a, 1.f/l) : F3{0,0,1}; }
CPU_GPU F3  cross(F3 a, F3 b) {
	return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
}

// Numerically stable angle between two unit vectors (pbrt-v4 AngleBetween)
CPU_GPU float angleBetween(F3 a, F3 b) {
	if (dot(a, b) < 0.f) {
		F3 s = add(a, b);
		float hl = len(s) * 0.5f;
		if (hl > 1.f) hl = 1.f;
		return float(scalar_math_detail::kPi) - 2.f * std::asin(hl);
	} else {
		F3 d = sub(b, a);
		float hl = len(d) * 0.5f;
		if (hl > 1.f) hl = 1.f;
		return 2.f * std::asin(hl);
	}
}

// Rotate unit vector v by angle theta (radians) around unit axis k (Rodrigues)
CPU_GPU F3 rotateAround(F3 v, F3 k, float theta) {
	float cosT = std::cos(theta);
	float sinT = std::sin(theta);
	F3 kcv = cross(k, v);
	float kdv = dot(k, v);
	// v*cos + (k x v)*sin + k*(k.v)*(1-cos)
	return add(add(scale(v, cosT), scale(kcv, sinT)), scale(k, kdv * (1.f - cosT)));
}

} // namespace dc_detail

// ===========================================================================
// DirectionCone
// ===========================================================================

struct DirectionCone {
	// Axis direction (unit vector) and cosine of half-angle.
	// cosTheta == +Infinity signals an empty cone.
	float wx = 0.f, wy = 0.f, wz = 1.f;
	float cosTheta = std::numeric_limits<float>::infinity();

	// Constructors
	DirectionCone() = default;

	// Construct from axis (vx,vy,vz) and cosine of half-angle.
	// The axis is normalised on construction.
	CPU_GPU DirectionCone(float vx, float vy, float vz, float cosT) {
		float l = std::sqrt(vx*vx + vy*vy + vz*vz);
		if (l > 0.f) { wx = vx/l; wy = vy/l; wz = vz/l; }
		cosTheta = cosT;
	}

	// Construct a cone containing exactly one direction (half-angle = 0).
	CPU_GPU explicit DirectionCone(float vx, float vy, float vz)
		: DirectionCone(vx, vy, vz, 1.f) {}

	CPU_GPU bool IsEmpty() const {
		return cosTheta == std::numeric_limits<float>::infinity();
	}

	// Factory: cone covering the entire sphere.
	CPU_GPU static DirectionCone EntireSphere() {
		return DirectionCone(0.f, 0.f, 1.f, -1.f);
	}
};

// ===========================================================================
// Inside: is direction (vx,vy,vz) inside the cone?
// ===========================================================================
CPU_GPU bool Inside(const DirectionCone& d, float vx, float vy, float vz) {
	if (d.IsEmpty()) return false;
	float l = std::sqrt(vx*vx + vy*vy + vz*vz);
	if (l == 0.f) return false;
	float dotVal = (d.wx*vx + d.wy*vy + d.wz*vz) / l;
	return dotVal >= d.cosTheta;
}

// ===========================================================================
// BoundSubtendedDirections: smallest DirectionCone bounding all directions
// from point p toward the axis-aligned bounding box [bMin, bMax].
// Mirrors pbrt-v4 BoundSubtendedDirections(Bounds3f, Point3f).
// ===========================================================================
CPU_GPU DirectionCone BoundSubtendedDirections(
	float bMinX, float bMinY, float bMinZ,
	float bMaxX, float bMaxY, float bMaxZ,
	float px,    float py,    float pz)
{
	// Bounding sphere of the box: centre = midpoint, radius = half-diagonal
	float cx = (bMinX + bMaxX) * 0.5f;
	float cy = (bMinY + bMaxY) * 0.5f;
	float cz = (bMinZ + bMaxZ) * 0.5f;
	float dx = bMaxX - bMinX, dy = bMaxY - bMinY, dz = bMaxZ - bMinZ;
	float radius = 0.5f * std::sqrt(dx*dx + dy*dy + dz*dz);

	// If p is inside the sphere, the cone covers the whole sphere
	float d2 = (cx-px)*(cx-px) + (cy-py)*(cy-py) + (cz-pz)*(cz-pz);
	if (d2 < radius * radius)
		return DirectionCone::EntireSphere();

	// Axis from p toward centre; half-angle = asin(radius / dist)
	float dist = std::sqrt(d2);
	float sin2ThetaMax = (radius * radius) / d2;
	float cosThetaMax  = SafeSqrt(1.f - sin2ThetaMax);
	return DirectionCone(cx - px, cy - py, cz - pz, cosThetaMax);
}

// ===========================================================================
// Union: smallest DirectionCone containing both a and b.
// Mirrors pbrt-v4 Union(DirectionCone, DirectionCone) in vecmath.cpp.
// ===========================================================================
CPU_GPU DirectionCone Union(const DirectionCone& a, const DirectionCone& b) {
	// Handle empty cones
	if (a.IsEmpty()) return b;
	if (b.IsEmpty()) return a;

	using namespace dc_detail;
	F3 wa = {a.wx, a.wy, a.wz};
	F3 wb = {b.wx, b.wy, b.wz};

	float theta_a = SafeACos(a.cosTheta);
	float theta_b = SafeACos(b.cosTheta);
	float theta_d = angleBetween(wa, wb);

	// One cone already contains the other
	static constexpr float kPi = float(scalar_math_detail::kPi);
	if (std::min(theta_d + theta_b, kPi) <= theta_a) return a;
	if (std::min(theta_d + theta_a, kPi) <= theta_b) return b;

	// Merged spread angle
	float theta_o = (theta_a + theta_d + theta_b) * 0.5f;
	if (theta_o >= kPi)
		return DirectionCone::EntireSphere();

	// Rotate wa toward wb by (theta_o - theta_a) around their cross product
	float theta_r = theta_o - theta_a;
	F3 wr = cross(wa, wb);
	if (lenSq(wr) == 0.f)
		return DirectionCone::EntireSphere();
	F3 w = rotateAround(wa, normalize(wr), theta_r);
	return DirectionCone(w.x, w.y, w.z, std::cos(theta_o));
}

// ===========================================================================
// ClosestVectorInCone: returns the direction in cone d that is closest to wp.
// Output written to (ox, oy, oz).
// Mirrors pbrt-v4 DirectionCone::ClosestVectorInCone(Vector3f wp).
// ===========================================================================
CPU_GPU void ClosestVectorInCone(
	const DirectionCone& d,
	float wpx, float wpy, float wpz,
	float& ox,  float& oy,  float& oz)
{
	using namespace dc_detail;
	F3 w  = {d.wx, d.wy, d.wz};
	F3 wp = normalize({wpx, wpy, wpz});

	// If wp is already inside the cone, return it directly
	if (dot(wp, w) > d.cosTheta) {
		ox = wp.x; oy = wp.y; oz = wp.z;
		return;
	}

	// Rotate wp to touch the cone boundary (pbrt-v4 formula)
	float sinTheta = -SafeSqrt(1.f - d.cosTheta * d.cosTheta);
	F3 a = cross(wp, w);
	float aLen = len(a);
	// Build the closest direction using the Rodrigues-style formula from pbrt-v4
	float c = d.cosTheta;
	float s = sinTheta / aLen;
	F3 result = {
		c * w.x + s * (w.x*(wp.y*w.y + wp.z*w.z) - wp.x*(w.y*w.y + w.z*w.z)),
		c * w.y + s * (w.y*(wp.x*w.x + wp.z*w.z) - wp.y*(w.x*w.x + w.z*w.z)),
		c * w.z + s * (w.z*(wp.x*w.x + wp.y*w.y) - wp.z*(w.x*w.x + w.y*w.y))
	};
	F3 nr = normalize(result);
	ox = nr.x; oy = nr.y; oz = nr.z;
}

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
