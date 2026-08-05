// light_bounds.h
// Power-weighted directional bound for emitters, ported from pbrt-v4.
//
// LightBounds encodes everything needed to estimate a light's importance at a
// shading point without evaluating the full emission model:
//   - AABB of the emitter geometry
//   - Dominant emission direction w (unit vector)
//   - Total emitted power phi
//   - cosTheta_o: cosine of the emission spread cone (e.g. 0 for hemisphere)
//   - cosTheta_e: cosine of the effective emission cut-off (e.g. 0 for Lambertian)
//   - twoSided: whether the light emits from both faces
//
// API (mirrors pbrt-v4 lights.h + lights.cpp):
//   LightBounds(bMin,bMax, wx,wy,wz, phi, cosTheta_o, cosTheta_e, twoSided)
//   Importance(lb, px,py,pz, nx,ny,nz) -- estimated importance at point p / surface n
//   Union(a, b)                         -- merged LightBounds
//
// Dependencies:
//   direction_cone.h  -- DirectionCone, Union, BoundSubtendedDirections
//   scalar_math.h     -- SafeSqrt, Sqr
//
// References: pbrt-v4 src/pbrt/lights.h, src/pbrt/lights.cpp (Apache-2.0)

#pragma once
#include <cmath>
#include <algorithm>
#include "direction_cone.h"   // DirectionCone, Union, BoundSubtendedDirections
#include "scalar_math.h"      // SafeSqrt, Sqr

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

// ===========================================================================
// LightBounds
// ===========================================================================

struct LightBounds {
	// AABB of the emitter
	float bMinX = 0, bMinY = 0, bMinZ = 0;
	float bMaxX = 0, bMaxY = 0, bMaxZ = 0;

	// Dominant emission direction (unit vector)
	float wx = 0, wy = 0, wz = 1;

	// Total emitted power
	float phi = 0;

	// Emission cone cosines
	float cosTheta_o = 1;   // spread of emission (1 = single direction, -1 = omnidirectional)
	float cosTheta_e = 0;   // effective cut-off (Lambertian: 0)

	bool twoSided = false;

	// Default-construct an empty (zero-power) LightBounds
	LightBounds() = default;

	CPU_GPU LightBounds(
		float bMinX_, float bMinY_, float bMinZ_,
		float bMaxX_, float bMaxY_, float bMaxZ_,
		float wx_,   float wy_,   float wz_,
		float phi_,
		float cosTheta_o_, float cosTheta_e_,
		bool  twoSided_)
	{
		bMinX = bMinX_; bMinY = bMinY_; bMinZ = bMinZ_;
		bMaxX = bMaxX_; bMaxY = bMaxY_; bMaxZ = bMaxZ_;
		// Normalise w
		float len = std::sqrt(wx_*wx_ + wy_*wy_ + wz_*wz_);
		if (len > 0.f) { wx = wx_/len; wy = wy_/len; wz = wz_/len; }
		phi        = phi_;
		cosTheta_o = cosTheta_o_;
		cosTheta_e = cosTheta_e_;
		twoSided   = twoSided_;
	}

	CPU_GPU float CentroidX() const { return (bMinX + bMaxX) * 0.5f; }
	CPU_GPU float CentroidY() const { return (bMinY + bMaxY) * 0.5f; }
	CPU_GPU float CentroidZ() const { return (bMinZ + bMaxZ) * 0.5f; }
};

// ===========================================================================
// Importance(lb, p, n)
//
// Returns an estimate of lb's contribution at point (px,py,pz) on a surface
// with normal (nx,ny,nz).  Pass nx=ny=nz=0 for a non-surface query (e.g. a
// volume scatter point).
//
// Mirrors pbrt-v4 LightBounds::Importance(Point3f p, Normal3f n).
// ===========================================================================
CPU_GPU float Importance(const LightBounds& lb,
								 float px, float py, float pz,
								 float nx, float ny, float nz)
{
	// Clamped cosine-subtraction helpers (pbrt-v4 cosSubClamped / sinSubClamped)
	auto cosSubClamped = [](float sinA, float cosA, float sinB, float cosB) -> float {
		if (cosA > cosB) return 1.f;
		return cosA * cosB + sinA * sinB;
	};
	auto sinSubClamped = [](float sinA, float cosA, float sinB, float cosB) -> float {
		if (cosA > cosB) return 0.f;
		return sinA * cosB - cosA * sinB;
	};

	// Clamped squared distance to AABB centroid
	float cx = lb.CentroidX(), cy = lb.CentroidY(), cz = lb.CentroidZ();
	float dx = lb.bMaxX - lb.bMinX, dy = lb.bMaxY - lb.bMinY, dz = lb.bMaxZ - lb.bMinZ;
	float diagLen = std::sqrt(dx*dx + dy*dy + dz*dz);
	float d2raw = (cx-px)*(cx-px) + (cy-py)*(cy-py) + (cz-pz)*(cz-pz);
	float d2 = std::max(d2raw, diagLen * 0.5f);  // pbrt-v4: max(d2, Length(diagonal)/2)

	// Angle from light axis w to direction toward p
	float wix = px - cx, wiy = py - cy, wiz = pz - cz;
	float wiLen = std::sqrt(wix*wix + wiy*wiy + wiz*wiz);
	if (wiLen == 0.f) return 0.f;
	wix /= wiLen; wiy /= wiLen; wiz /= wiLen;

	float cosTheta_w = lb.wx*wix + lb.wy*wiy + lb.wz*wiz;
	if (lb.twoSided) cosTheta_w = std::abs(cosTheta_w);
	float sinTheta_w = SafeSqrt(1.f - Sqr(cosTheta_w));

	// Compute cosTheta_b for reference point (mirrors pbrt-v4 exactly)
	float cosTheta_b = BoundSubtendedDirections(
		lb.bMinX, lb.bMinY, lb.bMinZ, lb.bMaxX, lb.bMaxY, lb.bMaxZ, px, py, pz).cosTheta;
	float sinTheta_b = SafeSqrt(1.f - Sqr(cosTheta_b));

	// Compute cos(theta') = cos(theta_w - theta_o - theta_b) clamped
	float sinTheta_o = SafeSqrt(1.f - Sqr(lb.cosTheta_o));
	float cosTheta_x = cosSubClamped(sinTheta_w, cosTheta_w, sinTheta_o, lb.cosTheta_o);
	float sinTheta_x = sinSubClamped(sinTheta_w, cosTheta_w, sinTheta_o, lb.cosTheta_o);
	float cosThetap  = cosSubClamped(sinTheta_x, cosTheta_x, sinTheta_b, cosTheta_b);

	// If entirely outside the emission cone, importance is zero
	if (cosThetap <= lb.cosTheta_e) return 0.f;

	float importance = lb.phi * cosThetap / d2;

	// Account for cos(theta_i) at surface (skip for volume queries with n=0)
	float nLen = nx*nx + ny*ny + nz*nz;
	if (nLen > 0.f) {
		float cosTheta_i = std::abs(nx*wix + ny*wiy + nz*wiz);
		float sinTheta_i = SafeSqrt(1.f - Sqr(cosTheta_i));
		float cosThetap_i = cosSubClamped(sinTheta_i, cosTheta_i, sinTheta_b, cosTheta_b);
		importance *= cosThetap_i;
	}

	return std::max(importance, 0.f);
}

// ===========================================================================
// Union(LightBounds, LightBounds)
// Mirrors pbrt-v4 Union(const LightBounds &a, const LightBounds &b).
// ===========================================================================
CPU_GPU LightBounds Union(const LightBounds& a, const LightBounds& b) {
	// If one bound has zero power, return the other
	if (a.phi == 0.f) return b;
	if (b.phi == 0.f) return a;

	// Merge AABBs
	float mnX = std::min(a.bMinX, b.bMinX), mnY = std::min(a.bMinY, b.bMinY), mnZ = std::min(a.bMinZ, b.bMinZ);
	float mxX = std::max(a.bMaxX, b.bMaxX), mxY = std::max(a.bMaxY, b.bMaxY), mxZ = std::max(a.bMaxZ, b.bMaxZ);

	// Merge emission cones
	DirectionCone ca(a.wx, a.wy, a.wz, a.cosTheta_o);
	DirectionCone cb(b.wx, b.wy, b.wz, b.cosTheta_o);
	DirectionCone cu = Union(ca, cb);

	float cosTheta_e = std::min(a.cosTheta_e, b.cosTheta_e);

	return LightBounds(mnX, mnY, mnZ, mxX, mxY, mxZ,
					   cu.wx, cu.wy, cu.wz,
					   a.phi + b.phi,
					   cu.cosTheta, cosTheta_e,
					   a.twoSided | b.twoSided);
}

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
