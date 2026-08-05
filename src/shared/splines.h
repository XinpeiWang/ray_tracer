// splines.h -- Cubic BÃƒÂ©zier and B-spline utilities, ported from pbrt-v4 util/splines.h
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// Apache License, Version 2.0.
//
// Functions:
//   BlossomCubicBezier<P>(cp, u0, u1, u2)       -- de Casteljau blossoming
//   EvaluateCubicBezier<P>(cp, u)               -- point on curve at u
//   EvaluateCubicBezierD(cp, u, deriv)          -- point + derivative (float[3])
//   SubdivideCubicBezier(cp, out7)              -- midpoint split -> 7 points
//   CubicBezierControlPoints(cp, uMin, uMax, out4) -- sub-interval reparametrize
//   BoundCubicBezierBox(cp, outMin, outMax)     -- AABB of 4 control points
//   ElevateQuadraticBezierToCubic(cp3, out4)    -- degree elevation Q->C
//   QuadraticBSplineToBezier(cp3, out3)         -- uniform quad B-spline -> Bezier
//   CubicBSplineToBezier(cp4, out4)             -- uniform cubic B-spline -> Bezier
//
// All functions are CPU/GPU compatible.
// Control points are passed as plain C arrays of float[3] (xyz).

#pragma once

#include <cmath>
#include <cstring>

#ifndef CPU_GPU
#  ifdef __CUDACC__
#    define CPU_GPU __host__ __device__
#  else
#    define CPU_GPU
#  endif
#endif

namespace splines {

// --------------------------------------------------------------------------
// Internal helpers
// --------------------------------------------------------------------------

// Lerp between two float[3] points: out = (1-t)*a + t*b
CPU_GPU void lerp3(float t, const float* a, const float* b, float* out) {
	out[0] = (1.f - t) * a[0] + t * b[0];
	out[1] = (1.f - t) * a[1] + t * b[1];
	out[2] = (1.f - t) * a[2] + t * b[2];
}

// Copy float[3]
CPU_GPU void copy3(const float* src, float* dst) {
	dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
}

// --------------------------------------------------------------------------
// BlossomCubicBezier
// Evaluates the blossom of the cubic BÃƒÂ©zier defined by cp[4] at (u0, u1, u2).
// Result written to out[3].
// Reference: pbrt-v4 BlossomCubicBezier
// --------------------------------------------------------------------------
CPU_GPU void BlossomCubicBezier(const float cp[4][3],
										float u0, float u1, float u2,
										float out[3]) {
	float a[3][3], b[2][3];
	lerp3(u0, cp[0], cp[1], a[0]);
	lerp3(u0, cp[1], cp[2], a[1]);
	lerp3(u0, cp[2], cp[3], a[2]);
	lerp3(u1, a[0], a[1], b[0]);
	lerp3(u1, a[1], a[2], b[1]);
	lerp3(u2, b[0], b[1], out);
}

// --------------------------------------------------------------------------
// EvaluateCubicBezier  (point only)
// --------------------------------------------------------------------------
CPU_GPU void EvaluateCubicBezier(const float cp[4][3], float u, float out[3]) {
	BlossomCubicBezier(cp, u, u, u, out);
}

// --------------------------------------------------------------------------
// EvaluateCubicBezierD  (point + derivative)
// deriv[3] receives 3*(cp2[1] - cp2[0]).
// Reference: pbrt-v4 EvaluateCubicBezier(Point3f, deriv*)
// --------------------------------------------------------------------------
CPU_GPU void EvaluateCubicBezierD(const float cp[4][3], float u,
										  float out[3], float deriv[3]) {
	float cp1[3][3], cp2[2][3];
	lerp3(u, cp[0], cp[1], cp1[0]);
	lerp3(u, cp[1], cp[2], cp1[1]);
	lerp3(u, cp[2], cp[3], cp1[2]);
	lerp3(u, cp1[0], cp1[1], cp2[0]);
	lerp3(u, cp1[1], cp1[2], cp2[1]);
	lerp3(u, cp2[0], cp2[1], out);

	// Derivative: 3*(cp2[1] - cp2[0]); fall back to cp[3]-cp[0] if degenerate.
	float dx = cp2[1][0] - cp2[0][0];
	float dy = cp2[1][1] - cp2[0][1];
	float dz = cp2[1][2] - cp2[0][2];
	float lenSq = dx*dx + dy*dy + dz*dz;
	if (lenSq > 0.f) {
		deriv[0] = 3.f * dx;
		deriv[1] = 3.f * dy;
		deriv[2] = 3.f * dz;
	} else {
		deriv[0] = cp[3][0] - cp[0][0];
		deriv[1] = cp[3][1] - cp[0][1];
		deriv[2] = cp[3][2] - cp[0][2];
	}
}

// --------------------------------------------------------------------------
// SubdivideCubicBezier
// Midpoint subdivision: splits cp[4] into out7[7] control points.
// out7[0..3] is the left half, out7[3..6] is the right half (shared midpoint).
// Reference: pbrt-v4 SubdivideCubicBezier
// --------------------------------------------------------------------------
CPU_GPU void SubdivideCubicBezier(const float cp[4][3], float out7[7][3]) {
	copy3(cp[0], out7[0]);

	float m01[3], m12[3], m23[3], m012[3], m123[3];
	lerp3(0.5f, cp[0], cp[1], m01);
	lerp3(0.5f, cp[1], cp[2], m12);
	lerp3(0.5f, cp[2], cp[3], m23);
	lerp3(0.5f, m01,  m12,  m012);
	lerp3(0.5f, m12,  m23,  m123);

	copy3(m01,  out7[1]);
	copy3(m012, out7[2]);
	lerp3(0.5f, m012, m123, out7[3]);
	copy3(m123, out7[4]);
	copy3(m23,  out7[5]);
	copy3(cp[3], out7[6]);
}

// --------------------------------------------------------------------------
// CubicBezierControlPoints
// Reparametrizes cp[4] to the sub-interval [uMin, uMax] -> out4[4].
// Reference: pbrt-v4 CubicBezierControlPoints
// --------------------------------------------------------------------------
CPU_GPU void CubicBezierControlPoints(const float cp[4][3],
											  float uMin, float uMax,
											  float out4[4][3]) {
	BlossomCubicBezier(cp, uMin, uMin, uMin, out4[0]);
	BlossomCubicBezier(cp, uMin, uMin, uMax, out4[1]);
	BlossomCubicBezier(cp, uMin, uMax, uMax, out4[2]);
	BlossomCubicBezier(cp, uMax, uMax, uMax, out4[3]);
}

// --------------------------------------------------------------------------
// BoundCubicBezierBox
// Loose AABB: union of bounding boxes of pairs of control points.
// outMin[3] and outMax[3] receive the result.
// Reference: pbrt-v4 BoundCubicBezier
// --------------------------------------------------------------------------
CPU_GPU void BoundCubicBezierBox(const float cp[4][3],
										 float outMin[3], float outMax[3]) {
	for (int i = 0; i < 3; ++i) {
		float a = cp[0][i] < cp[1][i] ? cp[0][i] : cp[1][i];
		float b = cp[0][i] > cp[1][i] ? cp[0][i] : cp[1][i];
		float c = cp[2][i] < cp[3][i] ? cp[2][i] : cp[3][i];
		float d = cp[2][i] > cp[3][i] ? cp[2][i] : cp[3][i];
		outMin[i] = a < c ? a : c;
		outMax[i] = b > d ? b : d;
	}
}

// --------------------------------------------------------------------------
// ElevateQuadraticBezierToCubic
// Degree elevation: quadratic cp3[3] -> cubic out4[4].
// Reference: pbrt-v4 ElevateQuadraticBezierToCubic
// --------------------------------------------------------------------------
CPU_GPU void ElevateQuadraticBezierToCubic(const float cp3[3][3],
												   float out4[4][3]) {
	copy3(cp3[0], out4[0]);
	lerp3(2.f / 3.f, cp3[0], cp3[1], out4[1]);
	lerp3(1.f / 3.f, cp3[1], cp3[2], out4[2]);
	copy3(cp3[2], out4[3]);
}

// --------------------------------------------------------------------------
// QuadraticBSplineToBezier
// Uniform quadratic B-spline segment cp3[3] -> BÃƒÂ©zier out3[3].
// Reference: pbrt-v4 QuadraticBSplineToBezier
// --------------------------------------------------------------------------
CPU_GPU void QuadraticBSplineToBezier(const float cp3[3][3], float out3[3][3]) {
	lerp3(0.5f, cp3[0], cp3[1], out3[0]);  // p11
	copy3(cp3[1], out3[1]);                  // p12
	lerp3(0.5f, cp3[1], cp3[2], out3[2]);  // p22
}

// --------------------------------------------------------------------------
// CubicBSplineToBezier
// Uniform cubic B-spline segment cp4[4] -> BÃƒÂ©zier out4[4].
// Reference: pbrt-v4 CubicBSplineToBezier
// --------------------------------------------------------------------------
CPU_GPU void CubicBSplineToBezier(const float cp4[4][3], float out4[4][3]) {
	float p012[3], p123[3], p234[3], p345[3];
	copy3(cp4[0], p012);
	copy3(cp4[1], p123);
	copy3(cp4[2], p234);
	copy3(cp4[3], p345);

	float p122[3], p223[3], p233[3], p334[3];
	lerp3(2.f / 3.f, p012, p123, p122);
	lerp3(1.f / 3.f, p123, p234, p223);
	lerp3(2.f / 3.f, p123, p234, p233);
	lerp3(1.f / 3.f, p234, p345, p334);

	lerp3(0.5f, p122, p223, out4[0]);  // p222
	copy3(p223, out4[1]);
	copy3(p233, out4[2]);
	lerp3(0.5f, p233, p334, out4[3]);  // p333
}

} // namespace splines
