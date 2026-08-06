#pragma once
// ---------------------------------------------------------------------------
// mis_sampling.h -- MIS heuristics and uniform sphere/cone/triangle samplers
//
// Ports of pbrt-v4 util/sampling.h functions:
//   BalanceHeuristic, PowerHeuristic
//   SampleUniformSphere / UniformSpherePDF / InvertUniformSphereSample
//   SampleUniformHemisphere / UniformHemispherePDF / InvertUniformHemisphereSample
//   SampleUniformCone / UniformConePDF / InvertUniformConeSample
//   SampleUniformTriangle / InvertUniformTriangleSample
//
// Design rules (same as sampling.h):
//   - Plain functions, CPU_GPU tagged
//   - No virtual functions, no heap allocation
//   - Uses double precision
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU inline
#   endif
#endif

#include "scalar_math.h"
#include <cmath>
#include <limits>

// ---------------------------------------------------------------------------
// MIS heuristics (pbrt-v4 util/sampling.h §13.10)
// ---------------------------------------------------------------------------

// BalanceHeuristic(nf, fPdf, ng, gPdf)
// Returns the balance-heuristic MIS weight for strategy f.
// pbrt-v4: BalanceHeuristic
CPU_GPU double BalanceHeuristic(int nf, double fPdf, int ng, double gPdf) {
	return (nf * fPdf) / (nf * fPdf + ng * gPdf);
}

// PowerHeuristic(nf, fPdf, ng, gPdf)
// Returns the power-heuristic (beta=2) MIS weight for strategy f.
// Numerically guarded: returns 1 if f^2 overflows to infinity.
// pbrt-v4: PowerHeuristic
CPU_GPU double PowerHeuristic(int nf, double fPdf, int ng, double gPdf) {
	double f = nf * fPdf, g = ng * gPdf;
	double f2 = f * f, g2 = g * g;
	if (f2 == std::numeric_limits<double>::infinity()) return 1.0;
	return f2 / (f2 + g2);
}

// ---------------------------------------------------------------------------
// SampleUniformSphere / UniformSpherePDF / InvertUniformSphereSample
// pdf = 1/(4*pi)
// pbrt-v4: SampleUniformSphere / UniformSpherePDF / InvertUniformSphereSample
// ---------------------------------------------------------------------------

CPU_GPU void SampleUniformSphere(double u0, double u1,
										double& wx, double& wy, double& wz) {
	namespace D = scalar_math_detail;
	double z   = 1.0 - 2.0 * u0;
	double r   = SafeSqrt(1.0 - z * z);
	double phi = 2.0 * D::kPi * u1;
	wx = r * std::cos(phi);
	wy = r * std::sin(phi);
	wz = z;
}

CPU_GPU double UniformSpherePDF() {
	return scalar_math_detail::kInv4Pi;
}

CPU_GPU void InvertUniformSphereSample(double wx, double wy, double wz,
											  double& u0, double& u1) {
	namespace D = scalar_math_detail;
	double phi = std::atan2(wy, wx);
	if (phi < 0.0) phi += 2.0 * D::kPi;
	u0 = (1.0 - wz) / 2.0;
	u1 = phi / (2.0 * D::kPi);
}

// ---------------------------------------------------------------------------
// SampleUniformHemisphere / UniformHemispherePDF / InvertUniformHemisphereSample
// pdf = 1/(2*pi),  wz >= 0
// pbrt-v4: SampleUniformHemisphere / UniformHemispherePDF
// ---------------------------------------------------------------------------

CPU_GPU void SampleUniformHemisphere(double u0, double u1,
											double& wx, double& wy, double& wz) {
	namespace D = scalar_math_detail;
	double z   = u0;
	double r   = SafeSqrt(1.0 - z * z);
	double phi = 2.0 * D::kPi * u1;
	wx = r * std::cos(phi);
	wy = r * std::sin(phi);
	wz = z;
}

CPU_GPU double UniformHemispherePDF() {
	return scalar_math_detail::kInv2Pi;
}

CPU_GPU void InvertUniformHemisphereSample(double wx, double wy, double wz,
												  double& u0, double& u1) {
	namespace D = scalar_math_detail;
	double phi = std::atan2(wy, wx);
	if (phi < 0.0) phi += 2.0 * D::kPi;
	u0 = wz;
	u1 = phi / (2.0 * D::kPi);
}

// ---------------------------------------------------------------------------
// SampleUniformCone / UniformConePDF / InvertUniformConeSample
// Cone of half-angle acos(cosThetaMax) about +Z.  pdf = 1/(2*pi*(1-cosThetaMax))
// pbrt-v4: SampleUniformCone / UniformConePDF / InvertUniformConeSample
// ---------------------------------------------------------------------------

CPU_GPU double UniformConePDF(double cosThetaMax) {
	namespace D = scalar_math_detail;
	return 1.0 / (2.0 * D::kPi * (1.0 - cosThetaMax));
}

CPU_GPU void SampleUniformCone(double u0, double u1, double cosThetaMax,
									  double& wx, double& wy, double& wz) {
	namespace D = scalar_math_detail;
	double cosTheta = (1.0 - u0) + u0 * cosThetaMax;
	double sinTheta = SafeSqrt(1.0 - cosTheta * cosTheta);
	double phi      = u1 * 2.0 * D::kPi;
	wx = std::cos(phi) * sinTheta;
	wy = std::sin(phi) * sinTheta;
	wz = cosTheta;
}

CPU_GPU void InvertUniformConeSample(double wx, double wy, double wz,
											double cosThetaMax,
											double& u0, double& u1) {
	namespace D = scalar_math_detail;
	double phi = std::atan2(wy, wx);
	if (phi < 0.0) phi += 2.0 * D::kPi;
	u0 = (wz - 1.0) / (cosThetaMax - 1.0);
	u1 = phi / (2.0 * D::kPi);
}

// ---------------------------------------------------------------------------
// SampleUniformTriangle / InvertUniformTriangleSample
// Maps (u0,u1) to barycentric coords (b0,b1,b2) with b0+b1+b2=1.
// pbrt-v4: SampleUniformTriangle / InvertUniformTriangleSample
// ---------------------------------------------------------------------------

CPU_GPU void SampleUniformTriangleD(double u0, double u1,
										   double& b0, double& b1, double& b2) {
	if (u0 < u1) {
		b0 = u0 * 0.5;
		b1 = u1 - b0;
	} else {
		b1 = u1 * 0.5;
		b0 = u0 - b1;
	}
	b2 = 1.0 - b0 - b1;
}

CPU_GPU void InvertUniformTriangleSample(double b0, double b1,
												double& u0, double& u1) {
	if (b0 > b1) {
		u0 = b0 + b1;
		u1 = 2.0 * b1;
	} else {
		u0 = 2.0 * b0;
		u1 = b1 + b0;
	}
}
