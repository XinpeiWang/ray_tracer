#pragma once
// ---------------------------------------------------------------------------
// sampling_sphere_cone.h -- Uniform sphere, hemisphere, cone, and disk
//                           sampling + inversion utilities.
//
// Direct ports of pbrt-v4 util/sampling.h:
//   SampleUniformDiskPolar         / InvertUniformDiskPolarSample
//   InvertUniformDiskConcentricSample
//   SampleUniformSphere            / UniformSpherePDF
//                                  / InvertUniformSphereSample
//   SampleUniformHemisphere        / UniformHemispherePDF
//                                  / InvertUniformHemisphereSample
//   SampleUniformCone              / UniformConePDF
//                                  / InvertUniformConeSample
//   InvertCosineHemisphereSample
//   InvertUniformTriangleSample
//
// Design rules (same as sampling.h):
//   - Plain functions, CPU_GPU tagged
//   - Templated on scalar type (float / double)
//   - No heap allocation, no virtual functions
// ---------------------------------------------------------------------------

#include "sampling.h"   // for SampleUniformDiskConcentric, SampleUniformTriangle,
						// SampleCosineHemisphere, CPU_GPU macro

#include <cmath>

// ===========================================================================
// Uniform disk -- polar parameterisation
// pbrt-v4: SampleUniformDiskPolar / InvertUniformDiskPolarSample
// ===========================================================================

// Maps u=(u0,u1) in [0,1)^2 to the unit disk via polar parameterisation.
// r = sqrt(u0), phi = 2*pi*u1
template<typename T>
CPU_GPU void SampleUniformDiskPolar(T u0, T u1, T& dx, T& dy) {
	T r   = std::sqrt(u0);
	T phi = T(2) * T(3.14159265358979323846) * u1;
	dx = r * std::cos(phi);
	dy = r * std::sin(phi);
}

// Invert SampleUniformDiskPolar: recover (u0,u1) from disk point (dx,dy).
// pbrt-v4: InvertUniformDiskPolarSample
template<typename T>
CPU_GPU void InvertUniformDiskPolarSample(T dx, T dy, T& u0, T& u1) {
	T phi = std::atan2(dy, dx);
	if (phi < T(0)) phi += T(2) * T(3.14159265358979323846);
	u0 = dx*dx + dy*dy;
	u1 = phi / (T(2) * T(3.14159265358979323846));
}

// Invert SampleUniformDiskConcentric: recover (u0,u1) from disk point (dx,dy).
// pbrt-v4: InvertUniformDiskConcentricSample (util/sampling.h)
template<typename T>
CPU_GPU void InvertUniformDiskConcentricSample(T dx, T dy, T& u0, T& u1) {
	constexpr T Pi      = T(3.14159265358979323846);
	constexpr T PiOver4 = Pi / T(4);
	constexpr T PiOver2 = Pi / T(2);

	T theta = std::atan2(dy, dx); // -pi..pi
	T r     = std::sqrt(dx*dx + dy*dy);

	T ox, oy; // result in [-1,1]^2
	if (std::abs(theta) < PiOver4 || std::abs(theta) > T(3)*PiOver4) {
		// x-dominant octants
		ox = std::copysign(r, dx);
		if (dx < T(0)) {
			oy = (dy < T(0) ? Pi + theta : theta - Pi) * ox / PiOver4;
		} else {
			oy = (theta * ox) / PiOver4;
		}
	} else {
		// y-dominant octants
		oy = std::copysign(r, dy);
		if (dy < T(0)) {
			ox = -(PiOver2 + theta) * oy / PiOver4;
		} else {
			ox = (PiOver2 - theta) * oy / PiOver4;
		}
	}
	u0 = (ox + T(1)) / T(2);
	u1 = (oy + T(1)) / T(2);
}

// ===========================================================================
// Uniform sphere
// pbrt-v4: SampleUniformSphere / UniformSpherePDF / InvertUniformSphereSample
// ===========================================================================

// Sample a direction uniformly over the full unit sphere.  pdf = 1/(4*pi)
template<typename T>
CPU_GPU void SampleUniformSphere(T u0, T u1, T& wx, T& wy, T& wz) {
	wz = T(1) - T(2)*u0;
	T r   = std::sqrt(std::max(T(0), T(1) - wz*wz));
	T phi = T(2) * T(3.14159265358979323846) * u1;
	wx = r * std::cos(phi);
	wy = r * std::sin(phi);
}

// PDF of SampleUniformSphere: 1/(4*pi)
template<typename T>
CPU_GPU T UniformSpherePDF() {
	return T(1) / (T(4) * T(3.14159265358979323846));
}

// Invert SampleUniformSphere: recover (u0,u1) from direction (wx,wy,wz).
template<typename T>
CPU_GPU void InvertUniformSphereSample(T wx, T wy, T wz, T& u0, T& u1) {
	T phi = std::atan2(wy, wx);
	if (phi < T(0)) phi += T(2) * T(3.14159265358979323846);
	u0 = (T(1) - wz) / T(2);
	u1 = phi / (T(2) * T(3.14159265358979323846));
}

// ===========================================================================
// Uniform hemisphere
// pbrt-v4: SampleUniformHemisphere / UniformHemispherePDF / InvertUniformHemisphereSample
// ===========================================================================

// Sample a direction uniformly over the upper hemisphere (wz >= 0).  pdf = 1/(2*pi)
template<typename T>
CPU_GPU void SampleUniformHemisphere(T u0, T u1, T& wx, T& wy, T& wz) {
	wz = u0;
	T r   = std::sqrt(std::max(T(0), T(1) - wz*wz));
	T phi = T(2) * T(3.14159265358979323846) * u1;
	wx = r * std::cos(phi);
	wy = r * std::sin(phi);
}

// PDF of SampleUniformHemisphere: 1/(2*pi)
template<typename T>
CPU_GPU T UniformHemispherePDF() {
	return T(1) / (T(2) * T(3.14159265358979323846));
}

// Invert SampleUniformHemisphere: recover (u0,u1) from direction (wx,wy,wz).
template<typename T>
CPU_GPU void InvertUniformHemisphereSample(T wx, T wy, T wz, T& u0, T& u1) {
	T phi = std::atan2(wy, wx);
	if (phi < T(0)) phi += T(2) * T(3.14159265358979323846);
	u0 = wz;
	u1 = phi / (T(2) * T(3.14159265358979323846));
}

// ===========================================================================
// Uniform cone  (around +z axis)
// pbrt-v4: SampleUniformCone / UniformConePDF / InvertUniformConeSample
// Samples directions inside a cone of half-angle arccos(cosThetaMax).
// pdf = 1 / (2*pi*(1-cosThetaMax))
// ===========================================================================

// Sample a direction uniformly inside a cone around +z.
template<typename T>
CPU_GPU void SampleUniformCone(T u0, T u1, T cosThetaMax,
									  T& wx, T& wy, T& wz) {
	T cosTheta  = (T(1) - u0) + u0 * cosThetaMax;
	T sinTheta  = std::sqrt(std::max(T(0), T(1) - cosTheta*cosTheta));
	T phi       = u1 * T(2) * T(3.14159265358979323846);
	wx = sinTheta * std::cos(phi);
	wy = sinTheta * std::sin(phi);
	wz = cosTheta;
}

// PDF for SampleUniformCone.
template<typename T>
CPU_GPU T UniformConePDF(T cosThetaMax) {
	return T(1) / (T(2) * T(3.14159265358979323846) * (T(1) - cosThetaMax));
}

// Invert SampleUniformCone: recover (u0,u1) from direction (wx,wy,wz).
template<typename T>
CPU_GPU void InvertUniformConeSample(T wx, T wy, T wz, T cosThetaMax,
											T& u0, T& u1) {
	T phi = std::atan2(wy, wx);
	if (phi < T(0)) phi += T(2) * T(3.14159265358979323846);
	u0 = (wz - T(1)) / (cosThetaMax - T(1));
	u1 = phi / (T(2) * T(3.14159265358979323846));
}

// ===========================================================================
// Invert cosine-hemisphere sample
// pbrt-v4: InvertCosineHemisphereSample (util/sampling.h)
// Recovers (u0,u1) from a cosine-hemisphere direction by inverting the
// concentric disk mapping applied to (wx, wy).
// ===========================================================================
template<typename T>
CPU_GPU void InvertCosineHemisphereSample(T wx, T wy, T /*wz*/, T& u0, T& u1) {
	InvertUniformDiskConcentricSample(wx, wy, u0, u1);
}

// ===========================================================================
// Invert SampleUniformTriangle
// pbrt-v4: InvertUniformTriangleSample (util/sampling.h)
// Given barycentric coords (b0,b1,b2) produced by SampleUniformTriangle,
// recover the original (u0,u1) sample.
// ===========================================================================
template<typename T>
CPU_GPU void InvertUniformTriangleSample(T b0, T b1, T /*b2*/, T& u0, T& u1) {
	if (b0 > b1) {
		// b0 = u0 - u1/2,  b1 = u1/2
		u0 = b0 + b1;
		u1 = T(2) * b1;
	} else {
		// b1 = u1 - u0/2,  b0 = u0/2
		u0 = T(2) * b0;
		u1 = b1 + b0;
	}
}
