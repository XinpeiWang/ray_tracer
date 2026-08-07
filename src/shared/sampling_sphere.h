#pragma once
#include "sampling_helpers.h"


// ===========================================================================
// Equal-Area Sphere Mapping -- pbrt-v4 util/math.cpp (Clarberg 2008)
//
// Maps the unit square [0,1]^2 bijectively to the unit sphere S^2 with
// uniform area distortion (equal-area property). Eliminates the polar
// singularity of equirectangular/lat-long mappings.
//
// Also provides SampleUniformHemisphereConcentric -- a hemisphere sampler
// that reuses the concentric disk mapping with a lifted z component,
// producing a more uniform distribution than the naive polar method.
//
// Functions:
//   EqualAreaSquareToSphere(u, v)         -> (wx, wy, wz) unit vector
//   EqualAreaSphereToSquare(wx,wy,wz)     -> (u, v) in [0,1]^2
//   WrapEqualAreaSquare(u, v)             -> wrapped (u,v) for tiling
//   SampleUniformHemisphereConcentric(u0,u1) -> (wx,wy,wz) upper hemi
//
// References:
//   pbrt-v4 util/math.cpp EqualAreaSquareToSphere / EqualAreaSphereToSquare
//   pbrt-v4 util/sampling.h SampleUniformHemisphereConcentric
//   Clarberg, "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD", 2008
// ===========================================================================

// Map (u,v) in [0,1]^2 to a unit direction on S^2 with equal-area property.
// pbrt-v4: EqualAreaSquareToSphere (util/math.cpp)
CPU_GPU void EqualAreaSquareToSphere(double u, double v,
											 double& wx, double& wy, double& wz) {
	// Transform to [-1,1]^2
	double uu = 2.0*u - 1.0, vv = 2.0*v - 1.0;
	double up = std::abs(uu), vp = std::abs(vv);

	// Signed distance from diagonal
	double signed_dist = 1.0 - (up + vp);
	double d = std::abs(signed_dist);
	double r = 1.0 - d;

	// Angle phi in [0, pi/2] based on position within quadrant
	double phi = (r == 0.0 ? 1.0 : (vp - up) / r + 1.0) * (3.14159265358979323846 / 4.0);

	// z coordinate: 1-r^2 with sign from hemisphere
	wz = std::copysign(1.0 - r*r, signed_dist);

	// xy components: cos/sin of phi, scaled by r*sqrt(2-r^2), with original signs
	double cos_phi = std::cos(phi);
	double sin_phi = std::sin(phi);
	double xy_r = r * std::sqrt(std::max(0.0, 2.0 - r*r));
	wx = std::copysign(cos_phi * xy_r, uu);
	wy = std::copysign(sin_phi * xy_r, vv);
}

// Map a unit direction to (u,v) in [0,1]^2 (inverse of EqualAreaSquareToSphere).
// pbrt-v4: EqualAreaSphereToSquare (util/math.cpp)
CPU_GPU void EqualAreaSphereToSquare(double wx, double wy, double wz,
											 double& u, double& v) {
	double x = std::abs(wx), y = std::abs(wy), z = std::abs(wz);
	double r = std::sqrt(std::max(0.0, 1.0 - z));
	double a = (x > y) ? x : y;
	double b = (x > y) ? y : x;
	b = (a == 0.0) ? 0.0 : b / a;

	const double t1 =  0.406758566246788489601959989e-5;
	const double t2 =  0.636226545274016134946890922156;
	const double t3 =  0.61572017898280213493197203466e-2;
	const double t4 = -0.247333733281268944196501420480;
	const double t5 =  0.881770664775316294736387951347e-1;
	const double t6 =  0.419038818029165735901852432784e-1;
	const double t7 = -0.251390972343483509333252996350e-1;
	double phi = t1 + b*(t2 + b*(t3 + b*(t4 + b*(t5 + b*(t6 + b*t7)))));
	if (x < y) phi = 1.0 - phi;

	double vv = phi * r;
	double uu = r - vv;
	if (wz < 0.0) { double tmp = uu; uu = 1.0 - vv; vv = 1.0 - tmp; }
	uu = std::copysign(uu, wx);
	vv = std::copysign(vv, wy);
	u = 0.5*(uu + 1.0);
	v = 0.5*(vv + 1.0);
}

// Wrap (u,v) outside [0,1]^2 back onto the equal-area square by mirroring.
// pbrt-v4: WrapEqualAreaSquare (util/math.cpp)
CPU_GPU void WrapEqualAreaSquare(double& u, double& v) {
	if (u < 0.0) { u = -u;      v = 1.0 - v; }
	else if (u > 1.0) { u = 2.0 - u; v = 1.0 - v; }
	if (v < 0.0) { u = 1.0 - u; v = -v; }
	else if (v > 1.0) { u = 1.0 - u; v = 2.0 - v; }
}

// Sample a direction uniformly on the upper hemisphere using the concentric
// disk mapping lifted to the hemisphere: wz = 1 - r^2, xy scaled by sqrt(2-r^2).
// Avoids the polar bunching of the naive polar method.
// pbrt-v4: SampleUniformHemisphereConcentric (util/sampling.h)
CPU_GPU void SampleUniformHemisphereConcentric(double u0, double u1,
													   double& wx, double& wy, double& wz) {
	// Map to [-1,1]^2
	double ux = 2.0*u0 - 1.0, uy = 2.0*u1 - 1.0;
	if (ux == 0.0 && uy == 0.0) { wx = 0.0; wy = 0.0; wz = 1.0; return; }

	double r, theta;
	if (std::abs(ux) > std::abs(uy)) {
		r = ux;
		theta = (3.14159265358979323846 / 4.0) * (uy / ux);
	} else {
		r = uy;
		theta = (3.14159265358979323846 / 2.0) - (3.14159265358979323846 / 4.0) * (ux / uy);
	}
	// Lift disk to hemisphere: z = 1-r^2, xy radius = r*sqrt(2-r^2)
	wz = 1.0 - r*r;
	double xy_scale = r * std::sqrt(std::max(0.0, 2.0 - r*r));
	wx = std::cos(theta) * xy_scale;
	wy = std::sin(theta) * xy_scale;
}

// ===========================================================================
// Cosine-weighted hemisphere sampling (no out_pdf overload)
// SampleUniformDiskPolar, InvertUniformDiskPolarSample,
// SampleUniformDiskConcentric, and InvertUniformDiskConcentricSample are
// defined in sampling_sphere_cone.h.
// pbrt-v4: SampleCosineHemisphere / CosineHemispherePDF (util/sampling.h)
// ===========================================================================

// Sample a cosine-weighted direction on the upper hemisphere via concentric disk.
// pbrt-v4: SampleCosineHemisphere
template<typename T>
CPU_GPU void SampleCosineHemisphere(T u0, T u1,
									T& wx, T& wy, T& wz) {
	T dx, dy;
	SampleUniformDiskConcentric(u0, u1, dx, dy);
	T z = std::sqrt(std::max(T(0), T(1) - dx*dx - dy*dy));
	wx = dx; wy = dy; wz = z;
}

// ===========================================================================
// Power-heuristic for MIS (Multiple Importance Sampling)
// pbrt-v4: PowerHeuristic (util/sampling.h)
// ===========================================================================

// Balance heuristic with beta=2: w = (n_f * f)^2 / ((n_f*f)^2 + (n_g*g)^2)
// pbrt-v4: if (IsInf(Sqr(f))) return 1 -- handles the case where one PDF
// dominates so heavily it overflows float.
CPU_GPU float PowerHeuristic(int nf, float f_pdf, int ng, float g_pdf) {
	float f = static_cast<float>(nf) * f_pdf;
	float g = static_cast<float>(ng) * g_pdf;
	float f2 = f * f;
	if (std::isinf(f2)) return 1.f;
	return f2 / (f2 + g * g);
}

// Note: LogisticPDF, SampleLogistic, TrimmedLogisticPDF, SampleTrimmedLogistic,
// InvertTrimmedLogisticSample, SmoothStepPDF, SampleSmoothStep,
// InvertSmoothStepSample, and VarianceEstimator are defined in
// scalar_distributions.h (which includes this file).
