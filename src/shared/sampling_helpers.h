#pragma once
// ---------------------------------------------------------------------------
// sampling.h -- Shared CPU/GPU sampling utilities
//
// Mirrors pbrt-v4 util/sampling.cpp (SampleSphericalRectangle) and
// shapes.h (Sphere solid-angle sampling small-angle fix).
//
// Key addition over Book-3:
//   SampleSphericalRectangle / SphericalRectanglePDF
//     -- Ureña et al. 2013 "An Area-Preserving Parametrization for
//        Spherical Rectangles". Samples a point on a quad directly and
//        uniformly in solid-angle measure from a reference point.
//        PDF = 1 / solidAngle  (constant, no distance² / cosine conversion).
//
// Design rules (same as bxdfs.h, noise.h):
//   - Plain structs/functions, CPU_GPU tagged
//   - No virtual functions, no heap allocation
//   - Uses double precision throughout (float on GPU via T)
// ---------------------------------------------------------------------------

#include "cpu_gpu.h"

#if defined(__CUDACC__)
#   include <math_functions.h>
#else
#   include <cmath>
#   include <cstring>
#endif
#include "scalar_math.h"
#include "vec3_frame.h"

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace sampling_detail {

template<typename T>
CPU_GPU T safe_sqrt(T x) {
	return SafeSqrt(x);
}

template<typename T>
CPU_GPU T clamp01(T x) {
	return x < T(0) ? T(0) : (x > T(1) ? T(1) : x);
}

// Vec3<T> / angle_between<T> / Frame<T> -- canonical definitions now live in
// vec3_frame.h (shared with pil_detail in portal_image_infinite_light.h),
// included above. Deliberately NOT re-declared as aliases in this namespace:
// this file's call sites reach them unqualified via
// "using namespace sampling_detail;" plus ordinary lookup finding the
// global ::Vec3/::Frame/::angle_between directly -- a same-named alias
// here would make both candidates visible at once and be ambiguous.

} // namespace sampling_detail

// ---------------------------------------------------------------------------
// SampleUniformDiskConcentric
// Shirley & Chiu 1997 area-preserving concentric mapping of [0,1)^2 to disk.
// Mirrors pbrt-v4 util/sampling.h SampleUniformDiskConcentric.
//
// Maps (u0,u1) to (dx,dy) on the unit disk with uniform area distribution.
// Uses the concentric mapping: every square annulus maps to a circular annulus
// with the same fractional area, preserving stratification quality.
//
// Returns: (dx, dy) on the unit disk (radius <= 1).
// ---------------------------------------------------------------------------
template<typename T>
CPU_GPU void SampleUniformDiskConcentric(T u0, T u1, T& dx, T& dy) {
    // Remap to [-1, 1]^2
    T ux = T(2) * u0 - T(1);
    T uy = T(2) * u1 - T(1);

    if (ux == T(0) && uy == T(0)) {
        dx = T(0); dy = T(0);
        return;
    }

    T r, theta;
    if (std::abs(ux) > std::abs(uy)) {
        // Map to first/fourth octant
        r = ux;
        theta = T(3.14159265358979323846 / 4.0) * (uy / ux);
    } else {
        // Map to second/third octant
        r = uy;
        theta = T(3.14159265358979323846 / 2.0) -
                T(3.14159265358979323846 / 4.0) * (ux / uy);
    }
    dx = r * std::cos(theta);
    dy = r * std::sin(theta);
}

// ---------------------------------------------------------------------------
// SampleCosineHemisphere
// Cosine-weighted hemisphere sample via concentric disk projection.
// Mirrors pbrt-v4 util/sampling.h SampleCosineHemisphere.
//
// Maps (u0,u1) to a direction on the unit hemisphere with pdf = cos(theta)/pi.
// The concentric disk mapping preserves stratification from [0,1)^2 samples.
//
// Returns: (wx, wy, wz) unit direction with wz >= 0; pdf via out_pdf.
// ---------------------------------------------------------------------------
template<typename T>
CPU_GPU void SampleCosineHemisphere(T u0, T u1, T& wx, T& wy, T& wz, T& out_pdf) {
    T dx, dy;
    SampleUniformDiskConcentric(u0, u1, dx, dy);
    T z2 = T(1) - dx*dx - dy*dy;
    wz = (z2 > T(0)) ? std::sqrt(z2) : T(0);
    wx = dx;
    wy = dy;
    // pdf = cos(theta) / pi
    const T inv_pi = T(1.0 / 3.14159265358979323846);
    out_pdf = wz * inv_pi;
}

// PDF for cosine-weighted hemisphere sampling.
// pdf(theta) = cos(theta) / pi
template<typename T>
CPU_GPU T CosineHemispherePDF(T cos_theta) {
    const T inv_pi = T(1.0 / 3.14159265358979323846);
    return (cos_theta > T(0)) ? cos_theta * inv_pi : T(0);
}

// ---------------------------------------------------------------------------
// SphericalRectangleSolidAngle
// Compute the solid angle subtended by a planar rectangle (quad) at point p.
//
// Parameters (match pbrt-v4 SampleSphericalRectangle):
//   p      : reference point (shading point)
//   s      : rectangle corner (origin of u/v edges)
//   ex     : first edge vector  (NOT normalised, length = width)
//   ey     : second edge vector (NOT normalised, length = height)
//
// Returns solid angle in steradians, or 0 if degenerate.
// ---------------------------------------------------------------------------
CPU_GPU double SphericalRectangleSolidAngle(
		double px, double py, double pz,
		double sx, double sy, double sz,
		double ex_x, double ex_y, double ex_z,
		double ey_x, double ey_y, double ey_z) {

	using namespace sampling_detail;
	using V3 = Vec3<double>;

	V3 p(px,py,pz), s(sx,sy,sz);
	V3 ex(ex_x, ex_y, ex_z), ey(ey_x, ey_y, ey_z);

	double exl = ex.length(), eyl = ey.length();
	if (exl < 1e-12 || eyl < 1e-12) return 0.0;

	Frame<double> R = Frame<double>::from_xy(ex*(1.0/exl), ey*(1.0/eyl));
	V3 dLocal = R.to_local(s - p);
	double z0 = dLocal.z;
	if (z0 > 0) { R.z = R.z * -1.0; z0 = -z0; }

	double x0 = dLocal.x, y0 = dLocal.y;
	double x1 = x0 + exl,  y1 = y0 + eyl;

	V3 v00(x0,y0,z0), v01(x0,y1,z0), v10(x1,y0,z0), v11(x1,y1,z0);
	V3 n0 = v00.cross(v10).normalized();
	V3 n1 = v10.cross(v11).normalized();
	V3 n2 = v11.cross(v01).normalized();
	V3 n3 = v01.cross(v00).normalized();

	double g0 = angle_between(n0*-1, n1);
	double g1 = angle_between(n1*-1, n2);
	double g2 = angle_between(n2*-1, n3);
	double g3 = angle_between(n3*-1, n0);

	double solidAngle = g0 + g1 + g2 + g3 - 2.0 * 3.14159265358979323846;
	return solidAngle > 0.0 ? solidAngle : 0.0;
}

// ---------------------------------------------------------------------------
// SphericalRectanglePDF
// Returns 1/solidAngle (constant over the whole quad), or 0 if degenerate.
// ---------------------------------------------------------------------------
CPU_GPU double SphericalRectanglePDF(
		double px, double py, double pz,
		double sx, double sy, double sz,
		double ex_x, double ex_y, double ex_z,
		double ey_x, double ey_y, double ey_z) {

	double sa = SphericalRectangleSolidAngle(px,py,pz, sx,sy,sz,
											 ex_x,ex_y,ex_z, ey_x,ey_y,ey_z);
	return (sa > 1e-10) ? 1.0 / sa : 0.0;
}

// ---------------------------------------------------------------------------
// SampleSphericalRectangle
// Direct port of pbrt-v4 SampleSphericalRectangle (util/sampling.cpp).
// Uniformly samples a point on the quad in solid-angle measure.
//
// Parameters:
//   p          : reference (shading) point
//   s          : rectangle corner (origin vertex)
//   ex, ey     : edge vectors (un-normalised)
//   u0, u1     : uniform random numbers in [0,1)
//   out_pdf    : (output) 1/solidAngle, or 0 if degenerate
//
// Returns: sampled point on the rectangle in world space.
//          Falls back to uniform area sample if solid angle is degenerate.
// ---------------------------------------------------------------------------
CPU_GPU void SampleSphericalRectangle(
		double  px,  double  py,  double  pz,     // reference point
		double  sx,  double  sy,  double  sz,     // rectangle corner s
		double ex_x, double ex_y, double ex_z,    // edge ex
		double ey_x, double ey_y, double ey_z,    // edge ey
		double u0,   double u1,                   // uniform samples [0,1)
		double* out_x, double* out_y, double* out_z,  // sampled point
		double* out_pdf) {                             // PDF

	using namespace sampling_detail;
	using V3 = Vec3<double>;

	const double Pi = 3.14159265358979323846;

	V3 p(px,py,pz), s(sx,sy,sz);
	V3 ex(ex_x,ex_y,ex_z), ey(ey_x,ey_y,ey_z);

	double exl = ex.length(), eyl = ey.length();

	// Build local frame aligned to rectangle edges
	Frame<double> R = Frame<double>::from_xy(ex*(1.0/exl), ey*(1.0/eyl));
	V3 dLocal = R.to_local(s - p);
	double z0 = dLocal.z;

	// Flip z so it points away from the quad (toward the reference point)
	if (z0 > 0) { R.z = R.z * -1.0; z0 = -z0; }
	double x0 = dLocal.x, y0 = dLocal.y;
	double x1 = x0 + exl,  y1 = y0 + eyl;

	// Compute edge plane normals
	V3 v00(x0,y0,z0), v01(x0,y1,z0), v10(x1,y0,z0), v11(x1,y1,z0);
	V3 n0 = v00.cross(v10).normalized();
	V3 n1 = v10.cross(v11).normalized();
	V3 n2 = v11.cross(v01).normalized();
	V3 n3 = v01.cross(v00).normalized();

	double g0 = angle_between(n0*-1, n1);
	double g1 = angle_between(n1*-1, n2);
	double g2 = angle_between(n2*-1, n3);
	double g3 = angle_between(n3*-1, n0);

	double solidAngle = g0 + g1 + g2 + g3 - 2.0 * Pi;

	// Degenerate: fall back to uniform area sample
	if (solidAngle <= 1e-10) {
		if (out_pdf) *out_pdf = 0.0;
		V3 q = s + ex*u0 + ey*u1;
		*out_x = q.x; *out_y = q.y; *out_z = q.z;
		return;
	}

	if (out_pdf) *out_pdf = 1.0 / solidAngle;

	// Small solid angle: fall back to area sample (pbrt-v4: < 1e-3 threshold)
	if (solidAngle < 1e-3) {
		V3 q = s + ex*u0 + ey*u1;
		*out_x = q.x; *out_y = q.y; *out_z = q.z;
		return;
	}

	// -- Ureña et al. 2013 sampling --

	// Sample cu (cosine of u-angle) by inverting solid angle CDF
	double b0 = n0.z, b1 = n2.z;
	double au = u0 * (g0 + g1 - 2.0*Pi) + (u0 - 1.0) * (g2 + g3);
	double fu = (std::cos(au) * b0 - b1) / std::sin(au);

	// copysign: cu has sign of fu, magnitude clamped to avoid NaN
	double fu2b0 = fu*fu + b0*b0;
	double cu = (fu >= 0.0 ? 1.0 : -1.0) / std::sqrt(fu2b0 > 1e-30 ? fu2b0 : 1e-30);
	cu = clamp01(std::abs(cu)) * (fu >= 0.0 ? 1.0 : -1.0);
	// Clamp to (-1+eps, 1-eps) to avoid corner NaNs
	const double OmEps = 1.0 - 1e-15;
	cu = cu < -OmEps ? -OmEps : (cu > OmEps ? OmEps : cu);

	// xu: x-coordinate of sample along ex edge
	double xu = -(cu * z0) / sampling_detail::safe_sqrt(1.0 - cu*cu);
	xu = xu < x0 ? x0 : (xu > x1 ? x1 : xu);  // clamp to [x0, x1]

	// yv: y-coordinate of sample along ey edge
	double dd  = std::sqrt(xu*xu + z0*z0);
	double h0  = y0 / std::sqrt(dd*dd + y0*y0);
	double h1  = y1 / std::sqrt(dd*dd + y1*y1);
	double hv  = h0 + u1 * (h1 - h0);
	double hvsq = hv * hv;
	double yv  = (hvsq < 1.0 - 1e-6) ? (hv * dd) / std::sqrt(1.0 - hvsq) : y1;

	// Map back to world space
	V3 local_pt(xu, yv, z0);
	V3 world_pt = p + R.from_local(local_pt);
	*out_x = world_pt.x;
	*out_y = world_pt.y;
	*out_z = world_pt.z;
}

// ---------------------------------------------------------------------------
// SphericalTriangleSolidAngle
// Solid angle of the triangle (v0,v1,v2) as seen from reference point p.
// Mirrors pbrt-v4: A = alpha + beta + gamma - Pi  (spherical excess)
// Returns 0 if degenerate.
// ---------------------------------------------------------------------------
CPU_GPU double SphericalTriangleSolidAngle(
		double v0x, double v0y, double v0z,
		double v1x, double v1y, double v1z,
		double v2x, double v2y, double v2z,
		double px,  double py,  double pz) {

	using namespace sampling_detail;
	using V3 = Vec3<double>;

	V3 a = V3(v0x-px, v0y-py, v0z-pz).normalized();
	V3 b = V3(v1x-px, v1y-py, v1z-pz).normalized();
	V3 c = V3(v2x-px, v2y-py, v2z-pz).normalized();

	V3 n_ab = a.cross(b); if (n_ab.length_squared() < 1e-30) return 0.0;
	V3 n_bc = b.cross(c); if (n_bc.length_squared() < 1e-30) return 0.0;
	V3 n_ca = c.cross(a); if (n_ca.length_squared() < 1e-30) return 0.0;
	n_ab = n_ab.normalized(); n_bc = n_bc.normalized(); n_ca = n_ca.normalized();

	double alpha = angle_between(n_ab,     n_ca*-1);
	double beta  = angle_between(n_bc,     n_ab*-1);
	double gamma = angle_between(n_ca,     n_bc*-1);

	double sa = alpha + beta + gamma - 3.14159265358979323846;
	return sa > 0.0 ? sa : 0.0;
}

// ---------------------------------------------------------------------------
// SampleSphericalTriangle
// Direct port of pbrt-v4 SampleSphericalTriangle (util/sampling.cpp:28-110).
// Uniformly samples a direction toward the triangle in solid-angle measure,
// returning barycentric coordinates (b0,b1,b2) of the sampled point.
//
// Parameters:
//   v0..v2  : triangle vertices in world space
//   p       : reference (shading) point
//   u0, u1  : uniform random numbers in [0,1)
//   out_b0..b2 : barycentric coordinates of the sampled point
//   out_pdf    : 1/solidAngle, or 0 if degenerate
// ---------------------------------------------------------------------------
CPU_GPU void SampleSphericalTriangle(
		double v0x, double v0y, double v0z,
		double v1x, double v1y, double v1z,
		double v2x, double v2y, double v2z,
		double px,  double py,  double pz,
		double u0,  double u1,
		double* out_b0, double* out_b1, double* out_b2,
		double* out_pdf) {

	using namespace sampling_detail;
	using V3 = Vec3<double>;
	const double Pi = 3.14159265358979323846;

	if (out_pdf) *out_pdf = 0.0;
	*out_b0 = *out_b1 = *out_b2 = 1.0/3.0;

	// Direction vectors from p to each vertex, normalised
	V3 a = V3(v0x-px, v0y-py, v0z-pz).normalized();
	V3 b = V3(v1x-px, v1y-py, v1z-pz).normalized();
	V3 c = V3(v2x-px, v2y-py, v2z-pz).normalized();

	// Edge plane normals
	V3 n_ab = a.cross(b); if (n_ab.length_squared() < 1e-30) return;
	V3 n_bc = b.cross(c); if (n_bc.length_squared() < 1e-30) return;
	V3 n_ca = c.cross(a); if (n_ca.length_squared() < 1e-30) return;
	n_ab = n_ab.normalized(); n_bc = n_bc.normalized(); n_ca = n_ca.normalized();

	// Internal angles at each vertex  (pbrt-v4: alpha, beta, gamma)
	double alpha = angle_between(n_ab, n_ca*-1);
	double beta  = angle_between(n_bc, n_ab*-1);
	double gamma = angle_between(n_ca, n_bc*-1);

	double A = alpha + beta + gamma - Pi;  // spherical excess = solid angle
	if (A <= 0.0) return;
	if (out_pdf) *out_pdf = 1.0 / A;

	// Sample a sub-area Ap uniformly in [pi, alpha+beta+gamma]
	double Ap_pi = u0 * (alpha + beta + gamma) + (1.0 - u0) * Pi;  // = Lerp(u0, Pi, A_pi)

	// Compute cosBp from Ap (pbrt-v4 cosBp formula)
	double cosAlpha = std::cos(alpha), sinAlpha = std::sin(alpha);
	double sinPhi   = std::sin(Ap_pi) * cosAlpha - std::cos(Ap_pi) * sinAlpha;
	double cosPhi   = std::cos(Ap_pi) * cosAlpha + std::sin(Ap_pi) * sinAlpha;
	double cosc     = a.dot(b);  // cos(arc c) = dot(a,b)
	double k1 = cosPhi + cosAlpha;
	double k2 = sinPhi - sinAlpha * cosc;
	// DifferenceOfProducts / SumOfProducts (exact)
	double num   = k2 + (k2*cosPhi - k1*sinPhi) * cosAlpha;
	double denom = (k2*sinPhi + k1*cosPhi) * sinAlpha;
	double cosBp = (std::abs(denom) > 1e-30) ? num / denom : 0.0;
	cosBp = cosBp < -1.0 ? -1.0 : (cosBp > 1.0 ? 1.0 : cosBp);

	// cp = cosBp*a + sinBp * GramSchmidt(c, a)
	double sinBp = std::sqrt(1.0 - cosBp*cosBp > 0.0 ? 1.0 - cosBp*cosBp : 0.0);
	// GramSchmidt(c, a) = normalise(c - dot(c,a)*a)
	V3 gs = c - a * c.dot(a);
	double gsl = gs.length();
	V3 gs_n = (gsl > 1e-15) ? gs * (1.0/gsl) : V3(0,0,0);
	V3 cp = a * cosBp + gs_n * sinBp;

	// Sample direction w along the arc between cp and b
	double cosTheta = 1.0 - u1 * (1.0 - cp.dot(b));
	double sinTheta = std::sqrt(1.0 - cosTheta*cosTheta > 0.0 ? 1.0 - cosTheta*cosTheta : 0.0);
	// GramSchmidt(cp, b)
	V3 gs2 = cp - b * cp.dot(b);
	double gs2l = gs2.length();
	V3 gs2_n = (gs2l > 1e-15) ? gs2 * (1.0/gs2l) : V3(0,0,0);
	V3 w = b * cosTheta + gs2_n * sinTheta;

	// Find barycentric coordinates for direction w via Möller-Trumbore
	V3 p_pt(px, py, pz);
	V3 v0(v0x,v0y,v0z), v1(v1x,v1y,v1z), v2(v2x,v2y,v2z);
	V3 e1 = v1 - v0, e2 = v2 - v0;
	V3 s1 = w.cross(e2);
	double divisor = s1.dot(e1);
	if (std::abs(divisor) < 1e-30) { *out_b0=*out_b1=*out_b2=1.0/3.0; return; }
	double inv = 1.0 / divisor;
	V3 s = p_pt - v0;
	double b1 = s.dot(s1) * inv;
	double b2 = w.dot(s.cross(e1)) * inv;

	// Clamp barycentrics (pbrt-v4: Clamp + normalise if b1+b2 > 1)
	b1 = b1 < 0.0 ? 0.0 : (b1 > 1.0 ? 1.0 : b1);
	b2 = b2 < 0.0 ? 0.0 : (b2 > 1.0 ? 1.0 : b2);
	if (b1 + b2 > 1.0) { double s = b1 + b2; b1 /= s; b2 /= s; }

	*out_b0 = 1.0 - b1 - b2;
	*out_b1 = b1;
	*out_b2 = b2;
}

