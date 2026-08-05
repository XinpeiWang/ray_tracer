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

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU inline
#   endif
#endif

#if defined(__CUDACC__)
#   include <math_functions.h>
#else
#   include <cmath>
#   include <cstring>
#endif
#include "scalar_math.h"

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace sampling_detail {

template<typename T>
CPU_GPU T safe_sqrt(T x) {
	return std::sqrt(x > T(0) ? x : T(0));
}

template<typename T>
CPU_GPU T clamp01(T x) {
	return x < T(0) ? T(0) : (x > T(1) ? T(1) : x);
}

// 3-component vector for internal use (avoids dependency on vec3.h)
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

// Minimal orthonormal Frame from two axes (pbrt-v4 Frame::FromXY)
template<typename T>
struct Frame {
	Vec3<T> x, y, z;

	CPU_GPU static Frame from_xy(Vec3<T> ex_n, Vec3<T> ey_n) {
		Frame f;
		f.x = ex_n;
		f.y = ey_n;
		f.z = ex_n.cross(ey_n).normalized();
		return f;
	}

	// Project world vector into local frame
	CPU_GPU Vec3<T> to_local(Vec3<T> v) const {
		return Vec3<T>(v.dot(x), v.dot(y), v.dot(z));
	}

	// Reconstruct world vector from local coords
	CPU_GPU Vec3<T> from_local(Vec3<T> v) const {
		return x*v.x + y*v.y + z*v.z;
	}
};

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

// ===========================================================================
// Bilinear sampling -- pbrt-v4 util/sampling.h
//
// Samples a point in [0,1]^2 from a bilinear distribution defined by four
// corner weights w[0..3] ordered as:
//   w[0]=f(0,0)  w[1]=f(1,0)  w[2]=f(0,1)  w[3]=f(1,1)
//
// References:
//   pbrt-v4 LinearPDF / SampleLinear / InvertLinearSample
//   pbrt-v4 BilinearPDF / SampleBilinear / InvertBilinearSample
// ===========================================================================

// PDF of a linear distribution on [0,1] with end-points a (at x=0) and b (at x=1).
// pbrt-v4: LinearPDF(x, a, b) = 2 * Lerp(x, a, b) / (a + b)
template<typename T>
CPU_GPU T LinearPDF(T x, T a, T b) {
	if (x < T(0) || x > T(1)) return T(0);
	T sum = a + b;
	if (sum == T(0)) return T(1);
	return T(2) * (a + (b - a) * x) / sum;
}

// Invert the CDF of the linear distribution.
// pbrt-v4: InvertLinearSample(x, a, b) = x * (a*(2-x) + b*x) / (a+b)
template<typename T>
CPU_GPU T InvertLinearSample(T x, T a, T b) {
	T sum = a + b;
	if (sum == T(0)) return x;
	return x * (a * (T(2) - x) + b * x) / sum;
}

// Sample x in [0,1) from a linear distribution with f(0)=a, f(1)=b.
// pbrt-v4: SampleLinear(u, a, b)
template<typename T>
CPU_GPU T SampleLinear(T u, T a, T b) {
	if (u == T(0) && a == T(0)) return T(0);
	T sum = a + b;
	if (sum == T(0)) return u;
	// Invert CDF: u = x*(2a + (b-a)*x) / (a+b)  =>  quadratic in x
	// Solved as: x = u*(a+b) / (a + sqrt(Lerp(u, a^2, b^2)))
	T lerp_sq = a * a + (b * b - a * a) * u;
	if (lerp_sq < T(0)) lerp_sq = T(0);
	T x = u * sum / (a + std::sqrt(lerp_sq));
	// Clamp to [0, 1-epsilon)
	if (x >= T(1)) x = T(1) - T(1e-7);
	return x;
}

// PDF of a bilinear distribution at point (px, py) in [0,1]^2.
// w[0]=f(0,0), w[1]=f(1,0), w[2]=f(0,1), w[3]=f(1,1).
// pbrt-v4: BilinearPDF(p, w)
template<typename T>
CPU_GPU T BilinearPDF(T px, T py, T w0, T w1, T w2, T w3) {
	if (px < T(0) || px > T(1) || py < T(0) || py > T(1)) return T(0);
	T total = w0 + w1 + w2 + w3;
	if (total == T(0)) return T(1);
	// Bilinear interpolation of weight at (px,py):
	//   f = (1-px)*(1-py)*w0 + px*(1-py)*w1 + (1-px)*py*w2 + px*py*w3
	T f = (T(1)-px)*(T(1)-py)*w0 + px*(T(1)-py)*w1
		+ (T(1)-px)*py       *w2 + px*py       *w3;
	return T(4) * f / total;
}

// Sample (px, py) from a bilinear distribution.
// Returns sampled point in [0,1)^2.
// pbrt-v4: SampleBilinear(u, w) -- marginal-y then conditional-x
template<typename T>
CPU_GPU void SampleBilinear(T u0, T u1, T w0, T w1, T w2, T w3,
							 T& out_px, T& out_py) {
	// Marginal: f_y(y) proportional to (w0+w1)*(1-y) + (w2+w3)*y
	out_py = SampleLinear(u1, w0 + w1, w2 + w3);
	// Conditional: f_x(x|y) proportional to lerp(y,w0,w2)*(1-x) + lerp(y,w1,w3)*x
	T ax = w0 + (w2 - w0) * out_py;
	T bx = w1 + (w3 - w1) * out_py;
	out_px = SampleLinear(u0, ax, bx);
}

// Invert SampleBilinear to recover (u0, u1) from a sampled point (px, py).
// pbrt-v4: InvertBilinearSample(p, w)
template<typename T>
CPU_GPU void InvertBilinearSample(T px, T py, T w0, T w1, T w2, T w3,
								   T& out_u0, T& out_u1) {
	T ax = w0 + (w2 - w0) * py;
	T bx = w1 + (w3 - w1) * py;
	out_u0 = InvertLinearSample(px, ax, bx);
	out_u1 = InvertLinearSample(py, w0 + w1, w2 + w3);
}

// ===========================================================================
// SampleUniformTriangle -- pbrt-v4 util/sampling.h
//
// Maps a uniform 2D sample (u0, u1) in [0,1)^2 to barycentric coordinates
// (b0, b1, b2) with b0+b1+b2 = 1, uniformly over a triangle.
//
// Uses the Heitz (2019) diagonal-fold method -- same as pbrt-v4:
//   if u0 < u1:  b0 = u0/2,       b1 = u1 - u0/2
//   else:        b0 = u0 - u1/2,  b1 = u1/2
//   b2 = 1 - b0 - b1
//
// Reference: pbrt-v4 SampleUniformTriangle (util/sampling.h)
// ===========================================================================
template<typename T>
CPU_GPU void SampleUniformTriangle(T u0, T u1, T& b0, T& b1, T& b2) {
	if (u0 < u1) {
		b0 = u0 * T(0.5);
		b1 = u1 - b0;
	} else {
		b1 = u1 * T(0.5);
		b0 = u0 - b1;
	}
	b2 = T(1) - b0 - b1;
}

// ===========================================================================
// Catmull-Rom spline CDF inversion -- pbrt-v4 util/sampling.cpp
//
// These functions implement importance sampling of a 1D or 2D function
// represented as a Catmull-Rom spline on a uniform or non-uniform grid.
//
// Key helpers:
//   find_interval       -- binary search for interval containing x
//   evaluate_polynomial -- Horner's method for polynomial evaluation
//   newton_bisection    -- hybrid Newton-bisection root finding
//   catmull_rom_weights -- compute spline interpolation weights
//   integrate_catmull_rom -- build CDF from spline (trapezoidal + derivative)
//   SampleCatmullRom    -- invert 1D Catmull-Rom CDF
//   SampleCatmullRom2D  -- invert 2D Catmull-Rom CDF (marginal + conditional)
//
// References:
//   pbrt-v4 util/math.h (CatmullRomWeights, IntegrateCatmullRom,
//            FindInterval, NewtonBisection, EvaluatePolynomial)
//   pbrt-v4 util/sampling.cpp (SampleCatmullRom, SampleCatmullRom2D)
// ===========================================================================

// Binary search: find largest i in [0, n-2] s.t. pred(i) is true.
// pbrt-v4: FindInterval (util/math.h)
template<typename Predicate>
inline int catmullrom_find_interval(int n, Predicate pred) {
	int size = n - 2, first = 1;
	while (size > 0) {
		int half   = size >> 1;
		int middle = first + half;
		if (pred(middle)) { first = middle + 1; size -= half + 1; }
		else              { size   = half; }
	}
	int r = first - 1;
	if (r < 0)   r = 0;
	if (r > n-2) r = n-2;
	return r;
}

// Evaluate polynomial by Horner's method.
// pbrt-v4: EvaluatePolynomial(t, c0, c1, ..., cn) = c0 + t*(c1 + t*(...))
inline double evaluate_polynomial(double /*t*/) { return 0.0; }
inline double evaluate_polynomial(double /*t*/, double c) { return c; }
template<typename... Rest>
inline double evaluate_polynomial(double t, double c0, Rest... rest) {
	return c0 + t * evaluate_polynomial(t, static_cast<double>(rest)...);
}

// Hybrid Newton-bisection: find root of f in [x0, x1].
// f must return {f(x), f'(x)}.
// pbrt-v4: NewtonBisection (util/math.h)
template<typename Func>
inline double catmullrom_newton_bisection(double x0, double x1, Func f,
										  double x_eps = 1e-6, double f_eps = 1e-6) {
	double fx0 = f(x0).first, fx1 = f(x1).first;
	if (std::abs(fx0) < f_eps) return x0;
	if (std::abs(fx1) < f_eps) return x1;
	bool neg_start = (fx0 < 0);
	double xMid = x0 + (x1 - x0) * (-fx0) / (fx1 - fx0);
	for (int iter = 0; iter < 64; ++iter) {
		if (!(x0 < xMid && xMid < x1))
			xMid = (x0 + x1) * 0.5;
		auto [fMid, dfMid] = f(xMid);
		if (std::abs(fMid) < f_eps || x1 - x0 < x_eps) break;
		if ((fMid < 0) == neg_start) x0 = xMid;
		else                          x1 = xMid;
		xMid -= fMid / dfMid;
	}
	return xMid;
}

// Compute Catmull-Rom interpolation weights for value x in the node array.
// Returns false if x is out of bounds.
// offset is set to idx-1 (first of 4 contributing nodes).
// pbrt-v4: CatmullRomWeights (util/math.cpp)
inline bool catmull_rom_weights(const double* nodes, int n, double x,
								 int* offset, double weights[4]) {
	if (!(x >= nodes[0] && x <= nodes[n-1])) return false;
	int idx = catmullrom_find_interval(n, [&](int i){ return nodes[i] <= x; });
	*offset = idx - 1;
	double x0 = nodes[idx], x1 = nodes[idx+1];
	double t  = (x - x0) / (x1 - x0), t2 = t*t, t3 = t2*t;
	weights[1] =  2*t3 - 3*t2 + 1;
	weights[2] = -2*t3 + 3*t2;
	// w0
	if (idx > 0) {
		double w0 = (t3 - 2*t2 + t) * (x1 - x0) / (x1 - nodes[idx-1]);
		weights[0] = -w0;
		weights[2] += w0;
	} else {
		double w0 = t3 - 2*t2 + t;
		weights[0] = 0;
		weights[1] -= w0;
		weights[2] += w0;
	}
	// w3
	if (idx + 2 < n) {
		double w3 = (t3 - t2) * (x1 - x0) / (nodes[idx+2] - x0);
		weights[1] -= w3;
		weights[3]  = w3;
	} else {
		double w3 = t3 - t2;
		weights[1] -= w3;
		weights[2] += w3;
		weights[3]  = 0;
	}
	return true;
}

// Build a CDF (cumulative integral) for a Catmull-Rom spline.
// cdf must have n entries; cdf[0] = 0. Returns total integral.
// pbrt-v4: IntegrateCatmullRom (util/math.cpp)
inline double integrate_catmull_rom(const double* nodes, const double* f,
									 int n, double* cdf) {
	double sum = 0;
	cdf[0] = 0;
	for (int i = 0; i < n-1; ++i) {
		double x0 = nodes[i], x1 = nodes[i+1];
		double f0 = f[i],     f1 = f[i+1];
		double w  = x1 - x0;
		double d0 = (i > 0)   ? w * (f1 - f[i-1]) / (x1 - nodes[i-1]) : (f1 - f0);
		double d1 = (i+2 < n) ? w * (f[i+2] - f0) / (nodes[i+2] - x0) : (f1 - f0);
		sum += w * ((f0 + f1) * 0.5 + (d0 - d1) / 12.0);
		cdf[i+1] = sum;
	}
	return sum;
}

// Sample a value from a 1D Catmull-Rom spline distribution.
//
// Parameters:
//   nodes  -- n node positions (sorted ascending)
//   f      -- n function values f(nodes[i])
//   F      -- n CDF values (precomputed by integrate_catmull_rom)
//   n      -- number of nodes
//   u      -- uniform sample in [0,1)
//   fval   -- (optional) output: interpolated f at sampled position
//   pdf    -- (optional) output: pdf at sampled position
//
// Returns the sampled position.
// pbrt-v4: SampleCatmullRom (util/sampling.cpp)
inline double SampleCatmullRom(const double* nodes, const double* f,
								const double* F, int n,
								double u, double* fval = nullptr, double* pdf = nullptr) {
	u *= F[n-1];
	int i = catmullrom_find_interval(n, [&](int k){ return F[k] <= u; });
	double x0 = nodes[i], x1 = nodes[i+1];
	double f0 = f[i],     f1 = f[i+1];
	double w  = x1 - x0;
	double d0 = (i > 0)   ? w * (f1 - f[i-1]) / (x1 - nodes[i-1]) : (f1 - f0);
	double d1 = (i+2 < n) ? w * (f[i+2] - f0) / (nodes[i+2] - x0) : (f1 - f0);
	u = (u - F[i]) / w;

	double Fhat = 0, fhat = 0;
	auto eval = [&](double t) -> std::pair<double,double> {
		Fhat = evaluate_polynomial(t, 0.0, f0, 0.5*d0,
				   (1.0/3.0)*(-2*d0 - d1) + f1 - f0,
					0.25*(d0 + d1) + 0.5*(f0 - f1));
		fhat = evaluate_polynomial(t, f0, d0,
				   -2*d0 - d1 + 3*(f1 - f0),
					d0 + d1 + 2*(f0 - f1));
		return {Fhat - u, fhat};
	};
	double t = catmullrom_newton_bisection(0.0, 1.0, eval);
	// Trigger final eval to update fhat
	eval(t);

	if (fval) *fval = fhat;
	if (pdf)  *pdf  = fhat / F[n-1];
	return x0 + w * t;
}

// Sample a value from a 2D Catmull-Rom spline distribution.
//
// The 2D table is indexed as values[i * n2 + j], cdf[i * n2 + j],
// where i indexes nodes1 (the "alpha" / first dimension to condition on)
// and j indexes nodes2 (the dimension to sample).
//
// Parameters:
//   nodes1  -- n1 node positions for the first (conditioning) dimension
//   nodes2  -- n2 node positions for the second (sampled) dimension
//   values  -- n1*n2 table of function values
//   cdf     -- n1*n2 precomputed CDF values (integrate_catmull_rom per row)
//   n1, n2  -- grid dimensions
//   alpha   -- value in nodes1 range (conditioning variable)
//   u       -- uniform sample in [0,1) (for sampling the second dimension)
//   fval    -- (optional) output: interpolated function value
//   pdf     -- (optional) output: pdf at sampled position
//
// Returns the sampled position in nodes2 range.
// pbrt-v4: SampleCatmullRom2D (util/sampling.cpp)
inline double SampleCatmullRom2D(const double* nodes1, const double* nodes2,
								  const double* values, const double* cdf,
								  int n1, int n2,
								  double alpha, double u,
								  double* fval = nullptr, double* pdf = nullptr) {
	int   offset;
	double weights[4];
	if (!catmull_rom_weights(nodes1, n1, alpha, &offset, weights))
		return 0.0;

	// Interpolate row values using the 4 Catmull-Rom weights
	auto interpolate = [&](const double* arr, int col) {
		double v = 0;
		for (int k = 0; k < 4; ++k)
			if (weights[k] != 0 && offset + k >= 0 && offset + k < n1)
				v += arr[(offset + k) * n2 + col] * weights[k];
		return v;
	};

	double maximum = interpolate(cdf, n2-1);
	u *= maximum;
	int idx = catmullrom_find_interval(n2, [&](int j){ return interpolate(cdf, j) <= u; });

	double f0 = interpolate(values, idx),   f1 = interpolate(values, idx+1);
	double x0 = nodes2[idx],               x1 = nodes2[idx+1];
	double w  = x1 - x0;
	double d0 = (idx > 0)    ? w * (f1 - interpolate(values, idx-1)) / (x1 - nodes2[idx-1])
							  : f1 - f0;
	double d1 = (idx+2 < n2) ? w * (interpolate(values, idx+2) - f0) / (nodes2[idx+2] - x0)
							  : f1 - f0;
	u = (u - interpolate(cdf, idx)) / w;

	double Fhat = 0, fhat = 0;
	auto eval = [&](double t) -> std::pair<double,double> {
		Fhat = evaluate_polynomial(t, 0.0, f0, 0.5*d0,
				   (1.0/3.0)*(-2*d0 - d1) + f1 - f0,
					0.25*(d0 + d1) + 0.5*(f0 - f1));
		fhat = evaluate_polynomial(t, f0, d0,
				   -2*d0 - d1 + 3*(f1 - f0),
					d0 + d1 + 2*(f0 - f1));
		return {Fhat - u, fhat};
	};
	double t = catmullrom_newton_bisection(0.0, 1.0, eval);
	eval(t);

	if (fval) *fval = fhat;
	if (pdf)  *pdf  = fhat / maximum;
	return x0 + w * t;
}

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

// =============================================================================
// Scalar 1-D distribution family  --  pbrt-v4 util/sampling.h
//
// Each entry provides three functions:
//   XxxPDF(x, ...)            -- probability density at x
//   SampleXxx(u, ...)         -- map uniform u in [0,1) to a sample
//   InvertXxxSample(x, ...)   -- invert the CDF (recover u from x)
//
// Families ported:
//   Tent             -- triangular filter, support [-r, r]
//   Exponential      -- e^(-a*x), x >= 0
//   TrimmedExponential -- e^(-c*x), x in [0, xMax]
//   Normal / TwoNormal -- Gaussian via ErfInv / Box-Muller
//   Logistic / TrimmedLogistic -- heavy-tailed; used in hair BxDF
//   SmoothStep       -- C1-smooth bump on [a,b]
//
// All helpers (SmoothStep, Gaussian, Logistic, ErfInv, NewtonBisection)
// live in scalar_math.h which is already included above.
// =============================================================================

// Tent distribution  -- support [-r, r]
// PDF(x) = 1/r - |x|/r^2
// pbrt-v4: TentPDF / SampleTent / InvertTentSample
CPU_GPU inline float TentPDF(float x, float r) {
	if (std::abs(x) >= r) return 0.f;
	return 1.f / r - std::abs(x) / (r * r);
}

CPU_GPU inline float SampleTent(float u, float r) {
	if (u < 0.5f) {
		u = 2.f * u;
		return -r + r * SampleLinear(u, 0.f, 1.f);
	} else {
		u = 2.f * (u - 0.5f);
		return r * SampleLinear(u, 1.f, 0.f);
	}
}

CPU_GPU inline float InvertTentSample(float x, float r) {
	if (x <= 0.f)
		return (1.f - InvertLinearSample(-x / r, 1.f, 0.f)) * 0.5f;
	else
		return 0.5f + InvertLinearSample(x / r, 1.f, 0.f) * 0.5f;
}

// Exponential distribution  -- support [0, inf)
// pbrt-v4: ExponentialPDF / SampleExponential / InvertExponentialSample
CPU_GPU inline float ExponentialPDF(float x, float a) {
	return a * std::exp(-a * x);
}

CPU_GPU inline float SampleExponential(float u, float a) {
	return -std::log(1.f - u) / a;
}

CPU_GPU inline float InvertExponentialSample(float x, float a) {
	return 1.f - std::exp(-a * x);
}

// TrimmedExponential -- e^(-c*x) on [0, xMax]
// pbrt-v4: TrimmedExponentialPDF / SampleTrimmedExponential / InvertTrimmedExponentialSample
CPU_GPU inline float TrimmedExponentialPDF(float x, float c, float xMax) {
	if (x < 0.f || x > xMax) return 0.f;
	return c / (1.f - std::exp(-c * xMax)) * std::exp(-c * x);
}

CPU_GPU inline float SampleTrimmedExponential(float u, float c, float xMax) {
	return std::log(1.f - u * (1.f - std::exp(-c * xMax))) / -c;
}

CPU_GPU inline float InvertTrimmedExponentialSample(float x, float c, float xMax) {
	return (1.f - std::exp(-c * x)) / (1.f - std::exp(-c * xMax));
}

// Normal (Gaussian) distribution
// pbrt-v4: NormalPDF / SampleNormal / InvertNormalSample / SampleTwoNormal
CPU_GPU inline float NormalPDF(float x, float mu = 0.f, float sigma = 1.f) {
	return Gaussian(x, mu, sigma);
}

CPU_GPU inline float SampleNormal(float u, float mu = 0.f, float sigma = 1.f) {
	static constexpr float kSqrt2 = 1.41421356237f;
	return mu + kSqrt2 * sigma * ErfInv(2.f * u - 1.f);
}

CPU_GPU inline float InvertNormalSample(float x, float mu = 0.f, float sigma = 1.f) {
	static constexpr float kSqrt2 = 1.41421356237f;
	return 0.5f * (1.f + std::erf((x - mu) / (sigma * kSqrt2)));
}

// Box-Muller: two independent N(mu, sigma^2) samples
CPU_GPU inline void SampleTwoNormal(float u0, float u1,
									float& out0, float& out1,
									float mu = 0.f, float sigma = 1.f) {
	float r2  = -2.f * std::log(std::max(1.f - u0, std::numeric_limits<float>::min()));
	float phi = 6.28318530717958647692f * u1;
	out0 = mu + sigma * std::sqrt(r2) * std::cos(phi);
	out1 = mu + sigma * std::sqrt(r2) * std::sin(phi);
}

// Logistic / TrimmedLogistic distribution
// pbrt-v4: LogisticPDF / SampleLogistic / InvertLogisticSample
//          TrimmedLogisticPDF / SampleTrimmedLogistic / InvertTrimmedLogisticSample
CPU_GPU inline float LogisticPDF(float x, float s) {
	x = std::abs(x);
	float ex = std::exp(-x / s);
	return ex / (s * (1.f + ex) * (1.f + ex));
}

CPU_GPU inline float SampleLogistic(float u, float s) {
	return -s * std::log(1.f / u - 1.f);
}

CPU_GPU inline float InvertLogisticSample(float x, float s) {
	return 1.f / (1.f + std::exp(-x / s));
}

CPU_GPU inline float TrimmedLogisticPDF(float x, float s, float a, float b) {
	if (x < a || x > b) return 0.f;
	return LogisticPDF(x, s) / (InvertLogisticSample(b, s) - InvertLogisticSample(a, s));
}

CPU_GPU inline float SampleTrimmedLogistic(float u, float s, float a, float b) {
	float pa = InvertLogisticSample(a, s);
	float pb = InvertLogisticSample(b, s);
	float x  = SampleLogistic(pa + u * (pb - pa), s);
	return std::max(a, std::min(b, x));
}

CPU_GPU inline float InvertTrimmedLogisticSample(float x, float s, float a, float b) {
	float pa = InvertLogisticSample(a, s);
	float pb = InvertLogisticSample(b, s);
	return (InvertLogisticSample(x, s) - pa) / (pb - pa);
}

// SmoothStep distribution  -- C1-smooth bump on [a, b]
// pbrt-v4: SmoothStepPDF / SampleSmoothStep / InvertSmoothStepSample
CPU_GPU inline float SmoothStepPDF(float x, float a, float b) {
	if (x < a || x > b) return 0.f;
	return (2.f / (b - a)) * SmoothStep(x, a, b);
}

CPU_GPU inline float SampleSmoothStep(float u, float a, float b) {
	auto cdfMinusU = [=](float x) -> std::pair<float,float> {
		float t  = (x - a) / (b - a);
		float P  = 2.f * t * t * t - t * t * t * t;
		float dP = SmoothStepPDF(x, a, b);
		return {P - u, dP};
	};
	return NewtonBisection(a, b, cdfMinusU);
}

CPU_GPU inline float InvertSmoothStepSample(float x, float a, float b) {
	float t = (x - a) / (b - a);
	return 2.f * t * t * t - t * t * t * t;
}

// =============================================================================
// InvertSphericalRectangleSample
// Direct port of pbrt-v4 InvertSphericalRectangleSample (util/sampling.cpp:222-346).
//
// Recovers the uniform sample (u0, u1) in [0,1)^2 that SampleSphericalRectangle
// would have produced for the given hit-point pRect on the rectangle.
//
// Parameters:
//   px, py, pz    : reference (shading) point
//   sx, sy, sz    : rectangle origin (corner v00)
//   ex*, ey*      : unnormalised edge vectors (length = width/height)
//   prx, pry, prz : the sampled point on the rectangle (world space)
//
// Returns (u0, u1) via out_u0, out_u1.  Returns (0,0) on degenerate input.
// =============================================================================
CPU_GPU void InvertSphericalRectangleSample(
		double px,  double py,  double pz,
		double sx,  double sy,  double sz,
		double exx, double exy, double exz,
		double eyx, double eyy, double eyz,
		double prx, double pry, double prz,
		double* out_u0, double* out_u1)
{
	using namespace sampling_detail;
	using V3 = Vec3<double>;
	const double Pi = 3.14159265358979323846;

	*out_u0 = *out_u1 = 0.0;

	V3 ex(exx, exy, exz), ey(eyx, eyy, eyz);
	double exl = ex.length(), eyl = ey.length();
	if (exl < 1e-30 || eyl < 1e-30) return;

	// Build local reference frame R (identical to SampleSphericalRectangle)
	Frame<double> R = Frame<double>::from_xy(ex * (1.0/exl), ey * (1.0/eyl));

	// Compute rectangle coords in local frame
	V3 d(sx - px, sy - py, sz - pz);
	double z0  = R.z.dot(d);
	if (z0 > 0.0) { R.z = R.z * -1.0; z0 = -z0; }
	double z0sq = z0 * z0;
	double x0  = R.x.dot(d),  x1 = x0 + exl;
	double y0  = R.y.dot(d),  y1 = y0 + eyl;
	double y0sq = y0 * y0,    y1sq = y1 * y1;

	// Edge plane normals
	V3 v00(x0,y0,z0), v01(x0,y1,z0), v10(x1,y0,z0), v11(x1,y1,z0);
	V3 n0 = v00.cross(v10).normalized();
	V3 n1 = v10.cross(v11).normalized();
	V3 n2 = v11.cross(v01).normalized();
	V3 n3 = v01.cross(v00).normalized();

	double g0 = angle_between(n0 * -1.0, n1);
	double g1 = angle_between(n1 * -1.0, n2);
	double g2 = angle_between(n2 * -1.0, n3);
	double g3 = angle_between(n3 * -1.0, n0);
	double solidAngle = g0 + g1 + g2 + g3 - 2.0 * Pi;

	if (solidAngle < 1e-3) {
		// Degenerate: fall back to planar coordinates
		V3 pq(prx - sx, pry - sy, prz - sz);
		*out_u0 = pq.dot(ex) / (exl * exl);
		*out_u1 = pq.dot(ey) / (eyl * eyl);
		*out_u0 = *out_u0 < 0.0 ? 0.0 : (*out_u0 > 1.0 ? 1.0 : *out_u0);
		*out_u1 = *out_u1 < 0.0 ? 0.0 : (*out_u1 > 1.0 ? 1.0 : *out_u1);
		return;
	}

	double b0 = n0.z, b1 = n2.z;
	double b0sq = b0 * b0;

	// Transform pRect into local frame to get (xu, yv)
	V3 pRect(prx - px, pry - py, prz - pz);
	double xu = R.x.dot(pRect);
	double yv = R.y.dot(pRect);
	xu = xu < x0 ? x0 : (xu > x1 ? x1 : xu);
	if (xu == 0.0) xu = 1e-10;

	// Invert the x-direction (cu) to recover u0.
	// Mirrors pbrt-v4 exactly: sqrt(DifferenceOfProducts(b0,b0,b1,b1)+fusq)
	// then atan2 with copysign(b0*sqrt, fu*b0).
	double invcusq = 1.0 + z0sq / (xu * xu);
	double fusq = invcusq - b0sq;
	if (fusq < 0.0) fusq = 0.0;
	double fu = xu > 0.0 ? std::sqrt(fusq) : -std::sqrt(fusq);

	// pbrt-v4: SafeSqrt(DifferenceOfProducts(b0,b0,b1,b1) + fusq)
	double sq_arg = b0 * b0 - b1 * b1 + fusq;
	double sqrt_val = sq_arg > 0.0 ? std::sqrt(sq_arg) : 0.0;
	// pbrt-v4: atan2(-(b1*fu) - copysign(b0*sqrt, fu*b0), b0*b1 - sqrt*|fu|)
	double au = std::atan2(-(b1 * fu) - std::copysign(b0 * sqrt_val, fu * b0),
							b0 * b1 - sqrt_val * std::abs(fu));
	if (au > 0.0) au -= 2.0 * Pi;
	if (fu == 0.0) au = Pi;

	*out_u0 = (au + g2 + g3) / solidAngle;
	*out_u0 = *out_u0 < 0.0 ? 0.0 : (*out_u0 > 1.0 ? 1.0 : *out_u0);

	// Invert the y-direction (yv) to recover u1
	double ddsq = xu * xu + z0sq;
	double dd   = std::sqrt(ddsq);
	double h0   = y0 / std::sqrt(ddsq + y0sq);
	double h1_v = y1 / std::sqrt(ddsq + y1sq);

	double yvsq = yv * yv;
	double det  = (h0 - h1_v) * (h0 - h1_v);
	if (det < 1e-30) { *out_u1 = 0.5; return; }

	// Two candidate solutions; pick the one whose reconstructed y matches yv
	double disc = std::abs(h0 - h1_v) * std::sqrt(yvsq * (ddsq + yvsq)) / (ddsq + yvsq);
	double nom  = h0 * h0 - h0 * h1_v;  // = DifferenceOfProducts(h0,h0, h0,h1_v)
	double u1a  = (nom - disc) / det;
	double u1b  = (nom + disc) / det;

	double hva  = h0 + u1a * (h1_v - h0);
	double hvb  = h0 + u1b * (h1_v - h0);
	double hvasq = hva * hva, hvbsq = hvb * hvb;
	double yza  = (hvasq < 1.0 - 1e-6) ? hva * dd / std::sqrt(1.0 - hvasq) : y1;
	double yzb  = (hvbsq < 1.0 - 1e-6) ? hvb * dd / std::sqrt(1.0 - hvbsq) : y1;

	double u1 = (std::abs(yza - yv) < std::abs(yzb - yv)) ? u1a : u1b;
	*out_u1 = u1 < 0.0 ? 0.0 : (u1 > 1.0 ? 1.0 : u1);
}

// =============================================================================
// InvertSphericalTriangleSample
// Direct port of pbrt-v4 InvertSphericalTriangleSample (util/sampling.cpp:110-160).
// (Via Jim Arvo's SphTri.C)
//
// Recovers (u0, u1) in [0,1)^2 such that SampleSphericalTriangle with those
// samples would produce a direction toward the point on the triangle closest
// to direction w from reference point p.
//
// Parameters:
//   v0..v2  : triangle vertices (world space)
//   px,py,pz: reference (shading) point
//   wx,wy,wz: the sampled unit direction
//   out_u0, out_u1: recovered canonical samples
// =============================================================================
CPU_GPU void InvertSphericalTriangleSample(
		double v0x, double v0y, double v0z,
		double v1x, double v1y, double v1z,
		double v2x, double v2y, double v2z,
		double px,  double py,  double pz,
		double wx,  double wy,  double wz,
		double* out_u0, double* out_u1)
{
	using namespace sampling_detail;
	using V3 = Vec3<double>;
	const double Pi = 3.14159265358979323846;

	*out_u0 = *out_u1 = 0.5;

	V3 a(v0x-px, v0y-py, v0z-pz); if (a.length_squared() < 1e-30) return; a = a.normalized();
	V3 b(v1x-px, v1y-py, v1z-pz); if (b.length_squared() < 1e-30) return; b = b.normalized();
	V3 c(v2x-px, v2y-py, v2z-pz); if (c.length_squared() < 1e-30) return; c = c.normalized();

	V3 n_ab = a.cross(b); if (n_ab.length_squared() < 1e-30) return; n_ab = n_ab.normalized();
	V3 n_bc = b.cross(c); if (n_bc.length_squared() < 1e-30) return; n_bc = n_bc.normalized();
	V3 n_ca = c.cross(a); if (n_ca.length_squared() < 1e-30) return; n_ca = n_ca.normalized();

	double alpha = angle_between(n_ab,     n_ca * -1.0);
	double beta  = angle_between(n_bc,     n_ab * -1.0);
	double gamma = angle_between(n_ca,     n_bc * -1.0);
	double A     = alpha + beta + gamma - Pi;
	if (A <= 0.0) return;

	// Find c' = intersection of plane(b,w) with arc(a,c)
	V3 w(wx, wy, wz);
	V3 cp = b.cross(w).cross(c.cross(a)).normalized();
	if (cp.dot(a + c) < 0.0) cp = cp * -1.0;

	double u0;
	if (a.dot(cp) > 0.99999847691) {   // < 0.1 degrees: c' == a
		u0 = 0.0;
	} else {
		V3 n_cpb = cp.cross(b);  if (n_cpb.length_squared() < 1e-30) { *out_u0 = 0.5; *out_u1 = 0.5; return; }
		V3 n_acp = a.cross(cp);  if (n_acp.length_squared() < 1e-30) { *out_u0 = 0.5; *out_u1 = 0.5; return; }
		n_cpb = n_cpb.normalized();
		n_acp = n_acp.normalized();
		double Ap = alpha + angle_between(n_ab, n_cpb) + angle_between(n_acp, n_cpb * -1.0) - Pi;
		u0 = Ap / A;
	}

	double u1 = (1.0 - w.dot(b)) / (1.0 - cp.dot(b));

	*out_u0 = u0 < 0.0 ? 0.0 : (u0 > 1.0 ? 1.0 : u0);
	*out_u1 = u1 < 0.0 ? 0.0 : (u1 > 1.0 ? 1.0 : u1);
}

// =============================================================================
// VarianceEstimator<Float>  --  online Welford mean/variance estimator
// pbrt-v4: VarianceEstimator (util/sampling.h)
// =============================================================================
template<typename Float>
class VarianceEstimator {
public:
	void Add(Float x) {
		++n;
		Float delta  = x - mean;
		mean += delta / static_cast<Float>(n);
		Float delta2 = x - mean;
		S += delta * delta2;
	}

	Float   Mean()             const { return mean; }
	Float   Variance()         const { return (n > 1) ? S / static_cast<Float>(n - 1) : Float(0); }
	int64_t Count()            const { return n; }
	Float   RelativeVariance() const {
		return (n < 1 || mean == Float(0)) ? Float(0) : Variance() / Mean();
	}

	void Merge(const VarianceEstimator& ve) {
		if (ve.n == 0) return;
		Float combined = static_cast<Float>(n + ve.n);
		S    = S + ve.S + (ve.mean - mean) * (ve.mean - mean)
			   * static_cast<Float>(n) * static_cast<Float>(ve.n) / combined;
		mean = (static_cast<Float>(n) * mean + static_cast<Float>(ve.n) * ve.mean) / combined;
		n   += ve.n;
	}

private:
	Float   mean = Float(0), S = Float(0);
	int64_t n    = 0;
};
