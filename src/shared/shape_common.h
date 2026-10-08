#pragma once
// shape_common.h -- part of shapes.h (which includes it): the helper math, ShapeHit, SamplingContext and ShapeSample that every shape uses.

#include "cpu_gpu.h"
#include "scalar_math.h"

#include "sampling_sphere_cone.h"   // SampleUniformSphere, SampleUniformDiskConcentric,
								// SampleUniformCone, UniformConePDF, etc.
#include "shading_frame.h"          // ShadingFrame<T>
#include "splines.h"                // CubicBezierControlPoints, EvaluateCubicBezierD, SubdivideCubicBezier
#include "interval_vec.h"           // Point3fi, OffsetRayOrigin, SpawnRay

#include <cmath>
#include <optional>
#include <algorithm>
#include <limits>

// ===========================================================================
// Helper math (mirrors pbrt-v4 util/math.h subset)
// ===========================================================================

namespace shapes_detail {

template<typename T> CPU_GPU T sq(T x) { return x * x; }

// Stable quadratic solver; returns false when no real roots exist.
// Roots are ordered: t0 <= t1.
template<typename T>
CPU_GPU bool solve_quadratic(T a, T b, T c, T& t0, T& t1) {
	// Use double precision for discriminant to avoid catastrophic cancellation
	double da = (double)a, db = (double)b, dc = (double)c;
	double disc = db*db - 4.0*da*dc;
	if (disc < 0.0) return false;
	double sqrt_disc = std::sqrt(disc);
	double q = (db < 0) ? -0.5*(db - sqrt_disc) : -0.5*(db + sqrt_disc);
	t0 = (T)(q / da);
	t1 = (T)(dc / q);
	if (t0 > t1) { T tmp = t0; t0 = t1; t1 = tmp; }
	return true;
}

// Safe arccos clamped to [-1,1]
// T is instantiated with both float (GPU shapes) and double (CPU shapes) -
// unsuffixed fmax/fmin overload-resolve correctly for either, unlike
// fmaxf/fminf which would silently truncate a double instantiation.
template<typename T> CPU_GPU T safe_acos(T x) {
#if defined(__CUDACC__)
	return std::acos(fmax(T(-1), fmin(T(1), x)));
#else
	return std::acos(std::max(T(-1), std::min(T(1), x)));
#endif
}

// Safe sqrt (clamps negative argument to zero)
template<typename T> CPU_GPU T safe_sqrt(T x) {
	return SafeSqrt(x);
}

// Length of 3-vector
template<typename T> CPU_GPU T len3(T x, T y, T z) {
	return std::sqrt(x*x + y*y + z*z);
}

// Dot product
template<typename T> CPU_GPU T dot3(T ax,T ay,T az, T bx,T by,T bz) {
	return ax*bx + ay*by + az*bz;
}

// Cross product (cx,cy,cz) = a x b
template<typename T>
CPU_GPU void cross3(T ax,T ay,T az, T bx,T by,T bz,
					T& cx, T& cy, T& cz) {
	cx = ay*bz - az*by;
	cy = az*bx - ax*bz;
	cz = ax*by - ay*bx;
}

// Normalize in-place
template<typename T>
CPU_GPU void normalize3(T& x, T& y, T& z) {
	T len = len3(x, y, z);
	if (len > T(0)) { x /= len; y /= len; z /= len; }
}

// gamma(n) -- conservative floating-point rounding error bound
// gamma(n) = (n * eps) / (1 - n * eps), eps = 0.5 * machine_eps
template<typename T>
CPU_GPU T gamma_fp(int n) {
	const T half_eps = std::numeric_limits<T>::epsilon() * T(0.5);
	return (T(n) * half_eps) / (T(1) - T(n) * half_eps);
}

// DifferenceOfProducts -- Kahan-style (a*b - c*d) with better precision
template<typename T>
CPU_GPU T diff_of_products(T a, T b, T c, T d) {
	T cd = c * d;
	T err = std::fma(-c, d, cd);
	T result = std::fma(a, b, -cd);
	return result + err;
}

// Max absolute component index of (x,y,z)
template<typename T>
CPU_GPU int max_abs_index(T x, T y, T z) {
	T ax = std::abs(x), ay = std::abs(y), az = std::abs(z);
	if (ax >= ay && ax >= az) return 0;
	if (ay >= az) return 1;
	return 2;
}

// Permute (x,y,z) by axes (i0,i1,i2)
template<typename T>
CPU_GPU void permute3(T x, T y, T z, int i0, int i1, int i2,
					  T& ox, T& oy, T& oz) {
	T v[3] = {x,y,z};
	ox = v[i0]; oy = v[i1]; oz = v[i2];
}

} // namespace shapes_detail


// ===========================================================================
// Data types
// ===========================================================================

// Returned by intersect()
template<typename T>
struct ShapeHit {
	T t;            // ray parameter of intersection
	T nx, ny, nz;   // outward surface normal (unit, in world space)
	T u, v;         // surface (u,v) parameterisation
	// Per-component absolute position-error bounds (pbrt-v4 pError).
	// Filled by each intersector; zero means "use flat 0.001 bias instead".
	T ex{}, ey{}, ez{};

	// Reconstruct a Point3fi for the hit position given ray (ox,oy,oz, dx,dy,dz)
	// so callers can invoke SpawnRay / OffsetRayOrigin directly.
	// p_hit = (ox + t*dx, oy + t*dy, oz + t*dz), error = (ex, ey, ez).
	CPU_GPU Point3fi ToPoint3fi(T ox, T oy, T oz,
								 T dx, T dy, T dz) const {
		return Point3fi(
			double(ox) + double(t)*double(dx),
			double(oy) + double(t)*double(dy),
			double(oz) + double(t)*double(dz),
			double(ex), double(ey), double(ez));
	}
};

// Point used as origin for solid-angle sampling
template<typename T>
struct SamplingContext {
	T px, py, pz;   // shading point (world space)
	T nx, ny, nz;   // shading normal (used only to offset ray origin; may be 0)
};

// Returned by sample() and sample_from()
template<typename T>
struct ShapeSample {
	T px, py, pz;   // sampled point on surface (world space)
	T nx, ny, nz;   // surface normal at sampled point
	T u, v;         // surface parameterisation at sampled point
	T pdf;          // probability density (per unit area or per unit solid angle)
};

// Converts an area-measure pdf to the solid-angle measure every shape's own
// sample_from()/pdf_from() below needs to return (pbrt-v4's "convert to
// solid angle" step, pdf *= dist^2/cos_theta) - factored out because
// Disk/Cylinder/Cone/Paraboloid each independently reimplemented this exact
// formula, and a code-review pass found the copies had already begun
// drifting: some guarded the final result with std::isfinite (a genuinely
// degenerate area/distance ratio can produce inf or NaN) and others didn't.
// Two entry points, matching the two call shapes this formula appears in:
//  - solid_angle_pdf_from_sample(): sample_from()'s own case - a freshly-
//    drawn ShapeSample position/normal and a raw (unnormalized) vector to
//    the reference point ctx, not yet known to be nonzero-length.
//  - solid_angle_pdf_from_hit(): pdf_from()'s own case - intersect() has
//    already returned a hit with its own normal, and the caller already
//    has the query direction as a UNIT vector plus the hit's own distance
//    (hit->t) - reusing those avoids a second, redundant normalize/sqrt
//    over the same points solid_angle_pdf_from_sample() would otherwise
//    repeat.
template<typename T>
CPU_GPU T solid_angle_pdf_from_sample(T sample_px, T sample_py, T sample_pz,
                                        T sample_nx, T sample_ny, T sample_nz,
                                        const SamplingContext<T>& ctx, T pdf_area) {
	using namespace shapes_detail;
	T wix = sample_px - ctx.px, wiy = sample_py - ctx.py, wiz = sample_pz - ctx.pz;
	T dist2 = wix*wix + wiy*wiy + wiz*wiz;
	if (dist2 == T(0)) return T(0);
	T inv_d = T(1) / std::sqrt(dist2);
	T cos_theta = std::abs(dot3(sample_nx, sample_ny, sample_nz,
	                             -wix*inv_d, -wiy*inv_d, -wiz*inv_d));
	if (cos_theta == T(0)) return T(0);
	T pdf = pdf_area * dist2 / cos_theta;
	return std::isfinite(pdf) ? pdf : T(0);
}

template<typename T>
CPU_GPU T solid_angle_pdf_from_hit(T hit_nx, T hit_ny, T hit_nz,
                                     T wi_ux, T wi_uy, T wi_uz, T hit_t, T pdf_area) {
	using namespace shapes_detail;
	T cos_theta = std::abs(dot3(hit_nx, hit_ny, hit_nz, -wi_ux, -wi_uy, -wi_uz));
	if (cos_theta == T(0)) return T(0);
	T pdf = pdf_area * (hit_t * hit_t) / cos_theta;
	return std::isfinite(pdf) ? pdf : T(0);
}

