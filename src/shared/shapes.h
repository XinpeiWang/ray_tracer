#pragma once
// ---------------------------------------------------------------------------
// shapes.h -- Shared CPU/GPU shape intersection and sampling primitives.
//
// Ports pbrt-v4 src/pbrt/shapes.h/.cpp for three fundamental shapes:
//   SphereShape<T>   -- full sphere (or z-clipped) with quadric intersection
//   DiskShape<T>     -- flat disk aligned to XY plane (z = height)
//   TriangleShape<T> -- Watertight triangle (Woop/Igehy style, pbrt-v4)
//
// Each shape exposes:
//   intersect(ray_ox,oy,oz, rd_dx,dy,dz, tMin, tMax)
//       -> optional<ShapeHit<T>>
//   area()                      -> T
//   sample(u0, u1)              -> ShapeSample<T>   (area-uniform)
//   pdf_area()                  -> T   (= 1/area())
//   sample_from(ctx, u0, u1)    -> ShapeSample<T>   (solid-angle from point)
//   pdf_from(ctx, wi_dx,wy,wz)  -> T   (solid-angle PDF)
//
// SamplingContext<T> holds the shading point used for solid-angle sampling.
//
// Design rules (same as bxdfs.h / sampling.h):
//   - No virtual functions, no heap allocation
//   - Template parameter T: double on CPU, float on GPU
//   - CPU_GPU macro: __host__ __device__ under NVCC, inline otherwise
//   - Coordinate convention: right-handed, y-up (matches local codebase)
//
// Reference: pbrt-v4 src/pbrt/shapes.h, shapes.cpp
//            Wald et al. 2014 "Watertight Ray/Triangle Intersection"
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU inline
#   endif
#endif

#include "sampling_sphere_cone.h"   // SampleUniformSphere, SampleUniformDiskConcentric,
								// SampleUniformCone, UniformConePDF, etc.
#include "shading_frame.h"          // ShadingFrame<T>
#include "splines.h"                // CubicBezierControlPoints, EvaluateCubicBezierD, SubdivideCubicBezier

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
template<typename T> CPU_GPU T safe_acos(T x) {
	return std::acos(std::max(T(-1), std::min(T(1), x)));
}

// Safe sqrt (clamps negative argument to zero)
template<typename T> CPU_GPU T safe_sqrt(T x) {
	return std::sqrt(std::max(T(0), x));
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


// ===========================================================================
// SphereShape<T>
// ===========================================================================
//
// Full sphere of radius r centered at (cx, cy, cz).
// Optional z-clipping: [z_min, z_max] and azimuthal clipping [0, phi_max].
// When phi_max = 2*pi and z_min = -r, z_max = +r -> complete sphere.
//
// Reference: pbrt-v4 Sphere (shapes.h / shapes.cpp)
// ===========================================================================

template<typename T>
struct SphereShape {
	T cx, cy, cz;   // center
	T r;            // radius
	T z_min, z_max; // z clipping in object space (default -r, +r)
	T phi_max;      // azimuthal extent in radians (default 2*pi)

	// Convenience constructor: full sphere
	CPU_GPU static SphereShape make(T cx, T cy, T cz, T radius) {
		const T pi2 = T(2) * T(3.14159265358979323846);
		return SphereShape{cx, cy, cz, radius, -radius, radius, pi2};
	}

	// Constructor with clipping
	CPU_GPU static SphereShape make_clipped(T cx, T cy, T cz, T radius,
											T zmin, T zmax, T phi_max_rad) {
		return SphereShape{cx, cy, cz, radius, zmin, zmax, phi_max_rad};
	}

	// -----------------------------------------------------------------------
	// Intersection -- quadric solve in object space, then clip
	// Reference: pbrt-v4 Sphere::Intersect
	// -----------------------------------------------------------------------
	CPU_GPU std::optional<ShapeHit<T>>
	intersect(T rox, T roy, T roz,
			  T rdx, T rdy, T rdz,
			  T t_min, T t_max) const
	{
		using namespace shapes_detail;

		// Transform ray to object space (sphere centered at origin)
		T ox = rox - cx, oy = roy - cy, oz = roz - cz;
		T dx = rdx, dy = rdy, dz = rdz;

		// Quadratic coefficients: |d|^2 t^2 + 2(o.d)t + (|o|^2 - r^2) = 0
		// Use double precision internally (pbrt-v4 approach)
		double a = (double)dx*dx + (double)dy*dy + (double)dz*dz;
		double b = 2.0 * ((double)ox*dx + (double)oy*dy + (double)oz*dz);
		double c = (double)ox*ox + (double)oy*oy + (double)oz*oz - (double)r*r;

		double disc = b*b - 4.0*a*c;
		if (disc < 0.0) return {};

		double sqrt_disc = std::sqrt(disc);
		double q = (b < 0) ? -0.5*(b - sqrt_disc) : -0.5*(b + sqrt_disc);
		T t0 = (T)(q / a);
		T t1 = (T)(c / q);
		if (t0 > t1) { T tmp = t0; t0 = t1; t1 = tmp; }

		if (t0 > t_max || t1 < t_min) return {};

		T t_hit = (t0 >= t_min) ? t0 : t1;
		if (t_hit > t_max) return {};

		// Hit point in object space
		T hx = ox + t_hit * dx;
		T hy = oy + t_hit * dy;
		T hz = oz + t_hit * dz;

		// Refine to exactly lie on sphere surface
		T len = safe_sqrt(hx*hx + hy*hy + hz*hz);
		if (len > T(0)) { hx *= r/len; hy *= r/len; hz *= r/len; }

		// Z clipping
		T th_z_min = std::max(T(-1), std::min(T(1), z_min / r));
		T th_z_max = std::max(T(-1), std::min(T(1), z_max / r));
		if (hz < z_min || hz > z_max) {
			// Try other root
			t_hit = (t_hit == t0) ? t1 : t0;
			if (t_hit < t_min || t_hit > t_max) return {};
			hx = ox + t_hit * dx;
			hy = oy + t_hit * dy;
			hz = oz + t_hit * dz;
			T len2 = safe_sqrt(hx*hx + hy*hy + hz*hz);
			if (len2 > T(0)) { hx *= r/len2; hy *= r/len2; hz *= r/len2; }
			if (hz < z_min || hz > z_max) return {};
		}

		// Phi clipping
		T phi = std::atan2(hy, hx);
		if (phi < T(0)) phi += T(2) * T(3.14159265358979323846);
		if (phi > phi_max) {
			// Try other root
			T t_other = (t_hit == t0) ? t1 : t0;
			if (t_other < t_min || t_other > t_max) return {};
			hx = ox + t_other * dx;
			hy = oy + t_other * dy;
			hz = oz + t_other * dz;
			T len2 = safe_sqrt(hx*hx + hy*hy + hz*hz);
			if (len2 > T(0)) { hx *= r/len2; hy *= r/len2; hz *= r/len2; }
			phi = std::atan2(hy, hx);
			if (phi < T(0)) phi += T(2) * T(3.14159265358979323846);
			if (hz < z_min || hz > z_max || phi > phi_max) return {};
			t_hit = t_other;
		}

		// Compute outward normal (object space -> world space: just shift back)
		T nnx = hx / r, nny = hy / r, nnz = hz / r;
		// Transform to world space (sphere has no rotation, just translation)
		// Normal is already correct in world space since sphere is axis-aligned

		// Compute (u,v)
		const T pi = T(3.14159265358979323846);
		T cos_theta = std::max(T(-1), std::min(T(1), hz / r));
		T theta = std::acos(cos_theta);
		T theta_z_min = std::acos(th_z_max);  // note: cos is monotone decreasing
		T theta_z_max = std::acos(th_z_min);
		T u_coord = phi / phi_max;
		T v_coord = (theta_z_max > theta_z_min)
					? (theta - theta_z_min) / (theta_z_max - theta_z_min)
					: T(0);

		ShapeHit<T> hit;
		hit.t  = t_hit;
		hit.nx = nnx; hit.ny = nny; hit.nz = nnz;
		hit.u  = u_coord; hit.v = v_coord;
		return hit;
	}

	// -----------------------------------------------------------------------
	// Area (full sphere = 4*pi*r^2; clipped = phi_max * r * (z_max - z_min))
	// Reference: pbrt-v4 Sphere::Area
	// -----------------------------------------------------------------------
	CPU_GPU T area() const {
		return phi_max * r * (z_max - z_min);
	}

	CPU_GPU T pdf_area() const { return T(1) / area(); }

	// -----------------------------------------------------------------------
	// Area-uniform surface sample
	// Reference: pbrt-v4 Sphere::Sample(Point2f u)
	// -----------------------------------------------------------------------
	CPU_GPU ShapeSample<T> sample(T u0, T u1) const {
		// Sample uniform direction on full sphere
		T wx, wy, wz;
		SampleUniformSphere(u0, u1, wx, wy, wz);

		// Object-space hit point on sphere surface
		T px_obj = r * wx;
		T py_obj = r * wy;
		T pz_obj = r * wz;

		// Reproject to exact sphere surface
		T len = shapes_detail::len3(px_obj, py_obj, pz_obj);
		if (len > T(0)) { px_obj *= r/len; py_obj *= r/len; pz_obj *= r/len; }

		// World-space position
		T px_w = cx + px_obj, py_w = cy + py_obj, pz_w = cz + pz_obj;

		// Outward normal
		T nnx = px_obj / r, nny = py_obj / r, nnz = pz_obj / r;

		// (u,v)
		const T pi = T(3.14159265358979323846);
		T cos_theta = std::max(T(-1), std::min(T(1), pz_obj / r));
		T theta = std::acos(cos_theta);
		T phi   = std::atan2(py_obj, px_obj);
		if (phi < T(0)) phi += T(2) * pi;
		T th_z_min = std::max(T(-1), std::min(T(1), z_min / r));
		T th_z_max = std::max(T(-1), std::min(T(1), z_max / r));
		T theta_z_min = std::acos(th_z_max);
		T theta_z_max = std::acos(th_z_min);
		T u_coord = phi / phi_max;
		T v_coord = (theta_z_max > theta_z_min)
					? (theta - theta_z_min) / (theta_z_max - theta_z_min)
					: T(0);

		return ShapeSample<T>{px_w, py_w, pz_w,
							  nnx, nny, nnz,
							  u_coord, v_coord,
							  pdf_area()};
	}

	// -----------------------------------------------------------------------
	// Solid-angle sample from a context point
	// Reference: pbrt-v4 Sphere::Sample(const ShapeSampleContext&, Point2f)
	// -----------------------------------------------------------------------
	CPU_GPU ShapeSample<T> sample_from(const SamplingContext<T>& ctx,
									   T u0, T u1) const {
		using namespace shapes_detail;
		const T pi = T(3.14159265358979323846);

		T dist2 = sq(ctx.px - cx) + sq(ctx.py - cy) + sq(ctx.pz - cz);

		// If inside sphere, fall back to area sampling and convert PDF
		if (dist2 <= sq(r)) {
			ShapeSample<T> ss = sample(u0, u1);
			T wix = ss.px - ctx.px;
			T wiy = ss.py - ctx.py;
			T wiz = ss.pz - ctx.pz;
			T wi_len2 = wix*wix + wiy*wiy + wiz*wiz;
			if (wi_len2 == T(0)) { ss.pdf = T(0); return ss; }
			T wi_len = std::sqrt(wi_len2);
			T wi_nx = wix/wi_len, wi_ny = wiy/wi_len, wi_nz = wiz/wi_len;
			T cos_theta_n = std::abs(dot3(ss.nx, ss.ny, ss.nz, -wi_nx, -wi_ny, -wi_nz));
			if (cos_theta_n == T(0)) { ss.pdf = T(0); return ss; }
			ss.pdf = ss.pdf * wi_len2 / cos_theta_n;
			return ss;
		}

		// Cone sampling: sample uniformly within the subtended cone
		T sin2_theta_max = sq(r) / dist2;
		T cos_theta_max  = safe_sqrt(T(1) - sin2_theta_max);
		T one_minus_cos  = T(1) - cos_theta_max;

		// Sample cosTheta
		T cos_theta = (cos_theta_max - T(1)) * u0 + T(1);
		T sin2_theta = T(1) - sq(cos_theta);

		// Small-angle Taylor expansion
		if (sin2_theta_max < T(0.00068523)) {
			sin2_theta = sin2_theta_max * u0;
			cos_theta  = std::sqrt(T(1) - sin2_theta);
			one_minus_cos = sin2_theta_max * T(0.5);
		}

		// Alpha angle (sphere surface to hit point)
		T sin_theta_max = safe_sqrt(sin2_theta_max);
		T cos_alpha = sin2_theta / sin_theta_max +
					  cos_theta * safe_sqrt(T(1) - sin2_theta / sin2_theta_max);
		T sin_alpha = safe_sqrt(T(1) - sq(cos_alpha));

		// Build local frame around (center - ctx.p) direction
		T frame_z_x = cx - ctx.px;
		T frame_z_y = cy - ctx.py;
		T frame_z_z = cz - ctx.pz;
		normalize3(frame_z_x, frame_z_y, frame_z_z);

		ShadingFrame<T> frame = ShadingFrame<T>::from_normal(frame_z_x, frame_z_y, frame_z_z);

		// Sampled direction in local frame: (sin_alpha*cos_phi, sin_alpha*sin_phi, cos_alpha)
		T phi = u1 * T(2) * pi;
		T wx_local = sin_alpha * std::cos(phi);
		T wy_local = sin_alpha * std::sin(phi);
		T wz_local = cos_alpha;

		// Convert to world space (note: w is pointing from ctx toward sphere)
		// The actual surface normal n = -FromLocal(w) in pbrt-v4's convention
		T nx_world, ny_world, nz_world;
		frame.to_world(wx_local, wy_local, wz_local, nx_world, ny_world, nz_world);
		// Normal on sphere surface points outward (away from center)
		T nnx = -nx_world, nny = -ny_world, nnz = -nz_world;

		T px_w = cx + r * nnx;
		T py_w = cy + r * nny;
		T pz_w = cz + r * nnz;

		// (u,v) at sampled point
		T px_obj = px_w - cx, py_obj = py_w - cy, pz_obj = pz_w - cz;
		T cos_theta_uv = std::max(T(-1), std::min(T(1), pz_obj / r));
		T theta_uv = std::acos(cos_theta_uv);
		T phi_uv   = std::atan2(py_obj, px_obj);
		if (phi_uv < T(0)) phi_uv += T(2) * pi;
		T th_z_min = std::max(T(-1), std::min(T(1), z_min / r));
		T th_z_max = std::max(T(-1), std::min(T(1), z_max / r));
		T theta_z_min = std::acos(th_z_max);
		T theta_z_max = std::acos(th_z_min);
		T u_coord = phi_uv / phi_max;
		T v_coord = (theta_z_max > theta_z_min)
					? (theta_uv - theta_z_min) / (theta_z_max - theta_z_min)
					: T(0);

		T pdf_val = (one_minus_cos > T(0))
					? T(1) / (T(2) * pi * one_minus_cos)
					: T(0);

		return ShapeSample<T>{px_w, py_w, pz_w,
							  nnx, nny, nnz,
							  u_coord, v_coord,
							  pdf_val};
	}

	// -----------------------------------------------------------------------
	// Solid-angle PDF
	// Reference: pbrt-v4 Sphere::PDF(const ShapeSampleContext&, Vector3f wi)
	// -----------------------------------------------------------------------
	CPU_GPU T pdf_from(const SamplingContext<T>& ctx,
					   T wi_dx, T wi_dy, T wi_dz) const {
		using namespace shapes_detail;
		const T pi = T(3.14159265358979323846);

		T dist2 = sq(ctx.px - cx) + sq(ctx.py - cy) + sq(ctx.pz - cz);

		if (dist2 <= sq(r)) {
			// Inside sphere: use area PDF converted to solid angle
			// Trace ray and compute PDF
			// Approximate: 1/Area * dist^2 / |cos_theta|
			// We just return 0 here as a conservative fallback (caller should
			// use sample_from's pdf field which is correctly computed)
			return T(0);
		}

		T sin2_theta_max = sq(r) / dist2;
		T cos_theta_max  = safe_sqrt(T(1) - sin2_theta_max);
		T one_minus_cos  = T(1) - cos_theta_max;
		if (sin2_theta_max < T(0.00068523))
			one_minus_cos = sin2_theta_max * T(0.5);

		// Verify wi falls inside the cone
		T wi_len = len3(wi_dx, wi_dy, wi_dz);
		if (wi_len == T(0)) return T(0);
		T wix = wi_dx/wi_len, wiy = wi_dy/wi_len, wiz = wi_dz/wi_len;

		T frame_z_x = cx - ctx.px;
		T frame_z_y = cy - ctx.py;
		T frame_z_z = cz - ctx.pz;
		normalize3(frame_z_x, frame_z_y, frame_z_z);

		T cos_wi = dot3(wix,wiy,wiz, frame_z_x,frame_z_y,frame_z_z);
		if (cos_wi < cos_theta_max) return T(0);

		return (one_minus_cos > T(0))
			   ? T(1) / (T(2) * pi * one_minus_cos)
			   : T(0);
	}
};


// ===========================================================================
// DiskShape<T>
// ===========================================================================
//
// Flat circular disk in the plane z = height, centered at (cx, cy, height).
// Inner radius inner_r allows annular disks (default 0).
//
// Reference: pbrt-v4 Disk (shapes.h / shapes.cpp)
// ===========================================================================

template<typename T>
struct DiskShape {
	T cx, cy, height;   // center x/y and z position of disk plane
	T outer_r;          // outer radius
	T inner_r;          // inner (hole) radius, 0 for solid disk
	T phi_max;          // azimuthal extent in radians (default 2*pi)

	CPU_GPU static DiskShape make(T cx, T cy, T height, T radius) {
		const T pi2 = T(2) * T(3.14159265358979323846);
		return DiskShape{cx, cy, height, radius, T(0), pi2};
	}

	CPU_GPU static DiskShape make_annular(T cx, T cy, T height,
										  T outer_r, T inner_r,
										  T phi_max_rad) {
		return DiskShape{cx, cy, height, outer_r, inner_r, phi_max_rad};
	}

	// -----------------------------------------------------------------------
	// Intersection
	// Reference: pbrt-v4 Disk::Intersect
	// -----------------------------------------------------------------------
	CPU_GPU std::optional<ShapeHit<T>>
	intersect(T rox, T roy, T roz,
			  T rdx, T rdy, T rdz,
			  T t_min, T t_max) const
	{
		// Ray parallel to disk plane?
		if (rdz == T(0)) return {};

		// t at intersection with z = height plane
		T t_hit = (height - roz) / rdz;
		if (t_hit < t_min || t_hit > t_max) return {};

		// Hit point
		T hx = rox + t_hit * rdx - cx;
		T hy = roy + t_hit * rdy - cy;
		T dist2 = hx*hx + hy*hy;
		if (dist2 > outer_r*outer_r || dist2 < inner_r*inner_r) return {};

		// Phi clipping
		T phi = std::atan2(hy, hx);
		if (phi < T(0)) phi += T(2) * T(3.14159265358979323846);
		if (phi > phi_max) return {};

		// (u,v): u = phi/phi_max, v = 1 - r/outer_r (annular: (r - inner_r)/(outer_r-inner_r))
		T dist = std::sqrt(dist2);
		T u_coord = phi / phi_max;
		T v_coord = (outer_r > inner_r)
					? T(1) - (dist - inner_r) / (outer_r - inner_r)
					: T(0);

		// Normal always points up (+z) for un-flipped disk
		ShapeHit<T> hit;
		hit.t  = t_hit;
		hit.nx = T(0); hit.ny = T(0); hit.nz = T(1);
		hit.u  = u_coord; hit.v = v_coord;
		return hit;
	}

	// -----------------------------------------------------------------------
	// Area
	// Reference: pbrt-v4 Disk::Area
	// -----------------------------------------------------------------------
	CPU_GPU T area() const {
		const T pi = T(3.14159265358979323846);
		return phi_max * T(0.5) * (outer_r*outer_r - inner_r*inner_r);
	}

	CPU_GPU T pdf_area() const { return T(1) / area(); }

	// -----------------------------------------------------------------------
	// Area-uniform surface sample
	// Reference: pbrt-v4 Disk::Sample(Point2f u)
	// pbrt-v4 scales SampleUniformDiskConcentric directly by outer_r.
	// -----------------------------------------------------------------------
	CPU_GPU ShapeSample<T> sample(T u0, T u1) const {
		T dx, dy;
		SampleUniformDiskConcentric(u0, u1, dx, dy);

		// Scale by outer radius (pbrt-v4: pd.x * radius, pd.y * radius)
		T px_obj = dx * outer_r;
		T py_obj = dy * outer_r;

		T px_w = cx + px_obj;
		T py_w = cy + py_obj;
		T pz_w = height;

		const T pi = T(3.14159265358979323846);
		T phi_samp = std::atan2(py_obj, px_obj);
		if (phi_samp < T(0)) phi_samp += T(2) * pi;
		T u_coord = phi_samp / phi_max;
		T radius_samp = std::sqrt(px_obj*px_obj + py_obj*py_obj);
		// pbrt-v4: v = (radius - radiusSample) / (radius - innerRadius)
		T v_coord = (outer_r > inner_r)
					? (outer_r - radius_samp) / (outer_r - inner_r)
					: T(0);

		return ShapeSample<T>{px_w, py_w, pz_w,
							  T(0), T(0), T(1),
							  u_coord, v_coord,
							  pdf_area()};
	}

	// -----------------------------------------------------------------------
	// Solid-angle sample from a context point
	// Falls back to area sampling + Jacobian conversion
	// -----------------------------------------------------------------------
	CPU_GPU ShapeSample<T> sample_from(const SamplingContext<T>& ctx,
									   T u0, T u1) const {
		ShapeSample<T> ss = sample(u0, u1);
		T wix = ss.px - ctx.px;
		T wiy = ss.py - ctx.py;
		T wiz = ss.pz - ctx.pz;
		T wi_len2 = wix*wix + wiy*wiy + wiz*wiz;
		if (wi_len2 == T(0)) { ss.pdf = T(0); return ss; }
		T wi_len = std::sqrt(wi_len2);
		T wi_nx = wix/wi_len, wi_ny = wiy/wi_len, wi_nz = wiz/wi_len;
		T cos_theta_n = std::abs(shapes_detail::dot3(ss.nx, ss.ny, ss.nz,
													 -wi_nx, -wi_ny, -wi_nz));
		if (cos_theta_n == T(0)) { ss.pdf = T(0); return ss; }
		ss.pdf = ss.pdf * wi_len2 / cos_theta_n;
		return ss;
	}

	CPU_GPU T pdf_from(const SamplingContext<T>& ctx,
					   T wi_dx, T wi_dy, T wi_dz) const {
		// Intersect the wi ray with the disk and compute Jacobian
		T wi_len = shapes_detail::len3(wi_dx, wi_dy, wi_dz);
		if (wi_len == T(0)) return T(0);
		// Ray from ctx in direction wi
		auto hit = intersect(ctx.px, ctx.py, ctx.pz,
							 wi_dx/wi_len, wi_dy/wi_len, wi_dz/wi_len,
							 T(1e-4), std::numeric_limits<T>::max());
		if (!hit) return T(0);
		T dist = hit->t;
		T cos_theta_n = std::abs(shapes_detail::dot3(hit->nx, hit->ny, hit->nz,
													 -wi_dx/wi_len, -wi_dy/wi_len, -wi_dz/wi_len));
		if (cos_theta_n == T(0)) return T(0);
		return pdf_area() * dist*dist / cos_theta_n;
	}
};


// ===========================================================================
// CylinderShape<T>
// ===========================================================================
//
// Cylinder of radius r along the Z axis in object space, translated to world
// space by center (cx, cy, cz). Extents are [z_min, z_max] in object space;
// azimuthal extent is [0, phi_max] (default 2*pi = full cylinder).
//
// Normal convention: outward radial (ignores end caps, matching pbrt-v4).
//
// Reference: pbrt-v4 src/pbrt/shapes.h  Cylinder class
// ===========================================================================

template<typename T>
struct CylinderShape {
	T cx, cy, cz;   // center (world origin of cylinder axis)
	T r;            // radius
	T z_min, z_max; // z extent in object space
	T phi_max;      // azimuthal extent in radians (default 2*pi)

	// Convenience constructors
	CPU_GPU static CylinderShape make(T cx, T cy, T cz, T radius,
									   T zmin, T zmax) {
		const T pi2 = T(2) * T(3.14159265358979323846);
		return CylinderShape{cx, cy, cz, radius, zmin, zmax, pi2};
	}

	CPU_GPU static CylinderShape make_partial(T cx, T cy, T cz, T radius,
											   T zmin, T zmax, T phi_max_rad) {
		return CylinderShape{cx, cy, cz, radius, zmin, zmax, phi_max_rad};
	}

	// -----------------------------------------------------------------------
	// Area
	// Reference: pbrt-v4 Cylinder::Area = (zMax - zMin) * radius * phiMax
	// -----------------------------------------------------------------------
	CPU_GPU T area() const {
		return (z_max - z_min) * r * phi_max;
	}

	CPU_GPU T pdf_area() const { return T(1) / area(); }

	// -----------------------------------------------------------------------
	// Intersection -- quadric solve in object space, z/phi clip
	// Reference: pbrt-v4 Cylinder::BasicIntersect
	// -----------------------------------------------------------------------
	CPU_GPU std::optional<ShapeHit<T>>
	intersect(T rox, T roy, T roz,
			  T rdx, T rdy, T rdz,
			  T t_min, T t_max) const
	{
		using namespace shapes_detail;
		const T pi = T(3.14159265358979323846);

		// Transform ray to object space (cylinder axis at origin)
		T ox = rox - cx, oy = roy - cy, oz = roz - cz;
		T dx = rdx, dy = rdy, dz = rdz;

		// Quadratic coefficients (only x,y matter – ignore z component)
		// a*t^2 + b*t + c = 0  where a = dx^2+dy^2, c = ox^2+oy^2-r^2
		double a = (double)dx*dx + (double)dy*dy;
		double b = 2.0 * ((double)ox*dx + (double)oy*dy);
		double c = (double)ox*ox + (double)oy*oy - (double)r*(double)r;

		// pbrt-v4 discriminant formula (numerically stable)
		double f = b / (2.0 * a);
		double vx = (double)ox - f * (double)dx;
		double vy = (double)oy - f * (double)dy;
		double length_v = std::sqrt(vx*vx + vy*vy);
		double discrim = 4.0 * a * ((double)r + length_v) * ((double)r - length_v);
		if (discrim < 0.0) return {};

		double sqrt_disc = std::sqrt(discrim);
		double q = (b < 0.0) ? -0.5*(b - sqrt_disc) : -0.5*(b + sqrt_disc);
		T t0 = (T)(q / a);
		T t1 = (T)(c / q);
		if (t0 > t1) { T tmp = t0; t0 = t1; t1 = tmp; }

		if (t0 > t_max || t1 < t_min) return {};

		// Helper lambda: compute hit point, refine, compute phi, check clips
		auto check = [&](T t) -> std::optional<ShapeHit<T>> {
			if (t < t_min || t > t_max) return {};
			T hx = ox + t * dx;
			T hy = oy + t * dy;
			T hz = oz + t * dz;
			// Refine to cylinder surface
			T hitRad = safe_sqrt(hx*hx + hy*hy);
			if (hitRad > T(0)) { hx *= r / hitRad; hy *= r / hitRad; }
			T phi = std::atan2(hy, hx);
			if (phi < T(0)) phi += T(2) * pi;
			if (hz < z_min || hz > z_max || phi > phi_max) return {};
			// Outward radial normal (object space == world space, no rotation)
			T nnx = hx / r, nny = hy / r, nnz = T(0);
			// UV: u = phi/phi_max, v = (z - z_min)/(z_max - z_min)
			T u = phi / phi_max;
			T v = (hz - z_min) / (z_max - z_min);
			return ShapeHit<T>{t, nnx, nny, nnz, u, v};
		};

		auto hit = check(t0);
		if (hit) return hit;
		return check(t1);
	}

	// -----------------------------------------------------------------------
	// Area-uniform surface sample
	// Reference: pbrt-v4 Cylinder::Sample(Point2f u)
	// -----------------------------------------------------------------------
	CPU_GPU ShapeSample<T> sample(T u0, T u1) const {
		const T pi = T(3.14159265358979323846);
		T z   = z_min + u0 * (z_max - z_min);
		T phi = u1 * phi_max;
		T px_obj = r * std::cos(phi);
		T py_obj = r * std::sin(phi);
		// Refine to exact radius
		T hitRad = safe_sqrt(px_obj*px_obj + py_obj*py_obj);
		if (hitRad > T(0)) { px_obj *= r / hitRad; py_obj *= r / hitRad; }
		T nx = px_obj / r, ny = py_obj / r, nz = T(0);
		T uv = phi / phi_max;
		T vv = (z - z_min) / (z_max - z_min);
		return ShapeSample<T>{cx + px_obj, cy + py_obj, cz + z,
							  nx, ny, nz,
							  uv, vv,
							  pdf_area()};
	}

	// -----------------------------------------------------------------------
	// Solid-angle sample from a context point (area sampling + Jacobian)
	// Reference: pbrt-v4 Cylinder::Sample(ShapeSampleContext, Point2f)
	// -----------------------------------------------------------------------
	CPU_GPU ShapeSample<T> sample_from(const SamplingContext<T>& ctx,
										T u0, T u1) const {
		ShapeSample<T> ss = sample(u0, u1);
		T wix = ss.px - ctx.px;
		T wiy = ss.py - ctx.py;
		T wiz = ss.pz - ctx.pz;
		T wi_len2 = wix*wix + wiy*wiy + wiz*wiz;
		if (wi_len2 == T(0)) { ss.pdf = T(0); return ss; }
		T wi_len = std::sqrt(wi_len2);
		T wi_nx = wix/wi_len, wi_ny = wiy/wi_len, wi_nz = wiz/wi_len;
		T cos_theta_n = std::abs(shapes_detail::dot3(ss.nx, ss.ny, ss.nz,
													  -wi_nx, -wi_ny, -wi_nz));
		if (cos_theta_n == T(0)) { ss.pdf = T(0); return ss; }
		ss.pdf = ss.pdf * wi_len2 / cos_theta_n;
		return ss;
	}

	// -----------------------------------------------------------------------
	// Solid-angle PDF from context (intersect + Jacobian)
	// Reference: pbrt-v4 Cylinder::PDF(ShapeSampleContext, Vector3f)
	// -----------------------------------------------------------------------
	CPU_GPU T pdf_from(const SamplingContext<T>& ctx,
						T wi_dx, T wi_dy, T wi_dz) const {
		T wi_len = shapes_detail::len3(wi_dx, wi_dy, wi_dz);
		if (wi_len == T(0)) return T(0);
		auto hit = intersect(ctx.px, ctx.py, ctx.pz,
							 wi_dx/wi_len, wi_dy/wi_len, wi_dz/wi_len,
							 T(1e-4), std::numeric_limits<T>::max());
		if (!hit) return T(0);
		T dist = hit->t;
		T cos_theta_n = std::abs(shapes_detail::dot3(hit->nx, hit->ny, hit->nz,
													  -wi_dx/wi_len, -wi_dy/wi_len, -wi_dz/wi_len));
		if (cos_theta_n == T(0)) return T(0);
		return pdf_area() * dist*dist / cos_theta_n;
	}
};


// ===========================================================================
// TriangleShape<T>
// ===========================================================================
//
// Triangle with vertices p0, p1, p2 (world space).
//
// Intersection: Watertight algorithm (Woop, Benthin, Wald 2013) exactly as
// in pbrt-v4 src/pbrt/shapes.cpp:IntersectTriangle.
//
// Sampling: either area-uniform (barycentric) or solid-angle (via spherical
// triangle sampling) depending on the solid angle as seen from ctx.
//
// Reference: pbrt-v4 Triangle (shapes.h / shapes.cpp)
// ===========================================================================

template<typename T>
struct TriangleShape {
	T p0x, p0y, p0z;
	T p1x, p1y, p1z;
	T p2x, p2y, p2z;

	// Optional per-vertex normals for shading (zeroed = use geometric normal)
	T n0x, n0y, n0z;
	T n1x, n1y, n1z;
	T n2x, n2y, n2z;
	bool has_shading_normals = false;

	CPU_GPU static TriangleShape make(T p0x, T p0y, T p0z,
									  T p1x, T p1y, T p1z,
									  T p2x, T p2y, T p2z) {
		TriangleShape t;
		t.p0x=p0x; t.p0y=p0y; t.p0z=p0z;
		t.p1x=p1x; t.p1y=p1y; t.p1z=p1z;
		t.p2x=p2x; t.p2y=p2y; t.p2z=p2z;
		t.n0x=t.n0y=t.n0z=T(0);
		t.n1x=t.n1y=t.n1z=T(0);
		t.n2x=t.n2y=t.n2z=T(0);
		t.has_shading_normals = false;
		return t;
	}

	// -----------------------------------------------------------------------
	// Geometric normal (outward, not normalised)
	// -----------------------------------------------------------------------
	CPU_GPU void geometric_normal(T& nx, T& ny, T& nz) const {
		T e1x = p1x-p0x, e1y = p1y-p0y, e1z = p1z-p0z;
		T e2x = p2x-p0x, e2y = p2y-p0y, e2z = p2z-p0z;
		shapes_detail::cross3(e1x,e1y,e1z, e2x,e2y,e2z, nx,ny,nz);
	}

	// -----------------------------------------------------------------------
	// Area = 0.5 * |e1 x e2|
	// -----------------------------------------------------------------------
	CPU_GPU T area() const {
		T nx, ny, nz;
		geometric_normal(nx,ny,nz);
		return T(0.5) * shapes_detail::len3(nx,ny,nz);
	}

	CPU_GPU T pdf_area() const { return T(1) / area(); }

	// -----------------------------------------------------------------------
	// Watertight ray-triangle intersection
	// Direct port of pbrt-v4 IntersectTriangle (shapes.cpp:168)
	// -----------------------------------------------------------------------
	CPU_GPU std::optional<ShapeHit<T>>
	intersect(T rox, T roy, T roz,
			  T rdx, T rdy, T rdz,
			  T t_min, T t_max) const
	{
		using namespace shapes_detail;

		// Degenerate check
		T gnx, gny, gnz;
		geometric_normal(gnx, gny, gnz);
		if (gnx*gnx + gny*gny + gnz*gnz == T(0)) return {};

		// Translate to ray origin
		T p0x_t = p0x - rox, p0y_t = p0y - roy, p0z_t = p0z - roz;
		T p1x_t = p1x - rox, p1y_t = p1y - roy, p1z_t = p1z - roz;
		T p2x_t = p2x - rox, p2y_t = p2y - roy, p2z_t = p2z - roz;

		// Choose dominant axis
		int kz = max_abs_index(rdx, rdy, rdz);
		int kx = (kz+1) % 3;
		int ky = (kx+1) % 3;

		T dxp, dyp, dzp;
		permute3(rdx, rdy, rdz, kx, ky, kz, dxp, dyp, dzp);

		T p0x_tp, p0y_tp, p0z_tp;
		T p1x_tp, p1y_tp, p1z_tp;
		T p2x_tp, p2y_tp, p2z_tp;
		permute3(p0x_t, p0y_t, p0z_t, kx, ky, kz, p0x_tp, p0y_tp, p0z_tp);
		permute3(p1x_t, p1y_t, p1z_t, kx, ky, kz, p1x_tp, p1y_tp, p1z_tp);
		permute3(p2x_t, p2y_t, p2z_t, kx, ky, kz, p2x_tp, p2y_tp, p2z_tp);

		// Shear transform
		T Sx = -dxp / dzp;
		T Sy = -dyp / dzp;
		T Sz =  T(1) / dzp;
		p0x_tp += Sx * p0z_tp;  p0y_tp += Sy * p0z_tp;
		p1x_tp += Sx * p1z_tp;  p1y_tp += Sy * p1z_tp;
		p2x_tp += Sx * p2z_tp;  p2y_tp += Sy * p2z_tp;

		// Edge functions
		T e0 = diff_of_products(p1x_tp, p2y_tp, p1y_tp, p2x_tp);
		T e1 = diff_of_products(p2x_tp, p0y_tp, p2y_tp, p0x_tp);
		T e2 = diff_of_products(p0x_tp, p1y_tp, p0y_tp, p1x_tp);

		// Double-precision fallback for edges near zero
		if (sizeof(T) == sizeof(float) && (e0==T(0) || e1==T(0) || e2==T(0))) {
			double p2txp1ty = (double)p2x_tp * (double)p1y_tp;
			double p2typ1tx = (double)p2y_tp * (double)p1x_tp;
			e0 = (T)(p2typ1tx - p2txp1ty);
			double p0txp2ty = (double)p0x_tp * (double)p2y_tp;
			double p0typ2tx = (double)p0y_tp * (double)p2x_tp;
			e1 = (T)(p0typ2tx - p0txp2ty);
			double p1txp0ty = (double)p1x_tp * (double)p0y_tp;
			double p1typ0tx = (double)p1y_tp * (double)p0x_tp;
			e2 = (T)(p1typ0tx - p1txp0ty);
		}

		// Reject based on edge sign
		if ((e0<T(0)||e1<T(0)||e2<T(0)) && (e0>T(0)||e1>T(0)||e2>T(0))) return {};
		T det = e0 + e1 + e2;
		if (det == T(0)) return {};

		// Compute scaled t
		p0z_tp *= Sz; p1z_tp *= Sz; p2z_tp *= Sz;
		T tScaled = e0*p0z_tp + e1*p1z_tp + e2*p2z_tp;
		if (det < T(0) && (tScaled >= T(0) || tScaled < t_max * det)) return {};
		else if (det > T(0) && (tScaled <= T(0) || tScaled > t_max * det)) return {};

		// Barycentrics and t
		T invDet = T(1) / det;
		T b0 = e0 * invDet;
		T b1 = e1 * invDet;
		T b2 = e2 * invDet;
		T t_hit = tScaled * invDet;

		// Conservative t error bound (pbrt-v4 style)
		T maxZt = std::max({std::abs(p0z_tp), std::abs(p1z_tp), std::abs(p2z_tp)});
		T maxXt = std::max({std::abs(p0x_tp), std::abs(p1x_tp), std::abs(p2x_tp)});
		T maxYt = std::max({std::abs(p0y_tp), std::abs(p1y_tp), std::abs(p2y_tp)});
		T deltaZ = gamma_fp<T>(3) * maxZt;
		T deltaX = gamma_fp<T>(5) * (maxXt + maxZt);
		T deltaY = gamma_fp<T>(5) * (maxYt + maxZt);
		T deltaE = T(2) * (gamma_fp<T>(2)*maxXt*maxYt + deltaY*maxXt + deltaX*maxYt);
		T maxE   = std::max({std::abs(e0), std::abs(e1), std::abs(e2)});
		T deltaT = T(3) * (gamma_fp<T>(3)*maxE*maxZt + deltaE*maxZt + deltaZ*maxE)
				   * std::abs(invDet);
		if (t_hit <= deltaT || t_hit < t_min) return {};

		// Interpolated hit point
		T hx = b0*p0x + b1*p1x + b2*p2x;
		T hy = b0*p0y + b1*p1y + b2*p2y;
		T hz = b0*p0z + b1*p1z + b2*p2z;

		// Normal: geometric or interpolated shading normal
		T nnx, nny, nnz;
		if (has_shading_normals) {
			nnx = b0*n0x + b1*n1x + b2*n2x;
			nny = b0*n0y + b1*n1y + b2*n2y;
			nnz = b0*n0z + b1*n1z + b2*n2z;
			T nlen = len3(nnx, nny, nnz);
			if (nlen > T(0)) { nnx /= nlen; nny /= nlen; nnz /= nlen; }
		} else {
			nnx = gnx; nny = gny; nnz = gnz;
			normalize3(nnx, nny, nnz);
		}

		// (u,v) = barycentric (b0, b1)
		ShapeHit<T> hit;
		hit.t  = t_hit;
		hit.nx = nnx; hit.ny = nny; hit.nz = nnz;
		hit.u  = b0; hit.v = b1;
		return hit;
	}

	// -----------------------------------------------------------------------
	// Area-uniform sample using uniform barycentric sampling
	// Reference: pbrt-v4 Triangle::Sample(Point2f u)
	// -----------------------------------------------------------------------
	CPU_GPU ShapeSample<T> sample(T u0, T u1) const {
		T b0, b1, b2;
		SampleUniformTriangle(u0, u1, b0, b1, b2);

		T px_w = b0*p0x + b1*p1x + b2*p2x;
		T py_w = b0*p0y + b1*p1y + b2*p2y;
		T pz_w = b0*p0z + b1*p1z + b2*p2z;

		T nnx, nny, nnz;
		if (has_shading_normals) {
			nnx = b0*n0x + b1*n1x + b2*n2x;
			nny = b0*n0y + b1*n1y + b2*n2y;
			nnz = b0*n0z + b1*n1z + b2*n2z;
			shapes_detail::normalize3(nnx, nny, nnz);
		} else {
			geometric_normal(nnx, nny, nnz);
			shapes_detail::normalize3(nnx, nny, nnz);
		}

		return ShapeSample<T>{px_w, py_w, pz_w,
							  nnx, nny, nnz,
							  b0, b1,
							  pdf_area()};
	}

	// -----------------------------------------------------------------------
	// Solid-angle sample from context point
	//
	// Uses spherical triangle sampling when solid angle is large enough;
	// falls back to area sampling + Jacobian otherwise.
	// Reference: pbrt-v4 Triangle::Sample(const ShapeSampleContext&, Point2f u)
	// -----------------------------------------------------------------------
	CPU_GPU ShapeSample<T> sample_from(const SamplingContext<T>& ctx,
									   T u0, T u1) const {
		using namespace shapes_detail;
		// Mirror pbrt-v4 MinSphericalSampleArea=3e-4, MaxSphericalSampleArea=6.22
		const T solid_angle_min = T(3e-4);
		const T solid_angle_max = T(6.22);

		// Compute solid angle of triangle as seen from ctx
		T sa = (T)SphericalTriangleSolidAngle(
			p0x, p0y, p0z,
			p1x, p1y, p1z,
			p2x, p2y, p2z,
			ctx.px, ctx.py, ctx.pz);

		if (sa < solid_angle_min || sa > solid_angle_max) {
			// Fall back to area sampling
			ShapeSample<T> ss = sample(u0, u1);
			T wix = ss.px - ctx.px;
			T wiy = ss.py - ctx.py;
			T wiz = ss.pz - ctx.pz;
			T wi_len2 = wix*wix + wiy*wiy + wiz*wiz;
			if (wi_len2 == T(0)) { ss.pdf = T(0); return ss; }
			T wi_len = std::sqrt(wi_len2);
			T cos_theta_n = std::abs(dot3(ss.nx, ss.ny, ss.nz,
										  -wix/wi_len, -wiy/wi_len, -wiz/wi_len));
			if (cos_theta_n == T(0)) { ss.pdf = T(0); return ss; }
			ss.pdf = ss.pdf * wi_len2 / cos_theta_n;
			return ss;
		}

		// Solid-angle sample via spherical triangle
		double b0d, b1d, b2d, sa_out_d;
		SampleSphericalTriangle(
			(double)p0x, (double)p0y, (double)p0z,
			(double)p1x, (double)p1y, (double)p1z,
			(double)p2x, (double)p2y, (double)p2z,
			(double)ctx.px, (double)ctx.py, (double)ctx.pz,
			(double)u0, (double)u1,
			&b0d, &b1d, &b2d, &sa_out_d);
		T b0 = (T)b0d, b1 = (T)b1d, b2 = (T)b2d;
		T sa_out = (T)sa_out_d;

		T px_w = b0*p0x + b1*p1x + b2*p2x;
		T py_w = b0*p0y + b1*p1y + b2*p2y;
		T pz_w = b0*p0z + b1*p1z + b2*p2z;

		T nnx, nny, nnz;
		if (has_shading_normals) {
			nnx = b0*n0x + b1*n1x + b2*n2x;
			nny = b0*n0y + b1*n1y + b2*n2y;
			nnz = b0*n0z + b1*n1z + b2*n2z;
			normalize3(nnx, nny, nnz);
		} else {
			geometric_normal(nnx, nny, nnz);
			normalize3(nnx, nny, nnz);
		}

		T pdf_val = (sa_out > T(0)) ? T(1) / sa_out : T(0);

		return ShapeSample<T>{px_w, py_w, pz_w,
							  nnx, nny, nnz,
							  b0, b1,
							  pdf_val};
	}

	// -----------------------------------------------------------------------
	// Solid-angle PDF
	// Reference: pbrt-v4 Triangle::PDF(const ShapeSampleContext&, Vector3f wi)
	// -----------------------------------------------------------------------
	CPU_GPU T pdf_from(const SamplingContext<T>& ctx,
					   T wi_dx, T wi_dy, T wi_dz) const {
		using namespace shapes_detail;
		// Mirror pbrt-v4 MinSphericalSampleArea=3e-4, MaxSphericalSampleArea=6.22
		const T solid_angle_min = T(3e-4);
		const T solid_angle_max = T(6.22);

		T sa = (T)SphericalTriangleSolidAngle(
			p0x, p0y, p0z,
			p1x, p1y, p1z,
			p2x, p2y, p2z,
			ctx.px, ctx.py, ctx.pz);

			if (sa < solid_angle_min || sa > solid_angle_max) {
					// Area PDF -> solid-angle: intersect ray and apply Jacobian
					T wi_len = len3(wi_dx, wi_dy, wi_dz);
					if (wi_len == T(0)) return T(0);
					auto hit = intersect(ctx.px, ctx.py, ctx.pz,
										 wi_dx/wi_len, wi_dy/wi_len, wi_dz/wi_len,
										 T(1e-4), std::numeric_limits<T>::max());
					if (!hit) return T(0);
					T dist = hit->t;
					T cos_theta_n = std::abs(dot3(hit->nx, hit->ny, hit->nz,
												  -wi_dx/wi_len, -wi_dy/wi_len, -wi_dz/wi_len));
					if (cos_theta_n == T(0)) return T(0);
					return pdf_area() * dist*dist / cos_theta_n;
				}

				// Solid-angle sampling: verify wi hits triangle, then return 1/sa
				T wi_len = len3(wi_dx, wi_dy, wi_dz);
				if (wi_len == T(0)) return T(0);
				auto hit = intersect(ctx.px, ctx.py, ctx.pz,
									 wi_dx/wi_len, wi_dy/wi_len, wi_dz/wi_len,
									 T(1e-4), std::numeric_limits<T>::max());
				if (!hit) return T(0);
				return (sa > T(0)) ? T(1) / sa : T(0);
			}
		};


		// ===========================================================================
		// CylinderShape<T>
		// ===========================================================================
		//
		// Axis-aligned cylinder along the Z axis, centered at (cx, cy, *).
		// Clipped to [z_min, z_max] and to an azimuthal sweep phi_max (radians).
		//
		// Reference: pbrt-v4 Cylinder (shapes.h / shapes.cpp)
		//
		// API matches SphereShape / DiskShape:
		//   intersect(rox,roy,roz, rdx,rdy,rdz, t_min, t_max)
		//   area()
		//   pdf_area()
		//   sample(u0, u1)           -- area-uniform
		//   sample_from(ctx, u0, u1) -- solid-angle from shading point
		//   pdf_from(ctx, wi_*)      -- solid-angle PDF
		// ===========================================================================

		template<typename T>
		struct CylinderShape {
			T cx, cy;       // axis center x/y (the axis runs parallel to Z through (cx,cy))
			T z_min, z_max; // z extent
			T radius;       // cylinder radius
			T phi_max;      // azimuthal sweep in radians (2*pi for full cylinder)

			CPU_GPU static CylinderShape make(T cx, T cy, T z_min, T z_max, T radius) {
				const T pi2 = T(2) * T(3.14159265358979323846);
				return CylinderShape{cx, cy, z_min, z_max, radius, pi2};
			}

			CPU_GPU static CylinderShape make_partial(T cx, T cy, T z_min, T z_max,
													  T radius, T phi_max_rad) {
				return CylinderShape{cx, cy, z_min, z_max, radius, phi_max_rad};
			}

			// -----------------------------------------------------------------------
			// Area
			// pbrt-v4: (zMax - zMin) * radius * phiMax
			// -----------------------------------------------------------------------
			CPU_GPU T area() const { return (z_max - z_min) * radius * phi_max; }
			CPU_GPU T pdf_area() const { return T(1) / area(); }

			// -----------------------------------------------------------------------
			// Intersection
			// pbrt-v4 Cylinder::BasicIntersect -- object-space quadratic:
			//   a = dx^2 + dy^2,  b = 2*(dx*ox + dy*oy),  c = ox^2 + oy^2 - r^2
			// Discriminant computed via stable displaced-centre formula.
			// -----------------------------------------------------------------------
			CPU_GPU std::optional<ShapeHit<T>>
			intersect(T rox, T roy, T roz,
					  T rdx, T rdy, T rdz,
					  T t_min, T t_max) const
			{
				using namespace shapes_detail;

				// Shift ray origin to cylinder-axis frame
				T ox = rox - cx, oy = roy - cy, oz = roz;
				T dx = rdx,       dy = rdy,       dz = rdz;

				T a = dx*dx + dy*dy;
				T b = T(2) * (dx*ox + dy*oy);
				T c = ox*ox + oy*oy - radius*radius;

				// Degenerate: ray parallel to cylinder axis
				if (a == T(0)) return {};

				T t0, t1;
				if (!solve_quadratic(a, b, c, t0, t1)) return {};

				// Pick the nearest valid root
				auto try_root = [&](T t) -> std::optional<ShapeHit<T>> {
					if (t < t_min || t > t_max) return {};
					T hx = ox + t*dx;
					T hy = oy + t*dy;
					T hz = oz + t*dz;

					// z-clip
					if (hz < z_min || hz > z_max) return {};

					// phi-clip
					T phi = std::atan2(hy, hx);
					if (phi < T(0)) phi += T(2) * T(3.14159265358979323846);
					if (phi > phi_max) return {};

					// Reproject hit point to exact cylinder surface (remove floating-point drift)
					T hit_rad = std::sqrt(hx*hx + hy*hy);
					if (hit_rad > T(0)) { hx *= radius/hit_rad; hy *= radius/hit_rad; }

					// (u, v) parameterization matching pbrt-v4
					T u_coord = phi / phi_max;
					T v_coord = (hz - z_min) / (z_max - z_min);

					// Outward normal in world space (cylinder axis is Z, center at cx,cy)
					T nx = hx / radius;
					T ny = hy / radius;
					T nz = T(0);

					ShapeHit<T> hit;
					hit.t  = t;
					hit.nx = nx; hit.ny = ny; hit.nz = nz;
					hit.u  = u_coord; hit.v = v_coord;
					return hit;
				};

				auto h = try_root(t0);
				if (h) return h;
				return try_root(t1);
			}

			// -----------------------------------------------------------------------
			// Area-uniform surface sample
			// pbrt-v4 Cylinder::Sample(Point2f u):
			//   z   = Lerp(u[0], zMin, zMax)
			//   phi = u[1] * phiMax
			//   p   = (r*cos(phi), r*sin(phi), z)
			//   n   = Normalize((px, py, 0))
			//   pdf = 1 / Area()
			// -----------------------------------------------------------------------
			CPU_GPU ShapeSample<T> sample(T u0, T u1) const {
				T z   = z_min + u0 * (z_max - z_min);   // Lerp
				T phi = u1 * phi_max;

				T lx  = radius * std::cos(phi);
				T ly  = radius * std::sin(phi);

				// Reprojection (remove drift, same as pbrt-v4)
				T hit_rad = std::sqrt(lx*lx + ly*ly);
				if (hit_rad > T(0)) { lx *= radius/hit_rad; ly *= radius/hit_rad; }

				// World-space position
				T px_w = cx + lx;
				T py_w = cy + ly;
				T pz_w = z;

				// Outward normal
				T nx = lx / radius, ny = ly / radius, nz_val = T(0);

				T u_coord = phi / phi_max;
				T v_coord = (z - z_min) / (z_max - z_min);

				return ShapeSample<T>{px_w, py_w, pz_w,
									  nx, ny, nz_val,
									  u_coord, v_coord,
									  pdf_area()};
			}

			// -----------------------------------------------------------------------
			// Solid-angle sample from shading point
			// pbrt-v4 Cylinder::Sample(const ShapeSampleContext&, Point2f u):
			//   Sample uniformly by area, convert PDF to solid angle.
			// -----------------------------------------------------------------------
			CPU_GPU ShapeSample<T> sample_from(const SamplingContext<T>& ctx,
											   T u0, T u1) const {
				ShapeSample<T> ss = sample(u0, u1);

				T wi_x = ss.px - ctx.px;
				T wi_y = ss.py - ctx.py;
				T wi_z = ss.pz - ctx.pz;
				T dist2 = wi_x*wi_x + wi_y*wi_y + wi_z*wi_z;
				if (dist2 == T(0)) { ss.pdf = T(0); return ss; }

				T inv_dist = T(1) / std::sqrt(dist2);
				T wix = wi_x * inv_dist;
				T wiy = wi_y * inv_dist;
				T wiz = wi_z * inv_dist;

				using namespace shapes_detail;
				T cos_theta = std::abs(dot3(ss.nx, ss.ny, ss.nz, -wix, -wiy, -wiz));
				if (cos_theta == T(0)) { ss.pdf = T(0); return ss; }

				// pdf_area * dist^2 / |cos_theta|
				ss.pdf = pdf_area() * dist2 / cos_theta;
				return ss;
			}

			// -----------------------------------------------------------------------
			// Solid-angle PDF
			// pbrt-v4 Cylinder::PDF(const ShapeSampleContext&, Vector3f wi):
			//   Intersect ray, then pdf_area * dist^2 / |cos_theta|
			// -----------------------------------------------------------------------
			CPU_GPU T pdf_from(const SamplingContext<T>& ctx,
							   T wi_dx, T wi_dy, T wi_dz) const {
				using namespace shapes_detail;
				T wi_len = std::sqrt(wi_dx*wi_dx + wi_dy*wi_dy + wi_dz*wi_dz);
				if (wi_len == T(0)) return T(0);
				T wix = wi_dx/wi_len, wiy = wi_dy/wi_len, wiz = wi_dz/wi_len;

				auto hit = intersect(ctx.px, ctx.py, ctx.pz,
									 wix, wiy, wiz,
									 T(1e-4), std::numeric_limits<T>::max());
				if (!hit) return T(0);

				T dist2 = sq(hit->t);
				T cos_theta = std::abs(dot3(hit->nx, hit->ny, hit->nz, -wix, -wiy, -wiz));
				if (cos_theta == T(0)) return T(0);

					T pdf = pdf_area() * dist2 / cos_theta;
						return std::isfinite(pdf) ? pdf : T(0);
					}
				};

				// ===========================================================================
				// CurveShape
				// ===========================================================================
				//
				// A single cubic Bezier curve segment (hair / ribbon / cylinder).
				// Mirrors pbrt-v4 Curve (shapes.h / shapes.cpp).
				//
				// Three curve types (CurveType enum):
				//   Flat     -- always faces the ray (camera-facing ribbon).
				//   Cylinder -- round cross-section, normal rotates around dpdu.
				//   Ribbon   -- fixed orientation via per-endpoint normals (n0,n1).
				//
				// The curve stores 4 Bezier control points in world space, plus tapered
				// widths width0 (at u=uMin) and width1 (at u=uMax).  A full hair strand
				// is split into several CurveShape segments each covering a [uMin,uMax]
				// sub-interval of the original spline.
				//
				// Intersection algorithm mirrors pbrt-v4 exactly:
				//   1. Transform ray into a "ray-facing" frame (LookAt-style).
				//   2. Project control points into that frame so the ray becomes +Z.
				//   3. Recursively subdivide until the 2D bounding box test is small
				//      enough, then test the leaf segment analytically.
				//
				// Reference: pbrt-v4 Curve (shapes.h / shapes.cpp)
				//
				// API:
				//   intersect(rox,roy,roz, rdx,rdy,rdz, t_min, t_max, hit)
				//   area()
				// ===========================================================================

				enum class CurveType { Flat, Cylinder, Ribbon };

				template<typename T>
				struct CurveShape {
					// 4 cubic Bezier control points in world space
					T cpx[4], cpy[4], cpz[4];
					// Parametric range [uMin, uMax] within the parent strand
					T uMin, uMax;
					// Tapered width at uMin and uMax
					T width0, width1;
					// Curve type
					CurveType type;
					// Per-endpoint normals for Ribbon type (world space, unit length)
					// Only valid when type == CurveType::Ribbon
					T n0x, n0y, n0z;   // normal at uMin endpoint
					T n1x, n1y, n1z;   // normal at uMax endpoint
					T normalAngle;      // angle between n0 and n1
					T invSinNormalAngle;

					// ---------------------------------------------------------------
					// Factory helpers
					// ---------------------------------------------------------------

					// Make a Flat or Cylinder curve segment (no ribbon normals needed)
					CPU_GPU static CurveShape make(
						const T cpx_[4], const T cpy_[4], const T cpz_[4],
						T uMin_, T uMax_,
						T width0_, T width1_,
						CurveType type_ = CurveType::Flat)
					{
						CurveShape c;
						for (int i = 0; i < 4; ++i) {
							c.cpx[i] = cpx_[i]; c.cpy[i] = cpy_[i]; c.cpz[i] = cpz_[i];
						}
						c.uMin = uMin_; c.uMax = uMax_;
						c.width0 = width0_; c.width1 = width1_;
						c.type = type_;
						c.n0x = c.n0y = c.n0z = T(0);
						c.n1x = c.n1y = c.n1z = T(0);
						c.normalAngle = T(0); c.invSinNormalAngle = T(0);
						return c;
					}

					// Make a Ribbon curve segment (requires endpoint normals)
					CPU_GPU static CurveShape make_ribbon(
						const T cpx_[4], const T cpy_[4], const T cpz_[4],
						T uMin_, T uMax_,
						T width0_, T width1_,
						T n0x_, T n0y_, T n0z_,
						T n1x_, T n1y_, T n1z_)
					{
						CurveShape c = make(cpx_, cpy_, cpz_, uMin_, uMax_, width0_, width1_,
											CurveType::Ribbon);
						// Normalize endpoint normals
						T len0 = std::sqrt(n0x_*n0x_ + n0y_*n0y_ + n0z_*n0z_);
						T len1 = std::sqrt(n1x_*n1x_ + n1y_*n1y_ + n1z_*n1z_);
						if (len0 > T(0)) { n0x_ /= len0; n0y_ /= len0; n0z_ /= len0; }
						if (len1 > T(0)) { n1x_ /= len1; n1y_ /= len1; n1z_ /= len1; }
						c.n0x = n0x_; c.n0y = n0y_; c.n0z = n0z_;
						c.n1x = n1x_; c.n1y = n1y_; c.n1z = n1z_;
						T cosAngle = n0x_*n1x_ + n0y_*n1y_ + n0z_*n1z_;
						cosAngle = cosAngle < T(-1) ? T(-1) : (cosAngle > T(1) ? T(1) : cosAngle);
						c.normalAngle = std::acos(cosAngle);
						c.invSinNormalAngle = (c.normalAngle > T(1e-6))
							? T(1) / std::sin(c.normalAngle) : T(0);
						return c;
					}

					// ---------------------------------------------------------------
					// area() -- approximate arc-length × average width
					// Mirrors pbrt-v4 Curve::Area()
					// ---------------------------------------------------------------
					CPU_GPU T area() const {
						// Sub-interval control points
						float cp[4][3];
						_sub_cp(cp);
						T w0 = _lerp(uMin, width0, width1);
						T w1 = _lerp(uMax, width0, width1);
						T avgWidth = (w0 + w1) * T(0.5);
						T approxLength = T(0);
						for (int i = 0; i < 3; ++i) {
							T dx = T(cp[i+1][0] - cp[i][0]);
							T dy = T(cp[i+1][1] - cp[i][1]);
							T dz = T(cp[i+1][2] - cp[i][2]);
							approxLength += std::sqrt(dx*dx + dy*dy + dz*dz);
						}
						return approxLength * avgWidth;
					}

					// ---------------------------------------------------------------
					// CurveHit -- result of intersect()
					// ---------------------------------------------------------------
					struct CurveHit {
						T t;
						T u, v;          // (u,v) on curve: u along strand, v across width
						T dpdu_x, dpdu_y, dpdu_z;  // tangent along curve at hit
						T dpdv_x, dpdv_y, dpdv_z;  // tangent across curve at hit
						T nx, ny, nz;    // shading normal (world space)
					};

					// ---------------------------------------------------------------
					// intersect()
					// Returns true if the ray hits the curve within [t_min, t_max].
					// On hit, *out is filled with hit details.
					// Mirrors pbrt-v4 Curve::IntersectRay + RecursiveIntersect.
					// ---------------------------------------------------------------
					CPU_GPU bool intersect(
						T rox, T roy, T roz,
						T rdx, T rdy, T rdz,
						T t_min, T t_max,
						CurveHit* out = nullptr) const
					{
						// Sub-interval control points in world space (float precision)
						float cpW[4][3];
						_sub_cp(cpW);

						// Build a "LookAt" transform that maps the ray to +Z axis.
						// dx is a direction in the plane perpendicular to the ray, used
						// to orient the frame. We pick Cross(ray.d, cp[3]-cp[0]).
						float rdF[3] = { float(rdx), float(rdy), float(rdz) };
						float diff[3] = { cpW[3][0] - cpW[0][0],
										  cpW[3][1] - cpW[0][1],
										  cpW[3][2] - cpW[0][2] };
						float dx[3], dy_unused[3];
						_cross3(rdF, diff, dx);
						if (_dot3(dx,dx) == 0.f)
							_coord_system(rdF, dx, dy_unused);

						// Compute ray-from-object 4×4 LookAt transform
						// (position = ray.o, forward = normalize(ray.d), up = dx)
						float ro[3] = { float(rox), float(roy), float(roz) };
						float M[4][4]; // row-major world-to-ray transform
						_look_at(ro, rdF, dx, M);

						// Transform control points to ray space
						float cp[4][3];
						for (int i = 0; i < 4; ++i)
							_transform_point(M, cpW[i], cp[i]);

						// Broad-phase AABB test in ray space
						float maxWidth = std::max(
							_lerp(float(uMin), float(width0), float(width1)),
							_lerp(float(uMax), float(width0), float(width1)));
						// Curve bounding box in ray space (union of edge midpoints)
						float bbMin[3], bbMax[3];
						_bound4(cp, bbMin, bbMax);
						for (int k = 0; k < 3; ++k) {
							bbMin[k] -= 0.5f * maxWidth;
							bbMax[k] += 0.5f * maxWidth;
						}
						// Ray in ray space is origin=(0,0,0) dir=(0,0,|d|)
						float rayLen = std::sqrt(_dot3(rdF,rdF));
						// Ray AABB: x,y in [-w,w], z in [0, rayLen*t_max]
						if (bbMax[0] < 0.f || bbMin[0] > 0.f ||
							bbMax[1] < 0.f || bbMin[1] > 0.f ||
							bbMax[2] < 0.f             || bbMin[2] > rayLen * float(t_max))
							return false;

						// Compute maximum recursion depth: mirrors pbrt-v4 log4 formula
						float L0 = 0.f;
						for (int i = 0; i < 2; ++i) {
							for (int k = 0; k < 3; ++k) {
								float v = std::abs(cp[i][k] - 2.f*cp[i+1][k] + cp[i+2][k]);
								if (v > L0) L0 = v;
							}
						}
						int maxDepth = 0;
						if (L0 > 0.f) {
							float maxW = std::max(float(width0), float(width1));
							float eps  = maxW * 0.05f;
							float arg  = 1.41421356237f * 6.f * L0 / (8.f * eps);
							if (arg > 1.f) {
								int r0 = (int)(std::log(arg) / std::log(4.f));
								maxDepth = r0 < 0 ? 0 : (r0 > 10 ? 10 : r0);
							}
						}

						// Recursive intersection in ray space
						// objectFromRay = inverse of M (a rigid LookAt, so transpose of rotation)
						float Minv[4][4];
						_invert_lookat(M, Minv);

						T tBest = t_max;
						bool hit = _recursive_intersect(
							ro, rdF, rayLen,
							cp, M, Minv, cpW,
							float(uMin), float(uMax),
							float(t_min), float(t_max),
							maxDepth,
							tBest, out);
						return hit;
					}

				private:
					// ---------------------------------------------------------------
					// Internal helpers
					// ---------------------------------------------------------------

					CPU_GPU static T _lerp(T t, T a, T b) { return (T(1)-t)*a + t*b; }
					CPU_GPU static float _lerp(float t, float a, float b) { return (1.f-t)*a + t*b; }

					CPU_GPU static float _dot3(const float* a, const float* b) {
						return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
					}
					CPU_GPU static float _len3(const float* a) {
						return std::sqrt(_dot3(a,a));
					}
					CPU_GPU static void _cross3(const float* a, const float* b, float* out) {
						out[0] = a[1]*b[2] - a[2]*b[1];
						out[1] = a[2]*b[0] - a[0]*b[2];
						out[2] = a[0]*b[1] - a[1]*b[0];
					}
					CPU_GPU static void _normalize3(float* v) {
						float len = _len3(v);
						if (len > 0.f) { v[0]/=len; v[1]/=len; v[2]/=len; }
					}

					// Frisvad-style orthonormal basis (Duff et al. 2017)
					CPU_GPU static void _coord_system(const float* n, float* t, float* b) {
						float sign = (n[2] >= 0.f) ? 1.f : -1.f;
						float a  = -1.f / (sign + n[2]);
						float bv = n[0] * n[1] * a;
						t[0] = 1.f + sign * n[0]*n[0]*a;
						t[1] = sign * bv;
						t[2] = -sign * n[0];
						b[0] = bv;
						b[1] = sign + n[1]*n[1]*a;
						b[2] = -n[1];
					}

					// Build LookAt 4×4 row-major matrix: maps world-space points so that
					// ro -> origin, (ro + normalize(dir)) -> +Z, up guides X axis.
					// Mirrors pbrt-v4 LookAt(pos, pos+dir, up).
					CPU_GPU static void _look_at(const float* pos, const float* dir,
												 const float* up, float M[4][4])
					{
						float fwd[3] = { dir[0], dir[1], dir[2] };
						_normalize3(fwd);
						float right[3];
						float up_tmp[3] = { up[0], up[1], up[2] };
						_normalize3(up_tmp);
						_cross3(up_tmp, fwd, right);
						if (_dot3(right,right) < 1e-12f) {
							float dummy[3];
							_coord_system(fwd, right, dummy);
						}
						_normalize3(right);
						float newUp[3];
						_cross3(fwd, right, newUp);

						// Camera-to-world columns: right, newUp, fwd
						// World-to-camera (what we want) is transpose * translate
						// Row 0: right
						M[0][0]=right[0]; M[0][1]=right[1]; M[0][2]=right[2];
						M[0][3]=-(right[0]*pos[0]+right[1]*pos[1]+right[2]*pos[2]);
						// Row 1: newUp
						M[1][0]=newUp[0]; M[1][1]=newUp[1]; M[1][2]=newUp[2];
						M[1][3]=-(newUp[0]*pos[0]+newUp[1]*pos[1]+newUp[2]*pos[2]);
						// Row 2: fwd
						M[2][0]=fwd[0]; M[2][1]=fwd[1]; M[2][2]=fwd[2];
						M[2][3]=-(fwd[0]*pos[0]+fwd[1]*pos[1]+fwd[2]*pos[2]);
						// Row 3: homogeneous
						M[3][0]=0; M[3][1]=0; M[3][2]=0; M[3][3]=1;
					}

					// Invert a LookAt (rigid body): transpose rotation, re-derive translation
					CPU_GPU static void _invert_lookat(const float M[4][4], float Minv[4][4]) {
						// Rotation part is rows 0-2 cols 0-2; inverse = transpose
						for (int i = 0; i < 3; ++i)
							for (int j = 0; j < 3; ++j)
								Minv[i][j] = M[j][i];
						// Translation: -R^T * t
						for (int i = 0; i < 3; ++i)
							Minv[i][3] = -(Minv[i][0]*M[0][3] + Minv[i][1]*M[1][3] + Minv[i][2]*M[2][3]);
						Minv[3][0]=0; Minv[3][1]=0; Minv[3][2]=0; Minv[3][3]=1;
					}

					// Apply 4×4 row-major matrix to a 3D point (w=1)
					CPU_GPU static void _transform_point(const float M[4][4],
														 const float p[3], float out[3]) {
						out[0] = M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]*p[2] + M[0][3];
						out[1] = M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]*p[2] + M[1][3];
						out[2] = M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]*p[2] + M[2][3];
					}

					// Apply 4×4 row-major matrix to a 3D vector (w=0)
					CPU_GPU static void _transform_vector(const float M[4][4],
														  const float v[3], float out[3]) {
						out[0] = M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2];
						out[1] = M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2];
						out[2] = M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2];
					}

					// AABB of 4 control points
					CPU_GPU static void _bound4(const float cp[4][3],
												float bbMin[3], float bbMax[3]) {
						for (int k = 0; k < 3; ++k) {
							bbMin[k] = cp[0][k]; bbMax[k] = cp[0][k];
							for (int i = 1; i < 4; ++i) {
								if (cp[i][k] < bbMin[k]) bbMin[k] = cp[i][k];
								if (cp[i][k] > bbMax[k]) bbMax[k] = cp[i][k];
							}
						}
					}

					// Compute sub-interval control points via CubicBezierControlPoints
					CPU_GPU void _sub_cp(float out[4][3]) const {
						float src[4][3] = {
							{float(cpx[0]),float(cpy[0]),float(cpz[0])},
							{float(cpx[1]),float(cpy[1]),float(cpz[1])},
							{float(cpx[2]),float(cpy[2]),float(cpz[2])},
							{float(cpx[3]),float(cpy[3]),float(cpz[3])}
						};
						splines::CubicBezierControlPoints(src, float(uMin), float(uMax), out);
					}

					// Evaluate cubic Bezier point and optional derivative at parameter w ∈ [0,1]
					// Uses the raw control points already in a given space.
					CPU_GPU static void _eval_bezier(const float cp[4][3], float w,
													 float p[3], float dp[3]) {
						float src[4][3];
						for (int i = 0; i < 4; ++i)
							for (int k = 0; k < 3; ++k) src[i][k] = cp[i][k];
						splines::EvaluateCubicBezierD(src, w, p, dp);
					}

					// Subdivide 4 control points into 7 (left 4 + right 4 sharing midpoint)
					CPU_GPU static void _subdivide(const float cp[4][3], float out7[7][3]) {
						float src[4][3];
						for (int i = 0; i < 4; ++i)
							for (int k = 0; k < 3; ++k) src[i][k] = cp[i][k];
						splines::SubdivideCubicBezier(src, out7);
					}

					// Recursive intersection (ray-space, mirrors pbrt-v4 RecursiveIntersect)
					CPU_GPU bool _recursive_intersect(
						const float ro[3], const float rd[3], float rayLen,
						const float cp[4][3],      // control points in ray space
						const float M[4][4],       // world->ray transform  (rayFromObject)
						const float Minv[4][4],    // ray->world transform  (objectFromRay)
						const float cpW[4][3],     // control points in world space (for normals)
						float u0, float u1,
						float t_min, float t_max,
						int depth,
						T& tBest, CurveHit* out) const
					{
						if (depth > 0) {
							// Split into two halves
							float cpSplit[7][3];
							_subdivide(cp, cpSplit);
							float uMid = 0.5f * (u0 + u1);
							float u[3] = { u0, uMid, u1 };
							bool hit = false;
							for (int seg = 0; seg < 2; ++seg) {
								// Build child cp from split
								float child[4][3];
								for (int i = 0; i < 4; ++i)
									for (int k = 0; k < 3; ++k)
										child[i][k] = cpSplit[3*seg + i][k];

								// Child bounding box test
								float maxW = std::max(
									_lerp(u[seg],   float(width0), float(width1)),
									_lerp(u[seg+1], float(width0), float(width1)));
								float bbMin[3], bbMax[3];
								_bound4(child, bbMin, bbMax);
								for (int k = 0; k < 3; ++k) {
									bbMin[k] -= 0.5f*maxW;
									bbMax[k] += 0.5f*maxW;
								}
								if (bbMax[0] < 0.f || bbMin[0] > 0.f ||
									bbMax[1] < 0.f || bbMin[1] > 0.f ||
									bbMax[2] < 0.f         || bbMin[2] > rayLen * float(tBest))
									continue;

								bool childHit = _recursive_intersect(
								ro, rd, rayLen, child, M, Minv, cpW,
									u[seg], u[seg+1], t_min, t_max,
									depth - 1, tBest, out);
								if (childHit) {
									hit = true;
									if (!out) return true; // shadow ray: early out
								}
							}
							return hit;

						} else {
							// Leaf: analytic intersection test
							// Test perpendicular tangent clipping planes at endpoints
							// (mirrors pbrt-v4 RecursiveIntersect leaf case)
							float edge0 = (cp[1][1]-cp[0][1])*(-cp[0][1])
										+ cp[0][0]*(cp[0][0]-cp[1][0]);
							if (edge0 < 0.f) return false;
							float edge1 = (cp[2][1]-cp[3][1])*(-cp[3][1])
										+ cp[3][0]*(cp[3][0]-cp[2][0]);
							if (edge1 < 0.f) return false;

							// Find parameter w minimizing distance in XY plane
							float sx = cp[3][0] - cp[0][0];
							float sy = cp[3][1] - cp[0][1];
							float denom = sx*sx + sy*sy;
							if (denom == 0.f) return false;
							float w = (-cp[0][0]*sx - cp[0][1]*sy) / denom;

							// Curve parameter and hit width
							float u = u0 + w * (u1 - u0);
							u = u < u0 ? u0 : (u > u1 ? u1 : u);
							float hitWidth = _lerp(u, float(width0), float(width1));

							// For ribbon: scale hitWidth by |dot(normal, ray.d)|
							float nHitX = 0.f, nHitY = 0.f, nHitZ = 0.f;
							if (type == CurveType::Ribbon) {
								if (normalAngle == T(0)) {
									nHitX = float(n0x); nHitY = float(n0y); nHitZ = float(n0z);
								} else {
									float uN = u;  // global u
									float sin0 = std::sin((1.f - uN) * float(normalAngle))
											   * float(invSinNormalAngle);
									float sin1 = std::sin(uN * float(normalAngle))
											   * float(invSinNormalAngle);
									nHitX = sin0*float(n0x) + sin1*float(n1x);
									nHitY = sin0*float(n0y) + sin1*float(n1y);
									nHitZ = sin0*float(n0z) + sin1*float(n1z);
								}
								// dot(nHit, ray.d) / |ray.d|
								float ndotd = nHitX*rd[0] + nHitY*rd[1] + nHitZ*rd[2];
								float absNdotd = ndotd < 0.f ? -ndotd : ndotd;
								hitWidth *= absNdotd / rayLen;
							}

							// Evaluate curve at w to find closest point
							w = w < 0.f ? 0.f : (w > 1.f ? 1.f : w);
							float pc[3], dpcdw[3];
							_eval_bezier(cp, w, pc, dpcdw);

							float ptCurveDist2 = pc[0]*pc[0] + pc[1]*pc[1];
							float halfW = 0.5f * hitWidth;
							if (ptCurveDist2 > halfW*halfW) return false;
							if (pc[2] < 0.f || pc[2] > rayLen * float(tBest)) return false;

							float tHit = pc[2] / rayLen;
							if (tHit < float(t_min) || tHit > float(tBest)) return false;

							tBest = T(tHit);

							if (out) {
								// v coordinate (position across width)
								float ptCurveDist = std::sqrt(ptCurveDist2);
								float edgeFunc = dpcdw[0]*(-pc[1]) + pc[0]*dpcdw[1];
								float v = (edgeFunc > 0.f)
									? 0.5f + ptCurveDist / hitWidth
									: 0.5f - ptCurveDist / hitWidth;

								// dpdu: tangent along curve at hit point (world space)
								// cpW covers [u0,u1]; remap global u -> local t in [0,1]
								float uLocal = (u1 > u0) ? (u - u0) / (u1 - u0) : 0.f;
								uLocal = uLocal < 0.f ? 0.f : (uLocal > 1.f ? 1.f : uLocal);
								float pc_w[3], dpdu_ray[3];
								_eval_bezier(cpW, uLocal, pc_w, dpdu_ray);

								// dpdv: across-width direction
								// pbrt-v4: dpduPlane = objectFromRay.ApplyInverse(dpdu)
								//   ApplyInverse on vector applies forward (world->ray) M
								float dpduPlane[3];
								_transform_vector(M, dpdu_ray, dpduPlane);
								// pbrt-v4: Normalize(Vector3f(-dpduPlane.y, dpduPlane.x, 0))
								// uses XY-only magnitude, not full 3D length
								float len_dpduXY = std::sqrt(dpduPlane[0]*dpduPlane[0] + dpduPlane[1]*dpduPlane[1]);
								if (len_dpduXY == 0.f) len_dpduXY = 1.f;
								float dpdvPlane[3] = {
									-dpduPlane[1] / len_dpduXY * hitWidth,
									 dpduPlane[0] / len_dpduXY * hitWidth,
									 0.f
								};
								if (type == CurveType::Cylinder) {
									// Rotate dpdvPlane around dpduPlane by -theta
									// pbrt-v4: Rotate(-theta, dpduPlane) where theta = Lerp(v,-90,90)
									float theta = v * 180.f - 90.f;
									float rad   = -theta * 3.14159265358979323846f / 180.f;
									float cosT  = std::cos(rad), sinT = std::sin(rad);
									// Rodrigues rotation around unit axis along dpduPlane
									float len_dpdu = _len3(dpduPlane);
									if (len_dpdu == 0.f) len_dpdu = 1.f;
									float ax = dpduPlane[0]/len_dpdu,
										  ay = dpduPlane[1]/len_dpdu,
										  az = dpduPlane[2]/len_dpdu;
									float aAxis[3] = {ax, ay, az};
									float dot = ax*dpdvPlane[0]+ay*dpdvPlane[1]+az*dpdvPlane[2];
									float crossVec[3];
									_cross3_s(ax,ay,az, dpdvPlane[0],dpdvPlane[1],dpdvPlane[2], crossVec);
									for (int k = 0; k < 3; ++k)
										dpdvPlane[k] = cosT*dpdvPlane[k]
													 + sinT*crossVec[k]
													 + (1.f-cosT)*dot*aAxis[k];
								}

								float dpdv_world[3];
								if (type == CurveType::Ribbon) {
									// dpdv = normalize(cross(nHit, dpdu)) * hitWidth
									float nH[3] = {nHitX, nHitY, nHitZ};
									_cross3(nH, dpdu_ray, dpdv_world);
									float l = _len3(dpdv_world);
									if (l > 0.f) for (int k=0; k<3; ++k) dpdv_world[k] *= hitWidth/l;
								} else {
									_transform_vector(Minv, dpdvPlane, dpdv_world);
								}

								// Shading normal = normalize(cross(dpdu, dpdv))
								float nv[3];
								_cross3_s(dpdu_ray[0],dpdu_ray[1],dpdu_ray[2],
										  dpdv_world[0],dpdv_world[1],dpdv_world[2], nv);
								float nl = _len3(nv);
								if (nl > 0.f) { nv[0]/=nl; nv[1]/=nl; nv[2]/=nl; }

								out->t      = T(tHit);
								out->u      = T(u);
								out->v      = T(v);
								out->dpdu_x = T(dpdu_ray[0]);
								out->dpdu_y = T(dpdu_ray[1]);
								out->dpdu_z = T(dpdu_ray[2]);
								out->dpdv_x = T(dpdv_world[0]);
								out->dpdv_y = T(dpdv_world[1]);
								out->dpdv_z = T(dpdv_world[2]);
								out->nx     = T(nv[0]);
								out->ny     = T(nv[1]);
								out->nz     = T(nv[2]);
							}
							return true;
						}
					}

					CPU_GPU static void _cross3_s(float ax, float ay, float az,
												  float bx, float by, float bz,
												  float* out) {
						out[0] = ay*bz - az*by;
						out[1] = az*bx - ax*bz;
						out[2] = ax*by - ay*bx;
					}
				};


