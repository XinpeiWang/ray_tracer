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
