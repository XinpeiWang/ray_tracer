#pragma once
// shape_sphere.h -- part of shapes.h (which includes it): SphereShape<T>.

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

#include "shape_common.h"

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
#if defined(__CUDACC__)
		T th_z_min = fmax(T(-1), fmin(T(1), z_min / r));
		T th_z_max = fmax(T(-1), fmin(T(1), z_max / r));
#else
		T th_z_min = std::max(T(-1), std::min(T(1), z_min / r));
		T th_z_max = std::max(T(-1), std::min(T(1), z_max / r));
#endif
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
#if defined(__CUDACC__)
		T cos_theta = fmax(T(-1), fmin(T(1), hz / r));
#else
		T cos_theta = std::max(T(-1), std::min(T(1), hz / r));
#endif
		T theta = std::acos(cos_theta);
		T theta_z_min = std::acos(th_z_max);  // note: cos is monotone decreasing
		T theta_z_max = std::acos(th_z_min);
		T u_coord = phi / phi_max;
		T v_coord = (theta_z_max > theta_z_min)
					? (theta - theta_z_min) / (theta_z_max - theta_z_min)
					: T(0);

		// pbrt-v4 pError for sphere: gamma(5) * |p_hit| per component
		// Reference: Sphere::InteractionFromIntersection, shapes.h -- gamma(5)*Abs(pHit)
		T g5 = gamma_fp<T>(5);
		ShapeHit<T> hit;
		hit.t  = t_hit;
		hit.nx = nnx; hit.ny = nny; hit.nz = nnz;
		hit.u  = u_coord; hit.v = v_coord;
		hit.ex = std::abs(hx) * g5;
		hit.ey = std::abs(hy) * g5;
		hit.ez = std::abs(hz) * g5;
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
#if defined(__CUDACC__)
		T cos_theta = fmax(T(-1), fmin(T(1), pz_obj / r));
#else
		T cos_theta = std::max(T(-1), std::min(T(1), pz_obj / r));
#endif
		T theta = std::acos(cos_theta);
		T phi   = std::atan2(py_obj, px_obj);
		if (phi < T(0)) phi += T(2) * pi;
#if defined(__CUDACC__)
		T th_z_min = fmax(T(-1), fmin(T(1), z_min / r));
		T th_z_max = fmax(T(-1), fmin(T(1), z_max / r));
#else
		T th_z_min = std::max(T(-1), std::min(T(1), z_min / r));
		T th_z_max = std::max(T(-1), std::min(T(1), z_max / r));
#endif
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
		// Qualified: a TU that also pulls in bxdfs_base.h's own global
		// normalize3<T> (e.g. via a material header) sees both that one and
		// this file's `using namespace shapes_detail;` version, which is
		// ambiguous for an unqualified call - SphereShape<T> was never
		// actually instantiated in such a TU until sphere_clipped_hittable.h
		// started using it, so this was latent rather than previously dead.
		shapes_detail::normalize3(frame_z_x, frame_z_y, frame_z_z);

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
#if defined(__CUDACC__)
		T cos_theta_uv = fmax(T(-1), fmin(T(1), pz_obj / r));
#else
		T cos_theta_uv = std::max(T(-1), std::min(T(1), pz_obj / r));
#endif
		T theta_uv = std::acos(cos_theta_uv);
		T phi_uv   = std::atan2(py_obj, px_obj);
		if (phi_uv < T(0)) phi_uv += T(2) * pi;
#if defined(__CUDACC__)
		T th_z_min = fmax(T(-1), fmin(T(1), z_min / r));
		T th_z_max = fmax(T(-1), fmin(T(1), z_max / r));
#else
		T th_z_min = std::max(T(-1), std::min(T(1), z_min / r));
		T th_z_max = std::max(T(-1), std::min(T(1), z_max / r));
#endif
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
		// Qualified - see sample_from()'s identical fix above for why.
		shapes_detail::normalize3(frame_z_x, frame_z_y, frame_z_z);

		T cos_wi = dot3(wix,wiy,wiz, frame_z_x,frame_z_y,frame_z_z);
		if (cos_wi < cos_theta_max) return T(0);

		return (one_minus_cos > T(0))
			   ? T(1) / (T(2) * pi * one_minus_cos)
			   : T(0);
	}
};

