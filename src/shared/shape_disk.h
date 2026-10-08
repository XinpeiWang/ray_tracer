#pragma once
// shape_disk.h -- part of shapes.h (which includes it): DiskShape<T>.

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
		// pbrt-v4 pError for disk: gamma(3) * |pHit.x|, gamma(3) * |pHit.y|, 0
		// (z = height plane is exact; x,y accumulate rounding from t*rdx/rdy)
		// Reference: Cylinder::InteractionFromIntersection, shapes.h -- gamma(3)*Abs(x,y,0)
		T g3 = shapes_detail::gamma_fp<T>(3);
		ShapeHit<T> hit;
		hit.t  = t_hit;
		hit.nx = T(0); hit.ny = T(0); hit.nz = T(1);
		hit.u  = u_coord; hit.v = v_coord;
		hit.ex = std::abs(hx) * g3;
		hit.ey = std::abs(hy) * g3;
		hit.ez = T(0);   // z = height is exact (plane equation)
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
		ss.pdf = solid_angle_pdf_from_sample(ss.px, ss.py, ss.pz,
		                                      ss.nx, ss.ny, ss.nz, ctx, ss.pdf);
		return ss;
	}

	CPU_GPU T pdf_from(const SamplingContext<T>& ctx,
					   T wi_dx, T wi_dy, T wi_dz) const {
		// Intersect the wi ray with the disk and compute Jacobian
		T wi_len = shapes_detail::len3(wi_dx, wi_dy, wi_dz);
		if (wi_len == T(0)) return T(0);
		T wix = wi_dx/wi_len, wiy = wi_dy/wi_len, wiz = wi_dz/wi_len;
		// Ray from ctx in direction wi
		auto hit = intersect(ctx.px, ctx.py, ctx.pz, wix, wiy, wiz,
							 T(1e-4), std::numeric_limits<T>::max());
		if (!hit) return T(0);
		return solid_angle_pdf_from_hit(hit->nx, hit->ny, hit->nz,
		                                 wix, wiy, wiz, hit->t, pdf_area());
	}
};

