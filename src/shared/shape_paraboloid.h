#pragma once
// shape_paraboloid.h -- part of shapes.h (which includes it): ParaboloidShape<T>.

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
// ParaboloidShape<T>
// ===========================================================================
//
// Paraboloid of revolution: z = k*(x^2+y^2), k = zmax/radius^2, clipped to
// z in [z_min, z_max] (the underlying paraboloid's own apex, r=0, is always
// at z=0 - z_min>0 clips off the tip without changing k, matching pbrt-v4's
// "radius is measured at zmax" convention). phi_max is the azimuthal sweep
// in radians.
//
// sample()/sample_from()/pdf_from() below support a real AreaLightSource
// the identical way ConeShape's own do (that shape's own comment) - z drawn
// uniformly over [z_min,z_max], each sample's pdf is its own real,
// position-dependent area density.
//
// Reference: pbrt-v4 src/pbrt/shapes.h/.cpp  Paraboloid class
// ===========================================================================

template<typename T>
struct ParaboloidShape {
	T radius;    // radius at z = z_max
	T z_min, z_max;
	T phi_max;   // azimuthal extent in radians

	CPU_GPU static ParaboloidShape make(T radius, T z_min, T z_max, T phi_max_rad) {
		return ParaboloidShape{radius, z_min, z_max, phi_max_rad};
	}

	// Exact surface area of the paraboloid z = k r^2 (k = zmax/radius^2) between z_min and z_max, over phi_max: the surface of revolution of r(z) = sqrt(z/k),
	//   Area = phi_max * integral sqrt(z/k + 1/(4k^2)) dz = (2 phi_max / (3 k^2)) * ((k zmax + 1/4)^(3/2) - (k zmin + 1/4)^(3/2)).
	// This is pbrt-v4's Paraboloid::Area (its k is 4x ours: (r^4 phiMax / (12 zmax^2)) * ((k zmax + 1)^1.5 - (k zmin + 1)^1.5)). It used to use the
	// coefficient r^4 phi_max / (6 k^2) where 2 phi_max / (3 k^2) belongs - r^4/4 of the true area (a unit paraboloid z = r^2, z in [0,1], full turn, is 5.330, not
	// 1.333). Nothing depended on it until the paraboloid became an emitter whose emission density is 1 / area.
	CPU_GPU T area() const {
		if (radius == T(0)) return T(0);
		T radius2 = radius*radius;
		T k = z_max / radius2;
		T lo = k*z_min + T(0.25), hi = k*z_max + T(0.25);
		lo = (lo < T(0)) ? T(0) : lo;
		hi = (hi < T(0)) ? T(0) : hi;
		return (T(2)*phi_max / (T(3)*k*k)) *
			   (std::pow(hi, T(1.5)) - std::pow(lo, T(1.5)));
	}

	CPU_GPU std::optional<ShapeHit<T>>
	intersect(T rox, T roy, T roz,
	          T rdx, T rdy, T rdz,
	          T t_min, T t_max) const
	{
		using namespace ::shapes_detail;
		const T pi = T(3.14159265358979323846);
		if (radius == T(0)) return {};
		const T k = z_max / (radius*radius);

		double ox=(double)rox, oy=(double)roy, oz=(double)roz;
		double dx=(double)rdx, dy=(double)rdy, dz=(double)rdz;
		double dk=(double)k;

		double a = dk*(dx*dx + dy*dy);
		double b = 2.0*dk*(ox*dx + oy*dy) - dz;
		double c = dk*(ox*ox + oy*oy) - oz;

		// a==0 only when dx=dy=0 (a ray exactly ON the symmetry axis) - the
		// quadratic degenerates to linear and this shape doesn't special-
		// case it (matches CylinderShape::intersect's own identical "a==0 ->
		// no hit" simplification just above), even though the axis DOES
		// carry one real surface point here (the apex) unlike a cylinder's
		// own axis. Accepted rather than fixed: a ray landing exactly on the
		// axis is a measure-zero event in a real Monte Carlo renderer - see
		// ShapesParaboloid.IntersectMissesExactlyOnAxis (shapes_tests.cpp)
		// for the documented behavior this produces.
		if (a == 0.0) return {};
		double discrim = b*b - 4.0*a*c;
		if (discrim < 0.0) return {};
		double sqrt_disc = std::sqrt(discrim);
		double q = (b < 0.0) ? -0.5*(b - sqrt_disc) : -0.5*(b + sqrt_disc);
		T t0 = (T)(q/a), t1 = (T)(c/q);
		if (t0 > t1) { T tmp = t0; t0 = t1; t1 = tmp; }
		if (t0 > t_max || t1 < t_min) return {};

		auto check = [&](T t) -> std::optional<ShapeHit<T>> {
			if (t < t_min || t > t_max) return {};
			T hx = rox + t*rdx, hy = roy + t*rdy, hz = roz + t*rdz;
			if (hz < z_min || hz > z_max) return {};
			T phi = std::atan2(hy, hx);
			if (phi < T(0)) phi += T(2)*pi;
			if (phi > phi_max) return {};
			// Outward normal ∝ (x, y, -1/(2k)) - dpdu×dpdv derivation, same
			// method as ConeShape's own (see that shape's header comment).
			T nx = hx, ny = hy, nz = -T(1) / (T(2)*k);
			T nlen = safe_sqrt(nx*nx + ny*ny + nz*nz);
			if (nlen > T(0)) { nx /= nlen; ny /= nlen; nz /= nlen; }
			T u = phi / phi_max, v = (hz - z_min) / (z_max - z_min);
			return ShapeHit<T>{t, nx, ny, nz, u, v};
		};
		auto hit = check(t0);
		if (hit) return hit;
		return check(t1);
	}

	// -----------------------------------------------------------------------
	// Area-uniform-ISH surface sample: z drawn uniformly over [z_min,z_max] -
	// see ConeShape::sample()'s own comment for the identical "position-
	// dependent ss.pdf, not a shape-wide constant" technique and rationale.
	// -----------------------------------------------------------------------
	CPU_GPU ShapeSample<T> sample(T u0, T u1) const {
		using namespace ::shapes_detail;
		if (radius == T(0)) return ShapeSample<T>{0,0,0, 0,0,1, 0,0, T(0)};
		const T k = z_max / (radius*radius);
		T z = z_min + u0*(z_max - z_min);
		T phi = u1 * phi_max;
		T r = safe_sqrt(z / k);
		T lx = r*std::cos(phi), ly = r*std::sin(phi);
		T nx = lx, ny = ly, nz = -T(1) / (T(2)*k);
		T nlen = safe_sqrt(nx*nx + ny*ny + nz*nz);
		if (nlen > T(0)) { nx /= nlen; ny /= nlen; nz /= nlen; }
		// dA/dz (this shape's own header comment - the exact same
		// integrand its own closed-form area() already integrates) times
		// the uniform-in-z sampling density (1/(z_max-z_min)) inverted
		// gives this SPECIFIC sample's own area pdf.
		T dAdz = (phi_max / k) * safe_sqrt(k*z + T(0.25));
		T pdfArea = (dAdz > T(0)) ? (T(1) / (z_max - z_min)) / dAdz : T(0);
		T uv = phi / phi_max, vv = (z - z_min) / (z_max - z_min);
		return ShapeSample<T>{lx, ly, z, nx, ny, nz, uv, vv, pdfArea};
	}

	// -----------------------------------------------------------------------
	// Solid-angle sample from a shading point - see ConeShape::sample_from()'s
	// own comment (identical shape/rationale).
	// -----------------------------------------------------------------------
	CPU_GPU ShapeSample<T> sample_from(const SamplingContext<T>& ctx,
	                                    T u0, T u1) const {
		ShapeSample<T> ss = sample(u0, u1);
		ss.pdf = solid_angle_pdf_from_sample(ss.px, ss.py, ss.pz,
		                                      ss.nx, ss.ny, ss.nz, ctx, ss.pdf);
		return ss;
	}

	// -----------------------------------------------------------------------
	// Solid-angle PDF from a shading point - see ConeShape::pdf_from()'s own
	// comment (identical shape/rationale).
	// -----------------------------------------------------------------------
	CPU_GPU T pdf_from(const SamplingContext<T>& ctx,
	                    T wi_dx, T wi_dy, T wi_dz) const {
		// The area density sample() draws a point at height z with (see there), evaluated at a hit. (Defined before the using-directive below, and with qualified
		// names: MSVC reads a lambda inside a using-directive's scope as ambiguous in a TU that has `using namespace scene_doc;`.)
		const T k = z_max / (radius*radius);
		const auto pdfAreaAt = [&](const ShapeHit<T>& h) {
			T z = z_min + h.v * (z_max - z_min);
			T dAdz = (phi_max / k) * ::shapes_detail::safe_sqrt(k*z + T(0.25));
			return (dAdz > T(0)) ? (T(1) / (z_max - z_min)) / dAdz : T(0);
		};
		using namespace ::shapes_detail;
		if (radius == T(0)) return T(0);
		T wi_len=len3(wi_dx,wi_dy,wi_dz);
		if (wi_len==T(0)) return T(0);
		T wix=wi_dx/wi_len, wiy=wi_dy/wi_len, wiz=wi_dz/wi_len;
		auto hit=intersect(ctx.px,ctx.py,ctx.pz,wix,wiy,wiz,
		                   T(1e-4),std::numeric_limits<T>::max());
		if(!hit) return T(0);
		T pdf = solid_angle_pdf_from_hit(hit->nx, hit->ny, hit->nz,
		                                  wix, wiy, wiz, hit->t, pdfAreaAt(*hit));
		// Both crossings of a ray count, as in ConeShape::pdf_from() and CylinderShape::pdf_from() (see there): the sampler reaches the far side of the bowl as
		// readily as the near one, so the direction's density is their sum.
		auto hit2 = intersect(ctx.px,ctx.py,ctx.pz,wix,wiy,wiz,
		                      hit->t + T(1e-4), std::numeric_limits<T>::max());
		if (hit2)
			pdf += solid_angle_pdf_from_hit(hit2->nx, hit2->ny, hit2->nz,
			                                 wix, wiy, wiz, hit2->t, pdfAreaAt(*hit2));
		return pdf;
	}
};

