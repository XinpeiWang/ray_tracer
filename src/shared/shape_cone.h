#pragma once
// shape_cone.h -- part of shapes.h (which includes it): ConeShape<T>.

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
// ConeShape<T>
// ===========================================================================
//
// Cone with base radius `radius` at z=0, narrowing linearly to a point (the
// apex) at z=height. phi_max is the azimuthal sweep in radians.
//
// Implicit surface: x^2 + y^2 - (radius - k*z)^2 = 0, k = radius/height,
// for z in [0, height] - matches pbrt-v4 Cone semantics exactly.
//
// sample()/sample_from()/pdf_from() below support a real AreaLightSource
// (pbrt_flatten.h's Cone struct comment) via a deliberately simpler-than-
// pbrt-v4 technique: z is drawn UNIFORMLY over [0,height] (not area-
// uniform - the lateral surface's true density is linear in z, heaviest at
// the wide base) but each sample's own pdf is the REAL, position-dependent
// area density it was actually drawn from, so the estimator stays unbiased
// even though it isn't pbrt-v4's own variance-optimal closed-form inverse
// CDF - see sample()'s own comment.
//
// Reference: pbrt-v4 src/pbrt/shapes.h/.cpp  Cone class
// ===========================================================================

template<typename T>
struct ConeShape {
	T radius;    // base radius, at z=0
	T height;    // apex is at z=height
	T phi_max;   // azimuthal extent in radians

	CPU_GPU static ConeShape make(T radius, T height, T phi_max_rad) {
		return ConeShape{radius, height, phi_max_rad};
	}

	// pbrt-v4 Cone::Area = radius * sqrt(height*height + radius*radius) * phiMax / 2
	CPU_GPU T area() const {
		return radius * std::sqrt(height*height + radius*radius) * phi_max / T(2);
	}

	CPU_GPU std::optional<ShapeHit<T>>
	intersect(T rox, T roy, T roz,
	          T rdx, T rdy, T rdz,
	          T t_min, T t_max) const
	{
		using namespace shapes_detail;
		const T pi = T(3.14159265358979323846);
		if (height == T(0)) return {};
		const T k = radius / height;

		double ox=(double)rox, oy=(double)roy, oz=(double)roz;
		double dx=(double)rdx, dy=(double)rdy, dz=(double)rdz;
		double dk=(double)k, dr=(double)radius;

		double a = dx*dx + dy*dy - dk*dk*dz*dz;
		double b = 2.0*(ox*dx + oy*dy) - 2.0*dk*dk*oz*dz + 2.0*dr*dk*dz;
		double c = ox*ox + oy*oy - dr*dr + 2.0*dr*dk*oz - dk*dk*oz*oz;

		if (a == 0.0) {
			// Degenerate to a linear equation (ray exactly parallel to a
			// generator line's projection) - rare enough in practice that a
			// clean miss is an acceptable v1 answer, matching CylinderShape's
			// own "a==0 -> no hit" precedent just above.
			return {};
		}
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
			if (hz < T(0) || hz > height) return {};
			T phi = std::atan2(hy, hx);
			if (phi < T(0)) phi += T(2)*pi;
			if (phi > phi_max) return {};
			// Outward normal ∝ (x, y, k*(radius - k*z)) - see this shape's
			// own header comment for the dpdu×dpdv derivation confirming
			// this orientation (verified two independent ways, not guessed).
			T r_local = radius - k*hz;
			T nx = hx, ny = hy, nz = k*r_local;
			T nlen = safe_sqrt(nx*nx + ny*ny + nz*nz);
			if (nlen > T(0)) { nx /= nlen; ny /= nlen; nz /= nlen; }
			T u = phi / phi_max, v = hz / height;
			return ShapeHit<T>{t, nx, ny, nz, u, v};
		};
		auto hit = check(t0);
		if (hit) return hit;
		return check(t1);
	}

	// -----------------------------------------------------------------------
	// Area-uniform-ISH surface sample: z drawn uniformly over [0,height]
	// (NOT area-uniform - the lateral surface's true area density is
	// LINEAR in z, dA/dz = phi_max*r(z)*sqrt(1+k^2), heaviest at the wide
	// base and zero at the apex), but ss.pdf below is the REAL, position-
	// dependent area density this sample was actually drawn from - a
	// smaller, guaranteed-correct (if not variance-optimal) alternative to
	// deriving this shape's own closed-form inverse CDF. Reference: no
	// direct pbrt-v4 equivalent (pbrt-v4 does derive the exact inverse CDF
	// for Cone::Sample) - this is a deliberately simpler, still-unbiased
	// substitute for this loader's own v1 NEE support.
	// -----------------------------------------------------------------------
	CPU_GPU ShapeSample<T> sample(T u0, T u1) const {
		using namespace shapes_detail;
		// Mirrors intersect()'s own "height==0 -> return {}" guard just
		// above, and ParaboloidShape::sample()'s identical "radius==0"
		// guard below - without it, k=radius/height=inf and r_local below
		// becomes inf*0=NaN, propagating a NaN sample/pdf all the way out
		// through sample_from() (a code-review pass found this reachable
		// from an ordinary pbrt scene: `Shape "cone" "float height" [0]`
		// under an `AreaLightSource` has no clamp anywhere in the loader).
		if (height == T(0)) return ShapeSample<T>{0,0,0, 0,0,1, 0,0, T(0)};
		const T k = radius / height;
		T z = u0 * height;
		T phi = u1 * phi_max;
		T r_local = radius - k*z;
		T lx = r_local*std::cos(phi), ly = r_local*std::sin(phi);
		T nx = lx, ny = ly, nz = k*r_local;
		T nlen = safe_sqrt(nx*nx + ny*ny + nz*nz);
		if (nlen > T(0)) { nx /= nlen; ny /= nlen; nz /= nlen; }
		// dA/dz (this shape's own header comment) times the uniform-in-z
		// sampling density (1/height) inverted gives this SPECIFIC sample's
		// own area pdf - not a shape-wide constant.
		T dAdz = phi_max * r_local * std::sqrt(T(1) + k*k);
		T pdfArea = (dAdz > T(0)) ? (T(1) / height) / dAdz : T(0);
		T uv = phi / phi_max, vv = z / height;
		return ShapeSample<T>{lx, ly, z, nx, ny, nz, uv, vv, pdfArea};
	}

	// -----------------------------------------------------------------------
	// Solid-angle sample from a shading point (area sample + Jacobian) -
	// same shape as CylinderShape::sample_from, EXCEPT this reuses the
	// per-sample ss.pdf sample() already computed (a shape CONSTANT
	// pdf_area() doesn't exist here - this shape's area density genuinely
	// varies with z, unlike Cylinder's own uniform one).
	// -----------------------------------------------------------------------
	CPU_GPU ShapeSample<T> sample_from(const SamplingContext<T>& ctx,
	                                    T u0, T u1) const {
		ShapeSample<T> ss = sample(u0, u1);
		ss.pdf = solid_angle_pdf_from_sample(ss.px, ss.py, ss.pz,
		                                      ss.nx, ss.ny, ss.nz, ctx, ss.pdf);
		return ss;
	}

	// -----------------------------------------------------------------------
	// Solid-angle PDF from a shading point (intersect + recompute the SAME
	// per-point area density sample() would have drawn, evaluated at the
	// hit instead of a fresh random draw - matches sample_from()'s own
	// distribution exactly, required for MIS to be unbiased).
	// -----------------------------------------------------------------------
	CPU_GPU T pdf_from(const SamplingContext<T>& ctx,
	                    T wi_dx, T wi_dy, T wi_dz) const {
		using namespace shapes_detail;
		T wi_len=len3(wi_dx,wi_dy,wi_dz);
		if (wi_len==T(0)) return T(0);
		T wix=wi_dx/wi_len, wiy=wi_dy/wi_len, wiz=wi_dz/wi_len;
		auto hit=intersect(ctx.px,ctx.py,ctx.pz,wix,wiy,wiz,
		                   T(1e-4),std::numeric_limits<T>::max());
		if(!hit) return T(0);
		const T k = radius / height;
		// The area density sample() draws a point at height z with (see there), evaluated at a hit.
		const auto pdfAreaAt = [&](const ShapeHit<T>& h) {
			T z = h.v * height;
			T r_local = radius - k*z;
			T dAdz = phi_max * r_local * std::sqrt(T(1) + k*k);
			return (dAdz > T(0)) ? (T(1) / height) / dAdz : T(0);
		};
		T pdf = solid_angle_pdf_from_hit(hit->nx, hit->ny, hit->nz,
		                                  wix, wiy, wiz, hit->t, pdfAreaAt(*hit));
		// A ray can cross the cone twice (the near and the far side of its lateral surface). An integrator that samples a DIRECTION toward the shape and credits
		// whatever emitter the ray hits first (this renderer's NEE, via hittable::random()/pdf_value()) is unbiased only if the direction's density counts BOTH
		// crossings, because sample() lands on the far side as readily as on the near one - see CylinderShape::pdf_from(), which already does this. With the first
		// crossing alone a floor next to a cone light read 3.4x too bright (closed form 0.4255, path tracer 1.4549) and --sppm 3x.
		auto hit2 = intersect(ctx.px,ctx.py,ctx.pz,wix,wiy,wiz,
		                      hit->t + T(1e-4), std::numeric_limits<T>::max());
		if (hit2)
			pdf += solid_angle_pdf_from_hit(hit2->nx, hit2->ny, hit2->nz,
			                                 wix, wiy, wiz, hit2->t, pdfAreaAt(*hit2));
		return pdf;
	}
};

