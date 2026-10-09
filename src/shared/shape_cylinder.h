#pragma once
// shape_cylinder.h -- part of shapes.h (which includes it): CylinderShape<T>.

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
// CylinderShape<T>
// ===========================================================================
//
// Cylinder of radius `radius` along the Z axis centered at (cx, cy, *).
// z_min and z_max are world-space Z extents; phi_max is the azimuthal sweep
// in radians (2*pi for a full cylinder).
//
// Matches pbrt-v4 Cylinder semantics:
//   Area  = (z_max - z_min) * radius * phi_max
//   Normal = outward radial, nz = 0 (no end caps)
//   UV     = (phi/phi_max, (z-z_min)/(z_max-z_min))
//
// Reference: pbrt-v4 src/pbrt/shapes.h  Cylinder class
// ===========================================================================

template<typename T>
struct CylinderShape {
	T cx, cy;       // world-space axis center (axis runs parallel to Z)
	T z_min, z_max; // world-space Z extent
	T radius;       // cylinder radius
	T phi_max;      // azimuthal extent in radians (default 2*pi)

	// make(cx, cy, z_min, z_max, radius) -- full cylinder
	CPU_GPU static CylinderShape make(T cx, T cy, T z_min, T z_max, T radius) {
		const T pi2 = T(2) * T(3.14159265358979323846);
		return CylinderShape{cx, cy, z_min, z_max, radius, pi2};
	}

	// make_partial -- partial azimuthal sweep
	CPU_GPU static CylinderShape make_partial(T cx, T cy, T z_min, T z_max,
	                                           T radius, T phi_max_rad) {
		return CylinderShape{cx, cy, z_min, z_max, radius, phi_max_rad};
	}

	// -----------------------------------------------------------------------
	// Area
	// Reference: pbrt-v4 Cylinder::Area = (zMax-zMin)*radius*phiMax
	// -----------------------------------------------------------------------
	CPU_GPU T area() const { return (z_max - z_min) * radius * phi_max; }
	CPU_GPU T pdf_area() const { return T(1) / area(); }

	// -----------------------------------------------------------------------
	// Intersection -- quadric in XY plane, world-space Z clip
	// Reference: pbrt-v4 Cylinder::BasicIntersect (stable discriminant)
	// -----------------------------------------------------------------------
	CPU_GPU std::optional<ShapeHit<T>>
	intersect(T rox, T roy, T roz,
	          T rdx, T rdy, T rdz,
	          T t_min, T t_max) const
	{
		using namespace ::shapes_detail;
		const T pi = T(3.14159265358979323846);
		T ox = rox-cx, oy = roy-cy, oz = roz;
		T dx = rdx, dy = rdy, dz = rdz;
		double da=(double)dx, db=(double)dy, oa=(double)ox, ob=(double)oy;
		double a=da*da+db*db, b=2.0*(oa*da+ob*db);
		double c=oa*oa+ob*ob-(double)radius*(double)radius;
		if (a==0.0) return {};
		double f=b/(2.0*a), vx=oa-f*da, vy=ob-f*db;
		double len_v=std::sqrt(vx*vx+vy*vy);
		double discrim=4.0*a*((double)radius+len_v)*((double)radius-len_v);
		if (discrim<0.0) return {};
		double sqrt_disc=std::sqrt(discrim);
		double q=(b<0.0)?-0.5*(b-sqrt_disc):-0.5*(b+sqrt_disc);
		T t0=(T)(q/a), t1=(T)(c/q);
		if (t0>t1){T tmp=t0;t0=t1;t1=tmp;}
		if (t0>t_max||t1<t_min) return {};
		auto check=[&](T t)->std::optional<ShapeHit<T>>{
			if(t<t_min||t>t_max) return {};
			T hx=ox+t*dx,hy=oy+t*dy,hz=oz+t*dz;
			if(hz<z_min||hz>z_max) return {};
			T phi=std::atan2(hy,hx);
			if(phi<T(0)) phi+=T(2)*pi;
			if(phi>phi_max) return {};
			T hitRad=safe_sqrt(hx*hx+hy*hy);
			if(hitRad>T(0)){hx*=radius/hitRad;hy*=radius/hitRad;}
			T u=phi/phi_max, v=(hz-z_min)/(z_max-z_min);
			return ShapeHit<T>{t,hx/radius,hy/radius,T(0),u,v};
		};
		auto hit=check(t0);
		if(hit) return hit;
		return check(t1);
	}

	// -----------------------------------------------------------------------
	// Area-uniform surface sample
	// Reference: pbrt-v4 Cylinder::Sample(Point2f u)
	// -----------------------------------------------------------------------
	CPU_GPU ShapeSample<T> sample(T u0, T u1) const {
		using namespace ::shapes_detail;
		T z=z_min+u0*(z_max-z_min), phi=u1*phi_max;
		T lx=radius*std::cos(phi), ly=radius*std::sin(phi);
		T hitRad=safe_sqrt(lx*lx+ly*ly);
		if(hitRad>T(0)){lx*=radius/hitRad;ly*=radius/hitRad;}
		T nx=lx/radius, ny=ly/radius, nz=T(0);
		T uv=phi/phi_max, vv=(z-z_min)/(z_max-z_min);
		return ShapeSample<T>{cx+lx,cy+ly,z,nx,ny,nz,uv,vv,pdf_area()};
	}

	// -----------------------------------------------------------------------
	// Solid-angle sample from a shading point (area sample + Jacobian)
	// Reference: pbrt-v4 Cylinder::Sample(ShapeSampleContext, Point2f)
	// -----------------------------------------------------------------------
	CPU_GPU ShapeSample<T> sample_from(const SamplingContext<T>& ctx,
	                                    T u0, T u1) const {
		ShapeSample<T> ss=sample(u0,u1);
		ss.pdf = solid_angle_pdf_from_sample(ss.px, ss.py, ss.pz,
		                                      ss.nx, ss.ny, ss.nz, ctx, ss.pdf);
		return ss;
	}

	// -----------------------------------------------------------------------
	// Solid-angle PDF from a shading point (intersect + Jacobian)
	// Reference: pbrt-v4 Cylinder::PDF(ShapeSampleContext, Vector3f)
	// -----------------------------------------------------------------------
	CPU_GPU T pdf_from(const SamplingContext<T>& ctx,
	                    T wi_dx, T wi_dy, T wi_dz) const {
		using namespace ::shapes_detail;
		T wi_len=len3(wi_dx,wi_dy,wi_dz);
		if(wi_len==T(0)) return T(0);
		T wix=wi_dx/wi_len, wiy=wi_dy/wi_len, wiz=wi_dz/wi_len;
		auto hit=intersect(ctx.px,ctx.py,ctx.pz,wix,wiy,wiz,
		                   T(1e-4),std::numeric_limits<T>::max());
		if(!hit) return T(0);
		T pdf=solid_angle_pdf_from_hit(hit->nx, hit->ny, hit->nz,
		                                wix, wiy, wiz, hit->t, pdf_area());
		// A ray can cross this open tube twice. An integrator that samples a DIRECTION toward the shape and then
		// credits whatever emitter the ray hits first (this renderer's NEE, via hittable::random()/pdf_value()) is
		// unbiased only if that direction's density counts BOTH crossings: the sampler lands on the far wall as
		// readily as on the near one, and pbrt's first-hit-only PDF() halves the denominator exactly where the near
		// wall hides the far one. Next to a cylinder light that rendered the floor up to 3x too bright against
		// the same light built from quads.
		auto hit2=intersect(ctx.px,ctx.py,ctx.pz,wix,wiy,wiz,
		                    hit->t+T(1e-4),std::numeric_limits<T>::max());
		if(hit2)
			pdf+=solid_angle_pdf_from_hit(hit2->nx, hit2->ny, hit2->nz,
			                               wix, wiy, wiz, hit2->t, pdf_area());
		return pdf;
	}
};

