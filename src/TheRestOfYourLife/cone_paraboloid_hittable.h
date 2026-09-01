#pragma once
// cone_paraboloid_hittable.h -- pbrt-v4 Shape "cone" / "paraboloid", CPU
// backend.
//
// Both wrap the shared ConeShape<T>/ParaboloidShape<T> math in
// src/shared/shapes.h (mirrors DiskShape<T>/CylinderShape<T>'s own object-
// space-plus-unbaked-CTM technique - see disk_cylinder_hittable.h's file
// header comment for the full rationale on why these aren't baked to
// world-space parameters the way Sphere is).
//
// v1 scope, geometry-only: no random()/pdf_value() override (base class
// hittable's own defaults - pdf_value=0, random=an arbitrary unit vector -
// are exactly right for "this shape is never sampled as a light", matching
// pbrt_flatten.h's Cone/Paraboloid structs never carrying an areaLight
// field at all). No medium-boundary support either, for the identical
// reason bilinearmesh/trianglemesh don't have one.

#include "hittable.h"
#include "material.h"
#include "../shared/pbrt_scene.h"
#include "../shared/shapes.h"

#include "affine_transform_apply.h"

#include <cmath>
#include <memory>


class cone_hittable : public hittable {
  public:
	cone_hittable(double radius, double height, double phi_max,
				  const pbrt_scene::Matrix4& object_to_world, shared_ptr<material> mat)
		: shape_(ConeShape<double>::make(radius, height, phi_max)),
		  mat_(mat), o2w_(object_to_world)
	{
		valid_ = o2w_.inverseAffine(w2o_);
		if (!valid_) return;
		bbox_ = affine_transform::transformed_bbox(o2w_,
			-radius, radius, -radius, radius, 0.0, height);
	}

	bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
		if (!valid_) return false;
		using namespace affine_transform;
		const point3 ro = apply_point(w2o_, r.origin());
		const vec3   rd = apply_vector(w2o_, r.direction());   // NOT normalised - see file comment

		const auto h = shape_.intersect(ro.x(), ro.y(), ro.z(), rd.x(), rd.y(), rd.z(),
										 ray_t.min, ray_t.max);
		if (!h) return false;

		rec.t = h->t;
		const double hx = ro.x() + h->t * rd.x();
		const double hy = ro.y() + h->t * rd.y();
		const double hz = ro.z() + h->t * rd.z();
		rec.p = apply_point(o2w_, point3(hx, hy, hz));
		rec.u = h->u;
		rec.v = h->v;

		// pbrt-v4 Cone::Intersect: dpdu tangent to increasing phi, dpdv along
		// the slant from base to apex. p(phi,z) = (r(z)cos(phi), r(z)sin(phi), z),
		// r(z) = radius - k*z (k=radius/height); v = z/height, so
		// dpdv = dp/dz * dz/dv = (-k*cos(phi), -k*sin(phi), 1) * height
		//      = (-radius*hx/r(z), -radius*hy/r(z), height) - falls back to
		// the pure-axial direction at the apex (r(z)->0) where phi is
		// undefined, matching CylinderShape's own "hitRad>0" degenerate guard.
		vec3 dpdu_obj(-shape_.phi_max * hy, shape_.phi_max * hx, 0.0);
		vec3 dpdv_obj(0.0, 0.0, shape_.height);
		{
			const double r_local = shape_.radius - (shape_.radius/shape_.height) * hz;
			if (r_local > 1e-12) {
				dpdv_obj = vec3(-shape_.radius * hx / r_local, -shape_.radius * hy / r_local, shape_.height);
			}
		}
		rec.dpdu = apply_vector(o2w_, dpdu_obj);
		rec.dpdv = apply_vector(o2w_, dpdv_obj);

		vec3 n = apply_normal(w2o_, vec3(h->nx, h->ny, h->nz));
		const double len = n.length();
		if (len > 0) n = n / len;
		rec.set_face_normal(r, n);
		rec.mat = mat_;
		return true;
	}

	aabb bounding_box() const override { return bbox_; }

	shared_ptr<material> get_material() const { return mat_; }

  private:
	ConeShape<double> shape_;
	shared_ptr<material> mat_;
	pbrt_scene::Matrix4 o2w_, w2o_;
	bool valid_ = false;
	aabb bbox_;
};


class paraboloid_hittable : public hittable {
  public:
	paraboloid_hittable(double radius, double z_min, double z_max, double phi_max,
						 const pbrt_scene::Matrix4& object_to_world, shared_ptr<material> mat)
		: shape_(ParaboloidShape<double>::make(radius, z_min, z_max, phi_max)),
		  mat_(mat), o2w_(object_to_world)
	{
		valid_ = o2w_.inverseAffine(w2o_);
		if (!valid_) return;
		bbox_ = affine_transform::transformed_bbox(o2w_,
			-radius, radius, -radius, radius, z_min, z_max);
	}

	bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
		if (!valid_) return false;
		using namespace affine_transform;
		const point3 ro = apply_point(w2o_, r.origin());
		const vec3   rd = apply_vector(w2o_, r.direction());   // NOT normalised - see file comment

		const auto h = shape_.intersect(ro.x(), ro.y(), ro.z(), rd.x(), rd.y(), rd.z(),
										 ray_t.min, ray_t.max);
		if (!h) return false;

		rec.t = h->t;
		const double hx = ro.x() + h->t * rd.x();
		const double hy = ro.y() + h->t * rd.y();
		const double hz = ro.z() + h->t * rd.z();
		rec.p = apply_point(o2w_, point3(hx, hy, hz));
		rec.u = h->u;
		rec.v = h->v;

		// pbrt-v4 Paraboloid::Intersect: dpdu tangent to increasing phi, dpdv
		// along the meridian (increasing z/radius). p(u,v) = (r(z)*cos(phi),
		// r(z)*sin(phi), z), z = zmin + v*(zmax-zmin); dr/dz = 1/(2*k*r(z))
		// (see ParaboloidShape's own header comment) -> dpdv = dp/dz * dz/dv
		// = (dr/dz*cos(phi), dr/dz*sin(phi), 1) * (zmax-zmin), falling back
		// to the pure-axial direction at the apex (r(z)->0, phi undefined).
		vec3 dpdu_obj(-shape_.phi_max * hy, shape_.phi_max * hx, 0.0);
		const double dz_dv = shape_.z_max - shape_.z_min;
		vec3 dpdv_obj(0.0, 0.0, dz_dv);
		{
			const double k = shape_.z_max / (shape_.radius * shape_.radius);
			const double r_local = std::sqrt(hx*hx + hy*hy);
			if (r_local > 1e-12 && k != 0.0) {
				const double dr_dz = 1.0 / (2.0 * k * r_local);
				dpdv_obj = vec3(hx / r_local * dr_dz, hy / r_local * dr_dz, 1.0) * dz_dv;
			}
		}
		rec.dpdu = apply_vector(o2w_, dpdu_obj);
		rec.dpdv = apply_vector(o2w_, dpdv_obj);

		vec3 n = apply_normal(w2o_, vec3(h->nx, h->ny, h->nz));
		const double len = n.length();
		if (len > 0) n = n / len;
		rec.set_face_normal(r, n);
		rec.mat = mat_;
		return true;
	}

	aabb bounding_box() const override { return bbox_; }

	shared_ptr<material> get_material() const { return mat_; }

  private:
	ParaboloidShape<double> shape_;
	shared_ptr<material> mat_;
	pbrt_scene::Matrix4 o2w_, w2o_;
	bool valid_ = false;
	aabb bbox_;
};
