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
// Real AreaLightSource support (random()/pdf_value() below, solid-angle NEE
// sampling via ConeShape/ParaboloidShape's own sample_from()/pdf_from() -
// see those functions' own comments in shapes.h) and real MediumInterface
// support (via constant_medium, wired the identical way Sphere/Disk/
// Cylinder already are - pbrt_cpu_builder.h's addMediumIfPresent()) -
// mirrors disk_hittable's own random()/pdf_value() pattern exactly, minus
// the motion-blur MotionState indirection (Cone::xform/Paraboloid::xform
// don't carry object motion blur, unlike Disk/Cylinder - a static o2w_/w2o_
// pair is always valid to sample against here).

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

	// Solid-angle NEE sampling - see this file's own header comment.
	vec3 random(const point3& origin) const override {
		if (!valid_) return vec3(1, 0, 0);
		using namespace affine_transform;
		const point3 ctx_obj = apply_point(w2o_, origin);
		const SamplingContext<double> ctx{ctx_obj.x(), ctx_obj.y(), ctx_obj.z(), 0, 0, 0};
		const auto ss = shape_.sample_from(ctx, random_double(), random_double());
		const point3 p_world = apply_point(o2w_, point3(ss.px, ss.py, ss.pz));
		return p_world - origin;
	}

	double pdf_value(const point3& origin, const vec3& direction) const override {
		if (!valid_) return 0.0;
		using namespace affine_transform;
		const point3 ctx_obj = apply_point(w2o_, origin);
		const vec3 dir_obj = apply_vector(w2o_, direction);
		const SamplingContext<double> ctx{ctx_obj.x(), ctx_obj.y(), ctx_obj.z(), 0, 0, 0};
		return shape_.pdf_from(ctx, dir_obj.x(), dir_obj.y(), dir_obj.z());
	}

	// Real entry/exit interval for a MediumInterface-wrapped cone - see
	// hittable::volume_bounds()'s own comment for why this is needed at
	// all (the generic constant_medium two-hit() fallback is wrong for
	// this shape's open base). Same "infinite quadric surface intersected
	// with a z-slab" technique as cylinder_hittable::volume_bounds()
	// (disk_cylinder_hittable.h), just with the cone's own a/b/c quadratic
	// coefficients - identical to intersect()'s own (this shape's
	// unclipped-by-z/phi roots), and the z-slab is [0, height] rather than
	// an independent [z_min, z_max] since a cone's base is always at z=0.
	bool supports_volume_bounds() const override { return true; }

	bool volume_bounds(const ray& r, double& out_t0, double& out_t1) const override {
		if (!valid_ || shape_.height == 0.0) return false;
		using namespace affine_transform;
		const point3 ro = apply_point(w2o_, r.origin());
		const vec3   rd = apply_vector(w2o_, r.direction());   // NOT normalised - see file comment

		const double k = shape_.radius / shape_.height;
		const double a = rd.x()*rd.x() + rd.y()*rd.y() - k*k*rd.z()*rd.z();
		const double b = 2.0*(ro.x()*rd.x() + ro.y()*rd.y()) - 2.0*k*k*ro.z()*rd.z()
		               + 2.0*shape_.radius*k*rd.z();
		const double c = ro.x()*ro.x() + ro.y()*ro.y() - shape_.radius*shape_.radius
		               + 2.0*shape_.radius*k*ro.z() - k*k*ro.z()*ro.z();

		double tube_t0 = -infinity, tube_t1 = infinity;
		bool hasTube;
		if (a == 0.0) {
			// Matches ConeShape::intersect()'s own "a==0 -> no hit" precedent
			// (see that function's comment) - a ray exactly parallel to a
			// generator line's slope, rare enough to accept as a clean miss.
			hasTube = false;
		} else {
			const double discrim = b*b - 4.0*a*c;
			if (discrim < 0.0) {
				hasTube = false;
			} else {
				const double sqrt_disc = std::sqrt(discrim);
				const double q = (b < 0.0) ? -0.5*(b - sqrt_disc) : -0.5*(b + sqrt_disc);
				tube_t0 = q / a;
				tube_t1 = c / q;
				if (tube_t0 > tube_t1) std::swap(tube_t0, tube_t1);
				hasTube = true;
			}
		}

		double z_t0 = -infinity, z_t1 = infinity;
		bool hasZSlab = true;
		if (rd.z() == 0.0) {
			hasZSlab = (ro.z() >= 0.0 && ro.z() <= shape_.height);
		} else {
			const double za = (0.0 - ro.z()) / rd.z();
			const double zb = (shape_.height - ro.z()) / rd.z();
			z_t0 = std::min(za, zb);
			z_t1 = std::max(za, zb);
		}

		out_t0 = std::max(tube_t0, z_t0);
		out_t1 = std::min(tube_t1, z_t1);
		return hasTube && hasZSlab && out_t0 < out_t1;
	}

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

	// Solid-angle NEE sampling - see this file's own header comment.
	vec3 random(const point3& origin) const override {
		if (!valid_) return vec3(1, 0, 0);
		using namespace affine_transform;
		const point3 ctx_obj = apply_point(w2o_, origin);
		const SamplingContext<double> ctx{ctx_obj.x(), ctx_obj.y(), ctx_obj.z(), 0, 0, 0};
		const auto ss = shape_.sample_from(ctx, random_double(), random_double());
		const point3 p_world = apply_point(o2w_, point3(ss.px, ss.py, ss.pz));
		return p_world - origin;
	}

	double pdf_value(const point3& origin, const vec3& direction) const override {
		if (!valid_) return 0.0;
		using namespace affine_transform;
		const point3 ctx_obj = apply_point(w2o_, origin);
		const vec3 dir_obj = apply_vector(w2o_, direction);
		const SamplingContext<double> ctx{ctx_obj.x(), ctx_obj.y(), ctx_obj.z(), 0, 0, 0};
		return shape_.pdf_from(ctx, dir_obj.x(), dir_obj.y(), dir_obj.z());
	}

	// Real entry/exit interval for a MediumInterface-wrapped paraboloid -
	// see cone_hittable::volume_bounds()'s own comment (identical
	// "infinite quadric intersected with a z-slab" rationale and
	// technique, cylinder_hittable::volume_bounds() in disk_cylinder_
	// hittable.h). a/b/c mirror intersect()'s own unclipped-by-z/phi
	// quadratic coefficients; the z-slab is this shape's own independent
	// [z_min, z_max] (unlike a cone, a paraboloid's near end need not be
	// z=0/the apex).
	bool supports_volume_bounds() const override { return true; }

	bool volume_bounds(const ray& r, double& out_t0, double& out_t1) const override {
		if (!valid_ || shape_.radius == 0.0) return false;
		using namespace affine_transform;
		const point3 ro = apply_point(w2o_, r.origin());
		const vec3   rd = apply_vector(w2o_, r.direction());   // NOT normalised - see file comment

		const double k = shape_.z_max / (shape_.radius*shape_.radius);
		const double a = k*(rd.x()*rd.x() + rd.y()*rd.y());
		const double b = 2.0*k*(ro.x()*rd.x() + ro.y()*rd.y()) - rd.z();
		const double c = k*(ro.x()*ro.x() + ro.y()*ro.y()) - ro.z();

		double tube_t0 = -infinity, tube_t1 = infinity;
		bool hasTube;
		if (a == 0.0) {
			// Matches ParaboloidShape::intersect()'s own "a==0 -> no hit"
			// precedent (see that function's comment) - a ray exactly on
			// the symmetry axis, rare enough to accept as a clean miss.
			hasTube = false;
		} else {
			const double discrim = b*b - 4.0*a*c;
			if (discrim < 0.0) {
				hasTube = false;
			} else {
				const double sqrt_disc = std::sqrt(discrim);
				const double q = (b < 0.0) ? -0.5*(b - sqrt_disc) : -0.5*(b + sqrt_disc);
				tube_t0 = q / a;
				tube_t1 = c / q;
				if (tube_t0 > tube_t1) std::swap(tube_t0, tube_t1);
				hasTube = true;
			}
		}

		double z_t0 = -infinity, z_t1 = infinity;
		bool hasZSlab = true;
		if (rd.z() == 0.0) {
			hasZSlab = (ro.z() >= shape_.z_min && ro.z() <= shape_.z_max);
		} else {
			const double za = (shape_.z_min - ro.z()) / rd.z();
			const double zb = (shape_.z_max - ro.z()) / rd.z();
			z_t0 = std::min(za, zb);
			z_t1 = std::max(za, zb);
		}

		out_t0 = std::max(tube_t0, z_t0);
		out_t1 = std::min(tube_t1, z_t1);
		return hasTube && hasZSlab && out_t0 < out_t1;
	}

	shared_ptr<material> get_material() const { return mat_; }

  private:
	ParaboloidShape<double> shape_;
	shared_ptr<material> mat_;
	pbrt_scene::Matrix4 o2w_, w2o_;
	bool valid_ = false;
	aabb bbox_;
};
