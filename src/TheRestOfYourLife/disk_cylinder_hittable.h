#pragma once
// disk_cylinder_hittable.h -- pbrt-v4 Shape "disk" / "cylinder", CPU backend.
//
// Both wrap the shared, already-tested DiskShape<T>/CylinderShape<T> math in
// src/shared/shapes.h (built for this purpose but never instantiated by any
// hittable - see docs/PBRT_SUPPORT.md and this feature's own commit history).
// That math is object-space only (disk centered at the z-axis in the plane
// z=height; cylinder centered on the z-axis from z_min to z_max) - exactly
// pbrt-v4's own convention, where the scene's CTM does the real positioning.
//
// Rather than baking the CTM into world-space parameters (the sphere's own
// approach - see pbrt_flatten.h's Sphere handling), both classes here carry
// the object-to-world transform directly and intersect by carrying the RAY
// into object space, matching transform_instance.h's technique exactly (see
// that file's own header comment for why: the ray moves, the geometry
// doesn't, and t survives the round trip unmodified because the direction is
// carried across without renormalising). This gives EXACT results under any
// affine transform including rotation, unlike the sphere's "bake to a single
// world-space radius, warn on anisotropic scale" approximation - a disk or
// cylinder is not rotation-invariant, so that approximation would be wrong
// far more often than it is for a sphere.
//
// One inherited caveat: solid-angle NEE sampling (random()/pdf_value()) does
// its geometry in OBJECT space and returns the result as if that were also
// the world-space solid angle. That equality holds exactly for a similarity
// transform (translation + rotation + UNIFORM scale - the dominant case in
// practice) and is only approximate under non-uniform scale or shear, for
// exactly the reason transform_instance.h's own header comment gives for why
// normals need the inverse transpose: solid angle is not preserved by an
// arbitrary affine map. This mirrors the sphere's own accepted "approximate
// under anisotropic scale" precedent rather than introducing new machinery
// to detect and warn about it.

#include "hittable.h"
#include "material.h"
#include "../shared/pbrt_scene.h"
#include "../shared/shapes.h"

#include <cmath>
#include <memory>

namespace disk_cylinder_detail {

inline point3 apply_point(const pbrt_scene::Matrix4& m, const point3& p) {
	return point3(
		m.m[0] * p.x() + m.m[1] * p.y() + m.m[2]  * p.z() + m.m[3],
		m.m[4] * p.x() + m.m[5] * p.y() + m.m[6]  * p.z() + m.m[7],
		m.m[8] * p.x() + m.m[9] * p.y() + m.m[10] * p.z() + m.m[11]);
}

// A direction: the translation column is deliberately not applied. Used for
// both ray directions and tangent vectors (dpdu/dpdv transform the same way
// a direction does - matches transform_instance.h's own apply_vector, which
// this is a direct copy of, kept local rather than shared to avoid coupling
// this file to transform_instance.h's own evolution).
inline vec3 apply_vector(const pbrt_scene::Matrix4& m, const vec3& v) {
	return vec3(
		m.m[0] * v.x() + m.m[1] * v.y() + m.m[2]  * v.z(),
		m.m[4] * v.x() + m.m[5] * v.y() + m.m[6]  * v.z(),
		m.m[8] * v.x() + m.m[9] * v.y() + m.m[10] * v.z());
}

// Object -> world for a normal is the TRANSPOSE of world -> object (same
// reasoning as transform_instance.h's apply_normal), so this reads w2o
// column-wise rather than taking o2w row-wise.
inline vec3 apply_normal(const pbrt_scene::Matrix4& w2o, const vec3& n) {
	return vec3(
		w2o.m[0] * n.x() + w2o.m[4] * n.y() + w2o.m[8]  * n.z(),
		w2o.m[1] * n.x() + w2o.m[5] * n.y() + w2o.m[9]  * n.z(),
		w2o.m[2] * n.x() + w2o.m[6] * n.y() + w2o.m[10] * n.z());
}

// World-space AABB of an object-space box, transformed corner-by-corner (not
// the transform of the box's own min/max) - identical reasoning to
// transform_instance.h's constructor: those differ the moment a rotation is
// involved, and transforming just the corners of the untransformed box
// produces a box that clips the geometry it's supposed to contain.
inline aabb transformed_bbox(const pbrt_scene::Matrix4& o2w,
							  double xlo, double xhi, double ylo, double yhi,
							  double zlo, double zhi) {
	point3 lo(0, 0, 0), hi(0, 0, 0);
	bool first = true;
	for (int corner = 0; corner < 8; ++corner) {
		const double x = (corner & 1) ? xhi : xlo;
		const double y = (corner & 2) ? yhi : ylo;
		const double z = (corner & 4) ? zhi : zlo;
		const point3 p = apply_point(o2w, point3(x, y, z));
		if (first) { lo = hi = p; first = false; continue; }
		lo = point3(std::fmin(lo.x(), p.x()), std::fmin(lo.y(), p.y()), std::fmin(lo.z(), p.z()));
		hi = point3(std::fmax(hi.x(), p.x()), std::fmax(hi.y(), p.y()), std::fmax(hi.z(), p.z()));
	}
	return aabb(lo, hi);
}

} // namespace disk_cylinder_detail


class disk_hittable : public hittable {
  public:
	disk_hittable(double outer_r, double inner_r, double height, double phi_max,
				  const pbrt_scene::Matrix4& object_to_world, shared_ptr<material> mat)
		: shape_(DiskShape<double>::make_annular(0.0, 0.0, height, outer_r, inner_r, phi_max)),
		  mat_(mat), o2w_(object_to_world)
	{
		valid_ = o2w_.inverseAffine(w2o_);
		if (!valid_) return;
		bbox_ = disk_cylinder_detail::transformed_bbox(o2w_,
			-outer_r, outer_r, -outer_r, outer_r, height, height);
	}

	bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
		if (!valid_) return false;
		using namespace disk_cylinder_detail;
		const point3 ro = apply_point(w2o_, r.origin());
		const vec3   rd = apply_vector(w2o_, r.direction());   // NOT normalised - see file comment

		const auto h = shape_.intersect(ro.x(), ro.y(), ro.z(), rd.x(), rd.y(), rd.z(),
										 ray_t.min, ray_t.max);
		if (!h) return false;

		rec.t = h->t;
		const double hx = ro.x() + h->t * rd.x();
		const double hy = ro.y() + h->t * rd.y();
		rec.p = apply_point(o2w_, point3(hx, hy, shape_.height));
		rec.u = h->u;
		rec.v = h->v;

		// pbrt-v4 Disk::Intersect: dpdu tangent to increasing phi, dpdv radial
		// (pointing from the outer edge toward the inner edge/center).
		vec3 dpdu_obj(-shape_.phi_max * hy, shape_.phi_max * hx, 0.0);
		const double dist = std::sqrt(hx * hx + hy * hy);
		vec3 dpdv_obj = (dist > 1e-12)
			? vec3(hx, hy, 0.0) * ((shape_.inner_r - shape_.outer_r) / dist)
			: vec3(1.0, 0.0, 0.0);
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

	// Solid-angle NEE sampling - see file header comment for the uniform-
	// scale caveat.
	vec3 random(const point3& origin) const override {
		if (!valid_) return vec3(1, 0, 0);
		using namespace disk_cylinder_detail;
		const point3 ctx_obj = apply_point(w2o_, origin);
		const SamplingContext<double> ctx{ctx_obj.x(), ctx_obj.y(), ctx_obj.z(), 0, 0, 0};
		const auto ss = shape_.sample_from(ctx, random_double(), random_double());
		const point3 p_world = apply_point(o2w_, point3(ss.px, ss.py, ss.pz));
		return p_world - origin;
	}

	double pdf_value(const point3& origin, const vec3& direction) const override {
		if (!valid_) return 0.0;
		using namespace disk_cylinder_detail;
		const point3 ctx_obj = apply_point(w2o_, origin);
		const vec3 dir_obj = apply_vector(w2o_, direction);
		const SamplingContext<double> ctx{ctx_obj.x(), ctx_obj.y(), ctx_obj.z(), 0, 0, 0};
		return shape_.pdf_from(ctx, dir_obj.x(), dir_obj.y(), dir_obj.z());
	}

  private:
	DiskShape<double> shape_;
	shared_ptr<material> mat_;
	pbrt_scene::Matrix4 o2w_, w2o_;
	bool valid_ = false;
	aabb bbox_;
};


class cylinder_hittable : public hittable {
  public:
	cylinder_hittable(double radius, double z_min, double z_max, double phi_max,
					   const pbrt_scene::Matrix4& object_to_world, shared_ptr<material> mat)
		: shape_(CylinderShape<double>::make_partial(0.0, 0.0, z_min, z_max, radius, phi_max)),
		  mat_(mat), o2w_(object_to_world)
	{
		valid_ = o2w_.inverseAffine(w2o_);
		if (!valid_) return;
		bbox_ = disk_cylinder_detail::transformed_bbox(o2w_,
			-radius, radius, -radius, radius, z_min, z_max);
	}

	bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
		if (!valid_) return false;
		using namespace disk_cylinder_detail;
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

		// pbrt-v4 Cylinder::Intersect: dpdu tangent to increasing phi, dpdv
		// along the axis.
		vec3 dpdu_obj(-shape_.phi_max * hy, shape_.phi_max * hx, 0.0);
		vec3 dpdv_obj(0.0, 0.0, shape_.z_max - shape_.z_min);
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

	// Solid-angle NEE sampling - see file header comment for the uniform-
	// scale caveat.
	vec3 random(const point3& origin) const override {
		if (!valid_) return vec3(1, 0, 0);
		using namespace disk_cylinder_detail;
		const point3 ctx_obj = apply_point(w2o_, origin);
		const SamplingContext<double> ctx{ctx_obj.x(), ctx_obj.y(), ctx_obj.z(), 0, 0, 0};
		const auto ss = shape_.sample_from(ctx, random_double(), random_double());
		const point3 p_world = apply_point(o2w_, point3(ss.px, ss.py, ss.pz));
		return p_world - origin;
	}

	double pdf_value(const point3& origin, const vec3& direction) const override {
		if (!valid_) return 0.0;
		using namespace disk_cylinder_detail;
		const point3 ctx_obj = apply_point(w2o_, origin);
		const vec3 dir_obj = apply_vector(w2o_, direction);
		const SamplingContext<double> ctx{ctx_obj.x(), ctx_obj.y(), ctx_obj.z(), 0, 0, 0};
		return shape_.pdf_from(ctx, dir_obj.x(), dir_obj.y(), dir_obj.z());
	}

  private:
	CylinderShape<double> shape_;
	shared_ptr<material> mat_;
	pbrt_scene::Matrix4 o2w_, w2o_;
	bool valid_ = false;
	aabb bbox_;
};
