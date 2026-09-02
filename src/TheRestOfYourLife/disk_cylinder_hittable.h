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
//
// Object motion blur (pbrt-v4 ActiveTransform "StartTime"/"EndTime" around a
// Shape "disk"/"cylinder"): both classes carry a SECOND object-to-world
// transform and resolve the real one at each ray's own r.time() via
// AnimatedTransform (src/shared/animated_transform.h) - real TRS
// decomposition (translation lerp, rotation slerp, scale lerp), not a naive
// per-element matrix lerp, which would visibly shear a rotating shape partway
// through its motion. The two endpoint transforms are cached as their own
// AT_Mat44 in the constructor; a STATIC shape (the overwhelmingly common
// case - AnimatedTransform::IsAnimated() false, both endpoints identical)
// also caches the resolved object<->world Matrix4 pair once there, so hit()
// pays for real per-ray interpolation and a 4x4 inversion only when the
// shape is genuinely moving, exactly the "true no-op when static" contract
// sphere.h's own motion blur established. random()/pdf_value() (NEE
// sampling) intentionally still resolve against the STARTING transform only,
// matching sphere.h's own identical, documented limitation (its random()/
// pdf_value() use center.at(0) unconditionally) - not a new gap this file
// introduces.

#include "hittable.h"
#include "material.h"
#include "../shared/pbrt_scene.h"
#include "../shared/shapes.h"
#include "../shared/animated_transform.h"

#include "affine_transform_apply.h"

#include <cmath>
#include <memory>

namespace disk_cylinder_detail {

inline AT_Mat44 toAnimMat(const pbrt_scene::Matrix4 &m) {
	AT_Mat44 r;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			r.m[i][j] = m.m[i * 4 + j];
	return r;
}

inline pbrt_scene::Matrix4 fromAnimMat(const AT_Mat44 &m) {
	pbrt_scene::Matrix4 r;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			r.m[i * 4 + j] = m.m[i][j];
	return r;
}

} // namespace disk_cylinder_detail


class disk_hittable : public hittable {
  public:
	disk_hittable(double outer_r, double inner_r, double height, double phi_max,
				  const pbrt_scene::Matrix4& object_to_world,
				  const pbrt_scene::Matrix4& object_to_world_end,
				  shared_ptr<material> mat)
		: shape_(DiskShape<double>::make_annular(0.0, 0.0, height, outer_r, inner_r, phi_max)),
		  mat_(mat),
		  anim_(disk_cylinder_detail::toAnimMat(object_to_world), 0.0,
				disk_cylinder_detail::toAnimMat(object_to_world_end), 1.0),
		  o2w_static_(object_to_world)
	{
		valid_ = o2w_static_.inverseAffine(w2o_static_);
		if (!valid_) return;
		if (!anim_.IsAnimated()) {
			bbox_ = affine_transform::transformed_bbox(o2w_static_,
				-outer_r, outer_r, -outer_r, outer_r, height, height);
		} else {
			const aabb box0 = affine_transform::transformed_bbox(object_to_world,
				-outer_r, outer_r, -outer_r, outer_r, height, height);
			const aabb box1 = affine_transform::transformed_bbox(object_to_world_end,
				-outer_r, outer_r, -outer_r, outer_r, height, height);
			bbox_ = aabb(box0, box1);
		}
	}

	bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
		if (!valid_) return false;
		using namespace affine_transform;
		pbrt_scene::Matrix4 o2w, w2o;
		if (!resolveTransforms(r.time(), o2w, w2o)) return false;
		const point3 ro = apply_point(w2o, r.origin());
		const vec3   rd = apply_vector(w2o, r.direction());   // NOT normalised - see file comment

		const auto h = shape_.intersect(ro.x(), ro.y(), ro.z(), rd.x(), rd.y(), rd.z(),
										 ray_t.min, ray_t.max);
		if (!h) return false;

		rec.t = h->t;
		const double hx = ro.x() + h->t * rd.x();
		const double hy = ro.y() + h->t * rd.y();
		rec.p = apply_point(o2w, point3(hx, hy, shape_.height));
		rec.u = h->u;
		rec.v = h->v;

		// pbrt-v4 Disk::Intersect: dpdu tangent to increasing phi, dpdv radial
		// (pointing from the outer edge toward the inner edge/center).
		vec3 dpdu_obj(-shape_.phi_max * hy, shape_.phi_max * hx, 0.0);
		const double dist = std::sqrt(hx * hx + hy * hy);
		vec3 dpdv_obj = (dist > 1e-12)
			? vec3(hx, hy, 0.0) * ((shape_.inner_r - shape_.outer_r) / dist)
			: vec3(1.0, 0.0, 0.0);
		rec.dpdu = apply_vector(o2w, dpdu_obj);
		rec.dpdv = apply_vector(o2w, dpdv_obj);

		vec3 n = apply_normal(w2o, vec3(h->nx, h->ny, h->nz));
		const double len = n.length();
		if (len > 0) n = n / len;
		rec.set_face_normal(r, n);
		rec.mat = mat_;
		return true;
	}

	aabb bounding_box() const override { return bbox_; }

	// Solid-angle NEE sampling - see file header comment for the uniform-
	// scale caveat and the "resolves against the starting transform only"
	// motion-blur caveat.
	vec3 random(const point3& origin) const override {
		if (!valid_) return vec3(1, 0, 0);
		using namespace affine_transform;
		pbrt_scene::Matrix4 o2w, w2o;
		if (!resolveTransforms(0.0, o2w, w2o)) return vec3(1, 0, 0);
		const point3 ctx_obj = apply_point(w2o, origin);
		const SamplingContext<double> ctx{ctx_obj.x(), ctx_obj.y(), ctx_obj.z(), 0, 0, 0};
		const auto ss = shape_.sample_from(ctx, random_double(), random_double());
		const point3 p_world = apply_point(o2w, point3(ss.px, ss.py, ss.pz));
		return p_world - origin;
	}

	double pdf_value(const point3& origin, const vec3& direction) const override {
		if (!valid_) return 0.0;
		using namespace affine_transform;
		pbrt_scene::Matrix4 o2w, w2o;
		if (!resolveTransforms(0.0, o2w, w2o)) return 0.0;
		const point3 ctx_obj = apply_point(w2o, origin);
		const vec3 dir_obj = apply_vector(w2o, direction);
		const SamplingContext<double> ctx{ctx_obj.x(), ctx_obj.y(), ctx_obj.z(), 0, 0, 0};
		return shape_.pdf_from(ctx, dir_obj.x(), dir_obj.y(), dir_obj.z());
	}

	shared_ptr<material> get_material() const { return mat_; }

  private:
	DiskShape<double> shape_;
	shared_ptr<material> mat_;
	AnimatedTransform anim_;
	pbrt_scene::Matrix4 o2w_static_, w2o_static_;
	bool valid_ = false;
	aabb bbox_;

	// Resolves the real object<->world transform pair for a ray at the given
	// time - see this file's own header comment for the static-vs-animated
	// cost tradeoff. Returns false (o2w/w2o left untouched) only for the
	// animated path's own degenerate-matrix case (zero scale at that
	// instant) - the static path's degeneracy is already excluded by
	// valid_ at construction time.
	bool resolveTransforms(double time, pbrt_scene::Matrix4& o2w, pbrt_scene::Matrix4& w2o) const {
		if (!anim_.IsAnimated()) {
			o2w = o2w_static_;
			w2o = w2o_static_;
			return true;
		}
		const AT_Mat44 mat = anim_.Interpolate(time);
		AT_Mat44 inv;
		if (!at_invert(mat, inv)) return false;
		o2w = disk_cylinder_detail::fromAnimMat(mat);
		w2o = disk_cylinder_detail::fromAnimMat(inv);
		return true;
	}
};


class cylinder_hittable : public hittable {
  public:
	cylinder_hittable(double radius, double z_min, double z_max, double phi_max,
					   const pbrt_scene::Matrix4& object_to_world,
					   const pbrt_scene::Matrix4& object_to_world_end,
					   shared_ptr<material> mat)
		: shape_(CylinderShape<double>::make_partial(0.0, 0.0, z_min, z_max, radius, phi_max)),
		  mat_(mat),
		  anim_(disk_cylinder_detail::toAnimMat(object_to_world), 0.0,
				disk_cylinder_detail::toAnimMat(object_to_world_end), 1.0),
		  o2w_static_(object_to_world)
	{
		valid_ = o2w_static_.inverseAffine(w2o_static_);
		if (!valid_) return;
		if (!anim_.IsAnimated()) {
			bbox_ = affine_transform::transformed_bbox(o2w_static_,
				-radius, radius, -radius, radius, z_min, z_max);
		} else {
			const aabb box0 = affine_transform::transformed_bbox(object_to_world,
				-radius, radius, -radius, radius, z_min, z_max);
			const aabb box1 = affine_transform::transformed_bbox(object_to_world_end,
				-radius, radius, -radius, radius, z_min, z_max);
			bbox_ = aabb(box0, box1);
		}
	}

	bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
		if (!valid_) return false;
		using namespace affine_transform;
		pbrt_scene::Matrix4 o2w, w2o;
		if (!resolveTransforms(r.time(), o2w, w2o)) return false;
		const point3 ro = apply_point(w2o, r.origin());
		const vec3   rd = apply_vector(w2o, r.direction());   // NOT normalised - see file comment

		const auto h = shape_.intersect(ro.x(), ro.y(), ro.z(), rd.x(), rd.y(), rd.z(),
										 ray_t.min, ray_t.max);
		if (!h) return false;

		rec.t = h->t;
		const double hx = ro.x() + h->t * rd.x();
		const double hy = ro.y() + h->t * rd.y();
		const double hz = ro.z() + h->t * rd.z();
		rec.p = apply_point(o2w, point3(hx, hy, hz));
		rec.u = h->u;
		rec.v = h->v;

		// pbrt-v4 Cylinder::Intersect: dpdu tangent to increasing phi, dpdv
		// along the axis.
		vec3 dpdu_obj(-shape_.phi_max * hy, shape_.phi_max * hx, 0.0);
		vec3 dpdv_obj(0.0, 0.0, shape_.z_max - shape_.z_min);
		rec.dpdu = apply_vector(o2w, dpdu_obj);
		rec.dpdv = apply_vector(o2w, dpdv_obj);

		vec3 n = apply_normal(w2o, vec3(h->nx, h->ny, h->nz));
		const double len = n.length();
		if (len > 0) n = n / len;
		rec.set_face_normal(r, n);
		rec.mat = mat_;
		return true;
	}

	aabb bounding_box() const override { return bbox_; }

	// Solid-angle NEE sampling - see file header comment for the uniform-
	// scale caveat and the "resolves against the starting transform only"
	// motion-blur caveat.
	vec3 random(const point3& origin) const override {
		if (!valid_) return vec3(1, 0, 0);
		using namespace affine_transform;
		pbrt_scene::Matrix4 o2w, w2o;
		if (!resolveTransforms(0.0, o2w, w2o)) return vec3(1, 0, 0);
		const point3 ctx_obj = apply_point(w2o, origin);
		const SamplingContext<double> ctx{ctx_obj.x(), ctx_obj.y(), ctx_obj.z(), 0, 0, 0};
		const auto ss = shape_.sample_from(ctx, random_double(), random_double());
		const point3 p_world = apply_point(o2w, point3(ss.px, ss.py, ss.pz));
		return p_world - origin;
	}

	double pdf_value(const point3& origin, const vec3& direction) const override {
		if (!valid_) return 0.0;
		using namespace affine_transform;
		pbrt_scene::Matrix4 o2w, w2o;
		if (!resolveTransforms(0.0, o2w, w2o)) return 0.0;
		const point3 ctx_obj = apply_point(w2o, origin);
		const vec3 dir_obj = apply_vector(w2o, direction);
		const SamplingContext<double> ctx{ctx_obj.x(), ctx_obj.y(), ctx_obj.z(), 0, 0, 0};
		return shape_.pdf_from(ctx, dir_obj.x(), dir_obj.y(), dir_obj.z());
	}

	shared_ptr<material> get_material() const { return mat_; }

	// Volumetric near/far entry-exit for a MediumInterface-attached fog
	// cylinder (constant_medium's own boundary use, not surface rendering) -
	// a code-review pass found constant_medium::hit()'s generic "two
	// sequential hit() calls" pattern (correct for a CLOSED boundary like a
	// sphere) can silently miss real medium extent on this OPEN (uncapped)
	// shape: a ray traveling near-axially through the open top/bottom
	// crosses no lateral wall within [z_min,z_max] at all, and even when it
	// does, the tube quadric's own two roots don't necessarily correspond to
	// "this ray's real entry/exit through this open shape" the way a closed
	// shape's hit()/hit() pair does - hit() only ever models the lateral
	// wall, since an uncapped cylinder has no surface at its flat ends.
	// Mirrors GPU's own identical fix (gpu/optix/optix_intersection_disk_
	// cylinder.h's MaterialType::Medium branch, dc_solve_tube_quadratic)
	// exactly: solve the (infinite) tube's own quadratic, intersect that
	// interval with the z-slab's own [z_min,z_max] interval taken directly
	// from the z-planes - correct whether the ray's real entry/exit is
	// through the wall or the open ends, and GPU's own confirmed-correct
	// reference implementation for this exact shape/use case.
	//
	// r is the real WORLD-space ray (this method carries it into object
	// space itself, matching hit()'s own apply_point/apply_vector
	// convention - NOT renormalized, so t stays in the same units as every
	// other method here, i.e. divide by r.direction().length() to convert
	// to true world distance, same as hit()'s own rec.t consumers already
	// do). Returns false when this object is invalid, or the tube and
	// z-slab intervals don't overlap at all (no medium extent along this
	// ray) - out_t0/out_t1 are still written in that case but meaningless,
	// the caller must check the return value first. When true, out_t0/
	// out_t1 are otherwise left UNCLAMPED (may be negative, e.g. an origin
	// already inside the medium) - matching hit()'s own rec1.t/rec2.t, the
	// caller is responsible for the same ray_t.min/max clamping it already
	// does for other boundary shapes.
	bool volume_bounds(const ray& r, double& out_t0, double& out_t1) const {
		if (!valid_) return false;
		using namespace affine_transform;
		pbrt_scene::Matrix4 o2w, w2o;
		if (!resolveTransforms(r.time(), o2w, w2o)) return false;
		const point3 ro = apply_point(w2o, r.origin());
		const vec3   rd = apply_vector(w2o, r.direction());   // NOT normalised - see file comment

		double tube_t0 = -infinity, tube_t1 = infinity;
		bool hasTube;
		const double a = rd.x() * rd.x() + rd.y() * rd.y();
		if (a == 0.0) {
			hasTube = ro.x() * ro.x() + ro.y() * ro.y() <= shape_.radius * shape_.radius;
		} else {
			const double b = 2.0 * (ro.x() * rd.x() + ro.y() * rd.y());
			const double c = ro.x() * ro.x() + ro.y() * ro.y() - shape_.radius * shape_.radius;
			const double f = b / (2.0 * a);
			const double vx = ro.x() - f * rd.x(), vy = ro.y() - f * rd.y();
			const double len_v = std::sqrt(vx * vx + vy * vy);
			const double discrim = 4.0 * a * (shape_.radius + len_v) * (shape_.radius - len_v);
			if (discrim < 0.0) {
				hasTube = false;
			} else {
				const double sqrt_disc = std::sqrt(discrim);
				const double q = (b < 0.0) ? -0.5 * (b - sqrt_disc) : -0.5 * (b + sqrt_disc);
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

  private:
	CylinderShape<double> shape_;
	shared_ptr<material> mat_;
	AnimatedTransform anim_;
	pbrt_scene::Matrix4 o2w_static_, w2o_static_;
	bool valid_ = false;
	aabb bbox_;

	// See disk_hittable::resolveTransforms's own comment - identical idiom.
	bool resolveTransforms(double time, pbrt_scene::Matrix4& o2w, pbrt_scene::Matrix4& w2o) const {
		if (!anim_.IsAnimated()) {
			o2w = o2w_static_;
			w2o = w2o_static_;
			return true;
		}
		const AT_Mat44 mat = anim_.Interpolate(time);
		AT_Mat44 inv;
		if (!at_invert(mat, inv)) return false;
		o2w = disk_cylinder_detail::fromAnimMat(mat);
		w2o = disk_cylinder_detail::fromAnimMat(inv);
		return true;
	}
};
