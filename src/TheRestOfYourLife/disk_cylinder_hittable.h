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
// Shape "disk"/"cylinder"): both classes hold a disk_cylinder_detail::
// MotionState (below), which carries a SECOND object-to-world transform and
// resolves the real one at each ray's own r.time() via AnimatedTransform
// (src/shared/animated_transform.h) - real TRS decomposition (translation
// lerp, rotation slerp, scale lerp), not a naive per-element matrix lerp,
// which would visibly shear a rotating shape partway through its motion. A
// STATIC shape (the overwhelmingly common case - AnimatedTransform::
// IsAnimated() false, both endpoints identical) resolves to a cached
// object<->world Matrix4 pair with no per-ray interpolation or inversion at
// all - the "true no-op when static" contract sphere.h's own motion blur
// established. random()/pdf_value() (NEE sampling) intentionally still
// resolve against the STARTING transform only, matching sphere.h's own
// identical, documented limitation (its random()/pdf_value() use
// center.at(0) unconditionally) - not a new gap this file introduces; see
// MotionState::o2wStart()/w2oStart() for why this needs no per-call
// resolution at all, animated or not.

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

// Shared per-shape motion state for Disk/Cylinder - factored out of
// disk_hittable/cylinder_hittable (previously two hand-duplicated copies of
// this exact logic, caught by a code-review pass on the commit that first
// added it) so a future fix to the degenerate-matrix handling, or an
// extension of this motion-blur idiom to a third shape, needs only one edit.
// This codebase has already shipped a real bug once from the analogous
// duplicated-with-a-comment-pointer pattern in bvh_aggregate_hittable - see
// that file's own history for the "silently dropped ~50% of medium scatter
// events" incident this precedent warns against.
// Two disclosed, accepted limitations, both raised and deliberately left
// unfixed by a code-review pass on the commit that added this class -
// narrow enough in practice, and big enough to fix properly, that they were
// scoped out rather than papered over:
//   - The two keyframes are always t0=0.0/t1=1.0 here, regardless of the
//     scene's own TransformTimes directive (pbrt_scene::Scene::
//     transformTimeStart/transformTimeEnd) - camera motion blur detects and
//     warns about exactly this TransformTimes/shutter mismatch
//     (pbrt_flatten.h's Camera-building code), Disk/Cylinder (and
//     pre-existing Sphere motion blur) do not. Fixing this for real means
//     threading the camera's own shutter/TransformTimes context down into
//     every shape builder, not a local change here.
//   - random()/pdf_value() (see disk_hittable/cylinder_hittable below)
//     resolve at time=0 unconditionally, so an animated EMISSIVE disk/
//     cylinder's NEE sample point and its own hit()-time shadow-ray test
//     can disagree once real motion has accumulated - matching sphere.h's
//     own identical, already-shipped center.at(0)-only limitation; MIS
//     against BSDF sampling partially self-corrects this, same as it
//     already does for sphere. A real fix needs a time parameter on the
//     whole hittable::random()/pdf_value() interface, not just this class.
class MotionState {
  public:
	MotionState(const pbrt_scene::Matrix4 &object_to_world,
				const pbrt_scene::Matrix4 &object_to_world_end)
		: anim_(toAnimMat(object_to_world), 0.0, toAnimMat(object_to_world_end), 1.0),
		  o2wStart_(object_to_world)
	{
		startValid_ = o2wStart_.inverseAffine(w2oStart_);
	}

	// True iff the StartTime pose itself is invertible. Only gates the
	// STATIC fast path and NEE sampling (both of which use exactly this
	// pose, cached) - deliberately NOT the animated hit()-time path, whose
	// own validity is checked per-ray by resolve() instead. A shape whose
	// StartTime keyframe is a legitimate degenerate matrix (e.g. a "grow
	// from nothing" Scale 0 0 0 opening keyframe, a real pbrt authoring
	// idiom) must still render correctly once the interpolated matrix
	// becomes non-degenerate - gating the animated path on THIS flag too
	// (an earlier version of this code did) made such a shape permanently
	// invisible for its entire lifetime, caught by a code-review pass.
	bool startValid() const { return startValid_; }

	// The StartTime pose, precomputed once at construction regardless of
	// whether this shape is animated - the single source both the static
	// fast path AND NEE sampling (random()/pdf_value(), which always wants
	// time=0 - see this file's own header comment) read directly, with no
	// per-call interpolation or inversion either way.
	const pbrt_scene::Matrix4 &o2wStart() const { return o2wStart_; }
	const pbrt_scene::Matrix4 &w2oStart() const { return w2oStart_; }

	// Resolves the real object<->world transform pair for a ray at the
	// given time. STATIC shape: the cached StartTime pose, true no-op cost
	// - fails only if that pose is itself degenerate (startValid() false).
	// ANIMATED shape: real per-ray TRS interpolation (AnimatedTransform)
	// composed via a closed-form AFFINE inverse (Matrix4::inverseAffine()),
	// not the generic 4x4 Gauss-Jordan at_invert() - the interpolated
	// matrix is always a pure affine transform (no perspective row), so the
	// same cheaper affine-specific inverse the static path already relies
	// on applies here too, at a fraction of the cost. Gated ONLY on this
	// time's own interpolated matrix being invertible, never on
	// startValid() - see that accessor's own comment for why.
	bool resolve(double time, pbrt_scene::Matrix4 &o2w, pbrt_scene::Matrix4 &w2o) const {
		if (!anim_.IsAnimated()) {
			if (!startValid_) return false;
			o2w = o2wStart_;
			w2o = w2oStart_;
			return true;
		}
		o2w = fromAnimMat(anim_.Interpolate(time));
		return o2w.inverseAffine(w2o);
	}

	// World-space bounding box across the shape's full swept motion, given
	// a callback that returns transformed_bbox() for a specific
	// object<->world matrix (the shape-specific object-space extent is the
	// caller's, not this class's, concern). A static shape needs only its
	// one pose. An animated one samples the interpolated pose at several
	// points across [0,1], not just the two endpoints, and unions all of
	// them: unioning only the two ENDPOINT boxes underestimates the true
	// swept volume whenever the motion includes rotation - e.g. a disk
	// rotating from 45 to 135 degrees about an axis in its own plane sweeps
	// through 90 degrees at the midpoint, extending further along that axis
	// than either endpoint alone. A code-review pass on the commit that
	// first added animated Disk/Cylinder motion blur (translation-only
	// union of two boxes) caught this as a real gap: BVH traversal could
	// cull rays that should hit the shape mid-rotation, since
	// bounding_box() feeds it directly. This is a conservative
	// approximation, not an exact analytic bound (pbrt-v4's own
	// AnimatedTransform::BoundPointMotion solves for the true per-axis
	// rotational extremum) - kMotionSamples samples closes the gap for any
	// realistic single-frame shutter rotation without that extra
	// complexity; a rotation whipping through more than a full turn within
	// one shutter interval (vanishingly rare in practice) could still poke
	// outside between samples.
	template <typename BoxForTransform>
	aabb motionBoundingBox(BoxForTransform boxForTransform) const {
		if (!anim_.IsAnimated()) return boxForTransform(o2wStart_);
		aabb box = boxForTransform(o2wStart_);
		constexpr int kMotionSamples = 17;
		for (int i = 1; i < kMotionSamples; ++i) {
			const double t = static_cast<double>(i) / (kMotionSamples - 1);
			box = aabb(box, boxForTransform(fromAnimMat(anim_.Interpolate(t))));
		}
		return box;
	}

  private:
	AnimatedTransform anim_;
	pbrt_scene::Matrix4 o2wStart_, w2oStart_;
	bool startValid_ = false;
};

} // namespace disk_cylinder_detail


class disk_hittable : public hittable {
  public:
	disk_hittable(double outer_r, double inner_r, double height, double phi_max,
				  const pbrt_scene::Matrix4& object_to_world,
				  const pbrt_scene::Matrix4& object_to_world_end,
				  shared_ptr<material> mat)
		: shape_(DiskShape<double>::make_annular(0.0, 0.0, height, outer_r, inner_r, phi_max)),
		  mat_(mat),
		  motion_(object_to_world, object_to_world_end)
	{
		bbox_ = motion_.motionBoundingBox([&](const pbrt_scene::Matrix4 &o2w) {
			return affine_transform::transformed_bbox(o2w,
				-outer_r, outer_r, -outer_r, outer_r, height, height);
		});
	}

	bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
		using namespace affine_transform;
		pbrt_scene::Matrix4 o2w, w2o;
		if (!motion_.resolve(r.time(), o2w, w2o)) return false;
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
		if (!motion_.startValid()) return vec3(1, 0, 0);
		using namespace affine_transform;
		const point3 ctx_obj = apply_point(motion_.w2oStart(), origin);
		const SamplingContext<double> ctx{ctx_obj.x(), ctx_obj.y(), ctx_obj.z(), 0, 0, 0};
		const auto ss = shape_.sample_from(ctx, random_double(), random_double());
		const point3 p_world = apply_point(motion_.o2wStart(), point3(ss.px, ss.py, ss.pz));
		return p_world - origin;
	}

	double pdf_value(const point3& origin, const vec3& direction) const override {
		if (!motion_.startValid()) return 0.0;
		using namespace affine_transform;
		const point3 ctx_obj = apply_point(motion_.w2oStart(), origin);
		const vec3 dir_obj = apply_vector(motion_.w2oStart(), direction);
		const SamplingContext<double> ctx{ctx_obj.x(), ctx_obj.y(), ctx_obj.z(), 0, 0, 0};
		return shape_.pdf_from(ctx, dir_obj.x(), dir_obj.y(), dir_obj.z());
	}

	shared_ptr<material> get_material() const { return mat_; }

  private:
	DiskShape<double> shape_;
	shared_ptr<material> mat_;
	disk_cylinder_detail::MotionState motion_;
	aabb bbox_;
};


class cylinder_hittable : public hittable {
  public:
	cylinder_hittable(double radius, double z_min, double z_max, double phi_max,
					   const pbrt_scene::Matrix4& object_to_world,
					   const pbrt_scene::Matrix4& object_to_world_end,
					   shared_ptr<material> mat)
		: shape_(CylinderShape<double>::make_partial(0.0, 0.0, z_min, z_max, radius, phi_max)),
		  mat_(mat),
		  motion_(object_to_world, object_to_world_end)
	{
		bbox_ = motion_.motionBoundingBox([&](const pbrt_scene::Matrix4 &o2w) {
			return affine_transform::transformed_bbox(o2w,
				-radius, radius, -radius, radius, z_min, z_max);
		});
	}

	bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
		using namespace affine_transform;
		pbrt_scene::Matrix4 o2w, w2o;
		if (!motion_.resolve(r.time(), o2w, w2o)) return false;
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
		if (!motion_.startValid()) return vec3(1, 0, 0);
		using namespace affine_transform;
		const point3 ctx_obj = apply_point(motion_.w2oStart(), origin);
		const SamplingContext<double> ctx{ctx_obj.x(), ctx_obj.y(), ctx_obj.z(), 0, 0, 0};
		const auto ss = shape_.sample_from(ctx, random_double(), random_double());
		const point3 p_world = apply_point(motion_.o2wStart(), point3(ss.px, ss.py, ss.pz));
		return p_world - origin;
	}

	double pdf_value(const point3& origin, const vec3& direction) const override {
		if (!motion_.startValid()) return 0.0;
		using namespace affine_transform;
		const point3 ctx_obj = apply_point(motion_.w2oStart(), origin);
		const vec3 dir_obj = apply_vector(motion_.w2oStart(), direction);
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
		using namespace affine_transform;
		pbrt_scene::Matrix4 o2w, w2o;
		if (!motion_.resolve(r.time(), o2w, w2o)) return false;
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
	disk_cylinder_detail::MotionState motion_;
	aabb bbox_;
};
