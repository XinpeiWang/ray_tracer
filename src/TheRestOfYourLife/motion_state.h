#pragma once
// motion_state.h -- generic per-shape object motion blur state (real TRS
// interpolation via AnimatedTransform, plus a conservative swept-AABB
// bounding_box() helper), factored out of disk_cylinder_hittable.h (that
// file's own header comment: "an extension of this motion-blur idiom to a
// third shape needs only one edit") so animated_transform_instance.h (mesh/
// curve/bilinear-patch object motion blur) can reuse the EXACT SAME class
// rather than a second hand-duplicated copy - this codebase has already
// shipped a real bug once from that duplicated-with-a-comment-pointer
// pattern (bvh_aggregate_hittable's "silently dropped ~50% of medium
// scatter events" incident).
//
// Not tied to Disk/Cylinder in any way: MotionState only resolves an
// object<->world Matrix4 pair at a given ray time (and sweeps a caller-
// supplied per-transform bounding box across the motion) - what shape the
// transform is placing is entirely the caller's concern.

#include "../shared/pbrt_scene.h"
#include "../shared/animated_transform.h"

#include "aabb.h"

inline AT_Mat44 motionStateToAnimMat(const pbrt_scene::Matrix4 &m) {
	AT_Mat44 r;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			r.m[i][j] = m.m[i * 4 + j];
	return r;
}

inline pbrt_scene::Matrix4 motionStateFromAnimMat(const AT_Mat44 &m) {
	pbrt_scene::Matrix4 r;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			r.m[i * 4 + j] = m.m[i][j];
	return r;
}

// See disk_cylinder_hittable.h's own header comment for the full design
// rationale (object motion blur via ActiveTransform "StartTime"/"EndTime",
// real TRS decomposition rather than naive per-element matrix lerp, the
// "true no-op when static" contract, and why random()/pdf_value() intentionally
// still resolve against the STARTING transform only).
class MotionState {
  public:
	MotionState(const pbrt_scene::Matrix4 &object_to_world,
				const pbrt_scene::Matrix4 &object_to_world_end)
		: anim_(motionStateToAnimMat(object_to_world), 0.0,
				motionStateToAnimMat(object_to_world_end), 1.0),
		  o2wStart_(object_to_world)
	{
		startValid_ = o2wStart_.inverseAffine(w2oStart_);
	}

	// True iff the StartTime pose itself is invertible - see disk_cylinder_
	// hittable.h's own startValid()/resolve() comments for why this gates
	// only the static fast path and NEE sampling, never the animated
	// hit()-time path (whose own validity is checked per-ray by resolve()).
	bool startValid() const { return startValid_; }

	const pbrt_scene::Matrix4 &o2wStart() const { return o2wStart_; }
	const pbrt_scene::Matrix4 &w2oStart() const { return w2oStart_; }

	// Resolves the real object<->world transform pair for a ray at the
	// given time - the cached StartTime pose for a static shape (true
	// no-op cost), or a real per-ray TRS interpolation (AnimatedTransform)
	// composed via a closed-form affine inverse for an animated one. See
	// disk_cylinder_hittable.h's own resolve() comment for the full
	// rationale.
	bool resolve(double time, pbrt_scene::Matrix4 &o2w, pbrt_scene::Matrix4 &w2o) const {
		if (!anim_.IsAnimated()) {
			if (!startValid_) return false;
			o2w = o2wStart_;
			w2o = w2oStart_;
			return true;
		}
		o2w = motionStateFromAnimMat(anim_.Interpolate(time));
		return o2w.inverseAffine(w2o);
	}

	// World-space bounding box across the shape's full swept motion - see
	// disk_cylinder_hittable.h's own motionBoundingBox() comment for why a
	// kMotionSamples-sample sweep (not just the two endpoints) is needed
	// once rotation is involved.
	template <typename BoxForTransform>
	aabb motionBoundingBox(BoxForTransform boxForTransform) const {
		if (!anim_.IsAnimated()) return boxForTransform(o2wStart_);
		aabb box = boxForTransform(o2wStart_);
		constexpr int kMotionSamples = 17;
		for (int i = 1; i < kMotionSamples; ++i) {
			const double t = static_cast<double>(i) / (kMotionSamples - 1);
			box = aabb(box, boxForTransform(motionStateFromAnimMat(anim_.Interpolate(t))));
		}
		return box;
	}

  private:
	AnimatedTransform anim_;
	pbrt_scene::Matrix4 o2wStart_, w2oStart_;
	bool startValid_ = false;
};
