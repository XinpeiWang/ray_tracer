#pragma once
// animated_transform_instance.h -- real object motion blur for
// trianglemesh/plymesh/loopsubdiv (pbrt-v4 ActiveTransform "StartTime"/
// "EndTime" around one of these Shape types) - CPU only.
//
// These shapes bake their geometry to WORLD space once, at flatten() time
// (pbrt_flatten.h transforms every vertex by the StartTime CTM and never
// carries the transform itself any further) - unlike Sphere/Disk/Cylinder,
// which keep the transform separate and carry the RAY into object space at
// intersection time (see transform_instance.h's own header comment for
// that technique: the ray moves, the geometry doesn't, and t survives the
// round trip unmodified because the direction is carried across without
// renormalising). Real per-ray-time motion blur for a mesh needs the SAME
// trick - just with MotionState's real per-ray-time TRS interpolation
// (motion_state.h, shared with disk_cylinder_hittable.h's own motion blur)
// standing in for transform_instance's single static transform.
//
// `object` must therefore be built in OBJECT space (pbrt_flatten::
// AnimatedTriangleMesh's own raw, untransformed triangles - see that
// struct's own comment), not the world-space bake pbrt_cpu_builder.h's
// static path produces. Only ever constructed when a shape's own xformEnd
// genuinely differs from xform; a static mesh keeps using the existing
// zero-overhead world-space-bake path entirely unchanged.
//
// Like transform_instance.h, this does NOT override random()/pdf_value()
// (NEE sampling) - a pre-existing, disclosed gap for object-space-
// instanced geometry (see that file's own lack of an override), not a new
// one this class introduces. pbrt_flatten.h already excludes an emissive
// mesh from taking this path at all (falls back to a static bake instead,
// warned) for exactly this reason - a shape wrapped here is never added to
// the `lights` list, so the gap is inert in practice.

#include <memory>

#include "../shared/pbrt_scene.h"

#include "aabb.h"
#include "affine_transform_apply.h"
#include "hittable.h"
#include "motion_state.h"

class animated_transform_instance : public hittable {
  public:
    // `object` is the OBJECT-space geometry (already built into a BVH by
    // the caller, mirroring transform_instance's own contract).
    // `object_to_world`/`object_to_world_end` are the shape's own real
    // StartTime/EndTime CTM (pbrt_flatten::AnimatedTriangleMesh::xform/
    // xformEnd).
    animated_transform_instance(std::shared_ptr<hittable> object,
                                const pbrt_scene::Matrix4& object_to_world,
                                const pbrt_scene::Matrix4& object_to_world_end)
      : object_(object), motion_(object_to_world, object_to_world_end)
    {
        const aabb b = object_->bounding_box();
        bbox_ = motion_.motionBoundingBox([&](const pbrt_scene::Matrix4& o2w) {
            return affine_transform::transformed_bbox(o2w, b.x.min, b.x.max,
                                                        b.y.min, b.y.max,
                                                        b.z.min, b.z.max);
        });
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        pbrt_scene::Matrix4 o2w, w2o;
        if (!motion_.resolve(r.time(), o2w, w2o)) return false;

        const ray local(affine_transform::apply_point(w2o, r.origin()),
                        affine_transform::apply_vector(w2o, r.direction()),   // NOT normalised - see transform_instance.h
                        r.time());
        if (!object_->hit(local, ray_t, rec)) return false;

        rec.p = affine_transform::apply_point(o2w, rec.p);
        // set_face_normal() decides front/back from the ray direction, so
        // it is given the WORLD ray - see transform_instance.h's own hit()
        // comment for why (keeps a mirroring transform consistent with the
        // geometry the camera actually sees).
        vec3 n = affine_transform::apply_normal(w2o, rec.normal);
        const double len = n.length();
        if (len > 0) n = n / len;
        rec.set_face_normal(r, n);
        // dpdu/dpdv are real tangent VECTORS (triangle.h's own UV-edge
        // Jacobian solve, computed in object space by the child hit() call
        // above) - a plain forward transform, unlike the inverse-transpose
        // a normal needs (transform_instance.h's hit() has no dpdu/dpdv to
        // transform at all; triangle hit_records do, and normal-mapping/
        // texture-differential filtering both read them, so leaving them
        // in object space here would misdirect both).
        rec.dpdu = affine_transform::apply_vector(o2w, rec.dpdu);
        rec.dpdv = affine_transform::apply_vector(o2w, rec.dpdv);
        return true;
    }

    aabb bounding_box() const override { return bbox_; }

  private:
    std::shared_ptr<hittable> object_;
    MotionState motion_;
    aabb bbox_;
};
