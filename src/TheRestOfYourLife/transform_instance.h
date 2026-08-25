#pragma once
// transform_instance.h -- a hittable that places shared geometry under an
// arbitrary affine transform.
//
// This is what makes object instancing worth doing. hittable.h already has
// `translate` and `rotate_y`, which are this class with all but one degree of
// freedom removed; a pbrt instance can carry any composition of translate,
// rotate, scale and a raw 4x4 Transform, so neither of them is enough.
//
// HOW IT WORKS, AND WHY t SURVIVES
// --------------------------------
// The geometry is never moved. Instead the RAY is carried into the object's
// own coordinate system, intersected there, and the resulting hit carried
// back out. That is the entire trade instancing makes: one matrix multiply per
// ray per instance entered, in exchange for storing the geometry once however
// many times it appears.
//
// The ray's direction is transformed WITHOUT renormalising, which is what
// keeps the ray parameter meaningful. Under a scale, a normalised direction
// would make distances in object space differ from distances in world space,
// so the caller's [t_min, t_max] window would mean something different inside
// than outside - and the returned rec.t would be wrong in world terms. Leaving
// the direction unnormalised makes t identical in both spaces, so the interval
// passes straight through and the hit record needs no rescaling.
//
// NORMALS NEED THE INVERSE TRANSPOSE
// ----------------------------------
// A normal is not a direction; under a non-uniform scale, pushing it through
// the same matrix as a point tilts it off the surface. Going object -> world
// therefore uses the transpose of the world -> object matrix, which is already
// to hand because the inverse is what the ray needed.

#include <cmath>
#include <memory>

#include "../shared/pbrt_scene.h"

#include "aabb.h"
#include "affine_transform_apply.h"
#include "hittable.h"

class transform_instance : public hittable {
  public:
    // `object_to_world` places the shared geometry. Construction inverts it
    // once; a singular matrix (a zero scale) leaves the instance empty rather
    // than producing infinities that would surface much later as geometry
    // mysteriously missing from the render.
    transform_instance(std::shared_ptr<hittable> object,
                       const pbrt_scene::Matrix4& object_to_world)
      : object(object), o2w(object_to_world)
    {
        valid = o2w.inverseAffine(w2o);
        if (!valid) return;

        // The world-space bounds are the bounds of the transformed corners,
        // not the transform of the bounds' corners - those differ the moment
        // a rotation is involved, and using the wrong one produces a box that
        // clips the object it is supposed to contain.
        const aabb b = object->bounding_box();
        bbox = affine_transform::transformed_bbox(o2w, b.x.min, b.x.max,
                                                   b.y.min, b.y.max,
                                                   b.z.min, b.z.max);
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        if (!valid) return false;

        const ray local(affine_transform::apply_point(w2o, r.origin()),
                        affine_transform::apply_vector(w2o, r.direction()),   // NOT normalised - see header
                        r.time());
        if (!object->hit(local, ray_t, rec)) return false;

        rec.p = affine_transform::apply_point(o2w, rec.p);
        // set_face_normal() decides front/back from the ray direction, so it
        // is given the WORLD ray. Calling it again rather than transforming
        // the child's flag keeps a mirroring transform (negative determinant,
        // which flips winding) consistent with the geometry the camera sees.
        vec3 n = affine_transform::apply_normal(w2o, rec.normal);
        const double len = n.length();
        if (len > 0) n = n / len;
        rec.set_face_normal(r, n);
        return true;
    }

    aabb bounding_box() const override { return bbox; }

    // Accessor for structural walkers (e.g. --spectral's material-scan
    // walker in cpu_interface.cpp) that need to recurse past this wrapper
    // into the instanced geometry, mirroring translate/rotate_y's own
    // get_object() in hittable.h.
    std::shared_ptr<hittable> get_object() const { return object; }

  private:
    std::shared_ptr<hittable> object;
    pbrt_scene::Matrix4 o2w;
    pbrt_scene::Matrix4 w2o;
    bool valid = false;
    aabb bbox;
};
