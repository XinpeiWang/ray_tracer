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
        bool first = true;
        point3 lo(0, 0, 0), hi(0, 0, 0);
        for (int corner = 0; corner < 8; ++corner) {
            const double x = (corner & 1) ? b.x.max : b.x.min;
            const double y = (corner & 2) ? b.y.max : b.y.min;
            const double z = (corner & 4) ? b.z.max : b.z.min;
            const point3 p = apply_point(o2w, point3(x, y, z));
            if (first) { lo = hi = p; first = false; continue; }
            lo = point3(std::fmin(lo.x(), p.x()), std::fmin(lo.y(), p.y()),
                        std::fmin(lo.z(), p.z()));
            hi = point3(std::fmax(hi.x(), p.x()), std::fmax(hi.y(), p.y()),
                        std::fmax(hi.z(), p.z()));
        }
        bbox = aabb(lo, hi);
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        if (!valid) return false;

        const ray local(apply_point(w2o, r.origin()),
                        apply_vector(w2o, r.direction()),   // NOT normalised - see header
                        r.time());
        if (!object->hit(local, ray_t, rec)) return false;

        rec.p = apply_point(o2w, rec.p);
        // set_face_normal() decides front/back from the ray direction, so it
        // is given the WORLD ray. Calling it again rather than transforming
        // the child's flag keeps a mirroring transform (negative determinant,
        // which flips winding) consistent with the geometry the camera sees.
        vec3 n = apply_normal(w2o, rec.normal);
        const double len = n.length();
        if (len > 0) n = n / len;
        rec.set_face_normal(r, n);
        return true;
    }

    aabb bounding_box() const override { return bbox; }

  private:
    static point3 apply_point(const pbrt_scene::Matrix4& m, const point3& p) {
        return point3(
            m.m[0] * p.x() + m.m[1] * p.y() + m.m[2]  * p.z() + m.m[3],
            m.m[4] * p.x() + m.m[5] * p.y() + m.m[6]  * p.z() + m.m[7],
            m.m[8] * p.x() + m.m[9] * p.y() + m.m[10] * p.z() + m.m[11]);
    }

    // A direction, so the translation column is deliberately not applied.
    static vec3 apply_vector(const pbrt_scene::Matrix4& m, const vec3& v) {
        return vec3(
            m.m[0] * v.x() + m.m[1] * v.y() + m.m[2]  * v.z(),
            m.m[4] * v.x() + m.m[5] * v.y() + m.m[6]  * v.z(),
            m.m[8] * v.x() + m.m[9] * v.y() + m.m[10] * v.z());
    }

    // Object -> world for a normal is the TRANSPOSE of world -> object, which
    // is why this takes w2o and reads it column-wise.
    static vec3 apply_normal(const pbrt_scene::Matrix4& w2o, const vec3& n) {
        return vec3(
            w2o.m[0] * n.x() + w2o.m[4] * n.y() + w2o.m[8]  * n.z(),
            w2o.m[1] * n.x() + w2o.m[5] * n.y() + w2o.m[9]  * n.z(),
            w2o.m[2] * n.x() + w2o.m[6] * n.y() + w2o.m[10] * n.z());
    }

    std::shared_ptr<hittable> object;
    pbrt_scene::Matrix4 o2w;
    pbrt_scene::Matrix4 w2o;
    bool valid = false;
    aabb bbox;
};
