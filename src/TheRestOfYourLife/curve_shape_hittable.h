#ifndef CURVE_SHAPE_HITTABLE_H
#define CURVE_SHAPE_HITTABLE_H

#include "hittable.h"
#include "../shared/shapes.h"

// Wraps one CurveShape<double> segment (a cubic Bezier hair/fiber strand)
// as a hittable. This is real curve geometry - a genuine ray-Bezier
// intersection test - unlike scene B11 (Hair Fibers), which applies
// HairBxDF shading to ordinary spheres rather than curving the geometry
// itself. See CurveShape's header comment in shapes.h for the algorithm
// (mirrors pbrt-v4 Curve: ray-facing frame + recursive subdivision).
//
// CPU only: OptiX has hand-written custom-intersection programs for every
// other non-triangle primitive here (sphere, quad, bilinear patch), but
// none yet for curves - CurveShape's math is CPU_GPU-tagged and could run
// on device, but the missing piece is the OptiX-side plumbing (AABB
// build input, __intersection__ program, hit attribute passing) that
// would let it. That is real, separate engineering, not a port of this
// file.
class curve_shape_hittable : public hittable {
  public:
    curve_shape_hittable(const CurveShape<double>& curve, shared_ptr<material> mat)
      : curve(curve), mat(mat)
    {
        // Conservative bound: the 4 control points' hull, padded by half
        // the widest cross-section. CurveShape has no bounds() of its own
        // (only area()/intersect()) - hit() below does the exact test
        // regardless, so this only needs to be a safe superset for the
        // BVH, the same relationship pbrt's own Curve::Bounds() has to
        // Curve::Intersect().
        double half_w = 0.5 * std::max(curve.width0, curve.width1);
        point3 lo(+infinity, +infinity, +infinity);
        point3 hi(-infinity, -infinity, -infinity);
        for (int i = 0; i < 4; ++i) {
            lo = point3(std::fmin(lo.x(), curve.cpx[i]),
                        std::fmin(lo.y(), curve.cpy[i]),
                        std::fmin(lo.z(), curve.cpz[i]));
            hi = point3(std::fmax(hi.x(), curve.cpx[i]),
                        std::fmax(hi.y(), curve.cpy[i]),
                        std::fmax(hi.z(), curve.cpz[i]));
        }
        vec3 pad(half_w, half_w, half_w);
        bbox = aabb(lo - pad, hi + pad);
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        CurveShape<double>::CurveHit ch;
        bool got = curve.intersect(
            r.origin().x(), r.origin().y(), r.origin().z(),
            r.direction().x(), r.direction().y(), r.direction().z(),
            ray_t.min, ray_t.max, &ch);
        if (!got)
            return false;

        rec.t = ch.t;
        rec.p = r.at(rec.t);
        vec3 outward_normal = unit_vector(vec3(ch.nx, ch.ny, ch.nz));
        rec.set_face_normal(r, outward_normal);
        rec.u = ch.u;
        rec.v = ch.v;
        rec.dpdu = unit_vector(vec3(ch.dpdu_x, ch.dpdu_y, ch.dpdu_z));
        rec.mat = mat;
        return true;
    }

    aabb bounding_box() const override { return bbox; }

  private:
    CurveShape<double> curve;
    shared_ptr<material> mat;
    aabb bbox;
};

#endif
