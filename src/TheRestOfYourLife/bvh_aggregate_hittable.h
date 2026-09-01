#ifndef BVH_AGGREGATE_HITTABLE_H
#define BVH_AGGREGATE_HITTABLE_H
//==============================================================================================
// bvh_aggregate_hittable.h -- hittable wrapper around src/shared/bvh_aggregate.h's BvhTree
//
// bvh_node (bvh.h) is this project's own pre-existing, real SAH BVH (pbrt-v4
// BVHAggregate section 7.3) and is already what every scene uses by default -
// SAH is also pbrt-v4's own real Accelerator "bvh" default, so a scene with
// no Accelerator directive (or an explicit "splitmethod" "sah") already gets
// correct, real SAH behavior without this file. This wrapper exists only to
// honor an EXPLICIT non-default splitmethod ("middle"/"equal"/"hlbvh"): same
// primitives, same converged image, different tree-build strategy (a
// build-speed/tree-shape choice, not a rendering-behavior one) - see
// pbrt_flatten.h's FlatScene::acceleratorSplitMethod comment.
//
// BvhTree<T,Prim>'s own duck-typed Prim concept returns a slim BvhHit<T>
// (t, normal[3], uv[2], prim_id) from intersect() - far less than this
// project's own hit_record (p, normal, dpdu, dpdv, mat, t, u, v, front_face,
// texture-footprint derivatives). HittablePrim below bridges this the
// standard way for a slim-interface acceleration structure wrapping a
// richer one: intersect() calls the real hittable::hit() once (to get a
// correct t/normal/uv for the BVH's own traversal decision), and
// bvh_aggregate_hittable::hit() below re-calls hit() a SECOND time - but
// only on the single WINNING primitive, an O(1) primitive test, not a
// second tree traversal - to repopulate the full hit_record. This is
// deliberate, not accidental duplication: BvhHit has no room for
// mat/dpdu/dpdv, and stashing them via a mutable side-channel on the shared,
// once-built HittablePrim would be a data race across this project's
// parallel per-scanline render threads (every thread queries the same BVH
// concurrently).
//
// Known, disclosed scope cut: BvhTree::intersect()'s org[3]/dir[3] signature
// carries no ray time, so a scene with object motion blur (pbrt_flatten.h's
// Sphere::center1 != Sphere::center) cannot be correctly routed through this
// wrapper - pbrt_cpu_builder.h checks for this and falls back to bvh_node
// (with a warning) rather than silently rendering a moving object frozen at
// time 0. Modifying bvh_aggregate.h itself to thread a time parameter
// through was deliberately avoided, to keep that already-tested,
// self-contained template untouched.
//==============================================================================================

#include "hittable.h"
#include "hittable_list.h"
#include "../shared/bvh_aggregate.h"

#include <memory>
#include <optional>
#include <vector>

// ---------------------------------------------------------------------------
// HittablePrim -- BvhTree<double, Prim> duck-typed primitive wrapping one
// shared_ptr<hittable>. original_index is assigned before build() (which
// reorders BvhTree's own internal primitive array), so bvh_aggregate_
// hittable::hit() can map a returned BvhHit::prim_id back to the correct
// original object for the full-hit_record re-query - see this file's own
// top comment.
// ---------------------------------------------------------------------------
struct HittablePrim {
    std::shared_ptr<hittable> obj;
    int original_index = 0;

    void bbox(double out_min[3], double out_max[3]) const {
        aabb b = obj->bounding_box();
        out_min[0] = b.x.min; out_min[1] = b.y.min; out_min[2] = b.z.min;
        out_max[0] = b.x.max; out_max[1] = b.y.max; out_max[2] = b.z.max;
    }

    std::optional<BvhHit<double>> intersect(const double org[3], const double dir[3],
                                             double t_min, double t_max) const {
        // time=0: correct for every stationary shape (the only kind ever
        // routed through this wrapper - see this file's top comment on the
        // object-motion-blur scope cut enforced by the caller).
        ray r(point3(org[0], org[1], org[2]), vec3(dir[0], dir[1], dir[2]), 0.0);
        hit_record rec;
        if (!obj->hit(r, interval(t_min, t_max), rec)) return std::nullopt;
        BvhHit<double> h;
        h.t = rec.t;
        h.normal[0] = rec.normal.x(); h.normal[1] = rec.normal.y(); h.normal[2] = rec.normal.z();
        h.uv[0] = rec.u; h.uv[1] = rec.v;
        h.prim_id = original_index;
        return h;
    }

    bool intersect_p(const double org[3], const double dir[3], double t_max) const {
        ray r(point3(org[0], org[1], org[2]), vec3(dir[0], dir[1], dir[2]), 0.0);
        hit_record rec;
        return obj->hit(r, interval(0.001, t_max), rec);
    }
};

// ---------------------------------------------------------------------------
// bvh_aggregate_hittable -- hittable adapter around BvhTree<double, HittablePrim>
// ---------------------------------------------------------------------------
class bvh_aggregate_hittable : public hittable {
  public:
    bvh_aggregate_hittable(const hittable_list& list, BvhSplitMethod split_method,
                            int max_prims_in_node) {
        original_objects_ = list.objects;
        bbox_ = aabb::empty;
        std::vector<HittablePrim> prims;
        prims.reserve(list.objects.size());
        for (size_t i = 0; i < list.objects.size(); ++i) {
            bbox_ = aabb(bbox_, list.objects[i]->bounding_box());
            HittablePrim p;
            p.obj = list.objects[i];
            p.original_index = static_cast<int>(i);
            prims.push_back(std::move(p));
        }
        tree_.build(std::move(prims), max_prims_in_node, split_method);
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        double org[3] = {r.origin().x(), r.origin().y(), r.origin().z()};
        double dir[3] = {r.direction().x(), r.direction().y(), r.direction().z()};
        auto hit = tree_.intersect(org, dir, ray_t.min, ray_t.max);
        if (!hit) return false;
        // Re-query the single winning primitive (see this file's top comment)
        // to repopulate the full hit_record (mat/dpdu/dpdv/p/front_face) that
        // BvhHit's slim result can't carry - deterministic: same ray, same
        // shape, a tight interval around the already-found root re-finds the
        // exact same surface point.
        if (hit->prim_id < 0 || hit->prim_id >= static_cast<int>(original_objects_.size()))
            return false;
        double eps = 1e-6 * (1.0 + std::abs(hit->t));
        return original_objects_[hit->prim_id]->hit(
            r, interval(ray_t.min, hit->t + eps), rec);
    }

    aabb bounding_box() const override { return bbox_; }

    // Accessor for the --spectral material-scan walker (cpu_interface.cpp) -
    // mirrors bvh_leaf::get_prims()'s own identical pattern (bvh.h).
    const std::vector<std::shared_ptr<hittable>>& get_prims() const { return original_objects_; }

  private:
    BvhTree<double, HittablePrim> tree_;
    std::vector<std::shared_ptr<hittable>> original_objects_;
    aabb bbox_;
};

#endif
