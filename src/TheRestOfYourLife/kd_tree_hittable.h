#ifndef KD_TREE_HITTABLE_H
#define KD_TREE_HITTABLE_H
//==============================================================================================
// kd_tree_hittable.h -- hittable wrapper around src/shared/kd_tree.h's KdTree
//
// bvh_node (bvh.h) is this project's own pre-existing, real SAH BVH and is
// what every scene uses by default - pbrt-v4's own Accelerator "bvh" is also
// its own real default. This wrapper exists to honor an EXPLICIT
// Accelerator "kdtree" directive with a real, different acceleration
// structure (src/shared/kd_tree.h's own KdTree<double,...>, a faithful port
// of pbrt-v4's real KdTreeAggregate - see that file's own top comment) -
// same primitives, same converged image, just a different traversal
// structure, exactly like bvh_aggregate_hittable.h's own non-"sah"
// splitmethod wrapper is for "bvh".
//
// KdTree<T,Prim>'s own duck-typed Prim concept returns a slim KdHit<T>
// (t, nx/ny/nz, u/v, prim_id) from intersect() - far less than this
// project's own hit_record. KdHittablePrim below bridges this the way
// bvh_aggregate_hittable.h's own HittablePrim does - see that file's own top
// comment for the base rationale (a second hit() call is wrong for a
// stochastic hittable like constant_medium/grid_medium_hittable/
// rgb_grid_medium_hittable, which samples a free-path scattering distance
// via random_double() INSIDE hit() itself; caching a result instead of
// re-querying sidesteps that).
//
// One IMPORTANT way this differs from bvh_aggregate_hittable.h's cache,
// found by code review: BvhTree guarantees at most one leaf per primitive,
// so HittablePrim::intersect() there is provably called at most once per
// primitive per traversal - a plain "cache the first result" is airtight.
// KdTree does NOT have that guarantee: build_tree()'s own SAH split
// classification (src/shared/kd_tree.h) can put a primitive whose bounding
// box straddles the chosen split plane into BOTH children, so ONE primitive
// can be tested from TWO different leaves within a single traversal. Simply
// caching-and-reusing "the first answer" would still call obj->hit() a
// second, independent time on the second leaf visit - exactly the bug this
// pattern exists to prevent, just moved one level down. KdHittablePrim::
// intersect() below instead checks the cache FIRST: if this original_index
// was already queried this traversal (hit OR miss - both are cached, via
// std::optional<hit_record>), it reuses that outcome, range-checked against
// the CURRENT [t_min, t_max] rather than blindly trusted. That range check
// is always valid, never a second sample: within one top-level
// kd_tree_hittable::hit() call, t_min is fixed and t_max only ever narrows
// as a closer best-hit is found (see hit()'s own loop), so the interval a
// primitive is re-tested with is always a subset of the one its cached
// result was drawn against - narrowing an already-drawn outcome's valid
// window is a deterministic range check, not a fresh stochastic draw.
//
// thread_local for the identical reason bvh_aggregate_hittable.h's own
// cache is: many rays traced concurrently across per-scanline worker
// threads against the SAME built tree, safe because this project's control
// flow never re-enters a hit() call on the same top-level accelerator from
// within another one still in progress on the same thread.
//==============================================================================================

#include "hittable.h"
#include "hittable_list.h"
#include "../shared/kd_tree.h"

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace kd_tree_hittable_detail {
// See this file's own top comment for why this must be thread_local, why
// that's safe for this project's render loop, and why the value type is
// std::optional<hit_record> (present-but-nullopt = "already queried this
// traversal, was a miss" - a real, deliberately-cached outcome, not the
// same as "absent = never queried"). Cleared at the start of every
// top-level kd_tree_hittable::hit() call (not per-Prim-test).
inline thread_local std::unordered_map<int, std::optional<hit_record>> g_lastHitCache;
}

// ---------------------------------------------------------------------------
// KdHittablePrim -- KdTree<double, Prim> duck-typed primitive wrapping one
// shared_ptr<hittable>. original_index is assigned before build() (which
// reorders KdTree's own internal primitive array), so
// kd_tree_hittable::hit() can map a returned KdHit::prim_id back to the
// cached full hit_record for the correct original object - see this file's
// own top comment, including the hit-OR-miss caching and range-check-on-
// reuse this needs that bvh_aggregate_hittable.h's own HittablePrim does
// not. (Named differently from that file's HittablePrim only to avoid a
// global-scope name collision when both headers land in the same
// translation unit.)
// ---------------------------------------------------------------------------
struct KdHittablePrim {
    std::shared_ptr<hittable> obj;
    int original_index = 0;

    void bbox(double out_min[3], double out_max[3]) const {
        aabb b = obj->bounding_box();
        out_min[0] = b.x.min; out_min[1] = b.y.min; out_min[2] = b.z.min;
        out_max[0] = b.x.max; out_max[1] = b.y.max; out_max[2] = b.z.max;
    }

    static KdHit<double> toKdHit(const hit_record& rec, int original_index) {
        KdHit<double> h;
        h.t = rec.t;
        h.nx = rec.normal.x(); h.ny = rec.normal.y(); h.nz = rec.normal.z();
        h.u = rec.u; h.v = rec.v;
        h.prim_id = original_index;
        return h;
    }

    std::optional<KdHit<double>> intersect(const double org[3], const double dir[3],
                                            double t_min, double t_max) const {
        auto& cache = kd_tree_hittable_detail::g_lastHitCache;
        auto found = cache.find(original_index);
        if (found != cache.end()) {
            // Already queried this primitive earlier in THIS traversal (see
            // this file's own top comment for why a kd-tree, unlike
            // BvhTree, can visit one primitive from two different leaves) -
            // reuse that outcome instead of calling obj->hit() again.
            if (!found->second) return std::nullopt;  // cached miss
            const hit_record& rec = *found->second;
            if (rec.t < t_min || rec.t > t_max) return std::nullopt;
            return toKdHit(rec, original_index);
        }
        // time=0: correct for every stationary shape (the only kind ever
        // routed through this wrapper - flatten() falls the "kdtree"
        // accelerator type back to "bvh" on any scene with object motion
        // blur, same as the non-"sah" splitmethod case - see that fallback's
        // own comment).
        ray r(point3(org[0], org[1], org[2]), vec3(dir[0], dir[1], dir[2]), 0.0);
        hit_record rec;
        if (!obj->hit(r, interval(t_min, t_max), rec)) {
            cache[original_index] = std::nullopt;
            return std::nullopt;
        }
        cache[original_index] = rec;
        return toKdHit(rec, original_index);
    }

    // Required by KdTree's Prim concept (see kd_tree.h's own doc comment),
    // but currently unreached in practice - same as bvh_aggregate_hittable.h's
    // own identical method (see its own comment): kd_tree_hittable below
    // only ever calls tree_.intersect() (closest-hit), never
    // tree_.intersect_p().
    bool intersect_p(const double org[3], const double dir[3], double t_max) const {
        ray r(point3(org[0], org[1], org[2]), vec3(dir[0], dir[1], dir[2]), 0.0);
        hit_record rec;
        return obj->hit(r, interval(0.001, t_max), rec);
    }
};

// ---------------------------------------------------------------------------
// kd_tree_hittable -- hittable adapter around KdTree<double, KdHittablePrim>
// ---------------------------------------------------------------------------
class kd_tree_hittable : public hittable {
  public:
    kd_tree_hittable(const hittable_list& list, const KdTreeAccelParams& params) {
        original_objects_ = list.objects;
        bbox_ = aabb::empty;
        std::vector<KdHittablePrim> prims;
        prims.reserve(list.objects.size());
        for (size_t i = 0; i < list.objects.size(); ++i) {
            bbox_ = aabb(bbox_, list.objects[i]->bounding_box());
            KdHittablePrim p;
            p.obj = list.objects[i];
            p.original_index = static_cast<int>(i);
            prims.push_back(std::move(p));
        }
        // std::move: this constructor's own `prims` is never needed again -
        // KdTree::build() takes it by value specifically so a caller in our
        // position can hand off ownership instead of paying for an extra
        // internal copy (see that function's own comment).
        tree_.build(std::move(prims), KdTree<double, KdHittablePrim>::Params(params));
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        double org[3] = {r.origin().x(), r.origin().y(), r.origin().z()};
        double dir[3] = {r.direction().x(), r.direction().y(), r.direction().z()};
        auto& cache = kd_tree_hittable_detail::g_lastHitCache;
        cache.clear();
        auto hit = tree_.intersect(org, dir, ray_t.min, ray_t.max);
        if (!hit) return false;
        // The winning primitive's full hit_record was already cached by
        // KdHittablePrim::intersect() above, during this same traversal - see
        // this file's own top comment for why that beats a second call.
        auto it = cache.find(hit->prim_id);
        if (it == cache.end() || !it->second) return false;  // should not happen
        rec = *it->second;
        return true;
    }

    aabb bounding_box() const override { return bbox_; }

    // Accessor for cpu_interface.cpp's --spectral material-scan walker AND
    // emitter_discovery.h's BDPT/MLT/SPPM light-discovery scan - mirrors
    // bvh_aggregate_hittable::get_prims()'s own identical pattern (that
    // file's own comment is stale in the same way - it also gained a second
    // caller without its comment being updated).
    const std::vector<std::shared_ptr<hittable>>& get_prims() const { return original_objects_; }

  private:
    KdTree<double, KdHittablePrim> tree_;
    std::vector<std::shared_ptr<hittable>> original_objects_;
    aabb bbox_;
};

#endif
