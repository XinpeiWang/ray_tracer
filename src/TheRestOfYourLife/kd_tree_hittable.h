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
// project's own hit_record. KdHittablePrim below bridges this EXACTLY the way
// bvh_aggregate_hittable.h's own HittablePrim does - see that file's own top
// comment for the full rationale (a second hit() call is wrong for a
// stochastic hittable like constant_medium/grid_medium_hittable/
// rgb_grid_medium_hittable, which samples a free-path scattering distance
// via random_double() INSIDE hit() itself; caching the ORIGINAL call's own
// result instead of re-querying sidesteps that entirely). thread_local for
// the identical reason bvh_aggregate_hittable.h's own cache is: many rays
// traced concurrently across per-scanline worker threads against the SAME
// built tree, safe because this project's control flow never re-enters a
// hit() call on the same top-level accelerator from within another one
// still in progress on the same thread.
//==============================================================================================

#include "hittable.h"
#include "hittable_list.h"
#include "../shared/kd_tree.h"

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace kd_tree_hittable_detail {
// See this file's own top comment for why this must be thread_local and why
// that's safe for this project's render loop. Cleared at the start of every
// top-level kd_tree_hittable::hit() call (not per-Prim-test), and written to
// by every leaf primitive KdHittablePrim::intersect() actually tests during
// that ONE traversal - at most once per original index, since a built
// KdTree partitions each original primitive into exactly one leaf.
inline thread_local std::unordered_map<int, hit_record> g_lastHitCache;
}

// ---------------------------------------------------------------------------
// KdHittablePrim -- KdTree<double, Prim> duck-typed primitive wrapping one
// shared_ptr<hittable>. original_index is assigned before build() (which
// reorders KdTree's own internal primitive array), so
// kd_tree_hittable::hit() can map a returned KdHit::prim_id back to the
// cached full hit_record for the correct original object - see this file's
// own top comment. Mirrors bvh_aggregate_hittable.h's own HittablePrim
// exactly (renamed here only to avoid a global-scope name collision when
// both headers land in the same translation unit), just against
// KdHit<double> instead of BvhHit<double>.
// ---------------------------------------------------------------------------
struct KdHittablePrim {
    std::shared_ptr<hittable> obj;
    int original_index = 0;

    void bbox(double out_min[3], double out_max[3]) const {
        aabb b = obj->bounding_box();
        out_min[0] = b.x.min; out_min[1] = b.y.min; out_min[2] = b.z.min;
        out_max[0] = b.x.max; out_max[1] = b.y.max; out_max[2] = b.z.max;
    }

    std::optional<KdHit<double>> intersect(const double org[3], const double dir[3],
                                            double t_min, double t_max) const {
        // time=0: correct for every stationary shape (the only kind ever
        // routed through this wrapper - flatten() falls the "kdtree"
        // accelerator type back to "bvh" on any scene with object motion
        // blur, same as the non-"sah" splitmethod case - see that fallback's
        // own comment).
        ray r(point3(org[0], org[1], org[2]), vec3(dir[0], dir[1], dir[2]), 0.0);
        hit_record rec;
        if (!obj->hit(r, interval(t_min, t_max), rec)) return std::nullopt;
        kd_tree_hittable_detail::g_lastHitCache[original_index] = rec;
        KdHit<double> h;
        h.t = rec.t;
        h.nx = rec.normal.x(); h.ny = rec.normal.y(); h.nz = rec.normal.z();
        h.u = rec.u; h.v = rec.v;
        h.prim_id = original_index;
        return h;
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
    kd_tree_hittable(const hittable_list& list, int isect_cost, int traversal_cost,
                      double empty_bonus, int max_prims_leaf, int max_depth) {
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
        KdTree<double, KdHittablePrim>::Params params;
        params.isect_cost = isect_cost;
        params.traversal_cost = traversal_cost;
        params.empty_bonus = empty_bonus;
        params.max_prims_leaf = max_prims_leaf;
        params.max_depth = max_depth;
        tree_.build(prims, params);
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
        if (it == cache.end()) return false;  // should not happen
        rec = it->second;
        return true;
    }

    aabb bounding_box() const override { return bbox_; }

    // Accessor for the --spectral material-scan walker (cpu_interface.cpp) -
    // mirrors bvh_aggregate_hittable::get_prims()'s own identical pattern.
    const std::vector<std::shared_ptr<hittable>>& get_prims() const { return original_objects_; }

  private:
    KdTree<double, KdHittablePrim> tree_;
    std::vector<std::shared_ptr<hittable>> original_objects_;
    aabb bbox_;
};

#endif
