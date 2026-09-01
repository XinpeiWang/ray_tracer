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
// texture-footprint derivatives). HittablePrim below bridges this by having
// intersect() cache the FULL hit_record it already computed (to get a
// correct t/normal/uv for the BVH's own traversal decision) into a
// thread_local scratch map, keyed by this primitive's original index;
// bvh_aggregate_hittable::hit() below looks the winning entry back up from
// there instead of re-calling hittable::hit() a second time.
//
// A second call was the first design tried here, and is WRONG: several
// hittables this wrapper can receive (constant_medium, grid_medium_hittable/
// rgb_grid_medium_hittable - any pbrt MediumInterface-wrapped shape) sample a
// free-path scattering distance via random_double() INSIDE hit() itself, so
// a second independent call draws a fresh random sample - it can legitimately
// return a different t (silently rendering a different scatter point than
// what the BVH traversal decided) or fail to reproduce a hit at all (an
// exponential free-path re-draw lands beyond the now-tightened interval on
// roughly half of all such rays), silently punching holes through a
// participating medium. Caching the ORIGINAL call's own result sidesteps
// this entirely - it's simply never re-evaluated.
//
// thread_local, not a mutable member on the (shared, built-once) HittablePrim
// itself, because this project's render loop traces many rays concurrently
// across per-scanline worker threads against the SAME built BVH - a shared
// mutable cache would be a data race. Safe as thread_local specifically
// because this project's control flow never re-enters a hit() call on the
// same top-level accelerator from within another one still in progress on
// the same thread (a primary ray's world->hit() always fully returns before
// any shadow/NEE ray for the same bounce is cast) - see the two write/read
// sites below for the exact contract this relies on.
//==============================================================================================

#include "hittable.h"
#include "hittable_list.h"
#include "../shared/bvh_aggregate.h"

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace bvh_aggregate_hittable_detail {
// See this file's own top comment for why this must be thread_local and why
// that's safe for this project's render loop. Cleared at the start of every
// top-level bvh_aggregate_hittable::hit() call (not per-Prim-test), and
// written to by every leaf primitive HittablePrim::intersect() actually
// tests during that ONE traversal - at most once per original index, since a
// built BvhTree partitions each original primitive into exactly one leaf.
inline thread_local std::unordered_map<int, hit_record> g_lastHitCache;
}

// ---------------------------------------------------------------------------
// HittablePrim -- BvhTree<double, Prim> duck-typed primitive wrapping one
// shared_ptr<hittable>. original_index is assigned before build() (which
// reorders BvhTree's own internal primitive array), so
// bvh_aggregate_hittable::hit() can map a returned BvhHit::prim_id back to
// the cached full hit_record for the correct original object - see this
// file's own top comment.
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
        bvh_aggregate_hittable_detail::g_lastHitCache[original_index] = rec;
        BvhHit<double> h;
        h.t = rec.t;
        h.normal[0] = rec.normal.x(); h.normal[1] = rec.normal.y(); h.normal[2] = rec.normal.z();
        h.uv[0] = rec.u; h.uv[1] = rec.v;
        h.prim_id = original_index;
        return h;
    }

    // Required by BvhTree's Prim concept (see bvh_aggregate.h's own doc
    // comment), but currently unreached in practice: bvh_aggregate_hittable
    // below only ever calls tree_.intersect() (closest-hit), never
    // tree_.intersect_p() - this project's hittable interface has no
    // separate any-hit/shadow method for a BVH-level fast path to serve
    // (shadow rays reuse the same hit() through shadow_ray.h). Kept for
    // concept-completeness and in case a future caller wants the any-hit
    // fast path; the same stochastic-hittable caveat HittablePrim::
    // intersect()'s own top-of-file comment documents would apply here too
    // if it ever becomes reachable (a hit found here is never cached/
    // reused - it only reports true/false, so that specific pitfall doesn't
    // apply to this method itself, but a caller building on top of it still
    // could not treat "intersect_p found something at t_max" as reusable
    // state for a later closest-hit query).
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
        auto& cache = bvh_aggregate_hittable_detail::g_lastHitCache;
        cache.clear();
        auto hit = tree_.intersect(org, dir, ray_t.min, ray_t.max);
        if (!hit) return false;
        // The winning primitive's full hit_record was already cached by
        // HittablePrim::intersect() above, during this same traversal - see
        // this file's own top comment for why that beats a second call.
        auto it = cache.find(hit->prim_id);
        if (it == cache.end()) return false;  // should not happen
        rec = it->second;
        return true;
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
