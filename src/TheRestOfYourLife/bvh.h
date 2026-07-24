#ifndef BVH_H
#define BVH_H
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================
//
// SAH BVH -- Surface Area Heuristic build, following pbrt-v4 BVHAggregate (§7.3).
//
// Split strategy (per node):
//   For each of the 3 axes, divide the centroid AABB into kBuckets uniform buckets.
//   For each of the (kBuckets-1) candidate split planes, compute the SAH cost:
//
//     cost = C_trav + (SA(left)/SA(parent)) * N_left  * C_isect
//                   + (SA(right)/SA(parent)) * N_right * C_isect
//
//   C_trav = 1 (box test), C_isect = 2 (primitive test) -- standard pbrt-v4 values.
//   Pick the axis+bucket with minimum cost.
//   If min_cost >= C_isect * N (leaf is cheaper), make a leaf instead.
//   Leaf threshold: <= kMaxLeafPrims primitives are stored directly as a list.
//
// Complexity: O(N * kBuckets * 3) per level, O(N log N) overall (no per-axis sort needed).
//==============================================================================================

#include "aabb.h"
#include "hittable.h"
#include "hittable_list.h"

#include <algorithm>


// ---------------------------------------------------------------------------
// SAH constants (mirror pbrt-v4 BVHAggregate defaults)
// ---------------------------------------------------------------------------
static constexpr int    kBuckets      = 12;   // candidate split planes per axis
static constexpr int    kMaxLeafPrims = 4;    // max primitives in a leaf node
static constexpr double kCostTraversal  = 1.0; // relative cost of a box test
static constexpr double kCostIntersect  = 2.0; // relative cost of a primitive test


// ---------------------------------------------------------------------------
// Helper: surface area of an aabb  (pbrt-v4: SurfaceArea())
// ---------------------------------------------------------------------------
static inline double sah_surface_area(const aabb& b) {
    double dx = b.x.size();
    double dy = b.y.size();
    double dz = b.z.size();
    return 2.0 * (dx*dy + dy*dz + dz*dx);
}

// centroid of an aabb
static inline point3 sah_centroid(const aabb& b) {
    return point3(
        0.5 * (b.x.min + b.x.max),
        0.5 * (b.y.min + b.y.max),
        0.5 * (b.z.min + b.z.max)
    );
}


// ---------------------------------------------------------------------------
// Leaf node: holds primitives directly (avoids extra indirection for small sets)
// ---------------------------------------------------------------------------
class bvh_leaf : public hittable {
  public:
    bvh_leaf(std::vector<shared_ptr<hittable>>& objects, size_t start, size_t end) {
        bbox = aabb::empty;
        for (size_t i = start; i < end; ++i) {
            prims.push_back(objects[i]);
            bbox = aabb(bbox, objects[i]->bounding_box());
        }
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        bool hit_any = false;
        for (const auto& p : prims)
            if (p->hit(r, ray_t, rec)) { hit_any = true; ray_t.max = rec.t; }
        return hit_any;
    }

    aabb bounding_box() const override { return bbox; }

  private:
    std::vector<shared_ptr<hittable>> prims;
    aabb bbox;
};


// ---------------------------------------------------------------------------
// bvh_node: interior node with SAH build
// ---------------------------------------------------------------------------
class bvh_node : public hittable {
  public:
    // Convenience constructor from hittable_list (copies the object vector)
    bvh_node(hittable_list list) : bvh_node(list.objects, 0, list.objects.size()) {}

    bvh_node(std::vector<shared_ptr<hittable>>& objects, size_t start, size_t end) {
        size_t n = end - start;

        // Compute bounding box of all primitives in this span
        bbox = aabb::empty;
        for (size_t i = start; i < end; ++i)
            bbox = aabb(bbox, objects[i]->bounding_box());

        // Base cases: make a leaf
        if (n <= static_cast<size_t>(kMaxLeafPrims)) {
            left  = make_shared<bvh_leaf>(objects, start, end);
            right = nullptr;
            return;
        }

        // Compute centroid bounding box (pbrt-v4: centroidBounds)
        aabb centroid_bbox = aabb::empty;
        for (size_t i = start; i < end; ++i) {
            point3 c = sah_centroid(objects[i]->bounding_box());
            // Expand centroid_bbox by the centroid point (avoid std::min/max -- Windows macros)
            double cxmin = (c.x() < centroid_bbox.x.min) ? c.x() : centroid_bbox.x.min;
            double cxmax = (c.x() > centroid_bbox.x.max) ? c.x() : centroid_bbox.x.max;
            double cymin = (c.y() < centroid_bbox.y.min) ? c.y() : centroid_bbox.y.min;
            double cymax = (c.y() > centroid_bbox.y.max) ? c.y() : centroid_bbox.y.max;
            double czmin = (c.z() < centroid_bbox.z.min) ? c.z() : centroid_bbox.z.min;
            double czmax = (c.z() > centroid_bbox.z.max) ? c.z() : centroid_bbox.z.max;
            centroid_bbox = aabb(interval(cxmin, cxmax), interval(cymin, cymax), interval(czmin, czmax));
        }

        // SAH bucket sweep over all 3 axes
        struct Bucket { int count = 0; aabb bounds = aabb::empty; };

        double best_cost = 1e30;  // sentinel: larger than any real SAH cost
        int    best_axis = -1;
        int    best_bucket = -1;  // split after this bucket index
        double parent_sa = sah_surface_area(bbox);

        for (int axis = 0; axis < 3; ++axis) {
            double axis_min = centroid_bbox.axis_interval(axis).min;
            double axis_max = centroid_bbox.axis_interval(axis).max;
            double axis_size = axis_max - axis_min;

            // If all centroids are on the same plane along this axis, skip it
            if (axis_size < 1e-10) continue;

            // Assign primitives to buckets
            Bucket buckets[kBuckets];
            for (size_t i = start; i < end; ++i) {
                double c = sah_centroid(objects[i]->bounding_box())[axis];
                int b = static_cast<int>(kBuckets * (c - axis_min) / axis_size);
                if (b >= kBuckets) b = kBuckets - 1;
                buckets[b].count++;
                buckets[b].bounds = aabb(buckets[b].bounds, objects[i]->bounding_box());
            }

            // Sweep left→right: prefix SA and counts
            double left_sa[kBuckets - 1],  right_sa[kBuckets - 1];
            int    left_cnt[kBuckets - 1], right_cnt[kBuckets - 1];

            aabb left_box = aabb::empty;  int lc = 0;
            for (int i = 0; i < kBuckets - 1; ++i) {
                left_box = aabb(left_box, buckets[i].bounds);
                lc += buckets[i].count;
                left_sa[i]  = sah_surface_area(left_box);
                left_cnt[i] = lc;
            }

            aabb right_box = aabb::empty; int rc = 0;
            for (int i = kBuckets - 2; i >= 0; --i) {
                right_box = aabb(right_box, buckets[i + 1].bounds);
                rc += buckets[i + 1].count;
                right_sa[i]  = sah_surface_area(right_box);
                right_cnt[i] = rc;
            }

            // Find best split for this axis
            for (int i = 0; i < kBuckets - 1; ++i) {
                if (left_cnt[i] == 0 || right_cnt[i] == 0) continue;
                double cost = kCostTraversal
                    + (left_sa[i]  / parent_sa) * left_cnt[i]  * kCostIntersect
                    + (right_sa[i] / parent_sa) * right_cnt[i] * kCostIntersect;
                if (cost < best_cost) {
                    best_cost   = cost;
                    best_axis   = axis;
                    best_bucket = i;
                }
            }
        }

        // Leaf cost: no traversal overhead, just test all N primitives
        double leaf_cost = kCostIntersect * static_cast<double>(n);

        // If no valid split found or leaf is cheaper, make a leaf
        if (best_axis == -1 || best_cost >= leaf_cost) {
            left  = make_shared<bvh_leaf>(objects, start, end);
            right = nullptr;
            return;
        }

        // Partition objects around best split (pbrt-v4: std::partition)
        double axis_min  = centroid_bbox.axis_interval(best_axis).min;
        double axis_size = centroid_bbox.axis_interval(best_axis).max - axis_min;

        auto mid_it = std::partition(
            objects.begin() + start,
            objects.begin() + end,
            [&](const shared_ptr<hittable>& obj) {
                double c = sah_centroid(obj->bounding_box())[best_axis];
                int b = static_cast<int>(kBuckets * (c - axis_min) / axis_size);
                if (b >= kBuckets) b = kBuckets - 1;
                return b <= best_bucket;
            }
        );

        size_t mid = static_cast<size_t>(mid_it - objects.begin());

        // Guard against degenerate partition (all objects on one side)
        if (mid == start || mid == end)
            mid = start + n / 2;

        left  = make_shared<bvh_node>(objects, start, mid);
        right = make_shared<bvh_node>(objects, mid,   end);
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        if (!bbox.hit(r, ray_t))
            return false;

        bool hit_left  = left  && left->hit(r, ray_t, rec);
        // Narrow the interval for the right child if left already hit something
        bool hit_right = right && right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);

        return hit_left || hit_right;
    }

    aabb bounding_box() const override { return bbox; }

  private:
    shared_ptr<hittable> left;
    shared_ptr<hittable> right;  // nullptr when left is a leaf
    aabb bbox;
};


#endif
