#pragma once
// ---------------------------------------------------------------------------
// bvh_aggregate.h -- SAH BVH scene acceleration structure
//
// Ports pbrt-v4 BVHAggregate (cpu/aggregates.h / cpu/aggregates.cpp,
// Apache-2.0) as a self-contained, header-only C++ template.
//
// Provides:
//   BvhHit<T>           -- intersection result (t, surface normal, uv, prim_id)
//   BvhTree<T, Prim>    -- SAH BVH built from a list of duck-typed primitives
//
// Primitive concept (Prim):
//   // Fill axis-aligned bounding box.
//   void bbox(T out_min[3], T out_max[3]) const;
//
//   // Intersect ray (org, dir) in [t_min, t_max].
//   // Returns BvhHit<T> on hit, empty otherwise.
//   std::optional<BvhHit<T>>
//   intersect(const T org[3], const T dir[3], T t_min, T t_max) const;
//
//   // Shadow test: true if any intersection in (0, t_max).
//   bool intersect_p(const T org[3], const T dir[3], T t_max) const;
//   // (If not provided, intersect() is used as a fallback in intersect_p.)
//
// Usage:
//   std::vector<MyPrim> prims = ...;
//   BvhTree<float, MyPrim> bvh;
//   bvh.build(prims);
//   auto hit = bvh.intersect(org, dir, 0.0f, 1e30f);
//   bool shadow = bvh.intersect_p(org, dir, dist);
//
// Split methods (enum BvhSplitMethod):
//   SAH          -- surface area heuristic (default, best quality)
//   Middle       -- split at centroid midpoint; falls through to EqualCounts
//   EqualCounts  -- split so each child has the same number of primitives
//   HLBVH        -- Morton-code treelets + upper SAH; fast build, good quality
//
// Design rules (same as kd_tree.h):
//   - Header-only, no virtual functions, no heap allocation in hot paths
//   - Template T: float or double
//   - Template Prim: duck-typed (see concept above)
//   - Single-threaded build; traversal is read-only and thread-safe
//
// pbrt-v4 alignment:
//   bvh_detail::LinearBVHNode  <-> LinearBVHNode (32-byte flat node)
//   bvh_detail::BVHBuildNode   <-> BVHBuildNode  (build-time tree node)
//   bvh_detail::BVHPrimitive   <-> BVHPrimitive  (build-time prim wrapper)
//   BvhTree::build()           <-> BVHAggregate ctor
//   BvhTree::intersect()       <-> BVHAggregate::Intersect()
//   BvhTree::intersect_p()     <-> BVHAggregate::IntersectP()
//
// Default parameters match pbrt-v4:
//   max_prims_in_node=1, split_method=SAH
//
// Reference: pbrt-v4 src/pbrt/cpu/aggregates.h / aggregates.cpp (Apache-2.0)
//            "Physically Based Rendering: From Theory to Implementation",
//            Pharr, Jakob, Humphreys. 4th ed., §7.2.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

// ---------------------------------------------------------------------------
// BvhHit<T> -- result returned by BvhTree::intersect()
// ---------------------------------------------------------------------------
template <typename T>
struct BvhHit {
	T t{};
	T normal[3]{};
	T uv[2]{};
	int prim_id{-1};
};

// ---------------------------------------------------------------------------
// BvhSplitMethod -- selects the build strategy
// ---------------------------------------------------------------------------
enum class BvhSplitMethod { SAH, Middle, EqualCounts, HLBVH };

// ---------------------------------------------------------------------------
// Internal implementation details
// ---------------------------------------------------------------------------
namespace bvh_detail {

// ---- Axis-aligned bounding box helpers ------------------------------------

template <typename T>
struct Aabb {
	T lo[3];
	T hi[3];

	Aabb() {
		for (int i = 0; i < 3; ++i) {
			lo[i] = std::numeric_limits<T>::max();
			hi[i] = std::numeric_limits<T>::lowest();
		}
	}

	void expand(const T pt[3]) {
		for (int i = 0; i < 3; ++i) {
			lo[i] = std::min(lo[i], pt[i]);
			hi[i] = std::max(hi[i], pt[i]);
		}
	}

	bool is_empty() const { return lo[0] > hi[0]; }

	void expand(const Aabb<T>& o) {
		// Skip union with an empty (default-constructed) box — matches
		// pbrt-v4 Bounds3f::Union() identity behaviour with ±INF defaults.
		if (o.is_empty()) return;
		expand(o.lo);
		expand(o.hi);
	}

	void centroid(T out[3]) const {
		for (int i = 0; i < 3; ++i)
			out[i] = T(0.5) * (lo[i] + hi[i]);
	}

	// offset of a point in [0,1]^3
	T offset(const T pt[3], int axis) const {
		T d = hi[axis] - lo[axis];
		if (d == T(0)) return T(0);
		return (pt[axis] - lo[axis]) / d;
	}

	T surface_area() const {
		T d[3] = {hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
		T sa = T(2) * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2]);
		return (sa < T(0)) ? T(0) : sa;
	}

	int max_dim() const {
		T d[3] = {hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
		return (d[0] >= d[1] && d[0] >= d[2]) ? 0 : (d[1] >= d[2] ? 1 : 2);
	}

	// Ray-slab intersection; returns true if the ray hits [t_min, t_max]
	bool intersect_p(const T org[3], const T inv_dir[3],
					 const int dir_is_neg[3], T t_max) const {
		T t_min_v = T(0);
		for (int i = 0; i < 3; ++i) {
			T t0 = (lo[i] - org[i]) * inv_dir[i];
			T t1 = (hi[i] - org[i]) * inv_dir[i];
			if (dir_is_neg[i]) std::swap(t0, t1);
			t_min_v = std::max(t_min_v, t0);
			t_max   = std::min(t_max,   t1);
			if (t_max < t_min_v) return false;
		}
		return true;
	}
};

// ---- Build-time node -------------------------------------------------------

template <typename T>
struct BVHBuildNode {
	Aabb<T>        bounds;
	BVHBuildNode*  children[2]{nullptr, nullptr};
	int            split_axis{0};
	int            first_prim_offset{0};
	int            n_primitives{0};

	void init_leaf(int first, int n, const Aabb<T>& b) {
		first_prim_offset = first;
		n_primitives      = n;
		bounds            = b;
		children[0] = children[1] = nullptr;
	}

	void init_interior(int axis, BVHBuildNode* c0, BVHBuildNode* c1) {
		children[0]  = c0;
		children[1]  = c1;
		bounds       = c0->bounds;
		bounds.expand(c1->bounds);
		split_axis   = axis;
		n_primitives = 0;
	}
};

// ---- Build-time primitive wrapper -----------------------------------------

template <typename T>
struct BVHPrimitive {
	int      primitive_index{0};
	Aabb<T>  bounds;

	BVHPrimitive() = default;
	BVHPrimitive(int idx, const Aabb<T>& b)
		: primitive_index(idx), bounds(b) {}

	void centroid(T out[3]) const { bounds.centroid(out); }
};

// ---- Compact flat node (32 bytes, matches pbrt-v4 LinearBVHNode) -----------

template <typename T>
struct alignas(32) LinearBVHNode {
	Aabb<T> bounds;          // 24 bytes for float, 48 bytes for double

	union {
		int primitives_offset;  // leaf
		int second_child_offset; // interior
	};
	uint16_t n_primitives{0}; // 0 -> interior
	uint8_t  axis{0};         // interior: split axis
	uint8_t  pad{0};
};

// ---- SAH bucket helper ----------------------------------------------------

template <typename T>
struct SplitBucket {
	int    count{0};
	Aabb<T> bounds;
};

// ---- Morton-code helpers for HLBVH ----------------------------------------

inline uint32_t left_shift3(uint32_t x) {
	if (x == (1 << 10)) --x;
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x <<  8)) & 0x0300F00F;
	x = (x | (x <<  4)) & 0x030C30C3;
	x = (x | (x <<  2)) & 0x09249249;
	return x;
}

inline uint32_t encode_morton3(uint32_t x, uint32_t y, uint32_t z) {
	return (left_shift3(z) << 2) | (left_shift3(y) << 1) | left_shift3(x);
}

struct MortonPrimitive {
	int      primitive_index{0};
	uint32_t morton_code{0};
};

template <typename T>
struct LBVHTreelet {
	size_t          start_index{0};
	size_t          n_primitives{0};
	BVHBuildNode<T>* build_nodes{nullptr};
};

} // namespace bvh_detail

// ---------------------------------------------------------------------------
// BvhTree<T, Prim>
// ---------------------------------------------------------------------------
template <typename T, typename Prim>
class BvhTree {
public:
	using Hit  = BvhHit<T>;
	using Node = bvh_detail::LinearBVHNode<T>;

	BvhTree() = default;
	~BvhTree() = default;

	// Non-copyable, movable
	BvhTree(const BvhTree&) = delete;
	BvhTree& operator=(const BvhTree&) = delete;
	BvhTree(BvhTree&&) = default;
	BvhTree& operator=(BvhTree&&) = default;

	// Build the BVH from the given primitives.
	// max_prims_in_node: max primitives per leaf (capped at 255, default 1)
	// split_method: SAH / Middle / EqualCounts / HLBVH (default SAH)
	void build(std::vector<Prim> prims,
			   int max_prims = 1,
			   BvhSplitMethod split_method = BvhSplitMethod::SAH) {
		primitives_    = std::move(prims);
		max_prims_     = std::min(255, max_prims);
		split_method_  = split_method;
		nodes_.clear();

		if (primitives_.empty()) return;

		// Build BVHPrimitive array
		const size_t n = primitives_.size();
		std::vector<bvh_detail::BVHPrimitive<T>> bvh_prims(n);
		for (size_t i = 0; i < n; ++i) {
			bvh_detail::Aabb<T> b;
			T lo[3], hi[3];
			primitives_[i].bbox(lo, hi);
			b.expand(lo);
			b.expand(hi);
			bvh_prims[i] = bvh_detail::BVHPrimitive<T>(static_cast<int>(i), b);
		}

		std::vector<Prim> ordered_prims;
		ordered_prims.reserve(n);

		// Allocate build nodes from a flat pool (avoids many small allocs)
		build_pool_.clear();
		build_pool_.resize(2 * n);  // upper bound on number of nodes
		pool_ptr_ = 0;

		int total_nodes = 0;
		bvh_detail::BVHBuildNode<T>* root = nullptr;

		if (split_method_ == BvhSplitMethod::HLBVH) {
			ordered_prims.resize(n);
			root = build_hlbvh(bvh_prims, &total_nodes, ordered_prims);
		} else {
			int ordered_prims_offset = 0;
			ordered_prims.resize(n);
			root = build_recursive(bvh_prims, &total_nodes,
								   &ordered_prims_offset, ordered_prims);
		}

		primitives_.swap(ordered_prims);

		// Flatten into LinearBVHNode array
		nodes_.resize(total_nodes);
		int offset = 0;
		flatten_bvh(root, &offset);

		// Release build pools
		build_pool_.clear();
		build_pool_.shrink_to_fit();
		hlbvh_treelet_pools_.clear();
		hlbvh_treelet_pools_.shrink_to_fit();
	}

	bool empty() const { return nodes_.empty(); }

	// Returns the world-space AABB of all primitives
	void bounds(T out_lo[3], T out_hi[3]) const {
		if (nodes_.empty()) {
			for (int i = 0; i < 3; ++i) { out_lo[i] = T(0); out_hi[i] = T(0); }
			return;
		}
		for (int i = 0; i < 3; ++i) {
			out_lo[i] = nodes_[0].bounds.lo[i];
			out_hi[i] = nodes_[0].bounds.hi[i];
		}
	}

	// Closest-hit query.
	std::optional<Hit> intersect(const T org[3], const T dir[3],
								  T t_min, T t_max) const {
		if (nodes_.empty()) return std::nullopt;

		T inv_dir[3] = {T(1) / dir[0], T(1) / dir[1], T(1) / dir[2]};
		int dir_is_neg[3] = {inv_dir[0] < T(0) ? 1 : 0,
							  inv_dir[1] < T(0) ? 1 : 0,
							  inv_dir[2] < T(0) ? 1 : 0};

		std::optional<Hit> result;
		int nodes_to_visit[64];
		int to_visit_offset = 0, current = 0;

		while (true) {
			const Node* node = &nodes_[current];
			if (node->bounds.intersect_p(org, inv_dir, dir_is_neg, t_max)) {
				if (node->n_primitives > 0) {
					// Leaf: test each primitive
					for (int i = 0; i < node->n_primitives; ++i) {
						auto hit = primitives_[node->primitives_offset + i]
									   .intersect(org, dir, t_min, t_max);
						if (hit) {
							result = hit;
							t_max  = hit->t;
						}
					}
					if (to_visit_offset == 0) break;
					current = nodes_to_visit[--to_visit_offset];
				} else {
					// Interior: visit near child first
					if (dir_is_neg[node->axis]) {
						nodes_to_visit[to_visit_offset++] = current + 1;
						current = node->second_child_offset;
					} else {
						nodes_to_visit[to_visit_offset++] = node->second_child_offset;
						current = current + 1;
					}
				}
			} else {
				if (to_visit_offset == 0) break;
				current = nodes_to_visit[--to_visit_offset];
			}
		}
		return result;
	}

	// Shadow (any-hit) query.
	bool intersect_p(const T org[3], const T dir[3], T t_max) const {
		if (nodes_.empty()) return false;

		T inv_dir[3] = {T(1) / dir[0], T(1) / dir[1], T(1) / dir[2]};
		int dir_is_neg[3] = {inv_dir[0] < T(0) ? 1 : 0,
							  inv_dir[1] < T(0) ? 1 : 0,
							  inv_dir[2] < T(0) ? 1 : 0};

		int nodes_to_visit[64];
		int to_visit_offset = 0, current = 0;

		while (true) {
			const Node* node = &nodes_[current];
			if (node->bounds.intersect_p(org, inv_dir, dir_is_neg, t_max)) {
				if (node->n_primitives > 0) {
					for (int i = 0; i < node->n_primitives; ++i) {
						if (primitives_[node->primitives_offset + i]
								.intersect_p(org, dir, t_max)) return true;
					}
					if (to_visit_offset == 0) break;
					current = nodes_to_visit[--to_visit_offset];
				} else {
					if (dir_is_neg[node->axis]) {
						nodes_to_visit[to_visit_offset++] = current + 1;
						current = node->second_child_offset;
					} else {
						nodes_to_visit[to_visit_offset++] = node->second_child_offset;
						current = current + 1;
					}
				}
			} else {
				if (to_visit_offset == 0) break;
				current = nodes_to_visit[--to_visit_offset];
			}
		}
		return false;
	}

private:
	// -------------------------------------------------------------------------
	// Build helpers
	// -------------------------------------------------------------------------

	bvh_detail::BVHBuildNode<T>* alloc_node() {
		assert(pool_ptr_ < build_pool_.size());
		auto* n = &build_pool_[pool_ptr_++];
		*n = bvh_detail::BVHBuildNode<T>{};
		return n;
	}

	// Recursive SAH / Middle / EqualCounts build (mirrors pbrt-v4 buildRecursive)
	bvh_detail::BVHBuildNode<T>* build_recursive(
		std::vector<bvh_detail::BVHPrimitive<T>>& bvh_prims_span,
		int* total_nodes,
		int* ordered_prims_offset,
		std::vector<Prim>& ordered_prims,
		int span_start = 0, int span_end = -1)
	{
		if (span_end < 0) span_end = static_cast<int>(bvh_prims_span.size());
		int n = span_end - span_start;
		assert(n > 0);

		auto* node = alloc_node();
		++(*total_nodes);

		// Compute total bounds
		bvh_detail::Aabb<T> bounds;
		for (int i = span_start; i < span_end; ++i)
			bounds.expand(bvh_prims_span[i].bounds);

		if (bounds.surface_area() == T(0) || n == 1) {
			// Create leaf
			int first = *ordered_prims_offset;
			*ordered_prims_offset += n;
			for (int i = span_start; i < span_end; ++i)
				ordered_prims[first + (i - span_start)] =
					primitives_[bvh_prims_span[i].primitive_index];
			node->init_leaf(first, n, bounds);
			return node;
		}

		// Centroid bounds
		bvh_detail::Aabb<T> centroid_bounds;
		T c[3];
		for (int i = span_start; i < span_end; ++i) {
			bvh_prims_span[i].centroid(c);
			centroid_bounds.expand(c);
		}
		int dim = centroid_bounds.max_dim();

		if (centroid_bounds.hi[dim] == centroid_bounds.lo[dim]) {
			// All centroids coincide: create leaf
			int first = *ordered_prims_offset;
			*ordered_prims_offset += n;
			for (int i = span_start; i < span_end; ++i)
				ordered_prims[first + (i - span_start)] =
					primitives_[bvh_prims_span[i].primitive_index];
			node->init_leaf(first, n, bounds);
			return node;
		}

		int mid = span_start + n / 2;

		switch (split_method_) {
		case BvhSplitMethod::Middle: {
			T pmid = T(0.5) * (centroid_bounds.lo[dim] + centroid_bounds.hi[dim]);
			auto* split_it = std::partition(
				bvh_prims_span.data() + span_start,
				bvh_prims_span.data() + span_end,
				[dim, pmid](const bvh_detail::BVHPrimitive<T>& p) {
					T cc[3]; p.centroid(const_cast<T*>(cc));
					return cc[dim] < pmid;
				});
			mid = static_cast<int>(split_it - bvh_prims_span.data());
			if (mid == span_start || mid == span_end) {
				// Fall through to EqualCounts
				mid = span_start + n / 2;
				std::nth_element(
					bvh_prims_span.data() + span_start,
					bvh_prims_span.data() + mid,
					bvh_prims_span.data() + span_end,
					[dim](const bvh_detail::BVHPrimitive<T>& a,
						  const bvh_detail::BVHPrimitive<T>& b) {
						T ca[3], cb[3];
						a.centroid(const_cast<T*>(ca));
						b.centroid(const_cast<T*>(cb));
						return ca[dim] < cb[dim];
					});
			}
			break;
		}
		case BvhSplitMethod::EqualCounts: {
			mid = span_start + n / 2;
			std::nth_element(
				bvh_prims_span.data() + span_start,
				bvh_prims_span.data() + mid,
				bvh_prims_span.data() + span_end,
				[dim](const bvh_detail::BVHPrimitive<T>& a,
					  const bvh_detail::BVHPrimitive<T>& b) {
					T ca[3], cb[3];
					a.centroid(const_cast<T*>(ca));
					b.centroid(const_cast<T*>(cb));
					return ca[dim] < cb[dim];
				});
			break;
		}
		case BvhSplitMethod::SAH:
		default: {
			if (n <= 2) {
				mid = span_start + n / 2;
				std::nth_element(
					bvh_prims_span.data() + span_start,
					bvh_prims_span.data() + mid,
					bvh_prims_span.data() + span_end,
					[dim](const bvh_detail::BVHPrimitive<T>& a,
						  const bvh_detail::BVHPrimitive<T>& b) {
						T ca[3], cb[3];
						a.centroid(const_cast<T*>(ca));
						b.centroid(const_cast<T*>(cb));
						return ca[dim] < cb[dim];
					});
			} else {
				constexpr int n_buckets = 12;
				bvh_detail::SplitBucket<T> buckets[n_buckets];

				// Fill buckets
				for (int i = span_start; i < span_end; ++i) {
					T cc[3];
					bvh_prims_span[i].centroid(cc);
					int b = static_cast<int>(
						n_buckets * centroid_bounds.offset(cc, dim));
					if (b == n_buckets) b = n_buckets - 1;
					++buckets[b].count;
					buckets[b].bounds.expand(bvh_prims_span[i].bounds);
				}

				// Swept SAH costs (pbrt-v4 two-pass prefix scan)
				constexpr int n_splits = n_buckets - 1;
				T costs[n_splits]{};

				// Forward sweep: accumulate left side
				{
					bvh_detail::Aabb<T> b_below;
					int count_below = 0;
					for (int i = 0; i < n_splits; ++i) {
						b_below.expand(buckets[i].bounds);
						count_below += buckets[i].count;
						costs[i] += static_cast<T>(count_below) * b_below.surface_area();
					}
				}
				// Backward sweep: accumulate right side
				{
					bvh_detail::Aabb<T> b_above;
					int count_above = 0;
					for (int i = n_splits; i >= 1; --i) {
						b_above.expand(buckets[i].bounds);
						count_above += buckets[i].count;
						costs[i - 1] += static_cast<T>(count_above) * b_above.surface_area();
					}
				}

				// Find minimum cost split
				int min_cost_bucket = 0;
				T min_cost = std::numeric_limits<T>::max();
				for (int i = 0; i < n_splits; ++i) {
					if (costs[i] < min_cost) {
						min_cost          = costs[i];
						min_cost_bucket   = i;
					}
				}

				T leaf_cost   = static_cast<T>(n);
				T split_cost  = T(0.5) + min_cost / bounds.surface_area();

				if (n > max_prims_ || split_cost < leaf_cost) {
					// Split at chosen bucket
					auto* split_it = std::partition(
						bvh_prims_span.data() + span_start,
						bvh_prims_span.data() + span_end,
						[&](const bvh_detail::BVHPrimitive<T>& p) {
							T cc[3]; p.centroid(const_cast<T*>(cc));
							int b = static_cast<int>(
								n_buckets * centroid_bounds.offset(cc, dim));
							if (b == n_buckets) b = n_buckets - 1;
							return b <= min_cost_bucket;
						});
					mid = static_cast<int>(split_it - bvh_prims_span.data());
				} else {
					// Create leaf
					int first = *ordered_prims_offset;
					*ordered_prims_offset += n;
					for (int i = span_start; i < span_end; ++i)
						ordered_prims[first + (i - span_start)] =
							primitives_[bvh_prims_span[i].primitive_index];
					node->init_leaf(first, n, bounds);
					return node;
				}
			}
			break;
		}
		} // switch

		auto* c0 = build_recursive(bvh_prims_span, total_nodes,
								   ordered_prims_offset, ordered_prims,
								   span_start, mid);
		auto* c1 = build_recursive(bvh_prims_span, total_nodes,
								   ordered_prims_offset, ordered_prims,
								   mid, span_end);
		node->init_interior(dim, c0, c1);
		return node;
	}

	// ---- HLBVH build (mirrors pbrt-v4 buildHLBVH + emitLBVH + buildUpperSAH)

	bvh_detail::BVHBuildNode<T>* build_hlbvh(
		std::vector<bvh_detail::BVHPrimitive<T>>& bvh_prims,
		int* total_nodes,
		std::vector<Prim>& ordered_prims)
	{
		// Compute centroid bounds of all primitives
		bvh_detail::Aabb<T> centroid_bounds;
		T c[3];
		for (auto& p : bvh_prims) {
			p.centroid(c);
			centroid_bounds.expand(c);
		}

		// Compute Morton codes
		constexpr int morton_bits  = 10;
		constexpr int morton_scale = 1 << morton_bits;
		const size_t n = bvh_prims.size();

		std::vector<bvh_detail::MortonPrimitive> morton_prims(n);
		for (size_t i = 0; i < n; ++i) {
			morton_prims[i].primitive_index = bvh_prims[i].primitive_index;
			T cc[3]; bvh_prims[i].centroid(cc);
			T offset[3];
			for (int d = 0; d < 3; ++d) {
				T range = centroid_bounds.hi[d] - centroid_bounds.lo[d];
				offset[d] = (range > T(0))
					? (cc[d] - centroid_bounds.lo[d]) / range
					: T(0);
			}
			auto fx = static_cast<uint32_t>(std::min(offset[0] * morton_scale,
													  static_cast<T>(morton_scale - 1)));
			auto fy = static_cast<uint32_t>(std::min(offset[1] * morton_scale,
													  static_cast<T>(morton_scale - 1)));
			auto fz = static_cast<uint32_t>(std::min(offset[2] * morton_scale,
													  static_cast<T>(morton_scale - 1)));
			morton_prims[i].morton_code = bvh_detail::encode_morton3(fx, fy, fz);
		}

		// Sort by Morton code (radix sort via std::sort on uint32)
		std::sort(morton_prims.begin(), morton_prims.end(),
				  [](const bvh_detail::MortonPrimitive& a,
					 const bvh_detail::MortonPrimitive& b) {
					  return a.morton_code < b.morton_code;
				  });

		// Group into treelets by upper 12 Morton bits.
		// Store node pools in the member hlbvh_treelet_pools_ so that raw pointers
		// into them remain valid through flatten_bvh() (local vectors would be destroyed).
		struct TreeletMeta {
			size_t start_index{0};
			size_t n_primitives{0};
			bvh_detail::BVHBuildNode<T>* root{nullptr};
		};
		hlbvh_treelet_pools_.clear();
		std::vector<TreeletMeta> treelets;
		{
			constexpr uint32_t mask = 0b00111111111111000000000000000000;
			for (size_t start = 0, end = 1; end <= n; ++end) {
				if (end == n ||
					((morton_prims[start].morton_code & mask) !=
					 (morton_prims[end].morton_code & mask))) {
					size_t np = end - start;
					hlbvh_treelet_pools_.emplace_back(2 * np);
					treelets.push_back({start, np, nullptr});
					start = end;
				}
			}
		}

		// Build LBVH for each treelet
		int ordered_prims_offset = 0;
		for (size_t ti = 0; ti < treelets.size(); ++ti) {
			auto& treelet = treelets[ti];
			int nodes_created = 0;
			auto* pool_begin = hlbvh_treelet_pools_[ti].data();
			treelet.root = emit_lbvh(
				pool_begin, bvh_prims,
				morton_prims.data() + treelet.start_index,
				static_cast<int>(treelet.n_primitives),
				&nodes_created, ordered_prims,
				&ordered_prims_offset, 29 - 12);
			*total_nodes += nodes_created;
		}

		// Build upper SAH over treelet roots
		std::vector<bvh_detail::BVHBuildNode<T>*> finished_treelets;
		finished_treelets.reserve(treelets.size());
		for (auto& t : treelets)
			finished_treelets.push_back(t.root);

		return build_upper_sah(finished_treelets, 0,
							   static_cast<int>(finished_treelets.size()),
							   total_nodes);
	}

	bvh_detail::BVHBuildNode<T>* emit_lbvh(
		bvh_detail::BVHBuildNode<T>*& build_nodes,
		const std::vector<bvh_detail::BVHPrimitive<T>>& bvh_prims,
		bvh_detail::MortonPrimitive* morton_prims, int n_prims,
		int* total_nodes,
		std::vector<Prim>& ordered_prims,
		int* ordered_prims_offset, int bit_index)
	{
		if (bit_index == -1 || n_prims < max_prims_) {
			++(*total_nodes);
			auto* node = build_nodes++;
			*node = bvh_detail::BVHBuildNode<T>{};
			bvh_detail::Aabb<T> bounds;
			int first = *ordered_prims_offset;
			*ordered_prims_offset += n_prims;
			for (int i = 0; i < n_prims; ++i) {
				int idx = morton_prims[i].primitive_index;
				ordered_prims[first + i] = primitives_[idx];
				bounds.expand(bvh_prims[idx].bounds);
			}
			node->init_leaf(first, n_prims, bounds);
			return node;
		}

		uint32_t mask = 1u << bit_index;
		// If all prims have the same bit, recurse with lower bit
		if ((morton_prims[0].morton_code & mask) ==
			(morton_prims[n_prims - 1].morton_code & mask)) {
			return emit_lbvh(build_nodes, bvh_prims, morton_prims, n_prims,
							 total_nodes, ordered_prims, ordered_prims_offset,
							 bit_index - 1);
		}

		// Binary search for split
		int lo = 0, hi = n_prims - 1;
		while (lo + 1 < hi) {
			int m = (lo + hi) / 2;
			if ((morton_prims[lo].morton_code & mask) ==
				(morton_prims[m].morton_code & mask))
				lo = m;
			else
				hi = m;
		}
		int split_offset = hi;

		++(*total_nodes);
		auto* node = build_nodes++;
		*node = bvh_detail::BVHBuildNode<T>{};
		auto* lbvh0 = emit_lbvh(build_nodes, bvh_prims, morton_prims,
								  split_offset, total_nodes, ordered_prims,
								  ordered_prims_offset, bit_index - 1);
		auto* lbvh1 = emit_lbvh(build_nodes, bvh_prims,
								  morton_prims + split_offset,
								  n_prims - split_offset, total_nodes,
								  ordered_prims, ordered_prims_offset,
								  bit_index - 1);
		int axis = bit_index % 3;
		node->init_interior(axis, lbvh0, lbvh1);
		return node;
	}

	// Upper SAH over treelet roots (mirrors pbrt-v4 buildUpperSAH)
	bvh_detail::BVHBuildNode<T>* build_upper_sah(
		std::vector<bvh_detail::BVHBuildNode<T>*>& treelet_roots,
		int start, int end, int* total_nodes)
	{
		int n = end - start;
		if (n == 1) return treelet_roots[start];

		++(*total_nodes);
		auto* node = alloc_node();

		bvh_detail::Aabb<T> bounds;
		for (int i = start; i < end; ++i)
			bounds.expand(treelet_roots[i]->bounds);

		bvh_detail::Aabb<T> centroid_bounds;
		T c[3];
		for (int i = start; i < end; ++i) {
			c[0] = T(0.5) * (treelet_roots[i]->bounds.lo[0] + treelet_roots[i]->bounds.hi[0]);
			c[1] = T(0.5) * (treelet_roots[i]->bounds.lo[1] + treelet_roots[i]->bounds.hi[1]);
			c[2] = T(0.5) * (treelet_roots[i]->bounds.lo[2] + treelet_roots[i]->bounds.hi[2]);
			centroid_bounds.expand(c);
		}
		int dim = centroid_bounds.max_dim();

		constexpr int n_buckets = 12;
		bvh_detail::SplitBucket<T> buckets[n_buckets];
		for (int i = start; i < end; ++i) {
			T centroid = T(0.5) * (treelet_roots[i]->bounds.lo[dim] +
								   treelet_roots[i]->bounds.hi[dim]);
			T denom = centroid_bounds.hi[dim] - centroid_bounds.lo[dim];
			int b = (denom > T(0))
				? static_cast<int>(n_buckets * (centroid - centroid_bounds.lo[dim]) / denom)
				: 0;
			if (b == n_buckets) b = n_buckets - 1;
			++buckets[b].count;
			buckets[b].bounds.expand(treelet_roots[i]->bounds);
		}

		constexpr int n_splits = n_buckets - 1;
		T costs[n_splits]{};
		for (int i = 0; i < n_splits; ++i) {
			bvh_detail::Aabb<T> b0, b1;
			int c0 = 0, c1 = 0;
			for (int j = 0;     j <= i;        ++j) { b0.expand(buckets[j].bounds); c0 += buckets[j].count; }
			for (int j = i + 1; j < n_buckets; ++j) { b1.expand(buckets[j].bounds); c1 += buckets[j].count; }
			costs[i] = T(0.125) + (c0 * b0.surface_area() + c1 * b1.surface_area()) /
								   bounds.surface_area();
		}

		T min_cost = costs[0];
		int min_bucket = 0;
		for (int i = 1; i < n_splits; ++i) {
			if (costs[i] < min_cost) { min_cost = costs[i]; min_bucket = i; }
		}

		auto* mid_ptr = std::partition(
			treelet_roots.data() + start,
			treelet_roots.data() + end,
			[&](const bvh_detail::BVHBuildNode<T>* nd) {
				T centroid = T(0.5) * (nd->bounds.lo[dim] + nd->bounds.hi[dim]);
				T denom = centroid_bounds.hi[dim] - centroid_bounds.lo[dim];
				int b = (denom > T(0))
					? static_cast<int>(n_buckets * (centroid - centroid_bounds.lo[dim]) / denom)
					: 0;
				if (b == n_buckets) b = n_buckets - 1;
				return b <= min_bucket;
			});
		int mid = static_cast<int>(mid_ptr - treelet_roots.data());

		node->init_interior(dim,
			build_upper_sah(treelet_roots, start, mid,  total_nodes),
			build_upper_sah(treelet_roots, mid,   end,  total_nodes));
		return node;
	}

	// ---- Flatten build tree into LinearBVHNode array -----------------------

	int flatten_bvh(bvh_detail::BVHBuildNode<T>* node, int* offset) {
		auto* linear = &nodes_[*offset];
		linear->bounds = node->bounds;
		int my_offset  = (*offset)++;

		if (node->n_primitives > 0) {
			// Leaf
			linear->primitives_offset = node->first_prim_offset;
			linear->n_primitives      = static_cast<uint16_t>(node->n_primitives);
		} else {
			// Interior
			linear->axis       = static_cast<uint8_t>(node->split_axis);
			linear->n_primitives = 0;
			flatten_bvh(node->children[0], offset);
			linear->second_child_offset = flatten_bvh(node->children[1], offset);
		}
		return my_offset;
	}

	// -------------------------------------------------------------------------
	// Data members
	// -------------------------------------------------------------------------
	std::vector<Prim>                                    primitives_;
	std::vector<Node>                                    nodes_;
	std::vector<bvh_detail::BVHBuildNode<T>>             build_pool_;  // recursive build nodes
	int                                                  pool_ptr_{0};
	std::vector<std::vector<bvh_detail::BVHBuildNode<T>>> hlbvh_treelet_pools_; // HLBVH treelet nodes
	int                                                  max_prims_{1};
	BvhSplitMethod                                       split_method_{BvhSplitMethod::SAH};
};
