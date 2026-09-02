#pragma once
// ---------------------------------------------------------------------------
// kd_tree.h -- SAH kd-tree scene acceleration structure
//
// Ports pbrt-v4 KdTreeAggregate (cpu/aggregates.h / cpu/aggregates.cpp,
// Apache-2.0) as a self-contained, header-only C++ template.
//
// Provides:
//   KdHit<T>        -- intersection result (t, surface normal, uv, prim_id)
//   KdTree<T, Prim> -- SAH kd-tree built from a list of duck-typed primitives
//
// Primitive concept (Prim):
//   // Fill axis-aligned bounding box.
//   void bbox(T out_min[3], T out_max[3]) const;
//
//   // Intersect ray (org, dir) in [t_min, t_max].
//   // Returns KdHit<T> on hit, empty otherwise.
//   std::optional<KdHit<T>>
//   intersect(const T org[3], const T dir[3], T t_min, T t_max) const;
//
//   // Shadow test: true if any intersection in (0, t_max).
//   bool intersect_p(const T org[3], const T dir[3], T t_max) const;
//   // (If not provided, intersect() is used as a fallback in intersect_p.)
//
// Usage:
//   std::vector<MyPrim> prims = ...;
//   KdTree<double, MyPrim> kd;
//   kd.build(prims);
//   auto hit = kd.intersect(org, dir, 0.0, 1e30);
//   bool shadow = kd.intersect_p(org, dir, dist);
//
// Design rules (same as bdpt.h / mlt.h):
//   - Header-only, no virtual functions, no heap allocation in hot paths
//   - Template T: float or double
//   - Template Prim: duck-typed (see concept above)
//   - Single-threaded build; traversal is read-only and thread-safe
//
// pbrt-v4 alignment:
//   KdNode           <-> KdTreeNode (alignas(8), union split/leaf)
//   BoundEdge<T>     <-> BoundEdge (Start/End edges for SAH)
//   KdTree::build()  <-> KdTreeAggregate ctor + buildTree()
//   KdTree::intersect() <-> KdTreeAggregate::Intersect()
//   KdTree::intersect_p() <-> KdTreeAggregate::IntersectP()
//
// Default SAH parameters match pbrt-v4:
//   isect_cost=5, traversal_cost=1, empty_bonus=0.5, max_prims_leaf=1
//   max_depth = round(8 + 1.3 * log2(n))
//
// Reference: pbrt-v4 src/pbrt/cpu/aggregates.h / aggregates.cpp (Apache-2.0)
//            "Physically Based Rendering: From Theory to Implementation",
//            Pharr, Jakob, Humphreys. 4th ed., §7.3.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

#include "cpu_gpu.h"

// ---------------------------------------------------------------------------
// KdHit<T> -- result of a successful kd-tree ray intersection
// ---------------------------------------------------------------------------
template<typename T>
struct KdHit {
	T t;                    // ray parameter at hit
	T nx, ny, nz;           // surface normal (unit, world space)
	T u, v;                 // surface parameterisation
	int prim_id;            // index of the hit primitive in the build list
};

// ---------------------------------------------------------------------------
// kd_detail -- internal helpers
// ---------------------------------------------------------------------------
namespace kd_detail {

// Stable quadratic: roots ordered t0 <= t1.  Returns false when no real roots.
template<typename T>
inline bool solve_quadratic(T a, T b, T c, T& t0, T& t1) {
	double da = (double)a, db = (double)b, dc = (double)c;
	double disc = db*db - 4.0*da*dc;
	if (disc < 0.0) return false;
	double sq = std::sqrt(disc);
	double q  = (db < 0) ? -0.5*(db - sq) : -0.5*(db + sq);
	t0 = (T)(q / da);
	t1 = (T)(dc / q);
	if (t0 > t1) { T tmp = t0; t0 = t1; t1 = tmp; }
	return true;
}

// integer log2 (floor), matching pbrt-v4 Log2Int for size_t
inline int log2int(int64_t v) {
	assert(v > 0);
	int r = 0;
	while (v > 1) { v >>= 1; ++r; }
	return r;
}

// ---------------------------------------------------------------------------
// KdNode -- compact 8-byte node matching pbrt-v4 KdTreeNode layout
//
//   Interior: flags = axis | (aboveChild << 2)   [axis in {0,1,2}]
//   Leaf:     flags = 3 | (nPrims << 2)
// ---------------------------------------------------------------------------
struct alignas(8) KdNode {
	union {
		float split;                // interior: split position
		int   one_prim_index;       // leaf, nPrims==1: primitive index
		int   prim_indices_offset;  // leaf, nPrims>1: offset into prim_indices[]
	};

	uint32_t flags;

	// Interior node initialisation
	void init_interior(int axis, int above_child, float s) {
		split = s;
		flags = (uint32_t)axis | ((uint32_t)above_child << 2);
	}

	// Leaf node initialisation (primNums: list of prim indices for this leaf)
	void init_leaf(const int* prim_nums, int n,
				   std::vector<int>& prim_indices_store) {
		flags = 3u | ((uint32_t)n << 2);
		if (n == 0) {
			one_prim_index = 0;
		} else if (n == 1) {
			one_prim_index = prim_nums[0];
		} else {
			prim_indices_offset = (int)prim_indices_store.size();
			for (int i = 0; i < n; ++i)
				prim_indices_store.push_back(prim_nums[i]);
		}
	}

	bool   is_leaf()    const { return (flags & 3u) == 3u; }
	int    split_axis() const { return (int)(flags & 3u); }
	float  split_pos()  const { return split; }
	int    n_prims()    const { return (int)(flags >> 2); }
	int    above_child()const { return (int)(flags >> 2); }
};

// ---------------------------------------------------------------------------
// BoundEdge<T> -- a primitive's min/max endpoint along one axis (for SAH)
// ---------------------------------------------------------------------------
template<typename T>
struct BoundEdge {
	T   t;
	int prim_num;
	bool is_start;  // true = lower bound, false = upper bound

	BoundEdge() = default;
	BoundEdge(T t_, int p, bool start) : t(t_), prim_num(p), is_start(start) {}

	bool operator<(const BoundEdge& o) const {
		if (t != o.t) return t < o.t;
		// Start before End at the same position.
		// Mirrors pbrt-v4: std::tie(t, type) where Start=0 < End=1.
		return is_start && !o.is_start;
	}
};

// Simple AABB vs ray slab test; returns false if no overlap in [0, t_max].
template<typename T>
inline bool aabb_intersect_p(const T bmin[3], const T bmax[3],
							  const T org[3],  const T dir[3],
							  T t_max, T& t_min_out, T& t_max_out) {
	T t_min = T(0);
	T t_cur_max = t_max;
	for (int i = 0; i < 3; ++i) {
		T inv = (dir[i] != T(0)) ? T(1) / dir[i]
								 : (dir[i] >= 0 ? std::numeric_limits<T>::max()
												: -std::numeric_limits<T>::max());
		T t0 = (bmin[i] - org[i]) * inv;
		T t1 = (bmax[i] - org[i]) * inv;
		if (t0 > t1) { T tmp = t0; t0 = t1; t1 = tmp; }
		t_min     = std::max(t_min,     t0);
		t_cur_max = std::min(t_cur_max, t1);
		if (t_min > t_cur_max) return false;
	}
	t_min_out = t_min;
	t_max_out = t_cur_max;
	return true;
}

} // namespace kd_detail

// ---------------------------------------------------------------------------
// KdTreeAccelParams -- non-templated mirror of KdTree<T,Prim>::Params below,
// for callers (e.g. this project's own pbrt-scene loader/CPU builder) that
// need to carry an Accelerator "kdtree" directive's params through code that
// has no reason to know T/Prim - one value threaded through instead of 5
// separately-named, separately-declared scalars at every layer. Field-for-
// field identical to Params (same pbrt-v4 defaults - see that struct's own
// comment); KdTree<T,Prim>::build() below accepts either.
// ---------------------------------------------------------------------------
struct KdTreeAccelParams {
	int    intersectCost = 5;
	int    traversalCost = 1;
	double emptyBonus    = 0.5;
	int    maxPrims      = 1;
	int    maxDepth      = -1; // -1 = auto: round(8 + 1.3*log2(n))
};

// ---------------------------------------------------------------------------
// KdTree<T, Prim>
// ---------------------------------------------------------------------------
template<typename T, typename Prim>
class KdTree {
public:
	// SAH tuning parameters (pbrt-v4 defaults)
	struct Params {
		int   isect_cost      = 5;
		int   traversal_cost  = 1;
		T     empty_bonus     = T(0.5);
		int   max_prims_leaf  = 1;
		int   max_depth       = -1; // -1 = auto: round(8 + 1.3*log2(n))

		Params() = default;
		// From the non-templated KdTreeAccelParams above - see that
		// struct's own comment.
		Params(const KdTreeAccelParams& p)
			: isect_cost(p.intersectCost), traversal_cost(p.traversalCost),
			  empty_bonus(T(p.emptyBonus)), max_prims_leaf(p.maxPrims),
			  max_depth(p.maxDepth) {}
	};

	KdTree() = default;

	// Build the kd-tree from a list of primitives.
	// Each prim must satisfy the Prim concept (see file header).
	// Takes prims BY VALUE (not const&) so a caller that no longer needs its
	// own copy can std::move it in for a real move rather than an extra
	// internal copy - mirrors bvh_aggregate.h's own BvhTree::build() for the
	// identical reason. A caller that still needs its vector afterward can
	// simply pass an lvalue, which copies exactly once at the call boundary,
	// same cost as the old by-const-ref signature.
	void build(std::vector<Prim> prims,
			   const Params& params = Params{}) {
		params_ = params;
		prims_  = std::move(prims);
		nodes_.clear();
		prim_indices_.clear();

		int n = (int)prims_.size();
		if (n == 0) return;

		// Determine max depth
		int max_depth = params_.max_depth;
		if (max_depth <= 0)
			max_depth = (int)std::round(8.0 + 1.3 * kd_detail::log2int(n));

		// Compute per-primitive bounding boxes
		std::vector<std::array<T, 6>> prim_bounds(n); // [xmin,ymin,zmin, xmax,ymax,zmax]
		T bmin[3] = { std::numeric_limits<T>::max(),
					  std::numeric_limits<T>::max(),
					  std::numeric_limits<T>::max() };
		T bmax[3] = { std::numeric_limits<T>::lowest(),
					  std::numeric_limits<T>::lowest(),
					  std::numeric_limits<T>::lowest() };
		for (int i = 0; i < n; ++i) {
			T lo[3], hi[3];
			prims_[i].bbox(lo, hi);
			for (int k = 0; k < 3; ++k) {
				prim_bounds[i][k]   = lo[k];
				prim_bounds[i][k+3] = hi[k];
				bmin[k] = std::min(bmin[k], lo[k]);
				bmax[k] = std::max(bmax[k], hi[k]);
			}
		}
		for (int k = 0; k < 3; ++k) { bounds_min_[k] = bmin[k]; bounds_max_[k] = bmax[k]; }

		// Working arrays
		std::vector<std::vector<kd_detail::BoundEdge<T>>> edges(3);
		for (int axis = 0; axis < 3; ++axis)
			edges[axis].resize(2 * n);

		std::vector<int> prims0(n);
		std::vector<int> prims1((size_t)(max_depth + 1) * n);

		std::vector<int> all_prim_nums(n);
		for (int i = 0; i < n; ++i) all_prim_nums[i] = i;

		// Reserve a plausible upper bound for nodes (can grow)
		nodes_.reserve(2 * n);

		build_tree(0, bmin, bmax, prim_bounds, all_prim_nums.data(), n,
				   max_depth, edges, prims0.data(), prims1.data(), 0);
	}

	// Intersect a ray with the scene.  Returns the closest KdHit<T> or empty.
	std::optional<KdHit<T>> intersect(const T org[3], const T dir[3],
									  T t_min_ray, T t_max_ray) const {
		if (nodes_.empty()) return {};

		T t_entry, t_exit;
		if (!kd_detail::aabb_intersect_p(bounds_min_, bounds_max_,
										 org, dir, t_max_ray,
										 t_entry, t_exit))
			return {};
		t_entry = std::max(t_entry, t_min_ray);
		if (t_entry > t_exit) return {};

		T inv_dir[3];
		for (int k = 0; k < 3; ++k)
			inv_dir[k] = (dir[k] != T(0)) ? T(1) / dir[k] : T(0);

		struct NodeToVisit { const kd_detail::KdNode* node; T t_min, t_max; };
		constexpr int kMaxToVisit = 64;
		NodeToVisit todo[kMaxToVisit];
		int todo_idx = 0;

		std::optional<KdHit<T>> best;

		const kd_detail::KdNode* node = &nodes_[0];
		T cur_min = t_entry, cur_max = t_exit;

		while (node) {
			// If we already found a closer hit, skip this subtree
			if (best && best->t < cur_min) {
				// Pop next
				if (todo_idx > 0) {
					--todo_idx;
					node    = todo[todo_idx].node;
					cur_min = todo[todo_idx].t_min;
					cur_max = todo[todo_idx].t_max;
				} else break;
				continue;
			}

			if (!node->is_leaf()) {
				// Interior node
				int axis = node->split_axis();
				T t_split = (T(node->split_pos()) - org[axis]) *
							(inv_dir[axis] != T(0) ? inv_dir[axis]
							 : std::numeric_limits<T>::max());

				const kd_detail::KdNode *first, *second;
				bool below_first = (org[axis] < T(node->split_pos())) ||
								   (org[axis] == T(node->split_pos()) && dir[axis] <= T(0));
				if (below_first) {
					first  = node + 1;
					second = &nodes_[node->above_child()];
				} else {
					first  = &nodes_[node->above_child()];
					second = node + 1;
				}

				if (t_split > cur_max || t_split <= T(0)) {
					node = first;
				} else if (t_split < cur_min) {
					node = second;
				} else {
					// Visit both: enqueue second, descend first
					assert(todo_idx < kMaxToVisit);
					todo[todo_idx].node  = second;
					todo[todo_idx].t_min = t_split;
					todo[todo_idx].t_max = cur_max;
					++todo_idx;
					node    = first;
					cur_max = t_split;
				}
			} else {
				// Leaf node
				int np = node->n_prims();
				if (np == 1) {
					int idx = node->one_prim_index;
					auto h = prims_[idx].intersect(org, dir, t_min_ray, t_max_ray);
					if (h && (!best || h->t < best->t)) {
						best = h;
						best->prim_id = idx;
						t_max_ray = best->t;
					}
				} else {
					for (int i = 0; i < np; ++i) {
						int idx = prim_indices_[node->prim_indices_offset + i];
						auto h = prims_[idx].intersect(org, dir, t_min_ray, t_max_ray);
						if (h && (!best || h->t < best->t)) {
							best = h;
							best->prim_id = idx;
							t_max_ray = best->t;
						}
					}
				}

				// Pop next
				if (todo_idx > 0) {
					--todo_idx;
					node    = todo[todo_idx].node;
					cur_min = todo[todo_idx].t_min;
					cur_max = todo[todo_idx].t_max;
				} else break;
			}
		}
		return best;
	}

	// Shadow test: true if any primitive occludes the segment [0, t_max].
	bool intersect_p(const T org[3], const T dir[3], T t_max_ray) const {
		if (nodes_.empty()) return false;

		T t_entry, t_exit;
		if (!kd_detail::aabb_intersect_p(bounds_min_, bounds_max_,
										 org, dir, t_max_ray,
										 t_entry, t_exit))
			return false;

		T inv_dir[3];
		for (int k = 0; k < 3; ++k)
			inv_dir[k] = (dir[k] != T(0)) ? T(1) / dir[k] : T(0);

		struct NodeToVisit { const kd_detail::KdNode* node; T t_min, t_max; };
		constexpr int kMaxToVisit = 64;
		NodeToVisit todo[kMaxToVisit];
		int todo_idx = 0;

		const kd_detail::KdNode* node = &nodes_[0];
		T cur_min = t_entry, cur_max = t_exit;

		while (node) {
			if (!node->is_leaf()) {
				int axis = node->split_axis();
				T t_split = (T(node->split_pos()) - org[axis]) *
							(inv_dir[axis] != T(0) ? inv_dir[axis]
							 : std::numeric_limits<T>::max());

				const kd_detail::KdNode *first, *second;
				bool below_first = (org[axis] < T(node->split_pos())) ||
								   (org[axis] == T(node->split_pos()) && dir[axis] <= T(0));
				if (below_first) {
					first  = node + 1;
					second = &nodes_[node->above_child()];
				} else {
					first  = &nodes_[node->above_child()];
					second = node + 1;
				}

				if (t_split > cur_max || t_split <= T(0)) {
					node = first;
				} else if (t_split < cur_min) {
					node = second;
				} else {
					assert(todo_idx < kMaxToVisit);
					todo[todo_idx].node  = second;
					todo[todo_idx].t_min = t_split;
					todo[todo_idx].t_max = cur_max;
					++todo_idx;
					node    = first;
					cur_max = t_split;
				}
			} else {
				int np = node->n_prims();
				if (np == 1) {
					int idx = node->one_prim_index;
					if (prims_[idx].intersect_p(org, dir, t_max_ray)) return true;
				} else {
					for (int i = 0; i < np; ++i) {
						int idx = prim_indices_[node->prim_indices_offset + i];
						if (prims_[idx].intersect_p(org, dir, t_max_ray)) return true;
					}
				}

				if (todo_idx > 0) {
					--todo_idx;
					node    = todo[todo_idx].node;
					cur_min = todo[todo_idx].t_min;
					cur_max = todo[todo_idx].t_max;
				} else break;
			}
		}
		return false;
	}

	// Number of nodes allocated (useful for testing)
	int num_nodes() const { return (int)nodes_.size(); }

	// Number of primitives stored
	int num_prims() const { return (int)prims_.size(); }

private:
	// -----------------------------------------------------------------------
	// build_tree -- recursive SAH kd-tree construction
	//   Mirrors pbrt-v4 KdTreeAggregate::buildTree()
	// -----------------------------------------------------------------------
	void build_tree(int node_num,
					const T node_min[3], const T node_max[3],
					const std::vector<std::array<T,6>>& all_bounds,
					const int* prim_nums, int n_prims,
					int depth,
					std::vector<std::vector<kd_detail::BoundEdge<T>>>& edges,
					int* prims0, int* prims1,
					int bad_refines)
	{
		// Allocate a new node slot
		if ((int)nodes_.size() <= node_num) {
			nodes_.resize(node_num + 1);
		}

		// Leaf: too few prims, max depth, or too many bad splits
		if (n_prims <= params_.max_prims_leaf || depth == 0) {
			nodes_[node_num].init_leaf(prim_nums, n_prims, prim_indices_);
			return;
		}

		// Compute node surface area
		T d[3];
		for (int k = 0; k < 3; ++k) d[k] = node_max[k] - node_min[k];
		T total_sa  = T(2) * (d[0]*d[1] + d[1]*d[2] + d[0]*d[2]);
		T inv_total = (total_sa > T(0)) ? T(1) / total_sa : T(0);
		T leaf_cost = (T)params_.isect_cost * n_prims;

		int best_axis   = -1;
		int best_offset = -1;
		T   best_cost   = std::numeric_limits<T>::max();

		// Try splitting on the longest axis first, retry on others if needed
		int axis = 0;
		{ // find longest axis
			if (d[1] > d[0]) axis = 1;
			if (d[2] > d[axis]) axis = 2;
		}

		// We try each axis (at most 3) via a loop; mirrors pbrt-v4's goto retrySplit
		for (int try_count = 0; try_count < 3; ++try_count) {
			// Build edge list for this axis
			for (int i = 0; i < n_prims; ++i) {
				int pn = prim_nums[i];
				edges[axis][2*i]   = kd_detail::BoundEdge<T>(all_bounds[pn][axis],   pn, true);
				edges[axis][2*i+1] = kd_detail::BoundEdge<T>(all_bounds[pn][axis+3], pn, false);
			}
			std::sort(edges[axis].begin(), edges[axis].begin() + 2 * n_prims);

			// Scan split candidates
			int n_below = 0, n_above = n_prims;
			for (int i = 0; i < 2 * n_prims; ++i) {
				if (!edges[axis][i].is_start) --n_above;

				T edge_t = edges[axis][i].t;
				if (edge_t > node_min[axis] && edge_t < node_max[axis]) {
					// Child surface areas
					int other0 = (axis + 1) % 3, other1 = (axis + 2) % 3;
					T below_sa = T(2) * (d[other0]*d[other1] +
										 (edge_t - node_min[axis]) * (d[other0] + d[other1]));
					T above_sa = T(2) * (d[other0]*d[other1] +
										 (node_max[axis] - edge_t) * (d[other0] + d[other1]));
					T p_below  = below_sa * inv_total;
					T p_above  = above_sa * inv_total;
					T eb = (n_above == 0 || n_below == 0) ? params_.empty_bonus : T(0);
					T cost = (T)params_.traversal_cost +
							 (T)params_.isect_cost * (T(1) - eb) *
							 (p_below * n_below + p_above * n_above);
					if (cost < best_cost) {
						best_cost   = cost;
						best_axis   = axis;
						best_offset = i;
					}
				}
				if (edges[axis][i].is_start) ++n_below;
			}

			if (best_axis != -1) break;   // found a good split
			axis = (axis + 1) % 3;        // try next axis
		}

		// Create leaf if no good splits found
		if (best_cost > leaf_cost) ++bad_refines;
		if ((best_cost > T(4) * leaf_cost && n_prims < 16) ||
			best_axis == -1 || bad_refines == 3) {
			nodes_[node_num].init_leaf(prim_nums, n_prims, prim_indices_);
			return;
		}

		// Classify primitives to below/above child
		int n0 = 0, n1 = 0;
		for (int i = 0; i < best_offset; ++i)
			if (edges[best_axis][i].is_start)
				prims0[n0++] = edges[best_axis][i].prim_num;
		for (int i = best_offset + 1; i < 2 * n_prims; ++i)
			if (!edges[best_axis][i].is_start)
				prims1[n1++] = edges[best_axis][i].prim_num;

		// Split position
		T t_split = edges[best_axis][best_offset].t;

		// Child bounding boxes
		T bounds0_min[3], bounds0_max[3];
		T bounds1_min[3], bounds1_max[3];
		for (int k = 0; k < 3; ++k) {
			bounds0_min[k] = bounds1_min[k] = node_min[k];
			bounds0_max[k] = bounds1_max[k] = node_max[k];
		}
		bounds0_max[best_axis] = t_split;
		bounds1_min[best_axis] = t_split;

		// Recurse below child (node_num + 1)
		build_tree(node_num + 1, bounds0_min, bounds0_max, all_bounds,
				   prims0, n0, depth - 1, edges, prims0, prims1 + n1, bad_refines);

		// Interior node: record above_child = nextFreeNode (mirrors pbrt-v4).
		// build_tree(above_child,...) will allocate the slot at the top of its
		// call, so no pre-resize is needed here.
		int above_child = (int)nodes_.size();
		nodes_[node_num].init_interior(best_axis, above_child,
									   (float)t_split);

		// Recurse above child
		build_tree(above_child, bounds1_min, bounds1_max, all_bounds,
				   prims1, n1, depth - 1, edges, prims0, prims1 + n1, bad_refines);
	}

	// -----------------------------------------------------------------------
	// Members
	// -----------------------------------------------------------------------
	Params                  params_;
	std::vector<Prim>       prims_;
	std::vector<kd_detail::KdNode> nodes_;
	std::vector<int>        prim_indices_; // multi-prim leaf indices
	T bounds_min_[3]{};
	T bounds_max_[3]{};
};
