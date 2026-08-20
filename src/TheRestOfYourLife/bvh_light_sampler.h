#ifndef BVH_LIGHT_SAMPLER_H
#define BVH_LIGHT_SAMPLER_H
//==============================================================================
// bvh_light_sampler.h  -- Bounding-Cone BVH light sampler
//
// pbrt-v4 alignment: Â§12.6, lightsamplers.h BVHLightSampler.
//
// At construction a BVH is built over the registered lights.  Each node
// stores a BvhLightBounds:
//   phi       -- total emitted power of the subtree (area * Le_avg * pi)
//   w         -- mean emission direction (unit)
//   cosTheta_o -- half-angle of directional emission cone around w
//   cosTheta_e -- half-angle of the normal-facing cone (pi/2 for two-sided)
//   bbox       -- world-space AABB of all light geometry in the subtree
//   twoSided   -- whether the light emits from both sides
//
// Importance(p, n) at a shading point (pbrt-v4 Â§12.6.2):
//   1. cosThetap = max(0, cos(angle_to_centroid - theta_o - theta_b))
//      where theta_b = half-angle subtended by bbox from p
//   2. importance = phi * cosThetap / dÂ²
//   3. Optionally multiply by cos(angle_w_to_n) for surface normals
//
// Sample(origin, u) -- O(log N) descent:
//   At each interior node pick child proportional to Importance; rescale u.
//   Returns direction toward sampled light via light->random(origin).
//
// pdf_value(origin, direction) -- O(log N) via bit_trail replay:
//   Replays the BVH path for the light that owns the given direction,
//   multiplying child PMFs along the way, then returns
//   P(light) * light->pdf_value(origin, direction).
//
// Drop-in replacement for power_light_list with identical hittable interface.
//==============================================================================

#include "hittable.h"
#include "hittable_list.h"
#include "rtweekend.h"

#include <vector>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cassert>
#include <mutex>
#include <numeric>

// ---------------------------------------------------------------------------
// BvhLightBounds  (mirrors pbrt-v4 BvhLightBounds / CompactLightBounds)
// ---------------------------------------------------------------------------
struct BvhLightBounds {
	aabb   bbox;          // world-space bounding box of geometry
	vec3   w;             // mean emission direction (unit vector)
	double phi      = 0;  // total emitted power estimate
	double cosTheta_o = 1;// cos of directional emission cone half-angle
	double cosTheta_e = 0;// cos of normal-facing cone half-angle (0 = pi/2)
	bool   twoSided = false;

	BvhLightBounds() = default;

	BvhLightBounds(const aabb& bb, const vec3& w_, double phi_,
				double cosTheta_o_, double cosTheta_e_, bool two)
		: bbox(bb), w(unit_vector(w_)), phi(phi_),
		  cosTheta_o(cosTheta_o_), cosTheta_e(cosTheta_e_), twoSided(two)
	{}

	// Merge two BvhLightBounds (interior-node constructor).
	// phi is summed; cone is widened to enclose both.
	// bbox is union; w is mean of both axes.
	static BvhLightBounds merge(const BvhLightBounds& a, const BvhLightBounds& b) {
		BvhLightBounds r;
		r.bbox      = aabb(a.bbox, b.bbox);
		r.phi       = a.phi + b.phi;
		r.twoSided  = a.twoSided || b.twoSided;

		// Widen cone: new w = mean direction, cosTheta_o = min of originals
		// adjusted by the angle between axes (conservative).
		vec3 wSum = a.w + b.w;
		double wl = wSum.length();
		r.w = (wl > 1e-14) ? wSum / wl : a.w;

		// Conservative: cosTheta_o is the smaller (wider) of the two, then
		// subtract the opening angle between the two axes.
		// angle between axes: acos(clamp(dot(a.w, b.w), -1, 1))
		double cosAB  = dot(a.w, b.w);
		cosAB = cosAB < -1 ? -1 : (cosAB > 1 ? 1 : cosAB);
		double halfAB = std::acos(cosAB) * 0.5;

		// Widen each cone by half the inter-axis angle
		auto widen = [](double cosTheta, double dAngle) -> double {
			double angle = std::acos(cosTheta < -1 ? -1 : (cosTheta > 1 ? 1 : cosTheta));
			double widened = angle + dAngle;
			if (widened >= 3.14159265358979323846)
				return -1.0;
			return std::cos(widened);
		};

		r.cosTheta_o = std::min(widen(a.cosTheta_o, halfAB),
								widen(b.cosTheta_o, halfAB));
		r.cosTheta_e = std::min(a.cosTheta_e, b.cosTheta_e);
		return r;
	}

	// Importance at reference point p with surface normal n (unit, or zero).
	// Mirrors pbrt-v4 CompactLightBounds::Importance.
	double importance(const point3& p, const vec3& n) const {
		if (phi <= 0) return 0.0;

		// Centroid of bounding box
		point3 pc = point3(
			(bbox.x.min + bbox.x.max) * 0.5,
			(bbox.y.min + bbox.y.max) * 0.5,
			(bbox.z.min + bbox.z.max) * 0.5);

		// dÂ² = max(distanceÂ² to centroid, half-diagonalÂ²) to avoid dÂ²=0
		vec3 pc_p = pc - p;
		double d2 = pc_p.length_squared();
		double halfDiag = (bbox.x.max - bbox.x.min) * (bbox.x.max - bbox.x.min)
						+ (bbox.y.max - bbox.y.min) * (bbox.y.max - bbox.y.min)
						+ (bbox.z.max - bbox.z.min) * (bbox.z.max - bbox.z.min);
		halfDiag *= 0.25;
		d2 = std::max(d2, halfDiag);
		if (d2 < 1e-30) d2 = 1e-30;

		// cos-sub / sin-sub helpers (pbrt-v4 cosSubClamped / sinSubClamped)
		auto cosSubClamped = [](double sA, double cA, double sB, double cB) -> double {
			if (cA > cB) return 1.0;
			return cA * cB + sA * sB;
		};
		auto sinSubClamped = [](double sA, double cA, double sB, double cB) -> double {
			if (cA > cB) return 0.0;
			return sA * cB - cA * sB;
		};

		// Angle to centroid from shading point
		vec3 wi = (p - pc);   // direction from centroid toward shading point
		double wil = wi.length();
		if (wil < 1e-14) return phi / d2;  // p is inside bbox
		wi = wi / wil;

		double cosTheta_w = dot(w, wi);
		if (twoSided) cosTheta_w = std::fabs(cosTheta_w);
		cosTheta_w = cosTheta_w < -1 ? -1 : (cosTheta_w > 1 ? 1 : cosTheta_w);
		double sinTheta_w = std::sqrt(std::max(0.0, 1.0 - cosTheta_w * cosTheta_w));

		// theta_b: half-angle subtended by the bbox from p (BoundSubtendedDirections)
		double sinTheta_b2 = halfDiag / d2;
		double cosTheta_b, sinTheta_b;
		if (sinTheta_b2 >= 1.0) { cosTheta_b = 0.0; sinTheta_b = 1.0; }
		else { sinTheta_b = std::sqrt(sinTheta_b2); cosTheta_b = std::sqrt(1.0 - sinTheta_b2); }

		// sinTheta_o / cosTheta_o
		double co = cosTheta_o < -1 ? -1 : (cosTheta_o > 1 ? 1 : cosTheta_o);
		double so = std::sqrt(std::max(0.0, 1.0 - co * co));

		// cosThetap = cos(theta_w - theta_o - theta_b) clamped to [0,1]
		double cosTheta_x = cosSubClamped(sinTheta_w, cosTheta_w, so, co);
		double sinTheta_x = sinSubClamped(sinTheta_w, cosTheta_w, so, co);
		double cosThetap  = cosSubClamped(sinTheta_x, cosTheta_x, sinTheta_b, cosTheta_b);

		if (cosThetap <= cosTheta_e) return 0.0;

		double imp = phi * cosThetap / d2;

		// Weight by cos(theta_i) at surface if normal is provided
		if (n.length_squared() > 1e-14) {
			double cosTheta_i = std::fabs(dot(wi, n));
			double sinTheta_i = std::sqrt(std::max(0.0, 1.0 - cosTheta_i * cosTheta_i));
			double cosThetap_i = cosSubClamped(sinTheta_i, cosTheta_i,
											   sinTheta_b, cosTheta_b);
			imp *= cosThetap_i;
		}

		return std::max(0.0, imp);
	}
};

// ---------------------------------------------------------------------------
// LightBoundsForHittable
// Compute a BvhLightBounds for a single light hittable given its emitted power.
// For quads/triangles: twoSided=false, w = face normal (unknown here, so we
// use w=(0,0,0) -> widened cone = full sphere, cosTheta_o = -1).
// Callers can supply a proper w via the overload.
// ---------------------------------------------------------------------------
inline BvhLightBounds light_bounds_for(std::shared_ptr<hittable> obj,
									double phi,
									const vec3& w      = vec3(0,0,0),
									double cosTheta_o  = -1.0,  // full sphere
									double cosTheta_e  =  0.0,  // pi/2 half-space
									bool   twoSided    = false)
{
	aabb bb = obj->bounding_box();
	vec3 wn = (w.length_squared() > 1e-14) ? unit_vector(w) : vec3(0,1,0);
	return BvhLightBounds(bb, wn, phi, cosTheta_o, cosTheta_e, twoSided);
}


// ---------------------------------------------------------------------------
// BVHLightNode -- compact BVH node (leaf or interior)
// ---------------------------------------------------------------------------
struct BVHLightNode {
	BvhLightBounds bounds;
	// isLeaf=true : child_or_light = index into lights array
	// isLeaf=false: child_or_light = index of right child in nodes array
	//               left child is always nodeIndex+1
	int  child_or_light = 0;
	bool isLeaf         = false;

	static BVHLightNode make_leaf(int light_idx, const BvhLightBounds& lb) {
		BVHLightNode n; n.bounds = lb; n.child_or_light = light_idx; n.isLeaf = true; return n;
	}
	static BVHLightNode make_interior(int right_child, const BvhLightBounds& lb) {
		BVHLightNode n; n.bounds = lb; n.child_or_light = right_child; n.isLeaf = false; return n;
	}
};


// ---------------------------------------------------------------------------
// bvh_light_sampler
// Drop-in replacement for power_light_list with bounding-cone BVH selection.
// ---------------------------------------------------------------------------
class bvh_light_sampler : public hittable {
  public:
	bvh_light_sampler() = default;

	// Build from an existing hittable_list with equal (uniform) weights -
	// mirrors power_light_list's identical-signature constructor, so a
	// call site written against that class can switch types with no other
	// change (see power_light_list(const hittable_list&) in
	// power_light_sampler.h).
	explicit bvh_light_sampler(const hittable_list& list) {
		for (const auto& obj : list.objects)
			add(obj, 1.0);
	}

	// Hand-written copy ctor/assignment: build_mutex_ is neither copyable nor
	// movable (std::mutex), and dirty_ is std::atomic<bool> (also neither),
	// so the implicit versions would be deleted - which would break every
	// call site that builds one of these by value and returns/assigns it
	// (e.g. build_cornell_box_bvh_lights() in cornell_box_scene.h). A fresh
	// mutex is correct here regardless: it only ever guards this object's
	// OWN first-build race (see ensure_built()'s comment), so a copy/move
	// target needs its own, not the source's.
	bvh_light_sampler(const bvh_light_sampler& other)
		: lights_(other.lights_), light_bounds_(other.light_bounds_),
		  nodes_(other.nodes_), bit_trail_(other.bit_trail_), bbox_(other.bbox_),
		  dirty_(other.dirty_.load(std::memory_order_acquire)) {}

	bvh_light_sampler& operator=(const bvh_light_sampler& other) {
		if (this == &other) return *this;
		lights_ = other.lights_;
		light_bounds_ = other.light_bounds_;
		nodes_ = other.nodes_;
		bit_trail_ = other.bit_trail_;
		bbox_ = other.bbox_;
		dirty_.store(other.dirty_.load(std::memory_order_acquire), std::memory_order_release);
		return *this;
	}

	// Add a light with its BvhLightBounds.
	void add(std::shared_ptr<hittable> obj, const BvhLightBounds& lb) {
		lights_.push_back(obj);
		light_bounds_.push_back(lb);
		bbox_ = aabb(bbox_, obj->bounding_box());
		dirty_.store(true, std::memory_order_relaxed);
	}

	// Convenience: add with just a power estimate (full-sphere cone).
	void add(std::shared_ptr<hittable> obj, double phi = 1.0) {
		add(obj, light_bounds_for(obj, phi));
	}

	// Must be called before any sample/pdf call if lights were added.
	// build() is called lazily (and thread-safely - see ensure_built()) on
	// first use if dirty_.
	void build() const {
		nodes_.clear();
		bit_trail_.assign(lights_.size(), 0u);
		if (lights_.empty()) { dirty_.store(false, std::memory_order_release); return; }

		// Build with indices 0..N-1
		std::vector<int> indices(lights_.size());
		std::iota(indices.begin(), indices.end(), 0);
		build_recursive(indices, 0, (int)indices.size(), 0);
		dirty_.store(false, std::memory_order_release);
	}

	// ---- hittable interface ------------------------------------------------

	bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
		bool any = false;
		for (const auto& obj : lights_) {
			hit_record tmp;
			if (obj->hit(r, ray_t, tmp) && tmp.t < ray_t.max) {
				ray_t.max = tmp.t;
				rec = tmp;
				any = true;
			}
		}
		return any;
	}

	aabb bounding_box() const override { return bbox_; }

	// Sample a direction toward a BVH-selected light, importance-weighted.
	vec3 random(const point3& origin) const override {
		ensure_built();
		if (lights_.empty()) return vec3(1,0,0);
		int li = sample_light_index(origin, vec3(0,0,0), random_double());
		if (li < 0) {
			// Fallback: uniform
			int i = (int)(random_double() * lights_.size());
			if (i >= (int)lights_.size()) i = (int)lights_.size()-1;
			return lights_[i]->random(origin);
		}
		return lights_[li]->random(origin);
	}

	// pdf_value: P(light_i | p) * p_i(omega) summed over lights that
	// could have produced this direction, weighted by BVH selection PMF.
	double pdf_value(const point3& origin, const vec3& direction) const override {
		ensure_built();
		if (lights_.empty()) return 0.0;
		double sum = 0.0;
		for (int i = 0; i < (int)lights_.size(); ++i) {
			double p_light = light_pmf(i, origin, vec3(0,0,0));
			if (p_light > 0)
				sum += p_light * lights_[i]->pdf_value(origin, direction);
		}
		return sum;
	}

	int light_count() const { return (int)lights_.size(); }

	// Expose per-light selection PMF at a given origin (for testing)
	double light_pmf_at(int i, const point3& origin) const {
		ensure_built();
		return light_pmf(i, origin, vec3(0,0,0));
	}

	std::vector<std::shared_ptr<hittable>> objects;  // public alias for compat

  private:
	std::vector<std::shared_ptr<hittable>> lights_;
	std::vector<BvhLightBounds>               light_bounds_;
	mutable std::vector<BVHLightNode>      nodes_;
	mutable std::vector<uint32_t>          bit_trail_;  // per-light BVH path
	aabb                                   bbox_;
	mutable std::atomic<bool>              dirty_{true};
	mutable std::mutex                     build_mutex_;

	// camera::render() (src/TheRestOfYourLife/camera.h) spawns many worker
	// threads that all call random()/pdf_value() on this SAME instance for
	// NEE, starting from each thread's very first sample - so the lazy
	// "build on first use" this class advertises must be safe under
	// concurrent first-touch, not just single-threaded convenience. The
	// unguarded `if (dirty_) build()` this used to be let two threads both
	// enter build() at once, racing push_back/clear on nodes_/bit_trail_
	// (this is what actually happened - see CameraParametersTest.
	// ExtremeCoordinates, which segfaulted mid-render before this fix).
	// dirty_'s acquire/release pairing with build()'s final store makes the
	// fast path safe to read lock-free once built; the lock only serializes
	// the (rare) actual build.
	void ensure_built() const {
		if (!dirty_.load(std::memory_order_acquire)) return;
		std::lock_guard<std::mutex> lock(build_mutex_);
		if (dirty_.load(std::memory_order_relaxed)) build();
	}

	// ---- BVH build (recursive, depth-first, nodes stored in pre-order) ----

	// Returns index of the node just created in nodes_.
	int build_recursive(std::vector<int>& idx, int begin, int end, int depth) const {
		if (end - begin == 1) {
			int li = idx[begin];
			int node_idx = (int)nodes_.size();
			nodes_.push_back(BVHLightNode::make_leaf(li, light_bounds_[li]));
			bit_trail_[li] = 0u;  // will be OR-ed by parent
			return node_idx;
		}

		// Compute combined bounds for this range
		BvhLightBounds combined = light_bounds_[idx[begin]];
		for (int i = begin+1; i < end; ++i)
			combined = BvhLightBounds::merge(combined, light_bounds_[idx[i]]);

		// Split along longest bbox axis at median centroid
		int axis = longest_axis(combined.bbox);
		auto centroid = [&](int li) {
			const aabb& b = light_bounds_[li].bbox;
			if (axis == 0) return (b.x.min + b.x.max) * 0.5;
			if (axis == 1) return (b.y.min + b.y.max) * 0.5;
			return (b.z.min + b.z.max) * 0.5;
		};
		int mid = (begin + end) / 2;
		std::nth_element(idx.begin() + begin, idx.begin() + mid, idx.begin() + end,
			[&](int a, int b){ return centroid(a) < centroid(b); });

		// Reserve slot for this interior node, fill after children
		int node_idx = (int)nodes_.size();
		nodes_.push_back(BVHLightNode{});  // placeholder

		// Left child at node_idx+1 (immediately follows)
		build_recursive(idx, begin, mid, depth + 1);

		// Right child
		int right_idx = (int)nodes_.size();
		build_recursive(idx, mid, end, depth + 1);

		// Tag bit_trail for all lights in right subtree with bit at this depth
		tag_right_subtree(right_idx, depth);

		nodes_[node_idx] = BVHLightNode::make_interior(right_idx, combined);
		return node_idx;
	}

	// Set bit `depth` in bit_trail_ for every leaf in subtree rooted at node_idx.
	// All leaves in this subtree share the same "right turn at depth `depth`" bit.
	void tag_right_subtree(int node_idx, int depth) const {
		const BVHLightNode& n = nodes_[node_idx];
		if (n.isLeaf) {
			bit_trail_[n.child_or_light] |= (1u << depth);
		} else {
			tag_right_subtree(node_idx + 1,     depth);  // same depth for all leaves
			tag_right_subtree(n.child_or_light, depth);  // same depth for all leaves
		}
	}

	static int longest_axis(const aabb& bb) {
		double dx = bb.x.max - bb.x.min;
		double dy = bb.y.max - bb.y.min;
		double dz = bb.z.max - bb.z.min;
		if (dx >= dy && dx >= dz) return 0;
		if (dy >= dz)             return 1;
		return 2;
	}

	// ---- Traversal ---------------------------------------------------------

	// Sample a light index by descending the BVH from node 0.
	// Returns -1 if importance is zero everywhere.
	int sample_light_index(const point3& p, const vec3& n, double u) const {
		if (nodes_.empty()) return -1;
		int node_idx = 0;
		while (true) {
			const BVHLightNode& node = nodes_[node_idx];
			if (node.isLeaf)
				return node.child_or_light;

			const BVHLightNode& left  = nodes_[node_idx + 1];
			const BVHLightNode& right = nodes_[node.child_or_light];

			double ci0 = left.bounds.importance(p, n);
			double ci1 = right.bounds.importance(p, n);
			double sum = ci0 + ci1;
			if (sum <= 0) return -1;

			double p0 = ci0 / sum;
			int child;
			if (u < p0) {
				u /= p0;
				child = 0;
			} else {
				u = (u - p0) / (1.0 - p0);
				child = 1;
			}
			u = u < 0 ? 0 : (u > 0.9999999 ? 0.9999999 : u);
			node_idx = (child == 0) ? (node_idx + 1) : node.child_or_light;
		}
	}

	// Compute PMF of selecting light `li` at shading point p with normal n.
	// Uses bit_trail_ to replay the path.
	double light_pmf(int li, const point3& p, const vec3& n) const {
		if (nodes_.empty()) return 1.0 / (double)lights_.size();
		uint32_t trail = bit_trail_[li];
		int node_idx = 0;
		double pmf = 1.0;
		int depth = 0;
		while (true) {
			const BVHLightNode& node = nodes_[node_idx];
			if (node.isLeaf) return pmf;

			const BVHLightNode& left  = nodes_[node_idx + 1];
			const BVHLightNode& right = nodes_[node.child_or_light];

			double ci0 = left.bounds.importance(p, n);
			double ci1 = right.bounds.importance(p, n);
			double sum = ci0 + ci1;
			if (sum <= 0) return 0.0;

			int child = (trail >> depth) & 1;
			pmf *= (child == 0 ? ci0 : ci1) / sum;
			node_idx = (child == 0) ? (node_idx + 1) : node.child_or_light;
			++depth;
		}
	}
};


#endif
