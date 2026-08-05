// bvh_light_sampler2.h
// BVH-based importance light sampler, ported from pbrt-v4.
//
// BVHLightSampler2 builds a binary BVH over a set of lights described by
// their LightBounds.  At query time it descends the tree stochastically,
// picking children with probability proportional to their
// CompactLightBounds::Importance at the shading point, and returns a
// (lightIndex, pmf) pair.  PMF replay uses a per-light bit-trail.
//
// This is a header-only, allocator-free adaptation of pbrt-v4's
// BVHLightSampler.  The public interface is adapted to work with plain
// indices into a user-provided lights array rather than pbrt's Light type.
//
// API:
//   BVHLightSampler2(lightBoundsArray, count)
//   optional<SampledLight2> Sample(px,py,pz, nx,ny,nz, u)
//   float PMF(px,py,pz, nx,ny,nz, lightIndex)
//   bool  Empty() const
//
// Returns:
//   struct SampledLight2 { int lightIndex; float pmf; }
//
// Dependencies:
//   light_bvh_node.h  -- LightBVHNode, CompactLightBounds, LightBounds
//   scalar_math.h     -- SafeSqrt, Sqr, Pi
//
// References: pbrt-v4 src/pbrt/lightsamplers.h+cpp  (Apache-2.0)

#pragma once
#include <vector>
#include <algorithm>
#include <utility>
#include <optional>
#include <unordered_map>
#include <cmath>
#include <cassert>
#include "light_bvh_node.h"   // LightBVHNode -> CompactLightBounds -> LightBounds
#include "scalar_math.h"      // SafeSqrt, Sqr, Pi (= 3.14159...)

#ifndef CPU_GPU
#  ifdef __CUDACC__
#    define CPU_GPU __host__ __device__
#  else
#    define CPU_GPU inline
#  endif
#endif

// ---------------------------------------------------------------------------
// SampledLight2 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â result of a successful Sample() call
// ---------------------------------------------------------------------------
struct SampledLight2 {
	int   lightIndex = -1;
	float pmf        = 0.f;
};

// ---------------------------------------------------------------------------
// BVHLightSampler2
// ---------------------------------------------------------------------------
class BVHLightSampler2 {
public:
	// -----------------------------------------------------------------------
	// Constructor
	// Builds the BVH from an array of LightBounds.
	// lightBounds[i] must describe light i.
	// Lights with phi==0 are excluded from the BVH (treated as "infinite"
	// in pbrt-v4's sense, or simply zero-contribution here).
	// -----------------------------------------------------------------------
	BVHLightSampler2() = default;

	explicit BVHLightSampler2(const LightBounds* lightBoundsArr, int count)
	{
		// Partition lights into BVH-eligible (phi > 0) vs. skipped
		std::vector<std::pair<int,LightBounds>> bvhLights;
		bvhLights.reserve(count);

		// Scene-wide AABB ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â start empty (represented as inverted)
		allBMinX = allBMinY = allBMinZ =  1e30f;
		allBMaxX = allBMaxY = allBMaxZ = -1e30f;

		for (int i = 0; i < count; ++i) {
			const LightBounds& lb = lightBoundsArr[i];
			if (lb.phi > 0.f) {
				bvhLights.push_back({i, lb});
				// Expand scene AABB
				allBMinX = std::min(allBMinX, lb.bMinX);
				allBMinY = std::min(allBMinY, lb.bMinY);
				allBMinZ = std::min(allBMinZ, lb.bMinZ);
				allBMaxX = std::max(allBMaxX, lb.bMaxX);
				allBMaxY = std::max(allBMaxY, lb.bMaxY);
				allBMaxZ = std::max(allBMaxZ, lb.bMaxZ);
			}
		}

		if (!bvhLights.empty())
			buildBVH(bvhLights, 0, (int)bvhLights.size(), 0u, 0);
	}

	bool Empty() const { return nodes_.empty(); }

	// -----------------------------------------------------------------------
	// Sample(p, n, u) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â O(log N) stochastic BVH descent
	// Returns empty optional when no light can contribute.
	// u is consumed (rescaled at each level).
	// Mirrors pbrt-v4 BVHLightSampler::Sample(LightSampleContext, Float).
	// -----------------------------------------------------------------------
	std::optional<SampledLight2> Sample(float px, float py, float pz,
										 float nx, float ny, float nz,
										 float u) const
	{
		if (nodes_.empty()) return std::nullopt;

		int   nodeIndex = 0;
		float pmf       = 1.f;

		// Clamp u to [0, OneMinusEpsilon] as in pbrt-v4
		u = std::min(u, 1.f - 1e-7f);

		while (true) {
			const LightBVHNode& node = nodes_[nodeIndex];
			if (!node.isLeaf) {
				// Compute importance of both children
				const LightBVHNode* children[2] = {
					&nodes_[nodeIndex + 1],
					&nodes_[node.childOrLightIndex]
				};
				float ci[2] = {
					children[0]->lightBounds.Importance(
						px,py,pz, nx,ny,nz,
						allBMinX,allBMinY,allBMinZ, allBMaxX,allBMaxY,allBMaxZ),
					children[1]->lightBounds.Importance(
						px,py,pz, nx,ny,nz,
						allBMinX,allBMinY,allBMinZ, allBMaxX,allBMaxY,allBMaxZ)
				};
				if (ci[0] == 0.f && ci[1] == 0.f) return std::nullopt;

				// Pick child proportional to importance ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â mirrors pbrt-v4 SampleDiscrete
				float sum = ci[0] + ci[1];
				float nodePMF;
				int child;
				if (u < ci[0] / sum) {
					child    = 0;
					nodePMF  = ci[0] / sum;
					u        = u / nodePMF;          // rescale u for next level
				} else {
					child    = 1;
					nodePMF  = ci[1] / sum;
					u        = (u - ci[0]/sum) / nodePMF;
				}
				u = std::min(u, 1.f - 1e-7f);
				pmf *= nodePMF;
				nodeIndex = (child == 0) ? (nodeIndex + 1)
										 : (int)node.childOrLightIndex;
			} else {
				// Leaf ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â verify non-zero importance (skip root-only degenerate case)
				if (nodeIndex > 0 ||
					nodes_[nodeIndex].lightBounds.Importance(
						px,py,pz, nx,ny,nz,
						allBMinX,allBMinY,allBMinZ, allBMaxX,allBMaxY,allBMaxZ) > 0.f)
				{
					return SampledLight2{(int)node.childOrLightIndex, pmf};
				}
				return std::nullopt;
			}
		}
	}

	// -----------------------------------------------------------------------
	// PMF(p, n, lightIndex) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â O(log N) bitTrail replay
	// Mirrors pbrt-v4 BVHLightSampler::PMF(LightSampleContext, Light).
	// Returns 0 if the light is not in the BVH.
	// -----------------------------------------------------------------------
	float PMF(float px, float py, float pz,
			  float nx, float ny, float nz,
			  int lightIndex) const
	{
		auto it = lightToBitTrail_.find(lightIndex);
		if (it == lightToBitTrail_.end()) return 0.f;

		uint32_t bitTrail = it->second;
		float pmf = 1.f;
		int nodeIndex = 0;

		while (true) {
			const LightBVHNode& node = nodes_[nodeIndex];
			if (node.isLeaf) return pmf;

			const LightBVHNode* child0 = &nodes_[nodeIndex + 1];
			const LightBVHNode* child1 = &nodes_[node.childOrLightIndex];
			float ci[2] = {
				child0->lightBounds.Importance(
					px,py,pz, nx,ny,nz,
					allBMinX,allBMinY,allBMinZ, allBMaxX,allBMaxY,allBMaxZ),
				child1->lightBounds.Importance(
					px,py,pz, nx,ny,nz,
					allBMinX,allBMinY,allBMinZ, allBMaxX,allBMaxY,allBMaxZ)
			};
			float sum = ci[0] + ci[1];
			if (sum == 0.f) return 0.f;

			int branch = (int)(bitTrail & 1u);
			pmf      *= ci[branch] / sum;
			nodeIndex = (branch == 0) ? (nodeIndex + 1)
									  : (int)node.childOrLightIndex;
			bitTrail >>= 1;
		}
	}

	// Accessors for testing / introspection
	int NodeCount()  const { return (int)nodes_.size(); }
	float AllBMinX() const { return allBMinX; }
	float AllBMinY() const { return allBMinY; }
	float AllBMinZ() const { return allBMinZ; }
	float AllBMaxX() const { return allBMaxX; }
	float AllBMaxY() const { return allBMaxY; }
	float AllBMaxZ() const { return allBMaxZ; }

private:
	// -----------------------------------------------------------------------
	// EvaluateCost ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â solid-angle ÃƒÆ’Ã¢â‚¬â€ area SAH heuristic (pbrt-v4 Ãƒâ€šÃ‚Â§12.6)
	// -----------------------------------------------------------------------
	static float EvaluateCost(const LightBounds& b,
							   float bMinX, float bMinY, float bMinZ,
							   float bMaxX, float bMaxY, float bMaxZ,
							   int dim)
	{
		// Directional measure M_omega
		float theta_o   = std::acos(std::max(-1.f, std::min(1.f, b.cosTheta_o)));
		float theta_e   = std::acos(std::max(-1.f, std::min(1.f, b.cosTheta_e)));
		float theta_w   = std::min(theta_o + theta_e, kPi);
		float sinTheta_o = SafeSqrt(1.f - Sqr(b.cosTheta_o));
		float M_omega = 2.f * kPi * (1.f - b.cosTheta_o) +
						(kPi / 2.f) *
							(2.f * theta_w * sinTheta_o
							 - std::cos(theta_o - 2.f * theta_w)
							 - 2.f * theta_o * sinTheta_o
							 + b.cosTheta_o);

		// AABB surface area
		float dx = b.bMaxX - b.bMinX;
		float dy = b.bMaxY - b.bMinY;
		float dz = b.bMaxZ - b.bMinZ;
		float sa = 2.f * (dx*dy + dy*dz + dz*dx);

		// Kr: ratio of max diagonal component to the split-dimension component
		float diagX = bMaxX - bMinX;
		float diagY = bMaxY - bMinY;
		float diagZ = bMaxZ - bMinZ;
		float diag[3] = { diagX, diagY, diagZ };
		float maxDiag = std::max({diagX, diagY, diagZ});
		float Kr = (diag[dim] > 0.f) ? (maxDiag / diag[dim]) : 1.f;

		return b.phi * M_omega * Kr * sa;
	}

	// -----------------------------------------------------------------------
	// buildBVH ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â recursive SAH BVH construction (mirrors pbrt-v4 exactly)
	// Returns {nodeIndex, mergedLightBounds} for the subtree.
	// -----------------------------------------------------------------------
	std::pair<int,LightBounds> buildBVH(
		std::vector<std::pair<int,LightBounds>>& bvhLights,
		int start, int end, uint32_t bitTrail, int depth)
	{
		assert(start < end);

		// --- Leaf case ---
		if (end - start == 1) {
			int nodeIndex = (int)nodes_.size();
			const LightBounds& lb = bvhLights[start].second;
			CompactLightBounds cb(lb,
				allBMinX,allBMinY,allBMinZ, allBMaxX,allBMaxY,allBMaxZ);
			int lightIndex = bvhLights[start].first;
			nodes_.push_back(LightBVHNode::MakeLeaf((unsigned)lightIndex, cb));
			lightToBitTrail_[lightIndex] = bitTrail;
			return {nodeIndex, lb};
		}

		// --- Compute centroid bounds for split-dimension selection ---
		float cbMinX =  1e30f, cbMinY =  1e30f, cbMinZ =  1e30f;
		float cbMaxX = -1e30f, cbMaxY = -1e30f, cbMaxZ = -1e30f;
		float bMinX  =  1e30f, bMinY  =  1e30f, bMinZ  =  1e30f;
		float bMaxX  = -1e30f, bMaxY  = -1e30f, bMaxZ  = -1e30f;
		for (int i = start; i < end; ++i) {
			const LightBounds& lb = bvhLights[i].second;
			float cx = lb.CentroidX(), cy = lb.CentroidY(), cz = lb.CentroidZ();
			cbMinX = std::min(cbMinX, cx); cbMaxX = std::max(cbMaxX, cx);
			cbMinY = std::min(cbMinY, cy); cbMaxY = std::max(cbMaxY, cy);
			cbMinZ = std::min(cbMinZ, cz); cbMaxZ = std::max(cbMaxZ, cz);
			bMinX = std::min(bMinX, lb.bMinX); bMaxX = std::max(bMaxX, lb.bMaxX);
			bMinY = std::min(bMinY, lb.bMinY); bMaxY = std::max(bMaxY, lb.bMaxY);
			bMinZ = std::min(bMinZ, lb.bMinZ); bMaxZ = std::max(bMaxZ, lb.bMaxZ);
		}

		// --- 12-bucket SAH over all three dimensions ---
		constexpr int nBuckets = 12;
		float minCost  = 1e30f;
		int   minBucket = -1, minDim = -1;

		for (int dim = 0; dim < 3; ++dim) {
			float cbMin = (dim==0)?cbMinX:(dim==1)?cbMinY:cbMinZ;
			float cbMax = (dim==0)?cbMaxX:(dim==1)?cbMaxY:cbMaxZ;
			if (cbMin == cbMax) continue;

			LightBounds buckets[nBuckets];
			for (int i = start; i < end; ++i) {
				const LightBounds& lb = bvhLights[i].second;
				float c = (dim==0)?lb.CentroidX():(dim==1)?lb.CentroidY():lb.CentroidZ();
				int b = (int)(nBuckets * (c - cbMin) / (cbMax - cbMin));
				if (b == nBuckets) b = nBuckets - 1;
				buckets[b] = Union(buckets[b], lb);
			}

			float cost[nBuckets - 1];
			for (int i = 0; i < nBuckets - 1; ++i) {
				LightBounds b0, b1;
				for (int j = 0;     j <= i;          ++j) b0 = Union(b0, buckets[j]);
				for (int j = i + 1; j < nBuckets;    ++j) b1 = Union(b1, buckets[j]);
				cost[i] = EvaluateCost(b0, bMinX,bMinY,bMinZ, bMaxX,bMaxY,bMaxZ, dim)
						 + EvaluateCost(b1, bMinX,bMinY,bMinZ, bMaxX,bMaxY,bMaxZ, dim);
			}

			for (int i = 1; i < nBuckets - 1; ++i) {
				if (cost[i] > 0.f && cost[i] < minCost) {
					minCost   = cost[i];
					minBucket = i;
					minDim    = dim;
				}
			}
		}

		// --- Partition ---
		int mid;
		if (minDim == -1) {
			mid = (start + end) / 2;
		} else {
			float cbMin = (minDim==0)?cbMinX:(minDim==1)?cbMinY:cbMinZ;
			float cbMax = (minDim==0)?cbMaxX:(minDim==1)?cbMaxY:cbMaxZ;
			auto* pmid = std::partition(
				&bvhLights[start], &bvhLights[end - 1] + 1,
				[&](const std::pair<int,LightBounds>& l) {
					float c = (minDim==0)?l.second.CentroidX()
							 :(minDim==1)?l.second.CentroidY()
							 :            l.second.CentroidZ();
					int b = (int)(nBuckets * (c - cbMin) / (cbMax - cbMin));
					if (b == nBuckets) b = nBuckets - 1;
					return b <= minBucket;
				});
			mid = (int)(pmid - &bvhLights[0]);
			if (mid == start || mid == end)
				mid = (start + end) / 2;
		}

		// --- Allocate interior node placeholder, recurse ---
		int nodeIndex = (int)nodes_.size();
		nodes_.push_back(LightBVHNode());   // placeholder
		assert(depth < 64);

		auto [i0, lb0] = buildBVH(bvhLights, start, mid,
								   bitTrail, depth + 1);
		assert(nodeIndex + 1 == i0);
		auto [i1, lb1] = buildBVH(bvhLights, mid, end,
								   bitTrail | (1u << depth), depth + 1);

		LightBounds lb = Union(lb0, lb1);
		CompactLightBounds cb(lb,
			allBMinX,allBMinY,allBMinZ, allBMaxX,allBMaxY,allBMaxZ);
		nodes_[nodeIndex] = LightBVHNode::MakeInterior((unsigned)i1, cb);
		return {nodeIndex, lb};
	}

	// -----------------------------------------------------------------------
	// Private data  (mirrors pbrt-v4 BVHLightSampler private members)
	// -----------------------------------------------------------------------
	std::vector<LightBVHNode>         nodes_;
	std::unordered_map<int,uint32_t>  lightToBitTrail_;
	float allBMinX = 0.f, allBMinY = 0.f, allBMinZ = 0.f;
	float allBMaxX = 0.f, allBMaxY = 0.f, allBMaxZ = 0.f;

	static constexpr float kPi = 3.14159265358979323846f;
};
