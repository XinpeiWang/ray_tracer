// exhaustive_light_sampler.h
// ===========================================================================
// ExhaustiveLightSampler
//
// An O(N) importance-weighted light sampler that evaluates every bounded
// light at each query point using a WeightedReservoirSampler.  This gives
// exactly correct importance-proportional PMFs at the cost of linear time,
// making it the reference-quality sampler for validation and small scenes.
//
// This mirrors pbrt-v4 ExhaustiveLightSampler (src/pbrt/lightsamplers.h/.cpp).
//
// Algorithm (mirrors pbrt-v4 exactly):
//   Lights are split at construction into:
//     - "infinite" lights  (LightBounds with phi == 0 used as sentinel, or
//                           lights that have no spatial bounds)
//     - "bounded" lights   (have a valid LightBounds with phi > 0)
//
//   Sample(ctx, u):
//     1. With probability p_inf = n_inf / (n_inf + (n_bounded > 0 ? 1 : 0))
//        pick uniformly among infinite lights.
//     2. Otherwise remap u and use WeightedReservoirSampler over all bounded
//        lights weighted by LightBounds::Importance(lb, p, n).
//
//   PMF(ctx, lightIndex):
//     - Infinite lights: uniform share of p_inf.
//     - Bounded lights:  importance / importanceSum * (1 - p_inf).
//
// Design rules (same as bvh_light_sampler2.h / power_light_sampler.h):
//   - Header-only, no virtual functions, no pbrt Allocator.
//   - Template-free; lights are identified by integer index into the array
//     passed at construction.
//   - CPU_GPU tagged for future GPU portability.
//   - LightBounds for a light is identified by the same index; pass phi == 0
//     to mark a light as "infinite" (no spatial bounds).
//
// Usage:
//   // Build sampler (bounded lights have phi > 0 in their LightBounds)
//   std::vector<LightBounds> lbs = { ... };
//   ExhaustiveLightSampler sampler(lbs.data(), (int)lbs.size());
//
//   // Query at a surface point with normal (0,1,0)
//   auto result = sampler.Sample(px,py,pz, nx,ny,nz, u);
//   if (result.lightIndex >= 0) {
//       // use result.lightIndex, result.pmf
//   }
//
//   // Evaluate PMF for a specific light
//   float p = sampler.PMF(px,py,pz, nx,ny,nz, lightIndex);
//
// References:
//   pbrt-v4 src/pbrt/lightsamplers.h  ExhaustiveLightSampler  (Apache-2.0)
//   pbrt-v4 src/pbrt/lightsamplers.cpp ExhaustiveLightSampler (Apache-2.0)
// ===========================================================================

#pragma once

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

#include "light_bounds.h"
#include "reservoir_sampler.h"

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU inline
#   endif
#endif

// ---------------------------------------------------------------------------
// SampledLightEx -- result of ExhaustiveLightSampler::Sample
// ---------------------------------------------------------------------------
struct SampledLightEx {
	int   lightIndex = -1;  // index into original lights array (-1 = none)
	float pmf        = 0.f; // selection probability
};

// ---------------------------------------------------------------------------
// ExhaustiveLightSampler
// ---------------------------------------------------------------------------
class ExhaustiveLightSampler {
public:
	ExhaustiveLightSampler() = default;

	// Construct from an array of LightBounds, one per light.
	// A light with lb.phi == 0 is treated as an "infinite" light (no bounds).
	// Mirrors pbrt-v4 ExhaustiveLightSampler::ExhaustiveLightSampler.
	explicit ExhaustiveLightSampler(const LightBounds* lbs, int n)
	{
		for (int i = 0; i < n; ++i) {
			if (lbs[i].phi > 0.f) {
				boundedIndices_.push_back(i);
				boundedLB_.push_back(lbs[i]);
			} else {
				infiniteIndices_.push_back(i);
			}
		}
	}

	// Overload accepting std::vector<LightBounds>.
	explicit ExhaustiveLightSampler(const std::vector<LightBounds>& lbs)
		: ExhaustiveLightSampler(lbs.data(), (int)lbs.size()) {}

	bool empty() const {
		return infiniteIndices_.empty() && boundedIndices_.empty();
	}
	int total_lights()    const { return (int)(infiniteIndices_.size() + boundedIndices_.size()); }
	int infinite_count()  const { return (int)infiniteIndices_.size(); }
	int bounded_count()   const { return (int)boundedIndices_.size(); }

	// -----------------------------------------------------------------------
	// Sample(px,py,pz, nx,ny,nz, u)
	//   Returns SampledLightEx{lightIndex, pmf} or {-1, 0} if empty.
	//
	// Mirrors pbrt-v4 ExhaustiveLightSampler::Sample(ctx, u).
	// -----------------------------------------------------------------------
	SampledLightEx Sample(float px, float py, float pz,
						  float nx, float ny, float nz,
						  float u) const
	{
		if (empty()) return {};

		int nInf     = (int)infiniteIndices_.size();
		int nBounded = (int)boundedIndices_.size();

		// p_inf = n_inf / (n_inf + (any bounded lights ? 1 : 0))
		// mirrors pbrt-v4: Float(infiniteLights.size()) /
		//                  Float(infiniteLights.size() + (!lightBounds.empty() ? 1 : 0))
		float p_inf = float(nInf) / float(nInf + (nBounded > 0 ? 1 : 0));

		if (u < p_inf) {
			// Select uniformly among infinite lights
			u /= p_inf;
			int idx = std::min(int(u * nInf), nInf - 1);
			float pdf = p_inf / float(nInf);
			return { infiniteIndices_[idx], pdf };
		} else {
			// Remap u into [0,1) for the bounded branch
			float u_remapped = std::min((u - p_inf) / (1.f - p_inf), 1.f - 1e-7f);

			// Use WeightedReservoirSampler over all bounded lights
			// Seed deterministically from the remapped sample
			uint64_t seed = mix_bits(uint64_t(u_remapped * float(1ULL << 32)));
			WeightedReservoirSampler<int> wrs(seed);

			for (int i = 0; i < nBounded; ++i) {
				float imp = Importance(boundedLB_[i], px, py, pz, nx, ny, nz);
				wrs.add(i, double(imp));
			}

			if (!wrs.has_sample()) return {};

			int slot = wrs.sample();         // index into boundedIndices_
			float pmf = float((1.0 - p_inf) * wrs.sample_probability());
			return { boundedIndices_[slot], pmf };
		}
	}

	// -----------------------------------------------------------------------
	// PMF(px,py,pz, nx,ny,nz, lightIndex)
	//   Returns the probability that Sample() would return lightIndex.
	//
	// Mirrors pbrt-v4 ExhaustiveLightSampler::PMF(ctx, light).
	// -----------------------------------------------------------------------
	float PMF(float px, float py, float pz,
			  float nx, float ny, float nz,
			  int lightIndex) const
	{
		if (empty()) return 0.f;

		int nInf     = (int)infiniteIndices_.size();
		int nBounded = (int)boundedIndices_.size();
		float p_inf  = float(nInf) / float(nInf + (nBounded > 0 ? 1 : 0));

		// Check if lightIndex is infinite
		for (int i : infiniteIndices_) {
			if (i == lightIndex)
				return (nInf > 0) ? p_inf / float(nInf) : 0.f;
		}

		// Must be a bounded light -- find its LightBounds
		int slot = -1;
		for (int s = 0; s < nBounded; ++s) {
			if (boundedIndices_[s] == lightIndex) { slot = s; break; }
		}
		if (slot < 0) return 0.f;

		// importance / importanceSum * (1 - p_inf)
		// mirrors pbrt-v4 PMF: lightImportance / importanceSum * (1 - pInfinite)
		float importanceSum  = 0.f;
		float lightImportance = 0.f;
		for (int i = 0; i < nBounded; ++i) {
			float imp = Importance(boundedLB_[i], px, py, pz, nx, ny, nz);
			importanceSum += imp;
			if (i == slot) lightImportance = imp;
		}

		if (importanceSum == 0.f) return 0.f;
		return lightImportance / importanceSum * (1.f - p_inf);
	}

private:
	// MixBits: bijective 64-bit hash used to seed the reservoir RNG.
	// Mirrors pbrt-v4 MixBits (util/math.h).
	static uint64_t mix_bits(uint64_t v) {
		v ^= (v >> 31); v *= 0x7fb5d329728ea185ULL;
		v ^= (v >> 27); v *= 0x81dadef4bc2dd44dULL;
		v ^= (v >> 33);
		return v;
	}

	std::vector<int>         infiniteIndices_; // original light indices with phi==0
	std::vector<int>         boundedIndices_;  // original light indices with phi>0
	std::vector<LightBounds> boundedLB_;       // LightBounds parallel to boundedIndices_
};
