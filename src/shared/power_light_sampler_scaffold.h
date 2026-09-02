#pragma once
// ---------------------------------------------------------------------------
// power_light_sampler_scaffold.h -- Power-proportional light sampler with
// alias table (orphaned scaffolding - see docs/FEATURE_INVENTORY.md's
// "Present But Not Wired In" section)
//
// Renamed from power_light_sampler.h: that name collided with
// src/TheRestOfYourLife/power_light_sampler.h (a separate, actually-used
// CPU light sampler with the same filename, resolved only by C++'s
// same-directory-first / include-path-order include rules - fragile).
// Kept the "_scaffold" suffix rather than deleting it outright: it's a
// real, complete implementation with zero callers (this codebase's actual
// GPU alias-table light sampling, gpu/optix/optix_renderer_scene.cpp /
// optix_types.h, implements the same Vose-alias-table algorithm natively
// rather than through this class) - matching this file's existing peers
// in the same "present but not wired in" category (UniformLightSampler,
// ExhaustiveLightSampler), not a gap to close here. BVHLightSampler2
// (src/shared/bvh_light_sampler2.h) is NOT in this category despite
// looking similar - it IS instantiated in production
// (optix_renderer_scene.cpp builds a real tree from it); only its
// downstream device-side consumption is separately disabled (unrelated
// crash, tracked elsewhere), not orphaned scaffolding like this file.
//
// Mirrors pbrt-v4 PowerLightSampler (src/pbrt/lightsamplers.h/cpp).
//
// Algorithm: Vose alias table (1991) — O(n) build, O(1) sample.
//   For n lights with powers p[0..n-1]:
//     1. Normalise weights: q[i] = p[i] / sum(p).
//     2. Build alias table: each slot i stores (prob[i], alias[i]) so that
//          Sample(u) = i  if frac(u*n) < prob[i], else alias[i].
//     3. PMF(i) = p[i] / sum(p).
//
// Design rules (same as bxdfs.h, pmj02_sampler.h):
//   - Header-only, no heap allocation: maximum light count is fixed at
//     compile time via MaxLights (default 64).
//   - CPU_GPU tagged — usable on both CPU and GPU.
//   - Works with any collection of light structs that expose a power() method.
//
// Usage (with punctual_lights.h types):
//
//   PointLightData<double> lights[3] = { ... };
//   double powers[3];
//   for (int i = 0; i < 3; ++i) powers[i] = lights[i].power();
//   PowerLightSampler<64> sampler(powers, 3);
//
//   double u = /* uniform [0,1) */;
//   int idx = sampler.sample(u);           // light index
//   double pmf = sampler.pmf(idx);         // selection probability
//
// pbrt-v4 reference: src/pbrt/lightsamplers.h PowerLightSampler,
//                    src/pbrt/util/containers.h AliasTable
// ---------------------------------------------------------------------------

#include <cmath>

#include "cpu_gpu.h"

// ---------------------------------------------------------------------------
// PowerLightSampler<MaxLights>
//   MaxLights: compile-time upper bound on number of lights (default 64)
// ---------------------------------------------------------------------------
template <int MaxLights = 64>
class PowerLightSampler {
public:
	CPU_GPU PowerLightSampler() : n_(0) {}

	// Build the alias table from an array of n power values.
	// powers[i] must be >= 0. If all are zero, falls back to uniform.
	CPU_GPU PowerLightSampler(const double* powers, int n) {
		build(powers, n);
	}

	// Convenience overload for float arrays
	CPU_GPU PowerLightSampler(const float* powers, int n) {
		double tmp[MaxLights];
		for (int i = 0; i < n && i < MaxLights; ++i) tmp[i] = double(powers[i]);
		build(tmp, (n < MaxLights) ? n : MaxLights);
	}

	// Sample a light index given u in [0,1).
	// Returns -1 if no lights.
	CPU_GPU int sample(double u) const {
		if (n_ == 0) return -1;
		// Map u to slot index + fractional part
		double scaled = u * double(n_);
		int slot = int(scaled);
		if (slot >= n_) slot = n_ - 1;
		double frac = scaled - double(slot);
		return (frac < prob_[slot]) ? slot : alias_[slot];
	}

	// PMF of light index i (= normalised power).
	CPU_GPU double pmf(int i) const {
		if (n_ == 0 || i < 0 || i >= n_) return 0.0;
		return pmf_[i];
	}

	// Number of lights registered.
	CPU_GPU int size() const { return n_; }

	// Weighted sum of all powers (un-normalised total).
	CPU_GPU double total_power() const { return total_; }

private:
	int    n_;
	double prob_[MaxLights];   // alias table threshold for each slot
	int    alias_[MaxLights];  // alias index for each slot
	double pmf_[MaxLights];    // normalised probability mass for each light
	double total_;             // sum of all powers

	CPU_GPU void build(const double* powers, int n) {
		n_ = (n <= MaxLights) ? n : MaxLights;
		total_ = 0.0;
		for (int i = 0; i < n_; ++i) total_ += powers[i];

		// Handle degenerate case: all powers zero → uniform
		bool use_uniform = (total_ <= 0.0);
		double uniform_p = 1.0 / double(n_);

		for (int i = 0; i < n_; ++i)
			pmf_[i] = use_uniform ? uniform_p : powers[i] / total_;

		// Vose alias construction
		// scaled[i] = pmf[i] * n  (average = 1)
		double scaled[MaxLights];
		int small[MaxLights], large[MaxLights];
		int ns = 0, nl = 0;

		for (int i = 0; i < n_; ++i) {
			scaled[i] = pmf_[i] * double(n_);
			prob_[i]  = 0.0;
			alias_[i] = i;
			if (scaled[i] < 1.0)
				small[ns++] = i;
			else
				large[nl++] = i;
		}

		while (ns > 0 && nl > 0) {
			int s = small[--ns];
			int l = large[--nl];
			prob_[s]  = scaled[s];
			alias_[s] = l;
			scaled[l] = (scaled[l] + scaled[s]) - 1.0;
			if (scaled[l] < 1.0)
				small[ns++] = l;
			else
				large[nl++] = l;
		}
		// Remaining entries are exactly prob=1
		while (nl > 0) { int l = large[--nl]; prob_[l] = 1.0; alias_[l] = l; }
		while (ns > 0) { int s = small[--ns]; prob_[s] = 1.0; alias_[s] = s; }
	}
};
