#pragma once
//==============================================================================
// independent_sampler.h -- pbrt-v4 IndependentSampler
//
// The simplest sampler pbrt-v4 offers: every Get1D()/Get2D() call is a plain
// draw from a per-pixel-seeded RNG, with no stratification, low-discrepancy
// sequence, or dimension-aware structure at all. Included for `.pbrt`/
// --sampler fidelity (a scene authored with `Sampler "independent"` used to
// silently fall back to Sobol, see camera.h's sampler_kind_from_name()) -
// Sobol strictly dominates it in practice, so this exists to honor the
// directive faithfully rather than to be anyone's recommended choice.
//
// Backed by src/shared/rng.h's RNG (canonical PCG32, pbrt-v4 util/rng.h).
//
// Usage (matches SobolSampler's shape -- construct fresh per ray, no
// start_pixel_sample()/persistent per-thread state needed, since every draw
// is independent of any prior one anyway):
//   IndependentSampler s(sample_idx, pixel_x, pixel_y);
//   double u = s.get();
//==============================================================================

#include <cstdint>
#include "rng.h"

class IndependentSampler {
  public:
	// sample_idx  -- index of this sample within the current pixel (0-based)
	// pixel_x/y   -- pixel coordinates, used to build a per-pixel RNG seed
	IndependentSampler(int sample_idx, int pixel_x, int pixel_y) {
		// Same per-pixel/per-sample hash-combine shape as SobolSampler's own
		// seeding (sobol_sampler.h) - distinct large-prime multipliers per
		// axis before MixBits, so adjacent pixels/samples don't share
		// correlated PCG32 streams.
		uint64_t px = static_cast<uint64_t>(pixel_x);
		uint64_t py = static_cast<uint64_t>(pixel_y);
		uint64_t si = static_cast<uint64_t>(sample_idx);
		uint64_t seed = MixBits(px * 2654435761ull ^ py * 805459861ull ^ si * 3266489917ull);
		rng_.SetSequence(seed);
	}

	// Return the next sample value in [0, 1). No dimension concept: unlike
	// every other sampler in this codebase, IndependentSampler makes no
	// attempt to decorrelate successive calls - that's the entire point of
	// what it's modeling.
	double get() { return rng_.Uniform<double>(); }

	// No-op: kept only so IndependentSampler satisfies the same duck-typed
	// interface (SobolSampler::reset_dim()) camera.h's integrator may call
	// on any sampler it's templated over.
	void reset_dim(int = 0) {}

  private:
	RNG rng_;
};
