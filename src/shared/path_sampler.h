#pragma once
//==============================================================================================
// path_sampler.h -- Halton path sampler (pbrt-v4 HaltonSampler-inspired, CPU-only)
//
// Provides per-path, per-dimension low-discrepancy samples using the Halton sequence.
// Each "dimension" corresponds to one random decision along a path; currently used
// for Russian Roulette decisions in the integrator.  Material scatter() calls
// still use random_double() internally -- threading full LD sampling through
// scatter() would require a larger interface change and is left as future work.
//
// Each dimension uses a different Halton base (first 16 primes), exactly as
// pbrt-v4 assigns scrambled Halton dimensions to individual path decisions.
// Per-pixel scramble seeds prevent structured correlation between pixels.
//
// Usage:
//   PathSampler ps(sample_index, pixel_x, pixel_y);
//   double u = ps.get();   // dimension 0 (e.g., RR at bounce 1)
//   double v = ps.get();   // dimension 1 (e.g., RR at bounce 2)
//   // ... one call per integrator-level random decision
//

#include <cstdint>
#include <cmath>

// First 16 primes used as Halton bases (dimensions 0..15).
// Dimension 0 = base-2, dimension 1 = base-3, etc.
inline constexpr int halton_primes[16] = {
	2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53
};

// Radical inverse in base `base` for integer `a`.
// Returns a value in [0, 1).
inline double radical_inverse(int base, uint64_t a) {
	double inv_base = 1.0 / static_cast<double>(base);
	double reversed = 0.0;
	double inv_base_n = 1.0;
	while (a > 0) {
		uint64_t next  = a / base;
		uint64_t digit = a - next * base;
		inv_base_n *= inv_base;
		reversed   += digit * inv_base_n;
		a           = next;
	}
	return reversed;
}

// Owen-scrambled radical inverse (pbrt-v4 ScrambledRadicalInverse style).
// Uses a simple multiplicative hash as scramble seed per (dimension, pixel).
// This breaks correlation between pixels while preserving the LD structure.
inline double scrambled_radical_inverse(int base, uint64_t a, uint32_t seed) {
	double inv_base  = 1.0 / static_cast<double>(base);
	double reversed  = 0.0;
	double inv_base_n = 1.0;
	while (a > 0) {
		uint64_t next  = a / base;
		uint64_t digit = a - next * base;
		// Scramble digit with seed (simple hash, not full Owen-tree)
		digit = (digit + seed) % static_cast<uint64_t>(base);
		inv_base_n *= inv_base;
		reversed   += digit * inv_base_n;
		seed = seed * 1664525u + 1013904223u;   // LCG advance
		a    = next;
	}
	return std::min(reversed, 1.0 - 1e-15);
}

// ---------------------------------------------------------------------------
// PathSampler -- stateful per-path sampler.
//
// Construct once per camera ray (sample), then call get() for each random
// decision. Each call advances the internal dimension counter by one.
// Wraps back to dimension 0 if more than max_dims dimensions are requested
// (falls back to independent random for extra dimensions).
// ---------------------------------------------------------------------------
class PathSampler {
  public:
	static constexpr int max_dims = 16;

	// sample_idx  -- index of this sample within the current pixel (0-based)
	// pixel_x/y   -- pixel coordinates, used to build per-pixel scramble seeds
	PathSampler(int sample_idx, int pixel_x, int pixel_y)
		: index_(static_cast<uint64_t>(sample_idx)),
		  dim_(0)
	{
		// Build per-dimension, per-pixel scramble seeds using a mixing hash.
		// Different seed per pixel prevents structured correlation artifacts.
		uint32_t px = static_cast<uint32_t>(pixel_x);
		uint32_t py = static_cast<uint32_t>(pixel_y);
		for (int d = 0; d < max_dims; ++d) {
			uint32_t h = px * 2654435761u ^ py * 1013904223u ^ static_cast<uint32_t>(d) * 2246822519u;
			h ^= h >> 17; h *= 0xbf324c81u; h ^= h >> 11;
			seeds_[d] = h;
		}
		// Seed the beyond-max_dims fallback LCG (see get() below) from a hash
		// of pixel + sample index rather than a fixed constant - see
		// sobol_sampler.h's SobolSampler for why a shared fixed seed here is a
		// real bug, not just a cosmetic one: any path needing more than
		// max_dims random draws (e.g. Russian Roulette through many bounces)
		// would otherwise replay the identical "random" sequence for every
		// pixel and every sample past that point, producing a strong bias
		// that does not shrink with more spp.
		uint32_t fh = px * 2654435761u ^ py * 1013904223u
					^ static_cast<uint32_t>(sample_idx) * 2246822519u;
		fh ^= fh >> 17; fh *= 0xbf324c81u; fh ^= fh >> 11;
		fallback_ = (static_cast<uint64_t>(fh) << 32) | fh;
	}

	// Return the next sample dimension value in [0, 1).
	double get() {
		if (dim_ >= max_dims) {
			// Exceeded pre-computed dimensions: fall back to LCG random.
			fallback_ = fallback_ * 6364136223846793005ULL + 1442695040888963407ULL;
			return static_cast<double>((fallback_ >> 33) & 0x7FFFFFFF) / 2147483647.0;
		}
		int base = halton_primes[dim_];
		double val = scrambled_radical_inverse(base, index_, seeds_[dim_]);
		++dim_;
		return val;
	}

	// Reset dimension counter (e.g., to re-use sampler for another vertex).
	void reset_dim(int d = 0) { dim_ = d; }

  private:
	uint64_t index_;
	int      dim_;
	uint32_t seeds_[max_dims];
	uint64_t fallback_ = 0x123456789ABCDEFull;
};
