#pragma once
//==============================================================================
// bluenoise_sampler.h -- Blue-noise Cranley-Patterson sampler (pbrt-v4 §8.x)
//
// pbrt-v4 reference: src/pbrt/samplers.h  (PMJ02BNSampler)
//                    src/pbrt/util/bluenoise.h
//
// Algorithm (Cranley-Patterson rotation with blue-noise offsets):
//   A base Halton sequence provides low-discrepancy stratification.
//   A per-pixel, per-dimension blue-noise offset is added (mod 1) to break
//   structured correlation across pixels -- the key insight from pbrt-v4's
//   PMJ02BNSampler.  The blue-noise offsets have the property that their
//   error distributes as high-frequency noise, which is far less visible
//   than the low-frequency clumping produced by random offsets.
//
// Relationship to existing samplers:
//   - halton_sampler.h  : full Halton sequence, no blue-noise rotation
//   - path_sampler.h    : lightweight Halton for per-bounce decisions
//   - BlueNoiseSampler  : Halton + blue-noise Cranley-Patterson (this file)
//
// API (matches halton_sampler / path_sampler conventions):
//   BlueNoiseSampler s(samples_per_pixel, image_width, image_height);
//   s.start_pixel_sample(px, py, sample_index);
//   float u   = s.get_1d();        // one 1-D sample
//   auto [u,v] = s.get_2d();       // one 2-D sample pair
//
// Dimensions 0-1 are the pixel-2D (camera jitter) dimensions and use the
// first two Halton bases; higher dimensions use successive primes.
// Each dimension maps to a unique blue-noise texture (wraps mod 48).
//==============================================================================

#include <cstdint>
#include <cmath>
#include <cassert>
#include "bluenoise.h"

// ---------------------------------------------------------------------------
// Internal helpers: radical inverse (base-n Halton)
// ---------------------------------------------------------------------------
namespace bns_detail {

static constexpr double OneMinusEpsilon = 1.0 - 1.1102230246251565e-16;

// First 64 primes for Halton bases (dimensions 0..63)
inline const int* get_primes() {
	static const int p[64] = {
		2,   3,   5,   7,  11,  13,  17,  19,  23,  29,  31,  37,  41,  43,
		47,  53,  59,  61,  67,  71,  73,  79,  83,  89,  97, 101, 103, 107,
		109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
		191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263,
		269, 271, 277, 281, 283, 293, 307, 311
	};
	return p;
}

// Compute radical inverse of `a` in the prime base at index `base_idx`.
inline double radical_inverse(int base_idx, uint64_t a) {
	const int base = get_primes()[base_idx];
	const double inv_base = 1.0 / base;
	double inv_base_m = 1.0;
	uint64_t limit    = ~0ull / base - base;
	uint64_t reversed = 0;
	while (a && reversed < limit) {
		uint64_t next  = a / base;
		uint64_t digit = a - next * base;
		reversed = reversed * base + digit;
		inv_base_m *= inv_base;
		a = next;
	}
	return std::min(reversed * inv_base_m, OneMinusEpsilon);
}

// Fast 32-bit hash for per-pixel scramble seed.
inline uint32_t pixel_hash(int px, int py, int dim) {
	uint32_t h = (uint32_t)(px * 73856093 ^ py * 19349663 ^ dim * 83492791);
	h ^= h >> 16; h *= 0x45d9f3b; h ^= h >> 16;
	return h;
}

} // namespace bns_detail

// ---------------------------------------------------------------------------
// BlueNoiseSampler
// ---------------------------------------------------------------------------
class BlueNoiseSampler {
public:
	// samples_per_pixel : number of samples taken per pixel (for diagnostics)
	// image_w, image_h  : image dimensions (unused beyond documentation)
	BlueNoiseSampler(int samples_per_pixel = 64,
					 int /*image_w*/       = 0,
					 int /*image_h*/       = 0)
		: spp_(samples_per_pixel) {}

	// Call once per sample before calling get_1d() / get_2d().
	void start_pixel_sample(int px, int py, int sample_index) {
		px_        = px;
		py_        = py;
		sample_    = (uint64_t)sample_index;
		dimension_ = 0;
	}

	// Returns one 1-D sample in [0, 1) with blue-noise Cranley-Patterson offset.
	//
	// Base value : Halton(dimension_, sample_)   (stratified over samples)
	// BN offset  : blue_noise(dimension_, px_, py_)  (correlated across pixels
	//              in a visually pleasing high-frequency pattern)
	float get_1d() {
		int dim = dimension_++;
		// Halton base value
		double h = bns_detail::radical_inverse(dim % 64, sample_);
		// Cranley-Patterson rotation: add blue-noise offset mod 1
		float  bn = blue_noise(dim, px_, py_);
		double u  = h + bn;
		if (u >= 1.0) u -= 1.0;
		return (float)std::min(u, bns_detail::OneMinusEpsilon);
	}

	// Returns a 2-D sample pair in [0,1)^2.
	// Uses two consecutive dimensions so consecutive get_2d() calls advance
	// the dimension counter by 2, matching pbrt-v4's convention.
	std::pair<float, float> get_2d() {
		float u = get_1d();
		float v = get_1d();
		return {u, v};
	}

	// Convenience: same as get_2d() for camera jitter usage.
	std::pair<float, float> get_pixel_2d() { return get_2d(); }

	int samples_per_pixel() const { return spp_; }

private:
	int      spp_;
	int      px_  = 0;
	int      py_  = 0;
	uint64_t sample_    = 0;
	int      dimension_ = 0;
};
