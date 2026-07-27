#pragma once
//==============================================================================
// pmj02_sampler.h -- PMJ02BN Sampler (pbrt-v4 default sampler port)
//
// pbrt-v4 reference: src/pbrt/samplers.h        (PMJ02BNSampler class)
//                    src/pbrt/samplers.cpp       (constructor / sorting logic)
//                    src/pbrt/util/pmj02tables.h (GetPMJ02BNSample)
//                    src/pbrt/util/bluenoise.h   (BlueNoise / CP rotation)
//
// Algorithm:
//   PMJ02 (Progressive Multi-Jittered 02-sequence) generates 2D sample sets
//   that are simultaneously well-stratified in 1D projections onto both axes
//   AND their 2D strata (elementary intervals).  This gives better convergence
//   than Halton or Sobol for path-tracing integrands.
//
//   Key operations (matching pbrt-v4 exactly):
//
//   get_pixel_2d():
//     Uses pre-sorted per-pixel slices of set-0 of the PMJ02BN table.
//     Constructor bins all 65536 samples by their fractional pixel coordinate
//     into a pixelTileSize x pixelTileSize grid; get_pixel_2d() returns the
//     (px%tile, py%tile)-th slice.
//
//   get_2d():
//     For low dimensions (pmjInstance < nPMJ02bnSets=5): looks up the matching
//     PMJ02BN set directly.  For higher dimensions falls back to permuted index.
//     Applies blue-noise Cranley-Patterson rotation to de-correlate pixels.
//
//   get_1d():
//     Uses PermutationElement to map sampleIndex -> stratified position, adds
//     a blue-noise delta for visual quality, and returns (index+delta)/spp.
//
// API (matches halton_sampler / bluenoise_sampler conventions):
//   PMJ02BNSampler s(samples_per_pixel);     // spp should be power-of-4
//   s.start_pixel_sample(px, py, sample_index);
//   float  u    = s.get_1d();
//   auto [u, v] = s.get_2d();
//   auto [u, v] = s.get_pixel_2d();          // for camera lens/pixel jitter
//==============================================================================

#include <cstdint>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include <utility>

#include "pmj02_data.h"   // 5 x 65536 x 2 uint32_t table
#include "bluenoise.h"    // blue_noise(dim, px, py)

// ---------------------------------------------------------------------------
// Shared math utilities (self-contained, no dependency on halton_sampler.h)
// Verbatim ports of pbrt-v4 helpers.
// ---------------------------------------------------------------------------
namespace pmj02_detail {

static constexpr float OneMinusEpsilonF = 1.0f - 1.1920929e-7f;  // float version

// MixBits / MurmurHash64A / Hash -- pbrt-v4 src/pbrt/util/hash.h
inline uint64_t mix_bits(uint64_t v) {
	v ^= (v >> 31); v *= 0x7fb5d329728ea185ull;
	v ^= (v >> 27); v *= 0x81dadef4bc2dd44dull;
	v ^= (v >> 33);
	return v;
}

inline uint64_t murmur_hash64a(const unsigned char* key, size_t len, uint64_t seed) {
	const uint64_t m = 0xc6a4a7935bd1e995ull;
	const int r = 47;
	uint64_t h = seed ^ (len * m);
	const unsigned char* end = key + 8 * (len / 8);
	while (key != end) {
		uint64_t k; memcpy(&k, key, 8); key += 8;
		k *= m; k ^= k >> r; k *= m;
		h ^= k; h *= m;
	}
	switch (len & 7) {
		case 7: h ^= uint64_t(key[6]) << 48; [[fallthrough]];
		case 6: h ^= uint64_t(key[5]) << 40; [[fallthrough]];
		case 5: h ^= uint64_t(key[4]) << 32; [[fallthrough]];
		case 4: h ^= uint64_t(key[3]) << 24; [[fallthrough]];
		case 3: h ^= uint64_t(key[2]) << 16; [[fallthrough]];
		case 2: h ^= uint64_t(key[1]) << 8;  [[fallthrough]];
		case 1: h ^= uint64_t(key[0]); h *= m;
	}
	h ^= h >> r; h *= m; h ^= h >> r;
	return h;
}

template <typename... Args>
inline uint64_t pmj_hash(Args... args) {
	constexpr size_t sz = (sizeof(Args) + ... + 0);
	constexpr size_t n  = (sz + 7) / 8;
	uint64_t buf[n];
	char* p = reinterpret_cast<char*>(buf);
	auto pack = [&](auto v) { memcpy(p, &v, sizeof(v)); p += sizeof(v); };
	(pack(args), ...);
	return murmur_hash64a(reinterpret_cast<const unsigned char*>(buf), sz, 0);
}

// PermutationElement -- Feistel-cipher bijection over {0..l-1}
// Verbatim port of pbrt-v4 PermutationElement (src/pbrt/util/math.h)
inline uint32_t permutation_element(uint32_t i, uint32_t l, uint32_t p) {
	uint32_t w = l - 1;
	w |= w >> 1; w |= w >> 2; w |= w >> 4; w |= w >> 8; w |= w >> 16;
	do {
		i ^= p;            i *= 0xe170893d;
		i ^= p >> 16;      i ^= (i & w) >> 4;
		i ^= p >> 8;       i *= 0x0929eb3f;
		i ^= p >> 23;      i ^= (i & w) >> 1;
		i *= 1 | p >> 27;  i *= 0x6935fa69;
		i ^= (i & w) >> 11; i *= 0x74dcb303;
		i ^= (i & w) >> 2;  i *= 0x9e501cc3;
		i ^= (i & w) >> 2;  i *= 0xc860a3df;
		i &= w;             i ^= i >> 5;
	} while (i >= l);
	return (i + p) % l;
}

// IsPowerOf4 / Log4Int / RoundUpPow4 -- pbrt-v4 src/pbrt/util/math.h
inline bool is_power_of4(int v) {
	return v > 0 && (v & (v - 1)) == 0 && (v & 0xAAAAAAAA) == 0;
}

// log base-4 of a power-of-4 integer
inline int log4_int(int v) {
	int n = 0;
	while (v > 1) { v >>= 2; ++n; }
	return n;
}

// Round up to next power of 4
inline int round_up_pow4(int v) {
	// round up to power of 2, then ensure it's also a power of 4
	v--;
	v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
	v++;
	// if log2(v) is odd, double it
	if (v > 1 && (v & (v >> 1))) v <<= 1;
	return v;
}

// Convert fixed-point uint32 pair to float2 (pbrt-v4 GetPMJ02BNSample)
// Uses double precision for better pixel-sorting accuracy (matching pbrt-v4).
inline std::pair<float, float> get_pmj02bn_sample(int setIndex, int sampleIndex) {
	using namespace pmj02_detail;
	using namespace pmj02_detail;  // for constants from pmj02_data.h
	setIndex    %= pmj02_detail::nPMJ02bnSets;
	sampleIndex %= pmj02_detail::nPMJ02bnSamples;
	// Double precision scaling matches pbrt-v4's non-GPU path
	double x = pmj02_detail::pmj02bnSamples[setIndex][sampleIndex][0] * 0x1p-32;
	double y = pmj02_detail::pmj02bnSamples[setIndex][sampleIndex][1] * 0x1p-32;
	return { (float)x, (float)y };
}

} // namespace pmj02_detail

// ---------------------------------------------------------------------------
// PMJ02BNSampler
// ---------------------------------------------------------------------------
class PMJ02BNSampler {
public:
	// spp should be a power of 4 (1, 4, 16, 64, 256, ...) for best quality.
	// seed shifts the permutation per-scene to break inter-frame correlation.
	explicit PMJ02BNSampler(int spp = 16, int seed = 0)
		: spp_(spp), seed_(seed) {
		using namespace pmj02_detail;
		// Clamp spp to table limit
		if (spp_ > nPMJ02bnSamples) spp_ = nPMJ02bnSamples;

		// Compute tile size: how many pixel positions fit in the 65536-sample table
		// pixelTileSize = 4^(log4(65536) - log4(roundUpPow4(spp)))
		int spp4    = round_up_pow4(spp_);
		pixel_tile_ = 1 << (log4_int(nPMJ02bnSamples) - log4_int(spp4));

		// Pre-sort PMJ02BN set-0 samples into per-pixel bins
		// pixelSamples_[pixelOffset * spp + k] = k-th sample for that pixel
		int n_pixels  = pixel_tile_ * pixel_tile_;
		pixel_samples_.assign(n_pixels * spp_, {0.0f, 0.0f});

		std::vector<int> n_stored(n_pixels, 0);
		for (int i = 0; i < nPMJ02bnSamples; ++i) {
			auto [fx, fy] = get_pmj02bn_sample(0, i);
			// Scale to pixel tile space
			float px = fx * pixel_tile_;
			float py = fy * pixel_tile_;
			int pixel_offset = (int)px + (int)py * pixel_tile_;
			if (n_stored[pixel_offset] == spp_) continue;  // bucket full (non-pow4 spp)
			int slot = pixel_offset * spp_ + n_stored[pixel_offset];
			// Store fractional part within the pixel tile cell
			pixel_samples_[slot] = { px - (int)px, py - (int)py };
			++n_stored[pixel_offset];
		}
	}

	// Call once per sample before get_1d() / get_2d() / get_pixel_2d().
	void start_pixel_sample(int px, int py, int sample_index) {
		px_        = px;
		py_        = py;
		sample_    = sample_index;
		dimension_ = 0;
	}

	// 1-D sample: PermutationElement-scrambled index + blue-noise delta.
	// Exact port of pbrt-v4 PMJ02BNSampler::Get1D().
	float get_1d() {
		int dim = dimension_++;
		uint64_t h = pmj02_detail::pmj_hash(px_, py_, dim, seed_);
		int index = (int)pmj02_detail::permutation_element(
			(uint32_t)sample_, (uint32_t)spp_, (uint32_t)h);
		float delta = blue_noise(dim, px_, py_);
		return std::min((index + delta) / spp_, pmj02_detail::OneMinusEpsilonF);
	}

	// Pixel 2-D sample: from pre-sorted per-pixel PMJ02BN slice.
	// Exact port of pbrt-v4 PMJ02BNSampler::GetPixel2D().
	std::pair<float, float> get_pixel_2d() {
		int px = px_ % pixel_tile_;
		int py = py_ % pixel_tile_;
		int offset = (px + py * pixel_tile_) * spp_ + sample_;
		auto [u, v] = pixel_samples_[offset];
		return { std::min(u, pmj02_detail::OneMinusEpsilonF),
				 std::min(v, pmj02_detail::OneMinusEpsilonF) };
	}

	// 2-D sample: PMJ02BN table lookup + blue-noise CP rotation.
	// Exact port of pbrt-v4 PMJ02BNSampler::Get2D().
	std::pair<float, float> get_2d() {
		using namespace pmj02_detail;
		int dim = dimension_;
		dimension_ += 2;

		int pmj_instance = dim / 2;
		int index        = sample_;

		if (pmj_instance >= nPMJ02bnSets) {
			// Fall back to permuted index for extra dimensions
			uint64_t h = pmj_hash(px_, py_, dim, seed_);
			index = (int)permutation_element((uint32_t)sample_, (uint32_t)spp_, (uint32_t)h);
		}

		auto [u, v] = get_pmj02bn_sample(pmj_instance, index);

		// Blue-noise Cranley-Patterson rotation (pbrt-v4 exact)
		u += blue_noise(dim,     px_, py_);
		v += blue_noise(dim + 1, px_, py_);
		if (u >= 1.0f) u -= 1.0f;
		if (v >= 1.0f) v -= 1.0f;

		return { std::min(u, OneMinusEpsilonF), std::min(v, OneMinusEpsilonF) };
	}

	int samples_per_pixel() const { return spp_; }
	bool is_power_of4_spp()  const { return pmj02_detail::is_power_of4(spp_); }

private:
	int spp_;
	int seed_;
	int pixel_tile_;    // pixelTileSize from pbrt-v4
	int px_ = 0, py_ = 0, sample_ = 0, dimension_ = 0;
	std::vector<std::pair<float, float>> pixel_samples_;  // [pixel_tile_^2 * spp_]
};
