#ifndef HALTON_SAMPLER_H
#define HALTON_SAMPLER_H
//==============================================================================
// halton_sampler.h -- Low-discrepancy Halton sampler (pbrt-v4 §8.3)
//
// pbrt-v4 reference: src/pbrt/samplers.h  (HaltonSampler)
//                    src/pbrt/util/lowdiscrepancy.h
//                    src/pbrt/util/primes.h / primes.cpp
//
// Algorithm:
//   The Halton sequence generates n-dimensional sample points by computing
//   the radical inverse of the sample index in successive prime bases:
//     sample[i] = (RadicalInverse(base_0, i), RadicalInverse(base_1, i), ...)
//
//   Pixel mapping: each pixel (px,py) maps to a unique stride in the global
//   sequence via the Chinese Remainder Theorem (matching pbrt-v4 §8.3.1).
//   Dimensions 0 and 1 are always base-2 and base-3 (the pixel 2D dims).
//   Dimensions 2+ use Primes[2], Primes[3], ... for additional 1D samples.
//
//   Scrambling (RandomizeStrategy::PermuteDigits):
//   Per-dimension digit permutations are pre-computed from a seed using a
//   simple LCG-based Fisher-Yates shuffle, matching pbrt-v4's intent.
//
// Usage:
//   halton_sampler s(spp, image_width, image_height);
//   for each pixel (px, py), for each sample index i:
//     s.start_pixel_sample(px, py, i);
//     double u = s.get_1d();
//     auto [u1, u2] = s.get_2d();
//==============================================================================

#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cassert>

// ---------------------------------------------------------------------------
// Prime table -- first 1000 primes via Sieve of Eratosthenes
// (pbrt-v4 reference: src/pbrt/util/primes.cpp -- same values)
// Built once on first call; avoids MSVC large-brace-initializer limits.
// ---------------------------------------------------------------------------
namespace halton_detail {

static constexpr int PrimeTableSize = 1000;

inline const int* get_primes() {
	static int primes[PrimeTableSize];
	static bool built = false;
	if (!built) {
		// Sieve of Eratosthenes up to 7920 (7919 is the 1000th prime)
		static const int sieve_limit = 8000;
		static bool composite[sieve_limit] = {};
		int count = 0;
		for (int n = 2; n < sieve_limit && count < PrimeTableSize; ++n) {
			if (!composite[n]) {
				primes[count++] = n;
				for (int m = n * 2; m < sieve_limit; m += n)
					composite[m] = true;
			}
		}
		built = true;
	}
	return primes;
}

// ---------------------------------------------------------------------------
// Helpers: radical inverse and scrambled radical inverse
// Verbatim port of pbrt-v4 RadicalInverse / ScrambledRadicalInverse
// ---------------------------------------------------------------------------

static constexpr double OneMinusEpsilon = 1.0 - 1.1102230246251565e-16;

inline double radical_inverse(int base_index, uint64_t a) {
	unsigned int base = (unsigned int)get_primes()[base_index];
	uint64_t limit = ~0ull / base - base;
	double inv_base = 1.0 / base, inv_base_m = 1.0;
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

inline uint64_t inverse_radical_inverse(uint64_t inv, int base, int n_digits) {
	uint64_t index = 0;
	for (int i = 0; i < n_digits; ++i) {
		uint64_t digit = inv % base;
		inv /= base;
		index = index * base + digit;
	}
	return index;
}

// ---------------------------------------------------------------------------
// Hashing utilities -- verbatim port of pbrt-v4 MixBits, MurmurHash64A, Hash
// (src/pbrt/util/hash.h)
// ---------------------------------------------------------------------------

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
		uint64_t k;
		std::memcpy(&k, key, sizeof(uint64_t));
		key += 8;
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
inline uint64_t halton_hash(Args... args) {
	constexpr size_t sz = (sizeof(Args) + ... + 0);
	constexpr size_t n  = (sz + 7) / 8;
	uint64_t buf[n];
	// pack args into buf
	char* p = reinterpret_cast<char*>(buf);
	auto pack = [&](auto v) { std::memcpy(p, &v, sizeof(v)); p += sizeof(v); };
	(pack(args), ...);
	return murmur_hash64a(reinterpret_cast<const unsigned char*>(buf), sz, 0);
}

// ---------------------------------------------------------------------------
// PermutationElement -- Feistel-cipher bijection over {0..n-1}
// Verbatim port of pbrt-v4 PermutationElement (src/pbrt/util/math.h)
// ---------------------------------------------------------------------------
inline uint32_t permutation_element(uint32_t i, uint32_t l, uint32_t p) {
	uint32_t w = l - 1;
	w |= w >> 1; w |= w >> 2; w |= w >> 4; w |= w >> 8; w |= w >> 16;
	do {
		i ^= p;           i *= 0xe170893d;
		i ^= p >> 16;     i ^= (i & w) >> 4;
		i ^= p >> 8;      i *= 0x0929eb3f;
		i ^= p >> 23;     i ^= (i & w) >> 1;
		i *= 1 | p >> 27; i *= 0x6935fa69;
		i ^= (i & w) >> 11; i *= 0x74dcb303;
		i ^= (i & w) >> 2;  i *= 0x9e501cc3;
		i ^= (i & w) >> 2;  i *= 0xc860a3df;
		i &= w;           i ^= i >> 5;
	} while (i >= l);
	return (i + p) % l;
}

// ---------------------------------------------------------------------------
// DigitPermutation -- per-dimension scrambling table
// Exact port of pbrt-v4 DigitPermutation (src/pbrt/util/lowdiscrepancy.h)
// Uses Hash(base, digitIndex, seed) + PermutationElement for each slot.
// ---------------------------------------------------------------------------
struct DigitPermutation {
	int base     = 0;
	int n_digits = 0;
	std::vector<uint16_t> permutations; // [n_digits * base]

	DigitPermutation() = default;

	DigitPermutation(int b, uint32_t seed) : base(b) {
		// Compute how many base-b digits fit in a double (matching pbrt-v4)
		double inv_base = 1.0 / b, inv_base_m = 1.0;
		n_digits = 0;
		while (1.0 - (b - 1) * inv_base_m < 1.0) {
			++n_digits;
			inv_base_m *= inv_base;
		}
		permutations.resize(n_digits * base);

		// Compute random permutations: verbatim pbrt-v4 logic
		for (int digit_index = 0; digit_index < n_digits; ++digit_index) {
			uint64_t dseed = halton_hash(b, digit_index, seed);
			for (int digit_value = 0; digit_value < base; ++digit_value) {
				int index = digit_index * base + digit_value;
				permutations[index] = (uint16_t)permutation_element(
					(uint32_t)digit_value, (uint32_t)base, (uint32_t)dseed);
			}
		}
	}

	int permute(int digit_index, int digit_value) const {
		return permutations[digit_index * base + digit_value];
	}
};

inline double scrambled_radical_inverse(int base_index, uint64_t a,
										 const DigitPermutation& perm) {
	unsigned int base = (unsigned int)get_primes()[base_index];
	uint64_t limit = ~0ull / base - base;
	double inv_base = 1.0 / base, inv_base_m = 1.0;
	uint64_t reversed = 0;
	int digit_index = 0;
	while (1.0 - (base - 1) * inv_base_m < 1.0 && reversed < limit) {
		uint64_t next       = a / base;
		int digit_value     = (int)(a - next * base);
		reversed = reversed * base + perm.permute(digit_index, digit_value);
		inv_base_m *= inv_base;
		++digit_index;
		a = next;
	}
	return std::min(inv_base_m * reversed, OneMinusEpsilon);
}

} // namespace halton_detail


// ---------------------------------------------------------------------------
// RandomizeStrategy enum  (mirrors pbrt-v4)
// ---------------------------------------------------------------------------
enum class HaltonRandomize {
	None,           // pure Halton — deterministic, no scrambling
	PermuteDigits   // per-dimension digit permutations seeded from a value
};


// ---------------------------------------------------------------------------
// halton_sampler
// ---------------------------------------------------------------------------
class halton_sampler {
public:
	// Max image resolution that gets a perfect per-pixel stride.
	// Matches pbrt-v4's MaxHaltonResolution = 128.
	static constexpr int MaxResolution = 128;

	// -----------------------------------------------------------------------
	// Constructor
	//   samples_per_pixel : spp (any positive integer)
	//   image_width/height: used to compute pixel strides (clamped to 128)
	//   randomize         : None or PermuteDigits
	//   seed              : scrambling seed (ignored for None)
	// -----------------------------------------------------------------------
	halton_sampler(int samples_per_pixel,
				   int image_width, int image_height,
				   HaltonRandomize randomize = HaltonRandomize::PermuteDigits,
				   uint32_t seed = 0)
		: samples_per_pixel_(samples_per_pixel)
		, randomize_(randomize)
	{
		using namespace halton_detail;

		// Compute base scales and exponents for dims 0 (base 2) and 1 (base 3)
		// matching pbrt-v4's pixel-stride computation.
		int res[2] = {
			std::min(image_width,  MaxResolution),
			std::min(image_height, MaxResolution)
		};
		for (int i = 0; i < 2; ++i) {
			int base = (i == 0) ? 2 : 3;
			int scale = 1, exp = 0;
			while (scale < res[i]) { scale *= base; ++exp; }
			base_scales_[i]   = scale;
			base_exponents_[i] = exp;
		}

		// Multiplicative inverses: mult_inverse[i] s.t.
		// mult_inverse[0] * base_scales_[0] ≡ 1 (mod base_scales_[1])
		// mult_inverse[1] * base_scales_[1] ≡ 1 (mod base_scales_[0])
		mult_inverse_[0] = (int)multiplicative_inverse(base_scales_[0], base_scales_[1]);
		mult_inverse_[1] = (int)multiplicative_inverse(base_scales_[1], base_scales_[0]);

		// Store seed; digit permutations are built lazily on first use.
		if (randomize == HaltonRandomize::PermuteDigits) {
			perm_seed_ = seed;
			digit_perms_.resize(PrimeTableSize); // empty DigitPermutations (base==0)
		}
	}

	int samples_per_pixel() const { return samples_per_pixel_; }

	// -----------------------------------------------------------------------
	// start_pixel_sample -- called once per (pixel, sample_index) pair.
	//   px, py        : pixel coordinates
	//   sample_index  : [0, samples_per_pixel)
	//   dim           : starting dimension (default 0)
	// -----------------------------------------------------------------------
	void start_pixel_sample(int px, int py, int sample_index, int dim = 0) {
		using namespace halton_detail;

		halton_index_ = 0;
		int sample_stride = base_scales_[0] * base_scales_[1];

		if (sample_stride > 1) {
			int pmx = px % MaxResolution;
			int pmy = py % MaxResolution;

			// Dim 0 offset (base 2)
			uint64_t dim0_offset = inverse_radical_inverse(pmx, 2, base_exponents_[0]);
			// Dim 1 offset (base 3)
			uint64_t dim1_offset = inverse_radical_inverse(pmy, 3, base_exponents_[1]);

			halton_index_ += dim0_offset * (sample_stride / base_scales_[0]) * mult_inverse_[0];
			halton_index_ += dim1_offset * (sample_stride / base_scales_[1]) * mult_inverse_[1];
			halton_index_ %= sample_stride;
		}

		halton_index_ += (int64_t)sample_index * sample_stride;
		dimension_     = std::max(2, dim);
	}

	// -----------------------------------------------------------------------
	// get_1d -- next 1D sample in [0,1)
	// -----------------------------------------------------------------------
	double get_1d() {
		if (dimension_ >= halton_detail::PrimeTableSize)
			dimension_ = 2;
		return sample_dimension(dimension_++);
	}

	// Single-scalar convenience, matching SobolSampler's own get() - lets
	// camera.h's ray_color() (templated, duck-typed on `Sampler& sampler`
	// calling only sampler.get()) treat this class as a drop-in replacement.
	double get() { return get_1d(); }

	// -----------------------------------------------------------------------
	// get_2d -- next 2D sample in [0,1)^2
	// -----------------------------------------------------------------------
	std::pair<double,double> get_2d() {
		if (dimension_ + 1 >= halton_detail::PrimeTableSize)
			dimension_ = 2;
		int d = dimension_;
		dimension_ += 2;
		return { sample_dimension(d), sample_dimension(d + 1) };
	}

	// -----------------------------------------------------------------------
	// get_pixel_2d -- the canonical pixel sample (dims 0 and 1)
	// This always uses base-2 and base-3 without consuming dimension_ counter.
	// Matches pbrt-v4 HaltonSampler::GetPixel2D().
	// -----------------------------------------------------------------------
	std::pair<double,double> get_pixel_2d() const {
		// Qualified - path_sampler.h (already #include'd by any TU that also
		// wants SamplerKind, e.g. camera.h) declares its own global,
		// same-signature radical_inverse(int, uint64_t), which an unqualified
		// call here would find ambiguous against halton_detail's own once
		// both headers are included together.
		return {
			halton_detail::radical_inverse(0, halton_index_ >> base_exponents_[0]),
			halton_detail::radical_inverse(1, halton_index_ / base_scales_[1])
		};
	}

private:
	const halton_detail::DigitPermutation& get_perm(int dim) const {
		using namespace halton_detail;
		auto& p = digit_perms_[dim];
		if (p.base == 0)  // not yet built
			p = DigitPermutation(get_primes()[dim], perm_seed_);
		return p;
	}

	double sample_dimension(int dim) const {
		using namespace halton_detail;
		if (randomize_ == HaltonRandomize::None)
			// Qualified - see get_pixel_2d()'s own comment on why an
			// unqualified call here is ambiguous once path_sampler.h is
			// also included.
			return halton_detail::radical_inverse(dim, halton_index_);
		return scrambled_radical_inverse(dim, halton_index_, get_perm(dim));
	}

	// Extended Euclidean GCD
	static void extended_gcd(uint64_t a, uint64_t b, int64_t* x, int64_t* y) {
		if (b == 0) { *x = 1; *y = 0; return; }
		int64_t d = (int64_t)(a / b), xp, yp;
		extended_gcd(b, a % b, &xp, &yp);
		*x = yp;
		*y = xp - d * yp;
	}

	static uint64_t multiplicative_inverse(int64_t a, int64_t n) {
		int64_t x, y;
		extended_gcd((uint64_t)a, (uint64_t)n, &x, &y);
		// Ensure positive result
		return (uint64_t)((x % n + n) % n);
	}

	int     samples_per_pixel_;
	HaltonRandomize randomize_;
	int     base_scales_[2]   = {1, 1};
	int     base_exponents_[2] = {0, 0};
	int     mult_inverse_[2]  = {0, 0};
	int64_t halton_index_     = 0;
	int     dimension_        = 0;

	uint32_t perm_seed_ = 0;
	mutable std::vector<halton_detail::DigitPermutation> digit_perms_;
};


#endif // HALTON_SAMPLER_H
