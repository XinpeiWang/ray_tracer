#ifndef LOWDISCREPANCY_H
#define LOWDISCREPANCY_H
// ===========================================================================
// lowdiscrepancy.h -- Low-discrepancy sequence utilities
//
// Port of the self-contained, allocator-free subset of:
//   pbrt-v4  src/pbrt/util/lowdiscrepancy.h  (Apache-2.0)
//
// Provides:
//   Scrambler types (callable uint32_t -> uint32_t):
//     NoRandomizer         -- identity; no scrambling
//     BinaryPermuteScrambler -- XOR with a fixed permutation word
//     FastOwenScrambler    -- fast hash-based Owen scrambling (~5x faster)
//     OwenScrambler        -- full binary-tree Owen scrambling
//
//   RandomizeStrategy      -- enum matching pbrt-v4's four strategies
//
//   Free functions:
//     MultiplyGenerator    -- dot product of C matrix row with sample index
//     SobolSample<R>       -- generic Sobol sample with pluggable scrambler
//     InverseRadicalInverse-- reverse radical-inverse digit order
//     OwenScrambledRadicalInverse -- Owen-scrambled Halton for arbitrary base
//
// NOT included here (already in other headers):
//   RadicalInverse / ScrambledRadicalInverse  -> halton_sampler.h
//   BlueNoiseSample                           -> bluenoise_sampler.h
//   SobolIntervalToIndex                      -> requires VdCSobolMatrices
//                                                data table (not yet ported)
//
// Dependencies:
//   scalar_math.h   -- ReverseBits32, PermutationElement, FloatOneMinusEpsilon
//   pbrt_hash.h     -- MixBits
//
// References:
//   pbrt-v4 src/pbrt/util/lowdiscrepancy.h  (Apache-2.0)
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU inline
#   endif
#endif

#ifdef _MSC_VER
#   pragma warning(push)
#   pragma warning(disable: 4141) // 'inline' used more than once (CPU_GPU inline)
#endif

#include "float_bits.h"
#include "scalar_math.h"
#include "pbrt_hash.h"

#include <cstdint>

// ===========================================================================
// ReverseBits32 / ReverseBits64 / LeftShift2 / EncodeMorton2
//
// Bit-manipulation utilities required by the scrambler types and Morton
// index helpers.  These are also added to scalar_math.h; the guards here
// prevent double-definition when both headers are included.
// Matches pbrt-v4 ReverseBits32/64, LeftShift2, EncodeMorton2 (util/math.h).
// ===========================================================================
#ifndef SCALAR_MATH_REVERSE_BITS_DEFINED
#define SCALAR_MATH_REVERSE_BITS_DEFINED

CPU_GPU uint32_t ReverseBits32(uint32_t n) {
#if defined(__CUDACC__)
    return __brev(n);
#else
    n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff00ff) << 8)  | ((n & 0xff00ff00) >> 8);
    n = ((n & 0x0f0f0f0f) << 4)  | ((n & 0xf0f0f0f0) >> 4);
    n = ((n & 0x33333333) << 2)  | ((n & 0xcccccccc) >> 2);
    n = ((n & 0x55555555) << 1)  | ((n & 0xaaaaaaaa) >> 1);
    return n;
#endif
}

CPU_GPU uint64_t ReverseBits64(uint64_t n) {
    uint64_t n0 = ReverseBits32(static_cast<uint32_t>(n));
    uint64_t n1 = ReverseBits32(static_cast<uint32_t>(n >> 32));
    return (n0 << 32) | n1;
}

CPU_GPU uint64_t LeftShift2(uint64_t x) {
    x &= 0xffffffff;
    x = (x ^ (x << 16)) & 0x0000ffff0000ffff;
    x = (x ^ (x <<  8)) & 0x00ff00ff00ff00ff;
    x = (x ^ (x <<  4)) & 0x0f0f0f0f0f0f0f0f;
    x = (x ^ (x <<  2)) & 0x3333333333333333;
    x = (x ^ (x <<  1)) & 0x5555555555555555;
    return x;
}

CPU_GPU uint64_t EncodeMorton2(uint32_t x, uint32_t y) {
    return (LeftShift2(y) << 1) | LeftShift2(x);
}

#endif // SCALAR_MATH_REVERSE_BITS_DEFINED

// ===========================================================================
// NoRandomizer
//
// Identity scrambler -- passes the uint32 through unchanged.
// pbrt-v4: struct NoRandomizer (util/lowdiscrepancy.h)
// ===========================================================================
struct NoRandomizer {
	CPU_GPU uint32_t operator()(uint32_t v) const { return v; }
};

// ===========================================================================
// BinaryPermuteScrambler
//
// XOR-based scrambler: XORs the sample with a fixed permutation word.
// Preserves the (0,s)-sequence property.
// pbrt-v4: struct BinaryPermuteScrambler (util/lowdiscrepancy.h)
// ===========================================================================
struct BinaryPermuteScrambler {
	CPU_GPU explicit BinaryPermuteScrambler(uint32_t perm) : permutation(perm) {}
	CPU_GPU uint32_t operator()(uint32_t v) const { return permutation ^ v; }
	uint32_t permutation;
};

// ===========================================================================
// FastOwenScrambler
//
// Hash-based approximation of Owen scrambling.  About 5x faster than the
// full binary-tree version while still decorrelating pixels.
// pbrt-v4: struct FastOwenScrambler (util/lowdiscrepancy.h)
// ===========================================================================
struct FastOwenScrambler {
	CPU_GPU explicit FastOwenScrambler(uint32_t seed) : seed(seed) {}

	CPU_GPU uint32_t operator()(uint32_t v) const {
		v  = ReverseBits32(v);
		v ^= v * 0x3d20adea;
		v += seed;
		v *= (seed >> 16) | 1;
		v ^= v * 0x05526c56;
		v ^= v * 0x53a22864;
		return ReverseBits32(v);
	}

	uint32_t seed;
};

// ===========================================================================
// OwenScrambler
//
// Full binary-tree Owen scrambling.  Stronger statistical guarantees than
// FastOwenScrambler; slower due to the per-bit MixBits call.
// pbrt-v4: struct OwenScrambler (util/lowdiscrepancy.h)
// ===========================================================================
struct OwenScrambler {
	CPU_GPU explicit OwenScrambler(uint32_t seed) : seed(seed) {}

	CPU_GPU uint32_t operator()(uint32_t v) const {
		if (seed & 1)
			v ^= 1u << 31;
		for (int b = 1; b < 32; ++b) {
			// Apply Owen scrambling to binary digit b in v
			uint32_t mask = (~0u) << (32 - b);
			if (static_cast<uint32_t>(MixBits((v & mask) ^ seed)) & (1u << b))
				v ^= 1u << (31 - b);
		}
		return v;
	}

	uint32_t seed;
};

// ===========================================================================
// RandomizeStrategy
//
// Selects which scrambling strategy the sampler uses.
// pbrt-v4: enum class RandomizeStrategy (util/lowdiscrepancy.h)
// ===========================================================================
enum class RandomizeStrategy {
	None,           // NoRandomizer
	PermuteDigits,  // DigitPermutation (Halton)
	FastOwen,       // FastOwenScrambler (Sobol)
	Owen            // OwenScrambler (Sobol, higher quality)
};

// ===========================================================================
// MultiplyGenerator
//
// Multiplies a generator-matrix row C by sample index a using XOR:
//   v = XOR over set bits i of a: C[i]
// This is the standard Sobol/NiederreiterXing matrix-vector product in GF(2).
// pbrt-v4: MultiplyGenerator (util/lowdiscrepancy.h)
// ===========================================================================
CPU_GPU uint32_t MultiplyGenerator(const uint32_t* C, int matrix_size,
										  uint32_t a) {
	uint32_t v = 0;
	for (int i = 0; i < matrix_size && a != 0; ++i, a >>= 1)
		if (a & 1)
			v ^= C[i];
	return v;
}

// ===========================================================================
// SobolSample<R>
//
// Generic Sobol sample: reads the dimension's generator matrix row, computes
// the matrix-vector product, then randomizes via the callable R.
//
// Template parameter R must satisfy: uint32_t R::operator()(uint32_t) const
// e.g. NoRandomizer, BinaryPermuteScrambler, FastOwenScrambler, OwenScrambler
//
// Parameters:
//   matrices     -- flat array: matrices[dim * matrix_size + i] = C[dim][i]
//   matrix_size  -- number of entries per dimension (e.g. 32)
//   a            -- sample index (must be < 2^matrix_size)
//   dimension    -- which Sobol dimension to sample
//   randomizer   -- scrambler instance (called on the raw uint32 result)
//
// Returns a float in [0, 1).
// pbrt-v4: SobolSample<R> (util/lowdiscrepancy.h)
// ===========================================================================
template <typename R>
CPU_GPU float SobolSample(const uint32_t* matrices, int matrix_size,
								 int64_t a, int dimension, R randomizer) {
	const uint32_t* C = matrices + dimension * matrix_size;
	uint32_t v = 0;
	for (int i = 0; i < matrix_size && a != 0; a >>= 1, ++i)
		if (a & 1)
			v ^= C[i];
	v = randomizer(v);
	return std::min(v * 0x1p-32f, FloatOneMinusEpsilon);
}

// ===========================================================================
// InverseRadicalInverse
//
// Reverses the digit order of a radical-inverse value, recovering the
// original sample index from its radical inverse in a given base with a
// fixed number of digits.
// pbrt-v4: InverseRadicalInverse (util/lowdiscrepancy.h)
// ===========================================================================
CPU_GPU uint64_t InverseRadicalInverse(uint64_t inverse, int base,
											  int nDigits) {
	uint64_t index = 0;
	for (int i = 0; i < nDigits; ++i) {
		uint64_t digit = inverse % base;
		inverse /= base;
		index = index * base + digit;
	}
	return index;
}

// ===========================================================================
// OwenScrambledRadicalInverse
//
// Computes the radical inverse of a in the prime base indexed by baseIndex,
// with Owen scrambling applied per digit using a per-digit hash seeded from
// `hash`.  Higher quality than digit-permutation scrambling.
//
// Requires:
//   PermutationElement (scalar_math.h)
//   MixBits            (pbrt_hash.h)
//   get_primes()       -- accessible from halton_sampler.h or locally below
//
// pbrt-v4: OwenScrambledRadicalInverse (util/lowdiscrepancy.h)
// ===========================================================================

// Local minimal prime table for OwenScrambledRadicalInverse.
// Only the first 1024 primes are needed; this avoids a dependency on
// halton_sampler.h's halton_detail::get_primes().
namespace lowdisc_detail {

inline const int* GetPrimes() {
	// First 1024 primes (generated once via sieve).
	static int primes[1024];
	static bool built = false;
	if (!built) {
		built = true;
		constexpr int limit = 8200; // upper bound for 1024th prime
		static bool sieve[limit];
		for (int i = 2; i < limit; ++i) sieve[i] = true;
		for (int i = 2; i * i < limit; ++i)
			if (sieve[i])
				for (int j = i * i; j < limit; j += i) sieve[j] = false;
		int count = 0;
		for (int n = 2; n < limit && count < 1024; ++n)
			if (sieve[n]) primes[count++] = n;
	}
	return primes;
}

} // namespace lowdisc_detail

CPU_GPU float OwenScrambledRadicalInverse(int baseIndex, uint64_t a,
												 uint32_t hash) {
	const unsigned int base = static_cast<unsigned int>(
		lowdisc_detail::GetPrimes()[baseIndex]);
	const float invBase  = 1.0f / static_cast<float>(base);
	float       invBaseM = 1.0f;
	uint64_t limit = ~0ull / base - base;

	uint64_t reversedDigits = 0;
	int digitIndex = 0;
	while (1.f - invBaseM < 1.f && reversedDigits < limit) {
		uint64_t next       = a / base;
		int      digitValue = static_cast<int>(a - next * base);
		uint32_t digitHash  = static_cast<uint32_t>(
			MixBits(hash ^ static_cast<uint64_t>(reversedDigits)));
		digitValue = PermutationElement(static_cast<uint32_t>(digitValue),
										base, digitHash);
		reversedDigits = reversedDigits * base + digitValue;
		invBaseM *= invBase;
		++digitIndex;
		a = next;
	}
	return std::min(invBaseM * static_cast<float>(reversedDigits),
					FloatOneMinusEpsilon);
}

#ifdef _MSC_VER
#   pragma warning(pop)
#endif

#endif // LOWDISCREPANCY_H
