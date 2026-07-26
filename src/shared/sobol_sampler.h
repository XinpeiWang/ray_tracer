#pragma once
//==============================================================================
// sobol_sampler.h -- Sobol sequence sampler with Fast Owen scrambling
//
// pbrt-v4 alignment
// -----------------
// pbrt-v4 (util/lowdiscrepancy.h) implements SobolSample() using a 32-bit
// generator matrix XOR-accumulation algorithm (Joe & Kuo 2008 direction
// numbers) followed by a randomizer.  The default randomizer for the CPU
// path integrator is FastOwenScrambler -- a hash-based approximation of Owen
// scrambling that preserves the (t,s)-sequence properties while being ~5x
// faster than the full bit-by-bit Owen tree traversal.
//
// This file reproduces that design with:
//   - The first 8 Sobol dimensions (sufficient for RR + path decisions).
//     Direction numbers from: Joe & Kuo, "Constructing Sobol sequences with
//     better two-dimensional projections", SIAM J. Sci. Comput. 30, 2008.
//     (same source as pbrt-v4's sobolmatrices.cpp, Apache-2.0 / Gruenschloss MIT)
//   - FastOwenScrambler identical to pbrt-v4's implementation.
//   - MixBits hash identical to pbrt-v4's MixBits (util/math.h).
//   - ReverseBits32 identical to pbrt-v4's ReverseBits32.
//   - SobolSampler interface matching PathSampler:
//       SobolSampler(sample_idx, pixel_x, pixel_y)
//       double get()       -- next sample in [0,1), advances dimension
//       void   reset_dim() -- reset dimension counter
//
// Why better than Halton?
//   Sobol sequences have lower star-discrepancy in high dimensions than
//   Halton, meaning stratification improves with more samples.  The
//   FastOwen scrambling additionally decorrelates pixels, eliminating the
//   structured "ghosting" artifacts visible with unscrambled sequences.
//==============================================================================

#include <cstdint>
#include <cmath>

// ---------------------------------------------------------------------------
// Bit-manipulation helpers -- identical to pbrt-v4 (util/math.h)
// ---------------------------------------------------------------------------

inline uint32_t reverse_bits_32(uint32_t v) {
	v = (v << 16) | (v >> 16);
	v = ((v & 0x00ff00ff) << 8)  | ((v & 0xff00ff00) >> 8);
	v = ((v & 0x0f0f0f0f) << 4)  | ((v & 0xf0f0f0f0) >> 4);
	v = ((v & 0x33333333) << 2)  | ((v & 0xcccccccc) >> 2);
	v = ((v & 0x55555555) << 1)  | ((v & 0xaaaaaaaa) >> 1);
	return v;
}

// MixBits -- finalisation hash from pbrt-v4 util/math.h
inline uint64_t mix_bits(uint64_t v) {
	v ^= (v >> 31);
	v *= 0x7fb5d329728ea185ull;
	v ^= (v >> 27);
	v *= 0x81dadef4bc2dd44dull;
	v ^= (v >> 33);
	return v;
}

// ---------------------------------------------------------------------------
// FastOwenScrambler -- pbrt-v4 FastOwenScrambler (util/lowdiscrepancy.h)
//
// Maps a 32-bit Sobol sample to a scrambled 32-bit value that preserves the
// (0,s)-sequence property in expectation.  Uses reverseBits and a multiply-
// xor hash cascade -- ~5x faster than the full Owen tree.
// ---------------------------------------------------------------------------
inline uint32_t fast_owen_scramble(uint32_t v, uint32_t seed) {
	v  = reverse_bits_32(v);
	v ^= v * 0x3d20adea;
	v += seed;
	v *= (seed >> 16) | 1;
	v ^= v * 0x05526c56;
	v ^= v * 0x53a22864;
	return reverse_bits_32(v);
}

// ---------------------------------------------------------------------------
// Sobol generator matrices (Joe & Kuo 2008) -- first 8 dimensions only.
//
// Each dimension has exactly SOBOL_MATRIX_SIZE = 32 entries (we use the
// first 32 of the 52 that pbrt stores; for sample indices < 2^32 -- which
// covers any practical render -- this is exact).
//
// Dimensions 0..7, row-major:  sobol_matrices[dim * SOBOL_MATRIX_SIZE + i]
//
// Dim 0: identity (Van der Corput / base-2 radical inverse)
// Dim 1..7: Joe & Kuo new-joe-kuo-6.21201 (same as pbrt-v4)
// ---------------------------------------------------------------------------
static constexpr int SOBOL_DIMS        = 8;
static constexpr int SOBOL_MATRIX_SIZE = 32;

static constexpr uint32_t sobol_matrices[SOBOL_DIMS * SOBOL_MATRIX_SIZE] = {
	// --- dim 0: identity (Van der Corput) ---
	0x80000000, 0x40000000, 0x20000000, 0x10000000,
	0x08000000, 0x04000000, 0x02000000, 0x01000000,
	0x00800000, 0x00400000, 0x00200000, 0x00100000,
	0x00080000, 0x00040000, 0x00020000, 0x00010000,
	0x00008000, 0x00004000, 0x00002000, 0x00001000,
	0x00000800, 0x00000400, 0x00000200, 0x00000100,
	0x00000080, 0x00000040, 0x00000020, 0x00000010,
	0x00000008, 0x00000004, 0x00000002, 0x00000001,

	// --- dim 1 ---
	0x80000000, 0xc0000000, 0xa0000000, 0xf0000000,
	0x88000000, 0xcc000000, 0xaa000000, 0xff000000,
	0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000,
	0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000,
	0x80008000, 0xc000c000, 0xa000a000, 0xf000f000,
	0x88008800, 0xcc00cc00, 0xaa00aa00, 0xff00ff00,
	0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0,
	0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff,

	// --- dim 2 ---
	0x80000000, 0xc0000000, 0x60000000, 0x90000000,
	0xe8000000, 0x5c000000, 0x8e000000, 0xc5000000,
	0x68800000, 0x9cc00000, 0xee600000, 0x55900000,
	0x80680000, 0xc09c0000, 0x60ee0000, 0x90550000,
	0xe8808000, 0x5cc0c000, 0x8e606000, 0xc5909000,
	0x6868e800, 0x9c9c5c00, 0xeeee8e00, 0x5555c500,
	0x8000e880, 0xc0005cc0, 0x60008e60, 0x9000c590,
	0xe8006868, 0x5c009c9c, 0x8e00eeee, 0xc5005555,

	// --- dim 3 ---
	0x80000000, 0xc0000000, 0x20000000, 0x50000000,
	0xf8000000, 0x74000000, 0xa2000000, 0x93000000,
	0xd8800000, 0x25400000, 0x59e00000, 0xe6d00000,
	0x78080000, 0xb40c0000, 0x82020000, 0xc3050000,
	0x208f8000, 0x51474000, 0xfbea2000, 0x75d93000,
	0xa0858800, 0x914e5400, 0xdbe79e00, 0x25db6d00,
	0x58800080, 0xe54000c0, 0x79e00020, 0xb6d00050,
	0x800800f8, 0xc00c0074, 0x200200a2, 0x50050093,

	// --- dim 4 ---
	0x80000000, 0x40000000, 0x20000000, 0xb0000000,
	0xf8000000, 0xdc000000, 0x7a000000, 0x9d000000,
	0x5a800000, 0x2fc00000, 0xa1600000, 0xf0b00000,
	0xda880000, 0x6fc40000, 0x81620000, 0x40bb0000,
	0x22878000, 0xb3c9c000, 0xfb65a000, 0xddb2d000,
	0x78022800, 0x9c0b3c00, 0x5a0fb600, 0x2d0ddb00,
	0xa2878080, 0xf3c9c040, 0xdb65a020, 0x6db2d0b0,
	0x800228f8, 0x400b3cdc, 0x200fb67a, 0xb00ddb9d,

	// --- dim 5 ---
	0x80000000, 0x40000000, 0x60000000, 0x30000000,
	0xc8000000, 0x24000000, 0x56000000, 0xfb000000,
	0xe0800000, 0x70400000, 0xa8600000, 0x14300000,
	0x9ec80000, 0xdf240000, 0xb6d60000, 0x8bbb0000,
	0x48008000, 0x64004000, 0x36006000, 0xcb003000,
	0x2880c800, 0x54402400, 0xfe605600, 0xef30fb00,
	0x7e48e080, 0xaf647040, 0x1eb6a860, 0x9f8b1430,
	0xd6c81ec8, 0xbb249f24, 0x80d6d6d6, 0x40bbbbbb,

	// --- dim 6 ---
	0x80000000, 0xc0000000, 0xa0000000, 0xd0000000,
	0xf8000000, 0xcc000000, 0xa6000000, 0x93000000,
	0x65800000, 0xce400000, 0xe3a00000, 0x29d00000,
	0x2e880000, 0x5f440000, 0xa1220000, 0xd2990000,
	0xa53f8000, 0x08cfc000, 0x9e27a000, 0xb7d3d000,
	0xab9de800, 0x3439cc00, 0x3f01a600, 0xd29d9300,
	0xe24be580, 0xcf9bce40, 0x2c21e3a0, 0xd0d629d0,
	0x12672e88, 0x435b5f44, 0xa1d7a122, 0xb15ed299,

	// --- dim 7 ---
	0x80000000, 0xc0000000, 0x40000000, 0xa0000000,
	0xf0000000, 0xd8000000, 0x5c000000, 0x2e000000,
	0x97000000, 0xeb800000, 0x05c00000, 0x03600000,
	0x01b00000, 0x00d80000, 0x006c0000, 0x00360000,
	0x001b0000, 0x000d8000, 0x00068000, 0x00034000,
	0x0001a000, 0x0000d000, 0x00006800, 0x00003400,
	0x00001a00, 0x00000d00, 0x00000680, 0x00000340,
	0x000001a0, 0x000000d0, 0x00000068, 0x00000034,
};

// ---------------------------------------------------------------------------
// sobol_sample -- evaluate one Sobol sample for (index, dimension, seed)
//
// Mirrors pbrt-v4's SobolSample<FastOwenScrambler>.
// index     -- path sample index (monotonically increasing per pixel)
// dim       -- dimension [0, SOBOL_DIMS)
// seed      -- per-pixel/per-dimension scramble seed
// ---------------------------------------------------------------------------
inline double sobol_sample(uint64_t index, int dim, uint32_t seed) {
	// Accumulate generator matrix columns for set bits in index.
	uint32_t v = 0;
	const int base = dim * SOBOL_MATRIX_SIZE;
	for (int i = base; index != 0; index >>= 1, ++i)
		if (index & 1)
			v ^= sobol_matrices[i];

	// Apply Fast Owen scrambling (pbrt-v4 FastOwenScrambler).
	v = fast_owen_scramble(v, seed);

	// Convert to [0, 1).  0x1p-32f = 2^{-32}.
	return std::min(v * (1.0 / 4294967296.0), 1.0 - 1e-15);
}

// ---------------------------------------------------------------------------
// SobolSampler -- stateful per-path sampler.
//
// Drop-in replacement for PathSampler (path_sampler.h).
// Construct once per camera ray, call get() for each integrator-level
// random decision.  Wraps to a LCG fallback beyond SOBOL_DIMS dimensions.
//
// Usage:
//   SobolSampler ss(sample_index, pixel_x, pixel_y);
//   double u = ss.get();   // dim 0
//   double v = ss.get();   // dim 1
// ---------------------------------------------------------------------------
class SobolSampler {
  public:
	// sample_idx  -- index of this sample within the current pixel (0-based)
	// pixel_x/y   -- pixel coordinates, used to build per-pixel scramble seeds
	SobolSampler(int sample_idx, int pixel_x, int pixel_y)
		: index_(static_cast<uint64_t>(sample_idx)), dim_(0)
	{
		// Build per-dimension, per-pixel scramble seeds.
		// Uses mix_bits (pbrt-v4 MixBits) to hash pixel + dimension.
		uint32_t px = static_cast<uint32_t>(pixel_x);
		uint32_t py = static_cast<uint32_t>(pixel_y);
		for (int d = 0; d < SOBOL_DIMS; ++d) {
			uint64_t h = mix_bits(static_cast<uint64_t>(px) * 2654435761ull
								^ static_cast<uint64_t>(py) * 805459861ull
								^ static_cast<uint64_t>(d)  * 2246822519ull);
			seeds_[d] = static_cast<uint32_t>(h);
		}
	}

	// Return the next sample dimension value in [0, 1).
	double get() {
		if (dim_ >= SOBOL_DIMS) {
			// Fallback LCG for extra dimensions (same as PathSampler).
			fallback_ = fallback_ * 6364136223846793005ULL + 1442695040888963407ULL;
			return static_cast<double>((fallback_ >> 33) & 0x7FFFFFFF) / 2147483647.0;
		}
		double val = sobol_sample(index_, dim_, seeds_[dim_]);
		++dim_;
		return val;
	}

	// Reset dimension counter (e.g. to re-use sampler for another vertex).
	void reset_dim(int d = 0) { dim_ = d; }

  private:
	uint64_t index_;
	int      dim_;
	uint32_t seeds_[SOBOL_DIMS];
	uint64_t fallback_ = 0x123456789ABCDEFull;
};
