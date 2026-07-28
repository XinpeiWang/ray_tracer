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
	0x58000000, 0x94000000, 0x3e000000, 0xe3000000,
	0xbe800000, 0x23c00000, 0x1e200000, 0xf3100000,
	0x46780000, 0x67840000, 0x78460000, 0x84670000,
	0xc6788000, 0xa784c000, 0xd846a000, 0x5467d000,
	0x9e78d800, 0x33845400, 0xe6469e00, 0xb7673300,
	0x20f86680, 0x104477c0, 0xf8668020, 0x4477c010,
	0x668020f8, 0x77c01044, 0x8020f866, 0xc0104477,

	// --- dim 7 ---
	0x80000000, 0x40000000, 0xa0000000, 0x50000000,
	0x88000000, 0x24000000, 0x12000000, 0x2d000000,
	0x76800000, 0x9e400000, 0x08200000, 0x64100000,
	0xb2280000, 0x7d140000, 0xfea20000, 0xba490000,
	0x1a248000, 0x491b4000, 0xc4b5a000, 0xe3739000,
	0xf6800800, 0xde400400, 0xa8200a00, 0x34100500,
	0x3a280880, 0x59140240, 0xeca20120, 0x974902d0,
	0x6ca48768, 0xd75b49e4, 0xcc95a082, 0x87639641,
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

// ---------------------------------------------------------------------------
// ZSobolSampler  -- pbrt-v4 ZSobolSampler (samplers.h)
//
// The default production sampler in pbrt-v4.  Interleaves pixel coordinates
// via a Morton Z-curve so that nearby pixels draw from nearby Sobol strata,
// achieving blue-noise error distribution across the image plane.
//
// Key design:
//   - pixel (px,py) + sample index -> mortonIndex via EncodeMorton2
//   - mortonIndex is treated as a base-4 number of nBase4Digits digits
//   - each digit is randomly permuted using one of 24 precomputed 4-way
//     permutations selected deterministically from higherDigits ^ dimension
//   - The permuted index is passed to SobolSample with FastOwenScrambler
//
// Reference: pbrt-v4 samplers.h ZSobolSampler (Apache-2.0)
// ---------------------------------------------------------------------------

// Bit helpers required by ZSobolSampler
inline uint64_t left_shift2(uint64_t x) {
	x &= 0xffffffff;
	x = (x ^ (x << 16)) & 0x0000ffff0000ffff;
	x = (x ^ (x <<  8)) & 0x00ff00ff00ff00ff;
	x = (x ^ (x <<  4)) & 0x0f0f0f0f0f0f0f0f;
	x = (x ^ (x <<  2)) & 0x3333333333333333;
	x = (x ^ (x <<  1)) & 0x5555555555555555;
	return x;
}

// Interleave bits of x and y into a 64-bit Morton code
inline uint64_t encode_morton2(uint32_t x, uint32_t y) {
	return (left_shift2(y) << 1) | left_shift2(x);
}

// Integer log2 (floor) for uint32
inline int log2_int(uint32_t v) {
	if (v == 0) return 0;
	int r = 0;
	while (v >>= 1) ++r;
	return r;
}

// Round up to next power of 2 (int32)
inline int round_up_pow2(int v) {
	--v;
	v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
	return v + 1;
}

// ZSobolSampler: pixel-aware Sobol sampler with Z-curve Morton ordering.
//
// Usage:
//   ZSobolSampler s(spp, res_x, res_y, /*seed=*/0);
//   s.start_pixel_sample(px, py, sample_index, /*dim=*/0);
//   double v = s.get();   // next sample dimension
class ZSobolSampler {
  public:
	// spp           -- samples per pixel (ideally a power of 2)
	// full_res_x/y  -- full image resolution (determines nBase4Digits)
	// seed          -- global seed for scrambling
	ZSobolSampler(int spp, int full_res_x, int full_res_y, int seed = 0)
		: seed_(seed)
	{
		log2_spp_ = log2_int(static_cast<uint32_t>(spp > 0 ? spp : 1));
		int res = round_up_pow2(full_res_x > full_res_y ? full_res_x : full_res_y);
		int log4_spp = (log2_spp_ + 1) / 2;
		n_base4_digits_ = log2_int(static_cast<uint32_t>(res > 0 ? res : 1)) + log4_spp;
	}

	int samples_per_pixel() const { return 1 << log2_spp_; }

	// Call once per (pixel, sample) pair before fetching dimensions.
	void start_pixel_sample(int px, int py, int sample_idx, int start_dim = 0) {
		dimension_    = start_dim;
		morton_index_ = (encode_morton2(static_cast<uint32_t>(px),
										static_cast<uint32_t>(py))
						 << log2_spp_) | static_cast<uint64_t>(sample_idx);
	}

	// Fetch next 1D sample; advances the dimension counter.
	double get() {
		uint64_t si = get_sample_index();
		uint32_t hash = static_cast<uint32_t>(mix_bits(
			static_cast<uint64_t>(dimension_) ^ static_cast<uint64_t>(seed_) * 6364136223846793005ULL));
		++dimension_;
		uint32_t raw = sobol_sample_raw(si, 0);
		return (fast_owen_scramble(raw, hash) >> 8) * 0x1p-24;
	}

	// Fetch a correlated 2D sample pair.
	void get2d(double& u0, double& u1) {
		uint64_t si   = get_sample_index();
		uint64_t bits = mix_bits(static_cast<uint64_t>(dimension_)
								 ^ static_cast<uint64_t>(seed_) * 6364136223846793005ULL);
		uint32_t h0   = static_cast<uint32_t>(bits);
		uint32_t h1   = static_cast<uint32_t>(bits >> 32);
		dimension_ += 2;
		u0 = (fast_owen_scramble(sobol_sample_raw(si, 0), h0) >> 8) * 0x1p-24;
		u1 = (fast_owen_scramble(sobol_sample_raw(si, 1), h1) >> 8) * 0x1p-24;
	}

	void reset_dim(int d = 0) { dimension_ = d; }

  private:
	// Permute base-4 digits of morton_index to get the Sobol sample index.
	// Direct port of pbrt-v4 ZSobolSampler::GetSampleIndex().
	uint64_t get_sample_index() const {
		static const uint8_t perms[24][4] = {
			{0,1,2,3},{0,1,3,2},{0,2,1,3},{0,2,3,1},
			{0,3,2,1},{0,3,1,2},{1,0,2,3},{1,0,3,2},
			{1,2,0,3},{1,2,3,0},{1,3,2,0},{1,3,0,2},
			{2,1,0,3},{2,1,3,0},{2,0,1,3},{2,0,3,1},
			{2,3,0,1},{2,3,1,0},{3,1,2,0},{3,1,0,2},
			{3,2,1,0},{3,2,0,1},{3,0,2,1},{3,0,1,2}
		};

		uint64_t sample_index = 0;
		bool pow2_samples = (log2_spp_ & 1) != 0;
		int  last_digit   = pow2_samples ? 1 : 0;

		for (int i = n_base4_digits_ - 1; i >= last_digit; --i) {
			int digit_shift = 2 * i - (pow2_samples ? 1 : 0);
			int digit       = static_cast<int>((morton_index_ >> digit_shift) & 3);
			uint64_t higher = morton_index_ >> (digit_shift + 2);
			int p = static_cast<int>((mix_bits(higher ^ (0x55555555u * static_cast<uint64_t>(dimension_))) >> 24) % 24);
			digit = perms[p][digit];
			sample_index |= static_cast<uint64_t>(digit) << digit_shift;
		}

		if (pow2_samples) {
			int digit = static_cast<int>(morton_index_ & 1);
			sample_index |= static_cast<uint64_t>(
				digit ^ (mix_bits((morton_index_ >> 1) ^ (0x55555555u * static_cast<uint64_t>(dimension_))) & 1));
		}

		return sample_index;
	}

	// Raw (unscrambled) Sobol uint32 for dimension d at sample index idx.
	static uint32_t sobol_sample_raw(uint64_t idx, int dim) {
		uint32_t v = 0;
		for (int i = dim * SOBOL_MATRIX_SIZE; idx; idx >>= 1, ++i)
			if (idx & 1) v ^= sobol_matrices[i];
		return v;
	}

	int      seed_;
	int      log2_spp_;
	int      n_base4_digits_;
	uint64_t morton_index_ = 0;
	int      dimension_    = 0;
};

// ---------------------------------------------------------------------------
// PermutationElement -- pbrt-v4 util/math.h
//
// Maps index i to a pseudo-random permutation of {0, ..., l-1} seeded by p.
// Uses a cycle-walking algorithm with a hash cascade: the inner loop maps
// i into a bitmask-sized range (next power of 2 minus 1), then rejects
// values >= l and repeats -- expected 2 iterations for random l.
//
// Usage: int j = permutation_element(i, n, seed_hash);
// Reference: pbrt-v4 util/math.h PermutationElement
// ---------------------------------------------------------------------------
inline int permutation_element(uint32_t i, uint32_t l, uint32_t p) {
	uint32_t w = l - 1;
	w |= w >> 1; w |= w >> 2; w |= w >> 4; w |= w >> 8; w |= w >> 16;
	do {
		i ^= p;           i *= 0xe170893d;
		i ^= p >> 16;
		i ^= (i & w) >> 4;
		i ^= p >> 8;      i *= 0x0929eb3f;
		i ^= p >> 23;
		i ^= (i & w) >> 1;
		i *= 1 | p >> 27; i *= 0x6935fa69;
		i ^= (i & w) >> 11;
		i *= 0x74dcb303;
		i ^= (i & w) >> 2;
		i *= 0x9e501cc3;
		i ^= (i & w) >> 2;
		i *= 0xc860a3df;
		i &= w;
		i ^= i >> 5;
	} while (i >= l);
	return static_cast<int>((i + p) % l);
}

// ---------------------------------------------------------------------------
// PaddedSobolSampler -- pbrt-v4 samplers.h PaddedSobolSampler
//
// Per-pixel padded Sobol sampler: decorrelates pixels by shuffling the
// sample index using PermutationElement (a hash-based permutation seeded
// by pixel coordinates, dimension, and a global seed). Each dimension then
// draws from Sobol dim-0 or dim-1 with FastOwen scrambling.
//
// Design:
//   - start_pixel_sample(px, py, sample_idx, start_dim)
//   - get()   -> next 1D sample (Sobol dim 0, permuted index, Owen scramble)
//   - get2d() -> next 2D sample (Sobol dims 0+1, same permuted index)
//
// Reference: pbrt-v4 samplers.h PaddedSobolSampler (Apache-2.0)
// ---------------------------------------------------------------------------
class PaddedSobolSampler {
  public:
	PaddedSobolSampler(int spp, int seed = 0)
		: spp_(spp > 0 ? spp : 1), seed_(seed) {}

	int samples_per_pixel() const { return spp_; }

	void start_pixel_sample(int px, int py, int sample_idx, int start_dim = 0) {
		pixel_x_     = px;
		pixel_y_     = py;
		sample_idx_  = sample_idx;
		dimension_   = start_dim;
	}

	// Fetch next 1D sample using Sobol dim 0 with per-(pixel,dim) permutation.
	double get() {
		uint64_t h   = pixel_hash(dimension_);
		int idx      = permutation_element(static_cast<uint32_t>(sample_idx_),
										   static_cast<uint32_t>(spp_),
										   static_cast<uint32_t>(h));
		uint32_t raw = sobol_sample_raw(static_cast<uint64_t>(idx), 0);
		uint32_t scr = fast_owen_scramble(raw, static_cast<uint32_t>(h >> 32));
		++dimension_;
		return (scr >> 8) * 0x1p-24;
	}

	// Fetch a 2D sample pair using Sobol dims 0+1 with shared permuted index.
	void get2d(double& u0, double& u1) {
		uint64_t h   = pixel_hash(dimension_);
		int idx      = permutation_element(static_cast<uint32_t>(sample_idx_),
										   static_cast<uint32_t>(spp_),
										   static_cast<uint32_t>(h));
		uint32_t r0  = fast_owen_scramble(sobol_sample_raw(static_cast<uint64_t>(idx), 0),
										  static_cast<uint32_t>(h));
		uint32_t r1  = fast_owen_scramble(sobol_sample_raw(static_cast<uint64_t>(idx), 1),
										  static_cast<uint32_t>(h >> 32));
		dimension_ += 2;
		u0 = (r0 >> 8) * 0x1p-24;
		u1 = (r1 >> 8) * 0x1p-24;
	}

	void reset_dim(int d = 0) { dimension_ = d; }

  private:
	// Hash pixel coordinates + dimension + seed to get a per-dimension seed.
	uint64_t pixel_hash(int dim) const {
		return mix_bits(static_cast<uint64_t>(pixel_x_) * 2654435761ull
					  ^ static_cast<uint64_t>(pixel_y_) * 805459861ull
					  ^ static_cast<uint64_t>(dim)      * 2246822519ull
					  ^ static_cast<uint64_t>(seed_)    * 6364136223846793005ull);
	}

	// Raw Sobol uint32 for dimension d at index idx (same as ZSobolSampler).
	static uint32_t sobol_sample_raw(uint64_t idx, int dim) {
		uint32_t v = 0;
		for (int i = dim * SOBOL_MATRIX_SIZE; idx; idx >>= 1, ++i)
			if (idx & 1) v ^= sobol_matrices[i];
		return v;
	}

	int spp_;
	int seed_;
	int pixel_x_    = 0;
	int pixel_y_    = 0;
	int sample_idx_ = 0;
	int dimension_  = 0;
};
