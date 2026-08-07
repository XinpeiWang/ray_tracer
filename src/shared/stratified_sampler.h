#pragma once
// ---------------------------------------------------------------------------
// stratified_sampler.h -- Jittered stratified sampler
//
// Mirrors pbrt-v4 StratifiedSampler (src/pbrt/samplers.h).
//
// Algorithm:
//   For a pixel with nx*ny samples-per-pixel, the sample space [0,1)^2 is
//   subdivided into nx*ny strata (one per sample index).  A hash-scrambled
//   PermutationElement maps sampleIndex -> stratum so that the mapping is
//   random but deterministic per pixel+seed, eliminating inter-dimension
//   correlation.  Within each stratum, a PCG32 RNG provides the jitter.
//
// pbrt-v4 references:
//   src/pbrt/samplers.h          StratifiedSampler
//   src/pbrt/util/rng.h          RNG (PCG32)
//   src/pbrt/util/math.h         MixBits
//   src/shared/pmj02_sampler.h   permutation_element / pmj_hash
//
// Design rules (same as pmj02_sampler.h):
//   - Header-only, CPU_GPU tagged, no heap allocation
//   - Double precision on CPU, float on GPU via conditional compilation
//   - PermutationElement reused from pmj02_detail namespace
// ---------------------------------------------------------------------------

#include <cstdint>
#include <cmath>
#include "pmj02_sampler.h"   // for pmj02_detail::permutation_element / pmj_hash

#include "cpu_gpu.h"

namespace stratified_detail {

// ---------------------------------------------------------------------------
// PCG32 RNG -- exact port of pbrt-v4 RNG (util/rng.h)
// ---------------------------------------------------------------------------
struct PCG32 {
	uint64_t state = 0;
	uint64_t inc   = 1;

	static constexpr uint64_t PCG32_MULT = 6364136223846793005ULL;

	CPU_GPU void set_sequence(uint64_t seq_index, uint64_t seed) {
		state = 0u;
		inc   = (seq_index << 1u) | 1u;
		uniform_u32();
		state += seed;
		uniform_u32();
	}

	// Advance state by idelta steps (pbrt-v4 RNG::Advance)
	CPU_GPU void advance(int64_t idelta) {
		uint64_t cur_mult = PCG32_MULT, cur_plus = inc;
		uint64_t acc_mult = 1u, acc_plus = 0u;
		uint64_t delta = (uint64_t)idelta;
		while (delta > 0) {
			if (delta & 1) { acc_mult *= cur_mult; acc_plus = acc_plus * cur_mult + cur_plus; }
			cur_plus = (cur_mult + 1) * cur_plus;
			cur_mult *= cur_mult;
			delta >>= 1;
		}
		state = acc_mult * state + acc_plus;
	}

	CPU_GPU uint32_t uniform_u32() {
		uint64_t old = state;
		state = old * PCG32_MULT + inc;
		uint32_t xs  = (uint32_t)(((old >> 18u) ^ old) >> 27u);
		uint32_t rot = (uint32_t)(old >> 59u);
		return (xs >> rot) | (xs << ((~rot + 1u) & 31u));
	}

	// Uniform double in [0, 1) -- mirrors pbrt-v4 RNG::Uniform<float>()
	CPU_GPU double uniform_double() {
		uint64_t v0 = uniform_u32(), v1 = uniform_u32();
		uint64_t u  = (v0 << 32) | v1;
		double   d  = (double)u * 0x1p-64;
		// Clamp to < 1 (OneMinusEpsilon equivalent)
		return d < 1.0 ? d : (1.0 - 1.1102230246251565e-16);
	}
};

// MixBits -- pbrt-v4 util/math.h MixBits (finalizer for Murmur-style hash)
CPU_GPU uint64_t mix_bits(uint64_t v) {
	v ^= (v >> 31);
	v *= 0x7fb5d329728ea185ULL;
	v ^= (v >> 27);
	v *= 0x81dadef4bc2dd44dULL;
	v ^= (v >> 33);
	return v;
}

// Multi-arg hash -- mirrors pbrt-v4 Hash(...)
CPU_GPU uint64_t hash2(uint64_t a, uint64_t b) {
	return pmj02_detail::pmj_hash(a, b);
}
CPU_GPU uint64_t hash3(uint64_t a, uint64_t b, uint64_t c) {
	return pmj02_detail::pmj_hash(a, b, c);
}

} // namespace stratified_detail

// ---------------------------------------------------------------------------
// StratifiedSampler
//
// Usage:
//   StratifiedSampler s(4, 4, /*jitter=*/true, /*seed=*/0);
//   s.start_pixel_sample(px, py, sample_index, /*dim=*/0);
//   double u  = s.get_1d();
//   double ux = s.get_2d_x();   // call get_2d_x then get_2d_y in order
//   double uy = s.get_2d_y();
//   double px = s.get_pixel_2d_x();
//   double py = s.get_pixel_2d_y();
// ---------------------------------------------------------------------------
class StratifiedSampler {
public:
	// nx_samples * ny_samples = samples per pixel (total strata)
	// jitter: if false, each sample is at the stratum center (0.5)
	// seed:   per-render seed for decorrelation across frames
	CPU_GPU StratifiedSampler(int nx_samples = 4, int ny_samples = 4,
							   bool jitter = true, int seed = 0)
		: nx_(nx_samples), ny_(ny_samples), seed_(seed), jitter_(jitter) {}

	CPU_GPU int samples_per_pixel() const { return nx_ * ny_; }

	// Call once per sample.  dim=0 resets dimension counter to dim.
	CPU_GPU void start_pixel_sample(int px, int py, int sample_index, int dim = 0) {
		px_     = px;
		py_     = py;
		idx_    = sample_index;
		dim_    = dim;
		// Seed the RNG: one unique sequence per pixel (pbrt-v4 Hash(p, seed))
		uint64_t seq = stratified_detail::hash2((uint64_t)px ^ ((uint64_t)py << 32),
												(uint64_t)seed_);
		rng_.set_sequence(seq, stratified_detail::mix_bits(seq));
		// Advance past the dimensions consumed by earlier samples in this pixel
		// pbrt-v4: rng.Advance(sampleIndex * 65536ull + dimension)
		rng_.advance((int64_t)sample_index * 65536LL + dim_);
	}

	// Get1D: returns stratified sample in [0,1) for the current dimension.
	// Mirrors pbrt-v4 StratifiedSampler::Get1D().
	CPU_GPU double get_1d() {
		uint64_t hash = stratified_detail::hash3((uint64_t)px_ ^ ((uint64_t)py_ << 32),
												  (uint64_t)dim_, (uint64_t)seed_);
		int stratum = (int)pmj02_detail::permutation_element(
			(uint32_t)idx_, (uint32_t)samples_per_pixel(), (uint32_t)hash);
		++dim_;
		double delta = jitter_ ? rng_.uniform_double() : 0.5;
		return (stratum + delta) / samples_per_pixel();
	}

	// Get2D: returns stratified (x,y) pair in [0,1)^2.
	// Mirrors pbrt-v4 StratifiedSampler::Get2D().
	// Returns both components; call get_2d() to get both at once.
	struct Sample2D { double x, y; };
	CPU_GPU Sample2D get_2d() {
		uint64_t hash = stratified_detail::hash3((uint64_t)px_ ^ ((uint64_t)py_ << 32),
												  (uint64_t)dim_, (uint64_t)seed_);
		int stratum = (int)pmj02_detail::permutation_element(
			(uint32_t)idx_, (uint32_t)samples_per_pixel(), (uint32_t)hash);
		dim_ += 2;
		int sx = stratum % nx_, sy = stratum / nx_;
		double dx = jitter_ ? rng_.uniform_double() : 0.5;
		double dy = jitter_ ? rng_.uniform_double() : 0.5;
		return { (sx + dx) / nx_, (sy + dy) / ny_ };
	}

	// get_pixel_2d: same as get_2d in pbrt-v4 StratifiedSampler::GetPixel2D()
	CPU_GPU Sample2D get_pixel_2d() { return get_2d(); }

	// Convenience: current dimension index
	CPU_GPU int dimension() const { return dim_; }

private:
	int  nx_, ny_, seed_;
	bool jitter_;

	int  px_ = 0, py_ = 0, idx_ = 0, dim_ = 0;
	stratified_detail::PCG32 rng_;
};
