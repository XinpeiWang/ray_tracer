#pragma once
// ---------------------------------------------------------------------------
// reservoir_sampler.h -- WeightedReservoirSampler<T> and AliasTable
//
// Mirrors pbrt-v4 util/sampling.h (WeightedReservoirSampler, AliasTable)
// and util/rng.h (PCG32 RNG) adapted for our double-precision CPU-only use.
//
// Components:
//   PCG32Rng          -- minimal PCG-32 RNG (pbrt-v4 util/rng.h)
//   WeightedReservoirSampler<T> -- streaming weighted reservoir sampling
//                                  (Vitter 1985 / Talbot 2005)
//   AliasTable        -- O(1) discrete sampling via Walker's alias method
//
// Design rules (same as sobol_sampler.h / bssrdf.h):
//   - Plain structs/classes, no virtual functions, no heap-allocated polymorphism
//   - CPU_GPU tagged for portability
//   - No pbrt Allocator: AliasTable uses std::vector
//
// References:
//   pbrt-v4 src/pbrt/util/rng.h    (Apache-2.0)
//   pbrt-v4 src/pbrt/util/sampling.h (Apache-2.0)
//   pbrt-v4 src/pbrt/util/sampling.cpp (Apache-2.0)
//   Vitter, "Random sampling with a reservoir", 1985
//   Walker, "An efficient method for generating discrete random variables
//            with general distributions", 1977
// ---------------------------------------------------------------------------

#include "cpu_gpu.h"

#include <cstdint>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <limits>

// ===========================================================================
// PCG32Rng -- Permuted Congruential Generator (32-bit output)
//
// Mirrors pbrt-v4 RNG (util/rng.h). Used internally by WeightedReservoirSampler.
// ===========================================================================
struct PCG32Rng {
	// PCG-32 constants (pbrt-v4 / O'Neill 2014)
	static constexpr uint64_t kDefaultState  = 0x853c49e6748fea9bull;
	static constexpr uint64_t kDefaultStream = 0xda3e39cb94b95bdbull;
	static constexpr uint64_t kMult          = 0x5851f42d4c957f2dull;

	CPU_GPU PCG32Rng() : state_(kDefaultState), inc_(kDefaultStream) {}
	CPU_GPU explicit PCG32Rng(uint64_t seq_index) { set_sequence(seq_index); }
	CPU_GPU PCG32Rng(uint64_t seq_index, uint64_t seed) { set_sequence(seq_index, seed); }

	// pbrt-v4: RNG::SetSequence(sequenceIndex, seed)
	CPU_GPU void set_sequence(uint64_t seq_index, uint64_t seed) {
		state_ = 0u;
		inc_   = (seq_index << 1u) | 1u;
		uniform_u32();
		state_ += seed;
		uniform_u32();
	}

	// pbrt-v4: RNG::SetSequence(sequenceIndex)  uses MixBits as seed
	CPU_GPU void set_sequence(uint64_t seq_index) {
		// MixBits from pbrt-v4 util/math.h
		uint64_t s = seq_index;
		s ^= (s >> 31); s *= 0x7fb5d329728ea185ull;
		s ^= (s >> 27); s *= 0x81dadef4bc2dd44dull;
		s ^= (s >> 33);
		set_sequence(seq_index, s);
	}

	// Raw 32-bit output (pbrt-v4: RNG::Uniform<uint32_t>)
	CPU_GPU uint32_t uniform_u32() {
		uint64_t old = state_;
		state_ = old * kMult + inc_;
		uint32_t xs  = (uint32_t)(((old >> 18u) ^ old) >> 27u);
		uint32_t rot = (uint32_t)(old >> 59u);
		return (xs >> rot) | (xs << ((~rot + 1u) & 31u));
	}

	// Uniform float in [0, 1)  (pbrt-v4: RNG::Uniform<float>)
	CPU_GPU double uniform() {
		// Use 32-bit to match pbrt-v4's float path (good enough for reservoir)
		static constexpr double kOneMinusEps = 1.0 - std::numeric_limits<double>::epsilon() / 2.0;
		return std::min(kOneMinusEps, uniform_u32() * 0x1p-32);
	}

  private:
	uint64_t state_, inc_;
};

// ===========================================================================
// WeightedReservoirSampler<T>
//
// Streaming reservoir sampler: given a stream of (sample, weight) pairs,
// maintains a single reservoir such that the probability the reservoir holds
// sample s_i is proportional to weight_i / sum_of_all_weights.
//
// Usage:
//   WeightedReservoirSampler<int> wrs(seed);
//   for (int i = 0; i < n; ++i) wrs.add(i, weight[i]);
//   if (wrs.has_sample()) use(wrs.sample());
//
// Merge semantics: two reservoirs over disjoint sets can be merged; the
// combined reservoir is correct for the union.
//
// References:
//   pbrt-v4 WeightedReservoirSampler (util/sampling.h)
//   Talbot et al., "Importance Resampling for Global Illumination", 2005
// ===========================================================================
template <typename T>
class WeightedReservoirSampler {
  public:
	CPU_GPU WeightedReservoirSampler() = default;
	CPU_GPU explicit WeightedReservoirSampler(uint64_t rng_seed) : rng_(rng_seed) {}

	CPU_GPU void seed(uint64_t s) { rng_.set_sequence(s); }

	// Add a candidate (sample, weight) to the reservoir.
	// Returns true if the reservoir was updated (sample accepted).
	// pbrt-v4: WeightedReservoirSampler::Add(sample, weight)
	CPU_GPU bool add(const T& sample, double weight) {
		weight_sum_ += weight;
		double p = weight / weight_sum_;
		if (rng_.uniform() < p) {
			reservoir_       = sample;
			reservoir_weight_ = weight;
			return true;
		}
		return false;
	}

	// Add via lazy callback: only constructs the sample if accepted.
	// Useful when sample construction is expensive.
	// pbrt-v4: WeightedReservoirSampler::Add(func, weight)
	template <typename F>
	CPU_GPU bool add(F func, double weight) {
		weight_sum_ += weight;
		double p = weight / weight_sum_;
		if (rng_.uniform() < p) {
			reservoir_       = func();
			reservoir_weight_ = weight;
			return true;
		}
		return false;
	}

	// Merge another reservoir (over a disjoint set) into this one.
	// pbrt-v4: WeightedReservoirSampler::Merge
	CPU_GPU void merge(const WeightedReservoirSampler& other) {
		if (other.has_sample() && add(other.reservoir_, other.weight_sum_))
			reservoir_weight_ = other.reservoir_weight_;
	}

	// Copy state (weights + reservoir) but not the RNG state.
	CPU_GPU void copy_from(const WeightedReservoirSampler& other) {
		weight_sum_       = other.weight_sum_;
		reservoir_        = other.reservoir_;
		reservoir_weight_ = other.reservoir_weight_;
	}

	CPU_GPU bool   has_sample()         const { return weight_sum_ > 0.0; }
	CPU_GPU const T& sample()           const { return reservoir_; }
	CPU_GPU double weight_sum()         const { return weight_sum_; }
	CPU_GPU double reservoir_weight()   const { return reservoir_weight_; }

	// Probability that the current reservoir holds the sample it holds.
	// pbrt-v4: WeightedReservoirSampler::SampleProbability
	CPU_GPU double sample_probability() const {
		return (weight_sum_ > 0.0) ? reservoir_weight_ / weight_sum_ : 0.0;
	}

	CPU_GPU void reset() { weight_sum_ = reservoir_weight_ = 0.0; }

  private:
	PCG32Rng rng_;
	double   weight_sum_        = 0.0;
	double   reservoir_weight_  = 0.0;
	T        reservoir_         = T{};
};

// ===========================================================================
// AliasTable
//
// O(n) build / O(1) sample discrete probability distribution.
// Given a weight array w[], normalises to PMF p[i] = w[i]/sum(w),
// then builds Walker's alias table so that a single uniform sample
// in [0,1) maps to an index i with probability p[i].
//
// Usage:
//   AliasTable at({3.0, 1.0, 2.0});   // build from weights
//   double pmf;
//   int i = at.sample(u, &pmf);        // O(1) sample
//   double p = at.pmf(i);              // exact PMF at index i
//
// References:
//   pbrt-v4 AliasTable (util/sampling.h / util/sampling.cpp)
//   Walker, "An efficient method for generating discrete random variables
//            with general distributions", ACM TOMS, 1977
// ===========================================================================
class AliasTable {
  public:
	AliasTable() = default;

	// Build alias table from a vector of non-negative weights.
	// pbrt-v4: AliasTable::AliasTable(pstd::span<const Float> weights, ...)
	explicit AliasTable(const std::vector<double>& weights) {
		build(weights.data(), (int)weights.size());
	}
	AliasTable(const double* weights, int n) { build(weights, n); }

	// Sample an index in [0, n) with probability proportional to weights.
	// u        -- uniform sample in [0, 1)
	// out_pmf  -- (optional) receives the PMF at the returned index
	// out_u_remapped -- (optional) remapped uniform in the chosen bin's sub-range
	//
	// pbrt-v4: AliasTable::Sample(Float u, Float* pmf, Float* uRemapped)
	int sample(double u, double* out_pmf = nullptr, double* out_u_remapped = nullptr) const {
		int n = (int)bins_.size();
		int offset = std::min((int)(u * n), n - 1);
		double up  = std::min(u * n - offset, 1.0 - std::numeric_limits<double>::epsilon() / 2.0);

		if (up < bins_[offset].q) {
			if (out_pmf)        *out_pmf = bins_[offset].p;
			if (out_u_remapped) *out_u_remapped =
				std::min(up / bins_[offset].q, 1.0 - std::numeric_limits<double>::epsilon() / 2.0);
			return offset;
		} else {
			int alias = bins_[offset].alias;
			if (out_pmf)        *out_pmf = bins_[alias].p;
			if (out_u_remapped) {
				double denom = 1.0 - bins_[offset].q;
				*out_u_remapped = denom > 0.0
					? std::min((up - bins_[offset].q) / denom,
							   1.0 - std::numeric_limits<double>::epsilon() / 2.0)
					: 0.0;
			}
			return alias;
		}
	}

	// Exact normalized PMF at index i.
	double pmf(int i) const { return bins_[i].p; }

	int    size()    const { return (int)bins_.size(); }
	bool   empty()   const { return bins_.empty(); }

  private:
	struct Bin {
		double q;   // probability threshold within this bin (in [0,1])
		double p;   // true normalized PMF at this index
		int    alias; // alias index (-1 if none)
	};

	void build(const double* weights, int n) {
		bins_.resize(n);

		// Normalize weights to PMF
		double sum = 0.0;
		for (int i = 0; i < n; ++i) sum += weights[i];
		for (int i = 0; i < n; ++i) bins_[i].p = weights[i] / sum;

		// Walker's alias method: split into "under" and "over" lists
		struct Outcome { double p_hat; int index; };
		std::vector<Outcome> under, over;
		under.reserve(n); over.reserve(n);

		for (int i = 0; i < n; ++i) {
			double p_hat = bins_[i].p * n;
			if (p_hat < 1.0) under.push_back({p_hat, i});
			else             over.push_back({p_hat, i});
		}

		while (!under.empty() && !over.empty()) {
			Outcome un = under.back(); under.pop_back();
			Outcome ov = over.back();  over.pop_back();

			bins_[un.index].q     = un.p_hat;
			bins_[un.index].alias = ov.index;

			double p_excess = un.p_hat + ov.p_hat - 1.0;
			if (p_excess < 1.0) under.push_back({p_excess, ov.index});
			else                over.push_back ({p_excess, ov.index});
		}

		// Remaining items fill their entire bin
		while (!over.empty()) {
			Outcome ov = over.back(); over.pop_back();
			bins_[ov.index].q     = 1.0;
			bins_[ov.index].alias = -1;
		}
		while (!under.empty()) {
			Outcome un = under.back(); under.pop_back();
			bins_[un.index].q     = 1.0;
			bins_[un.index].alias = -1;
		}
	}

	std::vector<Bin> bins_;
};
