#pragma once
// ---------------------------------------------------------------------------
// restir.h -- ReSTIR / RIS primitives for direct illumination
//
// Ported from pbrt-v4 util/sampling.h (WeightedReservoirSampler) and
// lightsamplers.cpp (ExhaustiveLightSampler), aligned with Bitterli 2020.
//
// Alignment with pbrt-v4:
//   - Streaming uses WeightedReservoirSampler<RestirCandidate<T>> from
//     reservoir_sampler.h, mirroring pbrt-v4 WeightedReservoirSampler<Light>
//     in ExhaustiveLightSampler::Sample.
//   - temporal_update / spatial_merge use WRS::merge(), matching pbrt-v4
//     WeightedReservoirSampler::Merge.
//   - WRS seeded at construction (wrs(seed)), matching pbrt-v4 pattern:
//       uint64_t seed = MixBits(FloatToBits(u));
//       WeightedReservoirSampler<Light> wrs(seed);
//   - Candidate generation and WRS selection use separate RNG streams.
//   - No stack VLAs: candidates streamed directly into WRS (not staged).
//   - UCW formula: W = w_sum/(M*p_hat) [Bitterli 2020 eq.6]
//
// References:
//   pbrt-v4 src/pbrt/util/sampling.h   (WeightedReservoirSampler, Apache-2.0)
//   pbrt-v4 src/pbrt/lightsamplers.cpp (ExhaustiveLightSampler,  Apache-2.0)
//   Bitterli et al., "Spatiotemporal reservoir resampling for real-time ray
//     tracing with dynamic direct lighting", SIGGRAPH 2020.
//   Talbot et al., "Importance Resampling for Global Illumination", EGSR 2005.
// ---------------------------------------------------------------------------

#include "cpu_gpu.h"

#include "reservoir_sampler.h"   // WeightedReservoirSampler<T>, PCG32Rng
#include <cstdint>
#include <algorithm>

// ===========================================================================
// RestirCandidate<T>
//
// One candidate in the RIS stream.
//   index      -- user handle (e.g., light index). -1 = invalid.
//   source_pdf -- p(x):     density used to draw this candidate
//   target_pdf -- p_hat(x): target density proportional to the integrand
//   ris_weight() = p_hat / p  (0 when source_pdf == 0)
//
// Mirrors pbrt-v4 ExhaustiveLightSampler usage:
//   wrs.Add(boundedLights[i], lightBounds[i].Importance(ctx.p(), ctx.n))
// where Importance() is the target weight and uniform 1/N is the source.
// ===========================================================================
template <typename T>
struct RestirCandidate {
	int  index      = -1;
	T    source_pdf = T(0);
	T    target_pdf = T(0);

	// w_i = p_hat / p  (pbrt-v4: weight passed to WRS::Add)
	CPU_GPU T ris_weight() const {
		return (source_pdf > T(0)) ? target_pdf / source_pdf : T(0);
	}
};

// ===========================================================================
// Reservoir<T>
//
// Result of a RIS / ReSTIR pass over M candidates.
//   selected  <-> pbrt-v4 WRS::reservoir   (the currently held sample)
//   w_sum     <-> pbrt-v4 WRS::weightSum   (sum of all Add() weights)
//   M            number of candidates streamed (tracked for reuse)
//   W            UCW = w_sum/(M*p_hat); filled by reservoir_ucw()
// ===========================================================================
template <typename T>
struct Reservoir {
	RestirCandidate<T> selected{};
	T   w_sum = T(0);
	int M     = 0;
	T   W     = T(0);

	CPU_GPU bool valid() const { return selected.index >= 0 && w_sum > T(0); }
	CPU_GPU void clear() {
		selected = RestirCandidate<T>{};
		w_sum = T(0);
		M = 0;
		W = T(0);
	}
};

// ===========================================================================
// reservoir_ucw(r)
//
// Compute W = w_sum / (M * p_hat(selected)) and store in r.W.
// Call after ris_finish / temporal_update / spatial_merge.
// Estimator: L = f(r.selected) / r.selected.target_pdf * r.W
//
// pbrt-v4 equiv: pmf = wrs.SampleProbability() = reservoirWeight/weightSum
//   UCW = weightSum / (M * reservoirWeight) = 1/(M * pmf)
// Reference: Bitterli 2020 eq. (6).
// ===========================================================================
template <typename T>
CPU_GPU void reservoir_ucw(Reservoir<T>& r) {
	T denom = T(r.M) * r.selected.target_pdf;
	r.W = (denom > T(0)) ? r.w_sum / denom : T(0);
}

// ===========================================================================
// ris_add(wrs, candidate)
//
// Stream one candidate into a WRS using its RIS weight.
// Mirrors pbrt-v4: wrs.Add(light, importance)
// ===========================================================================
template <typename T>
CPU_GPU void ris_add(WeightedReservoirSampler<RestirCandidate<T>>& wrs,
	const RestirCandidate<T>& candidate)
{
	wrs.add(candidate, (double)candidate.ris_weight());
}

// ===========================================================================
// ris_finish(wrs, M) -> Reservoir<T>
//
// Convert a filled WRS to a Reservoir<T>. W is not computed; call
// reservoir_ucw(r) afterwards.
// pbrt-v4 equiv: SampledLight{wrs.GetSample(), wrs.SampleProbability()*scale}
// ===========================================================================
template <typename T>
CPU_GPU Reservoir<T> ris_finish(
	const WeightedReservoirSampler<RestirCandidate<T>>& wrs, int M)
{
	Reservoir<T> r;
	r.M     = M;
	r.w_sum = T(wrs.weight_sum());
	if (wrs.has_sample())
		r.selected = wrs.sample();
	return r;
}

// ===========================================================================
// ris_fill(r, candidates, n, seed)
//
// Batch: stream n pre-built candidates through a WRS seeded with `seed`.
// Mirrors pbrt-v4 ExhaustiveLightSampler::Sample:
//   uint64_t seed = MixBits(FloatToBits(u));
//   WeightedReservoirSampler<Light> wrs(seed);
//   for (...) wrs.Add(lights[i], importance[i]);
// Call reservoir_ucw(r) after to compute r.W.
// ===========================================================================
template <typename T>
CPU_GPU void ris_fill(Reservoir<T>& r,
	const RestirCandidate<T>* candidates, int n, uint64_t seed)
{
	WeightedReservoirSampler<RestirCandidate<T>> wrs(seed);
	for (int i = 0; i < n; ++i)
		ris_add(wrs, candidates[i]);
	r = ris_finish<T>(wrs, n);
}

// ===========================================================================
// temporal_update(current, prev, seed, max_M)
//
// Merge a previous-frame reservoir into current using WRS::merge(), cap M.
// Mirrors pbrt-v4 WeightedReservoirSampler::Merge:
//   if (wrs.HasSample() && Add(wrs.reservoir, wrs.weightSum))
//       reservoirWeight = wrs.reservoirWeight;
// M = min(current.M + prev.M, max_M). Call reservoir_ucw() afterwards.
// Reference: Bitterli 2020 §4.3.
// ===========================================================================
template <typename T>
CPU_GPU void temporal_update(Reservoir<T>& current,
	const Reservoir<T>& prev, uint64_t seed, int max_M = 20)
{
	if (!prev.valid()) return;

	WeightedReservoirSampler<RestirCandidate<T>> wrs(seed);
	if (current.valid())
		wrs.add(current.selected, (double)current.w_sum);

	// Build a WRS for prev and merge -- mirrors WRS::Merge
	WeightedReservoirSampler<RestirCandidate<T>> prev_wrs(seed ^ 0xdeadbeef89abcdefULL);
	prev_wrs.add(prev.selected, (double)prev.w_sum);
	wrs.merge(prev_wrs);

	int new_M = std::min(current.M + prev.M, max_M);
	current = ris_finish<T>(wrs, new_M);
}

// ===========================================================================
// spatial_merge(current, neighbors, k, seed, max_M)
//
// Merge k neighbor reservoirs into current using WRS::merge() per neighbor.
// Each neighbor treated as a super-candidate with weight = neighbor.w_sum.
// M = min(sum of all M values, max_M). Call reservoir_ucw() afterwards.
// Note: for full unbiasedness, re-evaluate p_hat for each neighbor's sample
// at this pixel's context and call reservoir_ucw() with the corrected value.
// Reference: Bitterli 2020 §4.4 Algorithm 3.
// ===========================================================================
template <typename T>
CPU_GPU void spatial_merge(Reservoir<T>& current,
	const Reservoir<T>* neighbors, int k, uint64_t seed, int max_M = 500)
{
	WeightedReservoirSampler<RestirCandidate<T>> wrs(seed);
	if (current.valid())
		wrs.add(current.selected, (double)current.w_sum);

	int total_M = current.M;
	for (int j = 0; j < k; ++j) {
		const Reservoir<T>& nb = neighbors[j];
		if (!nb.valid()) continue;

		uint64_t nb_seed = seed ^ (uint64_t(j + 1) * 0x9e3779b97f4a7c15ULL);
		WeightedReservoirSampler<RestirCandidate<T>> nb_wrs(nb_seed);
		nb_wrs.add(nb.selected, (double)nb.w_sum);
		wrs.merge(nb_wrs);

		total_M = std::min(total_M + nb.M, max_M);
	}
	current = ris_finish<T>(wrs, total_M);
}

// ===========================================================================
// restir_direct_sample<T, SourceFn>(M, source_fn, seed)
//
// Stream M candidates from source_fn through a seeded WRS, compute UCW.
// Candidate generation and WRS selection use separate RNG streams, matching
// pbrt-v4's design where WRS has its own seeded internal RNG independent of
// the caller's sampling state.
//
// source_fn signature: RestirCandidate<T>(PCG32Rng& rng)
//
// Mirrors pbrt-v4 ExhaustiveLightSampler::Sample:
//   WeightedReservoirSampler<Light> wrs(seed);
//   for (...) wrs.Add(lights[i], importance[i]);
//   return SampledLight{wrs.GetSample(), pmf};
//
// Estimator: L = f(r.selected) / r.selected.target_pdf * r.W
// Reference: Bitterli 2020 eq. (6).
// ===========================================================================
template <typename T, typename SourceFn>
CPU_GPU Reservoir<T> restir_direct_sample(int M, SourceFn source_fn, uint64_t seed)
{
	PCG32Rng gen_rng(seed ^ 0x9e3779b97f4a7c15ULL);  // candidate generation RNG
	WeightedReservoirSampler<RestirCandidate<T>> wrs(seed);  // selection RNG

	for (int i = 0; i < M; ++i)
		ris_add(wrs, source_fn(gen_rng));

	Reservoir<T> r = ris_finish<T>(wrs, M);
	reservoir_ucw(r);
	return r;
}
