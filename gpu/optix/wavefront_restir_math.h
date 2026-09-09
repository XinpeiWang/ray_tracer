#pragma once
// wavefront_restir_math.h -- ReSTIR DI's pure RIS/reservoir-combine math and
// spatial-reuse geometric guard: exactly the pieces that need no scene data
// (no SphereData*/QuadData*/etc, no alias table, no RNG draw beyond an
// already-generated [0,1) float) - split out of wavefront_restir_helpers.h
// (which also carries __device__-only scene-sampling logic depending on
// wavefront_device_helpers.h's wf_sample_*_light/wf_rand/wf_dc_* functions,
// and so can only be included from WITHIN that file, after those are already
// defined) purely so THESE formulas can be unit-tested from a plain host
// build with no OptiX-module include-order constraints - see
// tests/unit/wavefront_restir_tests.cpp. wavefront_restir_helpers.h includes
// this header for its own use; nothing here duplicates anything there.
//
// Mirrors src/shared/restir.h's RIS/reservoir math (RestirCandidate<T>,
// Reservoir<T>, ris_fill, reservoir_ucw - see that header for the reference
// formulas and its own test coverage), ported to GpuReservoir/float instead
// of that header's generic T/double, for the same reasons
// wavefront_restir_helpers.h's own header comment gives.
//
// restir.h's temporal_update()/spatial_merge() explicitly do NOT implement
// the reweighting needed for unbiased reuse (that header's own TODO comment:
// "for full unbiasedness, re-evaluate p_hat for each neighbor's sample at
// this pixel's context and call reservoir_ucw() with the corrected value").
// restir_reservoir_combine() below IS that missing piece - every caller
// (wf_restir_temporal_combine/restir_spatial_reuse, wavefront_restir_helpers.h/
// wavefront_kernels_restir.cu) re-evaluates the OTHER reservoir's stored
// sample's target function fresh, at the CURRENT pixel's own context, before
// folding it in here.

#include "optix_types.h"
#include "optix_math_helpers.h"   // dot()/fabsf() via <cmath> already pulled in transitively

// ===========================================================================
// Core RIS math - pure float in/out, no CUDA-only types beyond float3/
// GpuReservoir (both plain, host+device-safe structs - optix_types.h's own
// comment), so callable and unit-testable from plain host C++ as well as
// device code.
// ===========================================================================

// Streams one candidate into `r` via weighted reservoir sampling (Bitterli
// eq. 5's streaming RIS update). `risWeight` is target_pdf/source_pdf for
// this candidate; `candidateM` is how many underlying samples it represents
// (1 for a freshly-drawn candidate, an existing reservoir's own M when
// combining reservoirs - see restir_reservoir_combine); `candidatePHat` is
// the candidate's own target-function value, stored on acceptance so the
// caller can later call restir_finalize() without re-evaluating it again.
// `rand01` MUST be a fresh uniform random in [0,1) per call. Returns true iff
// `candidate` became (or stayed) the reservoir's selected sample.
CPU_GPU inline bool restir_reservoir_add(GpuReservoir& r, const GpuLightSample& candidate,
										  float risWeight, int candidateM, float candidatePHat,
										  float rand01) {
	r.weightSum += risWeight;
	r.M += candidateM;
	if (r.weightSum <= 0.0f || risWeight <= 0.0f) return false;
	if (rand01 * r.weightSum < risWeight) {
		r.sample = candidate;
		r.pHat = candidatePHat;
		return true;
	}
	return false;
}

// Unbiased contribution weight - Bitterli eq. 6: W = w_sum / (M * pHat).
// Mirrors restir.h's reservoir_ucw exactly (0 when M*pHat is non-positive).
CPU_GPU inline float restir_reservoir_ucw(float weightSum, int M, float pHat) {
	float denom = float(M) * pHat;
	return (denom > 0.0f) ? (weightSum / denom) : 0.0f;
}

CPU_GPU inline void restir_finalize(GpuReservoir& r) {
	r.W = restir_reservoir_ucw(r.weightSum, r.M, r.pHat);
}

// Combines an already-formed reservoir `other` into `dst`, both understood to
// describe the SAME pixel's context (`dst` is that pixel's own running
// reservoir; `other` is a temporal or spatial neighbor's reservoir being
// reused there). `otherPHatAtDstContext` MUST be `other.sample`'s target
// function freshly evaluated at `dst`'s own shading point (never `other`'s
// stored `pHat`, which was evaluated at `other`'s original pixel) - this is
// exactly restir.h's documented missing piece for full unbiasedness. Treats
// `other` as one weighted candidate of weight `otherPHatAtDstContext *
// other.W * other.M`, carrying `other.M` samples - the standard reservoir-
// combine identity (Bitterli Algorithm 4): reusing a whole reservoir's UCW
// as a single RIS candidate weight is valid because W is itself an unbiased
// estimator of 1/pHat integrated over that reservoir's own M candidates.
CPU_GPU inline bool restir_reservoir_combine(GpuReservoir& dst, const GpuReservoir& other,
											  float otherPHatAtDstContext, float rand01) {
	if (!other.valid() || other.M <= 0) return false;
	float w = otherPHatAtDstContext * other.W * float(other.M);
	return restir_reservoir_add(dst, other.sample, w, other.M, otherPHatAtDstContext, rand01);
}

// Resampling-only target-function proxy shared by candidate generation AND
// temporal/spatial reuse's re-evaluation step - a plain Lambertian-cosine-
// weighted luminance-like magnitude of a (already twoSided-gated) raw
// emission, NOT the exact per-material BSDF value (that's only evaluated
// once, for the FINAL winning sample, by wf_finish_material_scatter's own
// existing per-material dispatch). Every reservoir combine uses THIS SAME
// function for p_hat, which is what RIS/reservoir-combine actually requires
// for correctness (a fixed, consistently-applied target function); an
// approximate p_hat only costs variance, never correctness. Must be called
// with `dir` pointing FROM the query origin TOWARD the light sample (matches
// every wf_sample_*_light's own convention).
CPU_GPU inline float wf_restir_target_proxy(float3 rawEmission, float3 dir, float3 normal) {
	const float cosProxy = fmaxf(dot(dir, normal), 0.0f);
	return ((rawEmission.x + rawEmission.y + rawEmission.z) * (1.0f / 3.0f)) * cosProxy;
}

// Geometric-ratio robustness guard for spatial reuse (Bitterli 2020 eq. 11's
// cos/dist^2 ratio between the two shading points' view of the same sampled
// light point) - NOT an additional multiplicative correction on top of
// wf_reevaluate_light_geometry's fresh cosine/distance recompute (applying
// both would double the same geometric term; see wavefront_restir_helpers.h's
// own header comment for why the re-evaluation alone is already the full
// unbiased correction this codebase's restir.h identifies as missing). Used
// only to REJECT a spatial neighbor whose light-sample geometry differs too
// drastically between the current and neighbor shading points (e.g. the
// sample is steeply grazing from one point but not the other), the same
// variance-control role normal/depth neighbor rejection already plays - a
// far-from-1 ratio signals the neighbor's sample is a poor, high-variance fit
// for the current pixel, not that it is biased to include (used by
// wavefront_kernels_restir.cu's own neighbor-acceptance test).
CPU_GPU inline float wf_restir_jacobian(const float3& currentDir, float currentDist,
										 const float3& neighborDir, float neighborDist,
										 const float3& lightNormal) {
	const float cosCurrent = fabsf(dot(currentDir, lightNormal));
	const float cosNeighbor = fabsf(dot(neighborDir, lightNormal));
	if (cosNeighbor < 1e-6f || neighborDist < 1e-6f) return 0.0f;
	const float numerator = cosCurrent * neighborDist * neighborDist;
	const float denominator = cosNeighbor * currentDist * currentDist;
	return (denominator > 1e-12f) ? (numerator / denominator) : 0.0f;
}
