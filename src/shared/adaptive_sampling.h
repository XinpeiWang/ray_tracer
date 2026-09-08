#pragma once
// adaptive_sampling.h -- the pure per-pixel convergence-stopping decision
// camera.h's CPU default path tracer render loop uses to decide whether a
// pixel has converged well enough to stop spending more of its
// samples-per-pixel budget on it, redistributing that time to noisier
// pixels instead - the same idea as Blender Cycles' own adaptive sampling
// (use_adaptive_sampling/adaptive_threshold), scoped down to a single,
// pixel-local stopping test since this project's stratified sqrt_spp x
// sqrt_spp sampling loop (camera.h's render()) already visits pixels
// independently, one at a time, within a single thread - no cross-pixel
// budget reallocation machinery is needed, just an early `break` out of
// that loop.
//
// Tracks per-sample LUMINANCE (0.2126/0.7152/0.0722 Rec.709 weights - the
// one shared implementation bdpt_adapter.h's own Luminance() now delegates
// to, for MLT's acceptance-ratio use of the same formula), not the full
// RGB color - reducing to one scalar
// noise estimate is standard practice for this kind of heuristic (pbrt/
// Cycles/production adaptive-sampling implementations all reduce to one
// channel or luminance rather than tracking full per-channel covariance),
// and is what VarianceEstimator (src/shared/octahedral_variance.h,
// Welford's online algorithm, already used elsewhere in this codebase)
// already operates on.
//
// Pulled into its own pure, header-only, unit-tested function (rather than
// left inline in camera.h's render loop) for the same reason
// light_sampler_resolution.h was: testable without a full render - see
// tests/unit/adaptive_sampling_tests.cpp.

#include "lowdiscrepancy.h"  // ReverseBits32 - row_visit_order()'s own comment
#include "octahedral_variance.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Named pixel_convergence, not adaptive_sampling, deliberately - camera.h's
// own bool field is named adaptive_sampling (the flag this file backs), and
// an unqualified name inside a camera:: member function would resolve to
// that member, silently shadowing a same-named namespace rather than
// failing to compile - not a risk worth taking for the sake of a
// same-named namespace.
namespace pixel_convergence {

inline double luminance(double r, double g, double b) {
	return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

// True once the running estimate's standard error, relative to its own
// mean, drops below `threshold` - i.e. "confident enough in this pixel's
// average brightness that more samples wouldn't change it much". Requires
// at least 2 samples (Welford's variance is undefined below that).
//
// A near-black pixel (mean below `black_floor`, AFTER `exposure` is
// applied - see this parameter's own note below) is treated as converged
// unconditionally rather than divided into a permanently-large relative
// error by a near-zero mean - matches Cycles' own behavior of not
// endlessly re-sampling background/shadow pixels that are correctly
// converging to (near) zero.
//
// `exposure` - camera::exposure's own flat post-multiply, 1.0 (default) if
// not passed - is applied to `mean` ONLY for this near-black comparison,
// not to the relative-error check above (that ratio is already scale-
// invariant under any positive multiplier, exposure included). Without
// this, a raw pre-exposure radiance below `black_floor` would fast-track
// as "converged" after just 2 samples even under a large --exposure
// (common for a deliberately dark/night scene boosted in post) that turns
// those same pixels into visible, noise-sensitive midtones once applied -
// the floor would silently stop protecting exactly the pixels a chosen
// exposure was about to make matter.
inline bool has_converged(const VarianceEstimator<double> &estimator, double threshold,
						   double black_floor = 1e-4, double exposure = 1.0) {
	if (estimator.Count() < 2) return false;
	const double mean = estimator.Mean();
	if (mean * exposure < black_floor) return true;
	const double standard_error = std::sqrt(estimator.Variance() / static_cast<double>(estimator.Count()));
	return (standard_error / mean) < threshold;
}

// A permutation of {0, ..., n-1} such that ANY prefix of the permuted
// sequence is itself well-spread across the full range - built by sorting
// indices by their base-2 radical inverse (Van der Corput) value, the same
// bit-reversal construction pbrt-v4's own low-discrepancy sequences use
// (see ReverseBits32's own comment, lowdiscrepancy.h) - CONFIRMED here for
// arbitrary n, not just powers of 2: sorting is a well-defined total order
// regardless of how many of the reversed-bit values are actually distinct.
//
// Exists because camera.h's render loop visits stratified sample ROWS
// (camera::sample_unit_square_stratified's own comment) using this order
// instead of raster order 0,1,2,...,n-1. A raster-order prefix is NOT
// well-spread - it is confined to one contiguous, spatially-lopsided end of
// the pixel reconstruction filter's footprint (FilterSampler::sample()'s
// "u2 -> row via marginal CDF" comment, filter_sampler.h, maps a
// stratified row's u2 value monotonically to a vertical filter position) -
// so a pixel that stops early under --adaptive before visiting every row
// would otherwise systematically under-sample one side of the filter
// instead of taking a representative, if smaller, sample of it.
//
// Reordering which row is visited when has NO effect at all on the final
// pixel value once every row IS visited: camera.h's weighted_color/
// weight_sum accumulation is a commutative sum over (s_i, s_j) pairs, and
// each row's own u1/u2 stratification math (sample_unit_square_stratified)
// depends only on the (s_i, s_j) VALUES, not on which loop iteration
// visits them - so a non-adaptive (or never-converging) render is
// bit-for-bit unaffected by this reordering (modulo floating-point
// summation-order noise, the same as any other reassociation of a sum).
inline std::vector<int> row_visit_order(int n) {
	std::vector<int> order(n);
	for (int i = 0; i < n; ++i) order[i] = i;
	std::sort(order.begin(), order.end(), [](int a, int b) {
		return ReverseBits32(static_cast<uint32_t>(a)) < ReverseBits32(static_cast<uint32_t>(b));
	});
	return order;
}

} // namespace pixel_convergence
