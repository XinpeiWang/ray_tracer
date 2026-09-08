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
// Tracks per-sample LUMINANCE (0.2126/0.7152/0.0722 Rec.709 weights,
// matching bdpt_adapter.h's own Luminance(), the MLT acceptance-ratio use
// of the same formula), not the full RGB color - reducing to one scalar
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

#include "octahedral_variance.h"

#include <algorithm>
#include <cmath>

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
// A near-black pixel (mean below `black_floor`) is treated as converged
// unconditionally rather than divided into a permanently-large relative
// error by a near-zero mean - matches Cycles' own behavior of not
// endlessly re-sampling background/shadow pixels that are correctly
// converging to (near) zero.
inline bool has_converged(const VarianceEstimator<double> &estimator, double threshold,
						   double black_floor = 1e-4) {
	if (estimator.Count() < 2) return false;
	const double mean = estimator.Mean();
	if (mean < black_floor) return true;
	const double standard_error = std::sqrt(estimator.Variance() / static_cast<double>(estimator.Count()));
	return (standard_error / mean) < threshold;
}

} // namespace pixel_convergence
