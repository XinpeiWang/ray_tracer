// adaptive_sampling_tests.cpp -- unit tests for
// src/shared/adaptive_sampling.h's pixel_convergence::luminance()/
// has_converged(), the pure function camera.h's render() loop uses to
// decide whether a pixel has taken enough samples under --adaptive.
// Header-only, no rendering needed - same convention as
// light_sampler_resolution_tests.cpp's own top-of-file comment for why a
// pure resolution function gets its own direct test instead of only being
// exercised indirectly through a full render.

#include <gtest/gtest.h>
#include "../../src/shared/adaptive_sampling.h"

using pixel_convergence::has_converged;
using pixel_convergence::luminance;

TEST(PixelConvergenceLuminance, MatchesRec709Weights) {
	// Same formula as bdpt_adapter.h's own Luminance() - see
	// adaptive_sampling.h's own comment for why they're not shared code.
	EXPECT_DOUBLE_EQ(luminance(1.0, 0.0, 0.0), 0.2126);
	EXPECT_DOUBLE_EQ(luminance(0.0, 1.0, 0.0), 0.7152);
	EXPECT_DOUBLE_EQ(luminance(0.0, 0.0, 1.0), 0.0722);
	EXPECT_NEAR(luminance(1.0, 1.0, 1.0), 1.0, 1e-12);
}

TEST(PixelConvergenceHasConverged, FalseBelowTwoSamples) {
	VarianceEstimator<double> e;
	EXPECT_FALSE(has_converged(e, 0.01));
	e.Add(0.5);
	EXPECT_FALSE(has_converged(e, 0.01));  // still only 1 sample
}

TEST(PixelConvergenceHasConverged, ConvergesOnIdenticalSamples) {
	// Zero variance - any positive threshold should immediately pass, once
	// there are enough samples for Variance() to be defined.
	VarianceEstimator<double> e;
	e.Add(0.5);
	e.Add(0.5);
	EXPECT_TRUE(has_converged(e, 0.01));
}

TEST(PixelConvergenceHasConverged, DoesNotConvergeOnHighVarianceSamples) {
	// Wildly different samples (e.g. an unconverged caustic/fireflies) -
	// the relative standard error should stay well above a tight threshold.
	VarianceEstimator<double> e;
	e.Add(0.0);
	e.Add(100.0);
	EXPECT_FALSE(has_converged(e, 0.01));
}

TEST(PixelConvergenceHasConverged, MoreSamplesEventuallyConvergeTheSameTrueMean) {
	// Same distribution, just more of it - standard error shrinks as
	// samples accumulate (∝ 1/sqrt(n)), so what doesn't converge at n=2
	// should converge once enough samples of the SAME distribution accumulate.
	VarianceEstimator<double> e;
	for (int i = 0; i < 2; ++i) { e.Add(1.0); e.Add(1.02); }
	const bool convergedEarly = has_converged(e, 0.001);
	for (int i = 0; i < 200; ++i) { e.Add(1.0); e.Add(1.02); }
	EXPECT_FALSE(convergedEarly);
	EXPECT_TRUE(has_converged(e, 0.001));
}

TEST(PixelConvergenceHasConverged, NearBlackPixelConvergesUnconditionally) {
	// A pixel correctly converging toward zero (e.g. deep shadow) must not
	// be judged by RELATIVE error against a near-zero mean - that ratio
	// would stay large (or explode) forever otherwise.
	VarianceEstimator<double> e;
	e.Add(0.0);
	e.Add(1e-6);
	e.Add(2e-6);
	EXPECT_TRUE(has_converged(e, 0.001, /*black_floor=*/1e-4));
}

TEST(PixelConvergenceHasConverged, TighterThresholdNeedsMoreSamples) {
	VarianceEstimator<double> loose, tight;
	for (int i = 0; i < 4; ++i) {
		loose.Add(1.0); loose.Add(1.1);
		tight.Add(1.0); tight.Add(1.1);
	}
	EXPECT_TRUE(has_converged(loose, 0.2));
	EXPECT_FALSE(has_converged(tight, 0.001));
}
