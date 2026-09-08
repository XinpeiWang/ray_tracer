// adaptive_sampling_tests.cpp -- unit tests for
// src/shared/adaptive_sampling.h's pixel_convergence::luminance()/
// has_converged()/row_visit_order(), the pure functions camera.h's
// render() loop uses to decide whether (and how) a pixel has taken enough
// samples under --adaptive. Header-only, no rendering needed - same
// convention as light_sampler_resolution_tests.cpp's own top-of-file
// comment for why a pure resolution function gets its own direct test
// instead of only being exercised indirectly through a full render.

#include <gtest/gtest.h>
#include "../../src/shared/adaptive_sampling.h"

#include <algorithm>
#include <vector>

using pixel_convergence::has_converged;
using pixel_convergence::luminance;
using pixel_convergence::row_visit_order;

TEST(PixelConvergenceLuminance, MatchesRec709Weights) {
	// bdpt_adapter.h's own Luminance() delegates to this exact function -
	// see adaptive_sampling.h's own comment.
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

// The near-black shortcut must judge brightness AFTER exposure, not the
// raw pre-exposure radiance - otherwise a high --exposure (common for a
// deliberately dark scene boosted in post) would fast-track pixels as
// "converged" that become visible, noise-sensitive midtones once exposure
// is actually applied.
TEST(PixelConvergenceHasConverged, ExposureAffectsNearBlackShortcut) {
	VarianceEstimator<double> e;
	e.Add(0.0);
	e.Add(2e-5);  // below the 1e-4 default black_floor on its own
	// At exposure=1.0 (default), this is still near-black - converges
	// unconditionally regardless of its (high) relative variance.
	EXPECT_TRUE(has_converged(e, 0.001, 1e-4, 1.0));
	// At exposure=50.0, the same raw mean (1e-5) scales to 5e-4, ABOVE the
	// 1e-4 floor - no longer near-black, so it falls through to the
	// ordinary relative-error test, which this high-variance pair (0.0 vs
	// 2e-5, a 2x spread) fails.
	EXPECT_FALSE(has_converged(e, 0.001, 1e-4, 50.0));
}

TEST(PixelConvergenceHasConverged, DefaultExposureMatchesNoExposureArgument) {
	VarianceEstimator<double> e;
	e.Add(0.0);
	e.Add(5e-5);
	EXPECT_EQ(has_converged(e, 0.001, 1e-4), has_converged(e, 0.001, 1e-4, 1.0));
}

// ---------------------------------------------------------------------------
// row_visit_order()
// ---------------------------------------------------------------------------

TEST(PixelConvergenceRowVisitOrder, IsAValidPermutation) {
	for (int n : {1, 2, 3, 4, 5, 8, 16, 22}) {
		std::vector<int> order = row_visit_order(n);
		ASSERT_EQ((int)order.size(), n);
		std::vector<int> sorted = order;
		std::sort(sorted.begin(), sorted.end());
		for (int i = 0; i < n; ++i) EXPECT_EQ(sorted[i], i) << "for n=" << n;
	}
}

// The whole point: any PREFIX of the returned order should be spread across
// the full [0,n) range, not confined to one contiguous end - verified here
// by checking a short prefix's max-min span is close to n, not close to
// the prefix's own length (which raster order 0,1,2,... would give).
TEST(PixelConvergenceRowVisitOrder, ShortPrefixIsSpreadNotClustered) {
	std::vector<int> order = row_visit_order(16);
	std::vector<int> prefix(order.begin(), order.begin() + 4);
	const int span = *std::max_element(prefix.begin(), prefix.end())
					- *std::min_element(prefix.begin(), prefix.end());
	// Raster order's first-4 span would be 3 (indices 0,1,2,3). A
	// well-spread prefix should cover much more of the 0..15 range.
	EXPECT_GT(span, 8) << "first 4 of row_visit_order(16): "
						<< prefix[0] << "," << prefix[1] << "," << prefix[2] << "," << prefix[3];
}

TEST(PixelConvergenceRowVisitOrder, FirstElementIsAlwaysZero) {
	// Radical inverse of 0 is 0, the smallest possible value - it always
	// sorts first regardless of n.
	for (int n : {1, 4, 7, 16}) {
		EXPECT_EQ(row_visit_order(n).front(), 0);
	}
}
