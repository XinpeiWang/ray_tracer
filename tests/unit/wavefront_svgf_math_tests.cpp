// wavefront_svgf_math_tests.cpp
// Validation for gpu/optix/wavefront_svgf_math.h's pure-math SVGF primitives
// - variance estimation, the temporal blend schedule, and the A-trous
// edge-stopping weight functions. All CPU_GPU/plain-float functions, so no
// GPU hardware is required (mirrors wavefront_restir_gi_tests.cpp's own
// host-build-only approach).
//
// Tests:
// wf_svgf_luminance
//   1. White gives luminance 1; pure R/G/B give their own Rec.709 coefficient
// wf_svgf_variance
//   2. Matches moment2 - moment1^2 for a simple case
//   3. Clamped to 0 for a (numerically-possible) negative input
// wf_svgf_temporal_alpha
//   4. historyLength=0 gives alpha=1 (no history - trust the fresh sample
//      entirely)
//   5. Behaves as a running mean (alpha=1/(n+1)) while history is short
//   6. Clamps to targetAlpha once history is long enough
// wf_svgf_weight_normal / _depth / _luminance / edge_weight
//   7. Identical neighbors give weight 1 on every term
//   8. Perpendicular normals collapse the normal weight to 0
//   9. A depth difference far exceeding the locally-planar prediction
//      collapses the depth weight toward 0
//   10. A high-variance (noisy) pixel tolerates a much larger luminance
//       difference than a low-variance (converged) one before its weight
//       collapses
//   11. Combined edge_weight is the product of all three terms

#include <gtest/gtest.h>
#include "wavefront_svgf_math.h"
#include <cmath>

TEST(WfSvgfLuminance, MatchesRec709Coefficients) {
	EXPECT_NEAR(wf_svgf_luminance(make_float3(1.0f, 1.0f, 1.0f)), 1.0f, 1e-5f);
	EXPECT_NEAR(wf_svgf_luminance(make_float3(1.0f, 0.0f, 0.0f)), 0.2126f, 1e-5f);
	EXPECT_NEAR(wf_svgf_luminance(make_float3(0.0f, 1.0f, 0.0f)), 0.7152f, 1e-5f);
	EXPECT_NEAR(wf_svgf_luminance(make_float3(0.0f, 0.0f, 1.0f)), 0.0722f, 1e-5f);
}

TEST(WfSvgfVariance, MatchesSecondMinusFirstSquared) {
	EXPECT_NEAR(wf_svgf_variance(/*moment1=*/2.0f, /*moment2=*/5.0f), 1.0f, 1e-5f);
}

TEST(WfSvgfVariance, ClampedNonNegative) {
	// A numerically-possible case where moment2 < moment1^2 due to
	// floating-point cancellation on a near-zero-variance signal.
	EXPECT_FLOAT_EQ(wf_svgf_variance(/*moment1=*/3.0f, /*moment2=*/8.999f), 0.0f);
}

TEST(WfSvgfTemporalAlpha, OneWhenNoHistory) {
	EXPECT_NEAR(wf_svgf_temporal_alpha(/*historyLength=*/0.0f, /*targetAlpha=*/0.2f), 1.0f, 1e-5f);
}

TEST(WfSvgfTemporalAlpha, ActsAsRunningMeanWhileHistoryIsShort) {
	// Below the crossover (historyLength < 1/targetAlpha - 1 = 4), alpha
	// should exactly match a plain running-mean weight 1/(n+1).
	EXPECT_NEAR(wf_svgf_temporal_alpha(1.0f, 0.2f), 1.0f / 2.0f, 1e-5f);
	EXPECT_NEAR(wf_svgf_temporal_alpha(3.0f, 0.2f), 1.0f / 4.0f, 1e-5f);
}

TEST(WfSvgfTemporalAlpha, ClampsToTargetOnceHistoryIsLongEnough) {
	// At historyLength=4, 1/(4+1)=0.2 exactly matches targetAlpha - the
	// crossover point. Past it, the running-mean weight would keep
	// shrinking, but alpha must clamp at targetAlpha instead.
	EXPECT_NEAR(wf_svgf_temporal_alpha(4.0f, 0.2f), 0.2f, 1e-5f);
	EXPECT_NEAR(wf_svgf_temporal_alpha(50.0f, 0.2f), 0.2f, 1e-5f);
}

TEST(WfSvgfWeights, IdenticalNeighborsGiveWeightOneOnEveryTerm) {
	float3 normal = make_float3(0.0f, 1.0f, 0.0f);
	EXPECT_NEAR(wf_svgf_weight_normal(normal, normal, 128.0f), 1.0f, 1e-5f);
	EXPECT_NEAR(wf_svgf_weight_depth(/*depthP=*/5.0f, /*depthQ=*/5.0f, /*gradDotOffset=*/1.0f, /*sigmaDepth=*/1.0f), 1.0f, 1e-5f);
	EXPECT_NEAR(wf_svgf_weight_luminance(/*lumP=*/0.5f, /*lumQ=*/0.5f, /*sqrtVariance=*/0.1f, /*sigmaLuminance=*/4.0f), 1.0f, 1e-5f);
}

TEST(WfSvgfWeights, PerpendicularNormalsCollapseToZero) {
	float3 normalP = make_float3(0.0f, 1.0f, 0.0f);
	float3 normalQ = make_float3(1.0f, 0.0f, 0.0f);
	EXPECT_NEAR(wf_svgf_weight_normal(normalP, normalQ, 128.0f), 0.0f, 1e-6f);
}

TEST(WfSvgfWeights, DepthFarBeyondPlanarPredictionCollapsesToNearZero) {
	// A gentle, locally-planar gradient predicts only a tiny depth change
	// over this offset, but the actual difference is huge - a genuine
	// silhouette edge, which the weight should reject.
	float w = wf_svgf_weight_depth(/*depthP=*/5.0f, /*depthQ=*/500.0f, /*gradDotOffset=*/0.01f, /*sigmaDepth=*/1.0f);
	EXPECT_LT(w, 1e-4f);
}

TEST(WfSvgfWeights, HighVarianceToleratesLargerLuminanceDifference) {
	const float lumP = 1.0f, lumQ = 2.0f;  // same absolute luminance gap in both cases
	const float wLowVariance = wf_svgf_weight_luminance(lumP, lumQ, /*sqrtVariance=*/0.01f, /*sigmaLuminance=*/4.0f);
	const float wHighVariance = wf_svgf_weight_luminance(lumP, lumQ, /*sqrtVariance=*/5.0f, /*sigmaLuminance=*/4.0f);
	EXPECT_LT(wLowVariance, 0.01f);   // a converged pixel rejects this neighbor almost entirely
	EXPECT_GT(wHighVariance, 0.9f);   // a still-noisy pixel barely penalizes it at all
}

TEST(WfSvgfWeights, EdgeWeightIsProductOfAllThreeTerms) {
	float3 normalP = make_float3(0.0f, 1.0f, 0.0f);
	float3 normalQ = make_float3(0.0f, 1.0f, 0.0f);
	const float expected =
		wf_svgf_weight_normal(normalP, normalQ, 128.0f) *
		wf_svgf_weight_depth(5.0f, 5.2f, 1.0f, 1.0f) *
		wf_svgf_weight_luminance(0.5f, 0.6f, 0.2f, 4.0f);
	const float actual = wf_svgf_edge_weight(
		normalP, normalQ, 128.0f,
		5.0f, 5.2f, 1.0f, 1.0f,
		0.5f, 0.6f, 0.2f, 4.0f);
	EXPECT_NEAR(actual, expected, 1e-6f);
}
