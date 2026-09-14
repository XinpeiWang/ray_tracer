// wavefront_restir_volume_tests.cpp
// Validation for gpu/optix/wavefront_restir_volume_math.h's pure-math ReSTIR-
// for-volumetric-media primitives - see that header's own comment for why it
// is deliberately thin (no Jacobian derivation here; wf_restir_jacobian,
// already covered by wavefront_restir_tests.cpp, is reused unmodified).
//
// Tests:
// wf_restir_volume_mean_free_path
//   1. Returns 1/sigma for an ordinary positive extinction/majorant value
//   2. Floors at the same 1e-6f epsilon a near-zero/zero sigma would
//      otherwise divide-by-zero on, matching the header's own fmaxf(...,
//      1e-6f) guard
//
// wf_restir_volume_spatial_valid
//   3. Same medium, well within the mean-free-path-scaled threshold -> valid
//   4. Same medium, well beyond the threshold -> invalid
//   5. Different mediums, even at zero distance -> invalid (hard reject)
//   6. Either mediumMatIdx negative ("never written" sentinel) -> invalid
//   7. Boundary: distance exactly at the threshold -> valid (<=, not <)

#include <gtest/gtest.h>
#include "wavefront_restir_volume_math.h"
#include <cmath>

// ============================================================
// wf_restir_volume_mean_free_path
// ============================================================

TEST(WfRestirVolumeMeanFreePath, ReciprocalOfExtinction) {
	EXPECT_NEAR(wf_restir_volume_mean_free_path(2.0f), 0.5f, 1e-6f);
	EXPECT_NEAR(wf_restir_volume_mean_free_path(0.1f), 10.0f, 1e-4f);
}

TEST(WfRestirVolumeMeanFreePath, FloorsAtEpsilonForNearZeroSigma) {
	const float expected = 1.0f / 1e-6f;
	EXPECT_NEAR(wf_restir_volume_mean_free_path(0.0f), expected, 1.0f);
	EXPECT_NEAR(wf_restir_volume_mean_free_path(-5.0f), expected, 1.0f);
}

// ============================================================
// wf_restir_volume_spatial_valid
// ============================================================

TEST(WfRestirVolumeSpatialValid, SameMediumWithinThresholdIsValid) {
	float3 cur = make_float3(0.0f, 0.0f, 0.0f);
	float3 neighbor = make_float3(1.0f, 0.0f, 0.0f);  // distance 1
	const float meanFreePath = 1.0f;
	const float scale = 3.0f;  // threshold = 3
	EXPECT_TRUE(wf_restir_volume_spatial_valid(cur, neighbor, /*curMatIdx=*/2, /*nMatIdx=*/2, meanFreePath, scale));
}

TEST(WfRestirVolumeSpatialValid, SameMediumBeyondThresholdIsInvalid) {
	float3 cur = make_float3(0.0f, 0.0f, 0.0f);
	float3 neighbor = make_float3(10.0f, 0.0f, 0.0f);  // distance 10
	const float meanFreePath = 1.0f;
	const float scale = 3.0f;  // threshold = 3
	EXPECT_FALSE(wf_restir_volume_spatial_valid(cur, neighbor, 2, 2, meanFreePath, scale));
}

TEST(WfRestirVolumeSpatialValid, DifferentMediumsAlwaysInvalidRegardlessOfDistance) {
	float3 cur = make_float3(0.0f, 0.0f, 0.0f);
	float3 neighbor = make_float3(0.0f, 0.0f, 0.0f);  // coincident - distance 0
	EXPECT_FALSE(wf_restir_volume_spatial_valid(cur, neighbor, /*curMatIdx=*/2, /*nMatIdx=*/3, 1.0f, 3.0f));
}

TEST(WfRestirVolumeSpatialValid, NegativeMatIdxSentinelIsInvalid) {
	float3 cur = make_float3(0.0f, 0.0f, 0.0f);
	float3 neighbor = make_float3(0.0f, 0.0f, 0.0f);
	EXPECT_FALSE(wf_restir_volume_spatial_valid(cur, neighbor, -1, 2, 1.0f, 3.0f));
	EXPECT_FALSE(wf_restir_volume_spatial_valid(cur, neighbor, 2, -1, 1.0f, 3.0f));
	EXPECT_FALSE(wf_restir_volume_spatial_valid(cur, neighbor, -1, -1, 1.0f, 3.0f));
}

TEST(WfRestirVolumeSpatialValid, DistanceExactlyAtThresholdIsValid) {
	float3 cur = make_float3(0.0f, 0.0f, 0.0f);
	float3 neighbor = make_float3(3.0f, 0.0f, 0.0f);  // distance exactly 3
	const float meanFreePath = 1.0f;
	const float scale = 3.0f;  // threshold = 3
	EXPECT_TRUE(wf_restir_volume_spatial_valid(cur, neighbor, 2, 2, meanFreePath, scale));
}
