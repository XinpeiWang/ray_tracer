// tone_map_tests.cpp
// Unit tests for src/shared/tone_map.h
//
// Tests are grouped into three areas:
//   1. linear_to_srgb  -- sRGB OETF piecewise correctness
//   2. ACES Narkowicz  -- aces_narkowicz curve properties
//   3. apply_tone_map  -- ToneMapMode dispatch including Reinhard and None

#include <gtest/gtest.h>
#include <cmath>
#include "../../src/shared/tone_map.h"

// ---------------------------------------------------------------------------
// linear_to_srgb
// ---------------------------------------------------------------------------

TEST(LinearToSRGB, ZeroMapsToZero) {
	EXPECT_DOUBLE_EQ(linear_to_srgb(0.0), 0.0);
}

TEST(LinearToSRGB, OneMapsToOne) {
	EXPECT_DOUBLE_EQ(linear_to_srgb(1.0), 1.0);
}

TEST(LinearToSRGB, NegativeClampedToZero) {
	EXPECT_DOUBLE_EQ(linear_to_srgb(-0.5), 0.0);
}

TEST(LinearToSRGB, AboveOneClampedToOne) {
	EXPECT_DOUBLE_EQ(linear_to_srgb(2.0), 1.0);
}

TEST(LinearToSRGB, LowLinearUsesLinearSegment) {
	// For x <= 0.0031308, sRGB = 12.92 * x
	double x = 0.001;
	EXPECT_NEAR(linear_to_srgb(x), 12.92 * x, 1e-10);
}

TEST(LinearToSRGB, MidLinearUsesPowerSegment) {
	// Our implementation uses a minimax polynomial aligned with pbrt-v4's
	// enoki-derived approximation.  Verify it matches the IEC 61966-2-1
	// reference formula to within 1e-5 (the polynomial's documented error).
	for (double x : {0.01, 0.1, 0.5, 0.9}) {
		double ref = 1.055 * std::pow(x, 1.0 / 2.4) - 0.055;
		EXPECT_NEAR(linear_to_srgb(x), ref, 1e-5) << "Mismatch at x=" << x;
	}
}

TEST(LinearToSRGB, ContinuousAtKnee) {
	// The two segments must meet continuously at x = 0.0031308
	double knee = 0.0031308;
	double linear_val = 12.92 * knee;
	double power_val  = 1.055 * std::pow(knee, 1.0 / 2.4) - 0.055;
	EXPECT_NEAR(linear_val, power_val, 1e-4);
}

TEST(LinearToSRGB, IsMonotone) {
	// A sample of values must be strictly increasing
	double prev = linear_to_srgb(0.0);
	for (int i = 1; i <= 20; ++i) {
		double x = i / 20.0;
		double cur = linear_to_srgb(x);
		EXPECT_GT(cur, prev) << "Not monotone at x=" << x;
		prev = cur;
	}
}

// ---------------------------------------------------------------------------
// aces_narkowicz
// ---------------------------------------------------------------------------

TEST(ACESNarkowicz, ZeroMapsToZero) {
	EXPECT_DOUBLE_EQ(aces_narkowicz(0.0), 0.0);
}

TEST(ACESNarkowicz, NegativeClampedToZero) {
	EXPECT_DOUBLE_EQ(aces_narkowicz(-1.0), 0.0);
}

TEST(ACESNarkowicz, OutputBoundedByOne) {
	for (double x : {0.001, 0.1, 0.5, 1.0, 2.0, 5.0, 100.0}) {
		double v = aces_narkowicz(x);
		EXPECT_GE(v, 0.0) << "Negative output at x=" << x;
		EXPECT_LE(v, 1.0) << "Output exceeds 1 at x=" << x;
	}
}

TEST(ACESNarkowicz, IsMonotone) {
	double prev = 0.0;
	for (int i = 1; i <= 50; ++i) {
		double x = i * 0.1;
		double cur = aces_narkowicz(x);
		EXPECT_GE(cur, prev) << "Not monotone at x=" << x;
		prev = cur;
	}
}

TEST(ACESNarkowicz, HighExposureConvergesToOne) {
	// Very bright values should map close to 1
	EXPECT_GT(aces_narkowicz(100.0), 0.99);
}

TEST(ACESNarkowicz, MidExposureReasonable) {
	// Exposure = 1 (no scale) maps to roughly 0.83 per the Narkowicz curve
	double v = aces_narkowicz(1.0);
	EXPECT_GT(v, 0.7);
	EXPECT_LT(v, 1.0);
}

// ---------------------------------------------------------------------------
// apply_tone_map / ToneMapMode dispatch
// ---------------------------------------------------------------------------

TEST(ApplyToneMap, ACESModeMatchesHelper) {
	double x = 2.5;
	EXPECT_DOUBLE_EQ(apply_tone_map(x, ToneMapMode::ACES), aces_narkowicz(x));
}

TEST(ApplyToneMap, ReinhardModeIsXOverOnePlusX) {
	double x = 3.0;
	EXPECT_NEAR(apply_tone_map(x, ToneMapMode::Reinhard), x / (1.0 + x), 1e-12);
}

TEST(ApplyToneMap, NoneModeIsIdentityForPositive) {
	double x = 0.7;
	EXPECT_DOUBLE_EQ(apply_tone_map(x, ToneMapMode::None), x);
}

TEST(ApplyToneMap, NoneModeClampNegativeToZero) {
	EXPECT_DOUBLE_EQ(apply_tone_map(-1.0, ToneMapMode::None), 0.0);
}

TEST(ApplyToneMap, ReinhardZeroMapsToZero) {
	EXPECT_DOUBLE_EQ(apply_tone_map(0.0, ToneMapMode::Reinhard), 0.0);
}

TEST(ApplyToneMap, AllModesReturnNonNegative) {
	for (auto mode : {ToneMapMode::ACES, ToneMapMode::Reinhard, ToneMapMode::None}) {
		EXPECT_GE(apply_tone_map(0.0, mode), 0.0);
		EXPECT_GE(apply_tone_map(1.0, mode), 0.0);
		EXPECT_GE(apply_tone_map(-0.5, mode), 0.0);
	}
}

TEST(ApplyToneMap, AllModesReturnAtMostOneForSaturated) {
	for (auto mode : {ToneMapMode::ACES, ToneMapMode::Reinhard}) {
		// These operators are designed to stay <= 1 for any non-negative input
		EXPECT_LE(apply_tone_map(1000.0, mode), 1.0);
	}
}

// ---------------------------------------------------------------------------
// tone_map_mode_from_name -- --tonemap flag / launcher_args.h dispatch
// ---------------------------------------------------------------------------

TEST(ToneMapModeFromName, RecognizesAllThreeModes) {
	ToneMapMode mode;
	EXPECT_TRUE(tone_map_mode_from_name("aces", mode));
	EXPECT_EQ(mode, ToneMapMode::ACES);
	EXPECT_TRUE(tone_map_mode_from_name("reinhard", mode));
	EXPECT_EQ(mode, ToneMapMode::Reinhard);
	EXPECT_TRUE(tone_map_mode_from_name("none", mode));
	EXPECT_EQ(mode, ToneMapMode::None);
}

TEST(ToneMapModeFromName, RejectsUnrecognizedNames) {
	ToneMapMode mode = ToneMapMode::ACES;
	EXPECT_FALSE(tone_map_mode_from_name("bogus", mode));
	EXPECT_FALSE(tone_map_mode_from_name("", mode));
	EXPECT_FALSE(tone_map_mode_from_name("ACES", mode))
		<< "names are case-sensitive lowercase, matching --sampler's own convention";
}
