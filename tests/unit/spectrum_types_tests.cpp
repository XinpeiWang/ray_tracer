// spectrum_types_tests.cpp
// Unit tests for spectrum_types.h
//
// Covers:
//   Blackbody()             -- free function, Planck radiance formula
//   ConstantSpectrum        -- construction, evaluation, Sample()
//   DenselySampledSpectrum  -- construction, SampleFunction, operator(), MaxValue, Scale, Sample(), equality
//   PiecewiseLinearSpectrum -- construction, FromInterleaved, operator() lerp, MaxValue, Scale, Sample()
//   BlackbodySpectrum       -- construction, normalization, operator(), MaxValue, Sample()
//
// pbrt-v4 reference: src/pbrt/util/spectrum.h

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "../../src/shared/spectrum_types.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static constexpr int kN = 4;
using SS  = SampledSpectrum<kN>;
using SWL = SampledWavelengths<kN>;

static constexpr float kEps = 1e-5f;
static constexpr float kRelEps = 1e-4f;  // for blackbody (uses exp)

// Blackbody() free function tests are in color_utils_tests.cpp.

// ---------------------------------------------------------------------------
// ConstantSpectrum
// ---------------------------------------------------------------------------
TEST(ConstantSpectrum, EvaluatesToConstant) {
	ConstantSpectrum cs(0.75f);
	EXPECT_NEAR(cs(400.f), 0.75f, kEps);
	EXPECT_NEAR(cs(550.f), 0.75f, kEps);
	EXPECT_NEAR(cs(700.f), 0.75f, kEps);
}

TEST(ConstantSpectrum, MaxValue) {
	ConstantSpectrum cs(1.5f);
	EXPECT_NEAR(cs.MaxValue(), 1.5f, kEps);
}

TEST(ConstantSpectrum, Zero) {
	ConstantSpectrum cs(0.f);
	EXPECT_EQ(cs(500.f), 0.f);
	EXPECT_EQ(cs.MaxValue(), 0.f);
}

TEST(ConstantSpectrum, SampleFillsAllBands) {
	ConstantSpectrum cs(0.5f);
	SWL swl = SWL::SampleUniform(0.3f);
	SS s = cs.Sample(swl);
	for (int i = 0; i < kN; ++i)
		EXPECT_NEAR(s[i], 0.5f, kEps) << "band " << i;
}

// ---------------------------------------------------------------------------
// DenselySampledSpectrum
// ---------------------------------------------------------------------------
TEST(DenselySampledSpectrum, DefaultConstructorIsZero) {
	DenselySampledSpectrum ds;
	EXPECT_EQ(ds(500.f), 0.f);
	EXPECT_EQ(ds(360.f), 0.f);
	EXPECT_EQ(ds(830.f), 0.f);
}

TEST(DenselySampledSpectrum, SampleFunctionStoresCallableResult) {
	// f(lambda) = lambda / 1000
	auto ds = DenselySampledSpectrum::SampleFunction(
		[](float l) { return l / 1000.f; });
	EXPECT_NEAR(ds(500.f), 0.5f, kEps);
	EXPECT_NEAR(ds(700.f), 0.7f, kEps);
}

TEST(DenselySampledSpectrum, OutOfRangeReturnsZero) {
	auto ds = DenselySampledSpectrum::SampleFunction(
		[](float l) { return 1.f; });
	EXPECT_EQ(ds(359.f), 0.f);
	EXPECT_EQ(ds(831.f), 0.f);
}

TEST(DenselySampledSpectrum, SnapsToNearestNm) {
	// Values are stored at integer nm; operator() snaps with lround
	DenselySampledSpectrum ds;
	ds.values[140] = 1.f;  // offset 140 = 360 + 140 = 500 nm
	EXPECT_NEAR(ds(500.4f), 1.f, kEps);   // rounds to 500
	EXPECT_NEAR(ds(499.6f), 1.f, kEps);   // rounds to 500
	EXPECT_EQ(ds(500.6f), 0.f);            // rounds to 501
}

TEST(DenselySampledSpectrum, MaxValue) {
	auto ds = DenselySampledSpectrum::SampleFunction(
		[](float l) { return l == 600.f ? 3.f : 1.f; });
	EXPECT_NEAR(ds.MaxValue(), 3.f, kEps);
}

TEST(DenselySampledSpectrum, Scale) {
	auto ds = DenselySampledSpectrum::SampleFunction(
		[](float l) { return 2.f; });
	ds.Scale(3.f);
	EXPECT_NEAR(ds(500.f), 6.f, kEps);
}

TEST(DenselySampledSpectrum, Equality) {
	auto a = DenselySampledSpectrum::SampleFunction([](float l) { return l; });
	auto b = DenselySampledSpectrum::SampleFunction([](float l) { return l; });
	auto c = DenselySampledSpectrum::SampleFunction([](float l) { return l * 2.f; });
	EXPECT_EQ(a, b);
	EXPECT_NE(a, c);
}

TEST(DenselySampledSpectrum, SampleMatchesOperator) {
	auto ds = DenselySampledSpectrum::SampleFunction(
		[](float l) { return l / 500.f; });
	SWL swl = SWL::SampleUniform(0.5f);
	SS s = ds.Sample(swl);
	for (int i = 0; i < kN; ++i) {
		float expected = ds(swl.lambda[i]);
		EXPECT_NEAR(s[i], expected, kEps) << "band " << i;
	}
}

TEST(DenselySampledSpectrum, CustomRange) {
	DenselySampledSpectrum ds(400, 700);
	EXPECT_EQ(static_cast<int>(ds.values.size()), 301);
	EXPECT_EQ(ds(399.f), 0.f);
	EXPECT_EQ(ds(701.f), 0.f);
}

// ---------------------------------------------------------------------------
// PiecewiseLinearSpectrum
// ---------------------------------------------------------------------------
TEST(PiecewiseLinearSpectrum, DefaultConstructorIsZero) {
	PiecewiseLinearSpectrum ps;
	EXPECT_EQ(ps(500.f), 0.f);
	EXPECT_EQ(ps.MaxValue(), 0.f);
}

TEST(PiecewiseLinearSpectrum, ExactKnots) {
	PiecewiseLinearSpectrum ps({400.f, 600.f, 800.f}, {0.f, 1.f, 0.5f});
	EXPECT_NEAR(ps(400.f), 0.f,  kEps);
	EXPECT_NEAR(ps(600.f), 1.f,  kEps);
	EXPECT_NEAR(ps(800.f), 0.5f, kEps);
}

TEST(PiecewiseLinearSpectrum, Interpolates) {
	PiecewiseLinearSpectrum ps({400.f, 600.f}, {0.f, 1.f});
	// Midpoint should be 0.5
	EXPECT_NEAR(ps(500.f), 0.5f, kEps);
	// Quarter point
	EXPECT_NEAR(ps(450.f), 0.25f, kEps);
}

TEST(PiecewiseLinearSpectrum, OutsideRangeIsZero) {
	PiecewiseLinearSpectrum ps({400.f, 600.f}, {1.f, 1.f});
	EXPECT_EQ(ps(399.f), 0.f);
	EXPECT_EQ(ps(601.f), 0.f);
}

TEST(PiecewiseLinearSpectrum, MaxValue) {
	PiecewiseLinearSpectrum ps({400.f, 500.f, 600.f}, {0.2f, 3.0f, 1.0f});
	EXPECT_NEAR(ps.MaxValue(), 3.0f, kEps);
}

TEST(PiecewiseLinearSpectrum, Scale) {
	PiecewiseLinearSpectrum ps({400.f, 600.f}, {1.f, 2.f});
	ps.Scale(2.f);
	EXPECT_NEAR(ps(400.f), 2.f, kEps);
	EXPECT_NEAR(ps(600.f), 4.f, kEps);
}

TEST(PiecewiseLinearSpectrum, FromInterleaved) {
	std::vector<float> data = {400.f, 0.f, 600.f, 1.f, 800.f, 0.5f};
	auto ps = PiecewiseLinearSpectrum::FromInterleaved(data);
	EXPECT_NEAR(ps(400.f), 0.f,  kEps);
	EXPECT_NEAR(ps(600.f), 1.f,  kEps);
	EXPECT_NEAR(ps(800.f), 0.5f, kEps);
	EXPECT_NEAR(ps(500.f), 0.5f, kEps);  // midpoint
}

TEST(PiecewiseLinearSpectrum, FromInterleavedNormalized) {
	std::vector<float> data = {400.f, 0.f, 600.f, 4.f, 800.f, 2.f};
	auto ps = PiecewiseLinearSpectrum::FromInterleaved(data, /*normalize=*/true);
	EXPECT_NEAR(ps(600.f), 1.f, kEps);   // max was 4, now 1
	EXPECT_NEAR(ps(800.f), 0.5f, kEps);  // 2/4 = 0.5
}

TEST(PiecewiseLinearSpectrum, SingleKnot) {
	// Single knot: only exact match returns the value
	PiecewiseLinearSpectrum ps({500.f}, {0.8f});
	EXPECT_NEAR(ps(500.f), 0.8f, kEps);
	EXPECT_EQ(ps(499.f), 0.f);
	EXPECT_EQ(ps(501.f), 0.f);
}

TEST(PiecewiseLinearSpectrum, SampleMatchesOperator) {
	PiecewiseLinearSpectrum ps({360.f, 550.f, 830.f}, {0.f, 1.f, 0.5f});
	SWL swl = SWL::SampleUniform(0.4f);
	SS s = ps.Sample(swl);
	for (int i = 0; i < kN; ++i) {
		float expected = ps(swl.lambda[i]);
		EXPECT_NEAR(s[i], expected, kEps) << "band " << i;
	}
}

// BlackbodySpectrum property tests (MaxValueIsOne, ValuesInUnitRange, etc.)
// are in color_utils_tests.cpp. Here we test only Sample<N>() which is
// unique to spectrum_types.h.

TEST(BlackbodySpectrum, SampleMatchesOperator) {
	BlackbodySpectrum bs(6500.f);
	SWL swl = SWL::SampleVisible(0.5f);
	SS s = bs.Sample(swl);
	for (int i = 0; i < kN; ++i) {
		float expected = bs(swl.lambda[i]);
		EXPECT_NEAR(s[i], expected, kRelEps) << "band " << i;
	}
}

// ---------------------------------------------------------------------------
// Cross-type integration: DenselySampled built from BlackbodySpectrum
// ---------------------------------------------------------------------------
TEST(SpectrumTypesIntegration, DenseFromBlackbody) {
	BlackbodySpectrum bs(5500.f);
	auto ds = DenselySampledSpectrum::SampleFunction(
		[&](float l) { return bs(l); });

	// DenselySampled should match BlackbodySpectrum at integer nm values
	for (int lambda = 360; lambda <= 830; lambda += 50) {
		float expected = bs(static_cast<float>(lambda));
		float actual   = ds(static_cast<float>(lambda));
		EXPECT_NEAR(actual, expected, kRelEps) << "lambda=" << lambda;
	}
}

TEST(SpectrumTypesIntegration, PiecewiseApproximatesConstant) {
	// A piecewise spectrum with equal values should match a constant
	PiecewiseLinearSpectrum ps({360.f, 830.f}, {0.6f, 0.6f});
	ConstantSpectrum cs(0.6f);
	for (int lambda = 360; lambda <= 830; lambda += 50) {
		EXPECT_NEAR(ps(static_cast<float>(lambda)),
					cs(static_cast<float>(lambda)), kEps);
	}
}
