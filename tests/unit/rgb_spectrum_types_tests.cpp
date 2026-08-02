// rgb_spectrum_types_tests.cpp
// Unit tests for RGBToSpectrumTable, RGBAlbedoSpectrum,
// RGBUnboundedSpectrum, and RGBIlluminantSpectrum.
//
// pbrt-v4 reference:
//   src/pbrt/util/color.h   -- RGBToSpectrumTable
//   src/pbrt/util/spectrum.h -- RGBAlbedoSpectrum, RGBUnboundedSpectrum,
//                               RGBIlluminantSpectrum

#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>

#include "../../src/shared/spectrum_types.h"
#include "../../src/shared/spectral_math.h"
#include "../../src/shared/cie_data.h"

static constexpr float kEps    = 1e-5f;
static constexpr float kRelTol = 0.01f;  // 1 % MC tolerance
static constexpr int kN = 4;
using SS  = SampledSpectrum<kN>;
using SWL = SampledWavelengths<kN>;
static const RGBColorSpace& kSRGB = RGBColorSpace::sRGB();

// ===========================================================================
// RGBToSpectrumTable
// ===========================================================================

TEST(RGBToSpectrumTable, AchromaticGrey_IsSpecialCase) {
	// For r==g==b the operator() uses the analytic formula:
	//   s(x) at x=0 => 0.5 => reflectance = 0.5 at all wavelengths
	auto rsp = RGBToSpectrumTable::sRGB()(0.5f, 0.5f, 0.5f);
	for (float lam = 400.f; lam <= 700.f; lam += 50.f)
		EXPECT_NEAR(rsp(lam), 0.5f, kEps) << "lam=" << lam;
}

TEST(RGBToSpectrumTable, AchromaticBlack) {
	auto rsp = RGBToSpectrumTable::sRGB()(0.f, 0.f, 0.f);
	// sigmoid(c2) near 0 -> rsp evaluates near 0 for all lambda
	for (float lam = 380.f; lam <= 720.f; lam += 20.f)
		EXPECT_NEAR(rsp(lam), 0.f, 1e-4f) << "lambda=" << lam;
}

TEST(RGBToSpectrumTable, AchromaticWhite) {
	auto rsp = RGBToSpectrumTable::sRGB()(1.f, 1.f, 1.f);
	// For pure white sigmoid polynomial should return ~1 for all visible lambda
	for (float lam = 400.f; lam <= 700.f; lam += 20.f)
		EXPECT_NEAR(rsp(lam), 1.f, 0.01f) << "lambda=" << lam;
}

TEST(RGBToSpectrumTable, OutputRange_IsNonNegative) {
	// Table output (sigmoid) must always be in [0,1]
	float testRGB[][3] = {
		{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f},
		{0.5f, 0.5f, 0.f}, {1.f, 0.5f, 0.25f}, {0.1f, 0.9f, 0.3f}
	};
	for (auto& rgb : testRGB) {
		auto rsp = RGBToSpectrumTable::sRGB()(rgb[0], rgb[1], rgb[2]);
		for (float lam = 360.f; lam <= 830.f; lam += 10.f) {
			float v = rsp(lam);
			EXPECT_GE(v, 0.f) << "rgb=(" << rgb[0] << "," << rgb[1] << "," << rgb[2] << ") lam=" << lam;
			EXPECT_LE(v, 1.f + kEps) << "rgb=(" << rgb[0] << "," << rgb[1] << "," << rgb[2] << ") lam=" << lam;
		}
	}
}

TEST(RGBToSpectrumTable, Singleton_SameReference) {
	EXPECT_EQ(&RGBToSpectrumTable::sRGB(), &RGBToSpectrumTable::sRGB());
}

// ===========================================================================
// RGBAlbedoSpectrum
// ===========================================================================

TEST(RGBAlbedoSpectrum, Black_IsZero) {
	RGBAlbedoSpectrum s(kSRGB, 0.f, 0.f, 0.f);
	for (float lam = 380.f; lam <= 720.f; lam += 20.f)
		EXPECT_NEAR(s(lam), 0.f, 1e-4f);
}

TEST(RGBAlbedoSpectrum, White_IsNearOne) {
	RGBAlbedoSpectrum s(kSRGB, 1.f, 1.f, 1.f);
	for (float lam = 400.f; lam <= 700.f; lam += 20.f)
		EXPECT_NEAR(s(lam), 1.f, 0.01f);
}

TEST(RGBAlbedoSpectrum, ValuesInUnitInterval) {
	RGBAlbedoSpectrum s(kSRGB, 0.8f, 0.2f, 0.1f);
	for (float lam = 360.f; lam <= 830.f; lam += 10.f) {
		float v = s(lam);
		EXPECT_GE(v, 0.f) << "lam=" << lam;
		EXPECT_LE(v, 1.f + kEps) << "lam=" << lam;
	}
}

TEST(RGBAlbedoSpectrum, MaxValue_IsInUnitInterval) {
	RGBAlbedoSpectrum s(kSRGB, 0.3f, 0.7f, 0.5f);
	EXPECT_GE(s.MaxValue(), 0.f);
	EXPECT_LE(s.MaxValue(), 1.f + kEps);
}

TEST(RGBAlbedoSpectrum, Sample_MatchesOperator) {
	RGBAlbedoSpectrum s(kSRGB, 0.6f, 0.3f, 0.9f);
	SWL swl = SWL::SampleUniform(0.4f);
	SS ss = s.Sample(swl);
	for (int i = 0; i < kN; ++i)
		EXPECT_NEAR(ss[i], s(swl.lambda[i]), kEps) << "band " << i;
}

TEST(RGBAlbedoSpectrum, Clamps_OutOfRange_Input) {
	// Values outside [0,1] should be clamped
	RGBAlbedoSpectrum s_clamped(kSRGB, 1.5f, -0.1f, 0.5f);
	RGBAlbedoSpectrum s_ref(kSRGB, 1.0f, 0.0f, 0.5f);
	for (float lam = 400.f; lam <= 700.f; lam += 50.f)
		EXPECT_NEAR(s_clamped(lam), s_ref(lam), kEps) << "lam=" << lam;
}

// ===========================================================================
// RGBUnboundedSpectrum
// ===========================================================================

TEST(RGBUnboundedSpectrum, DefaultCtor_IsZero) {
	RGBUnboundedSpectrum s;
	EXPECT_EQ(s.MaxValue(), 0.f);
	EXPECT_NEAR(s(550.f), 0.f, kEps);
}

TEST(RGBUnboundedSpectrum, Scale_IsTwiceMax) {
	// scale = 2 * max(r,g,b)
	RGBUnboundedSpectrum s(kSRGB, 3.f, 1.f, 0.5f);
	EXPECT_NEAR(s.scale, 2.f * 3.f, kEps);
}

TEST(RGBUnboundedSpectrum, NormalisedEquivalence) {
	// RGBUnboundedSpectrum(rgb) / scale should equal RGBAlbedoSpectrum(rgb/scale)
	float r = 2.f, g = 0.4f, b = 0.8f;
	float m = std::max({r, g, b});
	float sc = 2.f * m;
	RGBUnboundedSpectrum su(kSRGB, r, g, b);
	RGBAlbedoSpectrum sa(kSRGB, r/sc, g/sc, b/sc);
	for (float lam = 400.f; lam <= 700.f; lam += 50.f)
		EXPECT_NEAR(su(lam) / sc, sa(lam), 1e-4f) << "lam=" << lam;
}

TEST(RGBUnboundedSpectrum, Sample_MatchesOperator) {
	RGBUnboundedSpectrum s(kSRGB, 1.5f, 0.5f, 0.25f);
	SWL swl = SWL::SampleVisible(0.6f);
	SS ss = s.Sample(swl);
	for (int i = 0; i < kN; ++i)
		EXPECT_NEAR(ss[i], s(swl.lambda[i]), kEps) << "band " << i;
}

TEST(RGBUnboundedSpectrum, MaxValue_ExceedsOne_ForHDR) {
	RGBUnboundedSpectrum s(kSRGB, 5.f, 2.f, 1.f);
	EXPECT_GT(s.MaxValue(), 1.f);
}

// ===========================================================================
// RGBIlluminantSpectrum
// ===========================================================================

TEST(RGBIlluminantSpectrum, DefaultCtor_IsZero) {
	RGBIlluminantSpectrum s;
	EXPECT_EQ(s.MaxValue(), 0.f);
	EXPECT_NEAR(s(550.f), 0.f, kEps);
}

TEST(RGBIlluminantSpectrum, NullIlluminant_IsZero) {
	RGBIlluminantSpectrum s(kSRGB, 1.f, 1.f, 1.f, nullptr);
	EXPECT_EQ(s.MaxValue(), 0.f);
	for (float lam = 400.f; lam <= 700.f; lam += 50.f)
		EXPECT_NEAR(s(lam), 0.f, kEps) << "lam=" << lam;
}

TEST(RGBIlluminantSpectrum, WithIlluminant_NonZero) {
	const DenselySampledSpectrum& illum = GetCIE_Y();
	RGBIlluminantSpectrum s(kSRGB, 1.f, 0.5f, 0.25f, &illum);
	// Should be nonzero in the visible range
	float sum = 0.f;
	for (float lam = 400.f; lam <= 700.f; lam += 20.f)
		sum += s(lam);
	EXPECT_GT(sum, 0.f);
}

TEST(RGBIlluminantSpectrum, Sample_MatchesOperator) {
	const DenselySampledSpectrum& illum = GetCIE_Y();
	RGBIlluminantSpectrum s(kSRGB, 0.8f, 0.4f, 0.2f, &illum);
	SWL swl = SWL::SampleVisible(0.3f);
	SS ss = s.Sample(swl);
	for (int i = 0; i < kN; ++i)
		EXPECT_NEAR(ss[i], s(swl.lambda[i]), kEps) << "band " << i;
}

TEST(RGBIlluminantSpectrum, Scale_IsTwiceMax) {
	const DenselySampledSpectrum& illum = GetCIE_Y();
	RGBIlluminantSpectrum s(kSRGB, 2.f, 1.f, 0.5f, &illum);
	EXPECT_NEAR(s.scale, 2.f * 2.f, kEps);
}

TEST(RGBIlluminantSpectrum, Illuminant_Accessor) {
	const DenselySampledSpectrum& illum = GetCIE_Y();
	RGBIlluminantSpectrum s(kSRGB, 0.5f, 0.5f, 0.5f, &illum);
	EXPECT_EQ(s.Illuminant(), &illum);
}

TEST(RGBIlluminantSpectrum, NullSample_IsZero) {
	RGBIlluminantSpectrum s(kSRGB, 1.f, 0.5f, 0.3f, nullptr);
	SWL swl = SWL::SampleVisible(0.5f);
	SS ss = s.Sample(swl);
	for (int i = 0; i < kN; ++i)
		EXPECT_NEAR(ss[i], 0.f, kEps) << "band " << i;
}

// ===========================================================================
// Integration: RGBAlbedoSpectrum -> ToXYZ roundtrip
// ===========================================================================

TEST(RGBSpectraIntegration, AlbedoWhite_ToXYZ_YNearOne) {
	// White spectrum should have Y ≈ 1 under many MC samples
	const int kMC = 400;
	float ySum = 0.f;
	RGBAlbedoSpectrum s(kSRGB, 1.f, 1.f, 1.f);
	for (int i = 0; i < kMC; ++i) {
		float u = (i + 0.5f) / float(kMC);
		SWL swl = SWL::SampleUniform(u);
		SS ss = s.Sample(swl);
		ySum += LuminanceY(ss, swl);
	}
	EXPECT_NEAR(ySum / float(kMC), 1.f, 0.02f);
}

TEST(RGBSpectraIntegration, AlbedoBlack_ToXYZ_IsZero) {
	RGBAlbedoSpectrum s(kSRGB, 0.f, 0.f, 0.f);
	SWL swl = SWL::SampleUniform(0.5f);
	SS ss = s.Sample(swl);
	XYZ xyz = ToXYZ(ss, swl);
	EXPECT_NEAR(xyz.X, 0.f, kEps);
	EXPECT_NEAR(xyz.Y, 0.f, kEps);
	EXPECT_NEAR(xyz.Z, 0.f, kEps);
}

TEST(RGBSpectraIntegration, UnboundedSpectrum_FiniteOutputs) {
	RGBUnboundedSpectrum s(kSRGB, 4.f, 2.f, 1.f);
	SWL swl = SWL::SampleVisible(0.7f);
	SS ss = s.Sample(swl);
	for (int i = 0; i < kN; ++i) {
		EXPECT_TRUE(std::isfinite(ss[i])) << "band " << i;
		EXPECT_GE(ss[i], 0.f) << "band " << i;
	}
}
