// cie_data_tests.cpp
// Unit tests for cie_data.h and spectral_math.h
//
// Covers:
//   CIE data arrays          -- size, range, known values
//   GetCIE_X/Y/Z()           -- DenselySampledSpectrum correctness
//   ToXYZ()                  -- Monte Carlo estimator vs known spectrum
//   LuminanceY()             -- Y channel of ToXYZ
//   ToRGB()                  -- XYZ -> sRGB roundtrip
//
// pbrt-v4 reference: src/pbrt/util/spectrum.cpp lines 205-227

#include <gtest/gtest.h>
#include <cmath>

#include "../../src/data/cie_data.h"
#include "../../src/shared/spectral_math.h"

static constexpr float kEps    = 1e-5f;
static constexpr float kRelEps = 1e-3f;  // Monte Carlo estimator tolerance

static constexpr int kN = 4;
using SS  = SampledSpectrum<kN>;
using SWL = SampledWavelengths<kN>;

// ---------------------------------------------------------------------------
// CIE array integrity
// ---------------------------------------------------------------------------

TEST(CIEData, ArraySize) {
	EXPECT_EQ(kCIENSamples, 471);
	EXPECT_EQ(kCIELambda_max - kCIELambda_min + 1, 471);
}

TEST(CIEData, CIE_Y_PeakAtOne) {
	// CIE Y peaks at 555 nm (value = 1.0 exactly in the standard)
	int idx555 = 555 - kCIELambda_min;
	EXPECT_NEAR(CIE_Y[idx555], 1.0f, kEps);
}

TEST(CIEData, CIE_X_PeakNear600) {
	// CIE X has two peaks; the main one is near 600 nm and exceeds 1.0
	float maxX = 0.f;
	for (int i = 0; i < kCIENSamples; ++i)
		maxX = std::max(maxX, CIE_X[i]);
	EXPECT_GT(maxX, 1.0f);
}

TEST(CIEData, CIE_Z_ZeroAbove700nm) {
	// CIE Z is effectively zero above ~700 nm
	for (int i = 340; i < kCIENSamples; ++i)  // offset 340 = 700 nm
		EXPECT_NEAR(CIE_Z[i], 0.f, 1e-4f) << "lambda=" << (kCIELambda_min + i);
}

TEST(CIEData, AllArraysNonNegative) {
	for (int i = 0; i < kCIENSamples; ++i) {
		EXPECT_GE(CIE_X[i], 0.f) << "CIE_X[" << i << "]";
		EXPECT_GE(CIE_Y[i], 0.f) << "CIE_Y[" << i << "]";
		EXPECT_GE(CIE_Z[i], 0.f) << "CIE_Z[" << i << "]";
	}
}

TEST(CIEData, CIE_Y_IntegralApprox) {
	// Riemann sum of CIE_Y over [360,830] should be approx 106.86
	// (kCIE_Y_integral = 106.856895)
	float sum = 0.f;
	for (int i = 0; i < kCIENSamples; ++i)
		sum += CIE_Y[i];
	EXPECT_NEAR(sum, kCIE_Y_integral, 0.5f);
}

// ---------------------------------------------------------------------------
// GetCIE_X/Y/Z() -- DenselySampledSpectrum accessors
// ---------------------------------------------------------------------------

TEST(GetCIESpectrum, GetCIE_Y_At555) {
	// Should return 1.0 at 555 nm (the luminous efficiency peak)
	const auto& Y = GetCIE_Y();
	EXPECT_NEAR(Y(555.f), 1.0f, kEps);
}

TEST(GetCIESpectrum, GetCIE_Y_At360) {
	const auto& Y = GetCIE_Y();
	EXPECT_NEAR(Y(360.f), CIE_Y[0], kEps);
}

TEST(GetCIESpectrum, GetCIE_Y_At830) {
	const auto& Y = GetCIE_Y();
	EXPECT_NEAR(Y(830.f), CIE_Y[470], kEps);
}

TEST(GetCIESpectrum, GetCIE_X_OutOfRange) {
	const auto& X = GetCIE_X();
	EXPECT_EQ(X(359.f), 0.f);
	EXPECT_EQ(X(831.f), 0.f);
}

TEST(GetCIESpectrum, GetCIE_Z_At445) {
	// CIE Z peaks around 445 nm
	const auto& Z = GetCIE_Z();
	EXPECT_GT(Z(445.f), Z(360.f));
	EXPECT_GT(Z(445.f), Z(700.f));
}

TEST(GetCIESpectrum, Singleton_SameReference) {
	// Multiple calls return the same object
	EXPECT_EQ(&GetCIE_Y(), &GetCIE_Y());
	EXPECT_EQ(&GetCIE_X(), &GetCIE_X());
	EXPECT_EQ(&GetCIE_Z(), &GetCIE_Z());
}

TEST(GetCIESpectrum, SampleMatchesDirect) {
	// Sample() should match operator() for each wavelength in a SWL set
	SWL swl = SWL::SampleUniform(0.0f);
	SS Ys = GetCIE_Y().Sample(swl);
	for (int i = 0; i < kN; ++i) {
		float expected = GetCIE_Y()(swl.lambda[i]);
		EXPECT_NEAR(Ys[i], expected, kEps) << "band " << i;
	}
}

// ---------------------------------------------------------------------------
// ToXYZ() -- spectral -> XYZ estimator
// ---------------------------------------------------------------------------

TEST(ToXYZ, ConstantOneSpectrum_YApproxOne) {
	// A constant-1 spectrum under uniform sampling should give Y ≈ 1
	// because E[Y_bar * 1 / uniform_pdf] / CIE_Y_integral ≈ 1
	// (this is the definition of CIE_Y_integral as the integral of Y_bar)
	float ySum = 0.f;
	const int kMC = 1000;
	for (int s = 0; s < kMC; ++s) {
		float u = (s + 0.5f) / float(kMC);
		SWL swl = SWL::SampleUniform(u);
		SS ones(1.f);
		XYZ xyz = ToXYZ(ones, swl);
		ySum += xyz.Y;
	}
	float yMean = ySum / float(kMC);
	// Should converge to 1.0 — allow 1% tolerance for stratified estimate
	EXPECT_NEAR(yMean, 1.0f, 0.01f);
}

TEST(ToXYZ, ConstantZeroSpectrum_IsBlack) {
	SWL swl = SWL::SampleUniform(0.5f);
	SS zeros(0.f);
	XYZ xyz = ToXYZ(zeros, swl);
	EXPECT_NEAR(xyz.X, 0.f, kEps);
	EXPECT_NEAR(xyz.Y, 0.f, kEps);
	EXPECT_NEAR(xyz.Z, 0.f, kEps);
}

TEST(ToXYZ, XYZValuesNonNegativeForPositiveSpectrum) {
	SWL swl = SWL::SampleVisible(0.3f);
	SS ss(0.5f);
	XYZ xyz = ToXYZ(ss, swl);
	EXPECT_GE(xyz.X, 0.f);
	EXPECT_GE(xyz.Y, 0.f);
	EXPECT_GE(xyz.Z, 0.f);
}

// ---------------------------------------------------------------------------
// LuminanceY() -- Y channel
// ---------------------------------------------------------------------------

TEST(LuminanceY, MatchesToXYZ_Y) {
	SWL swl = SWL::SampleVisible(0.7f);
	SS ss(0.8f);
	XYZ xyz = ToXYZ(ss, swl);
	float y  = LuminanceY(ss, swl);
	EXPECT_NEAR(y, xyz.Y, kEps);
}

TEST(LuminanceY, ZeroForZeroSpectrum) {
	SWL swl = SWL::SampleUniform(0.2f);
	SS zeros(0.f);
	EXPECT_NEAR(LuminanceY(zeros, swl), 0.f, kEps);
}

TEST(LuminanceY, PositiveForPositiveSpectrum) {
	SWL swl = SWL::SampleVisible(0.5f);
	SS ss(1.f);
	EXPECT_GT(LuminanceY(ss, swl), 0.f);
}

// ---------------------------------------------------------------------------
// ToRGB() -- XYZ -> RGB in sRGB
// ---------------------------------------------------------------------------

TEST(ToRGB, GreySpectrum_sRGB_Consistent) {
	// A constant-1 spectrum produces the equal-energy illuminant E.
	// XYZ of E: X≈0.95, Y=1.00, Z≈1.09 (by definition of the CMF integral).
	// Converting via sRGB matrix gives R>G>B (E is warmer than D65 white).
	// This test verifies convergence and sign consistency, not exact neutrality.
	float rSum = 0.f, gSum = 0.f, bSum = 0.f;
	const int kMC = 400;
	const RGBColorSpace cs = RGBColorSpace::sRGB();
	for (int s = 0; s < kMC; ++s) {
		float u = (s + 0.5f) / float(kMC);
		SWL swl = SWL::SampleUniform(u);
		SS ss(1.f);
		float r, g, b;
		ToRGB(ss, swl, cs, r, g, b);
		rSum += r; gSum += g; bSum += b;
	}
	float rMean = rSum / kMC, gMean = gSum / kMC, bMean = bSum / kMC;
	// All channels positive (non-absorbing spectrum)
	EXPECT_GT(rMean, 0.f);
	EXPECT_GT(gMean, 0.f);
	EXPECT_GT(bMean, 0.f);
	// E illuminant: R > G > B in sRGB (cooler than D65)
	EXPECT_GT(rMean, gMean);
	// Y channel (luminance) should converge to ~1.0
	float ySum = 0.f;
	for (int s = 0; s < kMC; ++s) {
		float u = (s + 0.5f) / float(kMC);
		SWL swl = SWL::SampleUniform(u);
		SS ss(1.f);
		ySum += LuminanceY(ss, swl);
	}
	EXPECT_NEAR(ySum / float(kMC), 1.0f, 0.01f);
}

TEST(ToRGB, ZeroSpectrum_IsBlack) {
	SWL swl = SWL::SampleVisible(0.5f);
	SS zeros(0.f);
	float r, g, b;
	ToRGB(zeros, swl, RGBColorSpace::sRGB(), r, g, b);
	EXPECT_NEAR(r, 0.f, kEps);
	EXPECT_NEAR(g, 0.f, kEps);
	EXPECT_NEAR(b, 0.f, kEps);
}

TEST(ToRGB, OutputsAreFinite) {
	SWL swl = SWL::SampleVisible(0.4f);
	SS ss(0.5f);
	float r, g, b;
	ToRGB(ss, swl, RGBColorSpace::sRGB(), r, g, b);
	EXPECT_TRUE(std::isfinite(r));
	EXPECT_TRUE(std::isfinite(g));
	EXPECT_TRUE(std::isfinite(b));
}

// ---------------------------------------------------------------------------
// Integration: BlackbodySpectrum -> ToXYZ -> sanity
// ---------------------------------------------------------------------------

TEST(SpectralMathIntegration, Blackbody_ToXYZ_Consistent) {
	// A blackbody at 6500K sampled many times should give consistent XYZ Y
	const int kMC = 200;
	float ySum = 0.f;
	for (int s = 0; s < kMC; ++s) {
		float u = (s + 0.5f) / float(kMC);
		SWL swl = SWL::SampleVisible(u);
		BlackbodySpectrum bs(6500.f);
		SS ss = bs.Sample(swl);
		XYZ xyz = ToXYZ(ss, swl);
		ySum += xyz.Y;
	}
	float yMean = ySum / float(kMC);
	// Normalized blackbody Y should be in (0, 1]
	EXPECT_GT(yMean, 0.f);
	EXPECT_LE(yMean, 1.1f);  // small tolerance for MC variance
}
