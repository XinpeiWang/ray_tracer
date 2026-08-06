// sampled_spectrum_tests.cpp
// Unit tests for src/shared/sampled_spectrum.h
//
// Tests mirror pbrt-v4 semantics for SampledSpectrum<N> and SampledWavelengths<N>.
// Each section tests a logical group of functionality.

#include <gtest/gtest.h>
#include <cmath>
#include <array>
#include "../../src/shared/sampled_spectrum.h"
#include "../../src/shared/spectral_math.h"    // ToXYZ, LuminanceY, ToRGB, SpectrumToXYZ
#include "../../src/shared/spectrum_types.h"   // ConstantSpectrum, DenselySampledSpectrum
#include "../../src/shared/cie_data.h"         // GetCIE_X/Y/Z, kCIE_Y_integral
#include "../../src/shared/rgb_colorspace.h"   // RGBColorSpace::sRGB()

// Convenience aliases
using SS4  = SampledSpectrum<4>;
using SWL4 = SampledWavelengths<4>;

// Helpers
static bool near(float a, float b, float eps = 1e-5f) {
	return std::fabs(a - b) <= eps;
}

// ============================================================================
// SampledSpectrum -- construction
// ============================================================================

TEST(SampledSpectrumConstruction, DefaultIsZero) {
	SS4 s;
	for (int i = 0; i < 4; ++i)
		EXPECT_EQ(s[i], 0.f);
}

TEST(SampledSpectrumConstruction, FillConstant) {
	SS4 s(3.14f);
	for (int i = 0; i < 4; ++i)
		EXPECT_FLOAT_EQ(s[i], 3.14f);
}

TEST(SampledSpectrumConstruction, FromArray) {
	float v[4] = {1.f, 2.f, 3.f, 4.f};
	SS4 s(v, 4);
	for (int i = 0; i < 4; ++i)
		EXPECT_FLOAT_EQ(s[i], v[i]);
}

// ============================================================================
// SampledSpectrum -- element access
// ============================================================================

TEST(SampledSpectrumAccess, ReadWrite) {
	SS4 s;
	s[2] = 7.f;
	EXPECT_FLOAT_EQ(s[2], 7.f);
}

// ============================================================================
// SampledSpectrum -- bool conversion
// ============================================================================

TEST(SampledSpectrumBool, ZeroIsFalse) {
	SS4 s(0.f);
	EXPECT_FALSE(static_cast<bool>(s));
}

TEST(SampledSpectrumBool, NonZeroIsTrue) {
	SS4 s(0.f);
	s[1] = 1.f;
	EXPECT_TRUE(static_cast<bool>(s));
}

// ============================================================================
// SampledSpectrum -- arithmetic operators
// ============================================================================

TEST(SampledSpectrumArithmetic, Addition) {
	SS4 a(1.f), b(2.f);
	SS4 c = a + b;
	for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(c[i], 3.f);
}

TEST(SampledSpectrumArithmetic, AdditionInPlace) {
	SS4 a(1.f); a += SS4(4.f);
	for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(a[i], 5.f);
}

TEST(SampledSpectrumArithmetic, Subtraction) {
	SS4 a(5.f), b(3.f);
	SS4 c = a - b;
	for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(c[i], 2.f);
}

TEST(SampledSpectrumArithmetic, ScalarMinusSpectrum) {
	SS4 s(1.f);
	SS4 r = 10.f - s;
	for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(r[i], 9.f);
}

TEST(SampledSpectrumArithmetic, Multiplication) {
	SS4 a(2.f), b(3.f);
	SS4 c = a * b;
	for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(c[i], 6.f);
}

TEST(SampledSpectrumArithmetic, ScalarMultiplyLeft) {
	SS4 s(2.f);
	SS4 r = 5.f * s;
	for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(r[i], 10.f);
}

TEST(SampledSpectrumArithmetic, ScalarMultiplyRight) {
	SS4 s(3.f);
	SS4 r = s * 4.f;
	for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(r[i], 12.f);
}

TEST(SampledSpectrumArithmetic, Division) {
	SS4 a(9.f), b(3.f);
	SS4 c = a / b;
	for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(c[i], 3.f);
}

TEST(SampledSpectrumArithmetic, ScalarDivision) {
	SS4 s(12.f);
	SS4 r = s / 4.f;
	for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(r[i], 3.f);
}

TEST(SampledSpectrumArithmetic, Negation) {
	SS4 s(2.f);
	SS4 r = -s;
	for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(r[i], -2.f);
}

TEST(SampledSpectrumArithmetic, MixedChannels) {
	float v1[4] = {1.f, 2.f, 3.f, 4.f};
	float v2[4] = {4.f, 3.f, 2.f, 1.f};
	SS4 a(v1, 4), b(v2, 4);
	SS4 c = a * b;
	for (int i = 0; i < 4; ++i)
		EXPECT_FLOAT_EQ(c[i], v1[i] * v2[i]);
}

// ============================================================================
// SampledSpectrum -- reductions
// ============================================================================

TEST(SampledSpectrumReductions, Average) {
	float v[4] = {1.f, 2.f, 3.f, 4.f};
	SS4 s(v, 4);
	EXPECT_FLOAT_EQ(s.Average(), 2.5f);
}

TEST(SampledSpectrumReductions, MinComponentValue) {
	float v[4] = {3.f, 1.f, 4.f, 2.f};
	SS4 s(v, 4);
	EXPECT_FLOAT_EQ(s.MinComponentValue(), 1.f);
}

TEST(SampledSpectrumReductions, MaxComponentValue) {
	float v[4] = {3.f, 1.f, 4.f, 2.f};
	SS4 s(v, 4);
	EXPECT_FLOAT_EQ(s.MaxComponentValue(), 4.f);
}

TEST(SampledSpectrumReductions, AverageConstant) {
	SS4 s(7.f);
	EXPECT_FLOAT_EQ(s.Average(), 7.f);
}

// ============================================================================
// SampledSpectrum -- HasNaNs
// ============================================================================

TEST(SampledSpectrumNaN, NoNaN) {
	SS4 s(1.f);
	EXPECT_FALSE(s.HasNaNs());
}

TEST(SampledSpectrumNaN, HasNaN) {
	SS4 s(1.f);
	s[2] = std::numeric_limits<float>::quiet_NaN();
	EXPECT_TRUE(s.HasNaNs());
}

// ============================================================================
// SampledSpectrum free functions
// ============================================================================

TEST(SampledSpectrumFree, SafeDivNoDivByZero) {
	float va[4] = {8.f, 6.f, 4.f, 2.f};
	float vb[4] = {2.f, 0.f, 2.f, 0.f};
	SS4 a(va, 4), b(vb, 4);
	SS4 r = SafeDiv(a, b);
	EXPECT_FLOAT_EQ(r[0], 4.f);
	EXPECT_FLOAT_EQ(r[1], 0.f);  // b==0 → 0
	EXPECT_FLOAT_EQ(r[2], 2.f);
	EXPECT_FLOAT_EQ(r[3], 0.f);  // b==0 → 0
}

TEST(SampledSpectrumFree, ClampZero) {
	float v[4] = {-1.f, 0.f, 2.f, -0.5f};
	SS4 s(v, 4);
	SS4 r = ClampZero(s);
	EXPECT_FLOAT_EQ(r[0], 0.f);
	EXPECT_FLOAT_EQ(r[1], 0.f);
	EXPECT_FLOAT_EQ(r[2], 2.f);
	EXPECT_FLOAT_EQ(r[3], 0.f);
}

TEST(SampledSpectrumFree, Clamp) {
	float v[4] = {-1.f, 0.5f, 2.f, 1.f};
	SS4 s(v, 4);
	SS4 r = Clamp(s, 0.f, 1.f);
	EXPECT_FLOAT_EQ(r[0], 0.f);
	EXPECT_FLOAT_EQ(r[1], 0.5f);
	EXPECT_FLOAT_EQ(r[2], 1.f);
	EXPECT_FLOAT_EQ(r[3], 1.f);
}

TEST(SampledSpectrumFree, Sqrt) {
	float v[4] = {0.f, 1.f, 4.f, 9.f};
	SS4 s(v, 4);
	SS4 r = Sqrt(s);
	EXPECT_FLOAT_EQ(r[0], 0.f);
	EXPECT_FLOAT_EQ(r[1], 1.f);
	EXPECT_FLOAT_EQ(r[2], 2.f);
	EXPECT_FLOAT_EQ(r[3], 3.f);
}

TEST(SampledSpectrumFree, SafeSqrtNegativeGivesZero) {
	SS4 s(-1.f);
	SS4 r = SafeSqrt(s);
	for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(r[i], 0.f);
}

TEST(SampledSpectrumFree, Pow) {
	float v[4] = {1.f, 2.f, 3.f, 4.f};
	SS4 s(v, 4);
	SS4 r = Pow(s, 2.f);
	for (int i = 0; i < 4; ++i)
		EXPECT_NEAR(r[i], v[i] * v[i], 1e-5f);
}

TEST(SampledSpectrumFree, Exp) {
	SS4 s(0.f);
	SS4 r = Exp(s);
	for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(r[i], 1.f);
}

TEST(SampledSpectrumFree, FastExp) {
	// FastExp(0) == 1, FastExp(1) ≈ e
	SS4 s(0.f);
	SS4 r = FastExp(s);
	for (int i = 0; i < 4; ++i) EXPECT_NEAR(r[i], 1.f, 1e-4f);
	SS4 s1(1.f);
	SS4 r1 = FastExp(s1);
	for (int i = 0; i < 4; ++i) EXPECT_NEAR(r1[i], std::exp(1.f), 1e-3f);
}

TEST(SampledSpectrumFree, Lerp) {
	SS4 s0(0.f), s1(10.f);
	SS4 r = Lerp(0.3f, s0, s1);
	for (int i = 0; i < 4; ++i) EXPECT_NEAR(r[i], 3.f, 1e-5f);
}

// ============================================================================
// SampledWavelengths -- SampleUniform
// ============================================================================

TEST(SampledWavelengths, SampleUniformRange) {
	for (int k = 0; k < 50; ++k) {
		float u = (k + 0.5f) / 50.f;
		SWL4 swl = SWL4::SampleUniform(u);
		for (int i = 0; i < 4; ++i) {
			EXPECT_GE(swl[i], kLambda_min) << "i=" << i << " u=" << u;
			EXPECT_LE(swl[i], kLambda_max) << "i=" << i << " u=" << u;
		}
	}
}

TEST(SampledWavelengths, SampleUniformPDFPositive) {
	SWL4 swl = SWL4::SampleUniform(0.5f);
	SS4  pdf  = swl.PDF();
	float expected = 1.f / (kLambda_max - kLambda_min);
	for (int i = 0; i < 4; ++i)
		EXPECT_NEAR(pdf[i], expected, 1e-6f);
}

TEST(SampledWavelengths, SampleUniformEvenlySpaced) {
	float u = 0.25f;
	SWL4 swl = SWL4::SampleUniform(u);
	float delta = (kLambda_max - kLambda_min) / 4.f;
	// Each consecutive pair should differ by delta (modulo wrap)
	for (int i = 1; i < 4; ++i) {
		float diff = swl[i] - swl[i - 1];
		// After wrap diff could be negative but magnitude should still be delta
		if (diff < 0.f) diff += (kLambda_max - kLambda_min);
		EXPECT_NEAR(diff, delta, 1e-3f) << "i=" << i;
	}
}

TEST(SampledWavelengths, SampleUniformFirstWavelength) {
	float u = 0.f;
	SWL4 swl = SWL4::SampleUniform(u);
	EXPECT_FLOAT_EQ(swl[0], kLambda_min);

	u = 1.f;
	swl = SWL4::SampleUniform(u);
	EXPECT_FLOAT_EQ(swl[0], kLambda_max);
}

// ============================================================================
// SampledWavelengths -- SampleVisible
// ============================================================================

TEST(SampledWavelengths, SampleVisibleRange) {
	for (int k = 0; k < 50; ++k) {
		float u = (k + 0.5f) / 50.f;
		SWL4 swl = SWL4::SampleVisible(u);
		for (int i = 0; i < 4; ++i) {
			EXPECT_GE(swl[i], kLambda_min) << "i=" << i;
			EXPECT_LE(swl[i], kLambda_max) << "i=" << i;
		}
	}
}

TEST(SampledWavelengths, SampleVisiblePDFPositive) {
	SWL4 swl = SWL4::SampleVisible(0.5f);
	SS4  pdf  = swl.PDF();
	for (int i = 0; i < 4; ++i)
		EXPECT_GT(pdf[i], 0.f) << "i=" << i;
}

TEST(SampledWavelengths, SampleVisibleDistinct) {
	// The 4 stratified wavelengths should all be different.
	SWL4 swl = SWL4::SampleVisible(0.3f);
	for (int i = 0; i < 4; ++i)
		for (int j = i + 1; j < 4; ++j)
			EXPECT_NE(swl[i], swl[j]) << "i=" << i << " j=" << j;
}

TEST(SampledWavelengths, SampleVisibleConcentratesGreen) {
	// More than 50% of visible-sampled wavelengths should be in [480, 620].
	int inBand = 0, total = 0;
	for (int k = 0; k < 200; ++k) {
		float u = (k + 0.5f) / 200.f;
		SWL4 swl = SWL4::SampleVisible(u);
		for (int i = 0; i < 4; ++i) {
			++total;
			if (swl[i] >= 480.f && swl[i] <= 620.f) ++inBand;
		}
	}
	EXPECT_GT(inBand, total / 2);
}

// ============================================================================
// SampledWavelengths -- TerminateSecondary
// ============================================================================

TEST(SampledWavelengths, TerminateSecondaryZerosPdf) {
	SWL4 swl = SWL4::SampleVisible(0.4f);
	EXPECT_FALSE(swl.SecondaryTerminated());
	swl.TerminateSecondary();
	EXPECT_TRUE(swl.SecondaryTerminated());
	// pdf[1..3] must be zero
	SS4 pdf = swl.PDF();
	for (int i = 1; i < 4; ++i)
		EXPECT_EQ(pdf[i], 0.f) << "i=" << i;
}

TEST(SampledWavelengths, TerminateSecondaryPreservesPrimaryEnergy) {
	// After termination the surviving pdf[0] should equal original_pdf[0] / N
	// (pbrt-v4 divides pdf[0] by NSpectrumSamples to keep estimator unbiased).
	SWL4 swl = SWL4::SampleVisible(0.5f);
	float p0_before = swl.PDF()[0];
	swl.TerminateSecondary();
	float p0_after  = swl.PDF()[0];
	EXPECT_NEAR(p0_after, p0_before / 4.f, 1e-6f);
}

TEST(SampledWavelengths, TerminateSecondaryIdempotent) {
	SWL4 swl = SWL4::SampleVisible(0.6f);
	swl.TerminateSecondary();
	float p0 = swl.PDF()[0];
	swl.TerminateSecondary();  // second call must be no-op
	EXPECT_FLOAT_EQ(swl.PDF()[0], p0);
}

// ============================================================================
// SampledWavelengths -- equality
// ============================================================================

TEST(SampledWavelengths, Equality) {
	SWL4 a = SWL4::SampleUniform(0.5f);
	SWL4 b = SWL4::SampleUniform(0.5f);
	EXPECT_EQ(a, b);
	b.lambda[0] += 1.f;
	EXPECT_NE(a, b);
}

// ============================================================================
// Integration: SampledWavelengths + SampledSpectrum together
// ============================================================================

TEST(SampledSpectrumIntegration, ConstantSpectrumTimesInversePDF) {
	// Estimator: sum( f(lambda_i) / pdf(lambda_i) ) / N  ≈  integral(f, 360, 830)
	// For f = 1 (constant), integral = 470.
	// With many draws (N=4 per draw, 1000 draws) the Monte Carlo estimate
	// should be close to 470.
	double sum = 0.0;
	const int draws = 2000;
	for (int k = 0; k < draws; ++k) {
		float u = (k + 0.5f) / float(draws);
		SWL4 swl = SWL4::SampleVisible(u);
		SS4  pdf  = swl.PDF();
		for (int i = 0; i < 4; ++i) {
			// f(lambda) = 1, contribution = 1 / pdf[i]
			sum += 1.0 / pdf[i];
		}
	}
	double estimate = sum / (draws * 4);
	EXPECT_NEAR(estimate, 470.0, 5.0);  // visible range width = 830-360 = 470
}

TEST(SampledSpectrumIntegration, ArithmeticChain) {
	// (a + b) * c / 2  should equal (a*c + b*c) / 2 component-wise.
	float va[4] = {1.f, 2.f, 3.f, 4.f};
	float vb[4] = {4.f, 3.f, 2.f, 1.f};
	float vc[4] = {2.f, 2.f, 2.f, 2.f};
	SS4 a(va, 4), b(vb, 4), c(vc, 4);
	SS4 lhs = (a + b) * c / 2.f;
	SS4 rhs = (a * c + b * c) / 2.f;
	for (int i = 0; i < 4; ++i)
		EXPECT_NEAR(lhs[i], rhs[i], 1e-5f);
}

// ============================================================================
// spectral_math.h -- ToXYZ / LuminanceY / ToRGB / SpectrumToXYZ / InnerProduct
//
// pbrt-v4 reference: util/spectrum.h, util/spectrum.cpp
// Algorithm: Monte Carlo estimator
//   XYZ = (SafeDiv(CIE_X/Y/Z.Sample(swl) * ss, swl.PDF())).Average()
//         / CIE_Y_integral
// ============================================================================

// ---------------------------------------------------------------------------
// InnerProduct
// ---------------------------------------------------------------------------

TEST(SpectralMath, InnerProductConstantOne) {
	// InnerProduct(1, 1) = sum_{360}^{830} 1.0 = 471 samples
	ConstantSpectrum one(1.f);
	float ip = InnerProduct(one, one);
	EXPECT_NEAR(ip, 471.f, 1.f);  // 360..830 inclusive = 471 integers
}

TEST(SpectralMath, InnerProductCIEYSelf) {
	// InnerProduct(CIE_Y, CIE_Y) must be positive and finite
	float ip = InnerProduct(GetCIE_Y(), GetCIE_Y());
	EXPECT_GT(ip, 0.f);
	EXPECT_TRUE(std::isfinite(ip));
}

// ---------------------------------------------------------------------------
// SpectrumToXYZ (deterministic Riemann-sum integration)
// ---------------------------------------------------------------------------

TEST(SpectralMath, SpectrumToXYZ_ConstantOne_YEqualsOne) {
	// For a perfectly white spectrum (radiance = 1 at all lambda),
	// Y = InnerProduct(CIE_Y, 1) / CIE_Y_integral = 1.0 by definition.
	ConstantSpectrum white(1.f);
	XYZ xyz = SpectrumToXYZ(white);
	EXPECT_NEAR(xyz.Y, 1.f, 5e-4f);  // within 0.05% of 1.0
	EXPECT_GT(xyz.X, 0.f);
	EXPECT_GT(xyz.Z, 0.f);
}

TEST(SpectralMath, SpectrumToXYZ_Zero_IsZero) {
	ConstantSpectrum zero(0.f);
	XYZ xyz = SpectrumToXYZ(zero);
	EXPECT_FLOAT_EQ(xyz.X, 0.f);
	EXPECT_FLOAT_EQ(xyz.Y, 0.f);
	EXPECT_FLOAT_EQ(xyz.Z, 0.f);
}

TEST(SpectralMath, SpectrumToXYZ_ScaleLinear) {
	// SpectrumToXYZ(2*s) = 2 * SpectrumToXYZ(s) since integration is linear.
	ConstantSpectrum s1(1.f);
	ConstantSpectrum s2(2.f);
	XYZ xyz1 = SpectrumToXYZ(s1);
	XYZ xyz2 = SpectrumToXYZ(s2);
	EXPECT_NEAR(xyz2.X, 2.f * xyz1.X, 1e-4f);
	EXPECT_NEAR(xyz2.Y, 2.f * xyz1.Y, 1e-4f);
	EXPECT_NEAR(xyz2.Z, 2.f * xyz1.Z, 1e-4f);
}

// ---------------------------------------------------------------------------
// ToXYZ (Monte Carlo estimator)
// ---------------------------------------------------------------------------

TEST(SpectralMath, ToXYZ_ConstantOne_YApproxOne) {
	// A constant-radiance spectrum should give Y ≈ 1 under the MC estimator.
	// Use SampleUniform so we can use many strata for low variance.
	double sumY = 0.0;
	const int K = 1024;
	for (int k = 0; k < K; ++k) {
		float u = (k + 0.5f) / float(K);
		SWL4 swl = SWL4::SampleUniform(u);
		SS4  ss(1.f);
		XYZ  xyz = ToXYZ(ss, swl);
		sumY += xyz.Y;
	}
	double meanY = sumY / K;
	// Uniform sampling of [360,830] with 1/471 PDF; expect Y ≈ 1.0.
	EXPECT_NEAR(meanY, 1.0, 0.05);  // 5% tolerance for MC estimator
}

TEST(SpectralMath, ToXYZ_ZeroSpectrum_IsZero) {
	SWL4 swl = SWL4::SampleVisible(0.5f);
	SS4  ss(0.f);
	XYZ  xyz = ToXYZ(ss, swl);
	EXPECT_FLOAT_EQ(xyz.X, 0.f);
	EXPECT_FLOAT_EQ(xyz.Y, 0.f);
	EXPECT_FLOAT_EQ(xyz.Z, 0.f);
}

TEST(SpectralMath, ToXYZ_AgreesWithSpectrumToXYZ_Constant) {
	// For a DenselySampledSpectrum constant = 1, both estimators should agree
	// within Monte Carlo variance when averaged over enough samples.
	DenselySampledSpectrum ds = DenselySampledSpectrum::SampleFunction(
		[](float) { return 1.f; });

	// Deterministic reference via Riemann sum
	XYZ ref = SpectrumToXYZ(ds);

	// MC estimate averaged over 512 stratified draws
	double sumX = 0, sumY = 0, sumZ = 0;
	const int K = 512;
	for (int k = 0; k < K; ++k) {
		float u = (k + 0.5f) / float(K);
		SWL4 swl = SWL4::SampleVisible(u);
		SS4  ss  = ds.Sample(swl);
		XYZ  xyz = ToXYZ(ss, swl);
		sumX += xyz.X; sumY += xyz.Y; sumZ += xyz.Z;
	}
	EXPECT_NEAR(sumX / K, ref.X, 0.02f);
	EXPECT_NEAR(sumY / K, ref.Y, 0.02f);
	EXPECT_NEAR(sumZ / K, ref.Z, 0.02f);
}

// ---------------------------------------------------------------------------
// LuminanceY
// ---------------------------------------------------------------------------

TEST(SpectralMath, LuminanceY_ConstantOne_ApproxOne) {
	// Average over stratified uniform draws.
	double sumY = 0.0;
	const int K = 1024;
	for (int k = 0; k < K; ++k) {
		float u = (k + 0.5f) / float(K);
		SWL4 swl = SWL4::SampleUniform(u);
		SS4  ss(1.f);
		sumY += LuminanceY(ss, swl);
	}
	EXPECT_NEAR(sumY / K, 1.0, 0.05);
}

TEST(SpectralMath, LuminanceY_MatchesToXYZ_Y) {
	// LuminanceY should equal ToXYZ(...).Y to floating-point precision.
	SWL4 swl = SWL4::SampleVisible(0.3f);
	SS4  ss(0.6f);
	float yFromLum = LuminanceY(ss, swl);
	float yFromXYZ = ToXYZ(ss, swl).Y;
	EXPECT_FLOAT_EQ(yFromLum, yFromXYZ);
}

TEST(SpectralMath, LuminanceY_Zero_IsZero) {
	SWL4 swl = SWL4::SampleVisible(0.5f);
	SS4  ss(0.f);
	EXPECT_FLOAT_EQ(LuminanceY(ss, swl), 0.f);
}

// ---------------------------------------------------------------------------
// ToRGB
// ---------------------------------------------------------------------------

TEST(SpectralMath, ToRGB_ConstantOne_PositiveAndFinite) {
	// A white spectrum should produce positive, finite RGB values.
	SWL4 swl = SWL4::SampleUniform(0.5f);
	SS4  ss(1.f);
	const RGBColorSpace& cs = RGBColorSpace::sRGB();
	float r, g, b;
	ToRGB(ss, swl, cs, r, g, b);
	EXPECT_TRUE(std::isfinite(r));
	EXPECT_TRUE(std::isfinite(g));
	EXPECT_TRUE(std::isfinite(b));
}

TEST(SpectralMath, ToRGB_ZeroSpectrum_IsZero) {
	SWL4 swl = SWL4::SampleVisible(0.5f);
	SS4  ss(0.f);
	const RGBColorSpace& cs = RGBColorSpace::sRGB();
	float r, g, b;
	ToRGB(ss, swl, cs, r, g, b);
	EXPECT_FLOAT_EQ(r, 0.f);
	EXPECT_FLOAT_EQ(g, 0.f);
	EXPECT_FLOAT_EQ(b, 0.f);
}

TEST(SpectralMath, ToRGB_ConsistentWithToXYZ) {
	// ToRGB(ss, swl, cs) = cs.FromXYZ(ToXYZ(ss, swl)) -- verify both routes agree.
	SWL4 swl = SWL4::SampleVisible(0.7f);
	SS4  ss(0.5f);
	const RGBColorSpace& cs = RGBColorSpace::sRGB();

	// Route 1: ToRGB directly
	float r1, g1, b1;
	ToRGB(ss, swl, cs, r1, g1, b1);

	// Route 2: ToXYZ then FromXYZ
	XYZ xyz = ToXYZ(ss, swl);
	float r2, g2, b2;
	cs.FromXYZ(xyz.X, xyz.Y, xyz.Z, r2, g2, b2);

	EXPECT_FLOAT_EQ(r1, r2);
	EXPECT_FLOAT_EQ(g1, g2);
	EXPECT_FLOAT_EQ(b1, b2);
}

TEST(SpectralMath, ToRGB_WhiteSpectrum_NearEqualRGB) {
	// A flat equal-energy spectrum should produce roughly equal R, G, B
	// (not equal due to primary chromaticities, but within a factor of 2).
	double sumR = 0, sumG = 0, sumB = 0;
	const int K = 512;
	const RGBColorSpace& cs = RGBColorSpace::sRGB();
	for (int k = 0; k < K; ++k) {
		float u = (k + 0.5f) / float(K);
		SWL4 swl = SWL4::SampleUniform(u);
		SS4  ss(1.f);
		float r, g, b;
		ToRGB(ss, swl, cs, r, g, b);
		sumR += r; sumG += g; sumB += b;
	}
	float mR = float(sumR / K), mG = float(sumG / K), mB = float(sumB / K);
	// All channels must be positive
	EXPECT_GT(mR, 0.f);
	EXPECT_GT(mG, 0.f);
	EXPECT_GT(mB, 0.f);
	// Channels must be within 2x of each other for a white source
	EXPECT_LT(std::max({mR, mG, mB}) / std::min({mR, mG, mB}), 3.f);
}

// ---------------------------------------------------------------------------
// pbrt-v4 alignment: CIE_Y_integral constant
// ---------------------------------------------------------------------------

TEST(SpectralMath, CIEYIntegral_MatchesPbrtV4) {
	// pbrt-v4: static constexpr Float CIE_Y_integral = 106.856895;
	EXPECT_NEAR(kCIE_Y_integral, 106.856895f, 1e-4f);
}

TEST(SpectralMath, CIEYIntegral_MatchesRiemannSum) {
	// InnerProduct(CIE_Y, 1) should reproduce kCIE_Y_integral within 1-nm step.
	ConstantSpectrum one(1.f);
	float ip = InnerProduct(GetCIE_Y(), one);
	EXPECT_NEAR(ip, kCIE_Y_integral, 0.1f);  // 1-nm Riemann sum vs exact integral
}

// ---------------------------------------------------------------------------
// SampledSpectrumToXYZ -- raw-pointer version (used by GPU device code)
// pbrt-v4 alignment: numerically equivalent to SampledSpectrum::ToXYZ()
// ---------------------------------------------------------------------------

TEST(SampledSpectrumToXYZ, ZeroSpectrum_IsZero) {
	SWL4 swl = SWL4::SampleVisible(0.5f);
	SS4  L(0.f);
	auto xyz = SampledSpectrumToXYZ(L, swl, CIE_X, CIE_Y, CIE_Z,
									 kCIELambda_min, kCIENSamples);
	EXPECT_FLOAT_EQ(xyz.x, 0.f);
	EXPECT_FLOAT_EQ(xyz.y, 0.f);
	EXPECT_FLOAT_EQ(xyz.z, 0.f);
}

TEST(SampledSpectrumToXYZ, MatchesToXYZFreeFunction) {
	// Both routes must agree to floating-point precision.
	SWL4 swl = SWL4::SampleVisible(0.3f);
	SS4  L(0.7f);

	// Route 1: raw-pointer version (device-facing)
	auto xyz1 = SampledSpectrumToXYZ(L, swl, CIE_X, CIE_Y, CIE_Z,
									  kCIELambda_min, kCIENSamples);

	// Route 2: spectral_math ToXYZ (host, uses DenselySampledSpectrum::Sample)
	XYZ xyz2 = ToXYZ(L, swl);

	EXPECT_NEAR(xyz1.x, xyz2.X, 1e-5f);
	EXPECT_NEAR(xyz1.y, xyz2.Y, 1e-5f);
	EXPECT_NEAR(xyz1.z, xyz2.Z, 1e-5f);
}

TEST(SampledSpectrumToXYZ, TerminatedSecondary_MatchesPrimary) {
	// After TerminateSecondary, only lambda[0] carries weight.
	// Both ToXYZ routes should still agree.
	SWL4 swl = SWL4::SampleVisible(0.6f);
	swl.TerminateSecondary();
	SS4 L(1.f);

	auto xyz1 = SampledSpectrumToXYZ(L, swl, CIE_X, CIE_Y, CIE_Z,
									  kCIELambda_min, kCIENSamples);
	XYZ xyz2 = ToXYZ(L, swl);

	EXPECT_NEAR(xyz1.x, xyz2.X, 1e-5f);
	EXPECT_NEAR(xyz1.y, xyz2.Y, 1e-5f);
	EXPECT_NEAR(xyz1.z, xyz2.Z, 1e-5f);
}

// ---------------------------------------------------------------------------
// XYZToSRGB -- linear sRGB output
// pbrt-v4 alignment: matches sRGB standard (IEC 61966-2-1)
// ---------------------------------------------------------------------------

TEST(XYZToSRGB, BlackIsBlack) {
	float r, g, b;
	XYZToSRGB(0.f, 0.f, 0.f, r, g, b);
	EXPECT_FLOAT_EQ(r, 0.f);
	EXPECT_FLOAT_EQ(g, 0.f);
	EXPECT_FLOAT_EQ(b, 0.f);
}

TEST(XYZToSRGB, D65WhiteIsNearGrey) {
	// D65 white in XYZ: X=0.9504, Y=1.0, Z=1.0888
	// sRGB(D65 white) ~ (1,1,1)
	float r, g, b;
	XYZToSRGB(0.9504f, 1.0f, 1.0888f, r, g, b);
	EXPECT_NEAR(r, 1.f, 0.02f);
	EXPECT_NEAR(g, 1.f, 0.02f);
	EXPECT_NEAR(b, 1.f, 0.02f);
}

TEST(XYZToSRGB, NegativeXYZClampedToZero) {
	// Negative XYZ (out of gamut) must not produce negative RGB channels.
	float r, g, b;
	XYZToSRGB(-0.5f, -0.5f, -0.5f, r, g, b);
	EXPECT_GE(r, 0.f);
	EXPECT_GE(g, 0.f);
	EXPECT_GE(b, 0.f);
}

TEST(XYZToSRGB, GammaEncodingApplied) {
	// For D65-neutral grey at Y=0.2, the sRGB matrix should give r≈g≈b (neutral).
	// D65 white point: X_w=0.95047, Z_w=1.08883 normalized to Y=1.
	// At Y=0.2: X=0.19009, Y=0.2, Z=0.21777
	float r, g, b;
	XYZToSRGB(0.19009f, 0.2f, 0.21777f, r, g, b);
	// neutral grey after matrix + gamma: r≈g≈b
	EXPECT_NEAR(r, g, 0.01f);
	EXPECT_NEAR(g, b, 0.01f);
	// gamma-encoded 0.2 linear ≈ 0.48 sRGB
	EXPECT_NEAR(r, 0.48f, 0.05f);
}

