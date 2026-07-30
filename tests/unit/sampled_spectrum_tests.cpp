// sampled_spectrum_tests.cpp
// Unit tests for src/shared/sampled_spectrum.h
//
// Tests mirror pbrt-v4 semantics for SampledSpectrum<N> and SampledWavelengths<N>.
// Each section tests a logical group of functionality.

#include <gtest/gtest.h>
#include <cmath>
#include <array>
#include "../../src/shared/sampled_spectrum.h"

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
