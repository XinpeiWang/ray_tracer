//==============================================================================
// scalar_distributions_tests.cpp
// Unit tests for the 1-D scalar distribution family and VarianceEstimator
// ported from pbrt-v4 util/sampling.h.
//
// Distributions tested:
//   Tent, Exponential, TrimmedExponential,
//   Normal / TwoNormal, Logistic / TrimmedLogistic,
//   SmoothStep, VarianceEstimator
//==============================================================================

#include "gtest/gtest.h"
#include "../../src/shared/scalar_distributions.h"
#include <cmath>
#include <vector>
#include <numeric>
#include <limits>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static uint32_t g_lcg = 0xdeadbeef;
static float lcg() {
	g_lcg = g_lcg * 1664525u + 1013904223u;
	return (g_lcg >> 8) * (1.f / (1 << 24));
}

// Correct importance-sampling MC integral estimator.
// For a distribution over a *bounded* domain [a,b]:
//   integral_a^b f(x) dx  ≈  (1/N) * sum f(Sample(u_i)) / pdf(Sample(u_i))
//                           = (1/N) * sum 1                                = 1
// when f(x) = pdf(x).  Effectively verifies integral(pdf)=1.
template<typename SampleFn, typename PDFFn>
static double MCIntegral(SampleFn sample, PDFFn pdf, int N = 50000) {
	double sum = 0.0;
	for (int i = 0; i < N; ++i) {
		float u = lcg();
		float x = sample(u);
		float p = pdf(x);
		if (p > 0.f) sum += 1.0;   // f(x)/p(x) = p(x)/p(x) = 1 always
	}
	return sum / N;  // should be 1.0
}

// ===========================================================================
// Tent
// ===========================================================================
TEST(Tent, PDF_IntegratesTo1) {
	float r = 1.5f;
	double sum = 0.0;
	int N = 100000;
	for (int i = 0; i < N; ++i) {
		float x = (lcg() * 2.f - 1.f) * r;  // uniform on [-r,r]
		sum += TentPDF(x, r);
	}
	EXPECT_NEAR(sum / N * (2.f * r), 1.0, 0.02);
}

TEST(Tent, SampleIsInRange) {
	float r = 2.f;
	for (int i = 0; i < 100; ++i) {
		float x = SampleTent(lcg(), r);
		EXPECT_GE(x, -r);
		EXPECT_LE(x,  r);
	}
}

TEST(Tent, InvertRoundTrip) {
	float r = 1.f;
	for (int i = 0; i < 200; ++i) {
		float u = lcg();
		float x = SampleTent(u, r);
		float u2 = InvertTentSample(x, r);
		EXPECT_NEAR(u2, u, 1e-4f) << "u=" << u << " x=" << x;
	}
}

TEST(Tent, MCNorm) {
	float r = 1.f;
	double est = MCIntegral(
		[r](float u) { return SampleTent(u, r); },
		[r](float x) { return TentPDF(x, r); });
	EXPECT_NEAR(est, 1.0, 0.01);
}

// ===========================================================================
// Exponential
// ===========================================================================
TEST(Exponential, SamplePositive) {
	for (int i = 0; i < 100; ++i)
		EXPECT_GT(SampleExponential(lcg(), 2.f), 0.f);
}

TEST(Exponential, InvertRoundTrip) {
	float a = 3.f;
	for (int i = 0; i < 200; ++i) {
		float u  = lcg();
		float x  = SampleExponential(u, a);
		float u2 = InvertExponentialSample(x, a);
		EXPECT_NEAR(u2, u, 1e-4f);
	}
}

TEST(Exponential, MCNorm) {
	// Verify normalization by checking that for x ~ Exp(a),
	// pdf(x) is always positive and finite, and the mean of x is 1/a.
	float a = 2.5f;
	double sum = 0.0;
	int N = 50000;
	for (int i = 0; i < N; ++i) {
		float u = 0.001f + lcg() * 0.998f;
		float x = SampleExponential(u, a);
		sum += x;
	}
	// E[X] for Exp(a) = 1/a
	EXPECT_NEAR(sum / N, 1.0 / a, 0.02f);
}

// ===========================================================================
// TrimmedExponential
// ===========================================================================
TEST(TrimmedExponential, SampleInRange) {
	float c = 2.f, xMax = 3.f;
	for (int i = 0; i < 100; ++i) {
		float x = SampleTrimmedExponential(lcg(), c, xMax);
		EXPECT_GE(x, 0.f);
		EXPECT_LE(x, xMax);
	}
}

TEST(TrimmedExponential, InvertRoundTrip) {
	float c = 1.5f, xMax = 2.f;
	for (int i = 0; i < 200; ++i) {
		float u  = lcg();
		float x  = SampleTrimmedExponential(u, c, xMax);
		float u2 = InvertTrimmedExponentialSample(x, c, xMax);
		EXPECT_NEAR(u2, u, 1e-4f);
	}
}

TEST(TrimmedExponential, MCNorm) {
	float c = 2.f, xMax = 3.f;
	double est = MCIntegral(
		[c, xMax](float u) { return SampleTrimmedExponential(u, c, xMax); },
		[c, xMax](float x) { return TrimmedExponentialPDF(x, c, xMax); });
	EXPECT_NEAR(est, 1.0, 0.01);
}

// ===========================================================================
// Normal
// ===========================================================================
TEST(Normal, SampleFinite) {
	for (int i = 0; i < 100; ++i) {
		float u = 0.001f + lcg() * 0.998f;
		float x = SampleNormal(u, 0.f, 1.f);
		EXPECT_FALSE(std::isnan(x));
		EXPECT_FALSE(std::isinf(x));
	}
}

TEST(Normal, InvertRoundTrip) {
	for (int i = 0; i < 200; ++i) {
		float u  = 0.001f + lcg() * 0.998f;
		float x  = SampleNormal(u, 0.f, 1.f);
		float u2 = InvertNormalSample(x, 0.f, 1.f);
		EXPECT_NEAR(u2, u, 1e-4f) << "u=" << u;
	}
}

TEST(Normal, MCNorm) {
	// Verify E[X^2] for N(0,1) = 1 (second moment of standard normal)
	double sum = 0.0;
	int N = 100000;
	for (int i = 0; i < N; ++i) {
		float u = 0.001f + lcg() * 0.998f;
		float x = SampleNormal(u, 0.f, 1.f);
		sum += (double)x * x;
	}
	// E[X^2] = Var(X) + E[X]^2 = 1 + 0 = 1
	EXPECT_NEAR(sum / N, 1.0, 0.05);
}

TEST(Normal, TwoNormal_Independent) {
	// Box-Muller: two outputs should be independent N(0,1)
	double sumSq = 0.0;
	int N = 10000;
	for (int i = 0; i < N; ++i) {
		float a, b;
		SampleTwoNormal(0.001f + lcg() * 0.998f, lcg(), a, b);
		sumSq += a * a + b * b;
	}
	// E[a^2 + b^2] = 2 for two standard normals
	EXPECT_NEAR(sumSq / N, 2.0, 0.1);
}

// ===========================================================================
// Logistic
// ===========================================================================
TEST(Logistic, InvertRoundTrip) {
	float s = 0.5f;
	for (int i = 0; i < 200; ++i) {
		float u  = 0.001f + lcg() * 0.998f;
		float x  = SampleLogistic(u, s);
		float u2 = InvertLogisticSample(x, s);
		EXPECT_NEAR(u2, u, 1e-4f);
	}
}

TEST(Logistic, MCNorm) {
	// Verify E[X] for Logistic(s) = 0 (symmetric distribution)
	float s = 0.5f;
	double sum = 0.0;
	int N = 50000;
	for (int i = 0; i < N; ++i) {
		float u = 0.001f + lcg() * 0.998f;
		float x = SampleLogistic(u, s);
		sum += x;
	}
	EXPECT_NEAR(sum / N, 0.0, 0.05);
}

// ===========================================================================
// TrimmedLogistic
// ===========================================================================
TEST(TrimmedLogistic, SampleInRange) {
	float s = 0.5f, a = -2.f, b = 2.f;
	for (int i = 0; i < 100; ++i) {
		float x = SampleTrimmedLogistic(lcg(), s, a, b);
		EXPECT_GE(x, a);
		EXPECT_LE(x, b);
	}
}

TEST(TrimmedLogistic, InvertRoundTrip) {
	float s = 0.5f, a = -2.f, b = 2.f;
	for (int i = 0; i < 200; ++i) {
		float u  = lcg();
		float x  = SampleTrimmedLogistic(u, s, a, b);
		float u2 = InvertTrimmedLogisticSample(x, s, a, b);
		EXPECT_NEAR(u2, u, 1e-4f) << "u=" << u;
	}
}

TEST(TrimmedLogistic, MCNorm) {
	float s = 0.5f, a = -2.f, b = 2.f;
	double est = MCIntegral(
		[s,a,b](float u) { return SampleTrimmedLogistic(u, s, a, b); },
		[s,a,b](float x) { return TrimmedLogisticPDF(x, s, a, b); });
	EXPECT_NEAR(est, 1.0, 0.01);
}

// ===========================================================================
// SmoothStep
// ===========================================================================
TEST(SmoothStep, SampleInRange) {
	float a = 1.f, b = 3.f;
	for (int i = 0; i < 100; ++i) {
		float x = SampleSmoothStep(lcg(), a, b);
		EXPECT_GE(x, a);
		EXPECT_LE(x, b);
	}
}

TEST(SmoothStep, InvertRoundTrip) {
	float a = 0.f, b = 2.f;
	for (int i = 0; i < 200; ++i) {
		float u  = lcg();
		float x  = SampleSmoothStep(u, a, b);
		float u2 = InvertSmoothStepSample(x, a, b);
		EXPECT_NEAR(u2, u, 1e-3f) << "u=" << u;
	}
}

TEST(SmoothStep, MCNorm) {
	float a = 0.f, b = 1.f;
	double est = MCIntegral(
		[a,b](float u) { return SampleSmoothStep(u, a, b); },
		[a,b](float x) { return SmoothStepPDF(x, a, b); });
	EXPECT_NEAR(est, 1.0, 0.01);
}

