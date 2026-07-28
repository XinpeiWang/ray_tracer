// sampling_distributions_tests.cpp
// Unit tests for src/shared/sampling_distributions.h
//
// Each distribution is tested for:
//   1. PDF non-negativity and approximate integral == 1
//   2. Sample/Invert round-trip (Sample then Invert recovers u)
//   3. Range of generated samples

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include "../../src/shared/sampling_distributions.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Numerically integrate f over [lo, hi] with N-step midpoint rule.
static double integrate(std::function<double(double)> f, double lo, double hi, int N = 4000) {
	double h = (hi - lo) / N;
	double s = 0.0;
	for (int i = 0; i < N; ++i)
		s += f(lo + (i + 0.5) * h);
	return s * h;
}

static constexpr int kRoundTrips = 64;

// ---------------------------------------------------------------------------
// 1. Tent
// ---------------------------------------------------------------------------

TEST(TentDistribution, PDFNonNeg) {
	double r = 1.5;
	for (double x = -3.0; x <= 3.0; x += 0.1)
		EXPECT_GE(TentPDF(x, r), 0.0) << "x=" << x;
}

TEST(TentDistribution, PDFIntegral) {
	double r = 1.5;
	double integral = integrate([r](double x){ return TentPDF(x, r); }, -r, r);
	EXPECT_NEAR(integral, 1.0, 1e-4);
}

TEST(TentDistribution, SampleInRange) {
	for (int i = 1; i <= kRoundTrips; ++i) {
		double u = i / double(kRoundTrips + 1);
		double x = SampleTent(u, 2.0);
		EXPECT_GE(x, -2.0);
		EXPECT_LE(x,  2.0);
	}
}

TEST(TentDistribution, RoundTrip) {
	double r = 1.5;
	for (int i = 1; i <= kRoundTrips; ++i) {
		double u = i / double(kRoundTrips + 1);
		double x = SampleTent(u, r);
		double u2 = InvertTentSample(x, r);
		EXPECT_NEAR(u2, u, 1e-5) << "u=" << u;
	}
}

// ---------------------------------------------------------------------------
// 2. Exponential
// ---------------------------------------------------------------------------

TEST(ExponentialDistribution, PDFNonNeg) {
	for (double x = 0.0; x <= 5.0; x += 0.25)
		EXPECT_GE(ExponentialPDF(x, 2.0), 0.0);
}

TEST(ExponentialDistribution, PDFIntegral) {
	double integral = integrate([](double x){ return ExponentialPDF(x, 2.0); }, 0.0, 20.0);
	EXPECT_NEAR(integral, 1.0, 1e-4);
}

TEST(ExponentialDistribution, SamplePositive) {
	for (int i = 1; i <= kRoundTrips; ++i) {
		double u = i / double(kRoundTrips + 1);
		EXPECT_GT(SampleExponential(u, 2.0), 0.0);
	}
}

TEST(ExponentialDistribution, RoundTrip) {
	double a = 2.0;
	for (int i = 1; i <= kRoundTrips; ++i) {
		double u = i / double(kRoundTrips + 1);
		double x = SampleExponential(u, a);
		double u2 = InvertExponentialSample(x, a);
		EXPECT_NEAR(u2, u, 1e-6) << "u=" << u;
	}
}

// ---------------------------------------------------------------------------
// 3. Trimmed Exponential
// ---------------------------------------------------------------------------

TEST(TrimmedExponentialDistribution, PDFIntegral) {
	double c = 1.5, xMax = 3.0;
	double integral = integrate([c,xMax](double x){ return TrimmedExponentialPDF(x, c, xMax); }, 0.0, xMax);
	EXPECT_NEAR(integral, 1.0, 1e-4);
}

TEST(TrimmedExponentialDistribution, SampleInRange) {
	double c = 1.5, xMax = 3.0;
	for (int i = 1; i <= kRoundTrips; ++i) {
		double u = i / double(kRoundTrips + 1);
		double x = SampleTrimmedExponential(u, c, xMax);
		EXPECT_GE(x, 0.0);
		EXPECT_LE(x, xMax);
	}
}

TEST(TrimmedExponentialDistribution, RoundTrip) {
	double c = 1.5, xMax = 3.0;
	for (int i = 1; i <= kRoundTrips; ++i) {
		double u = i / double(kRoundTrips + 1);
		double x = SampleTrimmedExponential(u, c, xMax);
		double u2 = InvertTrimmedExponentialSample(x, c, xMax);
		EXPECT_NEAR(u2, u, 1e-6) << "u=" << u;
	}
}

// ---------------------------------------------------------------------------
// 4. Normal
// ---------------------------------------------------------------------------

TEST(NormalDistribution, PDFNonNeg) {
	for (double x = -4.0; x <= 4.0; x += 0.25)
		EXPECT_GE(NormalPDF(x), 0.0);
}

TEST(NormalDistribution, PDFIntegral) {
	double integral = integrate([](double x){ return NormalPDF(x); }, -8.0, 8.0);
	EXPECT_NEAR(integral, 1.0, 1e-4);
}

TEST(NormalDistribution, SampleMedian) {
	// median of standard normal is 0
	double x = SampleNormal(0.5);
	EXPECT_NEAR(x, 0.0, 1e-4);
}

TEST(NormalDistribution, RoundTrip) {
	for (int i = 1; i <= kRoundTrips; ++i) {
		double u = i / double(kRoundTrips + 1);
		double x = SampleNormal(u);
		double u2 = InvertNormalSample(x);
		EXPECT_NEAR(u2, u, 1e-4) << "u=" << u;
	}
}

// ---------------------------------------------------------------------------
// 5. TwoNormal (Box-Muller)
// ---------------------------------------------------------------------------

TEST(TwoNormal, OutputsInReasonableRange) {
	double out0, out1;
	SampleTwoNormal(0.5, 0.3, out0, out1);
	EXPECT_LT(std::abs(out0), 6.0);
	EXPECT_LT(std::abs(out1), 6.0);
}

TEST(TwoNormal, Deterministic) {
	double a0, a1, b0, b1;
	SampleTwoNormal(0.2, 0.7, a0, a1);
	SampleTwoNormal(0.2, 0.7, b0, b1);
	EXPECT_DOUBLE_EQ(a0, b0);
	EXPECT_DOUBLE_EQ(a1, b1);
}

// ---------------------------------------------------------------------------
// 6. Logistic
// ---------------------------------------------------------------------------

TEST(LogisticDistribution, PDFNonNeg) {
	for (double x = -5.0; x <= 5.0; x += 0.5)
		EXPECT_GE(LogisticPDF(x, 1.0), 0.0);
}

TEST(LogisticDistribution, PDFIntegral) {
	double integral = integrate([](double x){ return LogisticPDF(x, 1.0); }, -20.0, 20.0);
	EXPECT_NEAR(integral, 1.0, 1e-4);
}

TEST(LogisticDistribution, RoundTrip) {
	double s = 0.5;
	for (int i = 1; i <= kRoundTrips; ++i) {
		double u = i / double(kRoundTrips + 1);
		double x = SampleLogistic(u, s);
		double u2 = InvertLogisticSample(x, s);
		EXPECT_NEAR(u2, u, 1e-6) << "u=" << u;
	}
}

// ---------------------------------------------------------------------------
// 7. Trimmed Logistic
// ---------------------------------------------------------------------------

TEST(TrimmedLogisticDistribution, PDFIntegral) {
	double s = 0.5, a = -1.0, b = 2.0;
	double integral = integrate([s,a,b](double x){ return TrimmedLogisticPDF(x, s, a, b); }, a, b);
	EXPECT_NEAR(integral, 1.0, 1e-4);
}

TEST(TrimmedLogisticDistribution, SampleInRange) {
	double s = 0.5, a = -1.0, b = 2.0;
	for (int i = 1; i <= kRoundTrips; ++i) {
		double u = i / double(kRoundTrips + 1);
		double x = SampleTrimmedLogistic(u, s, a, b);
		EXPECT_GE(x, a);
		EXPECT_LE(x, b);
	}
}

TEST(TrimmedLogisticDistribution, RoundTrip) {
	double s = 0.5, a = -1.0, b = 2.0;
	for (int i = 1; i <= kRoundTrips; ++i) {
		double u = i / double(kRoundTrips + 1);
		double x = SampleTrimmedLogistic(u, s, a, b);
		double u2 = InvertTrimmedLogisticSample(x, s, a, b);
		EXPECT_NEAR(u2, u, 1e-5) << "u=" << u;
	}
}

// ---------------------------------------------------------------------------
// 8. SmoothStep
// ---------------------------------------------------------------------------

TEST(SmoothStepDistribution, PDFNonNeg) {
	double a = 0.2, b = 0.8;
	for (double x = 0.0; x <= 1.0; x += 0.05)
		EXPECT_GE(SmoothStepPDF(x, a, b), 0.0);
}

TEST(SmoothStepDistribution, PDFIntegral) {
	double a = 0.2, b = 0.8;
	double integral = integrate([a,b](double x){ return SmoothStepPDF(x, a, b); }, a, b);
	EXPECT_NEAR(integral, 1.0, 1e-4);
}

TEST(SmoothStepDistribution, SampleInRange) {
	double a = 0.2, b = 0.8;
	for (int i = 1; i <= kRoundTrips; ++i) {
		double u = i / double(kRoundTrips + 1);
		double x = SampleSmoothStep(u, a, b);
		EXPECT_GE(x, a);
		EXPECT_LE(x, b);
	}
}

TEST(SmoothStepDistribution, RoundTrip) {
	double a = 0.2, b = 0.8;
	for (int i = 1; i <= kRoundTrips; ++i) {
		double u = i / double(kRoundTrips + 1);
		double x = SampleSmoothStep(u, a, b);
		double u2 = InvertSmoothStepSample(x, a, b);
		EXPECT_NEAR(u2, u, 1e-5) << "u=" << u;
	}
}

TEST(SmoothStepDistribution, InvertBoundaries) {
	// At x=a: t=0, P=0; at x=b: t=1, P=1
	EXPECT_NEAR(InvertSmoothStepSample(0.2, 0.2, 0.8), 0.0, 1e-10);
	EXPECT_NEAR(InvertSmoothStepSample(0.8, 0.2, 0.8), 1.0, 1e-10);
}

TEST(TwoNormal, SpreadOverManyDraws) {
	// Gather 100 pairs; both outputs should span both sides of the mean.
	double sumA = 0, sumB = 0;
	for (int i = 0; i < 100; ++i) {
		double u0 = (i + 0.5) / 100.0;
		double u1 = std::fmod((i * 0.61803398875) + 0.1, 1.0); // golden ratio sequence
		double a, b;
		SampleTwoNormal(u0, u1, a, b);
		sumA += a; sumB += b;
	}
	// Mean of 100 standard-normal draws should be near 0
	EXPECT_NEAR(sumA / 100.0, 0.0, 0.4);
	EXPECT_NEAR(sumB / 100.0, 0.0, 0.4);
}
