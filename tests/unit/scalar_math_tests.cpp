// scalar_math_tests.cpp -- Unit tests for src/shared/scalar_math.h
// Mirrors pbrt-v4 scalar math semantics.

#include <cmath>
#include <cstring>
#include <cstdint>
#include <utility>
#include "../../src/shared/scalar_math.h"
#include <gtest/gtest.h>
#include <utility>

// kPi for internal use -- always comes from the detail namespace to avoid
// redefinition issues regardless of whether scalar_math.h exports Pi globally.
static constexpr double kPi = scalar_math_detail::kPi;

// ---------------------------------------------------------------------------
// Lerp
// ---------------------------------------------------------------------------
TEST(LerpTest, Endpoints) {
	EXPECT_DOUBLE_EQ(Lerp(0.0, 2.0, 8.0), 2.0);
	EXPECT_DOUBLE_EQ(Lerp(1.0, 2.0, 8.0), 8.0);
}
TEST(LerpTest, Midpoint) {
	EXPECT_DOUBLE_EQ(Lerp(0.5, 0.0, 10.0), 5.0);
}

// ---------------------------------------------------------------------------
// Clamp
// ---------------------------------------------------------------------------
TEST(ClampTest, BelowLow)  { EXPECT_EQ(Clamp(1,   5, 10), 5);  }
TEST(ClampTest, AboveHigh) { EXPECT_EQ(Clamp(20,  5, 10), 10); }
TEST(ClampTest, Inside)    { EXPECT_EQ(Clamp(7,   5, 10), 7);  }

// ---------------------------------------------------------------------------
// Mod (integer)
// ---------------------------------------------------------------------------
TEST(ModTest, Positive)   { EXPECT_EQ(Mod(10, 3), 1);  }
TEST(ModTest, Zero)       { EXPECT_EQ(Mod(9,  3), 0);  }
TEST(ModTest, NegativeA)  { EXPECT_EQ(Mod(-1, 4), 3);  }

// ---------------------------------------------------------------------------
// Sqr / Pow
// ---------------------------------------------------------------------------
TEST(SqrTest, Double) { EXPECT_DOUBLE_EQ(Sqr(3.0), 9.0); }
TEST(SqrTest, Float)  { EXPECT_FLOAT_EQ(Sqr(4.f), 16.f); }

TEST(PowTest, Zero)   { EXPECT_FLOAT_EQ(Pow<0>(5.f), 1.f);  }
TEST(PowTest, One)    { EXPECT_FLOAT_EQ(Pow<1>(5.f), 5.f);  }
TEST(PowTest, Two)    { EXPECT_FLOAT_EQ(Pow<2>(3.f), 9.f);  }
TEST(PowTest, NegOne) { EXPECT_FLOAT_EQ(Pow<-1>(4.f), 0.25f); }

// ---------------------------------------------------------------------------
// Radians / Degrees
// ---------------------------------------------------------------------------
TEST(AngleConvTest, Radians) { EXPECT_NEAR(Radians(180.0), kPi, 1e-12); }
TEST(AngleConvTest, Degrees) { EXPECT_NEAR(Degrees(kPi), 180.0, 1e-10); }
TEST(AngleConvTest, RoundTrip) {
	EXPECT_NEAR(Degrees(Radians(45.0)), 45.0, 1e-10);
}

// ---------------------------------------------------------------------------
// SafeSqrt / SafeASin / SafeACos
// ---------------------------------------------------------------------------
TEST(SafeMathTest, SafeSqrtNegative) { EXPECT_DOUBLE_EQ(SafeSqrt(-1.0), 0.0); }
TEST(SafeMathTest, SafeSqrtPositive) { EXPECT_NEAR(SafeSqrt(4.0), 2.0, 1e-12); }
TEST(SafeMathTest, SafeASinClamp)    { EXPECT_NEAR(SafeASin(1.1f), SafeASin(1.f), 1e-6f); }
TEST(SafeMathTest, SafeACosClamp)    { EXPECT_NEAR(SafeACos(-2.f), SafeACos(-1.f), 1e-6f); }

// ---------------------------------------------------------------------------
// SinXOverX / Sinc / WindowedSinc
// ---------------------------------------------------------------------------
TEST(SincTest, SinXOverXAtZero)  { EXPECT_DOUBLE_EQ(SinXOverX(0.0), 1.0); }
TEST(SincTest, SincAtZero)       { EXPECT_DOUBLE_EQ(Sinc(0.0), 1.0); }
TEST(SincTest, SincAtOne)        {
	// Sinc(1) = sin(Pi)/Pi ~= 0
	EXPECT_NEAR(Sinc(1.0), 0.0, 1e-14);
}
TEST(SincTest, WindowedSincOutsideRadius) {
	EXPECT_DOUBLE_EQ(WindowedSinc(5.0, 4.0, 3.0), 0.0);
}
TEST(SincTest, WindowedSincAtZero) {
	EXPECT_NEAR(WindowedSinc(0.0, 4.0, 3.0), 1.0, 1e-12);
}

// ---------------------------------------------------------------------------
// EvaluatePolynomial
// ---------------------------------------------------------------------------
TEST(EvalPolyTest, Constant)  { EXPECT_DOUBLE_EQ(EvaluatePolynomial(2.0, 7.0), 7.0); }
TEST(EvalPolyTest, Linear)    { EXPECT_DOUBLE_EQ(EvaluatePolynomial(3.0, 1.0, 2.0), 1.0 + 3.0 * 2.0); }
TEST(EvalPolyTest, Quadratic) {
	// c0 + t*(c1 + t*c2) = 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
	EXPECT_DOUBLE_EQ(EvaluatePolynomial(2.0, 1.0, 2.0, 3.0), 17.0);
}

// ---------------------------------------------------------------------------
// FastExp
// ---------------------------------------------------------------------------
TEST(FastExpTest, Zero)     { EXPECT_NEAR(FastExp(0.f), 1.f, 1e-5f); }
TEST(FastExpTest, One)      { EXPECT_NEAR(FastExp(1.f), std::exp(1.f), 1e-3f); }
TEST(FastExpTest, Negative) { EXPECT_NEAR(FastExp(-2.f), std::exp(-2.f), 1e-4f); }
TEST(FastExpTest, Large)    { EXPECT_FALSE(std::isinf(FastExp(85.f))); }
TEST(FastExpTest, VeryNeg)  { EXPECT_FLOAT_EQ(FastExp(-200.f), 0.f); }

// ---------------------------------------------------------------------------
// SmoothStep
// ---------------------------------------------------------------------------
TEST(SmoothStepTest, BelowA)  { EXPECT_DOUBLE_EQ(SmoothStep(-1.0, 0.0, 1.0), 0.0); }
TEST(SmoothStepTest, AboveB)  { EXPECT_DOUBLE_EQ(SmoothStep(2.0,  0.0, 1.0), 1.0); }
TEST(SmoothStepTest, Midpoint){ EXPECT_DOUBLE_EQ(SmoothStep(0.5,  0.0, 1.0), 0.5); }
TEST(SmoothStepTest, EqualAB) { EXPECT_DOUBLE_EQ(SmoothStep(-1.0, 0.0, 0.0), 0.0); }

// ---------------------------------------------------------------------------
// Gaussian / GaussianIntegral
// ---------------------------------------------------------------------------
TEST(GaussianTest, PeakAtMu) {
	double g0 = Gaussian(0.0, 0.0, 1.0);
	double g1 = Gaussian(1.0, 0.0, 1.0);
	EXPECT_GT(g0, g1);
}
TEST(GaussianIntegralTest, FullIntegral) {
	double total = GaussianIntegral(-100.0, 100.0, 0.0, 1.0);
	EXPECT_NEAR(total, 1.0, 1e-6);
}

// ---------------------------------------------------------------------------
// Logistic / LogisticCDF / TrimmedLogistic
// ---------------------------------------------------------------------------
TEST(LogisticTest, Symmetric)  { EXPECT_NEAR(Logistic(0.0, 1.0), 0.25, 1e-12); }
TEST(LogisticCDFTest, AtZero)  { EXPECT_NEAR(LogisticCDF(0.0, 1.0), 0.5, 1e-12); }
TEST(TrimmedLogisticTest, Integrates) {
	// Trimmed over [-1, 1] with s=1: should integrate to ~1 numerically
	// Just check it's positive and finite
	double v = TrimmedLogistic(0.0, 1.0, -1.0, 1.0);
	EXPECT_GT(v, 0.0);
	EXPECT_FALSE(std::isinf(v));
}

// ---------------------------------------------------------------------------
// ErfInv
// ---------------------------------------------------------------------------
TEST(ErfInvTest, Zero)     { EXPECT_NEAR(ErfInv(0.f), 0.f, 1e-5f); }
TEST(ErfInvTest, RoundTrip) {
	float x = 0.5f;
	float y = std::erf(ErfInv(x));
	EXPECT_NEAR(y, x, 1e-5f);
}

// ---------------------------------------------------------------------------
// I0 / LogI0
// ---------------------------------------------------------------------------
TEST(BesselTest, LogI0AtZero){ EXPECT_NEAR(LogI0(0.0), 0.0, 1e-10); }
TEST(BesselTest, LogI0Large) {
	// For large x, LogI0(x) ~= x (dominated by x term)
	double val = LogI0(20.0);
	EXPECT_GT(val, 15.0);
}

// ---------------------------------------------------------------------------
// Integer math: Log2 / Log2Int / Log4Int / IsPowerOf2 / RoundUpPow2
// ---------------------------------------------------------------------------
TEST(IntMathTest, Log2) { EXPECT_NEAR(Log2(8.0), 3.0, 1e-12); }

TEST(IntMathTest, Log2IntU32) {
	EXPECT_EQ(Log2Int(uint32_t(1)),  0);
	EXPECT_EQ(Log2Int(uint32_t(2)),  1);
	EXPECT_EQ(Log2Int(uint32_t(8)),  3);
	EXPECT_EQ(Log2Int(uint32_t(16)), 4);
}
TEST(IntMathTest, Log4Int) {
	EXPECT_EQ(Log4Int(uint32_t(1)),  0);
	EXPECT_EQ(Log4Int(uint32_t(4)),  1);
	EXPECT_EQ(Log4Int(uint32_t(16)), 2);
}
TEST(IntMathTest, IsPowerOf2) {
	EXPECT_TRUE(IsPowerOf2(1));
	EXPECT_TRUE(IsPowerOf2(4));
	EXPECT_FALSE(IsPowerOf2(5));
	EXPECT_FALSE(IsPowerOf2(0));
}
TEST(IntMathTest, RoundUpPow2_32) {
	EXPECT_EQ(RoundUpPow2(int32_t(1)),  1);
	EXPECT_EQ(RoundUpPow2(int32_t(5)),  8);
	EXPECT_EQ(RoundUpPow2(int32_t(8)),  8);
	EXPECT_EQ(RoundUpPow2(int32_t(9)),  16);
}
TEST(IntMathTest, RoundUpPow2_64) {
	EXPECT_EQ(RoundUpPow2(int64_t(1)),   int64_t(1));
	EXPECT_EQ(RoundUpPow2(int64_t(100)), int64_t(128));
}

// ---------------------------------------------------------------------------
// FindInterval
// ---------------------------------------------------------------------------
TEST(FindIntervalTest, LookupInArray) {
	// nodes: {0, 1, 2, 3, 4}
	const double nodes[] = {0, 1, 2, 3, 4};
	size_t n = 5;
	// 1.5 should fall in interval [1,2] -> index 1
	size_t idx = FindInterval(n, [&](size_t i) { return nodes[i] <= 1.5; });
	EXPECT_EQ(idx, size_t(1));
}
TEST(FindIntervalTest, AtLastNode) {
	const double nodes[] = {0, 1, 2, 3, 4};
	size_t n = 5;
	size_t idx = FindInterval(n, [&](size_t i) { return nodes[i] <= 3.9; });
	EXPECT_EQ(idx, size_t(3));
}

// ---------------------------------------------------------------------------
// NewtonBisection
// ---------------------------------------------------------------------------
TEST(NewtonBisectionTest, SqrtOf2) {
	// Find root of f(x) = x^2 - 2, f'(x) = 2x in [1, 2]
	auto f = [](double x) -> std::pair<double,double> {
		return {x * x - 2.0, 2.0 * x};
	};
	double root = NewtonBisection(1.0, 2.0, f, 1e-10, 1e-10);
	EXPECT_NEAR(root, std::sqrt(2.0), 1e-8);
}
TEST(NewtonBisectionTest, LinearRoot) {
	// f(x) = x - 0.3, root at x=0.3
	auto f = [](double x) -> std::pair<double,double> { return {x - 0.3, 1.0}; };
	double root = NewtonBisection(0.0, 1.0, f, 1e-10, 1e-10);
	EXPECT_NEAR(root, 0.3, 1e-8);
}

// ---------------------------------------------------------------------------
// PermutationElement
// ---------------------------------------------------------------------------
TEST(PermutationElementTest, Bijection) {
	// For l=8, each seed p should produce a bijection on [0,8)
	uint32_t l = 8, p = 42;
	std::vector<int> seen(l, 0);
	for (uint32_t i = 0; i < l; ++i) {
		int mapped = PermutationElement(i, l, p);
		EXPECT_GE(mapped, 0);
		EXPECT_LT(mapped, static_cast<int>(l));
		seen[mapped]++;
	}
	for (uint32_t i = 0; i < l; ++i)
		EXPECT_EQ(seen[i], 1) << "index " << i << " appears " << seen[i] << " times";
}
TEST(PermutationElementTest, DifferentSeedsDiffer) {
	// Same input, different seeds should (almost always) differ
	int a = PermutationElement(3, 16, 1);
	int b = PermutationElement(3, 16, 2);
	EXPECT_NE(a, b);
}

// ---------------------------------------------------------------------------
// Gaussian uses FastExp (pbrt-v4 alignment)
// ---------------------------------------------------------------------------
TEST(GaussianTest, UsesApproxExp) {
	// Gaussian(x) uses FastExp internally (like pbrt-v4).
	// For x near 0, FastExp is very accurate: within 1e-4 of std::exp.
	double g = Gaussian(0.5, 0.0, 1.0);
	double g_ref = 1.0 / std::sqrt(2.0 * kPi) * std::exp(-0.125);
	EXPECT_NEAR(g, g_ref, 1e-4);
}
TEST(GaussianTest, MonotonicallyDecreasing) {
	// Gaussian falls off from peak at mu
	for (int i = 1; i <= 5; ++i) {
		EXPECT_GT(Gaussian(double(i - 1), 0.0, 1.0),
				  Gaussian(double(i),     0.0, 1.0));
	}
}

// ---------------------------------------------------------------------------
// I0 accuracy (pbrt-v4 series, 10 terms)
// ---------------------------------------------------------------------------
TEST(BesselTest, I0KnownValues) {
	// Known values from Wolfram Alpha / standard tables
	EXPECT_NEAR(I0(0.0), 1.0,      1e-10);
	EXPECT_NEAR(I0(1.0), 1.266066, 1e-5);   // I_0(1) = 1.2660658...
	EXPECT_NEAR(I0(2.0), 2.279585, 1e-4);   // I_0(2) = 2.2795853...
}
TEST(BesselTest, LogI0KnownValues) {
	EXPECT_NEAR(LogI0(0.0),  0.0,          1e-10);
	EXPECT_NEAR(LogI0(1.0),  std::log(I0(1.0)), 1e-10);
	// Large x: asymptotic formula should be close
	EXPECT_NEAR(LogI0(15.0), 15.0 + 0.5 * (-std::log(2.0 * kPi) +
				std::log(1.0 / 15.0) + 1.0 / (8.0 * 15.0)), 1e-4);
}

// ---------------------------------------------------------------------------
// Log2Int round-to-nearest (pbrt-v4 alignment)
// ---------------------------------------------------------------------------
TEST(IntMathTest, Log2IntFloat_RoundNearest) {
	// 2^1 = 2, 2^1.5 ~= 2.828 is the midpoint -> values below round to 1, above to 2
	EXPECT_EQ(Log2Int(2.0f),   1);
	EXPECT_EQ(Log2Int(2.5f),   1);   // below 2^1.5 ~ 2.828 -> rounds to 1
	EXPECT_EQ(Log2Int(3.0f),   2);   // above 2^1.5 ~ 2.828 -> rounds to 2
	EXPECT_EQ(Log2Int(4.0f),   2);
}
TEST(IntMathTest, Log2IntDouble_RoundNearest) {
	EXPECT_EQ(Log2Int(1.0),  0);
	EXPECT_EQ(Log2Int(2.0),  1);
	EXPECT_EQ(Log2Int(4.0),  2);
	EXPECT_EQ(Log2Int(0.5),  -1);
}

// ---------------------------------------------------------------------------
// PermutationElement: uniform distribution (pbrt-v4 PermutationElement/Uniform)
// For every starting index i in [0, n), the mapped values should be
// approximately uniformly distributed across [0, n) over many seeds.
// ---------------------------------------------------------------------------
TEST(PermutationElementTest, Uniform) {
	for (int n : {3, 5, 9, 16}) {
		std::vector<int> count(n * n, 0);
		const int numIters = 5000 * n;
		for (int seed = 0; seed < numIters; ++seed) {
			for (int i = 0; i < n; ++i) {
				int ip = PermutationElement(static_cast<uint32_t>(i),
											static_cast<uint32_t>(n),
											static_cast<uint32_t>(seed * 1013904223u + 1664525u));
				ASSERT_GE(ip, 0);
				ASSERT_LT(ip, n);
				++count[ip * n + i];
			}
		}
		// Expect each (i -> j) mapping to appear numIters/n times, ±5%
		const float tol = 0.05f;
		const int lo = static_cast<int>((1.f - tol) * numIters / n);
		const int hi = static_cast<int>((1.f + tol) * numIters / n + 1);
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				int c = count[j * n + i];
				EXPECT_GE(c, lo) << "n=" << n << " i=" << i << " j=" << j;
				EXPECT_LE(c, hi) << "n=" << n << " i=" << i << " j=" << j;
			}
		}
	}
}

// PermutationElement: delta distribution (pbrt-v4 PermutationElement/UniformDelta)
// For every i, the delta (ip - i) mod n should be approximately uniform over seeds.
TEST(PermutationElementTest, UniformDelta) {
	for (int n : {3, 5, 9, 16}) {
		std::vector<int> count(n * n, 0);
		const int numIters = 5000 * n;
		for (int seed = 0; seed < numIters; ++seed) {
			for (int i = 0; i < n; ++i) {
				int ip = PermutationElement(static_cast<uint32_t>(i),
											static_cast<uint32_t>(n),
											static_cast<uint32_t>(seed * 1013904223u + 1664525u));
				int delta = ip - i;
				if (delta < 0) delta += n;
				ASSERT_GE(delta, 0);
				ASSERT_LT(delta, n);
				++count[delta * n + i];
			}
		}
		const float tol = 0.05f;
		const int lo = static_cast<int>((1.f - tol) * numIters / n);
		const int hi = static_cast<int>((1.f + tol) * numIters / n + 1);
		for (int i = 0; i < n; ++i) {
			for (int d = 0; d < n; ++d) {
				int c = count[d * n + i];
				EXPECT_GE(c, lo) << "n=" << n << " i=" << i << " delta=" << d;
				EXPECT_LE(c, hi) << "n=" << n << " i=" << i << " delta=" << d;
			}
		}
	}
}
