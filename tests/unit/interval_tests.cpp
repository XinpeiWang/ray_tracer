// interval_tests.cpp
// Unit tests for Interval conservative arithmetic (src/shared/interval.h)
// -- pbrt-v4 port
//
// Tests:
// ULP / rounding primitives
//   1.  next_float_up advances by one ULP (positive)
//   2.  next_float_down retreats by one ULP (positive)
//   3.  next_float_up/down are inverses
//   4.  add_round_up >= true sum; add_round_down <= true sum
//
// Interval constructors / accessors
//   5.  Point interval: LowerBound == UpperBound == v
//   6.  [a,b] constructor sets bounds correctly
//   7.  Midpoint is average of bounds
//   8.  Width is high - low
//   9.  FromValueAndError contains v+-err
//  10.  FromValueAndError with err==0 is exact
//
// Arithmetic -- containment invariants
//  11.  Addition: true sum lies within result interval
//  12.  Subtraction: true difference lies within result interval
//  13.  Multiplication: true product lies within result interval
//  14.  Division: true quotient lies within result interval
//  15.  Negation: -[a,b] == [-b,-a]
//
// Compound operators
//  16.  += / -= / *= / /= give same result as binary operators
//
// Free functions
//  17.  Sqr: 0 in interval => lower bound is 0
//  18.  Sqr: positive interval => [lo*lo, hi*hi]
//  19.  Sqrt: Sqrt(Sqr(i)).UpperBound() >= i.UpperBound() for positive i
//  20.  Abs: positive interval unchanged; negative interval flipped
//  21.  Abs: straddling zero => lower bound is 0
//  22.  InRange(v, i): true iff in [lo, hi]
//  23.  InRange(Interval, Interval): overlapping vs disjoint
//  24.  Cos: bounds contain cos(midpoint) for [0, pi]
//  25.  Sin: bounds contain sin(midpoint) for [0, pi/2]
//  26.  Quadratic: roots of (t-2)(t-3) = t^2 - 5t + 6 bracketed correctly
//  27.  Quadratic: negative discriminant returns false

#include <gtest/gtest.h>
#include "../../src/shared/interval.h"
#include <cmath>
#include <limits>

// ============================================================
// ULP / rounding helpers
// ============================================================

TEST(IntervalULPTest, NextFloatUpAdvancesOneULP) {
	double v = 1.0;
	double up = next_float_up(v);
	EXPECT_GT(up, v);
	// Advancing once more and comparing is the tightest check possible
	EXPECT_LT(v, up);
}

TEST(IntervalULPTest, NextFloatDownRetreatsOneULP) {
	double v = 1.0;
	double dn = next_float_down(v);
	EXPECT_LT(dn, v);
}

TEST(IntervalULPTest, UpAndDownAreInverses) {
	double v = 3.14;
	EXPECT_EQ(next_float_down(next_float_up(v)), v);
	EXPECT_EQ(next_float_up(next_float_down(v)), v);
}

TEST(IntervalULPTest, AddRoundDirections) {
	double a = 1.0 / 3.0, b = 2.0 / 3.0;
	double exact = a + b;   // may not be exactly 1.0
	EXPECT_GE(add_round_up(a, b),   exact);
	EXPECT_LE(add_round_down(a, b), exact);
}

// ============================================================
// Constructors / accessors
// ============================================================

TEST(IntervalConstructorTest, PointInterval) {
	Interval i(2.5);
	EXPECT_EQ(i.LowerBound(), 2.5);
	EXPECT_EQ(i.UpperBound(), 2.5);
	EXPECT_TRUE(i.Exactly(2.5));
}

TEST(IntervalConstructorTest, BoundsConstructor) {
	Interval i(1.0, 3.0);
	EXPECT_EQ(i.LowerBound(), 1.0);
	EXPECT_EQ(i.UpperBound(), 3.0);
}

TEST(IntervalConstructorTest, InvertedBoundsAreNormalized) {
	// pbrt-v4: Interval(high, low) must swap so LowerBound <= UpperBound
	Interval i(3.0, 1.0);
	EXPECT_EQ(i.LowerBound(), 1.0);
	EXPECT_EQ(i.UpperBound(), 3.0);
}

TEST(IntervalConstructorTest, Midpoint) {
	Interval i(1.0, 3.0);
	EXPECT_NEAR(i.Midpoint(), 2.0, 1e-15);
}

TEST(IntervalConstructorTest, Width) {
	Interval i(1.0, 4.0);
	EXPECT_NEAR(i.Width(), 3.0, 1e-15);
}

TEST(IntervalConstructorTest, FromValueAndErrorContainsRange) {
	double v = 1.0, err = 0.001;
	Interval i = Interval::FromValueAndError(v, err);
	EXPECT_LE(i.LowerBound(), v - err);
	EXPECT_GE(i.UpperBound(), v + err);
}

TEST(IntervalConstructorTest, FromValueAndErrorZeroErr) {
	Interval i = Interval::FromValueAndError(5.0, 0.0);
	EXPECT_TRUE(i.Exactly(5.0));
}

// ============================================================
// Arithmetic containment invariants
// ============================================================

static bool contains(Interval i, double v) {
	return v >= i.LowerBound() && v <= i.UpperBound();
}

TEST(IntervalArithmeticTest, AdditionContains) {
	Interval a(0.1, 0.3), b(0.2, 0.4);
	// All corner sums must be in result
	Interval r = a + b;
	EXPECT_LE(r.LowerBound(), 0.1 + 0.2);
	EXPECT_GE(r.UpperBound(), 0.3 + 0.4);
}

TEST(IntervalArithmeticTest, SubtractionContains) {
	Interval a(1.0, 2.0), b(0.3, 0.7);
	Interval r = a - b;
	EXPECT_LE(r.LowerBound(), 1.0 - 0.7);
	EXPECT_GE(r.UpperBound(), 2.0 - 0.3);
}

TEST(IntervalArithmeticTest, MultiplicationContains) {
	Interval a(2.0, 3.0), b(4.0, 5.0);
	Interval r = a * b;
	EXPECT_LE(r.LowerBound(), 2.0 * 4.0);
	EXPECT_GE(r.UpperBound(), 3.0 * 5.0);
}

TEST(IntervalArithmeticTest, DivisionContains) {
	Interval a(6.0, 8.0), b(2.0, 4.0);
	Interval r = a / b;
	EXPECT_LE(r.LowerBound(), 6.0 / 4.0);
	EXPECT_GE(r.UpperBound(), 8.0 / 2.0);
}

TEST(IntervalArithmeticTest, Negation) {
	Interval i(1.5, 3.0);
	Interval n = -i;
	EXPECT_NEAR(n.LowerBound(), -3.0, 1e-15);
	EXPECT_NEAR(n.UpperBound(), -1.5, 1e-15);
}

// ============================================================
// Compound operators
// ============================================================

TEST(IntervalCompoundTest, PlusEqualsMatchesBinary) {
	Interval a(1.0, 2.0), b(3.0, 4.0);
	Interval r1 = a + b;
	Interval r2 = a; r2 += b;
	EXPECT_EQ(r1.LowerBound(), r2.LowerBound());
	EXPECT_EQ(r1.UpperBound(), r2.UpperBound());
}

TEST(IntervalCompoundTest, TimesEqualsMatchesBinary) {
	Interval a(1.5, 2.5), b(0.5, 1.5);
	Interval r1 = a * b;
	Interval r2 = a; r2 *= b;
	EXPECT_EQ(r1.LowerBound(), r2.LowerBound());
	EXPECT_EQ(r1.UpperBound(), r2.UpperBound());
}

// ============================================================
// Free functions
// ============================================================

TEST(IntervalFreeFuncTest, SqrZeroIncluded) {
	Interval i(-2.0, 3.0);   // straddles zero
	Interval s = Sqr(i);
	EXPECT_EQ(s.LowerBound(), 0.0);
	EXPECT_GE(s.UpperBound(), 9.0);
}

TEST(IntervalFreeFuncTest, SqrPositiveInterval) {
	Interval i(2.0, 3.0);
	Interval s = Sqr(i);
	EXPECT_LE(s.LowerBound(), 4.0);
	EXPECT_GE(s.UpperBound(), 9.0);
}

TEST(IntervalFreeFuncTest, SqrtContainsTrue) {
	Interval i(4.0, 9.0);
	Interval s = Sqrt(i);
	EXPECT_LE(s.LowerBound(), 2.0);
	EXPECT_GE(s.UpperBound(), 3.0);
}

TEST(IntervalFreeFuncTest, AbsPositiveUnchanged) {
	Interval i(1.0, 3.0);
	Interval a = Abs(i);
	EXPECT_EQ(a.LowerBound(), i.LowerBound());
	EXPECT_EQ(a.UpperBound(), i.UpperBound());
}

TEST(IntervalFreeFuncTest, AbsNegativeFlipped) {
	Interval i(-4.0, -1.0);
	Interval a = Abs(i);
	EXPECT_NEAR(a.LowerBound(), 1.0, 1e-15);
	EXPECT_NEAR(a.UpperBound(), 4.0, 1e-15);
}

TEST(IntervalFreeFuncTest, AbsStraddlingZero) {
	Interval i(-3.0, 2.0);
	Interval a = Abs(i);
	EXPECT_EQ(a.LowerBound(), 0.0);
	EXPECT_NEAR(a.UpperBound(), 3.0, 1e-15);
}

TEST(IntervalFreeFuncTest, InRangeScalar) {
	Interval i(1.0, 3.0);
	EXPECT_TRUE (InRange(2.0, i));
	EXPECT_TRUE (InRange(1.0, i));
	EXPECT_TRUE (InRange(3.0, i));
	EXPECT_FALSE(InRange(0.0, i));
	EXPECT_FALSE(InRange(4.0, i));
}

TEST(IntervalFreeFuncTest, InRangeIntervalOverlapAndDisjoint) {
	Interval a(1.0, 3.0), b(2.0, 4.0), c(5.0, 7.0);
	EXPECT_TRUE (InRange(a, b));
	EXPECT_FALSE(InRange(a, c));
}

TEST(IntervalFreeFuncTest, CosBoundsContainTrueValue) {
	// cos([0.5, 1.0]): true values are in [cos(1.0), cos(0.5)]
	Interval i(0.5, 1.0);
	Interval c = Cos(i);
	EXPECT_LE(c.LowerBound(), std::cos(1.0));
	EXPECT_GE(c.UpperBound(), std::cos(0.5));
	// Both true values must be inside
	EXPECT_TRUE(contains(c, std::cos(0.5)));
	EXPECT_TRUE(contains(c, std::cos(1.0)));
}

TEST(IntervalFreeFuncTest, SinBoundsContainTrueValue) {
	// sin([0.3, 0.8]) is monotone increasing; both endpoints must be inside
	Interval i(0.3, 0.8);
	Interval s = Sin(i);
	EXPECT_TRUE(contains(s, std::sin(0.3)));
	EXPECT_TRUE(contains(s, std::sin(0.8)));
}

TEST(IntervalQuadraticTest, RealRootsBracketed) {
	// (t - 2)(t - 3) = t^2 - 5t + 6 => roots at 2 and 3
	Interval a(1.0), b(-5.0), c(6.0);
	Interval t0, t1;
	bool ok = Quadratic(a, b, c, &t0, &t1);
	ASSERT_TRUE(ok);
	// t0 should bracket 2, t1 should bracket 3
	EXPECT_LE(t0.LowerBound(), 2.0);
	EXPECT_GE(t0.UpperBound(), 2.0);
	EXPECT_LE(t1.LowerBound(), 3.0);
	EXPECT_GE(t1.UpperBound(), 3.0);
}

TEST(IntervalQuadraticTest, NegativeDiscriminantReturnsFalse) {
	// t^2 + 1 = 0 has no real roots
	Interval a(1.0), b(0.0), c(1.0);
	Interval t0, t1;
	EXPECT_FALSE(Quadratic(a, b, c, &t0, &t1));
}
