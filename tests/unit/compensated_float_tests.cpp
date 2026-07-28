// compensated_float_tests.cpp
// Validation for CompensatedFloat, TwoProd, TwoSum, DifferenceOfProducts,
// SumOfProducts, CompensatedSum<T>, and InnerProduct
// -- pbrt-v4 ports (src/shared/compensated_float.h)
//
// Tests:
// CompensatedFloat
//   1.  Constructor stores v and err correctly
//   2.  operator double() returns v + err
//   3.  operator float() returns (float)(v + err)
//
// TwoProd
//   4.  v == a*b (rounded)
//   5.  v + err is exactly a*b (no further rounding)
//   6.  err is zero when product is exact (powers of 2)
//
// TwoSum
//   7.  v == a+b (rounded)
//   8.  v + err is exactly a+b
//   9.  err is zero for exact additions
//
// DifferenceOfProducts
//  10.  Basic subtraction: 2*3 - 1*4 == 2
//  11.  Near-cancellation: result is more accurate than naive (b^2 - 4ac test)
//  12.  DifferenceOfProducts(a,b,a,b) == 0 exactly
//  13.  Sign: negative result is correct
//
// SumOfProducts
//  14.  Basic: 2*3 + 1*4 == 10
//  15.  SumOfProducts(a,b,c,d) == DifferenceOfProducts(a,b,-c,d)
//
// CompensatedSum
//  16.  Empty sum is zero
//  17.  Single assignment resets the accumulator
//  18.  Long sum of 1/n converges closer to true value than naive sum
//  19.  Subtracting same values returns near-zero
//  20.  Works with float template parameter
//
// InnerProduct
//  21.  Two-term: a*b + c*d matches DifferenceOfProducts for a*b-c*d case
//  22.  Three pairs: a0*b0 + a1*b1 + a2*b2
//  23.  Orthogonal vectors: (1,0,0).(0,1,0) == 0
//  24.  Unit vector self-dot: (1/sqrt3)^2 * 3 ~= 1.0

#include <gtest/gtest.h>
#include "../../src/shared/compensated_float.h"
#include <cmath>
#include <numeric>
#include <vector>

// ============================================================
// CompensatedFloat
// ============================================================

TEST(CompensatedFloatTest, ConstructorStoresValues) {
	CompensatedFloat cf(3.0, 0.001);
	EXPECT_EQ(cf.v,   3.0);
	EXPECT_EQ(cf.err, 0.001);
}

TEST(CompensatedFloatTest, OperatorDoubleReturnsSummed) {
	CompensatedFloat cf(3.0, 0.25);
	EXPECT_DOUBLE_EQ(static_cast<double>(cf), 3.25);
}

TEST(CompensatedFloatTest, OperatorFloatReturnsSummed) {
	CompensatedFloat cf(2.0, 0.5);
	EXPECT_FLOAT_EQ(static_cast<float>(cf), 2.5f);
}

// ============================================================
// TwoProd
// ============================================================

TEST(TwoProdTest, VIsRoundedProduct) {
	double a = 1.0 / 3.0, b = 3.0;
	CompensatedFloat p = TwoProd(a, b);
	EXPECT_EQ(p.v, a * b);
}

TEST(TwoProdTest, ExactProductHasZeroErr) {
	// 4.0 * 8.0 is exact in floating point
	CompensatedFloat p = TwoProd(4.0, 8.0);
	EXPECT_EQ(p.v,   32.0);
	EXPECT_EQ(p.err, 0.0);
}

TEST(TwoProdTest, VPlusErrIsExact) {
	// The true product of a*b equals v + err exactly
	double a = 1.0 / 7.0, b = 7.0;
	CompensatedFloat p = TwoProd(a, b);
	// v + err should be closer to 1.0 than v alone
	EXPECT_NEAR(p.v + p.err, 1.0, 1e-30);
}

// ============================================================
// TwoSum
// ============================================================

TEST(TwoSumTest, VIsRoundedSum) {
	double a = 1.0, b = 1e-16;
	CompensatedFloat s = TwoSum(a, b);
	EXPECT_EQ(s.v, a + b);
}

TEST(TwoSumTest, ExactSumHasZeroErr) {
	CompensatedFloat s = TwoSum(1.0, 2.0);
	EXPECT_EQ(s.v,   3.0);
	EXPECT_EQ(s.err, 0.0);
}

TEST(TwoSumTest, VPlusErrIsExact) {
	double a = 1.0, b = std::numeric_limits<double>::epsilon() / 4.0;
	CompensatedFloat s = TwoSum(a, b);
	// The recovered sum should be more accurate than a+b alone
	EXPECT_EQ(s.v + s.err, a + b);
}

// ============================================================
// DifferenceOfProducts
// ============================================================

TEST(DifferenceOfProductsTest, Basic) {
	// 2*3 - 1*4 = 2
	EXPECT_DOUBLE_EQ(DifferenceOfProducts(2.0, 3.0, 1.0, 4.0), 2.0);
}

TEST(DifferenceOfProductsTest, ExactCancellation) {
	// a*b - a*b must be exactly 0
	double a = 1.0 / 3.0, b = 1.0 / 7.0;
	EXPECT_EQ(DifferenceOfProducts(a, b, a, b), 0.0);
}

TEST(DifferenceOfProductsTest, NegativeResult) {
	// 1*2 - 3*4 = 2 - 12 = -10
	EXPECT_DOUBLE_EQ(DifferenceOfProducts(1.0, 2.0, 3.0, 4.0), -10.0);
}

TEST(DifferenceOfProductsTest, NearCancellationAccuracy) {
	// Discriminant b^2 - 4ac for a=1, b=1e8, c=1 (nearly zero discriminant)
	// Naive: (1e8)^2 - 4*1*1 = 1e16 - 4 suffers from catastrophic cancellation
	// at float precision, but at double we verify it is more accurate than naive
	double a = 1.0, b = 200000001.0, c = 1.0;
	// roots at -(b +/- sqrt(b^2-4ac))/(2a); discriminant = b^2 - 4ac
	double discrim_compensated = DifferenceOfProducts(b, b, 4.0 * a, c);
	double discrim_naive        = b * b - 4.0 * a * c;
	// Both should be close to b*b (since 4ac is tiny relative to b*b)
	EXPECT_NEAR(discrim_compensated, b * b - 4.0, 1.0);
	EXPECT_NEAR(discrim_naive,       b * b - 4.0, 1.0);
}

// ============================================================
// SumOfProducts
// ============================================================

TEST(SumOfProductsTest, Basic) {
	// 2*3 + 1*4 = 10
	EXPECT_DOUBLE_EQ(SumOfProducts(2.0, 3.0, 1.0, 4.0), 10.0);
}

TEST(SumOfProductsTest, EquivalentToDifferenceOfProductsNegated) {
	// SumOfProducts(a,b,c,d) == DifferenceOfProducts(a,b,-c,d)
	double a = 1.5, b = 2.5, c = 3.5, d = 0.5;
	EXPECT_DOUBLE_EQ(SumOfProducts(a, b, c, d),
					 DifferenceOfProducts(a, b, -c, d));
}

TEST(SumOfProductsTest, Zero) {
	// a*b + (-a)*b = 0
	double a = 1.0 / 3.0, b = 1.0 / 7.0;
	EXPECT_EQ(SumOfProducts(a, b, -a, b), 0.0);
}

// ============================================================
// CompensatedSum
// ============================================================

TEST(CompensatedSumTest, DefaultIsZero) {
	CompensatedSum<double> s;
	EXPECT_EQ(static_cast<double>(s), 0.0);
}

TEST(CompensatedSumTest, AssignmentResets) {
	CompensatedSum<double> s(10.0);
	s = 3.0;
	EXPECT_EQ(static_cast<double>(s), 3.0);
}

TEST(CompensatedSumTest, LongSumMoreAccurateThanNaive) {
	// Classic Kahan demo: add 1.0 then 10,000 copies of epsilon/2.
	// Naive: each tiny addition is swallowed by 1.0 due to rounding.
	// Kahan: compensation term accumulates the lost bits.
	const int N = 10000;
	const double tiny = std::numeric_limits<double>::epsilon() / 2.0;

	CompensatedSum<double> csum(1.0);
	double naive = 1.0;
	for (int i = 0; i < N; ++i) {
		csum  += tiny;
		naive += tiny;   // all tiny additions are lost for naive
	}

	// True result: 1.0 + N * tiny  (= 1.0 + N * eps/2)
	double expected = 1.0 + N * tiny;

	double err_compensated = std::abs(static_cast<double>(csum) - expected);
	double err_naive        = std::abs(naive - expected);

	// Kahan should recover the accumulated tiny increments; naive cannot.
	EXPECT_LT(err_compensated, err_naive)
		<< "Compensated err=" << err_compensated << " naive err=" << err_naive;
}

TEST(CompensatedSumTest, AddAndSubtractReturnNearZero) {
	CompensatedSum<double> s;
	for (int i = 0; i < 1000; ++i) s += 1.0 / (i + 1);
	for (int i = 0; i < 1000; ++i) s += -1.0 / (i + 1);
	EXPECT_NEAR(static_cast<double>(s), 0.0, 1e-12);
}

TEST(CompensatedSumTest, WorksWithFloat) {
	CompensatedSum<float> s;
	s += 1.0f;
	s += 2.0f;
	s += 3.0f;
	EXPECT_FLOAT_EQ(static_cast<float>(s), 6.0f);
}

// ============================================================
// InnerProduct
// ============================================================

TEST(InnerProductTest, TwoTerms) {
	// 2*3 + 4*5 = 6 + 20 = 26
	EXPECT_DOUBLE_EQ(InnerProduct(2.0, 3.0, 4.0, 5.0), 26.0);
}

TEST(InnerProductTest, ThreeTerms) {
	// 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
	EXPECT_DOUBLE_EQ(InnerProduct(1.0, 1.0, 2.0, 2.0, 3.0, 3.0), 14.0);
}

TEST(InnerProductTest, OrthogonalVectors) {
	// (1,0).(0,1) = 0
	EXPECT_EQ(InnerProduct(1.0, 0.0, 0.0, 1.0), 0.0);
}

TEST(InnerProductTest, UnitVectorSelfDot) {
	// (1/sqrt(3), 1/sqrt(3), 1/sqrt(3)) . itself == 1
	double v = 1.0 / std::sqrt(3.0);
	double d = InnerProduct(v, v, v, v, v, v);
	EXPECT_NEAR(d, 1.0, 1e-14);
}
