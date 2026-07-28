// float_bits_tests.cpp -- Unit tests for src/shared/float_bits.h
// Mirrors pbrt-v4 util/float.h semantics and pbrt-v4 float_test.cpp.

#include <cmath>
#include <cstdint>
#include <limits>
#include "../../src/shared/float_bits.h"
#include <gtest/gtest.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
TEST(FloatBitsConstants, OneMinusEpsilonLessThanOne) {
	EXPECT_LT(OneMinusEpsilon, 1.0f);
	EXPECT_GT(OneMinusEpsilon, 0.9999f);
}
TEST(FloatBitsConstants, DoubleOneMinusEpsilonLessThanOne) {
	EXPECT_LT(DoubleOneMinusEpsilon, 1.0);
	EXPECT_GT(DoubleOneMinusEpsilon, 0.9999999999999998);
}
TEST(FloatBitsConstants, MachineEpsilonHalfULP) {
	// FloatMachineEpsilon = eps/2 = half the spacing between 1 and next float
	EXPECT_FLOAT_EQ(FloatMachineEpsilon, std::numeric_limits<float>::epsilon() * 0.5f);
	EXPECT_DOUBLE_EQ(DoubleMachineEpsilon, std::numeric_limits<double>::epsilon() * 0.5);
}

// ---------------------------------------------------------------------------
// IsNaN / IsInf / IsFinite
// ---------------------------------------------------------------------------
TEST(IsNaNTest, FloatNaN)    { EXPECT_TRUE(IsNaN(std::numeric_limits<float>::quiet_NaN())); }
TEST(IsNaNTest, FloatNormal) { EXPECT_FALSE(IsNaN(1.0f)); }
TEST(IsNaNTest, DoubleNaN)   { EXPECT_TRUE(IsNaN(std::numeric_limits<double>::quiet_NaN())); }
TEST(IsNaNTest, IntZero)     { EXPECT_FALSE(IsNaN(0)); }

TEST(IsInfTest, FloatInf)    { EXPECT_TRUE(IsInf(std::numeric_limits<float>::infinity())); }
TEST(IsInfTest, FloatNegInf) { EXPECT_TRUE(IsInf(-std::numeric_limits<float>::infinity())); }
TEST(IsInfTest, FloatNormal) { EXPECT_FALSE(IsInf(1.0f)); }
TEST(IsInfTest, IntValue)    { EXPECT_FALSE(IsInf(42)); }

TEST(IsFiniteTest, FloatNormal) { EXPECT_TRUE(IsFinite(1.0f)); }
TEST(IsFiniteTest, FloatNaN)    { EXPECT_FALSE(IsFinite(std::numeric_limits<float>::quiet_NaN())); }
TEST(IsFiniteTest, FloatInf)    { EXPECT_FALSE(IsFinite(std::numeric_limits<float>::infinity())); }
TEST(IsFiniteTest, IntValue)    { EXPECT_TRUE(IsFinite(7)); }

// ---------------------------------------------------------------------------
// FloatToBits / BitsToFloat -- round-trip
// ---------------------------------------------------------------------------
TEST(FloatBitsConvert, FloatRoundTrip) {
	float values[] = { 0.f, 1.f, -1.f, 3.14f, -0.f,
					   std::numeric_limits<float>::infinity(),
					   std::numeric_limits<float>::max() };
	for (float v : values) {
		uint32_t bits = FloatToBits(v);
		float back = BitsToFloat(bits);
		// Use memcmp for bit-exact equality (handles -0 vs +0 and NaN)
		EXPECT_EQ(std::memcmp(&v, &back, sizeof(v)), 0) << "v=" << v;
	}
}
TEST(FloatBitsConvert, DoubleRoundTrip) {
	double values[] = { 0., 1., -1., 3.14159265358979, -0.,
						std::numeric_limits<double>::infinity(),
						std::numeric_limits<double>::max() };
	for (double v : values) {
		uint64_t bits = FloatToBits(v);
		double back = BitsToFloat(bits);
		EXPECT_EQ(std::memcmp(&v, &back, sizeof(v)), 0) << "v=" << v;
	}
}
TEST(FloatBitsConvert, KnownFloatBits) {
	// 1.0f = 0x3F800000
	EXPECT_EQ(FloatToBits(1.0f), 0x3F800000u);
	EXPECT_FLOAT_EQ(BitsToFloat(0x3F800000u), 1.0f);
}
TEST(FloatBitsConvert, KnownDoubleBits) {
	// 1.0 = 0x3FF0000000000000
	EXPECT_EQ(FloatToBits(1.0), 0x3FF0000000000000ull);
	EXPECT_DOUBLE_EQ(BitsToFloat(0x3FF0000000000000ull), 1.0);
}

// ---------------------------------------------------------------------------
// Exponent / Significand / SignBit
// ---------------------------------------------------------------------------
TEST(FloatFieldsTest, ExponentFloat) {
	EXPECT_EQ(Exponent(1.0f), 0);
	EXPECT_EQ(Exponent(2.0f), 1);
	EXPECT_EQ(Exponent(0.5f), -1);
	EXPECT_EQ(Exponent(4.0f), 2);
}
TEST(FloatFieldsTest, ExponentDouble) {
	EXPECT_EQ(Exponent(1.0), 0);
	EXPECT_EQ(Exponent(2.0), 1);
	EXPECT_EQ(Exponent(0.5), -1);
}
TEST(FloatFieldsTest, SignificandFloat) {
	// 1.0f has zero significand (pure power of 2)
	EXPECT_EQ(Significand(1.0f), 0);
	// 1.5f = 1.1 binary: significand = bit 22 set = 0x400000
	EXPECT_EQ(Significand(1.5f), 1 << 22);
}
TEST(FloatFieldsTest, SignBitFloat) {
	EXPECT_EQ(SignBit(1.0f),  0u);
	EXPECT_NE(SignBit(-1.0f), 0u);
	EXPECT_EQ(SignBit(0.0f),  0u);
}
TEST(FloatFieldsTest, SignBitDouble) {
	EXPECT_EQ(SignBit(1.0),  0ull);
	EXPECT_NE(SignBit(-1.0), 0ull);
}

// ---------------------------------------------------------------------------
// NextFloatUp / NextFloatDown -- float
// ---------------------------------------------------------------------------
TEST(NextFloatTest, UpFromZero) {
	float v = NextFloatUp(0.f);
	EXPECT_GT(v, 0.f);
	EXPECT_LT(v, 1e-44f);  // smallest positive denormal
}
TEST(NextFloatTest, DownFromZero) {
	float v = NextFloatDown(0.f);
	EXPECT_LT(v, 0.f);
	EXPECT_GT(v, -1e-44f);
}
TEST(NextFloatTest, UpFromOne) {
	float v = NextFloatUp(1.0f);
	EXPECT_GT(v, 1.0f);
	EXPECT_FLOAT_EQ(v - 1.0f, std::numeric_limits<float>::epsilon());
}
TEST(NextFloatTest, DownFromOne) {
	float v = NextFloatDown(1.0f);
	EXPECT_LT(v, 1.0f);
}
TEST(NextFloatTest, UpDownInverse) {
	// NextFloatDown(NextFloatUp(x)) == x for normal values
	float x = 3.14159f;
	EXPECT_FLOAT_EQ(NextFloatDown(NextFloatUp(x)), x);
	EXPECT_FLOAT_EQ(NextFloatUp(NextFloatDown(x)), x);
}
TEST(NextFloatTest, PositiveInfStaysInf) {
	float inf = std::numeric_limits<float>::infinity();
	EXPECT_EQ(NextFloatUp(inf), inf);
}
TEST(NextFloatTest, NegativeInfStaysInf) {
	float ninf = -std::numeric_limits<float>::infinity();
	EXPECT_EQ(NextFloatDown(ninf), ninf);
}
TEST(NextFloatTest, NegativeZeroTreatedAsPositive) {
	// NextFloatUp(-0) = NextFloatUp(+0) = smallest positive denormal
	EXPECT_GT(NextFloatUp(-0.f), 0.f);
}

// ---------------------------------------------------------------------------
// NextFloatUp / NextFloatDown -- double
// ---------------------------------------------------------------------------
TEST(NextDoubleTest, UpFromOne) {
	double v = NextFloatUp(1.0);
	EXPECT_GT(v, 1.0);
	EXPECT_DOUBLE_EQ(v - 1.0, std::numeric_limits<double>::epsilon());
}
TEST(NextDoubleTest, UpDownInverse) {
	double x = 2.718281828;
	EXPECT_DOUBLE_EQ(NextFloatDown(NextFloatUp(x)), x);
}
TEST(NextDoubleTest, PositiveInfStaysInf) {
	double inf = std::numeric_limits<double>::infinity();
	EXPECT_EQ(NextFloatUp(inf), inf);
}

// FlipSign(a,b) XORs a's sign bit with b's sign bit.
// If b is positive (sign=0): XOR leaves a's sign unchanged.
// If b is negative (sign=1): XOR flips a's sign.
TEST(FlipSignTest, FloatPositiveFromNegative) {
	// FlipSign(2.0f, -1.0f): b negative -> flip sign of a -> -2.0f
	EXPECT_FLOAT_EQ(FlipSign(2.0f, -1.0f), -2.0f);
}
TEST(FlipSignTest, FloatNegativeFromPositive) {
	// FlipSign(-3.0f, 1.0f): b positive -> sign unchanged -> -3.0f
	EXPECT_FLOAT_EQ(FlipSign(-3.0f, 1.0f), -3.0f);
}
TEST(FlipSignTest, DoubleNoChange) {
	// b positive -> a's sign unchanged
	EXPECT_DOUBLE_EQ(FlipSign(5.0, 2.0), 5.0);
}
TEST(FlipSignTest, DoubleFlip) {
	// b negative -> flip sign of a (positive -> negative)
	EXPECT_DOUBLE_EQ(FlipSign(5.0, -2.0), -5.0);
}

// ---------------------------------------------------------------------------
// gamma(n)
// ---------------------------------------------------------------------------
TEST(GammaTest, GammaOne) {
	float g = gamma(1);
	EXPECT_NEAR(g, FloatMachineEpsilon / (1.f - FloatMachineEpsilon), 1e-10f);
}
TEST(GammaTest, GammaNPositive) {
	for (int n = 1; n <= 8; ++n) {
		float g = gamma(n);
		EXPECT_GT(g, 0.f);
		EXPECT_LT(g, 1.f);
	}
}
TEST(GammaTest, GammaMonotone) {
	for (int n = 1; n < 8; ++n)
		EXPECT_LT(gamma(n), gamma(n + 1));
}

// ---------------------------------------------------------------------------
// Rounding-mode arithmetic -- conservative bound: result strictly contains true value
// ---------------------------------------------------------------------------
TEST(RoundingOpsTest, AddRoundUpGreaterThanExact) {
	float a = 1.0f, b = 1e-7f;
	float exact = a + b;
	EXPECT_GE(AddRoundUp(a, b), exact);
}
TEST(RoundingOpsTest, AddRoundDownLessOrEqualExact) {
	float a = 1.0f, b = 1e-7f;
	float exact = a + b;
	EXPECT_LE(AddRoundDown(a, b), exact);
}
TEST(RoundingOpsTest, MulRoundBoundsValue) {
	double a = 1.0 / 3.0, b = 3.0;
	double lo = MulRoundDown(a, b);
	double hi = MulRoundUp(a, b);
	// The true value (1.0) must be in [lo, hi]
	EXPECT_LE(lo, 1.0);
	EXPECT_GE(hi, 1.0);
}
TEST(RoundingOpsTest, DivRoundBoundsValue) {
	double a = 1.0, b = 3.0;
	double lo = DivRoundDown(a, b);
	double hi = DivRoundUp(a, b);
	double exact = a / b;
	EXPECT_LE(lo, exact);
	EXPECT_GE(hi, exact);
}
TEST(RoundingOpsTest, SqrtRoundBoundsValue) {
	double a = 2.0;
	double lo = SqrtRoundDown(a);
	double hi = SqrtRoundUp(a);
	double exact = std::sqrt(a);
	EXPECT_LE(lo, exact);
	EXPECT_GE(hi, exact);
}
TEST(RoundingOpsTest, SqrtRoundDownNonNegative) {
	// SqrtRoundDown must not return negative even for tiny inputs
	EXPECT_GE(SqrtRoundDown(0.0), 0.0);
}
TEST(RoundingOpsTest, SubRoundBoundsValue) {
	double a = 1.0, b = 1.0 / 3.0;
	double lo = SubRoundDown(a, b);
	double hi = SubRoundUp(a, b);
	double exact = a - b;
	EXPECT_LE(lo, exact);
	EXPECT_GE(hi, exact);
}
TEST(RoundingOpsTest, FMARoundBoundsValue) {
	float a = 1.0f / 3.0f, b = 3.0f, c = 1e-6f;
	float lo = FMARoundDown(a, b, c);
	float hi = FMARoundUp(a, b, c);
	float exact = std::fma(a, b, c);
	EXPECT_LE(lo, exact);
	EXPECT_GE(hi, exact);
}
