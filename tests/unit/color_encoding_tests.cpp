// color_encoding_tests.cpp
// Unit tests for src/shared/color_encoding.h
// pbrt-v4 reference: src/pbrt/util/color.h + color.cpp
//
// Tests cover:
//   LinearToSRGB / SRGBToLinear round-trips
//   LinearToSRGB8 / SRGB8ToLinear LUT accuracy
//   LinearColorEncoding batch encode/decode
//   sRGBColorEncoding batch encode/decode + scalar
//   GammaColorEncoding LUT construction + round-trip
//   ColorEncoding variant dispatch (all three)

#include <gtest/gtest.h>
#include "../../src/shared/color_encoding.h"
#include <cmath>
#include <vector>

static constexpr float kEps = 1e-5f;

static bool near(float a, float b, float eps = kEps) {
	return std::fabs(a - b) <= eps;
}

// Helper: build ce_detail::Span from std::vector
template<typename T>
ce_detail::Span<T> MakeSpan(std::vector<T>& v) {
	return ce_detail::Span<T>(v.data(), v.size());
}
template<typename T>
ce_detail::Span<const T> MakeSpanConst(const std::vector<T>& v) {
	return ce_detail::Span<const T>(v.data(), v.size());
}

// ---------------------------------------------------------------------------
// Free function tests: LinearToSRGB / SRGBToLinear
// ---------------------------------------------------------------------------

TEST(LinearToSRGBTest, ZeroMapsToZero) {
	EXPECT_TRUE(near(LinearToSRGB(0.f), 0.f));
}

TEST(LinearToSRGBTest, OneMapsToOne) {
	EXPECT_TRUE(near(LinearToSRGB(1.f), 1.f));
}

TEST(LinearToSRGBTest, LinearSegmentBelow0031308) {
	// x <= 0.0031308: result = 12.92 * x
	float x = 0.001f;
	EXPECT_TRUE(near(LinearToSRGB(x), 12.92f * x, 1e-6f));
}

TEST(LinearToSRGBTest, KnownMidGray) {
	// sRGB 0.5 linear ~= 0.7353 encoded (standard value)
	float enc = LinearToSRGB(0.5f);
	EXPECT_GT(enc, 0.72f);
	EXPECT_LT(enc, 0.75f);
}

TEST(LinearToSRGBTest, NegativePassthrough) {
	// pbrt-v4 LinearToSRGB does NOT clamp negatives in the scalar function.
	// Clamping is done only by LinearToSRGB8 via uint8 output range.
	EXPECT_TRUE(LinearToSRGB(-1.f) < 0.f);
}

TEST(SRGBToLinearTest, ZeroMapsToZero) {
	EXPECT_TRUE(near(SRGBToLinear(0.f), 0.f));
}

TEST(SRGBToLinearTest, OneMapsToOne) {
	EXPECT_TRUE(near(SRGBToLinear(1.f), 1.f, 1e-4f));
}

TEST(SRGBToLinearTest, LinearSegmentBelow004045) {
	float x = 0.02f;
	EXPECT_TRUE(near(SRGBToLinear(x), x / 12.92f, 1e-5f));
}

TEST(SRGBToLinearTest, RoundTripMidGray) {
	float linear = 0.5f;
	float encoded = LinearToSRGB(linear);
	float decoded = SRGBToLinear(encoded);
	EXPECT_TRUE(near(decoded, linear, 1e-4f));
}

TEST(SRGBToLinearTest, RoundTripMultipleValues) {
	float vals[] = {0.01f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 1.0f};
	for (float v : vals) {
		float rt = SRGBToLinear(LinearToSRGB(v));
		EXPECT_TRUE(near(rt, v, 1e-4f)) << "round-trip failed for v=" << v;
	}
}

// ---------------------------------------------------------------------------
// LinearToSRGB8 / SRGB8ToLinear
// ---------------------------------------------------------------------------

TEST(LinearToSRGB8Test, ZeroGivesZero) {
	EXPECT_EQ(LinearToSRGB8(0.f), 0);
}

TEST(LinearToSRGB8Test, OneGives255) {
	EXPECT_EQ(LinearToSRGB8(1.f), 255);
}

TEST(LinearToSRGB8Test, MidGrayInRange) {
	uint8_t v = LinearToSRGB8(0.5f);
	EXPECT_GE(v, 185);
	EXPECT_LE(v, 190);
}

TEST(SRGB8ToLinearTest, ZeroGivesZero) {
	EXPECT_FLOAT_EQ(SRGB8ToLinear(0), 0.f);
}

TEST(SRGB8ToLinearTest, MaxGivesOne) {
	EXPECT_FLOAT_EQ(SRGB8ToLinear(255), 1.f);
}

TEST(SRGB8ToLinearTest, LUTMatchesFormula) {
	// LUT values should match SRGBToLinear applied to i/255 (approximately)
	for (int i = 1; i < 255; ++i) {
		float lut  = SRGB8ToLinear((uint8_t)i);
		float calc = SRGBToLinear(float(i) / 255.f);
		EXPECT_TRUE(near(lut, calc, 5e-4f)) << "LUT mismatch at i=" << i;
	}
}

TEST(SRGB8ToLinearTest, Round8bitRoundTripWithinOneStep) {
	// Encoding a LUT value and decoding should yield the original index
	for (int i = 0; i < 256; ++i) {
		float   linear  = SRGB8ToLinear((uint8_t)i);
		uint8_t encoded = LinearToSRGB8(linear);
		// Expect within 1 LSB
		EXPECT_LE(std::abs(int(encoded) - i), 1) << "round-trip LSB at i=" << i;
	}
}

// ---------------------------------------------------------------------------
// LinearColorEncoding
// ---------------------------------------------------------------------------

TEST(LinearColorEncodingTest, ToLinearDividesBy255) {
	LinearColorEncoding enc;
	std::vector<uint8_t> in  = {0, 128, 255};
	std::vector<float>   out(3);
	enc.ToLinear(MakeSpanConst(in), MakeSpan(out));
	EXPECT_TRUE(near(out[0], 0.f));
	EXPECT_TRUE(near(out[1], 128.f / 255.f, 1e-5f));
	EXPECT_TRUE(near(out[2], 1.f));
}

TEST(LinearColorEncodingTest, FromLinearMultipliesBy255) {
	LinearColorEncoding enc;
	std::vector<float>   in  = {0.f, 0.5f, 1.f};
	std::vector<uint8_t> out(3);
	enc.FromLinear(MakeSpanConst(in), MakeSpan(out));
	EXPECT_EQ(out[0], 0);
	EXPECT_EQ(out[1], 128);  // round(0.5*255) = 128
	EXPECT_EQ(out[2], 255);
}

TEST(LinearColorEncodingTest, ToFloatLinearIsIdentity) {
	LinearColorEncoding enc;
	EXPECT_FLOAT_EQ(enc.ToFloatLinear(0.7f), 0.7f);
}

TEST(LinearColorEncodingTest, RoundTripAllBytes) {
	LinearColorEncoding enc;
	for (int i = 0; i < 256; ++i) {
		std::vector<uint8_t> in  = {(uint8_t)i};
		std::vector<float>   mid(1);
		std::vector<uint8_t> out(1);
		enc.ToLinear(MakeSpanConst(in), MakeSpan(mid));
		enc.FromLinear(MakeSpanConst(mid), MakeSpan(out));
		EXPECT_EQ(out[0], (uint8_t)i) << "round-trip failed at i=" << i;
	}
}

// ---------------------------------------------------------------------------
// sRGBColorEncoding
// ---------------------------------------------------------------------------

TEST(sRGBColorEncodingTest, ToLinearUsesLUT) {
	sRGBColorEncoding enc;
	std::vector<uint8_t> in  = {0, 128, 255};
	std::vector<float>   out(3);
	enc.ToLinear(MakeSpanConst(in), MakeSpan(out));
	EXPECT_FLOAT_EQ(out[0], SRGB8ToLinear(0));
	EXPECT_FLOAT_EQ(out[1], SRGB8ToLinear(128));
	EXPECT_FLOAT_EQ(out[2], SRGB8ToLinear(255));
}

TEST(sRGBColorEncodingTest, FromLinearUsesLinearToSRGB8) {
	sRGBColorEncoding enc;
	std::vector<float>   in  = {0.f, 0.5f, 1.f};
	std::vector<uint8_t> out(3);
	enc.FromLinear(MakeSpanConst(in), MakeSpan(out));
	EXPECT_EQ(out[0], LinearToSRGB8(0.f));
	EXPECT_EQ(out[2], LinearToSRGB8(1.f));
}

TEST(sRGBColorEncodingTest, ToFloatLinearCallsSRGBToLinear) {
	sRGBColorEncoding enc;
	float v = 0.5f;
	EXPECT_TRUE(near(enc.ToFloatLinear(v), SRGBToLinear(v), 1e-6f));
}

TEST(sRGBColorEncodingTest, RoundTripAllBytesWithinOneLSB) {
	sRGBColorEncoding enc;
	for (int i = 0; i < 256; ++i) {
		std::vector<uint8_t> in  = {(uint8_t)i};
		std::vector<float>   mid(1);
		std::vector<uint8_t> out(1);
		enc.ToLinear(MakeSpanConst(in), MakeSpan(mid));
		enc.FromLinear(MakeSpanConst(mid), MakeSpan(out));
		EXPECT_LE(std::abs(int(out[0]) - i), 1)
			<< "sRGB round-trip out of 1 LSB at i=" << i;
	}
}

// ---------------------------------------------------------------------------
// GammaColorEncoding
// ---------------------------------------------------------------------------

TEST(GammaColorEncodingTest, Gamma1IsLinear) {
	GammaColorEncoding enc(1.f);
	// gamma=1: ToFloatLinear(v) = v^1 = v
	EXPECT_TRUE(near(enc.ToFloatLinear(0.5f), 0.5f, 1e-4f));
}

TEST(GammaColorEncodingTest, Gamma2DecodeKnownValue) {
	GammaColorEncoding enc(2.f);
	// (128/255)^2 ≈ 0.2520
	float decoded = enc.applyLUT()[128];
	float expected = std::pow(128.f / 255.f, 2.f);
	EXPECT_TRUE(near(decoded, expected, 1e-4f));
}

TEST(GammaColorEncodingTest, ToFloatLinearMatchesLUT) {
	GammaColorEncoding enc(2.2f);
	// Scalar ToFloatLinear should match pow(v, 2.2)
	float v = 0.6f;
	EXPECT_TRUE(near(enc.ToFloatLinear(v), std::pow(v, 2.2f), 1e-4f));
}

TEST(GammaColorEncodingTest, BatchRoundTripWithinOneLSB) {
	GammaColorEncoding enc(2.2f);
	for (int i = 0; i < 256; ++i) {
		std::vector<uint8_t> in  = {(uint8_t)i};
		std::vector<float>   mid(1);
		std::vector<uint8_t> out(1);
		enc.ToLinear(MakeSpanConst(in), MakeSpan(mid));
		enc.FromLinear(MakeSpanConst(mid), MakeSpan(out));
		EXPECT_LE(std::abs(int(out[0]) - i), 1)
			<< "gamma=2.2 round-trip out of 1 LSB at i=" << i;
	}
}

// ---------------------------------------------------------------------------
// ColorEncoding variant dispatch
// ---------------------------------------------------------------------------

TEST(ColorEncodingTest, LinearVariantDispatch) {
	ColorEncoding enc = ColorEncoding::Linear();
	std::vector<uint8_t> in  = {255};
	std::vector<float>   out(1);
	enc.ToLinear(MakeSpanConst(in), MakeSpan(out));
	EXPECT_TRUE(near(out[0], 1.f));
	EXPECT_TRUE(near(enc.ToFloatLinear(0.3f), 0.3f));
}

TEST(ColorEncodingTest, SRGBVariantDispatch) {
	ColorEncoding enc = ColorEncoding::sRGB();
	EXPECT_FLOAT_EQ(enc.ToFloatLinear(0.f), 0.f);
	EXPECT_TRUE(near(enc.ToFloatLinear(1.f), SRGBToLinear(1.f), 1e-5f));
}

TEST(ColorEncodingTest, GammaVariantDispatch) {
	ColorEncoding enc = ColorEncoding::Gamma(2.f);
	// ToFloatLinear(0.5) = 0.5^2 = 0.25
	EXPECT_TRUE(near(enc.ToFloatLinear(0.5f), 0.25f, 1e-4f));
}

TEST(ColorEncodingTest, VariantFromLinearDispatch) {
	ColorEncoding enc = ColorEncoding::sRGB();
	std::vector<float>   in  = {1.f};
	std::vector<uint8_t> out(1);
	enc.FromLinear(MakeSpanConst(in), MakeSpan(out));
	EXPECT_EQ(out[0], 255);
}
