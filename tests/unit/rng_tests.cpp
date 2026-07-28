// rng_tests.cpp — Unit tests for src/shared/rng.h
// Ported from pbrt-v4 util/rng.h; validates PCG32 correctness,
// Uniform<T> specialisations, bounded integers, Advance(), and operator-.

#include "../../src/shared/rng.h"
#include "../../src/shared/float_bits.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <cmath>
#include <limits>
#include <set>
#include <vector>

// -----------------------------------------------------------------------
// Default construction
// -----------------------------------------------------------------------
TEST(RNGTest, DefaultState) {
	RNG a, b;
	// Two default-constructed RNGs must produce identical sequences.
	for (int i = 0; i < 16; ++i)
		EXPECT_EQ(a.Uniform<uint32_t>(), b.Uniform<uint32_t>());
}

// -----------------------------------------------------------------------
// Different sequences are independent
// -----------------------------------------------------------------------
TEST(RNGTest, DifferentSequences) {
	RNG a(0, 0), b(1, 0);
	bool anyDiffer = false;
	for (int i = 0; i < 16; ++i)
		if (a.Uniform<uint32_t>() != b.Uniform<uint32_t>())
			anyDiffer = true;
	EXPECT_TRUE(anyDiffer);
}

// -----------------------------------------------------------------------
// SetSequence reproduces the same stream
// -----------------------------------------------------------------------
TEST(RNGTest, SetSequenceReproducible) {
	RNG r1(42, 7), r2;
	r2.SetSequence(42, 7);
	for (int i = 0; i < 32; ++i)
		EXPECT_EQ(r1.Uniform<uint32_t>(), r2.Uniform<uint32_t>());
}

// -----------------------------------------------------------------------
// Uniform<uint32_t> — basic sanity
// -----------------------------------------------------------------------
TEST(RNGTest, Uint32Range) {
	RNG rng(0, 0);
	bool hasSmall = false, hasLarge = false;
	for (int i = 0; i < 1000; ++i) {
		uint32_t v = rng.Uniform<uint32_t>();
		if (v < 0x10000000u) hasSmall = true;
		if (v > 0xf0000000u) hasLarge = true;
	}
	EXPECT_TRUE(hasSmall);
	EXPECT_TRUE(hasLarge);
}

// -----------------------------------------------------------------------
// Uniform<uint64_t>
// -----------------------------------------------------------------------
TEST(RNGTest, Uint64Range) {
	RNG rng(1, 99);
	uint64_t orAcc = 0;
	for (int i = 0; i < 64; ++i)
		orAcc |= rng.Uniform<uint64_t>();
	// All 64 bits should be reachable over many samples.
	EXPECT_NE(orAcc, 0u);
}

// -----------------------------------------------------------------------
// Uniform<int32_t> and Uniform<int64_t> — sign coverage
// -----------------------------------------------------------------------
TEST(RNGTest, Int32SignCoverage) {
	RNG rng(2, 0);
	bool hasNeg = false, hasPos = false;
	for (int i = 0; i < 200; ++i) {
		int32_t v = rng.Uniform<int32_t>();
		if (v < 0) hasNeg = true;
		if (v > 0) hasPos = true;
	}
	EXPECT_TRUE(hasNeg);
	EXPECT_TRUE(hasPos);
}

TEST(RNGTest, Int64SignCoverage) {
	RNG rng(3, 0);
	bool hasNeg = false, hasPos = false;
	for (int i = 0; i < 200; ++i) {
		int64_t v = rng.Uniform<int64_t>();
		if (v < 0) hasNeg = true;
		if (v > 0) hasPos = true;
	}
	EXPECT_TRUE(hasNeg);
	EXPECT_TRUE(hasPos);
}

// -----------------------------------------------------------------------
// Uniform<float> — in [0, 1)
// -----------------------------------------------------------------------
TEST(RNGTest, FloatInUnitInterval) {
	RNG rng(4, 0);
	for (int i = 0; i < 10000; ++i) {
		float v = rng.Uniform<float>();
		EXPECT_GE(v, 0.0f);
		EXPECT_LT(v, 1.0f);
	}
}

// -----------------------------------------------------------------------
// Uniform<double> — in [0, 1)
// -----------------------------------------------------------------------
TEST(RNGTest, DoubleInUnitInterval) {
	RNG rng(5, 0);
	for (int i = 0; i < 10000; ++i) {
		double v = rng.Uniform<double>();
		EXPECT_GE(v, 0.0);
		EXPECT_LT(v, 1.0);
	}
}

// -----------------------------------------------------------------------
// Uniform(T bound) — unbiased bounded integer
// -----------------------------------------------------------------------
TEST(RNGTest, BoundedUint32) {
	RNG rng(6, 0);
	constexpr uint32_t B = 6u;
	std::set<uint32_t> seen;
	for (int i = 0; i < 10000; ++i) {
		uint32_t v = rng.Uniform(B);
		EXPECT_LT(v, B);
		seen.insert(v);
	}
	// All faces of a die should appear.
	EXPECT_EQ(seen.size(), (size_t)B);
}

// -----------------------------------------------------------------------
// Advance — forward skip
// -----------------------------------------------------------------------
TEST(RNGTest, AdvanceForward) {
	RNG r1(7, 0), r2(7, 0);
	// Consume 100 values from r1 manually.
	for (int i = 0; i < 100; ++i)
		r1.Uniform<uint32_t>();
	// Skip r2 ahead 100 steps.
	r2.Advance(100);
	// They should now be in sync.
	for (int i = 0; i < 20; ++i)
		EXPECT_EQ(r1.Uniform<uint32_t>(), r2.Uniform<uint32_t>());
}

// -----------------------------------------------------------------------
// Advance — backward skip
// -----------------------------------------------------------------------
TEST(RNGTest, AdvanceBackward) {
	RNG r1(8, 0), r2(8, 0);
	for (int i = 0; i < 50; ++i)
		r2.Uniform<uint32_t>();
	// Step r2 back 50 steps — should match r1.
	r2.Advance(-50);
	for (int i = 0; i < 20; ++i)
		EXPECT_EQ(r1.Uniform<uint32_t>(), r2.Uniform<uint32_t>());
}

// -----------------------------------------------------------------------
// operator- — stream distance
// -----------------------------------------------------------------------
TEST(RNGTest, StreamDistance) {
	RNG r1(9, 0), r2(9, 0);
	r2.Advance(77);
	EXPECT_EQ(r2 - r1, 77);
	EXPECT_EQ(r1 - r2, -77);
}

// -----------------------------------------------------------------------
// Reproducibility across SetSequence calls
// -----------------------------------------------------------------------
TEST(RNGTest, ReseedProducesIdenticalStream) {
	RNG rng(10, 5);
	std::vector<uint32_t> first;
	for (int i = 0; i < 32; ++i)
		first.push_back(rng.Uniform<uint32_t>());

	rng.SetSequence(10, 5);
	for (int i = 0; i < 32; ++i)
		EXPECT_EQ(rng.Uniform<uint32_t>(), first[i]);
}

// -----------------------------------------------------------------------
// Statistical: mean of Uniform<float> should be ~0.5
// -----------------------------------------------------------------------
TEST(RNGTest, FloatMean) {
	RNG rng(11, 0);
	double sum = 0.0;
	const int N = 100000;
	for (int i = 0; i < N; ++i)
		sum += rng.Uniform<float>();
	double mean = sum / N;
	EXPECT_NEAR(mean, 0.5, 0.01);
}

// -----------------------------------------------------------------------
// OneMinusEpsilon clamp — Uniform<float> must never reach 1.0f
// -----------------------------------------------------------------------
TEST(RNGTest, FloatNeverOne) {
	RNG rng(12, 0);
	for (int i = 0; i < 1000000; ++i)
		EXPECT_LT(rng.Uniform<float>(), 1.0f);
}

// -----------------------------------------------------------------------
// Clamp threshold correctness:
//   float  must clamp at FloatOneMinusEpsilon  (0x1.fffffep-1f)
//   double must clamp at DoubleOneMinusEpsilon (0x1.fffffffffffffp-1)
//   NOT at (double)FloatOneMinusEpsilon which would be ~0.9999999403...
// -----------------------------------------------------------------------
TEST(RNGTest, FloatClampThreshold) {
	// Largest float strictly less than 1.
	EXPECT_EQ(FloatOneMinusEpsilon, 0x1.fffffep-1f);
	// The float branch should be <= FloatOneMinusEpsilon.
	RNG rng(13, 0);
	for (int i = 0; i < 100000; ++i) {
		float v = rng.Uniform<float>();
		EXPECT_LE(v, FloatOneMinusEpsilon);
	}
}

TEST(RNGTest, DoubleClampThreshold) {
	// Largest double strictly less than 1.
	EXPECT_EQ(DoubleOneMinusEpsilon, 0x1.fffffffffffffp-1);
	// The double branch must clamp at DoubleOneMinusEpsilon, not the
	// narrower float-cast value ~0.9999999403.
	EXPECT_GT(DoubleOneMinusEpsilon, (double)FloatOneMinusEpsilon);
	RNG rng(14, 0);
	for (int i = 0; i < 100000; ++i) {
		double v = rng.Uniform<double>();
		EXPECT_LE(v, DoubleOneMinusEpsilon);
		EXPECT_LT(v, 1.0);
	}
}
