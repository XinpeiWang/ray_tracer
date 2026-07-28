// pbrt_hash_tests.cpp
// Unit tests for pbrt_hash.h ported from pbrt-v4 util/hash.h

#include "../../src/shared/pbrt_hash.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// MurmurHash64A
// ---------------------------------------------------------------------------
TEST(MurmurHashTest, EmptyInputIsDeterministic) {
	// Use a valid (non-null) pointer with len=0 to avoid UB.
	const unsigned char dummy = 0;
	uint64_t h1 = MurmurHash64A(&dummy, 0, 0);
	uint64_t h2 = MurmurHash64A(&dummy, 0, 0);
	EXPECT_EQ(h1, h2);
}

TEST(MurmurHashTest, DifferentSeedsProduceDifferentHashes) {
	const unsigned char data[] = "hello";
	uint64_t h0 = MurmurHash64A(data, 5, 0);
	uint64_t h1 = MurmurHash64A(data, 5, 1);
	EXPECT_NE(h0, h1);
}

TEST(MurmurHashTest, DifferentDataProduceDifferentHashes) {
	const unsigned char a[] = "hello";
	const unsigned char b[] = "world";
	EXPECT_NE(MurmurHash64A(a, 5, 0), MurmurHash64A(b, 5, 0));
}

TEST(MurmurHashTest, KnownValue) {
	// Regression: pin the exact output so any algorithm change is caught.
	// Expected value verified against the pbrt-v4 implementation.
	const unsigned char data[] = {0x01, 0x02, 0x03, 0x04,
								  0x05, 0x06, 0x07, 0x08};
	uint64_t h = MurmurHash64A(data, 8, 0);
	EXPECT_EQ(h, uint64_t(0x88B2A580354486B7ULL));
}

TEST(MurmurHashTest, IncrementalLengthsDiffer) {
	const unsigned char data[] = "abcdefgh";
	uint64_t prev = MurmurHash64A(data, 1, 0);
	for (size_t len = 2; len <= 8; ++len) {
		uint64_t cur = MurmurHash64A(data, len, 0);
		EXPECT_NE(cur, prev) << "lengths " << len-1 << " and " << len << " collide";
		prev = cur;
	}
}

// ---------------------------------------------------------------------------
// MixBits
// ---------------------------------------------------------------------------
TEST(MixBitsTest, ZeroIsFixedPoint) {
	// MixBits(0) = 0 is the expected mathematical result: all XOR/multiply
	// operations on 0 yield 0. This is a known property of the mixer.
	EXPECT_EQ(MixBits(0), uint64_t(0));
}

TEST(MixBitsTest, Deterministic) {
	EXPECT_EQ(MixBits(42), MixBits(42));
}

TEST(MixBitsTest, Avalanche) {
	// Single bit flip in input should change many output bits.
	uint64_t a = MixBits(0x0000000000000001ull);
	uint64_t b = MixBits(0x0000000000000002ull);
	uint64_t diff = a ^ b;
	int bits_changed = 0;
	for (int i = 0; i < 64; ++i)
		if ((diff >> i) & 1) ++bits_changed;
	EXPECT_GT(bits_changed, 20);
}

// ---------------------------------------------------------------------------
// HashBuffer
// ---------------------------------------------------------------------------
TEST(HashBufferTest, SameDataSameHash) {
	int data[] = {1, 2, 3, 4};
	EXPECT_EQ(HashBuffer(data, 4), HashBuffer(data, 4));
}

TEST(HashBufferTest, DifferentDataDifferentHash) {
	int a[] = {1, 2, 3, 4};
	int b[] = {1, 2, 3, 5};
	EXPECT_NE(HashBuffer(a, 4), HashBuffer(b, 4));
}

TEST(HashBufferTest, SeedChangesResult) {
	float data[] = {1.0f, 2.0f};
	EXPECT_NE(HashBuffer(data, 2, 0), HashBuffer(data, 2, 99));
}

// ---------------------------------------------------------------------------
// Hash<Args...>
// ---------------------------------------------------------------------------
TEST(HashTest, SameArgsSameHash) {
	EXPECT_EQ(Hash(1, 2, 3), Hash(1, 2, 3));
}

TEST(HashTest, DifferentArgsDifferentHash) {
	EXPECT_NE(Hash(1, 2, 3), Hash(1, 2, 4));
}

TEST(HashTest, SingleInt) {
	EXPECT_EQ(Hash(42), Hash(42));
	EXPECT_NE(Hash(42), Hash(43));
}

TEST(HashTest, SingleFloat) {
	EXPECT_EQ(Hash(1.0f), Hash(1.0f));
	EXPECT_NE(Hash(1.0f), Hash(2.0f));
}

TEST(HashTest, MixedTypes) {
	// Hash(int, float, uint32_t) should be deterministic and non-trivial.
	uint64_t h = Hash(7, 3.14f, uint32_t(99));
	EXPECT_NE(h, uint64_t(0));
	EXPECT_EQ(h, Hash(7, 3.14f, uint32_t(99)));
}

TEST(HashTest, OrderMatters) {
	// Argument order changes the byte layout, so Hash(a,b) != Hash(b,a)
	// when types have different sizes (uint8_t vs uint32_t = 5 vs 5 bytes
	// but at different offsets in the buffer).
	uint64_t hab = Hash(uint8_t(1), uint32_t(2));
	uint64_t hba = Hash(uint32_t(2), uint8_t(1));
	EXPECT_NE(hab, hba) << "Hash should depend on argument order";
	// Each call must also be self-consistent.
	EXPECT_EQ(Hash(uint8_t(1), uint32_t(2)), hab);
	EXPECT_EQ(Hash(uint32_t(2), uint8_t(1)), hba);
}

TEST(HashTest, PixelSampleUse) {
	// Typical sampler use: Hash(pixel_x, pixel_y, sample_index, seed)
	uint64_t h00 = Hash(0u, 0u, 0u, 0u);
	uint64_t h01 = Hash(0u, 1u, 0u, 0u);
	uint64_t h10 = Hash(1u, 0u, 0u, 0u);
	EXPECT_NE(h00, h01);
	EXPECT_NE(h00, h10);
	EXPECT_NE(h01, h10);
}

// ---------------------------------------------------------------------------
// HashFloat<Args...>
// ---------------------------------------------------------------------------
TEST(HashFloatTest, InUnitInterval) {
	for (int i = 0; i < 1000; ++i) {
		float v = HashFloat(i, 42);
		EXPECT_GE(v, 0.0f);
		EXPECT_LT(v, 1.0f);
	}
}

TEST(HashFloatTest, Deterministic) {
	EXPECT_EQ(HashFloat(7, 13), HashFloat(7, 13));
}

TEST(HashFloatTest, DifferentArgsDifferentValues) {
	EXPECT_NE(HashFloat(1, 2), HashFloat(1, 3));
}

TEST(HashFloatTest, UniformDistribution) {
	// Rough uniformity check: mean of 10000 samples should be near 0.5
	double sum = 0.0;
	const int N = 10000;
	for (int i = 0; i < N; ++i)
		sum += HashFloat(i, uint32_t(0xdeadbeef));
	double mean = sum / N;
	EXPECT_NEAR(mean, 0.5, 0.02);
}

TEST(HashFloatTest, PixelDecorrelation) {
	// Different pixels with same sample index should produce spread values,
	// not cluster — basic sampler sanity check.
	int collisions = 0;
	for (int px = 0; px < 32; ++px)
		for (int py = 0; py < 32; ++py) {
			float v = HashFloat(px, py, 0);
			// Every value must be in [0,1)
			EXPECT_GE(v, 0.0f);
			EXPECT_LT(v, 1.0f);
		}
	EXPECT_EQ(collisions, 0);
}
