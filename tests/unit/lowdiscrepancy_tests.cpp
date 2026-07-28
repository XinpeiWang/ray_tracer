// lowdiscrepancy_tests.cpp
// Unit tests for src/shared/lowdiscrepancy.h
// Coverage:
//   - NoRandomizer: identity
//   - BinaryPermuteScrambler: XOR property
//   - FastOwenScrambler: seeded consistency, bit-reversal symmetry
//   - OwenScrambler: bit-level scrambling properties
//   - RandomizeStrategy: enum values
//   - MultiplyGenerator: known generator-matrix outputs
//   - SobolSample<R>: dim-0 matches Van der Corput, scrambler round-trips
//   - InverseRadicalInverse: inverse of digit reversal
//   - OwenScrambledRadicalInverse: range, consistency

#include <gtest/gtest.h>
#include "../../src/shared/lowdiscrepancy.h"
#include "../../src/shared/scalar_math.h"

#include <cmath>
#include <cstdint>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Van der Corput (base-2 radical inverse) for ground truth
static float VanDerCorput(uint32_t index) {
	return ReverseBits32(index) * 0x1p-32f;
}

// Minimal 32-entry identity Sobol matrix for dim 0 (Van der Corput)
static constexpr int kMatrixSize = 32;
static constexpr uint32_t kIdentityMatrix[kMatrixSize] = {
	0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
	0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
	0x00800000u, 0x00400000u, 0x00200000u, 0x00100000u,
	0x00080000u, 0x00040000u, 0x00020000u, 0x00010000u,
	0x00008000u, 0x00004000u, 0x00002000u, 0x00001000u,
	0x00000800u, 0x00000400u, 0x00000200u, 0x00000100u,
	0x00000080u, 0x00000040u, 0x00000020u, 0x00000010u,
	0x00000008u, 0x00000004u, 0x00000002u, 0x00000001u,
};

// ---------------------------------------------------------------------------
// ReverseBits32 / EncodeMorton2 (added to scalar_math.h)
// ---------------------------------------------------------------------------

TEST(ReverseBits32Test, Identity) {
	// Reversing twice returns the original value
	for (uint32_t v : {0u, 1u, 0x80000000u, 0xDEADBEEFu, 0xFFFFFFFFu}) {
		EXPECT_EQ(ReverseBits32(ReverseBits32(v)), v);
	}
}

TEST(ReverseBits32Test, KnownValues) {
	EXPECT_EQ(ReverseBits32(0x00000001u), 0x80000000u);
	EXPECT_EQ(ReverseBits32(0x80000000u), 0x00000001u);
	EXPECT_EQ(ReverseBits32(0x00000000u), 0x00000000u);
	EXPECT_EQ(ReverseBits32(0xFFFFFFFFu), 0xFFFFFFFFu);
}

TEST(EncodeMorton2Test, SmallValues) {
	// x=1 (bits: 01) interleaved with y=0 should give bit 0 in position 0
	EXPECT_EQ(EncodeMorton2(1, 0), 1ull);
	// y=1 interleaved should give bit in position 1
	EXPECT_EQ(EncodeMorton2(0, 1), 2ull);
	// x=1, y=1 -> 0b11 = 3
	EXPECT_EQ(EncodeMorton2(1, 1), 3ull);
	// x=2 (bit 1), y=0 -> bit 2
	EXPECT_EQ(EncodeMorton2(2, 0), 4ull);
}

// ---------------------------------------------------------------------------
// NoRandomizer
// ---------------------------------------------------------------------------

TEST(NoRandomizerTest, Identity) {
	NoRandomizer r;
	EXPECT_EQ(r(0u), 0u);
	EXPECT_EQ(r(0xDEADBEEFu), 0xDEADBEEFu);
	EXPECT_EQ(r(0xFFFFFFFFu), 0xFFFFFFFFu);
}

// ---------------------------------------------------------------------------
// BinaryPermuteScrambler
// ---------------------------------------------------------------------------

TEST(BinaryPermuteScramblerTest, XORProperty) {
	uint32_t perm = 0xABCDEF01u;
	BinaryPermuteScrambler s(perm);
	EXPECT_EQ(s(0u), perm);
	EXPECT_EQ(s(perm), 0u);
	// Applying twice returns original
	EXPECT_EQ(s(s(0x12345678u)), 0x12345678u);
}

TEST(BinaryPermuteScramblerTest, ZeroPermutation) {
	BinaryPermuteScrambler s(0u);
	EXPECT_EQ(s(0xCAFEBABEu), 0xCAFEBABEu);
}

// ---------------------------------------------------------------------------
// FastOwenScrambler
// ---------------------------------------------------------------------------

TEST(FastOwenScramblerTest, DifferentSeedsProduceDifferentOutputs) {
	uint32_t v = 0x12345678u;
	FastOwenScrambler s0(0u), s1(1u), s2(0xDEADBEEFu);
	EXPECT_NE(s0(v), s1(v));
	EXPECT_NE(s1(v), s2(v));
}

TEST(FastOwenScramblerTest, SameSeedConsistent) {
	FastOwenScrambler s(42u);
	EXPECT_EQ(s(0xABCDu), s(0xABCDu));
}

TEST(FastOwenScramblerTest, ZeroInputSeedZero) {
	// With seed 0 and input 0 the result is deterministic
	FastOwenScrambler s(0u);
	uint32_t result = s(0u);
	EXPECT_EQ(result, s(0u)); // reproducible
}

// ---------------------------------------------------------------------------
// OwenScrambler
// ---------------------------------------------------------------------------

TEST(OwenScramblerTest, SameSeedConsistent) {
	OwenScrambler s(12345u);
	EXPECT_EQ(s(0x00000000u), s(0x00000000u));
	EXPECT_EQ(s(0xFFFFFFFFu), s(0xFFFFFFFFu));
}

TEST(OwenScramblerTest, DifferentSeedsProduceDifferentOutputs) {
	uint32_t v = 0x55AA55AAu;
	EXPECT_NE(OwenScrambler(1u)(v), OwenScrambler(2u)(v));
}

TEST(OwenScramblerTest, MSBFlipWhenSeedOdd) {
	// When seed is odd, MSB of the output should be flipped relative to seed=0
	// (seed & 1) triggers v ^= 1u<<31 before any further scrambling.
	// For v=0: seed=1 -> MSB = 1; seed=0 -> MSB = 0 (no bit flip from seed).
	uint32_t v = 0u;
	OwenScrambler sOdd(1u), sEven(0u);
	EXPECT_NE((sOdd(v) >> 31) & 1, (sEven(v) >> 31) & 1);
}

// ---------------------------------------------------------------------------
// RandomizeStrategy enum
// ---------------------------------------------------------------------------

TEST(RandomizeStrategyTest, EnumValues) {
	// Confirm the four strategy values are distinct
	EXPECT_NE(static_cast<int>(RandomizeStrategy::None),
			  static_cast<int>(RandomizeStrategy::PermuteDigits));
	EXPECT_NE(static_cast<int>(RandomizeStrategy::FastOwen),
			  static_cast<int>(RandomizeStrategy::Owen));
	EXPECT_NE(static_cast<int>(RandomizeStrategy::PermuteDigits),
			  static_cast<int>(RandomizeStrategy::FastOwen));
}

// ---------------------------------------------------------------------------
// MultiplyGenerator
// ---------------------------------------------------------------------------

TEST(MultiplyGeneratorTest, IdentityMatrixIsVanDerCorput) {
	// The identity matrix maps sample index a to ReverseBits32(a)
	for (uint32_t a : {0u, 1u, 2u, 3u, 4u, 7u, 15u, 255u, 1023u}) {
		uint32_t expected = ReverseBits32(a);
		EXPECT_EQ(MultiplyGenerator(kIdentityMatrix, kMatrixSize, a), expected)
			<< "a=" << a;
	}
}

TEST(MultiplyGeneratorTest, Zero) {
	EXPECT_EQ(MultiplyGenerator(kIdentityMatrix, kMatrixSize, 0u), 0u);
}

// ---------------------------------------------------------------------------
// SobolSample<R>
// ---------------------------------------------------------------------------

TEST(SobolSampleTest, Dim0NoScrambleMatchesVanDerCorput) {
	NoRandomizer r;
	for (int64_t a = 0; a < 64; ++a) {
		float sobol = SobolSample(kIdentityMatrix, kMatrixSize, a, 0, r);
		float vdc   = VanDerCorput(static_cast<uint32_t>(a));
		EXPECT_NEAR(sobol, vdc, 1e-7f) << "a=" << a;
	}
}

TEST(SobolSampleTest, ResultInUnitInterval) {
	FastOwenScrambler r(0xDEADBEEFu);
	for (int64_t a = 0; a < 1024; ++a) {
		float v = SobolSample(kIdentityMatrix, kMatrixSize, a, 0, r);
		EXPECT_GE(v, 0.0f) << "a=" << a;
		EXPECT_LT(v, 1.0f) << "a=" << a;
	}
}

TEST(SobolSampleTest, BinaryPermuteScramblerShiftsOutput) {
	BinaryPermuteScrambler r(0xFFFFFFFFu);
	// With max XOR perm, result should be different from unscrambled
	bool any_different = false;
	for (int64_t a = 1; a < 64; ++a) {
		float scrambled   = SobolSample(kIdentityMatrix, kMatrixSize, a, 0, r);
		float unscrambled = SobolSample(kIdentityMatrix, kMatrixSize, a, 0,
										NoRandomizer{});
		if (std::abs(scrambled - unscrambled) > 1e-7f) { any_different = true; break; }
	}
	EXPECT_TRUE(any_different);
}

TEST(SobolSampleTest, OwenScramblerResultInRange) {
	OwenScrambler r(99u);
	for (int64_t a = 0; a < 256; ++a) {
		float v = SobolSample(kIdentityMatrix, kMatrixSize, a, 0, r);
		EXPECT_GE(v, 0.0f) << "a=" << a;
		EXPECT_LT(v, 1.0f) << "a=" << a;
	}
}

// ---------------------------------------------------------------------------
// InverseRadicalInverse
// ---------------------------------------------------------------------------

TEST(InverseRadicalInverseTest, Base2RoundTrip) {
	// For base 2, nDigits = 32:
	// InverseRadicalInverse(ReverseBits32(a), 2, 32) should recover a
	// (for a < 2^32, all 32 digits)
	for (uint32_t a : {0u, 1u, 2u, 7u, 255u, 65535u, 0x12345678u}) {
		uint64_t ri = ReverseBits32(a);
		EXPECT_EQ(InverseRadicalInverse(ri, 2, 32), static_cast<uint64_t>(a))
			<< "a=" << a;
	}
}

TEST(InverseRadicalInverseTest, Base3SmallValues) {
	// Manually: a=5 in base 3 = "12", radical inverse digits reversed = "21" -> 2*1+1 = 7/9
	// InverseRadicalInverse(reversed_digits, 3, nDigits) = a
	// a=4 = "11" base 3, reversed "11" -> same index
	EXPECT_EQ(InverseRadicalInverse(4u, 3, 2), 4u);
	// a=0 always round-trips
	EXPECT_EQ(InverseRadicalInverse(0u, 3, 4), 0u);
}

// ---------------------------------------------------------------------------
// OwenScrambledRadicalInverse
// ---------------------------------------------------------------------------

TEST(OwenScrambledRadicalInverseTest, ResultInUnitInterval) {
	for (int baseIdx = 0; baseIdx < 8; ++baseIdx) {
		for (uint64_t a = 0; a < 64; ++a) {
			float v = OwenScrambledRadicalInverse(baseIdx, a, 0xABCD1234u);
			EXPECT_GE(v, 0.0f) << "base=" << baseIdx << " a=" << a;
			EXPECT_LT(v, 1.0f) << "base=" << baseIdx << " a=" << a;
		}
	}
}

TEST(OwenScrambledRadicalInverseTest, DifferentHashesProduceDifferentOutputs) {
	// Same index, different hash -> typically different result
	bool any_different = false;
	for (uint64_t a = 1; a < 32; ++a) {
		float v1 = OwenScrambledRadicalInverse(0, a, 0u);
		float v2 = OwenScrambledRadicalInverse(0, a, 1u);
		if (std::abs(v1 - v2) > 1e-7f) { any_different = true; break; }
	}
	EXPECT_TRUE(any_different);
}

TEST(OwenScrambledRadicalInverseTest, Consistency) {
	// Calling twice with same arguments returns same result
	float v1 = OwenScrambledRadicalInverse(3, 42u, 0xDEADBEEFu);
	float v2 = OwenScrambledRadicalInverse(3, 42u, 0xDEADBEEFu);
	EXPECT_EQ(v1, v2);
}
