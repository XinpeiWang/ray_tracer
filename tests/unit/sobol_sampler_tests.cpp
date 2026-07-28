// sobol_sampler_tests.cpp
// Unit tests for src/shared/sobol_sampler.h
//
// Tests are grouped into:
//   1. Helper functions -- ReverseBits32, MixBits, FastOwenScrambler
//   2. sobol_sample     -- range, dim-0 Van der Corput property, dim independence
//   3. SobolSampler     -- interface, range, uniformity, pixel independence,
//                          fallback for extra dims, reset_dim

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include "../../src/shared/sobol_sampler.h"

// ---------------------------------------------------------------------------
// 1. Helper functions
// ---------------------------------------------------------------------------

TEST(ReverseBits32, Identity) {
	// reverse twice == identity
	for (uint32_t v : {0u, 1u, 0x80000000u, 0xDEADBEEFu, 0xFFFFFFFFu}) {
		EXPECT_EQ(ReverseBits32(ReverseBits32(v)), v);
	}
}

TEST(ReverseBits32, KnownValues) {
	EXPECT_EQ(ReverseBits32(0x00000001u), 0x80000000u);
	EXPECT_EQ(ReverseBits32(0x80000000u), 0x00000001u);
	EXPECT_EQ(ReverseBits32(0xFFFFFFFFu), 0xFFFFFFFFu);
	EXPECT_EQ(ReverseBits32(0x00000000u), 0x00000000u);
}

TEST(MixBits, OutputNotEqualInput) {
	// MixBits (pbrt_hash.h); should change value for typical inputs
	for (uint64_t v : {1ull, 42ull, 0xDEADBEEFull, 1000000ull}) {
		EXPECT_NE(MixBits(v), v);
	}
}

TEST(MixBits, ZeroStable) {
	// Not required to be 0, but must be deterministic
	EXPECT_EQ(MixBits(0), MixBits(0));
}

TEST(FastOwenScramble, DeterministicSameSeed) {
	uint32_t v = 0x12345678u, seed = 0xABCDEF01u;
	EXPECT_EQ(FastOwenScrambler{seed}(v), FastOwenScrambler{seed}(v));
}

TEST(FastOwenScramble, DifferentSeedsDifferentOutput) {
	uint32_t v = 0x12345678u;
	EXPECT_NE(FastOwenScrambler{1u}(v), FastOwenScrambler{2u}(v));
}

TEST(FastOwenScramble, CoversBitRange) {
	// Over many inputs with different seeds, scrambled values should spread
	// across both the high and low halves of the 32-bit range.
	uint32_t low_count = 0, high_count = 0;
	for (uint32_t i = 0; i < 1000; ++i) {
		uint32_t s = FastOwenScrambler{i * 2654435761u + 1013904223u}(i);
		if (s < 0x80000000u) low_count++;
		else                  high_count++;
	}
	// Expect reasonable spread -- neither half should capture all values
	EXPECT_GT(low_count,  100u) << "Too few values in lower half";
	EXPECT_GT(high_count, 100u) << "Too few values in upper half";
}

// ---------------------------------------------------------------------------
// 2. sobol_sample
// ---------------------------------------------------------------------------

TEST(SobolSample, RangeAllDims) {
	for (int dim = 0; dim < SOBOL_DIMS; ++dim) {
		for (int i = 0; i < 256; ++i) {
			double v = sobol_sample(static_cast<uint64_t>(i), dim, 0u);
			EXPECT_GE(v, 0.0) << "dim=" << dim << " i=" << i;
			EXPECT_LT(v, 1.0) << "dim=" << dim << " i=" << i;
		}
	}
}

TEST(SobolSample, Dim0IsVanDerCorput) {
	// Dim 0 with seed 0 is unscrambled Van der Corput (base 2).
	// sobol_sample(1, 0, 0) = 0.5, sobol_sample(2, 0, 0) = 0.25, etc.
	// With seed=0, FastOwenScrambler{0}(v) may alter bits; test the
	// property that 2^N consecutive samples cover [0,1) uniformly.
	std::vector<double> vals;
	for (int i = 0; i < 16; ++i)
		vals.push_back(sobol_sample(static_cast<uint64_t>(i), 0, 0u));
	std::sort(vals.begin(), vals.end());
	// Each value should be in its own bin of width 1/16
	for (int i = 0; i < 16; ++i) {
		double bin_lo = i / 16.0, bin_hi = (i + 1) / 16.0;
		EXPECT_GE(vals[i], bin_lo) << "bin " << i;
		EXPECT_LT(vals[i], bin_hi) << "bin " << i;
	}
}

TEST(SobolSample, DifferentDimsUncorrelated) {
	// Dim 0 and dim 1 should not be identical
	bool any_different = false;
	for (int i = 0; i < 64; ++i) {
		if (sobol_sample(i, 0, 42u) != sobol_sample(i, 1, 42u)) {
			any_different = true;
			break;
		}
	}
	EXPECT_TRUE(any_different);
}

TEST(SobolSample, Deterministic) {
	double v1 = sobol_sample(123ull, 3, 0xCAFEBABEu);
	double v2 = sobol_sample(123ull, 3, 0xCAFEBABEu);
	EXPECT_DOUBLE_EQ(v1, v2);
}

// ---------------------------------------------------------------------------
// 3. SobolSampler -- interface, uniformity, independence
// ---------------------------------------------------------------------------

TEST(SobolSampler, GetInRange) {
	SobolSampler ss(0, 100, 200);
	for (int i = 0; i < SOBOL_DIMS + 4; ++i) {
		double v = ss.get();
		EXPECT_GE(v, 0.0) << "dim=" << i;
		EXPECT_LT(v, 1.0) << "dim=" << i;
	}
}

TEST(SobolSampler, Deterministic) {
	SobolSampler a(7, 42, 99);
	SobolSampler b(7, 42, 99);
	for (int i = 0; i < SOBOL_DIMS; ++i)
		EXPECT_DOUBLE_EQ(a.get(), b.get()) << "dim=" << i;
}

TEST(SobolSampler, ResetDim) {
	SobolSampler ss(5, 10, 20);
	double first = ss.get();
	ss.reset_dim(0);
	double again = ss.get();
	EXPECT_DOUBLE_EQ(first, again);
}

TEST(SobolSampler, UniformitySingleDim) {
	// 256 consecutive samples for dim 0 should cover [0,1) with
	// low discrepancy: each 1/16 bin must be hit at least 8 times.
	std::vector<int> bins(16, 0);
	for (int s = 0; s < 256; ++s) {
		SobolSampler ss(s, 0, 0);
		double v = ss.get();
		bins[static_cast<int>(v * 16)]++;
	}
	for (int b = 0; b < 16; ++b)
		EXPECT_GE(bins[b], 8) << "bin " << b << " underpopulated";
}

TEST(SobolSampler, PixelIndependence) {
	// Two pixels with the same sample index should produce different dim-0 values
	SobolSampler a(0, 0, 0);
	SobolSampler b(0, 1, 0);
	SobolSampler c(0, 0, 1);
	EXPECT_NE(a.get(), b.get());
	// Reset and check c
	SobolSampler a2(0, 0, 0);
	EXPECT_NE(a2.get(), c.get());
}

TEST(SobolSampler, FallbackBeyondMaxDims) {
	// After SOBOL_DIMS calls, fallback LCG must still return [0,1)
	SobolSampler ss(3, 7, 8);
	for (int i = 0; i < SOBOL_DIMS; ++i) ss.get();  // exhaust Sobol dims
	for (int i = 0; i < 8; ++i) {
		double v = ss.get();
		EXPECT_GE(v, 0.0) << "fallback dim " << i;
		EXPECT_LT(v, 1.0) << "fallback dim " << i;
	}
}

TEST(SobolSampler, InterfaceMatchesPathSamplerPattern) {
	// Verifies the typical integrator usage pattern compiles and runs
	SobolSampler ss(42, 320, 240);
	double rr_dim0 = ss.get();
	double rr_dim1 = ss.get();
	EXPECT_GE(rr_dim0, 0.0); EXPECT_LT(rr_dim0, 1.0);
	EXPECT_GE(rr_dim1, 0.0); EXPECT_LT(rr_dim1, 1.0);
	EXPECT_NE(rr_dim0, rr_dim1);  // consecutive dims differ
}

TEST(SobolSampler, SampleIndexAdvancesSequence) {
	// Different sample indices for the same pixel must produce different values
	SobolSampler s0(0, 50, 60);
	SobolSampler s1(1, 50, 60);
	EXPECT_NE(s0.get(), s1.get());
}
