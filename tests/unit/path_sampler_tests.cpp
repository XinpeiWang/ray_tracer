// path_sampler_tests.cpp -- pbrt-v4-style tests for the Halton PathSampler
//
// Tests verify:
//   1. radical_inverse produces correct values for known inputs
//   2. PathSampler outputs are in [0, 1)
//   3. Successive samples from different dimensions are distinct (LD property)
//   4. Different pixels produce different sample streams (per-pixel scrambling)
//   5. Mean coverage: N samples from any dimension average near 0.5 (LD uniformity)
//   6. Fallback beyond max_dims still produces values in [0, 1)

#include <gtest/gtest.h>
#include "../../src/shared/path_sampler.h"
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

// ---------------------------------------------------------------------------
// Helper: run PathSampler for `n_dims` dims on sample 0 and return values
// ---------------------------------------------------------------------------
static std::vector<double> sample_dims(int sample_idx, int px, int py, int n_dims) {
	PathSampler ps(sample_idx, px, py);
	std::vector<double> vals(n_dims);
	for (int d = 0; d < n_dims; ++d)
		vals[d] = ps.get();
	return vals;
}

// ---------------------------------------------------------------------------
// RadicalInverseTest
// ---------------------------------------------------------------------------
TEST(RadicalInverseTest, Base2KnownValues) {
	// radical_inverse(2, n) = bit-reverse of n / 2^bits
	// a=1 -> 0.5, a=2 -> 0.25, a=3 -> 0.75
	EXPECT_NEAR(radical_inverse(2, 1), 0.5,  1e-12);
	EXPECT_NEAR(radical_inverse(2, 2), 0.25, 1e-12);
	EXPECT_NEAR(radical_inverse(2, 3), 0.75, 1e-12);
}

TEST(RadicalInverseTest, Base3KnownValues) {
	// a=1 -> 1/3, a=2 -> 2/3, a=3 -> 1/9
	EXPECT_NEAR(radical_inverse(3, 1), 1.0/3.0,  1e-12);
	EXPECT_NEAR(radical_inverse(3, 2), 2.0/3.0,  1e-12);
	EXPECT_NEAR(radical_inverse(3, 3), 1.0/9.0,  1e-12);
}

TEST(RadicalInverseTest, ZeroReturnsZero) {
	EXPECT_EQ(radical_inverse(2, 0), 0.0);
	EXPECT_EQ(radical_inverse(3, 0), 0.0);
}

TEST(RadicalInverseTest, AlwaysInUnitInterval) {
	for (int base : {2, 3, 5, 7}) {
		for (uint64_t a = 0; a < 1000; ++a) {
			double v = radical_inverse(base, a);
			EXPECT_GE(v, 0.0) << "base=" << base << " a=" << a;
			EXPECT_LT(v, 1.0) << "base=" << base << " a=" << a;
		}
	}
}

// ---------------------------------------------------------------------------
// PathSamplerRangeTest
// ---------------------------------------------------------------------------
TEST(PathSamplerRangeTest, AllDimensionsInUnitInterval) {
	for (int sample_idx = 0; sample_idx < 64; ++sample_idx) {
		PathSampler ps(sample_idx, 10, 20);
		for (int d = 0; d < PathSampler::max_dims + 4; ++d) {
			double v = ps.get();
			EXPECT_GE(v, 0.0) << "dim=" << d << " sample=" << sample_idx;
			EXPECT_LT(v, 1.0) << "dim=" << d << " sample=" << sample_idx;
		}
	}
}

TEST(PathSamplerRangeTest, DifferentSamplesGiveDifferentValues) {
	// Within the same pixel and dimension, different sample indices should differ.
	// Note: sample_idx=0 gives a=0 in radical inverse (always 0); use indices > 0.
	PathSampler ps1(1, 5, 7);
	PathSampler ps2(2, 5, 7);
	double v1 = ps1.get();
	double v2 = ps2.get();
	EXPECT_NE(v1, v2);
}

// ---------------------------------------------------------------------------
// PathSamplerUniformityTest -- check LD uniformity
// ---------------------------------------------------------------------------
TEST(PathSamplerUniformityTest, MeanNearHalfForLargeSampleCount) {
	// The first N Halton values should average near 0.5 for each dimension.
	// (This is the defining property of low-discrepancy sequences.)
	constexpr int N = 1024;
	constexpr int px = 0, py = 0;
	// Test first 8 dimensions
	for (int d = 0; d < 8; ++d) {
		double sum = 0.0;
		for (int s = 0; s < N; ++s) {
			PathSampler ps(s, px, py);
			// Advance to dimension d
			for (int k = 0; k < d; ++k) ps.get();
			sum += ps.get();
		}
		double mean = sum / N;
		// Allow wider tolerance since scrambling shifts means slightly
		EXPECT_NEAR(mean, 0.5, 0.05) << "dimension=" << d;
	}
}

TEST(PathSamplerUniformityTest, DiscrepancyLessThanRandom) {
	// Measure star discrepancy as max deviation of CDF from uniform.
	// A Halton sequence should have smaller discrepancy than pseudo-random.
	constexpr int N = 256;
	constexpr int px = 3, py = 7;

	std::vector<double> halton_vals(N);
	for (int s = 0; s < N; ++s) {
		PathSampler ps(s, px, py);
		halton_vals[s] = ps.get();   // dimension 0
	}
	std::sort(halton_vals.begin(), halton_vals.end());

	double max_dev = 0.0;
	for (int i = 0; i < N; ++i) {
		double expected = (i + 0.5) / N;
		max_dev = std::max(max_dev, std::abs(halton_vals[i] - expected));
	}
	// Halton base-2, N=256 -> discrepancy should be well below 0.05
	EXPECT_LT(max_dev, 0.05) << "Halton star discrepancy too high: " << max_dev;
}

// ---------------------------------------------------------------------------
// PathSamplerPixelIndependenceTest
// ---------------------------------------------------------------------------
TEST(PathSamplerPixelIndependenceTest, DifferentPixelsDifferentStreams) {
	// Two different pixels at the same sample index should produce different streams.
	// Use sample_idx=1 (a > 0) so scrambling affects the output.
	auto v0 = sample_dims(1, 10, 20, 4);
	auto v1 = sample_dims(1, 11, 20, 4);
	auto v2 = sample_dims(1, 10, 21, 4);

	// At least one dimension should differ between different pixels
	bool diff_01 = false, diff_02 = false;
	for (int d = 0; d < 4; ++d) {
		if (v0[d] != v1[d]) diff_01 = true;
		if (v0[d] != v2[d]) diff_02 = true;
	}
	EXPECT_TRUE(diff_01) << "Pixels (10,20) and (11,20) produce identical streams";
	EXPECT_TRUE(diff_02) << "Pixels (10,20) and (10,21) produce identical streams";
}

// ---------------------------------------------------------------------------
// PathSamplerFallbackTest
// ---------------------------------------------------------------------------
TEST(PathSamplerFallbackTest, BeyondMaxDimsStillInRange) {
	PathSampler ps(42, 0, 0);
	// Exhaust all pre-computed dimensions
	for (int d = 0; d < PathSampler::max_dims; ++d)
		ps.get();
	// Fallback dimensions should still be in [0,1)
	for (int d = 0; d < 8; ++d) {
		double v = ps.get();
		EXPECT_GE(v, 0.0);
		EXPECT_LT(v, 1.0);
	}
}

TEST(PathSamplerFallbackTest, ResetDimResetsToZero) {
	PathSampler ps(5, 2, 3);
	double first_call = ps.get();
	ps.reset_dim(0);
	double after_reset = ps.get();
	EXPECT_EQ(first_call, after_reset);
}
