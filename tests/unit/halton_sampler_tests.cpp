// halton_sampler_tests.cpp -- Unit tests for pbrt-v4-style Halton sampler
// pbrt-v4 reference: src/pbrt/samplers.h (HaltonSampler)
//                    src/pbrt/util/lowdiscrepancy.h

#include "../../src/shared/halton_sampler.h"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Van der Corput discrepancy bound: for n points in [0,1) the star-discrepancy
// is O(log(n)/n). We check that a simple histogram is reasonably uniform.
static double max_deviation_from_uniform(const std::vector<double>& samples,
										  int n_bins) {
	std::vector<int> counts(n_bins, 0);
	for (double v : samples) {
		int b = std::min((int)(v * n_bins), n_bins - 1);
		counts[b]++;
	}
	double expected = (double)samples.size() / n_bins;
	double max_dev = 0.0;
	for (int c : counts)
		max_dev = std::max(max_dev, std::abs(c - expected) / expected);
	return max_dev;
}

// ---------------------------------------------------------------------------
// radical_inverse correctness tests
// ---------------------------------------------------------------------------

// Canonical values (pbrt-v4 §8.3):
//   RadicalInverse(0 [base 2], 1) = 0.5    (binary: 1  -> .1)
//   RadicalInverse(0 [base 2], 2) = 0.25   (binary: 10 -> .01)
//   RadicalInverse(0 [base 2], 3) = 0.75   (binary: 11 -> .11)
//   RadicalInverse(1 [base 3], 1) = 1/3
//   RadicalInverse(1 [base 3], 2) = 2/3
//   RadicalInverse(1 [base 3], 3) = 1/9
TEST(HaltonRadicalInverse, KnownValues) {
	using halton_detail::radical_inverse;
	EXPECT_NEAR(radical_inverse(0, 1), 0.5,        1e-15);
	EXPECT_NEAR(radical_inverse(0, 2), 0.25,       1e-15);
	EXPECT_NEAR(radical_inverse(0, 3), 0.75,       1e-15);
	EXPECT_NEAR(radical_inverse(1, 1), 1.0/3.0,    1e-15);
	EXPECT_NEAR(radical_inverse(1, 2), 2.0/3.0,    1e-15);
	EXPECT_NEAR(radical_inverse(1, 3), 1.0/9.0,    1e-15);
}

TEST(HaltonRadicalInverse, ZeroInputIsZero) {
	using halton_detail::radical_inverse;
	EXPECT_EQ(radical_inverse(0, 0), 0.0);
	EXPECT_EQ(radical_inverse(1, 0), 0.0);
	EXPECT_EQ(radical_inverse(5, 0), 0.0);
}

TEST(HaltonRadicalInverse, OutputInUnitInterval) {
	using halton_detail::radical_inverse;
	for (int base_idx = 0; base_idx < 20; ++base_idx)
		for (uint64_t i = 0; i < 256; ++i) {
			double v = radical_inverse(base_idx, i);
			EXPECT_GE(v, 0.0) << "base_idx=" << base_idx << " i=" << i;
			EXPECT_LT(v, 1.0) << "base_idx=" << base_idx << " i=" << i;
		}
}

// The first n values of the base-2 van der Corput sequence should be
// exactly {k/n : k=0..n-1} for n = 2^m.
TEST(HaltonRadicalInverse, Base2PowerOfTwoPermutation) {
	using halton_detail::radical_inverse;
	const int n = 16;
	std::vector<double> vals(n);
	for (int i = 0; i < n; ++i)
		vals[i] = radical_inverse(0, (uint64_t)i);
	std::sort(vals.begin(), vals.end());
	for (int i = 0; i < n; ++i)
		EXPECT_NEAR(vals[i], (double)i / n, 1e-14) << "i=" << i;
}

// ---------------------------------------------------------------------------
// inverse_radical_inverse roundtrip
// ---------------------------------------------------------------------------
TEST(HaltonRadicalInverse, InverseRoundtrip) {
	using namespace halton_detail;
	// For base 2, 10 digits: InverseRadicalInverse(RadicalInverse(0, a)*2^10) == a % 2^10
	for (uint64_t a = 0; a < 64; ++a) {
		double ri = radical_inverse(0, a);
		// Recover digit string: multiply back
		uint64_t scaled = (uint64_t)std::round(ri * (1 << 10));
		uint64_t recovered = inverse_radical_inverse(scaled, 2, 10);
		EXPECT_EQ(recovered, a % (1 << 10)) << "a=" << a;
	}
}

// ---------------------------------------------------------------------------
// DigitPermutation tests
// ---------------------------------------------------------------------------
TEST(HaltonDigitPermutation, PermutationIsABijection) {
	using halton_detail::DigitPermutation;
	// For base 5, each digit slot should be a permutation of {0..4}
	DigitPermutation dp(5, 42u);
	for (int di = 0; di < dp.n_digits; ++di) {
		std::vector<int> seen(5, 0);
		for (int v = 0; v < 5; ++v)
			seen[dp.permute(di, v)]++;
		for (int v = 0; v < 5; ++v)
			EXPECT_EQ(seen[v], 1) << "digit_index=" << di << " value=" << v;
	}
}

TEST(HaltonDigitPermutation, DifferentSeedsGiveDifferentPermutations) {
	using halton_detail::DigitPermutation;
	DigitPermutation dp1(7, 0u);
	DigitPermutation dp2(7, 99u);
	bool differs = false;
	for (int di = 0; di < dp1.n_digits && !differs; ++di)
		for (int v = 0; v < 7 && !differs; ++v)
			if (dp1.permute(di, v) != dp2.permute(di, v))
				differs = true;
	EXPECT_TRUE(differs);
}

// ---------------------------------------------------------------------------
// halton_sampler interface tests
// ---------------------------------------------------------------------------

TEST(HaltonSampler, Get1DInUnitInterval) {
	halton_sampler s(16, 64, 64, HaltonRandomize::None);
	for (int px = 0; px < 4; ++px)
		for (int py = 0; py < 4; ++py)
			for (int si = 0; si < 16; ++si) {
				s.start_pixel_sample(px, py, si);
				double v = s.get_1d();
				EXPECT_GE(v, 0.0);
				EXPECT_LT(v, 1.0);
			}
}

TEST(HaltonSampler, Get2DInUnitSquare) {
	halton_sampler s(16, 64, 64, HaltonRandomize::None);
	for (int si = 0; si < 64; ++si) {
		s.start_pixel_sample(3, 7, si);
		auto [u, v] = s.get_2d();
		EXPECT_GE(u, 0.0); EXPECT_LT(u, 1.0);
		EXPECT_GE(v, 0.0); EXPECT_LT(v, 1.0);
	}
}

// All spp samples from the same pixel should be distinct (Halton sequences
// are injective for the first few hundred samples).
TEST(HaltonSampler, SamplesAreDistinctWithinPixel) {
	halton_sampler s(128, 512, 512, HaltonRandomize::None);
	std::vector<double> vals;
	vals.reserve(128);
	for (int si = 0; si < 128; ++si) {
		s.start_pixel_sample(0, 0, si);
		vals.push_back(s.get_1d());
	}
	std::sort(vals.begin(), vals.end());
	for (size_t i = 1; i < vals.size(); ++i)
		EXPECT_NE(vals[i], vals[i-1]);
}

// Two different pixels with the same sample_index should yield different values.
TEST(HaltonSampler, DifferentPixelsGiveDifferentSamples) {
	halton_sampler s(64, 128, 128, HaltonRandomize::None);
	s.start_pixel_sample(0, 0, 0);
	double v00 = s.get_1d();

	s.start_pixel_sample(1, 0, 0);
	double v10 = s.get_1d();

	s.start_pixel_sample(0, 1, 0);
	double v01 = s.get_1d();

	EXPECT_NE(v00, v10);
	EXPECT_NE(v00, v01);
	EXPECT_NE(v10, v01);
}

// Halton 1D sequence across many samples should be well-distributed.
TEST(HaltonSampler, Get1DEquidistribution) {
	const int n = 256;
	halton_sampler s(n, 512, 512, HaltonRandomize::None);
	std::vector<double> vals(n);
	for (int i = 0; i < n; ++i) {
		s.start_pixel_sample(0, 0, i);
		vals[i] = s.get_1d();
	}
	// Expect max bin deviation < 30% with 16 bins
	double dev = max_deviation_from_uniform(vals, 16);
	EXPECT_LT(dev, 0.30) << "max bin deviation = " << dev;
}

// Scrambled sampler samples should also be in [0,1)
TEST(HaltonSampler, ScrambledInUnitInterval) {
	halton_sampler s(64, 128, 128, HaltonRandomize::PermuteDigits, 1234u);
	for (int si = 0; si < 64; ++si) {
		s.start_pixel_sample(5, 3, si);
		double v = s.get_1d();
		EXPECT_GE(v, 0.0);
		EXPECT_LT(v, 1.0);
	}
}

// Scrambled and unscrambled should differ (with overwhelming probability)
TEST(HaltonSampler, ScrambledDiffersFromUnscrambled) {
	halton_sampler plain(32, 64, 64, HaltonRandomize::None);
	halton_sampler scrambled(32, 64, 64, HaltonRandomize::PermuteDigits, 7u);
	bool found_diff = false;
	for (int si = 0; si < 32 && !found_diff; ++si) {
		plain.start_pixel_sample(0, 0, si);
		scrambled.start_pixel_sample(0, 0, si);
		if (plain.get_1d() != scrambled.get_1d())
			found_diff = true;
	}
	EXPECT_TRUE(found_diff);
}

// get_pixel_2d should always be in [0,1)^2
TEST(HaltonSampler, GetPixel2DInUnitSquare) {
	halton_sampler s(64, 128, 128, HaltonRandomize::None);
	for (int px = 0; px < 8; ++px)
		for (int py = 0; py < 8; ++py)
			for (int si = 0; si < 8; ++si) {
				s.start_pixel_sample(px, py, si);
				auto [u, v] = s.get_pixel_2d();
				EXPECT_GE(u, 0.0); EXPECT_LT(u, 1.0);
				EXPECT_GE(v, 0.0); EXPECT_LT(v, 1.0);
			}
}

// samples_per_pixel accessor
TEST(HaltonSampler, SamplesPerPixelAccessor) {
	halton_sampler s(42, 100, 100);
	EXPECT_EQ(s.samples_per_pixel(), 42);
}
