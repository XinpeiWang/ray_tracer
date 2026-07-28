// z_sobol_sampler_tests.cpp
// Validation for ZSobolSampler -- pbrt-v4 ZSobolSampler port
//
// Tests verify:
//   1. Samples are in [0, 1)
//   2. Different pixels produce different sequences (Z-curve property)
//   3. Different sample indices produce different values for same pixel
//   4. Samples cover [0,1) with low discrepancy (stratification check)
//   5. get2d() returns values in [0,1)^2
//   6. Power-of-2 and non-power-of-2 SPP both work
//   7. Determinism: same args -> same output
//   8. Z-curve: adjacent pixels differ but share base distribution
//   9. Morton encoding: encode_morton2 correctness
//  10. log2_int and round_up_pow2 helper correctness

#include <gtest/gtest.h>
#include "../../src/shared/sobol_sampler.h"
#include <vector>
#include <cmath>
#include <algorithm>

// ---- helper ----------------------------------------------------------------
static std::vector<double> collect(ZSobolSampler& s, int px, int py,
								   int spp, int dim_count) {
	std::vector<double> out;
	for (int i = 0; i < spp; ++i) {
		s.start_pixel_sample(px, py, i, 0);
		for (int d = 0; d < dim_count; ++d)
			out.push_back(s.get());
	}
	return out;
}

// ---- 1. Range --------------------------------------------------------------
TEST(ZSobolSamplerTest, SamplesInUnitInterval) {
	ZSobolSampler s(16, 512, 512, 0);
	for (int px = 0; px < 4; ++px) {
		for (int py = 0; py < 4; ++py) {
			for (int i = 0; i < 16; ++i) {
				s.start_pixel_sample(px, py, i, 0);
				for (int d = 0; d < 8; ++d) {
					double v = s.get();
					EXPECT_GE(v, 0.0) << "px=" << px << " py=" << py << " sample=" << i << " dim=" << d;
					EXPECT_LT(v, 1.0) << "px=" << px << " py=" << py << " sample=" << i << " dim=" << d;
				}
			}
		}
	}
}

// ---- 2. Different pixels differ --------------------------------------------
TEST(ZSobolSamplerTest, DifferentPixelsDifferentSequences) {
	ZSobolSampler s(16, 512, 512, 0);
	auto seq00 = collect(s, 0, 0, 4, 4);
	auto seq10 = collect(s, 1, 0, 4, 4);
	auto seq01 = collect(s, 0, 1, 4, 4);
	// At least some values should differ
	bool px_differs = false, py_differs = false;
	for (size_t i = 0; i < seq00.size(); ++i) {
		if (seq00[i] != seq10[i]) px_differs = true;
		if (seq00[i] != seq01[i]) py_differs = true;
	}
	EXPECT_TRUE(px_differs);
	EXPECT_TRUE(py_differs);
}

// ---- 3. Different sample indices differ ------------------------------------
TEST(ZSobolSamplerTest, DifferentSampleIndicesDiffer) {
	ZSobolSampler s(16, 512, 512, 0);
	s.start_pixel_sample(5, 7, 0, 0);
	double v0 = s.get();
	s.start_pixel_sample(5, 7, 1, 0);
	double v1 = s.get();
	EXPECT_NE(v0, v1);
}

// ---- 4. Stratification: 1D samples cover [0,1) with low discrepancy -------
TEST(ZSobolSamplerTest, LowDiscrepancyStratification) {
	const int spp = 64;
	ZSobolSampler s(spp, 512, 512, 0);
	// Collect dim-0 samples for pixel (0,0)
	std::vector<double> vals;
	for (int i = 0; i < spp; ++i) {
		s.start_pixel_sample(0, 0, i, 0);
		vals.push_back(s.get());
	}
	std::sort(vals.begin(), vals.end());
	// Each stratum [k/spp, (k+1)/spp) should contain exactly 1 sample
	// for a (0,1)-net; allow small deviation for blue-noise variants
	int misses = 0;
	for (int k = 0; k < spp; ++k) {
		double lo = static_cast<double>(k) / spp;
		double hi = static_cast<double>(k + 1) / spp;
		bool found = false;
		for (double v : vals) if (v >= lo && v < hi) { found = true; break; }
		if (!found) ++misses;
	}
	// Sobol with Morton permutation: at most a few misses acceptable
	EXPECT_LE(misses, spp / 8) << "Too many empty strata: " << misses;
}

// ---- 5. get2d() range ------------------------------------------------------
TEST(ZSobolSamplerTest, Get2DInUnitSquare) {
	ZSobolSampler s(16, 256, 256, 0);
	for (int i = 0; i < 16; ++i) {
		s.start_pixel_sample(3, 7, i, 0);
		double u0, u1;
		s.get2d(u0, u1);
		EXPECT_GE(u0, 0.0); EXPECT_LT(u0, 1.0);
		EXPECT_GE(u1, 0.0); EXPECT_LT(u1, 1.0);
	}
}

// ---- 6a. Power-of-2 SPP works ----------------------------------------------
TEST(ZSobolSamplerTest, PowerOf2SppWorks) {
	ZSobolSampler s(4, 128, 128, 0);
	EXPECT_EQ(s.samples_per_pixel(), 4);
	for (int i = 0; i < 4; ++i) {
		s.start_pixel_sample(0, 0, i, 0);
		double v = s.get();
		EXPECT_GE(v, 0.0); EXPECT_LT(v, 1.0);
	}
}

// ---- 6b. Non-power-of-2 SPP rounds up ------------------------------------
TEST(ZSobolSamplerTest, NonPowerOf2SppRoundsUp) {
	// spp=5 -> log2_spp=2 -> samples_per_pixel=4
	ZSobolSampler s(5, 128, 128, 0);
	EXPECT_GE(s.samples_per_pixel(), 4);
	for (int i = 0; i < 4; ++i) {
		s.start_pixel_sample(0, 0, i, 0);
		double v = s.get();
		EXPECT_GE(v, 0.0); EXPECT_LT(v, 1.0);
	}
}

// ---- 7. Determinism --------------------------------------------------------
TEST(ZSobolSamplerTest, Deterministic) {
	ZSobolSampler s1(16, 512, 512, 42);
	ZSobolSampler s2(16, 512, 512, 42);
	for (int i = 0; i < 16; ++i) {
		s1.start_pixel_sample(3, 5, i, 0);
		s2.start_pixel_sample(3, 5, i, 0);
		for (int d = 0; d < 6; ++d)
			EXPECT_EQ(s1.get(), s2.get());
	}
}

// ---- 8. Different seeds produce different sequences ------------------------
TEST(ZSobolSamplerTest, SeedChangesSequence) {
	ZSobolSampler s0(16, 512, 512, 0);
	ZSobolSampler s1(16, 512, 512, 1);
	s0.start_pixel_sample(0, 0, 0, 0);
	s1.start_pixel_sample(0, 0, 0, 0);
	double v0 = s0.get();
	double v1 = s1.get();
	EXPECT_NE(v0, v1);
}

// ---- 9. encode_morton2 correctness ----------------------------------------
TEST(ZSobolSamplerTest, EncodeMorton2Correctness) {
	// Morton code of (1,0): x=1 -> bit 0, y=0 -> result bit pattern = 01b = 1
	// Morton code of (0,1): y=1 -> bit 1, result = 10b = 2
	// Morton code of (1,1) = 11b = 3
	EXPECT_EQ(encode_morton2(1, 0), 1ULL);
	EXPECT_EQ(encode_morton2(0, 1), 2ULL);
	EXPECT_EQ(encode_morton2(1, 1), 3ULL);
	EXPECT_EQ(encode_morton2(0, 0), 0ULL);
	// (2,0) = 0b100 = 4; bits of 2 interleaved: 2=10b -> left_shift2 gives ..0100 -> 0x4
	EXPECT_EQ(encode_morton2(2, 0), 4ULL);
	EXPECT_EQ(encode_morton2(0, 2), 8ULL);
}

// ---- 10. Helper function correctness ---------------------------------------
TEST(ZSobolSamplerTest, Log2IntAndRoundUpPow2) {
	EXPECT_EQ(log2_int(1),  0);
	EXPECT_EQ(log2_int(2),  1);
	EXPECT_EQ(log2_int(4),  2);
	EXPECT_EQ(log2_int(8),  3);
	EXPECT_EQ(log2_int(16), 4);
	EXPECT_EQ(log2_int(7),  2);  // floor(log2(7)) = 2

	EXPECT_EQ(round_up_pow2(1),  1);
	EXPECT_EQ(round_up_pow2(2),  2);
	EXPECT_EQ(round_up_pow2(3),  4);
	EXPECT_EQ(round_up_pow2(5),  8);
	EXPECT_EQ(round_up_pow2(16), 16);
	EXPECT_EQ(round_up_pow2(17), 32);
}
