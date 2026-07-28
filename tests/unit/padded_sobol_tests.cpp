// padded_sobol_tests.cpp
// Validation for PermutationElement and PaddedSobolSampler -- pbrt-v4 ports
//
// Tests:
//   1.  permutation_element: output in [0, l)
//   2.  permutation_element: is a permutation (bijection) for power-of-2 l
//   3.  permutation_element: is a permutation for non-power-of-2 l
//   4.  permutation_element: different seeds give different permutations
//   5.  PaddedSobolSampler: samples in [0, 1)
//   6.  PaddedSobolSampler: different pixels produce different sequences
//   7.  PaddedSobolSampler: different sample indices produce different values
//   8.  PaddedSobolSampler: deterministic (same args -> same output)
//   9.  PaddedSobolSampler: get2d() output in [0,1)^2
//  10.  PaddedSobolSampler: stratification -- samples cover [0,1) with low discrepancy
//  11.  PaddedSobolSampler: different seeds produce different sequences
//  12.  PaddedSobolSampler: dimension counter advances correctly

#include <gtest/gtest.h>
#include "../../src/shared/sobol_sampler.h"
#include <vector>
#include <algorithm>
#include <unordered_set>

// ---- 1. permutation_element: output in [0, l) ----------------------------
TEST(PaddedSobolTest, PermutationElementRange) {
	for (uint32_t l : {3u, 4u, 7u, 8u, 16u, 100u}) {
		for (uint32_t i = 0; i < l; ++i) {
			int r = permutation_element(i, l, 0xdeadbeef);
			EXPECT_GE(r, 0)              << "l=" << l << " i=" << i;
			EXPECT_LT(r, (int)l)         << "l=" << l << " i=" << i;
		}
	}
}

// ---- 2. permutation_element: bijection for power-of-2 l ------------------
TEST(PaddedSobolTest, PermutationElementBijectionPow2) {
	uint32_t l = 16;
	std::vector<bool> seen(l, false);
	for (uint32_t i = 0; i < l; ++i) {
		int r = permutation_element(i, l, 0x12345678);
		EXPECT_FALSE(seen[r]) << "collision at i=" << i << " r=" << r;
		seen[r] = true;
	}
	for (uint32_t i = 0; i < l; ++i) EXPECT_TRUE(seen[i]) << "missing " << i;
}

// ---- 3. permutation_element: bijection for non-power-of-2 l --------------
TEST(PaddedSobolTest, PermutationElementBijectionNonPow2) {
	uint32_t l = 10;
	std::vector<bool> seen(l, false);
	for (uint32_t i = 0; i < l; ++i) {
		int r = permutation_element(i, l, 0xabcdef01);
		EXPECT_GE(r, 0); EXPECT_LT(r, (int)l);
		EXPECT_FALSE(seen[r]) << "collision at i=" << i;
		seen[r] = true;
	}
}

// ---- 4. permutation_element: different seeds give different permutations --
TEST(PaddedSobolTest, PermutationElementSeedDiffers) {
	uint32_t l = 8;
	bool any_diff = false;
	for (uint32_t i = 0; i < l; ++i) {
		if (permutation_element(i, l, 1) != permutation_element(i, l, 2))
			any_diff = true;
	}
	EXPECT_TRUE(any_diff);
}

// ---- 5. PaddedSobolSampler: samples in [0, 1) ----------------------------
TEST(PaddedSobolTest, SamplesInUnitInterval) {
	PaddedSobolSampler s(16, 0);
	for (int px = 0; px < 4; ++px)
		for (int py = 0; py < 4; ++py)
			for (int i = 0; i < 16; ++i) {
				s.start_pixel_sample(px, py, i, 0);
				for (int d = 0; d < 6; ++d) {
					double v = s.get();
					EXPECT_GE(v, 0.0); EXPECT_LT(v, 1.0);
				}
			}
}

// ---- 6. Different pixels produce different sequences ---------------------
TEST(PaddedSobolTest, DifferentPixelsDiffer) {
	PaddedSobolSampler s(16, 0);
	s.start_pixel_sample(0, 0, 0, 0); double v00 = s.get();
	s.start_pixel_sample(1, 0, 0, 0); double v10 = s.get();
	s.start_pixel_sample(0, 1, 0, 0); double v01 = s.get();
	EXPECT_NE(v00, v10);
	EXPECT_NE(v00, v01);
}

// ---- 7. Different sample indices differ ----------------------------------
TEST(PaddedSobolTest, DifferentSampleIndicesDiffer) {
	PaddedSobolSampler s(16, 0);
	s.start_pixel_sample(3, 5, 0, 0); double v0 = s.get();
	s.start_pixel_sample(3, 5, 1, 0); double v1 = s.get();
	EXPECT_NE(v0, v1);
}

// ---- 8. Deterministic ----------------------------------------------------
TEST(PaddedSobolTest, Deterministic) {
	PaddedSobolSampler s1(16, 42), s2(16, 42);
	for (int i = 0; i < 16; ++i) {
		s1.start_pixel_sample(2, 7, i, 0);
		s2.start_pixel_sample(2, 7, i, 0);
		for (int d = 0; d < 4; ++d) EXPECT_EQ(s1.get(), s2.get());
	}
}

// ---- 9. get2d() output in [0,1)^2 ----------------------------------------
TEST(PaddedSobolTest, Get2DRange) {
	PaddedSobolSampler s(16, 0);
	for (int i = 0; i < 16; ++i) {
		s.start_pixel_sample(1, 2, i, 0);
		double u0, u1;
		s.get2d(u0, u1);
		EXPECT_GE(u0, 0.0); EXPECT_LT(u0, 1.0);
		EXPECT_GE(u1, 0.0); EXPECT_LT(u1, 1.0);
	}
}

// ---- 10. Stratification: samples cover strata ----------------------------
TEST(PaddedSobolTest, LowDiscrepancyStratification) {
	const int spp = 16;
	PaddedSobolSampler s(spp, 0);
	std::vector<double> vals;
	for (int i = 0; i < spp; ++i) {
		s.start_pixel_sample(0, 0, i, 0);
		vals.push_back(s.get());
	}
	std::sort(vals.begin(), vals.end());
	int misses = 0;
	for (int k = 0; k < spp; ++k) {
		double lo = k / (double)spp, hi = (k+1) / (double)spp;
		bool found = false;
		for (double v : vals) if (v >= lo && v < hi) { found = true; break; }
		if (!found) ++misses;
	}
	EXPECT_LE(misses, spp / 4);
}

// ---- 11. Different seeds produce different sequences ---------------------
TEST(PaddedSobolTest, SeedChangesSequence) {
	PaddedSobolSampler s0(16, 0), s1(16, 1);
	s0.start_pixel_sample(0, 0, 0, 0);
	s1.start_pixel_sample(0, 0, 0, 0);
	EXPECT_NE(s0.get(), s1.get());
}

// ---- 12. Dimension counter advances correctly ----------------------------
TEST(PaddedSobolTest, DimensionAdvances) {
	PaddedSobolSampler s(16, 0);
	s.start_pixel_sample(0, 0, 0, 0);
	double v0 = s.get(); // dim 0 -> 1
	double v1 = s.get(); // dim 1 -> 2
	// Same pixel, same sample, different dims -> different values
	EXPECT_NE(v0, v1);
	// Reset and confirm reproducibility
	s.reset_dim(0);
	EXPECT_EQ(s.get(), v0);
}

// ---- 13. Power-of-2 spp gives perfect per-stratum coverage ---------------
// pbrt-v4 notes that non-power-of-2 spp is suboptimal for Sobol samplers.
// With spp=pow2, the permuted sample indices cover all strata exactly once.
TEST(PaddedSobolTest, PowerOf2PerfectStratification) {
	const int spp = 16; // power of 2
	PaddedSobolSampler s(spp, 42);
	std::vector<double> vals;
	for (int i = 0; i < spp; ++i) {
		s.start_pixel_sample(5, 7, i, 0);
		vals.push_back(s.get());
	}
	std::sort(vals.begin(), vals.end());
	// With power-of-2 spp, all 16 strata must be covered exactly
	int misses = 0;
	for (int k = 0; k < spp; ++k) {
		double lo = k / (double)spp, hi = (k+1) / (double)spp;
		bool found = false;
		for (double v : vals) if (v >= lo && v < hi) { found = true; break; }
		if (!found) ++misses;
	}
	EXPECT_EQ(misses, 0) << "Power-of-2 spp must achieve perfect stratification";
}

// ---- 14. get2d() returns samples from two distinct Sobol dimensions ------
TEST(PaddedSobolTest, Get2DUsesTwoDimensions) {
	PaddedSobolSampler s(16, 0);
	// Collect paired samples and verify they are not identical
	int n_distinct = 0;
	for (int i = 0; i < 16; ++i) {
		s.start_pixel_sample(0, 0, i, 0);
		double u0, u1;
		s.get2d(u0, u1);
		if (u0 != u1) ++n_distinct;
	}
	// Expect most 2D pairs to have distinct u0, u1 (different Sobol dims)
	EXPECT_GE(n_distinct, 10);
}
