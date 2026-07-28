// reservoir_sampler_tests.cpp
// Validation for PCG32Rng, WeightedReservoirSampler<T>, and AliasTable
// -- pbrt-v4 ports (src/shared/reservoir_sampler.h)
//
// Tests:
// PCG32Rng
//   1.  Output is in [0, 1)
//   2.  Different seeds produce different sequences
//   3.  Reproducible (same seed -> same sequence)
//   4.  Distribution is approximately uniform (chi-squared bucket test)
//
// WeightedReservoirSampler
//   5.  Empty reservoir has no sample
//   6.  Single add -> always accepted (probability 1.0)
//   7.  Two equal weights -> each selected ~50% of the time
//   8.  Selection probability is proportional to weight (statistical)
//   9.  weight_sum() accumulates correctly
//  10.  sample_probability() = reservoir_weight / weight_sum
//  11.  Merge of two equal reservoirs produces correct distribution
//  12.  Reset clears the reservoir
//  13.  Callback (lazy) add works correctly
//
// AliasTable
//  14.  PMF sums to 1.0
//  15.  PMF matches input weights (normalized)
//  16.  Sample returns index in [0, n)
//  17.  Uniform weights: each index sampled ~equally (chi-squared)
//  18.  Non-uniform weights: sampling frequencies proportional to weights
//  19.  Single-element table always returns index 0
//  20.  out_pmf parameter is filled correctly
//  21.  out_u_remapped is in [0, 1)
//  22.  Size-2 table with one dominant weight

#include <gtest/gtest.h>
#include "../../src/shared/reservoir_sampler.h"
#include <vector>
#include <numeric>
#include <cmath>

// ============================================================
// PCG32Rng tests
// ============================================================

// ---- 1. Output in [0, 1) ----------------------------------------
TEST(ReservoirPCG32Test, OutputInUnitInterval) {
	PCG32Rng rng(12345);
	for (int i = 0; i < 10000; ++i) {
		double v = rng.uniform();
		EXPECT_GE(v, 0.0);
		EXPECT_LT(v, 1.0);
	}
}

// ---- 2. Different seeds produce different sequences --------------
TEST(ReservoirPCG32Test, DifferentSeedsDiffer) {
	PCG32Rng r1(1), r2(2);
	bool any_diff = false;
	for (int i = 0; i < 20; ++i)
		if (r1.uniform() != r2.uniform()) { any_diff = true; break; }
	EXPECT_TRUE(any_diff);
}

// ---- 3. Reproducible --------------------------------------------
TEST(ReservoirPCG32Test, Reproducible) {
	PCG32Rng r1(99), r2(99);
	for (int i = 0; i < 50; ++i)
		EXPECT_EQ(r1.uniform(), r2.uniform());
}

// ---- 4. Approximately uniform distribution ----------------------
TEST(ReservoirPCG32Test, ApproximatelyUniform) {
	const int N = 100000, K = 10;
	PCG32Rng rng(42);
	std::vector<int> counts(K, 0);
	for (int i = 0; i < N; ++i)
		counts[(int)(rng.uniform() * K)]++;
	double expected = N / (double)K;
	for (int k = 0; k < K; ++k) {
		double diff = counts[k] - expected;
		EXPECT_LT(std::abs(diff) / expected, 0.05)
			<< "bucket " << k << " count=" << counts[k];
	}
}

// ============================================================
// WeightedReservoirSampler tests
// ============================================================

// ---- 5. Empty reservoir has no sample ---------------------------
TEST(WeightedReservoirSamplerTest, EmptyHasNoSample) {
	WeightedReservoirSampler<int> wrs;
	EXPECT_FALSE(wrs.has_sample());
	EXPECT_EQ(wrs.weight_sum(), 0.0);
}

// ---- 6. Single add always accepted ------------------------------
TEST(WeightedReservoirSamplerTest, SingleAddAlwaysAccepted) {
	for (int seed = 0; seed < 20; ++seed) {
		WeightedReservoirSampler<int> wrs(seed);
		bool accepted = wrs.add(42, 1.5);
		EXPECT_TRUE(accepted);
		EXPECT_TRUE(wrs.has_sample());
		EXPECT_EQ(wrs.sample(), 42);
	}
}

// ---- 7. Two equal weights -> ~50% each --------------------------
TEST(WeightedReservoirSamplerTest, TwoEqualWeightsHalfHalf) {
	const int N = 10000;
	int count0 = 0;
	for (int trial = 0; trial < N; ++trial) {
		WeightedReservoirSampler<int> wrs(trial);
		wrs.add(0, 1.0);
		wrs.add(1, 1.0);
		if (wrs.sample() == 0) ++count0;
	}
	double frac = count0 / (double)N;
	EXPECT_NEAR(frac, 0.5, 0.03) << "Expected ~50% for equal weights";
}

// ---- 8. Selection probability proportional to weight -----------
TEST(WeightedReservoirSamplerTest, WeightedSelection) {
	const int N = 20000;
	// weights: 1, 2, 3 -> expected fractions: 1/6, 2/6, 3/6
	std::vector<double> weights = {1.0, 2.0, 3.0};
	double sum_w = 6.0;
	std::vector<int> counts(3, 0);
	for (int trial = 0; trial < N; ++trial) {
		WeightedReservoirSampler<int> wrs(trial * 2654435761ull);
		for (int i = 0; i < 3; ++i) wrs.add(i, weights[i]);
		counts[wrs.sample()]++;
	}
	for (int i = 0; i < 3; ++i) {
		double expected = N * weights[i] / sum_w;
		EXPECT_NEAR(counts[i] / (double)N, weights[i] / sum_w, 0.03)
			<< "index " << i;
	}
}

// ---- 9. weight_sum accumulates correctly ------------------------
TEST(WeightedReservoirSamplerTest, WeightSumAccumulates) {
	WeightedReservoirSampler<int> wrs(0);
	wrs.add(0, 1.0);
	EXPECT_NEAR(wrs.weight_sum(), 1.0, 1e-12);
	wrs.add(1, 2.5);
	EXPECT_NEAR(wrs.weight_sum(), 3.5, 1e-12);
	wrs.add(2, 0.5);
	EXPECT_NEAR(wrs.weight_sum(), 4.0, 1e-12);
}

// ---- 10. sample_probability = reservoir_weight / weight_sum -----
TEST(WeightedReservoirSamplerTest, SampleProbabilityCorrect) {
	WeightedReservoirSampler<int> wrs(7);
	wrs.add(0, 3.0);
	wrs.add(1, 1.0);
	double sp = wrs.sample_probability();
	EXPECT_GT(sp, 0.0);
	EXPECT_LE(sp, 1.0);
	EXPECT_NEAR(sp, wrs.reservoir_weight() / wrs.weight_sum(), 1e-12);
}

// ---- 11. Merge: combined distribution correct -------------------
TEST(WeightedReservoirSamplerTest, MergeCorrectDistribution) {
	// Merge wrs_A (item 0, w=2) + wrs_B (item 1, w=2) -> 50/50
	const int N = 10000;
	int count0 = 0;
	for (int trial = 0; trial < N; ++trial) {
		WeightedReservoirSampler<int> a(trial);
		a.add(0, 2.0);
		WeightedReservoirSampler<int> b(trial + 1000000ull);
		b.add(1, 2.0);
		a.merge(b);
		if (a.sample() == 0) ++count0;
	}
	EXPECT_NEAR(count0 / (double)N, 0.5, 0.03);
}

// ---- 12. Reset clears the reservoir -----------------------------
TEST(WeightedReservoirSamplerTest, ResetClearsState) {
	WeightedReservoirSampler<int> wrs(0);
	wrs.add(42, 5.0);
	EXPECT_TRUE(wrs.has_sample());
	wrs.reset();
	EXPECT_FALSE(wrs.has_sample());
	EXPECT_EQ(wrs.weight_sum(), 0.0);
}

// ---- 13. Callback (lazy) add ------------------------------------
TEST(WeightedReservoirSamplerTest, LazyCallbackAdd) {
	WeightedReservoirSampler<int> wrs(0);
	int call_count = 0;
	// With weight 1, first add must accept (probability 1.0)
	bool accepted = wrs.add([&]() { ++call_count; return 99; }, 1.0);
	EXPECT_TRUE(accepted);
	EXPECT_EQ(call_count, 1);
	EXPECT_EQ(wrs.sample(), 99);
}

// ============================================================
// AliasTable tests
// ============================================================

// ---- 14. PMF sums to 1 ------------------------------------------
TEST(ReservoirAliasTableTest, PMFSumsToOne) {
	AliasTable at({1.0, 2.0, 3.0, 4.0});
	double sum = 0.0;
	for (int i = 0; i < at.size(); ++i) sum += at.pmf(i);
	EXPECT_NEAR(sum, 1.0, 1e-12);
}

// ---- 15. PMF matches normalized input weights -------------------
TEST(ReservoirAliasTableTest, PMFMatchesNormalizedWeights) {
	std::vector<double> w = {1.0, 3.0, 2.0};
	double sum_w = 6.0;
	AliasTable at(w);
	for (int i = 0; i < (int)w.size(); ++i)
		EXPECT_NEAR(at.pmf(i), w[i] / sum_w, 1e-12);
}

// ---- 16. Sample returns index in [0, n) --------------------------
TEST(ReservoirAliasTableTest, SampleInRange) {
	AliasTable at({2.0, 5.0, 1.0, 3.0});
	PCG32Rng rng(0);
	for (int i = 0; i < 1000; ++i) {
		int idx = at.sample(rng.uniform());
		EXPECT_GE(idx, 0);
		EXPECT_LT(idx, at.size());
	}
}

// ---- 17. Uniform weights -> equal frequencies -------------------
TEST(ReservoirAliasTableTest, UniformWeightsEqualFrequencies) {
	const int N = 5, TRIALS = 50000;
	AliasTable at(std::vector<double>(N, 1.0));
	PCG32Rng rng(123);
	std::vector<int> counts(N, 0);
	for (int t = 0; t < TRIALS; ++t) counts[at.sample(rng.uniform())]++;
	double expected = TRIALS / (double)N;
	for (int i = 0; i < N; ++i)
		EXPECT_NEAR(counts[i] / (double)TRIALS, 1.0 / N, 0.03)
			<< "index " << i;
}

// ---- 18. Non-uniform weights: frequencies proportional ----------
TEST(ReservoirAliasTableTest, NonUniformWeightsProportional) {
	std::vector<double> w = {1.0, 4.0, 2.0, 3.0};
	AliasTable at(w);
	double sum_w = 10.0;
	const int TRIALS = 100000;
	PCG32Rng rng(456);
	std::vector<int> counts(4, 0);
	for (int t = 0; t < TRIALS; ++t) counts[at.sample(rng.uniform())]++;
	for (int i = 0; i < 4; ++i)
		EXPECT_NEAR(counts[i] / (double)TRIALS, w[i] / sum_w, 0.02)
			<< "index " << i;
}

// ---- 19. Single-element table always returns 0 ------------------
TEST(ReservoirAliasTableTest, SingleElementAlwaysZero) {
	AliasTable at({7.5});
	PCG32Rng rng(0);
	for (int i = 0; i < 100; ++i)
		EXPECT_EQ(at.sample(rng.uniform()), 0);
}

// ---- 20. out_pmf matches at.pmf(returned_index) -----------------
TEST(ReservoirAliasTableTest, OutPmfCorrect) {
	AliasTable at({1.0, 2.0, 3.0});
	PCG32Rng rng(789);
	for (int i = 0; i < 200; ++i) {
		double pmf_val;
		int idx = at.sample(rng.uniform(), &pmf_val);
		EXPECT_NEAR(pmf_val, at.pmf(idx), 1e-12);
	}
}

// ---- 21. out_u_remapped in [0, 1) -------------------------------
TEST(ReservoirAliasTableTest, OutURemappedInUnitInterval) {
	AliasTable at({1.0, 2.0, 3.0, 1.0});
	PCG32Rng rng(321);
	for (int i = 0; i < 1000; ++i) {
		double u_remap;
		at.sample(rng.uniform(), nullptr, &u_remap);
		EXPECT_GE(u_remap, 0.0);
		EXPECT_LT(u_remap, 1.0);
	}
}

// ---- 22. Size-2 table with dominant weight ----------------------
TEST(ReservoirAliasTableTest, DominantWeightSelected) {
	// weight[0]=99, weight[1]=1 -> index 0 should be selected ~99% of the time
	AliasTable at({99.0, 1.0});
	PCG32Rng rng(555);
	const int TRIALS = 10000;
	int count0 = 0;
	for (int t = 0; t < TRIALS; ++t)
		if (at.sample(rng.uniform()) == 0) ++count0;
	EXPECT_NEAR(count0 / (double)TRIALS, 0.99, 0.02);
}
