// restir_tests.cpp
// Validation for src/shared/restir.h (ReSTIR / RIS primitives)
//
// Tests:
// RestirCandidate
//   1.  ris_weight() returns target_pdf / source_pdf
//   2.  ris_weight() returns 0 when source_pdf == 0
//
// Reservoir
//   3.  Default reservoir is invalid (index==-1)
//   4.  clear() resets all fields
//
// ris_fill
//   5.  Single candidate is always selected
//   6.  Selected candidate is always one of the inputs
//   7.  w_sum equals sum of all RIS weights
//   8.  M is set to the number of candidates
//   9.  All-zero weights: reservoir stays empty (no valid selection)
//  10.  Selection probability is proportional to p_hat/p (statistical test)
//
// reservoir_ucw
//  11.  UCW formula: W = w_sum / (M * p_hat)
//  12.  UCW is 0 when target_pdf == 0
//
// temporal_update
//  13.  Merging an invalid previous reservoir leaves current unchanged
//  14.  Merging a dominant previous reservoir replaces current with high prob
//  15.  M is capped at max_M after merge
//  16.  w_sum is sum of both reservoirs' w_sums after merge
//
// spatial_merge
//  17.  Merging zero neighbors leaves current unchanged
//  18.  Merging one neighbor with larger w_sum selects it more often
//  19.  M is capped at max_M after spatial merge
//
// restir_direct_sample
//  20.  Returns a valid reservoir for non-zero source
//  21.  UCW is computed (W > 0) when valid candidate exists
//  22.  Integral estimate converges for uniform p_hat over N lights

#include <gtest/gtest.h>
#include "../../src/shared/restir.h"
#include <cmath>
#include <array>
#include <numeric>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static RestirCandidate<double> make_cand(int idx, double p, double p_hat) {
	RestirCandidate<double> c;
	c.index      = idx;
	c.source_pdf = p;
	c.target_pdf = p_hat;
	return c;
}

// ============================================================
// RestirCandidate tests
// ============================================================

TEST(RestirCandidate, RisWeightFormula) {
	auto c = make_cand(0, 0.5, 2.0);
	EXPECT_NEAR(c.ris_weight(), 2.0 / 0.5, 1e-12);
}

TEST(RestirCandidate, RisWeightZeroSourcePdf) {
	auto c = make_cand(0, 0.0, 1.0);
	EXPECT_EQ(c.ris_weight(), 0.0);
}

// ============================================================
// Reservoir tests
// ============================================================

TEST(Reservoir, DefaultIsInvalid) {
	Reservoir<double> r;
	EXPECT_FALSE(r.valid());
	EXPECT_EQ(r.selected.index, -1);
	EXPECT_EQ(r.M, 0);
}

TEST(Reservoir, ClearResetsAllFields) {
	Reservoir<double> r;
	r.selected = make_cand(5, 0.1, 0.5);
	r.w_sum = 3.0;
	r.M     = 7;
	r.W     = 1.2;
	r.clear();
	EXPECT_EQ(r.selected.index, -1);
	EXPECT_EQ(r.w_sum, 0.0);
	EXPECT_EQ(r.M, 0);
	EXPECT_EQ(r.W, 0.0);
}

// ============================================================
// ris_fill tests
// ============================================================

TEST(RisFill, SingleCandidateAlwaysSelected) {
	std::array<RestirCandidate<double>, 1> cands = { make_cand(7, 0.25, 1.0) };
	for (uint64_t seed = 0; seed < 10; ++seed) {
		Reservoir<double> r;
		ris_fill(r, cands.data(), 1, seed);
		EXPECT_EQ(r.selected.index, 7);
		EXPECT_EQ(r.M, 1);
	}
}

TEST(RisFill, SelectedIsAlwaysOneOfInputs) {
	const int N = 5;
	std::array<RestirCandidate<double>, N> cands;
	for (int i = 0; i < N; ++i) cands[i] = make_cand(i, 0.2, double(i + 1));
	for (uint64_t t = 0; t < 500; ++t) {
		Reservoir<double> r;
		ris_fill(r, cands.data(), N, t * 6364136223846793005ULL + 1);
		if (r.valid())
			EXPECT_GE(r.selected.index, 0) << "selected index out of range";
		if (r.valid())
			EXPECT_LT(r.selected.index, N) << "selected index out of range";
	}
}

TEST(RisFill, WeightSumEqualsSum) {
	std::array<RestirCandidate<double>, 4> cands = {
		make_cand(0, 0.25, 1.0),
		make_cand(1, 0.25, 2.0),
		make_cand(2, 0.25, 3.0),
		make_cand(3, 0.25, 4.0),
	};
	// sum of RIS weights = 1/0.25 + 2/0.25 + 3/0.25 + 4/0.25 = 4+8+12+16 = 40
	double expected_sum = 0.0;
	for (auto& c : cands) expected_sum += c.ris_weight();

	Reservoir<double> r;
	ris_fill(r, cands.data(), 4, /*seed=*/42ULL);
	EXPECT_NEAR(r.w_sum, expected_sum, 1e-10);
	EXPECT_EQ(r.M, 4);
}

TEST(RisFill, MIsSetToN) {
	const int N = 8;
	std::array<RestirCandidate<double>, N> cands;
	for (int i = 0; i < N; ++i) cands[i] = make_cand(i, 1.0 / N, 1.0);
	Reservoir<double> r;
	ris_fill(r, cands.data(), N, /*seed=*/1ULL);
	EXPECT_EQ(r.M, N);
}

TEST(RisFill, AllZeroWeightsEmptyReservoir) {
	std::array<RestirCandidate<double>, 3> cands = {
		make_cand(0, 0.0, 0.0),
		make_cand(1, 0.0, 0.0),
		make_cand(2, 0.0, 0.0),
	};
	Reservoir<double> r;
	ris_fill(r, cands.data(), 3, /*seed=*/5ULL);
	EXPECT_NEAR(r.w_sum, 0.0, 1e-15);
	// No acceptance can occur since w_i=0 for all candidates.
	EXPECT_FALSE(r.valid());
}

TEST(RisFill, SelectionProportionalToTargetPdf) {
	// Two candidates: p_hat 1 and 3, equal source. Expected selection ratio 1:3.
	std::array<RestirCandidate<double>, 2> cands = {
		make_cand(0, 0.5, 1.0),
		make_cand(1, 0.5, 3.0),
	};
	const int TRIALS = 100000;
	int count1 = 0;
	for (uint64_t t = 0; t < TRIALS; ++t) {
		Reservoir<double> r;
		ris_fill(r, cands.data(), 2, t * 6364136223846793005ULL + 99);
		if (r.valid() && r.selected.index == 1) ++count1;
	}
	double freq1 = count1 / (double)TRIALS;
	// Expected: 3/(1+3) = 0.75
	EXPECT_NEAR(freq1, 0.75, 0.02) << "RIS selection not proportional to p_hat";
}

// ============================================================
// reservoir_ucw tests
// ============================================================

TEST(ReservoirUCW, Formula) {
	// w_sum=10, M=4, p_hat=2.0  =>  W = 10/(4*2) = 1.25
	Reservoir<double> r;
	r.selected = make_cand(0, 0.5, 2.0);
	r.w_sum    = 10.0;
	r.M        = 4;
	reservoir_ucw(r);
	EXPECT_NEAR(r.W, 1.25, 1e-12);
}

TEST(ReservoirUCW, ZeroTargetPdfGivesZeroW) {
	Reservoir<double> r;
	r.selected = make_cand(0, 0.5, 0.0);
	r.w_sum    = 5.0;
	r.M        = 2;
	reservoir_ucw(r);
	EXPECT_EQ(r.W, 0.0);
}

// ============================================================
// temporal_update tests
// ============================================================

TEST(TemporalUpdate, InvalidPrevLeavesCurrentUnchanged) {
	// Build a valid current reservoir
	std::array<RestirCandidate<double>, 1> cands = { make_cand(3, 0.5, 1.0) };
	Reservoir<double> current;
	ris_fill(current, cands.data(), 1, /*seed=*/0ULL);
	double orig_wsum = current.w_sum;
	int    orig_M    = current.M;
	int    orig_idx  = current.selected.index;

	Reservoir<double> prev;  // default = invalid
	temporal_update(current, prev, /*seed=*/1ULL);

	EXPECT_EQ(current.selected.index, orig_idx);
	EXPECT_NEAR(current.w_sum, orig_wsum, 1e-12);
	EXPECT_EQ(current.M, orig_M);
}

TEST(TemporalUpdate, MIsCapped) {
	Reservoir<double> current;
	current.selected = make_cand(0, 0.5, 1.0);
	current.w_sum = 5.0;
	current.M = 15;

	Reservoir<double> prev;
	prev.selected = make_cand(1, 0.5, 1.0);
	prev.w_sum = 5.0;
	prev.M = 10;

	temporal_update(current, prev, /*seed=*/0ULL, /*max_M=*/20);
	EXPECT_LE(current.M, 20);
}

TEST(TemporalUpdate, WeightSumIsSum) {
	Reservoir<double> current;
	current.selected = make_cand(0, 0.5, 1.0);
	current.w_sum = 4.0;
	current.M = 2;

	Reservoir<double> prev;
	prev.selected = make_cand(1, 0.5, 1.0);
	prev.w_sum = 6.0;
	prev.M = 3;

	temporal_update(current, prev, /*seed=*/0ULL);
	EXPECT_NEAR(current.w_sum, 10.0, 1e-12);
}

TEST(TemporalUpdate, DominantPrevSelectedMostOften) {
	// prev has much larger w_sum -> should be selected most of the time
	const int TRIALS = 10000;
	int prev_count = 0;
	for (int t = 0; t < TRIALS; ++t) {
		Reservoir<double> current;
		current.selected = make_cand(0, 0.5, 1.0);
		current.w_sum = 1.0;
		current.M = 1;

		Reservoir<double> prev;
		prev.selected = make_cand(1, 0.5, 1.0);
		prev.w_sum = 9.0;
		prev.M = 1;

		temporal_update(current, prev, uint64_t(t) * 6364136223846793005ULL + 1);
		if (current.selected.index == 1) ++prev_count;
	}
	double freq = prev_count / (double)TRIALS;
	// Expected: 9/(1+9) = 0.9
	EXPECT_NEAR(freq, 0.9, 0.03);
}

// ============================================================
// spatial_merge tests
// ============================================================

TEST(SpatialMerge, ZeroNeighborsLeavesCurrentUnchanged) {
	Reservoir<double> current;
	current.selected = make_cand(0, 0.5, 2.0);
	current.w_sum = 3.0;
	current.M = 2;

	spatial_merge(current, static_cast<const Reservoir<double>*>(nullptr), 0, /*seed=*/0ULL);

	EXPECT_EQ(current.selected.index, 0);
	EXPECT_NEAR(current.w_sum, 3.0, 1e-12);
	EXPECT_EQ(current.M, 2);
}

TEST(SpatialMerge, MergeOneNeighborAccumulatesWsum) {
	Reservoir<double> current;
	current.selected = make_cand(0, 0.5, 1.0);
	current.w_sum = 2.0;
	current.M = 2;

	Reservoir<double> nb;
	nb.selected = make_cand(1, 0.5, 1.0);
	nb.w_sum = 3.0;
	nb.M = 3;

	spatial_merge(current, &nb, 1, /*seed=*/7ULL);

	EXPECT_NEAR(current.w_sum, 5.0, 1e-12);
	EXPECT_EQ(current.M, 5);
}

TEST(SpatialMerge, MIsCappedAtMaxM) {
	Reservoir<double> current;
	current.selected = make_cand(0, 0.5, 1.0);
	current.w_sum = 1.0;
	current.M = 100;

	std::array<Reservoir<double>, 3> nbs;
	for (int i = 0; i < 3; ++i) {
		nbs[i].selected = make_cand(i + 1, 0.5, 1.0);
		nbs[i].w_sum = 1.0;
		nbs[i].M = 100;
	}

	spatial_merge(current, nbs.data(), 3, /*seed=*/0ULL, /*max_M=*/250);
	EXPECT_LE(current.M, 250);
}

TEST(SpatialMerge, LargerNeighborSelectedMoreOften) {
	// neighbor has 4x larger w_sum -> selected ~80% of the time
	const int TRIALS = 20000;
	int nb_count = 0;
	for (int t = 0; t < TRIALS; ++t) {
		Reservoir<double> current;
		current.selected = make_cand(0, 0.5, 1.0);
		current.w_sum = 1.0;
		current.M = 1;

		Reservoir<double> nb;
		nb.selected = make_cand(1, 0.5, 1.0);
		nb.w_sum = 4.0;
		nb.M = 1;

		spatial_merge(current, &nb, 1, uint64_t(t) * 2862933555777941757ULL + 3);
		if (current.selected.index == 1) ++nb_count;
	}
	double freq = nb_count / (double)TRIALS;
	// Expected: 4/(1+4) = 0.8
	EXPECT_NEAR(freq, 0.8, 0.03);
}

// ============================================================
// restir_direct_sample integration tests
// ============================================================

TEST(RestirDirectSample, ValidReservoirForNonZeroSource) {
	// 4 lights with uniform source and varying p_hat
	int light_count = 4;
	double phat[4] = {0.1, 0.5, 1.5, 0.8};
	auto source_fn = [&](PCG32Rng& r) -> RestirCandidate<double> {
		int i = int(r.uniform() * light_count);
		i = std::min(i, light_count - 1);
		return make_cand(i, 1.0 / light_count, phat[i]);
	};
	auto res = restir_direct_sample<double>(8, source_fn, /*seed=*/42ULL);
	EXPECT_TRUE(res.valid());
	EXPECT_GT(res.W, 0.0);
}

TEST(RestirDirectSample, UCWComputedAfterFill) {
	auto source_fn = [](PCG32Rng& r) -> RestirCandidate<double> {
		return make_cand(0, 1.0, 2.0);
	};
	auto res = restir_direct_sample<double>(1, source_fn, /*seed=*/7ULL);
	// w_sum = p_hat/p = 2/1 = 2, M = 1, p_hat = 2 => W = 2/(1*2) = 1
	EXPECT_NEAR(res.W, 1.0, 1e-12);
}

TEST(RestirDirectSample, IntegralConvergesForConstantF) {
	// Setup: two lights, light 0 has radiance 0, light 1 has radiance 1.
	// p_hat = radiance, source = uniform (p = 0.5 each).
	// True integral (one-sample estimator) = f(y) / p_hat(y) * W = 1.
	// Average over many reservoirs should converge to 1.
	const int TRIALS = 20000;
	double sum = 0.0;

	for (uint64_t t = 0; t < TRIALS; ++t) {
		double radiance[2] = {0.0, 1.0};
		auto source_fn = [&](PCG32Rng& r) -> RestirCandidate<double> {
			int i = (r.uniform() < 0.5) ? 0 : 1;
			return make_cand(i, 0.5, radiance[i]);
		};
		auto res = restir_direct_sample<double>(4, source_fn,
			t * 6364136223846793005ULL + 123);
		if (res.valid()) {
			double f = radiance[res.selected.index];
			double p_hat = res.selected.target_pdf;
			double estimate = (p_hat > 0.0) ? (f / p_hat) * res.W : 0.0;
			sum += estimate;
		}
	}
	double avg = sum / TRIALS;
	// RIS with p_hat = f is an unbiased estimator for sum_i f(light_i)
	// = f(light_0) + f(light_1) = 0 + 1 = 1.0.
	// (The source probability p=0.5 is absorbed into w_sum/M; the result is
	//  not the expectation under p, but the unnormalized sum over the domain.)
	EXPECT_NEAR(avg, 1.0, 0.02) << "RIS integral estimate did not converge";
}
