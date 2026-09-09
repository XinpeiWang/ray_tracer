// wavefront_restir_tests.cpp
// Validation for gpu/optix/wavefront_restir_math.h's pure-math ReSTIR DI
// primitives - the GPU-native counterpart to restir_tests.cpp (src/shared/
// restir.h). Covers exactly the CPU-testable pieces the ReSTIR DI plan called
// out: the RIS accept/replace formula, reservoir_ucw, the reservoir-combine
// step, and the spatial-reuse Jacobian guard - all CPU_GPU/plain-float
// functions, so no GPU hardware is required to exercise them (mirrored here
// via a host build, same as restir_tests.cpp's own CPU-only Reservoir<T>).
//
// Tests:
// restir_reservoir_add
//   1.  A single candidate is always selected
//   2.  w_sum accumulates the RIS weight across multiple adds
//   3.  M accumulates candidateM across multiple adds
//   4.  Selection probability is proportional to weight (statistical test)
//   5.  A non-positive risWeight is absorbed into weightSum/M but never selected
//
// restir_reservoir_ucw / restir_finalize
//   6.  UCW formula: W = w_sum / (M * p_hat)
//   7.  UCW is 0 when p_hat == 0
//   8.  UCW is 0 when M == 0
//
// restir_reservoir_combine
//   9.  Combining an invalid reservoir leaves dst unchanged
//   10. Combining folds in the other reservoir's full M
//   11. A dominant other reservoir is selected with high probability
//
// wf_restir_jacobian
//   12. Returns 1 when both views see identical cosine/distance (no-op ratio)
//   13. Returns 0 when the neighbor's cosine is ~0 (grazing, degenerate)
//   14. Scales with the ratio of squared distances

#include <gtest/gtest.h>
#include "wavefront_restir_math.h"
#include <cmath>
#include <random>

namespace {

GpuLightSample make_sample(int lightIdx) {
	GpuLightSample s;
	s.lightIdx = lightIdx;
	s.kind = GpuLightKind::Quad;
	s.primIdx = 0;
	s.point = make_float3(0.0f, 0.0f, 0.0f);
	s.normal = make_float3(0.0f, 1.0f, 0.0f);
	return s;
}

} // namespace

// ============================================================
// restir_reservoir_add
// ============================================================

TEST(RestirReservoirAdd, SingleCandidateAlwaysSelected) {
	GpuReservoir r;
	GpuLightSample cand = make_sample(3);
	bool accepted = restir_reservoir_add(r, cand, /*risWeight=*/2.0f, /*candidateM=*/1, /*pHat=*/1.0f, /*rand01=*/0.5f);
	EXPECT_TRUE(accepted);
	EXPECT_TRUE(r.valid());
	EXPECT_EQ(r.sample.lightIdx, 3);
	EXPECT_FLOAT_EQ(r.weightSum, 2.0f);
	EXPECT_EQ(r.M, 1);
}

TEST(RestirReservoirAdd, WeightSumAccumulatesAcrossAdds) {
	GpuReservoir r;
	restir_reservoir_add(r, make_sample(0), 1.0f, 1, 1.0f, 0.99f);
	restir_reservoir_add(r, make_sample(1), 3.0f, 1, 1.0f, 0.99f);
	restir_reservoir_add(r, make_sample(2), 2.0f, 1, 1.0f, 0.99f);
	EXPECT_FLOAT_EQ(r.weightSum, 6.0f);
}

TEST(RestirReservoirAdd, MAccumulatesCandidateMAcrossAdds) {
	GpuReservoir r;
	restir_reservoir_add(r, make_sample(0), 1.0f, 1, 1.0f, 0.99f);
	restir_reservoir_add(r, make_sample(1), 1.0f, 5, 1.0f, 0.99f);
	restir_reservoir_add(r, make_sample(2), 1.0f, 3, 1.0f, 0.99f);
	EXPECT_EQ(r.M, 1 + 5 + 3);
}

TEST(RestirReservoirAdd, SelectionProbabilityProportionalToWeight) {
	// Candidate 0 has 9x the weight of candidate 1 - over many trials it
	// should end up selected roughly 9/10 of the time (same statistical-test
	// shape as restir_tests.cpp's RisFill.SelectionProbabilityMatchesWeights).
	std::mt19937 rng(12345);
	std::uniform_real_distribution<float> uni(0.0f, 1.0f);
	int selectedZero = 0;
	const int trials = 20000;
	for (int t = 0; t < trials; ++t) {
		GpuReservoir r;
		restir_reservoir_add(r, make_sample(0), 9.0f, 1, 1.0f, uni(rng));
		restir_reservoir_add(r, make_sample(1), 1.0f, 1, 1.0f, uni(rng));
		if (r.sample.lightIdx == 0) ++selectedZero;
	}
	const double frac = double(selectedZero) / trials;
	EXPECT_NEAR(frac, 0.9, 0.02);
}

TEST(RestirReservoirAdd, NonPositiveWeightNeverSelectedButStillCounted) {
	GpuReservoir r;
	restir_reservoir_add(r, make_sample(0), 5.0f, 1, 1.0f, 0.01f);  // becomes selected
	bool accepted = restir_reservoir_add(r, make_sample(1), 0.0f, 1, 0.0f, 0.0f);
	EXPECT_FALSE(accepted);
	EXPECT_EQ(r.sample.lightIdx, 0);   // unchanged
	EXPECT_EQ(r.M, 2);                 // still counted toward M
	EXPECT_FLOAT_EQ(r.weightSum, 5.0f); // unchanged (0 weight contributes nothing)
}

// ============================================================
// restir_reservoir_ucw / restir_finalize
// ============================================================

TEST(RestirReservoirUcw, MatchesWSumOverMTimesPHat) {
	EXPECT_NEAR(restir_reservoir_ucw(/*weightSum=*/10.0f, /*M=*/5, /*pHat=*/2.0f), 10.0f / (5.0f * 2.0f), 1e-6f);
}

TEST(RestirReservoirUcw, ZeroWhenPHatIsZero) {
	EXPECT_FLOAT_EQ(restir_reservoir_ucw(10.0f, 5, 0.0f), 0.0f);
}

TEST(RestirReservoirUcw, ZeroWhenMIsZero) {
	EXPECT_FLOAT_EQ(restir_reservoir_ucw(10.0f, 0, 2.0f), 0.0f);
}

TEST(RestirFinalize, SetsWFromWeightSumMAndPHat) {
	GpuReservoir r;
	restir_reservoir_add(r, make_sample(0), 4.0f, 1, 2.0f, 0.0f);
	restir_finalize(r);
	EXPECT_NEAR(r.W, 4.0f / (1.0f * 2.0f), 1e-6f);
}

// ============================================================
// restir_reservoir_combine
// ============================================================

TEST(RestirReservoirCombine, InvalidOtherLeavesDstUnchanged) {
	GpuReservoir dst;
	restir_reservoir_add(dst, make_sample(0), 1.0f, 1, 1.0f, 0.0f);
	restir_finalize(dst);
	GpuReservoir before = dst;

	GpuReservoir invalidOther;  // default-constructed: weightSum=0, invalid
	bool accepted = restir_reservoir_combine(dst, invalidOther, /*otherPHatAtDstContext=*/5.0f, 0.5f);

	EXPECT_FALSE(accepted);
	EXPECT_EQ(dst.sample.lightIdx, before.sample.lightIdx);
	EXPECT_FLOAT_EQ(dst.weightSum, before.weightSum);
	EXPECT_EQ(dst.M, before.M);
}

TEST(RestirReservoirCombine, FoldsInOthersFullM) {
	GpuReservoir dst;
	restir_reservoir_add(dst, make_sample(0), 1.0f, 1, 1.0f, 0.0f);

	GpuReservoir other;
	restir_reservoir_add(other, make_sample(1), 3.0f, 7, 3.0f, 0.0f);  // M=7
	restir_finalize(other);  // W = w_sum/(M*pHat) = 3/(7*3) = 1/7

	const int dstMBefore = dst.M;
	restir_reservoir_combine(dst, other, /*otherPHatAtDstContext=*/2.0f, 0.99f);
	EXPECT_EQ(dst.M, dstMBefore + other.M);
}

TEST(RestirReservoirCombine, DominantOtherSelectedWithHighProbability) {
	// `other` carries a much larger effective weight (pHatAtDstContext *
	// other.W * other.M) than dst's own single candidate - over many trials
	// it should be selected the large majority of the time.
	std::mt19937 rng(777);
	std::uniform_real_distribution<float> uni(0.0f, 1.0f);
	int selectedOther = 0;
	const int trials = 20000;
	for (int t = 0; t < trials; ++t) {
		GpuReservoir dst;
		restir_reservoir_add(dst, make_sample(0), 1.0f, 1, 1.0f, uni(rng));  // weak candidate

		GpuReservoir other;
		restir_reservoir_add(other, make_sample(1), 100.0f, 1, 100.0f, uni(rng));
		restir_finalize(other);  // W = 100/(1*100) = 1

		restir_reservoir_combine(dst, other, /*otherPHatAtDstContext=*/100.0f, uni(rng));
		if (dst.sample.lightIdx == 1) ++selectedOther;
	}
	EXPECT_GT(double(selectedOther) / trials, 0.95);
}

// ============================================================
// wf_restir_jacobian
// ============================================================

TEST(WfRestirJacobian, OneWhenViewsAreIdentical) {
	float3 normal = make_float3(0.0f, 1.0f, 0.0f);
	float3 dir = normalize(make_float3(0.3f, 1.0f, 0.2f));
	float j = wf_restir_jacobian(dir, 5.0f, dir, 5.0f, normal);
	EXPECT_NEAR(j, 1.0f, 1e-5f);
}

TEST(WfRestirJacobian, ZeroWhenNeighborViewIsGrazing) {
	float3 normal = make_float3(0.0f, 1.0f, 0.0f);
	float3 currentDir = normalize(make_float3(0.0f, 1.0f, 0.0f));
	float3 grazingDir = make_float3(1.0f, 0.0f, 0.0f);  // exactly perpendicular to normal - dot == 0
	float j = wf_restir_jacobian(currentDir, 5.0f, grazingDir, 5.0f, normal);
	EXPECT_FLOAT_EQ(j, 0.0f);
}

TEST(WfRestirJacobian, ScalesWithSquaredDistanceRatio) {
	float3 normal = make_float3(0.0f, 1.0f, 0.0f);
	float3 dir = normalize(make_float3(0.0f, 1.0f, 0.0f));
	// Same cosine on both sides (dir/normal unchanged) - ratio should reduce
	// to neighborDist^2 / currentDist^2.
	float j = wf_restir_jacobian(dir, 2.0f, dir, 4.0f, normal);
	EXPECT_NEAR(j, (4.0f * 4.0f) / (2.0f * 2.0f), 1e-4f);
}
