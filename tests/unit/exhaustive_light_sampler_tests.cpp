// exhaustive_light_sampler_tests.cpp
// ===========================================================================
// Regression tests for ExhaustiveLightSampler (src/shared/exhaustive_light_sampler.h)
//
// Test groups:
//   1. Construction / empty
//   2. Single bounded light -- PMF==1, Sample returns correct index
//   3. Single infinite light -- PMF==1, Sample returns correct index
//   4. Mixed infinite + bounded -- probability split
//   5. Two bounded lights at different distances -- nearer gets higher PMF
//   6. PMF sums to ~1 over all lights
//   7. Sample PMF matches PMF() query (consistency)
//   8. Outside-cone bounded light -- PMF is zero
// ===========================================================================

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>
#include "../../src/shared/exhaustive_light_sampler.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Make a unit-cube LightBounds at (ox,oy,oz) with +Z axis, given phi.
// phi == 0 => "infinite" light (no spatial bounds).
static LightBounds MakeLB(float ox, float oy, float oz,
						   float phi = 10.f, bool twoSided = false)
{
	return LightBounds(ox, oy, oz, ox+1.f, oy+1.f, oz+1.f,
					   0.f, 0.f, 1.f, phi, 0.f, 0.f, twoSided);
}

// "Infinite" sentinel: phi == 0
static LightBounds MakeInfiniteLB() {
	LightBounds lb;
	lb.phi = 0.f;
	return lb;
}

// Query point well above all lights, no surface normal (volume query)
static const float QX = 0.5f, QY = 0.5f, QZ = 20.f;
static const float NX = 0.f,  NY = 0.f,  NZ = 0.f;  // no surface normal

// ---------------------------------------------------------------------------
// 1. Construction / empty
// ---------------------------------------------------------------------------

TEST(ExhaustiveLightSampler, DefaultIsEmpty) {
	ExhaustiveLightSampler s;
	EXPECT_TRUE(s.empty());
	EXPECT_EQ(s.total_lights(), 0);
}

TEST(ExhaustiveLightSampler, ZeroPhiBecomeInfinite) {
	LightBounds lb = MakeInfiniteLB();
	ExhaustiveLightSampler s(&lb, 1);
	EXPECT_FALSE(s.empty());
	EXPECT_EQ(s.infinite_count(), 1);
	EXPECT_EQ(s.bounded_count(), 0);
}

TEST(ExhaustiveLightSampler, PositivePhiIsBounded) {
	LightBounds lb = MakeLB(0,0,0, 5.f);
	ExhaustiveLightSampler s(&lb, 1);
	EXPECT_FALSE(s.empty());
	EXPECT_EQ(s.bounded_count(), 1);
	EXPECT_EQ(s.infinite_count(), 0);
}

// ---------------------------------------------------------------------------
// 2. Single bounded light
// ---------------------------------------------------------------------------

TEST(ExhaustiveLightSampler, SingleBoundedPMFIsOne) {
	LightBounds lb = MakeLB(0,0,0, 8.f);
	ExhaustiveLightSampler s(&lb, 1);
	float pmf = s.PMF(QX, QY, QZ, NX, NY, NZ, 0);
	EXPECT_NEAR(pmf, 1.f, 1e-5f);
}

TEST(ExhaustiveLightSampler, SingleBoundedSampleReturnsIndex0) {
	LightBounds lb = MakeLB(0,0,0, 8.f);
	ExhaustiveLightSampler s(&lb, 1);
	auto result = s.Sample(QX, QY, QZ, NX, NY, NZ, 0.5f);
	EXPECT_EQ(result.lightIndex, 0);
	EXPECT_NEAR(result.pmf, 1.f, 1e-5f);
}

// ---------------------------------------------------------------------------
// 3. Single infinite light
// ---------------------------------------------------------------------------

TEST(ExhaustiveLightSampler, SingleInfinitePMFIsOne) {
	LightBounds lb = MakeInfiniteLB();
	ExhaustiveLightSampler s(&lb, 1);
	float pmf = s.PMF(QX, QY, QZ, NX, NY, NZ, 0);
	EXPECT_NEAR(pmf, 1.f, 1e-5f);
}

TEST(ExhaustiveLightSampler, SingleInfiniteSampleReturnsIndex0) {
	LightBounds lb = MakeInfiniteLB();
	ExhaustiveLightSampler s(&lb, 1);
	// u=0.5 should land in the infinite branch since p_inf=1
	auto result = s.Sample(QX, QY, QZ, NX, NY, NZ, 0.5f);
	EXPECT_EQ(result.lightIndex, 0);
	EXPECT_NEAR(result.pmf, 1.f, 1e-5f);
}

// ---------------------------------------------------------------------------
// 4. Mixed: one infinite + one bounded
// ---------------------------------------------------------------------------

TEST(ExhaustiveLightSampler, MixedProbabilitySplit) {
	// 1 infinite (phi=0) at index 0, 1 bounded (phi=10) at index 1
	LightBounds lbs[2];
	lbs[0] = MakeInfiniteLB();    // index 0 = infinite
	lbs[1] = MakeLB(0,0,0, 10.f); // index 1 = bounded
	ExhaustiveLightSampler s(lbs, 2);

	EXPECT_EQ(s.infinite_count(), 1);
	EXPECT_EQ(s.bounded_count(), 1);

	// p_inf = 1 / (1 + 1) = 0.5
	// PMF(infinite) = p_inf / 1 = 0.5
	float pmf_inf = s.PMF(QX, QY, QZ, NX, NY, NZ, 0);
	EXPECT_NEAR(pmf_inf, 0.5f, 1e-5f);

	// PMF(bounded) = importanceRatio * (1 - p_inf) = 1.0 * 0.5 = 0.5
	float pmf_bnd = s.PMF(QX, QY, QZ, NX, NY, NZ, 1);
	EXPECT_NEAR(pmf_bnd, 0.5f, 1e-4f);

	// Sum = 1
	EXPECT_NEAR(pmf_inf + pmf_bnd, 1.f, 1e-5f);
}

// ---------------------------------------------------------------------------
// 5. Two bounded lights at different distances -- nearer gets higher PMF
// ---------------------------------------------------------------------------

TEST(ExhaustiveLightSampler, NearerBoundedLightGetsHigherPMF) {
	// Light 0: near (z = 1), Light 1: far (z = 100)
	LightBounds lbs[2];
	lbs[0] = MakeLB(0, 0, 1,  10.f);  // near
	lbs[1] = MakeLB(0, 0, 100, 10.f); // far

	ExhaustiveLightSampler s(lbs, 2);

	// Query point at origin looking toward +Z
	float pmf0 = s.PMF(0.5f, 0.5f, 0.f, 0, 0, 0, 0);
	float pmf1 = s.PMF(0.5f, 0.5f, 0.f, 0, 0, 0, 1);

	// Near light should have strictly higher importance
	EXPECT_GT(pmf0, pmf1);
	// Both are valid probabilities
	EXPECT_GT(pmf0, 0.f);
	EXPECT_GT(pmf1, 0.f);
}

// ---------------------------------------------------------------------------
// 6. PMF sums to 1 over all lights (many bounded lights)
// ---------------------------------------------------------------------------

TEST(ExhaustiveLightSampler, PMFSumsToOne_AllBounded) {
	const int N = 8;
	std::vector<LightBounds> lbs(N);
	for (int i = 0; i < N; ++i)
		lbs[i] = MakeLB(float(i) * 3.f, 0, 0, 5.f + float(i));

	ExhaustiveLightSampler s(lbs);

	float sum = 0.f;
	for (int i = 0; i < N; ++i)
		sum += s.PMF(QX, QY, QZ, NX, NY, NZ, i);
	EXPECT_NEAR(sum, 1.f, 1e-4f);
}

TEST(ExhaustiveLightSampler, PMFSumsToOne_Mixed) {
	// 2 infinite + 4 bounded
	std::vector<LightBounds> lbs(6);
	lbs[0] = MakeInfiniteLB();
	lbs[1] = MakeInfiniteLB();
	for (int i = 2; i < 6; ++i)
		lbs[i] = MakeLB(float(i) * 2.f, 0, 0, 8.f);

	ExhaustiveLightSampler s(lbs);

	float sum = 0.f;
	for (int i = 0; i < 6; ++i)
		sum += s.PMF(QX, QY, QZ, NX, NY, NZ, i);
	EXPECT_NEAR(sum, 1.f, 1e-4f);
}

// ---------------------------------------------------------------------------
// 7. Sample PMF matches PMF() query (consistency check)
// ---------------------------------------------------------------------------

TEST(ExhaustiveLightSampler, SamplePmfMatchesPMFQuery) {
	const int N = 5;
	std::vector<LightBounds> lbs(N);
	for (int i = 0; i < N; ++i)
		lbs[i] = MakeLB(float(i) * 4.f, 0, 0, float(i+1) * 3.f);

	ExhaustiveLightSampler s(lbs);

	// Test several u values
	for (int k = 0; k < 20; ++k) {
		float u = (k + 0.5f) / 20.f;
		auto result = s.Sample(QX, QY, QZ, NX, NY, NZ, u);
		if (result.lightIndex < 0) continue;

		float pmf_query = s.PMF(QX, QY, QZ, NX, NY, NZ, result.lightIndex);
		// Sample PMF and PMF() should agree to within floating point tolerance
		EXPECT_NEAR(result.pmf, pmf_query, 1e-4f)
			<< "Mismatch at u=" << u << " lightIndex=" << result.lightIndex;
	}
}

// ---------------------------------------------------------------------------
// 8. Outside-cone bounded light -- PMF is zero
// ---------------------------------------------------------------------------

TEST(ExhaustiveLightSampler, OutsideConePMFIsZero) {
	// Light emitting in +X direction with a narrow cone:
	//   cosTheta_o = 0.866 (~30 deg inner), cosTheta_e = 0.866 (same = no falloff zone)
	// Query point at (0, 0, 10) is 90 degrees off the +X emission axis,
	// so cosThetap < cosTheta_e and Importance() returns 0.
	LightBounds lb(
		0, 0, 0,  1, 1, 1,   // bbox unit cube at origin
		1, 0, 0,              // emission axis +X
		5.f,                  // phi
		0.866f, 0.866f,       // narrow cone: cosTheta_o == cosTheta_e
		false
	);
	ExhaustiveLightSampler s(&lb, 1);

	// Query point is directly above (+Z): 90 deg off the +X cone axis
	float pmf = s.PMF(0.5f, 0.5f, 10.f,  0.f, 0.f, 0.f,  0);
	EXPECT_NEAR(pmf, 0.f, 1e-6f);
}
