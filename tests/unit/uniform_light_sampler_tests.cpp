// tests/unit/uniform_light_sampler_tests.cpp
// Unit tests for UniformLightSampler (mirrors pbrt-v4 UniformLightSampler)

#include <gtest/gtest.h>
#include "../../src/shared/uniform_light_sampler.h"

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(UniformLightSampler, DefaultConstructsEmpty) {
	UniformLightSampler s;
	EXPECT_EQ(s.size(), 0);
}

TEST(UniformLightSampler, ConstructWithN) {
	UniformLightSampler s(5);
	EXPECT_EQ(s.size(), 5);
}

TEST(UniformLightSampler, ConstructWithZeroIsEmpty) {
	UniformLightSampler s(0);
	EXPECT_EQ(s.size(), 0);
}

// ---------------------------------------------------------------------------
// sample()
// ---------------------------------------------------------------------------

TEST(UniformLightSampler, EmptySamplerReturnsMinusOne) {
	UniformLightSampler s;
	EXPECT_EQ(s.sample(0.0), -1);
	EXPECT_EQ(s.sample(0.5), -1);
	EXPECT_EQ(s.sample(0.999), -1);
}

TEST(UniformLightSampler, SingleLightAlwaysReturnsZero) {
	UniformLightSampler s(1);
	EXPECT_EQ(s.sample(0.0),   0);
	EXPECT_EQ(s.sample(0.5),   0);
	EXPECT_EQ(s.sample(0.999), 0);
}

TEST(UniformLightSampler, SampleInRange) {
	UniformLightSampler s(4);
	for (double u = 0.0; u < 1.0; u += 0.05) {
		int idx = s.sample(u);
		EXPECT_GE(idx, 0);
		EXPECT_LT(idx, 4);
	}
}

TEST(UniformLightSampler, SampleNearOneStaysInRange) {
	// u just below 1.0 must not return >= n
	UniformLightSampler s(3);
	EXPECT_LT(s.sample(0.9999999999), 3);
}

TEST(UniformLightSampler, SampleCoversAllIndices) {
	// Each of the n bins should be reachable
	const int n = 4;
	UniformLightSampler s(n);
	bool seen[4] = {};
	for (int i = 0; i < n; ++i) {
		double u = (i + 0.5) / n;
		seen[s.sample(u)] = true;
	}
	for (int i = 0; i < n; ++i)
		EXPECT_TRUE(seen[i]) << "index " << i << " never sampled";
}

// ---------------------------------------------------------------------------
// pmf()
// ---------------------------------------------------------------------------

TEST(UniformLightSampler, PMFIsOneOverN) {
	const int n = 7;
	UniformLightSampler s(n);
	double expected = 1.0 / n;
	for (int i = 0; i < n; ++i)
		EXPECT_NEAR(s.pmf(i), expected, 1e-12);
}

TEST(UniformLightSampler, PMFSumsToOne) {
	const int n = 5;
	UniformLightSampler s(n);
	double total = 0.0;
	for (int i = 0; i < n; ++i) total += s.pmf(i);
	EXPECT_NEAR(total, 1.0, 1e-12);
}

TEST(UniformLightSampler, PMFInvalidIndexReturnsZero) {
	UniformLightSampler s(3);
	EXPECT_EQ(s.pmf(-1), 0.0);
	EXPECT_EQ(s.pmf(3),  0.0);
	EXPECT_EQ(s.pmf(99), 0.0);
}

TEST(UniformLightSampler, PMFEmptyReturnsZero) {
	UniformLightSampler s;
	EXPECT_EQ(s.pmf(0), 0.0);
}

// ---------------------------------------------------------------------------
// Statistical: sampled distribution is uniform
// ---------------------------------------------------------------------------

TEST(UniformLightSampler, MonteCarloBinsAreUniform) {
	const int n = 5;
	UniformLightSampler s(n);
	int counts[5] = {};
	const int N = 10000;
	// Stratified samples for determinism
	for (int i = 0; i < N; ++i) {
		double u = (i + 0.5) / N;
		int idx = s.sample(u);
		ASSERT_GE(idx, 0);
		ASSERT_LT(idx, n);
		counts[idx]++;
	}
	// Each bin should have exactly N/n samples (stratified grid)
	int expected = N / n;
	for (int i = 0; i < n; ++i)
		EXPECT_EQ(counts[i], expected) << "bin " << i;
}
