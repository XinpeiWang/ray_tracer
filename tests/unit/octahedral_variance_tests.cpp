// octahedral_variance_tests.cpp
// Unit tests for src/shared/octahedral_variance.h
//
// OctahedralVector tests:
//   1. Round-trip: canonical axes recover exactly (or near-exactly)
//   2. Round-trip: random unit vectors recover within angular tolerance
//   3. Antipodal: opposite vectors encode to different values
//   4. Lower-hemisphere directions round-trip correctly
//
// VarianceEstimator tests:
//   5. Empty estimator has zero count, zero variance
//   6. Single sample: mean == sample, variance == 0
//   7. Known sequence: mean and variance match analytical values
//   8. Merge: two halves merged equal full sequence
//   9. RelativeVariance: zero when mean is zero

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include "../../src/shared/octahedral_variance.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float dot3(float ax, float ay, float az, float bx, float by, float bz) {
	return ax * bx + ay * by + az * bz;
}

static float angleError(float ax, float ay, float az, float bx, float by, float bz) {
	float d = dot3(ax, ay, az, bx, by, bz);
	d = std::max(-1.f, std::min(1.f, d));
	return std::acos(d) * (180.f / 3.14159265358979323846f);  // degrees
}

static void normalize3(float& x, float& y, float& z) {
	float len = std::sqrt(x*x + y*y + z*z);
	x /= len; y /= len; z /= len;
}

// ---------------------------------------------------------------------------
// OctahedralVector tests
// ---------------------------------------------------------------------------

TEST(OctahedralVector, CanonicalAxes) {
	// +X, -X, +Y, -Y, +Z, -Z should round-trip with < 0.1 degree error
	const float dirs[6][3] = {
		{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}
	};
	for (auto& d : dirs) {
		OctahedralVector ov(d[0], d[1], d[2]);
		float rx, ry, rz;
		ov.ToVec3(rx, ry, rz);
		float err = angleError(d[0], d[1], d[2], rx, ry, rz);
		EXPECT_LT(err, 0.1f) << "dir=(" << d[0] << "," << d[1] << "," << d[2] << ")";
	}
}

TEST(OctahedralVector, UpperHemisphereRoundTrip) {
	// Sample upper hemisphere directions and verify < 0.1 degree error
	const int N = 8;
	for (int i = 0; i < N; ++i) {
		float theta = (i + 0.5f) / N * 3.14159265f * 0.5f;  // [0, pi/2)
		for (int j = 0; j < N; ++j) {
			float phi = j / float(N) * 2.f * 3.14159265f;
			float vx = std::sin(theta) * std::cos(phi);
			float vy = std::sin(theta) * std::sin(phi);
			float vz = std::cos(theta);  // z > 0

			OctahedralVector ov(vx, vy, vz);
			float rx, ry, rz;
			ov.ToVec3(rx, ry, rz);
			float err = angleError(vx, vy, vz, rx, ry, rz);
			EXPECT_LT(err, 0.15f) << "theta=" << theta << " phi=" << phi;
		}
	}
}

TEST(OctahedralVector, LowerHemisphereRoundTrip) {
	// Sample lower hemisphere directions (z < 0) and verify < 0.15 degree error
	const int N = 8;
	for (int i = 0; i < N; ++i) {
		float theta = 3.14159265f * 0.5f + (i + 0.5f) / N * 3.14159265f * 0.5f;
		for (int j = 0; j < N; ++j) {
			float phi = j / float(N) * 2.f * 3.14159265f;
			float vx = std::sin(theta) * std::cos(phi);
			float vy = std::sin(theta) * std::sin(phi);
			float vz = std::cos(theta);  // z < 0

			OctahedralVector ov(vx, vy, vz);
			float rx, ry, rz;
			ov.ToVec3(rx, ry, rz);
			float err = angleError(vx, vy, vz, rx, ry, rz);
			EXPECT_LT(err, 0.15f) << "theta=" << theta << " phi=" << phi;
		}
	}
}

TEST(OctahedralVector, AntipodalEncodesDifferently) {
	OctahedralVector a(0.5f, 0.5f, 0.707f);
	OctahedralVector b(-0.5f, -0.5f, -0.707f);
	EXPECT_TRUE(a.x != b.x || a.y != b.y);
}

TEST(OctahedralVector, Deterministic) {
	OctahedralVector a(0.3f, 0.4f, 0.866f);
	OctahedralVector b(0.3f, 0.4f, 0.866f);
	EXPECT_EQ(a.x, b.x);
	EXPECT_EQ(a.y, b.y);
}

TEST(OctahedralVector, StorageSize) {
	EXPECT_EQ(sizeof(OctahedralVector), 4u);
}

// ---------------------------------------------------------------------------
// VarianceEstimator tests
// ---------------------------------------------------------------------------

TEST(VarianceEstimator, EmptyState) {
	VarianceEstimator<double> ve;
	EXPECT_EQ(ve.Count(), 0);
	EXPECT_DOUBLE_EQ(ve.Mean(), 0.0);
	EXPECT_DOUBLE_EQ(ve.Variance(), 0.0);
	EXPECT_DOUBLE_EQ(ve.RelativeVariance(), 0.0);
}

TEST(VarianceEstimator, SingleSample) {
	VarianceEstimator<double> ve;
	ve.Add(42.0);
	EXPECT_EQ(ve.Count(), 1);
	EXPECT_DOUBLE_EQ(ve.Mean(), 42.0);
	EXPECT_DOUBLE_EQ(ve.Variance(), 0.0);
}

TEST(VarianceEstimator, KnownSequence) {
	// {2, 4, 4, 4, 5, 5, 7, 9} -- Wikipedia Welford example
	// mean = 5, variance = 4 (population), sample variance = 4.571...
	VarianceEstimator<double> ve;
	for (double v : {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0})
		ve.Add(v);
	EXPECT_EQ(ve.Count(), 8);
	EXPECT_NEAR(ve.Mean(), 5.0, 1e-10);
	// Sample variance = sum of sq deviations / (n-1) = 32/7
	EXPECT_NEAR(ve.Variance(), 32.0 / 7.0, 1e-9);
}

TEST(VarianceEstimator, MergeEqualsFullPass) {
	// Split the sequence into two halves and merge; result must match one-pass
	std::vector<double> seq;
	for (int i = 1; i <= 100; ++i) seq.push_back(static_cast<double>(i));

	VarianceEstimator<double> full;
	for (double v : seq) full.Add(v);

	VarianceEstimator<double> half1, half2;
	for (int i = 0; i < 50; ++i) half1.Add(seq[i]);
	for (int i = 50; i < 100; ++i) half2.Add(seq[i]);
	half1.Merge(half2);

	EXPECT_EQ(half1.Count(), full.Count());
	EXPECT_NEAR(half1.Mean(), full.Mean(), 1e-9);
	EXPECT_NEAR(half1.Variance(), full.Variance(), 1e-6);
}

TEST(VarianceEstimator, MergeWithEmpty) {
	VarianceEstimator<double> ve, empty;
	ve.Add(1.0); ve.Add(2.0);
	double meanBefore = ve.Mean();
	double varBefore  = ve.Variance();
	ve.Merge(empty);  // merge empty -- should be no-op
	EXPECT_DOUBLE_EQ(ve.Mean(), meanBefore);
	EXPECT_DOUBLE_EQ(ve.Variance(), varBefore);
}

TEST(VarianceEstimator, RelativeVariance) {
	VarianceEstimator<double> ve;
	for (double v : {1.0, 2.0, 3.0, 4.0, 5.0}) ve.Add(v);
	double rv = ve.RelativeVariance();
	EXPECT_GT(rv, 0.0);
	EXPECT_NEAR(rv, ve.Variance() / ve.Mean(), 1e-10);
}

TEST(VarianceEstimator, FloatSpecialization) {
	VarianceEstimator<float> ve;
	ve.Add(1.f); ve.Add(3.f);
	EXPECT_NEAR(ve.Mean(), 2.f, 1e-5f);
	EXPECT_NEAR(ve.Variance(), 2.f, 1e-4f);  // (1-2)^2 + (3-2)^2 / 1 = 2
}
