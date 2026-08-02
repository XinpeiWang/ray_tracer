// sphere_cone_sampling_tests.cpp
// Unit tests for pbrt-v4-aligned sphere/cone/triangle sampling functions
// and MIS heuristics added to src/shared/sampling.h.
//
// Tests:
//   MIS heuristics:
//     1.  BalanceHeuristic: equal strategies -> 0.5
//     2.  BalanceHeuristic: dominant f -> approaches 1
//     3.  PowerHeuristic: equal strategies -> 0.5
//     4.  PowerHeuristic: dominant f -> > BalanceHeuristic result
//     5.  PowerHeuristic: zero g pdf -> 1
//
//   SampleUniformSphere:
//     6.  Output is unit vector
//     7.  pdf is 1/(4*pi)
//     8.  Roundtrip inverse recovers u0,u1
//     9.  wz uniform in [-1,1] (mean ~0)
//
//   SampleUniformHemisphere:
//    10.  Output is unit vector with wz >= 0
//    11.  pdf is 1/(2*pi)
//    12.  Roundtrip inverse recovers u0,u1
//
//   SampleUniformCone:
//    13.  Output is unit vector with wz >= cosThetaMax
//    14.  pdf matches UniformConePDF
//    15.  Roundtrip inverse recovers u0,u1
//    16.  Full sphere (cosThetaMax=-1) equivalent to SampleUniformSphere
//
//   SampleUniformTriangle:
//    17.  Barycentrics sum to 1, all >= 0
//    18.  Roundtrip inverse recovers u0,u1
//    19.  Covers triangle uniformly (mean of each barycentric ~1/3)

#include <gtest/gtest.h>
#include "../../src/shared/sampling.h"
#include <cmath>
#include <limits>

static const double kPi    = 3.14159265358979323846;
static const double kInv4Pi = 1.0 / (4.0 * kPi);
static const double kInv2Pi = 1.0 / (2.0 * kPi);
static const double kEps   = 1e-10;

// ---------------------------------------------------------------------------
// MIS heuristics
// ---------------------------------------------------------------------------

TEST(BalanceHeuristicTest, EqualStrategiesHalfWeight) {
	double w = BalanceHeuristic(1, 0.5, 1, 0.5);
	EXPECT_NEAR(w, 0.5, kEps);
}

TEST(BalanceHeuristicTest, DominantFApproachesOne) {
	double w = BalanceHeuristic(1, 1000.0, 1, 0.001);
	EXPECT_GT(w, 0.999);
}

TEST(PowerHeuristicTest, EqualStrategiesHalfWeight) {
	double w = PowerHeuristic(1, 0.5, 1, 0.5);
	EXPECT_NEAR(w, 0.5, kEps);
}

TEST(PowerHeuristicTest, DominantFGreaterThanBalance) {
	double balance = BalanceHeuristic(1, 10.0, 1, 1.0);
	double power   = PowerHeuristic(1, 10.0, 1, 1.0);
	EXPECT_GT(power, balance);
}

TEST(PowerHeuristicTest, ZeroGPdfReturnsOne) {
	double w = PowerHeuristic(1, 1.0, 1, 0.0);
	EXPECT_NEAR(w, 1.0, kEps);
}

// ---------------------------------------------------------------------------
// SampleUniformSphere
// ---------------------------------------------------------------------------

TEST(SampleUniformSphereTest, OutputIsUnitVector) {
	for (int i = 0; i < 100; ++i) {
		double u0 = (i + 0.5) / 100.0;
		double u1 = ((i * 37) % 100 + 0.5) / 100.0;
		double wx, wy, wz;
		SampleUniformSphere(u0, u1, wx, wy, wz);
		double len = std::sqrt(wx*wx + wy*wy + wz*wz);
		EXPECT_NEAR(len, 1.0, 1e-12);
	}
}

TEST(SampleUniformSphereTest, PDFIsInv4Pi) {
	EXPECT_NEAR(UniformSpherePDF(), kInv4Pi, kEps);
}

TEST(SampleUniformSphereTest, RoundtripInverse) {
	for (int i = 1; i < 10; ++i) {
		for (int j = 1; j < 10; ++j) {
			double u0 = i / 11.0, u1 = j / 11.0;
			double wx, wy, wz;
			SampleUniformSphere(u0, u1, wx, wy, wz);
			double ru0, ru1;
			InvertUniformSphereSample(wx, wy, wz, ru0, ru1);
			EXPECT_NEAR(ru0, u0, 1e-9);
			EXPECT_NEAR(ru1, u1, 1e-9);
		}
	}
}

TEST(SampleUniformSphereTest, MeanWzNearZero) {
	int N = 1000;
	double sum = 0.0;
	for (int i = 0; i < N; ++i) {
		double u0 = (i + 0.5) / N;
		double u1 = std::fmod((i * 0.618033988749895), 1.0);
		double wx, wy, wz;
		SampleUniformSphere(u0, u1, wx, wy, wz);
		sum += wz;
	}
	EXPECT_NEAR(sum / N, 0.0, 0.05);
}

// ---------------------------------------------------------------------------
// SampleUniformHemisphere
// ---------------------------------------------------------------------------

TEST(SampleUniformHemisphereTest, OutputIsUnitVectorWzNonNegative) {
	for (int i = 0; i < 100; ++i) {
		double u0 = (i + 0.5) / 100.0;
		double u1 = ((i * 37) % 100 + 0.5) / 100.0;
		double wx, wy, wz;
		SampleUniformHemisphere(u0, u1, wx, wy, wz);
		double len = std::sqrt(wx*wx + wy*wy + wz*wz);
		EXPECT_NEAR(len, 1.0, 1e-12);
		EXPECT_GE(wz, -kEps);
	}
}

TEST(SampleUniformHemisphereTest, PDFIsInv2Pi) {
	EXPECT_NEAR(UniformHemispherePDF(), kInv2Pi, kEps);
}

TEST(SampleUniformHemisphereTest, RoundtripInverse) {
	for (int i = 1; i < 10; ++i) {
		for (int j = 1; j < 10; ++j) {
			double u0 = i / 11.0, u1 = j / 11.0;
			double wx, wy, wz;
			SampleUniformHemisphere(u0, u1, wx, wy, wz);
			double ru0, ru1;
			InvertUniformHemisphereSample(wx, wy, wz, ru0, ru1);
			EXPECT_NEAR(ru0, u0, 1e-9);
			EXPECT_NEAR(ru1, u1, 1e-9);
		}
	}
}

// ---------------------------------------------------------------------------
// SampleUniformCone
// ---------------------------------------------------------------------------

TEST(SampleUniformConeTest, OutputIsUnitVectorWithinCone) {
	double cosThetaMax = 0.5;  // 60-degree cone
	for (int i = 0; i < 100; ++i) {
		double u0 = (i + 0.5) / 100.0;
		double u1 = ((i * 37) % 100 + 0.5) / 100.0;
		double wx, wy, wz;
		SampleUniformCone(u0, u1, cosThetaMax, wx, wy, wz);
		double len = std::sqrt(wx*wx + wy*wy + wz*wz);
		EXPECT_NEAR(len, 1.0, 1e-12);
		EXPECT_GE(wz, cosThetaMax - 1e-9);
	}
}

TEST(SampleUniformConeTest, PDFMatchesUniformConePDF) {
	double cosThetaMax = 0.7;
	double expected = 1.0 / (2.0 * kPi * (1.0 - cosThetaMax));
	EXPECT_NEAR(UniformConePDF(cosThetaMax), expected, kEps);
}

TEST(SampleUniformConeTest, RoundtripInverse) {
	double cosThetaMax = 0.6;
	for (int i = 1; i < 10; ++i) {
		for (int j = 1; j < 10; ++j) {
			double u0 = i / 11.0, u1 = j / 11.0;
			double wx, wy, wz;
			SampleUniformCone(u0, u1, cosThetaMax, wx, wy, wz);
			double ru0, ru1;
			InvertUniformConeSample(wx, wy, wz, cosThetaMax, ru0, ru1);
			EXPECT_NEAR(ru0, u0, 1e-9);
			EXPECT_NEAR(ru1, u1, 1e-9);
		}
	}
}

TEST(SampleUniformConeTest, FullSphereCoversBothHemispheres) {
	// cosThetaMax = -1 -> full sphere; wz should cover [-1,1]
	double cosThetaMax = -1.0;
	double minWz = 1.0, maxWz = -1.0;
	for (int i = 0; i < 100; ++i) {
		double u0 = (i + 0.5) / 100.0;
		double u1 = ((i * 37) % 100 + 0.5) / 100.0;
		double wx, wy, wz;
		SampleUniformCone(u0, u1, cosThetaMax, wx, wy, wz);
		if (wz < minWz) minWz = wz;
		if (wz > maxWz) maxWz = wz;
	}
	EXPECT_LT(minWz, -0.5);
	EXPECT_GT(maxWz,  0.5);
}

// ---------------------------------------------------------------------------
// SampleUniformTriangle
// ---------------------------------------------------------------------------

TEST(SampleUniformTriangleTest, BarycentricsSumToOneAndNonNegative) {
	for (int i = 0; i < 100; ++i) {
		double u0 = (i + 0.5) / 100.0;
		double u1 = ((i * 37) % 100 + 0.5) / 100.0;
		double b0, b1, b2;
		SampleUniformTriangle(u0, u1, b0, b1, b2);
		EXPECT_NEAR(b0 + b1 + b2, 1.0, 1e-12);
		EXPECT_GE(b0, -kEps);
		EXPECT_GE(b1, -kEps);
		EXPECT_GE(b2, -kEps);
	}
}

TEST(SampleUniformTriangleTest, RoundtripInverse) {
	for (int i = 1; i < 10; ++i) {
		for (int j = 1; j < 10; ++j) {
			double u0 = i / 11.0, u1 = j / 11.0;
			double b0, b1, b2;
			SampleUniformTriangle(u0, u1, b0, b1, b2);
			double ru0, ru1;
			InvertUniformTriangleSample(b0, b1, ru0, ru1);
			EXPECT_NEAR(ru0, u0, 1e-9);
			EXPECT_NEAR(ru1, u1, 1e-9);
		}
	}
}

TEST(SampleUniformTriangleTest, MeanBarycentricNearOneThird) {
	int N = 1000;
	double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0;
	for (int i = 0; i < N; ++i) {
		double u0 = (i + 0.5) / N;
		double u1 = std::fmod((i * 0.618033988749895), 1.0);
		double b0, b1, b2;
		SampleUniformTriangle(u0, u1, b0, b1, b2);
		sum0 += b0; sum1 += b1; sum2 += b2;
	}
	EXPECT_NEAR(sum0 / N, 1.0/3.0, 0.02);
	EXPECT_NEAR(sum1 / N, 1.0/3.0, 0.02);
	EXPECT_NEAR(sum2 / N, 1.0/3.0, 0.02);
}
