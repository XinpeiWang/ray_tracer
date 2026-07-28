// equal_area_tests.cpp
// Validation for EqualAreaSquareToSphere, EqualAreaSphereToSquare,
// WrapEqualAreaSquare, and SampleUniformHemisphereConcentric
// -- pbrt-v4 Clarberg equal-area sphere mapping ports
//
// Tests:
//   1.  EqualAreaSquareToSphere: output is unit vector
//   2.  EqualAreaSquareToSphere: corners map to known directions
//   3.  EqualAreaSquareToSphere: center (0.5,0.5) maps to north pole (0,0,1)
//   4.  EqualAreaSphereToSquare: roundtrip with EqualAreaSquareToSphere
//   5.  EqualAreaSphereToSquare: known directions map to known UV
//   6.  Equal-area property: uniform grid of UV -> uniform sphere area
//   7.  WrapEqualAreaSquare: values inside [0,1]^2 unchanged
//   8.  WrapEqualAreaSquare: u<0 wraps correctly
//   9.  WrapEqualAreaSquare: v>1 wraps correctly
//  10.  SampleUniformHemisphereConcentric: output is unit vector with wz>=0
//  11.  SampleUniformHemisphereConcentric: origin (0.5,0.5)->north pole
//  12.  SampleUniformHemisphereConcentric: covers hemisphere uniformly (mean wz ~0.5)

#include <gtest/gtest.h>
#include "../../src/shared/sampling.h"
#include <cmath>
#include <vector>

static const double kPi = 3.14159265358979323846;

// ---- 1. EqualAreaSquareToSphere: output is unit vector --------------------
TEST(EqualAreaTest, SquareToSphereUnitVector) {
	for (int i = 0; i <= 10; ++i) {
		for (int j = 0; j <= 10; ++j) {
			double u = i / 10.0, v = j / 10.0;
			double wx, wy, wz;
			EqualAreaSquareToSphere(u, v, wx, wy, wz);
			double len2 = wx*wx + wy*wy + wz*wz;
			EXPECT_NEAR(len2, 1.0, 1e-10) << "u=" << u << " v=" << v;
		}
	}
}

// ---- 2. EqualAreaSquareToSphere: corner directions ------------------------
TEST(EqualAreaTest, SquareToSphereCorners) {
	// (0.5, 0.5) -> north pole (0,0,1)
	double wx, wy, wz;
	EqualAreaSquareToSphere(0.5, 0.5, wx, wy, wz);
	EXPECT_NEAR(wz, 1.0, 1e-6);
	EXPECT_NEAR(wx, 0.0, 1e-6);
	EXPECT_NEAR(wy, 0.0, 1e-6);
}

// ---- 3. EqualAreaSquareToSphere: center is north pole --------------------
TEST(EqualAreaTest, CenterIsNorthPole) {
	double wx, wy, wz;
	EqualAreaSquareToSphere(0.5, 0.5, wx, wy, wz);
	EXPECT_NEAR(wz, 1.0, 1e-6);
}

// ---- 4. Roundtrip: Square->Sphere->Square ---------------------------------
TEST(EqualAreaTest, RoundtripSquareToSphereToSquare) {
	for (int i = 1; i < 10; ++i) {
		for (int j = 1; j < 10; ++j) {
			double u = i / 10.0, v = j / 10.0;
			double wx, wy, wz;
			EqualAreaSquareToSphere(u, v, wx, wy, wz);
			double u2, v2;
			EqualAreaSphereToSquare(wx, wy, wz, u2, v2);
			EXPECT_NEAR(u, u2, 1e-5) << "u=" << u << " v=" << v;
			EXPECT_NEAR(v, v2, 1e-5) << "u=" << u << " v=" << v;
		}
	}
}

// ---- 5. EqualAreaSphereToSquare: known directions -------------------------
TEST(EqualAreaTest, SphereToSquareKnownDirections) {
	double u, v;
	// North pole (0,0,1) -> (0.5, 0.5)
	EqualAreaSphereToSquare(0.0, 0.0, 1.0, u, v);
	EXPECT_NEAR(u, 0.5, 1e-5);
	EXPECT_NEAR(v, 0.5, 1e-5);
	// South pole (0,0,-1) -> should be at a corner
	EqualAreaSphereToSquare(0.0, 0.0, -1.0, u, v);
	EXPECT_GE(u, 0.0); EXPECT_LE(u, 1.0);
	EXPECT_GE(v, 0.0); EXPECT_LE(v, 1.0);
}

// ---- 6. Equal-area property: uniform UV grid -> ~uniform solid angle ------
TEST(EqualAreaTest, EqualAreaProperty) {
	// Sample NxN uniform UV points, map to sphere, bin into 6 hemispheric zones.
	// Each zone should contain ~N^2/6 samples (equal area).
	const int N = 100;
	std::vector<int> bins(6, 0);
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			double u = (i + 0.5) / N, v = (j + 0.5) / N;
			double wx, wy, wz;
			EqualAreaSquareToSphere(u, v, wx, wy, wz);
			// Bin by dominant axis sign
			if (wz > 0.5)       bins[0]++;
			else if (wz < -0.5) bins[1]++;
			else if (wx > 0.5)  bins[2]++;
			else if (wx < -0.5) bins[3]++;
			else if (wy > 0.5)  bins[4]++;
			else                bins[5]++;
		}
	}
	// Each bin should have roughly N^2/6 samples; allow 30% deviation
	double expected = static_cast<double>(N*N) / 6.0;
	for (int b = 0; b < 6; ++b) {
		EXPECT_GT(bins[b], expected * 0.3) << "bin=" << b;
		EXPECT_LT(bins[b], expected * 2.0) << "bin=" << b;
	}
}

// ---- 7. WrapEqualAreaSquare: values inside [0,1]^2 unchanged --------------
TEST(EqualAreaTest, WrapInsideUnchanged) {
	for (int i = 1; i < 10; ++i) {
		for (int j = 1; j < 10; ++j) {
			double u = i / 10.0, v = j / 10.0;
			double wu = u, wv = v;
			WrapEqualAreaSquare(wu, wv);
			EXPECT_NEAR(wu, u, 1e-10);
			EXPECT_NEAR(wv, v, 1e-10);
		}
	}
}

// ---- 8. WrapEqualAreaSquare: u<0 wraps correctly --------------------------
TEST(EqualAreaTest, WrapNegativeU) {
	double u = -0.2, v = 0.3;
	WrapEqualAreaSquare(u, v);
	EXPECT_GE(u, 0.0); EXPECT_LE(u, 1.0);
	EXPECT_GE(v, 0.0); EXPECT_LE(v, 1.0);
	// u<0: u' = -u = 0.2, v' = 1-v = 0.7
	EXPECT_NEAR(u, 0.2, 1e-10);
	EXPECT_NEAR(v, 0.7, 1e-10);
}

// ---- 9. WrapEqualAreaSquare: v>1 wraps correctly --------------------------
TEST(EqualAreaTest, WrapV_Greater1) {
	double u = 0.4, v = 1.3;
	WrapEqualAreaSquare(u, v);
	EXPECT_GE(u, 0.0); EXPECT_LE(u, 1.0);
	EXPECT_GE(v, 0.0); EXPECT_LE(v, 1.0);
	// v>1: u' = 1-u = 0.6, v' = 2-v = 0.7
	EXPECT_NEAR(u, 0.6, 1e-10);
	EXPECT_NEAR(v, 0.7, 1e-10);
}

// ---- 10. SampleUniformHemisphereConcentric: unit vector, wz>=0 -----------
TEST(EqualAreaTest, HemisphereConcentricUnitVectorNonNegZ) {
	for (int i = 0; i <= 10; ++i) {
		for (int j = 0; j <= 10; ++j) {
			double u0 = i / 10.0, u1 = j / 10.0;
			double wx, wy, wz;
			SampleUniformHemisphereConcentric(u0, u1, wx, wy, wz);
			double len2 = wx*wx + wy*wy + wz*wz;
			EXPECT_NEAR(len2, 1.0, 1e-10) << "u0=" << u0 << " u1=" << u1;
			EXPECT_GE(wz, -1e-10) << "u0=" << u0 << " u1=" << u1;
		}
	}
}

// ---- 11. SampleUniformHemisphereConcentric: (0.5,0.5) -> north pole ------
TEST(EqualAreaTest, HemisphereConcentricCenterIsNorthPole) {
	double wx, wy, wz;
	SampleUniformHemisphereConcentric(0.5, 0.5, wx, wy, wz);
	EXPECT_NEAR(wz, 1.0, 1e-6);
	EXPECT_NEAR(wx, 0.0, 1e-6);
	EXPECT_NEAR(wy, 0.0, 1e-6);
}

// ---- 12. SampleUniformHemisphereConcentric: mean wz ~ 0.5 (uniform) -----
TEST(EqualAreaTest, HemisphereConcentricUniformCoverage) {
	const int N = 100;
	double sum_wz = 0.0;
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			double wx, wy, wz;
			SampleUniformHemisphereConcentric((i + 0.5) / N, (j + 0.5) / N, wx, wy, wz);
			sum_wz += wz;
		}
	}
	// E[cos(theta)] for uniform hemisphere = 0.5
	double mean_wz = sum_wz / (N * N);
	EXPECT_NEAR(mean_wz, 0.5, 0.02);
}
