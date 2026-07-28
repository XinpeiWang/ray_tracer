// anisotropic_ggx_tests.cpp
// Unit tests for anisotropic GGX roughness support (alpha_x / alpha_y).
// Validates:
//   - Isotropic case (alpha_x == alpha_y) matches original behavior
//   - Anisotropic case (alpha_x != alpha_y) produces different lobes per axis
//   - All materials accept two-roughness constructors without errors
//   - BxDF sampling stays valid with anisotropic parameters
//   - Brushed-metal test: strong anisotropy produces elongated highlight

#include <gtest/gtest.h>
#include "../../src/shared/bxdfs.h"
#include "../../src/shared/microfacet.h"

#include <cmath>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// TrowbridgeReitz anisotropic D() is larger in the perpendicular rough axis
// ---------------------------------------------------------------------------

TEST(AnisotropicGGXTest, IsotropicSymmetry) {
	// With alpha_x == alpha_y, D(wm) should be equal for wm rotated 90deg in tangent plane
	TrowbridgeReitz<double> iso(0.3, 0.3);
	// Two half-vectors symmetric about z, same phi offset
	double phi1 = 0.3, phi2 = 0.3 + 3.14159265358979323846 / 2.0;
	double sin_t = std::sin(0.4), cos_t = std::cos(0.4);
	double d1 = iso.D(sin_t*std::cos(phi1), sin_t*std::sin(phi1), cos_t);
	double d2 = iso.D(sin_t*std::cos(phi2), sin_t*std::sin(phi2), cos_t);
	EXPECT_NEAR(d1, d2, 1e-10);
}

TEST(AnisotropicGGXTest, AnisotropicBreaksSymmetry) {
	// With alpha_x != alpha_y, D(wm) differs at 0 vs 90 deg phi
	TrowbridgeReitz<double> aniso(0.1, 0.6);
	double sin_t = std::sin(0.4), cos_t = std::cos(0.4);
	double d_phi0  = aniso.D(sin_t, 0.0, cos_t);  // phi=0 (along x)
	double d_phi90 = aniso.D(0.0, sin_t, cos_t);  // phi=pi/2 (along y)
	EXPECT_NE(d_phi0, d_phi90);
	// Rougher axis (y, alpha_y=0.6) should give higher D for phi=90
	EXPECT_GT(d_phi90, d_phi0);
}

// ---------------------------------------------------------------------------
// ConductorBxDF anisotropic sampling
// ---------------------------------------------------------------------------

TEST(AnisotropicGGXTest, ConductorBxDFIsotropicValid) {
	// Isotropic: alpha_x == alpha_y
	ConductorBxDF<double> b;
	b.eta_r = 0.2; b.eta_g = 0.4; b.eta_b = 1.5;
	b.k_r = 3.0; b.k_g = 2.5; b.k_b = 1.8;
	b.alpha_x = b.alpha_y = 0.3;

	int valid = 0;
	for (int i = 0; i < 500; ++i) {
		double u0 = (i + 0.5) / 500.0;
		double u1 = std::fmod(i * 0.6180339887 + 0.1, 1.0);
		auto res = b.sample_local(0.0, 0.0, 1.0, u0, u1);
		if (res.valid) {
			EXPECT_GE(res.wo_z, 0.0);
			++valid;
		}
	}
	EXPECT_GT(valid, 400);
}

TEST(AnisotropicGGXTest, ConductorBxDFAnisotropicValid) {
	// Anisotropic: alpha_x != alpha_y
	ConductorBxDF<double> b;
	b.eta_r = 0.2; b.eta_g = 0.4; b.eta_b = 1.5;
	b.k_r = 3.0; b.k_g = 2.5; b.k_b = 1.8;
	b.alpha_x = 0.05;  // sharp in x (brushed metal highlight)
	b.alpha_y = 0.5;   // rough in y

	int valid = 0;
	for (int i = 0; i < 500; ++i) {
		double u0 = (i + 0.5) / 500.0;
		double u1 = std::fmod(i * 0.6180339887 + 0.1, 1.0);
		auto res = b.sample_local(0.0, 0.0, 1.0, u0, u1);
		if (res.valid) {
			EXPECT_GE(res.wo_z, 0.0);
			++valid;
		}
	}
	EXPECT_GT(valid, 400);
}

TEST(AnisotropicGGXTest, ConductorBxDFAnisotropicSpreadsDifferently) {
	// With strong anisotropy, spread in wy should be larger than in wx
	ConductorBxDF<double> b;
	b.eta_r = 0.2; b.eta_g = 0.4; b.eta_b = 1.5;
	b.k_r = 3.0; b.k_g = 2.5; b.k_b = 1.8;
	b.alpha_x = 0.05;  // sharp x
	b.alpha_y = 0.6;   // rough y

	std::vector<double> wx_vals, wy_vals;
	for (int i = 0; i < 2000; ++i) {
		double u0 = (i + 0.5) / 2000.0;
		double u1 = std::fmod(i * 0.6180339887 + 0.1, 1.0);
		auto res = b.sample_local(0.0, 0.0, 1.0, u0, u1);
		if (res.valid) {
			wx_vals.push_back(std::fabs(res.wo_x));
			wy_vals.push_back(std::fabs(res.wo_y));
		}
	}
	ASSERT_GT(wx_vals.size(), 100u);

	// Compute mean absolute deviation in each axis
	auto mean = [](const std::vector<double>& v) {
		double s = 0; for (auto x : v) s += x; return s / v.size();
	};
	double mx = mean(wx_vals);
	double my = mean(wy_vals);

	// Rougher y-axis should produce larger mean |wy|
	EXPECT_GT(my, mx) << "Expected anisotropic spread: mean|wy|=" << my << " > mean|wx|=" << mx;
}

// ---------------------------------------------------------------------------
// RoughMetalBxDF anisotropic
// ---------------------------------------------------------------------------

TEST(AnisotropicGGXTest, RoughMetalBxDFAnisotropicValid) {
	RoughMetalBxDF<double> b;
	b.albedo_r = 0.9; b.albedo_g = 0.8; b.albedo_b = 0.7;
	b.alpha_x = 0.05;
	b.alpha_y = 0.5;

	int valid = 0;
	for (int i = 0; i < 500; ++i) {
		double u0 = (i + 0.5) / 500.0;
		double u1 = std::fmod(i * 0.6180339887 + 0.1, 1.0);
		auto res = b.sample_local(0.0, 0.0, 1.0, u0, u1);
		if (res.valid) {
			EXPECT_GE(res.wo_z, 0.0);
			++valid;
		}
	}
	EXPECT_GT(valid, 400);
}

// ---------------------------------------------------------------------------
// RoughDielectricBxDF anisotropic
// ---------------------------------------------------------------------------

TEST(AnisotropicGGXTest, RoughDielectricBxDFAnisotropicValid) {
	RoughDielectricBxDF<double> b;
	b.ior = 1.5;
	b.alpha_x = 0.05;
	b.alpha_y = 0.5;

	int valid = 0;
	for (int i = 0; i < 500; ++i) {
		double u0 = (i + 0.5) / 500.0;
		double u1 = std::fmod(i * 0.6180339887 + 0.1, 1.0);
		double u2 = std::fmod(i * 0.7548776662 + 0.2, 1.0);
		double eta = 1.0 / 1.5;
		auto res = b.sample_local(0.0, 0.0, 1.0, eta, u0, u1, u2);
		if (res.valid) ++valid;
	}
	EXPECT_GT(valid, 400);
}
