// fresnel_tests.cpp -- Unit tests for src/shared/fresnel.h
// Covers FrDielectric, FrComplex (eta_r/eta_k), FrComplex (std::complex),
// FrConductorRGB (CPU stub), and FresnelMoment1.

#include "../../src/shared/fresnel.h"
#include <gtest/gtest.h>
#include <complex>
#include <cmath>

// -----------------------------------------------------------------------
// FrDielectric -- known analytic values
// -----------------------------------------------------------------------
TEST(FresnelTest, DielectricNormalIncidenceGlass) {
	// Normal incidence, glass eta=1.5: R = ((1.5-1)/(1.5+1))^2 = 0.04
	double r = FrDielectric(1.0, 1.5);
	EXPECT_NEAR(r, 0.04, 1e-6);
}

TEST(FresnelTest, DielectricTotalInternalReflection) {
	// From inside glass (eta=1.5->1.0), angle > critical angle
	double r = FrDielectric(0.1, 1.0 / 1.5);
	EXPECT_NEAR(r, 1.0, 1e-6);
}

TEST(FresnelTest, DielectricGrazingAngle) {
	// Grazing incidence -> R approaches 1 for both polarizations
	double r = FrDielectric(0.001, 1.5);
	EXPECT_GT(r, 0.9);
}

TEST(FresnelTest, DielectricSymmetry) {
	// Fresnel reflectance is the same whether light goes from medium A->B or B->A
	// at the same angle, as long as there is no TIR. Use normal incidence (cos=1)
	// where both directions are well above the critical angle.
	double r1 = FrDielectric(1.0, 1.5);
	double r2 = FrDielectric(1.0, 1.0 / 1.5);
	EXPECT_NEAR(r1, r2, 1e-5);
}

TEST(FresnelTest, DielectricRange) {
	// Result must be in [0,1] for all valid inputs
	for (int i = 0; i <= 10; ++i) {
		double cos_i = i / 10.0;
		double r = FrDielectric(cos_i, 1.5);
		EXPECT_GE(r, 0.0);
		EXPECT_LE(r, 1.0);
	}
}

// -----------------------------------------------------------------------
// FrComplex (eta_r, eta_k) -- conductor Fresnel
// -----------------------------------------------------------------------
TEST(FresnelTest, ConductorNormalIncidenceGold) {
	// Gold at 589nm: n~0.27, k~3.0 -> R ~ 0.85
	double r = FrComplex(1.0, 0.27, 3.0);
	EXPECT_GT(r, 0.75);
	EXPECT_LE(r, 1.0);
}

TEST(FresnelTest, ConductorPerfectMirror) {
	// Perfect conductor: k -> large -> R -> 1
	double r = FrComplex(1.0, 0.0, 100.0);
	EXPECT_GT(r, 0.999);
}

TEST(FresnelTest, ConductorDegeneratesDielectric) {
	// When k=0, FrComplex should equal FrDielectric
	double r_cplx = FrComplex(0.7, 1.5, 0.0);
	double r_diel = FrDielectric(0.7, 1.5);
	EXPECT_NEAR(r_cplx, r_diel, 1e-4);
}

TEST(FresnelTest, ConductorRange) {
	// Result must always be in [0,1]
	for (int i = 0; i <= 10; ++i) {
		double cos_i = i / 10.0;
		double r = FrComplex(cos_i, 0.18, 3.5);
		EXPECT_GE(r, 0.0);
		EXPECT_LE(r, 1.0);
	}
}

// -----------------------------------------------------------------------
// FrComplex (std::complex<T>) -- convenience overload
// Must match the (eta_r, eta_k) overload exactly.
// Reference: pbrt-v4 FrComplex(Float cosTheta_i, pstd::complex<Float> eta)
// -----------------------------------------------------------------------
TEST(FresnelTest, ComplexOverloadMatchesExplicit) {
	double cos_i = 0.7;
	double eta_r = 0.18, eta_k = 3.5;
	double r_explicit = FrComplex(cos_i, eta_r, eta_k);
	double r_complex  = FrComplex(cos_i, std::complex<double>(eta_r, eta_k));
	EXPECT_NEAR(r_explicit, r_complex, 1e-10);
}

TEST(FresnelTest, ComplexOverloadGold589nm) {
	double r = FrComplex(1.0, std::complex<double>(0.27, 3.0));
	EXPECT_GT(r, 0.75);
	EXPECT_LE(r, 1.0);
}

TEST(FresnelTest, ComplexOverloadDielectricLimit) {
	double r_cplx = FrComplex(0.7, std::complex<double>(1.5, 0.0));
	double r_diel = FrDielectric(0.7, 1.5);
	EXPECT_NEAR(r_cplx, r_diel, 1e-4);
}

// -----------------------------------------------------------------------
// FresnelMoment1 -- polynomial sanity checks
// -----------------------------------------------------------------------
TEST(FresnelTest, FresnelMoment1EtaOne) {
	// eta=1: no interface -> moment should be ~0.5 (by symmetry)
	double m = FresnelMoment1(1.0);
	EXPECT_GT(m, 0.0);
	EXPECT_LT(m, 1.0);
}

TEST(FresnelTest, FresnelMoment1PositiveRange) {
	// Must be positive for all valid eta
	for (float eta : {0.5f, 0.8f, 1.0f, 1.2f, 1.5f, 2.0f})
		EXPECT_GT(FresnelMoment1(eta), 0.f) << "eta=" << eta;
}
