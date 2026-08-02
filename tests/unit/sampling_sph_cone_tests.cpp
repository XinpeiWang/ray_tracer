// Unit tests for the sphere/cone/disk sampling extensions in sampling_sphere_cone.h
// pbrt-v4 references: util/sampling.h
//
// Covers:
//   SampleUniformDiskPolar         / InvertUniformDiskPolarSample
//   InvertUniformDiskConcentricSample
//   SampleUniformSphere            / UniformSpherePDF / InvertUniformSphereSample
//   SampleUniformHemisphere        / UniformHemispherePDF / InvertUniformHemisphereSample
//   SampleUniformCone              / UniformConePDF / InvertUniformConeSample
//   InvertCosineHemisphereSample
//   InvertUniformTriangleSample

#include <gtest/gtest.h>
#include <cmath>
#include "../../src/shared/sampling_sphere_cone.h"

static constexpr double kPi2  = 3.14159265358979323846;
static constexpr double kEps2 = 1e-9;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static double len3sc(double x, double y, double z) {
	return std::sqrt(x*x + y*y + z*z);
}

// ===========================================================================
// SampleUniformDiskPolar
// ===========================================================================
TEST(SamplingSphereCone, DiskPolarUnitRadius) {
	double dx, dy;
	SampleUniformDiskPolar(1.0, 0.0, dx, dy);
	EXPECT_NEAR(std::sqrt(dx*dx + dy*dy), 1.0, kEps2);
}

TEST(SamplingSphereCone, DiskPolarOrigin) {
	double dx, dy;
	SampleUniformDiskPolar(0.0, 0.0, dx, dy);
	EXPECT_NEAR(dx, 0.0, kEps2);
	EXPECT_NEAR(dy, 0.0, kEps2);
}

TEST(SamplingSphereCone, DiskPolarInsideDisk) {
	for (double u0 : {0.1, 0.25, 0.5, 0.75, 0.99}) {
		for (double u1 : {0.0, 0.25, 0.5, 0.75}) {
			double dx, dy;
			SampleUniformDiskPolar(u0, u1, dx, dy);
			EXPECT_LE(dx*dx + dy*dy, 1.0 + 1e-12);
		}
	}
}

TEST(SamplingSphereCone, DiskPolarInvertRoundTrip) {
	double dx, dy;
	SampleUniformDiskPolar(0.36, 0.7, dx, dy);
	double ru0, ru1;
	InvertUniformDiskPolarSample(dx, dy, ru0, ru1);
	EXPECT_NEAR(ru0, 0.36, 1e-9);
	EXPECT_NEAR(ru1, 0.7,  1e-9);
}

TEST(SamplingSphereCone, DiskPolarInvertRoundTripB) {
	double dx, dy;
	SampleUniformDiskPolar(0.5, 0.8, dx, dy);
	double u0, u1;
	InvertUniformDiskPolarSample(dx, dy, u0, u1);
	EXPECT_NEAR(u0, 0.5, 1e-9);
	EXPECT_NEAR(u1, 0.8, 1e-9);
}

// ===========================================================================
// InvertUniformDiskConcentricSample
// ===========================================================================
TEST(SamplingSphereCone, DiskConcentricInvertRT1) {
	double dx, dy, ru0, ru1;
	SampleUniformDiskConcentric(0.1, 0.2, dx, dy);
	InvertUniformDiskConcentricSample(dx, dy, ru0, ru1);
	EXPECT_NEAR(ru0, 0.1, 1e-7);
	EXPECT_NEAR(ru1, 0.2, 1e-7);
}

TEST(SamplingSphereCone, DiskConcentricInvertRT2) {
	double dx, dy, ru0, ru1;
	SampleUniformDiskConcentric(0.9, 0.1, dx, dy);
	InvertUniformDiskConcentricSample(dx, dy, ru0, ru1);
	EXPECT_NEAR(ru0, 0.9, 1e-7);
	EXPECT_NEAR(ru1, 0.1, 1e-7);
}

TEST(SamplingSphereCone, DiskConcentricInvertRT3) {
	double dx, dy, ru0, ru1;
	SampleUniformDiskConcentric(0.8, 0.9, dx, dy);
	InvertUniformDiskConcentricSample(dx, dy, ru0, ru1);
	EXPECT_NEAR(ru0, 0.8, 1e-7);
	EXPECT_NEAR(ru1, 0.9, 1e-7);
}

// ===========================================================================
// SampleUniformSphere
// ===========================================================================
TEST(SamplingSphereCone, UniformSphereUnitLength) {
	for (auto [u0, u1] : std::initializer_list<std::pair<double,double>>{
			{0.0, 0.0}, {0.5, 0.5}, {1.0, 0.0}, {0.25, 0.75}}) {
		double wx, wy, wz;
		SampleUniformSphere(u0, u1, wx, wy, wz);
		EXPECT_NEAR(len3sc(wx, wy, wz), 1.0, 1e-12);
	}
}

TEST(SamplingSphereCone, UniformSphereNorthPole) {
	double wx, wy, wz;
	SampleUniformSphere(0.0, 0.0, wx, wy, wz);
	EXPECT_NEAR(wz, 1.0, kEps2);
}

TEST(SamplingSphereCone, UniformSphereSouthPole) {
	double wx, wy, wz;
	SampleUniformSphere(1.0, 0.0, wx, wy, wz);
	EXPECT_NEAR(wz, -1.0, kEps2);
}

TEST(SamplingSphereCone, UniformSpherePDFValue) {
	EXPECT_NEAR(UniformSpherePDF<double>(), 1.0 / (4.0 * kPi2), kEps2);
}

TEST(SamplingSphereCone, UniformSphereInvertRT1) {
	double wx, wy, wz, ru0, ru1;
	SampleUniformSphere(0.1, 0.2, wx, wy, wz);
	InvertUniformSphereSample(wx, wy, wz, ru0, ru1);
	EXPECT_NEAR(ru0, 0.1, 1e-9);
	EXPECT_NEAR(ru1, 0.2, 1e-9);
}

TEST(SamplingSphereCone, UniformSphereInvertRT2) {
	double wx, wy, wz, ru0, ru1;
	SampleUniformSphere(0.9, 0.8, wx, wy, wz);
	InvertUniformSphereSample(wx, wy, wz, ru0, ru1);
	EXPECT_NEAR(ru0, 0.9, 1e-9);
	EXPECT_NEAR(ru1, 0.8, 1e-9);
}

TEST(SamplingSphereCone, UniformSphereInvertRT3) {
	double wx, wy, wz, ru0, ru1;
	SampleUniformSphere(0.3, 0.0, wx, wy, wz);
	InvertUniformSphereSample(wx, wy, wz, ru0, ru1);
	EXPECT_NEAR(ru0, 0.3, 1e-9);
	EXPECT_NEAR(ru1, 0.0, 1e-9);
}

// ===========================================================================
// SampleUniformHemisphere
// ===========================================================================
TEST(SamplingSphereCone, UniformHemisphereUnitLength) {
	for (auto [u0, u1] : std::initializer_list<std::pair<double,double>>{
			{0.0, 0.0}, {0.5, 0.25}, {1.0, 0.5}}) {
		double wx, wy, wz;
		SampleUniformHemisphere(u0, u1, wx, wy, wz);
		EXPECT_NEAR(len3sc(wx, wy, wz), 1.0, 1e-12);
	}
}

TEST(SamplingSphereCone, UniformHemisphereUpperOnly) {
	for (double u0 : {0.0, 0.1, 0.5, 0.9, 1.0}) {
		double wx, wy, wz;
		SampleUniformHemisphere(u0, 0.5, wx, wy, wz);
		EXPECT_GE(wz, -kEps2);
	}
}

TEST(SamplingSphereCone, UniformHemispherePDFValue) {
	EXPECT_NEAR(UniformHemispherePDF<double>(), 1.0 / (2.0 * kPi2), kEps2);
}

TEST(SamplingSphereCone, UniformHemisphereInvertRT1) {
	double wx, wy, wz, ru0, ru1;
	SampleUniformHemisphere(0.1, 0.3, wx, wy, wz);
	InvertUniformHemisphereSample(wx, wy, wz, ru0, ru1);
	EXPECT_NEAR(ru0, 0.1, 1e-9);
	EXPECT_NEAR(ru1, 0.3, 1e-9);
}

TEST(SamplingSphereCone, UniformHemisphereInvertRT2) {
	double wx, wy, wz, ru0, ru1;
	SampleUniformHemisphere(0.9, 0.7, wx, wy, wz);
	InvertUniformHemisphereSample(wx, wy, wz, ru0, ru1);
	EXPECT_NEAR(ru0, 0.9, 1e-9);
	EXPECT_NEAR(ru1, 0.7, 1e-9);
}

TEST(SamplingSphereCone, UniformHemisphereInvertRT3) {
	double wx, wy, wz, ru0, ru1;
	SampleUniformHemisphere(0.0, 0.0, wx, wy, wz);
	InvertUniformHemisphereSample(wx, wy, wz, ru0, ru1);
	EXPECT_NEAR(ru0, 0.0, 1e-9);
	EXPECT_NEAR(ru1, 0.0, 1e-9);
}

// ===========================================================================
// SampleUniformCone
// ===========================================================================
TEST(SamplingSphereCone, UniformConeUnitLength) {
	double wx, wy, wz;
	SampleUniformCone(0.5, 0.5, 0.9, wx, wy, wz);
	EXPECT_NEAR(len3sc(wx, wy, wz), 1.0, 1e-12);
}

TEST(SamplingSphereCone, UniformConeInsideCone) {
	double cosThetaMax = 0.866;
	for (double u0 : {0.0, 0.25, 0.5, 0.75, 1.0}) {
		double wx, wy, wz;
		SampleUniformCone(u0, 0.5, cosThetaMax, wx, wy, wz);
		EXPECT_GE(wz, cosThetaMax - 1e-12);
		EXPECT_LE(wz, 1.0 + 1e-12);
	}
}

TEST(SamplingSphereCone, UniformConeAtNorthPole) {
	double wx, wy, wz;
	SampleUniformCone(0.0, 0.0, 0.5, wx, wy, wz);
	EXPECT_NEAR(wz, 1.0, kEps2);
}

TEST(SamplingSphereCone, UniformConeAtMaxAngle) {
	double cosThetaMax = 0.7;
	double wx, wy, wz;
	SampleUniformCone(1.0, 0.0, cosThetaMax, wx, wy, wz);
	EXPECT_NEAR(wz, cosThetaMax, kEps2);
}

TEST(SamplingSphereCone, UniformConePDFValue) {
	double ctm = 0.8;
	EXPECT_NEAR(UniformConePDF<double>(ctm), 1.0 / (2.0 * kPi2 * (1.0 - ctm)), kEps2);
}

TEST(SamplingSphereCone, UniformConeInvertRT1) {
	double cosThetaMax = 0.75;
	double wx, wy, wz, ru0, ru1;
	SampleUniformCone(0.1, 0.2, cosThetaMax, wx, wy, wz);
	InvertUniformConeSample(wx, wy, wz, cosThetaMax, ru0, ru1);
	EXPECT_NEAR(ru0, 0.1, 1e-9);
	EXPECT_NEAR(ru1, 0.2, 1e-9);
}

TEST(SamplingSphereCone, UniformConeInvertRT2) {
	double cosThetaMax = 0.75;
	double wx, wy, wz, ru0, ru1;
	SampleUniformCone(0.9, 0.8, cosThetaMax, wx, wy, wz);
	InvertUniformConeSample(wx, wy, wz, cosThetaMax, ru0, ru1);
	EXPECT_NEAR(ru0, 0.9, 1e-9);
	EXPECT_NEAR(ru1, 0.8, 1e-9);
}

// ===========================================================================
// InvertCosineHemisphereSample
// ===========================================================================
TEST(SamplingSphereCone, CosineHemisphereInvertRT1) {
	double wx, wy, wz, pdf;
	SampleCosineHemisphere(0.1, 0.2, wx, wy, wz, pdf);
	double ru0, ru1;
	InvertCosineHemisphereSample(wx, wy, wz, ru0, ru1);
	EXPECT_NEAR(ru0, 0.1, 1e-7);
	EXPECT_NEAR(ru1, 0.2, 1e-7);
}

TEST(SamplingSphereCone, CosineHemisphereInvertRT2) {
	double wx, wy, wz, pdf;
	SampleCosineHemisphere(0.8, 0.9, wx, wy, wz, pdf);
	double ru0, ru1;
	InvertCosineHemisphereSample(wx, wy, wz, ru0, ru1);
	EXPECT_NEAR(ru0, 0.8, 1e-7);
	EXPECT_NEAR(ru1, 0.9, 1e-7);
}

// ===========================================================================
// InvertUniformTriangleSample
// ===========================================================================
TEST(SamplingSphereCone, UniformTriangleInvertRT1) {
	double b0, b1, b2, ru0, ru1;
	SampleUniformTriangle(0.1, 0.2, b0, b1, b2);
	InvertUniformTriangleSample(b0, b1, b2, ru0, ru1);
	EXPECT_NEAR(ru0, 0.1, 1e-9);
	EXPECT_NEAR(ru1, 0.2, 1e-9);
}

TEST(SamplingSphereCone, UniformTriangleInvertRT2) {
	double b0, b1, b2, ru0, ru1;
	SampleUniformTriangle(0.3, 0.7, b0, b1, b2);
	InvertUniformTriangleSample(b0, b1, b2, ru0, ru1);
	EXPECT_NEAR(ru0, 0.3, 1e-9);
	EXPECT_NEAR(ru1, 0.7, 1e-9);
}

TEST(SamplingSphereCone, UniformTriangleBarycentricSumIsOne) {
	for (auto [u0, u1] : std::initializer_list<std::pair<double,double>>{
			{0.1, 0.2}, {0.9, 0.05}, {0.5, 0.5}}) {
		double b0, b1, b2;
		SampleUniformTriangle(u0, u1, b0, b1, b2);
		EXPECT_NEAR(b0 + b1 + b2, 1.0, 1e-12);
	}
}

// ===========================================================================
// Float specialisation smoke tests
// ===========================================================================
TEST(SamplingSphereCone, FloatSphereSmokeTest) {
	float wx, wy, wz;
	SampleUniformSphere(0.5f, 0.5f, wx, wy, wz);
	EXPECT_NEAR(std::sqrt(wx*wx + wy*wy + wz*wz), 1.0f, 1e-6f);
}

TEST(SamplingSphereCone, FloatConeSmokeTest) {
	float wx, wy, wz;
	SampleUniformCone(0.5f, 0.5f, 0.9f, wx, wy, wz);
	EXPECT_NEAR(std::sqrt(wx*wx + wy*wy + wz*wz), 1.0f, 1e-6f);
}

TEST(SamplingSphereCone, FloatHemisphereSmokeTest) {
	float wx, wy, wz;
	SampleUniformHemisphere(0.5f, 0.5f, wx, wy, wz);
	EXPECT_NEAR(std::sqrt(wx*wx + wy*wy + wz*wz), 1.0f, 1e-6f);
}
