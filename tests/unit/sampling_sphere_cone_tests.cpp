// Unit tests for the sphere/cone/disk sampling extensions to src/shared/sampling.h
// pbrt-v4 references: util/sampling.h
//
// Covers:
//   SampleUniformDiskPolar / InvertUniformDiskPolarSample
//   InvertUniformDiskConcentricSample
//   SampleUniformSphere / UniformSpherePDF / InvertUniformSphereSample
//   SampleUniformHemisphere / UniformHemispherePDF / InvertUniformHemisphereSample
//   SampleUniformCone / UniformConePDF / InvertUniformConeSample
//   InvertCosineHemisphereSample
//   InvertUniformTriangleSample

#include <gtest/gtest.h>
#include <cmath>
#include "../../src/shared/sampling.h"

static constexpr double kPi  = 3.14159265358979323846;
static constexpr double kEps = 1e-9;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static double dot3(double ax, double ay, double az,
				   double bx, double by, double bz) {
	return ax*bx + ay*by + az*bz;
}
static double len3(double x, double y, double z) {
	return std::sqrt(x*x + y*y + z*z);
}

// ===========================================================================
// SampleUniformDiskPolar
// ===========================================================================
TEST(SamplingSphereCone, DiskPolarUnitRadius) {
	double dx, dy;
	SampleUniformDiskPolar(1.0, 0.0, dx, dy);
	EXPECT_NEAR(std::sqrt(dx*dx + dy*dy), 1.0, kEps);
}

TEST(SamplingSphereCone, DiskPolarOrigin) {
	double dx, dy;
	SampleUniformDiskPolar(0.0, 0.0, dx, dy);
	EXPECT_NEAR(dx, 0.0, kEps);
	EXPECT_NEAR(dy, 0.0, kEps);
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

TEST(SamplingSphereCone, DiskPolarInvertAngleWrap) {
	// phi in (-pi, 0) range: make sure inversion wraps to [0,1]
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
TEST(SamplingSphereCone, DiskConcentricInvertRoundTrip) {
	for (auto [u0, u1] : std::initializer_list<std::pair<double,double>>{
			{0.1, 0.2}, {0.9, 0.1}, {0.5, 0.5}, {0.8, 0.9}, {0.3, 0.7}}) {
		double dx, dy;
		SampleUniformDiskConcentric(u0, u1, dx, dy);
		double ru0, ru1;
		InvertUniformDiskConcentricSample(dx, dy, ru0, ru1);
		EXPECT_NEAR(ru0, u0, 1e-7) << "u0=" << u0 << " u1=" << u1;
		EXPECT_NEAR(ru1, u1, 1e-7) << "u0=" << u0 << " u1=" << u1;
	}
}

// ===========================================================================
// SampleUniformSphere
// ===========================================================================
TEST(SamplingSphereCone, UniformSphereUnitLength) {
	for (auto [u0, u1] : std::initializer_list<std::pair<double,double>>{
			{0.0, 0.0}, {0.5, 0.5}, {1.0, 0.0}, {0.25, 0.75}}) {
		double wx, wy, wz;
		SampleUniformSphere(u0, u1, wx, wy, wz);
		EXPECT_NEAR(len3(wx, wy, wz), 1.0, 1e-12);
	}
}

TEST(SamplingSphereCone, UniformSphereCoversFullRange) {
	// u0=0 -> wz=1 (north pole), u0=1 -> wz=-1 (south pole)
	double wx, wy, wz;
	SampleUniformSphere(0.0, 0.0, wx, wy, wz);
	EXPECT_NEAR(wz, 1.0, kEps);
	SampleUniformSphere(1.0, 0.0, wx, wy, wz);
	EXPECT_NEAR(wz, -1.0, kEps);
}

TEST(SamplingSphereCone, UniformSpherePDFValue) {
	EXPECT_NEAR(UniformSpherePDF<double>(), 1.0 / (4.0 * kPi), kEps);
}

TEST(SamplingSphereCone, UniformSphereInvertRoundTrip) {
	for (auto [u0, u1] : std::initializer_list<std::pair<double,double>>{
			{0.1, 0.2}, {0.5, 0.5}, {0.9, 0.8}, {0.3, 0.0}}) {
		double wx, wy, wz;
		SampleUniformSphere(u0, u1, wx, wy, wz);
		double ru0, ru1;
		InvertUniformSphereSample(wx, wy, wz, ru0, ru1);
		EXPECT_NEAR(ru0, u0, 1e-9) << "u0=" << u0 << " u1=" << u1;
		EXPECT_NEAR(ru1, u1, 1e-9) << "u0=" << u0 << " u1=" << u1;
	}
}

// ===========================================================================
// SampleUniformHemisphere
// ===========================================================================
TEST(SamplingSphereCone, UniformHemisphereUnitLength) {
	for (auto [u0, u1] : std::initializer_list<std::pair<double,double>>{
			{0.0, 0.0}, {0.5, 0.25}, {1.0, 0.5}}) {
		double wx, wy, wz;
		SampleUniformHemisphere(u0, u1, wx, wy, wz);
		EXPECT_NEAR(len3(wx, wy, wz), 1.0, 1e-12);
	}
}

TEST(SamplingSphereCone, UniformHemisphereUpperOnly) {
	for (double u0 : {0.0, 0.1, 0.5, 0.9, 1.0}) {
		double wx, wy, wz;
		SampleUniformHemisphere(u0, 0.5, wx, wy, wz);
		EXPECT_GE(wz, -kEps);
	}
}

TEST(SamplingSphereCone, UniformHemispherePDFValue) {
	EXPECT_NEAR(UniformHemispherePDF<double>(), 1.0 / (2.0 * kPi), kEps);
}

TEST(SamplingSphereCone, UniformHemisphereInvertRoundTrip) {
	for (auto [u0, u1] : std::initializer_list<std::pair<double,double>>{
			{0.1, 0.3}, {0.5, 0.5}, {0.9, 0.7}, {0.0, 0.0}}) {
		double wx, wy, wz;
		SampleUniformHemisphere(u0, u1, wx, wy, wz);
		double ru0, ru1;
		InvertUniformHemisphereSample(wx, wy, wz, ru0, ru1);
		EXPECT_NEAR(ru0, u0, 1e-9) << "u0=" << u0 << " u1=" << u1;
		EXPECT_NEAR(ru1, u1, 1e-9) << "u0=" << u0 << " u1=" << u1;
	}
}

// ===========================================================================
// SampleUniformCone
// ===========================================================================
TEST(SamplingSphereCone, UniformConeUnitLength) {
	double wx, wy, wz;
	SampleUniformCone(0.5, 0.5, 0.9, wx, wy, wz);
	EXPECT_NEAR(len3(wx, wy, wz), 1.0, 1e-12);
}

TEST(SamplingSphereCone, UniformConeInsideCone) {
	double cosThetaMax = 0.866; // ~30 degrees
	for (double u0 : {0.0, 0.25, 0.5, 0.75, 1.0}) {
		double wx, wy, wz;
		SampleUniformCone(u0, 0.5, cosThetaMax, wx, wy, wz);
		// wz = cosTheta must be in [cosThetaMax, 1]
		EXPECT_GE(wz, cosThetaMax - 1e-12);
		EXPECT_LE(wz, 1.0 + 1e-12);
	}
}

TEST(SamplingSphereCone, UniformConeAtBoundaryU0_0) {
	// u0=0 -> cosTheta=1 -> north pole
	double wx, wy, wz;
	SampleUniformCone(0.0, 0.0, 0.5, wx, wy, wz);
	EXPECT_NEAR(wz, 1.0, kEps);
}

TEST(SamplingSphereCone, UniformConeAtBoundaryU0_1) {
	// u0=1 -> cosTheta=cosThetaMax
	double cosThetaMax = 0.7;
	double wx, wy, wz;
	SampleUniformCone(1.0, 0.0, cosThetaMax, wx, wy, wz);
	EXPECT_NEAR(wz, cosThetaMax, kEps);
}

TEST(SamplingSphereCone, UniformConePDFValue) {
	double ctm = 0.8;
	EXPECT_NEAR(UniformConePDF<double>(ctm), 1.0 / (2.0 * kPi * (1.0 - ctm)), kEps);
}

TEST(SamplingSphereCone, UniformConeInvertRoundTrip) {
	double cosThetaMax = 0.75;
	for (auto [u0, u1] : std::initializer_list<std::pair<double,double>>{
			{0.1, 0.2}, {0.5, 0.5}, {0.9, 0.8}, {0.3, 0.0}}) {
		double wx, wy, wz;
		SampleUniformCone(u0, u1, cosThetaMax, wx, wy, wz);
		double ru0, ru1;
		InvertUniformConeSample(wx, wy, wz, cosThetaMax, ru0, ru1);
		EXPECT_NEAR(ru0, u0, 1e-9) << "u0=" << u0 << " u1=" << u1;
		EXPECT_NEAR(ru1, u1, 1e-9) << "u0=" << u0 << " u1=" << u1;
	}
}

// ===========================================================================
// InvertCosineHemisphereSample
// ===========================================================================
TEST(SamplingSphereCone, CosineHemisphereInvertRoundTrip) {
	for (auto [u0, u1] : std::initializer_list<std::pair<double,double>>{
			{0.1, 0.2}, {0.5, 0.5}, {0.8, 0.9}, {0.3, 0.7}}) {
		double wx, wy, wz, pdf;
		SampleCosineHemisphere(u0, u1, wx, wy, wz, pdf);
		double ru0, ru1;
		InvertCosineHemisphereSample(wx, wy, wz, ru0, ru1);
		EXPECT_NEAR(ru0, u0, 1e-7) << "u0=" << u0 << " u1=" << u1;
		EXPECT_NEAR(ru1, u1, 1e-7) << "u0=" << u0 << " u1=" << u1;
	}
}

// ===========================================================================
// InvertUniformTriangleSample
// ===========================================================================
TEST(SamplingSphereCone, UniformTriangleInvertRoundTrip) {
	for (auto [u0, u1] : std::initializer_list<std::pair<double,double>>{
			{0.1, 0.2}, {0.5, 0.4}, {0.3, 0.7}, {0.8, 0.1}, {0.6, 0.6}}) {
		double b0, b1, b2;
		SampleUniformTriangle(u0, u1, b0, b1, b2);
		double ru0, ru1;
		InvertUniformTriangleSample(b0, b1, b2, ru0, ru1);
		EXPECT_NEAR(ru0, u0, 1e-9) << "u0=" << u0 << " u1=" << u1;
		EXPECT_NEAR(ru1, u1, 1e-9) << "u0=" << u0 << " u1=" << u1;
	}
}

TEST(SamplingSphereCone, UniformTriangleInvertBarycentricSum) {
	// Output barycentric coords from SampleUniformTriangle must sum to 1
	for (auto [u0, u1] : std::initializer_list<std::pair<double,double>>{
			{0.1, 0.2}, {0.9, 0.05}, {0.5, 0.5}}) {
		double b0, b1, b2;
		SampleUniformTriangle(u0, u1, b0, b1, b2);
		EXPECT_NEAR(b0 + b1 + b2, 1.0, 1e-12);
	}
}

// ===========================================================================
// Float specialisation (compile + basic smoke test)
// ===========================================================================
TEST(SamplingSphereCone, FloatSpecialisationSphere) {
	float wx, wy, wz;
	SampleUniformSphere(0.5f, 0.5f, wx, wy, wz);
	float l = std::sqrt(wx*wx + wy*wy + wz*wz);
	EXPECT_NEAR(l, 1.0f, 1e-6f);
}

TEST(SamplingSphereCone, FloatSpecialisationCone) {
	float wx, wy, wz;
	SampleUniformCone(0.5f, 0.5f, 0.9f, wx, wy, wz);
	float l = std::sqrt(wx*wx + wy*wy + wz*wz);
	EXPECT_NEAR(l, 1.0f, 1e-6f);
}
