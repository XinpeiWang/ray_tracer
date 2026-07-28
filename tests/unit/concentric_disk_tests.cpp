// concentric_disk_tests.cpp
// Unit tests for SampleUniformDiskConcentric and SampleCosineHemisphere
// (pbrt-v4 util/sampling.h alignment).
//
// Validates:
//   - Mapped points stay within unit disk
//   - Concentric mapping is area-preserving (chi-squared uniformity)
//   - Cosine hemisphere samples have z >= 0, unit length
//   - PDF equals cos(theta)/pi
//   - Origin maps to origin
//   - Hemisphere integral of pdf ~ 1

#include <gtest/gtest.h>
#include "../../src/shared/sampling.h"

#include <cmath>
#include <array>
#include <vector>

static constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// SampleUniformDiskConcentric
// ---------------------------------------------------------------------------

TEST(ConcentricDiskTest, MappedPointsInsideUnitDisk) {
	// All mapped points must satisfy dx^2 + dy^2 <= 1
	const int N = 10000;
	for (int i = 0; i < N; ++i) {
		double u0 = (i + 0.5) / N;
		double u1 = std::fmod((i * 0.6180339887) + 0.1, 1.0);
		double dx, dy;
		SampleUniformDiskConcentric(u0, u1, dx, dy);
		EXPECT_LE(dx*dx + dy*dy, 1.0 + 1e-9) << "i=" << i;
	}
}

TEST(ConcentricDiskTest, OriginMapsToOrigin) {
	double dx, dy;
	SampleUniformDiskConcentric(0.5, 0.5, dx, dy);
	EXPECT_NEAR(dx, 0.0, 1e-12);
	EXPECT_NEAR(dy, 0.0, 1e-12);
}

TEST(ConcentricDiskTest, UniformAreaDistribution) {
	// Divide disk into 4 equal-area quadrants; samples should be ~uniform.
	const int N = 80000;
	int quad[4] = {};
	for (int i = 0; i < N; ++i) {
		double u0 = (i + 0.3) / N;
		double u1 = std::fmod((i * 0.7548776662) + 0.2, 1.0);
		double dx, dy;
		SampleUniformDiskConcentric(u0, u1, dx, dy);
		if (dx >= 0 && dy >= 0) ++quad[0];
		else if (dx < 0 && dy >= 0) ++quad[1];
		else if (dx < 0 && dy < 0) ++quad[2];
		else ++quad[3];
	}
	double expected = N / 4.0;
	for (int q = 0; q < 4; ++q) {
		double rel_err = std::fabs(quad[q] - expected) / expected;
		EXPECT_LT(rel_err, 0.03) << "Quadrant " << q << " count=" << quad[q];
	}
}

// ---------------------------------------------------------------------------
// SampleCosineHemisphere
// ---------------------------------------------------------------------------

TEST(CosineHemisphereTest, ZIsNonNegative) {
	const int N = 5000;
	for (int i = 0; i < N; ++i) {
		double u0 = (i + 0.5) / N;
		double u1 = std::fmod((i * 0.6180339887) + 0.3, 1.0);
		double wx, wy, wz, pdf;
		SampleCosineHemisphere(u0, u1, wx, wy, wz, pdf);
		EXPECT_GE(wz, 0.0) << "i=" << i;
	}
}

TEST(CosineHemisphereTest, UnitLength) {
	const int N = 5000;
	for (int i = 0; i < N; ++i) {
		double u0 = (i + 0.5) / N;
		double u1 = std::fmod((i * 0.6180339887) + 0.3, 1.0);
		double wx, wy, wz, pdf;
		SampleCosineHemisphere(u0, u1, wx, wy, wz, pdf);
		double len2 = wx*wx + wy*wy + wz*wz;
		EXPECT_NEAR(len2, 1.0, 1e-9) << "i=" << i;
	}
}

TEST(CosineHemisphereTest, PDFEqualsCosOverPi) {
	const int N = 5000;
	for (int i = 0; i < N; ++i) {
		double u0 = (i + 0.5) / N;
		double u1 = std::fmod((i * 0.6180339887) + 0.3, 1.0);
		double wx, wy, wz, pdf;
		SampleCosineHemisphere(u0, u1, wx, wy, wz, pdf);
		double expected_pdf = wz / kPi;
		EXPECT_NEAR(pdf, expected_pdf, 1e-12) << "i=" << i;
	}
}

TEST(CosineHemisphereTest, PDFIntegratesOverHemisphere) {
	// Monte Carlo estimate of integral(pdf * dOmega) over upper hemisphere
	// should equal 1.  Use uniform hemisphere samples to estimate.
	const int N = 200000;
	double sum = 0.0;
	for (int i = 0; i < N; ++i) {
		double u0 = (i + 0.5) / N;
		double u1 = std::fmod((i * 0.6180339887) + 0.3, 1.0);
		// Uniform hemisphere: pdf_uniform = 1/(2*pi)
		double phi   = 2.0 * kPi * u0;
		double cos_t = u1;       // uniform in cos(theta)
		double sin_t = std::sqrt(1.0 - cos_t*cos_t);
		double wx = sin_t * std::cos(phi);
		double wy = sin_t * std::sin(phi);
		double wz = cos_t;
		// cosine-hemisphere pdf at this direction
		double f = CosineHemispherePDF(wz);
		// weight by dOmega = 2*pi (uniform hemisphere solid angle)
		sum += f * 2.0 * kPi;
	}
	double integral = sum / N;
	EXPECT_NEAR(integral, 1.0, 0.01);
}

TEST(CosineHemisphereTest, PDFMatchesCosineHemispherePDF) {
	const int N = 1000;
	for (int i = 0; i < N; ++i) {
		double u0 = (i + 0.5) / N;
		double u1 = std::fmod((i * 0.6180339887) + 0.4, 1.0);
		double wx, wy, wz, pdf;
		SampleCosineHemisphere(u0, u1, wx, wy, wz, pdf);
		EXPECT_NEAR(pdf, CosineHemispherePDF(wz), 1e-12);
	}
}
