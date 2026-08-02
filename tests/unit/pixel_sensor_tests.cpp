// ---------------------------------------------------------------------------
// pixel_sensor_tests.cpp -- Unit tests for pixel_sensor.h
//
// Tests cover:
//   InnerProduct          -- spectral inner product
//   SpectrumToXYZ         -- spectrum to CIE XYZ
//   GetD65Illuminant      -- D65 spectral data sanity
//   PixelSensor (XYZ)     -- CreateDefault, ToSensorRGB, XYZFromSensorRGB
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>
#include "../../src/shared/pixel_sensor.h"
#include "../../src/shared/spectrum_types.h"
#include "../../src/shared/spectral_math.h"

// ===========================================================================
// InnerProduct tests
// ===========================================================================

TEST(InnerProduct, ConstantSpectra) {
	// InnerProduct(1, 1) over [360, 830] should equal 471 (step 1nm inclusive)
	ConstantSpectrum one(1.f);
	float ip = InnerProduct(one, one);
	EXPECT_NEAR(ip, 471.f, 1e-3f);
}

TEST(InnerProduct, Symmetry) {
	// InnerProduct(f, g) == InnerProduct(g, f)
	ConstantSpectrum a(2.f);
	ConstantSpectrum b(3.f);
	EXPECT_NEAR(InnerProduct(a, b), InnerProduct(b, a), 1e-5f);
}

TEST(InnerProduct, ScaledConstant) {
	// InnerProduct(2, 3) = 2*3 * 471
	ConstantSpectrum two(2.f);
	ConstantSpectrum three(3.f);
	EXPECT_NEAR(InnerProduct(two, three), 6.f * 471.f, 1e-2f);
}

// ===========================================================================
// SpectrumToXYZ tests
// ===========================================================================

TEST(SpectrumToXYZ, WhiteSpectrum) {
	// SpectrumToXYZ(1) with equal-energy: Y should be 1.0 by definition.
	ConstantSpectrum one(1.f);
	XYZ xyz = SpectrumToXYZ(one);
	EXPECT_NEAR(xyz.Y, 1.f, 5e-3f);
	// CIE E white: X≈Y≈Z
	EXPECT_NEAR(xyz.X, xyz.Z, 0.05f);
}

TEST(SpectrumToXYZ, ZeroSpectrum) {
	ConstantSpectrum zero(0.f);
	XYZ xyz = SpectrumToXYZ(zero);
	EXPECT_NEAR(xyz.X, 0.f, 1e-6f);
	EXPECT_NEAR(xyz.Y, 0.f, 1e-6f);
	EXPECT_NEAR(xyz.Z, 0.f, 1e-6f);
}

TEST(SpectrumToXYZ, CIE_Y_Integral) {
	// InnerProduct(CIE_Y, ConstantSpectrum(1)) / kCIE_Y_integral should be 1
	ConstantSpectrum one(1.f);
	float ip = InnerProduct(GetCIE_Y(), one);
	EXPECT_NEAR(ip / kCIE_Y_integral, 1.f, 1e-4f);
}

// ===========================================================================
// D65 illuminant tests
// ===========================================================================

TEST(D65Illuminant, PeakInBlueRegion) {
	// D65 has a prominent peak near 450-460nm (>100)
	const DenselySampledSpectrum& d65 = GetD65Illuminant();
	EXPECT_GT(d65(450.f), 100.f);
}

TEST(D65Illuminant, NormalisedAt560nm) {
	// D65 is defined with 100.0 at 560nm
	const DenselySampledSpectrum& d65 = GetD65Illuminant();
	EXPECT_NEAR(d65(560.f), 100.f, 1.f);
}

TEST(D65Illuminant, AllPositive) {
	const DenselySampledSpectrum& d65 = GetD65Illuminant();
	for (int lambda = 360; lambda <= 830; ++lambda)
		EXPECT_GE(d65(static_cast<float>(lambda)), 0.f) << "lambda=" << lambda;
}

TEST(D65Illuminant, SpectrumToXYZ_WhiteY) {
	// D65 has absolute spectral values ~100, not normalised to Y=1.
	// SpectrumToXYZ(D65) integrates against CIE Y, giving Y in the ~95-105 range.
	// Verify Y is positive and in the expected magnitude.
	const DenselySampledSpectrum& d65 = GetD65Illuminant();
	XYZ xyz = SpectrumToXYZ(d65);
	EXPECT_GT(xyz.Y, 90.f);
	EXPECT_LT(xyz.Y, 110.f);
	EXPECT_GT(xyz.X, 0.f);
	EXPECT_GT(xyz.Z, 0.f);
}

// ===========================================================================
// PixelSensor tests
// ===========================================================================

TEST(PixelSensor, CreateDefault_MatrixIsIdentity) {
	// CreateDefault() produces XYZ sensor with no white-balance (null illuminant),
	// so XYZFromSensorRGB should remain the identity matrix.
	PixelSensor sensor = PixelSensor::CreateDefault();
	const auto& M = sensor.XYZFromSensorRGB;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			EXPECT_NEAR(static_cast<float>(M[i][j]), (i == j) ? 1.f : 0.f, 1e-6f)
				<< "M[" << i << "][" << j << "]";
}

TEST(PixelSensor, XYZSensor_ToSensorRGB_WhiteSpectrum) {
	// For the XYZ sensor, all channels should be positive and finite
	PixelSensor sensor = PixelSensor::CreateDefault();
	auto lambda = SampledWavelengths<4>::SampleUniform(0.5f);
	SampledSpectrum<4> L(1.f);
	SensorRGB rgb = sensor.ToSensorRGB(L, lambda);
	EXPECT_GT(rgb.r, 0.f);
	EXPECT_GT(rgb.g, 0.f);
	EXPECT_GT(rgb.b, 0.f);
	EXPECT_TRUE(std::isfinite(rgb.r));
	EXPECT_TRUE(std::isfinite(rgb.g));
	EXPECT_TRUE(std::isfinite(rgb.b));
}

TEST(PixelSensor, XYZSensor_ImagingRatio) {
	// Changing imagingRatio should scale all three channels linearly.
	PixelSensor sensor1(&RGBColorSpace::sRGB(), nullptr, 1.f);
	PixelSensor sensor2(&RGBColorSpace::sRGB(), nullptr, 2.f);
	auto lambda = SampledWavelengths<4>::SampleUniform(0.3f);
	SampledSpectrum<4> L(0.5f);
	SensorRGB rgb1 = sensor1.ToSensorRGB(L, lambda);
	SensorRGB rgb2 = sensor2.ToSensorRGB(L, lambda);
	EXPECT_NEAR(rgb2.r, 2.f * rgb1.r, 1e-5f);
	EXPECT_NEAR(rgb2.g, 2.f * rgb1.g, 1e-5f);
	EXPECT_NEAR(rgb2.b, 2.f * rgb1.b, 1e-5f);
}

TEST(PixelSensor, XYZSensor_ZeroRadiance) {
	PixelSensor sensor = PixelSensor::CreateDefault();
	auto lambda = SampledWavelengths<4>::SampleUniform(0.7f);
	SampledSpectrum<4> L(0.f);
	SensorRGB rgb = sensor.ToSensorRGB(L, lambda);
	EXPECT_NEAR(rgb.r, 0.f, 1e-7f);
	EXPECT_NEAR(rgb.g, 0.f, 1e-7f);
	EXPECT_NEAR(rgb.b, 0.f, 1e-7f);
}

TEST(PixelSensor, SensorRGB_Indexing) {
	SensorRGB rgb{0.1f, 0.2f, 0.3f};
	EXPECT_FLOAT_EQ(rgb[0], 0.1f);
	EXPECT_FLOAT_EQ(rgb[1], 0.2f);
	EXPECT_FLOAT_EQ(rgb[2], 0.3f);
	rgb[1] = 0.5f;
	EXPECT_FLOAT_EQ(rgb.g, 0.5f);
}

TEST(PixelSensor, XYZSensor_WithD65Illuminant_MatrixNotIdentity) {
	// When a D65 sensor illuminant is supplied, the white-balance matrix
	// should deviate from identity (D65 != equal-energy, so correction is needed).
	const DenselySampledSpectrum& d65 = GetD65Illuminant();
	PixelSensor sensor(&RGBColorSpace::sRGB(), &d65, 1.f);
	const auto& M = sensor.XYZFromSensorRGB;
	// Check it's not the identity matrix
	bool isIdentity = true;
	for (int i = 0; i < 3 && isIdentity; ++i)
		for (int j = 0; j < 3 && isIdentity; ++j)
			if (std::abs(static_cast<float>(M[i][j]) - (i == j ? 1.f : 0.f)) > 1e-4f)
				isIdentity = false;
	EXPECT_FALSE(isIdentity) << "D65 white balance should modify the matrix";
}

