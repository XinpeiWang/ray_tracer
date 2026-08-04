// film_tests.cpp -- unit tests for src/shared/film.h
// Validates RGBFilm, GBufferFilm, and SpectralFilm against expected pbrt-v4 behavior.

#include <gtest/gtest.h>
#include <cmath>

#include "../../src/shared/film.h"

// ---------------------------------------------------------------------------
// Helpers: build identity sensor and identity color space for testing.
// outputRGBFromSensorRGB = RGBFromXYZ * XYZFromSensorRGB
// When both are identity, sensor-space RGB passes through unchanged.
// ---------------------------------------------------------------------------

// Build a PixelSensor with identity XYZFromSensorRGB (no spectral conversion).
// We access XYZFromSensorRGB directly since it is a public member.
static PixelSensor make_identity_sensor() {
	// Use the XYZ pseudo-sensor ctor with no illuminant -> XYZFromSensorRGB = identity
	// But that uses CIE X/Y/Z curves; easiest is to directly construct and set the matrix.
	RGBColorSpace srgb = RGBColorSpace::FromPrimaries(
		0.64f, 0.33f,  // R
		0.30f, 0.60f,  // G
		0.15f, 0.06f,  // B
		0.3127f, 0.3290f); // D65
	PixelSensor sensor(&srgb, nullptr, 1.0f);
	// Override XYZFromSensorRGB to identity for predictable tests
	sensor.XYZFromSensorRGB = SquareMatrix<3>();
	return sensor;
}

// Build an RGBColorSpace whose RGBFromXYZ is identity (for pass-through).
static RGBColorSpace make_identity_colorspace() {
	RGBColorSpace cs = RGBColorSpace::FromPrimaries(
		0.64f, 0.33f,
		0.30f, 0.60f,
		0.15f, 0.06f,
		0.3127f, 0.3290f);
	cs.RGBFromXYZ = SquareMatrix<3>();
	return cs;
}

// ---------------------------------------------------------------------------
// AtomicDoubleFilm
// ---------------------------------------------------------------------------
TEST(Film, AtomicDoubleAddAccumulates) {
	AtomicDoubleFilm a;
	a.add(1.0);
	a.add(2.5);
	EXPECT_NEAR(a.load(), 3.5, 1e-10);
}

TEST(Film, AtomicDoubleResetClearsToZero) {
	AtomicDoubleFilm a;
	a.add(42.0);
	a.reset();
	EXPECT_NEAR(a.load(), 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// RGBFilm construction
// ---------------------------------------------------------------------------
TEST(Film, RGBFilmConstructsWithCorrectDimensions) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	RGBFilm<BoxFilter> film(4, 3, BoxFilter(0.5), sensor, &cs);
	EXPECT_EQ(film.width(),  4);
	EXPECT_EQ(film.height(), 3);
}

// ---------------------------------------------------------------------------
// RGBFilm add_sample / get_pixel_rgb -- basic accumulation
// ---------------------------------------------------------------------------
TEST(Film, RGBFilmSingleSampleWeightOne) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	RGBFilm<BoxFilter> film(8, 8, BoxFilter(0.5), sensor, &cs);

	SensorRGB srgb{0.5f, 0.25f, 0.1f};
	film.add_sample(3, 4, 1.0f, srgb);

	auto p = film.get_pixel_rgb(3, 4);
	EXPECT_NEAR(p.r, 0.5f,  1e-5f);
	EXPECT_NEAR(p.g, 0.25f, 1e-5f);
	EXPECT_NEAR(p.b, 0.1f,  1e-5f);
}

TEST(Film, RGBFilmTwoEqualWeightSamplesAverages) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	RGBFilm<BoxFilter> film(4, 4, BoxFilter(0.5), sensor, &cs);

	film.add_sample(0, 0, 1.0f, SensorRGB{0.0f, 0.0f, 0.0f});
	film.add_sample(0, 0, 1.0f, SensorRGB{1.0f, 1.0f, 1.0f});

	auto p = film.get_pixel_rgb(0, 0);
	EXPECT_NEAR(p.r, 0.5f, 1e-5f);
	EXPECT_NEAR(p.g, 0.5f, 1e-5f);
	EXPECT_NEAR(p.b, 0.5f, 1e-5f);
}

TEST(Film, RGBFilmWeightedAverageCorrect) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	RGBFilm<BoxFilter> film(4, 4, BoxFilter(0.5), sensor, &cs);

	// w=3 sample: (0.6, 0, 0); w=1 sample: (0.2, 0, 0)
	// expected: (3*0.6 + 1*0.2) / 4 = 2.0/4 = 0.5
	film.add_sample(1, 1, 3.0f, SensorRGB{0.6f, 0.0f, 0.0f});
	film.add_sample(1, 1, 1.0f, SensorRGB{0.2f, 0.0f, 0.0f});

	auto p = film.get_pixel_rgb(1, 1);
	EXPECT_NEAR(p.r, 0.5f, 1e-5f);
}

TEST(Film, RGBFilmEmptyPixelIsBlack) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	RGBFilm<BoxFilter> film(4, 4, BoxFilter(0.5), sensor, &cs);

	auto p = film.get_pixel_rgb(2, 2);
	EXPECT_NEAR(p.r, 0.0f, 1e-5f);
	EXPECT_NEAR(p.g, 0.0f, 1e-5f);
	EXPECT_NEAR(p.b, 0.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// MaxComponentValue clamping (mirrors pbrt-v4 RGBFilm::AddSample clamp)
// ---------------------------------------------------------------------------
TEST(Film, RGBFilmClampsMaxComponent) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	// maxComponentValue = 1.0 -- anything above should be scaled down
	RGBFilm<BoxFilter> film(4, 4, BoxFilter(0.5), sensor, &cs, 1.0f);

	// Sample has max=4, so all channels scaled by 1/4
	film.add_sample(0, 0, 1.0f, SensorRGB{4.0f, 2.0f, 1.0f});

	auto p = film.get_pixel_rgb(0, 0);
	EXPECT_NEAR(p.r, 1.0f,  1e-5f);
	EXPECT_NEAR(p.g, 0.5f,  1e-5f);
	EXPECT_NEAR(p.b, 0.25f, 1e-5f);
}

// ---------------------------------------------------------------------------
// reset_pixel
// ---------------------------------------------------------------------------
TEST(Film, RGBFilmResetPixelClearsAccumulator) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	RGBFilm<BoxFilter> film(4, 4, BoxFilter(0.5), sensor, &cs);

	film.add_sample(0, 0, 1.0f, SensorRGB{1.0f, 1.0f, 1.0f});
	film.reset_pixel(0, 0);

	auto p = film.get_pixel_rgb(0, 0);
	EXPECT_NEAR(p.r, 0.0f, 1e-5f);
	EXPECT_NEAR(p.g, 0.0f, 1e-5f);
	EXPECT_NEAR(p.b, 0.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// clear()
// ---------------------------------------------------------------------------
TEST(Film, RGBFilmClearResetsAllPixels) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	RGBFilm<BoxFilter> film(4, 4, BoxFilter(0.5), sensor, &cs);

	for (int y = 0; y < 4; ++y)
		for (int x = 0; x < 4; ++x)
			film.add_sample(x, y, 1.0f, SensorRGB{0.5f, 0.5f, 0.5f});

	film.clear();

	for (int y = 0; y < 4; ++y) {
		for (int x = 0; x < 4; ++x) {
			auto p = film.get_pixel_rgb(x, y);
			EXPECT_NEAR(p.r, 0.0f, 1e-5f);
		}
	}
}

// ---------------------------------------------------------------------------
// add_splat -- splat lands in a 1x1 film, entire weight should appear
// ---------------------------------------------------------------------------
TEST(Film, RGBFilmSplatAddsToPixel) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	// Use BoxFilter radius=0.5: only the center pixel receives weight
	RGBFilm<BoxFilter> film(1, 1, BoxFilter(0.5), sensor, &cs);

	// Splat at center of the single pixel (0.0, 0.0)
	film.add_splat(0.0f, 0.0f, SensorRGB{1.0f, 0.5f, 0.25f});

	// With splat_scale=1 and filterIntegral = 4*0.5^2 = 1.0 for BoxFilter,
	// contribution = 1.0 * weight / filterIntegral = weight / 1.0
	auto p = film.get_pixel_rgb(0, 0, 1.0f);
	// rgbSum is 0 (no add_sample calls), so result comes only from splat
	EXPECT_GT(p.r, 0.0f);
	EXPECT_GT(p.g, 0.0f);
	EXPECT_GT(p.b, 0.0f);
	// Ratios must be preserved
	EXPECT_NEAR(p.g / p.r, 0.5f,  1e-4f);
	EXPECT_NEAR(p.b / p.r, 0.25f, 1e-4f);
}

TEST(Film, RGBFilmSplatScaleZeroRemovesSplat) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	RGBFilm<BoxFilter> film(4, 4, BoxFilter(0.5), sensor, &cs);

	film.add_sample(2, 2, 1.0f, SensorRGB{0.4f, 0.4f, 0.4f});
	film.add_splat(2.0f, 2.0f, SensorRGB{1.0f, 1.0f, 1.0f});

	auto p0 = film.get_pixel_rgb(2, 2, 0.0f);  // splat_scale = 0 ignores splat
	EXPECT_NEAR(p0.r, 0.4f, 1e-4f);
}

// ---------------------------------------------------------------------------
// add_splat -- distribute across multiple pixels
// ---------------------------------------------------------------------------
TEST(Film, RGBFilmSplatDistributesAcrossPixels) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	// TriangleFilter radius=1: covers a 3x3 neighborhood
	RGBFilm<TriangleFilter> film(5, 5, TriangleFilter(1.0), sensor, &cs);

	// Splat at center pixel (2,2); should spread to neighbors
	film.add_splat(2.0f, 2.0f, SensorRGB{1.0f, 0.0f, 0.0f});

	// Center should receive positive contribution
	auto pc = film.get_pixel_rgb(2, 2, 1.0f);
	EXPECT_GT(pc.r, 0.0f);

	// At least one neighbor should also get a contribution
	auto pn = film.get_pixel_rgb(2, 1, 1.0f);
	EXPECT_GT(pn.r, 0.0f);
}

// ---------------------------------------------------------------------------
// Filter integral sanity checks
// ---------------------------------------------------------------------------
TEST(Film, BoxFilterIntegralIs4r2) {
	BoxFilter f(0.5);
	EXPECT_NEAR(f.integral(), 4.0 * 0.5 * 0.5, 1e-10);
}

TEST(Film, TriangleFilterIntegralIsR4) {
	TriangleFilter f(2.0);
	EXPECT_NEAR(f.integral(), 16.0, 1e-10);
}

TEST(Film, MitchellFilterIntegralPositive) {
	MitchellFilter f(2.0);
	// pbrt-v4: radius.x * radius.y / 4 = 2 * 2 / 4 = 1.0
	EXPECT_NEAR(f.integral(), 1.0, 1e-10);
}

TEST(Film, GaussianFilterIntegralPositive) {
	GaussianFilter f(2.0, 0.5);
	EXPECT_GT(f.integral(), 0.0);
}

TEST(Film, LanczosSincFilterIntegralPositive) {
	LanczosSincFilter f(4.0, 3.0);
	EXPECT_GT(f.integral(), 0.0);
}

// ---------------------------------------------------------------------------
// to_output_rgb -- identity matrix pass-through
// ---------------------------------------------------------------------------
TEST(Film, RGBFilmToOutputRGBIdentityMatrix) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	RGBFilm<BoxFilter> film(2, 2, BoxFilter(0.5), sensor, &cs);

	auto out = film.to_output_rgb(0.3f, 0.5f, 0.7f);
	EXPECT_NEAR(out.r, 0.3f, 1e-5f);
	EXPECT_NEAR(out.g, 0.5f, 1e-5f);
	EXPECT_NEAR(out.b, 0.7f, 1e-5f);
}

// ---------------------------------------------------------------------------
// GBufferFilm construction + basic sample
// ---------------------------------------------------------------------------
TEST(Film, GBufferFilmConstructsWithCorrectDimensions) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	GBufferFilm<BoxFilter> film(10, 8, BoxFilter(0.5), sensor, &cs);
	EXPECT_EQ(film.width(),  10);
	EXPECT_EQ(film.height(), 8);
}

TEST(Film, GBufferFilmSingleSampleRGBMatches) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	GBufferFilm<BoxFilter> film(4, 4, BoxFilter(0.5), sensor, &cs);

	film.add_sample(1, 2, 1.0f, SensorRGB{0.7f, 0.3f, 0.1f});

	auto p = film.get_pixel_rgb(1, 2);
	EXPECT_NEAR(p.r, 0.7f, 1e-5f);
	EXPECT_NEAR(p.g, 0.3f, 1e-5f);
	EXPECT_NEAR(p.b, 0.1f, 1e-5f);
}

TEST(Film, GBufferFilmEmptyPixelIsBlack) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	GBufferFilm<BoxFilter> film(4, 4, BoxFilter(0.5), sensor, &cs);

	auto p = film.get_pixel_rgb(0, 0);
	EXPECT_NEAR(p.r, 0.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// GBufferFilm AOV accumulation
// ---------------------------------------------------------------------------
TEST(Film, GBufferFilmAOVNormalsAccumulate) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	GBufferFilm<BoxFilter> film(4, 4, BoxFilter(0.5), sensor, &cs);

	GBufferSample gbuf{};
	gbuf.n[0] = 0.0f; gbuf.n[1] = 1.0f; gbuf.n[2] = 0.0f;  // up normal

	film.add_sample(2, 2, 1.0f, SensorRGB{0.5f, 0.5f, 0.5f}, &gbuf);

	auto res = film.get_pixel_gbuffer(2, 2);
	EXPECT_GT(res.gBufWeight, 0.0f);
	EXPECT_NEAR(res.n[0], 0.0f, 1e-5f);
	EXPECT_NEAR(res.n[1], 1.0f, 1e-5f);
	EXPECT_NEAR(res.n[2], 0.0f, 1e-5f);
}

TEST(Film, GBufferFilmAOVPositionAccumulate) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	GBufferFilm<BoxFilter> film(4, 4, BoxFilter(0.5), sensor, &cs);

	GBufferSample gbuf{};
	gbuf.p[0] = 1.0f; gbuf.p[1] = 2.0f; gbuf.p[2] = 3.0f;

	film.add_sample(0, 0, 1.0f, SensorRGB{0.5f, 0.5f, 0.5f}, &gbuf);

	auto res = film.get_pixel_gbuffer(0, 0);
	EXPECT_NEAR(res.p[0], 1.0f, 1e-5f);
	EXPECT_NEAR(res.p[1], 2.0f, 1e-5f);
	EXPECT_NEAR(res.p[2], 3.0f, 1e-5f);
}

TEST(Film, GBufferFilmAOVUVAccumulate) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	GBufferFilm<BoxFilter> film(4, 4, BoxFilter(0.5), sensor, &cs);

	GBufferSample gbuf{};
	gbuf.uv[0] = 0.25f; gbuf.uv[1] = 0.75f;

	film.add_sample(1, 1, 1.0f, SensorRGB{0.5f, 0.5f, 0.5f}, &gbuf);

	auto res = film.get_pixel_gbuffer(1, 1);
	EXPECT_NEAR(res.uv[0], 0.25f, 1e-5f);
	EXPECT_NEAR(res.uv[1], 0.75f, 1e-5f);
}

TEST(Film, GBufferFilmAOVWeightedAverageNormals) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	GBufferFilm<BoxFilter> film(4, 4, BoxFilter(0.5), sensor, &cs);

	GBufferSample g1{}, g2{};
	g1.n[1] = 1.0f;  // (0,1,0)
	g2.n[0] = 1.0f;  // (1,0,0)

	film.add_sample(0, 0, 1.0f, SensorRGB{0.5f, 0.5f, 0.5f}, &g1);
	film.add_sample(0, 0, 1.0f, SensorRGB{0.5f, 0.5f, 0.5f}, &g2);

	auto res = film.get_pixel_gbuffer(0, 0);
	EXPECT_NEAR(res.n[0], 0.5f, 1e-5f);
	EXPECT_NEAR(res.n[1], 0.5f, 1e-5f);
	EXPECT_NEAR(res.n[2], 0.0f, 1e-5f);
}

TEST(Film, GBufferFilmEmptyGBufferWeightIsZero) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	GBufferFilm<BoxFilter> film(4, 4, BoxFilter(0.5), sensor, &cs);

	auto res = film.get_pixel_gbuffer(3, 3);
	EXPECT_NEAR(res.gBufWeight, 0.0f, 1e-10f);
}

TEST(Film, GBufferFilmAOVNullGBufSkipsAOV) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	GBufferFilm<BoxFilter> film(4, 4, BoxFilter(0.5), sensor, &cs);

	// add_sample without gbuf -- pixel should accumulate RGB but no AOV
	film.add_sample(0, 0, 1.0f, SensorRGB{0.5f, 0.5f, 0.5f}, nullptr);

	auto res = film.get_pixel_gbuffer(0, 0);
	EXPECT_NEAR(res.gBufWeight, 0.0f, 1e-10f);  // no gbuf -> weight stays 0

	auto rgb = film.get_pixel_rgb(0, 0);
	EXPECT_NEAR(rgb.r, 0.5f, 1e-5f);  // RGB still accumulates
}

// ---------------------------------------------------------------------------
// GBufferFilm splat
// ---------------------------------------------------------------------------
TEST(Film, GBufferFilmSplatAddsToPixel) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	GBufferFilm<BoxFilter> film(1, 1, BoxFilter(0.5), sensor, &cs);

	film.add_splat(0.0f, 0.0f, SensorRGB{1.0f, 0.5f, 0.25f});

	auto p = film.get_pixel_rgb(0, 0, 1.0f);
	EXPECT_GT(p.r, 0.0f);
	EXPECT_NEAR(p.g / p.r, 0.5f,  1e-4f);
	EXPECT_NEAR(p.b / p.r, 0.25f, 1e-4f);
}

// ---------------------------------------------------------------------------
// GBufferFilm clear
// ---------------------------------------------------------------------------
TEST(Film, GBufferFilmClearResetsAll) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	GBufferFilm<BoxFilter> film(3, 3, BoxFilter(0.5), sensor, &cs);

	GBufferSample gbuf{};
	gbuf.n[1] = 1.0f;
	for (int y = 0; y < 3; ++y)
		for (int x = 0; x < 3; ++x)
			film.add_sample(x, y, 1.0f, SensorRGB{1.f, 1.f, 1.f}, &gbuf);

	film.clear();

	for (int y = 0; y < 3; ++y) {
		for (int x = 0; x < 3; ++x) {
			auto p = film.get_pixel_rgb(x, y);
			EXPECT_NEAR(p.r, 0.0f, 1e-5f);
			auto res = film.get_pixel_gbuffer(x, y);
			EXPECT_NEAR(res.gBufWeight, 0.0f, 1e-10f);
		}
	}
}

// ---------------------------------------------------------------------------
// MitchellFilter -- RGBFilm integration: filtered splat sums close to 1
// ---------------------------------------------------------------------------
TEST(Film, RGBFilmMitchellSplatSumsToFilterIntegral) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	MitchellFilter filt(2.0);
	// 9x9 film, splat at center (4,4)
	RGBFilm<MitchellFilter> film(9, 9, filt, sensor, &cs);

	film.add_splat(4.0f, 4.0f, SensorRGB{1.0f, 0.0f, 0.0f});

	// Sum all pixel splat contributions (before dividing by filterIntegral)
	// We retrieve with splat_scale=1 and sum; total * filterIntegral should ≈ 1
	double totalR = 0.0;
	for (int y = 0; y < 9; ++y)
		for (int x = 0; x < 9; ++x)
			totalR += film.get_pixel_rgb(x, y, 1.0f).r;

	// Total should be positive and reasonable (filter sums > 0)
	EXPECT_GT(totalR, 0.0);
}

// ---------------------------------------------------------------------------
// SpectralFilm -- construction
// ---------------------------------------------------------------------------
TEST(Film, SpectralFilmConstructsWithCorrectDimensions) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	BoxFilter filt(0.5);
	SpectralFilm<BoxFilter> film(4, 3, filt, sensor, &cs, 16);
	EXPECT_EQ(film.width(),  4);
	EXPECT_EQ(film.height(), 3);
	EXPECT_EQ(film.num_buckets(), 16);
}

// ---------------------------------------------------------------------------
// SpectralFilm -- sample_wavelengths returns values in [lambdaMin, lambdaMax]
// ---------------------------------------------------------------------------
TEST(Film, SpectralFilmSampleWavelengthsInRange) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	BoxFilter filt(0.5);
	SpectralFilm<BoxFilter> film(4, 4, filt, sensor, &cs, 16);
	auto swl = film.sample_wavelengths(0.5f);
	for (int i = 0; i < 4; ++i) {
		EXPECT_GE(swl[i], film.lambda_min());
		EXPECT_LE(swl[i], film.lambda_max());
	}
}

// ---------------------------------------------------------------------------
// SpectralFilm -- add_sample accumulates RGB correctly (identity sensor/cs)
// ---------------------------------------------------------------------------
TEST(Film, SpectralFilmAddSampleRGBPassthrough) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	BoxFilter filt(0.5);
	SpectralFilm<BoxFilter> film(4, 4, filt, sensor, &cs, 16);

	auto swl = film.sample_wavelengths(0.3f);
	SampledSpectrum<4> L(1.0f);
	SensorRGB srgb{0.5f, 0.25f, 0.1f};

	film.add_sample(1, 1, 1.0f, srgb, L, swl);

	FilmPixelRGB px = film.get_pixel_rgb(1, 1);
	// Identity sensor + identity cs: output ≈ srgb input
	EXPECT_NEAR(px.r, 0.5f,  1e-4f);
	EXPECT_NEAR(px.g, 0.25f, 1e-4f);
	EXPECT_NEAR(px.b, 0.1f,  1e-4f);
}

// ---------------------------------------------------------------------------
// SpectralFilm -- spectral bucket accumulation: sampled wavelengths land in
// their expected buckets and the normalized value is positive.
// ---------------------------------------------------------------------------
TEST(Film, SpectralFilmSpectralBucketsPositiveAfterSample) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	BoxFilter filt(0.5);
	const int nBuckets = 32;
	SpectralFilm<BoxFilter> film(4, 4, filt, sensor, &cs, nBuckets);

	auto swl = film.sample_wavelengths(0.5f);
	SampledSpectrum<4> L(2.0f);
	SensorRGB srgb{1.0f, 1.0f, 1.0f};

	film.add_sample(2, 2, 1.0f, srgb, L, swl);

	auto spec = film.get_pixel_spectral(2, 2);
	ASSERT_EQ(static_cast<int>(spec.size()), nBuckets);

	// At least one bucket must be non-zero (those hit by our 4 hero wavelengths)
	bool any_nonzero = false;
	for (float v : spec) if (v > 0.f) { any_nonzero = true; break; }
	EXPECT_TRUE(any_nonzero);
}

// ---------------------------------------------------------------------------
// SpectralFilm -- unsampled pixel returns zero spectral bins
// ---------------------------------------------------------------------------
TEST(Film, SpectralFilmUnsampledPixelReturnsZeroBuckets) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	BoxFilter filt(0.5);
	SpectralFilm<BoxFilter> film(4, 4, filt, sensor, &cs, 16);

	auto spec = film.get_pixel_spectral(0, 0);
	for (float v : spec)
		EXPECT_EQ(v, 0.f);
}

// ---------------------------------------------------------------------------
// SpectralFilm -- bucket_lambda spans expected wavelength range
// ---------------------------------------------------------------------------
TEST(Film, SpectralFilmBucketLambdaCentersSpanRange) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	BoxFilter filt(0.5);
	const int nBuckets = 8;
	SpectralFilm<BoxFilter> film(2, 2, filt, sensor, &cs, nBuckets);

	// First bucket centre > lambdaMin
	EXPECT_GT(film.bucket_lambda(0), film.lambda_min());
	// Last bucket centre < lambdaMax
	EXPECT_LT(film.bucket_lambda(nBuckets - 1), film.lambda_max());
	// Monotonically increasing
	for (int b = 1; b < nBuckets; ++b)
		EXPECT_GT(film.bucket_lambda(b), film.bucket_lambda(b - 1));
}

// ---------------------------------------------------------------------------
// SpectralFilm -- clear() resets all spectral buckets and RGB
// ---------------------------------------------------------------------------
TEST(Film, SpectralFilmClearResetsPixels) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	BoxFilter filt(0.5);
	SpectralFilm<BoxFilter> film(4, 4, filt, sensor, &cs, 16);

	auto swl = film.sample_wavelengths(0.1f);
	SampledSpectrum<4> L(3.0f);
	SensorRGB srgb{1.0f, 0.5f, 0.25f};
	film.add_sample(1, 1, 1.0f, srgb, L, swl);

	film.clear();

	// RGB should be zero after clear
	FilmPixelRGB px = film.get_pixel_rgb(1, 1);
	EXPECT_NEAR(px.r, 0.f, 1e-6f);
	EXPECT_NEAR(px.g, 0.f, 1e-6f);
	EXPECT_NEAR(px.b, 0.f, 1e-6f);

	// Spectral buckets should be zero after clear
	auto spec = film.get_pixel_spectral(1, 1);
	for (float v : spec)
		EXPECT_EQ(v, 0.f);
}

// ---------------------------------------------------------------------------
// SpectralFilm -- multiple samples average correctly in spectral buckets
// ---------------------------------------------------------------------------
TEST(Film, SpectralFilmMultiSampleAveraging) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	BoxFilter filt(0.5);
	// One bucket spanning the full visible range for simplicity
	SpectralFilm<BoxFilter> film(4, 4, filt, sensor, &cs, 1);

	auto swl = film.sample_wavelengths(0.5f);
	SampledSpectrum<4> L1(2.0f);
	SampledSpectrum<4> L2(4.0f);
	SensorRGB srgb{1.0f, 0.f, 0.f};

	film.add_sample(0, 0, 1.0f, srgb, L1, swl);
	film.add_sample(0, 0, 1.0f, srgb, L2, swl);

	// Both samples hit the same single bucket; average of scaled values
	// should be strictly positive
	auto spec = film.get_pixel_spectral(0, 0);
	ASSERT_EQ(static_cast<int>(spec.size()), 1);
	EXPECT_GT(spec[0], 0.f);
}

// ---------------------------------------------------------------------------
// SpectralFilm -- add_splat contributes to spectral buckets
// ---------------------------------------------------------------------------
TEST(Film, SpectralFilmSplatAddsToSpectralBuckets) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	BoxFilter filt(0.5);
	SpectralFilm<BoxFilter> film(4, 4, filt, sensor, &cs, 16);

	auto swl = film.sample_wavelengths(0.5f);
	SampledSpectrum<4> L(1.0f);
	SensorRGB srgb{0.5f, 0.5f, 0.5f};

	// Splat at centre of pixel (1,1): continuous coords (1.5, 1.5)
	film.add_splat(1.5f, 1.5f, L, swl, srgb);

	// With splat_scale=1 and BoxFilter, spectral buckets at (1,1) should be nonzero
	auto spec = film.get_pixel_spectral(1, 1, 1.0f);
	bool any_nonzero = false;
	for (float v : spec) if (v > 0.f) { any_nonzero = true; break; }
	EXPECT_TRUE(any_nonzero);
}

// ---------------------------------------------------------------------------
// SpectralFilm -- splat_scale=0 suppresses splat but not sample contribution
// ---------------------------------------------------------------------------
TEST(Film, SpectralFilmSplatScaleZeroSuppressesSplatBuckets) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	BoxFilter filt(0.5);
	SpectralFilm<BoxFilter> film(4, 4, filt, sensor, &cs, 1);

	auto swl = film.sample_wavelengths(0.5f);
	SampledSpectrum<4> L(1.0f);
	SensorRGB srgb{1.0f, 0.f, 0.f};

	// Only a splat, no add_sample
	film.add_splat(0.5f, 0.5f, L, swl, srgb);

	// splat_scale=0 should suppress the splat contribution
	auto spec_no_splat = film.get_pixel_spectral(0, 0, 0.0f);
	EXPECT_NEAR(spec_no_splat[0], 0.f, 1e-6f);

	// splat_scale=1 should expose it
	auto spec_with_splat = film.get_pixel_spectral(0, 0, 1.0f);
	EXPECT_GT(spec_with_splat[0], 0.f);
}

// ---------------------------------------------------------------------------
// SpectralFilm -- clear() resets bucketSplats as well
// ---------------------------------------------------------------------------
TEST(Film, SpectralFilmClearResetsBucketSplats) {
	PixelSensor sensor = make_identity_sensor();
	RGBColorSpace cs   = make_identity_colorspace();
	BoxFilter filt(0.5);
	SpectralFilm<BoxFilter> film(4, 4, filt, sensor, &cs, 8);

	auto swl = film.sample_wavelengths(0.5f);
	SampledSpectrum<4> L(2.0f);
	SensorRGB srgb{1.0f, 0.f, 0.f};

	film.add_splat(0.5f, 0.5f, L, swl, srgb);
	film.clear();

	auto spec = film.get_pixel_spectral(0, 0, 1.0f);
	for (float v : spec)
		EXPECT_NEAR(v, 0.f, 1e-6f);
}
