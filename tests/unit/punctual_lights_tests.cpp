// punctual_lights_tests.cpp
// Regression tests for PointLightData, SpotLightData, DistantLightData.
// Focus on the newly added SpotLightData::sample_le and pdf_le methods
// that mirror pbrt-v4 SpotLight::SampleLe / SpotLight::PDF_Le.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../../src/shared/punctual_lights.h"

static constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static double length3(double x, double y, double z) {
	return std::sqrt(x*x + y*y + z*z);
}

static double dot3(double ax, double ay, double az,
				   double bx, double by, double bz) {
	return ax*bx + ay*by + az*bz;
}

// Make a SpotLight aimed along +Z with given full/falloff half-angles (degrees).
static SpotLightData<double> make_spot(double full_deg, double falloff_deg,
										double scale = 1.0) {
	SpotLightData<double> s;
	s.pos_x = 0; s.pos_y = 0; s.pos_z = 0;
	s.dir_x = 0; s.dir_y = 0; s.dir_z = 1;  // +Z cone axis
	s.ir = 1; s.ig = 1; s.ib = 1;
	s.scale = scale;
	s.cos_falloff_start = std::cos(full_deg     * kPi / 180.0);
	s.cos_falloff_end   = std::cos(falloff_deg  * kPi / 180.0);
	return s;
}

// Make a SpotLight aimed along an arbitrary axis.
static SpotLightData<double> make_spot_dir(double dx, double dy, double dz,
											double full_deg, double falloff_deg) {
	double len = length3(dx, dy, dz);
	SpotLightData<double> s;
	s.pos_x = 0; s.pos_y = 0; s.pos_z = 0;
	s.dir_x = dx/len; s.dir_y = dy/len; s.dir_z = dz/len;
	s.ir = 1; s.ig = 1; s.ib = 1;
	s.scale = 1.0;
	s.cos_falloff_start = std::cos(full_deg    * kPi / 180.0);
	s.cos_falloff_end   = std::cos(falloff_deg * kPi / 180.0);
	return s;
}

// ---------------------------------------------------------------------------
// cone_frame_basis tests
// ---------------------------------------------------------------------------

TEST(SpotLightSampleLe, ConeFrameBasisOrthonormal) {
	auto s = make_spot(30.0, 45.0);
	double tx, ty, tz, bx, by, bz;
	s.cone_frame_basis(tx, ty, tz, bx, by, bz);

	// All three vectors should be unit length
	EXPECT_NEAR(length3(tx, ty, tz), 1.0, 1e-9);
	EXPECT_NEAR(length3(bx, by, bz), 1.0, 1e-9);
	EXPECT_NEAR(length3(s.dir_x, s.dir_y, s.dir_z), 1.0, 1e-9);

	// Mutually orthogonal
	EXPECT_NEAR(dot3(tx,ty,tz, bx,by,bz), 0.0, 1e-9);
	EXPECT_NEAR(dot3(tx,ty,tz, s.dir_x,s.dir_y,s.dir_z), 0.0, 1e-9);
	EXPECT_NEAR(dot3(bx,by,bz, s.dir_x,s.dir_y,s.dir_z), 0.0, 1e-9);
}

TEST(SpotLightSampleLe, ConeFrameBasisArbitraryAxis) {
	auto s = make_spot_dir(1.0, 2.0, 3.0, 20.0, 40.0);
	double tx, ty, tz, bx, by, bz;
	s.cone_frame_basis(tx, ty, tz, bx, by, bz);
	EXPECT_NEAR(length3(tx, ty, tz), 1.0, 1e-9);
	EXPECT_NEAR(length3(bx, by, bz), 1.0, 1e-9);
	EXPECT_NEAR(dot3(tx,ty,tz, bx,by,bz), 0.0, 1e-9);
	EXPECT_NEAR(dot3(tx,ty,tz, s.dir_x,s.dir_y,s.dir_z), 0.0, 1e-9);
}

// ---------------------------------------------------------------------------
// sample_le basic tests
// ---------------------------------------------------------------------------

TEST(SpotLightSampleLe, DirectionIsNormalized) {
	auto s = make_spot(30.0, 45.0);
	double wx, wy, wz, pdf;
	s.sample_le(0.5, 0.3, 0.7, wx, wy, wz, pdf);
	EXPECT_NEAR(length3(wx, wy, wz), 1.0, 1e-9);
}

TEST(SpotLightSampleLe, PdfPositive) {
	auto s = make_spot(30.0, 45.0);
	double wx, wy, wz, pdf;
	s.sample_le(0.5, 0.5, 0.5, wx, wy, wz, pdf);
	EXPECT_GT(pdf, 0.0);
}

TEST(SpotLightSampleLe, DirectionInsideCone) {
	// All sampled directions should be within the outer cone
	auto s = make_spot(20.0, 40.0);
	double cf_end = std::cos(40.0 * kPi / 180.0);
	for (int i = 0; i < 20; ++i) {
		double ru  = (i + 0.5) / 20.0;
		double rv0 = ((i * 7 + 1) % 20 + 0.5) / 20.0;
		double rv1 = ((i * 13 + 3) % 20 + 0.5) / 20.0;
		double wx, wy, wz, pdf;
		s.sample_le(ru, rv0, rv1, wx, wy, wz, pdf);
		double ct = dot3(wx, wy, wz, s.dir_x, s.dir_y, s.dir_z);
		EXPECT_GE(ct, cf_end - 1e-6) << "i=" << i;
	}
}

TEST(SpotLightSampleLe, ArbitraryAxisDirectionInsideCone) {
	auto s = make_spot_dir(1.0, 1.0, 1.0, 20.0, 40.0);
	double cf_end = std::cos(40.0 * kPi / 180.0);
	for (int i = 0; i < 16; ++i) {
		double ru  = (i + 0.5) / 16.0;
		double rv0 = ((i * 5 + 2) % 16 + 0.5) / 16.0;
		double rv1 = ((i * 11 + 7) % 16 + 0.5) / 16.0;
		double wx, wy, wz, pdf;
		s.sample_le(ru, rv0, rv1, wx, wy, wz, pdf);
		double ct = dot3(wx, wy, wz, s.dir_x, s.dir_y, s.dir_z);
		EXPECT_GE(ct, cf_end - 1e-6) << "i=" << i;
	}
}

// ---------------------------------------------------------------------------
// pdf_le tests
// ---------------------------------------------------------------------------

TEST(SpotLightSampleLe, PdfLeZeroOutsideCone) {
	auto s = make_spot(20.0, 30.0);
	// Direction perpendicular to cone axis -- completely outside
	EXPECT_DOUBLE_EQ(s.pdf_le(1.0, 0.0, 0.0), 0.0);
	// Direction anti-parallel to cone
	EXPECT_DOUBLE_EQ(s.pdf_le(0.0, 0.0, -1.0), 0.0);
}

TEST(SpotLightSampleLe, PdfLePositiveInsideCone) {
	auto s = make_spot(20.0, 40.0);
	// Straight along cone axis -- in inner cone
	double pdf = s.pdf_le(0.0, 0.0, 1.0);
	EXPECT_GT(pdf, 0.0);
}

TEST(SpotLightSampleLe, PdfLePositiveInFalloff) {
	// Direction at exactly the midpoint of the falloff region
	auto s = make_spot(20.0, 40.0);
	double mid_angle = (20.0 + 40.0) / 2.0 * kPi / 180.0;
	double pdf = s.pdf_le(std::sin(mid_angle), 0.0, std::cos(mid_angle));
	EXPECT_GT(pdf, 0.0);
}

// ---------------------------------------------------------------------------
// Consistency: pdf_le(sample_le direction) == pdf reported by sample_le
// ---------------------------------------------------------------------------

TEST(SpotLightSampleLe, PdfLeMatchesSampleLe_InnerCone) {
	// ru near 0 -> inner cone branch
	auto s = make_spot(30.0, 50.0);
	double wx, wy, wz, pdf_sample;
	s.sample_le(0.01, 0.4, 0.7, wx, wy, wz, pdf_sample);
	double pdf_eval = s.pdf_le(wx, wy, wz);
	EXPECT_NEAR(pdf_eval, pdf_sample, pdf_sample * 0.01 + 1e-12);
}

TEST(SpotLightSampleLe, PdfLeMatchesSampleLe_Falloff) {
	// ru near 1 -> falloff branch (assuming p1 large enough)
	auto s = make_spot(20.0, 60.0);
	double wx, wy, wz, pdf_sample;
	s.sample_le(0.99, 0.4, 0.7, wx, wy, wz, pdf_sample);
	double pdf_eval = s.pdf_le(wx, wy, wz);
	EXPECT_NEAR(pdf_eval, pdf_sample, pdf_sample * 0.01 + 1e-12);
}

TEST(SpotLightSampleLe, PdfLeMatchesSampleLe_Sweep) {
	// Sweep many samples; each pdf_le should match pdf_sample
	auto s = make_spot(25.0, 50.0);
	for (int i = 0; i < 30; ++i) {
		double ru  = (i + 0.5) / 30.0;
		double rv0 = ((i * 7 + 3) % 30 + 0.5) / 30.0;
		double rv1 = ((i * 11 + 5) % 30 + 0.5) / 30.0;
		double wx, wy, wz, pdf_sample;
		s.sample_le(ru, rv0, rv1, wx, wy, wz, pdf_sample);
		double pdf_eval = s.pdf_le(wx, wy, wz);
		EXPECT_NEAR(pdf_eval, pdf_sample, pdf_sample * 0.02 + 1e-12)
			<< "i=" << i;
	}
}

// ---------------------------------------------------------------------------
// Normalisation: integral of pdf should be ~1 (Monte Carlo estimate)
// ---------------------------------------------------------------------------

TEST(SpotLightSampleLe, PdfIntegratesApprox1) {
	// Estimates integral(pdf_le dw) via importance sampling.
	// If pdf_le and sample_le agree: E[pdf_le/pdf_sample] = 1.
	auto s = make_spot(30.0, 60.0);
	const int N = 2000;
	double sum = 0.0;
	int valid = 0;
	for (int i = 0; i < N; ++i) {
		double ru  = (i + 0.5) / N;
		double rv0 = ((i * 7  + 3) % N + 0.5) / N;
		double rv1 = ((i * 13 + 7) % N + 0.5) / N;
		double wx, wy, wz, pdf_sample;
		s.sample_le(ru, rv0, rv1, wx, wy, wz, pdf_sample);
		if (pdf_sample > 1e-12) {
			sum += s.pdf_le(wx, wy, wz) / pdf_sample;
			++valid;
		}
	}
	if (valid > 0)
		EXPECT_NEAR(sum / valid, 1.0, 0.05) << "valid=" << valid;
}

// ---------------------------------------------------------------------------
// Existing SpotLightData sanity: power and pdf_Li unchanged
// ---------------------------------------------------------------------------

TEST(SpotLightData, PowerPositive) {
	auto s = make_spot(20.0, 40.0);
	EXPECT_GT(s.power(), 0.0);
}

TEST(SpotLightData, PdfLiAlwaysZero) {
	auto s = make_spot(20.0, 40.0);
	EXPECT_DOUBLE_EQ(s.pdf_Li(), 0.0);
}
