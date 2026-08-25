/**
 * @file scatter_record_transmission_tests.cpp
 * @brief Directly verifies scatter_record::eta/is_transmission are
 * correctly populated by scatter(), for skip_pdf materials -- catches the
 * exact class of bug fixed in commit b8d355d.
 *
 * The end-to-end brightness tests in skip_pdf_material_brightness_tests.cpp
 * exercise the real render path but turned out NOT to reliably reproduce
 * the original bug on demand: it depended on whatever value happened to be
 * sitting in scatter_record's stack slot at the time (genuinely undefined
 * behavior prior to the fix), which varies by call site, compiler, and
 * optimization level -- rerunning the pre-fix code against those tests
 * showed most runs passing by pure luck. This test instead checks
 * scatter()'s output directly and deterministically: does rough_dielectric
 * actually assign a real transmission event, not just leave
 * scatter_record's now-safe class-level defaults untouched?
 */

#include <gtest/gtest.h>
#include "rtweekend.h"
#include "hittable.h"
#include "sphere.h"
#include "material.h"

// A rough_dielectric sphere hit at near-normal incidence should transmit
// on most samples (Fresnel reflectance is low near normal incidence for
// IOR 1.5) -- across many trials, at least some calls must report a real
// transmission event with eta != 1.0, not just the struct's default.
TEST(ScatterRecordTransmission, RoughDielectricReportsRealTransmissionEvents) {
	auto mat = std::make_shared<rough_dielectric>(1.5, 0.2);
	sphere sph(point3(0, 0, 0), 1.0, mat);

	// Off-axis, not aimed at the sphere's dead center: a ray aimed exactly
	// at a sphere's center always hits at exactly-normal incidence
	// (cos_i == 1), a degenerate case for the underlying GGX VNDF sampling
	// math (unrelated to eta/is_transmission) that made scatter() itself
	// return false on the very first call, regardless of which code path
	// (buggy or fixed) was under test.
	ray r_in(point3(0, 0, -5), unit_vector(vec3(0.3, 0.2, 5)));
	hit_record rec;
	ASSERT_TRUE(sph.hit(r_in, interval(0.001, infinity), rec));

	// scatter() occasionally returns false on its own (a rejected/backfacing
	// VNDF microfacet sample -- normal, expected microfacet-BSDF behavior,
	// not related to eta/is_transmission), so those calls are skipped
	// rather than asserted on.
	int transmission_count = 0;
	for (int i = 0; i < 200; ++i) {
		scatter_record srec;
		if (!mat->scatter(r_in, rec, srec)) continue;
		if (srec.is_transmission) {
			EXPECT_NE(srec.eta, 1.0) << "is_transmission=true but eta wasn't actually set";
			++transmission_count;
		}
	}
	EXPECT_GT(transmission_count, 0)
		<< "rough_dielectric never reported a transmission event across 200 samples at "
		   "near-normal incidence (low Fresnel reflectance) -- scatter() may have stopped "
		   "propagating RoughDielectricBxDF::sample_local's res.is_transmission/res.eta";
}

// Pure-reflective delta materials should never report a transmission event
// -- confirms the safe class-level defaults (scatter_record::is_transmission
// = false, eta = 1.0, added in commit b8d355d) match what these materials
// actually produce, not just paper over an absent field.
TEST(ScatterRecordTransmission, MetalNeverReportsTransmission) {
	auto mat = std::make_shared<metal>(color(0.8, 0.8, 0.8), 0.0);
	sphere sph(point3(0, 0, 0), 1.0, mat);

	// Off-axis, not aimed at the sphere's dead center: a ray aimed exactly
	// at a sphere's center always hits at exactly-normal incidence
	// (cos_i == 1), a degenerate case for the underlying GGX VNDF sampling
	// math (unrelated to eta/is_transmission) that made scatter() itself
	// return false on the very first call, regardless of which code path
	// (buggy or fixed) was under test.
	ray r_in(point3(0, 0, -5), unit_vector(vec3(0.3, 0.2, 5)));
	hit_record rec;
	ASSERT_TRUE(sph.hit(r_in, interval(0.001, infinity), rec));

	scatter_record srec;
	ASSERT_TRUE(mat->scatter(r_in, rec, srec));
	EXPECT_FALSE(srec.is_transmission);
	EXPECT_DOUBLE_EQ(srec.eta, 1.0);
}

// dielectric's new dispersive constructor/scatter_dispersive() (added for
// --spectral chromatic-dispersion support) - confirms the class-level
// integration point end to end: a dispersive instance actually reports a
// DIFFERENT eta at two different hero wavelengths, while a plain
// (non-dispersive) instance ignores lambda_nm entirely and always reports
// the flat constructor eta - matching is_dispersive()'s own contract.
TEST(ScatterRecordTransmission, DispersiveDielectricEtaVariesByWavelength) {
	// Same (eta_d=1.52, Abbe=59) crown-glass-like values as fresnel_tests.cpp's
	// CauchyEtaNormalDispersion, constructed here via dielectric's own
	// factory instead of the raw formula.
	auto mat_ptr = dielectric::make_dispersive(1.52, 59.0);
	dielectric& mat = *mat_ptr;
	EXPECT_TRUE(mat.is_dispersive());

	sphere sph(point3(0, 0, 0), 1.0, mat_ptr);
	ray r_in(point3(0, 0, -5), unit_vector(vec3(0.3, 0.2, 5)));
	hit_record rec;
	ASSERT_TRUE(sph.hit(r_in, interval(0.001, infinity), rec));

	// Pin the RNG-driven reflect/transmit choice by trying enough samples
	// that at least one of each wavelength's calls actually transmits
	// (rejection loop, same reasoning as this file's own rough_dielectric
	// test above).
	double eta_red = 0.0, eta_violet = 0.0;
	for (int i = 0; i < 50 && eta_red == 0.0; ++i) {
		scatter_record srec;
		if (mat.scatter_dispersive(r_in, rec, srec, 650.f) && srec.is_transmission)
			eta_red = srec.eta;
	}
	for (int i = 0; i < 50 && eta_violet == 0.0; ++i) {
		scatter_record srec;
		if (mat.scatter_dispersive(r_in, rec, srec, 450.f) && srec.is_transmission)
			eta_violet = srec.eta;
	}
	ASSERT_NE(eta_red, 0.0) << "never got a transmission event at 650nm across 50 samples";
	ASSERT_NE(eta_violet, 0.0) << "never got a transmission event at 450nm across 50 samples";
	// DielectricBxDF::sample() reports srec.eta as eta_i/eta_t for an
	// entering transmission (confirmed empirically: dielectric(1.5)'s own
	// ordinary scatter() reports eta=1/1.5, not 1.5, for this same
	// entering-ray setup) - the INVERSE of the material's own IOR. Higher
	// true IOR (violet) therefore means a SMALLER reported srec.eta here.
	// The physically-precise, convention-independent check (does CauchyEta
	// itself increase as wavelength decreases) is fresnel_tests.cpp's own
	// CauchyEtaNormalDispersion - this test only needs to prove
	// scatter_dispersive() actually threads a different lambda_nm into a
	// different eta at all.
	EXPECT_NE(eta_red, eta_violet) << "dispersive dielectric reported the same eta at 650nm and 450nm";
	EXPECT_GT(eta_red, eta_violet) << "red's (smaller true IOR) reported eta should be larger than "
	                                   "violet's, given DielectricBxDF's eta_i/eta_t convention";
}

TEST(ScatterRecordTransmission, NonDispersiveDielectricIgnoresWavelength) {
	dielectric mat(1.5);  // plain flat-IOR constructor - not dispersive
	EXPECT_FALSE(mat.is_dispersive());

	sphere sph(point3(0, 0, 0), 1.0, std::make_shared<dielectric>(mat));
	ray r_in(point3(0, 0, -5), unit_vector(vec3(0.3, 0.2, 5)));
	hit_record rec;
	ASSERT_TRUE(sph.hit(r_in, interval(0.001, infinity), rec));

	// Reference value: whatever the ORDINARY (non-spectral) scatter() path
	// reports for this exact setup - avoids hardcoding an assumption about
	// DielectricBxDF's eta convention (entering vs exiting, eta vs 1/eta).
	double eta_reference = 0.0;
	for (int i = 0; i < 50 && eta_reference == 0.0; ++i) {
		scatter_record srec;
		if (mat.scatter(r_in, rec, srec) && srec.is_transmission)
			eta_reference = srec.eta;
	}
	ASSERT_NE(eta_reference, 0.0) << "never got a transmission event via scatter() across 50 samples";

	double eta_red = 0.0, eta_violet = 0.0;
	for (int i = 0; i < 50 && eta_red == 0.0; ++i) {
		scatter_record srec;
		if (mat.scatter_dispersive(r_in, rec, srec, 650.f) && srec.is_transmission)
			eta_red = srec.eta;
	}
	for (int i = 0; i < 50 && eta_violet == 0.0; ++i) {
		scatter_record srec;
		if (mat.scatter_dispersive(r_in, rec, srec, 450.f) && srec.is_transmission)
			eta_violet = srec.eta;
	}
	ASSERT_NE(eta_red, 0.0);
	ASSERT_NE(eta_violet, 0.0);
	EXPECT_DOUBLE_EQ(eta_red, eta_violet) << "non-dispersive dielectric must ignore lambda_nm entirely";
	EXPECT_DOUBLE_EQ(eta_red, eta_reference)
		<< "scatter_dispersive() on a non-dispersive instance must match ordinary scatter()";
}
