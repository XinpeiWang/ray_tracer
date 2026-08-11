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
