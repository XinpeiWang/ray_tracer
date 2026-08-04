// homogeneous_medium_tests.cpp
// Unit tests for HomogeneousMedium<T> and HomogeneousMajorantIterator<T>.
// Mirrors pbrt-v4 src/pbrt/media.h HomogeneousMedium (lines 217-263).
#include <gtest/gtest.h>
#include "../../src/shared/homogeneous_medium.h"
#include <cmath>
#include <optional>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(HomogeneousMedium, DefaultConstructsToZero) {
	HomogeneousMedium<double> med;
	EXPECT_EQ(med.sigma_a(), 0.0);
	EXPECT_EQ(med.sigma_s(), 0.0);
	EXPECT_EQ(med.sigma_t(), 0.0);
	EXPECT_EQ(med.Le(), 0.0);
	EXPECT_EQ(med.g(), 0.0);
}

TEST(HomogeneousMedium, ConstructsWithParameters) {
	HomogeneousMedium<double> med(0.1, 0.5, 0.0, 0.3);
	EXPECT_DOUBLE_EQ(med.sigma_a(), 0.1);
	EXPECT_DOUBLE_EQ(med.sigma_s(), 0.5);
	EXPECT_DOUBLE_EQ(med.sigma_t(), 0.6);
	EXPECT_DOUBLE_EQ(med.Le(), 0.0);
	EXPECT_DOUBLE_EQ(med.g(), 0.3);
}

TEST(HomogeneousMedium, SigmaScaleApplied) {
	// pbrt-v4: sigma_a_spec.Scale(sigmaScale); sigma_s_spec.Scale(sigmaScale)
	HomogeneousMedium<double> med(1.0, 2.0, 0.0, 0.0, /*sigmaScale=*/0.5);
	EXPECT_DOUBLE_EQ(med.sigma_a(), 0.5);
	EXPECT_DOUBLE_EQ(med.sigma_s(), 1.0);
	EXPECT_DOUBLE_EQ(med.sigma_t(), 1.5);
}

TEST(HomogeneousMedium, LeScaleApplied) {
	// pbrt-v4: Le_spec.Scale(LeScale)
	HomogeneousMedium<double> med(0.0, 0.0, 2.0, 0.0, /*sigmaScale=*/1.0, /*LeScale=*/3.0);
	EXPECT_DOUBLE_EQ(med.Le(), 6.0);
}

// ---------------------------------------------------------------------------
// IsEmissive
// ---------------------------------------------------------------------------

TEST(HomogeneousMedium, NonEmissiveWhenLeZero) {
	HomogeneousMedium<double> med(0.1, 0.5, 0.0, 0.0);
	EXPECT_FALSE(med.is_emissive());
}

TEST(HomogeneousMedium, EmissiveWhenLePositive) {
	HomogeneousMedium<double> med(0.1, 0.5, 1.0, 0.0);
	EXPECT_TRUE(med.is_emissive());
}

// ---------------------------------------------------------------------------
// sample_point — MediumPoint properties
// ---------------------------------------------------------------------------

TEST(HomogeneousMedium, SamplePointReturnsConstantCoeffs) {
	HomogeneousMedium<double> med(0.2, 0.8, 0.5, 0.7);
	auto p = med.sample_point();
	EXPECT_DOUBLE_EQ(p.sigma_a, 0.2);
	EXPECT_DOUBLE_EQ(p.sigma_s, 0.8);
	EXPECT_DOUBLE_EQ(p.Le, 0.5);
	EXPECT_DOUBLE_EQ(p.g, 0.7);
}

TEST(HomogeneousMedium, SamplePointSigmaTMatchesSumOfCoeffs) {
	HomogeneousMedium<double> med(0.3, 0.7, 0.0, 0.0);
	auto p = med.sample_point();
	EXPECT_DOUBLE_EQ(p.sigma_t(), 1.0);
}

TEST(HomogeneousMedium, SamplePointIsUniform) {
	// For a homogeneous medium, every point returns identical properties
	HomogeneousMedium<double> med(0.1, 0.9, 0.2, 0.5);
	auto p1 = med.sample_point();
	auto p2 = med.sample_point();
	EXPECT_DOUBLE_EQ(p1.sigma_a, p2.sigma_a);
	EXPECT_DOUBLE_EQ(p1.sigma_s, p2.sigma_s);
	EXPECT_DOUBLE_EQ(p1.Le,      p2.Le);
}

// ---------------------------------------------------------------------------
// HomogeneousMajorantIterator — sample_ray
// ---------------------------------------------------------------------------

TEST(HomogeneousMedium, SampleRayReturnsOneSegment) {
	HomogeneousMedium<double> med(0.1, 0.4, 0.0, 0.0);
	auto it = med.sample_ray(10.0);
	auto seg = it.next();
	ASSERT_TRUE(seg.has_value());
	EXPECT_DOUBLE_EQ(seg->tMin, 0.0);
	EXPECT_DOUBLE_EQ(seg->tMax, 10.0);
	// sigma_maj = sigma_a + sigma_s = 0.5
	EXPECT_DOUBLE_EQ(seg->sigma_maj, 0.5);
}

TEST(HomogeneousMedium, SampleRayExhaustsAfterOneSegment) {
	// pbrt-v4: HomogeneousMajorantIterator yields exactly one segment then {}
	HomogeneousMedium<double> med(0.1, 0.4, 0.0, 0.0);
	auto it = med.sample_ray(5.0);
	auto seg1 = it.next();
	ASSERT_TRUE(seg1.has_value());
	auto seg2 = it.next();
	EXPECT_FALSE(seg2.has_value());
}

TEST(HomogeneousMedium, SampleRaySegmentCoversFullInterval) {
	HomogeneousMedium<double> med(0.05, 0.15, 0.0, 0.0);
	double tMax = 100.0;
	auto it = med.sample_ray(tMax);
	auto seg = it.next();
	ASSERT_TRUE(seg.has_value());
	EXPECT_DOUBLE_EQ(seg->tMin, 0.0);
	EXPECT_DOUBLE_EQ(seg->tMax, tMax);
}

TEST(HomogeneousMedium, SampleRayMajorantEqualsExactSigmaT) {
	// For homogeneous medium: majorant == sigma_t (no over-estimation needed)
	HomogeneousMedium<double> med(0.3, 0.7, 0.0, 0.0);
	auto it = med.sample_ray(1.0);
	auto seg = it.next();
	ASSERT_TRUE(seg.has_value());
	EXPECT_DOUBLE_EQ(seg->sigma_maj, med.sigma_t());
}

TEST(HomogeneousMedium, SampleRayZeroExtinctionYearsZeroMajorant) {
	HomogeneousMedium<double> med(0.0, 0.0, 0.0, 0.0);
	auto it = med.sample_ray(1.0);
	auto seg = it.next();
	ASSERT_TRUE(seg.has_value());
	EXPECT_DOUBLE_EQ(seg->sigma_maj, 0.0);
}

// ---------------------------------------------------------------------------
// HomogeneousMajorantIterator standalone
// ---------------------------------------------------------------------------

TEST(HomogeneousMajorantIterator, DefaultIsExhausted) {
	HomogeneousMajorantIterator<double> it;
	EXPECT_FALSE(it.next().has_value());
}

TEST(HomogeneousMajorantIterator, ConstructedYieldsSegmentThenExhausts) {
	HomogeneousMajorantIterator<double> it(1.0, 5.0, 2.5);
	auto seg = it.next();
	ASSERT_TRUE(seg.has_value());
	EXPECT_DOUBLE_EQ(seg->tMin, 1.0);
	EXPECT_DOUBLE_EQ(seg->tMax, 5.0);
	EXPECT_DOUBLE_EQ(seg->sigma_maj, 2.5);
	EXPECT_FALSE(it.next().has_value());
}

// ---------------------------------------------------------------------------
// Phase function accessor
// ---------------------------------------------------------------------------

TEST(HomogeneousMedium, PhaseAccessorReturnsCorrectG) {
	// Verify the phase function differentiates forward vs backward scatter.
	// With g=0.8 the HG function should peak in one direction.
	// We test both cos_theta values are different (not equal) and that the
	// isotropic case (g=0) gives equal values.
	HomogeneousMedium<double> med(0.1, 0.5, 0.0, 0.8);
	const auto& ph = med.phase();
	double pdf_fwd = ph.p(1.0);
	double pdf_bwd = ph.p(-1.0);
	EXPECT_NE(pdf_fwd, pdf_bwd);
	// Both should be non-negative
	EXPECT_GE(pdf_fwd, 0.0);
	EXPECT_GE(pdf_bwd, 0.0);
}

TEST(HomogeneousMedium, PhaseIsotropicForGZero) {
	HomogeneousMedium<double> med(0.1, 0.5, 0.0, 0.0);
	const auto& ph = med.phase();
	double pdf_fwd = ph.p(1.0);
	double pdf_bwd = ph.p(-1.0);
	EXPECT_NEAR(pdf_fwd, pdf_bwd, 1e-10);
}
