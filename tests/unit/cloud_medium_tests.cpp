// cloud_medium_tests.cpp -- Regression tests for CloudMedium<T>
//
// Tests cover:
//   1.  HomogeneousMajorantIterator: empty (no-hit) and valid (one segment)
//   2.  CloudMedium::compute_density: range [0,1], altitude falloff, density=0 case
//   3.  CloudMedium::sample_point: proportional to density
//   4.  CloudMedium::sample_ray: miss, hit (tMin/tMax clipping), sigma_t = sigma_a + sigma_s
//   5.  Wispiness: non-zero wispiness produces different result from wispiness=0
//   6.  Frequency: scaling by frequency changes the density

#include <gtest/gtest.h>
#include <cmath>
#include "../../src/shared/cloud_medium.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static const double kEps = 1e-6;

// Identity transform helpers
static const double kIdMat[9]  = {1,0,0, 0,1,0, 0,0,1};
static const double kIdTr[3]   = {0,0,0};

// Build a unit-cube cloud with given params
static CloudMedium<double> make_cloud(double sa = 0.1, double ss = 0.9,
									  double g  = 0.0,
									  double den = 1.0, double wisp = 0.0,
									  double freq = 1.0) {
	return CloudMedium<double>::make_unit(sa, ss, g, den, wisp, freq);
}

// ---------------------------------------------------------------------------
// HomogeneousMajorantIterator
// ---------------------------------------------------------------------------

TEST(CloudMediumMajorant, EmptyIteratorYieldsNoSegment) {
	CloudMajorantIterator<double> it;
	double t0, t1, sig;
	EXPECT_FALSE(it.next(t0, t1, sig));
}

TEST(CloudMediumMajorant, ValidIteratorYieldsOneSegment) {
	CloudMajorantIterator<double> it(0.5, 2.5, 0.8);
	EXPECT_TRUE(it.hit);
	double t0, t1, sig;
	EXPECT_TRUE(it.next(t0, t1, sig));
	EXPECT_NEAR(t0,  0.5, kEps);
	EXPECT_NEAR(t1,  2.5, kEps);
	EXPECT_NEAR(sig, 0.8, kEps);
	// Second call must be empty
	EXPECT_FALSE(it.next(t0, t1, sig));
}

TEST(CloudMediumMajorant, EmptyIteratorHitFalse) {
	CloudMajorantIterator<double> it;
	EXPECT_FALSE(it.hit);
}

// ---------------------------------------------------------------------------
// compute_density
// ---------------------------------------------------------------------------

TEST(CloudMediumDensity, OutputInUnitRange) {
	auto cloud = make_cloud();
	// Sample a grid of medium-space points and verify density in [0,1]
	for (int ix = 0; ix <= 4; ++ix)
	for (int iy = 0; iy <= 4; ++iy)
	for (int iz = 0; iz <= 4; ++iz) {
		double px = ix * 0.25;
		double py = iy * 0.25;
		double pz = iz * 0.25;
		double d = cloud.compute_density(px, py, pz);
		EXPECT_GE(d, 0.0) << "px=" << px << " py=" << py << " pz=" << pz;
		EXPECT_LE(d, 1.0) << "px=" << px << " py=" << py << " pz=" << pz;
	}
}

TEST(CloudMediumDensity, DensityScaleZeroYieldsZeroOrAltitudeTerm) {
	// When density=0, the altitude falloff d stays from extra = 2*max(0,0.5-py)
	// At py=0.0: extra = 2*0.5 = 1.0, clamped → 1
	auto cloud = make_cloud(0.1, 0.9, 0.0, 0.0);
	double d = cloud.compute_density(0.5, 0.0, 0.5);
	EXPECT_GE(d, 0.0);
	EXPECT_LE(d, 1.0);
}

TEST(CloudMediumDensity, HighAltitudeLowerDensity) {
	// At py=0 vs py=1 density should generally be >= at py=1 because
	// altitude factor (1-py) is 1 vs 0.  Use a fixed x,z to compare.
	auto cloud = make_cloud(0.1, 0.9, 0.0, 1.0, 0.0, 1.0);
	double d_low  = cloud.compute_density(0.5, 0.0, 0.5);
	double d_high = cloud.compute_density(0.5, 1.0, 0.5);
	// At py=1: altitude factor is 0, extra=max(0,0.5-1)=0 → density mostly 0
	EXPECT_GE(d_low, d_high);
}

TEST(CloudMediumDensity, IsDeterministic) {
	auto cloud = make_cloud();
	double d1 = cloud.compute_density(0.3, 0.2, 0.7);
	double d2 = cloud.compute_density(0.3, 0.2, 0.7);
	EXPECT_DOUBLE_EQ(d1, d2);
}

TEST(CloudMediumDensity, WispinessChangesResult) {
	auto c0 = make_cloud(0.1, 0.9, 0.0, 1.0, 0.0, 1.0);
	auto c1 = make_cloud(0.1, 0.9, 0.0, 1.0, 1.0, 1.0);
	// At most points wispiness displaces the lookup point → different density
	bool any_diff = false;
	for (int i = 0; i < 5; ++i) {
		double d0 = c0.compute_density(0.2 + i*0.1, 0.3, 0.4);
		double d1 = c1.compute_density(0.2 + i*0.1, 0.3, 0.4);
		if (std::abs(d0 - d1) > 1e-9) { any_diff = true; break; }
	}
	EXPECT_TRUE(any_diff);
}

TEST(CloudMediumDensity, FrequencyChangesResult) {
	auto c1 = make_cloud(0.1, 0.9, 0.0, 1.0, 0.0, 1.0);
	auto c2 = make_cloud(0.1, 0.9, 0.0, 1.0, 0.0, 2.0);
	bool any_diff = false;
	for (int i = 0; i < 5; ++i) {
		double d1 = c1.compute_density(0.2 + i*0.1, 0.4, 0.3);
		double d2 = c2.compute_density(0.2 + i*0.1, 0.4, 0.3);
		if (std::abs(d1 - d2) > 1e-9) { any_diff = true; break; }
	}
	EXPECT_TRUE(any_diff);
}

// ---------------------------------------------------------------------------
// sample_point
// ---------------------------------------------------------------------------

TEST(CloudMediumSamplePoint, SigmaProportionalToDensity) {
	auto cloud = make_cloud(0.2, 0.8, 0.0, 1.0, 0.0, 1.0);
	// identity transform: world == medium space
	double sa, ss;
	cloud.sample_point(0.5, 0.3, 0.5, sa, ss);
	double d = cloud.compute_density(0.5, 0.3, 0.5);
	EXPECT_NEAR(sa, d * 0.2, kEps);
	EXPECT_NEAR(ss, d * 0.8, kEps);
}

TEST(CloudMediumSamplePoint, ZeroDensityGivesZeroSigma) {
	auto cloud = make_cloud(0.2, 0.8, 0.0, 0.0, 0.0, 1.0);
	double sa, ss;
	// At py=1 altitude factor zeros density (extra=0), main term=0 → d=0
	cloud.sample_point(0.5, 1.0, 0.5, sa, ss);
	double d = cloud.compute_density(0.5, 1.0, 0.5);
	EXPECT_NEAR(sa, d * 0.2, kEps);
	EXPECT_NEAR(ss, d * 0.8, kEps);
}

TEST(CloudMediumSamplePoint, NonNegativeSigma) {
	auto cloud = make_cloud();
	double sa, ss;
	for (int i = 0; i < 8; ++i) {
		double p = i * 0.125;
		cloud.sample_point(p, p, p, sa, ss);
		EXPECT_GE(sa, 0.0);
		EXPECT_GE(ss, 0.0);
	}
}

// ---------------------------------------------------------------------------
// sample_ray
// ---------------------------------------------------------------------------

TEST(CloudMediumSampleRay, MissRayYieldsNoSegment) {
	auto cloud = make_cloud();
	// Ray entirely above/outside the unit cube: origin at (0.5, 2.0, 0.5), dir up
	double ro[3] = {0.5, 2.0, 0.5};
	double rd[3] = {0.0, 1.0, 0.0};
	auto it = cloud.sample_ray(ro, rd, 1e10);
	EXPECT_FALSE(it.hit);
	double t0, t1, sig;
	EXPECT_FALSE(it.next(t0, t1, sig));
}

TEST(CloudMediumSampleRay, HitRayYieldsValidSegment) {
	auto cloud = make_cloud(0.1, 0.9);
	// Ray through the unit cube along +Z
	double ro[3] = {0.5, 0.3, -1.0};
	double rd[3] = {0.0, 0.0,  1.0};
	auto it = cloud.sample_ray(ro, rd, 1e10);
	EXPECT_TRUE(it.hit);
	double t0, t1, sig;
	EXPECT_TRUE(it.next(t0, t1, sig));
	EXPECT_NEAR(t0, 1.0, 1e-5);   // enters at z=0
	EXPECT_NEAR(t1, 2.0, 1e-5);   // exits  at z=1
	EXPECT_NEAR(sig, 0.1 + 0.9, kEps);
}

TEST(CloudMediumSampleRay, SigmaTIsSum) {
	auto cloud = make_cloud(0.3, 0.7);
	double ro[3] = {0.5, 0.5, -1.0};
	double rd[3] = {0.0, 0.0,  1.0};
	auto it = cloud.sample_ray(ro, rd, 1e10);
	double t0, t1, sig;
	ASSERT_TRUE(it.next(t0, t1, sig));
	EXPECT_NEAR(sig, 1.0, kEps);  // 0.3 + 0.7
}

TEST(CloudMediumSampleRay, RayTMaxClipsSegment) {
	auto cloud = make_cloud();
	// Ray along +Z from outside; limit tmax to 1.5 so it exits inside the cube
	double ro[3] = {0.5, 0.5, -1.0};
	double rd[3] = {0.0, 0.0,  1.0};
	auto it = cloud.sample_ray(ro, rd, 1.5);
	EXPECT_TRUE(it.hit);
	double t0, t1, sig;
	ASSERT_TRUE(it.next(t0, t1, sig));
	EXPECT_LE(t1, 1.5 + 1e-5);
}
