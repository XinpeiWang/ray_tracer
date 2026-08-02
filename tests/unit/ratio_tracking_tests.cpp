// ratio_tracking_tests.cpp -- unit tests for src/shared/ratio_tracking.h
// Validates alignment with pbrt-v4 SampleT_maj / VolPathIntegrator semantics.

#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

#include "../../src/shared/ratio_tracking.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static HomogeneousMediumData<double> make_medium(
	double sigma_a, double sigma_s, double g = 0.0) {
	return HomogeneousMediumData<double>(sigma_a, sigma_a, sigma_a,
										 sigma_s, sigma_s, sigma_s, g);
}

static RNG make_rng(uint64_t seed = 12345) {
	return RNG(seed, 0);
}

// ===========================================================================
// SampleExponential
// ===========================================================================

TEST(SampleExponential, DistributionMean) {
	// E[SampleExponential(u, a)] = 1/a
	const double a = 2.0;
	const int N = 100000;
	double sum = 0.0;
	RNG rng = make_rng(1);
	for (int i = 0; i < N; ++i)
		sum += SampleExponential<double>(rng.Uniform<float>(), a);
	double mean = sum / N;
	EXPECT_NEAR(mean, 1.0 / a, 0.05 / a);  // 5% tolerance
}

TEST(SampleExponential, ReturnsPositive) {
	RNG rng = make_rng(42);
	for (int i = 0; i < 1000; ++i)
		EXPECT_GT(SampleExponential<double>(rng.Uniform<float>(), 1.0), 0.0);
}

// ===========================================================================
// HomogeneousMajorantIterator
// ===========================================================================

TEST(HomogeneousMajorantIterator, ReturnsOneThenEmpty) {
	auto med = make_medium(0.1, 0.3);
	HomogeneousMajorantIterator<double> it(0.0, 5.0, med);

	auto seg = it.Next();
	ASSERT_TRUE(seg.has_value());
	EXPECT_NEAR(seg->tMin, 0.0, 1e-10);
	EXPECT_NEAR(seg->tMax, 5.0, 1e-10);
	// sigma_maj = sigma_a + sigma_s = 0.4
	EXPECT_NEAR(seg->sigma_maj_r, 0.4, 1e-10);

	auto seg2 = it.Next();
	EXPECT_FALSE(seg2.has_value());
}

TEST(HomogeneousMajorantIterator, DefaultIsEmpty) {
	HomogeneousMajorantIterator<double> it;
	EXPECT_FALSE(it.Next().has_value());
}

TEST(HomogeneousMajorantIterator, SigmaMajEqualsExtinction) {
	auto med = make_medium(0.2, 0.5);
	HomogeneousMajorantIterator<double> it(1.0, 3.0, med);
	auto seg = it.Next();
	ASSERT_TRUE(seg.has_value());
	EXPECT_NEAR(seg->sigma_maj_r, 0.7, 1e-10);
	EXPECT_NEAR(seg->sigma_maj_g, 0.7, 1e-10);
	EXPECT_NEAR(seg->sigma_maj_b, 0.7, 1e-10);
}

// ===========================================================================
// SampleMediumInteraction
// ===========================================================================

TEST(SampleMediumInteraction, VacuumReturnsNull) {
	auto med = make_medium(0.0, 0.0);
	RNG rng = make_rng();
	auto s = SampleMediumInteraction<double>(med, 0.0, 10.0, rng);
	EXPECT_EQ(s.event_type, MediumEventType::kNull);
}

TEST(SampleMediumInteraction, HighDensityAlwaysScattersOrAbsorbs) {
	// Very high density: almost certainly an event before tMax
	auto med = make_medium(5.0, 5.0);
	RNG rng = make_rng(99);
	int null_count = 0;
	const int N = 1000;
	for (int i = 0; i < N; ++i) {
		auto s = SampleMediumInteraction<double>(med, 0.0, 10.0, rng);
		if (s.event_type == MediumEventType::kNull) ++null_count;
	}
	// With sigma_t=10 and tMax=10, P(no event) = exp(-100) ≈ 0
	EXPECT_LT(null_count, 5);
}

TEST(SampleMediumInteraction, PureAbsorberAlwaysAbsorbs) {
	// sigma_s = 0 -> scatter prob = 0, only absorption or null
	auto med = make_medium(10.0, 0.0);
	RNG rng = make_rng(7);
	int scatter_count = 0;
	for (int i = 0; i < 500; ++i) {
		auto s = SampleMediumInteraction<double>(med, 0.0, 5.0, rng);
		if (s.event_type == MediumEventType::kScatter) ++scatter_count;
	}
	EXPECT_EQ(scatter_count, 0);
}

TEST(SampleMediumInteraction, PureScattererAlwaysScatters) {
	// sigma_a = 0 -> absorption prob = 0
	auto med = make_medium(0.0, 10.0);
	RNG rng = make_rng(3);
	int absorb_count = 0;
	for (int i = 0; i < 500; ++i) {
		auto s = SampleMediumInteraction<double>(med, 0.0, 5.0, rng);
		if (s.event_type == MediumEventType::kAbsorption) ++absorb_count;
	}
	EXPECT_EQ(absorb_count, 0);
}

TEST(SampleMediumInteraction, ScatterRatioMatchesProbability) {
	// P(scatter | event) = sigma_s / (sigma_a + sigma_s)
	// With sigma_a = 1, sigma_s = 3: P(scatter) = 0.75
	auto med = make_medium(1.0, 3.0);
	RNG rng = make_rng(42);
	int scatter = 0, absorb = 0;
	const int N = 50000;
	for (int i = 0; i < N; ++i) {
		auto s = SampleMediumInteraction<double>(med, 0.0, 100.0, rng);
		if (s.event_type == MediumEventType::kScatter) ++scatter;
		else if (s.event_type == MediumEventType::kAbsorption) ++absorb;
	}
	double total = scatter + absorb;
	ASSERT_GT(total, 1000);
	double frac = scatter / total;
	EXPECT_NEAR(frac, 0.75, 0.02);  // within 2%
}

TEST(SampleMediumInteraction, ScatterTInRange) {
	auto med = make_medium(0.5, 0.5);
	RNG rng = make_rng(55);
	const double tMin = 1.0, tMax = 5.0;
	for (int i = 0; i < 1000; ++i) {
		auto s = SampleMediumInteraction<double>(med, tMin, tMax, rng);
		if (s.event_type != MediumEventType::kNull) {
			EXPECT_GE(s.t, tMin);
			EXPECT_LE(s.t, tMax);
		}
	}
}

TEST(SampleMediumInteraction, TMajNonNegativeAndLeOne) {
	auto med = make_medium(0.3, 0.7);
	RNG rng = make_rng(77);
	for (int i = 0; i < 500; ++i) {
		auto s = SampleMediumInteraction<double>(med, 0.0, 3.0, rng);
		EXPECT_GE(s.T_maj_r, 0.0);
		EXPECT_GE(s.T_maj_g, 0.0);
		EXPECT_GE(s.T_maj_b, 0.0);
		EXPECT_LE(s.T_maj_r, 1.0 + 1e-9);
		EXPECT_LE(s.T_maj_g, 1.0 + 1e-9);
		EXPECT_LE(s.T_maj_b, 1.0 + 1e-9);
	}
}

// ===========================================================================
// RatioTrackingTr
// ===========================================================================

TEST(RatioTrackingTr, VacuumIsOne) {
	auto med = make_medium(0.0, 0.0);
	RNG rng = make_rng();
	double r, g, b;
	RatioTrackingTr<double>(med, 0.0, 5.0, rng, r, g, b);
	EXPECT_NEAR(r, 1.0, 1e-9);
	EXPECT_NEAR(g, 1.0, 1e-9);
	EXPECT_NEAR(b, 1.0, 1e-9);
}

TEST(RatioTrackingTr, HomogeneousMatchesBeerLambert) {
	// For HomogeneousMediumData, sigma_n = max(0, sigma_maj - sigma_a - sigma_s) = 0
	// because sigma_maj = sigma_a + sigma_s = sigma_t exactly.
	// The stochastic ratio-tracking estimator degenerates for sigma_n = 0
	// (every sampled null-scatter weight = sigma_n/sigma_maj = 0).
	// RatioTrackingTr therefore computes Beer-Lambert directly:
	//   Tr = exp(-sigma_t * (tMax - tMin))
	// This is deterministic -- verify exactly, not statistically.
	const double sigma_a = 0.3, sigma_s = 0.2;
	const double sigma_t = sigma_a + sigma_s;
	const double tMin = 0.0, tMax = 3.0;
	const double expected_Tr = std::exp(-sigma_t * (tMax - tMin));

	auto med = make_medium(sigma_a, sigma_s);
	RNG rng = make_rng(42);  // not actually used for homogeneous media
	double r, g, b;
	RatioTrackingTr<double>(med, tMin, tMax, rng, r, g, b);

	EXPECT_NEAR(r, expected_Tr, 1e-12);
	EXPECT_NEAR(g, expected_Tr, 1e-12);
	EXPECT_NEAR(b, expected_Tr, 1e-12);
}

TEST(RatioTrackingTr, LongRayNearlyZeroTr) {
	// Very optically thick medium: Tr should be near 0
	auto med = make_medium(5.0, 5.0);
	RNG rng = make_rng(999);
	double r, g, b;
	RatioTrackingTr<double>(med, 0.0, 10.0, rng, r, g, b);
	// exp(-100) ≈ 3.7e-44; with ratio-tracking it may round to 0
	EXPECT_LE(r, 1e-10);
	EXPECT_LE(g, 1e-10);
	EXPECT_LE(b, 1e-10);
}

TEST(RatioTrackingTr, ResultInZeroOne) {
	auto med = make_medium(0.4, 0.6);
	for (int i = 0; i < 500; ++i) {
		RNG rng = make_rng(static_cast<uint64_t>(i * 31 + 7));
		double r, g, b;
		RatioTrackingTr<double>(med, 0.0, 2.0, rng, r, g, b);
		EXPECT_GE(r, 0.0);  EXPECT_LE(r, 1.0 + 1e-9);
		EXPECT_GE(g, 0.0);  EXPECT_LE(g, 1.0 + 1e-9);
		EXPECT_GE(b, 0.0);  EXPECT_LE(b, 1.0 + 1e-9);
	}
}

TEST(RatioTrackingTr, MonotonicallyDecreasingWithDistance) {
	// Longer segments -> lower expected Tr
	auto med = make_medium(0.5, 0.5);
	const int N = 5000;
	double sum1 = 0.0, sum2 = 0.0;
	for (int i = 0; i < N; ++i) {
		RNG rng1 = make_rng(static_cast<uint64_t>(i));
		RNG rng2 = make_rng(static_cast<uint64_t>(i + N));
		double r1, g1, b1, r2, g2, b2;
		RatioTrackingTr<double>(med, 0.0, 1.0, rng1, r1, g1, b1);
		RatioTrackingTr<double>(med, 0.0, 5.0, rng2, r2, g2, b2);
		sum1 += r1; sum2 += r2;
	}
	EXPECT_GT(sum1 / N, sum2 / N);
}

// ===========================================================================
// MediumEventType enum coverage
// ===========================================================================

TEST(MediumEventType, AllValuesDistinct) {
	EXPECT_NE(MediumEventType::kAbsorption, MediumEventType::kScatter);
	EXPECT_NE(MediumEventType::kScatter,    MediumEventType::kNull);
	EXPECT_NE(MediumEventType::kAbsorption, MediumEventType::kNull);
}

// ===========================================================================
// Integration: SampleMediumInteraction + path weight consistency
// ===========================================================================

TEST(RatioTracking, ScatterEventCarriesCorrectSigmaS) {
	auto med = make_medium(0.1, 0.9);
	RNG rng = make_rng(123);
	// Find a scatter event and verify sigma_s fields are set correctly
	for (int i = 0; i < 200; ++i) {
		auto s = SampleMediumInteraction<double>(med, 0.0, 5.0, rng);
		if (s.event_type == MediumEventType::kScatter) {
			EXPECT_NEAR(s.sigma_s_r, 0.9, 1e-9);
			EXPECT_NEAR(s.sigma_s_g, 0.9, 1e-9);
			EXPECT_NEAR(s.sigma_s_b, 0.9, 1e-9);
			return;
		}
	}
	FAIL() << "No scatter event produced in 200 trials";
}

TEST(RatioTracking, AbsorptionEventCarriesCorrectSigmaA) {
	auto med = make_medium(0.9, 0.1);
	RNG rng = make_rng(456);
	for (int i = 0; i < 200; ++i) {
		auto s = SampleMediumInteraction<double>(med, 0.0, 5.0, rng);
		if (s.event_type == MediumEventType::kAbsorption) {
			EXPECT_NEAR(s.sigma_a_r, 0.9, 1e-9);
			return;
		}
	}
	FAIL() << "No absorption event produced in 200 trials";
}
