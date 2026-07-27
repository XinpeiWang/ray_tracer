// filter_sampler_tests.cpp
// Unit tests for FilterSampler<T,N> (src/shared/filter_sampler.h)
// and LanczosSincFilter (src/shared/filter.h)
// Mirrors pbrt-v4 FilterSampler / LanczosSincFilter validation approach.

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>

#include "../../src/shared/filter_sampler.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static double radical_inverse2(uint32_t n) {
	uint32_t bits = (n << 16) | (n >> 16);
	bits = ((bits & 0x55555555u) << 1) | ((bits & 0xAAAAAAAAu) >> 1);
	bits = ((bits & 0x33333333u) << 2) | ((bits & 0xCCCCCCCCu) >> 2);
	bits = ((bits & 0x0F0F0F0Fu) << 4) | ((bits & 0xF0F0F0F0u) >> 4);
	bits = ((bits & 0x00FF00FFu) << 8) | ((bits & 0xFF00FF00u) >> 8);
	return bits * 2.3283064365386963e-10;
}
static double radical_inverse3(uint32_t n) {
	double inv = 1.0 / 3.0, result = 0.0, f = inv;
	while (n > 0) { result += (n % 3) * f; n /= 3; f *= inv; }
	return result;
}

// ---------------------------------------------------------------------------
// Test: FilterSampler constructs without error and radius is preserved
// ---------------------------------------------------------------------------
TEST(FilterSampler, ConstructsForAllFilterTypes) {
	{
		BoxFilter f;
		FilterSampler<double> fs(f);
		EXPECT_NEAR(fs.radius(), 0.5, 1e-10);
	}
	{
		MitchellFilter f(0.5);
		FilterSampler<double> fs(f);
		EXPECT_NEAR(fs.radius(), 0.5, 1e-10);
	}
	{
		GaussianFilter f(0.5);
		FilterSampler<double> fs(f);
		EXPECT_NEAR(fs.radius(), 0.5, 1e-10);
	}
}

// ---------------------------------------------------------------------------
// Test: Sampled positions are always within [-radius, +radius]
// ---------------------------------------------------------------------------
TEST(FilterSampler, SampledPositionsInDomain) {
	MitchellFilter f(0.5);
	FilterSampler<double> fs(f);

	for (uint32_t i = 0; i < 500; ++i) {
		auto s = fs.sample(radical_inverse2(i), radical_inverse3(i));
		EXPECT_GE(s.p_x, -0.5 - 1e-9);
		EXPECT_LE(s.p_x,  0.5 + 1e-9);
		EXPECT_GE(s.p_y, -0.5 - 1e-9);
		EXPECT_LE(s.p_y,  0.5 + 1e-9);
	}
}

// ---------------------------------------------------------------------------
// Test: Sampled weights are non-negative (filter is non-negative in our table)
// ---------------------------------------------------------------------------
TEST(FilterSampler, WeightsNonNegative) {
	MitchellFilter f(0.5);
	FilterSampler<double> fs(f);

	for (uint32_t i = 0; i < 200; ++i) {
		auto s = fs.sample(radical_inverse2(i), radical_inverse3(i));
		EXPECT_GE(s.weight, 0.0);
	}
}

// ---------------------------------------------------------------------------
// Test: Integral matches numerical estimate of filter integral
//   BoxFilter on [-0.5,0.5]² has integral = 1.0
// ---------------------------------------------------------------------------
TEST(FilterSampler, BoxFilterIntegralIsOne) {
	BoxFilter f;
	FilterSampler<double> fs(f);
	EXPECT_NEAR(fs.integral(), 1.0, 0.02);  // 32x32 grid gives near 1.0
}

// ---------------------------------------------------------------------------
// Test: Integral of Mitchell filter matches brute-force numerical integration
// ---------------------------------------------------------------------------
TEST(FilterSampler, MitchellIntegralMatchesBruteForce) {
	MitchellFilter f(0.5);
	FilterSampler<double> fs(f);

	// Brute-force 1000x1000 Riemann sum
	const int M = 1000;
	double sum = 0.0;
	double step = 1.0 / M;
	for (int iy = 0; iy < M; ++iy) {
		for (int ix = 0; ix < M; ++ix) {
			double ox = -0.5 + (ix + 0.5) * step;
			double oy = -0.5 + (iy + 0.5) * step;
			double v = f.evaluate(ox, oy);
			sum += (v > 0.0 ? v : 0.0);
		}
	}
	double brute = sum * step * step;

	EXPECT_NEAR(fs.integral(), brute, brute * 0.02)
		<< "FilterSampler integral=" << fs.integral() << " brute=" << brute;
}

// ---------------------------------------------------------------------------
// Test: Importance sampling produces correct estimator for BoxFilter
//   BoxFilter is uniform: every sample should have weight ≈ integral
//   because f(p)/pdf = integral for a uniform distribution.
// ---------------------------------------------------------------------------
TEST(FilterSampler, BoxFilterImportanceSamplingWeightIsIntegral) {
	BoxFilter f;
	FilterSampler<double> fs(f);

	double sumW = 0.0;
	const int N = 1000;
	for (int i = 0; i < N; ++i) {
		auto s = fs.sample(radical_inverse2((uint32_t)i),
						   radical_inverse3((uint32_t)i));
		sumW += s.weight;
	}
	double avg = sumW / N;
	// For a uniform distribution, E[f/pdf] = integral
	EXPECT_NEAR(avg, fs.integral(), fs.integral() * 0.05)
		<< "Average weight=" << avg << " expected integral=" << fs.integral();
}

// ---------------------------------------------------------------------------
// Test: Importance sampling of Mitchell filter: E[1] = 1 (IS normalization)
//   Because E_p~f[w] = E[f(p)/pdf(p)] = integral, and we verify the
//   IS estimator for the constant function 1 over the domain gives the integral.
// ---------------------------------------------------------------------------
TEST(FilterSampler, MitchellImportanceSamplingNormalised) {
	MitchellFilter f(0.5);
	FilterSampler<double> fs(f);

	double sumW = 0.0;
	const int N = 2000;
	for (int i = 0; i < N; ++i) {
		auto s = fs.sample(radical_inverse2((uint32_t)i),
						   radical_inverse3((uint32_t)i));
		sumW += s.weight;
	}
	double avg = sumW / N;
	// E[f/pdf] = integral (constant-function IS identity)
	EXPECT_NEAR(avg, fs.integral(), fs.integral() * 0.05)
		<< "Mitchell IS avg=" << avg << " integral=" << fs.integral();
}

// ---------------------------------------------------------------------------
// Test: Gaussian filter — same IS normalization check
// ---------------------------------------------------------------------------
TEST(FilterSampler, GaussianImportanceSamplingNormalised) {
	GaussianFilter f(0.5, 0.5);
	FilterSampler<double> fs(f);

	double sumW = 0.0;
	const int N = 2000;
	for (int i = 0; i < N; ++i) {
		auto s = fs.sample(radical_inverse2((uint32_t)i),
						   radical_inverse3((uint32_t)i));
		sumW += s.weight;
	}
	double avg = sumW / N;
	EXPECT_NEAR(avg, fs.integral(), fs.integral() * 0.05)
		<< "Gaussian IS avg=" << avg << " integral=" << fs.integral();
}

// ---------------------------------------------------------------------------
// Test: Reproducibility — same (u1,u2) always gives the same result
// ---------------------------------------------------------------------------
TEST(FilterSampler, Reproducibility) {
	MitchellFilter f(0.5);
	FilterSampler<double> fs(f);

	auto s1 = fs.sample(0.3, 0.7);
	auto s2 = fs.sample(0.3, 0.7);
	EXPECT_EQ(s1.p_x, s2.p_x);
	EXPECT_EQ(s1.p_y, s2.p_y);
	EXPECT_EQ(s1.weight, s2.weight);
}

// ---------------------------------------------------------------------------
// Test: PDF is non-negative over the whole domain
// ---------------------------------------------------------------------------
TEST(FilterSampler, PDFNonNegative) {
	MitchellFilter f(0.5);
	FilterSampler<double> fs(f);

	for (uint32_t i = 0; i < 200; ++i) {
		auto s = fs.sample(radical_inverse2(i), radical_inverse3(i));
		double p = fs.pdf(s.p_x, s.p_y);
		EXPECT_GE(p, 0.0);
	}
}

// ---------------------------------------------------------------------------
// Test: Importance sampling concentrates samples in high-weight region
//   For Mitchell filter (positive centre, lower wings), more samples
//   should land near (0,0) than near the corners.
// ---------------------------------------------------------------------------
TEST(FilterSampler, MitchellSamplesConcentrateAtCentre) {
	MitchellFilter f(0.5);
	FilterSampler<double> fs(f);

	int near_centre = 0, near_corner = 0;
	const int N = 2000;
	for (int i = 0; i < N; ++i) {
		auto s = fs.sample(radical_inverse2((uint32_t)i),
						   radical_inverse3((uint32_t)i));
		double r = std::sqrt(s.p_x*s.p_x + s.p_y*s.p_y);
		if (r < 0.2) ++near_centre;
		if (r > 0.4) ++near_corner;
	}
	EXPECT_GT(near_centre, near_corner)
		<< "Mitchell IS should favour centre over corners";
}

// ---------------------------------------------------------------------------
// Test: larger N gives more accurate integral estimate
// ---------------------------------------------------------------------------
TEST(FilterSampler, HighResolutionIntegralAccuracy) {
	MitchellFilter f(0.5);
	FilterSampler<double, 64> fs_hi(f);
	FilterSampler<double, 16> fs_lo(f);

	// Brute-force reference
	const int M = 2000;
	double sum = 0.0, step = 1.0 / M;
	for (int iy = 0; iy < M; ++iy)
		for (int ix = 0; ix < M; ++ix) {
			double v = f.evaluate(-0.5+(ix+0.5)*step, -0.5+(iy+0.5)*step);
			sum += (v > 0.0 ? v : 0.0);
		}
	double ref = sum * step * step;

	double err_hi = std::fabs(fs_hi.integral() - ref);
	double err_lo = std::fabs(fs_lo.integral() - ref);
	EXPECT_LT(err_hi, err_lo * 2.0)
		<< "Higher resolution should give comparable or better integral accuracy";
}

// ---------------------------------------------------------------------------
// Test: Intra-cell interpolation produces continuous, non-quantized positions
//   pbrt-v4 uses fractional CDF interpolation so that two nearby u values give
//   nearby positions (not always snapped to the same cell center).
//   We verify: for N=4, distinct u values in the same cell give distinct px.
// ---------------------------------------------------------------------------
TEST(FilterSampler, SampledPositionsAreContinuous) {
	BoxFilter f(0.5);
	FilterSampler<double, 4> fs(f);  // coarse grid to make quantization obvious

	// Two u1 values very close but distinct, same u2
	auto s1 = fs.sample(0.0,   0.5);
	auto s2 = fs.sample(0.001, 0.5);
	auto s3 = fs.sample(0.499, 0.5);
	auto s4 = fs.sample(0.500, 0.5);

	// With cell-center snapping they'd all return -0.375 or -0.125.
	// With interpolation, s1.p_x < s2.p_x < s3.p_x and s4 may be in the next cell.
	EXPECT_LT(s1.p_x, s2.p_x)
		<< "Interpolated positions should be strictly ordered within a cell";
	EXPECT_LT(s2.p_x, s3.p_x)
		<< "Interpolated positions should increase with u1 within a cell";
}

// ===========================================================================
// LanczosSincFilter tests
// pbrt-v4 reference: filters.h LanczosSincFilter, math.h WindowedSinc/Sinc
// ===========================================================================

// ---------------------------------------------------------------------------
// Test: evaluate(0,0) == 1  (sinc(0)*sinc(0) = 1*1 = 1)
// ---------------------------------------------------------------------------
TEST(LanczosSincFilter, PeakAtOriginIsOne) {
	LanczosSincFilter f(4.0, 3.0);
	EXPECT_NEAR(f.evaluate(0.0, 0.0), 1.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Test: evaluate returns 0 outside [-radius, +radius]
// ---------------------------------------------------------------------------
TEST(LanczosSincFilter, ZeroOutsideRadius) {
	LanczosSincFilter f(4.0, 3.0);
	EXPECT_EQ(f.evaluate(4.001, 0.0), 0.0);
	EXPECT_EQ(f.evaluate(0.0, -4.001), 0.0);
	EXPECT_EQ(f.evaluate(5.0, 5.0), 0.0);
}

// ---------------------------------------------------------------------------
// Test: evaluate is symmetric (even function)
// ---------------------------------------------------------------------------
TEST(LanczosSincFilter, Symmetric) {
	LanczosSincFilter f(4.0, 3.0);
	for (double x : {0.5, 1.0, 1.5, 2.0, 3.0, 3.9}) {
		EXPECT_NEAR(f.evaluate(x, 1.0), f.evaluate(-x, 1.0), 1e-12)
			<< "Filter should be symmetric in x at x=" << x;
		EXPECT_NEAR(f.evaluate(1.0, x), f.evaluate(1.0, -x), 1e-12)
			<< "Filter should be symmetric in y at y=" << x;
	}
}

// ---------------------------------------------------------------------------
// Test: evaluate is separable: f(x,y) == f(x,0)*f(0,y) / f(0,0)
//   WindowedSinc(x)*WindowedSinc(y), and since f(0,0)=1, f(x,y)=f1d(x)*f1d(y)
// ---------------------------------------------------------------------------
TEST(LanczosSincFilter, Separable) {
	LanczosSincFilter f(4.0, 3.0);
	for (double x : {0.3, 1.2, 2.5, 3.7}) {
		for (double y : {0.1, 0.9, 2.0, 3.5}) {
			double fx = f.evaluate(x, 0.0);
			double fy = f.evaluate(0.0, y);
			EXPECT_NEAR(f.evaluate(x, y), fx * fy, 1e-12)
				<< "Separability failed at (" << x << "," << y << ")";
		}
	}
}

// ---------------------------------------------------------------------------
// Test: tau parameter changes the filter (wider tau = more lobes = sharper)
// ---------------------------------------------------------------------------
TEST(LanczosSincFilter, TauAffectsFilterShape) {
	LanczosSincFilter f1(4.0, 1.0);  // narrow: 1 lobe
	LanczosSincFilter f3(4.0, 3.0);  // default: 3 lobes
	// At x=1.5 (first negative lobe region), tau=1 differs from tau=3
	double v1 = f1.evaluate(1.5, 0.0);
	double v3 = f3.evaluate(1.5, 0.0);
	EXPECT_NE(v1, v3) << "Different tau should produce different filter values";
}

// ---------------------------------------------------------------------------
// Test: radius() and tau() accessors return the constructed values
// ---------------------------------------------------------------------------
TEST(LanczosSincFilter, AccessorsCorrect) {
	LanczosSincFilter f(3.5, 2.5);
	EXPECT_NEAR(f.radius(), 3.5, 1e-12);
	EXPECT_NEAR(f.tau(),    2.5, 1e-12);
}

// ---------------------------------------------------------------------------
// Test: LanczosSinc works with FilterSampler — IS normalisation check
//   E[f/pdf] = integral  (same identity as for other filter types)
// ---------------------------------------------------------------------------
TEST(LanczosSincFilter, FilterSamplerISNormalised) {
	// Use small radius so the 32x32 grid is dense enough
	LanczosSincFilter f(2.0, 3.0);
	FilterSampler<double> fs(f);

	EXPECT_GT(fs.integral(), 0.0) << "Lanczos integral should be positive";

	double sumW = 0.0;
	const int N = 3000;
	for (int i = 0; i < N; ++i) {
		auto s = fs.sample(radical_inverse2((uint32_t)i),
						   radical_inverse3((uint32_t)i));
		sumW += s.weight;
	}
	double avg = sumW / N;
	EXPECT_NEAR(avg, fs.integral(), fs.integral() * 0.05)
		<< "Lanczos IS avg=" << avg << " integral=" << fs.integral();
}

// ---------------------------------------------------------------------------
// Test: Sampled positions stay within [-radius, +radius] for Lanczos
// ---------------------------------------------------------------------------
TEST(LanczosSincFilter, FilterSamplerPositionsInDomain) {
	LanczosSincFilter f(2.0, 3.0);
	FilterSampler<double> fs(f);

	for (uint32_t i = 0; i < 300; ++i) {
		auto s = fs.sample(radical_inverse2(i), radical_inverse3(i));
		EXPECT_GE(s.p_x, -2.0 - 1e-9);
		EXPECT_LE(s.p_x,  2.0 + 1e-9);
		EXPECT_GE(s.p_y, -2.0 - 1e-9);
		EXPECT_LE(s.p_y,  2.0 + 1e-9);
	}
}
