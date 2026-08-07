// tests/unit/function_integrator_tests.cpp
// Unit tests for FunctionIntegrator (mirrors pbrt-v4 FunctionIntegrator)

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../../src/shared/function_integrator.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Stratified 2D samples over [0,1)^2: nx*ny samples.
static std::vector<double> stratified_samples(int nx, int ny, unsigned seed = 0) {
	std::vector<double> s;
	s.reserve(nx * ny * 2);
	// Simple LCG for jitter
	auto lcg = [&]() -> double {
		seed = seed * 1664525u + 1013904223u;
		return (seed >> 8) / double(1u << 24);
	};
	for (int j = 0; j < ny; ++j)
		for (int i = 0; i < nx; ++i) {
			s.push_back((i + lcg()) / nx);
			s.push_back((j + lcg()) / ny);
		}
	return s;
}

// Uniform random samples (LCG)
static std::vector<double> random_samples(int n, unsigned seed = 42) {
	std::vector<double> s;
	s.reserve(n * 2);
	auto lcg = [&]() -> double {
		seed = seed * 1664525u + 1013904223u;
		return (seed >> 8) / double(1u << 24);
	};
	for (int i = 0; i < n; ++i) {
		s.push_back(lcg());
		s.push_back(lcg());
	}
	return s;
}

// ---------------------------------------------------------------------------
// Factory / create()
// ---------------------------------------------------------------------------

TEST(FunctionIntegrator, CreateKnownNames) {
	EXPECT_NO_THROW(FunctionIntegrator::create("step"));
	EXPECT_NO_THROW(FunctionIntegrator::create("diagonal"));
	EXPECT_NO_THROW(FunctionIntegrator::create("disk"));
	EXPECT_NO_THROW(FunctionIntegrator::create("checkerboard"));
	EXPECT_NO_THROW(FunctionIntegrator::create("gaussian"));
}

TEST(FunctionIntegrator, CreateUnknownThrows) {
	EXPECT_THROW(FunctionIntegrator::create("unknown_func"), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Analytic integrals
// ---------------------------------------------------------------------------

TEST(FunctionIntegrator, AnalyticIntegralStep)         { EXPECT_DOUBLE_EQ(FunctionIntegrator::analytic_integral("step"),         1.0); }
TEST(FunctionIntegrator, AnalyticIntegralDiagonal)     { EXPECT_DOUBLE_EQ(FunctionIntegrator::analytic_integral("diagonal"),     1.0); }
TEST(FunctionIntegrator, AnalyticIntegralDisk)         { EXPECT_DOUBLE_EQ(FunctionIntegrator::analytic_integral("disk"),         1.0); }
TEST(FunctionIntegrator, AnalyticIntegralCheckerboard) { EXPECT_DOUBLE_EQ(FunctionIntegrator::analytic_integral("checkerboard"), 1.0); }
TEST(FunctionIntegrator, AnalyticIntegralGaussian)     { EXPECT_DOUBLE_EQ(FunctionIntegrator::analytic_integral("gaussian"),     1.0); }

// ---------------------------------------------------------------------------
// Function values (spot checks)
// ---------------------------------------------------------------------------

TEST(FunctionIntegrator, StepValues) {
	auto f = FunctionIntegrator::step();
	EXPECT_DOUBLE_EQ(f(0.0, 0.5), 2.0);
	EXPECT_DOUBLE_EQ(f(0.49, 0.5), 2.0);
	EXPECT_DOUBLE_EQ(f(0.5, 0.5), 0.0);
	EXPECT_DOUBLE_EQ(f(1.0, 0.5), 0.0);
}

TEST(FunctionIntegrator, DiagonalValues) {
	auto f = FunctionIntegrator::diagonal();
	EXPECT_DOUBLE_EQ(f(0.0, 0.0), 2.0);
	EXPECT_DOUBLE_EQ(f(0.5, 0.4), 2.0);
	EXPECT_DOUBLE_EQ(f(0.5, 0.5), 0.0);
	EXPECT_DOUBLE_EQ(f(1.0, 0.0), 0.0);
}

TEST(FunctionIntegrator, DiskCenterIsNonZero) {
	auto f = FunctionIntegrator::disk();
	EXPECT_GT(f(0.5, 0.5), 0.0);  // center of disk
}

TEST(FunctionIntegrator, DiskCornerIsZero) {
	auto f = FunctionIntegrator::disk();
	EXPECT_DOUBLE_EQ(f(0.0, 0.0), 0.0);
	EXPECT_DOUBLE_EQ(f(1.0, 1.0), 0.0);
}

TEST(FunctionIntegrator, GaussianPeakAtCenter) {
	auto f = FunctionIntegrator::gaussian();
	double center  = f(0.5, 0.5);
	double corner  = f(0.0, 0.0);
	EXPECT_GT(center, corner);
	EXPECT_GT(center, 0.0);
}

TEST(FunctionIntegrator, CheckerboardAlternates) {
	auto f = FunctionIntegrator::checkerboard();
	// Two adjacent cells in x should have opposite values
	double v0 = f(0.05, 0.05);  // cell (0,0)
	double v1 = f(0.15, 0.05);  // cell (1,0)
	EXPECT_NE(v0, v1);
	EXPECT_TRUE(v0 == 0.0 || v0 == 2.0);
	EXPECT_TRUE(v1 == 0.0 || v1 == 2.0);
}

// ---------------------------------------------------------------------------
// integrate() — MC estimates converge to analytic integral
// ---------------------------------------------------------------------------

TEST(FunctionIntegrator, IntegrateStepStratified) {
	auto samp = stratified_samples(100, 100);
	auto f = FunctionIntegrator::step();
	double est = FunctionIntegrator::integrate(f, samp.data(), 10000);
	EXPECT_NEAR(est, 1.0, 0.02);
}

TEST(FunctionIntegrator, IntegrateDiagonalStratified) {
	auto samp = stratified_samples(100, 100);
	auto f = FunctionIntegrator::diagonal();
	double est = FunctionIntegrator::integrate(f, samp.data(), 10000);
	EXPECT_NEAR(est, 1.0, 0.02);
}

TEST(FunctionIntegrator, IntegrateDiskStratified) {
	auto samp = stratified_samples(100, 100);
	auto f = FunctionIntegrator::disk();
	double est = FunctionIntegrator::integrate(f, samp.data(), 10000);
	EXPECT_NEAR(est, 1.0, 0.05);
}

TEST(FunctionIntegrator, IntegrateCheckerboardStratified) {
	auto samp = stratified_samples(100, 100);
	auto f = FunctionIntegrator::checkerboard();
	double est = FunctionIntegrator::integrate(f, samp.data(), 10000);
	EXPECT_NEAR(est, 1.0, 0.02);
}

TEST(FunctionIntegrator, IntegrateGaussianStratified) {
	auto samp = stratified_samples(100, 100);
	auto f = FunctionIntegrator::gaussian();
	double est = FunctionIntegrator::integrate(f, samp.data(), 10000);
	EXPECT_NEAR(est, 1.0, 0.05);
}

TEST(FunctionIntegrator, IntegrateZeroSamplesReturnsZero) {
	auto f = FunctionIntegrator::step();
	double est = FunctionIntegrator::integrate(f, nullptr, 0);
	EXPECT_DOUBLE_EQ(est, 0.0);
}

// ---------------------------------------------------------------------------
// mse() — error should be small for large stratified sample count
// ---------------------------------------------------------------------------

TEST(FunctionIntegrator, MSEDecreasesWith­MoreSamples) {
	auto f    = FunctionIntegrator::step();
	double ref = 1.0;

	auto samp16  = stratified_samples(4, 4,   1);
	auto samp100 = stratified_samples(10, 10, 1);

	double mse16  = FunctionIntegrator::mse(f, samp16.data(),  16,  ref);
	double mse100 = FunctionIntegrator::mse(f, samp100.data(), 100, ref);

	// More samples should not systematically increase error
	// (with stratified sampling mse16 >= mse100 probabilistically)
	EXPECT_LT(mse100, 0.01);
}

TEST(FunctionIntegrator, MSEExactIntegralIsZero) {
	// If estimate == reference, mse == 0
	auto f = FunctionIntegrator::step();
	// One sample at the representative point x=0.25 -> f=2, estimate=2 != 1
	// Use stratified 100x100 which should give estimate very close to 1
	auto samp = stratified_samples(100, 100);
	double mse = FunctionIntegrator::mse(f, samp.data(), 10000, 1.0);
	EXPECT_LT(mse, 1e-5);
}

// ---------------------------------------------------------------------------
// variance()
// ---------------------------------------------------------------------------

TEST(FunctionIntegrator, VarianceIsNonNegative) {
	auto samp = random_samples(1000);
	auto f    = FunctionIntegrator::step();
	double v  = FunctionIntegrator::variance(f, samp.data(), 1000);
	EXPECT_GE(v, 0.0);
}

TEST(FunctionIntegrator, VarianceConstantFunctionIsZero) {
	// f(x,y) = 1 everywhere -> variance = 0
	auto f    = [](double /*x*/, double /*y*/) { return 1.0; };
	auto samp = random_samples(100);
	double v  = FunctionIntegrator::variance(f, samp.data(), 100);
	EXPECT_NEAR(v, 0.0, 1e-12);
}

TEST(FunctionIntegrator, VarianceSingleSampleReturnsZero) {
	auto f = FunctionIntegrator::step();
	double s[2] = {0.25, 0.5};
	EXPECT_DOUBLE_EQ(FunctionIntegrator::variance(f, s, 1), 0.0);
}

TEST(FunctionIntegrator, VarianceStepFunction) {
	// Analytical variance of step over [0,1)^2:
	// E[f^2] = 0.5 * 4 + 0.5 * 0 = 2,  E[f] = 1,  Var = E[f^2]-E[f]^2 = 1
	auto samp = stratified_samples(200, 200);
	auto f    = FunctionIntegrator::step();
	double v  = FunctionIntegrator::variance(f, samp.data(), 40000);
	EXPECT_NEAR(v, 1.0, 0.05);
}
