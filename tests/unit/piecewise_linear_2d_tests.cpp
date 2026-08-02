//==============================================================================
// piecewise_linear_2d_tests.cpp
// Unit tests for PiecewiseLinear2D<Dimension>
// Mirrors pbrt-v4 util/sampling_test.cpp behavior where applicable.
//==============================================================================

#include "gtest/gtest.h"
#include "../../src/shared/piecewise_linear_2d.h"
#include <cmath>
#include <vector>
#include <numeric>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Uniform grid: all vertices = value
static std::vector<float> MakeUniform(int nx, int ny, float value = 1.f) {
	return std::vector<float>((size_t)(nx * ny), value);
}

// Grid with a single spike at (cx, cy)
static std::vector<float> MakeSpike(int nx, int ny, int cx, int cy) {
	std::vector<float> v((size_t)(nx * ny), 0.f);
	v[(size_t)(cy * nx + cx)] = 1.f;
	return v;
}

// Simple LCG
static uint32_t lcg_state = 0xabcdef01u;
static float lcg_float() {
	lcg_state = lcg_state * 1664525u + 1013904223u;
	return (lcg_state >> 8) * (1.f / (1 << 24));
}

// ---------------------------------------------------------------------------
// Dimension == 0 : build tests
// ---------------------------------------------------------------------------

TEST(PiecewiseLinear2D, BuildUniform_NormalizesToOne) {
	// For a uniform f=1 grid, Eval at any interior point should be ~1
	// (pdf of uniform distribution over [0,1]^2 is 1 everywhere)
	auto v = MakeUniform(8, 8, 1.f);
	PiecewiseLinear2D<0> d(v.data(), 8, 8);
	EXPECT_FALSE(d.IsEmpty());
	EXPECT_NEAR(d.Eval(0.5f, 0.5f), 1.f, 1e-5f);
	EXPECT_NEAR(d.Eval(0.1f, 0.9f), 1.f, 1e-5f);
}

TEST(PiecewiseLinear2D, BuildZero_DoesNotCrash) {
	auto v = MakeUniform(4, 4, 0.f);
	PiecewiseLinear2D<0> d(v.data(), 4, 4);
	EXPECT_FALSE(d.IsEmpty());
	EXPECT_NEAR(d.Eval(0.5f, 0.5f), 0.f, 1e-10f);
}

TEST(PiecewiseLinear2D, BuildNoCDF_EvalOnly) {
	auto v = MakeUniform(4, 4, 2.f);
	PiecewiseLinear2D<0> d(v.data(), 4, 4, /*normalize=*/true, /*build_cdf=*/false);
	// normalize=true, build_cdf=false: data is scaled so patches average to 1.
	// For uniform f=2, normalization factor is 1/sum(patches) = 1/(2*(nx-1)*(ny-1)).
	// Eval returns value * inv_patch_area. For uniform=2 with (nx-1)*(ny-1)=9 patches,
	// sum = 9 * 2 * 0.25 * 4 = 18 (bilinear average per patch * 4 = 2, patches = 9).
	// Check that Eval returns something finite and positive.
	float e = d.Eval(0.5f, 0.5f);
	EXPECT_GT(e, 0.f);
	EXPECT_FALSE(std::isnan(e));
}

// ---------------------------------------------------------------------------
// Dimension == 0 : Sample tests
// ---------------------------------------------------------------------------

TEST(PiecewiseLinear2D, SampleUniform_InUnitSquare) {
	auto v = MakeUniform(16, 16, 1.f);
	PiecewiseLinear2D<0> d(v.data(), 16, 16);
	for (int i = 0; i < 20; ++i) {
		float u0 = lcg_float(), u1 = lcg_float();
		PLSample s = d.Sample(u0, u1);
		EXPECT_GE(s.px, 0.f);
		EXPECT_LE(s.px, 1.f);
		EXPECT_GE(s.py, 0.f);
		EXPECT_LE(s.py, 1.f);
		EXPECT_GT(s.pdf, 0.f);
	}
}

TEST(PiecewiseLinear2D, SampleUniform_PDFIsOne) {
	// Uniform distribution: every sample should have pdf close to 1.
	auto v = MakeUniform(8, 8, 1.f);
	PiecewiseLinear2D<0> d(v.data(), 8, 8);
	for (int i = 0; i < 50; ++i) {
		float u0 = lcg_float(), u1 = lcg_float();
		PLSample s = d.Sample(u0, u1);
		EXPECT_NEAR(s.pdf, 1.f, 1e-4f) << "u=(" << u0 << "," << u1 << ")";
	}
}

TEST(PiecewiseLinear2D, SamplePDFMatchesEval) {
	// Sample pdf must equal Eval(p)
	auto v = MakeUniform(8, 8, 1.f);
	PiecewiseLinear2D<0> d(v.data(), 8, 8);
	for (int i = 0; i < 30; ++i) {
		PLSample s = d.Sample(lcg_float(), lcg_float());
		float ev = d.Eval(s.px, s.py);
		EXPECT_NEAR(s.pdf, ev, 1e-4f);
	}
}

TEST(PiecewiseLinear2D, SampleSpike_ConcentratesSamples) {
	// Single spike at (3,3) in a 4x4 grid.
	// Samples should cluster in the top-right quadrant [0.75,1]x[0.75,1].
	auto v = MakeSpike(4, 4, 3, 3);
	PiecewiseLinear2D<0> d(v.data(), 4, 4);
	for (int i = 0; i < 30; ++i) {
		PLSample s = d.Sample(lcg_float(), lcg_float());
		EXPECT_GE(s.px, 0.5f) << "spike should pull samples right";
		EXPECT_GE(s.py, 0.5f) << "spike should pull samples up";
	}
}

// ---------------------------------------------------------------------------
// Dimension == 0 : integral conservation via Monte Carlo
// ---------------------------------------------------------------------------

TEST(PiecewiseLinear2D, Integral_UniformIsOne) {
	// Importance-sample a uniform distribution: sample mean of 1/pdf should be 1.
	auto v = MakeUniform(8, 8, 1.f);
	PiecewiseLinear2D<0> d(v.data(), 8, 8);

	const int N = 10000;
	double sum = 0.0;
	for (int i = 0; i < N; ++i) {
		PLSample s = d.Sample(lcg_float(), lcg_float());
		if (s.pdf > 0.f) sum += 1.0 / s.pdf;
	}
	double est = sum / N;
	EXPECT_NEAR(est, 1.0, 0.02);  // within 2% of 1.0
}

TEST(PiecewiseLinear2D, Integral_NonUniform_MonteCarlo) {
	// f(x,y) = x+y (linear ramp).  Exact integral over [0,1]^2 = 1.
	const int NX = 16, NY = 16;
	std::vector<float> v((size_t)(NX * NY));
	for (int y = 0; y < NY; ++y)
		for (int x = 0; x < NX; ++x)
			v[(size_t)(y * NX + x)] = (float)x / (NX-1) + (float)y / (NY-1);
	PiecewiseLinear2D<0> d(v.data(), NX, NY);

	// Importance sampling: E[f(p)/pdf(p)] should = integral of f = 1 (after normalizing)
	const int N = 20000;
	double sum_inv = 0.0;
	for (int i = 0; i < N; ++i) {
		PLSample s = d.Sample(lcg_float(), lcg_float());
		if (s.pdf > 0.f) sum_inv += 1.0 / s.pdf;
	}
	// sum_inv / N should be approximately 1 (the distribution integrates to 1)
	EXPECT_NEAR(sum_inv / N, 1.0, 0.03);
}

// ---------------------------------------------------------------------------
// Dimension == 0 : Eval grid at vertices matches input (after normalizing)
// ---------------------------------------------------------------------------

TEST(PiecewiseLinear2D, Eval_VertexConsistency) {
	// For uniform f=c, all vertices after normalization are c/integral = 1.
	// Check corners and center.
	const float c = 3.f;
	auto v = MakeUniform(5, 5, c);
	PiecewiseLinear2D<0> d(v.data(), 5, 5);
	// After normalization, pdf = 1 everywhere for uniform input.
	EXPECT_NEAR(d.Eval(0.f, 0.f),  1.f, 1e-5f);
	EXPECT_NEAR(d.Eval(1.f, 1.f),  1.f, 1e-5f);
	EXPECT_NEAR(d.Eval(0.5f, 0.5f),1.f, 1e-5f);
}

TEST(PiecewiseLinear2D, Eval_LinearRamp_MatchesExpected) {
	// f(x,y) = x only (independent of y).
	// Integral = 0.5 over [0,1]^2.
	// After normalization, pdf(x,y) = f/integral = 2x.
	// At x=0.5, y=any: pdf should be ~1.0 (2 * 0.5).
	const int N = 9;
	std::vector<float> v((size_t)(N * N));
	for (int y = 0; y < N; ++y)
		for (int x = 0; x < N; ++x)
			v[(size_t)(y * N + x)] = (float)x / (N - 1);
	PiecewiseLinear2D<0> d(v.data(), N, N);
	EXPECT_NEAR(d.Eval(0.5f, 0.5f), 1.0f, 5e-3f);
	EXPECT_NEAR(d.Eval(0.5f, 0.1f), 1.0f, 5e-3f);
	// At x=0.25, pdf should be ~0.5
	EXPECT_NEAR(d.Eval(0.25f, 0.5f), 0.5f, 5e-3f);
}

// ---------------------------------------------------------------------------
// Dimension == 1 : conditional distribution test
// ---------------------------------------------------------------------------

TEST(PiecewiseLinear2D_Dim1, SampleInUnitSquare) {
	const int NX = 4, NY = 4;
	std::vector<float> v((size_t)(NX * NY * 2), 1.f);
	float param_vals[2] = {0.f, 1.f};
	int   param_res[1]  = {2};
	const float* pv[1]  = {param_vals};

	PiecewiseLinear2D<1> d(v.data(), NX, NY, param_res, pv);

	for (int i = 0; i < 20; ++i) {
		float param = lcg_float();
		PLSample s = d.Sample(lcg_float(), lcg_float(), param);
		EXPECT_GE(s.px, 0.f); EXPECT_LE(s.px, 1.f);
		EXPECT_GE(s.py, 0.f); EXPECT_LE(s.py, 1.f);
		EXPECT_GT(s.pdf, 0.f);
	}
}

TEST(PiecewiseLinear2D_Dim1, PDFIsOneForUniformSlices) {
	const int NX = 8, NY = 8;
	std::vector<float> v((size_t)(NX * NY * 2), 1.f);
	float param_vals[2] = {0.f, 1.f};
	int   param_res[1]  = {2};
	const float* pv[1]  = {param_vals};

	PiecewiseLinear2D<1> d(v.data(), NX, NY, param_res, pv);

	for (int i = 0; i < 30; ++i) {
		PLSample s = d.Sample(lcg_float(), lcg_float(), lcg_float());
		EXPECT_NEAR(s.pdf, 1.f, 1e-4f);
	}
}

// ---------------------------------------------------------------------------
// Invert round-trip tests
// ---------------------------------------------------------------------------

TEST(PiecewiseLinear2D, Invert_RoundTrip_Uniform) {
	// Sample then Invert must recover the original canonical sample (u0,u1).
	auto v = MakeUniform(8, 8, 1.f);
	PiecewiseLinear2D<0> d(v.data(), 8, 8);
	for (int i = 0; i < 50; ++i) {
		float u0 = lcg_float(), u1 = lcg_float();
		PLSample s  = d.Sample(u0, u1);
		PLSample inv = d.Invert(s.px, s.py);
		EXPECT_NEAR(inv.px, u0, 1e-3f) << "u0=" << u0 << " u1=" << u1;
		EXPECT_NEAR(inv.py, u1, 1e-3f) << "u0=" << u0 << " u1=" << u1;
		EXPECT_NEAR(inv.pdf, s.pdf, 1e-4f);
	}
}

TEST(PiecewiseLinear2D, Invert_RoundTrip_NonUniform) {
	// Linear ramp f(x,y) = x + y
	const int NX = 16, NY = 16;
	std::vector<float> v((size_t)(NX * NY));
	for (int y = 0; y < NY; ++y)
		for (int x = 0; x < NX; ++x)
			v[(size_t)(y * NX + x)] = (float)x / (NX-1) + (float)y / (NY-1);
	PiecewiseLinear2D<0> d(v.data(), NX, NY);
	for (int i = 0; i < 50; ++i) {
		float u0 = lcg_float(), u1 = lcg_float();
		PLSample s   = d.Sample(u0, u1);
		PLSample inv = d.Invert(s.px, s.py);
		EXPECT_NEAR(inv.px, u0, 2e-3f) << "i=" << i;
		EXPECT_NEAR(inv.py, u1, 2e-3f) << "i=" << i;
	}
}
