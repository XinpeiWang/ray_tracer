//==============================================================================
// summed_area_table_tests.cpp
// Unit tests for SummedAreaTable and WindowedPiecewiseConstant2D
// Mirrors pbrt-v4 util/sampling.h behavior.
//==============================================================================

#include "gtest/gtest.h"
#include "../../src/shared/summed_area_table.h"
#include <cmath>
#include <numeric>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static Array2D<float> MakeUniform(int nx, int ny, float value = 1.f) {
	Array2D<float> a(nx, ny);
	for (int y = 0; y < ny; ++y)
		for (int x = 0; x < nx; ++x)
			a(x, y) = value;
	return a;
}

// Build a 4x4 grid where only cell (cx, cy) is non-zero.
static Array2D<float> MakeSingleSpike(int nx, int ny, int cx, int cy) {
	Array2D<float> a(nx, ny, 0.f);
	a(cx, cy) = 1.f;
	return a;
}

// ---------------------------------------------------------------------------
// SummedAreaTable -- Integral tests
// ---------------------------------------------------------------------------

TEST(SummedAreaTable, FullExtentEqualsOne_UniformUnit) {
	// Integral over [0,1]^2 of a uniform-1 field should be 1.
	auto f = MakeUniform(8, 8, 1.f);
	SummedAreaTable sat(f);
	SAT_Bounds2f full(SAT_Point2f(0.f, 0.f), SAT_Point2f(1.f, 1.f));
	EXPECT_NEAR(sat.Integral(full), 1.f, 1e-5f);
}

TEST(SummedAreaTable, HalfExtentEqualsHalf_UniformUnit) {
	// Left half of [0,1]^2
	auto f = MakeUniform(8, 8, 1.f);
	SummedAreaTable sat(f);
	SAT_Bounds2f half(SAT_Point2f(0.f, 0.f), SAT_Point2f(0.5f, 1.f));
	EXPECT_NEAR(sat.Integral(half), 0.5f, 1e-5f);
}

TEST(SummedAreaTable, QuarterExtentEqualsQuarter_UniformUnit) {
	auto f = MakeUniform(8, 8, 1.f);
	SummedAreaTable sat(f);
	SAT_Bounds2f q(SAT_Point2f(0.25f, 0.25f), SAT_Point2f(0.75f, 0.75f));
	EXPECT_NEAR(sat.Integral(q), 0.25f, 1e-4f);
}

TEST(SummedAreaTable, ScalesWithValue) {
	// Uniform constant c: integral over [0,1]^2 should be c.
	auto f = MakeUniform(4, 4, 3.f);
	SummedAreaTable sat(f);
	SAT_Bounds2f full(SAT_Point2f(0.f, 0.f), SAT_Point2f(1.f, 1.f));
	EXPECT_NEAR(sat.Integral(full), 3.f, 1e-4f);
}

TEST(SummedAreaTable, ZeroFunctionReturnsZero) {
	auto f = MakeUniform(4, 4, 0.f);
	SummedAreaTable sat(f);
	SAT_Bounds2f full(SAT_Point2f(0.f, 0.f), SAT_Point2f(1.f, 1.f));
	EXPECT_EQ(sat.Integral(full), 0.f);
}

TEST(SummedAreaTable, SpikeIntegralIsWeightedByArea) {
	// 4x4 grid, spike at (2,1). Each cell covers 1/16 of [0,1]^2.
	// Integral over the whole domain should be 1/16.
	auto f = MakeSingleSpike(4, 4, 2, 1);
	SummedAreaTable sat(f);
	SAT_Bounds2f full(SAT_Point2f(0.f, 0.f), SAT_Point2f(1.f, 1.f));
	EXPECT_NEAR(sat.Integral(full), 1.f / 16.f, 1e-5f);
}

TEST(SummedAreaTable, NonOverlappingWindowsPartition) {
	// Two halves should sum to the whole.
	auto f = MakeUniform(8, 8, 2.f);
	SummedAreaTable sat(f);
	SAT_Bounds2f left (SAT_Point2f(0.f, 0.f), SAT_Point2f(0.5f, 1.f));
	SAT_Bounds2f right(SAT_Point2f(0.5f, 0.f), SAT_Point2f(1.f, 1.f));
	SAT_Bounds2f full (SAT_Point2f(0.f, 0.f), SAT_Point2f(1.f, 1.f));
	EXPECT_NEAR(sat.Integral(left) + sat.Integral(right), sat.Integral(full), 1e-4f);
}

// Mirrors pbrt-v4 SummedArea.Constant — checks several partial bounds exactly
TEST(SummedAreaTable, Constant_PartialBounds) {
	auto f = MakeUniform(4, 4, 1.f);
	SummedAreaTable sat(f);
	EXPECT_NEAR(sat.Integral(SAT_Bounds2f(SAT_Point2f(0.f, 0.f), SAT_Point2f(1.f, 1.f))),   1.f,       1e-6f);
	EXPECT_NEAR(sat.Integral(SAT_Bounds2f(SAT_Point2f(0.f, 0.f), SAT_Point2f(1.f, 0.5f))),  0.5f,      1e-6f);
	EXPECT_NEAR(sat.Integral(SAT_Bounds2f(SAT_Point2f(0.f, 0.f), SAT_Point2f(0.5f, 1.f))),  0.5f,      1e-6f);
	EXPECT_NEAR(sat.Integral(SAT_Bounds2f(SAT_Point2f(0.f, 0.f), SAT_Point2f(0.25f, 0.75f))), 3.f/16.f, 1e-6f);
	EXPECT_NEAR(sat.Integral(SAT_Bounds2f(SAT_Point2f(0.5f, 0.25f), SAT_Point2f(0.75f, 1.f))), 3.f/16.f, 1e-6f);
}

// Mirrors pbrt-v4 SummedArea.Rect — exhaustive cell-aligned bounds, v(x,y)=x+y
TEST(SummedAreaTable, Rect_ExhaustiveCellAligned) {
	const int NX = 8, NY = 4;
	Array2D<float> v(NX, NY);
	for (int y = 0; y < NY; ++y)
		for (int x = 0; x < NX; ++x)
			v(x, y) = (float)(x + y);
	SummedAreaTable sat(v);

	for (int y0 = 0; y0 < NY; ++y0)
		for (int x0 = 0; x0 < NX; ++x0)
			for (int y1 = y0; y1 <= NY; ++y1)
				for (int x1 = x0; x1 <= NX; ++x1) {
					double mySum = 0;
					for (int y = y0; y < y1; ++y)
						for (int x = x0; x < x1; ++x)
							mySum += v(x, y);
					mySum /= (NX * NY);

					SAT_Bounds2f b(SAT_Point2f((float)x0 / NX, (float)y0 / NY),
								   SAT_Point2f((float)x1 / NX, (float)y1 / NY));
					EXPECT_NEAR(sat.Integral(b), (float)mySum, 1e-5f)
						<< "box [" << x0 << "," << y0 << "]-[" << x1 << "," << y1 << "]";
				}
}

// Simple LCG for deterministic random values (no external RNG dependency)
static uint32_t lcg_next(uint32_t& s) { s = s * 1664525u + 1013904223u; return s; }
static float lcg_float(uint32_t& s) { return (lcg_next(s) >> 8) * (1.f / (1 << 24)); }

// Mirrors pbrt-v4 SummedArea.Randoms — random dims/values/bounds vs brute-force
TEST(SummedAreaTable, Randoms_BruteForce) {
	struct Dims { int nx, ny; };
	Dims dims[] = {{1,6},{6,1},{12,19},{16,16},{49,2}};
	uint32_t seed = 0xdeadbeef;

	for (auto d : dims) {
		Array2D<float> v(d.nx, d.ny);
		for (int y = 0; y < d.ny; ++y)
			for (int x = 0; x < d.nx; ++x)
				v(x, y) = (float)(lcg_next(seed) % 32);
		SummedAreaTable sat(v);

		for (int i = 0; i < 50; ++i) {
			int x0 = (int)(lcg_float(seed) * d.nx);
			int y0 = (int)(lcg_float(seed) * d.ny);
			int x1 = (int)(lcg_float(seed) * d.nx);
			int y1 = (int)(lcg_float(seed) * d.ny);
			if (x0 > x1) std::swap(x0, x1);
			if (y0 > y1) std::swap(y0, y1);
			if (x0 == x1 || y0 == y1) continue;

			double ref = 0;
			for (int y = y0; y < y1; ++y)
				for (int x = x0; x < x1; ++x)
					ref += v(x, y);
			ref /= (d.nx * d.ny);

			SAT_Bounds2f bf(SAT_Point2f((float)x0 / d.nx, (float)y0 / d.ny),
							SAT_Point2f((float)x1 / d.nx, (float)y1 / d.ny));
			double s = sat.Integral(bf);
			if (ref != 0)
				EXPECT_LT(std::abs((ref - s) / ref), 1e-3)
					<< "dims " << d.nx << "x" << d.ny << " box [" << x0 << "," << y0 << "]-[" << x1 << "," << y1 << "]";
			else
				EXPECT_NEAR(s, 0.f, 1e-6f);
		}
	}
}

// Mirrors pbrt-v4 SummedArea.NonCellAligned — non-grid-aligned bounds vs Monte Carlo
TEST(SummedAreaTable, NonCellAligned_MonteCarlo) {
	struct Dims { int nx, ny; };
	Dims dims[] = {{12,19},{16,16},{49,2}};
	uint32_t seed = 0xc0ffee42;

	for (auto d : dims) {
		Array2D<float> v(d.nx, d.ny);
		for (int y = 0; y < d.ny; ++y)
			for (int x = 0; x < d.nx; ++x)
				v(x, y) = (float)(lcg_next(seed) % 32);
		SummedAreaTable sat(v);

		// Random non-cell-aligned bounds
		float bx0 = lcg_float(seed), by0 = lcg_float(seed);
		float bx1 = lcg_float(seed), by1 = lcg_float(seed);
		if (bx0 > bx1) std::swap(bx0, bx1);
		if (by0 > by1) std::swap(by0, by1);
		if (bx0 == bx1 || by0 == by1) continue;

		SAT_Bounds2f b(SAT_Point2f(bx0, by0), SAT_Point2f(bx1, by1));

		// Monte Carlo estimate using Halton-like stratified samples
		const int nSamples = 50000;
		double sampledSum = 0;
		for (int k = 0; k < nSamples; ++k) {
			// Van der Corput + Halton base-2/3 low-discrepancy
			float u = lcg_float(seed), vv = lcg_float(seed);
			float px = bx0 + u * (bx1 - bx0);
			float py = by0 + vv * (by1 - by0);
			int ix = std::min((int)(px * d.nx), d.nx - 1);
			int iy = std::min((int)(py * d.ny), d.ny - 1);
			sampledSum += v(ix, iy);
		}
		float area = (bx1 - bx0) * (by1 - by0);
		double sampledResult = sampledSum * area / nSamples;
		double s = sat.Integral(b);
		if (sampledResult > 1e-4)
			EXPECT_LT(std::abs((sampledResult - s) / sampledResult), 5e-2)
				<< "dims " << d.nx << "x" << d.ny;
	}
}

// ---------------------------------------------------------------------------
// WindowedPiecewiseConstant2D -- sampling tests
// ---------------------------------------------------------------------------

TEST(WindowedPiecewiseConstant2D, ZeroFunctionReturnsEmpty) {
	auto f = MakeUniform(4, 4, 0.f);
	WindowedPiecewiseConstant2D dist(f);
	SAT_Bounds2f full(SAT_Point2f(0.f, 0.f), SAT_Point2f(1.f, 1.f));
	float pdf = 0.f;
	auto result = dist.Sample(SAT_Point2f(0.5f, 0.5f), full, &pdf);
	EXPECT_FALSE(result.has_value());
}

TEST(WindowedPiecewiseConstant2D, SampleInsideWindow) {
	auto f = MakeUniform(8, 8, 1.f);
	WindowedPiecewiseConstant2D dist(f);
	SAT_Bounds2f b(SAT_Point2f(0.2f, 0.3f), SAT_Point2f(0.8f, 0.9f));
	float pdf = 0.f;
	auto p = dist.Sample(SAT_Point2f(0.5f, 0.5f), b, &pdf);
	ASSERT_TRUE(p.has_value());
	EXPECT_GE(p->x, b.pMin.x);
	EXPECT_LE(p->x, b.pMax.x);
	EXPECT_GE(p->y, b.pMin.y);
	EXPECT_LE(p->y, b.pMax.y);
	EXPECT_GT(pdf, 0.f);
}

TEST(WindowedPiecewiseConstant2D, UniformPDFIsConstant) {
	// For a uniform function f=1 over [0,1]^2, PDF at any point in b
	// should equal 1 / Integral(b) = 1 / area_fraction (since f=const=1).
	auto f = MakeUniform(8, 8, 1.f);
	WindowedPiecewiseConstant2D dist(f);
	SAT_Bounds2f b(SAT_Point2f(0.f, 0.f), SAT_Point2f(1.f, 1.f));

	// Expected: PDF = f(p) / Integral(b) = 1 / 1 = 1
	float pdf1 = dist.PDF(SAT_Point2f(0.25f, 0.25f), b);
	float pdf2 = dist.PDF(SAT_Point2f(0.75f, 0.75f), b);
	EXPECT_NEAR(pdf1, 1.f, 1e-4f);
	EXPECT_NEAR(pdf2, 1.f, 1e-4f);
}

TEST(WindowedPiecewiseConstant2D, PDFMatchesSamplePDF) {
	// Sample a point and verify the returned pdf matches PDF(p, b).
	auto f = MakeUniform(8, 8, 1.f);
	WindowedPiecewiseConstant2D dist(f);
	SAT_Bounds2f b(SAT_Point2f(0.1f, 0.1f), SAT_Point2f(0.9f, 0.9f));
	float samplePdf = 0.f;
	auto p = dist.Sample(SAT_Point2f(0.3f, 0.7f), b, &samplePdf);
	ASSERT_TRUE(p.has_value());
	float evalPdf = dist.PDF(*p, b);
	EXPECT_NEAR(samplePdf, evalPdf, 1e-4f);
}

TEST(WindowedPiecewiseConstant2D, SpikeConcentratesSamples) {
	// 4x4, spike at cell (3,3) (top-right). All samples over the full
	// domain should land in the top-right quadrant.
	auto f = MakeSingleSpike(4, 4, 3, 3);
	WindowedPiecewiseConstant2D dist(f);
	SAT_Bounds2f full(SAT_Point2f(0.f, 0.f), SAT_Point2f(1.f, 1.f));

	for (int i = 0; i <= 4; ++i) {
		float u = (i + 0.5f) / 5.f;
		for (int j = 0; j <= 4; ++j) {
			float v = (j + 0.5f) / 5.f;
			float pdf = 0.f;
			auto p = dist.Sample(SAT_Point2f(u, v), full, &pdf);
			ASSERT_TRUE(p.has_value());
			// Spike is in [0.75, 1.0] x [0.75, 1.0]
			EXPECT_GE(p->x, 0.74f);
			EXPECT_GE(p->y, 0.74f);
		}
	}
}

TEST(WindowedPiecewiseConstant2D, SmallWindowSampling) {
	auto f = MakeUniform(16, 16, 1.f);
	WindowedPiecewiseConstant2D dist(f);
	SAT_Bounds2f b(SAT_Point2f(0.4f, 0.4f), SAT_Point2f(0.6f, 0.6f));
	for (int i = 0; i < 5; ++i) {
		float u = (i + 0.5f) / 5.f;
		float pdf = 0.f;
		auto p = dist.Sample(SAT_Point2f(u, u), b, &pdf);
		ASSERT_TRUE(p.has_value());
		EXPECT_GE(p->x, b.pMin.x - 1e-4f);
		EXPECT_LE(p->x, b.pMax.x + 1e-4f);
		EXPECT_GE(p->y, b.pMin.y - 1e-4f);
		EXPECT_LE(p->y, b.pMax.y + 1e-4f);
	}
}
