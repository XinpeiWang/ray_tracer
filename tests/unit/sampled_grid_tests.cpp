// ---------------------------------------------------------------------------
// sampled_grid_tests.cpp
// Unit tests for sampled_grid.h
//
// Mirrors pbrt-v4 SampledGrid<T> (util/containers.h) and
// GridMedium::SamplePoint + majorant construction (media.h).
//
// Tests cover:
//   SampledGrid  : direct voxel lookup, trilinear interpolation (corners,
//                  center, along axes), out-of-bounds clamping, MaxValue,
//                  convert overload
//   GridMediumData: construction, sample_point, sample_sigma,
//                   majorant_grid voxel values, DDA integration
// ---------------------------------------------------------------------------
#include <gtest/gtest.h>
#include "../../src/shared/sampled_grid.h"
#include "../../src/shared/grid_medium.h"

#include <cmath>
#include <numeric>
#include <vector>

// ============================================================
// SampledGrid<double> — direct voxel access
// ============================================================

TEST(SampledGrid, DirectLookupInBounds) {
	// 2×2×2 grid, values 0..7
	std::vector<double> v = {0,1,2,3,4,5,6,7};
	SampledGrid<double> g(v, 2, 2, 2);
	// layout: (z*ny+y)*nx+x  => (0*2+0)*2+0=0, (0*2+0)*2+1=1, ...
	EXPECT_DOUBLE_EQ(g.lookup(0,0,0), 0.0);
	EXPECT_DOUBLE_EQ(g.lookup(1,0,0), 1.0);
	EXPECT_DOUBLE_EQ(g.lookup(0,1,0), 2.0);
	EXPECT_DOUBLE_EQ(g.lookup(1,1,0), 3.0);
	EXPECT_DOUBLE_EQ(g.lookup(0,0,1), 4.0);
	EXPECT_DOUBLE_EQ(g.lookup(1,0,1), 5.0);
	EXPECT_DOUBLE_EQ(g.lookup(0,1,1), 6.0);
	EXPECT_DOUBLE_EQ(g.lookup(1,1,1), 7.0);
}

TEST(SampledGrid, DirectLookupOutOfBoundsReturnsZero) {
	std::vector<double> v = {1.0, 2.0, 3.0, 4.0};
	SampledGrid<double> g(v, 2, 2, 1);
	EXPECT_DOUBLE_EQ(g.lookup(-1, 0, 0), 0.0);
	EXPECT_DOUBLE_EQ(g.lookup( 2, 0, 0), 0.0);
	EXPECT_DOUBLE_EQ(g.lookup( 0, 2, 0), 0.0);
	EXPECT_DOUBLE_EQ(g.lookup( 0, 0, 1), 0.0);
}

// ============================================================
// SampledGrid<double> — trilinear lookup
// ============================================================

// Uniform grid: lookup at voxel centers returns the constant value.
// Note: at p=0 and p=1 the pbrt-v4 convention (pSamples = p*n - 0.5) blends
// with an implicit zero-border, so boundary values are <constant.
// Voxel centers for N=4 are at (2i+1)/(2*N), i.e. 0.125, 0.375, 0.625, 0.875.
TEST(SampledGrid, TrilinearUniformGrid) {
	const int N = 4;
	std::vector<double> v(N * N * N, 3.14);
	SampledGrid<double> g(v, N, N, N);
	// At interior voxel centers: no blending with the zero-border
	EXPECT_NEAR(g.lookup(0.5, 0.5, 0.5), 3.14, 1e-12);
	EXPECT_NEAR(g.lookup(0.125, 0.375, 0.625), 3.14, 1e-12);
	EXPECT_NEAR(g.lookup(0.875, 0.875, 0.875), 3.14, 1e-12);
	// Boundary p=0/1 blends with zero-border (result < 3.14)
	EXPECT_LT(g.lookup(0.0, 0.5, 0.5), 3.14);
	EXPECT_LT(g.lookup(1.0, 1.0, 1.0), 3.14);
}

// 1×1×1 grid: the voxel center is at p=(0.5, 0.5, 0.5).
// pbrt-v4: pSamples = p*1 - 0.5 = 0 at p=0.5 → no OOB blending → returns full value.
// At p=0: pSamples = -0.5 → pi=(-1,-1,-1), only corner (0,0,0) contributes
//   with weight 0.5^3 = 0.125 per axis → 0.125*5 rounded for each axis blend = 0.625.
TEST(SampledGrid, SingleVoxelLookup) {
	std::vector<double> v = {5.0};
	SampledGrid<double> g(v, 1, 1, 1);
	EXPECT_NEAR(g.lookup(0.5, 0.5, 0.5), 5.0, 1e-12);
	// At boundaries, value blends toward zero (implicit zero-border)
	EXPECT_NEAR(g.lookup(0.0, 0.0, 0.0), 0.625, 1e-12);  // 5 * 0.5 * 0.5 * 0.5 * 2 per-axis
	EXPECT_NEAR(g.lookup(1.0, 1.0, 1.0), 0.625, 1e-12);
}

// Linear ramp along X: Lookup(p) should be exactly linear in p.x
TEST(SampledGrid, LinearRampAlongX) {
	// 4×1×1 grid, values 0,1,2,3
	std::vector<double> v = {0.0, 1.0, 2.0, 3.0};
	SampledGrid<double> g(v, 4, 1, 1);

	// At voxel centers: x=1/8, 3/8, 5/8, 7/8 (center of each 1/4 cell)
	EXPECT_NEAR(g.lookup(1.0/8.0, 0.5, 0.5), 0.0, 1e-9);
	EXPECT_NEAR(g.lookup(3.0/8.0, 0.5, 0.5), 1.0, 1e-9);
	EXPECT_NEAR(g.lookup(5.0/8.0, 0.5, 0.5), 2.0, 1e-9);
	EXPECT_NEAR(g.lookup(7.0/8.0, 0.5, 0.5), 3.0, 1e-9);

	// Between centers: should interpolate
	EXPECT_NEAR(g.lookup(2.0/8.0, 0.5, 0.5), 0.5, 1e-9);
	EXPECT_NEAR(g.lookup(4.0/8.0, 0.5, 0.5), 1.5, 1e-9);
}

// Linear ramp along Z: analogous test
TEST(SampledGrid, LinearRampAlongZ) {
	// 1×1×4 grid, values 0,1,2,3
	std::vector<double> v = {0.0, 1.0, 2.0, 3.0};
	SampledGrid<double> g(v, 1, 1, 4);

	EXPECT_NEAR(g.lookup(0.5, 0.5, 1.0/8.0), 0.0, 1e-9);
	EXPECT_NEAR(g.lookup(0.5, 0.5, 3.0/8.0), 1.0, 1e-9);
	EXPECT_NEAR(g.lookup(0.5, 0.5, 5.0/8.0), 2.0, 1e-9);
	EXPECT_NEAR(g.lookup(0.5, 0.5, 7.0/8.0), 3.0, 1e-9);
}

// ============================================================
// SampledGrid<double> — MaxValue
// ============================================================

TEST(SampledGrid, MaxValueFullGrid) {
	std::vector<double> v = {1.0, 3.0, 2.0, 5.0, 4.0, 0.5};
	SampledGrid<double> g(v, 3, 2, 1);
	double mv = g.max_value(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	EXPECT_NEAR(mv, 5.0, 1e-12);
}

// MaxValue sub-region test.
// For a 4×1×1 grid with values {1,2,3,4}, the pSamples = p*4 - 0.5 convention
// means the first half (0..0.5) maps sample range -0.5..1.5 → voxels 0,1,2 (max=3).
// The second half (0.5..1.0) maps sample range 1.5..3.5 → voxels 1,2,3 (max=4).
TEST(SampledGrid, MaxValueSubRegion) {
	std::vector<double> v = {1.0, 2.0, 3.0, 4.0};
	SampledGrid<double> g(v, 4, 1, 1);
	// First half: touches voxels 0,1,2 → max = 3
	double mv = g.max_value(0.0, 0.0, 0.0, 0.5, 1.0, 1.0);
	EXPECT_NEAR(mv, 3.0, 1e-12);
	// Second half: touches voxels 1,2,3 → max = 4
	mv = g.max_value(0.5, 0.0, 0.0, 1.0, 1.0, 1.0);
	EXPECT_NEAR(mv, 4.0, 1e-12);
	// Full grid: max = 4
	mv = g.max_value(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	EXPECT_NEAR(mv, 4.0, 1e-12);
}

// ============================================================
// SampledGrid — convert overload
// ============================================================

TEST(SampledGrid, LookupConvert) {
	std::vector<double> v = {1.0, 2.0, 3.0, 4.0};
	SampledGrid<double> g(v, 2, 2, 1);
	// Convert doubles to floats
	auto result = g.lookup_convert(0.5, 0.5, 0.5, [](double d) { return static_cast<float>(d); });
	EXPECT_GT(result, 0.5f);
	EXPECT_LT(result, 5.0f);

	// Uniform grid at voxel center (0.25, 0.25, 0.25) for 2×2×2: no OOB blending
	// Voxel center of (0,0,0) in a 2×2×2 grid is at (0.5/2, 0.5/2, 0.5/2) = (0.25, 0.25, 0.25)
	std::vector<double> u(8, 7.0);
	SampledGrid<double> ug(u, 2, 2, 2);
	float r = ug.lookup_convert(0.25, 0.25, 0.25, [](double d) { return static_cast<float>(d); });
	EXPECT_NEAR(r, 7.0f, 1e-5f);
}

// ============================================================
// GridMediumData<double> — construction and sample_point
// ============================================================

// Test at voxel centers to avoid the implicit zero-border blending.
// For N=4, voxel centers are at (2i+1)/(2N), e.g. 3/8 for voxel 1.
TEST(GridMediumData, SamplePointUniformDensity) {
	const int N = 4;
	std::vector<double> density(N * N * N, 0.8);
	Bounds3<double> bounds(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	GridMediumData<double> gm(density, N, N, N, bounds, 1.0, 0.5, 0.0);
	// At the exact center of the grid
	EXPECT_NEAR(gm.sample_point(0.5, 0.5, 0.5), 0.8, 1e-9);
	// At the center of voxel (1,1,1): p = 3/8
	EXPECT_NEAR(gm.sample_point(3.0/8, 3.0/8, 3.0/8), 0.8, 1e-9);
}

TEST(GridMediumData, SampleSigmaScaling) {
	const int N = 2;
	std::vector<double> density(N * N * N, 0.5);
	Bounds3<double> bounds(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	const double sa = 2.0, ss = 3.0;
	GridMediumData<double> gm(density, N, N, N, bounds, sa, ss, 0.0);

	double sa_out, ss_out;
	gm.sample_sigma(0.5, 0.5, 0.5, sa_out, ss_out);
	EXPECT_NEAR(sa_out, sa * 0.5, 1e-12);
	EXPECT_NEAR(ss_out, ss * 0.5, 1e-12);
}

TEST(GridMediumData, ZeroDensityRegion) {
	// Grid: top half density=1, bottom half density=0
	const int N = 4;
	std::vector<double> density(N * N * N, 0.0);
	for (int z = N/2; z < N; ++z)
		for (int y = 0; y < N; ++y)
			for (int x = 0; x < N; ++x)
				density[(z * N + y) * N + x] = 1.0;

	Bounds3<double> bounds(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	GridMediumData<double> gm(density, N, N, N, bounds, 1.0, 1.0, 0.0);

	// Near z=0 → near zero density
	EXPECT_LT(gm.sample_point(0.5, 0.5, 0.1), 0.1);
	// Well inside the dense top half: center of voxel (0,0,3) is at z=7/8
	EXPECT_NEAR(gm.sample_point(0.5, 0.5, 7.0/8.0), 1.0, 1e-9);
}

// ============================================================
// GridMediumData — majorant grid correctness
// ============================================================

// Majorant grid voxel must be >= max density in its sub-region
TEST(GridMediumData, MajorantGridUpperBound) {
	const int N = 8;
	std::vector<double> density(N * N * N);
	// Fill with gradient 0..1 along X
	for (int z = 0; z < N; ++z)
		for (int y = 0; y < N; ++y)
			for (int x = 0; x < N; ++x)
				density[(z * N + y) * N + x] = static_cast<double>(x) / (N - 1);

	Bounds3<double> bounds(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	GridMediumData<double> gm(density, N, N, N, bounds, 1.0, 1.0, 0.0);

	// For every majorant voxel, the stored value must be >=
	// the actual maximum density in that region.
	const auto& mg = gm.majorant_grid;
	for (int mz = 0; mz < mg.res[2]; ++mz) {
		for (int my = 0; my < mg.res[1]; ++my) {
			for (int mx = 0; mx < mg.res[0]; ++mx) {
				double majorant = mg.lookup(mx, my, mz);
				// Get the normalized bounds of this majorant voxel
				double pMin[3], pMax[3];
				mg.voxel_bounds(mx, my, mz, pMin, pMax);
				// Sample a grid of points within this voxel and check
				for (int sx = 0; sx <= 4; ++sx)
					for (int sy = 0; sy <= 4; ++sy)
						for (int sz = 0; sz <= 4; ++sz) {
							double px = pMin[0] + (pMax[0]-pMin[0]) * sx / 4.0;
							double py = pMin[1] + (pMax[1]-pMin[1]) * sy / 4.0;
							double pz = pMin[2] + (pMax[2]-pMin[2]) * sz / 4.0;
							double d = gm.sample_point(px, py, pz);
							EXPECT_GE(majorant + 1e-9, d)
								<< "Majorant voxel (" << mx << "," << my << "," << mz
								<< ") = " << majorant << " < density " << d
								<< " at (" << px << "," << py << "," << pz << ")";
						}
			}
		}
	}
}

// ============================================================
// GridMediumData + DDAMajorantIterator integration
// ============================================================

// Confirm that the DDA iterator built from GridMediumData's majorant grid
// produces contiguous segments that cover the full ray interval,
// and that each segment's sigma_maj >= true sigma at any point in that segment.
TEST(GridMediumData, DDAIntegrationCoverageAndUpperBound) {
	const int N = 8;
	std::vector<double> density(N * N * N);
	// Checkerboard-ish: density = x/N
	for (int z = 0; z < N; ++z)
		for (int y = 0; y < N; ++y)
			for (int x = 0; x < N; ++x)
				density[(z * N + y) * N + x] = static_cast<double>(x + 1) / N;

	Bounds3<double> bounds(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	const double sigma_t = 2.0;
	GridMediumData<double> gm(density, N, N, N, bounds, sigma_t, 0.0, 0.0);

	// Ray along X
	Ray3<double> ray(0.0, 0.5, 0.5,  1.0, 0.0, 0.0);
	double tMin, tMax;
	bool hit = bounds.intersect_ray(ray.o, ray.d, 1e10, &tMin, &tMax);
	ASSERT_TRUE(hit);

	DDAMajorantIterator<double> it(ray, tMin, tMax, &gm.majorant_grid, sigma_t);

	double prevT = tMin;
	for (;;) {
		auto seg = it.Next();
		if (!seg.has_value()) break;
		// Contiguous
		EXPECT_NEAR(seg->tMin, prevT, 1e-11);
		prevT = seg->tMax;

		// Upper bound: majorant >= sigma_t * density at any point in segment
		int nSteps = 8;
		for (int s = 0; s <= nSteps; ++s) {
			double t = seg->tMin + (seg->tMax - seg->tMin) * s / nSteps;
			double px = ray.o[0] + t * ray.d[0];
			double py = ray.o[1] + t * ray.d[1];
			double pz = ray.o[2] + t * ray.d[2];
			double d = gm.sample_point(px, py, pz);
			EXPECT_GE(seg->sigma_maj + 1e-9, sigma_t * d);
		}
	}
	EXPECT_NEAR(prevT, tMax, 1e-9);
}

// Transmittance from DDA segments with uniform grid matches Beer-Lambert
TEST(GridMediumData, DDAUniformTransmittance) {
	const int N = 4;
	std::vector<double> density(N * N * N, 1.0);
	Bounds3<double> bounds(0.0, 0.0, 0.0, 2.0, 1.0, 1.0);
	const double sigma_t = 3.0;
	GridMediumData<double> gm(density, N, N, N, bounds, sigma_t, 0.0, 0.0);

	Ray3<double> ray(0.0, 0.5, 0.5,  1.0, 0.0, 0.0);
	double tMin, tMax;
	bounds.intersect_ray(ray.o, ray.d, 1e10, &tMin, &tMax);

	DDAMajorantIterator<double> it(ray, tMin, tMax, &gm.majorant_grid, sigma_t);
	double optical_depth = 0.0;
	for (;;) {
		auto seg = it.Next();
		if (!seg.has_value()) break;
		optical_depth += seg->sigma_maj * (seg->tMax - seg->tMin);
	}

	// With uniform density=1 and path length 2: OD = sigma_t * 1 * 2 = 6
	EXPECT_NEAR(optical_depth, sigma_t * 1.0 * (tMax - tMin), 1e-9);
	EXPECT_NEAR(std::exp(-optical_depth), std::exp(-sigma_t * (tMax - tMin)), 1e-9);
}
