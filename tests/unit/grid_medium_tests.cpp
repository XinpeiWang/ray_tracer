// ---------------------------------------------------------------------------
// grid_medium_tests.cpp
// Unit tests for grid_medium.h
//
// Mirrors pbrt-v4 MajorantGrid and DDAMajorantIterator (media.h).
// Tests cover:
//   Bounds3         : offset, diagonal, ray intersection
//   MajorantGrid    : lookup/set, voxel_bounds
//   DDAMajorantIterator: axis-aligned rays, diagonal ray, segment coverage,
//                        uniform-density transmittance consistency
// ---------------------------------------------------------------------------
#include <gtest/gtest.h>
#include "../../src/shared/grid_medium.h"

#include <cmath>
#include <numeric>

// ============================================================
// Bounds3 tests
// ============================================================

TEST(Bounds3, DiagonalAndOffset) {
	Bounds3<double> b(0.0, 0.0, 0.0, 2.0, 4.0, 8.0);
	EXPECT_DOUBLE_EQ(b.diagonal(0), 2.0);
	EXPECT_DOUBLE_EQ(b.diagonal(1), 4.0);
	EXPECT_DOUBLE_EQ(b.diagonal(2), 8.0);

	EXPECT_DOUBLE_EQ(b.offset(1.0, 0), 0.5);
	EXPECT_DOUBLE_EQ(b.offset(2.0, 1), 0.5);
	EXPECT_DOUBLE_EQ(b.offset(8.0, 2), 1.0);
	EXPECT_DOUBLE_EQ(b.offset(0.0, 2), 0.0);
}

TEST(Bounds3, RayIntersectHit) {
	Bounds3<double> b(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	double orig[3] = {0.5, 0.5, -1.0};
	double dir[3]  = {0.0, 0.0,  1.0};
	double tMin = 0.0, tMax = 0.0;
	bool hit = b.intersect_ray(orig, dir, 1e10, &tMin, &tMax);
	EXPECT_TRUE(hit);
	EXPECT_NEAR(tMin, 1.0, 1e-10);
	EXPECT_NEAR(tMax, 2.0, 1e-9);
}

TEST(Bounds3, RayIntersectMiss) {
	Bounds3<double> b(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	double orig[3] = {2.0, 2.0, 0.0};
	double dir[3]  = {0.0, 0.0, 1.0};
	double tMin = 0.0, tMax = 0.0;
	bool hit = b.intersect_ray(orig, dir, 1e10, &tMin, &tMax);
	EXPECT_FALSE(hit);
}

TEST(Bounds3, RayIntersectFromInside) {
	Bounds3<double> b(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	double orig[3] = {0.5, 0.5, 0.5};
	double dir[3]  = {1.0, 0.0, 0.0};
	double tMin = 0.0, tMax = 0.0;
	bool hit = b.intersect_ray(orig, dir, 1e10, &tMin, &tMax);
	EXPECT_TRUE(hit);
	EXPECT_NEAR(tMin, 0.0, 1e-10);
	EXPECT_NEAR(tMax, 0.5, 1e-9);
}

// ============================================================
// MajorantGrid tests
// ============================================================

TEST(MajorantGrid, LookupAndSet) {
	Bounds3<double> b(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	MajorantGrid<double> grid(b, 4, 4, 4);

	// All zero initially
	EXPECT_DOUBLE_EQ(grid.lookup(0, 0, 0), 0.0);
	EXPECT_DOUBLE_EQ(grid.lookup(3, 3, 3), 0.0);

	grid.set(1, 2, 3, 0.75);
	EXPECT_DOUBLE_EQ(grid.lookup(1, 2, 3), 0.75);

	// Others still zero
	EXPECT_DOUBLE_EQ(grid.lookup(0, 0, 0), 0.0);
	EXPECT_DOUBLE_EQ(grid.lookup(1, 2, 2), 0.0);
}

TEST(MajorantGrid, VoxelBounds) {
	Bounds3<double> b(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	MajorantGrid<double> grid(b, 2, 2, 2);

	double pMin[3], pMax[3];
	grid.voxel_bounds(0, 0, 0, pMin, pMax);
	EXPECT_NEAR(pMin[0], 0.0, 1e-15);
	EXPECT_NEAR(pMax[0], 0.5, 1e-15);

	grid.voxel_bounds(1, 1, 1, pMin, pMax);
	EXPECT_NEAR(pMin[0], 0.5, 1e-15);
	EXPECT_NEAR(pMax[0], 1.0, 1e-15);
}

TEST(MajorantGrid, FillDensities) {
	Bounds3<double> b(0.0, 0.0, 0.0, 3.0, 1.0, 1.0);
	MajorantGrid<double> grid(b, 3, 1, 1);
	grid.set(0, 0, 0, 1.0);
	grid.set(1, 0, 0, 2.0);
	grid.set(2, 0, 0, 3.0);

	EXPECT_DOUBLE_EQ(grid.lookup(0, 0, 0), 1.0);
	EXPECT_DOUBLE_EQ(grid.lookup(1, 0, 0), 2.0);
	EXPECT_DOUBLE_EQ(grid.lookup(2, 0, 0), 3.0);
}

// ============================================================
// DDAMajorantIterator tests
// ============================================================

// Helper: collect all segments from an iterator
template<typename T>
std::vector<RayMajorantSegment<T>> collect_segments(DDAMajorantIterator<T>& it) {
	std::vector<RayMajorantSegment<T>> segs;
	for (;;) {
		auto seg = it.Next();
		if (!seg.has_value()) break;
		segs.push_back(seg.value());
		if (segs.size() > 10000) break; // safety
	}
	return segs;
}

// Single-voxel grid: iterator must return exactly one segment covering [tMin,tMax].
TEST(DDAMajorantIterator, SingleVoxelGrid) {
	Bounds3<double> b(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	MajorantGrid<double> grid(b, 1, 1, 1);
	grid.set(0, 0, 0, 0.5);

	// X-axis aligned ray through the middle
	Ray3<double> ray(0.5, 0.5, -1.0,  0.0, 0.0, 1.0);
	// Intersect: enters at t=1, exits at t=2
	double tMin, tMax;
	bool hit = b.intersect_ray(ray.o, ray.d, 1e10, &tMin, &tMax);
	ASSERT_TRUE(hit);

	DDAMajorantIterator<double> it(ray, tMin, tMax, &grid, 1.0);
	auto segs = collect_segments(it);

	ASSERT_EQ(segs.size(), 1u);
	EXPECT_NEAR(segs[0].tMin,     tMin, 1e-10);
	EXPECT_NEAR(segs[0].tMax,     tMax, 1e-9);
	EXPECT_NEAR(segs[0].sigma_maj, 0.5, 1e-12);
}

// X-axis aligned ray through a 1x1x1 grid of N voxels along X:
// the iterator must visit all N voxels in order.
TEST(DDAMajorantIterator, XAxisAlignedNVoxels) {
	const int N = 8;
	Bounds3<double> b(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	MajorantGrid<double> grid(b, N, 1, 1);
	for (int i = 0; i < N; ++i)
		grid.set(i, 0, 0, static_cast<double>(i + 1));

	// Ray along X from outside, through the full box
	Ray3<double> ray(-0.5, 0.5, 0.5,  1.0, 0.0, 0.0);
	double tMin, tMax;
	bool hit = b.intersect_ray(ray.o, ray.d, 1e10, &tMin, &tMax);
	ASSERT_TRUE(hit);
	EXPECT_NEAR(tMin, 0.5, 1e-10);
	EXPECT_NEAR(tMax, 1.5, 1e-9);

	DDAMajorantIterator<double> it(ray, tMin, tMax, &grid, 1.0);
	auto segs = collect_segments(it);

	ASSERT_EQ(segs.size(), static_cast<size_t>(N));

	// Segments must be contiguous and cover [tMin, tMax]
	EXPECT_NEAR(segs.front().tMin, tMin, 1e-10);
	EXPECT_NEAR(segs.back().tMax,  tMax, 1e-9);
	for (size_t i = 1; i < segs.size(); ++i)
		EXPECT_NEAR(segs[i].tMin, segs[i-1].tMax, 1e-12);

	// Each segment's majorant should match expected density
	for (int i = 0; i < N; ++i)
		EXPECT_NEAR(segs[i].sigma_maj, static_cast<double>(i + 1), 1e-12);
}

// Ray along Y axis
TEST(DDAMajorantIterator, YAxisAligned) {
	const int N = 4;
	Bounds3<double> b(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	MajorantGrid<double> grid(b, 1, N, 1);
	for (int j = 0; j < N; ++j)
		grid.set(0, j, 0, static_cast<double>(j + 1));

	Ray3<double> ray(0.5, -0.5, 0.5,  0.0, 1.0, 0.0);
	double tMin, tMax;
	bool hit = b.intersect_ray(ray.o, ray.d, 1e10, &tMin, &tMax);
	ASSERT_TRUE(hit);

	DDAMajorantIterator<double> it(ray, tMin, tMax, &grid, 1.0);
	auto segs = collect_segments(it);

	ASSERT_EQ(segs.size(), static_cast<size_t>(N));
	for (int j = 0; j < N; ++j)
		EXPECT_NEAR(segs[j].sigma_maj, static_cast<double>(j + 1), 1e-12);
}

// Diagonal ray: must still produce contiguous, non-overlapping segments
// that cover [tMin, tMax] completely.
TEST(DDAMajorantIterator, DiagonalRayCoverage) {
	Bounds3<double> b(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	MajorantGrid<double> grid(b, 4, 4, 4);
	// Fill all voxels with density 1
	for (int x = 0; x < 4; ++x)
		for (int y = 0; y < 4; ++y)
			for (int z = 0; z < 4; ++z)
				grid.set(x, y, z, 1.0);

	// Diagonal ray along (1,1,1)
	double sq3inv = 1.0 / std::sqrt(3.0);
	Ray3<double> ray(0.0, 0.0, 0.0,  sq3inv, sq3inv, sq3inv);
	double tMin = 0.0;
	double tMax = std::sqrt(3.0);  // exits at (1,1,1)

	DDAMajorantIterator<double> it(ray, tMin, tMax, &grid, 1.0);
	auto segs = collect_segments(it);

	// Must have at least one segment
	ASSERT_GT(segs.size(), 0u);

	// Contiguous
	for (size_t i = 1; i < segs.size(); ++i)
		EXPECT_NEAR(segs[i].tMin, segs[i-1].tMax, 1e-11);

	// Cover full interval
	EXPECT_NEAR(segs.front().tMin, tMin, 1e-11);
	EXPECT_NEAR(segs.back().tMax,  tMax, 1e-9);

	// Total length sum matches interval length
	double totalLen = 0.0;
	for (const auto& seg : segs) totalLen += seg.tMax - seg.tMin;
	EXPECT_NEAR(totalLen, tMax - tMin, 1e-9);
}

// Negative-direction ray (stepping backwards)
TEST(DDAMajorantIterator, NegativeDirectionRay) {
	const int N = 5;
	Bounds3<double> b(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	MajorantGrid<double> grid(b, N, 1, 1);
	for (int i = 0; i < N; ++i)
		grid.set(i, 0, 0, static_cast<double>(N - i)); // densities: 5,4,3,2,1

	// Ray traveling in -X direction, entering from the positive-X side
	Ray3<double> ray(1.5, 0.5, 0.5,  -1.0, 0.0, 0.0);
	double tMin, tMax;
	bool hit = b.intersect_ray(ray.o, ray.d, 1e10, &tMin, &tMax);
	ASSERT_TRUE(hit);

	DDAMajorantIterator<double> it(ray, tMin, tMax, &grid, 1.0);
	auto segs = collect_segments(it);

	ASSERT_EQ(segs.size(), static_cast<size_t>(N));

	// Contiguous
	for (size_t i = 1; i < segs.size(); ++i)
		EXPECT_NEAR(segs[i].tMin, segs[i-1].tMax, 1e-12);

	// When traveling in -X, we first hit voxel N-1 (density=1), then N-2, ...
	for (int i = 0; i < N; ++i)
		EXPECT_NEAR(segs[i].sigma_maj, static_cast<double>(i + 1), 1e-12);
}

// Uniform grid: total transmittance from DDA segments == Beer-Lambert.
// pbrt-v4: this is the key correctness invariant — delta tracking uses these
// segments and the correct majorant ensures unbiased sampling.
TEST(DDAMajorantIterator, UniformGridTransmittanceConsistency) {
	const double sigma_t  = 2.0;   // world-space extinction
	const double density  = 1.0;   // voxel density (sigma_maj = sigma_t * density)
	const double grid_len = 3.0;   // path length through the medium

	Bounds3<double> b(0.0, 0.0, 0.0, grid_len, 1.0, 1.0);
	MajorantGrid<double> grid(b, 6, 1, 1);
	for (int i = 0; i < 6; ++i)
		grid.set(i, 0, 0, density);

	Ray3<double> ray(0.0, 0.5, 0.5,  1.0, 0.0, 0.0);
	double tMin, tMax;
	bool hit = b.intersect_ray(ray.o, ray.d, 1e10, &tMin, &tMax);
	ASSERT_TRUE(hit);
	EXPECT_NEAR(tMax - tMin, grid_len, 1e-9);

	DDAMajorantIterator<double> it(ray, tMin, tMax, &grid, sigma_t);
	double accumulated_optical_depth = 0.0;
	for (;;) {
		auto seg = it.Next();
		if (!seg.has_value()) break;
		accumulated_optical_depth += seg->sigma_maj * (seg->tMax - seg->tMin);
	}

	double expected_od = sigma_t * density * grid_len;  // = 6.0
	EXPECT_NEAR(accumulated_optical_depth, expected_od, 1e-9);

	// Beer-Lambert: Tr = exp(-optical_depth)
	double Tr = std::exp(-accumulated_optical_depth);
	EXPECT_NEAR(Tr, std::exp(-expected_od), 1e-12);
}

// sigma_t scaling: iterator must scale voxel density by sigma_t
TEST(DDAMajorantIterator, SigmaScaling) {
	Bounds3<double> b(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	MajorantGrid<double> grid(b, 2, 1, 1);
	grid.set(0, 0, 0, 0.3);
	grid.set(1, 0, 0, 0.7);

	const double sigma_t = 5.0;
	Ray3<double> ray(-0.5, 0.5, 0.5,  1.0, 0.0, 0.0);
	double tMin, tMax;
	b.intersect_ray(ray.o, ray.d, 1e10, &tMin, &tMax);

	DDAMajorantIterator<double> it(ray, tMin, tMax, &grid, sigma_t);
	auto segs = collect_segments(it);

	ASSERT_EQ(segs.size(), 2u);
	EXPECT_NEAR(segs[0].sigma_maj, sigma_t * 0.3, 1e-12);
	EXPECT_NEAR(segs[1].sigma_maj, sigma_t * 0.7, 1e-12);
}

// Empty iterator (tMin >= tMax) returns no segments
TEST(DDAMajorantIterator, EmptyRange) {
	Bounds3<double> b(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
	MajorantGrid<double> grid(b, 2, 2, 2);
	Ray3<double> ray(0.5, 0.5, -1.0,  0.0, 0.0, 1.0);
	// Pass tMin == tMax: should produce no segments
	DDAMajorantIterator<double> it(ray, 1.0, 1.0, &grid, 1.0);
	auto segs = collect_segments(it);
	EXPECT_EQ(segs.size(), 0u);
}
