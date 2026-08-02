// ---------------------------------------------------------------------------
// rgb_grid_medium_tests.cpp
// Unit tests for rgb_grid_medium.h
//
// Mirrors pbrt-v4 RGBGridMedium (src/pbrt/media.h).
//
// Tests cover:
//   RGBGridMediumData construction (with/without optional grids)
//   sample_point: absent grids default to 1 (sigma) / 0 (Le)
//   sample_point: sigma_scale applied correctly
//   sample_point: per-channel values at known grid positions
//   Le_scale and is_emissive
//   Majorant grid: built with correct per-voxel maxima
//   DDA iteration via sample_ray
//   Float specialization
// ---------------------------------------------------------------------------
#include <gtest/gtest.h>
#include "../../src/shared/rgb_grid_medium.h"

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static Bounds3<double> unit_cube() {
	return Bounds3<double>(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
}

// Make a 2x2x2 flat voxel array filled with a constant value
static std::vector<double> const_vol(double v, int n = 8) {
	return std::vector<double>(n, v);
}

// ---------------------------------------------------------------------------
// Construction Tests
// ---------------------------------------------------------------------------

TEST(RGBGridMediumData, DefaultConstructorIsEmpty) {
	RGBGridMediumData<double> med;
	EXPECT_FALSE(med.sigma_a_grids.has_value());
	EXPECT_FALSE(med.sigma_s_grids.has_value());
	EXPECT_FALSE(med.Le_grids.has_value());
	EXPECT_FALSE(med.is_emissive());
}

TEST(RGBGridMediumData, BuildWithNoGrids) {
	// All empty vectors -> no optional grids set
	auto med = RGBGridMediumData<double>::build(
		{}, {}, {},   // sigma_a absent
		{}, {}, {},   // sigma_s absent
		{}, {}, {},   // Le absent
		2, 2, 2,
		unit_cube(),
		/*sigma_scale=*/1.0, /*Le_scale=*/0.0, /*g=*/0.0);

	EXPECT_FALSE(med.sigma_a_grids.has_value());
	EXPECT_FALSE(med.sigma_s_grids.has_value());
	EXPECT_FALSE(med.Le_grids.has_value());
	EXPECT_FALSE(med.is_emissive());
}

TEST(RGBGridMediumData, BuildWithAllGrids) {
	auto v = const_vol(0.5);
	auto med = RGBGridMediumData<double>::build(
		v, v, v,  v, v, v,  v, v, v,
		2, 2, 2, unit_cube(), 1.0, 1.0, 0.0);

	EXPECT_TRUE(med.sigma_a_grids.has_value());
	EXPECT_TRUE(med.sigma_s_grids.has_value());
	EXPECT_TRUE(med.Le_grids.has_value());
	EXPECT_TRUE(med.is_emissive());
}

// ---------------------------------------------------------------------------
// Absent-grid defaults (pbrt-v4: absent sigma -> 1, absent Le -> 0)
// ---------------------------------------------------------------------------

TEST(RGBGridMediumData, AbsentSigmaDefaultsToOne) {
	// No sigma_a or sigma_s grids; sigma_scale = 2
	auto med = RGBGridMediumData<double>::build(
		{}, {}, {},
		{}, {}, {},
		{}, {}, {},
		2, 2, 2, unit_cube(), /*sigma_scale=*/2.0, 0.0, 0.0);

	double sa[3], ss[3], Le[3];
	med.sample_point(0.5, 0.5, 0.5, sa, ss, Le);

	// sigma_a = sigma_scale * 1  (absent grid -> 1)
	for (int c = 0; c < 3; ++c) {
		EXPECT_DOUBLE_EQ(sa[c], 2.0) << "sigma_a channel " << c;
		EXPECT_DOUBLE_EQ(ss[c], 2.0) << "sigma_s channel " << c;
		EXPECT_DOUBLE_EQ(Le[c], 0.0) << "Le channel " << c;
	}
}

TEST(RGBGridMediumData, AbsentLeDefaultsToZero) {
	auto v = const_vol(1.0);
	auto med = RGBGridMediumData<double>::build(
		v, v, v,
		v, v, v,
		{}, {}, {},  // Le absent
		2, 2, 2, unit_cube(), 1.0, /*Le_scale=*/5.0, 0.0);

	double sa[3], ss[3], Le[3];
	med.sample_point(0.5, 0.5, 0.5, sa, ss, Le);

	for (int c = 0; c < 3; ++c)
		EXPECT_DOUBLE_EQ(Le[c], 0.0) << "Le channel " << c << " should be 0 (absent grid)";
}

// ---------------------------------------------------------------------------
// sigma_scale applied
// ---------------------------------------------------------------------------

TEST(RGBGridMediumData, SigmaScaleMultipliesOutput) {
	auto v = const_vol(0.4);
	double scale = 3.0;
	auto med = RGBGridMediumData<double>::build(
		v, v, v,  v, v, v,  {}, {}, {},
		2, 2, 2, unit_cube(), scale, 0.0, 0.0);

	double sa[3], ss[3], Le[3];
	med.sample_point(0.5, 0.5, 0.5, sa, ss, Le);

	for (int c = 0; c < 3; ++c) {
		EXPECT_NEAR(sa[c], scale * 0.4, 1e-12) << "sigma_a channel " << c;
		EXPECT_NEAR(ss[c], scale * 0.4, 1e-12) << "sigma_s channel " << c;
	}
}

// ---------------------------------------------------------------------------
// Le_scale and is_emissive
// ---------------------------------------------------------------------------

TEST(RGBGridMediumData, IsEmissiveRequiresLegridAndPositiveLeScale) {
	auto v = const_vol(1.0);

	// Le grid present but Le_scale = 0 -> not emissive
	auto med0 = RGBGridMediumData<double>::build(
		{}, {}, {},  {}, {}, {},  v, v, v,
		2, 2, 2, unit_cube(), 1.0, /*Le_scale=*/0.0, 0.0);
	EXPECT_FALSE(med0.is_emissive());

	// Le grid present AND Le_scale > 0 -> emissive
	auto med1 = RGBGridMediumData<double>::build(
		{}, {}, {},  {}, {}, {},  v, v, v,
		2, 2, 2, unit_cube(), 1.0, /*Le_scale=*/2.0, 0.0);
	EXPECT_TRUE(med1.is_emissive());
}

TEST(RGBGridMediumData, LeScaleMultipliesOutput) {
	auto v = const_vol(0.6);
	double le_scale = 4.0;
	auto med = RGBGridMediumData<double>::build(
		{}, {}, {},  {}, {}, {},  v, v, v,
		2, 2, 2, unit_cube(), 1.0, le_scale, 0.0);

	double sa[3], ss[3], Le[3];
	med.sample_point(0.5, 0.5, 0.5, sa, ss, Le);

	for (int c = 0; c < 3; ++c)
		EXPECT_NEAR(Le[c], le_scale * 0.6, 1e-12) << "Le channel " << c;
}

// ---------------------------------------------------------------------------
// Per-channel independence
// ---------------------------------------------------------------------------

TEST(RGBGridMediumData, PerChannelValuesAreIndependent) {
	// Channels have different constant values: R=0.1, G=0.5, B=0.9
	std::vector<double> vr = const_vol(0.1);
	std::vector<double> vg = const_vol(0.5);
	std::vector<double> vb = const_vol(0.9);

	auto med = RGBGridMediumData<double>::build(
		vr, vg, vb,  // sigma_a
		vr, vg, vb,  // sigma_s
		vr, vg, vb,  // Le
		2, 2, 2, unit_cube(), 1.0, 1.0, 0.0);

	double sa[3], ss[3], Le[3];
	med.sample_point(0.5, 0.5, 0.5, sa, ss, Le);

	EXPECT_NEAR(sa[0], 0.1, 1e-12);
	EXPECT_NEAR(sa[1], 0.5, 1e-12);
	EXPECT_NEAR(sa[2], 0.9, 1e-12);

	EXPECT_NEAR(ss[0], 0.1, 1e-12);
	EXPECT_NEAR(ss[1], 0.5, 1e-12);
	EXPECT_NEAR(ss[2], 0.9, 1e-12);

	EXPECT_NEAR(Le[0], 0.1, 1e-12);
	EXPECT_NEAR(Le[1], 0.5, 1e-12);
	EXPECT_NEAR(Le[2], 0.9, 1e-12);
}

// ---------------------------------------------------------------------------
// Majorant grid
// ---------------------------------------------------------------------------

TEST(RGBGridMediumData, MajorantGridBuiltCorrectly) {
	// Constant grids with sigma_a=0.3, sigma_s=0.7 per channel -> sigma_t = 1.0
	// sigma_scale = 2 -> majorant = 2.0 * 1.0 = 2.0
	auto sa = const_vol(0.3);
	auto ss = const_vol(0.7);
	auto med = RGBGridMediumData<double>::build(
		sa, sa, sa,  ss, ss, ss,  {}, {}, {},
		2, 2, 2, unit_cube(), /*sigma_scale=*/2.0, 0.0, 0.0,
		/*maj_res=*/4);

	// Every majorant voxel should be 2.0 * (0.3 + 0.7) = 2.0
	const auto& mg = med.majorant_grid;
	for (int z = 0; z < mg.res[2]; ++z)
		for (int y = 0; y < mg.res[1]; ++y)
			for (int x = 0; x < mg.res[0]; ++x)
				EXPECT_NEAR(mg.lookup(x, y, z), 2.0, 1e-12)
					<< "voxel (" << x << "," << y << "," << z << ")";
}

TEST(RGBGridMediumData, MajorantGridReflectsMaxChannel) {
	// R channel sigma_a+sigma_s = 0.2+0.3=0.5
	// G channel sigma_a+sigma_s = 0.4+0.4=0.8  <- max
	// B channel sigma_a+sigma_s = 0.1+0.2=0.3
	// sigma_scale=1 -> majorant = 0.8
	auto sa_r = const_vol(0.2);
	auto sa_g = const_vol(0.4);
	auto sa_b = const_vol(0.1);
	auto ss_r = const_vol(0.3);
	auto ss_g = const_vol(0.4);
	auto ss_b = const_vol(0.2);

	auto med = RGBGridMediumData<double>::build(
		sa_r, sa_g, sa_b,  ss_r, ss_g, ss_b,  {}, {}, {},
		2, 2, 2, unit_cube(), 1.0, 0.0, 0.0, 4);

	const auto& mg = med.majorant_grid;
	for (int z = 0; z < mg.res[2]; ++z)
		for (int y = 0; y < mg.res[1]; ++y)
			for (int x = 0; x < mg.res[0]; ++x)
				EXPECT_NEAR(mg.lookup(x, y, z), 0.8, 1e-12)
					<< "voxel (" << x << "," << y << "," << z << ")";
}

// ---------------------------------------------------------------------------
// DDA traversal via sample_ray
// ---------------------------------------------------------------------------

TEST(RGBGridMediumData, SampleRayYieldsSegments) {
	auto v = const_vol(1.0, 64);
	auto med = RGBGridMediumData<double>::build(
		v, v, v,  v, v, v,  {}, {}, {},
		4, 4, 4, unit_cube(), 1.0, 0.0, 0.0, 4);

	// Axis-aligned ray through the unit cube
	Ray3<double> ray(0.5, 0.5, 0.0,  0.0, 0.0, 1.0);
	double tMin, tMax;
	ASSERT_TRUE(med.intersect_ray(ray, 1e10, tMin, tMax));

	auto it = med.sample_ray(ray, tMin, tMax);
	int count = 0;
	double t_prev = tMin;
	for (;;) {
		auto seg = it.Next();
		if (!seg.has_value()) break;
		EXPECT_GE(seg->tMin, t_prev - 1e-12);
		EXPECT_GT(seg->tMax, seg->tMin);
		EXPECT_GT(seg->sigma_maj, 0.0);
		t_prev = seg->tMax;
		++count;
	}
	EXPECT_GT(count, 0) << "DDA should yield at least one segment";
	EXPECT_NEAR(t_prev, tMax, 1e-10) << "Segments should cover full ray interval";
}

TEST(RGBGridMediumData, SampleRayDiagonalCoversTotalInterval) {
	auto v = const_vol(0.5, 64);
	auto med = RGBGridMediumData<double>::build(
		v, v, v,  v, v, v,  {}, {}, {},
		4, 4, 4, unit_cube(), 1.0, 0.0, 0.0, 4);

	// Diagonal ray
	double invSqrt3 = 1.0 / std::sqrt(3.0);
	Ray3<double> ray(0.0, 0.0, 0.0, invSqrt3, invSqrt3, invSqrt3);
	double tMin, tMax;
	ASSERT_TRUE(med.intersect_ray(ray, 1e10, tMin, tMax));

	auto it = med.sample_ray(ray, tMin, tMax);
	double total = 0.0;
	for (;;) {
		auto seg = it.Next();
		if (!seg.has_value()) break;
		total += seg->tMax - seg->tMin;
	}

	EXPECT_NEAR(total, tMax - tMin, 1e-9);
}

// ---------------------------------------------------------------------------
// sigma_maj value correctness
// Majorant grid stores sigma_scale * max_c(sigma_a[c]+sigma_s[c]).
// DDA sigma_t_ = 1, so seg->sigma_maj = grid.lookup(voxel) * 1.
// pbrt-v4: majorantGrid.Set(x,y,z, sigmaScale * maxSigma_t)
//          DDAMajorantIterator(..., sigma_t = SampledSpectrum(1))
// ---------------------------------------------------------------------------
TEST(RGBGridMediumData, SampleRaySegmentSigmaMajMatchesMajorantGrid) {
	// Constant grids: sigma_a = 0.3, sigma_s = 0.5 per channel
	// sigma_t per channel = 0.8; sigma_scale = 2 -> majorant = 2 * 0.8 = 1.6
	// 4x4x4 grid = 64 voxels
	std::vector<double> sa(64, 0.3);
	std::vector<double> ss(64, 0.5);
	double expected_maj = 2.0 * (0.3 + 0.5); // = 1.6

	auto med = RGBGridMediumData<double>::build(
		sa, sa, sa,  ss, ss, ss,  {}, {}, {},
		4, 4, 4, unit_cube(), /*sigma_scale=*/2.0, 0.0, 0.0, /*maj_res=*/4);

	Ray3<double> ray(0.5, 0.5, 0.0,  0.0, 0.0, 1.0);
	double tMin, tMax;
	ASSERT_TRUE(med.intersect_ray(ray, 1e10, tMin, tMax));

	auto it = med.sample_ray(ray, tMin, tMax);
	int count = 0;
	for (;;) {
		auto seg = it.Next();
		if (!seg.has_value()) break;
		EXPECT_NEAR(seg->sigma_maj, expected_maj, 1e-12)
			<< "segment " << count << " sigma_maj";
		++count;
	}
	EXPECT_GT(count, 0);
}

// ---------------------------------------------------------------------------
// intersect_ray
// ---------------------------------------------------------------------------

TEST(RGBGridMediumData, IntersectRayHitsUnitCube) {
	auto med = RGBGridMediumData<double>::build(
		{}, {}, {},  {}, {}, {},  {}, {}, {},
		2, 2, 2, unit_cube(), 1.0, 0.0, 0.0);

	// Ray along +Z through centre
	Ray3<double> ray(0.5, 0.5, -1.0,  0.0, 0.0, 1.0);
	double tMin, tMax;
	EXPECT_TRUE(med.intersect_ray(ray, 1e10, tMin, tMax));
	EXPECT_NEAR(tMin, 1.0, 1e-12);
	EXPECT_NEAR(tMax, 2.0, 1e-12);
}

TEST(RGBGridMediumData, IntersectRayMisses) {
	auto med = RGBGridMediumData<double>::build(
		{}, {}, {},  {}, {}, {},  {}, {}, {},
		2, 2, 2, unit_cube(), 1.0, 0.0, 0.0);

	// Ray that misses the unit cube entirely
	Ray3<double> ray(2.0, 2.0, -1.0,  0.0, 0.0, 1.0);
	double tMin, tMax;
	EXPECT_FALSE(med.intersect_ray(ray, 1e10, tMin, tMax));
}

// ---------------------------------------------------------------------------
// float specialization
// ---------------------------------------------------------------------------

TEST(RGBGridMediumDataFloat, BasicSamplePoint) {
	std::vector<float> v(8, 0.5f);
	Bounds3<float> b(0.f, 0.f, 0.f, 1.f, 1.f, 1.f);
	auto med = RGBGridMediumData<float>::build(
		v, v, v,  v, v, v,  {}, {}, {},
		2, 2, 2, b, 2.0f, 0.0f, 0.0f);

	float sa[3], ss[3], Le[3];
	med.sample_point(0.5f, 0.5f, 0.5f, sa, ss, Le);

	for (int c = 0; c < 3; ++c) {
		EXPECT_NEAR(sa[c], 1.0f, 1e-5f) << "sigma_a channel " << c;
		EXPECT_NEAR(ss[c], 1.0f, 1e-5f) << "sigma_s channel " << c;
		EXPECT_EQ  (Le[c], 0.0f) << "Le channel " << c;
	}
}
