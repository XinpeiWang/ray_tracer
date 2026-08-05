// bvh_aggregate_tests.cpp
// Unit tests for src/shared/bvh_aggregate.h
// pbrt-v4 reference: src/pbrt/cpu/aggregates.h / aggregates.cpp -- BVHAggregate

#include "../../src/shared/bvh_aggregate.h"
#include <gtest/gtest.h>
#include <array>
#include <cmath>

// ---------------------------------------------------------------------------
// Mock primitive: axis-aligned sphere (bounding box + analytic intersection)
// Same layout as kd_tree_tests.cpp MockSphere, adapted for BvhHit.
// ---------------------------------------------------------------------------
struct MockSphere {
	double cx, cy, cz, r;
	int    id = 0;

	void bbox(double out_min[3], double out_max[3]) const {
		out_min[0] = cx - r; out_min[1] = cy - r; out_min[2] = cz - r;
		out_max[0] = cx + r; out_max[1] = cy + r; out_max[2] = cz + r;
	}

	std::optional<BvhHit<double>>
	intersect(const double org[3], const double dir[3],
			  double t_min, double t_max) const {
		double ox = org[0]-cx, oy = org[1]-cy, oz = org[2]-cz;
		double a = dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2];
		double b = 2*(ox*dir[0] + oy*dir[1] + oz*dir[2]);
		double c = ox*ox + oy*oy + oz*oz - r*r;
		double disc = b*b - 4*a*c;
		if (disc < 0) return {};
		double sq = std::sqrt(disc);
		double t0 = (-b - sq) / (2*a);
		double t1 = (-b + sq) / (2*a);
		double t = (t0 >= t_min && t0 <= t_max) ? t0
				 : (t1 >= t_min && t1 <= t_max) ? t1 : -1.0;
		if (t < 0) return {};
		BvhHit<double> h;
		h.t        = t;
		h.normal[0] = (org[0] + t*dir[0] - cx) / r;
		h.normal[1] = (org[1] + t*dir[1] - cy) / r;
		h.normal[2] = (org[2] + t*dir[2] - cz) / r;
		h.uv[0] = h.uv[1] = 0;
		h.prim_id = id;
		return h;
	}

	bool intersect_p(const double org[3], const double dir[3],
					 double t_max) const {
		return intersect(org, dir, 1e-6, t_max).has_value();
	}
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void ray_down(double org[3], double dir[3]) {
	org[0]=0; org[1]=0; org[2]=5;
	dir[0]=0; dir[1]=0; dir[2]=-1;
}

// Build a BvhTree from a list of spheres using the given split method.
static BvhTree<double, MockSphere> make_bvh(
	std::vector<MockSphere> spheres,
	BvhSplitMethod method = BvhSplitMethod::SAH,
	int max_prims = 1)
{
	BvhTree<double, MockSphere> bvh;
	bvh.build(std::move(spheres), max_prims, method);
	return bvh;
}

// ---------------------------------------------------------------------------
// Empty tree
// ---------------------------------------------------------------------------
TEST(BvhAggregate, EmptyTree_MissesEverything) {
	BvhTree<double, MockSphere> bvh;
	double org[3]={0,0,5}, dir[3]={0,0,-1};
	EXPECT_FALSE(bvh.intersect(org, dir, 0.0, 1e30).has_value());
	EXPECT_FALSE(bvh.intersect_p(org, dir, 1e30));
	EXPECT_TRUE(bvh.empty());
}

// ---------------------------------------------------------------------------
// Single sphere: hit and miss
// ---------------------------------------------------------------------------
TEST(BvhAggregate, SingleSphere_Hit) {
	std::vector<MockSphere> prims = { {0,0,0, 1.0, 7} };
	auto bvh = make_bvh(prims);
	double org[3]={0,0,5}, dir[3]={0,0,-1};
	auto hit = bvh.intersect(org, dir, 0.0, 1e30);
	ASSERT_TRUE(hit.has_value());
	EXPECT_NEAR(hit->t, 4.0, 1e-9);
	EXPECT_EQ(hit->prim_id, 7);
}

TEST(BvhAggregate, SingleSphere_Miss) {
	std::vector<MockSphere> prims = { {0,0,0, 1.0, 0} };
	auto bvh = make_bvh(prims);
	double org[3]={0,5,5}, dir[3]={0,0,-1}; // offset in Y, misses sphere
	auto hit = bvh.intersect(org, dir, 0.0, 1e30);
	EXPECT_FALSE(hit.has_value());
}

// ---------------------------------------------------------------------------
// Two spheres: returns the closer one
// ---------------------------------------------------------------------------
TEST(BvhAggregate, TwoSpheres_ReturnsClosest) {
	std::vector<MockSphere> prims = {
		{0, 0, 0, 1.0,  1},  // farther: at z=0, hit at t=4
		{0, 0, 2, 1.0,  2}   // closer : at z=2, hit at t=2
	};
	auto bvh = make_bvh(prims);
	double org[3]={0,0,5}, dir[3]={0,0,-1};
	auto hit = bvh.intersect(org, dir, 0.0, 1e30);
	ASSERT_TRUE(hit.has_value());
	EXPECT_EQ(hit->prim_id, 2);
	EXPECT_NEAR(hit->t, 2.0, 1e-9);
}

// ---------------------------------------------------------------------------
// Many spheres: all are hittable (parallel rays from above each center)
// ---------------------------------------------------------------------------
TEST(BvhAggregate, ManySpheres_AllHittable) {
	std::vector<MockSphere> prims;
	for (int i = 0; i < 20; ++i)
		prims.push_back({static_cast<double>(i), 0.0, 0.0, 0.4, i});

	auto bvh = make_bvh(prims);

	for (int i = 0; i < 20; ++i) {
		double org[3] = {static_cast<double>(i), 0.0, 5.0};
		double dir[3] = {0, 0, -1};
		auto hit = bvh.intersect(org, dir, 0.0, 1e30);
		ASSERT_TRUE(hit.has_value()) << "sphere " << i << " not hit";
		EXPECT_EQ(hit->prim_id, i);
	}
}

// ---------------------------------------------------------------------------
// intersect_p: shadow ray hits and misses
// ---------------------------------------------------------------------------
TEST(BvhAggregate, ShadowRay_HitAndMiss) {
	std::vector<MockSphere> prims = { {0, 0, 0, 1.0, 42} };
	auto bvh = make_bvh(prims);

	double org_hit[3] = {0,0,5}, dir[3] = {0,0,-1};
	EXPECT_TRUE(bvh.intersect_p(org_hit, dir, 1e30));

	double org_miss[3] = {5,5,5};
	EXPECT_FALSE(bvh.intersect_p(org_miss, dir, 1e30));
}

// ---------------------------------------------------------------------------
// Split method coverage: Middle, EqualCounts, HLBVH all find all spheres
// ---------------------------------------------------------------------------
TEST(BvhAggregate, SplitMiddle_AllHittable) {
	std::vector<MockSphere> prims;
	for (int i = 0; i < 16; ++i)
		prims.push_back({static_cast<double>(i) * 2.0, 0, 0, 0.4, i});

	auto bvh = make_bvh(prims, BvhSplitMethod::Middle);
	for (int i = 0; i < 16; ++i) {
		double org[3] = {static_cast<double>(i) * 2.0, 0, 5};
		double dir[3] = {0, 0, -1};
		auto hit = bvh.intersect(org, dir, 0.0, 1e30);
		ASSERT_TRUE(hit.has_value()) << "Middle: sphere " << i << " not hit";
		EXPECT_EQ(hit->prim_id, i);
	}
}

TEST(BvhAggregate, SplitEqualCounts_AllHittable) {
	std::vector<MockSphere> prims;
	for (int i = 0; i < 16; ++i)
		prims.push_back({static_cast<double>(i) * 2.0, 0, 0, 0.4, i});

	auto bvh = make_bvh(prims, BvhSplitMethod::EqualCounts);
	for (int i = 0; i < 16; ++i) {
		double org[3] = {static_cast<double>(i) * 2.0, 0, 5};
		double dir[3] = {0, 0, -1};
		auto hit = bvh.intersect(org, dir, 0.0, 1e30);
		ASSERT_TRUE(hit.has_value()) << "EqualCounts: sphere " << i << " not hit";
		EXPECT_EQ(hit->prim_id, i);
	}
}

TEST(BvhAggregate, SplitHLBVH_AllHittable) {
	std::vector<MockSphere> prims;
	for (int i = 0; i < 16; ++i)
		prims.push_back({static_cast<double>(i) * 2.0, 0, 0, 0.4, i});

	auto bvh = make_bvh(prims, BvhSplitMethod::HLBVH);
	for (int i = 0; i < 16; ++i) {
		double org[3] = {static_cast<double>(i) * 2.0, 0, 5};
		double dir[3] = {0, 0, -1};
		auto hit = bvh.intersect(org, dir, 0.0, 1e30);
		ASSERT_TRUE(hit.has_value()) << "HLBVH: sphere " << i << " not hit";
		EXPECT_EQ(hit->prim_id, i);
	}
}

// ---------------------------------------------------------------------------
// max_prims_in_node > 1: multiple prims per leaf, still correct
// ---------------------------------------------------------------------------
TEST(BvhAggregate, MaxPrimsPerLeaf4_AllHittable) {
	std::vector<MockSphere> prims;
	for (int i = 0; i < 12; ++i)
		prims.push_back({static_cast<double>(i) * 2.0, 0, 0, 0.4, i});

	auto bvh = make_bvh(prims, BvhSplitMethod::SAH, 4);
	for (int i = 0; i < 12; ++i) {
		double org[3] = {static_cast<double>(i) * 2.0, 0, 5};
		double dir[3] = {0, 0, -1};
		auto hit = bvh.intersect(org, dir, 0.0, 1e30);
		ASSERT_TRUE(hit.has_value()) << "Leaf4: sphere " << i << " not hit";
		EXPECT_EQ(hit->prim_id, i);
	}
}

// ---------------------------------------------------------------------------
// t_max cutoff: ray too short to reach sphere
// ---------------------------------------------------------------------------
TEST(BvhAggregate, TMaxCutoff_NoHit) {
	std::vector<MockSphere> prims = { {0, 0, 0, 1.0, 0} };
	auto bvh = make_bvh(prims);
	double org[3]={0,0,5}, dir[3]={0,0,-1};
	// sphere front surface at t=4, but t_max=3.0 — should miss
	auto hit = bvh.intersect(org, dir, 0.0, 3.0);
	EXPECT_FALSE(hit.has_value());
}

// ---------------------------------------------------------------------------
// Bounds: returned AABB should contain all sphere centers
// ---------------------------------------------------------------------------
TEST(BvhAggregate, Bounds_ContainAllPrimitives) {
	std::vector<MockSphere> prims = {
		{-5, 0, 0, 1.0, 0},
		{ 5, 0, 0, 1.0, 1},
		{ 0, 3, 0, 1.0, 2}
	};
	auto bvh = make_bvh(prims);
	double lo[3], hi[3];
	bvh.bounds(lo, hi);
	EXPECT_LE(lo[0], -6.0);
	EXPECT_GE(hi[0],  6.0);
	EXPECT_LE(lo[1], -1.0);
	EXPECT_GE(hi[1],  4.0);
}

// ---------------------------------------------------------------------------
// Regression: two spheres with coincident centroids fall back to leaf
// (mirrors the kd-tree degenerate-centroid test)
// ---------------------------------------------------------------------------
TEST(BvhAggregate, CoincidentCentroids_BothReachable) {
	// Two spheres with the same center but different radii.
	// Their centroids coincide, so SAH must create a leaf containing both.
	std::vector<MockSphere> prims = {
		{0, 0, 0, 0.5, 10},
		{0, 0, 0, 1.0, 20}
	};
	auto bvh = make_bvh(prims);
	double org[3]={0,0,5}, dir[3]={0,0,-1};
	auto hit = bvh.intersect(org, dir, 0.0, 1e30);
	ASSERT_TRUE(hit.has_value());
	// Both spheres share the same center. The leaf tests both.
	// The outer sphere (r=1.0, prim_id=20) front face is at t=4.0 (5-1),
	// closer than the inner sphere (r=0.5, prim_id=10) front face at t=4.5 (5-0.5).
	EXPECT_EQ(hit->prim_id, 20);
	EXPECT_NEAR(hit->t, 4.0, 1e-9);
}

// ---------------------------------------------------------------------------
// Regression: dirIsNeg child ordering (pbrt-v4 alignment)
// Build two spheres on opposite sides of the origin along X.
// Ray traveling in -X direction should visit the right child first.
// ---------------------------------------------------------------------------
TEST(BvhAggregate, DirIsNeg_ChildOrderingCorrect) {
	std::vector<MockSphere> prims = {
		{-3, 0, 0, 0.4,  1},  // left
		{ 3, 0, 0, 0.4,  2}   // right
	};
	auto bvh = make_bvh(prims);
	// Ray from +X traveling in -X: hits sphere 2 (right) first
	double org[3]={8,0,0}, dir[3]={-1,0,0};
	auto hit = bvh.intersect(org, dir, 0.0, 1e30);
	ASSERT_TRUE(hit.has_value());
	EXPECT_EQ(hit->prim_id, 2);
	EXPECT_NEAR(hit->t, 4.6, 1e-9);
}
