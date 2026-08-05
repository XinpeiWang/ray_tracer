// kd_tree_tests.cpp
// Unit tests for src/shared/kd_tree.h
// pbrt-v4 reference: src/pbrt/cpu/aggregates.h / aggregates.cpp -- KdTreeAggregate

#include "../../src/shared/kd_tree.h"
#include <gtest/gtest.h>
#include <array>
#include <cmath>

// ---------------------------------------------------------------------------
// Mock primitive: axis-aligned sphere (bounding box + analytic intersection)
// ---------------------------------------------------------------------------
struct MockSphere {
	double cx, cy, cz, r; // center + radius
	int    id = 0;         // for identity checks

	void bbox(double out_min[3], double out_max[3]) const {
		out_min[0] = cx - r; out_min[1] = cy - r; out_min[2] = cz - r;
		out_max[0] = cx + r; out_max[1] = cy + r; out_max[2] = cz + r;
	}

	std::optional<KdHit<double>>
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
		KdHit<double> h;
		h.t = t;
		h.nx = (org[0] + t*dir[0] - cx) / r;
		h.ny = (org[1] + t*dir[1] - cy) / r;
		h.nz = (org[2] + t*dir[2] - cz) / r;
		h.u = h.v = 0;
		h.prim_id = id;
		return h;
	}

	bool intersect_p(const double org[3], const double dir[3],
					 double t_max) const {
		return intersect(org, dir, 1e-6, t_max).has_value();
	}
};

// Helper: unit ray in -Z direction from (0,0,5)
static void ray_down(double org[3], double dir[3]) {
	org[0]=0; org[1]=0; org[2]=5;
	dir[0]=0; dir[1]=0; dir[2]=-1;
}

// ---------------------------------------------------------------------------
// EmptyTree: no primitives
// ---------------------------------------------------------------------------
TEST(KdTree, EmptyTree_MissesEverything) {
	KdTree<double, MockSphere> kd;
	kd.build({});
	double org[3]={0,0,5}, dir[3]={0,0,-1};
	EXPECT_FALSE(kd.intersect(org, dir, 0, 1e30).has_value());
	EXPECT_FALSE(kd.intersect_p(org, dir, 1e30));
	EXPECT_EQ(kd.num_prims(), 0);
}

// ---------------------------------------------------------------------------
// SingleSphere: basic hit and miss
// ---------------------------------------------------------------------------
TEST(KdTree, SingleSphere_Hit) {
	MockSphere s{0,0,0,1,42};
	KdTree<double, MockSphere> kd;
	kd.build({s});

	double org[3]={0,0,5}, dir[3]={0,0,-1};
	auto h = kd.intersect(org, dir, 0, 1e30);
	ASSERT_TRUE(h.has_value());
	EXPECT_NEAR(h->t, 4.0, 1e-9);   // sphere at z=0, r=1, hit at z=1 -> t=4
	EXPECT_NEAR(h->nz, 1.0, 1e-9);  // outward normal in +Z
}

TEST(KdTree, SingleSphere_Miss_WrongDirection) {
	MockSphere s{0,0,0,1,0};
	KdTree<double, MockSphere> kd;
	kd.build({s});
	double org[3]={0,5,0}, dir[3]={0,1,0}; // ray moving away from sphere
	EXPECT_FALSE(kd.intersect(org, dir, 0, 1e30).has_value());
}

TEST(KdTree, SingleSphere_Miss_TMaxTooShort) {
	MockSphere s{0,0,0,1,0};
	KdTree<double, MockSphere> kd;
	kd.build({s});
	double org[3]={0,0,5}, dir[3]={0,0,-1};
	// t_hit = 4, but t_max = 2: should miss
	EXPECT_FALSE(kd.intersect(org, dir, 0, 2.0).has_value());
}

TEST(KdTree, SingleSphere_ShadowHit) {
	MockSphere s{0,0,0,1,0};
	KdTree<double, MockSphere> kd;
	kd.build({s});
	double org[3]={0,0,5}, dir[3]={0,0,-1};
	EXPECT_TRUE(kd.intersect_p(org, dir, 1e30));
}

TEST(KdTree, SingleSphere_ShadowMiss_TooFar) {
	MockSphere s{0,0,0,1,0};
	KdTree<double, MockSphere> kd;
	kd.build({s});
	double org[3]={0,0,5}, dir[3]={0,0,-1};
	EXPECT_FALSE(kd.intersect_p(org, dir, 2.0));
}

// ---------------------------------------------------------------------------
// TwoSpheres: returns closest hit
// ---------------------------------------------------------------------------
TEST(KdTree, TwoSpheres_ReturnsClosest) {
	// Sphere A at z=0, sphere B at z=-3, both radius 0.5
	// Ray from z=5 going in -z, hits A first (t=4.5), then B (t=7.5)
	MockSphere a{0,0, 0, 0.5, 1};
	MockSphere b{0,0,-3, 0.5, 2};
	KdTree<double, MockSphere> kd;
	kd.build({a, b});

	double org[3]={0,0,5}, dir[3]={0,0,-1};
	auto h = kd.intersect(org, dir, 0, 1e30);
	ASSERT_TRUE(h.has_value());
	EXPECT_NEAR(h->t, 4.5, 1e-9);   // hits sphere A first
	EXPECT_EQ(h->prim_id, 0);
}

TEST(KdTree, TwoSpheres_OccludesSecond) {
	MockSphere a{0,0, 0, 0.5, 1};
	MockSphere b{0,0,-3, 0.5, 2};
	KdTree<double, MockSphere> kd;
	kd.build({a, b});
	double org[3]={0,0,5}, dir[3]={0,0,-1};
	// shadow test: is anything between z=5 and z=2? sphere A at t=4.5 -- yes
	EXPECT_TRUE(kd.intersect_p(org, dir, 1e30));
}

// ---------------------------------------------------------------------------
// ManySpheres: SAH builds a tree with multiple nodes
// ---------------------------------------------------------------------------
TEST(KdTree, ManySpheres_AllHittable) {
	// 10 spheres along X axis, r=0.4
	std::vector<MockSphere> spheres;
	for (int i = 0; i < 10; ++i)
		spheres.push_back(MockSphere{(double)i, 0, 0, 0.4, i});

	KdTree<double, MockSphere> kd;
	kd.build(spheres);
	EXPECT_GT(kd.num_nodes(), 1); // tree should have more than 1 node

	// Fire a ray at each sphere and verify it's found
	for (int i = 0; i < 10; ++i) {
		double org[3] = {(double)i, 0, 5};
		double dir[3] = {0, 0, -1};
		auto h = kd.intersect(org, dir, 0, 1e30);
		ASSERT_TRUE(h.has_value()) << "Missed sphere " << i;
		EXPECT_EQ(h->prim_id, i);
	}
}

TEST(KdTree, ManySpheres_MissGaps) {
	// Same 10 spheres, fire ray between them
	std::vector<MockSphere> spheres;
	for (int i = 0; i < 10; ++i)
		spheres.push_back(MockSphere{(double)i, 0, 0, 0.4, i});

	KdTree<double, MockSphere> kd;
	kd.build(spheres);

	// Ray between sphere 0 (x=0) and sphere 1 (x=1): x=0.5, gap
	double org[3]={0.5, 0, 5};
	double dir[3]={0, 0, -1};
	EXPECT_FALSE(kd.intersect(org, dir, 0, 1e30).has_value());
}

// ---------------------------------------------------------------------------
// AABB misses root bounds: should not traverse any nodes
// ---------------------------------------------------------------------------
TEST(KdTree, RayOutsideBounds_NoHit) {
	MockSphere s{0,0,0, 0.5, 0};
	KdTree<double, MockSphere> kd;
	kd.build({s});
	// Ray far off to the side
	double org[3]={100, 0, 5}, dir[3]={0, 0, -1};
	EXPECT_FALSE(kd.intersect(org, dir, 0, 1e30).has_value());
	EXPECT_FALSE(kd.intersect_p(org, dir, 1e30));
}

// ---------------------------------------------------------------------------
// CustomDepth: respect max_depth = 1 (shallow tree forces leaf early)
// ---------------------------------------------------------------------------
TEST(KdTree, CustomMaxDepth_ShallowTree) {
	std::vector<MockSphere> spheres;
	for (int i = 0; i < 8; ++i)
		spheres.push_back(MockSphere{(double)i, 0, 0, 0.4, i});

	KdTree<double, MockSphere>::Params p;
	p.max_depth = 1;

	KdTree<double, MockSphere> kd;
	kd.build(spheres, p);

	// All spheres should still be reachable
	for (int i = 0; i < 8; ++i) {
		double org[3] = {(double)i, 0, 5};
		double dir[3] = {0, 0, -1};
		auto h = kd.intersect(org, dir, 0, 1e30);
		EXPECT_TRUE(h.has_value()) << "Missed sphere " << i;
	}
}

// ---------------------------------------------------------------------------
// NumNodes grows with scene complexity
// ---------------------------------------------------------------------------
TEST(KdTree, NumNodes_IncreasesWithMorePrims) {
	KdTree<double, MockSphere> kd1, kd10, kd100;
	kd1.build(  {MockSphere{0,0,0,0.1,0}});
	std::vector<MockSphere> s10, s100;
	for (int i = 0; i < 10;  ++i) s10.push_back( {(double)i,0,0,0.1,i});
	for (int i = 0; i < 100; ++i) s100.push_back({(double)i,0,0,0.1,i});
	kd10.build(s10);
	kd100.build(s100);

	EXPECT_LE(kd1.num_nodes(),  kd10.num_nodes());
	EXPECT_LE(kd10.num_nodes(), kd100.num_nodes());
}
