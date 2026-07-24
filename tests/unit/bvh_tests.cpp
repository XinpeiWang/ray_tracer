/**
 * @file bvh_tests.cpp
 * @brief Unit tests for the SAH BVH implementation
 *
 * Tests the Surface-Area Heuristic BVH (pbrt-v4 style):
 * - Leaf creation for small primitive counts
 * - Interior node SAH split for larger scenes
 * - Hit detection correctness (all primitives reachable)
 * - Bounding box accuracy
 * - Degenerate cases (1 primitive, all co-planar centroids)
 */

#include <gtest/gtest.h>
#include "rtweekend.h"
#include "bvh.h"
#include "sphere.h"
#include "hittable_list.h"
#include "material.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a grid of N lambertian spheres with radius 0.5 placed at (i,0,0).
static hittable_list make_sphere_row(int n) {
	hittable_list world;
	auto mat = make_shared<lambertian>(color(0.8, 0.8, 0.8));
	for (int i = 0; i < n; ++i)
		world.add(make_shared<sphere>(point3(i * 2.0, 0.0, 0.0), 0.5, mat));
	return world;
}

// ---------------------------------------------------------------------------
// BVH Bounding-box tests
// ---------------------------------------------------------------------------

TEST(BvhTest, SinglePrimitiveBoundingBox) {
	hittable_list world;
	auto mat = make_shared<lambertian>(color(1, 0, 0));
	world.add(make_shared<sphere>(point3(0, 0, 0), 1.0, mat));

	bvh_node bvh(world);
	aabb bbox = bvh.bounding_box();

	EXPECT_LE(bbox.x.min, -1.0);
	EXPECT_GE(bbox.x.max,  1.0);
	EXPECT_LE(bbox.y.min, -1.0);
	EXPECT_GE(bbox.y.max,  1.0);
	EXPECT_LE(bbox.z.min, -1.0);
	EXPECT_GE(bbox.z.max,  1.0);
}

TEST(BvhTest, MultiplePrimitivesBoundingBoxContainsAll) {
	hittable_list world = make_sphere_row(10);
	bvh_node bvh(world);
	aabb bbox = bvh.bounding_box();

	// Spheres at x = 0,2,4,...,18 with radius 0.5
	// So overall x range: [-0.5, 18.5]
	EXPECT_LE(bbox.x.min, -0.4);
	EXPECT_GE(bbox.x.max,  18.4);
}

// ---------------------------------------------------------------------------
// Hit detection tests
// ---------------------------------------------------------------------------

TEST(BvhTest, HitsAllSpheresInRow) {
	const int N = 20;
	hittable_list world = make_sphere_row(N);
	bvh_node bvh(world);

	auto mat = make_shared<lambertian>(color(1, 1, 1));

	// Fire a ray at each sphere centre from far below (+y direction)
	for (int i = 0; i < N; ++i) {
		point3 origin(i * 2.0, -100.0, 0.0);
		vec3   dir(0.0, 1.0, 0.0);
		ray r(origin, dir);
		hit_record rec;
		bool hit = bvh.hit(r, interval(0.001, infinity), rec);
		EXPECT_TRUE(hit) << "Missed sphere " << i;
	}
}

TEST(BvhTest, MissesEmptySpace) {
	hittable_list world = make_sphere_row(4);
	bvh_node bvh(world);

	// Fire a ray that completely misses all spheres (far to the side)
	ray r(point3(100.0, 100.0, 100.0), vec3(0.0, 0.0, 1.0));
	hit_record rec;
	bool hit = bvh.hit(r, interval(0.001, infinity), rec);
	EXPECT_FALSE(hit);
}

TEST(BvhTest, NearestHitIsReturned) {
	// Two spheres on the same ray path; BVH should return the closer one.
	hittable_list world;
	auto mat = make_shared<lambertian>(color(0.5, 0.5, 0.5));
	world.add(make_shared<sphere>(point3(0, 0, 3),  1.0, mat));  // closer
	world.add(make_shared<sphere>(point3(0, 0, 8),  1.0, mat));  // farther

	bvh_node bvh(world);

	ray r(point3(0, 0, -10), vec3(0, 0, 1));
	hit_record rec;
	bool hit = bvh.hit(r, interval(0.001, infinity), rec);

	EXPECT_TRUE(hit);
	// The closer sphere surface is at z = 3 - 1 = 2 from origin at z=-10 => t ~ 11
	EXPECT_NEAR(rec.p.z(), 2.0, 0.01);
}

// ---------------------------------------------------------------------------
// Leaf vs interior node tests
// ---------------------------------------------------------------------------

TEST(BvhTest, SmallSceneUsesLeaf) {
	// kMaxLeafPrims = 4; with 3 prims the root should produce a leaf,
	// which still must hit correctly.
	hittable_list world = make_sphere_row(3);
	bvh_node bvh(world);

	for (int i = 0; i < 3; ++i) {
		ray r(point3(i * 2.0, -10.0, 0.0), vec3(0, 1, 0));
		hit_record rec;
		EXPECT_TRUE(bvh.hit(r, interval(0.001, infinity), rec))
			<< "Leaf missed sphere " << i;
	}
}

TEST(BvhTest, LargeSceneBuildAndHit) {
	// Build a BVH over 64 spheres to exercise the recursive SAH split path.
	const int N = 64;
	hittable_list world = make_sphere_row(N);
	bvh_node bvh(world);

	// Spot-check 8 evenly spaced spheres
	for (int i = 0; i < N; i += 8) {
		ray r(point3(i * 2.0, -10.0, 0.0), vec3(0, 1, 0));
		hit_record rec;
		EXPECT_TRUE(bvh.hit(r, interval(0.001, infinity), rec))
			<< "Large scene missed sphere " << i;
	}
}

// ---------------------------------------------------------------------------
// Degenerate case: all centroids are co-planar
// ---------------------------------------------------------------------------

TEST(BvhTest, CoplanarCentroidsStillHits) {
	// All spheres on the same plane (z = 0); centroid AABB has zero depth.
	// The BVH should fall back gracefully and still hit all spheres.
	hittable_list world;
	auto mat = make_shared<lambertian>(color(0.3, 0.6, 0.9));
	for (int i = 0; i < 8; ++i)
		for (int j = 0; j < 8; ++j)
			world.add(make_shared<sphere>(point3(i * 2.0, j * 2.0, 0.0), 0.4, mat));

	bvh_node bvh(world);

	// Hit a few spheres from front
	for (int i : {0, 3, 7}) {
		for (int j : {0, 4, 7}) {
			ray r(point3(i * 2.0, j * 2.0, -10.0), vec3(0, 0, 1));
			hit_record rec;
			EXPECT_TRUE(bvh.hit(r, interval(0.001, infinity), rec))
				<< "Coplanar missed (" << i << "," << j << ")";
		}
	}
}

// ---------------------------------------------------------------------------
// SAH cost constants sanity
// ---------------------------------------------------------------------------

TEST(BvhTest, SahConstantsArePositive) {
	EXPECT_GT(kCostTraversal, 0.0);
	EXPECT_GT(kCostIntersect, 0.0);
	EXPECT_GT(kBuckets, 1);
	EXPECT_GT(kMaxLeafPrims, 0);
}
