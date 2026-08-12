/**
 * @file pbrt_quadify_tests.cpp
 * @brief Unit tests for rejoining triangle pairs into parallelogram quads
 *
 * The interesting cases are all refusals. Merging two triangles that really do
 * form a parallelogram is arithmetic; deciding that a given pair does NOT is
 * what keeps a light where the scene put it, so most of these tests hand it
 * something that looks mergeable and check that it declines.
 */

#include <gtest/gtest.h>

#include "pbrt_quadify.h"

#include <cmath>

namespace {

// One triangle, with the emissive/material tags that decide its fate.
pbrt_flatten::Triangle tri(double ax, double ay, double az,
						   double bx, double by, double bz,
						   double cx, double cy, double cz,
						   int material = 0, int areaLight = -1) {
	pbrt_flatten::Triangle t;
	const double v[9] = {ax, ay, az, bx, by, bz, cx, cy, cz};
	for (int i = 0; i < 9; ++i) t.v[i] = v[i];
	t.material = material;
	t.areaLight = areaLight;
	return t;
}

// The two halves of a unit square in the y=0 plane, written the way pbrt
// writes them: indices [0 1 2, 0 2 3] over four corners, so the two triangles
// share the 0-2 diagonal.
std::vector<pbrt_flatten::Triangle> squareHalves(int areaLight = 0) {
	return {
		tri(0, 0, 0,  1, 0, 0,  1, 0, 1, 0, areaLight),
		tri(0, 0, 0,  1, 0, 1,  0, 0, 1, 0, areaLight),
	};
}

double len(const double *v) {
	return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

} // namespace

TEST(PbrtQuadify, MergesTheTwoHalvesOfAnEmissiveSquare) {
	const pbrt_quadify::Result r = pbrt_quadify::quadify(squareHalves());
	ASSERT_EQ(r.quads.size(), 1u);
	EXPECT_TRUE(r.leftover.empty());

	const pbrt_quadify::Quad &q = r.quads[0];
	EXPECT_NEAR(len(q.u), 1.0, 1e-12);
	EXPECT_NEAR(len(q.v), 1.0, 1e-12);
	// Q + u + v must land on the far corner, which is the whole point of the
	// representation.
	for (int k = 0; k < 3; ++k) {
		const double corner = q.Q[k] + q.u[k] + q.v[k];
		const double expected[3] = {1, 0, 1};
		EXPECT_NEAR(corner, expected[k], 1e-12);
	}
}

TEST(PbrtQuadify, CarriesTheMaterialAndLightThrough) {
	std::vector<pbrt_flatten::Triangle> t = squareHalves(7);
	t[0].material = t[1].material = 3;
	const pbrt_quadify::Result r = pbrt_quadify::quadify(t);
	ASSERT_EQ(r.quads.size(), 1u);
	EXPECT_EQ(r.quads[0].material, 3);
	EXPECT_EQ(r.quads[0].areaLight, 7);
}

TEST(PbrtQuadify, LeavesNonEmissiveGeometryAlone) {
	// Ordinary geometry renders correctly as triangles, so merging it would be
	// work with no beneficiary - and would lose the ability to represent
	// anything that is not a parallelogram.
	std::vector<pbrt_flatten::Triangle> t = squareHalves(-1);
	const pbrt_quadify::Result r = pbrt_quadify::quadify(t);
	EXPECT_TRUE(r.quads.empty());
	EXPECT_EQ(r.leftover.size(), 2u);
}

TEST(PbrtQuadify, RefusesTrianglesThatShareAnEdgeButAreNotAParallelogram) {
	// A kite: shares the diagonal, is planar, is convex - and is still not
	// expressible as Q + u + v. Merging it would move the emitting surface.
	std::vector<pbrt_flatten::Triangle> t = {
		tri(0, 0, 0,  1, 0, 0,  1, 0, 1, 0, 0),
		tri(0, 0, 0,  1, 0, 1,  0, 0, 5, 0, 0),   // far corner pulled out
	};
	const pbrt_quadify::Result r = pbrt_quadify::quadify(t);
	EXPECT_TRUE(r.quads.empty());
	EXPECT_EQ(r.leftover.size(), 2u);
}

TEST(PbrtQuadify, RefusesTrianglesThatDoNotShareAnEdge) {
	std::vector<pbrt_flatten::Triangle> t = {
		tri(0, 0, 0,  1, 0, 0,  1, 0, 1, 0, 0),
		tri(9, 0, 9,  8, 0, 9,  8, 0, 8, 0, 0),
	};
	const pbrt_quadify::Result r = pbrt_quadify::quadify(t);
	EXPECT_TRUE(r.quads.empty());
	EXPECT_EQ(r.leftover.size(), 2u);
}

TEST(PbrtQuadify, RefusesToMergeAcrossTwoDifferentLights) {
	// Two lights that happen to be adjacent must not become one quad emitting
	// somewhere between them.
	std::vector<pbrt_flatten::Triangle> t = squareHalves(0);
	t[1].areaLight = 1;
	const pbrt_quadify::Result r = pbrt_quadify::quadify(t);
	EXPECT_TRUE(r.quads.empty());
	EXPECT_EQ(r.leftover.size(), 2u);
}

TEST(PbrtQuadify, RefusesToMergeAcrossTwoDifferentMaterials) {
	std::vector<pbrt_flatten::Triangle> t = squareHalves(0);
	t[1].material = 2;
	const pbrt_quadify::Result r = pbrt_quadify::quadify(t);
	EXPECT_TRUE(r.quads.empty());
	EXPECT_EQ(r.leftover.size(), 2u);
}

TEST(PbrtQuadify, ALoneEmissiveTriangleSurvivesAsATriangle) {
	// A genuine single-triangle light is legal pbrt. It must not be dropped,
	// and it must be reported as unmerged so the caller can say so.
	const std::vector<pbrt_flatten::Triangle> t = {
		tri(0, 0, 0,  1, 0, 0,  1, 0, 1, 0, 0)};
	const pbrt_quadify::Result r = pbrt_quadify::quadify(t);
	EXPECT_TRUE(r.quads.empty());
	ASSERT_EQ(r.leftover.size(), 1u);
	EXPECT_EQ(pbrt_quadify::unmergedEmissiveCount(r), 1u);
}

TEST(PbrtQuadify, UnmergedCountIgnoresOrdinaryGeometry) {
	// Non-emissive leftovers are the expected case, not a shortfall - counting
	// them would make every scene look like it had unsamplable lights.
	std::vector<pbrt_flatten::Triangle> t = squareHalves(-1);
	const pbrt_quadify::Result r = pbrt_quadify::quadify(t);
	EXPECT_EQ(pbrt_quadify::unmergedEmissiveCount(r), 0u);
}

TEST(PbrtQuadify, MergesAtCornellBoxScaleWhereAFixedEpsilonWouldNot) {
	// The bundled example's ceiling light, at pbrt's usual 0-555 scale. A
	// tolerance that does not scale with the coordinates either rejects this
	// or accepts things it should not, depending which constant was picked.
	const std::vector<pbrt_flatten::Triangle> t = {
		tri(213, 548.7, 227,  343, 548.7, 227,  343, 548.7, 332, 0, 0),
		tri(213, 548.7, 227,  343, 548.7, 332,  213, 548.7, 332, 0, 0),
	};
	const pbrt_quadify::Result r = pbrt_quadify::quadify(t);
	ASSERT_EQ(r.quads.size(), 1u);
	EXPECT_NEAR(len(r.quads[0].u), 130.0, 1e-9);
	EXPECT_NEAR(len(r.quads[0].v), 105.0, 1e-9);
}

TEST(PbrtQuadify, MergesSeveralLightsAndKeepsGeometryInOrder) {
	std::vector<pbrt_flatten::Triangle> t;
	t.push_back(tri(0, 0, 0,  1, 0, 0,  0, 1, 0));            // wall
	const std::vector<pbrt_flatten::Triangle> lightA = squareHalves(0);
	t.insert(t.end(), lightA.begin(), lightA.end());
	t.push_back(tri(5, 5, 5,  6, 5, 5,  5, 6, 5));            // wall
	std::vector<pbrt_flatten::Triangle> lightB = squareHalves(1);
	for (auto &x : lightB) { x.areaLight = 1; for (int k = 0; k < 9; k += 3) x.v[k] += 10; }
	t.insert(t.end(), lightB.begin(), lightB.end());

	const pbrt_quadify::Result r = pbrt_quadify::quadify(t);
	EXPECT_EQ(r.quads.size(), 2u);
	EXPECT_EQ(r.leftover.size(), 2u);
	EXPECT_EQ(pbrt_quadify::unmergedEmissiveCount(r), 0u);
	EXPECT_EQ(r.quads[0].areaLight, 0);
	EXPECT_EQ(r.quads[1].areaLight, 1);
}
