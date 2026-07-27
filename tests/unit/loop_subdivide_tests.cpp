// loop_subdivide_tests.cpp -- Unit tests for pbrt-v4-style Loop subdivision
// pbrt-v4 reference: src/pbrt/util/loopsubdiv.cpp

#include "../../src/TheRestOfYourLife/loop_subdivide.h"

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a single triangle mesh (1 triangle, 3 vertices)
static std::shared_ptr<triangle_mesh_data> make_single_tri() {
	auto m = std::make_shared<triangle_mesh_data>();
	m->positions = {
		point3(0,0,0),
		point3(1,0,0),
		point3(0,1,0)
	};
	m->indices = {0,1,2};
	return m;
}

// Build a simple tetrahedron (4 faces, 4 vertices)
static std::shared_ptr<triangle_mesh_data> make_tetrahedron() {
	auto m = std::make_shared<triangle_mesh_data>();
	m->positions = {
		point3( 1, 1, 1),
		point3(-1,-1, 1),
		point3(-1, 1,-1),
		point3( 1,-1,-1)
	};
	// 4 faces, winding consistent (all CCW from outside)
	m->indices = {
		0, 1, 2,
		0, 2, 3,
		0, 3, 1,
		1, 3, 2
	};
	return m;
}

// Build a 6-sided cube from 12 triangles
static std::shared_ptr<triangle_mesh_data> make_cube() {
	auto m = std::make_shared<triangle_mesh_data>();
	// 8 corners of a unit cube
	m->positions = {
		point3(-1,-1,-1), // 0
		point3( 1,-1,-1), // 1
		point3( 1, 1,-1), // 2
		point3(-1, 1,-1), // 3
		point3(-1,-1, 1), // 4
		point3( 1,-1, 1), // 5
		point3( 1, 1, 1), // 6
		point3(-1, 1, 1), // 7
	};
	// 12 triangles (2 per face)
	m->indices = {
		// -z face
		0, 2, 1,   0, 3, 2,
		// +z face
		4, 5, 6,   4, 6, 7,
		// -x face
		0, 1, 5,   0, 5, 4,
		// +x face
		2, 3, 7,   2, 7, 6,
		// -y face
		1, 2, 6,   1, 6, 5,
		// +y face
		0, 4, 7,   0, 7, 3
	};
	return m;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// 0 levels should return the same topology
TEST(LoopSubdivide, ZeroLevels_ReturnsCopy) {
	auto in  = make_single_tri();
	auto out = loop_subdivide(*in, 0);

	ASSERT_EQ(out->positions.size(), in->positions.size());
	ASSERT_EQ(out->indices.size(),   in->indices.size());
}

// 1 level of subdivision of a single triangle -> 4 triangles
TEST(LoopSubdivide, SingleTriangle_OneLevel_FourTris) {
	auto in  = make_single_tri();
	auto out = loop_subdivide(*in, 1);

	EXPECT_EQ(out->num_triangles(), 4);
	// 6 unique vertices (3 original + 3 edge midpoints)
	EXPECT_EQ((int)out->positions.size(), 6);
}

// Index count must be exactly 3 * nTris_in * 4^nLevels
TEST(LoopSubdivide, IndexCountGrowthRule) {
	auto in = make_tetrahedron();  // 4 tris

	for (int lvl = 0; lvl <= 3; ++lvl) {
		auto out = loop_subdivide(*in, lvl);
		int expected_tris = 4;
		for (int i = 0; i < lvl; ++i) expected_tris *= 4;
		EXPECT_EQ(out->num_triangles(), expected_tris)
			<< "Level " << lvl;
	}
}

// All normals must be unit vectors
TEST(LoopSubdivide, NormalsAreUnit) {
	auto in  = make_tetrahedron();
	auto out = loop_subdivide(*in, 2);

	ASSERT_TRUE(out->has_normals());
	for (size_t i = 0; i < out->normals.size(); ++i) {
		double len = out->normals[i].length();
		EXPECT_NEAR(len, 1.0, 1e-10) << "Vertex " << i;
	}
}

// All indices must be valid
TEST(LoopSubdivide, IndicesInRange) {
	auto in  = make_cube();
	auto out = loop_subdivide(*in, 2);

	int nV = (int)out->positions.size();
	for (int idx : out->indices) {
		EXPECT_GE(idx, 0);
		EXPECT_LT(idx, nV);
	}
}

// Every vertex must appear in at least one triangle
TEST(LoopSubdivide, AllVerticesReferenced) {
	auto in  = make_tetrahedron();
	auto out = loop_subdivide(*in, 2);

	std::vector<bool> used(out->positions.size(), false);
	for (int idx : out->indices) used[idx] = true;
	for (size_t i = 0; i < used.size(); ++i)
		EXPECT_TRUE(used[i]) << "Vertex " << i << " unreferenced";
}

// After subdivision, positions of a cube's 8 original corners should move
// inward (limit surface pulls toward centroid of the shape)
TEST(LoopSubdivide, CubeLimitSurfaceContracted) {
	auto in  = make_cube();
	auto out = loop_subdivide(*in, 3);

	// All output vertices should have |p| <= sqrt(3) (original corner radius)
	double original_r = std::sqrt(3.0);
	for (const auto& p : out->positions) {
		double r = std::sqrt(p.x()*p.x() + p.y()*p.y() + p.z()*p.z());
		EXPECT_LE(r, original_r + 1e-10);
	}
}

// Subdivision of a shape symmetric about the origin should yield symmetric output
TEST(LoopSubdivide, TetrahedronCentroidNearOrigin) {
	auto in  = make_tetrahedron();
	auto out = loop_subdivide(*in, 3);

	point3 centroid(0,0,0);
	for (const auto& p : out->positions)
		centroid = centroid + (1.0 / out->positions.size()) * p;

	EXPECT_NEAR(centroid.x(), 0.0, 1e-10);
	EXPECT_NEAR(centroid.y(), 0.0, 1e-10);
	EXPECT_NEAR(centroid.z(), 0.0, 1e-10);
}

// Sanity: no degenerate (zero-area) triangles after subdivision
TEST(LoopSubdivide, NoZeroAreaTriangles) {
	auto in  = make_tetrahedron();
	auto out = loop_subdivide(*in, 2);

	for (int i = 0; i < out->num_triangles(); ++i) {
		const point3& p0 = out->positions[out->indices[3*i + 0]];
		const point3& p1 = out->positions[out->indices[3*i + 1]];
		const point3& p2 = out->positions[out->indices[3*i + 2]];
		vec3 e1 = p1 - p0;
		vec3 e2 = p2 - p0;
		double area2 = cross(e1, e2).length();
		EXPECT_GT(area2, 1e-14) << "Degenerate triangle " << i;
	}
}
