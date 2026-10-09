// nanovdb_dense_tests.cpp - src/shared/nanovdb_dense.h: a NanoVDB file becomes a dense density grid with a world placement. Run from the repository root: the bundled .nvdb
// files are found by a relative path. (The OptiX builder side is in pbrt_gpu_nanovdb_tests.cpp, which needs the OptiX SDK.)
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include "../../src/shared/nanovdb_dense.h"

namespace {

const char* kSphere = "pbrt_scenes/nanovdb-sphere.nvdb";
const char* kFire = "pbrt_scenes/nanovdb-fire-test.nvdb";

const double kIdentity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

}  // namespace

TEST(NanoVdbDenseTest, TheBundledFogSphereReadsAsADenseGridDensestAtTheCentre) {
	const nanovdb_dense::Grid g = nanovdb_dense::readGrid(kSphere, "density", "");
	ASSERT_TRUE(g.ok());
	ASSERT_GT(g.nx, 1);
	EXPECT_EQ(g.density.size(), static_cast<size_t>(g.nx) * g.ny * g.nz);
	EXPECT_TRUE(g.temperature.empty());
	const auto at = [&](int x, int y, int z) { return g.density[(static_cast<size_t>(z) * g.ny + y) * g.nx + x]; };
	const float centre = at(g.nx / 2, g.ny / 2, g.nz / 2);
	EXPECT_GT(centre, 0.5f) << "a fog-volume sphere is dense in the middle";
	EXPECT_LT(at(0, 0, 0), centre) << "and falls off toward the corner";
	EXPECT_EQ(*std::max_element(g.density.begin(), g.density.end()) >= centre, true);
	// Reading it again gives the same numbers (no hidden state).
	EXPECT_EQ(nanovdb_dense::readGrid(kSphere, "density", "").density, g.density);
}

TEST(NanoVdbDenseTest, AMissingFileOrGridNameIsAnEmptyGridNotACrash) {
	EXPECT_FALSE(nanovdb_dense::readGrid("pbrt_scenes/does-not-exist.nvdb", "density", "").ok());
	EXPECT_FALSE(nanovdb_dense::readGrid(kSphere, "no-such-grid", "").ok());
}

TEST(NanoVdbDenseTest, ATemperatureGridIsReadAtTheSameVoxels) {
	const nanovdb_dense::Grid g = nanovdb_dense::readGrid(kFire, "density", "temperature");
	ASSERT_TRUE(g.ok());
	ASSERT_EQ(g.temperature.size(), g.density.size());
	EXPECT_NEAR(*std::max_element(g.density.begin(), g.density.end()), 0.5f, 1e-6f);
	EXPECT_NEAR(*std::max_element(g.temperature.begin(), g.temperature.end()), 3000.0f, 1e-3f);
	// A temperature grid that is not there only drops the emission.
	const nanovdb_dense::Grid noTemp = nanovdb_dense::readGrid(kFire, "density", "no-such-grid");
	ASSERT_TRUE(noTemp.ok());
	EXPECT_TRUE(noTemp.temperature.empty());
}

TEST(NanoVdbDenseTest, ThePlacementFollowsTheSceneTransform) {
	const nanovdb_dense::Grid g = nanovdb_dense::readGrid(kSphere, "density", "");
	ASSERT_TRUE(g.ok());
	nanovdb_dense::Placement plain, moved;
	ASSERT_TRUE(nanovdb_dense::placeInWorld(g, kIdentity, plain));
	// Scale 40 about the origin, then translate (278, 280, 250), as nanovdb-medium.pbrt does (row-major, translation in the last column).
	const double xf[16] = {40, 0, 0, 278, 0, 40, 0, 280, 0, 0, 40, 250, 0, 0, 0, 1};
	ASSERT_TRUE(nanovdb_dense::placeInWorld(g, xf, moved));
	for (int a = 0; a < 3; ++a) {
		EXPECT_LT(plain.worldMin[a], plain.worldMax[a]);
		const double offset[3] = {278, 280, 250};
		EXPECT_NEAR(moved.worldMin[a], 40.0 * plain.worldMin[a] + offset[a], 1e-6);
		EXPECT_NEAR(moved.worldMax[a], 40.0 * plain.worldMax[a] + offset[a], 1e-6);
	}
	// A point inside the moved box maps into [0,1]^3 medium space.
	const double c[3] = {0.5 * (moved.worldMin[0] + moved.worldMax[0]), 0.5 * (moved.worldMin[1] + moved.worldMax[1]), 0.5 * (moved.worldMin[2] + moved.worldMax[2])};
	for (int r = 0; r < 3; ++r) {
		const double u = moved.toMediumMat[r * 3] * c[0] + moved.toMediumMat[r * 3 + 1] * c[1] + moved.toMediumMat[r * 3 + 2] * c[2] + moved.toMediumTranslate[r];
		EXPECT_NEAR(u, 0.5, 1e-9);
	}
	const double flat[16] = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
	nanovdb_dense::Placement none;
	EXPECT_FALSE(nanovdb_dense::placeInWorld(g, flat, none)) << "a transform that flattens the grid cannot be inverted";
}
