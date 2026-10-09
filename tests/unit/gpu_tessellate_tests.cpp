// gpu_tessellate_tests.cpp - src/shared/gpu_tessellate.h: the shapes a GPU backend cannot trace become triangles, and only the kinds the backend asks for.
#include <gtest/gtest.h>

#include <cmath>

#include "../../src/shared/gpu_tessellate.h"

using namespace gpu_tessellate;

namespace {

double triangleArea(const pbrt_flatten::Triangle& t) {
	const double ax = t.v[3] - t.v[0], ay = t.v[4] - t.v[1], az = t.v[5] - t.v[2];
	const double bx = t.v[6] - t.v[0], by = t.v[7] - t.v[1], bz = t.v[8] - t.v[2];
	const double cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
	return 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
}

double totalArea(const pbrt_flatten::FlatScene& s) {
	double a = 0;
	for (const auto& t : s.triangles) a += triangleArea(t);
	return a;
}

pbrt_flatten::FlatScene sceneWithEveryKind() {
	pbrt_flatten::FlatScene s;
	pbrt_flatten::BilinearPatch bp;   // a flat 2 x 3 rectangle: its area is exact
	const double c[4][3] = {{0, 0, 0}, {2, 0, 0}, {0, 3, 0}, {2, 3, 0}};
	for (int i = 0; i < 4; ++i)
		for (int a = 0; a < 3; ++a) bp.p[i][a] = c[i][a];
	s.bilinearPatches.push_back(bp);
	pbrt_flatten::Cone cone;   // radius 1, height 2
	cone.radius = 1.0;
	cone.height = 2.0;
	s.cones.push_back(cone);
	pbrt_flatten::Paraboloid pa;
	pa.radius = 1.0;
	pa.zMin = 0.0;
	pa.zMax = 1.0;
	s.paraboloids.push_back(pa);
	pbrt_flatten::Curve cv;   // one straight Bezier segment
	cv.nSegments = 1;
	cv.cp = {0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0};
	cv.width0 = cv.width1 = 0.2;
	s.curves.push_back(cv);
	return s;
}

}  // namespace

TEST(GpuTessellateTest, AskingForEverythingTurnsEveryKindIntoTriangles) {
	pbrt_flatten::FlatScene s = sceneWithEveryKind();
	FiberTangents tangents;
	const size_t added = tessellateForBackend(s, TessellationCaps::all(), &tangents);
	EXPECT_EQ(added, s.triangles.size());
	// 16x16 patch cells, 48x4 cone cells, 48x24 paraboloid cells (2 triangles each) plus the curve's tube.
	EXPECT_GE(added, size_t(2 * (16 * 16 + 48 * 4 + 48 * 24)));
	EXPECT_TRUE(s.bilinearPatches.empty() && s.cones.empty() && s.paraboloids.empty() && s.curves.empty());
	EXPECT_FALSE(tangents.empty()) << "the curve's triangles carry their fibre direction";
	for (const auto& kv : tangents) {
		ASSERT_LT(static_cast<size_t>(kv.first), s.triangles.size());
		const auto& d = kv.second;
		EXPECT_NEAR(d[0] * d[0] + d[1] * d[1] + d[2] * d[2], 1.0, 1e-4);
	}
}

TEST(GpuTessellateTest, OnlyTheRequestedKindsAreRewritten) {
	pbrt_flatten::FlatScene s = sceneWithEveryKind();
	const size_t added = tessellateForBackend(s, TessellationCaps::conesAndParaboloids(), nullptr);
	EXPECT_EQ(added, size_t(2 * (48 * 4 + 48 * 24)));
	EXPECT_TRUE(s.cones.empty() && s.paraboloids.empty());
	EXPECT_EQ(s.bilinearPatches.size(), 1u) << "a backend with native patches keeps them";
	EXPECT_EQ(s.curves.size(), 1u) << "and its own curve dicing";
	pbrt_flatten::FlatScene none = sceneWithEveryKind();
	EXPECT_EQ(tessellateForBackend(none, TessellationCaps{}, nullptr), 0u);
	EXPECT_TRUE(none.triangles.empty());
}

TEST(GpuTessellateTest, AFlatPatchKeepsItsAreaAndAConeIsCloseToTheTrueOne) {
	pbrt_flatten::FlatScene patch = sceneWithEveryKind();
	patch.cones.clear();
	patch.paraboloids.clear();
	patch.curves.clear();
	tessellateForBackend(patch, TessellationCaps::all(), nullptr);
	EXPECT_NEAR(totalArea(patch), 6.0, 1e-9);   // 2 x 3
	pbrt_flatten::FlatScene cone = sceneWithEveryKind();
	cone.bilinearPatches.clear();
	cone.paraboloids.clear();
	cone.curves.clear();
	tessellateForBackend(cone, TessellationCaps::all(), nullptr);
	const double slant = std::sqrt(1.0 + 4.0);   // radius 1, height 2
	const double lateral = 3.14159265358979 * 1.0 * slant;
	EXPECT_NEAR(totalArea(cone), lateral, lateral * 0.01) << "within 1% of pi * r * slant";
}

TEST(GpuTessellateTest, AShapeOfAnInterfaceMaterialIsSkipped) {
	pbrt_flatten::FlatScene s = sceneWithEveryKind();
	pbrt_flatten::Material iface;
	iface.kind = pbrt_flatten::MaterialKind::Interface;
	s.materials.push_back(iface);
	s.cones[0].material = 0;
	s.paraboloids[0].material = 0;
	const size_t added = tessellateForBackend(s, TessellationCaps::all(), nullptr);
	EXPECT_LT(added, size_t(2 * (48 * 4 + 48 * 24)));   // neither quadric was turned into triangles; the patch and curve were
	EXPECT_TRUE(s.cones.empty() && s.paraboloids.empty()) << "they are dropped, not left for the backend";
}

TEST(GpuTessellateTest, MaterialAndAreaLightCarryOverToTheTriangles) {
	pbrt_flatten::FlatScene s = sceneWithEveryKind();
	s.cones[0].material = 3;
	s.cones[0].areaLight = 5;
	s.bilinearPatches.clear();
	s.paraboloids.clear();
	s.curves.clear();
	tessellateForBackend(s, TessellationCaps::all(), nullptr);
	ASSERT_FALSE(s.triangles.empty());
	for (const auto& t : s.triangles) {
		EXPECT_EQ(t.material, 3);
		EXPECT_EQ(t.areaLight, 5);
	}
}
