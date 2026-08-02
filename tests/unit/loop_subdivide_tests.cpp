// loop_subdivide_tests.cpp
// Unit tests for src/shared/loop_subdivide.h
// pbrt-v4 reference: util/loopsubdiv.cpp

#include "../../src/shared/loop_subdivide.h"
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <vector>

static LoopSubdivResult<double> single_triangle(int levels) {
    std::vector<std::array<double, 3>> pos = {
        {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.5, 0.866025, 0.0}
    };
    return loop_subdivide<double>(pos, std::vector<int>{0, 1, 2}, levels);
}

static LoopSubdivResult<double> tetrahedron(int levels) {
    std::vector<std::array<double, 3>> pos = {
        { 1.0,  1.0,  1.0}, {-1.0, -1.0,  1.0},
        {-1.0,  1.0, -1.0}, { 1.0, -1.0, -1.0}
    };
    std::vector<int> idx = {0,1,2, 0,3,1, 0,2,3, 1,3,2};
    return loop_subdivide<double>(pos, idx, levels);
}

TEST(LoopSubdivide, ZeroLevelsTriangle) {
    auto r = single_triangle(0);
    EXPECT_EQ(r.indices.size(), 3u);
    EXPECT_EQ(r.positions.size(), 3u);
    EXPECT_EQ(r.normals.size(), 3u);
}
TEST(LoopSubdivide, ZeroLevelsTetrahedron) {
    auto r = tetrahedron(0);
    EXPECT_EQ(r.indices.size() / 3, 4u);
    EXPECT_EQ(r.positions.size(), 4u);
}
TEST(LoopSubdivide, FaceCountLevel1Triangle) {
    EXPECT_EQ(single_triangle(1).indices.size() / 3, 4u);
}
TEST(LoopSubdivide, FaceCountLevel2Triangle) {
    EXPECT_EQ(single_triangle(2).indices.size() / 3, 16u);
}
TEST(LoopSubdivide, FaceCountLevel1Tetrahedron) {
    EXPECT_EQ(tetrahedron(1).indices.size() / 3, 16u);
}
TEST(LoopSubdivide, FaceCountLevel2Tetrahedron) {
    EXPECT_EQ(tetrahedron(2).indices.size() / 3, 64u);
}
TEST(LoopSubdivide, FaceCountGrowthRule) {
    for (int lv = 0; lv <= 3; ++lv) {
        int expected = 4;
        for (int i = 0; i < lv; ++i) expected *= 4;
        EXPECT_EQ((int)(tetrahedron(lv).indices.size() / 3), expected) << "level " << lv;
    }
}
TEST(LoopSubdivide, VertexCountLevel1Triangle) {
    EXPECT_EQ(single_triangle(1).positions.size(), 6u);
}
TEST(LoopSubdivide, VertexCountLevel1Tetrahedron) {
    EXPECT_EQ(tetrahedron(1).positions.size(), 10u);
}
TEST(LoopSubdivide, NormalsNonZero) {
    auto r = tetrahedron(2);
    ASSERT_EQ(r.normals.size(), r.positions.size());
    for (size_t i = 0; i < r.normals.size(); ++i) {
        const auto& n = r.normals[i];
        double len2 = n[0]*n[0] + n[1]*n[1] + n[2]*n[2];
        EXPECT_GT(len2, 1e-20) << "zero normal at vertex " << i;
    }
}
TEST(LoopSubdivide, NormalsCountMatchesPositions) {
    for (int lv = 0; lv <= 3; ++lv) {
        auto r = tetrahedron(lv);
        EXPECT_EQ(r.normals.size(), r.positions.size()) << "level " << lv;
    }
}
TEST(LoopSubdivide, IndicesInRange) {
    auto r = tetrahedron(3);
    for (int i : r.indices) { EXPECT_GE(i, 0); EXPECT_LT(i, (int)r.positions.size()); }
}
TEST(LoopSubdivide, IndicesInRangeOpenMesh) {
    auto r = single_triangle(3);
    for (int i : r.indices) { EXPECT_GE(i, 0); EXPECT_LT(i, (int)r.positions.size()); }
}
TEST(LoopSubdivide, PositionsFinite) {
    for (const auto& p : tetrahedron(4).positions) {
        EXPECT_TRUE(std::isfinite(p[0])); EXPECT_TRUE(std::isfinite(p[1])); EXPECT_TRUE(std::isfinite(p[2]));
    }
}
TEST(LoopSubdivide, FloatPrecision) {
    std::vector<std::array<float, 3>> pos = {{0.f,0.f,0.f},{1.f,0.f,0.f},{0.5f,0.866025f,0.f}};
    auto r = loop_subdivide<float>(pos, std::vector<int>{0,1,2}, 2);
    EXPECT_EQ(r.indices.size() / 3, 16u);
}
TEST(LoopSubdivide, TetrahedronCentroidNearOrigin) {
    auto r = tetrahedron(3);
    double cx = 0, cy = 0, cz = 0;
    for (const auto& p : r.positions) { cx += p[0]; cy += p[1]; cz += p[2]; }
    double n = (double)r.positions.size();
    EXPECT_NEAR(cx/n, 0.0, 1e-10); EXPECT_NEAR(cy/n, 0.0, 1e-10); EXPECT_NEAR(cz/n, 0.0, 1e-10);
}
