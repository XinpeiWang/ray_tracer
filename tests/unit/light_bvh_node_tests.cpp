// light_bvh_node_tests.cpp
// Unit tests for src/shared/light_bvh_node.h
//
// Groups:
//   1. MakeLeaf / MakeInterior factories
//   2. isLeaf flag
//   3. childOrLightIndex storage
//   4. lightBounds roundtrip
//   5. alignas(32) size/alignment check
#include <gtest/gtest.h>
#include <cstddef>
#include "../../src/shared/light_bvh_node.h"

// Scene AABB for CompactLightBounds construction
static const float S0 = 0.f, S1 = 10.f;

static CompactLightBounds MakeCLB(float phi = 5.f)
{
    LightBounds lb(1,1,1, 2,2,2, 0,0,1, phi, 0.f, 0.f, false);
    return CompactLightBounds(lb, S0,S0,S0, S1,S1,S1);
}

// ---------------------------------------------------------------------------
// 1 & 2. MakeLeaf factory and isLeaf flag
// ---------------------------------------------------------------------------

TEST(LightBVHNode, MakeLeafSetsIsLeaf) {
    auto clb = MakeCLB();
    auto node = LightBVHNode::MakeLeaf(42u, clb);
    EXPECT_EQ(node.isLeaf, 1u);
}

TEST(LightBVHNode, MakeInteriorClearsIsLeaf) {
    auto clb = MakeCLB();
    auto node = LightBVHNode::MakeInterior(7u, clb);
    EXPECT_EQ(node.isLeaf, 0u);
}

// ---------------------------------------------------------------------------
// 3. childOrLightIndex storage
// ---------------------------------------------------------------------------

TEST(LightBVHNode, MakeLeafStoresLightIndex) {
    auto clb = MakeCLB();
    auto node = LightBVHNode::MakeLeaf(99u, clb);
    EXPECT_EQ(node.childOrLightIndex, 99u);
}

TEST(LightBVHNode, MakeInteriorStoresChild1Index) {
    auto clb = MakeCLB();
    auto node = LightBVHNode::MakeInterior(512u, clb);
    EXPECT_EQ(node.childOrLightIndex, 512u);
}

TEST(LightBVHNode, MaxLightIndex) {
    // 31-bit field: max value = 2^31 - 1 = 2147483647
    auto clb = MakeCLB();
    unsigned int maxIdx = (1u << 31) - 1u;
    auto node = LightBVHNode::MakeLeaf(maxIdx, clb);
    EXPECT_EQ(node.childOrLightIndex, maxIdx);
}

// ---------------------------------------------------------------------------
// 4. lightBounds stored and accessible
// ---------------------------------------------------------------------------

TEST(LightBVHNode, LightBoundsPhi) {
    auto clb = MakeCLB(8.f);
    auto node = LightBVHNode::MakeLeaf(0u, clb);
    EXPECT_FLOAT_EQ(node.lightBounds.phi, 8.f);
}

TEST(LightBVHNode, LightBoundsTwoSided) {
    LightBounds lb(1,1,1,2,2,2, 0,0,1, 1.f, 0.f, 0.f, true);
    CompactLightBounds clb(lb, S0,S0,S0, S1,S1,S1);
    auto node = LightBVHNode::MakeInterior(3u, clb);
    EXPECT_TRUE(node.lightBounds.TwoSided());
}

TEST(LightBVHNode, ImportanceForwardedCorrectly) {
    // Leaf node importance should match the CompactLightBounds directly
    auto clb = MakeCLB(10.f);
    auto node = LightBVHNode::MakeLeaf(0u, clb);
    float imp = node.lightBounds.Importance(1.5f, 1.5f, 8.f,
                                             0,0,0,
                                             S0,S0,S0, S1,S1,S1);
    EXPECT_GT(imp, 0.f);
}

// ---------------------------------------------------------------------------
// 5. alignas(32) and size
// ---------------------------------------------------------------------------

TEST(LightBVHNode, AlignmentIs32) {
    EXPECT_EQ(alignof(LightBVHNode), 32u);
}

TEST(LightBVHNode, SizeIsMultipleOf32) {
    // pbrt-v4 uses alignas(32); size should be a multiple of 32
    EXPECT_EQ(sizeof(LightBVHNode) % 32u, 0u);
}

TEST(LightBVHNode, DefaultConstructible) {
    LightBVHNode node;
    (void)node;  // must compile and not crash
}
