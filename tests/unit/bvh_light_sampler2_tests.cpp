// bvh_light_sampler2_tests.cpp
// Unit tests for src/shared/bvh_light_sampler2.h
//
// Groups:
//   1. Construction / empty
//   2. Single-light: PMF == 1, Sample returns correct index
//   3. Two equal lights: each gets ~0.5 PMF from a neutral query point
//   4. Two lights at different distances: nearer light gets higher PMF
//   5. PMF sums to 1 over all lights (many lights)
//   6. PMF replay matches Sample pmf
#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>
#include "../../src/shared/bvh_light_sampler2.h"

// Helper: make a unit-cube LightBounds at position (ox,oy,oz), +Z axis, phi
static LightBounds MakeLB(float ox, float oy, float oz, float phi = 10.f, bool twoSided = false)
{
    return LightBounds(ox, oy, oz, ox+1.f, oy+1.f, oz+1.f,
                       0.f, 0.f, 1.f, phi, 0.f, 0.f, twoSided);
}

// ---------------------------------------------------------------------------
// 1. Construction / empty
// ---------------------------------------------------------------------------

TEST(BVHLightSampler2, DefaultIsEmpty) {
    BVHLightSampler2 s;
    EXPECT_TRUE(s.Empty());
}

TEST(BVHLightSampler2, ZeroPhiLightsProduceEmptySampler) {
    LightBounds lb = MakeLB(0,0,0, 0.f);
    BVHLightSampler2 s(&lb, 1);
    EXPECT_TRUE(s.Empty());
}

TEST(BVHLightSampler2, SingleLightNotEmpty) {
    LightBounds lb = MakeLB(0,0,0, 5.f);
    BVHLightSampler2 s(&lb, 1);
    EXPECT_FALSE(s.Empty());
    EXPECT_EQ(s.NodeCount(), 1);
}

// ---------------------------------------------------------------------------
// 2. Single light: PMF == 1, Sample returns light 0
// ---------------------------------------------------------------------------

TEST(BVHLightSampler2, SingleLightPMFIsOne) {
    LightBounds lb = MakeLB(0,0,0, 8.f);
    BVHLightSampler2 s(&lb, 1);
    float pmf = s.PMF(0.5f,0.5f,5.f, 0,0,0, 0);
    EXPECT_NEAR(pmf, 1.f, 1e-5f);
}

TEST(BVHLightSampler2, SingleLightSampleReturnsIndex0) {
    LightBounds lb = MakeLB(0,0,0, 8.f);
    BVHLightSampler2 s(&lb, 1);
    auto result = s.Sample(0.5f,0.5f,5.f, 0,0,0, 0.5f);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->lightIndex, 0);
    EXPECT_NEAR(result->pmf, 1.f, 1e-5f);
}

// ---------------------------------------------------------------------------
// 3. Two identical lights: each ~0.5 PMF from equidistant point
// ---------------------------------------------------------------------------

TEST(BVHLightSampler2, TwoEqualLightsEachHalfPMF) {
    LightBounds lbs[2] = { MakeLB(0,0,0, 5.f), MakeLB(4,0,0, 5.f) };
    BVHLightSampler2 s(lbs, 2);
    // Query from equidistant point between the two lights
    float p0 = s.PMF(2.f, 0.5f, 5.f, 0,0,0, 0);
    float p1 = s.PMF(2.f, 0.5f, 5.f, 0,0,0, 1);
    EXPECT_NEAR(p0 + p1, 1.f, 1e-4f);
    // Each should be close to 0.5 from a symmetric point
    EXPECT_NEAR(p0, 0.5f, 0.1f);
    EXPECT_NEAR(p1, 0.5f, 0.1f);
}

// ---------------------------------------------------------------------------
// 4. PMF sums to 1 over all lights (many lights)
// ---------------------------------------------------------------------------

TEST(BVHLightSampler2, PMFSumsToOne) {
    const int N = 8;
    LightBounds lbs[N];
    for (int i = 0; i < N; ++i)
        lbs[i] = MakeLB((float)(i*3), 0, 0, (float)(i+1));
    BVHLightSampler2 s(lbs, N);

    float total = 0.f;
    for (int i = 0; i < N; ++i)
        total += s.PMF(12.f, 0.5f, 8.f, 0,0,0, i);
    EXPECT_NEAR(total, 1.f, 1e-4f);
}

// ---------------------------------------------------------------------------
// 5. Sample PMF matches PMF() replay
// ---------------------------------------------------------------------------

TEST(BVHLightSampler2, SamplePMFMatchesReplay) {
    const int N = 6;
    LightBounds lbs[N];
    for (int i = 0; i < N; ++i)
        lbs[i] = MakeLB((float)(i*2), 0, 0, 5.f);
    BVHLightSampler2 s(lbs, N);

    float px = 5.f, py = 0.5f, pz = 8.f;
    // Try several u values
    float uVals[] = {0.05f, 0.25f, 0.5f, 0.75f, 0.95f};
    for (float u : uVals) {
        auto result = s.Sample(px,py,pz, 0,0,0, u);
        if (!result.has_value()) continue;
        float replay = s.PMF(px,py,pz, 0,0,0, result->lightIndex);
        EXPECT_NEAR(result->pmf, replay, 1e-4f)
            << "u=" << u << " lightIndex=" << result->lightIndex;
    }
}

// ---------------------------------------------------------------------------
// 6. Brighter light selected more often (stochastic, 10000 samples)
// ---------------------------------------------------------------------------

TEST(BVHLightSampler2, BrightLightSelectedMoreOften) {
    // Light 0: phi=1, Light 1: phi=9 (9x brighter)
    LightBounds lbs[2] = { MakeLB(0,0,0,1.f), MakeLB(4,0,0,9.f) };
    BVHLightSampler2 s(lbs, 2);

    int count1 = 0;
    const int N = 10000;
    for (int i = 0; i < N; ++i) {
        float u = (i + 0.5f) / N;
        auto r = s.Sample(2.f, 0.5f, 8.f, 0,0,0, u);
        if (r.has_value() && r->lightIndex == 1) ++count1;
    }
    // Bright light should be chosen >> 50% of the time
    EXPECT_GT(count1, N / 2);
}

// ---------------------------------------------------------------------------
// 7. PMF of unknown light is 0
// ---------------------------------------------------------------------------

TEST(BVHLightSampler2, UnknownLightPMFIsZero) {
    LightBounds lb = MakeLB(0,0,0,5.f);
    BVHLightSampler2 s(&lb, 1);
    EXPECT_EQ(s.PMF(0.5f,0.5f,5.f, 0,0,0, 99), 0.f);
}

// ---------------------------------------------------------------------------
// 8. Node array invariants (GPU-side traversal contract)
//
// buildBVH()'s recursive shape (allocate an interior placeholder, then
// recurse left-then-right, each recursion's own first action allocating
// its node at whatever nodes_.size() is at that moment) means the LAST
// node index in the flat array is always reached at the bottom of the
// right-most recursion chain - i.e. it must always be a leaf, for any
// light count/tree shape. gpu_light_bvh_sample_index()/gpu_light_bvh_pmf()
// (optix_device_helpers_lighting.h) both traverse via nodeIndex+1 (left,
// implicit) / node.childOrLightIndex (right, explicit) exactly like
// BVHLightSampler2's own host-side Sample()/PMF() below - this invariant
// is what a code-review round's own GPU diagnosis (a real, previously-
// unresolved illegal-memory-access crash, pbrt_scenes/triangle-fan-light.
// pbrt) empirically confirmed on real hardware for a 5-light/9-node tree
// (the host- and device-side node dumps matched, ruling out a struct-
// layout/ABI mismatch) - kept here as a permanent regression test for the
// invariant the traversal code relies on, independent of the GPU crash
// itself (which the device-side bounds guards now handle defensively).
// ---------------------------------------------------------------------------

TEST(BVHLightSampler2, LastNodeIsAlwaysALeaf) {
    for (int n = 1; n <= 12; ++n) {
        std::vector<LightBounds> lights;
        for (int i = 0; i < n; ++i)
            lights.push_back(MakeLB(static_cast<float>(i) * 2.f, 0.f, 0.f, 5.f + static_cast<float>(i)));
        BVHLightSampler2 s(lights.data(), n);
        ASSERT_GT(s.NodeCount(), 0) << "n=" << n;
        EXPECT_TRUE(s.Nodes()[s.NodeCount() - 1].isLeaf)
            << "n=" << n << " nodeCount=" << s.NodeCount();
    }
}

TEST(BVHLightSampler2, FiveLightNodeCountMatchesTheRealCrashScene) {
    // pbrt_scenes/triangle-fan-light.pbrt has exactly 5 lights; a full
    // binary tree over N leaves has 2N-1 nodes total.
    std::vector<LightBounds> lights;
    for (int i = 0; i < 5; ++i)
        lights.push_back(MakeLB(static_cast<float>(i) * 2.f, 0.f, 0.f, 5.f + static_cast<float>(i)));
    BVHLightSampler2 s(lights.data(), 5);
    EXPECT_EQ(s.NodeCount(), 9);
    EXPECT_TRUE(s.Nodes()[8].isLeaf);
}
