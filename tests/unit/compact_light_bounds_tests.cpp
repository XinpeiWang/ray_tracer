// compact_light_bounds_tests.cpp
// Unit tests for src/shared/compact_light_bounds.h
//
// Groups:
//   1. Construction / accessors
//   2. Quantisation round-trip fidelity
//   3. Importance() vs LightBounds::Importance() consistency
//   4. Bounds() reconstruction
#include <gtest/gtest.h>
#include <cmath>
#include "../../src/shared/compact_light_bounds.h"

// Scene AABB used by all tests: [0,0,0] -> [10,10,10]
static const float SMIN[3] = {0.f, 0.f, 0.f};
static const float SMAX[3] = {10.f, 10.f, 10.f};

// Helper: make a LightBounds at a given sub-box, +Z axis, hemispherical
static LightBounds MakeLB(float bx0, float by0, float bz0,
                           float bx1, float by1, float bz1,
                           float phi = 10.f, bool twoSided = false)
{
    return LightBounds(bx0, by0, bz0, bx1, by1, bz1,
                       0.f, 0.f, 1.f, phi, 0.f, 0.f, twoSided);
}

// ---------------------------------------------------------------------------
// 1. Construction / accessors
// ---------------------------------------------------------------------------

TEST(CompactLightBounds, DefaultZeroPhi) {
    CompactLightBounds clb;
    EXPECT_EQ(clb.phi, 0.f);
}

TEST(CompactLightBounds, TwoSidedStored) {
    LightBounds lbT = MakeLB(1,1,1,2,2,2,5.f,true);
    LightBounds lbF = MakeLB(1,1,1,2,2,2,5.f,false);
    CompactLightBounds clbT(lbT, SMIN[0],SMIN[1],SMIN[2], SMAX[0],SMAX[1],SMAX[2]);
    CompactLightBounds clbF(lbF, SMIN[0],SMIN[1],SMIN[2], SMAX[0],SMAX[1],SMAX[2]);
    EXPECT_TRUE(clbT.TwoSided());
    EXPECT_FALSE(clbF.TwoSided());
}

TEST(CompactLightBounds, PhiPreserved) {
    LightBounds lb = MakeLB(1,1,1,2,2,2,7.5f);
    CompactLightBounds clb(lb, SMIN[0],SMIN[1],SMIN[2], SMAX[0],SMAX[1],SMAX[2]);
    EXPECT_FLOAT_EQ(clb.phi, 7.5f);
}

// ---------------------------------------------------------------------------
// 2. Quantisation round-trip fidelity
// ---------------------------------------------------------------------------

TEST(CompactLightBounds, CosThetaORoundTrip) {
    // cosTheta_o = 0.5 -> quantise -> dequantise; expect < 0.001 error
    LightBounds lb(1,1,1,2,2,2, 0,0,1, 1.f, 0.5f, 0.0f, false);
    CompactLightBounds clb(lb, SMIN[0],SMIN[1],SMIN[2], SMAX[0],SMAX[1],SMAX[2]);
    EXPECT_NEAR(clb.CosTheta_o(), 0.5f, 1e-3f);
}

TEST(CompactLightBounds, CosThetaERoundTrip) {
    LightBounds lb(1,1,1,2,2,2, 0,0,1, 1.f, 0.f, 0.3f, false);
    CompactLightBounds clb(lb, SMIN[0],SMIN[1],SMIN[2], SMAX[0],SMAX[1],SMAX[2]);
    EXPECT_NEAR(clb.CosTheta_e(), 0.3f, 1e-3f);
}

TEST(CompactLightBounds, BoundsReconstructedConservatively) {
    // qb[0] floors, qb[1] ceils -> reconstructed min <= original, max >= original
    LightBounds lb = MakeLB(2.f, 3.f, 1.f, 4.f, 6.f, 5.f);
    CompactLightBounds clb(lb, SMIN[0],SMIN[1],SMIN[2], SMAX[0],SMAX[1],SMAX[2]);
    float outMin[3], outMax[3];
    clb.Bounds(SMIN[0],SMIN[1],SMIN[2], SMAX[0],SMAX[1],SMAX[2], outMin, outMax);
    // Allow 1-ULP slack from the 16-bit quantisation step (max error = 10/65535 ~ 0.00016)
    EXPECT_LE(outMin[0], 2.f + 1e-3f);
    EXPECT_LE(outMin[1], 3.f + 1e-3f);
    EXPECT_LE(outMin[2], 1.f + 1e-3f);
    EXPECT_GE(outMax[0], 4.f - 1e-3f);
    EXPECT_GE(outMax[1], 6.f - 1e-3f);
    EXPECT_GE(outMax[2], 5.f - 1e-3f);
}

// ---------------------------------------------------------------------------
// 3. Importance() consistency with LightBounds::Importance()
// ---------------------------------------------------------------------------

TEST(CompactLightBounds, ImportanceMatchesLightBounds) {
    LightBounds lb = MakeLB(2,2,2, 3,3,3, 8.f);
    CompactLightBounds clb(lb, SMIN[0],SMIN[1],SMIN[2], SMAX[0],SMAX[1],SMAX[2]);

    float px = 2.5f, py = 2.5f, pz = 8.f;
    float impFull = Importance(lb, px,py,pz, 0,0,0);
    float impCpct = clb.Importance(px,py,pz, 0,0,0,
                                   SMIN[0],SMIN[1],SMIN[2],
                                   SMAX[0],SMAX[1],SMAX[2]);

    // Allow up to 5% relative error from quantisation
    if (impFull > 1e-6f) {
        EXPECT_NEAR(impCpct / impFull, 1.f, 0.05f);
    } else {
        EXPECT_NEAR(impCpct, impFull, 1e-6f);
    }
}

TEST(CompactLightBounds, ImportanceNeverNegative) {
    LightBounds lb = MakeLB(1,1,1,3,3,3,5.f);
    CompactLightBounds clb(lb, SMIN[0],SMIN[1],SMIN[2], SMAX[0],SMAX[1],SMAX[2]);
    float pts[4][3] = {{2,2,8},{2,2,-5},{8,2,2},{-5,2,2}};
    for (auto& p : pts) {
        EXPECT_GE(clb.Importance(p[0],p[1],p[2], 0,0,0,
                                 SMIN[0],SMIN[1],SMIN[2],
                                 SMAX[0],SMAX[1],SMAX[2]), 0.f);
    }
}

TEST(CompactLightBounds, ImportanceDecreaseWithDistance) {
    LightBounds lb = MakeLB(4,4,4,6,6,6,10.f);
    CompactLightBounds clb(lb, SMIN[0],SMIN[1],SMIN[2], SMAX[0],SMAX[1],SMAX[2]);
    float impNear = clb.Importance(5,5,7,  0,0,0, SMIN[0],SMIN[1],SMIN[2], SMAX[0],SMAX[1],SMAX[2]);
    float impFar  = clb.Importance(5,5,50, 0,0,0, SMIN[0],SMIN[1],SMIN[2], SMAX[0],SMAX[1],SMAX[2]);
    EXPECT_GT(impNear, impFar);
}

TEST(CompactLightBounds, TwoSidedImportanceBothSides) {
    LightBounds lb = MakeLB(4,4,4,6,6,6,10.f,true);
    CompactLightBounds clb(lb, SMIN[0],SMIN[1],SMIN[2], SMAX[0],SMAX[1],SMAX[2]);
    float impFront = clb.Importance(5,5,8,  0,0,0, SMIN[0],SMIN[1],SMIN[2], SMAX[0],SMAX[1],SMAX[2]);
    float impBack  = clb.Importance(5,5,-8, 0,0,0, SMIN[0],SMIN[1],SMIN[2], SMAX[0],SMAX[1],SMAX[2]);
    EXPECT_GT(impFront, 0.f);
    EXPECT_GT(impBack,  0.f);
}

// ---------------------------------------------------------------------------
// 4. Degenerate / edge cases
// ---------------------------------------------------------------------------

TEST(CompactLightBounds, ZeroPhiImportanceIsZero) {
    LightBounds lb = MakeLB(1,1,1,2,2,2, 0.f);
    CompactLightBounds clb(lb, SMIN[0],SMIN[1],SMIN[2], SMAX[0],SMAX[1],SMAX[2]);
    EXPECT_EQ(clb.Importance(5,5,5, 0,0,0,
                              SMIN[0],SMIN[1],SMIN[2],
                              SMAX[0],SMAX[1],SMAX[2]), 0.f);
}

TEST(CompactLightBounds, CosThetaExtremesRoundTrip) {
    // Test +1 and -1 (boundary values for cosine)
    LightBounds lb1(1,1,1,2,2,2, 0,0,1, 1.f, 1.f, -1.f, false);
    CompactLightBounds clb1(lb1, SMIN[0],SMIN[1],SMIN[2], SMAX[0],SMAX[1],SMAX[2]);
    EXPECT_NEAR(clb1.CosTheta_o(),  1.f, 1e-3f);
    EXPECT_NEAR(clb1.CosTheta_e(), -1.f, 1e-3f);
}
