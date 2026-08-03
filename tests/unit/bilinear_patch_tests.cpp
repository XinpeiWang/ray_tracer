// bilinear_patch_tests.cpp
// Unit tests for src/shared/bilinear_patch.h
// pbrt-v4 reference: src/pbrt/shapes.h / shapes.cpp -- BilinearPatch

#include "../../src/shared/bilinear_patch.h"
#include <gtest/gtest.h>
#include <cmath>
#include <array>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static const float kEps = 1e-4f;

static bool near(float a, float b, float tol = kEps) {
    return std::abs(a - b) <= tol * (1.f + std::abs(b));
}

// Unit square in XY plane: p00=(0,0,0), p10=(1,0,0), p01=(0,1,0), p11=(1,1,0)
static const float sq00[3] = {0,0,0};
static const float sq10[3] = {1,0,0};
static const float sq01[3] = {0,1,0};
static const float sq11[3] = {1,1,0};

// Trapezoid (non-rectangle): p11 shifted
static const float tr00[3] = {0,0,0};
static const float tr10[3] = {2,0,0};
static const float tr01[3] = {0,1,0};
static const float tr11[3] = {1,1,0};

// ---------------------------------------------------------------------------
// blp_intersect
// ---------------------------------------------------------------------------

TEST(BilinearPatch, IntersectHitsCenter) {
    // Ray from above center of unit square, pointing -z
    float ro[3]={0.5f,0.5f,1.f}, rd[3]={0,0,-1.f};
    auto h = blp_intersect(ro, rd, 10.f, sq00, sq10, sq01, sq11);
    ASSERT_TRUE(h.has_value());
    EXPECT_NEAR(h->t, 1.f, 1e-5f);
    EXPECT_NEAR(h->u, 0.5f, 1e-4f);
    EXPECT_NEAR(h->v, 0.5f, 1e-4f);
}

TEST(BilinearPatch, IntersectHitsCorner) {
    float ro[3]={0.f,0.f,1.f}, rd[3]={0,0,-1.f};
    auto h = blp_intersect(ro, rd, 10.f, sq00, sq10, sq01, sq11);
    ASSERT_TRUE(h.has_value());
    EXPECT_NEAR(h->t, 1.f, 1e-5f);
    EXPECT_NEAR(h->u, 0.f, 2e-4f);
    EXPECT_NEAR(h->v, 0.f, 2e-4f);
}

TEST(BilinearPatch, IntersectMissesOutside) {
    // Ray aimed outside the patch
    float ro[3]={2.f,2.f,1.f}, rd[3]={0,0,-1.f};
    auto h = blp_intersect(ro, rd, 10.f, sq00, sq10, sq01, sq11);
    EXPECT_FALSE(h.has_value());
}

TEST(BilinearPatch, IntersectMissesBeyondTMax) {
    float ro[3]={0.5f,0.5f,5.f}, rd[3]={0,0,-1.f};
    auto h = blp_intersect(ro, rd, 2.f, sq00, sq10, sq01, sq11);
    EXPECT_FALSE(h.has_value());
}

TEST(BilinearPatch, IntersectObliqueRay) {
    // Oblique ray: starts at (0,0,2), direction (0.5,0.5,-2) normalized
    float d[3] = {0.5f, 0.5f, -2.f};
    float dlen = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
    d[0]/=dlen; d[1]/=dlen; d[2]/=dlen;
    float ro[3]={0,0,2.f};
    auto h = blp_intersect(ro, d, 10.f, sq00, sq10, sq01, sq11);
    ASSERT_TRUE(h.has_value());
    EXPECT_GT(h->t, 0.f);
    EXPECT_GE(h->u, 0.f); EXPECT_LE(h->u, 1.f);
    EXPECT_GE(h->v, 0.f); EXPECT_LE(h->v, 1.f);
}

TEST(BilinearPatch, IntersectTrapezoidCenter) {
    float ro[3]={0.75f,0.5f,1.f}, rd[3]={0,0,-1.f};
    auto h = blp_intersect(ro, rd, 10.f, tr00, tr10, tr01, tr11);
    ASSERT_TRUE(h.has_value());
    EXPECT_NEAR(h->t, 1.f, 1e-5f);
}

// ---------------------------------------------------------------------------
// blp_point / blp_normal
// ---------------------------------------------------------------------------

TEST(BilinearPatch, PointAtCorners) {
    float p[3];
    blp_point(sq00, sq10, sq01, sq11, 0,0, p, nullptr, nullptr);
    EXPECT_NEAR(p[0],0,1e-6f); EXPECT_NEAR(p[1],0,1e-6f); EXPECT_NEAR(p[2],0,1e-6f);

    blp_point(sq00, sq10, sq01, sq11, 1,0, p, nullptr, nullptr);
    EXPECT_NEAR(p[0],1,1e-6f); EXPECT_NEAR(p[1],0,1e-6f);

    blp_point(sq00, sq10, sq01, sq11, 0,1, p, nullptr, nullptr);
    EXPECT_NEAR(p[0],0,1e-6f); EXPECT_NEAR(p[1],1,1e-6f);

    blp_point(sq00, sq10, sq01, sq11, 1,1, p, nullptr, nullptr);
    EXPECT_NEAR(p[0],1,1e-6f); EXPECT_NEAR(p[1],1,1e-6f);
}

TEST(BilinearPatch, PointAtCenter) {
    float p[3];
    blp_point(sq00, sq10, sq01, sq11, 0.5f,0.5f, p, nullptr, nullptr);
    EXPECT_NEAR(p[0], 0.5f, 1e-6f);
    EXPECT_NEAR(p[1], 0.5f, 1e-6f);
    EXPECT_NEAR(p[2], 0.f,  1e-6f);
}

TEST(BilinearPatch, NormalUnitSquareIsZ) {
    float n[3];
    blp_normal(sq00, sq10, sq01, sq11, 0.5f, 0.5f, n);
    EXPECT_NEAR(std::abs(n[2]), 1.f, 1e-5f);
    EXPECT_NEAR(n[0], 0.f, 1e-5f);
    EXPECT_NEAR(n[1], 0.f, 1e-5f);
}

TEST(BilinearPatch, NormalIsUnitLength) {
    for (float u : {0.f, 0.3f, 0.7f, 1.f}) {
        for (float v : {0.f, 0.3f, 0.7f, 1.f}) {
            float n[3];
            blp_normal(tr00, tr10, tr01, tr11, u, v, n);
            float len2 = n[0]*n[0]+n[1]*n[1]+n[2]*n[2];
            EXPECT_NEAR(len2, 1.f, 1e-4f) << "u=" << u << " v=" << v;
        }
    }
}

// ---------------------------------------------------------------------------
// blp_is_rectangle
// ---------------------------------------------------------------------------

TEST(BilinearPatch, UnitSquareIsRectangle) {
    EXPECT_TRUE(blp_is_rectangle(sq00, sq10, sq01, sq11));
}

TEST(BilinearPatch, TrapezoidNotRectangle) {
    EXPECT_FALSE(blp_is_rectangle(tr00, tr10, tr01, tr11));
}

TEST(BilinearPatch, RectangleNotAxisAligned) {
    // 45-degree rotated rectangle
    float r00[3]={0,0,0}, r10[3]={1,1,0}, r01[3]={-1,1,0}, r11[3]={0,2,0};
    EXPECT_TRUE(blp_is_rectangle(r00, r10, r01, r11));
}

TEST(BilinearPatch, NonPlanarNotRectangle) {
    float p00[3]={0,0,0}, p10[3]={1,0,0}, p01[3]={0,1,0}, p11[3]={1,1,1};
    EXPECT_FALSE(blp_is_rectangle(p00, p10, p01, p11));
}

// ---------------------------------------------------------------------------
// blp_area
// ---------------------------------------------------------------------------

TEST(BilinearPatch, AreaUnitSquare) {
    EXPECT_NEAR(blp_area(sq00, sq10, sq01, sq11), 1.f, 1e-4f);
}

TEST(BilinearPatch, AreaScaledRectangle) {
    float p00[3]={0,0,0}, p10[3]={3,0,0}, p01[3]={0,4,0}, p11[3]={3,4,0};
    EXPECT_NEAR(blp_area(p00, p10, p01, p11), 12.f, 1e-3f);
}

TEST(BilinearPatch, AreaTrapezoid) {
    // Trapezoid area = 0.5*(2+1)*1 = 1.5
    float area = blp_area(tr00, tr10, tr01, tr11);
    EXPECT_NEAR(area, 1.5f, 5e-3f);
}

TEST(BilinearPatch, AreaPositive) {
    EXPECT_GT(blp_area(tr00, tr10, tr01, tr11), 0.f);
    EXPECT_GT(blp_area(sq00, sq10, sq01, sq11), 0.f);
}

// ---------------------------------------------------------------------------
// blp_sample
// ---------------------------------------------------------------------------

TEST(BilinearPatch, SamplePointOnPatch) {
    float u2[2] = {0.5f, 0.5f};
    float p[3], n[3], pdf;
    blp_sample(sq00, sq10, sq01, sq11, u2, p, n, &pdf);
    EXPECT_NEAR(p[2], 0.f, 1e-5f);
    EXPECT_GE(p[0], -1e-4f); EXPECT_LE(p[0], 1+1e-4f);
    EXPECT_GE(p[1], -1e-4f); EXPECT_LE(p[1], 1+1e-4f);
}

TEST(BilinearPatch, SampleNormalIsUnit) {
    float u2[2] = {0.3f, 0.7f};
    float p[3], n[3], pdf;
    blp_sample(sq00, sq10, sq01, sq11, u2, p, n, &pdf);
    float len2 = n[0]*n[0]+n[1]*n[1]+n[2]*n[2];
    EXPECT_NEAR(len2, 1.f, 1e-4f);
}

TEST(BilinearPatch, SamplePdfPositive) {
    float u2[2] = {0.5f, 0.5f};
    float p[3], n[3], pdf;
    blp_sample(sq00, sq10, sq01, sq11, u2, p, n, &pdf);
    EXPECT_GT(pdf, 0.f);
}

TEST(BilinearPatch, SamplePdfApproxInverseArea) {
    // For a unit square, area-sampling pdf should be ~1/area = 1
    float u2[2] = {0.5f, 0.5f};
    float p[3], n[3], pdf;
    blp_sample(sq00, sq10, sq01, sq11, u2, p, n, &pdf);
    EXPECT_NEAR(pdf, 1.f, 1e-3f);
}

TEST(BilinearPatch, SampleVaryingU) {
    // Multiple sample points should all land on the patch
    float pts[4][2] = {{0.1f,0.1f},{0.2f,0.8f},{0.9f,0.3f},{0.6f,0.6f}};
    for (auto& u2 : pts) {
        float p[3], n[3], pdf;
        blp_sample(tr00, tr10, tr01, tr11, u2, p, n, &pdf);
        EXPECT_GE(p[1], -1e-4f); EXPECT_LE(p[1], 1+1e-4f);
        EXPECT_GT(pdf, 0.f);
    }
}

// ---------------------------------------------------------------------------
// blp_pdf_wi
// ---------------------------------------------------------------------------

TEST(BilinearPatch, PdfWiHitIsPositive) {
    float ref[3] = {0.5f, 0.5f, 2.f};
    float wi[3]  = {0.f, 0.f, -1.f};
    float pdf = blp_pdf_wi(sq00, sq10, sq01, sq11, ref, wi);
    EXPECT_GT(pdf, 0.f);
}

TEST(BilinearPatch, PdfWiMissIsZero) {
    float ref[3] = {5.f, 5.f, 2.f};
    float wi[3]  = {0.f, 0.f, -1.f};
    float pdf = blp_pdf_wi(sq00, sq10, sq01, sq11, ref, wi);
    EXPECT_EQ(pdf, 0.f);
}

TEST(BilinearPatch, PdfWiConsistentWithArea) {
    // For a 1x1 patch at z=0, ref at (0.5,0.5,2):
    // area_pdf = 1, dist2 = 4, cos_theta = 1 => solid_angle_pdf = 4
    float ref[3] = {0.5f, 0.5f, 2.f};
    float wi[3]  = {0.f, 0.f, -1.f};
    float pdf = blp_pdf_wi(sq00, sq10, sq01, sq11, ref, wi);
    EXPECT_NEAR(pdf, 4.f, 0.01f);
}

// ---------------------------------------------------------------------------
// Intersection UV continuity
// ---------------------------------------------------------------------------

TEST(BilinearPatch, UVContinuous) {
    // Sample a grid of rays through the patch and check u,v in [0,1]
    for (int i = 0; i <= 5; ++i) {
        for (int j = 0; j <= 5; ++j) {
            float xu = float(i)/5.f, xv = float(j)/5.f;
            float ro[3] = {xu, xv, 2.f};
            float rd[3] = {0,0,-1.f};
            auto h = blp_intersect(ro, rd, 10.f, sq00, sq10, sq01, sq11);
            if (h) {
                EXPECT_GE(h->u, -1e-4f); EXPECT_LE(h->u, 1+1e-4f);
                EXPECT_GE(h->v, -1e-4f); EXPECT_LE(h->v, 1+1e-4f);
            }
        }
    }
}
