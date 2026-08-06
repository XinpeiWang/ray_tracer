// bilinear_patch_tests.cpp
// Unit tests for src/shared/bilinear_patch.h
// pbrt-v4 reference: src/pbrt/shapes.h / shapes.cpp -- BilinearPatch

#include "../../src/shared/bilinear_patch.h"
#include <gtest/gtest.h>
#include <cmath>
#include <array>

// BilinearPatchShape<T> needs ShapeHit / ShapeSample / SamplingContext from shapes.h
// (included directly by bilinear_patch.h via shapes.h)

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

// ===========================================================================
// BilinearPatchShape<T> tests (template API)
// ===========================================================================

// Unit square in XY plane: p00=(0,0,0), p10=(1,0,0), p01=(0,1,0), p11=(1,1,0)
static BilinearPatchShape<double> unit_blp() {
    return BilinearPatchShape<double>::make(
        0.0,0.0,0.0,  1.0,0.0,0.0,
        0.0,1.0,0.0,  1.0,1.0,0.0);
}

static const double BLP_PI = 3.14159265358979323846;

// -----------------------------------------------------------------------
// Area
// -----------------------------------------------------------------------
TEST(BilinearPatchShapeT, AreaUnitSquare) {
    // Unit square area = 1
    auto p = unit_blp();
    EXPECT_NEAR(p.area(), 1.0, 1e-6);
}

TEST(BilinearPatchShapeT, AreaRectangle) {
    // 2x3 rectangle area = 6
    auto p = BilinearPatchShape<double>::make(
        0,0,0,  2,0,0,
        0,3,0,  2,3,0);
    EXPECT_NEAR(p.area(), 6.0, 1e-5);
}

TEST(BilinearPatchShapeT, PdfAreaIsInverseArea) {
    auto p = unit_blp();
    EXPECT_NEAR(p.pdf_area(), 1.0 / p.area(), 1e-12);
}

// -----------------------------------------------------------------------
// Intersection
// -----------------------------------------------------------------------
TEST(BilinearPatchShapeT, IntersectHitsCenter) {
    auto p = unit_blp();
    auto h = p.intersect(0.5,0.5,2.0, 0,0,-1, 1e-4,10.0);
    ASSERT_TRUE(h.has_value());
    EXPECT_NEAR(h->t, 2.0, 1e-6);
    EXPECT_NEAR(h->u, 0.5, 1e-5);
    EXPECT_NEAR(h->v, 0.5, 1e-5);
}

TEST(BilinearPatchShapeT, IntersectNormalPointsUp) {
    auto p = unit_blp();
    auto h = p.intersect(0.5,0.5,2.0, 0,0,-1, 1e-4,10.0);
    ASSERT_TRUE(h.has_value());
    // Normal should point toward ray origin (up, +z)
    EXPECT_NEAR(std::abs(h->nz), 1.0, 1e-5);
}

TEST(BilinearPatchShapeT, IntersectMissOutside) {
    auto p = unit_blp();
    auto h = p.intersect(2.0,0.5,2.0, 0,0,-1, 1e-4,10.0);
    EXPECT_FALSE(h.has_value());
}

TEST(BilinearPatchShapeT, IntersectMissBeyondTMax) {
    auto p = unit_blp();
    auto h = p.intersect(0.5,0.5,2.0, 0,0,-1, 1e-4,1.5);
    EXPECT_FALSE(h.has_value());
}

TEST(BilinearPatchShapeT, IntersectHitPointOnSurface) {
    auto p = unit_blp();
    auto h = p.intersect(0.3,0.7,3.0, 0,0,-1, 1e-4,10.0);
    ASSERT_TRUE(h.has_value());
    // Hit point reconstructed from t should have z~0
    double hz = 3.0 + h->t * (-1.0);
    EXPECT_NEAR(hz, 0.0, 1e-6);
}

// -----------------------------------------------------------------------
// Sample
// -----------------------------------------------------------------------
TEST(BilinearPatchShapeT, SamplePdfIsInverseArea) {
    auto p = unit_blp();
    auto s = p.sample(0.3, 0.7);
    EXPECT_NEAR(s.pdf, p.pdf_area(), 1e-12);
}

TEST(BilinearPatchShapeT, SamplePointOnPatch) {
    auto p = unit_blp();
    auto s = p.sample(0.4, 0.6);
    // Unit square: px in [0,1], py in [0,1], pz = 0
    EXPECT_GE(s.px, -1e-9); EXPECT_LE(s.px, 1+1e-9);
    EXPECT_GE(s.py, -1e-9); EXPECT_LE(s.py, 1+1e-9);
    EXPECT_NEAR(s.pz, 0.0, 1e-9);
}

TEST(BilinearPatchShapeT, SampleNormalIsUnit) {
    auto p = unit_blp();
    auto s = p.sample(0.5, 0.5);
    double nlen = std::sqrt(s.nx*s.nx + s.ny*s.ny + s.nz*s.nz);
    EXPECT_NEAR(nlen, 1.0, 1e-9);
}

TEST(BilinearPatchShapeT, SampleIsDeterministic) {
    auto p = unit_blp();
    auto s1 = p.sample(0.123, 0.456);
    auto s2 = p.sample(0.123, 0.456);
    EXPECT_EQ(s1.px, s2.px);
    EXPECT_EQ(s1.py, s2.py);
    EXPECT_EQ(s1.pz, s2.pz);
}

// -----------------------------------------------------------------------
// sample_from / pdf_from consistency
// -----------------------------------------------------------------------
TEST(BilinearPatchShapeT, SampleFromPdfConsistent) {
    auto p = unit_blp();
    SamplingContext<double> ctx{0.5, 0.5, 3.0, 0,0,0};
    auto ss = p.sample_from(ctx, 0.4, 0.6);
    ASSERT_GT(ss.pdf, 0.0);
    double wix = ss.px-ctx.px, wiy = ss.py-ctx.py, wiz = ss.pz-ctx.pz;
    double pdf2 = p.pdf_from(ctx, wix, wiy, wiz);
    EXPECT_NEAR(ss.pdf, pdf2, 0.01*(ss.pdf+pdf2)*0.5 + 1e-10);
}

TEST(BilinearPatchShapeT, PdfFromZeroForMiss) {
    auto p = unit_blp();
    SamplingContext<double> ctx{0.5, 0.5, 3.0, 0,0,0};
    // Direction pointing away (+z, patch is at z=0 below ctx)
    double pdf = p.pdf_from(ctx, 0.0, 0.0, 1.0);
    EXPECT_EQ(pdf, 0.0);
}

TEST(BilinearPatchShapeT, SampleFromPdfPositive) {
    auto p = unit_blp();
    SamplingContext<double> ctx{0.5, 0.5, 2.0, 0,0,0};
    auto ss = p.sample_from(ctx, 0.5, 0.5);
    EXPECT_GT(ss.pdf, 0.0);
}
