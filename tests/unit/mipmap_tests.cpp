// mipmap_tests.cpp -- pbrt-v4-style MipMap unit tests
// pbrt-v4 reference: src/pbrt/util/mipmap.h / mipmap.cpp

#include "../../src/TheRestOfYourLife/rtweekend.h"
#include "../../src/shared/mipmap.h"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<color> solid_img(int w, int h, color c) {
    return std::vector<color>(w * h, c);
}

static std::vector<color> checker_img(int w, int h) {
    std::vector<color> p(w * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            p[y*w+x] = ((x + y) % 2 == 0) ? color(0,0,0) : color(1,1,1);
    return p;
}

// ---------------------------------------------------------------------------
// Pyramid construction (pbrt-v4 Image::GeneratePyramid alignment)
// ---------------------------------------------------------------------------

TEST(MipMap, LevelCountPowerOfTwo) {
    // 8x8 -> levels 8,4,2,1 = 4
    mipmap m(solid_img(8, 8, color(1,0,0)), 8, 8);
    EXPECT_EQ(m.levels(), 4);
}

TEST(MipMap, LevelCountNonPowerOfTwo) {
    mipmap m(solid_img(6, 6, color(1,1,1)), 6, 6);
    EXPECT_GE(m.levels(), 2);
    EXPECT_EQ(m.width(m.levels()-1),  1);
    EXPECT_EQ(m.height(m.levels()-1), 1);
}

TEST(MipMap, Level0IsFullResolution) {
    mipmap m(solid_img(32, 16, color(0.5,0.5,0.5)), 32, 16);
    EXPECT_EQ(m.width(0),  32);
    EXPECT_EQ(m.height(0), 16);
}

TEST(MipMap, EachLevelHalfPrevious) {
    mipmap m(solid_img(16, 16, color(1,1,1)), 16, 16);
    for (int lv = 1; lv < m.levels(); ++lv) {
        EXPECT_EQ(m.width(lv),  std::max(1, m.width(lv-1)  / 2)) << "level " << lv;
        EXPECT_EQ(m.height(lv), std::max(1, m.height(lv-1) / 2)) << "level " << lv;
    }
}

// ---------------------------------------------------------------------------
// Solid colour: all filter modes must return the original colour
// (pbrt-v4: Filter() dispatches to Point/Bilinear/Trilinear/EWA)
// ---------------------------------------------------------------------------

TEST(MipMap, SolidColor_Point) {
    MipMapOptions o; o.filter = MipFilter::Point;
    mipmap m(solid_img(8, 8, color(1,0,0)), 8, 8, o);
    color r = m.filter(0.5f, 0.5f, 0,0, 0,0);
    EXPECT_NEAR(r.x(), 1.0, 1e-5);
    EXPECT_NEAR(r.y(), 0.0, 1e-5);
    EXPECT_NEAR(r.z(), 0.0, 1e-5);
}

TEST(MipMap, SolidColor_Bilinear) {
    MipMapOptions o; o.filter = MipFilter::Bilinear;
    mipmap m(solid_img(16, 16, color(0,1,0)), 16, 16, o);
    color r = m.filter(0.5f, 0.5f, 0,0, 0,0);
    EXPECT_NEAR(r.y(), 1.0, 1e-4);
}

TEST(MipMap, SolidColor_Trilinear) {
    MipMapOptions o; o.filter = MipFilter::Trilinear;
    mipmap m(solid_img(32, 32, color(0,0,1)), 32, 32, o);
    color r = m.filter(0.5f, 0.5f, 0.5f, 0, 0, 0.5f);
    EXPECT_NEAR(r.z(), 1.0, 1e-4);
}

TEST(MipMap, SolidColor_EWA) {
    MipMapOptions o; o.filter = MipFilter::EWA;
    mipmap m(solid_img(16, 16, color(0.4,0.6,0.2)), 16, 16, o);
    color r = m.filter(0.5f, 0.5f, 0.01f, 0, 0, 0.01f);
    EXPECT_NEAR(r.x(), 0.4, 1e-3);
    EXPECT_NEAR(r.y(), 0.6, 1e-3);
    EXPECT_NEAR(r.z(), 0.2, 1e-3);
}

// ---------------------------------------------------------------------------
// Output bounded [0-eps, 1+eps]  (EWA Gaussian can slightly overshoot)
// ---------------------------------------------------------------------------

TEST(MipMap, OutputBounded) {
    MipMapOptions o; o.filter = MipFilter::EWA;
    mipmap m(checker_img(8, 8), 8, 8, o);
    const double eps = 1e-3;
    for (float u = 0.05f; u < 1.0f; u += 0.15f) {
        for (float v = 0.05f; v < 1.0f; v += 0.15f) {
            color r = m.filter(u, v, 0.02f, 0, 0, 0.02f);
            EXPECT_GE(r.x(), -eps) << u << "," << v;
            EXPECT_LE(r.x(), 1.0+eps) << u << "," << v;
            EXPECT_GE(r.y(), -eps);
            EXPECT_LE(r.y(), 1.0+eps);
            EXPECT_GE(r.z(), -eps);
            EXPECT_LE(r.z(), 1.0+eps);
        }
    }
}

// ---------------------------------------------------------------------------
// Bilinear: single-pixel image must return exact colour
// ---------------------------------------------------------------------------

TEST(MipMap, BilinearSinglePixel) {
    MipMapOptions o; o.filter = MipFilter::Bilinear;
    mipmap m(solid_img(1, 1, color(0.3,0.7,0.5)), 1, 1, o);
    color r = m.filter(0.5f, 0.5f, 0,0, 0,0);
    EXPECT_NEAR(r.x(), 0.3, 1e-5);
    EXPECT_NEAR(r.y(), 0.7, 1e-5);
    EXPECT_NEAR(r.z(), 0.5, 1e-5);
}

// ---------------------------------------------------------------------------
// Trilinear: coarsest LOD of a checkerboard approaches the 0.5 average
// ---------------------------------------------------------------------------

TEST(MipMap, TrilinearCoarseLODApproachesAverage) {
    MipMapOptions o; o.filter = MipFilter::Trilinear;
    mipmap m(checker_img(16, 16), 16, 16, o);
    color r = m.filter_lod(0.5f, 0.5f, (float)(m.levels() - 1));
    EXPECT_NEAR(r.x(), 0.5, 0.15);
}

// ---------------------------------------------------------------------------
// Wrap Clamp: OOB UV stays at border (pbrt-v4 WrapMode::Clamp)
// ---------------------------------------------------------------------------

TEST(MipMap, WrapClamp_OOBReturnsBorderColor) {
    int w = 4, h = 4;
    std::vector<color> px(w*h, color(0,0,1));
    for (int y = 0; y < h; ++y) px[y*w] = color(1,0,0); // left col = red
    MipMapOptions o; o.filter = MipFilter::Bilinear; o.wrap = MipWrapMode::Clamp;
    mipmap m(px, w, h, o);
    color r = m.filter(-0.1f, 0.5f, 0,0, 0,0);
    EXPECT_GT(r.x(), r.z()); // more red than blue
}

// ---------------------------------------------------------------------------
// filter_lod: explicit LOD helpers
// ---------------------------------------------------------------------------

TEST(MipMap, FilterLod_Level0ReturnsBilerp) {
    mipmap m(solid_img(8, 8, color(0.8,0.2,0.4)), 8, 8);
    color r = m.filter_lod(0.5f, 0.5f, 0.0f);
    EXPECT_NEAR(r.x(), 0.8, 1e-4);
}

TEST(MipMap, FilterLod_LargeValueClamped) {
    mipmap m(solid_img(8, 8, color(1,1,1)), 8, 8);
    color r = m.filter_lod(0.5f, 0.5f, 999.0f);
    EXPECT_GE(r.x(), -0.01); EXPECT_LE(r.x(), 1.01);
}

// ---------------------------------------------------------------------------
// EWA robustness (pbrt-v4: EWA with zero shorterVecLength falls back)
// ---------------------------------------------------------------------------

TEST(MipMap, EWA_ZeroDerivatives_NoCrash) {
    MipMapOptions o; o.filter = MipFilter::EWA;
    mipmap m(checker_img(8, 8), 8, 8, o);
    EXPECT_NO_THROW({
        color r = m.filter(0.5f, 0.5f, 0,0, 0,0);
        (void)r;
    });
}

TEST(MipMap, EWA_HighAnisotropy_NoCrash) {
    // pbrt-v4 clamps the anisotropy ratio to maxAnisotropy
    MipMapOptions o; o.filter = MipFilter::EWA; o.max_anisotropy = 8.0f;
    mipmap m(checker_img(16, 16), 16, 16, o);
    EXPECT_NO_THROW({
        color r = m.filter(0.5f, 0.5f, 0.5f, 0, 0, 0.001f);
        (void)r;
    });
}
