// bilinear_patch_hittable_tests.cpp
// Unit tests for bilinear_patch_hittable's NEE hooks (src/TheRestOfYourLife/scenes_advanced.h)
//
// bilinear_patch_hittable is the CPU hittable a pbrt "bilinearmesh" shape
// flattens into (see pbrt_cpu_builder.h). Before this task it only overrode
// hit()/bounding_box() - a bilinear patch could be hit directly but never
// explicitly sampled for next-event estimation. These pin the two new
// overrides, pdf_value()/random(), against each other: for a light sampled
// via random(), the solid-angle density pdf_value() reports for that exact
// direction must match what uniform-area sampling actually implies, or NEE
// on a bilinear-patch light (e.g. sportscar-area-lights.pbrt's 5 studio
// panels) would be biased - too bright or too dark, not just noisy.

#include <gtest/gtest.h>

#include "rtweekend.h"
#include "scenes_advanced.h"
#include "material.h"

#include <cmath>

namespace {

// Axis-aligned unit square in the XY plane (z=0), area = 1, normal = +-Z.
// A rectangle keeps the "expected" side of the consistency check closed-form:
// uniform-area sampling on a rectangle has constant area-pdf = 1/area.
bilinear_patch_hittable make_unit_square_patch() {
    auto mat = make_shared<lambertian>(color(1, 1, 1));
    return bilinear_patch_hittable(
        point3(0, 0, 0), point3(1, 0, 0), point3(0, 1, 0), point3(1, 1, 0), mat);
}

} // namespace

TEST(BilinearPatchHittable, RandomProducesPointsOnThePatch) {
    bilinear_patch_hittable patch = make_unit_square_patch();
    point3 origin(0.5, 0.5, 2.0);
    for (int i = 0; i < 200; ++i) {
        vec3 wi = patch.random(origin);
        point3 p = origin + wi;
        EXPECT_NEAR(p.z(), 0.0, 1e-6) << "sampled point must lie in the patch's plane";
        EXPECT_GE(p.x(), -1e-6); EXPECT_LE(p.x(), 1.0 + 1e-6);
        EXPECT_GE(p.y(), -1e-6); EXPECT_LE(p.y(), 1.0 + 1e-6);
    }
}

TEST(BilinearPatchHittable, PdfValueIsZeroForADirectionThatMissesThePatch) {
    bilinear_patch_hittable patch = make_unit_square_patch();
    point3 origin(0.5, 0.5, 2.0);
    // Straight up: away from the patch entirely.
    EXPECT_EQ(patch.pdf_value(origin, vec3(0, 0, 1)), 0.0);
    // Off to the side, well clear of the unit square's footprint.
    EXPECT_EQ(patch.pdf_value(origin, unit_vector(vec3(10, 10, -2))), 0.0);
}

TEST(BilinearPatchHittable, PdfValueIsPositiveForADirectionThatHitsThePatch) {
    bilinear_patch_hittable patch = make_unit_square_patch();
    point3 origin(0.5, 0.5, 2.0);
    EXPECT_GT(patch.pdf_value(origin, vec3(0, 0, -1)), 0.0);
}

// The core unbiasedness pin: random() samples uniformly over the patch's
// AREA, and for this rectangle that means a constant area-pdf of 1/area
// everywhere on it. pdf_value() independently recomputes a solid-angle pdf
// via the area-to-solid-angle Jacobian (area_pdf * dist^2 / cos_theta - see
// blp_pdf_wi in src/shared/bilinear_patch.h). If the two disagreed, NEE
// would weight a bilinear-patch light sample by the wrong pdf and the
// estimator would be biased rather than merely noisy.
TEST(BilinearPatchHittable, RandomAndPdfValueAgreeWithTheKnownUniformAreaDensity) {
    bilinear_patch_hittable patch = make_unit_square_patch();
    const double area = 1.0;
    point3 origin(0.3, 0.7, 2.5);
    for (int i = 0; i < 500; ++i) {
        vec3 wi = patch.random(origin);
        const double dist2 = wi.length_squared();
        const double cos_theta = std::fabs(unit_vector(wi).z());
        ASSERT_GT(cos_theta, 1e-6);
        const double expected_pdf = (1.0 / area) * dist2 / cos_theta;
        const double actual_pdf = patch.pdf_value(origin, wi);
        EXPECT_NEAR(actual_pdf, expected_pdf, 0.02 * expected_pdf + 1e-6);
    }
}

TEST(BilinearPatchHittable, PdfValueAcceptsAnUnnormalizedDirectionLikeRandomReturns) {
    // random()'s contract returns an unnormalized origin->point vector (see
    // its own comment, matching quad::random()) - pdf_value must be able to
    // consume that directly, since that is exactly how NEE calls it.
    bilinear_patch_hittable patch = make_unit_square_patch();
    point3 origin(0.5, 0.5, 2.0);
    vec3 wi(0, 0, -2.0); // unnormalized: hits patch center, |wi| = 2
    const double pdf_raw = patch.pdf_value(origin, wi);
    const double pdf_unit = patch.pdf_value(origin, unit_vector(wi));
    ASSERT_GT(pdf_raw, 0.0);
    EXPECT_NEAR(pdf_raw, pdf_unit, 1e-6);
}
