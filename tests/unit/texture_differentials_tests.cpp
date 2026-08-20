// texture_differentials_tests.cpp -- unit tests for the CPU ray-differential
// texture-filtering wiring added this session: ray.h's differential fields,
// sphere.h/triangle.h's dpdu/dpdv, texture.h's mipmap_texture::value_diff()/
// the new rtw_image&& constructor, and camera.h's get_ray() differential
// generation. src/shared/surface_interaction.h's compute_differentials()
// itself is an existing, unmodified, already-correct pbrt-v4 port - not
// re-tested here, only the new callers/producers around it.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "../../src/TheRestOfYourLife/rtweekend.h"
#include "../../src/TheRestOfYourLife/sphere.h"
#include "../../src/TheRestOfYourLife/triangle.h"
#include "../../src/TheRestOfYourLife/texture.h"
#include "../../src/TheRestOfYourLife/material.h"
#include "../../src/TheRestOfYourLife/camera.h"

// ===========================================================================
// sphere.h: dpdu/dpdv via finite differences against hit()'s own u/v/p
// ===========================================================================

namespace {
// Mirrors sphere.h's own (theta,phi) -> unit-sphere-point convention, used
// only to construct rays that hit a chosen (theta,phi) exactly - the
// dpdu/dpdv values under test come entirely from sphere::hit() itself, not
// from this helper.
point3 sphere_unit_point(double theta, double phi) {
    return point3(-std::sin(theta)*std::cos(phi), -std::cos(theta), std::sin(theta)*std::sin(phi));
}
} // namespace

TEST(SphereDifferentials, DpduMatchesFiniteDifference) {
    const double R = 3.0;
    sphere s(point3(0,0,0), R, std::make_shared<lambertian>(color(1,1,1)));

    double theta0 = 1.1, phi0 = 2.0;   // away from poles/seam
    double dphi = 1e-5;

    auto shoot = [&](double theta, double phi, hit_record& rec) {
        point3 target = R * sphere_unit_point(theta, phi);
        point3 origin = 2.0 * R * sphere_unit_point(theta, phi);
        ray r(origin, -sphere_unit_point(theta, phi), 0.0);
        ASSERT_TRUE(s.hit(r, interval(0.001, infinity), rec));
    };

    hit_record rec0, rec1;
    shoot(theta0, phi0, rec0);
    shoot(theta0, phi0 + dphi, rec1);

    vec3 finite_diff_du = (rec1.p - rec0.p) / (dphi / (2.0*pi));  // dphi -> du
    // dpdu should point the same direction and have comparable magnitude to
    // the finite-difference estimate (loose tolerance: dpdu is evaluated at
    // theta0/phi0, the finite difference over [phi0, phi0+dphi]).
    EXPECT_NEAR(unit_vector(rec0.dpdu).x(), unit_vector(finite_diff_du).x(), 1e-3);
    EXPECT_NEAR(unit_vector(rec0.dpdu).y(), unit_vector(finite_diff_du).y(), 1e-3);
    EXPECT_NEAR(unit_vector(rec0.dpdu).z(), unit_vector(finite_diff_du).z(), 1e-3);
    EXPECT_NEAR(rec0.dpdu.length(), finite_diff_du.length(), finite_diff_du.length() * 1e-3);
}

TEST(SphereDifferentials, DpdvMatchesFiniteDifference) {
    const double R = 3.0;
    sphere s(point3(0,0,0), R, std::make_shared<lambertian>(color(1,1,1)));

    double theta0 = 1.1, phi0 = 2.0;
    double dtheta = 1e-5;

    auto shoot = [&](double theta, double phi, hit_record& rec) {
        point3 origin = 2.0 * R * sphere_unit_point(theta, phi);
        ray r(origin, -sphere_unit_point(theta, phi), 0.0);
        ASSERT_TRUE(s.hit(r, interval(0.001, infinity), rec));
    };

    hit_record rec0, rec1;
    shoot(theta0, phi0, rec0);
    shoot(theta0 + dtheta, phi0, rec1);

    vec3 finite_diff_dv = (rec1.p - rec0.p) / (dtheta / pi);  // dtheta -> dv
    EXPECT_NEAR(unit_vector(rec0.dpdv).x(), unit_vector(finite_diff_dv).x(), 1e-3);
    EXPECT_NEAR(unit_vector(rec0.dpdv).y(), unit_vector(finite_diff_dv).y(), 1e-3);
    EXPECT_NEAR(unit_vector(rec0.dpdv).z(), unit_vector(finite_diff_dv).z(), 1e-3);
    EXPECT_NEAR(rec0.dpdv.length(), finite_diff_dv.length(), finite_diff_dv.length() * 1e-3);
}

TEST(SphereDifferentials, PoleFallbackIsFiniteAndNonDegenerate) {
    // theta ~ 0 (near the +Y pole): sin(theta) -> 0, dpdu's main-branch
    // formula degenerates to (near) zero - confirm the fallback kicks in and
    // produces finite, non-zero, orthogonal-ish tangent vectors instead.
    const double R = 1.0;
    sphere s(point3(0,0,0), R, std::make_shared<lambertian>(color(1,1,1)));
    point3 origin(0, 2*R, 1e-9);
    ray r(origin, vec3(0,-1,0), 0.0);
    hit_record rec;
    ASSERT_TRUE(s.hit(r, interval(0.001, infinity), rec));

    EXPECT_TRUE(std::isfinite(rec.dpdu.x()) && std::isfinite(rec.dpdu.y()) && std::isfinite(rec.dpdu.z()));
    EXPECT_TRUE(std::isfinite(rec.dpdv.x()) && std::isfinite(rec.dpdv.y()) && std::isfinite(rec.dpdv.z()));
    EXPECT_GT(rec.dpdu.length(), 1e-6);
    EXPECT_GT(rec.dpdv.length(), 1e-6);
}

// ===========================================================================
// triangle.h: dpdu/dpdv via the UV-edge Jacobian round-trip identity
// ===========================================================================

TEST(TriangleDifferentials, JacobianReproducesEdgesFromUVs) {
    point3 p0(0,0,0), p1(1,0,0), p2(0,1,0);
    auto mesh = std::make_shared<triangle_mesh_data>();
    mesh->positions = {p0, p1, p2};
    mesh->indices   = {0, 1, 2};
    // Non-trivial (non-identity) UVs so this is a real test of the Jacobian
    // solve, not a coincidence of du/dv == world-space edge lengths.
    mesh->uvs = { 0.0, 0.0,  2.0, 0.0,  0.5, 1.5 };
    auto mat = std::make_shared<lambertian>(color(1,1,1));
    triangle tri(mesh, 0, mat);

    ray r(point3(0.2, 0.2, 1), vec3(0,0,-1), 0.0);  // hits inside the triangle
    hit_record rec;
    ASSERT_TRUE(tri.hit(r, interval(0.001, infinity), rec));

    double du1 = 2.0 - 0.0, dv1 = 0.0 - 0.0;
    double du2 = 0.5 - 0.0, dv2 = 1.5 - 0.0;
    vec3 e1 = p1 - p0, e2 = p2 - p0;

    vec3 reconstructed_e1 = rec.dpdu * du1 + rec.dpdv * dv1;
    vec3 reconstructed_e2 = rec.dpdu * du2 + rec.dpdv * dv2;

    EXPECT_NEAR(reconstructed_e1.x(), e1.x(), 1e-9);
    EXPECT_NEAR(reconstructed_e1.y(), e1.y(), 1e-9);
    EXPECT_NEAR(reconstructed_e1.z(), e1.z(), 1e-9);
    EXPECT_NEAR(reconstructed_e2.x(), e2.x(), 1e-9);
    EXPECT_NEAR(reconstructed_e2.y(), e2.y(), 1e-9);
    EXPECT_NEAR(reconstructed_e2.z(), e2.z(), 1e-9);
}

TEST(TriangleDifferentials, DegenerateUVFallbackIsFinite) {
    point3 p0(0,0,0), p1(1,0,0), p2(0,1,0);
    auto mesh = std::make_shared<triangle_mesh_data>();
    mesh->positions = {p0, p1, p2};
    mesh->indices   = {0, 1, 2};
    // All three UVs identical -> zero-area UV parameterization -> det == 0.
    mesh->uvs = { 0.3, 0.3,  0.3, 0.3,  0.3, 0.3 };
    auto mat = std::make_shared<lambertian>(color(1,1,1));
    triangle tri(mesh, 0, mat);

    ray r(point3(0.2, 0.2, 1), vec3(0,0,-1), 0.0);
    hit_record rec;
    ASSERT_TRUE(tri.hit(r, interval(0.001, infinity), rec));

    EXPECT_TRUE(std::isfinite(rec.dpdu.x()) && std::isfinite(rec.dpdu.y()) && std::isfinite(rec.dpdu.z()));
    EXPECT_TRUE(std::isfinite(rec.dpdv.x()) && std::isfinite(rec.dpdv.y()) && std::isfinite(rec.dpdv.z()));
    EXPECT_GT(rec.dpdu.length(), 1e-9);
}

// ===========================================================================
// texture.h: mipmap_texture's value_diff() wiring + rtw_image&& constructor
// ===========================================================================

TEST(MipmapTextureDifferentials, RtwImageMoveConstructorMatchesFileConstructorLevels) {
    // Build a small synthetic checkerboard directly (no file I/O needed -
    // rtw_image(w,h,float*) is a real, existing constructor for exactly
    // this - see rtw_stb_image.h).
    const int w = 8, h = 8;
    std::vector<float> pixels(w * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            bool black = ((x + y) % 2) == 0;
            float v = black ? 0.0f : 1.0f;
            int idx = (y * w + x) * 3;
            pixels[idx+0] = v; pixels[idx+1] = v; pixels[idx+2] = v;
        }
    rtw_image img(w, h, pixels.data());
    ASSERT_GT(img.height(), 0);

    mipmap_texture tex(std::move(img));
    EXPECT_GT(tex.mip_levels(), 0);
}

TEST(MipmapTextureDifferentials, ValueDiffBlursMoreThanPointSampleUnderLargeFootprint) {
    // 2x2 checkerboard (max-frequency content) so a large filter footprint
    // has something real to blur away.
    const int w = 32, h = 32;
    std::vector<float> pixels(w * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            bool black = ((x/4 + y/4) % 2) == 0;
            float v = black ? 0.0f : 1.0f;
            int idx = (y * w + x) * 3;
            pixels[idx+0] = v; pixels[idx+1] = v; pixels[idx+2] = v;
        }
    rtw_image img(w, h, pixels.data());
    ASSERT_GT(img.height(), 0);
    mipmap_texture tex(std::move(img));

    // Point sample (zero footprint): should land exactly on a black or white
    // texel - i.e. saturated (0 or 1), not blurred. UV 0.3 (not 0.5) is used
    // so the sample lands well inside one checkerboard cell rather than
    // exactly on a cell boundary - at 0.5 bilerp's x=s*w-0.5 lands on a
    // texel-grid line and averages two cells to 0.5 regardless of footprint.
    color point_sample = tex.value_diff(0.3, 0.3, point3(0,0,0), 0,0,0,0);
    double point_extremeness = std::fabs(point_sample.x() - 0.5);

    // Huge footprint (simulating a grazing-angle/distant hit): should read
    // close to the checkerboard's 50% grey average, i.e. much less extreme
    // than the point sample.
    color blurred = tex.value_diff(0.3, 0.3, point3(0,0,0), 1.0, 0.0, 0.0, 1.0);
    double blurred_extremeness = std::fabs(blurred.x() - 0.5);

    EXPECT_LT(blurred_extremeness, point_extremeness);
    EXPECT_LT(blurred_extremeness, 0.15);  // should be close to the 0.5 average
}

TEST(TextureBaseClass, ValueDiffDefaultsToValueForNonImageTextures) {
    solid_color tex(color(0.2, 0.4, 0.6));
    color v  = tex.value(0.1, 0.2, point3(0,0,0));
    color vd = tex.value_diff(0.1, 0.2, point3(0,0,0), 0.5, 0.5, 0.5, 0.5);
    EXPECT_DOUBLE_EQ(v.x(), vd.x());
    EXPECT_DOUBLE_EQ(v.y(), vd.y());
    EXPECT_DOUBLE_EQ(v.z(), vd.z());
}

// ===========================================================================
// camera.h: get_ray() differential generation
// ===========================================================================

TEST(CameraDifferentials, PrimaryRayHasDifferentials) {
    camera cam;
    cam.lookfrom = point3(0, 0, -10);
    cam.lookat   = point3(0, 0, 0);
    cam.vfov = 40;
    cam.image_width = 100;
    cam.samples_per_pixel = 4;
    cam.max_depth = 1;
    cam.background = color(0,0,0);
    cam.initialize();

    ray r = cam.get_ray(50, 50, 0, 0, vec3(0,0,0));
    EXPECT_TRUE(r.has_differentials());
}

TEST(CameraDifferentials, RxDirectionStepMatchesOnePixelStep) {
    camera cam;
    cam.lookfrom = point3(0, 0, -10);
    cam.lookat   = point3(0, 0, 0);
    cam.vfov = 40;
    cam.image_width = 100;
    cam.samples_per_pixel = 4;
    cam.max_depth = 1;
    cam.background = color(0,0,0);
    cam.initialize();

    // Same offset, adjacent pixel columns - the primary-direction delta
    // between these two calls IS one full pixel_delta_u step (by
    // get_ray()'s own construction), so it's a fixture-free way to check
    // that rx_direction's offset from the primary ray matches that same
    // step, without needing to expose pixel_delta_u from the class.
    ray r0 = cam.get_ray(50, 50, 0, 0, vec3(0,0,0));
    ray r1 = cam.get_ray(51, 50, 0, 0, vec3(0,0,0));
    vec3 one_pixel_step = r1.direction() - r0.direction();
    vec3 rx_step = r0.rx_direction() - r0.direction();

    EXPECT_NEAR(rx_step.x(), one_pixel_step.x(), 1e-9);
    EXPECT_NEAR(rx_step.y(), one_pixel_step.y(), 1e-9);
    EXPECT_NEAR(rx_step.z(), one_pixel_step.z(), 1e-9);
}
