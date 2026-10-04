/**
 * @file shadow_ray_tests.cpp
 * @brief Regression tests for shadow_ray.h's transmissive-material occlusion skip.
 *
 * Covers the bug fixed alongside this file: a NEE shadow ray used to treat
 * ANY hit (including glass) as full occlusion, so a dielectric sitting
 * between a shading point and a light silently zeroed a valid light sample.
 * GPU's optix_anyhit_shadow.h has always skipped transmissive materials via
 * optixIgnoreIntersection() -- these tests pin CPU's shadow_ray_hit() to the
 * same behavior.
 */

#include <gtest/gtest.h>
#include "rtweekend.h"
#include "shadow_ray.h"
#include "hittable_list.h"
#include "quad.h"
#include "sphere.h"
#include "material.h"
#include "optix_math_helpers.h"

// Glass is opaque to shadow rays, like every pbrt-v4 surface that has a material (VolPathIntegrator::
// SampleLd returns 0 for any hit with a material, integrators.cpp:1335). Walking straight through it
// counted a light seen through glass twice (here and on the specular BSDF path) and ignored refraction.
// shadow_ray_hit() must therefore stop at the glass sphere, which is not an emitter.
TEST(ShadowRayTest, DielectricOccludesLight) {
    hittable_list world;
    auto glass = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 0, -5), 1.0, glass));

    auto light_mat = make_shared<diffuse_light>(color(4, 4, 4));
    world.add(make_shared<quad>(point3(-1, -1, -10), vec3(2, 0, 0), vec3(0, 2, 0), light_mat));

    ray shadow_ray(point3(0, 0, 0), vec3(0, 0, -1));
    hit_record rec;
    ASSERT_TRUE(shadow_ray_hit(world, shadow_ray, rec));
    color Le = rec.mat->emitted(shadow_ray, rec, rec.u, rec.v, rec.p);
    EXPECT_EQ(Le.x(), 0.0) << "the shadow ray found the light through the glass";
}

// ...except a glass SHELL around a participating medium: the loader's fog-boundary idiom is a
// near-invisible dielectric (eta ~1.001) + MediumInterface, which must stay transparent to NEE.
TEST(ShadowRayTest, MediumBoundaryDielectricStaysTransparent) {
    hittable_list world;
    auto glass = make_shared<dielectric>(1.001);
    glass->mark_medium_boundary();
    world.add(make_shared<sphere>(point3(0, 0, -5), 1.0, glass));

    auto light_mat = make_shared<diffuse_light>(color(4, 4, 4));
    world.add(make_shared<quad>(point3(-1, -1, -10), vec3(2, 0, 0), vec3(0, 2, 0), light_mat));

    ray shadow_ray(point3(0, 0, 0), vec3(0, 0, -1));
    hit_record rec;
    ASSERT_TRUE(shadow_ray_hit(world, shadow_ray, rec));
    color Le = rec.mat->emitted(shadow_ray, rec, rec.u, rec.v, rec.p);
    EXPECT_GT(Le.x(), 0.0);
}

// An opaque (Lambertian) surface between the origin and a light DOES occlude:
// shadow_ray_hit() must stop at the first opaque hit, not walk through it.
TEST(ShadowRayTest, OpaqueSurfaceOccludesLight) {
    hittable_list world;
    auto opaque = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0, 0, -5), 1.0, opaque));

    auto light_mat = make_shared<diffuse_light>(color(4, 4, 4));
    world.add(make_shared<quad>(point3(-1, -1, -10), vec3(2, 0, 0), vec3(0, 2, 0), light_mat));

    ray shadow_ray(point3(0, 0, 0), vec3(0, 0, -1));
    hit_record rec;
    ASSERT_TRUE(shadow_ray_hit(world, shadow_ray, rec));
    color Le = rec.mat->emitted(shadow_ray, rec, rec.u, rec.v, rec.p);
    EXPECT_EQ(Le.x(), 0.0);
}

// t_max must still bound the search: a light beyond t_max is correctly
// reported as "nothing found" even though nothing opaque is in the way.
TEST(ShadowRayTest, RespectsTMax) {
    hittable_list world;
    auto light_mat = make_shared<diffuse_light>(color(4, 4, 4));
    world.add(make_shared<quad>(point3(-1, -1, -10), vec3(2, 0, 0), vec3(0, 2, 0), light_mat));

    ray shadow_ray(point3(0, 0, 0), vec3(0, 0, -1));
    hit_record rec;
    EXPECT_FALSE(shadow_ray_hit(world, shadow_ray, rec, /*t_max=*/5.0));
}

// Two glass spheres in a row must both be skipped -- exercises the walk-past
// loop taking more than a single step before it finds the opaque surface.
TEST(ShadowRayTest, MultipleDielectricsInARowAreSkipped) {
    hittable_list world;
    auto glass = make_shared<dielectric>(1.5);
    glass->mark_medium_boundary();   // plain glass blocks; shells around a medium are walked past
    world.add(make_shared<sphere>(point3(0, 0, -3), 0.5, glass));
    world.add(make_shared<sphere>(point3(0, 0, -5), 0.5, glass));

    auto light_mat = make_shared<diffuse_light>(color(4, 4, 4));
    world.add(make_shared<quad>(point3(-1, -1, -10), vec3(2, 0, 0), vec3(0, 2, 0), light_mat));

    ray shadow_ray(point3(0, 0, 0), vec3(0, 0, -1));
    hit_record rec;
    ASSERT_TRUE(shadow_ray_hit(world, shadow_ray, rec));
    color Le = rec.mat->emitted(shadow_ray, rec, rec.u, rec.v, rec.p);
    EXPECT_GT(Le.x(), 0.0);
}

// The wavefront backend shifts a shadow ray's origin along the surface normal as well as the ray, so it
// has to re-aim at the sampled point: keeping the direction crossed a grazing light's plane early, and
// with emitters acting as occluders the ray was blocked by the light it targeted.
TEST(ShadowRayTowardTest, ShiftedRayLandsOnTheTargetPoint) {
    const float3 hit    = make_float3(0.0f, 0.0f, 0.0f);
    const float3 normal = make_float3(0.0f, 1.0f, 0.0f);
    const float3 dir    = normalize(make_float3(1.0f, 0.2f, 0.0f));     // grazing the surface
    const float  dist   = 5.0f;
    const float3 shifted = hit + 0.01f * normal + 0.01f * dir;

    float3 out_dir; float out_tmax;
    shadow_ray_toward(hit, shifted, dir, dist, out_dir, out_tmax);

    const float3 end = shifted + out_dir * out_tmax;
    const float3 target = hit + dir * dist;
    EXPECT_NEAR(end.x, target.x, 1e-4f);
    EXPECT_NEAR(end.y, target.y, 1e-4f);
    EXPECT_NEAR(end.z, target.z, 1e-4f);
    EXPECT_NEAR(length(out_dir), 1.0f, 1e-5f);
}

TEST(ShadowRayTowardTest, UnboundedRayKeepsItsDirectionAndLength) {
    const float3 hit = make_float3(0.0f, 0.0f, 0.0f);
    const float3 dir = make_float3(0.0f, 1.0f, 0.0f);
    float3 out_dir; float out_tmax;
    shadow_ray_toward(hit, make_float3(0.0f, 0.01f, 0.01f), dir, 1e30f, out_dir, out_tmax);
    EXPECT_FLOAT_EQ(out_dir.y, 1.0f);
    EXPECT_GE(out_tmax, 1e29f);
}
