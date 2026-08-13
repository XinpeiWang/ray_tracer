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

// A glass sphere directly between the origin and a quad "light" must NOT
// occlude the shadow ray: shadow_ray_hit() should see through it and find
// the quad on the far side.
TEST(ShadowRayTest, DielectricDoesNotOccludeLight) {
    hittable_list world;
    auto glass = make_shared<dielectric>(1.5);
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
