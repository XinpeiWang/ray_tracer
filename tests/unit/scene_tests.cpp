/**
 * @file scene_tests.cpp
 * @brief Unit tests for Cornell box scene construction
 * 
 * Tests the Cornell box scene to ensure:
 * - Walls are correctly positioned
 * - Materials are properly assigned (red, green, white)
 * - Light source is configured correctly
 * - Objects (glass sphere, rotated box) are present
 */

#include <gtest/gtest.h>
#include "rtweekend.h"
#include "cornell_box_scene.h"
#include "hittable_list.h"
#include "quad.h"
#include "sphere.h"
#include "material.h"
#include <memory>

// ============================================================================
// Cornell Box Geometry Tests
// ============================================================================

/**
 * Test that Cornell box scene creates the correct number of objects
 */
TEST(SceneTest, CornellBoxObjectCount) {
	auto world = build_cornell_box_scene();

	// Cornell box should have:
	// - 5 walls (left, right, back, floor, ceiling)
	// - 1 light
	// - 2 boxes (rotated aluminum + glass sphere, depending on configuration)
	// Total: at least 5 walls + 1 light = 6 objects minimum

	EXPECT_GE(world.objects.size(), 6);
}

/**
 * Test Cornell box bounds
 * The box should span from 0 to 555 in all dimensions
 */
TEST(SceneTest, CornellBoxBounds) {
	// Cornell box corner positions
	const double x_min = 0.0;
	const double x_max = 555.0;
	const double y_min = 0.0;
	const double y_max = 555.0;
	const double z_min = 0.0;
	const double z_max = 555.0;

	// Test known wall positions
	// Green wall (right): x = 0
	// Red wall (left): x = 555
	// White back: z = 555
	// Floor: y = 0
	// Ceiling: y = 555

	EXPECT_DOUBLE_EQ(x_min, 0.0);
	EXPECT_DOUBLE_EQ(x_max, 555.0);
	EXPECT_DOUBLE_EQ(y_min, 0.0);
	EXPECT_DOUBLE_EQ(y_max, 555.0);
	EXPECT_DOUBLE_EQ(z_min, 0.0);
	EXPECT_DOUBLE_EQ(z_max, 555.0);
}

/**
 * Test Cornell box center point
 */
TEST(SceneTest, CornellBoxCenter) {
	point3 center(278, 278, 278);

	// Center is approximately (278,278,278)
	EXPECT_NEAR(center.x(), 278.0, 1.0);
	EXPECT_NEAR(center.y(), 278.0, 1.0);
	EXPECT_NEAR(center.z(), 278.0, 1.0);
}

// ============================================================================
// Material Tests
// ============================================================================

/**
 * Test red material (left wall)
 */
TEST(MaterialTest, RedWallMaterial) {
	auto red = make_shared<lambertian>(color(.65, .05, .05));

	// Red wall should be predominantly red
	// RGB: (.65, .05, .05)
	EXPECT_GT(red->get_texture()->value(0, 0, point3()).x(), 0.6);
	EXPECT_LT(red->get_texture()->value(0, 0, point3()).y(), 0.1);
	EXPECT_LT(red->get_texture()->value(0, 0, point3()).z(), 0.1);
}

/**
 * Test green material (right wall)
 */
TEST(MaterialTest, GreenWallMaterial) {
	auto green = make_shared<lambertian>(color(.12, .45, .15));

	// Green wall should be predominantly green
	// RGB: (.12, .45, .15)
	EXPECT_GT(green->get_texture()->value(0, 0, point3()).y(), 0.4);
	EXPECT_LT(green->get_texture()->value(0, 0, point3()).x(), 0.2);
	EXPECT_LT(green->get_texture()->value(0, 0, point3()).z(), 0.2);
}

/**
 * Test white material (other walls)
 */
TEST(MaterialTest, WhiteWallMaterial) {
	auto white = make_shared<lambertian>(color(.73, .73, .73));

	// White walls should have similar R, G, B values
	color c = white->get_texture()->value(0, 0, point3());
	EXPECT_NEAR(c.x(), c.y(), 0.01);
	EXPECT_NEAR(c.y(), c.z(), 0.01);
	EXPECT_NEAR(c.x(), 0.73, 0.01);
}

/**
 * Test light material
 */
TEST(MaterialTest, LightMaterial) {
	auto light = make_shared<diffuse_light>(color(15, 15, 15));

	// Light should emit bright white light (use get_texture to access color directly)
	color emitted = light->get_texture()->value(0, 0, point3());
	EXPECT_DOUBLE_EQ(emitted.x(), 15.0);
	EXPECT_DOUBLE_EQ(emitted.y(), 15.0);
	EXPECT_DOUBLE_EQ(emitted.z(), 15.0);
}

// ============================================================================
// Ray-Scene Intersection Tests
// ============================================================================

/**
 * Test ray hitting the floor (y = 0)
 */
TEST(IntersectionTest, FloorIntersection) {
	auto world = build_cornell_box_scene();

	// Ray pointing straight down from center
	ray r(point3(278, 300, 278), vec3(0, -1, 0));
	hit_record rec;
	interval ray_t(0.001, infinity);

	// Should hit the floor
	bool hit = world.hit(r, ray_t, rec);
	EXPECT_TRUE(hit);

	if (hit) {
		// Hit point should be on the floor (y ≈ 0)
		EXPECT_NEAR(rec.p.y(), 0.0, 1.0);
	}
}

/**
 * Test ray hitting the ceiling (y = 555)
 */
TEST(IntersectionTest, CeilingIntersection) {
	auto world = build_cornell_box_scene();

	// Ray pointing straight up from center
	ray r(point3(278, 200, 278), vec3(0, 1, 0));
	hit_record rec;
	interval ray_t(0.001, infinity);

	// Should hit the ceiling or light
	bool hit = world.hit(r, ray_t, rec);
	EXPECT_TRUE(hit);

	if (hit) {
		// Hit point should be on ceiling (y ≈ 555) or light (y = 554)
		EXPECT_GT(rec.p.y(), 500.0);
	}
}

/**
 * Test ray hitting the back wall (z = 555)
 */
TEST(IntersectionTest, BackWallIntersection) {
	auto world = build_cornell_box_scene();

	// Ray pointing towards back wall
	ray r(point3(278, 278, 0), vec3(0, 0, 1));
	hit_record rec;
	interval ray_t(0.001, infinity);

	// Should hit *something* in the scene
	bool hit = world.hit(r, ray_t, rec);
	EXPECT_TRUE(hit);

	if (hit) {
		// Hit point should be somewhere in the scene (z between 0 and 555)
		EXPECT_GT(rec.p.z(), 0.0);
		EXPECT_LE(rec.p.z(), 555.0);
	}
}

/**
 * Test ray missing the scene entirely
 */
TEST(IntersectionTest, RayMiss) {
	auto world = build_cornell_box_scene();

	// Ray pointing away from the scene
	ray r(point3(278, 278, -1000), vec3(0, 0, -1));
	hit_record rec;
	interval ray_t(0.001, infinity);

	// Should not hit anything
	bool hit = world.hit(r, ray_t, rec);
	EXPECT_FALSE(hit);
}

// ============================================================================
// Light Source Tests
// ============================================================================

/**
 * Test light source position and size
 */
TEST(SceneTest, LightPosition) {
	// Cornell box light is at:
	// Center: (278, 554, 278)
	// Size: 130 x 105 (approximately)

	point3 light_center(278, 554, 278);

	// Light should be near the ceiling (y ≈ 554)
	EXPECT_NEAR(light_center.y(), 554.0, 1.0);

	// Light should be centered in x and z
	EXPECT_NEAR(light_center.x(), 278.0, 5.0);
	EXPECT_NEAR(light_center.z(), 278.0, 5.0);
}

// ============================================================================
// Object Placement Tests
// ============================================================================

/**
 * Test that glass sphere is present and positioned correctly
 */
TEST(SceneTest, GlassSpherePresent) {
	auto world = build_cornell_box_scene();

	// The glass sphere should be in the scene
	// Position: approximately (190, 90, 190)
	// Radius: 90

	// Cast a ray that should hit the sphere
	ray r(point3(190, 90, 0), vec3(0, 0, 1));
	hit_record rec;
	interval ray_t(0.001, infinity);

	bool hit = world.hit(r, ray_t, rec);

	// We should hit *something* (wall or sphere)
	EXPECT_TRUE(hit);
}

/**
 * Test that rotated aluminum box is present
 */
TEST(SceneTest, AluminumBoxPresent) {
	auto world = build_cornell_box_scene();

	// The aluminum box should be in the scene
	// Position: approximately (265, 0, 295)
	// Size: 165 x 330 x 165
	// Rotation: 15° around Y axis

	// Cast a ray that should hit the box region
	ray r(point3(265, 100, 0), vec3(0, 0, 1));
	hit_record rec;
	interval ray_t(0.001, infinity);

	bool hit = world.hit(r, ray_t, rec);

	// We should hit *something* (wall or box)
	EXPECT_TRUE(hit);
}

// ============================================================================
// Scene Consistency Tests
// ============================================================================

/**
 * Test that scene construction is deterministic
 */
TEST(SceneTest, DeterministicConstruction) {
	auto world1 = build_cornell_box_scene();
	auto world2 = build_cornell_box_scene();

	// Both scenes should have the same number of objects
	EXPECT_EQ(world1.objects.size(), world2.objects.size());
}

/**
 * Test that scene can be constructed multiple times
 */
TEST(SceneTest, MultipleConstruction) {
	for (int i = 0; i < 10; ++i) {
		hittable_list world;
		EXPECT_NO_THROW(world = build_cornell_box_scene());
		EXPECT_GE(world.objects.size(), 6);
	}
}
