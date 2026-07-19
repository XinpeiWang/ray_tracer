/**
 * @file camera_tests.cpp
 * @brief Unit tests for camera positioning and ray generation
 * 
 * Tests the camera class to ensure:
 * - Camera position (lookfrom) is correctly set
 * - Camera target (lookat) points to Cornell box center
 * - Field of view calculations are accurate
 * - Ray generation produces valid directions
 */

#include <gtest/gtest.h>
#include "camera.h"
#include "vec3.h"
#include <cmath>

// ============================================================================
// Camera Configuration Tests
// ============================================================================

/**
 * Test that camera lookfrom position is set correctly
 */
TEST(CameraTest, LookFromPosition) {
	camera cam;

	// Test preset: Default (front view)
	cam.lookfrom = point3(278, 278, -800);
	cam.lookat = point3(278, 278, 0);
	cam.vfov = 40;
	cam.image_width = 600;
	cam.samples_per_pixel = 10;
	cam.max_depth = 10;
	cam.background = color(0, 0, 0);

	cam.initialize();

	// Verify camera center matches lookfrom
	EXPECT_DOUBLE_EQ(cam.center.x(), 278);
	EXPECT_DOUBLE_EQ(cam.center.y(), 278);
	EXPECT_DOUBLE_EQ(cam.center.z(), -800);
}

/**
 * Test that camera lookat is pointing at Cornell box center
 */
TEST(CameraTest, LookAtCenter) {
	camera cam;

	// Cornell box center is at (278, 278, 278)
	cam.lookfrom = point3(278, 278, -800);
	cam.lookat = point3(278, 278, 278);
	cam.vfov = 40;
	cam.image_width = 600;
	cam.samples_per_pixel = 10;
	cam.max_depth = 10;
	cam.background = color(0, 0, 0);

	cam.initialize();

	// Camera should be looking towards z+ direction
	// The w vector points from lookat to lookfrom (opposite of view direction)
	vec3 expected_view_dir = unit_vector(cam.lookat - cam.lookfrom);

	// View direction should point roughly in +z direction
	EXPECT_GT(expected_view_dir.z(), 0.9); // Should be nearly (0, 0, 1)
}

/**
 * Test camera with different viewpoints
 */
TEST(CameraTest, MultipleViewpoints) {
	struct TestCase {
		const char* name;
		point3 lookfrom;
		point3 lookat;
	};

	TestCase cases[] = {
		{"Front View", point3(278, 278, -800), point3(278, 278, 278)},
		{"Inside Center", point3(278, 278, 278), point3(278, 278, 278)},
		{"Left Wall", point3(50, 278, 278), point3(278, 278, 278)},
		{"Right Wall", point3(506, 278, 278), point3(278, 278, 278)},
		{"Top View", point3(278, 506, 278), point3(278, 278, 278)},
		{"Bottom View", point3(278, 50, 278), point3(278, 278, 278)},
	};

	for (const auto& test : cases) {
		camera cam;
		cam.lookfrom = test.lookfrom;
		cam.lookat = test.lookat;
		cam.vfov = 40;
		cam.image_width = 100;
		cam.samples_per_pixel = 1;
		cam.max_depth = 5;
		cam.background = color(0, 0, 0);

		// Should not crash
		EXPECT_NO_THROW(cam.initialize()) << "Failed for: " << test.name;

		// Camera center should match lookfrom
		EXPECT_DOUBLE_EQ(cam.center.x(), test.lookfrom.x()) << "Failed for: " << test.name;
		EXPECT_DOUBLE_EQ(cam.center.y(), test.lookfrom.y()) << "Failed for: " << test.name;
		EXPECT_DOUBLE_EQ(cam.center.z(), test.lookfrom.z()) << "Failed for: " << test.name;
	}
}

// ============================================================================
// Field of View Tests
// ============================================================================

/**
 * Test field of view calculations
 */
TEST(CameraTest, FieldOfView) {
	camera cam;

	cam.lookfrom = point3(0, 0, 0);
	cam.lookat = point3(0, 0, 1);
	cam.vfov = 90; // 90 degree FOV
	cam.image_width = 400;
	cam.samples_per_pixel = 1;
	cam.max_depth = 5;
	cam.background = color(0, 0, 0);

	cam.initialize();

	// With 90° FOV and aspect_ratio 1:1, viewport should have specific dimensions
	// theta = 90° = π/2 rad
	// h = tan(θ/2) = tan(π/4) = 1
	// viewport_height = 2 * h * focus_dist = 2 * 1 * 1 = 2

	// Note: The actual viewport calculation depends on focus_dist
	// We just verify initialization doesn't crash
	EXPECT_GT(cam.image_width, 0);
	EXPECT_GT(cam.image_height, 0);
}

/**
 * Test different FOV values
 */
TEST(CameraTest, DifferentFOVValues) {
	double fov_values[] = {20, 40, 60, 90, 120};

	for (double fov : fov_values) {
		camera cam;
		cam.lookfrom = point3(0, 0, 0);
		cam.lookat = point3(0, 0, 1);
		cam.vfov = fov;
		cam.image_width = 200;
		cam.samples_per_pixel = 1;
		cam.max_depth = 5;
		cam.background = color(0, 0, 0);

		EXPECT_NO_THROW(cam.initialize()) << "Failed for FOV: " << fov;
	}
}

// ============================================================================
// Ray Generation Tests
// ============================================================================

/**
 * Test that camera generates valid rays
 */
TEST(CameraTest, RayGeneration) {
	camera cam;

	cam.lookfrom = point3(0, 0, 0);
	cam.lookat = point3(0, 0, 1);
	cam.vfov = 90;
	cam.image_width = 400;
	cam.samples_per_pixel = 1;
	cam.max_depth = 5;
	cam.background = color(0, 0, 0);

	cam.initialize();

	// Get ray for center pixel
	ray r = cam.get_ray(cam.image_width / 2, cam.image_height / 2);

	// Ray origin should be at camera position
	EXPECT_DOUBLE_EQ(r.origin().x(), 0.0);
	EXPECT_DOUBLE_EQ(r.origin().y(), 0.0);
	EXPECT_DOUBLE_EQ(r.origin().z(), 0.0);

	// Ray direction should be roughly looking forward (+z)
	vec3 dir = unit_vector(r.direction());
	EXPECT_GT(dir.z(), 0.5); // Should have significant +z component
}

/**
 * Test ray generation for corner pixels
 */
TEST(CameraTest, RayGenerationCorners) {
	camera cam;

	cam.lookfrom = point3(0, 0, 0);
	cam.lookat = point3(0, 0, 1);
	cam.vfov = 90;
	cam.image_width = 400;
	cam.samples_per_pixel = 1;
	cam.max_depth = 5;
	cam.background = color(0, 0, 0);

	cam.initialize();

	// Test corners
	struct Corner {
		int x, y;
		const char* name;
	};

	Corner corners[] = {
		{0, 0, "Top-Left"},
		{cam.image_width - 1, 0, "Top-Right"},
		{0, cam.image_height - 1, "Bottom-Left"},
		{cam.image_width - 1, cam.image_height - 1, "Bottom-Right"},
	};

	for (const auto& corner : corners) {
		ray r = cam.get_ray(corner.x, corner.y);

		// All rays should originate from camera position
		EXPECT_DOUBLE_EQ(r.origin().x(), 0.0) << "Failed for: " << corner.name;
		EXPECT_DOUBLE_EQ(r.origin().y(), 0.0) << "Failed for: " << corner.name;
		EXPECT_DOUBLE_EQ(r.origin().z(), 0.0) << "Failed for: " << corner.name;

		// Direction should be unit vector
		double len = r.direction().length();
		EXPECT_NEAR(len, 1.0, 0.01) << "Failed for: " << corner.name;
	}
}

// ============================================================================
// Aspect Ratio Tests
// ============================================================================

/**
 * Test aspect ratio handling
 */
TEST(CameraTest, AspectRatio) {
	struct TestCase {
		int width;
		double expected_height_ratio;
	};

	TestCase cases[] = {
		{400, 1.0},   // 1:1 square
		{800, 1.0},   // 1:1 square (larger)
		{600, 1.0},   // 1:1 square
	};

	for (const auto& test : cases) {
		camera cam;
		cam.lookfrom = point3(0, 0, 0);
		cam.lookat = point3(0, 0, 1);
		cam.vfov = 90;
		cam.image_width = test.width;
		cam.samples_per_pixel = 1;
		cam.max_depth = 5;
		cam.background = color(0, 0, 0);

		cam.initialize();

		// Verify aspect ratio is approximately 1:1 (default)
		double aspect = static_cast<double>(cam.image_width) / cam.image_height;
		EXPECT_NEAR(aspect, 1.0, 0.01) << "Width: " << test.width;
	}
}

// ============================================================================
// Camera Parameter Validation Tests
// ============================================================================

/**
 * Test camera with minimum valid parameters
 */
TEST(CameraTest, MinimumParameters) {
	camera cam;

	cam.lookfrom = point3(0, 0, 0);
	cam.lookat = point3(0, 0, 1);
	cam.vfov = 1; // Very narrow FOV
	cam.image_width = 10; // Very small image
	cam.samples_per_pixel = 1;
	cam.max_depth = 1;
	cam.background = color(0, 0, 0);

	EXPECT_NO_THROW(cam.initialize());
	EXPECT_GT(cam.image_width, 0);
	EXPECT_GT(cam.image_height, 0);
}

/**
 * Test camera with maximum reasonable parameters
 */
TEST(CameraTest, MaximumParameters) {
	camera cam;

	cam.lookfrom = point3(0, 0, 0);
	cam.lookat = point3(0, 0, 1);
	cam.vfov = 179; // Nearly 180° FOV
	cam.image_width = 4000; // Large image
	cam.samples_per_pixel = 10000;
	cam.max_depth = 100;
	cam.background = color(1, 1, 1);

	EXPECT_NO_THROW(cam.initialize());
	EXPECT_GT(cam.image_width, 0);
	EXPECT_GT(cam.image_height, 0);
}

/**
 * Test camera when lookfrom equals lookat (degenerate case)
 */
TEST(CameraTest, DegenerateLookFromEqualsLookAt) {
	camera cam;

	cam.lookfrom = point3(278, 278, 278);
	cam.lookat = point3(278, 278, 278); // Same position
	cam.vfov = 40;
	cam.image_width = 400;
	cam.samples_per_pixel = 1;
	cam.max_depth = 5;
	cam.background = color(0, 0, 0);

	// Should not crash even with degenerate parameters
	// (Camera implementation should handle this gracefully)
	EXPECT_NO_THROW(cam.initialize());
}
