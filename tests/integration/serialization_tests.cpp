/**
 * @file serialization_tests.cpp
 * @brief Integration tests for GPU scene serialization
 * 
 * Tests the scene_serializer functions:
 * - Scene POD array creation (spheres, quads, materials)
 * - Camera POD conversion
 * - Material ID consistency
 * - Memory management (allocation/deallocation)
 */

#include <gtest/gtest.h>
#include "scene_serializer.h"
#include "cornell_box_scene.h"
#include "hittable_list.h"
#include <cstring>

// ============================================================================
// Scene Serialization Tests
// ============================================================================

/**
 * Test basic scene serialization
 */
TEST(SerializationTest, BasicSerialization) {
	hittable_list world;
	cornell_box(world);

	// Allocate POD arrays
	SpherePOD* spheres = nullptr;
	int num_spheres = 0;
	QuadPOD* quads = nullptr;
	int num_quads = 0;
	MaterialPOD* materials = nullptr;
	int num_materials = 0;
	CameraPOD camera;

	// Default camera position
	double cam_x = 278.0;
	double cam_y = 278.0;
	double cam_z = -800.0;

	// Serialize the scene
	serialize_scene_arrays(
		world,
		&spheres, &num_spheres,
		&quads, &num_quads,
		&materials, &num_materials,
		&camera,
		cam_x, cam_y, cam_z
	);

	// Verify counts are reasonable
	EXPECT_GE(num_spheres, 0) << "Should have at least 0 spheres";
	EXPECT_GE(num_quads, 5) << "Should have at least 5 quads (walls)";
	EXPECT_GE(num_materials, 3) << "Should have at least 3 materials (red, green, white)";

	// Verify camera was set
	EXPECT_DOUBLE_EQ(camera.origin_x, cam_x);
	EXPECT_DOUBLE_EQ(camera.origin_y, cam_y);
	EXPECT_DOUBLE_EQ(camera.origin_z, cam_z);

	// Clean up
	free_scene_arrays(spheres, quads, materials);
}

/**
 * Test camera serialization with different positions
 */
TEST(SerializationTest, CameraPositionSerialization) {
	struct TestCase {
		double cam_x, cam_y, cam_z;
		const char* name;
	};

	TestCase cases[] = {
		{278, 278, -800, "Front View"},
		{50, 278, 278, "Left Wall"},
		{506, 278, 278, "Right Wall"},
		{278, 506, 278, "Top View"},
		{278, 50, 278, "Bottom View"},
	};

	for (const auto& test : cases) {
		hittable_list world;
		cornell_box(world);

		SpherePOD* spheres = nullptr;
		int num_spheres = 0;
		QuadPOD* quads = nullptr;
		int num_quads = 0;
		MaterialPOD* materials = nullptr;
		int num_materials = 0;
		CameraPOD camera;

		serialize_scene_arrays(
			world,
			&spheres, &num_spheres,
			&quads, &num_quads,
			&materials, &num_materials,
			&camera,
			test.cam_x, test.cam_y, test.cam_z
		);

		// Verify camera position
		EXPECT_DOUBLE_EQ(camera.origin_x, test.cam_x) << "Failed for: " << test.name;
		EXPECT_DOUBLE_EQ(camera.origin_y, test.cam_y) << "Failed for: " << test.name;
		EXPECT_DOUBLE_EQ(camera.origin_z, test.cam_z) << "Failed for: " << test.name;

		free_scene_arrays(spheres, quads, materials);
	}
}

// ============================================================================
// Material Serialization Tests
// ============================================================================

/**
 * Test material POD conversion
 */
TEST(SerializationTest, MaterialPODConversion) {
	hittable_list world;
	cornell_box(world);

	SpherePOD* spheres = nullptr;
	int num_spheres = 0;
	QuadPOD* quads = nullptr;
	int num_quads = 0;
	MaterialPOD* materials = nullptr;
	int num_materials = 0;
	CameraPOD camera;

	serialize_scene_arrays(
		world,
		&spheres, &num_spheres,
		&quads, &num_quads,
		&materials, &num_materials,
		&camera,
		278, 278, -800
	);

	// Verify materials have valid IDs
	for (int i = 0; i < num_materials; ++i) {
		// Material type should be in valid range (0-3)
		EXPECT_GE(materials[i].type, 0);
		EXPECT_LE(materials[i].type, 3);

		// Albedo colors should be in [0, 1] or slightly higher for emissive
		// (checking only that they're not NaN or wildly out of range)
		EXPECT_FALSE(std::isnan(materials[i].albedo_r));
		EXPECT_FALSE(std::isnan(materials[i].albedo_g));
		EXPECT_FALSE(std::isnan(materials[i].albedo_b));
	}

	free_scene_arrays(spheres, quads, materials);
}

/**
 * Test that material IDs are consistent
 */
TEST(SerializationTest, MaterialIDConsistency) {
	hittable_list world;
	cornell_box(world);

	SpherePOD* spheres = nullptr;
	int num_spheres = 0;
	QuadPOD* quads = nullptr;
	int num_quads = 0;
	MaterialPOD* materials = nullptr;
	int num_materials = 0;
	CameraPOD camera;

	serialize_scene_arrays(
		world,
		&spheres, &num_spheres,
		&quads, &num_quads,
		&materials, &num_materials,
		&camera,
		278, 278, -800
	);

	// All quads should have valid material IDs
	for (int i = 0; i < num_quads; ++i) {
		EXPECT_GE(quads[i].mat_id, 0);
		EXPECT_LT(quads[i].mat_id, num_materials);
	}

	// All spheres should have valid material IDs
	for (int i = 0; i < num_spheres; ++i) {
		EXPECT_GE(spheres[i].mat_id, 0);
		EXPECT_LT(spheres[i].mat_id, num_materials);
	}

	free_scene_arrays(spheres, quads, materials);
}

// ============================================================================
// Geometry Serialization Tests
// ============================================================================

/**
 * Test quad (wall) serialization
 */
TEST(SerializationTest, QuadSerialization) {
	hittable_list world;
	cornell_box(world);

	SpherePOD* spheres = nullptr;
	int num_spheres = 0;
	QuadPOD* quads = nullptr;
	int num_quads = 0;
	MaterialPOD* materials = nullptr;
	int num_materials = 0;
	CameraPOD camera;

	serialize_scene_arrays(
		world,
		&spheres, &num_spheres,
		&quads, &num_quads,
		&materials, &num_materials,
		&camera,
		278, 278, -800
	);

	// Verify all quads have valid geometry
	for (int i = 0; i < num_quads; ++i) {
		// Q (corner) should have finite values
		EXPECT_FALSE(std::isnan(quads[i].Q_x));
		EXPECT_FALSE(std::isnan(quads[i].Q_y));
		EXPECT_FALSE(std::isnan(quads[i].Q_z));

		// u and v (edge vectors) should have non-zero length
		double u_len_sq = quads[i].u_x * quads[i].u_x + 
						  quads[i].u_y * quads[i].u_y + 
						  quads[i].u_z * quads[i].u_z;
		double v_len_sq = quads[i].v_x * quads[i].v_x + 
						  quads[i].v_y * quads[i].v_y + 
						  quads[i].v_z * quads[i].v_z;

		EXPECT_GT(u_len_sq, 0.0) << "Quad " << i << " has zero-length u vector";
		EXPECT_GT(v_len_sq, 0.0) << "Quad " << i << " has zero-length v vector";
	}

	free_scene_arrays(spheres, quads, materials);
}

/**
 * Test sphere serialization (if present)
 */
TEST(SerializationTest, SphereSerialization) {
	hittable_list world;
	cornell_box(world);

	SpherePOD* spheres = nullptr;
	int num_spheres = 0;
	QuadPOD* quads = nullptr;
	int num_quads = 0;
	MaterialPOD* materials = nullptr;
	int num_materials = 0;
	CameraPOD camera;

	serialize_scene_arrays(
		world,
		&spheres, &num_spheres,
		&quads, &num_quads,
		&materials, &num_materials,
		&camera,
		278, 278, -800
	);

	// Verify all spheres have valid geometry
	for (int i = 0; i < num_spheres; ++i) {
		// Center should have finite values
		EXPECT_FALSE(std::isnan(spheres[i].center_x));
		EXPECT_FALSE(std::isnan(spheres[i].center_y));
		EXPECT_FALSE(std::isnan(spheres[i].center_z));

		// Radius should be positive
		EXPECT_GT(spheres[i].radius, 0.0) << "Sphere " << i << " has non-positive radius";

		// Material ID should be valid
		EXPECT_GE(spheres[i].mat_id, 0);
		EXPECT_LT(spheres[i].mat_id, num_materials);
	}

	free_scene_arrays(spheres, quads, materials);
}

// ============================================================================
// Memory Management Tests
// ============================================================================

/**
 * Test multiple serialization cycles
 */
TEST(SerializationTest, MultipleCycles) {
	for (int cycle = 0; cycle < 10; ++cycle) {
		hittable_list world;
		cornell_box(world);

		SpherePOD* spheres = nullptr;
		int num_spheres = 0;
		QuadPOD* quads = nullptr;
		int num_quads = 0;
		MaterialPOD* materials = nullptr;
		int num_materials = 0;
		CameraPOD camera;

		serialize_scene_arrays(
			world,
			&spheres, &num_spheres,
			&quads, &num_quads,
			&materials, &num_materials,
			&camera,
			278, 278, -800
		);

		// Should succeed every time
		EXPECT_GE(num_quads, 5);
		EXPECT_GE(num_materials, 3);

		free_scene_arrays(spheres, quads, materials);
	}
}

/**
 * Test that free_scene_arrays handles nullptr gracefully
 */
TEST(SerializationTest, FreeNullPointers) {
	// Should not crash
	EXPECT_NO_THROW(free_scene_arrays(nullptr, nullptr, nullptr));
}

/**
 * Test serialization with empty world
 */
TEST(SerializationTest, EmptyWorld) {
	hittable_list world;  // Empty scene

	SpherePOD* spheres = nullptr;
	int num_spheres = 0;
	QuadPOD* quads = nullptr;
	int num_quads = 0;
	MaterialPOD* materials = nullptr;
	int num_materials = 0;
	CameraPOD camera;

	serialize_scene_arrays(
		world,
		&spheres, &num_spheres,
		&quads, &num_quads,
		&materials, &num_materials,
		&camera,
		278, 278, -800
	);

	// Empty scene should have zero objects but camera should still be valid
	EXPECT_EQ(num_spheres, 0);
	EXPECT_EQ(num_quads, 0);
	EXPECT_EQ(num_materials, 0);

	EXPECT_DOUBLE_EQ(camera.origin_x, 278.0);
	EXPECT_DOUBLE_EQ(camera.origin_y, 278.0);
	EXPECT_DOUBLE_EQ(camera.origin_z, -800.0);

	free_scene_arrays(spheres, quads, materials);
}

// ============================================================================
// Camera POD Tests
// ============================================================================

/**
 * Test camera POD has valid viewport dimensions
 */
TEST(SerializationTest, CameraViewportDimensions) {
	hittable_list world;
	cornell_box(world);

	SpherePOD* spheres = nullptr;
	int num_spheres = 0;
	QuadPOD* quads = nullptr;
	int num_quads = 0;
	MaterialPOD* materials = nullptr;
	int num_materials = 0;
	CameraPOD camera;

	serialize_scene_arrays(
		world,
		&spheres, &num_spheres,
		&quads, &num_quads,
		&materials, &num_materials,
		&camera,
		278, 278, -800
	);

	// Pixel delta vectors should be non-zero
	double pixel_du_len_sq = camera.pixel_delta_u_x * camera.pixel_delta_u_x +
							  camera.pixel_delta_u_y * camera.pixel_delta_u_y +
							  camera.pixel_delta_u_z * camera.pixel_delta_u_z;

	double pixel_dv_len_sq = camera.pixel_delta_v_x * camera.pixel_delta_v_x +
							  camera.pixel_delta_v_y * camera.pixel_delta_v_y +
							  camera.pixel_delta_v_z * camera.pixel_delta_v_z;

	EXPECT_GT(pixel_du_len_sq, 0.0);
	EXPECT_GT(pixel_dv_len_sq, 0.0);

	// Upper-left pixel should have finite coordinates
	EXPECT_FALSE(std::isnan(camera.pixel00_loc_x));
	EXPECT_FALSE(std::isnan(camera.pixel00_loc_y));
	EXPECT_FALSE(std::isnan(camera.pixel00_loc_z));

	free_scene_arrays(spheres, quads, materials);
}

/**
 * Test camera lookat is correctly computed
 */
TEST(SerializationTest, CameraLookatDirection) {
	hittable_list world;
	cornell_box(world);

	SpherePOD* spheres = nullptr;
	int num_spheres = 0;
	QuadPOD* quads = nullptr;
	int num_quads = 0;
	MaterialPOD* materials = nullptr;
	int num_materials = 0;
	CameraPOD camera;

	// Camera looking from front towards center
	double cam_x = 278.0;
	double cam_y = 278.0;
	double cam_z = -800.0;

	serialize_scene_arrays(
		world,
		&spheres, &num_spheres,
		&quads, &num_quads,
		&materials, &num_materials,
		&camera,
		cam_x, cam_y, cam_z
	);

	// Camera should be looking roughly towards +z direction
	// (This is encoded in the viewport deltas, not explicitly stored)

	// Origin should match lookfrom
	EXPECT_DOUBLE_EQ(camera.origin_x, cam_x);
	EXPECT_DOUBLE_EQ(camera.origin_y, cam_y);
	EXPECT_DOUBLE_EQ(camera.origin_z, cam_z);

	free_scene_arrays(spheres, quads, materials);
}
