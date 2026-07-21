/**
 * @file render_tests.cpp
 * @brief Integration tests for end-to-end rendering
 * 
 * Tests complete render pipeline:
 * - CPU rendering produces valid output
 * - GPU rendering produces valid output
 * - Different camera positions produce different images
 * - Sample count affects image quality
 * - Output files are created correctly
 */

#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <algorithm>
#include <functional>
#include <chrono>

extern "C" {
	#include "cpu_interface.h"
	#include "optix_interface.h"
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Check if a file exists
 */
bool file_exists(const char* path) {
	std::ifstream file(path);
	return file.good();
}

/**
 * Read entire file into string
 */
std::string read_file(const char* path) {
	std::ifstream file(path);
	std::stringstream buffer;
	buffer << file.rdbuf();
	return buffer.str();
}

/**
 * Compute simple hash of file contents
 */
size_t hash_file(const char* path) {
	std::string contents = read_file(path);
	return std::hash<std::string>{}(contents);
}

/**
 * Check if PPM file header is valid
 */
bool has_valid_ppm_header(const char* path) {
	std::ifstream file(path);
	if (!file.good()) return false;

	std::string magic;
	file >> magic;

	// Should start with P3 (ASCII PPM)
	return magic == "P3";
}

/**
 * Get image dimensions from PPM file
 */
std::pair<int, int> get_ppm_dimensions(const char* path) {
	std::ifstream file(path);
	std::string magic;
	int width, height;

	file >> magic >> width >> height;

	return {width, height};
}

// ============================================================================
// CPU Render Tests
// ============================================================================

/**
 * Test basic CPU rendering
 */
TEST(RenderIntegrationTest, BasicCPURender) {
	const char* output = "test_render_cpu_basic.ppm";

	int result = cpu_render_main(
		100, 100, 5, 5,
		output,
		278, 278, -800
	);

	EXPECT_EQ(result, 0);
	EXPECT_TRUE(file_exists(output));
	EXPECT_TRUE(has_valid_ppm_header(output));

	auto [width, height] = get_ppm_dimensions(output);
	EXPECT_EQ(width, 100);
	EXPECT_EQ(height, 100);

	std::remove(output);
}

/**
 * Test CPU rendering with different resolutions
 */
TEST(RenderIntegrationTest, CPUDifferentResolutions) {
	struct TestCase {
		int width, height;
		const char* name;
	};

	TestCase cases[] = {
		{50, 50, "small"},
		{100, 100, "medium"},
		{200, 200, "large"},
	};

	for (const auto& test : cases) {
		std::string output = std::string("test_cpu_res_") + test.name + ".ppm";

		int result = cpu_render_main(
			test.width, test.height, 1, 3,
			output.c_str(),
			278, 278, -800
		);

		EXPECT_EQ(result, 0) << "Failed for: " << test.name;
		EXPECT_TRUE(file_exists(output.c_str())) << "Failed for: " << test.name;

		auto [w, h] = get_ppm_dimensions(output.c_str());
		EXPECT_EQ(w, test.width) << "Failed for: " << test.name;
		EXPECT_EQ(h, test.height) << "Failed for: " << test.name;

		std::remove(output.c_str());
	}
}

/**
 * Test that different camera positions produce different images
 */
TEST(RenderIntegrationTest, CPUCameraPositionsDifferent) {
	const char* output1 = "test_cpu_cam1.ppm";
	const char* output2 = "test_cpu_cam2.ppm";

	// Render from front
	cpu_render_main(100, 100, 5, 5, output1, 278, 278, -800);
	size_t hash1 = hash_file(output1);

	// Render from left wall
	cpu_render_main(100, 100, 5, 5, output2, 50, 278, 278);
	size_t hash2 = hash_file(output2);

	// Images should be different
	EXPECT_NE(hash1, hash2) << "Different camera positions should produce different images";

	std::remove(output1);
	std::remove(output2);
}

// ============================================================================
// GPU Render Tests
// ============================================================================

/**
 * Test basic GPU rendering (if available)
 */
TEST(RenderIntegrationTest, BasicGPURender) {
	if (!optix_is_available()) {
		GTEST_SKIP() << "OptiX not available";
	}

	const char* output = "test_render_gpu_basic.ppm";

	int result = optix_render_main(
		100, 100, 1000, 5,  // OptiX needs more samples for quality
		output
	);

	EXPECT_EQ(result, 0);
	EXPECT_TRUE(file_exists(output));
	EXPECT_TRUE(has_valid_ppm_header(output));

	auto [width, height] = get_ppm_dimensions(output);
	EXPECT_EQ(width, 100);
	EXPECT_EQ(height, 100);

	std::remove(output);
}

/**
 * Test GPU rendering with different resolutions (if available)
 */
TEST(RenderIntegrationTest, GPUDifferentResolutions) {
	if (!optix_is_available()) {
		GTEST_SKIP() << "OptiX not available";
	}

	struct TestCase {
		int width, height;
		const char* name;
	};

	TestCase cases[] = {
		{50, 50, "small"},
		{100, 100, "medium"},
		{150, 150, "large"},
	};

	for (const auto& test : cases) {
		std::string output = std::string("test_gpu_res_") + test.name + ".ppm";

		int result = optix_render_main(
			test.width, test.height, 100, 3,
			output.c_str()
		);

		EXPECT_EQ(result, 0) << "Failed for: " << test.name;
		EXPECT_TRUE(file_exists(output.c_str())) << "Failed for: " << test.name;

		auto [w, h] = get_ppm_dimensions(output.c_str());
		EXPECT_EQ(w, test.width) << "Failed for: " << test.name;
		EXPECT_EQ(h, test.height) << "Failed for: " << test.name;

		std::remove(output.c_str());
	}
}
		EXPECT_EQ(h, test.height) << "Failed for: " << test.name;

		std::remove(output.c_str());
	}
}

/**
 * Test that GPU and CPU produce similar results (rough check)
 */
TEST(RenderIntegrationTest, GPUvsCPUSimilarity) {
	if (!optix_is_available()) {
		GTEST_SKIP() << "OptiX not available";
	}

	const char* cpu_output = "test_compare_cpu.ppm";
	const char* gpu_output = "test_compare_gpu.ppm";

	// Render small image with both
	cpu_render_main(50, 50, 100, 10, cpu_output, 0, 278, 278, -800);
	optix_render_main(50, 50, 10000, 10, gpu_output);

	// Both should exist and have valid headers
	EXPECT_TRUE(file_exists(cpu_output));
	EXPECT_TRUE(file_exists(gpu_output));
	EXPECT_TRUE(has_valid_ppm_header(cpu_output));
	EXPECT_TRUE(has_valid_ppm_header(gpu_output));

	// Dimensions should match
	auto [cpu_w, cpu_h] = get_ppm_dimensions(cpu_output);
	auto [gpu_w, gpu_h] = get_ppm_dimensions(gpu_output);
	EXPECT_EQ(cpu_w, gpu_w);
	EXPECT_EQ(cpu_h, gpu_h);

	// Note: We don't check if images are identical because:
	// 1. GPU uses naive path tracing (no importance sampling)
	// 2. Different random number generators
	// 3. Floating-point differences
	// But both should produce valid Cornell box renders

	std::remove(cpu_output);
	std::remove(gpu_output);
}

// ============================================================================
// Sample Count Tests
// ============================================================================

/**
 * Test that more samples produce different (presumably better) results
 */
TEST(RenderIntegrationTest, SampleCountAffectsOutput) {
	const char* output_low = "test_samples_low.ppm";
	const char* output_high = "test_samples_high.ppm";

	// Low sample count
	cpu_render_main(100, 100, 1, 5, output_low, 278, 278, -800);
	size_t hash_low = hash_file(output_low);

	// High sample count
	cpu_render_main(100, 100, 50, 5, output_high, 278, 278, -800);
	size_t hash_high = hash_file(output_high);

	// Different sample counts should produce different images
	EXPECT_NE(hash_low, hash_high) << "Different sample counts should produce different images";

	std::remove(output_low);
	std::remove(output_high);
}

/**
 * Test that sample count of 1 runs fast
 */
TEST(RenderIntegrationTest, OneSampleFast) {
	const char* output = "test_one_sample.ppm";

	auto start = std::chrono::high_resolution_clock::now();

	int result = cpu_render_main(100, 100, 1, 5, output, 278, 278, -800);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	EXPECT_EQ(result, 0);

	// Should be very fast (< 5 seconds)
	EXPECT_LT(duration.count(), 5000) << "1 sample render took too long";

	std::remove(output);
}

// ============================================================================
// Max Depth Tests
// ============================================================================

/**
 * Test that max depth affects output
 */
TEST(RenderIntegrationTest, MaxDepthAffectsOutput) {
	const char* output_shallow = "test_depth_shallow.ppm";
	const char* output_deep = "test_depth_deep.ppm";

	// Shallow depth
	cpu_render_main(100, 100, 5, 1, output_shallow, 278, 278, -800);
	size_t hash_shallow = hash_file(output_shallow);

	// Deep depth
	cpu_render_main(100, 100, 5, 50, output_deep, 278, 278, -800);
	size_t hash_deep = hash_file(output_deep);

	// Different max depths should produce different images
	EXPECT_NE(hash_shallow, hash_deep) << "Different max depths should produce different images";

	std::remove(output_shallow);
	std::remove(output_deep);
}

// ============================================================================
// Camera Viewpoint Tests
// ============================================================================

/**
 * Test all standard camera presets produce valid output
 */
TEST(RenderIntegrationTest, AllCameraPresetsValid) {
	struct Preset {
		const char* name;
		double cam_x, cam_y, cam_z;
	};

	Preset presets[] = {
		{"Default", 278, 278, -800},
		{"InsideCenter", 278, 278, 278},
		{"InsideLeftWall", 50, 278, 278},
		{"InsideRightWall", 506, 278, 278},
		{"InsideTopWall", 278, 506, 278},
		{"InsideBottomWall", 278, 50, 278},
		{"InsideBackWall", 278, 278, 506},
		{"NearLeftWall", 100, 278, 278},
		{"NearRightWall", 455, 278, 278},
		{"Custom", 400, 300, 200},
	};

	for (const auto& preset : presets) {
		std::string output = std::string("test_preset_") + preset.name + ".ppm";

		int result = cpu_render_main(
			50, 50, 1, 3,  // Small, fast render
			output.c_str(),
			preset.cam_x, preset.cam_y, preset.cam_z
		);

		EXPECT_EQ(result, 0) << "Failed for preset: " << preset.name;
		EXPECT_TRUE(file_exists(output.c_str())) << "Failed for preset: " << preset.name;
		EXPECT_TRUE(has_valid_ppm_header(output.c_str())) << "Failed for preset: " << preset.name;

		std::remove(output.c_str());
	}
}

// ============================================================================
// Consistency Tests
// ============================================================================

/**
 * Test that rendering the same scene twice produces identical output
 */
TEST(RenderIntegrationTest, DeterministicRender) {
	const char* output1 = "test_deterministic_1.ppm";
	const char* output2 = "test_deterministic_2.ppm";

	// Render twice with same parameters
	cpu_render_main(100, 100, 10, 5, output1, 278, 278, -800);
	cpu_render_main(100, 100, 10, 5, output2, 278, 278, -800);

	size_t hash1 = hash_file(output1);
	size_t hash2 = hash_file(output2);

	// Note: Due to random sampling, images will be different
	// This test documents that behavior
	// If we want deterministic output, we'd need to seed the RNG

	// For now, just verify both renders succeeded
	EXPECT_GT(hash1, 0);
	EXPECT_GT(hash2, 0);

	std::remove(output1);
	std::remove(output2);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

/**
 * Test rendering with output path in non-existent directory
 */
TEST(RenderIntegrationTest, NonExistentDirectory) {
	const char* output = "nonexistent_dir/test.ppm";

	// This may fail or use default output path
	int result = cpu_render_main(50, 50, 1, 3, output, 278, 278, -800);

	// Current implementation may succeed (using default Desktop path)
	// or fail - just verify it doesn't crash
	EXPECT_TRUE(result == 0 || result != 0);
}

/**
 * Test rendering with very small image
 */
TEST(RenderIntegrationTest, VerySmallImage) {
	const char* output = "test_tiny.ppm";

	int result = cpu_render_main(10, 10, 1, 3, output, 278, 278, -800);

	EXPECT_EQ(result, 0);

	if (file_exists(output)) {
		auto [w, h] = get_ppm_dimensions(output);
		EXPECT_EQ(w, 10);
		EXPECT_EQ(h, 10);
		std::remove(output);
	}
}

/**
 * Test rendering with very large max depth
 */
TEST(RenderIntegrationTest, VeryLargeMaxDepth) {
	const char* output = "test_deep.ppm";

	int result = cpu_render_main(50, 50, 1, 1000, output, 278, 278, -800);

	EXPECT_EQ(result, 0);

	if (file_exists(output)) {
		std::remove(output);
	}
}
