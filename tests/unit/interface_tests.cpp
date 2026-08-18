/**
 * @file interface_tests.cpp
 * @brief Unit tests for CPU and GPU renderer interfaces
 * 
 * Tests the external C interfaces:
 * - cpu_render_main() parameter handling
 * - optix_render_main() parameter handling
 * - optix_is_available() detection
 * - Return code validation
 */

#include <gtest/gtest.h>
#include <string>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include <chrono>
#include "../../src/TheRestOfYourLife/error_codes.h"

// Include the C interfaces
extern "C" {
	#include "cpu_interface.h"
	#include "optix_interface.h"
}

// ============================================================================
// CPU Interface Tests
// ============================================================================

/**
 * Test CPU renderer with valid parameters
 */
TEST(CPUInterfaceTest, ValidParameters) {
	// Small test render: 100x100, 1 sample
	int width = 16;
	int height = 16;
	int spp = 1;
	int max_depth = 5;
	const char* output_path = "test_cpu_output.ppm";

	// Default camera position (front view)
	double cam_x = 278.0;
	double cam_y = 278.0;
	double cam_z = -800.0;

	int result = cpu_render_main(width, height, spp, max_depth, output_path, "A1", cam_x, cam_y, cam_z);

	// Should return 0 for success
	EXPECT_EQ(result, 0);

	// Clean up
	std::remove(output_path);
}

/**
 * Test CPU renderer with different camera positions
 */
TEST(CPUInterfaceTest, DifferentCameraPositions) {
	struct TestCase {
		const char* name;
		double cam_x, cam_y, cam_z;
	};

	TestCase cases[] = {
		{"Front View", 278, 278, -800},
		{"Left Wall", 50, 278, 278},
		{"Right Wall", 506, 278, 278},
		{"Top View", 278, 506, 278},
	};

	for (const auto& test : cases) {
		std::string output_path = std::string("test_cpu_") + test.name + ".ppm";

		int result = cpu_render_main(
			50, 50, 1, 3,  // Small, fast render
			output_path.c_str(),
			"A1", test.cam_x, test.cam_y, test.cam_z
		);

		EXPECT_EQ(result, 0) << "Failed for: " << test.name;

		// Clean up
		std::remove(output_path.c_str());
	}
}

/**
 * Test CPU renderer with minimum parameters
 */
TEST(CPUInterfaceTest, MinimumParameters) {
	int result = cpu_render_main(
		10, 10, 1, 1,  // Minimum size
		"test_cpu_min.ppm",
		"A1", 278, 278, -800
	);

	EXPECT_EQ(result, 0);
	std::remove("test_cpu_min.ppm");
}

/**
 * Test CPU renderer with larger parameters
 */
TEST(CPUInterfaceTest, LargerParameters) {
	int result = cpu_render_main(
		32, 32, 2, 5,  // Reduced for speed
		"test_cpu_large.ppm",
		"A1", 278, 278, -800
	);

	EXPECT_EQ(result, 0);
	std::remove("test_cpu_large.ppm");
}

// ============================================================================
// GPU Interface Tests
// ============================================================================

/**
 * Test GPU availability detection
 */
TEST(GPUInterfaceTest, GPUAvailability) {
	int available = optix_is_available();

	// Should return 0 (no OptiX) or 1 (OptiX available)
	EXPECT_TRUE(available == 0 || available == 1);

	if (available) {
		std::cout << "OptiX detected and available for testing" << std::endl;
	} else {
		std::cout << "OptiX not available - GPU tests will be skipped" << std::endl;
	}
}

/**
 * Test GPU renderer with valid parameters (if GPU available)
 */
TEST(GPUInterfaceTest, ValidParameters) {
	if (!optix_is_available()) {
		GTEST_SKIP() << "OptiX not available, skipping test";
	}

	// Small test render: 100x100, 100 samples
	int width = 32;
	int height = 32;
	int spp = 50;
	int max_depth = 5;
	const char* output_path = "test_gpu_output.ppm";

	int result = optix_render_main(width, height, spp, max_depth, output_path, "A1", 278.0, 278.0, -800.0);

	// Should return 0 for success
	EXPECT_EQ(result, 0);

	// Clean up
	std::remove(output_path);
}

/**
 * Test GPU renderer with different camera positions (if GPU available)
 */
TEST(GPUInterfaceTest, DifferentCameraPositions) {
	if (!optix_is_available()) {
		GTEST_SKIP() << "OptiX not available, skipping test";
	}

	struct TestCase {
		const char* name;
		double cam_x, cam_y, cam_z;
	};

	TestCase cases[] = {
		{"Front View", 278, 278, -800},
		{"Left Wall", 50, 278, 278},
		{"Right Wall", 506, 278, 278},
		{"Top View", 278, 506, 278},
	};

	for (const auto& test : cases) {
		std::string output_path = std::string("test_gpu_") + test.name + ".ppm";

		int result = optix_render_main(
			32, 32, 50, 3,  // Small, fast render
			output_path.c_str(),
			"A1", test.cam_x, test.cam_y, test.cam_z
		);

		EXPECT_EQ(result, 0) << "Failed for: " << test.name;

		// Clean up
		std::remove(output_path.c_str());
	}
}

// ============================================================================
// Parameter Validation Tests
// ============================================================================

/**
 * Test invalid width (zero)
 */
TEST(InterfaceValidationTest, ZeroWidth) {
	int result = cpu_render_main(
		0, 16, 1, 5,  // Zero width
		"test_invalid.ppm",
		"A1", 278, 278, -800
	);

	EXPECT_EQ(result, ERR_INVALID_DIMENSIONS);

	std::remove("test_invalid.ppm");
}

/**
 * Test invalid height (zero)
 */
TEST(InterfaceValidationTest, ZeroHeight) {
	int result = cpu_render_main(
		16, 0, 1, 5,  // Zero height
		"test_invalid.ppm",
		"A1", 278, 278, -800
	);

	EXPECT_EQ(result, ERR_INVALID_DIMENSIONS);
	std::remove("test_invalid.ppm");
}

/**
 * Test negative samples per pixel
 */
TEST(InterfaceValidationTest, NegativeSamples) {
	int result = cpu_render_main(
		16, 16, -1, 5,  // Negative samples
		"test_invalid.ppm",
		"A1", 278, 278, -800
	);

	EXPECT_EQ(result, ERR_INVALID_SAMPLE_COUNT);
	std::remove("test_invalid.ppm");
}

/**
 * Test null output path
 */
TEST(InterfaceValidationTest, NullOutputPath) {
	int result = cpu_render_main(
		16, 16, 1, 5,
		nullptr,
		"A1", 278, 278, -800
	);

	EXPECT_EQ(result, ERR_OUTPUT_PATH_INVALID);
}

// ============================================================================
// Camera Parameter Tests
// ============================================================================

/**
 * Test camera at Cornell box center
 */
TEST(CameraParametersTest, CenterPosition) {
	int result = cpu_render_main(
		50, 50, 1, 3,
		"test_center.ppm",
		"A1", 278, 278, 278  // At center
	);

	EXPECT_EQ(result, 0);
	std::remove("test_center.ppm");
}

/**
 * Test camera far away from scene
 */
TEST(CameraParametersTest, FarAwayCamera) {
	int result = cpu_render_main(
		50, 50, 1, 3,
		"test_far.ppm",
		"A1", 278, 278, -5000  // Very far
	);

	EXPECT_EQ(result, 0);
	std::remove("test_far.ppm");
}

/**
 * Test camera with extreme coordinates
 */
TEST(CameraParametersTest, ExtremeCoordinates) {
	int result = cpu_render_main(
		50, 50, 1, 3,
		"test_extreme.ppm",
		"A1", 10000, 10000, 10000  // Far outside scene
	);

	EXPECT_EQ(result, 0);
	std::remove("test_extreme.ppm");
}

// ============================================================================
// Output File Tests
// ============================================================================

/**
 * Test that output file is created
 */
TEST(OutputFileTest, FileCreation) {
	const char* output_path = "test_file_creation.ppm";

	int result = cpu_render_main(
		50, 50, 1, 3,
		output_path,
		"A1", 278, 278, -800
	);
	std::ifstream file(output_path);
	EXPECT_TRUE(file.good()) << "Output file was not created";
	file.close();

	std::remove(output_path);
}

/**
 * Test output path with directory
 */
TEST(OutputFileTest, PathWithDirectory) {
	// cpu_render_main creates missing parent directories for output_path
	// (cpu_interface.cpp's std::filesystem::create_directories call), so a
	// valid render request should succeed even into a not-yet-existing dir.
	const char* output_path = "test_output/test_render.ppm";

	int result = cpu_render_main(
		50, 50, 1, 3,
		output_path,
		"A1", 278, 278, -800
	);

	EXPECT_EQ(result, SUCCESS);

	std::remove(output_path);
	std::filesystem::remove("test_output");
}

// ============================================================================
// Performance Benchmarks (Optional)
// ============================================================================

/**
 * Benchmark small CPU render (for regression testing)
 */
TEST(BenchmarkTest, SmallCPURender) {
	auto start = std::chrono::high_resolution_clock::now();

	int result = cpu_render_main(
		100, 100, 1, 5,
		"benchmark_cpu.ppm",
		"A1", 278, 278, -800
	);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	EXPECT_EQ(result, 0);

	// Log performance (should complete in reasonable time)
	std::cout << "Small CPU render (100x100, 1 spp): " 
			  << duration.count() << " ms" << std::endl;

	// Sanity check: should complete within 30 seconds
	EXPECT_LT(duration.count(), 30000);

	std::remove("benchmark_cpu.ppm");
}

/**
 * Benchmark small GPU render (if available)
 */
TEST(BenchmarkTest, SmallGPURender) {
	if (!optix_is_available()) {
		GTEST_SKIP() << "GPU not available";
	}

	auto start = std::chrono::high_resolution_clock::now();

	int result = optix_render_main(
		100, 100, 10, 5,
		"benchmark_gpu.ppm",
		"A1", 278.0, 278.0, -800.0
	);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	EXPECT_EQ(result, 0);

	std::cout << "Small GPU render (100x100, 10 spp): " 
			  << duration.count() << " ms" << std::endl;

	// GPU should be fast for small renders
	EXPECT_LT(duration.count(), 10000);

	std::remove("benchmark_gpu.ppm");
}
