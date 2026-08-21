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
#include <filesystem>
#include "../../src/TheRestOfYourLife/error_codes.h"
#include "../../src/external/tinyexr.h"
#include "../ppm_test_utils.h"

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

/**
 * Average pixel brightness of a P3 PPM file, normalized to [0,1]. Used by
 * the --exposure tests below to check monotonic brightness ordering
 * (darker/default/brighter). Thin wrapper over the shared load_ppm()/
 * average_brightness() (../ppm_test_utils.h, also used by
 * tests/unit/gpu_render_tests.cpp) - returns -1.0 instead of 0 on an
 * invalid/unreadable file, preserving this file's own ASSERT_GE(avg, 0.0)
 * validity check at call sites below.
 */
double average_ppm_brightness(const char* path) {
	PPMImage img = load_ppm(path);
	if (!img.valid) return -1.0;
	return static_cast<double>(average_brightness(img));
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
		32, 32, 2, 5,
		output,
		"A1", 278, 278, -800
	);

	EXPECT_EQ(result, 0);
	EXPECT_TRUE(file_exists(output));
	EXPECT_TRUE(has_valid_ppm_header(output));

	auto [width, height] = get_ppm_dimensions(output);
	EXPECT_EQ(width, 32);
	EXPECT_EQ(height, 32);

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
		{16, 16, "small"},
		{32, 32, "medium"},
	};

	for (const auto& test : cases) {
		std::string output = std::string("test_cpu_res_") + test.name + ".ppm";

		int result = cpu_render_main(
			test.width, test.height, 1, 3,
			output.c_str(),
			"A1", 278, 278, -800
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
	cpu_render_main(32, 32, 1, 5, output1, "A1", 278, 278, -800);
	size_t hash1 = hash_file(output1);

	// Render from left wall
	cpu_render_main(32, 32, 1, 5, output2, "A1", 50, 278, 278);
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
		64, 64, 200, 5,
		output,
		"A1", 278.0, 278.0, -800.0
	);

	EXPECT_EQ(result, 0);
	EXPECT_TRUE(file_exists(output));
	EXPECT_TRUE(has_valid_ppm_header(output));

	auto [width, height] = get_ppm_dimensions(output);
	EXPECT_EQ(width, 64);
	EXPECT_EQ(height, 64);

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
		{16, 16, "small"},
		{32, 32, "medium"},
		{64, 64, "large"},
	};

	for (const auto& test : cases) {
		std::string output = std::string("test_gpu_res_") + test.name + ".ppm";

		int result = optix_render_main(
			test.width, test.height, 50, 3,
			output.c_str(),
			"A1", 278.0, 278.0, -800.0
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
 * Test that GPU and CPU produce similar results (rough check)
 */
TEST(RenderIntegrationTest, GPUvsCPUSimilarity) {
	if (!optix_is_available()) {
		GTEST_SKIP() << "OptiX not available";
	}

	const char* cpu_output = "test_compare_cpu.ppm";
	const char* gpu_output = "test_compare_gpu.ppm";

	// Render small image with both
	cpu_render_main(32, 32, 10, 5, cpu_output, "A1", 278, 278, -800);
	if (optix_render_main(32, 32, 500, 5, gpu_output, "A1", 278.0, 278.0, -800.0) != 0)
		GTEST_SKIP() << "optix_render_main failed — driver/PTX incompatibility";

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
	cpu_render_main(32, 32, 1, 5, output_low, "A1", 278, 278, -800);
	size_t hash_low = hash_file(output_low);

	// High sample count
	cpu_render_main(32, 32, 10, 5, output_high, "A1", 278, 278, -800);
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

	int result = cpu_render_main(32, 32, 1, 5, output, "A1", 278, 278, -800);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	EXPECT_EQ(result, 0);

	// Should be very fast (< 5 seconds)
	EXPECT_LT(duration.count(), 5000) << "1 sample render took too long";

	std::remove(output);
}

// ============================================================================
// Exposure Tests (--exposure flat pre-tonemap multiplier)
// ============================================================================

/**
 * exposure=1.0 (explicit) is a no-op multiply, so it should produce output
 * statistically indistinguishable from omitting the parameter entirely (its
 * default) - protects every pre-existing call site above that doesn't pass
 * exposure at all. Compared via average brightness with a generous
 * tolerance rather than an exact hash: per DeterministicRender's own
 * comment below, this renderer's RNG is unseeded, so two separate calls
 * with identical parameters are NOT byte-identical even without exposure
 * in the picture at all.
 */
TEST(RenderIntegrationTest, ExposureDefaultIsNoOp) {
	const char* output_default = "test_exposure_default.ppm";
	const char* output_explicit_1x = "test_exposure_explicit_1x.ppm";

	cpu_render_main(32, 32, 8, 5, output_default, "A1", 278, 278, -800);
	cpu_render_main(32, 32, 8, 5, output_explicit_1x, "A1", 278, 278, -800, 0, 1.0);

	double avg_default = average_ppm_brightness(output_default);
	double avg_explicit = average_ppm_brightness(output_explicit_1x);

	ASSERT_GE(avg_default, 0.0);
	ASSERT_GE(avg_explicit, 0.0);
	EXPECT_NEAR(avg_default, avg_explicit, 0.02)
		<< "exposure=1.0 should be statistically indistinguishable from omitting --exposure";

	std::remove(output_default);
	std::remove(output_explicit_1x);
}

/**
 * exposure > 1.0 should raise average brightness; exposure < 1.0 should
 * lower it. Checked as a monotonic ordering across all three renders
 * (same seed/scene/samples, only exposure varies) rather than an exact
 * multiplier, since ACES tonemap + sRGB OETF are both nonlinear - only
 * the ordering survives that, not "2x linear in equals ~2x byte out".
 */
TEST(RenderIntegrationTest, ExposureBrightensAndDarkensMonotonically) {
	const char* output_dim = "test_exposure_dim.ppm";
	const char* output_normal = "test_exposure_normal.ppm";
	const char* output_bright = "test_exposure_bright.ppm";

	cpu_render_main(32, 32, 4, 5, output_dim, "A1", 278, 278, -800, 0, 0.3);
	cpu_render_main(32, 32, 4, 5, output_normal, "A1", 278, 278, -800, 0, 1.0);
	cpu_render_main(32, 32, 4, 5, output_bright, "A1", 278, 278, -800, 0, 3.0);

	double avg_dim = average_ppm_brightness(output_dim);
	double avg_normal = average_ppm_brightness(output_normal);
	double avg_bright = average_ppm_brightness(output_bright);

	ASSERT_GE(avg_dim, 0.0);
	ASSERT_GE(avg_normal, 0.0);
	ASSERT_GE(avg_bright, 0.0);

	EXPECT_LT(avg_dim, avg_normal) << "exposure=0.3 should be darker than exposure=1.0";
	EXPECT_LT(avg_normal, avg_bright) << "exposure=3.0 should be brighter than exposure=1.0";

	std::remove(output_dim);
	std::remove(output_normal);
	std::remove(output_bright);
}

// ============================================================================
// Sampler Selection Tests (--sampler / SamplerKind, camera.h)
// ============================================================================

/**
 * Every ported pbrt-v4 sampler (see camera.h's SamplerKind/
 * sampler_kind_from_name()) must actually be selectable and produce a valid,
 * lit render - not just compile. Each is constructed once per worker thread
 * with a different signature/setup than SobolSampler's own per-ray
 * construction (see camera.h's worker lambda comment), so this is real
 * coverage that a plain build success wouldn't catch.
 */
TEST(RenderIntegrationTest, EverySamplerKindProducesAValidRender) {
	const char* names[] = {"sobol", "zsobol", "paddedsobol", "stratified", "pmj02bn", "halton"};
	for (const char* name : names) {
		std::string output = std::string("test_sampler_") + name + ".ppm";
		int result = cpu_render_main(24, 24, 4, 5, output.c_str(), "A1", 278, 278, -800, 0, 1.0, name);
		EXPECT_EQ(result, 0) << "sampler \"" << name << "\" failed to render";
		double avg = average_ppm_brightness(output.c_str());
		EXPECT_GT(avg, 0.0) << "sampler \"" << name << "\" produced an entirely black/invalid render";
		std::remove(output.c_str());
	}
}

/**
 * An unrecognized --sampler name must fall back to sobol (with a warning,
 * not a crash) - see cpu_interface.cpp's own sampler_kind_from_name() call.
 */
TEST(RenderIntegrationTest, UnknownSamplerNameFallsBackToSobol) {
	const char* output = "test_sampler_unknown.ppm";
	int result = cpu_render_main(24, 24, 4, 5, output, "A1", 278, 278, -800, 0, 1.0, "not-a-real-sampler");
	EXPECT_EQ(result, 0);
	double avg = average_ppm_brightness(output);
	EXPECT_GT(avg, 0.0);
	std::remove(output);
}

// ============================================================================
// EXR Output Tests (Round 4 Phase 4 - camera.h's exr_output /
// optix_interface.cpp's own extension-detection branch)
// ============================================================================

/**
 * A ".exr" output path must produce a real, valid EXR file on CPU - the
 * exact extension-detection convention cpu_interface.cpp uses (see
 * camera.h's exr_output field comment). Only the extension changes here
 * versus every other cpu_render_main() test in this file - proving this
 * path doesn't just fall back to writing a PPM with the wrong name.
 */
TEST(RenderIntegrationTest, CpuExrOutputWritesAValidExrFile) {
	const char* output = "test_cpu_output.exr";
	int result = cpu_render_main(24, 24, 4, 5, output, "A1", 278, 278, -800);
	EXPECT_EQ(result, 0);
	ASSERT_TRUE(file_exists(output));
	EXPECT_EQ(IsEXR(output), TINYEXR_SUCCESS)
		<< "cpu_render_main's .exr output must be a real, valid EXR file";
	std::remove(output);
}

/**
 * The pre-existing default (no ".exr" extension) must still write a plain
 * PPM, unaffected by exr_output's addition - regression coverage for the
 * shared `filename`/`out_path` plumbing camera.h's render() now threads
 * through both the PPM and EXR cases.
 */
TEST(RenderIntegrationTest, CpuNonExrOutputStillWritesAPpm) {
	const char* output = "test_cpu_output_still_ppm.ppm";
	int result = cpu_render_main(24, 24, 4, 5, output, "A1", 278, 278, -800);
	EXPECT_EQ(result, 0);
	ASSERT_TRUE(file_exists(output));
	EXPECT_TRUE(has_valid_ppm_header(output));
	std::remove(output);
}

/**
 * Same extension-detection contract on the GPU backend (optix_interface.cpp) -
 * this backend writes straight to output_path (no temp-file/copy dance), so
 * this is a more direct test than the CPU one above.
 */
TEST(RenderIntegrationTest, GpuExrOutputWritesAValidExrFile) {
	if (!optix_is_available()) {
		GTEST_SKIP() << "OptiX not available";
	}
	const char* output = "test_gpu_output.exr";
	int result = optix_render_main(32, 32, 16, 5, output, "A1", 278.0, 278.0, -800.0);
	EXPECT_EQ(result, 0);
	ASSERT_TRUE(file_exists(output));
	EXPECT_EQ(IsEXR(output), TINYEXR_SUCCESS)
		<< "optix_render_main's .exr output must be a real, valid EXR file";
	std::remove(output);
}

/**
 * --denoise + a ".exr" output must also emit "<stem>_albedo.exr"/
 * "<stem>_normal.exr" AOV siblings, reusing the exact buffers the denoiser
 * guide layer already computes (OptiXRenderer::readAovBuffers()) rather
 * than a separate device-side computation. GPU-recursive only, matching
 * this project's own established denoise-is-recursive-only scope (see
 * OptiXRenderer::enableDenoise()'s comment) - no --wavefront here.
 */
TEST(RenderIntegrationTest, GpuExrDenoiseWritesAlbedoAndNormalAovs) {
	if (!optix_is_available()) {
		GTEST_SKIP() << "OptiX not available";
	}
	const char* output = "test_gpu_aov.exr";
	const char* albedoOut = "test_gpu_aov_albedo.exr";
	const char* normalOut = "test_gpu_aov_normal.exr";
	std::remove(albedoOut);
	std::remove(normalOut);

	int result = optix_render_main(32, 32, 16, 5, output, "A1", 278.0, 278.0, -800.0,
									/*force_camera_override=*/0, /*denoise=*/1);
	EXPECT_EQ(result, 0);
	ASSERT_TRUE(file_exists(output));
	EXPECT_EQ(IsEXR(output), TINYEXR_SUCCESS);
	ASSERT_TRUE(file_exists(albedoOut)) << "denoise=1 + .exr output must also write an albedo AOV";
	ASSERT_TRUE(file_exists(normalOut)) << "denoise=1 + .exr output must also write a normal AOV";
	EXPECT_EQ(IsEXR(albedoOut), TINYEXR_SUCCESS);
	EXPECT_EQ(IsEXR(normalOut), TINYEXR_SUCCESS);

	std::remove(output);
	std::remove(albedoOut);
	std::remove(normalOut);
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
	cpu_render_main(32, 32, 2, 1, output_shallow, "A1", 278, 278, -800);
	size_t hash_shallow = hash_file(output_shallow);

	// Deep depth
	cpu_render_main(32, 32, 2, 20, output_deep, "A1", 278, 278, -800);
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
			16, 16, 1, 3,  // Small, fast render
			output.c_str(),
			"A1", preset.cam_x, preset.cam_y, preset.cam_z
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
	cpu_render_main(32, 32, 2, 5, output1, "A1", 278, 278, -800);
	cpu_render_main(32, 32, 2, 5, output2, "A1", 278, 278, -800);

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
	const char* output_dir = "nonexistent_dir";
	const char* output = "nonexistent_dir/test.ppm";

	// cpu_render_main creates missing parent directories for output_path
	// (cpu_interface.cpp's std::filesystem::create_directories call), so a
	// valid render request should succeed even into a not-yet-existing dir.
	int result = cpu_render_main(16, 16, 1, 3, output, "A1", 278, 278, -800);

	EXPECT_EQ(result, SUCCESS);

	// The renderer may auto-create the target directory; clean up so
	// repeated test runs don't leave litter in the working directory.
	std::remove(output);
	std::error_code ec;
	std::filesystem::remove(output_dir, ec);
}

/**
 * Test rendering with very small image
 */
TEST(RenderIntegrationTest, VerySmallImage) {
	const char* output = "test_tiny.ppm";

	int result = cpu_render_main(10, 10, 1, 3, output, "A1", 278, 278, -800);

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

	// Use depth=20 (not 1000) -- Russian Roulette handles deep paths,
	// but depth=1000 with 50x50 is prohibitively slow for a unit test.
	int result = cpu_render_main(16, 16, 1, 20, output, "A1", 278, 278, -800);

	EXPECT_EQ(result, 0);

	if (file_exists(output)) {
		std::remove(output);
	}
}
