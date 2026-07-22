/**
 * @file cpu_gpu_comparison_tests.cpp
 * @brief CPU vs GPU renderer comparison tests (pbrt-v4 style)
 *
 * Inspired by pbrt-v4's multi-integrator test pattern where the same scene
 * is rendered by different integrators and results are compared.
 *
 * Key insight: CPU (importance sampling) and GPU (naive path tracing) should:
 * - Both produce non-black output for the same scene
 * - Converge toward similar average brightness at sufficient sample counts
 * - Both correctly represent the Cornell box color palette (warm/cool tones)
 *
 * Tests:
 * - BothProduceNonBlackOutput:       neither renderer is broken
 * - BothHaveSimilarBrightness:       converge to same integral value
 * - CPUIsBrighterAtLowSPP:           CPU importance sampling is more efficient
 * - BothShowColorVariation:          red/green walls produce color in image
 * - HighSPPConvergence:              both approach same answer at high samples
 */

#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

extern "C" {
	#include "cpu_interface.h"
	#include "optix_interface.h"
}

// ============================================================================
// Shared utilities
// ============================================================================

struct Image {
	int width = 0, height = 0;
	std::vector<float> pixels;  // RGB normalized [0,1]
	bool valid = false;
};

Image load_image(const char* path) {
	Image img;
	std::ifstream f(path);
	if (!f.good()) return img;
	std::string magic;
	int maxVal;
	f >> magic >> img.width >> img.height >> maxVal;
	if (magic != "P3" || maxVal <= 0) return img;
	int total = img.width * img.height * 3;
	img.pixels.resize(total);
	for (int i = 0; i < total; ++i) {
		int v; f >> v;
		img.pixels[i] = static_cast<float>(v) / static_cast<float>(maxVal);
	}
	img.valid = true;
	return img;
}

float avg_brightness(const Image& img) {
	if (img.pixels.empty()) return 0.0f;
	float s = std::accumulate(img.pixels.begin(), img.pixels.end(), 0.0f);
	return s / static_cast<float>(img.pixels.size());
}

// Average red, green, blue channels separately
struct RGBAverage { float r, g, b; };
RGBAverage avg_channels(const Image& img) {
	RGBAverage out{0, 0, 0};
	int n = img.width * img.height;
	if (n == 0) return out;
	for (int i = 0; i < n; ++i) {
		out.r += img.pixels[i * 3 + 0];
		out.g += img.pixels[i * 3 + 1];
		out.b += img.pixels[i * 3 + 2];
	}
	out.r /= n; out.g /= n; out.b /= n;
	return out;
}

// ============================================================================
// CPU vs GPU Comparison Tests
// ============================================================================

class CPUGPUComparisonTest : public ::testing::Test {
protected:
	void SetUp() override {
		gpuAvailable_ = optix_is_available();
	}

	void TearDown() override {
		for (const auto& f : files_) std::remove(f.c_str());
	}

	Image renderCPU(const char* filename, int w, int h, int spp, int depth) {
		files_.push_back(filename);
		cpu_render_main(w, h, spp, depth, filename, 0, 278.0, 278.0, -800.0);
		return load_image(filename);
	}

	Image renderGPU(const char* filename, int w, int h, int spp, int depth) {
		files_.push_back(filename);
		optix_render_main(w, h, spp, depth, filename, 0, 278.0, 278.0, -800.0);
		return load_image(filename);
	}

	bool gpuAvailable_ = false;
	std::vector<std::string> files_;
};

// ----------------------------------------------------------------------------
// Test 1: Both renderers produce non-black output on the same scene
// (pbrt-v4 equivalent: all integrators produce non-zero output)
// ----------------------------------------------------------------------------
TEST_F(CPUGPUComparisonTest, BothProduceNonBlackOutput) {
	Image cpu = renderCPU("cmp_cpu_nonblack.ppm", 80, 80, 20, 5);
	ASSERT_TRUE(cpu.valid) << "CPU render failed";

	float cpuBrightness = avg_brightness(cpu);
	EXPECT_GT(cpuBrightness, 0.001f)
		<< "CPU output is black. Average brightness: " << cpuBrightness;

	if (!gpuAvailable_) {
		GTEST_SKIP() << "OptiX not available — GPU comparison skipped";
	}

	Image gpu = renderGPU("cmp_gpu_nonblack.ppm", 80, 80, 500, 5);
	ASSERT_TRUE(gpu.valid) << "GPU render failed";

	float gpuBrightness = avg_brightness(gpu);
	EXPECT_GT(gpuBrightness, 0.001f)
		<< "GPU output is black. Average brightness: " << gpuBrightness;
}

// ----------------------------------------------------------------------------
// Test 2: At high sample counts, CPU and GPU should have similar brightness
// (pbrt-v4 pattern: multiple integrators converge to same value)
// Allow 50% relative difference since GPU uses naive vs CPU importance sampling
// ----------------------------------------------------------------------------
TEST_F(CPUGPUComparisonTest, HighSPPBrightnessConverges) {
	if (!gpuAvailable_) GTEST_SKIP() << "OptiX not available";

	// Use sufficient samples for both to have meaningful statistics
	Image cpu = renderCPU("cmp_cpu_converge.ppm", 80, 80, 100,  10);
	Image gpu = renderGPU("cmp_gpu_converge.ppm", 80, 80, 2000, 10);

	ASSERT_TRUE(cpu.valid) << "CPU render failed";
	ASSERT_TRUE(gpu.valid) << "GPU render failed";

	float cpuBright = avg_brightness(cpu);
	float gpuBright = avg_brightness(gpu);

	// Both must be non-zero
	EXPECT_GT(cpuBright, 0.001f) << "CPU result is black";
	EXPECT_GT(gpuBright, 0.001f) << "GPU result is black";

	// Relative difference should be < 50%
	// (generous because GPU is naive path tracing vs CPU importance sampling)
	float maxBright = std::max(cpuBright, gpuBright);
	float relDiff = std::abs(cpuBright - gpuBright) / maxBright;

	EXPECT_LT(relDiff, 0.50f)
		<< "CPU brightness=" << cpuBright << " GPU brightness=" << gpuBright
		<< " relative difference=" << relDiff
		<< " — renderers disagree by more than 50%. "
		<< "They should converge to the same scene integral.";
}

// ----------------------------------------------------------------------------
// Test 3: CPU importance sampling should be brighter than GPU naive at same SPP
// (This validates that MIS/NEE is actually helping efficiency)
// ----------------------------------------------------------------------------
TEST_F(CPUGPUComparisonTest, CPUIsBrighterAtLowSPP) {
	if (!gpuAvailable_) GTEST_SKIP() << "OptiX not available";

	// Low SPP where efficiency difference is most visible
	Image cpu = renderCPU("cmp_cpu_lowspp.ppm", 80, 80, 10, 5);
	Image gpu = renderGPU("cmp_gpu_lowspp.ppm", 80, 80, 10, 5);

	ASSERT_TRUE(cpu.valid) << "CPU render failed";
	ASSERT_TRUE(gpu.valid) << "GPU render failed";

	float cpuBright = avg_brightness(cpu);
	float gpuBright = avg_brightness(gpu);

	// CPU with importance sampling should produce a brighter result at low SPP
	// because it finds the light more efficiently
	EXPECT_GT(cpuBright, gpuBright)
		<< "CPU brightness=" << cpuBright << " GPU brightness=" << gpuBright
		<< " — CPU importance sampling should outperform GPU naive tracing at low spp. "
		<< "If GPU is brighter, something may be wrong with CPU sampling.";
}

// ----------------------------------------------------------------------------
// Test 4: Cornell box should show color variation (red left, green right wall)
// Both renderers should produce images where one channel dominates in corners
// ----------------------------------------------------------------------------
TEST_F(CPUGPUComparisonTest, CPUShowsColorVariation) {
	Image img = renderCPU("cmp_cpu_color.ppm", 100, 100, 50, 10);
	ASSERT_TRUE(img.valid) << "CPU render failed";

	RGBAverage avg = avg_channels(img);
	float total = avg.r + avg.g + avg.b;

	if (total < 0.001f) GTEST_SKIP() << "Image too dark to analyze color";

	// Cornell box has red left wall and green right wall
	// The image should have meaningful content in both R and G channels
	float rFrac = avg.r / total;
	float gFrac = avg.g / total;

	// Neither channel should be totally absent (would mean we're missing a wall)
	EXPECT_GT(rFrac, 0.05f) << "Red channel fraction " << rFrac << " too low — red wall missing?";
	EXPECT_GT(gFrac, 0.05f) << "Green channel fraction " << gFrac << " too low — green wall missing?";
}

TEST_F(CPUGPUComparisonTest, GPUShowsColorVariation) {
	if (!gpuAvailable_) GTEST_SKIP() << "OptiX not available";

	Image img = renderGPU("cmp_gpu_color.ppm", 100, 100, 1000, 10);
	ASSERT_TRUE(img.valid) << "GPU render failed";

	RGBAverage avg = avg_channels(img);
	float total = avg.r + avg.g + avg.b;

	if (total < 0.001f) GTEST_SKIP() << "Image too dark to analyze color";

	float rFrac = avg.r / total;
	float gFrac = avg.g / total;

	EXPECT_GT(rFrac, 0.05f) << "GPU red channel fraction " << rFrac << " too low — red wall missing?";
	EXPECT_GT(gFrac, 0.05f) << "GPU green channel fraction " << gFrac << " too low — green wall missing?";
}

// ----------------------------------------------------------------------------
// Test 5: Scene ID 0 (Cornell box) must render the same for both renderers
// at matching resolutions — output files must exist and have same dimensions
// ----------------------------------------------------------------------------
TEST_F(CPUGPUComparisonTest, BothProduceSameDimensions) {
	if (!gpuAvailable_) GTEST_SKIP() << "OptiX not available";

	struct Case { int w, h; };
	for (auto [w, h] : std::vector<Case>{{64, 64}, {128, 128}}) {
		std::string cpuFile = "cmp_dims_cpu_" + std::to_string(w) + ".ppm";
		std::string gpuFile = "cmp_dims_gpu_" + std::to_string(w) + ".ppm";

		Image cpu = renderCPU(cpuFile.c_str(), w, h, 5, 3);
		Image gpu = renderGPU(gpuFile.c_str(), w, h, 100, 3);

		ASSERT_TRUE(cpu.valid) << "CPU render failed at " << w << "x" << h;
		ASSERT_TRUE(gpu.valid) << "GPU render failed at " << w << "x" << h;

		EXPECT_EQ(cpu.width,  gpu.width)  << "Width mismatch at " << w << "x" << h;
		EXPECT_EQ(cpu.height, gpu.height) << "Height mismatch at " << w << "x" << h;
		EXPECT_EQ(cpu.width,  w) << "CPU width incorrect";
		EXPECT_EQ(gpu.width,  w) << "GPU width incorrect";
	}
}
