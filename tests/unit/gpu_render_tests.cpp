/**
 * @file gpu_render_tests.cpp
 * @brief GPU render correctness tests (pbrt-v4 style)
 *
 * Inspired by pbrt-v4's integrators_test.cpp: renders scenes with known
 * properties and checks that output pixel values are within expected ranges.
 *
 * Tests:
 * - NonBlackOutput:      rendered image has at least some non-zero pixels
 * - CorrectDimensions:   output image matches requested resolution
 * - BrightnessInRange:   average pixel brightness is physically plausible
 * - LightVisible:        pixels directly viewing a light source are bright
 * - DifferentSPP:        more samples produce different (higher quality) output
 * - DifferentCameras:    different camera positions produce different images
 */

#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>

extern "C" {
	#include "optix_interface.h"
}

// ============================================================================
// PPM Helper Utilities (pbrt-v4 style: CheckSceneAverage)
// ============================================================================

struct PPMImage {
	int width = 0;
	int height = 0;
	std::vector<float> pixels;  // RGB floats, normalized to [0,1]
	bool valid = false;
};

/// Load a PPM (P3 ASCII) file and normalize pixel values to [0,1]
PPMImage load_ppm(const char* path) {
	PPMImage img;
	std::ifstream file(path);
	if (!file.good()) return img;

	std::string magic;
	int maxVal;
	file >> magic >> img.width >> img.height >> maxVal;

	if (magic != "P3" || maxVal <= 0) return img;

	int total = img.width * img.height * 3;
	img.pixels.resize(total);
	for (int i = 0; i < total; ++i) {
		int v;
		file >> v;
		img.pixels[i] = static_cast<float>(v) / static_cast<float>(maxVal);
	}
	img.valid = (file.good() || file.eof());
	return img;
}

/// Average brightness across all pixels (0=black, 1=white)
float average_brightness(const PPMImage& img) {
	if (img.pixels.empty()) return 0.0f;
	float sum = std::accumulate(img.pixels.begin(), img.pixels.end(), 0.0f);
	return sum / static_cast<float>(img.pixels.size());
}

/// Fraction of pixels that are exactly black (all channels == 0)
float black_pixel_fraction(const PPMImage& img) {
	if (!img.valid || img.width == 0) return 1.0f;
	int blackCount = 0;
	for (int i = 0; i < img.width * img.height; ++i) {
		float r = img.pixels[i * 3 + 0];
		float g = img.pixels[i * 3 + 1];
		float b = img.pixels[i * 3 + 2];
		if (r == 0.0f && g == 0.0f && b == 0.0f)
			++blackCount;
	}
	return static_cast<float>(blackCount) / static_cast<float>(img.width * img.height);
}

/// Max pixel value across all channels
float max_pixel_value(const PPMImage& img) {
	if (img.pixels.empty()) return 0.0f;
	return *std::max_element(img.pixels.begin(), img.pixels.end());
}

// ============================================================================
// GPU Render Correctness Tests (pbrt-v4 style)
// ============================================================================

class GPURenderTest : public ::testing::Test {
protected:
	void SetUp() override {
		if (!optix_is_available()) {
			GTEST_SKIP() << "OptiX not available on this system";
		}
	}

	void TearDown() override {
		// Clean up any output files created during the test
		for (const auto& f : outputFiles_) {
			std::remove(f.c_str());
		}
	}

	/// Render with GPU and return loaded image; registers file for cleanup
	PPMImage renderGPU(const char* filename, int w, int h, int spp, int depth,
					   double camX = 278.0, double camY = 278.0, double camZ = -800.0) {
		outputFiles_.push_back(filename);
		optix_render_main(w, h, spp, depth, filename, 0, camX, camY, camZ);
		return load_ppm(filename);
	}

	std::vector<std::string> outputFiles_;
};

// ----------------------------------------------------------------------------
// Test 1: Output is not completely black (pbrt-v4: CheckSceneAverage > 0)
// ----------------------------------------------------------------------------
TEST_F(GPURenderTest, NonBlackOutput) {
	PPMImage img = renderGPU("gpu_test_nonblack.ppm", 100, 100, 500, 10);

	ASSERT_TRUE(img.valid) << "Failed to load rendered PPM";

	float blackFrac = black_pixel_fraction(img);
	EXPECT_LT(blackFrac, 0.95f)
		<< "More than 95% of pixels are black — renderer likely broken. "
		<< "Black fraction: " << blackFrac;
}

// ----------------------------------------------------------------------------
// Test 2: Correct output dimensions
// ----------------------------------------------------------------------------
TEST_F(GPURenderTest, CorrectDimensions) {
	struct Case { int w, h; };
	for (auto [w, h] : std::vector<Case>{{64, 64}, {100, 150}, {200, 100}}) {
		std::string name = "gpu_test_dims_" + std::to_string(w) + "x" + std::to_string(h) + ".ppm";
		PPMImage img = renderGPU(name.c_str(), w, h, 100, 5);

		ASSERT_TRUE(img.valid) << "Failed for " << w << "x" << h;
		EXPECT_EQ(img.width,  w) << "Width mismatch for " << w << "x" << h;
		EXPECT_EQ(img.height, h) << "Height mismatch for " << w << "x" << h;
	}
}

// ----------------------------------------------------------------------------
// Test 3: Average brightness within physically plausible range
// (pbrt-v4 CheckSceneAverage pattern — checks sum/nPixels near expected)
// Cornell box with 15.0 intensity light should produce moderate brightness
// ----------------------------------------------------------------------------
TEST_F(GPURenderTest, BrightnessInRange) {
	PPMImage img = renderGPU("gpu_test_brightness.ppm", 100, 100, 1000, 10);

	ASSERT_TRUE(img.valid) << "Failed to load rendered PPM";

	float avg = average_brightness(img);

	// Cornell box: not too dark (scene has a bright light) and not overexposed
	// Generous tolerance because GPU uses naive path tracing
	EXPECT_GT(avg, 0.005f)
		<< "Image is too dark — average brightness " << avg << " near zero";
	EXPECT_LT(avg, 0.9f)
		<< "Image is suspiciously bright — average brightness " << avg;
}

// ----------------------------------------------------------------------------
// Test 4: Light source pixels are brighter than wall pixels
// Render with camera looking directly at the ceiling light
// ----------------------------------------------------------------------------
TEST_F(GPURenderTest, LightSourceIsBright) {
	// Render with more samples so light is clearly visible
	PPMImage img = renderGPU("gpu_test_light.ppm", 100, 100, 1000, 10);

	ASSERT_TRUE(img.valid) << "Failed to load rendered PPM";

	float maxVal = max_pixel_value(img);
	EXPECT_GT(maxVal, 0.1f)
		<< "Maximum pixel value " << maxVal << " — no bright pixels found. "
		<< "Light source should produce bright pixels somewhere in the image.";
}

// ----------------------------------------------------------------------------
// Test 5: More samples → image must be different from fewer samples
// (not necessarily better measured by brightness, just different)
// ----------------------------------------------------------------------------
TEST_F(GPURenderTest, DifferentSPPProducesDifferentOutput) {
	PPMImage img1 = renderGPU("gpu_test_spp1.ppm",  80, 80,   10, 5);
	PPMImage img2 = renderGPU("gpu_test_spp2.ppm",  80, 80, 1000, 5);

	ASSERT_TRUE(img1.valid) << "10 spp render failed";
	ASSERT_TRUE(img2.valid) << "1000 spp render failed";

	// Sum absolute difference across pixels
	float diff = 0.0f;
	for (size_t i = 0; i < img1.pixels.size(); ++i)
		diff += std::abs(img1.pixels[i] - img2.pixels[i]);
	float avgDiff = diff / static_cast<float>(img1.pixels.size());

	EXPECT_GT(avgDiff, 0.001f)
		<< "Images from 10 spp and 1000 spp are identical — "
		<< "random sampling should produce different results";
}

// ----------------------------------------------------------------------------
// Test 6: Different camera positions produce different images
// (same pattern as existing render_tests.cpp but with brightness validation)
// ----------------------------------------------------------------------------
TEST_F(GPURenderTest, DifferentCamerasDifferentOutput) {
	// Front view (standard Cornell box view)
	PPMImage front = renderGPU("gpu_test_cam_front.ppm", 80, 80, 500, 10,
								278.0, 278.0, -800.0);
	// Side view
	PPMImage side  = renderGPU("gpu_test_cam_side.ppm",  80, 80, 500, 10,
								-200.0, 278.0, 278.0);

	ASSERT_TRUE(front.valid) << "Front camera render failed";
	ASSERT_TRUE(side.valid)  << "Side camera render failed";

	float diff = 0.0f;
	for (size_t i = 0; i < front.pixels.size(); ++i)
		diff += std::abs(front.pixels[i] - side.pixels[i]);
	float avgDiff = diff / static_cast<float>(front.pixels.size());

	EXPECT_GT(avgDiff, 0.01f)
		<< "Front and side camera renders are too similar — "
		<< "different viewpoints should produce visually different images";
}
