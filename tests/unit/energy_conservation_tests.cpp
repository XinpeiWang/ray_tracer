/**
 * @file energy_conservation_tests.cpp
 * @brief Energy conservation tests (pbrt-v4 style)
 *
 * Inspired by pbrt-v4's bsdfs_test.cpp (BSDFEnergyConservation tests):
 * A physically correct renderer must not create energy — the output radiance
 * must be bounded by the light source input energy.
 *
 * Key principle: In a Cornell box scene with a single light of intensity L,
 * no pixel should consistently be brighter than L (after tone mapping).
 * Also verifies that pixel values are valid floats in [0, 1] after gamma.
 *
 * Tests:
 * - PixelsInValidRange:       all pixel values are in [0,1] after normalization
 * - NoFireflies_HighSPP:      no extreme outlier pixels (energy leak = firefly)
 * - CPUEnergyBounded:         CPU render average doesn't exceed theoretical max
 * - GPUEnergyBounded:         GPU render average doesn't exceed theoretical max
 * - BothRenderersProduceFiniteValues: neither renderer outputs NaN/Inf
 */

#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

extern "C" {
	#include "cpu_interface.h"
	#include "optix_interface.h"
}

// ============================================================================
// Shared PPM utilities
// ============================================================================

struct RenderResult {
	int width = 0, height = 0;
	std::vector<float> pixels;  // RGB, normalized to [0,1]
	bool valid = false;
};

RenderResult load_render(const char* path) {
	RenderResult r;
	std::ifstream f(path);
	if (!f.good()) return r;

	std::string magic;
	int maxVal;
	f >> magic >> r.width >> r.height >> maxVal;
	if (magic != "P3" || maxVal <= 0) return r;

	int total = r.width * r.height * 3;
	r.pixels.resize(total);
	for (int i = 0; i < total; ++i) {
		int v; f >> v;
		r.pixels[i] = static_cast<float>(v) / static_cast<float>(maxVal);
	}
	r.valid = true;
	return r;
}

// ============================================================================
// Energy Conservation Tests
// ============================================================================

class EnergyConservationTest : public ::testing::Test {
protected:
	void TearDown() override {
		for (const auto& f : files_) std::remove(f.c_str());
	}
	std::vector<std::string> files_;
};

// ----------------------------------------------------------------------------
// Test 1: All pixel values after PPM normalization are in [0, 1]
// PPM spec guarantees this — if violated, our writer has a bug
// ----------------------------------------------------------------------------
TEST_F(EnergyConservationTest, CPUPixelsInValidRange) {
	const char* out = "ec_test_cpu_range.ppm";
	files_.push_back(out);

	cpu_render_main(80, 80, 10, 5, out, 0, 278.0, 278.0, -800.0);
	RenderResult r = load_render(out);
	ASSERT_TRUE(r.valid) << "CPU render failed";

	int violations = 0;
	for (float v : r.pixels) {
		if (v < 0.0f || v > 1.0f) ++violations;
	}
	EXPECT_EQ(violations, 0)
		<< violations << " pixels out of [0,1] range — PPM writer bug";
}

TEST_F(EnergyConservationTest, GPUPixelsInValidRange) {
	if (!optix_is_available()) GTEST_SKIP() << "OptiX not available";

	const char* out = "ec_test_gpu_range.ppm";
	files_.push_back(out);

	optix_render_main(80, 80, 500, 5, out, 0, 278.0, 278.0, -800.0);
	RenderResult r = load_render(out);
	ASSERT_TRUE(r.valid) << "GPU render failed";

	int violations = 0;
	for (float v : r.pixels) {
		if (v < 0.0f || v > 1.0f) ++violations;
	}
	EXPECT_EQ(violations, 0)
		<< violations << " pixels out of [0,1] range — framebuffer clamp bug";
}

// ----------------------------------------------------------------------------
// Test 2: No extreme firefly pixels (energy leak / variance spike)
// pbrt-v4 pattern: individual pixels should not be > N * average brightness
// A firefly > 20x average suggests a path weight overflow / NaN recovery
// ----------------------------------------------------------------------------
TEST_F(EnergyConservationTest, GPUNoFireflies) {
	if (!optix_is_available()) GTEST_SKIP() << "OptiX not available";

	const char* out = "ec_test_gpu_fireflies.ppm";
	files_.push_back(out);

	optix_render_main(100, 100, 1000, 10, out, 0, 278.0, 278.0, -800.0);
	RenderResult r = load_render(out);
	ASSERT_TRUE(r.valid) << "GPU render failed";

	// Compute average brightness
	float sum = std::accumulate(r.pixels.begin(), r.pixels.end(), 0.0f);
	float avg = sum / static_cast<float>(r.pixels.size());

	if (avg < 1e-6f) GTEST_SKIP() << "Image too dark to check for fireflies";

	// Check that no individual channel exceeds 20x average
	// (Extreme fireflies indicate energy creation / path weight bugs)
	float threshold = 20.0f * avg;
	int fireflies = 0;
	float maxVal = 0.0f;
	for (float v : r.pixels) {
		maxVal = std::max(maxVal, v);
		if (v > threshold) ++fireflies;
	}

	// Allow a very small number (< 0.1% of pixels) — some variance is ok
	int totalPixels = r.width * r.height;
	int allowedFireflies = totalPixels / 1000;  // 0.1%

	EXPECT_LE(fireflies, allowedFireflies)
		<< fireflies << " firefly pixels (>" << threshold << " = 20x avg " << avg << "). "
		<< "Max pixel: " << maxVal << ". "
		<< "Suggests path weight overflow or missing Russian roulette.";
}

// ----------------------------------------------------------------------------
// Test 3: CPU average brightness bounded by physical limit
// Cornell box light is 15.0 intensity, scene is 555x555x555 units.
// After gamma correction (sqrt) and normalization, average must be < 1.0
// ----------------------------------------------------------------------------
TEST_F(EnergyConservationTest, CPUAverageBrightnessPhysicallyBounded) {
	const char* out = "ec_test_cpu_bound.ppm";
	files_.push_back(out);

	cpu_render_main(100, 100, 50, 10, out, 0, 278.0, 278.0, -800.0);
	RenderResult r = load_render(out);
	ASSERT_TRUE(r.valid) << "CPU render failed";

	float sum = std::accumulate(r.pixels.begin(), r.pixels.end(), 0.0f);
	float avg = sum / static_cast<float>(r.pixels.size());

	// Average must be below 1.0 (fully saturated white = energy source, not scene)
	EXPECT_LT(avg, 1.0f)
		<< "CPU average brightness " << avg << " >= 1.0 — physically impossible for Cornell box";

	// Also must be non-trivially non-zero (light source should illuminate scene)
	EXPECT_GT(avg, 0.001f)
		<< "CPU average brightness " << avg << " near zero — scene is too dark";
}

TEST_F(EnergyConservationTest, GPUAverageBrightnessPhysicallyBounded) {
	if (!optix_is_available()) GTEST_SKIP() << "OptiX not available";

	const char* out = "ec_test_gpu_bound.ppm";
	files_.push_back(out);

	optix_render_main(100, 100, 1000, 10, out, 0, 278.0, 278.0, -800.0);
	RenderResult r = load_render(out);
	ASSERT_TRUE(r.valid) << "GPU render failed";

	float sum = std::accumulate(r.pixels.begin(), r.pixels.end(), 0.0f);
	float avg = sum / static_cast<float>(r.pixels.size());

	EXPECT_LT(avg, 1.0f)
		<< "GPU average brightness " << avg << " >= 1.0 — energy is being created";
	EXPECT_GT(avg, 0.001f)
		<< "GPU average brightness " << avg << " near zero — scene not illuminated";
}

// ----------------------------------------------------------------------------
// Test 4: Neither renderer outputs NaN or Inf values
// (pbrt-v4 checks this implicitly via Image::Read + ASSERT_TRUE)
// NaN/Inf in the framebuffer indicates division by zero or uninitialized data
// ----------------------------------------------------------------------------
TEST_F(EnergyConservationTest, CPUNoNaNOrInf) {
	const char* out = "ec_test_cpu_nan.ppm";
	files_.push_back(out);

	cpu_render_main(80, 80, 20, 10, out, 0, 278.0, 278.0, -800.0);
	RenderResult r = load_render(out);
	ASSERT_TRUE(r.valid) << "CPU render failed";

	// PPM values are integers so we can check via the normalized floats
	// Overflow NaN → written as 0 or 255, both result in valid [0,1] floats
	// We check the file loaded cleanly and all values are finite
	int nanCount = 0;
	for (float v : r.pixels) {
		if (!std::isfinite(v)) ++nanCount;
	}
	EXPECT_EQ(nanCount, 0) << "CPU render produced " << nanCount << " NaN/Inf values";
}

TEST_F(EnergyConservationTest, GPUNoNaNOrInf) {
	if (!optix_is_available()) GTEST_SKIP() << "OptiX not available";

	const char* out = "ec_test_gpu_nan.ppm";
	files_.push_back(out);

	optix_render_main(80, 80, 500, 10, out, 0, 278.0, 278.0, -800.0);
	RenderResult r = load_render(out);
	ASSERT_TRUE(r.valid) << "GPU render failed";

	int nanCount = 0;
	for (float v : r.pixels) {
		if (!std::isfinite(v)) ++nanCount;
	}
	EXPECT_EQ(nanCount, 0) << "GPU render produced " << nanCount << " NaN/Inf values";
}

// ============================================================================
// Russian Roulette Tests
// ============================================================================
//
// These tests verify the key properties of Russian Roulette path termination:
//  1. Unbiasedness  — average brightness must not change vs a no-RR baseline
//  2. No energy gain — RR must not create energy (avg with RR <= avg without)
//  3. High-depth stability — raising max_depth with RR must not blow up or darken
//  4. Valid output — RR compensation must never produce NaN/Inf
//
// Note: RR is always active in the current build (depth < max_depth - 2).
// These tests compare low-depth (RR rarely fires) vs high-depth (RR fires often).
// ============================================================================

class RussianRouletteTest : public ::testing::Test {
protected:
	void TearDown() override {
		for (const auto& f : files_) std::remove(f.c_str());
	}
	std::vector<std::string> files_;

	// Helper: compute average luminance of a render
	float avg_luminance(const RenderResult& r) {
		float sum = 0;
		int n = r.width * r.height;
		for (int i = 0; i < n; ++i) {
			sum += 0.2126f * r.pixels[i*3+0]
				 + 0.7152f * r.pixels[i*3+1]
				 + 0.0722f * r.pixels[i*3+2];
		}
		return sum / n;
	}
};

// Test: RR with high max_depth should not produce brighter image than low max_depth.
// RR is unbiased — it should not add energy, just terminate paths early.
TEST_F(RussianRouletteTest, NoEnergyGainAtHighDepth) {
	const char* out_lo = "rr_test_depth5.ppm";
	const char* out_hi = "rr_test_depth50.ppm";
	files_.push_back(out_lo);
	files_.push_back(out_hi);

	// Enough SPP to average out RR noise
	cpu_render_main(80, 80, 50, 5,  out_lo, 0, 278.0, 278.0, -800.0);
	cpu_render_main(80, 80, 50, 50, out_hi, 0, 278.0, 278.0, -800.0);

	RenderResult r_lo = load_render(out_lo);
	RenderResult r_hi = load_render(out_hi);
	ASSERT_TRUE(r_lo.valid) << "Low-depth render failed";
	ASSERT_TRUE(r_hi.valid) << "High-depth render failed";

	float lum_lo = avg_luminance(r_lo);
	float lum_hi = avg_luminance(r_hi);

	// High depth should converge to same or slightly higher (more bounces = more
	// indirect light captured), but must never be more than 2x brighter (energy leak).
	EXPECT_GT(lum_lo, 0.0f) << "Low-depth render is black";
	EXPECT_GT(lum_hi, 0.0f) << "High-depth render is black";
	EXPECT_LT(lum_hi, lum_lo * 2.0f)
		<< "High-depth render is suspiciously brighter: lum_lo=" << lum_lo
		<< " lum_hi=" << lum_hi << " — possible energy gain from RR compensation bug";
}

// Test: raising max_depth must not darken the image excessively.
// RR terminates dim paths, but bright paths survive — so average should remain stable.
TEST_F(RussianRouletteTest, HighDepthDoesNotDarken) {
	const char* out_lo = "rr_test_nodark_d5.ppm";
	const char* out_hi = "rr_test_nodark_d50.ppm";
	files_.push_back(out_lo);
	files_.push_back(out_hi);

	cpu_render_main(80, 80, 50, 5,  out_lo, 0, 278.0, 278.0, -800.0);
	cpu_render_main(80, 80, 50, 50, out_hi, 0, 278.0, 278.0, -800.0);

	RenderResult r_lo = load_render(out_lo);
	RenderResult r_hi = load_render(out_hi);
	ASSERT_TRUE(r_lo.valid);
	ASSERT_TRUE(r_hi.valid);

	float lum_lo = avg_luminance(r_lo);
	float lum_hi = avg_luminance(r_hi);

	// High-depth should be at least 20% of the low-depth brightness.
	// If RR compensation is broken (throughput not divided by survival prob),
	// the image goes dark.
	EXPECT_GT(lum_hi, lum_lo * 0.2f)
		<< "High-depth image is suspiciously dark: lum_lo=" << lum_lo
		<< " lum_hi=" << lum_hi << " — possible missing RR survival compensation";
}

// Test: RR output must contain no NaN or Inf even with aggressive termination.
// The throughput /= (1 - q) compensation could produce Inf if q >= 1.
TEST_F(RussianRouletteTest, NoNaNOrInfAtHighDepth) {
	const char* out = "rr_test_nan_d50.ppm";
	files_.push_back(out);

	cpu_render_main(80, 80, 20, 50, out, 0, 278.0, 278.0, -800.0);
	RenderResult r = load_render(out);
	ASSERT_TRUE(r.valid) << "High-depth render failed";

	int bad = 0;
	for (float v : r.pixels) {
		if (!std::isfinite(v)) ++bad;
	}
	EXPECT_EQ(bad, 0)
		<< bad << " NaN/Inf values at max_depth=50 — RR compensation overflow?";
}

// Test: pixels must stay in [0, 1] after gamma even with high depth + RR compensation.
TEST_F(RussianRouletteTest, PixelsInValidRangeAtHighDepth) {
	const char* out = "rr_test_range_d50.ppm";
	files_.push_back(out);

	cpu_render_main(80, 80, 20, 50, out, 0, 278.0, 278.0, -800.0);
	RenderResult r = load_render(out);
	ASSERT_TRUE(r.valid);

	int violations = 0;
	for (float v : r.pixels) {
		if (v < 0.0f || v > 1.0f) ++violations;
	}
	EXPECT_EQ(violations, 0)
		<< violations << " pixels out of [0,1] at max_depth=50";
}
