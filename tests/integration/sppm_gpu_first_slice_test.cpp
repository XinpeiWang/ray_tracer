// sppm_gpu_first_slice_test.cpp
// First real end-to-end verification of the GPU SPPM port (sub-phase 1f,
// see C:\Users\xinpe\.claude\plans\cached-wobbling-ritchie.md), mirroring
// tests/integration/sppm_first_slice_test.cpp's own CPU version and
// tests/unit/gpu_render_tests.cpp's pattern of driving the renderer through
// its real CLI-level C entry point (optix_render_main_sppm()) rather than
// poking OptiXRenderer directly -- that entry point already owns OptiX
// context init/scene upload/PTX loading, exactly what a real --sppm --gpu
// invocation goes through.
//
// Small iteration/photon counts throughout: this is about correctness (does
// a real GPU SPPM render produce finite, non-negative, non-black output and
// respect Phase 1's scene-11-only scope), not visual convergence quality.
#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
	#include "optix_interface.h"
}

namespace {

struct PPMImage {
	int width = 0;
	int height = 0;
	std::vector<int> pixels;  // RGB ints, 0-255
	bool valid = false;
};

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
	for (int i = 0; i < total; ++i) file >> img.pixels[i];
	img.valid = (file.good() || file.eof());
	return img;
}

class SppmGpuFirstSliceTest : public ::testing::Test {
  protected:
	void SetUp() override {
		if (!optix_is_available()) {
			GTEST_SKIP() << "OptiX not available on this system";
		}
	}
	void TearDown() override {
		for (const auto& f : outputFiles_) std::remove(f.c_str());
	}
	std::vector<std::string> outputFiles_;
};

} // namespace

TEST_F(SppmGpuFirstSliceTest, CornellRoughGlassProducesFiniteNonBlackImage) {
	const char* path = "sppm_gpu_first_slice_test.ppm";
	outputFiles_.push_back(path);

	int result = optix_render_main_sppm(
		/*image_width=*/48, /*image_height=*/48,
		/*iterations=*/10, /*photons=*/2000, /*max_depth=*/5,
		path, /*scene_id=*/11,
		278.0, 278.0, -800.0, /*force_camera_override=*/1);
	ASSERT_EQ(result, 0) << "optix_render_main_sppm failed on scene 11";

	PPMImage img = load_ppm(path);
	ASSERT_TRUE(img.valid) << "Failed to load rendered PPM";
	ASSERT_EQ(img.width, 48);
	ASSERT_EQ(img.height, 48);

	// PPM values are already byte-clamped/tonemapped ints (0-255); the C
	// entry point itself already guards NaN/Inf before writing (see
	// optix_render_main_sppm's own NaN/Inf guard, matching write_color()),
	// so what's left to check here is that every value survived that guard
	// in-range and that the image isn't degenerate (all black).
	int nonzero_count = 0;
	for (int v : img.pixels) {
		ASSERT_GE(v, 0);
		ASSERT_LE(v, 255);
		if (v > 0) ++nonzero_count;
	}
	EXPECT_GT(nonzero_count, 0) << "Rendered image is entirely black -- SPPM camera/photon pass "
	                            << "likely produced no direct or indirect lighting contribution";
}

// Phase 1 scope guard: any scene other than 11 must be rejected cleanly
// (not crash, not silently render garbage) -- see optix_render_main_sppm's
// own doc comment on why scene 11 is the only supported scene right now.
TEST_F(SppmGpuFirstSliceTest, RejectsOutOfScopeScene) {
	const char* path = "sppm_gpu_first_slice_test_scene0.ppm";
	outputFiles_.push_back(path);

	int result = optix_render_main_sppm(
		32, 32, /*iterations=*/5, /*photons=*/500, /*max_depth=*/5,
		path, /*scene_id=*/0,
		278.0, 278.0, -800.0, 1);
	EXPECT_NE(result, 0) << "GPU SPPM should reject scene 0 (Phase 1 supports scene 11 only)";
}

// Regression guard: the real multi-iteration SPPM path (renderSPPM(),
// reached via optix_render_main_sppm()) must be fully additive -- it must
// not disturb the regular GPU path tracer's own render on the same scene in
// the same process (both share the same g_renderer singleton and uploaded-
// scene cache in optix_interface.cpp).
TEST_F(SppmGpuFirstSliceTest, PlainGpuRenderStillWorksAfterSppmRender) {
	const char* sppmPath = "sppm_gpu_first_slice_test_pre.ppm";
	const char* plainPath = "sppm_gpu_first_slice_test_plain.ppm";
	outputFiles_.push_back(sppmPath);
	outputFiles_.push_back(plainPath);

	ASSERT_EQ(optix_render_main_sppm(48, 48, 10, 2000, 5, sppmPath, 11,
	                                  278.0, 278.0, -800.0, 1), 0);

	int plainResult = optix_render_main(48, 48, /*samples_per_pixel=*/50, /*max_depth=*/5,
	                                     plainPath, /*scene_id=*/11,
	                                     278.0, 278.0, -800.0, 1);
	ASSERT_EQ(plainResult, 0) << "Plain GPU render failed after an SPPM render in the same process";

	PPMImage img = load_ppm(plainPath);
	ASSERT_TRUE(img.valid);
	int nonzero_count = 0;
	for (int v : img.pixels) if (v > 0) ++nonzero_count;
	EXPECT_GT(nonzero_count, 0) << "Plain GPU render is all-black after an SPPM render -- possible "
	                            << "state leak between SPPMPathTracer and the regular path tracer";
}
