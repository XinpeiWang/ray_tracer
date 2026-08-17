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
// a real GPU SPPM render produce finite, non-negative, non-black output),
// not visual convergence quality.
//
// Scope note: this file originally only exercised B3 (CornellRoughGlass) --
// Phase 1's one hardcoded scene -- plus a "GPU SPPM rejects any other scene"
// guard. A later generalization pass (see optix_interface.cpp's
// sppm_gpu_unsupported_reason() and sppm_programs.cu's
// sppm_is_delta_material()/sppm_sample_delta_material()) extended GPU SPPM's
// material dispatch to also cover Lambertian+Metal+Dielectric+Conductor
// scenes built purely from spheres/quads with area lights, so this file now
// also renders A1 (CornellBox, smooth Dielectric glass) and B4
// (CornellConductor, GGX conductor) end-to-end, and the old "reject anything
// but B3" test was replaced with one that rejects a scene using a
// MaterialType genuinely still unsupported (B5 CornellCoatedDiffuse) --
// see that test's own comment.
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
		path, /*scene_id=*/"B3",
		278.0, 278.0, -800.0, /*force_camera_override=*/1);
	ASSERT_EQ(result, 0) << "optix_render_main_sppm failed on scene B3";

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

// A1 (CornellBox): Lambertian walls/box + a smooth Dielectric glass sphere +
// a DiffuseLight ceiling quad -- exercises the Dielectric branch added to
// sppm_is_delta_material()/sppm_sample_delta_material() (sppm_programs.cu)
// by the generalization pass documented at this file's own top comment.
// Previously rejected outright by Phase 1's hardcoded "only B3" guard.
TEST_F(SppmGpuFirstSliceTest, CornellBoxDielectricProducesFiniteNonBlackImage) {
	const char* path = "sppm_gpu_first_slice_test_a1.ppm";
	outputFiles_.push_back(path);

	int result = optix_render_main_sppm(
		/*image_width=*/48, /*image_height=*/48,
		/*iterations=*/10, /*photons=*/2000, /*max_depth=*/5,
		path, /*scene_id=*/"A1",
		278.0, 278.0, -800.0, /*force_camera_override=*/1);
	ASSERT_EQ(result, 0) << "optix_render_main_sppm failed on scene A1";

	PPMImage img = load_ppm(path);
	ASSERT_TRUE(img.valid) << "Failed to load rendered PPM";
	ASSERT_EQ(img.width, 48);
	ASSERT_EQ(img.height, 48);

	int nonzero_count = 0;
	for (int v : img.pixels) {
		ASSERT_GE(v, 0);
		ASSERT_LE(v, 255);
		if (v > 0) ++nonzero_count;
	}
	EXPECT_GT(nonzero_count, 0) << "Rendered image is entirely black";
}

// B4 (CornellConductor): Cornell walls + a polished-gold sphere and a
// polished-aluminium box, both MaterialType::Conductor (GGX VNDF + complex
// Fresnel) -- exercises the Conductor branch added by the same
// generalization pass. Previously rejected outright by Phase 1's hardcoded
// "only B3" guard.
TEST_F(SppmGpuFirstSliceTest, CornellConductorProducesFiniteNonBlackImage) {
	const char* path = "sppm_gpu_first_slice_test_b4.ppm";
	outputFiles_.push_back(path);

	int result = optix_render_main_sppm(
		/*image_width=*/48, /*image_height=*/48,
		/*iterations=*/10, /*photons=*/2000, /*max_depth=*/5,
		path, /*scene_id=*/"B4",
		278.0, 278.0, -800.0, /*force_camera_override=*/1);
	ASSERT_EQ(result, 0) << "optix_render_main_sppm failed on scene B4";

	PPMImage img = load_ppm(path);
	ASSERT_TRUE(img.valid) << "Failed to load rendered PPM";
	ASSERT_EQ(img.width, 48);
	ASSERT_EQ(img.height, 48);

	int nonzero_count = 0;
	for (int v : img.pixels) {
		ASSERT_GE(v, 0);
		ASSERT_LE(v, 255);
		if (v > 0) ++nonzero_count;
	}
	EXPECT_GT(nonzero_count, 0) << "Rendered image is entirely black";
}

// Scope guard, updated for the post-generalization rule set: GPU SPPM must
// still cleanly reject (not crash, not silently mis-render) a scene using a
// MaterialType its camera/photon-pass dispatch genuinely has no BSDF sampler
// for. B5 (CornellCoatedDiffuse) is a good example -- pure sphere/quad
// geometry (so it isn't rejected for the unrelated triangle-mesh reason),
// but its sphere/box use MaterialType::CoatedDiffuse, which is intentionally
// NOT in sppm_is_delta_material()'s covered set (see that function's own
// comment on why: CPU's own coated_diffuse handling involves a multi-bounce
// coat-escape random walk this GPU port hasn't ported yet) -- so this locks
// in real, current out-of-scope behavior rather than an arbitrary
// placeholder id.
TEST_F(SppmGpuFirstSliceTest, RejectsSceneWithUnsupportedMaterial) {
	const char* path = "sppm_gpu_first_slice_test_b5.ppm";
	outputFiles_.push_back(path);

	int result = optix_render_main_sppm(
		32, 32, /*iterations=*/5, /*photons=*/500, /*max_depth=*/5,
		path, /*scene_id=*/"B5",
		278.0, 278.0, -800.0, 1);
	EXPECT_NE(result, 0) << "GPU SPPM should reject scene B5 (CoatedDiffuse has no GPU SPPM BSDF sampler)";
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

	ASSERT_EQ(optix_render_main_sppm(48, 48, 10, 2000, 5, sppmPath, "B3",
	                                  278.0, 278.0, -800.0, 1), 0);

	int plainResult = optix_render_main(48, 48, /*samples_per_pixel=*/50, /*max_depth=*/5,
	                                     plainPath, /*scene_id=*/"B3",
	                                     278.0, 278.0, -800.0, 1);
	ASSERT_EQ(plainResult, 0) << "Plain GPU render failed after an SPPM render in the same process";

	PPMImage img = load_ppm(plainPath);
	ASSERT_TRUE(img.valid);
	int nonzero_count = 0;
	for (int v : img.pixels) if (v > 0) ++nonzero_count;
	EXPECT_GT(nonzero_count, 0) << "Plain GPU render is all-black after an SPPM render -- possible "
	                            << "state leak between SPPMPathTracer and the regular path tracer";
}
