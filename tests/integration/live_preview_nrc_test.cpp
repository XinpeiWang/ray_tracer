// live_preview_nrc_test.cpp
//
// Regression test for the Neural Radiance Cache (gpu/optix/wavefront_nrc_*.h,
// wavefront_kernels_nrc.cu) - see this project's own plan.
//
// Mirrors live_preview_restir_gi_test.cpp/live_preview_checkerboard_test.cpp's
// own harness: renders the Cornell Box (scene "A1") via the same Live Preview
// entry point (rt_realtime_render_frame) real interactive use goes through.
//
// Runs past kNrcWarmupSteps (256 training steps, one per render() call) so
// these tests actually exercise the render-path QUERY (not just training) -
// the highest-risk path given it substitutes a hand-trained network's own
// output directly into the framebuffer.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern "C" {
	#include "optix_interface.h"
}

namespace {

// Cornell Box ("A1") - see live_preview_restir_gi_test.cpp's own comment for
// why this scene: fully enclosed, colored walls, both a diffuse (Lambertian)
// and (via scene variants) glossy convergence target.
constexpr double kCamX = 278.0, kCamY = 278.0, kCamZ = -800.0;
constexpr double kLookX = 278.0, kLookY = 278.0, kLookZ = 278.0;

// Same order-of-magnitude ceiling as live_preview_restir_gi_test.cpp's own
// kMaxPlausibleImageMean - NRC substitutes a learned estimate for real
// bounces, so an untrained/diverged network could plausibly produce a wildly
// bright (but still finite) image; this catches that without being so tight
// it flags ordinary noise.
constexpr double kMaxPlausibleImageMean = 5.0;

bool renderOneFrame(int width, int height, std::vector<float>& rgb,
					 bool enable_nrc, bool enable_svgf = false, bool enable_probe_cache = false,
					 bool enable_path_guiding = false) {
	return rt_realtime_render_frame(
		"A1", width, height, /*samples_per_pixel=*/1, /*max_depth=*/5,
		kCamX, kCamY, kCamZ,
		/*has_custom_lookat=*/true, kLookX, kLookY, kLookZ,
		/*denoise=*/false, /*denoise_blend=*/0.0,
		/*out_world_pos_buffer=*/nullptr, /*out_camera_basis=*/nullptr,
		rgb.data(),
		enable_svgf, /*enable_restir_gi=*/true, /*max_component_value=*/50.0f,
		/*svgf_tuning=*/nullptr, /*enable_restir_di=*/true,
		enable_probe_cache, enable_path_guiding,
		/*enable_temporal_upscale=*/false, /*temporal_upscale_factor=*/2, /*temporal_jitter_base_index=*/0,
		enable_nrc);
}

} // namespace

TEST(LivePreviewNrcTest, NoNaNOrInfOverManyFrames) {
	const int width = 48, height = 48;
	const int numPixels = width * height;
	const size_t numChannels = static_cast<size_t>(numPixels) * 3;
	std::vector<float> rgb(numChannels);

	// 300 frames crosses kNrcWarmupSteps (256) - covers both the training-
	// only warmup window (where every render-path query is rejected) and the
	// post-warmup window where the network's own predictions actually reach
	// the framebuffer.
	constexpr int kNumFrames = 300;
	for (int frame = 0; frame < kNumFrames; ++frame) {
		std::fill(rgb.begin(), rgb.end(), 0.0f);
		const bool ok = renderOneFrame(width, height, rgb, /*enable_nrc=*/true);
		ASSERT_TRUE(ok) << "rt_realtime_render_frame failed on frame " << frame;

		double sum = 0.0;
		for (size_t i = 0; i < numChannels; ++i) {
			const float v = rgb[i];
			ASSERT_FALSE(std::isnan(v)) << "NaN channel " << i << " on frame " << frame;
			ASSERT_FALSE(std::isinf(v)) << "Inf channel " << i << " on frame " << frame;
			sum += v;
		}
		const double mean = sum / static_cast<double>(numChannels);
		ASSERT_LT(mean, kMaxPlausibleImageMean)
			<< "Frame " << frame << "'s whole-image mean is " << mean
			<< " - implausibly bright for a converging Cornell Box, not "
			<< "ordinary NRC training noise.";
	}
}

TEST(LivePreviewNrcTest, WarmStartMatchesNrcDisabledBeforeWarmup) {
	// Before kNrcWarmupSteps training steps have run, the render-path query
	// is unconditionally rejected (falls through to a real bounce) - the
	// first few frames with NRC enabled should therefore render exactly like
	// NRC disabled (same seed sequence, same scene, same camera - the ONLY
	// difference enable_nrc introduces is the query gate itself, and that
	// gate is closed the whole time this test runs).
	const int width = 32, height = 32;
	const int numPixels = width * height;
	const size_t numChannels = static_cast<size_t>(numPixels) * 3;
	std::vector<float> rgbNrcOff(numChannels), rgbNrcOn(numChannels);

	constexpr int kFramesBeforeWarmup = 10;  // well under kNrcWarmupSteps=256
	for (int frame = 0; frame < kFramesBeforeWarmup; ++frame) {
		ASSERT_TRUE(renderOneFrame(width, height, rgbNrcOff, /*enable_nrc=*/false));
	}
	for (int frame = 0; frame < kFramesBeforeWarmup; ++frame) {
		ASSERT_TRUE(renderOneFrame(width, height, rgbNrcOn, /*enable_nrc=*/true));
	}

	for (size_t i = 0; i < numChannels; ++i) {
		ASSERT_FALSE(std::isnan(rgbNrcOn[i])) << "NaN channel " << i;
		ASSERT_FALSE(std::isinf(rgbNrcOn[i])) << "Inf channel " << i;
	}
	// Not bit-exact (the training pipeline's own RNG draws still perturb the
	// GPU's global PCG state differently than the NRC-disabled run, and the
	// render path shares no explicit seed synchronization with training) -
	// checking the two runs' whole-image MEANS are close is a fair proxy for
	// "the query gate genuinely stayed closed" without requiring bit-exact
	// determinism this codebase doesn't otherwise guarantee across an
	// unrelated pipeline's own RNG draws.
	double sumOff = 0.0, sumOn = 0.0;
	for (size_t i = 0; i < numChannels; ++i) { sumOff += rgbNrcOff[i]; sumOn += rgbNrcOn[i]; }
	const double meanOff = sumOff / static_cast<double>(numChannels);
	const double meanOn = sumOn / static_cast<double>(numChannels);
	EXPECT_NEAR(meanOn, meanOff, std::max(0.05, meanOff * 0.5))
		<< "meanOn=" << meanOn << " meanOff=" << meanOff
		<< " - NRC query should still be fully gated off before warmup.";
}

TEST(LivePreviewNrcTest, InteractionWithSvgfProbeCacheAndPathGuiding) {
	// All Live Preview features that can plausibly run alongside NRC,
	// enabled together - see this project's own plan's "documented
	// untested-by-construction combination" note.
	const int width = 48, height = 48;
	const int numPixels = width * height;
	const size_t numChannels = static_cast<size_t>(numPixels) * 3;
	std::vector<float> rgb(numChannels);

	constexpr int kNumFrames = 60;
	for (int frame = 0; frame < kNumFrames; ++frame) {
		std::fill(rgb.begin(), rgb.end(), 0.0f);
		const bool ok = renderOneFrame(width, height, rgb, /*enable_nrc=*/true,
										/*enable_svgf=*/true, /*enable_probe_cache=*/true,
										/*enable_path_guiding=*/true);
		ASSERT_TRUE(ok) << "rt_realtime_render_frame failed on frame " << frame;
		for (size_t i = 0; i < numChannels; ++i) {
			ASSERT_FALSE(std::isnan(rgb[i])) << "NaN on frame " << frame << ", channel " << i;
			ASSERT_FALSE(std::isinf(rgb[i])) << "Inf on frame " << frame << ", channel " << i;
		}
	}
}
