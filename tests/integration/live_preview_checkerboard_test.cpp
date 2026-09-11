// live_preview_checkerboard_test.cpp
//
// Regression test for checkerboard temporal upsampling (WavefrontPathTracer::
// render()'s own checkerboardActive - see wavefront_svgf_math.h's
// wf_checkerboard_pixel_active, wavefront_kernels_svgf.cu's
// svgf_checkerboard_clear_frame/svgf_temporal_integrate, and
// wavefront_kernels_restir.cu's restir_gi_finalize hold-over branch).
//
// Checkerboarding activates automatically whenever SVGF is enabled, once
// both SVGF's and ReSTIR GI's own history are warm (ReSTIR GI is
// unconditionally on for every Live Preview call, with no way to disable
// it) - so live_preview_svgf_test.cpp's own tests already exercise it
// implicitly from frame 2 onward. This file adds the two checks that test
// specifically GUARDS against the failure modes checkerboarding's own
// buffer-hold-over design depends on getting right:
//
//   1. No NaN/Inf, ever - a broken world-pos/albedo/normal hold-over would
//      most likely surface as SVGF's reprojection or edge-stopping math
//      dividing by/reading degenerate held state.
//   2. No persistent EVEN/ODD-frame brightness bias once converged - the
//      single most direct fingerprint of any of the three hold-over paths
//      (AOV/world-pos clear, SVGF temporal integrate, ReSTIR GI reservoir)
//      silently reverting to "wipe this pixel instead of holding it" for
//      whichever half of pixels is inactive on a given frame's parity would
//      be a systematic two-frame-periodic dip in brightness (half the image
//      losing its denoised/GI-lit value every other frame) - exactly what
//      this test checks for directly, rather than inferring it indirectly
//      from live_preview_svgf_test.cpp's own broader variance-reduction
//      assertion.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern "C" {
	#include "optix_interface.h"
}

namespace {

// Same scene/camera every other Live Preview SVGF/ReSTIR regression test in
// this project uses - see live_preview_svgf_test.cpp's own comment for why.
constexpr double kCamX = 278.0, kCamY = 278.0, kCamZ = -800.0;
constexpr double kLookX = 278.0, kLookY = 278.0, kLookZ = 278.0;

} // namespace

TEST(LivePreviewCheckerboardTest, NoNaNAcrossManyFramesOnceCheckerboardingIsActive) {
	const int width = 64, height = 64;
	const int numPixels = width * height;
	const size_t numChannels = static_cast<size_t>(numPixels) * 3;
	std::vector<float> rgb(numChannels);

	// Checkerboarding needs BOTH svgfHistoryValid_ and restirGiHistoryValid_
	// warm (WavefrontPathTracer::render()'s own checkerboardActive comment) -
	// both go valid after their own first successful call, so frame 0 is the
	// only guaranteed-inactive one; 40 frames gives ample steady-state
	// coverage well past any warm-up.
	constexpr int kNumFrames = 40;
	for (int frame = 0; frame < kNumFrames; ++frame) {
		const bool ok = rt_realtime_render_frame(
			"A1", width, height, /*samples_per_pixel=*/1, /*max_depth=*/5,
			kCamX, kCamY, kCamZ,
			/*has_custom_lookat=*/true, kLookX, kLookY, kLookZ,
			/*denoise=*/false, /*denoise_blend=*/0.0,
			/*out_world_pos_buffer=*/nullptr, /*out_camera_basis=*/nullptr,
			rgb.data(), /*enable_svgf=*/true);
		ASSERT_TRUE(ok) << "rt_realtime_render_frame failed on frame " << frame;
		for (size_t i = 0; i < numChannels; ++i) {
			ASSERT_FALSE(std::isnan(rgb[i])) << "NaN on frame " << frame << ", channel " << i;
			ASSERT_FALSE(std::isinf(rgb[i])) << "Inf on frame " << frame << ", channel " << i;
		}
	}
}

TEST(LivePreviewCheckerboardTest, NoPersistentEvenOddFrameBrightnessBias) {
	const int width = 64, height = 64;
	const int numPixels = width * height;
	const size_t numChannels = static_cast<size_t>(numPixels) * 3;
	std::vector<float> rgb(numChannels);

	// Skip enough frames for checkerboarding to be active AND for SVGF's own
	// temporal integration to have converged past its variance-bootstrap
	// window (kSvgfVarianceBootstrapFrames=4 real samples - up to 8 wall-
	// clock frames at half-rate sampling, wavefront_kernels_svgf.cu's own
	// comment) before measuring, so this isn't just catching ordinary
	// convergence transients.
	constexpr int kWarmupFrames = 20;
	constexpr int kMeasuredFrames = 30;

	std::vector<double> frameMeans;
	frameMeans.reserve(kMeasuredFrames);

	for (int frame = 0; frame < kWarmupFrames + kMeasuredFrames; ++frame) {
		const bool ok = rt_realtime_render_frame(
			"A1", width, height, /*samples_per_pixel=*/1, /*max_depth=*/5,
			kCamX, kCamY, kCamZ,
			/*has_custom_lookat=*/true, kLookX, kLookY, kLookZ,
			/*denoise=*/false, /*denoise_blend=*/0.0,
			/*out_world_pos_buffer=*/nullptr, /*out_camera_basis=*/nullptr,
			rgb.data(), /*enable_svgf=*/true);
		ASSERT_TRUE(ok) << "rt_realtime_render_frame failed on frame " << frame;

		if (frame >= kWarmupFrames) {
			double sum = 0.0;
			for (size_t i = 0; i < numChannels; ++i) sum += rgb[i];
			frameMeans.push_back(sum / static_cast<double>(numChannels));
		}
	}

	// Split the measured window by parity (which of the two checkerboard
	// phases was primary-sampled that frame) and compare the two groups'
	// own average brightness - a real hold-over bug (any of the 3 described
	// in this file's own header comment) would make one parity's frames
	// systematically dimmer, since that parity's own held-over contribution
	// would be missing/degenerate instead of a genuine one-frame-old value.
	double evenSum = 0.0, oddSum = 0.0;
	int evenCount = 0, oddCount = 0;
	for (size_t i = 0; i < frameMeans.size(); ++i) {
		if (i % 2 == 0) { evenSum += frameMeans[i]; ++evenCount; }
		else { oddSum += frameMeans[i]; ++oddCount; }
	}
	const double evenMean = evenSum / evenCount;
	const double oddMean = oddSum / oddCount;
	const double avgOfBoth = (evenMean + oddMean) / 2.0;
	ASSERT_GT(avgOfBoth, 1e-6) << "converged image is degenerately dark - can't evaluate parity bias";

	const double relativeDiff = std::abs(evenMean - oddMean) / avgOfBoth;
	EXPECT_LT(relativeDiff, 0.05)
		<< "Even-frame mean (" << evenMean << ") and odd-frame mean (" << oddMean
		<< ") differ by " << (relativeDiff * 100.0) << "% once converged - this is the "
		<< "signature of a broken checkerboard buffer hold-over (AOV/world-pos, SVGF "
		<< "temporal state, or ReSTIR GI reservoir) reverting to wiping the inactive "
		<< "parity instead of holding its last real value.";
}
