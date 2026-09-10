// live_preview_svgf_test.cpp
//
// Regression test for the SVGF spatiotemporal denoiser (gpu/optix/
// wavefront_svgf_math.h, wavefront_kernels_svgf.cu) - the denoising
// counterpart to live_preview_restir_gi_test.cpp (ReSTIR GI). Two things
// are checked: the same whole-image-mean-stays-bounded-and-flat regression
// this project's own ReSTIR tests already established (SVGF must not
// introduce a new runaway-bias mechanism of its own), and the actual thing
// SVGF exists for - a substantial reduction in per-pixel temporal variance
// versus the same scene rendered without it.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern "C" {
	#include "optix_interface.h"
}

namespace {

// Cornell Box ("A1") - same scene/camera every other Live Preview ReSTIR
// regression test in this project uses; see live_preview_restir_gi_test.cpp's
// own comment for why (canonical convergence test, and SVGF's own value is
// most visible on a scene with real indirect noise to clean up).
constexpr double kCamX = 278.0, kCamY = 278.0, kCamZ = -800.0;
constexpr double kLookX = 278.0, kLookY = 278.0, kLookZ = 278.0;

// Renders `numFrames` frames at a fixed camera and returns the per-pixel
// mean value over the LAST `varianceWindow` frames' own per-pixel variance,
// averaged across every pixel/channel - a single scalar summarizing how
// noisy the image still is once (if) it has had time to converge/stabilize.
double RenderAndMeasureTemporalVariance(int width, int height, bool enableSvgf,
										 int numFrames, int varianceWindow) {
	const int numPixels = width * height;
	const size_t numChannels = static_cast<size_t>(numPixels) * 3;
	std::vector<float> rgb(numChannels);
	std::vector<double> sum(numChannels, 0.0);
	std::vector<double> sumSq(numChannels, 0.0);

	for (int frame = 0; frame < numFrames; ++frame) {
		const bool ok = rt_realtime_render_frame(
			"A1", width, height, /*samples_per_pixel=*/1, /*max_depth=*/5,
			kCamX, kCamY, kCamZ,
			/*has_custom_lookat=*/true, kLookX, kLookY, kLookZ,
			/*denoise=*/false, /*denoise_blend=*/0.0,
			/*out_world_pos_buffer=*/nullptr, /*out_camera_basis=*/nullptr,
			rgb.data(), enableSvgf);
		EXPECT_TRUE(ok) << "rt_realtime_render_frame failed on frame " << frame
						 << " (enableSvgf=" << enableSvgf << ")";
		if (!ok) return -1.0;

		if (frame >= numFrames - varianceWindow) {
			for (size_t i = 0; i < numChannels; ++i) {
				const double v = rgb[i];
				sum[i] += v;
				sumSq[i] += v * v;
			}
		}
	}

	double totalVariance = 0.0;
	for (size_t i = 0; i < numChannels; ++i) {
		const double mean = sum[i] / varianceWindow;
		const double meanSq = sumSq[i] / varianceWindow;
		totalVariance += std::max(0.0, meanSq - mean * mean);
	}
	return totalVariance / static_cast<double>(numChannels);
}

} // namespace

TEST(LivePreviewSvgfTest, CornellBoxImageMeanStaysBoundedWithSvgfEnabled) {
	const int width = 64, height = 64;
	const int numPixels = width * height;
	const size_t numChannels = static_cast<size_t>(numPixels) * 3;
	std::vector<float> rgb(numChannels);

	// Same order-of-magnitude reasoning as live_preview_restir_gi_test.cpp's
	// own ceiling - SVGF's own temporally-integrated color starts from the
	// same raw radiance DI/GI already produce, so it inherits their own
	// converged-mean scale, not a new one.
	constexpr double kMaxPlausibleImageMean = 2.0;
	constexpr int kNumFrames = 60;

	for (int frame = 0; frame < kNumFrames; ++frame) {
		const bool ok = rt_realtime_render_frame(
			"A1", width, height, /*samples_per_pixel=*/1, /*max_depth=*/5,
			kCamX, kCamY, kCamZ,
			/*has_custom_lookat=*/true, kLookX, kLookY, kLookZ,
			/*denoise=*/false, /*denoise_blend=*/0.0,
			/*out_world_pos_buffer=*/nullptr, /*out_camera_basis=*/nullptr,
			rgb.data(), /*enable_svgf=*/true);
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
			<< " with SVGF enabled - a widespread blowup, not ordinary variance.";
	}
}

TEST(LivePreviewSvgfTest, SubstantiallyReducesTemporalVarianceOnAFixedCamera) {
	const int width = 64, height = 64;
	constexpr int kNumFrames = 60;
	constexpr int kVarianceWindow = 10;

	const double baselineVariance = RenderAndMeasureTemporalVariance(
		width, height, /*enableSvgf=*/false, kNumFrames, kVarianceWindow);
	const double svgfVariance = RenderAndMeasureTemporalVariance(
		width, height, /*enableSvgf=*/true, kNumFrames, kVarianceWindow);

	ASSERT_GE(baselineVariance, 0.0);
	ASSERT_GE(svgfVariance, 0.0);
	// SVGF's whole point is trading a bit of spatial/temporal lag for a much
	// more stable image - once its own temporal integration has had
	// kNumFrames - kVarianceWindow frames to build up history, the
	// remaining per-pixel variance should be a small fraction of the raw,
	// undenoised baseline's own. A generous 5x margin (not 10x+) to stay
	// robust across GPUs/driver versions while still clearly failing if
	// SVGF stops doing anything (e.g. a future change that accidentally
	// makes temporal integration a no-op).
	EXPECT_LT(svgfVariance, baselineVariance / 5.0)
		<< "SVGF variance (" << svgfVariance << ") is not substantially lower than "
		<< "the undenoised baseline (" << baselineVariance << ") - SVGF may not be "
		<< "actually filtering anything.";
}
