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

// A directly-viewed emissive surface (the Cornell Box's own ceiling light,
// kLightIntensity=15 in scene_builder.cpp vs. every reflective wall's own
// sub-1.0 linear albedo-scaled radiance) is excluded from the variance
// metric below - not because SVGF handles it badly, but because it isn't
// the kind of signal SVGF exists to clean up in the first place (a direct
// light hit has no Monte Carlo noise of its own to speak of; SVGF's whole
// purpose, per Schied et al.'s own paper, is denoising the STOCHASTIC
// indirect-lighting estimate on the reflective surfaces around it). Left
// in, its own ~15x-brighter-than-the-walls magnitude would dominate a
// plain averaged variance (variance scales with the square of a signal's
// own magnitude) badly enough that this metric would mostly be measuring
// the light's own residual noise rather than the walls'/floor's real
// indirect-lighting noise this test actually cares about.
constexpr double kVarianceMetricBrightnessCutoff = 2.0;

// Renders `numFrames` frames at a fixed camera and returns the per-pixel
// mean value over the LAST `varianceWindow` frames' own per-pixel variance,
// averaged across every included pixel/channel (see
// kVarianceMetricBrightnessCutoff's own comment for what's excluded and
// why) - a single scalar summarizing how noisy the image still is once (if)
// it has had time to converge/stabilize.
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
	size_t includedChannels = 0;
	for (size_t i = 0; i < numChannels; ++i) {
		const double mean = sum[i] / varianceWindow;
		if (mean > kVarianceMetricBrightnessCutoff) continue;  // the light source itself - see this file's own comment
		const double meanSq = sumSq[i] / varianceWindow;
		totalVariance += std::max(0.0, meanSq - mean * mean);
		++includedChannels;
	}
	return totalVariance / static_cast<double>(includedChannels);
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
	// Bumped again from 120/30 (which itself replaced an earlier 60/10 - see
	// git history for that first bump's own reasoning, still accurate): even
	// at 120/30 this test was observed to fail intermittently in a full-suite
	// run (measured ratio landing just under the 1.3x threshold on one run
	// out of several identical repeats) - the same "short measurement window
	// is a small, noisy sample of the residual per-pixel variance" mechanism
	// the first bump's own comment already diagnosed, just not yet reduced
	// enough. Doubling both again (240/60, same 0.25 window-to-total ratio)
	// halves the measurement window's own sampling error by roughly sqrt(2)
	// - if this still flakes, keep following this same remedy (more frames/
	// a wider window), not a lower 1.3x threshold - see that constant's own
	// comment for why loosening it defeats the point of this test.
	constexpr int kNumFrames = 240;
	constexpr int kVarianceWindow = 60;

	const double baselineVariance = RenderAndMeasureTemporalVariance(
		width, height, /*enableSvgf=*/false, kNumFrames, kVarianceWindow);
	const double svgfVariance = RenderAndMeasureTemporalVariance(
		width, height, /*enableSvgf=*/true, kNumFrames, kVarianceWindow);

	ASSERT_GE(baselineVariance, 0.0);
	ASSERT_GE(svgfVariance, 0.0);
	// SVGF's whole point is trading a bit of spatial/temporal lag for a much
	// more stable image - once its own temporal integration has had
	// kNumFrames - kVarianceWindow frames to build up history, the
	// remaining per-pixel variance should be meaningfully lower than the
	// raw, undenoised baseline's own. 1.3x (not the much larger margin an
	// earlier version of this test used) is what this Cornell Box scene's
	// own real indirect-lighting noise, on the reflective walls/floor the
	// brightness cutoff above leaves in the metric, was actually measured to
	// achieve (~1.4x-2.5x across repeated runs and execution contexts,
	// isolated and under the full suite) - the earlier, much larger margin
	// only passed because svgf_finalize's own albedo-floor bug (fixed
	// alongside this test) forced the light source's own pixels to a
	// constant black every frame, an unrelated rendering defect that
	// happened to look like "perfect convergence" to an unweighted variance
	// metric. 1.3x stays below every measured value seen so far (margin for
	// run-to-run/execution-context variability) while still clearly failing
	// if SVGF stops doing anything (e.g. a future change that accidentally
	// makes temporal integration a no-op, which would put this ratio at
	// ~1.0). If this still flakes, the fix is more frames/a wider window
	// above, not a lower threshold - a value much below this stops
	// distinguishing "SVGF works" from "SVGF is barely doing anything."
	EXPECT_LT(svgfVariance, baselineVariance / 1.3)
		<< "SVGF variance (" << svgfVariance << ") is not substantially lower than "
		<< "the undenoised baseline (" << baselineVariance << ") - SVGF may not be "
		<< "actually filtering anything.";
}
