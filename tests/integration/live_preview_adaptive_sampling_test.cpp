// live_preview_adaptive_sampling_test.cpp
//
// Regression test for Live Preview's adaptive-sampling pixel skip (Stage 2a
// of this project's own plan): rt_realtime_render_frame()'s
// enable_adaptive_sampling/in_active_pixel_mask parameters, threaded through
// WavefrontPathTracer::setActivePixelMask() to generate_camera_rays' own
// wf_adaptive_pixel_active() gate (gpu/optix/wavefront_svgf_math.h) - the
// same "never enqueue, never touch weightBuffer" shape
// wf_checkerboard_pixel_active() already uses, so a masked-inactive pixel's
// out_rgb_buffer value comes back exactly 0.0 via normalize_framebuffer's
// own zero-weight guard (there is no cross-call accumulation at this DLL
// layer - that's the caller's job, see qt_gui/realtime_preview_session.cpp).
//
// What this guards against:
//   1. A real mask with SOME pixels marked inactive leaves exactly those
//      pixels at 0 in out_rgb_buffer, while active pixels still render real
//      (non-degenerate) content - the direct, end-to-end confirmation that
//      wf_adaptive_pixel_active()'s pure-math unit tests (wavefront_svgf_
//      math_tests.cpp) actually reach the device correctly through the
//      whole host-upload/kernel-launch chain.
//   2. enable_adaptive_sampling=false ignores in_active_pixel_mask entirely,
//      even an all-zero ("every pixel inactive") one - optix_interface.cpp's
//      own gate ("false always wins", matching every sibling enable_X
//      parameter) must not let stale/leftover mask data silently blank the
//      image once the feature is toggled off.
//   3. No NaN/Inf across many frames with the feature active - same basic
//      sanity every other Live Preview regression test in this project
//      checks first.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern "C" {
	#include "optix_interface.h"
}

namespace {

// Same scene/camera every other Live Preview regression test in this
// project uses - see live_preview_checkerboard_test.cpp's own comment.
constexpr double kCamX = 278.0, kCamY = 278.0, kCamZ = -800.0;
constexpr double kLookX = 278.0, kLookY = 278.0, kLookZ = 278.0;

} // namespace

TEST(LivePreviewAdaptiveSamplingTest, NoNaNAcrossManyFramesWhileActive) {
	const int width = 64, height = 64;
	const int numPixels = width * height;
	const size_t numChannels = static_cast<size_t>(numPixels) * 3;
	std::vector<float> rgb(numChannels);
	// Every pixel active - the common case (nothing has converged yet).
	std::vector<unsigned char> mask(numPixels, 1);

	constexpr int kNumFrames = 20;
	for (int frame = 0; frame < kNumFrames; ++frame) {
		const bool ok = rt_realtime_render_frame(
			"A1", width, height, /*samples_per_pixel=*/1, /*max_depth=*/5,
			kCamX, kCamY, kCamZ,
			/*has_custom_lookat=*/true, kLookX, kLookY, kLookZ,
			/*denoise=*/false, /*denoise_blend=*/0.0,
			/*out_world_pos_buffer=*/nullptr, /*out_camera_basis=*/nullptr,
			rgb.data(), /*enable_svgf=*/false, /*enable_restir_gi=*/true,
			/*max_component_value=*/50.0f, /*svgf_tuning=*/nullptr,
			/*enable_restir_di=*/true, /*enable_probe_cache=*/false,
			/*enable_path_guiding=*/false,
			/*enable_temporal_upscale=*/false, /*temporal_upscale_factor=*/2,
			/*temporal_jitter_base_index=*/0,
			/*enable_nrc=*/false, /*enable_neural_upscale=*/false,
			/*out_neural_upscale_buffer=*/nullptr,
			/*aperture_override=*/-1.0, /*focus_distance_override=*/-1.0,
			/*enable_adaptive_sampling=*/true, mask.data());
		ASSERT_TRUE(ok) << "rt_realtime_render_frame failed on frame " << frame;
		for (size_t i = 0; i < numChannels; ++i) {
			ASSERT_FALSE(std::isnan(rgb[i])) << "NaN on frame " << frame << ", channel " << i;
			ASSERT_FALSE(std::isinf(rgb[i])) << "Inf on frame " << frame << ", channel " << i;
		}
	}
}

TEST(LivePreviewAdaptiveSamplingTest, MaskedInactivePixelsComeBackExactlyBlack) {
	const int width = 64, height = 64;
	const int numPixels = width * height;
	const size_t numChannels = static_cast<size_t>(numPixels) * 3;
	std::vector<float> rgb(numChannels, -1.0f);  // sentinel: every entry must be overwritten below
	std::vector<unsigned char> mask(numPixels, 1);

	// Mark the left half of the image inactive ("already converged") -
	// exactly this call's own worth of pixels never gets a queued ray, so
	// out_rgb_buffer for them must come back untouched-by-this-call, which
	// normalize_framebuffer's zero-weight guard reports as exactly 0.0.
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width / 2; ++x) mask[y * width + x] = 0;
	}

	ASSERT_TRUE(rt_realtime_render_frame(
		"A1", width, height, /*samples_per_pixel=*/4, /*max_depth=*/5,
		kCamX, kCamY, kCamZ,
		/*has_custom_lookat=*/true, kLookX, kLookY, kLookZ,
		/*denoise=*/false, /*denoise_blend=*/0.0,
		/*out_world_pos_buffer=*/nullptr, /*out_camera_basis=*/nullptr,
		rgb.data(), /*enable_svgf=*/false, /*enable_restir_gi=*/true,
		/*max_component_value=*/50.0f, /*svgf_tuning=*/nullptr,
		/*enable_restir_di=*/true, /*enable_probe_cache=*/false,
		/*enable_path_guiding=*/false,
		/*enable_temporal_upscale=*/false, /*temporal_upscale_factor=*/2,
		/*temporal_jitter_base_index=*/0,
		/*enable_nrc=*/false, /*enable_neural_upscale=*/false,
		/*out_neural_upscale_buffer=*/nullptr,
		/*aperture_override=*/-1.0, /*focus_distance_override=*/-1.0,
		/*enable_adaptive_sampling=*/true, mask.data()));

	int blackInactiveCount = 0;
	double activeSum = 0.0;
	int activeChannelCount = 0;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			const int pixel = y * width + x;
			const size_t idx = static_cast<size_t>(pixel) * 3;
			if (x < width / 2) {
				for (int c = 0; c < 3; ++c) {
					EXPECT_EQ(rgb[idx + c], 0.0f)
						<< "masked-inactive pixel (" << x << "," << y << ") channel " << c
						<< " should be exactly 0.0 (never enqueued this call), got " << rgb[idx + c];
				}
				++blackInactiveCount;
			} else {
				for (int c = 0; c < 3; ++c) { activeSum += rgb[idx + c]; ++activeChannelCount; }
			}
		}
	}
	ASSERT_EQ(blackInactiveCount, (width / 2) * height);
	// The active half is a real render of the Cornell Box's own lit
	// interior - degenerately black here would mean the mask accidentally
	// suppressed the ACTIVE half too, not just the inactive one.
	const double activeMean = activeSum / static_cast<double>(activeChannelCount);
	EXPECT_GT(activeMean, 1e-4)
		<< "active half of the image is degenerately dark (mean=" << activeMean
		<< ") - the mask may be suppressing pixels it was not supposed to.";
}

TEST(LivePreviewAdaptiveSamplingTest, DisabledFlagIgnoresAnAllZeroMask) {
	const int width = 64, height = 64;
	const int numPixels = width * height;
	const size_t numChannels = static_cast<size_t>(numPixels) * 3;
	std::vector<float> rgb(numChannels);
	// Every pixel marked inactive - if enable_adaptive_sampling=false didn't
	// unconditionally win (optix_interface.cpp's own gate), this would
	// render an entirely black frame instead of a normal one.
	std::vector<unsigned char> allInactiveMask(numPixels, 0);

	ASSERT_TRUE(rt_realtime_render_frame(
		"A1", width, height, /*samples_per_pixel=*/4, /*max_depth=*/5,
		kCamX, kCamY, kCamZ,
		/*has_custom_lookat=*/true, kLookX, kLookY, kLookZ,
		/*denoise=*/false, /*denoise_blend=*/0.0,
		/*out_world_pos_buffer=*/nullptr, /*out_camera_basis=*/nullptr,
		rgb.data(), /*enable_svgf=*/false, /*enable_restir_gi=*/true,
		/*max_component_value=*/50.0f, /*svgf_tuning=*/nullptr,
		/*enable_restir_di=*/true, /*enable_probe_cache=*/false,
		/*enable_path_guiding=*/false,
		/*enable_temporal_upscale=*/false, /*temporal_upscale_factor=*/2,
		/*temporal_jitter_base_index=*/0,
		/*enable_nrc=*/false, /*enable_neural_upscale=*/false,
		/*out_neural_upscale_buffer=*/nullptr,
		/*aperture_override=*/-1.0, /*focus_distance_override=*/-1.0,
		/*enable_adaptive_sampling=*/false, allInactiveMask.data()));

	double sum = 0.0;
	for (float v : rgb) sum += v;
	const double mean = sum / static_cast<double>(numChannels);
	EXPECT_GT(mean, 1e-4)
		<< "enable_adaptive_sampling=false rendered a degenerately dark image (mean=" << mean
		<< ") - an all-inactive mask should have no effect at all when the feature is disabled.";
}
