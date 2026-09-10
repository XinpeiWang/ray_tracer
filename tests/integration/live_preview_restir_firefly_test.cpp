// live_preview_restir_firefly_test.cpp
//
// Regression test for a Live Preview "firefly" bug found interactively (a
// visibly corrupted, blocky, non-converging render) and root-caused via
// direct GPU instrumentation rather than code reading alone. The actual
// mechanism: Live Preview's zero-init default pixel reconstruction filter is
// Gaussian (GpuCameraParams::filterKind==0), whose per-sample weight
// legitimately approaches zero for a sample whose random sub-pixel jitter
// lands near the filter's own support edge (wavefront_kernels_camera.cu's
// own filterWeight comment already calls this "tiny in absolute magnitude").
// normalize_framebuffer (wavefront_kernels_accumulate.cu) divides accumulated
// radiance by the SUM of these weights - correct once many samples have
// accumulated, but at Live Preview's samples_per_pixel=1-per-call
// granularity a single near-zero-weight sample IS the entire divisor for
// that call, turning an entirely ordinary raw radiance value (confirmed via
// instrumentation: 0.02-2.7, nothing anomalous) into a huge displayed spike.
// This is NOT a ReSTIR defect - it predates this feature and is a known
// characteristic of Gaussian/Mitchell-style filters at low sample counts -
// but ReSTIR's own weight-seeking candidate resampling does make it MORE
// VISIBLE, by tending to produce somewhat brighter raw per-sample radiance
// than the classic single-draw path, which is why this surfaced now. The
// real fix (optix_interface.cpp's rt_realtime_render_frame) forces a box
// filter (weight always 1.0) for Live Preview specifically, removing the
// divide-by-near-zero mechanism entirely - matching every other real-time/
// progressive renderer's own reasoning that filters like Gaussian only pay
// for themselves once many samples have accumulated.
//
// No prior test exercised rt_realtime_render_frame() at all (every other
// GPU test goes through the batch entry point, optix_render_main(), which
// never sets OptiXRenderer::enableRestir(true) or hits this filter code path
// at 1 sample per pixel) - this test closes that coverage gap by calling the
// SAME entry point Live Preview uses, repeatedly, checking the WHOLE-IMAGE
// mean rather than any single pixel's own value: a pixel immediately
// adjacent to an area light can legitimately produce an occasional brighter
// 1-spp NEE sample without that being a bug, but a genuine blowup (whether
// from the filter-weight mechanism above or a future regression in either
// the filter or the ReSTIR reservoir math) measurably moves the whole
// image's mean, not just one pixel.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern "C" {
	#include "optix_interface.h"
}

namespace {

// Cornell Box ("A1") - see src/TheRestOfYourLife/scene_registry.h's
// kCornellBoxCamera. Its only light emits (15,15,15); every material is
// Lambertian or a fully-specular dielectric, so this scene's own true
// converged image mean is a small multiple of that, not orders of
// magnitude above it.
constexpr double kCamX = 278.0, kCamY = 278.0, kCamZ = -800.0;
constexpr double kLookX = 278.0, kLookY = 278.0, kLookZ = 278.0;

} // namespace

TEST(LivePreviewRestirTest, CornellBoxImageMeanNeverBlowsUp) {
	const int width = 64, height = 64;
	const int numPixels = width * height;
	const size_t numChannels = static_cast<size_t>(numPixels) * 3;
	std::vector<float> rgb(numChannels);

	// A converged Cornell Box render's mean pixel value sits well under the
	// light's own emission (15) - most of the image is dimmer wall/floor/
	// box reflection, with the small bright light quad itself and its
	// immediate surroundings pulling the mean up only slightly. This
	// ceiling is a generous multiple of that (a single unconverged 1-spp
	// frame is noisier than a converged image), while still being many
	// orders of magnitude below what the original bug's filter-weight
	// divide-by-near-zero (or a genuine reservoir blowup) would produce.
	constexpr double kMaxPlausibleImageMean = 25.0;
	constexpr int kNumFrames = 100;

	for (int frame = 0; frame < kNumFrames; ++frame) {
		std::fill(rgb.begin(), rgb.end(), 0.0f);
		const bool ok = rt_realtime_render_frame(
			"A1", width, height, /*samples_per_pixel=*/1, /*max_depth=*/5,
			kCamX, kCamY, kCamZ,
			/*has_custom_lookat=*/true, kLookX, kLookY, kLookZ,
			/*denoise=*/false, /*denoise_blend=*/0.0,
			/*out_world_pos_buffer=*/nullptr, /*out_camera_basis=*/nullptr,
			rgb.data());
		ASSERT_TRUE(ok) << "rt_realtime_render_frame failed on frame " << frame;

		double sum = 0.0;
		float maxVal = 0.0f;
		for (size_t i = 0; i < numChannels; ++i) {
			const float v = rgb[i];
			ASSERT_FALSE(std::isnan(v)) << "NaN channel " << i << " on frame " << frame;
			ASSERT_FALSE(std::isinf(v)) << "Inf channel " << i << " on frame " << frame;
			sum += v;
			if (v > maxVal) maxVal = v;
		}
		const double mean = sum / static_cast<double>(numChannels);
		ASSERT_LT(mean, kMaxPlausibleImageMean)
			<< "Frame " << frame << "'s whole-image mean is " << mean << " (brightest pixel channel "
			<< maxVal << ") - a widespread blowup, not ordinary single-pixel NEE variance near the light.";
	}
}
