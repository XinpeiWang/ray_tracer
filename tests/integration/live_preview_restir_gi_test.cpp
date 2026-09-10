// live_preview_restir_gi_test.cpp
//
// Regression test for ReSTIR GI (gpu/optix/wavefront_restir_gi_math.h,
// wavefront_kernels_restir.cu's restir_gi_finalize/restir_gi_spatial_reuse) -
// the GI counterpart to live_preview_restir_firefly_test.cpp (ReSTIR DI).
//
// This project's own ReSTIR DI feature shipped with a subtle bug (a
// reservoir's M clamped AFTER weightSum had already been built from the
// uncapped value, inflating W - see wavefront_restir_helpers.h's own
// wf_restir_temporal_combine comment) that a lenient, one-shot mean
// threshold failed to catch; it only surfaced once the whole-image mean was
// checked ACROSS many frames for a slow upward drift. GI's own temporal/
// spatial reuse shares that exact "M clamp before combine" discipline (see
// restir_gi_finalize's own comment) and the SAME compounding-bias failure
// mode is possible in principle if a future change reintroduces it - so
// this test checks for drift explicitly, not just a single lenient ceiling.
//
// Renders the Cornell Box (scene "A1") via the same Live Preview entry point
// (rt_realtime_render_frame) real interactive use goes through, with ReSTIR
// GI enabled (the current unconditional default for that entry point - see
// optix_interface.cpp's own enableRestirGi(true) comment).

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern "C" {
	#include "optix_interface.h"
}

namespace {

// Cornell Box ("A1") - see src/TheRestOfYourLife/scene_registry.h's
// kCornellBoxCamera, and live_preview_restir_firefly_test.cpp's own comment
// for why this is the scene of choice: a small, fully enclosed box with
// colored walls is both this renderer's canonical convergence test AND the
// classic ReSTIR GI showcase (diffuse-to-diffuse color bleeding needs a
// Lambertian secondary bounce, which this scene's own walls/floor/ceiling
// provide in abundance).
constexpr double kCamX = 278.0, kCamY = 278.0, kCamZ = -800.0;
constexpr double kLookX = 278.0, kLookY = 278.0, kLookZ = 278.0;

} // namespace

TEST(LivePreviewRestirGiTest, CornellBoxImageMeanStaysBoundedAndFlat) {
	const int width = 64, height = 64;
	const int numPixels = width * height;
	const size_t numChannels = static_cast<size_t>(numPixels) * 3;
	std::vector<float> rgb(numChannels);

	// Same order-of-magnitude ceiling as live_preview_restir_firefly_test.cpp's
	// own kMaxPlausibleImageMean (classic NEE's own converged mean for this
	// exact scene/camera/resolution is ~0.16 - measured directly during this
	// session's own DI bug investigation) - GI adds indirect bounces on top
	// of that, plausibly a few times brighter for a fully enclosed box, but
	// nowhere near this ceiling absent a real bug.
	constexpr double kMaxPlausibleImageMean = 2.0;
	constexpr int kNumFrames = 150;

	// Split into an early and late window - a compounding-bias bug (like
	// DI's own M-cap mistake) shows up as the LATE window's mean drifting
	// well above the EARLY window's, not as an immediately-implausible
	// single-frame value - exactly the shape that let DI's own bug hide
	// behind a lenient one-shot ceiling for as long as it did.
	constexpr int kEarlyWindowFrames = 20;
	constexpr int kLateWindowStart = 100;
	double earlyWindowMeanSum = 0.0;
	double lateWindowMeanSum = 0.0;
	int lateWindowCount = 0;

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
		for (size_t i = 0; i < numChannels; ++i) {
			const float v = rgb[i];
			ASSERT_FALSE(std::isnan(v)) << "NaN channel " << i << " on frame " << frame;
			ASSERT_FALSE(std::isinf(v)) << "Inf channel " << i << " on frame " << frame;
			sum += v;
		}
		const double mean = sum / static_cast<double>(numChannels);
		ASSERT_LT(mean, kMaxPlausibleImageMean)
			<< "Frame " << frame << "'s whole-image mean is " << mean
			<< " - a widespread blowup, not ordinary 1-spp NEE/GI variance.";

		if (frame < kEarlyWindowFrames) {
			earlyWindowMeanSum += mean;
		} else if (frame >= kLateWindowStart) {
			lateWindowMeanSum += mean;
			++lateWindowCount;
		}
	}

	const double earlyWindowMean = earlyWindowMeanSum / kEarlyWindowFrames;
	const double lateWindowMean = lateWindowMeanSum / lateWindowCount;
	// A real compounding-bias bug inflates the mean many-fold over this many
	// frames (this session's own DI bug reached ~44x by frame 250) - 3x is a
	// generous margin over ordinary run-to-run noise while still catching
	// that class of regression well before it reaches "obviously broken".
	EXPECT_LT(lateWindowMean, earlyWindowMean * 3.0)
		<< "Late-window mean (" << lateWindowMean << ") drifted well above the "
		<< "early-window mean (" << earlyWindowMean << ") - looks like a "
		<< "frame-over-frame compounding bias, not ordinary variance.";
}
