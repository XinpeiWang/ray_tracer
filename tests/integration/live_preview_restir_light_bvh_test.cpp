// live_preview_restir_light_bvh_test.cpp
//
// Regression coverage for wavefront's bounding-cone light BVH (gpu/optix/
// wavefront_restir_helpers.h's wf_light_bvh_sample_index()/wf_light_bvh_pmf())
// feeding both ReSTIR DI candidate generation and classic NEE - no prior test
// exercised this GPU device code at all (bvh_light_sampler2_tests.cpp/
// light_bvh_node_tests.cpp/compact_light_bounds_tests.cpp validate the same
// algorithm host-side, on the CPU, which is a different code path entirely).
// The light BVH is built unconditionally by OptiXRenderer::buildScene()
// whenever a scene has at least one emissive light (optix_renderer_scene.cpp),
// so BOTH tests below exercise it for real, on whatever GPU this runs on -
// there is no special "enable the light BVH" scene setup needed.
//
// Two specific, previously-unguarded regressions motivate this file:
//
//   1. ReSTIR DI's winning reservoir sample's selection_pdf used to be
//      re-derived from the (fixed, power-only) alias table unconditionally,
//      even when the light BVH's own position-dependent pmf was what
//      actually drew that sample - silently corrupting the MIS weight
//      whenever ReSTIR DI and the light BVH are both active (every Live
//      Preview session with at least one light). RestirOnAndOffAgreeOnMean
//      below catches this directly: a wrong selection_pdf biases ReSTIR's
//      own converged mean away from classic NEE's (independently correct)
//      converged mean by a large, systematic margin - not ordinary Monte
//      Carlo noise.
//   2. wf_generate_restir_candidate()'s own per-draw light-BVH rejection
//      (a shading point outside every currently-considered light's cone of
//      influence for that one random draw) used to `break` the whole RIS
//      candidate loop instead of `continue`-ing, discarding the rest of the
//      frame's resampling budget on the first unlucky draw. Both tests below
//      exercise this on every frame; a regression back to `break` shows up
//      as elevated noise/darker frames in RoughGlassImageMeanNeverBlowsUp's
//      own converged-mean ceiling, though CATCHING it precisely would need a
//      much larger sample count than a fast regression test can afford - the
//      MIS-weight bug above is the sharper, more reliable signal.
//
// Both tests render via rt_realtime_render_frame(), the SAME entry point
// real interactive Live Preview use goes through (always the wavefront
// backend - see that function's own comment) - not optix_render_main(),
// which never enables ReSTIR DI.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern "C" {
	#include "optix_interface.h"
}

namespace {

// Cornell Box ("A1") - see src/TheRestOfYourLife/scene_registry.h's
// kCornellBoxCamera and live_preview_restir_firefly_test.cpp's own comment.
// All-Lambertian (plus one fully-specular dielectric sphere, which never
// reaches NEE) - the simplest scene to reason "ReSTIR-on and ReSTIR-off
// should converge to the same mean" about, since there is no glossy BSDF
// noise of its own to conflate with a light-selection bug.
constexpr double kCamX = 278.0, kCamY = 278.0, kCamZ = -800.0;
constexpr double kLookX = 278.0, kLookY = 278.0, kLookZ = 278.0;

// "B3" - Cornell box with a GGX rough-dielectric sphere (pbrt-v4
// RoughDielectricBxDF, see scene_registry_data.h) - unlike A1's fully-
// specular glass, RoughDielectric's NEE is real and non-trivial (evalGlossyF,
// two-sided reflection+transmission - see wavefront_device_helpers.h's own
// comment on this exact NEE extension), and it's the material type this
// session's own light-BVH wiring pass initially left stuck on the alias-
// table-only path (wavefront_kernels_materials_dielectric.cu's evaluate_
// materials_dielectric()) until fixed to also receive the light BVH context.
constexpr const char* kRoughGlassSceneId = "B3";

double renderMeanBrightness(const char* sceneId, int width, int height, int numFrames,
							 bool enableRestirDi, bool enableRestirGi) {
	const int numPixels = width * height;
	const size_t numChannels = static_cast<size_t>(numPixels) * 3;
	std::vector<float> rgb(numChannels);
	double meanSum = 0.0;

	for (int frame = 0; frame < numFrames; ++frame) {
		std::fill(rgb.begin(), rgb.end(), 0.0f);
		const bool ok = rt_realtime_render_frame(
			sceneId, width, height, /*samples_per_pixel=*/1, /*max_depth=*/5,
			kCamX, kCamY, kCamZ,
			/*has_custom_lookat=*/true, kLookX, kLookY, kLookZ,
			/*denoise=*/false, /*denoise_blend=*/0.0,
			/*out_world_pos_buffer=*/nullptr, /*out_camera_basis=*/nullptr,
			rgb.data(),
			/*enable_svgf=*/false, enableRestirGi, /*max_component_value=*/50.0f,
			/*svgf_tuning=*/nullptr, enableRestirDi);
		EXPECT_TRUE(ok) << "rt_realtime_render_frame failed on frame " << frame
						<< " (scene " << sceneId << ", restirDi=" << enableRestirDi << ")";
		if (!ok) return -1.0;

		double sum = 0.0;
		for (size_t i = 0; i < numChannels; ++i) {
			const float v = rgb[i];
			EXPECT_FALSE(std::isnan(v)) << "NaN channel " << i << " on frame " << frame
										 << " (scene " << sceneId << ", restirDi=" << enableRestirDi << ")";
			EXPECT_FALSE(std::isinf(v)) << "Inf channel " << i << " on frame " << frame
										 << " (scene " << sceneId << ", restirDi=" << enableRestirDi << ")";
			sum += v;
		}
		meanSum += sum / static_cast<double>(numChannels);
	}
	return meanSum / static_cast<double>(numFrames);
}

} // namespace

// Exercises the light BVH feeding a RoughDielectric material's own NEE (the
// dielectric kernel's own light-BVH wiring, wavefront_kernels_materials_
// dielectric.cu) with ReSTIR DI on (this entry point's default) - same
// blow-up/NaN/Inf shape as live_preview_restir_firefly_test.cpp's own
// CornellBoxImageMeanNeverBlowsUp, just on a scene that actually reaches the
// dielectric kernel instead of only Lambertian/Metal.
TEST(LivePreviewRestirLightBvhTest, RoughGlassImageMeanNeverBlowsUp) {
	const int width = 64, height = 64;
	// Same order-of-magnitude ceiling as the other Live Preview ReSTIR
	// regression tests - a rough-glass sphere in an otherwise-Lambertian box
	// is not plausibly brighter than a fully diffuse scene, so no separate
	// ceiling is needed for this material.
	constexpr double kMaxPlausibleImageMean = 2.0;
	const double mean = renderMeanBrightness(kRoughGlassSceneId, width, height,
											  /*numFrames=*/100,
											  /*enableRestirDi=*/true, /*enableRestirGi=*/true);
	ASSERT_GE(mean, 0.0) << "rt_realtime_render_frame failed - see per-frame failures above";
	EXPECT_LT(mean, kMaxPlausibleImageMean)
		<< "RoughGlass scene's whole-image mean is " << mean
		<< " - a widespread blowup, not ordinary 1-spp NEE/ReSTIR variance.";
}

// Direct regression guard for the MIS-weight bug described in this file's own
// header comment: with a wrong selection_pdf, ReSTIR DI's converged mean
// diverges from classic NEE's (independently correct) converged mean by a
// large, systematic margin - not the ordinary run-to-run noise a correct
// implementation produces between two different (but statistically
// equivalent) estimators of the same quantity. ReSTIR GI is held off in both
// arms so only the DI toggle differs.
TEST(LivePreviewRestirLightBvhTest, RestirOnAndOffAgreeOnMean) {
	const int width = 64, height = 64;
	constexpr int kNumFrames = 150;

	const double restirOnMean = renderMeanBrightness("A1", width, height, kNumFrames,
													   /*enableRestirDi=*/true, /*enableRestirGi=*/false);
	const double restirOffMean = renderMeanBrightness("A1", width, height, kNumFrames,
														/*enableRestirDi=*/false, /*enableRestirGi=*/false);
	ASSERT_GE(restirOnMean, 0.0) << "ReSTIR-on render failed - see per-frame failures above";
	ASSERT_GE(restirOffMean, 0.0) << "ReSTIR-off render failed - see per-frame failures above";
	ASSERT_GT(restirOffMean, 0.0) << "Classic NEE's own converged mean should be a real positive value";

	// A wrong selection_pdf's contribution error compounds through the power
	// heuristic (wf_mis squares the pdf ratio) - this session's own
	// investigation of the bug estimated up to ~100x for a nearby-bright-
	// light case. 50% is generous headroom over the two estimators' own
	// ordinary disagreement at 150 frames of 1 spp each, while still failing
	// fast and hard on any reintroduction of that class of bug.
	const double ratio = restirOnMean / restirOffMean;
	EXPECT_GT(ratio, 0.5) << "ReSTIR DI mean (" << restirOnMean << ") is far below classic NEE's ("
						  << restirOffMean << ") - looks like a light-selection pdf/MIS-weight bug.";
	EXPECT_LT(ratio, 2.0) << "ReSTIR DI mean (" << restirOnMean << ") is far above classic NEE's ("
						  << restirOffMean << ") - looks like a light-selection pdf/MIS-weight bug.";
}
