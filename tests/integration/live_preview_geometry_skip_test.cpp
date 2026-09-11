// live_preview_geometry_skip_test.cpp
//
// Regression test for SceneData::skipExpensiveGeometryLoad (scene_builder.h)
// - the optimization that lets prepareSceneAndCamera() (optix_interface.cpp)
// skip load_obj_triangles_gpu()/load_obj_triangles_mtl_gpu()/
// build_loaded_pbrt_scene()'s own expensive per-triangle work entirely on a
// pure camera-move Live Preview frame (scene_id unchanged, only cam_x/y/z/
// lookat differ), since the GPU already has this scene_id's correct geometry
// uploaded (g_uploaded_scene_id's own skip a few lines further down the same
// call chain) and a freshly-rebuilt host-side SceneData would only be
// discarded unused.
//
// The risk this guards against: if that skip ever accidentally left the GPU
// rendering stale/wrong geometry, or broke camera_params/cameraExtra
// derivation for the NEW camera position, this would show up as either a
// degenerate (black/NaN) image or an image that doesn't actually change when
// the camera moves - both checked below on a real triangle-mesh scene (the
// exact case this optimization targets: G1, Stanford Bunny, 69,451
// triangles via load_obj_triangles_gpu()).

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern "C" {
	#include "optix_interface.h"
}

namespace {

double MeanBrightness(const std::vector<float>& rgb) {
	double sum = 0.0;
	for (float v : rgb) sum += v;
	return sum / static_cast<double>(rgb.size());
}

} // namespace

TEST(LivePreviewGeometrySkipTest, CameraMoveOnUnchangedSceneStillRendersCorrectly) {
	const int width = 64, height = 64;
	const size_t numChannels = static_cast<size_t>(width) * height * 3;
	std::vector<float> rgbA(numChannels), rgbB(numChannels);

	// First call for this scene_id in this process - scene.skipExpensiveGeometryLoad
	// is false here (or true if an earlier test already rendered G1 - either
	// way, this call's own output must be a real, correctly-lit render).
	ASSERT_TRUE(rt_realtime_render_frame(
		"G1", width, height, /*samples_per_pixel=*/4, /*max_depth=*/5,
		0.0, 3.0, 7.0,
		/*has_custom_lookat=*/true, 0.0, 1.5, 0.0,
		/*denoise=*/false, /*denoise_blend=*/0.0,
		/*out_world_pos_buffer=*/nullptr, /*out_camera_basis=*/nullptr,
		rgbA.data(), /*enable_svgf=*/false));

	// Second call, SAME scene_id, DIFFERENT camera - this is exactly the
	// call that now sets scene.skipExpensiveGeometryLoad=true internally
	// (g_uploaded_scene_id already matches "G1" from the call above).
	ASSERT_TRUE(rt_realtime_render_frame(
		"G1", width, height, /*samples_per_pixel=*/4, /*max_depth=*/5,
		4.0, 3.5, 5.0,
		/*has_custom_lookat=*/true, 0.0, 1.5, 0.0,
		/*denoise=*/false, /*denoise_blend=*/0.0,
		/*out_world_pos_buffer=*/nullptr, /*out_camera_basis=*/nullptr,
		rgbB.data(), /*enable_svgf=*/false));

	for (size_t i = 0; i < numChannels; ++i) {
		ASSERT_FALSE(std::isnan(rgbA[i])) << "NaN in first render at channel " << i;
		ASSERT_FALSE(std::isinf(rgbA[i])) << "Inf in first render at channel " << i;
		ASSERT_FALSE(std::isnan(rgbB[i])) << "NaN in second (geometry-skip) render at channel " << i;
		ASSERT_FALSE(std::isinf(rgbB[i])) << "Inf in second (geometry-skip) render at channel " << i;
	}

	// Neither render should be degenerately black - a bunny lit by an
	// overhead area light, viewed from two different front-ish angles,
	// should have real, non-trivial average brightness in both cases.
	const double meanA = MeanBrightness(rgbA);
	const double meanB = MeanBrightness(rgbB);
	EXPECT_GT(meanA, 1e-4) << "First render is degenerately dark/black (mean=" << meanA << ")";
	EXPECT_GT(meanB, 1e-4) << "Second (geometry-skip) render is degenerately dark/black (mean=" << meanB
		<< ") - the geometry-skip optimization may have left the GPU with no real geometry to hit";

	// The two camera positions are different enough (and the bunny/ground
	// are asymmetric enough) that the two renders must differ - if the skip
	// optimization somehow froze the output (e.g. cameraExtra not actually
	// updated for the second call), this would catch it as a near-identical
	// image despite the moved camera.
	double sumAbsDiff = 0.0;
	for (size_t i = 0; i < numChannels; ++i) sumAbsDiff += std::abs(rgbA[i] - rgbB[i]);
	const double meanAbsDiff = sumAbsDiff / static_cast<double>(numChannels);
	EXPECT_GT(meanAbsDiff, 1e-4)
		<< "Moving the camera produced a near-identical image (mean abs diff=" << meanAbsDiff
		<< ") - camera_params/cameraExtra may not be updating correctly when "
		<< "scene.skipExpensiveGeometryLoad skips the geometry rebuild.";
}
