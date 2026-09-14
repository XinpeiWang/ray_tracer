// pbrt_dof_override_test.cpp
//
// Regression test for the depth-of-field render-option override (see this
// project's own DOF plan): gpu/optix/scene_builder.cpp's one pbrt-scene
// camera branch (inside build_loaded_pbrt_scene(), reached through
// build_scene()'s new has_dof_override/aperture_override/
// focus_distance_override trailing params) recomputes GpuCameraParams::
// defocus_disk_u/v from the override instead of the scene file's own
// "lensradius"/"focaldistance" Camera directive.
//
// build_scene() is pure host-side C++ (no OptiX/CUDA device calls - see
// gpu_scene_light_count()'s own comment in optix_interface.h, and
// scene_loader_transform_cache_test.cpp's identical use of this fact), so
// this test inspects GpuCameraParams directly - no GPU, no rendering.
//
// Uses the bundled pbrt_scenes/depth-of-field.pbrt example scene (already
// covered end-to-end by pbrt_example_scenes_tests.cpp) via the same
// find_example_scene()-by-stem lookup that file uses, since a Custom Scene's
// numeric id isn't stable enough to hardcode.

#include <gtest/gtest.h>
#include <cmath>
#include <string>

extern "C" {
	#include "optix_interface.h"
}
#include "scene_builder.h"
#include "scene_registry.h"

namespace {

const SceneDescriptor* find_example_scene(const char* stem) {
	for (const auto& s : get_scene_registry()) {
		if (s.category == SceneCategories::CustomScenes && s.name && std::string(s.name) == stem)
			return &s;
	}
	return nullptr;
}

bool isZeroDisk(const GpuCameraParams& extra) {
	return extra.defocus_disk_u.x == 0.0f && extra.defocus_disk_u.y == 0.0f && extra.defocus_disk_u.z == 0.0f &&
		   extra.defocus_disk_v.x == 0.0f && extra.defocus_disk_v.y == 0.0f && extra.defocus_disk_v.z == 0.0f;
}

} // namespace

TEST(PbrtDofOverrideTest, OverrideChangesDefocusDiskFromScenesOwnValue) {
	const SceneDescriptor* s = find_example_scene("depth-of-field");
	if (!s) GTEST_SKIP() << "depth-of-field.pbrt was not discovered - is pbrt_scenes/ present?";

	SceneData sceneBaseline;
	float cameraParamsBaseline[12];
	GpuCameraParams extraBaseline{};
	ASSERT_TRUE(build_scene(s->id.c_str(), 64, 64, sceneBaseline, cameraParamsBaseline,
							 278.0, 278.0, -800.0, &extraBaseline))
		<< "depth-of-field.pbrt failed to build (scene id " << s->id << ")";
	// The scene's own Camera directive sets "lensradius"/"focaldistance"
	// (see pbrt_scenes/depth-of-field.pbrt), so it must already have a
	// nonzero defocus disk with no override applied - otherwise this test
	// couldn't tell "override changed it" from "there was nothing to blur
	// in the first place".
	ASSERT_FALSE(isZeroDisk(extraBaseline))
		<< "depth-of-field.pbrt's own camera produced a zero defocus disk - "
		   "scene file may have changed, or GPU DOF support regressed";

	// A different aperture/focus distance than whatever the scene file
	// itself requests - the override's own two independent knobs.
	SceneData sceneOverridden;
	float cameraParamsOverridden[12];
	GpuCameraParams extraOverridden{};
	const double overrideAperture = extraBaseline.defocus_disk_u.x != 0.0f ? 0.05 : 5.0;  // avoid an accidental match
	const double overrideFocusDistance = 37.5;
	ASSERT_TRUE(build_scene(s->id.c_str(), 64, 64, sceneOverridden, cameraParamsOverridden,
							 278.0, 278.0, -800.0, &extraOverridden,
							 /*force_camera_override=*/false, /*has_custom_lookat=*/false,
							 0.0, 0.0, 0.0,
							 /*has_dof_override=*/true, overrideAperture, overrideFocusDistance));

	EXPECT_NE(extraBaseline.defocus_disk_u.x, extraOverridden.defocus_disk_u.x);
	EXPECT_FALSE(isZeroDisk(extraOverridden))
		<< "override aperture " << overrideAperture << " is nonzero, so the resulting disk must be too";
}

TEST(PbrtDofOverrideTest, ZeroApertureOverrideDisablesDof) {
	const SceneDescriptor* s = find_example_scene("depth-of-field");
	if (!s) GTEST_SKIP() << "depth-of-field.pbrt was not discovered - is pbrt_scenes/ present?";

	SceneData scene;
	float cameraParams[12];
	GpuCameraParams extra{};
	ASSERT_TRUE(build_scene(s->id.c_str(), 64, 64, scene, cameraParams,
							 278.0, 278.0, -800.0, &extra,
							 /*force_camera_override=*/false, /*has_custom_lookat=*/false,
							 0.0, 0.0, 0.0,
							 /*has_dof_override=*/true, /*aperture_override=*/0.0, /*focus_distance_override=*/10.0));

	// An explicit 0.0 aperture override must turn DOF fully off, even
	// though the scene file's own Camera directive requests real blur -
	// this is the "override can REMOVE a scene's own DOF" half of the
	// plan's own "both ADD and change/remove" requirement.
	EXPECT_TRUE(isZeroDisk(extra));
}
