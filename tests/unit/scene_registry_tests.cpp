/**
 * @file scene_registry_tests.cpp
 * @brief Unit tests for the pbrt-v4-style scene registry
 *
 * Covers:
 *  - Registry completeness (count, IDs, metadata fields)
 *  - find_scene() lookup by id (hit + miss)
 *  - cpu_scene_* C API correctness and out-of-range guards
 *  - Builder callback smoke-tests (each scene builds without crash)
 *  - Light callbacks: Cornell-family scenes return non-empty lights
 *  - GUI mirror count matches registry (so the two tables can't drift silently)
 */

#include <gtest/gtest.h>
#include <string>
#include <set>

// C++ registry (available to the test project via TheRestOfYourLife include path)
#include "scene_registry.h"

// C API (declared with C linkage in cpu_interface.h)
extern "C" {
#include "cpu_interface.h"
}

// ===========================================================================
// Registry structure tests
// ===========================================================================

TEST(SceneRegistryTest, RegistryIsNonEmpty) {
	EXPECT_GT(scene_count(), 0);
}

TEST(SceneRegistryTest, RegistryHasExpectedCount) {
	// We currently register 38 scenes (ids 0-37).
	// This test will fail if a scene is accidentally added or removed -
	// update this count (and kGuiSceneCount below) when that's intentional.
	EXPECT_EQ(scene_count(), 38);
}

TEST(SceneRegistryTest, AllIDsAreUnique) {
	std::set<int> seen;
	for (const auto& s : get_scene_registry()) {
		EXPECT_TRUE(seen.insert(s.id).second)
			<< "Duplicate scene id: " << s.id;
	}
}

TEST(SceneRegistryTest, IDsAreContiguousFromZero) {
	// IDs should be 0..N-1 so the GUI combobox index == scene id.
	int n = scene_count();
	for (int i = 0; i < n; ++i) {
		const SceneDescriptor* s = find_scene(i);
		ASSERT_NE(s, nullptr) << "Missing scene id: " << i;
		EXPECT_EQ(s->id, i);
	}
}

TEST(SceneRegistryTest, AllNamesAreNonEmpty) {
	for (const auto& s : get_scene_registry()) {
		EXPECT_NE(s.name, nullptr);
		EXPECT_GT(std::string(s.name).size(), 0u) << "Empty name for id " << s.id;
	}
}

TEST(SceneRegistryTest, AllDescriptionsAreNonEmpty) {
	for (const auto& s : get_scene_registry()) {
		EXPECT_NE(s.description, nullptr);
		EXPECT_GT(std::string(s.description).size(), 0u)
			<< "Empty description for id " << s.id;
	}
}

TEST(SceneRegistryTest, AllPerformanceStringsAreValid) {
	static const std::set<std::string> kValid = {"Fast", "Medium", "Slow", "Very Slow"};
	for (const auto& s : get_scene_registry()) {
		EXPECT_NE(s.performance, nullptr);
		EXPECT_GT(kValid.count(s.performance), 0u)
			<< "Unexpected performance string '" << s.performance
			<< "' for scene id " << s.id;
	}
}

TEST(SceneRegistryTest, AllRecommendedSppArePositive) {
	for (const auto& s : get_scene_registry()) {
		EXPECT_GT(s.recommended_spp, 0) << "Bad spp for id " << s.id;
	}
}

TEST(SceneRegistryTest, AllBuildWorldCallbacksAreSet) {
	for (const auto& s : get_scene_registry()) {
		EXPECT_TRUE(static_cast<bool>(s.build_world))
			<< "Null build_world for id " << s.id;
	}
}

TEST(SceneRegistryTest, AllBuildLightsCallbacksAreSet) {
	for (const auto& s : get_scene_registry()) {
		EXPECT_TRUE(static_cast<bool>(s.build_lights))
			<< "Null build_lights for id " << s.id;
	}
}

// ===========================================================================
// find_scene() tests
// ===========================================================================

TEST(FindSceneTest, FindsAllRegisteredScenes) {
	for (const auto& s : get_scene_registry()) {
		const SceneDescriptor* found = find_scene(s.id);
		ASSERT_NE(found, nullptr) << "find_scene failed for id " << s.id;
		EXPECT_EQ(found->id, s.id);
	}
}

TEST(FindSceneTest, ReturnsNullForUnknownId) {
	EXPECT_EQ(find_scene(-1), nullptr);
	EXPECT_EQ(find_scene(9999), nullptr);
	EXPECT_EQ(find_scene(scene_count()), nullptr); // one past the end
}

TEST(FindSceneTest, CorrectNameLookup) {
	const SceneDescriptor* s = find_scene(0);
	ASSERT_NE(s, nullptr);
	EXPECT_STREQ(s->name, "Cornell Box");
}

TEST(FindSceneTest, EarthSceneRequiresFiles) {
	const SceneDescriptor* s = find_scene(3);
	ASSERT_NE(s, nullptr);
	EXPECT_TRUE(s->requires_files);
}

TEST(FindSceneTest, CornellBoxIsGpuCompatible) {
	const SceneDescriptor* s = find_scene(0);
	ASSERT_NE(s, nullptr);
	EXPECT_TRUE(s->gpu_compatible);
}

TEST(FindSceneTest, BouncingSpheresIsGpuCompatible) {
	// Scene 1 uses moving spheres (motion blur) - gpu/optix/scene_builder.cpp's
	// build_bouncing_spheres() + OptiXRenderer::buildScene()'s sceneHasMotion_
	// detection give it real OptiX native motion blur (SphereData::center1,
	// GAS motion keys, optixGetRayTime() interpolation in
	// optix_intersection_sphere.h) - see optix_types.h's SphereData comment.
	const SceneDescriptor* s = find_scene(1);
	ASSERT_NE(s, nullptr);
	EXPECT_TRUE(s->gpu_compatible);
}

TEST(FindSceneTest, PbrtV4ScenesAreGpuCompatible) {
	// Scenes 10-17 all have GPU implementations in scene_builder.cpp
	for (int id : {10, 11, 12, 13, 14, 15, 16, 17}) {
		const SceneDescriptor* s = find_scene(id);
		ASSERT_NE(s, nullptr) << "Missing scene id: " << id;
		EXPECT_TRUE(s->gpu_compatible) << "Scene " << id << " should be gpu_compatible";
	}
}

TEST(FindSceneTest, PunctualLightScenesAreGpuCompatible) {
	// Scenes 25-27 (Spotlight/Distant/Point Cornell) have GPU implementations
	// in scene_builder.cpp via build_spotlight_cornell_gpu/build_distant_light_cornell_gpu/
	// build_point_light_cornell_gpu + PunctualLightGPU NEE in the Lambertian
	// case of optix_intersection_{sphere,quad}.h. Scenes 28-29 (Goniometric/
	// Projection Cornell) extend the same PunctualLightGPU NEE path with two
	// image-based light kinds (build_goniometric_cornell_gpu/
	// build_projection_cornell_gpu, GoniometricLightGPU/ProjectionLightGPU).
	for (int id : {25, 26, 27, 28, 29}) {
		const SceneDescriptor* s = find_scene(id);
		ASSERT_NE(s, nullptr) << "Missing scene id: " << id;
		EXPECT_TRUE(s->gpu_compatible) << "Scene " << id << " should be gpu_compatible";
	}
}

TEST(FindSceneTest, NonDefaultCameraScenesAreGpuCompatible) {
	// Scene 9 (RoughMetalSpheres) already had a working GPU handler but a
	// stale gpu_compatible=false flag. Scenes 22/32/33 add the three
	// non-default GPU camera models (GpuCameraParams/CameraKind in
	// optix_types.h): 22 DepthOfField (thin-lens perspective DOF), 32
	// OrthographicCamera, 33 SphericalCamera (equirectangular) - see
	// generate_primary_ray in optix_device_helpers.h (recursive path) and
	// wf_generate_primary_ray in wavefront_kernels.cu (wavefront path).
	for (int id : {9, 22, 32, 33}) {
		const SceneDescriptor* s = find_scene(id);
		ASSERT_NE(s, nullptr) << "Missing scene id: " << id;
		EXPECT_TRUE(s->gpu_compatible) << "Scene " << id << " should be gpu_compatible";
	}
}

TEST(FindSceneTest, BackgroundColorScenesAreGpuCompatible) {
	// Scenes 24 (HdriSky) and 35 (PortalInfiniteLight) both actually use a
	// flat-color sky_light on the CPU side (scenes_advanced.h's
	// build_hdri_sky()/build_portal_sky() - the importance-sampled-image
	// machinery in src/shared/image_infinite_light.h is unused dead code,
	// never wired to any scene), so the GPU port only needed a constant
	// background color for missed rays (GpuCameraParams::backgroundColor in
	// optix_types.h, consumed by optix_miss.h and wavefront_kernels.cu's
	// accumulate_miss), not a full environment-map sampler.
	for (int id : {24, 35}) {
		const SceneDescriptor* s = find_scene(id);
		ASSERT_NE(s, nullptr) << "Missing scene id: " << id;
		EXPECT_TRUE(s->gpu_compatible) << "Scene " << id << " should be gpu_compatible";
	}
}

TEST(FindSceneTest, VolumetricMediumScenesAreGpuCompatible) {
	// Scenes 7 (CornellSmoke), 30 (HomogeneousMedium), 31 (CloudMedium) all use
	// the CPU's closed-form constant_medium (Beer-Lambert free-path sampling +
	// Henyey-Greenstein phase function) - the heterogeneous grid/Perlin-noise
	// density machinery those CPU scenes construct is unused dead code (see
	// cloud_medium.h/grid_medium.h), so the GPU port only needed the simple
	// homogeneous case: MaterialType::Medium in optix_types.h, reusing sphere
	// intersection to find entry/exit roots (optix_intersection_sphere.h /
	// __closesthit__wf_sphere in wavefront_programs.cu) and
	// sample_henyey_greenstein/wf_sample_henyey_greenstein for the scatter
	// direction. Scene 7's two rotated boxes are approximated as spheres.
	for (int id : {7, 30, 31}) {
		const SceneDescriptor* s = find_scene(id);
		ASSERT_NE(s, nullptr) << "Missing scene id: " << id;
		EXPECT_TRUE(s->gpu_compatible) << "Scene " << id << " should be gpu_compatible";
	}
}

TEST(FindSceneTest, BilinearPatchSceneIsGpuCompatible) {
	// Scene 23's two patches are genuinely curved (non-planar) ruled surfaces
	// (verified against src/shared/bilinear_patch.h's corner coordinates - the
	// four corners are off-plane by orders of magnitude, not a numerical-
	// tolerance edge case), so unlike scene 7's medium boxes this needed a
	// real bilinear-surface intersection routine (quadratic solve, Ramsey et
	// al. 2004 / pbrt-v4 IntersectBilinearPatch) as its own GPU geometry type
	// - see optix_intersection_bilinear_patch.h and BilinearPatchData in
	// optix_types.h - rather than an approximation with existing shapes.
	const SceneDescriptor* s = find_scene(23);
	ASSERT_NE(s, nullptr);
	EXPECT_TRUE(s->gpu_compatible);
}

TEST(FindSceneTest, HairFibersSceneIsGpuCompatible) {
	// Scene 19 turned out not to need any new GPU geometry type at all: its
	// "hair fibers" are MaterialType::Hair (Marschner/Chiang fiber scattering,
	// src/shared/bxdfs_hair.h's HairBxDF, already CPU_GPU-tagged) applied
	// directly to 5 ordinary spheres, using the shading normal as a fiber-
	// tangent proxy - matching src/TheRestOfYourLife/hair_material.h's own
	// simplification exactly (src/shared/shapes.h's CurveShape, literal
	// fiber-strand geometry, is unused dead code, never wired to any scene).
	const SceneDescriptor* s = find_scene(19);
	ASSERT_NE(s, nullptr);
	EXPECT_TRUE(s->gpu_compatible);
}

TEST(FindSceneTest, MeasuredBrdfSceneIsGpuCompatible) {
	// Scene 34's "measured BRDF" is a misnomer at the CPU level, not just the
	// GPU level: src/TheRestOfYourLife/scenes_advanced.h's measured_material::
	// scatter() builds a MeasuredBRDFData member but never reads it - it's
	// byte-for-byte a Lambertian material (cosine-hemisphere sampling, flat
	// tint attenuation). The real pbrt-v4 MeasuredBxDF importance-sampling
	// chain (src/shared/measured_bxdf.h + piecewise_linear_2d.h) is fully
	// implemented and unit-tested elsewhere in this codebase but never wired
	// to this scene, so the GPU port matches actual CPU behavior with plain
	// MaterialType::Lambertian rather than porting unused tensor-BRDF math.
	const SceneDescriptor* s = find_scene(34);
	ASSERT_NE(s, nullptr);
	EXPECT_TRUE(s->gpu_compatible);
}

TEST(FindSceneTest, RealisticCameraSceneIsGpuCompatible) {
	// Scene 36's pbrt-v4 RealisticCamera (src/shared/cameras.h) is ported by
	// having gpu/optix/scene_builder.cpp directly instantiate a host-side
	// RealisticCamera<float> at scene-build time - the expensive one-time
	// precomputes (FocusThickLens, BoundExitPupil) reuse the CPU C++ class
	// as-is rather than being re-implemented in CUDA. Only the per-ray hot
	// path (film-plane mapping, exit-pupil sampling, per-element Snell's-law
	// trace) is ported to device code, in both the recursive
	// (optix_device_helpers.h) and wavefront (wavefront_kernels.cu) strategies.
	const SceneDescriptor* s = find_scene(36);
	ASSERT_NE(s, nullptr);
	EXPECT_TRUE(s->gpu_compatible);
}

TEST(FindSceneTest, TriangleMeshSceneIsGpuCompatible) {
	// Scene 37 is new this session (not an existing CPU scene ported to GPU
	// like the other 10 gaps): src/TheRestOfYourLife/triangle.h/mesh.h's real
	// watertight Moller-Trumbore triangle intersection existed but was never
	// wired into any scene, so there was no CPU-vs-GPU parity gap to close in
	// the usual sense - both a CPU builder (build_triangle_mesh_scene) and a
	// GPU port (a new TriangleData custom-primitive geometry type, mirroring
	// the sphere/quad/bilinear-patch pattern) were added together. The scene
	// is a procedurally-generated icosahedron (no external .obj file needed).
	const SceneDescriptor* s = find_scene(37);
	ASSERT_NE(s, nullptr);
	EXPECT_TRUE(s->gpu_compatible);
}

// ===========================================================================
// C API tests
// ===========================================================================

TEST(CpuSceneApiTest, CountMatchesCppRegistry) {
	EXPECT_EQ(cpu_scene_count(), scene_count());
}

TEST(CpuSceneApiTest, IdByIndexMatchesCppRegistry) {
	for (int i = 0; i < cpu_scene_count(); ++i) {
		EXPECT_EQ(cpu_scene_id(i), get_scene_registry()[i].id)
			<< "Mismatch at index " << i;
	}
}

TEST(CpuSceneApiTest, NameByIndexMatchesCppRegistry) {
	for (int i = 0; i < cpu_scene_count(); ++i) {
		EXPECT_STREQ(cpu_scene_name(i), get_scene_registry()[i].name)
			<< "Name mismatch at index " << i;
	}
}

TEST(CpuSceneApiTest, DescriptionByIndexMatchesCppRegistry) {
	for (int i = 0; i < cpu_scene_count(); ++i) {
		EXPECT_STREQ(cpu_scene_description(i), get_scene_registry()[i].description)
			<< "Description mismatch at index " << i;
	}
}

TEST(CpuSceneApiTest, PerformanceByIndexMatchesCppRegistry) {
	for (int i = 0; i < cpu_scene_count(); ++i) {
		EXPECT_STREQ(cpu_scene_performance(i), get_scene_registry()[i].performance)
			<< "Performance mismatch at index " << i;
	}
}

TEST(CpuSceneApiTest, RecommendedSppByIndexMatchesCppRegistry) {
	for (int i = 0; i < cpu_scene_count(); ++i) {
		EXPECT_EQ(cpu_scene_recommended_spp(i), get_scene_registry()[i].recommended_spp)
			<< "Spp mismatch at index " << i;
	}
}

TEST(CpuSceneApiTest, RequiresFilesByIndexMatchesCppRegistry) {
	for (int i = 0; i < cpu_scene_count(); ++i) {
		int expected = get_scene_registry()[i].requires_files ? 1 : 0;
		EXPECT_EQ(cpu_scene_requires_files(i), expected)
			<< "requires_files mismatch at index " << i;
	}
}

TEST(CpuSceneApiTest, GpuCompatibleByIndexMatchesCppRegistry) {
	for (int i = 0; i < cpu_scene_count(); ++i) {
		int expected = get_scene_registry()[i].gpu_compatible ? 1 : 0;
		EXPECT_EQ(cpu_scene_gpu_compatible(i), expected)
			<< "gpu_compatible mismatch at index " << i;
	}
}

// Out-of-range guards
TEST(CpuSceneApiTest, OutOfRangeIdReturnsMinusOne) {
	EXPECT_EQ(cpu_scene_id(-1), -1);
	EXPECT_EQ(cpu_scene_id(cpu_scene_count()), -1);
}

TEST(CpuSceneApiTest, OutOfRangeNameReturnsEmptyString) {
	EXPECT_STREQ(cpu_scene_name(-1), "");
	EXPECT_STREQ(cpu_scene_name(cpu_scene_count()), "");
}

TEST(CpuSceneApiTest, OutOfRangeSppReturnsZero) {
	// The original implementation returns 100 as the safe default for out-of-range
	EXPECT_EQ(cpu_scene_recommended_spp(-1), 100);
	EXPECT_EQ(cpu_scene_recommended_spp(cpu_scene_count()), 100);
}

// ===========================================================================
// Builder callback smoke tests
// ===========================================================================

TEST(SceneBuilderTest, AllScenesProduceNonEmptyWorld) {
	for (const auto& s : get_scene_registry()) {
		// Skip scenes requiring external files (earthmap.jpg may not be present in CI)
		if (s.requires_files) continue;
		hittable_list world;
		EXPECT_NO_THROW(world = s.build_world())
			<< "build_world threw for id " << s.id;
		EXPECT_GT(world.objects.size(), 0u)
			<< "Empty world for id " << s.id;
	}
}

TEST(SceneBuilderTest, CornellFamilyLightsAreNonEmpty) {
	// Scenes that use Cornell box lights: 0, 7, 10
	for (int id : {0, 7, 10}) {
		const SceneDescriptor* s = find_scene(id);
		ASSERT_NE(s, nullptr);
		hittable_list lights;
		EXPECT_NO_THROW(lights = s->build_lights())
			<< "build_lights threw for id " << id;
		EXPECT_GT(lights.objects.size(), 0u)
			<< "Empty lights for Cornell scene id " << id;
	}
}

TEST(SceneBuilderTest, SkyDummyLightsAreNonEmpty) {
	// All sky-lit scenes should still return a dummy light for PDF sampling
	for (int id : {1, 2, 4, 5, 9}) {
		const SceneDescriptor* s = find_scene(id);
		ASSERT_NE(s, nullptr);
		hittable_list lights;
		EXPECT_NO_THROW(lights = s->build_lights());
		EXPECT_GT(lights.objects.size(), 0u)
			<< "Empty sky dummy lights for id " << id;
	}
}

TEST(SceneBuilderTest, CornellBoxBuildsDetAndRepeatably) {
	// Registry-based determinism: same id always builds same object count
	const SceneDescriptor* s = find_scene(0);
	ASSERT_NE(s, nullptr);
	hittable_list w1 = s->build_world();
	hittable_list w2 = s->build_world();
	EXPECT_EQ(w1.objects.size(), w2.objects.size());
}

// ===========================================================================
// GUI mirror count guard
// ===========================================================================

// The Qt GUI builds its scene dropdown dynamically from get_all_scenes()
// (qt_gui/mainwindow_tabs.cpp createBasicTab()), not a hardcoded array, so it
// can't drift out of sync with the registry on its own. This constant exists
// as a tripwire: if it stops matching scene_count(), something changed the
// registry size and it's worth double-checking the GUI/error-hint text that
// mentions specific scene counts or ID ranges by hand.
TEST(SceneRegistryGuiConsistencyTest, GuiSceneCountMatchesRegistry) {
	constexpr int kGuiSceneCount = 38;
	EXPECT_EQ(scene_count(), kGuiSceneCount)
		<< "Registry size changed -- update kGuiSceneCount here to match.";
}
