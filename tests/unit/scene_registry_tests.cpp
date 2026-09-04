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
	// We currently compile in 74 scenes (legacy_ids 0-74 with one gap at 53 -
	// D5-D8 added the classic Cornell box rendered by each of D1-D4's camera
	// models for direct comparison, see scene_registry.h's comment above
	// those 4 rows; E3/E4 added two more volume scenes; F3/F4 added
	// Instanced Spheres and Curve Fibers to the Geometry category; G16 (VW
	// Beetle Alt, legacy_id 53) was removed as visually indistinguishable
	// from G15 - see scene_registry.h's G15 entry - leaving that legacy_id
	// permanently unused rather than renumbering G17-G24, matching this
	// registry's existing precedent of stable, non-contiguous ids; H4/H5
	// added Fireplace Room and San Miguel to the LargeScene category; H6-H9
	// added Sibenik Cathedral, Breakfast Room, Salle de Bain, and Gallery).
	// This test will fail if a scene is accidentally added or removed -
	// update this count (and kGuiSceneCount below) when that's intentional.
	//
	// Deliberately the BUILT-IN count, not scene_count(). The full registry
	// also contains whatever .pbrt files happen to be on the machine running
	// the tests, which is not a property of this source tree and must not
	// decide whether the suite passes.
	//
	// 81 -> 112: 31 curated pbrt_scenes/*.pbrt example entries added under
	// their real topic category (Materials/Lights/Cameras/Volumes/Geometry/
	// Models) instead of only the generic auto-discovered "Custom Scenes"
	// bucket - see pbrt_scene_registry::build_curated_pbrt_scene_descriptor()
	// and its call sites in get_builtin_scene_registry().
	//
	// 113 -> 117: I1-I4 added, a new Education category of curated Render
	// Options tab demos (Sampler/Spectral/Exposure+Tone mapping/Denoiser) -
	// each reuses an existing scene's build functions rather than being new
	// geometry, see their own comment block in get_builtin_scene_registry().
	//
	// 117 -> 118: B24 added, a frosted (rough_dielectric) sibling of B23's
	// dispersive prism - closes the "no rough_dielectric dispersion" gap
	// from docs/FEATURE_INVENTORY.md.
	//
	// 118 -> 119: C16 added, exercising the newly-wired "ColorSpace"
	// directive (identical to C10's blackbody-light.pbrt except for one
	// added "ColorSpace rec2020" line) - closes part of the "Accelerator/
	// CoordinateSystem/ColorSpace pbrt directives not parsed" gap from
	// docs/FEATURE_INVENTORY.md.
	//
	// 119 -> 120: D13 added, a Cornell Box panned across the exposure via
	// the newly-wired camera_is_animated/AnimatedTransform path - closes
	// the CPU half of the "no motion blur anywhere" gap from
	// docs/FEATURE_INVENTORY.md (GPU deferred).
	//
	// 120 -> 122: I5/I6 added, extending the Education category to the
	// newly GUI-selectable Integrator dropdown - I5 reuses B3 (Cornell
	// Rough Glass, the one CPU scene verified for --sppm) and I6 reuses A1
	// (Cornell Box, the one scene verified for --bdpt/--mlt).
	//
	// 122 -> 123: E9 added, pbrt's MakeNamedMedium "nanovdb" (a real
	// NanoVDB-format sparse density grid read from an external .nvdb file)
	// under Volumes - CPU only, see scene_registry.h's E9 entry.
	//
	// 123 -> 136: 13 curated pbrt-example entries added (B25-B28, C17-C20,
	// E10, F11-F14) for self-contained pbrt_scenes/*.pbrt files that were
	// previously only discoverable via the generic Custom Scenes tab.
	//
	// 136 -> 140: I7-I10 added, extending the Education category to the
	// render-transport/light-sampling/debug-integrator controls I1-I6
	// didn't cover yet - I7 (RandomWalk/SimplePath vs. the default MIS
	// path tracer) and I9 (Ambient Occlusion) both reuse A1 (Cornell Box);
	// I8 (Uniform/Power/BVH light sampler) is the one genuinely new
	// geometry in this category - a 5-lopsided-power-light variant of the
	// Cornell box (build_light_sampler_comparison(), cornell_box_scene.h) -
	// since no existing scene has enough lights of different power to
	// show a light-sampler difference at all; I10 (Regularize/
	// maxcomponentvalue firefly suppression) reuses B3 (Cornell Rough
	// Glass, the same hard-caustic scene I5 already reuses for SPPM).
	EXPECT_EQ(builtin_scene_count(), 140);
}

TEST(SceneRegistryTest, LoadedScenesAppendAfterTheBuiltInsWithoutDisturbingThem) {
	// The contract the ids depend on: loading scenes from disk may only ADD
	// entries at the end. If a loaded scene could take a lower id, every saved
	// setting and every script that passes a scene number would silently point
	// at a different scene than it did yesterday.
	const auto& all = get_scene_registry();
	ASSERT_GE(all.size(), static_cast<std::size_t>(builtin_scene_count()));

	const auto& builtins = get_builtin_scene_registry();
	for (std::size_t i = 0; i < builtins.size(); ++i) {
		EXPECT_EQ(all[i].id, builtins[i].id);
		EXPECT_STREQ(all[i].name, builtins[i].name);
	}
	// No built-in scene uses category CustomScenes, so loaded scenes start
	// numbering at 1 under CustomScenes's own letter - see
	// pbrt_scene_registry::append()'s user_number comment in
	// scene_registry.h. Derived via letter_for_category() rather than a
	// hardcoded literal (this used to hardcode "J" and broke the moment
	// CustomScenes's own position in kAll shifted for an unrelated new
	// category - the exact class of drift BuiltinIdLetterMatchesItsCategory
	// exists to catch on the builtin side; this is that same fix applied
	// here too).
	const char customScenesLetter = SceneCategories::letter_for_category(SceneCategories::CustomScenes);
	int user_number = 1;
	for (std::size_t i = builtins.size(); i < all.size(); ++i) {
		EXPECT_STREQ(all[i].category, SceneCategories::CustomScenes)
			<< "a scene past the built-ins should be a loaded one";
		EXPECT_EQ(all[i].id, std::string(1, customScenesLetter) + std::to_string(user_number++))
			<< "loaded scene ids must continue the CustomScenes sequence without gaps";
	}
}

TEST(SceneRegistryTest, AllIDsAreUnique) {
	std::set<std::string> seen;
	for (const auto& s : get_scene_registry()) {
		EXPECT_TRUE(seen.insert(s.id).second)
			<< "Duplicate scene id: " << s.id;
	}
}

TEST(SceneRegistryTest, AllLegacyIDsAreUnique) {
	// Pins the fix for a real bug: gpu/optix/scene_builder.cpp's build_scene()
	// switches on legacy_id, so two SceneDescriptors sharing one is not a
	// harmless duplicate the way a repeated `id` string would be - it means
	// GPU silently builds whichever one the switch's `case N:` was written
	// for, regardless of which scene was actually requested. pbrt_scene_registry
	// ::append() used to start its counter at builtin_scene_count() (the
	// builtin array's SIZE), which collided with H9 Gallery's own legacy_id
	// 78 once G16's removal left legacy_id 53 permanently unused (size 78 ==
	// highest id in use, not one past it) - the first scene loaded from a
	// .pbrt file on disk ("I1") got legacy_id 78 too, so GPU rendered
	// Gallery's framed-paintings interior for it instead of falling through
	// to the generic pbrt loader, while CPU (which never switches on
	// legacy_id) rendered the correct file. See pbrt_scene_registry::append()'s
	// legacy_id comment for the fix.
	std::set<int> seen;
	for (const auto& s : get_scene_registry()) {
		EXPECT_TRUE(seen.insert(s.legacy_id).second)
			<< "Duplicate legacy_id " << s.legacy_id << " on scene " << s.id;
	}
}

TEST(SceneRegistryTest, EveryIndexResolvesToAFindableId) {
	// IDs are category letter + number now, not contiguous ints (see
	// scene_registry.h's SceneDescriptor::id comment) - what stays true is
	// that every position in the registry has an id that find_scene() can
	// look back up, which is what cpu_scene_id(index) + find_scene(id)
	// (the GUI's actual enumeration path) depends on.
	int n = scene_count();
	for (int i = 0; i < n; ++i) {
		const std::string& id = get_scene_registry()[i].id;
		const SceneDescriptor* s = find_scene(id);
		ASSERT_NE(s, nullptr) << "Missing scene id: " << id;
		EXPECT_EQ(s->id, id);
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
	// "Unknown" is only ever produced by scenes loaded from a .pbrt file.
	// Nothing in a pbrt header says whether the world behind it holds three
	// triangles or ten million, and the geometry is deliberately not read
	// until the scene is rendered - so any of the four estimates below would
	// be a guess presented as a fact.
	static const std::set<std::string> kValid = {"Fast", "Medium", "Slow",
												"Very Slow", "Unknown"};
	for (const auto& s : get_scene_registry()) {
		EXPECT_NE(s.performance, nullptr);
		EXPECT_GT(kValid.count(s.performance), 0u)
			<< "Unexpected performance string '" << s.performance
			<< "' for scene id " << s.id;
	}
}

// ---------------------------------------------------------------------------
// Categories
// ---------------------------------------------------------------------------
// The Qt GUI builds its scene-browser tabs from SceneCategories::kAll and puts
// each scene under the tab matching its category string. A scene whose
// category is misspelled - or a category constant that no scene uses - both
// fail silently there: the scene simply appears under no tab, or an empty tab
// appears. These tests are what make either loud.

TEST(SceneRegistryTest, AllCategoriesAreKnownConstants) {
	static const std::set<std::string> kValid(
		SceneCategories::kAll, SceneCategories::kAll + SceneCategories::kAllCount);
	for (const auto& s : get_scene_registry()) {
		ASSERT_NE(s.category, nullptr) << "Null category for id " << s.id;
		EXPECT_GT(kValid.count(s.category), 0u)
			<< "Unknown category '" << s.category << "' for scene id " << s.id
			<< " - use a SceneCategories:: constant, not a literal";
	}
}

TEST(SceneRegistryTest, EveryCategoryHasAtLeastOneScene) {
	std::set<std::string> used;
	for (const auto& s : get_scene_registry())
		if (s.category) used.insert(s.category);

	for (std::size_t i = 0; i < SceneCategories::kAllCount; ++i) {
		// CustomScenes is populated from .pbrt files found on disk, so it is
		// legitimately empty on a machine with no scene collection installed -
		// including every CI machine. Every other category is compiled in, so
		// an empty one there really is the bug this test is looking for.
		if (std::string(SceneCategories::kAll[i]) == SceneCategories::CustomScenes)
			continue;
		EXPECT_GT(used.count(SceneCategories::kAll[i]), 0u)
			<< "Category '" << SceneCategories::kAll[i]
			<< "' has no scenes - it would render as an empty tab in the GUI";
	}
}

// SceneCategories::letter_for_category() derives a category's id letter
// from its POSITION in kAll (see scene_descriptor.h) - but
// every builtin scene's id is a hand-typed literal like "B10", not computed
// through that function (only the CustomScenes discovery loop actually calls
// it - see scene_registry.h). Nothing previously checked that a builtin
// id's letter still matched its category's position: reordering kAll for a
// cosmetic tab-order change, or a copy-paste typo giving a scene the wrong
// id prefix, would silently reassign what every id under one or more
// categories means, with every other registry test still passing (they
// check ids are unique and categories are known constants, not that the
// two agree with each other).
TEST(SceneRegistryTest, BuiltinIdLetterMatchesItsCategory) {
	for (const auto& s : get_builtin_scene_registry()) {
		ASSERT_FALSE(s.id.empty()) << "Empty id in builtin registry";
		const char expected = SceneCategories::letter_for_category(s.category);
		EXPECT_EQ(s.id[0], expected)
			<< "Scene '" << s.name << "' has id " << s.id << " (letter '" << s.id[0]
			<< "') but its category '" << s.category << "' maps to letter '"
			<< expected << "' - the id's category letter and its declared "
			<< "category have drifted apart.";
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
	EXPECT_EQ(find_scene(""), nullptr);
	EXPECT_EQ(find_scene("Z9999"), nullptr);
	EXPECT_EQ(find_scene("NotARealId"), nullptr);
}

TEST(FindSceneTest, CorrectNameLookup) {
	const SceneDescriptor* s = find_scene("A1");
	ASSERT_NE(s, nullptr);
	EXPECT_STREQ(s->name, "Cornell Box");
}

TEST(FindSceneTest, EarthSceneRequiresFiles) {
	const SceneDescriptor* s = find_scene("A4");
	ASSERT_NE(s, nullptr);
	EXPECT_TRUE(s->requires_files);
}

TEST(FindSceneTest, CornellBoxIsGpuCompatible) {
	const SceneDescriptor* s = find_scene("A1");
	ASSERT_NE(s, nullptr);
	EXPECT_TRUE(s->gpu_compatible);
}

TEST(FindSceneTest, BouncingSpheresIsGpuCompatible) {
	// Scene A2 uses moving spheres (motion blur) - gpu/optix/scene_builder.cpp's
	// build_bouncing_spheres() + OptiXRenderer::buildScene()'s sceneHasMotion_
	// detection give it real OptiX native motion blur (SphereData::center1,
	// GAS motion keys, optixGetRayTime() interpolation in
	// optix_intersection_sphere.h) - see optix_types.h's SphereData comment.
	const SceneDescriptor* s = find_scene("A2");
	ASSERT_NE(s, nullptr);
	EXPECT_TRUE(s->gpu_compatible);
}

TEST(FindSceneTest, PbrtV4ScenesAreGpuCompatible) {
	// Scenes B2-B9 (old flat ids 10-17) all have GPU implementations in scene_builder.cpp
	for (const std::string& id : {"B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"}) {
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
	for (const std::string& id : {"C2", "C3", "C4", "C5", "C6"}) {
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
	for (const std::string& id : {"B1", "D1", "D2", "D3"}) {
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
	for (const std::string& id : {"C1", "C7"}) {
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
	for (const std::string& id : {"A8", "E1", "E2"}) {
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
	const SceneDescriptor* s = find_scene("F1");
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
	const SceneDescriptor* s = find_scene("B11");
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
	const SceneDescriptor* s = find_scene("B14");
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
	const SceneDescriptor* s = find_scene("D4");
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
	const SceneDescriptor* s = find_scene("F2");
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

TEST(CpuSceneApiTest, CategoryByIdMatchesCppRegistry) {
	// The GUI reads categories only through this C API (via scene_metadata.dll),
	// never from the C++ registry directly, so the bridge needs its own check.
	for (const auto& s : get_scene_registry()) {
		EXPECT_STREQ(cpu_scene_category_by_id(s.id.c_str()), s.category)
			<< "Category mismatch for scene id " << s.id;
	}
}

TEST(CpuSceneApiTest, OutOfRangeCategoryReturnsEmptyString) {
	EXPECT_STREQ(cpu_scene_category_by_id("NotARealId"), "");
	EXPECT_STREQ(cpu_scene_category_by_id(""), "");
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
TEST(CpuSceneApiTest, OutOfRangeIdReturnsEmptyString) {
	EXPECT_STREQ(cpu_scene_id(-1), "");
	EXPECT_STREQ(cpu_scene_id(cpu_scene_count()), "");
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
	// Scenes that use Cornell box lights: A1, A8, B2 (old flat ids 0, 7, 10)
	for (const std::string& id : {"A1", "A8", "B2"}) {
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
	for (const std::string& id : {"A2", "A3", "A5", "A6", "B1"}) {
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
	const SceneDescriptor* s = find_scene("A1");
	ASSERT_NE(s, nullptr);
	hittable_list w1 = s->build_world();
	hittable_list w2 = s->build_world();
	EXPECT_EQ(w1.objects.size(), w2.objects.size());
}

// ===========================================================================
// GUI mirror count guard
// ===========================================================================

// The Qt GUI builds its scene dropdown dynamically from
// SceneMetadataClient::sceneCount()/sceneName() (qt_gui/mainwindow_tabs.cpp
// createBasicTab()), which query scene_metadata.dll -> this registry live,
// not a hardcoded array, so it can't drift out of sync with the registry
// on its own. This constant exists as a tripwire: if it stops matching
// scene_count(), something changed the registry size and it's worth
// double-checking the GUI/error-hint text that mentions specific scene
// counts or ID ranges by hand.
TEST(SceneRegistryGuiConsistencyTest, GuiSceneCountMatchesRegistry) {
	constexpr int kGuiSceneCount = 140;
	EXPECT_EQ(builtin_scene_count(), kGuiSceneCount)
		<< "Registry size changed -- update kGuiSceneCount here to match.";
}
