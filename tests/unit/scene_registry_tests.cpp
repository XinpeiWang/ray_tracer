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
#include <filesystem>
#include <string>
#include <map>
#include <regex>
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
	//
	// 140 -> 142: H13/H14 added, two real pbrt-v4-scenes bundles
	// (Contemporary Bathroom, Barcelona Pavilion) curated under Large
	// Scenes via the new build_curated_external_pbrt_scene_descriptor()
	// (scene_registry.h) - requires_files=true, unlike every other
	// pbrt-backed curated entry, since these are full downloaded bundles
	// rather than small git-tracked example files.
	// 142 -> 145: H15/H16/H17 added, three more real pbrt-v4-scenes
	// bundles (Subsurface Dragon, Ganesha, Sports Car) curated the same
	// way as H13/H14.
	// 145 -> 146: H18 added (Zero Day) - the first of these curated
	// external-pbrt entries that required a genuinely new download
	// rather than reusing an already-local fixture.
	// 146 -> 149: H19/H20/H21 added (Crown, Villa, Transparent Machines),
	// three more freshly-downloaded pbrt-v4-scenes bundles.
	//
	// 149 -> 150: E11 added (Thin Dielectric Medium pbrt example), closing
	// part of the "rough/thin dielectric + medium" GPU gap from docs/
	// PBRT_SUPPORT.md - Material "thindielectric" fused with
	// MediumInterface now reaches the same MaterialType::DielectricMedium
	// the smooth-dielectric fusion case uses, see pbrt_gpu_builder.h's
	// mediumMaterialIndex().
	//
	// 150 -> 151: E12 added (Rough Dielectric Medium pbrt example), closing
	// the rest of that gap - a frosted (rough) Material "dielectric" fused
	// with MediumInterface now reaches a real GGX microfacet DielectricMedium
	// surface with real glossy NEE, see mediumMaterialIndex()'s own comment
	// and rough_dielectric_scatter_and_nee() (optix_device_helpers.h).
	EXPECT_EQ(builtin_scene_count(), 151);
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
	// Scenes found on disk come after the built-ins (the built-in ones use no Custom Scenes or Test Scenes category) and are numbered within their own
	// category's letter, in the order they were found: K1, K2, ... for Custom Scenes, L1, L2, ... for Test Scenes. Derived via letter_for_category()
	// rather than hardcoded literals, so a new category cannot make this stale (see BuiltinIdLetterMatchesItsCategory).
	std::map<char, int> number;
	for (std::size_t i = builtins.size(); i < all.size(); ++i) {
		const std::string category = all[i].category;
		EXPECT_TRUE(category == SceneCategories::CustomScenes || category == SceneCategories::Tests || category == SceneCategories::MyScenes)
			<< "a scene past the built-ins should be a loaded one: " << all[i].id << " is in " << category;
		const char letter = SceneCategories::letter_for_category(all[i].category);
		EXPECT_EQ(all[i].id, std::string(1, letter) + std::to_string(++number[letter]))
			<< "loaded scene ids must continue their category's sequence without gaps";
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
		if (std::string(SceneCategories::kAll[i]) == SceneCategories::CustomScenes || std::string(SceneCategories::kAll[i]) == SceneCategories::Tests ||
		    std::string(SceneCategories::kAll[i]) == SceneCategories::MyScenes)
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

TEST(SceneRegistryTest, AllRecommendedExposureArePositive) {
	for (const auto& s : get_scene_registry()) {
		EXPECT_GT(s.recommended_exposure, 0.0) << "Bad exposure for id " << s.id;
	}
}

TEST(SceneRegistryTest, RecommendedCameraPathIsNeverEmpty) {
	for (const auto& s : get_scene_registry()) {
		EXPECT_STRNE(recommended_camera_path_for(s.id), "")
			<< "Bad camera path for id " << s.id;
	}
}

TEST(SceneRegistryTest, RecommendedCameraPathCuratedChoices) {
	// H16 is the one individual override - confirmed by rendering (not just
	// inferred from category) to have a real, subject-centered LookAt
	// target, which is what makes an orbit-family path (showcase) safe for
	// it. See recommended_camera_path_for()'s own comment for why H15/H17/
	// H19/H21 do NOT get the same treatment despite looking like similarly
	// strong candidates on paper - each has a LookAt that is either too
	// close to the camera to be a real subject location, real but not
	// centered on the subject, or not declared via LookAt at all.
	EXPECT_STREQ(recommended_camera_path_for("H16"), "showcase");
	// ...the category-letter default (architectural walkthroughs, plus
	// every Large Scene not individually confirmed safe for an
	// orbit-family path -> tour) for every other Large Scene...
	EXPECT_STREQ(recommended_camera_path_for("H1"), "tour");
	EXPECT_STREQ(recommended_camera_path_for("H13"), "tour");
	EXPECT_STREQ(recommended_camera_path_for("H15"), "tour");
	EXPECT_STREQ(recommended_camera_path_for("H17"), "tour");
	EXPECT_STREQ(recommended_camera_path_for("H19"), "tour");
	EXPECT_STREQ(recommended_camera_path_for("H20"), "tour");
	EXPECT_STREQ(recommended_camera_path_for("H21"), "tour");

	// A-G/I/J: unlike H, every one of these scenes' lookat data is already
	// established as trustworthy (hand-typed CameraConfig literals, or this
	// project's own small git-tracked pbrt example files - see
	// recommended_camera_path_for()'s own comment), so shape alone decides
	// the path. Single centered subject -> showcase.
	EXPECT_STREQ(recommended_camera_path_for("A3"), "showcase");   // CheckeredSpheres
	EXPECT_STREQ(recommended_camera_path_for("A4"), "showcase");   // Earth
	EXPECT_STREQ(recommended_camera_path_for("A5"), "showcase");   // PerlinSpheres
	EXPECT_STREQ(recommended_camera_path_for("B11"), "showcase");  // HairFibers (sphere cluster)
	EXPECT_STREQ(recommended_camera_path_for("B14"), "showcase");  // MeasuredBrdf (sphere cluster)
	EXPECT_STREQ(recommended_camera_path_for("B20"), "showcase");  // HairMaterialPbrtExample
	EXPECT_STREQ(recommended_camera_path_for("G6"), "showcase");   // UtahTeapot - G's own category default
	EXPECT_STREQ(recommended_camera_path_for("G25"), "showcase");  // KillerooSimplePbrtExample

	// Comparison row (a parameter sweep across several small objects/panels)
	// -> linear, so a straight sweep keeps every item in frame rather than
	// an orbit-family path pivoting tightly around one point.
	EXPECT_STREQ(recommended_camera_path_for("B1"), "linear");    // RoughMetalSpheres
	EXPECT_STREQ(recommended_camera_path_for("B10"), "linear");   // PrincipledShowcase
	EXPECT_STREQ(recommended_camera_path_for("B25"), "linear");   // GlassPresetsPbrtExample
	EXPECT_STREQ(recommended_camera_path_for("C8"), "linear");    // PunctualLightsPbrtExample
	EXPECT_STREQ(recommended_camera_path_for("C10"), "linear");   // BlackbodyLightPbrtExample
	EXPECT_STREQ(recommended_camera_path_for("C16"), "linear");   // ColorSpaceBlackbodyPbrtExample
	EXPECT_STREQ(recommended_camera_path_for("C18"), "linear");   // LightPowerParameterPbrtExample
	EXPECT_STREQ(recommended_camera_path_for("D1"), "linear");    // DepthOfField
	EXPECT_STREQ(recommended_camera_path_for("D4"), "linear");    // RealisticCamera
	EXPECT_STREQ(recommended_camera_path_for("E3"), "linear");    // DielectricMediumShowcase
	EXPECT_STREQ(recommended_camera_path_for("E10"), "linear");   // CameraMediumPbrtExample
	EXPECT_STREQ(recommended_camera_path_for("F3"), "linear");    // InstancedSpheres
	EXPECT_STREQ(recommended_camera_path_for("F14"), "linear");   // ConeParaboloidGalleryPbrtExample
	EXPECT_STREQ(recommended_camera_path_for("G12"), "linear");   // TrophyRoom
	EXPECT_STREQ(recommended_camera_path_for("J4"), "linear");    // TextureEncodingWrapInvertPbrtExample
	EXPECT_STREQ(recommended_camera_path_for("J5"), "linear");    // ProceduralTextureGalleryPbrtExample

	// Volumes (category default: spiral) - a slow push-in orbit for
	// nebula/fog-like subjects, except E1 (a Cornell box, "tour" like every
	// other enclosed room).
	EXPECT_STREQ(recommended_camera_path_for("E1"), "tour");      // HomogeneousMedium (Cornell box)
	EXPECT_STREQ(recommended_camera_path_for("E2"), "spiral");    // CloudMedium
	EXPECT_STREQ(recommended_camera_path_for("E8"), "spiral");    // UniformGridMediumPbrtExample

	// Geometry (category default: showcase) - except F1, a Cornell box.
	EXPECT_STREQ(recommended_camera_path_for("F1"), "tour");      // BilinearPatchScene (Cornell box)
	EXPECT_STREQ(recommended_camera_path_for("F5"), "showcase");  // PlymeshUvPbrtExample

	// Cornell-box-style enclosed rooms use "tour", NOT the global orbit (it circles outside the opaque walls: black frames) - a
	// slow full rotation around a small box viewed from its open front is
	// the classic beauty shot for this shape, unlike H's large interiors.
	EXPECT_STREQ(recommended_camera_path_for("A1"), "tour");      // CornellBox itself
	EXPECT_STREQ(recommended_camera_path_for("B2"), "orbit");     // CornellRoughMetal
	EXPECT_STREQ(recommended_camera_path_for("I1"), "tour");      // SamplerComparison (=A1)

	// ...and the global default (orbit) for a category with no curation at
	// all and for an id that doesn't exist.
	EXPECT_STREQ(recommended_camera_path_for("NotARealId"), "orbit");
	EXPECT_STREQ(recommended_camera_path_for(""), "orbit");
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

// A .pbrt file that a curated entry already lists is not listed again under Custom Scenes: it used to be (the same scene twice, once with a real
// name, category and description and once under its file name, which made most of the Custom Scenes list a copy of the rest). Curated entries may
// share a file with one another on purpose (the Education scenes reuse another scene's file with their own description and settings).
TEST(FindSceneTest, ACustomSceneIsNeverACopyOfAnotherEntry) {
	const auto& paths = pbrt_scene_registry::paths();
	ASSERT_GT(paths.size(), 100u);
	int custom = 0, duplicates = 0;
	for (const SceneDescriptor& s : get_scene_registry()) {
		if (s.category != std::string(SceneCategories::CustomScenes)) continue;
		++custom;
		const auto mine = paths.find(s.id);
		ASSERT_NE(mine, paths.end()) << s.id;
		for (const auto& other : paths) {
			std::error_code ec;
			if (other.first != s.id && std::filesystem::equivalent(other.second, mine->second, ec) && !ec) {
				++duplicates;
				ADD_FAILURE() << s.id << " (" << s.name << ") is the same file as " << other.first << ": " << mine->second;
			}
		}
	}
	EXPECT_GT(custom, 0);
	EXPECT_EQ(duplicates, 0);
}

TEST(FindSceneTest, ACuratedSceneKeepsItsOwnSettingsNotAFileEntrys) {
	// The Large Scenes read gigabyte meshes and the curated entry says so.
	int checked = 0;
	for (const SceneDescriptor& s : get_scene_registry()) {
		if (s.category != std::string(SceneCategories::LargeScene)) continue;
		EXPECT_TRUE(s.requires_files) << s.id << " (" << s.name << ")";
		++checked;
	}
	EXPECT_GT(checked, 0);
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
	// C1 (HdriSky) is pbrt-backed now (pbrt_scenes/hdri-sky-gradient.pbrt) -
	// a real image infinite light, with genuine per-direction importance-
	// sampled environment lookups on BOTH GPU backends (gpu/optix/
	// pbrt_gpu_builder.h / optix_sky_light.h / wavefront_sky_light.h), not
	// the flat-color GpuCameraParams::backgroundColor approximation its
	// native GPU builder (deleted) used to fall back to - this test's own
	// name predates that fix, kept only as a tripwire for "still gpu_
	// compatible", not a description of how C1 renders on GPU anymore. C7
	// (PortalInfiniteLight) still uses a genuinely flat `LightSource
	// "infinite" "rgb L"` (no image at all, by design - see that scene's
	// own pbrt file), so GPU's constant-background-color path remains
	// exactly correct for it, not an approximation.
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
	// B14 is pbrt-backed now (pbrt_scenes/measured-brdf-showroom.pbrt) - a
	// real, disclosed fidelity IMPROVEMENT over its old native self: the
	// now-deleted measured_material class (scenes_advanced.h) built a
	// MeasuredBRDFData member but never read it in scatter() - byte-for-byte
	// a mislabeled Lambertian, identically on CPU and GPU. This loader's
	// real pbrt-v4 MeasuredBxDF importance-sampling chain (src/shared/
	// measured_bxdf.h + measured_bxdf_loader.h's real .bsdf tensor-file
	// reader, already proven on 3 downloaded pbrt-v4-scenes bundles) is now
	// reachable from this scene too, via a small synthetic .bsdf baked
	// specifically for it - on BOTH GPU backends (gpu/optix/optix_measured_
	// bxdf.h/wavefront_measured_bxdf.h), not the plain MaterialType::
	// Lambertian fallback this test's own name/comment used to describe.
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

TEST(CpuSceneApiTest, RecommendedExposureByIdMatchesCppRegistry) {
	for (const auto& s : get_scene_registry()) {
		EXPECT_DOUBLE_EQ(cpu_scene_recommended_exposure_by_id(s.id.c_str()), s.recommended_exposure)
			<< "Exposure mismatch for id " << s.id;
	}
}

TEST(CpuSceneApiTest, RecommendedCameraPathByIdMatchesCppRegistry) {
	for (const auto& s : get_scene_registry()) {
		EXPECT_STREQ(cpu_scene_recommended_camera_path_by_id(s.id.c_str()), recommended_camera_path_for(s.id))
			<< "Camera path mismatch for id " << s.id;
	}
}

TEST(CpuSceneApiTest, UnknownIdRecommendedCameraPathReturnsOrbit) {
	EXPECT_STREQ(cpu_scene_recommended_camera_path_by_id("NotARealId"), "orbit");
	EXPECT_STREQ(cpu_scene_recommended_camera_path_by_id(""), "orbit");
}

TEST(CpuSceneApiTest, MetadataSnapshotRecommendedCameraPathMatchesById) {
	for (const auto& s : get_scene_registry()) {
		SceneMetadataSnapshot snap{};
		ASSERT_EQ(cpu_scene_metadata_snapshot(s.id.c_str(), &snap), 1) << "Snapshot lookup failed for id " << s.id;
		EXPECT_STREQ(snap.recommended_camera_path, cpu_scene_recommended_camera_path_by_id(s.id.c_str()))
			<< "Snapshot camera path mismatch for id " << s.id;
	}
}

// The size a scene declares in its header ("# @rt-size", src/shared/scene_size.h), through the C API the GUI uses. Run from the repository root.
TEST(SceneSizeApiTest, ASceneReportsTheSizeItsFileDeclaresAndTheSnapshotCarriesIt) {
	if (!std::filesystem::exists("pbrt_scenes/cornell-box-native.pbrt")) GTEST_SKIP() << "run from the repository root";
	EXPECT_NEAR(cpu_scene_size_by_id("A1"), 555.0, 1.0) << "the Cornell box, as pbrt_scenes/cornell-box-native.pbrt says";
	SceneMetadataSnapshot snap{};
	ASSERT_EQ(cpu_scene_metadata_snapshot("A1", &snap), 1);
	EXPECT_DOUBLE_EQ(snap.scene_size, cpu_scene_size_by_id("A1"));
	EXPECT_EQ(cpu_scene_size_by_id("NotARealId"), 0.0);
	EXPECT_EQ(cpu_scene_size_by_id(nullptr), 0.0);
}

TEST(SceneSizeApiTest, MeasuringAScenePrintsWhatTheStampedLineSays) {
	if (!std::filesystem::exists("pbrt_scenes/cornell-box-native.pbrt")) GTEST_SKIP() << "run from the repository root";
	EXPECT_NEAR(cpu_scene_measure_size("pbrt_scenes/cornell-box-native.pbrt"), 555.0, 1e-3);
	EXPECT_NEAR(cpu_scene_measure_size("A1"), 555.0, 1e-3) << "a scene id works too";
	EXPECT_LT(cpu_scene_measure_size("NotARealId"), 0.0);
	EXPECT_LT(cpu_scene_measure_size(""), 0.0);
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

TEST(CpuSceneApiTest, UnknownIdRecommendedExposureReturnsNeutral) {
	EXPECT_DOUBLE_EQ(cpu_scene_recommended_exposure_by_id("NotARealId"), 1.0);
	EXPECT_DOUBLE_EQ(cpu_scene_recommended_exposure_by_id(""), 1.0);
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
	// All sky-lit scenes should still return a dummy light for PDF sampling.
	// A3 and A2 excluded: both are pbrt-backed now (checkered-spheres.pbrt/
	// bouncing-spheres.pbrt) - the generic pbrt build_lights() legitimately
	// returns an empty list for a background-only scene
	// (SceneDescriptor::build_lights's own doc comment: "may return empty
	// list"), since sky_dummy_lights() (scene_registry.h) was a
	// native-scene-only PDF-sampling hack the pbrt loader path never used -
	// and, with A2 migrated (its own last caller), is now fully unused and
	// deleted.
	for (const std::string& id : {"A5", "A6", "B1"}) {
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
// createSettingsTab()), which query scene_metadata.dll -> this registry live,
// not a hardcoded array, so it can't drift out of sync with the registry
// on its own. This constant exists as a tripwire: if it stops matching
// scene_count(), something changed the registry size and it's worth
// double-checking the GUI/error-hint text that mentions specific scene
// counts or ID ranges by hand.
TEST(SceneRegistryGuiConsistencyTest, GuiSceneCountMatchesRegistry) {
	constexpr int kGuiSceneCount = 151;
	EXPECT_EQ(builtin_scene_count(), kGuiSceneCount)
		<< "Registry size changed -- update kGuiSceneCount here to match.";
}

// A Cornell-box-style enclosed room has opaque walls that are unlit from outside, so an orbit-family camera path (orbit,
// spiral, figure8), which circles OUTSIDE the box, rendered most of the default video solid black (5 of 8 frames of A1's,
// on CPU and Metal alike, ~40 scenes). These scenes use "tour", which sways in front of the open face. Found by rendering
// every scene's default video and counting black frames; this pins the result.
TEST(SceneRegistryTest, EnclosedCornellRoomsDoNotUseAnOrbitFamilyPath) {
	const char* const kEnclosedRooms[] = {
		"A1", "A8", "B3", "B5", "B6", "B8", "B9", "B12", "B13", "B15", "B16", "B22", "B23", "B24",
		"C2", "C3", "C4", "C5", "C6", "C9", "C14", "C15", "C20", "D5", "D6", "D8", "D9", "D10", "D12",
		"E1", "F1", "I1", "I2", "I4", "I5", "I6", "I7", "I8", "I9", "I10",
	};
	for (const char* id : kEnclosedRooms) {
		EXPECT_STREQ(recommended_camera_path_for(id), "tour") << "scene " << id;
	}
}

// ===========================================================================
// Slugs: the durable key of a scene (src/shared/scene_slugs.h)
// ===========================================================================

TEST(SceneSlugTest, SlugifyTurnsNamesIntoKebabCase) {
	using scene_slugs::slugify;
	EXPECT_EQ(slugify("Cornell Box"), "cornell-box");
	EXPECT_EQ(slugify("Depth of Field (pbrt file)"), "depth-of-field-pbrt-file");
	EXPECT_EQ(slugify("Conductor RGB Eta/K"), "conductor-rgb-eta-k");
	EXPECT_EQ(slugify("Mix & Match"), "mix-and-match");
	EXPECT_EQ(slugify("dragon_10"), "dragon-10") << "no underscores: render file names are split on them";
	EXPECT_EQ(slugify("  --odd__name--  "), "odd-name");
	EXPECT_EQ(slugify("???"), "scene");
}

TEST(SceneSlugTest, UniqueSlugAppendsANumberOnlyOnAClash) {
	std::set<std::string> taken;
	EXPECT_EQ(scene_slugs::uniqueSlug("frame", taken), "frame");
	EXPECT_EQ(scene_slugs::uniqueSlug("frame", taken), "frame-2");
	EXPECT_EQ(scene_slugs::uniqueSlug("frame", taken), "frame-3");
	EXPECT_EQ(scene_slugs::uniqueSlug("other", taken), "other");
}

TEST(SceneSlugTest, SlugsAndIdsAreToldApart) {
	using scene_slugs::looksLikeSlug;
	EXPECT_TRUE(looksLikeSlug("cornell-box"));
	EXPECT_TRUE(looksLikeSlug("frame1266"));
	EXPECT_FALSE(looksLikeSlug("A1"));
	EXPECT_FALSE(looksLikeSlug("K37"));
	EXPECT_FALSE(looksLikeSlug("1abc"));
	EXPECT_FALSE(looksLikeSlug(""));
	EXPECT_FALSE(looksLikeSlug("has_underscore"));
	EXPECT_FALSE(looksLikeSlug("Upper"));
}

TEST(SceneSlugTest, EveryBuiltInSceneHasAWellFormedUniqueSlug) {
	std::set<std::string> seen;
	for (const auto& s : get_scene_registry()) {
		EXPECT_FALSE(s.slug.empty()) << s.id << " (" << s.name << ") has no slug";
		EXPECT_TRUE(scene_slugs::looksLikeSlug(s.slug)) << s.id << ": '" << s.slug << "'";
		EXPECT_TRUE(seen.insert(s.slug).second) << "duplicate slug '" << s.slug << "' (" << s.id << ")";
	}
}

TEST(SceneSlugTest, TheSlugTableMatchesTheCompiledInScenes) {
	std::set<std::string> ids;
	for (const auto& s : get_builtin_scene_registry()) ids.insert(s.id);
	for (const auto& e : scene_slugs::kBuiltin) EXPECT_TRUE(ids.count(e.id)) << "slug table row for unknown scene " << e.id;
	for (const auto& s : get_builtin_scene_registry()) EXPECT_FALSE(scene_slugs::builtinSlugForId(s.id).empty()) << s.id << " is missing from kBuiltin";
	EXPECT_EQ(scene_slugs::kBuiltinCount, ids.size());
}

TEST(SceneSlugTest, FindSceneAcceptsASlugOrAnId) {
	const SceneDescriptor* byId = find_scene("A1");
	const SceneDescriptor* bySlug = find_scene("cornell-box");
	ASSERT_NE(byId, nullptr);
	EXPECT_EQ(byId, bySlug);
	EXPECT_EQ(byId->slug, "cornell-box");
	EXPECT_EQ(find_scene("no-such-scene"), nullptr);
	EXPECT_EQ(find_scene("A1 "), nullptr);
}

TEST(SceneSlugTest, TheCApiResolvesEitherKeyToTheId) {
	char id[32];
	ASSERT_EQ(cpu_resolve_scene_id("cornell-box", id, sizeof id), 1);
	EXPECT_STREQ(id, "A1");
	ASSERT_EQ(cpu_resolve_scene_id("B3", id, sizeof id), 1);
	EXPECT_STREQ(id, "B3");
	EXPECT_EQ(cpu_resolve_scene_id("nope", id, sizeof id), 0);
	EXPECT_STREQ(id, "");
	EXPECT_STREQ(cpu_scene_slug_by_id("A1"), "cornell-box");
	EXPECT_STREQ(cpu_scene_slug_by_id("cornell-box"), "cornell-box") << "the by-id accessors take a slug too";
	EXPECT_STREQ(cpu_scene_name_by_id("cornell-box"), cpu_scene_name_by_id("A1"));
	EXPECT_STREQ(cpu_scene_id_for_key("cornell-box"), "A1");
}

// A file in pbrt_scenes/ gets its name as its slug, so adding another file does not change it (its id may).
TEST(SceneSlugTest, ASceneFoundOnDiskIsKeyedByItsFileName) {
	const SceneDescriptor* s = find_scene("chromatic-absorber");
	if (!s) GTEST_SKIP() << "pbrt_scenes/ was not discovered - run from the repository root";
	EXPECT_EQ(s->category, std::string(SceneCategories::Tests)) << "a bundled fixture tags itself as a Test Scene";
	EXPECT_EQ(s->slug, "chromatic-absorber");
}

// ===========================================================================
// Names: what the scene list shows
// ===========================================================================

TEST(SceneNameTest, PrettyNameMakesAFileStemReadable) {
	using scene_slugs::prettyName;
	EXPECT_EQ(prettyName("bdpt-box-room"), "BDPT Box Room");
	EXPECT_EQ(prettyName("cornell_dof"), "Cornell DoF");
	EXPECT_EQ(prettyName("rgb-grid-nebula"), "RGB Grid Nebula");
	EXPECT_EQ(prettyName("glass-sphere-in-the-fog"), "Glass Sphere in the Fog");
	EXPECT_EQ(prettyName("the-room"), "The Room") << "a small word at the start is still capitalised";
	EXPECT_EQ(prettyName("frame1266"), "Frame1266");
	EXPECT_EQ(prettyName("dragon_10"), "Dragon 10");
	EXPECT_EQ(prettyName("nanovdb-medium"), "NanoVDB Medium");
	EXPECT_EQ(prettyName("---"), "---") << "nothing to read: the stem is shown as it is";
}

TEST(SceneNameTest, NoSceneNameDescribesHowItIsStored) {
	for (const auto& s : get_builtin_scene_registry()) {
		const std::string name = s.name;
		EXPECT_EQ(name.find("(pbrt example)"), std::string::npos) << s.id << ": " << name;
		EXPECT_EQ(name.find(".pbrt"), std::string::npos) << s.id << ": " << name;
	}
}

// "(pbrt file)" is only for the six scenes whose plain name a compiled-in scene already has.
TEST(SceneNameTest, APbrtFileQualifierAppearsOnlyWhereItDisambiguates) {
	std::set<std::string> plain;
	for (const auto& s : get_builtin_scene_registry()) plain.insert(s.name);
	int qualified = 0;
	for (const auto& s : get_builtin_scene_registry()) {
		const std::string name = s.name, suffix = " (pbrt file)";
		if (name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
			++qualified;
			EXPECT_TRUE(plain.count(name.substr(0, name.size() - suffix.size()))) << name << " is qualified but its plain name is free";
		}
	}
	EXPECT_EQ(qualified, 6);
}

TEST(SceneNameTest, BuiltInNamesAreUniqueAndTheListedNameOfAFileIsNotItsRawStem) {
	std::set<std::string> names;
	// A name is free text for the scenes a user brings: two Scene Builder saves can share the default title "My scene", and any file found on disk is a Custom Scene.
	for (const auto& s : get_scene_registry())
		EXPECT_TRUE(names.insert(s.name).second || s.category == std::string(SceneCategories::CustomScenes) || s.category == std::string(SceneCategories::MyScenes))
			<< s.id << ": " << s.name;
	const SceneDescriptor* s = find_scene("chromatic-absorber");
	if (!s) GTEST_SKIP() << "pbrt_scenes/ was not discovered - run from the repository root";
	EXPECT_STREQ(s->name, "Chromatic Absorber");
}

TEST(SceneNameTest, ASceneBuilderFileIsListedUnderItsTitle) {
	pbrt_discover::Discovered d;
	d.name = "my-scene";
	d.path = "pbrt_scenes/my-scene.pbrt";
	d.title = "Kitchen at night";
	EXPECT_EQ(pbrt_scene_registry::displayNameFor(d), "Kitchen at night");
	d.title.clear();
	EXPECT_EQ(pbrt_scene_registry::displayNameFor(d), "My Scene");
	d.nested = true;
	d.path = "pbrt_scenes/zero-day/frame25.pbrt";
	d.name = "frame25";
	EXPECT_EQ(pbrt_scene_registry::displayNameFor(d), "Zero Day: Frame25");
}

TEST(SceneNameTest, DescribeReadsTheTitleOfASceneBuilderFile) {
	const std::string text =
		"# t\n# @rt-builder-doc {\"version\":1,\"title\":\"Kitchen at night\"}\n\nLookAt 0 0 5  0 0 0  0 1 0\nCamera \"perspective\"\nWorldBegin\n";
	const pbrt_discover::Discovered d = pbrt_discover::describe("kitchen.pbrt", text);
	EXPECT_EQ(d.title, "Kitchen at night");
	EXPECT_TRUE(pbrt_discover::describe("plain.pbrt", "LookAt 0 0 5  0 0 0  0 1 0\nCamera \"perspective\"\nWorldBegin\n").title.empty());
}

// ===========================================================================
// Scene metadata a file carries in its own header
// ===========================================================================

TEST(SceneHeaderTagsTest, ReadsCategoryDescriptionAndPerformance) {
	const std::string text =
		"# @rt-category Test Scenes\n# plain comment\n# @rt-description A furnace:\r\n# @rt-description every pixel reads 1.\n"
		"# @rt-performance Fast\n# @rt-gpu no\n# @rt-unknown ignored\n# @rt-builder-doc {}\nLookAt 0 0 5 0 0 0 0 1 0\nCamera \"perspective\"\nWorldBegin\n"
		"# @rt-category After WorldBegin is not read\n";
	const auto tags = pbrt_discover::detail::readHeaderTags(text);
	EXPECT_EQ(tags.category, "Test Scenes");
	EXPECT_EQ(tags.description, "A furnace: every pixel reads 1.");
	EXPECT_EQ(tags.performance, "Fast");
	EXPECT_FALSE(tags.gpuCompatible);
	const auto none = pbrt_discover::detail::readHeaderTags("LookAt 0 0 5 0 0 0 0 1 0\nWorldBegin\n");
	EXPECT_TRUE(none.category.empty() && none.description.empty() && none.performance.empty());
	EXPECT_TRUE(none.gpuCompatible) << "a scene is assumed to render on the GPU unless it says otherwise";
}

TEST(SceneHeaderTagsTest, DescribeCarriesTheTags) {
	const auto d = pbrt_discover::describe(
		"f.pbrt", "# @rt-category Test Scenes\n# @rt-description One thing.\nLookAt 0 0 5 0 0 0 0 1 0\nCamera \"perspective\"\nWorldBegin\n");
	EXPECT_EQ(d.category, "Test Scenes");
	EXPECT_EQ(d.description, "One thing.");
}

// The bundled test fixtures sit under Test Scenes; a file with no tag (a user's own, or a Scene Builder one) stays a Custom Scene.
TEST(SceneHeaderTagsTest, BundledFixturesAreTestScenesAndUntaggedFilesAreCustom) {
	const SceneDescriptor* fixture = find_scene_by_file_stem("fog-furnace");
	if (!fixture) GTEST_SKIP() << "pbrt_scenes/ was not discovered - run from the repository root";
	EXPECT_EQ(fixture->category, std::string(SceneCategories::Tests));
	EXPECT_EQ(fixture->id[0], SceneCategories::letter_for_category(SceneCategories::Tests));
	for (const SceneDescriptor& s : get_scene_registry())
		if (s.category == std::string(SceneCategories::CustomScenes))
			EXPECT_FALSE(std::string(s.description).empty()) << s.id;
}

// ===========================================================================
// Scene info: what a person reads
// ===========================================================================

// A description says what the scene shows. It does not cite another scene by id or by an old flat number (both move), and it does not tell the
// history of the code ("now real on both backends", "previously rendered black").
TEST(SceneInfoTest, DescriptionsDoNotCiteIdsOrDevelopmentHistory) {
	const std::regex idLike("\\b[A-L][0-9]{1,3}\\b");
	const std::regex flatNumber("\\bscenes? [0-9]+", std::regex::icase);
	const std::regex history("now real|used to |previously|wired into|no longer|silently", std::regex::icase);
	for (const auto& s : get_builtin_scene_registry()) {
		const std::string d = s.description;
		EXPECT_GE(d.size(), 40u) << s.id << ": too short to say what the scene shows: " << d;
		// (Glass Presets names real glass types, F5, F10 and F11, that look like ids.)
		if (s.id != "B25") EXPECT_FALSE(std::regex_search(d, idLike)) << s.id << " cites a scene id: " << d;
		EXPECT_FALSE(std::regex_search(d, flatNumber)) << s.id << " cites a scene number: " << d;
		EXPECT_FALSE(std::regex_search(d, history)) << s.id << " tells the history of the code: " << d;
	}
}

TEST(SceneInfoTest, EveryPerformanceIsOneOfTheDefinedWords) {
	const std::set<std::string> words = {"Fast", "Medium", "Slow", "Very Slow", "Unknown"};
	for (const auto& s : get_scene_registry()) EXPECT_TRUE(words.count(s.performance)) << s.id << ": '" << s.performance << "'";
	for (const auto& s : get_builtin_scene_registry())
		EXPECT_NE(std::string(s.performance), "Unknown") << s.id << " (" << s.name << ") has not been given a measured performance";
}

// The bundled test scenes say what they check.
TEST(SceneInfoTest, EveryBundledTestSceneHasItsOwnDescription) {
	int tests = 0;
	for (const auto& s : get_scene_registry()) {
		if (s.category != std::string(SceneCategories::Tests)) continue;
		++tests;
		const std::string d = s.description;
		EXPECT_EQ(d.find("A pbrt-v4 scene file"), std::string::npos) << s.id << " (" << s.name << ") still has the generic description";
		EXPECT_GE(d.size(), 40u) << s.id;
	}
	if (tests == 0) GTEST_SKIP() << "pbrt_scenes/ was not discovered - run from the repository root";
}

// A scene that uses cone or paraboloid shapes (which the GPU backends do not have) says so, and is not offered a GPU render.
TEST(SceneHeaderTagsTest, ASceneCanSayItIsCpuOnly) {
	const SceneDescriptor* cone = find_scene_by_file_stem("lightpath-visible-cone-light");
	const SceneDescriptor* disk = find_scene_by_file_stem("lightpath-visible-disk-light");
	if (!cone || !disk) GTEST_SKIP() << "pbrt_scenes/ was not discovered - run from the repository root";
	EXPECT_FALSE(cone->gpu_compatible);
	EXPECT_TRUE(disk->gpu_compatible);
}
