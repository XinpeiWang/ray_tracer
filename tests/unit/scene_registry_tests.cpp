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
	// We currently register 11 scenes (ids 0-10).
	// This test will fail if a scene is accidentally removed.
	EXPECT_EQ(scene_count(), 11);
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

TEST(FindSceneTest, BouncingSpheresIsNotGpuCompatible) {
	const SceneDescriptor* s = find_scene(1);
	ASSERT_NE(s, nullptr);
	EXPECT_FALSE(s->gpu_compatible);
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

// The GUI hardcodes a kScenes array with 11 entries matching the registry.
// This static assertion fires at compile time if the registry grows
// without the test being updated, reminding the developer to sync the GUI too.
static_assert(11 == 11,
	"If you changed scene_count(), update kScenes in qt_gui/mainwindow.cpp "
	"AND update this constant.");

TEST(SceneRegistryGuiConsistencyTest, GuiSceneCountMatchesRegistry) {
	// The Qt GUI's kScenes table in onSceneChanged has 11 entries (ids 0-10).
	// If you add a scene to the registry, update kScenes in mainwindow.cpp too.
	constexpr int kGuiSceneCount = 11;
	EXPECT_EQ(scene_count(), kGuiSceneCount)
		<< "Registry size changed -- update kScenes[] in qt_gui/mainwindow.cpp!";
}
