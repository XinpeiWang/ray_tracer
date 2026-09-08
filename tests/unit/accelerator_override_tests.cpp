/**
 * @file accelerator_override_tests.cpp
 * @brief Unit tests for src/shared/accelerator_override.h and its consumer,
 * pbrt_load::loadFile()'s `accelOverride` parameter.
 *
 * launcher_args_accelerator_tests.cpp only covers --accelerator/--splitmethod
 * CLI-string parsing; this file covers the part that actually matters for
 * correctness - that a set() override reaches a loaded scene's resolved
 * FlatScene::acceleratorType/acceleratorSplitMethod, applied BEFORE
 * pbrt_flatten::flatten() runs (so it goes through flatten()'s own
 * validation/fallback logic, not a second, separate path), and that an
 * empty/default override changes nothing versus calling loadFile() with no
 * override argument at all (a regression guard for the "only copy the
 * parsed Scene when an override is actually present" optimization in
 * pbrt_load.h).
 *
 * accelerator_override::state() is a process-global - every TEST_F below
 * calls accelerator_override::reset() in TearDown() so one test's override
 * can never leak into another test in the same gtest binary (see
 * accelerator_override.h's own comment on why a leaked override would
 * otherwise silently persist forever, since wire_pbrt_backed_scene() caches
 * its BuildResult on first build).
 */

#include <gtest/gtest.h>

#include "accelerator_override.h"
#include "pbrt_load.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

namespace {

// Same scratch-directory fixture shape as pbrt_load_tests.cpp's TempTree.
class AcceleratorOverrideTest : public ::testing::Test {
protected:
	void SetUp() override {
		const char *tmp = std::getenv("TEMP");
		root_ = std::string(tmp ? tmp : ".") + "/accelerator_override_tests/";
		makeDir(root_);
	}
	void TearDown() override {
		for (const std::string &f : written_) std::remove(f.c_str());
		accelerator_override::reset();
	}

	void write(const std::string &relative, const std::string &contents) {
		const std::string full = root_ + relative;
		std::ofstream out(full, std::ios::binary);
		out << contents;
		out.close();
		written_.push_back(full);
	}

	std::string path(const std::string &relative) const { return root_ + relative; }

private:
	static void makeDir(const std::string &d) {
	#ifdef _WIN32
		std::string cmd = "if not exist \"" + d + "\" mkdir \"" + d + "\" >nul 2>&1";
		for (char &c : cmd) if (c == '/') c = '\\';
		std::system(cmd.c_str());
	#else
		std::system(("mkdir -p '" + d + "'").c_str());
	#endif
	}
	std::string root_;
	std::vector<std::string> written_;
};

// A minimal self-contained scene with no Accelerator directive of its own,
// so its resolved acceleratorType/acceleratorSplitMethod come purely from
// flatten()'s own "bvh"/"sah" defaults unless an override says otherwise.
const char *kNoAcceleratorDirective =
	"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
	"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n";

} // namespace

// ---------------------------------------------------------------------------
// accelerator_override::state()/set()/reset()/was_set() - pure, no filesystem
// ---------------------------------------------------------------------------

TEST(AcceleratorOverrideState, DefaultIsEmptyAndNotYetSet) {
	accelerator_override::reset();
	EXPECT_TRUE(accelerator_override::state().type.empty());
	EXPECT_TRUE(accelerator_override::state().split_method.empty());
	EXPECT_FALSE(accelerator_override::was_set());
}

TEST(AcceleratorOverrideState, SetStoresValuesAndMarksSet) {
	accelerator_override::reset();
	accelerator_override::set({"kdtree", "hlbvh"});
	EXPECT_EQ(accelerator_override::state().type, "kdtree");
	EXPECT_EQ(accelerator_override::state().split_method, "hlbvh");
	EXPECT_TRUE(accelerator_override::was_set());
	accelerator_override::reset();
}

TEST(AcceleratorOverrideState, ResetRestoresEmptyAndUnsetState) {
	accelerator_override::set({"bvh", "sah"});
	accelerator_override::reset();
	EXPECT_TRUE(accelerator_override::state().type.empty());
	EXPECT_TRUE(accelerator_override::state().split_method.empty());
	EXPECT_FALSE(accelerator_override::was_set());
}

// A real CLI invocation with neither --accelerator nor --splitmethod passed
// still calls set() (with both fields empty) - was_set() must distinguish
// that from "set() never ran at all" (see accelerator_override.h's own
// comment on why this couldn't be told apart using type/split_method alone).
TEST(AcceleratorOverrideState, SetWithEmptyValuesStillMarksSet) {
	accelerator_override::reset();
	accelerator_override::set({"", ""});
	EXPECT_TRUE(accelerator_override::state().type.empty());
	EXPECT_TRUE(accelerator_override::state().split_method.empty());
	EXPECT_TRUE(accelerator_override::was_set());
	accelerator_override::reset();
}

// ---------------------------------------------------------------------------
// pbrt_load::loadFile()'s accelOverride parameter
// ---------------------------------------------------------------------------

TEST_F(AcceleratorOverrideTest, NoOverrideResolvesToRealDefaults) {
	write("scene.pbrt", kNoAcceleratorDirective);
	const pbrt_load::LoadResult r = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.scene.acceleratorType, "bvh");
	EXPECT_EQ(r.scene.acceleratorSplitMethod, "sah");
}

TEST_F(AcceleratorOverrideTest, EmptyOverrideMatchesNoOverrideArgumentAtAll) {
	write("scene.pbrt", kNoAcceleratorDirective);
	const pbrt_load::LoadResult withDefaultArg = pbrt_load::loadFile(path("scene.pbrt"));
	const pbrt_load::LoadResult withExplicitEmpty =
		pbrt_load::loadFile(path("scene.pbrt"), accelerator_override::Override{});
	ASSERT_TRUE(withDefaultArg.ok) << withDefaultArg.error;
	ASSERT_TRUE(withExplicitEmpty.ok) << withExplicitEmpty.error;
	EXPECT_EQ(withDefaultArg.scene.acceleratorType, withExplicitEmpty.scene.acceleratorType);
	EXPECT_EQ(withDefaultArg.scene.acceleratorSplitMethod, withExplicitEmpty.scene.acceleratorSplitMethod);
}

TEST_F(AcceleratorOverrideTest, OverrideTypeReachesResolvedFlatScene) {
	write("scene.pbrt", kNoAcceleratorDirective);
	const pbrt_load::LoadResult r =
		pbrt_load::loadFile(path("scene.pbrt"), accelerator_override::Override{"kdtree", ""});
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.scene.acceleratorType, "kdtree");
}

TEST_F(AcceleratorOverrideTest, OverrideSplitMethodReachesResolvedFlatScene) {
	write("scene.pbrt", kNoAcceleratorDirective);
	const pbrt_load::LoadResult r =
		pbrt_load::loadFile(path("scene.pbrt"), accelerator_override::Override{"", "hlbvh"});
	ASSERT_TRUE(r.ok) << r.error;
	EXPECT_EQ(r.scene.acceleratorType, "bvh");  // unaffected - only splitmethod overridden
	EXPECT_EQ(r.scene.acceleratorSplitMethod, "hlbvh");
}

// The CLI override must win over the scene file's own explicit directive -
// this is the "CLI decides" half of the design (see pbrt_flatten.h's own
// acceleratorType resolution comment), not merely a fallback for when the
// scene made no request.
TEST_F(AcceleratorOverrideTest, OverrideWinsOverTheScenesOwnExplicitDirective) {
	write("scene.pbrt",
		  "Accelerator \"bvh\" \"string splitmethod\" [\"middle\"]\n"
		  "Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		  "  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	// Confirm the scene's own directive is honored with no override first.
	const pbrt_load::LoadResult noOverride = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(noOverride.ok) << noOverride.error;
	EXPECT_EQ(noOverride.scene.acceleratorSplitMethod, "middle");

	const pbrt_load::LoadResult overridden =
		pbrt_load::loadFile(path("scene.pbrt"), accelerator_override::Override{"", "hlbvh"});
	ASSERT_TRUE(overridden.ok) << overridden.error;
	EXPECT_EQ(overridden.scene.acceleratorSplitMethod, "hlbvh");
}
