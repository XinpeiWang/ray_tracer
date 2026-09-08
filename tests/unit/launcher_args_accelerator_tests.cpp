// launcher_args_accelerator_tests.cpp -- CLI flag parsing tests for
// --accelerator/--splitmethod (launcher/launcher_args.h), mirroring this
// codebase's existing convention of testing parse_launch_args() directly
// (header-only, no linking against the launcher executable needed), same as
// launcher_args_lightsampler_tests.cpp.
//
// Neither flag is resolved here - parse_launch_args() only validates and
// stores the raw string; the actual override-onto-the-scene's-own-directive
// resolution happens later, per-scene, via accelerator_override.h and
// pbrt_load::loadFile() - out of reach of a pure CLI-parsing test like this
// one (see accelerator_override_tests.cpp for that).

#include <gtest/gtest.h>
#include "../../launcher/launcher_args.h"

#include <vector>
#include <string>
#include <cstring>

namespace {

// Same argv-building helper as launcher_args_lightsampler_tests.cpp.
bool parse(std::vector<std::string> tokens, LaunchArgs& out_args) {
	std::vector<std::string> argv_storage;
	argv_storage.push_back("ray_tracer.exe");
	for (auto& t : tokens) argv_storage.push_back(t);

	std::vector<char*> argv_ptrs;
	for (auto& s : argv_storage) argv_ptrs.push_back(const_cast<char*>(s.c_str()));

	return parse_launch_args((int)argv_ptrs.size(), argv_ptrs.data(), out_args);
}

} // namespace

TEST(LauncherArgsAccelerator, DefaultIsEmpty) {
	LaunchArgs args;
	ASSERT_TRUE(parse({}, args));
	EXPECT_TRUE(args.accelerator.empty());
	EXPECT_TRUE(args.splitmethod.empty());
}

TEST(LauncherArgsAccelerator, RecognizesBothAcceleratorTypes) {
	for (const std::string& name : {"bvh", "kdtree"}) {
		LaunchArgs args;
		ASSERT_TRUE(parse({"--accelerator", name}, args));
		EXPECT_EQ(args.accelerator, name) << "for --accelerator " << name;
	}
}

TEST(LauncherArgsAccelerator, RecognizesEachSplitMethod) {
	for (const std::string& name : {"sah", "middle", "equal", "hlbvh"}) {
		LaunchArgs args;
		ASSERT_TRUE(parse({"--splitmethod", name}, args));
		EXPECT_EQ(args.splitmethod, name) << "for --splitmethod " << name;
	}
}

TEST(LauncherArgsAccelerator, IsCaseInsensitive) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--accelerator", "KDTREE"}, args));
	EXPECT_EQ(args.accelerator, "kdtree");

	LaunchArgs args2;
	ASSERT_TRUE(parse({"--splitmethod", "HLBVH"}, args2));
	EXPECT_EQ(args2.splitmethod, "hlbvh");
}

TEST(LauncherArgsAccelerator, InvalidAcceleratorFallsBackToDefault) {
	LaunchArgs args;
	// Matches --sampler/--lightsampler's own established "invalid value
	// warns and leaves the default (here: no override) in place" behavior,
	// not a parse failure.
	ASSERT_TRUE(parse({"--accelerator", "not-a-real-accelerator"}, args));
	EXPECT_TRUE(args.accelerator.empty());
}

TEST(LauncherArgsAccelerator, InvalidSplitMethodFallsBackToDefault) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--splitmethod", "not-a-real-splitmethod"}, args));
	EXPECT_TRUE(args.splitmethod.empty());
}

TEST(LauncherArgsAccelerator, BothCanBeSetTogether) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--accelerator", "bvh", "--splitmethod", "hlbvh"}, args));
	EXPECT_EQ(args.accelerator, "bvh");
	EXPECT_EQ(args.splitmethod, "hlbvh");
}
