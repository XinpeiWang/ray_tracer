// launcher_args_denoise_blend_tests.cpp -- CLI flag parsing tests for
// --denoise-blend (launcher/launcher_args.h), mirroring this codebase's
// existing convention of testing parse_launch_args() directly (header-
// only, no linking against the launcher executable needed), same as
// launcher_args_adaptive_tests.cpp.

#include <gtest/gtest.h>
#include "../../launcher/launcher_args.h"

#include <vector>
#include <string>
#include <cstring>

namespace {

// Same argv-building helper as launcher_args_adaptive_tests.cpp.
bool parse(std::vector<std::string> tokens, LaunchArgs& out_args) {
	std::vector<std::string> argv_storage;
	argv_storage.push_back("ray_tracer.exe");
	for (auto& t : tokens) argv_storage.push_back(t);

	std::vector<char*> argv_ptrs;
	for (auto& s : argv_storage) argv_ptrs.push_back(const_cast<char*>(s.c_str()));

	return parse_launch_args((int)argv_ptrs.size(), argv_ptrs.data(), out_args);
}

} // namespace

TEST(LauncherArgsDenoiseBlend, DefaultIsZero) {
	LaunchArgs args;
	ASSERT_TRUE(parse({}, args));
	EXPECT_DOUBLE_EQ(args.denoise_blend, 0.0);
}

TEST(LauncherArgsDenoiseBlend, AcceptsValueInRange) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--denoise", "--denoise-blend", "0.35"}, args));
	EXPECT_DOUBLE_EQ(args.denoise_blend, 0.35);
}

TEST(LauncherArgsDenoiseBlend, AcceptsBoundaryValues) {
	LaunchArgs args0;
	ASSERT_TRUE(parse({"--denoise-blend", "0.0"}, args0));
	EXPECT_DOUBLE_EQ(args0.denoise_blend, 0.0);

	LaunchArgs args1;
	ASSERT_TRUE(parse({"--denoise-blend", "1.0"}, args1));
	EXPECT_DOUBLE_EQ(args1.denoise_blend, 1.0);
}

TEST(LauncherArgsDenoiseBlend, AboveOneFallsBackToDefault) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--denoise-blend", "1.5"}, args));
	EXPECT_DOUBLE_EQ(args.denoise_blend, 0.0);
}

TEST(LauncherArgsDenoiseBlend, NegativeFallsBackToDefault) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--denoise-blend", "-0.1"}, args));
	EXPECT_DOUBLE_EQ(args.denoise_blend, 0.0);
}

TEST(LauncherArgsDenoiseBlend, InvalidValueFallsBackToDefaultAndConsumesToken) {
	LaunchArgs args;
	// Regression guard: a parse failure must still consume both tokens (the
	// flag AND its bad value), or the bad value leaks into positional
	// argument parsing and shifts width/spp/max_depth/scene_id - the exact
	// bug fixed for --adaptive-threshold's own catch block.
	ASSERT_TRUE(parse({"--denoise-blend", "bogus", "80", "8", "2", "A1"}, args));
	EXPECT_DOUBLE_EQ(args.denoise_blend, 0.0);
	EXPECT_EQ(args.image_width, 80);
	EXPECT_EQ(args.samples_per_pixel, 8);
	EXPECT_EQ(args.max_ray_depth, 2);
	EXPECT_EQ(args.scene_id, "A1");
}
