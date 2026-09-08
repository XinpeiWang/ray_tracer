// launcher_args_time_limit_tests.cpp -- CLI flag parsing tests for
// --time-limit (launcher/launcher_args.h), mirroring this codebase's
// existing convention of testing parse_launch_args() directly (header-
// only, no linking against the launcher executable needed), same as
// launcher_args_adaptive_tests.cpp.
//
// The actual scanline-deadline behavior isn't exercised here - this only
// covers CLI-string parsing/validation; camera::time_limit_seconds's own
// comment (camera.h) documents the render-loop side.

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

TEST(LauncherArgsTimeLimit, DefaultIsDisabled) {
	LaunchArgs args;
	ASSERT_TRUE(parse({}, args));
	EXPECT_DOUBLE_EQ(args.time_limit_seconds, 0.0);
}

TEST(LauncherArgsTimeLimit, AcceptsPositiveValue) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--time-limit", "120"}, args));
	EXPECT_DOUBLE_EQ(args.time_limit_seconds, 120.0);
}

TEST(LauncherArgsTimeLimit, ZeroFallsBackToDisabled) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--time-limit", "0"}, args));
	EXPECT_DOUBLE_EQ(args.time_limit_seconds, 0.0);
}

TEST(LauncherArgsTimeLimit, NegativeFallsBackToDisabled) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--time-limit", "-30"}, args));
	EXPECT_DOUBLE_EQ(args.time_limit_seconds, 0.0);
}

TEST(LauncherArgsTimeLimit, InvalidValueFallsBackToDisabledAndConsumesToken) {
	LaunchArgs args;
	// Regression guard for the same class of bug --adaptive-threshold's own
	// catch block had: a parse failure must still consume both tokens, or
	// the bad value leaks into positional argument parsing and shifts
	// width/spp/max_depth/scene_id.
	ASSERT_TRUE(parse({"--time-limit", "bogus", "80", "8", "2", "A1"}, args));
	EXPECT_DOUBLE_EQ(args.time_limit_seconds, 0.0);
	EXPECT_EQ(args.image_width, 80);
	EXPECT_EQ(args.samples_per_pixel, 8);
	EXPECT_EQ(args.max_ray_depth, 2);
	EXPECT_EQ(args.scene_id, "A1");
}

TEST(LauncherArgsTimeLimit, CanCombineWithAdaptive) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--adaptive", "--time-limit", "300"}, args));
	EXPECT_TRUE(args.adaptive_sampling);
	EXPECT_DOUBLE_EQ(args.time_limit_seconds, 300.0);
}
