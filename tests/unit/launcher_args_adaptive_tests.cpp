// launcher_args_adaptive_tests.cpp -- CLI flag parsing tests for
// --adaptive/--adaptive-threshold (launcher/launcher_args.h), mirroring
// this codebase's existing convention of testing parse_launch_args()
// directly (header-only, no linking against the launcher executable
// needed), same as launcher_args_accelerator_tests.cpp.
//
// The actual convergence-stopping decision isn't resolved here -
// parse_launch_args() only validates and stores the raw flag/value; see
// adaptive_sampling_tests.cpp for pixel_convergence::has_converged()'s own
// pure-logic coverage.

#include <gtest/gtest.h>
#include "../../launcher/launcher_args.h"

#include <vector>
#include <string>
#include <cstring>

namespace {

// Same argv-building helper as launcher_args_accelerator_tests.cpp.
bool parse(std::vector<std::string> tokens, LaunchArgs& out_args) {
	std::vector<std::string> argv_storage;
	argv_storage.push_back("ray_tracer.exe");
	for (auto& t : tokens) argv_storage.push_back(t);

	std::vector<char*> argv_ptrs;
	for (auto& s : argv_storage) argv_ptrs.push_back(const_cast<char*>(s.c_str()));

	return parse_launch_args((int)argv_ptrs.size(), argv_ptrs.data(), out_args);
}

} // namespace

TEST(LauncherArgsAdaptive, DefaultIsOffWithDefaultThreshold) {
	LaunchArgs args;
	ASSERT_TRUE(parse({}, args));
	EXPECT_FALSE(args.adaptive_sampling);
	EXPECT_DOUBLE_EQ(args.adaptive_threshold, 0.01);
}

TEST(LauncherArgsAdaptive, FlagAloneEnablesWithDefaultThreshold) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--adaptive"}, args));
	EXPECT_TRUE(args.adaptive_sampling);
	EXPECT_DOUBLE_EQ(args.adaptive_threshold, 0.01);
}

TEST(LauncherArgsAdaptive, ThresholdCanBeSetWithoutTheFlag) {
	// Matches --maxcomponentvalue-style flags: the value flag alone doesn't
	// imply the boolean gate is on - parse_launch_args() just stores what
	// was passed, the "has no effect unless --adaptive too" wiring happens
	// downstream (RenderOptions/camera.h), not here.
	LaunchArgs args;
	ASSERT_TRUE(parse({"--adaptive-threshold", "0.05"}, args));
	EXPECT_FALSE(args.adaptive_sampling);
	EXPECT_DOUBLE_EQ(args.adaptive_threshold, 0.05);
}

TEST(LauncherArgsAdaptive, BothTogether) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--adaptive", "--adaptive-threshold", "0.002"}, args));
	EXPECT_TRUE(args.adaptive_sampling);
	EXPECT_DOUBLE_EQ(args.adaptive_threshold, 0.002);
}

TEST(LauncherArgsAdaptive, ZeroThresholdFallsBackToDefault) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--adaptive-threshold", "0"}, args));
	EXPECT_DOUBLE_EQ(args.adaptive_threshold, 0.01);
}

TEST(LauncherArgsAdaptive, NegativeThresholdFallsBackToDefault) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--adaptive-threshold", "-0.5"}, args));
	EXPECT_DOUBLE_EQ(args.adaptive_threshold, 0.01);
}

TEST(LauncherArgsAdaptive, InvalidThresholdValueFallsBackToDefault) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--adaptive-threshold", "not-a-number"}, args));
	EXPECT_DOUBLE_EQ(args.adaptive_threshold, 0.01);
}
