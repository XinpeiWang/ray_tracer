// launcher_args_lightsampler_tests.cpp -- CLI flag parsing tests for
// --lightsampler (launcher/launcher_args.h), including the "auto" value
// added alongside uniform/power/bvh - mirroring this codebase's existing
// convention of testing parse_launch_args() directly (header-only, no
// linking against the launcher executable needed), same as
// launcher_args_bdpt_mlt_tests.cpp.
//
// "auto" itself isn't resolved here - parse_launch_args() only validates
// and stores the raw string; the actual "use whatever the scene's own
// Integrator lightsampler requested" resolution happens later, per-scene,
// in cpu_interface.cpp (see that resolution's own comment) once the
// scene's own recommendation is known - out of reach of a pure CLI-parsing
// test like this one.

#include <gtest/gtest.h>
#include "../../launcher/launcher_args.h"

#include <vector>
#include <string>
#include <cstring>

namespace {

// Same argv-building helper as launcher_args_bdpt_mlt_tests.cpp.
bool parse(std::vector<std::string> tokens, LaunchArgs& out_args) {
	std::vector<std::string> argv_storage;
	argv_storage.push_back("ray_tracer.exe");
	for (auto& t : tokens) argv_storage.push_back(t);

	std::vector<char*> argv_ptrs;
	for (auto& s : argv_storage) argv_ptrs.push_back(const_cast<char*>(s.c_str()));

	return parse_launch_args((int)argv_ptrs.size(), argv_ptrs.data(), out_args);
}

} // namespace

TEST(LauncherArgsLightSampler, DefaultIsEmpty) {
	LaunchArgs args;
	ASSERT_TRUE(parse({}, args));
	EXPECT_TRUE(args.lightsampler.empty());
}

TEST(LauncherArgsLightSampler, RecognizesEachRealImplementation) {
	for (const std::string& name : {"uniform", "power", "bvh"}) {
		LaunchArgs args;
		ASSERT_TRUE(parse({"--lightsampler", name}, args));
		EXPECT_EQ(args.lightsampler, name) << "for --lightsampler " << name;
	}
}

TEST(LauncherArgsLightSampler, RecognizesAuto) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--lightsampler", "auto"}, args));
	EXPECT_EQ(args.lightsampler, "auto");
}

TEST(LauncherArgsLightSampler, IsCaseInsensitive) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--lightsampler", "AUTO"}, args));
	EXPECT_EQ(args.lightsampler, "auto");
}

TEST(LauncherArgsLightSampler, InvalidValueFallsBackToDefault) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--lightsampler", "not-a-real-sampler"}, args));
	// Matches --sampler/--sppm-iterations's own established "invalid value
	// warns and leaves the default in place" behavior, not a parse failure.
	EXPECT_TRUE(args.lightsampler.empty());
}
