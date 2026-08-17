// launcher_args_bdpt_mlt_tests.cpp -- CLI flag parsing tests for the
// --bdpt/--mlt integration added to launcher/launcher_args.h, mirroring
// this codebase's existing convention of testing parse_launch_args()
// directly (header-only, no linking against the launcher executable
// needed) rather than spawning ray_tracer.exe as a subprocess.
//
// Covers: flag presence, numeric tunable overrides + their existing
// invalid-value-falls-back-to-default behavior (matches --sppm-iterations/
// --sppm-photons's own already-tested pattern), and that --bdpt/--mlt
// don't clobber each other's or --sppm's own defaults when unset.
//
// Mutual-exclusion / --gpu-ignored-with-warning validation (main.cpp, not
// launcher_args.h) is intentionally NOT tested here -- that logic lives in
// main.cpp's own dispatch, after parse_launch_args() returns, mirroring
// exactly where --sppm+--video's own equivalent check already lives (see
// main.cpp's own comment there). Testing that would mean either linking
// against main.cpp's logic (duplicated, not currently exposed as a
// separately-testable function -- same as --sppm+--video's own check) or a
// subprocess/exit-code integration test; out of scope for this file, which
// mirrors the pattern of testing parse_launch_args() alone.

#include <gtest/gtest.h>
#include "../../launcher/launcher_args.h"

#include <vector>
#include <string>
#include <cstring>

namespace {

// Builds an argv-compatible array from a list of string arguments (argv[0]
// is a fake program name, matching real argc/argv shape) and calls
// parse_launch_args(). Returns the parsed result via out_args; returns
// parse_launch_args()'s own bool.
bool parse(std::vector<std::string> tokens, LaunchArgs& out_args) {
	std::vector<std::string> argv_storage;
	argv_storage.push_back("ray_tracer.exe");
	for (auto& t : tokens) argv_storage.push_back(t);

	std::vector<char*> argv_ptrs;
	for (auto& s : argv_storage) argv_ptrs.push_back(const_cast<char*>(s.c_str()));

	return parse_launch_args((int)argv_ptrs.size(), argv_ptrs.data(), out_args);
}

} // namespace

// ---------------------------------------------------------------------------
// Defaults (no --bdpt/--mlt flags at all)
// ---------------------------------------------------------------------------

TEST(LauncherArgsBdptMlt, DefaultsAreOffWithSensibleTunables) {
	LaunchArgs args;
	ASSERT_TRUE(parse({}, args));
	EXPECT_FALSE(args.use_bdpt);
	EXPECT_FALSE(args.use_mlt);
	EXPECT_FALSE(args.use_sppm);
	EXPECT_GT(args.bdpt_max_depth, 0);
	EXPECT_GT(args.mlt_bootstrap, 0);
	EXPECT_GT(args.mlt_mutations, 0);
	EXPECT_GT(args.mlt_max_depth, 0);
	EXPECT_FALSE(args.gpu_flag_explicit);
}

// ---------------------------------------------------------------------------
// --bdpt
// ---------------------------------------------------------------------------

TEST(LauncherArgsBdptMlt, BdptFlagSetsUseBdpt) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--bdpt"}, args));
	EXPECT_TRUE(args.use_bdpt);
	EXPECT_FALSE(args.use_mlt);
	EXPECT_FALSE(args.use_sppm);
}

TEST(LauncherArgsBdptMlt, BdptMaxDepthOverridesDefault) {
	LaunchArgs args;
	int default_depth = LaunchArgs().bdpt_max_depth;
	ASSERT_TRUE(parse({"--bdpt", "--bdpt-max-depth", "12"}, args));
	EXPECT_TRUE(args.use_bdpt);
	EXPECT_EQ(args.bdpt_max_depth, 12);
	EXPECT_NE(args.bdpt_max_depth, default_depth);
}

TEST(LauncherArgsBdptMlt, InvalidBdptMaxDepthFallsBackToDefault) {
	LaunchArgs args;
	int default_depth = args.bdpt_max_depth;
	ASSERT_TRUE(parse({"--bdpt", "--bdpt-max-depth", "not_a_number"}, args));
	EXPECT_EQ(args.bdpt_max_depth, default_depth);
}

// ---------------------------------------------------------------------------
// --mlt
// ---------------------------------------------------------------------------

TEST(LauncherArgsBdptMlt, MltFlagSetsUseMlt) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--mlt"}, args));
	EXPECT_TRUE(args.use_mlt);
	EXPECT_FALSE(args.use_bdpt);
	EXPECT_FALSE(args.use_sppm);
}

TEST(LauncherArgsBdptMlt, MltTunablesOverrideDefaults) {
	LaunchArgs args;
	ASSERT_TRUE(parse({
		"--mlt",
		"--mlt-bootstrap", "5000",
		"--mlt-mutations", "123456789",
		"--mlt-max-depth", "9"
	}, args));
	EXPECT_TRUE(args.use_mlt);
	EXPECT_EQ(args.mlt_bootstrap, 5000);
	EXPECT_EQ(args.mlt_mutations, 123456789LL);
	EXPECT_EQ(args.mlt_max_depth, 9);
}

TEST(LauncherArgsBdptMlt, MltMutationsAcceptsLargeValueBeyond32Bit) {
	// mlt_mutations is a long long specifically so a large total mutation
	// budget (a real, expected use case for a high-quality MLT render)
	// doesn't silently wrap the way an int would past ~2.1 billion.
	LaunchArgs args;
	ASSERT_TRUE(parse({"--mlt", "--mlt-mutations", "5000000000"}, args));
	EXPECT_EQ(args.mlt_mutations, 5000000000LL);
}

TEST(LauncherArgsBdptMlt, InvalidMltTunablesFallBackToDefaults) {
	LaunchArgs defaults;
	LaunchArgs args;
	ASSERT_TRUE(parse({
		"--mlt",
		"--mlt-bootstrap", "xyz",
		"--mlt-mutations", "xyz",
		"--mlt-max-depth", "xyz"
	}, args));
	EXPECT_EQ(args.mlt_bootstrap, defaults.mlt_bootstrap);
	EXPECT_EQ(args.mlt_mutations, defaults.mlt_mutations);
	EXPECT_EQ(args.mlt_max_depth, defaults.mlt_max_depth);
}

// ---------------------------------------------------------------------------
// gpu_flag_explicit -- see launcher_args.h's own doc comment on this field:
// main.cpp uses it to warn about --gpu being ignored under --bdpt/--mlt
// without spamming that warning on the common `--bdpt` (no --gpu token at
// all) invocation, since use_gpu's own struct default is already true.
// ---------------------------------------------------------------------------

TEST(LauncherArgsBdptMlt, GpuFlagExplicitOnlyTrueWhenGpuTokenPresent) {
	LaunchArgs args_no_gpu;
	ASSERT_TRUE(parse({"--bdpt"}, args_no_gpu));
	EXPECT_FALSE(args_no_gpu.gpu_flag_explicit);
	EXPECT_TRUE(args_no_gpu.use_gpu);   // struct default, not from an explicit token

	LaunchArgs args_with_gpu;
	ASSERT_TRUE(parse({"--bdpt", "--gpu"}, args_with_gpu));
	EXPECT_TRUE(args_with_gpu.gpu_flag_explicit);
	EXPECT_TRUE(args_with_gpu.use_gpu);
}

// ---------------------------------------------------------------------------
// Coexistence with unrelated flags (positional args, --output, --sppm's own
// flags) -- confirms --bdpt/--mlt's new "else if" branches in the parser
// don't accidentally swallow or get swallowed by neighboring flags.
// ---------------------------------------------------------------------------

TEST(LauncherArgsBdptMlt, BdptCoexistsWithPositionalArgsAndOutput) {
	LaunchArgs args;
	ASSERT_TRUE(parse({"--bdpt", "--bdpt-max-depth", "7", "-o", "out.ppm", "400", "50", "10", "A1"}, args));
	EXPECT_TRUE(args.use_bdpt);
	EXPECT_EQ(args.bdpt_max_depth, 7);
	EXPECT_EQ(args.custom_output_path, "out.ppm");
	EXPECT_EQ(args.image_width, 400);
	EXPECT_EQ(args.samples_per_pixel, 50);
	EXPECT_EQ(args.max_ray_depth, 10);
	EXPECT_EQ(args.scene_id, "A1");
}
