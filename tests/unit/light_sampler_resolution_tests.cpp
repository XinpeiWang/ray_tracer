// light_sampler_resolution_tests.cpp -- unit tests for
// src/shared/light_sampler_resolution.h, the pure function cpu_interface.cpp
// uses to resolve --lightsampler (explicit value, "auto", or unset) plus a
// scene's own Integrator "string lightsampler" recommendation into what to
// actually construct and what to print. Header-only, no linking against the
// launcher/cpu_renderer executables needed - same convention as
// launcher_args_bdpt_mlt_tests.cpp's own top-of-file comment explains for
// launcher_args.h.
//
// Covers a real bug caught by code review: scene_desc->recommended_light_sampler
// is unvalidated text straight from a .pbrt file, so a scene requesting a
// real pbrt-v4 option this project doesn't implement (e.g. "exhaustive") -
// or just a typo/wrong case - must not be reported as honored (auto's
// "using it") or actionable (the plain warning's "pass --lightsampler X"),
// since light_sampler_choice silently falls back to "bvh" for it either way.

#include <gtest/gtest.h>
#include "../../src/shared/light_sampler_resolution.h"

using light_sampler_resolution::resolve;
using light_sampler_resolution::Result;

// ---------------------------------------------------------------------------
// No explicit --lightsampler
// ---------------------------------------------------------------------------

TEST(LightSamplerResolution, NoCliNoRecommendationIsSilentBvh) {
	Result r = resolve("A1", "", false, "");
	EXPECT_EQ(r.choice, "bvh");
	EXPECT_TRUE(r.mismatch_warning.empty());
	EXPECT_TRUE(r.auto_info.empty());
}

TEST(LightSamplerResolution, NoCliRecommendationBvhIsSilent) {
	// "bvh" is the real default - recommending it is a no-op, not a mismatch.
	Result r = resolve("A1", "bvh", false, "");
	EXPECT_EQ(r.choice, "bvh");
	EXPECT_TRUE(r.mismatch_warning.empty());
}

TEST(LightSamplerResolution, NoCliRecommendationPowerWarnsButStillUsesBvh) {
	Result r = resolve("A1", "power", false, "");
	EXPECT_EQ(r.choice, "bvh");
	EXPECT_FALSE(r.mismatch_warning.empty());
	EXPECT_NE(r.mismatch_warning.find("power"), std::string::npos);
	EXPECT_NE(r.mismatch_warning.find("A1"), std::string::npos);
}

TEST(LightSamplerResolution, NoCliRecommendationUnimplementedStaysSilent) {
	// The bug this test guards against: a scene requesting something this
	// project doesn't implement (a real pbrt-v4 option, a typo, wrong case)
	// must not be told "pass --lightsampler X explicitly" - X isn't a value
	// --lightsampler actually accepts, so that advice would be wrong.
	Result r = resolve("A1", "exhaustive", false, "");
	EXPECT_EQ(r.choice, "bvh");
	EXPECT_TRUE(r.mismatch_warning.empty());
	EXPECT_TRUE(r.auto_info.empty());
}

// ---------------------------------------------------------------------------
// Explicit --lightsampler (not "auto")
// ---------------------------------------------------------------------------

TEST(LightSamplerResolution, ExplicitValueAlwaysWinsRegardlessOfRecommendation) {
	for (const std::string& recommendation : {std::string(), std::string("bvh"), std::string("power"), std::string("exhaustive")}) {
		Result r = resolve("A1", recommendation, true, "uniform");
		EXPECT_EQ(r.choice, "uniform") << "for recommendation '" << recommendation << "'";
		EXPECT_TRUE(r.mismatch_warning.empty());
		EXPECT_TRUE(r.auto_info.empty());
	}
}

// ---------------------------------------------------------------------------
// --lightsampler auto
// ---------------------------------------------------------------------------

TEST(LightSamplerResolution, AutoWithNoRecommendationIsSilentBvh) {
	Result r = resolve("A1", "", true, "auto");
	EXPECT_EQ(r.choice, "bvh");
	EXPECT_TRUE(r.auto_info.empty());
	EXPECT_TRUE(r.mismatch_warning.empty());
}

TEST(LightSamplerResolution, AutoWithBvhRecommendationIsSilent) {
	Result r = resolve("A1", "bvh", true, "auto");
	EXPECT_EQ(r.choice, "bvh");
	EXPECT_TRUE(r.auto_info.empty());
}

TEST(LightSamplerResolution, AutoWithPowerRecommendationUsesItAndSaysSo) {
	Result r = resolve("A1", "power", true, "auto");
	EXPECT_EQ(r.choice, "power");
	EXPECT_FALSE(r.auto_info.empty());
	EXPECT_NE(r.auto_info.find("power"), std::string::npos);
	EXPECT_TRUE(r.mismatch_warning.empty());
}

TEST(LightSamplerResolution, AutoWithUniformRecommendationUsesItAndSaysSo) {
	Result r = resolve("B1", "uniform", true, "auto");
	EXPECT_EQ(r.choice, "uniform");
	EXPECT_FALSE(r.auto_info.empty());
	EXPECT_NE(r.auto_info.find("uniform"), std::string::npos);
}

TEST(LightSamplerResolution, AutoWithUnimplementedRecommendationFallsBackSilently) {
	// The exact bug this extraction fixes: previously this printed
	// "using it" while light_sampler_choice actually fell back to bvh.
	Result r = resolve("A1", "exhaustive", true, "auto");
	EXPECT_EQ(r.choice, "bvh");
	EXPECT_TRUE(r.auto_info.empty())
		<< "auto_info claimed to honor an unimplemented recommendation: " << r.auto_info;
	EXPECT_TRUE(r.mismatch_warning.empty());
}

TEST(LightSamplerResolution, AutoIsCaseSensitiveLikeEveryOtherComparisonHere) {
	// Documents current behavior rather than prescribing it: "Power"/"BVH"
	// (wrong case) are treated as unimplemented, same as "exhaustive" -
	// launcher_args.h's own CLI parsing already lowercases --lightsampler's
	// value before it ever reaches here, but a scene's own recommendation
	// (parsed by pbrt_scene.h) is never case-normalized.
	Result r = resolve("A1", "Power", true, "auto");
	EXPECT_EQ(r.choice, "bvh");
	EXPECT_TRUE(r.auto_info.empty());
}
