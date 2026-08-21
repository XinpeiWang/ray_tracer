// sampler_kind_tests.cpp -- SamplerKind/sampler_kind_from_name() (camera.h),
// the CPU render loop's --sampler/Sampler-directive dispatch added to adopt
// this project's already-ported pbrt-v4 samplers (Sobol/ZSobol/PaddedSobol/
// Stratified/PMJ02BN/Halton) into an actual runtime choice - see
// launcher_args.h's --sampler flag and cpu_interface.cpp's own wiring.

#include <gtest/gtest.h>

#include "../../src/TheRestOfYourLife/camera.h"

TEST(SamplerKindFromName, RecognizesEveryPortedSampler) {
	SamplerKind kind;
	EXPECT_TRUE(sampler_kind_from_name("sobol", kind));
	EXPECT_EQ(kind, SamplerKind::Sobol);
	EXPECT_TRUE(sampler_kind_from_name("zsobol", kind));
	EXPECT_EQ(kind, SamplerKind::ZSobol);
	EXPECT_TRUE(sampler_kind_from_name("paddedsobol", kind));
	EXPECT_EQ(kind, SamplerKind::PaddedSobol);
	EXPECT_TRUE(sampler_kind_from_name("stratified", kind));
	EXPECT_EQ(kind, SamplerKind::Stratified);
	EXPECT_TRUE(sampler_kind_from_name("pmj02bn", kind));
	EXPECT_EQ(kind, SamplerKind::PMJ02BN);
	EXPECT_TRUE(sampler_kind_from_name("halton", kind));
	EXPECT_EQ(kind, SamplerKind::Halton);
}

TEST(SamplerKindFromName, RejectsUnrecognizedOrPbrtOnlyNames) {
	SamplerKind kind = SamplerKind::Sobol;
	// "independent" is a real pbrt-v4 sampler this project never ported -
	// must fail closed (false), not silently alias to something else.
	EXPECT_FALSE(sampler_kind_from_name("independent", kind));
	EXPECT_FALSE(sampler_kind_from_name("", kind));
	EXPECT_FALSE(sampler_kind_from_name("Sobol", kind))
		<< "names are case-sensitive lowercase, matching pbrt's own directive spelling";
}

TEST(Camera, SamplerKindDefaultsToSobol) {
	camera cam;
	EXPECT_EQ(cam.sampler_kind, SamplerKind::Sobol)
		<< "must match this project's pre-existing hardcoded behavior so an "
		   "existing render is pixel-identical unless --sampler is passed";
}
