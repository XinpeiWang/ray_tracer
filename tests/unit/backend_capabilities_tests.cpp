// backend_capabilities_tests.cpp - src/shared/backend_capabilities.h: which options each renderer reads (the one table the launcher's warnings and the GUI's greyed-out
// controls both use).
#include <gtest/gtest.h>

#include <cstring>

#include "../../src/shared/backend_capabilities.h"

using namespace backend_caps;

TEST(BackendCapabilitiesTest, OnlyTheCpuPathTracerReadsTheCpuSamplingOptions) {
	for (Option o : {Option::Sampler, Option::LightSampler, Option::Spectral, Option::TimeLimit}) {
		EXPECT_TRUE(supports(Renderer::CpuDefault, o)) << flagName(o);
		EXPECT_FALSE(supports(Renderer::CpuOtherIntegrator, o)) << flagName(o);
		EXPECT_FALSE(supports(Renderer::GpuOptix, o)) << flagName(o);
		EXPECT_FALSE(supports(Renderer::GpuMetal, o)) << flagName(o);
	}
}

TEST(BackendCapabilitiesTest, MetalReadsAdaptiveSamplingAndOptixDoesNot) {
	// gpu/metal/metal_poc.mm reads options.adaptive_sampling and adaptive_threshold; the OptiX one-shot renderers do not.
	EXPECT_TRUE(supports(Renderer::GpuMetal, Option::AdaptiveSampling));
	EXPECT_TRUE(supports(Renderer::CpuDefault, Option::AdaptiveSampling));
	EXPECT_FALSE(supports(Renderer::GpuOptix, Option::AdaptiveSampling));
	EXPECT_FALSE(supports(Renderer::CpuOtherIntegrator, Option::AdaptiveSampling));
}

TEST(BackendCapabilitiesTest, TheGpuDifferencesAreTheDocumentedOnes) {
	EXPECT_TRUE(supports(Renderer::GpuOptix, Option::Regularize));
	EXPECT_FALSE(supports(Renderer::GpuMetal, Option::Regularize));
	EXPECT_TRUE(supports(Renderer::GpuOptix, Option::DofOverride));
	EXPECT_FALSE(supports(Renderer::GpuMetal, Option::DofOverride));
	EXPECT_TRUE(supports(Renderer::GpuOptix, Option::OptixValidate));
	EXPECT_FALSE(supports(Renderer::GpuMetal, Option::OptixValidate));
	EXPECT_FALSE(supports(Renderer::CpuDefault, Option::OptixValidate));
}

TEST(BackendCapabilitiesTest, WhatEveryPlainPathTracerReads) {
	for (Option o : {Option::MaxComponentValue, Option::Crop, Option::Seed, Option::Exposure, Option::Tonemap}) {
		for (Renderer r : {Renderer::CpuDefault, Renderer::GpuOptix, Renderer::GpuMetal}) EXPECT_TRUE(supports(r, o)) << flagName(o);
		EXPECT_FALSE(supports(Renderer::CpuOtherIntegrator, o)) << flagName(o);
	}
	// Scene construction is shared by every CPU integrator; a GPU always builds its own BVH.
	EXPECT_TRUE(supports(Renderer::CpuDefault, Option::Accelerator));
	EXPECT_TRUE(supports(Renderer::CpuOtherIntegrator, Option::Accelerator));
	EXPECT_FALSE(supports(Renderer::GpuOptix, Option::Accelerator));
	EXPECT_FALSE(supports(Renderer::GpuMetal, Option::Accelerator));
	// A denoiser exists for the default CPU path tracer (Open Image Denoise) and both GPU backends, not for the other CPU integrators.
	EXPECT_TRUE(supports(Renderer::CpuDefault, Option::Denoise));
	EXPECT_TRUE(supports(Renderer::GpuOptix, Option::Denoise));
	EXPECT_TRUE(supports(Renderer::GpuMetal, Option::Denoise));
	EXPECT_FALSE(supports(Renderer::CpuOtherIntegrator, Option::Denoise));
}

TEST(BackendCapabilitiesTest, EveryOptionAndRendererHasAName) {
	for (int o = 0; o <= static_cast<int>(Option::OptixValidate); ++o) EXPECT_GT(std::strlen(flagName(static_cast<Option>(o))), 2u);
	for (int r = 0; r <= static_cast<int>(Renderer::GpuMetal); ++r) EXPECT_GT(std::strlen(rendererName(static_cast<Renderer>(r))), 3u);
}
