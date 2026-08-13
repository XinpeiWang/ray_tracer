/**
 * @file optix_validation_sweep_test.cpp
 * @brief Opt-in deep-validation sweep for the GPU backends.
 *
 * Renders a curated set of scenes through both GPU backends (recursive and
 * wavefront) with OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL enabled (see
 * OptiXRenderer::createContext()'s own comment), and fails if OptiX's log
 * callback reports anything at level <= 3 (fatal/error/warning).
 *
 * This is the automated form of the manual investigation that found the
 * wavefront shadow-ray miss-SBT-index bug: that bug never crashed on its
 * own scene - it silently read adjacent heap memory and produced a
 * plausible-looking image - and was only caught by explicitly enabling
 * validation mode. A "does it render, is it non-black" smoke test would
 * never have caught it; this sweep exists to catch that whole class of bug.
 *
 * NOT part of the default test run. Deliberately excluded because:
 *   1. Validation mode has a real per-launch cost - it shouldn't tax every
 *      build the way a normal correctness test does.
 *   2. It's a slow sweep by design (many scene x backend combinations).
 * Run it explicitly and in its own process:
 *   ray_tracer_tests.exe --gtest_filter="OptixValidationSweepTest.*"
 * with RAY_TRACER_RUN_VALIDATION_SWEEP=1 set - SetUp() skips otherwise.
 *
 * "In its own process" is not optional: RAY_TRACER_OPTIX_VALIDATION (unlike
 * RAY_TRACER_WAVEFRONT) is read exactly once, at OptiX device context
 * creation - the first optix_render_main() call in the process, whichever
 * call that happens to be (see optix_validation_enabled()'s own comment).
 * If anything else already rendered earlier in this process, the context
 * is already built without validation and setting the env var now does
 * nothing. SetUp() checks optix_validation_enabled() after warming up the
 * context and skips loudly, with that explanation, rather than silently
 * passing a sweep that never actually validated anything.
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

extern "C" {
	#include "optix_interface.h"
}

namespace {

// Picked to cover every GPU-implemented primitive type (sphere, quad,
// triangle, bilinear patch, curve), every light kind (area quad, area
// sphere, punctual, zero lights), motion blur, textures/normal maps, and
// the two material types added this session (Principled,
// NormalMappedLambertian) - all without external mesh assets, so the
// sweep can't silently shrink on a machine that hasn't downloaded them.
// See gpu/optix/scene_builder.cpp's case comments for what each ID builds.
struct SweepScene {
	const char* id;
	const char* name;
};

const SweepScene kSweepScenes[] = {
	{"A1",  "CornellBox"},            // quads + sphere, area light - baseline
	{"A2",  "BouncingSpheres"},       // motion blur - task #107's bug class
	{"A6",  "ColoredQuads"},          // zero lights - task #106's bug class
	{"A9",  "FinalScene"},            // motion blur + medium + many primitives -
	                                    // the scene that first exposed both bugs
	{"B5",  "CornellCoatedDiffuse"},  // complex layered BxDF
	{"B10", "PrincipledShowcase"},    // Principled material, zero lights
	{"B11", "HairFibers"},            // curve geometry
	{"B12", "NormalMappedCornell"},   // textures + normal maps
	{"F1",  "BilinearPatchScene"},    // bilinear patch geometry
	{"C2",  "SpotlightCornell"},      // punctual light
	{"F2",  "TriangleMesh"},          // triangle geometry
};

}  // namespace

class OptixValidationSweepTest : public ::testing::Test {
protected:
	void SetUp() override {
		if (!optix_is_available()) {
			GTEST_SKIP() << "OptiX not available; skipping validation sweep";
		}
		if (!std::getenv("RAY_TRACER_RUN_VALIDATION_SWEEP")) {
			GTEST_SKIP() << "Opt-in only (extra per-launch validation cost) - "
			                "set RAY_TRACER_RUN_VALIDATION_SWEEP=1 to run";
		}
#ifdef _WIN32
		_putenv_s("RAY_TRACER_OPTIX_VALIDATION", "1");
#else
		setenv("RAY_TRACER_OPTIX_VALIDATION", "1", 1);
#endif
		// Warm up the context (this is the call that decides validation
		// mode for the rest of the process, if it hasn't been decided
		// already) before checking whether it actually took.
		optix_clear_validation_issues();
		optix_render_main(4, 4, 1, 1, warmupPath_.c_str(), "A1", 278.0, 278.0, -800.0);
		std::remove(warmupPath_.c_str());
		if (!optix_validation_enabled()) {
			GTEST_SKIP() << "Validation mode did not take effect - something "
			                "else already created the OptiX context earlier "
			                "in this process. Run this suite alone: "
			                "ray_tracer_tests.exe "
			                "--gtest_filter=\"OptixValidationSweepTest.*\"";
		}
	}

	void TearDown() override {
		std::remove(outPath_.c_str());
	}

	::testing::AssertionResult renderAndCheck(const char* sceneId, bool wavefront) {
#ifdef _WIN32
		_putenv_s("RAY_TRACER_WAVEFRONT", wavefront ? "1" : "");
#else
		if (wavefront) setenv("RAY_TRACER_WAVEFRONT", "1", 1);
		else            unsetenv("RAY_TRACER_WAVEFRONT");
#endif
		optix_clear_validation_issues();
		// Tiny resolution/spp/depth on purpose: this sweep exists to
		// exercise material/light/SBT code paths and let OptiX's
		// validation layer inspect the launches - the pixels themselves
		// are never read, so image quality is irrelevant.
		const int result = optix_render_main(32, 32, 8, 4, outPath_.c_str(),
			sceneId, 278.0, 278.0, -800.0);
		if (result != 0) {
			return ::testing::AssertionFailure()
				<< "render failed with error code " << result;
		}
		const int issues = optix_validation_issue_count();
		if (issues > 0) {
			::testing::AssertionResult failure = ::testing::AssertionFailure();
			failure << "OptiX reported " << issues << " issue(s):";
			for (int i = 0; i < issues; ++i) {
				failure << "\n  " << optix_validation_issue(i);
			}
			return failure;
		}
		return ::testing::AssertionSuccess();
	}

	std::string outPath_ = "validation_sweep_out.ppm";
	std::string warmupPath_ = "validation_sweep_warmup.ppm";
};

TEST_F(OptixValidationSweepTest, RecursiveBackend) {
	for (const auto& scene : kSweepScenes) {
		SCOPED_TRACE(std::string("scene ") + scene.id + " (" + scene.name + ")");
		EXPECT_TRUE(renderAndCheck(scene.id, /*wavefront=*/false));
	}
}

TEST_F(OptixValidationSweepTest, WavefrontBackend) {
	for (const auto& scene : kSweepScenes) {
		SCOPED_TRACE(std::string("scene ") + scene.id + " (" + scene.name + ")");
		EXPECT_TRUE(renderAndCheck(scene.id, /*wavefront=*/true));
	}
}

// Regression test for the exact sequence that surfaced the wavefront
// shadow miss-SBT-index bug during manual investigation (tracked as tasks
// #106/#107/#108): a zero-light scene immediately followed by a lit scene,
// same process, wavefront backend.
TEST_F(OptixValidationSweepTest, WavefrontZeroLightThenLitSceneSequence) {
	EXPECT_TRUE(renderAndCheck("A6", /*wavefront=*/true));  // Colored Quads, zero lights
	EXPECT_TRUE(renderAndCheck("A1", /*wavefront=*/true));  // Cornell Box, has lights
}
