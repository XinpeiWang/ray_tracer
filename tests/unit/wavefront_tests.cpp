/**
 * @file wavefront_tests.cpp
 * @brief Unit tests for the wavefront GPU path tracer.
 *
 * Tests aligned with pbrt-v4 wavefront integrator design:
 *   - WorkQueue type layout and size invariants
 *   - RayWorkItem / HitWorkItem / ShadowRayWorkItem field coverage
 *   - WavefrontLaunchParams layout sanity
 *   - Wavefront render output correctness (NonBlack, Dimensions, Brightness)
 *   - Wavefront and recursive renderers produce consistent output
 */

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cstdio>

// ============================================================================
// Work-item type tests (host-side, no CUDA)
// ============================================================================
// We can include wavefront_types.h from the host side: the __CUDACC__ guard
// hides device-only push/size methods, and the struct layout is plain C POD.
#ifndef __CUDACC__
#include "wavefront_types.h"  // from gpu/optix/ (added to include dirs in vcxproj)
#endif

// ============================================================================
// WorkQueue struct layout tests
// ============================================================================

TEST(WavefrontTypes, WorkQueueLayout) {
	// WorkQueue<T> must be a plain POD struct with pointer + counter + capacity.
	// This validates it hasn't accidentally grown (e.g. vtable, padding surprises).
	WorkQueue<RayWorkItem> q = {};
	EXPECT_EQ(q.items,    nullptr);
	EXPECT_EQ(q.counter,  nullptr);
	EXPECT_EQ(q.capacity, 0);
	(void)q;
}

TEST(WavefrontTypes, RayWorkItemHasSpecularBounce) {
	RayWorkItem r = {};
	r.specular_bounce = 1;
	EXPECT_EQ(r.specular_bounce, 1);
	r.specular_bounce = 0;
	EXPECT_EQ(r.specular_bounce, 0);
}

TEST(WavefrontTypes, HitWorkItemHasSpecularBounce) {
	HitWorkItem h = {};
	h.specular_bounce = 1;
	EXPECT_EQ(h.specular_bounce, 1);
}

TEST(WavefrontTypes, RayWorkItemFieldCoverage) {
	// Verify all expected fields exist and are zero-initializable.
	RayWorkItem r = {};
	EXPECT_EQ(r.origin.x,    0.0f);
	EXPECT_EQ(r.origin.y,    0.0f);
	EXPECT_EQ(r.origin.z,    0.0f);
	EXPECT_EQ(r.direction.x, 0.0f);
	// Spectral arrays (kWFNWavelengths = 4)
	EXPECT_EQ(r.throughput[0], 0.0f);
	EXPECT_EQ(r.radiance[0],   0.0f);
	EXPECT_EQ(r.wavelengths[0], 0.0f);
	EXPECT_EQ(r.wavelength_pdfs[0], 0.0f);
	EXPECT_EQ(r.seed,        0u);
	EXPECT_EQ(r.pixelIndex,  0);
	EXPECT_EQ(r.depth,       0);
	EXPECT_EQ(r.specular_bounce, 0);
	EXPECT_EQ(r.tMin,        0.0f);
	EXPECT_EQ(r.tMax,        0.0f);
}

TEST(WavefrontTypes, HitWorkItemFieldCoverage) {
	HitWorkItem h = {};
	EXPECT_EQ(h.hitPoint.x,  0.0f);
	EXPECT_EQ(h.normal.x,    0.0f);
	EXPECT_EQ(h.t,           0.0f);
	EXPECT_EQ(h.materialIdx, 0);
	EXPECT_EQ(h.geomType,    0);
	EXPECT_EQ(h.rayOrigin.x, 0.0f);
	EXPECT_EQ(h.rayDir.x,    0.0f);
	// Spectral arrays
	EXPECT_EQ(h.throughput[0], 0.0f);
	EXPECT_EQ(h.radiance[0],   0.0f);
	EXPECT_EQ(h.wavelengths[0], 0.0f);
	EXPECT_EQ(h.wavelength_pdfs[0], 0.0f);
	EXPECT_EQ(h.seed,        0u);
	EXPECT_EQ(h.pixelIndex,  0);
	EXPECT_EQ(h.depth,       0);
	EXPECT_EQ(h.specular_bounce, 0);
}

TEST(WavefrontTypes, ShadowRayWorkItemFieldCoverage) {
	ShadowRayWorkItem s = {};
	EXPECT_EQ(s.origin.x,    0.0f);
	EXPECT_EQ(s.direction.x, 0.0f);
	EXPECT_EQ(s.tMax,        0.0f);
	EXPECT_EQ(s.Ld[0],       0.0f);  // spectral direct-light; not float3 anymore
	EXPECT_EQ(s.pixelIndex,  0);
}

TEST(WavefrontTypes, MissWorkItemFieldCoverage) {
	MissWorkItem m = {};
	EXPECT_EQ(m.throughput[0], 0.0f);  // spectral array
	EXPECT_EQ(m.rayDir.x,      0.0f);
	EXPECT_EQ(m.pixelIndex,    0);
}

TEST(WavefrontTypes, WavefrontQueuesLayout) {
	// All queue fields must be zero-initializable.
	WavefrontQueues q = {};
	EXPECT_EQ(q.rayQueue.items,    nullptr);
	EXPECT_EQ(q.hitQueue.items,    nullptr);
	// Lambertian/Metal hits route here instead of hitQueue - see
	// WavefrontQueues::simpleHitQueue's comment (wavefront_types.h).
	EXPECT_EQ(q.simpleHitQueue.items, nullptr);
	EXPECT_EQ(q.missQueue.items,   nullptr);
	EXPECT_EQ(q.shadowQueue.items, nullptr);
	EXPECT_EQ(q.nextRayQueue.items, nullptr);
}

TEST(WavefrontTypes, WavefrontLaunchParamsLayout) {
#ifdef OPTIX_RENDERER_AVAILABLE
	WavefrontLaunchParams lp = {};
	EXPECT_EQ(lp.framebuffer,  nullptr);
	EXPECT_EQ(lp.width,        0u);
	EXPECT_EQ(lp.height,       0u);
	// These fields depend on SphereData*/QuadData*/MaterialData* from optix_types.h
	EXPECT_EQ(lp.spheres,      nullptr);
	EXPECT_EQ(lp.quads,        nullptr);
	EXPECT_EQ(lp.materials,    nullptr);
	EXPECT_EQ(lp.simpleHitQueue.items, nullptr);
	EXPECT_EQ(lp.numSpheres,   0u);
	EXPECT_EQ(lp.numQuads,     0u);
	EXPECT_EQ(lp.numMaterials, 0u);
	EXPECT_EQ(lp.numLights,    0u);
	EXPECT_EQ(lp.samplesPerPixel, 0u);
	EXPECT_EQ(lp.maxDepth,     0u);
#else
	GTEST_SKIP() << "WavefrontLaunchParams requires OPTIX_RENDERER_AVAILABLE";
#endif
}

// ============================================================================
// pbrt-v4 alignment: specular_bounce flag semantics
// ============================================================================

TEST(WavefrontTypes, PrimaryRayIsSpecularBounce) {
	// pbrt-v4 invariant: primary camera rays are treated as specular bounces
	// so that emissive surfaces hit directly by the camera are counted.
	RayWorkItem primary = {};
	primary.depth           = 0;
	primary.specular_bounce = 1;  // must be set for primary rays
	EXPECT_EQ(primary.specular_bounce, 1)
		<< "Primary rays must have specular_bounce=1 for correct emissive handling";
}

TEST(WavefrontTypes, NonSpecularBounceFlag) {
	// After a Lambertian scatter, specular_bounce must be 0
	// so that NEE shadow rays are not double-counted with emissive hits.
	RayWorkItem diffuse_bounce = {};
	diffuse_bounce.depth           = 1;
	diffuse_bounce.specular_bounce = 0;
	EXPECT_EQ(diffuse_bounce.specular_bounce, 0)
		<< "Diffuse/Lambertian bounces must have specular_bounce=0";
}

TEST(WavefrontTypes, SpecularBounceFlag) {
	// After a specular scatter (Metal, Dielectric, etc.), specular_bounce must be 1.
	RayWorkItem specular_bounce = {};
	specular_bounce.depth           = 1;
	specular_bounce.specular_bounce = 1;
	EXPECT_EQ(specular_bounce.specular_bounce, 1)
		<< "Specular bounces must have specular_bounce=1";
}

// ============================================================================
// pbrt-v4 alignment: ShadowRayWorkItem carries only Ld (not packed radiance)
// ============================================================================

TEST(WavefrontTypes, ShadowRayLdIsOnlyDirectLight) {
	// pbrt-v4 pattern: ShadowRayWorkItem.Ld is the pending direct-light contribution.
	// Prior-bounce accumulated radiance must be written to the framebuffer
	// unconditionally at the NEE site, NOT packed into Ld.
	// Spectral port: Ld is now float[kWFNWavelengths] (hero-wavelength spectral array).
	ShadowRayWorkItem s = {};
	for (int i = 0; i < kWFNWavelengths; ++i)
		s.Ld[i] = float(i + 1);
	for (int i = 0; i < kWFNWavelengths; ++i)
		EXPECT_FLOAT_EQ(s.Ld[i], float(i + 1));
}

// ============================================================================
// GPU wavefront render output tests
// ============================================================================
// These tests require OptiX and the wavefront PTX to be available.
// They are skipped gracefully when OptiX is not present.
// They only compile when OPTIX_RENDERER_AVAILABLE is defined (i.e., when
// optix_renderer.lib is linked into the test binary).

#ifdef OPTIX_RENDERER_AVAILABLE

extern "C" {
	#include "optix_interface.h"
}

/// Simple PPM loader (ASCII P3)
struct WFPPMImage {
	int width = 0, height = 0;
	std::vector<float> pixels;  // normalized [0,1] RGB
	bool valid = false;
};

static WFPPMImage wf_load_ppm(const char* path) {
	WFPPMImage img;
	std::ifstream f(path);
	if (!f.good()) return img;
	std::string magic;
	int maxVal;
	f >> magic >> img.width >> img.height >> maxVal;
	if (magic != "P3" || maxVal <= 0) return img;
	int total = img.width * img.height * 3;
	img.pixels.resize(total);
	for (int i = 0; i < total; ++i) {
		int v; f >> v;
		img.pixels[i] = float(v) / float(maxVal);
	}
	img.valid = (f.good() || f.eof());
	return img;
}

static float wf_avg_brightness(const WFPPMImage& img) {
	if (img.pixels.empty()) return 0.0f;
	float s = std::accumulate(img.pixels.begin(), img.pixels.end(), 0.0f);
	return s / float(img.pixels.size());
}

static float wf_black_fraction(const WFPPMImage& img) {
	if (!img.valid || img.width == 0) return 1.0f;
	int cnt = 0;
	for (int i = 0; i < img.width * img.height; ++i) {
		if (img.pixels[i*3]==0 && img.pixels[i*3+1]==0 && img.pixels[i*3+2]==0) ++cnt;
	}
	return float(cnt) / float(img.width * img.height);
}

class WavefrontRenderTest : public ::testing::Test {
protected:
	void SetUp() override {
		if (!optix_is_available()) {
			GTEST_SKIP() << "OptiX not available; skipping wavefront GPU render tests";
		}
		// Enable wavefront mode for this test class
		prev_wf_ = getenv("RAY_TRACER_WAVEFRONT");
#ifdef _WIN32
		_putenv_s("RAY_TRACER_WAVEFRONT", "1");
#else
		setenv("RAY_TRACER_WAVEFRONT", "1", 1);
#endif
	}

	void TearDown() override {
		// Restore env var
#ifdef _WIN32
		if (prev_wf_) {
			_putenv_s("RAY_TRACER_WAVEFRONT", prev_wf_);
		} else {
			_putenv_s("RAY_TRACER_WAVEFRONT", "");
		}
#else
		if (prev_wf_) setenv("RAY_TRACER_WAVEFRONT", prev_wf_, 1);
		else          unsetenv("RAY_TRACER_WAVEFRONT");
#endif
		for (const auto& f : files_) std::remove(f.c_str());
	}

	WFPPMImage renderWF(const char* name, int w, int h, int spp, int depth,
						double cx = 278, double cy = 278, double cz = -800,
						const char* scene_id = "A1") {
		files_.push_back(name);
		if (optix_render_main(w, h, spp, depth, name, scene_id, cx, cy, cz) != 0)
			return {};
		return wf_load_ppm(name);
	}

	std::vector<std::string> files_;
	const char* prev_wf_ = nullptr;
};

// Test 1: Wavefront output is not completely black
TEST_F(WavefrontRenderTest, NonBlackOutput) {
	auto img = renderWF("wf_test_nonblack.ppm", 80, 80, 200, 8);
	ASSERT_TRUE(img.valid) << "Wavefront render failed to produce a valid PPM";
	float bf = wf_black_fraction(img);
	EXPECT_LT(bf, 0.95f)
		<< "More than 95% of pixels are black (black fraction=" << bf
		<< "). Wavefront renderer may not be working.";
}

// Regression test for task #106: a wavefront scene with ZERO emissive
// lights, immediately followed by any scene WITH lights in the same
// process, used to crash the shadow pipeline (illegal memory access /
// cudaStreamSynchronize failure at wavefront_path_tracer.cpp's shadow
// launch sync). Bisected to be independent of material type, textures,
// geometry composition, and resolution - reproduced with a scene using
// only plain pre-existing Lambertian quads and no textures (scene 5)
// followed by the Cornell box (scene 0). compute-sanitizer's memcheck,
// racecheck, and synccheck all reported zero errors against this repro,
// which made sense once the real cause surfaced while fixing task #107:
// __raygen__wf_shadow (wavefront_programs.cu) passed missSBTIndex=1 to
// optixTrace, but shadowSBT_ has exactly one miss record (index 0) - an
// out-of-bounds read into adjacent heap memory whose contents depend on
// allocation history, which is exactly why a zero-light scene (different
// alloc/free pattern for the light index/kind/alias-table buffers) right
// before a lit scene was what it took to land on bytes that faulted.
TEST_F(WavefrontRenderTest, ZeroLightSceneThenLitSceneDoesNotCrash) {
	auto img1 = renderWF("wf_diag_zerolight1.ppm", 80, 80, 50, 5, 278, 278, -800, "A6");
	ASSERT_TRUE(img1.valid) << "Scene A6 (zero lights) render failed";
	auto img2 = renderWF("wf_diag_zerolight2.ppm", 80, 80, 50, 5, 278, 278, -800, "A1");
	ASSERT_TRUE(img2.valid) << "Scene A1 render after scene A6 (zero lights) failed";
}

// Regression test for task #107: scene 8 (Final Scene) rendered solid black
// on the wavefront backend because WavefrontPathTracer::initialize() set
// pipelineCompileOptions_.usesMotionBlur=false while tracing against the
// SAME IAS/GAS the recursive backend builds via OptiXRenderer::buildScene() -
// which gets motionOptions.numKeys=2 (see its sceneHasMotion_ detection)
// whenever a scene has a moving sphere, as scene 8's does. Tracing a
// motion-enabled traversable from a pipeline compiled with
// usesMotionBlur=false is undefined behavior in OptiX; here it made every
// primary ray report a miss on the very first intersect launch. Fixed by
// setting usesMotionBlur=true unconditionally (matching the recursive
// pipeline's own always-on setting, safe for motion and non-motion scenes
// alike - see both files' comments at that assignment).
TEST_F(WavefrontRenderTest, Scene8FinalSceneIsNotBlack) {
	// cx/cy/cz are ignored for scene 8 (a Fixed-mode camera scene - see the
	// case 8 block in scene_builder.cpp), so the renderWF() defaults are
	// fine here.
	auto img = renderWF("wf_test_scene8.ppm", 80, 80, 200, 8, 278, 278, -800, "A9");
	ASSERT_TRUE(img.valid) << "Wavefront render of scene A9 failed to produce a valid PPM";
	float bf = wf_black_fraction(img);
	EXPECT_LT(bf, 0.95f)
		<< "More than 95% of pixels are black (black fraction=" << bf
		<< "). Scene 8's motion-blur/traversable mismatch may have regressed.";
}

// Test 2: Correct output dimensions
TEST_F(WavefrontRenderTest, CorrectDimensions) {
	struct C { int w, h; };
	for (auto [w, h] : std::vector<C>{{64,64},{100,80},{80,100}}) {
		std::string name = "wf_test_dims_" + std::to_string(w) + "x" + std::to_string(h) + ".ppm";
		auto img = renderWF(name.c_str(), w, h, 50, 5);
		ASSERT_TRUE(img.valid) << "Failed for " << w << "x" << h;
		EXPECT_EQ(img.width,  w);
		EXPECT_EQ(img.height, h);
	}
}

// Test 3: Average brightness in physically plausible range
// Cornell box lit with area light — not too dark, not saturated.
TEST_F(WavefrontRenderTest, BrightnessInRange) {
	auto img = renderWF("wf_test_brightness.ppm", 80, 80, 500, 8);
	ASSERT_TRUE(img.valid) << "Wavefront render failed";
	float avg = wf_avg_brightness(img);
	EXPECT_GT(avg, 0.002f) << "Image too dark (avg=" << avg << ")";
	EXPECT_LT(avg, 0.95f)  << "Image too bright/saturated (avg=" << avg << ")";
}

// Test 4: pbrt-v4 alignment — wavefront and recursive renderers produce
// similar average brightness for the same scene (within 3x factor).
// The wavefront uses naive path tracing (no importance sampling), so it needs
// more samples for the same quality, but the average illumination should be
// in the same order of magnitude.
TEST_F(WavefrontRenderTest, WavefrontBrightnessConsistentWithRecursive) {
	// Render with wavefront (already set via SetUp)
	auto wf = renderWF("wf_test_consistency_wf.ppm", 60, 60, 1000, 8);
	ASSERT_TRUE(wf.valid) << "Wavefront render failed";

	// Disable wavefront and render with recursive tracer
#ifdef _WIN32
	_putenv_s("RAY_TRACER_WAVEFRONT", "0");
#else
	setenv("RAY_TRACER_WAVEFRONT", "0", 1);
#endif
	// Recursive has real importance sampling (MIS/NEE), so it doesn't need
	// anywhere near wavefront's 1000spp to produce a stable average for a
	// 10x-factor tolerance check.
	files_.push_back("wf_test_consistency_rec.ppm");
	WFPPMImage rec;
	if (optix_render_main(60, 60, 150, 8, "wf_test_consistency_rec.ppm", "A1",
						  278, 278, -800) == 0) {
		rec = wf_load_ppm("wf_test_consistency_rec.ppm");
	}
	// Re-enable wavefront for TearDown
#ifdef _WIN32
	_putenv_s("RAY_TRACER_WAVEFRONT", "1");
#else
	setenv("RAY_TRACER_WAVEFRONT", "1", 1);
#endif

	if (!rec.valid) {
		GTEST_SKIP() << "Recursive render failed; skipping consistency check";
	}

	float wf_avg  = wf_avg_brightness(wf);
	float rec_avg = wf_avg_brightness(rec);

	// Both should produce similar scene illumination within 3x factor
	EXPECT_GT(wf_avg, rec_avg * 0.1f)
		<< "Wavefront too dark vs recursive: wf=" << wf_avg << " rec=" << rec_avg;
	EXPECT_LT(wf_avg, rec_avg * 10.0f)
		<< "Wavefront too bright vs recursive: wf=" << wf_avg << " rec=" << rec_avg;
}

// Test 5: Different samples per pixel produce different wavefront output
TEST_F(WavefrontRenderTest, DifferentSPPProducesDifferentOutput) {
	// Only needs to prove random sampling noise differs between the two
	// runs (avg_diff > 0.001) - 100 vs 10 is already an easy margin for that.
	auto img_lo = renderWF("wf_test_spp_lo.ppm", 60, 60,  10, 5);
	auto img_hi = renderWF("wf_test_spp_hi.ppm", 60, 60, 100, 5);
	ASSERT_TRUE(img_lo.valid) << "Low-SPP wavefront render failed";
	ASSERT_TRUE(img_hi.valid) << "High-SPP wavefront render failed";

	float diff = 0.0f;
	for (size_t i = 0; i < img_lo.pixels.size(); ++i)
		diff += std::abs(img_lo.pixels[i] - img_hi.pixels[i]);
	float avg_diff = diff / float(img_lo.pixels.size());

	EXPECT_GT(avg_diff, 0.001f)
		<< "Wavefront renders at 10spp and 1000spp are identical "
		<< "(avg_diff=" << avg_diff << "); random sampling should differ";
}

#endif // OPTIX_RENDERER_AVAILABLE
