/**
 * @file cpu_gpu_comparison_tests.cpp
 * @brief CPU vs GPU renderer comparison tests (pbrt-v4 style)
 *
 * Inspired by pbrt-v4's multi-integrator test pattern where the same scene
 * is rendered by different integrators and results are compared.
 *
 * Key insight: CPU (importance sampling) and GPU (naive path tracing) should:
 * - Both produce non-black output for the same scene
 * - Converge toward similar average brightness at sufficient sample counts
 * - Both correctly represent the Cornell box color palette (warm/cool tones)
 *
 * Tests:
 * - BothProduceNonBlackOutput:       neither renderer is broken
 * - BothHaveSimilarBrightness:       converge to same integral value
 * - CPUIsBrighterAtLowSPP:           CPU importance sampling is more efficient
 * - BothShowColorVariation:          red/green walls produce color in image
 * - HighSPPConvergence:              both approach same answer at high samples
 */

#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <memory>
#include <cctype>

#include "scene_registry.h"

extern "C" {
	#include "cpu_interface.h"
	#include "optix_interface.h"
}

// ============================================================================
// Shared utilities
// ============================================================================

struct Image {
	int width = 0, height = 0;
	std::vector<float> pixels;  // RGB normalized [0,1]
	bool valid = false;
};

Image load_image(const char* path) {
	Image img;
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
		img.pixels[i] = static_cast<float>(v) / static_cast<float>(maxVal);
	}
	img.valid = true;
	return img;
}

float avg_brightness(const Image& img) {
	if (img.pixels.empty()) return 0.0f;
	float s = std::accumulate(img.pixels.begin(), img.pixels.end(), 0.0f);
	return s / static_cast<float>(img.pixels.size());
}

// Average red, green, blue channels separately
struct RGBAverage { float r, g, b; };
RGBAverage avg_channels(const Image& img) {
	RGBAverage out{0, 0, 0};
	int n = img.width * img.height;
	if (n == 0) return out;
	for (int i = 0; i < n; ++i) {
		out.r += img.pixels[i * 3 + 0];
		out.g += img.pixels[i * 3 + 1];
		out.b += img.pixels[i * 3 + 2];
	}
	out.r /= n; out.g /= n; out.b /= n;
	return out;
}

// ============================================================================
// CPU vs GPU Comparison Tests
// ============================================================================

class CPUGPUComparisonTest : public ::testing::Test {
protected:
	void SetUp() override {
		gpuAvailable_ = optix_is_available();
	}

	void TearDown() override {
		for (const auto& f : files_) std::remove(f.c_str());
	}

	Image renderCPU(const char* filename, int w, int h, int spp, int depth) {
		files_.push_back(filename);
		cpu_render_main(w, h, spp, depth, filename, "A1", 278.0, 278.0, -800.0);
		return load_image(filename);
	}

	Image renderGPU(const char* filename, int w, int h, int spp, int depth) {
		files_.push_back(filename);
		if (optix_render_main(w, h, spp, depth, filename, "A1", 278.0, 278.0, -800.0) != 0)
			return {};   // invalid image; caller should check image.valid
		return load_image(filename);
	}

	bool gpuAvailable_ = false;
	std::vector<std::string> files_;
};

// ----------------------------------------------------------------------------
// Test 1: Both renderers produce non-black output on the same scene
// (pbrt-v4 equivalent: all integrators produce non-zero output)
// ----------------------------------------------------------------------------
TEST_F(CPUGPUComparisonTest, BothProduceNonBlackOutput) {
	Image cpu = renderCPU("cmp_cpu_nonblack.ppm", 80, 80, 20, 5);
	ASSERT_TRUE(cpu.valid) << "CPU render failed";

	float cpuBrightness = avg_brightness(cpu);
	EXPECT_GT(cpuBrightness, 0.001f)
		<< "CPU output is black. Average brightness: " << cpuBrightness;

	if (!gpuAvailable_) {
		GTEST_SKIP() << "OptiX not available — GPU comparison skipped";
	}

	Image gpu = renderGPU("cmp_gpu_nonblack.ppm", 80, 80, 500, 5);
	ASSERT_TRUE(gpu.valid) << "GPU render failed";

	float gpuBrightness = avg_brightness(gpu);
	EXPECT_GT(gpuBrightness, 0.001f)
		<< "GPU output is black. Average brightness: " << gpuBrightness;
}

// ----------------------------------------------------------------------------
// Test 2: At high sample counts, CPU and GPU should have similar brightness
// (pbrt-v4 pattern: multiple integrators converge to same value)
// Allow 50% relative difference since GPU uses naive vs CPU importance sampling
// ----------------------------------------------------------------------------
TEST_F(CPUGPUComparisonTest, HighSPPBrightnessConverges) {
	if (!gpuAvailable_) GTEST_SKIP() << "OptiX not available";

	// Use sufficient samples for both to have meaningful statistics. GPU spp
	// is well above CPU's since it's naive (no importance sampling) - it
	// needs more samples for the same variance, but not 2000: the 50%
	// tolerance below is generous enough that 500 already converges reliably.
	Image cpu = renderCPU("cmp_cpu_converge.ppm", 80, 80, 100, 10);
	Image gpu = renderGPU("cmp_gpu_converge.ppm", 80, 80, 500, 10);

	ASSERT_TRUE(cpu.valid) << "CPU render failed";
	ASSERT_TRUE(gpu.valid) << "GPU render failed";

	float cpuBright = avg_brightness(cpu);
	float gpuBright = avg_brightness(gpu);

	// Both must be non-zero
	EXPECT_GT(cpuBright, 0.001f) << "CPU result is black";
	EXPECT_GT(gpuBright, 0.001f) << "GPU result is black";

	// Relative difference should be < 50%
	// (generous because GPU is naive path tracing vs CPU importance sampling)
	float maxBright = std::max(cpuBright, gpuBright);
	float relDiff = std::abs(cpuBright - gpuBright) / maxBright;

	EXPECT_LT(relDiff, 0.50f)
		<< "CPU brightness=" << cpuBright << " GPU brightness=" << gpuBright
		<< " relative difference=" << relDiff
		<< " — renderers disagree by more than 50%. "
		<< "They should converge to the same scene integral.";
}

// ----------------------------------------------------------------------------
// Test 3: Both CPU and GPU should produce adequately bright results at low SPP
// The GPU uses MIS/NEE just like CPU, so both should be in a similar range.
// ----------------------------------------------------------------------------
TEST_F(CPUGPUComparisonTest, CPUIsBrighterAtLowSPP) {
	if (!gpuAvailable_) GTEST_SKIP() << "OptiX not available";

	// Low SPP where efficiency difference is most visible
	Image cpu = renderCPU("cmp_cpu_lowspp.ppm", 80, 80, 10, 5);
	Image gpu = renderGPU("cmp_gpu_lowspp.ppm", 80, 80, 10, 5);

	ASSERT_TRUE(cpu.valid) << "CPU render failed";
	ASSERT_TRUE(gpu.valid) << "GPU render failed";

	float cpuBright = avg_brightness(cpu);
	float gpuBright = avg_brightness(gpu);

	// Both renderers use importance sampling (MIS/NEE), so both should produce
	// a meaningful (non-black) image at low SPP.
	EXPECT_GT(cpuBright, 0.05f) << "CPU image is unexpectedly dark at low SPP";
	EXPECT_GT(gpuBright, 0.05f) << "GPU image is unexpectedly dark at low SPP";

	// Neither should be more than 3x brighter than the other
	// (guards against one renderer being completely broken)
	float ratio = (cpuBright > gpuBright) ? cpuBright / gpuBright : gpuBright / cpuBright;
	EXPECT_LT(ratio, 3.0f)
		<< "Large brightness gap: CPU=" << cpuBright << " GPU=" << gpuBright
		<< " — one renderer may have a sampling bug.";
}

// ----------------------------------------------------------------------------
// Test 4: Cornell box should show color variation (red left, green right wall)
// Both renderers should produce images where one channel dominates in corners
// ----------------------------------------------------------------------------
TEST_F(CPUGPUComparisonTest, CPUShowsColorVariation) {
	Image img = renderCPU("cmp_cpu_color.ppm", 100, 100, 50, 10);
	ASSERT_TRUE(img.valid) << "CPU render failed";

	RGBAverage avg = avg_channels(img);
	float total = avg.r + avg.g + avg.b;

	if (total < 0.001f) GTEST_SKIP() << "Image too dark to analyze color";

	// Cornell box has red left wall and green right wall
	// The image should have meaningful content in both R and G channels
	float rFrac = avg.r / total;
	float gFrac = avg.g / total;

	// Neither channel should be totally absent (would mean we're missing a wall)
	EXPECT_GT(rFrac, 0.05f) << "Red channel fraction " << rFrac << " too low — red wall missing?";
	EXPECT_GT(gFrac, 0.05f) << "Green channel fraction " << gFrac << " too low — green wall missing?";
}

TEST_F(CPUGPUComparisonTest, GPUShowsColorVariation) {
	if (!gpuAvailable_) GTEST_SKIP() << "OptiX not available";

	// Only checking gross per-channel fractions (>5%), not fine brightness
	// matching - doesn't need anywhere near 1000spp to be stable.
	Image img = renderGPU("cmp_gpu_color.ppm", 100, 100, 150, 10);
	ASSERT_TRUE(img.valid) << "GPU render failed";

	RGBAverage avg = avg_channels(img);
	float total = avg.r + avg.g + avg.b;

	if (total < 0.001f) GTEST_SKIP() << "Image too dark to analyze color";

	float rFrac = avg.r / total;
	float gFrac = avg.g / total;

	EXPECT_GT(rFrac, 0.05f) << "GPU red channel fraction " << rFrac << " too low — red wall missing?";
	EXPECT_GT(gFrac, 0.05f) << "GPU green channel fraction " << gFrac << " too low — green wall missing?";
}

// ----------------------------------------------------------------------------
// Test 5: Scene ID 0 (Cornell box) must render the same for both renderers
// at matching resolutions — output files must exist and have same dimensions
// ----------------------------------------------------------------------------
TEST_F(CPUGPUComparisonTest, BothProduceSameDimensions) {
	if (!gpuAvailable_) GTEST_SKIP() << "OptiX not available";

	struct Case { int w, h; };
	for (auto [w, h] : std::vector<Case>{{64, 64}, {128, 128}}) {
		std::string cpuFile = "cmp_dims_cpu_" + std::to_string(w) + ".ppm";
		std::string gpuFile = "cmp_dims_gpu_" + std::to_string(w) + ".ppm";

		Image cpu = renderCPU(cpuFile.c_str(), w, h, 5, 3);
		Image gpu = renderGPU(gpuFile.c_str(), w, h, 100, 3);

		ASSERT_TRUE(cpu.valid) << "CPU render failed at " << w << "x" << h;
		ASSERT_TRUE(gpu.valid) << "GPU render failed at " << w << "x" << h;

		EXPECT_EQ(cpu.width,  gpu.width)  << "Width mismatch at " << w << "x" << h;
		EXPECT_EQ(cpu.height, gpu.height) << "Height mismatch at " << w << "x" << h;
		EXPECT_EQ(cpu.width,  w) << "CPU width incorrect";
		EXPECT_EQ(gpu.width,  w) << "GPU width incorrect";
	}
}

// ============================================================================
// CPU/GPU scene-definition parity: light count
//
// Every scene's CPU builder (src/TheRestOfYourLife/scenes*.h) and GPU builder
// (gpu/optix/scene_builder.cpp) are independent, hand-written functions that
// are supposed to describe the same scene - there's no shared source of
// truth, so they can silently drift (this is exactly how scene 0's Cornell
// box ended up with a CPU-only accent light: scenes_book.h::build_cornell_box()
// has 2 diffuse_light emitters, scene_builder.cpp::build_cornell_box() only
// uploads 1). Pixel-brightness comparisons (see BothProduceNonBlackOutput /
// HighSPPBrightnessConverges above) don't catch this class of bug: a small
// accent light easily hides inside their existing 50%/3x tolerance bands.
// This check instead compares scene *structure* directly - number of
// emissive primitives - which is exact, not statistical.
// ============================================================================

// Recursively counts emissive (diffuse_light-material) quad/sphere/triangle
// primitives in a CPU hittable subtree. Descends through hittable_list/
// translate/rotate_y, the only wrapper types this codebase's scene builders
// place area lights inside/under. Treated as opaque (0 lights) otherwise:
// bvh_node, constant_medium, bilinear_patch, curve, etc. - verified against
// every current scene builder that no light is ever nested inside one of
// these (bvh_node in particular wraps bulk non-emissive geometry piles, e.g.
// build_triangle_mesh_scene's icosahedron, with lights always added as
// separate top-level/translate/rotate_y siblings). If a future scene breaks
// that assumption, this under-counts on the CPU side and the test below
// fails loudly rather than silently passing.
//
// triangle is counted alongside quad/sphere, not left opaque like the rest
// of that list: pbrt_cpu_builder.h's AreaLightSource handling (see
// pbrt_cpu_builder.h) adds individual emissive triangles straight to the
// world/lights lists for a trianglemesh area light - exactly the shape every
// bundled pbrt_scenes/*.pbrt example scene's light uses (two triangles
// forming a quad), so leaving triangle opaque here silently reported 0 CPU
// lights against a real, non-zero GPU count (GpuLightKind::Triangle) for
// every one of them the moment they stopped being skipped by
// SceneDescriptor::requires_files.
static int count_cpu_emissive_lights(const std::shared_ptr<hittable>& h);

static int count_cpu_emissive_lights_list(const hittable_list& list) {
	int n = 0;
	for (const auto& obj : list.objects) n += count_cpu_emissive_lights(obj);
	return n;
}

static int count_cpu_emissive_lights(const std::shared_ptr<hittable>& h) {
	if (!h) return 0;
	if (auto hl = std::dynamic_pointer_cast<hittable_list>(h)) return count_cpu_emissive_lights_list(*hl);
	if (auto tr = std::dynamic_pointer_cast<translate>(h))     return count_cpu_emissive_lights(tr->get_object());
	if (auto ry = std::dynamic_pointer_cast<rotate_y>(h))      return count_cpu_emissive_lights(ry->get_object());
	if (auto q = std::dynamic_pointer_cast<quad>(h))
		return std::dynamic_pointer_cast<diffuse_light>(q->get_material()) ? 1 : 0;
	if (auto s = std::dynamic_pointer_cast<sphere>(h))
		return std::dynamic_pointer_cast<diffuse_light>(s->get_material()) ? 1 : 0;
	if (auto t = std::dynamic_pointer_cast<triangle>(h))
		return std::dynamic_pointer_cast<diffuse_light>(t->get_material()) ? 1 : 0;
	return 0;
}

class CpuGpuLightParityTest : public ::testing::TestWithParam<int> {};

TEST_P(CpuGpuLightParityTest, LightCountMatches) {
	// GetParam() is a registry POSITION, not a scene id (ids are category
	// letter + number now, not contiguous ints - see scene_registry.h's
	// SceneDescriptor::id comment), so resolve id via the registry first.
	const SceneDescriptor& desc = get_scene_registry()[GetParam()];
	const SceneDescriptor* s = find_scene(desc.id);
	ASSERT_NE(s, nullptr) << "Missing scene id " << desc.id;
	if (!s->gpu_compatible) GTEST_SKIP() << s->name << " is not GPU-compatible";
	if (s->requires_files) GTEST_SKIP() << s->name << " requires external assets";

	// pbrt-loaded scenes (category CustomScenes) are the one case where
	// build_world()'s top-level walk can't see the lights at all:
	// pbrt_cpu_builder.h's build() wraps the ENTIRE world - lights included -
	// in a single bvh_node for trace performance (see its own "a flat list
	// would make every ray test every primitive" comment), and bvh_node is
	// intentionally opaque to count_cpu_emissive_lights (its own comment
	// explains why: every OTHER scene builder keeps lights as un-BVH'd
	// top-level/translate/rotate_y siblings, so walking past a bvh_node was
	// safe to skip - until a builder came along that BVH-wraps the lights
	// too). build_lights() sidesteps this cleanly: pbrt_cpu_builder.h adds
	// every emissive triangle/sphere/patch to it in the same statement that
	// adds it to world (see e.g. its `world.add(tri); ... lights.add(tri);`
	// pairing), so it is always a flat, accurate, un-BVH'd mirror of the
	// world's emissive shapes for these scenes specifically. Every other
	// category keeps using build_world() - unchanged from before - since
	// build_lights() isn't guaranteed to carry real emissive materials
	// there: e.g. cornell_box_scene.h's build_cornell_box_lights() (used by
	// A1 and several others) returns NEE sampling-target shapes built with
	// an empty_material placeholder, not the actual emissive objects -
	// correct for its own purpose (picking directions toward known light
	// shapes) but a silent 0 if fed through count_cpu_emissive_lights here.
	//
	// F3 is included alongside category CustomScenes despite being a curated
	// Geometry-category entry, not an auto-generated "I<N>" one: its
	// build_lights lambda (scene_registry.h) returns pbrt_cpu::BuildResult's
	// own `lights` list, identical in shape and origin to every CustomScenes
	// entry's - see its own header comment on why it exists as a curated
	// entry "rather than an auto-generated I<N> Custom Scene" while still
	// loading pbrt_scenes/instanced-spheres.pbrt through the same pipeline.
	const bool isPbrtLoaded =
		std::string(s->category) == SceneCategories::CustomScenes || s->id == "F3";
	int cpuLights = isPbrtLoaded
		? count_cpu_emissive_lights_list(s->build_lights())
		: count_cpu_emissive_lights_list(s->build_world());
	int gpuLights = gpu_scene_light_count(s->id.c_str(), 100, 100);

	ASSERT_GE(gpuLights, 0) << s->name << " (id " << s->id << "): GPU scene build failed";

	if (isPbrtLoaded) {
		// Exact equality doesn't hold here for a real, deliberate reason:
		// pbrt_gpu_builder.h merges a pair of coplanar, same-emission
		// triangles that together form a parallelogram into ONE quad-
		// sampled light (see scene_builder.cpp's own "took the per-triangle
		// path rather than the cheaper merged-quad one" comment) - a real
		// GPU-side sampling-efficiency optimization pbrt_cpu_builder.h's
		// flat per-triangle `lights` list has no equivalent for (every
		// bundled pbrt_scenes/*.pbrt example's light is authored as exactly
		// this two-triangle-rectangle shape, which is why this was never
		// exercised before these scenes had the correct requires_files
		// value). Each merge accounts for EXACTLY two CPU triangles - never
		// more, never a triangle vanishing outright - so gpuLights is
		// bounded tightly: at most cpuLights (nothing merges away for
		// free), at least half of it rounded up (the most any merge pass
		// could combine). A real drift - a light present on one side and
		// silently missing on the other - still falls outside this range
		// and fails loudly.
		const int minPlausibleGpuLights = (cpuLights + 1) / 2;
		EXPECT_GE(gpuLights, minPlausibleGpuLights)
			<< s->name << " (id " << s->id << "): CPU scene has " << cpuLights
			<< " emissive light(s), GPU scene has only " << gpuLights
			<< " - more than a parallelogram-pair merge can explain.";
		EXPECT_LE(gpuLights, cpuLights)
			<< s->name << " (id " << s->id << "): GPU scene has " << gpuLights
			<< " emissive light(s), more than CPU's " << cpuLights
			<< " - a merge can only reduce the count, never grow it.";
	} else {
		EXPECT_EQ(cpuLights, gpuLights)
			<< s->name << " (id " << s->id << "): CPU scene has " << cpuLights
			<< " emissive light(s), GPU scene has " << gpuLights
			<< " - the two builders have drifted out of sync.";
	}
}

INSTANTIATE_TEST_SUITE_P(
	AllScenes, CpuGpuLightParityTest,
	::testing::Range(0, static_cast<int>(get_scene_registry().size())),
	[](const ::testing::TestParamInfo<int>& info) {
		const SceneDescriptor& desc = get_scene_registry()[info.param];
		std::string name = desc.name ? desc.name : "Unknown";
		std::string sanitized;
		for (char c : name) sanitized += std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
		return "Scene" + std::to_string(info.param) + "_" + sanitized;
	});
