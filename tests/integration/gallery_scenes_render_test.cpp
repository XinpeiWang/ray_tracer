/**
 * @file gallery_scenes_render_test.cpp
 * @brief Renders every scene the rest of the suite skips for "requires external assets", when those assets are on disk.
 *
 * The registry marks the model gallery (G/H), the downloaded pbrt collections (environment-*) and a few texture scenes with
 * SceneDescriptor::requires_files, and the parity and light-count suites skip them so a checkout without the data still passes.
 * Skipping them everywhere means ~90 scenes were never rendered by any test: a loader, builder or shader change that broke only
 * them (a mesh format, a large-BVH path, a texture type that only they use) went unnoticed. This suite renders each one that is
 * actually present on a tiny frame on the recursive GPU backend and checks the result is a real image: the render succeeds and
 * is not black. It is a smoke test with deliberately wide bounds, not a parity check.
 *
 * Cost, measured over the 83 gallery scenes: the GPU renders take ~170 s in total, dominated by loading the meshes, and 89 s of that
 * is the 12 "Large Scenes" (gigabyte OBJs: Power Plant, San Miguel, Rungholt, ...); the CPU renders take ~580 s for the same reason.
 * So by default the Large Scenes are skipped (RT_GALLERY_ALL=1 includes them), and the CPU render with its CPU-vs-GPU brightness
 * check is opt-in (RT_GALLERY_CPU=1). Several registry entries are twins of one pbrt file (a curated H entry and its auto-discovered
 * K entry): only the first is rendered. RT_SKIP_GALLERY=1 skips the whole suite.
 *
 * The test name contains "Gpu", so scripts/run_tests_parallel.ps1 puts it in the Slow tier with the other GPU tests.
 */
#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <set>
#include <string>

#include "../ppm_test_utils.h"
#include "scene_registry.h"

extern "C" {
	#include "cpu_interface.h"
	#include "optix_interface.h"
}

namespace {

constexpr int kWidth = 32;
constexpr int kHeight = 32;
constexpr int kDepth = 4;
constexpr int kCpuSpp = 8;
constexpr int kGpuSpp = 32;

// True for the second and later registry entries that load the same pbrt file as an earlier one (matched like scene_registry.h does,
// with std::filesystem::equivalent, since the same file can be reached through differently spelled paths).
bool isTwinOfAnEarlierScene(const std::string& id) {
	const auto& paths = pbrt_scene_registry::paths();
	const auto mine = paths.find(id);
	if (mine == paths.end()) return false;
	for (const SceneDescriptor& other : get_scene_registry()) {
		if (other.id == id) return false;   // reached this scene first: it is the original
		const auto theirs = paths.find(other.id);
		if (theirs == paths.end() || !other.requires_files) continue;
		std::error_code ec;
		if (std::filesystem::equivalent(theirs->second, mine->second, ec) && !ec) return true;
	}
	return false;
}

double secondsSince(std::chrono::steady_clock::time_point t) {
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

class GalleryGpuRenderTest : public ::testing::TestWithParam<int> {};

TEST_P(GalleryGpuRenderTest, RendersARealImage) {
	const SceneDescriptor& desc = get_scene_registry()[GetParam()];
	const SceneDescriptor* s = find_scene(desc.id);
	ASSERT_NE(s, nullptr) << "Missing scene id " << desc.id;
	if (!s->requires_files) GTEST_SKIP() << s->name << " is covered by the other suites";
	if (!s->gpu_compatible) GTEST_SKIP() << s->name << " is not GPU-compatible";
	if (!optix_is_available()) GTEST_SKIP() << "OptiX not available";
	if (std::getenv("RT_SKIP_GALLERY")) GTEST_SKIP() << "RT_SKIP_GALLERY is set";
	if (std::strcmp(s->category, SceneCategories::LargeScene) == 0 && !std::getenv("RT_GALLERY_ALL"))
		GTEST_SKIP() << s->name << " is a Large Scene (set RT_GALLERY_ALL=1 to render it)";
	if (isTwinOfAnEarlierScene(s->id)) GTEST_SKIP() << s->name << " loads the same file as an earlier entry";

	const bool withCpu = std::getenv("RT_GALLERY_CPU") != nullptr;
	const std::string base = "gallery_" + s->id;
	double cpuSec = 0.0;
	PPMImage cpu;
	if (withCpu) {
		const auto cpuStart = std::chrono::steady_clock::now();
		cpu_render_main(kWidth, kHeight, kCpuSpp, kDepth, (base + "_cpu.ppm").c_str(), s->id.c_str(),
		                s->camera.lookfrom_x, s->camera.lookfrom_y, s->camera.lookfrom_z);
		cpuSec = secondsSince(cpuStart);
		cpu = load_ppm((base + "_cpu.ppm").c_str());
		std::remove((base + "_cpu.ppm").c_str());
	}

#ifdef _WIN32
	_putenv_s("RAY_TRACER_WAVEFRONT", "0");
#else
	setenv("RAY_TRACER_WAVEFRONT", "0", 1);
#endif
	const auto gpuStart = std::chrono::steady_clock::now();
	const int rc = optix_render_main(kWidth, kHeight, kGpuSpp, kDepth, (base + "_gpu.ppm").c_str(), s->id.c_str(),
	                                 s->camera.lookfrom_x, s->camera.lookfrom_y, s->camera.lookfrom_z);
	const double gpuSec = secondsSince(gpuStart);
	const PPMImage gpu = rc == 0 ? load_ppm((base + "_gpu.ppm").c_str()) : PPMImage{};
	std::remove((base + "_gpu.ppm").c_str());

	const float cpuMean = cpu.valid ? average_brightness(cpu) : -1.0f;
	const float gpuMean = gpu.valid ? average_brightness(gpu) : -1.0f;
	std::printf("[gallery] %s %s: cpu %.3f (%.1fs) gpu %.3f (%.1fs)\n", s->id.c_str(), s->name, cpuMean, cpuSec, gpuMean, gpuSec);

	if (withCpu) ASSERT_TRUE(cpu.valid) << s->name << " (" << s->id << "): the CPU render produced no readable image";
	ASSERT_EQ(rc, 0) << s->name << " (" << s->id << "): the GPU render failed";
	ASSERT_TRUE(gpu.valid) << s->name << " (" << s->id << "): the GPU render produced no readable image";
	// Not exactly black, no more: a night scene at 32 spp and depth 4 (villa-lights-on) is legitimately this dim (0.0012), and a scene that
	// failed to load renders an all-zero image.
	EXPECT_GT(gpuMean, 0.0003f) << s->name << " (" << s->id << "): the GPU image is black";
	if (withCpu) EXPECT_GT(cpuMean, 0.0003f) << s->name << " (" << s->id << "): the CPU image is black";
	if (withCpu && cpuMean > 0.0003f && gpuMean > 0.0003f) {
		const float ratio = gpuMean / cpuMean;
		EXPECT_GT(ratio, 0.4f) << s->name << " (" << s->id << "): GPU is much darker than CPU";
		EXPECT_LT(ratio, 2.5f) << s->name << " (" << s->id << "): GPU is much brighter than CPU";
	}
}

INSTANTIATE_TEST_SUITE_P(
	AllScenes, GalleryGpuRenderTest,
	::testing::Range(0, static_cast<int>(get_scene_registry().size())),
	[](const ::testing::TestParamInfo<int>& info) {
		const SceneDescriptor& desc = get_scene_registry()[info.param];
		std::string name = desc.name ? desc.name : "Unknown";
		std::string sanitized;
		for (char c : name) sanitized += std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
		return "Scene" + std::to_string(info.param) + "_" + sanitized;
	});
