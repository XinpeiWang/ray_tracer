/**
 * @file pbrt_example_scenes_tests.cpp
 * @brief Render-smoke tests for the small, self-contained example .pbrt
 * scenes bundled in pbrt_scenes/ (kExampleSceneStems below - grows as new
 * pbrt-loader gaps get their own dedicated demo scene; add the new stem here
 * too whenever one is created, or it gets zero automated render coverage).
 *
 * These exist to demonstrate the Custom Scenes discovery feature and, until
 * now, had no automated coverage at all: CpuGpuLightParityTest
 * (cpu_gpu_comparison_tests.cpp) used to skip every pbrt-loaded scene
 * outright, because SceneDescriptor::requires_files was set unconditionally
 * true for anything pbrt_discover found - large downloaded collection or
 * small bundled file alike, with nothing distinguishing them. It now reads
 * pbrt_discover::Discovered::nested instead (see that field's comment), so a
 * scene sitting flat in pbrt_scenes/ - like these - reports requires_files
 * false and is no longer skipped there; this fixture remains useful as a
 * fast, targeted, always-on regression check independent of that broader
 * suite. BundledPbrtLightCoverageTest (pbrt_gpu_light_coverage_tests.cpp)
 * only checks that the GPU can sample every emissive shape - it never
 * actually renders a frame. Unlike the large pbrt-v4-scenes downloads
 * (gitignored, not always present - see pbrt_scenes/README.md), these
 * files are git-tracked and need no external assets beyond what ships in
 * the repo (killeroo-simple.pbrt's `Include` of
 * pbrt_scenes/geometry/killeroo.pbrt is itself bundled), so they are always
 * present and cheap to render - exactly what a fast regression fixture needs.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

extern "C" {
	#include "cpu_interface.h"
	#include "optix_interface.h"
}

#include "scene_registry.h"

namespace {

// Mirrors energy_conservation_tests.cpp's own load_render() - each test file
// in this suite that reads back a PPM keeps its own small copy rather than
// sharing one header, matching the existing convention.
struct RenderResult {
	int width = 0, height = 0;
	std::vector<float> pixels;  // RGB, normalized to [0,1]
	bool valid = false;
};

RenderResult load_render(const char* path) {
	RenderResult r;
	std::ifstream f(path);
	if (!f.good()) return r;

	std::string magic;
	int maxVal;
	f >> magic >> r.width >> r.height >> maxVal;
	if (magic != "P3" || maxVal <= 0) return r;

	int total = r.width * r.height * 3;
	r.pixels.resize(total);
	for (int i = 0; i < total; ++i) {
		int v; f >> v;
		r.pixels[i] = static_cast<float>(v) / static_cast<float>(maxVal);
	}
	r.valid = true;
	return r;
}

bool anyNonBlack(const RenderResult& r) {
	for (float p : r.pixels) if (p > 1e-4f) return true;
	return false;
}

bool allFinite(const RenderResult& r) {
	for (float p : r.pixels) if (!std::isfinite(p)) return false;
	return true;
}

// The self-contained example scenes, named by file stem (not id - see
// find_example_scene()'s comment on why a stable id can't be hardcoded).
// The last 4 were added specifically to close feature-coverage gaps the
// first 5 left open (see docs/PBRT_SUPPORT.md's capability matrix): those 5
// only ever exercised 4 of 11 material kinds, 1 of 9 light kinds, and 1 of 6
// camera kinds between them - no punctual light, alternate camera, or "mix"
// material (the documented largest CPU/GPU gap) had render-level coverage
// from this folder at all.
constexpr const char* kExampleSceneStems[] = {
	"flush-ceiling-light",
	"emissive-octahedron-furnace",
	"example-cornell",
	"instanced-spheres",
	"killeroo-simple",
	"triangle-fan-light",
	"two-sphere-lights",
	"mix-material",
	"punctual-lights",
	"depth-of-field",
	"orthographic-camera",
	"layered-materials",
	"infinite-light",
	"spherical-camera",
	"named-material-and-texture",
	"plymesh-geometry",
	"realistic-camera",
	"textured-twosided-lights",
	"plymesh-uv",
	"goniometric-projection",
	"coateddiffuse-texture",
	"diffusetransmission-texture",
	"conductor-rgb-eta-k",
	"nested-checker-texture",
	"blackbody-light",
	"colorspace-blackbody",
};

// Resolved by NAME at test-run time rather than a hardcoded id string:
// pbrt-loaded scenes are numbered by discovery order across every .pbrt
// pbrt_discover finds (see scene_registry.h's SceneDescriptor::id comment
// and pbrt_scenes/README.md's "Ids" section), so "example-cornell.pbrt" is
// not reliably "I1" - a machine with additional downloaded pbrt-v4-scenes
// collections sitting in pbrt_scenes/ (this repo's own dev machines
// sometimes do - see that README) sorts differently and shifts every id
// after whatever sorts earlier. Matching by SceneDescriptor::name (the
// file's stem) is stable regardless of what else happens to be discovered
// alongside these 5.
const SceneDescriptor* find_example_scene(const char* stem) {
	for (const auto& s : get_scene_registry()) {
		if (s.category == SceneCategories::CustomScenes && s.name && std::string(s.name) == stem)
			return &s;
	}
	return nullptr;
}

} // namespace

class PbrtExampleSceneTest : public ::testing::TestWithParam<const char*> {
protected:
	void TearDown() override {
		for (const auto& f : files_) std::remove(f.c_str());
	}
	std::vector<std::string> files_;
};

// force_camera_override is deliberately left at its default (0): these
// files declare their own Camera/LookAt, and the point of this fixture is
// to exercise that declared camera the same way a real user selecting the
// scene by id would - not to override it with an arbitrary position. The
// cam_x/y/z arguments are therefore ignored by the renderer for these
// scenes and their exact values don't matter.
TEST_P(PbrtExampleSceneTest, CpuRendersWithoutCrashingOrGoingBlack) {
	const char* stem = GetParam();
	const SceneDescriptor* s = find_example_scene(stem);
	if (!s) GTEST_SKIP() << stem << ".pbrt was not discovered - is pbrt_scenes/ present?";

	const std::string out = std::string("pbrt_example_cpu_") + stem + ".ppm";
	files_.push_back(out);
	const int rc = cpu_render_main(64, 64, 16, 5, out.c_str(), s->id.c_str(), 0.0, 0.0, 0.0);
	ASSERT_EQ(rc, 0) << stem << ": CPU render failed (scene id " << s->id << ")";

	const RenderResult r = load_render(out.c_str());
	ASSERT_TRUE(r.valid) << stem << ": CPU render produced no readable PPM";
	EXPECT_TRUE(allFinite(r)) << stem << ": CPU render produced NaN/Inf pixels";
	EXPECT_TRUE(anyNonBlack(r)) << stem << ": CPU render is entirely black";
}

TEST_P(PbrtExampleSceneTest, GpuRendersWithoutCrashingOrGoingBlack) {
	if (!optix_is_available()) GTEST_SKIP() << "OptiX not available";
	const char* stem = GetParam();
	const SceneDescriptor* s = find_example_scene(stem);
	if (!s) GTEST_SKIP() << stem << ".pbrt was not discovered - is pbrt_scenes/ present?";
	if (!s->gpu_compatible) GTEST_SKIP() << stem << " is not GPU-compatible";

	const std::string out = std::string("pbrt_example_gpu_") + stem + ".ppm";
	files_.push_back(out);
	const int rc = optix_render_main(64, 64, 16, 5, out.c_str(), s->id.c_str(), 0.0, 0.0, 0.0);
	ASSERT_EQ(rc, 0) << stem << ": GPU render failed (scene id " << s->id << ")";

	const RenderResult r = load_render(out.c_str());
	ASSERT_TRUE(r.valid) << stem << ": GPU render produced no readable PPM";
	EXPECT_TRUE(allFinite(r)) << stem << ": GPU render produced NaN/Inf pixels";
	EXPECT_TRUE(anyNonBlack(r)) << stem << ": GPU render is entirely black";
}

INSTANTIATE_TEST_SUITE_P(
	Bundled, PbrtExampleSceneTest,
	::testing::ValuesIn(kExampleSceneStems),
	[](const ::testing::TestParamInfo<const char*>& info) {
		std::string sanitized;
		for (const char* p = info.param; *p; ++p)
			sanitized += std::isalnum(static_cast<unsigned char>(*p)) ? *p : '_';
		return sanitized;
	});

// CPU vs GPU-recursive vs GPU-wavefront mean brightness of one bundled scene; the CPU is the reference.
static void expectBackendsAgree(const char* stem, int spp, double lo, double hi) {
	if (!optix_is_available()) GTEST_SKIP() << "OptiX not available";
	const SceneDescriptor* s = find_example_scene(stem);
	if (!s) GTEST_SKIP() << stem << ".pbrt was not discovered - is pbrt_scenes/ present?";

	const auto mean = [](const RenderResult& r) {
		double sum = 0.0;
		for (float p : r.pixels) sum += p;
		return r.pixels.empty() ? 0.0 : sum / static_cast<double>(r.pixels.size());
	};
	const auto setWavefront = [](const char* v) {
#ifdef _WIN32
		_putenv_s("RAY_TRACER_WAVEFRONT", v);
#else
		setenv("RAY_TRACER_WAVEFRONT", v, 1);
#endif
	};
	const auto renderGpu = [&](const char* wavefront, const std::string& out) {
		setWavefront(wavefront);
		const int rc = optix_render_main(64, 64, spp, 8, out.c_str(), s->id.c_str(), 0.0, 0.0, 0.0);
		EXPECT_EQ(rc, 0);
		RenderResult r = load_render(out.c_str());
		std::remove(out.c_str());
		return r;
	};

	const std::string base = std::string("pbrt_agree_") + stem;
	ASSERT_EQ(cpu_render_main(64, 64, spp, 8, (base + "_cpu.ppm").c_str(), s->id.c_str(), 0.0, 0.0, 0.0), 0);
	const RenderResult cpu = load_render((base + "_cpu.ppm").c_str());
	std::remove((base + "_cpu.ppm").c_str());
	const RenderResult recursive = renderGpu("0", base + "_rec.ppm");
	const RenderResult wavefront = renderGpu("1", base + "_wf.ppm");
	setWavefront("0");
	ASSERT_TRUE(cpu.valid && recursive.valid && wavefront.valid);
	const double c = mean(cpu);
	ASSERT_GT(c, 0.05) << stem << ": the CPU reference came out black";
	EXPECT_GT(mean(recursive), lo * c) << stem << ": GPU-recursive too dark vs CPU";
	EXPECT_LT(mean(recursive), hi * c) << stem << ": GPU-recursive too bright vs CPU";
	EXPECT_GT(mean(wavefront), lo * c) << stem << ": GPU-wavefront too dark vs CPU";
	EXPECT_LT(mean(wavefront), hi * c) << stem << ": GPU-wavefront too bright vs CPU";
}

// A finite light sitting closer to the surface behind it than the GPU's 0.01 shadow-ray nudge. The GPU
// used to measure a shadow ray's length from the un-nudged shading point, overshooting the light by the nudge
// and ending inside the ceiling 0.005 behind it, so the sample counted as blocked: recursive rendered this
// room at 22% of the CPU's brightness and wavefront at 37%. See pbrt_scenes/flush-ceiling-light.pbrt.
TEST(PbrtBackendAgreementTest, FlushCeilingLightAgreesAcrossBackends) {
	expectBackendsAgree("flush-ceiling-light", 64, 0.85, 1.15);
}

// A multi-faced emitter must occlude its own far side: the GPU used to skip emissive surfaces in shadow
// any-hits, so a ray aimed at the back of this octahedron passed through its front and every face counted
// as visible - both GPU backends rendered it 48% too bright. The CPU matches the closed-form answer.
TEST(PbrtBackendAgreementTest, EmissiveOctahedronFurnaceAgreesAcrossBackends) {
	expectBackendsAgree("emissive-octahedron-furnace", 128, 0.88, 1.12);
}

// A BSDF-sampled bounce that lands on a lamp has to be counted, MIS-weighted against the NEE sample taken at the
// previous vertex. The wavefront backend dropped it for every non-specular bounce while still weighting its NEE
// sample, so with a big lamp close to bright walls it lost a fifth of the indirect light (83% of CPU on Fireplace
// Room). Tight bounds: the three backends agree to about 1% here.
TEST(PbrtBackendAgreementTest, LargeAreaLightRoomAgreesAcrossBackends) {
	expectBackendsAgree("large-area-light-room", 64, 0.96, 1.04);
}
