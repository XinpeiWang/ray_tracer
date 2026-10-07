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
#include "../../src/external/tinyexr.h"
#include <algorithm>
#include <array>
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
#include "agreement_test_helpers.h"

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
//
// Compared as LINEAR float means (each backend writes the pre-tone-map radiance when the output path ends in .exr), not as the
// mean of the 8-bit tone-mapped picture: the ACES curve and sRGB encoding compress a real radiance error to a fraction of its size
// and saturate bright pixels, so a +5% error in the glass of a scene read as +1% (or nothing) there. The ratios are always printed
// ("[agree] stem: recursive 100.2%  wavefront 99.8%") so the bounds can be set from what the backends really do: most scenes sit
// within 0.5% of the CPU and are bounded at +-1.5%; the wider ones are measured: rgbgrid-medium (now within 0.1%) and camera-medium-absorbing (wavefront ~2.5% above recursive, and the CPU lands on one of the two between
// runs: 1.572, 1.572, 1.613 - a heavy-tailed point light in fog, noise rather than bias). bump-mapped-plane and normal-mapped-cornell were
// 98.4-99.0% and 100.9% until the CPU stopped requantizing the decoded height image to bytes and both backends began reading normal
// maps linear, as pbrt does.
static void expectBackendsAgree(const char* stem, int spp, double lo, double hi, int cpuRepeats = 1) {
	if (!optix_is_available()) GTEST_SKIP() << "OptiX not available";
	const SceneDescriptor* s = find_example_scene(stem);
	if (!s) GTEST_SKIP() << stem << ".pbrt was not discovered - is pbrt_scenes/ present?";

	const auto setWavefront = [](const char* v) {
#ifdef _WIN32
		_putenv_s("RAY_TRACER_WAVEFRONT", v);
#else
		setenv("RAY_TRACER_WAVEFRONT", v, 1);
#endif
	};
	const auto renderGpu = [&](const char* wavefront, const std::string& out, double& mean) {
		setWavefront(wavefront);
		const int rc = optix_render_main(64, 64, spp, 8, out.c_str(), s->id.c_str(), 0.0, 0.0, 0.0);
		EXPECT_EQ(rc, 0);
		const bool ok = loadLinearMean(out, mean);
		std::remove(out.c_str());
		return ok;
	};

	const std::string base = std::string("pbrt_agree_") + stem;
	double c = 0.0, rec = 0.0, wf = 0.0;
	// The CPU reference is the median of `cpuRepeats` renders: a scene whose CPU estimate has a heavy tail (a point light in fog) lands
	// on a different mean every run, and one draw is not a reference.
	std::vector<double> cpuMeans;
	bool cpuOk = true;
	for (int i = 0; i < cpuRepeats && cpuOk; ++i) {
		ASSERT_EQ(cpu_render_main(64, 64, spp, 8, (base + "_cpu.exr").c_str(), s->id.c_str(), 0.0, 0.0, 0.0), 0);
		double m = 0.0;
		cpuOk = loadLinearMean(base + "_cpu.exr", m);
		cpuMeans.push_back(m);
		std::remove((base + "_cpu.exr").c_str());
	}
	if (cpuOk) {
		std::sort(cpuMeans.begin(), cpuMeans.end());
		c = cpuMeans[cpuMeans.size() / 2];
	}
	const bool recOk = renderGpu("0", base + "_rec.exr", rec);
	const bool wfOk = renderGpu("1", base + "_wf.exr", wf);
	setWavefront("0");
	ASSERT_TRUE(cpuOk && recOk && wfOk) << stem << ": a render produced no readable EXR";
	ASSERT_GT(c, 1e-3) << stem << ": the CPU reference came out black";
	std::printf("[agree] %s: recursive %.1f%%  wavefront %.1f%%  (CPU linear mean %.4f, bounds %.1f%%..%.1f%%)\n",
	            stem, 100.0 * rec / c, 100.0 * wf / c, c, 100.0 * lo, 100.0 * hi);
	EXPECT_GT(rec, lo * c) << stem << ": GPU-recursive too dark vs CPU";
	EXPECT_LT(rec, hi * c) << stem << ": GPU-recursive too bright vs CPU";
	EXPECT_GT(wf, lo * c) << stem << ": GPU-wavefront too dark vs CPU";
	EXPECT_LT(wf, hi * c) << stem << ": GPU-wavefront too bright vs CPU";
}

// Per-channel linear means of an EXR (non-finite samples ignored): what a chromatic scene is judged on, since the mean over the three
// channels hides a medium that is thicker in one of them.
static bool loadLinearChannelMeans(const std::string& path, double mean[3]) {
	float* rgba = nullptr;
	int w = 0, h = 0;
	const char* err = nullptr;
	if (LoadEXR(&rgba, &w, &h, path.c_str(), &err) != TINYEXR_SUCCESS) {
		if (err) FreeEXRErrorMessage(err);
		return false;
	}
	double sum[3] = {0.0, 0.0, 0.0};
	for (int i = 0; i < w * h; ++i)
		for (int c = 0; c < 3; ++c) {
			const float v = rgba[4 * i + c];
			if (std::isfinite(v)) sum[c] += v;
		}
	free(rgba);
	for (int c = 0; c < 3; ++c) mean[c] = (w > 0 && h > 0) ? sum[c] / (static_cast<double>(w) * h) : 0.0;
	return true;
}

// One bundled scene on all three backends, each channel's linear mean against a known answer (a closed form, or a furnace's 1.0):
// the CPU and recursive backends sample in RGB and must be within `tol` of it, the spectral wavefront backend within `tolWavefront`
// (its uplifted sigma(lambda) is not exactly the per-channel exp(-sigma t) - see pbrt_scenes/chromatic-absorber.pbrt).
static void expectChannelMeans(const char* stem, int spp, int depth, const double* expectedIn, double tol, double tolWavefront, int cpuRepeats = 1) {
	if (!optix_is_available()) GTEST_SKIP() << "OptiX not available";
	const SceneDescriptor* s = find_example_scene(stem);
	if (!s) GTEST_SKIP() << stem << ".pbrt was not discovered - is pbrt_scenes/ present?";

	const auto setWavefront = [](const char* v) {
#ifdef _WIN32
		_putenv_s("RAY_TRACER_WAVEFRONT", v);
#else
		setenv("RAY_TRACER_WAVEFRONT", v, 1);
#endif
	};
	const std::string base = std::string("pbrt_chroma_") + stem;
	double cpu[3], rec[3], wf[3];
	// The CPU reference is the per-channel median of `cpuRepeats` renders: a scene whose CPU estimate has a heavy tail (one very bright path
	// sample in a thousand) otherwise fails now and then on a single outlier while the GPU backends, which converge, stay put.
	bool cpuOk = true;
	std::vector<std::array<double, 3>> cpuRuns;
	for (int rep = 0; rep < cpuRepeats && cpuOk; ++rep) {
		ASSERT_EQ(cpu_render_main(64, 64, spp, depth, (base + "_cpu.exr").c_str(), s->id.c_str(), 0.0, 0.0, 0.0), 0);
		double one[3];
		cpuOk = loadLinearChannelMeans(base + "_cpu.exr", one);
		std::remove((base + "_cpu.exr").c_str());
		cpuRuns.push_back({one[0], one[1], one[2]});
	}
	if (cpuOk) {
		for (int c = 0; c < 3; ++c) {
			std::vector<double> v;
			for (const auto& r : cpuRuns) v.push_back(r[c]);
			std::sort(v.begin(), v.end());
			cpu[c] = v[v.size() / 2];
		}
	}
	bool gpuOk[2];
	double* gpuMeans[2] = {rec, wf};
	const char* gpuMode[2] = {"0", "1"};
	const char* gpuTag[2] = {"_rec.exr", "_wf.exr"};
	for (int g = 0; g < 2; ++g) {
		setWavefront(gpuMode[g]);
		EXPECT_EQ(optix_render_main(64, 64, spp, depth, (base + gpuTag[g]).c_str(), s->id.c_str(), 0.0, 0.0, 0.0), 0);
		gpuOk[g] = loadLinearChannelMeans(base + gpuTag[g], gpuMeans[g]);
		std::remove((base + gpuTag[g]).c_str());
	}
	setWavefront("0");
	ASSERT_TRUE(cpuOk && gpuOk[0] && gpuOk[1]) << stem << ": a render produced no readable EXR";
	// No closed form: the CPU (RGB, per-channel) is the reference and only the GPU backends are judged against it.
	double expectedFromCpu[3] = {cpu[0], cpu[1], cpu[2]};
	const double* expected = expectedIn ? expectedIn : expectedFromCpu;
	std::printf("[chroma] %s: expected %.4f %.4f %.4f | cpu %.4f %.4f %.4f | recursive %.4f %.4f %.4f | wavefront %.4f %.4f %.4f\n", stem,
	            expected[0], expected[1], expected[2], cpu[0], cpu[1], cpu[2], rec[0], rec[1], rec[2], wf[0], wf[1], wf[2]);
	static const char* const kChannel[3] = {"R", "G", "B"};
	for (int c = 0; c < 3; ++c) {
		if (expectedIn) EXPECT_NEAR(cpu[c], expected[c], tol * expected[c]) << stem << " CPU " << kChannel[c];
		EXPECT_NEAR(rec[c], expected[c], tol * expected[c]) << stem << " GPU-recursive " << kChannel[c];
		EXPECT_NEAR(wf[c], expected[c], tolWavefront * expected[c]) << stem << " GPU-wavefront " << kChannel[c];
	}
}

// A medium thicker in blue than in red used to render GREY on every backend: the models took one extinction (the luminance of
// sigma_t) for all three channels and only tinted the albedo. sigma_a = (0.1, 0.4, 0.9) over a chord of 2 is exp(-sigma * 2)
// = (0.819, 0.449, 0.165); the scalar model read 0.47 in each. See pbrt_scenes/chromatic-absorber.pbrt.
TEST(PbrtBackendAgreementTest, ChromaticAbsorberFollowsBeerLambertPerChannel) {
	const double chord = 1.9945;   // mean over the 2-degree frame
	const double expected[3] = {std::exp(-0.1 * chord), std::exp(-0.4 * chord), std::exp(-0.9 * chord)};
	// Measured: CPU and recursive within 0.1% of the closed form; wavefront 0.780/0.446/0.153 (-4.8%, -0.9%, -7.8%) - spectral rendering
	// of a strongly chromatic sigma, whose uplift is not the per-channel exp (the sRGB matrix has negative lobes), as in pbrt.
	expectChannelMeans("chromatic-absorber", 128, 8, expected, 0.01, 0.09);
}

// The same absorber as the camera medium (pbrt's unbounded fog around the camera, MediumInterface before Camera): the pass-through weight
// of the camera medium's free-flight sample. The GPU backends used to apply the luminance extinction to every channel here.
TEST(PbrtBackendAgreementTest, ChromaticCameraMediumAbsorberFollowsBeerLambertPerChannel) {
	const double expected[3] = {std::exp(-0.1 * 2.0), std::exp(-0.4 * 2.0), std::exp(-0.9 * 2.0)};
	expectChannelMeans("chromatic-camera-medium-absorber", 128, 8, expected, 0.01, 0.09);
}

// A lit room under a camera medium with a different extinction per colour: collisions, pass-through, area and point light shadow rays
// all carry per-channel weights. No closed form; the CPU is the reference, per channel. Measured: recursive +0.5/+1.2/+2.2% (R/G/B), wavefront
// +1.5/+0.2/+0.8%.
TEST(PbrtBackendAgreementTest, ChromaticCameraMediumRoomAgreesPerChannel) {
	// The CPU reference is the median of seven renders: one render's blue channel ranged 0.0785-0.0821 across runs and once read 0.469 (a
	// single very bright path sample), which failed the 4% bound now and then, while the GPU backends are steady.
	expectChannelMeans("chromatic-camera-medium", 256, 8, nullptr, 0.04, 0.04, /*cpuRepeats=*/7);
}

// A heterogeneous RGB grid (MakeNamedMedium "rgbgrid") that only absorbs. Both GPU backends ignored sigma_a in a grid entirely (the absorber
// rendered invisible) and the CPU read one grey value. The grid lookup is zero beyond the outermost voxel centres (pbrt's SampledGrid), so
// the mean density over the 2-unit chord is 0.875 and each channel is exp(-sigma_a * 1.9945 * 0.875). Wavefront is spectral (a basis spread
// of the RGB coefficients, wf_rgb_wavelength_basis), hence the wider bound. See pbrt_scenes/chromatic-rgbgrid-absorber.pbrt.
TEST(PbrtBackendAgreementTest, ChromaticRgbGridAbsorberFollowsBeerLambertPerChannel) {
	const double depth = 1.9945 * 0.875;
	const double expected[3] = {std::exp(-0.1 * depth), std::exp(-0.4 * depth), std::exp(-0.9 * depth)};
	expectChannelMeans("chromatic-rgbgrid-absorber", 128, 8, expected, 0.03, 0.07);
}

// A diffuse sphere (rho 0.5) under a uniform white sky radiates exactly 0.5 everywhere, at every maximum depth from 1 up: one bounce, then the sky.
// The path tracers used to stop at the last scattering vertex without tracing its continuation ray, so a depth-limited render carried only the MIS
// share light sampling got there and never the BSDF-sampled share (0.09 at depth 1; a Cornell box 1% / 3% / 10% dark at depth 8 / 4 / 2). pbrt-v4
// adds the emission or sky the last continuation ray sees before it tests the depth. Depth 1 is the strictest case. See path-depth-furnace.pbrt.
TEST(PbrtBackendAgreementTest, DiffuseSphereUnderSkyIsExactAtDepthOneOnEveryBackend) {
	const double expected[3] = {0.5, 0.5, 0.5};
	expectChannelMeans("path-depth-furnace", 256, 1, expected, 0.01, 0.03);
}

// A grid that scatters without absorbing, differently per colour, is invisible under a uniform sky in every channel. The earlier model
// (extinction of the brightest channel, albedo sigma_s / max) made the lesser channels absorb: red 0.60, green 0.77 on every backend.
TEST(PbrtBackendAgreementTest, ChromaticRgbGridFurnaceStaysInvisibleInEveryChannel) {
	const double expected[3] = {1.0, 1.0, 1.0};
	expectChannelMeans("chromatic-rgbgrid-furnace", 128, 24, expected, 0.015, 0.015);
}

// A fog whose scattering differs by colour but absorbs nothing is invisible under a uniform sky in every channel. A collision weight
// that does not average to the transmittance (the balance heuristic across channels, volume_scattering.h) shows up here as a cast.
// A measured-BSDF sphere under a uniform white sky: each pixel's radiance is the table's directional albedo for that view angle. Two
// defects made every backend read it wrong, identically (so a CPU-vs-GPU comparison could not see either): the material returned the bare f
// instead of pbrt's path weight f * |cos| / pdf (a blue table read 7.8/8.1/3.5, a metallic one 12, white paper 0.34), and sample_f
// evaluated the spectra at u_wm instead of the luminance-warped point (so its f disagreed with f() by up to 26%).
//
// The reference is independent of all the sampling code: the albedo at each view angle integrated from f() alone (cosine-weighted
// directions), averaged over the pixels the camera sees (LookAt 0 0 6, fov 14, a unit sphere: the view angle runs from 0 to ~55 degrees,
// and the few corner pixels past the silhouette see the sky, 1.0).
TEST(PbrtBackendAgreementTest, MeasuredFurnaceReadsTheTablesOwnAlbedoOnEveryBackend) {
	std::string loadError;
	const auto table = measured_bxdf_io::GetMeasuredBRDFDataCached("pbrt_scenes/synthetic-gold.bsdf", loadError);
	ASSERT_TRUE(table) << "synthetic-gold.bsdf did not load: " << loadError;
	ASSERT_TRUE(table->isotropic) << "the reference ignores the azimuth, so it needs an isotropic table";
	MeasuredBxDF<double> bxdf(table.get(), 612.0f, 549.0f, 465.0f);

	const double kPi = 3.141592653589793;
	const int kAngles = 91, kSamples = 20000;
	std::vector<std::array<double, 3>> albedoAt(kAngles);
	for (int a = 0; a < kAngles; ++a) {
		const double theta = std::min(a, 89) * kPi / 180.0, wox = std::sin(theta), woz = std::cos(theta);
		double sum[3] = {0.0, 0.0, 0.0};
		for (int i = 0; i < kSamples; ++i) {
			const double u0 = (i + 0.5) / kSamples, u1 = std::fmod((i + 0.5) * 0.6180339887498949, 1.0);   // a low-discrepancy pair
			const double r = std::sqrt(u0), phi = 2.0 * kPi * u1;
			double fr, fg, fb;
			bxdf.f(wox, 0.0, woz, r * std::cos(phi), r * std::sin(phi), std::sqrt(std::max(0.0, 1.0 - u0)), fr, fg, fb);
			sum[0] += fr; sum[1] += fg; sum[2] += fb;
		}
		for (int c = 0; c < 3; ++c) albedoAt[a][c] = kPi * sum[c] / kSamples;
	}
	double expected[3] = {0.0, 0.0, 0.0};
	const int res = 64;   // the resolution expectChannelMeans renders at
	const double tanHalfFov = std::tan(7.0 * kPi / 180.0), distance = 6.0;
	for (int py = 0; py < res; ++py) {
		for (int px = 0; px < res; ++px) {
			const double x = (2.0 * (px + 0.5) / res - 1.0) * tanHalfFov, y = (2.0 * (py + 0.5) / res - 1.0) * tanHalfFov;
			const double len = std::sqrt(x * x + y * y + 1.0), dx = x / len, dy = y / len, dz = -1.0 / len;
			// ray from (0, 0, distance) against the unit sphere at the origin
			const double b = distance * dz, disc = b * b - (distance * distance - 1.0);
			if (disc < 0.0) { for (int c = 0; c < 3; ++c) expected[c] += 1.0; continue; }
			const double t = -b - std::sqrt(disc);
			const double hx = t * dx, hy = t * dy, hz = distance + t * dz;   // the hit point, which is also the unit normal
			const double cosTheta = std::max(0.0, -(hx * dx + hy * dy + hz * dz));
			const double deg = std::acos(std::min(1.0, cosTheta)) * 180.0 / kPi;
			const int lo = std::min(static_cast<int>(deg), kAngles - 2);
			const double w = std::min(1.0, deg - lo);
			for (int c = 0; c < 3; ++c) expected[c] += albedoAt[lo][c] * (1.0 - w) + albedoAt[lo + 1][c] * w;
		}
	}
	for (int c = 0; c < 3; ++c) expected[c] /= res * res;
	std::printf("[measured-furnace] reference albedo %.4f %.4f %.4f\n", expected[0], expected[1], expected[2]);
	expectChannelMeans("measured-furnace", 256, 4, expected, 0.03, 0.03);
}

// The measured material takes direct-light samples with MIS (CPU, recursive and wavefront), instead of sampling its BSDF alone. This scene has
// a small area light and a point light: the point light is a delta light that BSDF sampling can never hit, so the old estimator never lit the
// material from it at all. The three backends must agree (each is a separate implementation of the BxDF's f() and pdf()).
TEST(PbrtBackendAgreementTest, MeasuredLightsAgreeAcrossBackends) {
	expectBackendsAgree("measured-lights", 256, 0.985, 1.015);
}

// The same scene with the point light removed: the area light alone, whose mean the old BSDF-sampling-only estimator could get right. It
// converged to a luminance of 0.0608 (CPU 0.0606, recursive 0.0610, wavefront 0.0609 at 4096 samples, ~75% per-pixel noise); light sampling
// must read the same mean with ~1.5% noise, so this checks the new estimator is unbiased against an independent one. The expected RGB is the
// light-sampled estimator's, whose luminance (0.2126 R + 0.7152 G + 0.0722 B) is that 0.0608.
TEST(PbrtBackendAgreementTest, MeasuredAreaLightKeepsTheBsdfSamplingMean) {
	const double expected[3] = {0.0685, 0.0608, 0.0384};
	expectChannelMeans("measured-lights-area", 512, 6, expected, 0.02, 0.02);
}

// A closed diffuse-transmission shell under a uniform sky (pbrt_scenes/diffuse-transmission-furnace.pbrt, R = 0.2, T = 0.6) has a closed form at every depth:
// 0.2, 0.56, 0.632, 0.6464, ... -> 0.65. Three defects hid in that one material and each integrator read something different (CPU 0.39, recursive GPU 0.25,
// --simplepath 1.40 at depth 30): the CPU weight carried the lobe probability pr / (pr + pt) twice (scattering_pdf() and the committed lobe), both GPU backends
// left it out altogether (weight R instead of R / p_lobe), and shadow rays passed through the shell, so light sampling saw the sky through it while BSDF sampling
// could not - MIS then mixed two estimators of different integrals (depth 1 read 0.35 on the CPU, whose closed form is 0.2).
TEST(PbrtBackendAgreementTest, DiffuseTransmissionShellHasItsClosedFormOnEveryBackend) {
	const double expected[3] = {0.6464, 0.6464, 0.6464};   // depth 4
	expectChannelMeans("diffuse-transmission-furnace", 256, 4, expected, 0.01, 0.02);
}

TEST(PbrtBackendAgreementTest, DiffuseTransmissionShellReadsOnlyItsReflectionAtDepthOne) {
	const double expected[3] = {0.2, 0.2, 0.2};
	expectChannelMeans("diffuse-transmission-furnace", 256, 1, expected, 0.01, 0.02);
}

// GPU SPPM against the CPU: it read 0.33x (Earth globe, spherical-camera panorama) to 0.47x (Perlin spheres) of the path tracer because it never sampled a
// material's texture and had no sky at all (an escaping ray and a visible point lit by a uniform sky contributed nothing), and its camera-pass seed ignored the
// SPPM iteration, so every iteration drew the same light sample and film position. A constant sky + one convex diffuse object has no interreflection, so the
// closed form of the furnace is exact; the textured sphere and the two bundled scenes are compared with the CPU path tracer / CPU SPPM.
static bool gpuSppmMeans(const std::string& sceneId, int size, int iterations, int photons, int depth, double mean[3]) {
	const std::string out = "pbrt_gpu_sppm_" + sceneId + ".exr";
	if (optix_render_main_sppm(size, size, iterations, photons, depth, out.c_str(), sceneId.c_str(), 0.0, 0.0, 0.0) != 0) return false;
	const bool ok = loadLinearChannelMeans(out, mean);
	std::remove(out.c_str());
	return ok;
}

TEST(PbrtBackendAgreementTest, GpuSppmUniformSkyFurnaceIsExact) {
	if (!optix_is_available()) GTEST_SKIP() << "OptiX not available";
	const SceneDescriptor* s = find_example_scene("path-depth-furnace");
	if (!s) GTEST_SKIP() << "path-depth-furnace.pbrt was not discovered - is pbrt_scenes/ present?";
	double m[3];
	ASSERT_TRUE(gpuSppmMeans(s->id, 32, 100, 20000, 4, m));
	std::printf("[gpu sppm] sky furnace: %.4f %.4f %.4f (0.5 expected)\n", m[0], m[1], m[2]);
	for (int c = 0; c < 3; ++c) EXPECT_NEAR(m[c], 0.5, 0.015) << "channel " << c;
}

TEST(PbrtBackendAgreementTest, GpuSppmSamplesDiffuseTexturesLikeThePathTracer) {
	if (!optix_is_available()) GTEST_SKIP() << "OptiX not available";
	const SceneDescriptor* s = find_example_scene("sppm-textured-sky");
	if (!s) GTEST_SKIP() << "sppm-textured-sky.pbrt was not discovered - is pbrt_scenes/ present?";
	ASSERT_EQ(cpu_render_main(32, 32, 256, 4, "pbrt_gpu_sppm_tex_pt.exr", s->id.c_str(), 0.0, 0.0, 0.0), 0);
	double pt[3], sppm[3];
	ASSERT_TRUE(loadLinearChannelMeans("pbrt_gpu_sppm_tex_pt.exr", pt));
	std::remove("pbrt_gpu_sppm_tex_pt.exr");
	ASSERT_TRUE(gpuSppmMeans(s->id, 32, 100, 20000, 4, sppm));
	for (int c = 0; c < 3; ++c) {
		std::printf("[gpu sppm] textured sphere ch%d: sppm %.4f path %.4f (%.1f%%)\n", c, sppm[c], pt[c], 100.0 * sppm[c] / pt[c]);
		EXPECT_NEAR(sppm[c], pt[c], 0.03 * pt[c]) << "channel " << c;
	}
}

TEST(PbrtBackendAgreementTest, GpuSppmAgreesWithCpuSppmOnSkyLitAndSphericalCameraScenes) {
	if (!optix_is_available()) GTEST_SKIP() << "OptiX not available";
	// A7: Perlin spheres with emissive lights; D3: a 360-degree spherical camera under a uniform sky. Both read ~0.5x on the GPU before.
	for (const char* id : {"A7", "D3"}) {
		ASSERT_EQ(cpu_render_main_sppm(48, 48, 100, 20000, 4, "pbrt_gpu_sppm_cpu.exr", id, 0.0, 0.0, 0.0), 0) << id;
		double cpu[3], gpu[3];
		ASSERT_TRUE(loadLinearChannelMeans("pbrt_gpu_sppm_cpu.exr", cpu)) << id;
		std::remove("pbrt_gpu_sppm_cpu.exr");
		ASSERT_TRUE(gpuSppmMeans(id, 48, 100, 20000, 4, gpu)) << id;
		for (int c = 0; c < 3; ++c) {
			std::printf("[gpu sppm] %s ch%d: gpu %.4f cpu %.4f (%.1f%%)\n", id, c, gpu[c], cpu[c], 100.0 * gpu[c] / cpu[c]);
			EXPECT_NEAR(gpu[c], cpu[c], 0.06 * cpu[c]) << id << " channel " << c;
		}
	}
}

// A diffuse-transmission plate (R = 0.3, T = 0.5) lit by one point light has a closed form at the centre (pbrt_scenes/diffuse-transmission-point-light*.pbrt):
// E = I / d^2 = 10, so 0.5 / pi * 10 = 1.5915 seen through the plate from a light behind it and 0.3 / pi * 10 = 0.9549 seen from the lit side. Both GPU backends
// sampled no light at a diffuse-transmission vertex (a BSDF-only estimator), which cannot find a point light at all: the plate rendered black, in both cases.
// They now sample lights on either side of the surface, with MIS, as the CPU always did.
TEST(PbrtBackendAgreementTest, DiffuseTransmissionPlateUnderAPointLightBehindIt) {
	const double expected[3] = {1.5915, 1.5915, 1.5915};
	expectChannelMeans("diffuse-transmission-point-light", 64, 4, expected, 0.01, 0.02);
}

TEST(PbrtBackendAgreementTest, DiffuseTransmissionPlateUnderAPointLightInFrontOfIt) {
	const double expected[3] = {0.9549, 0.9549, 0.9549};
	expectChannelMeans("diffuse-transmission-point-light-front", 64, 4, expected, 0.01, 0.02);
}


// A closed diffuse sphere lit by a point light at its centre has a closed form at every depth (pbrt_scenes/bdpt-room-furnace.pbrt): after d bounces
// every wall point reads 0.5 * (1 + 0.5 + ... + 0.5^(d-1)), here 0.9375 at depth 4. It pins the absolute interreflection energy of the CPU and both GPU
// backends, and of BDPT - which a path-tracer-vs-BDPT comparison alone could not, since the two could be wrong together.
TEST(PbrtBackendAgreementTest, InterreflectionFurnaceHasItsClosedFormAtDepthFourOnEveryIntegrator) {
	const double expected[3] = {0.9375, 0.9375, 0.9375};
	expectChannelMeans("bdpt-room-furnace", 256, 4, expected, 0.01, 0.03);
	const SceneDescriptor* s = find_example_scene("bdpt-room-furnace");
	ASSERT_TRUE(s);
	const std::string out = "pbrt_agree_bdpt-room-furnace_bdpt.exr";
	double bdptMean = 0.0;
	ASSERT_EQ(cpu_render_main_bdpt(48, 48, 256, 4, out.c_str(), s->id.c_str(), 0.0, 0.0, 0.0), 0);
	ASSERT_TRUE(loadLinearMean(out, bdptMean));
	std::remove(out.c_str());
	std::printf("[furnace] bdpt-room-furnace depth 4: bdpt %.4f (closed form 0.9375)\n", bdptMean);
	EXPECT_NEAR(bdptMean, 0.9375, 0.01 * 0.9375);
}

TEST(PbrtBackendAgreementTest, ChromaticFogFurnaceStaysInvisibleInEveryChannel) {
	const double expected[3] = {1.0, 1.0, 1.0};
	// Measured: every backend within 0.3% of 1 in every channel.
	expectChannelMeans("chromatic-fog-furnace", 128, 24, expected, 0.01, 0.01);
}

// A finite light sitting closer to the surface behind it than the GPU's 0.01 shadow-ray nudge. The GPU
// used to measure a shadow ray's length from the un-nudged shading point, overshooting the light by the nudge
// and ending inside the ceiling 0.005 behind it, so the sample counted as blocked: recursive rendered this
// room at 22% of the CPU's brightness and wavefront at 37%. See pbrt_scenes/flush-ceiling-light.pbrt.
TEST(PbrtBackendAgreementTest, FlushCeilingLightAgreesAcrossBackends) {
	expectBackendsAgree("flush-ceiling-light", 64, 0.97, 1.03);
}

// A multi-faced emitter must occlude its own far side: the GPU used to skip emissive surfaces in shadow
// any-hits, so a ray aimed at the back of this octahedron passed through its front and every face counted
// as visible - both GPU backends rendered it 48% too bright. The CPU matches the closed-form answer.
TEST(PbrtBackendAgreementTest, EmissiveOctahedronFurnaceAgreesAcrossBackends) {
	expectBackendsAgree("emissive-octahedron-furnace", 128, 0.98, 1.02);
}

// A BSDF-sampled bounce that lands on a lamp has to be counted, MIS-weighted against the NEE sample taken at the
// previous vertex. The wavefront backend dropped it for every non-specular bounce while still weighting its NEE
// sample, so with a big lamp close to bright walls it lost a fifth of the indirect light (83% of CPU on Fireplace
// Room). Tight bounds: the three backends agree to about 1% here.
TEST(PbrtBackendAgreementTest, LargeAreaLightRoomAgreesAcrossBackends) {
	expectBackendsAgree("large-area-light-room", 64, 0.985, 1.015);
}

// Participating media: a point-lit fog sphere whose full multiple-scattering answer is known from an
// independent Monte Carlo solution (see the scene's header). The CPU used to attenuate a shadow ray through
// fog only on the rays that happened to scatter (T*(2-T) instead of T: 24% too bright here) and the
// recursive GPU backend never sampled punctual lights at a scatter event (black).
TEST(PbrtBackendAgreementTest, FogPointLightAgreesAcrossBackends) {
	expectBackendsAgree("fog-point-light", 128, 0.98, 1.02);
}

// A medium that absorbs as well as scatters: the GPU builder dropped the sigma_s/sigma_t factor, so every
// collision scattered at full strength (the fog came out ~14% too bright in this scene, a pure absorber 2.2x).
TEST(PbrtBackendAgreementTest, AbsorbingFogAgreesAcrossBackends) {
	// Wavefront reads 101.0-101.1% alone and 101.6% inside a full-suite run (recursive 100.5%), so +-1.5% sat on the edge of its own noise.
	expectBackendsAgree("absorbing-fog", 128, 0.985, 1.02);
}

// A fuzzed metal (the GPU loader's build for a conductor given only a reflectance) has to absorb the rays its
// fuzz sends below the surface, as the CPU's metal and the wavefront backend do. The recursive backend used to
// reflect them instead, so the plane kept all 0.9 of its reflectance (140% of the CPU's 0.645 at roughness 1).
TEST(PbrtBackendAgreementTest, FuzzedMetalFurnaceAgreesAcrossBackends) {
	expectBackendsAgree("fuzzed-metal-furnace", 128, 0.985, 1.015);
}

// A grayscale "texture displacement" is a bump map: the CPU perturbs the shading normal with it and the GPU
// ignored it, so a strongly bumped plane rendered ~58% too bright there (and Sibenik 12-15% too dark).
TEST(PbrtBackendAgreementTest, BumpMappedPlaneAgreesAcrossBackends) {
	expectBackendsAgree("bump-mapped-plane", 128, 0.985, 1.015);
}

// The CPU caps a path's throughput at 50; the GPU backends had no cap, so a closed hair shape (whose BSDF sample
// weight averages ~4 and compounds over its interior bounces) rendered 2.1x too bright there (B11: 2.3-2.5x).
TEST(PbrtBackendAgreementTest, HairSphereDimSkyAgreesAcrossBackends) {
	expectBackendsAgree("hair-sphere-dim-sky", 128, 0.985, 1.015);
}

// pbrt's camera medium on all three backends: CPU and recursive GPU attenuated a ray that passed through the fog
// twice, recursive GPU ignored the scattering albedo (and its builder dropped sigma_s/sigma_t), wavefront GPU had no
// camera medium at all. See the scene's header for the Monte Carlo numbers they now match.
TEST(PbrtBackendAgreementTest, CameraMediumAbsorbingAgreesAcrossBackends) {
	// 16 CPU renders of this scene: 1.537-1.683, median 1.577 (a point light in fog: heavy tail, upward). The GPU backends are steady
	// (recursive 1.57, wavefront 1.54-1.55 in every run), but a median of five CPU draws still ranged 1.556-1.627, which put the
	// wavefront at 94.8-99.1% of it and failed the 95% bound now and then. The CPU reference is the median of fifteen draws, whose
	// spread is about 1.7x smaller, and the bounds leave room for what remains.
	expectBackendsAgree("camera-medium-absorbing", 128, 0.95, 1.06, /*cpuRepeats=*/15);
}

// Disk and cylinder area lights. The CPU's cylinder pdf counted only the first of a ray's two crossings of the
// tube while its NEE credits whatever emitter the ray hits first, so the floor beside the cylinder light rendered up
// to 3x too bright and the whole scene ~8% above both GPU backends (which evaluate emission at the sampled point).
TEST(PbrtBackendAgreementTest, DiskCylinderLightAgreesAcrossBackends) {
	expectBackendsAgree("disk-cylinder-light", 128, 0.985, 1.015);
}

// A medium boundary written the way pbrt-v4 does it (Material "interface") must be a free crossing that leaves the MIS
// state of the last real vertex alone. The loader did not know the name "interface" (it fell to a grey diffuse), the
// recursive backend's free crossings never advanced the ray, and both GPU backends reset the MIS state at every
// medium pass-through, so a non-absorbing fog under a sky read 23-52% bright. See pbrt_scenes/fog-furnace.pbrt.
TEST(PbrtBackendAgreementTest, FogFurnaceAgreesAcrossBackends) {
	expectBackendsAgree("fog-furnace", 128, 0.985, 1.015);
}

// The heterogeneous media. The wavefront backend never received its uniform-grid arrays, so NEE shadow rays inside
// a "uniformgrid" medium were never attenuated (E8 rendered +10% bright, a sky-lit furnace 4x); the CPU sampled free
// flights along an unnormalised camera ray, a medium |d| times too thin for primary rays (4-10% too bright).
TEST(PbrtBackendAgreementTest, UniformGridMediumAgreesAcrossBackends) {
	expectBackendsAgree("uniformgrid-medium", 128, 0.985, 1.015);
}
TEST(PbrtBackendAgreementTest, RgbGridMediumAgreesAcrossBackends) {
	expectBackendsAgree("rgbgrid-medium", 128, 0.985, 1.015);  // measured 100.0% / 99.9% (it was 106% before the grid media were sampled per channel, with sigma_a)
}
TEST(PbrtBackendAgreementTest, CloudMediumAgreesAcrossBackends) {
	expectBackendsAgree("cloud-medium", 128, 0.985, 1.015);
}

// A glass shape that bounds a medium is opaque to NEE shadow rays, as every pbrt-v4 surface with a material is. The
// loader made it transparent (a near-invisible-shell idiom), which counted every escape from the fog twice. All three
// backends now block at the shell and agree. See pbrt_scenes/glass-fog-furnace.pbrt.
TEST(PbrtBackendAgreementTest, GlassFogFurnaceAgreesAcrossBackends) {
	expectBackendsAgree("glass-fog-furnace", 128, 0.985, 1.015);
}

// Normal-mapped and bump-mapped surfaces. The wavefront backend lit a normal-mapped surface with a white BSDF colour
// instead of its albedo (a blue sphere read grey, 164-188% of the CPU over it), and both GPU backends skipped bump
// mapping on a triangle mesh that has no UVs, which the CPU bumps through its barycentric fallback.
TEST(PbrtBackendAgreementTest, NormalMappedCornellAgreesAcrossBackends) {
	expectBackendsAgree("normal-mapped-cornell", 128, 0.985, 1.015);
}

// Rough glass under a small lamp, where NEE at the vertices that leave the glass carries the image. The shared
// RoughDielectricBxDF's transmission f() and pdf() both carried an extra eta^2 (2.25 for glass), so NEE there was counted
// 2.25x too strongly and MIS weighed against the wrong magnitude (CPU +20% against a pbrt-v4 path-level reference,
// GPU +15%); its GPU samplers also kept refracted samples that land on the wrong side of the surface, which pbrt
// rejects. See pbrt_scenes/rough-glass-lamp.pbrt and scripts/pbrt_rough_glass_reference.py.
TEST(PbrtBackendAgreementTest, RoughGlassUnderALampAgreesAcrossBackends) {
	// This scene's mean is dominated by the directly visible lamp, so it is a coarse check (see rough-glass-from-inside for
	// one whose every pixel is a glass vertex). The harness compares TONEMAPPED means, and a clipped, noisy image reads
	// lower the noisier it is (the tonemap is concave): at 128 spp the CPU read 6% under the GPU although the linear means
	// agree to 0.3%, so this takes more samples, and the CPU's mean on a 64x64 image still wanders by a few percent between
	// runs at 512 spp (hence 2048 and 5% bounds; the broken behaviour was 10-20% off).
	expectBackendsAgree("rough-glass-lamp", 2048, 0.985, 1.015);
}

// The camera inside a rough glass sphere: every pixel is an interior vertex where a large share of BSDF samples is rejected.
// Both GPU backends returned before the NEE step when the continuation sample was rejected, so they read 1/(1+alpha^2) of
// the CPU (63% at this scene's roughness 0.6; 50% at 1.0). pbrt takes the direct-light sample at every vertex regardless.
TEST(PbrtBackendAgreementTest, RoughGlassFromInsideAgreesAcrossBackends) {
	expectBackendsAgree("rough-glass-from-inside", 128, 0.985, 1.015);
}

// A rough copper sphere lit by an out-of-view lamp, so the whole image is glossy direct light. Both GPU backends skipped NEE
// at vertices whose BSDF continuation sample was rejected (86% / 75% of the CPU at roughness 0.25 / 1.0), and the CPU took the
// conductor's Fresnel at the view-normal cosine instead of the half-vector (104% / 108% of a pbrt-v4 path-level reference).
// See pbrt_scenes/rough-metal-lamp.pbrt.
TEST(PbrtBackendAgreementTest, RoughMetalLampAgreesAcrossBackends) {
	expectBackendsAgree("rough-metal-lamp", 512, 0.985, 1.015);
}

// An emissive SPHERE that is not the light being sampled has to block shadow rays like any other surface. Both GPU backends'
// sphere shadow any-hit programs ignored every emissive sphere, so a glowing ball under a lamp cast no shadow: 116.9% of the CPU.
// See pbrt_scenes/emissive-sphere-occluder.pbrt.
TEST(PbrtBackendAgreementTest, EmissiveSphereOccluderAgreesAcrossBackends) {
	expectBackendsAgree("emissive-sphere-occluder", 128, 0.985, 1.015);
}

// Smooth glass on the wavefront backend chose reflect-vs-refract with Schlick's approximation fed the incident cosine, which
// under-reflects for a ray leaving the glass: the light through two refractions was ~2.2x too bright and the glass read 105.3% of
// a pbrt path-level reference (CPU and recursive, which use the exact Fresnel, 100.4% / 100.9%). See pbrt_scenes/glass-sphere-lamp.pbrt.
TEST(PbrtBackendAgreementTest, GlassSphereLampAgreesAcrossBackends) {
	expectBackendsAgree("glass-sphere-lamp", 512, 0.985, 1.015);
}

// A smooth coat over smooth copper (pbrt's coatedconductor default) is a mirror whose reflectance has a closed form, and a smooth
// coat over a rough conductor is the layered BSDF's mixed case. The coated materials were a simplified model that never refracted
// at the coat: CPU 33% of the closed form and the GPU 24% of the CPU for the first, 72-84% of the CPU for rough ones. See
// pbrt_scenes/coated-conductor-lamp.pbrt and coated-conductor-glossy-lamp.pbrt.
TEST(PbrtBackendAgreementTest, CoatedConductorLampAgreesAcrossBackends) {
	expectBackendsAgree("coated-conductor-lamp", 256, 0.985, 1.015);
}

TEST(PbrtBackendAgreementTest, CoatedConductorGlossyLampAgreesAcrossBackends) {
	expectBackendsAgree("coated-conductor-glossy-lamp", 512, 0.985, 1.015);
}
