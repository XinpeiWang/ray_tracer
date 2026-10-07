/**
 * @file cpu_integrator_agreement_tests.cpp
 * @brief The CPU integrators (path tracer, BDPT, MLT, SPPM, --simplepath/--randomwalk/--lightpath) against each other and against closed forms
 *
 * Every test here renders on the CPU only, so it runs in the portable CMake target (and so in GitHub CI) as well as in the full MSVC suite.
 * They were written one by one as defects were found - each scene file under pbrt_scenes/ says which - and moved here from
 * pbrt_example_scenes_tests.cpp, whose remaining tests need the OptiX backends.
 */

#include "agreement_test_helpers.h"

#include <array>
#include <fstream>

// BDPT, MLT and SPPM draw a medium event from constant_medium::hit() - one luminance extinction - so a chromatic medium renders attenuated
// by the grey average there (the chromatic absorber reads 0.47 in every channel under --bdpt, --mlt and --sppm instead of 0.82/0.45/0.17). They
// say so instead of silently greying it: the warning names the scene and the integrator, and a grey medium or a medium-free scene gets none.
TEST(ChromaticMediaIntegratorWarningTest, NamesChromaticMediaAndOnlyThose) {
	const auto warn = [](const char* stem, const char* flags) {
		const SceneDescriptor* s = find_example_scene(stem);
		return s ? chromatic_media_integrator_warning(*s, stem, flags) : std::string("<scene missing>");
	};
	for (const char* stem : {"chromatic-absorber", "chromatic-camera-medium-absorber", "chromatic-camera-medium", "dielectric-medium-showcase"}) {
		const std::string w = warn(stem, "--bdpt/--mlt");
		EXPECT_NE(w.find("extinction differs between colour channels"), std::string::npos) << stem << ": " << w;
		EXPECT_NE(w.find(stem), std::string::npos) << stem << " should be named";
		EXPECT_NE(w.find("--bdpt/--mlt"), std::string::npos) << stem;
	}
	for (const char* stem : {"fog-furnace", "absorbing-fog", "camera-medium-absorbing", "flush-ceiling-light"})
		EXPECT_EQ(warn(stem, "--sppm"), "") << stem << " has no chromatic medium";
}

// BDPT and MLT must agree with the path tracer on a distant light. They built a distant light as a surface vertex at an invented point in
// SampleLight() and on the bounding disk in SampleLightLe(), so the MIS weights of the light-sampling and light-tracing strategies summed above 1
// and every distant-light scene read 1.34x too bright at every depth (pbrt-v4 treats LightType::DeltaDirection as an infinite light vertex). A
// diffuse sphere with direct lighting only is smooth, so all integrators read the same mean; the three of them are compared here.
TEST(PbrtBackendAgreementTest, BdptDistantLightAgreesWithPathTracer) {
	const SceneDescriptor* s = find_example_scene("distant-light-sphere");
	if (!s) GTEST_SKIP() << "distant-light-sphere.pbrt was not discovered - is pbrt_scenes/ present?";
	const std::string base = "pbrt_agree_distant-light-sphere";
	double pathMean = 0.0, bdptMean = 0.0;
	ASSERT_EQ(cpu_render_main(48, 48, 256, 4, (base + "_path.exr").c_str(), s->id.c_str(), 0.0, 0.0, 0.0), 0);
	ASSERT_TRUE(loadLinearMean(base + "_path.exr", pathMean));
	ASSERT_EQ(cpu_render_main_bdpt(48, 48, 256, 4, (base + "_bdpt.exr").c_str(), s->id.c_str(), 0.0, 0.0, 0.0), 0);
	ASSERT_TRUE(loadLinearMean(base + "_bdpt.exr", bdptMean));
	std::remove((base + "_path.exr").c_str());
	std::remove((base + "_bdpt.exr").c_str());
	std::printf("[agree] distant-light-sphere: path %.4f  bdpt %.4f (%.1f%%)\n", pathMean, bdptMean, 100.0 * bdptMean / pathMean);
	ASSERT_GT(pathMean, 0.01);
	EXPECT_GT(bdptMean, 0.98 * pathMean) << "BDPT too dark vs the path tracer on a distant light";
	EXPECT_LT(bdptMean, 1.02 * pathMean) << "BDPT too bright vs the path tracer on a distant light";
}

// BDPT against the path tracer on one scene at one depth: both are unbiased estimators of the same image, so the means agree to the noise (~0.2% here).
static void expectBdptAgreesWithPathTracer(const char* stem, int depth, double tol) {
	const SceneDescriptor* s = find_example_scene(stem);
	if (!s) GTEST_SKIP() << stem << ".pbrt was not discovered - is pbrt_scenes/ present?";
	const std::string base = std::string("pbrt_agree_") + stem;
	double pathMean = 0.0, bdptMean = 0.0;
	ASSERT_EQ(cpu_render_main(48, 48, 256, depth, (base + "_path.exr").c_str(), s->id.c_str(), 0.0, 0.0, 0.0), 0);
	ASSERT_TRUE(loadLinearMean(base + "_path.exr", pathMean));
	ASSERT_EQ(cpu_render_main_bdpt(48, 48, 256, depth, (base + "_bdpt.exr").c_str(), s->id.c_str(), 0.0, 0.0, 0.0), 0);
	ASSERT_TRUE(loadLinearMean(base + "_bdpt.exr", bdptMean));
	std::remove((base + "_path.exr").c_str());
	std::remove((base + "_bdpt.exr").c_str());
	std::printf("[agree] %s depth %d: path %.4f  bdpt %.4f (%.1f%%)\n", stem, depth, pathMean, bdptMean, 100.0 * bdptMean / pathMean);
	ASSERT_GT(pathMean, 0.01);
	EXPECT_GT(bdptMean, (1.0 - tol) * pathMean) << stem << ": BDPT too dark vs the path tracer";
	EXPECT_LT(bdptMean, (1.0 + tol) * pathMean) << stem << ": BDPT too bright vs the path tracer";
}

// A sky is an infinite-light vertex with no position; BDPTVertex::PDF() derived the direction toward such a `prev` vertex from its zeroed position (toward the
// world origin), so the MIS weights of s = 1 and 2 against s >= 3 were wrong whenever light entered a transmissive object from a sky: a diffuse-transmission
// shell read +14% at depth 3 and a rough glass sphere +8%. Both have an independent answer here (a closed form; the path tracer).
TEST(PbrtBackendAgreementTest, BdptSkyLitDiffuseTransmissionShellHasItsClosedForm) {
	const SceneDescriptor* s = find_example_scene("diffuse-transmission-furnace");
	if (!s) GTEST_SKIP() << "diffuse-transmission-furnace.pbrt was not discovered - is pbrt_scenes/ present?";
	const std::string out = "pbrt_agree_dtf_bdpt.exr";
	double bdptMean = 0.0;
	ASSERT_EQ(cpu_render_main_bdpt(48, 48, 512, 4, out.c_str(), s->id.c_str(), 0.0, 0.0, 0.0), 0);
	ASSERT_TRUE(loadLinearMean(out, bdptMean));
	std::remove(out.c_str());
	std::printf("[furnace] diffuse-transmission-furnace depth 4: bdpt %.4f (closed form 0.6464)\n", bdptMean);
	EXPECT_NEAR(bdptMean, 0.6464, 0.01 * 0.6464);
}

TEST(PbrtBackendAgreementTest, BdptSkyLitRoughGlassSphereAgreesWithPathTracer) {
	expectBdptAgreesWithPathTracer("bdpt-sky-rough-glass", 6, 0.012);
}

// SPPM against the path tracer on a scene smaller than its old fixed gather radius of 10 world units. That radius spans this whole 8-unit room, so the photon
// estimate averaged photons from all over it and the shrinking radius needed thousands of iterations to recover (0.30 after 100 iterations, 0.47 after 1600,
// against 0.65). The radius now starts at 2% of the scene's bounding radius (never above 10). See pbrt_scenes/sppm-transmissive-furnace.pbrt.
TEST(PbrtBackendAgreementTest, SppmInASmallSceneAgreesWithPathTracer) {
	const SceneDescriptor* s = find_example_scene("sppm-transmissive-furnace");
	if (!s) GTEST_SKIP() << "sppm-transmissive-furnace.pbrt was not discovered - is pbrt_scenes/ present?";
	const std::string pathOut = "pbrt_agree_sppm_path.exr", sppmOut = "pbrt_agree_sppm_sppm.exr";
	double pathMean = 0.0, sppmMean = 0.0;
	ASSERT_EQ(cpu_render_main(32, 32, 256, 30, pathOut.c_str(), s->id.c_str(), 0.0, 0.0, 0.0), 0);
	ASSERT_TRUE(loadLinearMean(pathOut, pathMean));
	ASSERT_EQ(cpu_render_main_sppm(32, 32, 100, 20000, 30, sppmOut.c_str(), s->id.c_str(), 0.0, 0.0, 0.0), 0);
	ASSERT_TRUE(loadLinearMean(sppmOut, sppmMean));
	std::remove(pathOut.c_str());
	std::remove(sppmOut.c_str());
	std::printf("[agree] sppm-transmissive-furnace: path %.4f  sppm %.4f (%.1f%%)\n", pathMean, sppmMean, 100.0 * sppmMean / pathMean);
	ASSERT_GT(pathMean, 0.3);
	EXPECT_GT(sppmMean, 0.95 * pathMean) << "SPPM too dark vs the path tracer";
	EXPECT_LT(sppmMean, 1.05 * pathMean) << "SPPM too bright vs the path tracer";
}

// Disk and cylinder emitters (Shape "disk"/"cylinder" with an AreaLightSource). BDPT, MLT and SPPM build their light list from shapes that can sample their
// own surface (sample_area()) and whose material they can read; the disk and cylinder had neither, so a scene lit only by them rendered with no light at all
// under --bdpt/--mlt/--sppm (0.001 of the path tracer's brightness on the lit surfaces). See pbrt_scenes/disk-cylinder-light.pbrt.
TEST(PbrtBackendAgreementTest, BdptDiskAndCylinderLightsAgreeWithPathTracer) {
	expectBdptAgreesWithPathTracer("disk-cylinder-light", 3, 0.02);
}

TEST(PbrtBackendAgreementTest, SppmDiskAndCylinderLightsAgreeWithPathTracer) {
	const SceneDescriptor* s = find_example_scene("disk-cylinder-light");
	if (!s) GTEST_SKIP() << "disk-cylinder-light.pbrt was not discovered - is pbrt_scenes/ present?";
	const std::string pathOut = "pbrt_agree_dcl_path.exr", sppmOut = "pbrt_agree_dcl_sppm.exr";
	double pathMean = 0.0, sppmMean = 0.0;
	ASSERT_EQ(cpu_render_main(48, 48, 256, 4, pathOut.c_str(), s->id.c_str(), 0.0, 0.0, 0.0), 0);
	ASSERT_TRUE(loadLinearMean(pathOut, pathMean));
	ASSERT_EQ(cpu_render_main_sppm(48, 48, 60, 20000, 4, sppmOut.c_str(), s->id.c_str(), 0.0, 0.0, 0.0), 0);
	ASSERT_TRUE(loadLinearMean(sppmOut, sppmMean));
	std::remove(pathOut.c_str());
	std::remove(sppmOut.c_str());
	std::printf("[agree] disk-cylinder-light: path %.4f  sppm %.4f (%.1f%%)\n", pathMean, sppmMean, 100.0 * sppmMean / pathMean);
	ASSERT_GT(pathMean, 0.1);
	EXPECT_GT(sppmMean, 0.95 * pathMean) << "SPPM too dark vs the path tracer";
	EXPECT_LT(sppmMean, 1.05 * pathMean) << "SPPM too bright vs the path tracer";
}

TEST(PbrtBackendAgreementTest, BdptAndSimplePathFilterTheirCameraRaysLikeThePathTracer) {
	const SceneDescriptor* s = find_example_scene("cornell-box-native");
	if (!s) GTEST_SKIP() << "cornell-box-native.pbrt was not discovered - is pbrt_scenes/ present?";
	const int N = 64, kDepth = 1, kSpp = 512;   // depth 1: the visible light plus direct light; no indirect noise
	int w = 0, h = 0;
	std::vector<float> path, bdpt, simple;
	const std::string a = "pbrt_agree_filter_path.exr", b = "pbrt_agree_filter_bdpt.exr", c = "pbrt_agree_filter_simple.exr";
	ASSERT_EQ(cpu_render_main(N, N, kSpp, kDepth, a.c_str(), s->id.c_str(), 0.0, 0.0, 0.0), 0);
	ASSERT_EQ(cpu_render_main_bdpt(N, N, kSpp, kDepth, b.c_str(), s->id.c_str(), 0.0, 0.0, 0.0), 0);
	ASSERT_EQ(cpu_render_main_simplepath(N, N, kSpp, kDepth, 1, 1, c.c_str(), s->id.c_str(), 0.0, 0.0, 0.0, 0), 0);
	ASSERT_TRUE(loadLinearRgbPixels(a, w, h, path));
	ASSERT_TRUE(loadLinearRgbPixels(b, w, h, bdpt));
	ASSERT_TRUE(loadLinearRgbPixels(c, w, h, simple));
	std::remove(a.c_str());
	std::remove(b.c_str());
	std::remove(c.c_str());
	// The light is near the top of the frame; take the window around it from the path tracer's own brightest pixel.
	int by = 0, bx = 0;
	float best = -1.0f;
	for (int y = 0; y < h; ++y)
		for (int x = 0; x < w; ++x) {
			const float v = path[3 * (static_cast<size_t>(y) * w + x) + 1];
			if (v > best) { best = v; by = y; bx = x; }
		}
	double sumPath = 0.0, diffBdpt = 0.0, diffSimple = 0.0;
	for (int y = std::max(0, by - 6); y <= std::min(h - 1, by + 6); ++y)
		for (int x = std::max(0, bx - 10); x <= std::min(w - 1, bx + 10); ++x) {
			const size_t i = 3 * (static_cast<size_t>(y) * w + x) + 1;
			sumPath += path[i];
			diffBdpt += std::fabs(path[i] - bdpt[i]);
			diffSimple += std::fabs(path[i] - simple[i]);
		}
	ASSERT_GT(sumPath, 1.0);
	std::printf("[agree] filter window around the light: bdpt differs from the path tracer by %.1f%%, simplepath by %.1f%%\n",
	            100.0 * diffBdpt / sumPath, 100.0 * diffSimple / sumPath);
	EXPECT_LT(diffBdpt / sumPath, 0.10) << "BDPT's light edge does not match the path tracer's filtered one";
	EXPECT_LT(diffSimple / sumPath, 0.10) << "--simplepath's light edge does not match the path tracer's filtered one";
}

// MLT has no estimator of its own beyond the Markov chain over BDPT's path space, so it inherits every defect in BDPT's strategy weights and bridge; it had no test
// of its own. Averaged over seeds it agrees with the path tracer to ~0.5% on every scene checked (a single run scatters by about +/-1%, which is where an earlier
// "MLT reads 1.5% low" came from - it was noise, not a bias), so the bound here is loose enough for one seed and still catches a defect like the BDPT ones fixed
// above (those read 0.78x-1.1x).
static void expectMltAgreesWithPathTracer(const char* stem, int depth, double tol) {
	const SceneDescriptor* s = find_example_scene(stem);
	if (!s) GTEST_SKIP() << stem << ".pbrt was not discovered - is pbrt_scenes/ present?";
	const std::string pathOut = std::string("pbrt_agree_mlt_") + stem + "_path.exr", mltOut = std::string("pbrt_agree_mlt_") + stem + "_mlt.exr";
	double pathMean = 0.0, mltMean = 0.0;
	ASSERT_EQ(cpu_render_main(48, 48, 256, depth, pathOut.c_str(), s->id.c_str(), 0.0, 0.0, 0.0), 0);
	ASSERT_TRUE(loadLinearMean(pathOut, pathMean));
	ASSERT_EQ(cpu_render_main_mlt(48, 48, 200000, 8000000, depth, mltOut.c_str(), s->id.c_str(), 0.0, 0.0, 0.0), 0);
	ASSERT_TRUE(loadLinearMean(mltOut, mltMean));
	std::remove(pathOut.c_str());
	std::remove(mltOut.c_str());
	std::printf("[agree] %s depth %d: path %.4f  mlt %.4f (%.1f%%)\n", stem, depth, pathMean, mltMean, 100.0 * mltMean / pathMean);
	ASSERT_GT(pathMean, 0.05);
	EXPECT_GT(mltMean, (1.0 - tol) * pathMean) << stem << ": MLT too dark vs the path tracer";
	EXPECT_LT(mltMean, (1.0 + tol) * pathMean) << stem << ": MLT too bright vs the path tracer";
}

TEST(PbrtBackendAgreementTest, MltClosedBoxWithLargeLightAgreesWithPathTracer) {
	expectMltAgreesWithPathTracer("bdpt-box-room", 6, 0.03);
}

TEST(PbrtBackendAgreementTest, MltSkyLitDiffuseTransmissionShellAgreesWithPathTracer) {
	expectMltAgreesWithPathTracer("diffuse-transmission-furnace", 4, 0.03);
}

// The debug integrators (--simplepath, --randomwalk, --lightpath) against the path tracer / a closed form. They were written for the scene convention
// "BSDFf() returns f * |cos|" but the BDPT adapter hands back the plain f, so they dropped the cosine at every bounce and every direct-light sample
// (--simplepath read 0.5 / 1.0 / 2.0 / 3.95 at depths 1 / 2 / 4 / 8 on a furnace whose answer is 0.5 / 0.75 / 0.94 / 1.0); --lightpath also counted the
// light-selection probability twice (once folded into pdf_pos, once as p_light), 2x too bright with two emitters.
static double renderMeanWith(const char* integrator, const SceneDescriptor* s, int depth, int spp) {
	const std::string out = std::string("pbrt_agree_dbg_") + integrator + ".exr";
	int rc = -1;
	const std::string name = integrator;
	if (name == "path") rc = cpu_render_main(32, 32, spp, depth, out.c_str(), s->id.c_str(), 0.0, 0.0, 0.0);
	else if (name == "simplepath") rc = cpu_render_main_simplepath(32, 32, spp, depth, 1, 1, out.c_str(), s->id.c_str(), 0.0, 0.0, 0.0, 0);
	else if (name == "randomwalk") rc = cpu_render_main_randomwalk(32, 32, spp, depth, out.c_str(), s->id.c_str(), 0.0, 0.0, 0.0, 0);
	else if (name == "lightpath") rc = cpu_render_main_lightpath(32, 32, spp, depth, out.c_str(), s->id.c_str(), 0.0, 0.0, 0.0, 0);
	double mean = -1.0;
	if (rc == 0) loadLinearMean(out, mean);
	std::remove(out.c_str());
	return mean;
}

TEST(PbrtBackendAgreementTest, SimplePathHasTheInterreflectionFurnaceClosedForm) {
	const SceneDescriptor* s = find_example_scene("bdpt-room-furnace");
	if (!s) GTEST_SKIP() << "bdpt-room-furnace.pbrt was not discovered - is pbrt_scenes/ present?";
	const double kDepth4 = 0.9375;   // 0.5 * (1 + 0.5 + 0.25 + 0.125)
	const double m = renderMeanWith("simplepath", s, 4, 128);
	std::printf("[debug-integrators] bdpt-room-furnace depth 4: simplepath %.4f (closed form %.4f)\n", m, kDepth4);
	EXPECT_NEAR(m, kDepth4, 0.015 * kDepth4);
}

TEST(PbrtBackendAgreementTest, DebugIntegratorsAgreeWithPathTracerInAClosedBox) {
	const SceneDescriptor* s = find_example_scene("bdpt-box-room");
	if (!s) GTEST_SKIP() << "bdpt-box-room.pbrt was not discovered - is pbrt_scenes/ present?";
	const int depth = 4, spp = 512;
	const double path = renderMeanWith("path", s, depth, spp);
	ASSERT_GT(path, 0.05);
	const struct { const char* name; double tol; } kCases[] = {{"simplepath", 0.03}, {"randomwalk", 0.05}, {"lightpath", 0.06}};
	for (const auto& c : kCases) {
		const double m = renderMeanWith(c.name, s, depth, spp);
		std::printf("[debug-integrators] bdpt-box-room depth %d: path %.4f  %s %.4f (%.1f%%)\n", depth, path, c.name, m, 100.0 * m / path);
		EXPECT_GT(m, (1.0 - c.tol) * path) << c.name << " too dark vs the path tracer";
		EXPECT_LT(m, (1.0 + c.tol) * path) << c.name << " too bright vs the path tracer";
	}
}

// Glossy and refracting materials in the BDPT bridge (bsdf_bridge.h). Each of these rooms read measurably off the path tracer before; the whole-image
// mean at depth 6 is what shows it (the lit surfaces dominate - the light is facing the ceiling and not in view).
TEST(PbrtBackendAgreementTest, BdptSmoothGlassSphereAgreesWithPathTracer) {
	expectBdptAgreesWithPathTracer("bdpt-box-room-glass", 6, 0.012);
}

TEST(PbrtBackendAgreementTest, BdptRoughGlassSphereAgreesWithPathTracer) {
	expectBdptAgreesWithPathTracer("bdpt-box-room-roughglass", 6, 0.012);
}

TEST(PbrtBackendAgreementTest, BdptRoughConductorSphereAgreesWithPathTracer) {
	expectBdptAgreesWithPathTracer("bdpt-box-room-roughmetal", 6, 0.012);
}

// A spherical area light: BDPT's light sampling drew a point over the whole sphere but divided by the visible-cone density, and dropped the samples
// on the far side - 0.48x of the path tracer at every depth. See pbrt_scenes/bdpt-room-spherelight.pbrt.
TEST(PbrtBackendAgreementTest, BdptSphereLightAgreesWithPathTracer) {
	expectBdptAgreesWithPathTracer("bdpt-room-spherelight", 4, 0.02);
}

// A two-sided light: the MIS weights' direction density (LightPDFLe) ignored the two sides that SampleLightLe emits to - 0.78x at depth 2.
// See pbrt_scenes/bdpt-room-twosided-light.pbrt.
TEST(PbrtBackendAgreementTest, BdptTwoSidedLightAgreesWithPathTracer) {
	expectBdptAgreesWithPathTracer("bdpt-room-twosided-light", 4, 0.02);
}

// A large light in a closed box, and its edge in view: the random walk left the light vertex's pdfRev unset (every strategy with s >= 3 then
// weighted wrongly: 0.75x at depth 6), and the camera rays all went through the same point of their pixel (hard-edged light, mean energy off).
// See pbrt_scenes/bdpt-box-room.pbrt.
TEST(PbrtBackendAgreementTest, BdptClosedBoxWithLargeLightAgreesWithPathTracer) {
	expectBdptAgreesWithPathTracer("bdpt-box-room", 6, 0.02);
}
