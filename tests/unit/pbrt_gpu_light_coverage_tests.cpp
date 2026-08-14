/**
 * @file pbrt_gpu_light_coverage_tests.cpp
 * @brief No emissive geometry may go missing from the GPU light list.
 *
 * This is the drift check the existing CPU/GPU parity test cannot make for
 * pbrt scenes. That one walks the CPU hittable tree, which for a pbrt scene is
 * triangles behind a BVH it does not descend into, so it skips these scenes
 * entirely - which is how the GPU quietly failing to sample non-parallelogram
 * area lights went unnoticed in the first place.
 *
 * Comparing light COUNTS between the backends would be wrong: the GPU merges a
 * rectangle's two triangles into one quad and the CPU keeps both, so the two
 * legitimately disagree about how many lights a scene has. What must agree is
 * the total emissive AREA, which no amount of merging or splitting changes.
 *
 * The reference is computed straight from the flattened scene - not from
 * either builder - so this cannot be satisfied by the two backends making the
 * same mistake.
 */

#include <gtest/gtest.h>

#include "pbrt_gpu_builder.h"
#include "../../src/shared/pbrt_flatten.h"
#include "../../src/shared/pbrt_scene.h"
#include "../../src/shared/pbrt_discover.h"
#include "../../src/shared/pbrt_load.h"

#include <cmath>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

// Total emissive surface area according to the flattened scene itself - the
// neutral reference both builders are measured against.
double flatEmissiveArea(const pbrt_flatten::FlatScene &scene) {
	double total = 0.0;
	for (const pbrt_flatten::Triangle &t : scene.triangles) {
		if (t.areaLight < 0) continue;
		const double ex[3] = {t.v[3] - t.v[0], t.v[4] - t.v[1], t.v[5] - t.v[2]};
		const double ey[3] = {t.v[6] - t.v[0], t.v[7] - t.v[1], t.v[8] - t.v[2]};
		const double cx = ex[1]*ey[2] - ex[2]*ey[1];
		const double cy = ex[2]*ey[0] - ex[0]*ey[2];
		const double cz = ex[0]*ey[1] - ex[1]*ey[0];
		total += 0.5 * std::sqrt(cx*cx + cy*cy + cz*cz);
	}
	for (const pbrt_flatten::Sphere &s : scene.spheres) {
		if (s.areaLight < 0) continue;
		total += 4.0 * kPi * s.radius * s.radius;
	}
	return total;
}

// Total area of everything the GPU builder actually registered as sampleable,
// each entry measured by the same rule the renderer's alias table uses.
double gpuSampledLightArea(const SceneData &scene) {
	double total = 0.0;
	for (std::size_t i = 0; i < scene.lightIndices.size(); ++i) {
		const int idx = scene.lightIndices[i];
		switch (scene.lightKinds[i]) {
		case GpuLightKind::Sphere: {
			const SphereData &s = scene.spheres[idx];
			total += 4.0 * kPi * double(s.radius) * double(s.radius);
			break;
		}
		case GpuLightKind::Quad: {
			const QuadData &q = scene.quads[idx];
			total += double(length(cross(q.u, q.v)));   // w = u x v, |w| = area
			break;
		}
		case GpuLightKind::Triangle: {
			const TriangleData &t = scene.triangles[idx];
			total += 0.5 * double(length(cross(t.p1 - t.p0, t.p2 - t.p0)));
			break;
		}
		}
	}
	return total;
}

// Parses inline source rather than reading a file, so the invariant is pinned
// even if every bundled scene is deleted.
double areaRatioFor(const char *sceneSource) {
	const pbrt_scene::ParseResult r = pbrt_scene::parse(sceneSource);
	EXPECT_TRUE(r.ok) << r.error;
	const pbrt_flatten::FlatScene flat = pbrt_flatten::flatten(r.scene);

	SceneData gpu;
	pbrt_gpu::build(flat, gpu);

	const double reference = flatEmissiveArea(flat);
	EXPECT_GT(reference, 0.0) << "the test scene has no emissive geometry at all";
	return gpuSampledLightArea(gpu) / reference;
}

} // namespace

TEST(PbrtGpuLightCoverageTest, ARectangleLightIsFullyCovered) {
	// The ordinary case: two triangles merge into one quad, one light, same
	// total area.
	EXPECT_NEAR(areaRatioFor(
		"WorldBegin\n"
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 10 10 10 ]\n"
		"  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ]\n"
		"    \"point3 P\" [ 0 100 0   120 100 0   120 100 90   0 100 90 ]\n"
		"AttributeEnd\n"), 1.0, 1e-4);
}

TEST(PbrtGpuLightCoverageTest, AnIrregularFanLightIsFullyCovered) {
	// The case this whole light-kind effort exists for. Before emissive
	// triangles could be sampled, the GPU registered NONE of these, so the
	// ratio here was 0 - the light existed as glowing geometry and next-event
	// estimation could not aim at any of it.
	EXPECT_NEAR(areaRatioFor(
		"WorldBegin\n"
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 10 10 10 ]\n"
		"  Shape \"trianglemesh\"\n"
		"    \"integer indices\" [ 0 1 2   0 2 3   0 3 1 ]\n"
		"    \"point3 P\" [ 0 100 0   90 100 0   -20 100 70   -35 100 -55 ]\n"
		"AttributeEnd\n"), 1.0, 1e-4);
}

TEST(PbrtGpuLightCoverageTest, AMixOfShapesIsFullyCovered) {
	// A sphere light, a mergeable rectangle and an odd triangle at once, so
	// all three GpuLightKinds are exercised against one reference.
	EXPECT_NEAR(areaRatioFor(
		"WorldBegin\n"
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 8 8 8 ]\n"
		"  Translate 200 200 200\n"
		"  Shape \"sphere\" \"float radius\" [ 30 ]\n"
		"AttributeEnd\n"
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 10 10 10 ]\n"
		"  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ]\n"
		"    \"point3 P\" [ 0 100 0   120 100 0   120 100 90   0 100 90 ]\n"
		"AttributeEnd\n"
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 10 10 10 ]\n"
		"  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"    \"point3 P\" [ -200 100 0   -110 100 25   -160 100 95 ]\n"
		"AttributeEnd\n"), 1.0, 1e-4);
}

// Every .pbrt that ships with the project, checked the same way. Parameterised
// over whatever is on disk rather than a fixed list, so a scene added later is
// covered without anyone remembering to add it here.
class BundledPbrtLightCoverageTest
	: public ::testing::TestWithParam<pbrt_discover::Discovered> {};

TEST_P(BundledPbrtLightCoverageTest, EveryEmissiveShapeIsSampleable) {
	const pbrt_discover::Discovered &d = GetParam();
	if (!d.ok) GTEST_SKIP() << d.path << ": " << d.error;

	// loadFile() parses AND flattens, so this is already world-space geometry.
	// A full load can still fail even though the lightweight header scan
	// above succeeded - e.g. a NamedMaterial referenced before its
	// MakeNamedMaterial is declared, which real pbrt-v4 tolerates (it
	// resolves named entities in a deferred second pass) and this parser's
	// single-pass model does not. That is a real gap in scene-format
	// coverage, not something this drift check exists to catch, so it skips
	// the same way an unparseable header does rather than failing the suite.
	const pbrt_load::LoadResult r = pbrt_load::loadFile(d.path);
	if (!r.ok) GTEST_SKIP() << d.path << ": " << r.error;
	const pbrt_flatten::FlatScene &flat = r.scene;

	const double reference = flatEmissiveArea(flat);
	if (reference <= 0.0) GTEST_SKIP() << d.name << " has no area lights";

	SceneData gpu;
	pbrt_gpu::build(flat, gpu);

	EXPECT_NEAR(gpuSampledLightArea(gpu) / reference, 1.0, 1e-4)
		<< d.name << ": the GPU light list does not cover all of this scene's "
		   "emissive area - some of it emits when hit but cannot be sampled.";
}

INSTANTIATE_TEST_SUITE_P(
	Bundled, BundledPbrtLightCoverageTest,
	::testing::ValuesIn(pbrt_discover::scanDefaultPaths()),
	[](const ::testing::TestParamInfo<pbrt_discover::Discovered> &info) {
		std::string sanitized;
		for (char c : info.param.name)
			sanitized += std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
		return "Scene" + std::to_string(info.index) + "_" + sanitized;
	});
