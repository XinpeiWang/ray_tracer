/**
 * @file pbrt_gpu_triangle_light_tests.cpp
 * @brief Emissive triangles that will not merge are still sampleable lights.
 *
 * pbrt has no quad shape, so an area light arrives as a `trianglemesh`.
 * pbrt_quadify.h rejoins the usual two-triangle rectangle into one
 * parallelogram, which the GPU's quad light sampler handles. What would not
 * merge - an odd triangle, a fan, anything non-parallelogram - used to be
 * emitted as geometry and left out of the light list: it glowed when a ray
 * happened to hit it, but next-event estimation could not aim at it, so the
 * GPU image came out darker and noisier than the CPU's with nothing on screen
 * to explain why.
 *
 * These pin the builder's half of the fix - that such triangles reach the
 * light list at all, tagged as GpuLightKind::Triangle and indexed into the
 * TRIANGLE array rather than the quad one. The device half (sample_triangle_
 * light's uniform point and solid-angle pdf) was verified separately by
 * rendering one rectangle both ways: through the trusted quad sampler and,
 * with merging disabled, through the triangle sampler. They agreed to 0.12%
 * at 2000 spp, which is Monte Carlo noise between two strategies for the same
 * light rather than a difference in what they converge to.
 */

#include <gtest/gtest.h>

#include "pbrt_gpu_builder.h"
#include "../../src/shared/pbrt_flatten.h"
#include "../../src/shared/pbrt_scene.h"

#include <cmath>

namespace {

// A three-triangle fan at irregular radii. Irregular matters: a REGULAR fan
// has equilateral wedges, adjacent pairs form rhombi, and quadify merges the
// whole light back into quads - so the obvious test scene tests nothing. An
// odd triangle count is the backstop.
pbrt_flatten::FlatScene flattenFanLight() {
	const pbrt_scene::ParseResult r = pbrt_scene::parse(
		"WorldBegin\n"
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 12 10 8 ]\n"
		"  Shape \"trianglemesh\"\n"
		"    \"integer indices\" [ 0 1 2   0 2 3   0 3 1 ]\n"
		"    \"point3 P\" [ 0 100 0   90 100 0   -20 100 70   -35 100 -55 ]\n"
		"AttributeEnd\n");
	EXPECT_TRUE(r.ok) << r.error;
	return pbrt_flatten::flatten(r.scene);
}

// The ordinary case, kept as a control: two triangles forming a rectangle,
// which must still take the cheaper merged-quad route.
pbrt_flatten::FlatScene flattenRectLight() {
	const pbrt_scene::ParseResult r = pbrt_scene::parse(
		"WorldBegin\n"
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 12 10 8 ]\n"
		"  Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ]\n"
		"    \"point3 P\" [ 0 100 0   100 100 0   100 100 80   0 100 80 ]\n"
		"AttributeEnd\n");
	EXPECT_TRUE(r.ok) << r.error;
	return pbrt_flatten::flatten(r.scene);
}

} // namespace

TEST(PbrtGpuTriangleLightTest, UnmergeableEmissiveTrianglesReachTheLightList) {
	const pbrt_flatten::FlatScene flat = flattenFanLight();
	SceneData scene;
	pbrt_gpu::build(flat, scene);

	ASSERT_TRUE(scene.quads.empty())
		<< "an irregular fan must not merge into quads, or this tests nothing";
	EXPECT_EQ(scene.lightIndices.size(), 3u)
		<< "every wedge of the fan should be sampleable";
	ASSERT_EQ(scene.lightKinds.size(), scene.lightIndices.size());
	for (const GpuLightKind k : scene.lightKinds)
		EXPECT_EQ(k, GpuLightKind::Triangle);
}

TEST(PbrtGpuTriangleLightTest, TriangleLightIndicesAddressTheTriangleArray) {
	const pbrt_flatten::FlatScene flat = flattenFanLight();
	SceneData scene;
	pbrt_gpu::build(flat, scene);

	// The whole point of the kind tag: the same integer means a different
	// array depending on it, and pointing a triangle light at params.quads is
	// how the previous light-table bug crashed the renderer outright.
	ASSERT_FALSE(scene.lightIndices.empty());
	for (std::size_t i = 0; i < scene.lightIndices.size(); ++i) {
		ASSERT_EQ(scene.lightKinds[i], GpuLightKind::Triangle);
		const int idx = scene.lightIndices[i];
		EXPECT_GE(idx, 0);
		EXPECT_LT(static_cast<std::size_t>(idx), scene.triangles.size());
		// And it must actually be an emitter.
		const MaterialData &m = scene.materials[scene.triangles[idx].materialIdx];
		EXPECT_EQ(m.type, MaterialType::DiffuseLight);
	}
}

TEST(PbrtGpuTriangleLightTest, EachLightIndexIsDistinct) {
	const pbrt_flatten::FlatScene flat = flattenFanLight();
	SceneData scene;
	pbrt_gpu::build(flat, scene);

	// Registering before the push_back means an off-by-one here would make
	// every wedge claim the same triangle - which would still render, just
	// with the light in the wrong place and the wrong total power.
	ASSERT_EQ(scene.lightIndices.size(), 3u);
	EXPECT_NE(scene.lightIndices[0], scene.lightIndices[1]);
	EXPECT_NE(scene.lightIndices[1], scene.lightIndices[2]);
	EXPECT_NE(scene.lightIndices[0], scene.lightIndices[2]);
}

TEST(PbrtGpuTriangleLightTest, AnOrdinaryRectangleStillTakesTheQuadPath) {
	const pbrt_flatten::FlatScene flat = flattenRectLight();
	SceneData scene;
	pbrt_gpu::build(flat, scene);

	// The triangle path is the fallback, not a replacement: merging two
	// triangles into one quad is both cheaper to sample and what nearly every
	// real pbrt area light wants.
	EXPECT_EQ(scene.quads.size(), 1u);
	ASSERT_EQ(scene.lightKinds.size(), 1u);
	EXPECT_EQ(scene.lightKinds[0], GpuLightKind::Quad);
}

TEST(PbrtGpuTriangleLightTest, NonEmissiveTrianglesAreNotLights) {
	const pbrt_scene::ParseResult r = pbrt_scene::parse(
		"WorldBegin\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0   10 0 0   0 10 0 ]\n");
	ASSERT_TRUE(r.ok) << r.error;
	const pbrt_flatten::FlatScene flat = pbrt_flatten::flatten(r.scene);

	SceneData scene;
	pbrt_gpu::build(flat, scene);

	EXPECT_EQ(scene.triangles.size(), 1u);
	EXPECT_TRUE(scene.lightIndices.empty())
		<< "a plain triangle is geometry, not a light";
	EXPECT_TRUE(scene.lightKinds.empty());
}
