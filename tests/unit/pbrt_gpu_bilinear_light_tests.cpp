/**
 * @file pbrt_gpu_bilinear_light_tests.cpp
 * @brief Emissive bilinear patches (pbrt "bilinearmesh") reach the GPU light list.
 *
 * sportscar-area-lights.pbrt authors all 5 of its studio light panels as
 * `Shape "bilinearmesh"` with an AreaLightSource, not trianglemesh/quads.
 * Before this task pbrt_flatten.h dropped the shape entirely (a genuinely
 * curved ruled surface has no quad/triangle representation to fall back to),
 * so that scene lost 100% of its light geometry and rendered pure black.
 *
 * These pin the builder's half of the fix - structural copy of
 * pbrt_gpu_triangle_light_tests.cpp for the same reason: an emissive
 * bilinear patch must reach scene.bilinearPatches, be tagged
 * GpuLightKind::BilinearPatch, and be indexed into the bilinearPatches array
 * rather than quads/triangles (see optix_renderer.cpp's alias-table power
 * loop, whose trailing Quad-assuming `else` this light kind had to get its
 * own explicit branch ahead of).
 */

#include <gtest/gtest.h>

#include "pbrt_gpu_builder.h"
#include "../../src/shared/pbrt_flatten.h"
#include "../../src/shared/pbrt_scene.h"

namespace {

pbrt_flatten::FlatScene flattenBilinearLight() {
	const pbrt_scene::ParseResult r = pbrt_scene::parse(
		"WorldBegin\n"
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 12 10 8 ]\n"
		"  Shape \"bilinearmesh\" \"point3 P\" [ 0 100 0   100 100 0   0 100 80   100 100 80 ]\n"
		"AttributeEnd\n");
	EXPECT_TRUE(r.ok) << r.error;
	return pbrt_flatten::flatten(r.scene);
}

} // namespace

TEST(PbrtGpuBilinearLightTest, EmissiveBilinearPatchReachesTheLightList) {
	const pbrt_flatten::FlatScene flat = flattenBilinearLight();
	SceneData scene;
	pbrt_gpu::build(flat, scene);

	ASSERT_EQ(scene.bilinearPatches.size(), 1u);
	ASSERT_EQ(scene.lightIndices.size(), 1u);
	ASSERT_EQ(scene.lightKinds.size(), 1u);
	EXPECT_EQ(scene.lightKinds[0], GpuLightKind::BilinearPatch);
}

TEST(PbrtGpuBilinearLightTest, LightIndexAddressesTheBilinearPatchArray) {
	const pbrt_flatten::FlatScene flat = flattenBilinearLight();
	SceneData scene;
	pbrt_gpu::build(flat, scene);

	// The same integer means a different array depending on the kind tag -
	// pointing a bilinear-patch light at params.quads (the alias-table
	// loop's old bare `else`) is exactly how this would silently misread
	// geometry it was never given: sportscar-area-lights.pbrt has 0 quads
	// and 5 bilinear-patch lights.
	ASSERT_EQ(scene.lightIndices.size(), 1u);
	ASSERT_EQ(scene.lightKinds[0], GpuLightKind::BilinearPatch);
	const int idx = scene.lightIndices[0];
	EXPECT_GE(idx, 0);
	EXPECT_LT(static_cast<std::size_t>(idx), scene.bilinearPatches.size());
	const MaterialData &m = scene.materials[scene.bilinearPatches[idx].materialIdx];
	EXPECT_EQ(m.type, MaterialType::DiffuseLight);
}

TEST(PbrtGpuBilinearLightTest, NonEmissiveBilinearPatchIsGeometryNotALight) {
	const pbrt_scene::ParseResult r = pbrt_scene::parse(
		"WorldBegin\n"
		"Shape \"bilinearmesh\" \"point3 P\" [ 0 0 0  1 0 0  0 1 0  1 1 0 ]\n");
	ASSERT_TRUE(r.ok) << r.error;
	const pbrt_flatten::FlatScene flat = pbrt_flatten::flatten(r.scene);

	SceneData scene;
	pbrt_gpu::build(flat, scene);

	EXPECT_EQ(scene.bilinearPatches.size(), 1u);
	EXPECT_TRUE(scene.lightIndices.empty())
		<< "a plain bilinear patch is geometry, not a light";
	EXPECT_TRUE(scene.lightKinds.empty());
}

TEST(PbrtGpuBilinearLightTest, MultiplePatchesEachGetADistinctLightIndex) {
	const pbrt_scene::ParseResult r = pbrt_scene::parse(
		"WorldBegin\n"
		"AttributeBegin\n"
		"  AreaLightSource \"diffuse\" \"rgb L\" [ 5 5 5 ]\n"
		"  Shape \"bilinearmesh\" \"point3 P\" [ 0 0 0  1 0 0  0 1 0  1 1 0 ]\n"
		"  Shape \"bilinearmesh\" \"point3 P\" [ 0 0 5  1 0 5  0 1 5  1 1 5 ]\n"
		"AttributeEnd\n");
	ASSERT_TRUE(r.ok) << r.error;
	const pbrt_flatten::FlatScene flat = pbrt_flatten::flatten(r.scene);

	SceneData scene;
	pbrt_gpu::build(flat, scene);

	ASSERT_EQ(scene.bilinearPatches.size(), 2u);
	ASSERT_EQ(scene.lightIndices.size(), 2u);
	EXPECT_NE(scene.lightIndices[0], scene.lightIndices[1]);
	for (const GpuLightKind k : scene.lightKinds)
		EXPECT_EQ(k, GpuLightKind::BilinearPatch);
}
