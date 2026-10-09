// pbrt_gpu_nanovdb_tests.cpp - the OptiX builder turns a "nanovdb" medium into the dense grid medium "uniformgrid" already uses (scene E9 used to render as flat fog on the
// GPU). Host-side field checks only, like the other pbrt_gpu_* tests: pbrt_gpu::build() is plain host code. Needs the OptiX SDK headers, so MSBuild only. Run from the
// repository root: the bundled .nvdb files are found by a relative path.
#include <gtest/gtest.h>

#include "pbrt_gpu_builder.h"
#include "../../src/shared/pbrt_flatten.h"
#include "../../src/shared/pbrt_scene.h"

namespace {

pbrt_flatten::FlatScene flattenSource(const std::string& text) {
	const pbrt_scene::ParseResult r = pbrt_scene::parse(text);
	EXPECT_TRUE(r.ok) << r.error;
	return pbrt_flatten::flatten(r.scene);
}

}  // namespace

TEST(PbrtGpuNanoVdbTest, TheOptixBuilderTurnsANanoVdbMediumIntoAGridMedium) {
	const pbrt_flatten::FlatScene flat = flattenSource(
		"MakeNamedMedium \"fog\" \"string type\" [ \"nanovdb\" ] \"string filename\" [ \"pbrt_scenes/nanovdb-sphere.nvdb\" ] "
		"\"rgb sigma_a\" [ 0 0 0 ] \"rgb sigma_s\" [ 0.6 0.6 0.6 ] \"float g\" [ 0.2 ]\n"
		"AttributeBegin\n"
		"  Material \"interface\"\n"
		"  MediumInterface \"fog\" \"\"\n"
		"  Shape \"sphere\" \"float radius\" [ 3 ]\n"
		"AttributeEnd\n");
	SceneData scene;
	const pbrt_gpu::BuildStats stats = pbrt_gpu::build(flat, scene);
	EXPECT_TRUE(stats.unsupportedMediumTypeCounts.empty()) << "nanovdb is a real medium now, not flat fog";
	ASSERT_EQ(scene.gridMediums.size(), 1u);
	const GpuGridMedium& m = scene.gridMediums[0];
	EXPECT_GT(m.nx, 1);
	EXPECT_EQ(scene.gridData.size() - static_cast<size_t>(m.dataOffset), static_cast<size_t>(m.nx) * m.ny * m.nz);
	EXPECT_NEAR(m.sigma_scale, 0.6f, 1e-6f);
	EXPECT_NEAR(m.phase_g, 0.2f, 1e-6f);
	EXPECT_GT(m.sigma_maj, 0.6f * 0.5f) << "the majorant covers the grid's densest voxel";
	for (int a = 0; a < 3; ++a) {
		EXPECT_LT(m.bounds_min[a], 0.0f);
		EXPECT_GT(m.bounds_max[a], 0.0f);
	}
	bool anyGridMaterial = false;
	for (const MaterialData& d : scene.materials) anyGridMaterial = anyGridMaterial || d.type == MaterialType::GridMedium;
	EXPECT_TRUE(anyGridMaterial);
}

TEST(PbrtGpuNanoVdbTest, AnUnreadableNanoVdbFileIsAnInvisibleGridAndATemperatureGridIsCountedAsDropped) {
	const pbrt_flatten::FlatScene broken = flattenSource(
		"MakeNamedMedium \"fog\" \"string type\" [ \"nanovdb\" ] \"string filename\" [ \"pbrt_scenes/does-not-exist.nvdb\" ] \"rgb sigma_s\" [ 1 1 1 ]\n"
		"AttributeBegin\n  Material \"interface\"\n  MediumInterface \"fog\" \"\"\n  Shape \"sphere\" \"float radius\" [ 3 ]\nAttributeEnd\n");
	SceneData scene;
	pbrt_gpu::build(broken, scene);
	ASSERT_EQ(scene.gridMediums.size(), 1u);
	EXPECT_EQ(scene.gridMediums[0].sigma_maj, 0.0f) << "an all-zero grid never scatters";

	const pbrt_flatten::FlatScene fire = flattenSource(
		"MakeNamedMedium \"fire\" \"string type\" [ \"nanovdb\" ] \"string filename\" [ \"pbrt_scenes/nanovdb-fire-test.nvdb\" ] "
		"\"string temperaturename\" [ \"temperature\" ] \"rgb sigma_a\" [ 1 1 1 ] \"rgb sigma_s\" [ 1 1 1 ]\n"
		"AttributeBegin\n  Material \"interface\"\n  MediumInterface \"fire\" \"\"\n  Shape \"sphere\" \"float radius\" [ 3 ]\nAttributeEnd\n");
	SceneData scene2;
	const pbrt_gpu::BuildStats stats2 = pbrt_gpu::build(fire, scene2);
	EXPECT_EQ(stats2.nanovdbEmissionDropped, 1);
	ASSERT_EQ(scene2.gridMediums.size(), 1u);
}
