/**
 * @file pbrt_gpu_mix_material_tests.cpp
 * @brief Host-side field-correctness tests for Material "mix" on the GPU
 *        builder (pbrt_gpu_builder.h's MaterialKind::Mix case).
 *
 * pbrt_gpu::build() is plain host code with no GPU/OptiX dependency (same
 * split as pbrt_gpu_disk_cylinder_tests.cpp's own file comment), so these
 * run everywhere without hardware. They check that a Mix material resolves
 * to a real MaterialType::Mix entry whose mix_extra indices point at real,
 * correctly-typed sub-material MaterialData entries in scene.materials -
 * the actual per-shading-point stochastic pick itself (resolve_mix_
 * material()/wf_resolve_mix_material(), optix_device_helpers.h/
 * wavefront_kernels.cu) is device code, verified instead by rendering
 * pbrt_scenes/mix-material.pbrt across all three backends and comparing.
 */

#include <gtest/gtest.h>

#include "pbrt_gpu_builder.h"
#include "../../src/shared/pbrt_flatten.h"
#include "../../src/shared/pbrt_scene.h"

namespace {

pbrt_flatten::FlatScene flattenSource(const std::string &text) {
	const pbrt_scene::ParseResult r = pbrt_scene::parse(text);
	EXPECT_TRUE(r.ok) << r.error;
	return pbrt_flatten::flatten(r.scene);
}

} // namespace

TEST(PbrtGpuMixMaterialTest, ResolvesToRealMixTypeWithRealSubMaterialIndices) {
	const pbrt_flatten::FlatScene flat = flattenSource(
		"MakeNamedMaterial \"a\" \"string type\" [ \"diffuse\" ] \"rgb reflectance\" [ 1 0 0 ]\n"
		// A named metal spectrum (not "rgb reflectance") is what actually
		// resolves to a real MaterialType::Conductor on GPU - an explicit
		// RGB eta/k, or no eta/k at all, falls back to the fuzz-sphere Metal
		// approximation instead (docs/PBRT_SUPPORT.md's `conductor` row).
		"MakeNamedMaterial \"b\" \"string type\" [ \"conductor\" ] "
			"\"spectrum eta\" [ \"metal-Cu-eta\" ] \"spectrum k\" [ \"metal-Cu-k\" ]\n"
		"Material \"mix\" \"string materials\" [ \"a\" \"b\" ] \"float amount\" [ 0.25 ]\n"
		"Shape \"sphere\" \"float radius\" [ 1 ]\n");
	SceneData scene;
	const pbrt_gpu::BuildStats stats = pbrt_gpu::build(flat, scene);
	ASSERT_EQ(stats.spheres, 1u);
	ASSERT_EQ(scene.spheres.size(), 1u);

	const int mixIdx = scene.spheres[0].materialIdx;
	ASSERT_GE(mixIdx, 0);
	ASSERT_LT(static_cast<std::size_t>(mixIdx), scene.materials.size());
	const MaterialData &mix = scene.materials[static_cast<std::size_t>(mixIdx)];
	EXPECT_EQ(mix.type, MaterialType::Mix);
	EXPECT_FLOAT_EQ(mix.mix_extra.mixWeight, 0.25f);

	const int idxA = static_cast<int>(mix.mix_extra.mixMaterialAIdx);
	const int idxB = static_cast<int>(mix.mix_extra.mixMaterialBIdx);
	ASSERT_GE(idxA, 0);
	ASSERT_LT(static_cast<std::size_t>(idxA), scene.materials.size());
	ASSERT_GE(idxB, 0);
	ASSERT_LT(static_cast<std::size_t>(idxB), scene.materials.size());
	// Real MaterialData for each sub-material, not merely its flat colour -
	// the whole point of this feature over the old averaged-Lambertian
	// fallback (a Conductor sub-material must stay a real Conductor so it
	// keeps its specular highlight on GPU, not become flat diffuse).
	EXPECT_EQ(scene.materials[static_cast<std::size_t>(idxA)].type, MaterialType::Lambertian);
	EXPECT_EQ(scene.materials[static_cast<std::size_t>(idxB)].type, MaterialType::Conductor);
}

TEST(PbrtGpuMixMaterialTest, NestedMixOfMixResolvesBothLevelsToRealMixTypes) {
	const pbrt_flatten::FlatScene flat = flattenSource(
		"MakeNamedMaterial \"a\" \"string type\" [ \"diffuse\" ] \"rgb reflectance\" [ 1 0 0 ]\n"
		"MakeNamedMaterial \"b\" \"string type\" [ \"diffuse\" ] \"rgb reflectance\" [ 0 1 0 ]\n"
		"MakeNamedMaterial \"inner\" \"string type\" [ \"mix\" ] "
			"\"string materials\" [ \"a\" \"b\" ] \"float amount\" [ 0.5 ]\n"
		"MakeNamedMaterial \"c\" \"string type\" [ \"conductor\" ]\n"
		"Material \"mix\" \"string materials\" [ \"inner\" \"c\" ] \"float amount\" [ 0.5 ]\n"
		"Shape \"sphere\" \"float radius\" [ 1 ]\n");
	SceneData scene;
	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.spheres.size(), 1u);

	const int outerIdx = scene.spheres[0].materialIdx;
	ASSERT_GE(outerIdx, 0);
	const MaterialData &outer = scene.materials[static_cast<std::size_t>(outerIdx)];
	EXPECT_EQ(outer.type, MaterialType::Mix);

	const int innerIdx = static_cast<int>(outer.mix_extra.mixMaterialAIdx);
	ASSERT_GE(innerIdx, 0);
	ASSERT_LT(static_cast<std::size_t>(innerIdx), scene.materials.size());
	EXPECT_EQ(scene.materials[static_cast<std::size_t>(innerIdx)].type, MaterialType::Mix);

	const int conductorIdx = static_cast<int>(outer.mix_extra.mixMaterialBIdx);
	ASSERT_GE(conductorIdx, 0);
	ASSERT_LT(static_cast<std::size_t>(conductorIdx), scene.materials.size());
	EXPECT_EQ(scene.materials[static_cast<std::size_t>(conductorIdx)].type, MaterialType::Conductor);
}

TEST(PbrtGpuMixMaterialTest, AmountDefaultsToOneHalf) {
	const pbrt_flatten::FlatScene flat = flattenSource(
		"MakeNamedMaterial \"a\" \"string type\" [ \"diffuse\" ]\n"
		"MakeNamedMaterial \"b\" \"string type\" [ \"diffuse\" ]\n"
		"Material \"mix\" \"string materials\" [ \"a\" \"b\" ]\n"
		"Shape \"sphere\" \"float radius\" [ 1 ]\n");
	SceneData scene;
	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.spheres.size(), 1u);

	const int mixIdx = scene.spheres[0].materialIdx;
	ASSERT_GE(mixIdx, 0);
	EXPECT_FLOAT_EQ(scene.materials[static_cast<std::size_t>(mixIdx)].mix_extra.mixWeight, 0.5f);
}

// A mix with the SAME two named sub-materials, used on two different
// shapes, must dedupe to one MaterialData entry each - mirroring every
// other material kind's existing dedup-by-(material,areaLight) cache
// (build()'s own `cache` map), not something Mix's own materialIndexDepth
// plumbing should bypass.
TEST(PbrtGpuMixMaterialTest, SharedMixMaterialDedupesAcrossShapes) {
	const pbrt_flatten::FlatScene flat = flattenSource(
		"MakeNamedMaterial \"a\" \"string type\" [ \"diffuse\" ]\n"
		"MakeNamedMaterial \"b\" \"string type\" [ \"conductor\" ]\n"
		"Material \"mix\" \"string materials\" [ \"a\" \"b\" ]\n"
		"Shape \"sphere\" \"float radius\" [ 1 ]\n"
		"Shape \"sphere\" \"float radius\" [ 2 ]\n");
	SceneData scene;
	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.spheres.size(), 2u);
	EXPECT_EQ(scene.spheres[0].materialIdx, scene.spheres[1].materialIdx);
}
