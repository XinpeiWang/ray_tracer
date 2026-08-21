/**
 * @file pbrt_gpu_texture_tests.cpp
 * @brief Tests for pbrt Diffuse "reflectance" imagemap textures on the GPU
 *        builder (Round 4 Phase 2)
 *
 * Host-side field-correctness only, matching this project's established split
 * for pbrt GPU builder work (see pbrt_gpu_disk_cylinder_tests.cpp's own file
 * comment): pbrt_gpu::build() (pbrt_gpu_builder.h) is plain host code with no
 * GPU/OptiX dependency, so these run everywhere without hardware.
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

TEST(PbrtGpuTextureTest, DiffuseWithNoTextureLeavesTextureIdxUnset) {
	const pbrt_flatten::FlatScene flat = flattenSource(
		"Material \"diffuse\" \"rgb reflectance\" [ .5 .5 .5 ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	SceneData scene;
	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.materials.size(), 1u);
	EXPECT_EQ(scene.materials[0].type, MaterialType::Lambertian);
	EXPECT_EQ(scene.materials[0].textureIdx, -1);
	EXPECT_TRUE(scene.textures.empty());
}

TEST(PbrtGpuTextureTest, MissingImagemapFileLeavesTextureIdxUnset) {
	// flattenSource() alone (no pbrt_load.h pass) never resolves
	// textureFilename to a real path, so this exercises the same "file could
	// not be decoded" fallback pbrt_load.h's own resolution failure would
	// hit: getOrBuildPbrtImageTexture() must degrade to flat albedo, not
	// crash or leave textureIdx pointing at a bad entry.
	const pbrt_flatten::FlatScene flat = flattenSource(
		"Texture \"tmap\" \"spectrum\" \"imagemap\" \"string filename\" [ \"nope.png\" ]\n"
		"Material \"diffuse\" \"texture reflectance\" [ \"tmap\" ]\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ 0 0 0  1 0 0  0 1 0 ]\n");
	SceneData scene;
	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.materials.size(), 1u);
	EXPECT_EQ(scene.materials[0].type, MaterialType::Lambertian);
	EXPECT_EQ(scene.materials[0].textureIdx, -1);
}

TEST(PbrtGpuTextureTest, RealImagemapFileBuildsATextureIdxAndUploadsPixels) {
	// Uses the actual bundled ganesha statue texture (pbrt_scenes/ganesha/
	// textures/ganesha.png) - the scene that motivated this feature (see
	// pbrt_flatten.h's own "A parameter bound to a Texture" comment) - so
	// this proves the real stbi_loadf decode path, not just that a filled-in
	// string reaches the right field. Run from the repo root, same as every
	// other bundled-scene test (relative path resolves via CWD).
	pbrt_flatten::Material m;
	m.kind = pbrt_flatten::MaterialKind::Diffuse;
	m.textureFilename = "pbrt_scenes/ganesha/textures/ganesha.png";
	pbrt_flatten::FlatScene flat;
	flat.materials.push_back(m);

	SceneData scene;
	// pbrt_gpu::build() only builds materials that a shape actually
	// references (materialIndex lambda), so give it one triangle using this
	// material to force the Diffuse case to run.
	pbrt_flatten::Triangle tri{};
	tri.material = 0;
	tri.areaLight = -1;
	flat.triangles.push_back(tri);

	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.materials.size(), 1u);
	EXPECT_EQ(scene.materials[0].type, MaterialType::Lambertian);
	ASSERT_GE(scene.materials[0].textureIdx, 0)
		<< "ganesha.png must decode successfully when run from the repo root";
	const TextureData &tex = scene.textures[static_cast<std::size_t>(scene.materials[0].textureIdx)];
	EXPECT_EQ(tex.kind, TextureKind::Image);
	EXPECT_GT(tex.width, 0);
	EXPECT_GT(tex.height, 0);
	EXPECT_FALSE(scene.texturePixels.empty());
}

TEST(PbrtGpuTextureTest, ShapeAlphaSetsAlphaMaskTexIdxOnTheOwningMaterial) {
	// Round 4 Phase 3: Material::alphaTextureFilename (a Shape "alpha"
	// cutout mask, see that field's own comment) must reach
	// MaterialData::alphaMaskTexIdx - the SAME field OBJ/MTL's own map_d
	// already wires into both GPU backends' any-hit/closest-hit alpha tests
	// (optix_anyhit_shadow.h, optix_intersection_triangle.h,
	// wavefront_programs.cu), so setting it here is sufficient to make both
	// backends respect a pbrt-authored alpha mask with no further wiring.
	pbrt_flatten::Material m;
	m.kind = pbrt_flatten::MaterialKind::Diffuse;
	m.alphaTextureFilename = "pbrt_scenes/ganesha/textures/ganesha.png";
	pbrt_flatten::FlatScene flat;
	flat.materials.push_back(m);
	pbrt_flatten::Triangle tri{};
	tri.material = 0;
	tri.areaLight = -1;
	flat.triangles.push_back(tri);

	SceneData scene;
	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.materials.size(), 1u);
	ASSERT_GE(scene.materials[0].alphaMaskTexIdx, 0)
		<< "ganesha.png must decode successfully when run from the repo root";
	const TextureData &tex = scene.textures[static_cast<std::size_t>(scene.materials[0].alphaMaskTexIdx)];
	EXPECT_EQ(tex.kind, TextureKind::Image);
	EXPECT_GT(tex.width, 0);
}

TEST(PbrtGpuTextureTest, NoAlphaParameterLeavesAlphaMaskTexIdxUnset) {
	pbrt_flatten::Material m;
	m.kind = pbrt_flatten::MaterialKind::Diffuse;
	pbrt_flatten::FlatScene flat;
	flat.materials.push_back(m);
	pbrt_flatten::Triangle tri{};
	tri.material = 0;
	tri.areaLight = -1;
	flat.triangles.push_back(tri);

	SceneData scene;
	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.materials.size(), 1u);
	EXPECT_EQ(scene.materials[0].alphaMaskTexIdx, -1);
}
