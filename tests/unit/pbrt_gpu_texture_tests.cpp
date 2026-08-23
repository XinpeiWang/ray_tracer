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

#include <cstdio>
#include <cstdlib>
#include <fstream>

#include "pbrt_gpu_builder.h"
#include "../../src/shared/pbrt_flatten.h"
#include "../../src/shared/pbrt_scene.h"

namespace {

pbrt_flatten::FlatScene flattenSource(const std::string &text) {
	const pbrt_scene::ParseResult r = pbrt_scene::parse(text);
	EXPECT_TRUE(r.ok) << r.error;
	return pbrt_flatten::flatten(r.scene);
}

// Hand-authored minimal 1x1 24bpp BMP, same technique as pbrt_alpha_cutout_
// tests.cpp's own solidBmp1x1() (duplicated rather than shared - these are
// small, self-contained test-file helpers, not production code). A 1x1
// image is sufficient for is_grayscale_texture_gpu()/isPbrtTextureGrayscale()
// - its 8x8 sample grid maps every sample to pixel (0,0) regardless.
std::string solidBmp1x1(unsigned char r, unsigned char g, unsigned char b) {
	std::string bytes;
	const auto u16 = [&](unsigned short v) {
		bytes.push_back(static_cast<char>(v & 0xFF));
		bytes.push_back(static_cast<char>((v >> 8) & 0xFF));
	};
	const auto u32 = [&](unsigned int v) {
		bytes.push_back(static_cast<char>(v & 0xFF));
		bytes.push_back(static_cast<char>((v >> 8) & 0xFF));
		bytes.push_back(static_cast<char>((v >> 16) & 0xFF));
		bytes.push_back(static_cast<char>((v >> 24) & 0xFF));
	};
	bytes.push_back('B'); bytes.push_back('M');
	u32(14 + 40 + 4);
	u32(0);
	u32(14 + 40);
	u32(40);
	u32(1);
	u32(1);
	u16(1);
	u16(24);
	u32(0);
	u32(4);
	u32(0); u32(0);
	u32(0); u32(0);
	bytes.push_back(static_cast<char>(b));
	bytes.push_back(static_cast<char>(g));
	bytes.push_back(static_cast<char>(r));
	bytes.push_back('\0');
	return bytes;
}

class GpuTextureTempTree : public ::testing::Test {
protected:
	void SetUp() override {
		const char *tmp = std::getenv("TEMP");
		root_ = std::string(tmp ? tmp : ".") + "/pbrt_gpu_texture_tests/";
		std::string cmd = "if not exist \"" + root_ + "\" mkdir \"" + root_ + "\" >nul 2>&1";
		for (char &c : cmd) if (c == '/') c = '\\';
		std::system(cmd.c_str());
	}
	void TearDown() override {
		for (const std::string &f : written_) std::remove(f.c_str());
	}
	void write(const std::string &relative, const std::string &contents) {
		const std::string full = root_ + relative;
		std::ofstream out(full, std::ios::binary);
		out << contents;
		out.close();
		written_.push_back(full);
	}
	std::string path(const std::string &relative) const { return root_ + relative; }
private:
	std::string root_;
	std::vector<std::string> written_;
};

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

TEST_F(GpuTextureTempTree, DisplacementWithARealNormalMapBecomesNormalMappedLambertian) {
	// Round 5 Phase 1: a Lambertian material with a "texture displacement"
	// binding, resolved to a tangent-space RGB normal map (not grayscale -
	// see isPbrtTextureGrayscale()'s own comment), must upgrade from plain
	// Lambertian to MaterialType::NormalMappedLambertian with textureIdx
	// pointing at the decoded normal map - mirroring scene_builder.cpp's own
	// OBJ/MTL map_Bump dispatch for the RGB case exactly.
	write("normal.bmp", solidBmp1x1(128, 128, 255));

	pbrt_flatten::Material m;
	m.kind = pbrt_flatten::MaterialKind::Diffuse;
	m.displacementTextureFilename = path("normal.bmp");
	pbrt_flatten::FlatScene flat;
	flat.materials.push_back(m);
	pbrt_flatten::Triangle tri{};
	tri.material = 0;
	tri.areaLight = -1;
	flat.triangles.push_back(tri);

	SceneData scene;
	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.materials.size(), 1u);
	EXPECT_EQ(scene.materials[0].type, MaterialType::NormalMappedLambertian);
	ASSERT_GE(scene.materials[0].textureIdx, 0);
	const TextureData &tex = scene.textures[static_cast<std::size_t>(scene.materials[0].textureIdx)];
	EXPECT_EQ(tex.kind, TextureKind::Image);
}

TEST_F(GpuTextureTempTree, DisplacementWithARealGrayscaleBumpMapStaysPlainLambertian) {
	// The scalar/grayscale case: GPU has no device-side scalar bump
	// perturbation path (see makeMaterial()'s own comment on this), so a
	// real grayscale displacement image must leave the material as plain
	// Lambertian rather than mis-rendering it through the normal-map unpack
	// path (which would decode a flat R==G==B image as a degenerate normal
	// offset only along one fixed diagonal).
	write("bump.bmp", solidBmp1x1(128, 128, 128));

	pbrt_flatten::Material m;
	m.kind = pbrt_flatten::MaterialKind::Diffuse;
	m.displacementTextureFilename = path("bump.bmp");
	pbrt_flatten::FlatScene flat;
	flat.materials.push_back(m);
	pbrt_flatten::Triangle tri{};
	tri.material = 0;
	tri.areaLight = -1;
	flat.triangles.push_back(tri);

	SceneData scene;
	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.materials.size(), 1u);
	EXPECT_EQ(scene.materials[0].type, MaterialType::Lambertian);
}

TEST(PbrtGpuTextureTest, NoDisplacementParameterLeavesMaterialUnperturbed) {
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
	EXPECT_EQ(scene.materials[0].type, MaterialType::Lambertian);
	EXPECT_EQ(scene.materials[0].textureIdx, -1);
}

TEST(PbrtGpuTextureTest, CheckerReflectanceBuildsAUVCheckerTexture) {
	// Round 5 Phase 2: Material::hasCheckerReflectance must reach a real
	// TextureKind::UVChecker entry in out.textures with the resolved
	// colours/scales, and MaterialData::textureIdx pointing at it.
	pbrt_flatten::Material m;
	m.kind = pbrt_flatten::MaterialKind::Diffuse;
	m.hasCheckerReflectance = true;
	m.checkerColor1[0] = 1.0; m.checkerColor1[1] = 0.0; m.checkerColor1[2] = 0.0;
	m.checkerColor2[0] = 0.0; m.checkerColor2[1] = 0.0; m.checkerColor2[2] = 1.0;
	m.checkerUScale = 4.0;
	m.checkerVScale = 8.0;
	pbrt_flatten::FlatScene flat;
	flat.materials.push_back(m);
	pbrt_flatten::Triangle tri{};
	tri.material = 0;
	tri.areaLight = -1;
	flat.triangles.push_back(tri);

	SceneData scene;
	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.materials.size(), 1u);
	EXPECT_EQ(scene.materials[0].type, MaterialType::Lambertian);
	ASSERT_GE(scene.materials[0].textureIdx, 0);
	const TextureData &tex = scene.textures[static_cast<std::size_t>(scene.materials[0].textureIdx)];
	EXPECT_EQ(tex.kind, TextureKind::UVChecker);
	EXPECT_FLOAT_EQ(tex.color1.x, 1.0f);
	EXPECT_FLOAT_EQ(tex.color2.z, 1.0f);
	EXPECT_FLOAT_EQ(tex.uScale, 4.0f);
	EXPECT_FLOAT_EQ(tex.vScale, 8.0f);
}

TEST(PbrtGpuTextureTest, CheckerNestedImagemapTex1BuildsATex1ImageIdx) {
	// One-level-nested tex1 (Material::checkerTex1Filename - see that
	// field's own comment) must build a REAL decoded image entry and set
	// TextureData::tex1ImageIdx to it, leaving color1 unused; tex2 stays the
	// flat literal it always was. Uses the same real bundled image the
	// RealImagemapFileBuildsATextureIdxAndUploadsPixels test above already
	// proves decodes successfully from the repo root.
	pbrt_flatten::Material m;
	m.kind = pbrt_flatten::MaterialKind::Diffuse;
	m.hasCheckerReflectance = true;
	m.checkerTex1Filename = "pbrt_scenes/ganesha/textures/ganesha.png";
	m.checkerColor2[0] = 0.0; m.checkerColor2[1] = 0.0; m.checkerColor2[2] = 1.0;
	m.checkerUScale = 1.0;
	m.checkerVScale = 1.0;
	pbrt_flatten::FlatScene flat;
	flat.materials.push_back(m);
	pbrt_flatten::Triangle tri{};
	tri.material = 0;
	tri.areaLight = -1;
	flat.triangles.push_back(tri);

	SceneData scene;
	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.materials.size(), 1u);
	ASSERT_GE(scene.materials[0].textureIdx, 0);
	const TextureData &tex = scene.textures[static_cast<std::size_t>(scene.materials[0].textureIdx)];
	EXPECT_EQ(tex.kind, TextureKind::UVChecker);
	ASSERT_GE(tex.tex1ImageIdx, 0) << "tex1 must resolve to a real decoded image entry";
	EXPECT_EQ(tex.tex2ImageIdx, -1) << "tex2 stayed a flat literal, must not get an image index";
	const TextureData &tex1Img = scene.textures[static_cast<std::size_t>(tex.tex1ImageIdx)];
	EXPECT_EQ(tex1Img.kind, TextureKind::Image);
	EXPECT_GT(tex1Img.width, 0);
	EXPECT_FLOAT_EQ(tex.color2.z, 1.0f);
}

TEST(PbrtGpuTextureTest, FbmReflectanceBuildsAnFBmTexture) {
	// Round 6 Phase 1: Material::hasFbmReflectance must reach a real
	// TextureKind::FBm entry with the resolved octaves/roughness.
	pbrt_flatten::Material m;
	m.kind = pbrt_flatten::MaterialKind::Diffuse;
	m.hasFbmReflectance = true;
	m.fbmOctaves = 4;
	m.fbmRoughness = 0.3;
	pbrt_flatten::FlatScene flat;
	flat.materials.push_back(m);
	pbrt_flatten::Triangle tri{};
	tri.material = 0;
	tri.areaLight = -1;
	flat.triangles.push_back(tri);

	SceneData scene;
	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.materials.size(), 1u);
	EXPECT_EQ(scene.materials[0].type, MaterialType::Lambertian);
	ASSERT_GE(scene.materials[0].textureIdx, 0);
	const TextureData &tex = scene.textures[static_cast<std::size_t>(scene.materials[0].textureIdx)];
	EXPECT_EQ(tex.kind, TextureKind::FBm);
	EXPECT_EQ(tex.octaves, 4);
	EXPECT_FLOAT_EQ(tex.omega, 0.3f);
}

TEST(PbrtGpuTextureTest, MarbleReflectanceBuildsAMarbleTexture) {
	// Round 6 Phase 1: Material::hasMarbleReflectance must reach a real
	// TextureKind::Marble entry with the resolved octaves/roughness/scale/
	// variation.
	pbrt_flatten::Material m;
	m.kind = pbrt_flatten::MaterialKind::Diffuse;
	m.hasMarbleReflectance = true;
	m.marbleOctaves = 6;
	m.marbleRoughness = 0.4;
	m.marbleScale = 2.0;
	m.marbleVariation = 0.3;
	pbrt_flatten::FlatScene flat;
	flat.materials.push_back(m);
	pbrt_flatten::Triangle tri{};
	tri.material = 0;
	tri.areaLight = -1;
	flat.triangles.push_back(tri);

	SceneData scene;
	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.materials.size(), 1u);
	EXPECT_EQ(scene.materials[0].type, MaterialType::Lambertian);
	ASSERT_GE(scene.materials[0].textureIdx, 0);
	const TextureData &tex = scene.textures[static_cast<std::size_t>(scene.materials[0].textureIdx)];
	EXPECT_EQ(tex.kind, TextureKind::Marble);
	EXPECT_EQ(tex.octaves, 6);
	EXPECT_FLOAT_EQ(tex.omega, 0.4f);
	EXPECT_FLOAT_EQ(tex.marbleScale, 2.0f);
	EXPECT_FLOAT_EQ(tex.marbleVariation, 0.3f);
}

TEST(PbrtGpuTextureTest, MixReflectanceBuildsAMixTexture) {
	// Round 6 Phase 1: Material::hasMixReflectance must reach a real
	// TextureKind::Mix entry with the resolved colours/amount.
	pbrt_flatten::Material m;
	m.kind = pbrt_flatten::MaterialKind::Diffuse;
	m.hasMixReflectance = true;
	m.mixColor1[0] = 1.0; m.mixColor1[1] = 0.0; m.mixColor1[2] = 0.0;
	m.mixColor2[0] = 0.0; m.mixColor2[1] = 0.0; m.mixColor2[2] = 1.0;
	m.mixAmount = 0.25;
	pbrt_flatten::FlatScene flat;
	flat.materials.push_back(m);
	pbrt_flatten::Triangle tri{};
	tri.material = 0;
	tri.areaLight = -1;
	flat.triangles.push_back(tri);

	SceneData scene;
	pbrt_gpu::build(flat, scene);
	ASSERT_EQ(scene.materials.size(), 1u);
	EXPECT_EQ(scene.materials[0].type, MaterialType::Lambertian);
	ASSERT_GE(scene.materials[0].textureIdx, 0);
	const TextureData &tex = scene.textures[static_cast<std::size_t>(scene.materials[0].textureIdx)];
	EXPECT_EQ(tex.kind, TextureKind::Mix);
	EXPECT_FLOAT_EQ(tex.color1.x, 1.0f);
	EXPECT_FLOAT_EQ(tex.color2.z, 1.0f);
	EXPECT_FLOAT_EQ(tex.mixAmount, 0.25f);
}
