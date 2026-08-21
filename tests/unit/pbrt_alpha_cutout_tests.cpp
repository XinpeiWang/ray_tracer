/**
 * @file pbrt_alpha_cutout_tests.cpp
 * @brief End-to-end tests for pbrt Shape "alpha" cutout masks (Round 4
 *        Phase 3) - barcelona-pavilion's foliage is exactly this: each leaf
 *        Shape "plymesh" gives its own "texture alpha".
 *
 * Unlike pbrt_cpu_builder_tests.cpp (which builds from in-memory scene text
 * only), an alpha-cutout ray test needs a REAL decodable image on disk -
 * Material::alphaTextureFilename is only ever populated by pbrt_load.h's
 * post-flatten resolution pass, and triangle::hit()'s alpha test needs the
 * image's actual decoded pixels, not just a filename string. So this goes
 * through the full pbrt_load::loadFile() -> pbrt_cpu::build() chain against
 * real temp files, the same "these tests write real files" rationale
 * pbrt_load_tests.cpp's own file comment already gives.
 */

#include <gtest/gtest.h>

#include "pbrt_cpu_builder.h"
#include "pbrt_gpu_builder.h"
#include "../../src/shared/pbrt_load.h"

#include <cstdio>
#include <fstream>
#include <string>

namespace {

// A minimal uncompressed 24bpp BMP, 1x1, solid colour - stb_image decodes
// BMP with no extra build configuration (src/external/stb_image_impl.cpp
// defines no STBI_NO_*/STBI_ONLY_* restriction macros). One pixel is enough:
// triangle::hit()'s alpha test only cares whether the sampled value crosses
// kAlphaCutoutThreshold (0.5), not spatial detail - a uniform image removes
// any dependency on pbrt trianglemesh's own barycentric-fallback UV (see
// docs/PBRT_SUPPORT.md's UV-threading gap note) picking a particular point.
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
	// File header (14 bytes)
	bytes.push_back('B'); bytes.push_back('M');
	u32(14 + 40 + 4);   // file size: headers + one padded pixel row
	u32(0);             // reserved
	u32(14 + 40);       // pixel data offset
	// DIB header (BITMAPINFOHEADER, 40 bytes)
	u32(40);            // header size
	u32(1);             // width
	u32(1);             // height
	u16(1);             // planes
	u16(24);            // bits per pixel
	u32(0);             // no compression
	u32(4);             // image data size (one padded row)
	u32(0); u32(0);     // pixels per meter x/y
	u32(0); u32(0);     // colors used / important
	// Pixel data: BGR + 1 padding byte (rows padded to a 4-byte boundary)
	bytes.push_back(static_cast<char>(b));
	bytes.push_back(static_cast<char>(g));
	bytes.push_back(static_cast<char>(r));
	bytes.push_back('\0');
	return bytes;
}

class AlphaCutoutTempTree : public ::testing::Test {
protected:
	void SetUp() override {
		const char *tmp = std::getenv("TEMP");
		root_ = std::string(tmp ? tmp : ".") + "/pbrt_alpha_cutout_tests/";
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

// A unit quad (two triangles) in the z=0 plane spanning (0,0)..(1,1), with
// its "alpha" bound to a Texture named by the caller.
std::string quadSceneWithAlpha(const std::string &textureName) {
	return "Texture \"" + textureName + "\" \"float\" \"imagemap\" "
			"\"string filename\" [ \"mask.bmp\" ]\n"
		   "Material \"diffuse\" \"rgb reflectance\" [ .5 .5 .5 ]\n"
		   "Shape \"trianglemesh\" \"texture alpha\" [ \"" + textureName + "\" ]\n"
		   "  \"integer indices\" [ 0 1 2  0 2 3 ]\n"
		   "  \"point3 P\" [ 0 0 0  1 0 0  1 1 0  0 1 0 ]\n";
}

} // namespace

TEST(AlphaCutoutBundledSceneTest, BarcelonaPavilionFoliageResolvesFiveAlphaTextures) {
	// The actual motivating scene (Round 4 Phase 3's own task description) -
	// 5 foliage materials (Betula pendula / Tilia tomentosa leaves) each bind
	// "texture alpha" to an imagemap sharing their colour photo, pbrt reading
	// its red channel as an alpha-cutout mask (mirrors OBJ/MTL's own map_d
	// convention - see getOrBuildPbrtImageTexture's/alphaMaskFor's own
	// comments for why only the red channel is sampled). Both pbrt_load.h's
	// resolution pass and the GPU builder's decode-and-upload path are
	// exercised end to end against the real bundled files, not synthetic
	// ones - the two are otherwise covered by AlphaCutoutTempTree below and
	// PbrtGpuTextureTest's own alpha tests.
	const pbrt_load::LoadResult r = pbrt_load::loadFile(
		"pbrt_scenes/barcelona-pavilion/pavilion-day.pbrt");
	ASSERT_TRUE(r.ok) << r.error;

	int withAlpha = 0;
	for (const auto &m : r.scene.materials)
		if (!m.alphaTextureFilename.empty()) withAlpha++;
	EXPECT_EQ(withAlpha, 5);

	SceneData scene;
	pbrt_gpu::build(r.scene, scene);
	int gpuWithAlpha = 0;
	for (const auto &m : scene.materials) if (m.alphaMaskTexIdx >= 0) gpuWithAlpha++;
	EXPECT_EQ(gpuWithAlpha, 5);
	EXPECT_EQ(scene.textures.size(), 5u);
	EXPECT_FALSE(scene.texturePixels.empty());
}

TEST_F(AlphaCutoutTempTree, AWhiteAlphaMaskLeavesTheShapeFullyHittable) {
	write("scene.pbrt", quadSceneWithAlpha("mask"));
	write("mask.bmp", solidBmp1x1(255, 255, 255));
	const pbrt_load::LoadResult loaded = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(loaded.ok) << loaded.error;
	ASSERT_EQ(loaded.scene.materials.size(), 1u);
	ASSERT_EQ(loaded.scene.materials[0].alphaTextureFilename, path("mask.bmp"));

	const pbrt_cpu::BuildResult b = pbrt_cpu::build(loaded.scene);
	hit_record rec;
	EXPECT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec))
		<< "an alpha value of 1.0 (fully opaque) must not cut the shape out";
}

TEST_F(AlphaCutoutTempTree, ABlackAlphaMaskCutsTheShapeOutEntirely) {
	write("scene.pbrt", quadSceneWithAlpha("mask"));
	write("mask.bmp", solidBmp1x1(0, 0, 0));
	const pbrt_load::LoadResult loaded = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(loaded.ok) << loaded.error;

	const pbrt_cpu::BuildResult b = pbrt_cpu::build(loaded.scene);
	hit_record rec;
	EXPECT_FALSE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							  interval(0.001, infinity), rec))
		<< "an alpha value of 0.0 (fully transparent) must cut the whole shape "
		   "out, exactly like OBJ/MTL's own map_d";
}

TEST_F(AlphaCutoutTempTree, AShapeWithNoAlphaParameterIsUnaffected) {
	write("scene.pbrt",
		  "Material \"diffuse\" \"rgb reflectance\" [ .5 .5 .5 ]\n"
		  "Shape \"trianglemesh\" \"integer indices\" [ 0 1 2  0 2 3 ]\n"
		  "  \"point3 P\" [ 0 0 0  1 0 0  1 1 0  0 1 0 ]\n");
	const pbrt_load::LoadResult loaded = pbrt_load::loadFile(path("scene.pbrt"));
	ASSERT_TRUE(loaded.ok) << loaded.error;

	const pbrt_cpu::BuildResult b = pbrt_cpu::build(loaded.scene);
	hit_record rec;
	EXPECT_TRUE(b.world->hit(ray(point3(0.5, 0.5, -5), vec3(0, 0, 1)),
							 interval(0.001, infinity), rec));
}
