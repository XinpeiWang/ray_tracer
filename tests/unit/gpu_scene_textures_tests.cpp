// gpu_scene_textures_tests.cpp - src/shared/gpu_scene_textures.h: the image decode, alpha-cutout mask and grayscale test every GPU backend shares.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>

#include "../../src/shared/gpu_scene_textures.h"

using namespace gpu_scene_textures;

namespace {

struct Rgb { unsigned char r, g, b; };

// A w x h 24-bit BMP whose pixel (x, y) (y = 0 at the top) is colourAt(x, y).
std::string bmp(int w, int h, const std::function<Rgb(int, int)>& colourAt) {
	std::string bytes;
	const auto u16 = [&](unsigned v) { bytes.push_back(char(v & 0xFF)); bytes.push_back(char((v >> 8) & 0xFF)); };
	const auto u32 = [&](unsigned v) { u16(v & 0xFFFF); u16(v >> 16); };
	const int rowBytes = (w * 3 + 3) & ~3;
	bytes += "BM";
	u32(14 + 40 + rowBytes * h); u32(0); u32(14 + 40);
	u32(40); u32(w); u32(h); u16(1); u16(24); u32(0); u32(rowBytes * h); u32(0); u32(0); u32(0); u32(0);
	for (int row = h - 1; row >= 0; --row) {   // bottom-up
		int written = 0;
		for (int x = 0; x < w; ++x) {
			const Rgb c = colourAt(x, row);
			bytes.push_back(char(c.b)); bytes.push_back(char(c.g)); bytes.push_back(char(c.r));
			written += 3;
		}
		while (written++ < rowBytes) bytes.push_back('\0');
	}
	return bytes;
}

class GpuSceneTexturesTest : public ::testing::Test {
protected:
	void SetUp() override {
		const char* tmp = std::getenv("TEMP");
		if (!tmp) tmp = std::getenv("TMPDIR");
		root_ = std::string(tmp ? tmp : ".") + "/gpu_scene_textures_tests/";
		std::filesystem::create_directories(root_);
	}
	void TearDown() override { std::filesystem::remove_all(root_); }
	std::string write(const std::string& name, const std::string& bytes) {
		const std::string path = root_ + name;
		std::ofstream(path, std::ios::binary) << bytes;
		return path;
	}
	std::string root_;
};

}  // namespace

TEST_F(GpuSceneTexturesTest, TheDefaultGammaKeepsTheFilesOwnSrgbBytes) {
	const std::string path = write("a.bmp", bmp(2, 2, [](int x, int y) { return Rgb{(unsigned char)(20 + x), (unsigned char)(100 + y), 200}; }));
	const DecodedImage img = decodeColourImage(path);
	ASSERT_TRUE(img.found);
	EXPECT_TRUE(img.srgb) << "the GPU decodes each texel; the bytes are not baked";
	EXPECT_EQ(img.width, 2);
	EXPECT_EQ(img.height, 2);
	ASSERT_EQ(img.pixels.size(), size_t(2 * 2 * 3));
	EXPECT_EQ(img.pixels[0], 20);    // pixel (0,0), top-left
	EXPECT_EQ(img.pixels[1], 100);
	EXPECT_EQ(img.pixels[2], 200);
	const unsigned char* bottomRight = &img.pixels[(1 * 2 + 1) * 3];
	EXPECT_EQ(bottomRight[0], 21);
	EXPECT_EQ(bottomRight[1], 101);
}

TEST_F(GpuSceneTexturesTest, AnotherGammaIsBakedIntoLinearBytes) {
	const std::string path = write("b.bmp", bmp(1, 1, [](int, int) { return Rgb{128, 128, 128}; }));
	const DecodedImage g22 = decodeColourImage(path, 2.2f, true);   // invert forces the baked path even at the default gamma
	ASSERT_TRUE(g22.found);
	EXPECT_FALSE(g22.srgb);
	const DecodedImage g1 = decodeColourImage(path, 1.0f);
	ASSERT_TRUE(g1.found);
	EXPECT_FALSE(g1.srgb);
	EXPECT_NEAR(g1.pixels[0], 128, 1) << "gamma 1 leaves the value (within the requantization)";
	const DecodedImage g3 = decodeColourImage(path, 3.0f);
	const int expected = int(256.0f * std::pow(128.0f / 255.0f, 3.0f));
	EXPECT_NEAR(g3.pixels[0], expected, 1) << "pow(c, gamma), then 8-bit linear";
	EXPECT_LT(g3.pixels[0], g1.pixels[0]);
}

TEST_F(GpuSceneTexturesTest, InvertFlipsTheQuantizedByte) {
	const std::string path = write("c.bmp", bmp(1, 1, [](int, int) { return Rgb{0, 100, 255}; }));
	const DecodedImage inv = decodeColourImage(path, 1.0f, true);
	ASSERT_TRUE(inv.found);
	EXPECT_EQ(inv.pixels[0], 255);
	EXPECT_NEAR(inv.pixels[1], 155, 1);
	EXPECT_EQ(inv.pixels[2], 0);
}

TEST_F(GpuSceneTexturesTest, AMissingOrBrokenFileIsNotFound) {
	EXPECT_FALSE(decodeColourImage(root_ + "nothing.png").found);
	EXPECT_FALSE(decodeAlphaMask(root_ + "nothing.png").found);
	const std::string junk = write("junk.bmp", "this is not an image");
	EXPECT_FALSE(decodeColourImage(junk).found);
	EXPECT_FALSE(decodeAlphaMask(junk).found);
}

TEST_F(GpuSceneTexturesTest, AlphaMaskIsTheSrgbDecodedMeanReplicatedInEveryChannel) {
	const std::string path = write("m.bmp", bmp(3, 1, [](int x, int) {
		if (x == 0) return Rgb{255, 255, 255};
		if (x == 1) return Rgb{153, 153, 153};   // an authored 0.6: sRGB-decoded it is about 0.32, below the cutout threshold
		return Rgb{255, 0, 0};                    // the mean of (1, 0, 0)
	}));
	const DecodedImage mask = decodeAlphaMask(path);
	ASSERT_TRUE(mask.found);
	EXPECT_FALSE(mask.srgb) << "already decoded";
	ASSERT_EQ(mask.pixels.size(), size_t(3 * 3));
	EXPECT_EQ(mask.pixels[0], 255);
	const int grey = int(srgb_decode::byteToLinear(153) * 255.0f + 0.5f);
	EXPECT_EQ(mask.pixels[3], grey);
	EXPECT_LT(mask.pixels[3], 128);
	EXPECT_EQ(mask.pixels[6], int((1.0f / 3.0f) * 255.0f + 0.5f));
	for (int p = 0; p < 3; ++p) {
		EXPECT_EQ(mask.pixels[p * 3 + 0], mask.pixels[p * 3 + 1]);
		EXPECT_EQ(mask.pixels[p * 3 + 1], mask.pixels[p * 3 + 2]);
	}
}

TEST(GpuSceneTexturesGrayscaleTest, TenLevelsOfChannelSpreadStillCountAsGray) {
	const unsigned char gray[3] = {100, 100, 100};
	const unsigned char slightly[3] = {100, 110, 100};
	const unsigned char tinted[3] = {100, 111, 100};
	EXPECT_TRUE(isGrayscaleRgb8(gray, 1, 1));
	EXPECT_TRUE(isGrayscaleRgb8(slightly, 1, 1));
	EXPECT_FALSE(isGrayscaleRgb8(tinted, 1, 1));
}

TEST(GpuSceneTexturesGrayscaleTest, ANormalMapIsNotGrayAndADegenerateImageIs) {
	std::vector<unsigned char> normalMap(16 * 16 * 3);
	for (size_t i = 0; i < normalMap.size(); i += 3) { normalMap[i] = 128; normalMap[i + 1] = 128; normalMap[i + 2] = 255; }   // flat tangent-space normal
	EXPECT_FALSE(isGrayscaleRgb8(normalMap.data(), 16, 16));
	EXPECT_TRUE(isGrayscaleRgb8(nullptr, 0, 0));
	EXPECT_TRUE(isGrayscaleRgb8(nullptr, 5, 0));
}

TEST(GpuSceneTexturesGrayscaleTest, OnlyTheEightByEightGridIsLookedAt) {
	std::vector<unsigned char> img(16 * 16 * 3, 50);
	const auto set = [&](int x, int y) { unsigned char* p = &img[(y * 16 + x) * 3]; p[0] = 255; p[1] = 0; p[2] = 0; };
	set(1, 1);   // grid samples sit at multiples of 2 for a 16-pixel side, so this pixel is never read
	EXPECT_TRUE(isGrayscaleRgb8(img.data(), 16, 16));
	set(2, 4);   // a grid point
	EXPECT_FALSE(isGrayscaleRgb8(img.data(), 16, 16));
}

TEST(GpuSceneTexturesGrayscaleTest, ACustomAccessorGetsXThenY) {
	int calls = 0;
	const bool gray = isGrayscaleImage(8, 4, [&](int x, int y) {
		EXPECT_LT(x, 8);
		EXPECT_LT(y, 4);
		++calls;
		static const unsigned char px[3] = {7, 7, 7};
		return px;
	});
	EXPECT_TRUE(gray);
	EXPECT_EQ(calls, 64);
}
