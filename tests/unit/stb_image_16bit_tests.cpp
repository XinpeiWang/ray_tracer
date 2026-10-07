/**
 * @file stb_image_16bit_tests.cpp
 * @brief The bundled stb_image must decode 16-bit-per-channel PNGs.
 *
 * The vendored stb_image.h was v2.06, which refuses them ("PNG not supported: 1/2/4/8-bit only"). Real scene assets use them - pbrt-v4-scenes'
 * zero-day folder alone has about a hundred - and every renderer loads textures through this header, so such a texture silently fell back to a
 * constant colour (or a missing displacement map) on the CPU, OptiX and Metal paths alike. This builds a tiny 16-bit PNG in memory (no file
 * fixture) and checks both the 8-bit and the float decode paths the renderers use.
 */

#include <gtest/gtest.h>

#include "../../src/external/stb_image.h"

#define MINIZ_NO_ZLIB_COMPATIBLE_NAMES
#include "../../src/external/miniz.h"

#include <cstdint>
#include <string>
#include <vector>

namespace {

void putBE32(std::string &s, uint32_t v) { for (int i = 3; i >= 0; --i) s.push_back(static_cast<char>((v >> (8 * i)) & 0xff)); }

void appendChunk(std::string &png, const char type[5], const std::string &data) {
	putBE32(png, static_cast<uint32_t>(data.size()));
	std::string body = std::string(type, 4) + data;
	png += body;
	putBE32(png, static_cast<uint32_t>(mz_crc32(MZ_CRC32_INIT, reinterpret_cast<const mz_uint8 *>(body.data()), body.size())));
}

// A 3x2 RGB PNG, 16 bits per channel, filter type 0 on every row. Channel value of pixel p, channel c is (1 + 7p + c) * 0x0505.
std::string make16BitPng(int width, int height) {
	std::string raw;
	for (int y = 0; y < height; ++y) {
		raw.push_back(0);   // filter: none
		for (int x = 0; x < width; ++x)
			for (int c = 0; c < 3; ++c) {
				const uint32_t v = (1 + 7 * (y * width + x) + c) * 0x0505;
				raw.push_back(static_cast<char>(v >> 8));
				raw.push_back(static_cast<char>(v & 0xff));
			}
	}
	mz_ulong zlen = mz_compressBound(static_cast<mz_ulong>(raw.size()));
	std::vector<unsigned char> z(zlen);
	EXPECT_EQ(mz_compress(z.data(), &zlen, reinterpret_cast<const unsigned char *>(raw.data()), static_cast<mz_ulong>(raw.size())), MZ_OK);
	std::string ihdr;
	putBE32(ihdr, static_cast<uint32_t>(width));
	putBE32(ihdr, static_cast<uint32_t>(height));
	ihdr += std::string({16, 2, 0, 0, 0});   // bit depth 16, colour type 2 (RGB), deflate, adaptive filtering, no interlace
	std::string png = "\x89PNG\r\n\x1a\n";
	appendChunk(png, "IHDR", ihdr);
	appendChunk(png, "IDAT", std::string(reinterpret_cast<const char *>(z.data()), zlen));
	appendChunk(png, "IEND", "");
	return png;
}

}  // namespace

TEST(StbImage16BitTest, A16BitPngDecodesTo8BitByKeepingTheHighByte) {
	const std::string png = make16BitPng(3, 2);
	int w = 0, h = 0, ch = 0;
	unsigned char *px = stbi_load_from_memory(reinterpret_cast<const unsigned char *>(png.data()), static_cast<int>(png.size()), &w, &h, &ch, 3);
	ASSERT_NE(px, nullptr) << stbi_failure_reason();
	EXPECT_EQ(w, 3);
	EXPECT_EQ(h, 2);
	for (int p = 0; p < 6; ++p)
		for (int c = 0; c < 3; ++c)
			EXPECT_EQ(px[p * 3 + c], ((1 + 7 * p + c) * 0x0505) >> 8) << "pixel " << p << " channel " << c;
	stbi_image_free(px);
}

TEST(StbImage16BitTest, A16BitPngDecodesToFloatLikeTheTextureLoaderExpects) {
	const std::string png = make16BitPng(3, 2);
	int w = 0, h = 0, ch = 0;
	float *px = stbi_loadf_from_memory(reinterpret_cast<const unsigned char *>(png.data()), static_cast<int>(png.size()), &w, &h, &ch, 3);
	ASSERT_NE(px, nullptr) << stbi_failure_reason();
	ASSERT_EQ(w, 3);
	ASSERT_EQ(h, 2);
	// Values are the 16-bit samples / 65535, through stb's gamma step (stbi_loadf applies 2.2 to LDR input): well inside (0, 1) and increasing.
	for (int p = 1; p < 6; ++p) EXPECT_GT(px[p * 3], px[(p - 1) * 3]) << "pixel " << p;
	EXPECT_GT(px[0], 0.0f);
	EXPECT_LT(px[17], 1.0f);
	stbi_image_free(px);
}
