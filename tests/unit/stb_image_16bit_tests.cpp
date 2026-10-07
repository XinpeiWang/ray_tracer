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
#include "../../src/shared/stb_load_large.h"

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

// --- stb_load_large.h: the fallback for images whose float buffer exceeds stb's 2 GB limit -------------------------------------------------
// The limit itself cannot be reached in a unit test (it needs a 3 GB buffer), so this checks what the fallback computes: decoding 8-bit and
// converting must equal what stbi_loadf produces for the same bytes, for every channel count and gamma the call sites use.

namespace {
void expectFallbackMatchesStbiLoadf(int reqComp, float gamma) {
	const std::string png = make16BitPng(3, 2);
	const auto *bytes = reinterpret_cast<const unsigned char *>(png.data());
	const int len = static_cast<int>(png.size());
	stbi_ldr_to_hdr_gamma(gamma);
	int w = 0, h = 0, c = 0;
	float *expected = stbi_loadf_from_memory(bytes, len, &w, &h, &c, reqComp);
	stbi_ldr_to_hdr_gamma(2.2f);   // stb's default; the global must not leak into other tests
	ASSERT_NE(expected, nullptr) << stbi_failure_reason();
	int w2 = 0, h2 = 0, c2 = 0;
	float *got = stbi_loadf_via_8bit_from_memory(bytes, len, &w2, &h2, &c2, reqComp, gamma);
	ASSERT_NE(got, nullptr);
	EXPECT_EQ(w2, w);
	EXPECT_EQ(h2, h);
	EXPECT_EQ(c2, c);
	for (int i = 0; i < w * h * reqComp; ++i) EXPECT_NEAR(got[i], expected[i], 1e-6f) << "value " << i << " (channels " << reqComp << ", gamma " << gamma << ")";
	stbi_image_free(expected);
	stbi_image_free(got);
}
}  // namespace

TEST(StbLoadLargeTest, TheEightBitFallbackMatchesStbiLoadfForRgbAtStbsDefaultGamma) { expectFallbackMatchesStbiLoadf(3, 2.2f); }
TEST(StbLoadLargeTest, TheEightBitFallbackMatchesStbiLoadfForRgbWithGammaOne) { expectFallbackMatchesStbiLoadf(3, 1.0f); }
TEST(StbLoadLargeTest, TheEightBitFallbackKeepsAlphaLinear) { expectFallbackMatchesStbiLoadf(4, 2.2f); }

TEST(StbLoadLargeTest, ThePublicEntryPointsLoadWhenStbCanAndRefuseWhatIsNotAnImage) {
	const std::string png = make16BitPng(3, 2);
	int w = 0, h = 0, c = 0;
	float *p = stbi_loadf_from_memory_large(reinterpret_cast<const unsigned char *>(png.data()), static_cast<int>(png.size()), &w, &h, &c, 3);
	ASSERT_NE(p, nullptr);
	EXPECT_EQ(w, 3);
	stbi_image_free(p);
	const unsigned char junk[16] = {1, 2, 3};
	EXPECT_EQ(stbi_loadf_from_memory_large(junk, sizeof(junk), &w, &h, &c, 3), nullptr);
	EXPECT_EQ(stbi_loadf_large("/nonexistent/definitely-not-here.png", &w, &h, &c, 3), nullptr);
}
