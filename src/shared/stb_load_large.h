#pragma once
// stb_load_large.h - stbi_loadf that still loads very large 8-bit images.
//
// stb_image sizes its buffers in `int`. Since v2.2x it refuses any image whose FLOAT buffer would exceed 2 GB (w * h * channels * 4 bytes), where
// the v2.06 this project used to vendor just allocated it. A real scene needs that: the Gallery environment scene's 16384 x 16384 JPEG is 3.2 GB
// as floats. Its 8-bit buffer (805 MB) is fine, so when the float load fails on a non-HDR image these decode 8-bit and convert to float exactly the
// way stbi__ldr_to_hdr does (pow(byte / 255, gamma) for colour channels, alpha linear), with no stb-side size limit on the float buffer. The result is
// malloc'd, so stbi_image_free() frees it.
//
// Callers that set stbi_ldr_to_hdr_gamma() must pass the same value as `ldrGamma` (stb's global is private to its translation unit).

#include "../external/stb_image.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// The conversion alone, from `bytes` decoded as 8-bit. Exposed for the tests, which compare it with stbi_loadf on a small image.
inline float *stbi_loadf_via_8bit_from_memory(const stbi_uc *buffer, int len, int *x, int *y, int *comp, int reqComp, float ldrGamma) {
	int w = 0, h = 0, c = 0;
	stbi_uc *bytes = stbi_load_from_memory(buffer, len, &w, &h, &c, reqComp);
	if (!bytes) return nullptr;
	const int n = reqComp ? reqComp : c;
	const std::size_t count = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * static_cast<std::size_t>(n);
	float *out = static_cast<float *>(std::malloc(count * sizeof(float)));
	if (!out) { stbi_image_free(bytes); return nullptr; }
	float lut[256];
	for (int i = 0; i < 256; ++i) lut[i] = std::pow(static_cast<float>(i) / 255.0f, ldrGamma);
	const int colourChannels = (n & 1) ? n : n - 1;   // an even channel count ends in alpha, which stays linear
	for (std::size_t p = 0; p < count; p += static_cast<std::size_t>(n)) {
		for (int k = 0; k < colourChannels; ++k) out[p + k] = lut[bytes[p + k]];
		if (colourChannels < n) out[p + n - 1] = static_cast<float>(bytes[p + n - 1]) / 255.0f;
	}
	stbi_image_free(bytes);
	if (x) *x = w;
	if (y) *y = h;
	if (comp) *comp = c;
	return out;
}

inline float *stbi_loadf_from_memory_large(const stbi_uc *buffer, int len, int *x, int *y, int *comp, int reqComp, float ldrGamma = 2.2f) {
	if (float *p = stbi_loadf_from_memory(buffer, len, x, y, comp, reqComp)) return p;
	if (stbi_is_hdr_from_memory(buffer, len)) return nullptr;   // a Radiance image has no 8-bit form to fall back to
	return stbi_loadf_via_8bit_from_memory(buffer, len, x, y, comp, reqComp, ldrGamma);
}

inline float *stbi_loadf_large(const char *filename, int *x, int *y, int *comp, int reqComp, float ldrGamma = 2.2f) {
	if (float *p = stbi_loadf(filename, x, y, comp, reqComp)) return p;
	std::FILE *f = std::fopen(filename, "rb");
	if (!f) return nullptr;
	std::vector<stbi_uc> bytes;
	unsigned char chunk[1 << 16];
	for (std::size_t got; (got = std::fread(chunk, 1, sizeof(chunk), f)) > 0;) bytes.insert(bytes.end(), chunk, chunk + got);
	std::fclose(f);
	if (bytes.empty() || bytes.size() > static_cast<std::size_t>(0x7fffffff)) return nullptr;
	if (stbi_is_hdr_from_memory(bytes.data(), static_cast<int>(bytes.size()))) return nullptr;
	return stbi_loadf_via_8bit_from_memory(bytes.data(), static_cast<int>(bytes.size()), x, y, comp, reqComp, ldrGamma);
}
