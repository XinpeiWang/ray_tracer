#pragma once
// srgb_decode.h -- the exact sRGB transfer function and the helpers built on it.
//
// pbrt-v4 decodes an 8-bit image with an exact sRGB lookup table (util/color.h, SRGB8ToLinear):
// linear = c/12.92 below 0.04045, ((c+0.055)/1.055)^2.4 above. This codebase decoded with pow(c, 2.2)
// (stb_image's own gamma hook), which is about 2x off in the darks (byte 20: 0.0070 exact, 0.0037 with
// 2.2) and, because the result was then requantized to 8-bit linear, crushed dark texels to zero.
// Plain C++ so the CPU image loader, the CPU/GPU pbrt builders and the tests share one definition.

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace srgb_decode {

// Exact IEC 61966-2-1 decode of a [0,1] encoded value.
inline float toLinear(float c) {
	if (c <= 0.0f) return 0.0f;
	if (c >= 1.0f) return 1.0f;
	return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

// 256-entry byte -> linear table, built once.
inline const std::array<float, 256> &byteTable() {
	static const std::array<float, 256> table = [] {
		std::array<float, 256> t{};
		for (int i = 0; i < 256; ++i) t[static_cast<std::size_t>(i)] = toLinear(static_cast<float>(i) / 255.0f);
		return t;
	}();
	return table;
}

inline float byteToLinear(unsigned char b) { return byteTable()[b]; }

// pbrt-v4 reads a float imagemap (an alpha-cutout mask, a bump height) from a 3-channel image as the mean
// of its channels, each sRGB-decoded first when the image is 8-bit with the default encoding
// (util/mipmap.cpp:396-405, textures.cpp:436). Returns pixels*3 bytes, the decoded mean replicated into
// R, G and B, so a consumer that reads the red channel (our alpha test) sees pbrt's value. The 8-bit
// requantization is harmless for a mask that is only ever compared against a threshold or a hash.
inline std::vector<unsigned char> alphaMaskFromRgb8(const unsigned char *rgb, std::size_t pixels, bool srgb = true) {
	std::vector<unsigned char> out(pixels * 3);
	for (std::size_t i = 0; i < pixels; ++i) {
		float sum = 0.0f;
		for (int k = 0; k < 3; ++k)
			sum += srgb ? byteToLinear(rgb[i * 3 + k]) : static_cast<float>(rgb[i * 3 + k]) / 255.0f;
		float v = sum / 3.0f;
		v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
		const unsigned char q = static_cast<unsigned char>(v * 255.0f + 0.5f);
		out[i * 3 + 0] = out[i * 3 + 1] = out[i * 3 + 2] = q;
	}
	return out;
}

} // namespace srgb_decode
