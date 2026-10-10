#pragma once
// gpu_scene_textures.h -- how a GPU backend turns an image file into texture pixels: the 8-bit decode (sRGB bytes kept or gamma baked), the alpha-cutout mask and the
// "is this a height map or a normal map" test. GPU scene stage 1 (docs/GPU_SCENE_COMMON.md).
//
// These were inside the OptiX pbrt builder (pbrt_gpu_builder_materials.h), and the grayscale test was copied four times (CPU mesh_mtl.h, OptiX, Metal's displacement loader and an
// older OptiX loader). What stays per backend is everything about WHERE the pixels go: OptiX appends to one shared byte buffer and records an offset, Metal binds the first image
// that asks for a slot, and each keeps its own cache. Only the decisions are here, so the two cannot disagree about what a texel's value is.
//
// The decode functions read a file and need stb_image's implementation linked in (every executable here already has it). Nothing else is required.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include "../external/stb_image.h"
#include "srgb_decode.h"

namespace gpu_scene_textures {

// pbrt-v4's default 8-bit encoding is sRGB; a "gamma" other than this value asks for a power curve instead (Texture "imagemap" `string encoding "gamma 1.8"`).
constexpr float kDefaultImagemapGamma = 2.2f;

// Three bytes per pixel, row-major. `srgb` says the bytes are the file's own sRGB-encoded values (the GPU decodes each texel exactly when it looks it up); false means they
// are already linear, with the gamma (and invert) baked in.
struct DecodedImage {
	bool found = false;
	bool srgb = false;
	int width = 0, height = 0;
	std::vector<unsigned char> pixels;
};

// A colour image (reflectance, transmittance, roughness...). The default gamma without invert keeps the file's bytes as they are; any other gamma, invert, or an HDR
// source goes through stb's float decode and is requantized to linear bytes (invert applied after the quantization, as the CPU's mipmap does, so both land on the same byte).
inline DecodedImage decodeColourImage(const std::string& path, float gamma = kDefaultImagemapGamma, bool invert = false) {
	DecodedImage entry;
	int width = 0, height = 0, channels = 0;
	if (gamma == kDefaultImagemapGamma && !invert && !stbi_is_hdr(path.c_str())) {
		unsigned char* raw = stbi_load(path.c_str(), &width, &height, &channels, 3);
		if (raw) {
			entry.width = width;
			entry.height = height;
			entry.pixels.assign(raw, raw + static_cast<std::size_t>(width) * height * 3);
			entry.srgb = true;
			entry.found = true;
			stbi_image_free(raw);
			return entry;
		}
	}
	// stbi_ldr_to_hdr_gamma is process-global state: set it for this one load and put it back.
	static std::mutex gammaMutex;
	float* fdata = nullptr;
	{
		std::lock_guard<std::mutex> gammaLock(gammaMutex);
		stbi_ldr_to_hdr_gamma(gamma);
		fdata = stbi_loadf(path.c_str(), &width, &height, &channels, 3);
		stbi_ldr_to_hdr_gamma(kDefaultImagemapGamma);
	}
	if (!fdata) return entry;
	entry.found = true;
	entry.width = width;
	entry.height = height;
	const std::size_t total = static_cast<std::size_t>(width) * height * 3;
	entry.pixels.resize(total);
	for (std::size_t i = 0; i < total; ++i) {
		const float v = fdata[i];
		unsigned char q = (v <= 0.0f) ? 0 : (v >= 1.0f ? 255 : static_cast<unsigned char>(256.0f * v));
		if (invert) {
			const float reconstructed = q / 255.0f;
			const float inverted = std::fmax(0.0f, 1.0f - reconstructed);
			q = (inverted <= 0.0f) ? 0 : (inverted >= 1.0f ? 255 : static_cast<unsigned char>(256.0f * inverted));
		}
		entry.pixels[i] = q;
	}
	stbi_image_free(fdata);
	return entry;
}

// The decode this follows (stb's float load) turns an 8-bit picture into linear values with gamma 2.2. A Texture "imagemap" with another "encoding" ("linear", "gamma 1.8") or
// "invert" wants its bytes read differently: this redoes the curve from stb's result (v = byte^2.2 -> byte^gamma) and inverts. A float picture (.exr, .hdr) is linear already.
// Found by scripts/consistency_sweep.py: a reflectance image with encoding "linear" was drawn 12% darker on Metal than on the CPU.
inline void applyImagemapEncoding(std::vector<float>& pixels, const std::string& filename, double gamma, bool invert) {
    std::string lower = filename;
    for (char& c : lower) c = (char)tolower((unsigned char)c);
    const bool floatPicture = lower.size() >= 4 && (lower.compare(lower.size() - 4, 4, ".exr") == 0 || lower.compare(lower.size() - 4, 4, ".hdr") == 0 || lower.compare(lower.size() - 4, 4, ".pic") == 0);
    if (floatPicture) return;
    const float exponent = (float)(gamma / 2.2);
    for (float& v : pixels) {
        if (exponent != 1.0f) v = powf(std::max(v, 0.0f), exponent);
        if (invert) v = std::max(0.0f, 1.0f - v);
    }
}

// A Shape "alpha" cutout mask: pbrt reads it as the mean of the channels, sRGB-decoded, so the result holds that value replicated into R, G and B (a consumer that reads the
// red channel sees pbrt's number). It is never gamma-decoded the way colour is: a coverage fraction is not a display colour.
inline DecodedImage decodeAlphaMask(const std::string& path) {
	DecodedImage entry;
	int width = 0, height = 0, channels = 0;
	unsigned char* bytes = stbi_load(path.c_str(), &width, &height, &channels, 3);
	if (!bytes) return entry;
	entry.found = true;
	entry.srgb = false;   // already decoded
	entry.width = width;
	entry.height = height;
	entry.pixels = srgb_decode::alphaMaskFromRgb8(bytes, static_cast<std::size_t>(width) * height);
	stbi_image_free(bytes);
	return entry;
}

// Whether an image is a grayscale height map (true) or a colour image such as a tangent-space normal map (false): an 8x8 grid of samples, grayscale when no sample's channels
// differ by more than 10. `pixelAt(x, y)` returns a pointer to that pixel's first byte (RGB order); a degenerate image counts as grayscale.
template <class PixelAt>
inline bool isGrayscaleImage(int width, int height, PixelAt pixelAt) {
	if (width <= 0 || height <= 0) return true;
	constexpr int kGrid = 8;
	int maxDiff = 0;
	for (int sy = 0; sy < kGrid; ++sy) {
		const int y = (sy * height) / kGrid;
		for (int sx = 0; sx < kGrid; ++sx) {
			const int x = (sx * width) / kGrid;
			const unsigned char* p = pixelAt(x, y);
			const int r = p[0], g = p[1], b = p[2];
			maxDiff = std::max({maxDiff, std::abs(r - g), std::abs(g - b), std::abs(r - b)});
		}
	}
	return maxDiff <= 10;
}

// The same test on a tightly packed RGB8 buffer.
inline bool isGrayscaleRgb8(const unsigned char* rgb, int width, int height) {
	return isGrayscaleImage(width, height, [rgb, width](int x, int y) {
		return rgb + (static_cast<std::size_t>(y) * width + x) * 3;
	});
}

}  // namespace gpu_scene_textures
