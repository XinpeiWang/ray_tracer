#pragma once
// exr_writer.h -- thin wrapper around tinyexr's SaveEXR, shared by both
// render backends (CPU: src/TheRestOfYourLife/camera.h; GPU: gpu/optix/
// optix_interface.cpp). tinyexr's write API (SaveEXR) has been linked into
// the project since src/external/tinyexr_impl.cpp first vendored the
// library - see that file's TINYEXR_IMPLEMENTATION define - but until this
// header, only its read side (LoadEXRFromMemory, for infinite-light
// textures - pbrt_load.h) was ever actually called.
//
// Interleaved RGB float, full precision (not fp16): matches the linear
// pre-tonemap radiance both callers already hold, and this project's own
// EXR reader (pbrt_load.h) decodes into the same layout, so a render's own
// EXR output round-trips through it unchanged if ever fed back in as a
// texture.

#include "../external/tinyexr.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// Case-insensitive ".exr" extension check, shared by every render entry
// point that decides EXR-vs-PPM/PNG output by sniffing output_path (CPU/GPU
// recursive, BDPT/MLT/SPPM, and the launcher's post-render step) so the
// four independent copies of this check can't drift out of sync.
inline bool is_exr_output_path(const std::string &path) {
	if (path.size() < 4) return false;
	std::string ext = path.substr(path.size() - 4);
	for (char &c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	return ext == ".exr";
}

inline bool write_exr_image(const std::string &path, const float *rgb,
							int width, int height, std::string &error) {
	const char *err = nullptr;
	const int ret = SaveEXR(rgb, width, height, /*components=*/3,
							 /*save_as_fp16=*/0, path.c_str(), &err);
	if (ret != TINYEXR_SUCCESS) {
		if (err) {
			error = err;
			FreeEXRErrorMessage(err);
		} else {
			error = "SaveEXR failed with code " + std::to_string(ret);
		}
		return false;
	}
	return true;
}

// ---- Multi-channel (AOV / multilayer) EXR -------------------------------------------------------------------------------------------------------------------
// One float plane per named channel ("R", "G", "B", "A", "normal.X", "depth.Z", ...), full precision. The channels are written sorted by name, as the EXR format
// requires. `planes[i]` holds width*height floats for `names[i]`.
inline bool write_exr_channels(const std::string &path, int width, int height, const std::vector<std::string> &names,
							   const std::vector<std::vector<float>> &planes, std::string &error) {
	const int n = static_cast<int>(names.size());
	if (n == 0 || planes.size() != names.size()) { error = "no channels"; return false; }
	for (const auto &p : planes)
		if (p.size() != static_cast<size_t>(width) * static_cast<size_t>(height)) { error = "a channel does not match the image size"; return false; }
	std::vector<int> order(n);
	std::iota(order.begin(), order.end(), 0);
	std::sort(order.begin(), order.end(), [&](int a, int b) { return names[a] < names[b]; });

	EXRHeader header;
	InitEXRHeader(&header);
	EXRImage image;
	InitEXRImage(&image);
	image.num_channels = n;
	std::vector<float *> ptrs(n);   // tinyexr's image pointers are not const; SaveEXRImageToFile only reads them
	for (int i = 0; i < n; ++i) ptrs[i] = const_cast<float *>(planes[order[i]].data());
	image.images = reinterpret_cast<unsigned char **>(ptrs.data());
	image.width = width;
	image.height = height;
	header.num_channels = n;
	header.channels = static_cast<EXRChannelInfo *>(std::malloc(sizeof(EXRChannelInfo) * static_cast<size_t>(n)));
	header.pixel_types = static_cast<int *>(std::malloc(sizeof(int) * static_cast<size_t>(n)));
	header.requested_pixel_types = static_cast<int *>(std::malloc(sizeof(int) * static_cast<size_t>(n)));
	for (int i = 0; i < n; ++i) {
		std::memset(header.channels[i].name, 0, sizeof(header.channels[i].name));
		{   // (a bounded memcpy: MSVC rejects strncpy as unsafe, and the name was zeroed above)
			const std::string &channelName = names[order[i]];
			const size_t room = sizeof(header.channels[i].name) - 1;
			std::memcpy(header.channels[i].name, channelName.c_str(), channelName.size() < room ? channelName.size() : room);
		}
		header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
		header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
	}
	const char *err = nullptr;
	const int ret = SaveEXRImageToFile(&image, &header, path.c_str(), &err);
	std::free(header.channels);
	std::free(header.pixel_types);
	std::free(header.requested_pixel_types);
	if (ret != TINYEXR_SUCCESS) {
		error = err ? err : "SaveEXRImageToFile failed with code " + std::to_string(ret);
		if (err) FreeEXRErrorMessage(err);
		return false;
	}
	return true;
}

// Reads every channel of a scanline EXR as a float plane (half and uint channels are converted).
inline bool read_exr_channels(const std::string &path, int &width, int &height, std::vector<std::string> &names,
							  std::vector<std::vector<float>> &planes, std::string &error) {
	EXRVersion version;
	if (ParseEXRVersionFromFile(&version, path.c_str()) != TINYEXR_SUCCESS || version.multipart || version.tiled) {
		error = "not a single-part scanline EXR file: " + path;
		return false;
	}
	EXRHeader header;
	InitEXRHeader(&header);
	const char *err = nullptr;
	if (ParseEXRHeaderFromFile(&header, &version, path.c_str(), &err) != TINYEXR_SUCCESS) {
		error = err ? err : "could not read the EXR header";
		if (err) FreeEXRErrorMessage(err);
		return false;
	}
	for (int i = 0; i < header.num_channels; ++i) header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
	EXRImage image;
	InitEXRImage(&image);
	if (LoadEXRImageFromFile(&image, &header, path.c_str(), &err) != TINYEXR_SUCCESS) {
		error = err ? err : "could not read the EXR pixels";
		if (err) FreeEXRErrorMessage(err);
		FreeEXRHeader(&header);
		return false;
	}
	width = image.width;
	height = image.height;
	names.clear();
	planes.clear();
	for (int c = 0; c < header.num_channels; ++c) {
		names.push_back(header.channels[c].name);
		const float *src = reinterpret_cast<const float *>(image.images[c]);
		planes.emplace_back(src, src + static_cast<size_t>(width) * static_cast<size_t>(height));
	}
	FreeEXRImage(&image);
	FreeEXRHeader(&header);
	return true;
}
