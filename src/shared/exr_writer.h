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

#include <cctype>
#include <string>

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
