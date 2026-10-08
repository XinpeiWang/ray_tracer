#pragma once
// oidn_runtime.h -- Intel Open Image Denoise (OIDN, Apache-2.0) loaded at RUN time, so nothing links against it and a machine without it is not affected.
//
// Why: the Mac has no AI denoiser (the OptiX one is NVIDIA-only), so a Metal render at a modest sample count stays grainy. OIDN runs on Apple silicon's GPU
// (and on Intel Macs' CPUs) and is what Blender's Cycles uses off NVIDIA. Its library is ~50 MB (the network weights), so it is not bundled in the app: the GUI
// downloads it on request into the per-user folder (see docs/DENOISING.md), and this header finds it there.
//
// Where the library is looked for, in order:
//   1. $RT_OIDN_DIR (a folder with libOpenImageDenoise next to it or in lib/ below it) - tests and power users;
//   2. $RAY_TRACER_USER_ASSETS/denoiser/lib (what the GUI's installer writes: the lib/ folder of an OIDN release);
//   3. Homebrew's /opt/homebrew/lib and /usr/local/lib.
//
// Only the C API's handful of entry points used here are declared (the constants below are the ones in OIDN's oidn.h, which has kept them stable across 2.x), so
// building needs no OIDN headers. std-only apart from dlopen.

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace oidn_runtime {

namespace detail {

typedef struct OIDNDeviceImpl* Device;
typedef struct OIDNFilterImpl* Filter;
typedef struct OIDNBufferImpl* Buffer;

constexpr int kDeviceTypeDefault = 0;
constexpr int kFormatFloat3 = 3;
constexpr int kQualityHigh = 6;
constexpr int kErrorNone = 0;

struct Api {
	void* handle = nullptr;
	Device (*newDevice)(int) = nullptr;
	void (*commitDevice)(Device) = nullptr;
	int (*getDeviceError)(Device, const char**) = nullptr;
	void (*releaseDevice)(Device) = nullptr;
	Buffer (*newBuffer)(Device, size_t) = nullptr;
	void (*writeBuffer)(Buffer, size_t, size_t, const void*) = nullptr;
	void (*readBuffer)(Buffer, size_t, size_t, void*) = nullptr;
	void (*releaseBuffer)(Buffer) = nullptr;
	Filter (*newFilter)(Device, const char*) = nullptr;
	void (*setFilterImage)(Filter, const char*, Buffer, int, size_t, size_t, size_t, size_t, size_t) = nullptr;
	void (*setFilterBool)(Filter, const char*, bool) = nullptr;
	void (*setFilterInt)(Filter, const char*, int) = nullptr;
	void (*commitFilter)(Filter) = nullptr;
	void (*executeFilter)(Filter) = nullptr;
	void (*releaseFilter)(Filter) = nullptr;
	std::string path;     // the library that was loaded
	std::string error;    // why none was
};

inline bool fileExists(const std::string& p) {
	FILE* f = std::fopen(p.c_str(), "rb");
	if (!f) return false;
	std::fclose(f);
	return true;
}

// An environment variable's value, "" if unset. (_dupenv_s on Windows: MSVC rejects getenv in a project without _CRT_SECURE_NO_WARNINGS.)
inline std::string envValue(const char* name) {
#ifdef _WIN32
	char* value = nullptr;
	size_t length = 0;
	std::string out;
	if (_dupenv_s(&value, &length, name) == 0 && value) out = value;
	std::free(value);
	return out;
#else
	const char* value = std::getenv(name);
	return value ? std::string(value) : std::string();
#endif
}

inline const char* libraryName() {
#if defined(__APPLE__)
	return "libOpenImageDenoise.dylib";
#elif defined(_WIN32)
	return "OpenImageDenoise.dll";
#else
	return "libOpenImageDenoise.so";
#endif
}

// The candidate library files, best first.
inline std::vector<std::string> candidates() {
	std::vector<std::string> out;
	const std::string name = libraryName();
	auto addDir = [&](const std::string& dir) {
		if (dir.empty()) return;
		out.push_back(dir + "/" + name);
		out.push_back(dir + "/lib/" + name);
	};
	addDir(envValue("RT_OIDN_DIR"));
	if (const std::string u = envValue("RAY_TRACER_USER_ASSETS"); !u.empty()) addDir(u + "/denoiser");
	addDir("/opt/homebrew");
	addDir("/usr/local");
	return out;
}

inline Api& api() {
	static Api a = []() {
		Api x;
#if defined(_WIN32)
		x.error = "the denoiser is not wired up on Windows (the OptiX denoiser is used there)";
		return x;
#else
		for (const std::string& path : candidates()) {
			if (!fileExists(path)) continue;
			x.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
			if (!x.handle) {
				const char* why = dlerror();
				x.error = std::string("could not load ") + path + ": " + (why ? why : "unknown error");
				continue;
			}
			x.path = path;
			break;
		}
		if (!x.handle) {
			if (x.error.empty()) x.error = std::string("Open Image Denoise (") + libraryName() + ") was not found; install it from the Diagnostics tab or set RT_OIDN_DIR";
			return x;
		}
		auto sym = [&](const char* n) { return dlsym(x.handle, n); };
#define RT_OIDN_BIND(field, name) *reinterpret_cast<void**>(&x.field) = sym(name)
		RT_OIDN_BIND(newDevice, "oidnNewDevice");
		RT_OIDN_BIND(commitDevice, "oidnCommitDevice");
		RT_OIDN_BIND(getDeviceError, "oidnGetDeviceError");
		RT_OIDN_BIND(releaseDevice, "oidnReleaseDevice");
		RT_OIDN_BIND(newBuffer, "oidnNewBuffer");
		RT_OIDN_BIND(writeBuffer, "oidnWriteBuffer");
		RT_OIDN_BIND(readBuffer, "oidnReadBuffer");
		RT_OIDN_BIND(releaseBuffer, "oidnReleaseBuffer");
		RT_OIDN_BIND(newFilter, "oidnNewFilter");
		RT_OIDN_BIND(setFilterImage, "oidnSetFilterImage");
		RT_OIDN_BIND(setFilterBool, "oidnSetFilterBool");
		RT_OIDN_BIND(setFilterInt, "oidnSetFilterInt");
		RT_OIDN_BIND(commitFilter, "oidnCommitFilter");
		RT_OIDN_BIND(executeFilter, "oidnExecuteFilter");
		RT_OIDN_BIND(releaseFilter, "oidnReleaseFilter");
#undef RT_OIDN_BIND
		if (!x.newDevice || !x.commitDevice || !x.getDeviceError || !x.releaseDevice || !x.newBuffer || !x.writeBuffer || !x.readBuffer || !x.releaseBuffer ||
		    !x.newFilter || !x.setFilterImage || !x.setFilterBool || !x.setFilterInt || !x.commitFilter || !x.executeFilter || !x.releaseFilter) {
			x.error = std::string(x.path) + " is not an Open Image Denoise 2.x library (an entry point is missing)";
			dlclose(x.handle);
			x.handle = nullptr;
		}
		return x;
#endif
	}();
	return a;
}

}  // namespace detail

// True when an OIDN library was found and loaded.
inline bool available() { return detail::api().handle != nullptr; }
inline std::string libraryPath() { return detail::api().path; }
// Why it is not available (empty when it is).
inline std::string unavailableReason() { return available() ? std::string() : detail::api().error; }

// Denoises the HDR colour image `pixels` in place. `pixels` holds width*height pixels of `pixelStrideFloats` floats each (3 for RGB, 4 for RGBA: only the first three
// are read and written). `blend` in [0,1] mixes the result with the original (1 = fully denoised, the default). Returns false and says why in `error` when OIDN is
// missing or fails; `pixels` is then left unchanged.
inline bool denoiseHdr(float* pixels, int width, int height, int pixelStrideFloats, float blend, std::string& error) {
	using namespace detail;
	Api& a = api();
	if (!a.handle) { error = a.error; return false; }
	if (!pixels || width < 1 || height < 1 || (pixelStrideFloats != 3 && pixelStrideFloats != 4)) { error = "bad image arguments"; return false; }
	const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
	std::vector<float> color(count * 3), result(count * 3);
	for (size_t i = 0; i < count; ++i)
		for (int c = 0; c < 3; ++c) {
			const float v = pixels[i * pixelStrideFloats + c];
			color[i * 3 + c] = (v == v && v > 0.0f) ? v : 0.0f;   // OIDN wants finite, non-negative radiance
		}
	const size_t bytes = count * 3 * sizeof(float);

	Device device = a.newDevice(kDeviceTypeDefault);
	if (!device) { error = "could not create an Open Image Denoise device"; return false; }
	a.commitDevice(device);
	const char* message = nullptr;
	if (a.getDeviceError(device, &message) != kErrorNone) {
		error = std::string("Open Image Denoise: ") + (message ? message : "device error");
		a.releaseDevice(device);
		return false;
	}
	Buffer colorBuf = a.newBuffer(device, bytes), outBuf = a.newBuffer(device, bytes);
	Filter filter = a.newFilter(device, "RT");
	bool ok = colorBuf && outBuf && filter;
	if (ok) {
		a.writeBuffer(colorBuf, 0, bytes, color.data());
		const size_t pixelStride = 3 * sizeof(float), rowStride = pixelStride * static_cast<size_t>(width);
		a.setFilterImage(filter, "color", colorBuf, kFormatFloat3, static_cast<size_t>(width), static_cast<size_t>(height), 0, pixelStride, rowStride);
		a.setFilterImage(filter, "output", outBuf, kFormatFloat3, static_cast<size_t>(width), static_cast<size_t>(height), 0, pixelStride, rowStride);
		a.setFilterBool(filter, "hdr", true);
		a.setFilterInt(filter, "quality", kQualityHigh);
		a.commitFilter(filter);
		a.executeFilter(filter);
		if (a.getDeviceError(device, &message) != kErrorNone) {
			error = std::string("Open Image Denoise: ") + (message ? message : "error while denoising");
			ok = false;
		} else {
			a.readBuffer(outBuf, 0, bytes, result.data());
		}
	} else {
		error = "could not allocate Open Image Denoise buffers";
	}
	if (filter) a.releaseFilter(filter);
	if (colorBuf) a.releaseBuffer(colorBuf);
	if (outBuf) a.releaseBuffer(outBuf);
	a.releaseDevice(device);
	if (!ok) return false;

	const float t = blend < 0.0f ? 0.0f : (blend > 1.0f ? 1.0f : blend);
	for (size_t i = 0; i < count; ++i)
		for (int c = 0; c < 3; ++c) {
			float& p = pixels[i * pixelStrideFloats + c];
			p = p + t * (result[i * 3 + c] - p);
		}
	return true;
}

}  // namespace oidn_runtime
