// Stand-in for gpu/optix/optix_interface.h on builds with no CUDA/OptiX SDK
// (the new CMake-based macOS/CPU-only build - see root CMakeLists.txt and
// launcher/main.cpp's RT_HAVE_OPTIX guard). Windows MSBuild always defines
// RT_HAVE_OPTIX and includes the real header instead; this file is never
// compiled into that build.
//
// Every function here mirrors optix_interface.h's exact signature so main.cpp
// needs no call-site changes. optix_is_available() returning false is the
// only behavior that matters: main.cpp's existing GPU dispatch already
// treats "OptiX unavailable" as a fully-handled state at every call site
// (see optix_is_available()'s own doc comment in optix_interface.h), so none
// of the other stubs below are ever actually invoked - they exist only to
// satisfy the linker/type-checker, and return obviously-invalid values
// (false/-1/empty/no-op) so a mistaken call fails loudly rather than
// pretending to succeed.

#pragma once

#include <algorithm>
#include <string>

// RenderOptions has no CUDA/OptiX dependency at all (just doubles/bools/
// const char*), so including it directly here doesn't compromise this
// file's "no CUDA/OptiX SDK required" purpose the way including the real
// optix_interface.h would - unlike OptixDiagnostics below, it doesn't
// need a locally-duplicated mirror.
#include "../src/shared/render_options.h"

// Mirrors optix_interface.h's OptixDiagnostics exactly (same field layout) -
// see that file's own comment for why this is a plain-POD struct rather than
// a shared header: this file stands in for optix_interface.h wholesale, so
// it must be self-contained, same as every function below.
struct OptixDiagnostics {
	bool available;
	char device_name[256];
	int  cuda_driver_version;
	int  cuda_runtime_version;
	int  optix_abi_version;
	unsigned long long vram_free_bytes;
	unsigned long long vram_total_bytes;
	char failure_reason[256];
};

#ifdef __cplusplus
extern "C" {
#endif

inline bool optix_is_available() { return false; }

inline bool optix_get_diagnostics(OptixDiagnostics* out) {
	if (!out) return false;
	*out = OptixDiagnostics{};
	out->available = false;
	// std::string::copy instead of strncpy - avoids MSVC's C4996 "unsafe
	// function" flag (this header can be compiled under MSVC too: the root
	// CMakeLists.txt's CPU-only ray_tracer target builds it there whenever
	// RT_BUILD_GPU is off, not just on macOS/Linux).
	std::string reason("Built without RT_HAVE_OPTIX (no CUDA/OptiX SDK at build time)");
	size_t n = (std::min)(reason.size(), sizeof(out->failure_reason) - 1);
	reason.copy(out->failure_reason, n);
	out->failure_reason[n] = '\0';
	return false;
}

inline int optix_render_main(
	int /*image_width*/,
	int /*image_height*/,
	int /*samples_per_pixel*/,
	int /*max_depth*/,
	const char* /*output_path*/,
	const char* /*scene_id*/,
	double /*cam_x*/,
	double /*cam_y*/,
	double /*cam_z*/,
	int /*force_camera_override*/ = 0,
	const RenderOptions& /*options*/ = {}
) {
	return -1;
}

inline int optix_render_main_sppm(
	int /*image_width*/,
	int /*image_height*/,
	int /*iterations*/,
	int /*photons*/,
	int /*max_depth*/,
	const char* /*output_path*/,
	const char* /*scene_id*/,
	double /*cam_x*/,
	double /*cam_y*/,
	double /*cam_z*/,
	int /*force_camera_override*/ = 0
) {
	return -1;
}

inline int gpu_scene_light_count(const char* /*scene_id*/, int /*image_width*/, int /*image_height*/) {
	return -1;
}

inline int optix_validation_issue_count() { return 0; }

inline const char* optix_validation_issue(int /*index*/) { return ""; }

inline void optix_clear_validation_issues() {}

inline bool optix_validation_enabled() { return false; }

#ifdef __cplusplus
}
#endif
