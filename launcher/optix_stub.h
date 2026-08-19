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

#ifdef __cplusplus
extern "C" {
#endif

inline bool optix_is_available() { return false; }

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
	int /*force_camera_override*/ = 0
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
