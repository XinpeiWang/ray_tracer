// scene_metadata_dll.cpp -- Thin DLL boundary exposing cpu_renderer's
// scene-metadata C API (cpu_interface.h) to toolchains that can't link
// cpu_renderer.lib directly - specifically the MinGW-built Qt GUI (see
// qt_gui/RayTracerGUI.pro), which is ABI-incompatible with this MSVC-built
// static lib. x64 Windows has a single calling convention regardless of
// compiler, so a MinGW binary CAN call these extern "C" exports via
// LoadLibrary/GetProcAddress even though it can't link the .lib directly.
//
// Every export here is a direct passthrough to cpu_interface.h - no logic
// lives in this file. Add a new export here (and to
// qt_gui/scene_metadata_client.h/.cpp) whenever the GUI needs another piece
// of scene_registry.h's data instead of duplicating it locally.
//
// Note: this links the whole cpu_renderer.lib, not just these two
// functions - cpu_interface.cpp is one translation unit containing both
// this metadata lookup AND cpu_render_main (the full CPU path tracer), so
// the linker pulls in all of it, including every scene builder's
// std::function entry in scene_registry.h's table. The resulting DLL
// (~400KB) is small enough that this hasn't mattered in practice; splitting
// cpu_interface.cpp would be the fix if it ever does.
#include "../cpu_renderer/cpu_interface.h"

#define SCENE_METADATA_API extern "C" __declspec(dllexport)

SCENE_METADATA_API int scene_metadata_gpu_compatible(int scene_id) {
	return cpu_scene_gpu_compatible_by_id(scene_id);
}

SCENE_METADATA_API int scene_metadata_recommended_camera(int scene_id,
	double* lookfrom_x, double* lookfrom_y, double* lookfrom_z,
	double* lookat_x, double* lookat_y, double* lookat_z) {
	return cpu_scene_recommended_camera(scene_id, lookfrom_x, lookfrom_y, lookfrom_z,
	                                     lookat_x, lookat_y, lookat_z);
}

// Everything below serves the presentational fields (name/description/
// performance/recommended_spp/requires_files) that used to live in the
// GUI's own duplicated src/shared/scene_descriptor.h table - see that
// header's comment for why it was removed. scene_metadata_count() plus a
// by-id lookup per field is all the GUI needs to enumerate every scene
// (ids are contiguous from 0, enforced by
// tests/unit/scene_registry_tests.cpp's IDsAreContiguousFromZero), so
// there's no separate by-index-vs-by-id split here the way cpu_interface.h
// has for its own historical reasons.
SCENE_METADATA_API int scene_metadata_count() {
	return cpu_scene_count();
}

SCENE_METADATA_API const char* scene_metadata_name(int scene_id) {
	return cpu_scene_name_by_id(scene_id);
}

SCENE_METADATA_API const char* scene_metadata_category(int scene_id) {
	return cpu_scene_category_by_id(scene_id);
}

SCENE_METADATA_API const char* scene_metadata_description(int scene_id) {
	return cpu_scene_description_by_id(scene_id);
}

SCENE_METADATA_API const char* scene_metadata_performance(int scene_id) {
	return cpu_scene_performance_by_id(scene_id);
}

SCENE_METADATA_API int scene_metadata_recommended_spp(int scene_id) {
	return cpu_scene_recommended_spp_by_id(scene_id);
}

SCENE_METADATA_API int scene_metadata_requires_files(int scene_id) {
	return cpu_scene_requires_files_by_id(scene_id);
}
