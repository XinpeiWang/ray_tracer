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
