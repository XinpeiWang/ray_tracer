// realtime_renderer_dll.cpp -- Thin DLL boundary exposing optix_renderer's
// live-preview C API (gpu/optix/optix_interface.h's rt_realtime_render_frame())
// to toolchains that can't link optix_renderer.lib directly - specifically
// the MinGW-built Qt GUI (see qt_gui/RayTracerGUI.pro), which is ABI-
// incompatible with this MSVC-built static lib. x64 Windows has a single
// calling convention regardless of compiler, so a MinGW binary CAN call
// this extern "C" export via LoadLibrary/GetProcAddress even though it
// can't link the .lib directly - exact same pattern as
// scene_metadata/scene_metadata_dll.cpp uses for cpu_renderer.lib.
//
// Two exports, both direct passthroughs to optix_interface.h - no logic
// lives in this file.
#include "../gpu/optix/optix_interface.h"

// __declspec(dllexport) is MSVC/MinGW-only - matches
// scene_metadata_dll.cpp's own comment on why this is Windows-specific.
#if defined(_WIN32)
#define RT_REALTIME_API extern "C" __declspec(dllexport)
#else
#define RT_REALTIME_API extern "C" __attribute__((visibility("default")))
#endif

RT_REALTIME_API bool realtime_render_frame(
	const char* scene_id,
	int image_width,
	int image_height,
	int samples_per_pixel,
	int max_depth,
	double cam_x,
	double cam_y,
	double cam_z,
	bool has_custom_lookat,
	double lookat_x,
	double lookat_y,
	double lookat_z,
	bool denoise,
	double denoise_blend,
	float* out_world_pos_buffer,
	float* out_camera_basis,
	float* out_rgb_buffer,
	bool enable_svgf,
	bool enable_restir_gi,
	float max_component_value,
	const SvgfTuningParams* svgf_tuning,
	bool enable_restir_di,
	bool enable_probe_cache
) {
	return rt_realtime_render_frame(scene_id, image_width, image_height,
		samples_per_pixel, max_depth, cam_x, cam_y, cam_z,
		has_custom_lookat, lookat_x, lookat_y, lookat_z,
		denoise, denoise_blend, out_world_pos_buffer, out_camera_basis, out_rgb_buffer,
		enable_svgf, enable_restir_gi, max_component_value, svgf_tuning, enable_restir_di,
		enable_probe_cache);
}

RT_REALTIME_API const char* realtime_get_last_error() {
	return rt_realtime_get_last_error();
}
