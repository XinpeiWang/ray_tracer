#pragma once
// realtime_api.h -- the C ABI of the Live Preview renderer library (realtime_renderer.dll on Windows, realtime_renderer.dylib on macOS), declared ONCE.
//
// Three programs touch it: the Qt GUI loads the library at run time (qt_gui/realtime_preview_session.cpp, through RealtimeRenderFrameFn below),
// realtime_renderer/realtime_renderer_dll.cpp implements it on top of OptiX, and realtime_renderer/realtime_renderer_mac.cpp implements it on top of
// Metal. All three include this header, so a parameter added, removed or retyped in one place no longer compiles in the others - the signatures
// used to be copied by hand into each, with only comments to keep them in step.
//
// New parameters are always appended at the end, never inserted in the middle. A backend that does not have a feature accepts its flag and ignores
// it (the Metal library has no ReSTIR, SVGF, probe cache and so on): the GUI asks RealtimeBackendFeatures (below) which controls to offer.
//
// Plain C types only (plus SvgfTuningParams, a plain struct with no includes), so every toolchain that builds one of the three - MSVC, MinGW, clang -
// reads it the same way.

#include "../../gpu/optix/svgf_tuning_params.h"

#if defined(_WIN32)
#define RT_REALTIME_API extern "C" __declspec(dllexport)
#else
#define RT_REALTIME_API extern "C" __attribute__((visibility("default")))
#endif

// Renders one frame of `scene_id` into out_rgb_buffer (linear radiance, width*height*3 floats, top row first). The GUI accumulates frames itself.
// out_world_pos_buffer (width*height*4) and out_camera_basis (12 floats) feed the caller's temporal reprojection; either may be null.
// Returns false on failure; realtime_get_last_error() has the detail.
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
	bool enable_probe_cache,
	bool enable_path_guiding,
	bool enable_temporal_upscale,
	int temporal_upscale_factor,
	unsigned int temporal_jitter_base_index,
	bool enable_nrc,
	bool enable_neural_upscale,
	float* out_neural_upscale_buffer,
	double aperture_override,
	double focus_distance_override,
	bool enable_adaptive_sampling,
	const unsigned char* in_active_pixel_mask);

// Detail of the most recent failure on the calling thread ("" if none). Valid until the next realtime_render_frame() on that thread.
RT_REALTIME_API const char* realtime_get_last_error();

// What the GUI loads the two functions as.
using RealtimeRenderFrameFn = decltype(&realtime_render_frame);
using RealtimeGetLastErrorFn = decltype(&realtime_get_last_error);

// Which of realtime_render_frame()'s optional features a backend really implements. A flag for a feature it lacks is accepted and ignored, so the GUI
// should not offer the control. (The OptiX library does all of these; the Metal library does none of them yet.)
struct RealtimeBackendFeatures {
	bool aiDenoiser;
	bool svgf;
	bool restirGi;
	bool restirDi;
	bool probeCache;
	bool pathGuiding;
	bool temporalUpscale;
	bool neuralRadianceCache;
	bool neuralUpscale;
	bool adaptiveSampling;
	bool depthOfFieldOverride;
};

// Fills `out` with what this library implements. (Absent from a library built before this existed: the GUI then assumes the OptiX set.)
RT_REALTIME_API void realtime_backend_features(RealtimeBackendFeatures* out);
using RealtimeBackendFeaturesFn = decltype(&realtime_backend_features);
