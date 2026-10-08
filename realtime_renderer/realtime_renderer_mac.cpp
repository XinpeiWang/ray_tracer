// realtime_renderer_mac.cpp - the macOS realtime_renderer.dylib: the same C ABI as realtime_renderer_dll.cpp (what the Qt GUI's
// Live Preview loads, qt_gui/realtime_preview_session.cpp), backed by the Metal live session (gpu/metal/metal_live_preview.h)
// instead of OptiX.
//
// realtime_render_frame() is declared once, in src/shared/realtime_api.h, which the GUI and the OptiX library include too, so this definition cannot drift from
// them. Most of its flags are OptiX-only features (AI denoiser, SVGF, ReSTIR, probe cache, path guiding, temporal/neural upscale, NRC, adaptive
// sampling); the Metal backend has none of them, so they are accepted and ignored: Metal returns the plain per-pixel mean of `samples_per_pixel`
// samples and the GUI accumulates frames itself. realtime_backend_features() tells the GUI which controls to offer.
#include <atomic>

#include "../gpu/metal/metal_live_preview.h"
#include "../src/shared/realtime_api.h"

RT_REALTIME_API bool realtime_render_frame(
	const char* scene_id, int image_width, int image_height, int samples_per_pixel, int max_depth,
	double cam_x, double cam_y, double cam_z, bool has_custom_lookat, double lookat_x, double lookat_y, double lookat_z,
	bool /*denoise*/, double /*denoise_blend*/, float* out_world_pos_buffer, float* out_camera_basis, float* out_rgb_buffer,
	bool /*enable_svgf*/, bool /*enable_restir_gi*/, float max_component_value, const SvgfTuningParams* /*svgf_tuning*/,
	bool /*enable_restir_di*/, bool /*enable_probe_cache*/, bool /*enable_path_guiding*/,
	bool /*enable_temporal_upscale*/, int /*temporal_upscale_factor*/, unsigned int /*temporal_jitter_base_index*/,
	bool /*enable_nrc*/, bool /*enable_neural_upscale*/, float* /*out_neural_upscale_buffer*/,
	double aperture_override, double focus_distance_override,
	bool /*enable_adaptive_sampling*/, const unsigned char* /*in_active_pixel_mask*/
) {
	// The GUI does not pass a frame index, so number the calls here: successive frames need different noise.
	static std::atomic<unsigned int> frame{0};
	return metal_live_render_frame(scene_id, image_width, image_height, samples_per_pixel, max_depth,
		cam_x, cam_y, cam_z, has_custom_lookat, lookat_x, lookat_y, lookat_z,
		max_component_value, frame.fetch_add(1u), out_rgb_buffer, out_world_pos_buffer, out_camera_basis,
		aperture_override, focus_distance_override);
}

RT_REALTIME_API const char* realtime_get_last_error() {
	return metal_live_last_error();
}

// The Metal library has the thin-lens depth-of-field override and nothing else of the optional set yet.
RT_REALTIME_API void realtime_backend_features(RealtimeBackendFeatures* out) {
	if (out) *out = RealtimeBackendFeatures{false, false, false, false, false, false, false, false, false, false, true};
}
