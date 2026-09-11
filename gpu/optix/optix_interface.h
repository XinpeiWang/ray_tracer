// OptiX renderer C API
// Matches the signature of cpu_renderer/cpu_interface.h so the launcher can
// call either backend through the same shape of call.

#pragma once

#include "../../src/shared/render_options.h"
#include "svgf_tuning_params.h"

// Plain-POD result of a full GPU/CUDA/OptiX capability probe (--diagnose,
// see launcher/diagnostics.h). Fixed-size char buffers rather than
// std::string, matching every other type that crosses this extern "C"
// boundary - main.cpp never includes CUDA/OptiX headers directly, only this
// file (see this file's own header comment / launcher/optix_stub.h's).
struct OptixDiagnostics {
	bool available;
	char device_name[256];
	int  cuda_driver_version;   // cudaDriverGetVersion(), e.g. 12040 = 12.4
	int  cuda_runtime_version;  // cudaRuntimeGetVersion()
	int  optix_abi_version;     // OPTIX_VERSION (compile-time SDK macro)
	unsigned long long vram_free_bytes;
	unsigned long long vram_total_bytes;
	char failure_reason[256];   // populated only when available == false
};

#ifdef __cplusplus
extern "C" {
#endif

// Check if GPU/OptiX rendering is available
bool optix_is_available();

// Full capability probe for --diagnose: device name, driver/CUDA/OptiX
// versions, VRAM. Runs the same probe sequence as optix_is_available() (see
// OptiXRenderer::isAvailable()'s comment) but reports what it found instead
// of just true/false, and fills failure_reason on the step that failed
// rather than discarding it to stderr. Always returns the same bool as
// out->available for convenience.
bool optix_get_diagnostics(OptixDiagnostics* out);

// Main OptiX rendering entry point
// Supports multiple scenes via scene_id parameter
// force_camera_override: 1 = always use cam_x/y/z, even for scenes that
// otherwise ignore it and use their own fixed lookfrom (see
// scene_builder.cpp's scene 1/2 cases). main.cpp passes 1 for both
// single-image and video-mode calls today; the default of 0 below is for
// other/future callers that want a scene's own fixed lookfrom honored.
// options: render-behavior flags (denoise/exposure/tonemap - sampler/
// spectral are CPU-only and ignored here) bundled into one struct - see
// src/shared/render_options.h's own field-by-field comments. A default-
// constructed RenderOptions matches every existing caller's prior
// behavior (no denoise, exposure=1.0/no-op, ACES tone mapping). exposure
// and tonemap both apply to the recursive AND wavefront backends (the
// final pixel-writing loop in optix_interface.cpp is the single shared
// output path for both).
int optix_render_main(
	int image_width,
	int image_height,
	int samples_per_pixel,
	int max_depth,
	const char* output_path,
	const char* scene_id,
	double cam_x,
	double cam_y,
	double cam_z,
	int force_camera_override = 0,
	const RenderOptions& options = {}
);

// Live-preview (progressive-refinement mode) entry point - see this
// project's own real-time-preview plan. One call per preview frame: renders
// samples_per_pixel samples via the wavefront backend and writes the raw
// LINEAR (pre-tonemap) RGB result into out_rgb_buffer (caller-allocated,
// >= image_width*image_height*3 floats) - no exposure/tonemap/gamma
// applied, unlike optix_render_main() above. The caller is expected to
// accumulate these across many low-spp calls and tonemap only the
// accumulated result once per displayed frame. Caches scene_id/resolution/
// camera/look-at from the last call - a repeated call where NONE of those
// changed skips scene rebuild, GPU re-upload, AND SBT rebuild entirely (all
// real per-call costs otherwise - see optix_interface.cpp's own comment),
// so only an actual camera or look-at move pays the full cost, matching
// how the caller only needs a fresh frame at all in that case (see
// RealtimePreviewWorker::setCamera()'s own reset-on-move design). Returns
// false on any failure (unsupported scene, GPU error, or wavefront mode
// unavailable) - check it every call, don't assume out_rgb_buffer was
// filled.
// has_custom_lookat/lookat_x/y/z: overrides every scene's own hardcoded
// look-at point - see build_scene()'s own comment (scene_builder.h). Live
// Preview's free-fly camera always passes has_custom_lookat=true, since it
// needs to look wherever it's facing rather than always re-aiming at each
// scene's fixed subject.
// denoise/denoise_blend: same OptiX AI denoiser already used by
// --denoise/--denoise-blend for batch/video rendering (see
// OptiXRenderer::enableDenoise()/setDenoiseBlend()'s own comments) - applied
// fresh every call, so unlike the cache above there's nothing to key on
// here. denoise_blend is the fraction of the ORIGINAL noisy signal kept
// (0.0 = fully denoised, 1.0 = denoiser disabled in all but name).
// out_world_pos_buffer: optional (pass nullptr to skip - see
// OptiXRenderer::enableWorldPosOutput()'s own comment for the cost of NOT
// skipping it), caller-allocated >= image_width*image_height*4 floats -
// filled with each pixel's world-space primary-hit point (xyz) and a
// validity flag (w: 1.0 = real hit, 0.0 = miss). Used by Live Preview's
// temporal reprojection (qt_gui/camera_math.h's projectToScreen(),
// qt_gui/realtime_preview_session.cpp) to find where a surface point
// visible in THIS frame would have appeared in a PREVIOUS frame's camera,
// so that frame's already-accumulated sample can be reused instead of
// starting over from noise on every camera move.
// out_camera_basis: optional (pass nullptr to skip), caller-allocated
// >= 12 floats - filled with the ACTUAL GpuCameraParams::origin/
// lower_left_corner/horizontal/vertical (float3 each, in that order) this
// call rendered with. This is the pairing partner out_world_pos_buffer
// needs: vfov/aspect are never available to any caller (every scene case
// bakes its own vfov into a hardcoded literal deep inside build_scene(),
// with nothing surfacing it) - reprojection instead reads back the EXACT
// basis the GPU used via camera_math.h's CameraBasis/projectToScreen(),
// not a re-derived approximation. Populated from whatever the scene's
// actual CameraKind produced; a non-Perspective/Orthographic camera (rare -
// Spherical/Realistic scenes) leaves these near/at zero, which
// projectToScreen() already treats as "nothing to project onto" rather
// than dividing by zero.
bool rt_realtime_render_frame(
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
	// SVGF spatiotemporal denoiser (gpu/optix/wavefront_svgf_math.h) - an
	// alternative to `denoise` above (the OptiX AI denoiser), not layered on
	// top of it. When true, out_rgb_buffer already holds the temporally-
	// stable, spatially-filtered result - the caller should NOT also blend
	// it into its own running-mean accumulation (see qt_gui/
	// realtime_preview_session.cpp's own effectiveShowLatest comment for why
	// SVGF's output is unconditionally treated as "already final", exactly
	// like a denoise+showLatest frame already is). Defaults false so every
	// existing call site keeps today's exact behavior unchanged.
	bool enable_svgf = false,
	// ReSTIR GI (gpu/optix/wavefront_restir_gi_math.h) resampled one-bounce
	// indirect lighting - independent from enable_svgf above. Defaults true
	// so every existing call site keeps today's exact (always-on) behavior
	// unchanged; the GUI exposes this as a real toggle.
	bool enable_restir_gi = true,
	// Firefly clamp (GpuCameraParams::maxComponentValue) - caps the brightest
	// possible sample value to suppress fireflies, at the cost of clipping
	// genuinely bright highlights. Defaults 50.0f, matching this function's
	// own previously-hardcoded literal, so every existing call site keeps
	// today's exact behavior unchanged.
	float max_component_value = 50.0f,
	// SVGF advanced tuning (gpu/optix/svgf_tuning_params.h) - nullptr (the
	// default) means "use SvgfTuningParams{}'s own literature defaults",
	// exactly matching this function's own previously-hardcoded kSvgf*
	// constants (wavefront_kernels_svgf.cu). Only meaningful when enable_svgf
	// is true. Not retained past this call - copied by value before returning.
	const SvgfTuningParams* svgf_tuning = nullptr
);

// GPU SPPM (Stochastic Progressive Photon Mapping) rendering entry point,
// mirrors cpu_render_main_sppm()'s signature (cpu_renderer/cpu_interface.h)
// so main.cpp's --sppm --gpu branch (sub-phase 1e) can call either with the
// same argument shape. Scope is determined dynamically from the built
// scene's actual materials/geometry/lights (see optix_interface.cpp's
// sppm_gpu_unsupported_reason() for the full, current rule set) rather than
// a hardcoded scene-id allowlist -- as of this writing that covers scenes
// built purely from spheres/quads, area (quad/sphere) lights only, and the
// {Lambertian, Metal, Dielectric, RoughDielectric, Conductor, DiffuseLight}
// material subset; anything else is rejected with a reason-specific
// ERR_GPU_UNSUPPORTED_SCENE message (mesh/instanced/bilinear-patch
// geometry, punctual/sky lighting, and remaining MaterialTypes like
// CoatedDiffuse/Hair/Subsurface all still fall outside that set).
int optix_render_main_sppm(
	int image_width,
	int image_height,
	int iterations,
	int photons,
	int max_depth,
	const char* output_path,
	const char* scene_id,
	double cam_x,
	double cam_y,
	double cam_z,
	int force_camera_override = 0
);

// Returns the number of emissive light primitives (quads+spheres) that
// gpu/optix/scene_builder.cpp's build_scene() would upload for scene_id, or
// -1 if the scene fails to build. build_scene() is pure host-side C++ (no
// OptiX/CUDA device calls) - this works without a GPU present, unlike
// optix_render_main(). Used by CPU/GPU scene-parity tests (see
// tests/integration/cpu_gpu_comparison_tests.cpp) to catch cases where a
// scene's CPU and GPU builders have drifted out of sync on light count.
int gpu_scene_light_count(const char* scene_id, int image_width, int image_height);

// The three functions below expose OptiXRenderer::loggedIssues() (see its
// own doc comment) for opt-in deep-validation sweeps - see
// tests/integration/optix_validation_sweep_test.cpp. Only meaningful when
// RAY_TRACER_OPTIX_VALIDATION=1 was set before the first optix_render_main()
// call in this process (validation mode is fixed for the context's whole
// lifetime); the count is always 0 otherwise. All three are no-ops (count 0,
// index out of range) before any render has created g_renderer.

// Number of OptiX log messages at level <= 3 (fatal/error/warning) recorded
// since the last optix_clear_validation_issues() call.
int optix_validation_issue_count();

// Message text for the issue at the given 0-based index (must be <
// optix_validation_issue_count()). Returns "" if out of range. Valid until
// the next optix_clear_validation_issues() call.
const char* optix_validation_issue(int index);

// Clears the recorded issue list - call between scenes/backends in a sweep
// so each iteration's pass/fail is independent of what came before it.
void optix_clear_validation_issues();

// Whether the CURRENT g_renderer's OptiX context was actually created with
// validation mode on. Unlike RAY_TRACER_WAVEFRONT (read fresh on every
// render call), RAY_TRACER_OPTIX_VALIDATION is read ONCE, at OptiX context
// creation - the first optix_render_main() call in the process, whichever
// call that happens to be. If something else already triggered context
// creation earlier in this process without the env var set, setting it now
// has no effect: the context (and every render after it) stays
// unvalidated. Callers that need validation guarantees MUST check this
// after their first render rather than trusting the env var alone - see
// tests/integration/optix_validation_sweep_test.cpp's SetUp().
bool optix_validation_enabled();

#ifdef __cplusplus
}
#endif
