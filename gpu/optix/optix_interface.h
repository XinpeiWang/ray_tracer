// OptiX renderer C API
// Matches the signature of gpu/cuda/gpu_interface.h for drop-in replacement

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Check if GPU/OptiX rendering is available
bool optix_is_available();

// Main OptiX rendering entry point
// Supports multiple scenes via scene_id parameter
// force_camera_override: 1 = always use cam_x/y/z, even for scenes that
// otherwise ignore it and use their own fixed lookfrom (see
// scene_builder.cpp's scene 1/2 cases). main.cpp's video-mode frame loop
// passes 1, since a video needs to honor its per-frame animated camera
// regardless of the scene's single-image default; single-image rendering
// passes 0 (default).
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
	int force_camera_override = 0
);

// GPU SPPM (Stochastic Progressive Photon Mapping) rendering entry point,
// mirrors cpu_render_main_sppm()'s signature (cpu_renderer/cpu_interface.h)
// so main.cpp's --sppm --gpu branch (sub-phase 1e) can call either with the
// same argument shape. Phase 1 scope only: rejects any scene_id other than
// "B3" (CornellRoughGlass, old flat id 11) with ERR_GPU_UNSUPPORTED_SCENE.
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
