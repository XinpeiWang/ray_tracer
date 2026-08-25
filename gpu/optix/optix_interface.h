// OptiX renderer C API
// Matches the signature of gpu/cuda/gpu_interface.h for drop-in replacement

#pragma once

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
// scene_builder.cpp's scene 1/2 cases). main.cpp's video-mode frame loop
// passes 1, since a video needs to honor its per-frame animated camera
// regardless of the scene's single-image default; single-image rendering
// passes 0 (default).
// denoise: 1 = run the OptiX AI denoiser on the finished render before
// writing output_path (recursive GPU backend only - silently has no effect
// under wavefront mode, see OptiXRenderer::enableDenoise()'s comment for
// why). 0 (default) = off, matching every existing caller's prior behavior.
// exposure: flat multiplier on linear color, applied right before tone-
// mapping (see the final pixel-writing loop in optix_interface.cpp, which
// covers both the recursive and wavefront backends). 1.0 (default) is a
// no-op; mirrors cpu_render_main()'s own exposure parameter.
// tonemap: which operator that same pixel-writing loop applies before the
// sRGB OETF (src/shared/tone_map.h's ToneMapMode) - one of "aces"/
// "reinhard"/"none"; nullptr, empty, or an unrecognized name all fall back
// to "aces" (this project's pre-existing hardcoded default). Also covers
// both the recursive and wavefront backends, for the same reason exposure
// does; mirrors cpu_render_main()'s own tonemap parameter.
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
	int denoise = 0,
	double exposure = 1.0,
	const char* tonemap = nullptr
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
