// Metal renderer C API - mirrors gpu/optix/optix_interface.h's own shape
// exactly (down to reusing the same RenderOptions struct), so launcher/main.cpp
// routes --gpu to this backend on macOS the same way it routes to
// optix_render_main() on Windows, with no Metal/Objective-C headers of its
// own ever needing to leak into main.cpp.
//
// WIRED IN: launcher/main.cpp's --gpu dispatch calls metal_render_main()
// directly on a RT_HAVE_METAL build (see docs/METAL_GPU_FEASIBILITY.md's
// own phase 3a/3b sections for how ray_tracer itself came to link this
// code, and main.cpp's own #ifdef RT_HAVE_METAL block for the call site).

#pragma once

#include "../../src/shared/render_options.h"

#ifdef __cplusplus
extern "C" {
#endif

// Mirrors gpu/optix/optix_interface.h's own OptixDiagnostics struct in
// spirit (same "available flag + device name + failure reason" shape,
// so launcher/diagnostics.cpp's own append_gpu() can report on whichever
// GPU backend this build actually has with one consistent report style)
// but NOT in exact field layout - CUDA driver/runtime versions and a
// separate VRAM free/total split have no Metal equivalent (Apple
// Silicon's unified memory is shared with the CPU, not a separate pool
// with its own free/total to query the way a discrete GPU's VRAM is),
// so those fields are replaced with what's actually meaningful here
// rather than padded out with unused CUDA-shaped ones.
struct MetalDiagnostics {
	bool available;
	char device_name[256];
	// Apple's own recommendedMaxWorkingSetSize - not a hard VRAM total
	// the way optix_get_diagnostics()'s own vram_total_bytes is (there
	// is no fixed GPU-only memory pool to report), but the closest
	// analogous "how much can this GPU comfortably use" figure Metal
	// itself exposes.
	unsigned long long recommended_max_working_set_bytes;
	char failure_reason[256];
};

// Reports whether a Metal device is available on this machine at all -
// true even on a build with RT_BUILD_METAL=OFF is impossible (this
// function only exists in that build's own metal_interface.h in the
// first place; a non-Metal build never links or declares it - see
// launcher/diagnostics.cpp's own #ifdef RT_HAVE_METAL guard around its
// one call site). False here means "this Mac itself has no usable
// Metal GPU" (e.g. running under conditions Metal doesn't support),
// not "this build lacks Metal support" - that second case is a
// compile-time, not runtime, fact.
bool metal_get_diagnostics(MetalDiagnostics* out);

// scene_id must resolve to a pbrt-file-backed scene (cpu_scene_pbrt_path_
// by_id() returns non-empty) - this backend's own scene loader
// (loadPbrtScene(), metal_poc.mm) doesn't yet reproduce this project's
// hand-authored built-in scenes the way gpu/optix/scene_builder.cpp's own
// switch-case does. Returns non-zero (not a crash) for any scene_id that
// doesn't resolve that way, same graceful-failure shape scene_builder.cpp's
// own default: case already established for GPU-unsupported scenes.
//
// force_camera_override/cam_x/y/z: when force_camera_override is set,
// overrides ONLY the camera's lookfrom position - lookat/up/vfov always
// come from the scene's own pbrt Camera block, matching
// cpu_interface.cpp's own applyCameraConfig() semantics exactly. cam_x/
// y/z are read in the scene's own pbrt-file-authored coordinate space
// (the same space cpu_scene_recommended_camera() and any explicit CLI
// --cam-x/y/z already use for every other backend) - see
// applyCameraOverride()'s own comment (metal_poc.mm) for the coordinate
// transform this applies before use.
//
// options: options.tonemap ("aces"/"reinhard"/"none", matching
// cpu_render_main()/optix_render_main()'s own convention), options.exposure
// (a flat pre-tonemap multiplier, section 148, docs/METAL_GPU_FEASIBILITY.md),
// options.seed (selects the per-pixel RNG stream - section 204), and
// options.isolate_pbrt_lighting (skips the hardcoded demo room's own
// lights for a loaded pbrt scene - section 197/199, docs/
// METAL_GPU_FEASIBILITY.md) are all read - every other RenderOptions field
// is a documented no-op for this backend (this POC doesn't implement
// sampler/adaptive_sampling/lightsampler/regularize/max_component_value/
// crop/aperture-or-focus-override/spectral/denoise yet), same "flag
// has no effect under X" convention render_options.h's own header comment
// already documents for other backend/field combinations.
int metal_render_main(
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

#ifdef __cplusplus
}
#endif
