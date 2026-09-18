// Metal renderer C API - mirrors gpu/optix/optix_interface.h's own shape
// exactly (down to reusing the same RenderOptions struct), so a future
// launcher/main.cpp caller could route --gpu to this backend on macOS the
// same way it already routes to optix_render_main() on Windows, with no
// Metal/Objective-C headers of its own ever needing to leak into main.cpp.
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
// options: only options.tonemap is read ("aces"/"reinhard"/"none",
// matching cpu_render_main()/optix_render_main()'s own convention) -
// every other RenderOptions field is a documented no-op for this backend
// (this POC doesn't implement exposure/sampler/adaptive_sampling/
// lightsampler/regularize/max_component_value/crop/aperture-or-focus-
// override/spectral/denoise/seed yet), same "flag has no effect under X"
// convention render_options.h's own header comment already documents for
// other backend/field combinations.
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
