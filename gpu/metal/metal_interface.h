// Metal renderer C API - mirrors gpu/optix/optix_interface.h's own shape
// exactly (down to reusing the same RenderOptions struct), so a future
// launcher/main.cpp caller could route --gpu to this backend on macOS the
// same way it already routes to optix_render_main() on Windows, with no
// Metal/Objective-C headers of its own ever needing to leak into main.cpp.
//
// NOT YET WIRED IN: metal_render_main() is implemented and works
// standalone (gpu/metal/metal_poc.mm, callable and unit-tested there), but
// nothing outside that file's own main()/tests calls it yet - ray_tracer's
// own CMake target doesn't compile or link metal_poc.mm's code at all
// (see docs/METAL_GPU_FEASIBILITY.md's own section on this phase for
// what's still needed: enabling OBJCXX on that target, a static-lib-style
// build analogous to optix_renderer's own, and the actual launcher/
// main.cpp dispatch branch). This header exists now so that future wiring
// has an already-settled, OptiX-shaped signature to call against.

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
// force_camera_override/cam_x/y/z: NOT YET IMPLEMENTED - honored only
// insofar as a non-zero force_camera_override prints a warning; the
// scene's own camera is always used. See metal_render_main()'s own
// definition (metal_poc.mm) for why (the coordinate rescale/recentre/
// offset loadPbrtScene() applies isn't yet exposed for a caller-supplied
// override to go through the same transform).
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
