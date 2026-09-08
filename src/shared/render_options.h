#pragma once
// render_options.h -- the render-behavior flag tail shared by
// cpu_render_main() (cpu_renderer/cpu_interface.h) and optix_render_main()
// (gpu/optix/optix_interface.h).
//
// Previously each of these took 4-5 trailing positional parameters
// (exposure, sampler, spectral, tonemap / denoise, exposure, tonemap) -
// several same-typed and adjacent, so a transposed pair at a call site
// compiled cleanly and broke silently at runtime. Bundling them into one
// named-field struct removes that risk at the actual function-call
// boundary. Not every field applies to every backend/mode - see each
// field's own comment - callers pass a default-constructed RenderOptions
// for the fields that don't apply to them, matching this project's
// existing "flag has no effect under X" warn-and-ignore convention rather
// than a compile-time restriction.
struct RenderOptions {
	// Flat multiplier on linear color, applied right before tone-mapping.
	// 1.0 (default) is a no-op. Both backends, default path tracer only.
	double exposure = 1.0;
	// Which ported pbrt-v4 sampler drives random decisions - one of
	// "sobol"/"zsobol"/"paddedsobol"/"stratified"/"pmj02bn"/"halton";
	// nullptr/empty/unrecognized all fall back to "sobol". CPU default
	// path tracer only.
	const char* sampler = nullptr;
	// Stops sampling a pixel early once it's converged, instead of always
	// spending the full samples-per-pixel budget on every pixel - see
	// camera::adaptive_sampling's own comment (camera.h) and
	// src/shared/adaptive_sampling.h. Off (default) renders exactly
	// samples_per_pixel samples per pixel, unchanged. CPU default path
	// tracer only, same scope cut as `sampler` above.
	bool adaptive_sampling = false;
	// Target relative standard error of a pixel's running luminance mean -
	// only consulted when adaptive_sampling is true. 0.01 (default) matches
	// Blender Cycles' own adaptive_threshold default.
	double adaptive_threshold = 0.01;
	// Stops claiming new scanlines once this many seconds have elapsed -
	// see camera::time_limit_seconds's own comment (camera.h). <= 0.0
	// (default) means "no limit", unchanged prior behavior. CPU default
	// path tracer only, same scope cut as `sampler`/adaptive_sampling
	// above - camera::render() is this integrator's own loop, not
	// something BDPT/MLT/SPPM's separate render loops call into.
	double time_limit_seconds = 0.0;
	// pbrt-v4 Integrator "string lightsampler" - which of this project's
	// own light-sampler classes selects the next-event-estimation light to
	// sample - one of "uniform"/"power"/"bvh", or "auto" (resolved by
	// light_sampler_resolution.h to the loaded scene's own Integrator
	// "string lightsampler" request, falling back to "bvh" if it made none
	// or requested something this project doesn't implement); nullptr/
	// empty/unrecognized all fall back to "bvh" (pbrt-v4's own real
	// default). CPU default path tracer only - affects convergence/variance,
	// not the converged image, same perf/quality-knob shape as `sampler`
	// above.
	const char* lightsampler = nullptr;
	// NOTE: --accelerator/--splitmethod are deliberately NOT fields here.
	// Unlike every field above (each read by cpu_render_main(), the CPU
	// default path tracer's own RenderOptions consumer), accelerator/
	// splitmethod affect scene CONSTRUCTION itself (scene_registry.h's
	// build_world(), shared by every CPU integrator - default path tracer,
	// BDPT/MLT, and SPPM all call it, and none of the latter take a
	// RenderOptions parameter at all - see launcher/main.cpp's own comment
	// on render_options_from_args()). Threading them through here would
	// only reach the default path tracer, silently missing the others.
	// Instead launcher/main.cpp sets src/shared/accelerator_override.h's
	// process-global directly, once, before any entry point's first scene
	// lookup - see that header's own comment for the full reasoning.
	// pbrt-v4 Integrator "bool regularize" as an explicit CLI request -
	// see LaunchArgs::regularize's own comment (launcher_args.h) for the
	// full "only ever forces ON, never overrides a scene's own true back
	// off" reasoning. Both backends, default path tracer only.
	bool regularize = false;
	// pbrt-v4 Film "maxcomponentvalue" (the firefly-clamp threshold) as an
	// explicit CLI request - 1e9 (matching camera_t::max_component_value's
	// own class default) means "not explicitly requested", so a scene's
	// own Film directive still applies unless this differs. CPU default
	// path tracer only - GPU has no equivalent clamp.
	double max_component_value = 1e9;
	// pbrt-v4 Film "cropwindow" (NDC fractions in [0,1]) as an explicit CLI
	// request - {0,0,1,1} (the full frame) means "not explicitly
	// requested", so a scene's own cropwindow/pixelbounds directive still
	// applies unless this differs. Both backends, default path tracer only.
	double crop_x0 = 0.0, crop_y0 = 0.0, crop_x1 = 1.0, crop_y1 = 1.0;
	// Real hero-wavelength spectral rendering instead of flat RGB. CPU
	// default path tracer only, 6-material whitelist (see camera.h's
	// ray_color_spectral()'s own comment).
	bool spectral = false;
	// Which operator write_color() applies before the sRGB OETF - one of
	// "aces"/"reinhard"/"none"; nullptr/empty/unrecognized all fall back
	// to "aces". Both backends, default path tracer only.
	const char* tonemap = nullptr;
	// Run the OptiX AI denoiser on the finished render. GPU only - both the
	// recursive and wavefront backends have their own denoiser. No effect
	// under GPU SPPM (launcher/main.cpp warns on --denoise --sppm --gpu)
	// or any CPU-only integrator.
	bool denoise = false;
	// OptiX's own blend between the noisy input and the fully denoised
	// output - 0.0 (default) = 100% denoised (this project's prior,
	// only-ever behavior), 1.0 = the original noisy image unchanged,
	// linearly interpolated in between. Only consulted when `denoise` is
	// true. Same GPU-only, both-backends, "no effect under GPU SPPM" scope
	// as `denoise` above - see gpu/optix/optix_denoiser.h's runDenoiser()
	// for where this actually reaches the OptiX API. Interpolates in the
	// linear, pre-tonemap HDR color buffer the denoiser itself operates on,
	// not the final tonemapped/gamma-corrected image.
	float denoise_blend = 0.0f;
	// An explicit CLI request for reproducible renders. -1 (default) means
	// "not requested" - both backends fall back to their own pre-existing
	// behavior (CPU: genuinely non-deterministic, seeded from hardware
	// entropy every run; GPU: already deterministic at frame 0, unchanged).
	// >= 0 makes CPU deterministic too (see camera_t::seed's own comment,
	// camera.h) and gives GPU an explicit, chosen starting seed instead of
	// always 0. Both backends, default path tracer only - same scope cut
	// as regularize/max_component_value/crop above.
	long long seed = -1;
};
