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
	// pbrt-v4 Integrator "string lightsampler" - which of this project's
	// own light-sampler classes selects the next-event-estimation light to
	// sample - one of "uniform"/"power"/"bvh"; nullptr/empty/unrecognized
	// all fall back to "bvh" (pbrt-v4's own real default). CPU default
	// path tracer only - affects convergence/variance, not the converged
	// image, same perf/quality-knob shape as `sampler` above.
	const char* lightsampler = nullptr;
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
