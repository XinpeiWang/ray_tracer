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
};
