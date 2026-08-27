#pragma once
// render_flag_names.h -- canonical CLI flag-name strings for exactly the
// flags that cross the GUI/CLI process boundary: launcher_args.h's parser
// (compiled into ray_tracer.exe, MSVC) reads them, and qt_gui's
// RenderController::start() (compiled into RayTracerGUI.exe, a separate
// MinGW/Qt binary that spawns ray_tracer.exe as a subprocess and builds
// its argv by hand) writes them. Before this header, each flag's name was
// typed as an independent string literal on both sides - a rename
// compiled cleanly on both and just broke the GUI at runtime. Referencing
// the same constant on both sides turns that into a compile error on
// whichever side still has the old name.
//
// Header-only constexpr constants: no linkage/ABI concerns crossing the
// two toolchains, since nothing here is ever linked - each binary just
// embeds its own copy of the string literal.
//
// Deliberately NOT exhaustive: only flags the GUI actually emits are
// listed here. A handful of CLI-only flags with no GUI control at all
// (e.g. --diagnose is GUI-adjacent but most debug/one-off flags aren't
// listed) stay plain literals in launcher_args.h - forcing every CLI flag
// into this header would suggest a GUI/CLI coupling that doesn't
// actually exist for the ones the GUI never touches.
namespace render_flags {
	constexpr const char* kGpu           = "--gpu";
	constexpr const char* kCpu           = "--cpu";
	constexpr const char* kWavefront     = "--wavefront";
	constexpr const char* kOutput        = "--output";
	constexpr const char* kVideo         = "--video";
	constexpr const char* kFrames        = "--frames";
	constexpr const char* kFps           = "--fps";
	constexpr const char* kSpeed         = "--speed";
	constexpr const char* kCameraPath    = "--camera-path";
	constexpr const char* kDenoise       = "--denoise";
	constexpr const char* kStats         = "--stats";
	constexpr const char* kOptixValidate = "--optix-validate";
	constexpr const char* kExposure      = "--exposure";
	constexpr const char* kSampler       = "--sampler";
	constexpr const char* kSpectral      = "--spectral";
	constexpr const char* kTonemap       = "--tonemap";
	constexpr const char* kDiagnose      = "--diagnose";

	// Alternate integrators and their sub-flags (both in the Integrator
	// group, Render Options tab) -
	// verified byte-for-byte against launcher/launcher_args.h's own
	// `arg == "..."` parsing.
	constexpr const char* kSppm               = "--sppm";
	constexpr const char* kSppmIterations     = "--sppm-iterations";
	constexpr const char* kSppmPhotons        = "--sppm-photons";
	constexpr const char* kBdpt               = "--bdpt";
	constexpr const char* kBdptMaxDepth       = "--bdpt-max-depth";
	constexpr const char* kMlt                = "--mlt";
	constexpr const char* kMltBootstrap       = "--mlt-bootstrap";
	constexpr const char* kMltMutations       = "--mlt-mutations";
	constexpr const char* kMltMaxDepth        = "--mlt-max-depth";
	constexpr const char* kRandomwalk         = "--randomwalk";
	constexpr const char* kAo                 = "--ao";
	constexpr const char* kAoMaxDist          = "--ao-max-dist";
	constexpr const char* kAoUniform          = "--ao-uniform";
	constexpr const char* kAoIllumScale       = "--ao-illum-scale";
	constexpr const char* kAoIllumRgb         = "--ao-illum-rgb";
	constexpr const char* kSimplepath         = "--simplepath";
	constexpr const char* kSimplepathNoLights = "--simplepath-no-lights";
	constexpr const char* kSimplepathNoBsdf   = "--simplepath-no-bsdf";
	constexpr const char* kSimplevolpath      = "--simplevolpath";
	constexpr const char* kLightpath          = "--lightpath";
}
