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
// listed here. The ~25 CLI-only flags (--bdpt/--mlt/--sppm, the debug
// integrators, their sub-flags) stay plain literals in launcher_args.h -
// the GUI never builds them, so there's no cross-process duplication to
// fix for those, and forcing them into this header would suggest a
// GUI/CLI coupling that doesn't actually exist.
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
}
