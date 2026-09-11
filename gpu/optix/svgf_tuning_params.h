#pragma once
// svgf_tuning_params.h -- runtime-tunable SVGF constants (gpu/optix/
// wavefront_kernels_svgf.cu), formerly hardcoded `constexpr` literals in that
// file plus wavefront_path_tracer.cpp's own atrous-pass-count loop constant.
// Plain POD, zero CUDA/OptiX includes - safe to reference from any host
// translation unit. In-class defaults reproduce today's literature-default
// behavior exactly (Schied et al. 2017 and the common reference
// implementations descended from it - see wavefront_kernels_svgf.cu's own
// header comment), so a default-constructed instance changes nothing.
//
// This type crosses the qt_gui (MinGW) <-> realtime_renderer.dll (MSVC) ABI
// boundary as a pointer (rt_realtime_render_frame()'s own `svgf_tuning`
// parameter) - see that function's own comment for why there's no shared
// header/versioning across that boundary. qt_gui hand-declares an IDENTICAL
// mirror struct of this same name next to its RenderFrameFn typedef
// (qt_gui/realtime_preview_session.h) rather than including this file
// directly - keep the two in sync by hand on every change, same as every
// other type/signature crossing that boundary.
struct SvgfTuningParams {
	// Temporal blend floor (wf_svgf_temporal_alpha) - acts like a plain
	// running mean while history is short, then clamps to this rate once
	// history grows past 1/temporalAlpha - 1 frames.
	float temporalAlpha = 0.2f;
	// Caps how many frames of history a converged pixel can accumulate -
	// bounds how "sticky" it gets.
	float maxHistoryLength = 32.0f;
	// Below this history length, variance is spatially prefiltered from
	// neighboring pixels instead of trusted alone.
	float varianceBootstrapFrames = 4.0f;
	// Box radius (in pixels) for the variance prefilter above - radius 3
	// means a 7x7 box.
	int varianceBootstrapRadius = 3;
	// A-trous edge-stopping sigma for the shading-normal term - higher
	// rejects a smaller normal difference.
	float sigmaNormal = 128.0f;
	// A-trous edge-stopping sigma for the depth term (relative to the local
	// depth gradient).
	float sigmaDepth = 1.0f;
	// A-trous edge-stopping sigma for the luminance term (relative to the
	// pixel's own estimated noise level).
	float sigmaLuminance = 4.0f;
	// A-trous filter footprint radius per pass - MUST stay in [0,2]:
	// svgf_atrous_pass's own binomial kernel-weight table (wavefront_kernels_
	// svgf.cu's kKernel[3]) only has 3 entries, indexed by this value. The
	// kernel itself clamps defensively, but the GUI's own spinbox range is
	// the primary guard - see mainwindow_tabs.cpp's own comment on this
	// control.
	int atrousRadius = 2;
	// Floor applied before dividing color by albedo (demodulation) - keeps a
	// near-zero-albedo pixel (background/sky) from round-tripping to black.
	float minAlbedo = 0.02f;
	// Number of A-trous filter passes (step sizes double each pass:
	// 1,2,4,8,...) - a host-side loop count, not a device kernel parameter.
	int atrousPasses = 4;
};
