#pragma once
// wavefront_temporal_upscale_math.h -- deterministic per-frame sub-pixel
// jitter sequence for Live Preview's temporal upscale feature (see this
// project's own plan). Render resolution stays exactly what it is today
// (still one small internal render per frame) - this only changes WHICH
// sub-pixel offset within a pixel's footprint each frame samples, so that
// a handful of low-res frames sweep out full coverage of a higher-
// resolution grid instead of landing at whitenoise-random offsets.
//
// wf_temporal_upscale_subcell() is the single source of truth: an EXACT
// bijection from a sample index to one of upscaleFactor*upscaleFactor
// sub-cells, via a hard-coded ordered-dither ("Bayer matrix") table - the
// standard real-time-rendering construction for "N samples, N cells, visit
// every cell exactly once per period, reasonably spread out along the way."
// An earlier version of this file used a quantized Halton(2,3) sequence
// instead (floor(haltonX*U), floor(haltonY*U)) - LOOKS like it should tile
// the grid evenly, but does not: this project's own unit tests caught it
// leaving 1 of 4 cells unvisited at upscaleFactor=2, and 2 of 16 at
// upscaleFactor=4, within a single period. A low-discrepancy sequence
// minimizes long-run gaps, not exact short-window coverage - the wrong
// property for a feature whose whole design depends on every high-res cell
// getting a fresh sample within one period. The Bayer table's small size
// (only 2 and 4 need supporting - this project's own v1 scope, see the
// plan's own Files-to-change section) makes hard-coding it directly cheap
// and exactly correct, with none of that risk.
//
// wf_temporal_upscale_jitter() derives the continuous [0,1) sub-pixel
// offset generate_camera_rays needs from the SAME sub-cell (its own
// center) - keeping the two functions consistent by construction: whichever
// sub-cell a sample's ray direction actually came from is exactly the cell
// RealtimePreviewWorker's own high-res splat step (using
// wf_temporal_upscale_subcell(), hand-duplicated into qt_gui/camera_math.h)
// will file it under.
//
// Dependency-free like wavefront_guiding.h's own header comment explains for
// the same reason: usable from a plain host build for unit testing, no
// ordering constraint on where it's included from.

#include "../../src/shared/cpu_gpu.h"

// See this file's own header comment. `sampleIndex` is taken mod the
// period (upscaleFactor*upscaleFactor) internally, so callers may pass an
// ever-increasing counter without pre-wrapping it themselves. Only
// upscaleFactor 2 and 4 are supported (this project's own v1 scope) -
// anything else falls back to the 2x table.
CPU_GPU void wf_temporal_upscale_subcell(unsigned int sampleIndex, int upscaleFactor, int& outCx, int& outCy) {
	if (upscaleFactor == 4) {
		// Standard 4x4 ordered-dither (Bayer) matrix, inverted (value ->
		// position) so it can be indexed directly by sample index:
		//    0  8  2 10
		//   12  4 14  6
		//    3 11  1  9
		//   15  7 13  5
		static const int kInverse4x4[16][2] = {
			{0, 0}, {2, 2}, {2, 0}, {0, 2}, {1, 1}, {3, 3}, {3, 1}, {1, 3},
			{1, 0}, {3, 2}, {3, 0}, {1, 2}, {0, 1}, {2, 3}, {2, 1}, {0, 3}
		};
		const unsigned int i = sampleIndex % 16u;
		outCx = kInverse4x4[i][0];
		outCy = kInverse4x4[i][1];
		return;
	}
	// upscaleFactor == 2 (this project's only other supported v1 factor).
	// 2x2 Bayer matrix [[0,2],[3,1]], inverted the same way.
	static const int kInverse2x2[4][2] = { {0, 0}, {1, 1}, {1, 0}, {0, 1} };
	const unsigned int i = sampleIndex % 4u;
	outCx = kInverse2x2[i][0];
	outCy = kInverse2x2[i][1];
}

// Continuous [0,1) sub-pixel jitter for sampleIndex - the center of the
// sub-cell wf_temporal_upscale_subcell() assigns it (see this file's own
// header comment for why the two must stay consistent).
CPU_GPU void wf_temporal_upscale_jitter(unsigned int sampleIndex, int upscaleFactor, float& outRx, float& outRy) {
	int cx = 0, cy = 0;
	wf_temporal_upscale_subcell(sampleIndex, upscaleFactor, cx, cy);
	// upscaleFactor is caller-controlled (ultimately from UI/host settings) -
	// guard against 0/negative reaching a divide, same defensive posture as
	// wf_temporal_upscale_subcell()'s own "anything else falls back" comment.
	const float factor = static_cast<float>(upscaleFactor > 0 ? upscaleFactor : 1);
	outRx = (static_cast<float>(cx) + 0.5f) / factor;
	outRy = (static_cast<float>(cy) + 0.5f) / factor;
}
