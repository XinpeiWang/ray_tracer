#ifndef TONE_MAP_H
#define TONE_MAP_H
//==============================================================================
// tone_map.h -- Filmic tone mapping and sRGB gamma encoding
//
// pbrt-v4 context
// ---------------
// pbrt-v4 outputs floating-point EXR images and defers tone mapping to a
// separate tool (imgtool).  Because this renderer outputs 8-bit PPM, tone
// mapping must happen in the write path.  We implement two widely-used
// industry operators:
//
//   ACES (default)   -- Narkowicz 2015 approximation of the ACES RRT+ODT.
//                       Same curve used by Unreal Engine 4.  Maps [0, inf)
//                       to [0, 1) with a filmic shoulder and pleasant
//                       highlight rolloff.  Reference:
//                       https://knarkowicz.wordpress.com/2016/01/06/
//                       aces-filmic-tone-mapping-curve/
//
//   Reinhard         -- Simple operator: L' = L / (1 + L).
//                       Preserves hue at low values; faster but less
//                       filmic at high exposure.
//
//   None             -- Clamp only (legacy behavior for scenes without HDR).
//
// sRGB gamma
// ----------
// The standard sRGB OETF is a piecewise function:
//   x <= 0.0031308  ->  12.92 * x
//   x >  0.0031308  ->  1.055 * x^(1/2.4) - 0.055
// This replaces the previous sqrt (gamma-2.0) approximation and more
// accurately matches display hardware.
//==============================================================================

#include <cmath>

// ---------------------------------------------------------------------------
// sRGB OETF  (linear float -> display-encoded float in [0,1])
// ---------------------------------------------------------------------------
inline double linear_to_srgb(double x) {
	if (x <= 0.0)           return 0.0;
	if (x >= 1.0)           return 1.0;
	if (x <= 0.0031308)     return 12.92 * x;
	return 1.055 * std::pow(x, 1.0 / 2.4) - 0.055;
}

// ---------------------------------------------------------------------------
// ACES filmic tone mapping -- Narkowicz 2015 approximation
// Input:  linear HDR value (any non-negative float)
// Output: tone-mapped value in [0, 1)
//
// Curve: f(x) = x * (a*x + b) / (x * (c*x + d) + e)
// with a=2.51, b=0.03, c=2.43, d=0.59, e=0.14
// ---------------------------------------------------------------------------
inline double aces_narkowicz(double x) {
	if (x <= 0.0) return 0.0;
	const double a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
	double mapped = (x * (a * x + b)) / (x * (c * x + d) + e);
	return mapped < 0.0 ? 0.0 : (mapped > 1.0 ? 1.0 : mapped);
}

// ---------------------------------------------------------------------------
// Reinhard tone mapping
// Input:  linear HDR value
// Output: value in [0, 1)
// ---------------------------------------------------------------------------
inline double reinhard(double x) {
	if (x <= 0.0) return 0.0;
	return x / (1.0 + x);
}

// ---------------------------------------------------------------------------
// ToneMapMode -- selects the active operator
// ---------------------------------------------------------------------------
enum class ToneMapMode {
	ACES,       // Narkowicz ACES RRT+ODT approximation (default)
	Reinhard,   // Simple Reinhard operator
	None        // No tone mapping -- clamp only (legacy behavior)
};

// ---------------------------------------------------------------------------
// apply_tone_map -- apply the selected operator to a single channel
// ---------------------------------------------------------------------------
inline double apply_tone_map(double x, ToneMapMode mode) {
	switch (mode) {
		case ToneMapMode::ACES:     return aces_narkowicz(x);
		case ToneMapMode::Reinhard: return reinhard(x);
		default:                    return x < 0.0 ? 0.0 : x;
	}
}

#endif
