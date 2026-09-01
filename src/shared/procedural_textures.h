#pragma once
// ---------------------------------------------------------------------------
// procedural_textures.h -- anti-aliased checkerboard filtering (pbrt-v4)
//
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// Apache License, Version 2.0.
//
// Originally a much larger port of pbrt-v4's non-image procedural textures
// (Bilerp/Checkerboard/Mix/Scaled/FBm/Wrinkled/Windy/Dots/Marble), added in
// an early bulk port of pbrt-v4 groundwork and never wired into the actual
// rendering pipeline (CPU's src/TheRestOfYourLife/texture.h and both GPU
// backends independently grew their own point-sampled equivalents of most
// of these - see e.g. windy_texture/wrinkled_texture/dots_texture/
// bilerp_texture in texture.h). Trimmed to just the one piece that WASN'T
// redundant: real anti-aliased checkerboard filtering (box-filtered
// footprint integration, pbrt-v4's own Checkerboard()) - texture.h's own
// checker_texture/uv_checker_texture do a hard binary cell test with no
// antialiasing. uv_checker_texture::value_diff() (texture.h) now calls
// checkerboard_weight_2d() below to blend between cells instead.
//
// checkerboard_weight_2d() only needs UV-space derivatives (dsdx/dsdy/
// dtdx/dtdy), which uv_checker_texture's value_diff() already receives -
// checker_texture (this project's own world-space 3D checker) is NOT
// wired to this: its value_diff() only carries UV derivatives, not the
// world-space point derivatives (dpdx/dpdy) real 3D antialiasing would
// need, and texture::value_diff()'s signature doesn't carry those - a
// real fix would need a broader interface change, out of scope here.
// ---------------------------------------------------------------------------

#include "texture_mapping.h"   // TexCoord2D

#include <cmath>
#include <algorithm>

namespace pt_detail {

// Anti-aliased 1D checkerboard integral function.
// pbrt-v4: d(x) in Checkerboard()
template<typename T>
CPU_GPU T checker_d(T x) {
	T y = x / T(2) - std::floor(x / T(2)) - T(0.5);
	return x / T(2) + y * (T(1) - T(2) * std::abs(y));
}

// Anti-aliased 1D checkerboard box filter.
// pbrt-v4: bf(x, r) in Checkerboard()
template<typename T>
CPU_GPU T checker_bf(T x, T r) {
	if (std::floor(x - r) == std::floor(x + r))
		return T(1) - T(2) * static_cast<T>(static_cast<int>(std::floor(x)) & 1);
	return (checker_d(x + r) - T(2) * checker_d(x) + checker_d(x - r)) / (r * r);
}

} // namespace pt_detail

// ===========================================================================
// checkerboard_weight_2d
//
// Returns a weight in [0,1] blending between two checkerboard cells, using
// anti-aliased box-filtered integration (pbrt-v4 Checkerboard()) - 0 for
// the "even" cell (matching (floor(s)+floor(t))%2==0), 1 for "odd", and a
// smooth blend when the filter footprint straddles a cell boundary.
// ===========================================================================
template<typename T>
CPU_GPU T checkerboard_weight_2d(const TexCoord2D& c) {
	T s = static_cast<T>(c.st.x);
	T t_ = static_cast<T>(c.st.y);
	T ds = std::max(std::abs(static_cast<T>(c.dsdx)), std::abs(static_cast<T>(c.dsdy)));
	T dt = std::max(std::abs(static_cast<T>(c.dtdx)), std::abs(static_cast<T>(c.dtdy)));
	ds *= T(1.5);
	dt *= T(1.5);
	return T(0.5) - pt_detail::checker_bf(s, ds) * pt_detail::checker_bf(t_, dt) / T(2);
}
