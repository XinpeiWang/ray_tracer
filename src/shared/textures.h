#pragma once
// ---------------------------------------------------------------------------
// textures.h -- pbrt-v4 Texture classes (header-only, CPU)
//
// pbrt-v4 reference: src/pbrt/textures.h
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// Apache License, Version 2.0.
//
// Provides:
//   FloatConstantTexture<T>       -- constant float value
//   RGBConstantTexture<T>         -- constant RGB triple
//   FloatBilerpTexture<T>         -- bilinear interp of 4 float corner values
//   RGBBilerpTexture<T>           -- bilinear interp of 4 RGB corner values
//   FloatScaleTexture<TA,TB,T>    -- multiply two float textures (pbrt-v4 FloatScaledTexture)
//   RGBScaleTexture<TA,TB,T>      -- multiply two RGB textures
//   FloatMixTexture<TA,TB,TW,T>   -- blend two float textures by weight texture
//   RGBMixTexture<TA,TB,TW,T>     -- blend two RGB textures by weight texture
//   UVTexture<T>                  -- debug: output UV as (u, v, 0)
//
// Note: FBmTexture, WrinkledTexture, WindyTexture, DotsTexture, and
//       MarbleTexture are already implemented in procedural_textures.h.
//       FloatImageTexture / RGBImageTexture delegate to mipmap.h; see note
//       below on the ImageTexture types.
//
// All textures expose:
//   evaluate(const TextureEvalContext&) -> float  (Float variants)
//   evaluate(const TextureEvalContext&) -> T[3]   (RGB variants, as std::array<T,3>)
//
// Design notes:
//   - Header-only; templated on scalar T (float or double).
//   - No virtual functions, no heap allocation, no pbrt infrastructure.
//   - Depends on texture_mapping.h (TextureEvalContext, TextureMapping2D,
//     TexCoord2D) which must be included first, or included below.
//   - Mirrors pbrt-v4 evaluate interface: Evaluate(TextureEvalContext ctx).
//
// pbrt-v4 alignment:
//   FloatConstantTexture   <- FloatConstantTexture
//   RGBConstantTexture     <- SpectrumConstantTexture (RGB-only simplified)
//   FloatBilerpTexture     <- FloatBilerpTexture
//   RGBBilerpTexture       <- SpectrumBilerpTexture
//   FloatScaleTexture      <- FloatScaledTexture
//   RGBScaleTexture        <- SpectrumScaledTexture
//   FloatMixTexture        <- FloatMixTexture
//   RGBMixTexture          <- SpectrumMixTexture
//   UVTexture              <- (pbrt-v4 has no UV debug texture; common in other renderers)
// ---------------------------------------------------------------------------

#include "texture_mapping.h"

#include <array>
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// Bilerp helper: bilinear interpolate four values at corners (0,0),(1,0),(0,1),(1,1)
// given normalised (s,t) in [0,1].
// pbrt-v4: Bilerp({s,t}, {v00,v10,v01,v11})
// ---------------------------------------------------------------------------
namespace tex_detail {

template<typename T>
inline T bilerp(T s, T t, T v00, T v10, T v01, T v11) {
	return (T(1)-s)*(T(1)-t)*v00 + s*(T(1)-t)*v10 +
		   (T(1)-s)*t*v01       + s*t*v11;
}

} // namespace tex_detail

// ===========================================================================
// FloatConstantTexture<T>
//
// Returns a constant float value regardless of surface point.
// pbrt-v4: FloatConstantTexture
// ===========================================================================
template<typename T = float>
struct FloatConstantTexture {
	T value = T(0);

	CPU_GPU T Evaluate(const TextureEvalContext& /*ctx*/) const {
		return value;
	}
};

// ===========================================================================
// RGBConstantTexture<T>
//
// Returns a constant RGB triple regardless of surface point.
// pbrt-v4: SpectrumConstantTexture (RGB-only simplified port)
// ===========================================================================
template<typename T = float>
struct RGBConstantTexture {
	std::array<T,3> rgb = {T(0), T(0), T(0)};

	CPU_GPU std::array<T,3> Evaluate(const TextureEvalContext& /*ctx*/) const {
		return rgb;
	}
};

// ===========================================================================
// FloatBilerpTexture<T>
//
// Bilinearly interpolates four float values at the UV corners (0,0),(1,0),(0,1),(1,1).
// pbrt-v4: FloatBilerpTexture
//
// Members:
//   mapping -- TextureMapping2D to get (s,t) coordinates
//   v00,v10,v01,v11 -- corner values: (u=0,v=0), (u=1,v=0), (u=0,v=1), (u=1,v=1)
// ===========================================================================
template<typename T = float>
struct FloatBilerpTexture {
	TextureMapping2D mapping = UVMapping(1.f, 1.f, 0.f, 0.f);
	T v00 = T(0), v10 = T(0), v01 = T(0), v11 = T(0);

	CPU_GPU T Evaluate(const TextureEvalContext& ctx) const {
		TexCoord2D c = mapping.Map(ctx);
		T s = static_cast<T>(c.st.x);
		T t = static_cast<T>(c.st.y);
		return tex_detail::bilerp(s, t, v00, v10, v01, v11);
	}
};

// ===========================================================================
// RGBBilerpTexture<T>
//
// Bilinearly interpolates four RGB values at UV corners.
// pbrt-v4: SpectrumBilerpTexture
//
// Corner layout: v00=(u=0,v=0), v10=(u=1,v=0), v01=(u=0,v=1), v11=(u=1,v=1)
// ===========================================================================
template<typename T = float>
struct RGBBilerpTexture {
	TextureMapping2D mapping = UVMapping(1.f, 1.f, 0.f, 0.f);
	std::array<T,3> v00 = {T(0),T(0),T(0)};
	std::array<T,3> v10 = {T(0),T(0),T(0)};
	std::array<T,3> v01 = {T(0),T(0),T(0)};
	std::array<T,3> v11 = {T(0),T(0),T(0)};

	CPU_GPU std::array<T,3> Evaluate(const TextureEvalContext& ctx) const {
		TexCoord2D c = mapping.Map(ctx);
		T s = static_cast<T>(c.st.x);
		T t = static_cast<T>(c.st.y);
		return {
			tex_detail::bilerp(s, t, v00[0], v10[0], v01[0], v11[0]),
			tex_detail::bilerp(s, t, v00[1], v10[1], v01[1], v11[1]),
			tex_detail::bilerp(s, t, v00[2], v10[2], v01[2], v11[2])
		};
	}
};

// ===========================================================================
// FloatScaleTexture<TexA, TexB, T>
//
// Multiplies two float textures: result = TexA.Evaluate(ctx) * TexB.Evaluate(ctx)
// pbrt-v4: FloatScaledTexture
//
// Template parameters:
//   TexA -- any type with T Evaluate(TextureEvalContext) (the base texture)
//   TexB -- any type with T Evaluate(TextureEvalContext) (the scale texture)
// ===========================================================================
template<typename TexA, typename TexB, typename T = float>
struct FloatScaleTexture {
	TexA tex;
	TexB scale;

	CPU_GPU T Evaluate(const TextureEvalContext& ctx) const {
		// pbrt-v4: evaluate scale first; skip base texture if scale is zero
		T sc = scale.Evaluate(ctx);
		if (sc == T(0)) return T(0);
		return tex.Evaluate(ctx) * sc;
	}
};

// ===========================================================================
// RGBScaleTexture<TexA, TexB, T>
//
// Multiplies an RGB texture by a float scale texture component-wise.
// pbrt-v4: SpectrumScaledTexture
//
//   TexA -- any type with std::array<T,3> Evaluate(TextureEvalContext) (RGB base)
//   TexB -- any type with T              Evaluate(TextureEvalContext) (float scale)
// ===========================================================================
template<typename TexA, typename TexB, typename T = float>
struct RGBScaleTexture {
	TexA tex;
	TexB scale;

	CPU_GPU std::array<T,3> Evaluate(const TextureEvalContext& ctx) const {
		// pbrt-v4: evaluate scale first; skip base texture if scale is zero
		T s = scale.Evaluate(ctx);
		if (s == T(0)) return {T(0), T(0), T(0)};
		auto rgb = tex.Evaluate(ctx);
		return { rgb[0]*s, rgb[1]*s, rgb[2]*s };
	}
};

// ===========================================================================
// FloatMixTexture<TexA, TexB, TexW, T>
//
// Linearly blends two float textures by a weight texture:
//   result = (1-w)*TexA.Evaluate(ctx) + w*TexB.Evaluate(ctx)
// pbrt-v4: FloatMixTexture
//
//   TexA -- any type with T Evaluate(TextureEvalContext) (texture at w=0)
//   TexB -- any type with T Evaluate(TextureEvalContext) (texture at w=1)
//   TexW -- any type with T Evaluate(TextureEvalContext) (blend weight in [0,1])
// ===========================================================================
template<typename TexA, typename TexB, typename TexW, typename T = float>
struct FloatMixTexture {
	TexA tex1;
	TexB tex2;
	TexW amount;

	CPU_GPU T Evaluate(const TextureEvalContext& ctx) const {
		T w = amount.Evaluate(ctx);
		// pbrt-v4: avoid evaluating the unused texture when weight is 0 or 1
		T t1 = T(0), t2 = T(0);
		if (w != T(1)) t1 = tex1.Evaluate(ctx);
		if (w != T(0)) t2 = tex2.Evaluate(ctx);
		return (T(1) - w) * t1 + w * t2;
	}
};

// ===========================================================================
// RGBMixTexture<TexA, TexB, TexW, T>
//
// Linearly blends two RGB textures by a float weight texture.
// pbrt-v4: SpectrumMixTexture
//
//   TexA -- any type with std::array<T,3> Evaluate(TextureEvalContext)
//   TexB -- any type with std::array<T,3> Evaluate(TextureEvalContext)
//   TexW -- any type with T              Evaluate(TextureEvalContext)
// ===========================================================================
template<typename TexA, typename TexB, typename TexW, typename T = float>
struct RGBMixTexture {
	TexA tex1;
	TexB tex2;
	TexW amount;

	CPU_GPU std::array<T,3> Evaluate(const TextureEvalContext& ctx) const {
		T w = amount.Evaluate(ctx);
		// pbrt-v4: avoid evaluating the unused texture when weight is 0 or 1
		std::array<T,3> c1 = {T(0),T(0),T(0)}, c2 = {T(0),T(0),T(0)};
		if (w != T(1)) c1 = tex1.Evaluate(ctx);
		if (w != T(0)) c2 = tex2.Evaluate(ctx);
		return {
			(T(1)-w)*c1[0] + w*c2[0],
			(T(1)-w)*c1[1] + w*c2[1],
			(T(1)-w)*c1[2] + w*c2[2]
		};
	}
};

// ===========================================================================
// UVTexture<T>
//
// Debug texture: outputs (u, v, 0) as an RGB triple so UV parameterisation
// can be visualised directly.  Not in pbrt-v4 (common in other renderers).
//
// Applies the 2D mapping before reading UVs so scale/offset is respected.
// ===========================================================================
template<typename T = float>
struct UVTexture {
	TextureMapping2D mapping = UVMapping(1.f, 1.f, 0.f, 0.f);

	CPU_GPU std::array<T,3> Evaluate(const TextureEvalContext& ctx) const {
		TexCoord2D c = mapping.Map(ctx);
		// Wrap s,t to [0,1] with fmod
		T s = static_cast<T>(c.st.x);
		T t = static_cast<T>(c.st.y);
		// Keep in [0,1] via fmod (matches pbrt-v4 convention: frac part)
		s = s - std::floor(s);
		t = t - std::floor(t);
		return { s, t, T(0) };
	}
};

// ===========================================================================
// Checkerboard band-limited filter helper
//
// Mirrors pbrt-v4 Checkerboard() in textures.cpp.
// Returns a continuous blend weight in [0,1] that antialiases the
// checkerboard boundary using a triangle-filtered box integral of the
// 1D square-wave function.
//
// For the 2D case (used by both CheckerboardTexture variants):
//   ds = 1.5 * max(|dsdx|, |dsdy|)   -- filter footprint in s
//   dt = 1.5 * max(|dtdx|, |dtdy|)   -- filter footprint in t
//   w  = 0.5 - bf(s, ds) * bf(t, dt) / 2
//
// Result w: 0.5 in fully-blended (antialiased) region, 0 or 1 in pure cells.
// The final texture value is: (1-w)*tex1 + w*tex2.
// ===========================================================================
namespace tex_detail {

// Integral of the 1D square-wave antiderivative: d(x) = x/2 + y*(1-2|y|)
// where y = x/2 - floor(x/2) - 0.5
inline float checker_d(float x) {
	float y = x * 0.5f - std::floor(x * 0.5f) - 0.5f;
	return x * 0.5f + y * (1.f - 2.f * std::abs(y));
}

// Band-limited 1D checkerboard weight in footprint [x-r, x+r].
// Returns +1 or -1 if entirely inside one cell, or a blended value.
inline float checker_bf(float x, float r) {
	if (std::floor(x - r) == std::floor(x + r)) {
		// Entirely within one cell -- return binary +1 or -1
		return 1.f - 2.f * (static_cast<int>(std::floor(x)) & 1);
	}
	return (checker_d(x + r) - 2.f * checker_d(x) + checker_d(x - r)) / (r * r);
}

// 2D checkerboard weight derived from texture derivatives.
// Mirrors pbrt-v4: w = 0.5 - bf(s, 1.5*ds) * bf(t, 1.5*dt) / 2
inline float checkerboard_weight(const TexCoord2D& c) {
	float ds = 1.5f * std::max(std::abs(c.dsdx), std::abs(c.dsdy));
	float dt = 1.5f * std::max(std::abs(c.dtdx), std::abs(c.dtdy));
	// Avoid degenerate zero-footprint (point sample fallback)
	if (ds == 0.f && dt == 0.f) {
		int si = static_cast<int>(std::floor(c.st.x));
		int ti = static_cast<int>(std::floor(c.st.y));
		return static_cast<float>((((si + ti) & 1) != 0) ? 1 : 0);
	}
	// Clamp minimum footprint to avoid precision issues at the boundary
	ds = std::max(ds, 1e-6f);
	dt = std::max(dt, 1e-6f);
	return 0.5f - checker_bf(c.st.x, ds) * checker_bf(c.st.y, dt) * 0.5f;
}

} // namespace tex_detail

// ===========================================================================
// CheckerboardTexture<TexA, TexB, T>  (float variant)
//
// 2D checkerboard pattern alternating between two float textures.
// Uses pbrt-v4's band-limited antialiased filter to avoid aliasing at
// high-frequency checkerboard boundaries.
// pbrt-v4: FloatCheckerboardTexture (2D variant)
//
//   tex1 -- "light" cells (even floor(s)+floor(t) sum)
//   tex2 -- "dark"  cells (odd  floor(s)+floor(t) sum)
//
// The blend weight w is computed via checkerboard_weight():
//   w=0 -> pure tex1, w=1 -> pure tex2, 0<w<1 -> antialiased boundary
// ===========================================================================
template<typename TexA, typename TexB, typename T = float>
struct CheckerboardTexture {
	TextureMapping2D mapping = UVMapping(1.f, 1.f, 0.f, 0.f);
	TexA tex1;  // "light" cells (w=0)
	TexB tex2;  // "dark"  cells (w=1)

	CPU_GPU T Evaluate(const TextureEvalContext& ctx) const {
		TexCoord2D c = mapping.Map(ctx);
		float w = tex_detail::checkerboard_weight(c);
		T t1 = T(0), t2 = T(0);
		if (w != 1.f) t1 = tex1.Evaluate(ctx);
		if (w != 0.f) t2 = tex2.Evaluate(ctx);
		return static_cast<T>((1.f - w)) * t1 + static_cast<T>(w) * t2;
	}
};

// ===========================================================================
// RGBCheckerboardTexture<TexA, TexB, T>
//
// 2D checkerboard pattern alternating between two RGB textures.
// Uses pbrt-v4's band-limited antialiased filter.
// pbrt-v4: SpectrumCheckerboardTexture (2D variant)
// ===========================================================================
template<typename TexA, typename TexB, typename T = float>
struct RGBCheckerboardTexture {
	TextureMapping2D mapping = UVMapping(1.f, 1.f, 0.f, 0.f);
	TexA tex1;
	TexB tex2;

	CPU_GPU std::array<T,3> Evaluate(const TextureEvalContext& ctx) const {
		TexCoord2D c = mapping.Map(ctx);
		float w = tex_detail::checkerboard_weight(c);
		std::array<T,3> c1 = {T(0),T(0),T(0)}, c2 = {T(0),T(0),T(0)};
		if (w != 1.f) c1 = tex1.Evaluate(ctx);
		if (w != 0.f) c2 = tex2.Evaluate(ctx);
		T wt = static_cast<T>(w);
		return {
			(T(1)-wt)*c1[0] + wt*c2[0],
			(T(1)-wt)*c1[1] + wt*c2[1],
			(T(1)-wt)*c1[2] + wt*c2[2]
		};
	}
};

// ===========================================================================
// RGBImageTexture<T>
//
// MIPmap-backed image texture with EWA / trilinear filtering.
// Mirrors pbrt-v4 FloatImageTexture / SpectrumImageTexture.
//
// Depends on mipmap.h (include it before this header, or include it here).
// Since mipmap.h depends on rtweekend.h `color` type, this class is
// conditionally compiled only when MIPMAP_H_INCLUDED is defined.
//
// Usage:
//   #include "mipmap.h"          // defines mipmap, color
//   #include "textures.h"        // now RGBImageTexture is available
//
// The mipmap is owned externally; RGBImageTexture holds a const pointer.
// This avoids copying the pyramid on every texture construction.
//
// pbrt-v4 alignment:
//   - mapping   <- TextureMapping2D (same)
//   - scale     <- Float scale member
//   - invert    <- bool invert (returns max(0, 1-v) per channel when true)
//   - flip_y    <- pbrt-v4 flips t: c.st[1] = 1 - c.st[1] for image coords
// ===========================================================================
#if defined(MIPMAP_H_INCLUDED) || defined(MIPMAP_H)

template<typename T = float>
struct RGBImageTexture {
	TextureMapping2D mapping = UVMapping(1.f, 1.f, 0.f, 0.f);
	const mipmap* mip = nullptr;  // externally owned
	T scale   = T(1);
	bool invert = false;

	// Evaluate: look up filtered RGB from the mipmap.
	// Mirrors pbrt-v4 FloatImageTexture::Evaluate() / SpectrumImageTexture::Evaluate().
	std::array<T,3> Evaluate(const TextureEvalContext& ctx) const {
		if (!mip) return {T(0), T(0), T(0)};
		TexCoord2D c = mapping.Map(ctx);
		// pbrt-v4: flip t so (0,0) is lower-left in image space
		float s = c.st.x;
		float t = 1.f - c.st.y;
		// Lookup with EWA derivatives
		color rgb = mip->filter(s, t,
								c.dsdx,  c.dtdx,   // d(st)/dx
								c.dsdy,  c.dtdy);  // d(st)/dy
		std::array<T,3> result = {
			static_cast<T>(rgb.x()) * scale,
			static_cast<T>(rgb.y()) * scale,
			static_cast<T>(rgb.z()) * scale
		};
		if (invert) {
			result[0] = std::max(T(0), T(1) - result[0]);
			result[1] = std::max(T(0), T(1) - result[1]);
			result[2] = std::max(T(0), T(1) - result[2]);
		}
		return result;
	}
};

template<typename T = float>
struct FloatImageTexture {
	TextureMapping2D mapping = UVMapping(1.f, 1.f, 0.f, 0.f);
	const mipmap* mip = nullptr;
	T scale   = T(1);
	bool invert = false;
	// Channel to sample (0=R, 1=G, 2=B); matches common single-channel use
	int channel = 0;

	// Evaluate: look up a single channel from the mipmap.
	// Mirrors pbrt-v4 FloatImageTexture::Evaluate().
	T Evaluate(const TextureEvalContext& ctx) const {
		if (!mip) return T(0);
		TexCoord2D c = mapping.Map(ctx);
		float s = c.st.x;
		float t = 1.f - c.st.y;
		color rgb = mip->filter(s, t, c.dsdx, c.dtdx, c.dsdy, c.dtdy);
		float v = 0.f;
		if      (channel == 0) v = static_cast<float>(rgb.x());
		else if (channel == 1) v = static_cast<float>(rgb.y());
		else                   v = static_cast<float>(rgb.z());
		T result = static_cast<T>(v) * scale;
		return invert ? std::max(T(0), T(1) - result) : result;
	}
};

#endif // MIPMAP_H_INCLUDED / MIPMAP_H
