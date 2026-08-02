#pragma once
// ---------------------------------------------------------------------------
// procedural_textures.h -- CPU/GPU procedural texture library
//
// Direct ports of pbrt-v4 textures.h / textures.cpp (non-image textures).
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// Apache License, Version 2.0.
//
// Provides (all templated on scalar type T = float or double):
//
//   Utility functions:
//     checkerboard_weight<T>     -- anti-aliased filtered checkerboard weight
//     inside_polka_dot<T>        -- polka-dot cell test (Noise-based)
//
//   Texture types (each has a Map2D or Map3D variant plus CPU_GPU Evaluate):
//     ConstantTexture<T>         -- constant scalar value
//     BilerpTexture<T>           -- bilinear blend of 4 corner values
//     CheckerboardTexture<T>     -- anti-aliased 2D or 3D checkerboard
//     MixTexture<T>              -- lerp between two child textures
//     ScaledTexture<T>           -- multiply child texture by a scalar
//     FBmTexture<T>              -- fractional Brownian motion
//     WrinkledTexture<T>         -- turbulence-based wrinkle texture
//     WindyTexture<T>            -- two-frequency FBm wind ripples
//     DotsTexture<T>             -- polka-dot pattern
//     MarbleTexture<T>           -- procedural marble (FBm + cubic spline)
//
// Design rules (same as sampling.h, noise.h):
//   - Header-only, no heap allocation, no virtual functions
//   - CPU_GPU tagged (expands to __host__ __device__ __forceinline__ on GPU,
//     inline on CPU)
//   - All textures take a TextureEvalContext and return T
//   - Child textures are passed as std::function<T(const TextureEvalContext&)>
//     so callers can compose arbitrarily
//
// Dependencies (all in src/shared/):
//   texture_mapping.h  -- TextureEvalContext, TexCoord2D/3D, TextureMapping2D/3D
//   noise.h            -- perlin_noise<T>, fbm<T>, turbulence<T>
// ---------------------------------------------------------------------------

#include "texture_mapping.h"
#include "noise.h"

#include <cmath>
#include <functional>
#include <algorithm>

// ===========================================================================
// Internal helpers (mirroring pbrt-v4 internal functions)
// ===========================================================================
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

// Evaluate a cubic Bezier segment at t in [0,1] using de Casteljau.
// pbrt-v4: EvaluateCubicBezier used in MarbleTexture::Evaluate
template<typename T>
CPU_GPU void bezier3(T t,
	T p0x, T p0y, T p0z,
	T p1x, T p1y, T p1z,
	T p2x, T p2y, T p2z,
	T p3x, T p3y, T p3z,
	T& rx, T& ry, T& rz)
{
	T u = T(1) - t;
	T w0 = u*u*u;
	T w1 = T(3)*u*u*t;
	T w2 = T(3)*u*t*t;
	T w3 = t*t*t;
	rx = w0*p0x + w1*p1x + w2*p2x + w3*p3x;
	ry = w0*p0y + w1*p1y + w2*p2y + w3*p3y;
	rz = w0*p0z + w1*p1z + w2*p2z + w3*p3z;
}

} // namespace pt_detail

// ===========================================================================
// checkerboard_weight
//
// Returns weight in [0,1] blending between two checkerboard cells.
// Uses anti-aliased box-filtered integration (pbrt-v4 Checkerboard()).
// Pass use_2d=true and a valid 2D coord, or use_2d=false for 3D.
// ===========================================================================

// 2D variant: takes TexCoord2D
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

// 3D variant: takes TexCoord3D
template<typename T>
CPU_GPU T checkerboard_weight_3d(const TexCoord3D& c) {
	T dx = T(1.5) * std::max(std::abs(static_cast<T>(c.dpdx.x)), std::abs(static_cast<T>(c.dpdy.x)));
	T dy = T(1.5) * std::max(std::abs(static_cast<T>(c.dpdx.y)), std::abs(static_cast<T>(c.dpdy.y)));
	T dz = T(1.5) * std::max(std::abs(static_cast<T>(c.dpdx.z)), std::abs(static_cast<T>(c.dpdy.z)));
	return T(0.5) - T(0.5) *
		pt_detail::checker_bf(static_cast<T>(c.p.x), dx) *
		pt_detail::checker_bf(static_cast<T>(c.p.y), dy) *
		pt_detail::checker_bf(static_cast<T>(c.p.z), dz);
}

// ===========================================================================
// inside_polka_dot
//
// Returns true if the 2D texture coordinate lies inside a noise-scattered
// polka dot cell.  pbrt-v4: InsidePolkaDot(Point2f st)
// Note: pbrt-v4 Noise(x, y) defaults z=0.5f; we match that here.
// ===========================================================================
template<typename T>
CPU_GPU bool inside_polka_dot(T s, T t_) {
	int sCell = static_cast<int>(std::floor(s + T(0.5)));
	int tCell = static_cast<int>(std::floor(t_ + T(0.5)));

	if (perlin_noise<T>(static_cast<T>(sCell) + T(0.5),
						static_cast<T>(tCell) + T(0.5),
						T(0.5)) > T(0))
	{
		T radius    = T(0.35);
		T maxShift  = T(0.5) - radius;
		T sCenter   = static_cast<T>(sCell) + maxShift *
			perlin_noise<T>(static_cast<T>(sCell) + T(1.5),
							static_cast<T>(tCell) + T(2.8),
							T(0.5));
		T tCenter   = static_cast<T>(tCell) + maxShift *
			perlin_noise<T>(static_cast<T>(sCell) + T(4.5),
							static_cast<T>(tCell) + T(9.8),
							T(0.5));
		T ds = s - sCenter, dt = t_ - tCenter;
		if (ds*ds + dt*dt < radius*radius)
			return true;
	}
	return false;
}

// ===========================================================================
// ConstantTexture<T>
//
// Returns a fixed scalar value regardless of surface point.
// pbrt-v4: FloatConstantTexture
// ===========================================================================
template<typename T>
struct ConstantTexture {
	T value = T(0);

	CPU_GPU T Evaluate(const TextureEvalContext& /*ctx*/) const {
		return value;
	}
};

// ===========================================================================
// BilerpTexture<T>
//
// Bilinear interpolation of four corner values over the unit square.
// pbrt-v4: FloatBilerpTexture
//   v00 = corner at (s=0,t=0), v10 = (1,0), v01 = (0,1), v11 = (1,1)
// ===========================================================================
template<typename T>
struct BilerpTexture {
	TextureMapping2D mapping = UVMapping(1.f, 1.f, 0.f, 0.f);
	T v00 = T(0), v01 = T(0), v10 = T(0), v11 = T(0);

	CPU_GPU T Evaluate(const TextureEvalContext& ctx) const {
		TexCoord2D c = mapping.Map(ctx);
		T s = static_cast<T>(c.st.x);
		T t_ = static_cast<T>(c.st.y);
		return (T(1)-s)*(T(1)-t_)*v00 + s*(T(1)-t_)*v10 +
			   (T(1)-s)*t_*v01       + s*t_*v11;
	}
};

// ===========================================================================
// CheckerboardTexture<T>
//
// Anti-aliased 2D checkerboard.  The weight returned by checkerboard_weight_2d
// smoothly blends between the two child textures when a filter footprint
// straddles a boundary.  pbrt-v4: FloatCheckerboardTexture (2D path).
//
// Child textures are plain T scalars; for function-based children use
// CheckerboardFnTexture below.
// ===========================================================================
template<typename T>
struct CheckerboardTexture {
	TextureMapping2D mapping = UVMapping(1.f, 1.f, 0.f, 0.f);
	T tex0 = T(0);   // value for "white" cells
	T tex1 = T(1);   // value for "black" cells

	CPU_GPU T Evaluate(const TextureEvalContext& ctx) const {
		TexCoord2D c = mapping.Map(ctx);
		T w = checkerboard_weight_2d<T>(c);
		return (T(1) - w) * tex0 + w * tex1;
	}
};

// 3D variant
template<typename T>
struct Checkerboard3DTexture {
	TextureMapping3D mapping = PointTransformMapping(TextureTransform{});
	T tex0 = T(0);
	T tex1 = T(1);

	CPU_GPU T Evaluate(const TextureEvalContext& ctx) const {
		TexCoord3D c = mapping.Map(ctx);
		T w = checkerboard_weight_3d<T>(c);
		return (T(1) - w) * tex0 + w * tex1;
	}
};

// ===========================================================================
// MixTexture<T>
//
// Linear interpolation between two constant child values by a scalar amount.
// pbrt-v4: FloatMixTexture
// For function-based children use MixFnTexture (see below).
// ===========================================================================
template<typename T>
struct MixTexture {
	T tex0 = T(0);
	T tex1 = T(1);
	T amount = T(0.5);  // blend weight; 0 -> tex0, 1 -> tex1

	CPU_GPU T Evaluate(const TextureEvalContext& /*ctx*/) const {
		return (T(1) - amount) * tex0 + amount * tex1;
	}
};

// ===========================================================================
// ScaledTexture<T>
//
// Multiplies a child texture value by a constant scale factor.
// pbrt-v4: FloatScaledTexture
// ===========================================================================
template<typename T>
struct ScaledTexture {
	T child_value = T(1);   // child texture evaluated to a constant for simplicity
	T scale       = T(1);

	CPU_GPU T Evaluate(const TextureEvalContext& /*ctx*/) const {
		return child_value * scale;
	}
};

// ===========================================================================
// FBmTexture<T>
//
// Fractional Brownian motion noise texture.
// pbrt-v4: FBmTexture
// ===========================================================================
template<typename T>
struct FBmTexture {
	TextureMapping3D mapping = PointTransformMapping(TextureTransform{});
	T     omega    = T(0.5);
	int   octaves  = 8;

	CPU_GPU T Evaluate(const TextureEvalContext& ctx) const {
		TexCoord3D c = mapping.Map(ctx);
		T dxX = static_cast<T>(c.dpdx.x), dxY = static_cast<T>(c.dpdx.y), dxZ = static_cast<T>(c.dpdx.z);
		T dyX = static_cast<T>(c.dpdy.x), dyY = static_cast<T>(c.dpdy.y), dyZ = static_cast<T>(c.dpdy.z);
		T dpdx_len2 = dxX*dxX + dxY*dxY + dxZ*dxZ;
		T dpdy_len2 = dyX*dyX + dyY*dyY + dyZ*dyZ;
		return fbm<T>(
			static_cast<T>(c.p.x), static_cast<T>(c.p.y), static_cast<T>(c.p.z),
			dpdx_len2, dpdy_len2, omega, octaves);
	}
};

// ===========================================================================
// WrinkledTexture<T>
//
// Turbulence (absolute value of FBm) noise -- looks like wrinkled paper.
// pbrt-v4: WrinkledTexture
// ===========================================================================
template<typename T>
struct WrinkledTexture {
	TextureMapping3D mapping = PointTransformMapping(TextureTransform{});
	T   omega   = T(0.5);
	int octaves = 8;

	CPU_GPU T Evaluate(const TextureEvalContext& ctx) const {
		TexCoord3D c = mapping.Map(ctx);
		T dxX = static_cast<T>(c.dpdx.x), dxY = static_cast<T>(c.dpdx.y), dxZ = static_cast<T>(c.dpdx.z);
		T dyX = static_cast<T>(c.dpdy.x), dyY = static_cast<T>(c.dpdy.y), dyZ = static_cast<T>(c.dpdy.z);
		T dpdx_len2 = dxX*dxX + dxY*dxY + dxZ*dxZ;
		T dpdy_len2 = dyX*dyX + dyY*dyY + dyZ*dyZ;
		return turbulence<T>(
			static_cast<T>(c.p.x), static_cast<T>(c.p.y), static_cast<T>(c.p.z),
			dpdx_len2, dpdy_len2, omega, octaves);
	}
};

// ===========================================================================
// WindyTexture<T>
//
// Two-frequency FBm simulating wind-disturbed water ripples.
// pbrt-v4: WindyTexture
//   windStrength = FBm(p*0.1, ..., 0.5, 3)
//   waveHeight   = FBm(p,     ..., 0.5, 6)
//   result = |windStrength| * waveHeight
// ===========================================================================
template<typename T>
struct WindyTexture {
	TextureMapping3D mapping = PointTransformMapping(TextureTransform{});

	CPU_GPU T Evaluate(const TextureEvalContext& ctx) const {
		TexCoord3D c = mapping.Map(ctx);
		T px = static_cast<T>(c.p.x),    py = static_cast<T>(c.p.y),    pz = static_cast<T>(c.p.z);
		T dxX = static_cast<T>(c.dpdx.x), dxY = static_cast<T>(c.dpdx.y), dxZ = static_cast<T>(c.dpdx.z);
		T dyX = static_cast<T>(c.dpdy.x), dyY = static_cast<T>(c.dpdy.y), dyZ = static_cast<T>(c.dpdy.z);

		// Low-frequency wind: scale point by 0.1, scale footprint by 0.01
		T wind_dpdx_len2 = T(0.01)*(dxX*dxX + dxY*dxY + dxZ*dxZ);
		T wind_dpdy_len2 = T(0.01)*(dyX*dyX + dyY*dyY + dyZ*dyZ);
		T windStrength = fbm<T>(T(0.1)*px, T(0.1)*py, T(0.1)*pz,
								wind_dpdx_len2, wind_dpdy_len2, T(0.5), 3);

		T dpdx_len2 = dxX*dxX + dxY*dxY + dxZ*dxZ;
		T dpdy_len2 = dyX*dyX + dyY*dyY + dyZ*dyZ;
		T waveHeight = fbm<T>(px, py, pz, dpdx_len2, dpdy_len2, T(0.5), 6);

		return std::abs(windStrength) * waveHeight;
	}
};

// ===========================================================================
// DotsTexture<T>
//
// Polka-dot pattern: returns inside_val where Noise-scattered dots appear,
// outside_val elsewhere.
// pbrt-v4: FloatDotsTexture
// ===========================================================================
template<typename T>
struct DotsTexture {
	TextureMapping2D mapping = UVMapping(1.f, 1.f, 0.f, 0.f);
	T outside_val = T(0);
	T inside_val  = T(1);

	CPU_GPU T Evaluate(const TextureEvalContext& ctx) const {
		TexCoord2D c = mapping.Map(ctx);
		return inside_polka_dot<T>(static_cast<T>(c.st.x), static_cast<T>(c.st.y))
			   ? inside_val
			   : outside_val;
	}
};

// ===========================================================================
// MarbleTexture<T>
//
// Procedural marble: FBm-perturbed sine wave mapped through a hardcoded RGB
// cubic-Bezier spline, returning a single luminance (greyscale) value.
//
// pbrt-v4: MarbleTexture::Evaluate (textures.cpp)
// The original returns SampledSpectrum; here we return luminance T:
//   lum = 0.2126*r + 0.7152*g + 0.0722*b  (sRGB)
// ===========================================================================
template<typename T>
struct MarbleTexture {
	TextureMapping3D mapping = PointTransformMapping(TextureTransform{});
	int octaves   = 8;
	T   omega     = T(0.5);
	T   scale_    = T(1);
	T   variation = T(0.2);

	CPU_GPU T Evaluate(const TextureEvalContext& ctx) const {
		TexCoord3D c = mapping.Map(ctx);
		T px = static_cast<T>(c.p.x) * scale_;
		T py = static_cast<T>(c.p.y) * scale_;
		T pz = static_cast<T>(c.p.z) * scale_;
		T dx = static_cast<T>(c.dpdx.x) * scale_;
		T dy = static_cast<T>(c.dpdx.y) * scale_;
		T dz = static_cast<T>(c.dpdx.z) * scale_;
		T ex = static_cast<T>(c.dpdy.x) * scale_;
		T ey = static_cast<T>(c.dpdy.y) * scale_;
		T ez = static_cast<T>(c.dpdy.z) * scale_;

		T dpdx_len2 = dx*dx + dy*dy + dz*dz;
		T dpdy_len2 = ex*ex + ey*ey + ez*ez;
		T marble = py + variation * fbm<T>(px, py, pz, dpdx_len2, dpdy_len2, omega, octaves);
		T t = T(0.5) + T(0.5) * std::sin(marble);

		// Marble color spline (9 RGB control points from pbrt-v4)
		// Each row: {r, g, b}
		constexpr int kN = 9;
		static const float colors[kN][3] = {
			{.58f,.58f,.6f}, {.58f,.58f,.6f}, {.58f,.58f,.6f},
			{.5f, .5f, .5f}, {.6f,.59f,.58f}, {.58f,.58f,.6f},
			{.58f,.58f,.6f}, {.2f,.2f,.33f},  {.58f,.58f,.6f}
		};

		int nSeg = kN - 3;  // 6 cubic-Bezier segments
		int first = std::min(static_cast<int>(std::floor(t * static_cast<T>(nSeg))), nSeg - 1);
		T seg_t = t * static_cast<T>(nSeg) - static_cast<T>(first);

		T r, g, b;
		pt_detail::bezier3(seg_t,
			static_cast<T>(colors[first][0]),   static_cast<T>(colors[first][1]),   static_cast<T>(colors[first][2]),
			static_cast<T>(colors[first+1][0]), static_cast<T>(colors[first+1][1]), static_cast<T>(colors[first+1][2]),
			static_cast<T>(colors[first+2][0]), static_cast<T>(colors[first+2][1]), static_cast<T>(colors[first+2][2]),
			static_cast<T>(colors[first+3][0]), static_cast<T>(colors[first+3][1]), static_cast<T>(colors[first+3][2]),
			r, g, b);

		// Scale by 1.5 (pbrt-v4 applies 1.5 factor)
		r *= T(1.5); g *= T(1.5); b *= T(1.5);

		// Return sRGB luminance
		return T(0.2126)*r + T(0.7152)*g + T(0.0722)*b;
	}
};

// ===========================================================================
// Function-based wrappers
//
// For composing textures as callable objects (std::function).
// These mirror how pbrt-v4 passes FloatTexture / SpectrumTexture as
// tagged-pointer handles.  Using std::function keeps us dependency-free.
// ===========================================================================

// A FloatTextureFn takes a TextureEvalContext and returns float.
using FloatTextureFn = std::function<float(const TextureEvalContext&)>;

// MixFnTexture -- lerp between two callable child textures
struct MixFnTexture {
	FloatTextureFn tex0;
	FloatTextureFn tex1;
	FloatTextureFn amount;  // callable returning blend weight

	float Evaluate(const TextureEvalContext& ctx) const {
		float amt = amount(ctx);
		float t0 = (amt != 1.f) ? tex0(ctx) : 0.f;
		float t1 = (amt != 0.f) ? tex1(ctx) : 0.f;
		return (1.f - amt)*t0 + amt*t1;
	}
};

// ScaledFnTexture -- multiply callable child by callable scale
struct ScaledFnTexture {
	FloatTextureFn tex;
	FloatTextureFn scale;

	float Evaluate(const TextureEvalContext& ctx) const {
		float sc = scale(ctx);
		return (sc == 0.f) ? 0.f : tex(ctx) * sc;
	}
};

// DirectionMixFnTexture -- mix by |dot(normal, dir)|
// pbrt-v4: FloatDirectionMixTexture
struct DirectionMixFnTexture {
	FloatTextureFn tex0;
	FloatTextureFn tex1;
	TxVector3f dir = {0,0,1};

	float Evaluate(const TextureEvalContext& ctx) const {
		float nx = ctx.n.x, ny = ctx.n.y, nz = ctx.n.z;
		float amt = std::abs(nx*dir.x + ny*dir.y + nz*dir.z);
		float t0 = (amt != 0.f) ? tex0(ctx) : 0.f;
		float t1 = (amt != 1.f) ? tex1(ctx) : 0.f;
		return amt*t0 + (1.f - amt)*t1;
	}
};
