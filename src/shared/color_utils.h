// color_utils.h -- Color conversion utilities, ported from pbrt-v4 util/color.h
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// Apache License, Version 2.0.
//
// Provides:
//   LinearToSRGB(float)         -- linear -> sRGB (IEC 61966-2-1)
//   SRGBToLinear(float)         -- sRGB -> linear
//   LinearToSRGB8(float, dither) -- linear -> uint8 with optional dither
//   SRGB8ToLinear(uint8_t)      -- uint8 sRGB -> linear (LUT)
//   XYZ struct                  -- CIE XYZ tristimulus type with operators
//   WhiteBalance(srcWhite, dstWhite, out3x3) -- Bradford chromatic adaptation
//
// Note: tone_map.h contains the same LinearToSRGB for the write pipeline.
// color_utils.h is the canonical shared version aligned with pbrt-v4.

#pragma once

#include "scalar_math.h"
#include "square_matrix.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#ifndef CPU_GPU
#  ifdef __CUDACC__
#    define CPU_GPU __host__ __device__
#  else
#    define CPU_GPU
#  endif
#endif

// ---------------------------------------------------------------------------
// LinearToSRGB -- linear float -> sRGB float in [0, 1]
// Uses the same minimax polynomial approximation as pbrt-v4 (from enoki).
// Reference: pbrt-v4 util/color.h LinearToSRGB
// ---------------------------------------------------------------------------
CPU_GPU inline float LinearToSRGB(float value) {
	if (value <= 0.0031308f)
		return 12.92f * value;
	float sqrtValue = std::sqrt(std::max(0.f, value));
	float p = EvaluatePolynomial(sqrtValue,
		-0.0016829072605308378f,  0.03453868659826638f,
		 0.7642611304733891f,     2.0041169284241644f,
		 0.7551545191665577f,    -0.016202083165206348f);
	float q = EvaluatePolynomial(sqrtValue,
		 4.178892964897981e-7f,  -0.00004375359692957097f,
		 0.03467195408529984f,    0.6085338522168684f,
		 1.8970238036421054f,     1.f);
	return (p / q) * value;
}

// ---------------------------------------------------------------------------
// LinearToSRGB8 -- linear float -> uint8 sRGB (0..255), with optional dither
// Reference: pbrt-v4 util/color.h LinearToSRGB8
// ---------------------------------------------------------------------------
CPU_GPU inline uint8_t LinearToSRGB8(float value, float dither = 0.f) {
	if (value <= 0.f) return 0;
	if (value >= 1.f) return 255;
	float encoded = LinearToSRGB(value);
	float v = 255.f * encoded + dither + 0.5f;
	if (v < 0.f)   return 0;
	if (v > 255.f) return 255;
	return static_cast<uint8_t>(v);
}

// ---------------------------------------------------------------------------
// SRGBToLinear -- sRGB float -> linear float
// Reference: pbrt-v4 util/color.h SRGBToLinear
// ---------------------------------------------------------------------------
CPU_GPU inline float SRGBToLinear(float value) {
	if (value <= 0.04045f)
		return value * (1.f / 12.92f);
	float p = EvaluatePolynomial(value,
		-0.0163933279112946f,  -0.7386328024653209f,
	   -11.199318357635072f,  -47.46726633009393f,
	   -36.04572663838034f);
	float q = EvaluatePolynomial(value,
		-0.004261480793199332f, -19.140923959601675f,
	   -59.096406619244426f,   -18.225745396846637f,
		 1.f);
	return (p / q) * value;
}

// ---------------------------------------------------------------------------
// SRGB8ToLinear -- uint8 sRGB -> linear float (precomputed LUT)
// Reference: pbrt-v4 util/color.h SRGB8ToLinear + SRGBToLinearLUT
// ---------------------------------------------------------------------------
namespace color_detail {
// Lazily initialised LUT for SRGB8ToLinear.
inline const float* GetSRGB8ToLinearLUT() {
	static float lut[256];
	static bool initialised = false;
	if (!initialised) {
		for (int i = 0; i < 256; ++i)
			lut[i] = SRGBToLinear(i / 255.f);
		initialised = true;
	}
	return lut;
}
} // namespace color_detail

inline float SRGB8ToLinear(uint8_t value) {
	return color_detail::GetSRGB8ToLinearLUT()[value];
}

// ---------------------------------------------------------------------------
// XYZ -- CIE XYZ tristimulus type
// Reference: pbrt-v4 util/color.h class XYZ
// ---------------------------------------------------------------------------
struct XYZ {
	float X = 0.f, Y = 0.f, Z = 0.f;

	CPU_GPU XYZ() = default;
	CPU_GPU XYZ(float X, float Y, float Z) : X(X), Y(Y), Z(Z) {}

	CPU_GPU float Average() const { return (X + Y + Z) / 3.f; }

	// Chromaticity coordinates (xy).
	CPU_GPU float x() const { return X / (X + Y + Z); }
	CPU_GPU float y() const { return Y / (X + Y + Z); }

	// Construct from chromaticity + luminance.
	CPU_GPU static XYZ FromxyY(float cx, float cy, float lum = 1.f) {
		if (cy == 0.f) return XYZ(0.f, 0.f, 0.f);
		return XYZ(cx * lum / cy, lum, (1.f - cx - cy) * lum / cy);
	}

	CPU_GPU XYZ operator+(const XYZ& o) const { return {X+o.X, Y+o.Y, Z+o.Z}; }
	CPU_GPU XYZ operator-(const XYZ& o) const { return {X-o.X, Y-o.Y, Z-o.Z}; }
	CPU_GPU XYZ operator*(float a)       const { return {X*a, Y*a, Z*a}; }
	CPU_GPU XYZ operator/(float a)       const { return {X/a, Y/a, Z/a}; }
	CPU_GPU XYZ& operator+=(const XYZ& o) { X+=o.X; Y+=o.Y; Z+=o.Z; return *this; }
	CPU_GPU XYZ& operator*=(float a)      { X*=a;  Y*=a;  Z*=a;  return *this; }
	CPU_GPU bool operator==(const XYZ& o) const { return X==o.X && Y==o.Y && Z==o.Z; }
	CPU_GPU bool operator!=(const XYZ& o) const { return !(*this == o); }
	CPU_GPU float operator[](int i) const { return i==0 ? X : i==1 ? Y : Z; }
	CPU_GPU float& operator[](int i) { return i==0 ? X : i==1 ? Y : Z; }

	friend CPU_GPU XYZ operator*(float a, const XYZ& x) { return x * a; }
};

CPU_GPU inline XYZ Lerp(float t, const XYZ& a, const XYZ& b) {
	return a * (1.f - t) + b * t;
}

// ---------------------------------------------------------------------------
// WhiteBalance -- Bradford chromatic adaptation matrix
// Returns a 3×3 matrix M such that M * XYZ_src_white ≈ XYZ_dst_white.
// srcWhite / dstWhite are CIE xy chromaticity coordinates.
// Reference: pbrt-v4 util/color.h WhiteBalance
// ---------------------------------------------------------------------------
inline SquareMatrix<3> WhiteBalance(float srcX, float srcY,
									 float dstX, float dstY) {
	// Bradford LMS-from-XYZ matrix (row-major)
	const SquareMatrix<3> LMSFromXYZ(
		 0.8951f,  0.2664f, -0.1614f,
		-0.7502f,  1.7135f,  0.0367f,
		 0.0389f, -0.0685f,  1.0296f);
	const SquareMatrix<3> XYZFromLMS(
		 0.986993f,   -0.147054f,  0.159963f,
		 0.432305f,    0.51836f,   0.0492912f,
		-0.00852866f,  0.0400428f, 0.968487f);

	XYZ srcXYZ = XYZ::FromxyY(srcX, srcY);
	XYZ dstXYZ = XYZ::FromxyY(dstX, dstY);

	// Transform to LMS
	float sLMS[3] = {
		LMSFromXYZ[0][0]*srcXYZ.X + LMSFromXYZ[0][1]*srcXYZ.Y + LMSFromXYZ[0][2]*srcXYZ.Z,
		LMSFromXYZ[1][0]*srcXYZ.X + LMSFromXYZ[1][1]*srcXYZ.Y + LMSFromXYZ[1][2]*srcXYZ.Z,
		LMSFromXYZ[2][0]*srcXYZ.X + LMSFromXYZ[2][1]*srcXYZ.Y + LMSFromXYZ[2][2]*srcXYZ.Z,
	};
	float dLMS[3] = {
		LMSFromXYZ[0][0]*dstXYZ.X + LMSFromXYZ[0][1]*dstXYZ.Y + LMSFromXYZ[0][2]*dstXYZ.Z,
		LMSFromXYZ[1][0]*dstXYZ.X + LMSFromXYZ[1][1]*dstXYZ.Y + LMSFromXYZ[1][2]*dstXYZ.Z,
		LMSFromXYZ[2][0]*dstXYZ.X + LMSFromXYZ[2][1]*dstXYZ.Y + LMSFromXYZ[2][2]*dstXYZ.Z,
	};

	// Diagonal correction matrix
	SquareMatrix<3> LMScorrect = SquareMatrix<3>::Diag(
		sLMS[0] != 0.f ? dLMS[0] / sLMS[0] : 1.f,
		sLMS[1] != 0.f ? dLMS[1] / sLMS[1] : 1.f,
		sLMS[2] != 0.f ? dLMS[2] / sLMS[2] : 1.f);

	return XYZFromLMS * LMScorrect * LMSFromXYZ;
}
