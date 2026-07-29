// color_utils.h -- Color conversion utilities, ported from pbrt-v4 util/color.h
//                  and spectral utilities from pbrt-v4 util/spectrum.h
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
//   Blackbody(lambda_nm, T_K)   -- Planck spectral radiance (pbrt-v4 spectrum.h)
//   BlackbodySpectrum            -- normalized blackbody callable (pbrt-v4 spectrum.h)
//   RGBSigmoidPolynomial         -- spectral upsampling polynomial (pbrt-v4 color.h)
//
// Note: tone_map.h contains the same LinearToSRGB for the write pipeline.
// color_utils.h is the canonical shared version aligned with pbrt-v4.

#pragma once

#include "scalar_math.h"
#include "square_matrix.h"

#include <algorithm>
#include <array>
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
// Thread-safe LUT: C++11 static local init is thread-safe.
inline const std::array<float, 256>& GetSRGB8ToLinearLUT() {
	static const std::array<float, 256> lut = []() {
		std::array<float, 256> t{};
		for (int i = 0; i < 256; ++i)
			t[i] = SRGBToLinear(i / 255.f);
		return t;
	}();
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

	// Transform to LMS via Bradford matrix
	XYZ srcLMS = LMSFromXYZ * srcXYZ;
	XYZ dstLMS = LMSFromXYZ * dstXYZ;

	// Diagonal correction matrix
	SquareMatrix<3> LMScorrect = SquareMatrix<3>::Diag(
		srcLMS[0] != 0.f ? dstLMS[0] / srcLMS[0] : 1.f,
		srcLMS[1] != 0.f ? dstLMS[1] / srcLMS[1] : 1.f,
		srcLMS[2] != 0.f ? dstLMS[2] / srcLMS[2] : 1.f);

	return XYZFromLMS * LMScorrect * LMSFromXYZ;
}

// ---------------------------------------------------------------------------
// Blackbody -- Planck spectral radiance at wavelength lambda_nm and temperature T_K
//
// pbrt-v4 reference: util/spectrum.h Blackbody(Float lambda, Float T)
//
// Returns spectral radiance Le [W / (m^2 · sr · m)] for a blackbody emitter
// at temperature T (Kelvin) and wavelength lambda (nanometres).
// Returns 0 for T <= 0 or lambda <= 0.
//
// Physical constants used (CODATA 2010, matching pbrt-v4):
//   c  = 299 792 458  m/s
//   h  = 6.626 069 57e-34  J·s
//   k_B = 1.380 648 8e-23  J/K
// ---------------------------------------------------------------------------
CPU_GPU inline float Blackbody(float lambda_nm, float T) {
	if (T <= 0.f || lambda_nm <= 0.f) return 0.f;
	const float c  = 299792458.f;
	const float h  = 6.62606957e-34f;
	const float kb = 1.3806488e-23f;
	float l  = lambda_nm * 1e-9f;                             // nm -> m
	float Le = (2.f * h * c * c) /
			   (Pow<5>(l) * (FastExp((h * c) / (l * kb * T)) - 1.f));
	return Le;
}

// ---------------------------------------------------------------------------
// BlackbodySpectrum -- normalized blackbody callable
//
// pbrt-v4 reference: util/spectrum.h BlackbodySpectrum
//
// Wraps Blackbody() and normalises by the peak emission so that the maximum
// value over the visible range is 1.  The peak wavelength is given by Wien's
// displacement law: lambda_max = b / T, b = 2.897 772 1e-3 m·K.
//
// Usage:
//   BlackbodySpectrum bb(6500.f);   // ~D65
//   float v = bb(550.f);            // normalised radiance at 550 nm in [0,1]
// ---------------------------------------------------------------------------
struct BlackbodySpectrum {
	// T_K: colour temperature in Kelvin
	explicit BlackbodySpectrum(float T_K) : T(T_K) {
		// Wien displacement: lambda_max [nm] = 2.897721e-3 / T  (in metres) * 1e9
		float lambdaMax_nm = 2.8977721e-3f / T * 1e9f;
		normalizationFactor = (lambdaMax_nm > 0.f)
			? 1.f / Blackbody(lambdaMax_nm, T)
			: 0.f;
	}

	// Evaluate normalised spectral radiance at wavelength lambda_nm.
	// Return value is in [0, 1] (1.0 at peak wavelength).
	// pbrt-v4: BlackbodySpectrum::operator()(Float lambda)
	CPU_GPU float operator()(float lambda_nm) const {
		return Blackbody(lambda_nm, T) * normalizationFactor;
	}

	// Maximum value is always 1 by construction.
	// pbrt-v4: BlackbodySpectrum::MaxValue()
	CPU_GPU float MaxValue() const { return 1.f; }

  private:
	float T;
	float normalizationFactor;
};

// ---------------------------------------------------------------------------
// RGBSigmoidPolynomial -- spectral upsampling via a quadratic sigmoid
//
// pbrt-v4 reference: util/color.h RGBSigmoidPolynomial
//
// Maps a wavelength lambda to a reflectance in [0,1] using:
//   f(lambda) = s(c0*lambda^2 + c1*lambda + c2)
// where s(x) = 0.5 + x / (2*sqrt(1+x^2))  is the smooth sigmoid in (0,1).
//
// Coefficients (c0, c1, c2) are obtained from the RGBToSpectrumTable lookup
// (not included here) or fit analytically.  Given them, evaluation is O(1).
//
// Usage:
//   RGBSigmoidPolynomial p(c0, c1, c2);
//   float r = p(550.f);     // reflectance at 550 nm
//   float m = p.MaxValue(); // peak reflectance over [360, 830] nm
// ---------------------------------------------------------------------------
struct RGBSigmoidPolynomial {
	RGBSigmoidPolynomial() = default;
	CPU_GPU RGBSigmoidPolynomial(float c0, float c1, float c2)
		: c0(c0), c1(c1), c2(c2) {}

	// Evaluate reflectance at wavelength lambda_nm.
	// pbrt-v4: RGBSigmoidPolynomial::operator()(Float lambda)
	CPU_GPU float operator()(float lambda) const {
		return s(EvaluatePolynomial(lambda, c2, c1, c0));
	}

	// Maximum reflectance over the visible range [360, 830] nm.
	// pbrt-v4: RGBSigmoidPolynomial::MaxValue()
	CPU_GPU float MaxValue() const {
		float result = std::max((*this)(360.f), (*this)(830.f));
		// Vertex of the quadratic: lambda* = -c1 / (2*c0)
		if (c0 != 0.f) {
			float lambdaStar = -c1 / (2.f * c0);
			if (lambdaStar >= 360.f && lambdaStar <= 830.f)
				result = std::max(result, (*this)(lambdaStar));
		}
		return result;
	}

  private:
	// Smooth sigmoid: s(x) = 0.5 + x/(2*sqrt(1+x^2)), maps R -> (0,1).
	// pbrt-v4: RGBSigmoidPolynomial::s(Float x)
	CPU_GPU static float s(float x) {
		if (std::isinf(x)) return x > 0.f ? 1.f : 0.f;
		return 0.5f + x / (2.f * std::sqrt(1.f + Sqr(x)));
	}

	float c0 = 0.f, c1 = 0.f, c2 = 0.f;
};
