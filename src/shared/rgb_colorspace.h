#pragma once
// ---------------------------------------------------------------------------
// rgb_colorspace.h -- RGB color space definitions with XYZ<->RGB matrices
//
// pbrt-v4 reference: src/pbrt/util/colorspace.h + colorspace.cpp
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// Apache License, Version 2.0.
//
// Provides:
//   RGBColorSpace          -- chromaticity primaries + XYZFromRGB / RGBFromXYZ
//   RGBColorSpace::sRGB()        -- IEC 61966-2-1 / IEC 61966-2-4
//   RGBColorSpace::DCI_P3()      -- SMPTE EG 432-1 (D65 white point)
//   RGBColorSpace::Rec2020()     -- ITU-R BT.2020
//   RGBColorSpace::ACES2065_1()  -- ACES AP0 (AMPAS S-2014-006)
//   ConvertRGBColorSpace(from, to) -- 3x3 cross-space conversion matrix
//
// Design notes:
//   - Header-only; no allocator, no DenselySampledSpectrum dependency.
//   - Matrices are derived analytically from CIE xy chromaticity primaries
//     using the same algorithm as pbrt-v4 RGBColorSpace::RGBColorSpace().
//   - XYZ type and SquareMatrix<3> are from color_utils.h / square_matrix.h.
//   - Uses float throughout (matching pbrt-v4 Float == float build).
// ---------------------------------------------------------------------------

#include "color_utils.h"            // XYZ, WhiteBalance
#include "square_matrix.h"          // SquareMatrix<3>, Inverse, Mul
#include "rgb_to_spectrum_table.h"  // RGBToSpectrumTable, RGBSigmoidPolynomial

#include <optional>
#include <string>
#include <cstring>

#ifndef CPU_GPU
#  ifdef __CUDACC__
#    define CPU_GPU __host__ __device__
#  else
#    define CPU_GPU
#  endif
#endif

// ---------------------------------------------------------------------------
// RGBColorSpace
// ---------------------------------------------------------------------------
struct RGBColorSpace {
	// CIE xy chromaticity of R, G, B primaries and white point
	float rx, ry;   // red primary
	float gx, gy;   // green primary
	float bx, by;   // blue primary
	float wx, wy;   // white point

	// 3x3 matrices (row-major, double precision matching SquareMatrix<3>)
	// pbrt-v4: XYZFromRGB, RGBFromXYZ
	SquareMatrix<3> XYZFromRGB;
	SquareMatrix<3> RGBFromXYZ;

	// -----------------------------------------------------------------------
	// Factory: build from CIE xy chromaticity primaries + white point
	//
	// Mirrors pbrt-v4 RGBColorSpace::RGBColorSpace():
	//   R = XYZ::FromxyY(r)  (Y=1)
	//   G = XYZ::FromxyY(g)
	//   B = XYZ::FromxyY(b)
	//   W = XYZ::FromxyY(w)
	//   rgb_mat = [R G B]  (column vectors)
	//   C       = Inverse(rgb_mat) * W
	//   XYZFromRGB = rgb_mat * Diag(C)
	//   RGBFromXYZ = Inverse(XYZFromRGB)
	// -----------------------------------------------------------------------
	static RGBColorSpace FromPrimaries(
		float rx_, float ry_,
		float gx_, float gy_,
		float bx_, float by_,
		float wx_, float wy_)
	{
		RGBColorSpace cs;
		cs.rx = rx_; cs.ry = ry_;
		cs.gx = gx_; cs.gy = gy_;
		cs.bx = bx_; cs.by = by_;
		cs.wx = wx_; cs.wy = wy_;

		// Convert xy -> XYZ (Y=1 assumption, pbrt-v4 XYZ::FromxyY)
		auto fromxy = [](float x, float y) -> XYZ {
			return XYZ::FromxyY(x, y);
		};

		XYZ R = fromxy(rx_, ry_);
		XYZ G = fromxy(gx_, gy_);
		XYZ B = fromxy(bx_, by_);
		XYZ W = fromxy(wx_, wy_);

		// Column-major layout: each column is one primary
		// rgb = [ R.X  G.X  B.X ]
		//       [ R.Y  G.Y  B.Y ]
		//       [ R.Z  G.Z  B.Z ]
		SquareMatrix<3> rgb(
			R.X, G.X, B.X,
			R.Y, G.Y, B.Y,
			R.Z, G.Z, B.Z);

		// C = Inverse(rgb) * W  -- scaling to match white point
		auto inv_rgb = Inverse(rgb);
		// inv_rgb * W
		double Cx = inv_rgb.value()[0][0]*W.X + inv_rgb.value()[0][1]*W.Y + inv_rgb.value()[0][2]*W.Z;
		double Cy = inv_rgb.value()[1][0]*W.X + inv_rgb.value()[1][1]*W.Y + inv_rgb.value()[1][2]*W.Z;
		double Cz = inv_rgb.value()[2][0]*W.X + inv_rgb.value()[2][1]*W.Y + inv_rgb.value()[2][2]*W.Z;

		// XYZFromRGB = rgb * Diag(Cx, Cy, Cz)
		cs.XYZFromRGB = rgb * SquareMatrix<3>::Diag(Cx, Cy, Cz);
		cs.RGBFromXYZ = Inverse(cs.XYZFromRGB).value();

		return cs;
	}

	// -----------------------------------------------------------------------
	// Named color space singletons (matching pbrt-v4 Init())
	// -----------------------------------------------------------------------

	// sRGB / IEC 61966-2-1, D65 white point
	// Primaries: r(0.64,0.33) g(0.30,0.60) b(0.15,0.06) w D65(0.3127,0.3290)
	static const RGBColorSpace& sRGB() {
		static RGBColorSpace cs = FromPrimaries(
			0.64f, 0.33f,
			0.30f, 0.60f,
			0.15f, 0.06f,
			0.3127f, 0.3290f);
		return cs;
	}

	// DCI-P3 (SMPTE EG 432-1), D65 white point
	// Primaries: r(0.680,0.320) g(0.265,0.690) b(0.150,0.060)
	static const RGBColorSpace& DCI_P3() {
		static RGBColorSpace cs = FromPrimaries(
			0.680f, 0.320f,
			0.265f, 0.690f,
			0.150f, 0.060f,
			0.3127f, 0.3290f);
		return cs;
	}

	// Rec. 2020 (ITU-R BT.2020), D65 white point
	// Primaries: r(0.708,0.292) g(0.170,0.797) b(0.131,0.046)
	static const RGBColorSpace& Rec2020() {
		static RGBColorSpace cs = FromPrimaries(
			0.708f, 0.292f,
			0.170f, 0.797f,
			0.131f, 0.046f,
			0.3127f, 0.3290f);
		return cs;
	}

	// ACES AP0 (ACES2065-1), ACES D60 white point
	// Primaries: r(0.7347,0.2653) g(0.0,1.0) b(0.0001,-0.077)
	// White: D60 (0.32168, 0.33767)
	static const RGBColorSpace& ACES2065_1() {
		static RGBColorSpace cs = FromPrimaries(
			0.7347f,  0.2653f,
			0.0f,     1.0f,
			0.0001f, -0.077f,
			0.32168f, 0.33767f);
		return cs;
	}

	// -----------------------------------------------------------------------
	// Conversion methods (mirrors pbrt-v4 RGBColorSpace::ToXYZ / ToRGB)
	// -----------------------------------------------------------------------

	// RGB -> XYZ  (pbrt-v4: RGBColorSpace::ToXYZ)
	CPU_GPU XYZ ToXYZ(float r, float g, float b) const {
		double X = XYZFromRGB[0][0]*r + XYZFromRGB[0][1]*g + XYZFromRGB[0][2]*b;
		double Y = XYZFromRGB[1][0]*r + XYZFromRGB[1][1]*g + XYZFromRGB[1][2]*b;
		double Z = XYZFromRGB[2][0]*r + XYZFromRGB[2][1]*g + XYZFromRGB[2][2]*b;
		return XYZ{(float)X, (float)Y, (float)Z};
	}

	// XYZ -> RGB  (pbrt-v4: RGBColorSpace::ToRGB)
	CPU_GPU void FromXYZ(float X, float Y, float Z,
						 float& r, float& g, float& b) const {
		r = (float)(RGBFromXYZ[0][0]*X + RGBFromXYZ[0][1]*Y + RGBFromXYZ[0][2]*Z);
		g = (float)(RGBFromXYZ[1][0]*X + RGBFromXYZ[1][1]*Y + RGBFromXYZ[1][2]*Z);
		b = (float)(RGBFromXYZ[2][0]*X + RGBFromXYZ[2][1]*Y + RGBFromXYZ[2][2]*Z);
	}

	// RGB -> RGBSigmoidPolynomial spectral upsampling coefficients
	// pbrt-v4: RGBColorSpace::ToRGBCoeffs(RGB rgb)
	// Requires sRGB color space (uses RGBToSpectrumTable::sRGB()).
	// Input r,g,b must be clamped to [0,1] (albedo) before calling.
	CPU_GPU RGBSigmoidPolynomial ToRGBCoeffs(float r, float g, float b) const {
		return RGBToSpectrumTable::sRGB()(r, g, b);
	}

	// Luminance coefficients from the second row of XYZFromRGB (Y channel)
	// pbrt-v4: RGBColorSpace::LuminanceVector()
	CPU_GPU void LuminanceVector(float& lr, float& lg, float& lb) const {
		lr = (float)XYZFromRGB[1][0];
		lg = (float)XYZFromRGB[1][1];
		lb = (float)XYZFromRGB[1][2];
	}

	// White point Y=1 XYZ roundtrip: ToXYZ(1,1,1) should equal W scaled
	// Equality check (primaries + white point)
	bool operator==(const RGBColorSpace& o) const {
		return rx==o.rx && ry==o.ry && gx==o.gx && gy==o.gy &&
			   bx==o.bx && by==o.by && wx==o.wx && wy==o.wy;
	}
	bool operator!=(const RGBColorSpace& o) const { return !(*this == o); }
};

// ---------------------------------------------------------------------------
// ConvertRGBColorSpace -- 3x3 matrix to convert RGB from one space to another
//
// pbrt-v4: ConvertRGBColorSpace(from, to) = to.RGBFromXYZ * from.XYZFromRGB
// Usage: out_rgb = M * in_rgb  (where M = ConvertRGBColorSpace(src, dst))
// ---------------------------------------------------------------------------
inline SquareMatrix<3> ConvertRGBColorSpace(const RGBColorSpace& from,
											 const RGBColorSpace& to) {
	if (from == to) return SquareMatrix<3>();  // identity
	// pbrt-v4: to.RGBFromXYZ * from.XYZFromRGB
	return to.RGBFromXYZ * from.XYZFromRGB;
}
