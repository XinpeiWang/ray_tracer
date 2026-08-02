#pragma once
// ---------------------------------------------------------------------------
// rgb_to_spectrum_table.h -- RGB to SampledSpectrum spectral upsampling table
//
// Ported from pbrt-v4 src/pbrt/util/color.h + color.cpp (Apache-2.0)
// Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
//
// The table encodes a mapping from any RGB triple in [0,1]^3 to a
// RGBSigmoidPolynomial (three coefficients c0,c1,c2) that, when evaluated
// at a wavelength lambda, reconstructs a plausible spectral reflectance.
//
// Algorithm: Jakob & Hanika 2019 "A Low-Dimensional Function Space for
//   Efficient Spectral Upsampling" (EG 2019).
//
// Table layout (matches pbrt-v4 exactly):
//   res = 64
//   Scale[res]              -- nonlinear z-axis knots (smoothstep of smoothstep)
//   Data[3][res][res][res][3] -- trilinear grid of (c0,c1,c2) coefficients
//     index 0: max-component axis (0=R dominant, 1=G dominant, 2=B dominant)
//     index 1: z (scaled by Scale[], maps to max-component value)
//     index 2: y (second-largest component / max  * (res-1))
//     index 3: x (smallest component   / max  * (res-1))
//     index 4: coefficient index (c0, c1, c2)
//
// Provides:
//   RGBToSpectrumTable         -- lookup class (mirrors pbrt-v4)
//   RGBToSpectrumTable::sRGB() -- singleton accessor for sRGB table
//
// Dependencies:
//   color_utils.h   -- RGBSigmoidPolynomial
//   scalar_math.h   -- FindInterval
// ---------------------------------------------------------------------------

#include "color_utils.h"
#include "scalar_math.h"

#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// RGBToSpectrumTable
//
// pbrt-v4: class RGBToSpectrumTable (util/color.h)
//
// Usage:
//   RGBSigmoidPolynomial rsp = RGBToSpectrumTable::sRGB()(rgb_r, rgb_g, rgb_b);
// ---------------------------------------------------------------------------
class RGBToSpectrumTable {
public:
	static constexpr int kRes = 64;

	// CoefficientArray[maxc][z][y][x][coeff]
	using CoefficientArray = float[3][kRes][kRes][kRes][3];

	// Construct from externally-owned zNodes and coeffs arrays.
	// Both must remain valid for the lifetime of the table.
	constexpr RGBToSpectrumTable(const float* zNodes,
								  const CoefficientArray* coeffs)
		: zNodes_(zNodes), coeffs_(coeffs) {}

	// ---------------------------------------------------------------------------
	// operator() -- map clamped RGB in [0,1]^3 to sigmoid polynomial coefficients
	//
	// pbrt-v4: RGBToSpectrumTable::operator()(RGB rgb) const (color.cpp)
	// ---------------------------------------------------------------------------
	CPU_GPU RGBSigmoidPolynomial operator()(float r, float g, float b) const {
		// Handle achromatic (greyscale) input
		// pbrt-v4: (rgb[0] - .5f) / std::sqrt(rgb[0] * (1 - rgb[0]))
		// s() handles ±inf for r=0 or r=1 (returns 0 or 1 respectively).
		if (r == g && g == b) {
			float v = (r - 0.5f) / std::sqrt(r * (1.f - r));
			return RGBSigmoidPolynomial(0.f, 0.f, v);
		}

		// Find max component and compute normalised coordinates
		int maxc = (r > g) ? ((r > b) ? 0 : 2) : ((g > b) ? 1 : 2);
		float rgb[3] = { r, g, b };
		float z = rgb[maxc];
		float x = rgb[(maxc + 1) % 3] * (kRes - 1) / z;
		float y = rgb[(maxc + 2) % 3] * (kRes - 1) / z;

		// Integer indices into the table
		int xi = std::min((int)x, kRes - 2);
		int yi = std::min((int)y, kRes - 2);
		int zi = FindInterval(kRes, [&](int i) { return zNodes_[i] < z; });
		float dx = x - xi;
		float dy = y - yi;
		float dz = (z - zNodes_[zi]) / (zNodes_[zi + 1] - zNodes_[zi]);

		// Trilinearly interpolate all three polynomial coefficients
		float c[3];
		for (int i = 0; i < 3; ++i) {
			auto co = [&](int ddx, int ddy, int ddz) -> float {
				return (*coeffs_)[maxc][zi + ddz][yi + ddy][xi + ddx][i];
			};
			c[i] = Lerp(dz,
						Lerp(dy, Lerp(dx, co(0,0,0), co(1,0,0)),
								 Lerp(dx, co(0,1,0), co(1,1,0))),
						Lerp(dy, Lerp(dx, co(0,0,1), co(1,0,1)),
								 Lerp(dx, co(0,1,1), co(1,1,1))));
		}
		return RGBSigmoidPolynomial(c[0], c[1], c[2]);
	}

	// Singleton accessor for the sRGB table.
	// Data is defined in src/shared/rgb_spectrum_table_data.cpp.
	static const RGBToSpectrumTable& sRGB();

private:
	const float*             zNodes_;
	const CoefficientArray*  coeffs_;
};

// ---------------------------------------------------------------------------
// External declarations for the sRGB table data defined in
// src/shared/rgb_spectrum_table_data.cpp
// ---------------------------------------------------------------------------
extern const float                         sRGBToSpectrumTable_Scale[RGBToSpectrumTable::kRes];
extern const RGBToSpectrumTable::CoefficientArray sRGBToSpectrumTable_Data;

inline const RGBToSpectrumTable& RGBToSpectrumTable::sRGB() {
	static const RGBToSpectrumTable table(sRGBToSpectrumTable_Scale,
										  &sRGBToSpectrumTable_Data);
	return table;
}
