#pragma once
// ---------------------------------------------------------------------------
// spectral_math.h -- SampledSpectrum-to-XYZ/RGB conversion utilities
//
// Ported from pbrt-v4 src/pbrt/util/spectrum.cpp (Apache-2.0)
// Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
//
// Provides free functions that extend SampledSpectrum<N> with CIE-based
// colour conversions. These are separate from sampled_spectrum.h to avoid
// circular includes (cie_data.h -> spectrum_types.h -> sampled_spectrum.h).
//
// Provides:
//   ToXYZ(ss, swl)        -- SampledSpectrum -> XYZ  (pbrt-v4: SampledSpectrum::ToXYZ)
//   LuminanceY(ss, swl)   -- SampledSpectrum -> float Y  (pbrt-v4: SampledSpectrum::y)
//   ToRGB(ss, swl, cs)    -- SampledSpectrum -> float[3] RGB  (pbrt-v4: SampledSpectrum::ToRGB)
//
// Algorithm (pbrt-v4 spectrum.cpp lines 205-227):
//   X_s = GetCIE_X().Sample(lambda)
//   Y_s = GetCIE_Y().Sample(lambda)
//   Z_s = GetCIE_Z().Sample(lambda)
//   pdf = lambda.PDF()
//   X = SafeDiv(X_s * ss, pdf).Average() / CIE_Y_integral
//   Y = SafeDiv(Y_s * ss, pdf).Average() / CIE_Y_integral
//   Z = SafeDiv(Z_s * ss, pdf).Average() / CIE_Y_integral
//   ToRGB: cs.FromXYZ(X,Y,Z, r,g,b)
//
// Dependencies:
//   sampled_spectrum.h  -- SampledSpectrum<N>, SampledWavelengths<N>, SafeDiv, kCIE_Y_integral
//   cie_data.h          -- GetCIE_X/Y/Z()
//   rgb_colorspace.h    -- RGBColorSpace, XYZ
// ---------------------------------------------------------------------------

#include "sampled_spectrum.h"
#include "../data/cie_data.h"
#include "rgb_colorspace.h"

// ---------------------------------------------------------------------------
// ToXYZ -- convert a SampledSpectrum to CIE XYZ tristimulus values
//
// pbrt-v4: SampledSpectrum::ToXYZ(const SampledWavelengths& lambda)
//
// Uses the Monte Carlo estimator:
//   X = E[X_bar(lambda) * L(lambda) / p(lambda)] / CIE_Y_integral
// where p(lambda) is the wavelength sampling PDF stored in swl.
// ---------------------------------------------------------------------------
template <int N = 4>
inline XYZ ToXYZ(const SampledSpectrum<N>& ss,
				 const SampledWavelengths<N>& swl) {
	SampledSpectrum<N> X = GetCIE_X().Sample(swl);
	SampledSpectrum<N> Y = GetCIE_Y().Sample(swl);
	SampledSpectrum<N> Z = GetCIE_Z().Sample(swl);
	SampledSpectrum<N> pdf = swl.PDF();
	return XYZ{
		SafeDiv(X * ss, pdf).Average() / kCIE_Y_integral,
		SafeDiv(Y * ss, pdf).Average() / kCIE_Y_integral,
		SafeDiv(Z * ss, pdf).Average() / kCIE_Y_integral
	};
}

// ---------------------------------------------------------------------------
// LuminanceY -- compute CIE Y (luminance) of a SampledSpectrum
//
// pbrt-v4: SampledSpectrum::y(const SampledWavelengths& lambda)
// ---------------------------------------------------------------------------
template <int N = 4>
inline float LuminanceY(const SampledSpectrum<N>& ss,
						const SampledWavelengths<N>& swl) {
	SampledSpectrum<N> Ys  = GetCIE_Y().Sample(swl);
	SampledSpectrum<N> pdf = swl.PDF();
	return SafeDiv(Ys * ss, pdf).Average() / kCIE_Y_integral;
}

// ---------------------------------------------------------------------------
// ToRGB -- convert a SampledSpectrum to RGB in a given colour space
//
// pbrt-v4: SampledSpectrum::ToRGB(const SampledWavelengths& lambda,
//                                  const RGBColorSpace& cs)
//
// Calls ToXYZ() then applies the colorspace XYZ->RGB matrix.
// Outputs are written to r, g, b (not clamped to [0,1]).
// ---------------------------------------------------------------------------
template <int N = 4>
inline void ToRGB(const SampledSpectrum<N>& ss,
			  const SampledWavelengths<N>& swl,
			  const RGBColorSpace& cs,
			  float& r, float& g, float& b) {
	XYZ xyz = ToXYZ(ss, swl);
	cs.FromXYZ(xyz.X, xyz.Y, xyz.Z, r, g, b);
}

// ---------------------------------------------------------------------------
// InnerProduct -- spectral inner product (Riemann sum over visible range)
//
// pbrt-v4: util/spectrum.h InnerProduct(Spectrum f, Spectrum g)
//
//   integral = sum_{lambda=360}^{830} f(lambda) * g(lambda)  (step 1 nm)
//
// Both F and G must provide operator()(float lambda) const.
// ---------------------------------------------------------------------------
template <typename F, typename G>
inline float InnerProduct(const F& f, const G& g) {
	static constexpr float Lambda_min = 360.f;
	static constexpr float Lambda_max = 830.f;
	float integral = 0.f;
	for (float lambda = Lambda_min; lambda <= Lambda_max; ++lambda)
		integral += f(lambda) * g(lambda);
	return integral;
}

// ---------------------------------------------------------------------------
// SpectrumToXYZ -- integrate a spectrum against CIE X/Y/Z basis
//
// pbrt-v4: util/spectrum.cpp SpectrumToXYZ(Spectrum s)
//
//   X = InnerProduct(CIE_X, s) / CIE_Y_integral
//   Y = InnerProduct(CIE_Y, s) / CIE_Y_integral
//   Z = InnerProduct(CIE_Z, s) / CIE_Y_integral
//
// S must provide operator()(float lambda) const.
// ---------------------------------------------------------------------------
template <typename S>
inline XYZ SpectrumToXYZ(const S& s) {
	return XYZ{
		InnerProduct(GetCIE_X(), s) / kCIE_Y_integral,
		InnerProduct(GetCIE_Y(), s) / kCIE_Y_integral,
		InnerProduct(GetCIE_Z(), s) / kCIE_Y_integral
	};
}
