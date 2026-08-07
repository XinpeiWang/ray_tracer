#pragma once
// ---------------------------------------------------------------------------
// pixel_sensor.h -- Camera sensor spectral response model
//
// Ported from pbrt-v4 src/pbrt/film.h  (Apache-2.0)
// Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// PhysLight code contributed by Anders Langlands and Luca Fascione
// Copyright (c) 2020, Weta Digital, Ltd.  (Apache-2.0)
//
// Provides:
//   PixelSensor              -- spectral sensor with r/g/b response curves
//     PixelSensor(cs, illum, imagingRatio)  -- CIE XYZ pseudo-sensor
//     PixelSensor(r,g,b, cs, illum, ratio)  -- RGB sensor with swatch calibration
//     ToSensorRGB(L, lambda)  -- map radiance SampledSpectrum -> SensorRGB
//     XYZFromSensorRGB        -- 3x3 calibrated sensor -> XYZ matrix
//
// Algorithm overview (pbrt-v4 film.h):
//   For XYZ sensor: use CIE X/Y/Z as r_bar/g_bar/b_bar, optionally apply
//     white-balance correction via WhiteBalance().
//   For camera sensor: fit XYZFromSensorRGB via LinearLeastSquares over 24
//     Macbeth ColorChecker swatch reflectances (Pascale 2004-2012).
//
// Dependencies:
//   sampled_spectrum.h   -- SampledSpectrum<N>, SampledWavelengths<N>, SafeDiv
//   spectrum_types.h     -- DenselySampledSpectrum, PiecewiseLinearSpectrum
//   spectral_math.h      -- InnerProduct, SpectrumToXYZ
//   cie_data.h           -- GetCIE_X/Y/Z()
//   rgb_colorspace.h     -- RGBColorSpace
//   square_matrix.h      -- SquareMatrix<3>, LinearLeastSquares<3>
//   color_utils.h        -- XYZ, WhiteBalance
// ---------------------------------------------------------------------------

#include "sampled_spectrum.h"
#include "spectrum_types.h"
#include "spectral_math.h"
#include "../data/cie_data.h"
#include "rgb_colorspace.h"
#include "square_matrix.h"
#include "color_utils.h"

#include <optional>
#include <array>

// ---------------------------------------------------------------------------
// SensorRGB -- lightweight 3-channel float triplet used by PixelSensor
// ---------------------------------------------------------------------------
struct SensorRGB {
	float r = 0.f, g = 0.f, b = 0.f;
	float& operator[](int i) {
		if (i == 0) return r;
		if (i == 1) return g;
		return b;
	}
	float operator[](int i) const {
		if (i == 0) return r;
		if (i == 1) return g;
		return b;
	}
	SensorRGB operator*(float s) const { return {r * s, g * s, b * s}; }
	SensorRGB operator/(float s) const { return {r / s, g / s, b / s}; }
	SensorRGB& operator+=(const SensorRGB& o) { r += o.r; g += o.g; b += o.b; return *this; }
};

// ---------------------------------------------------------------------------
// PixelSensor
// ---------------------------------------------------------------------------
class PixelSensor {
public:
	// -----------------------------------------------------------------------
	// Constructor 1: CIE XYZ pseudo-sensor  (pbrt-v4: PixelSensor(cs, illum, ratio, alloc))
	//
	// Uses CIE X/Y/Z as the sensor's r/g/b response curves.  If a sensor
	// illuminant is provided its white chromaticity is used to white-balance
	// the output colour space.
	// -----------------------------------------------------------------------
	PixelSensor(const RGBColorSpace* outputColorSpace,
				const DenselySampledSpectrum* sensorIllum,
				float imagingRatio)
		: r_bar(GetCIE_X()),
		  g_bar(GetCIE_Y()),
		  b_bar(GetCIE_Z()),
		  imagingRatio_(imagingRatio) {
		if (sensorIllum) {
			// Compute white balance from sensor illuminant chromaticity
			// to output colour space white point.
			XYZ srcXYZ = SpectrumToXYZ(*sensorIllum);
			// xy chromaticity
			float sum = srcXYZ.X + srcXYZ.Y + srcXYZ.Z;
			float srcx = (sum > 0.f) ? srcXYZ.X / sum : 1.f / 3.f;
			float srcy = (sum > 0.f) ? srcXYZ.Y / sum : 1.f / 3.f;
			XYZFromSensorRGB = WhiteBalance(srcx, srcy,
											outputColorSpace->wx, outputColorSpace->wy);
		}
		// else XYZFromSensorRGB remains identity (default-constructed)
	}

	// -----------------------------------------------------------------------
	// Constructor 2: Camera sensor (pbrt-v4: PixelSensor(r,g,b, cs, illum, ratio, alloc))
	//
	// Fits a 3x3 XYZFromSensorRGB matrix using LinearLeastSquares over the
	// 24 Macbeth ColorChecker swatch reflectances.
	// -----------------------------------------------------------------------
	PixelSensor(const DenselySampledSpectrum& r,
				const DenselySampledSpectrum& g,
				const DenselySampledSpectrum& b,
				const RGBColorSpace* outputColorSpace,
				const DenselySampledSpectrum* sensorIllum,
				float imagingRatio)
		: r_bar(r), g_bar(g), b_bar(b), imagingRatio_(imagingRatio) {
		// Build a flat illuminant if none provided
		ConstantSpectrum flat(1.f);

		// Compute rgbCamera for each training swatch
		double rgbCamera[kNSwatches][3]{};
		for (int i = 0; i < kNSwatches; ++i) {
			SensorRGB rgb = ProjectReflectance(swatchReflectances_[i], sensorIllum, r_bar, g_bar, b_bar);
			for (int c = 0; c < 3; ++c)
				rgbCamera[i][c] = static_cast<double>(rgb[c]);
		}

		// Compute xyzOutput for each training swatch
		double xyzOutput[kNSwatches][3]{};
		float sensorWhiteG = sensorIllum
			? InnerProduct(*sensorIllum, g_bar)
			: InnerProduct(flat, g_bar);
		float sensorWhiteY = sensorIllum
			? InnerProduct(*sensorIllum, GetCIE_Y())
			: InnerProduct(flat, GetCIE_Y());
		float whiteScale = (sensorWhiteG > 0.f && sensorWhiteY > 0.f)
						   ? (sensorWhiteY / sensorWhiteG) : 1.f;

		for (int i = 0; i < kNSwatches; ++i) {
			// pbrt-v4 uses outputColorSpace->illuminant (D65 for sRGB).
			// We approximate with equal-energy illuminant (ConstantSpectrum(1))
			// by projecting reflectance directly against CIE X/Y/Z.
			XYZ xyz = ProjectReflectanceXYZ(swatchReflectances_[i]);
			xyz = XYZ{ xyz.X * whiteScale, xyz.Y * whiteScale, xyz.Z * whiteScale };
			for (int c = 0; c < 3; ++c)
				xyzOutput[i][c] = static_cast<double>(xyz[c]);
		}

		// Fit XYZFromSensorRGB via normal equations
		auto m = LinearLeastSquares<3>(rgbCamera, xyzOutput, kNSwatches);
		if (m)
			XYZFromSensorRGB = *m;
	}

	// -----------------------------------------------------------------------
	// CreateDefault -- XYZ sensor with sRGB output, imagingRatio=1  (pbrt-v4)
	// -----------------------------------------------------------------------
	static PixelSensor CreateDefault() {
		return PixelSensor(&RGBColorSpace::sRGB(), nullptr, 1.f);
	}

	// -----------------------------------------------------------------------
	// ToSensorRGB -- convert radiance SampledSpectrum -> sensor RGB  (pbrt-v4)
	//
	//   L_div = SafeDiv(L, lambda.PDF())
	//   r = (r_bar.Sample(lambda) * L_div).Average()
	//   result *= imagingRatio
	// -----------------------------------------------------------------------
	template <int N = 4>
	SensorRGB ToSensorRGB(const SampledSpectrum<N>& L,
						  const SampledWavelengths<N>& lambda) const {
		SampledSpectrum<N> Ldiv = SafeDiv(L, lambda.PDF());
		return SensorRGB{
			(r_bar.Sample(lambda) * Ldiv).Average() * imagingRatio_,
			(g_bar.Sample(lambda) * Ldiv).Average() * imagingRatio_,
			(b_bar.Sample(lambda) * Ldiv).Average() * imagingRatio_
		};
	}

	// Public calibration matrix (pbrt-v4: PixelSensor::XYZFromSensorRGB)
	SquareMatrix<3> XYZFromSensorRGB;

private:
	// -----------------------------------------------------------------------
	// ProjectReflectance -- Riemann-sum projection of refl * illum onto basis
	//
	// pbrt-v4: PixelSensor::ProjectReflectance<Triplet>
	//
	//   result[i] = sum_lambda b_i(lambda) * refl(lambda) * illum(lambda)
	//   result   /= sum_lambda b2(lambda) * illum(lambda)    (normalise by G)
	// -----------------------------------------------------------------------
	static SensorRGB ProjectReflectance(const PiecewiseLinearSpectrum& refl,
										const DenselySampledSpectrum* illum,
										const DenselySampledSpectrum& b1,
										const DenselySampledSpectrum& b2,
										const DenselySampledSpectrum& b3) {
		static constexpr float Lambda_min = 360.f;
		static constexpr float Lambda_max = 830.f;
		SensorRGB result{};
		float g_integral = 0.f;
		for (float lambda = Lambda_min; lambda <= Lambda_max; ++lambda) {
			float illumVal = illum ? (*illum)(lambda) : 1.f;
			float r = refl(lambda);
			g_integral  += b2(lambda) * illumVal;
			result[0]   += b1(lambda) * r * illumVal;
			result[1]   += b2(lambda) * r * illumVal;
			result[2]   += b3(lambda) * r * illumVal;
		}
		if (g_integral > 0.f)
			return result / g_integral;
		return result;
	}

	// Project a swatch reflectance against the D65 illuminant using CIE X/Y/Z basis.
	//
	// pbrt-v4: ProjectReflectance<XYZ>(s, &outputColorSpace->illuminant, X, Y, Z)
	// where outputColorSpace->illuminant is D65 for sRGB/P3/Rec2020.
	// Normalised by sum(CIE_Y * D65) to match pbrt-v4's g_integral.
	static XYZ ProjectReflectanceXYZ(const PiecewiseLinearSpectrum& refl) {
		static constexpr float Lambda_min = 360.f;
		static constexpr float Lambda_max = 830.f;
		const DenselySampledSpectrum& d65 = GetD65Illuminant();
		XYZ result{};
		float g_integral = 0.f;
		for (float lambda = Lambda_min; lambda <= Lambda_max; ++lambda) {
			float r   = refl(lambda);
			float ill = d65(lambda);
			float y   = GetCIE_Y()(lambda);
			g_integral += y * ill;
			result.X   += GetCIE_X()(lambda) * r * ill;
			result.Y   += y                  * r * ill;
			result.Z   += GetCIE_Z()(lambda) * r * ill;
		}
		if (g_integral > 0.f) {
			result.X /= g_integral;
			result.Y /= g_integral;
			result.Z /= g_integral;
		}
		return result;
	}

	// -----------------------------------------------------------------------
	// Macbeth ColorChecker swatch reflectances (24 patches)
	// Data from Danny Pascale / BabelColor (used by permission)
	// http://www.babelcolor.com/index_htm_files/ColorChecker_RGB_and_spectra.zip
	// Matching pbrt-v4 film.cpp swatchReflectances[]
	// -----------------------------------------------------------------------
	static constexpr int kNSwatches = 24;

	static const PiecewiseLinearSpectrum swatchReflectances_[kNSwatches];

	// Sensor response curves (densely sampled)
	DenselySampledSpectrum r_bar, g_bar, b_bar;
	float imagingRatio_;
};

// ---------------------------------------------------------------------------
// Out-of-line static member: swatch reflectance data
// ---------------------------------------------------------------------------
inline const PiecewiseLinearSpectrum PixelSensor::swatchReflectances_[PixelSensor::kNSwatches] = {
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.055f,390.f,0.058f,400.f,0.061f,410.f,0.062f,420.f,0.062f,430.f,0.062f,
		 440.f,0.062f,450.f,0.062f,460.f,0.062f,470.f,0.062f,480.f,0.062f,490.f,0.063f,
		 500.f,0.065f,510.f,0.070f,520.f,0.076f,530.f,0.079f,540.f,0.081f,550.f,0.084f,
		 560.f,0.091f,570.f,0.103f,580.f,0.119f,590.f,0.134f,600.f,0.143f,610.f,0.147f,
		 620.f,0.151f,630.f,0.158f,640.f,0.168f,650.f,0.179f,660.f,0.188f,670.f,0.190f,
		 680.f,0.186f,690.f,0.181f,700.f,0.182f,710.f,0.187f,720.f,0.196f,730.f,0.209f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.117f,390.f,0.143f,400.f,0.175f,410.f,0.191f,420.f,0.196f,430.f,0.199f,
		 440.f,0.204f,450.f,0.213f,460.f,0.228f,470.f,0.251f,480.f,0.280f,490.f,0.309f,
		 500.f,0.329f,510.f,0.333f,520.f,0.315f,530.f,0.286f,540.f,0.273f,550.f,0.276f,
		 560.f,0.277f,570.f,0.289f,580.f,0.339f,590.f,0.420f,600.f,0.488f,610.f,0.525f,
		 620.f,0.546f,630.f,0.562f,640.f,0.578f,650.f,0.595f,660.f,0.612f,670.f,0.625f,
		 680.f,0.638f,690.f,0.656f,700.f,0.678f,710.f,0.700f,720.f,0.717f,730.f,0.734f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.130f,390.f,0.177f,400.f,0.251f,410.f,0.306f,420.f,0.324f,430.f,0.330f,
		 440.f,0.333f,450.f,0.331f,460.f,0.323f,470.f,0.311f,480.f,0.298f,490.f,0.285f,
		 500.f,0.269f,510.f,0.250f,520.f,0.231f,530.f,0.214f,540.f,0.199f,550.f,0.185f,
		 560.f,0.169f,570.f,0.157f,580.f,0.149f,590.f,0.145f,600.f,0.142f,610.f,0.141f,
		 620.f,0.141f,630.f,0.141f,640.f,0.143f,650.f,0.147f,660.f,0.152f,670.f,0.154f,
		 680.f,0.150f,690.f,0.144f,700.f,0.136f,710.f,0.132f,720.f,0.135f,730.f,0.147f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.051f,390.f,0.054f,400.f,0.056f,410.f,0.057f,420.f,0.058f,430.f,0.059f,
		 440.f,0.060f,450.f,0.061f,460.f,0.062f,470.f,0.063f,480.f,0.065f,490.f,0.067f,
		 500.f,0.075f,510.f,0.101f,520.f,0.145f,530.f,0.178f,540.f,0.184f,550.f,0.170f,
		 560.f,0.149f,570.f,0.133f,580.f,0.122f,590.f,0.115f,600.f,0.109f,610.f,0.105f,
		 620.f,0.104f,630.f,0.106f,640.f,0.109f,650.f,0.112f,660.f,0.114f,670.f,0.114f,
		 680.f,0.112f,690.f,0.112f,700.f,0.115f,710.f,0.120f,720.f,0.125f,730.f,0.130f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.144f,390.f,0.198f,400.f,0.294f,410.f,0.375f,420.f,0.408f,430.f,0.421f,
		 440.f,0.426f,450.f,0.426f,460.f,0.419f,470.f,0.403f,480.f,0.379f,490.f,0.346f,
		 500.f,0.311f,510.f,0.281f,520.f,0.254f,530.f,0.229f,540.f,0.214f,550.f,0.208f,
		 560.f,0.202f,570.f,0.194f,580.f,0.193f,590.f,0.200f,600.f,0.214f,610.f,0.230f,
		 620.f,0.241f,630.f,0.254f,640.f,0.279f,650.f,0.313f,660.f,0.348f,670.f,0.366f,
		 680.f,0.366f,690.f,0.359f,700.f,0.358f,710.f,0.365f,720.f,0.377f,730.f,0.398f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.136f,390.f,0.179f,400.f,0.247f,410.f,0.297f,420.f,0.320f,430.f,0.337f,
		 440.f,0.355f,450.f,0.381f,460.f,0.419f,470.f,0.466f,480.f,0.510f,490.f,0.546f,
		 500.f,0.567f,510.f,0.574f,520.f,0.569f,530.f,0.551f,540.f,0.524f,550.f,0.488f,
		 560.f,0.445f,570.f,0.400f,580.f,0.350f,590.f,0.299f,600.f,0.252f,610.f,0.221f,
		 620.f,0.204f,630.f,0.196f,640.f,0.191f,650.f,0.188f,660.f,0.191f,670.f,0.199f,
		 680.f,0.212f,690.f,0.223f,700.f,0.232f,710.f,0.233f,720.f,0.229f,730.f,0.229f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.054f,390.f,0.054f,400.f,0.053f,410.f,0.054f,420.f,0.054f,430.f,0.055f,
		 440.f,0.055f,450.f,0.055f,460.f,0.056f,470.f,0.057f,480.f,0.058f,490.f,0.061f,
		 500.f,0.068f,510.f,0.089f,520.f,0.125f,530.f,0.154f,540.f,0.174f,550.f,0.199f,
		 560.f,0.248f,570.f,0.335f,580.f,0.444f,590.f,0.538f,600.f,0.587f,610.f,0.595f,
		 620.f,0.591f,630.f,0.587f,640.f,0.584f,650.f,0.584f,660.f,0.590f,670.f,0.603f,
		 680.f,0.620f,690.f,0.639f,700.f,0.655f,710.f,0.663f,720.f,0.663f,730.f,0.667f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.122f,390.f,0.164f,400.f,0.229f,410.f,0.286f,420.f,0.327f,430.f,0.361f,
		 440.f,0.388f,450.f,0.400f,460.f,0.392f,470.f,0.362f,480.f,0.316f,490.f,0.260f,
		 500.f,0.209f,510.f,0.168f,520.f,0.138f,530.f,0.117f,540.f,0.104f,550.f,0.096f,
		 560.f,0.090f,570.f,0.086f,580.f,0.084f,590.f,0.084f,600.f,0.084f,610.f,0.084f,
		 620.f,0.084f,630.f,0.085f,640.f,0.090f,650.f,0.098f,660.f,0.109f,670.f,0.123f,
		 680.f,0.143f,690.f,0.169f,700.f,0.205f,710.f,0.244f,720.f,0.287f,730.f,0.332f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.096f,390.f,0.115f,400.f,0.131f,410.f,0.135f,420.f,0.133f,430.f,0.132f,
		 440.f,0.130f,450.f,0.128f,460.f,0.125f,470.f,0.120f,480.f,0.115f,490.f,0.110f,
		 500.f,0.105f,510.f,0.100f,520.f,0.095f,530.f,0.093f,540.f,0.092f,550.f,0.093f,
		 560.f,0.096f,570.f,0.108f,580.f,0.156f,590.f,0.265f,600.f,0.399f,610.f,0.500f,
		 620.f,0.556f,630.f,0.579f,640.f,0.588f,650.f,0.591f,660.f,0.593f,670.f,0.594f,
		 680.f,0.598f,690.f,0.602f,700.f,0.607f,710.f,0.609f,720.f,0.609f,730.f,0.610f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.092f,390.f,0.116f,400.f,0.146f,410.f,0.169f,420.f,0.178f,430.f,0.173f,
		 440.f,0.158f,450.f,0.139f,460.f,0.119f,470.f,0.101f,480.f,0.087f,490.f,0.075f,
		 500.f,0.066f,510.f,0.060f,520.f,0.056f,530.f,0.053f,540.f,0.051f,550.f,0.051f,
		 560.f,0.052f,570.f,0.052f,580.f,0.051f,590.f,0.052f,600.f,0.058f,610.f,0.073f,
		 620.f,0.096f,630.f,0.119f,640.f,0.141f,650.f,0.166f,660.f,0.194f,670.f,0.227f,
		 680.f,0.265f,690.f,0.309f,700.f,0.355f,710.f,0.396f,720.f,0.436f,730.f,0.478f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.061f,390.f,0.061f,400.f,0.062f,410.f,0.063f,420.f,0.064f,430.f,0.066f,
		 440.f,0.069f,450.f,0.075f,460.f,0.085f,470.f,0.105f,480.f,0.139f,490.f,0.192f,
		 500.f,0.271f,510.f,0.376f,520.f,0.476f,530.f,0.531f,540.f,0.549f,550.f,0.546f,
		 560.f,0.528f,570.f,0.504f,580.f,0.471f,590.f,0.428f,600.f,0.381f,610.f,0.347f,
		 620.f,0.327f,630.f,0.318f,640.f,0.312f,650.f,0.310f,660.f,0.314f,670.f,0.327f,
		 680.f,0.345f,690.f,0.363f,700.f,0.376f,710.f,0.381f,720.f,0.378f,730.f,0.379f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.063f,390.f,0.063f,400.f,0.063f,410.f,0.064f,420.f,0.064f,430.f,0.064f,
		 440.f,0.065f,450.f,0.066f,460.f,0.067f,470.f,0.068f,480.f,0.071f,490.f,0.076f,
		 500.f,0.087f,510.f,0.125f,520.f,0.206f,530.f,0.305f,540.f,0.383f,550.f,0.431f,
		 560.f,0.469f,570.f,0.518f,580.f,0.568f,590.f,0.607f,600.f,0.628f,610.f,0.637f,
		 620.f,0.640f,630.f,0.642f,640.f,0.645f,650.f,0.648f,660.f,0.651f,670.f,0.653f,
		 680.f,0.657f,690.f,0.664f,700.f,0.673f,710.f,0.680f,720.f,0.684f,730.f,0.688f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.066f,390.f,0.079f,400.f,0.102f,410.f,0.146f,420.f,0.200f,430.f,0.244f,
		 440.f,0.282f,450.f,0.309f,460.f,0.308f,470.f,0.278f,480.f,0.231f,490.f,0.178f,
		 500.f,0.130f,510.f,0.094f,520.f,0.070f,530.f,0.054f,540.f,0.046f,550.f,0.042f,
		 560.f,0.039f,570.f,0.038f,580.f,0.038f,590.f,0.038f,600.f,0.038f,610.f,0.039f,
		 620.f,0.039f,630.f,0.040f,640.f,0.041f,650.f,0.042f,660.f,0.044f,670.f,0.045f,
		 680.f,0.046f,690.f,0.046f,700.f,0.048f,710.f,0.052f,720.f,0.057f,730.f,0.065f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.052f,390.f,0.053f,400.f,0.054f,410.f,0.055f,420.f,0.057f,430.f,0.059f,
		 440.f,0.061f,450.f,0.066f,460.f,0.075f,470.f,0.093f,480.f,0.125f,490.f,0.178f,
		 500.f,0.246f,510.f,0.307f,520.f,0.337f,530.f,0.334f,540.f,0.317f,550.f,0.293f,
		 560.f,0.262f,570.f,0.230f,580.f,0.198f,590.f,0.165f,600.f,0.135f,610.f,0.115f,
		 620.f,0.104f,630.f,0.098f,640.f,0.094f,650.f,0.092f,660.f,0.093f,670.f,0.097f,
		 680.f,0.102f,690.f,0.108f,700.f,0.113f,710.f,0.115f,720.f,0.114f,730.f,0.114f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.050f,390.f,0.049f,400.f,0.048f,410.f,0.047f,420.f,0.047f,430.f,0.047f,
		 440.f,0.047f,450.f,0.047f,460.f,0.046f,470.f,0.045f,480.f,0.044f,490.f,0.044f,
		 500.f,0.045f,510.f,0.046f,520.f,0.047f,530.f,0.048f,540.f,0.049f,550.f,0.050f,
		 560.f,0.054f,570.f,0.060f,580.f,0.072f,590.f,0.104f,600.f,0.178f,610.f,0.312f,
		 620.f,0.467f,630.f,0.581f,640.f,0.644f,650.f,0.675f,660.f,0.690f,670.f,0.698f,
		 680.f,0.706f,690.f,0.715f,700.f,0.724f,710.f,0.730f,720.f,0.734f,730.f,0.738f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.058f,390.f,0.054f,400.f,0.052f,410.f,0.052f,420.f,0.053f,430.f,0.054f,
		 440.f,0.056f,450.f,0.059f,460.f,0.067f,470.f,0.081f,480.f,0.107f,490.f,0.152f,
		 500.f,0.225f,510.f,0.336f,520.f,0.462f,530.f,0.559f,540.f,0.616f,550.f,0.650f,
		 560.f,0.672f,570.f,0.694f,580.f,0.710f,590.f,0.723f,600.f,0.731f,610.f,0.739f,
		 620.f,0.746f,630.f,0.752f,640.f,0.758f,650.f,0.764f,660.f,0.769f,670.f,0.771f,
		 680.f,0.776f,690.f,0.782f,700.f,0.790f,710.f,0.796f,720.f,0.799f,730.f,0.804f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.145f,390.f,0.195f,400.f,0.283f,410.f,0.346f,420.f,0.362f,430.f,0.354f,
		 440.f,0.334f,450.f,0.306f,460.f,0.276f,470.f,0.248f,480.f,0.218f,490.f,0.190f,
		 500.f,0.168f,510.f,0.149f,520.f,0.127f,530.f,0.107f,540.f,0.100f,550.f,0.102f,
		 560.f,0.104f,570.f,0.109f,580.f,0.137f,590.f,0.200f,600.f,0.290f,610.f,0.400f,
		 620.f,0.516f,630.f,0.615f,640.f,0.687f,650.f,0.732f,660.f,0.760f,670.f,0.774f,
		 680.f,0.783f,690.f,0.793f,700.f,0.803f,710.f,0.812f,720.f,0.817f,730.f,0.825f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.108f,390.f,0.141f,400.f,0.192f,410.f,0.236f,420.f,0.261f,430.f,0.286f,
		 440.f,0.317f,450.f,0.353f,460.f,0.390f,470.f,0.426f,480.f,0.446f,490.f,0.444f,
		 500.f,0.423f,510.f,0.385f,520.f,0.337f,530.f,0.283f,540.f,0.231f,550.f,0.185f,
		 560.f,0.146f,570.f,0.118f,580.f,0.101f,590.f,0.090f,600.f,0.082f,610.f,0.076f,
		 620.f,0.074f,630.f,0.073f,640.f,0.073f,650.f,0.074f,660.f,0.076f,670.f,0.077f,
		 680.f,0.076f,690.f,0.075f,700.f,0.073f,710.f,0.072f,720.f,0.074f,730.f,0.079f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.189f,390.f,0.255f,400.f,0.423f,410.f,0.660f,420.f,0.811f,430.f,0.862f,
		 440.f,0.877f,450.f,0.884f,460.f,0.891f,470.f,0.896f,480.f,0.899f,490.f,0.904f,
		 500.f,0.907f,510.f,0.909f,520.f,0.911f,530.f,0.910f,540.f,0.911f,550.f,0.914f,
		 560.f,0.913f,570.f,0.916f,580.f,0.915f,590.f,0.916f,600.f,0.914f,610.f,0.915f,
		 620.f,0.918f,630.f,0.919f,640.f,0.921f,650.f,0.923f,660.f,0.924f,670.f,0.922f,
		 680.f,0.922f,690.f,0.925f,700.f,0.927f,710.f,0.930f,720.f,0.930f,730.f,0.933f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.171f,390.f,0.232f,400.f,0.365f,410.f,0.507f,420.f,0.567f,430.f,0.583f,
		 440.f,0.588f,450.f,0.590f,460.f,0.591f,470.f,0.590f,480.f,0.588f,490.f,0.588f,
		 500.f,0.589f,510.f,0.589f,520.f,0.591f,530.f,0.590f,540.f,0.590f,550.f,0.590f,
		 560.f,0.589f,570.f,0.591f,580.f,0.590f,590.f,0.590f,600.f,0.587f,610.f,0.585f,
		 620.f,0.583f,630.f,0.580f,640.f,0.578f,650.f,0.576f,660.f,0.574f,670.f,0.572f,
		 680.f,0.571f,690.f,0.569f,700.f,0.568f,710.f,0.568f,720.f,0.566f,730.f,0.566f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.144f,390.f,0.192f,400.f,0.272f,410.f,0.331f,420.f,0.350f,430.f,0.357f,
		 440.f,0.361f,450.f,0.363f,460.f,0.363f,470.f,0.361f,480.f,0.359f,490.f,0.358f,
		 500.f,0.358f,510.f,0.359f,520.f,0.360f,530.f,0.360f,540.f,0.361f,550.f,0.361f,
		 560.f,0.360f,570.f,0.362f,580.f,0.362f,590.f,0.361f,600.f,0.359f,610.f,0.358f,
		 620.f,0.355f,630.f,0.352f,640.f,0.350f,650.f,0.348f,660.f,0.345f,670.f,0.343f,
		 680.f,0.340f,690.f,0.338f,700.f,0.335f,710.f,0.334f,720.f,0.332f,730.f,0.331f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.105f,390.f,0.131f,400.f,0.163f,410.f,0.180f,420.f,0.186f,430.f,0.190f,
		 440.f,0.193f,450.f,0.194f,460.f,0.194f,470.f,0.192f,480.f,0.191f,490.f,0.191f,
		 500.f,0.191f,510.f,0.192f,520.f,0.192f,530.f,0.192f,540.f,0.192f,550.f,0.192f,
		 560.f,0.192f,570.f,0.193f,580.f,0.192f,590.f,0.192f,600.f,0.191f,610.f,0.189f,
		 620.f,0.188f,630.f,0.186f,640.f,0.184f,650.f,0.182f,660.f,0.181f,670.f,0.179f,
		 680.f,0.178f,690.f,0.176f,700.f,0.174f,710.f,0.173f,720.f,0.172f,730.f,0.171f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.068f,390.f,0.077f,400.f,0.084f,410.f,0.087f,420.f,0.089f,430.f,0.090f,
		 440.f,0.092f,450.f,0.092f,460.f,0.091f,470.f,0.090f,480.f,0.090f,490.f,0.090f,
		 500.f,0.090f,510.f,0.090f,520.f,0.090f,530.f,0.090f,540.f,0.090f,550.f,0.090f,
		 560.f,0.090f,570.f,0.090f,580.f,0.090f,590.f,0.089f,600.f,0.089f,610.f,0.088f,
		 620.f,0.087f,630.f,0.086f,640.f,0.086f,650.f,0.085f,660.f,0.084f,670.f,0.084f,
		 680.f,0.083f,690.f,0.083f,700.f,0.082f,710.f,0.081f,720.f,0.081f,730.f,0.081f}),
	PiecewiseLinearSpectrum::FromInterleaved(
		{380.f,0.031f,390.f,0.032f,400.f,0.032f,410.f,0.033f,420.f,0.033f,430.f,0.033f,
		 440.f,0.033f,450.f,0.033f,460.f,0.032f,470.f,0.032f,480.f,0.032f,490.f,0.032f,
		 500.f,0.032f,510.f,0.032f,520.f,0.032f,530.f,0.032f,540.f,0.032f,550.f,0.032f,
		 560.f,0.032f,570.f,0.032f,580.f,0.032f,590.f,0.032f,600.f,0.032f,610.f,0.032f,
		 620.f,0.032f,630.f,0.032f,640.f,0.032f,650.f,0.032f,660.f,0.032f,670.f,0.032f,
		 680.f,0.032f,690.f,0.032f,700.f,0.032f,710.f,0.032f,720.f,0.032f,730.f,0.033f})
};
