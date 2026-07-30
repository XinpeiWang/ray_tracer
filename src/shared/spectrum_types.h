#pragma once
// ---------------------------------------------------------------------------
// spectrum_types.h -- Spectrum type implementations for hero-wavelength rendering
//
// Ported from pbrt-v4 src/pbrt/util/spectrum.h (Apache-2.0)
// Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
//
// Provides:
//   Blackbody(lambda, T)        -- Planck spectral radiance (W/sr/m^2/nm)
//   ConstantSpectrum            -- uniform value over all wavelengths
//   DenselySampledSpectrum      -- one sample per nm in [lambda_min, lambda_max]
//   PiecewiseLinearSpectrum     -- tabulated (lambda, value) pairs with lerp
//   BlackbodySpectrum           -- normalized Planckian emitter
//
// All types provide:
//   float operator()(float lambda) const  -- evaluate at wavelength lambda (nm)
//   float MaxValue() const
//   SampledSpectrum<N> Sample(const SampledWavelengths<N>&) const
//
// Design notes:
//   - Header-only; no allocator dependency (uses std::vector).
//   - Exact numerical match to pbrt-v4 where possible.
//   - DenselySampledSpectrum::operator() snaps to nearest integer nm (lround),
//     matching pbrt-v4 exactly (no lerp between samples).
//   - Lambda range constants: Lambda_min = 360, Lambda_max = 830 nm.
//
// Dependencies:
//   sampled_spectrum.h   -- SampledSpectrum<N>, SampledWavelengths<N>
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "sampled_spectrum.h"

#ifndef CPU_GPU
#  if defined(__CUDACC__)
#    define CPU_GPU __host__ __device__ __forceinline__
#  else
#    define CPU_GPU inline
#  endif
#endif

// ---------------------------------------------------------------------------
// Spectrum range constants (matching pbrt-v4)
// ---------------------------------------------------------------------------
static constexpr float SpectrumLambda_min = 360.f;
static constexpr float SpectrumLambda_max = 830.f;

// ---------------------------------------------------------------------------
// Blackbody -- Planck spectral radiance
//
// pbrt-v4 reference: util/spectrum.h Blackbody()
// Returns spectral radiance in W/(sr * m^2 * nm) for temperature T (Kelvin)
// and wavelength lambda (nm). Returns 0 for T <= 0.
// ---------------------------------------------------------------------------
CPU_GPU inline float Blackbody(float lambda, float T) {
	if (T <= 0.f)
		return 0.f;
	const float c  = 299792458.f;
	const float h  = 6.62606957e-34f;
	const float kb = 1.3806488e-23f;
	float l  = lambda * 1e-9f;  // nm -> m
	float Le = (2.f * h * c * c) /
			   (std::pow(l, 5.f) * (std::exp((h * c) / (l * kb * T)) - 1.f));
	return Le;
}

// ---------------------------------------------------------------------------
// ConstantSpectrum
//
// pbrt-v4 reference: util/spectrum.h ConstantSpectrum
// Evaluates to a fixed value c for every wavelength.
// ---------------------------------------------------------------------------
struct ConstantSpectrum {
	CPU_GPU explicit ConstantSpectrum(float c) : c(c) {}

	CPU_GPU float operator()(float /*lambda*/) const { return c; }
	CPU_GPU float MaxValue() const { return c; }

	template <int N = 4>
	CPU_GPU SampledSpectrum<N> Sample(const SampledWavelengths<N>& /*lambda*/) const {
		return SampledSpectrum<N>(c);
	}

	float c;
};

// ---------------------------------------------------------------------------
// DenselySampledSpectrum
//
// pbrt-v4 reference: util/spectrum.h DenselySampledSpectrum
// Stores one float value per integer nm in [lambda_min, lambda_max].
// operator() snaps lambda to the nearest integer nm (std::lround) -- exact
// pbrt-v4 behaviour (no interpolation).
// ---------------------------------------------------------------------------
struct DenselySampledSpectrum {
	// Construct zero-filled spectrum over [lambda_min, lambda_max]
	explicit DenselySampledSpectrum(
		int lmin = static_cast<int>(SpectrumLambda_min),
		int lmax = static_cast<int>(SpectrumLambda_max))
		: lambda_min(lmin),
		  lambda_max(lmax),
		  values(static_cast<size_t>(lmax - lmin + 1), 0.f)
	{
		assert(lmax >= lmin);
	}

	// Construct from a callable f(float lambda) -> float
	// pbrt-v4: DenselySampledSpectrum::SampleFunction
	template <typename F>
	static DenselySampledSpectrum SampleFunction(
		F func,
		int lmin = static_cast<int>(SpectrumLambda_min),
		int lmax = static_cast<int>(SpectrumLambda_max))
	{
		DenselySampledSpectrum s(lmin, lmax);
		for (int lambda = lmin; lambda <= lmax; ++lambda)
			s.values[static_cast<size_t>(lambda - lmin)] =
				func(static_cast<float>(lambda));
		return s;
	}

	// Evaluate at lambda (nm) -- snaps to nearest integer nm
	CPU_GPU float operator()(float lambda) const {
		int offset = static_cast<int>(std::lround(lambda)) - lambda_min;
		if (offset < 0 || offset >= static_cast<int>(values.size()))
			return 0.f;
		return values[static_cast<size_t>(offset)];
	}

	CPU_GPU float MaxValue() const {
		if (values.empty()) return 0.f;
		return *std::max_element(values.begin(), values.end());
	}

	void Scale(float s) {
		for (float& v : values)
			v *= s;
	}

	template <int N = 4>
	SampledSpectrum<N> Sample(const SampledWavelengths<N>& swl) const {
		SampledSpectrum<N> s(0.f);
		for (int i = 0; i < N; ++i) {
			int offset = static_cast<int>(std::lround(swl.lambda[i])) - lambda_min;
			if (offset >= 0 && offset < static_cast<int>(values.size()))
				s[i] = values[static_cast<size_t>(offset)];
		}
		return s;
	}

	bool operator==(const DenselySampledSpectrum& d) const {
		return lambda_min == d.lambda_min &&
			   lambda_max == d.lambda_max &&
			   values == d.values;
	}
	bool operator!=(const DenselySampledSpectrum& d) const { return !(*this == d); }

	// Public data (matches pbrt-v4 private, exposed for testability)
	int lambda_min, lambda_max;
	std::vector<float> values;
};

// ---------------------------------------------------------------------------
// PiecewiseLinearSpectrum
//
// pbrt-v4 reference: util/spectrum.h PiecewiseLinearSpectrum
// Tabulated spectrum defined by sorted (lambda, value) pairs.
// Evaluates to 0 outside the defined range; linearly interpolates within.
// ---------------------------------------------------------------------------
struct PiecewiseLinearSpectrum {
	PiecewiseLinearSpectrum() = default;

	// Construct from separate sorted lambda and value arrays
	PiecewiseLinearSpectrum(const std::vector<float>& lambdas_in,
							const std::vector<float>& values_in)
		: lambdas(lambdas_in), values(values_in)
	{
		assert(lambdas.size() == values.size());
	}

	// FromInterleaved: construct from interleaved [lambda0, val0, lambda1, val1, ...]
	// pbrt-v4: PiecewiseLinearSpectrum::FromInterleaved
	// If normalize=true, scales so that the maximum value equals 1.
	static PiecewiseLinearSpectrum FromInterleaved(
		const std::vector<float>& samples,
		bool normalize = false)
	{
		assert(samples.size() % 2 == 0);
		PiecewiseLinearSpectrum s;
		for (size_t i = 0; i < samples.size(); i += 2) {
			s.lambdas.push_back(samples[i]);
			s.values.push_back(samples[i + 1]);
		}
		if (normalize) {
			float maxVal = *std::max_element(s.values.begin(), s.values.end());
			if (maxVal > 0.f)
				for (float& v : s.values)
					v /= maxVal;
		}
		return s;
	}

	// Evaluate at wavelength lambda (nm)
	// pbrt-v4: PiecewiseLinearSpectrum::operator()
	CPU_GPU float operator()(float lambda) const {
		if (lambdas.empty() || lambda < lambdas.front() || lambda > lambdas.back())
			return 0.f;
		// Find the segment containing lambda
		auto it = std::lower_bound(lambdas.begin(), lambdas.end(), lambda);
		if (it == lambdas.end())
			return values.back();
		if (it == lambdas.begin())
			return values.front();
		size_t hi = static_cast<size_t>(it - lambdas.begin());
		size_t lo = hi - 1;
		float t = (lambda - lambdas[lo]) / (lambdas[hi] - lambdas[lo]);
		return values[lo] + t * (values[hi] - values[lo]);
	}

	float MaxValue() const {
		if (values.empty()) return 0.f;
		return *std::max_element(values.begin(), values.end());
	}

	void Scale(float s) {
		for (float& v : values)
			v *= s;
	}

	template <int N = 4>
	SampledSpectrum<N> Sample(const SampledWavelengths<N>& swl) const {
		SampledSpectrum<N> s(0.f);
		for (int i = 0; i < N; ++i)
			s[i] = (*this)(swl.lambda[i]);
		return s;
	}

	// Public data (matches pbrt-v4 private, exposed for testability)
	std::vector<float> lambdas;
	std::vector<float> values;
};

// ---------------------------------------------------------------------------
// BlackbodySpectrum
//
// pbrt-v4 reference: util/spectrum.h BlackbodySpectrum
// Normalized Planckian emitter: Blackbody(lambda, T) / Blackbody(lambdaMax, T)
// so that MaxValue() == 1.
// lambdaMax is computed from Wien's displacement law: lambda_max = b / T
// where b = 2.8977721e-3 m*K.
// ---------------------------------------------------------------------------
struct BlackbodySpectrum {
	CPU_GPU explicit BlackbodySpectrum(float T) : T(T) {
		// Wien's displacement law: lambda_max (m) = 2.8977721e-3 / T
		float lambdaMax = 2.8977721e-3f / T;
		// normalizationFactor = 1 / Blackbody(lambdaMax_nm, T)
		normalizationFactor = 1.f / Blackbody(lambdaMax * 1e9f, T);
	}

	CPU_GPU float operator()(float lambda) const {
		return Blackbody(lambda, T) * normalizationFactor;
	}

	CPU_GPU float MaxValue() const { return 1.f; }

	template <int N = 4>
	CPU_GPU SampledSpectrum<N> Sample(const SampledWavelengths<N>& swl) const {
		SampledSpectrum<N> s(0.f);
		for (int i = 0; i < N; ++i)
			s[i] = Blackbody(swl.lambda[i], T) * normalizationFactor;
		return s;
	}

	float T;
	float normalizationFactor;
};
