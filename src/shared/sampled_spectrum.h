#pragma once
// ---------------------------------------------------------------------------
// sampled_spectrum.h -- Hero-wavelength spectral rendering primitives
//
// Ported from pbrt-v4 src/pbrt/util/spectrum.h (Apache-2.0)
// Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
//
// Provides:
//   SampledSpectrum<N>   -- Fixed array of N spectral-band values with full
//                           arithmetic, suitable for Monte Carlo integration.
//   SampledWavelengths<N>-- N hero wavelengths + PDFs; SampleUniform and
//                           SampleVisible factory methods; TerminateSecondary
//                           for non-fluorescent path termination.
//
// Free functions (element-wise, mirroring pbrt-v4):
//   SafeDiv, ClampZero, Clamp, Sqrt, SafeSqrt, Pow, Exp, FastExp, Lerp
//
// Usage:
//   #include "sampled_spectrum.h"
//   using SS  = SampledSpectrum<4>;
//   using SWL = SampledWavelengths<4>;
//   SWL swl = SWL::SampleVisible(rng_u);
//   SS  L(0.f);   // zero-initialised
//
// NOTE: ToXYZ / ToRGB / y() require CIE XYZ LUTs that live in the full
//       DenselySampledSpectrum system -- not yet ported. Those methods are
//       therefore excluded here. They can be added once spectrum data tables
//       are available.
//
// Dependencies:
//   scalar_math.h            -- Clamp, SafeSqrt, FastExp, Lerp
//   sampling_distributions.h -- SampleVisibleWavelengths, VisibleWavelengthsPDF
// ---------------------------------------------------------------------------

#include <array>
#include <cmath>
#include <algorithm>
#include <cassert>

#include "scalar_math.h"
#include "sampling_distributions.h"

#ifndef CPU_GPU
#  if defined(__CUDACC__)
#    define CPU_GPU __host__ __device__ __forceinline__
#  else
#    define CPU_GPU
#  endif
#endif

// ---------------------------------------------------------------------------
// Spectrum constants (pbrt-v4: spectrum.h)
// ---------------------------------------------------------------------------
static constexpr float kLambda_min = 360.f;
static constexpr float kLambda_max = 830.f;
static constexpr float kCIE_Y_integral = 106.856895f;

// ---------------------------------------------------------------------------
// SampledSpectrum<N>
//
// Mirrors pbrt-v4 class SampledSpectrum (NSpectrumSamples hard-coded to 4).
// Here we template on N so tests can use smaller sizes.
// ---------------------------------------------------------------------------
template <int N = 4>
class SampledSpectrum {
public:
	// ---- Construction -------------------------------------------------------
	SampledSpectrum() = default;

	CPU_GPU explicit SampledSpectrum(float c) { values.fill(c); }

	// Construct from raw array (pbrt-v4: SampledSpectrum(pstd::span<const Float>))
	CPU_GPU explicit SampledSpectrum(const float* v, int n) {
		assert(n == N);
		for (int i = 0; i < N; ++i) values[i] = v[i];
	}

	// ---- Element access -----------------------------------------------------
	CPU_GPU float  operator[](int i) const { return values[i]; }
	CPU_GPU float& operator[](int i)       { return values[i]; }

	// ---- Bool conversion (any non-zero component) ---------------------------
	CPU_GPU explicit operator bool() const {
		for (int i = 0; i < N; ++i)
			if (values[i] != 0.f) return true;
		return false;
	}

	// ---- Equality -----------------------------------------------------------
	CPU_GPU bool operator==(const SampledSpectrum& s) const { return values == s.values; }
	CPU_GPU bool operator!=(const SampledSpectrum& s) const { return values != s.values; }

	// ---- Arithmetic: SampledSpectrum op SampledSpectrum ---------------------
	CPU_GPU SampledSpectrum& operator+=(const SampledSpectrum& s) {
		for (int i = 0; i < N; ++i) values[i] += s.values[i];
		return *this;
	}
	CPU_GPU SampledSpectrum operator+(const SampledSpectrum& s) const {
		SampledSpectrum r = *this; r += s; return r;
	}

	CPU_GPU SampledSpectrum& operator-=(const SampledSpectrum& s) {
		for (int i = 0; i < N; ++i) values[i] -= s.values[i];
		return *this;
	}
	CPU_GPU SampledSpectrum operator-(const SampledSpectrum& s) const {
		SampledSpectrum r = *this; r -= s; return r;
	}
	// scalar - spectrum  (pbrt-v4 friend)
	CPU_GPU friend SampledSpectrum operator-(float a, const SampledSpectrum& s) {
		SampledSpectrum r;
		for (int i = 0; i < N; ++i) r.values[i] = a - s.values[i];
		return r;
	}

	CPU_GPU SampledSpectrum& operator*=(const SampledSpectrum& s) {
		for (int i = 0; i < N; ++i) values[i] *= s.values[i];
		return *this;
	}
	CPU_GPU SampledSpectrum operator*(const SampledSpectrum& s) const {
		SampledSpectrum r = *this; r *= s; return r;
	}

	CPU_GPU SampledSpectrum& operator/=(const SampledSpectrum& s) {
		for (int i = 0; i < N; ++i) values[i] /= s.values[i];
		return *this;
	}
	CPU_GPU SampledSpectrum operator/(const SampledSpectrum& s) const {
		SampledSpectrum r = *this; r /= s; return r;
	}

	// ---- Arithmetic: SampledSpectrum op scalar ------------------------------
	CPU_GPU SampledSpectrum& operator*=(float a) {
		for (int i = 0; i < N; ++i) values[i] *= a;
		return *this;
	}
	CPU_GPU SampledSpectrum operator*(float a) const {
		SampledSpectrum r = *this; r *= a; return r;
	}
	CPU_GPU friend SampledSpectrum operator*(float a, const SampledSpectrum& s) {
		return s * a;
	}

	CPU_GPU SampledSpectrum& operator/=(float a) {
		for (int i = 0; i < N; ++i) values[i] /= a;
		return *this;
	}
	CPU_GPU SampledSpectrum operator/(float a) const {
		SampledSpectrum r = *this; r /= a; return r;
	}

	CPU_GPU SampledSpectrum operator-() const {
		SampledSpectrum r;
		for (int i = 0; i < N; ++i) r.values[i] = -values[i];
		return r;
	}

	// ---- Reductions ---------------------------------------------------------
	CPU_GPU float MinComponentValue() const {
		float m = values[0];
		for (int i = 1; i < N; ++i) m = std::min(m, values[i]);
		return m;
	}
	CPU_GPU float MaxComponentValue() const {
		float m = values[0];
		for (int i = 1; i < N; ++i) m = std::max(m, values[i]);
		return m;
	}
	CPU_GPU float Average() const {
		float s = 0.f;
		for (int i = 0; i < N; ++i) s += values[i];
		return s / N;
	}

	// ---- Diagnostics --------------------------------------------------------
	CPU_GPU bool HasNaNs() const {
		for (int i = 0; i < N; ++i)
			if (values[i] != values[i]) return true;  // NaN != NaN
		return false;
	}

	// ---- Storage ------------------------------------------------------------
	std::array<float, N> values{};
};

// ---------------------------------------------------------------------------
// SampledSpectrum free functions (pbrt-v4: spectrum.h inline functions)
// ---------------------------------------------------------------------------

template <int N>
CPU_GPU inline SampledSpectrum<N> SafeDiv(SampledSpectrum<N> a, SampledSpectrum<N> b) {
	SampledSpectrum<N> r;
	for (int i = 0; i < N; ++i)
		r[i] = (b[i] != 0.f) ? a[i] / b[i] : 0.f;
	return r;
}

template <int N>
CPU_GPU inline SampledSpectrum<N> ClampZero(const SampledSpectrum<N>& s) {
	SampledSpectrum<N> r;
	for (int i = 0; i < N; ++i) r[i] = std::max(0.f, s[i]);
	return r;
}

template <int N>
CPU_GPU inline SampledSpectrum<N> Clamp(const SampledSpectrum<N>& s, float lo, float hi) {
	SampledSpectrum<N> r;
	for (int i = 0; i < N; ++i) r[i] = ::Clamp(s[i], lo, hi);
	return r;
}

template <int N>
CPU_GPU inline SampledSpectrum<N> Sqrt(const SampledSpectrum<N>& s) {
	SampledSpectrum<N> r;
	for (int i = 0; i < N; ++i) r[i] = std::sqrt(s[i]);
	return r;
}

template <int N>
CPU_GPU inline SampledSpectrum<N> SafeSqrt(const SampledSpectrum<N>& s) {
	SampledSpectrum<N> r;
	for (int i = 0; i < N; ++i) r[i] = ::SafeSqrt(s[i]);
	return r;
}

template <int N>
CPU_GPU inline SampledSpectrum<N> Pow(const SampledSpectrum<N>& s, float e) {
	SampledSpectrum<N> r;
	for (int i = 0; i < N; ++i) r[i] = std::pow(s[i], e);
	return r;
}

template <int N>
CPU_GPU inline SampledSpectrum<N> Exp(const SampledSpectrum<N>& s) {
	SampledSpectrum<N> r;
	for (int i = 0; i < N; ++i) r[i] = std::exp(s[i]);
	return r;
}

template <int N>
CPU_GPU inline SampledSpectrum<N> FastExp(const SampledSpectrum<N>& s) {
	SampledSpectrum<N> r;
	for (int i = 0; i < N; ++i) r[i] = ::FastExp(s[i]);
	return r;
}

template <int N>
CPU_GPU inline SampledSpectrum<N> Lerp(float t,
										const SampledSpectrum<N>& s1,
										const SampledSpectrum<N>& s2) {
	return (1.f - t) * s1 + t * s2;
}

// ---------------------------------------------------------------------------
// SampledWavelengths<N>
//
// Mirrors pbrt-v4 class SampledWavelengths.
// Holds N hero wavelengths (lambda[]) and their sampling PDFs (pdf[]).
//
// Factory methods:
//   SampleUniform(u, lambda_min, lambda_max)  -- stratified uniform sampling
//   SampleVisible(u)                          -- V(λ)-importance sampling
//
// TerminateSecondary() -- zero out pdf[1..N-1] for non-fluorescent paths
//                         (collapses to single-wavelength tracking).
// ---------------------------------------------------------------------------
template <int N = 4>
class SampledWavelengths {
public:
	// ---- Equality -----------------------------------------------------------
	CPU_GPU bool operator==(const SampledWavelengths& o) const {
		return lambda == o.lambda && pdf == o.pdf;
	}
	CPU_GPU bool operator!=(const SampledWavelengths& o) const {
		return !(*this == o);
	}

	// ---- Element access -----------------------------------------------------
	CPU_GPU float  operator[](int i) const { return lambda[i]; }
	CPU_GPU float& operator[](int i)       { return lambda[i]; }

	// ---- PDF accessor -------------------------------------------------------
	CPU_GPU SampledSpectrum<N> PDF() const {
		SampledSpectrum<N> s;
		for (int i = 0; i < N; ++i) s[i] = pdf[i];
		return s;
	}

	// ---- Factory: uniform stratified ----------------------------------------
	// pbrt-v4: SampledWavelengths::SampleUniform(Float u, Float lambda_min, lambda_max)
	CPU_GPU static SampledWavelengths SampleUniform(
			float u,
			float lambda_min = kLambda_min,
			float lambda_max = kLambda_max) {
		SampledWavelengths swl;
		// First wavelength via simple lerp
		swl.lambda[0] = lambda_min + u * (lambda_max - lambda_min);
		// Remaining wavelengths: evenly spaced with wrap-around
		float delta = (lambda_max - lambda_min) / N;
		for (int i = 1; i < N; ++i) {
			swl.lambda[i] = swl.lambda[i - 1] + delta;
			if (swl.lambda[i] > lambda_max)
				swl.lambda[i] = lambda_min + (swl.lambda[i] - lambda_max);
		}
		// Uniform PDF = 1 / (lambda_max - lambda_min)
		float p = 1.f / (lambda_max - lambda_min);
		for (int i = 0; i < N; ++i) swl.pdf[i] = p;
		return swl;
	}

	// ---- Factory: V(λ)-importance sampling ----------------------------------
	// pbrt-v4: SampledWavelengths::SampleVisible(Float u)
	// Uses SampleVisibleWavelengths / VisibleWavelengthsPDF from
	// sampling_distributions.h (ported in the previous session).
	CPU_GPU static SampledWavelengths SampleVisible(float u) {
		SampledWavelengths swl;
		for (int i = 0; i < N; ++i) {
			// Stratified offset: map u into the i-th stratum with wrap
			float up = u + float(i) / float(N);
			if (up > 1.f) up -= 1.f;
			swl.lambda[i] = static_cast<float>(SampleVisibleWavelengths(static_cast<double>(up)));
			swl.pdf[i]    = static_cast<float>(VisibleWavelengthsPDF(static_cast<double>(swl.lambda[i])));
		}
		return swl;
	}

	// ---- Secondary termination (non-fluorescent paths) ----------------------
	// pbrt-v4: TerminateSecondary()
	// After calling this, only lambda[0] carries a non-zero PDF.
	// The integrator should then only evaluate the primary wavelength and
	// compensate by dividing by the surviving PDF.
	CPU_GPU void TerminateSecondary() {
		if (SecondaryTerminated()) return;
		for (int i = 1; i < N; ++i) pdf[i] = 0.f;
		pdf[0] /= float(N);
	}

	CPU_GPU bool SecondaryTerminated() const {
		for (int i = 1; i < N; ++i)
			if (pdf[i] != 0.f) return false;
		return true;
	}

	// ---- Storage (public for direct access in tests) -----------------------
	std::array<float, N> lambda{};
	std::array<float, N> pdf{};
};
