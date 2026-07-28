// sampling_distributions.h
// Invertible 1-D sampling distributions ported from pbrt-v4 util/sampling.h
//
// Functions provided (each has a Sample*, *PDF, and Invert*Sample triple
// unless noted):
//
//   Tent             -- SampleTent, TentPDF, InvertTentSample
//   Exponential      -- SampleExponential, ExponentialPDF, InvertExponentialSample
//   TrimmedExponential -- SampleTrimmedExponential, TrimmedExponentialPDF,
//                        InvertTrimmedExponentialSample
//   Normal (Gaussian)-- SampleNormal, NormalPDF, InvertNormalSample
//   TwoNormal        -- SampleTwoNormal   (Box-Muller; no invert)
//   Logistic         -- SampleLogistic, LogisticPDF, InvertLogisticSample
//   TrimmedLogistic  -- SampleTrimmedLogistic, TrimmedLogisticPDF,
//                       InvertTrimmedLogisticSample
//   SmoothStep       -- SampleSmoothStep, SmoothStepPDF, InvertSmoothStepSample
//
// Dependencies (all already in this project):
//   scalar_math.h  -- Clamp, Lerp, Sqr, Pow<n>, SmoothStep, Gaussian, ErfInv,
//                     Logistic, LogisticCDF, NewtonBisection, kSqrt2, kPi
//
// NOTE: SampleLinear / InvertLinearSample are reproduced inline here to avoid
// pulling in the full sampling.h (which has non-inline ODR-sensitive functions).
//
// References: pbrt-v4 src/pbrt/util/sampling.h  (lines 195-310, 442-459)

#pragma once
#include <cmath>
#include "scalar_math.h"

// ---------------------------------------------------------------------------
// CPU_GPU macro guard (mirrors the convention used in lowdiscrepancy.h)
// ---------------------------------------------------------------------------
#ifndef CPU_GPU
#  ifdef __CUDACC__
#    define CPU_GPU __host__ __device__
#  else
#    define CPU_GPU
#  endif
#endif

// Suppress MSVC C4141 (duplicate 'inline') that can arise when CPU_GPU
// expands to nothing and the function is already declared inline.
#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4141)
#endif

// Convenience aliases for the constants used below
static constexpr double kSampDistSqrt2 = scalar_math_detail::kSqrt2;
static constexpr double kSampDistPi    = scalar_math_detail::kPi;

// ===========================================================================
// Local inline helpers from sampling.h (reproduced to avoid ODR issues)
// ===========================================================================

template<typename T>
CPU_GPU inline T SD_SampleLinear(T u, T a, T b) {
	if (u == T(0) && a == T(0)) return T(0);
	T sum = a + b;
	if (sum == T(0)) return u;
	T lerp_sq = a * a + (b * b - a * a) * u;
	if (lerp_sq < T(0)) lerp_sq = T(0);
	T x = u * sum / (a + std::sqrt(lerp_sq));
	if (x >= T(1)) x = T(1) - T(1e-7);
	return x;
}

template<typename T>
CPU_GPU inline T SD_InvertLinearSample(T x, T a, T b) {
	T sum = a + b;
	if (sum == T(0)) return x;
	return x * (a * (T(2) - x) + b * x) / sum;
}

// ===========================================================================
// Tent distribution  f(x) = (1/r - |x|/r^2)  on [-r, r]
// ===========================================================================

CPU_GPU inline double TentPDF(double x, double r) {
	if (std::abs(x) >= r) return 0.0;
	return 1.0 / r - std::abs(x) / (r * r);
}

// Sample a Tent-distributed value given a uniform u in [0,1).
// Uses the fact that the tent is the sum of two mirror-image linear pieces.
CPU_GPU inline double SampleTent(double u, double r) {
	if (u < 0.5) {
		u = u * 2.0;                            // re-map to [0,1)
		return -r + r * SD_SampleLinear<double>(u, 0.0, 1.0);
	} else {
		u = (u - 0.5) * 2.0;
		return r * SD_SampleLinear<double>(u, 1.0, 0.0);
	}
}

CPU_GPU inline double InvertTentSample(double x, double r) {
	if (x <= 0.0)
		return (1.0 - SD_InvertLinearSample<double>(-x / r, 1.0, 0.0)) / 2.0;
	else
		return 0.5 + SD_InvertLinearSample<double>(x / r, 1.0, 0.0) / 2.0;
}

// ===========================================================================
// Exponential distribution  f(x) = a * e^{-a x},  x >= 0
// ===========================================================================

CPU_GPU inline double ExponentialPDF(double x, double a) {
	return a * std::exp(-a * x);
}

CPU_GPU inline double SampleExponential(double u, double a) {
	return -std::log(1.0 - u) / a;
}

CPU_GPU inline double InvertExponentialSample(double x, double a) {
	return 1.0 - std::exp(-a * x);
}

// ===========================================================================
// Trimmed Exponential  f(x) = c*e^{-cx} / (1 - e^{-c*xMax}),  x in [0, xMax]
// ===========================================================================

CPU_GPU inline double TrimmedExponentialPDF(double x, double c, double xMax) {
	if (x < 0.0 || x > xMax) return 0.0;
	return c / (1.0 - std::exp(-c * xMax)) * std::exp(-c * x);
}

CPU_GPU inline double SampleTrimmedExponential(double u, double c, double xMax) {
	return std::log(1.0 - u * (1.0 - std::exp(-c * xMax))) / -c;
}

CPU_GPU inline double InvertTrimmedExponentialSample(double x, double c, double xMax) {
	return (1.0 - std::exp(-c * x)) / (1.0 - std::exp(-c * xMax));
}

// ===========================================================================
// Normal (Gaussian) distribution
// ===========================================================================

CPU_GPU inline double NormalPDF(double x, double mu = 0.0, double sigma = 1.0) {
	return Gaussian(x, mu, sigma);
}

CPU_GPU inline double SampleNormal(double u, double mu = 0.0, double sigma = 1.0) {
	return mu + kSampDistSqrt2 * sigma * static_cast<double>(ErfInv(float(2.0 * u - 1.0)));
}

CPU_GPU inline double InvertNormalSample(double x, double mu = 0.0, double sigma = 1.0) {
	return 0.5 * (1.0 + std::erf((x - mu) / (sigma * kSampDistSqrt2)));
}

// Box-Muller: generate two independent normal samples from two uniforms
// Returns {x, y} each drawn from N(mu, sigma^2).
CPU_GPU inline void SampleTwoNormal(double u0, double u1,
									 double& out0, double& out1,
									 double mu = 0.0, double sigma = 1.0) {
	double r2 = -2.0 * std::log1p(-u0);          // robust log(1 - u0)
	double r  = sigma * std::sqrt(r2);
	out0 = mu + r * std::cos(2.0 * kSampDistPi * u1);
	out1 = mu + r * std::sin(2.0 * kSampDistPi * u1);
}

// ===========================================================================
// Logistic distribution  f(x) = e^{-|x|/s} / (s * (1 + e^{-|x|/s})^2)
// ===========================================================================

CPU_GPU inline double LogisticPDF(double x, double s) {
	x = std::abs(x);
	double e = std::exp(-x / s);
	return e / (s * (1.0 + e) * (1.0 + e));
}

// CDF: sigma(x/s) = 1 / (1 + e^{-x/s})
CPU_GPU inline double InvertLogisticSample(double x, double s) {
	return 1.0 / (1.0 + std::exp(-x / s));
}

CPU_GPU inline double SampleLogistic(double u, double s) {
	return -s * std::log(1.0 / u - 1.0);
}

// ===========================================================================
// Trimmed Logistic  (logistic restricted to [a, b])
// ===========================================================================

CPU_GPU inline double TrimmedLogisticPDF(double x, double s, double a, double b) {
	if (x < a || x > b) return 0.0;
	return Logistic(x, s) / (InvertLogisticSample(b, s) - InvertLogisticSample(a, s));
}

CPU_GPU inline double SampleTrimmedLogistic(double u, double s, double a, double b) {
	double Pa = InvertLogisticSample(a, s);
	double Pb = InvertLogisticSample(b, s);
	double x  = SampleLogistic(Lerp(u, Pa, Pb), s);
	return Clamp(x, a, b);
}

CPU_GPU inline double InvertTrimmedLogisticSample(double x, double s, double a, double b) {
	double Pa = InvertLogisticSample(a, s);
	double Pb = InvertLogisticSample(b, s);
	return (InvertLogisticSample(x, s) - Pa) / (Pb - Pa);
}

// ===========================================================================
// SmoothStep distribution  f(x) proportional to SmoothStep(x, a, b)
//
// CDF: P(x) = 2t^3 - t^4  where t = (x-a)/(b-a)
// Sampled via Newton-bisection (same as pbrt-v4).
// ===========================================================================

CPU_GPU inline double SmoothStepPDF(double x, double a, double b) {
	if (x < a || x > b) return 0.0;
	return (2.0 / (b - a)) * SmoothStep(x, a, b);
}

CPU_GPU inline double SampleSmoothStep(double u, double a, double b) {
	auto cdfMinusU = [=](double x) -> std::pair<double, double> {
		double t = (x - a) / (b - a);
		double P = 2.0 * Pow<3>(t) - Pow<4>(t);
		double PDeriv = SmoothStepPDF(x, a, b);
		return { P - u, PDeriv };
	};
	return NewtonBisection(a, b, cdfMinusU);
}

CPU_GPU inline double InvertSmoothStepSample(double x, double a, double b) {
	double t = (x - a) / (b - a);
	double P = 2.0 * Pow<3>(t) - Pow<4>(t);
	return P;  // CDF at x normalised to [0,1] (P(a)=0, P(b)=1)
}

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
