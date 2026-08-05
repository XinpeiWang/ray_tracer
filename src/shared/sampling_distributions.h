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
//   VisibleWavelengths -- SampleVisibleWavelengths, VisibleWavelengthsPDF
//                        (no CDF-invert; the sample function IS the inverse CDF)
//
// Dependencies (all already in this project):
//   scalar_math.h  -- Clamp, Lerp, Sqr, Pow<n>, SmoothStep, Gaussian, ErfInv,
//                     Logistic, LogisticCDF, NewtonBisection, kSqrt2, kPi
//
// NOTE: SampleLinear / InvertLinearSample are reproduced inline here to avoid
// pulling in the full sampling.h (which has non-inline ODR-sensitive functions).
//
// References: pbrt-v4 src/pbrt/util/sampling.h  (lines 195-310, 442-459, VisibleWavelengths)

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
#    define CPU_GPU inline
#  endif
#endif

// Suppress MSVC C4141 (duplicate 'inline') that can arise when CPU_GPU
// expands to nothing and the function is already declared inline.
#if defined(_MSC_VER)
#  pragma warning(push)
#endif

// Convenience aliases for the constants used below
static constexpr double kSampDistSqrt2 = scalar_math_detail::kSqrt2;
static constexpr double kSampDistPi    = scalar_math_detail::kPi;

// ===========================================================================
// Local inline helpers from sampling.h (reproduced to avoid ODR issues)
// ===========================================================================

template<typename T>
CPU_GPU T SD_SampleLinear(T u, T a, T b) {
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
CPU_GPU T SD_InvertLinearSample(T x, T a, T b) {
	T sum = a + b;
	if (sum == T(0)) return x;
	return x * (a * (T(2) - x) + b * x) / sum;
}

// ===========================================================================
// Tent distribution  f(x) = (1/r - |x|/r^2)  on [-r, r]
// ===========================================================================

CPU_GPU double TentPDF(double x, double r) {
	if (std::abs(x) >= r) return 0.0;
	return 1.0 / r - std::abs(x) / (r * r);
}

// Sample a Tent-distributed value given a uniform u in [0,1).
// Uses the fact that the tent is the sum of two mirror-image linear pieces.
CPU_GPU double SampleTent(double u, double r) {
	if (u < 0.5) {
		u = u * 2.0;                            // re-map to [0,1)
		return -r + r * SD_SampleLinear<double>(u, 0.0, 1.0);
	} else {
		u = (u - 0.5) * 2.0;
		return r * SD_SampleLinear<double>(u, 1.0, 0.0);
	}
}

CPU_GPU double InvertTentSample(double x, double r) {
	if (x <= 0.0)
		return (1.0 - SD_InvertLinearSample<double>(-x / r, 1.0, 0.0)) / 2.0;
	else
		return 0.5 + SD_InvertLinearSample<double>(x / r, 1.0, 0.0) / 2.0;
}

// ===========================================================================
// Exponential distribution  f(x) = a * e^{-a x},  x >= 0
// ===========================================================================

CPU_GPU double ExponentialPDF(double x, double a) {
	return a * std::exp(-a * x);
}

CPU_GPU double SampleExponential(double u, double a) {
	return -std::log(1.0 - u) / a;
}

CPU_GPU double InvertExponentialSample(double x, double a) {
	return 1.0 - std::exp(-a * x);
}

// ===========================================================================
// Trimmed Exponential  f(x) = c*e^{-cx} / (1 - e^{-c*xMax}),  x in [0, xMax]
// ===========================================================================

CPU_GPU double TrimmedExponentialPDF(double x, double c, double xMax) {
	if (x < 0.0 || x > xMax) return 0.0;
	return c / (1.0 - std::exp(-c * xMax)) * std::exp(-c * x);
}

CPU_GPU double SampleTrimmedExponential(double u, double c, double xMax) {
	return std::log(1.0 - u * (1.0 - std::exp(-c * xMax))) / -c;
}

CPU_GPU double InvertTrimmedExponentialSample(double x, double c, double xMax) {
	return (1.0 - std::exp(-c * x)) / (1.0 - std::exp(-c * xMax));
}

// ===========================================================================
// Normal (Gaussian) distribution
// ===========================================================================

CPU_GPU double NormalPDF(double x, double mu = 0.0, double sigma = 1.0) {
	return Gaussian(x, mu, sigma);
}

CPU_GPU double SampleNormal(double u, double mu = 0.0, double sigma = 1.0) {
	return mu + kSampDistSqrt2 * sigma * static_cast<double>(ErfInv(float(2.0 * u - 1.0)));
}

CPU_GPU double InvertNormalSample(double x, double mu = 0.0, double sigma = 1.0) {
	return 0.5 * (1.0 + std::erf((x - mu) / (sigma * kSampDistSqrt2)));
}

// Box-Muller: generate two independent normal samples from two uniforms
// Returns {x, y} each drawn from N(mu, sigma^2).
CPU_GPU void SampleTwoNormal(double u0, double u1,
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

CPU_GPU double LogisticPDF(double x, double s) {
	x = std::abs(x);
	double e = std::exp(-x / s);
	return e / (s * (1.0 + e) * (1.0 + e));
}

// CDF: sigma(x/s) = 1 / (1 + e^{-x/s})
CPU_GPU double InvertLogisticSample(double x, double s) {
	return 1.0 / (1.0 + std::exp(-x / s));
}

CPU_GPU double SampleLogistic(double u, double s) {
	return -s * std::log(1.0 / u - 1.0);
}

// ===========================================================================
// Trimmed Logistic  (logistic restricted to [a, b])
// ===========================================================================

CPU_GPU double TrimmedLogisticPDF(double x, double s, double a, double b) {
	if (x < a || x > b) return 0.0;
	return Logistic(x, s) / (InvertLogisticSample(b, s) - InvertLogisticSample(a, s));
}

CPU_GPU double SampleTrimmedLogistic(double u, double s, double a, double b) {
	double Pa = InvertLogisticSample(a, s);
	double Pb = InvertLogisticSample(b, s);
	double x  = SampleLogistic(Lerp(u, Pa, Pb), s);
	return Clamp(x, a, b);
}

CPU_GPU double InvertTrimmedLogisticSample(double x, double s, double a, double b) {
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

CPU_GPU double SmoothStepPDF(double x, double a, double b) {
	if (x < a || x > b) return 0.0;
	return (2.0 / (b - a)) * SmoothStep(x, a, b);
}

CPU_GPU double SampleSmoothStep(double u, double a, double b) {
	auto cdfMinusU = [=](double x) -> std::pair<double, double> {
		double t = (x - a) / (b - a);
		double P = 2.0 * Pow<3>(t) - Pow<4>(t);
		double PDeriv = SmoothStepPDF(x, a, b);
		return { P - u, PDeriv };
	};
	return NewtonBisection(a, b, cdfMinusU);
}

CPU_GPU double InvertSmoothStepSample(double x, double a, double b) {
	double t = (x - a) / (b - a);
	double P = 2.0 * Pow<3>(t) - Pow<4>(t);
	// NOTE: pbrt-v4's version wraps P in a lambda then computes (P(x)-P(a))/(P(b)-P(a)),
	// but the lambda captures t (fixed) and ignores its argument, making the division
	// degenerate (0/0).  The correct CDF-inverse is simply P = 2t^3 - t^4, since
	// CDF(a)=0 and CDF(b)=1 by construction.
	return P;
}

// ---------------------------------------------------------------------------
// VisibleWavelengths -- importance-sample wavelength ÃƒÅ½Ã‚Â» proportional to V(ÃƒÅ½Ã‚Â»)
//
// pbrt-v4 reference: util/sampling.h  SampleVisibleWavelengths / VisibleWavelengthsPDF
//
// SampleVisibleWavelengths(u):
//   Maps u ~ Uniform[0,1) to ÃƒÅ½Ã‚Â» in nm, sampling proportional to the CIE
//   photopic luminous-efficiency function V(ÃƒÅ½Ã‚Â»).  The closed-form inverse CDF
//   concentrates Monte Carlo samples in the 450-650 nm range where human
//   vision is most sensitive, reducing variance in spectral rendering.
//
// VisibleWavelengthsPDF(ÃƒÅ½Ã‚Â»):
//   Returns the PDF of the distribution above.  Zero outside [360, 830] nm.
//   PDF = 0.0039398042 / coshÃƒâ€šÃ‚Â²(0.0072Ãƒâ€šÃ‚Â·(ÃƒÅ½Ã‚Â» ÃƒÂ¢Ã‹â€ Ã¢â‚¬â„¢ 538))
// ---------------------------------------------------------------------------

CPU_GPU double SampleVisibleWavelengths(double u) {
	// Inverse CDF: ÃƒÅ½Ã‚Â» = 538 ÃƒÂ¢Ã‹â€ Ã¢â‚¬â„¢ 138.888889Ãƒâ€šÃ‚Â·atanh(0.85691062 ÃƒÂ¢Ã‹â€ Ã¢â‚¬â„¢ 1.82750197Ãƒâ€šÃ‚Â·u)
	// pbrt-v4: return 538 - 138.888889f * std::atanh(0.85691062f - 1.82750197f * u);
	return 538.0 - 138.888889 * std::atanh(0.85691062 - 1.82750197 * u);
}

CPU_GPU double VisibleWavelengthsPDF(double lambda) {
	// Zero outside the visible range [360, 830] nm.
	if (lambda < 360.0 || lambda > 830.0)
		return 0.0;
	// PDF = 0.0039398042 / coshÃƒâ€šÃ‚Â²(0.0072Ãƒâ€šÃ‚Â·(ÃƒÅ½Ã‚Â» ÃƒÂ¢Ã‹â€ Ã¢â‚¬â„¢ 538))
	double c = std::cosh(0.0072 * (lambda - 538.0));
	return 0.0039398042 / (c * c);
}

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
