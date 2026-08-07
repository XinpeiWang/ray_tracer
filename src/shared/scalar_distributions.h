#pragma once
// =============================================================================
// scalar_distributions.h  --  Scalar 1-D distribution family + VarianceEstimator
//
// Mirrors pbrt-v4 util/sampling.h (inline scalar distribution functions).
//
// Each distribution provides three functions:
//   XxxPDF(x, ...)            -- probability density at x
//   SampleXxx(u, ...)         -- map uniform u in [0,1) to a sample
//   InvertXxxSample(x, ...)   -- invert the CDF (recover u from x)
//
// Distributions:
//   Tent             -- triangular filter, support [-r, r]
//   Exponential      -- e^(-a*x), x >= 0
//   TrimmedExponential -- e^(-c*x), x in [0, xMax]
//   Normal / TwoNormal -- Gaussian via ErfInv / Box-Muller
//   Logistic / TrimmedLogistic -- heavy-tailed, used in hair BxDF
//   SmoothStep       -- C1-smooth bump on [a, b]
//   VarianceEstimator -- online Welford mean/variance (with parallel Merge)
//
// Requires scalar_math.h for: Gaussian, ErfInv, SmoothStep, NewtonBisection,
//                              Logistic, LogisticCDF
// Requires sampling.h (or standalone) for: SampleLinear, InvertLinearSample
// =============================================================================

#include "cpu_gpu.h"

#include <cmath>
#include <limits>
#include <utility>
#include <cstdint>
#include "scalar_math.h"
#include "sampling.h"    // for SampleLinear / InvertLinearSample

// -----------------------------------------------------------------------------
// Tent distribution  -- support [-r, r],  integral = 1
// PDF(x) = 1/r - |x|/r^2
// pbrt-v4: TentPDF / SampleTent / InvertTentSample
// -----------------------------------------------------------------------------
CPU_GPU float TentPDF(float x, float r) {
	if (std::abs(x) >= r) return 0.f;
	return 1.f / r - std::abs(x) / (r * r);
}

CPU_GPU float SampleTent(float u, float r) {
	if (u < 0.5f) {
		u = 2.f * u;
		return -r + r * SampleLinear(u, 0.f, 1.f);
	} else {
		u = 2.f * (u - 0.5f);
		return r * SampleLinear(u, 1.f, 0.f);
	}
}

CPU_GPU float InvertTentSample(float x, float r) {
	if (x <= 0.f)
		return (1.f - InvertLinearSample(-x / r, 1.f, 0.f)) * 0.5f;
	else
		return 0.5f + InvertLinearSample(x / r, 1.f, 0.f) * 0.5f;
}

// -----------------------------------------------------------------------------
// Exponential distribution  -- support [0, inf),  PDF(x) = a*e^(-a*x)
// pbrt-v4: ExponentialPDF / SampleExponential / InvertExponentialSample
// -----------------------------------------------------------------------------
CPU_GPU float ExponentialPDF(float x, float a) {
	return a * std::exp(-a * x);
}

CPU_GPU float SampleExponential(float u, float a) {
	return -std::log(1.f - u) / a;
}

CPU_GPU float InvertExponentialSample(float x, float a) {
	return 1.f - std::exp(-a * x);
}

// -----------------------------------------------------------------------------
// TrimmedExponential -- e^(-c*x) on [0, xMax]
// pbrt-v4: TrimmedExponentialPDF / SampleTrimmedExponential / InvertTrimmedExponentialSample
// Used by media equi-angular sampling.
// -----------------------------------------------------------------------------
CPU_GPU float TrimmedExponentialPDF(float x, float c, float xMax) {
	if (x < 0.f || x > xMax) return 0.f;
	return c / (1.f - std::exp(-c * xMax)) * std::exp(-c * x);
}

CPU_GPU float SampleTrimmedExponential(float u, float c, float xMax) {
	return std::log(1.f - u * (1.f - std::exp(-c * xMax))) / -c;
}

CPU_GPU float InvertTrimmedExponentialSample(float x, float c, float xMax) {
	return (1.f - std::exp(-c * x)) / (1.f - std::exp(-c * xMax));
}

// -----------------------------------------------------------------------------
// Normal (Gaussian) distribution
// pbrt-v4: NormalPDF / SampleNormal / InvertNormalSample / SampleTwoNormal
// Requires ErfInv and Gaussian from scalar_math.h.
// -----------------------------------------------------------------------------
CPU_GPU float NormalPDF(float x, float mu = 0.f, float sigma = 1.f) {
	return static_cast<float>(Gaussian(x, mu, sigma));
}

CPU_GPU float SampleNormal(float u, float mu = 0.f, float sigma = 1.f) {
	static constexpr float kSqrt2 = 1.41421356237f;
	return mu + kSqrt2 * sigma * ErfInv(2.f * u - 1.f);
}

CPU_GPU float InvertNormalSample(float x, float mu = 0.f, float sigma = 1.f) {
	static constexpr float kSqrt2 = 1.41421356237f;
	return 0.5f * (1.f + std::erf((x - mu) / (sigma * kSqrt2)));
}

// Box-Muller: produce two independent N(mu, sigma^2) samples from a 2D uniform.
// pbrt-v4: SampleTwoNormal
CPU_GPU void SampleTwoNormal(float u0, float u1,
								float& out0, float& out1,
								float mu = 0.f, float sigma = 1.f) {
	// Use log1p(-u0) for numerical robustness near u0=1. pbrt-v4: SampleTwoNormal
	float r2  = -2.f * std::log1p(-u0);
	float phi = 6.28318530717958647692f * u1;
	out0 = mu + sigma * std::sqrt(r2) * std::cos(phi);
	out1 = mu + sigma * std::sqrt(r2) * std::sin(phi);
}

// -----------------------------------------------------------------------------
// Logistic distribution
// PDF(x,s) = exp(-|x|/s) / (s*(1+exp(-|x|/s))^2)
// pbrt-v4: LogisticPDF / SampleLogistic / InvertLogisticSample
// Used in hair BxDF (Marschner model).
// -----------------------------------------------------------------------------
CPU_GPU float LogisticPDF(float x, float s) {
	x = std::abs(x);
	float ex = std::exp(-x / s);
	return ex / (s * (1.f + ex) * (1.f + ex));
}

CPU_GPU float SampleLogistic(float u, float s) {
	return -s * std::log(1.f / u - 1.f);
}

CPU_GPU float InvertLogisticSample(float x, float s) {
	return 1.f / (1.f + std::exp(-x / s));
}

// TrimmedLogistic -- Logistic truncated to [a, b]
// pbrt-v4: TrimmedLogisticPDF / SampleTrimmedLogistic / InvertTrimmedLogisticSample
CPU_GPU float TrimmedLogisticPDF(float x, float s, float a, float b) {
	if (x < a || x > b) return 0.f;
	return LogisticPDF(x, s) / (InvertLogisticSample(b, s) - InvertLogisticSample(a, s));
}

CPU_GPU float SampleTrimmedLogistic(float u, float s, float a, float b) {
	float pa = InvertLogisticSample(a, s);
	float pb = InvertLogisticSample(b, s);
	float x  = SampleLogistic(pa + u * (pb - pa), s);
	return std::max(a, std::min(b, x));
}

CPU_GPU float InvertTrimmedLogisticSample(float x, float s, float a, float b) {
	float pa = InvertLogisticSample(a, s);
	float pb = InvertLogisticSample(b, s);
	return (InvertLogisticSample(x, s) - pa) / (pb - pa);
}

// -----------------------------------------------------------------------------
// SmoothStep distribution  -- C1-smooth bump on [a, b]
// PDF(x) = (2/(b-a)) * smoothstep(x,a,b);  CDF = 2t^3 - t^4,  t=(x-a)/(b-a)
// pbrt-v4: SmoothStepPDF / SampleSmoothStep / InvertSmoothStepSample
// Requires SmoothStep and NewtonBisection from scalar_math.h.
// -----------------------------------------------------------------------------
CPU_GPU float SmoothStepPDF(float x, float a, float b) {
	if (x < a || x > b) return 0.f;
	return (2.f / (b - a)) * SmoothStep(x, a, b);
}

CPU_GPU float SampleSmoothStep(float u, float a, float b) {
	auto cdfMinusU = [=](float x) -> std::pair<float,float> {
		float t  = (x - a) / (b - a);
		float P  = 2.f * t * t * t - t * t * t * t;    // CDF
		float dP = SmoothStepPDF(x, a, b);              // PDF = CDF'
		return {P - u, dP};
	};
	return NewtonBisection(a, b, cdfMinusU);
}

CPU_GPU float InvertSmoothStepSample(float x, float a, float b) {
	float t = (x - a) / (b - a);
	return 2.f * t * t * t - t * t * t * t;
}

// =============================================================================
// VarianceEstimator<Float>  --  online Welford mean/variance estimator
// Supports parallel merge via Chan et al. formula.
// pbrt-v4: VarianceEstimator (util/sampling.h)
// =============================================================================
template<typename Float>
class VarianceEstimator {
public:
	void Add(Float x) {
		++n;
		Float delta  = x - mean;
		mean += delta / static_cast<Float>(n);
		Float delta2 = x - mean;
		S += delta * delta2;
	}

	Float   Mean()             const { return mean; }
	Float   Variance()         const { return (n > 1) ? S / static_cast<Float>(n - 1) : Float(0); }
	int64_t Count()            const { return n; }
	Float   RelativeVariance() const {
		return (n < 1 || mean == Float(0)) ? Float(0) : Variance() / Mean();
	}

	// Parallel merge (Chan et al.) -- matches pbrt-v4 Merge()
	void Merge(const VarianceEstimator& ve) {
		if (ve.n == 0) return;
		Float combined = static_cast<Float>(n + ve.n);
		S    = S + ve.S + (ve.mean - mean) * (ve.mean - mean)
			   * static_cast<Float>(n) * static_cast<Float>(ve.n) / combined;
		mean = (static_cast<Float>(n) * mean + static_cast<Float>(ve.n) * ve.mean) / combined;
		n   += ve.n;
	}

private:
	Float   mean = Float(0), S = Float(0);
	int64_t n    = 0;
};
