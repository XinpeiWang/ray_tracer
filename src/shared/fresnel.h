#pragma once

// ---------------------------------------------------------------------------
// Shared Fresnel functions ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â usable on both CPU and GPU.
//
// Mirrors pbrt-v4's FrDielectric and FrComplex (src/pbrt/util/scattering.h),
// both annotated CPU_GPU so the same code runs on CPU and GPU.
//
// CPU usage  (MSVC / plain C++):  CPU_GPU is empty, T = double
// GPU usage  (NVCC / CUDA):       CPU_GPU = __host__ __device__, T = float
//
// FrComplex overloads:
//   FrComplex(cosTheta, eta_r, eta_k)      -- explicit real+imag (CPU+GPU)
//   FrComplex(cosTheta, complex<T> eta)    -- std::complex convenience (CPU)
// ---------------------------------------------------------------------------

#if defined(__CUDACC__)
#   define CPU_GPU __host__ __device__ __forceinline__
#else
#   define CPU_GPU inline
#endif

#if defined(__CUDACC__)
#   include <math_functions.h>   // fmaxf, fminf, sqrtf (device)
#else
#   include <cmath>
#   include <complex>
#endif

// ---------------------------------------------------------------------------
// FrDielectric -- real-valued Fresnel for dielectric interfaces (pbrt-v4)
// eta = eta_t / eta_i  (transmitted / incident IOR)
// ---------------------------------------------------------------------------
template <typename T>
CPU_GPU T FrDielectric(T cos_theta_i, T eta) {
	// Clamp to [-1, 1]
#if defined(__CUDACC__)
	cos_theta_i = fmaxf(-T(1), fminf(T(1), cos_theta_i));
#else
	cos_theta_i = std::fmax(T(-1), std::fmin(T(1), cos_theta_i));
#endif

	// If ray is inside the medium, flip the interface
	if (cos_theta_i < T(0)) {
		eta = T(1) / eta;
		cos_theta_i = -cos_theta_i;
	}

	T sin2_theta_i = T(1) - cos_theta_i * cos_theta_i;
	T sin2_theta_t = sin2_theta_i / (eta * eta);

	// Total internal reflection
	if (sin2_theta_t >= T(1))
		return T(1);

	// SafeSqrt: clamp argument to avoid NaN from floating-point rounding
#if defined(__CUDACC__)
	T cos_theta_t = sqrtf(fmaxf(T(0), T(1) - sin2_theta_t));
#else
	T cos_theta_t = std::sqrt(std::fmax(T(0), T(1) - sin2_theta_t));
#endif

	T r_parl = (eta * cos_theta_i - cos_theta_t)
			 / (eta * cos_theta_i + cos_theta_t);
	T r_perp = (cos_theta_i - eta * cos_theta_t)
			 / (cos_theta_i + eta * cos_theta_t);

	return (r_parl * r_parl + r_perp * r_perp) / T(2);
}

// ---------------------------------------------------------------------------
// FrComplex -- complex-valued Fresnel for conductor interfaces (pbrt-v4)
// Mirrors pbrt-v4 FrComplex(Float cosTheta_i, pstd::complex<Float> eta)
// in src/pbrt/util/scattering.h.
//
// eta_r, eta_k = real and imaginary parts of complex IOR (ÃƒÅ½Ã‚Â· + iÃƒâ€šÃ‚Â·k)
// Returns scalar reflectance in [0,1] for one wavelength channel.
//
// Implementation avoids std::complex<> for GPU compatibility:
// all complex arithmetic is expanded manually.
// ---------------------------------------------------------------------------
template <typename T>
CPU_GPU T FrComplex(T cos_theta_i, T eta_r, T eta_k) {
	// Clamp cosine to [0,1] (conductor: ray always from outside)
#if defined(__CUDACC__)
	cos_theta_i = fmaxf(T(0), fminf(T(1), cos_theta_i));
#else
	cos_theta_i = std::fmax(T(0), std::fmin(T(1), cos_theta_i));
#endif

	T sin2_i = T(1) - cos_theta_i * cos_theta_i;

	// Complex Snell's law: sin2_t = sin2_i / (eta_r + i*eta_k)^2
	// (eta_r + i*eta_k)^2 = (eta_r^2 - eta_k^2) + i*(2*eta_r*eta_k)
	T eta_r2 = eta_r * eta_r;
	T eta_k2 = eta_k * eta_k;
	T denom_r = eta_r2 - eta_k2;
	T denom_i = T(2) * eta_r * eta_k;
	// divide sin2_i (real) by complex denom: result = (a*re - b*im, -a*im + b*re) / |d|^2
	// where a = sin2_i, b = 0 (purely real numerator)
	T denom_sq = denom_r * denom_r + denom_i * denom_i;
	T sin2_t_r = sin2_i * denom_r / denom_sq;
	T sin2_t_i = -sin2_i * denom_i / denom_sq;

	// cos_t = sqrt(1 - sin2_t)  (complex sqrt)
	// 1 - sin2_t = (1 - sin2_t_r, -sin2_t_i)
	T c_r = T(1) - sin2_t_r;
	T c_i = -sin2_t_i;

	// Complex sqrt: sqrt(a + ib) = sqrt((|z|+a)/2) + i*sign(b)*sqrt((|z|-a)/2)
#if defined(__CUDACC__)
	T mag = sqrtf(c_r * c_r + c_i * c_i);
	T cos_t_r = sqrtf(fmaxf(T(0), (mag + c_r) / T(2)));
	T cos_t_i = (c_i >= T(0) ? T(1) : T(-1)) * sqrtf(fmaxf(T(0), (mag - c_r) / T(2)));
#else
	T mag = std::sqrt(c_r * c_r + c_i * c_i);
	T cos_t_r = std::sqrt(std::fmax(T(0), (mag + c_r) / T(2)));
	T cos_t_i = (c_i >= T(0) ? T(1) : T(-1)) * std::sqrt(std::fmax(T(0), (mag - c_r) / T(2)));
#endif

	// r_parl = (eta * cos_i - cos_t) / (eta * cos_i + cos_t)
	// eta * cos_i = (eta_r * cos_i, eta_k * cos_i)  (cos_i is real)
	T ec_r = eta_r * cos_theta_i - cos_t_r;
	T ec_i = eta_k * cos_theta_i - cos_t_i;
	T ed_r = eta_r * cos_theta_i + cos_t_r;
	T ed_i = eta_k * cos_theta_i + cos_t_i;
	T ed_sq = ed_r * ed_r + ed_i * ed_i;
	// complex divide: (a+ib)/(c+id) = ((ac+bd) + i(bc-ad)) / (c^2+d^2)
	T rp_r = (ec_r * ed_r + ec_i * ed_i) / ed_sq;
	T rp_i = (ec_i * ed_r - ec_r * ed_i) / ed_sq;
	T r_parl_sq = rp_r * rp_r + rp_i * rp_i;  // |r_parl|^2

	// r_perp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t)
	// eta * cos_t = (eta_r * cos_t_r - eta_k * cos_t_i, eta_r * cos_t_i + eta_k * cos_t_r)
	T etc_r = eta_r * cos_t_r - eta_k * cos_t_i;
	T etc_i = eta_r * cos_t_i + eta_k * cos_t_r;
	T nc_r = cos_theta_i - etc_r;
	T nc_i = -etc_i;
	T nd_r = cos_theta_i + etc_r;
	T nd_i = etc_i;
	T nd_sq = nd_r * nd_r + nd_i * nd_i;
	T rs_r = (nc_r * nd_r + nc_i * nd_i) / nd_sq;
	T rs_i = (nc_i * nd_r - nc_r * nd_i) / nd_sq;
	T r_perp_sq = rs_r * rs_r + rs_i * rs_i;  // |r_perp|^2

	return (r_parl_sq + r_perp_sq) / T(2);
}

// ---------------------------------------------------------------------------
// FrComplex(cosTheta, complex<T>) -- std::complex convenience overload (CPU)
// Matches the pbrt-v4 signature: FrComplex(Float cosTheta_i, complex<Float> eta)
// Reference: pbrt-v4 util/scattering.h
// ---------------------------------------------------------------------------
#if !defined(__CUDACC__)
template <typename T>
inline T FrComplex(T cos_theta_i, std::complex<T> eta) {
	return FrComplex(cos_theta_i, static_cast<T>(eta.real()), static_cast<T>(eta.imag()));
}
#endif

// ---------------------------------------------------------------------------
// FrConductorRGB -- evaluate FrComplex for R, G, B channels simultaneously
// Returns reflectance as (F_R, F_G, F_B).
// Mirrors pbrt-v4 SampledSpectrum FrComplex(cosTheta, eta[], k[]) but
// for exactly 3 RGB channels instead of NSpectrumSamples=4.
// ---------------------------------------------------------------------------
#if defined(__CUDACC__)
// GPU version: operates on float3
CPU_GPU float3 FrConductorRGB(float cos_theta_i,
									  float eta_r, float eta_g, float eta_b,
									  float k_r,   float k_g,   float k_b) {
	return make_float3(FrComplex(cos_theta_i, eta_r, k_r),
					   FrComplex(cos_theta_i, eta_g, k_g),
					   FrComplex(cos_theta_i, eta_b, k_b));
}
#endif  // __CUDACC__

// ---------------------------------------------------------------------------
// FresnelMoment1 -- first Fresnel moment integral (pbrt-v4 scattering.cpp)
// Used by NormalizedFresnelBxDF: c = 1 - 2 * FresnelMoment1(1/eta)
// Polynomial fit in two branches (eta < 1 and eta >= 1).
// ---------------------------------------------------------------------------
template <typename T>
CPU_GPU T FresnelMoment1(T eta) {
	T eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta, eta5 = eta4 * eta;
	if (eta < T(1))
		return T(0.45966f) - T(1.73965f)*eta + T(3.37668f)*eta2
			 - T(3.904945f)*eta3 + T(2.49277f)*eta4 - T(0.68441f)*eta5;
	else
		return T(-4.61686f) + T(11.1136f)*eta - T(10.4646f)*eta2
			 + T(5.11455f)*eta3 - T(1.27198f)*eta4 + T(0.12746f)*eta5;
}
