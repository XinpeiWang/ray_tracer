#pragma once

// ---------------------------------------------------------------------------
// Shared Fresnel functions — usable on both CPU and GPU.
//
// Mirrors pbrt-v4's FrDielectric and FrComplex (src/pbrt/util/scattering.h),
// both annotated PBRT_CPU_GPU so the same code runs on CPU and GPU.
//
// CPU usage  (MSVC / plain C++):  CPU_GPU is empty, T = double
// GPU usage  (NVCC / CUDA):       CPU_GPU = __host__ __device__, T = float
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
// eta_r, eta_k = real and imaginary parts of complex IOR (η + i·k)
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
// FrConductorRGB -- evaluate FrComplex for R, G, B channels simultaneously
// Returns reflectance as (F_R, F_G, F_B).
// Mirrors pbrt-v4 SampledSpectrum FrComplex(cosTheta, eta[], k[]) but
// for exactly 3 RGB channels instead of NSpectrumSamples=4.
// ---------------------------------------------------------------------------
#if defined(__CUDACC__)
// GPU version: operates on float3
CPU_GPU inline float3 FrConductorRGB(float cos_theta_i,
									  float eta_r, float eta_g, float eta_b,
									  float k_r,   float k_g,   float k_b) {
	return make_float3(FrComplex(cos_theta_i, eta_r, k_r),
					   FrComplex(cos_theta_i, eta_g, k_g),
					   FrComplex(cos_theta_i, eta_b, k_b));
}
#endif  // __CUDACC__
