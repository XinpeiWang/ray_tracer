#pragma once

// ---------------------------------------------------------------------------
// Shared exact Fresnel function — usable on both CPU and GPU.
//
// Mirrors pbrt-v4's FrDielectric (src/pbrt/util/scattering.h), which is
// annotated PBRT_CPU_GPU so the same code runs on both paths.
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
