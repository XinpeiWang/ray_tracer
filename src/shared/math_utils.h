#pragma once

// Shared math utilities -- CPU and GPU (pbrt-v4 pattern)
// scattering.h: Reflect, Refract | sampling.h: PowerHeuristic

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU inline
#   endif
#endif

// reflect (pbrt-v4 Reflect, PBRT_CPU_GPU)
template <typename Vec>
CPU_GPU Vec cpu_gpu_reflect(const Vec& v, const Vec& n) {
    return v - 2 * dot(v, n) * n;
}

// refract (pbrt-v4 Refract, PBRT_CPU_GPU) -- etai_over_etat = eta_i/eta_t
template <typename Vec, typename Scalar>
CPU_GPU Vec cpu_gpu_refract(const Vec& uv, const Vec& n, Scalar etai_over_etat) {
#if defined(__CUDACC__)
    Scalar cos_theta = fminf(dot(-uv, n), Scalar(1));
#else
    Scalar cos_theta = std::fmin(dot(-uv, n), Scalar(1));
#endif
    Vec r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Scalar len2 = dot(r_out_perp, r_out_perp);
#if defined(__CUDACC__)
    Vec r_out_parallel = -sqrtf(fabsf(Scalar(1) - len2)) * n;
#else
    Vec r_out_parallel = -std::sqrt(std::fabs(Scalar(1) - len2)) * n;
#endif
    return r_out_perp + r_out_parallel;
}

// PowerHeuristic (pbrt-v4 sampling.h, PBRT_CPU_GPU) -- beta=2, nf=ng=1
template <typename Scalar>
CPU_GPU Scalar PowerHeuristic(Scalar pdf_a, Scalar pdf_b) {
    if (pdf_a <= Scalar(0)) return Scalar(0);
    if (pdf_b <= Scalar(0)) return Scalar(1);
    Scalar a2 = pdf_a * pdf_a;
    Scalar b2 = pdf_b * pdf_b;
#if defined(__CUDACC__)
    if (isinf(a2)) return Scalar(1);
#else
    if (std::isinf(a2)) return Scalar(1);
#endif
    return a2 / (a2 + b2);
}