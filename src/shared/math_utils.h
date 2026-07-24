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

// ---------------------------------------------------------------------------
// Halton low-discrepancy sampler (pbrt-v4 HaltonSampler pattern)
//
// halton_radical_inverse(n, base): maps integer n to [0,1) by writing n in
//   the given base and mirroring the digits around the decimal point.
//   e.g. n=6, base=2: 6 = 1*4+1*2+0*1 = "110" -> "0.011" = 0.375
//
// halton2(n): base-2  sequence for pixel sub-pixel offset x
// halton3(n): base-3  sequence for pixel sub-pixel offset y
//
// Usage: replace random pixel jitter with halton2(sampleIndex) / halton3(sampleIndex).
// Bounce RNG (scatter directions, light sampling) keeps using PCG32/random_double().
// ---------------------------------------------------------------------------
CPU_GPU inline float halton_radical_inverse(unsigned int n, unsigned int base) {
    float result = 0.0f;
    float invBase = 1.0f / float(base);
    float factor  = invBase;
    while (n > 0) {
        result += float(n % base) * factor;
        n      /= base;
        factor *= invBase;
    }
    return result;
}

// Base-2 radical inverse (pixel x offset)
CPU_GPU inline float halton2(unsigned int n) {
    return halton_radical_inverse(n, 2u);
}

// Base-3 radical inverse (pixel y offset)
CPU_GPU inline float halton3(unsigned int n) {
    return halton_radical_inverse(n, 3u);
}