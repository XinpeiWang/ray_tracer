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
// Halton low-discrepancy sampler (pbrt-v4 HaltonSampler / RadicalInverse pattern)
//
// halton_radical_inverse(n, base): maps integer n to [0,1) using the radical
//   inverse construction: write n in the given base, mirror around the decimal
//   point, then convert to float.
//
//   Implementation follows pbrt-v4 lowdiscrepancy.h::RadicalInverse():
//   - integer digit accumulation (avoids float precision drift)
//   - clamped to < 1.0 via OneMinusEpsilon equivalent
//   - overflow guard: stops when reversedDigits approaches uint64 limit
//
// halton2(n, px, py): base-2 with per-pixel offset so adjacent pixels get
//   different sub-sequences (pbrt-v4 StartPixelSample pattern).
// halton3(n, px, py): base-3 with per-pixel offset.
//
// Usage:
//   pixel x offset = halton2(sampleIndex, pixelX, pixelY)
//   pixel y offset = halton3(sampleIndex, pixelX, pixelY)
//   Bounce RNG keeps using PCG32/random_double().
// ---------------------------------------------------------------------------

CPU_GPU inline float halton_radical_inverse(unsigned int n, unsigned int base) {
    // pbrt-v4 pattern: integer accumulation then single float conversion.
    // Overflow guard: stop when reversedDigits approaches uint64 limit.
    const unsigned long long limit = (~0ull) / base - base;
    unsigned long long reversedDigits = 0;
    float invBase   = 1.0f / float(base);
    float invBaseM  = 1.0f;
    while (n > 0 && reversedDigits < limit) {
        unsigned int digit = n % base;
        n /= base;
        reversedDigits = reversedDigits * base + digit;
        invBaseM *= invBase;
    }
    // Clamp to < 1.0 (matches pbrt-v4 std::min(..., OneMinusEpsilon))
    // 0x1.fffffep-1f is the largest float strictly less than 1.0
    float result = float(reversedDigits) * invBaseM;
    return result < 0x1.fffffep-1f ? result : 0x1.fffffep-1f;
}

// Per-pixel sample offset: mix pixel coordinates into the sample index so
// adjacent pixels use different sub-sequences of the Halton sequence.
// This replicates pbrt-v4 HaltonSampler::StartPixelSample() intent without
// its full pixel-stride machinery — a lightweight pixel hash offset.
CPU_GPU inline unsigned int halton_pixel_index(unsigned int sampleIndex,
                                               unsigned int px,
                                               unsigned int py) {
    // Simple but effective: offset sample index by a pixel-unique constant.
    // Uses pbrt-v4-style mixing: different primes so base-2 and base-3
    // sequences decorrelate independently across pixel dimensions.
    return sampleIndex + px * 0x9e3779b9u + py * 0x6c62272eu;
}

// Base-2 radical inverse (pixel x offset), per-pixel decorrelated
CPU_GPU inline float halton2(unsigned int sampleIndex,
                             unsigned int px = 0u,
                             unsigned int py = 0u) {
    return halton_radical_inverse(halton_pixel_index(sampleIndex, px, py), 2u);
}

// Base-3 radical inverse (pixel y offset), per-pixel decorrelated
CPU_GPU inline float halton3(unsigned int sampleIndex,
                             unsigned int px = 0u,
                             unsigned int py = 0u) {
    return halton_radical_inverse(halton_pixel_index(sampleIndex, py, px), 3u);
    // Note: halton3 swaps (py,px) so y-axis pixel correlation uses
    // a different offset pattern than x-axis, further reducing correlation.
}