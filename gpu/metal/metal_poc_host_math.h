#pragma once
// metal_poc_host_math.h
// Pure, GPU-independent host-side math extracted out of metal_poc.mm -
// blackbody colour, tonemap operators, the sRGB OETF, chromatic
// aberration, vignetting, and the power-proportional light-sampler alias
// table. None of these need a Metal device, an acceleration structure, or
// a rendered image to exercise: every one is a plain function of its own
// scalar/vector arguments. Header-only (not a .cpp) specifically so a
// tiny CPU-only test executable (metal_poc_math_tests.cpp) can include
// this ALONE, with no Metal/Foundation framework link dependency at all,
// and exercise the exact same code metal_poc.mm's own render path calls -
// not a hand-copied "shadow" reimplementation that could silently drift
// from the real thing.
//
// This closes a real, previously-undocumented gap: every one of this
// POC's ~58 documented increments (docs/METAL_GPU_FEASIBILITY.md) was
// verified by rendering once, by hand, via a throwaway numeric/visual
// script that was never committed - a future edit to any of THESE
// specific functions could silently reintroduce a fixed bug (the exact
// PR #43 chromatic-aberration coordinate-convention bug this file's own
// test suite now regression-tests directly) with nothing in `ctest`
// catching it. See metal_poc_math_tests.cpp for what's actually checked.

#include <simd/simd.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using simd::float3;

// A plain, tightly-packed 12-byte vertex type - MTLAccelerationStructure
// TriangleGeometryDescriptor reads raw bytes at vertexStride, and simd's
// own float3 is 16-byte-aligned/padded, not 12, so a hand-rolled struct
// (no simd padding) is what actually matches a tight vertexStride.
struct PackedFloat3 {
    float x, y, z;
};

// Mirrors metal_poc.metal's UV buffer element type - simd::float2 has no
// padding-inside-a-struct issue the way float3 does (2 floats is already
// its own natural 8-byte alignment), but named separately from simd::float2
// for the same "obviously the wire format, not incidentally compatible
// with it" clarity PackedFloat3 gives the position/normal buffers.
struct PackedFloat2 {
    float u, v;
};

// Mirrors metal_poc.metal's AreaLight byte-for-byte.
struct AreaLightData {
    PackedFloat3 center;
    PackedFloat3 edgeU;
    PackedFloat3 edgeV;
    PackedFloat3 normal;
    float area;
    PackedFloat3 emission;
    // Defaults keep every light before this one exactly flat (no
    // pattern) - see metal_poc.metal's own AreaLight comment.
    float patternTileB = 0.0f;
    float patternScale = 0.0f;
    // pbrt-v4's own "bool twosided" AreaLightSource parameter (section
    // 104) - 0.0 (every light before this one) keeps the original
    // one-sided-only behaviour exactly. See metal_poc.metal's own
    // AreaLight::twoSided comment.
    float twoSided = 0.0f;
    // A pbrt-loaded scene's own image-based AreaLightSource (section
    // 105) - 0.0 (every light before this one) keeps `emission` a flat
    // direct radiance value. See metal_poc.metal's own
    // AreaLight::useTexture comment.
    float useTexture = 0.0f;
    // Power-proportional light-picking data (Vose alias table, built
    // host-side by buildPowerLightSampler() below, mirroring
    // src/shared/power_light_sampler_scaffold.h's own PowerLightSampler)
    // - see metal_poc.metal's AreaLight/sampleAreaLight() for how these
    // three get used. Defaults reproduce exact uniform 1/N picking (every
    // light before this one) if this ever got skipped: pmf = 0 would be
    // wrong, so buildPowerLightSampler() always runs, never left at these
    // raw defaults for an actual render.
    float pmf = 0.0f;
    float aliasProb = 1.0f;
    uint32_t aliasIndex = 0;
};

// Power-proportional light picking - a direct port of the Vose alias-
// table CONSTRUCTION algorithm in src/shared/power_light_sampler_scaffold.h's
// own PowerLightSampler::build() (that class itself is documented there as
// orphaned scaffolding with zero callers anywhere in this project, not
// wired into either the CPU or OptiX-GPU renderer - but the algorithm it
// implements is a real, correct port of pbrt-v4's own PowerLightSampler,
// exactly the kind of "tested reference, not proven-in-production
// caller" src/shared/ can still be worth porting from). Replaces this
// POC's own uniform 1/N area-light picking (section 18/52's own
// documented simplification: "a real port would... sample lights
// proportional to their own power"), so a bright light gets picked (and
// therefore NEE-sampled) more often than a dim one, reducing variance on
// the bright light without wasting samples equally on a light contributing
// almost nothing to the image.
//
// `power` here is `luminance(emission) * area` per light - proportional
// to each quad light's true total radiant power up to a factor (pi times
// a Lambertian-emitter solid-angle constant) that's the SAME for every
// light in this POC (all planar, all diffuse emitters), so it cancels
// out of the relative weighting a sampler only ever needs.
//
// `logLine`, when non-null, receives the same per-light diagnostic
// fprintf metal_poc.mm's own render path always prints - factored out as
// a callback (rather than an unconditional fprintf(stderr, ...) baked in
// here) so metal_poc_math_tests.cpp can call this exact function without
// spamming stderr on every test run.
inline void buildPowerLightSampler(std::vector<AreaLightData>& lights,
                                    void (*logLine)(int, double, double, double) = nullptr) {
    int n = (int)lights.size();
    if (n == 0) return;
    std::vector<double> power(n), pmf(n), scaled(n);
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        const PackedFloat3& e = lights[i].emission;
        double luminance = 0.2126 * e.x + 0.7152 * e.y + 0.0722 * e.z;
        power[i] = luminance * lights[i].area;
        total += power[i];
    }
    bool useUniform = (total <= 0.0);
    double uniformP = 1.0 / (double)n;
    for (int i = 0; i < n; ++i) {
        pmf[i] = useUniform ? uniformP : power[i] / total;
        scaled[i] = pmf[i] * (double)n;
        lights[i].pmf = (float)pmf[i];
        lights[i].aliasProb = 0.0f;
        lights[i].aliasIndex = (uint32_t)i;
    }
    std::vector<int> small, large;
    for (int i = 0; i < n; ++i) {
        (scaled[i] < 1.0 ? small : large).push_back(i);
    }
    while (!small.empty() && !large.empty()) {
        int s = small.back(); small.pop_back();
        int l = large.back(); large.pop_back();
        lights[s].aliasProb = (float)scaled[s];
        lights[s].aliasIndex = (uint32_t)l;
        scaled[l] = (scaled[l] + scaled[s]) - 1.0;
        (scaled[l] < 1.0 ? small : large).push_back(l);
    }
    while (!large.empty()) { int l = large.back(); large.pop_back(); lights[l].aliasProb = 1.0f; lights[l].aliasIndex = (uint32_t)l; }
    while (!small.empty()) { int s = small.back(); small.pop_back(); lights[s].aliasProb = 1.0f; lights[s].aliasIndex = (uint32_t)s; }
    if (logLine) {
        for (int i = 0; i < n; ++i) logLine(i, power[i], pmf[i], 1.0 / (double)n);
    }
}

// Mirrors sampleAreaLight()'s own Vose alias-table LOOKUP in
// metal_poc.metal exactly (slot = floor(u*n), frac = remainder, accept
// slot if frac < aliasProb[slot] else fall through to aliasIndex[slot]) -
// used by metal_poc_math_tests.cpp to statistically verify the table
// buildPowerLightSampler() constructs actually reproduces each light's
// own `pmf` when sampled, not just that the `pmf` FIELD itself looks
// right. Not used by metal_poc.mm's own render path (that logic lives
// device-side, in the real metal_poc.metal shader) - this is a
// host-side, test-only mirror of it, kept deliberately in lock-step with
// the shader's own comment ("ported verbatim... ") so a future edit to
// one is expected to need the same edit here.
inline int sampleAreaLightAliasTable(const std::vector<AreaLightData>& lights, float u) {
    int n = (int)lights.size();
    if (n == 0) return -1;
    int lastIdx = n - 1;
    float scaled = u * (float)n;
    int slot = std::min((int)scaled, lastIdx);
    float frac = scaled - (float)slot;
    int idx = (frac < lights[slot].aliasProb) ? slot : (int)lights[slot].aliasIndex;
    return std::min(idx, lastIdx);
}

// Blackbody-temperature-to-RGB (Tanner Helland's widely-used polynomial
// fit to the Planckian locus) - converts an actual physical temperature
// in Kelvin to an approximate linear-ish RGB tint, valid over roughly
// [1000, 40000] K (clamped at the low end, where the fit diverges).
inline float3 blackbodyColor(float kelvinIn) {
    float kelvin = fmaxf(kelvinIn, 1000.0f) / 100.0f;
    float r, g, b;
    if (kelvin <= 66.0f) {
        r = 255.0f;
    } else {
        r = 329.698727446f * powf(kelvin - 60.0f, -0.1332047592f);
    }
    if (kelvin <= 66.0f) {
        g = 99.4708025861f * logf(kelvin) - 161.1195681661f;
    } else {
        g = 288.1221695283f * powf(kelvin - 60.0f, -0.0755148492f);
    }
    if (kelvin >= 66.0f) {
        b = 255.0f;
    } else if (kelvin <= 19.0f) {
        b = 0.0f;
    } else {
        b = 138.5177312231f * logf(kelvin - 10.0f) - 305.0447927307f;
    }
    r = fminf(fmaxf(r, 0.0f), 255.0f) / 255.0f;
    g = fminf(fmaxf(g, 0.0f), 255.0f) / 255.0f;
    b = fminf(fmaxf(b, 0.0f), 255.0f) / 255.0f;
    return float3{r, g, b};
}

// Natural (lens) vignetting - real camera lenses transmit less light to
// the sensor at the edges/corners of the frame than at the centre (the
// classic `cos^4` falloff law). `strength` scales how much the corners
// darken; 0.0 is an exact no-op.
inline float vignetteFactor(uint32_t x, uint32_t y, uint32_t width, uint32_t height, float strength) {
    float nx = (float(x) + 0.5f) / float(width) * 2.0f - 1.0f;
    float ny = (float(y) + 0.5f) / float(height) * 2.0f - 1.0f;
    float r2 = nx * nx + ny * ny;
    return 1.0f - strength * r2 * r2;
}

// Bilinearly samples ONE channel of the linear HDR pixel buffer at a
// fractional (x, y). Coordinates are clamped to the buffer's own bounds.
inline float sampleChannelBilinear(const std::vector<float>& pixels, uint32_t width, uint32_t height,
                                    float x, float y, int channel) {
    x = fmaxf(0.0f, fminf(x, float(width) - 1.001f));
    y = fmaxf(0.0f, fminf(y, float(height) - 1.001f));
    uint32_t x0 = (uint32_t)x;
    uint32_t y0 = (uint32_t)y;
    uint32_t x1 = std::min(x0 + 1, width - 1);
    uint32_t y1 = std::min(y0 + 1, height - 1);
    float fx = x - float(x0);
    float fy = y - float(y0);
    float v00 = pixels[(y0 * width + x0) * 4 + channel];
    float v10 = pixels[(y0 * width + x1) * 4 + channel];
    float v01 = pixels[(y1 * width + x0) * 4 + channel];
    float v11 = pixels[(y1 * width + x1) * 4 + channel];
    float v0 = v00 * (1.0f - fx) + v10 * fx;
    float v1 = v01 * (1.0f - fx) + v11 * fx;
    return v0 * (1.0f - fy) + v1 * fy;
}

// Lateral chromatic aberration - the red channel is resampled from a
// position scaled slightly OUTWARD from frame centre, blue slightly
// INWARD, green left untouched. `strength == 0.0` is an exact no-op.
//
// Index-space centre (NOT pixel-CENTRE `+0.5` convention) - matching
// sampleChannelBilinear()'s own convention, where an integer (x, y)
// means "exactly pixel (x, y)", not "the corner before it". Mixing the
// two conventions here was a REAL, SHIPPED bug (PR #43): computing `cx`
// as `width * 0.5` while adding `+ 0.5` to `px` left the "centre" pixel's
// own `dx` at exactly 0.5 instead of 0.0, so even the un-shifted red/blue
// channels sampled a blend of two neighbouring pixels there instead of
// the exact centre pixel itself - invisible at this POC's own default
// even-width resolution, only caught by rendering an ODD-sized image (so
// a true single centre pixel exists) and finding its R/B channels had
// shifted anyway, which is mathematically impossible at true zero shift.
// metal_poc_math_tests.cpp's own `chromatic_aberration_zero_shift_at_centre`
// test regression-tests exactly this at an odd width, permanently.
inline void chromaticAberration(const std::vector<float>& pixels, uint32_t width, uint32_t height,
                                 uint32_t px, uint32_t py, float strength,
                                 float* outR, float* outG, float* outB) {
    float cx = float(width - 1) * 0.5f;
    float cy = float(height - 1) * 0.5f;
    float dx = float(px) - cx;
    float dy = float(py) - cy;
    float redX = cx + dx * (1.0f + strength);
    float redY = cy + dy * (1.0f + strength);
    float blueX = cx + dx * (1.0f - strength);
    float blueY = cy + dy * (1.0f - strength);
    *outR = sampleChannelBilinear(pixels, width, height, redX, redY, 0);
    *outG = pixels[(py * width + px) * 4 + 1];
    *outB = sampleChannelBilinear(pixels, width, height, blueX, blueY, 2);
}

// ACES filmic tonemap (Krzysztof Narkowicz's widely-used fitted
// approximation of the ACES RRT+ODT curve). Applied per channel, in
// LINEAR space, BEFORE the sRGB-ish gamma encode below.
inline float acesFilmicTonemap(float x) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    float mapped = (x * (a * x + b)) / (x * (c * x + d) + e);
    return fminf(fmaxf(mapped, 0.0f), 1.0f);
}

// Reinhard tonemap (`L' = L / (1 + L)`) - ported directly from this
// project's own CPU/OptiX-shared reference (`src/shared/tone_map.h`'s own
// `reinhard()`).
inline float reinhardTonemap(float x) {
    if (x <= 0.0f) return 0.0f;
    return x / (1.0f + x);
}

// Mirrors `src/shared/tone_map.h`'s own `ToneMapMode`.
enum class ToneMapMode {
    ACES,
    Reinhard,
    None
};

// Same three names `--tonemap` accepts on the CPU/OptiX side
// (`tone_map_mode_from_name()`). Falls back to ACES (this POC's
// long-standing default) on anything unrecognized.
inline ToneMapMode parseToneMapMode(const char* name) {
    if (!name) return ToneMapMode::ACES;
    if (strcmp(name, "reinhard") == 0) return ToneMapMode::Reinhard;
    if (strcmp(name, "none") == 0) return ToneMapMode::None;
    return ToneMapMode::ACES;
}

inline float applyToneMap(float x, ToneMapMode mode) {
    switch (mode) {
        case ToneMapMode::Reinhard: return reinhardTonemap(x);
        case ToneMapMode::None:     return fminf(fmaxf(x, 0.0f), 1.0f);
        case ToneMapMode::ACES:
        default:                    return acesFilmicTonemap(x);
    }
}

// The REAL sRGB OETF (IEC 61966-2-1) - a numerically-stable minimax
// rational-polynomial approximation of the true piecewise curve
// (`12.92 * v` below a small threshold, `1.055 * v^(1/2.4) - 0.055`
// above it), ported directly from this project's own CPU renderer
// (src/shared/color_encoding.h's own LinearToSRGB()).
inline float linearToSRGB(float value) {
    if (value <= 0.0031308f) {
        return 12.92f * value;
    }
    float s = sqrtf(value);
    float p = -0.0016829072605308378f
            + s * (  0.03453868659826638f
            + s * (  0.7642611304733891f
            + s * (  2.0041169284241644f
            + s * (  0.7551545191665577f
            + s * (-0.016202083165206348f)))));
    float q =  4.178892964897981e-7f
            + s * (-0.00004375359692957097f
            + s * (  0.03467195408529984f
            + s * (  0.6085338522168684f
            + s * (  1.8970238036421054f
            + s))));
    return p / q * value;
}

// ===========================================================================
// Environment-map importance sampling (Distribution2D) - PHASE 1
//
// `earthTexture` (equirectangularUV(), section 18) is currently only
// ever REACHED on a miss ray - there is no NEE/importance-sampling
// strategy for it at all, so a bright, spatially-concentrated region of
// the environment image (a "sun") can only ever contribute light via a
// BSDF-sampled ray getting lucky enough to escape toward it, the same
// high-variance-under-a-bright-small-light problem NEE/MIS already
// solves for every other light type in this POC.
//
// This is the FIRST of two increments closing that gap: a piecewise-
// constant 2D importance-sampling structure built once, at load time,
// from the SAME decoded image bytes metal_poc.mm already has in hand
// for `earthTexture` - mirrors pbrt-v4's own Distribution2D/
// ImageInfiniteLight construction (row-marginal CDF + per-row
// conditional CDFs, each row weighted by BOTH its own pixel luminance
// AND the equirectangular mapping's own sin(theta) solid-angle
// Jacobian, so the poles - which cover less real solid angle per pixel
// than the equator - aren't over-sampled just for having more raw
// pixels pointed at them). Deliberately NOT yet wired into the shader's
// own NEE loops or uploaded to the GPU at all - this increment is
// scoped to the sampling STRUCTURE itself, built and independently
// verified (metal_poc_math_tests.cpp) before the (considerably larger,
// touching every material's own NEE code and the miss-path's own MIS
// weight) shader-wiring half lands as its own follow-up PR.
//
// CDF-only storage (no separate `pdf`/`funcInt` field): the local PDF
// for whichever bucket a sample lands in is always recoverable from
// that bucket's own CDF slope - `(cdf[i+1]-cdf[i]) * bucketCount` - so
// storing only the CDF arrays is both sufficient and exactly what a
// GPU-side binary search would want to upload as buffers later.
struct EnvDistribution2D {
    int width = 0;
    int height = 0;
    std::vector<float> marginalCDF;     // size height+1, monotonic 0..1
    std::vector<float> conditionalCDF;  // size height*(width+1), row-major
};

// sRGB (0-255) -> linear, the real piecewise inverse of linearToSRGB()
// above - matches this project's own decode convention (the same one
// `earthTexture`'s own `_sRGB` pixel format applies automatically in
// the shader) so the importance weights below are built from LINEAR
// luminance, not raw gamma-encoded byte values.
inline float srgbByteToLinear(unsigned char c) {
    float s = c / 255.0f;
    return (s <= 0.04045f) ? (s / 12.92f) : powf((s + 0.055f) / 1.055f, 2.4f);
}

// Builds the distribution from a decoded RGBA8 image (row-major, 4
// bytes/pixel - the exact layout `stbi_load(..., 4)` already produces
// for `earthPixels` in metal_poc.mm). An all-black row/image falls back
// to a UNIFORM CDF for that row/the marginal (rather than leaving every
// entry at its default-constructed 0.0f, which would make every
// interval degenerate and any sample land in bucket 0 forever) - a
// well-defined, if unlikely-to-matter, edge case rather than undefined
// behaviour.
inline void buildEnvDistribution2D(const unsigned char* rgba, int width, int height,
                                    EnvDistribution2D& dist) {
    dist.width = width;
    dist.height = height;
    dist.marginalCDF.assign((size_t)height + 1, 0.0f);
    dist.conditionalCDF.assign((size_t)height * (width + 1), 0.0f);

    std::vector<double> rowWeight((size_t)height, 0.0);
    std::vector<double> f((size_t)width, 0.0);
    for (int row = 0; row < height; ++row) {
        double sinTheta = sin(M_PI * (row + 0.5) / (double)height);
        double rowSum = 0.0;
        for (int col = 0; col < width; ++col) {
            const unsigned char* px = rgba + ((size_t)row * width + col) * 4;
            double lr = srgbByteToLinear(px[0]);
            double lg = srgbByteToLinear(px[1]);
            double lb = srgbByteToLinear(px[2]);
            double luminance = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb;
            f[col] = luminance * sinTheta;
            rowSum += f[col];
        }
        rowWeight[row] = rowSum;
        float* cdfRow = &dist.conditionalCDF[(size_t)row * (width + 1)];
        cdfRow[0] = 0.0f;
        if (rowSum > 0.0) {
            double acc = 0.0;
            for (int col = 0; col < width; ++col) {
                acc += f[col];
                cdfRow[col + 1] = (float)(acc / rowSum);
            }
        } else {
            for (int col = 0; col < width; ++col) cdfRow[col + 1] = (float)(col + 1) / (float)width;
        }
        cdfRow[width] = 1.0f;
    }
    double totalWeight = 0.0;
    for (int row = 0; row < height; ++row) totalWeight += rowWeight[row];
    dist.marginalCDF[0] = 0.0f;
    if (totalWeight > 0.0) {
        double acc = 0.0;
        for (int row = 0; row < height; ++row) {
            acc += rowWeight[row];
            dist.marginalCDF[row + 1] = (float)(acc / totalWeight);
        }
    } else {
        for (int row = 0; row < height; ++row) dist.marginalCDF[row + 1] = (float)(row + 1) / (float)height;
    }
    dist.marginalCDF[height] = 1.0f;
}

// Same as the RGBA8 overload above, but for an already-linear, already-
// decoded float RGB image (3 floats/pixel, row-major) - pbrt_load.h's
// own InfiniteLight::imagePixels is already resolved this way (a real
// HDR/EXR-style image decodes straight to linear float, no sRGB byte
// roundtrip to reverse), so this skips srgbByteToLinear() entirely
// rather than needing a synthetic 8-bit encode/decode roundtrip just to
// reuse the other overload. Otherwise identical (same luminance
// weights/sin(theta) solid-angle Jacobian/degenerate-row fallback).
inline void buildEnvDistribution2D(const float* rgb, int width, int height,
                                    EnvDistribution2D& dist) {
    dist.width = width;
    dist.height = height;
    dist.marginalCDF.assign((size_t)height + 1, 0.0f);
    dist.conditionalCDF.assign((size_t)height * (width + 1), 0.0f);

    std::vector<double> rowWeight((size_t)height, 0.0);
    std::vector<double> f((size_t)width, 0.0);
    for (int row = 0; row < height; ++row) {
        double sinTheta = sin(M_PI * (row + 0.5) / (double)height);
        double rowSum = 0.0;
        for (int col = 0; col < width; ++col) {
            const float* px = rgb + ((size_t)row * width + col) * 3;
            double luminance = 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2];
            f[col] = luminance * sinTheta;
            rowSum += f[col];
        }
        rowWeight[row] = rowSum;
        float* cdfRow = &dist.conditionalCDF[(size_t)row * (width + 1)];
        cdfRow[0] = 0.0f;
        if (rowSum > 0.0) {
            double acc = 0.0;
            for (int col = 0; col < width; ++col) {
                acc += f[col];
                cdfRow[col + 1] = (float)(acc / rowSum);
            }
        } else {
            for (int col = 0; col < width; ++col) cdfRow[col + 1] = (float)(col + 1) / (float)width;
        }
        cdfRow[width] = 1.0f;
    }
    double totalWeight = 0.0;
    for (int row = 0; row < height; ++row) totalWeight += rowWeight[row];
    dist.marginalCDF[0] = 0.0f;
    if (totalWeight > 0.0) {
        double acc = 0.0;
        for (int row = 0; row < height; ++row) {
            acc += rowWeight[row];
            dist.marginalCDF[row + 1] = (float)(acc / totalWeight);
        }
    } else {
        for (int row = 0; row < height; ++row) dist.marginalCDF[row + 1] = (float)(row + 1) / (float)height;
    }
    dist.marginalCDF[height] = 1.0f;
}

// Largest `i` such that `cdf[i] <= u`, clamped to `[0, n-1]` - a plain
// binary search (`cdf` has `n+1` monotonic entries, `cdf[0]==0`,
// `cdf[n]==1`); the same lookup a GPU-side binary search over the
// uploaded buffer would perform, exercised here host-side first.
inline int findCdfInterval(const float* cdf, int n, float u) {
    int lo = 0, hi = n;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] <= u) lo = mid; else hi = mid;
    }
    return (lo < n - 1) ? lo : (n - 1);
}

// Draws a continuous (u,v) in [0,1)^2 from the distribution given two
// independent uniform randoms, and returns the IMAGE-SPACE pdf (a
// density w.r.t. du*dv over the unit square, NOT yet a solid-angle
// pdf - that conversion needs the equirectangular Jacobian, which is
// direction-dependent and belongs on the device side once a world
// direction exists) - mirrors pbrt-v4's own Distribution2D::
// SampleContinuous: pick the row bucket via the marginal CDF, the
// column bucket via THAT row's own conditional CDF, then linearly
// interpolate a continuous offset within each bucket from the CDF's
// own local slope.
inline void sampleEnvDistribution2D(const EnvDistribution2D& dist, float u1, float u2,
                                     float& outU, float& outV, float& outPdf) {
    int row = findCdfInterval(dist.marginalCDF.data(), dist.height, u1);
    float rowLo = dist.marginalCDF[row], rowHi = dist.marginalCDF[row + 1];
    float rowSpan = (rowHi - rowLo > 1e-9f) ? (rowHi - rowLo) : 1e-9f;
    float dv = (u1 - rowLo) / rowSpan;
    outV = (row + dv) / (float)dist.height;
    float rowPdf = rowSpan * (float)dist.height;

    const float* condRow = &dist.conditionalCDF[(size_t)row * (dist.width + 1)];
    int col = findCdfInterval(condRow, dist.width, u2);
    float colLo = condRow[col], colHi = condRow[col + 1];
    float colSpan = (colHi - colLo > 1e-9f) ? (colHi - colLo) : 1e-9f;
    float du = (u2 - colLo) / colSpan;
    outU = (col + du) / (float)dist.width;
    float colPdf = colSpan * (float)dist.width;

    outPdf = rowPdf * colPdf;
}

// Evaluates the SAME image-space pdf at an arbitrary (u,v) - not by
// drawing a new sample, but by looking up which cell (u,v) already
// falls in and reading that cell's own CDF slope. This is what a
// BSDF-sampled ray that escaped toward some direction needs for its own
// MIS weight (section 65/66's own established "power heuristic against
// what the OTHER strategy's pdf would have been for THIS exact
// direction" pattern) - the direction was picked by the BSDF, not this
// sampler, so there is no "sample" here to draw, only a density to
// evaluate at a point.
inline float pdfEnvDistribution2D(const EnvDistribution2D& dist, float u, float v) {
    int row = (int)(v * dist.height);
    if (row < 0) row = 0; else if (row >= dist.height) row = dist.height - 1;
    int col = (int)(u * dist.width);
    if (col < 0) col = 0; else if (col >= dist.width) col = dist.width - 1;
    float rowPdf = (dist.marginalCDF[row + 1] - dist.marginalCDF[row]) * (float)dist.height;
    const float* condRow = &dist.conditionalCDF[(size_t)row * (dist.width + 1)];
    float colPdf = (condRow[col + 1] - condRow[col]) * (float)dist.width;
    return rowPdf * colPdf;
}

// ===========================================================================
// GGX multi-scatter energy compensation - PHASE 1
//
// A well-known, real limitation of the single-scatter GGX model this POC's
// own `ggxD()`/`ggxG()`/`sampleGGXVNDF()` (metal_poc.metal) implement: a
// rough conductor visibly DARKENS at high roughness relative to a real
// measured metal, because light that bounces more than once between
// microfacets before finally escaping is simply discarded by a model that
// only ever accounts for a single reflection off the sampled half-vector.
// pbrt-v4's own basic `ConductorBxDF` (already ported into this POC,
// section 66) has this SAME limitation - it isn't something this port
// introduced, but a gap in the reference this POC has followed all along.
//
// Found by using Blender Cycles as a SECOND, independent reference (per
// the user's own request) while spot-checking already-merged PRs:
// `kernel/closure/bsdf_microfacet.h`'s own `microfacet_ggx_preserve_energy()`
// applies exactly this correction, via a precomputed directional-albedo
// table (`ggx_E`/`ggx_Eavg`) - Cycles' own `app/cycles_precompute.cpp`
// builds these OFFLINE via ~8-64 million Monte Carlo samples per table and
// checks in no data, generating them at build/install time instead.
//
// This phase ports the SAME technique (Kulla & Conty, "Revisiting
// Physically Based Shading at Imageworks," SIGGRAPH 2017 course notes -
// the paper Cycles' own comment cites), scoped down for this POC's own
// real-time-precompute needs: a much smaller grid/sample count computed
// ONCE at host startup (not a separate offline tool + checked-in binary
// data), reusing this POC's OWN already-verified `ggxD`/`ggxG`/`ggxG1`/
// `sampleGGXVNDF` formulas (ported below as host mirrors, matching the
// established "host mirror of a device function, kept in lock-step"
// pattern metal_poc_math_tests.cpp's own alias-table test already uses)
// rather than re-deriving the integral from scratch.
//
// Deliberately scoped to the ACHROMATIC energy_scale term only
// (`1 + (1-E)/E`) - Cycles' own extra "multi-bounce Fresnel darkening"
// refinement (a per-channel Fss/E_avg-dependent tint on top of this) is
// a real further refinement, explicitly NOT attempted here; this phase
// closes the larger, more visible energy-LOSS gap first.
// ===========================================================================

struct GGXFloat3 { float x, y, z; };
inline GGXFloat3 ggxF3(float x, float y, float z) { GGXFloat3 r{x,y,z}; return r; }
inline float ggxDot3(GGXFloat3 a, GGXFloat3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline GGXFloat3 ggxCross3(GGXFloat3 a, GGXFloat3 b) {
    return ggxF3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
inline GGXFloat3 ggxNormalize3(GGXFloat3 a) {
    float len = sqrtf(ggxDot3(a, a));
    return (len > 1e-12f) ? ggxF3(a.x/len, a.y/len, a.z/len) : a;
}

// Host mirrors of metal_poc.metal's own ggxD()/ggxLambda()/ggxG1()/ggxG() -
// identical formulas, kept in lock-step with that file (see this section's
// own header comment for why a host copy exists at all).
inline float ggxDHost(GGXFloat3 hLocal, float alphaX, float alphaY) {
    GGXFloat3 hr = ggxF3(hLocal.x / alphaX, hLocal.y / alphaY, hLocal.z);
    float lenSq = fmaxf(ggxDot3(hr, hr), 1e-12f);
    return (1.0f / (float)M_PI) / fmaxf(alphaX * alphaY * lenSq * lenSq, 1e-12f);
}
inline float ggxLambdaHost(GGXFloat3 wLocal, float alphaX, float alphaY) {
    float wz2 = fmaxf(wLocal.z * wLocal.z, 1e-12f);
    float sqrAlphaTanN = (alphaX*alphaX*wLocal.x*wLocal.x + alphaY*alphaY*wLocal.y*wLocal.y) / wz2;
    return 0.5f * (sqrtf(1.0f + sqrAlphaTanN) - 1.0f);
}
inline float ggxG1Host(GGXFloat3 wLocal, float alphaX, float alphaY) {
    return 1.0f / (1.0f + ggxLambdaHost(wLocal, alphaX, alphaY));
}
inline float ggxGHost(GGXFloat3 woLocal, GGXFloat3 wiLocal, float alphaX, float alphaY) {
    return 1.0f / (1.0f + ggxLambdaHost(woLocal, alphaX, alphaY) + ggxLambdaHost(wiLocal, alphaX, alphaY));
}

// Host mirror of metal_poc.metal's own sampleGGXVNDF() (isotropic case,
// alphaX == alphaY == alpha - this table only ever needs the isotropic
// slice, since alpha is diagonalized to a single roughness axis for this
// energy integral exactly like Cycles' own `mu`/`rough` table axes are).
// `randFn` is any `() -> float in [0,1)` source - kept generic rather than
// tied to one RNG type so a test can supply a fixed sequence.
template<typename RandFn>
inline GGXFloat3 sampleGGXVNDFHost(GGXFloat3 woLocal, float alpha, RandFn& randFn) {
    GGXFloat3 Vh = ggxNormalize3(ggxF3(alpha * woLocal.x, alpha * woLocal.y, woLocal.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    GGXFloat3 T1 = (lensq > 0.0f) ? ggxF3(-Vh.y, Vh.x, 0.0f) : ggxF3(1.0f, 0.0f, 0.0f);
    if (lensq > 0.0f) { float s = sqrtf(lensq); T1 = ggxF3(T1.x/s, T1.y/s, T1.z/s); }
    GGXFloat3 T2 = ggxCross3(Vh, T1);

    float u1 = randFn();
    float u2 = randFn();
    float r = sqrtf(u1);
    float phi = 2.0f * (float)M_PI * u2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1*t1)) + s*t2;

    GGXFloat3 Nh = ggxF3(t1*T1.x + t2*T2.x + sqrtf(fmaxf(0.0f, 1.0f - t1*t1 - t2*t2)) * Vh.x,
                          t1*T1.y + t2*T2.y + sqrtf(fmaxf(0.0f, 1.0f - t1*t1 - t2*t2)) * Vh.y,
                          t1*T1.z + t2*T2.z + sqrtf(fmaxf(0.0f, 1.0f - t1*t1 - t2*t2)) * Vh.z);
    GGXFloat3 Ne = ggxF3(alpha * Nh.x, alpha * Nh.y, fmaxf(0.0f, Nh.z));
    return ggxNormalize3(Ne);
}

// Directional albedo E(alpha, mu) = integral of the (Fresnel-less, F=1)
// GGX reflection lobe over the hemisphere, for a fixed view direction
// wo = (sqrt(1-mu^2), 0, mu). VNDF importance sampling turns this
// integral into a plain average: f(wo,wi)*cos(wi)/pdf(wi) collapses to
// exactly G(wo,wi)/G1(wo) for a VNDF-sampled wi (the SAME reduction
// section 65/66's own throughput weight already relies on, algebra
// unchanged by dropping the Fresnel term to 1) - so E is just the mean of
// that ratio over `samplesPerCell` VNDF-sampled directions, no separate
// BRDF-value/pdf bookkeeping needed.
struct GGXEnergyTable {
    int roughRes = 0;
    int muRes = 0;
    std::vector<float> E;      // size roughRes*muRes, row-major (rough-major)
    std::vector<float> Eavg;   // size roughRes
};

template<typename RandFn>
inline void buildGGXEnergyTable(int roughRes, int muRes, int samplesPerCell,
                                 GGXEnergyTable& table, RandFn& randFn) {
    table.roughRes = roughRes;
    table.muRes = muRes;
    table.E.assign((size_t)roughRes * muRes, 1.0f);
    table.Eavg.assign(roughRes, 1.0f);

    for (int ri = 0; ri < roughRes; ++ri) {
        // roughness in (0,1], alpha = roughness^2 (this POC's own
        // perceptual-roughness-to-alpha convention, materialType 4/9's
        // own `mat.ior*mat.ior` mapping).
        float roughness = ((float)ri + 0.5f) / (float)roughRes;
        float alpha = fmaxf(roughness * roughness, 0.0009f);

        for (int mi = 0; mi < muRes; ++mi) {
            float mu = ((float)mi + 0.5f) / (float)muRes;
            mu = fmaxf(mu, 0.0001f);
            GGXFloat3 wo = ggxF3(sqrtf(fmaxf(0.0f, 1.0f - mu*mu)), 0.0f, mu);

            double sum = 0.0;
            int valid = 0;
            for (int s = 0; s < samplesPerCell; ++s) {
                GGXFloat3 h = sampleGGXVNDFHost(wo, alpha, randFn);
                float cosWoH = ggxDot3(wo, h);
                GGXFloat3 wi = ggxF3(2.0f*cosWoH*h.x - wo.x, 2.0f*cosWoH*h.y - wo.y, 2.0f*cosWoH*h.z - wo.z);
                if (wi.z <= 0.0f) continue;  // below-hemisphere VNDF sample - zero contribution
                float G = ggxGHost(wo, wi, alpha, alpha);
                float G1 = ggxG1Host(wo, alpha, alpha);
                sum += (double)(G / fmaxf(G1, 1e-6f));
                ++valid;
            }
            float E = (valid > 0) ? (float)(sum / valid) : 1.0f;
            table.E[(size_t)ri * muRes + mi] = fminf(fmaxf(E, 0.0f), 1.0f);
        }

        // Eavg(alpha) = 2 * integral_0^1 E(alpha,mu)*mu dmu - the cosine-
        // weighted hemispherical average (matches Cycles' own
        // `2.0f * mu * precompute_ggx_E(...)` integrand exactly), via a
        // plain midpoint-rule sum over the same mu grid already computed.
        double avgSum = 0.0;
        for (int mi = 0; mi < muRes; ++mi) {
            float mu = ((float)mi + 0.5f) / (float)muRes;
            avgSum += 2.0 * table.E[(size_t)ri * muRes + mi] * mu * (1.0 / muRes);
        }
        table.Eavg[ri] = (float)fmin(fmax(avgSum, 0.0), 1.0);
    }
}

// Bilinear lookup of E(roughness, mu) - the same interpolation a GPU-side
// texture sample would perform, exercised host-side first (this table's
// own device-side counterpart in phase 2 will read the SAME uploaded
// buffer via an equivalent bilinear lookup, not `texture2d::sample()`,
// since this is plain float data with no image-file/sRGB concerns).
inline float sampleGGXEnergyTable(const GGXEnergyTable& table, float roughness, float mu) {
    float rf = roughness * table.roughRes - 0.5f;
    float mf = mu * table.muRes - 0.5f;
    int r0 = (int)floorf(rf), m0 = (int)floorf(mf);
    float rt = rf - r0, mt = mf - m0;
    int r1 = r0 + 1, m1 = m0 + 1;
    r0 = std::min(std::max(r0, 0), table.roughRes - 1);
    r1 = std::min(std::max(r1, 0), table.roughRes - 1);
    m0 = std::min(std::max(m0, 0), table.muRes - 1);
    m1 = std::min(std::max(m1, 0), table.muRes - 1);
    auto at = [&](int r, int m) { return table.E[(size_t)r * table.muRes + m]; };
    float e00 = at(r0, m0), e10 = at(r1, m0), e01 = at(r0, m1), e11 = at(r1, m1);
    float e0 = e00 + (e10 - e00) * rt;
    float e1 = e01 + (e11 - e01) * rt;
    return e0 + (e1 - e0) * mt;
}

// The world-space aim direction of a pbrt-v4 Goniometric/Projection
// LightSource, recovered from src/shared/pbrt_flatten.h's own
// PunctualLight::worldToLight (a row-major 3x3 world->light ROTATION,
// built by that header's worldToLightRotation() by inverting the
// LightSource directive's own CTM - see that field's own comment: both
// kinds have no "from"/"to" of their own in pbrt-v4, so a scene aims
// either one purely by rotating the CTM before the LightSource
// directive). pbrt-v4's own convention for this kind of light's
// principal axis is local +Z; a real scene's own light-aiming CTM never
// includes a Scale (same comment), so worldToLight is a pure rotation
// and its own inverse is its transpose - the world-space representation
// of local +Z is therefore transpose(worldToLight) * (0,0,1), which for
// a row-major matrix is simply worldToLight's own THIRD ROW (indices
// 6/7/8), not a second matrix inversion.
inline simd::float3 punctualLightWorldForward(const double worldToLight[9]) {
    return simd::normalize(simd::float3{(float)worldToLight[6], (float)worldToLight[7],
                                         (float)worldToLight[8]});
}

// Same derivation as punctualLightWorldForward() above, for the light's
// local +Y ("up") axis instead of +Z - the SECOND row of worldToLight
// (indices 3/4/5), by the exact same transpose-of-a-pure-rotation
// argument. Added for section 98's own real per-light goniometric/
// projection profile IMAGES: unlike the Approx (no-image) case, which
// only ever needed the aim direction, a real image has a fixed roll
// around that axis (the image's own "up") that an arbitrary world-up
// hint (metal_poc.mm's makeProjectionLight()/makeGoniometricLight()'s
// own default, fine when there's no real image to get a roll wrong)
// would get wrong whenever the scene's own aiming rotation includes
// roll. Feeding THIS into those same helpers' existing cross-product
// formula (right = cross(forward, up), the final up = cross(right,
// forward)) recovers the real roll while keeping the exact same
// handedness/sign convention those helpers already establish - safer
// than deriving right/up directly from worldToLight's own remaining
// rows, whose handedness relative to this codebase's own right/up/
// forward convention was not independently confirmed to match.
inline simd::float3 punctualLightWorldUp(const double worldToLight[9]) {
    return simd::normalize(simd::float3{(float)worldToLight[3], (float)worldToLight[4],
                                         (float)worldToLight[5]});
}
