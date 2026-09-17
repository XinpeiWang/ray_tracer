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
