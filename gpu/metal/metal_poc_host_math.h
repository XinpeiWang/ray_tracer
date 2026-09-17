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
