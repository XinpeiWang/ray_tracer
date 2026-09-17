// metal_poc_math_tests.cpp
// Real regression coverage for metal_poc_host_math.h - see that header's
// own comment for the gap this closes. Deliberately NOT a testing
// framework dependency (no GoogleTest/Catch2) - matches metal_poc_validate.cpp's
// own existing style (plain main(), assert-and-print, PASS/FAIL to
// stderr, nonzero exit on any failure) so this stays a single small
// self-contained executable, consistent with every other tool in this
// directory. No Metal/Foundation framework needed at all - every
// function under test here is plain, GPU-independent C++/simd math, so
// this compiles and runs on any Mac without a GPU, and re-runs in
// milliseconds.
//
// What this checks, and why each one specifically:
//   - linearToSRGB() against the REAL piecewise sRGB formula at several
//     points - the same numeric check PR #49's own throwaway verification
//     script did once, made permanent instead of thrown away.
//   - chromaticAberration() at an ODD width has EXACTLY zero shift at the
//     true centre pixel - a direct regression test for the real,
//     shipped PR #43 bug (a pixel-centre-vs-pixel-index coordinate
//     convention mismatch that was invisible at this POC's own default
//     EVEN resolution).
//   - The three tonemap operators (ACES/Reinhard/None) via applyToneMap()
//     - each must actually produce a DIFFERENT value above 1.0 (the
//     whole point of adding more than one), and None must be an exact
//     clamp with no rolloff at all.
//   - buildPowerLightSampler()'s own pmf field sums to 1 and matches each
//     light's true power ratio, across three constructed cases (equal
//     powers, one dominant light, all-zero emission falling back to
//     uniform) - and sampleAreaLightAliasTable() actually REPRODUCES
//     those pmf values under repeated sampling, not just that the pmf
//     field itself looks right in isolation.
//   - blackbodyColor()/vignetteFactor() basic sanity (output range,
//     known monotonic behaviour) - cheaper properties, still worth
//     locking down since both feed directly into every light colour and
//     every rendered pixel's own final brightness.

#include "metal_poc_host_math.h"

#include <cmath>
#include <cstdio>

static int g_failures = 0;

static void expectNear(const char* label, double actual, double expected, double tolerance) {
    if (std::fabs(actual - expected) > tolerance) {
        fprintf(stderr, "FAIL: %s - got %.6f, expected %.6f (tolerance %.6f)\n",
                label, actual, expected, tolerance);
        ++g_failures;
    }
}

static void expectTrue(const char* label, bool condition) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", label);
        ++g_failures;
    }
}

// The real, piecewise IEC 61966-2-1 sRGB OETF, evaluated directly (not
// linearToSRGB()'s own minimax-polynomial approximation of it) - the
// independent reference this test checks that approximation against,
// the same formula PR #49's own throwaway verification script used.
static double referenceSRGB(double linear) {
    if (linear <= 0.0031308) return 12.92 * linear;
    return 1.055 * std::pow(linear, 1.0 / 2.4) - 0.055;
}

static void testLinearToSRGB() {
    const double points[] = {0.0, 0.001, 0.0031308, 0.01, 0.18, 0.5, 1.0};
    for (double v : points) {
        double got = linearToSRGB((float)v);
        double want = referenceSRGB(v);
        char label[128];
        snprintf(label, sizeof(label), "linearToSRGB(%.6f) matches reference sRGB OETF", v);
        // The minimax polynomial is a numerically-fit approximation, not
        // the piecewise formula reimplemented - PR #49's own verification
        // found it matches to 6 decimal places; 1e-4 leaves headroom for
        // float (not double) rounding while still catching any real
        // regression (the old pow(v,1/2.2) approximation this replaced
        // was off by 3x at v=0.001, nowhere close to this tolerance).
        expectNear(label, got, want, 1e-4);
    }
}

static void testChromaticAberrationZeroShiftAtCentre() {
    // Odd width/height (901) has one true, exactly-centred pixel - the
    // same technique PR #43's own fix was originally verified with,
    // since an even-width render can hide this bug entirely (the
    // "centre" pixel's own half-pixel error rounds to the same 8-bit
    // value as its neighbour).
    const uint32_t width = 21, height = 21;
    std::vector<float> pixels(width * height * 4);
    // A radial gradient with real per-channel structure (not a flat
    // colour) - if chromaticAberration() ever resampled the "centre"
    // pixel from anywhere other than its own exact position, R/G/B would
    // disagree here; a flat test image couldn't tell the difference.
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            float fx = (float)x / (float)(width - 1);
            float fy = (float)y / (float)(height - 1);
            uint32_t i = (y * width + x) * 4;
            pixels[i + 0] = fx;
            pixels[i + 1] = fy;
            pixels[i + 2] = fx * fy;
            pixels[i + 3] = 1.0f;
        }
    }
    uint32_t cx = width / 2, cy = height / 2; // (10, 10) - the true centre pixel
    float r, g, b;
    chromaticAberration(pixels, width, height, cx, cy, /*strength=*/0.05f, &r, &g, &b);
    uint32_t centreIdx = (cy * width + cx) * 4;
    expectNear("chromaticAberration centre pixel R unshifted", r, pixels[centreIdx + 0], 1e-6);
    expectNear("chromaticAberration centre pixel G unshifted", g, pixels[centreIdx + 1], 1e-6);
    expectNear("chromaticAberration centre pixel B unshifted", b, pixels[centreIdx + 2], 1e-6);

    // An OFF-centre pixel, by contrast, must show a real shift (strength
    // > 0) - confirms this isn't just "always returns the unshifted
    // value regardless of position", which would trivially also pass
    // the centre-only check above.
    chromaticAberration(pixels, width, height, /*px=*/2, /*py=*/2, /*strength=*/0.05f, &r, &g, &b);
    uint32_t offIdx = (2 * width + 2) * 4;
    bool anyChannelShifted = std::fabs(r - pixels[offIdx + 0]) > 1e-6 ||
                              std::fabs(b - pixels[offIdx + 2]) > 1e-6;
    expectTrue("chromaticAberration off-centre pixel shows a real R/B shift", anyChannelShifted);
}

static void testToneMapModesDiverge() {
    // At a linear value comfortably above 1.0 (every ceiling light in
    // this POC's own committed scene reaches this range), the three
    // modes must disagree - if they didn't, adding Reinhard/None would
    // have been a no-op PR, not a real feature.
    const float hdrValue = 3.0f;
    float aces = applyToneMap(hdrValue, ToneMapMode::ACES);
    float reinhard = applyToneMap(hdrValue, ToneMapMode::Reinhard);
    float none = applyToneMap(hdrValue, ToneMapMode::None);

    expectTrue("ACES(3.0) is in [0,1]", aces >= 0.0f && aces <= 1.0f);
    expectTrue("Reinhard(3.0) is in [0,1]", reinhard >= 0.0f && reinhard <= 1.0f);
    // None is a hard clamp - at any value > 1.0, it must be EXACTLY 1.0,
    // no rolloff at all (the exact behaviour ACES/Reinhard were added to
    // move away from).
    expectNear("None(3.0) is an exact hard clamp to 1.0", none, 1.0, 1e-9);
    expectTrue("ACES(3.0) and Reinhard(3.0) actually differ",
               std::fabs(aces - reinhard) > 1e-4);
    expectTrue("Reinhard(3.0) matches x/(1+x) exactly",
               std::fabs(reinhard - (hdrValue / (1.0f + hdrValue))) < 1e-6);

    // Name parsing must dispatch to the same three modes applyToneMap()
    // itself switches on - a typo or case-sensitivity slip here would
    // silently fall back to ACES for every unrecognized name, exactly
    // the intended (documented) behaviour for a REAL typo, but a real
    // bug if it happens for "reinhard"/"none" themselves.
    expectTrue("parseToneMapMode(\"reinhard\") selects Reinhard",
               parseToneMapMode("reinhard") == ToneMapMode::Reinhard);
    expectTrue("parseToneMapMode(\"none\") selects None",
               parseToneMapMode("none") == ToneMapMode::None);
    expectTrue("parseToneMapMode(\"aces\") selects ACES",
               parseToneMapMode("aces") == ToneMapMode::ACES);
    expectTrue("parseToneMapMode(nullptr) falls back to ACES",
               parseToneMapMode(nullptr) == ToneMapMode::ACES);
    expectTrue("parseToneMapMode(unrecognized) falls back to ACES",
               parseToneMapMode("bogus") == ToneMapMode::ACES);
}

static AreaLightData makeTestLight(float emissionScale, float area) {
    AreaLightData light{};
    light.area = area;
    light.emission = PackedFloat3{emissionScale, emissionScale, emissionScale};
    return light;
}

static void checkPmfMatchesPowerRatio(const char* caseName, std::vector<AreaLightData>& lights,
                                       const std::vector<double>& expectedPower) {
    buildPowerLightSampler(lights);
    double total = 0.0;
    for (double p : expectedPower) total += p;
    double pmfSum = 0.0;
    for (size_t i = 0; i < lights.size(); ++i) {
        double expectedPmf = (total > 0.0) ? (expectedPower[i] / total) : (1.0 / (double)lights.size());
        char label[160];
        snprintf(label, sizeof(label), "%s: light %zu pmf matches its own power ratio", caseName, i);
        expectNear(label, lights[i].pmf, expectedPmf, 1e-4);
        pmfSum += lights[i].pmf;
    }
    char sumLabel[160];
    snprintf(sumLabel, sizeof(sumLabel), "%s: all pmf values sum to 1.0", caseName);
    expectNear(sumLabel, pmfSum, 1.0, 1e-4);
}

static void testPowerLightSamplerPmf() {
    {
        // Case 1: two lights, equal power (area * luminance identical) -
        // must reduce to exact uniform 1/N picking, same as this POC's
        // own committed scene (its two ceiling lights turned out nearly
        // equal in power too, PR #52's own honestly-reported finding).
        std::vector<AreaLightData> lights = {makeTestLight(1.0f, 1.0f), makeTestLight(1.0f, 1.0f)};
        checkPmfMatchesPowerRatio("equal-power case", lights, {1.0, 1.0});
    }
    {
        // Case 2: one light dominant by 25x - the same ratio PR #52's
        // own isolated diagnostic scene used to prove the mechanism
        // actually reduces variance (the render-level verification threw
        // this scene away afterward; this is that same test, made
        // permanent and numeric instead of visual).
        std::vector<AreaLightData> lights = {makeTestLight(1.0f, 1.0f), makeTestLight(25.0f, 1.0f)};
        checkPmfMatchesPowerRatio("25x-imbalanced case", lights, {1.0, 25.0});
    }
    {
        // Case 3: every light has zero emission (a degenerate scene this
        // POC never actually ships, but buildPowerLightSampler()'s own
        // comment explicitly documents a fallback for) - must fall back
        // to exact uniform picking, not divide by zero or produce NaN.
        std::vector<AreaLightData> lights = {makeTestLight(0.0f, 1.0f), makeTestLight(0.0f, 1.0f),
                                              makeTestLight(0.0f, 1.0f)};
        checkPmfMatchesPowerRatio("all-zero-emission fallback case", lights, {0.0, 0.0, 0.0});
    }
}

static void testPowerLightSamplerAliasTableSampling() {
    // Beyond the pmf FIELD looking right (testPowerLightSamplerPmf()
    // above), this confirms the alias table ACTUALLY reproduces those
    // probabilities when sampled the same way sampleAreaLight() samples
    // it device-side - the two-step "pmf is right" + "the table you
    // built from it is also right" verification PR #52's own review
    // should have covered with an automated test from the start.
    std::vector<AreaLightData> lights = {makeTestLight(1.0f, 1.0f), makeTestLight(3.0f, 1.0f),
                                          makeTestLight(6.0f, 1.0f)};
    buildPowerLightSampler(lights);

    const int kNumSamples = 200000;
    std::vector<long> counts(lights.size(), 0);
    // A fixed, deterministic LCG - not this project's own PCG32 (that one
    // lives device-side, in metal_poc.metal, not host C++) - any
    // decorrelated-enough sequence works here, this test only needs a
    // large, reproducible sample count, not cryptographic or path-tracer-
    // grade randomness.
    uint32_t state = 12345u;
    for (int i = 0; i < kNumSamples; ++i) {
        state = state * 1664525u + 1013904223u;
        float u = (float)state / 4294967296.0f;
        int idx = sampleAreaLightAliasTable(lights, u);
        expectTrue("sampleAreaLightAliasTable returns a valid index", idx >= 0 && idx < (int)lights.size());
        if (idx >= 0 && idx < (int)lights.size()) counts[idx]++;
    }
    for (size_t i = 0; i < lights.size(); ++i) {
        double empirical = (double)counts[i] / (double)kNumSamples;
        char label[160];
        snprintf(label, sizeof(label),
                 "alias table empirical frequency for light %zu matches its own pmf (%.4f)", i, lights[i].pmf);
        // 200k samples gives a binomial standard error well under 0.5%
        // for every pmf value in this test's own 1/10-to-6/10 range;
        // 0.01 (1 percentage point) is a generous margin above that,
        // avoiding test flakiness while still catching a real
        // construction bug (which would be off by a large, structural
        // amount, not a coin-flip's worth of noise).
        expectNear(label, empirical, lights[i].pmf, 0.01);
    }
}

static void testBlackbodyColor() {
    // The Sun's real photosphere temperature (5778 K, this POC's own
    // directional-light colour source) should read as a fairly neutral
    // white, not strongly tinted either way - a basic sanity check that
    // this isn't secretly returning a wildly wrong colour for a very
    // ordinary, frequently-used input.
    float3 sun = blackbodyColor(5778.0f);
    expectTrue("blackbodyColor(5778K) all channels in [0,1]",
               sun.x >= 0.0f && sun.x <= 1.0f && sun.y >= 0.0f && sun.y <= 1.0f &&
               sun.z >= 0.0f && sun.z <= 1.0f);
    expectTrue("blackbodyColor(5778K) reads roughly neutral (R and B within 0.15 of each other)",
               std::fabs(sun.x - sun.z) < 0.15f);

    // A low colour temperature (2700K, this POC's own "warm incandescent"
    // ceiling light) must read visibly WARMER (more red, less blue) than
    // a high one (20000K, its own "cool" ceiling light) - the whole
    // reason these two lights were given different temperatures at all.
    float3 warm = blackbodyColor(2700.0f);
    float3 cool = blackbodyColor(20000.0f);
    expectTrue("blackbodyColor(2700K) is warmer than blackbodyColor(20000K)",
               (warm.x - warm.z) > (cool.x - cool.z));

    // Every channel must stay clamped to [0,1] even far outside the
    // fit's own documented valid range - a real value this POC's own
    // hardcoded scene never passes, but a defensive property worth
    // locking down given the fit is a polynomial approximation with no
    // built-in bound outside [1000, 40000] K.
    float3 extreme = blackbodyColor(100.0f); // below the fit's own 1000K floor
    expectTrue("blackbodyColor(100K, below floor) still clamps to [0,1]",
               extreme.x >= 0.0f && extreme.x <= 1.0f && extreme.y >= 0.0f && extreme.y <= 1.0f &&
               extreme.z >= 0.0f && extreme.z <= 1.0f);
}

static void testVignetteFactor() {
    const uint32_t width = 101, height = 101;
    // 0.18 is the actual strength metal_poc.mm's own render path uses
    // (vignetteStrength, applied unconditionally to every render) - the
    // property this test cares about is "the real committed setting
    // behaves sanely," not an arbitrary strength value. vignetteFactor()
    // is a plain `1 - s*r^4` curve with no floor at all - a large enough
    // strength CAN legitimately go negative at the frame corner (solve
    // s > 1/r_max^4), which is correct behaviour for that function, not
    // a bug - downstream code (every tonemap operator) already clamps a
    // negative pre-tonemap radiance to black, so this is harmless in
    // practice; asserting non-negativity at an arbitrary large strength
    // here would be testing the wrong thing.
    const float committedStrength = 0.18f;
    float centre = vignetteFactor(width / 2, height / 2, width, height, committedStrength);
    float corner = vignetteFactor(0, 0, width, height, committedStrength);
    expectNear("vignetteFactor at frame centre is ~1.0 (no darkening)", centre, 1.0, 0.02);
    expectTrue("vignetteFactor darkens the corner relative to the centre", corner < centre);
    expectTrue("vignetteFactor stays non-negative at this POC's own actual committed strength (0.18)",
               corner >= 0.0f);

    // strength == 0.0 must be an EXACT no-op everywhere - the documented
    // "0 reproduces prior behaviour exactly" contract this POC's own
    // knobs (lensRadius, fogSigmaT, etc.) all share.
    expectNear("vignetteFactor(strength=0) is exactly 1.0 at the centre",
               vignetteFactor(width / 2, height / 2, width, height, 0.0f), 1.0, 1e-9);
    expectNear("vignetteFactor(strength=0) is exactly 1.0 at a corner too",
               vignetteFactor(0, 0, width, height, 0.0f), 1.0, 1e-9);
}

int main() {
    testLinearToSRGB();
    testChromaticAberrationZeroShiftAtCentre();
    testToneMapModesDiverge();
    testPowerLightSamplerPmf();
    testPowerLightSamplerAliasTableSampling();
    testBlackbodyColor();
    testVignetteFactor();

    if (g_failures > 0) {
        fprintf(stderr, "FAIL: %d check(s) failed\n", g_failures);
        return 1;
    }
    fprintf(stderr, "PASS: all metal_poc_host_math.h checks passed\n");
    return 0;
}
