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
#include <random>

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

// A small (32x16) synthetic RGBA8 test image: black everywhere except a
// bright white 8x4 block in one quadrant (rows 8-11, cols 16-23) - a
// deliberately EXTREME case (a "sun" against a black sky) so a sampler
// that isn't actually concentrating draws where the image is bright
// would be caught immediately, not hidden in a gentle gradient.
static void buildQuadrantTestImage(std::vector<unsigned char>& rgba, int width, int height,
                                    int bx0, int by0, int bx1, int by1) {
    rgba.assign((size_t)width * height * 4, 0);
    for (int y = by0; y < by1; ++y) {
        for (int x = bx0; x < bx1; ++x) {
            unsigned char* px = &rgba[((size_t)y * width + x) * 4];
            px[0] = px[1] = px[2] = 255;
            px[3] = 255;
        }
    }
}

static void testEnvDistribution2DConcentratesOnBrightRegion() {
    const int width = 32, height = 16;
    const int bx0 = 16, by0 = 8, bx1 = 24, by1 = 12;
    std::vector<unsigned char> rgba;
    buildQuadrantTestImage(rgba, width, height, bx0, by0, bx1, by1);

    EnvDistribution2D dist;
    buildEnvDistribution2D(rgba.data(), width, height, dist);

    // Draw a large number of samples via a simple deterministic
    // low-discrepancy-ish sweep (not std::rand - keeps this test
    // reproducible with no seed to manage) and check the overwhelming
    // majority land inside the bright block's own (u,v) footprint - the
    // bright block covers only (8*4)/(32*16) = 6.25% of the image's own
    // pixel AREA, so landing there >95% of the time is only possible if
    // the sampler is genuinely concentrating draws where the image is
    // bright, not sampling uniformly.
    const int N = 20000;
    int insideBlock = 0;
    for (int i = 0; i < N; ++i) {
        float u1 = ((float)i + 0.5f) / N;
        float u2 = fmodf(u1 * 97.0f + 0.37f, 1.0f);  // decorrelate the 2nd coordinate
        float su, sv, pdf;
        sampleEnvDistribution2D(dist, u1, u2, su, sv, pdf);
        expectTrue("sampleEnvDistribution2D pdf is positive and finite",
                   pdf > 0.0f && std::isfinite(pdf));
        int col = (int)(su * width), row = (int)(sv * height);
        if (col >= bx0 && col < bx1 && row >= by0 && row < by1) ++insideBlock;
    }
    double fraction = (double)insideBlock / N;
    expectTrue("sampleEnvDistribution2D concentrates the overwhelming majority of "
               "draws inside the image's own bright block (got a lower fraction than expected)",
               fraction > 0.95);
}

// Self-consistency: pdfEnvDistribution2D(u,v), evaluated at the EXACT
// (u,v) a sample just landed on, must equal that sample's own returned
// pdf - both are reading the same underlying CDF-slope arithmetic, so
// any mismatch here means the sampling path and the evaluation path
// have DRIFTED apart (e.g. a row/col indexing mistake in one but not
// the other) - a bug the bright-block test above couldn't distinguish
// from "correct but slightly mis-weighted."
static void testEnvDistribution2DPdfMatchesSample() {
    const int width = 64, height = 32;
    std::vector<unsigned char> rgba((size_t)width * height * 4);
    // A smooth (non-degenerate, non-constant) gradient image - every
    // row/column gets a genuinely different weight, unlike the single-
    // block image above, so this exercises many different CDF buckets.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned char* px = &rgba[((size_t)y * width + x) * 4];
            px[0] = (unsigned char)(x * 255 / (width - 1));
            px[1] = (unsigned char)(y * 255 / (height - 1));
            px[2] = 128;
            px[3] = 255;
        }
    }
    EnvDistribution2D dist;
    buildEnvDistribution2D(rgba.data(), width, height, dist);

    for (int i = 0; i < 500; ++i) {
        float u1 = ((float)i + 0.5f) / 500.0f;
        float u2 = fmodf(u1 * 61.0f + 0.13f, 1.0f);
        float su, sv, samplePdf;
        sampleEnvDistribution2D(dist, u1, u2, su, sv, samplePdf);
        float evalPdf = pdfEnvDistribution2D(dist, su, sv);
        char label[128];
        snprintf(label, sizeof(label), "pdfEnvDistribution2D(%.4f,%.4f) matches its own sample's pdf", su, sv);
        expectNear(label, evalPdf, samplePdf, 1e-3);
    }
}

// A uniform (constant-colour) image must reduce to an EXACTLY uniform
// distribution over u (every column bucket the same width) - the
// direction-independent degenerate case every importance sampler
// should collapse to when there is genuinely nothing to concentrate on,
// the same "0 reproduces uniform/prior behaviour" contract this POC's
// own other optional features already follow (see vignetteFactor's own
// strength=0 case). NOT true for v (a flat image's OWN row weights
// still vary via the sin(theta) solid-angle Jacobian - only genuinely
// equal-solid-angle-per-pixel would make v uniform too, which an
// equirectangular image never is), so only the conditional (per-row)
// CDF is checked here.
static void testEnvDistribution2DUniformImageIsUniformInU() {
    const int width = 16, height = 8;
    std::vector<unsigned char> rgba((size_t)width * height * 4, 200);
    for (size_t i = 3; i < rgba.size(); i += 4) rgba[i] = 255;  // alpha
    EnvDistribution2D dist;
    buildEnvDistribution2D(rgba.data(), width, height, dist);

    const float* row0 = &dist.conditionalCDF[0];
    for (int col = 0; col <= width; ++col) {
        char label[96];
        snprintf(label, sizeof(label), "uniform image's row-0 conditional CDF is linear at col=%d", col);
        expectNear(label, row0[col], (double)col / width, 1e-5);
    }
}

// The float-RGB overload of buildEnvDistribution2D (added for pbrt-
// loaded scenes' own already-linear-decoded InfiniteLight::imagePixels
// - metal_poc.mm's own loadPbrtScene() comment) should reproduce the
// EXACT same distribution as the RGBA8 overload for an equivalent
// image, since white (255,255,255) decodes to EXACTLY linear 1.0 and
// black to EXACTLY linear 0.0 (srgbByteToLinear's own piecewise
// formula at those two endpoints) - no rounding to tolerate, so this
// checks bit-for-bit-close equality, not just "both look concentrated."
static void testEnvDistribution2DFloatOverloadMatchesByteOverload() {
    const int width = 32, height = 16;
    const int bx0 = 16, by0 = 8, bx1 = 24, by1 = 12;
    std::vector<unsigned char> rgba;
    buildQuadrantTestImage(rgba, width, height, bx0, by0, bx1, by1);
    EnvDistribution2D byteDist;
    buildEnvDistribution2D(rgba.data(), width, height, byteDist);

    std::vector<float> rgb((size_t)width * height * 3, 0.0f);
    for (int y = by0; y < by1; ++y) {
        for (int x = bx0; x < bx1; ++x) {
            float* px = &rgb[((size_t)y * width + x) * 3];
            px[0] = px[1] = px[2] = 1.0f;
        }
    }
    EnvDistribution2D floatDist;
    buildEnvDistribution2D(rgb.data(), width, height, floatDist);

    expectTrue("float-overload distribution has the same dimensions as the byte-overload one",
               floatDist.width == byteDist.width && floatDist.height == byteDist.height);
    for (size_t i = 0; i < byteDist.marginalCDF.size(); ++i) {
        char label[96];
        snprintf(label, sizeof(label), "marginalCDF[%zu] matches between float and byte overloads", i);
        expectNear(label, floatDist.marginalCDF[i], byteDist.marginalCDF[i], 1e-5);
    }
    for (size_t i = 0; i < byteDist.conditionalCDF.size(); ++i) {
        char label[96];
        snprintf(label, sizeof(label), "conditionalCDF[%zu] matches between float and byte overloads", i);
        expectNear(label, floatDist.conditionalCDF[i], byteDist.conditionalCDF[i], 1e-5);
    }
}

// GGX multi-scatter energy-compensation table (found via spot-checking
// this POC's own GGX conductor material against Blender Cycles as a
// second reference - see buildGGXEnergyTable's own header comment for the
// full "why"). Checks the well-known Kulla-Conty trend: E(roughness,mu)
// must be close to 1.0 (near-zero energy loss) at low roughness, and
// must DECREASE as roughness increases - a real, checkable physical
// property, not an arbitrary numeric coincidence.
static void testGGXEnergyTableTrend() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);
    auto randFn = [&]() { return unitDist(rng); };

    GGXEnergyTable table;
    buildGGXEnergyTable(8, 8, 2048, table, randFn);

    // Every table entry must be a valid directional albedo: in (0,1].
    for (float e : table.E) {
        expectTrue("GGXEnergyTable E value is within (0,1]", e > 0.0f && e <= 1.0001f);
    }
    for (float e : table.Eavg) {
        expectTrue("GGXEnergyTable Eavg value is within (0,1]", e > 0.0f && e <= 1.0001f);
    }

    // Near-smooth (alpha close to the 0.0009 floor) must lose almost no
    // energy at any view angle - a mirror-like surface's own single-
    // bounce reflection already captures essentially all the light.
    float smoothE = sampleGGXEnergyTable(table, 0.03f, 0.7f);
    expectTrue("GGXEnergyTable E is close to 1.0 at near-zero roughness", smoothE > 0.97f);

    // Eavg must be monotonically non-increasing as roughness increases -
    // more microfacet self-shadowing/masking at high roughness can only
    // ever lose MORE energy to inter-reflection, never less.
    for (int ri = 1; ri < table.roughRes; ++ri) {
        char label[96];
        snprintf(label, sizeof(label), "GGXEnergyTable Eavg is non-increasing at roughness bin %d", ri);
        expectTrue(label, table.Eavg[ri] <= table.Eavg[ri - 1] + 1e-3f);
    }

    // Fully rough (alpha=1) must show REAL, substantial energy loss - the
    // whole reason this table exists - not just a rounding-level dip.
    expectTrue("GGXEnergyTable Eavg at full roughness shows real energy loss (< 0.9)",
               table.Eavg[table.roughRes - 1] < 0.9f);
}

int main() {
    testLinearToSRGB();
    testChromaticAberrationZeroShiftAtCentre();
    testToneMapModesDiverge();
    testPowerLightSamplerPmf();
    testPowerLightSamplerAliasTableSampling();
    testBlackbodyColor();
    testVignetteFactor();
    testEnvDistribution2DConcentratesOnBrightRegion();
    testEnvDistribution2DPdfMatchesSample();
    testEnvDistribution2DUniformImageIsUniformInU();
    testEnvDistribution2DFloatOverloadMatchesByteOverload();
    testGGXEnergyTableTrend();

    if (g_failures > 0) {
        fprintf(stderr, "FAIL: %d check(s) failed\n", g_failures);
        return 1;
    }
    fprintf(stderr, "PASS: all metal_poc_host_math.h checks passed\n");
    return 0;
}
