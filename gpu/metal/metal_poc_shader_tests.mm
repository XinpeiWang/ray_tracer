// metal_poc_shader_tests.mm
// Real regression coverage for the DEVICE-side half of metal_poc.metal -
// the half metal_poc_math_tests.cpp (docs/METAL_GPU_FEASIBILITY.md
// section 59) explicitly could NOT close, since that one only covers
// plain host C++ math with no GPU dependency at all. Everything checked
// here only ever runs on the GPU: frDielectric(), the GGX
// distribution/masking-shadowing functions, checkerColor(),
// spotLightFalloff(), fresnelSchlickConductor(),
// henyeyGreensteinPhase(), projectionLightRadiance(), and
// sampleAreaLight()'s own REAL alias-table lookup (not
// metal_poc_math_tests.cpp's own host-side test MIRROR of it).
//
// Dispatches a handful of small, purpose-built test kernels
// (metal_poc.metal's own "Device-side unit-test kernels" section, right
// after primaryRayKernel) with known inputs, reads the results back, and
// checks them against independently-derived reference values - the same
// "call the exact same code the real render path calls, don't
// reimplement it to test it" principle metal_poc_math_tests.cpp already
// established, just on the GPU side of the boundary instead of the host
// side. Needs a real Metal device (unlike metal_poc_math_tests.cpp),
// same MTLCopyAllDevices()/newLibraryWithSource: pattern metal_poc.mm's
// own main() uses - this only runs on a real Mac with Apple Silicon or a
// Metal-capable GPU, same requirement metal_poc/metal_poc_smoke_render
// already have.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <simd/simd.h>

#include "metal_poc_host_math.h"

using simd::float3;

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

// PackedFloat3 itself now comes from metal_poc_host_math.h (included
// above for EnvDistribution2D/buildEnvDistribution2D, section 71) -
// identical 12-byte layout to what this file used to define separately
// here, just no longer duplicated now that a real dependency on that
// header exists anyway.

// Mirrors metal_poc.metal's ProjectionLight byte-for-byte (see that
// struct's own comment) - needed to build the constant-buffer argument
// test_projectionLightRadiance expects.
struct ProjectionLightGPU {
    PackedFloat3 position;
    PackedFloat3 forward;
    PackedFloat3 right;
    PackedFloat3 up;
    float tanHalfFovX;
    float tanHalfFovY;
    float scale;
};

// Mirrors metal_poc.metal's GoniometricLight byte-for-byte - needed to
// build the constant-buffer argument test_goniometricLightRadiance
// expects.
struct GoniometricLightGPU {
    PackedFloat3 position;
    PackedFloat3 forward;
    PackedFloat3 right;
    PackedFloat3 up;
    PackedFloat3 emission;
    float scale;
};

// Mirrors metal_poc.metal's AreaLight byte-for-byte - needed to build the
// light list test_sampleAreaLight_pmf dispatches against.
struct AreaLightGPU {
    PackedFloat3 center;
    PackedFloat3 edgeU;
    PackedFloat3 edgeV;
    PackedFloat3 normal;
    float area;
    PackedFloat3 emission;
    float patternTileB;
    float patternScale;
    float pmf;
    float aliasProb;
    uint32_t aliasIndex;
};

// A direct, deliberately-independent re-implementation of
// buildPowerLightSampler()'s own Vose alias-table CONSTRUCTION (see
// metal_poc_host_math.h - not called from here, that header pulls in
// simd/PackedFloat3 definitions this file already declares its own
// versions of, and duplicating this one small function avoids a type
// clash between the two). Builds the SAME alias-table shape
// test_sampleAreaLight_pmf's own kernel then samples device-side -
// metal_poc_math_tests.cpp already checks this exact CONSTRUCTION
// algorithm's own correctness on the host side; this file's job is
// checking the DEVICE-side lookup against it, not re-deriving the
// construction step's own correctness a second time.
static void buildAliasTable(std::vector<AreaLightGPU>& lights) {
    int n = (int)lights.size();
    std::vector<double> pmf(n), scaled(n);
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        const PackedFloat3& e = lights[i].emission;
        total += (0.2126 * e.x + 0.7152 * e.y + 0.0722 * e.z) * lights[i].area;
    }
    for (int i = 0; i < n; ++i) {
        const PackedFloat3& e = lights[i].emission;
        double power = (0.2126 * e.x + 0.7152 * e.y + 0.0722 * e.z) * lights[i].area;
        pmf[i] = power / total;
        scaled[i] = pmf[i] * n;
        lights[i].pmf = (float)pmf[i];
        lights[i].aliasProb = 0.0f;
        lights[i].aliasIndex = (uint32_t)i;
    }
    std::vector<int> small, large;
    for (int i = 0; i < n; ++i) (scaled[i] < 1.0 ? small : large).push_back(i);
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
}

// Runs `functionName`, a test kernel taking a fixed argument list of
// device buffers (bound in order at buffer(0), buffer(1), ...) plus an
// optional texture at texture(0), dispatched over `count` threads.
static bool runKernel(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue,
                      NSString* functionName, NSArray<id<MTLBuffer>>* buffers,
                      id<MTLTexture> texture, uint32_t count) {
    NSError* error = nil;
    id<MTLFunction> fn = [library newFunctionWithName:functionName];
    if (!fn) {
        fprintf(stderr, "FAIL: no such kernel function %s\n", functionName.UTF8String);
        return false;
    }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "FAIL: pipeline creation for %s failed: %s\n",
                functionName.UTF8String, error.localizedDescription.UTF8String);
        return false;
    }
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    NSUInteger idx = 0;
    for (id<MTLBuffer> buf in buffers) {
        [enc setBuffer:buf offset:0 atIndex:idx];
        ++idx;
    }
    if (texture) {
        [enc setTexture:texture atIndex:0];
    }
    NSUInteger w = MIN((NSUInteger)count, pipeline.threadExecutionWidth);
    [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "FAIL: %s dispatch failed: %s\n", functionName.UTF8String, cmd.error.localizedDescription.UTF8String);
        return false;
    }
    return true;
}

static id<MTLBuffer> makeBuffer(id<MTLDevice> device, const void* data, size_t length) {
    return [device newBufferWithBytes:data length:length options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> makeOutputBuffer(id<MTLDevice> device, size_t length) {
    return [device newBufferWithLength:length options:MTLResourceStorageModeShared];
}

static void testFrDielectric(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // Reference values computed independently via a standalone double-
    // precision C program mirroring FrDielectric's own formula (not
    // copy-pasted from an earlier session's memory) - see this PR's own
    // commit message for that program's exact output.
    struct { float cosThetaI, eta, expected; } cases[] = {
        {1.0f, 1.5f, 0.04000000f},          // normal incidence: R0 = ((eta-1)/(eta+1))^2 exactly
        {0.5f, 1.5f, 0.08918671f},          // 60 degrees
        {0.08715574f, 1.5f, 0.61279965f},   // 85 degrees (near-grazing)
        {0.17364818f, 1.0f / 1.5f, 1.0f},   // total internal reflection (glass -> air, steep angle)
    };
    int n = 4;
    std::vector<simd::float2> inputs(n);
    for (int i = 0; i < n; ++i) inputs[i] = simd::float2{cases[i].cosThetaI, cases[i].eta};
    id<MTLBuffer> inBuf = makeBuffer(device, inputs.data(), inputs.size() * sizeof(simd::float2));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_frDielectric", @[inBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    for (int i = 0; i < n; ++i) {
        char label[128];
        snprintf(label, sizeof(label), "frDielectric(cos=%.5f, eta=%.5f)", cases[i].cosThetaI, cases[i].eta);
        expectNear(label, out[i], cases[i].expected, 1e-4);
    }
}

static void testGgxD(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // At normal incidence (hLocal == the shading normal), ggxD() has an
    // exact closed form regardless of alpha: D = 1/(pi*alpha^2) - see
    // that function's own definition (hr collapses to (0,0,1), lenSq
    // becomes exactly 1).
    float alphas[] = {0.05f, 0.2f, 0.5f, 0.9f};
    int n = 4;
    id<MTLBuffer> inBuf = makeBuffer(device, alphas, sizeof(alphas));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_ggxD", @[inBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    for (int i = 0; i < n; ++i) {
        double expected = 1.0 / (M_PI * alphas[i] * alphas[i]);
        char label[128];
        snprintf(label, sizeof(label), "ggxD at normal incidence, alpha=%.3f", alphas[i]);
        expectNear(label, out[i], expected, expected * 1e-3);
    }
}

static void testGgxG1SmoothLimit(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // Near-zero roughness must make G1 approach 1 (no masking/shadowing
    // at all) for any direction with a positive z - the smooth-surface
    // limit every microfacet model must reduce to.
    simd::float3 dirs[] = {
        simd::normalize(simd::float3{0.0f, 0.0f, 1.0f}),
        simd::normalize(simd::float3{0.3f, 0.4f, 0.8f}),
        simd::normalize(simd::float3{0.6f, -0.5f, 0.5f}),
    };
    int n = 3;
    id<MTLBuffer> inBuf = makeBuffer(device, dirs, sizeof(dirs));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_ggxG1_smoothLimit", @[inBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    for (int i = 0; i < n; ++i) {
        char label[128];
        snprintf(label, sizeof(label), "ggxG1 approaches 1 at near-zero roughness, direction %d", i);
        expectNear(label, out[i], 1.0, 0.01);
    }
}

static void testCheckerColor(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // scale=2 over uv in [0,1]: tile = floor(uv*2), parity = (tile.x+tile.y) mod 2.
    // (0.25,0.25) -> tile(0,0) parity 0 -> colorA=(1,0,0)
    // (0.75,0.25) -> tile(1,0) parity 1 -> colorB=(0,1,0)
    // (0.25,0.75) -> tile(0,1) parity 1 -> colorB
    // (0.75,0.75) -> tile(1,1) parity 0 -> colorA
    simd::float2 uvs[] = {{0.25f, 0.25f}, {0.75f, 0.25f}, {0.25f, 0.75f}, {0.75f, 0.75f}};
    bool expectA[] = {true, false, false, true};
    int n = 4;
    id<MTLBuffer> inBuf = makeBuffer(device, uvs, sizeof(uvs));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    if (!runKernel(device, library, queue, @"test_checkerColor", @[inBuf, outBuf], nil, n)) return;
    simd::float3* out = (simd::float3*)outBuf.contents;
    for (int i = 0; i < n; ++i) {
        char label[128];
        snprintf(label, sizeof(label), "checkerColor uv=(%.2f,%.2f) picks the expected tile", uvs[i].x, uvs[i].y);
        simd::float3 expected = expectA[i] ? simd::float3{1, 0, 0} : simd::float3{0, 1, 0};
        expectNear(label, simd::length(out[i] - expected), 0.0, 1e-5);
    }
}

static void testSpotLightFalloff(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    simd::float3 wis[3] = {
        simd::float3{0, 0, 1},   // exactly aligned with the spot's own aim
        simd::float3{0, 0, 1},
        simd::float3{1, 0, 0},   // 90 degrees off-axis
    };
    simd::float3 dirs[3] = {
        simd::float3{0, 0, 1},
        simd::float3{0, 0, 1},
        simd::float3{0, 0, 1},
    };
    simd::float2 angles[3] = {
        {-1.0f, -1.0f},          // omnidirectional flag - always full weight regardless of direction
        {0.5f, 0.9f},            // aligned (cosAngle=1) is inside the inner cone -> full weight
        {0.5f, 0.9f},            // 90 degrees (cosAngle=0) is outside the outer cone -> zero
    };
    float expected[3] = {1.0f, 1.0f, 0.0f};
    int n = 3;
    id<MTLBuffer> wiBuf = makeBuffer(device, wis, sizeof(wis));
    id<MTLBuffer> dirBuf = makeBuffer(device, dirs, sizeof(dirs));
    id<MTLBuffer> angleBuf = makeBuffer(device, angles, sizeof(angles));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_spotLightFalloff", @[wiBuf, dirBuf, angleBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    const char* names[3] = {"omnidirectional flag ignores direction", "aligned direction inside inner cone",
                             "90-degree direction outside outer cone"};
    for (int i = 0; i < n; ++i) {
        expectNear(names[i], out[i], expected[i], 1e-5);
    }
}

static void testFresnelSchlickConductor(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    float cosThetas[2] = {1.0f, 0.0f};
    simd::float3 f0s[2] = {simd::float3{0.9f, 0.6f, 0.2f}, simd::float3{0.9f, 0.6f, 0.2f}};
    int n = 2;
    id<MTLBuffer> cosBuf = makeBuffer(device, cosThetas, sizeof(cosThetas));
    id<MTLBuffer> f0Buf = makeBuffer(device, f0s, sizeof(f0s));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    if (!runKernel(device, library, queue, @"test_fresnelSchlickConductor", @[cosBuf, f0Buf, outBuf], nil, n)) return;
    simd::float3* out = (simd::float3*)outBuf.contents;
    // Normal incidence: F(1, F0) == F0 exactly (t = (1-1)^5 = 0).
    expectNear("fresnelSchlickConductor at normal incidence equals F0 exactly",
               simd::length(out[0] - f0s[0]), 0.0, 1e-5);
    // Grazing incidence: F(0, F0) == (1,1,1) exactly (t = (1-0)^5 = 1), regardless of F0.
    expectNear("fresnelSchlickConductor at grazing incidence is exactly white",
               simd::length(out[1] - simd::float3{1, 1, 1}), 0.0, 1e-5);
}

static void testFrComplexRGB(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // Reference values from a fresh standalone double-precision C port of
    // src/shared/fresnel.h's FrComplex (frcomplex_ref.c, not reused from
    // any earlier session): real gold (Au) and copper (Cu) complex IOR at
    // normal incidence, plus a k=0 degenerate case that must reduce to
    // the ORDINARY real-valued dielectric Fresnel formula exactly
    // (R0 = ((eta-1)/(eta+1))^2 = 0.04 for eta=1.5).
    float cosThetas[3] = {1.0f, 1.0f, 1.0f};
    simd::float3 etas[3] = {
        simd::float3{0.184f, 0.457f, 1.354f},  // Au
        simd::float3{0.246f, 1.072f, 1.155f},  // Cu
        simd::float3{1.5f, 1.5f, 1.5f},        // k=0 dielectric-degenerate
    };
    simd::float3 ks[3] = {
        simd::float3{3.070f, 2.408f, 1.818f},
        simd::float3{3.378f, 2.591f, 2.469f},
        simd::float3{0.0f, 0.0f, 0.0f},
    };
    simd::float3 expected[3] = {
        simd::float3{0.932020f, 0.769230f, 0.387776f},
        simd::float3{0.924094f, 0.610411f, 0.569832f},
        simd::float3{0.04f, 0.04f, 0.04f},
    };
    int n = 3;
    id<MTLBuffer> cosBuf = makeBuffer(device, cosThetas, sizeof(cosThetas));
    id<MTLBuffer> etaBuf = makeBuffer(device, etas, sizeof(etas));
    id<MTLBuffer> kBuf = makeBuffer(device, ks, sizeof(ks));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    if (!runKernel(device, library, queue, @"test_frComplexRGB", @[cosBuf, etaBuf, kBuf, outBuf], nil, n)) return;
    simd::float3* out = (simd::float3*)outBuf.contents;
    const char* names[3] = {"frComplexRGB matches reference for gold (Au) at normal incidence",
                             "frComplexRGB matches reference for copper (Cu) at normal incidence",
                             "frComplexRGB with k=0 reduces to the real dielectric Fresnel formula"};
    for (int i = 0; i < n; ++i) {
        expectNear(names[i], simd::length(out[i] - expected[i]), 0.0, 1e-4);
    }
}

static void testBuildAnisotropicOnb(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // Orthonormality check (tangent/bitangent/normal mutually
    // perpendicular, all unit length) at a spread of directions
    // INCLUDING both exact poles (0,0,+-1) and directions straddling the
    // OLD construction's own 0.999-dot-product discontinuity threshold
    // (see buildAnisotropicOnb's own comment) - the new construction has
    // no singularity anywhere, so every one of these must pass, not just
    // the "easy" arbitrary directions a pre-fix test might have picked.
    simd::float3 normals[6] = {
        simd::normalize(simd::float3{0.0f, 1.0f, 0.0f}),
        simd::normalize(simd::float3{0.0f, -1.0f, 0.0f}),
        simd::normalize(simd::float3{0.0f, 0.0f, 1.0f}),
        simd::normalize(simd::float3{0.02f, 0.9998f, 0.0f}),   // just inside the old 0.999 threshold
        simd::normalize(simd::float3{0.05f, 0.9990f, 0.0f}),   // just outside it
        simd::normalize(simd::float3{0.3f, 0.5f, 0.8124f}),
    };
    int n = 6;
    id<MTLBuffer> normalBuf = makeBuffer(device, normals, sizeof(normals));
    id<MTLBuffer> tangentBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    id<MTLBuffer> bitangentBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    if (!runKernel(device, library, queue, @"test_buildAnisotropicOnb",
                   @[normalBuf, tangentBuf, bitangentBuf], nil, n)) return;
    simd::float3* tangents = (simd::float3*)tangentBuf.contents;
    simd::float3* bitangents = (simd::float3*)bitangentBuf.contents;
    for (int i = 0; i < n; ++i) {
        char label[160];
        simd::float3 t = tangents[i], b = bitangents[i], nrm = normals[i];
        snprintf(label, sizeof(label), "buildAnisotropicOnb tangent is unit length (case %d)", i);
        expectNear(label, simd::length(t), 1.0, 1e-4);
        snprintf(label, sizeof(label), "buildAnisotropicOnb bitangent is unit length (case %d)", i);
        expectNear(label, simd::length(b), 1.0, 1e-4);
        snprintf(label, sizeof(label), "buildAnisotropicOnb tangent perpendicular to normal (case %d)", i);
        expectNear(label, simd::dot(t, nrm), 0.0, 1e-4);
        snprintf(label, sizeof(label), "buildAnisotropicOnb bitangent perpendicular to normal (case %d)", i);
        expectNear(label, simd::dot(b, nrm), 0.0, 1e-4);
        snprintf(label, sizeof(label), "buildAnisotropicOnb tangent perpendicular to bitangent (case %d)", i);
        expectNear(label, simd::dot(t, b), 0.0, 1e-4);
    }
}

static void testEnvironmentDirectionSampling(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // Same synthetic "bright block against a black background" image
    // metal_poc_math_tests.cpp's own testEnvDistribution2DConcentrates
    // OnBrightRegion() uses (section 69, phase 1) - reused here via
    // buildEnvDistribution2D() itself (the exact same host-side function
    // metal_poc.mm's own render path calls to build the real CDF arrays
    // this test then uploads) rather than hand-deriving a device-side-
    // only fixture, so this test exercises the SAME CDF layout the real
    // render path produces, not a hand-rolled approximation of it.
    const int width = 32, height = 16;
    std::vector<unsigned char> rgba((size_t)width * height * 4, 0);
    for (int y = 8; y < 12; ++y) {
        for (int x = 16; x < 24; ++x) {
            unsigned char* px = &rgba[((size_t)y * width + x) * 4];
            px[0] = px[1] = px[2] = 255;
            px[3] = 255;
        }
    }
    EnvDistribution2D dist;
    buildEnvDistribution2D(rgba.data(), width, height, dist);

    id<MTLBuffer> marginalBuf = makeBuffer(device, dist.marginalCDF.data(), dist.marginalCDF.size() * sizeof(float));
    id<MTLBuffer> conditionalBuf = makeBuffer(device, dist.conditionalCDF.data(), dist.conditionalCDF.size() * sizeof(float));
    simd::int2 dims[1] = {{width, height}};
    id<MTLBuffer> dimsBuf = makeBuffer(device, dims, sizeof(dims));

    const int n = 64;
    simd::float2 uvSamples[n];
    for (int i = 0; i < n; ++i) {
        float u1 = ((float)i + 0.5f) / n;
        uvSamples[i] = {u1, fmodf(u1 * 71.0f + 0.19f, 1.0f)};
    }
    id<MTLBuffer> uvBuf = makeBuffer(device, uvSamples, sizeof(uvSamples));
    id<MTLBuffer> dirOutBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    id<MTLBuffer> pdfOutBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_sampleEnvironmentDirection",
                   @[marginalBuf, conditionalBuf, dimsBuf, uvBuf, dirOutBuf, pdfOutBuf], nil, n)) return;
    simd::float3* dirs = (simd::float3*)dirOutBuf.contents;
    float* samplePdfs = (float*)pdfOutBuf.contents;

    // Every sampled direction must be unit length and every pdf strictly
    // positive/finite - the same basic sanity check the host-side
    // sampler's own test already established, now for the device-side
    // port of it.
    for (int i = 0; i < n; ++i) {
        char label[96];
        snprintf(label, sizeof(label), "sampleEnvironmentDirection returns a unit direction (case %d)", i);
        expectNear(label, simd::length(dirs[i]), 1.0, 1e-3);
        expectTrue("sampleEnvironmentDirection pdf is positive and finite",
                   samplePdfs[i] > 0.0f && std::isfinite(samplePdfs[i]));
    }

    // Self-consistency (mirrors metal_poc_math_tests.cpp's own
    // testEnvDistribution2DPdfMatchesSample()): re-evaluating
    // pdfEnvironmentDirection() at each sample's own returned direction
    // must reproduce that SAME sample's own pdf - the sampling path and
    // the evaluation path read the same underlying CDF arithmetic (plus
    // the same equirectangular Jacobian), so any drift between them
    // would show up here.
    id<MTLBuffer> dirInBuf = makeBuffer(device, dirs, n * sizeof(simd::float3));
    id<MTLBuffer> evalPdfOutBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_pdfEnvironmentDirection",
                   @[marginalBuf, conditionalBuf, dimsBuf, dirInBuf, evalPdfOutBuf], nil, n)) return;
    float* evalPdfs = (float*)evalPdfOutBuf.contents;
    for (int i = 0; i < n; ++i) {
        char label[96];
        snprintf(label, sizeof(label), "pdfEnvironmentDirection matches its own sample's pdf (case %d)", i);
        expectNear(label, evalPdfs[i], samplePdfs[i], 1e-2);
    }
}

static void testHenyeyGreensteinPhase(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // g == 0 (isotropic) must give the SAME value - 1/(4*pi) - for every
    // cosTheta, since an isotropic phase function has no directional
    // dependence at all.
    simd::float2 inputs[3] = {{0.3f, 0.0f}, {-0.7f, 0.0f}, {1.0f, 0.0f}};
    int n = 3;
    id<MTLBuffer> inBuf = makeBuffer(device, inputs, sizeof(inputs));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_henyeyGreensteinPhase", @[inBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    double expected = 1.0 / (4.0 * M_PI);
    for (int i = 0; i < n; ++i) {
        char label[128];
        snprintf(label, sizeof(label), "henyeyGreensteinPhase(g=0) is isotropic at cosTheta=%.2f", inputs[i].x);
        expectNear(label, out[i], expected, 1e-5);
    }
}

static void testProjectionLightRadiance(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // A tiny 4x4 test image, every texel a distinct flat colour except
    // one marked texel (index (2,2), Metal's own floor(uv*size) texel
    // for uv exactly at 0.5 - the projector's own dead-centre direction)
    // set to a bright, unmistakable red so a nearest-filtered sample at
    // the frustum's own centre can be checked exactly, not just
    // "produced SOME nonzero colour."
    const int texSize = 4;
    std::vector<uint8_t> pixels(texSize * texSize * 4, 0);
    for (int i = 0; i < texSize * texSize; ++i) pixels[i * 4 + 3] = 255; // alpha
    int centreTexel = (2 * texSize + 2) * 4;
    pixels[centreTexel + 0] = 255; // pure red at the centre texel

    MTLTextureDescriptor* texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                        width:texSize height:texSize mipmapped:NO];
    texDesc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> testImage = [device newTextureWithDescriptor:texDesc];
    [testImage replaceRegion:MTLRegionMake2D(0, 0, texSize, texSize) mipmapLevel:0
                    withBytes:pixels.data() bytesPerRow:texSize * 4];

    // A projector looking straight down -Z, 40-degree vertical FOV,
    // square aspect - forward/right/up form a trivial identity-ish
    // orthonormal frame, chosen so the expected screen-space mapping for
    // each test direction can be worked out by hand rather than needing
    // a second, independent "look-at" construction to trust.
    ProjectionLightGPU light{};
    light.position = PackedFloat3{0, 0, 0};
    light.forward = PackedFloat3{0, 0, -1};
    light.right = PackedFloat3{1, 0, 0};
    light.up = PackedFloat3{0, 1, 0};
    light.tanHalfFovX = tanf(20.0f * (float)M_PI / 180.0f);
    light.tanHalfFovY = tanf(20.0f * (float)M_PI / 180.0f);
    light.scale = 1.0f;

    simd::float3 wiFromLights[3] = {
        simd::float3{0, 0, -1},   // dead centre - must land on the marked red texel
        simd::float3{0, 0, 1},    // directly BEHIND the projector - must be exactly black
        simd::float3{1, 0, -0.01f}, // far outside the +/-20 degree frustum - must be exactly black
    };
    int n = 3;
    id<MTLBuffer> wiBuf = makeBuffer(device, wiFromLights, sizeof(wiFromLights));
    id<MTLBuffer> lightBuf = makeBuffer(device, &light, sizeof(light));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    if (!runKernel(device, library, queue, @"test_projectionLightRadiance", @[wiBuf, lightBuf, outBuf], testImage, n)) return;
    simd::float3* out = (simd::float3*)outBuf.contents;

    expectTrue("projectionLightRadiance at dead centre samples the marked red texel (R channel bright)",
               out[0].x > 0.5f);
    expectNear("projectionLightRadiance at dead centre has near-zero G", out[0].y, 0.0, 0.05);
    expectNear("projectionLightRadiance directly behind the projector is exactly black",
               simd::length(out[1]), 0.0, 1e-6);
    expectNear("projectionLightRadiance outside the frustum is exactly black",
               simd::length(out[2]), 0.0, 1e-6);
}

static void testSampleAreaLightAliasTable(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // Three lights, deliberately imbalanced (1x/3x/6x power, same shape
    // metal_poc_math_tests.cpp's own testPowerLightSamplerAliasTableSampling
    // uses for the HOST-side mirror) - this is the REAL device-side
    // sampleAreaLight() that mirror stood in for.
    std::vector<AreaLightGPU> lights(3);
    float emissions[3] = {1.0f, 3.0f, 6.0f};
    for (int i = 0; i < 3; ++i) {
        lights[i] = AreaLightGPU{};
        lights[i].area = 1.0f;
        lights[i].emission = PackedFloat3{emissions[i], emissions[i], emissions[i]};
        lights[i].center = PackedFloat3{0, 0, 0};
        lights[i].edgeU = PackedFloat3{1, 0, 0};
        lights[i].edgeV = PackedFloat3{0, 1, 0};
        lights[i].normal = PackedFloat3{0, 0, 1};
    }
    buildAliasTable(lights);

    uint32_t lightCount = 3;
    uint32_t seed = 777u;
    const uint32_t kNumSamples = 200000;
    id<MTLBuffer> lightsBuf = makeBuffer(device, lights.data(), lights.size() * sizeof(AreaLightGPU));
    id<MTLBuffer> countBuf = makeBuffer(device, &lightCount, sizeof(lightCount));
    id<MTLBuffer> seedBuf = makeBuffer(device, &seed, sizeof(seed));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, kNumSamples * sizeof(float));
    if (!runKernel(device, library, queue, @"test_sampleAreaLight_pmf",
                   @[lightsBuf, countBuf, seedBuf, outBuf], nil, kNumSamples)) return;
    float* out = (float*)outBuf.contents;

    long counts[3] = {0, 0, 0};
    long unmatched = 0;
    for (uint32_t i = 0; i < kNumSamples; ++i) {
        bool matched = false;
        for (int j = 0; j < 3; ++j) {
            if (std::fabs(out[i] - lights[j].pmf) < 1e-5f) { counts[j]++; matched = true; break; }
        }
        if (!matched) ++unmatched;
    }
    expectTrue("every sampleAreaLight() pmf output matches one of the three known light pmfs",
               unmatched == 0);
    for (int j = 0; j < 3; ++j) {
        double empirical = (double)counts[j] / (double)kNumSamples;
        char label[160];
        snprintf(label, sizeof(label),
                 "device-side sampleAreaLight() empirical frequency for light %d matches its own pmf (%.4f)",
                 j, lights[j].pmf);
        // Same 0.01 absolute-tolerance reasoning
        // metal_poc_math_tests.cpp's own host-side alias-table test uses
        // at this sample count.
        expectNear(label, empirical, lights[j].pmf, 0.01);
    }
}

static void testEqualAreaSphereToSquare(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // Reference values computed independently via a standalone double-
    // precision C program mirroring EqualAreaSphereToSquare's own formula
    // (src/shared/sampling_extra.h) - not copied from memory - see this
    // PR's own commit message for that program's exact output. Added
    // alongside PR #57's own GoniometricLight increment, which is the
    // first (and so far only) caller of this mapping in this file.
    // The fourth direction (0.3, 0.4, 0.866025) is already unit-length
    // (0.3^2 + 0.4^2 + 0.866025^2 = 1.0), matching the exact triple the
    // reference program used - written out explicitly rather than via
    // simd::normalize() so it's obviously the SAME input, not a
    // renormalized approximation of one.
    simd::float3 dirs[4] = {
        simd::float3{0, 0, 1},           // dead centre of the square
        simd::float3{0, 0, -1},          // opposite pole - the far corner (1,1)
        simd::float3{1, 0, 0},           // equator, +X axis
        simd::float3{0.3f, 0.4f, 0.866025f},
    };

    simd::float2 expected[4] = {
        {0.5f, 0.5f},
        {1.0f, 1.0f},
        {0.99999797f, 0.50000203f},
        {0.57497354f, 0.60803916f},
    };
    int n = 4;
    id<MTLBuffer> inBuf = makeBuffer(device, dirs, sizeof(dirs));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(simd::float2));
    if (!runKernel(device, library, queue, @"test_equalAreaSphereToSquare", @[inBuf, outBuf], nil, n)) return;
    simd::float2* out = (simd::float2*)outBuf.contents;
    const char* names[4] = {"equalAreaSphereToSquare(0,0,1) is the square's own dead centre",
                             "equalAreaSphereToSquare(0,0,-1) is the square's own far corner",
                             "equalAreaSphereToSquare(1,0,0) matches the reference program",
                             "equalAreaSphereToSquare(0.3,0.4,0.866) matches the reference program"};
    for (int i = 0; i < n; ++i) {
        expectNear(names[i], out[i].x, expected[i].x, 1e-4);
        char label[128];
        snprintf(label, sizeof(label), "%s (v component)", names[i]);
        expectNear(label, out[i].y, expected[i].y, 1e-4);
    }
}

static void testGoniometricLightRadiance(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // A tiny 4x4 single-channel test image (R8Unorm - the SAME pixel
    // format the real committed scene's own goniometric texture uses,
    // see metal_poc.mm's own buildGoniometricProfileImage() call site) -
    // goniometricLightRadiance() only ever reads the R channel
    // (`image.sample(s, uv).r`), so with a MONOCHROMATIC `emission`
    // every output channel is necessarily identical - there is no
    // "mark a different channel" trick available the way
    // testProjectionLightRadiance()'s own RGB test image used. Two
    // texels instead get two DIFFERENT R intensities.
    const int texSize = 4;
    std::vector<uint8_t> pixels(texSize * texSize, 0);
    // uv=(0.5,0.5) (the light's own forward direction, see below) lands
    // on texel (2,2) with nearest filtering on a 4x4 texture
    // (floor(0.5*4)=2) - full intensity.
    pixels[2 * texSize + 2] = 255;
    // uv=(1.0,1.0) (the light's own exact backward direction) clamps to
    // the last valid texel, (3,3) (floor(1.0*4)=4, clamped to size-1=3)
    // under clamp_to_edge addressing - HALF intensity, a different
    // value from the centre texel's, not a different channel.
    pixels[3 * texSize + 3] = 128;

    MTLTextureDescriptor* texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                        width:texSize height:texSize mipmapped:NO];
    texDesc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> testImage = [device newTextureWithDescriptor:texDesc];
    [testImage replaceRegion:MTLRegionMake2D(0, 0, texSize, texSize) mipmapLevel:0
                    withBytes:pixels.data() bytesPerRow:texSize];

    // A trivial identity-ish light frame (forward=+Z, right=+X, up=+Y) -
    // the same reasoning testProjectionLightRadiance() already used for
    // its own light frame: keeps the expected UV for each test direction
    // computable by hand instead of needing a second "look-at"
    // construction to trust.
    GoniometricLightGPU light{};
    light.position = PackedFloat3{0, 0, 0};
    light.forward = PackedFloat3{0, 0, 1};
    light.right = PackedFloat3{1, 0, 0};
    light.up = PackedFloat3{0, 1, 0};
    light.emission = PackedFloat3{1, 1, 1};
    light.scale = 2.0f;

    simd::float3 wiFromLights[2] = {
        simd::float3{0, 0, 1},  // exactly the light's own forward direction - should sample the full-intensity centre texel
        simd::float3{0, 0, -1}, // exactly backward - should sample the half-intensity back texel
    };
    int n = 2;
    id<MTLBuffer> wiBuf = makeBuffer(device, wiFromLights, sizeof(wiFromLights));
    id<MTLBuffer> lightBuf = makeBuffer(device, &light, sizeof(light));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    if (!runKernel(device, library, queue, @"test_goniometricLightRadiance", @[wiBuf, lightBuf, outBuf], testImage, n)) return;
    simd::float3* out = (simd::float3*)outBuf.contents;

    // `scale=2.0`, `emission=(1,1,1)`: forward direction (full-intensity
    // texel, 255/255=1.0) should read ~2.0; backward (half-intensity,
    // 128/255~=0.502) should read ~1.004 - both checked against the
    // ACTUAL expected number (not just "some positive value"), which
    // also confirms `scale`/`emission` genuinely multiply into the
    // result rather than the raw texture sample passing through
    // unscaled.
    expectNear("goniometricLightRadiance at forward direction reads ~2.0 (full-intensity texel * scale)",
               out[0].x, 2.0, 0.05);
    expectNear("goniometricLightRadiance at backward direction reads ~1.004 (half-intensity texel * scale)",
               out[1].x, 1.004, 0.05);
    expectTrue("goniometricLightRadiance's forward and backward results are genuinely different",
               std::fabs(out[0].x - out[1].x) > 0.5f);
}

int main() {
    @autoreleasepool {
        id<MTLDevice> device = nil;
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices.count > 0) device = devices[0];
        if (!device) {
            fprintf(stderr, "FAIL: no Metal device available\n");
            return 1;
        }
        if (!device.supportsRaytracing) {
            fprintf(stderr, "FAIL: device does not support hardware raytracing\n");
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];

        NSError* error = nil;
#ifdef RT_METAL_SHADER_DIR
        NSString* shaderDir = @(RT_METAL_SHADER_DIR);
#else
        NSString* shaderDir = [@(__FILE__) stringByDeletingLastPathComponent];
#endif
        NSString* shaderPath = [shaderDir stringByAppendingPathComponent:@"metal_poc.metal"];
        NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath encoding:NSUTF8StringEncoding error:&error];
        if (!shaderSource) {
            fprintf(stderr, "FAIL: could not read shader source at %s: %s\n",
                    shaderPath.UTF8String, error.localizedDescription.UTF8String);
            return 1;
        }
        MTLCompileOptions* compileOpts = [MTLCompileOptions new];
        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:compileOpts error:&error];
        if (!library) {
            fprintf(stderr, "FAIL: shader compile failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        testFrDielectric(device, library, queue);
        testGgxD(device, library, queue);
        testGgxG1SmoothLimit(device, library, queue);
        testCheckerColor(device, library, queue);
        testSpotLightFalloff(device, library, queue);
        testFresnelSchlickConductor(device, library, queue);
        testFrComplexRGB(device, library, queue);
        testBuildAnisotropicOnb(device, library, queue);
        testEnvironmentDirectionSampling(device, library, queue);
        testHenyeyGreensteinPhase(device, library, queue);
        testProjectionLightRadiance(device, library, queue);
        testSampleAreaLightAliasTable(device, library, queue);
        testEqualAreaSphereToSquare(device, library, queue);
        testGoniometricLightRadiance(device, library, queue);

        if (g_failures > 0) {
            fprintf(stderr, "FAIL: %d check(s) failed\n", g_failures);
            return 1;
        }
        fprintf(stderr, "PASS: all metal_poc.metal device-side checks passed\n");
        return 0;
    }
}
