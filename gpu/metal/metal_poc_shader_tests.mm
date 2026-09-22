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
#include <random>
#include <simd/simd.h>

#include "metal_poc_host_math.h"
#include "metal_poc_shader_files.h"

// HairBxDF<double> - the actual production CPU/OptiX reference template
// (src/shared/bxdfs_hair.h + bxdfs_principled.h's own hair_* helpers,
// pulled in via the umbrella header) - used here as ground truth for the
// numeric cross-check tests below, NOT reimplemented independently, since
// the question this file's own B11 tests need answered is specifically
// "does the MSL port match the SAME reference CPU/OptiX already render
// with", not "is pbrt's own hair model correct" (already established
// elsewhere). See metal_poc_materials_hair.metal's own header comment for
// why this cross-check exists (a hand-port of a 10-term Bessel I0 series
// is exactly the kind of code a silent 32-bit-integer-overflow mistake
// hides in, invisible in a full-scene render until traced term-by-term).
#include "../../src/shared/bxdfs.h"

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
    uint32_t usePbrtTexture = 0;
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
    uint32_t usePbrtTexture = 0;
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
    float twoSided = 0.0f;
    float useTexture = 0.0f;
    float pmf;
    float aliasProb;
    uint32_t aliasIndex;
    float kind = 0.0f;  // 0=quad, 1=sphere - see AreaLight::kind's own comment
    int32_t spherePrimId = -1;
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

static void testSampleGGXEnergyTableDevice(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // Builds the SAME table (via buildGGXEnergyTable(), the exact
    // function metal_poc.mm's own render path calls at startup) and
    // checks the device-side bilinear lookup reproduces the host-side
    // one (sampleGGXEnergyTable()) at a spread of (roughness, mu) pairs -
    // both read the identical uploaded array, so any drift here means
    // the two interpolation implementations have diverged, not that the
    // table itself is wrong (metal_poc_math_tests.cpp's own
    // testGGXEnergyTableTrend() already checks the table's own values
    // against the real Kulla-Conty physical trend).
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);
    auto randFn = [&]() { return unitDist(rng); };
    GGXEnergyTable table;
    buildGGXEnergyTable(8, 8, 512, table, randFn);

    id<MTLBuffer> tableBuf = makeBuffer(device, table.E.data(), table.E.size() * sizeof(float));
    simd::uint2 dims[1] = {{(uint32_t)table.roughRes, (uint32_t)table.muRes}};
    id<MTLBuffer> dimsBuf = makeBuffer(device, dims, sizeof(dims));

    const int n = 20;
    simd::float2 pairs[n];
    for (int i = 0; i < n; ++i) {
        float roughness = ((float)i + 0.5f) / n;
        float mu = fmodf(roughness * 53.0f + 0.31f, 1.0f);
        pairs[i] = {roughness, mu};
    }
    id<MTLBuffer> pairsBuf = makeBuffer(device, pairs, sizeof(pairs));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_sampleGGXEnergyTableDevice",
                   @[tableBuf, dimsBuf, pairsBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    for (int i = 0; i < n; ++i) {
        float hostValue = sampleGGXEnergyTable(table, pairs[i].x, pairs[i].y);
        char label[128];
        snprintf(label, sizeof(label), "sampleGGXEnergyTableDevice matches host lookup (roughness=%.3f, mu=%.3f)",
                 pairs[i].x, pairs[i].y);
        expectNear(label, out[i], hostValue, 1e-4);
    }
}

static void testOrenNayarF(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // Reference values from a fresh standalone double-precision C port
    // of Blender Cycles' own bsdf_oren_nayar.h (its single-scatter term
    // only - the multiscatter energy-compensation refinement is
    // deliberately not ported, see orenNayarF's own comment), not reused
    // from any earlier session: (1) sigma == 0 must reduce EXACTLY to
    // plain Lambertian's own 1/pi; (2) a grazing, azimuth-aligned view/
    // light pair at high roughness must read substantially BRIGHTER than
    // Lambertian (the real, well-known retroreflective "flat moon"
    // effect this whole material exists to capture) - checked against
    // the reference program's own computed value, not just "greater
    // than," so a sign error or a wrong-by-a-constant-factor bug would
    // still be caught.
    simd::float3 wos[3] = {
        {0.3f, 0.0f, 0.9539f}, {0.9539f, 0.0f, 0.3f}, {0.9539f, 0.0f, 0.3f},
    };
    simd::float3 wis[3] = {
        {-0.2f, 0.1f, 0.9747f}, {0.9747f, 0.0f, 0.2f}, {0.9747f, 0.0f, 0.2f},
    };
    simd::float3 ns[3] = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
    float sigmas[3] = {0.0f, 0.0f, 1.0f};
    float expected[3] = {1.0f / (float)M_PI, 1.0f / (float)M_PI, 1.0132f};
    int n = 3;
    id<MTLBuffer> woBuf = makeBuffer(device, wos, sizeof(wos));
    id<MTLBuffer> wiBuf = makeBuffer(device, wis, sizeof(wis));
    id<MTLBuffer> nBuf = makeBuffer(device, ns, sizeof(ns));
    id<MTLBuffer> sigmaBuf = makeBuffer(device, sigmas, sizeof(sigmas));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_orenNayarF",
                   @[woBuf, wiBuf, nBuf, sigmaBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    const char* names[3] = {
        "orenNayarF at sigma=0 matches Lambertian's own 1/pi (case 0, near-normal)",
        "orenNayarF at sigma=0 matches Lambertian's own 1/pi (case 1, grazing)",
        "orenNayarF at sigma=1 shows the real grazing-angle retroreflective brightening",
    };
    for (int i = 0; i < n; ++i) {
        expectNear(names[i], out[i], expected[i], 1e-3);
    }
}

static void testVelvetF(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // Reference values from a fresh standalone double-precision C port
    // of Blender Cycles' own bsdf_ashikhmin_velvet.h, not reused from
    // any earlier session: (1) perfectly head-on view AND light must be
    // EXACTLY zero - velvet's own signature "no highlight straight back
    // at you" behaviour, unlike every other glossy material in this
    // POC; (2) a grazing, azimuth-aligned view/light pair must show the
    // real, substantial "fuzzy rim" brightening this material exists to
    // capture, matched to the reference program's own exact value, not
    // just "greater than zero."
    simd::float3 wos[3] = {
        {0.0f, 0.0f, 1.0f}, {0.97072832f, 0.0f, 0.24018020f}, {0.96592583f, 0.0f, 0.25881905f},
    };
    simd::float3 wis[3] = {
        {0.0f, 0.0f, 1.0f}, {0.97072832f, 0.0f, 0.24018020f}, {0.96592583f, 0.0f, 0.25881905f},
    };
    simd::float3 ns[3] = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
    float sigmas[3] = {0.3f, 0.3f, 0.3f};
    float expected[3] = {0.0f, 0.24227969f, 0.23677936f};
    int n = 3;
    id<MTLBuffer> woBuf = makeBuffer(device, wos, sizeof(wos));
    id<MTLBuffer> wiBuf = makeBuffer(device, wis, sizeof(wis));
    id<MTLBuffer> nBuf = makeBuffer(device, ns, sizeof(ns));
    id<MTLBuffer> sigmaBuf = makeBuffer(device, sigmas, sizeof(sigmas));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_velvetF",
                   @[woBuf, wiBuf, nBuf, sigmaBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    const char* names[3] = {
        "velvetF is exactly zero at perfectly head-on view/light",
        "velvetF matches the reference at a grazing, aligned pair (case 1)",
        "velvetF matches the reference at a grazing, aligned pair (case 2, 75deg)",
    };
    for (int i = 0; i < n; ++i) {
        expectNear(names[i], out[i], expected[i], 1e-3);
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
    // sampleAreaLight() now takes a texture (section 105's own
    // AreaLight::useTexture) - every light above has useTexture=0.0f
    // (AreaLightGPU's own default), so this dummy 1x1 texture is never
    // actually sampled, just needs to be SOME real bound resource.
    MTLTextureDescriptor* dummyTexDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                              width:1 height:1 mipmapped:NO];
    dummyTexDesc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> dummyTexture = [device newTextureWithDescriptor:dummyTexDesc];
    uint8_t dummyPixel[4] = {0, 0, 0, 255};
    [dummyTexture replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:dummyPixel bytesPerRow:4];
    if (!runKernel(device, library, queue, @"test_sampleAreaLight_pmf",
                   @[lightsBuf, countBuf, seedBuf, outBuf], dummyTexture, kNumSamples)) return;
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

// Mirrors metal_poc_materials_hair.metal's own HairBxDFParams byte-for-
// byte (18 floats, same field order) - built from a real HairBxDF<double>
// instance's own public v[]/s/sin2kAlpha[]/cos2kAlpha[] fields (its
// constructor's own math, not re-derived here) so this test isolates
// eval_local()/scattering_pdf_local()/hair_Mp()/hair_Np() specifically,
// the functions metal_poc_materials_hair.metal actually had to hand-port.
struct HairBxDFParamsGPU {
    float h, eta;
    float sigma_r, sigma_g, sigma_b;
    float beta_m, beta_n;
    float v0, v1, v2, v3;
    float s;
    float sin2kAlpha0, sin2kAlpha1, sin2kAlpha2;
    float cos2kAlpha0, cos2kAlpha1, cos2kAlpha2;
};

static HairBxDFParamsGPU makeHairParamsGPU(const HairBxDF<double>& ref) {
    HairBxDFParamsGPU p{};
    p.h = (float)ref.h; p.eta = (float)ref.eta;
    p.sigma_r = (float)ref.sigma_r; p.sigma_g = (float)ref.sigma_g; p.sigma_b = (float)ref.sigma_b;
    p.beta_m = (float)ref.beta_m; p.beta_n = (float)ref.beta_n;
    p.v0 = (float)ref.v[0]; p.v1 = (float)ref.v[1]; p.v2 = (float)ref.v[2]; p.v3 = (float)ref.v[3];
    p.s = (float)ref.s;
    p.sin2kAlpha0 = (float)ref.sin2kAlpha[0]; p.sin2kAlpha1 = (float)ref.sin2kAlpha[1]; p.sin2kAlpha2 = (float)ref.sin2kAlpha[2];
    p.cos2kAlpha0 = (float)ref.cos2kAlpha[0]; p.cos2kAlpha1 = (float)ref.cos2kAlpha[1]; p.cos2kAlpha2 = (float)ref.cos2kAlpha[2];
    return p;
}

// hair_Mp() is the one function most exposed to the 32-bit-overflow risk
// metal_poc_materials_hair.metal's own header comment describes: its
// v>0.1 branch calls hair_I0(a) directly (no log-space/asymptotic guard
// at all), and a = cosTheta_i*cosTheta_o/v approaches 10 for realistic
// near-grazing-free directions at v just above 0.1 - exactly the range
// where the Bessel series' own i4*ifact^2 denominator term is large
// enough (billions, for i>=7) to overflow a naive 32-bit int port. Cases
// below deliberately sweep a up toward that regime in BOTH the v<=0.1
// (LogI0) and v>0.1 (direct I0) branches.
static void testHairMp(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    struct { double cosI, cosO, sinI, sinO, v; } cases[] = {
        {1.0, 1.0, 0.0, 0.0, 0.5},        // head-on, moderate v: sanity baseline
        {0.999, 0.999, 0.0447, 0.0447, 0.15},   // v>0.1 branch, a ~ 6.65 (mid series)
        {0.999, 0.999, 0.0447, 0.0447, 0.11},   // v>0.1 branch, a ~ 9.07 (late series terms matter)
        {0.9995, 0.9995, 0.0316, 0.0316, 0.09}, // v<=0.1 branch (LogI0), a ~ 11.1 (just under the x>12 asymptotic cutoff - full series)
        {0.7, 0.7, 0.7141, 0.7141, 0.05},       // v<=0.1 branch, smaller a (~9.8) with non-trivial sinTheta terms
        {0.999, -0.999, 0.0447, -0.0447, 0.11}, // opposite-sign sinTheta (b term flips sign)
    };
    int n = 6;
    std::vector<float> cosIs(n), cosOs(n), sinIs(n), sinOs(n), vs(n);
    std::vector<double> expected(n);
    for (int i = 0; i < n; ++i) {
        cosIs[i] = (float)cases[i].cosI; cosOs[i] = (float)cases[i].cosO;
        sinIs[i] = (float)cases[i].sinI; sinOs[i] = (float)cases[i].sinO; vs[i] = (float)cases[i].v;
        expected[i] = hair_Mp<double>(cases[i].cosI, cases[i].cosO, cases[i].sinI, cases[i].sinO, cases[i].v);
    }
    id<MTLBuffer> cosIBuf = makeBuffer(device, cosIs.data(), n * sizeof(float));
    id<MTLBuffer> cosOBuf = makeBuffer(device, cosOs.data(), n * sizeof(float));
    id<MTLBuffer> sinIBuf = makeBuffer(device, sinIs.data(), n * sizeof(float));
    id<MTLBuffer> sinOBuf = makeBuffer(device, sinOs.data(), n * sizeof(float));
    id<MTLBuffer> vBuf = makeBuffer(device, vs.data(), n * sizeof(float));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_hairMp",
                   @[cosIBuf, cosOBuf, sinIBuf, sinOBuf, vBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    for (int i = 0; i < n; ++i) {
        char label[160];
        snprintf(label, sizeof(label), "hairMp matches HairBxDF<double> reference (case %d, v=%.3f)", i, cases[i].v);
        expectNear(label, out[i], expected[i], std::max(1e-3, expected[i] * 0.02));
    }
}

static void testHairNp(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    struct { double phi; int p; double s, gammaO, gammaT; } cases[] = {
        {0.0, 0, 0.3, 0.1, 0.05},
        {0.5, 1, 0.3, 0.1, 0.05},
        {-1.2, 2, 0.15, -0.2, 0.15},
        {3.0, 0, 0.4, 0.0, 0.0},   // dphi near +-pi wraparound
    };
    int n = 4;
    std::vector<float> phis(n), ss(n), gOs(n), gTs(n);
    std::vector<int> ps(n);
    std::vector<double> expected(n);
    for (int i = 0; i < n; ++i) {
        phis[i] = (float)cases[i].phi; ps[i] = cases[i].p; ss[i] = (float)cases[i].s;
        gOs[i] = (float)cases[i].gammaO; gTs[i] = (float)cases[i].gammaT;
        expected[i] = hair_Np<double>(cases[i].phi, cases[i].p, cases[i].s, cases[i].gammaO, cases[i].gammaT);
    }
    id<MTLBuffer> phiBuf = makeBuffer(device, phis.data(), n * sizeof(float));
    id<MTLBuffer> pBuf = makeBuffer(device, ps.data(), n * sizeof(int));
    id<MTLBuffer> sBuf = makeBuffer(device, ss.data(), n * sizeof(float));
    id<MTLBuffer> gOBuf = makeBuffer(device, gOs.data(), n * sizeof(float));
    id<MTLBuffer> gTBuf = makeBuffer(device, gTs.data(), n * sizeof(float));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_hairNp",
                   @[phiBuf, pBuf, sBuf, gOBuf, gTBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    for (int i = 0; i < n; ++i) {
        char label[128];
        snprintf(label, sizeof(label), "hairNp matches HairBxDF<double> reference (case %d)", i);
        expectNear(label, out[i], expected[i], std::max(1e-4, expected[i] * 0.02));
    }
}

// The real end-to-end test: eval_local()'s fr/fg/fb (the numerator of the
// ratio the earlier B11 attempt found going unexpectedly negative) and
// scattering_pdf_local()'s pdf (the denominator - a SEPARATE code path
// with its own hair_Mp() calls the earlier attempt's bisection may not
// have fully isolated, since overriding Mp only inside eval_local
// wouldn't touch scattering_pdf_local's own independent Mp calls).
static void testHairEvalAndPdf(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    struct { double h, eta, sr, sg, sb, beta_m, beta_n, alpha; } materials[] = {
        {0.3, 1.55, 0.06, 0.1, 0.2, 0.25, 0.3, 2.0},   // typical brown hair
        {-0.5, 1.55, 0.4, 0.7, 1.3, 0.15, 0.2, 2.0},   // darker hair, lower roughness (stresses hair_Mp's a~10 regime)
        {0.0, 1.55, 0.02, 0.03, 0.05, 0.4, 0.4, 2.0},  // light hair, higher roughness
    };
    struct { double sinO, phiO, sinI, phiI; } dirs[] = {
        {0.1, 0.3, 0.15, 2.0},
        {-0.2, -1.0, 0.05, 0.5},
        {0.4, 2.5, -0.1, -2.5},
        {0.0, 0.0, 0.02, 3.0},
    };
    int nMat = 3, nDir = 4;
    for (int m = 0; m < nMat; ++m) {
        HairBxDF<double> ref(materials[m].h, materials[m].eta, materials[m].sr, materials[m].sg, materials[m].sb,
                              materials[m].beta_m, materials[m].beta_n, materials[m].alpha);
        HairBxDFParamsGPU gpuParams = makeHairParamsGPU(ref);
        id<MTLBuffer> paramsBuf = makeBuffer(device, &gpuParams, sizeof(HairBxDFParamsGPU));
        // test_hairEvalLocal/test_hairScatteringPdfLocal index params[tid]
        // (one struct per thread) - replicate the single material across
        // all nDir threads so each direction case shares the same params.
        std::vector<HairBxDFParamsGPU> paramsRep(nDir, gpuParams);
        id<MTLBuffer> paramsRepBuf = makeBuffer(device, paramsRep.data(), nDir * sizeof(HairBxDFParamsGPU));

        std::vector<simd::float3> wos(nDir), wis(nDir);
        std::vector<double> expectedFr(nDir), expectedFg(nDir), expectedFb(nDir), expectedPdf(nDir);
        for (int d = 0; d < nDir; ++d) {
            double cosO = std::sqrt(std::max(0.0, 1.0 - dirs[d].sinO * dirs[d].sinO));
            double cosI = std::sqrt(std::max(0.0, 1.0 - dirs[d].sinI * dirs[d].sinI));
            double wo_x = dirs[d].sinO, wo_y = cosO * std::cos(dirs[d].phiO), wo_z = cosO * std::sin(dirs[d].phiO);
            double wi_x = dirs[d].sinI, wi_y = cosI * std::cos(dirs[d].phiI), wi_z = cosI * std::sin(dirs[d].phiI);
            wos[d] = simd::float3{(float)wo_x, (float)wo_y, (float)wo_z};
            wis[d] = simd::float3{(float)wi_x, (float)wi_y, (float)wi_z};
            double fr, fg, fb;
            ref.eval_local(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z, fr, fg, fb);
            expectedFr[d] = fr; expectedFg[d] = fg; expectedFb[d] = fb;
            expectedPdf[d] = ref.scattering_pdf_local(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
        }
        id<MTLBuffer> woBuf = makeBuffer(device, wos.data(), nDir * sizeof(simd::float3));
        id<MTLBuffer> wiBuf = makeBuffer(device, wis.data(), nDir * sizeof(simd::float3));
        id<MTLBuffer> evalOutBuf = makeOutputBuffer(device, nDir * sizeof(simd::float3));
        id<MTLBuffer> pdfOutBuf = makeOutputBuffer(device, nDir * sizeof(float));
        if (!runKernel(device, library, queue, @"test_hairEvalLocal",
                       @[paramsRepBuf, woBuf, wiBuf, evalOutBuf], nil, nDir)) continue;
        if (!runKernel(device, library, queue, @"test_hairScatteringPdfLocal",
                       @[paramsRepBuf, woBuf, wiBuf, pdfOutBuf], nil, nDir)) continue;
        simd::float3* evalOut = (simd::float3*)evalOutBuf.contents;
        float* pdfOut = (float*)pdfOutBuf.contents;
        for (int d = 0; d < nDir; ++d) {
            char label[192];
            snprintf(label, sizeof(label), "hairEvalLocal.fr matches reference (material %d, dir %d)", m, d);
            expectNear(label, evalOut[d].x, expectedFr[d], std::max(1e-3, std::fabs(expectedFr[d]) * 0.02));
            snprintf(label, sizeof(label), "hairEvalLocal.fg matches reference (material %d, dir %d)", m, d);
            expectNear(label, evalOut[d].y, expectedFg[d], std::max(1e-3, std::fabs(expectedFg[d]) * 0.02));
            snprintf(label, sizeof(label), "hairEvalLocal.fb matches reference (material %d, dir %d)", m, d);
            expectNear(label, evalOut[d].z, expectedFb[d], std::max(1e-3, std::fabs(expectedFb[d]) * 0.02));
            snprintf(label, sizeof(label), "hairScatteringPdfLocal matches reference (material %d, dir %d)", m, d);
            expectNear(label, pdfOut[d], expectedPdf[d], std::max(1e-3, std::fabs(expectedPdf[d]) * 0.02));
            // The actual quantity that matters for rendering: fr/pdf must
            // be non-negative (a real BSDF ratio can never be negative) -
            // the earlier B11 attempt's central, unexplained finding was
            // that this went negative across large surface regions.
            snprintf(label, sizeof(label), "hairEvalLocal/hairScatteringPdfLocal ratio is non-negative (material %d, dir %d)", m, d);
            if (pdfOut[d] > 1e-8) {
                expectTrue(label, evalOut[d].x / pdfOut[d] > -1e-4);
            }
        }
    }
}

// Broad randomized sweep (not just the hand-picked stress cases above) -
// the earlier B11 attempt's own bug was subtle enough that a handful of
// chosen cases could plausibly miss it, so this dispatches many random
// (material, direction) pairs and checks both numeric agreement with the
// HairBxDF<double> reference AND the non-negativity property directly
// (fr/fg/fb and pdf must never go negative for a physically valid BSDF -
// this is the exact property that broke in the earlier attempt).
static void testHairRandomSweep(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::uniform_real_distribution<double> zeroOne(0.0, 1.0);
    std::uniform_real_distribution<double> angle(-M_PI, M_PI);

    const int nMat = 6;
    const int nDirPerMat = 12;
    for (int m = 0; m < nMat; ++m) {
        double h = unit(rng);
        double eta = 1.3 + zeroOne(rng) * 0.6;
        double sr = zeroOne(rng) * 1.5, sg = zeroOne(rng) * 1.5, sb = zeroOne(rng) * 1.5;
        double beta_m = 0.05 + zeroOne(rng) * 0.6;   // sweeps both v<=0.1 and v>0.1 branches
        double beta_n = 0.05 + zeroOne(rng) * 0.6;
        double alpha = zeroOne(rng) * 4.0;
        HairBxDF<double> ref(h, eta, sr, sg, sb, beta_m, beta_n, alpha);
        HairBxDFParamsGPU gpuParams = makeHairParamsGPU(ref);
        std::vector<HairBxDFParamsGPU> paramsRep(nDirPerMat, gpuParams);
        id<MTLBuffer> paramsBuf = makeBuffer(device, paramsRep.data(), nDirPerMat * sizeof(HairBxDFParamsGPU));

        std::vector<simd::float3> wos(nDirPerMat), wis(nDirPerMat);
        std::vector<double> expectedFr(nDirPerMat), expectedPdf(nDirPerMat);
        for (int d = 0; d < nDirPerMat; ++d) {
            double sinO = unit(rng), sinI = unit(rng);
            double phiO = angle(rng), phiI = angle(rng);
            double cosO = std::sqrt(std::max(0.0, 1.0 - sinO * sinO));
            double cosI = std::sqrt(std::max(0.0, 1.0 - sinI * sinI));
            double wo_x = sinO, wo_y = cosO * std::cos(phiO), wo_z = cosO * std::sin(phiO);
            double wi_x = sinI, wi_y = cosI * std::cos(phiI), wi_z = cosI * std::sin(phiI);
            wos[d] = simd::float3{(float)wo_x, (float)wo_y, (float)wo_z};
            wis[d] = simd::float3{(float)wi_x, (float)wi_y, (float)wi_z};
            double fr, fg, fb;
            ref.eval_local(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z, fr, fg, fb);
            expectedFr[d] = fr;
            expectedPdf[d] = ref.scattering_pdf_local(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
        }
        id<MTLBuffer> woBuf = makeBuffer(device, wos.data(), nDirPerMat * sizeof(simd::float3));
        id<MTLBuffer> wiBuf = makeBuffer(device, wis.data(), nDirPerMat * sizeof(simd::float3));
        id<MTLBuffer> evalOutBuf = makeOutputBuffer(device, nDirPerMat * sizeof(simd::float3));
        id<MTLBuffer> pdfOutBuf = makeOutputBuffer(device, nDirPerMat * sizeof(float));
        if (!runKernel(device, library, queue, @"test_hairEvalLocal",
                       @[paramsBuf, woBuf, wiBuf, evalOutBuf], nil, nDirPerMat)) continue;
        if (!runKernel(device, library, queue, @"test_hairScatteringPdfLocal",
                       @[paramsBuf, woBuf, wiBuf, pdfOutBuf], nil, nDirPerMat)) continue;
        simd::float3* evalOut = (simd::float3*)evalOutBuf.contents;
        float* pdfOut = (float*)pdfOutBuf.contents;
        for (int d = 0; d < nDirPerMat; ++d) {
            char label[192];
            snprintf(label, sizeof(label), "random sweep: hairEvalLocal.fr matches reference (mat %d beta_m=%.3f, dir %d)",
                     m, beta_m, d);
            expectNear(label, evalOut[d].x, expectedFr[d], std::max(1e-2, std::fabs(expectedFr[d]) * 0.05));
            snprintf(label, sizeof(label), "random sweep: pdf matches reference (mat %d beta_m=%.3f, dir %d)", m, beta_m, d);
            expectNear(label, pdfOut[d], expectedPdf[d], std::max(1e-2, std::fabs(expectedPdf[d]) * 0.05));
            snprintf(label, sizeof(label), "random sweep: fr is non-negative (mat %d beta_m=%.3f, dir %d)", m, beta_m, d);
            expectTrue(label, evalOut[d].x > -1e-4);
            snprintf(label, sizeof(label), "random sweep: pdf is non-negative (mat %d beta_m=%.3f, dir %d)", m, beta_m, d);
            expectTrue(label, pdfOut[d] > -1e-4);
        }
    }
}

// hairSample() end-to-end: the actual function shadeHair() will call at
// render time. Fixed (non-random) u1..u4 per case for exact reproducibility
// against ref.sample()'s own identical inverse-CDF math - any GPU/CPU
// divergence here would show up as a wrong wo direction or wrong r/g/b,
// not just a wrong scalar.
static void testHairSample(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    struct { double h, eta, sr, sg, sb, beta_m, beta_n, alpha; } materials[] = {
        {0.3, 1.55, 0.06, 0.1, 0.2, 0.25, 0.3, 2.0},
        {-0.5, 1.55, 0.4, 0.7, 1.3, 0.15, 0.2, 2.0},
        {0.0, 1.55, 0.02, 0.03, 0.05, 0.4, 0.4, 2.0},
    };
    struct { double tx, ty, tz, wix, wiy, wiz, u1, u2, u3, u4; } cases[] = {
        {1, 0, 0,  0.3, -0.8, 0.5,  0.1, 0.5, 0.3, 0.7},
        {0, 1, 0,  0.6, 0.2, -0.7,  0.6, 0.2, 0.8, 0.1},
        {0.577, 0.577, 0.577,  -0.4, -0.4, 0.8,  0.9, 0.9, 0.05, 0.4},
        {1, 0, 0,  -0.9, 0.1, 0.2,  0.99, 0.01, 0.6, 0.99},
    };
    int nMat = 3, nCase = 4;
    for (int m = 0; m < nMat; ++m) {
        HairBxDF<double> ref(materials[m].h, materials[m].eta, materials[m].sr, materials[m].sg, materials[m].sb,
                              materials[m].beta_m, materials[m].beta_n, materials[m].alpha);
        HairBxDFParamsGPU gpuParams = makeHairParamsGPU(ref);
        std::vector<HairBxDFParamsGPU> paramsRep(nCase, gpuParams);
        id<MTLBuffer> paramsBuf = makeBuffer(device, paramsRep.data(), nCase * sizeof(HairBxDFParamsGPU));

        std::vector<simd::float3> tangents(nCase), wis(nCase);
        std::vector<simd::float4> us(nCase);
        std::vector<BxDFSampleResult<double>> expected(nCase);
        for (int c = 0; c < nCase; ++c) {
            double tlen = std::sqrt(cases[c].tx * cases[c].tx + cases[c].ty * cases[c].ty + cases[c].tz * cases[c].tz);
            double tx = cases[c].tx / tlen, ty = cases[c].ty / tlen, tz = cases[c].tz / tlen;
            double wlen = std::sqrt(cases[c].wix * cases[c].wix + cases[c].wiy * cases[c].wiy + cases[c].wiz * cases[c].wiz);
            double wix = cases[c].wix / wlen, wiy = cases[c].wiy / wlen, wiz = cases[c].wiz / wlen;
            tangents[c] = simd::float3{(float)tx, (float)ty, (float)tz};
            wis[c] = simd::float3{(float)wix, (float)wiy, (float)wiz};
            us[c] = simd::float4{(float)cases[c].u1, (float)cases[c].u2, (float)cases[c].u3, (float)cases[c].u4};
            expected[c] = ref.sample(tx, ty, tz, wix, wiy, wiz, cases[c].u1, cases[c].u2, cases[c].u3, cases[c].u4);
        }
        id<MTLBuffer> tanBuf = makeBuffer(device, tangents.data(), nCase * sizeof(simd::float3));
        id<MTLBuffer> wiBuf = makeBuffer(device, wis.data(), nCase * sizeof(simd::float3));
        id<MTLBuffer> uBuf = makeBuffer(device, us.data(), nCase * sizeof(simd::float4));
        id<MTLBuffer> woOutBuf = makeOutputBuffer(device, nCase * sizeof(simd::float3));
        id<MTLBuffer> rgbOutBuf = makeOutputBuffer(device, nCase * sizeof(simd::float3));
        id<MTLBuffer> validOutBuf = makeOutputBuffer(device, nCase * sizeof(int32_t));
        if (!runKernel(device, library, queue, @"test_hairSample",
                       @[paramsBuf, tanBuf, wiBuf, uBuf, woOutBuf, rgbOutBuf, validOutBuf], nil, nCase)) continue;
        simd::float3* woOut = (simd::float3*)woOutBuf.contents;
        simd::float3* rgbOut = (simd::float3*)rgbOutBuf.contents;
        int32_t* validOut = (int32_t*)validOutBuf.contents;
        for (int c = 0; c < nCase; ++c) {
            char label[192];
            snprintf(label, sizeof(label), "hairSample valid flag matches reference (material %d, case %d)", m, c);
            expectTrue(label, (bool)validOut[c] == expected[c].valid);
            if (!expected[c].valid) continue;
            snprintf(label, sizeof(label), "hairSample wo.x matches reference (material %d, case %d)", m, c);
            expectNear(label, woOut[c].x, expected[c].wo_x, 1e-2);
            snprintf(label, sizeof(label), "hairSample wo.y matches reference (material %d, case %d)", m, c);
            expectNear(label, woOut[c].y, expected[c].wo_y, 1e-2);
            snprintf(label, sizeof(label), "hairSample wo.z matches reference (material %d, case %d)", m, c);
            expectNear(label, woOut[c].z, expected[c].wo_z, 1e-2);
            snprintf(label, sizeof(label), "hairSample r matches reference (material %d, case %d)", m, c);
            expectNear(label, rgbOut[c].x, expected[c].r, std::max(1e-2, std::fabs(expected[c].r) * 0.05));
            snprintf(label, sizeof(label), "hairSample g matches reference (material %d, case %d)", m, c);
            expectNear(label, rgbOut[c].y, expected[c].g, std::max(1e-2, std::fabs(expected[c].g) * 0.05));
            snprintf(label, sizeof(label), "hairSample b matches reference (material %d, case %d)", m, c);
            expectNear(label, rgbOut[c].z, expected[c].b, std::max(1e-2, std::fabs(expected[c].b) * 0.05));
            snprintf(label, sizeof(label), "hairSample r/g/b non-negative (material %d, case %d)", m, c);
            expectTrue(label, rgbOut[c].x > -1e-4 && rgbOut[c].y > -1e-4 && rgbOut[c].z > -1e-4);
        }
    }
}

// Diagnostic: B11's own "fine black fur" sphere (sigma_a=(0.50,0.55,0.60),
// beta_m=beta_n=0.15, alpha=2, eta=1.55 - scene_advanced.h's own
// build_hair_fibers() 5th sphere) renders fully black on GPU but as the
// BRIGHTEST sphere on CPU at the same sample count - not just noise. One
// plausible mechanism: if hairSample() returns invalid (pdf<1e-8) far more
// often on GPU than the reference for this specific high-absorption
// material, shadeHair() returns false and the bounce loop breaks
// immediately, contributing zero further radiance - which could turn an
// entire sphere black even though the individual eval/pdf/sample cross-
// checks above (lower-sigma materials) all passed. Sweeps many random
// (h, u1..u4, wi direction) draws for THIS exact material and compares
// the fraction of valid samples (and mean r/g/b when valid) against the
// HairBxDF<double> reference, to see whether GPU's invalid rate is
// systematically higher.
static void testHairBlackFurValidRate(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    const double sr = 0.50, sg = 0.55, sb = 0.60, beta_m = 0.15, beta_n = 0.15, alpha_deg = 2.0, eta = 1.55;
    std::mt19937 rng(777);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::uniform_real_distribution<double> zeroOne(0.0, 1.0);

    const int n = 200;
    std::vector<HairBxDFParamsGPU> paramsRep(n);
    std::vector<simd::float3> tangents(n), wis(n);
    std::vector<simd::float4> us(n);
    int refValidCount = 0;
    double refSumR = 0.0, refSumG = 0.0, refSumB = 0.0;
    for (int i = 0; i < n; ++i) {
        double h = unit(rng);
        HairBxDF<double> ref(h, eta, sr, sg, sb, beta_m, beta_n, alpha_deg);
        paramsRep[i] = makeHairParamsGPU(ref);

        // Random tangent direction (uniform-ish over the sphere) - a real
        // sphere's facing normal varies across its whole surface, unlike
        // this test's earlier fixed (0,1,0) version.
        double tx = unit(rng), ty = unit(rng), tz = unit(rng);
        double tlen = std::sqrt(tx * tx + ty * ty + tz * tz);
        if (tlen < 1e-6) tlen = 1.0;
        tx /= tlen; ty /= tlen; tz /= tlen;
        double wix = unit(rng), wiy = unit(rng), wiz = unit(rng);
        double wlen = std::sqrt(wix * wix + wiy * wiy + wiz * wiz);
        if (wlen < 1e-6) wlen = 1.0;
        wix /= wlen; wiy /= wlen; wiz /= wlen;
        double u1 = zeroOne(rng), u2 = zeroOne(rng), u3 = zeroOne(rng), u4 = zeroOne(rng);
        tangents[i] = simd::float3{(float)tx, (float)ty, (float)tz};
        wis[i] = simd::float3{(float)wix, (float)wiy, (float)wiz};
        us[i] = simd::float4{(float)u1, (float)u2, (float)u3, (float)u4};

        BxDFSampleResult<double> res = ref.sample(tx, ty, tz, wix, wiy, wiz, u1, u2, u3, u4);
        if (res.valid) {
            ++refValidCount;
            refSumR += res.r; refSumG += res.g; refSumB += res.b;
        }
    }

    id<MTLBuffer> paramsBuf = makeBuffer(device, paramsRep.data(), n * sizeof(HairBxDFParamsGPU));
    id<MTLBuffer> tanBuf = makeBuffer(device, tangents.data(), n * sizeof(simd::float3));
    id<MTLBuffer> wiBuf = makeBuffer(device, wis.data(), n * sizeof(simd::float3));
    id<MTLBuffer> uBuf = makeBuffer(device, us.data(), n * sizeof(simd::float4));
    id<MTLBuffer> woOutBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    id<MTLBuffer> rgbOutBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    id<MTLBuffer> validOutBuf = makeOutputBuffer(device, n * sizeof(int32_t));
    if (!runKernel(device, library, queue, @"test_hairSample",
                   @[paramsBuf, tanBuf, wiBuf, uBuf, woOutBuf, rgbOutBuf, validOutBuf], nil, n)) return;
    simd::float3* rgbOut = (simd::float3*)rgbOutBuf.contents;
    int32_t* validOut = (int32_t*)validOutBuf.contents;
    int gpuValidCount = 0;
    double gpuSumR = 0.0, gpuSumG = 0.0, gpuSumB = 0.0;
    for (int i = 0; i < n; ++i) {
        if (validOut[i]) {
            ++gpuValidCount;
            gpuSumR += rgbOut[i].x; gpuSumG += rgbOut[i].y; gpuSumB += rgbOut[i].z;
        }
    }
    fprintf(stderr, "[B11 black-fur diagnostic] reference valid: %d/%d (mean rgb when valid: %.4f %.4f %.4f)\n",
            refValidCount, n, refValidCount ? refSumR / refValidCount : 0.0,
            refValidCount ? refSumG / refValidCount : 0.0, refValidCount ? refSumB / refValidCount : 0.0);
    fprintf(stderr, "[B11 black-fur diagnostic] GPU valid:       %d/%d (mean rgb when valid: %.4f %.4f %.4f)\n",
            gpuValidCount, n, gpuValidCount ? gpuSumR / gpuValidCount : 0.0,
            gpuValidCount ? gpuSumG / gpuValidCount : 0.0, gpuValidCount ? gpuSumB / gpuValidCount : 0.0);
    char label[128];
    snprintf(label, sizeof(label), "B11 black-fur material: GPU valid rate matches reference (ref=%d, gpu=%d, n=%d)",
             refValidCount, gpuValidCount, n);
    expectTrue(label, std::abs(gpuValidCount - refValidCount) <= n / 10);
}

// Direct test of the DEGENERATE case a sphere's own surface normal used
// as HairBxDF's fiber tangent hits at the CENTER of its visible disc: the
// camera ray arrives ANTI-PARALLEL to the tangent (wi = -tangent exactly),
// meaning sinTheta_o = wo_x = 1, cosTheta_o = 0 - the extreme grazing-
// along-fiber case. The black-fur render's own mean brightness was FLAT
// between 256 and 2048 samples (27.7 vs 28.1, not trending toward CPU's
// 188.7) - a stable WRONG expected value, not slow convergence - so this
// averages hairSample()'s own r/g/b over MANY random u1..u4 draws at
// EXACTLY this geometry and compares the MEAN against the reference's
// own mean at the same fixed geometry, to see whether the two estimators'
// EXPECTED VALUES themselves disagree specifically in this regime (not
// just individual samples, which the earlier random-tangent sweep already
// found matching bit-for-bit on average, but that sweep rarely landed
// exactly at this measure-zero cosTheta_o=0 case).
static void testHairGrazingCenterBias(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    const double sr = 0.50, sg = 0.55, sb = 0.60, beta_m = 0.15, beta_n = 0.15, alpha_deg = 2.0, eta = 1.55;
    std::mt19937 rng(4242);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::uniform_real_distribution<double> zeroOne(0.0, 1.0);

    const int n = 500;
    std::vector<HairBxDFParamsGPU> paramsRep(n);
    std::vector<simd::float3> tangents(n), wis(n);
    std::vector<simd::float4> us(n);
    double refSumR = 0.0;
    int refValid = 0;
    for (int i = 0; i < n; ++i) {
        double h = unit(rng);
        HairBxDF<double> ref(h, eta, sr, sg, sb, beta_m, beta_n, alpha_deg);
        paramsRep[i] = makeHairParamsGPU(ref);
        // tangent = facing normal (world up, arbitrary but fixed), wi
        // EXACTLY anti-parallel to it - the exact center-of-disc geometry.
        double tx = 0.0, ty = 1.0, tz = 0.0;
        double wix = 0.0, wiy = -1.0, wiz = 0.0;
        tangents[i] = simd::float3{(float)tx, (float)ty, (float)tz};
        wis[i] = simd::float3{(float)wix, (float)wiy, (float)wiz};
        double u1 = zeroOne(rng), u2 = zeroOne(rng), u3 = zeroOne(rng), u4 = zeroOne(rng);
        us[i] = simd::float4{(float)u1, (float)u2, (float)u3, (float)u4};
        BxDFSampleResult<double> res = ref.sample(tx, ty, tz, wix, wiy, wiz, u1, u2, u3, u4);
        if (res.valid) { refSumR += res.r; ++refValid; }
    }

    id<MTLBuffer> paramsBuf = makeBuffer(device, paramsRep.data(), n * sizeof(HairBxDFParamsGPU));
    id<MTLBuffer> tanBuf = makeBuffer(device, tangents.data(), n * sizeof(simd::float3));
    id<MTLBuffer> wiBuf = makeBuffer(device, wis.data(), n * sizeof(simd::float3));
    id<MTLBuffer> uBuf = makeBuffer(device, us.data(), n * sizeof(simd::float4));
    id<MTLBuffer> woOutBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    id<MTLBuffer> rgbOutBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    id<MTLBuffer> validOutBuf = makeOutputBuffer(device, n * sizeof(int32_t));
    if (!runKernel(device, library, queue, @"test_hairSample",
                   @[paramsBuf, tanBuf, wiBuf, uBuf, woOutBuf, rgbOutBuf, validOutBuf], nil, n)) return;
    simd::float3* rgbOut = (simd::float3*)rgbOutBuf.contents;
    int32_t* validOut = (int32_t*)validOutBuf.contents;
    double gpuSumR = 0.0;
    int gpuValid = 0;
    for (int i = 0; i < n; ++i) {
        if (validOut[i]) { gpuSumR += rgbOut[i].x; ++gpuValid; }
    }
    fprintf(stderr, "[grazing-center diagnostic] reference: valid=%d/%d mean_r=%.4f (sum over ALL n, valid-or-not: %.4f)\n",
            refValid, n, refValid ? refSumR / refValid : 0.0, refSumR / n);
    fprintf(stderr, "[grazing-center diagnostic] GPU:       valid=%d/%d mean_r=%.4f (sum over ALL n, valid-or-not: %.4f)\n",
            gpuValid, n, gpuValid ? gpuSumR / gpuValid : 0.0, gpuSumR / n);
    char label[128];
    snprintf(label, sizeof(label), "grazing-center: GPU mean-over-all-n matches reference (gpu=%.4f, ref=%.4f)",
             gpuSumR / n, refSumR / n);
    expectNear(label, gpuSumR / n, refSumR / n, std::max(0.05, std::fabs(refSumR / n) * 0.1));
}

static void testAtan2Zero(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    simd::float2 inputs[9] = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f},
        {1e-3f, 1e-3f}, {1e-4f, 1e-4f}, {1e-5f, 1e-5f},
        {1e-6f, 1e-6f}, {1e-7f, 1e-7f}, {1e-8f, 1e-8f},
    };
    int n = 9;
    id<MTLBuffer> inBuf = makeBuffer(device, inputs, sizeof(inputs));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_atan2Zero", @[inBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    for (int i = 0; i < n; ++i) {
        fprintf(stderr, "[atan2 diagnostic] atan2(%.9f,%.9f) = %.6f  (expected ~%.6f)\n",
                inputs[i].y, inputs[i].x, out[i], std::atan2((double)inputs[i].y, (double)inputs[i].x));
    }
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
        // See metal_poc_shader_files.h's own comment - the shader source
        // is split across several files on disk but still compiled as
        // ONE concatenated string (metal_poc.mm's own loader does the
        // same thing, from the same ordered file list).
        NSMutableString* shaderSource = [NSMutableString string];
        int shaderFileCount = 0;
        const char* const* shaderFileNames = metalShaderFileNames(&shaderFileCount);
        for (int i = 0; i < shaderFileCount; ++i) {
            NSString* fragPath = [shaderDir stringByAppendingPathComponent:@(shaderFileNames[i])];
            NSString* fragSource = [NSString stringWithContentsOfFile:fragPath encoding:NSUTF8StringEncoding error:&error];
            if (!fragSource) {
                fprintf(stderr, "FAIL: could not read shader source at %s: %s\n",
                        fragPath.UTF8String, error.localizedDescription.UTF8String);
                return 1;
            }
            [shaderSource appendString:fragSource];
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
        testSampleGGXEnergyTableDevice(device, library, queue);
        testOrenNayarF(device, library, queue);
        testVelvetF(device, library, queue);
        testHenyeyGreensteinPhase(device, library, queue);
        testProjectionLightRadiance(device, library, queue);
        testSampleAreaLightAliasTable(device, library, queue);
        testEqualAreaSphereToSquare(device, library, queue);
        testGoniometricLightRadiance(device, library, queue);
        testHairMp(device, library, queue);
        testHairNp(device, library, queue);
        testHairEvalAndPdf(device, library, queue);
        testHairRandomSweep(device, library, queue);
        testHairSample(device, library, queue);
        testHairBlackFurValidRate(device, library, queue);
        testHairGrazingCenterBias(device, library, queue);
        testAtan2Zero(device, library, queue);

        if (g_failures > 0) {
            fprintf(stderr, "FAIL: %d check(s) failed\n", g_failures);
            return 1;
        }
        fprintf(stderr, "PASS: all metal_poc.metal device-side checks passed\n");
        return 0;
    }
}
