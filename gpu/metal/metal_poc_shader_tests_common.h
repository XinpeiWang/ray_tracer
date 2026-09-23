// metal_poc_shader_tests_common.h
// Shared setup for metal_poc_shader_tests's device-side test suite -
// includes, assertion helpers, GPU-mirror structs, and small kernel-
// dispatch helpers every metal_poc_shader_tests_*.mm split file below
// needs, plus prototypes for every test function so metal_poc_shader_
// tests.mm's own main() (now just the driver: device/library setup,
// calling each test in order, reporting g_failures) can call across
// translation units. Split out of metal_poc_shader_tests.mm once that
// one file's own 36 test functions grew past ~2000 lines - a pure
// code-motion refactor, no behaviour change, same precedent as metal_
// poc_app.h's own split (docs/METAL_GPU_FEASIBILITY.md section 187) and
// metal_poc_scenes_*.mm's before it.
//
// Free functions below were `static` in their original single-TU home;
// each is `inline` here instead (identical body, just ODR-safe to
// define in a header multiple translation units now include) - the
// ONLY mechanical change this split made to any of them. Test functions
// themselves lost `static` in their own new .mm files (real external
// linkage now needed, since main() calls them from a different TU) but
// otherwise moved verbatim.
//
// Real regression coverage for the DEVICE-side half of metal_poc.metal -
// the half metal_poc_math_tests.cpp (docs/METAL_GPU_FEASIBILITY.md
// section 59) explicitly could NOT close, since that one only covers
// plain host C++ math with no GPU dependency at all. Everything checked
// across this whole test suite only ever runs on the GPU: frDielectric(),
// the GGX distribution/masking-shadowing functions, checkerColor(),
// spotLightFalloff(), fresnelSchlickConductor(), henyeyGreensteinPhase(),
// projectionLightRadiance(), and sampleAreaLight()'s own REAL alias-table
// lookup (not metal_poc_math_tests.cpp's own host-side test MIRROR of
// it).
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
#pragma once

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

// perlin_noise<T>()/CloudMedium<T>/SampledGrid<T> - the CPU/OptiX
// reference for the medium-sphere numeric cross-checks below (section
// 195's own tests) - not pulled in by bxdfs.h above, which only covers
// BxDF-family headers.
#include "../../src/shared/noise.h"
#include "../../src/shared/cloud_medium.h"
#include "../../src/shared/sampled_grid.h"

using simd::float3;

inline int g_failures = 0;

inline void expectNear(const char* label, double actual, double expected, double tolerance) {
    if (std::fabs(actual - expected) > tolerance) {
        fprintf(stderr, "FAIL: %s - got %.6f, expected %.6f (tolerance %.6f)\n",
                label, actual, expected, tolerance);
        ++g_failures;
    }
}

inline void expectTrue(const char* label, bool condition) {
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
// Mirrors metal_poc_types.metal's GpuCloudMedium byte-for-byte - needed
// to build the constant-buffer argument test_gpuCloudDensity expects.
// Only `.density`/`.frequency` are actually read by gpuCloudDensity()
// itself; every other field is zero-initialized and unused by this test.
struct GpuCloudMediumGPU {
    float boundsMin[3]{};
    float boundsMax[3]{};
    float worldToMediumMat[9]{};
    float worldToMediumTranslate[3]{};
    float sigmaA = 0.0f;
    float sigmaS = 0.0f;
    float density = 1.0f;
    float wispiness = 0.0f;
    float frequency = 1.0f;
};

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
inline void buildAliasTable(std::vector<AreaLightGPU>& lights) {
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
inline bool runKernel(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue,
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

inline id<MTLBuffer> makeBuffer(id<MTLDevice> device, const void* data, size_t length) {
    return [device newBufferWithBytes:data length:length options:MTLResourceStorageModeShared];
}

inline id<MTLBuffer> makeOutputBuffer(id<MTLDevice> device, size_t length) {
    return [device newBufferWithLength:length options:MTLResourceStorageModeShared];
}

// --- Test function prototypes, grouped by which metal_poc_shader_tests_*.mm
// file below defines each - main() (metal_poc_shader_tests.mm) calls all 36
// in the same order the original single-file version did.

// metal_poc_shader_tests_materials.mm - BSDF/Fresnel/GGX/material numeric
// cross-checks and property tests, plus the misc numeric edge-case checks
// (checkerColor, atan2) that don't belong to media/lights/hair.
void testFrDielectric(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testLambertianAndDiffuseTransmissionPdf(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testGgxConductorFAndPdf(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testPrincipledPdf(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testNormalizedFresnelF(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testLayeredCoatedProperties(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testCauchyEta(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testGgxD(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testGgxG1SmoothLimit(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testCheckerColor(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testFresnelSchlickConductor(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testFrComplexRGB(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testBuildAnisotropicOnb(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testSampleGGXEnergyTableDevice(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testOrenNayarF(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testVelvetF(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testAtan2Zero(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);

// metal_poc_shader_tests_media.mm - participating-medium/volumetric and
// phase-function checks (Perlin noise, cloud/RGB-grid density, HG phase).
void testPerlinNoise3D(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testGpuCloudDensity(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testGpuRgbGridTrilinear(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testSampleHenyeyGreensteinProperties(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testHenyeyGreensteinPhase(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);

// metal_poc_shader_tests_lights.mm - light-type-specific checks (spot,
// environment-direction sampling, projection, area-light alias table,
// equal-area mapping, goniometric).
void testSpotLightFalloff(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testEnvironmentDirectionSampling(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testProjectionLightRadiance(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testSampleAreaLightAliasTable(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testEqualAreaSphereToSquare(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testGoniometricLightRadiance(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);

// metal_poc_shader_tests_hair.mm - the HairBxDF<double> cross-check series
// (section 183/197's own B11/C9 hair investigations).
void testHairMp(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testHairNp(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testHairEvalAndPdf(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testHairComputeAp(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testHairRandomSweep(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testHairSample(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testHairBlackFurValidRate(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
void testHairGrazingCenterBias(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue);
