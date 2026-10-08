// metal_poc_shader_tests_layout.mm - the host and shader copies of the shared structs have the same layout.
//
// metal_poc_gpu_types.h (host) and metal_poc_types.metal (shader) each declare Uniforms, TriangleMaterial and a dozen more structs by hand;
// the buffer a kernel reads is the host's bytes interpreted as the shader's struct. Nothing but the "append new fields at the end" rule kept
// them in step, so a field added on one side only, or a type whose size differs between C++ and MSL (a float3 vs a packed one), would shift
// every later field and show up as wrong pictures far from the cause. This asks the shader for its own sizes, alignments and Uniforms field
// offsets (test_structLayouts in metal_poc_test_kernels.metal) and compares them with the host's.
// This file includes the PRODUCTION host types (metal_poc_gpu_types.h), not the shader tests' common header: that header declares its own
// test copies of some of the same structs, which would defeat the comparison (and clash with these names). So it is self-contained and
// returns its number of failures for main() to add up.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_poc_gpu_types.h"

#include <cstddef>
#include <cstdio>
#include <vector>

int testStructLayouts(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    int failures = 0;
    std::vector<uint32_t> expected;
    std::vector<const char*> labels;
#define LAYOUT_PROBE(HostT, ShaderName) \
    expected.push_back((uint32_t)sizeof(HostT)); labels.push_back("sizeof(" ShaderName ")"); \
    expected.push_back((uint32_t)alignof(HostT)); labels.push_back("alignof(" ShaderName ")");
    LAYOUT_PROBE(Uniforms, "Uniforms") LAYOUT_PROBE(TriangleMaterial, "TriangleMaterial") LAYOUT_PROBE(GpuCloudMedium, "GpuCloudMedium")
    LAYOUT_PROBE(GpuRgbGridMedium, "GpuRgbGridMedium") LAYOUT_PROBE(SphereData, "SphereData") LAYOUT_PROBE(InstanceTransform, "InstanceTransform")
    LAYOUT_PROBE(DiskData, "DiskData") LAYOUT_PROBE(CylinderData, "CylinderData") LAYOUT_PROBE(PointLightData, "PointLight")
    LAYOUT_PROBE(DirectionalLightData, "DirectionalLight") LAYOUT_PROBE(ProjectionLightData, "ProjectionLight")
    LAYOUT_PROBE(GoniometricLightData, "GoniometricLight") LAYOUT_PROBE(GpuLensElementData, "LensElement")
    LAYOUT_PROBE(GpuExitPupilBoundsData, "ExitPupilBounds")
#undef LAYOUT_PROBE
#define FIELD_OFFSET(f) expected.push_back((uint32_t)offsetof(Uniforms, f)); labels.push_back("offsetof(Uniforms, " #f ")");
    FIELD_OFFSET(fireflyClamp) FIELD_OFFSET(debugCensus) FIELD_OFFSET(pathRegen) FIELD_OFFSET(fogChromatic) FIELD_OFFSET(fogSigmaT3)
    FIELD_OFFSET(liveWorldPos) FIELD_OFFSET(adaptiveThreshold) FIELD_OFFSET(cameraGlassPrim) FIELD_OFFSET(pbrtHasPortalLight)
    FIELD_OFFSET(pbrtPortalWidth) FIELD_OFFSET(pbrtPortalHeight) FIELD_OFFSET(pbrtPortalScale) FIELD_OFFSET(portalFrameX)
    FIELD_OFFSET(portalFrameY) FIELD_OFFSET(portalFrameZ) FIELD_OFFSET(portalP0) FIELD_OFFSET(portalP2) FIELD_OFFSET(pbrtInstanceFirst)
#undef FIELD_OFFSET

    Uniforms uniforms{};   // contents are irrelevant: the kernel only takes field addresses
    id<MTLBuffer> outBuf = [device newBufferWithLength:expected.size() * sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> uniformBuf = [device newBufferWithBytes:&uniforms length:sizeof(uniforms) options:MTLResourceStorageModeShared];
    id<MTLFunction> fn = [library newFunctionWithName:@"test_structLayouts"];
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = fn ? [device newComputePipelineStateWithFunction:fn error:&error] : nil;
    if (!pipeline) {
        fprintf(stderr, "FAIL: struct layout probe kernel unavailable: %s\n", error ? error.localizedDescription.UTF8String : "no such function");
        return 1;
    }
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:outBuf offset:0 atIndex:0];
    [enc setBuffer:uniformBuf offset:0 atIndex:1];
    [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status == MTLCommandBufferStatusError) { fprintf(stderr, "FAIL: struct layout probe dispatch failed\n"); return 1; }
    const uint32_t* got = (const uint32_t*)outBuf.contents;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (got[i] == expected[i]) continue;
        fprintf(stderr, "FAIL: host and shader disagree on %s: host %u, shader %u\n", labels[i], expected[i], got[i]);
        ++failures;
    }
    return failures;
}
