// metal_poc.mm
// Host side of the Metal ray tracing proof-of-concept - see
// docs/METAL_GPU_FEASIBILITY.md. Builds a small hardcoded Cornell-box-like
// triangle scene, uploads it into a Metal acceleration structure, dispatches
// metal_poc.metal's inline-intersection compute kernel, and writes the
// result to a PNG via this project's existing stb_image_write.h - same
// output path the CPU renderer already uses, so the two are trivially
// visually comparable.
//
// Deliberately standalone (own main(), not wired into launcher/main.cpp or
// CMakeLists.txt yet): this is the "self-contained, time-boxed spike" the
// feasibility doc's Suggested Next Step calls for, proving the pipeline
// shape works before any of the real material/light/shape porting work
// starts.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../src/external/stb_image_write.h"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <simd/simd.h>

using simd::float3;

// A plain, tightly-packed 12-byte vertex type - MTLAccelerationStructure
// TriangleGeometryDescriptor reads raw bytes at vertexStride, and simd's
// own float3 is 16-byte-aligned/padded, not 12, so a hand-rolled struct
// (no simd padding) is what actually matches a tight vertexStride.
struct PackedFloat3 {
    float x, y, z;
};

// Mirrors metal_poc.metal's Uniforms/TriangleMaterial byte-for-byte -
// PackedFloat3 (not simd::float3) for every vector field, same reasoning
// as that file's own comment: simd::float3 is 16-byte aligned inside a
// struct, PackedFloat3 is a plain 12-byte triple with no padding, and a
// host/device struct-layout mismatch is a silent, hard-to-spot bug class
// worth designing out rather than debugging into.
struct Uniforms {
    PackedFloat3 cameraPos;
    PackedFloat3 cameraForward;
    PackedFloat3 cameraRight;
    PackedFloat3 cameraUp;
    float tanHalfFov;
    float aspect;
    uint32_t width;
    uint32_t height;
    uint32_t samplesPerPixel;
    uint32_t maxDepth;
    uint32_t frameSeed;
};

// materialType: 0 = Lambertian, 1 = mirror - see metal_poc.metal's own
// comment on this struct for why it's this minimal.
struct TriangleMaterial {
    PackedFloat3 color;
    uint32_t materialType;
};

// A quad (4 verts, wound as 2 triangles) sharing one flat colour and
// material type - the smallest scene-authoring shape that can build a
// real Cornell box without hand-listing 30 individual vertices.
static void addQuad(std::vector<PackedFloat3>& verts,
                     std::vector<TriangleMaterial>& materials,
                     float3 a, float3 b, float3 c, float3 d,
                     float3 color, uint32_t materialType = 0) {
    // a-b-c-d wound so (a,b,c) and (a,c,d) both face outward consistently.
    auto push = [&](float3 v) { verts.push_back(PackedFloat3{v.x, v.y, v.z}); };
    push(a); push(b); push(c);
    push(a); push(c); push(d);
    PackedFloat3 packedColor{color.x, color.y, color.z};
    materials.push_back({packedColor, materialType});
    materials.push_back({packedColor, materialType});
}

int main(int argc, const char** argv) {
    @autoreleasepool {
        const uint32_t width = (argc > 1) ? (uint32_t)atoi(argv[1]) : 400;
        const uint32_t height = (argc > 2) ? (uint32_t)atoi(argv[2]) : 400;
        const char* outPath = (argc > 3) ? argv[3] : "/tmp/metal_poc_render.png";

        // MTLCreateSystemDefaultDevice() is explicitly documented as
        // unsupported for command-line/daemon processes (confirmed via
        // `log show`: "Use of MTLCreateSystemDefaultDevice is not
        // supported for non-interactive (commandline or daemon) apps. Use
        // MTLCopyAllDevices(WithObserver) instead.") - this POC is a plain
        // CLI tool, not an app bundle, so it needs the enumeration API.
        id<MTLDevice> device = nil;
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices.count > 0) device = devices[0];
        if (!device) {
            fprintf(stderr, "No Metal device available.\n");
            return 1;
        }
        fprintf(stderr, "Metal device: %s\n", device.name.UTF8String);
        if (!device.supportsRaytracing) {
            fprintf(stderr, "Device does not support hardware raytracing.\n");
            return 1;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];

        // --- Scene: a Cornell-box-like room, world units ~[-1,1] --------
        // Matches this project's CPU Cornell box in spirit (floor/ceiling/
        // back wall + coloured side walls + an object), not in exact
        // dimensions - this POC's scene is entirely separate authored data,
        // not a shared asset with cpu_renderer/.
        std::vector<PackedFloat3> verts;
        std::vector<TriangleMaterial> materials;

        const float3 white{0.73f, 0.73f, 0.73f};
        const float3 red{0.65f, 0.05f, 0.05f};
        const float3 green{0.12f, 0.45f, 0.15f};
        const float3 mirrorTint{0.95f, 0.95f, 0.95f};

        // Floor (y = -1)
        addQuad(verts, materials, float3{-1,-1,-1}, float3{1,-1,-1}, float3{1,-1,1}, float3{-1,-1,1}, white);
        // Ceiling (y = 1)
        addQuad(verts, materials, float3{-1,1,1}, float3{1,1,1}, float3{1,1,-1}, float3{-1,1,-1}, white);
        // Back wall (z = -1)
        addQuad(verts, materials, float3{-1,-1,-1}, float3{-1,1,-1}, float3{1,1,-1}, float3{1,-1,-1}, white);
        // Left wall (x = -1), red
        addQuad(verts, materials, float3{-1,-1,1}, float3{-1,1,1}, float3{-1,1,-1}, float3{-1,-1,-1}, red);
        // Right wall (x = 1), green
        addQuad(verts, materials, float3{1,-1,-1}, float3{1,1,-1}, float3{1,1,1}, float3{1,-1,1}, green);
        // A small tilted mirror in the middle, floating just above the
        // floor - proves per-primitive material-TYPE branching works (not
        // just per-primitive colour, which step 1 already covered): a
        // correct render shows the room reflected in it, not a flat grey
        // quad.
        addQuad(verts, materials,
                float3{-0.35f,-0.9f,-0.3f}, float3{0.25f,-0.9f,-0.5f},
                float3{0.25f,-0.2f,-0.5f}, float3{-0.35f,-0.2f,-0.3f},
                mirrorTint, /*materialType=*/1);

        const uint32_t triangleCount = (uint32_t)materials.size();
        fprintf(stderr, "Scene: %u triangles\n", triangleCount);

        id<MTLBuffer> vertexBuffer = [device newBufferWithBytes:verts.data()
            length:verts.size() * sizeof(PackedFloat3)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> materialBuffer = [device newBufferWithBytes:materials.data()
            length:materials.size() * sizeof(TriangleMaterial)
            options:MTLResourceStorageModeShared];

        // --- Primitive acceleration structure (the mesh's own BVH) ------
        MTLAccelerationStructureTriangleGeometryDescriptor* geomDesc =
            [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
        geomDesc.vertexBuffer = vertexBuffer;
        geomDesc.vertexStride = sizeof(PackedFloat3);
        geomDesc.triangleCount = triangleCount;

        MTLPrimitiveAccelerationStructureDescriptor* primDesc =
            [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        primDesc.geometryDescriptors = @[geomDesc];

        MTLAccelerationStructureSizes primSizes = [device accelerationStructureSizesWithDescriptor:primDesc];
        id<MTLAccelerationStructure> primAS = [device newAccelerationStructureWithSize:primSizes.accelerationStructureSize];
        id<MTLBuffer> primScratch = [device newBufferWithLength:primSizes.buildScratchBufferSize
            options:MTLResourceStorageModePrivate];

        id<MTLCommandBuffer> buildCmd = [queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> buildEnc = [buildCmd accelerationStructureCommandEncoder];
        [buildEnc buildAccelerationStructure:primAS descriptor:primDesc scratchBuffer:primScratch scratchBufferOffset:0];
        [buildEnc endEncoding];
        [buildCmd commit];
        [buildCmd waitUntilCompleted];
        if (buildCmd.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "Primitive AS build failed: %s\n", buildCmd.error.localizedDescription.UTF8String);
            return 1;
        }

        // --- Instance acceleration structure (wraps the primitive AS, --
        // identity transform - a real renderer would have one instance
        // per pbrt Shape/ObjectInstance; this POC has exactly one).
        MTLAccelerationStructureInstanceDescriptor instanceDesc = {};
        instanceDesc.accelerationStructureIndex = 0;
        instanceDesc.options = MTLAccelerationStructureInstanceOptionNone;
        instanceDesc.mask = 0xFF;
        instanceDesc.intersectionFunctionTableOffset = 0;
        MTLPackedFloat4x3 identity;
        identity.columns[0] = MTLPackedFloat3Make(1, 0, 0);
        identity.columns[1] = MTLPackedFloat3Make(0, 1, 0);
        identity.columns[2] = MTLPackedFloat3Make(0, 0, 1);
        identity.columns[3] = MTLPackedFloat3Make(0, 0, 0);
        instanceDesc.transformationMatrix = identity;

        id<MTLBuffer> instanceBuffer = [device newBufferWithBytes:&instanceDesc
            length:sizeof(MTLAccelerationStructureInstanceDescriptor)
            options:MTLResourceStorageModeShared];

        MTLInstanceAccelerationStructureDescriptor* instAccelDesc =
            [MTLInstanceAccelerationStructureDescriptor descriptor];
        instAccelDesc.instancedAccelerationStructures = @[primAS];
        instAccelDesc.instanceCount = 1;
        instAccelDesc.instanceDescriptorBuffer = instanceBuffer;

        MTLAccelerationStructureSizes instSizes = [device accelerationStructureSizesWithDescriptor:instAccelDesc];
        id<MTLAccelerationStructure> instAS = [device newAccelerationStructureWithSize:instSizes.accelerationStructureSize];
        id<MTLBuffer> instScratch = [device newBufferWithLength:instSizes.buildScratchBufferSize
            options:MTLResourceStorageModePrivate];

        id<MTLCommandBuffer> buildCmd2 = [queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> buildEnc2 = [buildCmd2 accelerationStructureCommandEncoder];
        [buildEnc2 buildAccelerationStructure:instAS descriptor:instAccelDesc scratchBuffer:instScratch scratchBufferOffset:0];
        [buildEnc2 endEncoding];
        [buildCmd2 commit];
        [buildCmd2 waitUntilCompleted];
        if (buildCmd2.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "Instance AS build failed: %s\n", buildCmd2.error.localizedDescription.UTF8String);
            return 1;
        }

        // --- Compile the shader library from source at runtime ---------
        NSError* error = nil;
        NSString* shaderPath = @(__FILE__);
        shaderPath = [[shaderPath stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"metal_poc.metal"];
        NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath encoding:NSUTF8StringEncoding error:&error];
        if (!shaderSource) {
            fprintf(stderr, "Failed to read shader source at %s: %s\n",
                shaderPath.UTF8String, error.localizedDescription.UTF8String);
            return 1;
        }
        MTLCompileOptions* compileOpts = [MTLCompileOptions new];
        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:compileOpts error:&error];
        if (!library) {
            fprintf(stderr, "Shader compile failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLFunction> kernelFn = [library newFunctionWithName:@"primaryRayKernel"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernelFn error:&error];
        if (!pipeline) {
            fprintf(stderr, "Pipeline creation failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        // --- Output texture + uniforms ----------------------------------
        MTLTextureDescriptor* texDesc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
            width:width height:height mipmapped:NO];
        texDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
        texDesc.storageMode = MTLStorageModeShared;
        id<MTLTexture> outTexture = [device newTextureWithDescriptor:texDesc];

        const uint32_t samplesPerPixel = (argc > 4) ? (uint32_t)atoi(argv[4]) : 64;
        const uint32_t maxDepth = (argc > 5) ? (uint32_t)atoi(argv[5]) : 8;
        fprintf(stderr, "Samples/pixel: %u, max depth: %u\n", samplesPerPixel, maxDepth);

        Uniforms uniforms{};
        uniforms.cameraPos = PackedFloat3{0.0f, 0.0f, 3.2f};
        float3 forward = simd::normalize(float3{0, 0, -1});
        uniforms.cameraForward = PackedFloat3{forward.x, forward.y, forward.z};
        uniforms.cameraRight = PackedFloat3{1, 0, 0};
        uniforms.cameraUp = PackedFloat3{0, 1, 0};
        uniforms.tanHalfFov = tanf(0.5f * 40.0f * (float)M_PI / 180.0f);
        uniforms.aspect = (float)width / (float)height;
        uniforms.width = width;
        uniforms.height = height;
        uniforms.samplesPerPixel = samplesPerPixel;
        uniforms.maxDepth = maxDepth;
        uniforms.frameSeed = 1u;
        id<MTLBuffer> uniformBuffer = [device newBufferWithBytes:&uniforms length:sizeof(Uniforms) options:MTLResourceStorageModeShared];

        // --- Dispatch ----------------------------------------------------
        id<MTLCommandBuffer> renderCmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [renderCmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setTexture:outTexture atIndex:0];
        [enc setAccelerationStructure:instAS atBufferIndex:0];
        [enc setBuffer:uniformBuffer offset:0 atIndex:1];
        [enc setBuffer:materialBuffer offset:0 atIndex:2];
        [enc setBuffer:vertexBuffer offset:0 atIndex:3];
        // Mark the AS + its dependent primitive AS as used so Metal knows
        // about the indirection - required for instance acceleration
        // structures referencing primitive ones.
        [enc useResource:primAS usage:MTLResourceUsageRead];

        MTLSize gridSize = MTLSizeMake(width, height, 1);
        NSUInteger w = pipeline.threadExecutionWidth;
        NSUInteger h = pipeline.maxTotalThreadsPerThreadgroup / w;
        MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [enc endEncoding];
        [renderCmd commit];
        [renderCmd waitUntilCompleted];
        if (renderCmd.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "Render dispatch failed: %s\n", renderCmd.error.localizedDescription.UTF8String);
            return 1;
        }

        // --- Read back + write PNG (8-bit sRGB-ish gamma, same simple ---
        // 1/2.2 approximation good enough for a POC comparison image)
        std::vector<float> pixels(width * height * 4);
        MTLRegion region = MTLRegionMake2D(0, 0, width, height);
        [outTexture getBytes:pixels.data() bytesPerRow:width * 4 * sizeof(float) fromRegion:region mipmapLevel:0];

        std::vector<uint8_t> ldr(width * height * 3);
        for (uint32_t i = 0; i < width * height; ++i) {
            for (int c = 0; c < 3; ++c) {
                float v = pixels[i * 4 + c];
                v = powf(fmaxf(v, 0.0f), 1.0f / 2.2f);
                v = fminf(v, 1.0f);
                ldr[i * 3 + c] = (uint8_t)(v * 255.0f + 0.5f);
            }
        }
        stbi_write_png(outPath, width, height, 3, ldr.data(), width * 3);
        fprintf(stderr, "Wrote %s (%ux%u)\n", outPath, width, height);
    }
    return 0;
}
