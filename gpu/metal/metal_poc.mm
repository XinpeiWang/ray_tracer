// metal_poc.mm
// Host side of the Metal ray tracing proof-of-concept - see
// docs/METAL_GPU_FEASIBILITY.md. Builds a small hardcoded Cornell-box-like
// triangle scene, uploads it into a Metal acceleration structure, dispatches
// metal_poc.metal's inline-intersection compute kernel, and writes the
// result to a PNG via this project's existing stb_image_write.h - same
// output path the CPU renderer already uses, so the two are trivially
// visually comparable.
//
// Deliberately standalone (own main(), a separate CLI tool from
// ray_tracer/scene_metadata, not called by launcher/main.cpp or the Qt
// GUI): this is the "self-contained, time-boxed spike" the feasibility
// doc's Suggested Next Step calls for, proving the pipeline shape works
// before any of the real material/light/shape porting work starts. It IS
// now CMake-integrated (root CMakeLists.txt's RT_BUILD_METAL option) as
// its own metal_poc target, separately from ray_tracer/optix_renderer -
// build with `cmake -B build -DRT_BUILD_METAL=ON && cmake --build build
// --target metal_poc`.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../src/external/stb_image_write.h"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <fstream>
#include <sstream>
#include <string>
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

// materialType: 0 = Lambertian, 1 = mirror, 2 = dielectric (glass) - see
// metal_poc.metal's own comment on this struct for why it's this minimal.
// ior is only meaningful for materialType == 2, emission only nonzero for
// the light quad - both carried on every entry anyway, see that file's
// comment on the same tradeoff.
struct TriangleMaterial {
    PackedFloat3 color;
    uint32_t materialType;
    float ior;
    PackedFloat3 emission;
};

// Mirrors metal_poc.metal's SphereData byte-for-byte.
struct SphereData {
    PackedFloat3 center;
    float radius;
};

// A quad (4 verts, wound as 2 triangles) sharing one flat colour and
// material type - the smallest scene-authoring shape that can build a
// real Cornell box without hand-listing 30 individual vertices.
static void addQuad(std::vector<PackedFloat3>& verts,
                     std::vector<TriangleMaterial>& materials,
                     float3 a, float3 b, float3 c, float3 d,
                     float3 color, uint32_t materialType = 0,
                     float3 emission = simd::make_float3(0, 0, 0)) {
    // a-b-c-d wound so (a,b,c) and (a,c,d) both face outward consistently.
    auto push = [&](float3 v) { verts.push_back(PackedFloat3{v.x, v.y, v.z}); };
    push(a); push(b); push(c);
    push(a); push(c); push(d);
    PackedFloat3 packedColor{color.x, color.y, color.z};
    PackedFloat3 packedEmission{emission.x, emission.y, emission.z};
    materials.push_back({packedColor, materialType, 1.0f, packedEmission});
    materials.push_back({packedColor, materialType, 1.0f, packedEmission});
}

// A minimal Wavefront OBJ loader: positions + faces only (`v`/`f`), no
// texcoords/materials/groups/smoothing - this project's own real loaders
// (src/shared/pbrt_load.h -> pbrt_cpu_builder.h/pbrt_gpu_builder.h) are
// full pbrt-v4 scene parsers; this is deliberately the smallest thing that
// can prove "load an arbitrary real mesh, not just hand-authored
// axis-aligned quads and an analytic sphere" - the first genuinely
// data-driven geometry in this POC. Faces are fan-triangulated (n>3
// polygon -> n-2 triangles sharing vertex 0), matching how this project's
// own CPU loader handles polygons that aren't already triangles. Vertex
// normals (`vn`) in the file are read but NOT used - every triangle still
// gets a flat face normal computed from its own 3 positions (faceNormalFor
// in metal_poc.metal), same as every other mesh in this scene; smooth
// per-vertex-normal interpolation would need barycentric-coordinate
// plumbing this POC doesn't have yet, so Suzanne renders faceted here, a
// real (if visually rougher) limitation, not a bug.
//
// The mesh is auto-fit to `targetSize` (its largest bounding-box
// dimension scaled to that value) and recentred at `center` - real .obj
// files come in whatever units/scale their author used, and this scene's
// room is a fixed [-1,1] box, so SOME normalization is unavoidable rather
// than a hardcoded scale constant that would only happen to work for this
// one file.
static bool loadObjMesh(const std::string& path,
                         std::vector<PackedFloat3>& verts,
                         std::vector<TriangleMaterial>& materials,
                         float3 color, float3 center, float targetSize) {
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "Could not open OBJ file: %s\n", path.c_str());
        return false;
    }

    std::vector<float3> positions;
    std::vector<std::vector<int>> faceVertexIndices; // 0-based, post-fixup

    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            positions.push_back(simd::make_float3(x, y, z));
        } else if (tag == "f") {
            std::vector<int> indices;
            std::string token;
            while (ss >> token) {
                // Token is "v", "v/vt", "v//vn", or "v/vt/vn" - only the
                // first (position) index matters here.
                int vIdx = std::atoi(token.c_str());
                // OBJ indices are 1-based; a negative index is relative to
                // the current vertex count (rare, but real files use it) -
                // both normalized to a plain 0-based index here.
                if (vIdx < 0) vIdx = (int)positions.size() + vIdx + 1;
                indices.push_back(vIdx - 1);
            }
            if (indices.size() >= 3) faceVertexIndices.push_back(indices);
        }
    }

    if (positions.empty() || faceVertexIndices.empty()) {
        fprintf(stderr, "OBJ file had no usable geometry: %s\n", path.c_str());
        return false;
    }

    float3 bboxMin = simd::make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 bboxMax = simd::make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (const float3& p : positions) {
        bboxMin = simd::min(bboxMin, p);
        bboxMax = simd::max(bboxMax, p);
    }

    float3 extent = bboxMax - bboxMin;
    float largestDim = std::max(extent.x, std::max(extent.y, extent.z));
    float scale = (largestDim > 0.0f) ? (targetSize / largestDim) : 1.0f;
    float3 bboxCenter = (bboxMin + bboxMax) * 0.5f;

    auto transform = [&](const float3& p) -> float3 {
        return (p - bboxCenter) * scale + center;
    };

    uint32_t triangleCount = 0;
    for (const std::vector<int>& face : faceVertexIndices) {
        // Fan triangulation from vertex 0 - correct for the convex/near-
        // convex polygons a typical modeled mesh's faces are (this file's
        // own quads included), not a general concave-polygon triangulator.
        for (size_t i = 1; i + 1 < face.size(); ++i) {
            int i0 = face[0], i1 = face[i], i2 = face[i + 1];
            if (i0 < 0 || i0 >= (int)positions.size() ||
                i1 < 0 || i1 >= (int)positions.size() ||
                i2 < 0 || i2 >= (int)positions.size()) {
                continue; // malformed index - skip rather than crash
            }
            float3 a = transform(positions[i0]);
            float3 b = transform(positions[i1]);
            float3 c = transform(positions[i2]);
            verts.push_back(PackedFloat3{a.x, a.y, a.z});
            verts.push_back(PackedFloat3{b.x, b.y, b.z});
            verts.push_back(PackedFloat3{c.x, c.y, c.z});
            ++triangleCount;
        }
    }

    PackedFloat3 packedColor{color.x, color.y, color.z};
    TriangleMaterial mat{packedColor, /*materialType=*/0, 1.0f, PackedFloat3{0, 0, 0}};
    for (uint32_t i = 0; i < triangleCount; ++i) materials.push_back(mat);

    fprintf(stderr, "Loaded %s: %zu positions, %u triangles (scale %.4f)\n",
            path.c_str(), positions.size(), triangleCount, scale);
    return true;
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
        // Suzanne (Blender's monkey mascot, models/suzanne.obj - a real
        // mesh, 500 faces) replaces the earlier flat tilted-quad "mirror
        // test object": mirror MATERIAL coverage is already proven (the
        // sphere/dielectric PR's own mirror-quad screenshots), what this
        // scene hadn't tested yet is real, DATA-DRIVEN geometry with
        // genuine per-triangle normal variation, not a hand-authored
        // axis-aligned quad. Lambertian so its form reads clearly via
        // shading rather than showing room reflections. RT_MODELS_DIR
        // mirrors RT_METAL_SHADER_DIR's own fallback shape below.
#ifdef RT_MODELS_DIR
        NSString* modelsDir = @(RT_MODELS_DIR);
#else
        NSString* modelsDir = [[@(__FILE__) stringByDeletingLastPathComponent]
            stringByAppendingPathComponent:@"../../models"];
#endif
        NSString* suzannePath = [modelsDir stringByAppendingPathComponent:@"suzanne.obj"];
        const float3 bronze{0.55f, 0.35f, 0.15f};
        if (!loadObjMesh(suzannePath.UTF8String, verts, materials, bronze,
                          /*center=*/float3{-0.05f, -0.55f, -0.3f}, /*targetSize=*/0.75f)) {
            fprintf(stderr, "Continuing without Suzanne - check RT_MODELS_DIR / models/suzanne.obj.\n");
        }

        // Area light: a small quad hanging just under the ceiling
        // (y=0.98, not y=1 itself - avoids z-fighting/coplanar overlap
        // with the ceiling's own quad above), facing straight down. Real
        // geometry with nonzero emission, replacing the earlier hardcoded
        // directional light entirely - see metal_poc.metal's own comment
        // on kLightCenter/kLightHalfExtents/kLightNormal, which have to
        // stay in sync with this quad's own position/size/orientation by
        // hand (this POC's one deliberately-hardcoded light, not a real
        // light-list abstraction - see that file's comment on why).
        addQuad(verts, materials,
                float3{-0.3f,0.98f,-0.3f}, float3{0.3f,0.98f,-0.3f},
                float3{0.3f,0.98f,0.3f}, float3{-0.3f,0.98f,0.3f},
                white, /*materialType=*/0, /*emission=*/float3{15.0f,15.0f,14.0f});

        // One glass sphere, right side of the floor - a custom (non-
        // triangle) primitive via a bounding-box acceleration structure +
        // intersection function (metal_poc.metal's sphereIntersectionFunction),
        // the one "Medium risk, unconfirmed" item docs/METAL_GPU_
        // FEASIBILITY.md section 3 flagged that the room/mirror geometry
        // so far hadn't actually exercised (triangles only). ior 1.5
        // matches common glass, same value this project's own CPU Cornell
        // box scene (A1) uses for its glass sphere.
        SphereData sphere{PackedFloat3{0.35f, -0.65f, 0.15f}, 0.35f};
        TriangleMaterial sphereMaterial{
            PackedFloat3{1.0f, 1.0f, 1.0f}, /*materialType=*/2, /*ior=*/1.5f, PackedFloat3{0, 0, 0}};

        const uint32_t triangleCount = (uint32_t)materials.size();
        fprintf(stderr, "Scene: %u triangles, 1 sphere\n", triangleCount);

        id<MTLBuffer> vertexBuffer = [device newBufferWithBytes:verts.data()
            length:verts.size() * sizeof(PackedFloat3)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> materialBuffer = [device newBufferWithBytes:materials.data()
            length:materials.size() * sizeof(TriangleMaterial)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> sphereBuffer = [device newBufferWithBytes:&sphere
            length:sizeof(SphereData) options:MTLResourceStorageModeShared];
        id<MTLBuffer> sphereMaterialBuffer = [device newBufferWithBytes:&sphereMaterial
            length:sizeof(TriangleMaterial) options:MTLResourceStorageModeShared];

        // --- Primitive acceleration structure (the mesh's own BVH) ------
        MTLAccelerationStructureTriangleGeometryDescriptor* geomDesc =
            [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
        geomDesc.vertexBuffer = vertexBuffer;
        geomDesc.vertexStride = sizeof(PackedFloat3);
        geomDesc.triangleCount = triangleCount;
        // Explicit, not relying on the default: opaque means no any-hit
        // shader gets consulted for this geometry's hits at all, so the
        // hardware triangle intersector's result is taken directly - this
        // matters once a function table is bound at trace time at all
        // (added below, for the sphere), since without this a triangle
        // hit could otherwise get routed through the SAME table slot the
        // sphere's own intersection function occupies.
        geomDesc.opaque = YES;

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

        // --- Second primitive acceleration structure: the sphere's own --
        // bounding-box geometry (a custom/non-triangle primitive has no
        // vertex data at all as far as the acceleration structure is
        // concerned - just an AABB per primitive, with the real
        // intersection test deferred to sphereIntersectionFunction at
        // trace time).
        MTLAxisAlignedBoundingBox sphereBounds;
        sphereBounds.min = MTLPackedFloat3Make(
            sphere.center.x - sphere.radius, sphere.center.y - sphere.radius, sphere.center.z - sphere.radius);
        sphereBounds.max = MTLPackedFloat3Make(
            sphere.center.x + sphere.radius, sphere.center.y + sphere.radius, sphere.center.z + sphere.radius);
        id<MTLBuffer> boundingBoxBuffer = [device newBufferWithBytes:&sphereBounds
            length:sizeof(MTLAxisAlignedBoundingBox) options:MTLResourceStorageModeShared];

        MTLAccelerationStructureBoundingBoxGeometryDescriptor* bboxGeomDesc =
            [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
        bboxGeomDesc.boundingBoxBuffer = boundingBoxBuffer;
        bboxGeomDesc.boundingBoxStride = sizeof(MTLAxisAlignedBoundingBox);
        bboxGeomDesc.boundingBoxCount = 1;
        // intersectionFunctionTableOffset here is this GEOMETRY's own
        // slot within whatever function table gets bound at trace time -
        // 0 since sphereIntersectionFunction is the only entry in it
        // (set up below, alongside the compute pipeline).
        bboxGeomDesc.intersectionFunctionTableOffset = 0;
        // Opaque here too: sphereIntersectionFunction is the REQUIRED
        // primitive-intersection test for this custom geometry (always
        // invoked, opaque or not - there's no hardware fallback for a
        // bounding-box primitive), so opaque just means "accept its
        // result directly," skip a second any-hit pass on top of it,
        // exactly this POC's one-test-decides-it shape.
        bboxGeomDesc.opaque = YES;

        MTLPrimitiveAccelerationStructureDescriptor* sphereAccelDesc =
            [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        sphereAccelDesc.geometryDescriptors = @[bboxGeomDesc];

        MTLAccelerationStructureSizes sphereSizes = [device accelerationStructureSizesWithDescriptor:sphereAccelDesc];
        id<MTLAccelerationStructure> sphereAS = [device newAccelerationStructureWithSize:sphereSizes.accelerationStructureSize];
        id<MTLBuffer> sphereScratch = [device newBufferWithLength:sphereSizes.buildScratchBufferSize
            options:MTLResourceStorageModePrivate];

        id<MTLCommandBuffer> sphereBuildCmd = [queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> sphereBuildEnc = [sphereBuildCmd accelerationStructureCommandEncoder];
        [sphereBuildEnc buildAccelerationStructure:sphereAS descriptor:sphereAccelDesc scratchBuffer:sphereScratch scratchBufferOffset:0];
        [sphereBuildEnc endEncoding];
        [sphereBuildCmd commit];
        [sphereBuildCmd waitUntilCompleted];
        if (sphereBuildCmd.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "Sphere AS build failed: %s\n", sphereBuildCmd.error.localizedDescription.UTF8String);
            return 1;
        }

        // --- Instance acceleration structure: two instances, one per ---
        // primitive AS above (identity transform on both - a real
        // renderer would have one instance per pbrt Shape/ObjectInstance,
        // scaled/positioned by its own transform; this POC's two shapes
        // are both already authored in world space directly).
        MTLAccelerationStructureInstanceDescriptor instanceDescs[2] = {};
        MTLPackedFloat4x3 identity;
        identity.columns[0] = MTLPackedFloat3Make(1, 0, 0);
        identity.columns[1] = MTLPackedFloat3Make(0, 1, 0);
        identity.columns[2] = MTLPackedFloat3Make(0, 0, 1);
        identity.columns[3] = MTLPackedFloat3Make(0, 0, 0);

        instanceDescs[0].accelerationStructureIndex = 0; // primAS (triangles)
        instanceDescs[0].options = MTLAccelerationStructureInstanceOptionNone;
        instanceDescs[0].mask = 0xFF;
        instanceDescs[0].intersectionFunctionTableOffset = 0;
        instanceDescs[0].transformationMatrix = identity;

        instanceDescs[1].accelerationStructureIndex = 1; // sphereAS (bounding box)
        instanceDescs[1].options = MTLAccelerationStructureInstanceOptionNone;
        instanceDescs[1].mask = 0xFF;
        instanceDescs[1].intersectionFunctionTableOffset = 0;
        instanceDescs[1].transformationMatrix = identity;

        id<MTLBuffer> instanceBuffer = [device newBufferWithBytes:instanceDescs
            length:sizeof(instanceDescs)
            options:MTLResourceStorageModeShared];

        MTLInstanceAccelerationStructureDescriptor* instAccelDesc =
            [MTLInstanceAccelerationStructureDescriptor descriptor];
        instAccelDesc.instancedAccelerationStructures = @[primAS, sphereAS];
        instAccelDesc.instanceCount = 2;
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
        // RT_METAL_SHADER_DIR is set by CMakeLists.txt's metal_poc target
        // (RT_BUILD_METAL=ON path) to gpu/metal/'s absolute source
        // directory. Falls back to a __FILE__-relative lookup for the
        // ad-hoc `clang++ metal_poc.mm ...` invocation this POC started
        // as (docs/METAL_GPU_FEASIBILITY.md section 7/8/9) and still
        // works fine for a quick manual rebuild without going through
        // CMake at all.
#ifdef RT_METAL_SHADER_DIR
        NSString* shaderDir = @(RT_METAL_SHADER_DIR);
#else
        NSString* shaderDir = [@(__FILE__) stringByDeletingLastPathComponent];
#endif
        NSString* shaderPath = [shaderDir stringByAppendingPathComponent:@"metal_poc.metal"];
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
        id<MTLFunction> sphereIntersectFn = [library newFunctionWithName:@"sphereIntersectionFunction"];

        // The intersection function has to be LINKED into the compute
        // pipeline (MTLLinkedFunctions) before an MTLIntersectionFunction
        // Table naming it can be built - a plain newComputePipelineState
        // WithFunction: (used for step 1/2's triangle-only pipeline) has
        // nowhere to put that linkage, hence the switch to the descriptor-
        // based pipeline creation call here.
        MTLComputePipelineDescriptor* pipelineDesc = [MTLComputePipelineDescriptor new];
        pipelineDesc.computeFunction = kernelFn;
        MTLLinkedFunctions* linkedFns = [MTLLinkedFunctions new];
        linkedFns.functions = @[sphereIntersectFn];
        pipelineDesc.linkedFunctions = linkedFns;

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:pipelineDesc
            options:MTLPipelineOptionNone reflection:nil error:&error];
        if (!pipeline) {
            fprintf(stderr, "Pipeline creation failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        // --- Intersection function table: one slot (index 0), matching --
        // every geometry's intersectionFunctionTableOffset above (bboxGeomDesc's
        // and instanceDescs[1]'s, both 0) - a real multi-custom-primitive
        // scene would have one slot per distinct intersection function,
        // indexed by whatever offsets those geometries/instances declare.
        MTLIntersectionFunctionTableDescriptor* fnTableDesc = [MTLIntersectionFunctionTableDescriptor new];
        fnTableDesc.functionCount = 1;
        id<MTLIntersectionFunctionTable> functionTable = [pipeline newIntersectionFunctionTableWithDescriptor:fnTableDesc];
        id<MTLFunctionHandle> sphereHandle = [pipeline functionHandleWithFunction:sphereIntersectFn];
        [functionTable setFunction:sphereHandle atIndex:0];
        // sphereIntersectionFunction reads its own `spheres` buffer
        // (metal_poc.metal buffer(0), a SEPARATE argument table from the
        // calling kernel's buffer(0..6) - see that file's own comment) -
        // bound here, on the function table, not on the compute encoder.
        [functionTable setBuffer:sphereBuffer offset:0 atIndex:0];

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
        [enc setBuffer:sphereMaterialBuffer offset:0 atIndex:4];
        [enc setBuffer:sphereBuffer offset:0 atIndex:5];
        [enc setIntersectionFunctionTable:functionTable atBufferIndex:6];
        // Mark the AS + its dependent primitive ASes as used so Metal
        // knows about the indirection - required for instance
        // acceleration structures referencing primitive ones (now two:
        // the triangle mesh's and the sphere's bounding-box geometry).
        [enc useResource:primAS usage:MTLResourceUsageRead];
        [enc useResource:sphereAS usage:MTLResourceUsageRead];

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
