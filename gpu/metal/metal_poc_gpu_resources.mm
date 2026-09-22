// metal_poc_gpu_resources.mm
// Stage 3: MetalPocApp::buildGPUResources() - uploads every host-side
// buffer built by buildScene()/buildHandAuthoredScene() into real Metal
// buffers/textures and builds the acceleration structures (primitive +
// instance) - split out of metal_poc.mm once its own Stage 3/4 pair grew
// past ~1200 combined lines (a pure code-motion refactor, no behaviour
// change - same precedent as metal_poc_scenes_a.mm's own split, and
// metal_poc_pbrt_loader.mm's own follow-up). Stage 4 (shader compile +
// dispatch + readback) is the sibling split, metal_poc_dispatch.mm -
// see that file's own header comment for why it's separate from this one
// rather than combined.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_poc_app.h"

// --- Stage 3: upload GPU buffers + build acceleration structures --------
bool MetalPocApp::buildGPUResources() {
    vertexBuffer = [device newBufferWithBytes:verts.data()
        length:verts.size() * sizeof(PackedFloat3)
        options:MTLResourceStorageModeShared];
    normalBuffer = [device newBufferWithBytes:normals.data()
        length:normals.size() * sizeof(PackedFloat3)
        options:MTLResourceStorageModeShared];
    uvBuffer = [device newBufferWithBytes:uvs.data()
        length:uvs.size() * sizeof(PackedFloat2)
        options:MTLResourceStorageModeShared];
    lightBuffer = [device newBufferWithBytes:lights.data()
        length:lights.size() * sizeof(AreaLightData)
        options:MTLResourceStorageModeShared];
    pointLightBuffer = [device newBufferWithBytes:pointLights.data()
        length:pointLights.size() * sizeof(PointLightData)
        options:MTLResourceStorageModeShared];
    directionalLightBuffer = [device newBufferWithBytes:directionalLights.data()
        length:directionalLights.size() * sizeof(DirectionalLightData)
        options:MTLResourceStorageModeShared];
    projectionLightBuffer = [device newBufferWithBytes:projectionLights.data()
        length:projectionLights.size() * sizeof(ProjectionLightData)
        options:MTLResourceStorageModeShared];
    goniometricLightBuffer = [device newBufferWithBytes:goniometricLights.data()
        length:goniometricLights.size() * sizeof(GoniometricLightData)
        options:MTLResourceStorageModeShared];
    // Realistic (multi-element-lens) camera's own lens/exit-pupil-bounds
    // tables (D4/D8, section 157) - empty for every earlier/other scene.
    // NOT the same "zero-length buffer" shape the other optional per-
    // scene buffers below already tolerate, despite this code's own
    // original comment claiming otherwise: every OTHER optional buffer
    // here (point/directional/projection/goniometric lights, etc.) is
    // actually always non-empty in practice, because buildScene()'s own
    // hardcoded base room (buildScene()'s own comment) unconditionally
    // adds at least one of each - so the "tolerates zero-length" claim
    // was never really exercised until these two, the first buffers
    // that ARE genuinely empty for every scene but D4/D8. Confirmed by
    // actually running a non-D4/D8 scene on GPU: newBufferWithBytes:
    // length:0 (std::vector::data() on an empty vector may legally
    // return null - cppreference) returned nil, which
    // checkGpuResource() below correctly treats as fatal, aborting
    // EVERY other hand-authored scene's own GPU render. Fixed by
    // allocating a real (uninitialized, but real) 1-element buffer via
    // newBufferWithLength: instead whenever empty - never read by the
    // shader for these scenes anyway (sampleRealisticCameraRay's own
    // numLensElements==0u/numExitPupilBounds==0u guard, driven by
    // uniforms.numLensElements/numExitPupilBounds below, which still
    // correctly read 0 from these vectors' own real (unpadded) size -
    // this padding is buffer-allocation-only, not a change to that
    // count).
    lensElementBuffer = realisticLensElements.empty()
        ? [device newBufferWithLength:sizeof(GpuLensElementData) options:MTLResourceStorageModeShared]
        : [device newBufferWithBytes:realisticLensElements.data()
              length:realisticLensElements.size() * sizeof(GpuLensElementData)
              options:MTLResourceStorageModeShared];
    exitPupilBoundsBuffer = realisticExitPupilBounds.empty()
        ? [device newBufferWithLength:sizeof(GpuExitPupilBoundsData) options:MTLResourceStorageModeShared]
        : [device newBufferWithBytes:realisticExitPupilBounds.data()
              length:realisticExitPupilBounds.size() * sizeof(GpuExitPupilBoundsData)
              options:MTLResourceStorageModeShared];
    // The goniometric light's own procedural intensity image (built by
    // buildScene()) - a plain single-channel (R8Unorm) texture, sampled
    // device-side via goniometricLightRadiance()'s own bilinear
    // `textureSampler`. `address::repeat` (the same sampler every other
    // texture read in this file already uses) is a genuine no-op here in
    // practice: equalAreaSphereToSquare() always returns a UV strictly
    // inside [0,1]^2 for a valid unit direction, never walking off the
    // image's own edge the way a perspective-projected UV occasionally
    // needs wrapping/clamping to handle.
    MTLTextureDescriptor* goniometricDesc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
        width:(NSUInteger)goniometricImageSize height:(NSUInteger)goniometricImageSize mipmapped:NO];
    goniometricDesc.usage = MTLTextureUsageShaderRead;
    goniometricDesc.storageMode = MTLStorageModeShared;
    goniometricTexture = [device newTextureWithDescriptor:goniometricDesc];
    [goniometricTexture replaceRegion:MTLRegionMake2D(0, 0, (NSUInteger)goniometricImageSize, (NSUInteger)goniometricImageSize)
        mipmapLevel:0 withBytes:goniometricImage.data() bytesPerRow:(NSUInteger)goniometricImageSize];
    materialBuffer = [device newBufferWithBytes:materials.data()
        length:materials.size() * sizeof(TriangleMaterial)
        options:MTLResourceStorageModeShared];
    sphereBuffer = [device newBufferWithBytes:spheres.data()
        length:spheres.size() * sizeof(SphereData) options:MTLResourceStorageModeShared];
    sphereMaterialBuffer = [device newBufferWithBytes:sphereMaterials.data()
        length:sphereMaterials.size() * sizeof(TriangleMaterial) options:MTLResourceStorageModeShared];
    diskBuffer = [device newBufferWithBytes:disks.data()
        length:disks.size() * sizeof(DiskData) options:MTLResourceStorageModeShared];
    diskMaterialBuffer = [device newBufferWithBytes:diskMaterials.data()
        length:diskMaterials.size() * sizeof(TriangleMaterial) options:MTLResourceStorageModeShared];
    // Genuinely empty for every scene but the handful with a real pbrt
    // Shape "cylinder" - the SAME "empty std::vector::data() can return
    // null, newBufferWithBytes:length:0 then returns nil" pitfall
    // lensElementBuffer/exitPupilBoundsBuffer's own comment just above
    // already documents (found there first) - allocating a real
    // 1-element buffer via newBufferWithLength: instead whenever empty,
    // same fix, never read by the shader either (cylinderCount==0 means
    // no bounding-box geometry ever calls cylinderIntersectionFunction
    // at all, and the shading loop's own isCylinder branch is
    // unreachable with no cylinder primitives in the accel structure).
    cylinderBuffer = cylinders.empty()
        ? [device newBufferWithLength:sizeof(CylinderData) options:MTLResourceStorageModeShared]
        : [device newBufferWithBytes:cylinders.data()
              length:cylinders.size() * sizeof(CylinderData) options:MTLResourceStorageModeShared];
    cylinderMaterialBuffer = cylinderMaterials.empty()
        ? [device newBufferWithLength:sizeof(TriangleMaterial) options:MTLResourceStorageModeShared]
        : [device newBufferWithBytes:cylinderMaterials.data()
              length:cylinderMaterials.size() * sizeof(TriangleMaterial) options:MTLResourceStorageModeShared];
    // E2's own cloudMediums - empty for every scene but E2, same "empty
    // std::vector::data() can return null, newBufferWithBytes:length:0
    // then returns nil" guard as cylinderBuffer/cylinderMaterialBuffer
    // just above.
    cloudMediumBuffer = cloudMediums.empty()
        ? [device newBufferWithLength:sizeof(GpuCloudMedium) options:MTLResourceStorageModeShared]
        : [device newBufferWithBytes:cloudMediums.data()
              length:cloudMediums.size() * sizeof(GpuCloudMedium) options:MTLResourceStorageModeShared];
    // E4's own rgbGridMediums/rgbGridData - same empty-buffer guard.
    rgbGridMediumBuffer = rgbGridMediums.empty()
        ? [device newBufferWithLength:sizeof(GpuRgbGridMedium) options:MTLResourceStorageModeShared]
        : [device newBufferWithBytes:rgbGridMediums.data()
              length:rgbGridMediums.size() * sizeof(GpuRgbGridMedium) options:MTLResourceStorageModeShared];
    rgbGridDataBuffer = rgbGridData.empty()
        ? [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared]
        : [device newBufferWithBytes:rgbGridData.data()
              length:rgbGridData.size() * sizeof(float) options:MTLResourceStorageModeShared];

    const uint32_t suzanneTriangleCount = (uint32_t)suzanneMaterials.size();
    suzanneVertexBuffer = [device newBufferWithBytes:suzanneVerts.data()
        length:suzanneVerts.size() * sizeof(PackedFloat3) options:MTLResourceStorageModeShared];
    suzanneNormalBuffer = [device newBufferWithBytes:suzanneNormals.data()
        length:suzanneNormals.size() * sizeof(PackedFloat3) options:MTLResourceStorageModeShared];
    suzanneMaterialBuffer = [device newBufferWithBytes:suzanneMaterials.data()
        length:suzanneMaterials.size() * sizeof(TriangleMaterial) options:MTLResourceStorageModeShared];

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
    primAS = [device newAccelerationStructureWithSize:primSizes.accelerationStructureSize];
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
        return false;
    }

    // --- Second primitive acceleration structure: the spheres' own --
    // bounding-box geometry (a custom/non-triangle primitive has no
    // vertex data at all as far as the acceleration structure is
    // concerned - just an AABB per primitive, with the real
    // intersection test deferred to sphereIntersectionFunction at
    // trace time). One AABB per entry in `spheres`, same index order -
    // sphereIntersectionFunction's own primitive_id indexes both this
    // buffer and `spheres`/`sphereMaterials` identically.
    std::vector<MTLAxisAlignedBoundingBox> sphereBoundsList;
    for (const SphereData& s : spheres) {
        MTLAxisAlignedBoundingBox bounds;
        bounds.min = MTLPackedFloat3Make(s.center.x - s.radius, s.center.y - s.radius, s.center.z - s.radius);
        bounds.max = MTLPackedFloat3Make(s.center.x + s.radius, s.center.y + s.radius, s.center.z + s.radius);
        sphereBoundsList.push_back(bounds);
    }
    id<MTLBuffer> boundingBoxBuffer = [device newBufferWithBytes:sphereBoundsList.data()
        length:sphereBoundsList.size() * sizeof(MTLAxisAlignedBoundingBox) options:MTLResourceStorageModeShared];

    MTLAccelerationStructureBoundingBoxGeometryDescriptor* bboxGeomDesc =
        [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
    bboxGeomDesc.boundingBoxBuffer = boundingBoxBuffer;
    bboxGeomDesc.boundingBoxStride = sizeof(MTLAxisAlignedBoundingBox);
    bboxGeomDesc.boundingBoxCount = (uint32_t)sphereBoundsList.size();
    // intersectionFunctionTableOffset here is this GEOMETRY's own
    // slot within whatever function table gets bound at trace time -
    // 0, sphereIntersectionFunction's own slot (set up below,
    // alongside the compute pipeline). The disk geometry added below
    // uses slot 1 instead - the first time this POC's function table
    // has needed more than one entry.
    bboxGeomDesc.intersectionFunctionTableOffset = 0;
    // Opaque here too: sphereIntersectionFunction is the REQUIRED
    // primitive-intersection test for this custom geometry (always
    // invoked, opaque or not - there's no hardware fallback for a
    // bounding-box primitive), so opaque just means "accept its
    // result directly," skip a second any-hit pass on top of it,
    // exactly this POC's one-test-decides-it shape.
    bboxGeomDesc.opaque = YES;

    // The disk's own bounding-box geometry, a SECOND geometryDescriptor
    // within the SAME primitive AS as the spheres (not a separate AS -
    // Metal supports multiple heterogeneous geometries in one
    // acceleration structure, distinguished at trace time by
    // `geometry_id`, matching this array's own index order: spheres
    // at 0, disk at 1 - see metal_poc.metal's own comment on why that
    // distinction is needed now). Padded uniformly by a small epsilon
    // in every axis (not just the disk's own zero-thickness normal
    // axis) - simplest bound that's correct regardless of which axis
    // the disk's normal happens to be aligned with, at the cost of a
    // slightly looser-than-optimal box for a single small primitive.
    const float diskBoundsEpsilon = 0.01f;
    std::vector<MTLAxisAlignedBoundingBox> diskBoundsList;
    for (const DiskData& d : disks) {
        float r = d.radius + diskBoundsEpsilon;
        MTLAxisAlignedBoundingBox bounds;
        bounds.min = MTLPackedFloat3Make(d.center.x - r, d.center.y - r, d.center.z - r);
        bounds.max = MTLPackedFloat3Make(d.center.x + r, d.center.y + r, d.center.z + r);
        diskBoundsList.push_back(bounds);
    }
    id<MTLBuffer> diskBoundingBoxBuffer = [device newBufferWithBytes:diskBoundsList.data()
        length:diskBoundsList.size() * sizeof(MTLAxisAlignedBoundingBox) options:MTLResourceStorageModeShared];

    MTLAccelerationStructureBoundingBoxGeometryDescriptor* diskGeomDesc =
        [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
    diskGeomDesc.boundingBoxBuffer = diskBoundingBoxBuffer;
    diskGeomDesc.boundingBoxStride = sizeof(MTLAxisAlignedBoundingBox);
    diskGeomDesc.boundingBoxCount = (uint32_t)diskBoundsList.size();
    diskGeomDesc.intersectionFunctionTableOffset = 1; // diskIntersectionFunction's own slot
    diskGeomDesc.opaque = YES;

    // The cylinder's own bounding-box geometry (section 171) - a THIRD
    // geometryDescriptor in this SAME primitive AS, slot 2. A world-
    // space-axis-aligned box around the tube's own finite extent: the
    // base/top endpoints each widened by `radius` in every axis (a
    // simple, always-correct-but-not-maximally-tight bound for an
    // arbitrarily-oriented tube, same "simplest correct box, not the
    // tightest one" choice diskGeomDesc's own epsilon-padding comment
    // already made for a disk).
    std::vector<MTLAxisAlignedBoundingBox> cylinderBoundsList;
    for (const CylinderData& cy : cylinders) {
        float3 base{cy.base.x, cy.base.y, cy.base.z};
        float3 top = base + float3{cy.axis.x, cy.axis.y, cy.axis.z} * cy.height;
        float3 lo = simd::min(base, top) - cy.radius;
        float3 hi = simd::max(base, top) + cy.radius;
        MTLAxisAlignedBoundingBox bounds;
        bounds.min = MTLPackedFloat3Make(lo.x, lo.y, lo.z);
        bounds.max = MTLPackedFloat3Make(hi.x, hi.y, hi.z);
        cylinderBoundsList.push_back(bounds);
    }
    // Same empty-vector nil-buffer pitfall as cylinderBuffer/
    // cylinderMaterialBuffer's own comment above (buildGPUResources()) -
    // guarded here too, since boundingBoxCount==0 below means this
    // buffer is never actually read regardless.
    id<MTLBuffer> cylinderBoundingBoxBuffer = cylinderBoundsList.empty()
        ? [device newBufferWithLength:sizeof(MTLAxisAlignedBoundingBox) options:MTLResourceStorageModeShared]
        : [device newBufferWithBytes:cylinderBoundsList.data()
              length:cylinderBoundsList.size() * sizeof(MTLAxisAlignedBoundingBox) options:MTLResourceStorageModeShared];

    MTLAccelerationStructureBoundingBoxGeometryDescriptor* cylinderGeomDesc =
        [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
    cylinderGeomDesc.boundingBoxBuffer = cylinderBoundingBoxBuffer;
    cylinderGeomDesc.boundingBoxStride = sizeof(MTLAxisAlignedBoundingBox);
    cylinderGeomDesc.boundingBoxCount = (uint32_t)cylinderBoundsList.size();
    cylinderGeomDesc.intersectionFunctionTableOffset = 2; // cylinderIntersectionFunction's own slot
    cylinderGeomDesc.opaque = YES;

    MTLPrimitiveAccelerationStructureDescriptor* sphereAccelDesc =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    sphereAccelDesc.geometryDescriptors = @[bboxGeomDesc, diskGeomDesc, cylinderGeomDesc];

    MTLAccelerationStructureSizes sphereSizes = [device accelerationStructureSizesWithDescriptor:sphereAccelDesc];
    sphereAS = [device newAccelerationStructureWithSize:sphereSizes.accelerationStructureSize];
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
        return false;
    }

    // --- Third primitive acceleration structure: Suzanne's own ------
    // geometry, built once, referenced by TWO different instances
    // below with two different transforms - unlike primAS/sphereAS
    // (each instanced exactly once, at identity), this is what
    // actually exercises instancing's whole point: reusing one GPU-
    // resident BVH from more than one world-space placement, rather
    // than building/storing the geometry twice.
    MTLAccelerationStructureTriangleGeometryDescriptor* suzanneGeomDesc =
        [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    suzanneGeomDesc.vertexBuffer = suzanneVertexBuffer;
    suzanneGeomDesc.vertexStride = sizeof(PackedFloat3);
    suzanneGeomDesc.triangleCount = suzanneTriangleCount;
    suzanneGeomDesc.opaque = YES;

    MTLPrimitiveAccelerationStructureDescriptor* suzanneAccelDesc =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    suzanneAccelDesc.geometryDescriptors = @[suzanneGeomDesc];

    MTLAccelerationStructureSizes suzanneSizes = [device accelerationStructureSizesWithDescriptor:suzanneAccelDesc];
    suzanneAS = [device newAccelerationStructureWithSize:suzanneSizes.accelerationStructureSize];
    id<MTLBuffer> suzanneScratch = [device newBufferWithLength:suzanneSizes.buildScratchBufferSize
        options:MTLResourceStorageModePrivate];

    id<MTLCommandBuffer> suzanneBuildCmd = [queue commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> suzanneBuildEnc = [suzanneBuildCmd accelerationStructureCommandEncoder];
    [suzanneBuildEnc buildAccelerationStructure:suzanneAS descriptor:suzanneAccelDesc scratchBuffer:suzanneScratch scratchBufferOffset:0];
    [suzanneBuildEnc endEncoding];
    [suzanneBuildCmd commit];
    [suzanneBuildCmd waitUntilCompleted];
    if (suzanneBuildCmd.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "Suzanne AS build failed: %s\n", suzanneBuildCmd.error.localizedDescription.UTF8String);
        return false;
    }

    // --- Instance acceleration structure: four instances over three -
    // primitive ASes (primAS/sphereAS each instanced once at
    // identity, suzanneAS instanced TWICE with different transforms -
    // see that AS's own comment). `addInstance()` builds one
    // MTLAccelerationStructureInstanceDescriptor AND its matching
    // InstanceTransform side-channel entry from the SAME
    // column/translation values in one place, so the two can't drift
    // out of sync with each other the way two independently-hand-
    // authored copies of the same transform could.
    std::vector<MTLAccelerationStructureInstanceDescriptor> instanceDescs;
    std::vector<InstanceTransform> instanceTransforms;
    auto addInstance = [&](uint32_t accelStructureIndex, float3 col0, float3 col1, float3 col2, float3 col3) {
        MTLAccelerationStructureInstanceDescriptor desc{};
        desc.accelerationStructureIndex = accelStructureIndex;
        desc.options = MTLAccelerationStructureInstanceOptionNone;
        desc.mask = 0xFF;
        desc.intersectionFunctionTableOffset = 0;
        desc.transformationMatrix.columns[0] = MTLPackedFloat3Make(col0.x, col0.y, col0.z);
        desc.transformationMatrix.columns[1] = MTLPackedFloat3Make(col1.x, col1.y, col1.z);
        desc.transformationMatrix.columns[2] = MTLPackedFloat3Make(col2.x, col2.y, col2.z);
        desc.transformationMatrix.columns[3] = MTLPackedFloat3Make(col3.x, col3.y, col3.z);
        instanceDescs.push_back(desc);
        instanceTransforms.push_back(InstanceTransform{
            PackedFloat3{col0.x, col0.y, col0.z}, PackedFloat3{col1.x, col1.y, col1.z},
            PackedFloat3{col2.x, col2.y, col2.z}, PackedFloat3{col3.x, col3.y, col3.z}});
    };

    const float3 identityCol0{1, 0, 0}, identityCol1{0, 1, 0}, identityCol2{0, 0, 1}, identityCol3{0, 0, 0};
    addInstance(0, identityCol0, identityCol1, identityCol2, identityCol3); // primAS (room + Spot)
    addInstance(1, identityCol0, identityCol1, identityCol2, identityCol3); // sphereAS

    // Suzanne instance A: translation only, at the same world position
    // the single non-instanced Suzanne used to sit at - an identity-
    // rotation instance is the direct continuation of every earlier
    // screenshot's own Suzanne placement.
    addInstance(2, identityCol0, identityCol1, identityCol2, float3{-0.05f, -0.55f, -0.3f});

    // Suzanne instance B: rotated 45 degrees about Y and scaled down
    // (uniform scale only - transformNormalByInstance()'s own
    // rigid-transform assumption over in metal_poc.metal stays valid
    // under a uniform scale, since normalize() cancels a uniform
    // factor exactly; it would NOT under a non-uniform one), placed
    // high near the back of the ceiling. A first attempt at a back-
    // left-corner floor placement (x=-0.7, z=-0.55) turned out to sit
    // along almost the same camera sightline as the gold sphere
    // (x/z ratio ~0.19 vs. the gold sphere's own ~0.2) and was
    // nearly fully hidden behind it - the exact same 2D-screen-space-
    // occlusion lesson Spot's own placement (step 12) and the rough
    // dielectric sphere's own placement (step 13) already ran into,
    // caught here the same way: render, look, reposition. The one
    // instance in this whole scene whose object-space normals
    // actually need transforming before shading - everywhere else,
    // an identity transform makes that transform a no-op.
    {
        const float theta = 0.785398f; // 45 degrees, radians
        const float s = 0.55f;
        float3 rotCol0{s * cosf(theta), 0.0f, -s * sinf(theta)};
        float3 rotCol1{0.0f, s, 0.0f};
        float3 rotCol2{s * sinf(theta), 0.0f, s * cosf(theta)};
        addInstance(2, rotCol0, rotCol1, rotCol2, float3{0.0f, 0.75f, -0.3f});
    }

    id<MTLBuffer> instanceBuffer = [device newBufferWithBytes:instanceDescs.data()
        length:instanceDescs.size() * sizeof(MTLAccelerationStructureInstanceDescriptor)
        options:MTLResourceStorageModeShared];
    instanceTransformBuffer = [device newBufferWithBytes:instanceTransforms.data()
        length:instanceTransforms.size() * sizeof(InstanceTransform)
        options:MTLResourceStorageModeShared];

    MTLInstanceAccelerationStructureDescriptor* instAccelDesc =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    instAccelDesc.instancedAccelerationStructures = @[primAS, sphereAS, suzanneAS];
    instAccelDesc.instanceCount = (uint32_t)instanceDescs.size();
    instAccelDesc.instanceDescriptorBuffer = instanceBuffer;

    MTLAccelerationStructureSizes instSizes = [device accelerationStructureSizesWithDescriptor:instAccelDesc];
    instAS = [device newAccelerationStructureWithSize:instSizes.accelerationStructureSize];
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
        return false;
    }
    return true;
}
