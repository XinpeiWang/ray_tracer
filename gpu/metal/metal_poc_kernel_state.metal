// metal_poc_kernel_state.metal
// The shared state of primaryRayKernel, which used to be one 1,631-line function whose locals everything read and wrote.
// It is now a handful of functions (metal_poc_kernel_camera / _media / _surface / _bounce.metal) that take these three bundles:
//
//   KernelRes    everything the kernel is bound to: its textures, buffers, the acceleration structure, the intersection function table,
//                the configured intersector. Read-only; built once at the top of the kernel.
//   PathState    what one sample's path carries from bounce to bounce: the ray, throughput, radiance, MIS bookkeeping, medium state,
//                random state, depth, the census record.
//   BounceState  what one bounce computes and passes along: the intersection, the medium flags, the hit (primitive, normal, material,
//                albedo...).
//
// The functions are pure code motion from the old kernel body. Each starts with the alias macros below, which give the bundle members
// back their old names (`rayOrigin`, `uniforms`, `lights`, `mat`...), so the bodies read exactly as they did when they were inline.
// MSL inlines them all; the render of every scene is bit-identical to the one before the split (see docs/METAL_KERNEL_STRUCTURE.md).
//
// A function that can end the path returns bool: false means "the path ends here" (what `break` meant in the old do { } while (false)
// bounce body), true means "carry on". This is the convention the shadeXxx() material functions already use.

// Divergence census probe (see Uniforms::debugCensus): packs the material/event code of each bounce of sample 0 into
// cen0/cen1 (4 codes of 8 bits each) and counts them in cenN. A no-op unless the census is on.
#define CENSUS_PUSH(code) if (cenRec) { uint c_ = (uint(code)) & 0xFFu; if (cenN < 4u) cen0 |= c_ << (8u * cenN); else if (cenN < 8u) cen1 |= c_ << (8u * (cenN - 4u)); ++cenN; }

struct KernelRes {
    texture2d<float, access::sample> earthTexture;
    texture2d<float, access::sample> goniometricTexture;
    texture2d<float, access::sample> pbrtEnvTexture;
    texture2d<float, access::sample> pbrtGoniometricTexture;
    texture2d<float, access::sample> pbrtProjectionTexture;
    texture2d<float, access::sample> pbrtAreaLightTexture;
    texture2d<float, access::sample> pbrtDiffuseTexture;
    texture2d<float, access::sample> pbrtTransmitTexture;
    instance_acceleration_structure accelStructure;
    constant Uniforms* uniformsPtr;
    device const TriangleMaterial* triMaterials;
    device const packed_float3* vertices;
    device const TriangleMaterial* sphereMaterials;
    device const SphereData* spheres;
    intersection_function_table<instancing, triangle_data> functionTable;
    device const packed_float3* normals;
    device const packed_float2* uvs;
    device const AreaLight* lights;
    device const packed_float3* suzanneNormals;
    device const TriangleMaterial* suzanneMaterials;
    device const InstanceTransform* instanceTransforms;
    device const DiskData* disks;
    device const TriangleMaterial* diskMaterials;
    device const PointLight* pointLights;
    device const DirectionalLight* directionalLights;
    device const ProjectionLight* projectionLights;
    device const GoniometricLight* goniometricLights;
    device const float* envMarginalCDF;
    device const float* envConditionalCDF;
    device const float* ggxEnergyTable;
    device const float* pbrtEnvMarginalCDF;
    device const float* pbrtEnvConditionalCDF;
    device const LensElement* lensElements;
    device const ExitPupilBounds* exitPupilBounds;
    device const CylinderData* cylinders;
    device const TriangleMaterial* cylinderMaterials;
    device const GpuCloudMedium* cloudMediums;
    device const GpuRgbGridMedium* rgbGridMediums;
    device const float* rgbGridData;
    intersector<instancing, triangle_data> isect;
};

struct PathState {
    uint rngState = 0u;
    MetalFilterSample filterSample;
    float shutterT = 0.0;
    float3 rayOrigin = float3(0.0);
    float3 rayDir = float3(0.0);
    float3 throughput = float3(0.0);
    float3 radiance = float3(0.0);
    bool specularBounce = true;
    float bsdfPdf = 0.0;
    float mediumSkippedDist = 0.0;
    bool pathTouchedHair = false;
    bool envLightSampled = false;   // a cloud scatter already sampled the constant environment light for the current ray (see shadeEscapedRay)
    bool inGlass = false;
    float3 glassSigmaT3 = float3(0.0);
    float glassG = 0.0;
    float3 glassAlbedo = float3(1.0);
    int glassChan = -1;
    int fogChan = -1;
    uint rgbChannel = kRgbChannelUnset;
    uint cen0 = 0u;
    uint cen1 = 0u;
    uint cenN = 0u;
    bool cenRec = false;
    uint depth = 0;
};

struct BounceState {
    bool lastBounce = false;
    SpherePayload spherePayload = SpherePayload{0.0, false};
    SpherePayload shadowSpherePayload = SpherePayload{0.0, true};
    intersection_result<instancing, triangle_data> result;
    float3 mediumOriginBefore = float3(0.0);
    bool scatteredInMedium = false;
    bool passedThroughMediumSphere = false;
    uint primId = 0u;
    float3 hitPoint = float3(0.0);
    bool isBoundingBox = false;
    bool isDisk = false;
    bool isCylinder = false;
    bool isSphere = false;
    bool isTriangleHit = false;
    bool isPbrtInstance = false;
    bool isSuzanneInstance = false;
    float3 normal = float3(0.0);
    TriangleMaterial mat;
    bool frontFace = false;
    float3 facingNormal = float3(0.0);
    float3 albedo = float3(0.0);
};

// Gives the members of a bundle their old local names. Unused ones cost nothing (the compiler drops them).
#define KERNEL_RES_ALIASES(R) \
    texture2d<float, access::sample> earthTexture = (R).earthTexture; \
    texture2d<float, access::sample> goniometricTexture = (R).goniometricTexture; \
    texture2d<float, access::sample> pbrtEnvTexture = (R).pbrtEnvTexture; \
    texture2d<float, access::sample> pbrtGoniometricTexture = (R).pbrtGoniometricTexture; \
    texture2d<float, access::sample> pbrtProjectionTexture = (R).pbrtProjectionTexture; \
    texture2d<float, access::sample> pbrtAreaLightTexture = (R).pbrtAreaLightTexture; \
    texture2d<float, access::sample> pbrtDiffuseTexture = (R).pbrtDiffuseTexture; \
    texture2d<float, access::sample> pbrtTransmitTexture = (R).pbrtTransmitTexture; \
    instance_acceleration_structure accelStructure = (R).accelStructure; \
    constant Uniforms& uniforms = *(R).uniformsPtr; \
    device const TriangleMaterial* triMaterials = (R).triMaterials; \
    device const packed_float3* vertices = (R).vertices; \
    device const TriangleMaterial* sphereMaterials = (R).sphereMaterials; \
    device const SphereData* spheres = (R).spheres; \
    intersection_function_table<instancing, triangle_data> functionTable = (R).functionTable; \
    device const packed_float3* normals = (R).normals; \
    device const packed_float2* uvs = (R).uvs; \
    device const AreaLight* lights = (R).lights; \
    device const packed_float3* suzanneNormals = (R).suzanneNormals; \
    device const TriangleMaterial* suzanneMaterials = (R).suzanneMaterials; \
    device const InstanceTransform* instanceTransforms = (R).instanceTransforms; \
    device const DiskData* disks = (R).disks; \
    device const TriangleMaterial* diskMaterials = (R).diskMaterials; \
    device const PointLight* pointLights = (R).pointLights; \
    device const DirectionalLight* directionalLights = (R).directionalLights; \
    device const ProjectionLight* projectionLights = (R).projectionLights; \
    device const GoniometricLight* goniometricLights = (R).goniometricLights; \
    device const float* envMarginalCDF = (R).envMarginalCDF; \
    device const float* envConditionalCDF = (R).envConditionalCDF; \
    device const float* ggxEnergyTable = (R).ggxEnergyTable; \
    device const float* pbrtEnvMarginalCDF = (R).pbrtEnvMarginalCDF; \
    device const float* pbrtEnvConditionalCDF = (R).pbrtEnvConditionalCDF; \
    device const LensElement* lensElements = (R).lensElements; \
    device const ExitPupilBounds* exitPupilBounds = (R).exitPupilBounds; \
    device const CylinderData* cylinders = (R).cylinders; \
    device const TriangleMaterial* cylinderMaterials = (R).cylinderMaterials; \
    device const GpuCloudMedium* cloudMediums = (R).cloudMediums; \
    device const GpuRgbGridMedium* rgbGridMediums = (R).rgbGridMediums; \
    device const float* rgbGridData = (R).rgbGridData; \
    intersector<instancing, triangle_data> isect = (R).isect; \
    constexpr sampler textureSampler(coord::normalized, address::repeat, filter::linear);

#define PATH_STATE_ALIASES(P) \
    thread uint& rngState = (P).rngState; \
    thread MetalFilterSample& filterSample = (P).filterSample; \
    thread float& shutterT = (P).shutterT; \
    thread float3& rayOrigin = (P).rayOrigin; \
    thread float3& rayDir = (P).rayDir; \
    thread float3& throughput = (P).throughput; \
    thread float3& radiance = (P).radiance; \
    thread bool& specularBounce = (P).specularBounce; \
    thread float& bsdfPdf = (P).bsdfPdf; \
    thread float& mediumSkippedDist = (P).mediumSkippedDist; \
    thread bool& pathTouchedHair = (P).pathTouchedHair; \
    thread bool& envLightSampled = (P).envLightSampled; \
    thread bool& inGlass = (P).inGlass; \
    thread float3& glassSigmaT3 = (P).glassSigmaT3; \
    thread float& glassG = (P).glassG; \
    thread float3& glassAlbedo = (P).glassAlbedo; \
    thread int& glassChan = (P).glassChan; \
    thread int& fogChan = (P).fogChan; \
    thread uint& rgbChannel = (P).rgbChannel; \
    thread uint& cen0 = (P).cen0; \
    thread uint& cen1 = (P).cen1; \
    thread uint& cenN = (P).cenN; \
    thread bool& cenRec = (P).cenRec; \
    thread uint& depth = (P).depth;

#define BOUNCE_STATE_ALIASES(B) \
    thread bool& lastBounce = (B).lastBounce; \
    thread SpherePayload& spherePayload = (B).spherePayload; \
    thread SpherePayload& shadowSpherePayload = (B).shadowSpherePayload; \
    thread intersection_result<instancing, triangle_data>& result = (B).result; \
    thread float3& mediumOriginBefore = (B).mediumOriginBefore; \
    thread bool& scatteredInMedium = (B).scatteredInMedium; \
    thread bool& passedThroughMediumSphere = (B).passedThroughMediumSphere; \
    thread uint& primId = (B).primId; \
    thread float3& hitPoint = (B).hitPoint; \
    thread bool& isBoundingBox = (B).isBoundingBox; \
    thread bool& isDisk = (B).isDisk; \
    thread bool& isCylinder = (B).isCylinder; \
    thread bool& isSphere = (B).isSphere; \
    thread bool& isTriangleHit = (B).isTriangleHit; \
    thread bool& isPbrtInstance = (B).isPbrtInstance; \
    thread bool& isSuzanneInstance = (B).isSuzanneInstance; \
    thread float3& normal = (B).normal; \
    thread TriangleMaterial& mat = (B).mat; \
    thread bool& frontFace = (B).frontFace; \
    thread float3& facingNormal = (B).facingNormal; \
    thread float3& albedo = (B).albedo;
