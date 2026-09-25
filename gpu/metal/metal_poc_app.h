// metal_poc_app.h
// Shared types, host-side helper functions, and the full MetalPocApp
// class declaration used by metal_poc.mm (core app/GPU-resource
// plumbing), every metal_poc_scenes_*.mm file (one per hand-authored
// scene category), and metal_poc_pbrt_loader.mm (real pbrt scene
// loading). Split out of metal_poc.mm as a pure code-motion refactor
// (no behaviour change) once that file's own scene-builder count made
// it unwieldy - see docs/METAL_GPU_FEASIBILITY.md.
//
// Free functions below were `static` in their original single-TU home;
// each is `inline` here instead (identical body, just ODR-safe to
// define in a header multiple translation units now include) - the
// ONLY mechanical change this split made to any of them.
#pragma once

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstring>
#include <functional>
#include <mach-o/dyld.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <simd/simd.h>

#include "metal_poc_host_math.h"
#include "../../src/shared/pbrt_load.h"
#include "../../src/shared/cornell_box_data.h"
#include "../../src/shared/conductor_data.h"
// RealisticCamera<T> (D4/D8, section 157) - a portable, host-only
// precompute class (CPU_GPU-tagged but its own constructor/exit-pupil-
// bounding never runs device-side anywhere in this project), the EXACT
// same class gpu/optix/scene_builder.cpp already instantiates directly
// on its own host side to precompute a GPU-portable lens/exit-pupil
// table - see Uniforms::cameraRealistic's own comment.
#include "../../src/shared/realistic_camera.h"

#include "metal_poc_gpu_types.h"
#include "metal_poc_scene_helpers.h"

// main()'s own body used to be one ~1020-line function carrying roughly
// thirty local variables from CLI parsing all the way through to the
// final PNG write - device/queue, every scene-data vector, every GPU
// buffer, every acceleration structure, the linear HDR pixel buffer -
// with no natural place to split it without threading most of that state
// through explicit parameters at each new boundary. `MetalPocApp` turns
// those locals into members instead: every method below relocates a
// contiguous span of the ORIGINAL main() body largely unchanged
// (extracted mechanically via `sed`, not retyped from scratch, so
// `verts`/`device`/`pixels`/etc. keep resolving exactly the way they did
// as bare local variables, via implicit `this->`) - EXCEPT each such
// name's own FIRST declaration in the original code (e.g. `id<MTLBuffer>
// vertexBuffer = ...`) had its type annotation stripped down to a plain
// assignment (`vertexBuffer = ...`), and every no-initializer declaration
// (e.g. `std::vector<PackedFloat3> verts;`) was deleted outright - both
// real, necessary fixes, not cosmetic ones: left as originally written,
// each would have silently REDECLARED a same-named LOCAL that shadows
// the member of the same name for the rest of that one function, leaving
// the actual member permanently nil/empty for every OTHER method to read
// - caught by an actual crash (`buildGPUResources()`'s own instance-AS
// setup dereferencing a still-nil `primAS`) and a silently-wrong CLI
// default (width/height/outPath staying at their compiled-in defaults
// regardless of argv) during this refactor's own verification, not
// assumed safe from the mechanical extraction alone. A plain struct, not
// a class with any encapsulation of its own - every member public,
// matching this POC's existing preference for direct scene-authoring
// code over machinery it has no use for.

struct MetalPocApp {
    // --- Parsed CLI config, set by parseArgsAndCreateDevice() ----------
    uint32_t width = 400;
    uint32_t height = 400;
    const char* outPath = "/tmp/metal_poc_render.png";
    ToneMapMode toneMapMode = ToneMapMode::ACES;
    // Flat multiplier on linear colour, applied right before tone-
    // mapping (postProcessAndWrite()'s own comment) - RenderOptions::
    // exposure's own doc comment, matching cpu_render_main()/
    // optix_render_main()'s identical semantics. NOT parsed from the
    // positional argv[] array parseArgsAndCreateDevice() reads (that
    // 9-slot shape is shared with the standalone `metal_poc` CLI binary,
    // metal_poc_main.mm - adding a new slot there would shift every
    // later index for that separate entry point too) - metal_render_main()
    // pokes this field directly instead, the same "set a field metal_render_main()
    // itself needs, argv doesn't carry" shape force_camera_override's own
    // applyCameraOverride() call already uses. 1.0 (default) is a no-op,
    // unaffected for every scene/caller that never touches this field.
    float exposureValue = 1.0f;
    // Skips buildScene()'s own hardcoded demo room lights (sun/fill/area/
    // projection/goniometric) when set - RenderOptions::isolate_pbrt_lighting's
    // own doc comment for the full "why" (section 197's C9 finding).
    // Same "metal_render_main() pokes this field directly, argv doesn't
    // carry it" shape as exposureValue above. False (default) is a no-op,
    // unaffected for every scene/caller that never touches this field.
    bool isolatePbrtLighting = false;
    // Optional 7th positional CLI arg - a real .pbrt scene file to load
    // via src/shared/pbrt_load.h INSTEAD of buildScene()'s own hardcoded
    // room (see loadPbrtScene()'s own comment for exactly what subset of
    // pbrt this v1 supports). Empty (the default) keeps every existing
    // CLI invocation's behavior identical to before this existed.
    std::string pbrtScenePath;
    // Optional 9th positional CLI arg (see parseArgsAndCreateDevice()'s own
    // argv[8] handling) - a scene_id one of buildHandAuthoredScene()'s own
    // real cases covers (section 116, docs/METAL_GPU_FEASIBILITY.md), for a
    // scene with NO pbrt file backing it at all. Mutually exclusive with
    // pbrtScenePath in practice - metal_render_main() only ever sets one of
    // the two, matching cpu_scene_pbrt_path_by_id() either resolving a real
    // path (pbrtScenePath) or not (this one, only if cpu_scene_metal_hand_
    // authored_supported() says so).
    std::string handAuthoredSceneId;
    // Set by loadPbrtScene() when pbrtScenePath was given and loaded
    // successfully - buildScene()'s camera-setup call in
    // compileShaderAndDispatch() reads these instead of its own hardcoded
    // literals whenever this is true.
    bool havePbrtCamera = false;
    float3 pbrtCameraPos{0, 0, 0};
    float3 pbrtCameraForward{0, 0, -1};
    float3 pbrtCameraRight{1, 0, 0};
    float3 pbrtCameraUp{0, 1, 0};
    float pbrtTanHalfFov = 1.0f;
    // The scene's own lookat point (world/transformed space) and raw
    // (untransformed - it's a direction) up vector, plus the bbox-rescale
    // transform loadPbrtScene() derived - all saved so applyCameraOverride()
    // can recompute pbrtCameraPos/Forward/Right/Up for a caller-supplied
    // lookfrom in the scene's OWN pbrt-file coordinate space, the same
    // space cpu_scene_recommended_camera()/cam_x/y/z already use for every
    // other backend, without needing loadPbrtScene() to run again.
    float3 pbrtCameraLookAtWorld{0, 0, 0};
    float3 pbrtCameraUpRaw{0, 1, 0};
    // Thin-lens DOF for a hand-authored scene (D5, section 137) - both
    // already in this SCENE's own rescaled/offset unit system (a caller
    // computes them from the scene's own real defocus_angle/focus_dist
    // the same way camera.h's own `defocus_radius = focus_dist *
    // tan(defocus_angle/2)` does, then multiplies both by sceneScale).
    // Defaults (0/1) preserve every earlier hand-authored scene's own
    // "no DOF" behaviour exactly - this is a genuinely SEPARATE gap from
    // this struct's own pre-existing "pbrt v1 doesn't parse the pbrt
    // file's own lensradius/focaldistance Camera parameters" limitation
    // (metal_render_main()'s own comment) - unrelated, a hand-authored
    // scene has no pbrt file to parse from at all.
    float pbrtLensRadius = 0.0f;
    float pbrtFocusDistance = 1.0f;
    // Camera shutter motion blur for a hand-authored scene (D13, section
    // 174) - same shape as pbrtLensRadius/pbrtFocusDistance immediately
    // above: a scene builder sets this (already in this scene's own
    // rescaled unit system, sceneScale-multiplied like every other
    // world-space delta here), and compileShaderAndDispatch() reads it
    // into uniforms.cameraVelocity instead of its own previous
    // unconditional zero for every `havePbrtCamera` scene. Default
    // {0,0,0} preserves every OTHER scene's own "no camera motion blur"
    // behaviour exactly - a provable no-op, same as SphereData::
    // centerDelta1's own identical default-preserving shape (F11,
    // section 167).
    PackedFloat3 sceneCameraVelocity{0, 0, 0};
    // Orthographic (parallel-projection) camera for a hand-authored
    // scene (D2/D6, section 150) - mirrors pbrtLensRadius/
    // pbrtFocusDistance's own shape immediately above: a scene builder
    // sets this true and reuses pbrtTanHalfFov as the orthographic
    // screen window's own world-space half-extent (already in this
    // scene's own rescaled unit system, same sceneScale-multiplied
    // convention every other world-space distance here uses) - see
    // Uniforms::cameraOrthographic's own comment (metal_poc.metal) for
    // the full ray-generation mechanism. Default false preserves every
    // earlier hand-authored scene's own perspective-camera behaviour
    // exactly.
    bool havePbrtOrthographic = false;
    // Spherical (360-degree equirectangular panorama) camera for a
    // hand-authored scene (D7, section 152) - mirrors
    // havePbrtOrthographic's own shape immediately above. No extra
    // world-space parameter needed (unlike orthographic's own
    // pbrtTanHalfFov reuse) - a panoramic camera has no screen window
    // or FOV at all, only a position and orientation, both already
    // carried by pbrtCameraPos/Forward/Right/Up. See
    // Uniforms::cameraSpherical's own comment (metal_poc.metal) for the
    // full ray-generation mechanism. Default false preserves every
    // earlier hand-authored scene's own behaviour exactly.
    bool havePbrtSpherical = false;
    // Which mapping havePbrtSpherical's own camera uses - see
    // Uniforms::sphericalMappingEqualArea's own comment
    // (metal_poc_types.metal). Only ever set true by loadPbrtScene()
    // itself (a real pbrt file's own Camera "spherical" "string
    // mapping" ["equalarea"], D11/section 159) - no hand-authored
    // scene (D3/D7) requests it, so this stays false for those,
    // matching their own already-shipped EquiRectangular look exactly.
    bool havePbrtSphericalEqualArea = false;
    float3 pbrtBboxCenter{0, 0, 0};
    float pbrtSceneScale = 1.0f;
    float3 pbrtSceneOffset{0, 0, 0};

    // Set by loadPbrtScene() when the scene has a valid homogeneous
    // camera medium (scene.cameraMediumIndex, already validated by
    // pbrt_flatten.h's own resolution pass - see that field's own
    // comment) - read instead of the hardcoded fog defaults in
    // compileShaderAndDispatch()'s own havePbrtCamera override block.
    bool havePbrtMedium = false;
    float pbrtFogSigmaT = 0.0f;
    float3 pbrtFogAlbedo{1, 1, 1};
    float pbrtFogAsymmetryG = 0.0f;

    // Set by loadPbrtScene() for a constant-colour (no image)
    // LightSource "infinite" - see that function's own comment and
    // metal_poc.metal's own mirrored one on why this is miss-path-only,
    // no NEE/MIS strategy.
    bool havePbrtConstantEnvLight = false;
    float3 pbrtEnvColor{0, 0, 0};

    // Set by loadPbrtScene() for an image-based LightSource "infinite" -
    // the decoded pixels are already linear float RGB, row-major
    // (pbrt_load::loadFile()'s own resolution of InfiniteLight::
    // imagePixels - no filesystem/decode work needed on this side).
    // Mutually exclusive with havePbrtConstantEnvLight above.
    bool havePbrtImageEnvLight = false;
    std::vector<float> pbrtEnvImagePixels;
    int pbrtEnvImageWidth = 0;
    int pbrtEnvImageHeight = 0;

    // Set by loadPbrtScene() for the FIRST pbrt-loaded goniometric/
    // projection light that names a real profile/slide image and
    // successfully decodes (section 98) - unlike infinite light's own
    // imagePixels above, PunctualLight::filename is NOT pre-resolved/
    // decoded by pbrt_load.h (see that struct's own comment: only
    // InfiniteLight gets that treatment), so this loader does its own
    // pbrt_load::loadFileNear() + pbrt_load::detail::
    // decodeInfiniteLightImage() call - reusing that decoder rather than
    // writing a second one, since a goniometric/projection image is
    // decoded exactly the same way (extension-dispatched, EXR via
    // tinyexr or stb_image's float loader for everything else). Only ONE
    // slot per kind, mirroring this POC's existing "one shared texture"
    // architecture (goniometricTexture/earthTexture, section 57/91) -
    // a SECOND image-based light of the same kind in one scene is warned
    // and falls back to the Approx case (section 91/92), same as an
    // unsupported light kind entirely. Row-major linear float RGB, same
    // layout as pbrtEnvImagePixels above.
    bool havePbrtGoniometricImage = false;
    std::vector<float> pbrtGoniometricImagePixels;
    int pbrtGoniometricImageWidth = 0;
    int pbrtGoniometricImageHeight = 0;

    bool havePbrtProjectionImage = false;
    std::vector<float> pbrtProjectionImagePixels;
    int pbrtProjectionImageWidth = 0;
    int pbrtProjectionImageHeight = 0;

    // A pbrt-loaded scene's own image-based AreaLightSource
    // ("string filename", section 105) - same "one shared slot, first
    // light wins" constraint as the goniometric/projection images
    // above, and QUAD-shaped lights only (see AreaLight::useTexture's
    // own comment for why a disk-shaped one still falls back to flat L).
    bool havePbrtAreaLightImage = false;
    std::vector<float> pbrtAreaLightImagePixels;
    int pbrtAreaLightImageWidth = 0;
    int pbrtAreaLightImageHeight = 0;

    // A pbrt-loaded scene's own Diffuse/CoatedDiffuse material with a
    // "texture reflectance" bound to a bare "imagemap" Texture (F5/F9,
    // section 166) - same "one shared slot, first material wins"
    // constraint as every other pbrt-loaded image above.
    bool havePbrtDiffuseImage = false;
    std::vector<float> pbrtDiffuseImagePixels;
    int pbrtDiffuseImageWidth = 0;
    int pbrtDiffuseImageHeight = 0;
    // The filename actually loaded into the fields above - lets a
    // SECOND triangle sharing the SAME already-loaded textured material
    // (e.g. a 2-triangle quad) correctly get materialType 26 too,
    // distinguishing "the same material's own second triangle" from "a
    // genuinely different second texture" (metal_poc.mm's own
    // MaterialKind::Diffuse case comment has the full story).
    std::string pbrtDiffuseImageFilename;

    // --- Metal device/queue, set by parseArgsAndCreateDevice() ---------
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;

    // --- Host-side scene data, built by buildScene() --------------------
    std::vector<PackedFloat3> verts, normals;
    std::vector<PackedFloat2> uvs;
    std::vector<TriangleMaterial> materials;
    std::vector<PackedFloat3> suzanneVerts, suzanneNormals;
    std::vector<PackedFloat2> suzanneUVs;
    std::vector<TriangleMaterial> suzanneMaterials;
    std::vector<AreaLightData> lights;
    std::vector<SphereData> spheres;
    std::vector<TriangleMaterial> sphereMaterials;
    std::vector<DiskData> disks;
    std::vector<TriangleMaterial> diskMaterials;
    std::vector<CylinderData> cylinders;
    std::vector<TriangleMaterial> cylinderMaterials;
    // E2's own backing buffer for materialType 29 (GpuCloudMedium's own
    // comment) - empty for every scene but E2.
    std::vector<GpuCloudMedium> cloudMediums;
    // E4's own backing buffers for materialType 30 (GpuRgbGridMedium's
    // own comment) - both empty for every scene but E4. `rgbGridData` is
    // the flat, concatenated R-then-G-then-B voxel data every
    // `rgbGridMediums` entry's own `dataOffset` indexes into.
    std::vector<GpuRgbGridMedium> rgbGridMediums;
    std::vector<float> rgbGridData;
    std::vector<PointLightData> pointLights;
    std::vector<DirectionalLightData> directionalLights;
    std::vector<ProjectionLightData> projectionLights;
    std::vector<GoniometricLightData> goniometricLights;
    // The goniometric light's own procedurally-generated intensity image
    // (see buildGoniometricProfileImage()'s own comment) - built by
    // buildScene() alongside `goniometricLights`, uploaded as a real
    // MTLTexture by buildGPUResources() (needs `device`, not available
    // yet inside buildScene()).
    std::vector<uint8_t> goniometricImage;
    int goniometricImageSize = 0;
    uint32_t triangleCount = 0;
    // Realistic (multi-element-lens) camera (D4/D8, section 157) - built
    // by a hand-authored scene's own builder (metal_poc_scenes_d.mm),
    // directly reading a REAL, host-instantiated `RealisticCamera<float>`'s
    // own GPU-port accessors - see Uniforms::cameraRealistic's own
    // comment for the full mechanism.
    bool havePbrtRealisticCamera = false;
    std::vector<GpuLensElementData> realisticLensElements;
    std::vector<GpuExitPupilBoundsData> realisticExitPupilBounds;
    float realisticFilmHalfX = 0.0f, realisticFilmHalfY = 0.0f, realisticLensRearZ = 0.0f;

    // --- GPU-resident buffers + acceleration structures, built by
    // buildGPUResources() ------------------------------------------------
    id<MTLBuffer> vertexBuffer, normalBuffer, uvBuffer, lightBuffer;
    id<MTLBuffer> pointLightBuffer, directionalLightBuffer, projectionLightBuffer;
    id<MTLBuffer> goniometricLightBuffer;
    id<MTLBuffer> materialBuffer, sphereBuffer, sphereMaterialBuffer;
    id<MTLBuffer> diskBuffer, diskMaterialBuffer;
    id<MTLBuffer> cylinderBuffer, cylinderMaterialBuffer;
    id<MTLBuffer> cloudMediumBuffer;
    id<MTLBuffer> rgbGridMediumBuffer, rgbGridDataBuffer;
    id<MTLBuffer> suzanneVertexBuffer, suzanneNormalBuffer, suzanneMaterialBuffer;
    id<MTLBuffer> instanceTransformBuffer;
    id<MTLBuffer> lensElementBuffer, exitPupilBoundsBuffer;
    id<MTLAccelerationStructure> primAS, sphereAS, suzanneAS, instAS;
    id<MTLTexture> goniometricTexture = nil;

    // --- Render output: written by compileShaderAndDispatch(), read by
    // postProcessAndWrite() ----------------------------------------------
    std::vector<float> pixels;

    bool parseArgsAndCreateDevice(int argc, const char** argv);
    void buildScene();
    // See its own comment (defined just above buildScene()) - loads a
    // real .pbrt file's geometry/materials/lights/camera into the SAME
    // vectors buildScene()'s own hardcoded room uses. Only called from
    // buildScene() itself, when pbrtScenePath is non-empty.
    void loadPbrtScene();
    // The 9 phases loadPbrtScene() itself is now just a thin dispatcher
    // over (section 109 - split from one ~950-line function, mirroring
    // the SAME "monolithic function -> named phase methods" refactor
    // main()/PR #56 and primaryRayKernel/PR #65 already went through
    // once each, at a comparable size). Every phase takes the shared
    // read-only state it needs as EXPLICIT parameters (never a NEW
    // class member) - toWorld/materialFor as std::function, matching
    // this file's own already-established idiom for mapMaterial -
    // specifically to avoid PR #56's own documented bug class (a
    // local's type annotation left in place after converting it to a
    // member, silently redeclaring a same-named shadowing local): there
    // is nothing here to accidentally redeclare, since none of this
    // shared state becomes a member at all. Every phase writes its own
    // results straight into the ALREADY-existing MetalPocApp members
    // (spheres/disks/lights/pointLights/...) exactly as the single
    // monolithic function used to - only the CALLING convention changed,
    // not where any result actually lives.
    using PbrtToWorldFn = std::function<float3(float3)>;
    using PbrtMaterialForFn = std::function<TriangleMaterial(int)>;
    // Area lights ("single quad, 2 triangles" shape only) - populates
    // `lights`, and the two out-params the very next phase
    // (loadPbrtRemainingTriangles) needs: which triangles this phase
    // already consumed, and any light emission a too-complex (non-quad)
    // light still needs applied directly to its own triangles' material.
    void loadPbrtAreaLights(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld,
        const PbrtMaterialForFn& materialFor, std::vector<bool>& triangleHandled,
        std::unordered_map<int, std::pair<float3, bool>>& unhandledLightEmission);
    // Every triangle loadPbrtAreaLights() didn't already consume as a
    // light's own quad - ordinary geometry, or an unhandled (non-quad)
    // light's own triangles (emissive via `unhandledLightEmission`, not
    // NEE-registered).
    void loadPbrtRemainingTriangles(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld,
        const PbrtMaterialForFn& materialFor, const std::vector<bool>& triangleHandled,
        const std::unordered_map<int, std::pair<float3, bool>>& unhandledLightEmission);
    void loadPbrtSpheres(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld,
        const PbrtMaterialForFn& materialFor, float sceneScale);
    void loadPbrtDisks(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld,
        const PbrtMaterialForFn& materialFor, float sceneScale);
    void loadPbrtCylinders(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld,
        const PbrtMaterialForFn& materialFor, float sceneScale);
    void loadPbrtObjectInstances(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld,
        const PbrtMaterialForFn& materialFor);
    void loadPbrtPunctualLights(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld, float sceneScale);
    void loadPbrtMedium(const pbrt_flatten::FlatScene& scene, float sceneScale);
    void loadPbrtInfiniteLight(const pbrt_flatten::FlatScene& scene);
    void loadPbrtCamera(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld,
        float3 bboxCenter, float sceneScale, float3 sceneOffset);
    // A HAND-AUTHORED (no pbrt file at all) scene, dispatched by scene_id -
    // see cpu_scene_metal_hand_authored_supported()'s own comment
    // (cpu_interface.h) for the one canonical "which ids does this cover"
    // list. Populates the exact same members loadPbrtScene() does
    // (verts/materials/spheres/lights/camera/...), via the SAME shared,
    // backend-agnostic data headers (e.g. src/shared/cornell_box_data.h)
    // CPU's/OptiX's own hand-authored builders already read, rather than
    // re-deriving geometry numbers by hand - see section 116, docs/
    // METAL_GPU_FEASIBILITY.md. Returns false (and prints why) for a
    // scene_id this function doesn't have a real case for yet - callers
    // should not have reached here for one at all (metal_render_main()'s
    // own gate already checked cpu_scene_metal_hand_authored_supported()
    // first), so this is a safety net, not the primary guard.
    bool buildHandAuthoredScene(const std::string& scene_id);
    void buildCornellBoxA1();
    // A8: Cornell Smoke - A1's own 5 walls plus two bounded medium
    // spheres approximating CPU's own two rotated smoke boxes. See this
    // method's own definition comment (metal_poc_scenes_a.mm) for the
    // full "why," including materialType 28's own new mechanism.
    // Section 176, docs/METAL_GPU_FEASIBILITY.md.
    void buildCornellSmoke();
    // Shared "mesh gallery" pattern (category G/Models - section 117, docs/
    // METAL_GPU_FEASIBILITY.md) - a flat ground quad, one imported OBJ mesh
    // in a caller-chosen material, and one small quad area light above,
    // mirroring gpu/optix/scene_builder_mesh_gallery.h's own repeated
    // "ground + mesh + light" shape (its own comment: ~50 near-identical
    // scenes) with two deliberate simplifications: a flat quad ground, not
    // a huge checker SPHERE (metal_poc.metal's sphere-intersection path has
    // no UV computation checkerColor() could read at all); a small quad
    // light, not a sphere light (this loader's only NEE-sampled light
    // representation, AreaLightData, is an analytic QUAD - a sphere light
    // would need genuinely new NEE-sampling code, out of scope for this
    // increment). objFilename is resolved the SAME executableDir()-first,
    // RT_MODELS_DIR-fallback way loadObjMesh()'s own existing Suzanne/Spot
    // callers already do (section 110's own fix covers this automatically -
    // no new path-resolution code needed here).
    void buildMeshGalleryScene(const std::string& objFilename, float3 meshColor,
        uint32_t meshMaterialType, float meshRoughness, float3 meshConductorEta,
        float3 meshConductorK, float meshTargetSize,
        // meshIor: materialType==2 (dielectric) only, e.g. Glass Dragon
        // (G13). flipXZ: loadObjMesh()'s own 180-degree-about-Y flip, for
        // a mesh authored facing away from this app's own camera
        // convention (G7 Spot Cow, G10 Horse) - section 119, docs/
        // METAL_GPU_FEASIBILITY.md. Both trailing-defaulted so G1-G24's
        // own existing calls are untouched.
        float meshIor = 1.0f, bool flipXZ = false);
    void buildStanfordBunny();
    void buildStanfordArmadillo();
    void buildStanfordHappyBuddha();
    // G4-G24 (minus G7/G10 - need a 180-degree mesh flip buildMeshGallery
    // Scene() doesn't support yet; G12 - four meshes, not one; G13 -
    // dielectric, not conductor - all deferred to a later increment,
    // section 118, docs/METAL_GPU_FEASIBILITY.md). Each is a one-line
    // buildMeshGalleryScene() call with that scene's own real OptiX
    // albedo/roughness (gpu/optix/scene_builder_mesh_gallery.h) converted
    // the same reflectanceToConductorK() way G1-G3 already are.
    void buildStanfordLucy();
    void buildStanfordDragon();
    void buildUtahTeapot();
    void buildSuzanneGallery();
    void buildNefertiti();
    void buildCheburashka();
    void buildBeast();
    void buildVWBeetle();
    void buildBimba();
    void buildCowGallery();
    void buildFandisk();
    void buildHomer();
    void buildIgea();
    void buildMaxPlanck();
    void buildOgre();
    void buildRockerArm();
    // G7/G10/G13 (section 119) - the last 3 single-mesh gallery scenes
    // buildMeshGalleryScene()'s now-extended signature (flipXZ/meshIor)
    // can cover. G12 (Trophy Room, four meshes) stays deferred - a
    // genuinely different, bespoke shape, not this pattern.
    void buildSpotCow();
    void buildHorse();
    void buildGlassDragon();
    // G12: Trophy Room - the last category-G scene. A bespoke builder, not
    // buildMeshGalleryScene(), since that helper only places ONE mesh -
    // this is four (bunny/teapot/Suzanne/Spot the Cow, matching CPU's
    // build_trophy_room() and OptiX's build_trophy_room_gpu() exactly in
    // mesh choice/material tones), the first hand-authored Metal scene to
    // combine multiple external OBJ meshes in one composition. Section
    // 120, docs/METAL_GPU_FEASIBILITY.md.
    void buildTrophyRoom();
    // A3: Checkered Spheres - the first category-A ("Basics") scene beyond
    // A1's own Cornell box. Real, direct spheres.push_back() calls (no
    // mesh import, no shared helper) - the geometry is simple enough
    // (5 spheres, all analytic) not to need one, and CPU's/OptiX's own
    // camera params port DIRECTLY (unlike G12's - see that function's own
    // comment) since nothing here goes through loadObjMesh()'s targetSize/
    // centre-based auto-fit convention. Introduces materialType 16 (real
    // 3D world-space checker) - see checker3DColor()'s own declaration
    // comment, metal_poc.metal. Section 121, docs/METAL_GPU_FEASIBILITY.md.
    void buildCheckeredSpheres();
    // A2: Bouncing Spheres (In One Weekend's own final scene) - see this
    // method's own definition comment (metal_poc_scenes_a.mm) for the
    // full "why," including the fixed-seed pseudo-random grid layout.
    // Section 175, docs/METAL_GPU_FEASIBILITY.md.
    void buildBouncingSpheres();
    // A6: Colored Quads - 5 flat-colour wall quads plus one emissive lamp
    // quad, matching CPU's build_quads()/build_quads_lights() exactly.
    // Pure addQuad() calls (no new material/geometry machinery at all -
    // every quad here is materialType 0, the same shape buildCornellBoxA1()
    // already uses for its own walls). Section 122, docs/
    // METAL_GPU_FEASIBILITY.md.
    void buildColoredQuads();
    // A4: Earth - a globe sphere textured with the SAME earthTexture the
    // hardcoded room's own back wall/sky already sample, reusing
    // materialType 9's own already-established equirectangularUV(normal)
    // technique (metal_poc.metal's own materialType==3 comment) to get a
    // sphere a UV coordinate with no real per-primitive UV
    // parameterization at all. Plus a small grey "moon" accent sphere
    // and a rim-light quad, matching CPU's build_earth()/
    // build_earth_lights() exactly. Section 123, docs/
    // METAL_GPU_FEASIBILITY.md.
    void buildEarth();
    // A5: Perlin Spheres - 4 spheres (a giant ground sphere + main sphere,
    // both `noise_texture(4)`; 2 smaller companion spheres, both
    // `noise_texture(8)`) sharing materialType 17 (real Perlin marble -
    // see turbulenceSimple()'s own declaration comment, metal_poc.metal)
    // plus a warm key-light quad, matching CPU's
    // build_perlin_spheres()/build_perlin_spheres_lights() exactly.
    // Section 124, docs/METAL_GPU_FEASIBILITY.md.
    void buildPerlinSpheres();
    // A7: Simple Light - reuses A5's own materialType 17 (Perlin marble)
    // for its ground+main sphere, plus a warm emissive SPHERE light and
    // a cool emissive quad light - matching CPU's own build_simple_light()
    // EXACTLY, including its real choice of `no_lights` (neither light
    // is NEE-registered even on CPU) - see this method's own definition
    // for why that's a deliberate fidelity choice, not a missing
    // feature. Section 125, docs/METAL_GPU_FEASIBILITY.md.
    void buildSimpleLight();
    // Shared "Cornell family" pattern (category B/Materials) - the SAME
    // 5 walls + ceiling light + rotated box + sphere shell
    // buildCornellBoxA1() already builds for A1, but with the box's and
    // sphere's own MATERIAL as caller-chosen parameters instead of
    // always white-Lambertian/glass - mirrors CPU's own
    // add_cornell_walls_and_main_light() + per-scene box/sphere swap
    // shape (src/TheRestOfYourLife/scenes_materials.h's own comment: "10
    // more Cornell-family scenes... swap in different sphere/box
    // materials"). Only materialType 0 (Lambertian)/2 (dielectric)/4
    // (GGX conductor)/5 (rough dielectric, sphere-only so far - section
    // 128) are supported by this helper so far - the materials this
    // Metal backend already fully implements; a scene needing a
    // not-yet-supported one (coated diffuse/conductor, subsurface, hair,
    // measured) stays out of scope until this helper (or a dedicated
    // builder) grows to cover
    // it. Section 126, docs/METAL_GPU_FEASIBILITY.md.
    void buildCornellFamilyScene(
        uint32_t boxMaterialType, float3 boxColor, float boxRoughness,
        float3 boxConductorEta, float3 boxConductorK, float boxIor,
        uint32_t sphereMaterialType, float3 sphereColor, float sphereRoughness,
        float3 sphereConductorEta, float3 sphereConductorK, float sphereIor,
        // Diffuse TRANSMITTANCE tint, materialType == 12 (diffuse
        // transmission, section 131) only - defaults to black (every
        // caller before B8 never used materialType 12 on the sphere).
        float3 sphereTransmitColor = simd::make_float3(0, 0, 0));
    // B2: Cornell Rough Metal - rough aluminium box + rough gold sphere,
    // both materialType 4 (GGX conductor), matching CPU's own
    // build_cornell_rough_metal() exactly.
    void buildCornellRoughMetal();
    // B4: Cornell Conductor - polished gold sphere + polished aluminium
    // box, both materialType 4 with REAL measured eta/k spectra (src/
    // shared/conductor_data.h's kConductorAu/kConductorAl - already used
    // elsewhere in this file, e.g. the hardcoded room's own gold accent
    // sphere) instead of B2's own flat-albedo reflectanceToConductorK()
    // approximation - matches CPU's own build_cornell_conductor(), which
    // uses the SAME real presets via its own `conductor` material class
    // (as opposed to B2's simpler `rough_metal`). Section 127, docs/
    // METAL_GPU_FEASIBILITY.md.
    void buildCornellConductor();
    // B3: Cornell Rough Glass - white diffuse box (unchanged from A1's
    // own) + a rough/frosted-dielectric sphere (materialType 5,
    // roughness 0.2, ior 1.5), matching CPU's own
    // build_cornell_rough_glass() exactly. First sphere-only use of
    // buildCornellFamilyScene()'s materialType 5 support. Section 128,
    // docs/METAL_GPU_FEASIBILITY.md.
    void buildCornellRoughGlass();
    // B6: Cornell Thin Glass - NOT built via buildCornellFamilyScene():
    // this scene has its own real shape (CPU's build_cornell_thin_glass(),
    // scenes_materials.h) - the standard 5 walls (still
    // cornell_box_data::kQuads[0..4], reused directly) and the same
    // rotated white box, but a DIFFERENT ceiling light (smaller, off-
    // centre) and an extra rotated vertical thin-glass PANEL (materialType
    // 11) splitting the box - no sphere at all. Section 129, docs/
    // METAL_GPU_FEASIBILITY.md.
    void buildCornellThinGlass();
    // B1: Rough Metal Spheres - NOT a Cornell-shell scene at all (a
    // different category-B shape: a row of 5 GGX conductor spheres over
    // a ground plane, lit by one real NEE-sampled quad light), matching
    // CPU's own build_rough_metal_spheres() exactly. Ground is a flat
    // quad, not CPU's own radius-1000 sphere - the SAME clearance issue
    // A5's own ground already had (section 124's own comment), same fix.
    // Section 130, docs/METAL_GPU_FEASIBILITY.md.
    void buildRoughMetalSpheres();
    // B8: Cornell Wax Slab - back to the buildCornellFamilyScene() shape
    // (walls+light+box+sphere), box unchanged white Lambertian, sphere
    // now materialType 12 (diffuse transmission - already implemented,
    // just newly wired into this helper). Section 131, docs/
    // METAL_GPU_FEASIBILITY.md.
    void buildCornellWaxSlab();
    // I8: Light Sampler Comparison - the one category-I (Education)
    // scene with genuinely NEW geometry (see the I-batch's own dispatch
    // comment): same A1 Cornell shell (walls/box/glass sphere, still
    // cornell_box_data::kQuads[0..4]/kBox/kGlassSphere) but the single
    // ceiling light is replaced by FIVE small quad lights of
    // deliberately lopsided power (~1:2:6:15:80), matching CPU's own
    // build_light_sampler_comparison() exactly. Bespoke builder (like
    // buildCornellThinGlass()) - buildCornellFamilyScene() assumes
    // exactly one light. Section 133, docs/METAL_GPU_FEASIBILITY.md.
    void buildLightSamplerComparison();
    // Shared geometry for category C's own "Cornell box, no ceiling
    // light, lit by ONE punctual light instead" family (C2-C6) - matches
    // CPU's own cornell_walls_no_light() exactly: the 5 standard walls
    // (kQuads[0..4], no light quad) plus a white diffuse sphere and a
    // metal accent sphere. Returns the SAME toWorld() lambda every
    // caller needs to place its own punctual light in this scene's own
    // rescaled coordinate space. Section 134, docs/
    // METAL_GPU_FEASIBILITY.md.
    std::function<float3(float3)> buildCornellNoLightWalls();
    void buildSpotlightCornell();
    void buildDistantLightCornell();
    void buildPointLightCornell();
    // C5/C6: Goniometric/Projection Light Cornell - same
    // buildCornellNoLightWalls() shell as C2-C4, but lit by a real
    // synthetic profile IMAGE instead of a plain scalar cone - reuses
    // the SAME dedicated pbrtGoniometricTexture/pbrtProjectionTexture
    // upload path sections 98/105 already established for a pbrt-
    // loaded light's own real image (havePbrtGoniometricImage/
    // pbrtGoniometricImagePixels etc. - the upload code itself reads
    // these members regardless of WHERE they were populated from, pbrt
    // file or, here, a hand-generated image), rather than the room's
    // own separate shared goniometricTexture/projectionTexture slot
    // (which stays reserved for the hardcoded demo room's own lights,
    // avoiding any texture-slot conflict between the two). Section 135,
    // docs/METAL_GPU_FEASIBILITY.md.
    void buildGoniometricLightCornell();
    void buildProjectionLightCornell();
    // F2: Triangle Mesh - a procedurally-generated icosahedron (12
    // vertices, 20 triangular faces, flat per-face normals), matching
    // CPU's own build_triangle_mesh_scene() exactly. Real triangle
    // geometry this loader already fully supports (no new primitive
    // TYPE needed at all, unlike F1's bilinear patch or F4's real
    // curve geometry - both genuinely bigger lifts, correctly
    // deferred) - only the vertex DATA is new. Section 136, docs/
    // METAL_GPU_FEASIBILITY.md.
    void buildTriangleMeshScene();
    // D5: Depth of Field Cornell Box - the SAME A1 Cornell box geometry
    // (build_cornell_box on CPU), just with real thin-lens defocus blur
    // (defocus_angle=2.0, focus_dist=800.0 in the scene's own pbrt-file-
    // scale units) - a genuinely free reuse of buildCornellBoxA1() plus
    // pbrtLensRadius/pbrtFocusDistance (this struct's own newly added
    // fields, see their own declaration comment). Section 137, docs/
    // METAL_GPU_FEASIBILITY.md.
    void buildDepthOfFieldCornellBox();
    // D1: Depth of Field - an open (non-Cornell) row-of-spheres scene
    // demonstrating the SAME real thin-lens defocus blur D5 already
    // wired up (pbrtLensRadius/pbrtFocusDistance), just at natural
    // scale (no 555-unit rescale, sceneScale=1.0 - the F2/B10/C1
    // convention) instead of a Cornell box. 6 fixed-material spheres
    // (2 out-of-focus lambertian, 1 in-focus glass, 1 in-focus metal,
    // 1 more out-of-focus lambertian) plus a checker ground and a row
    // of small accent spheres. Section 149, docs/METAL_GPU_FEASIBILITY.md.
    void buildDepthOfField();
    // D6: Orthographic Camera Cornell Box - the EXACT SAME A1 Cornell
    // box geometry (build_cornell_box on CPU, same as D5), just with a
    // real orthographic (parallel-projection) camera instead of the
    // usual perspective one - a genuinely free geometry reuse of
    // buildCornellBoxA1(), same shape as D5's own reuse, plus a NEW
    // camera projection mode (havePbrtOrthographic/
    // Uniforms::cameraOrthographic - this loader's first ever
    // non-perspective camera). Section 150, docs/METAL_GPU_FEASIBILITY.md.
    void buildOrthoCornellBox();
    // D2: Orthographic Camera (open scene) - an open (non-Cornell)
    // column-of-spheres scene demonstrating the SAME orthographic mode
    // D6 already wired up (havePbrtOrthographic), just at natural scale
    // (sceneScale=1.0, the D1/F2/B10/C1 convention) with a screen-window
    // half-extent of 5 (matching build_ortho_camera_scene()'s own alt-
    // camera lambda) instead of D6's own 320-for-a-555-unit-Cornell-box
    // value. Needs the SAME right-vector sign correction D6's own
    // section 150 finding established (CPU's alt-camera path is
    // internally inconsistent with its own primary one). Section 151,
    // docs/METAL_GPU_FEASIBILITY.md.
    void buildOrthoCameraScene();
    // D7: Spherical Camera Cornell Box - the EXACT SAME A1 Cornell box
    // geometry (same as D5/D6/D8), viewed from the box's OWN CENTER as
    // a real 360-degree equirectangular panorama (pbrt-v4
    // SphericalCamera) instead of a windowed perspective/orthographic
    // view - this loader's SECOND new camera projection mode
    // (havePbrtSpherical/Uniforms::cameraSpherical), a genuinely
    // different (non-linear, direction-only) ray-generation formula
    // than D6's own simple origin-offset one. Section 152, docs/
    // METAL_GPU_FEASIBILITY.md.
    void buildSphericalCornellBox();
    // D3: Spherical Camera (open scene) - an open (non-Cornell) ring-
    // of-spheres scene demonstrating the SAME equirectangular panorama
    // mode D7 already wired up (havePbrtSpherical), at natural scale.
    // Uses the SAME fixed-+Z-world-forward degenerate-cross-product
    // workaround D7's own build already established (D3's own registry
    // camera also has lookfrom directly above lookat, the same
    // degenerate input) - not a new construction to verify, the
    // identical one. Section 153, docs/METAL_GPU_FEASIBILITY.md.
    void buildSphericalCameraScene();
    // D4: Realistic Camera (open scene) - 5 spheres viewed through a
    // real 9-element simplified Double-Gauss lens (pbrt-v4
    // RealisticCamera), demonstrating genuine per-element-refraction
    // bokeh instead of the thin-lens DOF approximation D1/D5 use.
    // Builds a real, portable `RealisticCamera<float>` host-side and
    // reads its own GPU-port accessors directly - see
    // Uniforms::cameraRealistic's own comment for the full mechanism.
    // Section 157, docs/METAL_GPU_FEASIBILITY.md.
    void buildRealisticCameraScene();
    // D8: Realistic Camera Cornell Box - the EXACT SAME A1 Cornell box
    // geometry (same as D5/D6/D7), viewed through the SAME 9-element
    // lens D4 uses, just with the aperture diameter scaled up (350mm
    // vs D4's own 8mm) to keep a comparable defocus-cone ANGLE at this
    // scene's own much larger (555-unit) scale - matches CPU's own
    // setup_camera lambda exactly, the same "display convenience for
    // an already non-physical simplified lens" scaling CPU's own
    // comment documents, not a new choice.
    void buildRealisticCornellBox();
    // D13: Camera Motion Blur - the SAME A1 Cornell box geometry, plus
    // a real (approximated) camera shutter dolly - see this method's
    // own definition comment (metal_poc_scenes_d.mm) for the full
    // "why," including the honest translate-only approximation scope.
    // Section 174, docs/METAL_GPU_FEASIBILITY.md.
    void buildCameraMotionBlurCornellBox();
    // F1: Bilinear Patch - a Cornell box (the SAME 5 walls + ceiling
    // light literal as A1/E1, no box/sphere) containing TWO curved,
    // non-planar bilinear-patch surfaces (a saddle + a ramp), each a
    // GGX conductor. Ported as a fine tessellated triangle grid
    // (`addBilinearPatch()`) rather than a real new custom-primitive
    // type - see that helper's own declaration comment for the full
    // rationale. Section 154, docs/METAL_GPU_FEASIBILITY.md.
    void buildBilinearPatchScene();
    // F4: Curve Fibers - 70 windswept, tapered Bezier hair strands in a
    // Fibonacci-disk root arrangement, matching build_curve_fibers_scene()'s
    // own deterministic (non-RNG) placement exactly. Ported as
    // tessellated tapered tubes (`addTaperedTube()`) - matches this
    // scene's OWN registry description exactly ("GPU renders... tubes
    // of bilinear patches... rather than an exact curve intersection"),
    // the same documented simplification F1 already used. Section 155,
    // docs/METAL_GPU_FEASIBILITY.md.
    void buildCurveFibersScene();
    // E1: Homogeneous Medium - the standard A1 Cornell box WALLS (all 6
    // of kQuads[0..5] including the light - CPU's own scene reuses the
    // exact same light quad, no box/sphere at all) filled with a real
    // homogeneous scattering fog, reusing the ALREADY-EXISTING
    // pbrtFogSigmaT/pbrtFogAlbedo/pbrtFogAsymmetryG + havePbrtMedium
    // mechanism (previously only ever set by loadPbrtScene() for a real
    // pbrt file's own Medium block) directly from hand-authored C++
    // instead - that mechanism itself has no pbrt-specific logic at all,
    // it is a general "fill this scene's own enclosed interior with a
    // homogeneous medium" uniform, so reusing it needed no shader
    // changes. Section 138, docs/METAL_GPU_FEASIBILITY.md.
    void buildHomogeneousMediumScene();
    // E3: Dielectric Medium Showcase - 3 tinted-glass spheres
    // approximating CPU's own dielectric-plus-real-medium combination.
    // See this method's own definition comment (metal_poc_scenes_e.mm)
    // for the full "why," including the honest scope cut. Section 177,
    // docs/METAL_GPU_FEASIBILITY.md.
    void buildDielectricMediumShowcase();
    // E2: Cloud Medium - a real heterogeneous, procedural Perlin-noise
    // cloud (pbrt-v4 CloudMedium), delta-tracked through its own world-
    // space AABB via a trigger sphere, the same "invisible bounding
    // sphere plus analytic geometry" convention A8's own homogeneous
    // medium (materialType 28) established. See this method's own
    // definition comment (metal_poc_scenes_e.mm) for the full mechanism.
    // Section 178, docs/METAL_GPU_FEASIBILITY.md.
    void buildCloudMediumScene();
    // E4: RGB Grid Medium - a real heterogeneous "nebula" with an
    // independent per-voxel R/G/B scattering grid (pbrt-v4
    // RGBGridMedium), delta-tracked via the SAME trigger-sphere
    // convention as E2's own materialType 29 - see this method's own
    // definition comment (metal_poc_scenes_e.mm) for the full mechanism.
    // Section 179, docs/METAL_GPU_FEASIBILITY.md.
    void buildRgbGridMediumScene();
    // B9: Cornell Crystal - buildCornellFamilyScene() with the sphere as
    // materialType 18 (NormalizedFresnelBxDF - a genuinely NEW material,
    // not previously implemented before this PR - see
    // shadeNormalizedFresnel()'s own declaration comment, metal_poc.metal).
    // Section 139, docs/METAL_GPU_FEASIBILITY.md.
    void buildCornellCrystal();
    // B5: Cornell Coated Diffuse - buildCornellFamilyScene() with BOTH box
    // and sphere as materialType 19 (CoatedDiffuseBxDF, a genuinely NEW
    // stochastic layered-material shader function - see
    // shadeCoatedDiffuse()'s own declaration comment, metal_poc.metal).
    // Section 140, docs/METAL_GPU_FEASIBILITY.md.
    void buildCornellCoatedDiffuse();
    // B7: Cornell Coated Conductor - buildCornellFamilyScene() with BOTH
    // box and sphere as materialType 20 (CoatedConductorBxDF - the SAME
    // coat random walk as materialType 19's own CoatedDiffuseBxDF, just
    // with a GGX-conductor bottom bounce instead of Lambertian - see
    // shadeCoatedConductor()'s own declaration comment, metal_poc.metal).
    // Section 141, docs/METAL_GPU_FEASIBILITY.md.
    void buildCornellCoatedConductor();
    // B12: Normal Mapped Cornell - buildCornellFamilyScene() with the
    // sphere as materialType 21 (checker-driven normal-mapped
    // Lambertian - see the materialType==21u normal-perturbation
    // branch's own comment, metal_poc.metal). The box/back-wall's own
    // CPU-side bump map is a real, verified NO-OP (see
    // buildNormalMappedCornell()'s own comment) so the box stays plain
    // materialType 0, matching CPU's own actually-rendered behavior.
    // Section 142, docs/METAL_GPU_FEASIBILITY.md.
    void buildNormalMappedCornell();
    // B23: Glass Prism Dispersion - a real triangular prism (3 quads + 2
    // triangle end caps) under a single directional light, splitting
    // into a visible chromatic fan on a catcher screen. materialType 22
    // (recursive-backend dispersive dielectric - see
    // shadeDispersiveDielectric()'s own declaration comment,
    // metal_poc.metal). Not a Cornell-family scene at all - its own
    // bespoke geometry/camera/light, matching CPU's own
    // build_prism_dispersion_geometry() exactly.
    // Section 143, docs/METAL_GPU_FEASIBILITY.md.
    void buildPrismDispersion();
    // Shared by buildPrismDispersion() (B23) and
    // buildPrismDispersionRough() (B24) - identical geometry/camera/
    // light, differing only in the glass material/roughness passed in.
    void buildPrismDispersionGeometry(uint32_t glassMaterialType, float roughness);
    // B24: Frosted Prism Dispersion - same prism/light/screen as B23,
    // materialType 23 (dispersive rough dielectric - see
    // shadeDispersiveRoughDielectric()'s own declaration comment,
    // metal_poc.metal) instead of smooth. Section 144,
    // docs/METAL_GPU_FEASIBILITY.md.
    void buildPrismDispersionRough();
    // B10: Principled Showcase - a row of 7 spheres demonstrating pbrt-
    // v4's PrincipledBxDF (materialType 24 - see shadePrincipled()'s own
    // declaration comment, metal_poc.metal), matte diffuse through to
    // fully metallic/clearcoated, over a checkered ground plane, under
    // one overhead area light. NOT a Cornell-family scene - its own
    // standalone geometry/camera, matching build_principled_showcase()
    // exactly (F2's own "natural scale, no Cornell-style rescale"
    // convention, since this scene's own extent is already compact).
    // Section 145, docs/METAL_GPU_FEASIBILITY.md.
    void buildPrincipledShowcase();
    // B11: Hair Fibers - 5 spheres (materialType 31, shadeHair() -
    // metal_poc_materials_hair.metal) with pbrt-v4's own HairBxDF applied
    // as a fur/fiber material, matching build_hair_fibers() exactly. See
    // that Metal function's own comment for the field-reuse layout and
    // section 183, docs/METAL_GPU_FEASIBILITY.md.
    void buildHairFibersScene();
    // B14: Measured BRDF - 5 Lambertian spheres (matching CPU's own
    // measured_material, which never actually reads its own tabulated
    // data - see that class's own comment) under the first emissive
    // SPHERE light this loader supports (AreaLight::kind==1, new sphere-
    // light NEE - metal_poc_sampling.metal's sampleAreaLight()). See
    // buildMeasuredBrdfScene()'s own comment and section 184,
    // docs/METAL_GPU_FEASIBILITY.md.
    void buildMeasuredBrdfScene();
    // B13: Subsurface Slab - a Cornell box with a dielectric wax slab
    // (box) and jade sphere, each approximated as tinted glass
    // (materialType 2, E3's own precedent) rather than a real internal
    // scattering medium. See buildSubsurfaceSlab()'s own comment and
    // section 185, docs/METAL_GPU_FEASIBILITY.md.
    void buildSubsurfaceSlab();
    // A9: Final Scene - Book 2's own combined finale (400-box ground,
    // moving/dielectric/fuzzy-metal/tinted-glass/earth/Perlin-marble
    // spheres, a 1000-sphere cluster, and a whole-scene faint fog),
    // reusing several already-shipped mechanisms rather than needing
    // anything genuinely new. See buildFinalScene()'s own comment and
    // section 186, docs/METAL_GPU_FEASIBILITY.md.
    void buildFinalScene();
    // C1: HDRI Sky - an open scene (ground + 3 spheres: diffuse, rough
    // metal, glass) lit ENTIRELY by a procedural gradient sky, no other
    // light at all. Needs NO new materialType or shader code at all -
    // reuses the SAME real image-based infinite-light mechanism
    // loadPbrtScene() already populates for a real pbrt scene's own
    // ImageInfiniteLight (havePbrtImageEnvLight/pbrtEnvImageWidth/
    // Height/Pixels - the texture upload AND importance-sampling CDF
    // construction, metal_render_main()'s own GPU-resource setup, both
    // already trigger generically off these fields regardless of who
    // populated them), just with a HOST-side-generated 64x32 gradient
    // image (matching CPU's own build_hdri_sky() pixel formula exactly)
    // in place of a real loaded HDRI file. Section 146,
    // docs/METAL_GPU_FEASIBILITY.md.
    void buildHdriSky();
    // C7: Portal Infinite Light - a Cornell-family room (right/left/
    // ceiling/floor, NO ceiling light) whose back wall has an actual
    // rectangular window cut into it (4 border quads instead of one
    // solid quad), with a CONSTANT-colour sky visible through the
    // opening as the room's only light source (havePbrtConstantEnvLight/
    // pbrtEnvColor - the same miss-path-only mechanism buildCornellFamilyScene()'s
    // own background use and buildPrincipledShowcase() already exercise,
    // just as the scene's ONLY light instead of a supplement to an area
    // light). One metal sphere (materialType 4 via
    // reflectanceToConductorK(), approximating CPU's own simple
    // metal(...) - same B2/C1 substitution). Needs no new materialType/
    // shader code at all - the only genuinely new PART is the window
    // aperture itself, which is just 4 quads with a gap between them,
    // not a new geometry primitive. Section 147, docs/METAL_GPU_FEASIBILITY.md.
    void buildPortalLightScene();
    // Recomputes pbrtCameraPos/Forward/Right/Up for a new lookfrom in the
    // loaded scene's own pbrt-file coordinate space, keeping lookat/up/fov
    // exactly as loadPbrtScene() read them from the scene - see this
    // method's own definition (just after loadPbrtScene()) and
    // metal_render_main()'s own comment for why only lookfrom moves. Only
    // valid to call after a successful loadPbrtScene() (havePbrtCamera).
    void applyCameraOverride(double cam_x, double cam_y, double cam_z);
    bool buildGPUResources();
    bool compileShaderAndDispatch(int argc, const char** argv);
    bool postProcessAndWrite();  // false if the output file could not be written
};

