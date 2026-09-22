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

#include <cstring>
#include <functional>
#include <mach-o/dyld.h>

// Declare-only: src/external/image_writer.cpp is this project's one owner
// of STB_IMAGE_WRITE_IMPLEMENTATION (mirrors stb_image_impl.cpp/
// tinyexr_impl.cpp's own single-owner convention for the read-side
// libraries). Whichever executable links this file must also link
// image_writer.cpp - see root CMakeLists.txt's metal_poc/metal_renderer
// targets.
#include "../../src/external/stb_image_write.h"
// STB_IMAGE_IMPLEMENTATION here is a separate translation unit from
// src/external/stb_image_impl.cpp's own definition of it (that one is
// compiled into cpu_renderer, which metal_poc doesn't link against at
// all - two different executables, no ODR conflict) - loading
// images/earthmap.jpg for the textured-material test below.
//
// #undef immediately after - stb_image.h's own implementation section has
// no include-once guard of its own (only its DECLARATIONS do), relying on
// the convention that STB_IMAGE_IMPLEMENTATION is defined in exactly one
// .c/.cpp file project-wide. metal_poc.mm now also transitively includes
// src/shared/gzip_inflate.h (via pbrt_load.h -> ply_mesh.h, for real pbrt
// scene loading), which does its OWN plain #include "../external/
// stb_image.h" expecting just declarations - left defined, that second
// inclusion re-expands the whole implementation a second time in this
// same translation unit and fails with "redefinition of stbi__malloc" and
// a dozen more like it. Undefining right after this header's own
// (intentional, singular) implementation use is the correct scoping,
// same as any other single-header library's "define, include, undef"
// idiom - not a workaround, just doing it properly for the first time
// since nothing needed a second stb_image.h include in this file before.
#define STB_IMAGE_IMPLEMENTATION
#include "../../src/external/stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION

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

// PackedFloat3/PackedFloat2/AreaLightData/buildPowerLightSampler and every
// pure post-process function (blackbodyColor, vignetteFactor,
// chromaticAberration, the three tonemap operators, linearToSRGB) now live
// in this GPU-independent shared header instead of here directly, so
// metal_poc_math_tests.cpp can exercise the exact same code as a plain
// CPU-only unit test, with no Metal/Foundation link dependency at all -
// see that file and metal_poc_host_math.h's own comment for the full
// "why" (closing a real, previously-undocumented testing gap).
#include "metal_poc_host_math.h"

// pbrt_load.h -> pbrt_flatten.h's InfiniteLight image decode path needs
// tinyexr's real implementation linked in somewhere - cpu_renderer gets
// it from its own separate src/external/tinyexr_impl.cpp translation
// unit, but metal_poc is a standalone binary that doesn't link against
// that library at all, so it needs its own copy, same "define, include,
// undef" scoping as stb_image.h above and for the same reason.
#define TINYEXR_IMPLEMENTATION
#include "../../src/external/tinyexr.h"
#undef TINYEXR_IMPLEMENTATION
#include "../../src/shared/pbrt_load.h"
#include "../../src/shared/cornell_box_data.h"
// kConductorAu/kConductorCu - real (eta,k) presets for B7's own lacquered-
// gold sphere/lacquered-copper box (buildCornellCoatedConductor()) -
// reused directly rather than re-transcribed by hand, same as this
// file's own cornell_box_data.h include just above.
#include "../../src/shared/conductor_data.h"
// metal_render_main()'s own callable signature (section 79) - matches
// gpu/optix/optix_interface.h's own optix_render_main() shape exactly,
// down to reusing this SAME struct, so a future launcher/main.cpp caller
// (once this target is linked into ray_tracer itself, still TODO) needs
// no Metal-specific parameter shape of its own.
#include "../../src/shared/render_options.h"
// cpu_scene_pbrt_path_by_id() - resolves a scene_id string to its pbrt
// file path, the SAME shared C-ABI accessor gpu/optix/scene_builder.cpp
// already uses for pbrt-file-backed scenes, rather than re-scanning
// pbrt_scenes/ a second, independent way.
#include "../../cpu_renderer/cpu_interface.h"
// metal_render_main()'s own extern "C" declaration (with its default
// arguments - a C++ default argument may only be specified once, so this
// function's own DEFINITION below deliberately omits them) - included
// here, before that definition, so the compiler sees them as the same
// declaration rather than two independent ones.
#include "metal_interface.h"

// Mirrors metal_poc.metal's Uniforms/TriangleMaterial byte-for-byte -
// PackedFloat3 (not simd::float3) for every vector field, same reasoning
// as that file's own comment: simd::float3 is 16-byte aligned inside a
// struct, PackedFloat3 is a plain 12-byte triple with no padding, and a
// host/device struct-layout mismatch is a silent, hard-to-spot bug class
// worth designing out rather than debugging into.

#include "metal_poc_app.h"
#include "metal_poc_shader_files.h"


// --- Stage 1: CLI args + Metal device -----------------------------------
bool MetalPocApp::parseArgsAndCreateDevice(int argc, const char** argv) {
    width = (argc > 1) ? (uint32_t)atoi(argv[1]) : 400;
    height = (argc > 2) ? (uint32_t)atoi(argv[2]) : 400;
    outPath = (argc > 3) ? argv[3] : "/tmp/metal_poc_render.png";
    toneMapMode = parseToneMapMode((argc > 6) ? argv[6] : nullptr);
    if (argc > 7) pbrtScenePath = argv[7];
    if (argc > 8) handAuthoredSceneId = argv[8];

    // MTLCreateSystemDefaultDevice() is explicitly documented as
    // unsupported for command-line/daemon processes (confirmed via
    // `log show`: "Use of MTLCreateSystemDefaultDevice is not
    // supported for non-interactive (commandline or daemon) apps. Use
    // MTLCopyAllDevices(WithObserver) instead.") - this POC is a plain
    // CLI tool, not an app bundle, so it needs the enumeration API.
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (devices.count > 0) device = devices[0];
    if (!device) {
        fprintf(stderr, "No Metal device available.\n");
        return false;
    }
    fprintf(stderr, "Metal device: %s\n", device.name.UTF8String);
    if (!device.supportsRaytracing) {
        fprintf(stderr, "Device does not support hardware raytracing.\n");
        return false;
    }

    queue = [device newCommandQueue];
    return true;
}

// --- Stage 2: build the scene's own host-side data ----------------------
void MetalPocApp::buildScene() {
    // --- Scene: a Cornell-box-like room, world units ~[-1,1] --------
    // Matches this project's CPU Cornell box in spirit (floor/ceiling/
    // back wall + coloured side walls + an object), not in exact
    // dimensions - this POC's scene is entirely separate authored data,
    // not a shared asset with cpu_renderer/.

    const float3 white{0.73f, 0.73f, 0.73f};
    const float3 red{0.65f, 0.05f, 0.05f};
    const float3 green{0.12f, 0.45f, 0.15f};

    // Floor (y = -1) - procedural checkerboard (materialType 6), the
    // one surface in the scene that samples NO texture/image at all
    // for its albedo, a genuinely different technique from
    // materialType 3's earthTexture lookup (analytic UV math instead
    // of a sampler call). addQuad()'s own planar 0-1 UVs across the
    // whole floor, combined with checkerColor()'s own 8-tiles-per-UV-
    // unit scale, give 8x8 tiles across the room's own floor.
    addQuad(verts, normals, uvs, materials, float3{-1,-1,-1}, float3{1,-1,-1}, float3{1,-1,1}, float3{-1,-1,1}, white, /*materialType=*/6);
    // Ceiling (y = 1)
    addQuad(verts, normals, uvs, materials, float3{-1,1,1}, float3{1,1,1}, float3{1,1,-1}, float3{-1,1,-1}, white);
    // Back wall (z = -1) - textured (materialType 3): the one surface
    // in the scene that samples earthTexture, chosen because it's the
    // large flat backdrop the camera looks straight at, showing the
    // whole 0-1 UV mapping unobstructed.
    addQuad(verts, normals, uvs, materials, float3{-1,-1,-1}, float3{-1,1,-1}, float3{1,1,-1}, float3{1,-1,-1}, white, /*materialType=*/3);
    // Left wall (x = -1), red
    addQuad(verts, normals, uvs, materials, float3{-1,-1,1}, float3{-1,1,1}, float3{-1,1,-1}, float3{-1,-1,-1}, red);
    // Right wall (x = 1), green
    addQuad(verts, normals, uvs, materials, float3{1,-1,-1}, float3{1,1,-1}, float3{1,1,1}, float3{1,-1,1}, green);
    // A small procedurally bump-mapped panel (materialType 7),
    // flush-mounted just in front of the back wall (z = -0.99, the
    // same off-surface margin the mirror disk/other flush-mounted
    // geometry already uses to avoid z-fighting) rather than a side
    // wall - the one surface in the scene whose shading normal is
    // perturbed AWAY from its own true (perfectly flat) geometric
    // normal, an analytic egg-carton height field rather than a
    // stored normal-map texture (no new image asset needed - see
    // metal_poc.metal's own proceduralBumpNormal() comment).
    // Positioned in the region the directional light (see
    // metal_poc.metal's own DirectionalLight comment) hits closest to
    // head-on, not tucked against a side wall - a first attempt
    // mounted on the red wall got barely any direct light at all
    // (nearly the same shallow self-shadowing the directional light's
    // own doc describes for that wall), making the bump invisible
    // under GI-only ambient lighting; moved here after that render
    // came back looking completely flat, not assumed correct from
    // the code alone. `roughness` here means bump strength, not a
    // BRDF parameter - materialType 7's own reuse of that field, see
    // TriangleMaterial's comment above.
    addQuad(verts, normals, uvs, materials,
            float3{0.15f, -0.3f, -0.99f}, float3{0.15f, 0.3f, -0.99f},
            float3{0.75f, 0.3f, -0.99f}, float3{0.75f, -0.3f, -0.99f},
            float3{0.55f, 0.5f, 0.45f}, /*materialType=*/7,
            /*emission=*/simd::make_float3(0, 0, 0), /*lightId=*/-1,
            /*roughness(bump strength)=*/0.6f);
    // A thin dielectric "glass pane" (materialType 11) - see
    // metal_poc.metal's own comment on this materialType for the full
    // "why" (a zero-thickness slab, transmits straight through with no
    // bending at all, unlike materialType 2/5's own SOLID glass sphere).
    // Floating in open space above the sphere cluster and below the
    // ceiling lights (y=0.05-0.55), facing the camera directly (normal
    // along +Z) so the mural on the back wall reads UNDISTORTED through
    // it - the one visual signature that actually distinguishes this
    // from a solid dielectric, which would show visible bending/
    // magnification of whatever's behind it. `color` is unused by this
    // materialType (see that comment) - passed as white only because
    // addQuad() itself has no "no colour" concept, not because it means
    // anything here.
    addQuad(verts, normals, uvs, materials,
            float3{-0.25f, 0.05f, 0.0f}, float3{0.25f, 0.05f, 0.0f},
            float3{0.25f, 0.55f, 0.0f}, float3{-0.25f, 0.55f, 0.0f},
            white, /*materialType=*/11,
            /*emission=*/simd::make_float3(0, 0, 0), /*lightId=*/-1,
            /*roughness=*/0.0f, /*ior=*/1.5f);
    // A translucent "leaf" panel (materialType 12, diffuse transmission) -
    // see metal_poc.metal's own comment on this materialType for the full
    // "why" (a two-sided diffuser, unlike materialType 11's own
    // undistorted-straight-through specular transmission just above).
    // Deliberately placed almost exactly AT the first point light's own
    // depth (PackedFloat3{0,0.3,0.3} below, z=0.3 - this quad sits at
    // z=0.35, just in front of it from the camera's own +Z-facing view)
    // so that light is genuinely BACKLIT through this panel, not merely
    // side-lit - the one placement that actually exercises the
    // transmission lobe's own NEE path, not just the reflection lobe
    // every other diffuse material in this scene already covers.
    // Reflectance (`color`) and transmittance are DIFFERENT tints (a
    // real leaf's own transmitted colour is warmer/brighter, not just a
    // dimmer copy of its reflected one) - the classic "backlit leaf
    // glows a lighter green" effect this placement is chosen to show.
    addQuad(verts, normals, uvs, materials,
            float3{-0.22f, 0.08f, 0.35f}, float3{0.22f, 0.08f, 0.35f},
            float3{0.22f, 0.5f, 0.35f}, float3{-0.22f, 0.5f, 0.35f},
            /*color(reflectance)=*/float3{0.25f, 0.45f, 0.12f}, /*materialType=*/12,
            /*emission=*/simd::make_float3(0, 0, 0), /*lightId=*/-1,
            /*roughness=*/0.0f, /*ior=*/1.0f,
            /*transmitColor=*/float3{0.18f, 0.6f, 0.1f});
    // Suzanne (Blender's monkey mascot, models/suzanne.obj - a real
    // mesh, 500 faces) replaces the earlier flat tilted-quad "mirror
    // test object": mirror MATERIAL coverage is already proven (the
    // sphere/dielectric PR's own mirror-quad screenshots), what this
    // scene hadn't tested yet is real, DATA-DRIVEN geometry with
    // genuine per-triangle normal variation, not a hand-authored
    // axis-aligned quad. Lambertian so its form reads clearly via
    // shading rather than showing room reflections. RT_MODELS_DIR
    // mirrors RT_METAL_SHADER_DIR's own fallback shape below.
    //
    // Loaded into ITS OWN vertex/normal/uv/material vectors (not the
    // shared `verts`/`normals`/`uvs`/`materials` the room quads and
    // Spot still use) - Suzanne gets her own acceleration structure
    // below, instanced twice with two DIFFERENT transforms, the one
    // piece of this scene that actually exercises a non-identity
    // instance transform. A shared-buffer object can only ever have
    // ONE position (it's baked directly into world-space vertex
    // positions at load time); testing instancing needs the SAME
    // object-space geometry referenced from more than one instance
    // descriptor, which needs its own acceleration structure.
    NSString* modelsDir = nil;
    {
        NSString* exeDir = executableDir();
        NSString* candidate = [exeDir stringByAppendingPathComponent:@"models"];
        if (exeDir && [[NSFileManager defaultManager] fileExistsAtPath:
                [candidate stringByAppendingPathComponent:@"suzanne.obj"]]) {
            modelsDir = candidate;
        }
    }
    if (!modelsDir) {
#ifdef RT_MODELS_DIR
        modelsDir = @(RT_MODELS_DIR);
#else
        modelsDir = [[@(__FILE__) stringByDeletingLastPathComponent]
            stringByAppendingPathComponent:@"../../models"];
#endif
    }
    NSString* suzannePath = [modelsDir stringByAppendingPathComponent:@"suzanne.obj"];
    const float3 bronze{0.55f, 0.35f, 0.15f};
    if (!loadObjMesh(suzannePath.UTF8String, suzanneVerts, suzanneNormals, suzanneUVs, suzanneMaterials, bronze,
                      /*center=*/float3{0.0f, 0.0f, 0.0f}, /*targetSize=*/0.75f)) {
        fprintf(stderr, "Continuing without Suzanne - check RT_MODELS_DIR / models/suzanne.obj.\n");
    }

    // Spot (Keenan Crane's textured cow model, models/spot.obj) - unlike
    // suzanne.obj, this file has real per-face-corner `vt` data (3225
    // entries) and NO `vn` at all, the exact inverse case from Suzanne's
    // own (vn on every face, no vt) - loading it exercises
    // loadObjMesh()'s real-UV path (not just its zero-fallback path)
    // and its flat-normal fallback path in the same call, closing the
    // texture-mapping PR's own explicitly-deferred "real vt/f v/vt/vn
    // parsing" item. materialType 3 (textured) reuses `earthTexture` -
    // wrapping a world map onto a cow is a deliberately silly texture
    // choice for a mesh that was never authored to use it, but it's
    // exactly what makes this a REAL demonstration of per-vertex UV
    // interpolation rather than a coincidentally-plausible-looking
    // result: the world map's grid lines and coastlines have to follow
    // spot's actual surface curvature for this to look right at all.
    NSString* spotPath = [modelsDir stringByAppendingPathComponent:@"spot.obj"];
    if (!loadObjMesh(spotPath.UTF8String, verts, normals, uvs, materials, white,
                      /*center=*/float3{0.78f, -0.75f, 0.6f}, /*targetSize=*/0.42f,
                      /*materialType=*/3)) {
        fprintf(stderr, "Continuing without Spot - check RT_MODELS_DIR / models/spot.obj.\n");
    }

    // Area lights: real geometry, hanging just under the ceiling
    // (y=0.98, not y=1 itself - avoids z-fighting/coplanar overlap
    // with the ceiling's own quad above), facing straight down. Two
    // separate lights (not one, as every previous PR up through #11
    // had) - the smallest scene change that actually exercises
    // `lights` as a genuine LIST rather than a single renamed
    // constant: a warm light and a cool light side by side prove the
    // shader's own light-picking/MIS code path handles more than one
    // entry, visibly (two independently-coloured highlights/shadow
    // directions), not just structurally. `addAreaLight` keeps each
    // call's geometry (addQuad, tagged with this light's own index
    // via materialType/lightId) and its AreaLightData entry (same
    // corners, reduced to center/edgeU/edgeV/normal/area) in sync by
    // construction, rather than needing two hand-authored, separately-
    // maintained descriptions of the same quad the way the single-
    // light version's kLightCenter/kLightHalfExtents/kLightNormal
    // constants over in metal_poc.metal used to (see that file's
    // AreaLight struct comment for the "replacing..." history).
    // `patternTileB`/`patternScale` default to 0 (flat emission,
    // materialType 0) - see AreaLightData's own comment. Passing a
    // nonzero `patternScale` switches the light's own triangles to
    // materialType 10 too, so a direct hit and an NEE sample both
    // evaluate the SAME checker pattern (just at each one's own
    // different point on the light - see metal_poc.metal's own
    // comments on materialType 10 and AreaLight for why that needs
    // two separate evaluations, not one shared value).
    auto addAreaLight = [&](float3 a, float3 b, float3 c, float3 d, float3 emission,
                             float patternTileB = 0.0f, float patternScale = 0.0f) {
        int32_t lightId = (int32_t)lights.size();
        uint32_t materialType = (patternScale > 0.0f) ? 10u : 0u;
        addQuad(verts, normals, uvs, materials, a, b, c, d, white,
                materialType, emission, lightId, /*roughness(pattern tileB)=*/patternTileB);
        float3 edgeU = b - a;
        float3 edgeV = d - a;
        float3 normal = simd::normalize(simd::cross(edgeU, edgeV));
        float area = simd::length(simd::cross(edgeU, edgeV));
        float3 center = a + 0.5f * edgeU + 0.5f * edgeV;
        lights.push_back(AreaLightData{
            PackedFloat3{center.x, center.y, center.z},
            PackedFloat3{edgeU.x, edgeU.y, edgeU.z},
            PackedFloat3{edgeV.x, edgeV.y, edgeV.z},
            PackedFloat3{normal.x, normal.y, normal.z},
            area,
            PackedFloat3{emission.x, emission.y, emission.z},
            patternTileB,
            patternScale});
    };
    // Both ceiling lights' own colours now come from blackbodyColor()
    // (see that function's own comment, added for the point/spot/
    // directional lights) rather than the hand-picked tuples above -
    // 2700 K (a standard incandescent bulb) for the warm one, 20000 K
    // (near the top of the fit's own valid range, a very hot/blue-
    // white source) for the cool one. Notably, 20000K's own derived
    // colour is nowhere near as saturated a blue as the hand-picked
    // (6, 10, 18) it replaces - real blackbody radiation never gets
    // that saturated; no amount of temperature produces a deeply
    // saturated blue the way an artistic RGB pick can. A real,
    // honestly-reported limitation of deriving colour from physical
    // temperature, not swept under the rug.
    const float3 warmAreaLightColor = blackbodyColor(2700.0f) * 15.5f;
    const float3 coolAreaLightColor = blackbodyColor(20000.0f) * 13.9f;
    addAreaLight(float3{-0.58f,0.98f,-0.25f}, float3{-0.22f,0.98f,-0.25f},
                 float3{-0.22f,0.98f,0.25f}, float3{-0.58f,0.98f,0.25f},
                 /*emission=*/warmAreaLightColor);
    // The cool light also gets a patterned diffuser-grid look
    // (materialType 10 - see that comment for the full "why"), a
    // real fixture detail the flat-emission warm light doesn't have:
    // tile B at 40% of tile A's own brightness (a translucent grid,
    // not fully opaque black bars) across a 6x6 tiling of the
    // light's own 0-1 UV span.
    addAreaLight(float3{0.22f,0.98f,-0.25f}, float3{0.58f,0.98f,-0.25f},
                 float3{0.58f,0.98f,0.25f}, float3{0.22f,0.98f,0.25f},
                 /*emission=*/coolAreaLightColor,
                 /*patternTileB=*/0.4f, /*patternScale=*/6.0f);
    // Real pbrt scene loading (see loadPbrtScene()'s own comment):
    // ADDITIVE, not a replacement for the hardcoded room above - its own
    // geometry gets recentred/rescaled/offset well clear of this room's
    // own [-1,1] region (loadPbrtScene()'s own bounding-box normalization
    // step), so the two coexist without visually interfering, without
    // needing to conditionally skip building any of buildGPUResources()'s
    // existing per-geometry-type acceleration structures/buffers (which
    // assume every one of these vectors is always non-empty - true
    // before this, and still true now). MUST run before
    // buildPowerLightSampler() just below (which needs the FULL, final
    // light list - the same reason every addAreaLight() call above also
    // precedes it) - a first version called this after that build call
    // instead, and every light this function adds silently became
    // unreachable by NEE (see buildPowerLightSampler()'s own call site
    // history for the full story).
    if (!pbrtScenePath.empty()) loadPbrtScene();
    // Same ADDITIVE reasoning as loadPbrtScene() just above (its own
    // comment) - mutually exclusive with it in practice (metal_render_main()
    // only ever sets one of pbrtScenePath/handAuthoredSceneId), but if both
    // were somehow set, both would just coexist harmlessly the same way a
    // pbrt scene and this hardcoded room already do.
    if (!handAuthoredSceneId.empty()) buildHandAuthoredScene(handAuthoredSceneId);

    // Builds each light's own pmf/aliasProb/aliasIndex in place - see
    // buildPowerLightSampler()'s own comment. Must run after every
    // addAreaLight() call above (needs the full, final light list) and
    // before `lights` gets uploaded to the GPU buffer below.
    buildPowerLightSampler(lights, logPowerLightSamplerLine);

    // Two spheres, both custom (non-triangle) primitives via a shared
    // bounding-box acceleration structure + intersection function
    // (metal_poc.metal's sphereIntersectionFunction indexes into
    // `spheres`/`sphereMaterials` by primitive_id, so any number of
    // spheres share one geometry/one intersection function - adding a
    // second one below is purely a host-side array-of-2 change, no
    // shader change) - the one "Medium risk, unconfirmed" item
    // docs/METAL_GPU_FEASIBILITY.md section 3 originally flagged that
    // the room/mirror geometry alone hadn't exercised (triangles only).
    // Sphere 0: glass, ior 1.5 matches common glass, same value this
    // project's own CPU Cornell box scene (A1) uses for its glass
    // sphere. Sphere 1: a rough (GGX) conductor - gold-ish F0, roughness
    // 0.15 (a fairly tight but visibly non-mirror highlight), placed on
    // the opposite side of the room so both new-material spheres read
    // clearly side by side.
    // Sphere 2: rough (frosted) dielectric, materialType 5 - small,
    // front-and-centre between the other two spheres and just in
    // front of Suzanne. An earlier back-left-corner placement turned
    // out to sit almost exactly along the camera-to-gold-sphere
    // sightline (both ~20% off-axis, gold sphere much closer/larger)
    // and was fully hidden - caught by actually rendering and
    // inspecting the image, not by the bounding-region math alone
    // (which only rules out 3D overlap, not 2D screen-space
    // occlusion) - this position was checked against both.
    // Sphere 3: a procedurally roughness-mapped GGX conductor
    // (materialType 9) - a "worn/scratched copper" look, patches of
    // near-mirror-smooth and rough microfacet regions on the SAME
    // surface (see metal_poc.metal's own materialType 9 comment).
    // Genuinely different from sphere 1's own anisotropic roughness:
    // that one varies BY DIRECTION at a single point (one alpha per
    // tangent axis, constant everywhere on the sphere); this one
    // varies BY LOCATION (alpha itself is a function of where on the
    // sphere you look, isotropic at any single point). Placed
    // resting on the floor (`y = -1 + radius`, matching every other
    // floor sphere's own convention) at the room's front-right,
    // clear of the gold sphere/Spot/the disk mirror - checked by
    // rendering and inspecting, not just the bounding-sphere math.
    // Sphere 4: clearcoat/glossy-plastic (materialType 8) - a deep
    // red "car paint" look, a sharp specular highlight riding on top
    // of a genuinely diffuse (not metallic) coloured base, the
    // signature that distinguishes this from every reflective
    // material already in the scene (mirror/GGX conductor are
    // colour-tinted AT the reflection itself; clearcoat's own
    // reflection stays colourless/white, only the diffuse base
    // beneath carries colour). Placed at the room's front-right,
    // deliberately at a different x AND z from Spot-the-cow (this
    // POC's own established near-miss from sphere 3's own placement:
    // sharing an x coordinate with a closer foreground object hid it
    // completely) and the gold sphere.
    // .insert(spheres.begin(), {...}) rather than a plain `spheres = {...}`
    // assignment - loadPbrtScene() (see its own call site, above
    // buildPowerLightSampler()) may already have pushed pbrt-loaded
    // spheres onto this vector by the time this code runs; a plain `=`
    // would silently wipe those back out. Inserting the hardcoded room's
    // own spheres at the FRONT keeps their existing index convention
    // (every hand-picked index/comment below, e.g. "gold sphere's own
    // z"/spheres[1], is unaffected) while anything loadPbrtScene() added
    // earlier lands after them, not lost.
    spheres.insert(spheres.begin(), {
        SphereData{PackedFloat3{0.35f, -0.65f, 0.15f}, 0.35f},
        SphereData{PackedFloat3{-0.55f, -0.65f, 0.45f}, 0.35f},
        SphereData{PackedFloat3{-0.05f, -0.82f, 0.6f}, 0.18f},
        SphereData{PackedFloat3{-0.78f, -0.78f, 0.7f}, 0.18f},
        SphereData{PackedFloat3{0.8f, -0.85f, 1.0f}, 0.15f},
        // A 6th sphere (materialType 13, Oren-Nayar rough diffuse) - a
        // different x from every sphere above it (the established "same
        // x as a nearer object hides the new one completely" lesson,
        // step 34's own note). z=0.9/y=-0.85 keeps it just inside the
        // room's own [-1,1] bounding box and resting on the floor -
        // an EARLIER placement attempt (z=1.3, y=-0.88) put it outside
        // the room entirely and clipping through the floor, confirmed
        // invisible via a diagnostic magenta-Lambertian recolour render
        // before this fix, not assumed correct from the coordinates
        // alone.
        SphereData{PackedFloat3{-0.3f, -0.85f, 0.9f}, 0.15f},
        // A 7th sphere (materialType 14, Ashikhmin velvet) - a further
        // different x again from every sphere above (same lesson as the
        // Oren-Nayar sphere's own comment), and kept well inside the
        // room's own [-1,1] bounding box/resting on the floor this time,
        // not repeating that sphere's own first-attempt mistake.
        SphereData{PackedFloat3{0.15f, -0.85f, 0.75f}, 0.15f},
    });
    // Sphere 0 and 2's own `color` is now a Beer-Lambert ABSORPTION
    // coefficient (see metal_poc.metal's own applyBeerLambertAbsorption()
    // comment), not a reflectance/tint the way every other material's
    // `color` field is read - {1,1,1} would have meant "absorb
    // everything, render black" under this new interpretation, so
    // both dielectric spheres' old placeholder {1,1,1} "clear glass"
    // values were replaced with real per-channel absorption:
    // sphere 0 is emerald-tinted (absorbs red/blue faster than
    // green, getting more richly green toward its own thicker
    // centre), sphere 2 a much milder amber (still reads mostly
    // frosted-white, just warmed slightly).
    // Same insert-at-front reasoning as `spheres` above - keeps this
    // parallel array's own indices aligned with it (loadPbrtScene()'s own
    // sphereMaterials.push_back() calls, if any, already ran earlier).
    sphereMaterials.insert(sphereMaterials.begin(), {
        TriangleMaterial{PackedFloat3{0.5f, 0.05f, 0.35f}, /*materialType=*/2, /*ior=*/1.5f, PackedFloat3{0, 0, 0}},
        // Genuinely ANISOTROPIC now (alphaX from `ior`, alphaY from
        // `roughness` - see TriangleMaterial's own comment): a tight
        // 0.08 in one tangent direction and a much broader 0.45 in
        // the other, the classic "brushed metal" look - a real,
        // deliberate change from the previously-isotropic 0.15 this
        // sphere used through step 22, not a value chosen to
        // preserve the old appearance (that A/B check is done via a
        // dedicated verification render instead, not the committed
        // scene - see docs/METAL_GPU_FEASIBILITY.md's own note).
        // conductorEta/conductorK: real gold (Au) complex IOR, sampled at
        // the sRGB primary wavelengths (630/532/467nm) from pbrt-v4's own
        // spectral tables - src/shared/conductor_data.h's kConductorAu,
        // matching this sphere's own approximate gold tint above (which
        // is now vestigial as a Fresnel input - see this branch's own
        // comment in metal_poc.metal - but left in place unchanged so
        // every other reader of `color` on this material, if any existed,
        // stays unaffected).
        TriangleMaterial{PackedFloat3{1.0f, 0.86f, 0.57f}, /*materialType=*/4, /*alphaX=*/0.08f, PackedFloat3{0, 0, 0},
                         /*lightId=*/-1, /*alphaY=*/0.45f,
                         /*conductorEta=*/PackedFloat3{0.184f, 0.457f, 1.354f},
                         /*conductorK=*/PackedFloat3{3.070f, 2.408f, 1.818f}},
        TriangleMaterial{PackedFloat3{0.12f, 0.08f, 0.02f}, /*materialType=*/5, /*ior=*/1.5f, PackedFloat3{0, 0, 0},
                         /*lightId=*/-1, /*roughness=*/0.35f},
        // materialType 9: `ior` is the SMOOTH patch's own perceptual
        // roughness, `roughness` the ROUGH patch's - both squared
        // into GGX alpha exactly like materialType 4 already does,
        // just picked between by an analytic UV-space checker
        // pattern instead of being one constant.
        // conductorEta/conductorK: real copper (Cu) complex IOR (same
        // source/sampling as the gold sphere above's own comment),
        // matching this sphere's own approximate copper tint.
        TriangleMaterial{PackedFloat3{0.8f, 0.45f, 0.2f}, /*materialType=*/9, /*ior(smooth)=*/0.05f, PackedFloat3{0, 0, 0},
                         /*lightId=*/-1, /*roughness(rough)=*/0.6f,
                         /*conductorEta=*/PackedFloat3{0.246f, 1.072f, 1.155f},
                         /*conductorK=*/PackedFloat3{3.378f, 2.591f, 2.469f}},
        // materialType 8: `color` is the diffuse BASE colour under
        // the coat (materialType 0's own convention) - a deep,
        // fairly saturated red, since the coat's own reflection
        // stays colourless regardless.
        TriangleMaterial{PackedFloat3{0.55f, 0.05f, 0.06f}, /*materialType=*/8, /*ior=*/1.0f, PackedFloat3{0, 0, 0}},
        // materialType 13: Oren-Nayar rough diffuse - `roughness` is this
        // material's own sigma parameter (see TriangleMaterial's own
        // comment), 0.9 deliberately high so the grazing-angle
        // retroreflective brightening/head-on darkening relative to
        // plain Lambertian reads clearly, not subtly. A warm terracotta/
        // clay tint - the real-world material family this BxDF was
        // originally designed to model.
        TriangleMaterial{PackedFloat3{0.75f, 0.55f, 0.4f}, /*materialType=*/13, /*ior=*/0.0f, PackedFloat3{0, 0, 0},
                         /*lightId=*/-1, /*roughness(sigma)=*/0.9f},
        // materialType 14: Ashikhmin velvet - `ior` is this material's
        // own sigma (spread) parameter (see TriangleMaterial's own
        // comment), 0.3 matching the value this material's own
        // verification reference program used. A deep red "velvet
        // cloth" tint - the real-world material family this BxDF was
        // originally designed to model.
        TriangleMaterial{PackedFloat3{0.5f, 0.05f, 0.15f}, /*materialType=*/14, /*ior(sigma)=*/0.3f, PackedFloat3{0, 0, 0}},
    });

    // A wall-mounted mirror disk (materialType 1) - a second, distinct
    // custom-primitive SHAPE, not just another sphere. Every custom
    // primitive so far (however many) has gone through the SAME
    // intersection function at function-table slot 0; this is what
    // actually exercises a second, different function at slot 1 (see
    // metal_poc.metal's own comment on diskIntersectionFunction).
    // Also, incidentally, the first object in this whole scene to use
    // materialType 1 (mirror) at all - it's existed in the shader
    // since the very first multi-material step but nothing had
    // actually used it since the mirror test quad was replaced by the
    // dielectric sphere back in step 6.
    // .insert(...begin(), {...}) rather than a plain `= {...}` assignment -
    // loadPbrtScene() doesn't populate disks/diskMaterials today, but it's
    // the same latent wipe-pbrt-data-out bug class `spheres`'s own
    // insert-at-front comment documents, so this is fixed proactively
    // rather than left as a trap for whenever a future increment adds
    // pbrt disk support.
    disks.insert(disks.begin(), {
        DiskData{PackedFloat3{0.97f, 0.3f, -0.3f}, PackedFloat3{-1.0f, 0.0f, 0.0f}, 0.22f},
    });
    diskMaterials.insert(diskMaterials.begin(), {
        TriangleMaterial{PackedFloat3{0.9f, 0.9f, 0.9f}, /*materialType=*/1, /*ior=*/1.0f, PackedFloat3{0, 0, 0}},
    });

    // A true delta point light - genuinely different from every
    // AreaLight above (zero area, hard-edged shadows, no NEE/MIS
    // weighting needed at all - see metal_poc.metal's own PointLight
    // comment). Placed off-axis from both area lights so it adds a
    // THIRD, distinctly-positioned specular highlight to the
    // reflective spheres/disk rather than blending into an existing
    // one - the easiest way to visually confirm it's really
    // contributing light, not just present in the buffer unused.
    // Second point light: a genuine SPOT (cone-restricted), unlike the
    // first one's omnidirectional glow - aimed down at Spot-the-cow's
    // own floor area, a real "pool of light" cone signature an
    // omnidirectional point light cannot produce at all (its own
    // illumination falls off with distance everywhere, never with
    // ANGLE the way a spot's does). 25 degree outer / 15 degree inner
    // cone (smoothstep-blended between them, not a hard edge).
    //
    // All three lights below get their own colour from
    // blackbodyColor() at a NAMED physical temperature rather than a
    // hand-picked RGB tuple - 9000 K (cool, moonlight-ish) for this
    // first point light, 3000 K (warm tungsten) for the spot, 5778 K
    // (the Sun's own real photosphere temperature) for the
    // directional light below. Each still scaled by a plain
    // intensity multiplier chosen to land in roughly the same
    // brightness range this scene's own lights already used - only
    // the HUE is now derived, not the overall exposure.
    const float3 spotPos = float3{0.65f, 0.9f, -0.1f};
    const float3 spotTarget = float3{0.7f, -1.0f, 0.4f};
    const float3 spotDir = simd::normalize(spotTarget - spotPos);
    const float3 pointLight1Color = blackbodyColor(9000.0f) * 1.0f;
    const float3 spotLightColor = blackbodyColor(3000.0f) * 11.3f;
    // .insert(pointLights.begin(), {...}) rather than a plain `pointLights =
    // {...}` assignment - loadPbrtScene() (called earlier, above
    // buildPowerLightSampler()) may already have pushed pbrt-parsed
    // point/spot lights onto this vector by the time this code runs; a
    // plain `=` would silently wipe those back out, the exact same bug
    // class `spheres`'s own insert-at-front (see that vector's own
    // comment) was already fixed for.
    pointLights.insert(pointLights.begin(), {
        PointLightData{PackedFloat3{0.0f, 0.3f, 0.3f}, PackedFloat3{pointLight1Color.x, pointLight1Color.y, pointLight1Color.z}},
        PointLightData{PackedFloat3{spotPos.x, spotPos.y, spotPos.z}, PackedFloat3{spotLightColor.x, spotLightColor.y, spotLightColor.z},
                       PackedFloat3{spotDir.x, spotDir.y, spotDir.z},
                       /*cosOuterAngle=*/cosf(25.0f * (float)M_PI / 180.0f),
                       /*cosInnerAngle=*/cosf(15.0f * (float)M_PI / 180.0f)},
    });

    // A directional ("sun") light - genuinely different in KIND from
    // both point-light entries above: parallel rays with no position
    // and no distance falloff at all, rather than one more delta
    // light radiating from a finite point (see metal_poc.metal's own
    // DirectionalLight comment). Aimed through the room's own open
    // front (the z=1 face has no wall - see the floor/ceiling/wall
    // addQuad() calls above): `direction` points mostly along -z with
    // a slight -y/+x tilt, so tracing back toward the light from
    // anywhere in the [-1,1]^3 room exits through that open face
    // before it would cross the ceiling (y=1) or either side wall,
    // rather than being trivially self-shadowed by this room's own
    // geometry on every shading point.
    const float3 sunColor = blackbodyColor(5778.0f) * 2.7f;
    // Same insert-at-front reasoning as `pointLights` just above - a plain
    // `=` here would wipe out any distant light loadPbrtScene() already
    // pushed.
    directionalLights.insert(directionalLights.begin(), {
        DirectionalLightData{PackedFloat3{0.1f, -0.15f, -1.0f}, PackedFloat3{sunColor.x, sunColor.y, sunColor.z}},
    });

    // A "slide projector" light - see metal_poc.metal's own
    // ProjectionLight comment. Mounted near the ceiling on the room's
    // centre-left, aimed down and across at the green (right, x=1)
    // wall's own lower-mid area - a plain, otherwise-undecorated flat
    // Lambertian surface (unlike the back wall, already carrying its
    // own mural texture), so the projected earthmap image reads as
    // unambiguously new rather than blending into existing detail.
    // Reuses `earthTexture` (already loaded for materialType 3's own
    // back-wall texture, see that PR's own comment) as the projected
    // image, rather than a separate asset - the same "a world map
    // projected like a slide" idea, just illuminating a wall instead
    // of decorating one. 32 degree (full vertical) FOV keeps this
    // reading as a defined projector "beam," not floodlighting the
    // whole wall; aspect 2.0 roughly matches earthmap.jpg's own
    // 2048x1025 (~2:1) proportions so the projected image isn't
    // visibly stretched.
    // insert(...begin(), ...) - same latent-wipe-bug reasoning as
    // disks/diskMaterials above (loadPbrtScene() doesn't populate this
    // vector today, but this proactively closes the same trap for
    // whenever it does).
    projectionLights.insert(projectionLights.begin(), {
        makeProjectionLight(/*position=*/float3{0.1f, 0.85f, -0.15f},
                             /*target=*/float3{1.0f, -0.05f, -0.15f},
                             /*worldUp=*/float3{0.0f, 1.0f, 0.0f},
                             /*fovDegrees=*/38.0f, /*aspect=*/2.0f, /*scale=*/25.0f),
    });

    // A goniometric ("IES-profile") light - see metal_poc.metal's own
    // GoniometricLight comment. Mounted near the ceiling on the room's
    // centre-right, aimed down and across at the RED (left, x=-1) wall's
    // own lower-mid area - a clean mirror of the projection light's own
    // placement on the green wall above, on the one remaining plain,
    // undecorated flat wall this scene has. 35-degree cosCutoff keeps
    // this reading as a defined beam, not floodlighting the whole wall;
    // ringFrequency=25 gives a handful of visible concentric bands within
    // that cone - tuned by rendering and inspecting (the same way every
    // other post-process/lighting knob in this POC was), not picked from
    // theory alone.
    const float3 goniometricColor = blackbodyColor(4500.0f) * 13.0f;
    // Same insert-at-front reasoning as projectionLights just above.
    goniometricLights.insert(goniometricLights.begin(), {
        makeGoniometricLight(/*position=*/float3{-0.1f, 0.85f, -0.15f},
                              /*target=*/float3{-1.0f, -0.05f, -0.15f},
                              /*worldUp=*/float3{0.0f, 1.0f, 0.0f},
                              /*emission=*/goniometricColor, /*scale=*/1.0f),
    });
    goniometricImageSize = 64;
    goniometricImage =
        buildGoniometricProfileImage(goniometricImageSize, /*cosCutoff=*/cosf(35.0f * (float)M_PI / 180.0f),
                                      /*ringFrequency=*/25.0f);

    // (Real pbrt scene loading, when pbrtScenePath is set, now happens
    // EARLIER - see the loadPbrtScene() call right before
    // buildPowerLightSampler() above, not here. A first version called
    // it here instead and it seemed to load correctly, but rendered
    // almost entirely black: buildPowerLightSampler()'s own alias table
    // is built from `lights` BEFORE this point in the function, so a
    // light appended here is invisible to every NEE draw - every direct-
    // lighting sample kept picking one of the two hardcoded room's own
    // (spatially unrelated) lights instead, never the genuinely nearby
    // one. Moving the call earlier, before that build call, was the
    // actual fix - not a lighting/exposure bug at all.)

    triangleCount = (uint32_t)materials.size();
    fprintf(stderr, "Scene: %u triangles, %zu spheres, %zu disks, %zu lights, %zu point lights, %zu directional lights, %zu projection lights, %zu goniometric lights\n",
            triangleCount, spheres.size(), disks.size(), lights.size(), pointLights.size(), directionalLights.size(),
            projectionLights.size(), goniometricLights.size());
}

// --- Stage 2.5: real pbrt scene loading - loadPbrtScene() and its 9
// phase methods now live in metal_poc_pbrt_loader.mm (pure code motion,
// see that file's own header comment) ------------------------------------

// See buildHandAuthoredScene()'s own declaration comment for the "which
// scene_ids" contract (cpu_scene_metal_hand_authored_supported(),
// cpu_interface.h) and section 116 (docs/METAL_GPU_FEASIBILITY.md).
bool MetalPocApp::buildHandAuthoredScene(const std::string& scene_id) {
    if (scene_id == "A1") {
        buildCornellBoxA1();
        return true;
    }
    if (scene_id == "G1") { buildStanfordBunny(); return true; }
    if (scene_id == "G2") { buildStanfordArmadillo(); return true; }
    if (scene_id == "G3") { buildStanfordHappyBuddha(); return true; }
    if (scene_id == "G4") { buildStanfordLucy(); return true; }
    if (scene_id == "G5") { buildStanfordDragon(); return true; }
    if (scene_id == "G6") { buildUtahTeapot(); return true; }
    if (scene_id == "G8") { buildSuzanneGallery(); return true; }
    if (scene_id == "G9") { buildNefertiti(); return true; }
    if (scene_id == "G11") { buildCheburashka(); return true; }
    if (scene_id == "G14") { buildBeast(); return true; }
    if (scene_id == "G15") { buildVWBeetle(); return true; }
    if (scene_id == "G17") { buildBimba(); return true; }
    if (scene_id == "G18") { buildCowGallery(); return true; }
    if (scene_id == "G19") { buildFandisk(); return true; }
    if (scene_id == "G20") { buildHomer(); return true; }
    if (scene_id == "G21") { buildIgea(); return true; }
    if (scene_id == "G22") { buildMaxPlanck(); return true; }
    if (scene_id == "G23") { buildOgre(); return true; }
    if (scene_id == "G24") { buildRockerArm(); return true; }
    if (scene_id == "G7") { buildSpotCow(); return true; }
    if (scene_id == "G10") { buildHorse(); return true; }
    if (scene_id == "G13") { buildGlassDragon(); return true; }
    if (scene_id == "G12") { buildTrophyRoom(); return true; }
    if (scene_id == "A3") { buildCheckeredSpheres(); return true; }
    if (scene_id == "A2") { buildBouncingSpheres(); return true; }
    if (scene_id == "A8") { buildCornellSmoke(); return true; }
    if (scene_id == "A6") { buildColoredQuads(); return true; }
    if (scene_id == "A4") { buildEarth(); return true; }
    if (scene_id == "A5") { buildPerlinSpheres(); return true; }
    if (scene_id == "A7") { buildSimpleLight(); return true; }
    if (scene_id == "B2") { buildCornellRoughMetal(); return true; }
    if (scene_id == "B4") { buildCornellConductor(); return true; }
    if (scene_id == "B3") { buildCornellRoughGlass(); return true; }
    if (scene_id == "B6") { buildCornellThinGlass(); return true; }
    if (scene_id == "B1") { buildRoughMetalSpheres(); return true; }
    if (scene_id == "B8") { buildCornellWaxSlab(); return true; }
    // Category I (Education) - several entries deliberately reuse ANOTHER
    // scene's own geometry verbatim under a different id/description,
    // pointing at a render-OPTION (sampler/integrator/exposure/light-
    // sampler/firefly-suppression) rather than new geometry at all (see
    // each one's own registry comment, scene_registry_data.h) - the
    // render-option toggle itself is a GUI/CPU-integrator concern, out of
    // scope here (Metal only ever runs its own fixed path tracer
    // regardless of scene_id), but the underlying SCENE is one this
    // backend already builds, so there's no reason not to claim it too.
    // I1/I4/I6/I7/I9 all use build_cornell_box (identical to A1); I5/I10
    // both use build_cornell_rough_glass (identical to B3). Section 132,
    // docs/METAL_GPU_FEASIBILITY.md.
    if (scene_id == "I1" || scene_id == "I4" || scene_id == "I6" ||
        scene_id == "I7" || scene_id == "I9") { buildCornellBoxA1(); return true; }
    if (scene_id == "I5" || scene_id == "I10") { buildCornellRoughGlass(); return true; }
    if (scene_id == "I8") { buildLightSamplerComparison(); return true; }
    if (scene_id == "C2") { buildSpotlightCornell(); return true; }
    if (scene_id == "C3") { buildDistantLightCornell(); return true; }
    if (scene_id == "C4") { buildPointLightCornell(); return true; }
    if (scene_id == "C5") { buildGoniometricLightCornell(); return true; }
    if (scene_id == "C6") { buildProjectionLightCornell(); return true; }
    if (scene_id == "F2") { buildTriangleMeshScene(); return true; }
    if (scene_id == "D5") { buildDepthOfFieldCornellBox(); return true; }
    if (scene_id == "D1") { buildDepthOfField(); return true; }
    if (scene_id == "D6") { buildOrthoCornellBox(); return true; }
    if (scene_id == "D2") { buildOrthoCameraScene(); return true; }
    if (scene_id == "D7") { buildSphericalCornellBox(); return true; }
    if (scene_id == "D3") { buildSphericalCameraScene(); return true; }
    if (scene_id == "D4") { buildRealisticCameraScene(); return true; }
    if (scene_id == "D8") { buildRealisticCornellBox(); return true; }
    if (scene_id == "D13") { buildCameraMotionBlurCornellBox(); return true; }
    if (scene_id == "F1") { buildBilinearPatchScene(); return true; }
    if (scene_id == "F4") { buildCurveFibersScene(); return true; }
    if (scene_id == "E1") { buildHomogeneousMediumScene(); return true; }
    if (scene_id == "E3") { buildDielectricMediumShowcase(); return true; }
    if (scene_id == "B9") { buildCornellCrystal(); return true; }
    if (scene_id == "B5") { buildCornellCoatedDiffuse(); return true; }
    if (scene_id == "B7") { buildCornellCoatedConductor(); return true; }
    if (scene_id == "B12") { buildNormalMappedCornell(); return true; }
    if (scene_id == "B23") { buildPrismDispersion(); return true; }
    if (scene_id == "B24") { buildPrismDispersionRough(); return true; }
    if (scene_id == "B10") { buildPrincipledShowcase(); return true; }
    if (scene_id == "C1") { buildHdriSky(); return true; }
    if (scene_id == "C7") { buildPortalLightScene(); return true; }
    // I3 (ExposureToneMapping): the SAME world/lights/sky as C1
    // (scene_registry_data.h's own comment: "Same world/lights/sky as
    // C1... GPU-compatible: see gpu/optix/scene_builder.cpp's case 134,
    // a near-verbatim copy of case 24, C1's own GPU case") - this
    // education scene is purely a Render-Options exercise (raise/lower
    // --exposure, compare --tonemap modes) against C1's own bright-sky/
    // shadowed-sphere geometry, not a different scene. Section 148.
    if (scene_id == "I3") { buildHdriSky(); return true; }
    // I2 (SpectralDispersionEducation): "Same glass prism as B23" verbatim
    // (scene_registry_data.h's own comment) - the missed entry from this
    // same Education-category batch, caught by a fresh scoping pass
    // (section 168). No new shader/materialType code needed - B23's own
    // materialType 22 dispersive dielectric already handles it.
    if (scene_id == "I2") { buildPrismDispersion(); return true; }
    fprintf(stderr, "buildHandAuthoredScene: scene '%s' has no real hand-authored builder yet - "
                    "this should not normally be reachable (metal_render_main()'s own gate "
                    "already checks cpu_scene_metal_hand_authored_supported() first).\n",
            scene_id.c_str());
    return false;
}

// The classic Cornell box (glass sphere + rotated white box) - scene A1,
// and (once cpu_scene_metal_hand_authored_supported()'s own list grows to
// include them - not yet, section 116) every other scene_id that reuses
// CPU's own build_cornell_box()/OptiX's own build_cornell_box() verbatim
// (several Cameras/Education-category scenes per scene_registry_data.h -
// same geometry, only id/category/description/camera differ).
//
// Reads src/shared/cornell_box_data.h directly - the SAME already-shared,
// backend-agnostic data both CPU's build_cornell_box() (scenes_book.h) and
// OptiX's build_cornell_box() (gpu/optix/scene_builder.cpp) already read,
// rather than re-deriving/re-typing the wall/box/sphere numbers a third
// time (that header's own comment explains why it exists at all - two
// independent hand-copies of this exact data already drifted apart once).

void MetalPocApp::applyCameraOverride(double cam_x, double cam_y, double cam_z) {
    const float3 rawLookfrom{(float)cam_x, (float)cam_y, (float)cam_z};
    const float3 lookfrom = (rawLookfrom - pbrtBboxCenter) * pbrtSceneScale + pbrtSceneOffset;
    const float3 forward = simd::normalize(pbrtCameraLookAtWorld - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, pbrtCameraUpRaw));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
}

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

// Every buffer/texture the compute encoder below binds can be nil if its
// own newBufferWith.../newTextureWithDescriptor call failed - OOM, or a
// requested length exceeding device.maxBufferLength (a real, finite,
// GPU-dependent ceiling a sufficiently large baked pbrt scene could
// plausibly hit, e.g. many ObjectInstance placements each duplicating
// full geometry rather than sharing one buffer - see section 86).
// Metal's own setBuffer:/setTexture: silently UNBIND that slot on a nil
// argument instead of erroring, so an unchecked failure here would let
// the shader read back zeroed/garbage data at that one binding and
// render a WRONG image with NO error anywhere - the hardest kind of bug
// to diagnose (found via a logging/debugging-focused review, not a
// symptom - see section 107). Deliberately does NOT re-check the
// acceleration-structure-only scratch/geometry buffers built earlier in
// this same stage (primScratch/sphereScratch/suzanneScratch/instScratch,
// boundingBoxBuffer/diskBoundingBoxBuffer/instanceBuffer) - those are
// never bound to THIS encoder, and a nil one there already fails loud
// via that build's own existing `buildCmd.status ==
// MTLCommandBufferStatusError` check just above it.
static void checkGpuResource(id resource, const char* name, id<MTLDevice> device, bool* anyFailed) {
    if (!resource) {
        fprintf(stderr, "GPU resource allocation FAILED: '%s' is nil (likely out of memory, "
                        "or this scene is too large for this GPU's max buffer length of %llu "
                        "bytes) - aborting before the render would silently produce a wrong "
                        "image instead of a visible error.\n",
                name, (unsigned long long)device.maxBufferLength);
        *anyFailed = true;
    }
}

// --- Stage 4: compile the shader, dispatch the render, read back -------
bool MetalPocApp::compileShaderAndDispatch(int argc, const char** argv) {
    // --- Compile the shader library from source at runtime ---------
    NSError* error = nil;
    // Three DIRECTORY candidates, tried in priority order (unchanged from
    // before the shader source was split into several files - see
    // metal_poc_shader_files.h's own comment for why every file below is
    // read from this SAME directory rather than resolved independently):
    // 1. Right next to the CURRENTLY RUNNING executable
    //    (_NSGetExecutablePath(), not NSBundle - resolves correctly for
    //    a plain (non-app-bundle) CLI binary too, which is exactly how
    //    a shipped .app's own Contents/MacOS/ray_tracer runs when the
    //    Qt GUI spawns it as a subprocess). This is the only candidate
    //    that works once the binary has been copied/installed anywhere
    //    other than the machine that built it - a real, previously-
    //    undiscovered bug found by actually testing a packaged release
    //    (section 110): RT_METAL_SHADER_DIR below is a compile-time
    //    absolute path into the BUILD MACHINE's own source tree, so a
    //    distributed .dmg's own bundled ray_tracer would report Metal
    //    available (metal_get_diagnostics() never touches this shader
    //    path at all) yet fail every actual GPU render once it got
    //    here, silently, on every machine except the one that built it.
    //    build_and_deploy_macos.sh now also copies every metal_poc_*.metal
    //    file next to the bundled ray_tracer specifically so this
    //    candidate finds them.
    // 2. RT_METAL_SHADER_DIR (set by CMakeLists.txt's metal_poc target,
    //    RT_BUILD_METAL=ON path) - gpu/metal/'s absolute SOURCE
    //    directory, correct only on the machine that built this binary
    //    (a plain `cmake --build` dev loop, never distributed).
    // 3. A __FILE__-relative lookup, for the ad-hoc `clang++
    //    metal_poc.mm ...` invocation this POC started as
    //    (docs/METAL_GPU_FEASIBILITY.md section 7/8/9).
    NSString* shaderDir = nil;
    {
        char exePathBuf[4096];
        uint32_t exePathSize = sizeof(exePathBuf);
        if (_NSGetExecutablePath(exePathBuf, &exePathSize) == 0) {
            NSString* exeDir = [@(exePathBuf) stringByDeletingLastPathComponent];
            int firstCount = 0;
            NSString* candidate = [exeDir stringByAppendingPathComponent:
                @(metalShaderFileNames(&firstCount)[0])];
            if ([[NSFileManager defaultManager] fileExistsAtPath:candidate]) shaderDir = exeDir;
        }
    }
    if (!shaderDir) {
#ifdef RT_METAL_SHADER_DIR
        shaderDir = @(RT_METAL_SHADER_DIR);
#else
        shaderDir = [@(__FILE__) stringByDeletingLastPathComponent];
#endif
    }
    // Concatenate every shader fragment, in metal_poc_shader_files.h's own
    // declared order, into ONE source string - Metal compiles from a
    // single in-memory string (newLibraryWithSource: below), so the split
    // into several files on disk is a source-organization change only,
    // not a real separate-translation-unit split the way the `.mm`
    // side's own per-category files are; every later file's own function
    // still needs every earlier file's own struct/function already
    // defined in the SAME string it's handed.
    NSMutableString* shaderSource = [NSMutableString string];
    int shaderFileCount = 0;
    const char* const* shaderFileNames = metalShaderFileNames(&shaderFileCount);
    for (int i = 0; i < shaderFileCount; ++i) {
        NSString* fragPath = [shaderDir stringByAppendingPathComponent:@(shaderFileNames[i])];
        NSString* fragSource = [NSString stringWithContentsOfFile:fragPath encoding:NSUTF8StringEncoding error:&error];
        if (!fragSource) {
            fprintf(stderr, "Failed to read shader source at %s: %s\n",
                fragPath.UTF8String, error.localizedDescription.UTF8String);
            return false;
        }
        [shaderSource appendString:fragSource];
    }
    MTLCompileOptions* compileOpts = [MTLCompileOptions new];
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:compileOpts error:&error];
    if (!library) {
        fprintf(stderr, "Shader compile failed: %s\n", error.localizedDescription.UTF8String);
        return false;
    }
    id<MTLFunction> kernelFn = [library newFunctionWithName:@"primaryRayKernel"];
    id<MTLFunction> sphereIntersectFn = [library newFunctionWithName:@"sphereIntersectionFunction"];
    id<MTLFunction> diskIntersectFn = [library newFunctionWithName:@"diskIntersectionFunction"];
    id<MTLFunction> cylinderIntersectFn = [library newFunctionWithName:@"cylinderIntersectionFunction"];

    // The intersection function has to be LINKED into the compute
    // pipeline (MTLLinkedFunctions) before an MTLIntersectionFunction
    // Table naming it can be built - a plain newComputePipelineState
    // WithFunction: (used for step 1/2's triangle-only pipeline) has
    // nowhere to put that linkage, hence the switch to the descriptor-
    // based pipeline creation call here.
    MTLComputePipelineDescriptor* pipelineDesc = [MTLComputePipelineDescriptor new];
    pipelineDesc.computeFunction = kernelFn;
    MTLLinkedFunctions* linkedFns = [MTLLinkedFunctions new];
    linkedFns.functions = @[sphereIntersectFn, diskIntersectFn, cylinderIntersectFn];
    pipelineDesc.linkedFunctions = linkedFns;

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:pipelineDesc
        options:MTLPipelineOptionNone reflection:nil error:&error];
    if (!pipeline) {
        fprintf(stderr, "Pipeline creation failed: %s\n", error.localizedDescription.UTF8String);
        return false;
    }

    // --- Intersection function table: two slots now, matching --------
    // bboxGeomDesc's own intersectionFunctionTableOffset (0) and
    // diskGeomDesc's (1) above - this POC's first real "one slot per
    // distinct intersection function" table, not just one slot
    // reused by every custom primitive. `setBuffer:atIndex:N` here
    // sets buffer N in the table's OWN shared argument namespace
    // (every function in ONE table draws from the same set of bound
    // buffers/textures) - sphereIntersectionFunction and
    // diskIntersectionFunction each declare a DIFFERENT `[[buffer(N)]]`
    // in their own MSL signature (0 and 1 respectively) specifically
    // so binding sphereBuffer at atIndex:0 and diskBuffer at
    // atIndex:1 here reaches the right function's own data, not a
    // shared/overwritten slot.
    MTLIntersectionFunctionTableDescriptor* fnTableDesc = [MTLIntersectionFunctionTableDescriptor new];
    fnTableDesc.functionCount = 3;
    id<MTLIntersectionFunctionTable> functionTable = [pipeline newIntersectionFunctionTableWithDescriptor:fnTableDesc];
    id<MTLFunctionHandle> sphereHandle = [pipeline functionHandleWithFunction:sphereIntersectFn];
    id<MTLFunctionHandle> diskHandle = [pipeline functionHandleWithFunction:diskIntersectFn];
    id<MTLFunctionHandle> cylinderHandle = [pipeline functionHandleWithFunction:cylinderIntersectFn];
    [functionTable setFunction:sphereHandle atIndex:0];
    [functionTable setFunction:diskHandle atIndex:1];
    [functionTable setFunction:cylinderHandle atIndex:2];
    // sphereIntersectionFunction/diskIntersectionFunction/
    // cylinderIntersectionFunction each read their own geometry buffer
    // (metal_poc.metal buffer(0)/buffer(1)/buffer(2) respectively - a
    // SEPARATE argument table from the calling kernel's own
    // buffer(0..27), see that file's own comment) - bound here, on the
    // function table, not on the compute encoder.
    [functionTable setBuffer:sphereBuffer offset:0 atIndex:0];
    [functionTable setBuffer:diskBuffer offset:0 atIndex:1];
    [functionTable setBuffer:cylinderBuffer offset:0 atIndex:2];
    // A8/section 176: sphereIntersectionFunction's own SpherePayload::
    // isShadowRay check needs sphereMaterials too, to tell a real
    // medium sphere (materialType 28) apart from an ordinary one - the
    // SAME sphereMaterialBuffer the calling kernel already binds at its
    // own buffer(4), bound a SECOND time here since this table has its
    // own independent argument namespace (diskGeomDesc's own comment).
    [functionTable setBuffer:sphereMaterialBuffer offset:0 atIndex:3];

    // --- Output texture + uniforms ----------------------------------
    MTLTextureDescriptor* texDesc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
        width:width height:height mipmapped:NO];
    texDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    texDesc.storageMode = MTLStorageModeShared;
    id<MTLTexture> outTexture = [device newTextureWithDescriptor:texDesc];

    // --- Earth texture (the back wall's materialType=3 source) -----
    // stb_image decodes straight to interleaved 8-bit RGBA regardless
    // of the source JPEG's channel count (the 4th `desiredChannels`
    // arg below), which is exactly MTLPixelFormatRGBA8Unorm_sRGB's own
    // BYTE layout - no repacking needed between stbi_load's buffer and
    // replaceRegion:. The `_sRGB` pixel format (not plain
    // `RGBA8Unorm`, this POC's own format up through PR #37) matters
    // for more than naming: an ordinary 8-bit JPEG/PNG's own stored
    // bytes are sRGB-gamma-ENCODED (perceptually, not linearly,
    // spaced) - every earlier render sampled those bytes directly as
    // if they were already linear radiance, silently darkening every
    // midtone the earth texture (and, via GI, everything it bounces
    // light onto) ever produced. `_sRGB` makes the texture SAMPLE
    // instruction itself convert sRGB to linear before the shader
    // ever sees a value - the standard, hardware-accelerated way to
    // do this, rather than a manual `pow(c, 2.2)` after sampling in
    // the shader.
    NSString* imagesDir = nil;
    {
        NSString* exeDir = executableDir();
        NSString* candidate = [exeDir stringByAppendingPathComponent:@"images"];
        if (exeDir && [[NSFileManager defaultManager] fileExistsAtPath:
                [candidate stringByAppendingPathComponent:@"earthmap.jpg"]]) {
            imagesDir = candidate;
        }
    }
    if (!imagesDir) {
#ifdef RT_MODELS_DIR
        imagesDir = [[@(RT_MODELS_DIR) stringByDeletingLastPathComponent]
            stringByAppendingPathComponent:@"images"];
#else
        imagesDir = [[@(__FILE__) stringByDeletingLastPathComponent]
            stringByAppendingPathComponent:@"../../images"];
#endif
    }
    NSString* earthPath = [imagesDir stringByAppendingPathComponent:@"earthmap.jpg"];
    int earthW = 0, earthH = 0, earthChannels = 0;
    unsigned char* earthPixels = stbi_load(earthPath.UTF8String, &earthW, &earthH, &earthChannels, 4);
    id<MTLTexture> earthTexture = nil;
    // Environment-map importance sampling (phase 2 - see section 69 for
    // phase 1's own host-side EnvDistribution2D, built and independently
    // verified there but not yet wired to anything). Built from the SAME
    // decoded `earthPixels` bytes right before they're freed below - the
    // shader-side NEE counterpart to `earthTexture`'s own miss-path
    // lookup (equirectangularUV()), which previously had no importance-
    // sampling strategy at all. Left default-constructed (empty arrays)
    // in the fallback/missing-JPEG case below; `envMapWidth`/
    // `envMapHeight` stay 0 in that case too, which the shader's own
    // Uniforms comment documents as "skip this NEE strategy entirely."
    EnvDistribution2D envDist;
    if (earthPixels) {
        MTLTextureDescriptor* earthDesc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB
            width:(NSUInteger)earthW height:(NSUInteger)earthH mipmapped:NO];
        earthDesc.usage = MTLTextureUsageShaderRead;
        earthDesc.storageMode = MTLStorageModeShared;
        earthTexture = [device newTextureWithDescriptor:earthDesc];
        MTLRegion earthRegion = MTLRegionMake2D(0, 0, (NSUInteger)earthW, (NSUInteger)earthH);
        [earthTexture replaceRegion:earthRegion mipmapLevel:0 withBytes:earthPixels
            bytesPerRow:(NSUInteger)earthW * 4];
        buildEnvDistribution2D(earthPixels, earthW, earthH, envDist);
        stbi_image_free(earthPixels);
        fprintf(stderr, "Loaded %s: %dx%d, %d channels\n", earthPath.UTF8String, earthW, earthH, earthChannels);
    } else {
        fprintf(stderr, "Could not load %s - back wall will read black/undefined texture data.\n",
            earthPath.UTF8String);
        // A 1x1 white fallback keeps the shader's unconditional
        // texture bind valid (Metal requires SOME texture at the
        // bound slot) even if the JPEG is missing. `_sRGB` for
        // consistency with the real texture above, though pure white
        // (255,255,255) round-trips through the sRGB<->linear
        // conversion unchanged either way.
        MTLTextureDescriptor* fallbackDesc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB width:1 height:1 mipmapped:NO];
        fallbackDesc.usage = MTLTextureUsageShaderRead;
        fallbackDesc.storageMode = MTLStorageModeShared;
        earthTexture = [device newTextureWithDescriptor:fallbackDesc];
        uint8_t white4[4] = {255, 255, 255, 255};
        [earthTexture replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:white4 bytesPerRow:4];
    }

    // A pbrt-loaded scene's own image-based infinite light - a
    // genuinely SEPARATE texture from earthTexture above (see
    // loadPbrtScene()'s own comment on why). pbrtEnvImagePixels is
    // already decoded, linear float RGB (3 floats/pixel) - stored
    // straight into an RGBA32Float texture (padding alpha=1, no sRGB
    // format/decode needed at all: unlike earthTexture's own 8-bit JPEG
    // source, there's no gamma curve to reverse here). Falls back to a
    // 1x1 black texture (Metal requires SOME texture bound at every
    // used slot) when the scene has no image-based infinite light -
    // harmless, since pbrtHasImageEnvLight gates whether the shader
    // ever actually samples it.
    id<MTLTexture> pbrtEnvTexture = nil;
    {
        const uint32_t pw = havePbrtImageEnvLight ? (uint32_t)pbrtEnvImageWidth : 1u;
        const uint32_t ph = havePbrtImageEnvLight ? (uint32_t)pbrtEnvImageHeight : 1u;
        MTLTextureDescriptor* pbrtEnvDesc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
            width:pw height:ph mipmapped:NO];
        pbrtEnvDesc.usage = MTLTextureUsageShaderRead;
        pbrtEnvDesc.storageMode = MTLStorageModeShared;
        pbrtEnvTexture = [device newTextureWithDescriptor:pbrtEnvDesc];
        std::vector<float> rgba((size_t)pw * ph * 4, 0.0f);
        if (havePbrtImageEnvLight) {
            for (size_t i = 0; i < (size_t)pw * ph; ++i) {
                rgba[i * 4 + 0] = pbrtEnvImagePixels[i * 3 + 0];
                rgba[i * 4 + 1] = pbrtEnvImagePixels[i * 3 + 1];
                rgba[i * 4 + 2] = pbrtEnvImagePixels[i * 3 + 2];
                rgba[i * 4 + 3] = 1.0f;
            }
        }
        [pbrtEnvTexture replaceRegion:MTLRegionMake2D(0, 0, pw, ph) mipmapLevel:0
            withBytes:rgba.data() bytesPerRow:(NSUInteger)pw * 4 * sizeof(float)];
    }

    // A pbrt-loaded scene's own real per-light goniometric/projection
    // profile images (section 98) - same upload pattern as pbrtEnvTexture
    // above (already-decoded linear float RGB -> RGBA32Float, 1x1 black
    // fallback when there's no such light, harmless since usePbrtTexture
    // gates whether any light actually reads it).
    id<MTLTexture> pbrtGoniometricTexture = nil;
    {
        const uint32_t pw = havePbrtGoniometricImage ? (uint32_t)pbrtGoniometricImageWidth : 1u;
        const uint32_t ph = havePbrtGoniometricImage ? (uint32_t)pbrtGoniometricImageHeight : 1u;
        MTLTextureDescriptor* desc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
            width:pw height:ph mipmapped:NO];
        desc.usage = MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared;
        pbrtGoniometricTexture = [device newTextureWithDescriptor:desc];
        std::vector<float> rgba((size_t)pw * ph * 4, 0.0f);
        if (havePbrtGoniometricImage) {
            for (size_t i = 0; i < (size_t)pw * ph; ++i) {
                rgba[i * 4 + 0] = pbrtGoniometricImagePixels[i * 3 + 0];
                rgba[i * 4 + 1] = pbrtGoniometricImagePixels[i * 3 + 1];
                rgba[i * 4 + 2] = pbrtGoniometricImagePixels[i * 3 + 2];
                rgba[i * 4 + 3] = 1.0f;
            }
        }
        [pbrtGoniometricTexture replaceRegion:MTLRegionMake2D(0, 0, pw, ph) mipmapLevel:0
            withBytes:rgba.data() bytesPerRow:(NSUInteger)pw * 4 * sizeof(float)];
    }

    id<MTLTexture> pbrtProjectionTexture = nil;
    {
        const uint32_t pw = havePbrtProjectionImage ? (uint32_t)pbrtProjectionImageWidth : 1u;
        const uint32_t ph = havePbrtProjectionImage ? (uint32_t)pbrtProjectionImageHeight : 1u;
        MTLTextureDescriptor* desc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
            width:pw height:ph mipmapped:NO];
        desc.usage = MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared;
        pbrtProjectionTexture = [device newTextureWithDescriptor:desc];
        std::vector<float> rgba((size_t)pw * ph * 4, 0.0f);
        if (havePbrtProjectionImage) {
            for (size_t i = 0; i < (size_t)pw * ph; ++i) {
                rgba[i * 4 + 0] = pbrtProjectionImagePixels[i * 3 + 0];
                rgba[i * 4 + 1] = pbrtProjectionImagePixels[i * 3 + 1];
                rgba[i * 4 + 2] = pbrtProjectionImagePixels[i * 3 + 2];
                rgba[i * 4 + 3] = 1.0f;
            }
        }
        [pbrtProjectionTexture replaceRegion:MTLRegionMake2D(0, 0, pw, ph) mipmapLevel:0
            withBytes:rgba.data() bytesPerRow:(NSUInteger)pw * 4 * sizeof(float)];
    }

    // A pbrt-loaded scene's own image-based AreaLightSource (section
    // 105) - same upload pattern as pbrtGoniometricTexture/
    // pbrtProjectionTexture above.
    id<MTLTexture> pbrtAreaLightTexture = nil;
    {
        const uint32_t pw = havePbrtAreaLightImage ? (uint32_t)pbrtAreaLightImageWidth : 1u;
        const uint32_t ph = havePbrtAreaLightImage ? (uint32_t)pbrtAreaLightImageHeight : 1u;
        MTLTextureDescriptor* desc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
            width:pw height:ph mipmapped:NO];
        desc.usage = MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared;
        pbrtAreaLightTexture = [device newTextureWithDescriptor:desc];
        std::vector<float> rgba((size_t)pw * ph * 4, 0.0f);
        if (havePbrtAreaLightImage) {
            for (size_t i = 0; i < (size_t)pw * ph; ++i) {
                rgba[i * 4 + 0] = pbrtAreaLightImagePixels[i * 3 + 0];
                rgba[i * 4 + 1] = pbrtAreaLightImagePixels[i * 3 + 1];
                rgba[i * 4 + 2] = pbrtAreaLightImagePixels[i * 3 + 2];
                rgba[i * 4 + 3] = 1.0f;
            }
        }
        [pbrtAreaLightTexture replaceRegion:MTLRegionMake2D(0, 0, pw, ph) mipmapLevel:0
            withBytes:rgba.data() bytesPerRow:(NSUInteger)pw * 4 * sizeof(float)];
    }

    // A pbrt-loaded Diffuse/CoatedDiffuse material's own real imagemap-
    // bound "texture reflectance" (F5/F9, section 166) - same upload
    // pattern as pbrtAreaLightTexture/pbrtGoniometricTexture above.
    id<MTLTexture> pbrtDiffuseTexture = nil;
    {
        const uint32_t pw = havePbrtDiffuseImage ? (uint32_t)pbrtDiffuseImageWidth : 1u;
        const uint32_t ph = havePbrtDiffuseImage ? (uint32_t)pbrtDiffuseImageHeight : 1u;
        MTLTextureDescriptor* desc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
            width:pw height:ph mipmapped:NO];
        desc.usage = MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared;
        pbrtDiffuseTexture = [device newTextureWithDescriptor:desc];
        std::vector<float> rgba((size_t)pw * ph * 4, 0.0f);
        if (havePbrtDiffuseImage) {
            for (size_t i = 0; i < (size_t)pw * ph; ++i) {
                rgba[i * 4 + 0] = pbrtDiffuseImagePixels[i * 3 + 0];
                rgba[i * 4 + 1] = pbrtDiffuseImagePixels[i * 3 + 1];
                rgba[i * 4 + 2] = pbrtDiffuseImagePixels[i * 3 + 2];
                rgba[i * 4 + 3] = 1.0f;
            }
        }
        [pbrtDiffuseTexture replaceRegion:MTLRegionMake2D(0, 0, pw, ph) mipmapLevel:0
            withBytes:rgba.data() bytesPerRow:(NSUInteger)pw * 4 * sizeof(float)];
    }

    // envMarginalCDF/envConditionalCDF buffers - a real (non-empty)
    // envDist above uploads its own arrays directly; the fallback case
    // (missing JPEG) still needs SOME buffer bound at these indices
    // (Metal requires a real resource at every declared buffer slot,
    // the same reason earthTexture's own fallback path exists above),
    // hence the single-float dummy - `envMapWidth`/`envMapHeight`
    // staying 0 is what actually keeps the shader from ever reading
    // past it.
    uint32_t envMapWidth = 0, envMapHeight = 0;
    id<MTLBuffer> envMarginalCDFBuffer;
    id<MTLBuffer> envConditionalCDFBuffer;
    if (!envDist.marginalCDF.empty()) {
        envMarginalCDFBuffer = [device newBufferWithBytes:envDist.marginalCDF.data()
            length:envDist.marginalCDF.size() * sizeof(float) options:MTLResourceStorageModeShared];
        envConditionalCDFBuffer = [device newBufferWithBytes:envDist.conditionalCDF.data()
            length:envDist.conditionalCDF.size() * sizeof(float) options:MTLResourceStorageModeShared];
        envMapWidth = (uint32_t)envDist.width;
        envMapHeight = (uint32_t)envDist.height;
    } else {
        float dummy = 0.0f;
        envMarginalCDFBuffer = [device newBufferWithBytes:&dummy length:sizeof(float) options:MTLResourceStorageModeShared];
        envConditionalCDFBuffer = [device newBufferWithBytes:&dummy length:sizeof(float) options:MTLResourceStorageModeShared];
    }

    // pbrtEnvMarginalCDF/pbrtEnvConditionalCDF buffers (section 96) - the
    // NEE-importance-sampling counterpart to pbrtEnvTexture above, same
    // idea as envMarginalCDF/envConditionalCDF just above but for a
    // SEPARATE image (a pbrt-loaded scene's own image-based infinite
    // light, not earthTexture). Uses the float-RGB buildEnvDistribution2D()
    // overload directly on pbrtEnvImagePixels (already linear, already
    // decoded - no srgbByteToLinear() round-trip needed the way
    // earthPixels' own RGBA8 bytes require). Empty/dummy fallback when
    // there's no image-based infinite light, same "0 disables it" pattern
    // as envMapWidth's own fallback.
    EnvDistribution2D pbrtEnvDist;
    if (havePbrtImageEnvLight) {
        buildEnvDistribution2D(pbrtEnvImagePixels.data(), pbrtEnvImageWidth, pbrtEnvImageHeight, pbrtEnvDist);
    }
    uint32_t pbrtEnvMapWidth = 0, pbrtEnvMapHeight = 0;
    id<MTLBuffer> pbrtEnvMarginalCDFBuffer;
    id<MTLBuffer> pbrtEnvConditionalCDFBuffer;
    if (!pbrtEnvDist.marginalCDF.empty()) {
        pbrtEnvMarginalCDFBuffer = [device newBufferWithBytes:pbrtEnvDist.marginalCDF.data()
            length:pbrtEnvDist.marginalCDF.size() * sizeof(float) options:MTLResourceStorageModeShared];
        pbrtEnvConditionalCDFBuffer = [device newBufferWithBytes:pbrtEnvDist.conditionalCDF.data()
            length:pbrtEnvDist.conditionalCDF.size() * sizeof(float) options:MTLResourceStorageModeShared];
        pbrtEnvMapWidth = (uint32_t)pbrtEnvDist.width;
        pbrtEnvMapHeight = (uint32_t)pbrtEnvDist.height;
    } else {
        float pbrtEnvDummy = 0.0f;
        pbrtEnvMarginalCDFBuffer = [device newBufferWithBytes:&pbrtEnvDummy length:sizeof(float) options:MTLResourceStorageModeShared];
        pbrtEnvConditionalCDFBuffer = [device newBufferWithBytes:&pbrtEnvDummy length:sizeof(float) options:MTLResourceStorageModeShared];
    }

    // GGX multi-scatter energy-compensation table (phase 2 - see phase
    // 1's own header comment in metal_poc_host_math.h for the full
    // "why," found by spot-checking this POC's GGX conductor material
    // against Blender Cycles as a second reference). Built ONCE here at
    // startup (a real-time Monte Carlo precompute, not Cycles' own
    // offline 8-64-million-sample tool) and uploaded as a single GPU
    // buffer - only `E` is needed on the device side; `Eavg` only feeds
    // the "multi-bounce Fresnel darkening" refinement this phase
    // deliberately doesn't attempt (see that same header comment).
    // std::mt19937 (not this POC's own device-side randFloat()) is fine
    // here - this is a host-only precompute of a smooth, low-frequency
    // table, not a per-pixel render decision that needs to match the
    // shader's own RNG stream bit-for-bit.
    std::mt19937 ggxEnergyRng(1337);
    std::uniform_real_distribution<float> ggxEnergyDist(0.0f, 1.0f);
    auto ggxEnergyRandFn = [&]() { return ggxEnergyDist(ggxEnergyRng); };
    GGXEnergyTable ggxEnergyTable;
    buildGGXEnergyTable(/*roughRes=*/32, /*muRes=*/32, /*samplesPerCell=*/2048,
                        ggxEnergyTable, ggxEnergyRandFn);
    id<MTLBuffer> ggxEnergyTableBuffer = [device newBufferWithBytes:ggxEnergyTable.E.data()
        length:ggxEnergyTable.E.size() * sizeof(float) options:MTLResourceStorageModeShared];

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
    uniforms.lightCount = (uint32_t)lights.size();
    // Thin-lens depth of field: focused on the gold conductor sphere
    // (the nearest object to the camera), so it renders pixel-sharp
    // while the dielectric sphere just behind it and the back
    // wall/Suzanne further back show progressively more defocus blur -
    // the falloff is what actually demonstrates this is a real lens
    // model, not just a uniform blur filter over the whole frame.
    uniforms.lensRadius = 0.05f;
    uniforms.focusDistance = uniforms.cameraPos.z - spheres[1].center.z; // gold sphere's own z
    // A hexagonal (6-blade) aperture rather than a perfectly circular
    // one - see Uniforms' own apertureBlades comment. The classic
    // photographic blade count; out-of-focus highlights (area/point/
    // spot light reflections on the defocused back-wall geometry)
    // should now read as hexagons, not perfect circles.
    uniforms.apertureBlades = 6;
    // Shutter motion blur: a small horizontal dolly over the frame's
    // simulated exposure - chosen (over, say, an object moving) since
    // it needs no acceleration-structure/intersection-function
    // changes at all, purely a primary-ray-generation addition (see
    // metal_poc.metal's own comment on why: a moving CUSTOM primitive
    // would need per-sample time threaded into
    // sphereIntersectionFunction's own, separate argument table, real
    // additional Metal API surface this increment intentionally
    // doesn't take on).
    uniforms.cameraVelocity = PackedFloat3{0.015f, 0.0f, 0.0f};
    // Homogeneous fog filling the whole room - subtle (transmittance
    // ~0.7 over the ~4-unit camera-to-back-wall sightline: exp(-0.08*4)
    // ~ 0.73), meant to read as a light atmospheric haze visible in
    // the light shafts/depth falloff, not an opaque room-filling mist
    // that would fight every other material's own visibility.
    uniforms.fogSigmaT = 0.05f;
    uniforms.fogAlbedo = PackedFloat3{0.85f, 0.88f, 0.95f}; // mostly-scattering, faint cool tint
    // On: a miss ray samples earthTexture by direction (equirectangular)
    // instead of the flat two-colour gradient - the room's open front
    // means most miss rays are secondary/GI bounces (a mirror/glass
    // surface reflecting/refracting outward), not primary camera rays,
    // so this mostly shows up subtly rather than as an obvious visible
    // backdrop - see docs/METAL_GPU_FEASIBILITY.md's own note on
    // verifying this with a dedicated wide-FOV test render.
    uniforms.useEnvironmentMap = 1u;
    // Moderate forward scattering (real fog/haze skews strongly
    // forward in reality - Mie scattering off water droplets often
    // has g around 0.7-0.9 - 0.4 is deliberately more modest, so the
    // difference from isotropic reads as a stylistic tint on the fog
    // rather than a dramatic visible change).
    uniforms.fogAsymmetryG = 0.4f;
    uniforms.pointLightCount = (uint32_t)pointLights.size();
    uniforms.directionalLightCount = (uint32_t)directionalLights.size();
    uniforms.projectionLightCount = (uint32_t)projectionLights.size();
    uniforms.goniometricLightCount = (uint32_t)goniometricLights.size();
    // Single-pass adaptive sampling (see metal_poc.metal's own
    // shading-loop comment) - enabled by default for this scene:
    // converged pixels (most of the flat-coloured walls/ceiling)
    // stop well short of the full samplesPerPixel budget, spending
    // it instead on the noisier fog/specular/caustic regions this
    // scene already has plenty of - a real render-TIME win at
    // (ideally) no visible quality cost, verified via a dedicated
    // A/B render, not assumed.
    uniforms.adaptiveSampling = 1u;
    uniforms.pbrtEnvMapWidth = pbrtEnvMapWidth;
    uniforms.pbrtEnvMapHeight = pbrtEnvMapHeight;
    uniforms.envMapWidth = envMapWidth;
    uniforms.envMapHeight = envMapHeight;
    uniforms.ggxEnergyRoughRes = (uint32_t)ggxEnergyTable.roughRes;
    uniforms.ggxEnergyMuRes = (uint32_t)ggxEnergyTable.muRes;

    if (havePbrtCamera) {
        // A real pbrt scene was loaded (loadPbrtScene()) - override every
        // scale/placement-dependent field the defaults above assumed,
        // all tuned for the hardcoded room's own [-1,1] scale, not a
        // real pbrt scene's own (often much larger - e.g. a ~500-unit
        // classic Cornell box) world scale. fogSigmaT=0.05 alone would
        // otherwise make an 800-unit sightline read as solid black
        // (exp(-0.05*800) ~ 0) - not a subtle atmospheric tweak, a
        // completely broken render.
        uniforms.cameraPos = PackedFloat3{pbrtCameraPos.x, pbrtCameraPos.y, pbrtCameraPos.z};
        uniforms.cameraForward = PackedFloat3{pbrtCameraForward.x, pbrtCameraForward.y, pbrtCameraForward.z};
        uniforms.cameraRight = PackedFloat3{pbrtCameraRight.x, pbrtCameraRight.y, pbrtCameraRight.z};
        uniforms.cameraUp = PackedFloat3{pbrtCameraUp.x, pbrtCameraUp.y, pbrtCameraUp.z};
        uniforms.tanHalfFov = pbrtTanHalfFov;
        // D13 (section 174): real orbit-style camera motion blur, gated
        // on the scene's own sceneCameraVelocity actually being nonzero
        // (every other scene leaves it at its own {0,0,0} default, so
        // this is 0/false for all of them - a provable no-op). Always
        // copies pbrtCameraLookAtWorld/pbrtCameraUpRaw regardless -
        // both already exist for applyCameraOverride()'s own sake, and
        // are simply unread by the shader whenever hasCameraOrbitBlur
        // is 0.
        uniforms.cameraLookAtBlur = PackedFloat3{pbrtCameraLookAtWorld.x, pbrtCameraLookAtWorld.y, pbrtCameraLookAtWorld.z};
        uniforms.cameraUpRawBlur = PackedFloat3{pbrtCameraUpRaw.x, pbrtCameraUpRaw.y, pbrtCameraUpRaw.z};
        uniforms.hasCameraOrbitBlur = (sceneCameraVelocity.x != 0.0f || sceneCameraVelocity.y != 0.0f ||
                                        sceneCameraVelocity.z != 0.0f) ? 1u : 0u;
        // pbrt v1 still doesn't read a pbrt FILE's own "float lensradius"/
        // "float focaldistance" Camera parameters (a genuinely separate,
        // still-open gap) - but a hand-authored scene (D5, section 137)
        // has no pbrt file to parse from at all, so its own
        // pbrtLensRadius/pbrtFocusDistance (defaulting to 0/1, "no DOF" -
        // every scene before D5) are read directly here instead.
        // focusDistance is unread by the shader whenever lensRadius == 0.
        uniforms.lensRadius = pbrtLensRadius;
        uniforms.focusDistance = pbrtFocusDistance;
        // See MetalPocApp::havePbrtOrthographic's own comment - reuses
        // pbrtTanHalfFov (already copied to uniforms.tanHalfFov above)
        // as the orthographic screen window's own half-extent.
        uniforms.cameraOrthographic = havePbrtOrthographic ? 1u : 0u;
        // See MetalPocApp::havePbrtSpherical's own comment.
        uniforms.cameraSpherical = havePbrtSpherical ? 1u : 0u;
        // See MetalPocApp::havePbrtSphericalEqualArea's own comment.
        uniforms.sphericalMappingEqualArea = havePbrtSphericalEqualArea ? 1u : 0u;
        // See MetalPocApp::havePbrtRealisticCamera's own comment.
        uniforms.cameraRealistic = havePbrtRealisticCamera ? 1u : 0u;
        uniforms.numLensElements = (uint32_t)realisticLensElements.size();
        uniforms.numExitPupilBounds = (uint32_t)realisticExitPupilBounds.size();
        uniforms.filmHalfX = realisticFilmHalfX;
        uniforms.filmHalfY = realisticFilmHalfY;
        uniforms.lensRearZ = realisticLensRearZ;
        // Adaptive sampling (enabled by default just above,
        // uniforms.adaptiveSampling=1) OFF for a real multi-element-lens
        // camera - a real, found-by-rendering bug, not a style choice.
        // Its convergence test (metal_poc.metal's own shading-loop
        // comment) checks standardError/mean against a fixed threshold,
        // which assumes a roughly UNIMODAL per-sample radiance
        // distribution (true for every earlier camera mode's own noise,
        // which is why the heuristic was tuned/validated against those).
        // A real lens's own per-sample radiance is instead strongly
        // BIMODAL: most exit-pupil samples land outside the true
        // (non-rectangular) aperture and get vignetted to EXACTLY 0
        // (sampleRealisticCameraRay's own comment), while the rest carry
        // the entire `cameraWeight`-scaled contribution. Many exact
        // zeros pull the running mean/variance down together, so the
        // ratio can cross the threshold after just a handful of samples
        // even though the NONZERO tail is still wildly undersampled -
        // confirmed empirically: with adaptive sampling on, a 3000spp
        // D4 render finished in ~4s (barely slower than 100spp) and was
        // JUST as speckled; forcing every sample through to the full
        // requested budget (this override) made that same 3000spp
        // render converge to a clean, smooth image. The visible
        // "speckle" itself is this per-pixel undersampling residual,
        // amplified into colour fringing by postProcessAndWrite()'s own
        // always-on chromatic aberration (a small, fixed per-channel
        // pixel-space shift - invisible on an already-smooth image, but
        // it visibly decorrelates R/G/B on a noisy one).
        if (havePbrtRealisticCamera) uniforms.adaptiveSampling = 0u;
        // D13 (section 174): reads the scene's own sceneCameraVelocity
        // instead of unconditionally zeroing this out - see that
        // member's own comment. {0,0,0} (every scene but D13) is an
        // exact no-op, identical to this line's own previous literal.
        uniforms.cameraVelocity = sceneCameraVelocity;
        if (havePbrtMedium) {
            uniforms.fogSigmaT = pbrtFogSigmaT;
            uniforms.fogAlbedo = PackedFloat3{pbrtFogAlbedo.x, pbrtFogAlbedo.y, pbrtFogAlbedo.z};
            uniforms.fogAsymmetryG = pbrtFogAsymmetryG;
        } else {
            uniforms.fogSigmaT = 0.0f;                      // no participating medium in this scene
        }
        uniforms.useEnvironmentMap = 0u;                    // this scene's own sky, if any, replaces the hardcoded room's earthTexture-based one below
        // See Uniforms::isPbrtScene's own comment (metal_poc_types.metal)
        // - a real pbrt scene with no infinite light gets a BLACK
        // miss-path background, not the hardcoded room's own sky
        // gradient. Deliberately `!pbrtScenePath.empty()`, NOT the
        // broader `havePbrtCamera` this whole block is already gated
        // on - havePbrtCamera is true for every HAND-AUTHORED scene's
        // own camera setup too (buildCornellBoxA1() and nearly every
        // other builder set it), not just a genuinely loaded pbrt FILE.
        // A first version of this fix used havePbrtCamera directly and
        // made A1/G1/every other hand-authored scene's own background
        // incorrectly black too (caught by this PR's own before/after
        // hash sweep across EVERY scene, not just the pbrt-file ones
        // this fix was meant for - exactly the discipline that check
        // exists to catch).
        uniforms.isPbrtScene = pbrtScenePath.empty() ? 0u : 1u;
        if (havePbrtConstantEnvLight) {
            uniforms.pbrtHasConstantEnvLight = 1u;
            uniforms.pbrtEnvColor = PackedFloat3{pbrtEnvColor.x, pbrtEnvColor.y, pbrtEnvColor.z};
        } else if (havePbrtImageEnvLight) {
            uniforms.pbrtHasImageEnvLight = 1u;
        }
    }

    id<MTLBuffer> uniformBuffer = [device newBufferWithBytes:&uniforms length:sizeof(Uniforms) options:MTLResourceStorageModeShared];

    // Checked once, here, right before they're all bound below - see
    // checkGpuResource()'s own comment for why this one spot (not each
    // allocation call site above) and why the AS-only scratch/geometry
    // buffers aren't included.
    bool anyResourceFailed = false;
    checkGpuResource(uniformBuffer, "uniformBuffer", device, &anyResourceFailed);
    checkGpuResource(materialBuffer, "materialBuffer", device, &anyResourceFailed);
    checkGpuResource(vertexBuffer, "vertexBuffer", device, &anyResourceFailed);
    checkGpuResource(sphereMaterialBuffer, "sphereMaterialBuffer", device, &anyResourceFailed);
    checkGpuResource(sphereBuffer, "sphereBuffer", device, &anyResourceFailed);
    checkGpuResource(normalBuffer, "normalBuffer", device, &anyResourceFailed);
    checkGpuResource(uvBuffer, "uvBuffer", device, &anyResourceFailed);
    checkGpuResource(lightBuffer, "lightBuffer", device, &anyResourceFailed);
    checkGpuResource(suzanneNormalBuffer, "suzanneNormalBuffer", device, &anyResourceFailed);
    checkGpuResource(suzanneMaterialBuffer, "suzanneMaterialBuffer", device, &anyResourceFailed);
    checkGpuResource(instanceTransformBuffer, "instanceTransformBuffer", device, &anyResourceFailed);
    checkGpuResource(diskBuffer, "diskBuffer", device, &anyResourceFailed);
    checkGpuResource(diskMaterialBuffer, "diskMaterialBuffer", device, &anyResourceFailed);
    checkGpuResource(cylinderBuffer, "cylinderBuffer", device, &anyResourceFailed);
    checkGpuResource(cylinderMaterialBuffer, "cylinderMaterialBuffer", device, &anyResourceFailed);
    checkGpuResource(pointLightBuffer, "pointLightBuffer", device, &anyResourceFailed);
    checkGpuResource(directionalLightBuffer, "directionalLightBuffer", device, &anyResourceFailed);
    checkGpuResource(projectionLightBuffer, "projectionLightBuffer", device, &anyResourceFailed);
    checkGpuResource(goniometricLightBuffer, "goniometricLightBuffer", device, &anyResourceFailed);
    checkGpuResource(lensElementBuffer, "lensElementBuffer", device, &anyResourceFailed);
    checkGpuResource(exitPupilBoundsBuffer, "exitPupilBoundsBuffer", device, &anyResourceFailed);
    checkGpuResource(envMarginalCDFBuffer, "envMarginalCDFBuffer", device, &anyResourceFailed);
    checkGpuResource(envConditionalCDFBuffer, "envConditionalCDFBuffer", device, &anyResourceFailed);
    checkGpuResource(ggxEnergyTableBuffer, "ggxEnergyTableBuffer", device, &anyResourceFailed);
    checkGpuResource(pbrtEnvMarginalCDFBuffer, "pbrtEnvMarginalCDFBuffer", device, &anyResourceFailed);
    checkGpuResource(pbrtEnvConditionalCDFBuffer, "pbrtEnvConditionalCDFBuffer", device, &anyResourceFailed);
    checkGpuResource(outTexture, "outTexture", device, &anyResourceFailed);
    checkGpuResource(earthTexture, "earthTexture", device, &anyResourceFailed);
    checkGpuResource(goniometricTexture, "goniometricTexture", device, &anyResourceFailed);
    checkGpuResource(pbrtEnvTexture, "pbrtEnvTexture", device, &anyResourceFailed);
    checkGpuResource(pbrtGoniometricTexture, "pbrtGoniometricTexture", device, &anyResourceFailed);
    checkGpuResource(pbrtProjectionTexture, "pbrtProjectionTexture", device, &anyResourceFailed);
    checkGpuResource(pbrtAreaLightTexture, "pbrtAreaLightTexture", device, &anyResourceFailed);
    if (anyResourceFailed) return false;

    // --- Dispatch ----------------------------------------------------
    id<MTLCommandBuffer> renderCmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [renderCmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setTexture:outTexture atIndex:0];
    [enc setTexture:earthTexture atIndex:1];
    [enc setTexture:goniometricTexture atIndex:2];
    [enc setTexture:pbrtEnvTexture atIndex:3];
    [enc setTexture:pbrtGoniometricTexture atIndex:4];
    [enc setTexture:pbrtProjectionTexture atIndex:5];
    [enc setTexture:pbrtAreaLightTexture atIndex:6];
    [enc setTexture:pbrtDiffuseTexture atIndex:7];
    [enc setAccelerationStructure:instAS atBufferIndex:0];
    [enc setBuffer:uniformBuffer offset:0 atIndex:1];
    [enc setBuffer:materialBuffer offset:0 atIndex:2];
    [enc setBuffer:vertexBuffer offset:0 atIndex:3];
    [enc setBuffer:sphereMaterialBuffer offset:0 atIndex:4];
    [enc setBuffer:sphereBuffer offset:0 atIndex:5];
    [enc setIntersectionFunctionTable:functionTable atBufferIndex:6];
    [enc setBuffer:normalBuffer offset:0 atIndex:7];
    [enc setBuffer:uvBuffer offset:0 atIndex:8];
    [enc setBuffer:lightBuffer offset:0 atIndex:9];
    [enc setBuffer:suzanneNormalBuffer offset:0 atIndex:10];
    [enc setBuffer:suzanneMaterialBuffer offset:0 atIndex:11];
    [enc setBuffer:instanceTransformBuffer offset:0 atIndex:12];
    [enc setBuffer:diskBuffer offset:0 atIndex:13];
    [enc setBuffer:diskMaterialBuffer offset:0 atIndex:14];
    [enc setBuffer:pointLightBuffer offset:0 atIndex:15];
    [enc setBuffer:directionalLightBuffer offset:0 atIndex:16];
    [enc setBuffer:projectionLightBuffer offset:0 atIndex:17];
    [enc setBuffer:goniometricLightBuffer offset:0 atIndex:18];
    [enc setBuffer:envMarginalCDFBuffer offset:0 atIndex:19];
    [enc setBuffer:envConditionalCDFBuffer offset:0 atIndex:20];
    [enc setBuffer:ggxEnergyTableBuffer offset:0 atIndex:21];
    [enc setBuffer:pbrtEnvMarginalCDFBuffer offset:0 atIndex:22];
    [enc setBuffer:pbrtEnvConditionalCDFBuffer offset:0 atIndex:23];
    [enc setBuffer:lensElementBuffer offset:0 atIndex:24];
    [enc setBuffer:exitPupilBoundsBuffer offset:0 atIndex:25];
    [enc setBuffer:cylinderBuffer offset:0 atIndex:26];
    [enc setBuffer:cylinderMaterialBuffer offset:0 atIndex:27];
    // Mark the AS + its dependent primitive ASes as used so Metal
    // knows about the indirection - required for instance
    // acceleration structures referencing primitive ones (now three:
    // the room+Spot triangle mesh, the sphere's bounding-box
    // geometry, and Suzanne's own - referenced by TWO instances, but
    // only needs marking used once here, not once per instance).
    [enc useResource:primAS usage:MTLResourceUsageRead];
    [enc useResource:sphereAS usage:MTLResourceUsageRead];
    [enc useResource:suzanneAS usage:MTLResourceUsageRead];

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
        return false;
    }

    // --- Read back into `pixels` (post-processed and written to disk
    // by postProcessAndWrite(), below) --------------------------------
    pixels.resize(width * height * 4);
    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
    [outTexture getBytes:pixels.data() bytesPerRow:width * 4 * sizeof(float) fromRegion:region mipmapLevel:0];
    return true;
}

// --- Stage 5: post-process the linear HDR buffer and write the PNG -----
// (chromatic aberration, then lens vignette, then the selected tonemap
// operator, then the real sRGB OETF, then bilateral denoise)
void MetalPocApp::postProcessAndWrite() {
    const float vignetteStrength = 0.18f;
    const float chromaticAberrationStrength = 0.004f;
    std::vector<uint8_t> ldr(width * height * 3);
    for (uint32_t i = 0; i < width * height; ++i) {
        uint32_t px = i % width;
        uint32_t py = i / width;
        float vignette = vignetteFactor(px, py, width, height, vignetteStrength);
        float rgb[3];
        chromaticAberration(pixels, width, height, px, py, chromaticAberrationStrength,
                             &rgb[0], &rgb[1], &rgb[2]);
        for (int c = 0; c < 3; ++c) {
            float v = fmaxf(rgb[c], 0.0f) * vignette * exposureValue;
            v = applyToneMap(v, toneMapMode);
            v = linearToSRGB(v);
            ldr[i * 3 + c] = (uint8_t)(v * 255.0f + 0.5f);
        }
    }
    // Bilateral denoise - see that function's own comment. Radius 3
    // (7x7), sigmaSpatial 2.5, sigmaRange 20.0 (in 0-255 luminance
    // units) - tuned the same way every other post-process knob this
    // POC has added was: by rendering and comparing, not from theory
    // alone. A much more aggressive setting (radius 4, sigmaRange 80)
    // was also tried and rejected - it visibly softened the crystal
    // ball's own sharp specular highlight and the checkerboard
    // floor's own tile edges, confirming this knob really can wash
    // out real detail if pushed too far, not just theoretically.
    std::vector<uint8_t> denoised(width * height * 3);
    bilateralDenoise(ldr, denoised, width, height, /*radius=*/3, /*sigmaSpatial=*/2.5f, /*sigmaRange=*/20.0f);

    stbi_write_png(outPath, width, height, 3, denoised.data(), width * 3);
    fprintf(stderr, "Wrote %s (%ux%u)\n", outPath, width, height);
}

// --- Callable entry point (phase 2 of real GPU integration - see
// docs/METAL_GPU_FEASIBILITY.md's own section on this) -------------------
// Signature matches gpu/optix/optix_interface.h's own optix_render_main()
// exactly, down to reusing the same RenderOptions struct - the shape a
// future launcher/main.cpp caller would need once this target is actually
// linked into ray_tracer itself (still TODO, a separate/larger phase: this
// function exists and works standalone, but nothing calls it yet outside
// this file's own main() below and its own smoke test).
//
// See metal_interface.h's own comment for what this reports and why its
// fields differ from optix_get_diagnostics()'s own OptixDiagnostics
// shape. MTLCopyAllDevices() (NOT MTLCreateSystemDefaultDevice() -
// see parseArgsAndCreateDevice()'s own comment on why: that call is
// documented as unsupported for non-interactive CLI/daemon processes,
// confirmed via `log show`, and ray_tracer/metal_poc are both plain CLI
// tools, not app bundles) is the actual availability check - there is
// no separate "is Metal supported" query to make first, unlike CUDA/
// OptiX's own driver-then-device-then-SDK-ABI chain of things that can
// each fail independently.
bool metal_get_diagnostics(MetalDiagnostics* out) {
    if (!out) return false;
    *out = MetalDiagnostics{};
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        id<MTLDevice> device = (devices.count > 0) ? devices[0] : nil;
        if (!device) {
            out->available = false;
            std::string reason("MTLCopyAllDevices() returned no devices - "
                                "no usable Metal device on this machine");
            size_t n = (std::min)(reason.size(), sizeof(out->failure_reason) - 1);
            reason.copy(out->failure_reason, n);
            out->failure_reason[n] = '\0';
            return false;
        }
        out->available = true;
        NSString* name = device.name ? device.name : @"unknown Metal device";
        std::string nameStr(name.UTF8String);
        size_t n = (std::min)(nameStr.size(), sizeof(out->device_name) - 1);
        nameStr.copy(out->device_name, n);
        out->device_name[n] = '\0';
        out->recommended_max_working_set_bytes = device.recommendedMaxWorkingSetSize;
    }
    return true;
}

// scene_id resolution: this POC's own loadPbrtScene() only ever supported
// pbrt-FILE-backed scenes (see that function's own comment). Of this
// project's ~200 registered scenes, the ~93 hand-authored built-in ones
// (scene_registry.h, no pbrt file at all) are reproduced natively by
// gpu/optix/scene_builder.cpp's own ~900-line switch instead of going
// through pbrt_load.h - Metal's own equivalent is buildHandAuthoredScene()
// (section 116, docs/METAL_GPU_FEASIBILITY.md), which starts with real
// coverage for just "A1" and grows one scene (or small batch) at a time.
// scene_id here first tries cpu_scene_pbrt_path_by_id(); if that's empty,
// falls back to cpu_scene_metal_hand_authored_supported() - the ONE
// canonical list (cpu_interface.h's own comment) both this check and
// cpu_scene_metadata_snapshot()'s own metal_compatible field consult, so
// they can't drift apart. A scene covered by NEITHER is reported as
// not-yet-implemented and returns non-zero, the same "explain why, don't
// crash or silently render something else" precedent
// gpu/optix/scene_builder.cpp's own default: case already established.
//
// Camera override (cam_x/y/z, force_camera_override): applied via
// applyCameraOverride() below, right after buildScene() - see that
// method's own comment. Same "override lookfrom only, keep lookat/up/fov
// from the scene's own definition" semantics as cpu_interface.cpp's own
// applyCameraConfig(), and cam_x/y/z are read in the same coordinate
// space that backend's cpu_scene_recommended_camera()/CLI --cam-x/y/z
// already use for every scene.
int metal_render_main(int image_width, int image_height, int samples_per_pixel,
                       int max_depth, const char* output_path, const char* scene_id,
                       double cam_x, double cam_y, double cam_z,
                       int force_camera_override, const RenderOptions& options) {
    const char* pbrtPath = cpu_scene_pbrt_path_by_id(scene_id);
    const bool handAuthored = (!pbrtPath || !pbrtPath[0]) && cpu_scene_metal_hand_authored_supported(scene_id);
    if ((!pbrtPath || !pbrtPath[0]) && !handAuthored) {
        fprintf(stderr, "metal_render_main: scene '%s' has no pbrt file backing it and no "
                        "hand-authored Metal builder yet (see cpu_scene_metal_hand_authored_"
                        "supported()'s own comment, cpu_interface.h, for the current coverage "
                        "list). Use CPU or GPU (OptiX) for this scene instead.\n", scene_id);
        return 1;
    }

    // Builds the SAME positional-argv shape parseArgsAndCreateDevice()/
    // compileShaderAndDispatch() already parse for the standalone CLI
    // below, rather than giving those two functions a second, parallel
    // explicit-parameter entry point of their own - keeps this new
    // callable path exercising the EXACT SAME, already-tested parsing
    // code the CLI does, instead of two argument-handling implementations
    // that could silently drift apart. argv[8] (handAuthoredSceneId) is
    // only ever non-empty when argv[7] (pbrtScenePath) is empty - see
    // that member's own comment, MetalPocApp's own struct declaration.
    char widthStr[32], heightStr[32], sppStr[32], depthStr[32];
    snprintf(widthStr, sizeof(widthStr), "%d", image_width);
    snprintf(heightStr, sizeof(heightStr), "%d", image_height);
    snprintf(sppStr, sizeof(sppStr), "%d", samples_per_pixel);
    snprintf(depthStr, sizeof(depthStr), "%d", max_depth);
    const char* tonemapStr = (options.tonemap && options.tonemap[0]) ? options.tonemap : "aces";
    const char* args[9] = {"metal_render_main", widthStr, heightStr, output_path,
                            sppStr, depthStr, tonemapStr,
                            handAuthored ? "" : pbrtPath,
                            handAuthored ? scene_id : ""};
    const int argCount = 9;

    @autoreleasepool {
        MetalPocApp app;
        if (!app.parseArgsAndCreateDevice(argCount, args)) return 1;
        // See MetalPocApp::exposureValue's own comment for why this is a
        // direct field poke rather than a new argv[] slot.
        app.exposureValue = (float)options.exposure;
        app.buildScene();
        if (force_camera_override) {
            if (app.havePbrtCamera) {
                app.applyCameraOverride(cam_x, cam_y, cam_z);
            } else {
                // pbrtPath resolved above, but loadPbrtScene() itself failed
                // (bad/missing file - see its own stderr message) and
                // buildScene() fell back to the hardcoded room; nothing
                // meaningful to override.
                fprintf(stderr, "metal_render_main: camera override requested but the pbrt scene "
                                "failed to load (see loadPbrtScene's own message above) - ignoring "
                                "the override.\n");
            }
        }
        if (!app.buildGPUResources()) return 1;
        if (!app.compileShaderAndDispatch(argCount, args)) return 1;
        app.postProcessAndWrite();
    }
    return 0;
}

// Renamed from main() (phase 3 of real GPU integration - see docs/
// METAL_GPU_FEASIBILITY.md's own section on this): a real main() now
// needs to live OUTSIDE this file, since metal_poc.mm's own code (this
// function, MetalPocApp, metal_render_main()) is also linked into
// ray_tracer itself, which already has its own main() (launcher/
// main.cpp) - two definitions of main() in the same executable won't
// link. gpu/metal/metal_poc_main.mm - the standalone metal_poc
// executable's only other source file now - is just a one-line main()
// that calls straight through to this, so the standalone CLI's own
// behavior is completely unchanged.
int metal_poc_cli_main(int argc, const char** argv) {
    @autoreleasepool {
        MetalPocApp app;
        if (!app.parseArgsAndCreateDevice(argc, argv)) return 1;
        app.buildScene();
        if (!app.buildGPUResources()) return 1;
        if (!app.compileShaderAndDispatch(argc, argv)) return 1;
        app.postProcessAndWrite();
    }
    return 0;
}
