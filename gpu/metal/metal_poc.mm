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

// --- Stage 2.5: real pbrt scene loading (v1 - see docs/METAL_GPU_
// FEASIBILITY.md's own section on this) ----------------------------------
// Loads pbrtScenePath via src/shared/pbrt_load.h - the SAME front-end
// parser cpu_renderer and gpu/optix both already use (pbrt_cpu_builder.h/
// pbrt_gpu_builder.h) - and appends its geometry/materials/lights into
// this POC's own scene vectors, proving the pipeline shape (the FIRST
// concrete step toward real integration, not the whole thing - see this
// project's own status notes on what's still deliberately NOT done).
//
// Deliberately narrow v1 scope, each gap warned rather than silently
// wrong or a crash (matching gpu/optix/scene_builder.cpp's own graceful-
// failure precedent for GPU-unsupported scenes):
//   - Materials: Diffuse/Conductor/Dielectric only (this POC's own
//     materialType 0/4/2) - anything else falls back to gray Lambertian.
//   - Area lights: only a light attached to EXACTLY 2 triangles forming a
//     planar quad in addQuad()'s own a-b-c-d/a-c-d fan convention, or a
//     single full-circle disk, has a real NEE strategy - AreaLightData
//     (metal_poc.metal) is a parallelogram sampler (center/edgeU/edgeV),
//     not a general triangle-mesh/disk one. Any other emissive shape
//     (a non-quad triangle mesh, an annular/partial disk) is still
//     visible (direct hits, BSDF-sampled bounces) but has no explicit
//     NEE strategy sampling it - see section 100/101 of the docs.
//   - Shapes: triangle meshes, spheres, and full-circle disks (no inner
//     radius/phi-max, uniform scale only) - cylinder/cone/paraboloid/
//     bilinearmesh/curve shapes are still skipped entirely.
//   (ObjectInstance/instancing, infinite lights, and every punctual light
//   kind ARE now supported - this comment block predates those; see the
//   docs' own numbered sections for what shipped after this was written.)
// None of this needed any changes to buildGPUResources() below - see
// buildScene()'s own call-site comment for why (additive onto the
// existing hardcoded room, never leaves any vector newly empty).
void MetalPocApp::loadPbrtScene() {
    pbrt_load::LoadResult result = pbrt_load::loadFile(pbrtScenePath);
    if (!result.ok) {
        fprintf(stderr, "loadPbrtScene: %s\n", result.error.c_str());
        return;
    }
    const pbrt_flatten::FlatScene& scene = result.scene;
    for (const pbrt_scene::Warning& w : scene.warnings) {
        fprintf(stderr, "loadPbrtScene: pbrt loader warning: %s\n", w.message.c_str());
    }

    // Uniform scene-scale normalization: a real pbrt scene is typically
    // authored at a scale of hundreds of units (a classic Cornell box
    // spans ~555) - NOT this shader's own [-1,1]-ish hardcoded-room
    // scale. Every shadow-ray/reflection-ray self-intersection offset in
    // metal_poc.metal (`hitPoint + facingNormal * 0.001f`, ~80 call
    // sites) was tuned for that small scale; at ~500 units, 0.001 is a
    // numerically negligible fraction of the scene (0.0002%) - too small
    // to reliably escape the source triangle's own surface, so every
    // shadow ray immediately (re-)self-intersects and every light sample
    // reads as occluded. Confirmed directly: without this, the loaded
    // Cornell box rendered almost entirely black (only the light's own
    // direct camera-hit visible; every diffuse wall got zero NEE light).
    // Rescaling metal_poc.metal's own ~80 epsilons to be scene-relative
    // instead would be a much larger, riskier change than rescaling the
    // INPUT geometry once, here, at load time - computed from the
    // scene's own bounding box (triangles + spheres) so this generalizes
    // to any pbrt scene's own authored scale, not just this one file's.
    float3 bboxMin{FLT_MAX, FLT_MAX, FLT_MAX}, bboxMax{-FLT_MAX, -FLT_MAX, -FLT_MAX};
    auto growBounds = [&](float3 p) {
        bboxMin = simd::min(bboxMin, p);
        bboxMax = simd::max(bboxMax, p);
    };
    for (const pbrt_flatten::Triangle& t : scene.triangles) {
        for (int c = 0; c < 3; ++c)
            growBounds(float3{(float)t.v[c * 3 + 0], (float)t.v[c * 3 + 1], (float)t.v[c * 3 + 2]});
    }
    for (const pbrt_flatten::Sphere& s : scene.spheres) {
        const float3 c{(float)s.center[0], (float)s.center[1], (float)s.center[2]};
        const float r = (float)s.radius;
        growBounds(c - float3{r, r, r});
        growBounds(c + float3{r, r, r});
    }
    const float3 bboxExtent = bboxMax - bboxMin;
    const float maxExtent = fmaxf(bboxExtent.x, fmaxf(bboxExtent.y, bboxExtent.z));
    // Target: the loaded scene's own largest dimension maps to 2.0 units -
    // matching the hardcoded room's own [-1,1] (2-unit-across) scale, so
    // every one of those existing epsilons is meaningful again. Falls
    // back to 1.0 (no rescale) for a degenerate/empty scene rather than
    // dividing by ~0.
    const float sceneScale = (maxExtent > 1e-6f) ? (2.0f / maxExtent) : 1.0f;
    // Recentre on the scene's own bounding-box centre, THEN push it well
    // clear of the hardcoded room's own occupied [-1,1] region (a fixed
    // +8 in X - more than enough given the loaded scene's own rescaled
    // extent is ~2 units) - the pbrt scene's own coordinate origin has no
    // reason to relate to the hardcoded room's at all (e.g. this classic
    // Cornell box is authored spanning x/y/z ~[0,555], not centred at its
    // own origin), so simply rescaling in place (this function's own
    // first attempt) left the two scenes - and the camera, repositioned
    // to the loaded scene's own - confusingly overlapping in the SAME
    // small region of world space, with the render showing a hard-to-
    // interpret mix of both. Recentre + offset keeps this purely
    // ADDITIVE (see buildScene()'s own call-site comment on why - no
    // buildGPUResources() changes needed) while keeping the two scenes
    // visually and spatially separate, exactly as if they were two
    // different rooms.
    const float3 bboxCenter = 0.5f * (bboxMin + bboxMax);
    const float3 sceneOffset{8.0f, 0.0f, 0.0f};
    auto toWorld = [=](float3 p) { return (p - bboxCenter) * sceneScale + sceneOffset; };
    fprintf(stderr, "loadPbrtScene: scene bounding box extent %.1f units, rescaling by %.5f, "
                    "recentred and offset to +X\n", maxExtent, sceneScale);

    // Tracks which UNSUPPORTED material kind NAMES (not indices - the same
    // name can legitimately appear at more than one scene.materials index)
    // have already been warned about, so mapMaterial() below - called once
    // per TRIANGLE/sphere/instance referencing a material, not once per
    // distinct material - doesn't flood stderr with the identical message
    // for every single primitive. A mesh with tens of thousands of
    // triangles sharing one unsupported material (e.g. killeroo-simple.pbrt's
    // own 66,532-triangle "coateddiffuse" mesh) used to print that exact
    // line 66,532 times.
    std::unordered_set<std::string> warnedUnsupportedMaterialKinds;
    // std::function (not auto), capturing itself by reference, so the
    // Mix case below can recurse into mapMaterial() for its own two
    // named sub-materials - an ordinary auto lambda can't reference its
    // own name inside its own body (not yet in scope at that point).
    std::function<TriangleMaterial(const pbrt_flatten::Material&, int)> mapMaterial =
        [this, &warnedUnsupportedMaterialKinds, &scene, &mapMaterial](const pbrt_flatten::Material& m, int depth) -> TriangleMaterial {
        PackedFloat3 color{(float)m.color[0], (float)m.color[1], (float)m.color[2]};
        switch (m.kind) {
            case pbrt_flatten::MaterialKind::Diffuse: {
                // A "reflectance" bound to a "checkerboard" Texture (B22,
                // section 162) - materialType 25, pbrt-v4's real UV-space
                // 2-colour checker (see that materialType's own shading
                // comment, metal_poc_kernel.metal). Only the SIMPLE case
                // this loader can represent: tex1/tex2 each either a flat
                // literal or already flattened to one by
                // pbrt_flatten.h's own nestedProceduralAverageColor()
                // (m.checkerColor1/2 are ALWAYS valid flat colours by
                // this point, regardless of nesting depth - that
                // function's own comment) - a bare "imagemap" bound to
                // tex1/tex2 (m.checkerTex1Filename/checkerTex2Filename)
                // is the one sub-case this loader doesn't carry through
                // at all, unlike CPU's own full nested-texture support;
                // no bundled scene needs it (matches roughnessTextureFilename's
                // own identical "bare imagemap only" scope-narrowing,
                // pbrt_flatten.h) - falls through to the flat-colour
                // default below instead of misrendering it.
                if (m.hasCheckerReflectance && m.checkerTex1Filename.empty() && m.checkerTex2Filename.empty()) {
                    TriangleMaterial mat{
                        PackedFloat3{(float)m.checkerColor1[0], (float)m.checkerColor1[1], (float)m.checkerColor1[2]},
                        /*materialType=*/25u, /*ior=*/1.0f, PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
                    mat.transmitColor = PackedFloat3{(float)m.checkerColor2[0], (float)m.checkerColor2[1], (float)m.checkerColor2[2]};
                    mat.conductorEta = PackedFloat3{(float)m.checkerUScale, (float)m.checkerVScale, 0.0f};
                    return mat;
                }
                // A "reflectance" bound to a bare "imagemap" Texture
                // (F5/F9, section 166) - materialType 26, a real per-hit
                // image lookup via pbrtDiffuseTexture (see that
                // materialType's own shading comment,
                // metal_poc_kernel.metal). Only the FIRST DISTINCT such
                // texture FILENAME in the scene gets the one dedicated
                // texture slot (same "one shared texture" constraint
                // every other pbrt-loaded image here already has) - a
                // second, DIFFERENT filename, or a decode failure, falls
                // through to the flat-colour default below exactly as if
                // no texture were bound at all. Deliberately checked
                // against the ALREADY-LOADED filename, not just "already
                // loaded something" - `mapMaterial()` is called once per
                // TRIANGLE, not once per distinct Material, so a single
                // 2-triangle quad sharing ONE textured material calls
                // this twice for the SAME filename; a naive "first call
                // wins, every later call falls back" check (this
                // function's own first version) gave the quad's second
                // triangle a flat grey fallback instead of the same
                // texture its first triangle correctly got - a real,
                // visible bug (a hard diagonal split down the middle of
                // what should be one seamlessly textured quad), caught
                // by rendering and comparing against `--cpu`, not
                // assumed safe from the code alone.
                if (!m.textureFilename.empty() &&
                    (!havePbrtDiffuseImage || m.textureFilename == pbrtDiffuseImageFilename)) {
                    if (!havePbrtDiffuseImage) {
                        std::string bytes;
                        if (pbrt_load::loadFileNear(pbrtScenePath, m.textureFilename, bytes) &&
                            pbrt_load::detail::decodeInfiniteLightImage(m.textureFilename, bytes,
                                pbrtDiffuseImagePixels, pbrtDiffuseImageWidth, pbrtDiffuseImageHeight)) {
                            havePbrtDiffuseImage = true;
                            pbrtDiffuseImageFilename = m.textureFilename;
                        } else {
                            fprintf(stderr, "loadPbrtScene: Diffuse material's own texture '%s' could not be "
                                            "read/decoded; falling back to its flat colour\n", m.textureFilename.c_str());
                        }
                    }
                    if (havePbrtDiffuseImage) {
                        return TriangleMaterial{color, /*materialType=*/26u, /*ior=*/1.0f,
                                                 PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
                    }
                }
                return TriangleMaterial{color, /*materialType=*/0u, /*ior=*/1.0f,
                                         PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
            }
            case pbrt_flatten::MaterialKind::Conductor: {
                // RoughnessToAlpha (src/shared/microfacet.h) is sqrt(r) -
                // pbrt-v4's own "remaproughness" default (true) means the
                // authored value needs this remap; false means it already
                // IS alpha.
                const float alpha = (float)(m.remapRoughness ? std::sqrt(m.roughness) : m.roughness);
                TriangleMaterial mat{color, /*materialType=*/4u, /*ior(alphaX)=*/alpha,
                                     PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness(alphaY)=*/alpha};
                mat.conductorEta = PackedFloat3{(float)m.conductorEta[0], (float)m.conductorEta[1], (float)m.conductorEta[2]};
                mat.conductorK = PackedFloat3{(float)m.conductorK[0], (float)m.conductorK[1], (float)m.conductorK[2]};
                return mat;
            }
            case pbrt_flatten::MaterialKind::CoatedConductor: {
                // Approx tier (docs/PBRT_SUPPORT.md's own note, matching
                // both CPU's pbrt_cpu_builder.h and GPU-OptiX's own
                // pbrt_gpu_builder_materials.h): materialType 4 (GGX
                // conductor) - the "coat" itself (a separate rough
                // dielectric layer over the metal base) isn't modelled at
                // all, only the base conductor's own eta/k/roughness -
                // the same simplification both other backends already
                // accept for this kind, not a NEW approximation invented
                // here. A named metal spectrum or explicit "eta"/"k" (m.
                // hasConductorPreset) is used directly, already resolved
                // by pbrt_flatten.h identically to plain Conductor above;
                // otherwise `color` (the scene's own "reflectance", a
                // normal-incidence Schlick value) is converted via the
                // SAME eta=1/k-solved-from-r formula CPU's own
                // reflectanceToConductorK() uses (k = 2*sqrt(r) /
                // sqrt(max(1e-4, 1-r)), per channel) - not re-derived
                // independently, matching a real precedent already
                // established for exactly this "nothing given" case.
                const float alpha = (float)(m.remapRoughness ? std::sqrt(m.roughness) : m.roughness);
                TriangleMaterial mat{color, /*materialType=*/4u, /*ior(alphaX)=*/alpha,
                                     PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness(alphaY)=*/alpha};
                if (m.hasConductorPreset) {
                    mat.conductorEta = PackedFloat3{(float)m.conductorEta[0], (float)m.conductorEta[1], (float)m.conductorEta[2]};
                    mat.conductorK = PackedFloat3{(float)m.conductorK[0], (float)m.conductorK[1], (float)m.conductorK[2]};
                } else {
                    auto reflectanceToK = [](float r) {
                        r = r < 0.0f ? 0.0f : (r > 0.9999f ? 0.9999f : r);
                        return 2.0f * sqrtf(r) / sqrtf(std::max(1e-4f, 1.0f - r));
                    };
                    mat.conductorEta = PackedFloat3{1.0f, 1.0f, 1.0f};
                    mat.conductorK = PackedFloat3{reflectanceToK(color.x), reflectanceToK(color.y), reflectanceToK(color.z)};
                }
                return mat;
            }
            case pbrt_flatten::MaterialKind::Dielectric:
                return TriangleMaterial{color, /*materialType=*/2u, /*ior=*/(float)m.ior,
                                         PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
            case pbrt_flatten::MaterialKind::ThinDielectric:
                // materialType 11 - `color` is unused by this material
                // (metal_poc.mm's own hardcoded-room construction site
                // comment), only `ior` matters.
                return TriangleMaterial{color, /*materialType=*/11u, /*ior=*/(float)m.ior,
                                         PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
            case pbrt_flatten::MaterialKind::DiffuseTransmission: {
                // materialType 12 - `color` is this material's own
                // reflectance tint (same convention as Diffuse above);
                // `m.transmittance` is the SEPARATE transmitted tint
                // (Material::transmittance's own comment - a frosted-
                // panel default of 0.25 each way when the scene names
                // neither parameter, not a mirror-symmetric 0.5/0.5 split
                // of `color`).
                TriangleMaterial mat{color, /*materialType=*/12u, /*ior=*/1.0f,
                                     PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
                mat.transmitColor = PackedFloat3{(float)m.transmittance[0], (float)m.transmittance[1],
                                                  (float)m.transmittance[2]};
                return mat;
            }
            case pbrt_flatten::MaterialKind::CoatedDiffuse:
                // Approx tier (docs/PBRT_SUPPORT.md's own convention -
                // CPU/OptiX render this Full, a real stochastic layered
                // coat-over-Lambertian): materialType 8 (clearcoat) is
                // this POC's own simplified SINGLE-bounce smooth-
                // dielectric-coat-over-Lambertian, with a fixed
                // kClearcoatEta=1.5 shader-side constant - the scene's
                // own "ior"/"roughness" (a rough, scene-specified coat)
                // are silently NOT read here, unlike materialType 8's
                // hardcoded-room use, which never varies them either.
                // Still a real, honest improvement over the gray-
                // Lambertian default fallback below: the correct diffuse
                // albedo and a generic coat sheen both survive, just not
                // the exact coat IOR/roughness.
                return TriangleMaterial{color, /*materialType=*/8u, /*ior=*/1.0f,
                                         PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
            case pbrt_flatten::MaterialKind::Mix: {
                // Approx tier: CPU/OptiX both do a REAL per-shading-point
                // stochastic pick between the two named sub-materials
                // (pbrt-v4 MixMaterial - a hash of the hit point decides,
                // giving a fine-grained speckle of both materials' own
                // character, not a blended average - see
                // material_pbrt.h's own mix_material::scatter()). This
                // loader assigns materials once per triangle at LOAD
                // TIME, not per-ray-hit, so a real per-point stochastic
                // mix isn't representable without new per-pixel shader
                // logic - instead, deterministically resolves to
                // whichever of the two sub-materials mixWeight (pbrt's
                // own "amount", the probability weight toward B - see
                // mix_material::scatter()'s own `hash >= w ? A : B`)
                // favours, recursing into mapMaterial() for that ONE
                // sub-material's own real (possibly ALSO Approx-tier)
                // mapping. A uniform single-material triangle instead of
                // a fine speckle - not the real thing, but still a real,
                // honest improvement over gray Lambertian (the correct
                // material FAMILY and colour survive, just not the
                // per-point blend).
                // Depth-guarded (mirrors pbrt_cpu_builder.h's own
                // kMaxMixDepth=8 exactly - see that file's own comment:
                // "a cyclic/self-referential 'materials' list a
                // malformed scene could produce" is a real, anticipated
                // risk, not hypothetical - namedMaterialIndex is built
                // by scanning the WHOLE material list up front
                // (pbrt_flatten.h), before per-material resolution runs,
                // specifically so a "materials" list can name something
                // declared LATER in the file - which also means a Mix
                // material's own name can legally appear in its own
                // "materials" list, or two Mix materials can name each
                // other, with no cycle check anywhere in flatten() to
                // catch it. Without this guard, mapMaterial()'s own
                // recursion into such a scene would stack-overflow this
                // loader before any render starts - a real bug found by
                // code review, not exercised by any bundled scene.
                const int chosenIdx = (m.mixWeight >= 0.5) ? m.mixMaterialB : m.mixMaterialA;
                constexpr int kMaxMixDepth = 8;
                if (depth < kMaxMixDepth && chosenIdx >= 0 && chosenIdx < (int)scene.materials.size())
                    return mapMaterial(scene.materials[chosenIdx], depth + 1);
                // Both indices invalid (shouldn't happen - flatten()'s
                // own comment guarantees them valid whenever kind==Mix -
                // but this loader errs toward a safe fallback rather
                // than an out-of-bounds read), OR the depth guard above
                // fired (a cyclic/self-referential "materials" list) -
                // falls through to the same gray-Lambertian default
                // every other unsupported kind gets, same "safe fallback
                // over crashing" reasoning as every other malformed-
                // scene case in this file.
                [[fallthrough]];
            }
            default:
                if (warnedUnsupportedMaterialKinds.insert(m.pbrtType).second) {
                    fprintf(stderr, "loadPbrtScene: material kind '%s' not supported by this POC's "
                                    "scene loader yet, using gray Lambertian instead\n", m.pbrtType.c_str());
                }
                return TriangleMaterial{PackedFloat3{0.5f, 0.5f, 0.5f}, 0u, 1.0f,
                                         PackedFloat3{0, 0, 0}, -1, 0.0f};
        }
    };
    auto materialFor = [&](int idx) -> TriangleMaterial {
        if (idx < 0 || idx >= (int)scene.materials.size())
            return TriangleMaterial{PackedFloat3{0.5f, 0.5f, 0.5f}, 0u, 1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f};
        return mapMaterial(scene.materials[idx], /*depth=*/0);
    };
    auto vertexAt = [toWorld](const double* v, int i) {
        return toWorld(float3{(float)v[i * 3 + 0], (float)v[i * 3 + 1], (float)v[i * 3 + 2]});
    };

    std::vector<bool> triangleHandled(scene.triangles.size(), false);
    // Populated by loadPbrtAreaLights() below only for a light shape too
    // complex to represent as this loader's single analytic AreaLightData
    // quad (see that method's own comment) - read by
    // loadPbrtRemainingTriangles() right after to mark those triangles
    // emissive (but NOT NEE-light-registered) instead of silently
    // dropping their emission.
    std::unordered_map<int, std::pair<float3, bool>> unhandledLightEmission;
    loadPbrtAreaLights(scene, toWorld, materialFor, triangleHandled, unhandledLightEmission);
    loadPbrtRemainingTriangles(scene, toWorld, materialFor, triangleHandled, unhandledLightEmission);
    loadPbrtSpheres(scene, toWorld, materialFor, sceneScale);
    loadPbrtDisks(scene, toWorld, materialFor, sceneScale);
    loadPbrtObjectInstances(scene, toWorld, materialFor);
    loadPbrtPunctualLights(scene, toWorld, sceneScale);
    loadPbrtMedium(scene, sceneScale);
    loadPbrtInfiniteLight(scene);
    loadPbrtCamera(scene, toWorld, bboxCenter, sceneScale, sceneOffset);

    fprintf(stderr, "loadPbrtScene: loaded %s (%zu triangles, %zu spheres, %zu area lights)\n",
            pbrtScenePath.c_str(), scene.triangles.size(), scene.spheres.size(), scene.areaLights.size());
}

// --- Area lights: only the "single quad, 2 triangles" shape - see
// this function's own header comment.
void MetalPocApp::loadPbrtAreaLights(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld,
    const PbrtMaterialForFn& materialFor, std::vector<bool>& triangleHandled,
    std::unordered_map<int, std::pair<float3, bool>>& unhandledLightEmission) {
    auto vertexAt = [toWorld](const double* v, int i) {
        return toWorld(float3{(float)v[i * 3 + 0], (float)v[i * 3 + 1], (float)v[i * 3 + 2]});
    };
    std::unordered_map<int, std::vector<int>> trianglesByLight;
    for (int i = 0; i < (int)scene.triangles.size(); ++i) {
        const int al = scene.triangles[i].areaLight;
        if (al >= 0) trianglesByLight[al].push_back(i);
    }
    for (const auto& entry : trianglesByLight) {
        const int lightIdx = entry.first;
        const std::vector<int>& idxs = entry.second;
        const pbrt_flatten::Emission& em = scene.areaLights[lightIdx];
        float3 emission{(float)(em.L[0] * em.scale), (float)(em.L[1] * em.scale), (float)(em.L[2] * em.scale)};
        uint32_t quadMaterialType = 0u;
        float useTextureFlag = 0.0f;
        bool handled = false;
        if (idxs.size() == 2) {
            const pbrt_flatten::Triangle& t0 = scene.triangles[idxs[0]];
            const pbrt_flatten::Triangle& t1 = scene.triangles[idxs[1]];
            const float3 a = vertexAt(t0.v, 0), b = vertexAt(t0.v, 1), c = vertexAt(t0.v, 2);
            const float3 t1a = vertexAt(t1.v, 0), t1b = vertexAt(t1.v, 1), d = vertexAt(t1.v, 2);
            const float eps = 1e-4f;
            if (simd::length(a - t1a) < eps && simd::length(c - t1b) < eps) {
                // Image-based emission (section 105) - only for a
                // confirmed QUAD-shaped light (this branch - the vertex-
                // matching check just above already ruled out a 2-
                // triangle shape that ISN'T actually a quad, e.g. a
                // differently-diagonalized or non-planar pair; doing
                // this decode attempt BEFORE that check, as an earlier
                // version of this code did, would waste the one shared
                // texture slot - and leave `emission` wrongly set to a
                // bare `scale` instead of `L*scale` - on a light that
                // never actually becomes materialType 15 at all), and
                // only the FIRST such light in the scene (one shared
                // texture slot, matching the goniometric/projection
                // image precedent). A decode failure - or a SECOND
                // textured light - falls back to flat L exactly as if no
                // filename were named, same "safe fallback over dropping
                // the light" reasoning as every other image-based
                // feature in this loader.
                if (!em.filename.empty() && !havePbrtAreaLightImage) {
                    std::string bytes;
                    if (pbrt_load::loadFileNear(pbrtScenePath, em.filename, bytes) &&
                        pbrt_load::detail::decodeInfiniteLightImage(em.filename, bytes,
                            pbrtAreaLightImagePixels, pbrtAreaLightImageWidth, pbrtAreaLightImageHeight)) {
                        havePbrtAreaLightImage = true;
                        quadMaterialType = 15u;
                        useTextureFlag = 1.0f;
                        emission = float3{(float)em.scale, (float)em.scale, (float)em.scale};
                    } else {
                        fprintf(stderr, "loadPbrtScene: area light's own image '%s' could not be read/decoded; "
                                        "falling back to its flat colour\n", em.filename.c_str());
                    }
                }
                const TriangleMaterial lightMat = materialFor(t0.material);
                const int32_t lightId = (int32_t)lights.size();
                addQuad(verts, normals, uvs, materials, a, b, c, d,
                        float3{lightMat.color.x, lightMat.color.y, lightMat.color.z},
                        quadMaterialType, emission, lightId, /*roughness=*/0.0f,
                        /*ior=*/1.0f, /*transmitColor=*/simd::make_float3(0, 0, 0),
                        /*twoSided=*/em.twoSided);
                const float3 edgeU = b - a;
                const float3 edgeV = d - a;
                const float3 normalV = simd::normalize(simd::cross(edgeU, edgeV));
                const float area = simd::length(simd::cross(edgeU, edgeV));
                const float3 center = a + 0.5f * edgeU + 0.5f * edgeV;
                lights.push_back(AreaLightData{
                    PackedFloat3{center.x, center.y, center.z},
                    PackedFloat3{edgeU.x, edgeU.y, edgeU.z},
                    PackedFloat3{edgeV.x, edgeV.y, edgeV.z},
                    PackedFloat3{normalV.x, normalV.y, normalV.z},
                    area,
                    PackedFloat3{emission.x, emission.y, emission.z},
                    /*patternTileB=*/0.0f,
                    /*patternScale=*/0.0f,
                    /*twoSided=*/em.twoSided ? 1.0f : 0.0f,
                    /*useTexture=*/useTextureFlag});
                handled = true;
            }
        }
        if (handled) {
            triangleHandled[idxs[0]] = true;
            triangleHandled[idxs[1]] = true;
        } else {
            // Not a simple 2-triangle quad (a single triangle, an N>2
            // triangle mesh, ...) - this loader has no general per-
            // triangle NEE-sampled light representation (AreaLightData
            // is a single analytic QUAD, sampled as one unit; every
            // triangle of a real mesh light would need its own entry and
            // its own share of the light-picking pmf, real work this
            // loader doesn't do yet). Rather than silently dropping the
            // light's own emission entirely (this function's own PREVIOUS
            // behaviour - a real correctness bug, not just a missing
            // optimization: the light became fully invisible, not merely
            // higher-variance), each of its triangles is now marked
            // emissive directly (materialEmission below, lightId left at
            // -1 so the MIS-weight code below correctly treats every hit
            // on it as unweighted, same as a specular bounce) - reachable
            // by a camera ray or a BSDF-sampled bounce landing on it
            // directly, same as any other emissive surface, just with NO
            // NEE strategy sampling it explicitly. A real, unbiased,
            // higher-variance estimator - the same "correct but noisier"
            // tier this loader's own constant/image-based infinite light
            // miss-path-only cases already use (sections 89/90).
            for (int idx : idxs) unhandledLightEmission[idx] = {emission, em.twoSided};
            fprintf(stderr, "loadPbrtScene: area light with %zu triangle(s) is not a simple quad - "
                            "rendering it emissive but without an explicit NEE strategy (higher variance, "
                            "not invisible)\n", idxs.size());
        }
    }
}

// Every triangle loadPbrtAreaLights() didn't already consume as a
// light's own quad - ordinary geometry, or an unhandled (non-quad)
// light's own triangles (emissive via `unhandledLightEmission`, not
// NEE-registered).
void MetalPocApp::loadPbrtRemainingTriangles(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld,
    const PbrtMaterialForFn& materialFor, const std::vector<bool>& triangleHandled,
    const std::unordered_map<int, std::pair<float3, bool>>& unhandledLightEmission) {
    auto vertexAt = [toWorld](const double* v, int i) {
        return toWorld(float3{(float)v[i * 3 + 0], (float)v[i * 3 + 1], (float)v[i * 3 + 2]});
    };
    for (int i = 0; i < (int)scene.triangles.size(); ++i) {
        if (triangleHandled[i]) continue;
        const pbrt_flatten::Triangle& t = scene.triangles[i];
        const float3 v0 = vertexAt(t.v, 0), v1 = vertexAt(t.v, 1), v2 = vertexAt(t.v, 2);
        verts.push_back(PackedFloat3{v0.x, v0.y, v0.z});
        verts.push_back(PackedFloat3{v1.x, v1.y, v1.z});
        verts.push_back(PackedFloat3{v2.x, v2.y, v2.z});
        if (t.hasNormals) {
            // The vertex stream is FLAT (no index buffer - see addQuad()'s
            // own comment), so push all 3 corner normals, not one shared
            // flat value - shadingNormalFor() barycentric-interpolates
            // whatever sits in each of these 3 slots.
            for (int c = 0; c < 3; ++c)
                normals.push_back(PackedFloat3{(float)t.n[c * 3 + 0], (float)t.n[c * 3 + 1], (float)t.n[c * 3 + 2]});
        } else {
            const float3 faceN = simd::normalize(simd::cross(v1 - v0, v2 - v0));
            const PackedFloat3 packedN{faceN.x, faceN.y, faceN.z};
            normals.push_back(packedN); normals.push_back(packedN); normals.push_back(packedN);
        }
        if (t.hasUVs) {
            for (int c = 0; c < 3; ++c)
                uvs.push_back(PackedFloat2{(float)t.uv[c * 2 + 0], (float)t.uv[c * 2 + 1]});
        } else {
            uvs.push_back(PackedFloat2{0, 0}); uvs.push_back(PackedFloat2{0, 0}); uvs.push_back(PackedFloat2{0, 0});
        }
        TriangleMaterial mat = materialFor(t.material);
        auto unhandledIt = unhandledLightEmission.find(i);
        if (unhandledIt != unhandledLightEmission.end()) {
            const float3& unhandledEmission = unhandledIt->second.first;
            mat.emission = PackedFloat3{unhandledEmission.x, unhandledEmission.y, unhandledEmission.z};
            mat.lightId = -1;
            mat.twoSided = unhandledIt->second.second ? 1u : 0u;
        }
        materials.push_back(mat);
    }
}

// --- Spheres ---------------------------------------------------------
void MetalPocApp::loadPbrtSpheres(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld,
    const PbrtMaterialForFn& materialFor, float sceneScale) {
    for (const pbrt_flatten::Sphere& s : scene.spheres) {
        const float3 center = toWorld(float3{(float)s.center[0], (float)s.center[1], (float)s.center[2]});
        spheres.push_back(SphereData{
            PackedFloat3{center.x, center.y, center.z}, sceneScale * (float)s.radius});
        TriangleMaterial mat = materialFor(s.material);
        if (s.areaLight >= 0 && s.areaLight < (int)scene.areaLights.size()) {
            // Same "emissive, but not NEE-registered" tier loadPbrtDisks()
            // just above already established for a disk-shaped
            // AreaLightSource (that function's own comment) - this loader
            // has no sphere-shaped analytic light in its own `lights[]`
            // NEE list either, so a sphere area light is visible (direct
            // hit or a BSDF-sampled bounce landing on it) but not
            // explicitly sampled. `sphereMaterials` is plain
            // TriangleMaterial, so the SAME unconditional direct-hit
            // emissive/MIS-weight code every other emissive material
            // already goes through handles this correctly with no
            // further shader changes needed. A real, previously
            // undiscovered gap (section 164): unlike loadPbrtDisks() just
            // above, this function never read `s.areaLight` at all - a
            // sphere-shaped light rendered as a plain non-emissive grey
            // sphere, contributing NO light to the scene whatsoever (not
            // merely noisier/unsampled - fully absent), closer to the
            // ORIGINAL bug loadPbrtAreaLights()'s own comment describes
            // fixing for non-quad triangle-mesh lights than to the
            // disk-shaped case's own already-correct "noisier but
            // present" tier.
            const pbrt_flatten::Emission& em = scene.areaLights[s.areaLight];
            mat.emission = PackedFloat3{(float)(em.L[0] * em.scale), (float)(em.L[1] * em.scale), (float)(em.L[2] * em.scale)};
            mat.lightId = -1;
            mat.twoSided = em.twoSided ? 1u : 0u;
        }
        sphereMaterials.push_back(mat);
    }
}

// --- Disks (section 101) - the plain, common "full circle" case only:
    // this loader's own DiskData primitive (metal_poc.metal, matching the
    // hardcoded room's own single disk) is center/normal/radius with no
    // inner-radius/phi-max partial-disk support at all, unlike pbrt-v4's
    // real Disk (Disk::innerRadius/phiMaxDeg). An annular or wedge-shaped
    // disk is warned and skipped entirely (Approx tier's own honesty:
    // rendering a full disk in place of a wedge would be visibly WRONG,
    // not just simplified, so skipping is the safer choice here, unlike
    // e.g. CoatedDiffuse's own "close enough" Approx mapping).
    //
    // Disk::xform is a real 4x4 (translation + rotation + scale, possibly
    // non-uniform) - reuses pbrt_flatten::flatten_detail::transformPoint()/
    // transformNormal() directly (the SAME already-correct, cofactor/
    // adjugate-based utilities ObjectInstance baking above already uses),
    // rather than re-deriving the inverse-transpose normal transform by
    // hand. A disk's own local plane sits at object-space z=`height`
    // (pbrt-v4's own Disk convention) with its face normal along local
    // +Z - transformPoint()/transformNormal() applied to (0,0,height)/
    // (0,0,1) respectively give the real world-space center/normal
    // directly, correct even under non-uniform scale (unlike radius
    // below).
    //
    // Non-uniform scale is detected (not assumed): the LENGTHS of the
    // transformed local X/Y axis vectors must agree (within a loose
    // relative tolerance - transformPoint()'s own floating-point path
    // through a full 4x4, not a hand-verified-exact computation) for a
    // single scalar radius to mean anything at all - an ellipse-shaped
    // disk skips for the same "don't render something visibly wrong"
    // reason a partial disk does, mirroring skippedInstancedSpheres'
    // own non-uniform-scale precedent for instanced spheres.
void MetalPocApp::loadPbrtDisks(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld,
    const PbrtMaterialForFn& materialFor, float sceneScale) {
    size_t skippedDisks = 0;
    for (const pbrt_flatten::Disk& d : scene.disks) {
        if (d.innerRadius != 0.0 || d.phiMaxDeg != 360.0) { ++skippedDisks; continue; }

        pbrt_scene::Matrix4 dxform;
        for (int i = 0; i < 16; ++i) dxform.m[i] = d.xform[i];

        double worldOrigin[3], worldAxisX[3], worldAxisY[3];
        pbrt_flatten::flatten_detail::transformPoint(dxform, 0.0, 0.0, 0.0, worldOrigin);
        pbrt_flatten::flatten_detail::transformPoint(dxform, 1.0, 0.0, 0.0, worldAxisX);
        pbrt_flatten::flatten_detail::transformPoint(dxform, 0.0, 1.0, 0.0, worldAxisY);
        const float3 wOrigin{(float)worldOrigin[0], (float)worldOrigin[1], (float)worldOrigin[2]};
        const float scaleX = simd::length(float3{(float)worldAxisX[0], (float)worldAxisX[1], (float)worldAxisX[2]} - wOrigin);
        const float scaleY = simd::length(float3{(float)worldAxisY[0], (float)worldAxisY[1], (float)worldAxisY[2]} - wOrigin);
        if (fabsf(scaleX - scaleY) > 1e-3f * std::max(scaleX, scaleY)) { ++skippedDisks; continue; }

        double worldCenter[3], worldNormal[3];
        pbrt_flatten::flatten_detail::transformPoint(dxform, 0.0, 0.0, d.height, worldCenter);
        pbrt_flatten::flatten_detail::transformNormal(dxform, 0.0, 0.0, 1.0, worldNormal);
        const float3 center = toWorld(float3{(float)worldCenter[0], (float)worldCenter[1], (float)worldCenter[2]});
        const float3 normal = simd::normalize(float3{(float)worldNormal[0], (float)worldNormal[1], (float)worldNormal[2]});

        TriangleMaterial mat = materialFor(d.material);
        if (d.areaLight >= 0 && d.areaLight < (int)scene.areaLights.size()) {
            // Same "emissive, but not NEE-registered" tier PR #100 added
            // for non-quad triangle-mesh lights - this loader has no
            // disk-shaped analytic light in its own `lights[]` NEE list
            // either, so a disk area light is visible (direct hit or a
            // BSDF-sampled bounce landing on it) but not explicitly
            // sampled. `diskMaterials` is plain TriangleMaterial, so the
            // SAME unconditional direct-hit emissive/MIS-weight code PR
            // #100 fixed already handles this correctly with no further
            // shader changes.
            const pbrt_flatten::Emission& em = scene.areaLights[d.areaLight];
            mat.emission = PackedFloat3{(float)(em.L[0] * em.scale), (float)(em.L[1] * em.scale), (float)(em.L[2] * em.scale)};
            mat.lightId = -1;
            mat.twoSided = em.twoSided ? 1u : 0u;
        }
        disks.push_back(DiskData{PackedFloat3{center.x, center.y, center.z},
                                  PackedFloat3{normal.x, normal.y, normal.z},
                                  sceneScale * scaleX * (float)d.radius});
        diskMaterials.push_back(mat);
    }
    if (skippedDisks > 0)
        fprintf(stderr, "loadPbrtScene: %zu disk(s) skipped - only a full circle (no inner radius/phi-max) "
                        "under uniform scale is supported by this POC's own disk primitive\n", skippedDisks);
}

// --- ObjectInstance placements -----------------------------------
// Real instancing (scene.groups hold OBJECT-space geometry, defined
// once; scene.instances place them with a per-placement object->world
// transform - see FlatScene's own comment) is baked into plain
// world-space triangles here, rather than mirroring gpu/optix/
// pbrt_gpu_builder.h's own approach of a true GPU-level instance
// acceleration structure - this POC's own shader dispatch already
// resolves each geometry "kind" (room triangles, spheres, disks,
// Suzanne) via its own fixed buffer/intersection-function-table slot,
// so adding a genuinely general N-group instancing mechanism there
// would be a much larger change than this pbrt loader warrants for
// what's typically a handful of placements (e.g. this repo's own
// example-cornell.pbrt: 3). Baking duplicates geometry per placement
// instead of sharing one buffer - free for a scene with a few dozen
// instances, the case every pbrt scene this loader has seen uses.
//
// Emissive instanced shapes need no special handling here: flatten()
// itself already bakes those directly into scene.triangles (a light
// must be enumerable to be sampled - see pbrt_gpu_builder.h's own
// comment on the same point), so scene.groups/scene.instances only
// ever contain non-emissive geometry.
void MetalPocApp::loadPbrtObjectInstances(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld,
    const PbrtMaterialForFn& materialFor) {
    size_t instancedTriangleCount = 0, skippedInstancedSpheres = 0;
    for (const pbrt_flatten::Instance& inst : scene.instances) {
        if (inst.group < 0 || (size_t)inst.group >= scene.groups.size()) {
            fprintf(stderr, "loadPbrtScene: ObjectInstance with an invalid group index skipped\n");
            continue;
        }
        pbrt_scene::Matrix4 xform;
        for (int i = 0; i < 16; ++i) xform.m[i] = inst.xform[i];
        const pbrt_flatten::InstanceGroup& grp = scene.groups[inst.group];
        for (const pbrt_flatten::Triangle& t : grp.triangles) {
            double worldV[9];
            for (int c = 0; c < 3; ++c)
                pbrt_flatten::flatten_detail::transformPoint(xform, t.v[c * 3 + 0], t.v[c * 3 + 1], t.v[c * 3 + 2], &worldV[c * 3]);
            const float3 v0 = toWorld(float3{(float)worldV[0], (float)worldV[1], (float)worldV[2]});
            const float3 v1 = toWorld(float3{(float)worldV[3], (float)worldV[4], (float)worldV[5]});
            const float3 v2 = toWorld(float3{(float)worldV[6], (float)worldV[7], (float)worldV[8]});
            verts.push_back(PackedFloat3{v0.x, v0.y, v0.z});
            verts.push_back(PackedFloat3{v1.x, v1.y, v1.z});
            verts.push_back(PackedFloat3{v2.x, v2.y, v2.z});
            if (t.hasNormals) {
                for (int c = 0; c < 3; ++c) {
                    double worldN[3];
                    pbrt_flatten::flatten_detail::transformNormal(xform, t.n[c * 3 + 0], t.n[c * 3 + 1], t.n[c * 3 + 2], worldN);
                    const float3 n = simd::normalize(float3{(float)worldN[0], (float)worldN[1], (float)worldN[2]});
                    normals.push_back(PackedFloat3{n.x, n.y, n.z});
                }
            } else {
                const float3 faceN = simd::normalize(simd::cross(v1 - v0, v2 - v0));
                const PackedFloat3 packedN{faceN.x, faceN.y, faceN.z};
                normals.push_back(packedN); normals.push_back(packedN); normals.push_back(packedN);
            }
            if (t.hasUVs) {
                for (int c = 0; c < 3; ++c)
                    uvs.push_back(PackedFloat2{(float)t.uv[c * 2 + 0], (float)t.uv[c * 2 + 1]});
            } else {
                uvs.push_back(PackedFloat2{0, 0}); uvs.push_back(PackedFloat2{0, 0}); uvs.push_back(PackedFloat2{0, 0});
            }
            materials.push_back(materialFor(t.material));
            ++instancedTriangleCount;
        }
        // A non-uniformly-scaled sphere is an ellipsoid, which SphereData
        // (a plain centre+radius analytic primitive) can't represent -
        // baking it as a sphere anyway would silently render the wrong
        // shape, so this loader skips instanced spheres entirely rather
        // than risk that (matching every other "explain why, don't render
        // something wrong" gap this loader already documents). No pbrt
        // scene this loader has been run against uses one yet.
        skippedInstancedSpheres += grp.spheres.size();
    }
    if (instancedTriangleCount > 0)
        fprintf(stderr, "loadPbrtScene: baked %zu ObjectInstance placement(s) into %zu world-space "
                        "triangle(s)\n", scene.instances.size(), instancedTriangleCount);
    if (skippedInstancedSpheres > 0)
        fprintf(stderr, "loadPbrtScene: %zu instanced sphere(s) skipped - a non-uniformly-scaled "
                        "instanced sphere can't be represented by this loader's analytic sphere "
                        "primitive\n", skippedInstancedSpheres);
}

// --- Punctual lights (point/spot/distant) ---------------------------
void MetalPocApp::loadPbrtPunctualLights(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld, float sceneScale) {
    // Point and spot both reuse PointLightData - the SAME GPU buffer/
    // shading path the hardcoded room's own point+spot lights already use
    // (see that struct's own comment: direction/cosOuterAngle/
    // cosInnerAngle default to "omnidirectional" unless a spot cone
    // overrides them, exactly PointLightData's own existing convention).
    // Distant reuses DirectionalLightData.
    //
    // Intensity/radiance scale compensation: this app's own point/spot
    // falloff is a real 1/distance^2 term evaluated in ITS OWN internal
    // (rescaled) coordinate space (metal_poc.metal's own `/ plDistSq`
    // division, on toWorld()-transformed positions) - leaving a
    // pbrt-authored "I" unchanged while every point/spot light's own
    // distance to a hit point shrinks by sceneScale would inflate
    // apparent brightness by 1/sceneScale^2 relative to the same scene
    // rendered at its own native scale. Multiplying by sceneScale*
    // sceneScale exactly cancels that: irradiance = I/d^2, d'=d*
    // sceneScale => I'=I*sceneScale^2 keeps I'/d'^2 == I/d^2. Distant
    // lights need no such compensation - a directional light's own
    // contribution has no distance term at all (metal_poc.metal's own
    // DirectionalLight comment), the same reason this loader's area
    // lights above also need none (their own area shrinks by
    // sceneScale^2 in lockstep with d^2, cancelling exactly).
    //
    // Goniometric/Projection real profile/slide images: the FIRST light
    // of each kind that names one gets it (section 98, below); a SECOND
    // one of the same kind, or one whose image fails to load/decode,
    // falls back to the Approx (no-image) case and is counted here.
    const float intensityScale = sceneScale * sceneScale;
    size_t skippedImageBasedLights = 0;
    for (const pbrt_flatten::PunctualLight& pl : scene.punctualLights) {
        const float3 baseEmission{(float)(pl.intensity[0] * pl.scale),
                                   (float)(pl.intensity[1] * pl.scale),
                                   (float)(pl.intensity[2] * pl.scale)};
        switch (pl.kind) {
            case pbrt_flatten::PunctualLightKind::Point: {
                const float3 pos = toWorld(float3{(float)pl.pos[0], (float)pl.pos[1], (float)pl.pos[2]});
                const float3 emission = baseEmission * intensityScale;
                pointLights.push_back(PointLightData{
                    PackedFloat3{pos.x, pos.y, pos.z},
                    PackedFloat3{emission.x, emission.y, emission.z}});
                break;
            }
            case pbrt_flatten::PunctualLightKind::Spot: {
                const float3 pos = toWorld(float3{(float)pl.pos[0], (float)pl.pos[1], (float)pl.pos[2]});
                const float3 dir = simd::normalize(float3{(float)pl.dir[0], (float)pl.dir[1], (float)pl.dir[2]});
                const float3 emission = baseEmission * intensityScale;
                pointLights.push_back(PointLightData{
                    PackedFloat3{pos.x, pos.y, pos.z},
                    PackedFloat3{emission.x, emission.y, emission.z},
                    PackedFloat3{dir.x, dir.y, dir.z},
                    /*cosOuterAngle=*/cosf((float)pl.coneAngleDeg * (float)M_PI / 180.0f),
                    /*cosInnerAngle=*/cosf((float)pl.falloffStartAngleDeg * (float)M_PI / 180.0f)});
                break;
            }
            case pbrt_flatten::PunctualLightKind::Distant: {
                // pl.dir is "wi" (toward the light, see PunctualLight's own
                // field comment); DirectionalLightData::direction is the
                // light's own direction of TRAVEL - the sign convention
                // the hardcoded room's own existing directional light
                // already establishes (its `direction` points along -z,
                // "through the room's own open front", i.e. the way the
                // light travels, not back toward its source).
                const float3 wi = simd::normalize(float3{(float)pl.dir[0], (float)pl.dir[1], (float)pl.dir[2]});
                directionalLights.push_back(DirectionalLightData{
                    PackedFloat3{-wi.x, -wi.y, -wi.z},
                    PackedFloat3{baseEmission.x, baseEmission.y, baseEmission.z}});
                break;
            }
            case pbrt_flatten::PunctualLightKind::Goniometric: {
                // A real per-light IES profile image (section 98) - only
                // the FIRST such light in the scene gets one (this loader
                // has one dedicated pbrtGoniometricTexture slot, same
                // "one shared texture" constraint the hardcoded room's
                // own single goniometricTexture already has - see
                // metal_poc.h's own havePbrtGoniometricImage comment). A
                // second one, or a decode failure, falls back to the
                // Approx case below exactly as if no filename were named
                // at all - erring toward a safe, working (if less
                // accurate) render over dropping the light entirely.
                if (pl.hadImageFilename && !havePbrtGoniometricImage) {
                    std::string bytes;
                    if (pbrt_load::loadFileNear(pbrtScenePath, pl.filename, bytes) &&
                        pbrt_load::detail::decodeInfiniteLightImage(pl.filename, bytes,
                            pbrtGoniometricImagePixels, pbrtGoniometricImageWidth, pbrtGoniometricImageHeight)) {
                        havePbrtGoniometricImage = true;
                        const float3 pos = toWorld(float3{(float)pl.pos[0], (float)pl.pos[1], (float)pl.pos[2]});
                        const float3 emission = baseEmission * intensityScale;
                        // Same equal-area octahedral mapping the hardcoded
                        // room's own goniometric light already uses
                        // (equalAreaSphereToSquare(), section 57) - a
                        // right/up frame is needed to express "local"
                        // direction the same way. A real IES profile can
                        // have genuine azimuthal (non-rotationally-
                        // symmetric) variation, so the roll matters here
                        // too, not just for Projection below - using the
                        // scene's own real worldToLight-derived up
                        // (punctualLightWorldUp(), section 98) as the
                        // worldUp hint into the SAME cross-product formula
                        // makeGoniometricLight() already uses recovers
                        // that real roll instead of guessing at it.
                        const float3 forward = punctualLightWorldForward(pl.worldToLight);
                        const float3 worldUpHint = punctualLightWorldUp(pl.worldToLight);
                        const float3 right = simd::normalize(simd::cross(forward, worldUpHint));
                        const float3 up = simd::cross(right, forward);
                        goniometricLights.push_back(GoniometricLightData{
                            PackedFloat3{pos.x, pos.y, pos.z},
                            PackedFloat3{forward.x, forward.y, forward.z},
                            PackedFloat3{right.x, right.y, right.z},
                            PackedFloat3{up.x, up.y, up.z},
                            PackedFloat3{emission.x, emission.y, emission.z},
                            // pl.scale is ALREADY folded into `emission`
                            // above (baseEmission = intensity*pl.scale,
                            // see this function's own top-of-loop
                            // comment) - GoniometricLightData::scale is
                            // an INDEPENDENT multiplier the shader applies
                            // on top (the hardcoded room's own light,
                            // above, passes emission=I with no pl.scale
                            // baked in and a real scale=1.0f here);
                            // passing pl.scale a second time here would
                            // double-count it.
                            /*scale=*/1.0f,
                            /*usePbrtTexture=*/1u});
                        break;
                    }
                    fprintf(stderr, "loadPbrtScene: goniometric light's profile image '%s' could not be "
                                    "read/decoded; falling back to the Approx (uniform) case\n", pl.filename.c_str());
                }
                if (pl.hadImageFilename) ++skippedImageBasedLights;
                // No profile image named (or a second one/a decode
                // failure, both handled above) - pbrt-v4's own documented
                // "Approx" fallback (uniform isotropic intensity, see
                // pbrt_scenes/punctual-lights.pbrt's own header comment
                // for why this is the common case, not just a
                // simplification), which is EXACTLY what a plain
                // omnidirectional PointLightData already represents -
                // isotropic has no direction to get right at all, unlike
                // Projection's own cone/aim (still deferred below), so
                // this needed no new geometry work.
                const float3 pos = toWorld(float3{(float)pl.pos[0], (float)pl.pos[1], (float)pl.pos[2]});
                const float3 emission = baseEmission * intensityScale;
                pointLights.push_back(PointLightData{
                    PackedFloat3{pos.x, pos.y, pos.z},
                    PackedFloat3{emission.x, emission.y, emission.z}});
                break;
            }
            case pbrt_flatten::PunctualLightKind::Projection: {
                // A real per-light slide image (section 98) - same "one
                // dedicated slot, first light wins, decode failure falls
                // back to Approx" reasoning as Goniometric above.
                if (pl.hadImageFilename && !havePbrtProjectionImage) {
                    std::string bytes;
                    if (pbrt_load::loadFileNear(pbrtScenePath, pl.filename, bytes) &&
                        pbrt_load::detail::decodeInfiniteLightImage(pl.filename, bytes,
                            pbrtProjectionImagePixels, pbrtProjectionImageWidth, pbrtProjectionImageHeight)) {
                        havePbrtProjectionImage = true;
                        const float3 pos = toWorld(float3{(float)pl.pos[0], (float)pl.pos[1], (float)pl.pos[2]});
                        const float3 forward = punctualLightWorldForward(pl.worldToLight);
                        // Same look-at-style right/up derivation
                        // makeProjectionLight() already uses for the
                        // hardcoded room's own light, but with the
                        // scene's own REAL worldToLight-derived up
                        // (punctualLightWorldUp(), section 98) as the
                        // worldUp hint instead of a generic axis - a
                        // projected slide has a real, meaningful roll
                        // (the image's own up direction), which only the
                        // scene's own rotation can supply correctly.
                        const float3 worldUpHint = punctualLightWorldUp(pl.worldToLight);
                        const float3 right = simd::normalize(simd::cross(forward, worldUpHint));
                        const float3 up = simd::cross(right, forward);
                        // pl.fovDeg is pbrt's own single "fov" parameter,
                        // which real pbrt-v4 (and this project's own
                        // src/shared/projection_light.h port, already
                        // proven on the CPU/OptiX backends - see its own
                        // "screenBounds"/"aspect" comments) always applies
                        // to the SHORTER image axis: screen bounds are
                        // [-aspect,aspect]x[-1,1] when aspect=width/height
                        // >= 1 (a wide image, so Y/vertical is the fixed,
                        // fov-sized axis and X/horizontal is the wider
                        // one), or [-1,1]x[-1/aspect,1/aspect] when
                        // aspect < 1 (a tall image, X fixed, Y taller).
                        // tanHalfFovBase below is that fixed axis's own
                        // half-angle; the other axis scales by aspect (or
                        // 1/aspect) exactly as projection_light.h derives.
                        const float tanHalfFovBase = tanf((float)pl.fovDeg * 0.5f * (float)M_PI / 180.0f);
                        const float imageAspect = (pbrtProjectionImageHeight > 0)
                            ? (float)pbrtProjectionImageWidth / (float)pbrtProjectionImageHeight : 1.0f;
                        const bool wide = imageAspect >= 1.0f;
                        const float tanHalfFovX = wide ? tanHalfFovBase * imageAspect : tanHalfFovBase;
                        const float tanHalfFovYFinal = wide ? tanHalfFovBase : tanHalfFovBase / imageAspect;
                        projectionLights.push_back(ProjectionLightData{
                            PackedFloat3{pos.x, pos.y, pos.z},
                            PackedFloat3{forward.x, forward.y, forward.z},
                            PackedFloat3{right.x, right.y, right.z},
                            PackedFloat3{up.x, up.y, up.z},
                            tanHalfFovX,
                            tanHalfFovYFinal,
                            (float)pl.scale,
                            /*usePbrtTexture=*/1u});
                        break;
                    }
                    fprintf(stderr, "loadPbrtScene: projection light's slide image '%s' could not be "
                                    "read/decoded; falling back to the Approx (uniform beam) case\n", pl.filename.c_str());
                }
                if (pl.hadImageFilename) ++skippedImageBasedLights;
                // No slide image named - pbrt-v4's own documented
                // "Approx" fallback: a uniform white cone-shaped beam
                // (ProjectionLight::make_uniform(), matching this
                // project's own CPU builder - see pbrt_cpu_builder.h's
                // own kUniformSlide comment). Represented here as a
                // hard-edged PointLightData spot (cosOuterAngle ==
                // cosInnerAngle - spotLightFalloff()'s own
                // max(...,1e-6) denominator guard keeps this a clean
                // cutoff, not a divide-by-zero) rather than a genuinely
                // new light type: a uniform intensity within a cone and
                // zero outside it is exactly what a spot cone already
                // is, just without the smoothstep edge softening a real
                // spot's inner/outer split gives.
                //
                // Aim direction: see punctualLightWorldForward()'s own
                // comment (metal_poc_host_math.h) for the full
                // derivation - verified there against a known rotation
                // case, not just by hand here.
                const float3 pos = toWorld(float3{(float)pl.pos[0], (float)pl.pos[1], (float)pl.pos[2]});
                const float3 dir = punctualLightWorldForward(pl.worldToLight);
                const float3 emission = baseEmission * intensityScale;
                const float cosHalfFov = cosf(0.5f * (float)pl.fovDeg * (float)M_PI / 180.0f);
                pointLights.push_back(PointLightData{
                    PackedFloat3{pos.x, pos.y, pos.z},
                    PackedFloat3{emission.x, emission.y, emission.z},
                    PackedFloat3{dir.x, dir.y, dir.z},
                    /*cosOuterAngle=*/cosHalfFov,
                    /*cosInnerAngle=*/cosHalfFov});
                break;
            }
            default:
                break;
        }
    }
    if (skippedImageBasedLights > 0)
        fprintf(stderr, "loadPbrtScene: %zu goniometric/projection light(s) with a real profile/slide "
                        "image fell back to the Approx (uniform) case - only the first light of each "
                        "kind gets its own image, and a failed decode also falls back here\n",
                skippedImageBasedLights);
}

// --- Homogeneous participating medium (fog) -------------------------
// scene.cameraMediumIndex is already fully resolved and validated by
// pbrt_flatten.h's own post-pass (homogeneous type only, and only set
// at all when the scene has no conflicting real per-shape medium -
// see that field's own comment) - a direct, safe read, no further
// checking needed here.
void MetalPocApp::loadPbrtMedium(const pbrt_flatten::FlatScene& scene, float sceneScale) {
    if (scene.cameraMediumIndex >= 0 &&
        (size_t)scene.cameraMediumIndex < scene.media.size()) {
        const pbrt_flatten::Medium& m = scene.media[(size_t)scene.cameraMediumIndex];
        double sigmaT[3];
        double meanSigmaT = 0.0;
        for (int c = 0; c < 3; ++c) {
            sigmaT[c] = m.sigma_a[c] + m.sigma_s[c];
            meanSigmaT += sigmaT[c];
        }
        meanSigmaT /= 3.0;
        // Extinction has units of inverse length - the SAME rescale that
        // shrinks every position/distance by sceneScale (see this
        // function's own toWorld() comment) shrinks a ray's own travelled
        // distance in lockstep, so leaving sigmaT at its pbrt-native value
        // would UNDER-attenuate by the same factor (optical depth =
        // sigmaT*dist; dist'=dist*sceneScale, so sigmaT'=sigmaT/sceneScale
        // is what keeps sigmaT'*dist' == sigmaT*dist). The OPPOSITE
        // direction from the punctual-light intensity compensation above
        // (which multiplies by sceneScale^2) - that one compensates a
        // squared-length falloff term, this one a single inverse-length
        // one, so the correction is an inverse first power, not a square.
        havePbrtMedium = true;
        // The LOCAL `sceneScale` (computed above, already applied to
        // every vertex/light/camera position via toWorld()) - not the
        // MEMBER `pbrtSceneScale`, which this same function only
        // assigns much later, in its own Camera section below (for
        // applyCameraOverride()'s later use). Reading the member here
        // was a real, previously-latent bug: since loadPbrtScene() runs
        // exactly once, that member is still its 1.0f default-
        // initializer value at this point in EVERY call, so every real
        // pbrt scene's own fog silently used 1/sceneScale too little
        // attenuation (e.g. ~10x for camera-medium.pbrt's own ~20-unit
        // scale, worse for a larger scene) - found via code review
        // while mapping this function for section 108's own refactor,
        // not a symptom (the fog was still visibly present just too
        // faint, and PR #88's own on/off verification wasn't sensitive
        // to the WRONG-MAGNITUDE case, only presence/absence).
        pbrtFogSigmaT = (float)(meanSigmaT / sceneScale);
        // fogAlbedo is single-scattering albedo (sigma_s/sigma_t) PER
        // CHANNEL (metal_poc.metal's own field comment) - unlike sigmaT
        // itself, this ratio is dimensionless and scale-invariant, so the
        // real per-channel colour survives even though the overall
        // interaction RATE above is reduced to one achromatic scalar -
        // the same simplification this shader's own hardcoded-room fog
        // already makes (a single fogSigmaT, never a per-channel one).
        for (int c = 0; c < 3; ++c) {
            const float t = (float)sigmaT[c];
            pbrtFogAlbedo[c] = (t > 1e-9f) ? (float)(m.sigma_s[c] / sigmaT[c]) : 0.0f;
        }
        pbrtFogAsymmetryG = (float)m.g;   // dimensionless, no scale compensation needed
        fprintf(stderr, "loadPbrtScene: homogeneous camera medium found (sigma_t~%.5g/unit, g=%.3f)\n",
                meanSigmaT, m.g);
    }
}

// --- Infinite light ---------------------------------------------------
// earthTexture is ALSO the hardcoded room's own materialType-3
// back-wall albedo, sampled from a totally separate code path in
// primaryRayKernel (the `albedo = earthTexture.sample(...)` line, run
// BEFORE any of the 6 material-shading functions are even called) -
// repointing that one at a pbrt-provided image would silently corrupt
// that unrelated, still-active geometry. So this uses its OWN,
// genuinely separate texture (pbrtEnvTexture) instead - see
// primaryRayKernel's own texture-argument comment.
//
// Deliberately miss-path-only for BOTH the constant-colour and
// image cases (no NEE/MIS light-sampling strategy) - see
// metal_poc.metal's own mirrored comments on why that's accepted
// scope, not an oversight; a future NEE upgrade already has a
// tested building block waiting (metal_poc_host_math.h's own
// float-RGB buildEnvDistribution2D() overload, added but not yet
// wired to anything - the exact same "phase 1 before phase 2"
// staging earthTexture's own NEE support went through, sections
// 69/71).
void MetalPocApp::loadPbrtInfiniteLight(const pbrt_flatten::FlatScene& scene) {
    if (scene.infiniteLight.present) {
        if (scene.infiniteLight.imageWidth > 0 && scene.infiniteLight.imageHeight > 0 &&
            !scene.infiniteLight.imagePixels.empty()) {
            havePbrtImageEnvLight = true;
            pbrtEnvImageWidth = scene.infiniteLight.imageWidth;
            pbrtEnvImageHeight = scene.infiniteLight.imageHeight;
            // scale applied here (not left for a shader-side multiply) -
            // matches how the constant-colour case below bakes L*scale
            // at load time too, and how src/TheRestOfYourLife/
            // pbrt_cpu_builder.h's own sky_light(...) constructor takes
            // scale as a SEPARATE multiplier on the raw image samples
            // (not pre-baked into scene.infiniteLight.imagePixels itself).
            const float scale = (float)scene.infiniteLight.scale;
            pbrtEnvImagePixels = scene.infiniteLight.imagePixels;
            for (float& v : pbrtEnvImagePixels) v *= scale;
            fprintf(stderr, "loadPbrtScene: image-based infinite light found (%dx%d, scale=%.3g)\n",
                    pbrtEnvImageWidth, pbrtEnvImageHeight, scale);
        } else {
            havePbrtConstantEnvLight = true;
            pbrtEnvColor = float3{(float)(scene.infiniteLight.L[0] * scene.infiniteLight.scale),
                                   (float)(scene.infiniteLight.L[1] * scene.infiniteLight.scale),
                                   (float)(scene.infiniteLight.L[2] * scene.infiniteLight.scale)};
            fprintf(stderr, "loadPbrtScene: constant-colour infinite light found (L*scale ~ %.3g %.3g %.3g)\n",
                    pbrtEnvColor.x, pbrtEnvColor.y, pbrtEnvColor.z);
        }
    }
    // Disks are now handled above (section 101, common full-circle case);
    // cylinder/cone/paraboloid/bilinearmesh/curve remain a real gap.
    if (!scene.cylinders.empty() || !scene.cones.empty() ||
        !scene.paraboloids.empty() || !scene.bilinearPatches.empty() || !scene.curves.empty())
        fprintf(stderr, "loadPbrtScene: cylinder/cone/paraboloid/bilinearmesh/curve shapes skipped - "
                        "only triangle mesh, sphere, and (full-circle) disk shapes are supported by "
                        "this POC's scene loader yet\n");
}

// --- Camera ------------------------------------------------------------
void MetalPocApp::loadPbrtCamera(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld,
    float3 bboxCenter, float sceneScale, float3 sceneOffset) {
    const pbrt_flatten::Camera& cam = scene.camera;
    // lookfrom/lookat are positions, scaled the same as every vertex
    // above; `up` is a direction (scale-invariant, and normalized below
    // regardless).
    const float3 lookfrom = toWorld(float3{(float)cam.lookfrom[0], (float)cam.lookfrom[1], (float)cam.lookfrom[2]});
    const float3 lookat = toWorld(float3{(float)cam.lookat[0], (float)cam.lookat[1], (float)cam.lookat[2]});
    const float3 up{(float)cam.up[0], (float)cam.up[1], (float)cam.up[2]};
    const float3 forward = simd::normalize(lookat - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    pbrtTanHalfFov = tanf(0.5f * (float)cam.vfov * (float)M_PI / 180.0f);
    havePbrtCamera = true;
    // Saved for applyCameraOverride() - see that method's own comment.
    pbrtCameraLookAtWorld = lookat;
    pbrtCameraUpRaw = up;
    pbrtBboxCenter = bboxCenter;
    pbrtSceneScale = sceneScale;
    pbrtSceneOffset = sceneOffset;

    // Perspective/orthographic thin-lens DOF ("float lensradius"/"float
    // focaldistance", pbrt_flatten.h's own c.aperture/c.focusDistance -
    // c.aperture is already lensradius*2, a world-space DIAMETER, see
    // that field's own comment) - D9's own real gap (section 159): every
    // hand-authored D1/D5/D6 scene already sets pbrtLensRadius/
    // pbrtFocusDistance directly (defaulting to 0/1, "no DOF"), but a
    // REAL loaded pbrt file's own lensradius/focaldistance parameters
    // were never actually read here at all until now - depth-of-
    // field.pbrt rendered pinhole-sharp regardless of its own "float
    // lensradius" [20] before this fix. Same sceneScale-multiplied
    // world-space-distance convention pbrtLensRadius/pbrtFocusDistance
    // already use everywhere else (D5's own comment). Spherical/
    // Realistic cameras never read lensRadius/focusDistance at all
    // (Uniforms::cameraRealistic's own DOF-skip guard, metal_poc_kernel.
    // metal) - deliberately NOT set for those two below, matching real
    // pbrt (RealisticCamera has its own, wholly different depth-of-
    // field mechanism built into the lens trace itself; SphericalCamera
    // has none).
    if (cam.type == "perspective" || cam.type == "orthographic") {
        pbrtLensRadius = (float)(cam.aperture * 0.5) * sceneScale;
        pbrtFocusDistance = (float)pbrt_flatten::focusDistanceFor(cam) * sceneScale;
    }

    // Non-perspective camera types loaded from a REAL pbrt file (D10/
    // D11/D12, section 159) - the generic, data-driven counterpart to
    // D2/D6 (orthographic)/D3/D7 (spherical)/D4/D8 (realistic)'s own
    // hand-authored scenes, which set these same havePbrt*/pbrt* fields
    // directly instead of reading them from a Camera directive's own
    // parameters. Mirrors gpu/optix/scene_builder.cpp's own generic
    // Camera-type dispatch (the already-shipped CUDA reference this
    // block is ported from) - same fields, same formulas, just this
    // loader's own Objective-C++ types/RealisticCamera<float> instead
    // of OptiX's CUDA structs.
    if (cam.type == "orthographic") {
        // Mirrors D6's own buildOrthoCornellBox() - pbrtTanHalfFov
        // repurposed as the orthographic screen window's own
        // (necessarily symmetric - Uniforms::cameraOrthographic's own
        // comment, metal_poc_types.metal) world-space half-extent, in
        // THIS loader's own rescaled units. `screenwindow`'s own
        // xmax (== -xmin for every screenwindow this loader's own
        // scenes actually use, including orthographic-camera.pbrt's
        // own symmetric [-320,320,-320,320]) supplies it directly when
        // given; pbrt's own "roughly 1-unit-across" default otherwise
        // (pbrt_flatten::Camera::screenWindow's own comment).
        havePbrtOrthographic = true;
        pbrtTanHalfFov = (cam.hasScreenWindow ? (float)cam.screenWindow[1] : 1.0f) * sceneScale;
    } else if (cam.type == "spherical" || cam.type == "environment") {
        // Mirrors D7's own buildSphericalCornellBox() - a panoramic
        // camera has no screen window/FOV/DOF at all, only the
        // position/orientation already set above. `sphericalMapping`
        // additionally selects EquiRectangular (pbrt-v4's own default,
        // matching every hand-authored D3/D7 scene already) vs.
        // EqualArea (spherical-camera.pbrt's own "string mapping"
        // ["equalarea"] - the first scene, hand-authored or loaded,
        // to actually request it - Uniforms::sphericalMappingEqualArea's
        // own comment).
        havePbrtSpherical = true;
        havePbrtSphericalEqualArea = (cam.sphericalMapping == "equalarea");
    } else if (cam.type == "realistic") {
        // Mirrors D4/D8's own buildRealisticCameraScene()/
        // buildRealisticCornellBox() construction exactly, just with the
        // lens TABLE itself read from a real file (pbrt_load::
        // loadFileNear()/parseLensFile(), both already shared with CPU/
        // OptiX - the "half the lensfile-loading path a compiled-in
        // scene never exercised" gap D12's own registry description
        // names) instead of a literal std::vector in this function's own
        // source. Film half-extents are DERIVED from filmDiagonalMM +
        // this render's own aspect ratio (pbrt-v4's real convention,
        // and gpu/optix/scene_builder.cpp's own identical formula) -
        // D4/D8's own scenes instead gave film_x_mm/film_y_mm directly,
        // since a hand-authored scene has no separate "diagonal"
        // parameter to derive them from.
        std::string lensText;
        if (cam.lensFile.empty()) {
            fprintf(stderr, "loadPbrtCamera: a realistic camera has no \"lensfile\"; "
                            "rendering as perspective instead (should not normally be "
                            "reachable - pbrt_flatten.h already falls back to "
                            "\"perspective\" at flatten time when lensfile is missing).\n");
        } else if (!pbrt_load::loadFileNear(pbrtScenePath, cam.lensFile, lensText)) {
            fprintf(stderr, "loadPbrtCamera: realistic camera lensfile '%s' not found "
                            "near '%s'; rendering as perspective instead.\n",
                    cam.lensFile.c_str(), pbrtScenePath.c_str());
        } else {
            const std::vector<double> lensD = pbrt_load::parseLensFile(lensText);
            if (lensD.empty()) {
                fprintf(stderr, "loadPbrtCamera: realistic camera lensfile '%s' has no "
                                "usable rows; rendering as perspective instead.\n",
                        cam.lensFile.c_str());
            } else {
                std::vector<float> lensParams;
                lensParams.reserve(lensD.size());
                for (double v : lensD) lensParams.push_back((float)v);
                const float aspectF = (height > 0) ? (float)width / (float)height : 1.0f;
                const float halfY = (float)cam.filmDiagonalMM / (2.0f * sqrtf(aspectF * aspectF + 1.0f));
                const float halfX = aspectF * halfY;
                const float focusDist = (float)pbrt_flatten::focusDistanceFor(cam);
                const float apertureDiameter = (float)cam.apertureDiameterMM;
                RealisticCamera<float> realCam(Mat4<float>{}, halfX, halfY, focusDist,
                                                apertureDiameter, lensParams);
                realisticLensElements.clear();
                for (int i = 0; i < realCam.num_elements(); ++i) {
                    realisticLensElements.push_back(GpuLensElementData{
                        realCam.lens_curvature_radius(i), realCam.lens_thickness(i),
                        realCam.lens_eta(i), realCam.lens_aperture_radius(i)});
                }
                realisticExitPupilBounds.clear();
                for (int i = 0; i < realCam.num_exit_pupil_bounds(); ++i) {
                    realisticExitPupilBounds.push_back(GpuExitPupilBoundsData{
                        realCam.exit_pupil_xmin(i), realCam.exit_pupil_xmax(i),
                        realCam.exit_pupil_ymin(i), realCam.exit_pupil_ymax(i),
                        realCam.exit_pupil_degenerate(i) ? 1u : 0u});
                }
                realisticFilmHalfX = realCam.film_half_x();
                realisticFilmHalfY = realCam.film_half_y();
                realisticLensRearZ = realCam.lens_rear_z();
                havePbrtRealisticCamera = true;
            }
        }
    }
}

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
    if (scene_id == "F1") { buildBilinearPatchScene(); return true; }
    if (scene_id == "F4") { buildCurveFibersScene(); return true; }
    if (scene_id == "E1") { buildHomogeneousMediumScene(); return true; }
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

    MTLPrimitiveAccelerationStructureDescriptor* sphereAccelDesc =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    sphereAccelDesc.geometryDescriptors = @[bboxGeomDesc, diskGeomDesc];

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

    // The intersection function has to be LINKED into the compute
    // pipeline (MTLLinkedFunctions) before an MTLIntersectionFunction
    // Table naming it can be built - a plain newComputePipelineState
    // WithFunction: (used for step 1/2's triangle-only pipeline) has
    // nowhere to put that linkage, hence the switch to the descriptor-
    // based pipeline creation call here.
    MTLComputePipelineDescriptor* pipelineDesc = [MTLComputePipelineDescriptor new];
    pipelineDesc.computeFunction = kernelFn;
    MTLLinkedFunctions* linkedFns = [MTLLinkedFunctions new];
    linkedFns.functions = @[sphereIntersectFn, diskIntersectFn];
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
    fnTableDesc.functionCount = 2;
    id<MTLIntersectionFunctionTable> functionTable = [pipeline newIntersectionFunctionTableWithDescriptor:fnTableDesc];
    id<MTLFunctionHandle> sphereHandle = [pipeline functionHandleWithFunction:sphereIntersectFn];
    id<MTLFunctionHandle> diskHandle = [pipeline functionHandleWithFunction:diskIntersectFn];
    [functionTable setFunction:sphereHandle atIndex:0];
    [functionTable setFunction:diskHandle atIndex:1];
    // sphereIntersectionFunction/diskIntersectionFunction each read
    // their own geometry buffer (metal_poc.metal buffer(0)/buffer(1)
    // respectively - a SEPARATE argument table from the calling
    // kernel's own buffer(0..14), see that file's own comment) -
    // bound here, on the function table, not on the compute encoder.
    [functionTable setBuffer:sphereBuffer offset:0 atIndex:0];
    [functionTable setBuffer:diskBuffer offset:0 atIndex:1];

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
        uniforms.cameraVelocity = PackedFloat3{0, 0, 0};   // no motion blur
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
