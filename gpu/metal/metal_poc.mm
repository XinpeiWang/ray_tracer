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
// resolve_crop_pixel_bounds() - the shared NDC-fraction -> pixel-rectangle
// resolver OptiX already uses for --crop, so Metal rounds/clamps identically.
#include "../../src/shared/cameras.h"
// ERR_FILE_WRITE_FAILED - so a failed output write reaches the launcher/GUI as
// the specific "check the folder is writable" error instead of a generic 1.
#include "../../src/TheRestOfYourLife/error_codes.h"
// is_exr_output_path()/write_exr_image() - the same shared helpers the CPU
// and OptiX backends use to honor a ".exr" output path (postProcessAndWrite()
// below). Its own tinyexr.h include is a no-op here: that header's include
// guard is already set by the TINYEXR_IMPLEMENTATION include above.
#include "../../src/shared/exr_writer.h"
#include "../../src/shared/cornell_box_data.h"
// kConductorAu/kConductorCu - real (eta,k) presets for B7's own lacquered-
// gold sphere/lacquered-copper box (buildCornellCoatedConductor()) -
// reused directly rather than re-transcribed by hand, same as this
// file's own cornell_box_data.h include just above.
#include "../../src/shared/conductor_data.h"
// metal_render_main()'s own callable signature (section 79) - matches
// gpu/optix/optix_interface.h's own optix_render_main() shape exactly,
// down to reusing this SAME struct, so launcher/main.cpp's --gpu dispatch
// (already linked and calling this, on a RT_HAVE_METAL build) needs no
// Metal-specific parameter shape of its own.
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
    // isolatePbrtLighting (RenderOptions::isolate_pbrt_lighting, section
    // 197/199 - see MetalPocApp::isolatePbrtLighting's own comment) skips
    // every hardcoded-room light below, this pair included, so a loaded
    // pbrt scene's own lighting can be judged without this room's
    // lights competing with it. False (default) leaves this block
    // running exactly as before - unaffected for every existing caller.
    if (!isolatePbrtLighting) {
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
    }
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
    // Same isolatePbrtLighting gate as the two addAreaLight() calls
    // above - see that comment. Skips this pair, the sun, the
    // projection light, and the goniometric light below (5 blocks
    // total), leaving only whatever loadPbrtScene() itself pushed.
    if (!isolatePbrtLighting) {
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
    }  // if (!isolatePbrtLighting) - see this block's own opening comment.
    // Deliberately OUTSIDE the gate above, unlike the goniometricLights
    // light itself just above: buildGPUResources() (metal_poc_gpu_resources.mm)
    // unconditionally builds a goniometricTexture sized goniometricImageSize
    // x goniometricImageSize, with no zero-size guard - the same reason
    // `exitPupilBoundsBuffer` just above it falls back to a valid dummy
    // buffer rather than a zero-length one when realisticExitPupilBounds is
    // empty. Skipping this too (under isolatePbrtLighting, with a pbrt
    // scene that has no real goniometric light of its own, e.g. it failed
    // to load or simply has none) left goniometricImageSize at its 0
    // default and crashed MTLTextureDescriptor's own validation - caught by
    // actually running the new flag against C9, not assumed safe from
    // reading the code alone. The room's own light being skipped above
    // means this placeholder image is simply never sampled in that case,
    // same "built but unused" shape as the dummy exitPupilBoundsBuffer.
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
    if (scene_id == "E2") { buildCloudMediumScene(); return true; }
    if (scene_id == "E3") { buildDielectricMediumShowcase(); return true; }
    if (scene_id == "E4") { buildRgbGridMediumScene(); return true; }
    if (scene_id == "B9") { buildCornellCrystal(); return true; }
    if (scene_id == "B5") { buildCornellCoatedDiffuse(); return true; }
    if (scene_id == "B7") { buildCornellCoatedConductor(); return true; }
    if (scene_id == "B12") { buildNormalMappedCornell(); return true; }
    if (scene_id == "B23") { buildPrismDispersion(); return true; }
    if (scene_id == "B24") { buildPrismDispersionRough(); return true; }
    if (scene_id == "B10") { buildPrincipledShowcase(); return true; }
    if (scene_id == "B11") { buildHairFibersScene(); return true; }
    if (scene_id == "B14") { buildMeasuredBrdfScene(); return true; }
    if (scene_id == "B13") { buildSubsurfaceSlab(); return true; }
    if (scene_id == "A9") { buildFinalScene(); return true; }
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
// MetalPocApp::buildGPUResources() - split out into its own file,
// metal_poc_gpu_resources.mm, once this file's own Stage 3/4 pair grew
// past ~1200 combined lines (pure code motion, no behaviour change - see
// that file's own header comment, and docs/METAL_GPU_FEASIBILITY.md).

// --- Stage 4: compile the shader, dispatch the render, read back -------
// MetalPocApp::compileShaderAndDispatch() (plus its own checkGpuResource()
// helper) - split out into its own file, metal_poc_dispatch.mm, alongside
// the Stage 3 split just above (same reasoning - see that file's own
// header comment, and docs/METAL_GPU_FEASIBILITY.md).


// --- Stage 5: post-process the linear HDR buffer and write the PNG -----
// (chromatic aberration, then lens vignette, then the selected tonemap
// operator, then the real sRGB OETF, then bilateral denoise)
bool MetalPocApp::postProcessAndWrite() {
    // ".exr" output: linear HDR radiance straight from the GPU buffer, no
    // tonemap/vignette/chromatic-aberration/denoise/quantization - the same
    // contract the CPU and OptiX backends' EXR path has (see
    // src/shared/exr_writer.h). Without this branch a ".exr" path silently
    // received 8-bit PNG bytes from stbi_write_png() below.
    if (is_exr_output_path(outPath)) {
        std::vector<float> rgb(size_t(width) * height * 3);
        for (size_t i = 0; i < size_t(width) * height; ++i) {
            rgb[i * 3 + 0] = pixels[i * 4 + 0];
            rgb[i * 3 + 1] = pixels[i * 4 + 1];
            rgb[i * 3 + 2] = pixels[i * 4 + 2];
        }
        std::string exrError;
        if (write_exr_image(outPath, rgb.data(), int(width), int(height), exrError)) {
            fprintf(stderr, "Wrote %s (%ux%u, linear HDR EXR)\n", outPath, width, height);
            return true;
        }
        fprintf(stderr, "ERROR: failed to write EXR %s: %s\n", outPath, exrError.c_str());
        return false;
    }

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

    // stbi_write_png() returns 0 on failure (unwritable/nonexistent
    // directory, read-only volume, ...) - previously ignored, so a failed
    // write still printed "Wrote ..." and both callers returned success,
    // and launcher/main.cpp then reported "Render complete" for a file
    // that was never created.
    if (!stbi_write_png(outPath, width, height, 3, denoised.data(), width * 3)) {
        fprintf(stderr, "ERROR: failed to write PNG %s (check the folder exists and is writable)\n", outPath);
        return false;
    }
    fprintf(stderr, "Wrote %s (%ux%u)\n", outPath, width, height);
    return true;
}

// --- Callable entry point (phase 2 of real GPU integration - see
// docs/METAL_GPU_FEASIBILITY.md's own section on this) -------------------
// Signature matches gpu/optix/optix_interface.h's own optix_render_main()
// exactly, down to reusing the same RenderOptions struct. launcher/
// main.cpp's own --gpu dispatch (its RT_HAVE_METAL branch) calls straight
// into this now - see CMakeLists.txt's own RT_BUILD_METAL block for the
// link step that makes that resolve, and metal_interface.h's own top
// comment ("WIRED IN").
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
        app.isolatePbrtLighting = options.isolate_pbrt_lighting;
        // --seed: options.seed < 0 means "not requested" (leave the default
        // stream). Otherwise offset by 2 so seed 0 maps to 2, never to the
        // default stream's own 1u - every explicit
        // seed selects a stream distinct from a run that never passed one.
        if (options.seed >= 0) app.frameSeedValue = (uint32_t)options.seed + 2u;
        // --crop: NDC fractions -> pixel bounds via the SAME shared resolver
        // OptiX uses (src/shared/cameras.h), so rounding/clamping match. All
        // four at their defaults (0,0,1,1) means not requested - leave the crop
        // fields unset (full image, byte-identical to before crop existed).
        if (options.crop_x0 != 0.0 || options.crop_y0 != 0.0 ||
            options.crop_x1 != 1.0 || options.crop_y1 != 1.0) {
            const CropPixelBounds b = resolve_crop_pixel_bounds(
                options.crop_x0, options.crop_x1, options.crop_y0, options.crop_y1,
                image_width, image_height);
            if (b.wasDegenerate) {
                fprintf(stderr, "Warning: --crop resolves to an empty pixel range at %dx%d - "
                                "rendering the full frame instead.\n", image_width, image_height);
            } else {
                app.cropX0 = b.x0; app.cropX1 = b.x1; app.cropY0 = b.y0; app.cropY1 = b.y1;
            }
        }
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
        if (!app.postProcessAndWrite()) return ERR_FILE_WRITE_FAILED;
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
        if (!app.postProcessAndWrite()) return ERR_FILE_WRITE_FAILED;
    }
    return 0;
}
