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
    uint32_t lightCount;
    float lensRadius;
    float focusDistance;
    uint32_t apertureBlades;
    PackedFloat3 cameraVelocity;
    float fogSigmaT;
    PackedFloat3 fogAlbedo;
    uint32_t useEnvironmentMap;
    float fogAsymmetryG;
    uint32_t pointLightCount;
    uint32_t directionalLightCount;
    uint32_t projectionLightCount;
    uint32_t goniometricLightCount;
    uint32_t adaptiveSampling;
    // Environment-map importance sampling - see metal_poc.metal's own
    // mirrored comment.
    uint32_t envMapWidth = 0;
    uint32_t envMapHeight = 0;
    // GGX multi-scatter energy-compensation table dimensions - see
    // metal_poc.metal's own mirrored comment.
    uint32_t ggxEnergyRoughRes = 0;
    uint32_t ggxEnergyMuRes = 0;
    // A pbrt-loaded scene's own LightSource "infinite" with no image
    // (constant "rgb L"/"scale" only) - see loadPbrtScene()'s own
    // comment. Deliberately miss-path-only, no NEE/MIS strategy (unlike
    // useEnvironmentMap's own earthTexture-based one) - see
    // metal_poc.metal's own mirrored comment for why that's a real,
    // accepted scope cut, not an oversight. Mutually exclusive with
    // useEnvironmentMap in practice (that one is forced to 0 for every
    // pbrt-loaded scene), but not asserted as such here.
    uint32_t pbrtHasConstantEnvLight = 0;
    PackedFloat3 pbrtEnvColor{0, 0, 0};
    // A pbrt-loaded scene's own image-based LightSource "infinite" -
    // mutually exclusive with pbrtHasConstantEnvLight above. Reads
    // pbrtEnvTexture via a plain equirectangular lookup - see
    // metal_poc.metal's own mirrored comment.
    uint32_t pbrtHasImageEnvLight = 0;
    // pbrtEnvTexture's own SEPARATE EnvDistribution2D dimensions (section
    // 96) - see metal_poc.metal's own mirrored comment on
    // Uniforms::pbrtEnvMapWidth.
    uint32_t pbrtEnvMapWidth = 0;
    uint32_t pbrtEnvMapHeight = 0;
};

// AreaLightData/buildPowerLightSampler now live in metal_poc_host_math.h
// (included above) - see that header's own comment.

// buildPowerLightSampler()'s own optional per-light diagnostic line,
// preserved exactly as it read before the header extraction - passed in
// as a callback rather than baked into the (now shared, test-included)
// header itself, so metal_poc_math_tests.cpp's own calls don't spam
// stderr on every test run.
static void logPowerLightSamplerLine(int i, double power, double pmf, double uniformPmf) {
    fprintf(stderr, "Light %d: power=%.4f pmf=%.4f (uniform would be %.4f)\n", i, power, pmf, uniformPmf);
}

// Mirrors metal_poc.metal's PointLight byte-for-byte.
struct PointLightData {
    PackedFloat3 position;
    PackedFloat3 emission;
    // Defaults make every point light omnidirectional (see
    // metal_poc.metal's own PointLight/spotLightFalloff() comments) unless
    // explicitly overridden - the existing point light's own literal
    // doesn't need to change at all for this to stay backward compatible.
    PackedFloat3 direction = PackedFloat3{0.0f, -1.0f, 0.0f};
    float cosOuterAngle = -1.0f;
    float cosInnerAngle = -1.0f;
};

// Mirrors metal_poc.metal's DirectionalLight byte-for-byte.
struct DirectionalLightData {
    PackedFloat3 direction;
    PackedFloat3 emission;
};

// Mirrors metal_poc.metal's ProjectionLight byte-for-byte.
struct ProjectionLightData {
    PackedFloat3 position;
    PackedFloat3 forward;
    PackedFloat3 right;
    PackedFloat3 up;
    float tanHalfFovX;
    float tanHalfFovY;
    float scale;
    // True for a pbrt-loaded light with its own real slide image (section
    // 98) - selects pbrtProjectionTexture instead of the room's own
    // shared earthTexture at the shading call site. False (every light
    // before this one, including every hardcoded-room light and every
    // pbrt-loaded Approx-fallback light) keeps reading earthTexture
    // exactly as before - purely additive.
    uint32_t usePbrtTexture = 0;
};

// Builds a ProjectionLightData aimed from `position` at `target`, with a
// world-space `worldUp` hint (the usual "look-at" convention every other
// camera/light-aiming helper uses) to disambiguate the roll around the
// forward axis - mirrors metal_poc.metal's own ProjectionLight comment on
// why `right`/`up`/`forward` are precomputed once here rather than
// re-derived per shading sample. `fovDegrees` is the FULL vertical field
// of view (matching this project's own camera.h/cameras.h convention,
// not a half-angle); `aspect` (width/height of the projected image, NOT
// the render's own output aspect) sets the horizontal FOV independently,
// the same way projection_light.h's own screenBounds derivation folds an
// image's aspect ratio into an otherwise-square frustum.
static ProjectionLightData makeProjectionLight(float3 position, float3 target, float3 worldUp,
                                                float fovDegrees, float aspect, float scale) {
    float3 forward = simd::normalize(target - position);
    float3 right = simd::normalize(simd::cross(forward, worldUp));
    float3 up = simd::cross(right, forward);
    float tanHalfFovY = tanf(fovDegrees * 0.5f * (float)M_PI / 180.0f);
    float tanHalfFovX = tanHalfFovY * aspect;
    return ProjectionLightData{
        PackedFloat3{position.x, position.y, position.z},
        PackedFloat3{forward.x, forward.y, forward.z},
        PackedFloat3{right.x, right.y, right.z},
        PackedFloat3{up.x, up.y, up.z},
        tanHalfFovX,
        tanHalfFovY,
        scale};
}

// Mirrors metal_poc.metal's GoniometricLight byte-for-byte.
struct GoniometricLightData {
    PackedFloat3 position;
    PackedFloat3 forward;
    PackedFloat3 right;
    PackedFloat3 up;
    PackedFloat3 emission;
    float scale;
    // Same idea as ProjectionLightData::usePbrtTexture above - selects
    // pbrtGoniometricTexture instead of the room's own shared
    // goniometricTexture. False by default, purely additive.
    uint32_t usePbrtTexture = 0;
};

// Builds a GoniometricLightData the same look-at way makeProjectionLight()
// above builds a ProjectionLightData - see that function's own comment.
static GoniometricLightData makeGoniometricLight(float3 position, float3 target, float3 worldUp,
                                                  float3 emission, float scale) {
    float3 forward = simd::normalize(target - position);
    float3 right = simd::normalize(simd::cross(forward, worldUp));
    float3 up = simd::cross(right, forward);
    return GoniometricLightData{
        PackedFloat3{position.x, position.y, position.z},
        PackedFloat3{forward.x, forward.y, forward.z},
        PackedFloat3{right.x, right.y, right.z},
        PackedFloat3{up.x, up.y, up.z},
        PackedFloat3{emission.x, emission.y, emission.z},
        scale};
}

// pbrt-v4's own equal-area octahedral SQUARE-to-sphere mapping
// (src/shared/sampling_extra.h's EqualAreaSquareToSphere(), the INVERSE
// of metal_poc.metal's own equalAreaSphereToSquare() - ported here, host-
// side, purely to GENERATE the procedural goniometric image below: for
// each texel's own (u,v), this recovers exactly which direction that
// texel represents, so the image can be filled in by a function of
// DIRECTION (the natural way to author a goniometric profile) rather
// than needing to reason about the forward mapping's own distorted
// texel layout directly.
static float3 equalAreaSquareToSphere(float u, float v) {
    float uu = 2.0f * u - 1.0f, vv = 2.0f * v - 1.0f;
    float up_ = fabsf(uu), vp = fabsf(vv);
    float signedDist = 1.0f - (up_ + vp);
    float d = fabsf(signedDist);
    float r = 1.0f - d;
    float phi = (r == 0.0f ? 1.0f : (vp - up_) / r + 1.0f) * ((float)M_PI / 4.0f);
    float z = copysignf(1.0f - r * r, signedDist);
    float cosPhi = cosf(phi);
    float sinPhi = sinf(phi);
    float xyR = r * sqrtf(fmaxf(0.0f, 2.0f - r * r));
    float x = copysignf(cosPhi * xyR, uu);
    float y = copysignf(sinPhi * xyR, vv);
    return float3{x, y, z};
}

// Procedurally generates a small goniometric intensity image - see
// metal_poc.metal's own GoniometricLight comment for the full "why" (no
// real IES data file exists in this repo, and this POC's own established
// pattern for a missing real-world asset is an analytic stand-in, the
// same choice bump mapping's own egg-carton height field and the
// checkerboard/patterned-light materials already made). Single-channel
// (R8Unorm), `size`x`size`, equal-area-indexed (texel (u,v) represents
// the direction equalAreaSquareToSphere(u,v) gives, in the light's own
// local frame - +Z is the light's own `forward` aim direction).
//
// Two multiplied factors, each independently verifiable: a forward-
// facing LOBE (a smoothstep falloff in the direction's own Z component -
// full intensity dead-centre, fading to zero by `cosCutoff`, zero
// everywhere behind the light) and a RING modulation (`cos` of the polar
// angle scaled by `ringFrequency`, remapped to [0,1]) - the ring term is
// what makes this a genuine test of a real 2D-image-indexed light
// instead of a reskinned spot light: spotLightFalloff() can only ever
// produce a single monotonic centre-to-edge falloff, never concentric
// bright/dark BANDS within one cone, since it has no notion of a stored
// image at all.
static std::vector<uint8_t> buildGoniometricProfileImage(int size, float cosCutoff, float ringFrequency) {
    // Plain C++ has no built-in smoothstep (that's a GLSL/MSL intrinsic,
    // not a standard-library function) - the textbook 3t^2-2t^3 Hermite
    // form, same formula this project's own metal_poc.metal spells out
    // by hand for spotLightFalloff()'s own cone falloff.
    auto smoothstepHermite = [](float edge0, float edge1, float x) -> float {
        float t = fminf(fmaxf((x - edge0) / fmaxf(edge1 - edge0, 1e-6f), 0.0f), 1.0f);
        return t * t * (3.0f - 2.0f * t);
    };
    std::vector<uint8_t> image(size * size);
    for (int v = 0; v < size; ++v) {
        for (int u = 0; u < size; ++u) {
            float uu = (float(u) + 0.5f) / float(size);
            float vv = (float(v) + 0.5f) / float(size);
            float3 dir = equalAreaSquareToSphere(uu, vv);
            float cosTheta = fminf(fmaxf(dir.z, -1.0f), 1.0f);
            float lobe = smoothstepHermite(cosCutoff, 1.0f, cosTheta);
            float theta = acosf(cosTheta);
            float ring = 0.5f + 0.5f * cosf(theta * ringFrequency);
            float intensity = lobe * ring;
            image[v * size + u] = (uint8_t)(fminf(fmaxf(intensity, 0.0f), 1.0f) * 255.0f + 0.5f);
        }
    }
    return image;
}

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
    // Index into the AreaLight list this material's own triangles belong
    // to, or -1 for every non-emissive material - see metal_poc.metal's
    // own comment on the mirrored field.
    int32_t lightId = -1;
    // Rough-dielectric roughness (materialType == 5), ALSO reused as
    // anisotropic alphaY for materialType == 4 (0.0 there falls back to
    // isotropic), ALSO reused again as procedural bump strength for
    // materialType == 7 (0.0 there falls back to a perfectly flat
    // Lambertian, identical to materialType == 0) - see metal_poc.metal's
    // own comment on the mirrored field for the full explanation.
    float roughness = 0.0f;
    // Complex IOR (eta + i*k) per RGB channel, materialType == 4/9 only -
    // see metal_poc.metal's own mirrored comment.
    PackedFloat3 conductorEta{0, 0, 0};
    PackedFloat3 conductorK{0, 0, 0};
    // Diffuse TRANSMITTANCE tint, materialType == 12 only (`color` is
    // this material's own diffuse REFLECTANCE tint, same convention as
    // materialType 0) - see metal_poc.metal's own mirrored comment.
    PackedFloat3 transmitColor{0, 0, 0};
    // pbrt-v4's own "bool twosided" AreaLightSource parameter (section
    // 104) - see metal_poc.metal's own mirrored comment. 0 for every
    // non-emissive material.
    uint32_t twoSided = 0;
};

// Mirrors metal_poc.metal's SphereData byte-for-byte.
struct SphereData {
    PackedFloat3 center;
    float radius;
};

// Mirrors metal_poc.metal's InstanceTransform byte-for-byte (4 packed
// columns, same layout MTLPackedFloat4x3 itself uses - see that file's
// own comment on why this side-channel buffer exists at all).
struct InstanceTransform {
    PackedFloat3 col0;
    PackedFloat3 col1;
    PackedFloat3 col2;
    PackedFloat3 col3;
};

// Mirrors metal_poc.metal's DiskData byte-for-byte.
struct DiskData {
    PackedFloat3 center;
    PackedFloat3 normal;
    float radius;
};

// A quad (4 verts, wound as 2 triangles) sharing one flat colour and
// material type - the smallest scene-authoring shape that can build a
// real Cornell box without hand-listing 30 individual vertices. `normals`
// is parallel to `verts` (same per-triangle-corner indexing
// metal_poc.metal's shadingNormalFor() reads) - every corner of a flat
// quad gets the SAME computed face normal, which is what makes the
// shader's barycentric interpolation reduce to exact flat shading here
// (only a mesh with genuinely different per-corner normals, i.e.
// loadObjMesh() below, produces a different, smoothly-varying result).
// `uvs` is the same shape again but for texCoordFor() - a standard
// planar (0,0)-(1,1) mapping across a-b-c-d, meaningful only when
// materialType == 3 (textured); every other quad's UVs are simply never
// read by the shader.
static void addQuad(std::vector<PackedFloat3>& verts,
                     std::vector<PackedFloat3>& normals,
                     std::vector<PackedFloat2>& uvs,
                     std::vector<TriangleMaterial>& materials,
                     float3 a, float3 b, float3 c, float3 d,
                     float3 color, uint32_t materialType = 0,
                     float3 emission = simd::make_float3(0, 0, 0),
                     int32_t lightId = -1, float roughness = 0.0f,
                     // Defaults to 1.0 (every quad before materialType 11's
                     // own thin-dielectric PR) - matches every earlier
                     // call site's own hardcoded `ior` literal exactly, so
                     // adding this parameter changes nothing for any of
                     // them; only a quad that actually needs a REAL
                     // refraction index (materialType 11's own glass-pane
                     // object) passes something else.
                     float ior = 1.0f,
                     // Diffuse TRANSMITTANCE tint, materialType == 12
                     // only - defaults to black (every quad before that
                     // material existed had no transmission at all).
                     float3 transmitColor = simd::make_float3(0, 0, 0),
                     // pbrt-v4's own "bool twosided" AreaLightSource
                     // parameter (section 104) - defaults to false, every
                     // quad before this one stays one-sided exactly as
                     // before. Meaningless (ignored) for a non-emissive
                     // quad, same as every other emission-only field
                     // above.
                     bool twoSided = false) {
    // a-b-c-d wound so (a,b,c) and (a,c,d) both face outward consistently.
    auto push = [&](float3 v) { verts.push_back(PackedFloat3{v.x, v.y, v.z}); };
    push(a); push(b); push(c);
    push(a); push(c); push(d);
    float3 faceNormal = simd::normalize(simd::cross(b - a, c - a));
    PackedFloat3 packedNormal{faceNormal.x, faceNormal.y, faceNormal.z};
    for (int i = 0; i < 6; ++i) normals.push_back(packedNormal);
    uvs.push_back(PackedFloat2{0, 0});
    uvs.push_back(PackedFloat2{1, 0});
    uvs.push_back(PackedFloat2{1, 1});
    uvs.push_back(PackedFloat2{0, 0});
    uvs.push_back(PackedFloat2{1, 1});
    uvs.push_back(PackedFloat2{0, 1});
    PackedFloat3 packedColor{color.x, color.y, color.z};
    PackedFloat3 packedEmission{emission.x, emission.y, emission.z};
    TriangleMaterial mat{packedColor, materialType, ior, packedEmission, lightId, roughness};
    mat.transmitColor = PackedFloat3{transmitColor.x, transmitColor.y, transmitColor.z};
    mat.twoSided = twoSided ? 1u : 0u;
    materials.push_back(mat);
    materials.push_back(mat);
}

// A minimal Wavefront OBJ loader: positions, vertex normals, texture
// coordinates, and faces (`v`/`vn`/`vt`/`f`) - no materials/groups. This
// project's own real loaders (src/shared/pbrt_load.h ->
// pbrt_cpu_builder.h/pbrt_gpu_builder.h) are full pbrt-v4 scene parsers;
// this is deliberately the smallest thing that can prove "load an
// arbitrary real mesh, not just hand-authored axis-aligned quads and an
// analytic sphere" - the first genuinely data-driven geometry in this
// POC. Faces are fan-triangulated (n>3 polygon -> n-2 triangles sharing
// vertex 0), matching how this project's own CPU loader handles polygons
// that aren't already triangles.
//
// Per-corner shading normal: a face token's own `vn` index is used when
// present (`v//vn` or `v/vt/vn`); a face missing normal indices entirely
// falls back to that triangle's own computed flat face normal - real
// files are inconsistent about this in practice (suzanne.obj has `vn` on
// every face; spot.obj, added for real UV testing below, has NONE at
// all, so both code paths get real exercise across this POC's two mesh
// files, not just the fallback-free one). This is what makes
// metal_poc.metal's barycentric shadingNormalFor() interpolation produce
// a genuinely smooth result instead of the flat-per-triangle look every
// other object in this scene has.
//
// Per-corner texture coordinate: same idea, a face token's own `vt`
// index (`v/vt` or `v/vt/vn`) when present, (0,0) fallback otherwise -
// this is the real-data counterpart to addQuad()'s hand-authored planar
// UVs, the piece the texture-mapping PR explicitly deferred ("parsing
// real vt/f v/vt/vn tokens was judged out of scope for this increment").
// suzanne.obj has no `vt` data at all (every corner falls back), so it
// keeps its materialType 0; spot.obj DOES (3225 `vt` entries, one per
// face corner), the caller passes materialType 3 for it, and
// texCoordFor()'s barycentric interpolation on real per-corner data
// produces a real (if mismatched-content, earthmap.jpg was never meant
// for a cow) textured mesh, not just a textured flat quad.
//
// The mesh is auto-fit to `targetSize` (its largest bounding-box
// dimension scaled to that value) and recentred at `center` - real .obj
// files come in whatever units/scale their author used, and this scene's
// room is a fixed [-1,1] box, so SOME normalization is unavoidable rather
// than a hardcoded scale constant that would only happen to work for this
// one file.
// The directory containing the CURRENTLY RUNNING executable, or nil if
// unavailable (_NSGetExecutablePath, not NSBundle - resolves correctly
// for a plain non-app-bundle CLI binary too). Checked FIRST, before any
// RT_..._DIR compile-time fallback (RT_MODELS_DIR/RT_METAL_SHADER_DIR),
// at every asset-lookup site that has one - those are absolute paths
// into the machine that BUILT this binary, meaningless once it's been
// copied/installed anywhere else (section 110's own real bug: a
// distributed .dmg's bundled `ray_tracer`, run on a genuinely different
// machine, would otherwise fail to find its own shader source AND
// models/suzanne.obj, models/spot.obj, images/earthmap.jpg - found by
// actually testing a packaged build from a clean, relocated install,
// not assumed correct from the code alone).
static NSString* executableDir() {
    char exePathBuf[4096];
    uint32_t exePathSize = sizeof(exePathBuf);
    if (_NSGetExecutablePath(exePathBuf, &exePathSize) != 0) return nil;
    return [@(exePathBuf) stringByDeletingLastPathComponent];
}

static bool loadObjMesh(const std::string& path,
                         std::vector<PackedFloat3>& verts,
                         std::vector<PackedFloat3>& normals,
                         std::vector<PackedFloat2>& uvs,
                         std::vector<TriangleMaterial>& materials,
                         float3 color, float3 center, float targetSize,
                         uint32_t materialType = 0,
                         // materialType==4 only - see this function's own
                         // TriangleMaterial-construction comment below for
                         // why these three exist. meshConductorEta defaults
                         // to (1,1,1) (not 0) since eta==0 is unphysical and
                         // every REAL conductor preset this codebase already
                         // uses (conductor_data.h) keeps eta near 1 anyway -
                         // a caller that forgets to override k too still
                         // gets a real (if flat/grey) metal, not black.
                         float meshRoughness = 0.0f,
                         float3 meshConductorEta = simd::make_float3(1.0f, 1.0f, 1.0f),
                         float3 meshConductorK = simd::make_float3(0.0f, 0.0f, 0.0f),
                         // materialType==2 (dielectric) only - see this
                         // function's own TriangleMaterial-construction
                         // comment below. Default 1.0 (no refraction) keeps
                         // every OTHER materialType's behaviour identical to
                         // before this parameter existed.
                         float meshIor = 1.0f,
                         // Mirrors gpu/optix/scene_builder.cpp's own
                         // load_obj_triangles_gpu()'s flip_xz parameter
                         // exactly: negates x and z (a 180-degree rotation
                         // about Y) - some raw meshes (Spot the Cow, Horse)
                         // face away from this app's own camera convention
                         // without it (that file's own comment: "the raw
                         // mesh faces away from the camera"). Applied to
                         // every position AND normal, before centring/
                         // scaling - section 119, docs/METAL_GPU_
                         // FEASIBILITY.md.
                         bool flipXZ = false) {
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "Could not open OBJ file: %s\n", path.c_str());
        return false;
    }

    std::vector<float3> positions;
    std::vector<float3> fileNormals;
    std::vector<simd::float2> fileUVs;
    // Each face vertex is (positionIndex, normalIndex-or--1,
    // uvIndex-or--1), 0-based post-fixup - keeping the triple together
    // (rather than three parallel index lists) is what lets a face's own
    // vn/vt references survive fan triangulation below unchanged.
    struct FaceVertex { int posIdx; int normalIdx; int uvIdx; };
    std::vector<std::vector<FaceVertex>> faces;

    auto parseObjIndex = [](const std::string& token, size_t countAtParseTime) -> int {
        int idx = std::atoi(token.c_str());
        if (idx == 0) return -1; // absent (e.g. the "vt" slot in "v/vt/vn")
        // OBJ indices are 1-based; a negative index is relative to the
        // current count (rare, but real files use it).
        if (idx < 0) idx = (int)countAtParseTime + idx + 1;
        return idx - 1;
    };

    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            positions.push_back(simd::make_float3(x, y, z));
        } else if (tag == "vn") {
            float x, y, z;
            ss >> x >> y >> z;
            fileNormals.push_back(simd::make_float3(x, y, z));
        } else if (tag == "vt") {
            float u, v;
            ss >> u >> v;
            fileUVs.push_back(simd::float2{u, v});
        } else if (tag == "f") {
            std::vector<FaceVertex> faceVerts;
            std::string token;
            while (ss >> token) {
                // Token is "v", "v/vt", "v//vn", or "v/vt/vn".
                size_t firstSlash = token.find('/');
                size_t lastSlash = token.rfind('/');
                int posIdx = parseObjIndex(token.substr(0, firstSlash), positions.size());
                int normalIdx = -1;
                int uvIdx = -1;
                if (firstSlash != std::string::npos && lastSlash != firstSlash) {
                    normalIdx = parseObjIndex(token.substr(lastSlash + 1), fileNormals.size());
                }
                if (firstSlash != std::string::npos) {
                    // The vt slot sits between the two slashes for
                    // "v/vt/vn", or from the first slash to the token's
                    // end for "v/vt" (no vn at all, spot.obj's own
                    // format) - lastSlash == firstSlash in that case, so
                    // this substring naturally runs to end-of-string.
                    size_t vtEnd = (lastSlash != firstSlash) ? lastSlash : token.size();
                    std::string vtToken = token.substr(firstSlash + 1, vtEnd - firstSlash - 1);
                    uvIdx = parseObjIndex(vtToken, fileUVs.size());
                }
                faceVerts.push_back({posIdx, normalIdx, uvIdx});
            }
            if (faceVerts.size() >= 3) faces.push_back(faceVerts);
        }
    }

    if (positions.empty() || faces.empty()) {
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

    // flipXZ applied to the CENTRED delta, not the raw position - the
    // bounding box above is computed from raw positions either way (a
    // pure x/z negation is a reflection, which preserves the extent used
    // for `scale`, so no separate flipped-bbox pass is needed).
    auto transform = [&](const float3& p) -> float3 {
        float3 delta = p - bboxCenter;
        if (flipXZ) { delta.x = -delta.x; delta.z = -delta.z; }
        return delta * scale + center;
    };
    // Normals only need the scale's sign/shear behaviour, not translation -
    // a uniform positive scale (this loader's only kind) leaves direction
    // unchanged, so this is really just "no-op, pass through," kept as its
    // own step for clarity and in case a future non-uniform scale needs it.
    // flipXZ needs the SAME x/z negation as transform() above (a normal
    // rotates with its surface).
    auto transformNormal = [&](const float3& n) -> float3 {
        float3 nn = n;
        if (flipXZ) { nn.x = -nn.x; nn.z = -nn.z; }
        return simd::normalize(nn);
    };

    uint32_t triangleCount = 0;
    uint32_t normalFallbackCount = 0;
    uint32_t uvFallbackCount = 0;
    for (const std::vector<FaceVertex>& face : faces) {
        // Fan triangulation from vertex 0 - correct for the convex/near-
        // convex polygons a typical modeled mesh's faces are (this file's
        // own quads included), not a general concave-polygon triangulator.
        for (size_t i = 1; i + 1 < face.size(); ++i) {
            FaceVertex fv0 = face[0], fv1 = face[i], fv2 = face[i + 1];
            if (fv0.posIdx < 0 || fv0.posIdx >= (int)positions.size() ||
                fv1.posIdx < 0 || fv1.posIdx >= (int)positions.size() ||
                fv2.posIdx < 0 || fv2.posIdx >= (int)positions.size()) {
                continue; // malformed index - skip rather than crash
            }
            float3 a = transform(positions[fv0.posIdx]);
            float3 b = transform(positions[fv1.posIdx]);
            float3 c = transform(positions[fv2.posIdx]);
            verts.push_back(PackedFloat3{a.x, a.y, a.z});
            verts.push_back(PackedFloat3{b.x, b.y, b.z});
            verts.push_back(PackedFloat3{c.x, c.y, c.z});

            bool haveAllUVs =
                fv0.uvIdx >= 0 && fv0.uvIdx < (int)fileUVs.size() &&
                fv1.uvIdx >= 0 && fv1.uvIdx < (int)fileUVs.size() &&
                fv2.uvIdx >= 0 && fv2.uvIdx < (int)fileUVs.size();
            if (haveAllUVs) {
                uvs.push_back(PackedFloat2{fileUVs[fv0.uvIdx].x, fileUVs[fv0.uvIdx].y});
                uvs.push_back(PackedFloat2{fileUVs[fv1.uvIdx].x, fileUVs[fv1.uvIdx].y});
                uvs.push_back(PackedFloat2{fileUVs[fv2.uvIdx].x, fileUVs[fv2.uvIdx].y});
            } else {
                uvs.push_back(PackedFloat2{0, 0});
                uvs.push_back(PackedFloat2{0, 0});
                uvs.push_back(PackedFloat2{0, 0});
                ++uvFallbackCount;
            }

            bool haveAllNormals =
                fv0.normalIdx >= 0 && fv0.normalIdx < (int)fileNormals.size() &&
                fv1.normalIdx >= 0 && fv1.normalIdx < (int)fileNormals.size() &&
                fv2.normalIdx >= 0 && fv2.normalIdx < (int)fileNormals.size();
            if (haveAllNormals) {
                float3 n0 = transformNormal(fileNormals[fv0.normalIdx]);
                float3 n1 = transformNormal(fileNormals[fv1.normalIdx]);
                float3 n2 = transformNormal(fileNormals[fv2.normalIdx]);
                normals.push_back(PackedFloat3{n0.x, n0.y, n0.z});
                normals.push_back(PackedFloat3{n1.x, n1.y, n1.z});
                normals.push_back(PackedFloat3{n2.x, n2.y, n2.z});
            } else {
                float3 flat = simd::normalize(simd::cross(b - a, c - a));
                PackedFloat3 packedFlat{flat.x, flat.y, flat.z};
                normals.push_back(packedFlat);
                normals.push_back(packedFlat);
                normals.push_back(packedFlat);
                ++normalFallbackCount;
            }
            ++triangleCount;
        }
    }

    PackedFloat3 packedColor{color.x, color.y, color.z};
    TriangleMaterial mat{packedColor, materialType, 1.0f, PackedFloat3{0, 0, 0}};
    // materialType==4 (real complex-Fresnel GGX conductor) needs
    // conductorEta/conductorK too, which this function's own signature had
    // no way to pass until section 117's own G-category (Models) increment
    // needed a metal-finish mesh for the first time (Suzanne/Spot, this
    // function's only callers before that, are both materialType 0/3). Left
    // at their struct default (eta/k = {0,0,0}, roughness = 0) for every
    // OTHER materialType - identical to this function's own behaviour
    // before these parameters existed.
    //
    // BOTH mat.ior (alphaX) and mat.roughness (alphaY) must be set to the
    // SAME value for isotropic roughness - mapMaterial()'s own Conductor
    // case (loadPbrtScene()) does this identically. Leaving `mat.ior` at
    // the `1.0f` this constructor already gives every material (a
    // DIELECTRIC default, meaningless for a conductor) while only setting
    // `mat.roughness` would silently make alphaX=1.0 (maximally rough) and
    // alphaY=meshRoughness - a real, easy-to-miss anisotropy bug caught
    // here before it ever rendered, not after.
    if (materialType == 4u) {
        mat.ior = meshRoughness;
        mat.roughness = meshRoughness;
        mat.conductorEta = PackedFloat3{meshConductorEta.x, meshConductorEta.y, meshConductorEta.z};
        mat.conductorK = PackedFloat3{meshConductorK.x, meshConductorK.y, meshConductorK.z};
    } else if (materialType == 2u) {
        // Smooth dielectric (materialType 2, e.g. Glass Dragon - section
        // 119) - `ior` here is a real refraction index (glass~1.5), not
        // the alphaX reuse materialType 4 gives it above.
        mat.ior = meshIor;
    }
    for (uint32_t i = 0; i < triangleCount; ++i) materials.push_back(mat);

    fprintf(stderr, "Loaded %s: %zu positions, %zu normals, %zu uvs, %u triangles "
                     "(%u flat-normal fallback, %u zero-uv fallback), scale %.4f\n",
            path.c_str(), positions.size(), fileNormals.size(), fileUVs.size(), triangleCount,
            normalFallbackCount, uvFallbackCount, scale);
    return true;
}

// blackbodyColor/vignetteFactor/sampleChannelBilinear/chromaticAberration/
// acesFilmicTonemap/reinhardTonemap/ToneMapMode/parseToneMapMode/
// applyToneMap/linearToSRGB now live in metal_poc_host_math.h (included
// above) - see that header's own comment.

// Edge-preserving bilateral denoise, applied to the final 8-bit LDR
// image (after tonemapping/gamma, not the linear HDR buffer - the
// standard display-referred way to do this: a range kernel compared
// directly against raw HDR values would be dominated by the huge
// magnitude gap between a light source and everything else, rather than
// meaningfully distinguishing "real edge" from "Monte Carlo noise").
// A follow-on to the firefly clamp: that PR found (and honestly
// reported) this scene's own worst noise - high-variance fog/volumetric
// sampling near the spot light's own cone - wasn't the rare-extreme-
// outlier kind firefly clamping targets, so it barely helped there.
// Spatial denoising targets exactly that kind of noise instead: every
// neighbouring pixel contributes to the output, weighted by BOTH how
// close it is (`sigmaSpatial`, a Gaussian in pixel distance) and how
// similar its own LUMINANCE is to the centre pixel's (`sigmaRange`, a
// Gaussian in luminance difference) - two nearby pixels with similar
// brightness (likely the same underlying surface, differing only by
// noise) get smoothed together; two nearby pixels with very different
// brightness (likely a real edge - a shadow boundary, a specular
// highlight, a checker tile seam) barely influence each other at all,
// which is what keeps this from just being a uniform blur. The SAME
// per-pixel weight (derived from luminance alone) is applied to all
// three colour channels together, not computed separately per channel -
// preserves each pixel's own hue relationship to its neighbours instead
// of letting R/G/B drift independently.
static void bilateralDenoise(const std::vector<uint8_t>& ldrIn, std::vector<uint8_t>& ldrOut,
                              uint32_t width, uint32_t height, int radius,
                              float sigmaSpatial, float sigmaRange) {
    std::vector<float> luminance(width * height);
    for (uint32_t i = 0; i < width * height; ++i) {
        luminance[i] = 0.2126f * ldrIn[i * 3 + 0] + 0.7152f * ldrIn[i * 3 + 1] + 0.0722f * ldrIn[i * 3 + 2];
    }
    float invSpatial2 = 1.0f / (2.0f * sigmaSpatial * sigmaSpatial);
    float invRange2 = 1.0f / (2.0f * sigmaRange * sigmaRange);
    for (int32_t y = 0; y < (int32_t)height; ++y) {
        for (int32_t x = 0; x < (int32_t)width; ++x) {
            uint32_t centerIdx = (uint32_t)y * width + (uint32_t)x;
            float centerLum = luminance[centerIdx];
            float sumWeight = 0.0f;
            float sumRGB[3] = {0.0f, 0.0f, 0.0f};
            for (int32_t dy = -radius; dy <= radius; ++dy) {
                int32_t ny = y + dy;
                if (ny < 0 || ny >= (int32_t)height) continue;
                for (int32_t dx = -radius; dx <= radius; ++dx) {
                    int32_t nx = x + dx;
                    if (nx < 0 || nx >= (int32_t)width) continue;
                    uint32_t nIdx = (uint32_t)ny * width + (uint32_t)nx;
                    float spatialTerm = float(dx * dx + dy * dy) * invSpatial2;
                    float lumDiff = luminance[nIdx] - centerLum;
                    float rangeTerm = lumDiff * lumDiff * invRange2;
                    float weight = expf(-(spatialTerm + rangeTerm));
                    sumWeight += weight;
                    sumRGB[0] += weight * float(ldrIn[nIdx * 3 + 0]);
                    sumRGB[1] += weight * float(ldrIn[nIdx * 3 + 1]);
                    sumRGB[2] += weight * float(ldrIn[nIdx * 3 + 2]);
                }
            }
            for (int c = 0; c < 3; ++c) {
                float v = sumRGB[c] / fmaxf(sumWeight, 1e-6f);
                ldrOut[centerIdx * 3 + c] = (uint8_t)fminf(fmaxf(v + 0.5f, 0.0f), 255.0f);
            }
        }
    }
}

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

    // --- GPU-resident buffers + acceleration structures, built by
    // buildGPUResources() ------------------------------------------------
    id<MTLBuffer> vertexBuffer, normalBuffer, uvBuffer, lightBuffer;
    id<MTLBuffer> pointLightBuffer, directionalLightBuffer, projectionLightBuffer;
    id<MTLBuffer> goniometricLightBuffer;
    id<MTLBuffer> materialBuffer, sphereBuffer, sphereMaterialBuffer;
    id<MTLBuffer> diskBuffer, diskMaterialBuffer;
    id<MTLBuffer> suzanneVertexBuffer, suzanneNormalBuffer, suzanneMaterialBuffer;
    id<MTLBuffer> instanceTransformBuffer;
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
    // Recomputes pbrtCameraPos/Forward/Right/Up for a new lookfrom in the
    // loaded scene's own pbrt-file coordinate space, keeping lookat/up/fov
    // exactly as loadPbrtScene() read them from the scene - see this
    // method's own definition (just after loadPbrtScene()) and
    // metal_render_main()'s own comment for why only lookfrom moves. Only
    // valid to call after a successful loadPbrtScene() (havePbrtCamera).
    void applyCameraOverride(double cam_x, double cam_y, double cam_z);
    bool buildGPUResources();
    bool compileShaderAndDispatch(int argc, const char** argv);
    void postProcessAndWrite();
};

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
        [&warnedUnsupportedMaterialKinds, &scene, &mapMaterial](const pbrt_flatten::Material& m, int depth) -> TriangleMaterial {
        PackedFloat3 color{(float)m.color[0], (float)m.color[1], (float)m.color[2]};
        switch (m.kind) {
            case pbrt_flatten::MaterialKind::Diffuse:
                return TriangleMaterial{color, /*materialType=*/0u, /*ior=*/1.0f,
                                         PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
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
        sphereMaterials.push_back(materialFor(s.material));
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
void MetalPocApp::buildCornellBoxA1() {
    using namespace cornell_box_data;
    // Same rescale/recentre/offset convention loadPbrtScene() uses for
    // every real pbrt scene (see that function's own comment) - this data
    // is ALREADY authored at the identical ~555-unit Cornell-box scale a
    // real pbrt Cornell box uses, so the exact same fixed transform
    // applies unchanged: largest dimension -> 2.0 units, recentred on the
    // room's own bounding-box centre, offset +8 in X clear of the
    // hardcoded POC room's own [-1,1] region.
    const float3 bboxMin{0.0f, 0.0f, 0.0f};
    const float3 bboxMax{555.0f, 555.0f, 555.0f};
    const float maxExtent = 555.0f;
    const float sceneScale = 2.0f / maxExtent;
    const float3 bboxCenter = 0.5f * (bboxMin + bboxMax);
    const float3 sceneOffset{8.0f, 0.0f, 0.0f};
    auto toWorld = [=](float3 p) { return (p - bboxCenter) * sceneScale + sceneOffset; };

    // The 5 walls + the main ceiling light.
    for (const QuadSpec& q : kQuads) {
        const float3 Q{(float)q.Q.x, (float)q.Q.y, (float)q.Q.z};
        const float3 u{(float)q.u.x, (float)q.u.y, (float)q.u.z};
        const float3 v{(float)q.v.x, (float)q.v.y, (float)q.v.z};
        const float3 a = toWorld(Q);
        const float3 b = toWorld(Q + u);
        const float3 c = toWorld(Q + u + v);
        const float3 d = toWorld(Q + v);
        const float3 color{(float)q.color.r, (float)q.color.g, (float)q.color.b};
        if (q.is_light) {
            // Same AreaLightData construction loadPbrtAreaLights() already
            // uses for a quad light - a flat, one-sided (pbrt's own
            // default) emitter, no pattern/texture.
            const int32_t lightId = (int32_t)lights.size();
            addQuad(verts, normals, uvs, materials, a, b, c, d, color,
                    /*materialType=*/0u, /*emission=*/color, lightId);
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
                PackedFloat3{color.x, color.y, color.z},
                /*patternTileB=*/0.0f, /*patternScale=*/0.0f,
                /*twoSided=*/0.0f, /*useTexture=*/0.0f});
        } else {
            addQuad(verts, normals, uvs, materials, a, b, c, d, color);
        }
    }

    // The rotated white box - the same 6-quad (Q,u,v per face) construction
    // src/TheRestOfYourLife/quad.h's own box() helper uses (front/right/
    // back/left/top/bottom), rotated about Y then translated in LOCAL
    // space before toWorld() - matches src/TheRestOfYourLife/hittable.h's
    // own rotate_y forward-transform formula exactly (newx=cos*x+sin*z,
    // newz=-sin*x+cos*z), not re-derived independently.
    {
        const float3 minC{(float)kBox.corner_min.x, (float)kBox.corner_min.y, (float)kBox.corner_min.z};
        const float3 maxC{(float)kBox.corner_max.x, (float)kBox.corner_max.y, (float)kBox.corner_max.z};
        const float3 dx{maxC.x - minC.x, 0.0f, 0.0f};
        const float3 dy{0.0f, maxC.y - minC.y, 0.0f};
        const float3 dz{0.0f, 0.0f, maxC.z - minC.z};
        const float theta = (float)(kBox.rotate_y_degrees * M_PI / 180.0);
        const float sinT = sinf(theta), cosT = cosf(theta);
        const float3 boxTranslate{(float)kBox.translate.x, (float)kBox.translate.y, (float)kBox.translate.z};
        auto rotateTranslate = [=](float3 p) -> float3 {
            const float newX = cosT * p.x + sinT * p.z;
            const float newZ = -sinT * p.x + cosT * p.z;
            return float3{newX, p.y, newZ} + boxTranslate;
        };
        const float3 boxColor{(float)kBox.color.r, (float)kBox.color.g, (float)kBox.color.b};
        struct Face { float3 Q, u, v; };
        const Face faces[6] = {
            {float3{minC.x, minC.y, maxC.z},  dx,  dy},  // front
            {float3{maxC.x, minC.y, maxC.z}, -dz,  dy},  // right
            {float3{maxC.x, minC.y, minC.z}, -dx,  dy},  // back
            {float3{minC.x, minC.y, minC.z},  dz,  dy},  // left
            {float3{minC.x, maxC.y, maxC.z},  dx, -dz},  // top
            {float3{minC.x, minC.y, minC.z},  dx,  dz},  // bottom
        };
        for (const Face& f : faces) {
            const float3 a = toWorld(rotateTranslate(f.Q));
            const float3 b = toWorld(rotateTranslate(f.Q + f.u));
            const float3 c = toWorld(rotateTranslate(f.Q + f.u + f.v));
            const float3 d = toWorld(rotateTranslate(f.Q + f.v));
            addQuad(verts, normals, uvs, materials, a, b, c, d, boxColor);
        }
    }

    // Glass sphere (materialType 2, dielectric).
    {
        const float3 center = toWorld(float3{(float)kGlassSphere.center.x,
            (float)kGlassSphere.center.y, (float)kGlassSphere.center.z});
        spheres.push_back(SphereData{PackedFloat3{center.x, center.y, center.z},
            sceneScale * (float)kGlassSphere.radius});
        sphereMaterials.push_back(TriangleMaterial{PackedFloat3{1, 1, 1}, /*materialType=*/2u,
            /*ior=*/(float)kGlassSphere.glass_ior, PackedFloat3{0, 0, 0}, /*lightId=*/-1,
            /*roughness=*/0.0f});
    }

    // Camera - kCornellBoxCamera's own literal values (scene_registry.h):
    // vfov=40, lookfrom=(278,278,-800), lookat=(278,278,278), world-up
    // (0,1,0) - the SAME numbers cpu_scene_recommended_camera()/CLI
    // --cam-x/y/z already send through applyCameraOverride()
    // unconditionally (launcher/main.cpp's own force_camera_override=1),
    // so this is really just the fallback/initial value, immediately
    // replaced by that override in every real invocation - matching
    // loadPbrtCamera()'s own comment on why the exact literal here barely
    // matters in practice.
    const float3 lookfrom = toWorld(float3{278.0f, 278.0f, -800.0f});
    const float3 lookat = toWorld(float3{278.0f, 278.0f, 278.0f});
    const float3 up{0.0f, 1.0f, 0.0f};
    const float3 forward = simd::normalize(lookat - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    pbrtTanHalfFov = tanf(0.5f * 40.0f * (float)M_PI / 180.0f);
    havePbrtCamera = true;
    pbrtCameraLookAtWorld = lookat;
    pbrtCameraUpRaw = up;
    pbrtBboxCenter = bboxCenter;
    pbrtSceneScale = sceneScale;
    pbrtSceneOffset = sceneOffset;

    fprintf(stderr, "buildHandAuthoredScene: built 'A1' (classic Cornell box, hand-authored, "
                    "no pbrt file - %d quads, 1 sphere, 1 light)\n", kNumQuads - 1 + 6);
}

// See buildMeshGalleryScene()'s own declaration comment (this struct's own
// definition) for the shape/simplifications this shares across every
// category-G (Models) scene - section 117, docs/METAL_GPU_FEASIBILITY.md.
// meshConductorEta/meshConductorK: pass {1,1,1}/a per-channel k computed
// from OptiX's own flat "albedo" via the SAME reflectance-to-k formula
// PR #103's own CoatedConductor "nothing given" fallback already
// established (k = 2*sqrt(r)/sqrt(max(1e-4,1-r)), eta=1) - not a NEW
// approximation invented here, reusing an already-shipped precedent for
// exactly this "a flat colour, not a real measured conductor spectrum"
// situation. meshMaterialType == 0 skips all of that (a plain diffuse
// mesh needs none of it) - meshConductorEta/K are simply ignored then.
void MetalPocApp::buildMeshGalleryScene(const std::string& objFilename, float3 meshColor,
        uint32_t meshMaterialType, float meshRoughness, float3 meshConductorEta,
        float3 meshConductorK, float meshTargetSize, float meshIor, bool flipXZ) {
    // Same "push well clear of the hardcoded POC room's own [-1,1] region"
    // convention loadPbrtScene()/buildCornellBoxA1() both already use (see
    // either one's own comment) - a real, previously-shipped bug found
    // here first (before it ever reached review): this scene's own
    // geometry is authored directly at the app's own working scale (no
    // rescale needed, unlike A1's 555-unit Cornell box), but was left
    // OVERLAPPING the hardcoded room's own already-occupied space, not
    // offset clear of it - the first render looked like garbled noise, not
    // a recognizable mesh, because it genuinely WAS two unrelated scenes'
    // geometry interleaved in the same few world-space units.
    const float3 sceneOffset{8.0f, 0.0f, 0.0f};

    // Ground: a large flat quad (not OptiX's own huge checker SPHERE - see
    // buildMeshGalleryScene()'s own declaration comment for why), light
    // grey diffuse, centred under the mesh.
    const float3 groundColor{0.5f, 0.5f, 0.5f};
    addQuad(verts, normals, uvs, materials,
            float3{-2.0f, 0.0f, -2.0f} + sceneOffset, float3{2.0f, 0.0f, -2.0f} + sceneOffset,
            float3{2.0f, 0.0f, 2.0f} + sceneOffset, float3{-2.0f, 0.0f, 2.0f} + sceneOffset, groundColor);

    // The mesh itself, auto-fit to meshTargetSize and recentred at
    // sceneOffset (loadObjMesh()'s own `center` parameter IS the mesh's
    // real world-space placement, not a separate local-then-place step -
    // see its own Suzanne/Spot call sites) - loadObjMesh()'s own existing
    // Suzanne/Spot convention otherwise (section 110's executableDir()-
    // first path resolution applies automatically, no new lookup code
    // needed here).
    NSString* modelsDir = nil;
    {
        NSString* exeDir = executableDir();
        NSString* candidate = [exeDir stringByAppendingPathComponent:@"models"];
        if (exeDir && [[NSFileManager defaultManager] fileExistsAtPath:
                [candidate stringByAppendingPathComponent:@(objFilename.c_str())]]) {
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
    NSString* meshPath = [modelsDir stringByAppendingPathComponent:@(objFilename.c_str())];
    if (!loadObjMesh(meshPath.UTF8String, verts, normals, uvs, materials, meshColor,
                      /*center=*/float3{0.0f, 0.45f, 0.0f} + sceneOffset, meshTargetSize, meshMaterialType,
                      meshRoughness, meshConductorEta, meshConductorK, meshIor, flipXZ)) {
        fprintf(stderr, "buildMeshGalleryScene: continuing without '%s' - check RT_MODELS_DIR / "
                        "models/%s.\n", objFilename.c_str(), objFilename.c_str());
    }

    // A small quad area light above the mesh (this loader's only NEE-
    // sampled light shape - see buildMeshGalleryScene()'s own declaration
    // comment for why not a sphere light like OptiX's own).
    {
        const float3 a = float3{-0.4f, 1.6f, -0.4f} + sceneOffset, b = float3{0.4f, 1.6f, -0.4f} + sceneOffset,
                     c = float3{0.4f, 1.6f, 0.4f} + sceneOffset, d = float3{-0.4f, 1.6f, 0.4f} + sceneOffset;
        const float3 lightColor{6.0f, 6.0f, 6.0f};
        const int32_t lightId = (int32_t)lights.size();
        addQuad(verts, normals, uvs, materials, a, b, c, d, lightColor,
                /*materialType=*/0u, /*emission=*/lightColor, lightId);
        const float3 edgeU = b - a, edgeV = d - a;
        const float3 normalV = simd::normalize(simd::cross(edgeU, edgeV));
        const float area = simd::length(simd::cross(edgeU, edgeV));
        const float3 center = a + 0.5f * edgeU + 0.5f * edgeV;
        lights.push_back(AreaLightData{
            PackedFloat3{center.x, center.y, center.z},
            PackedFloat3{edgeU.x, edgeU.y, edgeU.z},
            PackedFloat3{edgeV.x, edgeV.y, edgeV.z},
            PackedFloat3{normalV.x, normalV.y, normalV.z},
            area, PackedFloat3{lightColor.x, lightColor.y, lightColor.z},
            /*patternTileB=*/0.0f, /*patternScale=*/0.0f,
            /*twoSided=*/1.0f, /*useTexture=*/0.0f});
    }

    // Camera: a simple 3/4 elevated view of the mesh, close to OptiX's own
    // apply_mesh_camera() framing in spirit (not the exact literal, which
    // barely matters - see buildCornellBoxA1()'s own comment on why the
    // fallback camera here is immediately overridden in every real
    // invocation anyway). Same sceneOffset as every other element above -
    // lookfrom/lookat both need it too, or the camera would end up
    // pointed at the hardcoded room's own empty space instead of this
    // scene's own geometry.
    const float3 lookfrom = float3{0.0f, 0.9f, 2.2f} + sceneOffset;
    const float3 lookat = float3{0.0f, 0.45f, 0.0f} + sceneOffset;
    const float3 up{0.0f, 1.0f, 0.0f};
    const float3 forward = simd::normalize(lookat - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    pbrtTanHalfFov = tanf(0.5f * 35.0f * (float)M_PI / 180.0f);
    havePbrtCamera = true;
    pbrtCameraLookAtWorld = lookat;
    pbrtCameraUpRaw = up;
    // No rescale (unlike buildCornellBoxA1()'s own 555-unit Cornell box -
    // this scene is already authored directly at the app's own working
    // scale) - applyCameraOverride()'s own transform (bboxCenter=0,
    // sceneScale=1, then + sceneOffset) reduces to exactly this same
    // fixed translation for any later --cam-x/y/z override too.
    pbrtBboxCenter = float3{0.0f, 0.0f, 0.0f};
    pbrtSceneScale = 1.0f;
    pbrtSceneOffset = sceneOffset;
}

// Converts a flat OptiX-style metal "albedo" into an approximate complex
// conductor (eta,k) via PR #103's own already-shipped reflectance-to-k
// formula - see buildMeshGalleryScene()'s own declaration comment.
static float3 reflectanceToConductorK(float3 albedo) {
    auto k = [](float r) {
        r = r < 0.0f ? 0.0f : (r > 0.9999f ? 0.9999f : r);
        return 2.0f * sqrtf(r) / sqrtf(std::max(1e-4f, 1.0f - r));
    };
    return float3{k(albedo.x), k(albedo.y), k(albedo.z)};
}

// Scene G1: Stanford Bunny (69,451 triangles), polished bronze - matches
// gpu/optix/scene_builder_mesh_gallery.h's own build_stanford_bunny_gpu()
// material (bronze albedo (0.71,0.43,0.20), roughness 0.15) exactly.
void MetalPocApp::buildStanfordBunny() {
    const float3 bronze{0.71f, 0.43f, 0.20f};
    buildMeshGalleryScene("stanford-bunny.obj", bronze, /*materialType=*/4u,
        /*roughness=*/0.15f, /*eta=*/float3{1, 1, 1}, reflectanceToConductorK(bronze),
        /*targetSize=*/1.1f);
}

// Scene G2: Stanford Armadillo (99,976 triangles), gunmetal - matches
// build_stanford_armadillo_gpu()'s own material (albedo (0.55,0.56,0.58),
// roughness 0.08) exactly.
void MetalPocApp::buildStanfordArmadillo() {
    const float3 gunmetal{0.55f, 0.56f, 0.58f};
    buildMeshGalleryScene("armadillo.obj", gunmetal, /*materialType=*/4u,
        /*roughness=*/0.08f, /*eta=*/float3{1, 1, 1}, reflectanceToConductorK(gunmetal),
        /*targetSize=*/1.1f);
}

// Scene G3: Stanford Happy Buddha (98,601 triangles), polished gold -
// matches build_stanford_happy_buddha_gpu()'s own material (albedo
// (0.83,0.69,0.22), roughness 0.05) exactly.
void MetalPocApp::buildStanfordHappyBuddha() {
    const float3 gold{0.83f, 0.69f, 0.22f};
    buildMeshGalleryScene("happy-buddha.obj", gold, /*materialType=*/4u,
        /*roughness=*/0.05f, /*eta=*/float3{1, 1, 1}, reflectanceToConductorK(gold),
        /*targetSize=*/1.1f);
}

// Scenes G4-G24 below (minus G7/G10/G12/G13 - see this struct's own
// declaration comment for why those are deferred) - each matches
// gpu/optix/scene_builder_mesh_gallery.h's own real per-scene albedo/
// roughness exactly, same reflectanceToConductorK() conversion G1-G3
// already use.

// G4: Stanford Lucy (99,970 triangles), bright silver.
void MetalPocApp::buildStanfordLucy() {
    const float3 silver{0.85f, 0.85f, 0.88f};
    buildMeshGalleryScene("lucy.obj", silver, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(silver), 2.5f);
}

// G5: Stanford XYZRGB Dragon (249,882 triangles), bright silver.
void MetalPocApp::buildStanfordDragon() {
    const float3 silver{0.85f, 0.85f, 0.88f};
    buildMeshGalleryScene("xyzrgb_dragon.obj", silver, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(silver), 2.5f);
}

// G6: Utah Teapot (6,320 triangles), bright silver.
void MetalPocApp::buildUtahTeapot() {
    const float3 silver{0.85f, 0.85f, 0.88f};
    buildMeshGalleryScene("teapot.obj", silver, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(silver), 2.5f);
}

// G8: Suzanne (968 triangles, fan-triangulated from 500 mostly-quad
// faces), bright silver - a DIFFERENT material from the hardcoded POC
// room's own bronze Suzanne (buildScene()'s own materialType 0), same
// mesh file, genuinely separate scene/context.
void MetalPocApp::buildSuzanneGallery() {
    const float3 silver{0.85f, 0.85f, 0.88f};
    buildMeshGalleryScene("suzanne.obj", silver, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(silver), 1.1f);
}

// G9: Nefertiti Bust (99,938 triangles), bright silver.
void MetalPocApp::buildNefertiti() {
    const float3 silver{0.85f, 0.85f, 0.88f};
    buildMeshGalleryScene("nefertiti.obj", silver, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(silver), 2.5f);
}

// G11: Cheburashka bust (13,334 triangles), bright silver.
void MetalPocApp::buildCheburashka() {
    const float3 silver{0.85f, 0.85f, 0.88f};
    buildMeshGalleryScene("cheburashka.obj", silver, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(silver), 2.2f);
}

// G14: Beast, bronze (same tone as G1's bunny).
void MetalPocApp::buildBeast() {
    const float3 bronze{0.71f, 0.43f, 0.20f};
    buildMeshGalleryScene("beast.obj", bronze, 4u, 0.15f, float3{1, 1, 1},
        reflectanceToConductorK(bronze), 1.1f);
}

// G15: VW Beetle, bright silver.
void MetalPocApp::buildVWBeetle() {
    const float3 silver{0.85f, 0.85f, 0.88f};
    buildMeshGalleryScene("beetle.obj", silver, 4u, 0.08f, float3{1, 1, 1},
        reflectanceToConductorK(silver), 2.5f);
}

// G17: Bimba, gold (same tone as G3's buddha).
void MetalPocApp::buildBimba() {
    const float3 gold{0.83f, 0.69f, 0.22f};
    buildMeshGalleryScene("bimba.obj", gold, 4u, 0.05f, float3{1, 1, 1},
        reflectanceToConductorK(gold), 1.1f);
}

// G18: Cow, warm brass tone.
void MetalPocApp::buildCowGallery() {
    const float3 brass{0.80f, 0.65f, 0.28f};
    buildMeshGalleryScene("cow.obj", brass, 4u, 0.15f, float3{1, 1, 1},
        reflectanceToConductorK(brass), 1.1f);
}

// G19: Fandisk, gunmetal.
void MetalPocApp::buildFandisk() {
    const float3 gunmetal{0.55f, 0.56f, 0.58f};
    buildMeshGalleryScene("fandisk.obj", gunmetal, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(gunmetal), 2.5f);
}

// G20: Homer, warm gold tone.
void MetalPocApp::buildHomer() {
    const float3 gold{0.85f, 0.70f, 0.25f};
    buildMeshGalleryScene("homer.obj", gold, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(gold), 1.1f);
}

// G21: Igea, bright silver.
void MetalPocApp::buildIgea() {
    const float3 silver{0.85f, 0.85f, 0.88f};
    buildMeshGalleryScene("igea.obj", silver, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(silver), 1.1f);
}

// G22: Max Planck bust, warm copper tone.
void MetalPocApp::buildMaxPlanck() {
    const float3 copper{0.65f, 0.45f, 0.30f};
    buildMeshGalleryScene("max-planck.obj", copper, 4u, 0.2f, float3{1, 1, 1},
        reflectanceToConductorK(copper), 1.1f);
}

// G23: Ogre, muted green-tinted metal.
void MetalPocApp::buildOgre() {
    const float3 tint{0.45f, 0.50f, 0.35f};
    buildMeshGalleryScene("ogre.obj", tint, 4u, 0.2f, float3{1, 1, 1},
        reflectanceToConductorK(tint), 1.1f);
}

// G24: Rocker Arm, gunmetal.
void MetalPocApp::buildRockerArm() {
    const float3 gunmetal{0.55f, 0.56f, 0.58f};
    buildMeshGalleryScene("rocker-arm.obj", gunmetal, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(gunmetal), 2.5f);
}

// G7: Spot the Cow (Keenan Crane), bright silver - flipXZ=true, matching
// OptiX's own build_spot_cow_gpu() comment ("the raw mesh faces away
// from the camera").
void MetalPocApp::buildSpotCow() {
    const float3 silver{0.85f, 0.85f, 0.88f};
    buildMeshGalleryScene("spot.obj", silver, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(silver), 1.3f, /*ior=*/1.0f, /*flipXZ=*/true);
}

// G10: Horse (classic geometry-processing test model), bright silver -
// flipXZ=true, matching OptiX's own build_horse_gpu() comment.
void MetalPocApp::buildHorse() {
    const float3 silver{0.85f, 0.85f, 0.88f};
    buildMeshGalleryScene("horse.obj", silver, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(silver), 2.2f, /*ior=*/1.0f, /*flipXZ=*/true);
}

// G13: Glass Dragon - same mesh/scale as G5's metal dragon, clear glass
// (materialType 2, ior 1.5) instead of a conductor - matches OptiX's own
// build_glass_dragon_gpu(). meshColor/roughness/eta/k are all ignored for
// materialType 2 (see loadObjMesh()'s own TriangleMaterial-construction
// comment) - passed as harmless placeholders.
void MetalPocApp::buildGlassDragon() {
    buildMeshGalleryScene("xyzrgb_dragon.obj", float3{1, 1, 1}, /*materialType=*/2u,
        /*roughness=*/0.0f, float3{1, 1, 1}, float3{0, 0, 0}, /*targetSize=*/2.5f,
        /*ior=*/1.5f);
}

// G12: Trophy Room - the last category-G scene, four already-verified
// meshes (bunny/teapot/Suzanne/Spot the Cow) lined up on one shared shelf
// in bronze/chrome/gold/gunmetal, matching CPU's build_trophy_room()/
// OptiX's own build_trophy_room_gpu() in mesh choice and material tones -
// NOT their exact numeric scale/offset, which are tuned for OptiX's own
// raw-multiply-then-translate placement convention. This loader's
// loadObjMesh() instead auto-fits each mesh to a caller-chosen targetSize
// around a caller-chosen world-space centre (see buildMeshGalleryScene()'s
// own declaration comment) - a different enough convention that porting
// OptiX's literal numbers would not reproduce the same layout. The first
// hand-authored Metal scene to place multiple external OBJ meshes in one
// composition - genuinely a different shape from buildMeshGalleryScene()
// (single mesh only), so this is a bespoke builder, not a call to it.
// Section 120, docs/METAL_GPU_FEASIBILITY.md.
void MetalPocApp::buildTrophyRoom() {
    const float3 sceneOffset{8.0f, 0.0f, 0.0f};

    // Ground: one large flat quad wide enough for all four meshes (same
    // "flat quad, not a checker sphere" simplification as
    // buildMeshGalleryScene() - see its own declaration comment for why).
    const float3 groundColor{0.5f, 0.5f, 0.5f};
    addQuad(verts, normals, uvs, materials,
            float3{-5.0f, 0.0f, -2.5f} + sceneOffset, float3{5.0f, 0.0f, -2.5f} + sceneOffset,
            float3{5.0f, 0.0f, 2.5f} + sceneOffset, float3{-5.0f, 0.0f, 2.5f} + sceneOffset, groundColor);

    NSString* modelsDir = nil;
    {
        NSString* exeDir = executableDir();
        NSString* candidate = [exeDir stringByAppendingPathComponent:@"models"];
        if (exeDir && [[NSFileManager defaultManager] fileExistsAtPath:
                [candidate stringByAppendingPathComponent:@"stanford-bunny.obj"]]) {
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

    // Same bronze/chrome/gold/gunmetal tones as CPU's/OptiX's own trophy
    // room, spaced 2.4 units apart along the shelf's own x-axis, each
    // auto-fit to a size that keeps it clear of its neighbours.
    const float3 bronze{0.71f, 0.43f, 0.20f};
    const float3 chrome{0.85f, 0.85f, 0.88f};
    const float3 gold{0.83f, 0.69f, 0.22f};
    const float3 gunmetal{0.55f, 0.56f, 0.58f};
    const float3 meshCenterY{0.0f, 0.45f, 0.0f};

    NSString* bunnyPath = [modelsDir stringByAppendingPathComponent:@"stanford-bunny.obj"];
    if (!loadObjMesh(bunnyPath.UTF8String, verts, normals, uvs, materials, bronze,
                      meshCenterY + float3{-3.6f, 0.0f, 0.0f} + sceneOffset, /*targetSize=*/1.1f,
                      /*materialType=*/4u, /*roughness=*/0.15f, float3{1, 1, 1},
                      reflectanceToConductorK(bronze))) {
        fprintf(stderr, "buildTrophyRoom: continuing without stanford-bunny.obj.\n");
    }

    NSString* teapotPath = [modelsDir stringByAppendingPathComponent:@"teapot.obj"];
    if (!loadObjMesh(teapotPath.UTF8String, verts, normals, uvs, materials, chrome,
                      meshCenterY + float3{-1.2f, 0.0f, 0.0f} + sceneOffset, /*targetSize=*/1.4f,
                      /*materialType=*/4u, /*roughness=*/0.10f, float3{1, 1, 1},
                      reflectanceToConductorK(chrome))) {
        fprintf(stderr, "buildTrophyRoom: continuing without teapot.obj.\n");
    }

    NSString* suzannePath = [modelsDir stringByAppendingPathComponent:@"suzanne.obj"];
    if (!loadObjMesh(suzannePath.UTF8String, verts, normals, uvs, materials, gold,
                      meshCenterY + float3{1.2f, 0.0f, 0.0f} + sceneOffset, /*targetSize=*/1.1f,
                      /*materialType=*/4u, /*roughness=*/0.05f, float3{1, 1, 1},
                      reflectanceToConductorK(gold))) {
        fprintf(stderr, "buildTrophyRoom: continuing without suzanne.obj.\n");
    }

    // Spot the Cow: flipXZ=true, the same "raw mesh faces away from this
    // app's own camera" fix G7's own solo scene needed (section 119).
    NSString* spotPath = [modelsDir stringByAppendingPathComponent:@"spot.obj"];
    if (!loadObjMesh(spotPath.UTF8String, verts, normals, uvs, materials, gunmetal,
                      meshCenterY + float3{3.6f, 0.0f, 0.0f} + sceneOffset, /*targetSize=*/1.3f,
                      /*materialType=*/4u, /*roughness=*/0.08f, float3{1, 1, 1},
                      reflectanceToConductorK(gunmetal), /*ior=*/1.0f, /*flipXZ=*/true)) {
        fprintf(stderr, "buildTrophyRoom: continuing without spot.obj.\n");
    }

    // A wide quad area light spanning the whole shelf (this loader's only
    // NEE-sampled light shape, not OptiX's own sphere light - see
    // buildMeshGalleryScene()'s own declaration comment for why).
    {
        const float3 a = float3{-4.0f, 3.0f, -1.0f} + sceneOffset, b = float3{4.0f, 3.0f, -1.0f} + sceneOffset,
                     c = float3{4.0f, 3.0f, 1.0f} + sceneOffset, d = float3{-4.0f, 3.0f, 1.0f} + sceneOffset;
        const float3 lightColor{6.0f, 6.0f, 6.0f};
        const int32_t lightId = (int32_t)lights.size();
        addQuad(verts, normals, uvs, materials, a, b, c, d, lightColor,
                /*materialType=*/0u, /*emission=*/lightColor, lightId);
        const float3 edgeU = b - a, edgeV = d - a;
        const float3 normalV = simd::normalize(simd::cross(edgeU, edgeV));
        const float area = simd::length(simd::cross(edgeU, edgeV));
        const float3 center = a + 0.5f * edgeU + 0.5f * edgeV;
        lights.push_back(AreaLightData{
            PackedFloat3{center.x, center.y, center.z},
            PackedFloat3{edgeU.x, edgeU.y, edgeU.z},
            PackedFloat3{edgeV.x, edgeV.y, edgeV.z},
            PackedFloat3{normalV.x, normalV.y, normalV.z},
            area, PackedFloat3{lightColor.x, lightColor.y, lightColor.z},
            /*patternTileB=*/0.0f, /*patternScale=*/0.0f,
            /*twoSided=*/1.0f, /*useTexture=*/0.0f});
    }

    // Camera: wide enough to frame all four meshes across the shelf's own
    // ~7.2-unit spread, the same simple 3/4-elevated-view convention as
    // buildMeshGalleryScene()'s own fallback camera - not a literal port of
    // the CPU registry's own G12 camera row (vfov 34, lookfrom (0,2.3,14)),
    // which is tuned for OptiX's own raw-scale placement, not this
    // targetSize-based one (see this function's own declaration comment).
    const float3 lookfrom = float3{0.0f, 1.8f, 8.0f} + sceneOffset;
    const float3 lookat = float3{0.0f, 0.6f, 0.0f} + sceneOffset;
    const float3 up{0.0f, 1.0f, 0.0f};
    const float3 forward = simd::normalize(lookat - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    pbrtTanHalfFov = tanf(0.5f * 55.0f * (float)M_PI / 180.0f);
    havePbrtCamera = true;
    pbrtCameraLookAtWorld = lookat;
    pbrtCameraUpRaw = up;
    pbrtBboxCenter = float3{0.0f, 0.0f, 0.0f};
    pbrtSceneScale = 1.0f;
    pbrtSceneOffset = sceneOffset;
}

// A3: Checkered Spheres - matches CPU's build_checkered_spheres() (src/
// TheRestOfYourLife/scenes_book.h) exactly: same 5 spheres (2 giant
// checker "planets" + 3 small accent spheres), same positions/radii/
// materials/colours, just offset by sceneOffset. Direct spheres.push_back()
// calls, no shared helper - simple enough not to need one, and (unlike
// G12) nothing here goes through loadObjMesh()'s own auto-fit convention,
// so CPU's/OptiX's own real camera params port over directly too.
void MetalPocApp::buildCheckeredSpheres() {
    // Same +8 offset convention every other hand-authored scene uses
    // (A1/G1-G24/G12) - even though this scene's own two checker
    // spheres have a much bigger radius (10) than any earlier scene's
    // geometry, their actual SURFACE never reaches the hardcoded room's
    // own [-1,1] region at this offset (the closest a sphere centred at
    // (8,+-10,0) radius 10 gets to the room is still well outside it -
    // checked algebraically, not assumed), so there is no real geometric
    // overlap bug to work around here, just a much bigger bounding
    // volume than usual for the BVH to skip past. Section 121, docs/
    // METAL_GPU_FEASIBILITY.md's own note on this scene's real per-pixel
    // verification challenge (its own extreme-grazing-angle framing) -
    // read that before assuming a large Metal-vs-CPU pixel diff here
    // means a bug.
    const float3 sceneOffset{8.0f, 0.0f, 0.0f};

    // The two checker "planet" spheres - materialType 16 (real 3D world-
    // space checker - see checker3DColor()'s own declaration comment,
    // metal_poc.metal). `roughness` reused as the checker's own world-
    // space cell scale (0.32, matching CPU's checker_texture construction
    // parameter exactly); `color`/`transmitColor` hold the two REAL tile
    // colours CPU actually uses ((.2,.3,.1)/(.9,.9,.9)) - not a fixed-
    // fraction-of-one-colour approximation the way materialType 6 (UV-
    // checker, section 121's own declaration comment) needs to avoid the
    // emission-field collision; that collision doesn't apply here since
    // `transmitColor` is otherwise unused by any Lambertian-family
    // material and isn't read by the direct-hit emissive check at all.
    const float3 tileA{0.2f, 0.3f, 0.1f};
    const float3 tileB{0.9f, 0.9f, 0.9f};
    auto pushChecker = [&](float3 center, float radius) {
        TriangleMaterial mat{PackedFloat3{tileA.x, tileA.y, tileA.z}, /*materialType=*/16u,
                              /*ior=*/1.0f, PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.32f};
        mat.transmitColor = PackedFloat3{tileB.x, tileB.y, tileB.z};
        spheres.push_back(SphereData{PackedFloat3{center.x, center.y, center.z}, radius});
        sphereMaterials.push_back(mat);
    };
    pushChecker(float3{0.0f, -10.0f, 0.0f} + sceneOffset, 10.0f);
    pushChecker(float3{0.0f, 10.0f, 0.0f} + sceneOffset, 10.0f);

    // 3 small accent spheres resting on the lower "planet"'s visible cap -
    // same positions/radii/materials/colours as CPU's own
    // build_checkered_spheres(), just offset.
    {
        TriangleMaterial mat{PackedFloat3{0.55f, 0.15f, 0.10f}, /*materialType=*/0u,
                              1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f};
        const float3 c = float3{1.6f, 0.5f, 2.2f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 0.9f});
        sphereMaterials.push_back(mat);
    }
    {
        const float3 metalColor{0.8f, 0.75f, 0.6f};
        // Isotropic: ior (alphaX) and roughness (alphaY) both 0.05 - see
        // loadObjMesh()'s own comment on why both must match.
        TriangleMaterial mat{PackedFloat3{metalColor.x, metalColor.y, metalColor.z}, /*materialType=*/4u,
                              /*ior(alphaX)=*/0.05f, PackedFloat3{0, 0, 0}, -1, /*roughness(alphaY)=*/0.05f};
        const float3 k = reflectanceToConductorK(metalColor);
        mat.conductorEta = PackedFloat3{1.0f, 1.0f, 1.0f};
        mat.conductorK = PackedFloat3{k.x, k.y, k.z};
        const float3 c = float3{-1.4f, 0.45f, 1.6f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 0.7f});
        sphereMaterials.push_back(mat);
    }
    {
        TriangleMaterial mat{PackedFloat3{1.0f, 1.0f, 1.0f}, /*materialType=*/2u,
                              /*ior=*/1.5f, PackedFloat3{0, 0, 0}, -1, 0.0f};
        const float3 c = float3{0.1f, 0.15f, 3.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 0.6f});
        sphereMaterials.push_back(mat);
    }

    // No dedicated light source - matches CPU's own registry row for A3
    // (sky_dummy_lights: no real scene light at all), same as several
    // other Basics scenes. Every material above is lit purely by the
    // miss-path background below - noisier without an NEE strategy for
    // it (none exists for a pure background light in this shader), not
    // biased.

    // Real per-scene flat background colour, REUSING the existing pbrt-
    // constant-infinite-light mechanism (havePbrtConstantEnvLight/
    // pbrtEnvColor, wired in metal_render_main()'s own uniforms setup)
    // rather than adding a new uniform: semantically identical to what a
    // pbrt scene's own `LightSource "infinite" "rgb L"` already does for
    // the miss path, and havePbrtCamera is already true below for every
    // hand-authored scene, so this is picked up automatically with no
    // new plumbing. Matches CPU's own registry row for A3 exactly - bg
    // (0.90, 0.75, 0.55), a warm sunset tint, not metal_poc.metal's own
    // hardcoded blue-sky gradient (skyBottom/skyTop) that every OTHER
    // hand-authored scene so far has silently fallen back to (harmless
    // for those - none of them has a visible open sky in frame).
    havePbrtConstantEnvLight = true;
    pbrtEnvColor = float3{0.90f, 0.75f, 0.55f};

    // Camera: CPU's/OptiX's own real registry row for A3, ported
    // DIRECTLY (vfov 20, lookfrom (13,2,3), lookat (0,0,0)) - unlike
    // G12's own fallback camera, nothing in this scene goes through
    // loadObjMesh()'s targetSize/centre auto-fit convention, so there's
    // no placement-convention mismatch to work around here at all.
    const float3 lookfrom = float3{13.0f, 2.0f, 3.0f} + sceneOffset;
    const float3 lookat = float3{0.0f, 0.0f, 0.0f} + sceneOffset;
    const float3 up{0.0f, 1.0f, 0.0f};
    const float3 forward = simd::normalize(lookat - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    pbrtTanHalfFov = tanf(0.5f * 20.0f * (float)M_PI / 180.0f);
    havePbrtCamera = true;
    pbrtCameraLookAtWorld = lookat;
    pbrtCameraUpRaw = up;
    pbrtBboxCenter = float3{0.0f, 0.0f, 0.0f};
    pbrtSceneScale = 1.0f;
    pbrtSceneOffset = sceneOffset;
}

// A6: Colored Quads - matches CPU's build_quads()/build_quads_lights()
// (src/TheRestOfYourLife/scenes_book.h) exactly: 5 flat-colour wall
// quads (Q, u, v parallelograms, each converted to addQuad()'s own
// a/b/c/d corners as a=Q, b=Q+u, c=Q+u+v, d=Q+v - the SAME winding
// CPU's own quad class uses internally, normalize(cross(u,v)), so this
// is a direct, not reconstructed, port) plus one emissive lamp quad,
// registered as a real NEE-sampled AreaLight the same way
// buildCornellBoxA1()'s own light quad already is. Pure addQuad() calls -
// no new material/geometry machinery at all, every quad here is
// materialType 0 (plain Lambertian).
void MetalPocApp::buildColoredQuads() {
    const float3 sceneOffset{8.0f, 0.0f, 0.0f};

    auto pushWall = [&](float3 Q, float3 u, float3 v, float3 color) {
        const float3 a = Q + sceneOffset, b = Q + u + sceneOffset,
                     c = Q + u + v + sceneOffset, d = Q + v + sceneOffset;
        addQuad(verts, normals, uvs, materials, a, b, c, d, color);
    };
    pushWall(float3{-3, -2, 5}, float3{0, 0, -4}, float3{0, 4, 0}, float3{1.0f, 0.2f, 0.2f});  // left_red
    pushWall(float3{-2, -2, 0}, float3{4, 0, 0}, float3{0, 4, 0}, float3{0.2f, 1.0f, 0.2f});   // back_green
    pushWall(float3{3, -2, 1}, float3{0, 0, 4}, float3{0, 4, 0}, float3{0.2f, 0.2f, 1.0f});    // right_blue
    pushWall(float3{-2, 3, 1}, float3{4, 0, 0}, float3{0, 0, 4}, float3{1.0f, 0.5f, 0.0f});    // upper_orange
    pushWall(float3{-2, -3, 5}, float3{4, 0, 0}, float3{0, 0, -4}, float3{0.2f, 0.8f, 0.8f});  // lower_teal

    // The lamp quad - a real NEE-sampled AreaLight, same pattern
    // buildCornellBoxA1()'s own ceiling light and buildMeshGalleryScene()'s
    // own quad light already use.
    {
        const float3 Q{-1.0f, 0.5f, 3.0f}, u{2.0f, 0.0f, 0.0f}, v{0.0f, 1.0f, 0.0f};
        const float3 a = Q + sceneOffset, b = Q + u + sceneOffset,
                     c = Q + u + v + sceneOffset, d = Q + v + sceneOffset;
        const float3 lampColor{7.0f, 7.0f, 6.5f};
        const int32_t lightId = (int32_t)lights.size();
        addQuad(verts, normals, uvs, materials, a, b, c, d, lampColor,
                /*materialType=*/0u, /*emission=*/lampColor, lightId);
        const float3 edgeU = b - a, edgeV = d - a;
        const float3 normalV = simd::normalize(simd::cross(edgeU, edgeV));
        const float area = simd::length(simd::cross(edgeU, edgeV));
        const float3 center = a + 0.5f * edgeU + 0.5f * edgeV;
        lights.push_back(AreaLightData{
            PackedFloat3{center.x, center.y, center.z},
            PackedFloat3{edgeU.x, edgeU.y, edgeU.z},
            PackedFloat3{edgeV.x, edgeV.y, edgeV.z},
            PackedFloat3{normalV.x, normalV.y, normalV.z},
            area, PackedFloat3{lampColor.x, lampColor.y, lampColor.z},
            /*patternTileB=*/0.0f, /*patternScale=*/0.0f,
            /*twoSided=*/0.0f, /*useTexture=*/0.0f});
    }

    // Real per-scene flat background colour (sky blue, (0.70,0.80,1.00)) -
    // same reuse of the pbrt-constant-infinite-light mechanism A3's own
    // buildCheckeredSpheres() already established (see that function's
    // own comment) rather than a new uniform. Close to, but not exactly,
    // metal_poc.metal's own hardcoded skyBottom/skyTop gradient default -
    // set explicitly anyway for a real, not coincidental, match.
    havePbrtConstantEnvLight = true;
    pbrtEnvColor = float3{0.70f, 0.80f, 1.00f};

    // Camera: CPU's own real registry row for A6, ported directly
    // (vfov 80, lookfrom (0,0,9), lookat (0,0,0)) - same direct-port
    // convention A3's own camera already established (no mesh/targetSize
    // involved here either).
    const float3 lookfrom = float3{0.0f, 0.0f, 9.0f} + sceneOffset;
    const float3 lookat = float3{0.0f, 0.0f, 0.0f} + sceneOffset;
    const float3 up{0.0f, 1.0f, 0.0f};
    const float3 forward = simd::normalize(lookat - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    pbrtTanHalfFov = tanf(0.5f * 80.0f * (float)M_PI / 180.0f);
    havePbrtCamera = true;
    pbrtCameraLookAtWorld = lookat;
    pbrtCameraUpRaw = up;
    pbrtBboxCenter = float3{0.0f, 0.0f, 0.0f};
    pbrtSceneScale = 1.0f;
    pbrtSceneOffset = sceneOffset;
}

// A4: Earth - matches CPU's build_earth()/build_earth_lights() exactly:
// a globe sphere (radius 2, origin-centred before offset) textured with
// earthTexture (materialType 3 - see this file's own shader-side
// materialType==3 comment for how a SPHERE hit gets a UV at all), a
// small flat-grey "moon" accent sphere, and a rim-light quad behind the
// globe, registered as a real NEE-sampled AreaLight the same way every
// earlier hand-authored scene's own light quad already is.
void MetalPocApp::buildEarth() {
    const float3 sceneOffset{8.0f, 0.0f, 0.0f};

    // Earth globe - materialType 3, `color` unused (the real albedo
    // comes from earthTexture, sampled via equirectangularUV(normal) -
    // see this material's own shader-side comment).
    {
        TriangleMaterial mat{PackedFloat3{1.0f, 1.0f, 1.0f}, /*materialType=*/3u,
                              1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f};
        const float3 c = float3{0.0f, 0.0f, 0.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 2.0f});
        sphereMaterials.push_back(mat);
    }
    // Moon accent sphere - flat grey Lambertian, matches CPU exactly.
    {
        TriangleMaterial mat{PackedFloat3{0.6f, 0.6f, 0.62f}, /*materialType=*/0u,
                              1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f};
        const float3 c = float3{2.0f, 1.3f, 0.5f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 0.35f});
        sphereMaterials.push_back(mat);
    }
    // Rim light quad, behind the globe.
    {
        const float3 Q{-4.0f, -2.5f, -6.0f}, u{3.0f, 0.0f, 0.0f}, v{0.0f, 5.0f, 0.0f};
        const float3 a = Q + sceneOffset, b = Q + u + sceneOffset,
                     c = Q + u + v + sceneOffset, d = Q + v + sceneOffset;
        const float3 rimColor{0.9f, 1.0f, 1.3f};
        const int32_t lightId = (int32_t)lights.size();
        addQuad(verts, normals, uvs, materials, a, b, c, d, rimColor,
                /*materialType=*/0u, /*emission=*/rimColor, lightId);
        const float3 edgeU = b - a, edgeV = d - a;
        const float3 normalV = simd::normalize(simd::cross(edgeU, edgeV));
        const float area = simd::length(simd::cross(edgeU, edgeV));
        const float3 center = a + 0.5f * edgeU + 0.5f * edgeV;
        lights.push_back(AreaLightData{
            PackedFloat3{center.x, center.y, center.z},
            PackedFloat3{edgeU.x, edgeU.y, edgeU.z},
            PackedFloat3{edgeV.x, edgeV.y, edgeV.z},
            PackedFloat3{normalV.x, normalV.y, normalV.z},
            area, PackedFloat3{rimColor.x, rimColor.y, rimColor.z},
            /*patternTileB=*/0.0f, /*patternScale=*/0.0f,
            /*twoSided=*/0.0f, /*useTexture=*/0.0f});
    }

    // Real per-scene flat background (sky blue, (0.70,0.80,1.00)) - same
    // reuse of the pbrt-constant-infinite-light mechanism A3/A6 already
    // established.
    havePbrtConstantEnvLight = true;
    pbrtEnvColor = float3{0.70f, 0.80f, 1.00f};

    // Camera: CPU's own real registry row for A4, ported directly
    // (vfov 25, lookfrom (0,0,12), lookat (0,0,0)).
    const float3 lookfrom = float3{0.0f, 0.0f, 12.0f} + sceneOffset;
    const float3 lookat = float3{0.0f, 0.0f, 0.0f} + sceneOffset;
    const float3 up{0.0f, 1.0f, 0.0f};
    const float3 forward = simd::normalize(lookat - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    pbrtTanHalfFov = tanf(0.5f * 25.0f * (float)M_PI / 180.0f);
    havePbrtCamera = true;
    pbrtCameraLookAtWorld = lookat;
    pbrtCameraUpRaw = up;
    pbrtBboxCenter = float3{0.0f, 0.0f, 0.0f};
    pbrtSceneScale = 1.0f;
    pbrtSceneOffset = sceneOffset;
}

// A5: Perlin Spheres - matches CPU's build_perlin_spheres()/
// build_perlin_spheres_lights() exactly: 4 spheres sharing materialType
// 17 (real Perlin marble - see turbulenceSimple()'s own declaration
// comment, metal_poc.metal) at two different noise scales, plus a warm
// key-light quad.
void MetalPocApp::buildPerlinSpheres() {
    const float3 sceneOffset{8.0f, 0.0f, 0.0f};

    // `roughness` reused as the marble texture's own `scale` parameter
    // (matches materialType 16's own established reuse of the same
    // field for a different procedural texture's own scale).
    auto pushMarble = [&](float3 center, float radius, float noiseScale) {
        TriangleMaterial mat{PackedFloat3{1.0f, 1.0f, 1.0f}, /*materialType=*/17u,
                              1.0f, PackedFloat3{0, 0, 0}, -1, /*roughness=*/noiseScale};
        const float3 c = center + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, radius});
        sphereMaterials.push_back(mat);
    };
    // CPU's own "ground" is a radius-1000 sphere centred (0,-1000,0) - a
    // classic book trick for a near-flat plane at this scale, but its
    // surface stays within +-1 of y=0 out to roughly +-45 units in x/z
    // (checked algebraically: solving the sphere equation at y=+-1 gives
    // |x-centre.x| <= sqrt(2000-1) ~ 44.7) - the usual +8 sceneOffset
    // is NOWHERE near enough clearance, so the sphere's own surface
    // would genuinely intersect the hardcoded POC room's own already-
    // occupied [-1,1] region (unlike A3's own radius-10 checker spheres,
    // section 121's own comment, which really don't reach that far).
    // Using an offset large enough to clear a RADIUS-1000 sphere would
    // need ~50+ units, an awkward, easy-to-get-wrong magic number - so
    // this reuses the SAME "flat quad instead of a huge sphere"
    // simplification category-G's own mesh gallery already established
    // (section 117) for exactly this "near-flat surface, no real
    // curvature visible at this camera distance" situation. materialType
    // 17 needs no UV either way (world-space `hitPoint, same as
    // materialType 16), so a quad works identically to a sphere here.
    {
        const float3 groundColor{1.0f, 1.0f, 1.0f};
        addQuad(verts, normals, uvs, materials,
                float3{-15.0f, 0.0f, -15.0f} + sceneOffset, float3{15.0f, 0.0f, -15.0f} + sceneOffset,
                float3{15.0f, 0.0f, 15.0f} + sceneOffset, float3{-15.0f, 0.0f, 15.0f} + sceneOffset,
                groundColor, /*materialType=*/17u, /*emission=*/simd::make_float3(0, 0, 0),
                /*lightId=*/-1, /*roughness=*/4.0f);
    }
    pushMarble(float3{0.0f, 2.0f, 0.0f}, 2.0f, 4.0f);         // main sphere
    pushMarble(float3{2.2f, 0.8f, 1.0f}, 0.8f, 8.0f);         // companion 1
    pushMarble(float3{-1.8f, 0.6f, -1.2f}, 0.6f, 8.0f);       // companion 2

    // Warm key-light quad, upper-left.
    {
        const float3 Q{-4.0f, 6.0f, -3.0f}, u{4.0f, 0.0f, 0.0f}, v{0.0f, 0.0f, 4.0f};
        const float3 a = Q + sceneOffset, b = Q + u + sceneOffset,
                     c = Q + u + v + sceneOffset, d = Q + v + sceneOffset;
        const float3 keyColor{8.0f, 6.0f, 3.0f};
        const int32_t lightId = (int32_t)lights.size();
        addQuad(verts, normals, uvs, materials, a, b, c, d, keyColor,
                /*materialType=*/0u, /*emission=*/keyColor, lightId);
        const float3 edgeU = b - a, edgeV = d - a;
        const float3 normalV = simd::normalize(simd::cross(edgeU, edgeV));
        const float area = simd::length(simd::cross(edgeU, edgeV));
        const float3 center = a + 0.5f * edgeU + 0.5f * edgeV;
        lights.push_back(AreaLightData{
            PackedFloat3{center.x, center.y, center.z},
            PackedFloat3{edgeU.x, edgeU.y, edgeU.z},
            PackedFloat3{edgeV.x, edgeV.y, edgeV.z},
            PackedFloat3{normalV.x, normalV.y, normalV.z},
            area, PackedFloat3{keyColor.x, keyColor.y, keyColor.z},
            /*patternTileB=*/0.0f, /*patternScale=*/0.0f,
            /*twoSided=*/0.0f, /*useTexture=*/0.0f});
    }

    // Real per-scene flat background (sky blue) - same reuse of the
    // pbrt-constant-infinite-light mechanism A3/A6/A4 already
    // established.
    havePbrtConstantEnvLight = true;
    pbrtEnvColor = float3{0.70f, 0.80f, 1.00f};

    // Camera: CPU's own real registry row for A5, ported directly
    // (vfov 20, lookfrom (13,2,3), lookat (0,0,0)).
    const float3 lookfrom = float3{13.0f, 2.0f, 3.0f} + sceneOffset;
    const float3 lookat = float3{0.0f, 0.0f, 0.0f} + sceneOffset;
    const float3 up{0.0f, 1.0f, 0.0f};
    const float3 forward = simd::normalize(lookat - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    pbrtTanHalfFov = tanf(0.5f * 20.0f * (float)M_PI / 180.0f);
    havePbrtCamera = true;
    pbrtCameraLookAtWorld = lookat;
    pbrtCameraUpRaw = up;
    pbrtBboxCenter = float3{0.0f, 0.0f, 0.0f};
    pbrtSceneScale = 1.0f;
    pbrtSceneOffset = sceneOffset;
}

// Recomputes the camera basis for a new lookfrom position, in the SAME
// coordinate space (cam_x, cam_y, cam_z) already arrive in from every
// other backend - cpu_scene_recommended_camera()'s return values and any
// explicit CLI --cam-x/y/z the user typed are both in the scene's own
// pbrt-file-authored space (e.g. a classic ~555-unit Cornell box), not
// this app's internal rescaled/recentred/offset one, so the new lookfrom
// goes through the exact same transform loadPbrtScene() applied to every
// vertex/light/camera position it read. Mirrors cpu_interface.cpp's own
// applyCameraConfig(): only lookfrom moves - lookat, up, and vfov all
// stay exactly as the scene's own pbrt Camera block defined, matching
// every other backend's "override changes WHERE you stand, not WHAT
// you're looking at" semantics.
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
    // Three candidates, tried in priority order:
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
    //    build_and_deploy_macos.sh now also copies metal_poc.metal next
    //    to the bundled ray_tracer specifically so this candidate finds
    //    it.
    // 2. RT_METAL_SHADER_DIR (set by CMakeLists.txt's metal_poc target,
    //    RT_BUILD_METAL=ON path) - gpu/metal/'s absolute SOURCE
    //    directory, correct only on the machine that built this binary
    //    (a plain `cmake --build` dev loop, never distributed).
    // 3. A __FILE__-relative lookup, for the ad-hoc `clang++
    //    metal_poc.mm ...` invocation this POC started as
    //    (docs/METAL_GPU_FEASIBILITY.md section 7/8/9).
    NSString* shaderPath = nil;
    {
        char exePathBuf[4096];
        uint32_t exePathSize = sizeof(exePathBuf);
        if (_NSGetExecutablePath(exePathBuf, &exePathSize) == 0) {
            NSString* exeDir = [@(exePathBuf) stringByDeletingLastPathComponent];
            NSString* candidate = [exeDir stringByAppendingPathComponent:@"metal_poc.metal"];
            if ([[NSFileManager defaultManager] fileExistsAtPath:candidate]) shaderPath = candidate;
        }
    }
    if (!shaderPath) {
#ifdef RT_METAL_SHADER_DIR
        NSString* shaderDir = @(RT_METAL_SHADER_DIR);
#else
        NSString* shaderDir = [@(__FILE__) stringByDeletingLastPathComponent];
#endif
        shaderPath = [shaderDir stringByAppendingPathComponent:@"metal_poc.metal"];
    }
    NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath encoding:NSUTF8StringEncoding error:&error];
    if (!shaderSource) {
        fprintf(stderr, "Failed to read shader source at %s: %s\n",
            shaderPath.UTF8String, error.localizedDescription.UTF8String);
        return false;
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
        // No DOF yet - this v1 doesn't read pbrt's own "float lensradius"/
        // "float focaldistance" Camera parameters. focusDistance is
        // unread by the shader whenever lensRadius == 0.
        uniforms.lensRadius = 0.0f;
        uniforms.focusDistance = 1.0f;
        uniforms.cameraVelocity = PackedFloat3{0, 0, 0};   // no motion blur
        if (havePbrtMedium) {
            uniforms.fogSigmaT = pbrtFogSigmaT;
            uniforms.fogAlbedo = PackedFloat3{pbrtFogAlbedo.x, pbrtFogAlbedo.y, pbrtFogAlbedo.z};
            uniforms.fogAsymmetryG = pbrtFogAsymmetryG;
        } else {
            uniforms.fogSigmaT = 0.0f;                      // no participating medium in this scene
        }
        uniforms.useEnvironmentMap = 0u;                    // this scene's own sky, if any, replaces the hardcoded room's earthTexture-based one below
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
            float v = fmaxf(rgb[c], 0.0f) * vignette;
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
