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
    // Whether a real pbrt scene was loaded - see metal_poc_types.metal's
    // own mirrored Uniforms::isPbrtScene comment for the full "why".
    uint32_t isPbrtScene = 0;
    // Orthographic camera toggle - see metal_poc.metal's own mirrored
    // Uniforms::cameraOrthographic comment for the full mechanism.
    uint32_t cameraOrthographic = 0;
    // Spherical (equirectangular panorama) camera toggle - see
    // metal_poc.metal's own mirrored Uniforms::cameraSpherical comment
    // for the full mechanism.
    uint32_t cameraSpherical = 0;
    // Which mapping cameraSpherical's own camera uses - see
    // metal_poc_types.metal's own mirrored
    // Uniforms::sphericalMappingEqualArea comment.
    uint32_t sphericalMappingEqualArea = 0;
    // Realistic (multi-element-lens) camera toggle - pbrt-v4's own
    // RealisticCamera, D4/D8's own real port (section 157). 0 (every
    // earlier scene) keeps the existing ray generation exactly as
    // before - mutually exclusive with cameraOrthographic/cameraSpherical
    // in practice. != 0 replaces the ENTIRE primary ray generation with
    // a real per-element Snell's-law lens trace (`sampleRealisticCameraRay()`,
    // metal_poc.metal - a direct MSL port of `gpu/optix/
    // optix_device_helpers.h`'s own already-shipped CUDA
    // `sample_realistic_camera_ray()`, itself mirroring `src/shared/
    // realistic_camera.h`'s `RealisticCamera<T>::generate_ray()`/
    // `trace_lenses_from_film()`/`sample_exit_pupil()` exactly) reading
    // the `lensElements`/`exitPupilBounds` buffers below - the FOCUS-
    // ADJUSTED lens table and precomputed exit-pupil-bounds table are
    // both host-only, one-time precomputes (the same cost class as
    // building a BVH), done by directly instantiating a REAL, portable
    // `RealisticCamera<float>` host-side (`metal_poc_scenes_d.mm`) and
    // reading its own GPU-port accessors - the EXACT same strategy
    // `gpu/optix/scene_builder.cpp` already uses, not a re-derivation.
    // A sample can be fully VIGNETTED (genuinely blocked by the lens
    // system for that film position/exit-pupil sample - not an error);
    // `cameraRealisticWeight`'s own per-sample multiplicative weight
    // (cos^4(theta)/(pdf*lensRearZ^2), computed device-side, folded into
    // that sample's own initial `throughput`) naturally zeroes such a
    // sample's contribution without any special-cased early-exit.
    uint32_t cameraRealistic = 0;
    uint32_t numLensElements = 0;
    uint32_t numExitPupilBounds = 0;
    // Film half-extents/rear-element-Z, all in the SAME world-space
    // units `RealisticCamera<float>`'s own metres-based accessors
    // already convert to (its constructor divides every mm input by
    // 1000) - NOT further sceneScale-multiplied the way a hand-authored
    // scene's other world-space distances (e.g. `pbrtLensRadius`) are,
    // since D4/D8 are BOTH natural/Cornell scale already (D4 has no
    // rescale at all; D8 reuses A1's own 555-unit-to-2-unit rescale,
    // and RealisticCamera's own `camera_to_world` matrix - built from
    // the SAME already-rescaled `cameraPos`/`cameraForward`/
    // `cameraRight`/`cameraUp` this loader's every other camera mode
    // uses - already carries that scale into `su`/`sv`/`sw`, so the
    // lens system's own INTERNAL metres stay in the camera's own local
    // space, never needing a separate conversion here).
    float filmHalfX = 0.0f;
    float filmHalfY = 0.0f;
    float lensRearZ = 0.0f;
    // Real (not translate-only-approximated) camera shutter motion blur
    // (D13, section 174) - see metal_poc_types.metal's own mirrored
    // comment for the full mechanism. Defaults preserve every other
    // scene's own cameraVelocity-only path exactly.
    PackedFloat3 cameraLookAtBlur{0, 0, 0};
    PackedFloat3 cameraUpRawBlur{0, 1, 0};
    uint32_t hasCameraOrbitBlur = 0;
};

// Mirrors metal_poc.metal's own LensElement byte-for-byte - a single
// pbrt-v4 RealisticCamera lens surface, already in the FOCUS-ADJUSTED,
// metres-converted form `RealisticCamera<float>::lens_curvature_radius(i)`/
// etc. return (see Uniforms::cameraRealistic's own comment). A
// `curvatureRadius == 0` entry is the aperture STOP, not a refractive
// surface (matches `RealisticCamera<T>::LensElement`'s own `eta == 0`-
// means-stop convention exactly, just keyed on the OTHER field - see
// that struct's own comment, src/shared/realistic_camera.h, for why
// both conventions coexist there).
struct GpuLensElementData {
    float curvatureRadius;
    float thickness;
    float eta;
    float apertureRadius;
};

// Mirrors metal_poc.metal's own ExitPupilBounds byte-for-byte - one
// annulus slab's own precomputed 2D bounding box on the rear exit
// pupil (`RealisticCamera<float>::bound_exit_pupil()`'s own output,
// read back via its `exit_pupil_xmin/xmax/ymin/ymax/degenerate(i)`
// accessors) - `degenerate` true means NO valid ray leaves the lens
// system from this film radius at all (fully vignetted).
struct GpuExitPupilBoundsData {
    float xMin, xMax, yMin, yMax;
    uint32_t degenerate;
};

// AreaLightData/buildPowerLightSampler now live in metal_poc_host_math.h
// (included above) - see that header's own comment.

// buildPowerLightSampler()'s own optional per-light diagnostic line,
// preserved exactly as it read before the header extraction - passed in
// as a callback rather than baked into the (now shared, test-included)
// header itself, so metal_poc_math_tests.cpp's own calls don't spam
// stderr on every test run.
inline void logPowerLightSamplerLine(int i, double power, double pmf, double uniformPmf) {
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
inline ProjectionLightData makeProjectionLight(float3 position, float3 target, float3 worldUp,
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
inline GoniometricLightData makeGoniometricLight(float3 position, float3 target, float3 worldUp,
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
inline float3 equalAreaSquareToSphere(float u, float v) {
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
inline std::vector<uint8_t> buildGoniometricProfileImage(int size, float cosCutoff, float ringFrequency) {
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

// Mirrors metal_poc.metal's SphereData byte-for-byte. centerDelta1
// (F11, section 167) defaults to {0,0,0} - see that struct's own
// comment for why every existing 2-field `SphereData{center, radius}`
// construction site (every scene but F11) stays correct unmodified.
struct SphereData {
    PackedFloat3 center;
    float radius;
    PackedFloat3 centerDelta1{0, 0, 0};
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

// Mirrors metal_poc.metal's CylinderData byte-for-byte (section 171) -
// see that struct's own comment for why this is baked world-space
// base/axis/radius/height rather than an object-space shape plus a
// per-primitive transform.
struct CylinderData {
    PackedFloat3 base;
    PackedFloat3 axis;
    float radius;
    float height;
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
inline void addQuad(std::vector<PackedFloat3>& verts,
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
                     bool twoSided = false,
                     // Complex IOR (eta+i*k) per RGB channel, materialType
                     // == 4/9 only - see loadObjMesh()'s own identical
                     // pair for the full explanation (dual `ior`/
                     // `roughness` reuse as alphaX/alphaY for a GGX
                     // conductor). Defaults match loadObjMesh()'s own
                     // (eta=(1,1,1), not 0 - unphysical; k=0) so every
                     // pre-existing call site (never a conductor quad
                     // before section 126) is unaffected.
                     float3 conductorEta = simd::make_float3(1.0f, 1.0f, 1.0f),
                     float3 conductorK = simd::make_float3(0.0f, 0.0f, 0.0f)) {
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
    // materialType == 4 (GGX conductor): `ior` doubles as alphaX, so it
    // must equal `roughness` (alphaY) for isotropic roughness - the
    // SAME bug-prevention convention loadObjMesh() already established
    // (that function's own comment). This makes the invariant
    // impossible to violate by omission at any FUTURE conductor-quad
    // call site, rather than relying on every caller to remember it.
    const float effectiveIor = (materialType == 4u) ? roughness : ior;
    TriangleMaterial mat{packedColor, materialType, effectiveIor, packedEmission, lightId, roughness};
    mat.conductorEta = PackedFloat3{conductorEta.x, conductorEta.y, conductorEta.z};
    mat.conductorK = PackedFloat3{conductorK.x, conductorK.y, conductorK.z};
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
inline NSString* executableDir() {
    char exePathBuf[4096];
    uint32_t exePathSize = sizeof(exePathBuf);
    if (_NSGetExecutablePath(exePathBuf, &exePathSize) != 0) return nil;
    return [@(exePathBuf) stringByDeletingLastPathComponent];
}

inline bool loadObjMesh(const std::string& path,
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
inline void bilateralDenoise(const std::vector<uint8_t>& ldrIn, std::vector<uint8_t>& ldrOut,
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
    void postProcessAndWrite();
};

// Converts a flat OptiX-style metal "albedo" into an approximate complex
// conductor (eta,k) via PR #103's own already-shipped reflectance-to-k
// formula - see buildMeshGalleryScene()'s own declaration comment.
inline float3 reflectanceToConductorK(float3 albedo) {
    auto k = [](float r) {
        r = r < 0.0f ? 0.0f : (r > 0.9999f ? 0.9999f : r);
        return 2.0f * sqrtf(r) / sqrtf(std::max(1e-4f, 1.0f - r));
    };
    return float3{k(albedo.x), k(albedo.y), k(albedo.z)};
}

// Tessellates a bilinear patch (4 corners, possibly NON-PLANAR - F1's
// own genuinely new geometry primitive, pbrt-v4's own BilinearPatch
// shape) into a fine NxN triangle grid, instead of adding a real
// bounding-box custom-intersection-function primitive (this loader's
// existing sphere/disk precedent - see sphereIntersectionFunction/
// diskIntersectionFunction, metal_poc.metal's own comment on the disk
// one: "a second, genuinely DIFFERENT custom-primitive shape"). A
// deliberate, documented simplification: a fine enough tessellation of
// a bilinear (degree-1-per-axis, genuinely smooth) surface is visually
// indistinguishable from the true analytic surface at any reasonable
// render resolution, and reuses 100% already-proven triangle
// infrastructure instead of a real architecture change (a THIRD
// bounding-box geometry descriptor, a new intersection function, and
// updating every `!isSphere && !isDisk && !isSuzanneInstance`-style
// exclusion check already scattered through the main shading kernel) -
// section 154, docs/METAL_GPU_FEASIBILITY.md. Per-VERTEX normals are
// the REAL analytic bilinear-surface normal at that exact (u,v)
// (`cross(dPdu, dPdv)`, not a flat per-face fallback), smoothly
// interpolated across each triangle by the SAME barycentric
// `shadingNormalFor()` every other smooth mesh here already uses, so
// the tessellation seams stay invisible under shading even though the
// underlying triangles are flat. Normal SIGN is never resolved to a
// canonical "outward" direction (unlike a convex sphere/CPU's own
// analytic shape) - deliberately unnecessary: this helper is only ever
// used for a GGX conductor material (materialType 4), whose own
// shading always uses the ray-`facingNormal` (auto-flipped to the
// visible side, `shadeConductor`'s own `frontFace` check), never the
// raw geometric one, so an inconsistent or "inward" normal sign is
// self-correcting and invisible in the final render.
inline void addBilinearPatch(std::vector<PackedFloat3>& verts,
                              std::vector<PackedFloat3>& normals,
                              std::vector<PackedFloat2>& uvs,
                              std::vector<TriangleMaterial>& materials,
                              float3 p00, float3 p10, float3 p01, float3 p11,
                              float3 color, float roughness,
                              int subdivisions = 24) {
    auto evalP = [&](float u, float v) -> float3 {
        return (1.0f - u) * (1.0f - v) * p00 + u * (1.0f - v) * p10
             + (1.0f - u) * v * p01 + u * v * p11;
    };
    auto evalNormal = [&](float u, float v) -> float3 {
        const float3 dPdu = (1.0f - v) * (p10 - p00) + v * (p11 - p01);
        const float3 dPdv = (1.0f - u) * (p01 - p00) + u * (p11 - p10);
        return simd::normalize(simd::cross(dPdu, dPdv));
    };
    const float3 k = reflectanceToConductorK(color);
    TriangleMaterial mat{PackedFloat3{color.x, color.y, color.z}, /*materialType=*/4u,
        /*ior(alphaX)=*/roughness, PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness(alphaY)=*/roughness};
    mat.conductorEta = PackedFloat3{1.0f, 1.0f, 1.0f};
    mat.conductorK = PackedFloat3{k.x, k.y, k.z};

    auto pushV = [&](float3 p) { verts.push_back(PackedFloat3{p.x, p.y, p.z}); };
    auto pushN = [&](float3 n) { normals.push_back(PackedFloat3{n.x, n.y, n.z}); };
    for (int i = 0; i < subdivisions; ++i) {
        for (int j = 0; j < subdivisions; ++j) {
            const float u0 = (float)i / subdivisions, u1 = (float)(i + 1) / subdivisions;
            const float v0 = (float)j / subdivisions, v1 = (float)(j + 1) / subdivisions;
            const float3 g00 = evalP(u0, v0), g10 = evalP(u1, v0);
            const float3 g01 = evalP(u0, v1), g11 = evalP(u1, v1);
            const float3 n00 = evalNormal(u0, v0), n10 = evalNormal(u1, v0);
            const float3 n01 = evalNormal(u0, v1), n11 = evalNormal(u1, v1);

            pushV(g00); pushV(g10); pushV(g11);
            pushN(n00); pushN(n10); pushN(n11);
            uvs.push_back(PackedFloat2{u0, v0});
            uvs.push_back(PackedFloat2{u1, v0});
            uvs.push_back(PackedFloat2{u1, v1});
            materials.push_back(mat);

            pushV(g00); pushV(g11); pushV(g01);
            pushN(n00); pushN(n11); pushN(n01);
            uvs.push_back(PackedFloat2{u0, v0});
            uvs.push_back(PackedFloat2{u1, v1});
            uvs.push_back(PackedFloat2{u0, v1});
            materials.push_back(mat);
        }
    }
}

// Tessellates a single tapered cubic-Bezier tube (F4's own genuinely
// new geometry primitive, pbrt-v4's own CurveShape<Cylinder>, a real
// ray-curve intersection on CPU) into a triangulated "tube of quad
// rings" - matches this scene's OWN registry description exactly
// ("GPU renders the same 70 strands tessellated into tapered tubes of
// bilinear patches (matches pbrt-v4's own GPU curve strategy) rather
// than an exact curve intersection"), the SAME documented simplification
// F1's own addBilinearPatch() already established for a different
// smooth-but-not-exactly-triangle shape (section 154) - reusing the
// already-proven triangle path instead of a new custom-intersection-
// function primitive. `lengthSegments` rings of `radialSegments`
// points each are swept along the curve; each ring's own local frame
// is built by GRAM-SCHMIDT re-orthogonalizing the PREVIOUS ring's own
// right/up vectors against the new tangent (not an independent
// per-ring basis, which would twist/flip randomly ring to ring for a
// thin tube) - a simple, adequate rotation-minimizing-frame
// approximation for a gently-curving strand (not a tightly coiled
// spring, where a more careful RMF would matter). Per-vertex normals
// are the tube's own outward RADIAL direction in each ring's local
// frame (correct for a swept-circle tube, ignoring the curve's own
// typically-negligible curvature-induced normal skew) - smoothly
// interpolated the same way F1's own bilinear-patch normals already
// are. End caps are skipped entirely (the root sits at/below the
// ground plane, invisible; the tip tapers to a near-zero radius,
// visually negligible) - matches this loader's own established
// "skip what a control render shows is imperceptible" discipline.
inline void addTaperedTube(std::vector<PackedFloat3>& verts,
                            std::vector<PackedFloat3>& normals,
                            std::vector<PackedFloat2>& uvs,
                            std::vector<TriangleMaterial>& materials,
                            const float3 cp[4], float width0, float width1,
                            float3 color, int lengthSegments = 14, int radialSegments = 8) {
    auto evalBezier = [&](float t) -> float3 {
        const float mt = 1.0f - t;
        return mt * mt * mt * cp[0] + 3.0f * mt * mt * t * cp[1]
             + 3.0f * mt * t * t * cp[2] + t * t * t * cp[3];
    };
    auto evalTangent = [&](float t) -> float3 {
        const float mt = 1.0f - t;
        const float3 d = 3.0f * mt * mt * (cp[1] - cp[0]) + 6.0f * mt * t * (cp[2] - cp[1])
                        + 3.0f * t * t * (cp[3] - cp[2]);
        return simd::normalize(d);
    };

    TriangleMaterial mat{PackedFloat3{color.x, color.y, color.z}, /*materialType=*/0u,
        1.0f, PackedFloat3{0, 0, 0}, /*lightId=*/-1, 0.0f};

    // Ring 0's own initial frame: an arbitrary reference vector not
    // parallel to the tangent (world up, unless the strand starts out
    // near-vertical, in which case +X instead).
    const float3 tangent0 = evalTangent(0.0f);
    const float3 ref0 = (fabsf(tangent0.y) < 0.99f) ? float3{0, 1, 0} : float3{1, 0, 0};
    float3 right = simd::normalize(simd::cross(tangent0, ref0));
    float3 up = simd::cross(right, tangent0);

    std::vector<float3> prevRing(radialSegments), prevNormals(radialSegments);
    std::vector<float3> curRing(radialSegments), curNormals(radialSegments);
    for (int seg = 0; seg <= lengthSegments; ++seg) {
        const float t = (float)seg / (float)lengthSegments;
        const float3 center = evalBezier(t);
        const float3 tangent = evalTangent(t);
        if (seg > 0) {
            // Gram-Schmidt re-orthogonalize the running frame against
            // the new tangent - keeps the ring from twisting.
            right = simd::normalize(right - tangent * simd::dot(right, tangent));
            up = simd::cross(tangent, right);
        }
        const float radius = 0.5f * ((1.0f - t) * width0 + t * width1);
        for (int k = 0; k < radialSegments; ++k) {
            const float theta = 2.0f * (float)M_PI * (float)k / (float)radialSegments;
            const float3 radial = cosf(theta) * right + sinf(theta) * up;
            curRing[k] = center + radius * radial;
            curNormals[k] = radial;
        }
        if (seg > 0) {
            for (int k = 0; k < radialSegments; ++k) {
                const int k1 = (k + 1) % radialSegments;
                auto pushV = [&](float3 p) { verts.push_back(PackedFloat3{p.x, p.y, p.z}); };
                auto pushN = [&](float3 n) { normals.push_back(PackedFloat3{n.x, n.y, n.z}); };
                pushV(prevRing[k]); pushV(curRing[k]); pushV(curRing[k1]);
                pushN(prevNormals[k]); pushN(curNormals[k]); pushN(curNormals[k1]);
                uvs.push_back(PackedFloat2{0, 0}); uvs.push_back(PackedFloat2{1, 0}); uvs.push_back(PackedFloat2{1, 1});
                materials.push_back(mat);

                pushV(prevRing[k]); pushV(curRing[k1]); pushV(prevRing[k1]);
                pushN(prevNormals[k]); pushN(curNormals[k1]); pushN(prevNormals[k1]);
                uvs.push_back(PackedFloat2{0, 0}); uvs.push_back(PackedFloat2{1, 1}); uvs.push_back(PackedFloat2{0, 1});
                materials.push_back(mat);
            }
        }
        prevRing = curRing;
        prevNormals = curNormals;
    }
}

// pbrt-v4's own FresnelMoment1() polynomial fit (src/shared/fresnel.h,
// ported directly - HOST-side only, since `eta` never varies per-hit
// for a NormalizedFresnelBxDF material, so `c = 1 - 2*FresnelMoment1(1/eta)`
// can be precomputed once here rather than needing a device-side port
// at all - see shadeNormalizedFresnel()'s own declaration comment,
// metal_poc.metal).
inline float fresnelMoment1(float eta) {
    const float eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta, eta5 = eta4 * eta;
    if (eta < 1.0f) {
        return 0.45966f - 1.73965f * eta + 3.37668f * eta2
             - 3.904945f * eta3 + 2.49277f * eta4 - 0.68441f * eta5;
    }
    return -4.61686f + 11.1136f * eta - 10.4646f * eta2
         + 5.11455f * eta3 - 1.27198f * eta4 + 0.12746f * eta5;
}



// CauchyCoefficientsFromAbbe (src/shared/fresnel.h) - construction-time
// only (not per-ray), ported here rather than #included directly since
// it's the one piece of that header genuinely CPU-only (no CPU_GPU tag
// needed at all - the per-ray CauchyEta() counterpart IS ported, as
// `cauchyEta()`, into metal_poc.metal itself, next to
// shadeDispersiveDielectric()'s own declaration).
inline void cauchyCoefficientsFromAbbe(double etaD, double abbeNumber, double& A, double& B) {
    constexpr double lambdaF = 0.4861, lambdaC = 0.6563, lambdaD = 0.5893;
    B = (etaD - 1.0) / (abbeNumber * (1.0 / (lambdaF * lambdaF) - 1.0 / (lambdaC * lambdaC)));
    A = etaD - B / (lambdaD * lambdaD);
}

