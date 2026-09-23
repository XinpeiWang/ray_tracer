// metal_poc_gpu_types.h
// GPU-buffer-facing data structs (Uniforms, per-light-kind data, material/
// geometry structs) plus the small host-side helper functions used to
// CONSTRUCT some of them (makeProjectionLight/makeGoniometricLight, the
// procedural goniometric profile image) - split out of metal_poc_app.h as
// a pure code-motion refactor (no behaviour change) once that file grew
// past ~2000 lines, the same "one giant translation unit"-adjacent problem
// PR #156/metal_poc.mm's own earlier splits already addressed - see
// docs/METAL_GPU_FEASIBILITY.md. Included by metal_poc_app.h itself, so
// every existing includer of THAT header keeps seeing these types with no
// changes needed at any call site.
#pragma once

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <simd/simd.h>

#include "metal_poc_host_math.h"
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
    // Row-band progress reporting: `compileShaderAndDispatch()` now
    // dispatches primaryRayKernel() once PER HORIZONTAL BAND of rows
    // instead of once for the whole image, so the host gets a real
    // checkpoint to print "Scanlines remaining: N" from between bands -
    // the exact same problem OptiX's own wavefront backend already
    // solved for the same reason (see that file's own comment,
    // gpu/optix/wavefront_path_tracer.cpp - "unlike the recursive
    // backend... no host-visible checkpoint to report from"). Each
    // dispatch's own [[thread_position_in_grid]] only ever ranges over
    // [0, bandHeight) - `rowOffset` (this band's own first real row) is
    // added to it ONCE, right at the top of the kernel, before anything
    // else reads `tid` - see primaryRayKernel()'s own comment. Appended
    // at the very end of this struct (not inserted among the existing
    // fields) so no other field's own byte offset shifts - the C++/MSL
    // mirrors only need to agree on ONE new field's placement, not be
    // re-verified against forty already-correct ones.
    uint32_t rowOffset = 0;
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

// Mirrors metal_poc_types.metal's GpuCloudMedium byte-for-byte (E2,
// section 178) - see that struct's own comment.
struct GpuCloudMedium {
    float boundsMin[3];
    float boundsMax[3];
    float worldToMediumMat[9];
    float worldToMediumTranslate[3];
    float sigmaA;
    float sigmaS;
    float density;
    float wispiness;
    float frequency;
};

// Mirrors metal_poc_types.metal's GpuRgbGridMedium byte-for-byte (E4,
// section 179) - see that struct's own comment.
struct GpuRgbGridMedium {
    float boundsMin[3];
    float boundsMax[3];
    float worldToMediumMat[9];
    float worldToMediumTranslate[3];
    int nx, ny, nz;
    int dataOffset;
    float sigmaScale;
    float sigmaMaj;
    float phaseG;
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
