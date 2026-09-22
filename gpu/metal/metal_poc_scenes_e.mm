// metal_poc_scenes_e.mm
// Category E (Volumes) hand-authored scene builders -
// MetalPocApp::buildXxx() out-of-class definitions, split out of
// metal_poc.mm as a pure code-motion refactor (no behaviour change) -
// see metal_poc_app.h's own header comment and docs/METAL_GPU_FEASIBILITY.md.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_poc_app.h"

void MetalPocApp::buildHomogeneousMediumScene() {
    using namespace cornell_box_data;
    const float3 bboxMin{0.0f, 0.0f, 0.0f};
    const float3 bboxMax{555.0f, 555.0f, 555.0f};
    const float sceneScale = 2.0f / 555.0f;
    const float3 bboxCenter = 0.5f * (bboxMin + bboxMax);
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};
    auto toWorld = [=](float3 p) { return (p - bboxCenter) * sceneScale + sceneOffset; };

    // All 6 walls, including the light this time (unlike
    // buildCornellNoLightWalls()'s own 5-only loop).
    for (const QuadSpec& q : kQuads) {
        const float3 Q{(float)q.Q.x, (float)q.Q.y, (float)q.Q.z};
        const float3 u{(float)q.u.x, (float)q.u.y, (float)q.u.z};
        const float3 v{(float)q.v.x, (float)q.v.y, (float)q.v.z};
        const float3 a = toWorld(Q), b = toWorld(Q + u), c = toWorld(Q + u + v), d = toWorld(Q + v);
        const float3 color{(float)q.color.r, (float)q.color.g, (float)q.color.b};
        if (q.is_light) {
            const int32_t lightId = (int32_t)lights.size();
            addQuad(verts, normals, uvs, materials, a, b, c, d, color,
                    /*materialType=*/0u, /*emission=*/color, lightId);
            const float3 edgeU = b - a, edgeV = d - a;
            const float3 normalV = simd::normalize(simd::cross(edgeU, edgeV));
            const float area = simd::length(simd::cross(edgeU, edgeV));
            const float3 center = a + 0.5f * edgeU + 0.5f * edgeV;
            lights.push_back(AreaLightData{
                PackedFloat3{center.x, center.y, center.z},
                PackedFloat3{edgeU.x, edgeU.y, edgeU.z},
                PackedFloat3{edgeV.x, edgeV.y, edgeV.z},
                PackedFloat3{normalV.x, normalV.y, normalV.z},
                area, PackedFloat3{color.x, color.y, color.z},
                /*patternTileB=*/0.0f, /*patternScale=*/0.0f,
                /*twoSided=*/0.0f, /*useTexture=*/0.0f});
        } else {
            addQuad(verts, normals, uvs, materials, a, b, c, d, color);
        }
    }

    // Real homogeneous fog - CPU's own constant_medium(boundary,
    // density=0.005, albedo=(0.8,0.9,1.0), g=0.3): `density` IS sigma_t
    // directly (constant_medium.h's own comment: "density is sigma_t...
    // we treat density as sigma_s only... albedo = albedo param" - the
    // passed colour is already a flat scattering albedo, not a per-
    // channel sigma_s needing back-division the way loadPbrtScene()'s
    // own real pbrt Medium parsing needs). sigma_t needs the SAME
    // `/sceneScale` compensation loadPbrtScene() already established
    // (section 108's own fix) - a real-world extinction coefficient
    // over a shorter simulated distance needs scaling UP to keep the
    // same physical optical depth.
    havePbrtMedium = true;
    // A real architectural mismatch found and worked around, not a
    // simple scale bug: this loader's own fog-sampling code has no
    // notion of a separate medium BOUNDARY at all - it samples along
    // whatever distance the CURRENT ray already travels to its next
    // real hit, unconditionally, the same convention the hardcoded POC
    // room's own always-camera-adjacent fog was designed for. CPU's own
    // Cornell-family camera (kCornellBoxCamera, lookfrom z=-800) sits
    // OUTSIDE the open-fronted box, so a primary ray here travels
    // through ~800 units of genuinely empty space in front of the room
    // before ever reaching its own real 555-unit interior - CPU's own
    // real medium has an EXPLICIT, LOCALIZED boundary box (inset 5
    // units from each wall) that correctly excludes that empty
    // approach segment; this loader's own "fog fills whatever the ray
    // hits" convention does not, so the straightforward `/sceneScale`
    // conversion (correct for the pbrt-loaded-scene case this formula
    // was originally derived for, section 108) applies the SAME
    // density over a MUCH LONGER effective path here, over-fogging the
    // room to near-total whiteout with a first, literal port. A
    // genuine architectural gap, not fixable by re-deriving a cleaner
    // formula - reconciled instead by an empirically-calibrated
    // correction factor (found by rendering and comparing against a
    // real --cpu reference directly, the same discipline this whole
    // series already uses for other non-portable numbers, e.g.
    // targetSize in section 118) rather than a first-principles value.
    pbrtFogSigmaT = 0.005f / sceneScale / 8.0f;
    pbrtFogAlbedo = float3{0.8f, 0.9f, 1.0f};
    pbrtFogAsymmetryG = 0.3f;

    // Camera - kCornellBoxCamera, same as every Cornell-family scene.
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
}

// E3: Dielectric Medium Showcase (section 177) - matches CPU's own
// build_dielectric_medium_scene() (scenes_advanced.h) in STRUCTURE: a
// ground sphere plus 3 glass spheres, each wrapping CPU's own
// `constant_medium` (a real participating medium) INSIDE a dielectric
// boundary (thin red mist / medium green haze / dense blue fog, in
// increasing optical density).
//
// Approximated here as TINTED GLASS (materialType 2, already real,
// working infrastructure - `shadeDielectric()`'s own existing Beer-
// Lambert absorption, `applyBeerLambertAbsorption()`, needs ZERO
// changes), not as a real internal scattering medium the way A8's own
// materialType 28 (section 176) is: combining THAT mechanism with
// real dielectric refraction would need a second Fresnel reflect/
// refract decision at the medium's own FAR boundary (exiting the
// glass, not just "continue unchanged" the way A8's own pass-through
// case does, since a dielectric bends light at both surfaces) plus
// its own new NEE/shadow-ray-occlusion handling for a shape that's
// simultaneously refractive AND scattering - a real, substantially
// bigger combined feature, not attempted here. Deliberately, honestly
// scoped down instead: `Material::color` (this materialType's own
// Beer-Lambert absorption coefficient) is derived from the medium's
// own `sigma_t * (1 - albedo)` - the ABSORBED (not scattered) fraction
// per channel, so a channel with HIGH albedo (mostly scattering, e.g.
// red mist's own 0.9 red channel) survives a straight pass THROUGH
// the glass almost unattenuated, and a LOW-albedo channel (mostly
// absorbed, that same sphere's own 0.2 green/blue) gets filtered out -
// producing correctly-COLOURED, correctly-DENSITY-varying glass (thin/
// misty to dense/opaque, matching each sphere's own real sigma_t), just
// without CPU's own real internal single-scattering GLOW (no light
// gets redirected sideways from inside the glass toward the camera the
// way real scattering would - this is honestly a tinted-glass look,
// not a lit-from-within haze).
void MetalPocApp::buildDielectricMediumShowcase() {
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};

    // Ground - CPU's own literal colour, radius-1000 sphere (the same
    // A2/A3/A8-established "huge sphere as ground plane" convention).
    {
        const float3 c = float3{0.0f, -1000.0f, 0.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 1000.0f});
        sphereMaterials.push_back(TriangleMaterial{PackedFloat3{0.4f, 0.5f, 0.3f}, /*materialType=*/0u,
            1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f});
    }

    // The 3 "fog" spheres - CPU's own exact x/albedo/sigmaT literals
    // (build_dielectric_medium_scene()'s own fog_sphere table), radius
    // 1.5, dielectric ior 1.5.
    struct FogSphere { float x; float3 albedo; float sigmaT; };
    const FogSphere kSpheres[3] = {
        {-4.0f, {0.9f, 0.2f, 0.2f}, 0.5f},  // thin red mist
        { 0.0f, {0.2f, 0.8f, 0.3f}, 1.5f},  // medium green haze
        { 4.0f, {0.3f, 0.4f, 0.9f}, 3.0f},  // dense blue fog
    };
    const float radius = 1.5f;
    for (const FogSphere& fs : kSpheres) {
        const float3 absorption = fs.sigmaT * (float3{1, 1, 1} - fs.albedo);
        const float3 c = float3{fs.x, radius, 0.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, radius});
        sphereMaterials.push_back(TriangleMaterial{PackedFloat3{absorption.x, absorption.y, absorption.z},
            /*materialType=*/2u, /*ior=*/1.5f, PackedFloat3{0, 0, 0}, -1, 0.0f});
    }

    // Background - CPU's own registry row for E3, bg (0.5,0.7,1.0).
    havePbrtConstantEnvLight = true;
    pbrtEnvColor = float3{0.5f, 0.7f, 1.0f};

    // Camera - CPU's own registry row for E3 (vfov 40, lookfrom
    // (0,3,18), lookat (0,1.5,0)).
    const float3 lookfrom = float3{0.0f, 3.0f, 18.0f} + sceneOffset;
    const float3 lookat = float3{0.0f, 1.5f, 0.0f} + sceneOffset;
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
    pbrtBboxCenter = float3{0.0f, 0.0f, 0.0f};
    pbrtSceneScale = 1.0f;
    pbrtSceneOffset = sceneOffset;
}

