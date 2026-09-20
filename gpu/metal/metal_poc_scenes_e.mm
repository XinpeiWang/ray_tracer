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
    const float3 sceneOffset{8.0f, 0.0f, 0.0f};
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

// B9: Cornell Crystal - matches CPU's own build_cornell_crystal()
// exactly: the box stays plain white Lambertian (unchanged from A1's
// own), the sphere is materialType 18 (NormalizedFresnelBxDF, IOR
// 1.5) - a genuinely NEW material for this whole series, not an
// existing one reused (see shadeNormalizedFresnel()'s own declaration
// comment, metal_poc.metal, for the full derivation).

