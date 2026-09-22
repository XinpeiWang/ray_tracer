// metal_poc_scenes_b.mm
// Category B (Materials) hand-authored scene builders -
// MetalPocApp::buildXxx() out-of-class definitions, split out of
// metal_poc.mm as a pure code-motion refactor (no behaviour change) -
// see metal_poc_app.h's own header comment and docs/METAL_GPU_FEASIBILITY.md.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_poc_app.h"

void MetalPocApp::buildCornellFamilyScene(
        uint32_t boxMaterialType, float3 boxColor, float boxRoughness,
        float3 boxConductorEta, float3 boxConductorK, float boxIor,
        uint32_t sphereMaterialType, float3 sphereColor, float sphereRoughness,
        float3 sphereConductorEta, float3 sphereConductorK, float sphereIor,
        float3 sphereTransmitColor) {
    using namespace cornell_box_data;
    // Same rescale/recentre/offset convention buildCornellBoxA1() uses -
    // see that function's own comment.
    const float3 bboxMin{0.0f, 0.0f, 0.0f};
    const float3 bboxMax{555.0f, 555.0f, 555.0f};
    const float maxExtent = 555.0f;
    const float sceneScale = 2.0f / maxExtent;
    const float3 bboxCenter = 0.5f * (bboxMin + bboxMax);
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};
    auto toWorld = [=](float3 p) { return (p - bboxCenter) * sceneScale + sceneOffset; };

    // The 5 walls + the main ceiling light - identical to
    // buildCornellBoxA1()'s own loop.
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

    // The rotated box - same 6-face construction as buildCornellBoxA1(),
    // but using the CALLER's own box material instead of always white
    // Lambertian.
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
        struct Face { float3 Q, u, v; };
        const Face faces[6] = {
            {float3{minC.x, minC.y, maxC.z},  dx,  dy},  // front
            {float3{maxC.x, minC.y, maxC.z}, -dz,  dy},  // right
            {float3{maxC.x, minC.y, minC.z}, -dx,  dy},  // back
            {float3{minC.x, minC.y, minC.z},  dz,  dy},  // left
            {float3{minC.x, maxC.y, maxC.z},  dx, -dz},  // top
            {float3{minC.x, minC.y, minC.z},  dx,  dz},  // bottom
        };
        // materialType == 2 (dielectric)/5 (rough dielectric): boxIor is
        // a real refraction index, `boxRoughness` is meaningless for 2,
        // meaningful for 5. materialType == 4: boxIor is UNUSED here
        // (addQuad() itself derives alphaX==alphaY==boxRoughness
        // automatically, see that function's own comment) - passed
        // through anyway for a uniform call shape.
        for (const Face& f : faces) {
            const float3 a = toWorld(rotateTranslate(f.Q));
            const float3 b = toWorld(rotateTranslate(f.Q + f.u));
            const float3 c = toWorld(rotateTranslate(f.Q + f.u + f.v));
            const float3 d = toWorld(rotateTranslate(f.Q + f.v));
            addQuad(verts, normals, uvs, materials, a, b, c, d, boxColor,
                    boxMaterialType, /*emission=*/simd::make_float3(0, 0, 0),
                    /*lightId=*/-1, boxRoughness, boxIor,
                    /*transmitColor=*/simd::make_float3(0, 0, 0), /*twoSided=*/false,
                    boxConductorEta, boxConductorK);
        }
    }

    // The sphere, same position/radius as A1's own glass sphere
    // (kGlassSphere - only its material varies per caller; the geometry
    // itself is the one constant every Cornell-family scene shares).
    {
        const float3 center = toWorld(float3{(float)kGlassSphere.center.x,
            (float)kGlassSphere.center.y, (float)kGlassSphere.center.z});
        const float radius = sceneScale * (float)kGlassSphere.radius;
        TriangleMaterial mat{PackedFloat3{sphereColor.x, sphereColor.y, sphereColor.z},
            sphereMaterialType, /*ior=*/1.0f, PackedFloat3{0, 0, 0}, /*lightId=*/-1, sphereRoughness};
        if (sphereMaterialType == 4u) {
            mat.ior = sphereRoughness;  // alphaX == alphaY (isotropic) - see addQuad()'s own comment
            mat.conductorEta = PackedFloat3{sphereConductorEta.x, sphereConductorEta.y, sphereConductorEta.z};
            mat.conductorK = PackedFloat3{sphereConductorK.x, sphereConductorK.y, sphereConductorK.z};
        } else if (sphereMaterialType == 2u || sphereMaterialType == 5u) {
            // materialType 2 (smooth dielectric): `roughness` unread,
            // `ior` is the real refraction index. materialType 5 (rough/
            // frosted dielectric, section 128): BOTH matter - `roughness`
            // (already set via the constructor above) drives alpha
            // (shadeRoughDielectric()'s own `mat.roughness^2`), `ior` is
            // still the real refraction index, read independently - the
            // two fields aren't a dual-use pair here the way materialType
            // 4's ior/roughness are.
            mat.ior = sphereIor;
        } else if (sphereMaterialType == 12u) {
            // Diffuse transmission (materialType 12, already implemented
            // long before this Phase-B epic - see this file's own
            // materialType==12 shader comment) - `color` (already set
            // from sphereColor above) is the REFLECTED diffuse tint,
            // `transmitColor` the TRANSMITTED one - matches CPU's own
            // `diffuse_transmission(R, T)` constructor exactly, two
            // independently-authored colours, not a derived pair.
            mat.transmitColor = PackedFloat3{sphereTransmitColor.x, sphereTransmitColor.y, sphereTransmitColor.z};
        } else if (sphereMaterialType == 18u) {
            // NormalizedFresnelBxDF ("crystal" sphere, section 139) -
            // `ior` is the real surface eta (sphereRoughness/sphereIor
            // params are unused for this material; `sphereIor` doubles
            // as eta here since `ior` is this field's own natural
            // meaning already). `roughness` holds the precomputed
            // energy-renormalization constant `c`, computed HOST-side -
            // see fresnelMoment1()'s own declaration comment.
            mat.ior = sphereIor;
            mat.roughness = 1.0f - 2.0f * fresnelMoment1(1.0f / sphereIor);
        } else if (sphereMaterialType == 19u) {
            // CoatedDiffuseBxDF ("coated diffuse" sphere, section 140) -
            // `ior` is the dielectric coat's real refraction index
            // (`sphereIor`); `roughness` (already set from
            // `sphereRoughness` via the constructor above) is the
            // PRECOMPUTED GGX alpha (`RoughnessToAlpha(userRoughness) =
            // sqrt(userRoughness)`, computed HOST-side by the caller -
            // matches CPU's own `coated_diffuse` constructor exactly,
            // NOT this file's own materialType 4/9 "store roughness,
            // square it in the shader" convention, since this is a
            // brand-new shader function with no old convention to stay
            // consistent with).
            mat.ior = sphereIor;
        } else if (sphereMaterialType == 20u) {
            // CoatedConductorBxDF ("lacquered metal" sphere, section 141) -
            // `ior` is the dielectric coat's real refraction index
            // (`sphereIor`); `roughness` (already set from
            // `sphereRoughness` via the constructor above) is the
            // precomputed GGX alpha, same convention as materialType
            // 19's own sphere branch just above; `conductorEta`/
            // `conductorK` are the metal base's own real per-channel
            // complex IOR - unlike materialType 4's own branch, the
            // constructor above does NOT set these two fields by
            // default, so they need the same explicit assignment here.
            mat.ior = sphereIor;
            mat.conductorEta = PackedFloat3{sphereConductorEta.x, sphereConductorEta.y, sphereConductorEta.z};
            mat.conductorK = PackedFloat3{sphereConductorK.x, sphereConductorK.y, sphereConductorK.z};
        } else if (sphereMaterialType == 21u) {
            // Checker-driven normal map (B12's own sphere, section 142) -
            // `roughness` reused as the checker's own world-space cell
            // size, matching materialType 16's own established reuse of
            // the same field - CPU's own real cell size (8.0, in the
            // pbrt-scale 0-555 coordinate system) needs the SAME
            // `sceneScale` conversion every world-space distance already
            // gets in this function (an earlier version of this scene
            // hardcoded 8.0 directly in the SHADER instead, in the
            // RESCALED ~2-unit coordinate space - one giant cell
            // covering the whole sphere, a real bug caught by comparing
            // directly against a --cpu render showing a visibly dimpled
            // sphere against this version's own perfectly smooth one).
            mat.roughness = 8.0f * sceneScale;
        }
        spheres.push_back(SphereData{PackedFloat3{center.x, center.y, center.z}, radius});
        sphereMaterials.push_back(mat);
    }

    // Camera - identical to buildCornellBoxA1()'s own (every Cornell-
    // family scene shares kCornellBoxCamera, scene_registry.h).
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

// B2: Cornell Rough Metal - matches CPU's own build_cornell_rough_metal()
// exactly: a rough-aluminium box (roughness 0.15) and a rough-gold
// sphere (roughness 0.3), both materialType 4 (GGX conductor).
// conductorEta/K derived from the flat albedo via reflectanceToConductorK()
// (section 103's own already-shipped formula) - CPU's own `rough_metal`
// material is itself a flat-albedo Schlick-style approximation, not a
// real measured-spectrum conductor, so this is a faithful MATERIAL
// match, not a downgrade.
//
// The literal 0.15/0.3 "roughness" NUMBERS are NOT ported as-is,
// though - a real convention mismatch, found by comparing a first
// literal-port render directly against a real --cpu render of the same
// scene_id (much shinier/more mirror-like than CPU's own visibly
// matte-ish box/sphere). CPU's own `rough_metal` maps roughness->alpha
// via pbrt-v4's real `RoughnessToAlpha()` (src/shared/microfacet.h) -
// alpha = sqrt(roughness). This POC's own materialType 4 (shadeConductor(),
// metal_poc.metal) instead SQUARES the stored value - alpha = roughness^2
// - an already-established, already-shipped convention (used by every
// prior conductor material this whole series has ever added, not
// something to special-case away just for this one scene). To make
// this scene's own APPARENT roughness match CPU's real alpha despite
// the two backends using genuinely different roughness->alpha curves,
// the value passed here is `bookRoughness^0.25` (so that, after this
// POC's own squaring, the net alpha equals `sqrt(bookRoughness)` -
// exactly CPU's own real alpha) - not an arbitrary fudge, the exact
// algebraic value needed to reconcile the two curves. Section 126,
// docs/METAL_GPU_FEASIBILITY.md.
void MetalPocApp::buildCornellRoughMetal() {
    const float3 alum{0.8f, 0.85f, 0.88f};
    const float3 gold{0.95f, 0.78f, 0.28f};
    const float boxAlpha = powf(0.15f, 0.25f);    // ~0.622 - see this function's own comment
    const float sphereAlpha = powf(0.3f, 0.25f);  // ~0.740
    buildCornellFamilyScene(
        /*box=*/4u, alum, boxAlpha, float3{1, 1, 1}, reflectanceToConductorK(alum), 1.0f,
        /*sphere=*/4u, gold, sphereAlpha, float3{1, 1, 1}, reflectanceToConductorK(gold), 1.0f);
}

// B4: Cornell Conductor - matches CPU's own build_cornell_conductor()
// exactly: a polished gold sphere (roughness 0.1) and a polished
// aluminium box (roughness 0.05), both materialType 4 with REAL
// measured eta/k (src/shared/conductor_data.h's kConductorAu/
// kConductorAl - same literal values the hardcoded room's own gold
// accent sphere already uses, section 61) instead of B2's own flat-
// albedo reflectanceToConductorK() approximation. Same `roughness^0.25`
// conversion B2's own comment already derived - CPU's `conductor`
// material class uses the SAME pbrt-v4 RoughnessToAlpha()/sqrt(roughness)
// convention `rough_metal` does (both call the same helper,
// material_pbrt.h), so the identical reconciliation applies.
void MetalPocApp::buildCornellConductor() {
    const float3 goldEta{0.184f, 0.457f, 1.354f}, goldK{3.070f, 2.408f, 1.818f};
    const float3 alumEta{1.357f, 0.884f, 0.669f}, alumK{7.588f, 6.470f, 5.690f};
    const float boxAlpha = powf(0.05f, 0.25f);    // ~0.473
    const float sphereAlpha = powf(0.1f, 0.25f);  // ~0.562
    buildCornellFamilyScene(
        /*box=*/4u, float3{1, 1, 1}, boxAlpha, alumEta, alumK, 1.0f,
        /*sphere=*/4u, float3{1, 1, 1}, sphereAlpha, goldEta, goldK, 1.0f);
}

// B3: Cornell Rough Glass - matches CPU's own build_cornell_rough_glass()
// exactly: the box stays plain white Lambertian (unchanged from A1's
// own), only the sphere changes - materialType 5 (rough/frosted
// dielectric), ior 1.5, roughness converted the SAME `^0.25` way
// sections 126/127 already established (CPU's own `rough_dielectric`
// class calls the identical `RoughnessToAlpha()` helper `rough_metal`/
// `conductor` do - checked directly, not assumed, before reusing the
// conversion here).
void MetalPocApp::buildCornellRoughGlass() {
    const float3 white{0.73f, 0.73f, 0.73f};
    const float sphereAlpha = powf(0.2f, 0.25f);  // ~0.669
    buildCornellFamilyScene(
        /*box=*/0u, white, 0.0f, float3{1, 1, 1}, float3{0, 0, 0}, 1.0f,
        /*sphere=*/5u, float3{1, 1, 1}, sphereAlpha, float3{1, 1, 1}, float3{0, 0, 0}, /*ior=*/1.5f);
}

// B6: Cornell Thin Glass - matches CPU's own build_cornell_thin_glass()
// exactly. NOT buildCornellFamilyScene() (see this method's own
// declaration comment for why): the 5 walls are still
// cornell_box_data::kQuads[0..4] (identical numbers to CPU's own hand-
// listed ones here, confirmed directly, not assumed) and the box is
// still kBox, but the ceiling light and the extra thin-glass panel are
// this scene's own.
void MetalPocApp::buildCornellThinGlass() {
    using namespace cornell_box_data;
    const float3 bboxMin{0.0f, 0.0f, 0.0f};
    const float3 bboxMax{555.0f, 555.0f, 555.0f};
    const float sceneScale = 2.0f / 555.0f;
    const float3 bboxCenter = 0.5f * (bboxMin + bboxMax);
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};
    auto toWorld = [=](float3 p) { return (p - bboxCenter) * sceneScale + sceneOffset; };

    // The 5 walls only (kQuads[5], the standard ceiling light, is
    // skipped - this scene's own light is a different size/position).
    for (int i = 0; i < 5; ++i) {
        const QuadSpec& q = kQuads[i];
        const float3 Q{(float)q.Q.x, (float)q.Q.y, (float)q.Q.z};
        const float3 u{(float)q.u.x, (float)q.u.y, (float)q.u.z};
        const float3 v{(float)q.v.x, (float)q.v.y, (float)q.v.z};
        const float3 color{(float)q.color.r, (float)q.color.g, (float)q.color.b};
        addQuad(verts, normals, uvs, materials, toWorld(Q), toWorld(Q + u),
                toWorld(Q + u + v), toWorld(Q + v), color);
    }

    // This scene's own ceiling light - smaller, off-centre, brighter
    // (15,15,15) than the standard kQuads[5] (7,7,7).
    {
        const float3 Q{213.0f, 554.0f, 227.0f}, u{130.0f, 0.0f, 0.0f}, v{0.0f, 0.0f, 105.0f};
        const float3 a = toWorld(Q), b = toWorld(Q + u), c = toWorld(Q + u + v), d = toWorld(Q + v);
        const float3 lightColor{15.0f, 15.0f, 15.0f};
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
            /*twoSided=*/0.0f, /*useTexture=*/0.0f});
    }

    // The rotated white box - identical to buildCornellBoxA1()'s own.
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
            {float3{minC.x, minC.y, maxC.z},  dx,  dy},
            {float3{maxC.x, minC.y, maxC.z}, -dz,  dy},
            {float3{maxC.x, minC.y, minC.z}, -dx,  dy},
            {float3{minC.x, minC.y, minC.z},  dz,  dy},
            {float3{minC.x, maxC.y, maxC.z},  dx, -dz},
            {float3{minC.x, minC.y, minC.z},  dx,  dz},
        };
        for (const Face& f : faces) {
            addQuad(verts, normals, uvs, materials,
                    toWorld(rotateTranslate(f.Q)), toWorld(rotateTranslate(f.Q + f.u)),
                    toWorld(rotateTranslate(f.Q + f.u + f.v)), toWorld(rotateTranslate(f.Q + f.v)),
                    boxColor);
        }
    }

    // The thin-glass panel (materialType 11, ior 1.5) - built centred at
    // the local origin, rotated 62 degrees about Y (the SAME rotate_y
    // forward-transform formula as the box above - hittable.h's own
    // convention), then translated into place. Matches CPU's own
    // panel_quad construction exactly, including the 62-degree tilt
    // CPU's own comment explains is needed for the thin-film sheen to
    // actually read at IOR 1.5 (near-zero Fresnel reflectance at normal
    // incidence).
    {
        const float3 localQ{-177.5f, -277.5f, 0.0f}, localU{0.0f, 555.0f, 0.0f}, localV{355.0f, 0.0f, 0.0f};
        const float theta = 62.0f * (float)M_PI / 180.0f;
        const float sinT = sinf(theta), cosT = cosf(theta);
        const float3 panelTranslate{277.5f, 277.5f, 200.0f};
        auto rotateTranslate = [=](float3 p) -> float3 {
            const float newX = cosT * p.x + sinT * p.z;
            const float newZ = -sinT * p.x + cosT * p.z;
            return float3{newX, p.y, newZ} + panelTranslate;
        };
        const float3 a = toWorld(rotateTranslate(localQ));
        const float3 b = toWorld(rotateTranslate(localQ + localU));
        const float3 c = toWorld(rotateTranslate(localQ + localU + localV));
        const float3 d = toWorld(rotateTranslate(localQ + localV));
        addQuad(verts, normals, uvs, materials, a, b, c, d, float3{1, 1, 1},
                /*materialType=*/11u, /*emission=*/simd::make_float3(0, 0, 0),
                /*lightId=*/-1, /*roughness=*/0.0f, /*ior=*/1.5f);
    }

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

// B1: Rough Metal Spheres - matches CPU's own build_rough_metal_spheres()
// exactly: 5 GGX conductor spheres (roughness 0.05/0.2/0.4/0.6/0.8, warm
// gold-ish albedo) in a row over a ground plane, lit by one real NEE-
// sampled quad light. Ground is a flat quad, not CPU's own radius-1000
// sphere - the SAME clearance issue A5's own ground had (section 124),
// same fix. Roughness values go through the same `^0.25` conversion
// sections 126-128 already established (CPU's own `rough_metal` class,
// same as B2's).
void MetalPocApp::buildRoughMetalSpheres() {
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};

    // Ground - flat quad, dark grey Lambertian.
    {
        const float3 groundColor{0.2f, 0.2f, 0.2f};
        addQuad(verts, normals, uvs, materials,
                float3{-15.0f, 0.0f, -15.0f} + sceneOffset, float3{15.0f, 0.0f, -15.0f} + sceneOffset,
                float3{15.0f, 0.0f, 15.0f} + sceneOffset, float3{-15.0f, 0.0f, 15.0f} + sceneOffset,
                groundColor);
    }

    // 5 rough-metal spheres, roughness increasing left to right.
    {
        const float3 albedo{0.95f, 0.85f, 0.55f};
        const float3 eta{1, 1, 1};
        const float3 k = reflectanceToConductorK(albedo);
        const float roughnesses[5] = {0.05f, 0.2f, 0.4f, 0.6f, 0.8f};
        for (int i = 0; i < 5; ++i) {
            const float x = (float)(i - 2) * 2.5f;
            const float alpha = powf(roughnesses[i], 0.25f);  // see this function's own comment
            TriangleMaterial mat{PackedFloat3{albedo.x, albedo.y, albedo.z}, /*materialType=*/4u,
                                  /*ior=*/alpha, PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/alpha};
            mat.conductorEta = PackedFloat3{eta.x, eta.y, eta.z};
            mat.conductorK = PackedFloat3{k.x, k.y, k.z};
            const float3 c = float3{x, 1.0f, 0.0f} + sceneOffset;
            spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 1.0f});
            sphereMaterials.push_back(mat);
        }
    }

    // Real NEE-sampled area light.
    {
        const float3 Q{-6.0f, 6.0f, -4.0f}, u{12.0f, 0.0f, 0.0f}, v{0.0f, 0.0f, 8.0f};
        const float3 a = Q + sceneOffset, b = Q + u + sceneOffset,
                     c = Q + u + v + sceneOffset, d = Q + v + sceneOffset;
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
            /*twoSided=*/0.0f, /*useTexture=*/0.0f});
    }

    // Real per-scene flat background (dark grey) - same reuse of the
    // pbrt-constant-infinite-light mechanism every earlier open (non-
    // Cornell-box) hand-authored scene already established.
    havePbrtConstantEnvLight = true;
    pbrtEnvColor = float3{0.10f, 0.10f, 0.12f};

    // Camera: CPU's own real registry row for B1, ported directly
    // (vfov 42, lookfrom (0,2.7,17), lookat (0,1,0)).
    const float3 lookfrom = float3{0.0f, 2.7f, 17.0f} + sceneOffset;
    const float3 lookat = float3{0.0f, 1.0f, 0.0f} + sceneOffset;
    const float3 up{0.0f, 1.0f, 0.0f};
    const float3 forward = simd::normalize(lookat - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    pbrtTanHalfFov = tanf(0.5f * 42.0f * (float)M_PI / 180.0f);
    havePbrtCamera = true;
    pbrtCameraLookAtWorld = lookat;
    pbrtCameraUpRaw = up;
    pbrtBboxCenter = float3{0.0f, 0.0f, 0.0f};
    pbrtSceneScale = 1.0f;
    pbrtSceneOffset = sceneOffset;
}

// B8: Cornell Wax Slab - matches CPU's own build_cornell_wax_slab()
// exactly: box unchanged (white Lambertian), sphere is materialType 12
// (diffuse transmission) - warm ivory reflectance, warm amber
// transmittance, more transmittance than reflectance (a real wax-like
// translucency).
void MetalPocApp::buildCornellWaxSlab() {
    const float3 white{0.73f, 0.73f, 0.73f};
    const float3 reflectR{0.6f, 0.5f, 0.3f};
    const float3 transmitT{0.8f, 0.6f, 0.3f};
    buildCornellFamilyScene(
        /*box=*/0u, white, 0.0f, float3{1, 1, 1}, float3{0, 0, 0}, 1.0f,
        /*sphere=*/12u, reflectR, 0.0f, float3{1, 1, 1}, float3{0, 0, 0}, /*ior=*/1.0f,
        /*sphereTransmitColor=*/transmitT);
}

// I8: Light Sampler Comparison - matches CPU's own
// build_light_sampler_comparison() exactly: the same A1 Cornell shell
// (walls/box/glass sphere), but 5 quad lights of deliberately lopsided
// power (~1:2:6:15:80) instead of the usual single ceiling light - all
// 5 real NEE-sampled AreaLights (this loader's own light picking is
// already power-proportional, section 52 - a real, if incidental,
// architectural match to CPU's own "power"/"bvh" light-sampler choices
// this scene exists to contrast against "uniform").

void MetalPocApp::buildCornellCrystal() {
    const float3 white{0.73f, 0.73f, 0.73f};
    buildCornellFamilyScene(
        /*box=*/0u, white, 0.0f, float3{1, 1, 1}, float3{0, 0, 0}, 1.0f,
        /*sphere=*/18u, float3{1, 1, 1}, 0.0f, float3{1, 1, 1}, float3{0, 0, 0}, /*ior=*/1.5f);
}

// B5: Cornell Coated Diffuse - matches CPU's own
// build_cornell_coated_diffuse() exactly: an orange/terracotta coated-
// diffuse BOX (IOR 1.5, roughness 0.2) and a blue coated-diffuse SPHERE
// (IOR 1.5, roughness 0.1), both materialType 19 (CoatedDiffuseBxDF - a
// rough dielectric coat over a Lambertian base, pbrt-v4's own real
// stochastic layered-material random walk, ported from the ALREADY-
// SHIPPED, ALREADY-VERIFIED OptiX GPU reference - gpu/optix/
// optix_device_helpers.h's own MaterialType::CoatedDiffuse case - rather
// than re-derived from CPU's own src/shared/bxdfs_layered.h from
// scratch. See shadeCoatedDiffuse()'s own declaration comment,
// metal_poc.metal, for the full derivation and the one deliberate place
// this port's own Fresnel convention differs between its sample step and
// its NEE/MIS f() step, matching OptiX's own real (not unified) two-path
// behavior exactly rather than "fixing" an inconsistency that isn't a
// bug.
//
// `RoughnessToAlpha(r) = sqrt(max(r, 1e-4))` (pbrt-v4's own real
// roughness->alpha remap, `TrowbridgeReitz::RoughnessToAlpha` -
// src/shared/microfacet.h) is computed HOST-side here, matching BOTH
// CPU's own `coated_diffuse` constructor AND OptiX's own
// `mat.remapRoughness ? sqrtf(mat.fuzz) : mat.fuzz` line exactly - this
// is a genuinely NEW shader function with no old "square it in the
// shader" convention (materialType 4/9's own) to reconcile with, unlike
// every earlier GGX-conductor scene in this series.
void MetalPocApp::buildCornellCoatedDiffuse() {
    const float3 blue{0.2f, 0.3f, 0.9f};
    const float3 orange{0.75f, 0.35f, 0.1f};
    const float boxAlpha = sqrtf(std::max(0.2f, 1e-4f));
    const float sphereAlpha = sqrtf(std::max(0.1f, 1e-4f));
    buildCornellFamilyScene(
        /*box=*/19u, orange, boxAlpha, float3{1, 1, 1}, float3{0, 0, 0}, /*boxIor=*/1.5f,
        /*sphere=*/19u, blue, sphereAlpha, float3{1, 1, 1}, float3{0, 0, 0}, /*sphereIor=*/1.5f);
}

// B7: Cornell Coated Conductor - matches CPU's own
// build_cornell_coated_conductor() exactly: a lacquered-COPPER box
// (Cu conductor, IOR-1.5 coat, roughness 0.2) and a lacquered-GOLD
// sphere (Au conductor, IOR-1.5 coat, roughness 0.1), both materialType
// 20 (CoatedConductorBxDF) - the SAME rough-dielectric-coat random walk
// materialType 19 just built, with a GGX-conductor bottom bounce
// (complex per-channel Fresnel, `kConductorAu`/`kConductorCu` - the
// SAME real presets materialType 4/9 already use elsewhere in this
// series) in place of a Lambertian one. `color` is unused for this
// achromatic-base material (the metal's own colour comes entirely from
// `conductorEta`/`conductorK`, not a flat tint) - set to white purely
// so an accidental future read doesn't silently multiply by black.
void MetalPocApp::buildCornellCoatedConductor() {
    const float boxAlpha = sqrtf(std::max(0.2f, 1e-4f));
    const float sphereAlpha = sqrtf(std::max(0.1f, 1e-4f));
    const float3 cuEta{kConductorCu.eta_r, kConductorCu.eta_g, kConductorCu.eta_b};
    const float3 cuK{kConductorCu.k_r, kConductorCu.k_g, kConductorCu.k_b};
    const float3 auEta{kConductorAu.eta_r, kConductorAu.eta_g, kConductorAu.eta_b};
    const float3 auK{kConductorAu.k_r, kConductorAu.k_g, kConductorAu.k_b};
    buildCornellFamilyScene(
        /*box=*/20u, float3{1, 1, 1}, boxAlpha, cuEta, cuK, /*boxIor=*/1.5f,
        /*sphere=*/20u, float3{1, 1, 1}, sphereAlpha, auEta, auK, /*sphereIor=*/1.5f);
}

// B12: Normal Mapped Cornell - matches CPU's own
// build_normal_mapped_cornell() geometry exactly (same box/sphere
// position as every other Cornell-family scene - kBox/kGlassSphere),
// but its own materials need a real investigation, not a literal port:
// CPU's own `bump_map_material` (the back wall AND the rotated box)
// wraps a `noise_texture`, whose `value(u,v,p)` (src/TheRestOfYourLife/
// texture.h) ignores `u`/`v` ENTIRELY and reads only `p` - but
// `bump_map_material::apply()` (normal_map_materials.h) samples that
// SAME texture at `(rec.u, rec.v, rec.p)`, `(rec.u+step, rec.v,
// rec.p)`, and `(rec.u, rec.v+step, rec.p)` - THE SAME `rec.p` every
// time, since only `rec.u`/`rec.v` change and the texture never reads
// them. All three displacement samples are therefore IDENTICAL, the
// finite-difference gradient is exactly zero, and `apply_bump_map()`
// returns the ORIGINAL, unperturbed geometric normal - a real,
// verified (not assumed) pre-existing CPU bug that makes this whole
// bump-map effect a complete no-op. Confirmed by rendering B12 with
// `--cpu` and inspecting the box/back-wall surface directly: perfectly
// flat, no visible marble waviness at all, unlike the sphere (which
// DOES show a real, correct effect - see below). The box therefore
// stays plain materialType 0 here, matching CPU's own ACTUAL (buggy
// but real) rendered output rather than what the scene's own name
// implies it should look like - the same "match the reference's real
// behavior, not its stated intent" discipline this whole series always
// follows.
//
// The SPHERE'S own `normal_map_material` has no such bug: it directly
// DECODES a texture's own RGB value as a tangent-space normal (no
// finite difference at all), and its own `checker_texture` (SPATIAL,
// 3D `p`-based, same convention materialType 16 already uses in this
// loader) genuinely varies from point to point regardless of `u`/`v` -
// a real, working effect, confirmed visually (a fine dimpled
// checkerboard-of-cubes pattern) in the same `--cpu` reference render.
// Wired as materialType 21 (metal_poc.metal's own materialType==21u
// normal-perturbation branch, inserted alongside materialType 7's own
// bump-map branch, then falling through to the ALREADY-EXISTING plain
// Lambertian shading path - no new "shadeXxx" function needed at all,
// since this material has no BSDF of its own, only a perturbed input
// normal feeding one that already exists).
void MetalPocApp::buildNormalMappedCornell() {
    const float3 white{0.73f, 0.73f, 0.73f};
    const float3 blueBase{0.2f, 0.3f, 0.8f};
    buildCornellFamilyScene(
        /*box=*/0u, white, 0.0f, float3{1, 1, 1}, float3{0, 0, 0}, 1.0f,
        /*sphere=*/21u, blueBase, 0.0f, float3{1, 1, 1}, float3{0, 0, 0}, 1.0f);
}

// B23: Glass Prism Dispersion - matches CPU's own
// build_prism_dispersion_geometry() exactly: a real triangular prism
// (cross-section A(0,0,0)/B(0,0,140)/C(0,121,70) in the Y-Z plane,
// extruded +150 along X - same hand-derived outward-normal winding as
// the CPU reference's own comment documents) under a single directional
// light (`build_prism_dispersion_punct()`), splitting into a chromatic
// fan on a large catcher screen. NOT a Cornell-family scene - its own
// bespoke bounding box/camera/light, since kCornellBoxCamera's own
// 555-unit framing doesn't apply here at all (see kPrismCamera,
// scene_registry.h).
//
// materialType 22 (recursive-backend dispersive dielectric - see
// shadeDispersiveDielectric()'s own declaration comment,
// metal_poc.metal, for the full "stochastic RGB-channel selection"
// mechanism, ported from OptiX's own identical recursive-backend
// approximation, NOT CPU's/wavefront's real continuous spectral
// integration - matches this scene's own registry description exactly:
// "GPU-recursive (--gpu, no --wavefront): a simplified 3-representative-
// wavelength RGB-channel approximation") or 23 (its frosted sibling,
// shadeDispersiveRoughDielectric() - B24, section 144). `conductorEta.x/y`
// carries the precomputed Cauchy (A, B) pair - `CauchyCoefficientsFromAbbe(
// 1.52, 59.0)`, crown glass, matching CPU's own `dielectric::
// make_dispersive(1.52, 59.0)`/`rough_dielectric::make_dispersive(1.52,
// 59.0, roughness)` exactly. Shared by both build_prism_dispersion()
// (B23) and build_prism_dispersion_rough() (B24) on the CPU side - same
// geometry/camera/light, only the glass material differs - so this one
// function builds both, parameterized on `glassMaterialType`/`roughness`.
void MetalPocApp::buildPrismDispersionGeometry(uint32_t glassMaterialType, float roughness) {
    // Bounding box covers the prism (X:[0,150], Y:[0,121], Z:[0,140]) and
    // the catcher screen (X:[-300,300], Y:[-300,400], Z:600) - NOT the
    // camera position, same "geometry only" convention
    // buildCornellFamilyScene() already uses (the camera goes through
    // the SAME toWorld() transform separately, even though it sits
    // outside this box).
    const float3 bboxMin{-300.0f, -300.0f, 0.0f};
    const float3 bboxMax{300.0f, 400.0f, 600.0f};
    const float maxExtent = 700.0f;  // 600(x) vs 700(y) vs 600(z) - see bbox above
    const float sceneScale = 2.0f / maxExtent;
    const float3 bboxCenter = 0.5f * (bboxMin + bboxMax);
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};
    auto toWorld = [=](float3 p) { return (p - bboxCenter) * sceneScale + sceneOffset; };

    double cauchyA, cauchyB;
    cauchyCoefficientsFromAbbe(1.52, 59.0, cauchyA, cauchyB);
    const float3 glassColor{1.0f, 1.0f, 1.0f};  // untinted - no transmission_filter in the CPU reference
    TriangleMaterial glassMat{PackedFloat3{glassColor.x, glassColor.y, glassColor.z},
        glassMaterialType, /*ior=*/1.52f, PackedFloat3{0, 0, 0}, /*lightId=*/-1, roughness};
    glassMat.conductorEta = PackedFloat3{(float)cauchyA, (float)cauchyB, 0.0f};
    glassMat.conductorK = PackedFloat3{0, 0, 0};

    // 3 rectangular sides, same outward-normal winding as CPU's own
    // build_prism_dispersion_geometry() comment documents (u/v order is
    // (depth, edge), not (edge, depth) - this cross-section's apex-up
    // orientation is in Y-Z, not X-Y).
    {
        const float3 A{0, 0, 0}, B{0, 0, 140}, C{0, 121, 70};
        const float3 depth{150, 0, 0};
        auto addPrismQuad = [&](float3 q, float3 u, float3 v) {
            addQuad(verts, normals, uvs, materials,
                    toWorld(q), toWorld(q + u), toWorld(q + u + v), toWorld(q + v),
                    glassColor, glassMaterialType, /*emission=*/simd::make_float3(0, 0, 0),
                    /*lightId=*/-1, roughness, /*ior=*/1.52f,
                    /*transmitColor=*/simd::make_float3(0, 0, 0), /*twoSided=*/false,
                    float3{(float)cauchyA, (float)cauchyB, 0.0f}, float3{0, 0, 0});
        };
        addPrismQuad(A, depth, B - A);  // base (z=0..140 side, y=0)
        addPrismQuad(B, depth, C - B);  // exit slant (toward +z)
        addPrismQuad(C, depth, A - C);  // entry slant (toward -z)

        // 2 triangular end caps (raw push, same pattern
        // buildTriangleMeshScene()'s own icosahedron uses).
        const float3 Aw = toWorld(A), Bw = toWorld(B), Cw = toWorld(C);
        const float3 Adw = toWorld(A + depth), Bdw = toWorld(B + depth), Cdw = toWorld(C + depth);
        auto pushTri = [&](float3 a, float3 b, float3 c) {
            const float3 faceNormal = simd::normalize(simd::cross(b - a, c - a));
            const PackedFloat3 packedNormal{faceNormal.x, faceNormal.y, faceNormal.z};
            verts.push_back(PackedFloat3{a.x, a.y, a.z});
            verts.push_back(PackedFloat3{b.x, b.y, b.z});
            verts.push_back(PackedFloat3{c.x, c.y, c.z});
            normals.push_back(packedNormal);
            normals.push_back(packedNormal);
            normals.push_back(packedNormal);
            uvs.push_back(PackedFloat2{0, 0});
            uvs.push_back(PackedFloat2{1, 0});
            uvs.push_back(PackedFloat2{0, 1});
            materials.push_back(glassMat);
        };
        // x=0 cap: forward winding (A,B,C) -> outward -X. x=150 cap:
        // reversed winding (A',C',B') -> outward +X - matches CPU's own
        // mesh_data->indices = {0,1,2, 3,5,4} exactly.
        pushTri(Aw, Bw, Cw);
        pushTri(Adw, Cdw, Bdw);
    }

    // Catcher screen: large white diffuse wall.
    {
        const float3 screenColor{0.9f, 0.9f, 0.9f};
        addQuad(verts, normals, uvs, materials,
                toWorld(float3{-300, -300, 600}), toWorld(float3{300, -300, 600}),
                toWorld(float3{300, 400, 600}), toWorld(float3{-300, 400, 600}),
                screenColor);
    }

    // Single directional light - add_distant()'s own `dir` argument is
    // the direction TOWARD the light source (verified empirically for
    // C3, section 134 - the field's own doc comment in
    // punctual_light_objects.h is misleading), so `direction` here
    // (this loader's own "direction of travel" convention) is the
    // NEGATION of CPU's own `vec3(0.0, 0.06, -1.0)`.
    {
        const float3 dirToLight = simd::normalize(float3{0.0f, 0.06f, -1.0f});
        const float3 dirOfTravel = -dirToLight;
        const float3 emission{3.0f, 3.0f, 3.0f};  // radiance(1,1,1) * scale(3.0)
        directionalLights.push_back(DirectionalLightData{
            PackedFloat3{dirOfTravel.x, dirOfTravel.y, dirOfTravel.z},
            PackedFloat3{emission.x, emission.y, emission.z}});
    }

    // Black background (no_lights registry row - a minimal standalone
    // scene, not a Cornell box) - same fix as C2's own background-leak
    // bug (section 134): without this, a ray that escapes past the
    // catcher screen or through the prism's own end caps would read
    // this loader's hardcoded blue-sky gradient instead of black.
    havePbrtConstantEnvLight = true;
    pbrtEnvColor = float3{0.0f, 0.0f, 0.0f};

    // Camera (kPrismCamera, scene_registry.h): fov=30,
    // lookfrom=(75,60,-400), lookat=(75,75,250) - through the SAME
    // toWorld() transform as the geometry above.
    const float3 lookfrom = toWorld(float3{75.0f, 60.0f, -400.0f});
    const float3 lookat = toWorld(float3{75.0f, 75.0f, 250.0f});
    const float3 up{0.0f, 1.0f, 0.0f};
    const float3 forward = simd::normalize(lookat - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    pbrtTanHalfFov = tanf(0.5f * 30.0f * (float)M_PI / 180.0f);
    havePbrtCamera = true;
    pbrtCameraLookAtWorld = lookat;
    pbrtCameraUpRaw = up;
    pbrtBboxCenter = bboxCenter;
    pbrtSceneScale = sceneScale;
    pbrtSceneOffset = sceneOffset;
}

void MetalPocApp::buildPrismDispersion() {
    buildPrismDispersionGeometry(/*glassMaterialType=*/22u, /*roughness=*/0.0f);
}

// B24: Frosted Prism Dispersion - the SAME prism/light/screen as B23,
// frosted glass (materialType 23, GGX roughness 0.08 - matches CPU's
// own `rough_dielectric::make_dispersive(1.52, 59.0, 0.08)` exactly)
// instead of smooth. See buildPrismDispersionGeometry()'s own
// declaration comment - the two scenes share one geometry builder,
// differing only in which glass material/roughness gets passed in.
void MetalPocApp::buildPrismDispersionRough() {
    buildPrismDispersionGeometry(/*glassMaterialType=*/23u, /*roughness=*/0.08f);
}

// B10: Principled Showcase - matches CPU's own build_principled_showcase()
// exactly: 7 spheres (radius 1.0, spaced 2.0 apart along X) demonstrating
// materialType 24's own 3-lobe PrincipledBxDF, from pure matte diffuse
// through semi-metallic to fully metallic/clearcoated, over a checkered
// ground plane, under one overhead area light. Not a Cornell-family
// scene - own standalone geometry/camera, F2's own "natural scale, plain
// +8-in-X offset (now +60, section 169), no rescale" convention (this
// scene's own extent, a ~14x20-unit span, is already compact enough).
//
// CPU's own ground is a checker-textured radius-1000 SPHERE
// (`point3(0,-1000,0)`) - replaced here with a large flat quad using the
// SAME materialType 16 (real 3D world-space checker) instead, matching
// this whole series' own established "huge sphere as ground plane"
// simplification (A5/B1/F2's own precedent): a radius-1000 sphere at
// this scene's own scale, offset the (then) usual +8 in X, algebraically
// overlaps the hardcoded POC room's own [-1,1] cube by about 1 unit
// (checked via |x-offset| <= sqrt(2*radius-1) before ever rendering, not
// discovered by a garbled render) - the same failure mode already fixed
// twice before, avoided here from the start.
void MetalPocApp::buildPrincipledShowcase() {
    // Used to be its own special-cased +10 (not the +8 every other scene
    // in this series used) - the leftmost sphere (x=-6, radius 1) sat
    // close enough to the hardcoded POC room's own [-1,1] cube that +8
    // left it exactly tangent to the room's own right face (world x=1),
    // letting a sliver of the room's own always-present geometry peek
    // through right next to it in a real render - caught by inspecting
    // the rendered image directly, not assumed. Section 169 raised the
    // shared baseline every scene in this series uses from +8 to +60
    // (a real, unrelated leftover-room leak found in G19/G25 - see that
    // section's own comment), which independently gives this scene's own
    // -6 extent 54 units of clearance, far more than the 1 extra unit
    // this special case ever needed - so it's folded back into the same
    // shared value as everything else, no longer a special case.
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};

    // Ground: large flat checker quad (materialType 16, real 3D
    // world-space checker) - `roughness` reused as the checker's own
    // cell size, matching CPU's own `checker_texture(0.5, ...)` scale
    // exactly (see materialType 16's own established convention,
    // section 121).
    {
        const float3 darkA{0.1f, 0.1f, 0.12f}, darkB{0.2f, 0.2f, 0.22f};
        addQuad(verts, normals, uvs, materials,
                float3{-30, 0, -30} + sceneOffset, float3{30, 0, -30} + sceneOffset,
                float3{30, 0, 30} + sceneOffset, float3{-30, 0, 30} + sceneOffset,
                darkA, /*materialType=*/16u, /*emission=*/simd::make_float3(0, 0, 0),
                /*lightId=*/-1, /*roughness(cell size)=*/0.5f, /*ior=*/1.0f,
                darkB);
    }

    // 7 principled spheres - color/metallic/roughness/ior/clearcoat/
    // clearcoatRoughness, matching build_principled_showcase()'s own
    // 7 `principled(...)` constructor calls exactly.
    struct SphereSpec { float3 pos; float3 color; float metallic; float roughness; float clearcoat; float clearcoatRough; };
    const SphereSpec kSpheres[7] = {
        {{-6, 1, 0}, {0.8f, 0.1f, 0.1f}, 0.0f, 0.9f, 0.0f, 0.1f},
        {{-4, 1, 0}, {0.1f, 0.2f, 0.8f}, 0.0f, 0.2f, 0.0f, 0.1f},
        {{-2, 1, 0}, {0.1f, 0.7f, 0.2f}, 0.0f, 0.3f, 1.0f, 0.05f},
        {{ 0, 1, 0}, {0.9f, 0.7f, 0.2f}, 0.5f, 0.3f, 0.0f, 0.1f},
        {{ 2, 1, 0}, {0.8f, 0.45f, 0.2f}, 0.8f, 0.4f, 0.0f, 0.1f},
        {{ 4, 1, 0}, {0.9f, 0.9f, 0.9f}, 1.0f, 0.05f, 0.0f, 0.1f},
        {{ 6, 1, 0}, {0.9f, 0.7f, 0.1f}, 1.0f, 0.1f, 1.0f, 0.08f},
    };
    for (const SphereSpec& s : kSpheres) {
        const float3 center = s.pos + sceneOffset;
        TriangleMaterial mat{PackedFloat3{s.color.x, s.color.y, s.color.z},
            /*materialType=*/24u, /*ior=*/1.5f, PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/s.roughness};
        mat.conductorEta = PackedFloat3{s.metallic, s.clearcoat, s.clearcoatRough};
        spheres.push_back(SphereData{PackedFloat3{center.x, center.y, center.z}, 1.0f});
        sphereMaterials.push_back(mat);
    }

    // Overhead area light - point3(-7,7,-5), 14x10, diffuse_light(6,6,6).
    {
        const float3 a = float3{-7, 7, -5} + sceneOffset;
        const float3 edgeU{14, 0, 0};
        const float3 edgeV{0, 0, 10};
        const float3 lightColor{6.0f, 6.0f, 6.0f};
        const int32_t lightId = (int32_t)lights.size();
        addQuad(verts, normals, uvs, materials, a, a + edgeU, a + edgeU + edgeV, a + edgeV,
                lightColor, /*materialType=*/0u, /*emission=*/lightColor, lightId);
        const float3 normalV = simd::normalize(simd::cross(edgeU, edgeV));
        const float area = simd::length(simd::cross(edgeU, edgeV));
        const float3 center = a + 0.5f * edgeU + 0.5f * edgeV;
        lights.push_back(AreaLightData{
            PackedFloat3{center.x, center.y, center.z},
            PackedFloat3{edgeU.x, edgeU.y, edgeU.z},
            PackedFloat3{edgeV.x, edgeV.y, edgeV.z},
            PackedFloat3{normalV.x, normalV.y, normalV.z},
            area,
            PackedFloat3{lightColor.x, lightColor.y, lightColor.z}});
    }

    // Background - kPrincipledShowcaseCamera's own (0.10,0.10,0.12),
    // matching CPU's own CameraConfig background_r/g/b exactly (a dark
    // bluish ambient, NOT black - unlike B23/B24's own pure-black
    // standalone scenes).
    havePbrtConstantEnvLight = true;
    pbrtEnvColor = float3{0.10f, 0.10f, 0.12f};

    // Camera: fov=45, lookfrom=(0,2.7,17), lookat=(0,1,0).
    const float3 lookfrom = float3{0.0f, 2.7f, 17.0f} + sceneOffset;
    const float3 lookat = float3{0.0f, 1.0f, 0.0f} + sceneOffset;
    const float3 up{0.0f, 1.0f, 0.0f};
    const float3 forward = simd::normalize(lookat - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    pbrtTanHalfFov = tanf(0.5f * 45.0f * (float)M_PI / 180.0f);
    havePbrtCamera = true;
    pbrtCameraLookAtWorld = lookat;
    pbrtCameraUpRaw = up;
    pbrtBboxCenter = float3{0.0f, 0.0f, 0.0f};
    pbrtSceneScale = 1.0f;
    pbrtSceneOffset = sceneOffset;
}

// B11: Hair Fibers - matches src/TheRestOfYourLife/scenes_advanced.h's own
// build_hair_fibers() exactly: a giant-sphere ground plus 5 spheres (radius
// 1.0) using materialType 31 (shadeHair(), metal_poc_materials_hair.metal)
// in place of CPU's own hair_material, each with its own sigma_a/beta_m/
// beta_n/alpha_deg (see the HairSpec table below, transcribed directly
// from build_hair_fibers()'s own 5 hair_material(...) constructor calls -
// eta uses hair_material's own default of 1.55 for all 5, matching CPU
// exactly since none of the 5 calls there override it). The overhead
// light's own deliberately dim intensity (0.22,0.22,0.19, not this
// series' usual 6,6,6) is NOT a Metal-specific tuning choice - it mirrors
// a pre-existing CPU calibration already established in
// build_hair_fibers()'s own comment (hair's BSDF response is naturally far
// brighter than diffuse/glossy surfaces', so the "normal" intensity blows
// out the whole frame under ACES).
void MetalPocApp::buildHairFibersScene() {
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};

    // Dark floor - point3(0,-1000,0), radius 1000, lambertian(0.05,0.05,0.06).
    {
        const float3 center = float3{0.0f, -1000.0f, 0.0f} + sceneOffset;
        TriangleMaterial mat{PackedFloat3{0.05f, 0.05f, 0.06f},
            /*materialType=*/0u, /*ior=*/1.0f, PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
        spheres.push_back(SphereData{PackedFloat3{center.x, center.y, center.z}, 1000.0f});
        sphereMaterials.push_back(mat);
    }

    // 5 hair spheres - field reuse matches shadeHair()'s own comment:
    // color=sigma_a(rgb), ior=eta, roughness=beta_m, conductorEta.x=beta_n,
    // conductorEta.y=alpha_deg.
    struct HairSpec { float3 pos; float3 sigmaA; float betaM; float betaN; float alphaDeg; };
    const HairSpec kHairSpheres[5] = {
        {{-3.5f, 1.0f, 0.0f},  {0.06f, 0.10f, 0.20f},   0.25f, 0.25f, 2.0f},  // dark brown
        {{-1.2f, 1.0f, 0.4f},  {0.01f, 0.015f, 0.03f},  0.30f, 0.30f, 2.0f}, // blonde
        {{ 1.2f, 1.0f, -0.4f}, {0.02f, 0.08f, 0.18f},   0.20f, 0.20f, 3.0f}, // auburn
        {{ 3.5f, 1.0f, 0.0f},  {0.001f, 0.001f, 0.002f}, 0.45f, 0.45f, 1.0f}, // white/silver
        {{ 0.0f, 1.0f, 2.3f},  {0.50f, 0.55f, 0.60f},   0.15f, 0.15f, 2.0f},  // fine black
    };
    const float eta = 1.55f;
    for (const HairSpec& s : kHairSpheres) {
        const float3 center = s.pos + sceneOffset;
        TriangleMaterial mat{PackedFloat3{s.sigmaA.x, s.sigmaA.y, s.sigmaA.z},
            /*materialType=*/31u, /*ior=*/eta, PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/s.betaM};
        mat.conductorEta = PackedFloat3{s.betaN, s.alphaDeg, 0.0f};
        spheres.push_back(SphereData{PackedFloat3{center.x, center.y, center.z}, 1.0f});
        sphereMaterials.push_back(mat);
    }

    // Overhead area light - point3(-5,6,-5), 10x7, diffuse_light(0.22,0.22,0.19).
    {
        const float3 a = float3{-5.0f, 6.0f, -5.0f} + sceneOffset;
        const float3 edgeU{10.0f, 0.0f, 0.0f};
        const float3 edgeV{0.0f, 0.0f, 7.0f};
        const float3 lightColor{0.22f, 0.22f, 0.19f};
        const int32_t lightId = (int32_t)lights.size();
        addQuad(verts, normals, uvs, materials, a, a + edgeU, a + edgeU + edgeV, a + edgeV,
                lightColor, /*materialType=*/0u, /*emission=*/lightColor, lightId);
        const float3 normalV = simd::normalize(simd::cross(edgeU, edgeV));
        const float area = simd::length(simd::cross(edgeU, edgeV));
        const float3 center = a + 0.5f * edgeU + 0.5f * edgeV;
        lights.push_back(AreaLightData{
            PackedFloat3{center.x, center.y, center.z},
            PackedFloat3{edgeU.x, edgeU.y, edgeU.z},
            PackedFloat3{edgeV.x, edgeV.y, edgeV.z},
            PackedFloat3{normalV.x, normalV.y, normalV.z},
            area,
            PackedFloat3{lightColor.x, lightColor.y, lightColor.z}});
    }

    // Background - (0.05,0.05,0.07), matching B11's own CameraConfig
    // background_r/g/b exactly (scene_registry_data.h).
    havePbrtConstantEnvLight = true;
    pbrtEnvColor = float3{0.05f, 0.05f, 0.07f};

    // Camera: fov=45, lookfrom=(0,2.5,14), lookat=(0,1,0).
    const float3 lookfrom = float3{0.0f, 2.5f, 14.0f} + sceneOffset;
    const float3 lookat = float3{0.0f, 1.0f, 0.0f} + sceneOffset;
    const float3 up{0.0f, 1.0f, 0.0f};
    const float3 forward = simd::normalize(lookat - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    pbrtTanHalfFov = tanf(0.5f * 45.0f * (float)M_PI / 180.0f);
    havePbrtCamera = true;
    pbrtCameraLookAtWorld = lookat;
    pbrtCameraUpRaw = up;
    pbrtBboxCenter = float3{0.0f, 0.0f, 0.0f};
    pbrtSceneScale = 1.0f;
    pbrtSceneOffset = sceneOffset;
}

// C1: HDRI Sky - matches src/TheRestOfYourLife/scenes_advanced.h's own
// build_hdri_sky_world()/build_hdri_sky() exactly: a ground plane + 3
// spheres (diffuse, fuzzy-metal, glass), lit ENTIRELY by a procedural
// gradient sky image - no area/point/directional light of any kind.
// Needs zero new materialType or shader code (see this method's own
// forward-declaration comment) - just populates the same
// havePbrtImageEnvLight/pbrtEnvImageWidth/Height/Pixels fields
// loadPbrtScene() already populates for a REAL pbrt ImageInfiniteLight,
// with a host-generated buffer standing in for a loaded HDRI file.

