// metal_poc_scenes_i.mm
// Category I (Education) hand-authored scene builders -
// MetalPocApp::buildXxx() out-of-class definitions, split out of
// metal_poc.mm as a pure code-motion refactor (no behaviour change) -
// see metal_poc_app.h's own header comment and docs/METAL_GPU_FEASIBILITY.md.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_poc_app.h"

void MetalPocApp::buildLightSamplerComparison() {
    using namespace cornell_box_data;
    const float3 bboxMin{0.0f, 0.0f, 0.0f};
    const float3 bboxMax{555.0f, 555.0f, 555.0f};
    const float sceneScale = 2.0f / 555.0f;
    const float3 bboxCenter = 0.5f * (bboxMin + bboxMax);
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};
    auto toWorld = [=](float3 p) { return (p - bboxCenter) * sceneScale + sceneOffset; };

    // The 5 walls only (kQuads[5], the usual single ceiling light, is
    // skipped - replaced by the five lopsided-power lights below).
    for (int i = 0; i < 5; ++i) {
        const QuadSpec& q = kQuads[i];
        const float3 Q{(float)q.Q.x, (float)q.Q.y, (float)q.Q.z};
        const float3 u{(float)q.u.x, (float)q.u.y, (float)q.u.z};
        const float3 v{(float)q.v.x, (float)q.v.y, (float)q.v.z};
        const float3 color{(float)q.color.r, (float)q.color.g, (float)q.color.b};
        addQuad(verts, normals, uvs, materials, toWorld(Q), toWorld(Q + u),
                toWorld(Q + u + v), toWorld(Q + v), color);
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

    // The glass sphere - identical to buildCornellBoxA1()'s own.
    {
        const float3 center = toWorld(float3{(float)kGlassSphere.center.x,
            (float)kGlassSphere.center.y, (float)kGlassSphere.center.z});
        spheres.push_back(SphereData{PackedFloat3{center.x, center.y, center.z},
            sceneScale * (float)kGlassSphere.radius});
        sphereMaterials.push_back(TriangleMaterial{PackedFloat3{1, 1, 1}, /*materialType=*/2u,
            /*ior=*/(float)kGlassSphere.glass_ior, PackedFloat3{0, 0, 0}, /*lightId=*/-1,
            /*roughness=*/0.0f});
    }

    // 5 quad lights, ~1:2:6:15:80 power ratio - matches CPU's own real
    // positions/sizes/emission values exactly (quad area is uniform at
    // 40x40 for four of them, so the ratio is also each one's emission
    // scale directly - the fifth, the original A1-position light, is
    // the larger 130x105 quad at a matching 15x scale).
    auto pushLight = [&](float3 Q, float3 u, float3 v, float3 emission) {
        const float3 a = toWorld(Q), b = toWorld(Q + u), c = toWorld(Q + u + v), d = toWorld(Q + v);
        const int32_t lightId = (int32_t)lights.size();
        addQuad(verts, normals, uvs, materials, a, b, c, d, emission,
                /*materialType=*/0u, /*emission=*/emission, lightId);
        const float3 edgeU = b - a, edgeV = d - a;
        const float3 normalV = simd::normalize(simd::cross(edgeU, edgeV));
        const float area = simd::length(simd::cross(edgeU, edgeV));
        const float3 center = a + 0.5f * edgeU + 0.5f * edgeV;
        lights.push_back(AreaLightData{
            PackedFloat3{center.x, center.y, center.z},
            PackedFloat3{edgeU.x, edgeU.y, edgeU.z},
            PackedFloat3{edgeV.x, edgeV.y, edgeV.z},
            PackedFloat3{normalV.x, normalV.y, normalV.z},
            area, PackedFloat3{emission.x, emission.y, emission.z},
            /*patternTileB=*/0.0f, /*patternScale=*/0.0f,
            /*twoSided=*/0.0f, /*useTexture=*/0.0f});
    };
    pushLight(float3{30, 554, 30}, float3{40, 0, 0}, float3{0, 0, 40}, float3{1, 1, 1});
    pushLight(float3{485, 554, 30}, float3{40, 0, 0}, float3{0, 0, 40}, float3{2, 2, 2});
    pushLight(float3{30, 554, 485}, float3{40, 0, 0}, float3{0, 0, 40}, float3{6, 6, 6});
    pushLight(float3{213, 554, 227}, float3{130, 0, 0}, float3{0, 0, 105}, float3{15, 15, 15});
    pushLight(float3{485, 554, 485}, float3{40, 0, 0}, float3{0, 0, 40}, float3{80, 80, 80});

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

// See this method's own declaration comment. Also sets up the SAME
// camera every C2-C6 scene shares (kCornellBoxCamera) - a caller only
// needs to add its own punctual light after calling this.

