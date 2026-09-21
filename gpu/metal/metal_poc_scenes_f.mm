// metal_poc_scenes_f.mm
// Category F (Geometry) hand-authored scene builders -
// MetalPocApp::buildXxx() out-of-class definitions, split out of
// metal_poc.mm as a pure code-motion refactor (no behaviour change) -
// see metal_poc_app.h's own header comment and docs/METAL_GPU_FEASIBILITY.md.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_poc_app.h"

void MetalPocApp::buildTriangleMeshScene() {
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};

    // Ground.
    {
        const float3 tileA{0.15f, 0.15f, 0.15f}, tileB{0.85f, 0.85f, 0.85f};
        addQuad(verts, normals, uvs, materials,
                float3{-15.0f, 0.0f, -15.0f} + sceneOffset, float3{15.0f, 0.0f, -15.0f} + sceneOffset,
                float3{15.0f, 0.0f, 15.0f} + sceneOffset, float3{-15.0f, 0.0f, 15.0f} + sceneOffset,
                tileA, /*materialType=*/16u, /*emission=*/simd::make_float3(0, 0, 0),
                /*lightId=*/-1, /*roughness=*/0.8f, /*ior=*/1.0f, tileB);
    }

    // Icosahedron - 12 vertices at golden-ratio coordinates, scaled to
    // radius 1.5 and centred at (0,2.5,0).
    {
        const float phi = (1.0f + sqrtf(5.0f)) / 2.0f;
        const float radius = 1.5f;
        const float3 rawVerts[12] = {
            {-1, phi, 0}, {1, phi, 0}, {-1, -phi, 0}, {1, -phi, 0},
            {0, -1, phi}, {0, 1, phi}, {0, -1, -phi}, {0, 1, -phi},
            {phi, 0, -1}, {phi, 0, 1}, {-phi, 0, -1}, {-phi, 0, 1},
        };
        const float vertLen = simd::length(rawVerts[0]);
        const float3 center = float3{0.0f, 2.5f, 0.0f} + sceneOffset;
        float3 scaledVerts[12];
        for (int i = 0; i < 12; ++i) scaledVerts[i] = center + (radius / vertLen) * rawVerts[i];

        const int faces[20][3] = {
            {0,11,5}, {0,5,1}, {0,1,7}, {0,7,10}, {0,10,11},
            {1,5,9}, {5,11,4}, {11,10,2}, {10,7,6}, {7,1,8},
            {3,9,4}, {3,4,2}, {3,2,6}, {3,6,8}, {3,8,9},
            {4,9,5}, {2,4,11}, {6,2,10}, {8,6,7}, {9,8,1},
        };
        const float3 albedo{0.8f, 0.6f, 0.2f};
        const float alpha = 0.15f;
        const float3 k = reflectanceToConductorK(albedo);
        TriangleMaterial mat{PackedFloat3{albedo.x, albedo.y, albedo.z}, /*materialType=*/4u,
            /*ior=*/alpha, PackedFloat3{0, 0, 0}, -1, /*roughness=*/alpha};
        mat.conductorEta = PackedFloat3{1, 1, 1};
        mat.conductorK = PackedFloat3{k.x, k.y, k.z};
        for (const auto& f : faces) {
            const float3 a = scaledVerts[f[0]], b = scaledVerts[f[1]], c = scaledVerts[f[2]];
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
            materials.push_back(mat);
        }
    }

    // Overhead area light - direct-hit only (see this function's own
    // declaration comment).
    {
        const float3 lightColor{6.0f, 6.0f, 6.0f};
        const float3 c = float3{0.0f, 8.0f, 0.0f} + sceneOffset;
        TriangleMaterial mat{PackedFloat3{lightColor.x, lightColor.y, lightColor.z}, /*materialType=*/0u,
            1.0f, PackedFloat3{lightColor.x, lightColor.y, lightColor.z}, /*lightId=*/-1, 0.0f};
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 2.0f});
        sphereMaterials.push_back(mat);
    }

    // Real per-scene flat background.
    havePbrtConstantEnvLight = true;
    pbrtEnvColor = float3{0.05f, 0.05f, 0.08f};

    // Camera: CPU's own real registry row for F2, ported directly
    // (vfov 35, lookfrom (0,4,8), lookat (0,2.5,0)).
    const float3 lookfrom = float3{0.0f, 4.0f, 8.0f} + sceneOffset;
    const float3 lookat = float3{0.0f, 2.5f, 0.0f} + sceneOffset;
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
    pbrtBboxCenter = float3{0.0f, 0.0f, 0.0f};
    pbrtSceneScale = 1.0f;
    pbrtSceneOffset = sceneOffset;
}

// D5: Depth of Field Cornell Box - matches CPU's own registry row for
// D5 exactly: the identical A1 Cornell box geometry (build_cornell_box
// on CPU), with real thin-lens defocus blur added on top via a real
// defocus_angle=2.0/focus_dist=800.0 (both in the scene's own pbrt-
// file-scale units, matching CameraConfig's own field meaning -
// scene_registry.h). `defocus_radius = focus_dist *
// tan(defocus_angle/2)` is camera.h's own real formula (ported
// directly, not re-derived) - both the resulting lens radius AND the
// focus distance itself need the SAME `sceneScale` this scene's own
// geometry/camera position already go through (they are WORLD-SPACE
// distances in the pre-rescale coordinate system, just like a
// lookfrom/lookat position), or the defocus cone would be sized for
// the wrong (much larger) scale entirely.

void MetalPocApp::buildBilinearPatchScene() {
    using namespace cornell_box_data;
    const float3 bboxMin{0.0f, 0.0f, 0.0f};
    const float3 bboxMax{555.0f, 555.0f, 555.0f};
    const float sceneScale = 2.0f / 555.0f;
    const float3 bboxCenter = 0.5f * (bboxMin + bboxMax);
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};
    auto toWorld = [=](float3 p) { return (p - bboxCenter) * sceneScale + sceneOffset; };

    // All 6 walls, including the light - same pattern buildHomogeneousMediumScene()'s
    // own loop already uses.
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

    // Saddle patch: p00/p11 high, p10/p01 low (classic hyperbolic
    // paraboloid) - CPU's own metal(color(0.8,0.7,0.3), 0.15).
    addBilinearPatch(verts, normals, uvs, materials,
        toWorld(float3{150, 80, 200}), toWorld(float3{400, 50, 200}),
        toWorld(float3{150, 50, 400}), toWorld(float3{400, 80, 400}),
        float3{0.8f, 0.7f, 0.3f}, 0.15f);

    // Ramp patch: linear in u, curved in v - CPU's own
    // metal(color(0.2,0.4,0.8), 0.25).
    addBilinearPatch(verts, normals, uvs, materials,
        toWorld(float3{200, 200, 220}), toWorld(float3{370, 200, 220}),
        toWorld(float3{150, 380, 420}), toWorld(float3{420, 320, 420}),
        float3{0.2f, 0.4f, 0.8f}, 0.25f);

    // Camera - same dead-on A1/D5/D6 Cornell camera.
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

// F4: Curve Fibers - matches build_curve_fibers_scene() exactly: a
// checker ground + 70 windswept, tapered Bezier hair strands rooted in
// a Fibonacci-disk arrangement, lit by an overhead area light. CPU's
// own root placement is fully DETERMINISTIC (a seeded hash, not the
// engine's own RNG) - `hash01()` below is a direct, bit-for-bit port
// of that same formula, so this loader's own strand roots land in
// EXACTLY the same positions as CPU's, not just a visually-similar
// random field (unlike D1's own small accent spheres, which really are
// unseeded on the CPU side and don't need this).
void MetalPocApp::buildCurveFibersScene() {
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};

    // Ground: flat checker quad (materialType 16), not CPU's own
    // radius-1000 ground SPHERE - the established overlap-avoidance
    // substitution.
    {
        const float3 darkA{0.15f, 0.15f, 0.15f}, lightB{0.85f, 0.85f, 0.85f};
        addQuad(verts, normals, uvs, materials,
                float3{-30, 0, -30} + sceneOffset, float3{30, 0, -30} + sceneOffset,
                float3{30, 0, 30} + sceneOffset, float3{-30, 0, 30} + sceneOffset,
                darkA, /*materialType=*/16u, /*emission=*/simd::make_float3(0, 0, 0),
                /*lightId=*/-1, /*roughness(cell size)=*/0.8f, /*ior=*/1.0f, lightB);
    }

    // Deterministic per-strand pseudo-random in [0,1) - CPU's own
    // hash01() lambda, ported bit-for-bit (same unsigned-int constants
    // and operation order) so strand height/lean lands identically.
    auto hash01 = [](int i, int salt) -> float {
        uint32_t h = (uint32_t)i * 374761393u + (uint32_t)salt * 668265263u;
        h = (h ^ (h >> 13)) * 1274126177u;
        h ^= (h >> 16);
        return (float)(h & 0xFFFFFFu) / (float)0xFFFFFFu;
    };

    const float3 palette[5] = {
        {0.25f, 0.14f, 0.06f}, {0.80f, 0.65f, 0.35f}, {0.45f, 0.13f, 0.05f},
        {0.75f, 0.75f, 0.78f}, {0.03f, 0.03f, 0.03f},
    };

    const int strandCount = 70;
    const float diskRadius = 1.4f;
    const float goldenAngle = 2.399963229728653f;

    for (int i = 0; i < strandCount; ++i) {
        const float frac = ((float)i + 0.5f) / (float)strandCount;
        const float r = diskRadius * sqrtf(frac);
        const float angle = (float)i * goldenAngle;
        const float bx = r * cosf(angle), bz = r * sinf(angle);

        const float height = 0.9f + 0.5f * hash01(i, 1);
        const float lean = height * (0.35f + 0.35f * hash01(i, 2));

        float3 cp[4] = {
            float3{bx, 0.0f, bz} + sceneOffset,
            float3{bx + 0.15f * lean, height * 0.33f, bz} + sceneOffset,
            float3{bx + 0.55f * lean, height * 0.70f, bz} + sceneOffset,
            float3{bx + lean, height, bz} + sceneOffset,
        };
        addTaperedTube(verts, normals, uvs, materials, cp, 0.045f, 0.006f, palette[i % 5]);
    }

    // Overhead area light - quad(-2.5,4.0,-2.5), 5x5, diffuse_light(6,6,6).
    {
        const float3 a = float3{-2.5f, 4.0f, -2.5f} + sceneOffset;
        const float3 edgeU{5.0f, 0.0f, 0.0f};
        const float3 edgeV{0.0f, 0.0f, 5.0f};
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

    // Background - kCurveFibersCamera's own (0.04,0.045,0.06), a near-
    // black dark ambient.
    havePbrtConstantEnvLight = true;
    pbrtEnvColor = float3{0.04f, 0.045f, 0.06f};

    // Camera: fov=38, lookfrom=(0,2.0,6.5), lookat=(0,0.7,0).
    const float3 lookfrom = float3{0.0f, 2.0f, 6.5f} + sceneOffset;
    const float3 lookat = float3{0.0f, 0.7f, 0.0f} + sceneOffset;
    const float3 up{0.0f, 1.0f, 0.0f};
    const float3 forward = simd::normalize(lookat - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    pbrtTanHalfFov = tanf(0.5f * 38.0f * (float)M_PI / 180.0f);
    havePbrtCamera = true;
    pbrtCameraLookAtWorld = lookat;
    pbrtCameraUpRaw = up;
    pbrtBboxCenter = float3{0.0f, 0.0f, 0.0f};
    pbrtSceneScale = 1.0f;
    pbrtSceneOffset = sceneOffset;
}

// E1: Homogeneous Medium - matches CPU's own build_homogeneous_medium_scene()
// in GEOMETRY exactly: the standard 6 Cornell walls (kQuads[0..5],
// including the SAME light quad - CPU's own scene reuses these exact
// literal numbers), no box, no sphere, filled with a real homogeneous
// scattering fog. The fog DENSITY itself needed real empirical
// recalibration, not just CPU's own literal sigma_t - see
// pbrtFogSigmaT's own assignment below for the full explanation (a
// genuine architectural mismatch between this loader's own "fog fills
// whatever the ray already hits" convention and CPU's own explicit,
// localized medium-boundary volume, not a simple scale-formula bug).

