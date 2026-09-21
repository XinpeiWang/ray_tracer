// metal_poc_scenes_d.mm
// Category D (Cameras) hand-authored scene builders -
// MetalPocApp::buildXxx() out-of-class definitions, split out of
// metal_poc.mm as a pure code-motion refactor (no behaviour change) -
// see metal_poc_app.h's own header comment and docs/METAL_GPU_FEASIBILITY.md.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_poc_app.h"

void MetalPocApp::buildDepthOfFieldCornellBox() {
    buildCornellBoxA1();
    const float sceneScale = 2.0f / 555.0f;
    const float defocusAngleDeg = 2.0f;
    const float focusDistRaw = 800.0f;
    const float lensRadiusRaw = focusDistRaw * tanf(defocusAngleDeg * 0.5f * (float)M_PI / 180.0f);
    pbrtLensRadius = lensRadiusRaw * sceneScale;
    pbrtFocusDistance = focusDistRaw * sceneScale;
}

// D1: Depth of Field - matches build_depth_of_field() exactly: an open
// (non-Cornell) scene, checker ground, 4 "hero" spheres at varying
// depth from the lens (2 out-of-focus lambertian, 1 in-focus glass, 1
// in-focus metal), plus a row of 7 small accent spheres. Natural scale
// (sceneScale=1.0, the F2/B10/C1 convention), so pbrtLensRadius/
// pbrtFocusDistance need no scale conversion at all - unlike D5's own
// 555-unit Cornell-box case, the raw defocus_angle/focus_dist formula
// applies directly.
void MetalPocApp::buildDepthOfField() {
    const float3 sceneOffset{8.0f, 0.0f, 0.0f};

    // Checker ground (materialType 16, real 3D checker) - matches
    // CPU's own checker_texture(0.5, (0.2,0.3,0.1), (0.9,0.9,0.9)).
    {
        const float3 darkA{0.2f, 0.3f, 0.1f}, lightB{0.9f, 0.9f, 0.9f};
        addQuad(verts, normals, uvs, materials,
                float3{-30, 0, -30} + sceneOffset, float3{30, 0, -30} + sceneOffset,
                float3{30, 0, 30} + sceneOffset, float3{-30, 0, 30} + sceneOffset,
                darkA, /*materialType=*/16u, /*emission=*/simd::make_float3(0, 0, 0),
                /*lightId=*/-1, /*roughness(cell size)=*/0.5f, /*ior=*/1.0f, lightB);
    }

    // Background sphere (out of focus far), lambertian.
    {
        TriangleMaterial mat{PackedFloat3{0.4f, 0.2f, 0.1f}, 0u, 1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f};
        const float3 c = float3{-4.0f, 1.0f, -3.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 1.0f});
        sphereMaterials.push_back(mat);
    }
    // Near sphere (out of focus near), lambertian.
    {
        TriangleMaterial mat{PackedFloat3{0.7f, 0.3f, 0.3f}, 0u, 1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f};
        const float3 c = float3{-1.5f, 0.5f, 1.5f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 0.5f});
        sphereMaterials.push_back(mat);
    }
    // In-focus centre sphere, smooth dielectric (glass). `color`={0,0,0}
    // - the Beer-Lambert absorption coefficient this materialType
    // actually reads (section 146's own C1 finding), not a reflectance
    // tint - {0,0,0} is true clear glass.
    {
        TriangleMaterial mat{PackedFloat3{0.0f, 0.0f, 0.0f}, 2u, 1.5f, PackedFloat3{0, 0, 0}, -1, 0.0f};
        const float3 c = float3{0.0f, 1.0f, 0.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 1.0f});
        sphereMaterials.push_back(mat);
    }
    // In-focus metal sphere - CPU's own metal(color(0.7,0.6,0.5),
    // fuzz=0.05), approximated via reflectanceToConductorK() (materialType
    // 4), the same B2/C1/C7 substitution.
    {
        const float3 color{0.7f, 0.6f, 0.5f};
        TriangleMaterial mat{PackedFloat3{color.x, color.y, color.z}, /*materialType=*/4u,
                              /*ior(alphaX)=*/0.05f, PackedFloat3{0, 0, 0}, -1, /*roughness(alphaY)=*/0.05f};
        const float3 k = reflectanceToConductorK(color);
        mat.conductorEta = PackedFloat3{1.0f, 1.0f, 1.0f};
        mat.conductorK = PackedFloat3{k.x, k.y, k.z};
        const float3 c = float3{2.0f, 1.0f, 0.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 1.0f});
        sphereMaterials.push_back(mat);
    }
    // Far sphere (out of focus), lambertian.
    {
        TriangleMaterial mat{PackedFloat3{0.1f, 0.2f, 0.6f}, 0u, 1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f};
        const float3 c = float3{4.0f, 1.0f, -2.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 1.0f});
        sphereMaterials.push_back(mat);
    }
    // 7 small accent spheres, i=-3..3 - CPU's own version colours each
    // with an UNSEEDED random_double(0.3,0.9) per channel (no fixed
    // seed anywhere in build_depth_of_field()), so CPU's own reference
    // render already differs between runs here - an exact colour match
    // isn't a meaningful bar for this loop. A fixed, varied palette
    // instead (deterministic, reproducible across runs on this side).
    {
        const float3 kAccentColors[7] = {
            {0.85f, 0.4f, 0.5f}, {0.4f, 0.8f, 0.5f}, {0.5f, 0.5f, 0.85f}, {0.8f, 0.75f, 0.35f},
            {0.4f, 0.7f, 0.8f}, {0.75f, 0.45f, 0.8f}, {0.6f, 0.85f, 0.4f},
        };
        for (int i = -3; i <= 3; ++i) {
            const float3 color = kAccentColors[i + 3];
            TriangleMaterial mat{PackedFloat3{color.x, color.y, color.z}, 0u, 1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f};
            const float3 c = float3{i * 1.2f, 0.2f, 2.5f + i * 0.3f} + sceneOffset;
            spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 0.2f});
            sphereMaterials.push_back(mat);
        }
    }

    // Background - kDepthOfFieldCamera's own (0.70,0.80,1.00), the same
    // literal buildStanfordBunny()-family scenes already use for this
    // exact colour (havePbrtConstantEnvLight/pbrtEnvColor).
    havePbrtConstantEnvLight = true;
    pbrtEnvColor = float3{0.70f, 0.80f, 1.00f};

    // Camera: fov=62, lookfrom=(0,2,9), lookat=(0,1,0), defocus_angle=10,
    // focus_dist=9 - kDepthOfFieldCamera's own literal values. Natural
    // scale (no rescale), so the lens-radius formula applies directly,
    // no sceneScale multiply (unlike D5's own 555-unit case).
    const float3 lookfrom = float3{0.0f, 2.0f, 9.0f} + sceneOffset;
    const float3 lookat = float3{0.0f, 1.0f, 0.0f} + sceneOffset;
    const float3 up{0.0f, 1.0f, 0.0f};
    const float3 forward = simd::normalize(lookat - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    pbrtTanHalfFov = tanf(0.5f * 62.0f * (float)M_PI / 180.0f);
    havePbrtCamera = true;
    pbrtCameraLookAtWorld = lookat;
    pbrtCameraUpRaw = up;
    pbrtBboxCenter = float3{0.0f, 0.0f, 0.0f};
    pbrtSceneScale = 1.0f;
    pbrtSceneOffset = sceneOffset;

    const float defocusAngleDeg = 10.0f;
    const float focusDistRaw = 9.0f;
    pbrtLensRadius = focusDistRaw * tanf(defocusAngleDeg * 0.5f * (float)M_PI / 180.0f);
    pbrtFocusDistance = focusDistRaw;
}

// D6: Orthographic Camera Cornell Box - the EXACT SAME geometry/camera
// POSITION buildCornellBoxA1() already builds (same dead-on lookfrom/
// lookat as A1/D5 - CPU's own build_cornell_box() scene, reused
// unchanged), just switching the projection mode to orthographic
// afterward. `pbrtTanHalfFov` is repurposed as the orthographic screen
// window's own half-extent (compute_screen_window()'s own unscaled
// +-1/+-aspect window times CPU's own 320 literal, matching
// build_ortho_camera_cornell_box's own screen-window scale exactly -
// see cameras.h's own compute_screen_window() and
// Uniforms::cameraOrthographic's own comment, metal_poc.metal) - the
// SAME sceneScale-multiplied world-space-distance convention
// pbrtLensRadius already uses.
void MetalPocApp::buildOrthoCornellBox() {
    buildCornellBoxA1();
    const float sceneScale = 2.0f / 555.0f;
    havePbrtOrthographic = true;
    pbrtTanHalfFov = 320.0f * sceneScale;
}

// D2: Orthographic Camera - matches build_ortho_camera_scene() exactly:
// checker ground + 5 spheres in a row at varying x, lit by a constant
// sky, viewed through a real orthographic camera. Natural scale
// (sceneScale=1.0, no rescale needed) - the screen-window half-extent
// (5, matching the scene's own alt-camera lambda) needs no sceneScale
// multiply either, unlike D6's own 555-unit Cornell-box case.
void MetalPocApp::buildOrthoCameraScene() {
    const float3 sceneOffset{8.0f, 0.0f, 0.0f};

    // Ground: a large flat checker quad (materialType 16), not CPU's
    // own radius-100 ground SPHERE - the established A5/B1/F2/B10/C1
    // substitution to avoid overlapping the hardcoded room's own
    // [-1,1] cube. Matches checker_texture(1.0, (0.2,0.2,0.2),
    // (0.9,0.9,0.9)) exactly.
    {
        const float3 darkA{0.2f, 0.2f, 0.2f}, lightB{0.9f, 0.9f, 0.9f};
        addQuad(verts, normals, uvs, materials,
                float3{-30, 0, -30} + sceneOffset, float3{30, 0, -30} + sceneOffset,
                float3{30, 0, 30} + sceneOffset, float3{-30, 0, 30} + sceneOffset,
                darkA, /*materialType=*/16u, /*emission=*/simd::make_float3(0, 0, 0),
                /*lightId=*/-1, /*roughness(cell size)=*/1.0f, /*ior=*/1.0f, lightB);
    }

    // 5 spheres in a row, x = (i-2)*2.5, radius 1 - CPU's own
    // gradient colour formula (0.2+0.15*i, 0.3, 0.8-0.1*i) applied
    // exactly.
    for (int i = 0; i < 5; ++i) {
        const float3 color{0.2f + 0.15f * i, 0.3f, 0.8f - 0.1f * i};
        TriangleMaterial mat{PackedFloat3{color.x, color.y, color.z}, 0u, 1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f};
        const float3 c = float3{(i - 2) * 2.5f, 1.0f, 0.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 1.0f});
        sphereMaterials.push_back(mat);
    }

    // Sky - build_ortho_sky()'s own sky_light(color(0.5,0.7,1.0)),
    // constant colour.
    havePbrtConstantEnvLight = true;
    pbrtEnvColor = float3{0.5f, 0.7f, 1.0f};

    // Camera: lookfrom=(0,10,20), lookat=(0,1,0) - kOrthoCameraCamera's
    // own literal values. Orthographic screen-window half-extent 5,
    // matching build_ortho_camera_scene()'s own alt-camera lambda
    // (xmin*5/xmax*5/ymin*5/ymax*5) - no sceneScale multiply, this
    // scene is natural-scale.
    const float3 lookfrom = float3{0.0f, 10.0f, 20.0f} + sceneOffset;
    const float3 lookat = float3{0.0f, 1.0f, 0.0f} + sceneOffset;
    const float3 up{0.0f, 1.0f, 0.0f};
    const float3 forward = simd::normalize(lookat - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    havePbrtCamera = true;
    havePbrtOrthographic = true;
    pbrtTanHalfFov = 5.0f;
    pbrtCameraLookAtWorld = lookat;
    pbrtCameraUpRaw = up;
    pbrtBboxCenter = float3{0.0f, 0.0f, 0.0f};
    pbrtSceneScale = 1.0f;
    pbrtSceneOffset = sceneOffset;
}

// D7: Spherical Camera Cornell Box - the EXACT SAME A1 Cornell box
// geometry, viewed from the box's own CENTER (278,278,278 in raw
// pbrt-file units) as a real 360-degree equirectangular panorama.
// CPU's own alt-camera lambda avoids a degenerate cross(up,forward) by
// using a FIXED +Z world-forward reference (lookat = lookfrom + (0,0,1))
// instead of feeding the scene's own registry lookat through
// make_look_at() directly - ported the same way: forward is the
// constant (0,0,1), unaffected by toWorld()'s own recentre/offset
// (only a DIRECTION between two points, and toWorld() is a uniform
// scale + translate, so direction is preserved exactly).
void MetalPocApp::buildSphericalCornellBox() {
    buildCornellBoxA1();
    // A1's own outside-looking-in camera never sees past the box's own
    // open front, so buildCornellBoxA1() never needed to set an
    // explicit background - this loader's own default two-colour sky
    // gradient fallback was invisible. This panorama camera, viewed
    // from the box's own CENTER, DOES look straight out through that
    // open front - CPU's own reference (no infinite light set for this
    // scene at all) renders that direction as true BLACK (a miss ray
    // with no light source contributes zero radiance), not this
    // loader's own default sky gradient. A real difference found via
    // direct comparison (the open-front direction rendered as a pale
    // gradient here, solid black in --cpu) - fixed by forcing a true
    // black constant "environment" explicitly, the same
    // havePbrtConstantEnvLight/pbrtEnvColor={0,0,0} pattern several
    // earlier standalone (non-Cornell-family-background) scenes already
    // use for an intentionally black backdrop.
    havePbrtConstantEnvLight = true;
    pbrtEnvColor = float3{0.0f, 0.0f, 0.0f};
    const float3 bboxMin{0.0f, 0.0f, 0.0f};
    const float3 bboxMax{555.0f, 555.0f, 555.0f};
    const float sceneScale = 2.0f / 555.0f;
    const float3 bboxCenter = 0.5f * (bboxMin + bboxMax);
    const float3 sceneOffset{8.0f, 0.0f, 0.0f};
    auto toWorld = [=](float3 p) { return (p - bboxCenter) * sceneScale + sceneOffset; };

    const float3 lookfrom = toWorld(float3{278.0f, 278.0f, 278.0f});
    const float3 forward{0.0f, 0.0f, 1.0f};
    const float3 up{0.0f, 1.0f, 0.0f};
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    havePbrtCamera = true;
    havePbrtSpherical = true;
    pbrtCameraLookAtWorld = lookfrom + forward;
    pbrtCameraUpRaw = up;
    pbrtBboxCenter = bboxCenter;
    pbrtSceneScale = sceneScale;
    pbrtSceneOffset = sceneOffset;
}

// D3: Spherical Camera - matches build_spherical_camera_scene()/
// build_spherical_sky() exactly: ground + an 8-sphere ring + a central
// emissive sphere, viewed as a real 360-degree equirectangular
// panorama. Natural scale (sceneScale=1.0, no rescale needed).
void MetalPocApp::buildSphericalCameraScene() {
    const float3 sceneOffset{8.0f, 0.0f, 0.0f};

    // Ground: a large flat quad (materialType 0), not CPU's own
    // radius-1000 ground SPHERE - the established overlap-avoidance
    // substitution (A5/B1/F2/B10/C1/D1/D2).
    {
        const float3 groundColor{0.4f, 0.5f, 0.3f};
        addQuad(verts, normals, uvs, materials,
                float3{-30, 0, -30} + sceneOffset, float3{30, 0, -30} + sceneOffset,
                float3{30, 0, 30} + sceneOffset, float3{-30, 0, 30} + sceneOffset,
                groundColor);
    }

    // Ring of 8 coloured spheres, radius 4, y=1 - CPU's own angle-based
    // colour formula applied exactly.
    for (int i = 0; i < 8; ++i) {
        const float angle = (float)i * (2.0f * (float)M_PI / 8.0f);
        const float cx = 4.0f * cosf(angle), cz = 4.0f * sinf(angle);
        const float3 color{0.2f + 0.5f * fabsf(cosf(angle)),
                            0.2f + 0.5f * fabsf(sinf(angle)),
                            0.5f + 0.3f * cosf(2.0f * angle)};
        TriangleMaterial mat{PackedFloat3{color.x, color.y, color.z}, 0u, 1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f};
        const float3 c = float3{cx, 1.0f, cz} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 1.0f});
        sphereMaterials.push_back(mat);
    }

    // Central emissive sphere (0,3,0), radius 0.5, diffuse_light(10,10,10) -
    // direct-hit-only (materialType 0, `emission` set, no `lightId`
    // registration) - this loader has no sphere-light NEE strategy at
    // all, the same established F2/A7 limitation (section 125/130).
    {
        const float3 lightColor{10.0f, 10.0f, 10.0f};
        TriangleMaterial mat{PackedFloat3{lightColor.x, lightColor.y, lightColor.z}, 0u,
                              1.0f, PackedFloat3{lightColor.x, lightColor.y, lightColor.z}, -1, 0.0f};
        const float3 c = float3{0.0f, 3.0f, 0.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 0.5f});
        sphereMaterials.push_back(mat);
    }

    // Sky - build_spherical_sky()'s own sky_light(color(0.3,0.5,0.9)),
    // constant colour.
    havePbrtConstantEnvLight = true;
    pbrtEnvColor = float3{0.3f, 0.5f, 0.9f};

    // Camera: lookfrom=(0,1,0) - kSphericalCameraCamera's own literal
    // value. Fixed +Z world-forward reference (D3's own registry
    // lookat sits directly below lookfrom, the SAME degenerate
    // cross(up,forward) input D7's own construction already works
    // around) - not a new construction, the identical one.
    const float3 lookfrom = float3{0.0f, 1.0f, 0.0f} + sceneOffset;
    const float3 forward{0.0f, 0.0f, 1.0f};
    const float3 up{0.0f, 1.0f, 0.0f};
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    havePbrtCamera = true;
    havePbrtSpherical = true;
    pbrtCameraLookAtWorld = lookfrom + forward;
    pbrtCameraUpRaw = up;
    pbrtBboxCenter = float3{0.0f, 0.0f, 0.0f};
    pbrtSceneScale = 1.0f;
    pbrtSceneOffset = sceneOffset;
}

// D4: Realistic Camera - matches build_realistic_camera_scene() exactly:
// checker ground (materialType 16, established flat-quad substitution)
// + a row of 5 coloured spheres at varying depth (to show bokeh) + a
// direct-hit-only overhead light sphere (the established F2/A7/D3 "no
// sphere-light NEE strategy" limitation). Natural scale (no rescale) -
// the RealisticCamera<float> constructor takes CPU's own LITERAL mm/
// world-unit inputs UNMODIFIED (this is what makes its own internal
// exit-pupil-bounds precompute reproduce CPU's identical lens system),
// with NO sceneScale multiply needed anywhere - see
// Uniforms::cameraRealistic's own comment for why D8's own Cornell-box
// case (BELOW) needs one and this one doesn't.
void MetalPocApp::buildRealisticCameraScene() {
    const float3 sceneOffset{8.0f, 0.0f, 0.0f};

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

    // 5 spheres at varying depth (z=2,3.5,5,6.5,8), radius 0.8 - CPU's
    // own literal colours applied exactly.
    {
        const float3 sphereColors[5] = {
            {0.9f, 0.2f, 0.2f}, {0.2f, 0.8f, 0.2f}, {0.2f, 0.2f, 0.9f},
            {0.8f, 0.8f, 0.2f}, {0.8f, 0.2f, 0.8f},
        };
        for (int i = 0; i < 5; ++i) {
            const float z = 2.0f + (float)i * 1.5f;
            const float3 color = sphereColors[i];
            TriangleMaterial mat{PackedFloat3{color.x, color.y, color.z}, 0u, 1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f};
            const float3 c = float3{0.0f, 1.0f, z} + sceneOffset;
            spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 0.8f});
            sphereMaterials.push_back(mat);
        }
    }

    // Overhead light sphere (0,8,5), radius 2, diffuse_light(6,6,6) -
    // direct-hit-only, no NEE (this loader has no sphere-light NEE
    // strategy at all).
    {
        const float3 lightColor{6.0f, 6.0f, 6.0f};
        TriangleMaterial mat{PackedFloat3{lightColor.x, lightColor.y, lightColor.z}, 0u,
                              1.0f, PackedFloat3{lightColor.x, lightColor.y, lightColor.z}, -1, 0.0f};
        const float3 c = float3{0.0f, 8.0f, 5.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 2.0f});
        sphereMaterials.push_back(mat);
    }

    // Camera: lookfrom=(1.65,1.07,-6.85), lookat=(1.4,1,5.5) -
    // kRealisticCameraCamera's own literal values.
    const float3 lookfrom = float3{1.65f, 1.07f, -6.85f} + sceneOffset;
    const float3 lookat = float3{1.4f, 1.0f, 5.5f} + sceneOffset;
    const float3 up{0.0f, 1.0f, 0.0f};
    const float3 forward = simd::normalize(lookat - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    havePbrtCamera = true;
    pbrtCameraLookAtWorld = lookat;
    pbrtCameraUpRaw = up;
    pbrtBboxCenter = float3{0.0f, 0.0f, 0.0f};
    pbrtSceneScale = 1.0f;
    pbrtSceneOffset = sceneOffset;

    // Real 9-element simplified Double-Gauss lens (pbrt-v4's own
    // dgauss.22deg.dat sample, hand-trimmed) - CPU's own EXACT literal
    // lens/film/aperture/focus-distance values, unmodified: constructing
    // RealisticCamera<float> with these SAME inputs reproduces the
    // identical internal (metres-space) lens system and exit-pupil-
    // bounds table CPU's own reference computes - see
    // Uniforms::cameraRealistic's own comment for the full mechanism.
    havePbrtRealisticCamera = true;
    {
        const std::vector<float> lensParams = {
             35.98738f,  1.21638f, 1.54f,  23.716f,
             11.69718f,  9.9957f,  1.0f,   17.996f,
             13.08714f, 15.9948f,  1.77f,  12.364f,
            -22.63294f,  2.7757f,  1.617f,  9.812f,
              0.0f,      2.75f,    0.0f,    7.4f,     // aperture stop
             36.3581f,   8.9722f,  1.617f, 12.7f,
            -17.8595f,   1.2f,     1.0f,   12.7f,
            100.0f,      2.9804f,  1.567f, 14.478f,
            -24.5656f,   0.0f,     1.0f,   15.0f,
        };
        RealisticCamera<float> realCam(Mat4<float>{}, /*film_x_mm=*/3.0f, /*film_y_mm=*/2.0f,
                                        /*focus_distance=*/12.4f, /*aperture_diameter_mm=*/8.0f,
                                        lensParams, /*nSamples_pupil=*/512);
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
    }
}

// D8: Realistic Camera Cornell Box - the EXACT SAME A1 Cornell box
// geometry, viewed through the SAME 9-element lens D4 uses, aperture
// scaled up to 350mm (CPU's own "display convenience for an already-
// non-physical simplified lens" choice, its own comment - keeps a
// comparable defocus-cone ANGLE at this scene's much larger scale).
// UNLIKE D4, every length RealisticCamera<float> reads back (lens
// element geometry, film half-extents, rear-element Z, exit-pupil
// bounds - everything except the dimensionless `eta` fields and the
// `degenerate` flags) is scaled by `sceneScale` AFTER construction
// (construction itself still uses CPU's own literal, UNSCALED mm/
// world-unit inputs, to reproduce CPU's identical internal lens
// system) - uniformly rescaling an entire optical system by one
// constant preserves every angle/ratio that actually determines its
// rendered look (F-number, field of view, depth-of-field blur amount),
// exactly like measuring the same lens in different units - so this
// keeps the lens system's own geometric footprint correctly matched to
// THIS loader's own rescaled (~2-unit) Cornell box instead of `sceneScale`
// being applied nowhere (an internally-consistent but WRONG-size lens
// relative to the scene) or applied inconsistently to only some fields
// (a self-contradictory lens system, likely vignetting every ray).
void MetalPocApp::buildRealisticCornellBox() {
    buildCornellBoxA1();
    const float sceneScale = 2.0f / 555.0f;

    // Camera - full override (position AND orientation both differ
    // from A1's own dead-on view): lookfrom=(278,278,-420),
    // lookat=(278,278,278).
    const float3 bboxMin{0.0f, 0.0f, 0.0f};
    const float3 bboxMax{555.0f, 555.0f, 555.0f};
    const float3 bboxCenter = 0.5f * (bboxMin + bboxMax);
    const float3 sceneOffset{8.0f, 0.0f, 0.0f};
    auto toWorld = [=](float3 p) { return (p - bboxCenter) * sceneScale + sceneOffset; };
    const float3 lookfrom = toWorld(float3{278.0f, 278.0f, -420.0f});
    const float3 lookat = toWorld(float3{278.0f, 278.0f, 278.0f});
    const float3 up{0.0f, 1.0f, 0.0f};
    const float3 forward = simd::normalize(lookat - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    havePbrtCamera = true;
    pbrtCameraLookAtWorld = lookat;
    pbrtCameraUpRaw = up;
    pbrtBboxCenter = bboxCenter;
    pbrtSceneScale = sceneScale;
    pbrtSceneOffset = sceneOffset;

    // Same lens prescription as D4, CPU's own literal (unscaled) inputs
    // for construction; aperture diameter scaled up to 350mm per CPU's
    // own comment (a display choice, not a physical one).
    havePbrtRealisticCamera = true;
    {
        const std::vector<float> lensParams = {
             35.98738f,  1.21638f, 1.54f,  23.716f,
             11.69718f,  9.9957f,  1.0f,   17.996f,
             13.08714f, 15.9948f,  1.77f,  12.364f,
            -22.63294f,  2.7757f,  1.617f,  9.812f,
              0.0f,      2.75f,    0.0f,    7.4f,     // aperture stop
             36.3581f,   8.9722f,  1.617f, 12.7f,
            -17.8595f,   1.2f,     1.0f,   12.7f,
            100.0f,      2.9804f,  1.567f, 14.478f,
            -24.5656f,   0.0f,     1.0f,   15.0f,
        };
        RealisticCamera<float> realCam(Mat4<float>{}, /*film_x_mm=*/3.0f, /*film_y_mm=*/2.0f,
                                        /*focus_distance=*/420.0f, /*aperture_diameter_mm=*/350.0f,
                                        lensParams, /*nSamples_pupil=*/512);
        realisticLensElements.clear();
        for (int i = 0; i < realCam.num_elements(); ++i) {
            realisticLensElements.push_back(GpuLensElementData{
                realCam.lens_curvature_radius(i) * sceneScale, realCam.lens_thickness(i) * sceneScale,
                realCam.lens_eta(i), realCam.lens_aperture_radius(i) * sceneScale});
        }
        realisticExitPupilBounds.clear();
        for (int i = 0; i < realCam.num_exit_pupil_bounds(); ++i) {
            realisticExitPupilBounds.push_back(GpuExitPupilBoundsData{
                realCam.exit_pupil_xmin(i) * sceneScale, realCam.exit_pupil_xmax(i) * sceneScale,
                realCam.exit_pupil_ymin(i) * sceneScale, realCam.exit_pupil_ymax(i) * sceneScale,
                realCam.exit_pupil_degenerate(i) ? 1u : 0u});
        }
        realisticFilmHalfX = realCam.film_half_x() * sceneScale;
        realisticFilmHalfY = realCam.film_half_y() * sceneScale;
        realisticLensRearZ = realCam.lens_rear_z() * sceneScale;
    }
}

