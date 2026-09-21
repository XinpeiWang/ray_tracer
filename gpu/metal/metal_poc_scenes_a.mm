// metal_poc_scenes_a.mm
// Category A (Basics) hand-authored scene builders -
// MetalPocApp::buildXxx() out-of-class definitions, split out of
// metal_poc.mm as a pure code-motion refactor (no behaviour change) -
// see metal_poc_app.h's own header comment and docs/METAL_GPU_FEASIBILITY.md.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_poc_app.h"

void MetalPocApp::buildCornellBoxA1() {
    using namespace cornell_box_data;
    // Same rescale/recentre/offset convention loadPbrtScene() uses for
    // every real pbrt scene (see that function's own comment) - this data
    // is ALREADY authored at the identical ~555-unit Cornell-box scale a
    // real pbrt Cornell box uses, so the exact same fixed transform
    // applies unchanged: largest dimension -> 2.0 units, recentred on the
    // room's own bounding-box centre, offset +60 in X (section 169) clear
    // of the hardcoded POC room's own [-1,1] region.
    const float3 bboxMin{0.0f, 0.0f, 0.0f};
    const float3 bboxMax{555.0f, 555.0f, 555.0f};
    const float maxExtent = 555.0f;
    const float sceneScale = 2.0f / maxExtent;
    const float3 bboxCenter = 0.5f * (bboxMin + bboxMax);
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};
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
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};

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
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};

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
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};

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
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};

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

// A7: Simple Light - matches CPU's own build_simple_light() exactly:
// the SAME ground+main-sphere Perlin marble pair A5 already uses
// (materialType 17, noise scale 4), a warm emissive SPHERE light, and
// a cool emissive quad light. CPU's own registry row for this scene
// uses `no_lights` (see scene_registry_data.h) - deliberately NEITHER
// light is NEE-registered, even on CPU, so this is matched exactly
// here too: both lights are added as plain emissive geometry
// (`emission` set, `lightId` left at -1), relying purely on direct
// hits/BSDF-sampled bounces landing on them, the SAME "emissive but not
// NEE-registered" mechanism sections 100/101 already established for
// non-quad emissive shapes - not a missing feature, a real, deliberate
// match of CPU's own choice for this specific scene. bg (0,0,0) - a
// genuinely BLACK background (no ambient sky at all, unlike every
// earlier hand-authored scene's own flat sky colour) - still the same
// havePbrtConstantEnvLight/pbrtEnvColor mechanism, just set to black.
void MetalPocApp::buildSimpleLight() {
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};

    // Ground + main sphere - same materialType 17/noise-scale-4 pair
    // A5's own buildPerlinSpheres() already established (see that
    // function's own comment on why the ground is a flat quad, not a
    // huge sphere - the exact same radius-1000-sphere clearance issue
    // applies here too, CPU's own build_simple_light() uses an
    // identical ground sphere).
    {
        const float3 groundColor{1.0f, 1.0f, 1.0f};
        addQuad(verts, normals, uvs, materials,
                float3{-15.0f, 0.0f, -15.0f} + sceneOffset, float3{15.0f, 0.0f, -15.0f} + sceneOffset,
                float3{15.0f, 0.0f, 15.0f} + sceneOffset, float3{-15.0f, 0.0f, 15.0f} + sceneOffset,
                groundColor, /*materialType=*/17u, /*emission=*/simd::make_float3(0, 0, 0),
                /*lightId=*/-1, /*roughness=*/4.0f);
    }
    {
        TriangleMaterial mat{PackedFloat3{1.0f, 1.0f, 1.0f}, /*materialType=*/17u,
                              1.0f, PackedFloat3{0, 0, 0}, -1, /*roughness=*/4.0f};
        const float3 c = float3{0.0f, 2.0f, 0.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 2.0f});
        sphereMaterials.push_back(mat);
    }

    // Warm sphere light - emissive, NOT NEE-registered (lightId=-1),
    // matching CPU's own `no_lights` choice for this scene exactly (see
    // this function's own declaration comment).
    {
        const float3 warmColor{6.0f, 3.0f, 1.0f};
        TriangleMaterial mat{PackedFloat3{warmColor.x, warmColor.y, warmColor.z},
                             /*materialType=*/0u, 1.0f, PackedFloat3{warmColor.x, warmColor.y, warmColor.z},
                             /*lightId=*/-1, 0.0f};
        const float3 c = float3{0.0f, 7.0f, 0.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 2.0f});
        sphereMaterials.push_back(mat);
    }

    // Cool quad light - same "emissive, not NEE-registered" choice.
    {
        const float3 Q{3.5f, 1.0f, -3.0f}, u{2.0f, 0.0f, 0.0f}, v{0.0f, 2.0f, 0.0f};
        const float3 a = Q + sceneOffset, b = Q + u + sceneOffset,
                     c = Q + u + v + sceneOffset, d = Q + v + sceneOffset;
        const float3 coolColor{2.0f, 3.0f, 6.0f};
        addQuad(verts, normals, uvs, materials, a, b, c, d, coolColor,
                /*materialType=*/0u, /*emission=*/coolColor, /*lightId=*/-1);
    }

    // Real per-scene flat background - pure BLACK (0,0,0), matching
    // CPU's own registry row for A7 exactly (no ambient sky at all).
    havePbrtConstantEnvLight = true;
    pbrtEnvColor = float3{0.0f, 0.0f, 0.0f};

    // Camera: CPU's own real registry row for A7, ported directly
    // (vfov 20, lookfrom (26,3,6), lookat (0,2,0)).
    const float3 lookfrom = float3{26.0f, 3.0f, 6.0f} + sceneOffset;
    const float3 lookat = float3{0.0f, 2.0f, 0.0f} + sceneOffset;
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

// See this method's own declaration comment (this struct's own
// definition) for the shape - reuses buildCornellBoxA1()'s own real
// wall/light/rescale code verbatim, only the box's and sphere's own
// material differ, matching CPU's own `add_cornell_walls_and_main_light()`
// + per-scene box/sphere swap. Section 126, docs/METAL_GPU_FEASIBILITY.md.

