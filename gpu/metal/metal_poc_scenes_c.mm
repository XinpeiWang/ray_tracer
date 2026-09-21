// metal_poc_scenes_c.mm
// Category C (Lights) hand-authored scene builders -
// MetalPocApp::buildXxx() out-of-class definitions, split out of
// metal_poc.mm as a pure code-motion refactor (no behaviour change) -
// see metal_poc_app.h's own header comment and docs/METAL_GPU_FEASIBILITY.md.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_poc_app.h"

std::function<float3(float3)> MetalPocApp::buildCornellNoLightWalls() {
    using namespace cornell_box_data;
    const float3 bboxMin{0.0f, 0.0f, 0.0f};
    const float3 bboxMax{555.0f, 555.0f, 555.0f};
    const float sceneScale = 2.0f / 555.0f;
    const float3 bboxCenter = 0.5f * (bboxMin + bboxMax);
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};
    auto toWorld = [=](float3 p) { return (p - bboxCenter) * sceneScale + sceneOffset; };

    // The 5 walls only - no ceiling light quad (this family is lit by
    // one punctual light instead).
    for (int i = 0; i < 5; ++i) {
        const QuadSpec& q = kQuads[i];
        const float3 Q{(float)q.Q.x, (float)q.Q.y, (float)q.Q.z};
        const float3 u{(float)q.u.x, (float)q.u.y, (float)q.u.z};
        const float3 v{(float)q.v.x, (float)q.v.y, (float)q.v.z};
        const float3 color{(float)q.color.r, (float)q.color.g, (float)q.color.b};
        addQuad(verts, normals, uvs, materials, toWorld(Q), toWorld(Q + u),
                toWorld(Q + u + v), toWorld(Q + v), color);
    }

    // White diffuse sphere.
    {
        const float3 center = toWorld(float3{190.0f, 90.0f, 190.0f});
        spheres.push_back(SphereData{PackedFloat3{center.x, center.y, center.z}, sceneScale * 90.0f});
        sphereMaterials.push_back(TriangleMaterial{PackedFloat3{0.73f, 0.73f, 0.73f}, /*materialType=*/0u,
            1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f});
    }
    // Metal accent sphere. CPU's own `metal(albedo, fuzz)` is the
    // classic Book-1 fuzzy-mirror model (reflect + fuzz*random_unit_vector),
    // a genuinely DIFFERENT, simpler model than materialType 4's real GGX
    // (no RoughnessToAlpha-style formula connects "fuzz" to a GGX alpha -
    // they're different physical parameterizations, unlike section 126's
    // own rough_metal/conductor case where an exact reconciliation
    // existed). Approximated directly as a low-roughness materialType 4
    // conductor instead (fuzz=0.1 is a near-mirror, low-roughness look),
    // verified by eye against a real --cpu render rather than derived
    // algebraically - an honest Approx-tier substitution, not an exact
    // port.
    {
        const float3 metalColor{0.8f, 0.8f, 0.9f};
        const float3 center = toWorld(float3{370.0f, 120.0f, 380.0f});
        const float alpha = 0.1f;
        const float3 k = reflectanceToConductorK(metalColor);
        TriangleMaterial mat{PackedFloat3{metalColor.x, metalColor.y, metalColor.z}, /*materialType=*/4u,
            /*ior=*/alpha, PackedFloat3{0, 0, 0}, -1, /*roughness=*/alpha};
        mat.conductorEta = PackedFloat3{1, 1, 1};
        mat.conductorK = PackedFloat3{k.x, k.y, k.z};
        spheres.push_back(SphereData{PackedFloat3{center.x, center.y, center.z}, sceneScale * 120.0f});
        sphereMaterials.push_back(mat);
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

    // Real per-scene flat background - pure BLACK (0,0,0), matching
    // every C2-C6 registry row exactly. A real bug found here BEFORE
    // ever comparing renders would have hidden it: this room's own
    // walls (like every Cornell-family scene's) have no front wall -
    // the camera looks in through an open front, so a genuinely
    // ESCAPING ray (an indirect bounce, or a grazing camera ray past
    // the spheres) reaches the miss path and would otherwise read
    // metal_poc.metal's own hardcoded blue-sky gradient instead of
    // black. A1/B2/etc never needed this because their own dominant
    // ceiling-light illumination masks a small sky leak; this family's
    // OWN single, far more concentrated punctual light does not - the
    // leak reads as "the whole room is uniformly, implausibly brighter
    // than a real --cpu render of the same scene_id," not a subtle
    // colour cast, caught by exactly that direct comparison (section
    // 134's own verification note).
    havePbrtConstantEnvLight = true;
    pbrtEnvColor = float3{0.0f, 0.0f, 0.0f};

    return toWorld;
}

// C2: Spotlight Cornell - matches CPU's own build_spotlight_cornell()/
// build_spotlight_punct() exactly: a spotlight aimed straight down from
// the ceiling centre, 30-degree total cone, 15-degree falloff start.
// Intensity needs the SAME `sceneScale^2` compensation section 87's own
// pbrt punctual-light finding established (this scene's own intensity,
// 600000.0, is calibrated for the ORIGINAL 555-unit scale's 1/r^2
// falloff, not this app's rescaled ~2-unit one).
void MetalPocApp::buildSpotlightCornell() {
    auto toWorld = buildCornellNoLightWalls();
    const float sceneScale = 2.0f / 555.0f;
    const float3 pos = toWorld(float3{278.0f, 548.0f, 278.0f});
    const float3 dir{0.0f, -1.0f, 0.0f};
    const float3 emission = float3{1.0f, 0.95f, 0.85f} * 600000.0f * sceneScale * sceneScale;
    pointLights.push_back(PointLightData{
        PackedFloat3{pos.x, pos.y, pos.z},
        PackedFloat3{emission.x, emission.y, emission.z},
        PackedFloat3{dir.x, dir.y, dir.z},
        /*cosOuterAngle=*/cosf(30.0f * (float)M_PI / 180.0f),
        /*cosInnerAngle=*/cosf(15.0f * (float)M_PI / 180.0f)});
}

// C3: Distant Light Cornell - matches CPU's own
// build_distant_light_cornell()/build_distant_light_punct() exactly: a
// parallel sun-like light. A REAL sign-convention bug found and fixed
// via direct CPU comparison (not assumed from a doc comment alone,
// which turned out to be self-contradictory - see below): a first
// version passed `add_distant()`'s own `dir` argument straight through
// to `DirectionalLightData::direction` unchanged, reasoning (from
// `punctual_light_objects.h`'s own wrapper comment, "unit direction
// *toward* the scene") that it was already this loader's own "light's
// direction of TRAVEL" convention - this rendered an almost completely
// BLACK room (every surface facing away from the light). The ACTUAL
// authoritative source, `src/shared/punctual_lights.h`'s own
// `DistantLightData<T>` (the struct the CPU wrapper actually
// constructs), says the opposite one line later: `dir_x/y/z` is
// "unit world-space direction TOWARD THE SCENE (wi)" as a field
// comment, but `sample_wi()` right below it says "Direction toward
// LIGHT = dir" - `dir` is genuinely `wi` (pointing FROM the scene
// TOWARD the light), the SAME convention pbrt's own punctual lights
// use (section 87) - needing the SAME negation before this loader's
// own `DirectionalLightData::direction` (light's direction of TRAVEL).
// No sceneScale compensation needed either way -
// `DistantLightData::eval_Li` has no 1/r^2 term to correct for.
void MetalPocApp::buildDistantLightCornell() {
    buildCornellNoLightWalls();
    const float3 wiTowardLight = simd::normalize(float3{-0.4f, -1.0f, -0.2f});
    const float3 dirOfTravel = -wiTowardLight;
    const float3 emission = float3{1.0f, 0.98f, 0.92f} * 14.0f;
    directionalLights.push_back(DirectionalLightData{
        PackedFloat3{dirOfTravel.x, dirOfTravel.y, dirOfTravel.z},
        PackedFloat3{emission.x, emission.y, emission.z}});
}

// C4: Point Light Cornell - matches CPU's own build_point_light_cornell()/
// build_point_light_punct() exactly: a single overhead point light, the
// SAME sceneScale^2 compensation as C2's own spotlight.
void MetalPocApp::buildPointLightCornell() {
    auto toWorld = buildCornellNoLightWalls();
    const float sceneScale = 2.0f / 555.0f;
    const float3 pos = toWorld(float3{278.0f, 540.0f, 278.0f});
    const float3 emission = float3{1.0f, 0.98f, 0.90f} * 600000.0f * sceneScale * sceneScale;
    pointLights.push_back(PointLightData{
        PackedFloat3{pos.x, pos.y, pos.z},
        PackedFloat3{emission.x, emission.y, emission.z}});
}

// C5: Goniometric Light Cornell - matches CPU's own
// build_goniometric_light_scene()/build_goniometric_punct() exactly: a
// synthetic 16x8 greyscale profile (brighter toward the "bottom
// hemisphere," `0.2 + 0.8*t` where `t = row/rows`), same sceneScale^2
// intensity compensation as C2/C4's own point/spot lights. Uploaded via
// the pbrtGoniometricTexture path (see this struct's own
// buildGoniometricLightCornell()/buildProjectionLightCornell()
// declaration comment for why that slot, not the room's own separate
// one).
void MetalPocApp::buildGoniometricLightCornell() {
    auto toWorld = buildCornellNoLightWalls();
    const float sceneScale = 2.0f / 555.0f;

    const int NU = 16, NV = 8;
    pbrtGoniometricImageWidth = NU;
    pbrtGoniometricImageHeight = NV;
    pbrtGoniometricImagePixels.assign((size_t)NU * NV * 3, 0.0f);
    for (int v = 0; v < NV; ++v) {
        const float t = (float)v / (float)NV;
        const float value = 0.2f + 0.8f * t;
        for (int u = 0; u < NU; ++u) {
            const size_t idx = ((size_t)v * NU + u) * 3;
            pbrtGoniometricImagePixels[idx + 0] = value;
            pbrtGoniometricImagePixels[idx + 1] = value;
            pbrtGoniometricImagePixels[idx + 2] = value;
        }
    }
    havePbrtGoniometricImage = true;

    const float3 pos = toWorld(float3{278.0f, 520.0f, 278.0f});
    const float3 target = toWorld(float3{278.0f, 0.0f, 278.0f});  // identity rotation - looks straight down
    const float3 worldUp{0.0f, 0.0f, 1.0f};  // any axis not parallel to (target-pos) works for a look-down light
    const float3 emission = float3{1.0f, 0.9f, 0.7f} * 600000.0f * sceneScale * sceneScale;
    GoniometricLightData light = makeGoniometricLight(pos, target, worldUp, emission, /*scale=*/1.0f);
    light.usePbrtTexture = 1u;
    goniometricLights.push_back(light);
}

// C6: Projection Light Cornell - matches CPU's own
// build_projection_light_scene()/build_projection_punct() exactly: a
// synthetic 8x8 RGB checkerboard slide (1.0/0.05 alternating), 40-degree
// FOV, aimed from just outside the box's own open front straight down
// +Z (CPU's own `wtl` is the identity rotation, i.e. "projector looks
// +Z in world" - reproduced here by aiming at a point directly along
// +Z from the light's own position, same effect without needing a
// separate rotation-matrix path this loader doesn't have for a hand-
// authored light anyway). Needs the SAME `sceneScale^2` compensation as
// C2/C4/C5's own point/spot/goniometric lights - checked directly, not
// assumed, against `src/shared/projection_light.h`'s own `eval_Li()`
// (`Lr = r * inv_r2`) before writing this: unlike distant light, a
// projection light DOES have a real 1/r^2 falloff term, and Metal's own
// `metal_poc.metal` shading loop divides by `pjDistSq` at every one of
// its own NEE call sites too - the two backends agree on this, so the
// same compensation formula sections 87/134 already established
// applies unchanged.
void MetalPocApp::buildProjectionLightCornell() {
    auto toWorld = buildCornellNoLightWalls();

    const int NX = 8, NY = 8;
    pbrtProjectionImageWidth = NX;
    pbrtProjectionImageHeight = NY;
    pbrtProjectionImagePixels.assign((size_t)NX * NY * 3, 0.0f);
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            const float value = ((x + y) % 2 == 0) ? 1.0f : 0.05f;
            const size_t idx = ((size_t)y * NX + x) * 3;
            pbrtProjectionImagePixels[idx + 0] = value;
            pbrtProjectionImagePixels[idx + 1] = value;
            pbrtProjectionImagePixels[idx + 2] = value;
        }
    }
    havePbrtProjectionImage = true;

    const float sceneScale = 2.0f / 555.0f;
    const float3 pos = toWorld(float3{278.0f, 278.0f, -50.0f});
    const float3 target = toWorld(float3{278.0f, 278.0f, 278.0f});  // +Z, matching CPU's own identity wtl
    const float3 worldUp{0.0f, 1.0f, 0.0f};
    ProjectionLightData light = makeProjectionLight(pos, target, worldUp,
        /*fovDegrees=*/40.0f, /*aspect=*/1.0f, /*scale=*/1000000.0f * sceneScale * sceneScale);
    light.usePbrtTexture = 1u;
    projectionLights.push_back(light);
}

// F2: Triangle Mesh - matches CPU's own build_triangle_mesh_scene()
// exactly: a procedurally-generated regular icosahedron (12 vertices at
// golden-ratio coordinates, 20 triangular faces, no per-vertex normals -
// CPU's own triangle::hit() falls back to flat per-face geometric
// normals for exactly this reason, matching this loader's own addQuad()
// convention of one flat normal per face already). Ground is a flat
// quad (materialType 16, real 3D checker), not CPU's own radius-1000
// sphere - the SAME clearance fix sections 124/130 already established.
// The metal material approximates CPU's own simple `metal(albedo,
// fuzz=0.15)` the same honest way section 134's own accent sphere did
// (no algebraic reconciliation exists between "fuzz" and GGX alpha).
// The overhead light sphere is direct-hit-only (materialType 0,
// `emission` set, no NEE registration) - this loader has no sphere-
// light NEE strategy at all (A7's own established limitation, section
// 125), even though CPU's own registry row DOES register it for NEE.

void MetalPocApp::buildHdriSky() {
    // Spheres sit at raw x=-3/0/3 (radius 1) - comfortably clear of the
    // hardcoded POC room's own [-1,1] cube even at the series' usual +8
    // offset (unlike B10's own x=-6 sphere, which needed +10 - see that
    // scene's own comment), so no extra clearance is needed here.
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};

    // Ground: a large flat quad (materialType 0, matching CPU's own
    // lambertian(0.4,0.4,0.4)) - not CPU's own radius-1000 ground
    // SPHERE, which would algebraically overlap the hardcoded room's
    // own [-1,1] cube (the same substitution A5/B1/F2/B10 already
    // established).
    {
        const float3 groundColor{0.4f, 0.4f, 0.4f};
        addQuad(verts, normals, uvs, materials,
                float3{-50, 0, -50} + sceneOffset, float3{50, 0, -50} + sceneOffset,
                float3{50, 0, 50} + sceneOffset, float3{-50, 0, 50} + sceneOffset,
                groundColor);
    }

    // Sphere 1 (x=-3): diffuse, lambertian(0.7,0.3,0.2).
    {
        const float3 color{0.7f, 0.3f, 0.2f};
        TriangleMaterial mat{PackedFloat3{color.x, color.y, color.z}, /*materialType=*/0u,
                              1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f};
        const float3 c = float3{-3.0f, 1.0f, 0.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 1.0f});
        sphereMaterials.push_back(mat);
    }
    // Sphere 2 (x=0): CPU's own simple metal(color(0.8,0.8,0.9),
    // fuzz=0.05) - approximated as a real GGX conductor (materialType
    // 4) via reflectanceToConductorK(), the same faithful substitution
    // B2/G1-G3 already established. Isotropic: ior(alphaX) and
    // roughness(alphaY) both set directly to CPU's own fuzz value (the
    // same direct sqrt(alpha)-scale mapping the metal accent sphere in
    // buildMeshGalleryScene() already uses for its own small fuzz
    // value).
    {
        const float3 color{0.8f, 0.8f, 0.9f};
        TriangleMaterial mat{PackedFloat3{color.x, color.y, color.z}, /*materialType=*/4u,
                              /*ior(alphaX)=*/0.05f, PackedFloat3{0, 0, 0}, -1, /*roughness(alphaY)=*/0.05f};
        const float3 k = reflectanceToConductorK(color);
        mat.conductorEta = PackedFloat3{1.0f, 1.0f, 1.0f};
        mat.conductorK = PackedFloat3{k.x, k.y, k.z};
        const float3 c = float3{0.0f, 1.0f, 0.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 1.0f});
        sphereMaterials.push_back(mat);
    }
    // Sphere 3 (x=3): smooth dielectric, dielectric(1.5). `color` is
    // reinterpreted as a Beer-Lambert absorption COEFFICIENT for this
    // materialType, not a reflectance tint - {0,0,0} means zero
    // absorption, true clear glass (see applyBeerLambertAbsorption()'s
    // own comment) - a {1,1,1} "white" value here would make the glass
    // strongly absorptive/dark instead, matching nothing CPU's own
    // dielectric(1.5) (no absorption at all) does.
    {
        TriangleMaterial mat{PackedFloat3{0.0f, 0.0f, 0.0f}, /*materialType=*/2u,
                              /*ior=*/1.5f, PackedFloat3{0, 0, 0}, -1, 0.0f};
        const float3 c = float3{3.0f, 1.0f, 0.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 1.0f});
        sphereMaterials.push_back(mat);
    }

    // The sky: a 64x32 procedural HDR gradient, matching CPU's own
    // build_hdri_sky()'s per-pixel formula exactly (t = y/(H-1), 0 at
    // the top row down to 1 at the bottom row - r/g/b below are each
    // uniform across a row, varying only with t). Wraps the SAME
    // image-based infinite-light path loadPbrtScene() already wires up
    // generically (texture upload + importance-sampling CDF) - no
    // separate scale multiplier needed here (CPU's own sky_light(...)
    // call uses scale=1.0).
    {
        const int W = 64, H = 32;
        std::vector<float> pixels(W * H * 3);
        for (int y = 0; y < H; ++y) {
            const float t = (float)y / (float)(H - 1);
            const float r = 0.1f + 0.9f * t * t;
            const float g = 0.3f + 0.4f * (1.0f - fabsf(t - 0.5f) * 2.0f);
            const float b = 0.8f * (1.0f - t * t);
            for (int x = 0; x < W; ++x) {
                float* p = &pixels[(y * W + x) * 3];
                p[0] = r; p[1] = g; p[2] = b;
            }
        }
        havePbrtImageEnvLight = true;
        pbrtEnvImageWidth = W;
        pbrtEnvImageHeight = H;
        pbrtEnvImagePixels = std::move(pixels);
    }

    // Camera: fov=42, lookfrom=(0,2.3,15), lookat=(0,1,0) - matches
    // build_hdri_sky_world()'s own CameraConfig exactly.
    const float3 lookfrom = float3{0.0f, 2.3f, 15.0f} + sceneOffset;
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

// C7: Portal Infinite Light - matches build_portal_light_scene()/
// build_portal_sky() exactly. Same ~555-unit Cornell-box rescale/
// recentre/offset convention buildCornellBoxA1() already uses (this
// scene is authored at the identical scale) - deliberately NOT built
// through buildCornellFamilyScene(), since this room differs from
// every Cornell-family scene structurally (no ceiling light, no white
// box, a solid back wall replaced by 4 border quads around a window),
// not just by a swapped sphere material.
void MetalPocApp::buildPortalLightScene() {
    const float3 bboxMin{0.0f, 0.0f, 0.0f};
    const float3 bboxMax{555.0f, 555.0f, 555.0f};
    const float maxExtent = 555.0f;
    const float sceneScale = 2.0f / maxExtent;
    const float3 bboxCenter = 0.5f * (bboxMin + bboxMax);
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};
    auto toWorld = [=](float3 p) { return (p - bboxCenter) * sceneScale + sceneOffset; };

    // Right/left/ceiling/floor - same 4 walls buildCornellBoxA1()'s own
    // kQuads carries, just without their shared ceiling light (this
    // room's only light is the sky through the window below) and
    // without a front wall (the same "open front" convention every
    // Cornell-family scene here already uses).
    struct QuadDef { float3 Q, u, v; float3 color; };
    const QuadDef kWalls[4] = {
        {{555, 0, 0},   {0, 0, 555},  {0, 555, 0}, {0.12f, 0.45f, 0.15f}},  // right (green)
        {{0, 0, 555},   {0, 0, -555}, {0, 555, 0}, {0.65f, 0.05f, 0.05f}},  // left (red)
        {{0, 555, 0},   {555, 0, 0},  {0, 0, 555}, {0.73f, 0.73f, 0.73f}},  // ceiling (white)
        {{0, 0, 555},   {555, 0, 0},  {0, 0, -555}, {0.73f, 0.73f, 0.73f}}, // floor (white)
    };
    for (const QuadDef& q : kWalls) {
        const float3 a = toWorld(q.Q), b = toWorld(q.Q + q.u);
        const float3 c = toWorld(q.Q + q.u + q.v), d = toWorld(q.Q + q.v);
        addQuad(verts, normals, uvs, materials, a, b, c, d, q.color);
    }

    // Back wall with a 245x245 window cut into it (centred in the
    // 555x555 wall), built from 4 border quads around the opening -
    // matches build_portal_light_scene()'s own comment exactly.
    const QuadDef kWindowBorder[4] = {
        {{555, 400, 555}, {-555, 0, 0}, {0, 155, 0}, {0.73f, 0.73f, 0.73f}},  // top strip
        {{555, 0, 555},   {-555, 0, 0}, {0, 155, 0}, {0.73f, 0.73f, 0.73f}},  // bottom strip
        {{555, 155, 555}, {-155, 0, 0}, {0, 245, 0}, {0.73f, 0.73f, 0.73f}},  // right-of-window strip
        {{155, 155, 555}, {-155, 0, 0}, {0, 245, 0}, {0.73f, 0.73f, 0.73f}},  // left-of-window strip
    };
    for (const QuadDef& q : kWindowBorder) {
        const float3 a = toWorld(q.Q), b = toWorld(q.Q + q.u);
        const float3 c = toWorld(q.Q + q.u + q.v), d = toWorld(q.Q + q.v);
        addQuad(verts, normals, uvs, materials, a, b, c, d, q.color);
    }

    // Sphere: point3(190,100,190), radius 100, CPU's own simple
    // metal(color(0.8,0.8,0.9), fuzz=0.05) - approximated as a real GGX
    // conductor (materialType 4) via reflectanceToConductorK(), the
    // same B2/C1 substitution, isotropic ior(alphaX)/roughness(alphaY)
    // both set directly to CPU's own fuzz value.
    {
        const float3 color{0.8f, 0.8f, 0.9f};
        TriangleMaterial mat{PackedFloat3{color.x, color.y, color.z}, /*materialType=*/4u,
                              /*ior(alphaX)=*/0.05f, PackedFloat3{0, 0, 0}, -1, /*roughness(alphaY)=*/0.05f};
        const float3 k = reflectanceToConductorK(color);
        mat.conductorEta = PackedFloat3{1.0f, 1.0f, 1.0f};
        mat.conductorK = PackedFloat3{k.x, k.y, k.z};
        const float3 center = toWorld(float3{190.0f, 100.0f, 190.0f});
        spheres.push_back(SphereData{PackedFloat3{center.x, center.y, center.z}, sceneScale * 100.0f});
        sphereMaterials.push_back(mat);
    }

    // Sky visible through the window: a CONSTANT-colour light
    // (build_portal_sky()'s own sky_light(color(0.55,0.65,0.85))) - not
    // an image, so this is even simpler than C1's own HDRI case; no
    // image buffer to generate, just the same havePbrtConstantEnvLight/
    // pbrtEnvColor fields buildPrincipledShowcase()'s own dark-ambient
    // background already uses, here as the scene's ONLY light rather
    // than a background supplementing an area light.
    havePbrtConstantEnvLight = true;
    pbrtEnvColor = float3{0.55f, 0.65f, 0.85f};

    // Camera: vfov=40, lookfrom=(278,278,-800), lookat=(278,278,278) -
    // kPortalLightCamera's own literal values (scene_registry_data.h),
    // same as A1's own Cornell camera.
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

