// metal_poc_scenes_e.mm
// Category E (Volumes) hand-authored scene builders -
// MetalPocApp::buildXxx() out-of-class definitions, split out of
// metal_poc.mm as a pure code-motion refactor (no behaviour change) -
// see metal_poc_app.h's own header comment and docs/METAL_GPU_FEASIBILITY.md.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_poc_app.h"
#include "../../src/shared/rgb_nebula_generator.h"

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

// E2: Cloud Medium (section 178) - matches CPU's own
// build_cloud_medium_scene() (scenes_advanced.h) and
// gpu/optix/scene_builder.cpp's own build_cloud_medium_scene_gpu()
// exactly: a real heterogeneous, procedural Perlin-FBm-density medium
// (pbrt-v4 CloudMedium), not the flat-density constant_medium sphere
// this scene used to render as on CPU either (see that CPU function's
// own comment).
//
// Rendered here with the SAME "invisible trigger sphere, real geometry
// computed analytically/mathematically inside the kernel" convention
// A8's own materialType 28 (bounded homogeneous medium, section 176)
// established - see this file's own buildCornellSmoke() comment. The
// cloud's real bounding volume is an axis-aligned world-space box
// (cloud_min/cloud_max below); a NEW box/AABB custom-primitive geometry
// is NOT needed to render it, because the ray/box overlap needed for
// delta tracking is computed as pure math (cloudAabbSlabIntersect(),
// metal_poc_sampling.metal) against the SAME ray parameter t the
// trigger sphere's own hit already established - see that function's
// own comment for why an affine world-to-medium transform preserves t
// exactly. The trigger sphere here is sized to comfortably CONTAIN the
// box (its own half-diagonal, from the box's center) and exists only to
// get a ray into materialType 29's own kernel branch at all.
void MetalPocApp::buildCloudMediumScene() {
    // Every hand-authored scene builder is ADDITIVE on top of the
    // hardcoded default room (buildScene()'s own comment,
    // metal_poc.mm) - the SAME `sceneOffset{60,0,0}` A8/E3 (and every
    // other non-Cornell-family hand-authored scene) already apply to
    // move clear of that room's own [-1,1] region, applied here to
    // every point AND the camera. Missing on a first version of this
    // scene (E2's own cloud/ground/background spheres sat directly on
    // top of the default room's own small Cornell box and Suzanne/Spot
    // meshes) - caught immediately by comparing against `--cpu`, which
    // showed a clean cloud with no extra geometry at all.
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};

    // Ground - same literal as A2/A3/A8/E3's own huge-sphere convention.
    {
        const float3 c = float3{0.0f, -1000.0f, 0.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 1000.0f});
        sphereMaterials.push_back(TriangleMaterial{PackedFloat3{0.4f, 0.5f, 0.3f}, /*materialType=*/0u,
            1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f});
    }

    // World AABB - matches CPU's cloud_min/cloud_max exactly (offset by
    // `sceneOffset`). CloudMedium's own altitude-falloff term treats
    // medium-y=0 as the cloud's dense base and medium-y=1 as thinned to
    // nothing (pbrt-v4 convention), so the box's bottom face (world y=1)
    // reads as the cloud's base and its top (world y=4) tapers off
    // naturally.
    const float3 cloudMin = float3{-4.0f, 1.0f, -3.0f} + sceneOffset;
    const float3 cloudMax = float3{4.0f, 4.0f, 3.0f} + sceneOffset;
    const float sx = 1.0f / (cloudMax.x - cloudMin.x);
    const float sy = 1.0f / (cloudMax.y - cloudMin.y);
    const float sz = 1.0f / (cloudMax.z - cloudMin.z);

    GpuCloudMedium cloud{};
    cloud.boundsMin[0] = 0.0f; cloud.boundsMin[1] = 0.0f; cloud.boundsMin[2] = 0.0f;
    cloud.boundsMax[0] = 1.0f; cloud.boundsMax[1] = 1.0f; cloud.boundsMax[2] = 1.0f;
    cloud.worldToMediumMat[0] = sx;   cloud.worldToMediumMat[1] = 0.0f; cloud.worldToMediumMat[2] = 0.0f;
    cloud.worldToMediumMat[3] = 0.0f; cloud.worldToMediumMat[4] = sy;   cloud.worldToMediumMat[5] = 0.0f;
    cloud.worldToMediumMat[6] = 0.0f; cloud.worldToMediumMat[7] = 0.0f; cloud.worldToMediumMat[8] = sz;
    cloud.worldToMediumTranslate[0] = -cloudMin.x * sx;
    cloud.worldToMediumTranslate[1] = -cloudMin.y * sy;
    cloud.worldToMediumTranslate[2] = -cloudMin.z * sz;
    cloud.sigmaA = 0.0f;      // pure scattering, no absorption
    cloud.sigmaS = 10.0f;     // matches CPU/OptiX's own literal - see that
                               // function's own comment on why 10.0 (not
                               // the physically-motivated-but-much-slower
                               // 40.0) was chosen.
    cloud.density = 1.0f;
    cloud.wispiness = 1.0f;   // unread by gpuCloudDensity() - kept only
                               // for struct-layout parity, see
                               // GpuCloudMedium's own comment.
    cloud.frequency = 4.0f;
    const int cloudIdx = (int)cloudMediums.size();
    cloudMediums.push_back(cloud);

    TriangleMaterial cloudMat{};
    cloudMat.color = PackedFloat3{1.0f, 1.0f, 1.0f};  // albedo
    cloudMat.materialType = 29u;
    cloudMat.emission = PackedFloat3{0, 0, 0};
    cloudMat.lightId = -1;
    cloudMat.roughness = 0.3f;  // phase_g
    cloudMat.conductorEta = PackedFloat3{(float)cloudIdx, 0.0f, 0.0f};
    sphereMaterials.push_back(cloudMat);

    const float3 cloudCenter = 0.5f * (cloudMin + cloudMax);
    const float3 half = 0.5f * (cloudMax - cloudMin);
    const float triggerRadius = simd::length(half);
    spheres.push_back(SphereData{PackedFloat3{cloudCenter.x, cloudCenter.y, cloudCenter.z}, triggerRadius});

    // Background spheres for context - CPU's own exact x=+-6, z=4
    // literals.
    {
        const float3 c = float3{-6.0f, 0.5f, 4.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 0.5f});
        sphereMaterials.push_back(TriangleMaterial{PackedFloat3{0.9f, 0.3f, 0.2f}, /*materialType=*/0u,
            1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f});
    }
    {
        // CPU's own `metal(color(0.8,0.8,0.9), 0.05)` (a fuzzy Book-3
        // conductor) - approximated the SAME way A2's own static-metal
        // hero sphere already establishes (metal_poc_scenes_a.mm):
        // materialType 4 (anisotropic GGX conductor), fuzz standing in
        // for alphaX/alphaY, eta=1/k derived from the reflectance colour.
        const float3 metalColor{0.8f, 0.8f, 0.9f};
        TriangleMaterial m{PackedFloat3{metalColor.x, metalColor.y, metalColor.z}, /*materialType=*/4u,
                           /*ior(alphaX)=*/0.05f, PackedFloat3{0, 0, 0}, -1, /*roughness(alphaY)=*/0.05f};
        const float3 k = reflectanceToConductorK(metalColor);
        m.conductorEta = PackedFloat3{1.0f, 1.0f, 1.0f};
        m.conductorK = PackedFloat3{k.x, k.y, k.z};
        const float3 c = float3{6.0f, 0.5f, 4.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 0.5f});
        sphereMaterials.push_back(m);
    }

    // Background - CPU's own registry row for E2, bg (0.5,0.7,1.0).
    havePbrtConstantEnvLight = true;
    pbrtEnvColor = float3{0.5f, 0.7f, 1.0f};

    // Camera - CPU's own registry row for E2 (vfov 40, lookfrom
    // (0,4,26), lookat (0,2,0) - widened/pulled back from an earlier
    // (20deg, (0,5,20)) framing that overflowed the cloud's own AABB,
    // see scene_registry_data.h's own comment).
    const float3 lookfrom = float3{0.0f, 4.0f, 26.0f} + sceneOffset;
    const float3 lookat = float3{0.0f, 2.0f, 0.0f} + sceneOffset;
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

// E4: RGB Grid Medium (section 179) - matches CPU's own
// build_rgb_grid_medium_scene() (scenes_advanced.h) and
// gpu/optix/scene_builder.cpp's own build_rgb_grid_medium_scene_gpu()
// exactly: a real heterogeneous "nebula" whose R/G/B scattering
// coefficients are three INDEPENDENT 24x24x24 voxel grids (so the
// medium's own colour genuinely varies spatially, unlike E2's single
// grayscale Perlin field), generated by generate_nebula_channel()
// (src/shared/rgb_nebula_generator.h - real, shared, CPU-callable host
// code, called here directly rather than re-ported, since it already
// runs identically for both backends at scene-build time and only its
// OUTPUT, not the generator itself, needs to reach the GPU).
//
// Same "invisible trigger sphere, real geometry handled analytically"
// convention as E2's own materialType 29 (see that method's own
// comment) - and the SAME deliberate GPU simplification OptiX's own
// port already established (GpuRgbGridMedium's own comment,
// metal_poc_types.metal): a single GLOBAL majorant (sigmaMaj, computed
// here as the max voxel value across all 3 channels times sigmaScale,
// with a small safety margin) rather than CPU's real per-voxel DDA
// majorant grid.
void MetalPocApp::buildRgbGridMediumScene() {
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};

    // Ground - same literal as A2/A3/A8/E2/E3's own huge-sphere convention.
    {
        const float3 c = float3{0.0f, -1000.0f, 0.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 1000.0f});
        sphereMaterials.push_back(TriangleMaterial{PackedFloat3{0.4f, 0.5f, 0.3f}, /*materialType=*/0u,
            1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f});
    }

    // CPU's own exact nx/ny/nz/freq/per-channel-offset literals - same
    // values feed both backends' own generate_nebula_channel() calls, so
    // the resulting voxel data (float precision here vs CPU's own
    // double) is structurally, not necessarily bit-for-bit, the same -
    // the same "not an exact-match bar" precedent A2's own fixed-seed
    // random scattering already established.
    const int nx = 24, ny = 24, nz = 24;
    std::vector<float> ssR, ssG, ssB;
    generate_nebula_channel<float>(nx, ny, nz, 3.0f, 0.0f, 0.0f, 0.0f, ssR);
    generate_nebula_channel<float>(nx, ny, nz, 3.0f, 5.2f, 1.7f, 3.3f, ssG);
    generate_nebula_channel<float>(nx, ny, nz, 3.0f, 11.4f, 8.8f, 2.1f, ssB);

    float maxDensity = 0.0f;
    for (float v : ssR) maxDensity = std::max(maxDensity, v);
    for (float v : ssG) maxDensity = std::max(maxDensity, v);
    for (float v : ssB) maxDensity = std::max(maxDensity, v);

    const float3 worldMin = float3{-4.0f, 1.0f, -4.0f} + sceneOffset;
    const float3 worldMax = float3{4.0f, 5.0f, 4.0f} + sceneOffset;
    const float sx = 1.0f / (worldMax.x - worldMin.x);
    const float sy = 1.0f / (worldMax.y - worldMin.y);
    const float sz = 1.0f / (worldMax.z - worldMin.z);
    const float sigmaScale = 4.0f;  // matches CPU's sigma_scale

    GpuRgbGridMedium grid{};
    // Medium-space unit cube, NOT the world-space AABB - rgbGridAabbSlabIntersect()
    // (metal_poc_sampling.metal) compares these directly against the ray
    // ALREADY transformed into medium space, the same convention
    // GpuCloudMedium::boundsMin/boundsMax already uses. `worldMin`/
    // `worldMax` below feed the affine transform's own derivation only.
    grid.boundsMin[0] = 0.0f; grid.boundsMin[1] = 0.0f; grid.boundsMin[2] = 0.0f;
    grid.boundsMax[0] = 1.0f; grid.boundsMax[1] = 1.0f; grid.boundsMax[2] = 1.0f;
    grid.worldToMediumMat[0] = sx;   grid.worldToMediumMat[1] = 0.0f; grid.worldToMediumMat[2] = 0.0f;
    grid.worldToMediumMat[3] = 0.0f; grid.worldToMediumMat[4] = sy;   grid.worldToMediumMat[5] = 0.0f;
    grid.worldToMediumMat[6] = 0.0f; grid.worldToMediumMat[7] = 0.0f; grid.worldToMediumMat[8] = sz;
    grid.worldToMediumTranslate[0] = -worldMin.x * sx;
    grid.worldToMediumTranslate[1] = -worldMin.y * sy;
    grid.worldToMediumTranslate[2] = -worldMin.z * sz;
    grid.nx = nx; grid.ny = ny; grid.nz = nz;
    grid.dataOffset = (int)rgbGridData.size();
    grid.sigmaScale = sigmaScale;
    grid.sigmaMaj = maxDensity * sigmaScale * 1.01f;  // small safety margin
    grid.phaseG = 0.2f;
    const int gridIdx = (int)rgbGridMediums.size();
    rgbGridMediums.push_back(grid);
    // R-then-G-then-B, concatenated - matches GpuRgbGridMedium::dataOffset's
    // own "R block, then G at +voxelCount, then B at +2*voxelCount" layout.
    rgbGridData.insert(rgbGridData.end(), ssR.begin(), ssR.end());
    rgbGridData.insert(rgbGridData.end(), ssG.begin(), ssG.end());
    rgbGridData.insert(rgbGridData.end(), ssB.begin(), ssB.end());

    TriangleMaterial gridMat{};
    // color left default (unread) - see materialType 30's own kernel
    // comment: the per-scatter tint comes from the grid's own local
    // R/G/B ratio, not a flat albedo.
    gridMat.materialType = 30u;
    gridMat.emission = PackedFloat3{0, 0, 0};
    gridMat.lightId = -1;
    gridMat.conductorEta = PackedFloat3{(float)gridIdx, 0.0f, 0.0f};
    sphereMaterials.push_back(gridMat);

    const float3 center = 0.5f * (worldMin + worldMax);
    const float3 half = 0.5f * (worldMax - worldMin);
    const float triggerRadius = simd::length(half);
    spheres.push_back(SphereData{PackedFloat3{center.x, center.y, center.z}, triggerRadius});

    // Context spheres - CPU's own exact x=+-6, z=4 literals (E4's box is
    // taller than E2's own, but the context spheres sit at the same spot).
    {
        const float3 c = float3{-6.0f, 0.5f, 4.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 0.5f});
        sphereMaterials.push_back(TriangleMaterial{PackedFloat3{0.9f, 0.3f, 0.2f}, /*materialType=*/0u,
            1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f});
    }
    {
        // Same "CPU's fuzzy metal(color,0.05) approximated via
        // materialType 4" substitution E2's own background sphere uses.
        const float3 metalColor{0.8f, 0.8f, 0.9f};
        TriangleMaterial m{PackedFloat3{metalColor.x, metalColor.y, metalColor.z}, /*materialType=*/4u,
                           /*ior(alphaX)=*/0.05f, PackedFloat3{0, 0, 0}, -1, /*roughness(alphaY)=*/0.05f};
        const float3 k = reflectanceToConductorK(metalColor);
        m.conductorEta = PackedFloat3{1.0f, 1.0f, 1.0f};
        m.conductorK = PackedFloat3{k.x, k.y, k.z};
        const float3 c = float3{6.0f, 0.5f, 4.0f} + sceneOffset;
        spheres.push_back(SphereData{PackedFloat3{c.x, c.y, c.z}, 0.5f});
        sphereMaterials.push_back(m);
    }

    // Background - CPU's own registry row for E4, bg (0.5,0.7,1.0).
    havePbrtConstantEnvLight = true;
    pbrtEnvColor = float3{0.5f, 0.7f, 1.0f};

    // Camera - CPU's own registry row for E4 (vfov 45, lookfrom
    // (0,5,30), lookat (0,3,0) - pulled back further than E2's own,
    // since this box is taller and the context spheres sit further out).
    const float3 lookfrom4 = float3{0.0f, 5.0f, 30.0f} + sceneOffset;
    const float3 lookat4 = float3{0.0f, 3.0f, 0.0f} + sceneOffset;
    const float3 up4{0.0f, 1.0f, 0.0f};
    const float3 forward4 = simd::normalize(lookat4 - lookfrom4);
    const float3 right4 = simd::normalize(simd::cross(forward4, up4));
    const float3 trueUp4 = simd::cross(right4, forward4);
    pbrtCameraPos = lookfrom4;
    pbrtCameraForward = forward4;
    pbrtCameraRight = right4;
    pbrtCameraUp = trueUp4;
    pbrtTanHalfFov = tanf(0.5f * 45.0f * (float)M_PI / 180.0f);
    havePbrtCamera = true;
    pbrtCameraLookAtWorld = lookat4;
    pbrtCameraUpRaw = up4;
    pbrtBboxCenter = float3{0.0f, 0.0f, 0.0f};
    pbrtSceneScale = 1.0f;
    pbrtSceneOffset = sceneOffset;
}

