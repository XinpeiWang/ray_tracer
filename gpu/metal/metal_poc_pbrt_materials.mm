// metal_poc_pbrt_materials.mm - MetalPocApp::mapPbrtMaterial(): a flattened pbrt material -> the Metal shader's TriangleMaterial.
//
// This used to be a ~520-line lambda inside loadPbrtScene() (metal_poc_pbrt_loader.mm), which made that function ~700 lines and every new
// pbrt material feature a change to it. It is now its own function; loadPbrtScene() builds one PbrtMaterialMapState per scene and hands out
// `materialFor(index)` closures over it. The body is the original code, moved unchanged. Which TriangleMaterial fields carry which
// pbrt parameter (materialType 25's `conductorK.y` family, glass-with-medium, measured BRDF tables, ...) is documented where each is set below
// and in docs/METAL_PARITY_STATUS.md; the materialType names are in metal_poc_material_ids.metal.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_poc_app.h"
#include "../../src/shared/curve_tessellate.h"
#include "../../src/shared/fresnel.h"   // CauchyCoefficientsFromAbbe
#include "../../src/shared/gpu_scene_materials.h"   // reflectanceToConductorK
#include "../../src/shared/srgb_decode.h"
#include "../../src/shared/measured_bxdf_loader.h"   // MeasuredBRDFData + GetMeasuredBRDFDataCached (the CPU renderer's own cache)
#include "../../src/shared/portal_image_infinite_light.h"   // PortalImageInfiniteLightData: rectified image + sampling tables for portal[4]
#include <cstring>
#include <array>
#include <map>
#include <functional>
#include <unordered_map>
#include <unordered_set>

// A measured BRDF (pbrt "measured") for the Metal shader (materialType 32, metal_poc_materials_extra.metal): its five PiecewiseLinear2D
// tables go into the shared float buffer `g` behind a 76-int descriptor. Descriptor fields are ints stored bit-for-bit in float slots and
// every offset is an absolute index into `g`; the descriptor comes FIRST so its own index (stored in a float, exact below 2^24) stays small
// however large the tables are. Returns that index.
template <size_t Dim>
static void putMeasuredPL2D(const PiecewiseLinear2D<Dim>& t, std::vector<float>& g, size_t descBase) {
    int d[15] = {t.XSize(), t.YSize(), (int)Dim, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0, -1, -1};
    for (size_t i = 0; i < Dim; ++i) {
        d[3 + i] = (int)t.ParamRes()[i];
        d[6 + i] = (int)t.ParamStride()[i];
        d[9 + i] = (int)g.size();
        g.insert(g.end(), t.ParamValues(i).begin(), t.ParamValues(i).end());
    }
    d[12] = (int)g.size();
    g.insert(g.end(), t.Data().begin(), t.Data().end());
    if (!t.Mcdf().empty()) { d[13] = (int)g.size(); g.insert(g.end(), t.Mcdf().begin(), t.Mcdf().end()); }
    if (!t.Ccdf().empty()) { d[14] = (int)g.size(); g.insert(g.end(), t.Ccdf().begin(), t.Ccdf().end()); }
    std::memcpy(&g[descBase], d, sizeof(d));
}

static size_t appendMeasuredBrdf(const MeasuredBRDFData& m, std::vector<float>& g) {
    const size_t base = g.size();
    g.resize(base + 76, 0.0f);
    putMeasuredPL2D(m.ndf, g, base);
    putMeasuredPL2D(m.sigma, g, base + 15);
    putMeasuredPL2D(m.vndf, g, base + 30);
    putMeasuredPL2D(m.luminance, g, base + 45);
    putMeasuredPL2D(m.spectra, g, base + 60);
    const int iso = m.isotropic ? 1 : 0;
    std::memcpy(&g[base + 75], &iso, sizeof(iso));
    return base;
}

// pbrt Diffuse: flat colour, or one of the textured variants (checkerboard, procedural noise, image) that the shader implements as modes of
// materialType 25/26 (see the comments below). Split out of mapPbrtMaterial(), where this one case was 215 of its ~530 lines.
TriangleMaterial MetalPocApp::mapPbrtDiffuseMaterial(const pbrt_flatten::Material& m, float sceneScale, float3 bboxCenter, float3 sceneOffset) {
    PackedFloat3 color{(float)m.color[0], (float)m.color[1], (float)m.color[2]};
    // A "reflectance" bound to a "checkerboard" Texture (B22,
    // section 162) - materialType 25, pbrt-v4's real UV-space
    // 2-colour checker (see that materialType's own shading
    // comment, metal_poc_kernel.metal). Only the SIMPLE case
    // this loader can represent: tex1/tex2 each either a flat
    // literal or already flattened to one by
    // pbrt_flatten.h's own nestedProceduralAverageColor()
    // (m.checkerColor1/2 are ALWAYS valid flat colours by
    // this point, regardless of nesting depth - that
    // function's own comment) - a bare "imagemap" bound to
    // tex1/tex2 (m.checkerTex1Filename/checkerTex2Filename)
    // is the one sub-case this loader doesn't carry through
    // at all, unlike CPU's own full nested-texture support;
    // no bundled scene needs it (matches roughnessTextureFilename's
    // own identical "bare imagemap only" scope-narrowing,
    // pbrt_flatten.h) - falls through to the flat-colour
    // default below instead of misrendering it. Also excludes
    // m.checkerIs3D: a 3D checker is handled by its own case just above
    // (it needs a texture-space point, not UV tile frequencies).
    // pbrt-v4 procedural reflectance textures, each a mode of materialType 25 (conductorK.y):
    //   3 fbm, 4 windy, 5 wrinkled (both keyed on the pbrt-world hit point with NO scale, like CPU),
    //   6 dots (uv), 7 bilerp (uv). The world-point ones fold the scene rescale/recentre into
    //   t = k*p_metal + off exactly as marble does (k = 1/sceneScale,
    //   off = bboxCenter - sceneOffset/sceneScale; conductorK.x = k, conductorEta = off).
    //   wrinkled: transmitColor = (octaves, omega, 0). dots: color = inside, transmitColor =
    //   outside (flat colours only). bilerp: color = v00, transmitColor = v01, conductorEta = v10,
    //   v11 = (conductorK.x, roughness, conductorK.z).
    if (m.hasWindyReflectance || m.hasWrinkledReflectance || m.hasFbmReflectance) {
        TriangleMaterial mat{PackedFloat3{0.5f, 0.5f, 0.5f}, /*materialType=*/METAL_MAT_TEXTURE_FAMILY, /*ior=*/1.0f,
                             PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
        mat.conductorK = PackedFloat3{1.0f / sceneScale, m.hasFbmReflectance ? 3.0f : (m.hasWindyReflectance ? 4.0f : 5.0f), 0.0f};
        mat.conductorEta = PackedFloat3{bboxCenter.x - sceneOffset.x / sceneScale,
                                         bboxCenter.y - sceneOffset.y / sceneScale,
                                         bboxCenter.z - sceneOffset.z / sceneScale};
        if (m.hasWrinkledReflectance)
            mat.transmitColor = PackedFloat3{(float)m.wrinkledOctaves, (float)m.wrinkledRoughness, 0.0f};
        if (m.hasFbmReflectance)   // fbm (mode 3): transmitColor = (octaves, omega, 0), scale 1 like CPU
            mat.transmitColor = PackedFloat3{(float)m.fbmOctaves, (float)m.fbmRoughness, 0.0f};
        return mat;
    }
    if (m.hasDotsReflectance && m.dotsInsideTexFilename.empty() && m.dotsOutsideTexFilename.empty()) {
        TriangleMaterial mat{PackedFloat3{(float)m.dotsInsideColor[0], (float)m.dotsInsideColor[1], (float)m.dotsInsideColor[2]},
                             /*materialType=*/METAL_MAT_TEXTURE_FAMILY, /*ior=*/1.0f, PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
        mat.transmitColor = PackedFloat3{(float)m.dotsOutsideColor[0], (float)m.dotsOutsideColor[1], (float)m.dotsOutsideColor[2]};
        mat.conductorK = PackedFloat3{0.0f, 6.0f, 0.0f};
        return mat;
    }
    if (m.hasBilerpReflectance) {
        TriangleMaterial mat{PackedFloat3{(float)m.bilerpV00[0], (float)m.bilerpV00[1], (float)m.bilerpV00[2]},
                             /*materialType=*/METAL_MAT_TEXTURE_FAMILY, /*ior=*/1.0f, PackedFloat3{0, 0, 0}, /*lightId=*/-1,
                             /*roughness=*/(float)m.bilerpV11[1]};
        mat.transmitColor = PackedFloat3{(float)m.bilerpV01[0], (float)m.bilerpV01[1], (float)m.bilerpV01[2]};
        mat.conductorEta = PackedFloat3{(float)m.bilerpV10[0], (float)m.bilerpV10[1], (float)m.bilerpV10[2]};
        mat.conductorK = PackedFloat3{(float)m.bilerpV11[0], 7.0f, (float)m.bilerpV11[2]};
        return mat;
    }
    // pbrt-v4 "marble" reflectance texture (A5/A7): FBm-perturbed sine through a colour
    // spline, ported from CPU's marble_texture. Like CPU it is keyed on the (pbrt-world)
    // hit point times `marbleScale`, so the same fold as the 3D checker applies: the
    // kernel evaluates at t = k*p_metal + off, k = scale/sceneScale,
    // off = scale*(bboxCenter - sceneOffset/sceneScale). materialType 25 with
    // conductorK.y = 2 selects it; conductorK.x = k, conductorEta = off,
    // transmitColor = (octaves, omega, variation).
    if (m.hasMarbleReflectance) {
        const float s = (float)m.marbleScale;
        TriangleMaterial mat{PackedFloat3{0.5f, 0.5f, 0.5f}, /*materialType=*/METAL_MAT_TEXTURE_FAMILY, /*ior=*/1.0f,
                             PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
        mat.transmitColor = PackedFloat3{(float)m.marbleOctaves, (float)m.marbleRoughness, (float)m.marbleVariation};
        mat.conductorK = PackedFloat3{s / sceneScale, 2.0f, 0.0f};
        mat.conductorEta = PackedFloat3{s * (bboxCenter.x - sceneOffset.x / sceneScale),
                                         s * (bboxCenter.y - sceneOffset.y / sceneScale),
                                         s * (bboxCenter.z - sceneOffset.z / sceneScale)};
        return mat;
    }
    // 3D checkerboard ("integer dimension" [3]): pbrt-v4 keys the pattern on a
    // TEXTURE-SPACE point, floor(x)+floor(y)+floor(z) parity, exactly like CPU's
    // checker_texture. Supported for the only shape the bundled scenes author - a
    // pure uniform scale (checkerWorldToTexture diagonal s, no rotation/translation),
    // same restriction OptiX's builder applies. materialType 25 with
    // conductorK.y = 1 selects this mode: Metal world space is rescaled/recentred, so
    // fold that into the transform once here: t = k*p_metal + off with k = s/sceneScale
    // and off = s*(bboxCenter - sceneOffset/sceneScale) (conductorK.x = k,
    // conductorEta = off). Works on triangles and spheres alike (A2's ground is a sphere).
    // 2D checkerboard whose tex1 is a bare imagemap (J3), or a NESTED checkerboard whose own tex1
    // is one (J6, two levels): pbrt evaluates nested textures with the same (u,v), each checker
    // with its own uscale/vscale. materialType 25 mode 8 (conductorK.y = 8), using the single
    // diffuse-image slot (the first image filename wins, as for materialType 26):
    //   color = outer tex2, transmitColor = nested tex2, conductorEta = (outer uscale, vscale,
    //   nested?1:0), conductorK = (nested uscale, 8, nested vscale).
    if (m.hasCheckerReflectance && !m.checkerIs3D && m.checkerTex2Filename.empty()) {
        std::string imgFile;
        bool nestedLevel = false;
        double nU = 1.0, nV = 1.0;
        double nTex2[3] = {0, 0, 0};
        const pbrt_flatten::NestedProceduralTexture& nt = m.checkerTex1Nested;
        if (!m.checkerTex1Filename.empty()) {
            imgFile = m.checkerTex1Filename;
        } else if (nt.kind == "checkerboard" && !nt.tex1Filename.empty() && nt.tex2Filename.empty()) {
            imgFile = nt.tex1Filename; nestedLevel = true; nU = nt.uscale; nV = nt.vscale;
            for (int c = 0; c < 3; ++c) nTex2[c] = nt.color2[c];
        }
        if (!imgFile.empty() && (!havePbrtDiffuseImage || imgFile == pbrtDiffuseImageFilename)) {
            if (!havePbrtDiffuseImage) {
                std::string bytes;
                if (pbrt_load::loadFileNear(pbrtScenePath, imgFile, bytes) &&
                    pbrt_load::detail::decodeInfiniteLightImage(imgFile, bytes,
                        pbrtDiffuseImagePixels, pbrtDiffuseImageWidth, pbrtDiffuseImageHeight)) {
                    havePbrtDiffuseImage = true;
                    pbrtDiffuseImageFilename = imgFile;
                } else {
                    fprintf(stderr, "loadPbrtScene: checkerboard's image texture '%s' could not be "
                                    "read/decoded; falling back to its flat colour\n", imgFile.c_str());
                }
            }
            if (havePbrtDiffuseImage) {
                TriangleMaterial mat{
                    PackedFloat3{(float)m.checkerColor2[0], (float)m.checkerColor2[1], (float)m.checkerColor2[2]},
                    /*materialType=*/METAL_MAT_TEXTURE_FAMILY, /*ior=*/1.0f, PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
                mat.transmitColor = PackedFloat3{(float)nTex2[0], (float)nTex2[1], (float)nTex2[2]};
                mat.conductorEta = PackedFloat3{(float)m.checkerUScale, (float)m.checkerVScale, nestedLevel ? 1.0f : 0.0f};
                mat.conductorK = PackedFloat3{(float)nU, 8.0f, (float)nV};
                return mat;
            }
        }
    }
    if (m.hasCheckerReflectance && m.checkerIs3D &&
        m.checkerTex1Filename.empty() && m.checkerTex2Filename.empty()) {
        const double* w2t = m.checkerWorldToTexture;
        const bool uniformScale =
            fabs(w2t[1]) < 1e-9 && fabs(w2t[2]) < 1e-9 && fabs(w2t[3]) < 1e-9 &&
            fabs(w2t[4]) < 1e-9 && fabs(w2t[6]) < 1e-9 && fabs(w2t[7]) < 1e-9 &&
            fabs(w2t[8]) < 1e-9 && fabs(w2t[9]) < 1e-9 && fabs(w2t[11]) < 1e-9 &&
            fabs(w2t[0] - w2t[5]) < 1e-6 && fabs(w2t[0] - w2t[10]) < 1e-6;
        if (uniformScale) {
            const float s = (float)w2t[0];
            TriangleMaterial mat{
                PackedFloat3{(float)m.checkerColor1[0], (float)m.checkerColor1[1], (float)m.checkerColor1[2]},
                /*materialType=*/METAL_MAT_TEXTURE_FAMILY, /*ior=*/1.0f, PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
            mat.transmitColor = PackedFloat3{(float)m.checkerColor2[0], (float)m.checkerColor2[1], (float)m.checkerColor2[2]};
            const float k = s / sceneScale;
            mat.conductorK = PackedFloat3{k, 1.0f, 0.0f};
            mat.conductorEta = PackedFloat3{s * (bboxCenter.x - sceneOffset.x / sceneScale),
                                             s * (bboxCenter.y - sceneOffset.y / sceneScale),
                                             s * (bboxCenter.z - sceneOffset.z / sceneScale)};
            return mat;
        }
    }
    if (m.hasCheckerReflectance && !m.checkerIs3D &&
        m.checkerTex1Filename.empty() && m.checkerTex2Filename.empty()) {
        TriangleMaterial mat{
            PackedFloat3{(float)m.checkerColor1[0], (float)m.checkerColor1[1], (float)m.checkerColor1[2]},
            /*materialType=*/METAL_MAT_TEXTURE_FAMILY, /*ior=*/1.0f, PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
        mat.transmitColor = PackedFloat3{(float)m.checkerColor2[0], (float)m.checkerColor2[1], (float)m.checkerColor2[2]};
        mat.conductorEta = PackedFloat3{(float)m.checkerUScale, (float)m.checkerVScale, 0.0f};
        return mat;
    }
    // A "reflectance" bound to a bare "imagemap" Texture
    // (F5/F9, section 166) - materialType 26, a real per-hit
    // image lookup via pbrtDiffuseTexture (see that
    // materialType's own shading comment,
    // metal_poc_kernel.metal). Only the FIRST DISTINCT such
    // texture FILENAME in the scene gets the one dedicated
    // texture slot (same "one shared texture" constraint
    // every other pbrt-loaded image here already has) - a
    // second, DIFFERENT filename, or a decode failure, falls
    // through to the flat-colour default below exactly as if
    // no texture were bound at all. Deliberately checked
    // against the ALREADY-LOADED filename, not just "already
    // loaded something" - `mapPbrtMaterial()` is called once per
    // TRIANGLE, not once per distinct Material, so a single
    // 2-triangle quad sharing ONE textured material calls
    // this twice for the SAME filename; a naive "first call
    // wins, every later call falls back" check (this
    // function's own first version) gave the quad's second
    // triangle a flat grey fallback instead of the same
    // texture its first triangle correctly got - a real,
    // visible bug (a hard diagonal split down the middle of
    // what should be one seamlessly textured quad), caught
    // by rendering and comparing against `--cpu`, not
    // assumed safe from the code alone.
    if (!m.textureFilename.empty() &&
        (!havePbrtDiffuseImage || m.textureFilename == pbrtDiffuseImageFilename)) {
        if (!havePbrtDiffuseImage) {
            std::string bytes;
            if (pbrt_load::loadFileNear(pbrtScenePath, m.textureFilename, bytes) &&
                pbrt_load::detail::decodeInfiniteLightImage(m.textureFilename, bytes,
                    pbrtDiffuseImagePixels, pbrtDiffuseImageWidth, pbrtDiffuseImageHeight)) {
                havePbrtDiffuseImage = true;
                pbrtDiffuseImageFilename = m.textureFilename;
            } else {
                fprintf(stderr, "loadPbrtScene: Diffuse material's own texture '%s' could not be "
                                "read/decoded; falling back to its flat colour\n", m.textureFilename.c_str());
            }
        }
        if (havePbrtDiffuseImage) {
            // `roughness` reused as this hit's own texture
            // SCALE multiplier (J1/section 172 - a "scale"-
            // class Texture wrapping the bare imagemap,
            // Material::textureScale's own comment) - unused
            // by materialType 26's own shading otherwise,
            // same "one scalar slot, per-materialType
            // meaning" pattern every other TriangleMaterial
            // field reuse here already follows. 1.0 (a
            // provable no-op) when the scene's own texture
            // was a bare imagemap with no wrapping "scale" -
            // true for F5/F9, the scenes this materialType
            // was originally built for, so this change is a
            // no-op for them.
            return TriangleMaterial{color, /*materialType=*/METAL_MAT_IMAGE_LAMBERTIAN, /*ior=*/1.0f,
                                     PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/(float)m.textureScale};
        }
    }
    return TriangleMaterial{color, /*materialType=*/METAL_MAT_LAMBERTIAN, /*ior=*/1.0f,
                             PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
}

// Tracks which UNSUPPORTED material kind NAMES (not indices - the same
// name can legitimately appear at more than one scene.materials index)
// have already been warned about, so mapPbrtMaterial() - called once
// per TRIANGLE/sphere/instance referencing a material, not once per
// distinct material - doesn't flood stderr with the identical message
// for every single primitive. A mesh with tens of thousands of
// triangles sharing one unsupported material (e.g. killeroo-simple.pbrt's
// own 66,532-triangle "coateddiffuse" mesh) used to print that exact
// line 66,532 times.
// `depth` guards a Mix material that (wrongly) refers back to itself.
// A pbrt `mix` of two plain surface materials as one METAL_MAT_MIX entry: both sub-materials are mapped and stored whole in the shared float
// buffer, and resolveHit() picks one at each hit, the second with probability `amount`, as pbrt-v4's MixMaterial does (and OptiX and the CPU).
// `amount` <= 0 or >= 1 is just one material, left to the caller. Returns false (leaving `out` alone) for a nested Mix or a medium-bounding
// sub-material, which keep the older load-time pick.
bool MetalPocApp::mapPbrtPerHitMix(PbrtMaterialMapState& st, const pbrt_flatten::Material& m, int depth, TriangleMaterial& out) {
    const pbrt_flatten::FlatScene& scene = st.scene;
    constexpr int kMaxMixDepth = 8;
    auto validIdx = [&](int i) { return i >= 0 && i < (int)scene.materials.size(); };
    if (depth >= kMaxMixDepth || !validIdx(m.mixMaterialA) || !validIdx(m.mixMaterialB) || !(m.mixWeight > 0.0 && m.mixWeight < 1.0)) return false;
    auto cached = st.mixCache.find(&m);
    if (cached != st.mixCache.end()) { out = cached->second; return true; }
    const TriangleMaterial a = mapPbrtMaterial(st, scene.materials[m.mixMaterialA], depth + 1);
    const TriangleMaterial b = mapPbrtMaterial(st, scene.materials[m.mixMaterialB], depth + 1);
    auto plainSurface = [](const TriangleMaterial& t) {
        return t.materialType != METAL_MAT_MIX && !(t.materialType >= METAL_MAT_MEDIUM_HOMOGENEOUS && t.materialType <= METAL_MAT_MEDIUM_RGB_GRID);
    };
    if (!plainSurface(a) || !plainSurface(b)) return false;
    static_assert(sizeof(TriangleMaterial) % sizeof(float) == 0, "TriangleMaterial is stored in the float buffer");
    constexpr size_t kFloats = sizeof(TriangleMaterial) / sizeof(float);
    TriangleMaterial mix{};
    mix.materialType = METAL_MAT_MIX;
    mix.lightId = -1;
    mix.bumpOffset = (int32_t)rgbGridData.size();
    mix.roughness = (float)m.mixWeight;
    rgbGridData.resize(rgbGridData.size() + 2 * kFloats);
    std::memcpy(&rgbGridData[(size_t)mix.bumpOffset], &a, sizeof(a));
    std::memcpy(&rgbGridData[(size_t)mix.bumpOffset + kFloats], &b, sizeof(b));
    st.mixCache.emplace(&m, mix);
    out = mix;
    return true;
}

TriangleMaterial MetalPocApp::mapPbrtMaterial(PbrtMaterialMapState& st, const pbrt_flatten::Material& m, int depth) {
    const pbrt_flatten::FlatScene& scene = st.scene;
    const float sceneScale = st.sceneScale;
    const float3 bboxCenter = st.bboxCenter;
    const float3 sceneOffset = st.sceneOffset;
    std::unordered_set<std::string>& warnedUnsupportedMaterialKinds = st.warnedUnsupportedMaterialKinds;
    std::unordered_map<std::string, int>& measuredTableCache = st.measuredTableCache;
    PackedFloat3 color{(float)m.color[0], (float)m.color[1], (float)m.color[2]};
    // Complex IOR for a GGX conductor (materialType 4). A named metal spectrum or
    // explicit eta/k (m.hasConductorPreset) is used directly. Otherwise the scene gave
    // only a "reflectance" (`color`, a normal-incidence value), and m.conductorEta/K
    // are left at their {0,0,0} defaults - copying those verbatim makes the shader's
    // complex Fresnel divide by |eta+ik|^4 = 0 and produce NaN. Use the same
    // eta=1 / k=2*sqrt(r)/sqrt(1-r) conversion CPU's reflectanceToConductorK() does.
    auto setConductorOptics = [&color](TriangleMaterial& mat, const pbrt_flatten::Material& cm) {
        if (cm.hasConductorPreset) {
            mat.conductorEta = PackedFloat3{(float)cm.conductorEta[0], (float)cm.conductorEta[1], (float)cm.conductorEta[2]};
            mat.conductorK = PackedFloat3{(float)cm.conductorK[0], (float)cm.conductorK[1], (float)cm.conductorK[2]};
        } else {
            mat.conductorEta = PackedFloat3{1.0f, 1.0f, 1.0f};
            mat.conductorK = PackedFloat3{gpu_scene_materials::reflectanceToConductorK(color.x), gpu_scene_materials::reflectanceToConductorK(color.y),
                                          gpu_scene_materials::reflectanceToConductorK(color.z)};
        }
    };
    switch (m.kind) {
        case pbrt_flatten::MaterialKind::Diffuse:
            return mapPbrtDiffuseMaterial(m, sceneScale, bboxCenter, sceneOffset);
        case pbrt_flatten::MaterialKind::Conductor: {
            // RoughnessToAlpha (src/shared/microfacet.h) is sqrt(r) -
            // pbrt-v4's own "remaproughness" default (true) means the
            // authored value needs this remap; false means it already
            // IS alpha.
            const float alpha = (float)(m.remapRoughness ? std::sqrt(m.roughness) : m.roughness);
            // shadeConductor() takes a roughness-style value in `ior` / `roughness` and SQUARES it into alpha (the hand-written
            // scenes are authored that way), so it is handed sqrt(alpha): storing alpha itself made every pbrt conductor render with
            // alpha^2 - roughness 0.04 (alpha 0.2) got alpha 0.04, a far sharper highlight than the CPU and pbrt give.
            const float alphaStored = std::sqrt(alpha);
            TriangleMaterial mat{color, /*materialType=*/METAL_MAT_ROUGH_CONDUCTOR, /*ior(alphaX)=*/alphaStored,
                                 PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness(alphaY)=*/alphaStored};
            setConductorOptics(mat, m);
            return mat;
        }
        case pbrt_flatten::MaterialKind::CoatedConductor: {
            // materialType 20: pbrt-v4's real LayeredBxDF (coat dielectric over the
            // conductor, random-walk Sample_f) via metal_poc_layered_bxdf.metal - verified
            // against CPU on B5/B7. `ior` = the COAT's IOR, `roughness` = the COAT's alpha;
            // the conductor's own alpha and the coat thickness ride in transmitColor (below).
            // A named metal spectrum or explicit "eta"/"k" (m.
            // hasConductorPreset) is used directly, already resolved
            // by pbrt_flatten.h identically to plain Conductor above;
            // otherwise `color` (the scene's own "reflectance", a
            // normal-incidence Schlick value) is converted via the
            // SAME eta=1/k-solved-from-r formula CPU's own
            // reflectanceToConductorK() uses (k = 2*sqrt(r) /
            // sqrt(max(1e-4, 1-r)), per channel) - not re-derived
            // independently, matching a real precedent already
            // established for exactly this "nothing given" case.
            const float alpha = (float)(m.remapRoughness ? std::sqrt(m.roughness_u) : m.roughness_u);
            // pbrt-v4 LayeredBxDF coated conductor (materialType 20): `ior` = coat IOR, `roughness` = alpha.
            TriangleMaterial mat{color, /*materialType=*/METAL_MAT_COATED_CONDUCTOR, /*ior=*/(float)m.ior,
                                 PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/alpha};
            // The conductor has its own roughness (pbrt conductor.roughness; negative = none given, the coat's applies to both) and the coat
            // its thickness: transmitColor = (conductor alpha, thickness, 0), read by shadeCoatedConductor().
            const double baseRough = m.conductorRoughness_u >= 0.0 ? m.conductorRoughness_u : m.roughness_u;
            mat.transmitColor = PackedFloat3{(float)(m.remapRoughness ? std::sqrt(baseRough) : baseRough), (float)m.coatThickness, 0.0f};
            setConductorOptics(mat, m);
            return mat;
        }
        case pbrt_flatten::MaterialKind::NormalizedFresnel: {
            // materialType 18: `ior` = eta, `roughness` = the precomputed normalisation constant
            // c = 1 - 2*FresnelMoment1(1/eta) (a fixed function of eta, so computed here once; the
            // shader has no albedo tint - the BRDF is achromatic).
            const float eta = (float)m.ior;
            float nfC = 1.0f - 2.0f * fresnelMoment1(1.0f / eta);
            if (nfC <= 0.0f) nfC = 1e-6f;
            return TriangleMaterial{color, /*materialType=*/METAL_MAT_NORMALIZED_FRESNEL, /*ior=*/eta,
                                     PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/nfC};
        }
        case pbrt_flatten::MaterialKind::Principled: {
            // materialType 24 (this project's own non-pbrt "principled"): color = base colour,
            // ior, roughness = perceptual roughness, conductorEta = (metallic, clearcoat,
            // clearcoat roughness).
            TriangleMaterial mat{color, /*materialType=*/METAL_MAT_PRINCIPLED, /*ior=*/(float)m.ior,
                                 PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/(float)m.roughness};
            mat.conductorEta = PackedFloat3{(float)m.metallic, (float)m.clearcoat, (float)m.clearcoatRoughness};
            return mat;
        }
        case pbrt_flatten::MaterialKind::Measured: {
            // pbrt-v4 "measured" (tabulated BRDF) -> materialType 32 (shadeMeasured). conductorEta.x = the table descriptor's index in
            // rgbGridData. A file that cannot be read keeps the gray Lambertian fallback, as the CPU does.
            int tableIndex = -1;
            if (!m.measuredFilename.empty()) {
                const auto cached = measuredTableCache.find(m.measuredFilename);
                if (cached != measuredTableCache.end()) {
                    tableIndex = cached->second;
                } else {
                    std::string error;
                    const std::shared_ptr<const MeasuredBRDFData> data = measured_bxdf_io::GetMeasuredBRDFDataCached(m.measuredFilename, error);
                    if (data) {
                        const size_t base = appendMeasuredBrdf(*data, rgbGridData);
                        if (base < (size_t)(1 << 24)) tableIndex = (int)base;
                        else fprintf(stderr, "loadPbrtScene: measured BRDF '%s' cannot be addressed (table block starts past 2^24 floats)\n", m.measuredFilename.c_str());
                    } else {
                        fprintf(stderr, "loadPbrtScene: measured BRDF '%s' could not be loaded (%s); using gray Lambertian\n", m.measuredFilename.c_str(), error.c_str());
                    }
                    measuredTableCache.emplace(m.measuredFilename, tableIndex);
                }
            }
            if (tableIndex >= 0) {
                TriangleMaterial mat{PackedFloat3{1, 1, 1}, /*materialType=*/METAL_MAT_MEASURED, /*ior=*/1.0f,
                                     PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
                mat.conductorEta = PackedFloat3{(float)tableIndex, 0.0f, 0.0f};
                return mat;
            }
            return TriangleMaterial{PackedFloat3{0.5f, 0.5f, 0.5f}, 0u, 1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f};
        }
        case pbrt_flatten::MaterialKind::Hair: {
            // pbrt-v4 HairMaterial -> materialType 31 (Metal's port of hair_material.h).
            // Field reuse: color = sigma_a (already resolved from eumelanin/pheomelanin or
            // a literal sigma_a by flatten), ior = eta, roughness = beta_m,
            // conductorEta = (beta_n, alpha_deg, 0). conductorK is the per-triangle fibre
            // tangent for tessellated curves (set in loadPbrtRemainingTriangles); zero means
            // "use the shading normal as the tangent", CPU's default for non-curve shapes.
            TriangleMaterial mat{PackedFloat3{(float)m.sigma_a[0], (float)m.sigma_a[1], (float)m.sigma_a[2]},
                                 /*materialType=*/METAL_MAT_HAIR, /*ior=*/(float)m.ior,
                                 PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/(float)m.betaM};
            mat.conductorEta = PackedFloat3{(float)m.betaN, (float)m.alphaDeg, 0.0f};
            return mat;
        }
        case pbrt_flatten::MaterialKind::Dielectric:
            // materialType 2 reads `color` as a per-unit-distance ABSORPTION coefficient
            // (Beer-Lambert on exit), where {0,0,0} is perfectly clear glass. `color` here is
            // pbrt's generic material colour, which defaults to 0.5 for a dielectric (it has
            // no "reflectance"): passing it through made every pbrt glass absorb 0.5/unit.
            // Clear by default; loadPbrtSpheres() sets a real coefficient for a glass sphere
            // that bounds a homogeneous medium.
            // "float abbenumber" on a dielectric makes the glass dispersive (a prism's colour fan): materialType 22 (smooth) / 23 (frosted).
            // The Cauchy pair (A, B) is derived from (eta_d, Abbe number) exactly as OptiX and the CPU do, and rides in conductorEta.xy;
            // each path then picks one of the three colour channels at its first dispersive hit (see shadeDispersiveDielectric).
            if (m.abbeNumber > 0.0) {
                double cauchyA = 0.0, cauchyB = 0.0;
                CauchyCoefficientsFromAbbe(m.ior, m.abbeNumber, cauchyA, cauchyB);
                const bool frosted = m.roughness_u > 0.0 || m.roughness_v > 0.0;
                float roughnessSqrtAlpha = 0.0f;
                if (frosted) {
                    const double rr = std::max(m.roughness_u, m.roughness_v);
                    roughnessSqrtAlpha = std::sqrt((float)(m.remapRoughness ? std::sqrt(rr) : rr));   // same convention as the rough dielectric below
                }
                TriangleMaterial mat{PackedFloat3{0, 0, 0}, /*materialType=*/frosted ? METAL_MAT_DISPERSIVE_ROUGH_DIELECTRIC : METAL_MAT_DISPERSIVE_DIELECTRIC,
                                     /*ior=*/(float)m.ior, PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/roughnessSqrtAlpha};
                mat.conductorEta = PackedFloat3{(float)cauchyA, (float)cauchyB, 0.0f};
                return mat;
            }
            if (m.roughness_u > 0.0 || m.roughness_v > 0.0) {
                // Rough (frosted) dielectric -> materialType 5 (GGX microfacet interface). Same convention as the
                // conductor mapping above: the shader squares `roughness` into alpha, so store sqrt(alpha).
                const double rr = std::max(m.roughness_u, m.roughness_v);
                const float a = (float)(m.remapRoughness ? std::sqrt(rr) : rr);
                return TriangleMaterial{PackedFloat3{0, 0, 0}, /*materialType=*/METAL_MAT_ROUGH_DIELECTRIC, /*ior=*/(float)m.ior,
                                         PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/std::sqrt(a)};
            }
            return TriangleMaterial{PackedFloat3{0, 0, 0}, /*materialType=*/METAL_MAT_DIELECTRIC, /*ior=*/(float)m.ior,
                                     PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
        case pbrt_flatten::MaterialKind::ThinDielectric:
            // materialType 11 - `color` is unused by this material
            // (metal_poc.mm's own hardcoded-room construction site
            // comment), only `ior` matters.
            return TriangleMaterial{color, /*materialType=*/METAL_MAT_THIN_DIELECTRIC, /*ior=*/(float)m.ior,
                                     PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
        case pbrt_flatten::MaterialKind::DiffuseTransmission: {
            // materialType 12 - `color` is this material's own
            // reflectance tint (same convention as Diffuse above);
            // `m.transmittance` is the SEPARATE transmitted tint
            // (Material::transmittance's own comment - a frosted-
            // panel default of 0.25 each way when the scene names
            // neither parameter, not a mirror-symmetric 0.5/0.5 split
            // of `color`).
            TriangleMaterial mat{color, /*materialType=*/METAL_MAT_DIFFUSE_TRANSMISSION, /*ior=*/1.0f,
                                 PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
            mat.transmitColor = PackedFloat3{(float)m.transmittance[0], (float)m.transmittance[1],
                                              (float)m.transmittance[2]};
            // Image-textured reflectance and/or transmittance (J2): the reflectance image uses the shared
            // diffuse-image slot, the transmittance image its own second slot (first filename wins in each).
            // conductorK = (reflectance-is-image, 9 = this mode, transmittance-is-image); the kernel samples them.
            {
                bool reflImg = false, transImg = false;
                if (!m.textureFilename.empty() && (!havePbrtDiffuseImage || m.textureFilename == pbrtDiffuseImageFilename)) {
                    if (!havePbrtDiffuseImage) {
                        std::string bytes;
                        if (pbrt_load::loadFileNear(pbrtScenePath, m.textureFilename, bytes) &&
                            pbrt_load::detail::decodeInfiniteLightImage(m.textureFilename, bytes,
                                pbrtDiffuseImagePixels, pbrtDiffuseImageWidth, pbrtDiffuseImageHeight)) {
                            havePbrtDiffuseImage = true;
                            pbrtDiffuseImageFilename = m.textureFilename;
                        }
                    }
                    reflImg = havePbrtDiffuseImage;
                }
                if (!m.transmittanceTextureFilename.empty() &&
                    (!havePbrtTransmitImage || m.transmittanceTextureFilename == pbrtTransmitImageFilename)) {
                    if (!havePbrtTransmitImage) {
                        std::string bytes;
                        if (pbrt_load::loadFileNear(pbrtScenePath, m.transmittanceTextureFilename, bytes) &&
                            pbrt_load::detail::decodeInfiniteLightImage(m.transmittanceTextureFilename, bytes,
                                pbrtTransmitImagePixels, pbrtTransmitImageWidth, pbrtTransmitImageHeight)) {
                            havePbrtTransmitImage = true;
                            pbrtTransmitImageFilename = m.transmittanceTextureFilename;
                        }
                    }
                    transImg = havePbrtTransmitImage;
                }
                if (reflImg || transImg) mat.conductorK = PackedFloat3{reflImg ? 1.0f : 0.0f, 9.0f, transImg ? 1.0f : 0.0f};
            }
            return mat;
        }
        case pbrt_flatten::MaterialKind::CoatedDiffuse: {
            // materialType 19: pbrt-v4's real LayeredBxDF (coat dielectric over a
            // Lambertian base, random-walk Sample_f) via metal_poc_layered_bxdf.metal -
            // verified against CPU on B5. `ior` = the coat's IOR and `roughness` = its
            // alpha, both read from the scene. (This used to be materialType 8, a
            // single-bounce smooth clearcoat with a fixed IOR that ignored both.)
            // Still a real, honest improvement over the gray-
            // Lambertian default fallback below: the correct diffuse
            // albedo and a generic coat sheen both survive, just not
            // the exact coat IOR/roughness.
            //
            // A "reflectance" bound to a bare "imagemap" Texture (J1,
            // section 172) - materialType 27, the SAME per-hit image
            // lookup materialType 26 already gives plain Diffuse
            // (reusing the SAME shared `pbrtDiffuseTexture` slot/
            // `havePbrtDiffuseImage` cache - `Material::textureFilename`
            // is already the identical GENERIC field for both material
            // kinds' own reflectance, pbrt_flatten.h's own comment),
            // just routed through shadeClearcoat's own explicit
            // `albedo` parameter afterward instead of the plain
            // Lambertian path - see metal_poc_kernel.metal's own
            // materialType 26/27 albedo-selection branch (now shared)
            // and its materialType 8/27 shading-dispatch branch (also
            // shared). Ganesha's own statue (this scene's own header
            // comment) is the motivating real-world case: a
            // CoatedDiffuse whose reflectance is a bare imagemap.
            if (!m.textureFilename.empty() &&
                (!havePbrtDiffuseImage || m.textureFilename == pbrtDiffuseImageFilename)) {
                if (!havePbrtDiffuseImage) {
                    std::string bytes;
                    if (pbrt_load::loadFileNear(pbrtScenePath, m.textureFilename, bytes) &&
                        pbrt_load::detail::decodeInfiniteLightImage(m.textureFilename, bytes,
                            pbrtDiffuseImagePixels, pbrtDiffuseImageWidth, pbrtDiffuseImageHeight)) {
                        havePbrtDiffuseImage = true;
                        pbrtDiffuseImageFilename = m.textureFilename;
                    } else {
                        fprintf(stderr, "loadPbrtScene: CoatedDiffuse material's own texture '%s' could not be "
                                        "read/decoded; falling back to its flat colour\n", m.textureFilename.c_str());
                    }
                }
                if (havePbrtDiffuseImage) {
                    return TriangleMaterial{color, /*materialType=*/METAL_MAT_IMAGE_CLEARCOAT, /*ior=*/1.0f,
                                             PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/(float)m.textureScale};
                }
            }
            // pbrt-v4 LayeredBxDF coated diffuse (materialType 19): `ior` = coat IOR, `roughness` = alpha.
            {
                const double coatRough = m.roughness_u;
                const float coatAlpha = (float)(m.remapRoughness ? std::sqrt(coatRough) : coatRough);
                return TriangleMaterial{color, /*materialType=*/METAL_MAT_COATED_DIFFUSE, /*ior=*/(float)m.ior,
                                         PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/coatAlpha};
            }
        }
        case pbrt_flatten::MaterialKind::Mix: {
            // Approx tier: CPU/OptiX both do a REAL per-shading-point
            // stochastic pick between the two named sub-materials
            // (pbrt-v4 MixMaterial - a hash of the hit point decides,
            // giving a fine-grained speckle of both materials' own
            // character, not a blended average - see
            // material_pbrt.h's own mix_material::scatter()). This
            // loader assigns materials once per triangle at LOAD
            // TIME, not per-ray-hit, so a real per-point stochastic
            // mix isn't representable without new per-pixel shader
            // logic - instead, deterministically resolves to
            // whichever of the two sub-materials mixWeight (pbrt's
            // own "amount", the probability weight toward B - see
            // mix_material::scatter()'s own `hash >= w ? A : B`)
            // favours, recursing into mapPbrtMaterial() for that ONE
            // sub-material's own real (possibly ALSO Approx-tier)
            // mapping. A uniform single-material triangle instead of
            // a fine speckle - not the real thing, but still a real,
            // honest improvement over gray Lambertian (the correct
            // material FAMILY and colour survive, just not the
            // per-point blend).
            // Depth-guarded (mirrors pbrt_cpu_builder.h's own
            // kMaxMixDepth=8 exactly - see that file's own comment:
            // "a cyclic/self-referential 'materials' list a
            // malformed scene could produce" is a real, anticipated
            // risk, not hypothetical - namedMaterialIndex is built
            // by scanning the WHOLE material list up front
            // (pbrt_flatten.h), before per-material resolution runs,
            // specifically so a "materials" list can name something
            // declared LATER in the file - which also means a Mix
            // material's own name can legally appear in its own
            // "materials" list, or two Mix materials can name each
            // other, with no cycle check anywhere in flatten() to
            // catch it. Without this guard, mapPbrtMaterial()'s own
            // recursion into such a scene would stack-overflow this
            // loader before any render starts - a real bug found by
            // code review, not exercised by any bundled scene.
            constexpr int kMaxMixDepth = 8;
            // A real per-hit mix where both sub-materials are plain surfaces (see mapPbrtPerHitMix); otherwise the older load-time pick below.
            if (TriangleMaterial perHit; mapPbrtPerHitMix(st, m, depth, perHit)) return perHit;
            const int chosenIdx = (m.mixWeight >= 0.5) ? m.mixMaterialB : m.mixMaterialA;
            if (depth < kMaxMixDepth && chosenIdx >= 0 && chosenIdx < (int)scene.materials.size())
                return mapPbrtMaterial(st, scene.materials[chosenIdx], depth + 1);
            // Both indices invalid (shouldn't happen - flatten()'s
            // own comment guarantees them valid whenever kind==Mix -
            // but this loader errs toward a safe fallback rather
            // than an out-of-bounds read), OR the depth guard above
            // fired (a cyclic/self-referential "materials" list) -
            // falls through to the same gray-Lambertian default
            // every other unsupported kind gets, same "safe fallback
            // over crashing" reasoning as every other malformed-
            // scene case in this file.
            [[fallthrough]];
        }
        default:
            if (warnedUnsupportedMaterialKinds.insert(m.pbrtType).second) {
                fprintf(stderr, "loadPbrtScene: material kind '%s' not supported by this POC's "
                                "scene loader yet, using gray Lambertian instead\n", m.pbrtType.c_str());
            }
            return TriangleMaterial{PackedFloat3{0.5f, 0.5f, 0.5f}, 0u, 1.0f,
                                     PackedFloat3{0, 0, 0}, -1, 0.0f};
    }
}
