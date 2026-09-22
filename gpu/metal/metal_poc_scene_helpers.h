// metal_poc_scene_helpers.h
// Free-standing geometry/mesh-construction and postprocessing helper
// functions (addQuad, loadObjMesh, addBilinearPatch, addTaperedTube,
// reflectanceToConductorK, fresnelMoment1, cauchyCoefficientsFromAbbe,
// bilateralDenoise) used by MetalPocApp's own scene builders and render
// path - split out of metal_poc_app.h as a pure code-motion refactor (no
// behaviour change) once that file grew past ~2000 lines, the same "one
// giant translation unit"-adjacent problem PR #156/metal_poc.mm's own
// earlier splits already addressed - see docs/METAL_GPU_FEASIBILITY.md.
// Included by metal_poc_app.h itself, so every existing includer of THAT
// header keeps seeing these functions with no changes needed at any call
// site.
#pragma once

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mach-o/dyld.h>
#include <sstream>
#include <string>
#include <vector>
#include <simd/simd.h>

#include "metal_poc_gpu_types.h"

// A quad (4 verts, wound as 2 triangles) sharing one flat colour and
// material type - the smallest scene-authoring shape that can build a
// real Cornell box without hand-listing 30 individual vertices. `normals`
// is parallel to `verts` (same per-triangle-corner indexing
// metal_poc.metal's shadingNormalFor() reads) - every corner of a flat
// quad gets the SAME computed face normal, which is what makes the
// shader's barycentric interpolation reduce to exact flat shading here
// (only a mesh with genuinely different per-corner normals, i.e.
// loadObjMesh() below, produces a different, smoothly-varying result).
// `uvs` is the same shape again but for texCoordFor() - a standard
// planar (0,0)-(1,1) mapping across a-b-c-d, meaningful only when
// materialType == 3 (textured); every other quad's UVs are simply never
// read by the shader.
inline void addQuad(std::vector<PackedFloat3>& verts,
                     std::vector<PackedFloat3>& normals,
                     std::vector<PackedFloat2>& uvs,
                     std::vector<TriangleMaterial>& materials,
                     float3 a, float3 b, float3 c, float3 d,
                     float3 color, uint32_t materialType = 0,
                     float3 emission = simd::make_float3(0, 0, 0),
                     int32_t lightId = -1, float roughness = 0.0f,
                     // Defaults to 1.0 (every quad before materialType 11's
                     // own thin-dielectric PR) - matches every earlier
                     // call site's own hardcoded `ior` literal exactly, so
                     // adding this parameter changes nothing for any of
                     // them; only a quad that actually needs a REAL
                     // refraction index (materialType 11's own glass-pane
                     // object) passes something else.
                     float ior = 1.0f,
                     // Diffuse TRANSMITTANCE tint, materialType == 12
                     // only - defaults to black (every quad before that
                     // material existed had no transmission at all).
                     float3 transmitColor = simd::make_float3(0, 0, 0),
                     // pbrt-v4's own "bool twosided" AreaLightSource
                     // parameter (section 104) - defaults to false, every
                     // quad before this one stays one-sided exactly as
                     // before. Meaningless (ignored) for a non-emissive
                     // quad, same as every other emission-only field
                     // above.
                     bool twoSided = false,
                     // Complex IOR (eta+i*k) per RGB channel, materialType
                     // == 4/9 only - see loadObjMesh()'s own identical
                     // pair for the full explanation (dual `ior`/
                     // `roughness` reuse as alphaX/alphaY for a GGX
                     // conductor). Defaults match loadObjMesh()'s own
                     // (eta=(1,1,1), not 0 - unphysical; k=0) so every
                     // pre-existing call site (never a conductor quad
                     // before section 126) is unaffected.
                     float3 conductorEta = simd::make_float3(1.0f, 1.0f, 1.0f),
                     float3 conductorK = simd::make_float3(0.0f, 0.0f, 0.0f)) {
    // a-b-c-d wound so (a,b,c) and (a,c,d) both face outward consistently.
    auto push = [&](float3 v) { verts.push_back(PackedFloat3{v.x, v.y, v.z}); };
    push(a); push(b); push(c);
    push(a); push(c); push(d);
    float3 faceNormal = simd::normalize(simd::cross(b - a, c - a));
    PackedFloat3 packedNormal{faceNormal.x, faceNormal.y, faceNormal.z};
    for (int i = 0; i < 6; ++i) normals.push_back(packedNormal);
    uvs.push_back(PackedFloat2{0, 0});
    uvs.push_back(PackedFloat2{1, 0});
    uvs.push_back(PackedFloat2{1, 1});
    uvs.push_back(PackedFloat2{0, 0});
    uvs.push_back(PackedFloat2{1, 1});
    uvs.push_back(PackedFloat2{0, 1});
    PackedFloat3 packedColor{color.x, color.y, color.z};
    PackedFloat3 packedEmission{emission.x, emission.y, emission.z};
    // materialType == 4 (GGX conductor): `ior` doubles as alphaX, so it
    // must equal `roughness` (alphaY) for isotropic roughness - the
    // SAME bug-prevention convention loadObjMesh() already established
    // (that function's own comment). This makes the invariant
    // impossible to violate by omission at any FUTURE conductor-quad
    // call site, rather than relying on every caller to remember it.
    const float effectiveIor = (materialType == 4u) ? roughness : ior;
    TriangleMaterial mat{packedColor, materialType, effectiveIor, packedEmission, lightId, roughness};
    mat.conductorEta = PackedFloat3{conductorEta.x, conductorEta.y, conductorEta.z};
    mat.conductorK = PackedFloat3{conductorK.x, conductorK.y, conductorK.z};
    mat.transmitColor = PackedFloat3{transmitColor.x, transmitColor.y, transmitColor.z};
    mat.twoSided = twoSided ? 1u : 0u;
    materials.push_back(mat);
    materials.push_back(mat);
}

// A minimal Wavefront OBJ loader: positions, vertex normals, texture
// coordinates, and faces (`v`/`vn`/`vt`/`f`) - no materials/groups. This
// project's own real loaders (src/shared/pbrt_load.h ->
// pbrt_cpu_builder.h/pbrt_gpu_builder.h) are full pbrt-v4 scene parsers;
// this is deliberately the smallest thing that can prove "load an
// arbitrary real mesh, not just hand-authored axis-aligned quads and an
// analytic sphere" - the first genuinely data-driven geometry in this
// POC. Faces are fan-triangulated (n>3 polygon -> n-2 triangles sharing
// vertex 0), matching how this project's own CPU loader handles polygons
// that aren't already triangles.
//
// Per-corner shading normal: a face token's own `vn` index is used when
// present (`v//vn` or `v/vt/vn`); a face missing normal indices entirely
// falls back to that triangle's own computed flat face normal - real
// files are inconsistent about this in practice (suzanne.obj has `vn` on
// every face; spot.obj, added for real UV testing below, has NONE at
// all, so both code paths get real exercise across this POC's two mesh
// files, not just the fallback-free one). This is what makes
// metal_poc.metal's barycentric shadingNormalFor() interpolation produce
// a genuinely smooth result instead of the flat-per-triangle look every
// other object in this scene has.
//
// Per-corner texture coordinate: same idea, a face token's own `vt`
// index (`v/vt` or `v/vt/vn`) when present, (0,0) fallback otherwise -
// this is the real-data counterpart to addQuad()'s hand-authored planar
// UVs, the piece the texture-mapping PR explicitly deferred ("parsing
// real vt/f v/vt/vn tokens was judged out of scope for this increment").
// suzanne.obj has no `vt` data at all (every corner falls back), so it
// keeps its materialType 0; spot.obj DOES (3225 `vt` entries, one per
// face corner), the caller passes materialType 3 for it, and
// texCoordFor()'s barycentric interpolation on real per-corner data
// produces a real (if mismatched-content, earthmap.jpg was never meant
// for a cow) textured mesh, not just a textured flat quad.
//
// The mesh is auto-fit to `targetSize` (its largest bounding-box
// dimension scaled to that value) and recentred at `center` - real .obj
// files come in whatever units/scale their author used, and this scene's
// room is a fixed [-1,1] box, so SOME normalization is unavoidable rather
// than a hardcoded scale constant that would only happen to work for this
// one file.
// The directory containing the CURRENTLY RUNNING executable, or nil if
// unavailable (_NSGetExecutablePath, not NSBundle - resolves correctly
// for a plain non-app-bundle CLI binary too). Checked FIRST, before any
// RT_..._DIR compile-time fallback (RT_MODELS_DIR/RT_METAL_SHADER_DIR),
// at every asset-lookup site that has one - those are absolute paths
// into the machine that BUILT this binary, meaningless once it's been
// copied/installed anywhere else (section 110's own real bug: a
// distributed .dmg's bundled `ray_tracer`, run on a genuinely different
// machine, would otherwise fail to find its own shader source AND
// models/suzanne.obj, models/spot.obj, images/earthmap.jpg - found by
// actually testing a packaged build from a clean, relocated install,
// not assumed correct from the code alone).
inline NSString* executableDir() {
    char exePathBuf[4096];
    uint32_t exePathSize = sizeof(exePathBuf);
    if (_NSGetExecutablePath(exePathBuf, &exePathSize) != 0) return nil;
    return [@(exePathBuf) stringByDeletingLastPathComponent];
}

inline bool loadObjMesh(const std::string& path,
                         std::vector<PackedFloat3>& verts,
                         std::vector<PackedFloat3>& normals,
                         std::vector<PackedFloat2>& uvs,
                         std::vector<TriangleMaterial>& materials,
                         float3 color, float3 center, float targetSize,
                         uint32_t materialType = 0,
                         // materialType==4 only - see this function's own
                         // TriangleMaterial-construction comment below for
                         // why these three exist. meshConductorEta defaults
                         // to (1,1,1) (not 0) since eta==0 is unphysical and
                         // every REAL conductor preset this codebase already
                         // uses (conductor_data.h) keeps eta near 1 anyway -
                         // a caller that forgets to override k too still
                         // gets a real (if flat/grey) metal, not black.
                         float meshRoughness = 0.0f,
                         float3 meshConductorEta = simd::make_float3(1.0f, 1.0f, 1.0f),
                         float3 meshConductorK = simd::make_float3(0.0f, 0.0f, 0.0f),
                         // materialType==2 (dielectric) only - see this
                         // function's own TriangleMaterial-construction
                         // comment below. Default 1.0 (no refraction) keeps
                         // every OTHER materialType's behaviour identical to
                         // before this parameter existed.
                         float meshIor = 1.0f,
                         // Mirrors gpu/optix/scene_builder.cpp's own
                         // load_obj_triangles_gpu()'s flip_xz parameter
                         // exactly: negates x and z (a 180-degree rotation
                         // about Y) - some raw meshes (Spot the Cow, Horse)
                         // face away from this app's own camera convention
                         // without it (that file's own comment: "the raw
                         // mesh faces away from the camera"). Applied to
                         // every position AND normal, before centring/
                         // scaling - section 119, docs/METAL_GPU_
                         // FEASIBILITY.md.
                         bool flipXZ = false) {
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "Could not open OBJ file: %s\n", path.c_str());
        return false;
    }

    std::vector<float3> positions;
    std::vector<float3> fileNormals;
    std::vector<simd::float2> fileUVs;
    // Each face vertex is (positionIndex, normalIndex-or--1,
    // uvIndex-or--1), 0-based post-fixup - keeping the triple together
    // (rather than three parallel index lists) is what lets a face's own
    // vn/vt references survive fan triangulation below unchanged.
    struct FaceVertex { int posIdx; int normalIdx; int uvIdx; };
    std::vector<std::vector<FaceVertex>> faces;

    auto parseObjIndex = [](const std::string& token, size_t countAtParseTime) -> int {
        int idx = std::atoi(token.c_str());
        if (idx == 0) return -1; // absent (e.g. the "vt" slot in "v/vt/vn")
        // OBJ indices are 1-based; a negative index is relative to the
        // current count (rare, but real files use it).
        if (idx < 0) idx = (int)countAtParseTime + idx + 1;
        return idx - 1;
    };

    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            positions.push_back(simd::make_float3(x, y, z));
        } else if (tag == "vn") {
            float x, y, z;
            ss >> x >> y >> z;
            fileNormals.push_back(simd::make_float3(x, y, z));
        } else if (tag == "vt") {
            float u, v;
            ss >> u >> v;
            fileUVs.push_back(simd::float2{u, v});
        } else if (tag == "f") {
            std::vector<FaceVertex> faceVerts;
            std::string token;
            while (ss >> token) {
                // Token is "v", "v/vt", "v//vn", or "v/vt/vn".
                size_t firstSlash = token.find('/');
                size_t lastSlash = token.rfind('/');
                int posIdx = parseObjIndex(token.substr(0, firstSlash), positions.size());
                int normalIdx = -1;
                int uvIdx = -1;
                if (firstSlash != std::string::npos && lastSlash != firstSlash) {
                    normalIdx = parseObjIndex(token.substr(lastSlash + 1), fileNormals.size());
                }
                if (firstSlash != std::string::npos) {
                    // The vt slot sits between the two slashes for
                    // "v/vt/vn", or from the first slash to the token's
                    // end for "v/vt" (no vn at all, spot.obj's own
                    // format) - lastSlash == firstSlash in that case, so
                    // this substring naturally runs to end-of-string.
                    size_t vtEnd = (lastSlash != firstSlash) ? lastSlash : token.size();
                    std::string vtToken = token.substr(firstSlash + 1, vtEnd - firstSlash - 1);
                    uvIdx = parseObjIndex(vtToken, fileUVs.size());
                }
                faceVerts.push_back({posIdx, normalIdx, uvIdx});
            }
            if (faceVerts.size() >= 3) faces.push_back(faceVerts);
        }
    }

    if (positions.empty() || faces.empty()) {
        fprintf(stderr, "OBJ file had no usable geometry: %s\n", path.c_str());
        return false;
    }

    float3 bboxMin = simd::make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 bboxMax = simd::make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (const float3& p : positions) {
        bboxMin = simd::min(bboxMin, p);
        bboxMax = simd::max(bboxMax, p);
    }

    float3 extent = bboxMax - bboxMin;
    float largestDim = std::max(extent.x, std::max(extent.y, extent.z));
    float scale = (largestDim > 0.0f) ? (targetSize / largestDim) : 1.0f;
    float3 bboxCenter = (bboxMin + bboxMax) * 0.5f;

    // flipXZ applied to the CENTRED delta, not the raw position - the
    // bounding box above is computed from raw positions either way (a
    // pure x/z negation is a reflection, which preserves the extent used
    // for `scale`, so no separate flipped-bbox pass is needed).
    auto transform = [&](const float3& p) -> float3 {
        float3 delta = p - bboxCenter;
        if (flipXZ) { delta.x = -delta.x; delta.z = -delta.z; }
        return delta * scale + center;
    };
    // Normals only need the scale's sign/shear behaviour, not translation -
    // a uniform positive scale (this loader's only kind) leaves direction
    // unchanged, so this is really just "no-op, pass through," kept as its
    // own step for clarity and in case a future non-uniform scale needs it.
    // flipXZ needs the SAME x/z negation as transform() above (a normal
    // rotates with its surface).
    auto transformNormal = [&](const float3& n) -> float3 {
        float3 nn = n;
        if (flipXZ) { nn.x = -nn.x; nn.z = -nn.z; }
        return simd::normalize(nn);
    };

    uint32_t triangleCount = 0;
    uint32_t normalFallbackCount = 0;
    uint32_t uvFallbackCount = 0;
    for (const std::vector<FaceVertex>& face : faces) {
        // Fan triangulation from vertex 0 - correct for the convex/near-
        // convex polygons a typical modeled mesh's faces are (this file's
        // own quads included), not a general concave-polygon triangulator.
        for (size_t i = 1; i + 1 < face.size(); ++i) {
            FaceVertex fv0 = face[0], fv1 = face[i], fv2 = face[i + 1];
            if (fv0.posIdx < 0 || fv0.posIdx >= (int)positions.size() ||
                fv1.posIdx < 0 || fv1.posIdx >= (int)positions.size() ||
                fv2.posIdx < 0 || fv2.posIdx >= (int)positions.size()) {
                continue; // malformed index - skip rather than crash
            }
            float3 a = transform(positions[fv0.posIdx]);
            float3 b = transform(positions[fv1.posIdx]);
            float3 c = transform(positions[fv2.posIdx]);
            verts.push_back(PackedFloat3{a.x, a.y, a.z});
            verts.push_back(PackedFloat3{b.x, b.y, b.z});
            verts.push_back(PackedFloat3{c.x, c.y, c.z});

            bool haveAllUVs =
                fv0.uvIdx >= 0 && fv0.uvIdx < (int)fileUVs.size() &&
                fv1.uvIdx >= 0 && fv1.uvIdx < (int)fileUVs.size() &&
                fv2.uvIdx >= 0 && fv2.uvIdx < (int)fileUVs.size();
            if (haveAllUVs) {
                uvs.push_back(PackedFloat2{fileUVs[fv0.uvIdx].x, fileUVs[fv0.uvIdx].y});
                uvs.push_back(PackedFloat2{fileUVs[fv1.uvIdx].x, fileUVs[fv1.uvIdx].y});
                uvs.push_back(PackedFloat2{fileUVs[fv2.uvIdx].x, fileUVs[fv2.uvIdx].y});
            } else {
                uvs.push_back(PackedFloat2{0, 0});
                uvs.push_back(PackedFloat2{0, 0});
                uvs.push_back(PackedFloat2{0, 0});
                ++uvFallbackCount;
            }

            bool haveAllNormals =
                fv0.normalIdx >= 0 && fv0.normalIdx < (int)fileNormals.size() &&
                fv1.normalIdx >= 0 && fv1.normalIdx < (int)fileNormals.size() &&
                fv2.normalIdx >= 0 && fv2.normalIdx < (int)fileNormals.size();
            if (haveAllNormals) {
                float3 n0 = transformNormal(fileNormals[fv0.normalIdx]);
                float3 n1 = transformNormal(fileNormals[fv1.normalIdx]);
                float3 n2 = transformNormal(fileNormals[fv2.normalIdx]);
                normals.push_back(PackedFloat3{n0.x, n0.y, n0.z});
                normals.push_back(PackedFloat3{n1.x, n1.y, n1.z});
                normals.push_back(PackedFloat3{n2.x, n2.y, n2.z});
            } else {
                float3 flat = simd::normalize(simd::cross(b - a, c - a));
                PackedFloat3 packedFlat{flat.x, flat.y, flat.z};
                normals.push_back(packedFlat);
                normals.push_back(packedFlat);
                normals.push_back(packedFlat);
                ++normalFallbackCount;
            }
            ++triangleCount;
        }
    }

    PackedFloat3 packedColor{color.x, color.y, color.z};
    TriangleMaterial mat{packedColor, materialType, 1.0f, PackedFloat3{0, 0, 0}};
    // materialType==4 (real complex-Fresnel GGX conductor) needs
    // conductorEta/conductorK too, which this function's own signature had
    // no way to pass until section 117's own G-category (Models) increment
    // needed a metal-finish mesh for the first time (Suzanne/Spot, this
    // function's only callers before that, are both materialType 0/3). Left
    // at their struct default (eta/k = {0,0,0}, roughness = 0) for every
    // OTHER materialType - identical to this function's own behaviour
    // before these parameters existed.
    //
    // BOTH mat.ior (alphaX) and mat.roughness (alphaY) must be set to the
    // SAME value for isotropic roughness - mapMaterial()'s own Conductor
    // case (loadPbrtScene()) does this identically. Leaving `mat.ior` at
    // the `1.0f` this constructor already gives every material (a
    // DIELECTRIC default, meaningless for a conductor) while only setting
    // `mat.roughness` would silently make alphaX=1.0 (maximally rough) and
    // alphaY=meshRoughness - a real, easy-to-miss anisotropy bug caught
    // here before it ever rendered, not after.
    if (materialType == 4u) {
        mat.ior = meshRoughness;
        mat.roughness = meshRoughness;
        mat.conductorEta = PackedFloat3{meshConductorEta.x, meshConductorEta.y, meshConductorEta.z};
        mat.conductorK = PackedFloat3{meshConductorK.x, meshConductorK.y, meshConductorK.z};
    } else if (materialType == 2u) {
        // Smooth dielectric (materialType 2, e.g. Glass Dragon - section
        // 119) - `ior` here is a real refraction index (glass~1.5), not
        // the alphaX reuse materialType 4 gives it above.
        mat.ior = meshIor;
    }
    for (uint32_t i = 0; i < triangleCount; ++i) materials.push_back(mat);

    fprintf(stderr, "Loaded %s: %zu positions, %zu normals, %zu uvs, %u triangles "
                     "(%u flat-normal fallback, %u zero-uv fallback), scale %.4f\n",
            path.c_str(), positions.size(), fileNormals.size(), fileUVs.size(), triangleCount,
            normalFallbackCount, uvFallbackCount, scale);
    return true;
}

// blackbodyColor/vignetteFactor/sampleChannelBilinear/chromaticAberration/
// acesFilmicTonemap/reinhardTonemap/ToneMapMode/parseToneMapMode/
// applyToneMap/linearToSRGB now live in metal_poc_host_math.h (included
// above) - see that header's own comment.

// Edge-preserving bilateral denoise, applied to the final 8-bit LDR
// image (after tonemapping/gamma, not the linear HDR buffer - the
// standard display-referred way to do this: a range kernel compared
// directly against raw HDR values would be dominated by the huge
// magnitude gap between a light source and everything else, rather than
// meaningfully distinguishing "real edge" from "Monte Carlo noise").
// A follow-on to the firefly clamp: that PR found (and honestly
// reported) this scene's own worst noise - high-variance fog/volumetric
// sampling near the spot light's own cone - wasn't the rare-extreme-
// outlier kind firefly clamping targets, so it barely helped there.
// Spatial denoising targets exactly that kind of noise instead: every
// neighbouring pixel contributes to the output, weighted by BOTH how
// close it is (`sigmaSpatial`, a Gaussian in pixel distance) and how
// similar its own LUMINANCE is to the centre pixel's (`sigmaRange`, a
// Gaussian in luminance difference) - two nearby pixels with similar
// brightness (likely the same underlying surface, differing only by
// noise) get smoothed together; two nearby pixels with very different
// brightness (likely a real edge - a shadow boundary, a specular
// highlight, a checker tile seam) barely influence each other at all,
// which is what keeps this from just being a uniform blur. The SAME
// per-pixel weight (derived from luminance alone) is applied to all
// three colour channels together, not computed separately per channel -
// preserves each pixel's own hue relationship to its neighbours instead
// of letting R/G/B drift independently.
inline void bilateralDenoise(const std::vector<uint8_t>& ldrIn, std::vector<uint8_t>& ldrOut,
                              uint32_t width, uint32_t height, int radius,
                              float sigmaSpatial, float sigmaRange) {
    std::vector<float> luminance(width * height);
    for (uint32_t i = 0; i < width * height; ++i) {
        luminance[i] = 0.2126f * ldrIn[i * 3 + 0] + 0.7152f * ldrIn[i * 3 + 1] + 0.0722f * ldrIn[i * 3 + 2];
    }
    float invSpatial2 = 1.0f / (2.0f * sigmaSpatial * sigmaSpatial);
    float invRange2 = 1.0f / (2.0f * sigmaRange * sigmaRange);
    for (int32_t y = 0; y < (int32_t)height; ++y) {
        for (int32_t x = 0; x < (int32_t)width; ++x) {
            uint32_t centerIdx = (uint32_t)y * width + (uint32_t)x;
            float centerLum = luminance[centerIdx];
            float sumWeight = 0.0f;
            float sumRGB[3] = {0.0f, 0.0f, 0.0f};
            for (int32_t dy = -radius; dy <= radius; ++dy) {
                int32_t ny = y + dy;
                if (ny < 0 || ny >= (int32_t)height) continue;
                for (int32_t dx = -radius; dx <= radius; ++dx) {
                    int32_t nx = x + dx;
                    if (nx < 0 || nx >= (int32_t)width) continue;
                    uint32_t nIdx = (uint32_t)ny * width + (uint32_t)nx;
                    float spatialTerm = float(dx * dx + dy * dy) * invSpatial2;
                    float lumDiff = luminance[nIdx] - centerLum;
                    float rangeTerm = lumDiff * lumDiff * invRange2;
                    float weight = expf(-(spatialTerm + rangeTerm));
                    sumWeight += weight;
                    sumRGB[0] += weight * float(ldrIn[nIdx * 3 + 0]);
                    sumRGB[1] += weight * float(ldrIn[nIdx * 3 + 1]);
                    sumRGB[2] += weight * float(ldrIn[nIdx * 3 + 2]);
                }
            }
            for (int c = 0; c < 3; ++c) {
                float v = sumRGB[c] / fmaxf(sumWeight, 1e-6f);
                ldrOut[centerIdx * 3 + c] = (uint8_t)fminf(fmaxf(v + 0.5f, 0.0f), 255.0f);
            }
        }
    }
}

// Converts a flat OptiX-style metal "albedo" into an approximate complex
// conductor (eta,k) via PR #103's own already-shipped reflectance-to-k
// formula - see buildMeshGalleryScene()'s own declaration comment.
inline float3 reflectanceToConductorK(float3 albedo) {
    auto k = [](float r) {
        r = r < 0.0f ? 0.0f : (r > 0.9999f ? 0.9999f : r);
        return 2.0f * sqrtf(r) / sqrtf(std::max(1e-4f, 1.0f - r));
    };
    return float3{k(albedo.x), k(albedo.y), k(albedo.z)};
}

// Tessellates a bilinear patch (4 corners, possibly NON-PLANAR - F1's
// own genuinely new geometry primitive, pbrt-v4's own BilinearPatch
// shape) into a fine NxN triangle grid, instead of adding a real
// bounding-box custom-intersection-function primitive (this loader's
// existing sphere/disk precedent - see sphereIntersectionFunction/
// diskIntersectionFunction, metal_poc.metal's own comment on the disk
// one: "a second, genuinely DIFFERENT custom-primitive shape"). A
// deliberate, documented simplification: a fine enough tessellation of
// a bilinear (degree-1-per-axis, genuinely smooth) surface is visually
// indistinguishable from the true analytic surface at any reasonable
// render resolution, and reuses 100% already-proven triangle
// infrastructure instead of a real architecture change (a THIRD
// bounding-box geometry descriptor, a new intersection function, and
// updating every `!isSphere && !isDisk && !isSuzanneInstance`-style
// exclusion check already scattered through the main shading kernel) -
// section 154, docs/METAL_GPU_FEASIBILITY.md. Per-VERTEX normals are
// the REAL analytic bilinear-surface normal at that exact (u,v)
// (`cross(dPdu, dPdv)`, not a flat per-face fallback), smoothly
// interpolated across each triangle by the SAME barycentric
// `shadingNormalFor()` every other smooth mesh here already uses, so
// the tessellation seams stay invisible under shading even though the
// underlying triangles are flat. Normal SIGN is never resolved to a
// canonical "outward" direction (unlike a convex sphere/CPU's own
// analytic shape) - deliberately unnecessary: this helper is only ever
// used for a GGX conductor material (materialType 4), whose own
// shading always uses the ray-`facingNormal` (auto-flipped to the
// visible side, `shadeConductor`'s own `frontFace` check), never the
// raw geometric one, so an inconsistent or "inward" normal sign is
// self-correcting and invisible in the final render.
inline void addBilinearPatch(std::vector<PackedFloat3>& verts,
                              std::vector<PackedFloat3>& normals,
                              std::vector<PackedFloat2>& uvs,
                              std::vector<TriangleMaterial>& materials,
                              float3 p00, float3 p10, float3 p01, float3 p11,
                              float3 color, float roughness,
                              int subdivisions = 24) {
    auto evalP = [&](float u, float v) -> float3 {
        return (1.0f - u) * (1.0f - v) * p00 + u * (1.0f - v) * p10
             + (1.0f - u) * v * p01 + u * v * p11;
    };
    auto evalNormal = [&](float u, float v) -> float3 {
        const float3 dPdu = (1.0f - v) * (p10 - p00) + v * (p11 - p01);
        const float3 dPdv = (1.0f - u) * (p01 - p00) + u * (p11 - p10);
        return simd::normalize(simd::cross(dPdu, dPdv));
    };
    const float3 k = reflectanceToConductorK(color);
    TriangleMaterial mat{PackedFloat3{color.x, color.y, color.z}, /*materialType=*/4u,
        /*ior(alphaX)=*/roughness, PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness(alphaY)=*/roughness};
    mat.conductorEta = PackedFloat3{1.0f, 1.0f, 1.0f};
    mat.conductorK = PackedFloat3{k.x, k.y, k.z};

    auto pushV = [&](float3 p) { verts.push_back(PackedFloat3{p.x, p.y, p.z}); };
    auto pushN = [&](float3 n) { normals.push_back(PackedFloat3{n.x, n.y, n.z}); };
    for (int i = 0; i < subdivisions; ++i) {
        for (int j = 0; j < subdivisions; ++j) {
            const float u0 = (float)i / subdivisions, u1 = (float)(i + 1) / subdivisions;
            const float v0 = (float)j / subdivisions, v1 = (float)(j + 1) / subdivisions;
            const float3 g00 = evalP(u0, v0), g10 = evalP(u1, v0);
            const float3 g01 = evalP(u0, v1), g11 = evalP(u1, v1);
            const float3 n00 = evalNormal(u0, v0), n10 = evalNormal(u1, v0);
            const float3 n01 = evalNormal(u0, v1), n11 = evalNormal(u1, v1);

            pushV(g00); pushV(g10); pushV(g11);
            pushN(n00); pushN(n10); pushN(n11);
            uvs.push_back(PackedFloat2{u0, v0});
            uvs.push_back(PackedFloat2{u1, v0});
            uvs.push_back(PackedFloat2{u1, v1});
            materials.push_back(mat);

            pushV(g00); pushV(g11); pushV(g01);
            pushN(n00); pushN(n11); pushN(n01);
            uvs.push_back(PackedFloat2{u0, v0});
            uvs.push_back(PackedFloat2{u1, v1});
            uvs.push_back(PackedFloat2{u0, v1});
            materials.push_back(mat);
        }
    }
}

// Tessellates a single tapered cubic-Bezier tube (F4's own genuinely
// new geometry primitive, pbrt-v4's own CurveShape<Cylinder>, a real
// ray-curve intersection on CPU) into a triangulated "tube of quad
// rings" - matches this scene's OWN registry description exactly
// ("GPU renders the same 70 strands tessellated into tapered tubes of
// bilinear patches (matches pbrt-v4's own GPU curve strategy) rather
// than an exact curve intersection"), the SAME documented simplification
// F1's own addBilinearPatch() already established for a different
// smooth-but-not-exactly-triangle shape (section 154) - reusing the
// already-proven triangle path instead of a new custom-intersection-
// function primitive. `lengthSegments` rings of `radialSegments`
// points each are swept along the curve; each ring's own local frame
// is built by GRAM-SCHMIDT re-orthogonalizing the PREVIOUS ring's own
// right/up vectors against the new tangent (not an independent
// per-ring basis, which would twist/flip randomly ring to ring for a
// thin tube) - a simple, adequate rotation-minimizing-frame
// approximation for a gently-curving strand (not a tightly coiled
// spring, where a more careful RMF would matter). Per-vertex normals
// are the tube's own outward RADIAL direction in each ring's local
// frame (correct for a swept-circle tube, ignoring the curve's own
// typically-negligible curvature-induced normal skew) - smoothly
// interpolated the same way F1's own bilinear-patch normals already
// are. End caps are skipped entirely (the root sits at/below the
// ground plane, invisible; the tip tapers to a near-zero radius,
// visually negligible) - matches this loader's own established
// "skip what a control render shows is imperceptible" discipline.
inline void addTaperedTube(std::vector<PackedFloat3>& verts,
                            std::vector<PackedFloat3>& normals,
                            std::vector<PackedFloat2>& uvs,
                            std::vector<TriangleMaterial>& materials,
                            const float3 cp[4], float width0, float width1,
                            float3 color, int lengthSegments = 14, int radialSegments = 8) {
    auto evalBezier = [&](float t) -> float3 {
        const float mt = 1.0f - t;
        return mt * mt * mt * cp[0] + 3.0f * mt * mt * t * cp[1]
             + 3.0f * mt * t * t * cp[2] + t * t * t * cp[3];
    };
    auto evalTangent = [&](float t) -> float3 {
        const float mt = 1.0f - t;
        const float3 d = 3.0f * mt * mt * (cp[1] - cp[0]) + 6.0f * mt * t * (cp[2] - cp[1])
                        + 3.0f * t * t * (cp[3] - cp[2]);
        return simd::normalize(d);
    };

    TriangleMaterial mat{PackedFloat3{color.x, color.y, color.z}, /*materialType=*/0u,
        1.0f, PackedFloat3{0, 0, 0}, /*lightId=*/-1, 0.0f};

    // Ring 0's own initial frame: an arbitrary reference vector not
    // parallel to the tangent (world up, unless the strand starts out
    // near-vertical, in which case +X instead).
    const float3 tangent0 = evalTangent(0.0f);
    const float3 ref0 = (fabsf(tangent0.y) < 0.99f) ? float3{0, 1, 0} : float3{1, 0, 0};
    float3 right = simd::normalize(simd::cross(tangent0, ref0));
    float3 up = simd::cross(right, tangent0);

    std::vector<float3> prevRing(radialSegments), prevNormals(radialSegments);
    std::vector<float3> curRing(radialSegments), curNormals(radialSegments);
    for (int seg = 0; seg <= lengthSegments; ++seg) {
        const float t = (float)seg / (float)lengthSegments;
        const float3 center = evalBezier(t);
        const float3 tangent = evalTangent(t);
        if (seg > 0) {
            // Gram-Schmidt re-orthogonalize the running frame against
            // the new tangent - keeps the ring from twisting.
            right = simd::normalize(right - tangent * simd::dot(right, tangent));
            up = simd::cross(tangent, right);
        }
        const float radius = 0.5f * ((1.0f - t) * width0 + t * width1);
        for (int k = 0; k < radialSegments; ++k) {
            const float theta = 2.0f * (float)M_PI * (float)k / (float)radialSegments;
            const float3 radial = cosf(theta) * right + sinf(theta) * up;
            curRing[k] = center + radius * radial;
            curNormals[k] = radial;
        }
        if (seg > 0) {
            for (int k = 0; k < radialSegments; ++k) {
                const int k1 = (k + 1) % radialSegments;
                auto pushV = [&](float3 p) { verts.push_back(PackedFloat3{p.x, p.y, p.z}); };
                auto pushN = [&](float3 n) { normals.push_back(PackedFloat3{n.x, n.y, n.z}); };
                pushV(prevRing[k]); pushV(curRing[k]); pushV(curRing[k1]);
                pushN(prevNormals[k]); pushN(curNormals[k]); pushN(curNormals[k1]);
                uvs.push_back(PackedFloat2{0, 0}); uvs.push_back(PackedFloat2{1, 0}); uvs.push_back(PackedFloat2{1, 1});
                materials.push_back(mat);

                pushV(prevRing[k]); pushV(curRing[k1]); pushV(prevRing[k1]);
                pushN(prevNormals[k]); pushN(curNormals[k1]); pushN(prevNormals[k1]);
                uvs.push_back(PackedFloat2{0, 0}); uvs.push_back(PackedFloat2{1, 1}); uvs.push_back(PackedFloat2{0, 1});
                materials.push_back(mat);
            }
        }
        prevRing = curRing;
        prevNormals = curNormals;
    }
}

// pbrt-v4's own FresnelMoment1() polynomial fit (src/shared/fresnel.h,
// ported directly - HOST-side only, since `eta` never varies per-hit
// for a NormalizedFresnelBxDF material, so `c = 1 - 2*FresnelMoment1(1/eta)`
// can be precomputed once here rather than needing a device-side port
// at all - see shadeNormalizedFresnel()'s own declaration comment,
// metal_poc.metal).
inline float fresnelMoment1(float eta) {
    const float eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta, eta5 = eta4 * eta;
    if (eta < 1.0f) {
        return 0.45966f - 1.73965f * eta + 3.37668f * eta2
             - 3.904945f * eta3 + 2.49277f * eta4 - 0.68441f * eta5;
    }
    return -4.61686f + 11.1136f * eta - 10.4646f * eta2
         + 5.11455f * eta3 - 1.27198f * eta4 + 0.12746f * eta5;
}

// CauchyCoefficientsFromAbbe (src/shared/fresnel.h) - construction-time
// only (not per-ray), ported here rather than #included directly since
// it's the one piece of that header genuinely CPU-only (no CPU_GPU tag
// needed at all - the per-ray CauchyEta() counterpart IS ported, as
// `cauchyEta()`, into metal_poc.metal itself, next to
// shadeDispersiveDielectric()'s own declaration).
inline void cauchyCoefficientsFromAbbe(double etaD, double abbeNumber, double& A, double& B) {
    constexpr double lambdaF = 0.4861, lambdaC = 0.6563, lambdaD = 0.5893;
    B = (etaD - 1.0) / (abbeNumber * (1.0 / (lambdaF * lambdaF) - 1.0 / (lambdaC * lambdaC)));
    A = etaD - B / (lambdaD * lambdaD);
}
