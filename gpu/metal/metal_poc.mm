// metal_poc.mm
// Host side of the Metal ray tracing proof-of-concept - see
// docs/METAL_GPU_FEASIBILITY.md. Builds a small hardcoded Cornell-box-like
// triangle scene, uploads it into a Metal acceleration structure, dispatches
// metal_poc.metal's inline-intersection compute kernel, and writes the
// result to a PNG via this project's existing stb_image_write.h - same
// output path the CPU renderer already uses, so the two are trivially
// visually comparable.
//
// Deliberately standalone (own main(), a separate CLI tool from
// ray_tracer/scene_metadata, not called by launcher/main.cpp or the Qt
// GUI): this is the "self-contained, time-boxed spike" the feasibility
// doc's Suggested Next Step calls for, proving the pipeline shape works
// before any of the real material/light/shape porting work starts. It IS
// now CMake-integrated (root CMakeLists.txt's RT_BUILD_METAL option) as
// its own metal_poc target, separately from ray_tracer/optix_renderer -
// build with `cmake -B build -DRT_BUILD_METAL=ON && cmake --build build
// --target metal_poc`.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstring>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../src/external/stb_image_write.h"
// STB_IMAGE_IMPLEMENTATION here is a separate translation unit from
// src/external/stb_image_impl.cpp's own definition of it (that one is
// compiled into cpu_renderer, which metal_poc doesn't link against at
// all - two different executables, no ODR conflict) - loading
// images/earthmap.jpg for the textured-material test below.
#define STB_IMAGE_IMPLEMENTATION
#include "../../src/external/stb_image.h"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <fstream>
#include <sstream>
#include <string>
#include <simd/simd.h>

using simd::float3;

// A plain, tightly-packed 12-byte vertex type - MTLAccelerationStructure
// TriangleGeometryDescriptor reads raw bytes at vertexStride, and simd's
// own float3 is 16-byte-aligned/padded, not 12, so a hand-rolled struct
// (no simd padding) is what actually matches a tight vertexStride.
struct PackedFloat3 {
    float x, y, z;
};

// Mirrors metal_poc.metal's UV buffer element type - simd::float2 has no
// padding-inside-a-struct issue the way float3 does (2 floats is already
// its own natural 8-byte alignment), but named separately from simd::float2
// for the same "obviously the wire format, not incidentally compatible
// with it" clarity PackedFloat3 gives the position/normal buffers.
struct PackedFloat2 {
    float u, v;
};

// Mirrors metal_poc.metal's Uniforms/TriangleMaterial byte-for-byte -
// PackedFloat3 (not simd::float3) for every vector field, same reasoning
// as that file's own comment: simd::float3 is 16-byte aligned inside a
// struct, PackedFloat3 is a plain 12-byte triple with no padding, and a
// host/device struct-layout mismatch is a silent, hard-to-spot bug class
// worth designing out rather than debugging into.
struct Uniforms {
    PackedFloat3 cameraPos;
    PackedFloat3 cameraForward;
    PackedFloat3 cameraRight;
    PackedFloat3 cameraUp;
    float tanHalfFov;
    float aspect;
    uint32_t width;
    uint32_t height;
    uint32_t samplesPerPixel;
    uint32_t maxDepth;
    uint32_t frameSeed;
    uint32_t lightCount;
    float lensRadius;
    float focusDistance;
    uint32_t apertureBlades;
    PackedFloat3 cameraVelocity;
    float fogSigmaT;
    PackedFloat3 fogAlbedo;
    uint32_t useEnvironmentMap;
    float fogAsymmetryG;
    uint32_t pointLightCount;
    uint32_t directionalLightCount;
    uint32_t adaptiveSampling;
};

// Mirrors metal_poc.metal's AreaLight byte-for-byte.
struct AreaLightData {
    PackedFloat3 center;
    PackedFloat3 edgeU;
    PackedFloat3 edgeV;
    PackedFloat3 normal;
    float area;
    PackedFloat3 emission;
    // Defaults keep every light before this one exactly flat (no
    // pattern) - see metal_poc.metal's own AreaLight comment.
    float patternTileB = 0.0f;
    float patternScale = 0.0f;
    // Power-proportional light-picking data (Vose alias table, built
    // host-side by buildPowerLightSampler() below, mirroring
    // src/shared/power_light_sampler_scaffold.h's own PowerLightSampler)
    // - see metal_poc.metal's AreaLight/sampleAreaLight() for how these
    // three get used. Defaults reproduce exact uniform 1/N picking (every
    // light before this one) if this ever got skipped: pmf = 0 would be
    // wrong, so buildPowerLightSampler() always runs, never left at these
    // raw defaults for an actual render.
    float pmf = 0.0f;
    float aliasProb = 1.0f;
    uint32_t aliasIndex = 0;
};

// Power-proportional light picking - a direct port of the Vose alias-
// table CONSTRUCTION algorithm in src/shared/power_light_sampler_scaffold.h's
// own PowerLightSampler::build() (that class itself is documented there as
// orphaned scaffolding with zero callers anywhere in this project, not
// wired into either the CPU or OptiX-GPU renderer - but the algorithm it
// implements is a real, correct port of pbrt-v4's own PowerLightSampler,
// exactly the kind of "tested reference, not proven-in-production
// caller" src/shared/ can still be worth porting from). Replaces this
// POC's own uniform 1/N area-light picking (section 18/52's own
// documented simplification: "a real port would... sample lights
// proportional to their own power"), so a bright light gets picked (and
// therefore NEE-sampled) more often than a dim one, reducing variance on
// the bright light without wasting samples equally on a light contributing
// almost nothing to the image.
//
// `power` here is `luminance(emission) * area` per light - proportional
// to each quad light's true total radiant power up to a factor (pi times
// a Lambertian-emitter solid-angle constant) that's the SAME for every
// light in this POC (all planar, all diffuse emitters), so it cancels
// out of the relative weighting a sampler only ever needs.
static void buildPowerLightSampler(std::vector<AreaLightData>& lights) {
    int n = (int)lights.size();
    if (n == 0) return;
    std::vector<double> power(n), pmf(n), scaled(n);
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        const PackedFloat3& e = lights[i].emission;
        double luminance = 0.2126 * e.x + 0.7152 * e.y + 0.0722 * e.z;
        power[i] = luminance * lights[i].area;
        total += power[i];
    }
    bool useUniform = (total <= 0.0);
    double uniformP = 1.0 / (double)n;
    for (int i = 0; i < n; ++i) {
        pmf[i] = useUniform ? uniformP : power[i] / total;
        scaled[i] = pmf[i] * (double)n;
        lights[i].pmf = (float)pmf[i];
        lights[i].aliasProb = 0.0f;
        lights[i].aliasIndex = (uint32_t)i;
    }
    std::vector<int> small, large;
    for (int i = 0; i < n; ++i) {
        (scaled[i] < 1.0 ? small : large).push_back(i);
    }
    while (!small.empty() && !large.empty()) {
        int s = small.back(); small.pop_back();
        int l = large.back(); large.pop_back();
        lights[s].aliasProb = (float)scaled[s];
        lights[s].aliasIndex = (uint32_t)l;
        scaled[l] = (scaled[l] + scaled[s]) - 1.0;
        (scaled[l] < 1.0 ? small : large).push_back(l);
    }
    while (!large.empty()) { int l = large.back(); large.pop_back(); lights[l].aliasProb = 1.0f; lights[l].aliasIndex = (uint32_t)l; }
    while (!small.empty()) { int s = small.back(); small.pop_back(); lights[s].aliasProb = 1.0f; lights[s].aliasIndex = (uint32_t)s; }
    for (int i = 0; i < n; ++i) {
        fprintf(stderr, "Light %d: power=%.4f pmf=%.4f (uniform would be %.4f)\n",
                i, power[i], pmf[i], 1.0 / (double)n);
    }
}

// Mirrors metal_poc.metal's PointLight byte-for-byte.
struct PointLightData {
    PackedFloat3 position;
    PackedFloat3 emission;
    // Defaults make every point light omnidirectional (see
    // metal_poc.metal's own PointLight/spotLightFalloff() comments) unless
    // explicitly overridden - the existing point light's own literal
    // doesn't need to change at all for this to stay backward compatible.
    PackedFloat3 direction = PackedFloat3{0.0f, -1.0f, 0.0f};
    float cosOuterAngle = -1.0f;
    float cosInnerAngle = -1.0f;
};

// Mirrors metal_poc.metal's DirectionalLight byte-for-byte.
struct DirectionalLightData {
    PackedFloat3 direction;
    PackedFloat3 emission;
};

// materialType: 0 = Lambertian, 1 = mirror, 2 = dielectric (glass) - see
// metal_poc.metal's own comment on this struct for why it's this minimal.
// ior is only meaningful for materialType == 2, emission only nonzero for
// the light quad - both carried on every entry anyway, see that file's
// comment on the same tradeoff.
struct TriangleMaterial {
    PackedFloat3 color;
    uint32_t materialType;
    float ior;
    PackedFloat3 emission;
    // Index into the AreaLight list this material's own triangles belong
    // to, or -1 for every non-emissive material - see metal_poc.metal's
    // own comment on the mirrored field.
    int32_t lightId = -1;
    // Rough-dielectric roughness (materialType == 5), ALSO reused as
    // anisotropic alphaY for materialType == 4 (0.0 there falls back to
    // isotropic), ALSO reused again as procedural bump strength for
    // materialType == 7 (0.0 there falls back to a perfectly flat
    // Lambertian, identical to materialType == 0) - see metal_poc.metal's
    // own comment on the mirrored field for the full explanation.
    float roughness = 0.0f;
};

// Mirrors metal_poc.metal's SphereData byte-for-byte.
struct SphereData {
    PackedFloat3 center;
    float radius;
};

// Mirrors metal_poc.metal's InstanceTransform byte-for-byte (4 packed
// columns, same layout MTLPackedFloat4x3 itself uses - see that file's
// own comment on why this side-channel buffer exists at all).
struct InstanceTransform {
    PackedFloat3 col0;
    PackedFloat3 col1;
    PackedFloat3 col2;
    PackedFloat3 col3;
};

// Mirrors metal_poc.metal's DiskData byte-for-byte.
struct DiskData {
    PackedFloat3 center;
    PackedFloat3 normal;
    float radius;
};

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
static void addQuad(std::vector<PackedFloat3>& verts,
                     std::vector<PackedFloat3>& normals,
                     std::vector<PackedFloat2>& uvs,
                     std::vector<TriangleMaterial>& materials,
                     float3 a, float3 b, float3 c, float3 d,
                     float3 color, uint32_t materialType = 0,
                     float3 emission = simd::make_float3(0, 0, 0),
                     int32_t lightId = -1, float roughness = 0.0f) {
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
    materials.push_back({packedColor, materialType, 1.0f, packedEmission, lightId, roughness});
    materials.push_back({packedColor, materialType, 1.0f, packedEmission, lightId, roughness});
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
static bool loadObjMesh(const std::string& path,
                         std::vector<PackedFloat3>& verts,
                         std::vector<PackedFloat3>& normals,
                         std::vector<PackedFloat2>& uvs,
                         std::vector<TriangleMaterial>& materials,
                         float3 color, float3 center, float targetSize,
                         uint32_t materialType = 0) {
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

    auto transform = [&](const float3& p) -> float3 {
        return (p - bboxCenter) * scale + center;
    };
    // Normals only need the scale's sign/shear behaviour, not translation -
    // a uniform positive scale (this loader's only kind) leaves direction
    // unchanged, so this is really just "no-op, pass through," kept as its
    // own step for clarity and in case a future non-uniform scale needs it.
    auto transformNormal = [&](const float3& n) -> float3 { return simd::normalize(n); };

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
    for (uint32_t i = 0; i < triangleCount; ++i) materials.push_back(mat);

    fprintf(stderr, "Loaded %s: %zu positions, %zu normals, %zu uvs, %u triangles "
                     "(%u flat-normal fallback, %u zero-uv fallback), scale %.4f\n",
            path.c_str(), positions.size(), fileNormals.size(), fileUVs.size(), triangleCount,
            normalFallbackCount, uvFallbackCount, scale);
    return true;
}

// Blackbody-temperature-to-RGB (Tanner Helland's widely-used polynomial
// fit to the Planckian locus) - converts an actual physical temperature
// in Kelvin to an approximate linear-ish RGB tint, valid over roughly
// [1000, 40000] K (clamped at the low end, where the fit diverges).
// Used below to derive several of this scene's own light colours from a
// NAMED physical quantity (5778 K for the directional "sun" light - the
// Sun's real photosphere temperature; 3000 K for a warm tungsten-style
// spot; 9000 K for a cool, moonlight-style point light) instead of a
// hand-picked RGB tuple - the same "derive it, don't guess it"
// preference this POC's own Fresnel/GGX/Beer-Lambert math already favors
// over an arbitrary-looking constant.
static float3 blackbodyColor(float kelvinIn) {
    float kelvin = fmaxf(kelvinIn, 1000.0f) / 100.0f;
    float r, g, b;
    if (kelvin <= 66.0f) {
        r = 255.0f;
    } else {
        r = 329.698727446f * powf(kelvin - 60.0f, -0.1332047592f);
    }
    if (kelvin <= 66.0f) {
        g = 99.4708025861f * logf(kelvin) - 161.1195681661f;
    } else {
        g = 288.1221695283f * powf(kelvin - 60.0f, -0.0755148492f);
    }
    if (kelvin >= 66.0f) {
        b = 255.0f;
    } else if (kelvin <= 19.0f) {
        b = 0.0f;
    } else {
        b = 138.5177312231f * logf(kelvin - 10.0f) - 305.0447927307f;
    }
    r = fminf(fmaxf(r, 0.0f), 255.0f) / 255.0f;
    g = fminf(fmaxf(g, 0.0f), 255.0f) / 255.0f;
    b = fminf(fmaxf(b, 0.0f), 255.0f) / 255.0f;
    return float3{r, g, b};
}

// Natural (lens) vignetting - real camera lenses transmit less light to
// the sensor at the edges/corners of the frame than at the centre (the
// classic `cos^4` falloff law), darkening corners even with no light
// physically blocked by a hood/filter ring ("mechanical" vignetting,
// which this is NOT modelling). Applied as a MULTIPLICATIVE attenuation
// on LINEAR radiance, before tonemapping/gamma - the physically correct
// place for it, the same reason tonemapping itself happens on linear
// values rather than after the gamma encode. `strength` scales how much
// the corners darken; 0.0 is an exact no-op (`vignette` reduces to 1.0
// everywhere), the same "0 reproduces prior behaviour exactly" shape
// this POC's own `lensRadius`/`apertureBlades` already use.
static float vignetteFactor(uint32_t x, uint32_t y, uint32_t width, uint32_t height, float strength) {
    float nx = (float(x) + 0.5f) / float(width) * 2.0f - 1.0f;
    float ny = (float(y) + 0.5f) / float(height) * 2.0f - 1.0f;
    float r2 = nx * nx + ny * ny;
    return 1.0f - strength * r2 * r2;
}

// Bilinearly samples ONE channel of the linear HDR pixel buffer at a
// fractional (x, y) - the building block chromaticAberration() below
// needs to resample the red/blue channels at a shifted position, unlike
// vignetteFactor()/acesFilmicTonemap() which only ever touch a pixel's
// own already-fetched value. Coordinates are clamped to the buffer's
// own bounds (a shift can walk slightly off the edge, especially near
// the frame's own corners) rather than reading out of range.
static float sampleChannelBilinear(const std::vector<float>& pixels, uint32_t width, uint32_t height,
                                    float x, float y, int channel) {
    x = fmaxf(0.0f, fminf(x, float(width) - 1.001f));
    y = fmaxf(0.0f, fminf(y, float(height) - 1.001f));
    uint32_t x0 = (uint32_t)x;
    uint32_t y0 = (uint32_t)y;
    uint32_t x1 = std::min(x0 + 1, width - 1);
    uint32_t y1 = std::min(y0 + 1, height - 1);
    float fx = x - float(x0);
    float fy = y - float(y0);
    float v00 = pixels[(y0 * width + x0) * 4 + channel];
    float v10 = pixels[(y0 * width + x1) * 4 + channel];
    float v01 = pixels[(y1 * width + x0) * 4 + channel];
    float v11 = pixels[(y1 * width + x1) * 4 + channel];
    float v0 = v00 * (1.0f - fx) + v10 * fx;
    float v1 = v01 * (1.0f - fx) + v11 * fx;
    return v0 * (1.0f - fy) + v1 * fy;
}

// Lateral chromatic aberration - a real lens focuses different
// wavelengths at very slightly different magnifications, so red/blue
// fringe outward/inward from green at the frame's own edges (worse
// toward the corners, exactly zero at the optical centre) - the same
// "colour fringing around high-contrast edges near a photo's own
// border" real camera lenses are well known for. Modelled the simplest
// physically-motivated way: the red channel is resampled from a
// position scaled slightly OUTWARD from frame centre, blue slightly
// INWARD, green left untouched as the reference channel - a pure
// radial scale about the centre already gives zero shift exactly at
// `r == 0` and a shift growing with radius everywhere else, with no
// need to compute `r` explicitly. `strength == 0.0` is an exact no-op
// (scale factors of 1.0, sampling each channel at its own unshifted
// position).
static void chromaticAberration(const std::vector<float>& pixels, uint32_t width, uint32_t height,
                                 uint32_t px, uint32_t py, float strength,
                                 float* outR, float* outG, float* outB) {
    // Index-space centre (NOT pixel-CENTRE `+0.5` convention) - matching
    // sampleChannelBilinear()'s own convention, where an integer (x, y)
    // means "exactly pixel (x, y)", not "the corner before it". Mixing
    // the two conventions here was a real bug this PR's own first
    // attempt had: computing `cx` as `width * 0.5` while adding `+ 0.5`
    // to `px` left the "centre" pixel's own `dx` at exactly 0.5 instead
    // of 0.0, so even the un-shifted red/blue channels sampled a blend
    // of two neighbouring pixels there instead of the exact centre pixel
    // itself - caught by rendering an ODD-sized image (so a true single
    // centre pixel exists) and finding its R/B channels had shifted
    // anyway, which should be mathematically impossible at true zero
    // shift.
    float cx = float(width - 1) * 0.5f;
    float cy = float(height - 1) * 0.5f;
    float dx = float(px) - cx;
    float dy = float(py) - cy;
    float redX = cx + dx * (1.0f + strength);
    float redY = cy + dy * (1.0f + strength);
    float blueX = cx + dx * (1.0f - strength);
    float blueY = cy + dy * (1.0f - strength);
    *outR = sampleChannelBilinear(pixels, width, height, redX, redY, 0);
    *outG = pixels[(py * width + px) * 4 + 1];
    *outB = sampleChannelBilinear(pixels, width, height, blueX, blueY, 2);
}

// ACES filmic tonemap (Krzysztof Narkowicz's widely-used fitted
// approximation of the ACES RRT+ODT curve) - replaces this POC's old
// direct clamp-to-[0,1] before the 8-bit gamma encode. A raw linear
// radiance value above 1.0 used to hard-clip straight to solid white -
// every area/point/spot/directional light's own bright core, and any
// surface catching more than one of them at once, does this constantly,
// visible as flat white patches with no rolloff in this POC's own
// earlier renders (the ceiling light panels, the directional light's own
// raking highlight). This rolls off smoothly instead, the standard
// reason a renderer tonemaps before display rather than clamping raw
// linear radiance - applied per channel, in LINEAR space, BEFORE the
// sRGB-ish gamma encode below (the correct ordering; gamma-encoding
// first and tonemapping second would double-correct the curve).
static float acesFilmicTonemap(float x) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    float mapped = (x * (a * x + b)) / (x * (c * x + d) + e);
    return fminf(fmaxf(mapped, 0.0f), 1.0f);
}

// Reinhard tonemap (`L' = L / (1 + L)`) - ported directly from this
// project's own CPU/OptiX-shared reference (`src/shared/tone_map.h`'s own
// `reinhard()`). Simpler and cheaper than ACES, with a very different
// character: it preserves hue at low-to-moderate values (each channel maps
// independently, but the curve's own compression is gentle enough there
// that saturated colours stay saturated) but has no filmic "shoulder" at
// all - bright values approach 1.0 much more gradually and asymptotically
// rather than rolling off with ACES's own S-curve, which in practice reads
// as a flatter, less contrasty, slightly washed-out highlight rendition
// next to ACES on the same scene.
static float reinhardTonemap(float x) {
    if (x <= 0.0f) return 0.0f;
    return x / (1.0f + x);
}

// Mirrors `src/shared/tone_map.h`'s own `ToneMapMode` - kept as a plain
// enum (not the shared header itself) since that header also drags in
// `tone_map_mode_from_name()`'s std::string plumbing this positional-args
// POC has no use for; the three named values and their meaning are what's
// shared, not the parsing machinery.
enum class ToneMapMode {
    ACES,
    Reinhard,
    None
};

// Same three names `--tonemap` accepts on the CPU/OptiX side
// (`tone_map_mode_from_name()`), so a scene/workflow note like "compare
// with --tonemap reinhard" translates directly to this POC's own 6th
// positional argument. Falls back to ACES (this POC's long-standing
// default, unchanged) on anything unrecognized rather than failing the
// render outright - there's no scene/CLI split to warn about a mismatch
// for here, just one flat argument list.
static ToneMapMode parseToneMapMode(const char* name) {
    if (!name) return ToneMapMode::ACES;
    if (strcmp(name, "reinhard") == 0) return ToneMapMode::Reinhard;
    if (strcmp(name, "none") == 0) return ToneMapMode::None;
    return ToneMapMode::ACES;
}

static float applyToneMap(float x, ToneMapMode mode) {
    switch (mode) {
        case ToneMapMode::Reinhard: return reinhardTonemap(x);
        case ToneMapMode::None:     return fminf(fmaxf(x, 0.0f), 1.0f);
        case ToneMapMode::ACES:
        default:                    return acesFilmicTonemap(x);
    }
}

// The REAL sRGB OETF (IEC 61966-2-1) - a numerically-stable minimax
// rational-polynomial approximation of the true piecewise curve
// (`12.92 * v` below a small threshold, `1.055 * v^(1/2.4) - 0.055`
// above it), ported directly from this project's own CPU renderer
// (src/shared/color_encoding.h's own LinearToSRGB(), itself mirroring
// pbrt-v4/enoki exactly) - replaces this POC's own long-standing
// `powf(v, 1.0f/2.2f)` approximation, the SAME kind of "close but not
// exact" gap step 38's own sRGB-DECODE fix closed on the read side
// (loading a texture); this closes the matching gap on the WRITE side
// (encoding the final image). A flat 1/2.2 power curve has no linear
// toe segment near black at all, so the two curves diverge most in
// shadows/near-black tones, not midtones - exactly where this fix's own
// verification render should show the clearest difference.
static float linearToSRGB(float value) {
    if (value <= 0.0031308f) {
        return 12.92f * value;
    }
    float s = sqrtf(value);
    float p = -0.0016829072605308378f
            + s * (  0.03453868659826638f
            + s * (  0.7642611304733891f
            + s * (  2.0041169284241644f
            + s * (  0.7551545191665577f
            + s * (-0.016202083165206348f)))));
    float q =  4.178892964897981e-7f
            + s * (-0.00004375359692957097f
            + s * (  0.03467195408529984f
            + s * (  0.6085338522168684f
            + s * (  1.8970238036421054f
            + s))));
    return p / q * value;
}

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
static void bilateralDenoise(const std::vector<uint8_t>& ldrIn, std::vector<uint8_t>& ldrOut,
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

int main(int argc, const char** argv) {
    @autoreleasepool {
        const uint32_t width = (argc > 1) ? (uint32_t)atoi(argv[1]) : 400;
        const uint32_t height = (argc > 2) ? (uint32_t)atoi(argv[2]) : 400;
        const char* outPath = (argc > 3) ? argv[3] : "/tmp/metal_poc_render.png";
        const ToneMapMode toneMapMode = parseToneMapMode((argc > 6) ? argv[6] : nullptr);

        // MTLCreateSystemDefaultDevice() is explicitly documented as
        // unsupported for command-line/daemon processes (confirmed via
        // `log show`: "Use of MTLCreateSystemDefaultDevice is not
        // supported for non-interactive (commandline or daemon) apps. Use
        // MTLCopyAllDevices(WithObserver) instead.") - this POC is a plain
        // CLI tool, not an app bundle, so it needs the enumeration API.
        id<MTLDevice> device = nil;
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices.count > 0) device = devices[0];
        if (!device) {
            fprintf(stderr, "No Metal device available.\n");
            return 1;
        }
        fprintf(stderr, "Metal device: %s\n", device.name.UTF8String);
        if (!device.supportsRaytracing) {
            fprintf(stderr, "Device does not support hardware raytracing.\n");
            return 1;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];

        // --- Scene: a Cornell-box-like room, world units ~[-1,1] --------
        // Matches this project's CPU Cornell box in spirit (floor/ceiling/
        // back wall + coloured side walls + an object), not in exact
        // dimensions - this POC's scene is entirely separate authored data,
        // not a shared asset with cpu_renderer/.
        std::vector<PackedFloat3> verts;
        std::vector<PackedFloat3> normals;
        std::vector<PackedFloat2> uvs;
        std::vector<TriangleMaterial> materials;

        const float3 white{0.73f, 0.73f, 0.73f};
        const float3 red{0.65f, 0.05f, 0.05f};
        const float3 green{0.12f, 0.45f, 0.15f};

        // Floor (y = -1) - procedural checkerboard (materialType 6), the
        // one surface in the scene that samples NO texture/image at all
        // for its albedo, a genuinely different technique from
        // materialType 3's earthTexture lookup (analytic UV math instead
        // of a sampler call). addQuad()'s own planar 0-1 UVs across the
        // whole floor, combined with checkerColor()'s own 8-tiles-per-UV-
        // unit scale, give 8x8 tiles across the room's own floor.
        addQuad(verts, normals, uvs, materials, float3{-1,-1,-1}, float3{1,-1,-1}, float3{1,-1,1}, float3{-1,-1,1}, white, /*materialType=*/6);
        // Ceiling (y = 1)
        addQuad(verts, normals, uvs, materials, float3{-1,1,1}, float3{1,1,1}, float3{1,1,-1}, float3{-1,1,-1}, white);
        // Back wall (z = -1) - textured (materialType 3): the one surface
        // in the scene that samples earthTexture, chosen because it's the
        // large flat backdrop the camera looks straight at, showing the
        // whole 0-1 UV mapping unobstructed.
        addQuad(verts, normals, uvs, materials, float3{-1,-1,-1}, float3{-1,1,-1}, float3{1,1,-1}, float3{1,-1,-1}, white, /*materialType=*/3);
        // Left wall (x = -1), red
        addQuad(verts, normals, uvs, materials, float3{-1,-1,1}, float3{-1,1,1}, float3{-1,1,-1}, float3{-1,-1,-1}, red);
        // Right wall (x = 1), green
        addQuad(verts, normals, uvs, materials, float3{1,-1,-1}, float3{1,1,-1}, float3{1,1,1}, float3{1,-1,1}, green);
        // A small procedurally bump-mapped panel (materialType 7),
        // flush-mounted just in front of the back wall (z = -0.99, the
        // same off-surface margin the mirror disk/other flush-mounted
        // geometry already uses to avoid z-fighting) rather than a side
        // wall - the one surface in the scene whose shading normal is
        // perturbed AWAY from its own true (perfectly flat) geometric
        // normal, an analytic egg-carton height field rather than a
        // stored normal-map texture (no new image asset needed - see
        // metal_poc.metal's own proceduralBumpNormal() comment).
        // Positioned in the region the directional light (see
        // metal_poc.metal's own DirectionalLight comment) hits closest to
        // head-on, not tucked against a side wall - a first attempt
        // mounted on the red wall got barely any direct light at all
        // (nearly the same shallow self-shadowing the directional light's
        // own doc describes for that wall), making the bump invisible
        // under GI-only ambient lighting; moved here after that render
        // came back looking completely flat, not assumed correct from
        // the code alone. `roughness` here means bump strength, not a
        // BRDF parameter - materialType 7's own reuse of that field, see
        // TriangleMaterial's comment above.
        addQuad(verts, normals, uvs, materials,
                float3{0.15f, -0.3f, -0.99f}, float3{0.15f, 0.3f, -0.99f},
                float3{0.75f, 0.3f, -0.99f}, float3{0.75f, -0.3f, -0.99f},
                float3{0.55f, 0.5f, 0.45f}, /*materialType=*/7,
                /*emission=*/simd::make_float3(0, 0, 0), /*lightId=*/-1,
                /*roughness(bump strength)=*/0.6f);
        // Suzanne (Blender's monkey mascot, models/suzanne.obj - a real
        // mesh, 500 faces) replaces the earlier flat tilted-quad "mirror
        // test object": mirror MATERIAL coverage is already proven (the
        // sphere/dielectric PR's own mirror-quad screenshots), what this
        // scene hadn't tested yet is real, DATA-DRIVEN geometry with
        // genuine per-triangle normal variation, not a hand-authored
        // axis-aligned quad. Lambertian so its form reads clearly via
        // shading rather than showing room reflections. RT_MODELS_DIR
        // mirrors RT_METAL_SHADER_DIR's own fallback shape below.
        //
        // Loaded into ITS OWN vertex/normal/uv/material vectors (not the
        // shared `verts`/`normals`/`uvs`/`materials` the room quads and
        // Spot still use) - Suzanne gets her own acceleration structure
        // below, instanced twice with two DIFFERENT transforms, the one
        // piece of this scene that actually exercises a non-identity
        // instance transform. A shared-buffer object can only ever have
        // ONE position (it's baked directly into world-space vertex
        // positions at load time); testing instancing needs the SAME
        // object-space geometry referenced from more than one instance
        // descriptor, which needs its own acceleration structure.
#ifdef RT_MODELS_DIR
        NSString* modelsDir = @(RT_MODELS_DIR);
#else
        NSString* modelsDir = [[@(__FILE__) stringByDeletingLastPathComponent]
            stringByAppendingPathComponent:@"../../models"];
#endif
        NSString* suzannePath = [modelsDir stringByAppendingPathComponent:@"suzanne.obj"];
        const float3 bronze{0.55f, 0.35f, 0.15f};
        std::vector<PackedFloat3> suzanneVerts;
        std::vector<PackedFloat3> suzanneNormals;
        std::vector<PackedFloat2> suzanneUVs;
        std::vector<TriangleMaterial> suzanneMaterials;
        if (!loadObjMesh(suzannePath.UTF8String, suzanneVerts, suzanneNormals, suzanneUVs, suzanneMaterials, bronze,
                          /*center=*/float3{0.0f, 0.0f, 0.0f}, /*targetSize=*/0.75f)) {
            fprintf(stderr, "Continuing without Suzanne - check RT_MODELS_DIR / models/suzanne.obj.\n");
        }

        // Spot (Keenan Crane's textured cow model, models/spot.obj) - unlike
        // suzanne.obj, this file has real per-face-corner `vt` data (3225
        // entries) and NO `vn` at all, the exact inverse case from Suzanne's
        // own (vn on every face, no vt) - loading it exercises
        // loadObjMesh()'s real-UV path (not just its zero-fallback path)
        // and its flat-normal fallback path in the same call, closing the
        // texture-mapping PR's own explicitly-deferred "real vt/f v/vt/vn
        // parsing" item. materialType 3 (textured) reuses `earthTexture` -
        // wrapping a world map onto a cow is a deliberately silly texture
        // choice for a mesh that was never authored to use it, but it's
        // exactly what makes this a REAL demonstration of per-vertex UV
        // interpolation rather than a coincidentally-plausible-looking
        // result: the world map's grid lines and coastlines have to follow
        // spot's actual surface curvature for this to look right at all.
        NSString* spotPath = [modelsDir stringByAppendingPathComponent:@"spot.obj"];
        if (!loadObjMesh(spotPath.UTF8String, verts, normals, uvs, materials, white,
                          /*center=*/float3{0.78f, -0.75f, 0.6f}, /*targetSize=*/0.42f,
                          /*materialType=*/3)) {
            fprintf(stderr, "Continuing without Spot - check RT_MODELS_DIR / models/spot.obj.\n");
        }

        // Area lights: real geometry, hanging just under the ceiling
        // (y=0.98, not y=1 itself - avoids z-fighting/coplanar overlap
        // with the ceiling's own quad above), facing straight down. Two
        // separate lights (not one, as every previous PR up through #11
        // had) - the smallest scene change that actually exercises
        // `lights` as a genuine LIST rather than a single renamed
        // constant: a warm light and a cool light side by side prove the
        // shader's own light-picking/MIS code path handles more than one
        // entry, visibly (two independently-coloured highlights/shadow
        // directions), not just structurally. `addAreaLight` keeps each
        // call's geometry (addQuad, tagged with this light's own index
        // via materialType/lightId) and its AreaLightData entry (same
        // corners, reduced to center/edgeU/edgeV/normal/area) in sync by
        // construction, rather than needing two hand-authored, separately-
        // maintained descriptions of the same quad the way the single-
        // light version's kLightCenter/kLightHalfExtents/kLightNormal
        // constants over in metal_poc.metal used to (see that file's
        // AreaLight struct comment for the "replacing..." history).
        std::vector<AreaLightData> lights;
        // `patternTileB`/`patternScale` default to 0 (flat emission,
        // materialType 0) - see AreaLightData's own comment. Passing a
        // nonzero `patternScale` switches the light's own triangles to
        // materialType 10 too, so a direct hit and an NEE sample both
        // evaluate the SAME checker pattern (just at each one's own
        // different point on the light - see metal_poc.metal's own
        // comments on materialType 10 and AreaLight for why that needs
        // two separate evaluations, not one shared value).
        auto addAreaLight = [&](float3 a, float3 b, float3 c, float3 d, float3 emission,
                                 float patternTileB = 0.0f, float patternScale = 0.0f) {
            int32_t lightId = (int32_t)lights.size();
            uint32_t materialType = (patternScale > 0.0f) ? 10u : 0u;
            addQuad(verts, normals, uvs, materials, a, b, c, d, white,
                    materialType, emission, lightId, /*roughness(pattern tileB)=*/patternTileB);
            float3 edgeU = b - a;
            float3 edgeV = d - a;
            float3 normal = simd::normalize(simd::cross(edgeU, edgeV));
            float area = simd::length(simd::cross(edgeU, edgeV));
            float3 center = a + 0.5f * edgeU + 0.5f * edgeV;
            lights.push_back(AreaLightData{
                PackedFloat3{center.x, center.y, center.z},
                PackedFloat3{edgeU.x, edgeU.y, edgeU.z},
                PackedFloat3{edgeV.x, edgeV.y, edgeV.z},
                PackedFloat3{normal.x, normal.y, normal.z},
                area,
                PackedFloat3{emission.x, emission.y, emission.z},
                patternTileB,
                patternScale});
        };
        // Both ceiling lights' own colours now come from blackbodyColor()
        // (see that function's own comment, added for the point/spot/
        // directional lights) rather than the hand-picked tuples above -
        // 2700 K (a standard incandescent bulb) for the warm one, 20000 K
        // (near the top of the fit's own valid range, a very hot/blue-
        // white source) for the cool one. Notably, 20000K's own derived
        // colour is nowhere near as saturated a blue as the hand-picked
        // (6, 10, 18) it replaces - real blackbody radiation never gets
        // that saturated; no amount of temperature produces a deeply
        // saturated blue the way an artistic RGB pick can. A real,
        // honestly-reported limitation of deriving colour from physical
        // temperature, not swept under the rug.
        const float3 warmAreaLightColor = blackbodyColor(2700.0f) * 15.5f;
        const float3 coolAreaLightColor = blackbodyColor(20000.0f) * 13.9f;
        addAreaLight(float3{-0.58f,0.98f,-0.25f}, float3{-0.22f,0.98f,-0.25f},
                     float3{-0.22f,0.98f,0.25f}, float3{-0.58f,0.98f,0.25f},
                     /*emission=*/warmAreaLightColor);
        // The cool light also gets a patterned diffuser-grid look
        // (materialType 10 - see that comment for the full "why"), a
        // real fixture detail the flat-emission warm light doesn't have:
        // tile B at 40% of tile A's own brightness (a translucent grid,
        // not fully opaque black bars) across a 6x6 tiling of the
        // light's own 0-1 UV span.
        addAreaLight(float3{0.22f,0.98f,-0.25f}, float3{0.58f,0.98f,-0.25f},
                     float3{0.58f,0.98f,0.25f}, float3{0.22f,0.98f,0.25f},
                     /*emission=*/coolAreaLightColor,
                     /*patternTileB=*/0.4f, /*patternScale=*/6.0f);
        // Builds each light's own pmf/aliasProb/aliasIndex in place - see
        // buildPowerLightSampler()'s own comment. Must run after every
        // addAreaLight() call above (needs the full, final light list) and
        // before `lights` gets uploaded to the GPU buffer below.
        buildPowerLightSampler(lights);

        // Two spheres, both custom (non-triangle) primitives via a shared
        // bounding-box acceleration structure + intersection function
        // (metal_poc.metal's sphereIntersectionFunction indexes into
        // `spheres`/`sphereMaterials` by primitive_id, so any number of
        // spheres share one geometry/one intersection function - adding a
        // second one below is purely a host-side array-of-2 change, no
        // shader change) - the one "Medium risk, unconfirmed" item
        // docs/METAL_GPU_FEASIBILITY.md section 3 originally flagged that
        // the room/mirror geometry alone hadn't exercised (triangles only).
        // Sphere 0: glass, ior 1.5 matches common glass, same value this
        // project's own CPU Cornell box scene (A1) uses for its glass
        // sphere. Sphere 1: a rough (GGX) conductor - gold-ish F0, roughness
        // 0.15 (a fairly tight but visibly non-mirror highlight), placed on
        // the opposite side of the room so both new-material spheres read
        // clearly side by side.
        // Sphere 2: rough (frosted) dielectric, materialType 5 - small,
        // front-and-centre between the other two spheres and just in
        // front of Suzanne. An earlier back-left-corner placement turned
        // out to sit almost exactly along the camera-to-gold-sphere
        // sightline (both ~20% off-axis, gold sphere much closer/larger)
        // and was fully hidden - caught by actually rendering and
        // inspecting the image, not by the bounding-region math alone
        // (which only rules out 3D overlap, not 2D screen-space
        // occlusion) - this position was checked against both.
        // Sphere 3: a procedurally roughness-mapped GGX conductor
        // (materialType 9) - a "worn/scratched copper" look, patches of
        // near-mirror-smooth and rough microfacet regions on the SAME
        // surface (see metal_poc.metal's own materialType 9 comment).
        // Genuinely different from sphere 1's own anisotropic roughness:
        // that one varies BY DIRECTION at a single point (one alpha per
        // tangent axis, constant everywhere on the sphere); this one
        // varies BY LOCATION (alpha itself is a function of where on the
        // sphere you look, isotropic at any single point). Placed
        // resting on the floor (`y = -1 + radius`, matching every other
        // floor sphere's own convention) at the room's front-right,
        // clear of the gold sphere/Spot/the disk mirror - checked by
        // rendering and inspecting, not just the bounding-sphere math.
        // Sphere 4: clearcoat/glossy-plastic (materialType 8) - a deep
        // red "car paint" look, a sharp specular highlight riding on top
        // of a genuinely diffuse (not metallic) coloured base, the
        // signature that distinguishes this from every reflective
        // material already in the scene (mirror/GGX conductor are
        // colour-tinted AT the reflection itself; clearcoat's own
        // reflection stays colourless/white, only the diffuse base
        // beneath carries colour). Placed at the room's front-right,
        // deliberately at a different x AND z from Spot-the-cow (this
        // POC's own established near-miss from sphere 3's own placement:
        // sharing an x coordinate with a closer foreground object hid it
        // completely) and the gold sphere.
        std::vector<SphereData> spheres = {
            SphereData{PackedFloat3{0.35f, -0.65f, 0.15f}, 0.35f},
            SphereData{PackedFloat3{-0.55f, -0.65f, 0.45f}, 0.35f},
            SphereData{PackedFloat3{-0.05f, -0.82f, 0.6f}, 0.18f},
            SphereData{PackedFloat3{-0.78f, -0.78f, 0.7f}, 0.18f},
            SphereData{PackedFloat3{0.8f, -0.85f, 1.0f}, 0.15f},
        };
        // Sphere 0 and 2's own `color` is now a Beer-Lambert ABSORPTION
        // coefficient (see metal_poc.metal's own applyBeerLambertAbsorption()
        // comment), not a reflectance/tint the way every other material's
        // `color` field is read - {1,1,1} would have meant "absorb
        // everything, render black" under this new interpretation, so
        // both dielectric spheres' old placeholder {1,1,1} "clear glass"
        // values were replaced with real per-channel absorption:
        // sphere 0 is emerald-tinted (absorbs red/blue faster than
        // green, getting more richly green toward its own thicker
        // centre), sphere 2 a much milder amber (still reads mostly
        // frosted-white, just warmed slightly).
        std::vector<TriangleMaterial> sphereMaterials = {
            TriangleMaterial{PackedFloat3{0.5f, 0.05f, 0.35f}, /*materialType=*/2, /*ior=*/1.5f, PackedFloat3{0, 0, 0}},
            // Genuinely ANISOTROPIC now (alphaX from `ior`, alphaY from
            // `roughness` - see TriangleMaterial's own comment): a tight
            // 0.08 in one tangent direction and a much broader 0.45 in
            // the other, the classic "brushed metal" look - a real,
            // deliberate change from the previously-isotropic 0.15 this
            // sphere used through step 22, not a value chosen to
            // preserve the old appearance (that A/B check is done via a
            // dedicated verification render instead, not the committed
            // scene - see docs/METAL_GPU_FEASIBILITY.md's own note).
            TriangleMaterial{PackedFloat3{1.0f, 0.86f, 0.57f}, /*materialType=*/4, /*alphaX=*/0.08f, PackedFloat3{0, 0, 0},
                             /*lightId=*/-1, /*alphaY=*/0.45f},
            TriangleMaterial{PackedFloat3{0.12f, 0.08f, 0.02f}, /*materialType=*/5, /*ior=*/1.5f, PackedFloat3{0, 0, 0},
                             /*lightId=*/-1, /*roughness=*/0.35f},
            // materialType 9: `ior` is the SMOOTH patch's own perceptual
            // roughness, `roughness` the ROUGH patch's - both squared
            // into GGX alpha exactly like materialType 4 already does,
            // just picked between by an analytic UV-space checker
            // pattern instead of being one constant.
            TriangleMaterial{PackedFloat3{0.8f, 0.45f, 0.2f}, /*materialType=*/9, /*ior(smooth)=*/0.05f, PackedFloat3{0, 0, 0},
                             /*lightId=*/-1, /*roughness(rough)=*/0.6f},
            // materialType 8: `color` is the diffuse BASE colour under
            // the coat (materialType 0's own convention) - a deep,
            // fairly saturated red, since the coat's own reflection
            // stays colourless regardless.
            TriangleMaterial{PackedFloat3{0.55f, 0.05f, 0.06f}, /*materialType=*/8, /*ior=*/1.0f, PackedFloat3{0, 0, 0}},
        };

        // A wall-mounted mirror disk (materialType 1) - a second, distinct
        // custom-primitive SHAPE, not just another sphere. Every custom
        // primitive so far (however many) has gone through the SAME
        // intersection function at function-table slot 0; this is what
        // actually exercises a second, different function at slot 1 (see
        // metal_poc.metal's own comment on diskIntersectionFunction).
        // Also, incidentally, the first object in this whole scene to use
        // materialType 1 (mirror) at all - it's existed in the shader
        // since the very first multi-material step but nothing had
        // actually used it since the mirror test quad was replaced by the
        // dielectric sphere back in step 6.
        std::vector<DiskData> disks = {
            DiskData{PackedFloat3{0.97f, 0.3f, -0.3f}, PackedFloat3{-1.0f, 0.0f, 0.0f}, 0.22f},
        };
        std::vector<TriangleMaterial> diskMaterials = {
            TriangleMaterial{PackedFloat3{0.9f, 0.9f, 0.9f}, /*materialType=*/1, /*ior=*/1.0f, PackedFloat3{0, 0, 0}},
        };

        // A true delta point light - genuinely different from every
        // AreaLight above (zero area, hard-edged shadows, no NEE/MIS
        // weighting needed at all - see metal_poc.metal's own PointLight
        // comment). Placed off-axis from both area lights so it adds a
        // THIRD, distinctly-positioned specular highlight to the
        // reflective spheres/disk rather than blending into an existing
        // one - the easiest way to visually confirm it's really
        // contributing light, not just present in the buffer unused.
        // Second point light: a genuine SPOT (cone-restricted), unlike the
        // first one's omnidirectional glow - aimed down at Spot-the-cow's
        // own floor area, a real "pool of light" cone signature an
        // omnidirectional point light cannot produce at all (its own
        // illumination falls off with distance everywhere, never with
        // ANGLE the way a spot's does). 25 degree outer / 15 degree inner
        // cone (smoothstep-blended between them, not a hard edge).
        //
        // All three lights below get their own colour from
        // blackbodyColor() at a NAMED physical temperature rather than a
        // hand-picked RGB tuple - 9000 K (cool, moonlight-ish) for this
        // first point light, 3000 K (warm tungsten) for the spot, 5778 K
        // (the Sun's own real photosphere temperature) for the
        // directional light below. Each still scaled by a plain
        // intensity multiplier chosen to land in roughly the same
        // brightness range this scene's own lights already used - only
        // the HUE is now derived, not the overall exposure.
        const float3 spotPos = float3{0.65f, 0.9f, -0.1f};
        const float3 spotTarget = float3{0.7f, -1.0f, 0.4f};
        const float3 spotDir = simd::normalize(spotTarget - spotPos);
        const float3 pointLight1Color = blackbodyColor(9000.0f) * 1.0f;
        const float3 spotLightColor = blackbodyColor(3000.0f) * 11.3f;
        std::vector<PointLightData> pointLights = {
            PointLightData{PackedFloat3{0.0f, 0.3f, 0.3f}, PackedFloat3{pointLight1Color.x, pointLight1Color.y, pointLight1Color.z}},
            PointLightData{PackedFloat3{spotPos.x, spotPos.y, spotPos.z}, PackedFloat3{spotLightColor.x, spotLightColor.y, spotLightColor.z},
                           PackedFloat3{spotDir.x, spotDir.y, spotDir.z},
                           /*cosOuterAngle=*/cosf(25.0f * (float)M_PI / 180.0f),
                           /*cosInnerAngle=*/cosf(15.0f * (float)M_PI / 180.0f)},
        };

        // A directional ("sun") light - genuinely different in KIND from
        // both point-light entries above: parallel rays with no position
        // and no distance falloff at all, rather than one more delta
        // light radiating from a finite point (see metal_poc.metal's own
        // DirectionalLight comment). Aimed through the room's own open
        // front (the z=1 face has no wall - see the floor/ceiling/wall
        // addQuad() calls above): `direction` points mostly along -z with
        // a slight -y/+x tilt, so tracing back toward the light from
        // anywhere in the [-1,1]^3 room exits through that open face
        // before it would cross the ceiling (y=1) or either side wall,
        // rather than being trivially self-shadowed by this room's own
        // geometry on every shading point.
        const float3 sunColor = blackbodyColor(5778.0f) * 2.7f;
        std::vector<DirectionalLightData> directionalLights = {
            DirectionalLightData{PackedFloat3{0.1f, -0.15f, -1.0f}, PackedFloat3{sunColor.x, sunColor.y, sunColor.z}},
        };

        const uint32_t triangleCount = (uint32_t)materials.size();
        fprintf(stderr, "Scene: %u triangles, %zu spheres, %zu disks, %zu lights, %zu point lights, %zu directional lights\n",
                triangleCount, spheres.size(), disks.size(), lights.size(), pointLights.size(), directionalLights.size());

        id<MTLBuffer> vertexBuffer = [device newBufferWithBytes:verts.data()
            length:verts.size() * sizeof(PackedFloat3)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> normalBuffer = [device newBufferWithBytes:normals.data()
            length:normals.size() * sizeof(PackedFloat3)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> uvBuffer = [device newBufferWithBytes:uvs.data()
            length:uvs.size() * sizeof(PackedFloat2)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> lightBuffer = [device newBufferWithBytes:lights.data()
            length:lights.size() * sizeof(AreaLightData)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> pointLightBuffer = [device newBufferWithBytes:pointLights.data()
            length:pointLights.size() * sizeof(PointLightData)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> directionalLightBuffer = [device newBufferWithBytes:directionalLights.data()
            length:directionalLights.size() * sizeof(DirectionalLightData)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> materialBuffer = [device newBufferWithBytes:materials.data()
            length:materials.size() * sizeof(TriangleMaterial)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> sphereBuffer = [device newBufferWithBytes:spheres.data()
            length:spheres.size() * sizeof(SphereData) options:MTLResourceStorageModeShared];
        id<MTLBuffer> sphereMaterialBuffer = [device newBufferWithBytes:sphereMaterials.data()
            length:sphereMaterials.size() * sizeof(TriangleMaterial) options:MTLResourceStorageModeShared];
        id<MTLBuffer> diskBuffer = [device newBufferWithBytes:disks.data()
            length:disks.size() * sizeof(DiskData) options:MTLResourceStorageModeShared];
        id<MTLBuffer> diskMaterialBuffer = [device newBufferWithBytes:diskMaterials.data()
            length:diskMaterials.size() * sizeof(TriangleMaterial) options:MTLResourceStorageModeShared];

        const uint32_t suzanneTriangleCount = (uint32_t)suzanneMaterials.size();
        id<MTLBuffer> suzanneVertexBuffer = [device newBufferWithBytes:suzanneVerts.data()
            length:suzanneVerts.size() * sizeof(PackedFloat3) options:MTLResourceStorageModeShared];
        id<MTLBuffer> suzanneNormalBuffer = [device newBufferWithBytes:suzanneNormals.data()
            length:suzanneNormals.size() * sizeof(PackedFloat3) options:MTLResourceStorageModeShared];
        id<MTLBuffer> suzanneMaterialBuffer = [device newBufferWithBytes:suzanneMaterials.data()
            length:suzanneMaterials.size() * sizeof(TriangleMaterial) options:MTLResourceStorageModeShared];

        // --- Primitive acceleration structure (the mesh's own BVH) ------
        MTLAccelerationStructureTriangleGeometryDescriptor* geomDesc =
            [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
        geomDesc.vertexBuffer = vertexBuffer;
        geomDesc.vertexStride = sizeof(PackedFloat3);
        geomDesc.triangleCount = triangleCount;
        // Explicit, not relying on the default: opaque means no any-hit
        // shader gets consulted for this geometry's hits at all, so the
        // hardware triangle intersector's result is taken directly - this
        // matters once a function table is bound at trace time at all
        // (added below, for the sphere), since without this a triangle
        // hit could otherwise get routed through the SAME table slot the
        // sphere's own intersection function occupies.
        geomDesc.opaque = YES;

        MTLPrimitiveAccelerationStructureDescriptor* primDesc =
            [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        primDesc.geometryDescriptors = @[geomDesc];

        MTLAccelerationStructureSizes primSizes = [device accelerationStructureSizesWithDescriptor:primDesc];
        id<MTLAccelerationStructure> primAS = [device newAccelerationStructureWithSize:primSizes.accelerationStructureSize];
        id<MTLBuffer> primScratch = [device newBufferWithLength:primSizes.buildScratchBufferSize
            options:MTLResourceStorageModePrivate];

        id<MTLCommandBuffer> buildCmd = [queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> buildEnc = [buildCmd accelerationStructureCommandEncoder];
        [buildEnc buildAccelerationStructure:primAS descriptor:primDesc scratchBuffer:primScratch scratchBufferOffset:0];
        [buildEnc endEncoding];
        [buildCmd commit];
        [buildCmd waitUntilCompleted];
        if (buildCmd.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "Primitive AS build failed: %s\n", buildCmd.error.localizedDescription.UTF8String);
            return 1;
        }

        // --- Second primitive acceleration structure: the spheres' own --
        // bounding-box geometry (a custom/non-triangle primitive has no
        // vertex data at all as far as the acceleration structure is
        // concerned - just an AABB per primitive, with the real
        // intersection test deferred to sphereIntersectionFunction at
        // trace time). One AABB per entry in `spheres`, same index order -
        // sphereIntersectionFunction's own primitive_id indexes both this
        // buffer and `spheres`/`sphereMaterials` identically.
        std::vector<MTLAxisAlignedBoundingBox> sphereBoundsList;
        for (const SphereData& s : spheres) {
            MTLAxisAlignedBoundingBox bounds;
            bounds.min = MTLPackedFloat3Make(s.center.x - s.radius, s.center.y - s.radius, s.center.z - s.radius);
            bounds.max = MTLPackedFloat3Make(s.center.x + s.radius, s.center.y + s.radius, s.center.z + s.radius);
            sphereBoundsList.push_back(bounds);
        }
        id<MTLBuffer> boundingBoxBuffer = [device newBufferWithBytes:sphereBoundsList.data()
            length:sphereBoundsList.size() * sizeof(MTLAxisAlignedBoundingBox) options:MTLResourceStorageModeShared];

        MTLAccelerationStructureBoundingBoxGeometryDescriptor* bboxGeomDesc =
            [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
        bboxGeomDesc.boundingBoxBuffer = boundingBoxBuffer;
        bboxGeomDesc.boundingBoxStride = sizeof(MTLAxisAlignedBoundingBox);
        bboxGeomDesc.boundingBoxCount = (uint32_t)sphereBoundsList.size();
        // intersectionFunctionTableOffset here is this GEOMETRY's own
        // slot within whatever function table gets bound at trace time -
        // 0, sphereIntersectionFunction's own slot (set up below,
        // alongside the compute pipeline). The disk geometry added below
        // uses slot 1 instead - the first time this POC's function table
        // has needed more than one entry.
        bboxGeomDesc.intersectionFunctionTableOffset = 0;
        // Opaque here too: sphereIntersectionFunction is the REQUIRED
        // primitive-intersection test for this custom geometry (always
        // invoked, opaque or not - there's no hardware fallback for a
        // bounding-box primitive), so opaque just means "accept its
        // result directly," skip a second any-hit pass on top of it,
        // exactly this POC's one-test-decides-it shape.
        bboxGeomDesc.opaque = YES;

        // The disk's own bounding-box geometry, a SECOND geometryDescriptor
        // within the SAME primitive AS as the spheres (not a separate AS -
        // Metal supports multiple heterogeneous geometries in one
        // acceleration structure, distinguished at trace time by
        // `geometry_id`, matching this array's own index order: spheres
        // at 0, disk at 1 - see metal_poc.metal's own comment on why that
        // distinction is needed now). Padded uniformly by a small epsilon
        // in every axis (not just the disk's own zero-thickness normal
        // axis) - simplest bound that's correct regardless of which axis
        // the disk's normal happens to be aligned with, at the cost of a
        // slightly looser-than-optimal box for a single small primitive.
        const float diskBoundsEpsilon = 0.01f;
        std::vector<MTLAxisAlignedBoundingBox> diskBoundsList;
        for (const DiskData& d : disks) {
            float r = d.radius + diskBoundsEpsilon;
            MTLAxisAlignedBoundingBox bounds;
            bounds.min = MTLPackedFloat3Make(d.center.x - r, d.center.y - r, d.center.z - r);
            bounds.max = MTLPackedFloat3Make(d.center.x + r, d.center.y + r, d.center.z + r);
            diskBoundsList.push_back(bounds);
        }
        id<MTLBuffer> diskBoundingBoxBuffer = [device newBufferWithBytes:diskBoundsList.data()
            length:diskBoundsList.size() * sizeof(MTLAxisAlignedBoundingBox) options:MTLResourceStorageModeShared];

        MTLAccelerationStructureBoundingBoxGeometryDescriptor* diskGeomDesc =
            [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
        diskGeomDesc.boundingBoxBuffer = diskBoundingBoxBuffer;
        diskGeomDesc.boundingBoxStride = sizeof(MTLAxisAlignedBoundingBox);
        diskGeomDesc.boundingBoxCount = (uint32_t)diskBoundsList.size();
        diskGeomDesc.intersectionFunctionTableOffset = 1; // diskIntersectionFunction's own slot
        diskGeomDesc.opaque = YES;

        MTLPrimitiveAccelerationStructureDescriptor* sphereAccelDesc =
            [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        sphereAccelDesc.geometryDescriptors = @[bboxGeomDesc, diskGeomDesc];

        MTLAccelerationStructureSizes sphereSizes = [device accelerationStructureSizesWithDescriptor:sphereAccelDesc];
        id<MTLAccelerationStructure> sphereAS = [device newAccelerationStructureWithSize:sphereSizes.accelerationStructureSize];
        id<MTLBuffer> sphereScratch = [device newBufferWithLength:sphereSizes.buildScratchBufferSize
            options:MTLResourceStorageModePrivate];

        id<MTLCommandBuffer> sphereBuildCmd = [queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> sphereBuildEnc = [sphereBuildCmd accelerationStructureCommandEncoder];
        [sphereBuildEnc buildAccelerationStructure:sphereAS descriptor:sphereAccelDesc scratchBuffer:sphereScratch scratchBufferOffset:0];
        [sphereBuildEnc endEncoding];
        [sphereBuildCmd commit];
        [sphereBuildCmd waitUntilCompleted];
        if (sphereBuildCmd.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "Sphere AS build failed: %s\n", sphereBuildCmd.error.localizedDescription.UTF8String);
            return 1;
        }

        // --- Third primitive acceleration structure: Suzanne's own ------
        // geometry, built once, referenced by TWO different instances
        // below with two different transforms - unlike primAS/sphereAS
        // (each instanced exactly once, at identity), this is what
        // actually exercises instancing's whole point: reusing one GPU-
        // resident BVH from more than one world-space placement, rather
        // than building/storing the geometry twice.
        MTLAccelerationStructureTriangleGeometryDescriptor* suzanneGeomDesc =
            [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
        suzanneGeomDesc.vertexBuffer = suzanneVertexBuffer;
        suzanneGeomDesc.vertexStride = sizeof(PackedFloat3);
        suzanneGeomDesc.triangleCount = suzanneTriangleCount;
        suzanneGeomDesc.opaque = YES;

        MTLPrimitiveAccelerationStructureDescriptor* suzanneAccelDesc =
            [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        suzanneAccelDesc.geometryDescriptors = @[suzanneGeomDesc];

        MTLAccelerationStructureSizes suzanneSizes = [device accelerationStructureSizesWithDescriptor:suzanneAccelDesc];
        id<MTLAccelerationStructure> suzanneAS = [device newAccelerationStructureWithSize:suzanneSizes.accelerationStructureSize];
        id<MTLBuffer> suzanneScratch = [device newBufferWithLength:suzanneSizes.buildScratchBufferSize
            options:MTLResourceStorageModePrivate];

        id<MTLCommandBuffer> suzanneBuildCmd = [queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> suzanneBuildEnc = [suzanneBuildCmd accelerationStructureCommandEncoder];
        [suzanneBuildEnc buildAccelerationStructure:suzanneAS descriptor:suzanneAccelDesc scratchBuffer:suzanneScratch scratchBufferOffset:0];
        [suzanneBuildEnc endEncoding];
        [suzanneBuildCmd commit];
        [suzanneBuildCmd waitUntilCompleted];
        if (suzanneBuildCmd.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "Suzanne AS build failed: %s\n", suzanneBuildCmd.error.localizedDescription.UTF8String);
            return 1;
        }

        // --- Instance acceleration structure: four instances over three -
        // primitive ASes (primAS/sphereAS each instanced once at
        // identity, suzanneAS instanced TWICE with different transforms -
        // see that AS's own comment). `addInstance()` builds one
        // MTLAccelerationStructureInstanceDescriptor AND its matching
        // InstanceTransform side-channel entry from the SAME
        // column/translation values in one place, so the two can't drift
        // out of sync with each other the way two independently-hand-
        // authored copies of the same transform could.
        std::vector<MTLAccelerationStructureInstanceDescriptor> instanceDescs;
        std::vector<InstanceTransform> instanceTransforms;
        auto addInstance = [&](uint32_t accelStructureIndex, float3 col0, float3 col1, float3 col2, float3 col3) {
            MTLAccelerationStructureInstanceDescriptor desc{};
            desc.accelerationStructureIndex = accelStructureIndex;
            desc.options = MTLAccelerationStructureInstanceOptionNone;
            desc.mask = 0xFF;
            desc.intersectionFunctionTableOffset = 0;
            desc.transformationMatrix.columns[0] = MTLPackedFloat3Make(col0.x, col0.y, col0.z);
            desc.transformationMatrix.columns[1] = MTLPackedFloat3Make(col1.x, col1.y, col1.z);
            desc.transformationMatrix.columns[2] = MTLPackedFloat3Make(col2.x, col2.y, col2.z);
            desc.transformationMatrix.columns[3] = MTLPackedFloat3Make(col3.x, col3.y, col3.z);
            instanceDescs.push_back(desc);
            instanceTransforms.push_back(InstanceTransform{
                PackedFloat3{col0.x, col0.y, col0.z}, PackedFloat3{col1.x, col1.y, col1.z},
                PackedFloat3{col2.x, col2.y, col2.z}, PackedFloat3{col3.x, col3.y, col3.z}});
        };

        const float3 identityCol0{1, 0, 0}, identityCol1{0, 1, 0}, identityCol2{0, 0, 1}, identityCol3{0, 0, 0};
        addInstance(0, identityCol0, identityCol1, identityCol2, identityCol3); // primAS (room + Spot)
        addInstance(1, identityCol0, identityCol1, identityCol2, identityCol3); // sphereAS

        // Suzanne instance A: translation only, at the same world position
        // the single non-instanced Suzanne used to sit at - an identity-
        // rotation instance is the direct continuation of every earlier
        // screenshot's own Suzanne placement.
        addInstance(2, identityCol0, identityCol1, identityCol2, float3{-0.05f, -0.55f, -0.3f});

        // Suzanne instance B: rotated 45 degrees about Y and scaled down
        // (uniform scale only - transformNormalByInstance()'s own
        // rigid-transform assumption over in metal_poc.metal stays valid
        // under a uniform scale, since normalize() cancels a uniform
        // factor exactly; it would NOT under a non-uniform one), placed
        // high near the back of the ceiling. A first attempt at a back-
        // left-corner floor placement (x=-0.7, z=-0.55) turned out to sit
        // along almost the same camera sightline as the gold sphere
        // (x/z ratio ~0.19 vs. the gold sphere's own ~0.2) and was
        // nearly fully hidden behind it - the exact same 2D-screen-space-
        // occlusion lesson Spot's own placement (step 12) and the rough
        // dielectric sphere's own placement (step 13) already ran into,
        // caught here the same way: render, look, reposition. The one
        // instance in this whole scene whose object-space normals
        // actually need transforming before shading - everywhere else,
        // an identity transform makes that transform a no-op.
        {
            const float theta = 0.785398f; // 45 degrees, radians
            const float s = 0.55f;
            float3 rotCol0{s * cosf(theta), 0.0f, -s * sinf(theta)};
            float3 rotCol1{0.0f, s, 0.0f};
            float3 rotCol2{s * sinf(theta), 0.0f, s * cosf(theta)};
            addInstance(2, rotCol0, rotCol1, rotCol2, float3{0.0f, 0.75f, -0.3f});
        }

        id<MTLBuffer> instanceBuffer = [device newBufferWithBytes:instanceDescs.data()
            length:instanceDescs.size() * sizeof(MTLAccelerationStructureInstanceDescriptor)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> instanceTransformBuffer = [device newBufferWithBytes:instanceTransforms.data()
            length:instanceTransforms.size() * sizeof(InstanceTransform)
            options:MTLResourceStorageModeShared];

        MTLInstanceAccelerationStructureDescriptor* instAccelDesc =
            [MTLInstanceAccelerationStructureDescriptor descriptor];
        instAccelDesc.instancedAccelerationStructures = @[primAS, sphereAS, suzanneAS];
        instAccelDesc.instanceCount = (uint32_t)instanceDescs.size();
        instAccelDesc.instanceDescriptorBuffer = instanceBuffer;

        MTLAccelerationStructureSizes instSizes = [device accelerationStructureSizesWithDescriptor:instAccelDesc];
        id<MTLAccelerationStructure> instAS = [device newAccelerationStructureWithSize:instSizes.accelerationStructureSize];
        id<MTLBuffer> instScratch = [device newBufferWithLength:instSizes.buildScratchBufferSize
            options:MTLResourceStorageModePrivate];

        id<MTLCommandBuffer> buildCmd2 = [queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> buildEnc2 = [buildCmd2 accelerationStructureCommandEncoder];
        [buildEnc2 buildAccelerationStructure:instAS descriptor:instAccelDesc scratchBuffer:instScratch scratchBufferOffset:0];
        [buildEnc2 endEncoding];
        [buildCmd2 commit];
        [buildCmd2 waitUntilCompleted];
        if (buildCmd2.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "Instance AS build failed: %s\n", buildCmd2.error.localizedDescription.UTF8String);
            return 1;
        }

        // --- Compile the shader library from source at runtime ---------
        NSError* error = nil;
        // RT_METAL_SHADER_DIR is set by CMakeLists.txt's metal_poc target
        // (RT_BUILD_METAL=ON path) to gpu/metal/'s absolute source
        // directory. Falls back to a __FILE__-relative lookup for the
        // ad-hoc `clang++ metal_poc.mm ...` invocation this POC started
        // as (docs/METAL_GPU_FEASIBILITY.md section 7/8/9) and still
        // works fine for a quick manual rebuild without going through
        // CMake at all.
#ifdef RT_METAL_SHADER_DIR
        NSString* shaderDir = @(RT_METAL_SHADER_DIR);
#else
        NSString* shaderDir = [@(__FILE__) stringByDeletingLastPathComponent];
#endif
        NSString* shaderPath = [shaderDir stringByAppendingPathComponent:@"metal_poc.metal"];
        NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath encoding:NSUTF8StringEncoding error:&error];
        if (!shaderSource) {
            fprintf(stderr, "Failed to read shader source at %s: %s\n",
                shaderPath.UTF8String, error.localizedDescription.UTF8String);
            return 1;
        }
        MTLCompileOptions* compileOpts = [MTLCompileOptions new];
        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:compileOpts error:&error];
        if (!library) {
            fprintf(stderr, "Shader compile failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLFunction> kernelFn = [library newFunctionWithName:@"primaryRayKernel"];
        id<MTLFunction> sphereIntersectFn = [library newFunctionWithName:@"sphereIntersectionFunction"];
        id<MTLFunction> diskIntersectFn = [library newFunctionWithName:@"diskIntersectionFunction"];

        // The intersection function has to be LINKED into the compute
        // pipeline (MTLLinkedFunctions) before an MTLIntersectionFunction
        // Table naming it can be built - a plain newComputePipelineState
        // WithFunction: (used for step 1/2's triangle-only pipeline) has
        // nowhere to put that linkage, hence the switch to the descriptor-
        // based pipeline creation call here.
        MTLComputePipelineDescriptor* pipelineDesc = [MTLComputePipelineDescriptor new];
        pipelineDesc.computeFunction = kernelFn;
        MTLLinkedFunctions* linkedFns = [MTLLinkedFunctions new];
        linkedFns.functions = @[sphereIntersectFn, diskIntersectFn];
        pipelineDesc.linkedFunctions = linkedFns;

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:pipelineDesc
            options:MTLPipelineOptionNone reflection:nil error:&error];
        if (!pipeline) {
            fprintf(stderr, "Pipeline creation failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        // --- Intersection function table: two slots now, matching --------
        // bboxGeomDesc's own intersectionFunctionTableOffset (0) and
        // diskGeomDesc's (1) above - this POC's first real "one slot per
        // distinct intersection function" table, not just one slot
        // reused by every custom primitive. `setBuffer:atIndex:N` here
        // sets buffer N in the table's OWN shared argument namespace
        // (every function in ONE table draws from the same set of bound
        // buffers/textures) - sphereIntersectionFunction and
        // diskIntersectionFunction each declare a DIFFERENT `[[buffer(N)]]`
        // in their own MSL signature (0 and 1 respectively) specifically
        // so binding sphereBuffer at atIndex:0 and diskBuffer at
        // atIndex:1 here reaches the right function's own data, not a
        // shared/overwritten slot.
        MTLIntersectionFunctionTableDescriptor* fnTableDesc = [MTLIntersectionFunctionTableDescriptor new];
        fnTableDesc.functionCount = 2;
        id<MTLIntersectionFunctionTable> functionTable = [pipeline newIntersectionFunctionTableWithDescriptor:fnTableDesc];
        id<MTLFunctionHandle> sphereHandle = [pipeline functionHandleWithFunction:sphereIntersectFn];
        id<MTLFunctionHandle> diskHandle = [pipeline functionHandleWithFunction:diskIntersectFn];
        [functionTable setFunction:sphereHandle atIndex:0];
        [functionTable setFunction:diskHandle atIndex:1];
        // sphereIntersectionFunction/diskIntersectionFunction each read
        // their own geometry buffer (metal_poc.metal buffer(0)/buffer(1)
        // respectively - a SEPARATE argument table from the calling
        // kernel's own buffer(0..14), see that file's own comment) -
        // bound here, on the function table, not on the compute encoder.
        [functionTable setBuffer:sphereBuffer offset:0 atIndex:0];
        [functionTable setBuffer:diskBuffer offset:0 atIndex:1];

        // --- Output texture + uniforms ----------------------------------
        MTLTextureDescriptor* texDesc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
            width:width height:height mipmapped:NO];
        texDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
        texDesc.storageMode = MTLStorageModeShared;
        id<MTLTexture> outTexture = [device newTextureWithDescriptor:texDesc];

        // --- Earth texture (the back wall's materialType=3 source) -----
        // stb_image decodes straight to interleaved 8-bit RGBA regardless
        // of the source JPEG's channel count (the 4th `desiredChannels`
        // arg below), which is exactly MTLPixelFormatRGBA8Unorm_sRGB's own
        // BYTE layout - no repacking needed between stbi_load's buffer and
        // replaceRegion:. The `_sRGB` pixel format (not plain
        // `RGBA8Unorm`, this POC's own format up through PR #37) matters
        // for more than naming: an ordinary 8-bit JPEG/PNG's own stored
        // bytes are sRGB-gamma-ENCODED (perceptually, not linearly,
        // spaced) - every earlier render sampled those bytes directly as
        // if they were already linear radiance, silently darkening every
        // midtone the earth texture (and, via GI, everything it bounces
        // light onto) ever produced. `_sRGB` makes the texture SAMPLE
        // instruction itself convert sRGB to linear before the shader
        // ever sees a value - the standard, hardware-accelerated way to
        // do this, rather than a manual `pow(c, 2.2)` after sampling in
        // the shader.
#ifdef RT_MODELS_DIR
        NSString* imagesDir = [[@(RT_MODELS_DIR) stringByDeletingLastPathComponent]
            stringByAppendingPathComponent:@"images"];
#else
        NSString* imagesDir = [[@(__FILE__) stringByDeletingLastPathComponent]
            stringByAppendingPathComponent:@"../../images"];
#endif
        NSString* earthPath = [imagesDir stringByAppendingPathComponent:@"earthmap.jpg"];
        int earthW = 0, earthH = 0, earthChannels = 0;
        unsigned char* earthPixels = stbi_load(earthPath.UTF8String, &earthW, &earthH, &earthChannels, 4);
        id<MTLTexture> earthTexture = nil;
        if (earthPixels) {
            MTLTextureDescriptor* earthDesc = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB
                width:(NSUInteger)earthW height:(NSUInteger)earthH mipmapped:NO];
            earthDesc.usage = MTLTextureUsageShaderRead;
            earthDesc.storageMode = MTLStorageModeShared;
            earthTexture = [device newTextureWithDescriptor:earthDesc];
            MTLRegion earthRegion = MTLRegionMake2D(0, 0, (NSUInteger)earthW, (NSUInteger)earthH);
            [earthTexture replaceRegion:earthRegion mipmapLevel:0 withBytes:earthPixels
                bytesPerRow:(NSUInteger)earthW * 4];
            stbi_image_free(earthPixels);
            fprintf(stderr, "Loaded %s: %dx%d, %d channels\n", earthPath.UTF8String, earthW, earthH, earthChannels);
        } else {
            fprintf(stderr, "Could not load %s - back wall will read black/undefined texture data.\n",
                earthPath.UTF8String);
            // A 1x1 white fallback keeps the shader's unconditional
            // texture bind valid (Metal requires SOME texture at the
            // bound slot) even if the JPEG is missing. `_sRGB` for
            // consistency with the real texture above, though pure white
            // (255,255,255) round-trips through the sRGB<->linear
            // conversion unchanged either way.
            MTLTextureDescriptor* fallbackDesc = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB width:1 height:1 mipmapped:NO];
            fallbackDesc.usage = MTLTextureUsageShaderRead;
            fallbackDesc.storageMode = MTLStorageModeShared;
            earthTexture = [device newTextureWithDescriptor:fallbackDesc];
            uint8_t white4[4] = {255, 255, 255, 255};
            [earthTexture replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:white4 bytesPerRow:4];
        }

        const uint32_t samplesPerPixel = (argc > 4) ? (uint32_t)atoi(argv[4]) : 64;
        const uint32_t maxDepth = (argc > 5) ? (uint32_t)atoi(argv[5]) : 8;
        fprintf(stderr, "Samples/pixel: %u, max depth: %u\n", samplesPerPixel, maxDepth);

        Uniforms uniforms{};
        uniforms.cameraPos = PackedFloat3{0.0f, 0.0f, 3.2f};
        float3 forward = simd::normalize(float3{0, 0, -1});
        uniforms.cameraForward = PackedFloat3{forward.x, forward.y, forward.z};
        uniforms.cameraRight = PackedFloat3{1, 0, 0};
        uniforms.cameraUp = PackedFloat3{0, 1, 0};
        uniforms.tanHalfFov = tanf(0.5f * 40.0f * (float)M_PI / 180.0f);
        uniforms.aspect = (float)width / (float)height;
        uniforms.width = width;
        uniforms.height = height;
        uniforms.samplesPerPixel = samplesPerPixel;
        uniforms.maxDepth = maxDepth;
        uniforms.frameSeed = 1u;
        uniforms.lightCount = (uint32_t)lights.size();
        // Thin-lens depth of field: focused on the gold conductor sphere
        // (the nearest object to the camera), so it renders pixel-sharp
        // while the dielectric sphere just behind it and the back
        // wall/Suzanne further back show progressively more defocus blur -
        // the falloff is what actually demonstrates this is a real lens
        // model, not just a uniform blur filter over the whole frame.
        uniforms.lensRadius = 0.05f;
        uniforms.focusDistance = uniforms.cameraPos.z - spheres[1].center.z; // gold sphere's own z
        // A hexagonal (6-blade) aperture rather than a perfectly circular
        // one - see Uniforms' own apertureBlades comment. The classic
        // photographic blade count; out-of-focus highlights (area/point/
        // spot light reflections on the defocused back-wall geometry)
        // should now read as hexagons, not perfect circles.
        uniforms.apertureBlades = 6;
        // Shutter motion blur: a small horizontal dolly over the frame's
        // simulated exposure - chosen (over, say, an object moving) since
        // it needs no acceleration-structure/intersection-function
        // changes at all, purely a primary-ray-generation addition (see
        // metal_poc.metal's own comment on why: a moving CUSTOM primitive
        // would need per-sample time threaded into
        // sphereIntersectionFunction's own, separate argument table, real
        // additional Metal API surface this increment intentionally
        // doesn't take on).
        uniforms.cameraVelocity = PackedFloat3{0.015f, 0.0f, 0.0f};
        // Homogeneous fog filling the whole room - subtle (transmittance
        // ~0.7 over the ~4-unit camera-to-back-wall sightline: exp(-0.08*4)
        // ~ 0.73), meant to read as a light atmospheric haze visible in
        // the light shafts/depth falloff, not an opaque room-filling mist
        // that would fight every other material's own visibility.
        uniforms.fogSigmaT = 0.05f;
        uniforms.fogAlbedo = PackedFloat3{0.85f, 0.88f, 0.95f}; // mostly-scattering, faint cool tint
        // On: a miss ray samples earthTexture by direction (equirectangular)
        // instead of the flat two-colour gradient - the room's open front
        // means most miss rays are secondary/GI bounces (a mirror/glass
        // surface reflecting/refracting outward), not primary camera rays,
        // so this mostly shows up subtly rather than as an obvious visible
        // backdrop - see docs/METAL_GPU_FEASIBILITY.md's own note on
        // verifying this with a dedicated wide-FOV test render.
        uniforms.useEnvironmentMap = 1u;
        // Moderate forward scattering (real fog/haze skews strongly
        // forward in reality - Mie scattering off water droplets often
        // has g around 0.7-0.9 - 0.4 is deliberately more modest, so the
        // difference from isotropic reads as a stylistic tint on the fog
        // rather than a dramatic visible change).
        uniforms.fogAsymmetryG = 0.4f;
        uniforms.pointLightCount = (uint32_t)pointLights.size();
        uniforms.directionalLightCount = (uint32_t)directionalLights.size();
        // Single-pass adaptive sampling (see metal_poc.metal's own
        // shading-loop comment) - enabled by default for this scene:
        // converged pixels (most of the flat-coloured walls/ceiling)
        // stop well short of the full samplesPerPixel budget, spending
        // it instead on the noisier fog/specular/caustic regions this
        // scene already has plenty of - a real render-TIME win at
        // (ideally) no visible quality cost, verified via a dedicated
        // A/B render, not assumed.
        uniforms.adaptiveSampling = 1u;
        id<MTLBuffer> uniformBuffer = [device newBufferWithBytes:&uniforms length:sizeof(Uniforms) options:MTLResourceStorageModeShared];

        // --- Dispatch ----------------------------------------------------
        id<MTLCommandBuffer> renderCmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [renderCmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setTexture:outTexture atIndex:0];
        [enc setTexture:earthTexture atIndex:1];
        [enc setAccelerationStructure:instAS atBufferIndex:0];
        [enc setBuffer:uniformBuffer offset:0 atIndex:1];
        [enc setBuffer:materialBuffer offset:0 atIndex:2];
        [enc setBuffer:vertexBuffer offset:0 atIndex:3];
        [enc setBuffer:sphereMaterialBuffer offset:0 atIndex:4];
        [enc setBuffer:sphereBuffer offset:0 atIndex:5];
        [enc setIntersectionFunctionTable:functionTable atBufferIndex:6];
        [enc setBuffer:normalBuffer offset:0 atIndex:7];
        [enc setBuffer:uvBuffer offset:0 atIndex:8];
        [enc setBuffer:lightBuffer offset:0 atIndex:9];
        [enc setBuffer:suzanneNormalBuffer offset:0 atIndex:10];
        [enc setBuffer:suzanneMaterialBuffer offset:0 atIndex:11];
        [enc setBuffer:instanceTransformBuffer offset:0 atIndex:12];
        [enc setBuffer:diskBuffer offset:0 atIndex:13];
        [enc setBuffer:diskMaterialBuffer offset:0 atIndex:14];
        [enc setBuffer:pointLightBuffer offset:0 atIndex:15];
        [enc setBuffer:directionalLightBuffer offset:0 atIndex:16];
        // Mark the AS + its dependent primitive ASes as used so Metal
        // knows about the indirection - required for instance
        // acceleration structures referencing primitive ones (now three:
        // the room+Spot triangle mesh, the sphere's bounding-box
        // geometry, and Suzanne's own - referenced by TWO instances, but
        // only needs marking used once here, not once per instance).
        [enc useResource:primAS usage:MTLResourceUsageRead];
        [enc useResource:sphereAS usage:MTLResourceUsageRead];
        [enc useResource:suzanneAS usage:MTLResourceUsageRead];

        MTLSize gridSize = MTLSizeMake(width, height, 1);
        NSUInteger w = pipeline.threadExecutionWidth;
        NSUInteger h = pipeline.maxTotalThreadsPerThreadgroup / w;
        MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [enc endEncoding];
        [renderCmd commit];
        [renderCmd waitUntilCompleted];
        if (renderCmd.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "Render dispatch failed: %s\n", renderCmd.error.localizedDescription.UTF8String);
            return 1;
        }

        // --- Read back + write PNG (chromatic aberration, then lens ---
        // vignette, then ACES filmic tonemap, then the real sRGB OETF)
        std::vector<float> pixels(width * height * 4);
        MTLRegion region = MTLRegionMake2D(0, 0, width, height);
        [outTexture getBytes:pixels.data() bytesPerRow:width * 4 * sizeof(float) fromRegion:region mipmapLevel:0];

        const float vignetteStrength = 0.18f;
        const float chromaticAberrationStrength = 0.004f;
        std::vector<uint8_t> ldr(width * height * 3);
        for (uint32_t i = 0; i < width * height; ++i) {
            uint32_t px = i % width;
            uint32_t py = i / width;
            float vignette = vignetteFactor(px, py, width, height, vignetteStrength);
            float rgb[3];
            chromaticAberration(pixels, width, height, px, py, chromaticAberrationStrength,
                                 &rgb[0], &rgb[1], &rgb[2]);
            for (int c = 0; c < 3; ++c) {
                float v = fmaxf(rgb[c], 0.0f) * vignette;
                v = applyToneMap(v, toneMapMode);
                v = linearToSRGB(v);
                ldr[i * 3 + c] = (uint8_t)(v * 255.0f + 0.5f);
            }
        }
        // Bilateral denoise - see that function's own comment. Radius 3
        // (7x7), sigmaSpatial 2.5, sigmaRange 20.0 (in 0-255 luminance
        // units) - tuned the same way every other post-process knob this
        // POC has added was: by rendering and comparing, not from theory
        // alone. A much more aggressive setting (radius 4, sigmaRange 80)
        // was also tried and rejected - it visibly softened the crystal
        // ball's own sharp specular highlight and the checkerboard
        // floor's own tile edges, confirming this knob really can wash
        // out real detail if pushed too far, not just theoretically.
        std::vector<uint8_t> denoised(width * height * 3);
        bilateralDenoise(ldr, denoised, width, height, /*radius=*/3, /*sigmaSpatial=*/2.5f, /*sigmaRange=*/20.0f);

        stbi_write_png(outPath, width, height, 3, denoised.data(), width * 3);
        fprintf(stderr, "Wrote %s (%ux%u)\n", outPath, width, height);
    }
    return 0;
}
