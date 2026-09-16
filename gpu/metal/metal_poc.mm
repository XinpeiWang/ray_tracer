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
    PackedFloat3 cameraVelocity;
};

// Mirrors metal_poc.metal's AreaLight byte-for-byte.
struct AreaLightData {
    PackedFloat3 center;
    PackedFloat3 edgeU;
    PackedFloat3 edgeV;
    PackedFloat3 normal;
    float area;
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
    // Rough-dielectric-only roughness (materialType == 5) - see
    // metal_poc.metal's own comment on the mirrored field for why this
    // isn't just reusing `ior`'s slot the way materialType 4 does.
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
                     int32_t lightId = -1) {
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
    materials.push_back({packedColor, materialType, 1.0f, packedEmission, lightId});
    materials.push_back({packedColor, materialType, 1.0f, packedEmission, lightId});
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

int main(int argc, const char** argv) {
    @autoreleasepool {
        const uint32_t width = (argc > 1) ? (uint32_t)atoi(argv[1]) : 400;
        const uint32_t height = (argc > 2) ? (uint32_t)atoi(argv[2]) : 400;
        const char* outPath = (argc > 3) ? argv[3] : "/tmp/metal_poc_render.png";

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

        // Floor (y = -1)
        addQuad(verts, normals, uvs, materials, float3{-1,-1,-1}, float3{1,-1,-1}, float3{1,-1,1}, float3{-1,-1,1}, white);
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
        auto addAreaLight = [&](float3 a, float3 b, float3 c, float3 d, float3 emission) {
            int32_t lightId = (int32_t)lights.size();
            addQuad(verts, normals, uvs, materials, a, b, c, d, white,
                    /*materialType=*/0, emission, lightId);
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
                PackedFloat3{emission.x, emission.y, emission.z}});
        };
        addAreaLight(float3{-0.58f,0.98f,-0.25f}, float3{-0.22f,0.98f,-0.25f},
                     float3{-0.22f,0.98f,0.25f}, float3{-0.58f,0.98f,0.25f},
                     /*emission=*/float3{15.0f,10.0f,6.0f});   // warm
        addAreaLight(float3{0.22f,0.98f,-0.25f}, float3{0.58f,0.98f,-0.25f},
                     float3{0.58f,0.98f,0.25f}, float3{0.22f,0.98f,0.25f},
                     /*emission=*/float3{6.0f,10.0f,18.0f});   // cool

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
        std::vector<SphereData> spheres = {
            SphereData{PackedFloat3{0.35f, -0.65f, 0.15f}, 0.35f},
            SphereData{PackedFloat3{-0.55f, -0.65f, 0.45f}, 0.35f},
            SphereData{PackedFloat3{-0.05f, -0.82f, 0.6f}, 0.18f},
        };
        std::vector<TriangleMaterial> sphereMaterials = {
            TriangleMaterial{PackedFloat3{1.0f, 1.0f, 1.0f}, /*materialType=*/2, /*ior=*/1.5f, PackedFloat3{0, 0, 0}},
            TriangleMaterial{PackedFloat3{1.0f, 0.86f, 0.57f}, /*materialType=*/4, /*roughness=*/0.15f, PackedFloat3{0, 0, 0}},
            TriangleMaterial{PackedFloat3{1.0f, 1.0f, 1.0f}, /*materialType=*/5, /*ior=*/1.5f, PackedFloat3{0, 0, 0},
                             /*lightId=*/-1, /*roughness=*/0.35f},
        };

        const uint32_t triangleCount = (uint32_t)materials.size();
        fprintf(stderr, "Scene: %u triangles, %zu spheres, %zu lights\n", triangleCount, spheres.size(), lights.size());

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
        id<MTLBuffer> materialBuffer = [device newBufferWithBytes:materials.data()
            length:materials.size() * sizeof(TriangleMaterial)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> sphereBuffer = [device newBufferWithBytes:spheres.data()
            length:spheres.size() * sizeof(SphereData) options:MTLResourceStorageModeShared];
        id<MTLBuffer> sphereMaterialBuffer = [device newBufferWithBytes:sphereMaterials.data()
            length:sphereMaterials.size() * sizeof(TriangleMaterial) options:MTLResourceStorageModeShared];

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
        // 0 since sphereIntersectionFunction is the only entry in it
        // (set up below, alongside the compute pipeline).
        bboxGeomDesc.intersectionFunctionTableOffset = 0;
        // Opaque here too: sphereIntersectionFunction is the REQUIRED
        // primitive-intersection test for this custom geometry (always
        // invoked, opaque or not - there's no hardware fallback for a
        // bounding-box primitive), so opaque just means "accept its
        // result directly," skip a second any-hit pass on top of it,
        // exactly this POC's one-test-decides-it shape.
        bboxGeomDesc.opaque = YES;

        MTLPrimitiveAccelerationStructureDescriptor* sphereAccelDesc =
            [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        sphereAccelDesc.geometryDescriptors = @[bboxGeomDesc];

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

        // The intersection function has to be LINKED into the compute
        // pipeline (MTLLinkedFunctions) before an MTLIntersectionFunction
        // Table naming it can be built - a plain newComputePipelineState
        // WithFunction: (used for step 1/2's triangle-only pipeline) has
        // nowhere to put that linkage, hence the switch to the descriptor-
        // based pipeline creation call here.
        MTLComputePipelineDescriptor* pipelineDesc = [MTLComputePipelineDescriptor new];
        pipelineDesc.computeFunction = kernelFn;
        MTLLinkedFunctions* linkedFns = [MTLLinkedFunctions new];
        linkedFns.functions = @[sphereIntersectFn];
        pipelineDesc.linkedFunctions = linkedFns;

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:pipelineDesc
            options:MTLPipelineOptionNone reflection:nil error:&error];
        if (!pipeline) {
            fprintf(stderr, "Pipeline creation failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        // --- Intersection function table: one slot (index 0), matching --
        // every geometry's intersectionFunctionTableOffset above (bboxGeomDesc's
        // and instanceDescs[1]'s, both 0) - a real multi-custom-primitive
        // scene would have one slot per distinct intersection function,
        // indexed by whatever offsets those geometries/instances declare.
        MTLIntersectionFunctionTableDescriptor* fnTableDesc = [MTLIntersectionFunctionTableDescriptor new];
        fnTableDesc.functionCount = 1;
        id<MTLIntersectionFunctionTable> functionTable = [pipeline newIntersectionFunctionTableWithDescriptor:fnTableDesc];
        id<MTLFunctionHandle> sphereHandle = [pipeline functionHandleWithFunction:sphereIntersectFn];
        [functionTable setFunction:sphereHandle atIndex:0];
        // sphereIntersectionFunction reads its own `spheres` buffer
        // (metal_poc.metal buffer(0), a SEPARATE argument table from the
        // calling kernel's buffer(0..6) - see that file's own comment) -
        // bound here, on the function table, not on the compute encoder.
        [functionTable setBuffer:sphereBuffer offset:0 atIndex:0];

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
        // arg below), which is exactly MTLPixelFormatRGBA8Unorm's own
        // layout - no repacking needed between stbi_load's buffer and
        // replaceRegion:.
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
                texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
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
            // bound slot) even if the JPEG is missing.
            MTLTextureDescriptor* fallbackDesc = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:1 height:1 mipmapped:NO];
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

        // --- Read back + write PNG (8-bit sRGB-ish gamma, same simple ---
        // 1/2.2 approximation good enough for a POC comparison image)
        std::vector<float> pixels(width * height * 4);
        MTLRegion region = MTLRegionMake2D(0, 0, width, height);
        [outTexture getBytes:pixels.data() bytesPerRow:width * 4 * sizeof(float) fromRegion:region mipmapLevel:0];

        std::vector<uint8_t> ldr(width * height * 3);
        for (uint32_t i = 0; i < width * height; ++i) {
            for (int c = 0; c < 3; ++c) {
                float v = pixels[i * 4 + c];
                v = powf(fmaxf(v, 0.0f), 1.0f / 2.2f);
                v = fminf(v, 1.0f);
                ldr[i * 3 + c] = (uint8_t)(v * 255.0f + 0.5f);
            }
        }
        stbi_write_png(outPath, width, height, 3, ldr.data(), width * 3);
        fprintf(stderr, "Wrote %s (%ux%u)\n", outPath, width, height);
    }
    return 0;
}
