// metal_poc_pbrt_loader.mm
// Real pbrt scene loading - MetalPocApp::loadPbrtScene() and its 9 phase
// methods (area lights, remaining triangles, spheres, disks, cylinders,
// object instances, punctual lights, medium, infinite light, camera) -
// split out of metal_poc.mm once that one file's own "Stage 2.5" block
// grew past a third of it (a pure code-motion refactor, no behaviour
// change - see metal_poc_scenes_a.mm's own header comment for the
// identical precedent this follows, and docs/history/METAL_GPU_FEASIBILITY.md).
// buildScene() (the hardcoded room), buildHandAuthoredScene() (the
// scene_id dispatcher), and everything GPU-resource/dispatch-related
// stay in metal_poc.mm itself - this file is only "read a .pbrt file
// into the app's own scene data," nothing else.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_poc_app.h"
#include "../../src/shared/curve_tessellate.h"
#include "../../src/shared/srgb_decode.h"
#include "../../src/shared/measured_bxdf_loader.h"   // MeasuredBRDFData + GetMeasuredBRDFDataCached (the CPU renderer's own cache)
#include "../../src/shared/portal_image_infinite_light.h"   // PortalImageInfiniteLightData: rectified image + sampling tables for portal[4]
#include <cstring>
#include <array>
#include <map>
#include <functional>

// --- Stage 2.5: real pbrt scene loading (v1 - see docs/METAL_GPU_
// FEASIBILITY.md's own section on this) ----------------------------------
// Loads pbrtScenePath via src/shared/pbrt_load.h - the SAME front-end
// parser cpu_renderer and gpu/optix both already use (pbrt_cpu_builder.h/
// pbrt_gpu_builder.h) - and appends its geometry/materials/lights into
// this POC's own scene vectors, proving the pipeline shape (the FIRST
// concrete step toward real integration, not the whole thing - see this
// project's own status notes on what's still deliberately NOT done).
//
// Deliberately narrow v1 scope, each gap warned rather than silently
// wrong or a crash (matching gpu/optix/scene_builder.cpp's own graceful-
// failure precedent for GPU-unsupported scenes):
//   - Materials: Diffuse/Conductor/Dielectric only (this POC's own
//     materialType 0/4/2) - anything else falls back to gray Lambertian.
//   - Area lights: only a light attached to EXACTLY 2 triangles forming a
//     planar quad in addQuad()'s own a-b-c-d/a-c-d fan convention, or a
//     single full-circle disk, has a real NEE strategy - AreaLightData
//     (metal_poc.metal) is a parallelogram sampler (center/edgeU/edgeV),
//     not a general triangle-mesh/disk one. Any other emissive shape
//     (a non-quad triangle mesh, an annular/partial disk) is still
//     visible (direct hits, BSDF-sampled bounces) but has no explicit
//     NEE strategy sampling it - see section 100/101 of the docs.
//   - Shapes: triangle meshes, spheres, and full-circle disks (no inner
//     radius/phi-max, uniform scale only) - cylinder/cone/paraboloid/
//     bilinearmesh/curve shapes are still skipped entirely.
//   (ObjectInstance/instancing, infinite lights, and every punctual light
//   kind ARE now supported - this comment block predates those; see the
//   docs' own numbered sections for what shipped after this was written.)
// None of this needed any changes to buildGPUResources() below - see
// buildScene()'s own call-site comment for why (additive onto the
// existing hardcoded room, never leaves any vector newly empty).
// --- Shapes the Metal POC has no intersection primitive for, tessellated into triangles -----
// bilinearmesh, cone, paraboloid and curve were skipped outright ("shapes skipped" warning), so
// scenes using them (F1, F4, F7, F8, F14) rendered without that geometry. They are instead
// converted, once at load time and BEFORE the scene bounding box / materials are processed, into
// ordinary triangles appended to the flattened scene - so scale, materials, area lights and
// instancing all treat them like any other mesh. Same approach pbrt-v4's own GPU path and this
// project's OptiX builder take for curves (src/shared/curve_tessellate.h). The tessellation is a
// close approximation, not exact geometry: smooth analytic vertex normals for the quadrics and
// patches, flat-shaded 8-sided tubes for curves.
namespace {

void tessellateGrid(std::vector<pbrt_flatten::Triangle>& out, int nu, int nv, int material, int areaLight,
                    const std::function<void(double, double, double*, double*)>& eval) {
    for (int j = 0; j < nv; ++j) {
        for (int i = 0; i < nu; ++i) {
            const double u0 = (double)i / nu, u1 = (double)(i + 1) / nu;
            const double v0 = (double)j / nv, v1 = (double)(j + 1) / nv;
            double p[4][3], n[4][3];
            const double us[4] = {u0, u1, u0, u1}, vs[4] = {v0, v0, v1, v1};   // p00 p10 p01 p11
            for (int k = 0; k < 4; ++k) eval(us[k], vs[k], p[k], n[k]);
            const int tris[2][3] = {{0, 1, 3}, {0, 3, 2}};
            for (const auto& t : tris) {
                pbrt_flatten::Triangle tri;
                for (int c = 0; c < 3; ++c) {
                    for (int a = 0; a < 3; ++a) { tri.v[c * 3 + a] = p[t[c]][a]; tri.n[c * 3 + a] = n[t[c]][a]; }
                    tri.uv[c * 2 + 0] = us[t[c]]; tri.uv[c * 2 + 1] = vs[t[c]];
                }
                tri.hasNormals = true;
                tri.hasUVs = true;
                tri.material = material;
                tri.areaLight = areaLight;
                out.push_back(tri);
            }
        }
    }
}

void normalize3(double* v) {
    const double l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (l > 1e-300) { v[0] /= l; v[1] /= l; v[2] /= l; }
}

// Returns how many triangles were added (0 when the scene has none of these shapes).
size_t tessellateUnsupportedShapes(pbrt_flatten::FlatScene& scene,
                                std::unordered_map<int, PackedFloat3>& fiberTangent) {
    std::vector<pbrt_flatten::Triangle> out;
    std::vector<std::pair<size_t, PackedFloat3>> outTangents;   // (index into `out`, fibre direction)
    // A shape with `Material "interface"` only bounds a participating medium: skip it (transparent)
    // rather than turning it into an opaque gray mesh - same rule the sphere/disk/cylinder loaders use.
    auto isInterface = [&scene](int m) {
        return m >= 0 && m < (int)scene.materials.size() &&
               scene.materials[m].kind == pbrt_flatten::MaterialKind::Interface;
    };

    // Bilinear patch: p(u,v) = (1-u)(1-v)p00 + u(1-v)p10 + (1-u)v p01 + uv p11; normal = dpdu x dpdv.
    for (const pbrt_flatten::BilinearPatch& bp : scene.bilinearPatches) {
        if (isInterface(bp.material)) continue;
        const double (*P)[3] = bp.p;
        tessellateGrid(out, 16, 16, bp.material, bp.areaLight, [&](double u, double v, double* p, double* n) {
            double dpdu[3], dpdv[3];
            for (int a = 0; a < 3; ++a) {
                p[a] = (1 - u) * (1 - v) * P[0][a] + u * (1 - v) * P[1][a] + (1 - u) * v * P[2][a] + u * v * P[3][a];
                dpdu[a] = (1 - v) * (P[1][a] - P[0][a]) + v * (P[3][a] - P[2][a]);
                dpdv[a] = (1 - u) * (P[2][a] - P[0][a]) + u * (P[3][a] - P[1][a]);
            }
            n[0] = dpdu[1] * dpdv[2] - dpdu[2] * dpdv[1];
            n[1] = dpdu[2] * dpdv[0] - dpdu[0] * dpdv[2];
            n[2] = dpdu[0] * dpdv[1] - dpdu[1] * dpdv[0];
            normalize3(n);
        });
    }

    // Cone (open, no base cap): object space z in [0,height], radius R*(1 - z/height).
    for (const pbrt_flatten::Cone& c : scene.cones) {
        if (isInterface(c.material)) continue;
        pbrt_scene::Matrix4 xf;
        for (int i = 0; i < 16; ++i) xf.m[i] = c.xform[i];
        const double phiMax = c.phiMaxDeg * M_PI / 180.0;
        tessellateGrid(out, 48, 4, c.material, c.areaLight, [&](double u, double v, double* p, double* n) {
            const double ph = u * phiMax, rad = c.radius * (1.0 - v);
            const double po[3] = {rad * std::cos(ph), rad * std::sin(ph), v * c.height};
            pbrt_flatten::flatten_detail::transformPoint(xf, po[0], po[1], po[2], p);
            pbrt_flatten::flatten_detail::transformNormal(xf, c.height * std::cos(ph), c.height * std::sin(ph), c.radius, n);
            normalize3(n);
        });
    }

    // Paraboloid: z in [zMin,zMax], radius R*sqrt(z/zMax); outward normal is the gradient of
    // x^2 + y^2 - (R^2/zMax) z.
    for (const pbrt_flatten::Paraboloid& pa : scene.paraboloids) {
        if (isInterface(pa.material)) continue;
        pbrt_scene::Matrix4 xf;
        for (int i = 0; i < 16; ++i) xf.m[i] = pa.xform[i];
        const double phiMax = pa.phiMaxDeg * M_PI / 180.0;
        tessellateGrid(out, 48, 24, pa.material, pa.areaLight, [&](double u, double v, double* p, double* n) {
            const double ph = u * phiMax;
            const double z = pa.zMin + v * (pa.zMax - pa.zMin);
            const double rad = pa.radius * std::sqrt(std::max(z, 0.0) / pa.zMax);
            const double x = rad * std::cos(ph), y = rad * std::sin(ph);
            pbrt_flatten::flatten_detail::transformPoint(xf, x, y, z, p);
            pbrt_flatten::flatten_detail::transformNormal(xf, 2.0 * x, 2.0 * y, -pa.radius * pa.radius / pa.zMax, n);
            normalize3(n);
        });
    }

    // Curve: every Bezier segment becomes a tapered 8-sided tube (the same dicing density OptiX
    // uses); width interpolates across the WHOLE curve, as in pbrt-v4. Flat-shaded.
    for (const pbrt_flatten::Curve& cv : scene.curves) {
        if (isInterface(cv.material)) continue;
        std::vector<curve_tessellate::Quad> quads;
        for (int seg = 0; seg < cv.nSegments; ++seg) {
            float cp[4][3];
            for (int i = 0; i < 4; ++i)
                for (int a = 0; a < 3; ++a) cp[i][a] = (float)cv.cp[((size_t)seg * 4 + i) * 3 + a];
            const double t0 = (double)seg / cv.nSegments, t1 = (double)(seg + 1) / cv.nSegments;
            quads.clear();
            curve_tessellate::tessellate(cp, 0.0f, 1.0f, (float)(cv.width0 + (cv.width1 - cv.width0) * t0),
                                         (float)(cv.width0 + (cv.width1 - cv.width0) * t1), 10, 8, quads);
            for (const curve_tessellate::Quad& q : quads) {
                const float* c4[4] = {q.p00, q.p10, q.p01, q.p11};
                const int tris[2][3] = {{0, 1, 3}, {0, 3, 2}};
                for (const auto& t : tris) {
                    pbrt_flatten::Triangle tri;
                    for (int c = 0; c < 3; ++c)
                        for (int a = 0; a < 3; ++a) tri.v[c * 3 + a] = c4[t[c]][a];
                    tri.material = cv.material;
                    tri.areaLight = cv.areaLight;
                    // Fibre direction = along the tube (ring i -> ring i+1), for the hair shader.
                    {
                        float tx = q.p10[0] - q.p00[0], ty = q.p10[1] - q.p00[1], tz = q.p10[2] - q.p00[2];
                        const float tl = std::sqrt(tx * tx + ty * ty + tz * tz);
                        if (tl > 1e-20f) outTangents.push_back({out.size(), PackedFloat3{tx / tl, ty / tl, tz / tl}});
                    }
                    out.push_back(tri);
                }
            }
        }
    }

    const size_t baseIndex = scene.triangles.size();
    for (const auto& tp : outTangents) fiberTangent[(int)(baseIndex + tp.first)] = tp.second;
    scene.triangles.insert(scene.triangles.end(), out.begin(), out.end());
    scene.bilinearPatches.clear();
    scene.cones.clear();
    scene.paraboloids.clear();
    scene.curves.clear();
    return out.size();
}

}  // namespace

void MetalPocApp::loadPbrtScene() {
    pbrt_load::LoadResult result = pbrt_load::loadFile(pbrtScenePath);
    if (!result.ok) {
        fprintf(stderr, "loadPbrtScene: %s\n", result.error.c_str());
        if (!result.missingFiles.empty()) pbrtMissingAssetsError = result.error;
        return;
    }
    if (pbrtRequireAllMeshes && !result.scene.missingFiles.empty()) {
        pbrtMissingAssetsError = pbrt_load::detail::missingMeshesMessage(pbrtScenePath, result.scene.missingFiles, false);
        fprintf(stderr, "loadPbrtScene: %s\n", pbrtMissingAssetsError.c_str());
        return;
    }
    pbrtTriangleFiberTangent.clear();
    if (const size_t added = tessellateUnsupportedShapes(result.scene, pbrtTriangleFiberTangent))
        fprintf(stderr, "loadPbrtScene: tessellated bilinear patch/cone/paraboloid/curve shapes into %zu triangle(s)\n", added);
    const pbrt_flatten::FlatScene& scene = result.scene;
    for (const pbrt_scene::Warning& w : scene.warnings) {
        fprintf(stderr, "loadPbrtScene: pbrt loader warning: %s\n", w.message.c_str());
    }

    // Film's own PixelFilter - see MetalPocApp::pbrtFilterKind's own
    // comment. `scene.filter` is already per-kind-default-resolved by
    // flatten() (PixelFilter::radius's own comment, src/shared/
    // pbrt_flatten.h) - a scene with no explicit PixelFilter directive
    // reads back exactly pbrt-v4's own gaussian/1.5 default, same as
    // this struct's own in-class defaults, so this assignment is a true
    // no-op for the overwhelming majority of scenes that never set one.
    pbrtFilterKind = scene.filter.kind;
    pbrtFilterB = (float)scene.filter.B;
    pbrtFilterC = (float)scene.filter.C;
    pbrtFilterSigma = (float)scene.filter.sigma;
    pbrtFilterTau = (float)scene.filter.tau;
    pbrtFilterRadius = (float)scene.filter.radius;

    // Uniform scene-scale normalization: a real pbrt scene is typically
    // authored at a scale of hundreds of units (a classic Cornell box
    // spans ~555) - NOT this shader's own [-1,1]-ish hardcoded-room
    // scale. Every shadow-ray/reflection-ray self-intersection offset in
    // metal_poc.metal (`hitPoint + facingNormal * 0.001f`, ~80 call
    // sites) was tuned for that small scale; at ~500 units, 0.001 is a
    // numerically negligible fraction of the scene (0.0002%) - too small
    // to reliably escape the source triangle's own surface, so every
    // shadow ray immediately (re-)self-intersects and every light sample
    // reads as occluded. Confirmed directly: without this, the loaded
    // Cornell box rendered almost entirely black (only the light's own
    // direct camera-hit visible; every diffuse wall got zero NEE light).
    // Rescaling metal_poc.metal's own ~80 epsilons to be scene-relative
    // instead would be a much larger, riskier change than rescaling the
    // INPUT geometry once, here, at load time - computed from the
    // scene's own bounding box (triangles + spheres) so this generalizes
    // to any pbrt scene's own authored scale, not just this one file's.
    float3 bboxMin{FLT_MAX, FLT_MAX, FLT_MAX}, bboxMax{-FLT_MAX, -FLT_MAX, -FLT_MAX};
    auto growBounds = [&](float3 p) {
        bboxMin = simd::min(bboxMin, p);
        bboxMax = simd::max(bboxMax, p);
    };
    for (const pbrt_flatten::Triangle& t : scene.triangles) {
        for (int c = 0; c < 3; ++c)
            growBounds(float3{(float)t.v[c * 3 + 0], (float)t.v[c * 3 + 1], (float)t.v[c * 3 + 2]});
    }
    for (const pbrt_flatten::Sphere& s : scene.spheres) {
        const float3 c{(float)s.center[0], (float)s.center[1], (float)s.center[2]};
        const float r = (float)s.radius;
        growBounds(c - float3{r, r, r});
        growBounds(c + float3{r, r, r});
    }
    // A huge ground-plane-as-a-sphere (the common `Translate 0 -1000 0` + radius-1000
    // idiom, e.g. A2/A3/A5/A7/F2) would make the extent ~2000 and the rescale ~0.001,
    // shrinking every real object to ~0.001 units - the SAME size as the shaders'
    // fixed 0.001 ray offsets - so reflection/refraction rays off a radius-1 sphere
    // were displaced by a full radius (mirror sphere's lower half rendered black).
    // So: if excluding spheres whose diameter is >= 60% of the full extent shrinks the
    // extent by more than 4x, size the scene from that "content" box instead. The
    // huge sphere is still loaded, just no longer allowed to set the scale.
    {
        const float3 fullExtent = bboxMax - bboxMin;
        const float fullMax = fmaxf(fullExtent.x, fmaxf(fullExtent.y, fullExtent.z));
        float3 coreMin{FLT_MAX, FLT_MAX, FLT_MAX}, coreMax{-FLT_MAX, -FLT_MAX, -FLT_MAX};
        bool anyCore = false, droppedGiant = false;
        for (const pbrt_flatten::Triangle& t : scene.triangles) {
            for (int c = 0; c < 3; ++c) {
                const float3 p{(float)t.v[c * 3 + 0], (float)t.v[c * 3 + 1], (float)t.v[c * 3 + 2]};
                coreMin = simd::min(coreMin, p); coreMax = simd::max(coreMax, p); anyCore = true;
            }
        }
        for (const pbrt_flatten::Sphere& s : scene.spheres) {
            const float r = (float)s.radius;
            if (2.0f * r >= 0.6f * fullMax) { droppedGiant = true; continue; }
            const float3 c{(float)s.center[0], (float)s.center[1], (float)s.center[2]};
            coreMin = simd::min(coreMin, c - float3{r, r, r}); coreMax = simd::max(coreMax, c + float3{r, r, r}); anyCore = true;
        }
        if (droppedGiant && anyCore) {
            const float3 coreExtent = coreMax - coreMin;
            const float coreMaxExtent = fmaxf(coreExtent.x, fmaxf(coreExtent.y, coreExtent.z));
            if (coreMaxExtent > 1e-6f && coreMaxExtent < 0.25f * fullMax) {
                fprintf(stderr, "loadPbrtScene: ignoring a huge ground-like sphere when sizing the scene "
                                "(extent %.1f -> %.1f)\n", fullMax, coreMaxExtent);
                bboxMin = coreMin; bboxMax = coreMax;
            }
        }
    }
    const float3 bboxExtent = bboxMax - bboxMin;
    const float maxExtent = fmaxf(bboxExtent.x, fmaxf(bboxExtent.y, bboxExtent.z));
    // Target: the loaded scene's own largest dimension maps to 2.0 units -
    // matching the hardcoded room's own [-1,1] (2-unit-across) scale, so
    // every one of those existing epsilons is meaningful again. Falls
    // back to 1.0 (no rescale) for a degenerate/empty scene rather than
    // dividing by ~0.
    const float sceneScale = (maxExtent > 1e-6f) ? (2.0f / maxExtent) : 1.0f;
    // Recentre on the scene's own bounding-box centre, THEN push it well
    // clear of the hardcoded room's own occupied [-1,1] region (+60 in X
    // as of section 169 - was +8, "more than enough given the loaded
    // scene's own rescaled extent is ~2 units" turned out to be true only
    // for camera-frustum overlap, not for an OPEN scene's own shadow rays
    // reaching the room's own always-present, zero-falloff directional
    // light unoccluded, or a specular/mirror material reflecting the
    // room's own geometry from 8 units away - both real, found via G25/
    // G19 respectively, not assumed) - the pbrt scene's own coordinate
    // origin has no
    // reason to relate to the hardcoded room's at all (e.g. this classic
    // Cornell box is authored spanning x/y/z ~[0,555], not centred at its
    // own origin), so simply rescaling in place (this function's own
    // first attempt) left the two scenes - and the camera, repositioned
    // to the loaded scene's own - confusingly overlapping in the SAME
    // small region of world space, with the render showing a hard-to-
    // interpret mix of both. Recentre + offset keeps this purely
    // ADDITIVE (see buildScene()'s own call-site comment on why - no
    // buildGPUResources() changes needed) while keeping the two scenes
    // visually and spatially separate, exactly as if they were two
    // different rooms.
    const float3 bboxCenter = 0.5f * (bboxMin + bboxMax);
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};
    auto toWorld = [=](float3 p) { return (p - bboxCenter) * sceneScale + sceneOffset; };
    fprintf(stderr, "loadPbrtScene: scene bounding box extent %.1f units, rescaling by %.5f, "
                    "recentred and offset to +X\n", maxExtent, sceneScale);

    PbrtMaterialMapState materialState{scene, sceneScale, bboxCenter, sceneOffset, {}, {}};   // see metal_poc_pbrt_materials.mm
    auto materialFor = [&](int idx) -> TriangleMaterial {
        if (idx < 0 || idx >= (int)scene.materials.size())
            return TriangleMaterial{PackedFloat3{0.5f, 0.5f, 0.5f}, 0u, 1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f};
        return mapPbrtMaterial(materialState, scene.materials[idx], /*depth=*/0);
    };
    auto vertexAt = [toWorld](const double* v, int i) {
        return toWorld(float3{(float)v[i * 3 + 0], (float)v[i * 3 + 1], (float)v[i * 3 + 2]});
    };

    std::vector<bool> triangleHandled(scene.triangles.size(), false);
    // Populated by loadPbrtAreaLights() below only for a light shape too
    // complex to represent as this loader's single analytic AreaLightData
    // quad (see that method's own comment) - read by
    // loadPbrtRemainingTriangles() right after to mark those triangles
    // emissive (but NOT NEE-light-registered) instead of silently
    // dropping their emission.
    std::unordered_map<int, std::pair<float3, bool>> unhandledLightEmission;
    loadPbrtAreaLights(scene, toWorld, materialFor, triangleHandled, unhandledLightEmission);
    loadPbrtRemainingTriangles(scene, toWorld, materialFor, triangleHandled, unhandledLightEmission);
    loadPbrtSpheres(scene, toWorld, materialFor, sceneScale);
    loadPbrtDisks(scene, toWorld, materialFor, sceneScale);
    loadPbrtCylinders(scene, toWorld, materialFor, sceneScale);
    loadPbrtObjectInstances(scene, toWorld, materialFor);
    loadPbrtPunctualLights(scene, toWorld, sceneScale);
    loadPbrtMedium(scene, sceneScale);
    pbrtMaxComponentValue = (float)scene.maxComponentValue;
    loadPbrtInfiniteLight(scene, toWorld);
    loadPbrtCamera(scene, toWorld, bboxCenter, sceneScale, sceneOffset);

    fprintf(stderr, "loadPbrtScene: loaded %s (%zu triangles, %zu spheres, %zu area lights)\n",
            pbrtScenePath.c_str(), scene.triangles.size(), scene.spheres.size(), scene.areaLights.size());
}

// --- Area lights: only the "single quad, 2 triangles" shape - see
// this function's own header comment.
// Flat emission (radiance) for an area light. A plain light is L*scale. An image-textured one
// (AreaLightSource "diffuse" "string filename") can only be mapped per-pixel on a QUAD here
// (the shared texture slot); for any other shape the pattern is dropped, but using the
// image's AVERAGE colour * scale instead of a flat L (default 1, i.e. white) keeps the
// light's total energy right - white*scale made such lights ~2x too bright (C11).
// `radialRowWeight` is for a DISK: pbrt maps image row -> radius (row/H = r/R), so the area-uniform
// average weights each row by its radius (~ row+0.5), not equally.
static PackedFloat3 pbrtFlatLightEmission(const std::string& pbrtScenePath, const pbrt_flatten::Emission& em,
                                          bool radialRowWeight = false) {
    if (!em.filename.empty()) {
        std::string bytes;
        std::vector<float> px;
        int w = 0, h = 0;
        if (pbrt_load::loadFileNear(pbrtScenePath, em.filename, bytes) &&
            pbrt_load::detail::decodeInfiniteLightImage(em.filename, bytes, px, w, h) && w > 0 && h > 0 &&
            px.size() >= (size_t)w * h * 3) {
            double sum[3] = {0, 0, 0}, wsum = 0.0;
            for (int y = 0; y < h; ++y) {
                const double wgt = radialRowWeight ? (y + 0.5) : 1.0;
                for (int x = 0; x < w; ++x) {
                    const size_t i = (size_t)y * w + x;
                    sum[0] += wgt * px[i * 3 + 0]; sum[1] += wgt * px[i * 3 + 1]; sum[2] += wgt * px[i * 3 + 2];
                    wsum += wgt;
                }
            }
            return PackedFloat3{(float)(sum[0] / wsum * em.scale), (float)(sum[1] / wsum * em.scale), (float)(sum[2] / wsum * em.scale)};
        }
    }
    return PackedFloat3{(float)(em.L[0] * em.scale), (float)(em.L[1] * em.scale), (float)(em.L[2] * em.scale)};
}

// The shared area-light texture slot holds ONE image. A light that names an image gets it if the slot is free (the image is decoded into it)
// or already holds that same image, so a scene may light several shapes from one texture; a light naming a different image, or one that
// cannot be decoded, falls back to its flat colour.
bool MetalPocApp::claimAreaLightImage(const pbrt_flatten::Emission& em) {
    if (em.filename.empty()) return false;
    if (havePbrtAreaLightImage) return em.filename == pbrtAreaLightImageFilename;
    std::string bytes;
    if (pbrt_load::loadFileNear(pbrtScenePath, em.filename, bytes) &&
        pbrt_load::detail::decodeInfiniteLightImage(em.filename, bytes,
            pbrtAreaLightImagePixels, pbrtAreaLightImageWidth, pbrtAreaLightImageHeight)) {
        havePbrtAreaLightImage = true;
        pbrtAreaLightImageFilename = em.filename;
        return true;
    }
    fprintf(stderr, "loadPbrtScene: area light's own image '%s' could not be read/decoded; falling back to its flat colour\n", em.filename.c_str());
    return false;
}

void MetalPocApp::loadPbrtAreaLights(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld,
    const PbrtMaterialForFn& materialFor, std::vector<bool>& triangleHandled,
    std::unordered_map<int, std::pair<float3, bool>>& unhandledLightEmission) {
    auto vertexAt = [toWorld](const double* v, int i) {
        return toWorld(float3{(float)v[i * 3 + 0], (float)v[i * 3 + 1], (float)v[i * 3 + 2]});
    };
    std::unordered_map<int, std::vector<int>> trianglesByLight;
    for (int i = 0; i < (int)scene.triangles.size(); ++i) {
        const int al = scene.triangles[i].areaLight;
        if (al >= 0) trianglesByLight[al].push_back(i);
    }
    for (const auto& entry : trianglesByLight) {
        const int lightIdx = entry.first;
        const std::vector<int>& idxs = entry.second;
        const pbrt_flatten::Emission& em = scene.areaLights[lightIdx];
        const PackedFloat3 flatEm = pbrtFlatLightEmission(pbrtScenePath, em);
        float3 emission{flatEm.x, flatEm.y, flatEm.z};
        uint32_t quadMaterialType = 0u;
        float useTextureFlag = 0.0f;
        bool handled = false;
        if (idxs.size() == 2) {
            const pbrt_flatten::Triangle& t0 = scene.triangles[idxs[0]];
            const pbrt_flatten::Triangle& t1 = scene.triangles[idxs[1]];
            const float3 a = vertexAt(t0.v, 0), b = vertexAt(t0.v, 1), c = vertexAt(t0.v, 2);
            const float3 t1a = vertexAt(t1.v, 0), t1b = vertexAt(t1.v, 1), d = vertexAt(t1.v, 2);
            const float eps = 1e-4f;
            if (simd::length(a - t1a) < eps && simd::length(c - t1b) < eps) {
                // Image-based emission (section 105) - only for a
                // confirmed QUAD-shaped light (this branch - the vertex-
                // matching check just above already ruled out a 2-
                // triangle shape that ISN'T actually a quad, e.g. a
                // differently-diagonalized or non-planar pair; doing
                // this decode attempt BEFORE that check, as an earlier
                // version of this code did, would waste the one shared
                // texture slot - and leave `emission` wrongly set to a
                // bare `scale` instead of `L*scale` - on a light that
                // never actually becomes materialType 15 at all), and
                // only the FIRST such light in the scene (one shared
                // texture slot, matching the goniometric/projection
                // image precedent). A decode failure - or a SECOND
                // textured light - falls back to flat L exactly as if no
                // filename were named, same "safe fallback over dropping
                // the light" reasoning as every other image-based
                // feature in this loader.
                if (claimAreaLightImage(em)) {
                    quadMaterialType = 15u;
                    useTextureFlag = 1.0f;
                    emission = float3{(float)em.scale, (float)em.scale, (float)em.scale};
                }
                const TriangleMaterial lightMat = materialFor(t0.material);
                const int32_t lightId = (int32_t)lights.size();
                addQuad(verts, normals, uvs, materials, a, b, c, d,
                        float3{lightMat.color.x, lightMat.color.y, lightMat.color.z},
                        quadMaterialType, emission, lightId, /*roughness=*/0.0f,
                        /*ior=*/1.0f, /*transmitColor=*/simd::make_float3(0, 0, 0),
                        /*twoSided=*/em.twoSided);
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
                    PackedFloat3{emission.x, emission.y, emission.z},
                    /*patternTileB=*/0.0f,
                    /*patternScale=*/0.0f,
                    /*twoSided=*/em.twoSided ? 1.0f : 0.0f,
                    /*useTexture=*/useTextureFlag});
                handled = true;
            }
        }
        if (handled) {
            triangleHandled[idxs[0]] = true;
            triangleHandled[idxs[1]] = true;
        } else {
            // Not a simple 2-triangle quad (a single triangle, an N>2
            // triangle mesh, ...) - this loader has no general per-
            // triangle NEE-sampled light representation (AreaLightData
            // is a single analytic QUAD, sampled as one unit; every
            // triangle of a real mesh light would need its own entry and
            // its own share of the light-picking pmf, real work this
            // loader doesn't do yet). Rather than silently dropping the
            // light's own emission entirely (this function's own PREVIOUS
            // behaviour - a real correctness bug, not just a missing
            // optimization: the light became fully invisible, not merely
            // higher-variance), each of its triangles is now marked
            // emissive directly (materialEmission below, lightId left at
            // -1 so the MIS-weight code below correctly treats every hit
            // on it as unweighted, same as a specular bounce) - reachable
            // by a camera ray or a BSDF-sampled bounce landing on it
            // directly, same as any other emissive surface, just with NO
            // NEE strategy sampling it explicitly. A real, unbiased,
            // higher-variance estimator - the same "correct but noisier"
            // tier this loader's own constant/image-based infinite light
            // miss-path-only cases already use (sections 89/90).
            for (int idx : idxs) unhandledLightEmission[idx] = {emission, em.twoSided};
            // UPDATE: each triangle is now ALSO registered as its own NEE light (AreaLightData
            // kind 4 - one triangle, uniform-area sampled), so a fan/mesh light is sampled
            // explicitly. The light-picking pmf is power-proportional (luminance*area) via the
            // alias table, so large triangles get more samples. pbrtEmissiveTriangleLightId
            // hands each triangle its light index so the emissive-hit MIS weight pairs with NEE
            // instead of double counting. Image-textured lights keep the old unsampled tier.
            {
                for (int idx : idxs) {
                    const pbrt_flatten::Triangle& tl = scene.triangles[idx];
                    const float3 a = vertexAt(tl.v, 0), b = vertexAt(tl.v, 1), c = vertexAt(tl.v, 2);
                    const float3 e1 = b - a, e2 = c - a;
                    const float3 cr = simd::cross(e1, e2);
                    const float len = simd::length(cr);
                    if (len < 1e-12f) continue;
                    const float3 nrm = cr / len;
                    pbrtEmissiveTriangleLightId[idx] = (int32_t)lights.size();
                    lights.push_back(AreaLightData{
                        PackedFloat3{a.x, a.y, a.z}, PackedFloat3{e1.x, e1.y, e1.z}, PackedFloat3{e2.x, e2.y, e2.z},
                        PackedFloat3{nrm.x, nrm.y, nrm.z}, 0.5f * len,
                        PackedFloat3{emission.x, emission.y, emission.z},
                        /*patternTileB=*/0.0f, /*patternScale=*/0.0f,
                        /*twoSided=*/em.twoSided ? 1.0f : 0.0f, /*useTexture=*/0.0f,
                        /*pmf=*/0.0f, /*aliasProb=*/1.0f, /*aliasIndex=*/0u,
                        /*kind=*/4.0f, /*spherePrimId=*/-1});
                }
            }
            fprintf(stderr, "loadPbrtScene: area light with %zu triangle(s) is not a simple quad - "
                            "registered per triangle for NEE (one light entry each)\n", idxs.size());
        }
    }
}

// Every triangle loadPbrtAreaLights() didn't already consume as a
// light's own quad - ordinary geometry, or an unhandled (non-quad)
// light's own triangles (emissive via `unhandledLightEmission`, not
// NEE-registered).
void MetalPocApp::loadPbrtRemainingTriangles(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld,
    const PbrtMaterialForFn& materialFor, const std::vector<bool>& triangleHandled,
    const std::unordered_map<int, std::pair<float3, bool>>& unhandledLightEmission) {
    auto vertexAt = [toWorld](const double* v, int i) {
        return toWorld(float3{(float)v[i * 3 + 0], (float)v[i * 3 + 1], (float)v[i * 3 + 2]});
    };
    // Image bump maps ("texture displacement"): each distinct grayscale image is decoded once into the shared float buffer.
    // filename -> {element offset, width, height, 1 if a normal map}; width 0 = could not be used (missing/undecodable).
    // A grayscale image is a height map (1 float per texel, sRGB-decoded); anything else is read as a tangent-space RGB normal map
    // (3 floats per texel, LINEAR - a normal map holds vectors, not colours), as the CPU classifies and reads them.
    std::map<std::string, std::array<int, 4>> bumpImages;
    auto applyImageBump = [&](TriangleMaterial& mat, const pbrt_flatten::Material& m) {
        if (m.displacementTextureFilename.empty() || mat.lightId >= 0 || mat.twoSided != 0u) return;
        auto it = bumpImages.find(m.displacementTextureFilename);
        if (it == bumpImages.end()) {
            std::array<int, 4> entry{0, 0, 0, 0};
            std::string bytes;
            if (pbrt_load::loadFileNear(pbrtScenePath, m.displacementTextureFilename, bytes)) {
                int w = 0, h = 0, ch = 0;
                unsigned char* px = stbi_load_from_memory(reinterpret_cast<const unsigned char*>(bytes.data()),
                                                          (int)bytes.size(), &w, &h, &ch, 3);
                if (px && w > 0 && h > 0) {
                    // The CPU classifies a displacement image by content: grayscale = height map (bump), otherwise a normal map.
                    int maxDiff = 0;
                    for (int sy = 0; sy < 8; ++sy)
                        for (int sx = 0; sx < 8; ++sx) {
                            const unsigned char* p = px + ((size_t)((sy * h) / 8) * w + (sx * w) / 8) * 3;
                            maxDiff = std::max({maxDiff, std::abs((int)p[0] - (int)p[1]), std::abs((int)p[1] - (int)p[2]), std::abs((int)p[0] - (int)p[2])});
                        }
                    if (maxDiff <= 10) {
                        entry = {(int)rgbGridData.size(), w, h, 0};
                        rgbGridData.reserve(rgbGridData.size() + (size_t)w * h);
                        for (size_t k = 0; k < (size_t)w * h; ++k) rgbGridData.push_back(srgb_decode::byteToLinear(px[k * 3]));   // pbrt decodes an 8-bit image as sRGB
                    } else {
                        entry = {(int)rgbGridData.size(), w, h, 1};
                        rgbGridData.reserve(rgbGridData.size() + (size_t)w * h * 3);
                        for (size_t k = 0; k < (size_t)w * h * 3; ++k) rgbGridData.push_back(px[k] / 255.0f);
                    }
                } else {
                    fprintf(stderr, "loadPbrtScene: displacement image '%s' could not be decoded; ignoring it\n", m.displacementTextureFilename.c_str());
                }
                if (px) stbi_image_free(px);
            } else {
                fprintf(stderr, "loadPbrtScene: displacement image '%s' not found; ignoring it\n", m.displacementTextureFilename.c_str());
            }
            it = bumpImages.emplace(m.displacementTextureFilename, entry).first;
        }
        if (it->second[1] > 0) {
            mat.bumpOffset = it->second[0];
            mat.bumpWidth = it->second[1];
            mat.bumpHeight = it->second[2];
            mat.bumpScale = (float)m.displacementScale;
            mat.bumpIsNormalMap = it->second[3];
        }
    };
    for (int i = 0; i < (int)scene.triangles.size(); ++i) {
        if (triangleHandled[i]) continue;
        const pbrt_flatten::Triangle& t = scene.triangles[i];
        const float3 v0 = vertexAt(t.v, 0), v1 = vertexAt(t.v, 1), v2 = vertexAt(t.v, 2);
        verts.push_back(PackedFloat3{v0.x, v0.y, v0.z});
        verts.push_back(PackedFloat3{v1.x, v1.y, v1.z});
        verts.push_back(PackedFloat3{v2.x, v2.y, v2.z});
        if (t.hasNormals) {
            // The vertex stream is FLAT (no index buffer - see addQuad()'s
            // own comment), so push all 3 corner normals, not one shared
            // flat value - shadingNormalFor() barycentric-interpolates
            // whatever sits in each of these 3 slots.
            for (int c = 0; c < 3; ++c)
                normals.push_back(PackedFloat3{(float)t.n[c * 3 + 0], (float)t.n[c * 3 + 1], (float)t.n[c * 3 + 2]});
        } else {
            const float3 faceN = simd::normalize(simd::cross(v1 - v0, v2 - v0));
            const PackedFloat3 packedN{faceN.x, faceN.y, faceN.z};
            normals.push_back(packedN); normals.push_back(packedN); normals.push_back(packedN);
        }
        if (t.hasUVs) {
            for (int c = 0; c < 3; ++c)
                uvs.push_back(PackedFloat2{(float)t.uv[c * 2 + 0], (float)t.uv[c * 2 + 1]});
        } else {
            // pbrt-v4's own real default UV for a trianglemesh with no
            // authored "point2 uv" (J1, section 172; Triangle::hasUVs's
            // own comment, pbrt_flatten.h, already named this exact
            // convention: "CPU triangle.h's own barycentric fallback") -
            // vertex 0/1/2 map to (0,0)/(1,0)/(0,1), so texCoordFor()'s
            // EXISTING barycentric interpolation (unchanged) reduces to
            // the hit's own two barycentric weights directly, matching
            // src/TheRestOfYourLife/triangle.h's `rec.u = b1; rec.v = b2;`
            // exactly - not a flat (0,0) for every point on the triangle,
            // which is what this bare `{0,0}` for all three vertices
            // actually produced before (never a real varying UV at all,
            // however deep any subsequent texture sampling scaffolding
            // went - found via J1's own real-imagemap CoatedDiffuse
            // texture rendering as one FLAT, unvarying colour instead of
            // the checker image's own real pattern, not assumed).
            uvs.push_back(PackedFloat2{0, 0}); uvs.push_back(PackedFloat2{1, 0}); uvs.push_back(PackedFloat2{0, 1});
        }
        TriangleMaterial mat = materialFor(t.material);
        if (t.material >= 0 && t.material < (int)scene.materials.size()) applyImageBump(mat, scene.materials[t.material]);
        if (mat.materialType == METAL_MAT_HAIR) {
            auto tanIt = pbrtTriangleFiberTangent.find(i);
            if (tanIt != pbrtTriangleFiberTangent.end()) mat.conductorK = tanIt->second;   // real fibre tangent (curves)
        }
        auto unhandledIt = unhandledLightEmission.find(i);
        if (unhandledIt != unhandledLightEmission.end()) {
            const float3& unhandledEmission = unhandledIt->second.first;
            mat.emission = PackedFloat3{unhandledEmission.x, unhandledEmission.y, unhandledEmission.z};
            mat.lightId = -1;
            mat.twoSided = unhandledIt->second.second ? 1u : 0u;
            auto triLightIt = pbrtEmissiveTriangleLightId.find(i);
            if (triLightIt != pbrtEmissiveTriangleLightId.end()) mat.lightId = triLightIt->second;
        }
        materials.push_back(mat);
    }
}

// --- Spheres ---------------------------------------------------------
// The bounded-medium material (materialType 28, shadeHomogeneousMediumSpan) of a homogeneous pbrt medium: `color` = per-channel
// single-scattering albedo, `ior` = sigma_t (mean over RGB - the shader takes one scalar - divided by sceneScale so optical depth
// survives the rescale), `roughness` = HG g. A pure absorber (no scattering) can be chromatic, which one scalar sigma_t cannot
// express: sigma_t = 0 plus the per-channel absorption in conductorEta makes the shader apply exp(-sigma_a * chord) per channel.
static TriangleMaterial homogeneousMediumMaterial(const pbrt_flatten::Medium& m, float sceneScale) {
    double sigmaT[3], meanSigmaT = 0.0;
    for (int c = 0; c < 3; ++c) { sigmaT[c] = m.sigma_a[c] + m.sigma_s[c]; meanSigmaT += sigmaT[c]; }
    meanSigmaT /= 3.0;
    PackedFloat3 albedo{0, 0, 0};
    float* a = &albedo.x;
    for (int c = 0; c < 3; ++c) a[c] = sigmaT[c] > 1e-9 ? (float)(m.sigma_s[c] / sigmaT[c]) : 0.0f;
    TriangleMaterial mat{albedo, /*materialType=*/METAL_MAT_MEDIUM_HOMOGENEOUS, /*ior (sigma_t)=*/(float)(meanSigmaT / sceneScale),
                         PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness (g)=*/(float)m.g};
    if (m.sigma_s[0] == 0.0 && m.sigma_s[1] == 0.0 && m.sigma_s[2] == 0.0 && meanSigmaT > 0.0) {
        mat.ior = 0.0f;
        mat.conductorEta = PackedFloat3{(float)(m.sigma_a[0] / sceneScale), (float)(m.sigma_a[1] / sceneScale),
                                        (float)(m.sigma_a[2] / sceneScale)};
    }
    return mat;
}

void MetalPocApp::loadPbrtSpheres(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld,
    const PbrtMaterialForFn& materialFor, float sceneScale) {
    for (const pbrt_flatten::Sphere& s : scene.spheres) {
        const float3 center = toWorld(float3{(float)s.center[0], (float)s.center[1], (float)s.center[2]});
        SphereData sd{PackedFloat3{center.x, center.y, center.z}, sceneScale * (float)s.radius};
        // Object motion blur (F11, section 167): Sphere::center1 differs
        // from Sphere::center only when an ActiveTransform "StartTime"/
        // "EndTime" pair bracketed this shape's own placement (that
        // field's own comment in pbrt_flatten.h) - the shared front-end
        // parser already resolved this, Metal's own loader just never
        // read it before now, the same "already-done-upstream" pattern as
        // B18/B25/D9-D12/F5/F9. Stored as a world-space DELTA
        // (toWorld(center1) - toWorld(center)), not an absolute point, so
        // `toWorld`'s translation term cancels out and only its
        // rotation/scale acts on the raw displacement - correct even
        // though `toWorld` isn't a pure-linear function. Left at its
        // default {0,0,0} (a provable no-op, see SphereData's own
        // comment) for every non-moving sphere, i.e. every sphere in
        // every OTHER scene.
        if (s.center1[0] != s.center[0] || s.center1[1] != s.center[1] || s.center1[2] != s.center[2]) {
            const float3 center1 = toWorld(float3{(float)s.center1[0], (float)s.center1[1], (float)s.center1[2]});
            sd.centerDelta1 = PackedFloat3{center1.x - center.x, center1.y - center.y, center1.z - center.z};
        }
        spheres.push_back(sd);
        TriangleMaterial mat = materialFor(s.material);
        // A `Material "interface"` sphere bounding a homogeneous medium
        // (MediumInterface "fog" "") has no BSDF of its own: it is only the
        // boundary of a participating volume. materialFor() has no case for
        // MaterialKind::Interface, so it would fall back to an OPAQUE gray
        // Lambertian sphere - which, for a fog sphere filling a Cornell box
        // (E1), blocks every light path and renders the whole scene nearly
        // black. Map it to the bounded-medium sphere (materialType 28,
        // shadeHomogeneousMediumSphere) instead: `color` = per-channel
        // single-scattering albedo, `ior` = sigma_t, `roughness` = HG g
        // (that shader's own TriangleMaterial slot reuse). sigma_t is mean
        // over RGB (the shader takes one scalar) and divided by sceneScale
        // for the same reason loadPbrtMedium() does it for the camera fog:
        // optical depth = sigma_t * distance must survive the rescale.
        // Other medium types (cloud/rgbgrid/...) and a real surface material
        // on a medium-bounded sphere are not covered here.
        const bool interfaceSphere = s.areaLight < 0 && s.material >= 0 && s.material < (int)scene.materials.size() &&
            scene.materials[s.material].kind == pbrt_flatten::MaterialKind::Interface;
        if (s.medium >= 0 && s.medium < (int)scene.media.size() && s.areaLight < 0 &&
            s.material >= 0 && s.material < (int)scene.materials.size() &&
            scene.materials[s.material].kind == pbrt_flatten::MaterialKind::Interface &&
            scene.media[s.medium].type == "homogeneous") {
            mat = homogeneousMediumMaterial(scene.media[s.medium], sceneScale);
        } else if (interfaceSphere && s.medium >= 0 && s.medium < (int)scene.media.size() &&
                   scene.media[s.medium].type == "rgbgrid" && scene.media[s.medium].nx > 0 &&
                   scene.media[s.medium].ny > 0 && scene.media[s.medium].nz > 0) {
            // pbrt "rgbgrid" medium (E7): a per-voxel RGB scattering grid inside the interface sphere, rendered by
            // shadeRgbGridMediumSphere (materialType 30; this sphere is its trigger volume, the grid's own AABB
            // clips the actual medium). Distances are rescaled by sceneScale, so every sigma is divided by it.
            // pbrt gives the medium sigma_a = 1 when the scene has no sigma_a grid (a mostly ABSORBING nebula, which
            // is what CPU renders), so that is the constant absorption here; a sigma_a grid is approximated by its
            // mean (the shader has no per-voxel absorption).
            const pbrt_flatten::Medium& gm = scene.media[s.medium];
            const size_t voxels = (size_t)gm.nx * gm.ny * gm.nz;
            const bool hasScatter = gm.sigma_s_r.size() == voxels && gm.sigma_s_g.size() == voxels && gm.sigma_s_b.size() == voxels;
            const bool hasAbsorb = gm.sigma_a_r.size() == voxels && gm.sigma_a_g.size() == voxels && gm.sigma_a_b.size() == voxels;
            const float invScale = 1.0f / sceneScale;
            GpuRgbGridMedium grid{};
            for (int i = 0; i < 3; ++i) { grid.boundsMin[i] = (float)gm.p0[i]; grid.boundsMax[i] = (float)gm.p1[i]; }
            // pbrt world p = (pMetal - sceneOffset)/sceneScale + bboxCenter; medium = M p + t.
            const float3 cOff = -toWorld(float3{0.0f, 0.0f, 0.0f}) * invScale;   // = bboxCenter - sceneOffset/sceneScale
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) grid.worldToMediumMat[r * 3 + c] = (float)gm.toMediumMat[r * 3 + c] * invScale;
                grid.worldToMediumTranslate[r] = (float)gm.toMediumTranslate[r]
                    + (float)(gm.toMediumMat[r * 3 + 0] * cOff.x + gm.toMediumMat[r * 3 + 1] * cOff.y + gm.toMediumMat[r * 3 + 2] * cOff.z);
            }
            grid.nx = gm.nx; grid.ny = gm.ny; grid.nz = gm.nz;
            grid.dataOffset = (int)rgbGridData.size();
            grid.sigmaScale = invScale;
            grid.phaseG = (float)gm.g;
            // Per-channel sigma_s and sigma_a, both per voxel: the shader does spectral tracking (a real event with probability
            // mean_c(sigma_t_c)/majorant, per-channel weights - the same model as the CPU's rgb_grid_medium_hittable.h and
            // OptiX's heterogeneous_tracking_step), so it needs each channel's own coefficients, not a mean or a maximum.
            // `majorant` bounds max_c(sigma_s_c + sigma_a_c) everywhere (the sum of the per-channel voxel maxima, trilinear
            // interpolation never exceeds them).
            float maxS[3] = {0.0f, 0.0f, 0.0f}, maxA[3] = {0.0f, 0.0f, 0.0f};
            int ci = 0;
            for (const std::vector<double>* ch : {&gm.sigma_s_r, &gm.sigma_s_g, &gm.sigma_s_b}) {
                if (hasScatter) {
                    for (double v : *ch) { rgbGridData.push_back((float)v); maxS[ci] = std::max(maxS[ci], (float)v); }
                } else {
                    rgbGridData.insert(rgbGridData.end(), voxels, 1.0f);   // pbrt's default sigma_s grid value
                    maxS[ci] = 1.0f;
                }
                ++ci;
            }
            if (hasAbsorb) {
                grid.saDataOffset = (int)rgbGridData.size();
                grid.sigmaAConst = 0.0f;
                ci = 0;
                for (const std::vector<double>* ch : {&gm.sigma_a_r, &gm.sigma_a_g, &gm.sigma_a_b}) {
                    for (double v : *ch) { rgbGridData.push_back((float)v); maxA[ci] = std::max(maxA[ci], (float)v); }
                    ++ci;
                }
            } else {
                grid.saDataOffset = -1;
                grid.sigmaAConst = 1.0f * invScale;   // pbrt: no "rgb sigma_a" means sigma_a = 1 everywhere
                for (int c = 0; c < 3; ++c) maxA[c] = 1.0f;
            }
            // pbrt "rgb Le" per voxel (plus "Lescale"): the grid emits at its real collisions (shadeRgbGridMediumSphere).
            if (gm.Le_scale > 0.0 && gm.Le_r.size() == voxels && gm.Le_g.size() == voxels && gm.Le_b.size() == voxels) {
                grid.leDataOffset = (int)rgbGridData.size();
                grid.leScale = (float)gm.Le_scale;
                for (const std::vector<double>* ch : {&gm.Le_r, &gm.Le_g, &gm.Le_b})
                    for (double v : *ch) rgbGridData.push_back((float)v);
            } else {
                grid.leDataOffset = -1;
            }
            float maxT = 0.0f;
            for (int c = 0; c < 3; ++c) maxT = std::max(maxT, maxS[c] + maxA[c]);
            grid.sigmaMaj = maxT * invScale * 1.01f;
            const int gridIdx = (int)rgbGridMediums.size();
            rgbGridMediums.push_back(grid);
            mat = TriangleMaterial{};
            mat.materialType = METAL_MAT_MEDIUM_RGB_GRID;
            mat.lightId = -1;
            mat.conductorEta = PackedFloat3{(float)gridIdx, 0.0f, 0.0f};
        } else if (interfaceSphere) {
            // An interface sphere whose medium this loader cannot represent (rgbgrid/
            // nanovdb/cloud/..., or no medium at all). The gray-Lambertian fallback
            // would make it an OPAQUE ball; the boundary itself has no BSDF, so make it
            // transparent instead: a zero-density medium sphere. The volume's own
            // scattering is missing, but the surrounding scene is no longer blocked.
            mat = TriangleMaterial{PackedFloat3{1, 1, 1}, /*materialType=*/METAL_MAT_MEDIUM_HOMOGENEOUS, /*ior (sigma_t)=*/0.0f,
                                   PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness (g)=*/0.0f};
        }
        // A glass sphere (dielectric / thin / rough) that also bounds a homogeneous medium (E3/E11/E12/
        // A9/B13): the medium is simulated for real. The kernel tracks "inside a glass medium" per path and
        // free-flight-samples scattering inside the sphere; shadow rays pass through the sphere with
        // stochastic attenuation (sphereIntersectionFunction), as CPU's do. Parameters ride in this
        // material's spare fields: conductorEta = per-channel sigma_t, conductorK = (g, 1 = has medium, chromatic), transmitColor = per-channel
        // single-scattering albedo; sigma_t is the RGB mean divided by sceneScale (like the camera fog).
        // `color` (the exit-time Beer-Lambert absorption of a plain dielectric) is zeroed - absorption is now
        // part of the medium (albedo < 1), and applying both would double count.
        if ((mat.materialType == METAL_MAT_DIELECTRIC || mat.materialType == METAL_MAT_ROUGH_DIELECTRIC || mat.materialType == METAL_MAT_THIN_DIELECTRIC) &&
            s.medium >= 0 && s.medium < (int)scene.media.size() &&
            scene.media[s.medium].type == "homogeneous") {
            const pbrt_flatten::Medium& gm = scene.media[s.medium];
            double gSigmaT[3], gMeanSigmaT = 0.0;
            for (int c = 0; c < 3; ++c) { gSigmaT[c] = gm.sigma_a[c] + gm.sigma_s[c]; gMeanSigmaT += gSigmaT[c]; }
            gMeanSigmaT /= 3.0;
            mat.color = PackedFloat3{0, 0, 0};
            mat.transmitColor = PackedFloat3{gSigmaT[0] > 1e-9 ? (float)(gm.sigma_s[0] / gSigmaT[0]) : 0.0f,
                                             gSigmaT[1] > 1e-9 ? (float)(gm.sigma_s[1] / gSigmaT[1]) : 0.0f,
                                             gSigmaT[2] > 1e-9 ? (float)(gm.sigma_s[2] / gSigmaT[2]) : 0.0f};
            // conductorEta = per-channel sigma_t; conductorK = (g, has-medium flag, chromatic flag). A chromatic
            // medium makes each path pick ONE colour channel when it enters the sphere (see the kernel), a grey one
            // needs no such colour noise.
            const double gMaxSig = std::max(gSigmaT[0], std::max(gSigmaT[1], gSigmaT[2]));
            const double gMinSig = std::min(gSigmaT[0], std::min(gSigmaT[1], gSigmaT[2]));
            const bool chromatic = (gMaxSig - gMinSig) > 0.02 * std::max(gMaxSig, 1e-12);
            mat.conductorEta = PackedFloat3{(float)(gSigmaT[0] / sceneScale), (float)(gSigmaT[1] / sceneScale), (float)(gSigmaT[2] / sceneScale)};
            mat.conductorK = PackedFloat3{(float)gm.g, 1.0f, chromatic ? 1.0f : 0.0f};
            (void)gMeanSigmaT;
        }
        if (s.areaLight >= 0 && s.areaLight < (int)scene.areaLights.size()) {
            // Emissive hit handled here; NEE registration (a real sphere light
            // in `lights[]`) follows below. (Originally this tier was "visible
            // but not explicitly sampled", like a disk light still is.) `sphereMaterials` is plain
            // TriangleMaterial, so the SAME unconditional direct-hit
            // emissive/MIS-weight code every other emissive material
            // already goes through handles this correctly with no
            // further shader changes needed. A real, previously
            // undiscovered gap (section 164): unlike loadPbrtDisks() just
            // above, this function never read `s.areaLight` at all - a
            // sphere-shaped light rendered as a plain non-emissive grey
            // sphere, contributing NO light to the scene whatsoever (not
            // merely noisier/unsampled - fully absent), closer to the
            // ORIGINAL bug loadPbrtAreaLights()'s own comment describes
            // fixing for non-quad triangle-mesh lights than to the
            // disk-shaped case's own already-correct "noisier but
            // present" tier.
            const pbrt_flatten::Emission& em = scene.areaLights[s.areaLight];
            mat.emission = pbrtFlatLightEmission(pbrtScenePath, em);
            mat.lightId = -1;
            mat.twoSided = em.twoSided ? 1u : 0u;
            // Register it for NEE as a real sphere light (AreaLightData::kind 1,
            // uniform-area sampling - the B14 mechanism in sampleAreaLight()).
            // Left unregistered, a sphere light is only ever found by BSDF-
            // sampled rays: unbiased, but a fog scatter point (which can only
            // reach a light via NEE) never sees it at all (E10 rendered
            // near-black), and everything else is far noisier than CPU.
            // Registered, `lightId` must be set too so the emissive-hit MIS
            // weight pairs with the NEE strategy instead of double counting.
            // Skipped for a moving sphere (the light list holds a fixed
            // position) and an image-textured light (not representable here).
            const bool movingSphere = sd.centerDelta1.x != 0.0f || sd.centerDelta1.y != 0.0f || sd.centerDelta1.z != 0.0f;
            if (!movingSphere) {
                const int32_t lightId = (int32_t)lights.size();
                const float lightArea = 4.0f * (float)M_PI * sd.radius * sd.radius;
                lights.push_back(AreaLightData{
                    PackedFloat3{center.x, center.y, center.z},
                    PackedFloat3{sd.radius, 0.0f, 0.0f},      // edgeU.x = radius for a sphere light
                    PackedFloat3{0.0f, 0.0f, 0.0f},
                    PackedFloat3{0.0f, 0.0f, 0.0f},
                    lightArea,
                    mat.emission,
                    /*patternTileB=*/0.0f, /*patternScale=*/0.0f,
                    /*twoSided=*/0.0f, /*useTexture=*/0.0f,
                    /*pmf=*/0.0f, /*aliasProb=*/1.0f, /*aliasIndex=*/0u,
                    /*kind=*/1.0f,
                    /*spherePrimId=*/(int32_t)spheres.size() - 1});
                mat.lightId = lightId;
            }
        }
        sphereMaterials.push_back(mat);
    }
}

// --- Disks (section 101) - the plain, common "full circle" case only:
    // this loader's own DiskData primitive (metal_poc.metal, matching the
    // hardcoded room's own single disk) is center/normal/radius with no
    // inner-radius/phi-max partial-disk support at all, unlike pbrt-v4's
    // real Disk (Disk::innerRadius/phiMaxDeg). An annular or wedge-shaped
    // disk is warned and skipped entirely (Approx tier's own honesty:
    // rendering a full disk in place of a wedge would be visibly WRONG,
    // not just simplified, so skipping is the safer choice here, unlike
    // e.g. CoatedDiffuse's own "close enough" Approx mapping).
    //
    // Disk::xform is a real 4x4 (translation + rotation + scale, possibly
    // non-uniform) - reuses pbrt_flatten::flatten_detail::transformPoint()/
    // transformNormal() directly (the SAME already-correct, cofactor/
    // adjugate-based utilities ObjectInstance baking above already uses),
    // rather than re-deriving the inverse-transpose normal transform by
    // hand. A disk's own local plane sits at object-space z=`height`
    // (pbrt-v4's own Disk convention) with its face normal along local
    // +Z - transformPoint()/transformNormal() applied to (0,0,height)/
    // (0,0,1) respectively give the real world-space center/normal
    // directly, correct even under non-uniform scale (unlike radius
    // below).
    //
    // Non-uniform scale is detected (not assumed): the LENGTHS of the
    // transformed local X/Y axis vectors must agree (within a loose
    // relative tolerance - transformPoint()'s own floating-point path
    // through a full 4x4, not a hand-verified-exact computation) for a
    // single scalar radius to mean anything at all - an ellipse-shaped
    // disk skips for the same "don't render something visibly wrong"
    // reason a partial disk does, mirroring skippedInstancedSpheres'
    // own non-uniform-scale precedent for instanced spheres.
void MetalPocApp::loadPbrtDisks(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld,
    const PbrtMaterialForFn& materialFor, float sceneScale) {
    size_t skippedDisks = 0;
    auto isInterfaceMaterial = [&scene](int idx) {
        return idx >= 0 && idx < (int)scene.materials.size() &&
               scene.materials[idx].kind == pbrt_flatten::MaterialKind::Interface;
    };
    for (const pbrt_flatten::Disk& d : scene.disks) {
        if (isInterfaceMaterial(d.material)) continue;   // medium boundary only: transparent, not an opaque gray disk
        if (d.innerRadius != 0.0 || d.phiMaxDeg != 360.0) { ++skippedDisks; continue; }

        pbrt_scene::Matrix4 dxform;
        for (int i = 0; i < 16; ++i) dxform.m[i] = d.xform[i];

        double worldOrigin[3], worldAxisX[3], worldAxisY[3];
        pbrt_flatten::flatten_detail::transformPoint(dxform, 0.0, 0.0, 0.0, worldOrigin);
        pbrt_flatten::flatten_detail::transformPoint(dxform, 1.0, 0.0, 0.0, worldAxisX);
        pbrt_flatten::flatten_detail::transformPoint(dxform, 0.0, 1.0, 0.0, worldAxisY);
        const float3 wOrigin{(float)worldOrigin[0], (float)worldOrigin[1], (float)worldOrigin[2]};
        const float scaleX = simd::length(float3{(float)worldAxisX[0], (float)worldAxisX[1], (float)worldAxisX[2]} - wOrigin);
        const float scaleY = simd::length(float3{(float)worldAxisY[0], (float)worldAxisY[1], (float)worldAxisY[2]} - wOrigin);
        if (fabsf(scaleX - scaleY) > 1e-3f * std::max(scaleX, scaleY)) { ++skippedDisks; continue; }

        double worldCenter[3], worldNormal[3];
        pbrt_flatten::flatten_detail::transformPoint(dxform, 0.0, 0.0, d.height, worldCenter);
        pbrt_flatten::flatten_detail::transformNormal(dxform, 0.0, 0.0, 1.0, worldNormal);
        const float3 center = toWorld(float3{(float)worldCenter[0], (float)worldCenter[1], (float)worldCenter[2]});
        const float3 normal = simd::normalize(float3{(float)worldNormal[0], (float)worldNormal[1], (float)worldNormal[2]});

        TriangleMaterial mat = materialFor(d.material);
        if (d.areaLight >= 0 && d.areaLight < (int)scene.areaLights.size()) {
            // Same "emissive, but not NEE-registered" tier PR #100 added
            // for non-quad triangle-mesh lights - this loader has no
            // disk-shaped analytic light in its own `lights[]` NEE list
            // either, so a disk area light is visible (direct hit or a
            // BSDF-sampled bounce landing on it) but not explicitly
            // sampled. `diskMaterials` is plain TriangleMaterial, so the
            // SAME unconditional direct-hit emissive/MIS-weight code PR
            // #100 fixed already handles this correctly with no further
            // shader changes.
            const pbrt_flatten::Emission& em = scene.areaLights[d.areaLight];
            mat.emission = pbrtFlatLightEmission(pbrtScenePath, em, /*radialRowWeight=*/true);
            mat.lightId = -1;
            mat.twoSided = em.twoSided ? 1u : 0u;
            // An image-textured disk light: pbrt maps the image over the disk (u = angle / 2pi from the object x axis towards y, v = (R - r) / R).
            // The shared texture slot is sampled by the light sampler and by direct hits; emission is then the pure scale.
            const bool texturedLight = claimAreaLightImage(em);
            float3 objectX{1, 0, 0};
            if (texturedLight) {
                mat.emission = PackedFloat3{(float)em.scale, (float)em.scale, (float)em.scale};
                mat.materialType = METAL_MAT_IMAGE_EMISSIVE;
                objectX = simd::normalize(float3{(float)worldAxisX[0], (float)worldAxisX[1], (float)worldAxisX[2]} - wOrigin);
            }
            // Register as a real disk light (AreaLightData kind 2, uniform-area sampling),
            // with lightId set so the emissive-hit MIS weight pairs with NEE. Image-textured
            // lights keep the old "emissive but unsampled" tier.
            {
                const float diskRadius = sceneScale * scaleX * (float)d.radius;
                const int32_t lightId = (int32_t)lights.size();
                lights.push_back(AreaLightData{
                    PackedFloat3{center.x, center.y, center.z},
                    PackedFloat3{diskRadius, 0.0f, 0.0f}, PackedFloat3{objectX.x, objectX.y, objectX.z},   // edgeV: the object x axis (the image's u = 0 direction), used only when textured
                    PackedFloat3{normal.x, normal.y, normal.z},
                    (float)M_PI * diskRadius * diskRadius,
                    mat.emission,
                    /*patternTileB=*/0.0f, /*patternScale=*/0.0f,
                    /*twoSided=*/em.twoSided ? 1.0f : 0.0f, /*useTexture=*/texturedLight ? 1.0f : 0.0f,
                    /*pmf=*/0.0f, /*aliasProb=*/1.0f, /*aliasIndex=*/0u,
                    /*kind=*/2.0f, /*spherePrimId=*/-1});
                mat.lightId = lightId;
            }
        }
        disks.push_back(DiskData{PackedFloat3{center.x, center.y, center.z},
                                  PackedFloat3{normal.x, normal.y, normal.z},
                                  sceneScale * scaleX * (float)d.radius});
        diskMaterials.push_back(mat);
    }
    if (skippedDisks > 0)
        fprintf(stderr, "loadPbrtScene: %zu disk(s) skipped - only a full circle (no inner radius/phi-max) "
                        "under uniform scale is supported by this POC's own disk primitive\n", skippedDisks);
}

// --- Cylinders (section 171) - the plain, common "full tube, no ------
// motion blur" case only, the same scope cut loadPbrtDisks() just above
// already established for its own shape: a partial azimuthal sweep (a
// real, non-degenerate case a reference direction would need to be
// carried through, unlike a full disk/cylinder's own rotational
// symmetry) or a non-uniform scale (would make the tube's own cross-
// section an ellipse, not a circle) is warned and skipped entirely -
// the same "rendering the wrong shape would be visibly WRONG, not just
// simplified" reasoning loadPbrtDisks()'s own comment already gives.
// Motion blur (pbrt's own ActiveTransform "StartTime"/"EndTime" around
// a Shape "cylinder", pbrt_flatten::Cylinder::xformEnd) is silently NOT
// read here at all - the same already-accepted, already-documented
// "frozen at its start pose" tier disk-cylinder-motion-blur.pbrt's own
// registry description already gives every GPU backend for this shape,
// not a new gap this loader introduces.
void MetalPocApp::loadPbrtCylinders(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld,
    const PbrtMaterialForFn& materialFor, float sceneScale) {
    size_t skippedCylinders = 0;
    auto isInterfaceMaterial = [&scene](int idx) {
        return idx >= 0 && idx < (int)scene.materials.size() &&
               scene.materials[idx].kind == pbrt_flatten::MaterialKind::Interface;
    };
    for (const pbrt_flatten::Cylinder& cy : scene.cylinders) {
        // A Material "interface" tube is a medium boundary only. Bounding a homogeneous medium it becomes a bounded-medium tube
        // (materialType 28); any other interface tube stays transparent rather than an opaque gray one.
        const bool mediumTube = isInterfaceMaterial(cy.material) && cy.areaLight < 0 && cy.medium >= 0 &&
            cy.medium < (int)scene.media.size() && scene.media[cy.medium].type == "homogeneous";
        if (isInterfaceMaterial(cy.material) && !mediumTube) continue;
        if (cy.phiMaxDeg != 360.0) { ++skippedCylinders; continue; }

        pbrt_scene::Matrix4 cxform;
        for (int i = 0; i < 16; ++i) cxform.m[i] = cy.xform[i];

        double worldOrigin[3], worldAxisX[3], worldAxisY[3];
        pbrt_flatten::flatten_detail::transformPoint(cxform, 0.0, 0.0, 0.0, worldOrigin);
        pbrt_flatten::flatten_detail::transformPoint(cxform, 1.0, 0.0, 0.0, worldAxisX);
        pbrt_flatten::flatten_detail::transformPoint(cxform, 0.0, 1.0, 0.0, worldAxisY);
        const float3 wOrigin{(float)worldOrigin[0], (float)worldOrigin[1], (float)worldOrigin[2]};
        const float scaleX = simd::length(float3{(float)worldAxisX[0], (float)worldAxisX[1], (float)worldAxisX[2]} - wOrigin);
        const float scaleY = simd::length(float3{(float)worldAxisY[0], (float)worldAxisY[1], (float)worldAxisY[2]} - wOrigin);
        if (fabsf(scaleX - scaleY) > 1e-3f * std::max(scaleX, scaleY)) { ++skippedCylinders; continue; }

        double worldBase[3], worldTop[3];
        pbrt_flatten::flatten_detail::transformPoint(cxform, 0.0, 0.0, cy.zMin, worldBase);
        pbrt_flatten::flatten_detail::transformPoint(cxform, 0.0, 0.0, cy.zMax, worldTop);
        const float3 base = toWorld(float3{(float)worldBase[0], (float)worldBase[1], (float)worldBase[2]});
        const float3 top = toWorld(float3{(float)worldTop[0], (float)worldTop[1], (float)worldTop[2]});
        const float3 delta = top - base;
        const float height = simd::length(delta);
        if (height < 1e-6f) { ++skippedCylinders; continue; }
        const float3 axis = delta / height;

        TriangleMaterial mat = mediumTube ? homogeneousMediumMaterial(scene.media[cy.medium], sceneScale) : materialFor(cy.material);
        if (cy.areaLight >= 0 && cy.areaLight < (int)scene.areaLights.size()) {
            // Same "emissive, but not NEE-registered" tier loadPbrtDisks()
            // just above already established - visible (direct hit or a
            // BSDF-sampled bounce) but not explicitly sampled.
            const pbrt_flatten::Emission& em = scene.areaLights[cy.areaLight];
            mat.emission = pbrtFlatLightEmission(pbrtScenePath, em);
            mat.lightId = -1;
            mat.twoSided = em.twoSided ? 1u : 0u;
            // Image-textured: pbrt maps u = angle / 2pi (from the object x axis towards y), v = height fraction. The x axis goes in the
            // light's (otherwise unused) normal slot.
            const bool texturedLight = claimAreaLightImage(em);
            float3 objectX = axis;
            if (texturedLight) {
                mat.emission = PackedFloat3{(float)em.scale, (float)em.scale, (float)em.scale};
                mat.materialType = METAL_MAT_IMAGE_EMISSIVE;
                objectX = simd::normalize(float3{(float)worldAxisX[0], (float)worldAxisX[1], (float)worldAxisX[2]} - wOrigin);
            }
            // Register as a real cylinder light (AreaLightData kind 3, lateral surface only,
            // uniform-area sampling); lightId set so emissive-hit MIS pairs with NEE.
            {
                const float cylRadius = sceneScale * scaleX * (float)cy.radius;
                const int32_t lightId = (int32_t)lights.size();
                lights.push_back(AreaLightData{
                    PackedFloat3{base.x, base.y, base.z},
                    PackedFloat3{axis.x, axis.y, axis.z},          // edgeU = unit axis
                    PackedFloat3{cylRadius, height, 0.0f},         // edgeV = (radius, height)
                    PackedFloat3{objectX.x, objectX.y, objectX.z},   // normal slot: the object x axis when textured (axis otherwise, as before)
                    2.0f * (float)M_PI * cylRadius * height,
                    mat.emission,
                    /*patternTileB=*/0.0f, /*patternScale=*/0.0f,
                    /*twoSided=*/em.twoSided ? 1.0f : 0.0f, /*useTexture=*/texturedLight ? 1.0f : 0.0f,
                    /*pmf=*/0.0f, /*aliasProb=*/1.0f, /*aliasIndex=*/0u,
                    /*kind=*/3.0f, /*spherePrimId=*/-1});
                mat.lightId = lightId;
            }
        }
        // `height` above comes from base/top AFTER toWorld() (which
        // already applies sceneScale itself) - only `radius`, a bare
        // scalar that never goes through toWorld(), needs its own
        // explicit `sceneScale *` here (same reasoning as loadPbrtDisks()'s
        // own `sceneScale * scaleX * d.radius` just above).
        cylinders.push_back(CylinderData{
            PackedFloat3{base.x, base.y, base.z}, PackedFloat3{axis.x, axis.y, axis.z},
            sceneScale * scaleX * (float)cy.radius, height});
        cylinderMaterials.push_back(mat);
    }
    if (skippedCylinders > 0)
        fprintf(stderr, "loadPbrtScene: %zu cylinder(s) skipped - only a full tube (no phi-max sweep) "
                        "under uniform scale is supported by this POC's own cylinder primitive\n", skippedCylinders);
}

// --- ObjectInstance placements -----------------------------------
// Real instancing (scene.groups hold OBJECT-space geometry, defined
// once; scene.instances place them with a per-placement object->world
// transform - see FlatScene's own comment) is baked into plain
// world-space triangles here, rather than mirroring gpu/optix/
// pbrt_gpu_builder.h's own approach of a true GPU-level instance
// acceleration structure - this POC's own shader dispatch already
// resolves each geometry "kind" (room triangles, spheres, disks,
// Suzanne) via its own fixed buffer/intersection-function-table slot,
// so adding a genuinely general N-group instancing mechanism there
// would be a much larger change than this pbrt loader warrants for
// what's typically a handful of placements (e.g. this repo's own
// example-cornell.pbrt: 3). Baking duplicates geometry per placement
// instead of sharing one buffer - free for a scene with a few dozen
// instances, the case every pbrt scene this loader has seen uses.
//
// Emissive instanced shapes need no special handling here: flatten()
// itself already bakes those directly into scene.triangles (a light
// must be enumerable to be sampled - see pbrt_gpu_builder.h's own
// comment on the same point), so scene.groups/scene.instances only
// ever contain non-emissive geometry.
void MetalPocApp::loadPbrtObjectInstances(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld,
    const PbrtMaterialForFn& materialFor) {
    size_t instancedTriangleCount = 0, skippedInstancedSpheres = 0;
    for (const pbrt_flatten::Instance& inst : scene.instances) {
        if (inst.group < 0 || (size_t)inst.group >= scene.groups.size()) {
            fprintf(stderr, "loadPbrtScene: ObjectInstance with an invalid group index skipped\n");
            continue;
        }
        pbrt_scene::Matrix4 xform;
        for (int i = 0; i < 16; ++i) xform.m[i] = inst.xform[i];
        const pbrt_flatten::InstanceGroup& grp = scene.groups[inst.group];
        for (const pbrt_flatten::Triangle& t : grp.triangles) {
            double worldV[9];
            for (int c = 0; c < 3; ++c)
                pbrt_flatten::flatten_detail::transformPoint(xform, t.v[c * 3 + 0], t.v[c * 3 + 1], t.v[c * 3 + 2], &worldV[c * 3]);
            const float3 v0 = toWorld(float3{(float)worldV[0], (float)worldV[1], (float)worldV[2]});
            const float3 v1 = toWorld(float3{(float)worldV[3], (float)worldV[4], (float)worldV[5]});
            const float3 v2 = toWorld(float3{(float)worldV[6], (float)worldV[7], (float)worldV[8]});
            verts.push_back(PackedFloat3{v0.x, v0.y, v0.z});
            verts.push_back(PackedFloat3{v1.x, v1.y, v1.z});
            verts.push_back(PackedFloat3{v2.x, v2.y, v2.z});
            if (t.hasNormals) {
                for (int c = 0; c < 3; ++c) {
                    double worldN[3];
                    pbrt_flatten::flatten_detail::transformNormal(xform, t.n[c * 3 + 0], t.n[c * 3 + 1], t.n[c * 3 + 2], worldN);
                    const float3 n = simd::normalize(float3{(float)worldN[0], (float)worldN[1], (float)worldN[2]});
                    normals.push_back(PackedFloat3{n.x, n.y, n.z});
                }
            } else {
                const float3 faceN = simd::normalize(simd::cross(v1 - v0, v2 - v0));
                const PackedFloat3 packedN{faceN.x, faceN.y, faceN.z};
                normals.push_back(packedN); normals.push_back(packedN); normals.push_back(packedN);
            }
            if (t.hasUVs) {
                for (int c = 0; c < 3; ++c)
                    uvs.push_back(PackedFloat2{(float)t.uv[c * 2 + 0], (float)t.uv[c * 2 + 1]});
            } else {
                // Same real default-UV fix as loadPbrtRemainingTriangles()'s
                // own identical branch just above (J1, section 172) - see
                // that site's own comment for the full "why."
                uvs.push_back(PackedFloat2{0, 0}); uvs.push_back(PackedFloat2{1, 0}); uvs.push_back(PackedFloat2{0, 1});
            }
            materials.push_back(materialFor(t.material));
            ++instancedTriangleCount;
        }
        // A non-uniformly-scaled sphere is an ellipsoid, which SphereData
        // (a plain centre+radius analytic primitive) can't represent -
        // baking it as a sphere anyway would silently render the wrong
        // shape, so this loader skips instanced spheres entirely rather
        // than risk that (matching every other "explain why, don't render
        // something wrong" gap this loader already documents). No pbrt
        // scene this loader has been run against uses one yet.
        skippedInstancedSpheres += grp.spheres.size();
    }
    if (instancedTriangleCount > 0)
        fprintf(stderr, "loadPbrtScene: baked %zu ObjectInstance placement(s) into %zu world-space "
                        "triangle(s)\n", scene.instances.size(), instancedTriangleCount);
    if (skippedInstancedSpheres > 0)
        fprintf(stderr, "loadPbrtScene: %zu instanced sphere(s) skipped - a non-uniformly-scaled "
                        "instanced sphere can't be represented by this loader's analytic sphere "
                        "primitive\n", skippedInstancedSpheres);
}

// --- Punctual lights (point/spot/distant) ---------------------------
void MetalPocApp::loadPbrtPunctualLights(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld, float sceneScale) {
    // Point and spot both reuse PointLightData - the SAME GPU buffer/
    // shading path the hardcoded room's own point+spot lights already use
    // (see that struct's own comment: direction/cosOuterAngle/
    // cosInnerAngle default to "omnidirectional" unless a spot cone
    // overrides them, exactly PointLightData's own existing convention).
    // Distant reuses DirectionalLightData.
    //
    // Intensity/radiance scale compensation: this app's own point/spot
    // falloff is a real 1/distance^2 term evaluated in ITS OWN internal
    // (rescaled) coordinate space (metal_poc.metal's own `/ plDistSq`
    // division, on toWorld()-transformed positions) - leaving a
    // pbrt-authored "I" unchanged while every point/spot light's own
    // distance to a hit point shrinks by sceneScale would inflate
    // apparent brightness by 1/sceneScale^2 relative to the same scene
    // rendered at its own native scale. Multiplying by sceneScale*
    // sceneScale exactly cancels that: irradiance = I/d^2, d'=d*
    // sceneScale => I'=I*sceneScale^2 keeps I'/d'^2 == I/d^2. Distant
    // lights need no such compensation - a directional light's own
    // contribution has no distance term at all (metal_poc.metal's own
    // DirectionalLight comment), the same reason this loader's area
    // lights above also need none (their own area shrinks by
    // sceneScale^2 in lockstep with d^2, cancelling exactly).
    //
    // Goniometric/Projection real profile/slide images: the FIRST light
    // of each kind that names one gets it (section 98, below); a SECOND
    // one of the same kind, or one whose image fails to load/decode,
    // falls back to the Approx (no-image) case and is counted here.
    const float intensityScale = sceneScale * sceneScale;
    size_t skippedImageBasedLights = 0;
    for (const pbrt_flatten::PunctualLight& pl : scene.punctualLights) {
        const float3 baseEmission{(float)(pl.intensity[0] * pl.scale),
                                   (float)(pl.intensity[1] * pl.scale),
                                   (float)(pl.intensity[2] * pl.scale)};
        switch (pl.kind) {
            case pbrt_flatten::PunctualLightKind::Point: {
                const float3 pos = toWorld(float3{(float)pl.pos[0], (float)pl.pos[1], (float)pl.pos[2]});
                const float3 emission = baseEmission * intensityScale;
                pointLights.push_back(PointLightData{
                    PackedFloat3{pos.x, pos.y, pos.z},
                    PackedFloat3{emission.x, emission.y, emission.z}});
                break;
            }
            case pbrt_flatten::PunctualLightKind::Spot: {
                const float3 pos = toWorld(float3{(float)pl.pos[0], (float)pl.pos[1], (float)pl.pos[2]});
                const float3 dir = simd::normalize(float3{(float)pl.dir[0], (float)pl.dir[1], (float)pl.dir[2]});
                const float3 emission = baseEmission * intensityScale;
                pointLights.push_back(PointLightData{
                    PackedFloat3{pos.x, pos.y, pos.z},
                    PackedFloat3{emission.x, emission.y, emission.z},
                    PackedFloat3{dir.x, dir.y, dir.z},
                    /*cosOuterAngle=*/cosf((float)pl.coneAngleDeg * (float)M_PI / 180.0f),
                    /*cosInnerAngle=*/cosf((float)pl.falloffStartAngleDeg * (float)M_PI / 180.0f)});
                break;
            }
            case pbrt_flatten::PunctualLightKind::Distant: {
                // pl.dir is "wi" (toward the light, see PunctualLight's own
                // field comment); DirectionalLightData::direction is the
                // light's own direction of TRAVEL - the sign convention
                // the hardcoded room's own existing directional light
                // already establishes (its `direction` points along -z,
                // "through the room's own open front", i.e. the way the
                // light travels, not back toward its source).
                const float3 wi = simd::normalize(float3{(float)pl.dir[0], (float)pl.dir[1], (float)pl.dir[2]});
                directionalLights.push_back(DirectionalLightData{
                    PackedFloat3{-wi.x, -wi.y, -wi.z},
                    PackedFloat3{baseEmission.x, baseEmission.y, baseEmission.z}});
                break;
            }
            case pbrt_flatten::PunctualLightKind::Goniometric: {
                // A real per-light IES profile image (section 98) - only
                // the FIRST such light in the scene gets one (this loader
                // has one dedicated pbrtGoniometricTexture slot, same
                // "one shared texture" constraint the hardcoded room's
                // own single goniometricTexture already has - see
                // metal_poc.h's own havePbrtGoniometricImage comment). A
                // second one, or a decode failure, falls back to the
                // Approx case below exactly as if no filename were named
                // at all - erring toward a safe, working (if less
                // accurate) render over dropping the light entirely.
                if (pl.hadImageFilename && !havePbrtGoniometricImage) {
                    std::string bytes;
                    if (pbrt_load::loadFileNear(pbrtScenePath, pl.filename, bytes) &&
                        pbrt_load::detail::decodeInfiniteLightImage(pl.filename, bytes,
                            pbrtGoniometricImagePixels, pbrtGoniometricImageWidth, pbrtGoniometricImageHeight)) {
                        havePbrtGoniometricImage = true;
                        const float3 pos = toWorld(float3{(float)pl.pos[0], (float)pl.pos[1], (float)pl.pos[2]});
                        const float3 emission = baseEmission * intensityScale;
                        // Same equal-area octahedral mapping the hardcoded
                        // room's own goniometric light already uses
                        // (equalAreaSphereToSquare(), section 57) - a
                        // right/up frame is needed to express "local"
                        // direction the same way. A real IES profile can
                        // have genuine azimuthal (non-rotationally-
                        // symmetric) variation, so the roll matters here
                        // too, not just for Projection below - using the
                        // scene's own real worldToLight-derived up
                        // (punctualLightWorldUp(), section 98) as the
                        // worldUp hint into the SAME cross-product formula
                        // makeGoniometricLight() already uses recovers
                        // that real roll instead of guessing at it.
                        const float3 forward = punctualLightWorldForward(pl.worldToLight);
                        const float3 worldUpHint = punctualLightWorldUp(pl.worldToLight);
                        const float3 right = simd::normalize(simd::cross(forward, worldUpHint));
                        const float3 up = simd::cross(right, forward);
                        goniometricLights.push_back(GoniometricLightData{
                            PackedFloat3{pos.x, pos.y, pos.z},
                            PackedFloat3{forward.x, forward.y, forward.z},
                            PackedFloat3{right.x, right.y, right.z},
                            PackedFloat3{up.x, up.y, up.z},
                            PackedFloat3{emission.x, emission.y, emission.z},
                            // pl.scale is ALREADY folded into `emission`
                            // above (baseEmission = intensity*pl.scale,
                            // see this function's own top-of-loop
                            // comment) - GoniometricLightData::scale is
                            // an INDEPENDENT multiplier the shader applies
                            // on top (the hardcoded room's own light,
                            // above, passes emission=I with no pl.scale
                            // baked in and a real scale=1.0f here);
                            // passing pl.scale a second time here would
                            // double-count it.
                            /*scale=*/1.0f,
                            /*usePbrtTexture=*/1u});
                        break;
                    }
                    fprintf(stderr, "loadPbrtScene: goniometric light's profile image '%s' could not be "
                                    "read/decoded; falling back to the Approx (uniform) case\n", pl.filename.c_str());
                }
                if (pl.hadImageFilename) ++skippedImageBasedLights;
                // No profile image named (or a second one/a decode
                // failure, both handled above) - pbrt-v4's own documented
                // "Approx" fallback (uniform isotropic intensity, see
                // pbrt_scenes/punctual-lights.pbrt's own header comment
                // for why this is the common case, not just a
                // simplification), which is EXACTLY what a plain
                // omnidirectional PointLightData already represents -
                // isotropic has no direction to get right at all, unlike
                // Projection's own cone/aim (still deferred below), so
                // this needed no new geometry work.
                const float3 pos = toWorld(float3{(float)pl.pos[0], (float)pl.pos[1], (float)pl.pos[2]});
                const float3 emission = baseEmission * intensityScale;
                pointLights.push_back(PointLightData{
                    PackedFloat3{pos.x, pos.y, pos.z},
                    PackedFloat3{emission.x, emission.y, emission.z}});
                break;
            }
            case pbrt_flatten::PunctualLightKind::Projection: {
                // A real per-light slide image (section 98) - same "one
                // dedicated slot, first light wins, decode failure falls
                // back to Approx" reasoning as Goniometric above.
                if (pl.hadImageFilename && !havePbrtProjectionImage) {
                    std::string bytes;
                    if (pbrt_load::loadFileNear(pbrtScenePath, pl.filename, bytes) &&
                        pbrt_load::detail::decodeInfiniteLightImage(pl.filename, bytes,
                            pbrtProjectionImagePixels, pbrtProjectionImageWidth, pbrtProjectionImageHeight)) {
                        havePbrtProjectionImage = true;
                        const float3 pos = toWorld(float3{(float)pl.pos[0], (float)pl.pos[1], (float)pl.pos[2]});
                        const float3 forward = punctualLightWorldForward(pl.worldToLight);
                        // Same look-at-style right/up derivation
                        // makeProjectionLight() already uses for the
                        // hardcoded room's own light, but with the
                        // scene's own REAL worldToLight-derived up
                        // (punctualLightWorldUp(), section 98) as the
                        // worldUp hint instead of a generic axis - a
                        // projected slide has a real, meaningful roll
                        // (the image's own up direction), which only the
                        // scene's own rotation can supply correctly.
                        const float3 worldUpHint = punctualLightWorldUp(pl.worldToLight);
                        const float3 right = simd::normalize(simd::cross(forward, worldUpHint));
                        const float3 up = simd::cross(right, forward);
                        // pl.fovDeg is pbrt's own single "fov" parameter,
                        // which real pbrt-v4 (and this project's own
                        // src/shared/projection_light.h port, already
                        // proven on the CPU/OptiX backends - see its own
                        // "screenBounds"/"aspect" comments) always applies
                        // to the SHORTER image axis: screen bounds are
                        // [-aspect,aspect]x[-1,1] when aspect=width/height
                        // >= 1 (a wide image, so Y/vertical is the fixed,
                        // fov-sized axis and X/horizontal is the wider
                        // one), or [-1,1]x[-1/aspect,1/aspect] when
                        // aspect < 1 (a tall image, X fixed, Y taller).
                        // tanHalfFovBase below is that fixed axis's own
                        // half-angle; the other axis scales by aspect (or
                        // 1/aspect) exactly as projection_light.h derives.
                        const float tanHalfFovBase = tanf((float)pl.fovDeg * 0.5f * (float)M_PI / 180.0f);
                        const float imageAspect = (pbrtProjectionImageHeight > 0)
                            ? (float)pbrtProjectionImageWidth / (float)pbrtProjectionImageHeight : 1.0f;
                        const bool wide = imageAspect >= 1.0f;
                        const float tanHalfFovX = wide ? tanHalfFovBase * imageAspect : tanHalfFovBase;
                        const float tanHalfFovYFinal = wide ? tanHalfFovBase : tanHalfFovBase / imageAspect;
                        projectionLights.push_back(ProjectionLightData{
                            PackedFloat3{pos.x, pos.y, pos.z},
                            PackedFloat3{forward.x, forward.y, forward.z},
                            PackedFloat3{right.x, right.y, right.z},
                            PackedFloat3{up.x, up.y, up.z},
                            tanHalfFovX,
                            tanHalfFovYFinal,
                            (float)pl.scale * intensityScale,
                            /*usePbrtTexture=*/1u});
                        break;
                    }
                    fprintf(stderr, "loadPbrtScene: projection light's slide image '%s' could not be "
                                    "read/decoded; falling back to the Approx (uniform beam) case\n", pl.filename.c_str());
                }
                if (pl.hadImageFilename) ++skippedImageBasedLights;
                // No slide image named - pbrt-v4's own documented
                // "Approx" fallback: a uniform white cone-shaped beam
                // (ProjectionLight::make_uniform(), matching this
                // project's own CPU builder - see pbrt_cpu_builder.h's
                // own kUniformSlide comment). Represented here as a
                // hard-edged PointLightData spot (cosOuterAngle ==
                // cosInnerAngle - spotLightFalloff()'s own
                // max(...,1e-6) denominator guard keeps this a clean
                // cutoff, not a divide-by-zero) rather than a genuinely
                // new light type: a uniform intensity within a cone and
                // zero outside it is exactly what a spot cone already
                // is, just without the smoothstep edge softening a real
                // spot's inner/outer split gives.
                //
                // Aim direction: see punctualLightWorldForward()'s own
                // comment (metal_poc_host_math.h) for the full
                // derivation - verified there against a known rotation
                // case, not just by hand here.
                const float3 pos = toWorld(float3{(float)pl.pos[0], (float)pl.pos[1], (float)pl.pos[2]});
                const float3 dir = punctualLightWorldForward(pl.worldToLight);
                const float3 emission = baseEmission * intensityScale;
                const float cosHalfFov = cosf(0.5f * (float)pl.fovDeg * (float)M_PI / 180.0f);
                pointLights.push_back(PointLightData{
                    PackedFloat3{pos.x, pos.y, pos.z},
                    PackedFloat3{emission.x, emission.y, emission.z},
                    PackedFloat3{dir.x, dir.y, dir.z},
                    /*cosOuterAngle=*/cosHalfFov,
                    /*cosInnerAngle=*/cosHalfFov});
                break;
            }
            default:
                break;
        }
    }
    if (skippedImageBasedLights > 0)
        fprintf(stderr, "loadPbrtScene: %zu goniometric/projection light(s) with a real profile/slide "
                        "image fell back to the Approx (uniform) case - only the first light of each "
                        "kind gets its own image, and a failed decode also falls back here\n",
                skippedImageBasedLights);
}

// --- Homogeneous participating medium (fog) -------------------------
// scene.cameraMediumIndex is already fully resolved and validated by
// pbrt_flatten.h's own post-pass (homogeneous type only, and only set
// at all when the scene has no conflicting real per-shape medium -
// see that field's own comment) - a direct, safe read, no further
// checking needed here.
void MetalPocApp::loadPbrtMedium(const pbrt_flatten::FlatScene& scene, float sceneScale) {
    if (scene.cameraMediumIndex >= 0 &&
        (size_t)scene.cameraMediumIndex < scene.media.size()) {
        const pbrt_flatten::Medium& m = scene.media[(size_t)scene.cameraMediumIndex];
        double sigmaT[3];
        double meanSigmaT = 0.0;
        for (int c = 0; c < 3; ++c) {
            sigmaT[c] = m.sigma_a[c] + m.sigma_s[c];
            meanSigmaT += sigmaT[c];
        }
        meanSigmaT /= 3.0;
        // Extinction has units of inverse length - the SAME rescale that
        // shrinks every position/distance by sceneScale (see this
        // function's own toWorld() comment) shrinks a ray's own travelled
        // distance in lockstep, so leaving sigmaT at its pbrt-native value
        // would UNDER-attenuate by the same factor (optical depth =
        // sigmaT*dist; dist'=dist*sceneScale, so sigmaT'=sigmaT/sceneScale
        // is what keeps sigmaT'*dist' == sigmaT*dist). The OPPOSITE
        // direction from the punctual-light intensity compensation above
        // (which multiplies by sceneScale^2) - that one compensates a
        // squared-length falloff term, this one a single inverse-length
        // one, so the correction is an inverse first power, not a square.
        havePbrtMedium = true;
        // The LOCAL `sceneScale` (computed above, already applied to
        // every vertex/light/camera position via toWorld()) - not the
        // MEMBER `pbrtSceneScale`, which this same function only
        // assigns much later, in its own Camera section below (for
        // applyCameraOverride()'s later use). Reading the member here
        // was a real, previously-latent bug: since loadPbrtScene() runs
        // exactly once, that member is still its 1.0f default-
        // initializer value at this point in EVERY call, so every real
        // pbrt scene's own fog silently used 1/sceneScale too little
        // attenuation (e.g. ~10x for camera-medium.pbrt's own ~20-unit
        // scale, worse for a larger scene) - found via code review
        // while mapping this function for section 108's own refactor,
        // not a symptom (the fog was still visibly present just too
        // faint, and PR #88's own on/off verification wasn't sensitive
        // to the WRONG-MAGNITUDE case, only presence/absence).
        pbrtFogSigmaT = (float)(meanSigmaT / sceneScale);
        // Chromatic medium: the kernel follows one colour channel per path (like a chromatic glass medium) and samples free
        // flight with that channel's sigma_t, so the per-channel extinction has to be handed over too.
        pbrtFogSigmaT3 = float3{(float)(sigmaT[0] / sceneScale), (float)(sigmaT[1] / sceneScale), (float)(sigmaT[2] / sceneScale)};
        {
            double lo = sigmaT[0], hi = sigmaT[0];
            for (int c = 1; c < 3; ++c) { lo = std::min(lo, sigmaT[c]); hi = std::max(hi, sigmaT[c]); }
            pbrtFogChromatic = hi > 0.0 && (hi - lo) > 0.01 * hi;
        }
        // fogAlbedo is single-scattering albedo (sigma_s/sigma_t) PER
        // CHANNEL (metal_poc.metal's own field comment) - unlike sigmaT
        // itself, this ratio is dimensionless and scale-invariant, so the
        // real per-channel colour survives even though the overall
        // interaction RATE above is reduced to one achromatic scalar -
        // the same simplification this shader's own hardcoded-room fog
        // already makes (a single fogSigmaT, never a per-channel one).
        for (int c = 0; c < 3; ++c) {
            const float t = (float)sigmaT[c];
            pbrtFogAlbedo[c] = (t > 1e-9f) ? (float)(m.sigma_s[c] / sigmaT[c]) : 0.0f;
        }
        pbrtFogAsymmetryG = (float)m.g;   // dimensionless, no scale compensation needed
        fprintf(stderr, "loadPbrtScene: homogeneous camera medium found (sigma_t~%.5g/unit, g=%.3f)\n",
                meanSigmaT, m.g);
    }
}

// --- Infinite light ---------------------------------------------------
// earthTexture is ALSO the hardcoded room's own materialType-3
// back-wall albedo, sampled from a totally separate code path in
// primaryRayKernel (the `albedo = earthTexture.sample(...)` line, run
// BEFORE any of the 6 material-shading functions are even called) -
// repointing that one at a pbrt-provided image would silently corrupt
// that unrelated, still-active geometry. So this uses its OWN,
// genuinely separate texture (pbrtEnvTexture) instead - see
// primaryRayKernel's own texture-argument comment.
//
// The image case gets a real NEE/MIS light-sampling strategy (section 97):
// every material's own shading function that already does NEE against
// earthTexture's own environment map also importance-samples pbrtEnvTexture
// directly, via a SEPARATE EnvDistribution2D built from pbrtEnvImagePixels
// (metal_poc_host_math.h's own float-RGB buildEnvDistribution2D() overload,
// added in section 90, wired up in section 97 - the same "phase 1 before
// phase 2" staging earthTexture's own NEE support went through in sections
// 69/71) - see pbrtEnvMarginalCDF/pbrtEnvConditionalCDF/pbrtEnvMapWidth/
// pbrtEnvMapHeight (buffers 22/23, metal_poc_dispatch.mm) and each shadeXxx()
// function's own "pbrtEnv" NEE block. The constant-colour case stays
// miss-path-only deliberately, not as an open gap - a spatially uniform
// environment light needs no separate importance-sampling strategy at all;
// cosine-weighted (or the material's own specular/GGX) BSDF sampling is
// already optimal for a constant-radiance background, which is exactly why
// no material function has a "pbrtEnvColor NEE" block anywhere.
void MetalPocApp::loadPbrtInfiniteLight(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld) {
    if (scene.infiniteLight.present) {
        if (scene.infiniteLight.hasPortal) {
            // pbrt-v4 portal (windowed) infinite light. Same host recipe as the OptiX builder (gpu/optix/pbrt_gpu_builder.h): a real
            // PortalImageInfiniteLightData does the equal-area rectification and builds the summed-area table, and the shader reads those
            // arrays back. With no usable image the light fails CLOSED (no light at all), like the CPU - never an unwindowed sky.
            if (scene.infiniteLight.imageWidth > 0 && scene.infiniteLight.imageHeight > 0 && !scene.infiniteLight.imagePixels.empty()) {
                std::array<Vec3<double>, 4> corners;
                for (int i = 0; i < 4; ++i)
                    corners[i] = Vec3<double>(scene.infiniteLight.portal[i*3+0], scene.infiniteLight.portal[i*3+1], scene.infiniteLight.portal[i*3+2]);
                PortalImageInfiniteLightData<double> portalData(scene.infiniteLight.imagePixels.data(), scene.infiniteLight.imageWidth,
                                                                 scene.infiniteLight.imageHeight, scene.infiniteLight.scale, corners);
                havePbrtPortalLight = true;
                pbrtPortalWidth = portalData.width();
                pbrtPortalHeight = portalData.height();
                pbrtPortalScale = (float)portalData.scale();
                pbrtPortalRectified = portalData.rectified();
                const Array2D<float>& func = portalData.distribution().func();
                pbrtPortalDistFunc.assign(func.data().begin(), func.data().end());
                const std::vector<double>& sat = portalData.distribution().sat().sum().data();
                pbrtPortalSatSum.assign(sat.begin(), sat.end());
                // Frame::FromXY(p03, p01) as the CPU class builds it (nx = p03, nz = normalize(cross(nx, p01)), ny = cross(nz, nx)).
                const auto p01 = pil_detail::normalize(pil_detail::sub(corners[1], corners[0]));
                const auto nx = pil_detail::normalize(pil_detail::sub(corners[3], corners[0]));
                const auto nz = pil_detail::normalize(pil_detail::cross(nx, p01));
                const auto ny = pil_detail::cross(nz, nx);
                pbrtPortalFrameX = float3{(float)nx.x, (float)nx.y, (float)nx.z};
                pbrtPortalFrameY = float3{(float)ny.x, (float)ny.y, (float)ny.z};
                pbrtPortalFrameZ = float3{(float)nz.x, (float)nz.y, (float)nz.z};
                // The window's position matters (it is seen from each shading point), and loadPbrtScene() rescales/recentres/offsets all geometry
                // into Metal world space - so the corners go through the same transform. The frame is a direction, untouched by it.
                pbrtPortalP0 = toWorld(float3{(float)corners[0].x, (float)corners[0].y, (float)corners[0].z});
                pbrtPortalP2 = toWorld(float3{(float)corners[2].x, (float)corners[2].y, (float)corners[2].z});
                fprintf(stderr, "loadPbrtScene: portal infinite light found (%dx%d, scale=%.3g)\n", pbrtPortalWidth, pbrtPortalHeight, pbrtPortalScale);
            } else {
                fprintf(stderr, "loadPbrtScene: portal infinite light has no usable image; no environment light is added (the CPU does the same)\n");
            }
        } else
        if (scene.infiniteLight.imageWidth > 0 && scene.infiniteLight.imageHeight > 0 &&
            !scene.infiniteLight.imagePixels.empty()) {
            havePbrtImageEnvLight = true;
            pbrtEnvImageWidth = scene.infiniteLight.imageWidth;
            pbrtEnvImageHeight = scene.infiniteLight.imageHeight;
            // scale applied here (not left for a shader-side multiply) -
            // matches how the constant-colour case below bakes L*scale
            // at load time too, and how src/TheRestOfYourLife/
            // pbrt_cpu_builder.h's own sky_light(...) constructor takes
            // scale as a SEPARATE multiplier on the raw image samples
            // (not pre-baked into scene.infiniteLight.imagePixels itself).
            const float scale = (float)scene.infiniteLight.scale;
            pbrtEnvImagePixels = scene.infiniteLight.imagePixels;
            for (float& v : pbrtEnvImagePixels) v *= scale;
            fprintf(stderr, "loadPbrtScene: image-based infinite light found (%dx%d, scale=%.3g)\n",
                    pbrtEnvImageWidth, pbrtEnvImageHeight, scale);
        } else {
            havePbrtConstantEnvLight = true;
            pbrtEnvColor = float3{(float)(scene.infiniteLight.L[0] * scene.infiniteLight.scale),
                                   (float)(scene.infiniteLight.L[1] * scene.infiniteLight.scale),
                                   (float)(scene.infiniteLight.L[2] * scene.infiniteLight.scale)};
            fprintf(stderr, "loadPbrtScene: constant-colour infinite light found (L*scale ~ %.3g %.3g %.3g)\n",
                    pbrtEnvColor.x, pbrtEnvColor.y, pbrtEnvColor.z);
        }
    }
    // Disks and cylinders have their own primitives; cone/paraboloid/bilinearmesh/curve are
    // tessellated into triangles up front (tessellateUnsupportedShapes()). Only the ANIMATED
    // (motion-blurred) variants of patches and curves are still dropped.
    if (!scene.animatedBilinearPatches.empty() || !scene.animatedCurves.empty())
        fprintf(stderr, "loadPbrtScene: animated (motion-blurred) bilinearmesh/curve shapes skipped - "
                        "only their static forms are supported by this POC's scene loader\n");
}

// --- Camera ------------------------------------------------------------
void MetalPocApp::loadPbrtCamera(const pbrt_flatten::FlatScene& scene, const PbrtToWorldFn& toWorld,
    float3 bboxCenter, float sceneScale, float3 sceneOffset) {
    const pbrt_flatten::Camera& cam = scene.camera;
    // lookfrom/lookat are positions, scaled the same as every vertex
    // above; `up` is a direction (scale-invariant, and normalized below
    // regardless).
    const float3 lookfrom = toWorld(float3{(float)cam.lookfrom[0], (float)cam.lookfrom[1], (float)cam.lookfrom[2]});
    const float3 lookat = toWorld(float3{(float)cam.lookat[0], (float)cam.lookat[1], (float)cam.lookat[2]});
    const float3 up{(float)cam.up[0], (float)cam.up[1], (float)cam.up[2]};
    const float3 forward = simd::normalize(lookat - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    pbrtTanHalfFov = tanf(0.5f * (float)cam.vfov * (float)M_PI / 180.0f);
    havePbrtCamera = true;
    // Saved for applyCameraOverride() - see that method's own comment.
    pbrtCameraLookAtWorld = lookat;
    pbrtCameraUpRaw = up;
    pbrtBboxCenter = bboxCenter;
    pbrtSceneScale = sceneScale;
    pbrtSceneOffset = sceneOffset;

    // Perspective/orthographic thin-lens DOF ("float lensradius"/"float
    // focaldistance", pbrt_flatten.h's own c.aperture/c.focusDistance -
    // c.aperture is already lensradius*2, a world-space DIAMETER, see
    // that field's own comment) - D9's own real gap (section 159): every
    // hand-authored D1/D5/D6 scene already sets pbrtLensRadius/
    // pbrtFocusDistance directly (defaulting to 0/1, "no DOF"), but a
    // REAL loaded pbrt file's own lensradius/focaldistance parameters
    // were never actually read here at all until now - depth-of-
    // field.pbrt rendered pinhole-sharp regardless of its own "float
    // lensradius" [20] before this fix. Same sceneScale-multiplied
    // world-space-distance convention pbrtLensRadius/pbrtFocusDistance
    // already use everywhere else (D5's own comment). Spherical/
    // Realistic cameras never read lensRadius/focusDistance at all
    // (Uniforms::cameraRealistic's own DOF-skip guard, metal_poc_kernel.
    // metal) - deliberately NOT set for those two below, matching real
    // pbrt (RealisticCamera has its own, wholly different depth-of-
    // field mechanism built into the lens trace itself; SphericalCamera
    // has none).
    if (cam.type == "perspective" || cam.type == "orthographic") {
        pbrtLensRadius = (float)(cam.aperture * 0.5) * sceneScale;
        pbrtFocusDistance = (float)pbrt_flatten::focusDistanceFor(cam) * sceneScale;
    }

    // Non-perspective camera types loaded from a REAL pbrt file (D10/
    // D11/D12, section 159) - the generic, data-driven counterpart to
    // D2/D6 (orthographic)/D3/D7 (spherical)/D4/D8 (realistic)'s own
    // hand-authored scenes, which set these same havePbrt*/pbrt* fields
    // directly instead of reading them from a Camera directive's own
    // parameters. Mirrors gpu/optix/scene_builder.cpp's own generic
    // Camera-type dispatch (the already-shipped CUDA reference this
    // block is ported from) - same fields, same formulas, just this
    // loader's own Objective-C++ types/RealisticCamera<float> instead
    // of OptiX's CUDA structs.
    if (cam.type == "orthographic") {
        // Mirrors D6's own buildOrthoCornellBox() - pbrtTanHalfFov
        // repurposed as the orthographic screen window's own
        // (necessarily symmetric - Uniforms::cameraOrthographic's own
        // comment, metal_poc_types.metal) world-space half-extent, in
        // THIS loader's own rescaled units. `screenwindow`'s own
        // xmax (== -xmin for every screenwindow this loader's own
        // scenes actually use, including orthographic-camera.pbrt's
        // own symmetric [-320,320,-320,320]) supplies it directly when
        // given; pbrt's own "roughly 1-unit-across" default otherwise
        // (pbrt_flatten::Camera::screenWindow's own comment).
        havePbrtOrthographic = true;
        pbrtTanHalfFov = (cam.hasScreenWindow ? (float)cam.screenWindow[1] : 1.0f) * sceneScale;
    } else if (cam.type == "spherical" || cam.type == "environment") {
        // Mirrors D7's own buildSphericalCornellBox() - a panoramic
        // camera has no screen window/FOV/DOF at all, only the
        // position/orientation already set above. `sphericalMapping`
        // additionally selects EquiRectangular (pbrt-v4's own default,
        // matching every hand-authored D3/D7 scene already) vs.
        // EqualArea (spherical-camera.pbrt's own "string mapping"
        // ["equalarea"] - the first scene, hand-authored or loaded,
        // to actually request it - Uniforms::sphericalMappingEqualArea's
        // own comment).
        havePbrtSpherical = true;
        havePbrtSphericalEqualArea = (cam.sphericalMapping == "equalarea");
    } else if (cam.type == "realistic") {
        // Mirrors D4/D8's own buildRealisticCameraScene()/
        // buildRealisticCornellBox() construction exactly, just with the
        // lens TABLE itself read from a real file (pbrt_load::
        // loadFileNear()/parseLensFile(), both already shared with CPU/
        // OptiX - the "half the lensfile-loading path a compiled-in
        // scene never exercised" gap D12's own registry description
        // names) instead of a literal std::vector in this function's own
        // source. Film half-extents are DERIVED from filmDiagonalMM +
        // this render's own aspect ratio (pbrt-v4's real convention,
        // and gpu/optix/scene_builder.cpp's own identical formula) -
        // D4/D8's own scenes instead gave film_x_mm/film_y_mm directly,
        // since a hand-authored scene has no separate "diagonal"
        // parameter to derive them from.
        std::string lensText;
        if (cam.lensFile.empty()) {
            fprintf(stderr, "loadPbrtCamera: a realistic camera has no \"lensfile\"; "
                            "rendering as perspective instead (should not normally be "
                            "reachable - pbrt_flatten.h already falls back to "
                            "\"perspective\" at flatten time when lensfile is missing).\n");
        } else if (!pbrt_load::loadFileNear(pbrtScenePath, cam.lensFile, lensText)) {
            fprintf(stderr, "loadPbrtCamera: realistic camera lensfile '%s' not found "
                            "near '%s'; rendering as perspective instead.\n",
                    cam.lensFile.c_str(), pbrtScenePath.c_str());
        } else {
            const std::vector<double> lensD = pbrt_load::parseLensFile(lensText);
            if (lensD.empty()) {
                fprintf(stderr, "loadPbrtCamera: realistic camera lensfile '%s' has no "
                                "usable rows; rendering as perspective instead.\n",
                        cam.lensFile.c_str());
            } else {
                std::vector<float> lensParams;
                lensParams.reserve(lensD.size());
                for (double v : lensD) lensParams.push_back((float)v);
                const float aspectF = (height > 0) ? (float)width / (float)height : 1.0f;
                const float halfY = (float)cam.filmDiagonalMM / (2.0f * sqrtf(aspectF * aspectF + 1.0f));
                const float halfX = aspectF * halfY;
                const float focusDist = (float)pbrt_flatten::focusDistanceFor(cam);
                const float apertureDiameter = (float)cam.apertureDiameterMM;
                RealisticCamera<float> realCam(Mat4<float>{}, halfX, halfY, focusDist,
                                                apertureDiameter, lensParams);
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
                havePbrtRealisticCamera = true;
            }
        }
    }
}
