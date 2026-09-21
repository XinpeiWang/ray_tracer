// metal_poc_pbrt_loader.mm
// Real pbrt scene loading - MetalPocApp::loadPbrtScene() and its 9 phase
// methods (area lights, remaining triangles, spheres, disks, cylinders,
// object instances, punctual lights, medium, infinite light, camera) -
// split out of metal_poc.mm once that one file's own "Stage 2.5" block
// grew past a third of it (a pure code-motion refactor, no behaviour
// change - see metal_poc_scenes_a.mm's own header comment for the
// identical precedent this follows, and docs/METAL_GPU_FEASIBILITY.md).
// buildScene() (the hardcoded room), buildHandAuthoredScene() (the
// scene_id dispatcher), and everything GPU-resource/dispatch-related
// stay in metal_poc.mm itself - this file is only "read a .pbrt file
// into the app's own scene data," nothing else.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_poc_app.h"

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
void MetalPocApp::loadPbrtScene() {
    pbrt_load::LoadResult result = pbrt_load::loadFile(pbrtScenePath);
    if (!result.ok) {
        fprintf(stderr, "loadPbrtScene: %s\n", result.error.c_str());
        return;
    }
    const pbrt_flatten::FlatScene& scene = result.scene;
    for (const pbrt_scene::Warning& w : scene.warnings) {
        fprintf(stderr, "loadPbrtScene: pbrt loader warning: %s\n", w.message.c_str());
    }

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

    // Tracks which UNSUPPORTED material kind NAMES (not indices - the same
    // name can legitimately appear at more than one scene.materials index)
    // have already been warned about, so mapMaterial() below - called once
    // per TRIANGLE/sphere/instance referencing a material, not once per
    // distinct material - doesn't flood stderr with the identical message
    // for every single primitive. A mesh with tens of thousands of
    // triangles sharing one unsupported material (e.g. killeroo-simple.pbrt's
    // own 66,532-triangle "coateddiffuse" mesh) used to print that exact
    // line 66,532 times.
    std::unordered_set<std::string> warnedUnsupportedMaterialKinds;
    // std::function (not auto), capturing itself by reference, so the
    // Mix case below can recurse into mapMaterial() for its own two
    // named sub-materials - an ordinary auto lambda can't reference its
    // own name inside its own body (not yet in scope at that point).
    std::function<TriangleMaterial(const pbrt_flatten::Material&, int)> mapMaterial =
        [this, &warnedUnsupportedMaterialKinds, &scene, &mapMaterial](const pbrt_flatten::Material& m, int depth) -> TriangleMaterial {
        PackedFloat3 color{(float)m.color[0], (float)m.color[1], (float)m.color[2]};
        switch (m.kind) {
            case pbrt_flatten::MaterialKind::Diffuse: {
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
                // default below instead of misrendering it.
                if (m.hasCheckerReflectance && m.checkerTex1Filename.empty() && m.checkerTex2Filename.empty()) {
                    TriangleMaterial mat{
                        PackedFloat3{(float)m.checkerColor1[0], (float)m.checkerColor1[1], (float)m.checkerColor1[2]},
                        /*materialType=*/25u, /*ior=*/1.0f, PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
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
                // loaded something" - `mapMaterial()` is called once per
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
                        return TriangleMaterial{color, /*materialType=*/26u, /*ior=*/1.0f,
                                                 PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/(float)m.textureScale};
                    }
                }
                return TriangleMaterial{color, /*materialType=*/0u, /*ior=*/1.0f,
                                         PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
            }
            case pbrt_flatten::MaterialKind::Conductor: {
                // RoughnessToAlpha (src/shared/microfacet.h) is sqrt(r) -
                // pbrt-v4's own "remaproughness" default (true) means the
                // authored value needs this remap; false means it already
                // IS alpha.
                const float alpha = (float)(m.remapRoughness ? std::sqrt(m.roughness) : m.roughness);
                TriangleMaterial mat{color, /*materialType=*/4u, /*ior(alphaX)=*/alpha,
                                     PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness(alphaY)=*/alpha};
                mat.conductorEta = PackedFloat3{(float)m.conductorEta[0], (float)m.conductorEta[1], (float)m.conductorEta[2]};
                mat.conductorK = PackedFloat3{(float)m.conductorK[0], (float)m.conductorK[1], (float)m.conductorK[2]};
                return mat;
            }
            case pbrt_flatten::MaterialKind::CoatedConductor: {
                // Approx tier (docs/PBRT_SUPPORT.md's own note, matching
                // both CPU's pbrt_cpu_builder.h and GPU-OptiX's own
                // pbrt_gpu_builder_materials.h): materialType 4 (GGX
                // conductor) - the "coat" itself (a separate rough
                // dielectric layer over the metal base) isn't modelled at
                // all, only the base conductor's own eta/k/roughness -
                // the same simplification both other backends already
                // accept for this kind, not a NEW approximation invented
                // here. A named metal spectrum or explicit "eta"/"k" (m.
                // hasConductorPreset) is used directly, already resolved
                // by pbrt_flatten.h identically to plain Conductor above;
                // otherwise `color` (the scene's own "reflectance", a
                // normal-incidence Schlick value) is converted via the
                // SAME eta=1/k-solved-from-r formula CPU's own
                // reflectanceToConductorK() uses (k = 2*sqrt(r) /
                // sqrt(max(1e-4, 1-r)), per channel) - not re-derived
                // independently, matching a real precedent already
                // established for exactly this "nothing given" case.
                const float alpha = (float)(m.remapRoughness ? std::sqrt(m.roughness) : m.roughness);
                TriangleMaterial mat{color, /*materialType=*/4u, /*ior(alphaX)=*/alpha,
                                     PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness(alphaY)=*/alpha};
                if (m.hasConductorPreset) {
                    mat.conductorEta = PackedFloat3{(float)m.conductorEta[0], (float)m.conductorEta[1], (float)m.conductorEta[2]};
                    mat.conductorK = PackedFloat3{(float)m.conductorK[0], (float)m.conductorK[1], (float)m.conductorK[2]};
                } else {
                    auto reflectanceToK = [](float r) {
                        r = r < 0.0f ? 0.0f : (r > 0.9999f ? 0.9999f : r);
                        return 2.0f * sqrtf(r) / sqrtf(std::max(1e-4f, 1.0f - r));
                    };
                    mat.conductorEta = PackedFloat3{1.0f, 1.0f, 1.0f};
                    mat.conductorK = PackedFloat3{reflectanceToK(color.x), reflectanceToK(color.y), reflectanceToK(color.z)};
                }
                return mat;
            }
            case pbrt_flatten::MaterialKind::Dielectric:
                return TriangleMaterial{color, /*materialType=*/2u, /*ior=*/(float)m.ior,
                                         PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
            case pbrt_flatten::MaterialKind::ThinDielectric:
                // materialType 11 - `color` is unused by this material
                // (metal_poc.mm's own hardcoded-room construction site
                // comment), only `ior` matters.
                return TriangleMaterial{color, /*materialType=*/11u, /*ior=*/(float)m.ior,
                                         PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
            case pbrt_flatten::MaterialKind::DiffuseTransmission: {
                // materialType 12 - `color` is this material's own
                // reflectance tint (same convention as Diffuse above);
                // `m.transmittance` is the SEPARATE transmitted tint
                // (Material::transmittance's own comment - a frosted-
                // panel default of 0.25 each way when the scene names
                // neither parameter, not a mirror-symmetric 0.5/0.5 split
                // of `color`).
                TriangleMaterial mat{color, /*materialType=*/12u, /*ior=*/1.0f,
                                     PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
                mat.transmitColor = PackedFloat3{(float)m.transmittance[0], (float)m.transmittance[1],
                                                  (float)m.transmittance[2]};
                return mat;
            }
            case pbrt_flatten::MaterialKind::CoatedDiffuse: {
                // Approx tier (docs/PBRT_SUPPORT.md's own convention -
                // CPU/OptiX render this Full, a real stochastic layered
                // coat-over-Lambertian): materialType 8 (clearcoat) is
                // this POC's own simplified SINGLE-bounce smooth-
                // dielectric-coat-over-Lambertian, with a fixed
                // kClearcoatEta=1.5 shader-side constant - the scene's
                // own "ior"/"roughness" (a rough, scene-specified coat)
                // are silently NOT read here, unlike materialType 8's
                // hardcoded-room use, which never varies them either.
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
                        return TriangleMaterial{color, /*materialType=*/27u, /*ior=*/1.0f,
                                                 PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/(float)m.textureScale};
                    }
                }
                return TriangleMaterial{color, /*materialType=*/8u, /*ior=*/1.0f,
                                         PackedFloat3{0, 0, 0}, /*lightId=*/-1, /*roughness=*/0.0f};
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
                // favours, recursing into mapMaterial() for that ONE
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
                // catch it. Without this guard, mapMaterial()'s own
                // recursion into such a scene would stack-overflow this
                // loader before any render starts - a real bug found by
                // code review, not exercised by any bundled scene.
                const int chosenIdx = (m.mixWeight >= 0.5) ? m.mixMaterialB : m.mixMaterialA;
                constexpr int kMaxMixDepth = 8;
                if (depth < kMaxMixDepth && chosenIdx >= 0 && chosenIdx < (int)scene.materials.size())
                    return mapMaterial(scene.materials[chosenIdx], depth + 1);
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
    };
    auto materialFor = [&](int idx) -> TriangleMaterial {
        if (idx < 0 || idx >= (int)scene.materials.size())
            return TriangleMaterial{PackedFloat3{0.5f, 0.5f, 0.5f}, 0u, 1.0f, PackedFloat3{0, 0, 0}, -1, 0.0f};
        return mapMaterial(scene.materials[idx], /*depth=*/0);
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
    loadPbrtInfiniteLight(scene);
    loadPbrtCamera(scene, toWorld, bboxCenter, sceneScale, sceneOffset);

    fprintf(stderr, "loadPbrtScene: loaded %s (%zu triangles, %zu spheres, %zu area lights)\n",
            pbrtScenePath.c_str(), scene.triangles.size(), scene.spheres.size(), scene.areaLights.size());
}

// --- Area lights: only the "single quad, 2 triangles" shape - see
// this function's own header comment.
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
        float3 emission{(float)(em.L[0] * em.scale), (float)(em.L[1] * em.scale), (float)(em.L[2] * em.scale)};
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
                if (!em.filename.empty() && !havePbrtAreaLightImage) {
                    std::string bytes;
                    if (pbrt_load::loadFileNear(pbrtScenePath, em.filename, bytes) &&
                        pbrt_load::detail::decodeInfiniteLightImage(em.filename, bytes,
                            pbrtAreaLightImagePixels, pbrtAreaLightImageWidth, pbrtAreaLightImageHeight)) {
                        havePbrtAreaLightImage = true;
                        quadMaterialType = 15u;
                        useTextureFlag = 1.0f;
                        emission = float3{(float)em.scale, (float)em.scale, (float)em.scale};
                    } else {
                        fprintf(stderr, "loadPbrtScene: area light's own image '%s' could not be read/decoded; "
                                        "falling back to its flat colour\n", em.filename.c_str());
                    }
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
            fprintf(stderr, "loadPbrtScene: area light with %zu triangle(s) is not a simple quad - "
                            "rendering it emissive but without an explicit NEE strategy (higher variance, "
                            "not invisible)\n", idxs.size());
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
        auto unhandledIt = unhandledLightEmission.find(i);
        if (unhandledIt != unhandledLightEmission.end()) {
            const float3& unhandledEmission = unhandledIt->second.first;
            mat.emission = PackedFloat3{unhandledEmission.x, unhandledEmission.y, unhandledEmission.z};
            mat.lightId = -1;
            mat.twoSided = unhandledIt->second.second ? 1u : 0u;
        }
        materials.push_back(mat);
    }
}

// --- Spheres ---------------------------------------------------------
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
        if (s.areaLight >= 0 && s.areaLight < (int)scene.areaLights.size()) {
            // Same "emissive, but not NEE-registered" tier loadPbrtDisks()
            // just above already established for a disk-shaped
            // AreaLightSource (that function's own comment) - this loader
            // has no sphere-shaped analytic light in its own `lights[]`
            // NEE list either, so a sphere area light is visible (direct
            // hit or a BSDF-sampled bounce landing on it) but not
            // explicitly sampled. `sphereMaterials` is plain
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
            mat.emission = PackedFloat3{(float)(em.L[0] * em.scale), (float)(em.L[1] * em.scale), (float)(em.L[2] * em.scale)};
            mat.lightId = -1;
            mat.twoSided = em.twoSided ? 1u : 0u;
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
    for (const pbrt_flatten::Disk& d : scene.disks) {
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
            mat.emission = PackedFloat3{(float)(em.L[0] * em.scale), (float)(em.L[1] * em.scale), (float)(em.L[2] * em.scale)};
            mat.lightId = -1;
            mat.twoSided = em.twoSided ? 1u : 0u;
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
    for (const pbrt_flatten::Cylinder& cy : scene.cylinders) {
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

        TriangleMaterial mat = materialFor(cy.material);
        if (cy.areaLight >= 0 && cy.areaLight < (int)scene.areaLights.size()) {
            // Same "emissive, but not NEE-registered" tier loadPbrtDisks()
            // just above already established - visible (direct hit or a
            // BSDF-sampled bounce) but not explicitly sampled.
            const pbrt_flatten::Emission& em = scene.areaLights[cy.areaLight];
            mat.emission = PackedFloat3{(float)(em.L[0] * em.scale), (float)(em.L[1] * em.scale), (float)(em.L[2] * em.scale)};
            mat.lightId = -1;
            mat.twoSided = em.twoSided ? 1u : 0u;
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
                            (float)pl.scale,
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
// Deliberately miss-path-only for BOTH the constant-colour and
// image cases (no NEE/MIS light-sampling strategy) - see
// metal_poc.metal's own mirrored comments on why that's accepted
// scope, not an oversight; a future NEE upgrade already has a
// tested building block waiting (metal_poc_host_math.h's own
// float-RGB buildEnvDistribution2D() overload, added but not yet
// wired to anything - the exact same "phase 1 before phase 2"
// staging earthTexture's own NEE support went through, sections
// 69/71).
void MetalPocApp::loadPbrtInfiniteLight(const pbrt_flatten::FlatScene& scene) {
    if (scene.infiniteLight.present) {
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
    // Disks are handled above (section 101, common full-circle case);
    // cylinders are handled above too now (section 171, common full-tube
    // case, via loadPbrtCylinders()); cone/paraboloid/bilinearmesh/curve
    // remain a real gap.
    if (!scene.cones.empty() || !scene.paraboloids.empty() ||
        !scene.bilinearPatches.empty() || !scene.curves.empty())
        fprintf(stderr, "loadPbrtScene: cone/paraboloid/bilinearmesh/curve shapes skipped - "
                        "only triangle mesh, sphere, (full-circle) disk, and (full-tube) cylinder "
                        "shapes are supported by this POC's scene loader yet\n");
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
