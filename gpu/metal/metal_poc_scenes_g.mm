// metal_poc_scenes_g.mm
// Category G (Models) hand-authored scene builders -
// MetalPocApp::buildXxx() out-of-class definitions, split out of
// metal_poc.mm as a pure code-motion refactor (no behaviour change) -
// see metal_poc_app.h's own header comment and docs/METAL_GPU_FEASIBILITY.md.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_poc_app.h"

void MetalPocApp::buildMeshGalleryScene(const std::string& objFilename, float3 meshColor,
        uint32_t meshMaterialType, float meshRoughness, float3 meshConductorEta,
        float3 meshConductorK, float meshTargetSize, float meshIor, bool flipXZ) {
    // Same "push well clear of the hardcoded POC room's own [-1,1] region"
    // convention loadPbrtScene()/buildCornellBoxA1() both already use (see
    // either one's own comment) - a real, previously-shipped bug found
    // here first (before it ever reached review): this scene's own
    // geometry is authored directly at the app's own working scale (no
    // rescale needed, unlike A1's 555-unit Cornell box), but was left
    // OVERLAPPING the hardcoded room's own already-occupied space, not
    // offset clear of it - the first render looked like garbled noise, not
    // a recognizable mesh, because it genuinely WAS two unrelated scenes'
    // geometry interleaved in the same few world-space units.
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};

    // Ground: a large flat quad (not OptiX's own huge checker SPHERE - see
    // buildMeshGalleryScene()'s own declaration comment for why), light
    // grey diffuse, centred under the mesh.
    const float3 groundColor{0.5f, 0.5f, 0.5f};
    addQuad(verts, normals, uvs, materials,
            float3{-2.0f, 0.0f, -2.0f} + sceneOffset, float3{2.0f, 0.0f, -2.0f} + sceneOffset,
            float3{2.0f, 0.0f, 2.0f} + sceneOffset, float3{-2.0f, 0.0f, 2.0f} + sceneOffset, groundColor);

    // The mesh itself, auto-fit to meshTargetSize and recentred at
    // sceneOffset (loadObjMesh()'s own `center` parameter IS the mesh's
    // real world-space placement, not a separate local-then-place step -
    // see its own Suzanne/Spot call sites) - loadObjMesh()'s own existing
    // Suzanne/Spot convention otherwise (section 110's executableDir()-
    // first path resolution applies automatically, no new lookup code
    // needed here).
    NSString* modelsDir = nil;
    {
        NSString* exeDir = executableDir();
        NSString* candidate = [exeDir stringByAppendingPathComponent:@"models"];
        if (exeDir && [[NSFileManager defaultManager] fileExistsAtPath:
                [candidate stringByAppendingPathComponent:@(objFilename.c_str())]]) {
            modelsDir = candidate;
        }
    }
    if (!modelsDir) {
#ifdef RT_MODELS_DIR
        modelsDir = @(RT_MODELS_DIR);
#else
        modelsDir = [[@(__FILE__) stringByDeletingLastPathComponent]
            stringByAppendingPathComponent:@"../../models"];
#endif
    }
    NSString* meshPath = [modelsDir stringByAppendingPathComponent:@(objFilename.c_str())];
    if (!loadObjMesh(meshPath.UTF8String, verts, normals, uvs, materials, meshColor,
                      /*center=*/float3{0.0f, 0.45f, 0.0f} + sceneOffset, meshTargetSize, meshMaterialType,
                      meshRoughness, meshConductorEta, meshConductorK, meshIor, flipXZ)) {
        fprintf(stderr, "buildMeshGalleryScene: continuing without '%s' - check RT_MODELS_DIR / "
                        "models/%s.\n", objFilename.c_str(), objFilename.c_str());
    }

    // A small quad area light above the mesh (this loader's only NEE-
    // sampled light shape - see buildMeshGalleryScene()'s own declaration
    // comment for why not a sphere light like OptiX's own).
    {
        const float3 a = float3{-0.4f, 1.6f, -0.4f} + sceneOffset, b = float3{0.4f, 1.6f, -0.4f} + sceneOffset,
                     c = float3{0.4f, 1.6f, 0.4f} + sceneOffset, d = float3{-0.4f, 1.6f, 0.4f} + sceneOffset;
        const float3 lightColor{6.0f, 6.0f, 6.0f};
        const int32_t lightId = (int32_t)lights.size();
        addQuad(verts, normals, uvs, materials, a, b, c, d, lightColor,
                /*materialType=*/0u, /*emission=*/lightColor, lightId);
        const float3 edgeU = b - a, edgeV = d - a;
        const float3 normalV = simd::normalize(simd::cross(edgeU, edgeV));
        const float area = simd::length(simd::cross(edgeU, edgeV));
        const float3 center = a + 0.5f * edgeU + 0.5f * edgeV;
        lights.push_back(AreaLightData{
            PackedFloat3{center.x, center.y, center.z},
            PackedFloat3{edgeU.x, edgeU.y, edgeU.z},
            PackedFloat3{edgeV.x, edgeV.y, edgeV.z},
            PackedFloat3{normalV.x, normalV.y, normalV.z},
            area, PackedFloat3{lightColor.x, lightColor.y, lightColor.z},
            /*patternTileB=*/0.0f, /*patternScale=*/0.0f,
            /*twoSided=*/1.0f, /*useTexture=*/0.0f});
    }

    // Camera: a simple 3/4 elevated view of the mesh, close to OptiX's own
    // apply_mesh_camera() framing in spirit (not the exact literal, which
    // barely matters - see buildCornellBoxA1()'s own comment on why the
    // fallback camera here is immediately overridden in every real
    // invocation anyway). Same sceneOffset as every other element above -
    // lookfrom/lookat both need it too, or the camera would end up
    // pointed at the hardcoded room's own empty space instead of this
    // scene's own geometry.
    const float3 lookfrom = float3{0.0f, 0.9f, 2.2f} + sceneOffset;
    const float3 lookat = float3{0.0f, 0.45f, 0.0f} + sceneOffset;
    const float3 up{0.0f, 1.0f, 0.0f};
    const float3 forward = simd::normalize(lookat - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    pbrtTanHalfFov = tanf(0.5f * 35.0f * (float)M_PI / 180.0f);
    havePbrtCamera = true;
    pbrtCameraLookAtWorld = lookat;
    pbrtCameraUpRaw = up;
    // No rescale (unlike buildCornellBoxA1()'s own 555-unit Cornell box -
    // this scene is already authored directly at the app's own working
    // scale) - applyCameraOverride()'s own transform (bboxCenter=0,
    // sceneScale=1, then + sceneOffset) reduces to exactly this same
    // fixed translation for any later --cam-x/y/z override too.
    pbrtBboxCenter = float3{0.0f, 0.0f, 0.0f};
    pbrtSceneScale = 1.0f;
    pbrtSceneOffset = sceneOffset;
}


// Scene G1: Stanford Bunny (69,451 triangles), polished bronze - matches
// gpu/optix/scene_builder_mesh_gallery.h's own build_stanford_bunny_gpu()
// material (bronze albedo (0.71,0.43,0.20), roughness 0.15) exactly.
void MetalPocApp::buildStanfordBunny() {
    const float3 bronze{0.71f, 0.43f, 0.20f};
    buildMeshGalleryScene("stanford-bunny.obj", bronze, /*materialType=*/4u,
        /*roughness=*/0.15f, /*eta=*/float3{1, 1, 1}, reflectanceToConductorK(bronze),
        /*targetSize=*/1.1f);
}

// Scene G2: Stanford Armadillo (99,976 triangles), gunmetal - matches
// build_stanford_armadillo_gpu()'s own material (albedo (0.55,0.56,0.58),
// roughness 0.08) exactly.
void MetalPocApp::buildStanfordArmadillo() {
    const float3 gunmetal{0.55f, 0.56f, 0.58f};
    buildMeshGalleryScene("armadillo.obj", gunmetal, /*materialType=*/4u,
        /*roughness=*/0.08f, /*eta=*/float3{1, 1, 1}, reflectanceToConductorK(gunmetal),
        /*targetSize=*/1.1f);
}

// Scene G3: Stanford Happy Buddha (98,601 triangles), polished gold -
// matches build_stanford_happy_buddha_gpu()'s own material (albedo
// (0.83,0.69,0.22), roughness 0.05) exactly.
void MetalPocApp::buildStanfordHappyBuddha() {
    const float3 gold{0.83f, 0.69f, 0.22f};
    buildMeshGalleryScene("happy-buddha.obj", gold, /*materialType=*/4u,
        /*roughness=*/0.05f, /*eta=*/float3{1, 1, 1}, reflectanceToConductorK(gold),
        /*targetSize=*/1.1f);
}

// Scenes G4-G24 below (minus G7/G10/G12/G13 - see this struct's own
// declaration comment for why those are deferred) - each matches
// gpu/optix/scene_builder_mesh_gallery.h's own real per-scene albedo/
// roughness exactly, same reflectanceToConductorK() conversion G1-G3
// already use.

// G4: Stanford Lucy (99,970 triangles), bright silver.
void MetalPocApp::buildStanfordLucy() {
    const float3 silver{0.85f, 0.85f, 0.88f};
    buildMeshGalleryScene("lucy.obj", silver, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(silver), 2.5f);
}

// G5: Stanford XYZRGB Dragon (249,882 triangles), bright silver.
void MetalPocApp::buildStanfordDragon() {
    const float3 silver{0.85f, 0.85f, 0.88f};
    buildMeshGalleryScene("xyzrgb_dragon.obj", silver, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(silver), 2.5f);
}

// G6: Utah Teapot (6,320 triangles), bright silver.
void MetalPocApp::buildUtahTeapot() {
    const float3 silver{0.85f, 0.85f, 0.88f};
    buildMeshGalleryScene("teapot.obj", silver, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(silver), 2.5f);
}

// G8: Suzanne (968 triangles, fan-triangulated from 500 mostly-quad
// faces), bright silver - a DIFFERENT material from the hardcoded POC
// room's own bronze Suzanne (buildScene()'s own materialType 0), same
// mesh file, genuinely separate scene/context.
void MetalPocApp::buildSuzanneGallery() {
    const float3 silver{0.85f, 0.85f, 0.88f};
    buildMeshGalleryScene("suzanne.obj", silver, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(silver), 1.1f);
}

// G9: Nefertiti Bust (99,938 triangles), bright silver.
void MetalPocApp::buildNefertiti() {
    const float3 silver{0.85f, 0.85f, 0.88f};
    buildMeshGalleryScene("nefertiti.obj", silver, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(silver), 2.5f);
}

// G11: Cheburashka bust (13,334 triangles), bright silver.
void MetalPocApp::buildCheburashka() {
    const float3 silver{0.85f, 0.85f, 0.88f};
    buildMeshGalleryScene("cheburashka.obj", silver, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(silver), 2.2f);
}

// G14: Beast, bronze (same tone as G1's bunny).
void MetalPocApp::buildBeast() {
    const float3 bronze{0.71f, 0.43f, 0.20f};
    buildMeshGalleryScene("beast.obj", bronze, 4u, 0.15f, float3{1, 1, 1},
        reflectanceToConductorK(bronze), 1.1f);
}

// G15: VW Beetle, bright silver.
void MetalPocApp::buildVWBeetle() {
    const float3 silver{0.85f, 0.85f, 0.88f};
    buildMeshGalleryScene("beetle.obj", silver, 4u, 0.08f, float3{1, 1, 1},
        reflectanceToConductorK(silver), 2.5f);
}

// G17: Bimba, gold (same tone as G3's buddha).
void MetalPocApp::buildBimba() {
    const float3 gold{0.83f, 0.69f, 0.22f};
    buildMeshGalleryScene("bimba.obj", gold, 4u, 0.05f, float3{1, 1, 1},
        reflectanceToConductorK(gold), 1.1f);
}

// G18: Cow, warm brass tone.
void MetalPocApp::buildCowGallery() {
    const float3 brass{0.80f, 0.65f, 0.28f};
    buildMeshGalleryScene("cow.obj", brass, 4u, 0.15f, float3{1, 1, 1},
        reflectanceToConductorK(brass), 1.1f);
}

// G19: Fandisk, gunmetal.
void MetalPocApp::buildFandisk() {
    const float3 gunmetal{0.55f, 0.56f, 0.58f};
    buildMeshGalleryScene("fandisk.obj", gunmetal, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(gunmetal), 2.5f);
}

// G20: Homer, warm gold tone.
void MetalPocApp::buildHomer() {
    const float3 gold{0.85f, 0.70f, 0.25f};
    buildMeshGalleryScene("homer.obj", gold, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(gold), 1.1f);
}

// G21: Igea, bright silver.
void MetalPocApp::buildIgea() {
    const float3 silver{0.85f, 0.85f, 0.88f};
    buildMeshGalleryScene("igea.obj", silver, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(silver), 1.1f);
}

// G22: Max Planck bust, warm copper tone.
void MetalPocApp::buildMaxPlanck() {
    const float3 copper{0.65f, 0.45f, 0.30f};
    buildMeshGalleryScene("max-planck.obj", copper, 4u, 0.2f, float3{1, 1, 1},
        reflectanceToConductorK(copper), 1.1f);
}

// G23: Ogre, muted green-tinted metal.
void MetalPocApp::buildOgre() {
    const float3 tint{0.45f, 0.50f, 0.35f};
    buildMeshGalleryScene("ogre.obj", tint, 4u, 0.2f, float3{1, 1, 1},
        reflectanceToConductorK(tint), 1.1f);
}

// G24: Rocker Arm, gunmetal.
void MetalPocApp::buildRockerArm() {
    const float3 gunmetal{0.55f, 0.56f, 0.58f};
    buildMeshGalleryScene("rocker-arm.obj", gunmetal, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(gunmetal), 2.5f);
}

// G7: Spot the Cow (Keenan Crane), bright silver - flipXZ=true, matching
// OptiX's own build_spot_cow_gpu() comment ("the raw mesh faces away
// from the camera").
void MetalPocApp::buildSpotCow() {
    const float3 silver{0.85f, 0.85f, 0.88f};
    buildMeshGalleryScene("spot.obj", silver, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(silver), 1.3f, /*ior=*/1.0f, /*flipXZ=*/true);
}

// G10: Horse (classic geometry-processing test model), bright silver -
// flipXZ=true, matching OptiX's own build_horse_gpu() comment.
void MetalPocApp::buildHorse() {
    const float3 silver{0.85f, 0.85f, 0.88f};
    buildMeshGalleryScene("horse.obj", silver, 4u, 0.1f, float3{1, 1, 1},
        reflectanceToConductorK(silver), 2.2f, /*ior=*/1.0f, /*flipXZ=*/true);
}

// G13: Glass Dragon - same mesh/scale as G5's metal dragon, clear glass
// (materialType 2, ior 1.5) instead of a conductor - matches OptiX's own
// build_glass_dragon_gpu(). meshColor/roughness/eta/k are all ignored for
// materialType 2 (see loadObjMesh()'s own TriangleMaterial-construction
// comment) - passed as harmless placeholders.
void MetalPocApp::buildGlassDragon() {
    buildMeshGalleryScene("xyzrgb_dragon.obj", float3{1, 1, 1}, /*materialType=*/2u,
        /*roughness=*/0.0f, float3{1, 1, 1}, float3{0, 0, 0}, /*targetSize=*/2.5f,
        /*ior=*/1.5f);
}

// G12: Trophy Room - the last category-G scene, four already-verified
// meshes (bunny/teapot/Suzanne/Spot the Cow) lined up on one shared shelf
// in bronze/chrome/gold/gunmetal, matching CPU's build_trophy_room()/
// OptiX's own build_trophy_room_gpu() in mesh choice and material tones -
// NOT their exact numeric scale/offset, which are tuned for OptiX's own
// raw-multiply-then-translate placement convention. This loader's
// loadObjMesh() instead auto-fits each mesh to a caller-chosen targetSize
// around a caller-chosen world-space centre (see buildMeshGalleryScene()'s
// own declaration comment) - a different enough convention that porting
// OptiX's literal numbers would not reproduce the same layout. The first
// hand-authored Metal scene to place multiple external OBJ meshes in one
// composition - genuinely a different shape from buildMeshGalleryScene()
// (single mesh only), so this is a bespoke builder, not a call to it.
// Section 120, docs/METAL_GPU_FEASIBILITY.md.
void MetalPocApp::buildTrophyRoom() {
    const float3 sceneOffset{60.0f, 0.0f, 0.0f};

    // Ground: one large flat quad wide enough for all four meshes (same
    // "flat quad, not a checker sphere" simplification as
    // buildMeshGalleryScene() - see its own declaration comment for why).
    const float3 groundColor{0.5f, 0.5f, 0.5f};
    addQuad(verts, normals, uvs, materials,
            float3{-5.0f, 0.0f, -2.5f} + sceneOffset, float3{5.0f, 0.0f, -2.5f} + sceneOffset,
            float3{5.0f, 0.0f, 2.5f} + sceneOffset, float3{-5.0f, 0.0f, 2.5f} + sceneOffset, groundColor);

    NSString* modelsDir = nil;
    {
        NSString* exeDir = executableDir();
        NSString* candidate = [exeDir stringByAppendingPathComponent:@"models"];
        if (exeDir && [[NSFileManager defaultManager] fileExistsAtPath:
                [candidate stringByAppendingPathComponent:@"stanford-bunny.obj"]]) {
            modelsDir = candidate;
        }
    }
    if (!modelsDir) {
#ifdef RT_MODELS_DIR
        modelsDir = @(RT_MODELS_DIR);
#else
        modelsDir = [[@(__FILE__) stringByDeletingLastPathComponent]
            stringByAppendingPathComponent:@"../../models"];
#endif
    }

    // Same bronze/chrome/gold/gunmetal tones as CPU's/OptiX's own trophy
    // room, spaced 2.4 units apart along the shelf's own x-axis, each
    // auto-fit to a size that keeps it clear of its neighbours.
    const float3 bronze{0.71f, 0.43f, 0.20f};
    const float3 chrome{0.85f, 0.85f, 0.88f};
    const float3 gold{0.83f, 0.69f, 0.22f};
    const float3 gunmetal{0.55f, 0.56f, 0.58f};
    const float3 meshCenterY{0.0f, 0.45f, 0.0f};

    NSString* bunnyPath = [modelsDir stringByAppendingPathComponent:@"stanford-bunny.obj"];
    if (!loadObjMesh(bunnyPath.UTF8String, verts, normals, uvs, materials, bronze,
                      meshCenterY + float3{-3.6f, 0.0f, 0.0f} + sceneOffset, /*targetSize=*/1.1f,
                      /*materialType=*/4u, /*roughness=*/0.15f, float3{1, 1, 1},
                      reflectanceToConductorK(bronze))) {
        fprintf(stderr, "buildTrophyRoom: continuing without stanford-bunny.obj.\n");
    }

    NSString* teapotPath = [modelsDir stringByAppendingPathComponent:@"teapot.obj"];
    if (!loadObjMesh(teapotPath.UTF8String, verts, normals, uvs, materials, chrome,
                      meshCenterY + float3{-1.2f, 0.0f, 0.0f} + sceneOffset, /*targetSize=*/1.4f,
                      /*materialType=*/4u, /*roughness=*/0.10f, float3{1, 1, 1},
                      reflectanceToConductorK(chrome))) {
        fprintf(stderr, "buildTrophyRoom: continuing without teapot.obj.\n");
    }

    NSString* suzannePath = [modelsDir stringByAppendingPathComponent:@"suzanne.obj"];
    if (!loadObjMesh(suzannePath.UTF8String, verts, normals, uvs, materials, gold,
                      meshCenterY + float3{1.2f, 0.0f, 0.0f} + sceneOffset, /*targetSize=*/1.1f,
                      /*materialType=*/4u, /*roughness=*/0.05f, float3{1, 1, 1},
                      reflectanceToConductorK(gold))) {
        fprintf(stderr, "buildTrophyRoom: continuing without suzanne.obj.\n");
    }

    // Spot the Cow: flipXZ=true, the same "raw mesh faces away from this
    // app's own camera" fix G7's own solo scene needed (section 119).
    NSString* spotPath = [modelsDir stringByAppendingPathComponent:@"spot.obj"];
    if (!loadObjMesh(spotPath.UTF8String, verts, normals, uvs, materials, gunmetal,
                      meshCenterY + float3{3.6f, 0.0f, 0.0f} + sceneOffset, /*targetSize=*/1.3f,
                      /*materialType=*/4u, /*roughness=*/0.08f, float3{1, 1, 1},
                      reflectanceToConductorK(gunmetal), /*ior=*/1.0f, /*flipXZ=*/true)) {
        fprintf(stderr, "buildTrophyRoom: continuing without spot.obj.\n");
    }

    // A wide quad area light spanning the whole shelf (this loader's only
    // NEE-sampled light shape, not OptiX's own sphere light - see
    // buildMeshGalleryScene()'s own declaration comment for why).
    {
        const float3 a = float3{-4.0f, 3.0f, -1.0f} + sceneOffset, b = float3{4.0f, 3.0f, -1.0f} + sceneOffset,
                     c = float3{4.0f, 3.0f, 1.0f} + sceneOffset, d = float3{-4.0f, 3.0f, 1.0f} + sceneOffset;
        const float3 lightColor{6.0f, 6.0f, 6.0f};
        const int32_t lightId = (int32_t)lights.size();
        addQuad(verts, normals, uvs, materials, a, b, c, d, lightColor,
                /*materialType=*/0u, /*emission=*/lightColor, lightId);
        const float3 edgeU = b - a, edgeV = d - a;
        const float3 normalV = simd::normalize(simd::cross(edgeU, edgeV));
        const float area = simd::length(simd::cross(edgeU, edgeV));
        const float3 center = a + 0.5f * edgeU + 0.5f * edgeV;
        lights.push_back(AreaLightData{
            PackedFloat3{center.x, center.y, center.z},
            PackedFloat3{edgeU.x, edgeU.y, edgeU.z},
            PackedFloat3{edgeV.x, edgeV.y, edgeV.z},
            PackedFloat3{normalV.x, normalV.y, normalV.z},
            area, PackedFloat3{lightColor.x, lightColor.y, lightColor.z},
            /*patternTileB=*/0.0f, /*patternScale=*/0.0f,
            /*twoSided=*/1.0f, /*useTexture=*/0.0f});
    }

    // Camera: wide enough to frame all four meshes across the shelf's own
    // ~7.2-unit spread, the same simple 3/4-elevated-view convention as
    // buildMeshGalleryScene()'s own fallback camera - not a literal port of
    // the CPU registry's own G12 camera row (vfov 34, lookfrom (0,2.3,14)),
    // which is tuned for OptiX's own raw-scale placement, not this
    // targetSize-based one (see this function's own declaration comment).
    const float3 lookfrom = float3{0.0f, 1.8f, 8.0f} + sceneOffset;
    const float3 lookat = float3{0.0f, 0.6f, 0.0f} + sceneOffset;
    const float3 up{0.0f, 1.0f, 0.0f};
    const float3 forward = simd::normalize(lookat - lookfrom);
    const float3 right = simd::normalize(simd::cross(forward, up));
    const float3 trueUp = simd::cross(right, forward);
    pbrtCameraPos = lookfrom;
    pbrtCameraForward = forward;
    pbrtCameraRight = right;
    pbrtCameraUp = trueUp;
    pbrtTanHalfFov = tanf(0.5f * 55.0f * (float)M_PI / 180.0f);
    havePbrtCamera = true;
    pbrtCameraLookAtWorld = lookat;
    pbrtCameraUpRaw = up;
    pbrtBboxCenter = float3{0.0f, 0.0f, 0.0f};
    pbrtSceneScale = 1.0f;
    pbrtSceneOffset = sceneOffset;
}


