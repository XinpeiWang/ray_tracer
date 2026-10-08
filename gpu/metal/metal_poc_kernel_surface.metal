// metal_poc_kernel_surface.metal
// What primaryRayKernel does at the end of a ray: the sky or environment when it escapes, and on a surface hit the geometry and normal, the
// shading-normal perturbations (bump, normal map), the albedo, and the emission. Pure code motion out of the old kernel body; see
// metal_poc_kernel_state.metal.

// The ray escaped the scene: adds the environment map, the pbrt infinite light, a constant environment colour or the hand-authored room's sky,
// with the MIS weight against light sampling. The path ends after this.
inline void shadeEscapedRay(thread const KernelRes& R, thread PathState& P, thread BounceState& B) {
    KERNEL_RES_ALIASES(R)
    PATH_STATE_ALIASES(P)
    BOUNCE_STATE_ALIASES(B)
    const float3 skyTop = float3(0.9, 0.95, 1.0);
    const float3 skyBottom = float3(0.3, 0.5, 0.9);
        CENSUS_PUSH(255u);
        if (uniforms.useEnvironmentMap != 0u) {
            float2 envUV = envMapUV(normalize(rayDir));
            float3 envColor = earthTexture.sample(textureSampler, envUV).rgb;
            // MIS weight against the environment-light NEE
            // strategy's own pdf for this EXACT escaping
            // direction (section 71) - this ray was BSDF-
            // sampled, not env-light-sampled, so `bsdfPdf`
            // (recorded by whichever material's own shading
            // function ran last bounce) is the competing
            // strategy's own pdf here, same "check for a hit
            // reachable two ways" MIS shape the area light's own
            // direct-hit weight below already uses. `specularBounce`
            // (a delta/specular material has no NEE strategy to
            // weight against at all) or `envMapWidth == 0` (no
            // NEE strategy exists to double-count against in the
            // first place) both mean full weight, unweighted -
            // exactly the same two escape hatches the area
            // light's own weight below already has.
            float envMissWeight = 1.0;
            if (!specularBounce && uniforms.envMapWidth > 0u) {
                float pdfEnv = pdfEnvironmentDirection(envMarginalCDF, envConditionalCDF,
                                                        int(uniforms.envMapWidth), int(uniforms.envMapHeight),
                                                        normalize(rayDir));
                envMissWeight = (bsdfPdf * bsdfPdf) / (bsdfPdf * bsdfPdf + pdfEnv * pdfEnv);
            }
            radiance += throughput * envColor * envMissWeight;
        } else if (uniforms.pbrtHasImageEnvLight != 0u) {
            // A pbrt-loaded scene's own image-based infinite
            // light. Now MIS-weighted against the NEE strategy
            // added in section 96 (mirrors the useEnvironmentMap
            // arm above exactly) - every material's own shading
            // function that does NEE now also samples
            // pbrtEnvTexture directly via pbrtEnvMapWidth, so a
            // BSDF-sampled ray escaping toward it needs the same
            // double-count protection.
            float3 pbrtEnvColorSample = pbrtEnvLeAt(uniforms, pbrtEnvConditionalCDF, pbrtEnvTexture, textureSampler, rayOrigin, normalize(rayDir));
            float pbrtEnvMissWeight = 1.0;
            if (!specularBounce && uniforms.pbrtEnvMapWidth > 0u) {
                float pdfPbrtEnv = pbrtEnvPdfAt(uniforms, pbrtEnvMarginalCDF, pbrtEnvConditionalCDF,
                                                uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                                                rayOrigin, normalize(rayDir));
                pbrtEnvMissWeight = (bsdfPdf * bsdfPdf) / (bsdfPdf * bsdfPdf + pdfPbrtEnv * pdfPbrtEnv);
            }
            radiance += throughput * pbrtEnvColorSample * pbrtEnvMissWeight;
        } else if (uniforms.pbrtHasConstantEnvLight != 0u) {
            // A pbrt-loaded scene's own constant-colour
            // LightSource "infinite" (metal_poc.mm's own
            // loadPbrtScene() comment) - deliberately no MIS
            // weight (weight 1.0, same as the specularBounce/
            // envMapWidth==0 escape hatches just above): there is
            // no NEE strategy for this light to double-count
            // against, since none of this shader's material-
            // shading functions sample it explicitly yet.
            radiance += throughput * float3(uniforms.pbrtEnvColor);
        } else if (uniforms.isPbrtScene == 0u) {
            float skyT = 0.5 * (rayDir.y + 1.0);
            radiance += throughput * mix(skyBottom, skyTop, skyT);
        }
        // else: a real pbrt scene with no infinite light of any
        // kind - true black (no radiance added at all), matching
        // real pbrt-v4's own behaviour (Uniforms::isPbrtScene's
        // own comment).
}

// A surface was hit: finds which primitive kind (triangle, sphere, disk, cylinder, instanced mesh), its geometric normal and its material, and
// whether the ray hit the front face.
inline void resolveHit(thread const KernelRes& R, thread PathState& P, thread BounceState& B) {
    KERNEL_RES_ALIASES(R)
    PATH_STATE_ALIASES(P)
    BOUNCE_STATE_ALIASES(B)
    primId = result.primitive_id;
    hitPoint = rayOrigin + rayDir * result.distance;

    // Geometric normal: derived from the triangle's own vertices
    // for a triangle hit (as before), or from the sphere's centre
    // for a bounding-box hit - a sphere has no "vertices" to pull
    // a face normal from, but (hitPoint - centre) is exact for a
    // perfect sphere, no approximation.
    //
    // Spheres, the disk, AND the cylinder all report
    // intersection_type::bounding_box (none is a hardware-native
    // triangle) - they only stop being ambiguous once
    // `geometry_id` is checked too: sphereAS's own
    // geometryDescriptors array has the spheres' bounding-box
    // geometry at index 0, the disk's at index 1, and the
    // cylinder's at index 2 (metal_poc.mm's own
    // sphereAccelDesc.geometryDescriptors, section 171), and
    // geometry_id reports exactly that array index for a
    // bounding-box hit.
    isBoundingBox = (result.type == intersection_type::bounding_box);
    isDisk = isBoundingBox && (result.geometry_id == 1u);
    isCylinder = isBoundingBox && (result.geometry_id == 2u);
    isSphere = isBoundingBox && !isDisk && !isCylinder;
    // Suzanne is instanced TWICE (instance_id 2 and 3, matching
    // metal_poc.mm's own instanceDescs[] ordering - see that
    // file's addTransformedSuzanneInstance()) from the SAME
    // object-space geometry/AS - the one thing in this scene that
    // actually exercises a non-identity instance transform; every
    // other instance (the combined room+Spot geometry, the
    // sphere primitives) still uses the identity transform this
    // POC's very first version already had. Hardcoding the
    // instance_id threshold here (rather than deriving it) is
    // the same "explicitly documented, scene-specific constant"
    // approach this POC already uses for its light geometry.
    // pbrt ObjectInstance placements are instances numbered from uniforms.pbrtInstanceFirst on (see Uniforms::pbrtInstanceFirst); Suzanne's
    // are the ones between 2 and that. A pbrt placement's primitive id is local to its group's own acceleration structure.
    isTriangleHit = !isSphere && !isDisk && !isCylinder;
    isPbrtInstance = isTriangleHit && (result.instance_id >= uniforms.pbrtInstanceFirst);
    isSuzanneInstance = isTriangleHit && !isPbrtInstance && (result.instance_id >= 2u);
    if (isSphere) {
        SphereData sphere = spheres[primId];
        // Same interpolated centre sphereIntersectionFunction
        // itself tested the ray against (F11, section 167) - using
        // the STATIC sphere.center here instead would put the
        // shading normal off-centre from the surface point this
        // ray actually landed on for any moving sphere, a subtle
        // but real correctness bug distinct from the intersection
        // test itself.
        float3 center = float3(sphere.center) + shutterT * float3(sphere.centerDelta1);
        normal = normalize(hitPoint - center);
        mat = sphereMaterials[primId];
    } else if (isDisk) {
        // Flat and planar - the disk's own stored normal IS the
        // shading normal directly, no per-hit computation needed
        // (unlike a sphere's hitPoint-relative one or a triangle's
        // barycentric-interpolated one).
        DiskData disk = disks[primId];
        normal = float3(disk.normal);
        // NOT diskMaterials[0] - a real pre-existing bug (harmless
        // until now, since the hardcoded room ever only had ONE
        // disk, making [0] and [primId] the same value by
        // coincidence) found while adding real pbrt-loaded disks
        // (section 101): a second disk's own material was
        // silently ignored, every disk hit reading the room's own
        // disk material instead.
        mat = diskMaterials[primId];
    } else if (isCylinder) {
        // Radial direction perpendicular to the axis at the hit
        // point - project the hit onto the axis line to find the
        // nearest axis point, then point away from it. Exact for
        // a true cylinder, same "no approximation needed" note
        // as the sphere's own hitPoint-relative normal above.
        CylinderData cyl = cylinders[primId];
        float3 axis = float3(cyl.axis);
        float3 base = float3(cyl.base);
        float3 axisPoint = base + dot(hitPoint - base, axis) * axis;
        normal = normalize(hitPoint - axisPoint);
        mat = cylinderMaterials[primId];
    } else if (isPbrtInstance) {
        // The group's triangles live once in the shared arrays, at triBase + the group-local primitive id; from here on `primId` is that
        // global index, so every triangle-indexed lookup below (uvs, vertices, triMaterials) works unchanged in OBJECT space. Only the
        // normal needs the instance's own transform (its normal matrix), as for Suzanne.
        const InstanceTransform pbrtXf = instanceTransforms[result.instance_id];
        primId = pbrtXf.triBase + result.primitive_id;
        normal = transformNormalByInstance(shadingNormalFor(primId, result.triangle_barycentric_coord, normals, vertices), pbrtXf);
        mat = triMaterials[primId];
    } else if (isSuzanneInstance) {
        // Object-space normal (Suzanne's own per-vertex data,
        // just like the non-instanced case below) transformed
        // into world space by THIS hit's own instance transform -
        // the one piece of shading math instancing actually adds
        // over the room/Spot geometry's own single-identity-
        // instance path.
        float3 objectNormal = shadingNormalForNoFacet(primId, result.triangle_barycentric_coord, suzanneNormals);
        normal = transformNormalByInstance(objectNormal, instanceTransforms[result.instance_id]);
        mat = suzanneMaterials[primId];
    } else {
        normal = shadingNormalFor(primId, result.triangle_barycentric_coord, normals, vertices);
        mat = triMaterials[primId];
    }
    // Raw (outward, unflipped) normal kept separately from here -
    // dielectric handling below needs to know which side of the
    // surface the ray is entering from (front vs back face) to
    // pick the right eta ratio, information the flipped-to-face-
    // the-ray version used by every other material below discards.
    frontFace = dot(normal, rayDir) < 0.0;
    facingNormal = frontFace ? normal : -normal;
}

// Perturbs the shading normal (facingNormal) for the bump-mapped and normal-mapped materials. Also records the material in the census.
inline void perturbShadingNormal(thread const KernelRes& R, thread PathState& P, thread BounceState& B) {
    KERNEL_RES_ALIASES(R)
    PATH_STATE_ALIASES(P)
    BOUNCE_STATE_ALIASES(B)

    // materialType == 7 (procedurally bump-mapped Lambertian):
    // perturb ONLY the shading normal used by the BSDF/NEE math
    // below, never `normal`/the ray-offset direction above - the
    // textbook bump-mapping distinction between what LOOKS
    // perturbed (shading) and what stays geometrically flat (ray
    // origins, self-intersection avoidance). Guarded to the
    // primary triangle buffer, the only place tangentFor()'s own
    // primId-indexed vertex/uv lookup is valid (see materialType
    // 7's own comment above) - never true for a sphere/disk/
    // Suzanne-instance hit, so this is simply skipped for those.
    // Image bump map ("texture displacement"): perturbs ONLY the shading normal of a triangle hit, like materialType 7 below.
    if (mat.bumpWidth > 0 && !isBoundingBox && !isSuzanneInstance && !isPbrtInstance) {
        const float2 bumpUV = texCoordFor(primId, result.triangle_barycentric_coord, uvs);
        float3 bumpDpdu, bumpDpdv;
        triangleDpduDpdv(primId, vertices, uvs, facingNormal, bumpDpdu, bumpDpdv);
        if (mat.bumpIsNormalMap != 0) {
            facingNormal = imageNormalMapNormal(rgbGridData, mat.bumpOffset, mat.bumpWidth, mat.bumpHeight,
                                                bumpUV.x, bumpUV.y, facingNormal, bumpDpdu);
        } else {
            float footprint = 0.0;
            if (depth == 0u && uniforms.cameraOrthographic == 0u && uniforms.cameraSpherical == 0u &&
                uniforms.cameraRealistic == 0u && uniforms.hasCameraOrbitBlur == 0u) {
                footprint = bumpFootprintStep(uniforms, rayOrigin, rayDir, hitPoint, facingNormal, bumpDpdu, bumpDpdv);
            }
            facingNormal = imageBumpNormal(rgbGridData, mat.bumpOffset, mat.bumpWidth, mat.bumpHeight, mat.bumpScale,
                                           bumpUV.x, bumpUV.y, facingNormal, bumpDpdu, footprint);
        }
    }
    CENSUS_PUSH(mat.materialType);
    if (mat.materialType == METAL_MAT_BUMP_LAMBERTIAN && !isSphere && !isDisk && !isSuzanneInstance && !isPbrtInstance) {
        float2 bumpUV = texCoordFor(primId, result.triangle_barycentric_coord, uvs);
        float3 tangent = tangentFor(primId, vertices, uvs);
        facingNormal = proceduralBumpNormal(facingNormal, tangent, bumpUV, mat.roughness);
    } else if (mat.materialType == METAL_MAT_NORMAL_MAPPED_LAMBERTIAN) {
        // materialType 21 (checker-driven normal-mapped
        // Lambertian, B12's own sphere) - matches CPU's own
        // `normal_map_material` (a DIRECT tangent-space-normal
        // read, decoded from an RGB texture, not a finite-
        // difference displacement gradient the way materialType
        // 7's own bump map is). Only perturbs the SHADING
        // normal, same "shading vs geometric" split materialType
        // 7 already established. Reuses `checker3DColor()`'s own
        // 3D-spatial cell test directly on `hitPoint` (CPU's own
        // `checker_texture` is spatial, not UV-based, so this
        // needs no geometry-specific UV lookup at all - works
        // for this scene's sphere or any future triangle/disk
        // hit identically). The two decoded tangent-space
        // normals below are CPU's own two checker colours
        // (`build_normal_mapped_cornell()`'s own `norm_tex`)
        // pre-decoded by hand: `color(0.5,0.5,1.0)` -> `2*c-1` =
        // `(0,0,1)` (flat, no perturbation) and
        // `color(0.8,0.8,1.0)` -> `(0.6,0.6,1.0)` normalized (a
        // real diagonal tilt) - hardcoded rather than stored as
        // new material fields, since this checker's own two
        // colours are a fixed property of this one scene, not a
        // reusable general-purpose texture parameter.
        float3 tangent, bitangent;
        buildAnisotropicOnb(facingNormal, tangent, bitangent);
        float3 cell = floor(hitPoint / mat.roughness);
        float parity = fmod(abs(cell.x) + abs(cell.y) + abs(cell.z), 2.0);
        float3 nsLocal = (parity < 0.5) ? float3(0.0, 0.0, 1.0) : normalize(float3(0.6, 0.6, 1.0));
        float3 perturbed = normalize(nsLocal.x * tangent + nsLocal.y * bitangent + nsLocal.z * facingNormal);
        if (dot(perturbed, facingNormal) < 0.0) perturbed = -perturbed;
        facingNormal = perturbed;
    }
}

// The surface colour at the hit (B.albedo): flat colour, image, checker, noise and marble textures, the texture family. May also replace the
// material's transmit colour from its image.
inline void computeAlbedo(thread const KernelRes& R, thread PathState& P, thread BounceState& B) {
    KERNEL_RES_ALIASES(R)
    PATH_STATE_ALIASES(P)
    BOUNCE_STATE_ALIASES(B)

    // materialType == 3 (textured Lambertian): the hardcoded POC
    // room's own back wall (a triangle - texCoordFor()'s own
    // triangle-only inputs, primId/barycentric_coord, are valid
    // there) OR, since section 122, a hand-authored scene's own
    // textured SPHERE (A4 Earth) - texCoordFor() cannot run on a
    // sphere hit at all (no primId-indexed UV to look up), so
    // this reuses the exact technique materialType 9 already
    // established for exactly this situation (see that
    // material's own comment, just above its alphaX/alphaY
    // branch): equirectangularUV() on the hit's own (sphere-
    // centre-relative) normal gives a texture-space coordinate
    // for free, no real UV parameterization needed.
    if (mat.materialType == METAL_MAT_TEXTURED_LAMBERTIAN) {
        // isSphere: NOT a plain equirectangularUV(normal) call -
        // this project's own CPU get_sphere_uv() (sphere.h) uses
        // phi=atan2(-p.z,p.x)+pi, whereas equirectangularUV() uses
        // atan2(dir.z,dir.x) - algebraically or (since atan2 is
        // odd in its first argument), CPU's own u is exactly
        // equirectangularUV(x,y,-z).x, a longitude MIRROR of
        // equirectangularUV(normal).x, not merely a phase shift.
        // Latitude (v) already matches exactly with no
        // correction needed (both give v=0 at y=-1, v=1 at
        // y=+1) - checked algebraically, not assumed, after a
        // first version of this code rendered the correct
        // CONTINENTS-SHAPED but WRONG-LONGITUDE side of the
        // globe compared to a real --cpu render of the same
        // scene_id (caught by that comparison, not by eye alone -
        // an equirectangular texture looks equally "plausible"
        // from any longitude).
        float2 uv = isSphere ? equirectangularUV(float3(normal.x, normal.y, -normal.z))
                              : texCoordFor(primId, result.triangle_barycentric_coord, uvs);
        albedo = earthTexture.sample(textureSampler, uv).rgb;
    } else if (mat.materialType == METAL_MAT_CHECKER_LAMBERTIAN) {
        // Procedural checker (Lambertian, same BSDF/NEE code path
        // as 0/3 below - only where albedo comes from differs):
        // `color` is the tile-A colour; tile B is a fixed
        // fraction of it, not a second stored colour - reusing
        // `emission` for that (an otherwise-unused field on a
        // non-emissive material, the same kind of repurposing
        // `ior`/`roughness` already do for materialTypes 2/4/5)
        // would have been read by the UNCONDITIONAL emissive-hit
        // check below (`any(mat.emission) > 0`) as "this triangle
        // is a light," making the floor incorrectly glow - a real
        // near-miss caught before it shipped, not a hypothetical.
        float2 uv = texCoordFor(primId, result.triangle_barycentric_coord, uvs);
        albedo = checkerColor(uv, 8.0, float3(mat.color), float3(mat.color) * 0.15);
    } else if (mat.materialType == METAL_MAT_CHECKER3D_LAMBERTIAN) {
        // Real 3D world-space checker (see checker3DColor()'s own
        // declaration comment) - `color`/`transmitColor` hold the
        // two REAL, independent tile colours (unlike materialType
        // 6's own fixed-fraction-of-one-colour simplification,
        // this one has no UV-collision reason to avoid a second
        // stored colour - `transmitColor` is otherwise unused by
        // any Lambertian-family material, materialType 12's own
        // diffuse-transmission tint is a different context
        // entirely). `roughness` reused as the checker's own
        // world-space cell size (`scale` in checker3DColor()'s
        // own signature - matches this project's own CPU
        // checker_texture's `inv_scale` construction parameter,
        // section 121).
        albedo = checker3DColor(hitPoint, mat.roughness, float3(mat.color), float3(mat.transmitColor));
    } else if (mat.materialType == METAL_MAT_MARBLE_LAMBERTIAN) {
        // Real Perlin-noise "marble" (see turbulenceSimple()'s
        // own declaration comment) - matches CPU's own
        // noise_texture::value() exactly: grey (0.5,0.5,0.5)
        // modulated by 1+sin(scale*p.z + 10*turb(p,7)), depth 7/
        // omega 0.5 fixed (CPU's own perlin::turb() defaults,
        // never overridden by any Basics-category scene).
        // `roughness` reused as the texture's own `scale`
        // parameter (matches materialType 16's own established
        // reuse of the same field for an unrelated procedural
        // texture's own scale). World-space `hitPoint`, no UV
        // needed - same reason materialType 16 works on a
        // sphere with no real UV parameterization.
        float marble = 1.0 + sin(mat.roughness * hitPoint.z + 10.0 * turbulenceSimple(hitPoint, 0.5, 7));
        albedo = float3(0.5, 0.5, 0.5) * marble;
    } else if (mat.materialType == METAL_MAT_TEXTURE_FAMILY) {
        // A pbrt-v4 "checkerboard" Texture bound to a Diffuse
        // material's own "reflectance" (B22, section 162) -
        // pbrt-v4's real UV-space 2D checkerboard (NOT this
        // file's own materialType 6, whose tile B is a fixed
        // fraction of tile A rather than a second independent
        // colour, and NOT materialType 16's own WORLD-SPACE 3D
        // checker, which has no notion of "uscale"/"vscale" at
        // all). `color`/`transmitColor` hold the two independent
        // tile colours - same reuse materialType 16 already
        // established for the identical purpose (that function's
        // own comment). `conductorEta.x`/`.y` reused as
        // uscale/vscale (spare for every material type but 4/9,
        // same "one scalar slot, per-materialType meaning"
        // pattern every other TriangleMaterial field reuse here
        // already follows) - pbrt-v4's own two INDEPENDENT scale
        // factors, unlike materialType 6/16's own single shared
        // `scale`. Same isSphere/texCoordFor() UV split
        // materialType 3 above already established (equirect-
        // angular UV on a sphere hit, real triangle UV
        // otherwise) - this is the first OTHER material type to
        // need UV on a sphere at all, reusing that exact
        // mechanism rather than re-deriving it.
        const int texMode = int(mat.conductorK.y + 0.5);
        if (texMode == 8) {
            // Checkerboard of an imagemap, optionally nested one level (J3/J6; loader comment):
            // the same uv drives every level, each checker with its own scale.
            float2 uv8 = isSphere ? equirectangularUV(float3(normal.x, normal.y, -normal.z))
                                  : texCoordFor(primId, result.triangle_barycentric_coord, uvs);
            float2 outerTile = floor(uv8 * float2(mat.conductorEta.x, mat.conductorEta.y));
            if ((int(outerTile.x) + int(outerTile.y)) & 1) {
                albedo = float3(mat.color);                      // outer tex2
            } else {
                bool imageHere = true;
                float3 inner = float3(mat.transmitColor);          // nested tex2
                if (mat.conductorEta.z > 0.5) {
                    float2 innerTile = floor(uv8 * float2(mat.conductorK.x, mat.conductorK.z));
                    imageHere = ((int(innerTile.x) + int(innerTile.y)) & 1) == 0;
                }
                albedo = imageHere ? pbrtDiffuseTexture.sample(textureSampler, float2(uv8.x, 1.0 - uv8.y)).rgb : inner;
            }
        } else if (texMode == 3 || texMode == 4 || texMode == 5) {
            // fbm (3) / windy (4) / wrinkled (5), ports of CPU's fbm_texture / windy_texture / wrinkled_texture: keyed on the
            // pbrt-world point t = k*p + off, no scale.
            float3 wp = mat.conductorK.x * hitPoint + float3(mat.conductorEta);
            float tv;
            if (texMode == 3) {
                tv = 0.5 + 0.5 * fbmSimple(wp, mat.transmitColor.y, int(mat.transmitColor.x + 0.5));
            } else if (texMode == 4) {
                float windStrength = fbmSimple(0.1 * wp, 0.5, 3);
                float waveHeight = fbmSimple(wp, 0.5, 6);
                tv = 0.5 + 0.5 * (abs(windStrength) * waveHeight);
            } else {
                tv = turbulenceSimple(wp, mat.transmitColor.y, int(mat.transmitColor.x + 0.5));
            }
            albedo = float3(clamp(tv, 0.0, 1.0));
        } else if (texMode == 6 || texMode == 7) {
            float2 uvp = isSphere ? equirectangularUV(float3(normal.x, normal.y, -normal.z))
                                  : texCoordFor(primId, result.triangle_barycentric_coord, uvs);
            if (texMode == 7) {
                // bilerp: v00 = color, v01 = transmitColor, v10 = conductorEta, v11 = (K.x, roughness, K.z)
                float3 v11 = float3(mat.conductorK.x, mat.roughness, mat.conductorK.z);
                albedo = (1.0 - uvp.x) * (1.0 - uvp.y) * float3(mat.color) + uvp.x * (1.0 - uvp.y) * float3(mat.conductorEta)
                       + (1.0 - uvp.x) * uvp.y * float3(mat.transmitColor) + uvp.x * uvp.y * v11;
            } else {
                // dots: port of dots_texture::is_inside_dot (cell-jittered discs from Perlin noise)
                float sCell = floor(uvp.x + 0.5), tCell = floor(uvp.y + 0.5);
                bool inside = false;
                if (perlinNoise3D(float3(sCell + 0.5, tCell + 0.5, 0.5)) > 0.0) {
                    const float radius = 0.35, maxShift = 0.5 - radius;
                    float sC = sCell + maxShift * perlinNoise3D(float3(sCell + 1.5, tCell + 2.8, 0.5));
                    float tC = tCell + maxShift * perlinNoise3D(float3(sCell + 4.5, tCell + 9.8, 0.5));
                    float ds = uvp.x - sC, dt = uvp.y - tC;
                    inside = (ds * ds + dt * dt) < radius * radius;
                }
                albedo = inside ? float3(mat.color) : float3(mat.transmitColor);
            }
        } else if (texMode == 2) {
            // pbrt "marble" reflectance texture (loader comment): evaluated at
            // t = k*p + off; transmitColor = (octaves, omega, variation).
            float3 mp = mat.conductorK.x * hitPoint + float3(mat.conductorEta);
            albedo = pbrtMarbleColor(mp, mat.transmitColor.y, int(mat.transmitColor.x + 0.5), mat.transmitColor.z);
        } else if (texMode == 1) {
            // 3D checkerboard (see the loader's own comment): texture-space point
            // t = k*p + off, parity of floor(tx)+floor(ty)+floor(tz); even -> tex1.
            float3 tp = mat.conductorK.x * hitPoint + float3(mat.conductorEta);
            int cellSum = int(floor(tp.x)) + int(floor(tp.y)) + int(floor(tp.z));
            albedo = ((cellSum & 1) == 0) ? float3(mat.color) : float3(mat.transmitColor);
        } else {
        float2 uv = isSphere ? equirectangularUV(float3(normal.x, normal.y, -normal.z))
                              : texCoordFor(primId, result.triangle_barycentric_coord, uvs);
        float2 tile = floor(uv * float2(mat.conductorEta.x, mat.conductorEta.y));
        float parity = fmod(tile.x + tile.y, 2.0);
        albedo = (abs(parity) < 0.5) ? float3(mat.color) : float3(mat.transmitColor);
        }
    } else if (mat.materialType == METAL_MAT_IMAGE_LAMBERTIAN || mat.materialType == METAL_MAT_IMAGE_CLEARCOAT) {
        // A pbrt-v4 Diffuse (26) or CoatedDiffuse (27, J1,
        // section 172) material's own "texture reflectance"
        // bound to a bare "imagemap" Texture (F5/F9, section
        // 166) - a REAL image FILE (`Material::textureFilename`,
        // pbrt_flatten.h), not a procedural pattern like
        // materialType 25's own checkerboard - reads
        // `pbrtDiffuseTexture` (this scene's own decoded file)
        // instead of the hardcoded room's own earthTexture, the
        // same "separate slot" reasoning materialType 3 already
        // established for pbrt-loaded infinite-light images.
        // Same isSphere/texCoordFor() UV split materialType 3/25
        // above already established. `1.0 - uv.y`: pbrt-v4's own
        // UV convention has v=0 at the BOTTOM of the image (this
        // scene's own "point2 uv" authors v=0/v=1 that way), but
        // pbrt_load::detail::decodeInfiniteLightImage() (stb_image
        // under the hood) always returns row 0 as the TOP row,
        // the same row Metal's own `replaceRegion:` (this
        // texture's own upload call, metal_poc.mm) then places at
        // texture row 0 too - sampled with `uv.y` UNFLIPPED, that
        // combination reads v=0 from the TOP instead of the
        // BOTTOM, a real top/bottom mirror caught by comparing
        // this scene's own 4-quadrant checker against `--cpu`
        // (each quadrant's colour landed in the vertically
        // opposite corner) rather than assumed correct from a
        // symmetric test image. `* mat.roughness`: this hit's own
        // texture SCALE multiplier (a "scale"-class Texture
        // wrapping the bare imagemap, Material::textureScale's
        // own comment, mapMaterial()'s own comment on reusing
        // this field) - both materialType 26 and 27's own
        // TriangleMaterial construction now sets `roughness` to
        // this value (1.0, a provable no-op, when the scene's
        // own texture had no wrapping "scale").
        float2 uv = isSphere ? equirectangularUV(float3(normal.x, normal.y, -normal.z))
                              : texCoordFor(primId, result.triangle_barycentric_coord, uvs);
        albedo = pbrtDiffuseTexture.sample(textureSampler, float2(uv.x, 1.0 - uv.y)).rgb * mat.roughness;
    } else if (mat.materialType == METAL_MAT_DIFFUSE_TRANSMISSION && mat.conductorK.y > 8.5) {
        // DiffuseTransmission with image-textured reflectance (diffuse slot) and/or transmittance
        // (second slot) - J2. conductorK.x = 1 when reflectance is an image, conductorK.z = 1 when
        // transmittance is; otherwise the flat colour in color / transmitColor is used. The shade
        // function reads mat.transmitColor, so the sampled transmittance is written back into it.
        float2 uv9 = texCoordFor(primId, result.triangle_barycentric_coord, uvs);
        albedo = (mat.conductorK.x > 0.5)
            ? pbrtDiffuseTexture.sample(textureSampler, float2(uv9.x, 1.0 - uv9.y)).rgb
            : float3(mat.color);
        if (mat.conductorK.z > 0.5)
            mat.transmitColor = packed_float3(pbrtTransmitTexture.sample(textureSampler, float2(uv9.x, 1.0 - uv9.y)).rgb);
    } else {
        albedo = float3(mat.color);
    }
}

// Emission at the hit: an emissive surface adds its light to the radiance, weighted by MIS against light sampling unless the previous bounce
// was specular.
inline void addHitEmission(thread const KernelRes& R, thread PathState& P, thread BounceState& B) {
    KERNEL_RES_ALIASES(R)
    PATH_STATE_ALIASES(P)
    BOUNCE_STATE_ALIASES(B)

    // Unconditional check (regardless of material type) for
    // whether this hit is the light quad - covers a camera ray or
    // a GI bounce landing on it directly, mirror/glass included (a
    // mirror reflecting toward the light correctly shows it, since
    // this fires for THEIR reflected/refracted rays' next hit too,
    // not just Lambertian ones). Every non-light surface has
    // emission == 0, so `any(...)` below is false and this whole
    // block is a no-op for them.
    //
    // `&& (frontFace || mat.twoSided != 0u)`: an AreaLight only
    // emits from the side its own `normal` points toward, UNLESS
    // the scene named `"bool twosided" [true]` (section 104) -
    // the same one-sidedness (or lack of it) the NEE branches
    // below already enforce via their own `(cosLight > 0.0 ||
    // (ls.twoSided != 0.0 && cosLight < 0.0))` check (see e.g.
    // this scene's own ceiling lights, which only shine down
    // into the room, one-sided). Without the frontFace half of
    // this, a camera ray or BSDF-sampled bounce landing on the
    // BACK of a one-sided light quad would still read its
    // emission unconditionally - invisible in this committed
    // scene (every light is mounted flush against the ceiling,
    // its own back face physically inaccessible from inside the
    // room) but a genuine correctness gap: this is the one place
    // in the shader a light's own emission was reachable without
    // a facing check at all, inconsistent with every NEE
    // branch's own already-correct behaviour. `frontFace` is
    // already computed above for the dielectric branch's own eta
    // selection - reused here, not recomputed. `mat.twoSided`
    // (not `lights[mat.lightId].twoSided`) so this same check
    // also covers a non-quad/disk emissive shape with NO
    // AreaLightData entry at all (`mat.lightId < 0`, sections
    // 100/101) - every emissive TriangleMaterial carries its own
    // copy of the flag directly, needing no lights[] lookup.
    if (any(float3(mat.emission) > float3(0.0)) && (frontFace || mat.twoSided != 0u)) {
        // Patterned emission (materialType 10 - see AreaLight's
        // own comment): a DIRECT hit needs the checker pattern
        // evaluated at THIS hit's own interpolated UV
        // (texCoordFor(), the same triangle-only lookup
        // materialType 3/6's own albedo already uses), not the
        // light-sample point's (u.x, u.y) sampleAreaLight() uses -
        // two different points on the same light quad, each
        // needing its own UV. `roughness` reused a SIXTH way here
        // (after materialTypes 4/5/7/9's own reuses) as this
        // pattern's own tile-B fraction, mirroring AreaLight's
        // `patternTileB`.
        float3 hitEmission = float3(mat.emission);
        if (mat.materialType == METAL_MAT_PATTERNED_EMISSIVE) {
            float2 patUV = texCoordFor(primId, result.triangle_barycentric_coord, uvs);
            hitEmission = checkerColor(patUV, 6.0, hitEmission, hitEmission * mat.roughness);
        } else if (mat.materialType == METAL_MAT_IMAGE_EMISSIVE) {
            // Real image-based AreaLightSource (section 105) -
            // same UV lookup as materialType 10's own checker
            // pattern (a direct hit needs THIS hit's own
            // interpolated UV, not the NEE sample point's), but
            // sampling a real texture instead of a procedural
            // pattern. `mat.emission` here holds the pure
            // (scale,scale,scale) multiplier (see AreaLight::
            // useTexture's own comment), not a direct radiance -
            // pbrt-v4 ignores L entirely once an image is given.
            // A disk or cylinder light has no triangle UV: derive pbrt's (u, v) for it from the hit point and the light's own frame.
            float2 texUV;
            if (isDisk && mat.lightId >= 0) texUV = diskLightTexCoord(lights[mat.lightId], hitPoint);
            else if (isCylinder && mat.lightId >= 0) texUV = cylinderLightTexCoord(lights[mat.lightId], hitPoint);
            else texUV = texCoordFor(primId, result.triangle_barycentric_coord, uvs);
            hitEmission = pbrtAreaLightTexture.sample(textureSampler, texUV).rgb * hitEmission;
        }
        if (specularBounce || mat.lightId < 0) {
            // No competing NEE sample could have produced this
            // exact hit (camera ray, or a mirror/glass bounce -
            // both skip NEE entirely, see their own branches
            // below), so there's nothing to weight against.
            // `mat.lightId < 0` covers a SEPARATE case with the
            // same "nothing to weight against" shape: a pbrt-
            // loaded area light shape too complex for this
            // loader's own single-quad AreaLightData (loadPbrtScene()'s
            // own comment) - emissive, but deliberately not
            // registered in `lights[]` for any material function's
            // own NEE loop to have sampled it from in the first
            // place.
            radiance += throughput * hitEmission;
        } else {
            // Reached via a BSDF-sampled continuation ray (diffuse
            // or conductor) - weight by the power heuristic against
            // what the light-sampling strategy's own PDF would have
            // been for this exact hit. mat.lightId names exactly
            // which AreaLight this triangle belongs to, so this
            // works for any number of lights, not just one -
            // this light's own power-proportional picking pdf
            // (`light.pmf`, see AreaLight's own comment - replaces
            // this POC's old flat 1/lightCount) folded in alongside
            // the same area-to-solid-angle conversion the NEE
            // branches below use.
            AreaLight light = lights[mat.lightId];
            const float hitDistFromBounce = result.distance + mediumSkippedDist;
            float distSq = hitDistFromBounce * hitDistFromBounce;
            // abs(), not the old max(dot(...), 0.0001) alone - a
            // real, previously-latent bug found by code review
            // (section 106), exposed once section 104's own
            // frontFace||twoSided gate made a BACK-face hit on a
            // two-sided light reachable here at all. On that
            // face, dot(normal,-rayDir) is NEGATIVE; the old
            // max(...,0.0001) clamped it up to the epsilon
            // itself rather than reflecting it, making pdfLight
            // spuriously huge and crushing this MIS weight
            // toward 0 - silently dropping the BSDF-sampled
            // strategy's own contribution to a two-sided light's
            // back face, an energy-loss bias. abs() first, THEN
            // the same 0.0001 floor purely to avoid a division
            // by exact zero at a grazing angle - matches the
            // same abs(cosLight) fix the NEE branches below
            // already got in section 104, just missed here.
            //
            // A sphere light (AreaLight::kind==1, B14, section
            // 184) has no single fixed plane normal the way a
            // quad does - `light.normal` is unused/meaningless for
            // one (see that field's own comment) - so this uses
            // THIS hit's own already-computed geometric `normal`
            // (line ~787, `normalize(hitPoint - sphereCenter)`)
            // instead, the correct per-point outward normal a
            // curved surface needs for its own area-to-solid-angle
            // Jacobian.
            float3 lightNormalAtHit = (light.kind > 0.5 && light.kind < 1.5) || (light.kind > 2.5 && light.kind < 3.5) ? normal : float3(light.normal);
            float cosLight = max(abs(dot(lightNormalAtHit, -rayDir)), 0.0001);
            float pdfLight = (distSq / (light.area * cosLight)) * light.pmf;
            float weight = (bsdfPdf * bsdfPdf) / (bsdfPdf * bsdfPdf + pdfLight * pdfLight);
            radiance += throughput * hitEmission * weight;
        }
    }
}
