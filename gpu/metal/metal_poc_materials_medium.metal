// metal_poc_materials_medium.metal
// Bounded per-object volumetric medium sphere shading (materialType
// 28/29/30 - homogeneous, CloudMedium, RgbGridMedium respectively) -
// split out of metal_poc_kernel.metal's own primaryRayKernel() once that
// single function grew to ~1729 lines (pure code motion, no behaviour
// change - each function body below is character-for-character the same
// logic that used to live inline in the kernel's own materialType
// dispatch, just moved behind a real function call, the same "one
// standalone shadeXxx() per material" convention metal_poc_materials_
// specular/diffuse/layered/extra.metal already use for every SURFACE
// material). See each function's own header comment (and sections 176/
// 178/179, docs/METAL_GPU_FEASIBILITY.md) for what the algorithm itself
// does - this split changes none of it.
//
// `entryDistance` is the hit sphere's own `result.distance` from the
// primaryRayKernel's own initial bounding_box intersection test,
// renamed at the call boundary only to avoid shadowing shadeHomogeneous
// MediumSphere()'s own internal `entryT` local.
//
// Unlike the shadeXxx() surface-material functions (which return `bool`
// - false means "break the bounce loop, this was a specular/terminal
// event"), these three are `void`: a medium interaction never early-
// terminates the path, it only ever scatters or passes through, so the
// caller's own `scatteredInMedium`/`passedThroughMediumSphere` out-
// parameters (mutated here by reference) are all it needs to decide
// what happens next.

// materialType 28 (A8, section 176) - bounded, homogeneous free-flight
// scattering sphere. See metal_poc_scenes_a.mm's own buildCornellSmoke()
// comment and this function's own inline comments (unchanged from the
// original inline kernel code) for the full algorithm.
inline void shadeHomogeneousMediumSphere(
    TriangleMaterial mediumMat, uint mediumPrimId, float entryDistance,
    device const SphereData* spheres, float shutterT,
    device const AreaLight* lights, constant Uniforms& uniforms,
    texture2d<float, access::sample> pbrtAreaLightTexture, sampler textureSampler,
    intersector<instancing, triangle_data> isect,
    instance_acceleration_structure accelStructure,
    intersection_function_table<instancing, triangle_data> functionTable,
    SpherePayload shadowSpherePayload,
    thread float3& rayDir, thread float3& rayOrigin,
    thread float3& throughput, thread float3& radiance,
    thread float& bsdfPdf, thread bool& specularBounce, thread uint& rngState,
    thread bool& scatteredInMedium, thread bool& passedThroughMediumSphere) {
    SphereData mediumSphere = spheres[mediumPrimId];
    float3 sphereCenter = float3(mediumSphere.center) + shutterT * float3(mediumSphere.centerDelta1);
    float entryT = entryDistance;
    float exitT = 2.0 * dot(sphereCenter - rayOrigin, rayDir) - entryT;
    float sigmaT = mediumMat.ior;
    float u = randFloat(rngState);
    float tScatter = -log(max(1.0 - u, 1e-6)) / sigmaT;
    if (tScatter < (exitT - entryT)) {
        float3 scatterPoint = rayOrigin + rayDir * (entryT + tScatter);
        float3 wo = -rayDir;

        LightSample ls = sampleAreaLight(lights, uniforms.lightCount, rngState, pbrtAreaLightTexture, textureSampler);
        float3 toLight = ls.point - scatterPoint;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        float3 wi = toLight / dist;
        float cosLight = dot(ls.normal, -wi);
        if ((cosLight > 0.0 || (ls.twoSided != 0.0 && cosLight < 0.0))) {
            ray mediumShadowRay;
            mediumShadowRay.origin = scatterPoint;
            mediumShadowRay.direction = wi;
            mediumShadowRay.min_distance = 0.001f;
            mediumShadowRay.max_distance = dist - 0.002f;
            intersection_result<instancing, triangle_data> mediumShadowResult =
                isect.intersect(mediumShadowRay, accelStructure, functionTable, shadowSpherePayload);
            if (mediumShadowResult.type == intersection_type::none) {
                float pdfSolidAngle = (distSq / (ls.area * abs(cosLight))) * ls.pmf;
                float phaseValue = henyeyGreensteinPhase(dot(wo, wi), mediumMat.roughness);
                float weight = (pdfSolidAngle * pdfSolidAngle)
                    / (pdfSolidAngle * pdfSolidAngle + phaseValue * phaseValue);
                // This shadow ray's own path never leaves
                // THIS medium sphere (A8's own two smoke
                // volumes sit apart, never overlapping) -
                // exp(-sigmaT*dist) attenuates it by this
                // sphere's own transmittance along its
                // length, the identical reasoning the
                // global fog's own NEE block gives for
                // its own shadow ray.
                //
                // `* float3(mediumMat.color)`: THIS
                // scattering event's own albedo weight
                // (this whole block's own header comment:
                // "weight = sigmaS*T(t)/p(t) = sigmaS/
                // sigmaT, the albedo" applies to ANY
                // estimator reached FROM this scatter
                // point, not just the phase-sampled
                // continuation ray below) - `throughput`
                // here only carries what accumulated
                // BEFORE this bounce, so leaving this out
                // would silently render every medium
                // sphere's own NEE contribution as the
                // LIGHT's own colour (a near-white 7,7,7
                // here) with no tint from the medium's own
                // albedo at all - caught by comparing a
                // real dark-tinted sphere against a
                // warm-amber one and finding them
                // indistinguishable, not assumed correct
                // from the formula alone.
                float transmittance = exp(-sigmaT * dist);
                radiance += throughput * float3(mediumMat.color) * phaseValue * ls.emission * transmittance
                            / pdfSolidAngle * weight;
            }
        }

        float3 newDir = sampleHenyeyGreenstein(wo, mediumMat.roughness, rngState);
        throughput *= float3(mediumMat.color);
        bsdfPdf = henyeyGreensteinPhase(dot(wo, newDir), mediumMat.roughness);
        rayDir = newDir;
        rayOrigin = scatterPoint;
        specularBounce = false;
        scatteredInMedium = true;
    } else {
        // Survived to the far side - continue from the
        // EXIT point, same direction, unchanged throughput
        // (this block's own header comment: T(exit)/
        // P(survive to exit) == 1 exactly). Consumes one
        // iteration of this loop's own depth budget, same
        // as any other bounce.
        rayOrigin = rayOrigin + rayDir * exitT;
        passedThroughMediumSphere = true;
    }
}

// materialType 29 (E2, section 178) - heterogeneous, procedural
// Perlin-noise CloudMedium. See metal_poc_scenes_e.mm's own
// buildCloudMediumScene() comment and this function's own inline
// comments (unchanged from the original inline kernel code) for the
// full algorithm.
inline void shadeCloudMediumSphere(
    TriangleMaterial mediumMat, float entryDistance,
    device const GpuCloudMedium* cloudMediums,
    device const AreaLight* lights, constant Uniforms& uniforms,
    texture2d<float, access::sample> pbrtAreaLightTexture, sampler textureSampler,
    intersector<instancing, triangle_data> isect,
    instance_acceleration_structure accelStructure,
    intersection_function_table<instancing, triangle_data> functionTable,
    SpherePayload shadowSpherePayload,
    thread float3& rayDir, thread float3& rayOrigin,
    thread float3& throughput, thread float3& radiance,
    thread float& bsdfPdf, thread bool& specularBounce, thread uint& rngState,
    thread bool& scatteredInMedium, thread bool& passedThroughMediumSphere) {
    // E2/section 178: heterogeneous, procedural Perlin-
    // noise cloud - delta tracking (null-collision free-
    // path sampling, pbrt-v4 SampleT_maj) through the
    // medium's own world-space AABB, NOT this trigger
    // sphere's bounds (the trigger sphere only exists to
    // get the ray into this branch at all, sized to
    // comfortably contain that AABB - see
    // buildCloudMediumScene()'s own comment,
    // metal_poc_scenes_e.mm). Direct port of
    // gpu/optix/optix_intersection_sphere.h's own
    // MaterialType::CloudMedium closest-hit code - see
    // that block's own comment for the algorithm.
    GpuCloudMedium cloud = cloudMediums[uint(mediumMat.conductorEta.x)];
    float3 mo = worldToMediumPoint(cloud, rayOrigin);
    // Direction transforms by the matrix only (no
    // translation) - CloudMedium<T>::sample_ray()'s own
    // convention.
    float3 md = float3(
        cloud.worldToMediumMat[0]*rayDir.x + cloud.worldToMediumMat[1]*rayDir.y + cloud.worldToMediumMat[2]*rayDir.z,
        cloud.worldToMediumMat[3]*rayDir.x + cloud.worldToMediumMat[4]*rayDir.y + cloud.worldToMediumMat[5]*rayDir.z,
        cloud.worldToMediumMat[6]*rayDir.x + cloud.worldToMediumMat[7]*rayDir.y + cloud.worldToMediumMat[8]*rayDir.z);
    float segMin, segMax;
    bool hasSeg = cloudAabbSlabIntersect(cloud, mo, md, segMin, segMax);
    float sigmaMaj = cloud.sigmaA + cloud.sigmaS;

    bool didScatter = false;
    float3 mediumPoint = float3(0.0, 0.0, 0.0);
    float3 wo = -rayDir;
    float missedExitT = entryDistance; // trigger sphere's own entry hit
    if (hasSeg && sigmaMaj > 0.0) {
        float tt = max(segMin, 0.0);
        // Bounded iteration count - device code must not
        // risk an unbounded loop from a pathological
        // (near-zero majorant) configuration, same cap
        // OptiX's own port uses.
        for (int iter = 0; iter < 128 && !didScatter; ++iter) {
            float dt = -log(max(1.0 - randFloat(rngState), 1e-8)) / sigmaMaj;
            tt += dt;
            if (tt >= segMax) break;
            float3 p = rayOrigin + tt * rayDir;
            float3 mp = worldToMediumPoint(cloud, p);
            float d = gpuCloudDensity(cloud, mp.x, mp.y, mp.z);
            float sigmaSLocal = d * cloud.sigmaS;
            if (randFloat(rngState) < sigmaSLocal / sigmaMaj) {
                didScatter = true;
                mediumPoint = p;
            }
        }
        if (!didScatter) missedExitT = segMax;
    }

    if (didScatter) {
        // NEE only fires when a REAL area light is
        // registered - unlike A8/E1 (materialType 28's own
        // scenes), E2 has none: it's lit purely by the
        // constant background/environment colour, the
        // same "no NEE technique exists yet for that
        // light kind" scope cut every other light-type
        // (point/directional/projection/goniometric) NEE
        // block already documents. sampleAreaLight() has
        // no lightCount==0 guard of its own (every OTHER
        // call site is on a scene that always registers
        // at least one) - calling it unconditionally here
        // read lights[0] out of an empty (zero-filled)
        // buffer, giving ls.area==0/ls.pmf==0, and
        // `distSq/(0*cosLight)*0` evaluated to NaN
        // (INF*0), silently corrupting every pixel's own
        // accumulated radiance - caught by rendering and
        // comparing against `--cpu`, which showed a
        // washed-out, near-black speckled cloud instead
        // of CPU's own soft grey one, not assumed correct
        // from the formula alone.
        if (uniforms.lightCount > 0u) {
        LightSample ls = sampleAreaLight(lights, uniforms.lightCount, rngState, pbrtAreaLightTexture, textureSampler);
        float3 toLight = ls.point - mediumPoint;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        float3 wi = toLight / dist;
        float cosLight = dot(ls.normal, -wi);
        if ((cosLight > 0.0 || (ls.twoSided != 0.0 && cosLight < 0.0))) {
            ray cloudShadowRay;
            cloudShadowRay.origin = mediumPoint;
            cloudShadowRay.direction = wi;
            cloudShadowRay.min_distance = 0.001f;
            cloudShadowRay.max_distance = dist - 0.002f;
            intersection_result<instancing, triangle_data> cloudShadowResult =
                isect.intersect(cloudShadowRay, accelStructure, functionTable, shadowSpherePayload);
            if (cloudShadowResult.type == intersection_type::none) {
                float pdfSolidAngle = (distSq / (ls.area * abs(cosLight))) * ls.pmf;
                float phaseValue = henyeyGreensteinPhase(dot(wo, wi), mediumMat.roughness);
                float weight = (pdfSolidAngle * pdfSolidAngle)
                    / (pdfSolidAngle * pdfSolidAngle + phaseValue * phaseValue);
                // Same "* float3(mediumMat.color)" albedo-
                // weight fix materialType 28's own NEE
                // block above already established - see
                // that block's own comment. No extra
                // transmittance factor here (unlike
                // materialType 28's homogeneous case):
                // delta tracking's own null-collision
                // sampling already makes reaching a real
                // scatter event at `mediumPoint` an
                // unbiased estimator with unit weight, so
                // this NEE ray only needs the target
                // light's own occlusion test - the same
                // reasoning OptiX's own medium_phase_nee_mis()
                // relies on.
                radiance += throughput * float3(mediumMat.color) * phaseValue * ls.emission
                            / pdfSolidAngle * weight;
            }
        }
        }

        // NEE toward the constant background/environment
        // light (E2's own real illumination source - see
        // buildCloudMediumScene()'s own comment, no
        // AreaLight is ever registered for this scene).
        // Without this, escaping a dense sigma_s==10
        // medium (mean free path ~0.1 world units across
        // an ~8-unit box) relies ENTIRELY on the plain
        // phase-sampled continuation ray below randomly
        // walking all the way out before max_depth runs
        // out - an astronomically low-probability event
        // in practice, which rendered as a near-solid-
        // black cloud (caught by comparing against
        // `--cpu`, which shows a properly lit soft grey
        // one - CPU's own hg_phase_material has an
        // equivalent NEE-against-every-registered-light
        // technique, ratio-tracking the transmittance;
        // ports_medium/materialType 28's own NEE only
        // ever needed area lights before this, since
        // every OTHER scene combining a bounded medium
        // with real light used one).
        //
        // The shadow ray here needs no companion
        // transmittance estimate along its own path back
        // out through the REST of this same cloud (unlike
        // CPU's own ratio-tracking estimator) because
        // shadowSpherePayload's own isShadowRay==true
        // convention already makes every medium/cloud
        // trigger sphere (materialType 28 AND 29)
        // completely transparent to a pure occlusion
        // test (SpherePayload::isShadowRay's own comment,
        // metal_poc_types.metal) - so this ray only ever
        // reports a REAL blocker (ground/background
        // spheres/room walls), never the cloud's own
        // density along the way. A real, honest
        // simplification versus CPU's own continuous
        // partial-transmittance estimate (this ray is a
        // binary hit/miss, ignoring the cloud's own self-
        // attenuation along its own remaining path out),
        // not a bug - documented the same "single-
        // scattering, medium doesn't self-shadow its own
        // NEE ray" scope cut a real-time renderer would
        // also make. No MIS weight against the
        // continuation ray's own eventual (vanishingly
        // rare) escape-to-background contribution, same
        // "no NEE strategy to double-count against"
        // weight==1.0 convention `pbrtHasConstantEnvLight`'s
        // own miss-path arm above already uses - the two
        // ever co-occurring for the same path is
        // negligible given how rare an unassisted escape
        // already is.
        if (uniforms.pbrtHasConstantEnvLight != 0u) {
            float3 bgDir = sampleHenyeyGreenstein(wo, mediumMat.roughness, rngState);
            // This NEE ray's own remaining self-
            // attenuation through the REST of the cloud,
            // approximated via the conservative MAJORANT
            // sigma_t (sigmaMaj) over the analytic box-
            // exit distance in this direction (the same
            // ray/AABB slab test the delta-tracking loop
            // above already uses) - a coarse stand-in for
            // CPU's own real ratio-tracking transmittance
            // estimate (src/shared/ratio_tracking.h),
            // cheap (one more slab test, no extra rays)
            // and enough to fix the massive over-
            // brightening an earlier, fully-unattenuated
            // version of this NEE ray produced (every
            // scatter event along a path added a FULL,
            // un-decayed background contribution
            // regardless of how deep inside the cloud it
            // was, summing to a blown-out white box -
            // caught by comparing against `--cpu`, not
            // assumed correct from the formula alone). Not
            // a bias-free match to CPU's own continuous
            // estimator - a real, documented scope cut.
            float3 bgMo = worldToMediumPoint(cloud, mediumPoint);
            float3 bgMd = float3(
                cloud.worldToMediumMat[0]*bgDir.x + cloud.worldToMediumMat[1]*bgDir.y + cloud.worldToMediumMat[2]*bgDir.z,
                cloud.worldToMediumMat[3]*bgDir.x + cloud.worldToMediumMat[4]*bgDir.y + cloud.worldToMediumMat[5]*bgDir.z,
                cloud.worldToMediumMat[6]*bgDir.x + cloud.worldToMediumMat[7]*bgDir.y + cloud.worldToMediumMat[8]*bgDir.z);
            float bgSegMin, bgSegMax;
            bool bgHasSeg = cloudAabbSlabIntersect(cloud, bgMo, bgMd, bgSegMin, bgSegMax);
            float remainingDist = (bgHasSeg && bgSegMax > 0.0) ? bgSegMax : 0.0;
            float selfTransmittance = exp(-sigmaMaj * remainingDist);

            ray bgShadowRay;
            bgShadowRay.origin = mediumPoint;
            bgShadowRay.direction = bgDir;
            bgShadowRay.min_distance = 0.001f;
            bgShadowRay.max_distance = 1.0e6f;
            intersection_result<instancing, triangle_data> bgShadowResult =
                isect.intersect(bgShadowRay, accelStructure, functionTable, shadowSpherePayload);
            if (bgShadowResult.type == intersection_type::none) {
                radiance += throughput * float3(mediumMat.color) * float3(uniforms.pbrtEnvColor) * selfTransmittance;
            }
        }

        float3 newDir = sampleHenyeyGreenstein(wo, mediumMat.roughness, rngState);
        throughput *= float3(mediumMat.color);
        bsdfPdf = henyeyGreensteinPhase(dot(wo, newDir), mediumMat.roughness);
        rayDir = newDir;
        rayOrigin = mediumPoint;
        specularBounce = false;
        scatteredInMedium = true;
    } else {
        // Either missed the medium's own tighter AABB
        // entirely (a real, expected case - the trigger
        // sphere is a loose bound) or survived the whole
        // delta-tracking segment with no scatter -
        // continue unchanged from `missedExitT`, same
        // "no interaction, free pass-through" contract as
        // materialType 28's own else-branch above.
        rayOrigin = rayOrigin + rayDir * missedExitT;
        passedThroughMediumSphere = true;
    }
}

// materialType 30 (E4, section 179) - heterogeneous per-voxel R/G/B
// RgbGridMedium. See metal_poc_scenes_e.mm's own buildRgbGridMediumScene()
// comment and this function's own inline comments (unchanged from the
// original inline kernel code) for the full algorithm.
inline void shadeRgbGridMediumSphere(
    TriangleMaterial mediumMat, float entryDistance,
    device const GpuRgbGridMedium* rgbGridMediums, device const float* rgbGridData,
    device const AreaLight* lights, constant Uniforms& uniforms,
    texture2d<float, access::sample> pbrtAreaLightTexture, sampler textureSampler,
    intersector<instancing, triangle_data> isect,
    instance_acceleration_structure accelStructure,
    intersection_function_table<instancing, triangle_data> functionTable,
    SpherePayload shadowSpherePayload,
    thread float3& rayDir, thread float3& rayOrigin,
    thread float3& throughput, thread float3& radiance,
    thread float& bsdfPdf, thread bool& specularBounce, thread uint& rngState,
    thread bool& scatteredInMedium, thread bool& passedThroughMediumSphere) {
    // E4/section 179: heterogeneous per-voxel R/G/B
    // scattering grid (a real "nebula" - independently-
    // colored density per channel, unlike materialType
    // 29's single grayscale noise field). Same delta-
    // tracking shape as materialType 29's own block just
    // above - direct port of gpu/optix/
    // optix_intersection_sphere.h's own RgbGridMedium
    // closest-hit code (see that block's own comment).
    GpuRgbGridMedium grid = rgbGridMediums[uint(mediumMat.conductorEta.x)];
    float3 mo = rgbGridWorldToMediumPoint(grid, rayOrigin);
    float3 md = float3(
        grid.worldToMediumMat[0]*rayDir.x + grid.worldToMediumMat[1]*rayDir.y + grid.worldToMediumMat[2]*rayDir.z,
        grid.worldToMediumMat[3]*rayDir.x + grid.worldToMediumMat[4]*rayDir.y + grid.worldToMediumMat[5]*rayDir.z,
        grid.worldToMediumMat[6]*rayDir.x + grid.worldToMediumMat[7]*rayDir.y + grid.worldToMediumMat[8]*rayDir.z);
    float segMin, segMax;
    bool hasSeg = rgbGridAabbSlabIntersect(grid, mo, md, segMin, segMax);
    float sigmaMaj = grid.sigmaMaj;

    bool didScatter = false;
    float3 mediumPoint = float3(0.0, 0.0, 0.0);
    float3 wo = -rayDir;
    float missedExitT = entryDistance;
    // Per-scatter tint, derived from the LOCAL R/G/B
    // density ratio at the accepted scatter point (unlike
    // materialType 28/29's own fixed `mediumMat.color`
    // albedo) - a nebula's whole point is spatially-
    // varying colour, so this can't be a single constant.
    float3 scatterColor = float3(1.0, 1.0, 1.0);
    if (hasSeg && sigmaMaj > 0.0) {
        float tt = max(segMin, 0.0);
        const int voxelCount = grid.nx * grid.ny * grid.nz;
        device const float* rData = rgbGridData + grid.dataOffset;
        device const float* gData = rData + voxelCount;
        device const float* bData = gData + voxelCount;
        for (int iter = 0; iter < 128 && !didScatter; ++iter) {
            float dt = -log(max(1.0 - randFloat(rngState), 1e-8)) / sigmaMaj;
            tt += dt;
            if (tt >= segMax) break;
            float3 p = rayOrigin + tt * rayDir;
            float3 mp = rgbGridWorldToMediumPoint(grid, p);
            float dr = gpuRgbGridTrilinear(rData, grid.nx, grid.ny, grid.nz, mp.x, mp.y, mp.z);
            float dg = gpuRgbGridTrilinear(gData, grid.nx, grid.ny, grid.nz, mp.x, mp.y, mp.z);
            float db = gpuRgbGridTrilinear(bData, grid.nx, grid.ny, grid.nz, mp.x, mp.y, mp.z);
            float sr = dr * grid.sigmaScale, sg = dg * grid.sigmaScale, sb = db * grid.sigmaScale;
            float sigmaTLocal = max(sr, max(sg, sb));
            if (randFloat(rngState) < sigmaTLocal / sigmaMaj) {
                didScatter = true;
                mediumPoint = p;
                float maxc = max(sr, max(sg, max(sb, 1e-6)));
                scatterColor = float3(sr, sg, sb) / maxc;
            }
        }
        if (!didScatter) missedExitT = segMax;
    }

    if (didScatter) {
        if (uniforms.lightCount > 0u) {
        LightSample ls = sampleAreaLight(lights, uniforms.lightCount, rngState, pbrtAreaLightTexture, textureSampler);
        float3 toLight = ls.point - mediumPoint;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        float3 wi = toLight / dist;
        float cosLight = dot(ls.normal, -wi);
        if ((cosLight > 0.0 || (ls.twoSided != 0.0 && cosLight < 0.0))) {
            ray gridShadowRay;
            gridShadowRay.origin = mediumPoint;
            gridShadowRay.direction = wi;
            gridShadowRay.min_distance = 0.001f;
            gridShadowRay.max_distance = dist - 0.002f;
            intersection_result<instancing, triangle_data> gridShadowResult =
                isect.intersect(gridShadowRay, accelStructure, functionTable, shadowSpherePayload);
            if (gridShadowResult.type == intersection_type::none) {
                float pdfSolidAngle = (distSq / (ls.area * abs(cosLight))) * ls.pmf;
                float phaseValue = henyeyGreensteinPhase(dot(wo, wi), grid.phaseG);
                float weight = (pdfSolidAngle * pdfSolidAngle)
                    / (pdfSolidAngle * pdfSolidAngle + phaseValue * phaseValue);
                radiance += throughput * scatterColor * phaseValue * ls.emission
                            / pdfSolidAngle * weight;
            }
        }
        }

        // NEE toward the constant background light - same
        // majorant-based self-attenuation approximation
        // materialType 29's own block above established
        // (section 178's own comment there has the full
        // "why"), reusing `sigmaMaj` (this medium's own
        // global majorant) in place of CloudMedium's
        // sigma_a+sigma_s.
        if (uniforms.pbrtHasConstantEnvLight != 0u) {
            float3 bgDir = sampleHenyeyGreenstein(wo, grid.phaseG, rngState);
            float3 bgMo = rgbGridWorldToMediumPoint(grid, mediumPoint);
            float3 bgMd = float3(
                grid.worldToMediumMat[0]*bgDir.x + grid.worldToMediumMat[1]*bgDir.y + grid.worldToMediumMat[2]*bgDir.z,
                grid.worldToMediumMat[3]*bgDir.x + grid.worldToMediumMat[4]*bgDir.y + grid.worldToMediumMat[5]*bgDir.z,
                grid.worldToMediumMat[6]*bgDir.x + grid.worldToMediumMat[7]*bgDir.y + grid.worldToMediumMat[8]*bgDir.z);
            float bgSegMin, bgSegMax;
            bool bgHasSeg = rgbGridAabbSlabIntersect(grid, bgMo, bgMd, bgSegMin, bgSegMax);
            float remainingDist = (bgHasSeg && bgSegMax > 0.0) ? bgSegMax : 0.0;
            float selfTransmittance = exp(-sigmaMaj * remainingDist);

            ray bgShadowRay;
            bgShadowRay.origin = mediumPoint;
            bgShadowRay.direction = bgDir;
            bgShadowRay.min_distance = 0.001f;
            bgShadowRay.max_distance = 1.0e6f;
            intersection_result<instancing, triangle_data> bgShadowResult =
                isect.intersect(bgShadowRay, accelStructure, functionTable, shadowSpherePayload);
            if (bgShadowResult.type == intersection_type::none) {
                radiance += throughput * scatterColor * float3(uniforms.pbrtEnvColor) * selfTransmittance;
            }
        }

        float3 newDir = sampleHenyeyGreenstein(wo, grid.phaseG, rngState);
        throughput *= scatterColor;
        bsdfPdf = henyeyGreensteinPhase(dot(wo, newDir), grid.phaseG);
        rayDir = newDir;
        rayOrigin = mediumPoint;
        specularBounce = false;
        scatteredInMedium = true;
    } else {
        rayOrigin = rayOrigin + rayDir * missedExitT;
        passedThroughMediumSphere = true;
    }
}
