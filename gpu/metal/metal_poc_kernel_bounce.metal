// metal_poc_kernel_bounce.metal
// One bounce of primaryRayKernel's path: traceBounce() casts the ray and runs the steps in order (media, escape, hit, albedo, emission,
// the material's scattering, glass-medium bookkeeping, Russian roulette). Pure code motion out of the old kernel body; see
// metal_poc_kernel_state.metal.

// Scatters at the hit surface by its material (each shadeXxx() samples the next direction and adds the direct light). Returns false when the
// path ends (an absorbing or terminal event).
inline bool scatterAtSurface(thread const KernelRes& R, thread PathState& P, thread BounceState& B) {
    KERNEL_RES_ALIASES(R)
    PATH_STATE_ALIASES(P)
    BOUNCE_STATE_ALIASES(B)
    if (mat.materialType == METAL_MAT_DIELECTRIC) {
        if (!shadeDielectric(mat, hitPoint, normal, facingNormal, frontFace, result.distance,
                              rayDir, rayOrigin, throughput, specularBounce, rngState)) return false;
    } else if (mat.materialType == METAL_MAT_DISPERSIVE_DIELECTRIC) {
        if (!shadeDispersiveDielectric(mat, hitPoint, normal, facingNormal, frontFace, result.distance,
                              rayDir, rayOrigin, throughput, specularBounce, rngState, rgbChannel)) return false;
    } else if (mat.materialType == METAL_MAT_ROUGH_DIELECTRIC) {
        if (!shadeRoughDielectric(mat, hitPoint, normal, facingNormal, frontFace, result.distance,
                                   uniforms, lights, pbrtAreaLightTexture, textureSampler,
                                   isect, accelStructure, functionTable,
                                   rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState)) return false;
    } else if (mat.materialType == METAL_MAT_DISPERSIVE_ROUGH_DIELECTRIC) {
        if (!shadeDispersiveRoughDielectric(mat, hitPoint, normal, facingNormal, frontFace, result.distance,
                                   rayDir, rayOrigin, throughput, specularBounce, rngState, rgbChannel)) return false;
    } else if (mat.materialType == METAL_MAT_ROUGH_CONDUCTOR || mat.materialType == METAL_MAT_CHECKER_ROUGH_CONDUCTOR) {
        if (!shadeConductor(mat, hitPoint, normal, facingNormal, uniforms,
                             lights, pointLights, directionalLights, projectionLights, goniometricLights,
                             envMarginalCDF, envConditionalCDF, uniforms.envMapWidth, uniforms.envMapHeight,
                             pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                             ggxEnergyTable, uniforms.ggxEnergyRoughRes, uniforms.ggxEnergyMuRes,
                             earthTexture, pbrtEnvTexture, goniometricTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, textureSampler,
                             isect, accelStructure, functionTable,
                             rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState)) return false;
    } else if (mat.materialType == METAL_MAT_MIRROR) {
        if (!shadeMirror(albedo, hitPoint, facingNormal, rayDir, rayOrigin, throughput, specularBounce)) return false;
    } else if (mat.materialType == METAL_MAT_THIN_DIELECTRIC) {
        if (!shadeThinDielectric(mat, hitPoint, normal, facingNormal,
                                  rayDir, rayOrigin, throughput, specularBounce, rngState)) return false;
    } else if (mat.materialType == METAL_MAT_CLEARCOAT || mat.materialType == METAL_MAT_IMAGE_CLEARCOAT) {
        // materialType 27 (J1, section 172): the SAME clearcoat
        // shading materialType 8 already uses, just with `albedo`
        // (computed above) sourced from a real per-hit texture
        // sample instead of `mat.color` - shadeClearcoat() itself
        // needs no changes at all, it already takes `albedo` as
        // an explicit parameter rather than reading `mat.color`
        // directly.
        if (!shadeClearcoat(mat, albedo, hitPoint, facingNormal, uniforms,
                             lights, pointLights, directionalLights, projectionLights, goniometricLights,
                             envMarginalCDF, envConditionalCDF, uniforms.envMapWidth, uniforms.envMapHeight,
                             pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                             earthTexture, pbrtEnvTexture, goniometricTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, textureSampler,
                             isect, accelStructure, functionTable,
                             rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState)) return false;
    } else if (mat.materialType == METAL_MAT_DIFFUSE_TRANSMISSION) {
        if (!shadeDiffuseTransmission(mat, albedo, hitPoint, facingNormal, uniforms,
                                       lights, pointLights, directionalLights, projectionLights, goniometricLights,
                                       envMarginalCDF, envConditionalCDF, uniforms.envMapWidth, uniforms.envMapHeight,
                                       pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                                       earthTexture, pbrtEnvTexture, goniometricTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, textureSampler,
                                       isect, accelStructure, functionTable,
                                       rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState)) return false;
    } else if (mat.materialType == METAL_MAT_OREN_NAYAR) {
        if (!shadeOrenNayar(mat, albedo, hitPoint, facingNormal, uniforms,
                             lights, pointLights, directionalLights, projectionLights, goniometricLights,
                             envMarginalCDF, envConditionalCDF, uniforms.envMapWidth, uniforms.envMapHeight,
                             pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                             earthTexture, pbrtEnvTexture, goniometricTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, textureSampler,
                             isect, accelStructure, functionTable,
                             rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState)) return false;
    } else if (mat.materialType == METAL_MAT_VELVET) {
        if (!shadeVelvet(mat, albedo, hitPoint, facingNormal, uniforms,
                          lights, pointLights, directionalLights, projectionLights, goniometricLights,
                          envMarginalCDF, envConditionalCDF, uniforms.envMapWidth, uniforms.envMapHeight,
                          pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                          earthTexture, pbrtEnvTexture, goniometricTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, textureSampler,
                          isect, accelStructure, functionTable,
                          rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState)) return false;
    } else if (mat.materialType == METAL_MAT_PRINCIPLED) {
        if (!shadePrincipled(mat, hitPoint, facingNormal,
                          rayDir, rayOrigin, throughput, specularBounce, rngState)) return false;
    } else if (mat.materialType == METAL_MAT_MEASURED) {
        if (!shadeMeasured(mat, hitPoint, facingNormal, uniforms, rgbGridData,
                          lights, pointLights, directionalLights, projectionLights, goniometricLights,
                          envMarginalCDF, envConditionalCDF, uniforms.envMapWidth, uniforms.envMapHeight,
                          pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                          earthTexture, pbrtEnvTexture, goniometricTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, textureSampler,
                          isect, accelStructure, functionTable,
                          rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState)) return false;
    } else if (mat.materialType == METAL_MAT_SUBSURFACE) {
        if (!shadeSubsurface(R, P, B)) return false;
    } else if (mat.materialType == METAL_MAT_HAIR) {
        pathTouchedHair = true;
        if (!shadeHair(mat, hitPoint, facingNormal,
                          rayDir, rayOrigin, throughput, specularBounce, rngState)) return false;
    } else if (mat.materialType == METAL_MAT_NORMALIZED_FRESNEL) {
        if (!shadeNormalizedFresnel(mat, hitPoint, facingNormal, uniforms,
                          lights, pointLights, directionalLights, projectionLights, goniometricLights,
                          envMarginalCDF, envConditionalCDF, uniforms.envMapWidth, uniforms.envMapHeight,
                          pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                          earthTexture, pbrtEnvTexture, goniometricTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, textureSampler,
                          isect, accelStructure, functionTable,
                          rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState)) return false;
    } else if (mat.materialType == METAL_MAT_COATED_DIFFUSE) {
        if (!shadeCoatedDiffuse(mat, hitPoint, facingNormal, uniforms,
                          lights, pointLights, directionalLights, projectionLights, goniometricLights,
                          envMarginalCDF, envConditionalCDF, uniforms.envMapWidth, uniforms.envMapHeight,
                          pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                          earthTexture, pbrtEnvTexture, goniometricTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, textureSampler,
                          isect, accelStructure, functionTable,
                          rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState)) return false;
    } else if (mat.materialType == METAL_MAT_COATED_CONDUCTOR) {
        if (!shadeCoatedConductor(mat, hitPoint, facingNormal, uniforms,
                          lights, pointLights, directionalLights, projectionLights, goniometricLights,
                          envMarginalCDF, envConditionalCDF, uniforms.envMapWidth, uniforms.envMapHeight,
                          pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                          earthTexture, pbrtEnvTexture, goniometricTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, textureSampler,
                          isect, accelStructure, functionTable,
                          rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState)) return false;
    } else {
        if (!shadeLambertian(mat, albedo, hitPoint, facingNormal, uniforms,
                              lights, pointLights, directionalLights, projectionLights, goniometricLights,
                              envMarginalCDF, envConditionalCDF, uniforms.envMapWidth, uniforms.envMapHeight,
                              pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                              earthTexture, pbrtEnvTexture, goniometricTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, textureSampler,
                              isect, accelStructure, functionTable,
                              rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState)) return false;
    }
    return true;
}

// After a real bounce: starts a new segment for the emissive-hit MIS weight, and updates the glass-bounded medium state (entering or leaving a
// glass sphere that carries a medium, including the camera's own glass shell).
inline void updateMediumStateAfterBounce(thread const KernelRes& R, thread PathState& P, thread BounceState& B) {
    KERNEL_RES_ALIASES(R)
    PATH_STATE_ALIASES(P)
    BOUNCE_STATE_ALIASES(B)
        mediumSkippedDist = 0.0;  // a real bounce starts a new segment
        // After a dielectric event on a glass sphere that carries a medium: the path is now inside
        // the sphere iff its new direction points against the outward normal (true for refraction in,
        // reflection inside, and a thin dielectric passing straight through). `normal` is the sphere's
        // outward normal here.
        if (isSphere && mat.conductorK.y > 0.5 &&
            (mat.materialType == METAL_MAT_DIELECTRIC || mat.materialType == METAL_MAT_ROUGH_DIELECTRIC || mat.materialType == METAL_MAT_THIN_DIELECTRIC)) {
            const bool wasInGlass = inGlass;
            inGlass = dot(rayDir, normal) < 0.0;
            // Leaving a glass-medium sphere that sits INSIDE the camera's world-haze shell puts the path back in that shell's medium.
            const bool backInCameraGlass = !inGlass && uniforms.cameraGlassPrim > 0 && primId != uint(uniforms.cameraGlassPrim - 1);
            if (backInCameraGlass) {
                const TriangleMaterial camGlass = sphereMaterials[uniforms.cameraGlassPrim - 1];
                inGlass = true;
                glassSigmaT3 = float3(camGlass.conductorEta);
                glassG = camGlass.conductorK.x;
                glassAlbedo = float3(camGlass.transmitColor);
            } else if (inGlass) {
                glassSigmaT3 = float3(mat.conductorEta);
                glassG = mat.conductorK.x;
                glassAlbedo = float3(mat.transmitColor);
                // Entering from outside, or from the camera's world-haze shell into a nested sphere (a front-face hit on a sphere other than the shell).
                const bool enteringNested = wasInGlass && uniforms.cameraGlassPrim > 0 && frontFace && primId != uint(uniforms.cameraGlassPrim - 1);
                if (!wasInGlass || enteringNested) {
                    // Entering. A chromatic medium: follow ONE colour channel for this path (chosen
                    // uniformly, weight x3 on it) so free flight and shadow attenuation can use that
                    // channel's sigma_t; averaged over paths every channel is recovered. A grey medium
                    // needs no such colour noise.
                    if (fogChan >= 0) {
                        glassChan = fogChan;   // the camera medium already fixed this path's colour channel
                    } else if (enteringNested && glassChan >= 0) {
                        // the shell already fixed this path's hero channel; keep it
                    } else if (mat.conductorK.z > 0.5) {
                        glassChan = min(int(randFloat(rngState) * 3.0), 2);
                        float3 chanMask = float3(glassChan == 0 ? 3.0 : 0.0, glassChan == 1 ? 3.0 : 0.0, glassChan == 2 ? 3.0 : 0.0);
                        throughput *= chanMask;
                    } else {
                        glassChan = -1;
                    }
                }
            }
        }
}

// Traces one bounce of the path in `P`. Returns true if the path continues with another bounce, false when it ends (escaped, absorbed, at the
// depth limit, or killed by Russian roulette).
inline bool traceBounce(thread const KernelRes& R, thread PathState& P) {
    KERNEL_RES_ALIASES(R)
    PATH_STATE_ALIASES(P)
    BounceState B;
    BOUNCE_STATE_ALIASES(B)
    lastBounce = (depth >= uniforms.maxDepth);
    ray r;
    r.origin = rayOrigin;
    r.direction = rayDir;
    r.min_distance = 0.001f;
    r.max_distance = 1e6f;

    // F11/section 167: this sample's own shutterT (drawn once per
    // sample above, already used for camera motion blur) doubles
    // as the payload every intersect() call below passes to
    // sphereIntersectionFunction, so a moving sphere is tested at
    // the SAME simulated instant this ray's own camera position
    // was sampled at - a shadow ray cast later in this same
    // sample reuses the identical payload for the same reason.
    // isShadowRay=false explicit override (SpherePayload::
    // isShadowRay's own comment, section 176) - this is the ONE
    // ray that must actually be able to ENTER a materialType-28
    // medium sphere. shadowSpherePayload (used by this same
    // sample's own fog-NEE shadow rays just below) is the
    // opposite: isShadowRay=true, so a medium sphere stays
    // invisible to a pure occlusion test.
    spherePayload = SpherePayload{shutterT, false};
    shadowSpherePayload = SpherePayload{shutterT, true};
    result =
        isect.intersect(r, accelStructure, functionTable, spherePayload);
    mediumOriginBefore = rayOrigin;
    scatteredInMedium = false;
    passedThroughMediumSphere = false;
    if (!traverseBoundedMedia(R, P, B)) return false;
    if (passedThroughMediumSphere) mediumSkippedDist += length(rayOrigin - mediumOriginBefore);
    else if (scatteredInMedium) mediumSkippedDist = 0.0;

    if (!scatterInGlobalMedium(R, P, B)) return false;

    if (scatteredInMedium) mediumSkippedDist = 0.0;  // global-fog scatter: new segment
    if (scatteredInMedium) { CENSUS_PUSH(100u); fromMediumScatter = true; }
    if (!scatteredInMedium && !passedThroughMediumSphere) {
        if (result.type == intersection_type::none) {
            CENSUS_PUSH(255u);
            shadeEscapedRay(R, P, B);
            return false;
        }
        envLightSampled = false;   // a surface vertex: the environment is no longer sampled for this ray
        fromMediumScatter = false;
        resolveHit(R, P, B);
        perturbShadingNormal(R, P, B);
        computeAlbedo(R, P, B);
        addHitEmission(R, P, B);

        // The emission-only step (see traceBounce's caller): no scattering, the path ends.
        if (lastBounce) return false;

        if (!scatterAtSurface(R, P, B)) return false;
        updateMediumStateAfterBounce(R, P, B);
    }


    // Path-throughput ceiling: the CPU caps each channel of a path's throughput at 50 after every bounce (camera.h's
    // kMaxPathThroughput, also in OptiX) - the safety net for a BSDF whose per-bounce weight compounds. HairBxDF's sample
    // weight averages ~4 and a closed hair sphere gives several hair bounces in a row, so without the cap Metal rendered
    // the hair-sphere-dim-sky scene (K84) about 2x brighter than the CPU. Russian roulette only acts below 1.
    throughput = min(throughput, float3(50.0));

    // Russian roulette after a few bounces, same "let cheap paths
    // terminate early, keep expensive ones unbiased" shape as
    // this project's CPU integrator - throughput's max channel is
    // the survival probability, divided back in on survival so
    // the estimator stays unbiased. Clamped to 1.0 - a real,
    // previously-latent bug found via the realistic camera's own
    // noisy, non-converging first render: every earlier scene's
    // own `throughput` starts at EXACTLY (1,1,1) and only ever
    // SHRINKS via material albedo (<=1) multiplications, so
    // `max(throughput channels)` was always already <=1 there,
    // making this clamp an invisible no-op - but
    // `cameraRealistic`'s own `cameraWeight` (section 157) can
    // legitimately exceed 1.0 (a real pbrt-v4 importance-sampling
    // weight, unbounded by design), so `throughput` can too.
    // WITHOUT the clamp, a >1 `p` still guarantees survival
    // (`randFloat() > p` is never true for p>=1) but ALSO
    // divides `throughput` by that same large `p` anyway - an
    // unnecessary, UNCOMPENSATED shrink standard RR never
    // applies once survival is already certain (the division
    // exists ONLY to compensate for paths that DO die, keeping
    // the estimator unbiased when survival is a coin flip; once
    // survival is guaranteed, no compensation is needed at all).
    // Left uncorrected, every sample's own bright, unbounded
    // `cameraWeight` got silently and inconsistently divided
    // back down partway through its own path - exactly the
    // per-sample-inconsistent, non-converging speckle the first
    // render showed.
    if (depth > 3) {
        float p = min(max(throughput.x, max(throughput.y, throughput.z)), 1.0);
        if (randFloat(rngState) > p) return false;
        throughput /= max(p, 0.0001);
    }
    return true;
}
