inline bool shadeDiffuseTransmission(TriangleMaterial mat, float3 albedo, float3 hitPoint, float3 facingNormal,
                                      constant Uniforms& uniforms,
                                      device const AreaLight* lights,
                                      device const PointLight* pointLights,
                                      device const DirectionalLight* directionalLights,
                                      device const ProjectionLight* projectionLights,
                                      device const GoniometricLight* goniometricLights,
                                      device const float* envMarginalCDF,
                                      device const float* envConditionalCDF,
                                      uint envMapWidth, uint envMapHeight,
                                      device const float* pbrtEnvMarginalCDF,
                                      device const float* pbrtEnvConditionalCDF,
                                      uint pbrtEnvMapWidth, uint pbrtEnvMapHeight,
                                      texture2d<float, access::sample> earthTexture,
                                      texture2d<float, access::sample> pbrtEnvTexture,
                                      texture2d<float, access::sample> goniometricTexture,
                                      texture2d<float, access::sample> pbrtGoniometricTexture,
                                      texture2d<float, access::sample> pbrtProjectionTexture,
                                      texture2d<float, access::sample> pbrtAreaLightTexture,
                                      sampler textureSampler,
                                      intersector<instancing, triangle_data> isect,
                                      instance_acceleration_structure accelStructure,
                                      intersection_function_table<instancing, triangle_data> functionTable,
                                      thread float3& rayDir, thread float3& rayOrigin,
                                      thread float3& throughput, thread float3& radiance,
                                      thread float& bsdfPdf, thread bool& specularBounce, thread uint& rngState) {
    // Diffuse transmission (a two-sided translucent diffuser - paper, a
    // leaf, a thin frosted panel): pbrt-v4's own DiffuseTransmissionBxDF.
    // `albedo` (this material's own `color`) is the REFLECTANCE tint;
    // `mat.transmitColor` the TRANSMITTANCE tint. For any ONE hit point,
    // a given light/continuation direction falls on exactly one side of
    // `facingNormal` - which lobe applies is decided by that single
    // sign, not two separate passes over each light. See section 67.
    float pr = max(albedo.x, max(albedo.y, albedo.z));
    float pt = max(mat.transmitColor.x, max(mat.transmitColor.y, mat.transmitColor.z));
    float pSum = max(pr + pt, 1e-6);

    if (all(mat.emission == float3(0.0))) {
        LightSample ls = sampleAreaLight(lights, uniforms.lightCount, rngState, pbrtAreaLightTexture, textureSampler);
        float3 toLight = ls.point - hitPoint;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        float3 wi = toLight / dist;
        float cosSurface = dot(facingNormal, wi);
        float cosLight = dot(ls.normal, -wi);
        if (cosSurface != 0.0 && (cosLight > 0.0 || (ls.twoSided != 0.0 && cosLight < 0.0))) {
            bool reflect = cosSurface > 0.0;
            float3 lobeTint = reflect ? albedo : mat.transmitColor;
            float lobeProb = reflect ? (pr / pSum) : (pt / pSum);
            float absCos = abs(cosSurface);
            ray shadowRay;
            shadowRay.origin = hitPoint + (reflect ? facingNormal : -facingNormal) * 0.001f;
            shadowRay.direction = wi;
            shadowRay.min_distance = 0.001f;
            shadowRay.max_distance = dist - 0.002f;
            intersection_result<instancing, triangle_data> shadowResult =
                isect.intersect(shadowRay, accelStructure, functionTable);
            if (shadowResult.type == intersection_type::none) {
                float pdfSolidAngle = (distSq / (ls.area * abs(cosLight))) * ls.pmf;
                float pdfBsdfForThisDir = lobeProb * absCos / M_PI_F;
                float weight = (pdfSolidAngle * pdfSolidAngle)
                    / (pdfSolidAngle * pdfSolidAngle + pdfBsdfForThisDir * pdfBsdfForThisDir);
                float transmittance = exp(-uniforms.fogSigmaT * dist);
                radiance += throughput * lobeTint * (1.0 / M_PI_F)
                            * ls.emission * absCos * transmittance / pdfSolidAngle * weight;
            }
        }

        for (uint pli = 0; pli < uniforms.pointLightCount; ++pli) {
            PointLight pl = pointLights[pli];
            float3 toPointLight = float3(pl.position) - hitPoint;
            float plDistSq = dot(toPointLight, toPointLight);
            float plDist = sqrt(plDistSq);
            float3 plWi = toPointLight / plDist;
            float plCosSurface = dot(facingNormal, plWi);
            if (plCosSurface != 0.0) {
                bool plReflect = plCosSurface > 0.0;
                float3 plLobeTint = plReflect ? albedo : mat.transmitColor;
                float plAbsCos = abs(plCosSurface);
                ray plShadowRay;
                plShadowRay.origin = hitPoint + (plReflect ? facingNormal : -facingNormal) * 0.001f;
                plShadowRay.direction = plWi;
                plShadowRay.min_distance = 0.001f;
                plShadowRay.max_distance = plDist - 0.002f;
                intersection_result<instancing, triangle_data> plShadowResult =
                    isect.intersect(plShadowRay, accelStructure, functionTable);
                if (plShadowResult.type == intersection_type::none) {
                    float plTransmittance = exp(-uniforms.fogSigmaT * plDist);
                    float plSpot = spotLightFalloff(-plWi, float3(pl.direction), pl.cosOuterAngle, pl.cosInnerAngle);
                    radiance += throughput * plLobeTint * (1.0 / M_PI_F)
                                * float3(pl.emission) * plAbsCos * plSpot * plTransmittance / plDistSq;
                }
            }
        }

        for (uint dli = 0; dli < uniforms.directionalLightCount; ++dli) {
            DirectionalLight dl = directionalLights[dli];
            float3 dlWi = normalize(-float3(dl.direction));
            float dlCosSurface = dot(facingNormal, dlWi);
            if (dlCosSurface != 0.0) {
                bool dlReflect = dlCosSurface > 0.0;
                float3 dlLobeTint = dlReflect ? albedo : mat.transmitColor;
                float dlAbsCos = abs(dlCosSurface);
                ray dlShadowRay;
                dlShadowRay.origin = hitPoint + (dlReflect ? facingNormal : -facingNormal) * 0.001f;
                dlShadowRay.direction = dlWi;
                dlShadowRay.min_distance = 0.001f;
                dlShadowRay.max_distance = kDirectionalLightMaxDistance;
                intersection_result<instancing, triangle_data> dlShadowResult =
                    isect.intersect(dlShadowRay, accelStructure, functionTable);
                if (dlShadowResult.type == intersection_type::none) {
                    // Skip when there is no fog - rayBoxExitDistance()
                    // is only valid for an origin INSIDE the hardcoded
                    // room bounds (see that function.s own comment for
                    // why calling it unconditionally is unsafe).
                    float dlTransmittance = (uniforms.fogSigmaT > 0.0)
                        ? exp(-uniforms.fogSigmaT * rayBoxExitDistance(dlShadowRay.origin, dlWi, kRoomBoundsMin, kRoomBoundsMax))
                        : 1.0;
                    radiance += throughput * dlLobeTint * (1.0 / M_PI_F)
                                * float3(dl.emission) * dlAbsCos * dlTransmittance;
                }
            }
        }

        for (uint pji = 0; pji < uniforms.projectionLightCount; ++pji) {
            ProjectionLight pj = projectionLights[pji];
            float3 toProjLight = float3(pj.position) - hitPoint;
            float pjDistSq = dot(toProjLight, toProjLight);
            float pjDist = sqrt(pjDistSq);
            float3 pjWi = toProjLight / pjDist;
            float pjCosSurface = dot(facingNormal, pjWi);
            if (pjCosSurface != 0.0) {
                float3 pjRadiance = projectionLightRadiance(-pjWi, pj.forward, pj.right, pj.up,
                                                             pj.tanHalfFovX, pj.tanHalfFovY, pj.scale,
                                                             (pj.usePbrtTexture != 0u ? pbrtProjectionTexture : earthTexture), textureSampler);
                if (any(pjRadiance > float3(0.0))) {
                    bool pjReflect = pjCosSurface > 0.0;
                    float3 pjLobeTint = pjReflect ? albedo : mat.transmitColor;
                    float pjAbsCos = abs(pjCosSurface);
                    ray pjShadowRay;
                    pjShadowRay.origin = hitPoint + (pjReflect ? facingNormal : -facingNormal) * 0.001f;
                    pjShadowRay.direction = pjWi;
                    pjShadowRay.min_distance = 0.001f;
                    pjShadowRay.max_distance = pjDist - 0.002f;
                    intersection_result<instancing, triangle_data> pjShadowResult =
                        isect.intersect(pjShadowRay, accelStructure, functionTable);
                    if (pjShadowResult.type == intersection_type::none) {
                        float pjTransmittance = exp(-uniforms.fogSigmaT * pjDist);
                        radiance += throughput * pjLobeTint * (1.0 / M_PI_F)
                                    * pjRadiance * pjAbsCos * pjTransmittance / pjDistSq;
                    }
                }
            }
        }

        for (uint gli = 0; gli < uniforms.goniometricLightCount; ++gli) {
            GoniometricLight gl = goniometricLights[gli];
            float3 toGoniLight = float3(gl.position) - hitPoint;
            float glDistSq = dot(toGoniLight, toGoniLight);
            float glDist = sqrt(glDistSq);
            float3 glWi = toGoniLight / glDist;
            float glCosSurface = dot(facingNormal, glWi);
            if (glCosSurface != 0.0) {
                float3 glRadiance = goniometricLightRadiance(-glWi, gl.forward, gl.right, gl.up,
                                                              gl.emission, gl.scale,
                                                              (gl.usePbrtTexture != 0u ? pbrtGoniometricTexture : goniometricTexture), textureSampler);
                if (any(glRadiance > float3(0.0))) {
                    bool glReflect = glCosSurface > 0.0;
                    float3 glLobeTint = glReflect ? albedo : mat.transmitColor;
                    float glAbsCos = abs(glCosSurface);
                    ray glShadowRay;
                    glShadowRay.origin = hitPoint + (glReflect ? facingNormal : -facingNormal) * 0.001f;
                    glShadowRay.direction = glWi;
                    glShadowRay.min_distance = 0.001f;
                    glShadowRay.max_distance = glDist - 0.002f;
                    intersection_result<instancing, triangle_data> glShadowResult =
                        isect.intersect(glShadowRay, accelStructure, functionTable);
                    if (glShadowResult.type == intersection_type::none) {
                        float glTransmittance = exp(-uniforms.fogSigmaT * glDist);
                        radiance += throughput * glLobeTint * (1.0 / M_PI_F)
                                    * glRadiance * glAbsCos * glTransmittance / glDistSq;
                    }
                }
            }
        }

        // Environment map (importance-sampled NEE, section 71) - see
        // shadeLambertian's own comment for the full "why." Same signed-
        // lobe-pick shape as every other light above in this function:
        // the sampled direction can land on EITHER side of `facingNormal`
        // (unlike a fixed-position light, whose side is decided once per
        // hit point), so which lobe/tint/pdf-weight/shadow-ray-offset
        // applies is decided by that sign, exactly as above.
        // Gated on useEnvironmentMap too, not just envMapWidth - see
        // shadeConductor's own comment.
        if (envMapWidth > 0u && uniforms.useEnvironmentMap != 0u) {
            float envPdfSolidAngle;
            float3 envWi = sampleEnvironmentDirection(envMarginalCDF, envConditionalCDF,
                                                       int(envMapWidth), int(envMapHeight),
                                                       randFloat(rngState), randFloat(rngState), envPdfSolidAngle);
            float envCosSurface = dot(facingNormal, envWi);
            if (envCosSurface != 0.0 && envPdfSolidAngle > 1e-9) {
                bool envReflect = envCosSurface > 0.0;
                float3 envLobeTint = envReflect ? albedo : mat.transmitColor;
                float envLobeProb = envReflect ? (pr / pSum) : (pt / pSum);
                float envAbsCos = abs(envCosSurface);
                ray envShadowRay;
                envShadowRay.origin = hitPoint + (envReflect ? facingNormal : -facingNormal) * 0.001f;
                envShadowRay.direction = envWi;
                envShadowRay.min_distance = 0.001f;
                envShadowRay.max_distance = 1e5f;
                intersection_result<instancing, triangle_data> envShadowResult =
                    isect.intersect(envShadowRay, accelStructure, functionTable);
                if (envShadowResult.type == intersection_type::none) {
                    float2 envUV = equirectangularUV(envWi);
                    float3 envRadiance = earthTexture.sample(textureSampler, envUV).rgb;
                    float envPdfBsdf = envLobeProb * envAbsCos / M_PI_F;
                    float envWeight = (envPdfSolidAngle * envPdfSolidAngle)
                        / (envPdfSolidAngle * envPdfSolidAngle + envPdfBsdf * envPdfBsdf);
                    radiance += throughput * envLobeTint * (1.0 / M_PI_F)
                                * envRadiance * envAbsCos / envPdfSolidAngle * envWeight;
                }
            }
        }

        // Same NEE/MIS strategy, for a pbrt-loaded scene's own SEPARATE
        // image-based infinite light (section 96) - see shadeConductor's
        // own comment. Same signed-lobe-pick shape as the block above.
        if (pbrtEnvMapWidth > 0u) {
            float pbrtEnvPdfSolidAngle;
            float3 pbrtEnvWi = sampleEnvironmentDirection(pbrtEnvMarginalCDF, pbrtEnvConditionalCDF,
                                                           int(pbrtEnvMapWidth), int(pbrtEnvMapHeight),
                                                           randFloat(rngState), randFloat(rngState), pbrtEnvPdfSolidAngle);
            float pbrtEnvCosSurface = dot(facingNormal, pbrtEnvWi);
            if (pbrtEnvCosSurface != 0.0 && pbrtEnvPdfSolidAngle > 1e-9) {
                bool pbrtEnvReflect = pbrtEnvCosSurface > 0.0;
                float3 pbrtEnvLobeTint = pbrtEnvReflect ? albedo : mat.transmitColor;
                float pbrtEnvLobeProb = pbrtEnvReflect ? (pr / pSum) : (pt / pSum);
                float pbrtEnvAbsCos = abs(pbrtEnvCosSurface);
                ray pbrtEnvShadowRay;
                pbrtEnvShadowRay.origin = hitPoint + (pbrtEnvReflect ? facingNormal : -facingNormal) * 0.001f;
                pbrtEnvShadowRay.direction = pbrtEnvWi;
                pbrtEnvShadowRay.min_distance = 0.001f;
                pbrtEnvShadowRay.max_distance = 1e5f;
                intersection_result<instancing, triangle_data> pbrtEnvShadowResult =
                    isect.intersect(pbrtEnvShadowRay, accelStructure, functionTable);
                if (pbrtEnvShadowResult.type == intersection_type::none) {
                    float2 pbrtEnvUV = equirectangularUV(pbrtEnvWi);
                    float3 pbrtEnvRadianceSample = pbrtEnvTexture.sample(textureSampler, pbrtEnvUV).rgb;
                    float pbrtEnvPdfBsdf = pbrtEnvLobeProb * pbrtEnvAbsCos / M_PI_F;
                    float pbrtEnvWeight = (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle)
                        / (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle + pbrtEnvPdfBsdf * pbrtEnvPdfBsdf);
                    radiance += throughput * pbrtEnvLobeTint * (1.0 / M_PI_F)
                                * pbrtEnvRadianceSample * pbrtEnvAbsCos / pbrtEnvPdfSolidAngle * pbrtEnvWeight;
                }
            }
        }
    }

    bool reflect = randFloat(rngState) < (pr / pSum);
    float3 lobeNormal = reflect ? facingNormal : -facingNormal;
    rayDir = cosineSampleHemisphere(lobeNormal, rngState);
    rayOrigin = hitPoint + lobeNormal * 0.001f;
    throughput *= reflect ? albedo : mat.transmitColor;
    bsdfPdf = (reflect ? (pr / pSum) : (pt / pSum)) * max(dot(lobeNormal, rayDir), 0.0001) / M_PI_F;
    specularBounce = false;
    return true;
}

inline bool shadeLambertian(TriangleMaterial mat, float3 albedo, float3 hitPoint, float3 facingNormal,
                             constant Uniforms& uniforms,
                             device const AreaLight* lights,
                             device const PointLight* pointLights,
                             device const DirectionalLight* directionalLights,
                             device const ProjectionLight* projectionLights,
                             device const GoniometricLight* goniometricLights,
                             device const float* envMarginalCDF,
                             device const float* envConditionalCDF,
                             uint envMapWidth, uint envMapHeight,
                             device const float* pbrtEnvMarginalCDF,
                             device const float* pbrtEnvConditionalCDF,
                             uint pbrtEnvMapWidth, uint pbrtEnvMapHeight,
                             texture2d<float, access::sample> earthTexture,
                             texture2d<float, access::sample> pbrtEnvTexture,
                             texture2d<float, access::sample> goniometricTexture,
                             texture2d<float, access::sample> pbrtGoniometricTexture,
                             texture2d<float, access::sample> pbrtProjectionTexture,
                             texture2d<float, access::sample> pbrtAreaLightTexture,
                             sampler textureSampler,
                             intersector<instancing, triangle_data> isect,
                             instance_acceleration_structure accelStructure,
                             intersection_function_table<instancing, triangle_data> functionTable,
                             thread float3& rayDir, thread float3& rayOrigin,
                             thread float3& throughput, thread float3& radiance,
                             thread float& bsdfPdf, thread bool& specularBounce, thread uint& rngState) {
    // Lambertian (materialType 0, or 3/6/7/10 - the only difference
    // already resolved upstream into `albedo`, this BSDF/NEE math has no
    // idea where albedo came from): next-event estimation against a
    // randomly picked light, then continue the path via cosine-weighted
    // hemisphere sampling for indirect light.
    if (all(mat.emission == float3(0.0))) {
        LightSample ls = sampleAreaLight(lights, uniforms.lightCount, rngState, pbrtAreaLightTexture, textureSampler);
        float3 toLight = ls.point - hitPoint;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        float3 wi = toLight / dist;
        float cosSurface = dot(facingNormal, wi);
        float cosLight = dot(ls.normal, -wi);
        if (cosSurface > 0.0 && (cosLight > 0.0 || (ls.twoSided != 0.0 && cosLight < 0.0))) {
            ray shadowRay;
            shadowRay.origin = hitPoint + facingNormal * 0.001f;
            shadowRay.direction = wi;
            shadowRay.min_distance = 0.001f;
            shadowRay.max_distance = dist - 0.002f;
            intersection_result<instancing, triangle_data> shadowResult =
                isect.intersect(shadowRay, accelStructure, functionTable);
            if (shadowResult.type == intersection_type::none) {
                float pdfSolidAngle = (distSq / (ls.area * abs(cosLight))) * ls.pmf;
                float pdfBsdfForThisDir = cosSurface / M_PI_F;
                float weight = (pdfSolidAngle * pdfSolidAngle)
                    / (pdfSolidAngle * pdfSolidAngle + pdfBsdfForThisDir * pdfBsdfForThisDir);
                float transmittance = exp(-uniforms.fogSigmaT * dist);
                radiance += throughput * albedo * (1.0 / M_PI_F)
                            * ls.emission * cosSurface * transmittance / pdfSolidAngle * weight;
            }
        }

        for (uint pli = 0; pli < uniforms.pointLightCount; ++pli) {
            PointLight pl = pointLights[pli];
            float3 toPointLight = float3(pl.position) - hitPoint;
            float plDistSq = dot(toPointLight, toPointLight);
            float plDist = sqrt(plDistSq);
            float3 plWi = toPointLight / plDist;
            float plCosSurface = dot(facingNormal, plWi);
            if (plCosSurface > 0.0) {
                ray plShadowRay;
                plShadowRay.origin = hitPoint + facingNormal * 0.001f;
                plShadowRay.direction = plWi;
                plShadowRay.min_distance = 0.001f;
                plShadowRay.max_distance = plDist - 0.002f;
                intersection_result<instancing, triangle_data> plShadowResult =
                    isect.intersect(plShadowRay, accelStructure, functionTable);
                if (plShadowResult.type == intersection_type::none) {
                    float plTransmittance = exp(-uniforms.fogSigmaT * plDist);
                    float plSpot = spotLightFalloff(-plWi, float3(pl.direction), pl.cosOuterAngle, pl.cosInnerAngle);
                    radiance += throughput * albedo * (1.0 / M_PI_F)
                                * float3(pl.emission) * plCosSurface * plSpot * plTransmittance / plDistSq;
                }
            }
        }

        for (uint dli = 0; dli < uniforms.directionalLightCount; ++dli) {
            DirectionalLight dl = directionalLights[dli];
            float3 dlWi = normalize(-float3(dl.direction));
            float dlCosSurface = dot(facingNormal, dlWi);
            if (dlCosSurface > 0.0) {
                ray dlShadowRay;
                dlShadowRay.origin = hitPoint + facingNormal * 0.001f;
                dlShadowRay.direction = dlWi;
                dlShadowRay.min_distance = 0.001f;
                dlShadowRay.max_distance = kDirectionalLightMaxDistance;
                intersection_result<instancing, triangle_data> dlShadowResult =
                    isect.intersect(dlShadowRay, accelStructure, functionTable);
                if (dlShadowResult.type == intersection_type::none) {
                    // Skip when there is no fog - rayBoxExitDistance()
                    // is only valid for an origin INSIDE the hardcoded
                    // room bounds (see that function.s own comment for
                    // why calling it unconditionally is unsafe).
                    float dlTransmittance = (uniforms.fogSigmaT > 0.0)
                        ? exp(-uniforms.fogSigmaT * rayBoxExitDistance(dlShadowRay.origin, dlWi, kRoomBoundsMin, kRoomBoundsMax))
                        : 1.0;
                    radiance += throughput * albedo * (1.0 / M_PI_F)
                                * float3(dl.emission) * dlCosSurface * dlTransmittance;
                }
            }
        }

        for (uint pji = 0; pji < uniforms.projectionLightCount; ++pji) {
            ProjectionLight pj = projectionLights[pji];
            float3 toProjLight = float3(pj.position) - hitPoint;
            float pjDistSq = dot(toProjLight, toProjLight);
            float pjDist = sqrt(pjDistSq);
            float3 pjWi = toProjLight / pjDist;
            float pjCosSurface = dot(facingNormal, pjWi);
            if (pjCosSurface > 0.0) {
                float3 pjRadiance = projectionLightRadiance(-pjWi, pj.forward, pj.right, pj.up,
                                                             pj.tanHalfFovX, pj.tanHalfFovY, pj.scale,
                                                             (pj.usePbrtTexture != 0u ? pbrtProjectionTexture : earthTexture), textureSampler);
                if (any(pjRadiance > float3(0.0))) {
                    ray pjShadowRay;
                    pjShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    pjShadowRay.direction = pjWi;
                    pjShadowRay.min_distance = 0.001f;
                    pjShadowRay.max_distance = pjDist - 0.002f;
                    intersection_result<instancing, triangle_data> pjShadowResult =
                        isect.intersect(pjShadowRay, accelStructure, functionTable);
                    if (pjShadowResult.type == intersection_type::none) {
                        float pjTransmittance = exp(-uniforms.fogSigmaT * pjDist);
                        radiance += throughput * albedo * (1.0 / M_PI_F)
                                    * pjRadiance * pjCosSurface * pjTransmittance / pjDistSq;
                    }
                }
            }
        }

        for (uint gli = 0; gli < uniforms.goniometricLightCount; ++gli) {
            GoniometricLight gl = goniometricLights[gli];
            float3 toGoniLight = float3(gl.position) - hitPoint;
            float glDistSq = dot(toGoniLight, toGoniLight);
            float glDist = sqrt(glDistSq);
            float3 glWi = toGoniLight / glDist;
            float glCosSurface = dot(facingNormal, glWi);
            if (glCosSurface > 0.0) {
                float3 glRadiance = goniometricLightRadiance(-glWi, gl.forward, gl.right, gl.up,
                                                              gl.emission, gl.scale,
                                                              (gl.usePbrtTexture != 0u ? pbrtGoniometricTexture : goniometricTexture), textureSampler);
                if (any(glRadiance > float3(0.0))) {
                    ray glShadowRay;
                    glShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    glShadowRay.direction = glWi;
                    glShadowRay.min_distance = 0.001f;
                    glShadowRay.max_distance = glDist - 0.002f;
                    intersection_result<instancing, triangle_data> glShadowResult =
                        isect.intersect(glShadowRay, accelStructure, functionTable);
                    if (glShadowResult.type == intersection_type::none) {
                        float glTransmittance = exp(-uniforms.fogSigmaT * glDist);
                        radiance += throughput * albedo * (1.0 / M_PI_F)
                                    * glRadiance * glCosSurface * glTransmittance / glDistSq;
                    }
                }
            }
        }

        // Environment map (importance-sampled NEE, section 71): a NEW
        // light-sampling strategy for `earthTexture`'s own equirectangular
        // sample - previously reachable only via a BSDF-sampled ray that
        // happened to escape toward a bright region (high variance under
        // a small/bright environment feature, exactly the problem NEE/MIS
        // already solves for every light type above). Samples a
        // direction from the environment image's own importance
        // distribution (phase 1, section 69 - bright regions picked more
        // often), checks visibility with a shadow ray toward "infinity"
        // (`1e5f`, matching this scene's own room-scale units), and MIS-
        // weights against this material's own cosine-weighted BSDF pdf
        // for that same direction - the same NEE/MIS shape the area
        // light above already uses, just with a direction-dependent (not
        // point) light source and an importance-sampled (not uniform)
        // sampling strategy.
        // Gated on useEnvironmentMap too, not just envMapWidth - see
        // shadeConductor's own comment.
        if (envMapWidth > 0u && uniforms.useEnvironmentMap != 0u) {
            float envPdfSolidAngle;
            float3 envWi = sampleEnvironmentDirection(envMarginalCDF, envConditionalCDF,
                                                       int(envMapWidth), int(envMapHeight),
                                                       randFloat(rngState), randFloat(rngState), envPdfSolidAngle);
            float envCosSurface = dot(facingNormal, envWi);
            if (envCosSurface > 0.0 && envPdfSolidAngle > 1e-9) {
                ray envShadowRay;
                envShadowRay.origin = hitPoint + facingNormal * 0.001f;
                envShadowRay.direction = envWi;
                envShadowRay.min_distance = 0.001f;
                envShadowRay.max_distance = 1e5f;
                intersection_result<instancing, triangle_data> envShadowResult =
                    isect.intersect(envShadowRay, accelStructure, functionTable);
                if (envShadowResult.type == intersection_type::none) {
                    float2 envUV = equirectangularUV(envWi);
                    float3 envRadiance = earthTexture.sample(textureSampler, envUV).rgb;
                    float envPdfBsdf = envCosSurface / M_PI_F;
                    float envWeight = (envPdfSolidAngle * envPdfSolidAngle)
                        / (envPdfSolidAngle * envPdfSolidAngle + envPdfBsdf * envPdfBsdf);
                    radiance += throughput * albedo * (1.0 / M_PI_F)
                                * envRadiance * envCosSurface / envPdfSolidAngle * envWeight;
                }
            }
        }

        // Same NEE/MIS strategy, for a pbrt-loaded scene's own SEPARATE
        // image-based infinite light (section 96) - see shadeConductor's
        // own comment.
        if (pbrtEnvMapWidth > 0u) {
            float pbrtEnvPdfSolidAngle;
            float3 pbrtEnvWi = sampleEnvironmentDirection(pbrtEnvMarginalCDF, pbrtEnvConditionalCDF,
                                                           int(pbrtEnvMapWidth), int(pbrtEnvMapHeight),
                                                           randFloat(rngState), randFloat(rngState), pbrtEnvPdfSolidAngle);
            float pbrtEnvCosSurface = dot(facingNormal, pbrtEnvWi);
            if (pbrtEnvCosSurface > 0.0 && pbrtEnvPdfSolidAngle > 1e-9) {
                ray pbrtEnvShadowRay;
                pbrtEnvShadowRay.origin = hitPoint + facingNormal * 0.001f;
                pbrtEnvShadowRay.direction = pbrtEnvWi;
                pbrtEnvShadowRay.min_distance = 0.001f;
                pbrtEnvShadowRay.max_distance = 1e5f;
                intersection_result<instancing, triangle_data> pbrtEnvShadowResult =
                    isect.intersect(pbrtEnvShadowRay, accelStructure, functionTable);
                if (pbrtEnvShadowResult.type == intersection_type::none) {
                    float2 pbrtEnvUV = equirectangularUV(pbrtEnvWi);
                    float3 pbrtEnvRadianceSample = pbrtEnvTexture.sample(textureSampler, pbrtEnvUV).rgb;
                    float pbrtEnvPdfBsdf = pbrtEnvCosSurface / M_PI_F;
                    float pbrtEnvWeight = (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle)
                        / (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle + pbrtEnvPdfBsdf * pbrtEnvPdfBsdf);
                    radiance += throughput * albedo * (1.0 / M_PI_F)
                                * pbrtEnvRadianceSample * pbrtEnvCosSurface / pbrtEnvPdfSolidAngle * pbrtEnvWeight;
                }
            }
        }
    }

    rayDir = cosineSampleHemisphere(facingNormal, rngState);
    rayOrigin = hitPoint + facingNormal * 0.001f;
    throughput *= albedo;
    bsdfPdf = max(dot(facingNormal, rayDir), 0.0001) / M_PI_F;
    specularBounce = false;
    return true;
}

// "Improved Oren-Nayar" rough diffuse (Fujii's reformulation of Oren &
// Nayar 1994, https://mimosa-pudica.net/improved-oren-nayar.html) -
// found via Blender Cycles as reference (kernel/closure/
// bsdf_oren_nayar.h), where it's the modern replacement for the
// classic model. A rough (not perfectly Lambertian) diffuse surface -
// clay, plaster, the Moon's own regolith - reads visibly FLATTER/less
// shaded than ideal Lambertian: light and view directions both near
// grazing (and roughly aligned in azimuth) brighten noticeably (the
// classic "flat full moon" retroreflective look, real microfacet
// self-shadowing/masking within the rough surface concentrating
// reflected light back toward the source), while head-on illumination
// reads slightly DARKER than Lambertian - a real energy redistribution,
// not just a brightness knob. `mat.roughness` doubles as this
// material's own sigma parameter (materialType 5/4/7/9/10 already
// reuse this same field their own way, per TriangleMaterial's own
// comment - one more reuse, not a new struct field).
//
// Deliberately scoped to the SINGLE-scatter term only - Cycles' own
// version adds a further energy-preserving MULTI-scatter compensation
// term (OpenPBR-spec-based) on top of this, explicitly deferred here
// (the same "close the bigger, more visible gap first" staging this
// POC's own GGX energy-compensation work, sections 72/73, already
// used) - `sigma == 0` still reduces EXACTLY to plain Lambertian
// (`a == 1/pi`, `b == 0`), verified below, so this is a strict
// generalization, not a replacement with different edge-case behaviour.
