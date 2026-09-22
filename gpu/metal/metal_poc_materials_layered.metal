inline float orenNayarF(float3 wo, float3 wi, float3 n, float sigma) {
    float nl = max(dot(n, wi), 0.0);
    float nv = max(dot(n, wo), 0.0);
    float a = 1.0 / (M_PI_F + sigma * (M_PI_F * 0.5 - 2.0 / 3.0));
    float b = sigma * a;
    if (b <= 0.0) {
        return a;
    }
    float t = dot(wi, wo) - nl * nv;
    if (t > 0.0) {
        t /= max(max(nl, nv), 1e-6);
    }
    return a + b * t;
}

inline bool shadeOrenNayar(TriangleMaterial mat, float3 albedo, float3 hitPoint, float3 facingNormal,
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
    float3 woWorld = -rayDir;

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
                float pdfBsdfForThisDir = lambertianPdf(cosSurface);
                float weight = (pdfSolidAngle * pdfSolidAngle)
                    / (pdfSolidAngle * pdfSolidAngle + pdfBsdfForThisDir * pdfBsdfForThisDir);
                float transmittance = exp(-uniforms.fogSigmaT * dist);
                radiance += throughput * albedo * orenNayarF(woWorld, wi, facingNormal, mat.roughness)
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
                    radiance += throughput * albedo * orenNayarF(woWorld, plWi, facingNormal, mat.roughness)
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
                    radiance += throughput * albedo * orenNayarF(woWorld, dlWi, facingNormal, mat.roughness)
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
                        radiance += throughput * albedo * orenNayarF(woWorld, pjWi, facingNormal, mat.roughness)
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
                        radiance += throughput * albedo * orenNayarF(woWorld, glWi, facingNormal, mat.roughness)
                                    * glRadiance * glCosSurface * glTransmittance / glDistSq;
                    }
                }
            }
        }

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
                    float envPdfBsdf = lambertianPdf(envCosSurface);
                    float envWeight = (envPdfSolidAngle * envPdfSolidAngle)
                        / (envPdfSolidAngle * envPdfSolidAngle + envPdfBsdf * envPdfBsdf);
                    radiance += throughput * albedo * orenNayarF(woWorld, envWi, facingNormal, mat.roughness)
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
                    float pbrtEnvPdfBsdf = lambertianPdf(pbrtEnvCosSurface);
                    float pbrtEnvWeight = (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle)
                        / (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle + pbrtEnvPdfBsdf * pbrtEnvPdfBsdf);
                    radiance += throughput * albedo * orenNayarF(woWorld, pbrtEnvWi, facingNormal, mat.roughness)
                                * pbrtEnvRadianceSample * pbrtEnvCosSurface / pbrtEnvPdfSolidAngle * pbrtEnvWeight;
                }
            }
        }
    }

    // Continuation ray: still plain cosine-weighted hemisphere sampling
    // (NOT importance-sampled to this BRDF's own a+b*t shape - the same
    // simplification Cycles' own bsdf_oren_nayar_sample() makes too),
    // so the pdf stays cosTheta/pi exactly like Lambertian. The MC
    // weight f*cosTheta/pdf therefore collapses to
    // `albedo*orenNayarF(...)*pi` - Lambertian's own `throughput *=
    // albedo` is the special case of this at sigma == 0, where
    // orenNayarF() returns the constant `1/pi` and the two `pi`s cancel
    // back to exactly `albedo`.
    float3 newDir = cosineSampleHemisphere(facingNormal, rngState);
    rayDir = newDir;
    rayOrigin = hitPoint + facingNormal * 0.001f;
    throughput *= albedo * orenNayarF(woWorld, newDir, facingNormal, mat.roughness) * M_PI_F;
    bsdfPdf = lambertianPdf(max(dot(facingNormal, newDir), 0.0001f));
    specularBounce = false;
    return true;
}

// pbrt-v4's own NormalizedFresnelBxDF (src/shared/bxdfs_layered.h) -
// Fresnel-WEIGHTED diffuse reflection: `f(wi) = (1-FrDielectric(cos_wi,
// eta)) / (c*pi)`, `c = 1 - 2*FresnelMoment1(1/eta)` an energy-
// renormalization constant accounting for light trapped and re-emitted
// by internal reflection inside a dielectric-coated diffuse layer (the
// real physical basis: a "crystal" sphere - light exits MORE at
// grazing angles, since Fresnel reflectance is LOWEST there, the
// opposite intuition from a bare specular Fresnel surface). Genuinely
// ACHROMATIC (no albedo tint at all, unlike every OTHER diffuse-family
// material here) - `eta` (materialType 2/4/5/9/11 already each reuse
// this field their own way) and the precomputed `c` constant
// (`mat.roughness`, matching Oren-Nayar's own reuse of the same field
// for an unrelated per-material scalar) are its ONLY two parameters.
// `c` is computed HOST-SIDE (FresnelMoment1's own polynomial fit is a
// fixed function of a compile-time-known `eta`, needing no per-hit
// device evaluation at all - simpler than porting FresnelMoment1()
// itself to MSL, and exactly equivalent since eta never varies per-hit
// for this material).
inline float normalizedFresnelF(float3 wi, float3 n, float eta, float c) {
    float cosWi = max(dot(n, wi), 0.0);
    if (cosWi <= 0.0) return 0.0;
    float fr = frDielectric(cosWi, eta);
    float cv = max(c, 1e-6);
    return (1.0 - fr) / (cv * M_PI_F);
}

// Structurally identical to shadeOrenNayar() just above (same cosine-
// weighted NEE+continuation shape - see that function's own comment on
// why: `albedo*orenNayarF(...)` there becomes plain
// `float3(normalizedFresnelF(...))` here, since this material has no
// separate albedo tint at all, the BRDF value IS the whole weight).
inline bool shadeNormalizedFresnel(TriangleMaterial mat, float3 hitPoint, float3 facingNormal,
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
                float pdfBsdfForThisDir = lambertianPdf(cosSurface);
                float weight = (pdfSolidAngle * pdfSolidAngle)
                    / (pdfSolidAngle * pdfSolidAngle + pdfBsdfForThisDir * pdfBsdfForThisDir);
                float transmittance = exp(-uniforms.fogSigmaT * dist);
                radiance += throughput * float3(normalizedFresnelF(wi, facingNormal, mat.ior, mat.roughness))
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
                    radiance += throughput * float3(normalizedFresnelF(plWi, facingNormal, mat.ior, mat.roughness))
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
                    radiance += throughput * float3(normalizedFresnelF(dlWi, facingNormal, mat.ior, mat.roughness))
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
                        radiance += throughput * float3(normalizedFresnelF(pjWi, facingNormal, mat.ior, mat.roughness))
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
                        radiance += throughput * float3(normalizedFresnelF(glWi, facingNormal, mat.ior, mat.roughness))
                                    * glRadiance * glCosSurface * glTransmittance / glDistSq;
                    }
                }
            }
        }

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
                    float envPdfBsdf = lambertianPdf(envCosSurface);
                    float envWeight = (envPdfSolidAngle * envPdfSolidAngle)
                        / (envPdfSolidAngle * envPdfSolidAngle + envPdfBsdf * envPdfBsdf);
                    radiance += throughput * float3(normalizedFresnelF(envWi, facingNormal, mat.ior, mat.roughness))
                                * envRadiance * envCosSurface / envPdfSolidAngle * envWeight;
                }
            }
        }

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
                    float pbrtEnvPdfBsdf = lambertianPdf(pbrtEnvCosSurface);
                    float pbrtEnvWeight = (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle)
                        / (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle + pbrtEnvPdfBsdf * pbrtEnvPdfBsdf);
                    radiance += throughput * float3(normalizedFresnelF(pbrtEnvWi, facingNormal, mat.ior, mat.roughness))
                                * pbrtEnvRadianceSample * pbrtEnvCosSurface / pbrtEnvPdfSolidAngle * pbrtEnvWeight;
                }
            }
        }
    }

    // Continuation ray: cosine-weighted hemisphere sampling, same
    // pdf-cancellation shape as Oren-Nayar's own (see that function's
    // own comment) - `f*cosTheta/pdf` collapses to
    // `normalizedFresnelF(...)*pi`, i.e. `(1-Fr(cosWi,eta))/c` exactly
    // (matching this BxDF's own documented closed-form `sample()`
    // weight, `bxdfs_layered.h`'s own comment - not a coincidence, the
    // same algebra pbrt-v4 itself already derived).
    float3 newDir = cosineSampleHemisphere(facingNormal, rngState);
    rayDir = newDir;
    rayOrigin = hitPoint + facingNormal * 0.001f;
    throughput *= float3(normalizedFresnelF(newDir, facingNormal, mat.ior, mat.roughness)) * M_PI_F;
    bsdfPdf = lambertianPdf(max(dot(facingNormal, newDir), 0.0001f));
    specularBounce = false;
    return true;
}

// CoatedDiffuseBxDF (pbrt-v4) - a rough dielectric coat (GGX, real IOR)
// over a Lambertian base, the real physical model behind "lacquered
// wood"/"coated plastic": light either specularly reflects straight off
// the coat's own top surface, or transmits through, bounces around
// (possibly many times) between the coat's underside and the diffuse
// base below, and eventually re-exits through the top - pbrt-v4's own
// LayeredBxDF, a genuine stochastic random walk with NO closed-form
// BSDF value (unlike every earlier material in this file). This port
// deliberately transliterates the ALREADY-SHIPPED, ALREADY-VERIFIED
// OptiX GPU reference (gpu/optix/optix_device_helpers.h's own
// MaterialType::CoatedDiffuse case) rather than re-deriving from CPU's
// own src/shared/bxdfs_layered.h random walk from scratch - OptiX
// already worked out every hard design question (bounce budget, which
// Fresnel convention to use where, how to avoid the "one bounce then
// give up" darkness bug) and left comments explaining each one; this
// keeps that exact, tested shape rather than risking a fresh derivation.
//
// TWO DELIBERATELY DIFFERENT Fresnel conventions, matching OptiX's own
// real (not internally unified) behavior exactly:
//   - The CONTINUATION-ray sampler below (`shadeCoatedDiffuse`'s own
//     diffuse-escape loop) uses `frDielectric(cosOut, 1.0/eta)` (coat-
//     to-air, inverted) for its exit test - OptiX's own custom,
//     simplified per-launch bounce loop (NOT calling CPU's own
//     `layered_sample_local`), independently written and independently
//     tuned until it stopped rendering too dark.
//   - `layeredCoatedDiffuseF()` below (the stochastic NEE/MIS `f()`
//     value, called for real light sampling) uses `frDielectric(cos,
//     eta)` (uninverted) throughout, matching `src/shared/
//     bxdfs_layered.h`'s own `layered_f()` byte-for-byte - the function
//     OptiX's own shading code calls VERBATIM (it's `CPU_GPU`-tagged,
//     compiled unchanged for CUDA), not reimplemented.
// Unifying these into one "consistent" convention would be UNFAITHFUL
// to what OptiX's own already-verified code actually runs - not a bug
// to fix, two genuinely different code paths in the reference itself.
//
// No medium scattering (CoatedDiffuseBxDF never sets one - `medium_
// albedo` is always 0 in both CPU's and OptiX's own construction), so
// that whole branch of the shared CPU random walk is omitted here
// entirely, matching OptiX's own identical omission (its own comment:
// "medium scattering omitted - coateddiffuse never sets a scattering
// medium").
//
// `mat.color` = Lambertian base albedo, `mat.ior` = coat's real
// dielectric IOR, `mat.roughness` = PRECOMPUTED GGX alpha (host-side
// `RoughnessToAlpha`, not squared in-shader - see
// buildCornellCoatedDiffuse()'s own comment, metal_poc.mm).

// Stochastic BSDF value at an arbitrary queried direction (wiLocal
// toward the light, woLocal toward the camera, both already in the
// hit's own local shading frame, z-up) - a direct port of
// `layered_detail::layered_f()` (src/shared/bxdfs_layered.h), medium-
// free, `nSamples` folded to 1 (this scene's own default, matching
// every call site in this codebase). Consumes `rngState` directly
// (this file's own single-stream RNG convention) rather than a
// separate PCG32 sub-stream the way the CPU/OptiX reference does -
// Monte Carlo correctness only needs valid uniform draws, not a
// bitwise-identical sequence to CPU's own.
inline float3 layeredCoatedDiffuseF(float3 wiLocal, float3 woLocal, float eta, float alpha,
                                     float3 albedo, thread uint& rngState) {
    if (wiLocal.z <= 0.0 || woLocal.z <= 0.0) return float3(0.0);
    float3 result = float3(0.0);

    // Zero-bounce term: direct GGX reflection off the coat's own top
    // surface (no penetration at all) - matches the entrance-reflect
    // branch of the walk below exactly (standard GGX reflection shape
    // times FrDielectric at the wi/wo half-vector), deterministic, not
    // part of the stochastic average.
    {
        float3 h = wiLocal + woLocal;
        float hlen = length(h);
        if (hlen > 1e-8) {
            h /= hlen;
            float D = ggxD(h, alpha, alpha);
            float G = ggxG(woLocal, wiLocal, alpha, alpha);
            float cosWiH = dot(wiLocal, h);
            float F0 = frDielectric(cosWiH, eta);
            float val = D * G * F0 / max(4.0 * wiLocal.z * woLocal.z, 1e-8);
            result = float3(val);
        }
    }

    const int kMaxDepth = 10;
    const float kThickness = 0.01;

    float3 wm = sampleGGXVNDF(wiLocal, alpha, alpha, rngState);
    float cosI = dot(wiLocal, wm);
    float Fin = frDielectric(cosI, eta);
    float3 w = 2.0 * cosI * wm - wiLocal;
    w.z = -abs(w.z);
    if (w.z == 0.0) return result;

    float3 beta = float3(1.0 - Fin);
    float3 accum = float3(0.0);

    for (int depth = 0; depth < kMaxDepth; ++depth) {
        if (depth > 3) {
            float rrBeta = max(beta.x, max(beta.y, beta.z));
            if (rrBeta < 0.25) {
                float q = max(0.0, 1.0 - rrBeta);
                if (randFloat(rngState) < q) break;
                beta /= max(1.0 - q, 1e-6);
            }
        }

        // Beer-Lambert transmittance through the coat's own thickness
        // (no medium scattering - see this function's own header
        // comment) - advances to whichever interface `w` is heading
        // toward.
        beta *= exp(-kThickness / max(abs(w.z), 1e-6));
        bool atBottom = (w.z < 0.0);

        if (atBottom) {
            // Bottom-interface Lambertian bounce - always leaves `w`
            // pointing back upward (cosine-sampled about the local +z).
            w = cosineSampleHemisphere(float3(0.0, 0.0, 1.0), rngState);
            beta *= albedo;
        } else {
            // At the top interface from inside: connect toward wo using
            // a plain GGX reflection half-vector between the walk's
            // current direction and wo - the only exit point this
            // reflective BSDF has.
            float3 h2 = w + woLocal;
            float hlen2 = length(h2);
            if (hlen2 > 1e-8) {
                h2 /= hlen2;
                float D2 = ggxD(h2, alpha, alpha);
                float G2 = ggxG(woLocal, w, alpha, alpha);
                float cosWH = dot(w, h2);
                float Fexit = frDielectric(cosWH, eta);
                float shape = D2 * G2 / max(4.0 * w.z * woLocal.z, 1e-8);
                accum += beta * (shape * (1.0 - Fexit)) * woLocal.z;
            }

            // Continue the walk: deterministic internal reflection,
            // weighted by the actual reflectance (the transmission/exit
            // possibility was already accounted for by the connection
            // above).
            float3 wm2 = sampleGGXVNDF(w, alpha, alpha, rngState);
            float cos2 = dot(w, wm2);
            float Fout = frDielectric(cos2, eta);
            float3 r2 = 2.0 * cos2 * wm2 - w;
            r2.z = -abs(r2.z);
            w = r2;
            beta *= Fout;
        }
    }

    result += accum;
    return result;
}

// GGX VNDF reflection pdf (same `D*G1/(4*NdotO)` shape already used by
// materialType 4/9's own NEE weight, section ~104) - reused here as the
// cheap, shape-matched MIS proxy pdf for this material's own unbounded-
// depth random walk, which has no real closed-form pdf at all. Any
// valid pdf keeps MIS/NEE unbiased (only variance is affected) - the
// same choice OptiX's own `ggx_vndf_reflection_pdf` already makes for
// this exact material.
inline float coatedDiffuseProxyPdf(float3 woLocal, float3 wiLocal, float alpha) {
    float3 h = woLocal + wiLocal;
    float hlen = length(h);
    if (hlen < 1e-8) return 0.0;
    h /= hlen;
    float D = ggxD(h, alpha, alpha);
    float G1 = ggxG1(woLocal, alpha, alpha);
    return (D * G1) / max(4.0 * woLocal.z, 1e-6);
}

inline bool shadeCoatedDiffuse(TriangleMaterial mat, float3 hitPoint, float3 facingNormal,
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
    float alpha = max(mat.roughness, 0.0001);
    bool effectivelySmooth = alpha < 0.001;  // TrowbridgeReitz::EffectivelySmooth() threshold

    float3 tangent, bitangent;
    buildAnisotropicOnb(facingNormal, tangent, bitangent);
    float3 woWorld = -rayDir;
    float3 woLocal = float3(dot(woWorld, tangent), dot(woWorld, bitangent), dot(woWorld, facingNormal));
    woLocal.z = max(woLocal.z, 0.0001);

    if (!effectivelySmooth && all(mat.emission == float3(0.0))) {
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
                float3 wiLocal = float3(dot(wi, tangent), dot(wi, bitangent), dot(wi, facingNormal));
                float3 f = layeredCoatedDiffuseF(wiLocal, woLocal, mat.ior, alpha, float3(mat.color), rngState);
                float pdfSolidAngle = (distSq / (ls.area * abs(cosLight))) * ls.pmf;
                float pdfBsdf = coatedDiffuseProxyPdf(woLocal, wiLocal, alpha);
                float weight = (pdfSolidAngle * pdfSolidAngle)
                    / (pdfSolidAngle * pdfSolidAngle + pdfBsdf * pdfBsdf);
                float transmittance = exp(-uniforms.fogSigmaT * dist);
                radiance += throughput * f * ls.emission * cosSurface * transmittance / pdfSolidAngle * weight;
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
                    float3 plWiLocal = float3(dot(plWi, tangent), dot(plWi, bitangent), dot(plWi, facingNormal));
                    float3 plF = layeredCoatedDiffuseF(plWiLocal, woLocal, mat.ior, alpha, float3(mat.color), rngState);
                    float plTransmittance = exp(-uniforms.fogSigmaT * plDist);
                    float plSpot = spotLightFalloff(-plWi, float3(pl.direction), pl.cosOuterAngle, pl.cosInnerAngle);
                    radiance += throughput * plF * float3(pl.emission) * plCosSurface * plSpot * plTransmittance / plDistSq;
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
                    float3 dlWiLocal = float3(dot(dlWi, tangent), dot(dlWi, bitangent), dot(dlWi, facingNormal));
                    float3 dlF = layeredCoatedDiffuseF(dlWiLocal, woLocal, mat.ior, alpha, float3(mat.color), rngState);
                    // Skip when there is no fog - rayBoxExitDistance()
                    // is only valid for an origin INSIDE the hardcoded
                    // room bounds (see that function.s own comment for
                    // why calling it unconditionally is unsafe).
                    float dlTransmittance = (uniforms.fogSigmaT > 0.0)
                        ? exp(-uniforms.fogSigmaT * rayBoxExitDistance(dlShadowRay.origin, dlWi, kRoomBoundsMin, kRoomBoundsMax))
                        : 1.0;
                    radiance += throughput * dlF * float3(dl.emission) * dlCosSurface * dlTransmittance;
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
                        float3 pjWiLocal = float3(dot(pjWi, tangent), dot(pjWi, bitangent), dot(pjWi, facingNormal));
                        float3 pjF = layeredCoatedDiffuseF(pjWiLocal, woLocal, mat.ior, alpha, float3(mat.color), rngState);
                        float pjTransmittance = exp(-uniforms.fogSigmaT * pjDist);
                        radiance += throughput * pjF * pjRadiance * pjCosSurface * pjTransmittance / pjDistSq;
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
                        float3 glWiLocal = float3(dot(glWi, tangent), dot(glWi, bitangent), dot(glWi, facingNormal));
                        float3 glF = layeredCoatedDiffuseF(glWiLocal, woLocal, mat.ior, alpha, float3(mat.color), rngState);
                        float glTransmittance = exp(-uniforms.fogSigmaT * glDist);
                        radiance += throughput * glF * glRadiance * glCosSurface * glTransmittance;
                    }
                }
            }
        }
    }

    // Continuation ray: entrance test at the coat's own top surface,
    // then either a specular GGX reflection (probability F_in) or a
    // transmit-and-random-walk-through-the-base escape (probability
    // 1-F_in) - transliterated directly from OptiX's own already-
    // verified `MaterialType::CoatedDiffuse` bounce loop (see this
    // function's own header comment on why NOT CPU's
    // `layered_sample_local` instead).
    float3 wm = sampleGGXVNDF(woLocal, alpha, alpha, rngState);
    float cosI = dot(woLocal, wm);
    float Fin = frDielectric(cosI, mat.ior);
    float3 newDirLocal;
    float3 beta;
    if (randFloat(rngState) < Fin) {
        float3 reflLocal = 2.0 * cosI * wm - woLocal;
        if (reflLocal.z <= 0.0) return false;
        float G1 = ggxG1(woLocal, alpha, alpha);
        float G = ggxG(reflLocal, woLocal, alpha, alpha);
        float w = (G1 > 1e-8) ? G / G1 : 0.0;
        float fw = Fin * w;
        beta = float3(fw, fw, fw);
        newDirLocal = reflLocal;
    } else {
        constexpr int kMaxCoatBounces = 8;
        float3 b = float3(1.0 - Fin);
        float3 diffDirLocal = float3(0.0, 0.0, 1.0);
        bool escaped = false;
        for (int cb = 0; cb < kMaxCoatBounces; ++cb) {
            diffDirLocal = cosineSampleHemisphere(float3(0.0, 0.0, 1.0), rngState);
            b *= float3(mat.color);
            float3 wm2 = sampleGGXVNDF(diffDirLocal, alpha, alpha, rngState);
            float cosOut = dot(diffDirLocal, wm2);
            // Coat-to-air exit test - inverted eta, matching OptiX's own
            // sample loop exactly (see this function's own header
            // comment on the two deliberately different conventions).
            float Fout = frDielectric(cosOut, 1.0 / mat.ior);
            if (randFloat(rngState) < Fout) continue;  // TIR: bounce again
            b *= (1.0 - Fout);
            escaped = true;
            break;
        }
        if (!escaped) return false;
        beta = b;
        newDirLocal = diffDirLocal;
    }

    float3 newDirWorld = normalize(newDirLocal.x * tangent + newDirLocal.y * bitangent + newDirLocal.z * facingNormal);
    rayDir = newDirWorld;
    rayOrigin = hitPoint + facingNormal * 0.001f;
    throughput *= beta;
    if (!effectivelySmooth) {
        bsdfPdf = coatedDiffuseProxyPdf(woLocal, newDirLocal, alpha);
        specularBounce = false;
    } else {
        specularBounce = true;
    }
    return true;
}

// CoatedConductorBxDF (pbrt-v4) - the SAME rough-dielectric-coat random
// walk `shadeCoatedDiffuse()` just above already implements (see that
// function's own header comment for the full derivation/rationale -
// same OptiX-reference-not-CPU-reference porting strategy, same "two
// deliberately different Fresnel conventions" note), with the bottom
// interface swapped from a Lambertian cosine bounce to a GGX-conductor
// specular bounce (real per-channel complex Fresnel, `frComplexRGB()`).
// Ported from `gpu/optix/optix_device_helpers.h`'s own
// `MaterialType::CoatedConductor` case, which is genuinely SIMPLER than
// CoatedDiffuse's own sample step for one real physical reason: a
// conductor's bottom bounce is a single specular GGX reflection (one
// direction in, one direction out), not a Lambertian bounce that can
// need several retries before finding an exit angle that clears the
// coat - so the continuation sampler below needs no retry loop at all,
// unlike `shadeCoatedDiffuse()`'s own `kMaxCoatBounces` loop.
//
// `mat.conductorEta`/`mat.conductorK` = the metal base's own real
// per-channel complex IOR (same fields materialType 4/9 already use);
// `mat.color` is UNUSED (this material has no separate diffuse albedo
// at all - the metal's own colour comes entirely from its complex
// Fresnel). `mat.ior`/`mat.roughness` are the coat's own (real IOR,
// precomputed GGX alpha), same convention as materialType 19.

// Stochastic BSDF value (see `layeredCoatedDiffuseF()`'s own header
// comment for the full derivation) with a GGX-conductor bottom bounce
// in place of the Lambertian one - direct port of
// `layered_detail::ConductorBottomBounce::bounce()` (src/shared/
// bxdfs_layered.h) folded into the SAME shared random-walk shape
// `layeredCoatedDiffuseF()` already implements.
