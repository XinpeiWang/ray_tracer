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
            reaimShadowRay(shadowRay, hitPoint, dist);
            intersection_result<instancing, triangle_data> shadowResult =
                traceShadowAny(isect, shadowRay, accelStructure, functionTable);
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
                reaimShadowRay(plShadowRay, hitPoint, plDist);
                intersection_result<instancing, triangle_data> plShadowResult =
                    traceShadowAny(isect, plShadowRay, accelStructure, functionTable);
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
                    traceShadowAny(isect, dlShadowRay, accelStructure, functionTable);
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
                    reaimShadowRay(pjShadowRay, hitPoint, pjDist);
                    intersection_result<instancing, triangle_data> pjShadowResult =
                        traceShadowAny(isect, pjShadowRay, accelStructure, functionTable);
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
                    reaimShadowRay(glShadowRay, hitPoint, glDist);
                    intersection_result<instancing, triangle_data> glShadowResult =
                        traceShadowAny(isect, glShadowRay, accelStructure, functionTable);
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
                    traceShadowAny(isect, envShadowRay, accelStructure, functionTable);
                if (envShadowResult.type == intersection_type::none) {
                    float2 envUV = envMapUV(envWi);
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
            float3 pbrtEnvWi = pbrtEnvSampleDirection(uniforms, pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, pbrtEnvMapWidth, pbrtEnvMapHeight, hitPoint,
                                                           randFloat(rngState), randFloat(rngState), pbrtEnvPdfSolidAngle);
            float pbrtEnvCosSurface = dot(facingNormal, pbrtEnvWi);
            if (pbrtEnvCosSurface > 0.0 && pbrtEnvPdfSolidAngle > 1e-9) {
                ray pbrtEnvShadowRay;
                pbrtEnvShadowRay.origin = hitPoint + facingNormal * 0.001f;
                pbrtEnvShadowRay.direction = pbrtEnvWi;
                pbrtEnvShadowRay.min_distance = 0.001f;
                pbrtEnvShadowRay.max_distance = 1e5f;
                intersection_result<instancing, triangle_data> pbrtEnvShadowResult =
                    traceShadowAny(isect, pbrtEnvShadowRay, accelStructure, functionTable);
                if (pbrtEnvShadowResult.type == intersection_type::none) {
                    float3 pbrtEnvRadianceSample = pbrtEnvLeAt(uniforms, pbrtEnvConditionalCDF, pbrtEnvTexture, textureSampler, hitPoint, pbrtEnvWi);
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
            reaimShadowRay(shadowRay, hitPoint, dist);
            intersection_result<instancing, triangle_data> shadowResult =
                traceShadowAny(isect, shadowRay, accelStructure, functionTable);
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
                reaimShadowRay(plShadowRay, hitPoint, plDist);
                intersection_result<instancing, triangle_data> plShadowResult =
                    traceShadowAny(isect, plShadowRay, accelStructure, functionTable);
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
                    traceShadowAny(isect, dlShadowRay, accelStructure, functionTable);
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
                    reaimShadowRay(pjShadowRay, hitPoint, pjDist);
                    intersection_result<instancing, triangle_data> pjShadowResult =
                        traceShadowAny(isect, pjShadowRay, accelStructure, functionTable);
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
                    reaimShadowRay(glShadowRay, hitPoint, glDist);
                    intersection_result<instancing, triangle_data> glShadowResult =
                        traceShadowAny(isect, glShadowRay, accelStructure, functionTable);
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
                    traceShadowAny(isect, envShadowRay, accelStructure, functionTable);
                if (envShadowResult.type == intersection_type::none) {
                    float2 envUV = envMapUV(envWi);
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
            float3 pbrtEnvWi = pbrtEnvSampleDirection(uniforms, pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, pbrtEnvMapWidth, pbrtEnvMapHeight, hitPoint,
                                                           randFloat(rngState), randFloat(rngState), pbrtEnvPdfSolidAngle);
            float pbrtEnvCosSurface = dot(facingNormal, pbrtEnvWi);
            if (pbrtEnvCosSurface > 0.0 && pbrtEnvPdfSolidAngle > 1e-9) {
                ray pbrtEnvShadowRay;
                pbrtEnvShadowRay.origin = hitPoint + facingNormal * 0.001f;
                pbrtEnvShadowRay.direction = pbrtEnvWi;
                pbrtEnvShadowRay.min_distance = 0.001f;
                pbrtEnvShadowRay.max_distance = 1e5f;
                intersection_result<instancing, triangle_data> pbrtEnvShadowResult =
                    traceShadowAny(isect, pbrtEnvShadowRay, accelStructure, functionTable);
                if (pbrtEnvShadowResult.type == intersection_type::none) {
                    float3 pbrtEnvRadianceSample = pbrtEnvLeAt(uniforms, pbrtEnvConditionalCDF, pbrtEnvTexture, textureSampler, hitPoint, pbrtEnvWi);
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

// CoatedDiffuseBxDF (pbrt-v4) - a dielectric coat (real IOR, GGX roughness, smooth by default) over a Lambertian base, the
// physical model behind "lacquered wood"/"coated plastic": pbrt-v4's LayeredBxDF, a stochastic random walk with no closed-form
// BSDF value. The walk itself (Sample_f, f(), PDF()) lives in metal_poc_layered_bxdf.metal, a port of src/shared/bxdfs_layered.h;
// this file only builds the layers and wires the three pieces into shadeCoatedDiffuse() the way pbrt integrates them:
//   - the continuation direction and its weight come from the walk's own sample (layeredSample), a smooth coat's mirror
//     reflection being a specular sample (no MIS);
//   - the MIS density of a non-specular sample, and of each light sample, is pbrt's PDF() (layeredPdf);
//   - light samples use the stochastic f() (layeredF), at every vertex, whether or not the walk then succeeds.
//
// `mat.color` = Lambertian base albedo, `mat.ior` = coat's real dielectric IOR, `mat.roughness` = PRECOMPUTED GGX alpha
// (host-side `RoughnessToAlpha`, not squared in-shader - see buildCornellCoatedDiffuse()'s own comment, metal_poc.mm).

inline float3 layeredCoatedDiffuseF(float3 wiLocal, float3 woLocal, float eta, float alpha,
                                     float3 albedo, thread uint& rngState) {
    LayerTop top; LayerBottom bottom;
    layeredCoatedDiffuseParts(eta, alpha, albedo, top, bottom);
    return layeredF(top, bottom, kLayerThickness, woLocal, wiLocal, rngState);
}

inline float layeredCoatedDiffusePdf(float3 woLocal, float3 wiLocal, float eta, float alpha, float3 albedo) {
    LayerTop top; LayerBottom bottom;
    layeredCoatedDiffuseParts(eta, alpha, albedo, top, bottom);
    return layeredPdf(top, bottom, woLocal, wiLocal);
}

inline LayerSample layeredCoatedDiffuseSample(float3 woLocal, float eta, float alpha, float3 albedo, thread uint& rngState) {
    LayerTop top; LayerBottom bottom;
    layeredCoatedDiffuseParts(eta, alpha, albedo, top, bottom);
    return layeredSample(top, bottom, kLayerThickness, woLocal, rngState);
}

// GGX VNDF reflection pdf of the coat's own top surface - the MIS proxy the shaders used before the pbrt port. No longer used
// by shadeCoatedDiffuse()/shadeCoatedConductor() (layeredPdf replaced it); kept because the shader tests call it.
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
    float alpha = max(mat.roughness, 0.0);

    float3 tangent, bitangent;
    buildAnisotropicOnb(facingNormal, tangent, bitangent);
    float3 woWorld = -rayDir;
    float3 woLocal = float3(dot(woWorld, tangent), dot(woWorld, bitangent), dot(woWorld, facingNormal));
    woLocal.z = max(woLocal.z, 0.0001);

    // A Lambertian base always scatters diffusely: never a delta BSDF, so light samples are always taken (pbrt Flags()).
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
            reaimShadowRay(shadowRay, hitPoint, dist);
            intersection_result<instancing, triangle_data> shadowResult =
                traceShadowAny(isect, shadowRay, accelStructure, functionTable);
            if (shadowResult.type == intersection_type::none) {
                float3 wiLocal = float3(dot(wi, tangent), dot(wi, bitangent), dot(wi, facingNormal));
                float3 f = layeredCoatedDiffuseF(wiLocal, woLocal, mat.ior, alpha, float3(mat.color), rngState);
                float pdfSolidAngle = (distSq / (ls.area * abs(cosLight))) * ls.pmf;
                float pdfBsdf = layeredCoatedDiffusePdf(woLocal, wiLocal, mat.ior, alpha, float3(mat.color));
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
                reaimShadowRay(plShadowRay, hitPoint, plDist);
                intersection_result<instancing, triangle_data> plShadowResult =
                    traceShadowAny(isect, plShadowRay, accelStructure, functionTable);
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
                    traceShadowAny(isect, dlShadowRay, accelStructure, functionTable);
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
                    reaimShadowRay(pjShadowRay, hitPoint, pjDist);
                    intersection_result<instancing, triangle_data> pjShadowResult =
                        traceShadowAny(isect, pjShadowRay, accelStructure, functionTable);
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
                    reaimShadowRay(glShadowRay, hitPoint, glDist);
                    intersection_result<instancing, triangle_data> glShadowResult =
                        traceShadowAny(isect, glShadowRay, accelStructure, functionTable);
                    if (glShadowResult.type == intersection_type::none) {
                        float3 glWiLocal = float3(dot(glWi, tangent), dot(glWi, bitangent), dot(glWi, facingNormal));
                        float3 glF = layeredCoatedDiffuseF(glWiLocal, woLocal, mat.ior, alpha, float3(mat.color), rngState);
                        float glTransmittance = exp(-uniforms.fogSigmaT * glDist);
                        radiance += throughput * glF * glRadiance * glCosSurface * glTransmittance / glDistSq;
                    }
                }
            }
        }

        // Environment lights (the demo scene's picture, and a pbrt scene's own image-based infinite light): one light sample from the picture's own distribution,
        // MIS-weighted against the walk's sample by pbrt's PDF(). Without this the continuation ray's escape is MIS-weighted for a light sample that is never
        // taken, and an object under an image sky comes out dark (a coated ball read 6% low under a white picture sky and exactly right under the same sky as a constant:
        // found by scripts/consistency_sweep.py).
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
                    traceShadowAny(isect, envShadowRay, accelStructure, functionTable);
                if (envShadowResult.type == intersection_type::none) {
                    float3 envWiLocal = float3(dot(envWi, tangent), dot(envWi, bitangent), dot(envWi, facingNormal));
                    float3 envF = layeredCoatedDiffuseF(envWiLocal, woLocal, mat.ior, alpha, float3(mat.color), rngState);
                    float envPdfBsdf = layeredCoatedDiffusePdf(woLocal, envWiLocal, mat.ior, alpha, float3(mat.color));
                    float envWeight = (envPdfSolidAngle * envPdfSolidAngle) / (envPdfSolidAngle * envPdfSolidAngle + envPdfBsdf * envPdfBsdf);
                    float3 envRadiance = earthTexture.sample(textureSampler, envMapUV(envWi)).rgb;
                    radiance += throughput * envF * envRadiance * envCosSurface / envPdfSolidAngle * envWeight;
                }
            }
        }
        if (pbrtEnvMapWidth > 0u) {
            float pbrtEnvPdfSolidAngle;
            float3 pbrtEnvWi = pbrtEnvSampleDirection(uniforms, pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, pbrtEnvMapWidth, pbrtEnvMapHeight, hitPoint,
                                                           randFloat(rngState), randFloat(rngState), pbrtEnvPdfSolidAngle);
            float pbrtEnvCosSurface = dot(facingNormal, pbrtEnvWi);
            if (pbrtEnvCosSurface > 0.0 && pbrtEnvPdfSolidAngle > 1e-9) {
                ray pbrtEnvShadowRay;
                pbrtEnvShadowRay.origin = hitPoint + facingNormal * 0.001f;
                pbrtEnvShadowRay.direction = pbrtEnvWi;
                pbrtEnvShadowRay.min_distance = 0.001f;
                pbrtEnvShadowRay.max_distance = 1e5f;
                intersection_result<instancing, triangle_data> pbrtEnvShadowResult =
                    traceShadowAny(isect, pbrtEnvShadowRay, accelStructure, functionTable);
                if (pbrtEnvShadowResult.type == intersection_type::none) {
                    float3 pbrtEnvWiLocal = float3(dot(pbrtEnvWi, tangent), dot(pbrtEnvWi, bitangent), dot(pbrtEnvWi, facingNormal));
                    float3 pbrtEnvF = layeredCoatedDiffuseF(pbrtEnvWiLocal, woLocal, mat.ior, alpha, float3(mat.color), rngState);
                    float pbrtEnvPdfBsdf = layeredCoatedDiffusePdf(woLocal, pbrtEnvWiLocal, mat.ior, alpha, float3(mat.color));
                    float pbrtEnvWeight = (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle) / (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle + pbrtEnvPdfBsdf * pbrtEnvPdfBsdf);
                    float3 pbrtEnvRadianceSample = pbrtEnvLeAt(uniforms, pbrtEnvConditionalCDF, pbrtEnvTexture, textureSampler, hitPoint, pbrtEnvWi);
                    radiance += throughput * pbrtEnvF * pbrtEnvRadianceSample * pbrtEnvCosSurface / pbrtEnvPdfSolidAngle * pbrtEnvWeight;
                }
            }
        }
    }

    // Continuation ray: pbrt's LayeredBxDF::Sample_f random walk (see this file's header comment). A failed walk
    // (Russian roulette, total internal reflection, ...) ends the path here - the light samples above were already taken.
    LayerSample smp = layeredCoatedDiffuseSample(woLocal, mat.ior, alpha, float3(mat.color), rngState);
    if (!smp.valid) return false;
    float3 beta = smp.f * (abs(smp.wi.z) / smp.pdf);
    float3 newDirLocal = smp.wi;
    float3 newDirWorld = normalize(newDirLocal.x * tangent + newDirLocal.y * bitangent + newDirLocal.z * facingNormal);
    rayDir = newDirWorld;
    rayOrigin = hitPoint + facingNormal * 0.001f;
    throughput *= beta;
    if (smp.specular) {
        // A smooth coat's mirror reflection: nothing the light samples could have produced, so no MIS (pbrt specularBounce).
        specularBounce = true;
    } else {
        bsdfPdf = layeredCoatedDiffusePdf(woLocal, newDirLocal, mat.ior, alpha, float3(mat.color));
        specularBounce = false;
    }
    return true;
}

// CoatedConductorBxDF (pbrt-v4): the same layered walk with a GGX conductor as the base - its f()/pdf()/sample wrappers and
// shadeCoatedConductor() are in metal_poc_materials_extra.metal.
