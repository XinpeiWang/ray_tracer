// CoatedConductorBxDF (pbrt-v4): a dielectric coat over a GGX conductor, pbrt's LayeredBxDF - see metal_poc_layered_bxdf.metal for
// the walk and metal_poc_materials_layered.metal's CoatedDiffuse comment for how its three pieces (Sample_f, PDF(), f()) are
// integrated. `mat.conductorEta`/`mat.conductorK` = the metal's own complex IOR; as in pbrt's CoatedConductorMaterial::GetBxDF the
// conductor sits INSIDE the coat, so both are divided by the coat's IOR (done in layeredCoatedConductorParts()). `mat.color` is
// unused. `mat.ior`/`mat.roughness` are the coat's real IOR and PRECOMPUTED GGX alpha. `mat.transmitColor` = (conductor alpha, coat thickness, 0)
// when transmitColor.y > 0 (pbrt's conductor.roughness and thickness); otherwise (zero-initialised materials) the coat's alpha serves both
// interfaces and the thickness is kLayerThickness.
inline float3 layeredCoatedConductorF(float3 wiLocal, float3 woLocal, float eta, float alpha,
                                       float3 conductorEta, float3 conductorK, thread uint& rngState,
                                       float baseAlpha = -1.0, float thickness = kLayerThickness) {
    LayerTop top; LayerBottom bottom;
    layeredCoatedConductorParts(eta, alpha, conductorEta, conductorK, top, bottom, baseAlpha);
    return layeredF(top, bottom, thickness, woLocal, wiLocal, rngState);
}

inline float layeredCoatedConductorPdf(float3 woLocal, float3 wiLocal, float eta, float alpha,
                                        float3 conductorEta, float3 conductorK, float baseAlpha = -1.0) {
    LayerTop top; LayerBottom bottom;
    layeredCoatedConductorParts(eta, alpha, conductorEta, conductorK, top, bottom, baseAlpha);
    return layeredPdf(top, bottom, woLocal, wiLocal);
}

inline LayerSample layeredCoatedConductorSample(float3 woLocal, float eta, float alpha,
                                                 float3 conductorEta, float3 conductorK, thread uint& rngState,
                                                 float baseAlpha = -1.0, float thickness = kLayerThickness) {
    LayerTop top; LayerBottom bottom;
    layeredCoatedConductorParts(eta, alpha, conductorEta, conductorK, top, bottom, baseAlpha);
    return layeredSample(top, bottom, thickness, woLocal, rngState);
}

inline bool shadeCoatedConductor(TriangleMaterial mat, float3 hitPoint, float3 facingNormal,
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
    float3 conductorEta = float3(mat.conductorEta);
    float3 conductorK = float3(mat.conductorK);
    const bool hasBase = mat.transmitColor.y > 0.0;
    const float baseAlpha = hasBase ? max(float(mat.transmitColor.x), 0.0f) : -1.0f;
    const float coatThickness = hasBase ? float(mat.transmitColor.y) : kLayerThickness;

    // pbrt's Flags(): a smooth coat over a smooth conductor is a delta lobe (no light samples, no MIS); any roughness at all, on the
    // coat or the conductor, gets light samples. Here the one alpha serves both interfaces.
    LayerTop deltaTop; LayerBottom deltaBottom;
    layeredCoatedConductorParts(mat.ior, alpha, conductorEta, conductorK, deltaTop, deltaBottom, baseAlpha);
    const bool takesLightSamples = !layerIsDelta(deltaTop, deltaBottom);

    if (takesLightSamples && all(mat.emission == float3(0.0))) {
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
                float3 f = layeredCoatedConductorF(wiLocal, woLocal, mat.ior, alpha, conductorEta, conductorK, rngState, baseAlpha, coatThickness);
                float pdfSolidAngle = (distSq / (ls.area * abs(cosLight))) * ls.pmf;
                float pdfBsdf = layeredCoatedConductorPdf(woLocal, wiLocal, mat.ior, alpha, conductorEta, conductorK, baseAlpha);
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
                    float3 plF = layeredCoatedConductorF(plWiLocal, woLocal, mat.ior, alpha, conductorEta, conductorK, rngState, baseAlpha, coatThickness);
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
                    float3 dlF = layeredCoatedConductorF(dlWiLocal, woLocal, mat.ior, alpha, conductorEta, conductorK, rngState, baseAlpha, coatThickness);
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
                        float3 pjF = layeredCoatedConductorF(pjWiLocal, woLocal, mat.ior, alpha, conductorEta, conductorK, rngState, baseAlpha, coatThickness);
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
                        float3 glF = layeredCoatedConductorF(glWiLocal, woLocal, mat.ior, alpha, conductorEta, conductorK, rngState, baseAlpha, coatThickness);
                        float glTransmittance = exp(-uniforms.fogSigmaT * glDist);
                        radiance += throughput * glF * glRadiance * glCosSurface * glTransmittance;
                    }
                }
            }
        }
    }

    // Continuation ray: pbrt's LayeredBxDF::Sample_f random walk. A failed walk ends the path here (the light samples above were
    // already taken); a smooth coat's mirror reflection - or the whole chain of a delta lobe - is a specular sample (no MIS).
    LayerSample smp = layeredCoatedConductorSample(woLocal, mat.ior, alpha, conductorEta, conductorK, rngState, baseAlpha, coatThickness);
    if (!smp.valid) return false;
    float3 beta = smp.f * (abs(smp.wi.z) / smp.pdf);
    float3 newDirLocal = smp.wi;
    float3 newDirWorld = normalize(newDirLocal.x * tangent + newDirLocal.y * bitangent + newDirLocal.z * facingNormal);
    rayDir = newDirWorld;
    rayOrigin = hitPoint + facingNormal * 0.001f;
    throughput *= beta;
    if (smp.specular) {
        specularBounce = true;
    } else {
        bsdfPdf = layeredCoatedConductorPdf(woLocal, newDirLocal, mat.ior, alpha, conductorEta, conductorK, baseAlpha);
        specularBounce = false;
    }
    return true;
}

// Ashikhmin & Shirley's own velvet BRDF (2000; this exact port, though,
// mirrors Blender Cycles' own kernel/closure/bsdf_ashikhmin_velvet.h,
// itself adapted from Open Shading Language) - the classic fabric/
// cloth "fuzzy grazing-angle rim glow" look: a Blinn-Phong-shaped
// microfacet distribution `D` (peaked when the half-vector sits near
// the TANGENT plane, not near the normal the way every specular
// material in this POC so far peaks) times a heuristic geometric term
// `G`, giving a BRDF that's genuinely near-ZERO for head-on view/light
// (no highlight at all, unlike every other glossy material here) and
// rises toward a real peak somewhere around 75-80 degrees before
// falling off again approaching true grazing - not a monotonic curve,
// a real physical signature confirmed against a fresh reference
// program below, not assumed from the formula alone. `sigma` (this
// material's own roughness-like spread parameter) reuses `mat.ior`
// (materialType 2/4/5/9/11 already each reuse this same field their
// own way - one more reuse, not a new struct field); `G`'s own "TODO:
// derive G from D analytically" comment in the reference is Cycles'
// own, not this port's - a known, accepted heuristic in the original
// source, left exactly as-is here rather than silently "fixing" it.
inline float velvetF(float3 wo, float3 wi, float3 n, float sigma) {
    float invSigma2 = 1.0 / (sigma * sigma);
    float cosNO = dot(n, wo);
    float cosNI = dot(n, wi);
    if (cosNO <= 0.0 || cosNI <= 0.0) {
        return 0.0;
    }
    float3 h = normalize(wo + wi);
    float cosNH = dot(n, h);
    float cosH = abs(dot(wo, h));
    if (abs(cosNH) >= 1.0 - 1e-5 || cosH <= 1e-5) {
        return 0.0;
    }
    float cosNHdivH = max(cosNH / cosH, 1e-5);
    float fac1 = 2.0 * abs(cosNHdivH * cosNO);
    float fac2 = 2.0 * abs(cosNHdivH * cosNI);
    float sinNH2 = 1.0 - cosNH * cosNH;
    float sinNH4 = sinNH2 * sinNH2;
    float cot2 = (cosNH * cosNH) / sinNH2;
    float D = exp(-cot2 * invSigma2) * invSigma2 * M_1_PI_F / sinNH4;
    float G = min(1.0, min(fac1, fac2));
    return 0.25 * (D * G) / cosNO;
}

inline bool shadeVelvet(TriangleMaterial mat, float3 albedo, float3 hitPoint, float3 facingNormal,
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
    // UNIFORM hemisphere sampling (not cosine-weighted, see
    // sampleUniformHemisphere's own comment) - the competing BSDF pdf
    // for MIS against every light below is the CONSTANT 1/(2*pi), not
    // Lambertian/Oren-Nayar's own direction-dependent cosTheta/pi.
    float uniformPdf = 1.0 / (2.0 * M_PI_F);

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
                float weight = (pdfSolidAngle * pdfSolidAngle)
                    / (pdfSolidAngle * pdfSolidAngle + uniformPdf * uniformPdf);
                float transmittance = exp(-uniforms.fogSigmaT * dist);
                radiance += throughput * albedo * velvetF(woWorld, wi, facingNormal, mat.ior)
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
                    radiance += throughput * albedo * velvetF(woWorld, plWi, facingNormal, mat.ior)
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
                    radiance += throughput * albedo * velvetF(woWorld, dlWi, facingNormal, mat.ior)
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
                        radiance += throughput * albedo * velvetF(woWorld, pjWi, facingNormal, mat.ior)
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
                        radiance += throughput * albedo * velvetF(woWorld, glWi, facingNormal, mat.ior)
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
                    float envWeight = (envPdfSolidAngle * envPdfSolidAngle)
                        / (envPdfSolidAngle * envPdfSolidAngle + uniformPdf * uniformPdf);
                    radiance += throughput * albedo * velvetF(woWorld, envWi, facingNormal, mat.ior)
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
                    float pbrtEnvWeight = (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle)
                        / (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle + uniformPdf * uniformPdf);
                    radiance += throughput * albedo * velvetF(woWorld, pbrtEnvWi, facingNormal, mat.ior)
                                * pbrtEnvRadianceSample * pbrtEnvCosSurface / pbrtEnvPdfSolidAngle * pbrtEnvWeight;
                }
            }
        }
    }

    // Continuation ray: UNIFORM (not cosine-weighted) hemisphere
    // sampling, matching Cycles' own bsdf_ashikhmin_velvet_sample() -
    // pdf stays the constant 1/(2*pi) regardless of direction, so the MC
    // weight f*cosTheta/pdf is `albedo*velvetF(...)*cosTheta*2*pi`, NOT
    // Lambertian/Oren-Nayar's own pi-only factor (their own cosTheta/pi
    // pdf already cancels one power of cosTheta that this uniform pdf
    // does not).
    float3 newDir = sampleUniformHemisphere(facingNormal, rngState);
    rayDir = newDir;
    rayOrigin = hitPoint + facingNormal * 0.001f;
    float newDirCos = max(dot(facingNormal, newDir), 0.0001);
    throughput *= albedo * velvetF(woWorld, newDir, facingNormal, mat.ior) * newDirCos * (2.0 * M_PI_F);
    bsdfPdf = uniformPdf;
    specularBounce = false;
    return true;
}

// Schlick's Fresnel approximation, F0 + (1-F0)*(1-cosTheta)^5 - the SAME
// approximate formula src/shared/bxdfs_principled.h's own
// `schlick_fresnel()` uses (a deliberate simplification for this
// artist-friendly BSDF, unlike materialType 2/4/5's own EXACT
// frDielectric()/frComplex() - matching the reference exactly here
// means using its own approximation, not "upgrading" it).
inline float schlickFresnelPrincipled(float cosTheta, float F0) {
    float c = 1.0 - clamp(cosTheta, 0.0, 1.0);
    float c2 = c * c;
    float c5 = c2 * c2 * c;
    return F0 + (1.0 - F0) * c5;
}

// GGX specular BRDF value, local frame (z=normal) - same D*G/(4*cosO*cosI)
// shape ggxD()/ggxG() already compute for materialType 4/9's own
// shadeConductor(), just without a Fresnel term folded in (Principled's
// own 3-lobe blend applies Fresnel separately per lobe/channel).
inline float principledGgxBrdf(float3 wo, float3 wi, float alpha) {
    if (wo.z <= 0.0 || wi.z <= 0.0) return 0.0;
    float3 h = wo + wi;
    float hlen = length(h);
    if (hlen < 1e-8) return 0.0;
    h /= hlen;
    float D = ggxD(h, alpha, alpha);
    float G = ggxG(wo, wi, alpha, alpha);
    return D * G / max(4.0 * wo.z * wi.z, 1e-8);
}

// GGX VNDF sampling pdf, local frame - pbrt-v4's own
// `D(wm)*G1(wo)*AbsDot(wo,wm)/AbsCosTheta(wo) / (4*dot(wo,wm))`, matching
// `PrincipledBxDF::ggx_pdf()` (src/shared/bxdfs_principled.h) exactly.
inline float principledGgxPdf(float3 wo, float3 wi, float alpha) {
    if (wo.z <= 0.0 || wi.z <= 0.0) return 0.0;
    float3 h = wo + wi;
    float hlen = length(h);
    if (hlen < 1e-8) return 0.0;
    h /= hlen;
    float dotWoH = dot(wo, h);
    if (dotWoH <= 0.0) return 0.0;
    float D = ggxD(h, alpha, alpha);
    float G1 = ggxG1(wo, alpha, alpha);
    float pdfWm = D * G1 * dotWoH / max(wo.z, 1e-6);
    return pdfWm / max(4.0 * dotWoH, 1e-8);
}

// The combined 3-lobe (diffuse + specular + clearcoat) Principled PDF,
// given each lobe's already-computed selection probability - the exact
// same combine `shadePrincipled()`'s own tail performs, factored out so
// it's independently testable against `PrincipledBxDF<T>::
// scattering_pdf()`'s own identical combine (src/shared/
// bxdfs_principled.h) - same "extract the pure formula, don't duplicate
// it" rationale as `lambertianPdf()`/`ggxConductorPdf()` before this
// (sections 190/191). Does NOT re-derive `pDiff`/`pSpec`/`pCoat` itself
// (those stay computed once, at `shadePrincipled()`'s own top, for BOTH
// lobe selection AND this combine, matching the original code's own
// control flow exactly - no new computation, just a name for the
// existing tail expression).
inline float principledCombinedPdf(float pDiff, float pSpec, float pCoat, float alpha, float alphaCC,
                                    float3 woLocal, float3 wiLocal) {
    float pdfDiff = pDiff * wiLocal.z / M_PI_F;
    float pdfSpec = pSpec * principledGgxPdf(woLocal, wiLocal, alpha);
    float pdfCoat = pCoat * principledGgxPdf(woLocal, wiLocal, alphaCC);
    return pdfDiff + pdfSpec + pdfCoat;
}

// materialType 24 (B10, Principled Showcase) - pbrt-v4/Disney's own
// artist-friendly 3-lobe BSDF (diffuse + specular-dielectric-or-metal +
// clearcoat), a direct port of `PrincipledBxDF<T>::sample()`
// (src/shared/bxdfs_principled.h), the SAME shared header both CPU
// (`principled_material.h`) and OptiX (`sample_principled_material()`,
// `optix_device_helpers.h`) already build from directly. Deliberately
// has NO NEE/MIS at all - matches CPU's own `principled::scatter()`
// (`srec.skip_pdf = true`, no separate `scattering_pdf()`-driven light
// sampling loop) and OptiX's own identical `is_specular = true` choice
// for this exact material (`optix_intersection_sphere.h`'s own comment:
// "no NEE/MIS, res.r/g/b already divides by the sample pdf") - the
// BSDF's own `sample()` returns a complete `f*cos/pdf` weight in one
// call, the same "combined sample+eval, no separate NEE path" shape
// this loader's own `shadeDielectric()`/`shadeMirror()` already use for
// other delta-like materials, just with 3 stochastically-chosen lobes
// instead of 1. Field reuse (matching OptiX's own exact convention,
// `optix_device_helpers.h`'s `sample_principled_material()` comment):
// `mat.color`=base color, `mat.ior`=ior, `mat.roughness`=perceptual
// roughness, `mat.conductorEta.x`=metallic, `mat.conductorEta.y`=
// clearcoat, `mat.conductorEta.z`=clearcoat_roughness.
inline bool shadePrincipled(TriangleMaterial mat, float3 hitPoint, float3 facingNormal,
                            thread float3& rayDir, thread float3& rayOrigin,
                            thread float3& throughput, thread bool& specularBounce, thread uint& rngState) {
    float metallic = mat.conductorEta.x;
    float clearcoat = mat.conductorEta.y;
    float clearcoatRoughness = mat.conductorEta.z;
    // TrowbridgeReitz::RoughnessToAlpha(r) = sqrt(r) - the REAL pbrt-v4
    // formula, not materialType 4/9's own "square it in the shader"
    // convention (this is a brand-new shading function with no old
    // convention to reconcile with, same reasoning as materialType 19's
    // own comment, metal_poc.mm).
    float alpha = max(sqrt(max(mat.roughness, 0.0)), 0.0009);
    float alphaCC = max(sqrt(max(clearcoatRoughness, 0.0)), 0.0009);

    float3 tangent, bitangent;
    buildOnb(facingNormal, tangent, bitangent);
    // wi = the ray's OWN direction of travel (INTO the surface) - matches
    // CPU's own `in_dir = unit_vector(r_in.direction())` passed straight
    // into `bxdf.sample()` with NO negation; `wo = -wi` is derived
    // internally, exactly mirrored here.
    float3 wiWorld = normalize(rayDir);
    float3 wiLocal = float3(dot(wiWorld, tangent), dot(wiWorld, bitangent), dot(wiWorld, facingNormal));
    float3 woLocal = -wiLocal;
    if (woLocal.z <= 0.0) return false;

    float F0d = pow((mat.ior - 1.0) / (mat.ior + 1.0), 2.0);
    float Fspec = schlickFresnelPrincipled(woLocal.z, F0d);
    float wDiff = (1.0 - metallic) * (1.0 - Fspec);
    float wSpec = 1.0;
    float wCoat = clearcoat * 0.25;
    float wTotal = wDiff + wSpec + wCoat;
    if (wTotal < 1e-8) return false;
    float invW = 1.0 / wTotal;
    float pDiff = wDiff * invW;
    float pSpec = wSpec * invW;
    float pCoat = wCoat * invW;

    float u1 = randFloat(rngState);
    float3 woOutLocal;
    if (u1 < pDiff) {
        woOutLocal = cosineSampleHemisphere(float3(0.0, 0.0, 1.0), rngState);
    } else if (u1 < pDiff + pSpec) {
        float3 wm = sampleGGXVNDF(woLocal, alpha, alpha, rngState);
        float d = dot(woLocal, wm);
        woOutLocal = 2.0 * d * wm - woLocal;
    } else {
        float3 wm = sampleGGXVNDF(woLocal, alphaCC, alphaCC, rngState);
        float d = dot(woLocal, wm);
        woOutLocal = 2.0 * d * wm - woLocal;
    }
    if (woOutLocal.z <= 0.0) return false;

    float cosWiL = woOutLocal.z;
    float3 h = woLocal + woOutLocal;
    float hlen = length(h);
    float cosWm = (hlen > 1e-8) ? dot(woLocal, h / hlen) : woLocal.z;

    float FwiDiff = schlickFresnelPrincipled(cosWiL, F0d);
    float3 diffCol = float3(mat.color) * (1.0 / M_PI_F) * (1.0 - metallic) * (1.0 - FwiDiff);

    float specVal = principledGgxBrdf(woLocal, woOutLocal, alpha);
    float FspecWm = schlickFresnelPrincipled(cosWm, F0d);
    float3 FmetWm = float3(schlickFresnelPrincipled(cosWm, mat.color.x),
                            schlickFresnelPrincipled(cosWm, mat.color.y),
                            schlickFresnelPrincipled(cosWm, mat.color.z));
    float3 Fmix = (1.0 - metallic) * FspecWm + metallic * FmetWm;
    float3 specCol = Fmix * specVal;

    float ccF0 = 0.04;
    float Fcc = schlickFresnelPrincipled(cosWm, ccF0);
    float ccVal = principledGgxBrdf(woLocal, woOutLocal, alphaCC);
    float ccCol = clearcoat * 0.25 * Fcc * ccVal;

    float3 totalCol = diffCol + specCol + float3(ccCol);

    float pdf = principledCombinedPdf(pDiff, pSpec, pCoat, alpha, alphaCC, woLocal, woOutLocal);
    if (pdf < 1e-12) return false;

    float3 weight = totalCol * cosWiL / pdf;

    float3 newDirWorld = normalize(woOutLocal.x * tangent + woOutLocal.y * bitangent + woOutLocal.z * facingNormal);
    rayDir = newDirWorld;
    rayOrigin = hitPoint + facingNormal * 0.001f;
    throughput *= weight;
    specularBounce = true;
    return true;
}


// ---------------------------------------------------------------------------
// materialType 32 - tabulated measured BRDF (pbrt-v4 Material "measured", Dupuy & Jakob 2018): a port of src/shared/measured_bxdf.h and
// piecewise_linear_2d.h by way of gpu/optix/optix_measured_bxdf.h (same flat-table layout, same math). The tables (ndf, sigma, vndf,
// luminance, spectra) and the descriptors that locate them are float data in the shared `rgbGridData` buffer (the kernel has no free buffer
// slot); a descriptor field is an int stored bit-for-bit in a float slot (as_type<int>). The material's conductorEta.x holds the float index
// of its descriptor, so a table block must start below 2^24 floats (the loader checks).
//
// Descriptor layout: five MeasuredPL2D blocks of 15 ints - ndf, sigma, vndf, luminance, spectra - then one int, isotropic. Every offset in a
// block is an absolute index into rgbGridData.
// ---------------------------------------------------------------------------
struct MeasuredPL2D {
    int nx, ny, dim;
    int paramRes[3];
    int paramStride[3];
    int paramValueOffset[3];
    int dataOffset, mcdfOffset, ccdfOffset;
};

struct MeasuredTables {
    MeasuredPL2D ndf, sigma, vndf, lum, spec;
    int isotropic;
};

inline MeasuredPL2D measuredReadPL2D(device const float* g, int base) {
    MeasuredPL2D t;
    t.nx = as_type<int>(g[base + 0]);
    t.ny = as_type<int>(g[base + 1]);
    t.dim = as_type<int>(g[base + 2]);
    for (int k = 0; k < 3; ++k) {
        t.paramRes[k] = as_type<int>(g[base + 3 + k]);
        t.paramStride[k] = as_type<int>(g[base + 6 + k]);
        t.paramValueOffset[k] = as_type<int>(g[base + 9 + k]);
    }
    t.dataOffset = as_type<int>(g[base + 12]);
    t.mcdfOffset = as_type<int>(g[base + 13]);
    t.ccdfOffset = as_type<int>(g[base + 14]);
    return t;
}

inline MeasuredTables measuredReadTables(device const float* g, int base) {
    MeasuredTables m;
    m.ndf = measuredReadPL2D(g, base);
    m.sigma = measuredReadPL2D(g, base + 15);
    m.vndf = measuredReadPL2D(g, base + 30);
    m.lum = measuredReadPL2D(g, base + 45);
    m.spec = measuredReadPL2D(g, base + 60);
    m.isotropic = as_type<int>(g[base + 75]);
    return m;
}

// Folds the 2^dim corner values of one slice-blended lookup (PiecewiseLinear2D::LookImpl): `data` points at the table, i0 is the base flat
// index, sz the per-slice element count, pw the per-axis (1-t, t) weight pairs.
inline float measuredPL2DLook(device const float* data, uint i0, uint sz, thread const float* pw, thread const MeasuredPL2D& tab) {
    float result = 0.0f;
    const int combos = 1 << tab.dim;
    for (int mask = 0; mask < combos; ++mask) {
        uint idx = i0;
        float w = 1.0f;
        for (int k = 0; k < tab.dim; ++k) {
            const int bit = (mask >> k) & 1;
            if (bit) idx += uint(tab.paramStride[k]) * sz;
            w *= pw[2 * k + bit];
        }
        result += w * data[idx];
    }
    return result;
}

// PiecewiseLinear2D::FillPW: for each conditioning axis the bracketing pair and its interpolation weights, plus the slice offset.
inline void measuredPL2DFillPW(device const float* g, thread const MeasuredPL2D& tab, thread const float* p,
                               thread uint& soff, thread float* pw) {
    soff = 0u;
    for (int d = 0; d < tab.dim; ++d) {
        const int n = tab.paramRes[d];
        if (n <= 1) { pw[2 * d] = 1.0f; pw[2 * d + 1] = 0.0f; continue; }
        device const float* axis = g + tab.paramValueOffset[d];
        const float pv = p[d];
        uint lo = 0u, hi = uint(n);
        while (lo + 1u < hi) {
            const uint mid = (lo + hi) >> 1;
            if (axis[mid] <= pv) lo = mid; else hi = mid;
        }
        int idx = int(lo);
        if (idx + 1 >= n) idx = n - 2;
        const float p0 = axis[idx], p1 = axis[idx + 1];
        const float t = clamp((pv - p0) / (p1 - p0), 0.0f, 1.0f);
        pw[2 * d + 1] = t;
        pw[2 * d] = 1.0f - t;
        soff += uint(tab.paramStride[d]) * uint(idx);
    }
}

// PiecewiseLinear2D::Eval - the bilinearly interpolated normalised density at (px, py).
inline float measuredPL2DEval(device const float* g, thread const MeasuredPL2D& tab, float px, float py, thread const float* p) {
    float pw[6] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    uint soff;
    measuredPL2DFillPW(g, tab, p, soff, pw);
    const float invx = float(tab.nx - 1), invy = float(tab.ny - 1);
    const float x = px * invx, y = py * invy;
    int ix = min(int(x), tab.nx - 2);
    int iy = min(int(y), tab.ny - 2);
    const float wx1 = x - float(ix), wx0 = 1.0f - wx1;
    const float wy1 = y - float(iy), wy0 = 1.0f - wy1;
    uint idx = uint(ix + iy * tab.nx);
    const uint sz = uint(tab.nx * tab.ny);
    if (tab.dim != 0) idx += soff * sz;
    device const float* base = g + tab.dataOffset;
    const float v00 = measuredPL2DLook(base, idx, sz, pw, tab);
    const float v10 = measuredPL2DLook(base + 1, idx, sz, pw, tab);
    const float v01 = measuredPL2DLook(base + tab.nx, idx, sz, pw, tab);
    const float v11 = measuredPL2DLook(base + tab.nx + 1, idx, sz, pw, tab);
    return (wy0 * (wx0 * v00 + wx1 * v10) + wy1 * (wx0 * v01 + wx1 * v11)) * invx * invy;
}

// PiecewiseLinear2D::Sample - warps (u0, u1) through the table's bilinear-patch density: marginal row from mcdf, conditional column from
// ccdf, then an analytic quadratic inverse inside the patch.
inline void measuredPL2DSample(device const float* g, thread const MeasuredPL2D& tab, float u0, float u1, thread const float* p,
                               thread float& outPx, thread float& outPy, thread float& outPdf) {
    const float kOme = 1.0f - 1.19209e-7f;
    u0 = clamp(u0, 1.0f - kOme, kOme);
    u1 = clamp(u1, 1.0f - kOme, kOme);
    float pw[6] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    uint soff;
    measuredPL2DFillPW(g, tab, p, soff, pw);
    const int nx = tab.nx, ny = tab.ny;
    const uint sz = uint(nx * ny);
    device const float* dataBase = g + tab.dataOffset;
    device const float* mcdfBase = g + tab.mcdfOffset;
    device const float* ccdfBase = g + tab.ccdfOffset;

    // mcdf holds one value per row per slice, so its slice stride is ny, not nx*ny.
    const uint mb = (tab.dim != 0) ? (soff * uint(ny)) : 0u;
    uint rlo = 0u, rhi = uint(ny);
    while (rlo + 1u < rhi) {
        const uint mid = (rlo + rhi) >> 1;
        const float fm = measuredPL2DLook(mcdfBase, mb + mid, uint(ny), pw, tab);
        if (fm < u1) rlo = mid; else rhi = mid;
    }
    const uint row = rlo;
    u1 -= measuredPL2DLook(mcdfBase, mb + row, uint(ny), pw, tab);

    uint off = row * uint(nx) + ((tab.dim != 0) ? soff * sz : 0u);
    const float r0 = measuredPL2DLook(ccdfBase, off + uint(nx) - 1u, sz, pw, tab);
    const float r1 = measuredPL2DLook(ccdfBase, off + uint(nx * 2 - 1), sz, pw, tab);
    const bool ky = fabs(r0 - r1) < 1e-4f * (r0 + r1);
    u1 = ky ? (2.0f * u1) : (r0 - sqrt(max(0.0f, r0 * r0 - 2.0f * u1 * (r0 - r1))));
    u1 /= ky ? (r0 + r1) : (r0 - r1);

    u0 *= (1.0f - u1) * r0 + u1 * r1;
    uint clo = 0u, chi = uint(nx);
    while (clo + 1u < chi) {
        const uint mid = (clo + chi) >> 1;
        const float v0 = measuredPL2DLook(ccdfBase, off + mid, sz, pw, tab);
        const float v1 = measuredPL2DLook(ccdfBase + nx, off + mid, sz, pw, tab);
        if ((1.0f - u1) * v0 + u1 * v1 < u0) clo = mid; else chi = mid;
    }
    const uint col = clo;
    {
        const float v0 = measuredPL2DLook(ccdfBase, off + col, sz, pw, tab);
        const float v1 = measuredPL2DLook(ccdfBase + nx, off + col, sz, pw, tab);
        u0 -= (1.0f - u1) * v0 + u1 * v1;
    }
    off += col;

    const float v00 = measuredPL2DLook(dataBase, off, sz, pw, tab);
    const float v10 = measuredPL2DLook(dataBase + 1, off, sz, pw, tab);
    const float v01 = measuredPL2DLook(dataBase + nx, off, sz, pw, tab);
    const float v11 = measuredPL2DLook(dataBase + nx + 1, off, sz, pw, tab);
    const float c0 = (1.0f - u1) * v00 + u1 * v01;
    const float c1 = (1.0f - u1) * v10 + u1 * v11;
    const bool kx = fabs(c0 - c1) < 1e-4f * (c0 + c1);
    u0 = kx ? (2.0f * u0) : (c0 - sqrt(max(0.0f, c0 * c0 - 2.0f * u0 * (c0 - c1))));
    u0 /= kx ? (c0 + c1) : (c0 - c1);

    outPx = (float(col) + u0) / float(nx - 1);
    outPy = (float(row) + u1) / float(ny - 1);
    outPdf = ((1.0f - u0) * c0 + u0 * c1) * float(nx - 1) * float(ny - 1);
}

// PiecewiseLinear2D::Invert - the uniform sample Sample() would have mapped to (px, py), and the density there.
inline void measuredPL2DInvert(device const float* g, thread const MeasuredPL2D& tab, float px, float py, thread const float* p,
                               thread float& outPx, thread float& outPy, thread float& outPdf) {
    float pw[6] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    uint soff;
    measuredPL2DFillPW(g, tab, p, soff, pw);
    const int nx = tab.nx, ny = tab.ny;
    const float invx = float(nx - 1), invy = float(ny - 1);
    float sx = px * invx, sy = py * invy;
    const int ix = min(int(sx), nx - 2);
    const int iy = min(int(sy), ny - 2);
    sx -= float(ix); sy -= float(iy);
    const uint sz = uint(nx * ny);
    uint off = uint(ix + iy * nx);
    if (tab.dim != 0) off += soff * sz;
    device const float* dataBase = g + tab.dataOffset;
    device const float* mcdfBase = g + tab.mcdfOffset;
    device const float* ccdfBase = g + tab.ccdfOffset;

    const float v00 = measuredPL2DLook(dataBase, off, sz, pw, tab);
    const float v10 = measuredPL2DLook(dataBase + 1, off, sz, pw, tab);
    const float v01 = measuredPL2DLook(dataBase + nx, off, sz, pw, tab);
    const float v11 = measuredPL2DLook(dataBase + nx + 1, off, sz, pw, tab);
    const float w0y = 1.0f - sy, w1y = sy;
    const float c0 = w0y * v00 + w1y * v01;
    const float c1 = w0y * v10 + w1y * v11;
    const float pdf = (1.0f - sx) * c0 + sx * c1;

    float ux = sx * (c0 + 0.5f * sx * (c1 - c0));
    const float cdf0 = measuredPL2DLook(ccdfBase, off, sz, pw, tab);
    const float cdf1 = measuredPL2DLook(ccdfBase + nx, off, sz, pw, tab);
    ux += (1.0f - sy) * cdf0 + sy * cdf1;
    uint roff = uint(iy * nx);
    if (tab.dim != 0) roff += soff * sz;
    const float r0 = measuredPL2DLook(ccdfBase, roff + uint(nx) - 1u, sz, pw, tab);
    const float r1 = measuredPL2DLook(ccdfBase, roff + uint(nx * 2 - 1), sz, pw, tab);
    ux /= (1.0f - sy) * r0 + sy * r1;

    float uy = sy * (r0 + 0.5f * sy * (r1 - r0));
    uint moff = uint(iy);
    if (tab.dim != 0) moff += soff * uint(ny);
    uy += measuredPL2DLook(mcdfBase, moff, uint(ny), pw, tab);

    outPx = clamp(ux, 0.0f, 1.0f);
    outPy = clamp(uy, 0.0f, 1.0f);
    outPdf = pdf * invx * invy;
}

// MeasuredBxDF::f() and pdf() in one function (they share the half vector and the vndf inversion), local shading frame (z = normal).
// Returns false (f = 0, pdf = 0) for a pair the table gives no reflectance.
inline bool measuredFPdf(device const float* g, thread const MeasuredTables& tab, float3 wo, float3 wi,
                         thread float3& f, thread float& pdfOut) {
    const float kPi = 3.14159265358979323846f;
    const float3 lambda = float3(612.0f, 549.0f, 465.0f);   // the CPU's fixed R/G/B query wavelengths (material_pbrt.h kLambdaR/G/B)
    f = float3(0.0f); pdfOut = 0.0f;
    if (wo.z * wi.z <= 0.0f) return false;
    if (wo.z < 0.0f) { wo = -wo; wi = -wi; }
    float3 wm = wi + wo;
    const float wm2 = dot(wm, wm);
    if (wm2 == 0.0f) return false;
    wm *= rsqrt(wm2);

    const float thetaO = acos(clamp(wo.z, -1.0f, 1.0f)), phiO = atan2(wo.y, wo.x);
    const float thetaM = acos(clamp(wm.z, -1.0f, 1.0f)), phiM = atan2(wm.y, wm.x);
    const float uWoX = sqrt(thetaO * (2.0f / kPi)), uWoY = phiO * (0.5f / kPi) + 0.5f;
    const float uWmX = sqrt(thetaM * (2.0f / kPi));
    float uWmY = (tab.isotropic != 0 ? (phiM - phiO) : phiM) * (0.5f / kPi) + 0.5f;
    uWmY -= floor(uWmY);

    float p[3] = {phiO, thetaO, 0.0f};
    float uiX, uiY, uiPdf;
    measuredPL2DInvert(g, tab.vndf, uWmX, uWmY, p, uiX, uiY, uiPdf);

    float val[3];
    for (int c = 0; c < 3; ++c) {
        p[2] = lambda[c];
        val[c] = max(0.0f, measuredPL2DEval(g, tab.spec, uiX, uiY, p));
    }
    const float ndfVal = measuredPL2DEval(g, tab.ndf, uWmX, uWmY, p);
    const float sigmaVal = measuredPL2DEval(g, tab.sigma, uWoX, uWoY, p);
    const float denom = 4.0f * sigmaVal * fabs(wi.z);
    if (denom == 0.0f) return false;
    f = float3(val[0], val[1], val[2]) * (ndfVal / denom);

    p[2] = 0.0f;
    const float lum = measuredPL2DEval(g, tab.lum, uiX, uiY, p);
    const float sinThetaM = sqrt(max(0.0f, 1.0f - wm.z * wm.z));
    const float jacobian = 4.0f * dot(wo, wm) * max(2.0f * kPi * kPi * uWmX * sinThetaM, 1e-6f);
    if (jacobian == 0.0f) return true;   // f is valid, the sampling density is not
    pdfOut = uiPdf * lum / jacobian;
    return true;
}

// MeasuredBxDF::sample_f(): importance-samples wi (local frame) and returns the path weight f * |cos(wi)| / pdf, as pbrt-v4 applies it.
inline bool measuredSampleF(device const float* g, thread const MeasuredTables& tab, float3 wo, float u0, float u1,
                            thread float3& wiOut, thread float3& weightOut, thread float& pdfOut) {
    const float kPi = 3.14159265358979323846f;
    const float3 lambda = float3(612.0f, 549.0f, 465.0f);
    weightOut = float3(0.0f); pdfOut = 0.0f;
    const bool flip = wo.z < 0.0f;
    if (flip) wo = -wo;
    if (wo.z <= 0.0f) return false;
    const float thetaO = acos(clamp(wo.z, -1.0f, 1.0f)), phiO = atan2(wo.y, wo.x);

    float p[3] = {phiO, thetaO, 0.0f};
    float lumPx, lumPy, lumPdf;
    measuredPL2DSample(g, tab.lum, u0, u1, p, lumPx, lumPy, lumPdf);
    if (lumPdf == 0.0f) return false;
    float uWmX, uWmY, vndfPdf;
    measuredPL2DSample(g, tab.vndf, lumPx, lumPy, p, uWmX, uWmY, vndfPdf);
    if (vndfPdf == 0.0f) return false;

    float phiM = (2.0f * uWmY - 1.0f) * kPi;
    const float thetaM = uWmX * uWmX * (kPi * 0.5f);
    if (tab.isotropic != 0) phiM += phiO;
    // phiM can land in [-2pi, 2pi]; fast-math sin/cos lose accuracy outside [-pi, pi], so wrap back first.
    phiM -= 2.0f * kPi * floor((phiM + kPi) * (0.5f / kPi));
    const float sinThetaM = sin(thetaM), cosThetaM = cos(thetaM);
    const float3 wm = float3(sinThetaM * cos(phiM), sinThetaM * sin(phiM), cosThetaM);
    const float dotWoWm = dot(wo, wm);
    const float3 wi = 2.0f * dotWoWm * wm - wo;
    if (wi.z <= 0.0f) return false;

    // The spectra are evaluated at the luminance-warped point, as pbrt-v4 Sample_f (and f() via vndf.Invert) do.
    float val[3];
    for (int c = 0; c < 3; ++c) {
        p[2] = lambda[c];
        val[c] = max(0.0f, measuredPL2DEval(g, tab.spec, lumPx, lumPy, p));
    }
    const float uWoX = sqrt(thetaO * (2.0f / kPi)), uWoY = phiO * (0.5f / kPi) + 0.5f;
    const float ndfVal = measuredPL2DEval(g, tab.ndf, uWmX, uWmY, p);
    const float sigmaVal = measuredPL2DEval(g, tab.sigma, uWoX, uWoY, p);
    const float absCosWi = fabs(wi.z);
    const float denom = 4.0f * sigmaVal * absCosWi;
    if (denom == 0.0f) return false;
    const float scale = ndfVal / denom;
    const float jacobian = 4.0f * dotWoWm * max(2.0f * kPi * kPi * uWmX * sinThetaM, 1e-6f);
    if (jacobian == 0.0f) return false;
    const float finalPdf = vndfPdf * lumPdf / jacobian;
    if (finalPdf == 0.0f) return false;

    wiOut = flip ? -wi : wi;
    weightOut = float3(val[0], val[1], val[2]) * (scale * absCosWi / finalPdf);
    pdfOut = finalPdf;
    return true;
}

// Evaluates the measured BRDF for a WORLD-space direction `wi`: f (RGB) and the sampling density, both for the local frame (t, b, n).
inline bool measuredEvalWorld(device const float* g, thread const MeasuredTables& tab, float3 woLocal, float3 wi,
                              float3 t, float3 b, float3 n, thread float3& f, thread float& pdf) {
    const float3 wiLocal = float3(dot(wi, t), dot(wi, b), dot(wi, n));
    return measuredFPdf(g, tab, woLocal, wiLocal, f, pdf);
}

inline bool shadeMeasured(TriangleMaterial mat, float3 hitPoint, float3 facingNormal,
                          constant Uniforms& uniforms,
                          device const float* rgbGridData,
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
    // One VNDF-importance-sampled BxDF sample continues the path with weight f * |cos| / pdf; the vertex also takes direct-light samples
    // (area, punctual, environment) with MIS against that density, using the BxDF's own f() and pdf() - the same structure as the
    // OptiX backends (optix_device_helpers.h, MaterialType::Measured). A rejected sample (grazing wo, zero pdf, a reflection below the
    // horizon) still gets its direct-light samples, then ends the path.
    const MeasuredTables tab = measuredReadTables(rgbGridData, int(mat.conductorEta.x));
    float3 tangent, bitangent;
    buildAnisotropicOnb(facingNormal, tangent, bitangent);
    const float3 woWorld = -rayDir;
    float3 woLocal = float3(dot(woWorld, tangent), dot(woWorld, bitangent), dot(woWorld, facingNormal));
    const float3 origin = hitPoint + facingNormal * 0.001f;

    if (all(mat.emission == float3(0.0))) {
        LightSample ls = sampleAreaLight(lights, uniforms.lightCount, rngState, pbrtAreaLightTexture, textureSampler);
        float3 toLight = ls.point - hitPoint;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        float3 wi = toLight / dist;
        float cosSurface = dot(facingNormal, wi);
        float cosLight = dot(ls.normal, -wi);
        if (cosSurface > 0.0 && (cosLight > 0.0 || (ls.twoSided != 0.0 && cosLight < 0.0))) {
            float3 f; float pdfBsdf;
            if (measuredEvalWorld(rgbGridData, tab, woLocal, wi, tangent, bitangent, facingNormal, f, pdfBsdf)) {
                ray shadowRay;
                shadowRay.origin = origin;
                shadowRay.direction = wi;
                shadowRay.min_distance = 0.001f;
                shadowRay.max_distance = dist - 0.002f;
                reaimShadowRay(shadowRay, hitPoint, dist);
                if (traceShadowAny(isect, shadowRay, accelStructure, functionTable).type == intersection_type::none) {
                    float pdfSolidAngle = (distSq / (ls.area * abs(cosLight))) * ls.pmf;
                    float weight = (pdfSolidAngle * pdfSolidAngle) / (pdfSolidAngle * pdfSolidAngle + pdfBsdf * pdfBsdf);
                    float transmittance = exp(-uniforms.fogSigmaT * dist);
                    radiance += throughput * f * ls.emission * cosSurface * transmittance / pdfSolidAngle * weight;
                }
            }
        }

        for (uint pli = 0; pli < uniforms.pointLightCount; ++pli) {
            PointLight pl = pointLights[pli];
            float3 toP = float3(pl.position) - hitPoint;
            float pDistSq = dot(toP, toP);
            float pDist = sqrt(pDistSq);
            float3 pWi = toP / pDist;
            float pCos = dot(facingNormal, pWi);
            if (pCos > 0.0) {
                float3 f; float pdfUnused;
                if (measuredEvalWorld(rgbGridData, tab, woLocal, pWi, tangent, bitangent, facingNormal, f, pdfUnused)) {
                    ray shadowRay;
                    shadowRay.origin = origin;
                    shadowRay.direction = pWi;
                    shadowRay.min_distance = 0.001f;
                    shadowRay.max_distance = pDist - 0.002f;
                    reaimShadowRay(shadowRay, hitPoint, pDist);
                    if (traceShadowAny(isect, shadowRay, accelStructure, functionTable).type == intersection_type::none) {
                        float spot = spotLightFalloff(-pWi, float3(pl.direction), pl.cosOuterAngle, pl.cosInnerAngle);
                        radiance += throughput * f * float3(pl.emission) * pCos * spot * exp(-uniforms.fogSigmaT * pDist) / pDistSq;
                    }
                }
            }
        }

        for (uint dli = 0; dli < uniforms.directionalLightCount; ++dli) {
            DirectionalLight dl = directionalLights[dli];
            float3 dWi = normalize(-float3(dl.direction));
            float dCos = dot(facingNormal, dWi);
            if (dCos > 0.0) {
                float3 f; float pdfUnused;
                if (measuredEvalWorld(rgbGridData, tab, woLocal, dWi, tangent, bitangent, facingNormal, f, pdfUnused)) {
                    ray shadowRay;
                    shadowRay.origin = origin;
                    shadowRay.direction = dWi;
                    shadowRay.min_distance = 0.001f;
                    shadowRay.max_distance = kDirectionalLightMaxDistance;
                    if (traceShadowAny(isect, shadowRay, accelStructure, functionTable).type == intersection_type::none) {
                        float dTrans = (uniforms.fogSigmaT > 0.0)
                            ? exp(-uniforms.fogSigmaT * rayBoxExitDistance(shadowRay.origin, dWi, kRoomBoundsMin, kRoomBoundsMax))
                            : 1.0;
                        radiance += throughput * f * float3(dl.emission) * dCos * dTrans;
                    }
                }
            }
        }

        for (uint pji = 0; pji < uniforms.projectionLightCount; ++pji) {
            ProjectionLight pj = projectionLights[pji];
            float3 toP = float3(pj.position) - hitPoint;
            float pDistSq = dot(toP, toP);
            float pDist = sqrt(pDistSq);
            float3 pWi = toP / pDist;
            float pCos = dot(facingNormal, pWi);
            if (pCos > 0.0) {
                float3 pjRadiance = projectionLightRadiance(-pWi, pj.forward, pj.right, pj.up, pj.tanHalfFovX, pj.tanHalfFovY, pj.scale,
                                                            (pj.usePbrtTexture != 0u ? pbrtProjectionTexture : earthTexture), textureSampler);
                float3 f; float pdfUnused;
                if (any(pjRadiance > float3(0.0)) &&
                    measuredEvalWorld(rgbGridData, tab, woLocal, pWi, tangent, bitangent, facingNormal, f, pdfUnused)) {
                    ray shadowRay;
                    shadowRay.origin = origin;
                    shadowRay.direction = pWi;
                    shadowRay.min_distance = 0.001f;
                    shadowRay.max_distance = pDist - 0.002f;
                    reaimShadowRay(shadowRay, hitPoint, pDist);
                    if (traceShadowAny(isect, shadowRay, accelStructure, functionTable).type == intersection_type::none)
                        radiance += throughput * f * pjRadiance * pCos * exp(-uniforms.fogSigmaT * pDist) / pDistSq;
                }
            }
        }

        for (uint gli = 0; gli < uniforms.goniometricLightCount; ++gli) {
            GoniometricLight gl = goniometricLights[gli];
            float3 toG = float3(gl.position) - hitPoint;
            float gDistSq = dot(toG, toG);
            float gDist = sqrt(gDistSq);
            float3 gWi = toG / gDist;
            float gCos = dot(facingNormal, gWi);
            if (gCos > 0.0) {
                float3 glRadiance = goniometricLightRadiance(-gWi, gl.forward, gl.right, gl.up, gl.emission, gl.scale,
                                                             (gl.usePbrtTexture != 0u ? pbrtGoniometricTexture : goniometricTexture), textureSampler);
                float3 f; float pdfUnused;
                if (any(glRadiance > float3(0.0)) &&
                    measuredEvalWorld(rgbGridData, tab, woLocal, gWi, tangent, bitangent, facingNormal, f, pdfUnused)) {
                    ray shadowRay;
                    shadowRay.origin = origin;
                    shadowRay.direction = gWi;
                    shadowRay.min_distance = 0.001f;
                    shadowRay.max_distance = gDist - 0.002f;
                    reaimShadowRay(shadowRay, hitPoint, gDist);
                    if (traceShadowAny(isect, shadowRay, accelStructure, functionTable).type == intersection_type::none)
                        radiance += throughput * f * glRadiance * gCos * exp(-uniforms.fogSigmaT * gDist) / gDistSq;
                }
            }
        }

        // Image-based environments, importance-sampled with MIS (same gating as shadeConductor).
        if (envMapWidth > 0u && uniforms.useEnvironmentMap != 0u) {
            float envPdf;
            float3 envWi = sampleEnvironmentDirection(envMarginalCDF, envConditionalCDF, int(envMapWidth), int(envMapHeight),
                                                      randFloat(rngState), randFloat(rngState), envPdf);
            float envCos = dot(facingNormal, envWi);
            float3 f; float pdfBsdf;
            if (envCos > 0.0 && envPdf > 1e-9 &&
                measuredEvalWorld(rgbGridData, tab, woLocal, envWi, tangent, bitangent, facingNormal, f, pdfBsdf)) {
                ray shadowRay;
                shadowRay.origin = origin;
                shadowRay.direction = envWi;
                shadowRay.min_distance = 0.001f;
                shadowRay.max_distance = 1e5f;
                if (traceShadowAny(isect, shadowRay, accelStructure, functionTable).type == intersection_type::none) {
                    float3 envRadiance = earthTexture.sample(textureSampler, envMapUV(envWi)).rgb;
                    float envWeight = (envPdf * envPdf) / (envPdf * envPdf + pdfBsdf * pdfBsdf);
                    radiance += throughput * f * envRadiance * envCos / envPdf * envWeight;
                }
            }
        }
        if (pbrtEnvMapWidth > 0u) {
            float envPdf;
            float3 envWi = pbrtEnvSampleDirection(uniforms, pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, pbrtEnvMapWidth, pbrtEnvMapHeight, hitPoint,
                                                           randFloat(rngState), randFloat(rngState), envPdf);
            float envCos = dot(facingNormal, envWi);
            float3 f; float pdfBsdf;
            if (envCos > 0.0 && envPdf > 1e-9 &&
                measuredEvalWorld(rgbGridData, tab, woLocal, envWi, tangent, bitangent, facingNormal, f, pdfBsdf)) {
                ray shadowRay;
                shadowRay.origin = origin;
                shadowRay.direction = envWi;
                shadowRay.min_distance = 0.001f;
                shadowRay.max_distance = 1e5f;
                if (traceShadowAny(isect, shadowRay, accelStructure, functionTable).type == intersection_type::none) {
                    float3 envRadiance = pbrtEnvLeAt(uniforms, pbrtEnvConditionalCDF, pbrtEnvTexture, textureSampler, hitPoint, envWi);
                    float envWeight = (envPdf * envPdf) / (envPdf * envPdf + pdfBsdf * pdfBsdf);
                    radiance += throughput * f * envRadiance * envCos / envPdf * envWeight;
                }
            }
        }
    }

    float3 wiLocal, weight; float samplePdf;
    if (!measuredSampleF(rgbGridData, tab, woLocal, randFloat(rngState), randFloat(rngState), wiLocal, weight, samplePdf)) return false;
    throughput *= weight;
    rayDir = normalize(wiLocal.x * tangent + wiLocal.y * bitangent + wiLocal.z * facingNormal);
    rayOrigin = origin;
    bsdfPdf = samplePdf;
    specularBounce = false;
    return true;
}
