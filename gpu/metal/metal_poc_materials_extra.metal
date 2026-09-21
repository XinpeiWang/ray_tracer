inline float3 layeredCoatedConductorF(float3 wiLocal, float3 woLocal, float eta, float alpha,
                                       float3 conductorEta, float3 conductorK, thread uint& rngState) {
    if (wiLocal.z <= 0.0 || woLocal.z <= 0.0) return float3(0.0);
    float3 result = float3(0.0);

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

        beta *= exp(-kThickness / max(abs(w.z), 1e-6));
        bool atBottom = (w.z < 0.0);

        if (atBottom) {
            // GGX-conductor bottom bounce (ConductorBottomBounce::bounce()) -
            // flip to the conductor's own "incoming from above" frame,
            // sample a VNDF half-vector, reflect, weight by real complex
            // Fresnel times the height-correlated G/G1 ratio, and always
            // leave `w` pointing back upward.
            float3 fw = -w;
            float3 bwm = sampleGGXVNDF(fw, alpha, alpha, rngState);
            float cosC = dot(fw, bwm);
            float3 rwo = 2.0 * cosC * bwm - fw;
            float G1c = ggxG1(fw, alpha, alpha);
            float Gc = ggxG(rwo, fw, alpha, alpha);
            float wtC = (G1c > 1e-8) ? Gc / G1c : 0.0;
            beta *= frComplexRGB(cosC, conductorEta, conductorK) * wtC;
            w = float3(rwo.x, rwo.y, abs(rwo.z));
        } else {
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
    float alpha = max(mat.roughness, 0.0001);
    bool effectivelySmooth = alpha < 0.001;

    float3 tangent, bitangent;
    buildAnisotropicOnb(facingNormal, tangent, bitangent);
    float3 woWorld = -rayDir;
    float3 woLocal = float3(dot(woWorld, tangent), dot(woWorld, bitangent), dot(woWorld, facingNormal));
    woLocal.z = max(woLocal.z, 0.0001);
    float3 conductorEta = float3(mat.conductorEta);
    float3 conductorK = float3(mat.conductorK);

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
                float3 f = layeredCoatedConductorF(wiLocal, woLocal, mat.ior, alpha, conductorEta, conductorK, rngState);
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
                    float3 plF = layeredCoatedConductorF(plWiLocal, woLocal, mat.ior, alpha, conductorEta, conductorK, rngState);
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
                    float3 dlF = layeredCoatedConductorF(dlWiLocal, woLocal, mat.ior, alpha, conductorEta, conductorK, rngState);
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
                        float3 pjF = layeredCoatedConductorF(pjWiLocal, woLocal, mat.ior, alpha, conductorEta, conductorK, rngState);
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
                        float3 glF = layeredCoatedConductorF(glWiLocal, woLocal, mat.ior, alpha, conductorEta, conductorK, rngState);
                        float glTransmittance = exp(-uniforms.fogSigmaT * glDist);
                        radiance += throughput * glF * glRadiance * glCosSurface * glTransmittance;
                    }
                }
            }
        }
    }

    // Continuation ray: entrance test at the coat's own top surface,
    // then either a specular GGX reflection (probability F_in) or a
    // transmit -> single-bounce GGX-conductor reflection -> deterministic
    // exit test (probability 1-F_in) - NO retry loop needed (unlike
    // materialType 19's own diffuse-escape loop), since a conductor's
    // own bottom bounce is a single specular direction, not a spread
    // that can miss the coat's own exit cone and need another try.
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
        float3 wDown = 2.0 * cosI * wm - woLocal;
        if (wDown.z > 0.0) wDown.z = -wDown.z;
        if (wDown.z == 0.0) return false;

        float3 fw = -wDown;
        float3 bwm = sampleGGXVNDF(fw, alpha, alpha, rngState);
        float cosC = dot(fw, bwm);
        if (cosC <= 0.0) return false;
        float3 rwo = 2.0 * cosC * bwm - fw;
        if (rwo.z <= 0.0) return false;

        float G1c = ggxG1(fw, alpha, alpha);
        float Gc = ggxG(rwo, fw, alpha, alpha);
        float wtC = (G1c > 1e-8) ? Gc / G1c : 0.0;
        float3 Fc = frComplexRGB(cosC, conductorEta, conductorK) * wtC;

        // Coat-to-air exit test - inverted eta, matching OptiX's own
        // sample loop exactly (same convention shadeCoatedDiffuse()'s
        // own diffuse-escape loop uses, but a single deterministic test
        // here, not a retry loop - see this function's own header
        // comment).
        float Fout = frDielectric(rwo.z, 1.0 / mat.ior);
        float Tout = 1.0 - Fout;
        float Tin = 1.0 - Fin;
        beta = Fc * (Tin * Tout);
        newDirLocal = rwo;
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
            intersection_result<instancing, triangle_data> shadowResult =
                isect.intersect(shadowRay, accelStructure, functionTable);
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
                intersection_result<instancing, triangle_data> plShadowResult =
                    isect.intersect(plShadowRay, accelStructure, functionTable);
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
                    isect.intersect(dlShadowRay, accelStructure, functionTable);
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
                    intersection_result<instancing, triangle_data> pjShadowResult =
                        isect.intersect(pjShadowRay, accelStructure, functionTable);
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
                    intersection_result<instancing, triangle_data> glShadowResult =
                        isect.intersect(glShadowRay, accelStructure, functionTable);
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
                    isect.intersect(envShadowRay, accelStructure, functionTable);
                if (envShadowResult.type == intersection_type::none) {
                    float2 envUV = equirectangularUV(envWi);
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

    float pdfDiff = pDiff * cosWiL / M_PI_F;
    float pdfSpec = pSpec * principledGgxPdf(woLocal, woOutLocal, alpha);
    float pdfCoat = pCoat * principledGgxPdf(woLocal, woOutLocal, alphaCC);
    float pdf = pdfDiff + pdfSpec + pdfCoat;
    if (pdf < 1e-12) return false;

    float3 weight = totalCol * cosWiL / pdf;

    float3 newDirWorld = normalize(woOutLocal.x * tangent + woOutLocal.y * bitangent + woOutLocal.z * facingNormal);
    rayDir = newDirWorld;
    rayOrigin = hitPoint + facingNormal * 0.001f;
    throughput *= weight;
    specularBounce = true;
    return true;
}

