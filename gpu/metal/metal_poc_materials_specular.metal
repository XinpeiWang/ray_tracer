// ---------------------------------------------------------------------------
// Per-material shading functions
//
// Each function below implements ONE materialType's own NEE (where
// applicable) + BSDF-sampled-continuation logic - extracted out of
// primaryRayKernel's own `mat.materialType`-dispatched if/else chain
// (which used to inline all of this directly, growing that one function
// past 1,600 lines) so the kernel itself stays a short dispatcher. Every
// function takes the read-only per-hit context it needs (material,
// albedo, hit geometry) plus whatever scene resources its own NEE
// actually reads (only the four NEE-capable materials - conductor,
// clearcoat, diffuse transmission, Lambertian - need light buffers/
// textures/the intersector at all; the four purely-specular ones do
// not), and mutates the bounce loop's own running path state (`rayDir`,
// `rayOrigin`, `throughput`, `radiance`, `bsdfPdf`, `specularBounce`,
// `rngState`) by reference - the EXACT same variables the inlined code
// used to write to directly, just threaded through as `thread &`
// parameters instead of captured implicitly.
//
// Returns `false` only when this bounce's own sampled direction is
// genuinely invalid and the whole path should terminate (materialType
// 4/9's own below-the-hemisphere VNDF sample, the one `break` inside
// this whole dispatch chain before this refactor) - every other
// material always returns `true`. The caller checks this uniformly
// (`if (!shadeXxx(...)) break;`) for every branch, not just the one
// that can actually return false, so a future material added the same
// way doesn't need special-casing at the call site to get this right.

inline bool shadeMirror(float3 albedo, float3 hitPoint, float3 facingNormal,
                         thread float3& rayDir, thread float3& rayOrigin,
                         thread float3& throughput, thread bool& specularBounce) {
    // Mirror: deterministic reflection, no light sampling (a specular
    // surface has zero probability of the shadow ray toward a delta
    // light landing exactly on the reflection vector - NEE simply
    // doesn't apply here, same reason the CPU renderer's own BSDFs skip
    // NEE for specular lobes).
    //
    // Fresnel-weighted, not a flat `albedo` multiply the way every
    // earlier version of this branch did: a real mirror's reflectance
    // rises toward white/uncolored at grazing angles regardless of its
    // base tint (the same physical effect the GGX conductor material
    // already models via this exact function - `albedo` doubles as this
    // surface's own F0 here, the same "colour IS the normal-incidence
    // reflectance" convention that material already established).
    float3 newDir = reflect(rayDir, facingNormal);
    float cosTheta = max(dot(facingNormal, -rayDir), 0.0001);
    float3 fresnel = fresnelSchlickConductor(cosTheta, albedo);
    rayDir = newDir;
    rayOrigin = hitPoint + facingNormal * 0.001f;
    throughput *= fresnel;
    specularBounce = true;
    return true;
}

inline bool shadeDielectric(TriangleMaterial mat, float3 hitPoint, float3 normal, float3 facingNormal,
                             bool frontFace, float hitDistance,
                             thread float3& rayDir, thread float3& rayOrigin,
                             thread float3& throughput, thread bool& specularBounce, thread uint& rngState) {
    // Dielectric (glass): the exact Fresnel dielectric reflectance
    // (frDielectric(), see its own comment) decides reflect vs refract
    // stochastically each bounce - the "one importance-sampled choice
    // per hit, unbiased in expectation" approach pbrt-v4 and this
    // project's own CPU dielectric material use, not a 50/50 split of
    // energy.
    float refractionRatio = frontFace ? (1.0 / mat.ior) : mat.ior;
    float3 unitDir = normalize(rayDir);
    float cosTheta = min(dot(-unitDir, facingNormal), 1.0);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    bool cannotRefract = refractionRatio * sinTheta > 1.0;

    float3 newDir;
    if (cannotRefract || frDielectric(cosTheta, 1.0 / refractionRatio) > randFloat(rngState)) {
        newDir = reflect(unitDir, facingNormal);
    } else {
        newDir = refract(unitDir, facingNormal, refractionRatio);
    }
    rayDir = newDir;
    // Offset along the GEOMETRIC (unflipped-for-facing) normal signed
    // toward the new ray direction, not always `facingNormal` -
    // reflect() and refract() can each send the continuation ray to
    // either side of the surface here, and offsetting on the wrong side
    // re-intersects the same surface immediately (self-shadowing acne).
    rayOrigin = hitPoint + (dot(newDir, normal) > 0.0 ? normal : -normal) * 0.001f;
    applyBeerLambertAbsorption(throughput, mat.color, frontFace, hitDistance);
    specularBounce = true;
    return true;
}

// Recursive-backend dispersion (B23/B24, materialType 22) - the SAME
// simplified 3-representative-wavelength RGB-channel scheme OptiX's own
// recursive (non-wavefront) backend uses, per this scene's own registry
// comment (scene_registry_data.h) - NOT the real continuous spectral
// integration CPU's own `--spectral` path or GPU's own `--wavefront`
// path use, since this loader (like OptiX-recursive) has no per-
// wavelength camera ray/hero-wavelength infrastructure at all, only
// plain RGB throughput. Ported from `gpu/optix/optix_device_helpers.h`'s
// own `MaterialType::Dielectric` dispersive branch + `optix_raygen.h`'s
// own channel-masking step (there split across two programs by OptiX's
// own payload-register architecture; here, in ONE function, since this
// loader's shading kernel is already a single self-contained loop).
//
// Mechanism: `rgbChannel` is per-SAMPLE state (declared once before the
// bounce loop, `kRgbChannelUnset` = "no dispersive hit yet"). The FIRST
// time a path hits this material, a channel (0=R/1=G/2=B) is picked
// uniformly at random and PERSISTS for the rest of that sample (every
// later dispersive hit along the same path - e.g. exiting the same
// prism, or a second dispersive object - reuses it, never re-rolls) -
// matches CPU/wavefront's own "one hero wavelength for the whole path"
// convention. `throughput` is masked to that ONE channel with a
// compensating 3x weight AT THE MOMENT the channel is first chosen (a
// standard unbiased stochastic-channel-selection estimator: each of the
// 3 equally-likely channels, averaged over many samples, reconstructs
// the full-RGB expectation) - every later `radiance +=` naturally
// inherits this through throughput's own ongoing multiply chain, and a
// SAMPLE that never reaches a dispersive hit at all pays nothing extra.
// `kRgbChannelWavelengthNm` are the sRGB primaries' own commonly-cited
// dominant wavelengths - the SAME fixed values this project's own
// `measured` material (materialType 15, `src/TheRestOfYourLife/
// material_pbrt.h`'s `kLambdaR/G/B`) already uses for the identical
// "3 fixed representative wavelengths" purpose, not a fresh/independent
// choice.
//
// `mat.ior` = eta_d (unused directly - kept for parity/debugging only);
// `mat.conductorEta.x/y` reused as the precomputed Cauchy `(A, B)`
// coefficients (`CauchyCoefficientsFromAbbe()`, computed HOST-side once
// at scene-build time - construction-time math, not per-ray, matching
// CPU's own dielectric::make_dispersive() constructor exactly) - an
// otherwise-entirely-unused field for this materialType, the same
// "reuse a field with no other meaning here" convention every earlier
// materialType's own dual-use fields already follow.
constant uint kRgbChannelUnset = 3u;
constant float kRgbChannelWavelengthNm[3] = { 612.0f, 549.0f, 465.0f };

inline float cauchyEta(float lambdaNm, float A, float B) {
    float lambdaUm = lambdaNm * 0.001f;
    return A + B / (lambdaUm * lambdaUm);
}

inline bool shadeDispersiveDielectric(TriangleMaterial mat, float3 hitPoint, float3 normal, float3 facingNormal,
                             bool frontFace, float hitDistance,
                             thread float3& rayDir, thread float3& rayOrigin,
                             thread float3& throughput, thread bool& specularBounce,
                             thread uint& rngState, thread uint& rgbChannel) {
    if (rgbChannel == kRgbChannelUnset) {
        uint newChannel = min(uint(randFloat(rngState) * 3.0), 2u);
        float3 channelMask = float3(newChannel == 0u ? 3.0 : 0.0,
                                     newChannel == 1u ? 3.0 : 0.0,
                                     newChannel == 2u ? 3.0 : 0.0);
        throughput *= channelMask;
        rgbChannel = newChannel;
    }
    float dielectricIor = cauchyEta(kRgbChannelWavelengthNm[rgbChannel],
                                     mat.conductorEta.x, mat.conductorEta.y);

    float refractionRatio = frontFace ? (1.0 / dielectricIor) : dielectricIor;
    float3 unitDir = normalize(rayDir);
    float cosTheta = min(dot(-unitDir, facingNormal), 1.0);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    bool cannotRefract = refractionRatio * sinTheta > 1.0;

    float3 newDir;
    if (cannotRefract || frDielectric(cosTheta, 1.0 / refractionRatio) > randFloat(rngState)) {
        newDir = reflect(unitDir, facingNormal);
    } else {
        newDir = refract(unitDir, facingNormal, refractionRatio);
    }
    rayDir = newDir;
    rayOrigin = hitPoint + (dot(newDir, normal) > 0.0 ? normal : -normal) * 0.001f;
    applyBeerLambertAbsorption(throughput, mat.color, frontFace, hitDistance);
    specularBounce = true;
    return true;
}

inline bool shadeRoughDielectric(TriangleMaterial mat, float3 hitPoint, float3 normal, float3 facingNormal,
                                  bool frontFace, float hitDistance,
                                  thread float3& rayDir, thread float3& rayOrigin,
                                  thread float3& throughput, thread bool& specularBounce, thread uint& rngState) {
    // Rough (frosted) dielectric: the same Schlick-Fresnel reflect-vs-
    // refract decision as materialType 2, but taken about a GGX-VNDF-
    // SAMPLED microfacet normal instead of the smooth geometric one.
    // Energy-conserving throughput correction (G/G1(wo)) derived from
    // src/shared/bxdfs_conductor.h's own validated RoughDielectricBxDF
    // f()/pdf() pair - see docs/METAL_GPU_FEASIBILITY.md section 65 for
    // the full derivation.
    float alpha = max(mat.roughness * mat.roughness, 0.0009);
    float3 tangent, bitangent;
    buildOnb(facingNormal, tangent, bitangent);
    float3 woWorld = -rayDir;
    float3 woLocal = float3(dot(woWorld, tangent), dot(woWorld, bitangent), dot(woWorld, facingNormal));
    woLocal.z = max(woLocal.z, 0.0001);
    float3 hLocal = sampleGGXVNDF(woLocal, alpha, alpha, rngState);
    float3 hWorld = normalize(hLocal.x * tangent + hLocal.y * bitangent + hLocal.z * facingNormal);

    float refractionRatio = frontFace ? (1.0 / mat.ior) : mat.ior;
    float3 unitDir = normalize(rayDir);
    float cosTheta = clamp(dot(-unitDir, hWorld), 0.0, 1.0);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    bool cannotRefract = refractionRatio * sinTheta > 1.0;

    float3 newDir;
    if (cannotRefract || frDielectric(cosTheta, 1.0 / refractionRatio) > randFloat(rngState)) {
        newDir = reflect(unitDir, hWorld);
    } else {
        newDir = refract(unitDir, hWorld, refractionRatio);
    }
    float3 newDirLocal = float3(dot(newDir, tangent), dot(newDir, bitangent), dot(newDir, facingNormal));
    float roughDielectricG = ggxG(woLocal, newDirLocal, alpha, alpha);
    float roughDielectricG1 = ggxG1(woLocal, alpha, alpha);
    throughput *= roughDielectricG / max(roughDielectricG1, 1e-6);
    rayDir = newDir;
    rayOrigin = hitPoint + (dot(newDir, normal) > 0.0 ? normal : -normal) * 0.001f;
    applyBeerLambertAbsorption(throughput, mat.color, frontFace, hitDistance);
    specularBounce = true;
    return true;
}

// B24: the frosted (rough) sibling of materialType 22's own smooth
// dispersive dielectric - same OptiX reference, same
// `MaterialType::RoughDielectric` dispersive branch (a few lines below
// its own smooth `MaterialType::Dielectric` counterpart in
// `optix_device_helpers.h`, deliberately NOT factored into a shared
// helper there either - see that code's own comment on why "exactly 2
// occurrences" isn't worth it). Structurally just
// `shadeRoughDielectric()` (materialType 5) with the SAME `rgbChannel`
// resolution `shadeDispersiveDielectric()` (materialType 22) already
// uses, substituted in place of the flat `mat.ior` - the two dispersive
// materials share the exact same channel-selection mechanism, only the
// underlying (smooth vs. GGX-rough) BSDF differs, matching how
// OptiX's own two dispersive branches are independently-but-identically
// structured. `mat.conductorEta.x/y` is the SAME dual-use Cauchy
// `(A, B)` pair materialType 22 already established (both materials
// otherwise leave that field unused).
inline bool shadeDispersiveRoughDielectric(TriangleMaterial mat, float3 hitPoint, float3 normal, float3 facingNormal,
                                  bool frontFace, float hitDistance,
                                  thread float3& rayDir, thread float3& rayOrigin,
                                  thread float3& throughput, thread bool& specularBounce,
                                  thread uint& rngState, thread uint& rgbChannel) {
    if (rgbChannel == kRgbChannelUnset) {
        uint newChannel = min(uint(randFloat(rngState) * 3.0), 2u);
        float3 channelMask = float3(newChannel == 0u ? 3.0 : 0.0,
                                     newChannel == 1u ? 3.0 : 0.0,
                                     newChannel == 2u ? 3.0 : 0.0);
        throughput *= channelMask;
        rgbChannel = newChannel;
    }
    float dispersiveIor = cauchyEta(kRgbChannelWavelengthNm[rgbChannel],
                                     mat.conductorEta.x, mat.conductorEta.y);

    float alpha = max(mat.roughness * mat.roughness, 0.0009);
    float3 tangent, bitangent;
    buildOnb(facingNormal, tangent, bitangent);
    float3 woWorld = -rayDir;
    float3 woLocal = float3(dot(woWorld, tangent), dot(woWorld, bitangent), dot(woWorld, facingNormal));
    woLocal.z = max(woLocal.z, 0.0001);
    float3 hLocal = sampleGGXVNDF(woLocal, alpha, alpha, rngState);
    float3 hWorld = normalize(hLocal.x * tangent + hLocal.y * bitangent + hLocal.z * facingNormal);

    float refractionRatio = frontFace ? (1.0 / dispersiveIor) : dispersiveIor;
    float3 unitDir = normalize(rayDir);
    float cosTheta = clamp(dot(-unitDir, hWorld), 0.0, 1.0);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    bool cannotRefract = refractionRatio * sinTheta > 1.0;

    float3 newDir;
    if (cannotRefract || frDielectric(cosTheta, 1.0 / refractionRatio) > randFloat(rngState)) {
        newDir = reflect(unitDir, hWorld);
    } else {
        newDir = refract(unitDir, hWorld, refractionRatio);
    }
    float3 newDirLocal = float3(dot(newDir, tangent), dot(newDir, bitangent), dot(newDir, facingNormal));
    float roughDielectricG = ggxG(woLocal, newDirLocal, alpha, alpha);
    float roughDielectricG1 = ggxG1(woLocal, alpha, alpha);
    throughput *= roughDielectricG / max(roughDielectricG1, 1e-6);
    rayDir = newDir;
    rayOrigin = hitPoint + (dot(newDir, normal) > 0.0 ? normal : -normal) * 0.001f;
    applyBeerLambertAbsorption(throughput, mat.color, frontFace, hitDistance);
    specularBounce = true;
    return true;
}

inline bool shadeThinDielectric(TriangleMaterial mat, float3 hitPoint, float3 normal, float3 facingNormal,
                                 thread float3& rayDir, thread float3& rayOrigin,
                                 thread float3& throughput, thread bool& specularBounce, thread uint& rngState) {
    // Thin dielectric (pbrt-v4's own ThinDielectricBxDF) - a zero-
    // thickness slab: transmission passes straight through with no
    // bending, reflectance boosted by a closed-form multi-bounce
    // geometric series. Same `ior` regardless of front/back face - see
    // docs/METAL_GPU_FEASIBILITY.md section 64.
    float cosTheta = max(abs(dot(facingNormal, -rayDir)), 0.0001);
    float thinR = frDielectric(cosTheta, mat.ior);
    if (thinR < 1.0) {
        float thinT = 1.0 - thinR;
        thinR += thinT * thinT * thinR / max(1.0 - thinR * thinR, 1e-6);
    }
    float3 newDir = (randFloat(rngState) < thinR)
        ? reflect(rayDir, facingNormal)
        : rayDir; // straight-through, no bending - zero-thickness slab
    rayDir = newDir;
    rayOrigin = hitPoint + (dot(newDir, normal) > 0.0 ? normal : -normal) * 0.001f;
    // No Beer-Lambert absorption, no material tint - pbrt-v4's own
    // ThinDielectricBxDF::Sample_f hardcodes r=g=b=1, a deliberate,
    // faithful match to the reference, not an oversight.
    specularBounce = true;
    return true;
}

inline bool shadeConductor(TriangleMaterial mat, float3 hitPoint, float3 normal, float3 facingNormal,
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
                            device const float* ggxEnergyTable,
                            uint ggxEnergyRoughRes, uint ggxEnergyMuRes,
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
    // Rough conductor (GGX metal): structurally the same NEE + BSDF-
    // sampled-continuation + MIS shape as the Lambertian material below -
    // only the BRDF/sampling math changes, from a cosine-weighted
    // diffuse lobe to an importance-sampled microfacet one. Fresnel here
    // is `frComplexRGB(..., mat.conductorEta, mat.conductorK)` - the
    // real per-channel complex conductor Fresnel (section 66) - `albedo`
    // is unread by this material, unlike materialType 1's mirror.
    //
    // Genuinely ANISOTROPIC (materialType 4): `ior` gives alphaX,
    // `roughness` doubles as alphaY (0.0 falls back to isotropic).
    // Genuinely SPATIALLY-VARYING instead (materialType 9): alpha is
    // isotropic at any one point, but which of two roughness values
    // applies switches across the surface via a UV-space checker
    // pattern.
    float alphaX, alphaY;
    if (mat.materialType == 9u) {
        float2 roughnessUV = equirectangularUV(normal);
        float alphaSmooth = max(mat.ior * mat.ior, 0.0009);
        float alphaRough = max(mat.roughness * mat.roughness, 0.0009);
        alphaX = checkerColor(roughnessUV, 6.0, float3(alphaSmooth), float3(alphaRough)).x;
        alphaY = alphaX;
    } else {
        alphaX = max(mat.ior * mat.ior, 0.0009);
        alphaY = (mat.roughness > 0.0) ? max(mat.roughness * mat.roughness, 0.0009) : alphaX;
    }
    float3 tangent, bitangent;
    buildAnisotropicOnb(facingNormal, tangent, bitangent);
    float3 woWorld = -rayDir;
    float3 woLocal = float3(dot(woWorld, tangent), dot(woWorld, bitangent), dot(woWorld, facingNormal));
    woLocal.z = max(woLocal.z, 0.0001);

    // Multi-scatter energy compensation (section 72/73) - `energyScale`
    // recovers the energy single-scatter GGX discards to inter-
    // reflection between microfacets, applied as a flat multiplier on
    // every BRDF value below AND the continuation ray's own throughput
    // update at this function's own tail, all scaled by the SAME factor
    // since it depends only on this hit's own (alpha, view angle), not
    // on which light/direction is being evaluated. The table itself is
    // isotropic-only (built from a single alpha, section 72's own
    // buildGGXEnergyTable()); materialType 4's own genuinely anisotropic
    // alphaX/alphaY collapse to a representative isotropic
    // sqrt(alphaX*alphaY) for this lookup - an approximation, not exact,
    // but the SAME kind of "isotropic energy term applied to an
    // anisotropic lobe" approximation production renderers (including
    // Cycles itself) commonly make, since a full anisotropic energy
    // table would need a third table axis this phase doesn't build.
    float ggxEnergyIsoAlpha = sqrt(alphaX * alphaY);
    float ggxEnergyRoughness = sqrt(ggxEnergyIsoAlpha);
    float ggxE = sampleGGXEnergyTableDevice(ggxEnergyTable, ggxEnergyRoughRes, ggxEnergyMuRes,
                                             ggxEnergyRoughness, woLocal.z);
    float energyScale = 1.0 / max(ggxE, 0.05);

    if (all(mat.emission == float3(0.0))) {
        LightSample ls = sampleAreaLight(lights, uniforms.lightCount, rngState, pbrtAreaLightTexture, textureSampler);
        float3 toLight = ls.point - hitPoint;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        float3 wi = toLight / dist;
        float cosSurface = dot(facingNormal, wi);
        float cosLight = dot(ls.normal, -wi);
        if (cosSurface > 0.0 && (cosLight > 0.0 || (ls.twoSided != 0.0 && cosLight < 0.0))) {
            float3 wiLocal = float3(dot(wi, tangent), dot(wi, bitangent), dot(wi, facingNormal));
            float3 h = normalize(woLocal + wiLocal);
            float NdotO = woLocal.z;
            float NdotI = max(wiLocal.z, 0.0001);
            float Dh = ggxD(h, alphaX, alphaY);
            float G = ggxG(woLocal, wiLocal, alphaX, alphaY);
            float3 F = frComplexRGB(max(dot(woLocal, h), 0.0), mat.conductorEta, mat.conductorK);
            float3 brdf = Dh * G * F * energyScale / max(4.0 * NdotO * NdotI, 1e-6);

            ray shadowRay;
            shadowRay.origin = hitPoint + facingNormal * 0.001f;
            shadowRay.direction = wi;
            shadowRay.min_distance = 0.001f;
            shadowRay.max_distance = dist - 0.002f;
            intersection_result<instancing, triangle_data> shadowResult =
                isect.intersect(shadowRay, accelStructure, functionTable);
            if (shadowResult.type == intersection_type::none) {
                float pdfSolidAngle = (distSq / (ls.area * abs(cosLight))) * ls.pmf;
                float pdfBsdf = (Dh * ggxG1(woLocal, alphaX, alphaY)) / max(4.0 * NdotO, 1e-6);
                float weight = (pdfSolidAngle * pdfSolidAngle)
                    / (pdfSolidAngle * pdfSolidAngle + pdfBsdf * pdfBsdf);
                float transmittance = exp(-uniforms.fogSigmaT * dist);
                radiance += throughput * brdf * ls.emission * cosSurface * transmittance / pdfSolidAngle * weight;
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
                float3 plWiLocal = float3(dot(plWi, tangent), dot(plWi, bitangent), dot(plWi, facingNormal));
                float3 plH = normalize(woLocal + plWiLocal);
                float plNdotO = woLocal.z;
                float plNdotI = max(plWiLocal.z, 0.0001);
                float plDh = ggxD(plH, alphaX, alphaY);
                float plG = ggxG(woLocal, plWiLocal, alphaX, alphaY);
                float3 plF = frComplexRGB(max(dot(woLocal, plH), 0.0), mat.conductorEta, mat.conductorK);
                float3 plBrdf = plDh * plG * plF * energyScale / max(4.0 * plNdotO * plNdotI, 1e-6);

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
                    radiance += throughput * plBrdf * float3(pl.emission) * plCosSurface * plSpot * plTransmittance / plDistSq;
                }
            }
        }

        for (uint dli = 0; dli < uniforms.directionalLightCount; ++dli) {
            DirectionalLight dl = directionalLights[dli];
            float3 dlWi = normalize(-float3(dl.direction));
            float dlCosSurface = dot(facingNormal, dlWi);
            if (dlCosSurface > 0.0) {
                float3 dlWiLocal = float3(dot(dlWi, tangent), dot(dlWi, bitangent), dot(dlWi, facingNormal));
                float3 dlH = normalize(woLocal + dlWiLocal);
                float dlNdotO = woLocal.z;
                float dlNdotI = max(dlWiLocal.z, 0.0001);
                float dlDh = ggxD(dlH, alphaX, alphaY);
                float dlG = ggxG(woLocal, dlWiLocal, alphaX, alphaY);
                float3 dlF = frComplexRGB(max(dot(woLocal, dlH), 0.0), mat.conductorEta, mat.conductorK);
                float3 dlBrdf = dlDh * dlG * dlF * energyScale / max(4.0 * dlNdotO * dlNdotI, 1e-6);

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
                    radiance += throughput * dlBrdf * float3(dl.emission) * dlCosSurface * dlTransmittance;
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
                    float3 pjWiLocal = float3(dot(pjWi, tangent), dot(pjWi, bitangent), dot(pjWi, facingNormal));
                    float3 pjH = normalize(woLocal + pjWiLocal);
                    float pjNdotO = woLocal.z;
                    float pjNdotI = max(pjWiLocal.z, 0.0001);
                    float pjDh = ggxD(pjH, alphaX, alphaY);
                    float pjG = ggxG(woLocal, pjWiLocal, alphaX, alphaY);
                    float3 pjF = frComplexRGB(max(dot(woLocal, pjH), 0.0), mat.conductorEta, mat.conductorK);
                    float3 pjBrdf = pjDh * pjG * pjF * energyScale / max(4.0 * pjNdotO * pjNdotI, 1e-6);

                    ray pjShadowRay;
                    pjShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    pjShadowRay.direction = pjWi;
                    pjShadowRay.min_distance = 0.001f;
                    pjShadowRay.max_distance = pjDist - 0.002f;
                    intersection_result<instancing, triangle_data> pjShadowResult =
                        isect.intersect(pjShadowRay, accelStructure, functionTable);
                    if (pjShadowResult.type == intersection_type::none) {
                        float pjTransmittance = exp(-uniforms.fogSigmaT * pjDist);
                        radiance += throughput * pjBrdf * pjRadiance * pjCosSurface * pjTransmittance / pjDistSq;
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
                    float3 glWiLocal = float3(dot(glWi, tangent), dot(glWi, bitangent), dot(glWi, facingNormal));
                    float3 glH = normalize(woLocal + glWiLocal);
                    float glNdotO = woLocal.z;
                    float glNdotI = max(glWiLocal.z, 0.0001);
                    float glDh = ggxD(glH, alphaX, alphaY);
                    float glG = ggxG(woLocal, glWiLocal, alphaX, alphaY);
                    float3 glF = frComplexRGB(max(dot(woLocal, glH), 0.0), mat.conductorEta, mat.conductorK);
                    float3 glBrdf = glDh * glG * glF * energyScale / max(4.0 * glNdotO * glNdotI, 1e-6);

                    ray glShadowRay;
                    glShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    glShadowRay.direction = glWi;
                    glShadowRay.min_distance = 0.001f;
                    glShadowRay.max_distance = glDist - 0.002f;
                    intersection_result<instancing, triangle_data> glShadowResult =
                        isect.intersect(glShadowRay, accelStructure, functionTable);
                    if (glShadowResult.type == intersection_type::none) {
                        float glTransmittance = exp(-uniforms.fogSigmaT * glDist);
                        radiance += throughput * glBrdf * glRadiance * glCosSurface * glTransmittance / glDistSq;
                    }
                }
            }
        }

        // Environment map (importance-sampled NEE, section 71) - an
        // additional light-sampling strategy alongside the ones above,
        // not a replacement: see shadeLambertian's own comment for the
        // full "why." No fog-transmittance factor here (unlike the
        // delta lights above) - the miss-path's own existing
        // unconditional environment contribution (primaryRayKernel's
        // own comment) never applied one either, so this NEE addition
        // stays consistent with that pre-existing limitation rather
        // than introducing a new correctness asymmetry between the two.
        // envMapWidth>0 alone isn't enough to gate this - it's built
        // from earthPixels UNCONDITIONALLY (metal_poc.mm), independent of
        // whether the current scene's own miss path actually uses
        // earthTexture as its sky. A pbrt-loaded scene always sets
        // useEnvironmentMap=0 (its own sky, if any, replaces earthTexture
        // - see buildScene()'s own comment there), so without this check
        // every pbrt scene's materials would incorrectly importance-
        // sample and add light from earthTexture as if it were the
        // active environment, even though nothing in the miss path ever
        // shows it as sky for that render - a real bug found while
        // adding the pbrt-env NEE block below (section 96).
        if (envMapWidth > 0u && uniforms.useEnvironmentMap != 0u) {
            float envPdfSolidAngle;
            float3 envWi = sampleEnvironmentDirection(envMarginalCDF, envConditionalCDF,
                                                       int(envMapWidth), int(envMapHeight),
                                                       randFloat(rngState), randFloat(rngState), envPdfSolidAngle);
            float envCosSurface = dot(facingNormal, envWi);
            if (envCosSurface > 0.0 && envPdfSolidAngle > 1e-9) {
                float3 envWiLocal = float3(dot(envWi, tangent), dot(envWi, bitangent), dot(envWi, facingNormal));
                float3 envH = normalize(woLocal + envWiLocal);
                float envNdotO = woLocal.z;
                float envNdotI = max(envWiLocal.z, 0.0001);
                float envDh = ggxD(envH, alphaX, alphaY);
                float envG = ggxG(woLocal, envWiLocal, alphaX, alphaY);
                float3 envF = frComplexRGB(max(dot(woLocal, envH), 0.0), mat.conductorEta, mat.conductorK);
                float3 envBrdf = envDh * envG * envF * energyScale / max(4.0 * envNdotO * envNdotI, 1e-6);

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
                    float envPdfBsdf = (envDh * ggxG1(woLocal, alphaX, alphaY)) / max(4.0 * envNdotO, 1e-6);
                    float envWeight = (envPdfSolidAngle * envPdfSolidAngle)
                        / (envPdfSolidAngle * envPdfSolidAngle + envPdfBsdf * envPdfBsdf);
                    radiance += throughput * envBrdf * envRadiance * envCosSurface / envPdfSolidAngle * envWeight;
                }
            }
        }

        // Same NEE/MIS strategy, for a pbrt-loaded scene's own SEPARATE
        // image-based infinite light (section 96) - a genuinely
        // different texture/CDF pair, gated on its own pbrtEnvMapWidth
        // rather than envMapWidth so it's a pure no-op for every scene
        // without one.
        if (pbrtEnvMapWidth > 0u) {
            float pbrtEnvPdfSolidAngle;
            float3 pbrtEnvWi = sampleEnvironmentDirection(pbrtEnvMarginalCDF, pbrtEnvConditionalCDF,
                                                           int(pbrtEnvMapWidth), int(pbrtEnvMapHeight),
                                                           randFloat(rngState), randFloat(rngState), pbrtEnvPdfSolidAngle);
            float pbrtEnvCosSurface = dot(facingNormal, pbrtEnvWi);
            if (pbrtEnvCosSurface > 0.0 && pbrtEnvPdfSolidAngle > 1e-9) {
                float3 pbrtEnvWiLocal = float3(dot(pbrtEnvWi, tangent), dot(pbrtEnvWi, bitangent), dot(pbrtEnvWi, facingNormal));
                float3 pbrtEnvH = normalize(woLocal + pbrtEnvWiLocal);
                float pbrtEnvNdotO = woLocal.z;
                float pbrtEnvNdotI = max(pbrtEnvWiLocal.z, 0.0001);
                float pbrtEnvDh = ggxD(pbrtEnvH, alphaX, alphaY);
                float pbrtEnvG = ggxG(woLocal, pbrtEnvWiLocal, alphaX, alphaY);
                float3 pbrtEnvF = frComplexRGB(max(dot(woLocal, pbrtEnvH), 0.0), mat.conductorEta, mat.conductorK);
                float3 pbrtEnvBrdf = pbrtEnvDh * pbrtEnvG * pbrtEnvF * energyScale / max(4.0 * pbrtEnvNdotO * pbrtEnvNdotI, 1e-6);

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
                    float pbrtEnvPdfBsdf = (pbrtEnvDh * ggxG1(woLocal, alphaX, alphaY)) / max(4.0 * pbrtEnvNdotO, 1e-6);
                    float pbrtEnvWeight = (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle)
                        / (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle + pbrtEnvPdfBsdf * pbrtEnvPdfBsdf);
                    radiance += throughput * pbrtEnvBrdf * pbrtEnvRadianceSample * pbrtEnvCosSurface / pbrtEnvPdfSolidAngle * pbrtEnvWeight;
                }
            }
        }
    }

    float3 hLocal = sampleGGXVNDF(woLocal, alphaX, alphaY, rngState);
    float3 wiLocal = reflect(-woLocal, hLocal);
    if (wiLocal.z <= 0.0) {
        // Sampled a half-vector whose reflection lands below the
        // hemisphere (possible at grazing angles/high roughness) - a
        // real BRDF value of zero, not a bug; terminate this path
        // rather than continue with an invalid direction.
        return false;
    }
    float3 wiWorld = normalize(wiLocal.x * tangent + wiLocal.y * bitangent + wiLocal.z * facingNormal);

    float NdotO = woLocal.z;
    float G = ggxG(woLocal, wiLocal, alphaX, alphaY);
    float G1 = ggxG1(woLocal, alphaX, alphaY);
    float3 F = frComplexRGB(max(dot(woLocal, hLocal), 0.0), mat.conductorEta, mat.conductorK);
    throughput *= F * (G / max(G1, 1e-6)) * energyScale;

    rayDir = wiWorld;
    rayOrigin = hitPoint + facingNormal * 0.001f;
    bsdfPdf = (ggxD(hLocal, alphaX, alphaY) * G1) / max(4.0 * NdotO, 1e-6);
    specularBounce = false;
    return true;
}

inline bool shadeClearcoat(TriangleMaterial mat, float3 albedo, float3 hitPoint, float3 facingNormal,
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
    // Clearcoat (glossy plastic/car-paint) - `albedo` is the diffuse
    // base colour underneath the coat (materialType 0's own convention),
    // NOT an F0. A stochastic MIX of two lobes: a colourless specular
    // coat (fixed IOR 1.5, F0 = 0.04) with probability exactly equal to
    // its own reflectance (no extra scaling needed - see this material's
    // own section 51), or the diffuse base with the complementary
    // probability (full NEE + cosine-sampling, duplicated here rather
    // than shared with the Lambertian material below since the
    // stochastic coat-vs-base decision has to happen first).
    //
    // Entering-light coat attenuation (below): every diffuse-base
    // contribution is additionally scaled by `(1 - frDielectric(cosX,
    // kClearcoatEta))` at ITS OWN incidence angle - light reaching the
    // diffuse layer from any direction must first penetrate the SAME
    // dielectric coat, independent of which outgoing/view direction is
    // being evaluated. This is distinct from (and not already covered
    // by) the `coatFresnel` probability below: that term already
    // correctly reproduces the OUTGOING-direction attenuation in
    // expectation via unweighted stochastic lobe selection (a standard,
    // unbiased one-sample MC estimator - P(diffuse)=1-coatFresnel(wo),
    // then evaluate the chosen lobe's own BRDF unweighted), so adding an
    // extra `(1-coatFresnel)` factor here would double-count it. pbrt-v4's
    // `NormalizedFresnelBxDF` (src/shared/bxdfs_layered.h) also has a `c`
    // energy-renormalization constant (`1 - 2*FresnelMoment1(1/eta)`)
    // accounting for light trapped and re-emitted by internal reflection
    // inside the diffuse layer - deliberately DEFERRED here, same staging
    // as Oren-Nayar's/GGX's own multi-scatter compensation splits, since
    // it needs pbrt's fuller layered-BxDF stochastic-transport context to
    // get right rather than being a simple standalone factor.
    float cosThetaCoat = max(dot(facingNormal, -rayDir), 0.0001);
    const float kClearcoatF0 = 0.04;
    const float kClearcoatEta = 1.5;
    float coatFresnel = fresnelSchlickConductor(cosThetaCoat, float3(kClearcoatF0)).x;
    if (randFloat(rngState) < coatFresnel) {
        float3 newDir = reflect(rayDir, facingNormal);
        rayDir = newDir;
        rayOrigin = hitPoint + facingNormal * 0.001f;
        specularBounce = true;
    } else {
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
                    float coatTransmitIn = 1.0 - frDielectric(cosSurface, kClearcoatEta);
                    radiance += throughput * albedo * (1.0 / M_PI_F) * coatTransmitIn
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
                        float plCoatTransmitIn = 1.0 - frDielectric(plCosSurface, kClearcoatEta);
                        radiance += throughput * albedo * (1.0 / M_PI_F) * plCoatTransmitIn
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
                        // room bounds (see that function's own comment for
                        // why calling it unconditionally is unsafe).
                        float dlTransmittance = (uniforms.fogSigmaT > 0.0)
                            ? exp(-uniforms.fogSigmaT * rayBoxExitDistance(dlShadowRay.origin, dlWi, kRoomBoundsMin, kRoomBoundsMax))
                            : 1.0;
                        float dlCoatTransmitIn = 1.0 - frDielectric(dlCosSurface, kClearcoatEta);
                        radiance += throughput * albedo * (1.0 / M_PI_F) * dlCoatTransmitIn
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
                            float pjCoatTransmitIn = 1.0 - frDielectric(pjCosSurface, kClearcoatEta);
                            radiance += throughput * albedo * (1.0 / M_PI_F) * pjCoatTransmitIn
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
                            float glCoatTransmitIn = 1.0 - frDielectric(glCosSurface, kClearcoatEta);
                            radiance += throughput * albedo * (1.0 / M_PI_F) * glCoatTransmitIn
                                        * glRadiance * glCosSurface * glTransmittance / glDistSq;
                        }
                    }
                }
            }

            // Environment map (importance-sampled NEE, section 71) - see
            // shadeLambertian's own comment for the full "why." Gated on
            // useEnvironmentMap too, not just envMapWidth - see
            // shadeConductor's own comment on why (envMapWidth alone
            // stays nonzero for a pbrt-loaded scene, which always sets
            // useEnvironmentMap=0).
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
                        float envCoatTransmitIn = 1.0 - frDielectric(envCosSurface, kClearcoatEta);
                        radiance += throughput * albedo * (1.0 / M_PI_F) * envCoatTransmitIn
                                    * envRadiance * envCosSurface / envPdfSolidAngle * envWeight;
                    }
                }
            }

            // Same NEE/MIS strategy, for a pbrt-loaded scene's own
            // SEPARATE image-based infinite light (section 96) - see
            // shadeConductor's own comment.
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
                        float pbrtEnvCoatTransmitIn = 1.0 - frDielectric(pbrtEnvCosSurface, kClearcoatEta);
                        radiance += throughput * albedo * (1.0 / M_PI_F) * pbrtEnvCoatTransmitIn
                                    * pbrtEnvRadianceSample * pbrtEnvCosSurface / pbrtEnvPdfSolidAngle * pbrtEnvWeight;
                    }
                }
            }
        }

        rayDir = cosineSampleHemisphere(facingNormal, rngState);
        rayOrigin = hitPoint + facingNormal * 0.001f;
        float coatTransmitInNewDir = 1.0 - frDielectric(max(dot(facingNormal, rayDir), 0.0001), kClearcoatEta);
        throughput *= albedo * coatTransmitInNewDir;
        bsdfPdf = max(dot(facingNormal, rayDir), 0.0001) / M_PI_F;
        specularBounce = false;
    }
    return true;
}

