// pbrt-v4 LayeredBxDF<DielectricBxDF, DiffuseBxDF | ConductorBxDF, twoSided> in MSL: a port of src/shared/bxdfs_layered.h
// (itself a port of pbrt-v4's bxdfs.h), shared by shadeCoatedDiffuse() and shadeCoatedConductor(). It replaces an earlier
// hand port of OptiX's simplified single-bounce model, which never refracted at the coat and gave the three backends three
// different answers (see docs/PBRT_SUPPORT.md, coatedconductor).
//
// Conventions are pbrt's: directions are in the local shading frame (z = normal) pointing AWAY from the surface; `wo` is the
// direction toward the camera, `wi` the direction toward the light. The 1/eta'^2 factors of radiance-mode transmission cancel
// between entering and leaving the layer, so a coated BSDF carries no net eta^2.
//   layeredSample(): the random walk's own sample, f and pdf are running products - the unbiased path weight is
//                    f*|cos(wi)|/pdf (pbrt's BSDFSample::pdfIsProportional);
//   layeredPdf():    pbrt's approximate PDF(), used ONLY for MIS (seeded from the two directions so the light sample and the
//                    BSDF sample of the same direction see the same number);
//   layeredF():      the stochastic BSDF value for next-event estimation.
// Isotropic roughness, no scattering medium inside the layer (albedo = 0), maxDepth 10, nSamples 1 - what every scene here uses.
//
// NOT compiled or run on the machine that wrote it (no Metal toolchain): the algorithm is the C++ one in src/shared, which is
// covered by unit tests there (closed form for the smooth case, sampler vs f() integral for the rest); this file is a line-by-line
// translation to MSL, so build it on a Mac and run the shader tests before relying on it.

struct LayerSample {
    float3 f;
    float3 wi;
    float pdf;
    bool valid;
    bool reflection;
    bool transmission;
    bool specular;
};

// The coat (a dielectric interface) and the base (a Lambertian or a conductor).
struct LayerTop {
    float eta;     // the coat's IOR
    float alpha;   // GGX alpha
};

struct LayerBottom {
    int kind;      // 0 = Lambertian (albedo), 1 = conductor (eta, k - already divided by the coat's IOR)
    float3 albedo;
    float3 eta;
    float3 k;
    float alpha;   // conductor's GGX alpha
};

constant int kLayerReflection   = 1;
constant int kLayerTransmission = 2;
constant int kLayerAllLobes     = 3;
constant int kLayerMaxDepth     = 10;

inline LayerSample layerNoSample() {
    LayerSample s;
    s.f = float3(0.0);
    s.wi = float3(0.0, 0.0, 1.0);
    s.pdf = 0.0;
    s.valid = false;
    s.reflection = false;
    s.transmission = false;
    s.specular = false;
    return s;
}

inline bool layerSameHemisphere(float3 a, float3 b) { return a.z * b.z > 0.0; }
inline bool layerTopSmooth(LayerTop t) { return t.eta == 1.0 || t.alpha < 1.0e-3; }
inline bool layerBottomSpecular(LayerBottom b) { return b.kind == 1 && b.alpha < 1.0e-3; }
// pbrt's Flags(): no Diffuse/Glossy bit (so no NEE and no MIS) only for a smooth coat over a smooth conductor.
inline bool layerIsDelta(LayerTop t, LayerBottom b) { return layerTopSmooth(t) && layerBottomSpecular(b); }

inline float layerMaxComponent(float3 v) { return max(v.x, max(v.y, v.z)); }
inline bool layerIsZero(float3 v) { return v.x == 0.0 && v.y == 0.0 && v.z == 0.0; }
inline float layerPower2(float a, float b) { return (a * a) / max(a * a + b * b, 1.0e-30); }

// Beer-Lambert transmittance of dz of the layer along w (unit extinction), pbrt's Tr().
inline float layerTr(float dz, float3 w) {
    if (abs(dz) <= 1.17549435e-38) return 1.0;
    return exp(-abs(dz / w.z));
}

// pbrt's TrowbridgeReitzDistribution::Sample_wm: visible-normal sample, flipping w into the upper hemisphere first.
inline float3 layerSampleWm(float3 w, float alpha, thread uint& rng) {
    return sampleGGXVNDF(w.z < 0.0 ? -w : w, alpha, alpha, rng);
}

// pbrt's TrowbridgeReitzDistribution::PDF(w, wm) = D_visible.
inline float layerVisiblePdf(float3 w, float3 wm, float alpha) {
    return ggxG1(w, alpha, alpha) / max(abs(w.z), 1.0e-8) * ggxD(wm, alpha, alpha) * abs(dot(w, wm));
}

// pbrt's Refract(wi, n, eta, &etap, &wt): false on total internal reflection.
inline bool layerRefract(float3 wi, float3 n, float eta, thread float& etap, thread float3& wt) {
    float cosI = dot(n, wi);
    if (cosI < 0.0) { eta = 1.0 / eta; cosI = -cosI; n = -n; }
    float sin2I = max(0.0, 1.0 - cosI * cosI);
    float sin2T = sin2I / (eta * eta);
    if (sin2T >= 1.0) return false;
    float cosT = sqrt(max(0.0, 1.0 - sin2T));
    wt = -wi / eta + (cosI / eta - cosT) * n;
    etap = eta;
    return true;
}

// ---- the coat: pbrt's DielectricBxDF, both sides -------------------------------------------------------------------
inline float3 layerDielectricF(LayerTop top, float3 wo, float3 wi, bool radiance) {
    if (layerTopSmooth(top)) return float3(0.0);
    float cosO = wo.z, cosI = wi.z;
    bool isReflect = cosI * cosO > 0.0;
    float etap = 1.0;
    if (!isReflect) etap = cosO > 0.0 ? top.eta : 1.0 / top.eta;
    float3 wm = wi * etap + wo;
    if (cosI == 0.0 || cosO == 0.0 || dot(wm, wm) == 0.0) return float3(0.0);
    wm = normalize(wm);
    if (wm.z < 0.0) wm = -wm;
    if (dot(wm, wi) * cosI < 0.0 || dot(wm, wo) * cosO < 0.0) return float3(0.0);
    float F = frDielectric(dot(wo, wm), top.eta);
    float D = ggxD(wm, top.alpha, top.alpha);
    float G = ggxG(wo, wi, top.alpha, top.alpha);
    if (isReflect) return float3(D * G * F / abs(4.0 * cosI * cosO));
    float denom = (dot(wi, wm) + dot(wo, wm) / etap);
    denom = denom * denom * cosI * cosO;
    float ft = D * (1.0 - F) * G * abs(dot(wi, wm) * dot(wo, wm) / denom);
    if (radiance) ft /= etap * etap;
    return float3(ft);
}

inline float layerDielectricPdf(LayerTop top, float3 wo, float3 wi, int flags) {
    if (layerTopSmooth(top)) return 0.0;
    float cosO = wo.z, cosI = wi.z;
    bool isReflect = cosI * cosO > 0.0;
    float etap = 1.0;
    if (!isReflect) etap = cosO > 0.0 ? top.eta : 1.0 / top.eta;
    float3 wm = wi * etap + wo;
    if (cosI == 0.0 || cosO == 0.0 || dot(wm, wm) == 0.0) return 0.0;
    wm = normalize(wm);
    if (wm.z < 0.0) wm = -wm;
    if (dot(wm, wi) * cosI < 0.0 || dot(wm, wo) * cosO < 0.0) return 0.0;
    float R = frDielectric(dot(wo, wm), top.eta);
    float T = 1.0 - R;
    float pr = (flags & kLayerReflection) != 0 ? R : 0.0;
    float pt = (flags & kLayerTransmission) != 0 ? T : 0.0;
    if (pr == 0.0 && pt == 0.0) return 0.0;
    float pdfWm = layerVisiblePdf(wo, wm, top.alpha);
    if (isReflect) return pdfWm / (4.0 * abs(dot(wo, wm))) * pr / (pr + pt);
    float denom = dot(wi, wm) + dot(wo, wm) / etap;
    denom = denom * denom;
    float dwmDwi = abs(dot(wi, wm)) / max(denom, 1.0e-12);
    return pdfWm * dwmDwi * pt / (pr + pt);
}

inline LayerSample layerDielectricSample(LayerTop top, float3 wo, float uc, thread uint& rng, bool radiance, int flags) {
    LayerSample s = layerNoSample();
    if (layerTopSmooth(top)) {
        float R = frDielectric(wo.z, top.eta);
        float T = 1.0 - R;
        float pr = (flags & kLayerReflection) != 0 ? R : 0.0;
        float pt = (flags & kLayerTransmission) != 0 ? T : 0.0;
        if (pr == 0.0 && pt == 0.0) return s;
        if (uc < pr / (pr + pt)) {
            float3 wi = float3(-wo.x, -wo.y, wo.z);
            s.f = float3(R / abs(wi.z));
            s.wi = wi;
            s.pdf = pr / (pr + pt);
            s.valid = true; s.reflection = true; s.specular = true;
            return s;
        }
        float3 wi; float etap;
        if (!layerRefract(wo, float3(0.0, 0.0, 1.0), top.eta, etap, wi)) return s;
        float v = T / abs(wi.z);
        if (radiance) v /= etap * etap;
        s.f = float3(v);
        s.wi = wi;
        s.pdf = pt / (pr + pt);
        s.valid = true; s.transmission = true; s.specular = true;
        return s;
    }
    float3 wm = layerSampleWm(wo, top.alpha, rng);
    float R = frDielectric(dot(wo, wm), top.eta);
    float T = 1.0 - R;
    float pr = (flags & kLayerReflection) != 0 ? R : 0.0;
    float pt = (flags & kLayerTransmission) != 0 ? T : 0.0;
    if (pr == 0.0 && pt == 0.0) return s;
    if (uc < pr / (pr + pt)) {
        float d = dot(wo, wm);
        float3 wi = -wo + 2.0 * d * wm;
        if (!layerSameHemisphere(wo, wi)) return s;
        s.pdf = layerVisiblePdf(wo, wm, top.alpha) / (4.0 * abs(d)) * pr / (pr + pt);
        float v = ggxD(wm, top.alpha, top.alpha) * ggxG(wo, wi, top.alpha, top.alpha) * R / (4.0 * wi.z * wo.z);
        s.f = float3(v);
        s.wi = wi;
        s.valid = true; s.reflection = true;
        return s;
    }
    float etap; float3 wi;
    bool tir = !layerRefract(wo, wm, top.eta, etap, wi);
    if (tir || layerSameHemisphere(wo, wi) || wi.z == 0.0) return s;
    float denom = dot(wi, wm) + dot(wo, wm) / etap;
    denom = denom * denom;
    float dwmDwi = abs(dot(wi, wm)) / max(denom, 1.0e-12);
    s.pdf = layerVisiblePdf(wo, wm, top.alpha) * dwmDwi * pt / (pr + pt);
    float ft = T * ggxD(wm, top.alpha, top.alpha) * ggxG(wo, wi, top.alpha, top.alpha)
             * abs(dot(wi, wm) * dot(wo, wm) / (wi.z * wo.z * denom));
    if (radiance) ft /= etap * etap;
    s.f = float3(ft);
    s.wi = wi;
    s.valid = true; s.transmission = true;
    return s;
}

// ---- the base: pbrt's DiffuseBxDF / ConductorBxDF -------------------------------------------------------------------
inline float3 layerBottomF(LayerBottom b, float3 wo, float3 wi) {
    if (!layerSameHemisphere(wo, wi)) return float3(0.0);
    if (b.kind == 0) return b.albedo / M_PI_F;
    if (b.alpha < 1.0e-3) return float3(0.0);
    float cosO = abs(wo.z), cosI = abs(wi.z);
    if (cosI == 0.0 || cosO == 0.0) return float3(0.0);
    float3 wm = wi + wo;
    if (dot(wm, wm) == 0.0) return float3(0.0);
    wm = normalize(wm);
    float3 F = frComplexRGB(abs(dot(wo, wm)), b.eta, b.k);
    return F * (ggxD(wm, b.alpha, b.alpha) * ggxG(wo, wi, b.alpha, b.alpha) / (4.0 * cosI * cosO));
}

inline float layerBottomPdf(LayerBottom b, float3 wo, float3 wi, int flags) {
    if ((flags & kLayerReflection) == 0 || !layerSameHemisphere(wo, wi)) return 0.0;
    if (b.kind == 0) return abs(wi.z) / M_PI_F;
    if (b.alpha < 1.0e-3) return 0.0;
    float3 wm = wo + wi;
    if (dot(wm, wm) == 0.0) return 0.0;
    wm = normalize(wm);
    if (wm.z < 0.0) wm = -wm;
    return layerVisiblePdf(wo, wm, b.alpha) / (4.0 * abs(dot(wo, wm)));
}

inline LayerSample layerBottomSample(LayerBottom b, float3 wo, thread uint& rng, int flags) {
    LayerSample s = layerNoSample();
    if ((flags & kLayerReflection) == 0) return s;
    if (b.kind == 0) {
        float3 wi = cosineSampleHemisphere(float3(0.0, 0.0, 1.0), rng);
        if (wo.z < 0.0) wi.z = -wi.z;
        s.wi = wi;
        s.pdf = abs(wi.z) / M_PI_F;
        s.f = b.albedo / M_PI_F;
        s.valid = true; s.reflection = true;
        return s;
    }
    if (b.alpha < 1.0e-3) {
        float3 wi = float3(-wo.x, -wo.y, wo.z);
        float c = abs(wi.z);
        s.f = frComplexRGB(c, b.eta, b.k) / c;
        s.wi = wi;
        s.pdf = 1.0;
        s.valid = true; s.reflection = true; s.specular = true;
        return s;
    }
    if (wo.z == 0.0) return s;
    float3 wm = layerSampleWm(wo, b.alpha, rng);
    float d = dot(wo, wm);
    float3 wi = -wo + 2.0 * d * wm;
    if (!layerSameHemisphere(wo, wi)) return s;
    s.pdf = layerVisiblePdf(wo, wm, b.alpha) / (4.0 * abs(d));
    float cosO = abs(wo.z), cosI = abs(wi.z);
    if (cosI == 0.0 || cosO == 0.0) return s;
    float3 F = frComplexRGB(abs(d), b.eta, b.k);
    s.f = F * (ggxD(wm, b.alpha, b.alpha) * ggxG(wo, wi, b.alpha, b.alpha) / (4.0 * cosI * cosO));
    s.wi = wi;
    s.valid = true; s.reflection = true;
    return s;
}

// ---- LayeredBxDF::f(wo, wi) for wo.z > 0, wi.z > 0 ------------------------------------------------------------------
inline float3 layeredF(LayerTop top, LayerBottom bottom, float thickness, float3 wo, float3 wi, thread uint& rng) {
    float3 f = float3(0.0);
    if (wo.z <= 0.0 || wi.z <= 0.0) return f;
    const bool topSpecular = layerTopSmooth(top);
    const bool bottomSpecular = layerBottomSpecular(bottom);

    f = layerDielectricF(top, wo, wi, true);

    float uc = randFloat(rng);
    LayerSample wos = layerDielectricSample(top, wo, uc, rng, true, kLayerTransmission);
    if (!wos.valid || layerIsZero(wos.f) || wos.pdf == 0.0 || wos.wi.z == 0.0) return f;
    uc = randFloat(rng);
    LayerSample wis = layerDielectricSample(top, wi, uc, rng, false, kLayerTransmission);
    if (!wis.valid || layerIsZero(wis.f) || wis.pdf == 0.0 || wis.wi.z == 0.0) return f;

    float3 beta = wos.f * (abs(wos.wi.z) / wos.pdf);
    float z = thickness;
    float3 w = wos.wi;

    for (int depth = 0; depth < kLayerMaxDepth; ++depth) {
        if (depth > 3 && layerMaxComponent(beta) < 0.25) {
            float q = max(0.0, 1.0 - layerMaxComponent(beta));
            if (randFloat(rng) < q) break;
            beta /= 1.0 - q;
        }

        z = (z == thickness) ? 0.0 : thickness;
        beta *= layerTr(thickness, w);

        if (z == thickness) {
            // reflection at the exit (top) interface
            float uc2 = randFloat(rng);
            LayerSample bs = layerDielectricSample(top, -w, uc2, rng, true, kLayerReflection);
            if (!bs.valid || layerIsZero(bs.f) || bs.pdf == 0.0 || bs.wi.z == 0.0) break;
            beta *= bs.f * (abs(bs.wi.z) / bs.pdf);
            w = bs.wi;
        } else {
            // scattering at the non-exit (bottom) interface
            if (!bottomSpecular) {
                float wt = 1.0;
                if (!topSpecular) wt = layerPower2(wis.pdf, layerBottomPdf(bottom, -w, -wis.wi, kLayerAllLobes));
                f += beta * layerBottomF(bottom, -w, -wis.wi) * wis.f * (abs(wis.wi.z) * wt * layerTr(thickness, wis.wi) / wis.pdf);
            }
            float uc2 = randFloat(rng);
            LayerSample bs = layerBottomSample(bottom, -w, rng, kLayerReflection);
            if (!bs.valid || layerIsZero(bs.f) || bs.pdf == 0.0 || bs.wi.z == 0.0) break;
            beta *= bs.f * (abs(bs.wi.z) / bs.pdf);
            w = bs.wi;

            if (!topSpecular) {
                float3 fExit = layerDielectricF(top, -w, wi, true);
                if (!layerIsZero(fExit)) {
                    float wt = 1.0;
                    if (!bottomSpecular) {
                        // Verbatim pbrt: exitPDF is the density of generating wi from inside, not the competing strategy's,
                        // so the two MIS weights do not sum to one (see src/shared/bxdfs_layered.h).
                        float exitPdf = layerDielectricPdf(top, -w, wi, kLayerTransmission);
                        wt = layerPower2(bs.pdf, exitPdf);
                    }
                    f += beta * fExit * (layerTr(thickness, bs.wi) * wt);
                }
            }
            (void)uc2;
        }
    }
    return f;
}

// ---- LayeredBxDF::Sample_f(wo) ---------------------------------------------------------------------------------------
inline LayerSample layeredSample(LayerTop top, LayerBottom bottom, float thickness, float3 wo, thread uint& rng) {
    LayerSample none = layerNoSample();
    if (wo.z <= 0.0) return none;

    float uc = randFloat(rng);
    LayerSample bs = layerDielectricSample(top, wo, uc, rng, true, kLayerAllLobes);
    if (!bs.valid || layerIsZero(bs.f) || bs.pdf == 0.0 || bs.wi.z == 0.0) return none;
    if (bs.reflection) return bs;   // pbrt: pdfIsProportional

    float3 w = bs.wi;
    bool specularPath = bs.specular;
    float3 f = bs.f * abs(bs.wi.z);
    float pdf = bs.pdf;
    float z = thickness;

    for (int depth = 0; depth < kLayerMaxDepth; ++depth) {
        float rrBeta = layerMaxComponent(f) / pdf;
        if (depth > 3 && rrBeta < 0.25) {
            float q = max(0.0, 1.0 - rrBeta);
            if (randFloat(rng) < q) return none;
            pdf *= 1.0 - q;
        }
        if (w.z == 0.0) return none;

        z = (z == thickness) ? 0.0 : thickness;
        f *= layerTr(thickness, w);

        LayerSample is;
        float c0 = randFloat(rng);
        if (z == 0.0) is = layerBottomSample(bottom, -w, rng, kLayerAllLobes);
        else          is = layerDielectricSample(top, -w, c0, rng, true, kLayerAllLobes);
        if (!is.valid || layerIsZero(is.f) || is.pdf == 0.0 || is.wi.z == 0.0) return none;
        f *= is.f;
        pdf *= is.pdf;
        specularPath = specularPath && is.specular;
        w = is.wi;

        if (is.transmission) {
            LayerSample res = layerNoSample();
            res.valid = true;
            res.reflection = layerSameHemisphere(wo, w);
            res.transmission = !res.reflection;
            res.specular = specularPath;
            res.f = f;
            res.wi = w;
            res.pdf = pdf;
            return res;
        }
        f *= abs(is.wi.z);
    }
    return none;
}

// ---- LayeredBxDF::PDF(wo, wi) for wo.z > 0, wi.z > 0 -----------------------------------------------------------------
inline uint layerHashDir(float3 d, uint h) {
    h = (h ^ as_type<uint>(d.x)) * 747796405u + 2891336453u;
    h = (h ^ as_type<uint>(d.y)) * 747796405u + 2891336453u;
    h = (h ^ as_type<uint>(d.z)) * 747796405u + 2891336453u;
    h ^= h >> 15u;
    return h;
}

inline float layeredPdf(LayerTop top, LayerBottom bottom, float3 wo, float3 wi) {
    if (wo.z <= 0.0 || wi.z <= 0.0) return 0.0;
    uint rng = layerHashDir(wo, layerHashDir(wi, 0x9e3779b9u));
    const bool topSpecular = layerTopSmooth(top);
    const bool bottomSpecular = layerBottomSpecular(bottom);

    float pdfSum = layerDielectricPdf(top, wo, wi, kLayerReflection);

    float a = randFloat(rng);
    LayerSample wos = layerDielectricSample(top, wo, a, rng, true, kLayerTransmission);
    a = randFloat(rng);
    LayerSample wis = layerDielectricSample(top, wi, a, rng, false, kLayerTransmission);

    if (wos.valid && !layerIsZero(wos.f) && wos.pdf > 0.0 && wis.valid && !layerIsZero(wis.f) && wis.pdf > 0.0) {
        if (topSpecular) {
            pdfSum += layerBottomPdf(bottom, -wos.wi, -wis.wi, kLayerAllLobes);
        } else {
            LayerSample rs = layerBottomSample(bottom, -wos.wi, rng, kLayerAllLobes);
            if (rs.valid && !layerIsZero(rs.f) && rs.pdf > 0.0) {
                if (bottomSpecular) {
                    pdfSum += layerDielectricPdf(top, -rs.wi, wi, kLayerAllLobes);
                } else {
                    float rPdf = layerBottomPdf(bottom, -wos.wi, -wis.wi, kLayerAllLobes);
                    float wt = layerPower2(wis.pdf, rPdf);
                    pdfSum += wt * rPdf;

                    float tPdf = layerDielectricPdf(top, -rs.wi, wi, kLayerAllLobes);
                    wt = layerPower2(rs.pdf, tPdf);
                    pdfSum += wt * tPdf;
                }
            }
        }
    }
    // Lerp(0.9, 1/(4 pi), pdfSum)
    return 0.1 * (1.0 / (4.0 * M_PI_F)) + 0.9 * pdfSum;
}

// ---- constructors used by the material shaders ------------------------------------------------------------------------
// CoatedDiffuse: coat IOR + alpha over a Lambertian base.
inline void layeredCoatedDiffuseParts(float eta, float alpha, float3 albedo,
                                       thread LayerTop& top, thread LayerBottom& bottom) {
    top.eta = eta;
    top.alpha = alpha;
    bottom.kind = 0;
    bottom.albedo = albedo;
    bottom.eta = float3(1.0);
    bottom.k = float3(0.0);
    bottom.alpha = 0.0;
}

// CoatedConductor: as in pbrt's CoatedConductorMaterial::GetBxDF the conductor sits INSIDE the coat, so its complex IOR is
// relative to the coat - eta and k are both divided by the coat's IOR. `alpha` is the COAT's roughness; `baseAlpha` is the conductor's own
// (pbrt's conductor.roughness), or negative for "the coat's applies to both".
inline void layeredCoatedConductorParts(float eta, float alpha, float3 conductorEta, float3 conductorK,
                                         thread LayerTop& top, thread LayerBottom& bottom, float baseAlpha = -1.0) {
    top.eta = eta;
    top.alpha = alpha;
    bottom.kind = 1;
    bottom.albedo = float3(0.0);
    bottom.eta = conductorEta / eta;
    bottom.k = conductorK / eta;
    bottom.alpha = baseAlpha >= 0.0 ? baseAlpha : alpha;
}

constant float kLayerThickness = 0.01;
