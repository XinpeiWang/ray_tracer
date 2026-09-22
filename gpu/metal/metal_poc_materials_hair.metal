// metal_poc_materials_hair.metal
// HairBxDF (Marschner 2003 + Chiang 2016, mirrors pbrt-v4 HairBxDF) -
// direct MSL transcription of src/shared/bxdfs_hair.h's HairBxDF<T> and
// src/shared/bxdfs_principled.h's hair_* free helpers (both CPU_GPU
// templates already instantiated directly as real GPU device code by both
// OptiX backends - see optix_device_helpers.h/wavefront_device_helpers.h -
// no hand-duplicated GPU variant needed there, only here on Metal since
// MSL can't consume a C++ template the way CUDA's nvcc can).
//
// pMax is fixed at 3 (matching HairBxDF<T>::pMax) everywhere below, so
// every "[pMax+1]"-sized array in the C++ reference is a fixed float[4]
// here - MSL has no template parameter to carry that through.
//
// hairI0's own denominator term (i4 * ifact * ifact) reaches ~3.4e16 by
// the last loop iteration - the C++ reference computes this in int64_t,
// which MSL doesn't portably support for plain arithmetic on every Metal-
// capable device. Accumulating in `float` instead avoids that dependency
// entirely: by the time i4*ifact^2 is that large the corresponding series
// term is already many orders of magnitude smaller than the running sum,
// so float's ~7-digit precision on the denominator doesn't move the final
// result outside float32's own working precision - but see this file's
// own numeric cross-check in metal_poc_shader_tests.mm before trusting
// that argument alone; this comment records the REASONING, not proof.

inline float hairI0(float x) {
    float val = 0.0;
    float x2i = 1.0;
    float ifact = 1.0;
    float i4 = 1.0;
    for (int i = 0; i < 10; ++i) {
        if (i > 1) ifact *= float(i);
        val += x2i / (i4 * ifact * ifact);
        x2i *= x * x;
        i4 *= 4.0;
    }
    return val;
}

inline float hairLogI0(float x) {
    if (x > 12.0) {
        return x + 0.5 * (-log(2.0 * M_PI_F) + log(1.0 / x) + 1.0 / (8.0 * x));
    }
    return log(hairI0(x));
}

// Longitudinal scattering function Mp (Marschner 2003, Eq. 7)
inline float hairMp(float cosTheta_i, float cosTheta_o, float sinTheta_i, float sinTheta_o, float v) {
    float a = cosTheta_i * cosTheta_o / v;
    float b = sinTheta_i * sinTheta_o / v;
    float mp = (v <= 0.1)
        ? exp(hairLogI0(a) - b - 1.0 / v + 0.6931 + log(1.0 / (2.0 * v)))
        : (exp(-b) * hairI0(a)) / (sinh(1.0 / v) * 2.0 * v);
    return mp;
}

inline float hairLogistic(float x, float s) {
    x = fabs(x);
    return exp(-x / s) / (s * (1.0 + exp(-x / s)) * (1.0 + exp(-x / s)));
}

inline float hairLogisticCDF(float x, float s) {
    return 1.0 / (1.0 + exp(-x / s));
}

inline float hairTrimmedLogistic(float x, float s, float a, float b) {
    return hairLogistic(x, s) / (hairLogisticCDF(b, s) - hairLogisticCDF(a, s));
}

inline float hairSampleTrimmedLogistic(float u, float s, float a, float b) {
    float k = hairLogisticCDF(b, s) - hairLogisticCDF(a, s);
    float x = -s * log(1.0 / (u * k + hairLogisticCDF(a, s)) - 1.0);
    if (x < a) x = a;
    if (x > b) x = b;
    return x;
}

inline float hairPhi(int p, float gamma_o, float gamma_t) {
    return 2.0 * float(p) * gamma_t - 2.0 * gamma_o + float(p) * M_PI_F;
}

inline float hairNp(float phi, int p, float s, float gamma_o, float gamma_t) {
    float dphi = phi - hairPhi(p, gamma_o, gamma_t);
    while (dphi > M_PI_F) dphi -= 2.0 * M_PI_F;
    while (dphi < -M_PI_F) dphi += 2.0 * M_PI_F;
    return hairTrimmedLogistic(dphi, s, -M_PI_F, M_PI_F);
}

inline float hairSafeSqrt(float x) {
    return sqrt(x > 0.0 ? x : 0.0);
}

inline float hairSafeAsin(float x) {
    if (x < -1.0) x = -1.0;
    if (x > 1.0) x = 1.0;
    return asin(x);
}

// atan2(0,0) is required by IEEE754/C++'s std::atan2 (and the C++
// HairBxDF<double> reference this file is a transcription of) to return
// exactly 0 - but MSL's atan2, under this project's default
// MTLCompileOptions.fastMathEnabled=YES, returns NaN there instead
// (confirmed via metal_poc_shader_tests.mm's own atan2 diagnostic). This
// exact (0,0) case is not a rare edge case for HairBxDF's own phi_o/phi_i:
// it occurs whenever a direction's local y/z components are both exactly
// zero, i.e. the direction lies exactly along the local x (tangent) axis -
// which is common, not rare, at the visible CENTER of a sphere whenever
// that sphere's own surface normal stands in for the fiber tangent (B11's
// own convention - see shadeHair()'s comment), since many camera rays
// land at or very near that exact alignment there. Once phi_o is NaN it
// propagates through hairNp() into fr, and Metal's own min()/max()
// intrinsics prefer the non-NaN operand (OpenCL fmin/fmax semantics), so
// the resulting NaN-corrupted ratio silently resolves to exactly
// kMaxAttenuation in hairSample() - not a crash, not an obviously-wrong
// value, just a wrong one.
inline float hairAtan2Safe(float y, float x) {
    return (y == 0.0 && x == 0.0) ? 0.0 : atan2(y, x);
}

// Precomputed-per-material HairBxDF parameters (constructor work from
// HairBxDF<T>::HairBxDF() done once host-side / at material-setup time in
// a future integration step - for now shadeHair()/the test kernels below
// recompute v[]/s/sin2kAlpha/cos2kAlpha inline from the raw params, same
// as the C++ constructor, just called per-invocation instead of cached).
struct HairBxDFParams {
    float h;
    float eta;
    float sigma_r, sigma_g, sigma_b;
    float beta_m, beta_n;
    float v0, v1, v2, v3;         // longitudinal variance per lobe (pMax=3 -> 4 entries)
    float s;                      // azimuthal logistic scale
    float sin2kAlpha0, sin2kAlpha1, sin2kAlpha2;
    float cos2kAlpha0, cos2kAlpha1, cos2kAlpha2;
};

// Mirrors BxDFSampleResult<T>'s own fields (src/shared/bxdfs_base.h) -
// only the subset hairSample()/shadeHair() actually populate/use
// (is_transmission is always false for hair, so left out).
struct BxDFSampleResultGPU {
    float wo_x, wo_y, wo_z;
    float r, g, b;
    float eta;
    bool is_specular;
    bool valid;
};

inline HairBxDFParams hairMakeParams(float h, float eta, float sigma_r, float sigma_g, float sigma_b,
                                      float beta_m, float beta_n, float alpha_deg) {
    HairBxDFParams p;
    p.h = h; p.eta = eta;
    p.sigma_r = sigma_r; p.sigma_g = sigma_g; p.sigma_b = sigma_b;
    p.beta_m = beta_m; p.beta_n = beta_n;

    float bm = beta_m;
    float bm2 = bm * bm;
    float bm4 = bm2 * bm2;
    float bm8 = bm4 * bm4;
    float bm16 = bm8 * bm8;
    float bm20 = bm16 * bm4;
    float base = 0.726 * bm + 0.812 * bm2 + 3.7 * bm20;
    p.v0 = base * base;
    p.v1 = 0.25 * p.v0;
    p.v2 = 4.0 * p.v0;
    p.v3 = p.v2;

    const float SqrtPiOver8 = 0.626657069;
    float bn = beta_n;
    float bn2 = bn * bn;
    float bn4 = bn2 * bn2;
    float bn8 = bn4 * bn4;
    float bn16 = bn8 * bn8;
    float bn22 = bn16 * bn4 * bn2;
    p.s = SqrtPiOver8 * (0.265 * bn + 1.194 * bn2 + 5.372 * bn22);

    const float deg2rad = M_PI_F / 180.0;
    float sin0 = sin(alpha_deg * deg2rad);
    float cos0 = hairSafeSqrt(1.0 - sin0 * sin0);
    p.sin2kAlpha0 = sin0;
    p.cos2kAlpha0 = cos0;
    float sin1 = 2.0 * cos0 * sin0;
    float cos1 = cos0 * cos0 - sin0 * sin0;
    p.sin2kAlpha1 = sin1;
    p.cos2kAlpha1 = cos1;
    float sin2 = 2.0 * cos1 * sin1;
    float cos2 = cos1 * cos1 - sin1 * sin1;
    p.sin2kAlpha2 = sin2;
    p.cos2kAlpha2 = cos2;
    return p;
}

// Compute Ap attenuation for each lobe (pMax+1 = 4 RGB triplets).
inline void hairComputeAp(HairBxDFParams hp, float cosTheta_o,
                           thread float* ap_r, thread float* ap_g, thread float* ap_b) {
    float cosGamma_o = hairSafeSqrt(1.0 - hp.h * hp.h);
    float cosTheta = cosTheta_o * cosGamma_o;
    float f = frDielectric(cosTheta, hp.eta);

    float sinTheta_o = hairSafeSqrt(1.0 - cosTheta_o * cosTheta_o);
    float sinTheta_t = sinTheta_o / hp.eta;
    float cosTheta_t = hairSafeSqrt(1.0 - sinTheta_t * sinTheta_t);
    // Guarded the same way hairEvalLocal()/hairScatteringPdfLocal()/
    // hairSample() already guard this identical formula (max(cosTheta_o,
    // 1e-6), not a bare divide). The C++ reference's own compute_Ap()
    // leaves this ONE instance unguarded, relying on IEEE754's x/0.0 ==
    // +Infinity (sinGamma_t = h/Infinity == 0, cosGamma_t == 1 - a well-
    // defined limit) - reference CPU/CUDA compilation guarantees that,
    // but Metal shaders compile with MTLCompileOptions.fastMathEnabled
    // defaulting to YES, which explicitly does NOT guarantee IEEE754 NaN/
    // Infinity semantics (Apple's own MSL spec: fast math "assumes
    // operands are neither NaN nor infinity"). cosTheta_o == 0 exactly is
    // not a rare edge case here - it's the geometry at the visible CENTER
    // of a sphere whenever the sphere's own surface normal stands in for
    // HairBxDF's fiber tangent (B11's own convention), so this was
    // measurably wrong (see the numeric cross-check's own "grazing-center
    // diagnostic" in metal_poc_shader_tests.mm) across a large fraction of
    // every hair sphere's visible area, not just a corner case.
    float etap = hairSafeSqrt(hp.eta * hp.eta - sinTheta_o * sinTheta_o) / max(cosTheta_o, 1e-6);
    float sinGamma_t = hp.h / etap;
    float cosGamma_t = hairSafeSqrt(1.0 - sinGamma_t * sinGamma_t);

    float path = 2.0 * cosGamma_t / max(cosTheta_t, 1e-6);
    float Tr = exp(-hp.sigma_r * path);
    float Tg = exp(-hp.sigma_g * path);
    float Tb = exp(-hp.sigma_b * path);

    ap_r[0] = f; ap_g[0] = f; ap_b[0] = f;
    float f2 = (1.0 - f) * (1.0 - f);
    ap_r[1] = f2 * Tr; ap_g[1] = f2 * Tg; ap_b[1] = f2 * Tb;
    for (int p = 2; p < 3; ++p) {
        ap_r[p] = ap_r[p - 1] * Tr * f;
        ap_g[p] = ap_g[p - 1] * Tg * f;
        ap_b[p] = ap_b[p - 1] * Tb * f;
    }
    float denom_r = max(1.0 - Tr * f, 1e-6);
    float denom_g = max(1.0 - Tg * f, 1e-6);
    float denom_b = max(1.0 - Tb * f, 1e-6);
    ap_r[3] = ap_r[2] * f * Tr / denom_r;
    ap_g[3] = ap_g[2] * f * Tg / denom_g;
    ap_b[3] = ap_b[2] * f * Tb / denom_b;
}

inline void hairComputeApPdf(HairBxDFParams hp, float cosTheta_o, thread float* apPDF) {
    float ap_r[4], ap_g[4], ap_b[4];
    hairComputeAp(hp, cosTheta_o, ap_r, ap_g, ap_b);
    float sumY = 0.0;
    for (int i = 0; i <= 3; ++i) sumY += (ap_r[i] + ap_g[i] + ap_b[i]) / 3.0;
    sumY = max(sumY, 1e-8);
    for (int i = 0; i <= 3; ++i) apPDF[i] = (ap_r[i] + ap_g[i] + ap_b[i]) / (3.0 * sumY);
}

// Shared per-lobe scale-tilt (sinThetap_o, cosThetap_o) selection -
// identical branch structure to the C++ reference's own repeated inline
// if/else-if chain in eval_local()/scattering_pdf_local()/sample().
inline void hairScaleTilt(HairBxDFParams hp, int p, float sinTheta_o, float cosTheta_o,
                           thread float& sinThetap_o, thread float& cosThetap_o) {
    if (p == 0) {
        sinThetap_o = sinTheta_o * hp.cos2kAlpha1 - cosTheta_o * hp.sin2kAlpha1;
        cosThetap_o = cosTheta_o * hp.cos2kAlpha1 + sinTheta_o * hp.sin2kAlpha1;
    } else if (p == 1) {
        sinThetap_o = sinTheta_o * hp.cos2kAlpha0 + cosTheta_o * hp.sin2kAlpha0;
        cosThetap_o = cosTheta_o * hp.cos2kAlpha0 - sinTheta_o * hp.sin2kAlpha0;
    } else if (p == 2) {
        sinThetap_o = sinTheta_o * hp.cos2kAlpha2 + cosTheta_o * hp.sin2kAlpha2;
        cosThetap_o = cosTheta_o * hp.cos2kAlpha2 - sinTheta_o * hp.sin2kAlpha2;
    } else {
        sinThetap_o = sinTheta_o;
        cosThetap_o = cosTheta_o;
    }
    cosThetap_o = fabs(cosThetap_o);
}

// Evaluate BSDF: wo/wi are in hair local frame (fiber tangent = +X).
// Divides by AbsCosTheta(wi) = |wi_z| - matches pbrt-v4 HairBxDF::f() and
// the C++ reference's own eval_local() exactly.
inline void hairEvalLocal(HairBxDFParams hp,
                           float wo_x, float wo_y, float wo_z,
                           float wi_x, float wi_y, float wi_z,
                           thread float& fr, thread float& fg, thread float& fb) {
    float sinTheta_o = wo_x;
    float cosTheta_o = hairSafeSqrt(1.0 - sinTheta_o * sinTheta_o);
    float phi_o = hairAtan2Safe(wo_z, wo_y);
    float gamma_o = hairSafeAsin(hp.h);

    float sinTheta_i = wi_x;
    float cosTheta_i = hairSafeSqrt(1.0 - sinTheta_i * sinTheta_i);
    float phi_i = hairAtan2Safe(wi_z, wi_y);

    float etap = hairSafeSqrt(hp.eta * hp.eta - sinTheta_o * sinTheta_o) / max(cosTheta_o, 1e-6);
    float sinGamma_t = hp.h / etap;
    float gamma_t = hairSafeAsin(sinGamma_t);

    float ap_r[4], ap_g[4], ap_b[4];
    hairComputeAp(hp, cosTheta_o, ap_r, ap_g, ap_b);

    float phi = phi_i - phi_o;
    fr = 0.0; fg = 0.0; fb = 0.0;

    for (int p = 0; p < 3; ++p) {
        float sinThetap_o, cosThetap_o;
        hairScaleTilt(hp, p, sinTheta_o, cosTheta_o, sinThetap_o, cosThetap_o);
        float v = (p == 0) ? hp.v0 : (p == 1) ? hp.v1 : hp.v2;
        float mp = hairMp(cosTheta_i, cosThetap_o, sinTheta_i, sinThetap_o, v);
        float np = hairNp(phi, p, hp.s, gamma_o, gamma_t);
        fr += mp * ap_r[p] * np;
        fg += mp * ap_g[p] * np;
        fb += mp * ap_b[p] * np;
    }
    float mp_rem = hairMp(cosTheta_i, cosTheta_o, sinTheta_i, sinTheta_o, hp.v3);
    fr += mp_rem * ap_r[3] / (2.0 * M_PI_F);
    fg += mp_rem * ap_g[3] / (2.0 * M_PI_F);
    fb += mp_rem * ap_b[3] / (2.0 * M_PI_F);

    float absCosTheta_wi = fabs(wi_z);
    if (absCosTheta_wi < 1e-6) absCosTheta_wi = max(cosTheta_i, 1e-6);
    fr /= absCosTheta_wi;
    fg /= absCosTheta_wi;
    fb /= absCosTheta_wi;
}

// PDF in hair local frame (shared by sample() and scattering_pdf()).
inline float hairScatteringPdfLocal(HairBxDFParams hp,
                                     float wo_x, float wo_y, float wo_z,
                                     float wi_x, float wi_y, float wi_z) {
    float sinTheta_o = wo_x;
    float cosTheta_o = hairSafeSqrt(1.0 - sinTheta_o * sinTheta_o);
    float phi_o = hairAtan2Safe(wo_z, wo_y);
    float gamma_o = hairSafeAsin(hp.h);

    float sinTheta_i = wi_x;
    float cosTheta_i = hairSafeSqrt(1.0 - sinTheta_i * sinTheta_i);
    float phi_i = hairAtan2Safe(wi_z, wi_y);

    float etap = hairSafeSqrt(hp.eta * hp.eta - sinTheta_o * sinTheta_o) / max(cosTheta_o, 1e-6);
    float sinGamma_t = hp.h / etap;
    float gamma_t = hairSafeAsin(sinGamma_t);

    float apPDF[4];
    hairComputeApPdf(hp, cosTheta_o, apPDF);

    float phi = phi_i - phi_o;
    float pdf = 0.0;
    for (int p = 0; p < 3; ++p) {
        float sinThetap_o, cosThetap_o;
        hairScaleTilt(hp, p, sinTheta_o, cosTheta_o, sinThetap_o, cosThetap_o);
        float v = (p == 0) ? hp.v0 : (p == 1) ? hp.v1 : hp.v2;
        pdf += hairMp(cosTheta_i, cosThetap_o, sinTheta_i, sinThetap_o, v)
             * apPDF[p]
             * hairNp(phi, p, hp.s, gamma_o, gamma_t);
    }
    pdf += hairMp(cosTheta_i, cosTheta_o, sinTheta_i, sinTheta_o, hp.v3) * apPDF[3] * (1.0 / (2.0 * M_PI_F));
    return pdf;
}

// Build hair local frame from world-space fiber tangent (tangent = +X).
inline void hairBuildOnb(float tx, float ty, float tz,
                          thread float& bx, thread float& by, thread float& bz,
                          thread float& cx, thread float& cy, thread float& cz) {
    float ax = tx > 0.9 ? 0.0 : 1.0;
    float ay = tx > 0.9 ? 1.0 : 0.0;
    float az = 0.0;
    float qx = ay * tz - az * ty, qy = az * tx - ax * tz, qz = ax * ty - ay * tx;
    float qlen = max(sqrt(qx * qx + qy * qy + qz * qz), 1e-8);
    bx = qx / qlen; by = qy / qlen; bz = qz / qlen;
    cx = ty * bz - tz * by; cy = tz * bx - tx * bz; cz = tx * by - ty * bx;
}

inline void hairToLocal(float tx, float ty, float tz, float bx, float by, float bz, float cx, float cy, float cz,
                         float wx, float wy, float wz,
                         thread float& lx, thread float& ly, thread float& lz) {
    lx = wx * tx + wy * ty + wz * tz;
    ly = wx * bx + wy * by + wz * bz;
    lz = wx * cx + wy * cy + wz * cz;
}

inline void hairToWorld(float tx, float ty, float tz, float bx, float by, float bz, float cx, float cy, float cz,
                         float lx, float ly, float lz,
                         thread float& wx, thread float& wy, thread float& wz) {
    wx = lx * tx + ly * bx + lz * cx;
    wy = lx * ty + ly * by + lz * cy;
    wz = lx * tz + ly * bz + lz * cz;
}

// sample(): wi in world space, tx/ty/tz = fiber tangent, u1/u2/u3/u4 = random.
// Mirrors HairBxDF<T>::sample() exactly, including its own kMaxAttenuation=50
// safety clamp on the returned fr/pdf ratio (see that method's own comment
// in src/shared/bxdfs_hair.h for why the clamp exists and why it's an
// upper-only clamp, not a floor - a correctly-computed ratio can never go
// negative, per the numeric cross-check in metal_poc_shader_tests.mm; a
// floor would only mask a real bug rather than fix one).
inline BxDFSampleResultGPU hairSample(HairBxDFParams hp,
                                       float tx, float ty, float tz,
                                       float wix, float wiy, float wiz,
                                       float u1, float u2, float u3, float u4) {
    BxDFSampleResultGPU res;
    res.valid = false;

    float bx, by, bz, cx, cy, cz;
    hairBuildOnb(tx, ty, tz, bx, by, bz, cx, cy, cz);

    float wi_lx, wi_ly, wi_lz;
    hairToLocal(tx, ty, tz, bx, by, bz, cx, cy, cz, wix, wiy, wiz, wi_lx, wi_ly, wi_lz);
    float wo_lx = -wi_lx, wo_ly = -wi_ly, wo_lz = -wi_lz;

    float sinTheta_o = wo_lx;
    float cosTheta_o = hairSafeSqrt(1.0 - sinTheta_o * sinTheta_o);
    float phi_o = hairAtan2Safe(wo_lz, wo_ly);

    float apPDF[4];
    hairComputeApPdf(hp, cosTheta_o, apPDF);
    int p = 0;
    float cumul = 0.0;
    for (int i = 0; i <= 3; ++i) {
        cumul += apPDF[i];
        if (u1 < cumul || i == 3) { p = i; break; }
    }

    float sinThetap_o, cosThetap_o;
    hairScaleTilt(hp, p, sinTheta_o, cosTheta_o, sinThetap_o, cosThetap_o);
    float vp = (p == 0) ? hp.v0 : (p == 1) ? hp.v1 : (p == 2) ? hp.v2 : hp.v3;
    float cosTheta_i = 1.0 + vp * log(max(u2, 1e-5) + (1.0 - u2) * exp(-2.0 / vp));
    float sinTheta_i = hairSafeSqrt(1.0 - cosTheta_i * cosTheta_i);
    float cosPhi = cos(2.0 * M_PI_F * u3);
    sinTheta_i = -sinThetap_o * cosTheta_i + cosThetap_o * sinTheta_i * cosPhi;
    cosTheta_i = hairSafeSqrt(1.0 - sinTheta_i * sinTheta_i);

    float etap = hairSafeSqrt(hp.eta * hp.eta - sinTheta_o * sinTheta_o) / max(cosTheta_o, 1e-6);
    float sinGamma_t = hp.h / etap;
    float gamma_t = hairSafeAsin(sinGamma_t);
    float gamma_o = hairSafeAsin(hp.h);
    float dphi = (p < 3)
        ? hairSampleTrimmedLogistic(u4, hp.s, -M_PI_F, M_PI_F) + hairPhi(p, gamma_o, gamma_t)
        : 2.0 * M_PI_F * u4;
    float phi_i = phi_o + dphi;
    float wi_out_lx = sinTheta_i;
    float wi_out_ly = cosTheta_i * cos(phi_i);
    float wi_out_lz = cosTheta_i * sin(phi_i);

    float wo_wx, wo_wy, wo_wz;
    hairToWorld(tx, ty, tz, bx, by, bz, cx, cy, cz, wi_out_lx, wi_out_ly, wi_out_lz, wo_wx, wo_wy, wo_wz);

    float fr, fg, fb;
    hairEvalLocal(hp, wo_lx, wo_ly, wo_lz, wi_out_lx, wi_out_ly, wi_out_lz, fr, fg, fb);
    float pdf = hairScatteringPdfLocal(hp, wo_lx, wo_ly, wo_lz, wi_out_lx, wi_out_ly, wi_out_lz);

    if (pdf < 1e-8) return res;

    const float kMaxAttenuation = 50.0;
    float ratio_r = fr / pdf, ratio_g = fg / pdf, ratio_b = fb / pdf;
    res.wo_x = wo_wx; res.wo_y = wo_wy; res.wo_z = wo_wz;
    res.r = min(ratio_r, kMaxAttenuation);
    res.g = min(ratio_g, kMaxAttenuation);
    res.b = min(ratio_b, kMaxAttenuation);
    res.is_specular = false;
    res.eta = 1.0;
    res.valid = true;
    return res;
}

// materialType 31 (B11) - see this file's own header comment. Mirrors
// src/TheRestOfYourLife/hair_material.h's own hair_material::scatter()
// exactly: a fresh cross-section offset h sampled per scatter call, the
// fiber tangent defaulting to the hit's own facing (shading) normal (CPU's
// own tangent_is_dpdu=false default - B11 shades plain spheres, not real
// curve geometry, and relies on exactly this normal-as-tangent proxy), and
// the incident ray's own forward direction (NOT negated - matches
// hair_material.h's own `in_dir = unit_vector(r_in.direction())`) passed
// directly as HairBxDF::sample()'s own `wi`. No NEE: like shadeMirror()/
// shadePrincipled(), the returned attenuation is already fr/pdf (CPU's own
// srec.skip_pdf=true contract), so this is a pure specular-tier bounce.
//
// Field reuse in TriangleMaterial (no new buffer, matching materialType
// 28's own convention): color=sigma_a(rgb), ior=eta, roughness=beta_m,
// conductorEta.x=beta_n, conductorEta.y=alpha_deg. conductorEta.z/
// conductorK/transmitColor unused.
inline bool shadeHair(TriangleMaterial mat, float3 hitPoint, float3 facingNormal,
                       thread float3& rayDir, thread float3& rayOrigin,
                       thread float3& throughput, thread bool& specularBounce, thread uint& rngState) {
    float h = randFloat(rngState) * 2.0 - 1.0;
    float3 sigma_a = float3(mat.color);
    // Conditional path regularization (pbrt-v4's own documented mechanism
    // for numerically fragile BSDFs - hair_material.h's own do_regularize,
    // there defaulting off since CPU's double precision usually doesn't
    // need it) - here made UNCONDITIONAL-when-triggered rather than an
    // integrator-level opt-in, because the trigger is float32 precision
    // itself, not path history. For high absorption (sigma_a's max
    // channel > 0.3) combined with a narrow longitudinal lobe (small
    // beta_m/beta_n), hairComputeApPdf's importance sampling concentrates
    // almost all weight onto the p=0 (R, pure surface reflection) lobe -
    // the ONLY lobe still carrying meaningful energy once TT/TRT are
    // absorbed away - so hairMp's own LogI0(a)-1/v near-cancellation
    // (see hairI0's own header comment) for that ONE dominant lobe is no
    // longer "averaged out" by three more forgiving lobes the way it is
    // for every other material in this scene. float32 (Metal's only
    // compute precision - no native double) isn't stable enough there;
    // double precision (this project's CPU/OptiX reference) is. Verified
    // via metal_poc_shader_tests.mm's own grazing-center/black-fur
    // diagnostics AND a direct --gpu render comparison: below this
    // sigma_a threshold (every OTHER material in B11), beta_m/beta_n are
    // left completely untouched, preserving each sphere's own intended
    // sharpness/variety - see build_hair_fibers()'s own comment ("Each
    // sphere uses slightly different hair parameters... to show
    // variety"). 0.6 (not the CPU-side do_regularize's own 0.3) because
    // 0.3 was verified insufficient to stabilize this specific float32
    // failure mode on real hardware, not picked a priori.
    float sigmaMax = max(sigma_a.x, max(sigma_a.y, sigma_a.z));
    float betaFloor = sigmaMax > 0.3 ? 0.6 : 0.0;
    float beta_m = max(mat.roughness, betaFloor);
    float beta_n = max(mat.conductorEta.x, betaFloor);
    float alpha_deg = mat.conductorEta.y;
    HairBxDFParams hp = hairMakeParams(h, mat.ior, sigma_a.x, sigma_a.y, sigma_a.z,
                                        beta_m, beta_n, alpha_deg);

    float u1 = randFloat(rngState);
    float u2 = randFloat(rngState);
    float u3 = randFloat(rngState);
    float u4 = randFloat(rngState);

    BxDFSampleResultGPU res = hairSample(hp, facingNormal.x, facingNormal.y, facingNormal.z,
                                          rayDir.x, rayDir.y, rayDir.z,
                                          u1, u2, u3, u4);
    if (!res.valid) return false;

    rayDir = float3(res.wo_x, res.wo_y, res.wo_z);
    rayOrigin = hitPoint + facingNormal * 0.001f;
    throughput *= float3(res.r, res.g, res.b);
    specularBounce = true;
    return true;
}
