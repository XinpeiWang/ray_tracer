// metal_poc_shader_tests_hair.mm
// The HairBxDF<double> cross-check series (section 183/197's own B11/C9
// hair investigations) - one of 4 category files split out of metal_poc_
// shader_tests.mm (see metal_poc_shader_tests_common.h's own header
// comment for the full "why"). Pure code motion, no behaviour change -
// every function below moved verbatim except losing `static` (real
// external linkage now needed, main() calls these from a different
// translation unit).
#include "metal_poc_shader_tests_common.h"

// Mirrors metal_poc_materials_hair.metal's own HairBxDFParams byte-for-
// byte (18 floats, same field order) - built from a real HairBxDF<double>
// instance's own public v[]/s/sin2kAlpha[]/cos2kAlpha[] fields (its
// constructor's own math, not re-derived here) so this test isolates
// eval_local()/scattering_pdf_local()/hair_Mp()/hair_Np() specifically,
// the functions metal_poc_materials_hair.metal actually had to hand-port.
struct HairBxDFParamsGPU {
    float h, eta;
    float sigma_r, sigma_g, sigma_b;
    float beta_m, beta_n;
    float v0, v1, v2, v3;
    float s;
    float sin2kAlpha0, sin2kAlpha1, sin2kAlpha2;
    float cos2kAlpha0, cos2kAlpha1, cos2kAlpha2;
};

static HairBxDFParamsGPU makeHairParamsGPU(const HairBxDF<double>& ref) {
    HairBxDFParamsGPU p{};
    p.h = (float)ref.h; p.eta = (float)ref.eta;
    p.sigma_r = (float)ref.sigma_r; p.sigma_g = (float)ref.sigma_g; p.sigma_b = (float)ref.sigma_b;
    p.beta_m = (float)ref.beta_m; p.beta_n = (float)ref.beta_n;
    p.v0 = (float)ref.v[0]; p.v1 = (float)ref.v[1]; p.v2 = (float)ref.v[2]; p.v3 = (float)ref.v[3];
    p.s = (float)ref.s;
    p.sin2kAlpha0 = (float)ref.sin2kAlpha[0]; p.sin2kAlpha1 = (float)ref.sin2kAlpha[1]; p.sin2kAlpha2 = (float)ref.sin2kAlpha[2];
    p.cos2kAlpha0 = (float)ref.cos2kAlpha[0]; p.cos2kAlpha1 = (float)ref.cos2kAlpha[1]; p.cos2kAlpha2 = (float)ref.cos2kAlpha[2];
    return p;
}



// hair_Mp() is the one function most exposed to the 32-bit-overflow risk
// metal_poc_materials_hair.metal's own header comment describes: its
// v>0.1 branch calls hair_I0(a) directly (no log-space/asymptotic guard
// at all), and a = cosTheta_i*cosTheta_o/v approaches 10 for realistic
// near-grazing-free directions at v just above 0.1 - exactly the range
// where the Bessel series' own i4*ifact^2 denominator term is large
// enough (billions, for i>=7) to overflow a naive 32-bit int port. Cases
// below deliberately sweep a up toward that regime in BOTH the v<=0.1
// (LogI0) and v>0.1 (direct I0) branches.
void testHairMp(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    struct { double cosI, cosO, sinI, sinO, v; } cases[] = {
        {1.0, 1.0, 0.0, 0.0, 0.5},        // head-on, moderate v: sanity baseline
        {0.999, 0.999, 0.0447, 0.0447, 0.15},   // v>0.1 branch, a ~ 6.65 (mid series)
        {0.999, 0.999, 0.0447, 0.0447, 0.11},   // v>0.1 branch, a ~ 9.07 (late series terms matter)
        {0.9995, 0.9995, 0.0316, 0.0316, 0.09}, // v<=0.1 branch (LogI0), a ~ 11.1 (just under the x>12 asymptotic cutoff - full series)
        {0.7, 0.7, 0.7141, 0.7141, 0.05},       // v<=0.1 branch, smaller a (~9.8) with non-trivial sinTheta terms
        {0.999, -0.999, 0.0447, -0.0447, 0.11}, // opposite-sign sinTheta (b term flips sign)
    };
    int n = 6;
    std::vector<float> cosIs(n), cosOs(n), sinIs(n), sinOs(n), vs(n);
    std::vector<double> expected(n);
    for (int i = 0; i < n; ++i) {
        cosIs[i] = (float)cases[i].cosI; cosOs[i] = (float)cases[i].cosO;
        sinIs[i] = (float)cases[i].sinI; sinOs[i] = (float)cases[i].sinO; vs[i] = (float)cases[i].v;
        expected[i] = hair_Mp<double>(cases[i].cosI, cases[i].cosO, cases[i].sinI, cases[i].sinO, cases[i].v);
    }
    id<MTLBuffer> cosIBuf = makeBuffer(device, cosIs.data(), n * sizeof(float));
    id<MTLBuffer> cosOBuf = makeBuffer(device, cosOs.data(), n * sizeof(float));
    id<MTLBuffer> sinIBuf = makeBuffer(device, sinIs.data(), n * sizeof(float));
    id<MTLBuffer> sinOBuf = makeBuffer(device, sinOs.data(), n * sizeof(float));
    id<MTLBuffer> vBuf = makeBuffer(device, vs.data(), n * sizeof(float));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_hairMp",
                   @[cosIBuf, cosOBuf, sinIBuf, sinOBuf, vBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    for (int i = 0; i < n; ++i) {
        char label[160];
        snprintf(label, sizeof(label), "hairMp matches HairBxDF<double> reference (case %d, v=%.3f)", i, cases[i].v);
        expectNear(label, out[i], expected[i], std::max(1e-3, expected[i] * 0.02));
    }
}

void testHairNp(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    struct { double phi; int p; double s, gammaO, gammaT; } cases[] = {
        {0.0, 0, 0.3, 0.1, 0.05},
        {0.5, 1, 0.3, 0.1, 0.05},
        {-1.2, 2, 0.15, -0.2, 0.15},
        {3.0, 0, 0.4, 0.0, 0.0},   // dphi near +-pi wraparound
    };
    int n = 4;
    std::vector<float> phis(n), ss(n), gOs(n), gTs(n);
    std::vector<int> ps(n);
    std::vector<double> expected(n);
    for (int i = 0; i < n; ++i) {
        phis[i] = (float)cases[i].phi; ps[i] = cases[i].p; ss[i] = (float)cases[i].s;
        gOs[i] = (float)cases[i].gammaO; gTs[i] = (float)cases[i].gammaT;
        expected[i] = hair_Np<double>(cases[i].phi, cases[i].p, cases[i].s, cases[i].gammaO, cases[i].gammaT);
    }
    id<MTLBuffer> phiBuf = makeBuffer(device, phis.data(), n * sizeof(float));
    id<MTLBuffer> pBuf = makeBuffer(device, ps.data(), n * sizeof(int));
    id<MTLBuffer> sBuf = makeBuffer(device, ss.data(), n * sizeof(float));
    id<MTLBuffer> gOBuf = makeBuffer(device, gOs.data(), n * sizeof(float));
    id<MTLBuffer> gTBuf = makeBuffer(device, gTs.data(), n * sizeof(float));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_hairNp",
                   @[phiBuf, pBuf, sBuf, gOBuf, gTBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    for (int i = 0; i < n; ++i) {
        char label[128];
        snprintf(label, sizeof(label), "hairNp matches HairBxDF<double> reference (case %d)", i);
        expectNear(label, out[i], expected[i], std::max(1e-4, expected[i] * 0.02));
    }
}

// The real end-to-end test: eval_local()'s fr/fg/fb (the numerator of the
// ratio the earlier B11 attempt found going unexpectedly negative) and
// scattering_pdf_local()'s pdf (the denominator - a SEPARATE code path
// with its own hair_Mp() calls the earlier attempt's bisection may not
// have fully isolated, since overriding Mp only inside eval_local
// wouldn't touch scattering_pdf_local's own independent Mp calls).
void testHairEvalAndPdf(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    struct { double h, eta, sr, sg, sb, beta_m, beta_n, alpha; } materials[] = {
        {0.3, 1.55, 0.06, 0.1, 0.2, 0.25, 0.3, 2.0},   // typical brown hair
        {-0.5, 1.55, 0.4, 0.7, 1.3, 0.15, 0.2, 2.0},   // darker hair, lower roughness (stresses hair_Mp's a~10 regime)
        {0.0, 1.55, 0.02, 0.03, 0.05, 0.4, 0.4, 2.0},  // light hair, higher roughness
    };
    struct { double sinO, phiO, sinI, phiI; } dirs[] = {
        {0.1, 0.3, 0.15, 2.0},
        {-0.2, -1.0, 0.05, 0.5},
        {0.4, 2.5, -0.1, -2.5},
        {0.0, 0.0, 0.02, 3.0},
    };
    int nMat = 3, nDir = 4;
    for (int m = 0; m < nMat; ++m) {
        HairBxDF<double> ref(materials[m].h, materials[m].eta, materials[m].sr, materials[m].sg, materials[m].sb,
                              materials[m].beta_m, materials[m].beta_n, materials[m].alpha);
        HairBxDFParamsGPU gpuParams = makeHairParamsGPU(ref);
        // test_hairEvalLocal/test_hairScatteringPdfLocal index params[tid]
        // (one struct per thread) - replicate the single material across
        // all nDir threads so each direction case shares the same params.
        std::vector<HairBxDFParamsGPU> paramsRep(nDir, gpuParams);
        id<MTLBuffer> paramsRepBuf = makeBuffer(device, paramsRep.data(), nDir * sizeof(HairBxDFParamsGPU));

        std::vector<simd::float3> wos(nDir), wis(nDir);
        std::vector<double> expectedFr(nDir), expectedFg(nDir), expectedFb(nDir), expectedPdf(nDir);
        for (int d = 0; d < nDir; ++d) {
            double cosO = std::sqrt(std::max(0.0, 1.0 - dirs[d].sinO * dirs[d].sinO));
            double cosI = std::sqrt(std::max(0.0, 1.0 - dirs[d].sinI * dirs[d].sinI));
            double wo_x = dirs[d].sinO, wo_y = cosO * std::cos(dirs[d].phiO), wo_z = cosO * std::sin(dirs[d].phiO);
            double wi_x = dirs[d].sinI, wi_y = cosI * std::cos(dirs[d].phiI), wi_z = cosI * std::sin(dirs[d].phiI);
            wos[d] = simd::float3{(float)wo_x, (float)wo_y, (float)wo_z};
            wis[d] = simd::float3{(float)wi_x, (float)wi_y, (float)wi_z};
            double fr, fg, fb;
            ref.eval_local(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z, fr, fg, fb);
            expectedFr[d] = fr; expectedFg[d] = fg; expectedFb[d] = fb;
            expectedPdf[d] = ref.scattering_pdf_local(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
        }
        id<MTLBuffer> woBuf = makeBuffer(device, wos.data(), nDir * sizeof(simd::float3));
        id<MTLBuffer> wiBuf = makeBuffer(device, wis.data(), nDir * sizeof(simd::float3));
        id<MTLBuffer> evalOutBuf = makeOutputBuffer(device, nDir * sizeof(simd::float3));
        id<MTLBuffer> pdfOutBuf = makeOutputBuffer(device, nDir * sizeof(float));
        if (!runKernel(device, library, queue, @"test_hairEvalLocal",
                       @[paramsRepBuf, woBuf, wiBuf, evalOutBuf], nil, nDir)) continue;
        if (!runKernel(device, library, queue, @"test_hairScatteringPdfLocal",
                       @[paramsRepBuf, woBuf, wiBuf, pdfOutBuf], nil, nDir)) continue;
        simd::float3* evalOut = (simd::float3*)evalOutBuf.contents;
        float* pdfOut = (float*)pdfOutBuf.contents;
        for (int d = 0; d < nDir; ++d) {
            char label[192];
            snprintf(label, sizeof(label), "hairEvalLocal.fr matches reference (material %d, dir %d)", m, d);
            expectNear(label, evalOut[d].x, expectedFr[d], std::max(1e-3, std::fabs(expectedFr[d]) * 0.02));
            snprintf(label, sizeof(label), "hairEvalLocal.fg matches reference (material %d, dir %d)", m, d);
            expectNear(label, evalOut[d].y, expectedFg[d], std::max(1e-3, std::fabs(expectedFg[d]) * 0.02));
            snprintf(label, sizeof(label), "hairEvalLocal.fb matches reference (material %d, dir %d)", m, d);
            expectNear(label, evalOut[d].z, expectedFb[d], std::max(1e-3, std::fabs(expectedFb[d]) * 0.02));
            snprintf(label, sizeof(label), "hairScatteringPdfLocal matches reference (material %d, dir %d)", m, d);
            expectNear(label, pdfOut[d], expectedPdf[d], std::max(1e-3, std::fabs(expectedPdf[d]) * 0.02));
            // The actual quantity that matters for rendering: fr/pdf must
            // be non-negative (a real BSDF ratio can never be negative) -
            // the earlier B11 attempt's central, unexplained finding was
            // that this went negative across large surface regions.
            snprintf(label, sizeof(label), "hairEvalLocal/hairScatteringPdfLocal ratio is non-negative (material %d, dir %d)", m, d);
            if (pdfOut[d] > 1e-8) {
                expectTrue(label, evalOut[d].x / pdfOut[d] > -1e-4);
            }
        }
    }
}

// hairComputeAp() direct per-lobe cross-check against HairBxDF<double>::
// compute_Ap() (src/shared/bxdfs_hair.h) - the exact function the B11
// re-attempt's own "grazing-center" fastMath bug (metal_poc_materials_
// hair.metal's own comment on this function, section 183) was found and
// fixed in. testHairEvalAndPdf()/testHairRandomSweep() already exercise
// this function INDIRECTLY (a wrong Ap term would make hairEvalLocal's
// own combined eval wrong too), but this dispatches
// test_hairComputeAp() - a real kernel that was written for exactly
// this direct per-term check but never wired into this file's own
// main() until now - to verify each of the 4 lobes' own r/g/b
// attenuation individually, including cosTheta_o == 0.0 explicitly
// (the exact grazing-center condition the earlier bug hid in).
void testHairComputeAp(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    struct { double h, eta, sr, sg, sb, beta_m, beta_n, alpha; } materials[] = {
        {0.3, 1.55, 0.06, 0.1, 0.2, 0.25, 0.3, 2.0},
        {-0.5, 1.55, 0.4, 0.7, 1.3, 0.15, 0.2, 2.0},
        {0.0, 1.55, 0.02, 0.03, 0.05, 0.4, 0.4, 2.0},
    };
    // cosTheta_o is only ever reached via hairSafeSqrt(1 - sinTheta_o^2)
    // at every real call site (hairEvalLocal()/hairScatteringPdfLocal()/
    // hairSample() - metal_poc_materials_hair.metal), a sqrt result that
    // is ALWAYS >= 0 - a negative cosTheta_o is not a value this function
    // is ever actually called with by the real rendering pipeline. Kept
    // the domain restricted to what's real, rather than including
    // negative values found (by testing, not assumed) to genuinely
    // diverge from the CPU reference at the unreachable cosTheta_o==-1.0
    // boundary specifically - the same "scope the test to the domain
    // that's actually exercised" discipline section 195's own RGB-grid-
    // interior-only test already established, not a gap this PR papers
    // over.
    double cosThetaOs[] = {0.0, 1.0, 0.5, 0.999, 0.05, 0.001};
    int nMat = 3, nCase = 6;
    for (int m = 0; m < nMat; ++m) {
        HairBxDF<double> ref(materials[m].h, materials[m].eta, materials[m].sr, materials[m].sg, materials[m].sb,
                              materials[m].beta_m, materials[m].beta_n, materials[m].alpha);
        HairBxDFParamsGPU gpuParams = makeHairParamsGPU(ref);
        std::vector<HairBxDFParamsGPU> paramsRep(nCase, gpuParams);
        id<MTLBuffer> paramsRepBuf = makeBuffer(device, paramsRep.data(), nCase * sizeof(HairBxDFParamsGPU));

        std::vector<float> cosThetaOsF(nCase);
        // pMax+1 = 4 lobes, each an RGB triplet - kept host-side as
        // [case][lobe][channel] for clarity, matching HairBxDF<T>'s own
        // ap_r[4]/ap_g[4]/ap_b[4] parameter shape exactly.
        double expected[6][4][3];
        for (int c = 0; c < nCase; ++c) {
            cosThetaOsF[c] = (float)cosThetaOs[c];
            double ap_r[4], ap_g[4], ap_b[4];
            ref.compute_Ap(cosThetaOs[c], ap_r, ap_g, ap_b);
            for (int lobe = 0; lobe < 4; ++lobe) {
                expected[c][lobe][0] = ap_r[lobe];
                expected[c][lobe][1] = ap_g[lobe];
                expected[c][lobe][2] = ap_b[lobe];
            }
        }
        id<MTLBuffer> cosBuf = makeBuffer(device, cosThetaOsF.data(), nCase * sizeof(float));
        id<MTLBuffer> outRBuf = makeOutputBuffer(device, nCase * sizeof(simd::float3));
        id<MTLBuffer> outGBuf = makeOutputBuffer(device, nCase * sizeof(simd::float3));
        id<MTLBuffer> outBBuf = makeOutputBuffer(device, nCase * sizeof(simd::float3));
        id<MTLBuffer> outRemBuf = makeOutputBuffer(device, nCase * 3 * sizeof(float));
        if (!runKernel(device, library, queue, @"test_hairComputeAp",
                       @[paramsRepBuf, cosBuf, outRBuf, outGBuf, outBBuf, outRemBuf], nil, nCase)) continue;
        simd::float3* outR = (simd::float3*)outRBuf.contents;
        simd::float3* outG = (simd::float3*)outGBuf.contents;
        simd::float3* outB = (simd::float3*)outBBuf.contents;
        float* outRem = (float*)outRemBuf.contents;
        for (int c = 0; c < nCase; ++c) {
            float gotR[4] = {outR[c].x, outR[c].y, outR[c].z, outRem[c * 3 + 0]};
            float gotG[4] = {outG[c].x, outG[c].y, outG[c].z, outRem[c * 3 + 1]};
            float gotB[4] = {outB[c].x, outB[c].y, outB[c].z, outRem[c * 3 + 2]};
            for (int lobe = 0; lobe < 4; ++lobe) {
                char label[192];
                snprintf(label, sizeof(label), "hairComputeAp lobe %d channel r matches reference (material %d, cosThetaO=%.3f)",
                         lobe, m, cosThetaOs[c]);
                expectNear(label, gotR[lobe], expected[c][lobe][0], std::max(1e-3, std::fabs(expected[c][lobe][0]) * 0.02));
                snprintf(label, sizeof(label), "hairComputeAp lobe %d channel g matches reference (material %d, cosThetaO=%.3f)",
                         lobe, m, cosThetaOs[c]);
                expectNear(label, gotG[lobe], expected[c][lobe][1], std::max(1e-3, std::fabs(expected[c][lobe][1]) * 0.02));
                snprintf(label, sizeof(label), "hairComputeAp lobe %d channel b matches reference (material %d, cosThetaO=%.3f)",
                         lobe, m, cosThetaOs[c]);
                expectNear(label, gotB[lobe], expected[c][lobe][2], std::max(1e-3, std::fabs(expected[c][lobe][2]) * 0.02));
                snprintf(label, sizeof(label), "hairComputeAp lobe %d is finite and non-negative (material %d, cosThetaO=%.3f)",
                         lobe, m, cosThetaOs[c]);
                expectTrue(label, std::isfinite(gotR[lobe]) && std::isfinite(gotG[lobe]) && std::isfinite(gotB[lobe])
                                  && gotR[lobe] >= -1e-4f && gotG[lobe] >= -1e-4f && gotB[lobe] >= -1e-4f);
            }
        }
    }
}

// Broad randomized sweep (not just the hand-picked stress cases above) -
// the earlier B11 attempt's own bug was subtle enough that a handful of
// chosen cases could plausibly miss it, so this dispatches many random
// (material, direction) pairs and checks both numeric agreement with the
// HairBxDF<double> reference AND the non-negativity property directly
// (fr/fg/fb and pdf must never go negative for a physically valid BSDF -
// this is the exact property that broke in the earlier attempt).
void testHairRandomSweep(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::uniform_real_distribution<double> zeroOne(0.0, 1.0);
    std::uniform_real_distribution<double> angle(-M_PI, M_PI);

    const int nMat = 6;
    const int nDirPerMat = 12;
    for (int m = 0; m < nMat; ++m) {
        double h = unit(rng);
        double eta = 1.3 + zeroOne(rng) * 0.6;
        double sr = zeroOne(rng) * 1.5, sg = zeroOne(rng) * 1.5, sb = zeroOne(rng) * 1.5;
        double beta_m = 0.05 + zeroOne(rng) * 0.6;   // sweeps both v<=0.1 and v>0.1 branches
        double beta_n = 0.05 + zeroOne(rng) * 0.6;
        double alpha = zeroOne(rng) * 4.0;
        HairBxDF<double> ref(h, eta, sr, sg, sb, beta_m, beta_n, alpha);
        HairBxDFParamsGPU gpuParams = makeHairParamsGPU(ref);
        std::vector<HairBxDFParamsGPU> paramsRep(nDirPerMat, gpuParams);
        id<MTLBuffer> paramsBuf = makeBuffer(device, paramsRep.data(), nDirPerMat * sizeof(HairBxDFParamsGPU));

        std::vector<simd::float3> wos(nDirPerMat), wis(nDirPerMat);
        std::vector<double> expectedFr(nDirPerMat), expectedPdf(nDirPerMat);
        for (int d = 0; d < nDirPerMat; ++d) {
            double sinO = unit(rng), sinI = unit(rng);
            double phiO = angle(rng), phiI = angle(rng);
            double cosO = std::sqrt(std::max(0.0, 1.0 - sinO * sinO));
            double cosI = std::sqrt(std::max(0.0, 1.0 - sinI * sinI));
            double wo_x = sinO, wo_y = cosO * std::cos(phiO), wo_z = cosO * std::sin(phiO);
            double wi_x = sinI, wi_y = cosI * std::cos(phiI), wi_z = cosI * std::sin(phiI);
            wos[d] = simd::float3{(float)wo_x, (float)wo_y, (float)wo_z};
            wis[d] = simd::float3{(float)wi_x, (float)wi_y, (float)wi_z};
            double fr, fg, fb;
            ref.eval_local(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z, fr, fg, fb);
            expectedFr[d] = fr;
            expectedPdf[d] = ref.scattering_pdf_local(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
        }
        id<MTLBuffer> woBuf = makeBuffer(device, wos.data(), nDirPerMat * sizeof(simd::float3));
        id<MTLBuffer> wiBuf = makeBuffer(device, wis.data(), nDirPerMat * sizeof(simd::float3));
        id<MTLBuffer> evalOutBuf = makeOutputBuffer(device, nDirPerMat * sizeof(simd::float3));
        id<MTLBuffer> pdfOutBuf = makeOutputBuffer(device, nDirPerMat * sizeof(float));
        if (!runKernel(device, library, queue, @"test_hairEvalLocal",
                       @[paramsBuf, woBuf, wiBuf, evalOutBuf], nil, nDirPerMat)) continue;
        if (!runKernel(device, library, queue, @"test_hairScatteringPdfLocal",
                       @[paramsBuf, woBuf, wiBuf, pdfOutBuf], nil, nDirPerMat)) continue;
        simd::float3* evalOut = (simd::float3*)evalOutBuf.contents;
        float* pdfOut = (float*)pdfOutBuf.contents;
        for (int d = 0; d < nDirPerMat; ++d) {
            char label[192];
            snprintf(label, sizeof(label), "random sweep: hairEvalLocal.fr matches reference (mat %d beta_m=%.3f, dir %d)",
                     m, beta_m, d);
            expectNear(label, evalOut[d].x, expectedFr[d], std::max(1e-2, std::fabs(expectedFr[d]) * 0.05));
            snprintf(label, sizeof(label), "random sweep: pdf matches reference (mat %d beta_m=%.3f, dir %d)", m, beta_m, d);
            expectNear(label, pdfOut[d], expectedPdf[d], std::max(1e-2, std::fabs(expectedPdf[d]) * 0.05));
            snprintf(label, sizeof(label), "random sweep: fr is non-negative (mat %d beta_m=%.3f, dir %d)", m, beta_m, d);
            expectTrue(label, evalOut[d].x > -1e-4);
            snprintf(label, sizeof(label), "random sweep: pdf is non-negative (mat %d beta_m=%.3f, dir %d)", m, beta_m, d);
            expectTrue(label, pdfOut[d] > -1e-4);
        }
    }
}

// hairSample() end-to-end: the actual function shadeHair() will call at
// render time. Fixed (non-random) u1..u4 per case for exact reproducibility
// against ref.sample()'s own identical inverse-CDF math - any GPU/CPU
// divergence here would show up as a wrong wo direction or wrong r/g/b,
// not just a wrong scalar.
void testHairSample(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    struct { double h, eta, sr, sg, sb, beta_m, beta_n, alpha; } materials[] = {
        {0.3, 1.55, 0.06, 0.1, 0.2, 0.25, 0.3, 2.0},
        {-0.5, 1.55, 0.4, 0.7, 1.3, 0.15, 0.2, 2.0},
        {0.0, 1.55, 0.02, 0.03, 0.05, 0.4, 0.4, 2.0},
    };
    struct { double tx, ty, tz, wix, wiy, wiz, u1, u2, u3, u4; } cases[] = {
        {1, 0, 0,  0.3, -0.8, 0.5,  0.1, 0.5, 0.3, 0.7},
        {0, 1, 0,  0.6, 0.2, -0.7,  0.6, 0.2, 0.8, 0.1},
        {0.577, 0.577, 0.577,  -0.4, -0.4, 0.8,  0.9, 0.9, 0.05, 0.4},
        {1, 0, 0,  -0.9, 0.1, 0.2,  0.99, 0.01, 0.6, 0.99},
    };
    int nMat = 3, nCase = 4;
    for (int m = 0; m < nMat; ++m) {
        HairBxDF<double> ref(materials[m].h, materials[m].eta, materials[m].sr, materials[m].sg, materials[m].sb,
                              materials[m].beta_m, materials[m].beta_n, materials[m].alpha);
        HairBxDFParamsGPU gpuParams = makeHairParamsGPU(ref);
        std::vector<HairBxDFParamsGPU> paramsRep(nCase, gpuParams);
        id<MTLBuffer> paramsBuf = makeBuffer(device, paramsRep.data(), nCase * sizeof(HairBxDFParamsGPU));

        std::vector<simd::float3> tangents(nCase), wis(nCase);
        std::vector<simd::float4> us(nCase);
        std::vector<BxDFSampleResult<double>> expected(nCase);
        for (int c = 0; c < nCase; ++c) {
            double tlen = std::sqrt(cases[c].tx * cases[c].tx + cases[c].ty * cases[c].ty + cases[c].tz * cases[c].tz);
            double tx = cases[c].tx / tlen, ty = cases[c].ty / tlen, tz = cases[c].tz / tlen;
            double wlen = std::sqrt(cases[c].wix * cases[c].wix + cases[c].wiy * cases[c].wiy + cases[c].wiz * cases[c].wiz);
            double wix = cases[c].wix / wlen, wiy = cases[c].wiy / wlen, wiz = cases[c].wiz / wlen;
            tangents[c] = simd::float3{(float)tx, (float)ty, (float)tz};
            wis[c] = simd::float3{(float)wix, (float)wiy, (float)wiz};
            us[c] = simd::float4{(float)cases[c].u1, (float)cases[c].u2, (float)cases[c].u3, (float)cases[c].u4};
            expected[c] = ref.sample(tx, ty, tz, wix, wiy, wiz, cases[c].u1, cases[c].u2, cases[c].u3, cases[c].u4);
        }
        id<MTLBuffer> tanBuf = makeBuffer(device, tangents.data(), nCase * sizeof(simd::float3));
        id<MTLBuffer> wiBuf = makeBuffer(device, wis.data(), nCase * sizeof(simd::float3));
        id<MTLBuffer> uBuf = makeBuffer(device, us.data(), nCase * sizeof(simd::float4));
        id<MTLBuffer> woOutBuf = makeOutputBuffer(device, nCase * sizeof(simd::float3));
        id<MTLBuffer> rgbOutBuf = makeOutputBuffer(device, nCase * sizeof(simd::float3));
        id<MTLBuffer> validOutBuf = makeOutputBuffer(device, nCase * sizeof(int32_t));
        if (!runKernel(device, library, queue, @"test_hairSample",
                       @[paramsBuf, tanBuf, wiBuf, uBuf, woOutBuf, rgbOutBuf, validOutBuf], nil, nCase)) continue;
        simd::float3* woOut = (simd::float3*)woOutBuf.contents;
        simd::float3* rgbOut = (simd::float3*)rgbOutBuf.contents;
        int32_t* validOut = (int32_t*)validOutBuf.contents;
        for (int c = 0; c < nCase; ++c) {
            char label[192];
            snprintf(label, sizeof(label), "hairSample valid flag matches reference (material %d, case %d)", m, c);
            expectTrue(label, (bool)validOut[c] == expected[c].valid);
            if (!expected[c].valid) continue;
            snprintf(label, sizeof(label), "hairSample wo.x matches reference (material %d, case %d)", m, c);
            expectNear(label, woOut[c].x, expected[c].wo_x, 1e-2);
            snprintf(label, sizeof(label), "hairSample wo.y matches reference (material %d, case %d)", m, c);
            expectNear(label, woOut[c].y, expected[c].wo_y, 1e-2);
            snprintf(label, sizeof(label), "hairSample wo.z matches reference (material %d, case %d)", m, c);
            expectNear(label, woOut[c].z, expected[c].wo_z, 1e-2);
            snprintf(label, sizeof(label), "hairSample r matches reference (material %d, case %d)", m, c);
            expectNear(label, rgbOut[c].x, expected[c].r, std::max(1e-2, std::fabs(expected[c].r) * 0.05));
            snprintf(label, sizeof(label), "hairSample g matches reference (material %d, case %d)", m, c);
            expectNear(label, rgbOut[c].y, expected[c].g, std::max(1e-2, std::fabs(expected[c].g) * 0.05));
            snprintf(label, sizeof(label), "hairSample b matches reference (material %d, case %d)", m, c);
            expectNear(label, rgbOut[c].z, expected[c].b, std::max(1e-2, std::fabs(expected[c].b) * 0.05));
            snprintf(label, sizeof(label), "hairSample r/g/b non-negative (material %d, case %d)", m, c);
            expectTrue(label, rgbOut[c].x > -1e-4 && rgbOut[c].y > -1e-4 && rgbOut[c].z > -1e-4);
        }
    }
}

// Diagnostic: B11's own "fine black fur" sphere (sigma_a=(0.50,0.55,0.60),
// beta_m=beta_n=0.15, alpha=2, eta=1.55 - scene_advanced.h's own
// build_hair_fibers() 5th sphere) renders fully black on GPU but as the
// BRIGHTEST sphere on CPU at the same sample count - not just noise. One
// plausible mechanism: if hairSample() returns invalid (pdf<1e-8) far more
// often on GPU than the reference for this specific high-absorption
// material, shadeHair() returns false and the bounce loop breaks
// immediately, contributing zero further radiance - which could turn an
// entire sphere black even though the individual eval/pdf/sample cross-
// checks above (lower-sigma materials) all passed. Sweeps many random
// (h, u1..u4, wi direction) draws for THIS exact material and compares
// the fraction of valid samples (and mean r/g/b when valid) against the
// HairBxDF<double> reference, to see whether GPU's invalid rate is
// systematically higher.
void testHairBlackFurValidRate(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    const double sr = 0.50, sg = 0.55, sb = 0.60, beta_m = 0.15, beta_n = 0.15, alpha_deg = 2.0, eta = 1.55;
    std::mt19937 rng(777);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::uniform_real_distribution<double> zeroOne(0.0, 1.0);

    const int n = 200;
    std::vector<HairBxDFParamsGPU> paramsRep(n);
    std::vector<simd::float3> tangents(n), wis(n);
    std::vector<simd::float4> us(n);
    int refValidCount = 0;
    double refSumR = 0.0, refSumG = 0.0, refSumB = 0.0;
    for (int i = 0; i < n; ++i) {
        double h = unit(rng);
        HairBxDF<double> ref(h, eta, sr, sg, sb, beta_m, beta_n, alpha_deg);
        paramsRep[i] = makeHairParamsGPU(ref);

        // Random tangent direction (uniform-ish over the sphere) - a real
        // sphere's facing normal varies across its whole surface, unlike
        // this test's earlier fixed (0,1,0) version.
        double tx = unit(rng), ty = unit(rng), tz = unit(rng);
        double tlen = std::sqrt(tx * tx + ty * ty + tz * tz);
        if (tlen < 1e-6) tlen = 1.0;
        tx /= tlen; ty /= tlen; tz /= tlen;
        double wix = unit(rng), wiy = unit(rng), wiz = unit(rng);
        double wlen = std::sqrt(wix * wix + wiy * wiy + wiz * wiz);
        if (wlen < 1e-6) wlen = 1.0;
        wix /= wlen; wiy /= wlen; wiz /= wlen;
        double u1 = zeroOne(rng), u2 = zeroOne(rng), u3 = zeroOne(rng), u4 = zeroOne(rng);
        tangents[i] = simd::float3{(float)tx, (float)ty, (float)tz};
        wis[i] = simd::float3{(float)wix, (float)wiy, (float)wiz};
        us[i] = simd::float4{(float)u1, (float)u2, (float)u3, (float)u4};

        BxDFSampleResult<double> res = ref.sample(tx, ty, tz, wix, wiy, wiz, u1, u2, u3, u4);
        if (res.valid) {
            ++refValidCount;
            refSumR += res.r; refSumG += res.g; refSumB += res.b;
        }
    }

    id<MTLBuffer> paramsBuf = makeBuffer(device, paramsRep.data(), n * sizeof(HairBxDFParamsGPU));
    id<MTLBuffer> tanBuf = makeBuffer(device, tangents.data(), n * sizeof(simd::float3));
    id<MTLBuffer> wiBuf = makeBuffer(device, wis.data(), n * sizeof(simd::float3));
    id<MTLBuffer> uBuf = makeBuffer(device, us.data(), n * sizeof(simd::float4));
    id<MTLBuffer> woOutBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    id<MTLBuffer> rgbOutBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    id<MTLBuffer> validOutBuf = makeOutputBuffer(device, n * sizeof(int32_t));
    if (!runKernel(device, library, queue, @"test_hairSample",
                   @[paramsBuf, tanBuf, wiBuf, uBuf, woOutBuf, rgbOutBuf, validOutBuf], nil, n)) return;
    simd::float3* rgbOut = (simd::float3*)rgbOutBuf.contents;
    int32_t* validOut = (int32_t*)validOutBuf.contents;
    int gpuValidCount = 0;
    double gpuSumR = 0.0, gpuSumG = 0.0, gpuSumB = 0.0;
    for (int i = 0; i < n; ++i) {
        if (validOut[i]) {
            ++gpuValidCount;
            gpuSumR += rgbOut[i].x; gpuSumG += rgbOut[i].y; gpuSumB += rgbOut[i].z;
        }
    }
    fprintf(stderr, "[B11 black-fur diagnostic] reference valid: %d/%d (mean rgb when valid: %.4f %.4f %.4f)\n",
            refValidCount, n, refValidCount ? refSumR / refValidCount : 0.0,
            refValidCount ? refSumG / refValidCount : 0.0, refValidCount ? refSumB / refValidCount : 0.0);
    fprintf(stderr, "[B11 black-fur diagnostic] GPU valid:       %d/%d (mean rgb when valid: %.4f %.4f %.4f)\n",
            gpuValidCount, n, gpuValidCount ? gpuSumR / gpuValidCount : 0.0,
            gpuValidCount ? gpuSumG / gpuValidCount : 0.0, gpuValidCount ? gpuSumB / gpuValidCount : 0.0);
    char label[128];
    snprintf(label, sizeof(label), "B11 black-fur material: GPU valid rate matches reference (ref=%d, gpu=%d, n=%d)",
             refValidCount, gpuValidCount, n);
    expectTrue(label, std::abs(gpuValidCount - refValidCount) <= n / 10);
}

// Direct test of the DEGENERATE case a sphere's own surface normal used
// as HairBxDF's fiber tangent hits at the CENTER of its visible disc: the
// camera ray arrives ANTI-PARALLEL to the tangent (wi = -tangent exactly),
// meaning sinTheta_o = wo_x = 1, cosTheta_o = 0 - the extreme grazing-
// along-fiber case. The black-fur render's own mean brightness was FLAT
// between 256 and 2048 samples (27.7 vs 28.1, not trending toward CPU's
// 188.7) - a stable WRONG expected value, not slow convergence - so this
// averages hairSample()'s own r/g/b over MANY random u1..u4 draws at
// EXACTLY this geometry and compares the MEAN against the reference's
// own mean at the same fixed geometry, to see whether the two estimators'
// EXPECTED VALUES themselves disagree specifically in this regime (not
// just individual samples, which the earlier random-tangent sweep already
// found matching bit-for-bit on average, but that sweep rarely landed
// exactly at this measure-zero cosTheta_o=0 case).
void testHairGrazingCenterBias(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    const double sr = 0.50, sg = 0.55, sb = 0.60, beta_m = 0.15, beta_n = 0.15, alpha_deg = 2.0, eta = 1.55;
    std::mt19937 rng(4242);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::uniform_real_distribution<double> zeroOne(0.0, 1.0);

    const int n = 500;
    std::vector<HairBxDFParamsGPU> paramsRep(n);
    std::vector<simd::float3> tangents(n), wis(n);
    std::vector<simd::float4> us(n);
    double refSumR = 0.0;
    int refValid = 0;
    for (int i = 0; i < n; ++i) {
        double h = unit(rng);
        HairBxDF<double> ref(h, eta, sr, sg, sb, beta_m, beta_n, alpha_deg);
        paramsRep[i] = makeHairParamsGPU(ref);
        // tangent = facing normal (world up, arbitrary but fixed), wi
        // EXACTLY anti-parallel to it - the exact center-of-disc geometry.
        double tx = 0.0, ty = 1.0, tz = 0.0;
        double wix = 0.0, wiy = -1.0, wiz = 0.0;
        tangents[i] = simd::float3{(float)tx, (float)ty, (float)tz};
        wis[i] = simd::float3{(float)wix, (float)wiy, (float)wiz};
        double u1 = zeroOne(rng), u2 = zeroOne(rng), u3 = zeroOne(rng), u4 = zeroOne(rng);
        us[i] = simd::float4{(float)u1, (float)u2, (float)u3, (float)u4};
        BxDFSampleResult<double> res = ref.sample(tx, ty, tz, wix, wiy, wiz, u1, u2, u3, u4);
        if (res.valid) { refSumR += res.r; ++refValid; }
    }

    id<MTLBuffer> paramsBuf = makeBuffer(device, paramsRep.data(), n * sizeof(HairBxDFParamsGPU));
    id<MTLBuffer> tanBuf = makeBuffer(device, tangents.data(), n * sizeof(simd::float3));
    id<MTLBuffer> wiBuf = makeBuffer(device, wis.data(), n * sizeof(simd::float3));
    id<MTLBuffer> uBuf = makeBuffer(device, us.data(), n * sizeof(simd::float4));
    id<MTLBuffer> woOutBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    id<MTLBuffer> rgbOutBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    id<MTLBuffer> validOutBuf = makeOutputBuffer(device, n * sizeof(int32_t));
    if (!runKernel(device, library, queue, @"test_hairSample",
                   @[paramsBuf, tanBuf, wiBuf, uBuf, woOutBuf, rgbOutBuf, validOutBuf], nil, n)) return;
    simd::float3* rgbOut = (simd::float3*)rgbOutBuf.contents;
    int32_t* validOut = (int32_t*)validOutBuf.contents;
    double gpuSumR = 0.0;
    int gpuValid = 0;
    for (int i = 0; i < n; ++i) {
        if (validOut[i]) { gpuSumR += rgbOut[i].x; ++gpuValid; }
    }
    fprintf(stderr, "[grazing-center diagnostic] reference: valid=%d/%d mean_r=%.4f (sum over ALL n, valid-or-not: %.4f)\n",
            refValid, n, refValid ? refSumR / refValid : 0.0, refSumR / n);
    fprintf(stderr, "[grazing-center diagnostic] GPU:       valid=%d/%d mean_r=%.4f (sum over ALL n, valid-or-not: %.4f)\n",
            gpuValid, n, gpuValid ? gpuSumR / gpuValid : 0.0, gpuSumR / n);
    char label[128];
    snprintf(label, sizeof(label), "grazing-center: GPU mean-over-all-n matches reference (gpu=%.4f, ref=%.4f)",
             gpuSumR / n, refSumR / n);
    expectNear(label, gpuSumR / n, refSumR / n, std::max(0.05, std::fabs(refSumR / n) * 0.1));
}

