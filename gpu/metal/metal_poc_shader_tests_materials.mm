// metal_poc_shader_tests_materials.mm
// BSDF/Fresnel/GGX/material numeric cross-checks and property tests, plus
// misc numeric edge-case checks (checkerColor, atan2) that don't belong
// to the media/lights/hair splits - one of 4 category files split out of
// metal_poc_shader_tests.mm (see metal_poc_shader_tests_common.h's own
// header comment for the full "why"). Pure code motion, no behaviour
// change - every function below moved verbatim except losing `static`
// (real external linkage now needed, main() calls these from a different
// translation unit).
#include "metal_poc_shader_tests_common.h"

void testFrDielectric(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // Reference values computed independently via a standalone double-
    // precision C program mirroring FrDielectric's own formula (not
    // copy-pasted from an earlier session's memory) - see this PR's own
    // commit message for that program's exact output.
    struct { float cosThetaI, eta, expected; } cases[] = {
        {1.0f, 1.5f, 0.04000000f},          // normal incidence: R0 = ((eta-1)/(eta+1))^2 exactly
        {0.5f, 1.5f, 0.08918671f},          // 60 degrees
        {0.08715574f, 1.5f, 0.61279965f},   // 85 degrees (near-grazing)
        {0.17364818f, 1.0f / 1.5f, 1.0f},   // total internal reflection (glass -> air, steep angle)
    };
    int n = 4;
    std::vector<simd::float2> inputs(n);
    for (int i = 0; i < n; ++i) inputs[i] = simd::float2{cases[i].cosThetaI, cases[i].eta};
    id<MTLBuffer> inBuf = makeBuffer(device, inputs.data(), inputs.size() * sizeof(simd::float2));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_frDielectric", @[inBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    for (int i = 0; i < n; ++i) {
        char label[128];
        snprintf(label, sizeof(label), "frDielectric(cos=%.5f, eta=%.5f)", cases[i].cosThetaI, cases[i].eta);
        expectNear(label, out[i], cases[i].expected, 1e-4);
    }
}

namespace dielectric_test_detail {
// Host-side transcriptions of Metal's own `reflect`/`refract` built-ins
// (MSL matches the GLSL definitions exactly - this project already relies
// on that equivalence everywhere else this file uses double-precision
// stand-ins for GPU intrinsics), used ONLY as an independent reference to
// compare shadeDielectric()'s/shadeThinDielectric()'s ACTUAL returned
// direction against, never called from the shader itself.
simd::double3 reflectRef(simd::double3 I, simd::double3 N) {
    return I - 2.0 * simd::dot(N, I) * N;
}
// Returns {0,0,0} on TIR, matching Metal's own refract() - callers here
// only invoke it once cannotRefract has already been ruled out, same as
// shadeDielectric() itself does.
simd::double3 refractRef(simd::double3 I, simd::double3 N, double eta) {
    double NdotI = simd::dot(N, I);
    double k = 1.0 - eta * eta * (1.0 - NdotI * NdotI);
    if (k < 0.0) return simd::double3{0.0, 0.0, 0.0};
    return eta * I - (eta * NdotI + std::sqrt(k)) * N;
}
simd::double3 d3(simd::float3 v) { return simd::double3{v.x, v.y, v.z}; }
simd::float3 f3(simd::double3 v) { return simd::float3{(float)v.x, (float)v.y, (float)v.z}; }
bool isFiniteF3(simd::float3 v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }
}  // namespace dielectric_test_detail

// shadeDielectric() (metal_poc_materials_specular.metal, materialType 2) -
// the reflect-vs-refract branch selection, Snell-law refraction direction,
// and TIR routing that testFrDielectric() above does NOT exercise (it only
// checks the raw Fresnel-reflectance scalar, never the shading function
// that consumes it). Every case below is constructed so the RNG draw
// cannot change the outcome - matched-media (ior=1.0) forces transmission
// unconditionally (frDielectric(eta=1) is exactly 0, so "reflectance >
// randFloat" is false no matter what randFloat returns), and a beyond-
// critical-angle exit forces TIR unconditionally (shadeDielectric()'s own
// `cannotRefract ||` short-circuits before the RNG comparison) - so these
// are exact geometric checks, not statistical ones.
void testShadeDielectric(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    using namespace dielectric_test_detail;

    struct Case {
        const char *name;
        simd::float3 rayDir, normal, facingNormal;
        float ior;
        simd::float3 absorption;
        bool frontFace;
        float hitDistance;
        // What to check for this case.
        bool expectReflect;   // true: expect reflectRef(); false: expect refractRef()
        double refractEta;    // eta to feed refractRef() when expectReflect is false
    };

    // Incidence ~48.19 deg (cos=2/3): sin(48.19)=0.7454. Front-face air->
    // glass (ior=1.5): refractionRatio=1/1.5, ratio*sin=0.497 < 1, refracts.
    // Back-face glass->air (ior=1.5) at the SAME angle: refractionRatio=1.5,
    // ratio*sin=1.118 > 1 -> TIR, forced reflection regardless of RNG (this
    // is exactly the "steep exit angle" case METAL_GPU_FEASIBILITY.md and
    // this codebase's own frDielectric test already use for TIR, section
    // above's `{0.17364818f, 1.0f/1.5f, 1.0f}` case, just exercised through
    // the full shading function instead of the raw Fresnel scalar).
    const simd::float3 obliqueDir = simd::normalize(simd::float3{0.7454f, -0.6667f, 0.0f});
    const simd::float3 upNormal{0.0f, 1.0f, 0.0f};

    const simd::float3 noAbsorption = simd::float3{0.0f, 0.0f, 0.0f};
    std::vector<Case> cases = {
        {"front-face oblique refraction (air->glass, ior=1.5)",
         obliqueDir, upNormal, upNormal, 1.5f, noAbsorption, true, 0.0f,
         /*expectReflect=*/false, /*refractEta=*/1.0 / 1.5},

        {"back-face oblique TIR (glass->air, ior=1.5, beyond critical angle)",
         obliqueDir, upNormal, -upNormal, 1.5f, simd::float3{0.3f, 0.3f, 0.3f}, false, 2.0f,
         /*expectReflect=*/true, /*refractEta=*/0.0},

        {"matched-media straight-through (ior=1.0, oblique, front-face)",
         obliqueDir, upNormal, upNormal, 1.0f, noAbsorption, true, 0.0f,
         /*expectReflect=*/false, /*refractEta=*/1.0},

        {"matched-media straight-through (ior=1.0, oblique, back-face)",
         obliqueDir, upNormal, -upNormal, 1.0f, simd::float3{0.5f, 0.5f, 0.5f}, false, 3.0f,
         /*expectReflect=*/false, /*refractEta=*/1.0},
    };

    std::vector<DielectricTestInput> inputs;
    for (const Case &c : cases) {
        // Two different rngSeeds per case - the whole point is that the
        // outcome must NOT depend on which one lands, since every case
        // above is constructed to force a deterministic branch.
        for (uint32_t seed : {12345u, 999999937u}) {
            DielectricTestInput in{};
            in.rayDir = c.rayDir; in.normal = c.normal; in.facingNormal = c.facingNormal;
            in.ior = c.ior; in.absorption = c.absorption;
            in.frontFace = c.frontFace ? 1u : 0u; in.hitDistance = c.hitDistance;
            in.rngSeed = seed;
            inputs.push_back(in);
        }
    }
    id<MTLBuffer> inBuf = makeBuffer(device, inputs.data(), inputs.size() * sizeof(DielectricTestInput));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, inputs.size() * sizeof(DielectricTestOutput));
    if (!runKernel(device, library, queue, @"test_shadeDielectric", @[inBuf, outBuf], nil, (int)inputs.size())) return;
    DielectricTestOutput *out = (DielectricTestOutput *)outBuf.contents;

    for (size_t i = 0; i < cases.size(); ++i) {
        const Case &c = cases[i];
        for (int seedIdx = 0; seedIdx < 2; ++seedIdx) {
            const DielectricTestOutput &o = out[i * 2 + seedIdx];
            char label[192];

            snprintf(label, sizeof(label), "shadeDielectric: %s (seed %d) - direction is finite", c.name, seedIdx);
            expectTrue(label, isFiniteF3(simd::float3(o.rayDir)));

            simd::double3 unitDir = simd::normalize(d3(c.rayDir));
            simd::double3 expectedDir = c.expectReflect
                ? reflectRef(unitDir, d3(c.facingNormal))
                : refractRef(unitDir, d3(c.facingNormal), c.refractEta);
            simd::float3 expectedDirF = f3(expectedDir);
            snprintf(label, sizeof(label), "shadeDielectric: %s (seed %d) - direction matches %s",
                     c.name, seedIdx, c.expectReflect ? "reflect()" : "refract() (Snell's law)");
            expectNear(label, simd::length(simd::float3(o.rayDir) - expectedDirF), 0.0, 1e-4);

            snprintf(label, sizeof(label), "shadeDielectric: %s (seed %d) - specularBounce is true", c.name, seedIdx);
            expectTrue(label, o.specularBounce != 0);

            // Absorption only applies on the backface (applyBeerLambertAbsorption's
            // own gate) - verify the exact exp(-absorption*distance) factor when it
            // should fire, and that throughput is untouched (still 1,1,1) otherwise.
            simd::float3 expectedThroughput = c.frontFace
                ? simd::float3{1.0f, 1.0f, 1.0f}
                : simd::float3{(float)std::exp(-c.absorption.x * c.hitDistance),
                               (float)std::exp(-c.absorption.y * c.hitDistance),
                               (float)std::exp(-c.absorption.z * c.hitDistance)};
            snprintf(label, sizeof(label), "shadeDielectric: %s (seed %d) - Beer-Lambert throughput", c.name, seedIdx);
            expectNear(label, simd::length(simd::float3(o.throughput) - expectedThroughput), 0.0, 1e-5);
        }
    }
}

// shadeThinDielectric() (metal_poc_materials_specular.metal, materialType
// 23) - same "force a deterministic branch, check it exactly" approach as
// testShadeDielectric() above. ior=1.0 makes frDielectric() return exactly
// 0, so thinR (after the multi-bounce boost, which leaves 0 unchanged)
// stays 0 and the function transmits unconditionally; a near-grazing angle
// pushes frDielectric() (and therefore the boosted thinR) to within a
// tiny, explicitly-bounded distance of 1.0, making reflection overwhelmingly
// likely for any of the 8 independent seeds tried - not mathematically
// airtight the way the ior=1.0 case is, but the same "several independent
// seeds must agree" standard this file's own coated-material property
// tests already use, and the actual probability of a false failure here
// (a uniform draw landing inside a sub-1e-5-wide window, 8 times running)
// is astronomically smaller than that tolerance suggests.
void testShadeThinDielectric(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    using namespace dielectric_test_detail;

    const simd::float3 upNormal{0.0f, 1.0f, 0.0f};
    const simd::float3 obliqueDir = simd::normalize(simd::float3{0.7454f, -0.6667f, 0.0f});
    // cosTheta ~ 0.006 (theta ~ 89.6 deg): frDielectric(eta != 1) here is
    // above 0.999 for any eta this test uses (eta=1.5 -> R ~= 0.9996), so
    // the boosted thinR is even higher - "virtually always reflects."
    const simd::float3 grazingDir = simd::normalize(simd::float3{0.99998f, -0.006f, 0.0f});

    std::vector<uint32_t> seeds = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};

    // Case A: matched media (ior=1.0) - must transmit (straight-through,
    // unchanged direction) for EVERY seed, no exceptions.
    {
        std::vector<DielectricTestInput> inputs;
        for (uint32_t seed : seeds) {
            DielectricTestInput in{};
            in.rayDir = obliqueDir; in.normal = upNormal; in.facingNormal = upNormal;
            in.ior = 1.0f; in.rngSeed = seed;
            inputs.push_back(in);
        }
        id<MTLBuffer> inBuf = makeBuffer(device, inputs.data(), inputs.size() * sizeof(DielectricTestInput));
        id<MTLBuffer> outBuf = makeOutputBuffer(device, inputs.size() * sizeof(DielectricTestOutput));
        if (runKernel(device, library, queue, @"test_shadeThinDielectric", @[inBuf, outBuf], nil, (int)inputs.size())) {
            DielectricTestOutput *out = (DielectricTestOutput *)outBuf.contents;
            for (size_t i = 0; i < inputs.size(); ++i) {
                char label[160];
                snprintf(label, sizeof(label),
                         "shadeThinDielectric: matched media (ior=1.0) transmits straight-through (seed %u)", seeds[i]);
                expectNear(label, simd::length(simd::float3(out[i].rayDir) - obliqueDir), 0.0, 1e-5);
                snprintf(label, sizeof(label), "shadeThinDielectric: matched media - throughput untinted (seed %u)", seeds[i]);
                expectNear(label, simd::length(simd::float3(out[i].throughput) - simd::float3{1, 1, 1}), 0.0, 1e-6);
            }
        }
    }

    // Case B: near-grazing incidence, ior=1.5 - must reflect for every seed.
    {
        std::vector<DielectricTestInput> inputs;
        for (uint32_t seed : seeds) {
            DielectricTestInput in{};
            in.rayDir = grazingDir; in.normal = upNormal; in.facingNormal = upNormal;
            in.ior = 1.5f; in.rngSeed = seed;
            inputs.push_back(in);
        }
        id<MTLBuffer> inBuf = makeBuffer(device, inputs.data(), inputs.size() * sizeof(DielectricTestInput));
        id<MTLBuffer> outBuf = makeOutputBuffer(device, inputs.size() * sizeof(DielectricTestOutput));
        if (runKernel(device, library, queue, @"test_shadeThinDielectric", @[inBuf, outBuf], nil, (int)inputs.size())) {
            DielectricTestOutput *out = (DielectricTestOutput *)outBuf.contents;
            simd::double3 expectedReflect = reflectRef(d3(grazingDir), d3(upNormal));
            simd::float3 expectedReflectF = f3(expectedReflect);
            for (size_t i = 0; i < inputs.size(); ++i) {
                char label[160];
                snprintf(label, sizeof(label),
                         "shadeThinDielectric: near-grazing incidence reflects (seed %u)", seeds[i]);
                expectNear(label, simd::length(simd::float3(out[i].rayDir) - expectedReflectF), 0.0, 1e-3);
            }
        }
    }
}

// lambertianPdf()/diffuseTransmissionPdf() (metal_poc_sampling.metal) -
// numeric cross-check against pbrt-v4's own DiffuseBxDF<double>/
// DiffuseTransmissionBxDF<double>::scattering_pdf() (src/shared/
// bxdfs_simple.h, bxdfs_layered.h), the SAME trusted CPU/OptiX reference
// the CPU/OptiX renderers already use for these two materials - not a
// hand-copied duplicate, the same "call the exact production formula"
// principle testHairRandomSweep() already established for HairBxDF.
// These two functions are the ones shadeLambertian()/
// shadeDiffuseTransmission() (metal_poc_materials_diffuse.metal) call
// directly for their own NEE MIS weights, so this covers real, currently-
// running production code, not a parallel reimplementation of it.
void testLambertianAndDiffuseTransmissionPdf(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    std::mt19937 rng(4242);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::uniform_real_distribution<double> zeroOne(0.0, 1.0);

    // --- lambertianPdf(): fixed edge cases + a random sweep -------------
    {
        std::vector<float> cosWis;
        std::vector<double> expected;
        auto addCase = [&](double cosWi) {
            cosWis.push_back((float)cosWi);
            DiffuseBxDF<double> ref{};  // albedo unused by scattering_pdf(); pdf depends only on cos_theta
            double wox = std::sqrt(std::max(0.0, 1.0 - cosWi * cosWi));
            expected.push_back(ref.scattering_pdf(0.0, 0.0, 1.0, wox, 0.0, cosWi));
        };
        addCase(1.0);    // normal incidence: 1/pi
        addCase(0.0);    // grazing: 0
        addCase(-1.0);   // below surface: 0 (not -1/pi)
        addCase(0.5);
        for (int i = 0; i < 40; ++i) addCase(unit(rng));

        int n = (int)cosWis.size();
        id<MTLBuffer> inBuf = makeBuffer(device, cosWis.data(), n * sizeof(float));
        id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
        if (runKernel(device, library, queue, @"test_lambertianPdf", @[inBuf, outBuf], nil, n)) {
            float* out = (float*)outBuf.contents;
            for (int i = 0; i < n; ++i) {
                char label[128];
                snprintf(label, sizeof(label), "lambertianPdf(cosWi=%.5f) matches DiffuseBxDF::scattering_pdf", cosWis[i]);
                expectNear(label, out[i], expected[i], 1e-5);
                snprintf(label, sizeof(label), "lambertianPdf(cosWi=%.5f) is non-negative", cosWis[i]);
                expectTrue(label, out[i] >= 0.0f);
            }
        }
    }

    // --- diffuseTransmissionPdf(): fixed edge cases + a random sweep ----
    {
        std::vector<simd::float3> inputs;  // (cosWi, pr, pt)
        std::vector<double> expected;
        auto addCase = [&](double cosWi, double pr, double pt) {
            inputs.push_back(simd::float3{(float)cosWi, (float)pr, (float)pt});
            DiffuseTransmissionBxDF<double> ref{pr, pr, pr, pt, pt, pt};
            double wox = std::sqrt(std::max(0.0, 1.0 - cosWi * cosWi));
            expected.push_back(ref.scattering_pdf(0.0, 0.0, 1.0, wox, 0.0, cosWi));
        };
        addCase(1.0, 1.0, 0.0);    // pure reflector, normal incidence: 1/pi
        addCase(-1.0, 1.0, 0.0);   // pure reflector, transmission side: pdf 0 (pt=0)
        addCase(-1.0, 0.0, 1.0);   // pure transmitter, transmission side: 1/pi
        addCase(1.0, 0.5, 0.5);    // balanced lobes, reflection side: 0.5/pi
        addCase(-1.0, 0.5, 0.5);   // balanced lobes, transmission side: 0.5/pi
        for (int i = 0; i < 40; ++i) {
            double pr = zeroOne(rng), pt = zeroOne(rng);
            if (pr + pt < 1e-6) pr = 0.5;  // avoid the degenerate pr+pt==0 case addressed separately
            addCase(unit(rng), pr, pt);
        }

        int n = (int)inputs.size();
        id<MTLBuffer> inBuf = makeBuffer(device, inputs.data(), n * sizeof(simd::float3));
        id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
        if (runKernel(device, library, queue, @"test_diffuseTransmissionPdf", @[inBuf, outBuf], nil, n)) {
            float* out = (float*)outBuf.contents;
            for (int i = 0; i < n; ++i) {
                char label[160];
                snprintf(label, sizeof(label), "diffuseTransmissionPdf(cosWi=%.5f, pr=%.3f, pt=%.3f) matches reference",
                         inputs[i].x, inputs[i].y, inputs[i].z);
                expectNear(label, out[i], expected[i], 1e-5);
                snprintf(label, sizeof(label), "diffuseTransmissionPdf(cosWi=%.5f, pr=%.3f, pt=%.3f) is non-negative",
                         inputs[i].x, inputs[i].y, inputs[i].z);
                expectTrue(label, out[i] >= 0.0f);
            }
        }
    }
}

// ggxConductorF()/ggxConductorPdf() (metal_poc_sampling.metal), full
// pipeline (raw local-frame directions -> ggxD/ggxG/ggxG1/frComplexRGB
// -> the final combine) - numeric cross-check against pbrt-v4's own
// ConductorBxDF<double>::f()/pdf() (src/shared/bxdfs_conductor.h), the
// SAME trusted CPU/OptiX reference those renderers already use for
// rough conductors. Same "call the real production formula, don't
// duplicate it" principle as testLambertianAndDiffuseTransmissionPdf()
// (PR #190) and testHairRandomSweep() before it.
//
// Direction-naming note: ConductorBxDF's own `sample_local(wi, ...)`
// conditions its VNDF sample ON `wi` and produces `wo` - i.e. its "wi"
// is the GIVEN/starting direction and "wo" the sampled/queried one,
// confirmed by cross-reading `f()`'s/`pdf()`'s own formulas (symmetric
// in the D*G term, but `pdf()`'s own G1 is evaluated at "wi" only).
// shadeConductor()'s own `woLocal` (the view direction every NEE/BSDF-
// sample call in that function conditions on) is that same "given"
// role, and `wiLocal` (the queried light/sampled direction) is the
// "produced" one - so `ref.f(wi=woLocal, wo=wiLocal)` is the CORRECT
// mapping, not a naming coincidence to get backwards.
void testGgxConductorFAndPdf(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    std::mt19937 rng(9001);
    std::uniform_real_distribution<double> zeroOne(0.0, 1.0);
    std::uniform_real_distribution<double> alphaDist(0.01, 0.9);
    std::uniform_real_distribution<double> etaDist(0.2, 3.0);
    std::uniform_real_distribution<double> kDist(0.0, 4.0);

    auto randomUpperHemisphereDir = [&]() -> simd::double3 {
        // Uniform-ish (not physically meaningful, just needs z>0 and to
        // cover a range of grazing/near-normal angles) - cosine-weighted
        // would bias away from the grazing cases this test also wants
        // to exercise.
        double z = 0.05 + zeroOne(rng) * 0.9;  // keep away from the pole to avoid a degenerate h
        double phi = zeroOne(rng) * 2.0 * M_PI;
        double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        return simd::double3{r * std::cos(phi), r * std::sin(phi), z};
    };

    const int n = 60;
    std::vector<simd::float3> wos(n), wis(n);
    std::vector<simd::float2> alphas(n);
    std::vector<simd::float3> etas(n), ks(n);
    std::vector<double> expectedF(n), expectedPdf(n);
    for (int i = 0; i < n; ++i) {
        simd::double3 wo = randomUpperHemisphereDir();
        simd::double3 wi = randomUpperHemisphereDir();
        double alphaX = alphaDist(rng);
        double alphaY = alphaDist(rng);
        double eta = etaDist(rng), k = kDist(rng);
        wos[i] = simd::float3{(float)wo.x, (float)wo.y, (float)wo.z};
        wis[i] = simd::float3{(float)wi.x, (float)wi.y, (float)wi.z};
        alphas[i] = simd::float2{(float)alphaX, (float)alphaY};
        etas[i] = simd::float3{(float)eta, (float)eta, (float)eta};
        ks[i] = simd::float3{(float)k, (float)k, (float)k};

        ConductorBxDF<double> ref{eta, eta, eta, k, k, k, alphaX, alphaY};
        double fr, fg, fb;
        ref.f(wo.x, wo.y, wo.z, wi.x, wi.y, wi.z, fr, fg, fb);
        expectedF[i] = fr;  // eta/k are the same across channels above, so r==g==b
        expectedPdf[i] = ref.pdf(wo.x, wo.y, wo.z, wi.x, wi.y, wi.z);
    }

    id<MTLBuffer> woBuf = makeBuffer(device, wos.data(), n * sizeof(simd::float3));
    id<MTLBuffer> wiBuf = makeBuffer(device, wis.data(), n * sizeof(simd::float3));
    id<MTLBuffer> alphaBuf = makeBuffer(device, alphas.data(), n * sizeof(simd::float2));
    id<MTLBuffer> etaBuf = makeBuffer(device, etas.data(), n * sizeof(simd::float3));
    id<MTLBuffer> kBuf = makeBuffer(device, ks.data(), n * sizeof(simd::float3));
    id<MTLBuffer> fOutBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    id<MTLBuffer> pdfOutBuf = makeOutputBuffer(device, n * sizeof(float));

    if (runKernel(device, library, queue, @"test_ggxConductorF",
                  @[woBuf, wiBuf, alphaBuf, etaBuf, kBuf, fOutBuf], nil, n)) {
        simd::float3* fOut = (simd::float3*)fOutBuf.contents;
        for (int i = 0; i < n; ++i) {
            char label[160];
            snprintf(label, sizeof(label), "ggxConductorF matches ConductorBxDF::f (case %d)", i);
            expectNear(label, fOut[i].x, expectedF[i], std::max(1e-5, std::fabs(expectedF[i]) * 1e-3));
            snprintf(label, sizeof(label), "ggxConductorF is non-negative (case %d)", i);
            expectTrue(label, fOut[i].x >= 0.0f);
        }
    }
    if (runKernel(device, library, queue, @"test_ggxConductorPdf",
                  @[woBuf, wiBuf, alphaBuf, pdfOutBuf], nil, n)) {
        float* pdfOut = (float*)pdfOutBuf.contents;
        for (int i = 0; i < n; ++i) {
            char label[160];
            snprintf(label, sizeof(label), "ggxConductorPdf matches ConductorBxDF::pdf (case %d)", i);
            expectNear(label, pdfOut[i], expectedPdf[i], std::max(1e-5, std::fabs(expectedPdf[i]) * 1e-3));
            snprintf(label, sizeof(label), "ggxConductorPdf is non-negative (case %d)", i);
            expectTrue(label, pdfOut[i] >= 0.0f);
        }
    }

    // Below-hemisphere regression: a direction at/below the shading plane
    // must give exactly 0, NOT the spuriously positive value the old
    // max(4*NdotO*NdotI, 1e-6) clamp alone returned for a negative product
    // (case 0: queried wi below the surface; case 1: given wo below it).
    {
        std::vector<simd::float3> wos2 = {simd::float3{0.30f, 0.0f, 0.9539f}, simd::float3{0.30f, 0.0f, -0.20f}};
        std::vector<simd::float3> wis2 = {simd::float3{0.30f, 0.0f, -0.20f}, simd::float3{0.30f, 0.0f, 0.9539f}};
        std::vector<simd::float2> alphas2 = {simd::float2{0.3f, 0.3f}, simd::float2{0.3f, 0.3f}};
        std::vector<simd::float3> etas2 = {simd::float3{0.2f, 0.2f, 0.2f}, simd::float3{0.2f, 0.2f, 0.2f}};
        std::vector<simd::float3> ks2 = {simd::float3{3.0f, 3.0f, 3.0f}, simd::float3{3.0f, 3.0f, 3.0f}};
        const int n2 = 2;
        id<MTLBuffer> woBuf2 = makeBuffer(device, wos2.data(), n2 * sizeof(simd::float3));
        id<MTLBuffer> wiBuf2 = makeBuffer(device, wis2.data(), n2 * sizeof(simd::float3));
        id<MTLBuffer> alphaBuf2 = makeBuffer(device, alphas2.data(), n2 * sizeof(simd::float2));
        id<MTLBuffer> etaBuf2 = makeBuffer(device, etas2.data(), n2 * sizeof(simd::float3));
        id<MTLBuffer> kBuf2 = makeBuffer(device, ks2.data(), n2 * sizeof(simd::float3));
        id<MTLBuffer> fOutBuf2 = makeOutputBuffer(device, n2 * sizeof(simd::float3));
        id<MTLBuffer> pdfOutBuf2 = makeOutputBuffer(device, n2 * sizeof(float));
        if (runKernel(device, library, queue, @"test_ggxConductorF",
                      @[woBuf2, wiBuf2, alphaBuf2, etaBuf2, kBuf2, fOutBuf2], nil, n2)) {
            simd::float3* fOut2 = (simd::float3*)fOutBuf2.contents;
            expectNear("ggxConductorF is 0 when wi is below the hemisphere", fOut2[0].x, 0.0, 1e-9);
            expectNear("ggxConductorF is 0 when wo is below the hemisphere", fOut2[1].x, 0.0, 1e-9);
        }
        if (runKernel(device, library, queue, @"test_ggxConductorPdf",
                      @[woBuf2, wiBuf2, alphaBuf2, pdfOutBuf2], nil, n2)) {
            float* pdfOut2 = (float*)pdfOutBuf2.contents;
            expectNear("ggxConductorPdf is 0 when wo is below the hemisphere", pdfOut2[1], 0.0, 1e-9);
        }
    }
}

// Principled BSDF (materialType 24, B10) numeric cross-check - three
// pieces, mirroring section 191's own conductor test shape: the two
// already-standalone GGX helpers direct, then the combined 3-lobe pdf
// end to end. Reference: PrincipledBxDF<double> (src/shared/
// bxdfs_principled.h), the same shared header CPU/OptiX already build
// from.
void testPrincipledPdf(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    std::mt19937 rng(31415);
    std::uniform_real_distribution<double> zeroOne(0.0, 1.0);
    std::uniform_real_distribution<double> alphaDist(0.01, 0.9);
    std::uniform_real_distribution<double> iorDist(1.05, 2.5);

    auto randomUpperHemisphereDir = [&]() -> simd::double3 {
        double z = 0.05 + zeroOne(rng) * 0.9;
        double phi = zeroOne(rng) * 2.0 * M_PI;
        double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        return simd::double3{r * std::cos(phi), r * std::sin(phi), z};
    };

    const int n = 50;

    // --- principledGgxBrdf()/principledGgxPdf(): direct dispatch -----
    {
        std::vector<simd::float3> wos(n), wis(n);
        std::vector<float> alphas(n);
        std::vector<double> expectedBrdf(n), expectedPdf(n);
        for (int i = 0; i < n; ++i) {
            simd::double3 wo = randomUpperHemisphereDir();
            simd::double3 wi = randomUpperHemisphereDir();
            double alpha = alphaDist(rng);
            wos[i] = simd::float3{(float)wo.x, (float)wo.y, (float)wo.z};
            wis[i] = simd::float3{(float)wi.x, (float)wi.y, (float)wi.z};
            alphas[i] = (float)alpha;
            PrincipledBxDF<double> ref{};
            expectedBrdf[i] = ref.ggx_brdf(wo.x, wo.y, wo.z, wi.x, wi.y, wi.z, alpha);
            expectedPdf[i] = ref.ggx_pdf(wo.x, wo.y, wo.z, wi.x, wi.y, wi.z, alpha);
        }
        id<MTLBuffer> woBuf = makeBuffer(device, wos.data(), n * sizeof(simd::float3));
        id<MTLBuffer> wiBuf = makeBuffer(device, wis.data(), n * sizeof(simd::float3));
        id<MTLBuffer> alphaBuf = makeBuffer(device, alphas.data(), n * sizeof(float));
        id<MTLBuffer> brdfOutBuf = makeOutputBuffer(device, n * sizeof(float));
        id<MTLBuffer> pdfOutBuf = makeOutputBuffer(device, n * sizeof(float));
        if (runKernel(device, library, queue, @"test_principledGgxBrdf", @[woBuf, wiBuf, alphaBuf, brdfOutBuf], nil, n)) {
            float* out = (float*)brdfOutBuf.contents;
            for (int i = 0; i < n; ++i) {
                char label[128];
                snprintf(label, sizeof(label), "principledGgxBrdf matches PrincipledBxDF::ggx_brdf (case %d)", i);
                expectNear(label, out[i], expectedBrdf[i], std::max(1e-5, std::fabs(expectedBrdf[i]) * 1e-3));
            }
        }
        if (runKernel(device, library, queue, @"test_principledGgxPdf", @[woBuf, wiBuf, alphaBuf, pdfOutBuf], nil, n)) {
            float* out = (float*)pdfOutBuf.contents;
            for (int i = 0; i < n; ++i) {
                char label[128];
                snprintf(label, sizeof(label), "principledGgxPdf matches PrincipledBxDF::ggx_pdf (case %d)", i);
                expectNear(label, out[i], expectedPdf[i], std::max(1e-5, std::fabs(expectedPdf[i]) * 1e-3));
            }
        }
    }

    // --- principledCombinedPdf(): full pipeline vs scattering_pdf() --
    // Direction-naming note: PrincipledBxDF::scattering_pdf(n, wi, wo)
    // takes `wi` as the RAY'S OWN incident direction (toward the
    // surface) and `wo` as the newly scattered direction (away from
    // it), then negates `wi` internally to get its own local "view"
    // direction - i.e. `-wi` plays the role shadePrincipled()'s own
    // `woLocal` (the view/conditioning direction) does, and `wo` plays
    // the role `woOutLocal` (the queried/sampled direction) does. Using
    // an identity local frame (n=(0,0,1)) here since both GGX lobes are
    // isotropic (alpha_x==alpha_y always in this material), so the
    // world/local distinction doesn't affect the result and no real
    // tangent frame needs constructing.
    {
        std::vector<float> metallics(n), iors(n), clearcoats(n), roughnesses(n), clearcoatRoughnesses(n);
        std::vector<simd::float3> wos(n), wis(n);
        std::vector<double> expectedPdf(n);
        for (int i = 0; i < n; ++i) {
            simd::double3 woView = randomUpperHemisphereDir();      // shadePrincipled()'s own woLocal
            simd::double3 wiQuery = randomUpperHemisphereDir();     // shadePrincipled()'s own woOutLocal
            double metallic = zeroOne(rng);
            double ior = iorDist(rng);
            double clearcoat = zeroOne(rng);
            double roughness = zeroOne(rng);
            double clearcoatRoughness = zeroOne(rng);
            metallics[i] = (float)metallic;
            iors[i] = (float)ior;
            clearcoats[i] = (float)clearcoat;
            roughnesses[i] = (float)roughness;
            clearcoatRoughnesses[i] = (float)clearcoatRoughness;
            wos[i] = simd::float3{(float)woView.x, (float)woView.y, (float)woView.z};
            wis[i] = simd::float3{(float)wiQuery.x, (float)wiQuery.y, (float)wiQuery.z};

            PrincipledBxDF<double> ref{0.5, 0.5, 0.5, metallic, roughness, ior, clearcoat, clearcoatRoughness};
            simd::double3 cpuWi = -woView;  // ray's own incident direction (toward surface)
            expectedPdf[i] = ref.scattering_pdf(0.0, 0.0, 1.0,
                                                 cpuWi.x, cpuWi.y, cpuWi.z,
                                                 wiQuery.x, wiQuery.y, wiQuery.z);
        }
        id<MTLBuffer> metallicBuf = makeBuffer(device, metallics.data(), n * sizeof(float));
        id<MTLBuffer> iorBuf = makeBuffer(device, iors.data(), n * sizeof(float));
        id<MTLBuffer> clearcoatBuf = makeBuffer(device, clearcoats.data(), n * sizeof(float));
        id<MTLBuffer> roughBuf = makeBuffer(device, roughnesses.data(), n * sizeof(float));
        id<MTLBuffer> ccRoughBuf = makeBuffer(device, clearcoatRoughnesses.data(), n * sizeof(float));
        id<MTLBuffer> woBuf = makeBuffer(device, wos.data(), n * sizeof(simd::float3));
        id<MTLBuffer> wiBuf = makeBuffer(device, wis.data(), n * sizeof(simd::float3));
        id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
        if (runKernel(device, library, queue, @"test_principledCombinedPdf",
                      @[metallicBuf, iorBuf, clearcoatBuf, roughBuf, ccRoughBuf, woBuf, wiBuf, outBuf], nil, n)) {
            float* out = (float*)outBuf.contents;
            for (int i = 0; i < n; ++i) {
                char label[160];
                snprintf(label, sizeof(label), "principledCombinedPdf matches PrincipledBxDF::scattering_pdf (case %d)", i);
                expectNear(label, out[i], expectedPdf[i], std::max(1e-4, std::fabs(expectedPdf[i]) * 1e-2));
                snprintf(label, sizeof(label), "principledCombinedPdf is non-negative (case %d)", i);
                expectTrue(label, out[i] >= 0.0f);
            }
        }
    }
}

// normalizedFresnelF() numeric cross-check - already a standalone
// function (no extraction needed), matching (1-FrDielectric(cos,eta))/
// (c*pi), pbrt-v4's own NormalizedFresnelBxDF formula (src/shared/
// bxdfs_layered.h's own top comment). Uses the free FrDielectric<double>()
// template directly - see this file's own test_kernels.metal comment
// on why the class's own scattering_pdf() isn't the right reference
// here (it returns f*cos for this codebase's own NEE convention, not
// bare f). Also fixes a real (if harmless) code-duplication gap found
// while scoping this: shadeOrenNayar()/shadeNormalizedFresnel() both
// used to inline the exact same `cosSurface/M_PI_F` cosine-hemisphere
// pdf shadeLambertian() already has a tested, named function for
// (lambertianPdf(), section 190) - repointed both (8 call sites total)
// at it, a pure DRY fix with zero formula change, already covered by
// that existing test.
void testNormalizedFresnelF(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    std::mt19937 rng(2718);
    std::uniform_real_distribution<double> zeroOne(0.0, 1.0);
    std::uniform_real_distribution<double> etaDist(1.05, 2.5);
    std::uniform_real_distribution<double> cDist(0.05, 1.0);

    auto randomUpperHemisphereDir = [&]() -> simd::double3 {
        double z = 0.05 + zeroOne(rng) * 0.9;
        double phi = zeroOne(rng) * 2.0 * M_PI;
        double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        return simd::double3{r * std::cos(phi), r * std::sin(phi), z};
    };

    const int n = 40;
    std::vector<simd::float3> wis(n), ns(n);
    std::vector<float> etas(n), cs(n);
    std::vector<double> expected(n);
    for (int i = 0; i < n; ++i) {
        simd::double3 wi = randomUpperHemisphereDir();
        double eta = etaDist(rng);
        double c = cDist(rng);
        wis[i] = simd::float3{(float)wi.x, (float)wi.y, (float)wi.z};
        ns[i] = simd::float3{0.0f, 0.0f, 1.0f};
        etas[i] = (float)eta;
        cs[i] = (float)c;
        double cosWi = wi.z;
        double fr = FrDielectric<double>(cosWi, eta);
        double cv = std::max(c, 1e-6);
        expected[i] = (1.0 - fr) / (cv * M_PI);
    }
    id<MTLBuffer> wiBuf = makeBuffer(device, wis.data(), n * sizeof(simd::float3));
    id<MTLBuffer> nBuf = makeBuffer(device, ns.data(), n * sizeof(simd::float3));
    id<MTLBuffer> etaBuf = makeBuffer(device, etas.data(), n * sizeof(float));
    id<MTLBuffer> cBuf = makeBuffer(device, cs.data(), n * sizeof(float));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (runKernel(device, library, queue, @"test_normalizedFresnelF", @[wiBuf, nBuf, etaBuf, cBuf, outBuf], nil, n)) {
        float* out = (float*)outBuf.contents;
        for (int i = 0; i < n; ++i) {
            char label[128];
            snprintf(label, sizeof(label), "normalizedFresnelF matches (1-Fr)/(c*pi) (case %d)", i);
            expectNear(label, out[i], expected[i], std::max(1e-5, std::fabs(expected[i]) * 1e-3));
        }
    }
}

// layeredCoatedConductorF()/layeredCoatedDiffuseF()/coatedDiffuseProxyPdf()
// - property tests, not a numeric cross-check (see test_kernels.metal's
// own comment on why an exact/statistical comparison against the CPU
// reference isn't attempted for the two stochastic walk functions).
// coatedDiffuseProxyPdf() has no reference at all (this codebase's own
// heuristic MIS proxy, not a value with independent ground truth - same
// category, so tested the same way).
void testLayeredCoatedProperties(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    std::mt19937 rng(777);
    std::uniform_real_distribution<double> zeroOne(0.0, 1.0);
    std::uniform_real_distribution<double> alphaDist(0.01, 0.9);
    std::uniform_real_distribution<double> etaDist(1.05, 2.5);
    std::uniform_real_distribution<double> kDist(0.0, 4.0);

    auto randomUpperHemisphereDir = [&]() -> simd::double3 {
        double z = 0.05 + zeroOne(rng) * 0.9;
        double phi = zeroOne(rng) * 2.0 * M_PI;
        double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        return simd::double3{r * std::cos(phi), r * std::sin(phi), z};
    };
    auto isFiniteFloat3 = [](simd::float3 v) {
        return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
    };

    const int n = 30;
    std::vector<simd::float3> wiLocals(n), woLocals(n);
    std::vector<float> etas(n), alphas(n);
    std::vector<simd::float3> conductorEtas(n), conductorKs(n), albedos(n);
    std::vector<uint32_t> seedsA(n), seedsB(n);
    for (int i = 0; i < n; ++i) {
        simd::double3 wi = randomUpperHemisphereDir();
        simd::double3 wo = randomUpperHemisphereDir();
        wiLocals[i] = simd::float3{(float)wi.x, (float)wi.y, (float)wi.z};
        woLocals[i] = simd::float3{(float)wo.x, (float)wo.y, (float)wo.z};
        etas[i] = (float)etaDist(rng);
        alphas[i] = (float)alphaDist(rng);
        double k = kDist(rng);
        conductorEtas[i] = simd::float3{(float)etaDist(rng), (float)etaDist(rng), (float)etaDist(rng)};
        conductorKs[i] = simd::float3{(float)k, (float)k, (float)k};
        albedos[i] = simd::float3{(float)zeroOne(rng), (float)zeroOne(rng), (float)zeroOne(rng)};
        seedsA[i] = (uint32_t)rng();
        seedsB[i] = (uint32_t)rng();
    }

    id<MTLBuffer> wiBuf = makeBuffer(device, wiLocals.data(), n * sizeof(simd::float3));
    id<MTLBuffer> woBuf = makeBuffer(device, woLocals.data(), n * sizeof(simd::float3));
    id<MTLBuffer> etaBuf = makeBuffer(device, etas.data(), n * sizeof(float));
    id<MTLBuffer> alphaBuf = makeBuffer(device, alphas.data(), n * sizeof(float));
    id<MTLBuffer> condEtaBuf = makeBuffer(device, conductorEtas.data(), n * sizeof(simd::float3));
    id<MTLBuffer> condKBuf = makeBuffer(device, conductorKs.data(), n * sizeof(simd::float3));
    id<MTLBuffer> albedoBuf = makeBuffer(device, albedos.data(), n * sizeof(simd::float3));
    id<MTLBuffer> seedABuf = makeBuffer(device, seedsA.data(), n * sizeof(uint32_t));
    id<MTLBuffer> seedBBuf = makeBuffer(device, seedsB.data(), n * sizeof(uint32_t));
    id<MTLBuffer> ccOutABuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    id<MTLBuffer> ccOutBBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    id<MTLBuffer> cdOutABuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    id<MTLBuffer> cdOutBBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    id<MTLBuffer> proxyOutBuf = makeOutputBuffer(device, n * sizeof(float));

    bool ranCCA = runKernel(device, library, queue, @"test_layeredCoatedConductorFProperties",
                             @[wiBuf, woBuf, etaBuf, alphaBuf, condEtaBuf, condKBuf, seedABuf, ccOutABuf], nil, n);
    bool ranCCB = runKernel(device, library, queue, @"test_layeredCoatedConductorFProperties",
                             @[wiBuf, woBuf, etaBuf, alphaBuf, condEtaBuf, condKBuf, seedBBuf, ccOutBBuf], nil, n);
    bool ranCDA = runKernel(device, library, queue, @"test_layeredCoatedDiffuseFProperties",
                             @[wiBuf, woBuf, etaBuf, alphaBuf, albedoBuf, seedABuf, cdOutABuf], nil, n);
    bool ranCDB = runKernel(device, library, queue, @"test_layeredCoatedDiffuseFProperties",
                             @[wiBuf, woBuf, etaBuf, alphaBuf, albedoBuf, seedBBuf, cdOutBBuf], nil, n);
    bool ranProxy = runKernel(device, library, queue, @"test_coatedDiffuseProxyPdf",
                               @[woBuf, wiBuf, alphaBuf, proxyOutBuf], nil, n);

    int ccDifferingCount = 0, cdDifferingCount = 0;
    if (ranCCA && ranCCB) {
        simd::float3* outA = (simd::float3*)ccOutABuf.contents;
        simd::float3* outB = (simd::float3*)ccOutBBuf.contents;
        for (int i = 0; i < n; ++i) {
            char label[128];
            snprintf(label, sizeof(label), "layeredCoatedConductorF is finite (case %d, seed A)", i);
            expectTrue(label, isFiniteFloat3(outA[i]));
            snprintf(label, sizeof(label), "layeredCoatedConductorF is non-negative (case %d, seed A)", i);
            expectTrue(label, outA[i].x >= -1e-6f && outA[i].y >= -1e-6f && outA[i].z >= -1e-6f);
            if (simd::any(outA[i] != outB[i])) ++ccDifferingCount;
        }
        // Not every case needs to differ (some walks legitimately hit
        // the same deterministic early-out, e.g. a grazing wi/wo pair),
        // but MOST should, across 30 independently-seeded cases - this
        // is the "genuinely sampling, not silently constant" check.
        expectTrue("layeredCoatedConductorF varies across at least half of seeded pairs",
                   ccDifferingCount >= n / 2);
    }
    if (ranCDA && ranCDB) {
        simd::float3* outA = (simd::float3*)cdOutABuf.contents;
        simd::float3* outB = (simd::float3*)cdOutBBuf.contents;
        for (int i = 0; i < n; ++i) {
            char label[128];
            snprintf(label, sizeof(label), "layeredCoatedDiffuseF is finite (case %d, seed A)", i);
            expectTrue(label, isFiniteFloat3(outA[i]));
            snprintf(label, sizeof(label), "layeredCoatedDiffuseF is non-negative (case %d, seed A)", i);
            expectTrue(label, outA[i].x >= -1e-6f && outA[i].y >= -1e-6f && outA[i].z >= -1e-6f);
            if (simd::any(outA[i] != outB[i])) ++cdDifferingCount;
        }
        expectTrue("layeredCoatedDiffuseF varies across at least half of seeded pairs",
                   cdDifferingCount >= n / 2);
    }
    if (ranProxy) {
        float* out = (float*)proxyOutBuf.contents;
        for (int i = 0; i < n; ++i) {
            char label[128];
            snprintf(label, sizeof(label), "coatedDiffuseProxyPdf is finite and non-negative (case %d)", i);
            expectTrue(label, std::isfinite(out[i]) && out[i] >= -1e-6f);
        }
    }
}

// cauchyEta() numeric cross-check - the one genuinely new, previously-
// untested formula in the Mirror/Clearcoat/Dispersive material group
// (see this PR's own scoping note: shadeMirror()/shadeClearcoat() fully
// decompose into already-tested primitives - fresnelSchlickConductor(),
// frDielectric(), lambertianPdf()'s own cosine-pdf shape - and
// shadeRoughDielectric()'s own continuous pdf was already deferred in
// section 191, so shadeDispersiveRoughDielectric() inherits that same
// deferral rather than needing its own). Matches src/shared/fresnel.h's
// own CauchyEta<double>() template exactly.
void testCauchyEta(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    std::mt19937 rng(161803);
    std::uniform_real_distribution<double> lambdaDist(380.0, 750.0);  // visible spectrum, nm
    std::uniform_real_distribution<double> aDist(1.3, 1.8);
    std::uniform_real_distribution<double> bDist(0.001, 0.05);

    struct { double lambda, A, B; } fixedCases[] = {
        {589.3, 1.52, 0.00420},   // crown glass at the sodium D line
        {486.1, 1.52, 0.00420},   // same glass, F line (shorter wavelength -> higher eta)
        {656.3, 1.52, 0.00420},   // same glass, C line (longer wavelength -> lower eta)
        {550.0, 1.0, 0.0},        // B=0 degenerates to a flat, wavelength-independent eta
    };
    std::vector<simd::float3> inputs;
    std::vector<double> expected;
    for (auto& c : fixedCases) {
        inputs.push_back(simd::float3{(float)c.lambda, (float)c.A, (float)c.B});
        expected.push_back(CauchyEta<double>(c.lambda, c.A, c.B));
    }
    for (int i = 0; i < 30; ++i) {
        double lambda = lambdaDist(rng), A = aDist(rng), B = bDist(rng);
        inputs.push_back(simd::float3{(float)lambda, (float)A, (float)B});
        expected.push_back(CauchyEta<double>(lambda, A, B));
    }

    int n = (int)inputs.size();
    id<MTLBuffer> inBuf = makeBuffer(device, inputs.data(), n * sizeof(simd::float3));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (runKernel(device, library, queue, @"test_cauchyEta", @[inBuf, outBuf], nil, n)) {
        float* out = (float*)outBuf.contents;
        for (int i = 0; i < n; ++i) {
            char label[128];
            snprintf(label, sizeof(label), "cauchyEta(lambda=%.1f, A=%.4f, B=%.5f) matches CauchyEta reference",
                     inputs[i].x, inputs[i].y, inputs[i].z);
            expectNear(label, out[i], expected[i], std::max(1e-5, std::fabs(expected[i]) * 1e-4));
        }
        // Physically-real dispersion check: shorter wavelengths refract
        // MORE for any real glass (B>0) - the fixed cases above are
        // ordered F(486.1nm) > D(589.3nm) > C(656.3nm) for exactly this
        // reason. A sign error in the B/lambda^2 term wouldn't show up
        // in the raw magnitude check above if it happened to cancel at
        // one specific wavelength, but would invert this ordering.
        expectTrue("cauchyEta: shorter wavelength (F line) has higher eta than D line",
                   out[1] > out[0]);
        expectTrue("cauchyEta: D line has higher eta than longer wavelength (C line)",
                   out[0] > out[2]);
    }
}

void testGgxD(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // At normal incidence (hLocal == the shading normal), ggxD() has an
    // exact closed form regardless of alpha: D = 1/(pi*alpha^2) - see
    // that function's own definition (hr collapses to (0,0,1), lenSq
    // becomes exactly 1).
    float alphas[] = {0.05f, 0.2f, 0.5f, 0.9f};
    int n = 4;
    id<MTLBuffer> inBuf = makeBuffer(device, alphas, sizeof(alphas));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_ggxD", @[inBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    for (int i = 0; i < n; ++i) {
        double expected = 1.0 / (M_PI * alphas[i] * alphas[i]);
        char label[128];
        snprintf(label, sizeof(label), "ggxD at normal incidence, alpha=%.3f", alphas[i]);
        expectNear(label, out[i], expected, expected * 1e-3);
    }
}

void testGgxG1SmoothLimit(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // Near-zero roughness must make G1 approach 1 (no masking/shadowing
    // at all) for any direction with a positive z - the smooth-surface
    // limit every microfacet model must reduce to.
    simd::float3 dirs[] = {
        simd::normalize(simd::float3{0.0f, 0.0f, 1.0f}),
        simd::normalize(simd::float3{0.3f, 0.4f, 0.8f}),
        simd::normalize(simd::float3{0.6f, -0.5f, 0.5f}),
    };
    int n = 3;
    id<MTLBuffer> inBuf = makeBuffer(device, dirs, sizeof(dirs));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_ggxG1_smoothLimit", @[inBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    for (int i = 0; i < n; ++i) {
        char label[128];
        snprintf(label, sizeof(label), "ggxG1 approaches 1 at near-zero roughness, direction %d", i);
        expectNear(label, out[i], 1.0, 0.01);
    }
}

void testCheckerColor(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // scale=2 over uv in [0,1]: tile = floor(uv*2), parity = (tile.x+tile.y) mod 2.
    // (0.25,0.25) -> tile(0,0) parity 0 -> colorA=(1,0,0)
    // (0.75,0.25) -> tile(1,0) parity 1 -> colorB=(0,1,0)
    // (0.25,0.75) -> tile(0,1) parity 1 -> colorB
    // (0.75,0.75) -> tile(1,1) parity 0 -> colorA
    simd::float2 uvs[] = {{0.25f, 0.25f}, {0.75f, 0.25f}, {0.25f, 0.75f}, {0.75f, 0.75f}};
    bool expectA[] = {true, false, false, true};
    int n = 4;
    id<MTLBuffer> inBuf = makeBuffer(device, uvs, sizeof(uvs));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    if (!runKernel(device, library, queue, @"test_checkerColor", @[inBuf, outBuf], nil, n)) return;
    simd::float3* out = (simd::float3*)outBuf.contents;
    for (int i = 0; i < n; ++i) {
        char label[128];
        snprintf(label, sizeof(label), "checkerColor uv=(%.2f,%.2f) picks the expected tile", uvs[i].x, uvs[i].y);
        simd::float3 expected = expectA[i] ? simd::float3{1, 0, 0} : simd::float3{0, 1, 0};
        expectNear(label, simd::length(out[i] - expected), 0.0, 1e-5);
    }
}

void testFresnelSchlickConductor(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    float cosThetas[2] = {1.0f, 0.0f};
    simd::float3 f0s[2] = {simd::float3{0.9f, 0.6f, 0.2f}, simd::float3{0.9f, 0.6f, 0.2f}};
    int n = 2;
    id<MTLBuffer> cosBuf = makeBuffer(device, cosThetas, sizeof(cosThetas));
    id<MTLBuffer> f0Buf = makeBuffer(device, f0s, sizeof(f0s));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    if (!runKernel(device, library, queue, @"test_fresnelSchlickConductor", @[cosBuf, f0Buf, outBuf], nil, n)) return;
    simd::float3* out = (simd::float3*)outBuf.contents;
    // Normal incidence: F(1, F0) == F0 exactly (t = (1-1)^5 = 0).
    expectNear("fresnelSchlickConductor at normal incidence equals F0 exactly",
               simd::length(out[0] - f0s[0]), 0.0, 1e-5);
    // Grazing incidence: F(0, F0) == (1,1,1) exactly (t = (1-0)^5 = 1), regardless of F0.
    expectNear("fresnelSchlickConductor at grazing incidence is exactly white",
               simd::length(out[1] - simd::float3{1, 1, 1}), 0.0, 1e-5);
}

void testFrComplexRGB(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // Reference values from a fresh standalone double-precision C port of
    // src/shared/fresnel.h's FrComplex (frcomplex_ref.c, not reused from
    // any earlier session): real gold (Au) and copper (Cu) complex IOR at
    // normal incidence, plus a k=0 degenerate case that must reduce to
    // the ORDINARY real-valued dielectric Fresnel formula exactly
    // (R0 = ((eta-1)/(eta+1))^2 = 0.04 for eta=1.5).
    float cosThetas[3] = {1.0f, 1.0f, 1.0f};
    simd::float3 etas[3] = {
        simd::float3{0.184f, 0.457f, 1.354f},  // Au
        simd::float3{0.246f, 1.072f, 1.155f},  // Cu
        simd::float3{1.5f, 1.5f, 1.5f},        // k=0 dielectric-degenerate
    };
    simd::float3 ks[3] = {
        simd::float3{3.070f, 2.408f, 1.818f},
        simd::float3{3.378f, 2.591f, 2.469f},
        simd::float3{0.0f, 0.0f, 0.0f},
    };
    simd::float3 expected[3] = {
        simd::float3{0.932020f, 0.769230f, 0.387776f},
        simd::float3{0.924094f, 0.610411f, 0.569832f},
        simd::float3{0.04f, 0.04f, 0.04f},
    };
    int n = 3;
    id<MTLBuffer> cosBuf = makeBuffer(device, cosThetas, sizeof(cosThetas));
    id<MTLBuffer> etaBuf = makeBuffer(device, etas, sizeof(etas));
    id<MTLBuffer> kBuf = makeBuffer(device, ks, sizeof(ks));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    if (!runKernel(device, library, queue, @"test_frComplexRGB", @[cosBuf, etaBuf, kBuf, outBuf], nil, n)) return;
    simd::float3* out = (simd::float3*)outBuf.contents;
    const char* names[3] = {"frComplexRGB matches reference for gold (Au) at normal incidence",
                             "frComplexRGB matches reference for copper (Cu) at normal incidence",
                             "frComplexRGB with k=0 reduces to the real dielectric Fresnel formula"};
    for (int i = 0; i < n; ++i) {
        expectNear(names[i], simd::length(out[i] - expected[i]), 0.0, 1e-4);
    }
}

void testBuildAnisotropicOnb(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // Orthonormality check (tangent/bitangent/normal mutually
    // perpendicular, all unit length) at a spread of directions
    // INCLUDING both exact poles (0,0,+-1) and directions straddling the
    // OLD construction's own 0.999-dot-product discontinuity threshold
    // (see buildAnisotropicOnb's own comment) - the new construction has
    // no singularity anywhere, so every one of these must pass, not just
    // the "easy" arbitrary directions a pre-fix test might have picked.
    simd::float3 normals[6] = {
        simd::normalize(simd::float3{0.0f, 1.0f, 0.0f}),
        simd::normalize(simd::float3{0.0f, -1.0f, 0.0f}),
        simd::normalize(simd::float3{0.0f, 0.0f, 1.0f}),
        simd::normalize(simd::float3{0.02f, 0.9998f, 0.0f}),   // just inside the old 0.999 threshold
        simd::normalize(simd::float3{0.05f, 0.9990f, 0.0f}),   // just outside it
        simd::normalize(simd::float3{0.3f, 0.5f, 0.8124f}),
    };
    int n = 6;
    id<MTLBuffer> normalBuf = makeBuffer(device, normals, sizeof(normals));
    id<MTLBuffer> tangentBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    id<MTLBuffer> bitangentBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    if (!runKernel(device, library, queue, @"test_buildAnisotropicOnb",
                   @[normalBuf, tangentBuf, bitangentBuf], nil, n)) return;
    simd::float3* tangents = (simd::float3*)tangentBuf.contents;
    simd::float3* bitangents = (simd::float3*)bitangentBuf.contents;
    for (int i = 0; i < n; ++i) {
        char label[160];
        simd::float3 t = tangents[i], b = bitangents[i], nrm = normals[i];
        snprintf(label, sizeof(label), "buildAnisotropicOnb tangent is unit length (case %d)", i);
        expectNear(label, simd::length(t), 1.0, 1e-4);
        snprintf(label, sizeof(label), "buildAnisotropicOnb bitangent is unit length (case %d)", i);
        expectNear(label, simd::length(b), 1.0, 1e-4);
        snprintf(label, sizeof(label), "buildAnisotropicOnb tangent perpendicular to normal (case %d)", i);
        expectNear(label, simd::dot(t, nrm), 0.0, 1e-4);
        snprintf(label, sizeof(label), "buildAnisotropicOnb bitangent perpendicular to normal (case %d)", i);
        expectNear(label, simd::dot(b, nrm), 0.0, 1e-4);
        snprintf(label, sizeof(label), "buildAnisotropicOnb tangent perpendicular to bitangent (case %d)", i);
        expectNear(label, simd::dot(t, b), 0.0, 1e-4);
    }
}

void testSampleGGXEnergyTableDevice(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // Builds the SAME table (via buildGGXEnergyTable(), the exact
    // function metal_poc.mm's own render path calls at startup) and
    // checks the device-side bilinear lookup reproduces the host-side
    // one (sampleGGXEnergyTable()) at a spread of (roughness, mu) pairs -
    // both read the identical uploaded array, so any drift here means
    // the two interpolation implementations have diverged, not that the
    // table itself is wrong (metal_poc_math_tests.cpp's own
    // testGGXEnergyTableTrend() already checks the table's own values
    // against the real Kulla-Conty physical trend).
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);
    auto randFn = [&]() { return unitDist(rng); };
    GGXEnergyTable table;
    buildGGXEnergyTable(8, 8, 512, table, randFn);

    id<MTLBuffer> tableBuf = makeBuffer(device, table.E.data(), table.E.size() * sizeof(float));
    simd::uint2 dims[1] = {{(uint32_t)table.roughRes, (uint32_t)table.muRes}};
    id<MTLBuffer> dimsBuf = makeBuffer(device, dims, sizeof(dims));

    const int n = 20;
    simd::float2 pairs[n];
    for (int i = 0; i < n; ++i) {
        float roughness = ((float)i + 0.5f) / n;
        float mu = fmodf(roughness * 53.0f + 0.31f, 1.0f);
        pairs[i] = {roughness, mu};
    }
    id<MTLBuffer> pairsBuf = makeBuffer(device, pairs, sizeof(pairs));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_sampleGGXEnergyTableDevice",
                   @[tableBuf, dimsBuf, pairsBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    for (int i = 0; i < n; ++i) {
        float hostValue = sampleGGXEnergyTable(table, pairs[i].x, pairs[i].y);
        char label[128];
        snprintf(label, sizeof(label), "sampleGGXEnergyTableDevice matches host lookup (roughness=%.3f, mu=%.3f)",
                 pairs[i].x, pairs[i].y);
        expectNear(label, out[i], hostValue, 1e-4);
    }
}

void testOrenNayarF(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // Reference values from a fresh standalone double-precision C port
    // of Blender Cycles' own bsdf_oren_nayar.h (its single-scatter term
    // only - the multiscatter energy-compensation refinement is
    // deliberately not ported, see orenNayarF's own comment), not reused
    // from any earlier session: (1) sigma == 0 must reduce EXACTLY to
    // plain Lambertian's own 1/pi; (2) a grazing, azimuth-aligned view/
    // light pair at high roughness must read substantially BRIGHTER than
    // Lambertian (the real, well-known retroreflective "flat moon"
    // effect this whole material exists to capture) - checked against
    // the reference program's own computed value, not just "greater
    // than," so a sign error or a wrong-by-a-constant-factor bug would
    // still be caught.
    simd::float3 wos[3] = {
        {0.3f, 0.0f, 0.9539f}, {0.9539f, 0.0f, 0.3f}, {0.9539f, 0.0f, 0.3f},
    };
    simd::float3 wis[3] = {
        {-0.2f, 0.1f, 0.9747f}, {0.9747f, 0.0f, 0.2f}, {0.9747f, 0.0f, 0.2f},
    };
    simd::float3 ns[3] = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
    float sigmas[3] = {0.0f, 0.0f, 1.0f};
    float expected[3] = {1.0f / (float)M_PI, 1.0f / (float)M_PI, 1.0132f};
    int n = 3;
    id<MTLBuffer> woBuf = makeBuffer(device, wos, sizeof(wos));
    id<MTLBuffer> wiBuf = makeBuffer(device, wis, sizeof(wis));
    id<MTLBuffer> nBuf = makeBuffer(device, ns, sizeof(ns));
    id<MTLBuffer> sigmaBuf = makeBuffer(device, sigmas, sizeof(sigmas));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_orenNayarF",
                   @[woBuf, wiBuf, nBuf, sigmaBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    const char* names[3] = {
        "orenNayarF at sigma=0 matches Lambertian's own 1/pi (case 0, near-normal)",
        "orenNayarF at sigma=0 matches Lambertian's own 1/pi (case 1, grazing)",
        "orenNayarF at sigma=1 shows the real grazing-angle retroreflective brightening",
    };
    for (int i = 0; i < n; ++i) {
        expectNear(names[i], out[i], expected[i], 1e-3);
    }
}

void testVelvetF(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // Reference values from a fresh standalone double-precision C port
    // of Blender Cycles' own bsdf_ashikhmin_velvet.h, not reused from
    // any earlier session: (1) perfectly head-on view AND light must be
    // EXACTLY zero - velvet's own signature "no highlight straight back
    // at you" behaviour, unlike every other glossy material in this
    // POC; (2) a grazing, azimuth-aligned view/light pair must show the
    // real, substantial "fuzzy rim" brightening this material exists to
    // capture, matched to the reference program's own exact value, not
    // just "greater than zero."
    simd::float3 wos[3] = {
        {0.0f, 0.0f, 1.0f}, {0.97072832f, 0.0f, 0.24018020f}, {0.96592583f, 0.0f, 0.25881905f},
    };
    simd::float3 wis[3] = {
        {0.0f, 0.0f, 1.0f}, {0.97072832f, 0.0f, 0.24018020f}, {0.96592583f, 0.0f, 0.25881905f},
    };
    simd::float3 ns[3] = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
    float sigmas[3] = {0.3f, 0.3f, 0.3f};
    float expected[3] = {0.0f, 0.24227969f, 0.23677936f};
    int n = 3;
    id<MTLBuffer> woBuf = makeBuffer(device, wos, sizeof(wos));
    id<MTLBuffer> wiBuf = makeBuffer(device, wis, sizeof(wis));
    id<MTLBuffer> nBuf = makeBuffer(device, ns, sizeof(ns));
    id<MTLBuffer> sigmaBuf = makeBuffer(device, sigmas, sizeof(sigmas));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_velvetF",
                   @[woBuf, wiBuf, nBuf, sigmaBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    const char* names[3] = {
        "velvetF is exactly zero at perfectly head-on view/light",
        "velvetF matches the reference at a grazing, aligned pair (case 1)",
        "velvetF matches the reference at a grazing, aligned pair (case 2, 75deg)",
    };
    for (int i = 0; i < n; ++i) {
        expectNear(names[i], out[i], expected[i], 1e-3);
    }
}

void testAtan2Zero(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    simd::float2 inputs[9] = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f},
        {1e-3f, 1e-3f}, {1e-4f, 1e-4f}, {1e-5f, 1e-5f},
        {1e-6f, 1e-6f}, {1e-7f, 1e-7f}, {1e-8f, 1e-8f},
    };
    int n = 9;
    id<MTLBuffer> inBuf = makeBuffer(device, inputs, sizeof(inputs));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_atan2Zero", @[inBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    for (int i = 0; i < n; ++i) {
        double expected = std::atan2((double)inputs[i].y, (double)inputs[i].x);
        fprintf(stderr, "[atan2 diagnostic] atan2(%.9f,%.9f) = %.6f  (expected ~%.6f)\n",
                inputs[i].y, inputs[i].x, out[i], expected);
        // Real assertion for every case EXCEPT exact (0,0) - this used to be
        // diagnostic-only and so could never fail. (0,0) is deliberately
        // excluded: MSL's atan2(0,0) returns NaN under fastMathEnabled (a
        // documented, worked-around behavior - see docs/
        // METAL_GPU_FEASIBILITY.md's "atan2(0,0) returns NaN in MSL" bug),
        // unlike std::atan2(0,0) == 0, and asserting it would just fail on
        // known-correct behavior. Every other input, down to 1e-8, must be
        // finite and match std::atan2 - that's the regression this guards.
        if (inputs[i].x == 0.0f && inputs[i].y == 0.0f) continue;
        char label[96];
        snprintf(label, sizeof(label), "atan2 finite and matches std::atan2 (case %d)", i);
        expectTrue(label, std::isfinite(out[i]));
        expectNear(label, out[i], expected, 1e-3);
    }
}
