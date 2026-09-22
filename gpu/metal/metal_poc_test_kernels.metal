// ---------------------------------------------------------------------------
// Device-side unit-test kernels - see gpu/metal/metal_poc_shader_tests.mm's
// own comment for the full "why" (closing the device-side half of the
// testing gap docs/METAL_GPU_FEASIBILITY.md section 59 left open: every
// pure GPU-independent host function got a real CTest case there, but
// everything that only ever runs on the GPU - frDielectric(), the GGX
// microfacet math, checkerColor(), spotLightFalloff(),
// fresnelSchlickConductor(), henyeyGreensteinPhase(), the real
// projectionLightRadiance(), and sampleAreaLight()'s own alias-table
// lookup - had none at all, only the full-scene smoke test's own "not
// flat/black" check).
//
// Each kernel below calls exactly one already-existing function from this
// SAME file, with no reimplementation - metal_poc_shader_tests.mm
// dispatches each with known inputs and checks the outputs against
// independently-derived reference values, the same way
// metal_poc_math_tests.cpp already does for this file's host-side
// counterpart. Purely additive: nothing above this point is touched, and
// primaryRayKernel's own [[buffer(N)]] indices are irrelevant here - each
// test kernel is its own separate entry point with its own independent
// buffer(0..N) argument table, the same "intersection functions have
// their own independent argument table" reasoning
// sphereIntersectionFunction's own comment already established.
// ---------------------------------------------------------------------------

// lambertianPdf()/diffuseTransmissionPdf() (metal_poc_sampling.metal) -
// shadeLambertian()'s/shadeDiffuseTransmission()'s own real PDF formulas,
// factored out specifically so these two kernels can call them directly
// (same "test the actual production function, don't reimplement it"
// principle this file's own header comment states, and the same factoring
// hairScatteringPdfLocal() already went through for shadeHair() - see
// metal_poc_sampling.metal's own comment on both functions).
kernel void test_lambertianPdf(
    device const float* cosWis [[buffer(0)]],
    device float* outputs [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = lambertianPdf(cosWis[tid]);
}

kernel void test_diffuseTransmissionPdf(
    device const float3* inputs [[buffer(0)]],   // (cosWi, pr, pt)
    device float* outputs [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = diffuseTransmissionPdf(inputs[tid].x, inputs[tid].y, inputs[tid].z);
}

kernel void test_frDielectric(
    device const float2* inputs [[buffer(0)]],   // (cosThetaI, eta)
    device float* outputs [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = frDielectric(inputs[tid].x, inputs[tid].y);
}

kernel void test_ggxD(
    device const float* alphas [[buffer(0)]],
    device float* outputs [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    // Half-vector aligned exactly with the shading normal (hLocal ==
    // (0,0,1)) - the one case ggxD() has a known, exact closed form for:
    // D reduces to 1/(pi*alpha^2) regardless of alpha, since hr collapses
    // to (0,0,1) and lenSq becomes exactly 1.
    float alpha = alphas[tid];
    outputs[tid] = ggxD(float3(0.0, 0.0, 1.0), alpha, alpha);
}

kernel void test_ggxG1_smoothLimit(
    device const float3* directions [[buffer(0)]],
    device float* outputs [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    // A near-zero (not exactly zero, to avoid a literal divide-by-zero
    // in ggxLambda's own sqrAlphaTanN term) roughness must make G1
    // approach 1 for any direction with a positive z - a perfectly smooth
    // microfacet distribution has no masking/shadowing at all.
    const float alpha = 1e-4;
    outputs[tid] = ggxG1(directions[tid], alpha, alpha);
}

kernel void test_checkerColor(
    device const float2* uvs [[buffer(0)]],
    device float3* outputs [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = checkerColor(uvs[tid], /*scale=*/2.0, float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0));
}

kernel void test_spotLightFalloff(
    device const float3* wiFromLights [[buffer(0)]],
    device const float3* directions [[buffer(1)]],
    device const float2* angles [[buffer(2)]],   // (cosOuterAngle, cosInnerAngle)
    device float* outputs [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = spotLightFalloff(wiFromLights[tid], directions[tid], angles[tid].x, angles[tid].y);
}

kernel void test_fresnelSchlickConductor(
    device const float* cosThetas [[buffer(0)]],
    device const float3* f0s [[buffer(1)]],
    device float3* outputs [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = fresnelSchlickConductor(cosThetas[tid], f0s[tid]);
}

kernel void test_frComplexRGB(
    device const float* cosThetas [[buffer(0)]],
    device const float3* etas [[buffer(1)]],
    device const float3* ks [[buffer(2)]],
    device float3* outputs [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = frComplexRGB(cosThetas[tid], etas[tid], ks[tid]);
}

kernel void test_buildAnisotropicOnb(
    device const float3* normals [[buffer(0)]],
    device float3* tangentOutputs [[buffer(1)]],
    device float3* bitangentOutputs [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    float3 t, b;
    buildAnisotropicOnb(normals[tid], t, b);
    tangentOutputs[tid] = t;
    bitangentOutputs[tid] = b;
}

kernel void test_sampleEnvironmentDirection(
    device const float* marginalCDF [[buffer(0)]],
    device const float* conditionalCDF [[buffer(1)]],
    device const int2* dims [[buffer(2)]],
    device const float2* uvSamples [[buffer(3)]],
    device float3* dirOutputs [[buffer(4)]],
    device float* pdfOutputs [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    float pdf;
    float3 dir = sampleEnvironmentDirection(marginalCDF, conditionalCDF, dims[0].x, dims[0].y,
                                             uvSamples[tid].x, uvSamples[tid].y, pdf);
    dirOutputs[tid] = dir;
    pdfOutputs[tid] = pdf;
}

kernel void test_pdfEnvironmentDirection(
    device const float* marginalCDF [[buffer(0)]],
    device const float* conditionalCDF [[buffer(1)]],
    device const int2* dims [[buffer(2)]],
    device const float3* dirs [[buffer(3)]],
    device float* pdfOutputs [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    pdfOutputs[tid] = pdfEnvironmentDirection(marginalCDF, conditionalCDF, dims[0].x, dims[0].y, dirs[tid]);
}

kernel void test_sampleGGXEnergyTableDevice(
    device const float* E [[buffer(0)]],
    device const uint2* dims [[buffer(1)]],
    device const float2* roughnessMuPairs [[buffer(2)]],
    device float* outputs [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = sampleGGXEnergyTableDevice(E, dims[0].x, dims[0].y,
                                               roughnessMuPairs[tid].x, roughnessMuPairs[tid].y);
}

kernel void test_orenNayarF(
    device const float3* wos [[buffer(0)]],
    device const float3* wis [[buffer(1)]],
    device const float3* ns [[buffer(2)]],
    device const float* sigmas [[buffer(3)]],
    device float* outputs [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = orenNayarF(wos[tid], wis[tid], ns[tid], sigmas[tid]);
}

kernel void test_velvetF(
    device const float3* wos [[buffer(0)]],
    device const float3* wis [[buffer(1)]],
    device const float3* ns [[buffer(2)]],
    device const float* sigmas [[buffer(3)]],
    device float* outputs [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = velvetF(wos[tid], wis[tid], ns[tid], sigmas[tid]);
}

kernel void test_henyeyGreensteinPhase(
    device const float2* inputs [[buffer(0)]],   // (cosTheta, g)
    device float* outputs [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = henyeyGreensteinPhase(inputs[tid].x, inputs[tid].y);
}

// Mirrors ProjectionLight's own field layout exactly (see that struct's
// own comment) - a plain input-data buffer for this test kernel, not a
// re-declaration of the real struct.
kernel void test_projectionLightRadiance(
    device const float3* wiFromLights [[buffer(0)]],
    constant ProjectionLight& light [[buffer(1)]],
    device float3* outputs [[buffer(2)]],
    texture2d<float, access::sample> testImage [[texture(0)]],
    uint tid [[thread_position_in_grid]])
{
    // Nearest, clamp-to-edge sampling - deliberately DIFFERENT from
    // primaryRayKernel's own production sampler (bilinear, repeat), since
    // this test cares about projectionLightRadiance()'s own frustum/UV
    // MATH being right (an exact, predictable texel lookup), not
    // re-verifying Metal's own texture-sampling hardware, which isn't
    // this function's responsibility to get right or wrong.
    constexpr sampler nearestSampler(coord::normalized, address::clamp_to_edge, filter::nearest);
    outputs[tid] = projectionLightRadiance(wiFromLights[tid], light.forward, light.right, light.up,
                                            light.tanHalfFovX, light.tanHalfFovY, light.scale,
                                            testImage, nearestSampler);
}

// Dispatches sampleAreaLight() itself, many times, against a small
// host-supplied light list already run through buildPowerLightSampler()
// (metal_poc_host_math.h) - metal_poc_shader_tests.mm checks that the
// REAL device-side alias-table lookup this kernel calls empirically
// reproduces each light's own `pmf`, closing the exact gap
// docs/METAL_GPU_FEASIBILITY.md section 59 called out by name ("only
// this PR's own host-side test MIRROR of the alias-table lookup is
// directly regression-tested, not the actual device-side one it's
// mirroring").
kernel void test_sampleAreaLight_pmf(
    device const AreaLight* lights [[buffer(0)]],
    constant uint& lightCount [[buffer(1)]],
    constant uint& seed [[buffer(2)]],
    device float* outPmfs [[buffer(3)]],
    texture2d<float, access::sample> dummyTexture [[texture(0)]],
    uint tid [[thread_position_in_grid]])
{
    constexpr sampler nearestSampler(coord::normalized, address::clamp_to_edge, filter::nearest);
    uint rngState = tid * 9781u + seed * 26699u + 1u;
    LightSample ls = sampleAreaLight(lights, lightCount, rngState, dummyTexture, nearestSampler);
    outPmfs[tid] = ls.pmf;
}

// Added alongside PR #57's own GoniometricLight/equalAreaSphereToSquare()
// - neither had any device-side test coverage at all until now, the
// exact gap this whole "Device-side unit-test kernels" section exists to
// close for every OTHER function already here.
kernel void test_equalAreaSphereToSquare(
    device const float3* directions [[buffer(0)]],
    device float2* outputs [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = equalAreaSphereToSquare(directions[tid]);
}

// HairBxDF numeric cross-check kernels (B11 re-attempt, section 183 -
// see docs/METAL_GPU_FEASIBILITY.md) - each dispatches exactly one
// function from metal_poc_materials_hair.metal against a fixed input
// buffer, so metal_poc_shader_tests.mm can diff the result against the
// SAME src/shared/bxdfs_hair.h/bxdfs_principled.h template (T=double)
// this file is a hand transcription of, term-by-term rather than only
// via a full-scene render - the methodology the first B11 attempt didn't
// use and got stuck without.
kernel void test_atan2Zero(
    device const float2* inputs [[buffer(0)]],
    device float* outputs [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = atan2(inputs[tid].y, inputs[tid].x);
}

kernel void test_hairMp(
    device const float* cosThetaIs [[buffer(0)]],
    device const float* cosThetaOs [[buffer(1)]],
    device const float* sinThetaIs [[buffer(2)]],
    device const float* sinThetaOs [[buffer(3)]],
    device const float* vs [[buffer(4)]],
    device float* outputs [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = hairMp(cosThetaIs[tid], cosThetaOs[tid], sinThetaIs[tid], sinThetaOs[tid], vs[tid]);
}

kernel void test_hairNp(
    device const float* phis [[buffer(0)]],
    device const int* ps [[buffer(1)]],
    device const float* ss [[buffer(2)]],
    device const float* gammaOs [[buffer(3)]],
    device const float* gammaTs [[buffer(4)]],
    device float* outputs [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = hairNp(phis[tid], ps[tid], ss[tid], gammaOs[tid], gammaTs[tid]);
}

kernel void test_hairComputeAp(
    constant HairBxDFParams* params [[buffer(0)]],
    device const float* cosThetaOs [[buffer(1)]],
    device float3* outputsR [[buffer(2)]],  // ap_r[0..3] packed as two float3 rows below
    device float3* outputsG [[buffer(3)]],
    device float3* outputsB [[buffer(4)]],
    device float* outputsRem [[buffer(5)]],  // ap[3] (r,g,b) since float3 only holds 3
    uint tid [[thread_position_in_grid]])
{
    float ap_r[4], ap_g[4], ap_b[4];
    hairComputeAp(params[tid], cosThetaOs[tid], ap_r, ap_g, ap_b);
    outputsR[tid] = float3(ap_r[0], ap_r[1], ap_r[2]);
    outputsG[tid] = float3(ap_g[0], ap_g[1], ap_g[2]);
    outputsB[tid] = float3(ap_b[0], ap_b[1], ap_b[2]);
    outputsRem[tid * 3 + 0] = ap_r[3];
    outputsRem[tid * 3 + 1] = ap_g[3];
    outputsRem[tid * 3 + 2] = ap_b[3];
}

kernel void test_hairEvalLocal(
    constant HairBxDFParams* params [[buffer(0)]],
    device const float3* wos [[buffer(1)]],
    device const float3* wis [[buffer(2)]],
    device float3* outputs [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    float fr, fg, fb;
    hairEvalLocal(params[tid], wos[tid].x, wos[tid].y, wos[tid].z,
                   wis[tid].x, wis[tid].y, wis[tid].z, fr, fg, fb);
    outputs[tid] = float3(fr, fg, fb);
}

kernel void test_hairScatteringPdfLocal(
    constant HairBxDFParams* params [[buffer(0)]],
    device const float3* wos [[buffer(1)]],
    device const float3* wis [[buffer(2)]],
    device float* outputs [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = hairScatteringPdfLocal(params[tid], wos[tid].x, wos[tid].y, wos[tid].z,
                                           wis[tid].x, wis[tid].y, wis[tid].z);
}

kernel void test_hairSample(
    constant HairBxDFParams* params [[buffer(0)]],
    device const float3* tangents [[buffer(1)]],
    device const float3* wis [[buffer(2)]],
    device const float4* us [[buffer(3)]],  // (u1,u2,u3,u4)
    device float3* outWo [[buffer(4)]],
    device float3* outRgb [[buffer(5)]],
    device int* outValid [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    BxDFSampleResultGPU res = hairSample(params[tid], tangents[tid].x, tangents[tid].y, tangents[tid].z,
                                          wis[tid].x, wis[tid].y, wis[tid].z,
                                          us[tid].x, us[tid].y, us[tid].z, us[tid].w);
    outWo[tid] = float3(res.wo_x, res.wo_y, res.wo_z);
    outRgb[tid] = float3(res.r, res.g, res.b);
    outValid[tid] = res.valid ? 1 : 0;
}

kernel void test_goniometricLightRadiance(
    device const float3* wiFromLights [[buffer(0)]],
    constant GoniometricLight& light [[buffer(1)]],
    device float3* outputs [[buffer(2)]],
    texture2d<float, access::sample> testImage [[texture(0)]],
    uint tid [[thread_position_in_grid]])
{
    constexpr sampler nearestSampler(coord::normalized, address::clamp_to_edge, filter::nearest);
    outputs[tid] = goniometricLightRadiance(wiFromLights[tid], light.forward, light.right, light.up,
                                             light.emission, light.scale, testImage, nearestSampler);
}
