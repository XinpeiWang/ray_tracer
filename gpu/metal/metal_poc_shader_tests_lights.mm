// metal_poc_shader_tests_lights.mm
// Light-type-specific checks (spot, environment-direction sampling,
// projection, area-light alias table, equal-area mapping, goniometric) -
// one of 4 category files split out of metal_poc_shader_tests.mm (see
// metal_poc_shader_tests_common.h's own header comment for the full
// "why"). Pure code motion, no behaviour change - every function below
// moved verbatim except losing `static` (real external linkage now
// needed, main() calls these from a different translation unit).
#include "metal_poc_shader_tests_common.h"

void testSpotLightFalloff(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    simd::float3 wis[3] = {
        simd::float3{0, 0, 1},   // exactly aligned with the spot's own aim
        simd::float3{0, 0, 1},
        simd::float3{1, 0, 0},   // 90 degrees off-axis
    };
    simd::float3 dirs[3] = {
        simd::float3{0, 0, 1},
        simd::float3{0, 0, 1},
        simd::float3{0, 0, 1},
    };
    simd::float2 angles[3] = {
        {-1.0f, -1.0f},          // omnidirectional flag - always full weight regardless of direction
        {0.5f, 0.9f},            // aligned (cosAngle=1) is inside the inner cone -> full weight
        {0.5f, 0.9f},            // 90 degrees (cosAngle=0) is outside the outer cone -> zero
    };
    float expected[3] = {1.0f, 1.0f, 0.0f};
    int n = 3;
    id<MTLBuffer> wiBuf = makeBuffer(device, wis, sizeof(wis));
    id<MTLBuffer> dirBuf = makeBuffer(device, dirs, sizeof(dirs));
    id<MTLBuffer> angleBuf = makeBuffer(device, angles, sizeof(angles));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_spotLightFalloff", @[wiBuf, dirBuf, angleBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    const char* names[3] = {"omnidirectional flag ignores direction", "aligned direction inside inner cone",
                             "90-degree direction outside outer cone"};
    for (int i = 0; i < n; ++i) {
        expectNear(names[i], out[i], expected[i], 1e-5);
    }
}

void testEnvironmentDirectionSampling(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // Same synthetic "bright block against a black background" image
    // metal_poc_math_tests.cpp's own testEnvDistribution2DConcentrates
    // OnBrightRegion() uses (section 69, phase 1) - reused here via
    // buildEnvDistribution2D() itself (the exact same host-side function
    // metal_poc.mm's own render path calls to build the real CDF arrays
    // this test then uploads) rather than hand-deriving a device-side-
    // only fixture, so this test exercises the SAME CDF layout the real
    // render path produces, not a hand-rolled approximation of it.
    const int width = 32, height = 16;
    std::vector<unsigned char> rgba((size_t)width * height * 4, 0);
    for (int y = 8; y < 12; ++y) {
        for (int x = 16; x < 24; ++x) {
            unsigned char* px = &rgba[((size_t)y * width + x) * 4];
            px[0] = px[1] = px[2] = 255;
            px[3] = 255;
        }
    }
    EnvDistribution2D dist;
    buildEnvDistribution2D(rgba.data(), width, height, dist);

    id<MTLBuffer> marginalBuf = makeBuffer(device, dist.marginalCDF.data(), dist.marginalCDF.size() * sizeof(float));
    id<MTLBuffer> conditionalBuf = makeBuffer(device, dist.conditionalCDF.data(), dist.conditionalCDF.size() * sizeof(float));
    simd::int2 dims[1] = {{width, height}};
    id<MTLBuffer> dimsBuf = makeBuffer(device, dims, sizeof(dims));

    const int n = 64;
    simd::float2 uvSamples[n];
    for (int i = 0; i < n; ++i) {
        float u1 = ((float)i + 0.5f) / n;
        uvSamples[i] = {u1, fmodf(u1 * 71.0f + 0.19f, 1.0f)};
    }
    id<MTLBuffer> uvBuf = makeBuffer(device, uvSamples, sizeof(uvSamples));
    id<MTLBuffer> dirOutBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    id<MTLBuffer> pdfOutBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_sampleEnvironmentDirection",
                   @[marginalBuf, conditionalBuf, dimsBuf, uvBuf, dirOutBuf, pdfOutBuf], nil, n)) return;
    simd::float3* dirs = (simd::float3*)dirOutBuf.contents;
    float* samplePdfs = (float*)pdfOutBuf.contents;

    // Every sampled direction must be unit length and every pdf strictly
    // positive/finite - the same basic sanity check the host-side
    // sampler's own test already established, now for the device-side
    // port of it.
    for (int i = 0; i < n; ++i) {
        char label[96];
        snprintf(label, sizeof(label), "sampleEnvironmentDirection returns a unit direction (case %d)", i);
        expectNear(label, simd::length(dirs[i]), 1.0, 1e-3);
        expectTrue("sampleEnvironmentDirection pdf is positive and finite",
                   samplePdfs[i] > 0.0f && std::isfinite(samplePdfs[i]));
    }

    // Self-consistency (mirrors metal_poc_math_tests.cpp's own
    // testEnvDistribution2DPdfMatchesSample()): re-evaluating
    // pdfEnvironmentDirection() at each sample's own returned direction
    // must reproduce that SAME sample's own pdf - the sampling path and
    // the evaluation path read the same underlying CDF arithmetic (plus
    // the same equirectangular Jacobian), so any drift between them
    // would show up here.
    id<MTLBuffer> dirInBuf = makeBuffer(device, dirs, n * sizeof(simd::float3));
    id<MTLBuffer> evalPdfOutBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_pdfEnvironmentDirection",
                   @[marginalBuf, conditionalBuf, dimsBuf, dirInBuf, evalPdfOutBuf], nil, n)) return;
    float* evalPdfs = (float*)evalPdfOutBuf.contents;
    for (int i = 0; i < n; ++i) {
        char label[96];
        snprintf(label, sizeof(label), "pdfEnvironmentDirection matches its own sample's pdf (case %d)", i);
        expectNear(label, evalPdfs[i], samplePdfs[i], 1e-2);
    }
}

void testProjectionLightRadiance(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // A tiny 4x4 test image, every texel a distinct flat colour except
    // one marked texel (index (2,2), Metal's own floor(uv*size) texel
    // for uv exactly at 0.5 - the projector's own dead-centre direction)
    // set to a bright, unmistakable red so a nearest-filtered sample at
    // the frustum's own centre can be checked exactly, not just
    // "produced SOME nonzero colour."
    const int texSize = 4;
    std::vector<uint8_t> pixels(texSize * texSize * 4, 0);
    for (int i = 0; i < texSize * texSize; ++i) pixels[i * 4 + 3] = 255; // alpha
    int centreTexel = (2 * texSize + 2) * 4;
    pixels[centreTexel + 0] = 255; // pure red at the centre texel

    MTLTextureDescriptor* texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                        width:texSize height:texSize mipmapped:NO];
    texDesc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> testImage = [device newTextureWithDescriptor:texDesc];
    [testImage replaceRegion:MTLRegionMake2D(0, 0, texSize, texSize) mipmapLevel:0
                    withBytes:pixels.data() bytesPerRow:texSize * 4];

    // A projector looking straight down -Z, 40-degree vertical FOV,
    // square aspect - forward/right/up form a trivial identity-ish
    // orthonormal frame, chosen so the expected screen-space mapping for
    // each test direction can be worked out by hand rather than needing
    // a second, independent "look-at" construction to trust.
    ProjectionLightGPU light{};
    light.position = PackedFloat3{0, 0, 0};
    light.forward = PackedFloat3{0, 0, -1};
    light.right = PackedFloat3{1, 0, 0};
    light.up = PackedFloat3{0, 1, 0};
    light.tanHalfFovX = tanf(20.0f * (float)M_PI / 180.0f);
    light.tanHalfFovY = tanf(20.0f * (float)M_PI / 180.0f);
    light.scale = 1.0f;

    simd::float3 wiFromLights[3] = {
        simd::float3{0, 0, -1},   // dead centre - must land on the marked red texel
        simd::float3{0, 0, 1},    // directly BEHIND the projector - must be exactly black
        simd::float3{1, 0, -0.01f}, // far outside the +/-20 degree frustum - must be exactly black
    };
    int n = 3;
    id<MTLBuffer> wiBuf = makeBuffer(device, wiFromLights, sizeof(wiFromLights));
    id<MTLBuffer> lightBuf = makeBuffer(device, &light, sizeof(light));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    if (!runKernel(device, library, queue, @"test_projectionLightRadiance", @[wiBuf, lightBuf, outBuf], testImage, n)) return;
    simd::float3* out = (simd::float3*)outBuf.contents;

    expectTrue("projectionLightRadiance at dead centre samples the marked red texel (R channel bright)",
               out[0].x > 0.5f);
    expectNear("projectionLightRadiance at dead centre has near-zero G", out[0].y, 0.0, 0.05);
    expectNear("projectionLightRadiance directly behind the projector is exactly black",
               simd::length(out[1]), 0.0, 1e-6);
    expectNear("projectionLightRadiance outside the frustum is exactly black",
               simd::length(out[2]), 0.0, 1e-6);
}

void testSampleAreaLightAliasTable(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // Three lights, deliberately imbalanced (1x/3x/6x power, same shape
    // metal_poc_math_tests.cpp's own testPowerLightSamplerAliasTableSampling
    // uses for the HOST-side mirror) - this is the REAL device-side
    // sampleAreaLight() that mirror stood in for.
    std::vector<AreaLightGPU> lights(3);
    float emissions[3] = {1.0f, 3.0f, 6.0f};
    for (int i = 0; i < 3; ++i) {
        lights[i] = AreaLightGPU{};
        lights[i].area = 1.0f;
        lights[i].emission = PackedFloat3{emissions[i], emissions[i], emissions[i]};
        lights[i].center = PackedFloat3{0, 0, 0};
        lights[i].edgeU = PackedFloat3{1, 0, 0};
        lights[i].edgeV = PackedFloat3{0, 1, 0};
        lights[i].normal = PackedFloat3{0, 0, 1};
    }
    buildAliasTable(lights);

    uint32_t lightCount = 3;
    uint32_t seed = 777u;
    const uint32_t kNumSamples = 200000;
    id<MTLBuffer> lightsBuf = makeBuffer(device, lights.data(), lights.size() * sizeof(AreaLightGPU));
    id<MTLBuffer> countBuf = makeBuffer(device, &lightCount, sizeof(lightCount));
    id<MTLBuffer> seedBuf = makeBuffer(device, &seed, sizeof(seed));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, kNumSamples * sizeof(float));
    // sampleAreaLight() now takes a texture (section 105's own
    // AreaLight::useTexture) - every light above has useTexture=0.0f
    // (AreaLightGPU's own default), so this dummy 1x1 texture is never
    // actually sampled, just needs to be SOME real bound resource.
    MTLTextureDescriptor* dummyTexDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                              width:1 height:1 mipmapped:NO];
    dummyTexDesc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> dummyTexture = [device newTextureWithDescriptor:dummyTexDesc];
    uint8_t dummyPixel[4] = {0, 0, 0, 255};
    [dummyTexture replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:dummyPixel bytesPerRow:4];
    if (!runKernel(device, library, queue, @"test_sampleAreaLight_pmf",
                   @[lightsBuf, countBuf, seedBuf, outBuf], dummyTexture, kNumSamples)) return;
    float* out = (float*)outBuf.contents;

    long counts[3] = {0, 0, 0};
    long unmatched = 0;
    for (uint32_t i = 0; i < kNumSamples; ++i) {
        bool matched = false;
        for (int j = 0; j < 3; ++j) {
            if (std::fabs(out[i] - lights[j].pmf) < 1e-5f) { counts[j]++; matched = true; break; }
        }
        if (!matched) ++unmatched;
    }
    expectTrue("every sampleAreaLight() pmf output matches one of the three known light pmfs",
               unmatched == 0);
    for (int j = 0; j < 3; ++j) {
        double empirical = (double)counts[j] / (double)kNumSamples;
        char label[160];
        snprintf(label, sizeof(label),
                 "device-side sampleAreaLight() empirical frequency for light %d matches its own pmf (%.4f)",
                 j, lights[j].pmf);
        // Same 0.01 absolute-tolerance reasoning
        // metal_poc_math_tests.cpp's own host-side alias-table test uses
        // at this sample count.
        expectNear(label, empirical, lights[j].pmf, 0.01);
    }
}

void testEqualAreaSphereToSquare(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // Reference values computed independently via a standalone double-
    // precision C program mirroring EqualAreaSphereToSquare's own formula
    // (src/shared/sampling_extra.h) - not copied from memory - see this
    // PR's own commit message for that program's exact output. Added
    // alongside PR #57's own GoniometricLight increment, which is the
    // first (and so far only) caller of this mapping in this file.
    // The fourth direction (0.3, 0.4, 0.866025) is already unit-length
    // (0.3^2 + 0.4^2 + 0.866025^2 = 1.0), matching the exact triple the
    // reference program used - written out explicitly rather than via
    // simd::normalize() so it's obviously the SAME input, not a
    // renormalized approximation of one.
    simd::float3 dirs[4] = {
        simd::float3{0, 0, 1},           // dead centre of the square
        simd::float3{0, 0, -1},          // opposite pole - the far corner (1,1)
        simd::float3{1, 0, 0},           // equator, +X axis
        simd::float3{0.3f, 0.4f, 0.866025f},
    };

    simd::float2 expected[4] = {
        {0.5f, 0.5f},
        {1.0f, 1.0f},
        {0.99999797f, 0.50000203f},
        {0.57497354f, 0.60803916f},
    };
    int n = 4;
    id<MTLBuffer> inBuf = makeBuffer(device, dirs, sizeof(dirs));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(simd::float2));
    if (!runKernel(device, library, queue, @"test_equalAreaSphereToSquare", @[inBuf, outBuf], nil, n)) return;
    simd::float2* out = (simd::float2*)outBuf.contents;
    const char* names[4] = {"equalAreaSphereToSquare(0,0,1) is the square's own dead centre",
                             "equalAreaSphereToSquare(0,0,-1) is the square's own far corner",
                             "equalAreaSphereToSquare(1,0,0) matches the reference program",
                             "equalAreaSphereToSquare(0.3,0.4,0.866) matches the reference program"};
    for (int i = 0; i < n; ++i) {
        expectNear(names[i], out[i].x, expected[i].x, 1e-4);
        char label[128];
        snprintf(label, sizeof(label), "%s (v component)", names[i]);
        expectNear(label, out[i].y, expected[i].y, 1e-4);
    }
}

void testGoniometricLightRadiance(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // A tiny 4x4 single-channel test image (R8Unorm - the SAME pixel
    // format the real committed scene's own goniometric texture uses,
    // see metal_poc.mm's own buildGoniometricProfileImage() call site) -
    // goniometricLightRadiance() only ever reads the R channel
    // (`image.sample(s, uv).r`), so with a MONOCHROMATIC `emission`
    // every output channel is necessarily identical - there is no
    // "mark a different channel" trick available the way
    // testProjectionLightRadiance()'s own RGB test image used. Two
    // texels instead get two DIFFERENT R intensities.
    const int texSize = 4;
    std::vector<uint8_t> pixels(texSize * texSize, 0);
    // uv=(0.5,0.5) (the light's own forward direction, see below) lands
    // on texel (2,2) with nearest filtering on a 4x4 texture
    // (floor(0.5*4)=2) - full intensity.
    pixels[2 * texSize + 2] = 255;
    // uv=(1.0,1.0) (the light's own exact backward direction) clamps to
    // the last valid texel, (3,3) (floor(1.0*4)=4, clamped to size-1=3)
    // under clamp_to_edge addressing - HALF intensity, a different
    // value from the centre texel's, not a different channel.
    pixels[3 * texSize + 3] = 128;

    MTLTextureDescriptor* texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                        width:texSize height:texSize mipmapped:NO];
    texDesc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> testImage = [device newTextureWithDescriptor:texDesc];
    [testImage replaceRegion:MTLRegionMake2D(0, 0, texSize, texSize) mipmapLevel:0
                    withBytes:pixels.data() bytesPerRow:texSize];

    // A trivial identity-ish light frame (forward=+Z, right=+X, up=+Y) -
    // the same reasoning testProjectionLightRadiance() already used for
    // its own light frame: keeps the expected UV for each test direction
    // computable by hand instead of needing a second "look-at"
    // construction to trust.
    GoniometricLightGPU light{};
    light.position = PackedFloat3{0, 0, 0};
    light.forward = PackedFloat3{0, 0, 1};
    light.right = PackedFloat3{1, 0, 0};
    light.up = PackedFloat3{0, 1, 0};
    light.emission = PackedFloat3{1, 1, 1};
    light.scale = 2.0f;

    simd::float3 wiFromLights[2] = {
        simd::float3{0, 0, 1},  // exactly the light's own forward direction - should sample the full-intensity centre texel
        simd::float3{0, 0, -1}, // exactly backward - should sample the half-intensity back texel
    };
    int n = 2;
    id<MTLBuffer> wiBuf = makeBuffer(device, wiFromLights, sizeof(wiFromLights));
    id<MTLBuffer> lightBuf = makeBuffer(device, &light, sizeof(light));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    if (!runKernel(device, library, queue, @"test_goniometricLightRadiance", @[wiBuf, lightBuf, outBuf], testImage, n)) return;
    simd::float3* out = (simd::float3*)outBuf.contents;

    // `scale=2.0`, `emission=(1,1,1)`: forward direction (full-intensity
    // texel, 255/255=1.0) should read ~2.0; backward (half-intensity,
    // 128/255~=0.502) should read ~1.004 - both checked against the
    // ACTUAL expected number (not just "some positive value"), which
    // also confirms `scale`/`emission` genuinely multiply into the
    // result rather than the raw texture sample passing through
    // unscaled.
    expectNear("goniometricLightRadiance at forward direction reads ~2.0 (full-intensity texel * scale)",
               out[0].x, 2.0, 0.05);
    expectNear("goniometricLightRadiance at backward direction reads ~1.004 (half-intensity texel * scale)",
               out[1].x, 1.004, 0.05);
    expectTrue("goniometricLightRadiance's forward and backward results are genuinely different",
               std::fabs(out[0].x - out[1].x) > 0.5f);
}
