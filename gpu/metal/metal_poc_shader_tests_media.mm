// metal_poc_shader_tests_media.mm
// Participating-medium/volumetric and phase-function checks (Perlin
// noise, cloud/RGB-grid density, Henyey-Greenstein phase) - one of 4
// category files split out of metal_poc_shader_tests.mm (see metal_poc_
// shader_tests_common.h's own header comment for the full "why"). Pure
// code motion, no behaviour change - every function below moved verbatim
// except losing `static` (real external linkage now needed, main() calls
// these from a different translation unit).
#include "metal_poc_shader_tests_common.h"

// perlinNoise3D() numeric cross-check against src/shared/noise.h's own
// perlin_noise<double>() - the SAME fixed pbrt-v4 permutation table and
// formula, confirmed by reading both (not assumed identical just
// because the header comment says so).
void testPerlinNoise3D(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    std::mt19937 rng(90210);
    std::uniform_real_distribution<double> coordDist(-20.0, 20.0);

    const int n = 40;
    std::vector<simd::float3> points(n);
    std::vector<double> expected(n);
    for (int i = 0; i < n; ++i) {
        double x = coordDist(rng), y = coordDist(rng), z = coordDist(rng);
        points[i] = simd::float3{(float)x, (float)y, (float)z};
        expected[i] = perlin_noise<double>(x, y, z);
    }
    id<MTLBuffer> ptBuf = makeBuffer(device, points.data(), n * sizeof(simd::float3));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (runKernel(device, library, queue, @"test_perlinNoise3D", @[ptBuf, outBuf], nil, n)) {
        float* out = (float*)outBuf.contents;
        for (int i = 0; i < n; ++i) {
            char label[128];
            snprintf(label, sizeof(label), "perlinNoise3D matches perlin_noise<double> (case %d)", i);
            expectNear(label, out[i], expected[i], std::max(1e-4, std::fabs(expected[i]) * 1e-3));
        }
    }
}

// gpuCloudDensity() numeric cross-check against
// CloudMedium<double>::compute_density() called with wispiness=0 - see
// test_kernels.metal's own comment on why wispiness=0 is the right,
// honest comparison (gpuCloudDensity() is a deliberate GPU-only port
// that never implements the CPU reference's own wispiness perturbation
// at all, not a gap this test should paper over by faking one in).
void testGpuCloudDensity(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    std::mt19937 rng(31337);
    std::uniform_real_distribution<double> zeroOne(0.0, 1.0);
    std::uniform_real_distribution<double> freqDist(0.5, 4.0);
    std::uniform_real_distribution<double> densityDist(0.5, 3.0);

    const int n = 30;
    std::vector<GpuCloudMediumGPU> clouds(n);
    std::vector<simd::float3> points(n);
    std::vector<double> expected(n);
    for (int i = 0; i < n; ++i) {
        double frequency = freqDist(rng);
        double density = densityDist(rng);
        double mx = zeroOne(rng), my = zeroOne(rng), mz = zeroOne(rng);
        clouds[i] = GpuCloudMediumGPU{};
        clouds[i].density = (float)density;
        clouds[i].frequency = (float)frequency;
        points[i] = simd::float3{(float)mx, (float)my, (float)mz};

        CloudMedium<double> ref{};
        ref.density = density;
        ref.frequency = frequency;
        ref.wispiness = 0.0;  // GPU never implements this term - see above
        expected[i] = ref.compute_density(mx, my, mz);
    }
    id<MTLBuffer> cloudBuf = makeBuffer(device, clouds.data(), n * sizeof(GpuCloudMediumGPU));
    id<MTLBuffer> ptBuf = makeBuffer(device, points.data(), n * sizeof(simd::float3));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (runKernel(device, library, queue, @"test_gpuCloudDensity", @[cloudBuf, ptBuf, outBuf], nil, n)) {
        float* out = (float*)outBuf.contents;
        for (int i = 0; i < n; ++i) {
            char label[128];
            snprintf(label, sizeof(label), "gpuCloudDensity matches CloudMedium::compute_density, wispiness=0 (case %d)", i);
            expectNear(label, out[i], expected[i], std::max(1e-4, std::fabs(expected[i]) * 1e-3));
        }
    }
}

// gpuRgbGridTrilinear() numeric cross-check against
// SampledGrid<double>::lookup(px,py,pz) (src/shared/sampled_grid.h) -
// see test_kernels.metal's own comment on why only genuinely INTERIOR
// grid points are used (a real, documented divergence exists at the
// outermost half-voxel shell: Metal clamps out-of-range voxel reads to
// the nearest edge, pbrt-v4's own SampledGrid returns a hard zero - this
// test deliberately stays inside the region where both algorithms agree
// exactly, rather than either hitting a false failure or silently
// tolerating the wrong thing at the boundary).
void testGpuRgbGridTrilinear(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    std::mt19937 rng(24601);
    std::uniform_real_distribution<double> valDist(0.0, 5.0);
    std::uniform_real_distribution<double> interiorDist(0.2, 0.8);  // stays well inside [0,1]^3

    const int nx = 8, ny = 8, nz = 8;
    std::vector<float> gridDataF(nx * ny * nz);
    std::vector<double> gridDataD(nx * ny * nz);
    for (int i = 0; i < nx * ny * nz; ++i) {
        double v = valDist(rng);
        gridDataF[i] = (float)v;
        gridDataD[i] = v;
    }
    SampledGrid<double> refGrid(gridDataD.data(), gridDataD.size(), nx, ny, nz);

    const int n = 30;
    std::vector<simd::float3> points(n);
    std::vector<double> expected(n);
    for (int i = 0; i < n; ++i) {
        double px = interiorDist(rng), py = interiorDist(rng), pz = interiorDist(rng);
        points[i] = simd::float3{(float)px, (float)py, (float)pz};
        expected[i] = refGrid.lookup(px, py, pz);
    }
    id<MTLBuffer> gridBuf = makeBuffer(device, gridDataF.data(), gridDataF.size() * sizeof(float));
    simd::int3 dims{nx, ny, nz};
    id<MTLBuffer> dimsBuf = makeBuffer(device, &dims, sizeof(simd::int3));
    id<MTLBuffer> ptBuf = makeBuffer(device, points.data(), n * sizeof(simd::float3));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (runKernel(device, library, queue, @"test_gpuRgbGridTrilinear",
                  @[gridBuf, dimsBuf, ptBuf, outBuf], nil, n)) {
        float* out = (float*)outBuf.contents;
        for (int i = 0; i < n; ++i) {
            char label[128];
            snprintf(label, sizeof(label), "gpuRgbGridTrilinear matches SampledGrid::lookup, interior point (case %d)", i);
            expectNear(label, out[i], expected[i], std::max(1e-3, std::fabs(expected[i]) * 1e-3));
        }
    }
}

// sampleHenyeyGreenstein() - property test, not a numeric cross-check
// (see test_kernels.metal's own comment: same stochastic-estimator-on-a-
// different-RNG reasoning section 193 already established for the
// layered coated materials). henyeyGreensteinPhase() itself (the
// deterministic phase VALUE) already has its own dedicated test
// (testHenyeyGreensteinPhase) - this closes the one piece of the HG
// machinery that never had any coverage: the direction sampler.
void testSampleHenyeyGreensteinProperties(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    std::mt19937 rng(555);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::uniform_real_distribution<double> gDist(-0.9, 0.9);

    auto randomUnitDir = [&]() -> simd::double3 {
        simd::double3 v;
        do {
            v = simd::double3{unit(rng), unit(rng), unit(rng)};
        } while (simd::length_squared(v) < 1e-6);
        return simd::normalize(v);
    };

    const int n = 30;
    std::vector<simd::float3> wos(n);
    std::vector<float> gs(n);
    std::vector<uint32_t> seedsA(n), seedsB(n);
    for (int i = 0; i < n; ++i) {
        simd::double3 wo = randomUnitDir();
        wos[i] = simd::float3{(float)wo.x, (float)wo.y, (float)wo.z};
        gs[i] = (float)gDist(rng);
        seedsA[i] = (uint32_t)rng();
        seedsB[i] = (uint32_t)rng();
    }
    id<MTLBuffer> woBuf = makeBuffer(device, wos.data(), n * sizeof(simd::float3));
    id<MTLBuffer> gBuf = makeBuffer(device, gs.data(), n * sizeof(float));
    id<MTLBuffer> seedABuf = makeBuffer(device, seedsA.data(), n * sizeof(uint32_t));
    id<MTLBuffer> seedBBuf = makeBuffer(device, seedsB.data(), n * sizeof(uint32_t));
    id<MTLBuffer> outABuf = makeOutputBuffer(device, n * sizeof(simd::float3));
    id<MTLBuffer> outBBuf = makeOutputBuffer(device, n * sizeof(simd::float3));

    bool ranA = runKernel(device, library, queue, @"test_sampleHenyeyGreenstein", @[woBuf, gBuf, seedABuf, outABuf], nil, n);
    bool ranB = runKernel(device, library, queue, @"test_sampleHenyeyGreenstein", @[woBuf, gBuf, seedBBuf, outBBuf], nil, n);
    if (ranA && ranB) {
        simd::float3* outA = (simd::float3*)outABuf.contents;
        simd::float3* outB = (simd::float3*)outBBuf.contents;
        int differingCount = 0;
        for (int i = 0; i < n; ++i) {
            float len = simd::length(outA[i]);
            char label[128];
            snprintf(label, sizeof(label), "sampleHenyeyGreenstein returns a unit vector (case %d)", i);
            expectNear(label, len, 1.0, 1e-3);
            snprintf(label, sizeof(label), "sampleHenyeyGreenstein is finite (case %d)", i);
            expectTrue(label, std::isfinite(outA[i].x) && std::isfinite(outA[i].y) && std::isfinite(outA[i].z));
            if (simd::any(outA[i] != outB[i])) ++differingCount;
        }
        expectTrue("sampleHenyeyGreenstein varies across at least half of seeded pairs",
                   differingCount >= n / 2);
    }
}

void testHenyeyGreensteinPhase(id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    // g == 0 (isotropic) must give the SAME value - 1/(4*pi) - for every
    // cosTheta, since an isotropic phase function has no directional
    // dependence at all.
    simd::float2 inputs[3] = {{0.3f, 0.0f}, {-0.7f, 0.0f}, {1.0f, 0.0f}};
    int n = 3;
    id<MTLBuffer> inBuf = makeBuffer(device, inputs, sizeof(inputs));
    id<MTLBuffer> outBuf = makeOutputBuffer(device, n * sizeof(float));
    if (!runKernel(device, library, queue, @"test_henyeyGreensteinPhase", @[inBuf, outBuf], nil, n)) return;
    float* out = (float*)outBuf.contents;
    double expected = 1.0 / (4.0 * M_PI);
    for (int i = 0; i < n; ++i) {
        char label[128];
        snprintf(label, sizeof(label), "henyeyGreensteinPhase(g=0) is isotropic at cosTheta=%.2f", inputs[i].x);
        expectNear(label, out[i], expected, 1e-5);
    }
}

