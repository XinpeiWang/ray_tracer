// metal_poc_nanovdb_check.cpp - host-only checker for the metal_poc_nanovdb CTest (see CMakeLists.txt / metal_poc_nanovdb_test.sh): pbrt "nanovdb" media on Metal.
// The two absolute checks of the NanoVDB medium that the CPU and OptiX renderers already pass (tests/unit/pbrt_example_scenes_tests.cpp):
//  * nanovdb-furnace.pbrt: a purely scattering NanoVDB fog sphere under a uniform white sky is invisible, so the picture reads 1.000 (within 1.5%), and so does every
//    8x8 block of it (a stripe of black pixels where rays stalled on the grid's box face once made one block read 0.4).
//  * nanovdb-thin-slab.pbrt: a thin uniform cube lit by a distant light on black has the closed-form radiance 0.1 * 0.09375 / (4 pi) = 7.46e-4 (within 3%): the density, the
//    size, the placement and the distant-light next-event estimate all have to be right for that.
//
// usage: metal_poc_nanovdb_check <furnace.exr> <slab.exr>
#include "../../src/external/tinyexr.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool load(const char* path, int& w, int& h, std::vector<float>& rgb) {
    float* rgba = nullptr;
    const char* err = nullptr;
    if (LoadEXR(&rgba, &w, &h, path, &err) != TINYEXR_SUCCESS) { if (err) FreeEXRErrorMessage(err); return false; }
    rgb.resize((size_t)w * h * 3);
    for (int i = 0; i < w * h; ++i) for (int c = 0; c < 3; ++c) rgb[(size_t)i * 3 + c] = rgba[(size_t)i * 4 + c];
    free(rgba);
    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s furnace.exr slab.exr\n", argv[0]); return 2; }
    int bad = 0;
    int w = 0, h = 0;
    std::vector<float> img;
    if (!load(argv[1], w, h, img)) { fprintf(stderr, "FAIL: cannot read %s\n", argv[1]); return 1; }
    double mean[3] = {0, 0, 0};
    for (size_t i = 0; i < img.size(); ++i) mean[i % 3] += img[i];
    for (double& m : mean) m /= (double)w * h;
    double worstBlock = 1e9;
    for (int by = 0; by + 8 <= h; by += 8)
        for (int bx = 0; bx + 8 <= w; bx += 8) {
            double s = 0;
            for (int y = by; y < by + 8; ++y) for (int x = bx; x < bx + 8; ++x) s += img[((size_t)y * w + x) * 3 + 1];
            worstBlock = std::min(worstBlock, s / 64.0);
        }
    const bool furnaceOk = std::fabs(mean[0] - 1.0) < 0.015 && std::fabs(mean[1] - 1.0) < 0.015 && std::fabs(mean[2] - 1.0) < 0.015 && worstBlock > 0.95;
    printf("furnace: mean (%.4f, %.4f, %.4f), darkest 8x8 block %.3f (want 1.000 +-1.5%%, blocks above 0.95): %s\n", mean[0], mean[1], mean[2], worstBlock, furnaceOk ? "ok" : "WRONG");
    if (!furnaceOk) ++bad;

    if (!load(argv[2], w, h, img)) { fprintf(stderr, "FAIL: cannot read %s\n", argv[2]); return 1; }
    double slab[3] = {0, 0, 0};
    for (size_t i = 0; i < img.size(); ++i) slab[i % 3] += img[i];
    for (double& m : slab) m /= (double)w * h;
    const double expected = 0.1 * 0.09375 / (4.0 * 3.14159265358979323846);
    bool slabOk = true;
    for (double m : slab) slabOk = slabOk && std::fabs(m - expected) < 0.03 * expected;
    printf("thin slab: mean (%.3e, %.3e, %.3e) vs closed form %.3e (within 3%%): %s\n", slab[0], slab[1], slab[2], expected, slabOk ? "ok" : "WRONG");
    if (!slabOk) ++bad;
    return bad ? 1 : 0;
}
