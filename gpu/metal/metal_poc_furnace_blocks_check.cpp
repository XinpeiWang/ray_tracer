// metal_poc_furnace_blocks_check.cpp - host-only checker for the metal_poc_clipped_medium CTest (see CMakeLists.txt / metal_poc_clipped_medium_test.sh): a white-furnace
// picture must read the sky's value everywhere, not just on average. Fails when the picture's mean is off by more than <mean-tolerance-percent> or any 8x8 block of it
// (green channel) falls below <min-block> times the sky.
//
// usage: metal_poc_furnace_blocks_check <render.exr> <sky> <mean-tolerance-percent> <min-block>
#include "../../src/external/tinyexr.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc != 5) { fprintf(stderr, "usage: %s render.exr sky mean-tolerance-percent min-block\n", argv[0]); return 2; }
    float* rgba = nullptr;
    int w = 0, h = 0;
    const char* err = nullptr;
    if (LoadEXR(&rgba, &w, &h, argv[1], &err) != TINYEXR_SUCCESS) { if (err) FreeEXRErrorMessage(err); fprintf(stderr, "FAIL: cannot read %s\n", argv[1]); return 1; }
    const double sky = atof(argv[2]), tol = atof(argv[3]) / 100.0, minBlock = atof(argv[4]);
    double mean[3] = {0, 0, 0};
    for (int i = 0; i < w * h; ++i) for (int c = 0; c < 3; ++c) mean[c] += rgba[(size_t)i * 4 + c];
    for (double& m : mean) m /= (double)w * h;
    double darkest = 1e9;
    int darkX = 0, darkY = 0;
    for (int by = 0; by + 8 <= h; by += 8)
        for (int bx = 0; bx + 8 <= w; bx += 8) {
            double s = 0;
            for (int y = by; y < by + 8; ++y) for (int x = bx; x < bx + 8; ++x) s += rgba[((size_t)y * w + x) * 4 + 1];
            s /= 64.0 * sky;
            if (s < darkest) { darkest = s; darkX = bx; darkY = by; }
        }
    free(rgba);
    bool ok = darkest >= minBlock;
    for (double m : mean) ok = ok && std::fabs(m - sky) <= tol * sky;
    printf("furnace: mean (%.4f, %.4f, %.4f) vs sky %.4f (within %.1f%%), darkest 8x8 block %.3f of the sky at (%d,%d) (at least %.2f): %s\n", mean[0], mean[1], mean[2], sky, tol * 100.0,
           darkest, darkX, darkY, minBlock, ok ? "ok" : "WRONG");
    return ok ? 0 : 1;
}
