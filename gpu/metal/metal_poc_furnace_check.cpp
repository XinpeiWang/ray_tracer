// metal_poc_furnace_check.cpp - host-only checker for the metal_poc_cloud_furnace CTest (see CMakeLists.txt / metal_poc_furnace_test.sh).
// pbrt_scenes/cloud-furnace.pbrt is a non-absorbing cloud alone under a uniform sky: energy conservation says the cloud's box must look
// exactly like the sky around it. Compares the mean of two windows inside the box (at 300x300 the box spans about x 104-190, y 105-192)
// with the mean of a window of plain sky, per channel, and fails when they differ by more than the tolerance (a percentage).
//
// usage: metal_poc_furnace_check <render.png> <tolerance-percent>
#define STB_IMAGE_IMPLEMENTATION
#include "../../src/external/stb_image.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

static void windowMean(const unsigned char* px, int w, int x0, int y0, int x1, int y1, double out[3]) {
    out[0] = out[1] = out[2] = 0.0;
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x)
            for (int c = 0; c < 3; ++c) out[c] += px[(y * w + x) * 3 + c];
    const double n = (double)(x1 - x0) * (y1 - y0);
    for (int c = 0; c < 3; ++c) out[c] /= n;
}

int main(int argc, char** argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s render.png tolerance-percent\n", argv[0]); return 2; }
    int w = 0, h = 0, n = 0;
    unsigned char* px = stbi_load(argv[1], &w, &h, &n, 3);
    if (!px || w != 300 || h != 300) { fprintf(stderr, "FAIL: could not load a 300x300 image from %s\n", argv[1]); return 1; }
    const double tol = atof(argv[2]) / 100.0;
    double sky[3];
    windowMean(px, w, 10, 10, 90, 90, sky);
    struct Win { const char* what; int x0, y0, x1, y1; };
    const Win wins[] = {{"top of the box", 110, 108, 185, 130}, {"middle of the box", 110, 130, 185, 160}, {"bottom of the box", 110, 160, 185, 188}};
    int bad = 0;
    for (const Win& win : wins) {
        double m[3];
        windowMean(px, w, win.x0, win.y0, win.x1, win.y1, m);
        double worst = 0.0;
        for (int c = 0; c < 3; ++c) worst = std::fmax(worst, std::fabs(m[c] - sky[c]) / sky[c]);
        const bool ok = worst <= tol;
        printf("%-18s mean (%6.1f,%6.1f,%6.1f) vs sky (%6.1f,%6.1f,%6.1f): off by %.2f%%  %s\n", win.what, m[0], m[1], m[2], sky[0], sky[1], sky[2], worst * 100.0, ok ? "ok" : "TOO FAR");
        if (!ok) ++bad;
    }
    stbi_image_free(px);
    return bad ? 1 : 0;
}
