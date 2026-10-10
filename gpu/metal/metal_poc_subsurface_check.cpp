// metal_poc_subsurface_check.cpp - host-only checker for the metal_poc_subsurface CTest (see CMakeLists.txt / metal_poc_subsurface_test.sh).
// pbrt_scenes/subsurface-ball.pbrt is a Cornell box with two balls of pbrt's SubsurfaceMaterial (a tabulated BSSRDF). Rendered at 128x128 by the
// CPU renderer (512 spp), the mean 8-bit values of four windows are the numbers below; Metal's BSSRDF has to land within the tolerance of them. A renderer
// that falls back to an opaque grey diffuse (as Metal did before it had a BSSRDF), or lets the balls cast shadows, is far outside: the balls would come
// out about twice as bright. (The numbers are the CPU renderer's since the exit-normal fix: the balls read about 26% brighter than before it.)
//
// usage: metal_poc_subsurface_check <render.png> <tolerance-percent>
#define STB_IMAGE_IMPLEMENTATION
#include "../../src/external/stb_image.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s render.png tolerance-percent\n", argv[0]); return 2; }
    int w = 0, h = 0, n = 0;
    unsigned char* px = stbi_load(argv[1], &w, &h, &n, 3);
    if (!px || w != 128 || h != 128) { fprintf(stderr, "FAIL: could not load a 128x128 image from %s\n", argv[1]); return 1; }
    const double tol = atof(argv[2]) / 100.0;
    struct Win { const char* what; int x0, y0, x1, y1; double expect[3]; };
    const Win wins[] = {
        {"big ball",   32,  64,  64, 102, {76.6, 80.9, 74.4}},
        {"small ball", 78,  92,  96, 108, {55.9, 50.4, 44.8}},
        {"back wall",  38,  19,  90,  38, {98.8, 91.3, 84.1}},
        {"floor",      26, 108, 102, 124, {109.9, 108.5, 104.9}},
    };
    int bad = 0;
    for (const Win& win : wins) {
        double m[3] = {0, 0, 0};
        for (int y = win.y0; y < win.y1; ++y)
            for (int x = win.x0; x < win.x1; ++x)
                for (int c = 0; c < 3; ++c) m[c] += px[(y * w + x) * 3 + c];
        const double count = (double)(win.x1 - win.x0) * (win.y1 - win.y0);
        double worst = 0.0;
        for (int c = 0; c < 3; ++c) { m[c] /= count; worst = std::fmax(worst, std::fabs(m[c] - win.expect[c]) / win.expect[c]); }
        const bool ok = worst <= tol;
        printf("%-11s mean (%5.1f,%5.1f,%5.1f) vs CPU (%5.1f,%5.1f,%5.1f): off by %.1f%%  %s\n", win.what, m[0], m[1], m[2],
               win.expect[0], win.expect[1], win.expect[2], worst * 100.0, ok ? "ok" : "TOO FAR");
        if (!ok) ++bad;
    }
    stbi_image_free(px);
    return bad ? 1 : 0;
}
