// metal_poc_crop_check.cpp - host-only checker for the metal_poc_crop_window
// CTest (see CMakeLists.txt / metal_poc_crop_test.sh, docs/METAL_GPU_FEASIBILITY.md
// section 205). Compares an uncropped Metal render against a --crop render of
// the same scene/settings and checks the two properties that define a correct
// crop window:
//   1. Outside the window the crop render is exactly black.
//   2. Well inside the window it is BIT-IDENTICAL to the uncropped render -
//      cropping must only skip work, not change any pixel it does compute
//      (the kernel reconstructs the true full-image pixel coordinate, so the
//      per-pixel RNG stream is the same either way).
// Both images go through the same post-process (chromatic aberration, vignette,
// tonemap, 7x7 bilateral denoise), which reads NEIGHBOURING pixels - so right
// at the window's edge the crop render's pixels legitimately differ (they
// blend against the black outside). `margin` pixels of slack around every
// window edge that is not also an image edge are excluded from both checks.
//
// usage: metal_poc_crop_check <full.png> <crop.png> <x0> <x1> <y0> <y1> <margin>
#define STB_IMAGE_IMPLEMENTATION
#include "../../src/external/stb_image.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc != 8) { fprintf(stderr, "usage: %s full.png crop.png x0 x1 y0 y1 margin\n", argv[0]); return 2; }
    int wF, hF, wC, hC, n;
    unsigned char* full = stbi_load(argv[1], &wF, &hF, &n, 3);
    unsigned char* crop = stbi_load(argv[2], &wC, &hC, &n, 3);
    if (!full || !crop) { fprintf(stderr, "FAIL: could not load images\n"); return 1; }
    if (wF != wC || hF != hC) { fprintf(stderr, "FAIL: size mismatch %dx%d vs %dx%d\n", wF, hF, wC, hC); return 1; }
    const int x0 = atoi(argv[3]), x1 = atoi(argv[4]), y0 = atoi(argv[5]), y1 = atoi(argv[6]), m = atoi(argv[7]);
    const int W = wF, H = hF;
    // Slack only on window edges that are real interior edges of the image.
    const int sl = (x0 > 0) ? m : 0, sr = (x1 < W) ? m : 0, st = (y0 > 0) ? m : 0, sb = (y1 < H) ? m : 0;
    long outsideChecked = 0, outsideBad = 0, insideChecked = 0, insideBad = 0, insideLit = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const unsigned char* c = &crop[(y * W + x) * 3];
            const unsigned char* f = &full[(y * W + x) * 3];
            const bool inWindow = x >= x0 && x < x1 && y >= y0 && y < y1;
            if (!inWindow) {
                // Skip the m-pixel band just outside every real window edge.
                const bool nearEdge = (x >= x0 - sl && x < x1 + sr && y >= y0 - st && y < y1 + sb);
                if (nearEdge) continue;
                ++outsideChecked;
                if (c[0] || c[1] || c[2]) ++outsideBad;
            } else {
                if (x < x0 + sl || x >= x1 - sr || y < y0 + st || y >= y1 - sb) continue;
                ++insideChecked;
                if (c[0] != f[0] || c[1] != f[1] || c[2] != f[2]) ++insideBad;
                if (c[0] || c[1] || c[2]) ++insideLit;
            }
        }
    }
    stbi_image_free(full); stbi_image_free(crop);
    printf("outside: %ld checked, %ld not black | inside: %ld checked, %ld differ from uncropped, %ld non-black\n",
           outsideChecked, outsideBad, insideChecked, insideBad, insideLit);
    if (insideChecked == 0) { fprintf(stderr, "FAIL: no inside pixels to check (window too small for margin)\n"); return 1; }
    if (insideLit == 0) { fprintf(stderr, "FAIL: window is entirely black - nothing was rendered\n"); return 1; }
    if (outsideBad) { fprintf(stderr, "FAIL: %ld pixels outside the window are not black\n", outsideBad); return 1; }
    if (insideBad) { fprintf(stderr, "FAIL: %ld pixels inside the window differ from the uncropped render\n", insideBad); return 1; }
    printf("CROP_CHECK_OK\n");
    return 0;
}
