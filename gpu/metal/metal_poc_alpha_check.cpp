// metal_poc_alpha_check.cpp - host-only checker for the metal_poc_alpha_cutout CTest (see CMakeLists.txt / metal_poc_alpha_test.sh).
// pbrt_scenes/alpha-cutout.pbrt is a red square in front of a green wall; the square's "texture alpha" mask (images/alpha-cutout-mask.png,
// 64x64) is transparent in a round hole in the middle and in checker-shaped notches in its corners. Rendered at 128x128 the square spans
// pixels [14, 114) in x and y, so a texel (tx, ty) of the mask lands at ((tx + .5) / 64 * 100 + 14, (ty + .5) / 64 * 100 + 14). Samples
// four texels whose alpha is known: the hole (transparent: the green wall shows), a ring texel (opaque: red), a corner notch (transparent:
// green) and a neighbouring bar (opaque: red). A renderer that ignores alpha draws all four red.
//
// usage: metal_poc_alpha_check <render.png>
#define STB_IMAGE_IMPLEMENTATION
#include "../../src/external/stb_image.h"

#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s render.png\n", argv[0]); return 2; }
    int w = 0, h = 0, n = 0;
    unsigned char* px = stbi_load(argv[1], &w, &h, &n, 3);
    if (!px || w != 128 || h != 128) { fprintf(stderr, "FAIL: could not load a 128x128 image from %s\n", argv[1]); return 1; }
    struct Probe { const char* what; int tx, ty; bool expectGreen; };
    const Probe probes[] = {{"hole", 32, 32, true}, {"ring", 50, 32, false}, {"corner notch", 4, 4, true}, {"bar next to the notch", 12, 4, false}};
    int bad = 0;
    for (const Probe& p : probes) {
        const int x = (int)((p.tx + 0.5) / 64.0 * 100.0 + 14.0), y = (int)((p.ty + 0.5) / 64.0 * 100.0 + 14.0);
        const unsigned char* c = &px[(y * w + x) * 3];
        const bool green = c[1] > c[0] + 20, red = c[0] > c[1] + 20;
        const bool ok = p.expectGreen ? green : red;
        printf("%-22s pixel (%3d,%3d) = (%3d,%3d,%3d)  expected %s: %s\n", p.what, x, y, c[0], c[1], c[2], p.expectGreen ? "green (transparent)" : "red (opaque)", ok ? "ok" : "WRONG");
        if (!ok) ++bad;
    }
    stbi_image_free(px);
    return bad ? 1 : 0;
}
