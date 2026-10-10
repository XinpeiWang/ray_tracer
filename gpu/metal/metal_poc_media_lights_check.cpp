// metal_poc_media_lights_check.cpp - host-only checker for the metal_poc_media_lights CTest (see CMakeLists.txt / metal_poc_media_lights_test.sh): the mean of a picture
// Metal rendered must match the same scene rendered by the CPU renderer (the reference) within a tolerance. The scenes (gpu/metal/test_scenes/media-*.pbrt) are a medium
// lit by one distant light or one point light on a black background, so a renderer that does not sample that kind of light inside that kind of medium reads 0.
//
// usage: metal_poc_media_lights_check <name> <cpu.exr> <metal.exr> <tolerance-percent>
#include "../../src/external/tinyexr.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool meanOf(const char* path, double mean[3]) {
    float* rgba = nullptr;
    int w = 0, h = 0;
    const char* err = nullptr;
    if (LoadEXR(&rgba, &w, &h, path, &err) != TINYEXR_SUCCESS) { if (err) FreeEXRErrorMessage(err); return false; }
    mean[0] = mean[1] = mean[2] = 0.0;
    for (int i = 0; i < w * h; ++i) for (int c = 0; c < 3; ++c) mean[c] += rgba[(size_t)i * 4 + c];
    for (int c = 0; c < 3; ++c) mean[c] /= (double)w * h;
    free(rgba);
    return true;
}

int main(int argc, char** argv) {
    if (argc != 5) { fprintf(stderr, "usage: %s name cpu.exr metal.exr tolerance-percent\n", argv[0]); return 2; }
    double cpu[3], metal[3];
    if (!meanOf(argv[2], cpu) || !meanOf(argv[3], metal)) { fprintf(stderr, "FAIL: cannot read %s or %s\n", argv[2], argv[3]); return 1; }
    const double tol = atof(argv[4]) / 100.0;
    bool ok = true;
    double worst = 0.0;
    for (int c = 0; c < 3; ++c) {
        const double off = cpu[c] > 0 ? std::fabs(metal[c] - cpu[c]) / cpu[c] : (metal[c] > 0 ? 1.0 : 0.0);
        worst = std::fmax(worst, off);
        ok = ok && off <= tol && cpu[c] > 1e-4;   // a black reference would make the comparison empty
    }
    printf("%-22s CPU mean %.4f, Metal mean %.4f (Metal/CPU %.3f, off by %.1f%%, tolerance %.0f%%): %s\n", argv[1], cpu[1], metal[1], cpu[1] > 0 ? metal[1] / cpu[1] : 0.0,
           worst * 100.0, tol * 100.0, ok ? "ok" : "WRONG");
    return ok ? 0 : 1;
}
