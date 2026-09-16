// metal_poc_validate.cpp
// A minimal smoke-test validator for metal_poc's own rendered PNG output.
// This POC has grown to seventeen incremental features (docs/METAL_GPU_
// FEASIBILITY.md sections 8-26) with zero automated regression coverage
// the whole way - every increment was verified by rendering, looking at
// the image, and reasoning about it by hand, which catches real bugs
// (see section 20's own account of finding two of them) but leaves
// nothing running the NEXT time a shader edit silently breaks something
// that isn't the specific thing being changed. This is deliberately NOT
// full CPU/GPU parity testing (the "some parity coverage... reusing the
// existing MaterialCpuGpuParityTest pattern" section 5 point 3 of the
// feasibility doc calls for is real, standalone future work) - just the
// cheapest check that catches the two most common "something is
// seriously broken" failure modes for a renderer: a crashed/never-
// dispatched kernel (leaves the output texture at its cleared/zero
// state - an all-black image) and a shading bug that collapses every
// pixel to the same value (a flat, zero-variance image). Neither
// check knows anything about this specific scene's own expected
// colours, so it stays valid across every future scene change.
//
// Standalone from metal_poc.mm on purpose (a separate executable, not a
// --validate flag added to metal_poc itself) - keeps metal_poc's own
// Metal/rendering code untouched by this, and lets this run as its own
// separate CTest step with metal_poc's rendered output as its input.

#define STB_IMAGE_IMPLEMENTATION
#include "../../src/external/stb_image.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <png_path> <expected_width> <expected_height>\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];
    int expectedWidth = atoi(argv[2]);
    int expectedHeight = atoi(argv[3]);

    int width, height, channels;
    unsigned char* pixels = stbi_load(path, &width, &height, &channels, 3);
    if (!pixels) {
        fprintf(stderr, "FAIL: could not load %s\n", path);
        return 1;
    }
    if (width != expectedWidth || height != expectedHeight) {
        fprintf(stderr, "FAIL: dimensions %dx%d != expected %dx%d\n", width, height, expectedWidth, expectedHeight);
        stbi_image_free(pixels);
        return 1;
    }

    long pixelCount = (long)width * height * 3;
    double sum = 0.0;
    for (long i = 0; i < pixelCount; ++i) sum += pixels[i];
    double mean = sum / (double)pixelCount;

    double sqSum = 0.0;
    for (long i = 0; i < pixelCount; ++i) {
        double d = (double)pixels[i] - mean;
        sqSum += d * d;
    }
    double stddev = sqrt(sqSum / (double)pixelCount);
    stbi_image_free(pixels);

    fprintf(stderr, "%s: %dx%d, mean=%.2f, stddev=%.2f\n", path, width, height, mean, stddev);

    // Thresholds are deliberately loose (real scene content is nowhere
    // close to either boundary - this scene's own renders run well over
    // mean 30 and stddev 40) - the point is catching total failure, not
    // asserting anything about visual quality.
    if (mean < 2.0) {
        fprintf(stderr, "FAIL: mean pixel value %.2f is near-zero - render looks all-black\n", mean);
        return 1;
    }
    if (stddev < 2.0) {
        fprintf(stderr, "FAIL: pixel stddev %.2f is near-zero - render looks like a flat single colour\n", stddev);
        return 1;
    }

    fprintf(stderr, "PASS\n");
    return 0;
}
