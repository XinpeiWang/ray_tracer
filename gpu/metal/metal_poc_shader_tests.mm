// metal_poc_shader_tests.mm
// The driver for this device-side test suite: sets up the Metal device/
// shader library, calls every test function (now split across metal_poc_
// shader_tests_materials.mm/_media.mm/_lights.mm/_hair.mm - see metal_poc_
// shader_tests_common.h's own header comment for the full "why" this file
// used to be one ~2260-line file with all 36 tests inline, and what moved
// where), and reports g_failures. Pure code motion, no behaviour change -
// main() below is byte-for-byte the same body the original single file
// had, just now calling declarations from metal_poc_shader_tests_common.h
// instead of static functions defined a few hundred lines up in the same
// file.
#include "metal_poc_shader_tests_common.h"

int main() {
    @autoreleasepool {
        id<MTLDevice> device = nil;
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices.count > 0) device = devices[0];
        if (!device) {
            fprintf(stderr, "FAIL: no Metal device available\n");
            return 1;
        }
        if (!device.supportsRaytracing) {
            fprintf(stderr, "FAIL: device does not support hardware raytracing\n");
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];

        NSError* error = nil;
#ifdef RT_METAL_SHADER_DIR
        NSString* shaderDir = @(RT_METAL_SHADER_DIR);
#else
        NSString* shaderDir = [@(__FILE__) stringByDeletingLastPathComponent];
#endif
        // See metal_poc_shader_files.h's own comment - the shader source
        // is split across several files on disk but still compiled as
        // ONE concatenated string (metal_poc.mm's own loader does the
        // same thing, from the same ordered file list).
        NSMutableString* shaderSource = [NSMutableString string];
        int shaderFileCount = 0;
        const char* const* shaderFileNames = metalShaderFileNames(&shaderFileCount);
        for (int i = 0; i < shaderFileCount; ++i) {
            NSString* fragPath = [shaderDir stringByAppendingPathComponent:@(shaderFileNames[i])];
            NSString* fragSource = [NSString stringWithContentsOfFile:fragPath encoding:NSUTF8StringEncoding error:&error];
            if (!fragSource) {
                fprintf(stderr, "FAIL: could not read shader source at %s: %s\n",
                        fragPath.UTF8String, error.localizedDescription.UTF8String);
                return 1;
            }
            [shaderSource appendString:fragSource];
        }
        MTLCompileOptions* compileOpts = [MTLCompileOptions new];
        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:compileOpts error:&error];
        if (!library) {
            fprintf(stderr, "FAIL: shader compile failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        testFrDielectric(device, library, queue);
        testLambertianAndDiffuseTransmissionPdf(device, library, queue);
        testGgxConductorFAndPdf(device, library, queue);
        testPrincipledPdf(device, library, queue);
        testNormalizedFresnelF(device, library, queue);
        testLayeredCoatedProperties(device, library, queue);
        testCauchyEta(device, library, queue);
        testPerlinNoise3D(device, library, queue);
        testGpuCloudDensity(device, library, queue);
        testGpuRgbGridTrilinear(device, library, queue);
        testSampleHenyeyGreensteinProperties(device, library, queue);
        testGgxD(device, library, queue);
        testGgxG1SmoothLimit(device, library, queue);
        testCheckerColor(device, library, queue);
        testSpotLightFalloff(device, library, queue);
        testFresnelSchlickConductor(device, library, queue);
        testFrComplexRGB(device, library, queue);
        testBuildAnisotropicOnb(device, library, queue);
        testEnvironmentDirectionSampling(device, library, queue);
        testSampleGGXEnergyTableDevice(device, library, queue);
        testOrenNayarF(device, library, queue);
        testVelvetF(device, library, queue);
        testHenyeyGreensteinPhase(device, library, queue);
        testProjectionLightRadiance(device, library, queue);
        testSampleAreaLightAliasTable(device, library, queue);
        testEqualAreaSphereToSquare(device, library, queue);
        testGoniometricLightRadiance(device, library, queue);
        testHairMp(device, library, queue);
        testHairNp(device, library, queue);
        testHairEvalAndPdf(device, library, queue);
        testHairComputeAp(device, library, queue);
        testHairRandomSweep(device, library, queue);
        testHairSample(device, library, queue);
        testHairBlackFurValidRate(device, library, queue);
        testHairGrazingCenterBias(device, library, queue);
        testAtan2Zero(device, library, queue);

        if (g_failures > 0) {
            fprintf(stderr, "FAIL: %d check(s) failed\n", g_failures);
            return 1;
        }
        fprintf(stderr, "PASS: all metal_poc.metal device-side checks passed\n");
        return 0;
    }
}
