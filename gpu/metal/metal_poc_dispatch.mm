// metal_poc_dispatch.mm
// Stage 4: MetalPocApp::compileShaderAndDispatch() (plus its own
// checkGpuResource() helper) - concatenates+compiles the runtime Metal
// shader source (see metal_poc_shader_files.h's own comment), binds every
// GPU-resident buffer/texture buildGPUResources() built, dispatches the
// compute kernel, and reads the rendered HDR buffer back into `pixels` -
// split out of metal_poc.mm once its own Stage 3/4 pair grew past ~1200
// combined lines (a pure code-motion refactor, no behaviour change - same
// precedent as metal_poc_scenes_a.mm's own split, and metal_poc_pbrt_
// loader.mm's own follow-up). Stage 3 (GPU buffer/AS upload) is the
// sibling split, metal_poc_gpu_resources.mm - kept separate from this one
// since "upload the data" and "compile+dispatch+read back" are two
// genuinely different concerns that happen to run back to back, not one
// oversized stage that was arbitrarily cut in half.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_poc_app.h"
#include "metal_poc_shader_files.h"

// Every buffer/texture the compute encoder below binds can be nil if its
// own newBufferWith.../newTextureWithDescriptor call failed - OOM, or a
// requested length exceeding device.maxBufferLength (a real, finite,
// GPU-dependent ceiling a sufficiently large baked pbrt scene could
// plausibly hit, e.g. many ObjectInstance placements each duplicating
// full geometry rather than sharing one buffer - see section 86).
// Metal's own setBuffer:/setTexture: silently UNBIND that slot on a nil
// argument instead of erroring, so an unchecked failure here would let
// the shader read back zeroed/garbage data at that one binding and
// render a WRONG image with NO error anywhere - the hardest kind of bug
// to diagnose (found via a logging/debugging-focused review, not a
// symptom - see section 107). Deliberately does NOT re-check the
// acceleration-structure-only scratch/geometry buffers built earlier in
// this same stage (primScratch/sphereScratch/suzanneScratch/instScratch,
// boundingBoxBuffer/diskBoundingBoxBuffer/instanceBuffer) - those are
// never bound to THIS encoder, and a nil one there already fails loud
// via that build's own existing `buildCmd.status ==
// MTLCommandBufferStatusError` check just above it.
static void checkGpuResource(id resource, const char* name, id<MTLDevice> device, bool* anyFailed) {
    if (!resource) {
        fprintf(stderr, "GPU resource allocation FAILED: '%s' is nil (likely out of memory, "
                        "or this scene is too large for this GPU's max buffer length of %llu "
                        "bytes) - aborting before the render would silently produce a wrong "
                        "image instead of a visible error.\n",
                name, (unsigned long long)device.maxBufferLength);
        *anyFailed = true;
    }
}

// --- Stage 4: compile the shader, dispatch the render, read back -------
bool MetalPocApp::compileShaderAndDispatch(int argc, const char** argv) {
    // --- Compile the shader library from source at runtime ---------
    NSError* error = nil;
    // Three DIRECTORY candidates, tried in priority order (unchanged from
    // before the shader source was split into several files - see
    // metal_poc_shader_files.h's own comment for why every file below is
    // read from this SAME directory rather than resolved independently):
    // 1. Right next to the CURRENTLY RUNNING executable
    //    (_NSGetExecutablePath(), not NSBundle - resolves correctly for
    //    a plain (non-app-bundle) CLI binary too, which is exactly how
    //    a shipped .app's own Contents/MacOS/ray_tracer runs when the
    //    Qt GUI spawns it as a subprocess). This is the only candidate
    //    that works once the binary has been copied/installed anywhere
    //    other than the machine that built it - a real, previously-
    //    undiscovered bug found by actually testing a packaged release
    //    (section 110): RT_METAL_SHADER_DIR below is a compile-time
    //    absolute path into the BUILD MACHINE's own source tree, so a
    //    distributed .dmg's own bundled ray_tracer would report Metal
    //    available (metal_get_diagnostics() never touches this shader
    //    path at all) yet fail every actual GPU render once it got
    //    here, silently, on every machine except the one that built it.
    //    build_and_deploy_macos.sh now also copies every metal_poc_*.metal
    //    file next to the bundled ray_tracer specifically so this
    //    candidate finds them.
    // 2. RT_METAL_SHADER_DIR (set by CMakeLists.txt's metal_poc target,
    //    RT_BUILD_METAL=ON path) - gpu/metal/'s absolute SOURCE
    //    directory, correct only on the machine that built this binary
    //    (a plain `cmake --build` dev loop, never distributed).
    // 3. A __FILE__-relative lookup, for the ad-hoc `clang++
    //    metal_poc.mm ...` invocation this POC started as
    //    (docs/METAL_GPU_FEASIBILITY.md section 7/8/9).
    NSString* shaderDir = nil;
    {
        char exePathBuf[4096];
        uint32_t exePathSize = sizeof(exePathBuf);
        if (_NSGetExecutablePath(exePathBuf, &exePathSize) == 0) {
            NSString* exeDir = [@(exePathBuf) stringByDeletingLastPathComponent];
            int firstCount = 0;
            NSString* candidate = [exeDir stringByAppendingPathComponent:
                @(metalShaderFileNames(&firstCount)[0])];
            if ([[NSFileManager defaultManager] fileExistsAtPath:candidate]) shaderDir = exeDir;
        }
    }
    if (!shaderDir) {
#ifdef RT_METAL_SHADER_DIR
        shaderDir = @(RT_METAL_SHADER_DIR);
#else
        shaderDir = [@(__FILE__) stringByDeletingLastPathComponent];
#endif
    }
    // Concatenate every shader fragment, in metal_poc_shader_files.h's own
    // declared order, into ONE source string - Metal compiles from a
    // single in-memory string (newLibraryWithSource: below), so the split
    // into several files on disk is a source-organization change only,
    // not a real separate-translation-unit split the way the `.mm`
    // side's own per-category files are; every later file's own function
    // still needs every earlier file's own struct/function already
    // defined in the SAME string it's handed.
    NSMutableString* shaderSource = [NSMutableString string];
    int shaderFileCount = 0;
    const char* const* shaderFileNames = metalShaderFileNames(&shaderFileCount);
    for (int i = 0; i < shaderFileCount; ++i) {
        NSString* fragPath = [shaderDir stringByAppendingPathComponent:@(shaderFileNames[i])];
        NSString* fragSource = [NSString stringWithContentsOfFile:fragPath encoding:NSUTF8StringEncoding error:&error];
        if (!fragSource) {
            fprintf(stderr, "Failed to read shader source at %s: %s\n",
                fragPath.UTF8String, error.localizedDescription.UTF8String);
            return false;
        }
        [shaderSource appendString:fragSource];
    }
    MTLCompileOptions* compileOpts = [MTLCompileOptions new];
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:compileOpts error:&error];
    if (!library) {
        fprintf(stderr, "Shader compile failed: %s\n", error.localizedDescription.UTF8String);
        return false;
    }
    id<MTLFunction> kernelFn = [library newFunctionWithName:@"primaryRayKernel"];
    id<MTLFunction> sphereIntersectFn = [library newFunctionWithName:@"sphereIntersectionFunction"];
    id<MTLFunction> diskIntersectFn = [library newFunctionWithName:@"diskIntersectionFunction"];
    id<MTLFunction> cylinderIntersectFn = [library newFunctionWithName:@"cylinderIntersectionFunction"];

    // The intersection function has to be LINKED into the compute
    // pipeline (MTLLinkedFunctions) before an MTLIntersectionFunction
    // Table naming it can be built - a plain newComputePipelineState
    // WithFunction: (used for step 1/2's triangle-only pipeline) has
    // nowhere to put that linkage, hence the switch to the descriptor-
    // based pipeline creation call here.
    MTLComputePipelineDescriptor* pipelineDesc = [MTLComputePipelineDescriptor new];
    pipelineDesc.computeFunction = kernelFn;
    MTLLinkedFunctions* linkedFns = [MTLLinkedFunctions new];
    linkedFns.functions = @[sphereIntersectFn, diskIntersectFn, cylinderIntersectFn];
    pipelineDesc.linkedFunctions = linkedFns;

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:pipelineDesc
        options:MTLPipelineOptionNone reflection:nil error:&error];
    if (!pipeline) {
        fprintf(stderr, "Pipeline creation failed: %s\n", error.localizedDescription.UTF8String);
        return false;
    }

    // --- Intersection function table: two slots now, matching --------
    // bboxGeomDesc's own intersectionFunctionTableOffset (0) and
    // diskGeomDesc's (1) above - this POC's first real "one slot per
    // distinct intersection function" table, not just one slot
    // reused by every custom primitive. `setBuffer:atIndex:N` here
    // sets buffer N in the table's OWN shared argument namespace
    // (every function in ONE table draws from the same set of bound
    // buffers/textures) - sphereIntersectionFunction and
    // diskIntersectionFunction each declare a DIFFERENT `[[buffer(N)]]`
    // in their own MSL signature (0 and 1 respectively) specifically
    // so binding sphereBuffer at atIndex:0 and diskBuffer at
    // atIndex:1 here reaches the right function's own data, not a
    // shared/overwritten slot.
    MTLIntersectionFunctionTableDescriptor* fnTableDesc = [MTLIntersectionFunctionTableDescriptor new];
    fnTableDesc.functionCount = 3;
    id<MTLIntersectionFunctionTable> functionTable = [pipeline newIntersectionFunctionTableWithDescriptor:fnTableDesc];
    id<MTLFunctionHandle> sphereHandle = [pipeline functionHandleWithFunction:sphereIntersectFn];
    id<MTLFunctionHandle> diskHandle = [pipeline functionHandleWithFunction:diskIntersectFn];
    id<MTLFunctionHandle> cylinderHandle = [pipeline functionHandleWithFunction:cylinderIntersectFn];
    [functionTable setFunction:sphereHandle atIndex:0];
    [functionTable setFunction:diskHandle atIndex:1];
    [functionTable setFunction:cylinderHandle atIndex:2];
    // sphereIntersectionFunction/diskIntersectionFunction/
    // cylinderIntersectionFunction each read their own geometry buffer
    // (metal_poc.metal buffer(0)/buffer(1)/buffer(2) respectively - a
    // SEPARATE argument table from the calling kernel's own
    // buffer(0..27), see that file's own comment) - bound here, on the
    // function table, not on the compute encoder.
    [functionTable setBuffer:sphereBuffer offset:0 atIndex:0];
    [functionTable setBuffer:diskBuffer offset:0 atIndex:1];
    [functionTable setBuffer:cylinderBuffer offset:0 atIndex:2];
    // A8/section 176: sphereIntersectionFunction's own SpherePayload::
    // isShadowRay check needs sphereMaterials too, to tell a real
    // medium sphere (materialType 28) apart from an ordinary one - the
    // SAME sphereMaterialBuffer the calling kernel already binds at its
    // own buffer(4), bound a SECOND time here since this table has its
    // own independent argument namespace (diskGeomDesc's own comment).
    [functionTable setBuffer:sphereMaterialBuffer offset:0 atIndex:3];

    // --- Output texture + uniforms ----------------------------------
    MTLTextureDescriptor* texDesc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
        width:width height:height mipmapped:NO];
    texDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    texDesc.storageMode = MTLStorageModeShared;
    id<MTLTexture> outTexture = [device newTextureWithDescriptor:texDesc];

    // --- Earth texture (the back wall's materialType=3 source) -----
    // stb_image decodes straight to interleaved 8-bit RGBA regardless
    // of the source JPEG's channel count (the 4th `desiredChannels`
    // arg below), which is exactly MTLPixelFormatRGBA8Unorm_sRGB's own
    // BYTE layout - no repacking needed between stbi_load's buffer and
    // replaceRegion:. The `_sRGB` pixel format (not plain
    // `RGBA8Unorm`, this POC's own format up through PR #37) matters
    // for more than naming: an ordinary 8-bit JPEG/PNG's own stored
    // bytes are sRGB-gamma-ENCODED (perceptually, not linearly,
    // spaced) - every earlier render sampled those bytes directly as
    // if they were already linear radiance, silently darkening every
    // midtone the earth texture (and, via GI, everything it bounces
    // light onto) ever produced. `_sRGB` makes the texture SAMPLE
    // instruction itself convert sRGB to linear before the shader
    // ever sees a value - the standard, hardware-accelerated way to
    // do this, rather than a manual `pow(c, 2.2)` after sampling in
    // the shader.
    NSString* imagesDir = nil;
    {
        NSString* exeDir = executableDir();
        NSString* candidate = [exeDir stringByAppendingPathComponent:@"images"];
        if (exeDir && [[NSFileManager defaultManager] fileExistsAtPath:
                [candidate stringByAppendingPathComponent:@"earthmap.jpg"]]) {
            imagesDir = candidate;
        }
    }
    if (!imagesDir) {
#ifdef RT_MODELS_DIR
        imagesDir = [[@(RT_MODELS_DIR) stringByDeletingLastPathComponent]
            stringByAppendingPathComponent:@"images"];
#else
        imagesDir = [[@(__FILE__) stringByDeletingLastPathComponent]
            stringByAppendingPathComponent:@"../../images"];
#endif
    }
    NSString* earthPath = [imagesDir stringByAppendingPathComponent:@"earthmap.jpg"];
    int earthW = 0, earthH = 0, earthChannels = 0;
    unsigned char* earthPixels = stbi_load(earthPath.UTF8String, &earthW, &earthH, &earthChannels, 4);
    id<MTLTexture> earthTexture = nil;
    // Environment-map importance sampling (phase 2 - see section 69 for
    // phase 1's own host-side EnvDistribution2D, built and independently
    // verified there but not yet wired to anything). Built from the SAME
    // decoded `earthPixels` bytes right before they're freed below - the
    // shader-side NEE counterpart to `earthTexture`'s own miss-path
    // lookup (equirectangularUV()), which previously had no importance-
    // sampling strategy at all. Left default-constructed (empty arrays)
    // in the fallback/missing-JPEG case below; `envMapWidth`/
    // `envMapHeight` stay 0 in that case too, which the shader's own
    // Uniforms comment documents as "skip this NEE strategy entirely."
    EnvDistribution2D envDist;
    if (earthPixels) {
        MTLTextureDescriptor* earthDesc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB
            width:(NSUInteger)earthW height:(NSUInteger)earthH mipmapped:NO];
        earthDesc.usage = MTLTextureUsageShaderRead;
        earthDesc.storageMode = MTLStorageModeShared;
        earthTexture = [device newTextureWithDescriptor:earthDesc];
        MTLRegion earthRegion = MTLRegionMake2D(0, 0, (NSUInteger)earthW, (NSUInteger)earthH);
        [earthTexture replaceRegion:earthRegion mipmapLevel:0 withBytes:earthPixels
            bytesPerRow:(NSUInteger)earthW * 4];
        buildEnvDistribution2D(earthPixels, earthW, earthH, envDist);
        stbi_image_free(earthPixels);
        fprintf(stderr, "Loaded %s: %dx%d, %d channels\n", earthPath.UTF8String, earthW, earthH, earthChannels);
    } else {
        fprintf(stderr, "Could not load %s - back wall will read black/undefined texture data.\n",
            earthPath.UTF8String);
        // A 1x1 white fallback keeps the shader's unconditional
        // texture bind valid (Metal requires SOME texture at the
        // bound slot) even if the JPEG is missing. `_sRGB` for
        // consistency with the real texture above, though pure white
        // (255,255,255) round-trips through the sRGB<->linear
        // conversion unchanged either way.
        MTLTextureDescriptor* fallbackDesc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB width:1 height:1 mipmapped:NO];
        fallbackDesc.usage = MTLTextureUsageShaderRead;
        fallbackDesc.storageMode = MTLStorageModeShared;
        earthTexture = [device newTextureWithDescriptor:fallbackDesc];
        uint8_t white4[4] = {255, 255, 255, 255};
        [earthTexture replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:white4 bytesPerRow:4];
    }

    // A pbrt-loaded scene's own image-based infinite light - a
    // genuinely SEPARATE texture from earthTexture above (see
    // loadPbrtScene()'s own comment on why). pbrtEnvImagePixels is
    // already decoded, linear float RGB (3 floats/pixel) - stored
    // straight into an RGBA32Float texture (padding alpha=1, no sRGB
    // format/decode needed at all: unlike earthTexture's own 8-bit JPEG
    // source, there's no gamma curve to reverse here). Falls back to a
    // 1x1 black texture (Metal requires SOME texture bound at every
    // used slot) when the scene has no image-based infinite light -
    // harmless, since pbrtHasImageEnvLight gates whether the shader
    // ever actually samples it.
    id<MTLTexture> pbrtEnvTexture = nil;
    {
        const uint32_t pw = havePbrtImageEnvLight ? (uint32_t)pbrtEnvImageWidth : 1u;
        const uint32_t ph = havePbrtImageEnvLight ? (uint32_t)pbrtEnvImageHeight : 1u;
        MTLTextureDescriptor* pbrtEnvDesc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
            width:pw height:ph mipmapped:NO];
        pbrtEnvDesc.usage = MTLTextureUsageShaderRead;
        pbrtEnvDesc.storageMode = MTLStorageModeShared;
        pbrtEnvTexture = [device newTextureWithDescriptor:pbrtEnvDesc];
        std::vector<float> rgba((size_t)pw * ph * 4, 0.0f);
        if (havePbrtImageEnvLight) {
            for (size_t i = 0; i < (size_t)pw * ph; ++i) {
                rgba[i * 4 + 0] = pbrtEnvImagePixels[i * 3 + 0];
                rgba[i * 4 + 1] = pbrtEnvImagePixels[i * 3 + 1];
                rgba[i * 4 + 2] = pbrtEnvImagePixels[i * 3 + 2];
                rgba[i * 4 + 3] = 1.0f;
            }
        }
        [pbrtEnvTexture replaceRegion:MTLRegionMake2D(0, 0, pw, ph) mipmapLevel:0
            withBytes:rgba.data() bytesPerRow:(NSUInteger)pw * 4 * sizeof(float)];
    }

    // A pbrt-loaded scene's own real per-light goniometric/projection
    // profile images (section 98) - same upload pattern as pbrtEnvTexture
    // above (already-decoded linear float RGB -> RGBA32Float, 1x1 black
    // fallback when there's no such light, harmless since usePbrtTexture
    // gates whether any light actually reads it).
    id<MTLTexture> pbrtGoniometricTexture = nil;
    {
        const uint32_t pw = havePbrtGoniometricImage ? (uint32_t)pbrtGoniometricImageWidth : 1u;
        const uint32_t ph = havePbrtGoniometricImage ? (uint32_t)pbrtGoniometricImageHeight : 1u;
        MTLTextureDescriptor* desc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
            width:pw height:ph mipmapped:NO];
        desc.usage = MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared;
        pbrtGoniometricTexture = [device newTextureWithDescriptor:desc];
        std::vector<float> rgba((size_t)pw * ph * 4, 0.0f);
        if (havePbrtGoniometricImage) {
            for (size_t i = 0; i < (size_t)pw * ph; ++i) {
                rgba[i * 4 + 0] = pbrtGoniometricImagePixels[i * 3 + 0];
                rgba[i * 4 + 1] = pbrtGoniometricImagePixels[i * 3 + 1];
                rgba[i * 4 + 2] = pbrtGoniometricImagePixels[i * 3 + 2];
                rgba[i * 4 + 3] = 1.0f;
            }
        }
        [pbrtGoniometricTexture replaceRegion:MTLRegionMake2D(0, 0, pw, ph) mipmapLevel:0
            withBytes:rgba.data() bytesPerRow:(NSUInteger)pw * 4 * sizeof(float)];
    }

    id<MTLTexture> pbrtProjectionTexture = nil;
    {
        const uint32_t pw = havePbrtProjectionImage ? (uint32_t)pbrtProjectionImageWidth : 1u;
        const uint32_t ph = havePbrtProjectionImage ? (uint32_t)pbrtProjectionImageHeight : 1u;
        MTLTextureDescriptor* desc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
            width:pw height:ph mipmapped:NO];
        desc.usage = MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared;
        pbrtProjectionTexture = [device newTextureWithDescriptor:desc];
        std::vector<float> rgba((size_t)pw * ph * 4, 0.0f);
        if (havePbrtProjectionImage) {
            for (size_t i = 0; i < (size_t)pw * ph; ++i) {
                rgba[i * 4 + 0] = pbrtProjectionImagePixels[i * 3 + 0];
                rgba[i * 4 + 1] = pbrtProjectionImagePixels[i * 3 + 1];
                rgba[i * 4 + 2] = pbrtProjectionImagePixels[i * 3 + 2];
                rgba[i * 4 + 3] = 1.0f;
            }
        }
        [pbrtProjectionTexture replaceRegion:MTLRegionMake2D(0, 0, pw, ph) mipmapLevel:0
            withBytes:rgba.data() bytesPerRow:(NSUInteger)pw * 4 * sizeof(float)];
    }

    // A pbrt-loaded scene's own image-based AreaLightSource (section
    // 105) - same upload pattern as pbrtGoniometricTexture/
    // pbrtProjectionTexture above.
    id<MTLTexture> pbrtAreaLightTexture = nil;
    {
        const uint32_t pw = havePbrtAreaLightImage ? (uint32_t)pbrtAreaLightImageWidth : 1u;
        const uint32_t ph = havePbrtAreaLightImage ? (uint32_t)pbrtAreaLightImageHeight : 1u;
        MTLTextureDescriptor* desc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
            width:pw height:ph mipmapped:NO];
        desc.usage = MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared;
        pbrtAreaLightTexture = [device newTextureWithDescriptor:desc];
        std::vector<float> rgba((size_t)pw * ph * 4, 0.0f);
        if (havePbrtAreaLightImage) {
            for (size_t i = 0; i < (size_t)pw * ph; ++i) {
                rgba[i * 4 + 0] = pbrtAreaLightImagePixels[i * 3 + 0];
                rgba[i * 4 + 1] = pbrtAreaLightImagePixels[i * 3 + 1];
                rgba[i * 4 + 2] = pbrtAreaLightImagePixels[i * 3 + 2];
                rgba[i * 4 + 3] = 1.0f;
            }
        }
        [pbrtAreaLightTexture replaceRegion:MTLRegionMake2D(0, 0, pw, ph) mipmapLevel:0
            withBytes:rgba.data() bytesPerRow:(NSUInteger)pw * 4 * sizeof(float)];
    }

    // A pbrt-loaded Diffuse/CoatedDiffuse material's own real imagemap-
    // bound "texture reflectance" (F5/F9, section 166) - same upload
    // pattern as pbrtAreaLightTexture/pbrtGoniometricTexture above.
    id<MTLTexture> pbrtDiffuseTexture = nil;
    {
        const uint32_t pw = havePbrtDiffuseImage ? (uint32_t)pbrtDiffuseImageWidth : 1u;
        const uint32_t ph = havePbrtDiffuseImage ? (uint32_t)pbrtDiffuseImageHeight : 1u;
        MTLTextureDescriptor* desc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
            width:pw height:ph mipmapped:NO];
        desc.usage = MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared;
        pbrtDiffuseTexture = [device newTextureWithDescriptor:desc];
        std::vector<float> rgba((size_t)pw * ph * 4, 0.0f);
        if (havePbrtDiffuseImage) {
            for (size_t i = 0; i < (size_t)pw * ph; ++i) {
                rgba[i * 4 + 0] = pbrtDiffuseImagePixels[i * 3 + 0];
                rgba[i * 4 + 1] = pbrtDiffuseImagePixels[i * 3 + 1];
                rgba[i * 4 + 2] = pbrtDiffuseImagePixels[i * 3 + 2];
                rgba[i * 4 + 3] = 1.0f;
            }
        }
        [pbrtDiffuseTexture replaceRegion:MTLRegionMake2D(0, 0, pw, ph) mipmapLevel:0
            withBytes:rgba.data() bytesPerRow:(NSUInteger)pw * 4 * sizeof(float)];
    }

    // envMarginalCDF/envConditionalCDF buffers - a real (non-empty)
    // envDist above uploads its own arrays directly; the fallback case
    // (missing JPEG) still needs SOME buffer bound at these indices
    // (Metal requires a real resource at every declared buffer slot,
    // the same reason earthTexture's own fallback path exists above),
    // hence the single-float dummy - `envMapWidth`/`envMapHeight`
    // staying 0 is what actually keeps the shader from ever reading
    // past it.
    uint32_t envMapWidth = 0, envMapHeight = 0;
    id<MTLBuffer> envMarginalCDFBuffer;
    id<MTLBuffer> envConditionalCDFBuffer;
    if (!envDist.marginalCDF.empty()) {
        envMarginalCDFBuffer = [device newBufferWithBytes:envDist.marginalCDF.data()
            length:envDist.marginalCDF.size() * sizeof(float) options:MTLResourceStorageModeShared];
        envConditionalCDFBuffer = [device newBufferWithBytes:envDist.conditionalCDF.data()
            length:envDist.conditionalCDF.size() * sizeof(float) options:MTLResourceStorageModeShared];
        envMapWidth = (uint32_t)envDist.width;
        envMapHeight = (uint32_t)envDist.height;
    } else {
        float dummy = 0.0f;
        envMarginalCDFBuffer = [device newBufferWithBytes:&dummy length:sizeof(float) options:MTLResourceStorageModeShared];
        envConditionalCDFBuffer = [device newBufferWithBytes:&dummy length:sizeof(float) options:MTLResourceStorageModeShared];
    }

    // pbrtEnvMarginalCDF/pbrtEnvConditionalCDF buffers (section 96) - the
    // NEE-importance-sampling counterpart to pbrtEnvTexture above, same
    // idea as envMarginalCDF/envConditionalCDF just above but for a
    // SEPARATE image (a pbrt-loaded scene's own image-based infinite
    // light, not earthTexture). Uses the float-RGB buildEnvDistribution2D()
    // overload directly on pbrtEnvImagePixels (already linear, already
    // decoded - no srgbByteToLinear() round-trip needed the way
    // earthPixels' own RGBA8 bytes require). Empty/dummy fallback when
    // there's no image-based infinite light, same "0 disables it" pattern
    // as envMapWidth's own fallback.
    EnvDistribution2D pbrtEnvDist;
    if (havePbrtImageEnvLight) {
        buildEnvDistribution2D(pbrtEnvImagePixels.data(), pbrtEnvImageWidth, pbrtEnvImageHeight, pbrtEnvDist);
    }
    uint32_t pbrtEnvMapWidth = 0, pbrtEnvMapHeight = 0;
    id<MTLBuffer> pbrtEnvMarginalCDFBuffer;
    id<MTLBuffer> pbrtEnvConditionalCDFBuffer;
    if (!pbrtEnvDist.marginalCDF.empty()) {
        pbrtEnvMarginalCDFBuffer = [device newBufferWithBytes:pbrtEnvDist.marginalCDF.data()
            length:pbrtEnvDist.marginalCDF.size() * sizeof(float) options:MTLResourceStorageModeShared];
        pbrtEnvConditionalCDFBuffer = [device newBufferWithBytes:pbrtEnvDist.conditionalCDF.data()
            length:pbrtEnvDist.conditionalCDF.size() * sizeof(float) options:MTLResourceStorageModeShared];
        pbrtEnvMapWidth = (uint32_t)pbrtEnvDist.width;
        pbrtEnvMapHeight = (uint32_t)pbrtEnvDist.height;
    } else {
        float pbrtEnvDummy = 0.0f;
        pbrtEnvMarginalCDFBuffer = [device newBufferWithBytes:&pbrtEnvDummy length:sizeof(float) options:MTLResourceStorageModeShared];
        pbrtEnvConditionalCDFBuffer = [device newBufferWithBytes:&pbrtEnvDummy length:sizeof(float) options:MTLResourceStorageModeShared];
    }

    // GGX multi-scatter energy-compensation table (phase 2 - see phase
    // 1's own header comment in metal_poc_host_math.h for the full
    // "why," found by spot-checking this POC's GGX conductor material
    // against Blender Cycles as a second reference). Built ONCE here at
    // startup (a real-time Monte Carlo precompute, not Cycles' own
    // offline 8-64-million-sample tool) and uploaded as a single GPU
    // buffer - only `E` is needed on the device side; `Eavg` only feeds
    // the "multi-bounce Fresnel darkening" refinement this phase
    // deliberately doesn't attempt (see that same header comment).
    // std::mt19937 (not this POC's own device-side randFloat()) is fine
    // here - this is a host-only precompute of a smooth, low-frequency
    // table, not a per-pixel render decision that needs to match the
    // shader's own RNG stream bit-for-bit.
    std::mt19937 ggxEnergyRng(1337);
    std::uniform_real_distribution<float> ggxEnergyDist(0.0f, 1.0f);
    auto ggxEnergyRandFn = [&]() { return ggxEnergyDist(ggxEnergyRng); };
    GGXEnergyTable ggxEnergyTable;
    buildGGXEnergyTable(/*roughRes=*/32, /*muRes=*/32, /*samplesPerCell=*/2048,
                        ggxEnergyTable, ggxEnergyRandFn);
    id<MTLBuffer> ggxEnergyTableBuffer = [device newBufferWithBytes:ggxEnergyTable.E.data()
        length:ggxEnergyTable.E.size() * sizeof(float) options:MTLResourceStorageModeShared];

    const uint32_t samplesPerPixel = (argc > 4) ? (uint32_t)atoi(argv[4]) : 64;
    const uint32_t maxDepth = (argc > 5) ? (uint32_t)atoi(argv[5]) : 8;
    fprintf(stderr, "Samples/pixel: %u, max depth: %u\n", samplesPerPixel, maxDepth);

    Uniforms uniforms{};
    uniforms.cameraPos = PackedFloat3{0.0f, 0.0f, 3.2f};
    float3 forward = simd::normalize(float3{0, 0, -1});
    uniforms.cameraForward = PackedFloat3{forward.x, forward.y, forward.z};
    uniforms.cameraRight = PackedFloat3{1, 0, 0};
    uniforms.cameraUp = PackedFloat3{0, 1, 0};
    uniforms.tanHalfFov = tanf(0.5f * 40.0f * (float)M_PI / 180.0f);
    uniforms.aspect = (float)width / (float)height;
    uniforms.width = width;
    uniforms.height = height;
    uniforms.samplesPerPixel = samplesPerPixel;
    uniforms.maxDepth = maxDepth;
    uniforms.frameSeed = 1u;
    uniforms.lightCount = (uint32_t)lights.size();
    // Thin-lens depth of field: focused on the gold conductor sphere
    // (the nearest object to the camera), so it renders pixel-sharp
    // while the dielectric sphere just behind it and the back
    // wall/Suzanne further back show progressively more defocus blur -
    // the falloff is what actually demonstrates this is a real lens
    // model, not just a uniform blur filter over the whole frame.
    uniforms.lensRadius = 0.05f;
    uniforms.focusDistance = uniforms.cameraPos.z - spheres[1].center.z; // gold sphere's own z
    // A hexagonal (6-blade) aperture rather than a perfectly circular
    // one - see Uniforms' own apertureBlades comment. The classic
    // photographic blade count; out-of-focus highlights (area/point/
    // spot light reflections on the defocused back-wall geometry)
    // should now read as hexagons, not perfect circles.
    uniforms.apertureBlades = 6;
    // Shutter motion blur: a small horizontal dolly over the frame's
    // simulated exposure - chosen (over, say, an object moving) since
    // it needs no acceleration-structure/intersection-function
    // changes at all, purely a primary-ray-generation addition (see
    // metal_poc.metal's own comment on why: a moving CUSTOM primitive
    // would need per-sample time threaded into
    // sphereIntersectionFunction's own, separate argument table, real
    // additional Metal API surface this increment intentionally
    // doesn't take on).
    uniforms.cameraVelocity = PackedFloat3{0.015f, 0.0f, 0.0f};
    // Homogeneous fog filling the whole room - subtle (transmittance
    // ~0.7 over the ~4-unit camera-to-back-wall sightline: exp(-0.08*4)
    // ~ 0.73), meant to read as a light atmospheric haze visible in
    // the light shafts/depth falloff, not an opaque room-filling mist
    // that would fight every other material's own visibility.
    uniforms.fogSigmaT = 0.05f;
    uniforms.fogAlbedo = PackedFloat3{0.85f, 0.88f, 0.95f}; // mostly-scattering, faint cool tint
    // On: a miss ray samples earthTexture by direction (equirectangular)
    // instead of the flat two-colour gradient - the room's open front
    // means most miss rays are secondary/GI bounces (a mirror/glass
    // surface reflecting/refracting outward), not primary camera rays,
    // so this mostly shows up subtly rather than as an obvious visible
    // backdrop - see docs/METAL_GPU_FEASIBILITY.md's own note on
    // verifying this with a dedicated wide-FOV test render.
    uniforms.useEnvironmentMap = 1u;
    // Moderate forward scattering (real fog/haze skews strongly
    // forward in reality - Mie scattering off water droplets often
    // has g around 0.7-0.9 - 0.4 is deliberately more modest, so the
    // difference from isotropic reads as a stylistic tint on the fog
    // rather than a dramatic visible change).
    uniforms.fogAsymmetryG = 0.4f;
    uniforms.pointLightCount = (uint32_t)pointLights.size();
    uniforms.directionalLightCount = (uint32_t)directionalLights.size();
    uniforms.projectionLightCount = (uint32_t)projectionLights.size();
    uniforms.goniometricLightCount = (uint32_t)goniometricLights.size();
    // Single-pass adaptive sampling (see metal_poc.metal's own
    // shading-loop comment) - enabled by default for this scene:
    // converged pixels (most of the flat-coloured walls/ceiling)
    // stop well short of the full samplesPerPixel budget, spending
    // it instead on the noisier fog/specular/caustic regions this
    // scene already has plenty of - a real render-TIME win at
    // (ideally) no visible quality cost, verified via a dedicated
    // A/B render, not assumed.
    uniforms.adaptiveSampling = 1u;
    uniforms.pbrtEnvMapWidth = pbrtEnvMapWidth;
    uniforms.pbrtEnvMapHeight = pbrtEnvMapHeight;
    uniforms.envMapWidth = envMapWidth;
    uniforms.envMapHeight = envMapHeight;
    uniforms.ggxEnergyRoughRes = (uint32_t)ggxEnergyTable.roughRes;
    uniforms.ggxEnergyMuRes = (uint32_t)ggxEnergyTable.muRes;

    if (havePbrtCamera) {
        // A real pbrt scene was loaded (loadPbrtScene()) - override every
        // scale/placement-dependent field the defaults above assumed,
        // all tuned for the hardcoded room's own [-1,1] scale, not a
        // real pbrt scene's own (often much larger - e.g. a ~500-unit
        // classic Cornell box) world scale. fogSigmaT=0.05 alone would
        // otherwise make an 800-unit sightline read as solid black
        // (exp(-0.05*800) ~ 0) - not a subtle atmospheric tweak, a
        // completely broken render.
        uniforms.cameraPos = PackedFloat3{pbrtCameraPos.x, pbrtCameraPos.y, pbrtCameraPos.z};
        uniforms.cameraForward = PackedFloat3{pbrtCameraForward.x, pbrtCameraForward.y, pbrtCameraForward.z};
        uniforms.cameraRight = PackedFloat3{pbrtCameraRight.x, pbrtCameraRight.y, pbrtCameraRight.z};
        uniforms.cameraUp = PackedFloat3{pbrtCameraUp.x, pbrtCameraUp.y, pbrtCameraUp.z};
        uniforms.tanHalfFov = pbrtTanHalfFov;
        // D13 (section 174): real orbit-style camera motion blur, gated
        // on the scene's own sceneCameraVelocity actually being nonzero
        // (every other scene leaves it at its own {0,0,0} default, so
        // this is 0/false for all of them - a provable no-op). Always
        // copies pbrtCameraLookAtWorld/pbrtCameraUpRaw regardless -
        // both already exist for applyCameraOverride()'s own sake, and
        // are simply unread by the shader whenever hasCameraOrbitBlur
        // is 0.
        uniforms.cameraLookAtBlur = PackedFloat3{pbrtCameraLookAtWorld.x, pbrtCameraLookAtWorld.y, pbrtCameraLookAtWorld.z};
        uniforms.cameraUpRawBlur = PackedFloat3{pbrtCameraUpRaw.x, pbrtCameraUpRaw.y, pbrtCameraUpRaw.z};
        uniforms.hasCameraOrbitBlur = (sceneCameraVelocity.x != 0.0f || sceneCameraVelocity.y != 0.0f ||
                                        sceneCameraVelocity.z != 0.0f) ? 1u : 0u;
        // pbrt v1 still doesn't read a pbrt FILE's own "float lensradius"/
        // "float focaldistance" Camera parameters (a genuinely separate,
        // still-open gap) - but a hand-authored scene (D5, section 137)
        // has no pbrt file to parse from at all, so its own
        // pbrtLensRadius/pbrtFocusDistance (defaulting to 0/1, "no DOF" -
        // every scene before D5) are read directly here instead.
        // focusDistance is unread by the shader whenever lensRadius == 0.
        uniforms.lensRadius = pbrtLensRadius;
        uniforms.focusDistance = pbrtFocusDistance;
        // See MetalPocApp::havePbrtOrthographic's own comment - reuses
        // pbrtTanHalfFov (already copied to uniforms.tanHalfFov above)
        // as the orthographic screen window's own half-extent.
        uniforms.cameraOrthographic = havePbrtOrthographic ? 1u : 0u;
        // See MetalPocApp::havePbrtSpherical's own comment.
        uniforms.cameraSpherical = havePbrtSpherical ? 1u : 0u;
        // See MetalPocApp::havePbrtSphericalEqualArea's own comment.
        uniforms.sphericalMappingEqualArea = havePbrtSphericalEqualArea ? 1u : 0u;
        // See MetalPocApp::havePbrtRealisticCamera's own comment.
        uniforms.cameraRealistic = havePbrtRealisticCamera ? 1u : 0u;
        uniforms.numLensElements = (uint32_t)realisticLensElements.size();
        uniforms.numExitPupilBounds = (uint32_t)realisticExitPupilBounds.size();
        uniforms.filmHalfX = realisticFilmHalfX;
        uniforms.filmHalfY = realisticFilmHalfY;
        uniforms.lensRearZ = realisticLensRearZ;
        // Adaptive sampling (enabled by default just above,
        // uniforms.adaptiveSampling=1) OFF for a real multi-element-lens
        // camera - a real, found-by-rendering bug, not a style choice.
        // Its convergence test (metal_poc.metal's own shading-loop
        // comment) checks standardError/mean against a fixed threshold,
        // which assumes a roughly UNIMODAL per-sample radiance
        // distribution (true for every earlier camera mode's own noise,
        // which is why the heuristic was tuned/validated against those).
        // A real lens's own per-sample radiance is instead strongly
        // BIMODAL: most exit-pupil samples land outside the true
        // (non-rectangular) aperture and get vignetted to EXACTLY 0
        // (sampleRealisticCameraRay's own comment), while the rest carry
        // the entire `cameraWeight`-scaled contribution. Many exact
        // zeros pull the running mean/variance down together, so the
        // ratio can cross the threshold after just a handful of samples
        // even though the NONZERO tail is still wildly undersampled -
        // confirmed empirically: with adaptive sampling on, a 3000spp
        // D4 render finished in ~4s (barely slower than 100spp) and was
        // JUST as speckled; forcing every sample through to the full
        // requested budget (this override) made that same 3000spp
        // render converge to a clean, smooth image. The visible
        // "speckle" itself is this per-pixel undersampling residual,
        // amplified into colour fringing by postProcessAndWrite()'s own
        // always-on chromatic aberration (a small, fixed per-channel
        // pixel-space shift - invisible on an already-smooth image, but
        // it visibly decorrelates R/G/B on a noisy one).
        if (havePbrtRealisticCamera) uniforms.adaptiveSampling = 0u;
        // D13 (section 174): reads the scene's own sceneCameraVelocity
        // instead of unconditionally zeroing this out - see that
        // member's own comment. {0,0,0} (every scene but D13) is an
        // exact no-op, identical to this line's own previous literal.
        uniforms.cameraVelocity = sceneCameraVelocity;
        if (havePbrtMedium) {
            uniforms.fogSigmaT = pbrtFogSigmaT;
            uniforms.fogAlbedo = PackedFloat3{pbrtFogAlbedo.x, pbrtFogAlbedo.y, pbrtFogAlbedo.z};
            uniforms.fogAsymmetryG = pbrtFogAsymmetryG;
        } else {
            uniforms.fogSigmaT = 0.0f;                      // no participating medium in this scene
        }
        uniforms.useEnvironmentMap = 0u;                    // this scene's own sky, if any, replaces the hardcoded room's earthTexture-based one below
        // See Uniforms::isPbrtScene's own comment (metal_poc_types.metal)
        // - a real pbrt scene with no infinite light gets a BLACK
        // miss-path background, not the hardcoded room's own sky
        // gradient. Deliberately `!pbrtScenePath.empty()`, NOT the
        // broader `havePbrtCamera` this whole block is already gated
        // on - havePbrtCamera is true for every HAND-AUTHORED scene's
        // own camera setup too (buildCornellBoxA1() and nearly every
        // other builder set it), not just a genuinely loaded pbrt FILE.
        // A first version of this fix used havePbrtCamera directly and
        // made A1/G1/every other hand-authored scene's own background
        // incorrectly black too (caught by this PR's own before/after
        // hash sweep across EVERY scene, not just the pbrt-file ones
        // this fix was meant for - exactly the discipline that check
        // exists to catch).
        uniforms.isPbrtScene = pbrtScenePath.empty() ? 0u : 1u;
        if (havePbrtConstantEnvLight) {
            uniforms.pbrtHasConstantEnvLight = 1u;
            uniforms.pbrtEnvColor = PackedFloat3{pbrtEnvColor.x, pbrtEnvColor.y, pbrtEnvColor.z};
        } else if (havePbrtImageEnvLight) {
            uniforms.pbrtHasImageEnvLight = 1u;
        }
    }

    id<MTLBuffer> uniformBuffer = [device newBufferWithBytes:&uniforms length:sizeof(Uniforms) options:MTLResourceStorageModeShared];

    // Checked once, here, right before they're all bound below - see
    // checkGpuResource()'s own comment for why this one spot (not each
    // allocation call site above) and why the AS-only scratch/geometry
    // buffers aren't included.
    bool anyResourceFailed = false;
    checkGpuResource(uniformBuffer, "uniformBuffer", device, &anyResourceFailed);
    checkGpuResource(materialBuffer, "materialBuffer", device, &anyResourceFailed);
    checkGpuResource(vertexBuffer, "vertexBuffer", device, &anyResourceFailed);
    checkGpuResource(sphereMaterialBuffer, "sphereMaterialBuffer", device, &anyResourceFailed);
    checkGpuResource(cloudMediumBuffer, "cloudMediumBuffer", device, &anyResourceFailed);
    checkGpuResource(rgbGridMediumBuffer, "rgbGridMediumBuffer", device, &anyResourceFailed);
    checkGpuResource(rgbGridDataBuffer, "rgbGridDataBuffer", device, &anyResourceFailed);
    checkGpuResource(sphereBuffer, "sphereBuffer", device, &anyResourceFailed);
    checkGpuResource(normalBuffer, "normalBuffer", device, &anyResourceFailed);
    checkGpuResource(uvBuffer, "uvBuffer", device, &anyResourceFailed);
    checkGpuResource(lightBuffer, "lightBuffer", device, &anyResourceFailed);
    checkGpuResource(suzanneNormalBuffer, "suzanneNormalBuffer", device, &anyResourceFailed);
    checkGpuResource(suzanneMaterialBuffer, "suzanneMaterialBuffer", device, &anyResourceFailed);
    checkGpuResource(instanceTransformBuffer, "instanceTransformBuffer", device, &anyResourceFailed);
    checkGpuResource(diskBuffer, "diskBuffer", device, &anyResourceFailed);
    checkGpuResource(diskMaterialBuffer, "diskMaterialBuffer", device, &anyResourceFailed);
    checkGpuResource(cylinderBuffer, "cylinderBuffer", device, &anyResourceFailed);
    checkGpuResource(cylinderMaterialBuffer, "cylinderMaterialBuffer", device, &anyResourceFailed);
    checkGpuResource(pointLightBuffer, "pointLightBuffer", device, &anyResourceFailed);
    checkGpuResource(directionalLightBuffer, "directionalLightBuffer", device, &anyResourceFailed);
    checkGpuResource(projectionLightBuffer, "projectionLightBuffer", device, &anyResourceFailed);
    checkGpuResource(goniometricLightBuffer, "goniometricLightBuffer", device, &anyResourceFailed);
    checkGpuResource(lensElementBuffer, "lensElementBuffer", device, &anyResourceFailed);
    checkGpuResource(exitPupilBoundsBuffer, "exitPupilBoundsBuffer", device, &anyResourceFailed);
    checkGpuResource(envMarginalCDFBuffer, "envMarginalCDFBuffer", device, &anyResourceFailed);
    checkGpuResource(envConditionalCDFBuffer, "envConditionalCDFBuffer", device, &anyResourceFailed);
    checkGpuResource(ggxEnergyTableBuffer, "ggxEnergyTableBuffer", device, &anyResourceFailed);
    checkGpuResource(pbrtEnvMarginalCDFBuffer, "pbrtEnvMarginalCDFBuffer", device, &anyResourceFailed);
    checkGpuResource(pbrtEnvConditionalCDFBuffer, "pbrtEnvConditionalCDFBuffer", device, &anyResourceFailed);
    checkGpuResource(outTexture, "outTexture", device, &anyResourceFailed);
    checkGpuResource(earthTexture, "earthTexture", device, &anyResourceFailed);
    checkGpuResource(goniometricTexture, "goniometricTexture", device, &anyResourceFailed);
    checkGpuResource(pbrtEnvTexture, "pbrtEnvTexture", device, &anyResourceFailed);
    checkGpuResource(pbrtGoniometricTexture, "pbrtGoniometricTexture", device, &anyResourceFailed);
    checkGpuResource(pbrtProjectionTexture, "pbrtProjectionTexture", device, &anyResourceFailed);
    checkGpuResource(pbrtAreaLightTexture, "pbrtAreaLightTexture", device, &anyResourceFailed);
    if (anyResourceFailed) return false;

    // --- Dispatch ----------------------------------------------------
    id<MTLCommandBuffer> renderCmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [renderCmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setTexture:outTexture atIndex:0];
    [enc setTexture:earthTexture atIndex:1];
    [enc setTexture:goniometricTexture atIndex:2];
    [enc setTexture:pbrtEnvTexture atIndex:3];
    [enc setTexture:pbrtGoniometricTexture atIndex:4];
    [enc setTexture:pbrtProjectionTexture atIndex:5];
    [enc setTexture:pbrtAreaLightTexture atIndex:6];
    [enc setTexture:pbrtDiffuseTexture atIndex:7];
    [enc setAccelerationStructure:instAS atBufferIndex:0];
    [enc setBuffer:uniformBuffer offset:0 atIndex:1];
    [enc setBuffer:materialBuffer offset:0 atIndex:2];
    [enc setBuffer:vertexBuffer offset:0 atIndex:3];
    [enc setBuffer:sphereMaterialBuffer offset:0 atIndex:4];
    [enc setBuffer:sphereBuffer offset:0 atIndex:5];
    [enc setIntersectionFunctionTable:functionTable atBufferIndex:6];
    [enc setBuffer:normalBuffer offset:0 atIndex:7];
    [enc setBuffer:uvBuffer offset:0 atIndex:8];
    [enc setBuffer:lightBuffer offset:0 atIndex:9];
    [enc setBuffer:suzanneNormalBuffer offset:0 atIndex:10];
    [enc setBuffer:suzanneMaterialBuffer offset:0 atIndex:11];
    [enc setBuffer:instanceTransformBuffer offset:0 atIndex:12];
    [enc setBuffer:diskBuffer offset:0 atIndex:13];
    [enc setBuffer:diskMaterialBuffer offset:0 atIndex:14];
    [enc setBuffer:pointLightBuffer offset:0 atIndex:15];
    [enc setBuffer:directionalLightBuffer offset:0 atIndex:16];
    [enc setBuffer:projectionLightBuffer offset:0 atIndex:17];
    [enc setBuffer:goniometricLightBuffer offset:0 atIndex:18];
    [enc setBuffer:envMarginalCDFBuffer offset:0 atIndex:19];
    [enc setBuffer:envConditionalCDFBuffer offset:0 atIndex:20];
    [enc setBuffer:ggxEnergyTableBuffer offset:0 atIndex:21];
    [enc setBuffer:pbrtEnvMarginalCDFBuffer offset:0 atIndex:22];
    [enc setBuffer:pbrtEnvConditionalCDFBuffer offset:0 atIndex:23];
    [enc setBuffer:lensElementBuffer offset:0 atIndex:24];
    [enc setBuffer:exitPupilBoundsBuffer offset:0 atIndex:25];
    [enc setBuffer:cylinderBuffer offset:0 atIndex:26];
    [enc setBuffer:cylinderMaterialBuffer offset:0 atIndex:27];
    [enc setBuffer:cloudMediumBuffer offset:0 atIndex:28];
    [enc setBuffer:rgbGridMediumBuffer offset:0 atIndex:29];
    [enc setBuffer:rgbGridDataBuffer offset:0 atIndex:30];
    // Mark the AS + its dependent primitive ASes as used so Metal
    // knows about the indirection - required for instance
    // acceleration structures referencing primitive ones (now three:
    // the room+Spot triangle mesh, the sphere's bounding-box
    // geometry, and Suzanne's own - referenced by TWO instances, but
    // only needs marking used once here, not once per instance).
    [enc useResource:primAS usage:MTLResourceUsageRead];
    [enc useResource:sphereAS usage:MTLResourceUsageRead];
    [enc useResource:suzanneAS usage:MTLResourceUsageRead];

    MTLSize gridSize = MTLSizeMake(width, height, 1);
    NSUInteger w = pipeline.threadExecutionWidth;
    NSUInteger h = pipeline.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [enc endEncoding];
    [renderCmd commit];
    [renderCmd waitUntilCompleted];
    if (renderCmd.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "Render dispatch failed: %s\n", renderCmd.error.localizedDescription.UTF8String);
        return false;
    }

    // --- Read back into `pixels` (post-processed and written to disk
    // by postProcessAndWrite(), below) --------------------------------
    pixels.resize(width * height * 4);
    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
    [outTexture getBytes:pixels.data() bytesPerRow:width * 4 * sizeof(float) fromRegion:region mipmapLevel:0];
    return true;
}
