// metal_poc_kernel.metal
// primaryRayKernel: one thread per pixel. It sets up the bundles of metal_poc_kernel_state.metal, then loops over the pixel's samples,
// tracing each path bounce by bounce (metal_poc_kernel_camera / _media / _surface / _bounce.metal), accumulates the filtered radiance and
// stops early when adaptive sampling says the pixel has converged.

kernel void primaryRayKernel(
    texture2d<float, access::write> outTexture [[texture(0)]],
    texture2d<float, access::sample> earthTexture [[texture(1)]],
    texture2d<float, access::sample> goniometricTexture [[texture(2)]],
    // A pbrt-loaded scene's own image-based LightSource "infinite" -
    // deliberately a SEPARATE texture from earthTexture above (never
    // repointing that one - see metal_poc.mm's own loadPbrtScene()
    // comment on why doing so would corrupt the hardcoded room's own
    // materialType-3 wall, which reads earthTexture for a completely
    // different purpose). Miss-path-only, same scope cut as the
    // constant-colour case (section 88) - see the miss-path code below.
    texture2d<float, access::sample> pbrtEnvTexture [[texture(3)]],
    // A pbrt-loaded scene's own real per-light goniometric/projection
    // profile images (section 98) - same "separate slot, never repoint
    // the room's own shared texture" reasoning as pbrtEnvTexture above.
    // Only ONE of each kind is supported (metal_poc.mm's own
    // havePbrtGoniometricImage/havePbrtProjectionImage comment); which
    // light in a NEE loop reads which texture is picked per-light via
    // GoniometricLight::usePbrtTexture/ProjectionLight::usePbrtTexture.
    texture2d<float, access::sample> pbrtGoniometricTexture [[texture(4)]],
    texture2d<float, access::sample> pbrtProjectionTexture [[texture(5)]],
    // A pbrt-loaded scene's own image-based AreaLightSource
    // ("string filename", section 105) - same separate-slot reasoning.
    texture2d<float, access::sample> pbrtAreaLightTexture [[texture(6)]],
    // A pbrt-loaded scene's own Diffuse/CoatedDiffuse material with a
    // "texture reflectance" bound to a bare "imagemap" Texture (section
    // 166) - same separate-slot reasoning as every other pbrt-loaded
    // image above (never repointing earthTexture, which materialType 3
    // still reads for the hardcoded room's own completely unrelated
    // wall/A4-Earth-sphere purpose). Only the FIRST such material in the
    // scene gets a real per-hit texture lookup (materialType 26) - a
    // second one, or a decode failure, falls back to `color` exactly as
    // if no texture were bound at all, the same "safe, working fallback
    // over dropping the material" tier every other pbrt-loaded image
    // here already uses.
    texture2d<float, access::sample> pbrtDiffuseTexture [[texture(7)]],
    texture2d<float, access::sample> pbrtTransmitTexture [[texture(8)]],
    instance_acceleration_structure accelStructure [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]],
    device const TriangleMaterial* triMaterials [[buffer(2)]],
    device const packed_float3* vertices [[buffer(3)]],
    device const TriangleMaterial* sphereMaterials [[buffer(4)]],
    device const SphereData* spheres [[buffer(5)]],
    intersection_function_table<instancing, triangle_data> functionTable [[buffer(6)]],
    device const packed_float3* normals [[buffer(7)]],
    device const packed_float2* uvs [[buffer(8)]],
    device const AreaLight* lights [[buffer(9)]],
    device const packed_float3* suzanneNormals [[buffer(10)]],
    device const TriangleMaterial* suzanneMaterials [[buffer(11)]],
    device const InstanceTransform* instanceTransforms [[buffer(12)]],
    device const DiskData* disks [[buffer(13)]],
    device const TriangleMaterial* diskMaterials [[buffer(14)]],
    device const PointLight* pointLights [[buffer(15)]],
    device const DirectionalLight* directionalLights [[buffer(16)]],
    device const ProjectionLight* projectionLights [[buffer(17)]],
    device const GoniometricLight* goniometricLights [[buffer(18)]],
    device const float* envMarginalCDF [[buffer(19)]],
    device const float* envConditionalCDF [[buffer(20)]],
    device const float* ggxEnergyTable [[buffer(21)]],
    // pbrtEnvTexture's own SEPARATE EnvDistribution2D CDFs (section 96) -
    // see Uniforms::pbrtEnvMapWidth's own comment.
    device const float* pbrtEnvMarginalCDF [[buffer(22)]],
    device const float* pbrtEnvConditionalCDF [[buffer(23)]],
    // Realistic camera's own lens/exit-pupil-bounds tables (D4/D8,
    // section 157) - see sampleRealisticCameraRay()'s own comment.
    device const LensElement* lensElements [[buffer(24)]],
    device const ExitPupilBounds* exitPupilBounds [[buffer(25)]],
    // A full (no phi-max sweep, no motion blur) cylinder primitive
    // (section 171) - see CylinderData's own comment (metal_poc_types.metal)
    // for why this is baked world-space rather than object-space-plus-
    // transform. Separate from cylinderIntersectionFunction's own
    // buffer(2) (the function table's own argument namespace) exactly
    // like spheres/disks above - see SphereData/DiskData's own comment
    // on that split.
    device const CylinderData* cylinders [[buffer(26)]],
    device const TriangleMaterial* cylinderMaterials [[buffer(27)]],
    // E2/section 178: materialType 29's own backing buffer - see
    // GpuCloudMedium's own comment (metal_poc_types.metal). Index into
    // this buffer comes from TriangleMaterial::conductorEta.x on the hit
    // sphere's own material, read via `sphereMaterials` (already bound
    // above at buffer(4)).
    device const GpuCloudMedium* cloudMediums [[buffer(28)]],
    // E4/section 179: materialType 30's own backing buffer, same
    // conductorEta.x-index convention as cloudMediums above. Its own
    // per-voxel R/G/B data lives in a SEPARATE flat buffer
    // (`rgbGridData` - GpuRgbGridMedium::dataOffset indexes into it),
    // since a whole grid's own voxel count can't fit inline the way
    // GpuCloudMedium's own scalar parameters do.
    device const GpuRgbGridMedium* rgbGridMediums [[buffer(29)]],
    device const float* rgbGridData [[buffer(30)]],
    texture2d<uint, access::write> censusTexture [[texture(9)]],
    // Live Preview: the first surface point (or a miss: all zero) each pixel sees, in this scene's internal units - the host turns
    // it back into the GUI's units and hands it to the temporal reprojection. A 1x1 dummy when uniforms.liveWorldPos == 0.
    texture2d<float, access::write> worldPosTexture [[texture(10)]],
    uint2 tidInBand [[thread_position_in_grid]])
{
    // Row-band dispatch (see Uniforms::rowOffset's own comment) - this
    // dispatch only ever covers rows [0, bandHeight) of the actual
    // image; `tid` (used everywhere below, exactly as before this
    // field existed) is the REAL full-image pixel coordinate, computed
    // once here so nothing past this line needs to know bands exist at
    // all - a single-dispatch (whole-image) render still works
    // unchanged, since `rowOffset` defaults to 0.
    uint2 tid = uint2(tidInBand.x, tidInBand.y + uniforms.rowOffset);

    // Bilinear + repeat/wrap: the standard choice for a UV-mapped photo
    // texture (the earth-map's own left/right edges are meant to tile
    // seamlessly at the date line) - constexpr so it's resolved at
    // compile time, same as every Metal sample/tutorial's own pattern for
    // a sampler that never needs to change at runtime.
    constexpr sampler textureSampler(coord::normalized, address::repeat, filter::linear);
    if (tid.x >= uniforms.width || tid.y >= uniforms.height) return;

    // Area lights: real geometry (added via addQuad() host-side with a
    // matching AreaLight entry, see metal_poc.mm) with nonzero
    // TriangleMaterial::emission on their triangles, not a separate light
    // type - `lights`/`uniforms.lightCount` (bound above) is the light
    // LIST this project's own CPU src/TheRestOfYourLife/*light_sampler*.h
    // is the equivalent abstraction for, replacing the single hardcoded
    // light this POC started with (steps 4/5).


    // No assume_geometry_type() hint here (step 1/2 had one, for
    // triangle-only) - the scene now mixes triangle geometry (the room)
    // with bounding-box/custom geometry (the sphere) across two
    // instances in the same instance_acceleration_structure, so the
    // intersector genuinely needs to handle both.
    intersector<instancing, triangle_data> isect;
    // Every geometry in the acceleration structures is built opaque and is only ever triangles or bounding boxes
    // (spheres/disks/cylinders), so tell the traversal so: no opacity checks, no curve/motion paths. Measured on an
    // M2: A1 1.54 s -> 1.27 s, B2 1.84 s -> 1.55 s, images unchanged. (The shadow-ray intersector copies this one.)
    isect.assume_geometry_type(geometry_type::triangle | geometry_type::bounding_box);
#ifdef METAL_ALPHA_MASKS
    // A scene with an alpha-cutout mask (set when the library is compiled, see dsCompileLibrary): the main triangles are non-opaque and
    // alphaTriangleIntersectionFunction decides each hit, so the geometry's own opacity flags must be honoured.
    isect.force_opacity(forced_opacity::none);
#else
    isect.force_opacity(forced_opacity::opaque);
#endif

    // Everything the kernel is bound to, in one bundle for the functions below (metal_poc_kernel_state.metal).
    const KernelRes R{earthTexture, goniometricTexture, pbrtEnvTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, pbrtDiffuseTexture, pbrtTransmitTexture, accelStructure, &uniforms, triMaterials, vertices, sphereMaterials, spheres, functionTable, normals, uvs, lights, suzanneNormals, suzanneMaterials, instanceTransforms, disks, diskMaterials, pointLights, directionalLights, projectionLights, goniometricLights, envMarginalCDF, envConditionalCDF, ggxEnergyTable, pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, lensElements, exitPupilBounds, cylinders, cylinderMaterials, cloudMediums, rgbGridMediums, rgbGridData, isect};
    // What the sample in flight carries from bounce to bounce.
    PathState P;
    PATH_STATE_ALIASES(P)
    rngState = tid.x * 9781u + tid.y * 6271u + uniforms.frameSeed * 26699u + 1u;

    float3 accumColor = float3(0.0);
    // Single-pass adaptive sampling - ported from this project's own CPU
    // integrator (src/shared/adaptive_sampling.h's own
    // pixel_convergence::has_converged(), camera.h's render loop): once
    // a pixel's own running per-sample LUMINANCE estimate is confident
    // enough (relative standard error below kAdaptiveThreshold) that
    // more samples wouldn't change its mean much, stop early instead of
    // spending this pixel's full samplesPerPixel budget on it - a
    // genuinely converged sky/shadow/matte-wall region needs far fewer
    // samples than a noisy caustic or grazing-light region does. Unlike
    // the CPU's own version, this fires within ONE kernel dispatch's own
    // per-pixel loop (no cross-dispatch/cross-pixel budget
    // reallocation), the same single-thread-per-pixel structure the CPU
    // integrator's own per-pixel loop already has - ported faithfully,
    // not reinvented, using Welford's online algorithm (the same
    // mean/M2 update VarianceEstimator uses) rather than the CPU's own
    // templated class, since this is plain MSL, not C++.
    // `uniforms.adaptiveSampling == 0` (every scene before this one)
    // skips the convergence check entirely below - a true no-op, this
    // sample count and this loop behave EXACTLY as before.
    uint convergedCount = 0;
    float convergedMean = 0.0;
    float convergedM2 = 0.0;

    // Running filter-weight denominator for the accumColor normalization
    // below (accumColor /= weightSum, not /= actualSamples) - see
    // sampleFilterPosition()'s own comment and this loop's own
    // `accumColor += radiance * filterSample.weight` line.
    float weightSum = 0.0;
    // PATH REGENERATION. The 32 lanes of a SIMD group used to run sample s together, so every group-bounce cost as much as
    // its longest path (measured with METAL_CENSUS: only ~50% of lanes active per bounce even in A1). Now a lane whose path
    // ends immediately starts its next sample instead of waiting; each lane still draws its random numbers in exactly the
    // same order (sample after sample, bounce after bounce), so the image is unchanged.
    uint s = 0;                  // index of the sample whose path is in flight (or the next one to start)
    bool pathActive = false;
    bool converged = false;
    // Live Preview: where the ray through this pixel's CENTRE first hits (or that it misses), for the GUI's temporal reprojection. The
    // pixel centre, not the jittered sample the path tracer traces: reprojection compares this position with one projected from a
    // later camera, and on an oblique surface a one-pixel jitter is tens of scene units - far above its match tolerance. One extra
    // primary ray per pixel per frame; pinhole camera only (any other camera leaves "no data").
    if (uniforms.liveWorldPos != 0u) {
        float4 wp = float4(0.0);
        if (uniforms.cameraOrthographic == 0u && uniforms.cameraSpherical == 0u && uniforms.cameraRealistic == 0u && uniforms.hasCameraOrbitBlur == 0u) {
            float2 c = (float2(tid) + 0.5) / float2(uniforms.width, uniforms.height) * 2.0 - 1.0;
            c.y = -c.y;
            c.x *= uniforms.aspect;
            c *= uniforms.tanHalfFov;
            ray cr;
            cr.origin = float3(uniforms.cameraPos);
            cr.direction = normalize(float3(uniforms.cameraForward) + c.x * float3(uniforms.cameraRight) + c.y * float3(uniforms.cameraUp));
            cr.min_distance = 0.001f;
            cr.max_distance = 1e6f;
            SpherePayload centrePayload{0.0, false};
            intersection_result<instancing, triangle_data> cres = isect.intersect(cr, accelStructure, functionTable, centrePayload);
            if (cres.type != intersection_type::none) wp = float4(cr.origin + cr.direction * cres.distance, 1.0);
        }
        worldPosTexture.write(wp, tid);
    }
    while (true) {
        // Lockstep mode only needs to know whether any lane of the group is still mid-path (a SIMD vote over the lanes
        // that are still in this loop - every lane reaches this line together).
        const bool groupBusy = simd_any(pathActive);
        if (!pathActive) {
        if (s >= uniforms.samplesPerPixel || converged) break;
        if (uniforms.pathRegen == 0u && groupBusy) continue;   // lockstep: wait for the lanes still tracing
        const float cameraWeight = generateCameraRay(R, P, tid);
        initPathState(R, P, cameraWeight, s);
        pathActive = true;
        }
        bool continuePath = false;
        // pbrt-v4's (and the CPU's) depth convention: maxDepth scattering vertices, then ONE more step that only adds the emission (or sky) the last
        // sampled ray sees - weighted by the MIS share against the light sample taken at the vertex that sampled it - and stops: no light
        // sampling, no scattering. Without that step the BSDF-sampled share of the last vertex's light sample was simply lost (a diffuse sphere under
        // a uniform sky read 0 at depth 1 instead of 0.5; a room lit only indirectly read 3-19% dark).
        if (depth <= uniforms.maxDepth) continuePath = traceBounce(R, P);
        if (continuePath) {
            ++depth;
            if (depth <= uniforms.maxDepth) continue;   // next bounce of this path (other lanes may be starting new ones)
        }
        // The path is finished (terminated, or out of depth): finalize this sample below.

        if (cenRec) censusTexture.write(uint4(cen0, cen1, cenN, 0u), tid);

        // Firefly clamp - see Uniforms::fireflyClamp (metal_poc_gpu_types.h).
        // Applied once per SAMPLE, here, not per NEE contribution inside
        // the bounce loop above - simpler (one clamp site, not scattered
        // across every light-sampling branch) and still catches the same
        // outliers, since an extreme single-bounce contribution dominates
        // this sample's own total `radiance` regardless of which branch
        // produced it.
        // The threshold is the scene's own Film "maxcomponentvalue" (pbrt's default
        // is unbounded, as on CPU), NOT a fixed constant (the old one, 20, was
        // tuned for the old hand-authored room), and a pbrt scene can
        // legitimately exceed it (the ceiling right above a point light reads
        // 100-260), which made Metal render such scenes ~35% too dark.
        float fireflyLimit = uniforms.fireflyClamp > 0.0 ? uniforms.fireflyClamp : INFINITY;
        if (pathTouchedHair) fireflyLimit = min(fireflyLimit, 40.0f);
        float sampleMax = max(radiance.x, max(radiance.y, radiance.z));
        if (sampleMax > fireflyLimit) {
            radiance *= fireflyLimit / sampleMax;
        }
        // Filter-weighted accumulation - mirrors CPU's camera.h render
        // loop exactly (weighted_color += w*sample; weight_sum += w;
        // pixel = weighted_color/weight_sum), not a plain per-sample
        // average - see sampleFilterPosition()'s own comment for why
        // `filterSample.weight` is a scene-wide constant in practice,
        // not a per-sample-varying one; accumulated as a running sum
        // here anyway (not hoisted out as a single multiply at the end)
        // to match that reference algorithm's own shape exactly, the
        // same "port faithfully, don't hand-optimize away a generality
        // this port doesn't need yet" choice the adaptive-sampling
        // block just below already makes for CPU's Welford update.
        accumColor += radiance * filterSample.weight;
        weightSum += filterSample.weight;

        // Adaptive-sampling convergence check - see this kernel's own
        // opening comment. Welford's online update (matching
        // VarianceEstimator::Add()'s own formula exactly) on THIS
        // sample's own luminance, then the same has_converged() test the
        // CPU integrator uses: sample variance (M2/(n-1), undefined
        // below n=2, hence the `> 1u` guard) turned into a standard
        // error, compared against the running mean - relative, so scale-
        // invariant regardless of this scene's own absolute brightness.
        if (uniforms.adaptiveSampling != 0u) {
            float lum = 0.2126 * radiance.x + 0.7152 * radiance.y + 0.0722 * radiance.z;
            convergedCount += 1;
            float delta = lum - convergedMean;
            convergedMean += delta / float(convergedCount);
            float delta2 = lum - convergedMean;
            convergedM2 += delta * delta2;
            if (convergedCount >= kAdaptiveMinSamples) {
                bool blackConverged = convergedMean < kAdaptiveBlackFloor;
                bool relativeConverged = false;
                if (!blackConverged && convergedCount > 1u) {
                    float variance = convergedM2 / float(convergedCount - 1u);
                    float standardError = sqrt(variance / float(convergedCount));
                    relativeConverged = (standardError / convergedMean) < uniforms.adaptiveThreshold;
                }
                if (blackConverged || relativeConverged) {
                    // weightSum/accumColor already reflect only the
                    // samples actually taken through this break (both
                    // accumulated once per completed iteration above,
                    // never for one that hasn't run yet) - no separate
                    // "how many samples ran" count needed for the
                    // normalization below, unlike before this function
                    // divided by a plain actualSamples/sample-count.
                    converged = true;
                }
            }
        }
        ++s;
        pathActive = false;
    }

    // Normalize by filter weight sum, not actualSamples - mirrors
    // pbrt-v4/CPU's own film normalization (pixel.rgbSum/pixel.weightSum,
    // camera.h's own identical comment) exactly. Black (not a divide-by-
    // zero NaN) when weightSum is non-positive - unreachable today (every
    // filter kind's own tabulated integral is strictly positive, and a
    // scene can't reach this line with actualSamples==0), but matches
    // CPU's own explicit `(weight_sum > 0.0) ? ... : color(0,0,0)` guard
    // rather than relying on that being true.
    accumColor = (weightSum > 1e-6) ? (accumColor / weightSum) : float3(0.0);
    outTexture.write(float4(accumColor, 1.0), tid);
}
