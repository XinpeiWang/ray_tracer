// Divergence census probe (see Uniforms::debugCensus): packs the material/event code of each bounce of sample 0 into
// cen0/cen1 (4 codes of 8 bits each) and counts them in cenN. A no-op unless the census is on.
#define CENSUS_PUSH(code) if (cenRec) { uint c_ = (uint(code)) & 0xFFu; if (cenN < 4u) cen0 |= c_ << (8u * cenN); else if (cenN < 8u) cen1 |= c_ << (8u * (cenN - 4u)); ++cenN; }

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

    const float3 skyTop = float3(0.9, 0.95, 1.0);
    const float3 skyBottom = float3(0.3, 0.5, 0.9);

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
    isect.force_opacity(forced_opacity::opaque);

    uint rngState = tid.x * 9781u + tid.y * 6271u + uniforms.frameSeed * 26699u + 1u;

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
    uint depth = 0;
    MetalFilterSample filterSample;
    float shutterT = 0.0;
    float3 rayOrigin = float3(0.0), rayDir = float3(0.0);
    float3 throughput = float3(0.0), radiance = float3(0.0);
    bool specularBounce = true;
    float bsdfPdf = 0.0;
    float mediumSkippedDist = 0.0;
    bool pathTouchedHair = false;
    bool inGlass = false;
    float3 glassSigmaT3 = float3(0.0);
    float glassG = 0.0;
    float3 glassAlbedo = float3(1.0);
    int glassChan = -1;   // hero colour channel for a chromatic glass medium, else -1
    uint rgbChannel = kRgbChannelUnset;
    uint cen0 = 0u, cen1 = 0u, cenN = 0u;
    bool cenRec = false;
    while (true) {
        // Lockstep mode only needs to know whether any lane of the group is still mid-path (a SIMD vote over the lanes
        // that are still in this loop - every lane reaches this line together).
        const bool groupBusy = simd_any(pathActive);
        if (!pathActive) {
        if (s >= uniforms.samplesPerPixel || converged) break;
        if (uniforms.pathRegen == 0u && groupBusy) continue;   // lockstep: wait for the lanes still tracing
        // Filter-importance-sampled sub-pixel sample position - the
        // multi-sample loop's own antialiasing AND reconstruction-filter
        // reach in one draw (sampleFilterPosition()'s own comment);
        // without it every sample would retrace the exact same primary
        // ray. Replaces the old uniform-in-[0,1)-pixel jitter, which was
        // exactly a hardcoded 0.5-pixel-radius box filter - see section
        // 207, docs/METAL_GPU_FEASIBILITY.md.
        filterSample = sampleFilterPosition(uniforms, randFloat(rngState), randFloat(rngState));
        float2 pixelNDC = (float2(tid) + 0.5 + float2(filterSample.px, filterSample.py)) / float2(uniforms.width, uniforms.height);
        float2 screen = pixelNDC * 2.0 - 1.0;
        screen.y = -screen.y;
        screen.x *= uniforms.aspect;
        screen *= uniforms.tanHalfFov;

        // Shutter motion blur: this sample's own random point in time
        // over [0,1] decides how far along `cameraVelocity` the camera
        // has moved for THIS ray - drawn once per sample (not once per
        // pixel), same as the pixel jitter above, so different samples
        // genuinely see a moving camera rather than one shared static
        // offset re-jittered.
        shutterT = randFloat(rngState);
        // Real multi-element-lens camera's own per-sample weight (see
        // Uniforms::cameraRealistic's own comment) - 1.0 (a true no-op)
        // for every other camera mode, folded into `throughput`'s own
        // initial value below.
        float cameraWeight = 1.0;
        if (uniforms.cameraOrthographic != 0u) {
            // Orthographic (parallel-projection): every pixel's ray
            // shares the SAME direction (cameraForward) - `screen.x/y`
            // instead offsets the ray's ORIGIN across the screen
            // window, mirroring pbrt-v4 OrthographicCamera::GenerateRay()
            // exactly (Uniforms::cameraOrthographic's own comment).
            // `-screen.x` (NEGATED, unlike every perspective ray below) -
            // a real sign mismatch found via a mirrored first render,
            // not assumed: this loader's own `cameraRight` (shared by
            // every scene, including this one's own buildCornellBoxA1()
            // call) is built as `cross(forward, up)`, matching CPU's
            // OWN primary book-style camera (src/TheRestOfYourLife/
            // camera.h's `u = cross(vup, w)` where `w = -forward`,
            // algebraically the SAME `cross(forward, up)` sign). But
            // CPU's own D2/D6-D8 alt-camera path (src/shared/cameras.h's
            // `make_look_at()`, used ONLY for the orthographic/
            // spherical/realistic cameras, never the primary one) computes
            // `right = cross(up, forward)` instead - the OPPOSITE sign
            // from its own primary camera, a genuine internal
            // inconsistency between CPU's two separate camera-construction
            // code paths. Invisible for every perspective scene so far
            // (this loader's ray DIRECTION fan uses `cameraRight`, always
            // built the primary/book-style way, matching CPU's own
            // primary camera exactly), but D6 is the FIRST scene to
            // reuse `cameraRight` for an ORTHOGRAPHIC ray ORIGIN offset -
            // where it must instead match CPU's differently-signed ALT
            // camera, not the primary one.
            rayOrigin = float3(uniforms.cameraPos) + shutterT * float3(uniforms.cameraVelocity)
                        - screen.x * float3(uniforms.cameraRight)
                        + screen.y * float3(uniforms.cameraUp);
            rayDir = normalize(float3(uniforms.cameraForward));
        } else if (uniforms.cameraSpherical != 0u) {
            // Spherical (360-degree panorama), two mappings
            // (Uniforms::cameraSpherical's own comment) - `u`/`v` are the
            // SAME `pixelNDC.x/y` every other mode derives `screen` from,
            // used directly here (raw [0,1], BEFORE the `*2-1`/y-flip/
            // aspect/tanHalfFov transform those other modes need - this
            // mode has no screen window or FOV at all, it captures the
            // full sphere around one point).
            rayOrigin = float3(uniforms.cameraPos) + shutterT * float3(uniforms.cameraVelocity);
            if (uniforms.sphericalMappingEqualArea != 0u) {
                // EqualArea (D11, Uniforms::sphericalMappingEqualArea's
                // own comment): pbrt-v4 SphericalCamera::GenerateRay's
                // EqualArea branch - map (u,v) straight through
                // equalAreaSquareToSphere() (wrapped first, since a
                // per-sample jittered pixel coordinate can legitimately
                // land fractionally outside [0,1]^2, unlike a plain
                // image-index lookup). That function's own (wx,wy,wz)
                // treats Z as the mapping's pole axis; THIS camera's own
                // up axis is `cameraUp` (Y-like), not `cameraForward` -
                // swapping wy/wz on the way out (`w.z` feeds `cameraUp`,
                // `w.y` feeds `cameraForward`) re-homes the mapping's own
                // pole onto this camera's actual up axis, the same swap
                // gpu/optix/optix_device_helpers.h's own
                // generate_primary_ray() (CUDA's already-shipped mirror
                // of this exact camera model) makes for the identical
                // reason. Leading MINUS on the `cameraRight` term (`w.x`)
                // - the SAME sign correction the EquiRectangular branch
                // below and `cameraOrthographic`'s own branch above both
                // need (CPU's alt-camera path's own `right` is this
                // loader's `cameraRight` negated).
                float su = pixelNDC.x, sv = pixelNDC.y;
                wrapEqualAreaSquare(su, sv);
                float3 w = equalAreaSquareToSphere(su, sv);
                rayDir = normalize(-w.x * float3(uniforms.cameraRight)
                                    + w.z * float3(uniforms.cameraUp)
                                    + w.y * float3(uniforms.cameraForward));
            } else {
                // EquiRectangular (every scene before D11, including
                // D7/D8's own hand-authored ports): mirrors pbrt-v4
                // SphericalCamera::GenerateRay()'s own EquiRectangular
                // mapping exactly - `theta = pi*v, phi = 2*pi*u`.
                float theta = M_PI_F * pixelNDC.y;
                float phi = 2.0 * M_PI_F * pixelNDC.x;
                float sinTheta = sin(theta), cosTheta = cos(theta);
                // Same leading-MINUS sign correction the EqualArea branch
                // above needs, for the identical reason.
                rayDir = normalize(-sinTheta * cos(phi) * float3(uniforms.cameraRight)
                                    + cosTheta * float3(uniforms.cameraUp)
                                    + sinTheta * sin(phi) * float3(uniforms.cameraForward));
            }
        } else if (uniforms.cameraRealistic != 0u) {
            // Real multi-element-lens camera - see
            // sampleRealisticCameraRay()'s own comment for the full
            // mechanism. `cameraWeight` (default 1.0, every earlier
            // mode) folds the returned cos^4(theta)/(pdf*lensRearZ^2)
            // weight straight into this sample's own `throughput` below -
            // a fully-vignetted sample (function returns false) leaves
            // `cameraWeight` at 0, so its own throughput starts at
            // (0,0,0) and every subsequent `radiance +=` naturally
            // contributes nothing, no separate early-exit needed.
            float3 lensOrigin, lensDir;
            bool valid = sampleRealisticCameraRay(uniforms, lensElements, exitPupilBounds,
                                                   pixelNDC.x, pixelNDC.y, rngState,
                                                   float3(uniforms.cameraRight), float3(uniforms.cameraUp),
                                                   float3(uniforms.cameraForward), float3(uniforms.cameraPos),
                                                   lensOrigin, lensDir, cameraWeight);
            rayOrigin = lensOrigin + shutterT * float3(uniforms.cameraVelocity);
            rayDir = valid ? lensDir : float3(uniforms.cameraForward);
        } else if (uniforms.hasCameraOrbitBlur != 0u) {
            // Real (not translate-only) camera shutter motion blur
            // (D13, section 174, Uniforms::hasCameraOrbitBlur's own
            // comment) - recomputes the WHOLE forward/right/up basis
            // fresh from this sample's own interpolated origin toward
            // the fixed cameraLookAtBlur point, rather than translating
            // a FIXED basis (what the plain cameraVelocity path just
            // below does) - correct for a camera that keeps pointing at
            // the same subject while it moves, which the plain path
            // isn't.
            float3 origin0 = float3(uniforms.cameraPos);
            float3 origin1 = origin0 + float3(uniforms.cameraVelocity);
            float3 interpOrigin = mix(origin0, origin1, shutterT);
            float3 fwd = normalize(float3(uniforms.cameraLookAtBlur) - interpOrigin);
            float3 right = normalize(cross(fwd, float3(uniforms.cameraUpRawBlur)));
            float3 up = cross(right, fwd);
            rayOrigin = interpOrigin;
            rayDir = normalize(fwd + screen.x * right + screen.y * up);
        } else {
            rayOrigin = float3(uniforms.cameraPos) + shutterT * float3(uniforms.cameraVelocity);
            rayDir = normalize(float3(uniforms.cameraForward)
                                + screen.x * float3(uniforms.cameraRight)
                                + screen.y * float3(uniforms.cameraUp));
        }

        // Thin-lens depth of field: jitter the ray's ORIGIN across a disk
        // (the camera's simulated aperture) and re-aim it through the same
        // fixed point on the focus plane the un-jittered pinhole ray would
        // have hit - everything exactly at focusDistance stays pixel-sharp
        // (every jittered origin re-aims through the identical focus
        // point), everything nearer/farther blurs, because a jittered
        // origin's ray toward that SAME focus point diverges from the
        // pinhole ray more the further the actual hit surface is from the
        // focus plane. lensRadius == 0 (every earlier PR's own scenes)
        // skips this block entirely - see Uniforms' own comment.
        // ALSO skipped whenever cameraRealistic != 0u - that mode already
        // did its own, far more accurate per-element lens sampling above;
        // this simple thin-lens jitter would be a redundant (and wrong)
        // SECOND defocus applied on top. Never actually reachable today
        // (no scene sets both lensRadius>0 and cameraRealistic!=0), but
        // guarded explicitly rather than relying on that.
        if (uniforms.lensRadius > 0.0 && uniforms.cameraRealistic == 0u) {
            float2 apertureSample = (uniforms.apertureBlades >= 3u)
                ? samplePolygonAperture(uniforms.apertureBlades, rngState)
                : sampleUnitDisk(rngState);
            float2 lensSample = uniforms.lensRadius * apertureSample;
            float3 focusPoint = rayOrigin + rayDir * uniforms.focusDistance;
            rayOrigin += lensSample.x * float3(uniforms.cameraRight) + lensSample.y * float3(uniforms.cameraUp);
            rayDir = normalize(focusPoint - rayOrigin);
        }

        throughput = float3(cameraWeight);
        radiance = float3(0.0);
        // MIS bookkeeping across bounces: the light quad can be reached
        // two ways - explicit light sampling below (NEE), or landing on
        // it by chance via a Lambertian BSDF-sampled continuation ray.
        // Adding BOTH at full weight double-counts and adding only one
        // wastes the other strategy's lower-variance samples - the power
        // heuristic (beta=2, same choice this project's own CPU/GPU-OptiX
        // integrators make per docs/FEATURE_INVENTORY.md) blends them.
        // specularBounce starts true (a camera ray has no competing NEE
        // strategy to weight against, so its own hits - direct light
        // visibility - are always full weight, same as a mirror/glass
        // bounce's next hit); bsdfPdf is only meaningful when false.
        specularBounce = true;
        bsdfPdf = 0.0;
        // Distance this ray has already travelled since the last real bounce or
        // scatter, but which rayOrigin no longer reflects: stepping THROUGH a medium
        // sphere moves rayOrigin to its exit point. The emissive-hit MIS weight
        // needs the true distance from the previous bounce (that is what the NEE
        // strategy it is weighed against measured), not from the exit point.
        mediumSkippedDist = 0.0;
        // Set when this path bounced off a hair (materialType 31) surface. Metal's hair BSDF is float32,
        // where the near-cancelling logI0 terms occasionally produce absurd weights (single samples of
        // ~1e4 where CPU's double-precision hair peaks near 4); those compound across bounces. The scene-
        // driven Film maxcomponentvalue below is unbounded by default, so hair paths get the old fixed
        // per-sample clamp (40) - it is applied only to paths that touched hair, so every other material
        // stays unclamped.
        pathTouchedHair = false;
        // Glass-bounded medium state (a dielectric/thin/rough SPHERE whose material carries a homogeneous
        // medium in conductorEta = (sigma_t, g, 1) and transmitColor = albedo): true while this path
        // travels inside such a sphere. Updated after every dielectric event from the NEW direction vs the
        // outward normal. While true, the free-flight block below uses these parameters instead of the
        // camera medium's.
        inGlass = false;
        glassSigmaT3 = float3(0.0);
        glassG = 0.0;
        glassAlbedo = float3(1.0);
        glassChan = -1;   // hero colour channel for a chromatic glass medium, else -1
        // Recursive-backend dispersion state (materialType 22, B23/B24) -
        // kRgbChannelUnset means "no dispersive hit yet, this sample
        // stays full RGB". See shadeDispersiveDielectric()'s own
        // declaration comment for the full "stochastic channel
        // selection" rationale, ported from OptiX's own identical
        // per-path convention (gpu/optix/optix_raygen.h/
        // optix_device_helpers.h).
        rgbChannel = kRgbChannelUnset;

        cen0 = 0u; cen1 = 0u; cenN = 0u;
        cenRec = (uniforms.debugCensus != 0u) && (s == 0u);
        depth = 0;
        pathActive = true;
        }
        bool continuePath = false;
        if (depth < uniforms.maxDepth) do {
            ray r;
            r.origin = rayOrigin;
            r.direction = rayDir;
            r.min_distance = 0.001f;
            r.max_distance = 1e6f;

            // F11/section 167: this sample's own shutterT (drawn once per
            // sample above, already used for camera motion blur) doubles
            // as the payload every intersect() call below passes to
            // sphereIntersectionFunction, so a moving sphere is tested at
            // the SAME simulated instant this ray's own camera position
            // was sampled at - a shadow ray cast later in this same
            // sample reuses the identical payload for the same reason.
            // isShadowRay=false explicit override (SpherePayload::
            // isShadowRay's own comment, section 176) - this is the ONE
            // ray that must actually be able to ENTER a materialType-28
            // medium sphere. shadowSpherePayload (used by this same
            // sample's own fog-NEE shadow rays just below) is the
            // opposite: isShadowRay=true, so a medium sphere stays
            // invisible to a pure occlusion test.
            SpherePayload spherePayload{shutterT, false};
            SpherePayload shadowSpherePayload{shutterT, true};
            intersection_result<instancing, triangle_data> result =
                isect.intersect(r, accelStructure, functionTable, spherePayload);

            // Bounded, per-object homogeneous medium (A8, section 176) -
            // materialType 28, a sphere whose OWN interior is a real
            // participating medium instead of a solid surface (unlike the
            // global `fogSigmaT` medium just below, which fills the WHOLE
            // scene's interior up to whatever it hits first). Mirrors
            // gpu/optix/scene_builder.cpp's own build_cornell_smoke_gpu()
            // precedent exactly: CPU's own two rotated/translated BOXES
            // have no Metal (or OptiX) primitive to represent directly, so
            // both back-ends approximate each box as a sphere instead - an
            // accepted, already-documented CPU/GPU divergence, not a new
            // approximation invented here.
            //
            // The SAME free-flight sampling technique the global fog just
            // below already uses (this whole kernel's own comment on it
            // applies here identically), just bounded between the
            // sphere's own ENTRY distance (`result.distance`, already
            // known) and its EXIT distance - computed analytically, not
            // via a second intersect() call: a ray's two roots of
            // |O+tD-C|^2=r^2 sum to `2*dot(C-O,D)` (D already normalised,
            // so the quadratic's leading coefficient is exactly 1), so the
            // farther root is just that sum minus the already-known nearer
            // one - no extra acceleration-structure traversal needed.
            // `mat.color`/`mat.ior`/`mat.roughness` carry this sphere's
            // OWN albedo/sigmaT/HG-asymmetry-g (TriangleMaterial's usual
            // "one scalar slot, per-materialType meaning" reuse), letting
            // several differently-tinted/dense medium spheres coexist in
            // one scene, unlike the single shared `uniforms.fogSigmaT`.
            //
            // Real NEE to the scene's own registered area lights from a
            // scatter point inside this medium - the SAME sampleAreaLight()/
            // area-to-solid-angle/MIS machinery the global fog's own NEE
            // block just below already uses (mirrored, not duplicated
            // blindly: only the area-light case, since that's the only
            // light kind A8 - the one scene exercising this so far -
            // actually registers; point/directional/projection/
            // goniometric lights would need the identical treatment the
            // day a scene combining a bounded medium with one of those
            // shows up, not attempted here). A first version of this
            // shipped with NO NEE at all here (phase-sampled continuation
            // only, the same "visible but not NEE-sampled" tier sphere/
            // disk/cylinder-shaped area LIGHTS already established) -
            // rendering and comparing against `--cpu` showed why that
            // tier doesn't transfer to a participating MEDIUM the same
            // way: a light SOURCE is often reached directly or after one
            // bounce, but a scatter point deep in fog reaching a small
            // ceiling light by pure phase-sampled chance is a far
            // lower-probability event, so the smoke rendered as sparse
            // black flecks on an otherwise-empty room instead of
            // recognisable haze - a real correctness gap, not a
            // cosmetic one, fixed properly rather than shipped.
            const float3 mediumOriginBefore = rayOrigin;
            bool scatteredInMedium = false;
            bool passedThroughMediumSphere = false;
            if (result.type == intersection_type::bounding_box && result.geometry_id == 0u) {
                uint mediumPrimId = result.primitive_id;
                TriangleMaterial mediumMat = sphereMaterials[mediumPrimId];
                if (mediumMat.materialType == 28u) {
                    shadeHomogeneousMediumSphere(mediumMat, mediumPrimId, result.distance,
                        spheres, shutterT, lights, uniforms, pbrtAreaLightTexture, textureSampler,
                        isect, accelStructure, functionTable, shadowSpherePayload,
                        rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState,
                        scatteredInMedium, passedThroughMediumSphere);
                } else if (mediumMat.materialType == 29u) {
                    shadeCloudMediumSphere(mediumMat, result.distance,
                        cloudMediums, lights, uniforms, pbrtAreaLightTexture, textureSampler,
                        isect, accelStructure, functionTable, shadowSpherePayload,
                        rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState,
                        scatteredInMedium, passedThroughMediumSphere);
                } else if (mediumMat.materialType == 30u) {
                    shadeRgbGridMediumSphere(mediumMat, result.distance,
                        rgbGridMediums, rgbGridData, lights, uniforms, pbrtAreaLightTexture, textureSampler,
                        isect, accelStructure, functionTable, shadowSpherePayload,
                        rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState,
                        scatteredInMedium, passedThroughMediumSphere);
                }
            }

            if (passedThroughMediumSphere) mediumSkippedDist += length(rayOrigin - mediumOriginBefore);
            else if (scatteredInMedium) mediumSkippedDist = 0.0;

            // Homogeneous-medium free-flight distance sampling: draws a
            // random scattering distance from the medium's own
            // transmittance distribution (t = -ln(1-u)/sigmaT) and
            // compares it against the surface hit's own distance. This
            // ONE stochastic comparison is what makes BOTH "reached the
            // surface without scattering" and "scattered partway there"
            // come out unbiased with NO extra transmittance/pdf-ratio
            // multiplier needed in either case - a well-known result (the
            // pdf of sampling t this way, p(t) = sigmaT*exp(-sigmaT*t),
            // exactly cancels the extinction term(s) either way):
            //   scatter event (t < surfaceDist): weight = sigmaS*T(t)/p(t)
            //     = sigmaS/sigmaT (the albedo below, nothing else)
            //   reached surface (t >= surfaceDist): weight =
            //     T(surfaceDist)/P(t>=surfaceDist) = 1 exactly (survival
            //     probability under an exponential distribution IS the
            //     transmittance) - so the existing surface-shading code
            //     below needs NO changes at all for this case.
            //
            // Gated on an ACTUAL surface hit existing at all
            // (result.type != none) - the fog fills the scene's INTERIOR
            // (bounded implicitly by the room's own geometry, see this
            // struct's own comment), not empty space beyond a miss. An
            // earlier version of this code used FLT_MAX as a miss ray's
            // own "surface distance," which is a bug, not a deliberately
            // unbounded medium: since a sampled t is a finite real number
            // with probability 1, `t < FLT_MAX` is true for EVERY miss
            // ray, meaning no ray could ever actually reach the sky/
            // environment-map code below once fog was enabled at all -
            // every escaping ray incorrectly kept "scattering" in a
            // medium that should have already ended at the scene's own
            // boundary. Invisible in this POC's own default scene (the
            // room's 5 closed walls mean almost every PRIMARY ray already
            // hits something at 40 degrees FOV - only secondary/GI
            // bounces reflecting out through the open front ever actually
            // missed, a small enough fraction to not read as an obvious
            // artifact), but a real correctness bug, caught by this PR's
            // own wide-FOV/pulled-back verification render for the
            // environment-map feature - that render came back an
            // unexplained near-black speckled mess, and tracing why
            // surfaced this.
            // UPDATE: the fog is now only ever enabled by a pbrt scene's camera medium
            // (the hand-authored room is gone), and a pbrt camera medium extends to
            // infinity: a ray that escapes the scene still scatters with probability 1
            // (CPU does the same). So a miss ray gets an unbounded free-flight limit
            // instead of being skipped - without this the sky region of a foggy scene
            // (E10) rendered black instead of haze-lit.
            if (!scatteredInMedium && !passedThroughMediumSphere && (inGlass || uniforms.fogSigmaT > 0.0)) {
                const float curSigmaT = inGlass ? (glassChan >= 0 ? glassSigmaT3[glassChan] : glassSigmaT3.x) : uniforms.fogSigmaT;
                const float curG = inGlass ? glassG : uniforms.fogAsymmetryG;
                const float3 curAlbedo = inGlass ? glassAlbedo : float3(uniforms.fogAlbedo);
                // Transmittance along shadow rays: a glass sphere's own medium is attenuated stochastically in
                // sphereIntersectionFunction, so only the camera fog is applied analytically here.
                const float shadowSigmaT = inGlass ? 0.0f : uniforms.fogSigmaT;
                shadowSpherePayload.shadowChannel = inGlass ? glassChan : -1;
                float surfaceDist = (result.type == intersection_type::none) ? INFINITY : result.distance;
                float u = randFloat(rngState);
                float t = sampleFreePathDistance(u, curSigmaT);
                if (t < surfaceDist) {
                    scatteredInMedium = true;
                    float3 scatterPoint = rayOrigin + rayDir * t;

                    // NEE from the scatter point - same sampleAreaLight()/
                    // area-to-solid-angle/MIS machinery every surface
                    // material's own NEE branch already uses, with the
                    // Henyey-Greenstein phase function VALUE (no cosine
                    // term - a volume scattering event has no surface to
                    // cosine-weight against, unlike a BRDF) standing in
                    // for the BSDF value. `wo` (direction back toward
                    // where this ray came from) is captured BEFORE
                    // `rayDir` gets overwritten below by the sampled
                    // continuation direction, same `wo` convention the
                    // GGX conductor code uses. `exp(-sigmaT*dist)`
                    // attenuates this shadow ray's own contribution by the
                    // medium's transmittance along ITS length too - the
                    // free-flight sampling above only accounts for the
                    // PRIMARY ray's path, a shadow ray is a separate,
                    // deterministic occlusion test that needs this factor
                    // applied explicitly or it would silently ignore the
                    // fog lying between the scatter point and the light.
                    float3 wo = -rayDir;
                    LightSample ls = sampleAreaLight(lights, uniforms.lightCount, rngState, pbrtAreaLightTexture, textureSampler);
                    float3 toLight = ls.point - scatterPoint;
                    float distSq = dot(toLight, toLight);
                    float dist = sqrt(distSq);
                    float3 wi = toLight / dist;
                    float cosLight = dot(ls.normal, -wi);
                    if ((cosLight > 0.0 || (ls.twoSided != 0.0 && cosLight < 0.0))) {
                        ray shadowRay;
                        shadowRay.origin = scatterPoint;
                        shadowRay.direction = wi;
                        shadowRay.min_distance = 0.001f;
                        shadowRay.max_distance = dist - 0.002f;
                        intersection_result<instancing, triangle_data> shadowResult =
                            traceShadowAnyP(isect, shadowRay, accelStructure, functionTable, shadowSpherePayload);
                        if (shadowResult.type == intersection_type::none) {
                            float pdfSolidAngle = (distSq / (ls.area * abs(cosLight))) * ls.pmf;
                            // HG's own sampling pdf for direction wi EQUALS
                            // its own phase function value at the same
                            // cosTheta - a defining property (the phase
                            // function IS already a normalized pdf over
                            // the sphere), so the same call serves both as
                            // the "BSDF" value AND its own competing MIS
                            // pdf, no separate pdf expression needed the
                            // way a surface BRDF's importance-sampled pdf
                            // usually differs from its raw value.
                            float phaseValue = henyeyGreensteinPhase(dot(wo, wi), curG);
                            float weight = (pdfSolidAngle * pdfSolidAngle)
                                / (pdfSolidAngle * pdfSolidAngle + phaseValue * phaseValue);
                            float transmittance = exp(-shadowSigmaT * dist);
                            // `* curAlbedo`: this scattering
                            // event's own albedo weight - the SAME real bug
                            // materialType 28's own NEE block once had
                            // (section 176's own comment on that fix, found
                            // while diagnosing A8), left unfixed here at the
                            // time since the default fogAlbedo
                            // (0.85,0.88,0.95) makes the omission a sub-5%
                            // colour error, invisible in practice - fixed
                            // properly now that it's been flagged. Without
                            // this, every fog NEE contribution below (this
                            // one and all 4 delta-light ones) rendered as
                            // the LIGHT's own colour with no tint from the
                            // fog's own albedo at all.
                            radiance += throughput * curAlbedo * phaseValue * ls.emission * transmittance
                                        / pdfSolidAngle * weight;
                        }
                    }

                    // Point lights: summed unconditionally, not picked,
                    // full weight, no MIS - see PointLight's own comment.
                    for (uint pli = 0; pli < uniforms.pointLightCount; ++pli) {
                        PointLight pl = pointLights[pli];
                        float3 toPointLight = float3(pl.position) - scatterPoint;
                        float plDistSq = dot(toPointLight, toPointLight);
                        float plDist = sqrt(plDistSq);
                        float3 plWi = toPointLight / plDist;
                        ray plShadowRay;
                        plShadowRay.origin = scatterPoint;
                        plShadowRay.direction = plWi;
                        plShadowRay.min_distance = 0.001f;
                        plShadowRay.max_distance = plDist - 0.002f;
                        intersection_result<instancing, triangle_data> plShadowResult =
                            traceShadowAnyP(isect, plShadowRay, accelStructure, functionTable, shadowSpherePayload);
                        if (plShadowResult.type == intersection_type::none) {
                            float plPhaseValue = henyeyGreensteinPhase(dot(wo, plWi), curG);
                            float plTransmittance = exp(-shadowSigmaT * plDist);
                            float plSpot = spotLightFalloff(-plWi, float3(pl.direction), pl.cosOuterAngle, pl.cosInnerAngle);
                            radiance += throughput * curAlbedo * plPhaseValue * float3(pl.emission) * plSpot * plTransmittance / plDistSq;
                        }
                    }

                    // Directional lights: summed unconditionally, not
                    // picked - see DirectionalLight's own comment. No
                    // distance falloff, but fog attenuation now DOES
                    // apply, using rayBoxExitDistance() as the real path
                    // length - see that function's own comment.
                    for (uint dli = 0; dli < uniforms.directionalLightCount; ++dli) {
                        DirectionalLight dl = directionalLights[dli];
                        float3 dlWi = normalize(-float3(dl.direction));
                        ray dlShadowRay;
                        dlShadowRay.origin = scatterPoint;
                        dlShadowRay.direction = dlWi;
                        dlShadowRay.min_distance = 0.001f;
                        dlShadowRay.max_distance = kDirectionalLightMaxDistance;
                        intersection_result<instancing, triangle_data> dlShadowResult =
                            traceShadowAnyP(isect, dlShadowRay, accelStructure, functionTable, shadowSpherePayload);
                        if (dlShadowResult.type == intersection_type::none) {
                            float dlPhaseValue = henyeyGreensteinPhase(dot(wo, dlWi), curG);
                            float dlExitDist = inGlass ? 0.0f : rayBoxExitDistance(scatterPoint, dlWi, kRoomBoundsMin, kRoomBoundsMax);
                            float dlTransmittance = exp(-shadowSigmaT * dlExitDist);
                            radiance += throughput * curAlbedo * dlPhaseValue * float3(dl.emission) * dlTransmittance;
                        }
                    }

                    // Projection ("slide projector") lights: same
                    // unconditional-sum, no-MIS shape as every other
                    // delta light here - see ProjectionLight's own
                    // comment.
                    for (uint pji = 0; pji < uniforms.projectionLightCount; ++pji) {
                        ProjectionLight pj = projectionLights[pji];
                        float3 toProjLight = float3(pj.position) - scatterPoint;
                        float pjDistSq = dot(toProjLight, toProjLight);
                        float pjDist = sqrt(pjDistSq);
                        float3 pjWi = toProjLight / pjDist;
                        float3 pjRadiance = projectionLightRadiance(-pjWi, pj.forward, pj.right, pj.up,
                                                                     pj.tanHalfFovX, pj.tanHalfFovY, pj.scale,
                                                                     (pj.usePbrtTexture != 0u ? pbrtProjectionTexture : earthTexture), textureSampler);
                        if (any(pjRadiance > float3(0.0))) {
                            ray pjShadowRay;
                            pjShadowRay.origin = scatterPoint;
                            pjShadowRay.direction = pjWi;
                            pjShadowRay.min_distance = 0.001f;
                            pjShadowRay.max_distance = pjDist - 0.002f;
                            intersection_result<instancing, triangle_data> pjShadowResult =
                                traceShadowAnyP(isect, pjShadowRay, accelStructure, functionTable, shadowSpherePayload);
                            if (pjShadowResult.type == intersection_type::none) {
                                float pjPhaseValue = henyeyGreensteinPhase(dot(wo, pjWi), curG);
                                float pjTransmittance = exp(-shadowSigmaT * pjDist);
                                radiance += throughput * curAlbedo * pjPhaseValue * pjRadiance * pjTransmittance / pjDistSq;
                            }
                        }
                    }

                    // Goniometric lights: same unconditional-sum, no-MIS
                    // shape as every other delta light here - see
                    // GoniometricLight's own comment.
                    for (uint gli = 0; gli < uniforms.goniometricLightCount; ++gli) {
                        GoniometricLight gl = goniometricLights[gli];
                        float3 toGoniLight = float3(gl.position) - scatterPoint;
                        float glDistSq = dot(toGoniLight, toGoniLight);
                        float glDist = sqrt(glDistSq);
                        float3 glWi = toGoniLight / glDist;
                        float3 glRadiance = goniometricLightRadiance(-glWi, gl.forward, gl.right, gl.up,
                                                                      gl.emission, gl.scale,
                                                                      (gl.usePbrtTexture != 0u ? pbrtGoniometricTexture : goniometricTexture), textureSampler);
                        if (any(glRadiance > float3(0.0))) {
                            ray glShadowRay;
                            glShadowRay.origin = scatterPoint;
                            glShadowRay.direction = glWi;
                            glShadowRay.min_distance = 0.001f;
                            glShadowRay.max_distance = glDist - 0.002f;
                            intersection_result<instancing, triangle_data> glShadowResult =
                                traceShadowAnyP(isect, glShadowRay, accelStructure, functionTable, shadowSpherePayload);
                            if (glShadowResult.type == intersection_type::none) {
                                float glPhaseValue = henyeyGreensteinPhase(dot(wo, glWi), curG);
                                float glTransmittance = exp(-shadowSigmaT * glDist);
                                radiance += throughput * curAlbedo * glPhaseValue * glRadiance * glTransmittance / glDistSq;
                            }
                        }
                    }

                    float3 newDir = sampleHenyeyGreenstein(wo, curG, rngState);
                    rayDir = newDir;
                    rayOrigin = scatterPoint;
                    throughput *= curAlbedo;
                    bsdfPdf = henyeyGreensteinPhase(dot(wo, newDir), curG);
                    specularBounce = false;
                }
            }

            if (scatteredInMedium) mediumSkippedDist = 0.0;  // global-fog scatter: new segment
            if (scatteredInMedium) { CENSUS_PUSH(100u); }
            if (!scatteredInMedium && !passedThroughMediumSphere) {
            if (result.type == intersection_type::none) {
                CENSUS_PUSH(255u);
                if (uniforms.useEnvironmentMap != 0u) {
                    float2 envUV = equirectangularUV(normalize(rayDir));
                    float3 envColor = earthTexture.sample(textureSampler, envUV).rgb;
                    // MIS weight against the environment-light NEE
                    // strategy's own pdf for this EXACT escaping
                    // direction (section 71) - this ray was BSDF-
                    // sampled, not env-light-sampled, so `bsdfPdf`
                    // (recorded by whichever material's own shading
                    // function ran last bounce) is the competing
                    // strategy's own pdf here, same "check for a hit
                    // reachable two ways" MIS shape the area light's own
                    // direct-hit weight below already uses. `specularBounce`
                    // (a delta/specular material has no NEE strategy to
                    // weight against at all) or `envMapWidth == 0` (no
                    // NEE strategy exists to double-count against in the
                    // first place) both mean full weight, unweighted -
                    // exactly the same two escape hatches the area
                    // light's own weight below already has.
                    float envMissWeight = 1.0;
                    if (!specularBounce && uniforms.envMapWidth > 0u) {
                        float pdfEnv = pdfEnvironmentDirection(envMarginalCDF, envConditionalCDF,
                                                                int(uniforms.envMapWidth), int(uniforms.envMapHeight),
                                                                normalize(rayDir));
                        envMissWeight = (bsdfPdf * bsdfPdf) / (bsdfPdf * bsdfPdf + pdfEnv * pdfEnv);
                    }
                    radiance += throughput * envColor * envMissWeight;
                } else if (uniforms.pbrtHasImageEnvLight != 0u) {
                    // A pbrt-loaded scene's own image-based infinite
                    // light. Now MIS-weighted against the NEE strategy
                    // added in section 96 (mirrors the useEnvironmentMap
                    // arm above exactly) - every material's own shading
                    // function that does NEE now also samples
                    // pbrtEnvTexture directly via pbrtEnvMapWidth, so a
                    // BSDF-sampled ray escaping toward it needs the same
                    // double-count protection.
                    float2 pbrtEnvUV = equirectangularUV(normalize(rayDir));
                    float3 pbrtEnvColorSample = pbrtEnvTexture.sample(textureSampler, pbrtEnvUV).rgb;
                    float pbrtEnvMissWeight = 1.0;
                    if (!specularBounce && uniforms.pbrtEnvMapWidth > 0u) {
                        float pdfPbrtEnv = pdfEnvironmentDirection(pbrtEnvMarginalCDF, pbrtEnvConditionalCDF,
                                                                    int(uniforms.pbrtEnvMapWidth), int(uniforms.pbrtEnvMapHeight),
                                                                    normalize(rayDir));
                        pbrtEnvMissWeight = (bsdfPdf * bsdfPdf) / (bsdfPdf * bsdfPdf + pdfPbrtEnv * pdfPbrtEnv);
                    }
                    radiance += throughput * pbrtEnvColorSample * pbrtEnvMissWeight;
                } else if (uniforms.pbrtHasConstantEnvLight != 0u) {
                    // A pbrt-loaded scene's own constant-colour
                    // LightSource "infinite" (metal_poc.mm's own
                    // loadPbrtScene() comment) - deliberately no MIS
                    // weight (weight 1.0, same as the specularBounce/
                    // envMapWidth==0 escape hatches just above): there is
                    // no NEE strategy for this light to double-count
                    // against, since none of this shader's material-
                    // shading functions sample it explicitly yet.
                    radiance += throughput * float3(uniforms.pbrtEnvColor);
                } else if (uniforms.isPbrtScene == 0u) {
                    float skyT = 0.5 * (rayDir.y + 1.0);
                    radiance += throughput * mix(skyBottom, skyTop, skyT);
                }
                // else: a real pbrt scene with no infinite light of any
                // kind - true black (no radiance added at all), matching
                // real pbrt-v4's own behaviour (Uniforms::isPbrtScene's
                // own comment).
                break;
            }

            uint primId = result.primitive_id;
            float3 hitPoint = rayOrigin + rayDir * result.distance;

            // Geometric normal: derived from the triangle's own vertices
            // for a triangle hit (as before), or from the sphere's centre
            // for a bounding-box hit - a sphere has no "vertices" to pull
            // a face normal from, but (hitPoint - centre) is exact for a
            // perfect sphere, no approximation.
            //
            // Spheres, the disk, AND the cylinder all report
            // intersection_type::bounding_box (none is a hardware-native
            // triangle) - they only stop being ambiguous once
            // `geometry_id` is checked too: sphereAS's own
            // geometryDescriptors array has the spheres' bounding-box
            // geometry at index 0, the disk's at index 1, and the
            // cylinder's at index 2 (metal_poc.mm's own
            // sphereAccelDesc.geometryDescriptors, section 171), and
            // geometry_id reports exactly that array index for a
            // bounding-box hit.
            bool isBoundingBox = (result.type == intersection_type::bounding_box);
            bool isDisk = isBoundingBox && (result.geometry_id == 1u);
            bool isCylinder = isBoundingBox && (result.geometry_id == 2u);
            bool isSphere = isBoundingBox && !isDisk && !isCylinder;
            // Suzanne is instanced TWICE (instance_id 2 and 3, matching
            // metal_poc.mm's own instanceDescs[] ordering - see that
            // file's addTransformedSuzanneInstance()) from the SAME
            // object-space geometry/AS - the one thing in this scene that
            // actually exercises a non-identity instance transform; every
            // other instance (the combined room+Spot geometry, the
            // sphere primitives) still uses the identity transform this
            // POC's very first version already had. Hardcoding the
            // instance_id threshold here (rather than deriving it) is
            // the same "explicitly documented, scene-specific constant"
            // approach this POC already uses for its light geometry.
            bool isSuzanneInstance = !isSphere && !isDisk && !isCylinder && (result.instance_id >= 2u);
            float3 normal;
            TriangleMaterial mat;
            if (isSphere) {
                SphereData sphere = spheres[primId];
                // Same interpolated centre sphereIntersectionFunction
                // itself tested the ray against (F11, section 167) - using
                // the STATIC sphere.center here instead would put the
                // shading normal off-centre from the surface point this
                // ray actually landed on for any moving sphere, a subtle
                // but real correctness bug distinct from the intersection
                // test itself.
                float3 center = float3(sphere.center) + shutterT * float3(sphere.centerDelta1);
                normal = normalize(hitPoint - center);
                mat = sphereMaterials[primId];
            } else if (isDisk) {
                // Flat and planar - the disk's own stored normal IS the
                // shading normal directly, no per-hit computation needed
                // (unlike a sphere's hitPoint-relative one or a triangle's
                // barycentric-interpolated one).
                DiskData disk = disks[primId];
                normal = float3(disk.normal);
                // NOT diskMaterials[0] - a real pre-existing bug (harmless
                // until now, since the hardcoded room ever only had ONE
                // disk, making [0] and [primId] the same value by
                // coincidence) found while adding real pbrt-loaded disks
                // (section 101): a second disk's own material was
                // silently ignored, every disk hit reading the room's own
                // disk material instead.
                mat = diskMaterials[primId];
            } else if (isCylinder) {
                // Radial direction perpendicular to the axis at the hit
                // point - project the hit onto the axis line to find the
                // nearest axis point, then point away from it. Exact for
                // a true cylinder, same "no approximation needed" note
                // as the sphere's own hitPoint-relative normal above.
                CylinderData cyl = cylinders[primId];
                float3 axis = float3(cyl.axis);
                float3 base = float3(cyl.base);
                float3 axisPoint = base + dot(hitPoint - base, axis) * axis;
                normal = normalize(hitPoint - axisPoint);
                mat = cylinderMaterials[primId];
            } else if (isSuzanneInstance) {
                // Object-space normal (Suzanne's own per-vertex data,
                // just like the non-instanced case below) transformed
                // into world space by THIS hit's own instance transform -
                // the one piece of shading math instancing actually adds
                // over the room/Spot geometry's own single-identity-
                // instance path.
                float3 objectNormal = shadingNormalForNoFacet(primId, result.triangle_barycentric_coord, suzanneNormals);
                normal = transformNormalByInstance(objectNormal, instanceTransforms[result.instance_id]);
                mat = suzanneMaterials[primId];
            } else {
                normal = shadingNormalFor(primId, result.triangle_barycentric_coord, normals, vertices);
                mat = triMaterials[primId];
            }
            // Raw (outward, unflipped) normal kept separately from here -
            // dielectric handling below needs to know which side of the
            // surface the ray is entering from (front vs back face) to
            // pick the right eta ratio, information the flipped-to-face-
            // the-ray version used by every other material below discards.
            bool frontFace = dot(normal, rayDir) < 0.0;
            float3 facingNormal = frontFace ? normal : -normal;

            // materialType == 7 (procedurally bump-mapped Lambertian):
            // perturb ONLY the shading normal used by the BSDF/NEE math
            // below, never `normal`/the ray-offset direction above - the
            // textbook bump-mapping distinction between what LOOKS
            // perturbed (shading) and what stays geometrically flat (ray
            // origins, self-intersection avoidance). Guarded to the
            // primary triangle buffer, the only place tangentFor()'s own
            // primId-indexed vertex/uv lookup is valid (see materialType
            // 7's own comment above) - never true for a sphere/disk/
            // Suzanne-instance hit, so this is simply skipped for those.
            CENSUS_PUSH(mat.materialType);
            if (mat.materialType == 7u && !isSphere && !isDisk && !isSuzanneInstance) {
                float2 bumpUV = texCoordFor(primId, result.triangle_barycentric_coord, uvs);
                float3 tangent = tangentFor(primId, vertices, uvs);
                facingNormal = proceduralBumpNormal(facingNormal, tangent, bumpUV, mat.roughness);
            } else if (mat.materialType == 21u) {
                // materialType 21 (checker-driven normal-mapped
                // Lambertian, B12's own sphere) - matches CPU's own
                // `normal_map_material` (a DIRECT tangent-space-normal
                // read, decoded from an RGB texture, not a finite-
                // difference displacement gradient the way materialType
                // 7's own bump map is). Only perturbs the SHADING
                // normal, same "shading vs geometric" split materialType
                // 7 already established. Reuses `checker3DColor()`'s own
                // 3D-spatial cell test directly on `hitPoint` (CPU's own
                // `checker_texture` is spatial, not UV-based, so this
                // needs no geometry-specific UV lookup at all - works
                // for this scene's sphere or any future triangle/disk
                // hit identically). The two decoded tangent-space
                // normals below are CPU's own two checker colours
                // (`build_normal_mapped_cornell()`'s own `norm_tex`)
                // pre-decoded by hand: `color(0.5,0.5,1.0)` -> `2*c-1` =
                // `(0,0,1)` (flat, no perturbation) and
                // `color(0.8,0.8,1.0)` -> `(0.6,0.6,1.0)` normalized (a
                // real diagonal tilt) - hardcoded rather than stored as
                // new material fields, since this checker's own two
                // colours are a fixed property of this one scene, not a
                // reusable general-purpose texture parameter.
                float3 tangent, bitangent;
                buildAnisotropicOnb(facingNormal, tangent, bitangent);
                float3 cell = floor(hitPoint / mat.roughness);
                float parity = fmod(abs(cell.x) + abs(cell.y) + abs(cell.z), 2.0);
                float3 nsLocal = (parity < 0.5) ? float3(0.0, 0.0, 1.0) : normalize(float3(0.6, 0.6, 1.0));
                float3 perturbed = normalize(nsLocal.x * tangent + nsLocal.y * bitangent + nsLocal.z * facingNormal);
                if (dot(perturbed, facingNormal) < 0.0) perturbed = -perturbed;
                facingNormal = perturbed;
            }

            // materialType == 3 (textured Lambertian): the hardcoded POC
            // room's own back wall (a triangle - texCoordFor()'s own
            // triangle-only inputs, primId/barycentric_coord, are valid
            // there) OR, since section 122, a hand-authored scene's own
            // textured SPHERE (A4 Earth) - texCoordFor() cannot run on a
            // sphere hit at all (no primId-indexed UV to look up), so
            // this reuses the exact technique materialType 9 already
            // established for exactly this situation (see that
            // material's own comment, just above its alphaX/alphaY
            // branch): equirectangularUV() on the hit's own (sphere-
            // centre-relative) normal gives a texture-space coordinate
            // for free, no real UV parameterization needed.
            float3 albedo;
            if (mat.materialType == 3u) {
                // isSphere: NOT a plain equirectangularUV(normal) call -
                // this project's own CPU get_sphere_uv() (sphere.h) uses
                // phi=atan2(-p.z,p.x)+pi, whereas equirectangularUV() uses
                // atan2(dir.z,dir.x) - algebraically or (since atan2 is
                // odd in its first argument), CPU's own u is exactly
                // equirectangularUV(x,y,-z).x, a longitude MIRROR of
                // equirectangularUV(normal).x, not merely a phase shift.
                // Latitude (v) already matches exactly with no
                // correction needed (both give v=0 at y=-1, v=1 at
                // y=+1) - checked algebraically, not assumed, after a
                // first version of this code rendered the correct
                // CONTINENTS-SHAPED but WRONG-LONGITUDE side of the
                // globe compared to a real --cpu render of the same
                // scene_id (caught by that comparison, not by eye alone -
                // an equirectangular texture looks equally "plausible"
                // from any longitude).
                float2 uv = isSphere ? equirectangularUV(float3(normal.x, normal.y, -normal.z))
                                      : texCoordFor(primId, result.triangle_barycentric_coord, uvs);
                albedo = earthTexture.sample(textureSampler, uv).rgb;
            } else if (mat.materialType == 6u) {
                // Procedural checker (Lambertian, same BSDF/NEE code path
                // as 0/3 below - only where albedo comes from differs):
                // `color` is the tile-A colour; tile B is a fixed
                // fraction of it, not a second stored colour - reusing
                // `emission` for that (an otherwise-unused field on a
                // non-emissive material, the same kind of repurposing
                // `ior`/`roughness` already do for materialTypes 2/4/5)
                // would have been read by the UNCONDITIONAL emissive-hit
                // check below (`any(mat.emission) > 0`) as "this triangle
                // is a light," making the floor incorrectly glow - a real
                // near-miss caught before it shipped, not a hypothetical.
                float2 uv = texCoordFor(primId, result.triangle_barycentric_coord, uvs);
                albedo = checkerColor(uv, 8.0, float3(mat.color), float3(mat.color) * 0.15);
            } else if (mat.materialType == 16u) {
                // Real 3D world-space checker (see checker3DColor()'s own
                // declaration comment) - `color`/`transmitColor` hold the
                // two REAL, independent tile colours (unlike materialType
                // 6's own fixed-fraction-of-one-colour simplification,
                // this one has no UV-collision reason to avoid a second
                // stored colour - `transmitColor` is otherwise unused by
                // any Lambertian-family material, materialType 12's own
                // diffuse-transmission tint is a different context
                // entirely). `roughness` reused as the checker's own
                // world-space cell size (`scale` in checker3DColor()'s
                // own signature - matches this project's own CPU
                // checker_texture's `inv_scale` construction parameter,
                // section 121).
                albedo = checker3DColor(hitPoint, mat.roughness, float3(mat.color), float3(mat.transmitColor));
            } else if (mat.materialType == 17u) {
                // Real Perlin-noise "marble" (see turbulenceSimple()'s
                // own declaration comment) - matches CPU's own
                // noise_texture::value() exactly: grey (0.5,0.5,0.5)
                // modulated by 1+sin(scale*p.z + 10*turb(p,7)), depth 7/
                // omega 0.5 fixed (CPU's own perlin::turb() defaults,
                // never overridden by any Basics-category scene).
                // `roughness` reused as the texture's own `scale`
                // parameter (matches materialType 16's own established
                // reuse of the same field for an unrelated procedural
                // texture's own scale). World-space `hitPoint`, no UV
                // needed - same reason materialType 16 works on a
                // sphere with no real UV parameterization.
                float marble = 1.0 + sin(mat.roughness * hitPoint.z + 10.0 * turbulenceSimple(hitPoint, 0.5, 7));
                albedo = float3(0.5, 0.5, 0.5) * marble;
            } else if (mat.materialType == 25u) {
                // A pbrt-v4 "checkerboard" Texture bound to a Diffuse
                // material's own "reflectance" (B22, section 162) -
                // pbrt-v4's real UV-space 2D checkerboard (NOT this
                // file's own materialType 6, whose tile B is a fixed
                // fraction of tile A rather than a second independent
                // colour, and NOT materialType 16's own WORLD-SPACE 3D
                // checker, which has no notion of "uscale"/"vscale" at
                // all). `color`/`transmitColor` hold the two independent
                // tile colours - same reuse materialType 16 already
                // established for the identical purpose (that function's
                // own comment). `conductorEta.x`/`.y` reused as
                // uscale/vscale (spare for every material type but 4/9,
                // same "one scalar slot, per-materialType meaning"
                // pattern every other TriangleMaterial field reuse here
                // already follows) - pbrt-v4's own two INDEPENDENT scale
                // factors, unlike materialType 6/16's own single shared
                // `scale`. Same isSphere/texCoordFor() UV split
                // materialType 3 above already established (equirect-
                // angular UV on a sphere hit, real triangle UV
                // otherwise) - this is the first OTHER material type to
                // need UV on a sphere at all, reusing that exact
                // mechanism rather than re-deriving it.
                const int texMode = int(mat.conductorK.y + 0.5);
                if (texMode == 8) {
                    // Checkerboard of an imagemap, optionally nested one level (J3/J6; loader comment):
                    // the same uv drives every level, each checker with its own scale.
                    float2 uv8 = isSphere ? equirectangularUV(float3(normal.x, normal.y, -normal.z))
                                          : texCoordFor(primId, result.triangle_barycentric_coord, uvs);
                    float2 outerTile = floor(uv8 * float2(mat.conductorEta.x, mat.conductorEta.y));
                    if ((int(outerTile.x) + int(outerTile.y)) & 1) {
                        albedo = float3(mat.color);                      // outer tex2
                    } else {
                        bool imageHere = true;
                        float3 inner = float3(mat.transmitColor);          // nested tex2
                        if (mat.conductorEta.z > 0.5) {
                            float2 innerTile = floor(uv8 * float2(mat.conductorK.x, mat.conductorK.z));
                            imageHere = ((int(innerTile.x) + int(innerTile.y)) & 1) == 0;
                        }
                        albedo = imageHere ? pbrtDiffuseTexture.sample(textureSampler, float2(uv8.x, 1.0 - uv8.y)).rgb : inner;
                    }
                } else if (texMode == 3 || texMode == 4 || texMode == 5) {
                    // fbm (3) / windy (4) / wrinkled (5), ports of CPU's fbm_texture / windy_texture / wrinkled_texture: keyed on the
                    // pbrt-world point t = k*p + off, no scale.
                    float3 wp = mat.conductorK.x * hitPoint + float3(mat.conductorEta);
                    float tv;
                    if (texMode == 3) {
                        tv = 0.5 + 0.5 * fbmSimple(wp, mat.transmitColor.y, int(mat.transmitColor.x + 0.5));
                    } else if (texMode == 4) {
                        float windStrength = fbmSimple(0.1 * wp, 0.5, 3);
                        float waveHeight = fbmSimple(wp, 0.5, 6);
                        tv = 0.5 + 0.5 * (abs(windStrength) * waveHeight);
                    } else {
                        tv = turbulenceSimple(wp, mat.transmitColor.y, int(mat.transmitColor.x + 0.5));
                    }
                    albedo = float3(clamp(tv, 0.0, 1.0));
                } else if (texMode == 6 || texMode == 7) {
                    float2 uvp = isSphere ? equirectangularUV(float3(normal.x, normal.y, -normal.z))
                                          : texCoordFor(primId, result.triangle_barycentric_coord, uvs);
                    if (texMode == 7) {
                        // bilerp: v00 = color, v01 = transmitColor, v10 = conductorEta, v11 = (K.x, roughness, K.z)
                        float3 v11 = float3(mat.conductorK.x, mat.roughness, mat.conductorK.z);
                        albedo = (1.0 - uvp.x) * (1.0 - uvp.y) * float3(mat.color) + uvp.x * (1.0 - uvp.y) * float3(mat.conductorEta)
                               + (1.0 - uvp.x) * uvp.y * float3(mat.transmitColor) + uvp.x * uvp.y * v11;
                    } else {
                        // dots: port of dots_texture::is_inside_dot (cell-jittered discs from Perlin noise)
                        float sCell = floor(uvp.x + 0.5), tCell = floor(uvp.y + 0.5);
                        bool inside = false;
                        if (perlinNoise3D(float3(sCell + 0.5, tCell + 0.5, 0.5)) > 0.0) {
                            const float radius = 0.35, maxShift = 0.5 - radius;
                            float sC = sCell + maxShift * perlinNoise3D(float3(sCell + 1.5, tCell + 2.8, 0.5));
                            float tC = tCell + maxShift * perlinNoise3D(float3(sCell + 4.5, tCell + 9.8, 0.5));
                            float ds = uvp.x - sC, dt = uvp.y - tC;
                            inside = (ds * ds + dt * dt) < radius * radius;
                        }
                        albedo = inside ? float3(mat.color) : float3(mat.transmitColor);
                    }
                } else if (texMode == 2) {
                    // pbrt "marble" reflectance texture (loader comment): evaluated at
                    // t = k*p + off; transmitColor = (octaves, omega, variation).
                    float3 mp = mat.conductorK.x * hitPoint + float3(mat.conductorEta);
                    albedo = pbrtMarbleColor(mp, mat.transmitColor.y, int(mat.transmitColor.x + 0.5), mat.transmitColor.z);
                } else if (texMode == 1) {
                    // 3D checkerboard (see the loader's own comment): texture-space point
                    // t = k*p + off, parity of floor(tx)+floor(ty)+floor(tz); even -> tex1.
                    float3 tp = mat.conductorK.x * hitPoint + float3(mat.conductorEta);
                    int cellSum = int(floor(tp.x)) + int(floor(tp.y)) + int(floor(tp.z));
                    albedo = ((cellSum & 1) == 0) ? float3(mat.color) : float3(mat.transmitColor);
                } else {
                float2 uv = isSphere ? equirectangularUV(float3(normal.x, normal.y, -normal.z))
                                      : texCoordFor(primId, result.triangle_barycentric_coord, uvs);
                float2 tile = floor(uv * float2(mat.conductorEta.x, mat.conductorEta.y));
                float parity = fmod(tile.x + tile.y, 2.0);
                albedo = (abs(parity) < 0.5) ? float3(mat.color) : float3(mat.transmitColor);
                }
            } else if (mat.materialType == 26u || mat.materialType == 27u) {
                // A pbrt-v4 Diffuse (26) or CoatedDiffuse (27, J1,
                // section 172) material's own "texture reflectance"
                // bound to a bare "imagemap" Texture (F5/F9, section
                // 166) - a REAL image FILE (`Material::textureFilename`,
                // pbrt_flatten.h), not a procedural pattern like
                // materialType 25's own checkerboard - reads
                // `pbrtDiffuseTexture` (this scene's own decoded file)
                // instead of the hardcoded room's own earthTexture, the
                // same "separate slot" reasoning materialType 3 already
                // established for pbrt-loaded infinite-light images.
                // Same isSphere/texCoordFor() UV split materialType 3/25
                // above already established. `1.0 - uv.y`: pbrt-v4's own
                // UV convention has v=0 at the BOTTOM of the image (this
                // scene's own "point2 uv" authors v=0/v=1 that way), but
                // pbrt_load::detail::decodeInfiniteLightImage() (stb_image
                // under the hood) always returns row 0 as the TOP row,
                // the same row Metal's own `replaceRegion:` (this
                // texture's own upload call, metal_poc.mm) then places at
                // texture row 0 too - sampled with `uv.y` UNFLIPPED, that
                // combination reads v=0 from the TOP instead of the
                // BOTTOM, a real top/bottom mirror caught by comparing
                // this scene's own 4-quadrant checker against `--cpu`
                // (each quadrant's colour landed in the vertically
                // opposite corner) rather than assumed correct from a
                // symmetric test image. `* mat.roughness`: this hit's own
                // texture SCALE multiplier (a "scale"-class Texture
                // wrapping the bare imagemap, Material::textureScale's
                // own comment, mapMaterial()'s own comment on reusing
                // this field) - both materialType 26 and 27's own
                // TriangleMaterial construction now sets `roughness` to
                // this value (1.0, a provable no-op, when the scene's
                // own texture had no wrapping "scale").
                float2 uv = isSphere ? equirectangularUV(float3(normal.x, normal.y, -normal.z))
                                      : texCoordFor(primId, result.triangle_barycentric_coord, uvs);
                albedo = pbrtDiffuseTexture.sample(textureSampler, float2(uv.x, 1.0 - uv.y)).rgb * mat.roughness;
            } else if (mat.materialType == 12u && mat.conductorK.y > 8.5) {
                // DiffuseTransmission with image-textured reflectance (diffuse slot) and/or transmittance
                // (second slot) - J2. conductorK.x = 1 when reflectance is an image, conductorK.z = 1 when
                // transmittance is; otherwise the flat colour in color / transmitColor is used. The shade
                // function reads mat.transmitColor, so the sampled transmittance is written back into it.
                float2 uv9 = texCoordFor(primId, result.triangle_barycentric_coord, uvs);
                albedo = (mat.conductorK.x > 0.5)
                    ? pbrtDiffuseTexture.sample(textureSampler, float2(uv9.x, 1.0 - uv9.y)).rgb
                    : float3(mat.color);
                if (mat.conductorK.z > 0.5)
                    mat.transmitColor = packed_float3(pbrtTransmitTexture.sample(textureSampler, float2(uv9.x, 1.0 - uv9.y)).rgb);
            } else {
                albedo = float3(mat.color);
            }

            // Unconditional check (regardless of material type) for
            // whether this hit is the light quad - covers a camera ray or
            // a GI bounce landing on it directly, mirror/glass included (a
            // mirror reflecting toward the light correctly shows it, since
            // this fires for THEIR reflected/refracted rays' next hit too,
            // not just Lambertian ones). Every non-light surface has
            // emission == 0, so `any(...)` below is false and this whole
            // block is a no-op for them.
            //
            // `&& (frontFace || mat.twoSided != 0u)`: an AreaLight only
            // emits from the side its own `normal` points toward, UNLESS
            // the scene named `"bool twosided" [true]` (section 104) -
            // the same one-sidedness (or lack of it) the NEE branches
            // below already enforce via their own `(cosLight > 0.0 ||
            // (ls.twoSided != 0.0 && cosLight < 0.0))` check (see e.g.
            // this scene's own ceiling lights, which only shine down
            // into the room, one-sided). Without the frontFace half of
            // this, a camera ray or BSDF-sampled bounce landing on the
            // BACK of a one-sided light quad would still read its
            // emission unconditionally - invisible in this committed
            // scene (every light is mounted flush against the ceiling,
            // its own back face physically inaccessible from inside the
            // room) but a genuine correctness gap: this is the one place
            // in the shader a light's own emission was reachable without
            // a facing check at all, inconsistent with every NEE
            // branch's own already-correct behaviour. `frontFace` is
            // already computed above for the dielectric branch's own eta
            // selection - reused here, not recomputed. `mat.twoSided`
            // (not `lights[mat.lightId].twoSided`) so this same check
            // also covers a non-quad/disk emissive shape with NO
            // AreaLightData entry at all (`mat.lightId < 0`, sections
            // 100/101) - every emissive TriangleMaterial carries its own
            // copy of the flag directly, needing no lights[] lookup.
            if (any(float3(mat.emission) > float3(0.0)) && (frontFace || mat.twoSided != 0u)) {
                // Patterned emission (materialType 10 - see AreaLight's
                // own comment): a DIRECT hit needs the checker pattern
                // evaluated at THIS hit's own interpolated UV
                // (texCoordFor(), the same triangle-only lookup
                // materialType 3/6's own albedo already uses), not the
                // light-sample point's (u.x, u.y) sampleAreaLight() uses -
                // two different points on the same light quad, each
                // needing its own UV. `roughness` reused a SIXTH way here
                // (after materialTypes 4/5/7/9's own reuses) as this
                // pattern's own tile-B fraction, mirroring AreaLight's
                // `patternTileB`.
                float3 hitEmission = float3(mat.emission);
                if (mat.materialType == 10u) {
                    float2 patUV = texCoordFor(primId, result.triangle_barycentric_coord, uvs);
                    hitEmission = checkerColor(patUV, 6.0, hitEmission, hitEmission * mat.roughness);
                } else if (mat.materialType == 15u) {
                    // Real image-based AreaLightSource (section 105) -
                    // same UV lookup as materialType 10's own checker
                    // pattern (a direct hit needs THIS hit's own
                    // interpolated UV, not the NEE sample point's), but
                    // sampling a real texture instead of a procedural
                    // pattern. `mat.emission` here holds the pure
                    // (scale,scale,scale) multiplier (see AreaLight::
                    // useTexture's own comment), not a direct radiance -
                    // pbrt-v4 ignores L entirely once an image is given.
                    float2 texUV = texCoordFor(primId, result.triangle_barycentric_coord, uvs);
                    hitEmission = pbrtAreaLightTexture.sample(textureSampler, texUV).rgb * hitEmission;
                }
                if (specularBounce || mat.lightId < 0) {
                    // No competing NEE sample could have produced this
                    // exact hit (camera ray, or a mirror/glass bounce -
                    // both skip NEE entirely, see their own branches
                    // below), so there's nothing to weight against.
                    // `mat.lightId < 0` covers a SEPARATE case with the
                    // same "nothing to weight against" shape: a pbrt-
                    // loaded area light shape too complex for this
                    // loader's own single-quad AreaLightData (loadPbrtScene()'s
                    // own comment) - emissive, but deliberately not
                    // registered in `lights[]` for any material function's
                    // own NEE loop to have sampled it from in the first
                    // place.
                    radiance += throughput * hitEmission;
                } else {
                    // Reached via a BSDF-sampled continuation ray (diffuse
                    // or conductor) - weight by the power heuristic against
                    // what the light-sampling strategy's own PDF would have
                    // been for this exact hit. mat.lightId names exactly
                    // which AreaLight this triangle belongs to, so this
                    // works for any number of lights, not just one -
                    // this light's own power-proportional picking pdf
                    // (`light.pmf`, see AreaLight's own comment - replaces
                    // this POC's old flat 1/lightCount) folded in alongside
                    // the same area-to-solid-angle conversion the NEE
                    // branches below use.
                    AreaLight light = lights[mat.lightId];
                    const float hitDistFromBounce = result.distance + mediumSkippedDist;
                    float distSq = hitDistFromBounce * hitDistFromBounce;
                    // abs(), not the old max(dot(...), 0.0001) alone - a
                    // real, previously-latent bug found by code review
                    // (section 106), exposed once section 104's own
                    // frontFace||twoSided gate made a BACK-face hit on a
                    // two-sided light reachable here at all. On that
                    // face, dot(normal,-rayDir) is NEGATIVE; the old
                    // max(...,0.0001) clamped it up to the epsilon
                    // itself rather than reflecting it, making pdfLight
                    // spuriously huge and crushing this MIS weight
                    // toward 0 - silently dropping the BSDF-sampled
                    // strategy's own contribution to a two-sided light's
                    // back face, an energy-loss bias. abs() first, THEN
                    // the same 0.0001 floor purely to avoid a division
                    // by exact zero at a grazing angle - matches the
                    // same abs(cosLight) fix the NEE branches below
                    // already got in section 104, just missed here.
                    //
                    // A sphere light (AreaLight::kind==1, B14, section
                    // 184) has no single fixed plane normal the way a
                    // quad does - `light.normal` is unused/meaningless for
                    // one (see that field's own comment) - so this uses
                    // THIS hit's own already-computed geometric `normal`
                    // (line ~787, `normalize(hitPoint - sphereCenter)`)
                    // instead, the correct per-point outward normal a
                    // curved surface needs for its own area-to-solid-angle
                    // Jacobian.
                    float3 lightNormalAtHit = (light.kind > 0.5 && light.kind < 1.5) || (light.kind > 2.5 && light.kind < 3.5) ? normal : float3(light.normal);
                    float cosLight = max(abs(dot(lightNormalAtHit, -rayDir)), 0.0001);
                    float pdfLight = (distSq / (light.area * cosLight)) * light.pmf;
                    float weight = (bsdfPdf * bsdfPdf) / (bsdfPdf * bsdfPdf + pdfLight * pdfLight);
                    radiance += throughput * hitEmission * weight;
                }
            }

            if (mat.materialType == 2u) {
                const bool smoothMediumExit = isSphere && inGlass && mat.conductorK.y > 0.5 && !specularBounce;
                const float smoothPdfBefore = bsdfPdf;
                if (!shadeDielectric(mat, hitPoint, normal, facingNormal, frontFace, result.distance,
                                      rayDir, rayOrigin, throughput, specularBounce, rngState)) break;
                if (smoothMediumExit && dot(rayDir, normal) > 0.0) {
                    specularBounce = false;
                    bsdfPdf = smoothPdfBefore;
                }
            } else if (mat.materialType == 22u) {
                if (!shadeDispersiveDielectric(mat, hitPoint, normal, facingNormal, frontFace, result.distance,
                                      rayDir, rayOrigin, throughput, specularBounce, rngState, rgbChannel)) break;
            } else if (mat.materialType == 5u) {
                // Rough glass bounding a medium (see the thin-glass note below): keep the MIS state of a scattered
                // path across the boundary, so the light is not counted by NEE and again at full weight.
                const bool roughMediumExit = isSphere && inGlass && mat.conductorK.y > 0.5 && !specularBounce;
                const float roughPdfBefore = bsdfPdf;
                if (!shadeRoughDielectric(mat, hitPoint, normal, facingNormal, frontFace, result.distance,
                                           rayDir, rayOrigin, throughput, specularBounce, rngState)) break;
                if (roughMediumExit && dot(rayDir, normal) > 0.0) {
                    specularBounce = false;
                    bsdfPdf = roughPdfBefore;
                }
            } else if (mat.materialType == 23u) {
                if (!shadeDispersiveRoughDielectric(mat, hitPoint, normal, facingNormal, frontFace, result.distance,
                                           rayDir, rayOrigin, throughput, specularBounce, rngState, rgbChannel)) break;
            } else if (mat.materialType == 4u || mat.materialType == 9u) {
                if (!shadeConductor(mat, hitPoint, normal, facingNormal, uniforms,
                                     lights, pointLights, directionalLights, projectionLights, goniometricLights,
                                     envMarginalCDF, envConditionalCDF, uniforms.envMapWidth, uniforms.envMapHeight,
                                     pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                                     ggxEnergyTable, uniforms.ggxEnergyRoughRes, uniforms.ggxEnergyMuRes,
                                     earthTexture, pbrtEnvTexture, goniometricTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, textureSampler,
                                     isect, accelStructure, functionTable,
                                     rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState)) break;
            } else if (mat.materialType == 1u) {
                if (!shadeMirror(albedo, hitPoint, facingNormal, rayDir, rayOrigin, throughput, specularBounce)) break;
            } else if (mat.materialType == 11u) {
                // A thin glass sphere bounding a medium: a ray that scattered inside (NEE already added the light
                // with an MIS weight against this phase-sampled continuation) passes STRAIGHT through the boundary,
                // so it can hit the light directly. shadeThinDielectric marks every event specular, which would add
                // that hit at full weight on top of the NEE (~2x too bright, E11). Keep the pre-boundary MIS state
                // when the ray went straight through.
                const bool thinMediumExit = isSphere && inGlass && mat.conductorK.y > 0.5 && !specularBounce;
                const float3 thinDirBefore = rayDir;
                const float thinPdfBefore = bsdfPdf;
                if (!shadeThinDielectric(mat, hitPoint, normal, facingNormal,
                                          rayDir, rayOrigin, throughput, specularBounce, rngState)) break;
                if (thinMediumExit && dot(rayDir, thinDirBefore) > 0.9999) {
                    specularBounce = false;
                    bsdfPdf = thinPdfBefore;
                }
            } else if (mat.materialType == 8u || mat.materialType == 27u) {
                // materialType 27 (J1, section 172): the SAME clearcoat
                // shading materialType 8 already uses, just with `albedo`
                // (computed above) sourced from a real per-hit texture
                // sample instead of `mat.color` - shadeClearcoat() itself
                // needs no changes at all, it already takes `albedo` as
                // an explicit parameter rather than reading `mat.color`
                // directly.
                if (!shadeClearcoat(mat, albedo, hitPoint, facingNormal, uniforms,
                                     lights, pointLights, directionalLights, projectionLights, goniometricLights,
                                     envMarginalCDF, envConditionalCDF, uniforms.envMapWidth, uniforms.envMapHeight,
                                     pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                                     earthTexture, pbrtEnvTexture, goniometricTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, textureSampler,
                                     isect, accelStructure, functionTable,
                                     rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState)) break;
            } else if (mat.materialType == 12u) {
                if (!shadeDiffuseTransmission(mat, albedo, hitPoint, facingNormal, uniforms,
                                               lights, pointLights, directionalLights, projectionLights, goniometricLights,
                                               envMarginalCDF, envConditionalCDF, uniforms.envMapWidth, uniforms.envMapHeight,
                                               pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                                               earthTexture, pbrtEnvTexture, goniometricTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, textureSampler,
                                               isect, accelStructure, functionTable,
                                               rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState)) break;
            } else if (mat.materialType == 13u) {
                if (!shadeOrenNayar(mat, albedo, hitPoint, facingNormal, uniforms,
                                     lights, pointLights, directionalLights, projectionLights, goniometricLights,
                                     envMarginalCDF, envConditionalCDF, uniforms.envMapWidth, uniforms.envMapHeight,
                                     pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                                     earthTexture, pbrtEnvTexture, goniometricTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, textureSampler,
                                     isect, accelStructure, functionTable,
                                     rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState)) break;
            } else if (mat.materialType == 14u) {
                if (!shadeVelvet(mat, albedo, hitPoint, facingNormal, uniforms,
                                  lights, pointLights, directionalLights, projectionLights, goniometricLights,
                                  envMarginalCDF, envConditionalCDF, uniforms.envMapWidth, uniforms.envMapHeight,
                                  pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                                  earthTexture, pbrtEnvTexture, goniometricTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, textureSampler,
                                  isect, accelStructure, functionTable,
                                  rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState)) break;
            } else if (mat.materialType == 24u) {
                if (!shadePrincipled(mat, hitPoint, facingNormal,
                                  rayDir, rayOrigin, throughput, specularBounce, rngState)) break;
            } else if (mat.materialType == 31u) {
                pathTouchedHair = true;
                if (!shadeHair(mat, hitPoint, facingNormal,
                                  rayDir, rayOrigin, throughput, specularBounce, rngState)) break;
            } else if (mat.materialType == 18u) {
                if (!shadeNormalizedFresnel(mat, hitPoint, facingNormal, uniforms,
                                  lights, pointLights, directionalLights, projectionLights, goniometricLights,
                                  envMarginalCDF, envConditionalCDF, uniforms.envMapWidth, uniforms.envMapHeight,
                                  pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                                  earthTexture, pbrtEnvTexture, goniometricTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, textureSampler,
                                  isect, accelStructure, functionTable,
                                  rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState)) break;
            } else if (mat.materialType == 19u) {
                if (!shadeCoatedDiffuse(mat, hitPoint, facingNormal, uniforms,
                                  lights, pointLights, directionalLights, projectionLights, goniometricLights,
                                  envMarginalCDF, envConditionalCDF, uniforms.envMapWidth, uniforms.envMapHeight,
                                  pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                                  earthTexture, pbrtEnvTexture, goniometricTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, textureSampler,
                                  isect, accelStructure, functionTable,
                                  rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState)) break;
            } else if (mat.materialType == 20u) {
                if (!shadeCoatedConductor(mat, hitPoint, facingNormal, uniforms,
                                  lights, pointLights, directionalLights, projectionLights, goniometricLights,
                                  envMarginalCDF, envConditionalCDF, uniforms.envMapWidth, uniforms.envMapHeight,
                                  pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                                  earthTexture, pbrtEnvTexture, goniometricTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, textureSampler,
                                  isect, accelStructure, functionTable,
                                  rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState)) break;
            } else {
                if (!shadeLambertian(mat, albedo, hitPoint, facingNormal, uniforms,
                                      lights, pointLights, directionalLights, projectionLights, goniometricLights,
                                      envMarginalCDF, envConditionalCDF, uniforms.envMapWidth, uniforms.envMapHeight,
                                      pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                                      earthTexture, pbrtEnvTexture, goniometricTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, textureSampler,
                                      isect, accelStructure, functionTable,
                                      rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState)) break;
            }
                mediumSkippedDist = 0.0;  // a real bounce starts a new segment
                // After a dielectric event on a glass sphere that carries a medium: the path is now inside
                // the sphere iff its new direction points against the outward normal (true for refraction in,
                // reflection inside, and a thin dielectric passing straight through). `normal` is the sphere's
                // outward normal here.
                if (isSphere && mat.conductorK.y > 0.5 &&
                    (mat.materialType == 2u || mat.materialType == 5u || mat.materialType == 11u)) {
                    const bool wasInGlass = inGlass;
                    inGlass = dot(rayDir, normal) < 0.0;
                    if (inGlass) {
                        glassSigmaT3 = float3(mat.conductorEta);
                        glassG = mat.conductorK.x;
                        glassAlbedo = float3(mat.transmitColor);
                        if (!wasInGlass) {
                            // Entering. A chromatic medium: follow ONE colour channel for this path (chosen
                            // uniformly, weight x3 on it) so free flight and shadow attenuation can use that
                            // channel's sigma_t; averaged over paths every channel is recovered. A grey medium
                            // needs no such colour noise.
                            if (mat.conductorK.z > 0.5) {
                                glassChan = min(int(randFloat(rngState) * 3.0), 2);
                                float3 chanMask = float3(glassChan == 0 ? 3.0 : 0.0, glassChan == 1 ? 3.0 : 0.0, glassChan == 2 ? 3.0 : 0.0);
                                throughput *= chanMask;
                            } else {
                                glassChan = -1;
                            }
                        }
                    }
                }
            } // !scatteredInMedium && !passedThroughMediumSphere

            // Russian roulette after a few bounces, same "let cheap paths
            // terminate early, keep expensive ones unbiased" shape as
            // this project's CPU integrator - throughput's max channel is
            // the survival probability, divided back in on survival so
            // the estimator stays unbiased. Clamped to 1.0 - a real,
            // previously-latent bug found via the realistic camera's own
            // noisy, non-converging first render: every earlier scene's
            // own `throughput` starts at EXACTLY (1,1,1) and only ever
            // SHRINKS via material albedo (<=1) multiplications, so
            // `max(throughput channels)` was always already <=1 there,
            // making this clamp an invisible no-op - but
            // `cameraRealistic`'s own `cameraWeight` (section 157) can
            // legitimately exceed 1.0 (a real pbrt-v4 importance-sampling
            // weight, unbounded by design), so `throughput` can too.
            // WITHOUT the clamp, a >1 `p` still guarantees survival
            // (`randFloat() > p` is never true for p>=1) but ALSO
            // divides `throughput` by that same large `p` anyway - an
            // unnecessary, UNCOMPENSATED shrink standard RR never
            // applies once survival is already certain (the division
            // exists ONLY to compensate for paths that DO die, keeping
            // the estimator unbiased when survival is a coin flip; once
            // survival is guaranteed, no compensation is needed at all).
            // Left uncorrected, every sample's own bright, unbounded
            // `cameraWeight` got silently and inconsistently divided
            // back down partway through its own path - exactly the
            // per-sample-inconsistent, non-converging speckle the first
            // render showed.
            if (depth > 3) {
                float p = min(max(throughput.x, max(throughput.y, throughput.z)), 1.0);
                if (randFloat(rngState) > p) break;
                throughput /= max(p, 0.0001);
            }
            continuePath = true;
        } while (false);   // a `break` in the body above ends the PATH (every one of them did, in the old depth loop)
        if (continuePath) {
            ++depth;
            if (depth < uniforms.maxDepth) continue;   // next bounce of this path (other lanes may be starting new ones)
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
                    relativeConverged = (standardError / convergedMean) < kAdaptiveThreshold;
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

