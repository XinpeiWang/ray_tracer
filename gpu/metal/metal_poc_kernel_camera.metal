// metal_poc_kernel_camera.metal
// The start of one sample's path in primaryRayKernel: the camera ray (pinhole, orthographic, spherical, realistic lens, orbit blur, thin-lens
// depth of field) and the fresh per-path state. Pure code motion out of the old kernel body; see metal_poc_kernel_state.metal.

// Draws this sample's pixel position, shutter time and camera ray (rayOrigin / rayDir). Returns the camera weight (1.0 except for the realistic
// lens, where it carries the vignetting / cos^4 factor) that the path's throughput starts from.
inline float generateCameraRay(thread const KernelRes& R, thread PathState& P, uint2 tid) {
    KERNEL_RES_ALIASES(R)
    PATH_STATE_ALIASES(P)
    // Filter-importance-sampled sub-pixel sample position - the
    // multi-sample loop's own antialiasing AND reconstruction-filter
    // reach in one draw (sampleFilterPosition()'s own comment);
    // without it every sample would retrace the exact same primary
    // ray. Replaces the old uniform-in-[0,1)-pixel jitter, which was
    // exactly a hardcoded 0.5-pixel-radius box filter - see section
    // 207, docs/history/METAL_GPU_FEASIBILITY.md.
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
    return cameraWeight;
}

// Resets everything a new path carries: throughput, radiance, MIS bookkeeping, medium and glass state, census record. `s` is the sample index
// (the census records sample 0 only).
inline void initPathState(thread const KernelRes& R, thread PathState& P, float cameraWeight, uint s) {
    KERNEL_RES_ALIASES(R)
    PATH_STATE_ALIASES(P)

    throughput = float3(cameraWeight);
    // Chromatic camera medium: follow one colour channel for this whole path (chosen uniformly, weight x3 on it);
    // averaged over paths every channel is recovered, and free flight / shadow rays can use that channel's sigma_t.
    fogChan = -1;
    if (uniforms.fogChromatic != 0u && uniforms.fogSigmaT > 0.0) {
        fogChan = min(int(randFloat(rngState) * 3.0), 2);
        throughput *= float3(fogChan == 0 ? 3.0 : 0.0, fogChan == 1 ? 3.0 : 0.0, fogChan == 2 ? 3.0 : 0.0);
    }
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
    envLightSampled = false;   // PathState lives across the pixel's samples: every sample starts with the environment not yet sampled
    fromMediumScatter = false;
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
    // A camera inside a glass shell that bounds a medium (A9's radius-5000 world haze) starts INSIDE that medium: it never refracts in.
    if (uniforms.cameraGlassPrim > 0) {
        const TriangleMaterial camGlass = sphereMaterials[uniforms.cameraGlassPrim - 1];
        inGlass = true;
        glassSigmaT3 = float3(camGlass.conductorEta);
        glassG = camGlass.conductorK.x;
        glassAlbedo = float3(camGlass.transmitColor);
        if (fogChan >= 0) {
            glassChan = fogChan;
        } else if (camGlass.conductorK.z > 0.5) {
            glassChan = min(int(randFloat(rngState) * 3.0), 2);
            throughput *= float3(glassChan == 0 ? 3.0 : 0.0, glassChan == 1 ? 3.0 : 0.0, glassChan == 2 ? 3.0 : 0.0);
        }
    }
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
}
