// metal_poc.metal
// Metal ray tracing proof-of-concept, step 2 - see
// docs/METAL_GPU_FEASIBILITY.md section 8. Step 1 (committed in PR #2) was
// a single-bounce raycast preview with no shadow rays. This step turns it
// into an actual minimal Monte Carlo path tracer: shadow-ray occlusion, a
// real multi-bounce GI loop with Russian roulette, a second material
// (mirror) to prove per-primitive material-type branching works (not just
// per-primitive colour), and multi-sample antialiasing. Still deliberately
// minimal otherwise - one directional light, two material types, no
// textures, no area lights, no spectral/BSSRDF/volumetric anything. The
// goal remains proving the pipeline shape, now one level deeper (a real
// path integrator, not just primary-ray shading) rather than feature
// coverage.

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

// Every member here is either already 4-byte-scalar or packed_float3
// (size 12, align 4, no implicit padding) - deliberately avoiding plain
// `float3` (size 12 but ALIGN 16 inside a struct) anywhere two of these
// structs are shared byte-for-byte between the C++ host struct and this
// MSL one. A float3-vs-packed_float3 layout mismatch between host and
// device is a classic, silent GPU bug (wrong bytes land in the wrong
// field, still "compiles and runs," just renders garbage) - packed_float3
// on both sides everywhere a struct crosses the host/device boundary
// avoids the whole class of bug rather than reasoning through MSL's
// packing rules per-field.
struct Uniforms {
    packed_float3 cameraPos;
    packed_float3 cameraForward;
    packed_float3 cameraRight;
    packed_float3 cameraUp;
    float tanHalfFov;
    float aspect;
    uint width;
    uint height;
    uint samplesPerPixel;
    uint maxDepth;
    uint frameSeed;
    uint lightCount;
    // Thin-lens depth of field: lensRadius == 0 reduces to the original
    // pinhole camera exactly (no lens-sample branch taken below), so
    // every earlier PR's own screenshots stay reproducible bit-for-bit
    // reasoning-wise by just leaving this at 0 - DOF is purely additive,
    // not a replacement for the pinhole path.
    float lensRadius;
    float focusDistance;
    // Aperture SHAPE for the lens sample above - 0/1/2 (every scene
    // before this one) means "circular" (sampleUnitDisk()); 3 or more is
    // a regular polygon with this many sides/blades (samplePolygonAperture()),
    // the real reason out-of-focus highlights in an actual photo read as
    // hexagons/pentagons rather than perfect circles. Purely additive,
    // same "0 reproduces the exact prior behaviour" shape lensRadius == 0
    // itself already has - never affects anything when lensRadius == 0.
    uint apertureBlades;
    // Camera (shutter) motion blur: the camera translates by
    // `cameraVelocity` (world-space, full displacement) over the frame's
    // simulated [0,1] shutter interval - each primary-ray SAMPLE draws
    // its own uniform-random shutter time and offsets the camera origin
    // by that fraction of the velocity before casting, so different
    // samples for the same pixel see the camera at different points along
    // its path, and averaging them (the same multi-sample loop every
    // other feature in this POC already reuses) is what produces the
    // blur - no separate accumulation pass needed. `cameraVelocity ==
    // (0,0,0)` (every earlier PR's own scenes) makes every sample use the
    // exact same origin regardless of its sampled time, i.e. the original
    // static camera exactly - purely additive, like lensRadius == 0 above.
    packed_float3 cameraVelocity;
    // Homogeneous participating medium (fog) filling the WHOLE scene
    // volume - not attached to any one object's geometry, the simplest
    // possible "is this ray inside the medium" answer (always yes,
    // camera to first surface hit), avoiding needing boundary tracking
    // (entering/exiting a fog volume's own geometry) this POC doesn't
    // have any other reason to build yet. fogSigmaT == 0 (every earlier
    // PR's own scenes) skips the medium-interaction branch entirely -
    // purely additive, same pattern as lensRadius/cameraVelocity above.
    float fogSigmaT; // extinction coefficient (scalar - shared across
                      // RGB channels for distance sampling, not spectral)
    packed_float3 fogAlbedo; // single-scattering albedo (sigma_s/sigma_t), per channel
    // useEnvironmentMap == 0 (every earlier PR's own scenes) keeps the
    // original flat two-colour sky gradient exactly; != 0 replaces it
    // with an equirectangular sample of `earthTexture` by ray DIRECTION
    // instead of by surface UV - reusing the same texture already loaded
    // for materialType 3, sampled a completely different way. Purely
    // additive/toggleable, same pattern as every other uniform above.
    uint useEnvironmentMap;
    // Henyey-Greenstein phase-function asymmetry, [-1,1]. 0 (isotropic)
    // reproduces the same uniform-over-sphere DISTRIBUTION step 17's own
    // isotropic phase function used (HG's own g==0 case reduces to
    // uniform-over-sphere sampling, just relative to a local frame
    // instead of absolute world axes - a rotationally-invariant
    // distribution is identical either way) -
    // not a separate toggle, this field alone controls both the isotropic
    // and directional cases. Positive g = forward scattering (favours
    // continuing roughly the same direction light was already travelling
    // - fog/haze/water droplets in reality skew strongly forward, g
    // around 0.7-0.9 in Mie scattering terms), negative = backward.
    float fogAsymmetryG;
    // How many entries in the SEPARATE `pointLights` buffer are live -
    // not part of `lightCount`/`lights[]` above (see PointLight's own
    // comment on why delta lights aren't folded into the area-light
    // picking scheme). 0 (every earlier scene) skips the point-light NEE
    // loop entirely in every material branch - purely additive.
    uint pointLightCount;
    // Same idea as pointLightCount, for the separate `directionalLights`
    // buffer (see DirectionalLight's own comment). 0 (every earlier
    // scene) skips that loop entirely - purely additive.
    uint directionalLightCount;
    // Same idea again, for the separate `projectionLights` buffer (see
    // ProjectionLight's own comment) - a fourth delta-light type, summed
    // unconditionally like the other three, never picked. 0 (every
    // earlier scene) skips that loop entirely - purely additive.
    uint projectionLightCount;
    // Same idea again, for the separate `goniometricLights` buffer (see
    // GoniometricLight's own comment) - a fifth delta-light type, summed
    // unconditionally like the other four, never picked. 0 (every
    // earlier scene) skips that loop entirely - purely additive.
    uint goniometricLightCount;
    // Single-pass adaptive sampling toggle - see the shading loop's own
    // comment on `convergedCount`/`kAdaptiveThreshold` for the full
    // "why" and the CPU integrator this was ported from. 0 (every scene
    // before this one) skips the convergence check entirely, an exact
    // no-op.
    uint adaptiveSampling;
    // Environment-map importance sampling (section 69's own
    // EnvDistribution2D, uploaded as the `envMarginalCDF`/
    // `envConditionalCDF` buffers below) - the image dimensions its own
    // CDF arrays were built at, needed by the device-side binary search/
    // evaluation functions to index them correctly. 0 (every scene before
    // this one, or `useEnvironmentMap == 0`) skips the environment
    // light's own NEE sampling entirely in every material's own shading
    // function - purely additive, same "0 reproduces prior behaviour
    // exactly" pattern every other optional feature here already uses.
    uint envMapWidth;
    uint envMapHeight;
    // GGX multi-scatter energy-compensation table (section 72/73) - the
    // grid dimensions its own directional-albedo table (`ggxEnergyTable`
    // buffer below) was built at, needed to index it correctly. Applies
    // unconditionally to materialType 4/9's own GGX conductor (there is
    // no "0 disables this" case the way env-map/adaptive-sampling have -
    // the table is always built and always physically correct to apply,
    // unlike those genuinely optional features).
    uint ggxEnergyRoughRes;
    uint ggxEnergyMuRes;
    // A pbrt-loaded scene's own LightSource "infinite" with no image
    // (constant colour only) - see the miss-path code below and
    // metal_poc.mm's own loadPbrtScene() comment. Deliberately NO NEE/
    // MIS strategy the way useEnvironmentMap's earthTexture-based one
    // has (section 71) - every material's own shading function would
    // need its own new sampling+MIS block to add that, real work this
    // POC's own docs (METAL_GPU_FEASIBILITY.md) explicitly scope out for
    // now. A pure miss-path contribution is still a correct, unbiased
    // Monte Carlo estimator (just higher-variance than an NEE-augmented
    // one) - the exact same tradeoff this file's own env-map support
    // already had for a long time before section 71 added NEE on top of
    // it, and the same one envMapWidth==0 still falls back to today.
    uint pbrtHasConstantEnvLight;
    packed_float3 pbrtEnvColor;
    // A pbrt-loaded scene's own image-based LightSource "infinite" (an
    // actual named image file, as opposed to pbrtHasConstantEnvLight's
    // colour-only case just above) - mutually exclusive with that flag
    // (an infinite light is either constant-colour or image-based, never
    // both - see metal_poc.mm's own loadPbrtScene()). Reads pbrtEnvTexture
    // (see the kernel's own texture argument comment) via a plain
    // equirectangular lookup.
    uint pbrtHasImageEnvLight;
    // Importance-sampling dimensions for pbrtEnvTexture's own SEPARATE
    // EnvDistribution2D (section 96) - same idea as envMapWidth/
    // envMapHeight above, built from a DIFFERENT image (pbrtEnvTexture,
    // not earthTexture) via metal_poc_host_math.h's float-RGB
    // buildEnvDistribution2D() overload (pbrt's own infinite-light image
    // is already decoded linear float, unlike earthTexture's 8-bit JPEG
    // source). 0 (every scene without a pbrt-loaded image-based infinite
    // light) skips this NEE strategy entirely in every material's own
    // shading function, same "0 disables it" pattern as envMapWidth.
    uint pbrtEnvMapWidth;
    uint pbrtEnvMapHeight;
    // Orthographic (parallel-projection) camera - pbrt-v4's own
    // OrthographicCamera, D2/D6's own real port (section 150). 0 (every
    // earlier scene) keeps the existing pinhole/thin-lens PERSPECTIVE
    // fan exactly as before - purely additive, same "0 reproduces prior
    // behaviour" shape every other camera-feature flag above already
    // has. != 0 switches the primary ray generation from a fanned
    // direction (`cameraForward + screen.x*cameraRight + screen.y*
    // cameraUp`) to a CONSTANT direction (`cameraForward`) with the
    // SAME `screen.x/screen.y` instead offsetting the ray's ORIGIN -
    // `tanHalfFov` is reused unchanged as the orthographic screen
    // window's own half-extent (not a tangent at all here, just a
    // world-space distance) since the aspect-ratio scaling already
    // applied to `screen` above is IDENTICAL to pbrt-v4's own
    // `compute_screen_window()` shape (`xmax`, `ymax` in [-1,1] or
    // aspect-scaled), just missing that formula's own final `*320`
    // (or whatever literal) multiplier - which is exactly what setting
    // `tanHalfFov` to that literal (pre-scaled by `sceneScale`, the
    // same world-space-distance convention `pbrtLensRadius` already
    // uses) supplies.
    uint cameraOrthographic;
    // Spherical (360-degree equirectangular panorama) camera - pbrt-v4's
    // own SphericalCamera, D7/D8's own real port (section 152). 0
    // (every earlier scene) keeps the existing ray generation exactly
    // as before - purely additive, same shape as `cameraOrthographic`
    // immediately above (mutually exclusive with it in practice, never
    // both nonzero for the same scene). != 0 replaces the ENTIRE
    // perspective/orthographic direction computation with pbrt-v4's own
    // SphericalCamera::GenerateRay() formula (EquiRectangular mapping):
    // `theta = pi*v, phi = 2*pi*u` (`u`/`v` the SAME `pixelNDC.x/y`
    // already computed for every other mode, BEFORE the `screen.y =
    // -screen.y` flip that only the perspective/orthographic modes
    // need), `rayDir = -sin(theta)*cos(phi)*cameraRight +
    // cos(theta)*cameraUp + sin(theta)*sin(phi)*cameraForward` - the
    // leading MINUS on the `cameraRight` term is the SAME right-vector
    // sign correction `cameraOrthographic`'s own comment explains (CPU's
    // `cameras.h::make_look_at()` alt-camera path's `right` is the
    // negation of this loader's own `cameraRight`, which always matches
    // CPU's PRIMARY perspective camera instead). `cameraPos` is used
    // directly as the ray origin (no lens/DOF, no screen-window offset -
    // every pixel shares one origin, only the DIRECTION varies, the
    // defining trait of a panoramic camera).
    uint cameraSpherical;
    // Realistic (multi-element-lens) camera - pbrt-v4's own
    // RealisticCamera, D4/D8's own real port (section 157). 0 (every
    // earlier scene) keeps the existing ray generation exactly as
    // before - mutually exclusive with cameraOrthographic/cameraSpherical
    // in practice. != 0 replaces the ENTIRE primary ray generation with
    // sampleRealisticCameraRay()'s own real per-element Snell's-law
    // lens trace, reading the `lensElements`/`exitPupilBounds` buffers
    // below - see metal_poc.mm's own Uniforms::cameraRealistic mirrored
    // comment for the full host-side precompute mechanism.
    uint cameraRealistic;
    uint numLensElements;
    uint numExitPupilBounds;
    float filmHalfX;
    float filmHalfY;
    float lensRearZ;
};

// A real light LIST entry, replacing the single hardcoded kLightCenter/
// kLightHalfExtents/kLightNormal/kLightArea/kLightEmission constants
// steps 4/5's own comments explicitly flagged as "one hardcoded light's
// shape baked into the integrator... a real port would carry a proper
// light list instead." A general parallelogram (center + two full edge
// vectors, not axis-aligned half-extents) rather than the old x/z-only
// shape - edgeU/edgeV/normal/area are precomputed host-side once, not
// re-derived per shading sample.
struct AreaLight {
    packed_float3 center;
    packed_float3 edgeU;
    packed_float3 edgeV;
    packed_float3 normal;
    float area;
    packed_float3 emission;
    // Spatially-varying emission via the SAME checkerboard idea
    // materialType 6 already uses for albedo (checkerColor(), tile A is
    // `emission` above, tile B a fixed fraction of it) - `patternTileB`
    // is that fraction; 0.0 (every light before this one) means "tile B
    // is pure black," which combined with `patternScale <= 0.0` below
    // instead SKIPPING the pattern entirely reproduces flat, uniform
    // emission exactly - purely additive, the same "0 is an exact no-op"
    // shape this POC's own knobs already use.
    float patternTileB;
    // Tiles per 0-1 UV unit - reuses the light's OWN NEE sample point
    // (`u.x, u.y` in sampleAreaLight(), the same planar 0-1
    // parameterization addQuad()'s own UVs already establish for this
    // exact quad) as the pattern's own UV, needing no new per-light UV
    // data at all. <= 0.0 means "no pattern," see above.
    float patternScale;
    // pbrt-v4's own "bool twosided" AreaLightSource parameter (section
    // 104) - 0.0 (every light before this one) means this light only
    // emits from the side its own `normal` points toward, matching this
    // POC's original one-sided-only behaviour exactly; nonzero accepts
    // BOTH signs of cosLight in every NEE call site's own visibility
    // check, and the direct-hit code's own facing check (mirrored onto
    // TriangleMaterial::twoSided there, not read from here, so it also
    // covers a non-quad/disk emissive shape with no AreaLight entry).
    float twoSided;
    // A pbrt-loaded scene's own image-based AreaLightSource
    // ("string filename", section 105) - 0.0 (every light before this
    // one) keeps `emission` a flat, direct radiance value; nonzero
    // means `emission` instead holds a pure (scale,scale,scale)
    // MULTIPLIER (pbrt-v4 semantics: image pixel value * scale, no
    // meaningful "L" once an image is given - DiffuseAreaLight ignores
    // L entirely once "filename" is set), and sampleAreaLight() samples
    // pbrtAreaLightTexture at its own NEE sample point's (u.x, u.y)
    // instead of using `emission`/the checker pattern directly. Only
    // ONE textured light is supported (same "one shared slot" tier as
    // sections 90/98's own single-image texture slots) and only for a
    // QUAD-shaped light (this loader's own disk primitive has no UV
    // parameterization to sample a real image against at all, unlike a
    // quad's own addQuad()-assigned UVs) - a disk-shaped textured light
    // still falls back to flat L, a real, separate, still-open gap.
    float useTexture;
    // Power-proportional light-picking data, host-computed once by
    // metal_poc.mm's buildPowerLightSampler() (a direct port of
    // src/shared/power_light_sampler_scaffold.h's own PowerLightSampler -
    // see that function's own comment for the full "why"). `pmf` is this
    // light's own overall selection probability (`power[i] / totalPower`),
    // used directly by both sampleAreaLight() and the direct-hit MIS
    // branch below in place of this POC's old flat `1.0 / lightCount`.
    // `aliasProb`/`aliasIndex` are this light's own Vose alias-table slot
    // (threshold + fallback index) - together they let sampleAreaLight()
    // pick a power-weighted light index in O(1), no loop or running-sum
    // search over `lights` needed despite the non-uniform probabilities.
    float pmf;
    float aliasProb;
    uint aliasIndex;
};

// A true DELTA light - zero-area, zero-solid-angle, unlike every AreaLight
// above. Fundamentally simpler to importance-sample than an area light,
// not just a smaller version of one: a delta light has probability
// EXACTLY zero of ever being hit by a BSDF-sampled continuation ray (the
// same "measure zero" reasoning the mirror/dielectric materials already
// use to skip NEE for THEIR own delta lobes, mirrored here from the
// light's side instead of the material's), so there is no competing
// BSDF-sampling strategy to weight against at all - every point light's
// own NEE contribution is added at full weight, unconditionally, with no
// power-heuristic MIS blend and no area-to-solid-angle pdf conversion
// (a delta light's own "pdf" only ever appears as an idealized 1/distSq
// falloff, not a real probability density needing normalization the way
// a light-PICKING probability or an area-sampling pdf does). Point lights
// are NOT part of the `lights[]` array/light-picking scheme above at
// all - since there is no variance-reduction benefit to stochastically
// picking among a handful of always-fully-weighted point lights the way
// there is for choosing among many area lights, every point light's
// contribution is simply summed each bounce instead (see the shading
// loop's own point-light loop below).
// `direction`/`cosOuterAngle`/`cosInnerAngle` turn this into a SPOT light
// when set (Blinn/pbrt-v4's own smoothstep cone falloff between the two
// angles - full emission inside the inner cone, smoothly fading to zero
// at the outer one, not a hard-edged cone) - `cosOuterAngle <= -1.0`
// (every point light added before this one, and this struct's own
// default) means "omnidirectional, not a spot," skipping the falloff
// computation entirely rather than risking a degenerate `smoothstep`
// with equal edges. A spot light is still fundamentally the SAME delta
// light PointLight already is (zero area, no MIS/pdf needed, summed
// unconditionally, never picked) - the cone is a multiplicative falloff
// on its own emission, not a different light TYPE requiring different
// integration math.
struct PointLight {
    packed_float3 position;
    packed_float3 emission;
    packed_float3 direction;
    float cosOuterAngle;
    float cosInnerAngle;
};

// Blinn/pbrt-v4's own smooth cone falloff: 0 outside the outer cone, 1
// inside the inner cone, smoothstep-interpolated in between - `wiFromLight`
// points FROM the light TOWARD the shading point (the direction the
// spot's own light actually travels), compared against the spot's own
// aim `direction`.
inline float spotLightFalloff(float3 wiFromLight, float3 direction, float cosOuterAngle, float cosInnerAngle) {
    if (cosOuterAngle <= -1.0) {
        return 1.0; // omnidirectional - every point light before this one
    }
    float cosAngle = dot(normalize(wiFromLight), normalize(direction));
    if (cosAngle < cosOuterAngle) return 0.0;
    if (cosAngle > cosInnerAngle) return 1.0;
    float t = (cosAngle - cosOuterAngle) / max(cosInnerAngle - cosOuterAngle, 1e-6);
    return t * t * (3.0 - 2.0 * t); // smoothstep
}

// A directional ("sun") light - like PointLight, a true delta light
// (zero solid angle, always full NEE weight, no MIS, no pdf conversion),
// but positioned at infinity rather than at a finite point: every shading
// point sees the SAME fixed incoming direction, and there is no 1/distSq
// falloff at all (a real sun's own distance makes that falloff
// imperceptibly close to constant across any scene-sized region) - the
// limit of a point light as distance goes to infinity and emission grows
// to compensate, not a separate kind of light needing new integration
// theory. `direction` is the direction the light itself travels (from
// the sun toward the scene), matching PointLight's own `direction`
// convention for the spot cone.
//
// Originally shipped with fog attenuation skipped entirely for this
// light (unlike PointLight's own shadow ray, which has a known finite
// distance `exp(-fogSigmaT * dist)` can use directly) - a directional
// light's shadow ray has no target distance at all, the light being at
// infinity. Fixed below (`rayBoxExitDistance()`) rather than left as a
// permanent gap: since this scene's own fog fills exactly the room's
// solid geometry, and an UNOCCLUDED shadow ray (by definition, this
// branch only runs when the real scene intersection test found nothing)
// can only have exited through this room's one gap (the open front),
// the distance to where that ray crosses this room's own `[-1,1]^3`
// bounds is exactly the real fog path length - not an arbitrary
// sentinel standing in for one.
struct DirectionalLight {
    packed_float3 direction;
    packed_float3 emission;
};

// A "slide projector" light (pbrt-v4's own ProjectionLight, src/shared/
// projection_light.h section 12.5) - a FOURTH delta light type alongside
// PointLight/spot/DirectionalLight, same "summed unconditionally every
// bounce, never picked, no MIS/pdf needed" integration shape every
// earlier delta light already established. Unlike a spot light's own
// smooth cone falloff (a single scalar), this one projects an actual
// IMAGE through a perspective frustum - a real gobo/slide-projector
// effect, not just a differently-shaped intensity curve. Reuses the
// scene's own already-loaded `earthTexture` (bound at texture(1) since
// this POC's very first texture-mapping PR) as the projected image
// rather than needing a second texture binding - a world map projected
// onto a wall/floor like a slide projector, not a new asset.
// `right`/`up`/`forward` are a precomputed (host-side, once)
// orthonormal light-space basis - the same "precompute once, don't
// re-derive per shading sample" approach AreaLight's own edgeU/edgeV/
// normal already uses. `tanHalfFovX`/`tanHalfFovY` fold the frustum's
// half-angle AND aspect ratio into two scalars (rather than a separate
// fov + aspect the way pbrt-v4's own screenBounds derivation does),
// since this POC only ever needs the two independent screen-space
// extents, not pbrt-v4's own more general aspect>=1-vs-aspect<1 code
// path.
struct ProjectionLight {
    packed_float3 position;
    packed_float3 forward;
    packed_float3 right;
    packed_float3 up;
    float tanHalfFovX;
    float tanHalfFovY;
    float scale;
    // Mirrors metal_poc.mm's own ProjectionLightData::usePbrtTexture
    // comment (section 98) - selects pbrtProjectionTexture instead of
    // the room's own shared earthTexture at every call site below.
    uint usePbrtTexture;
};

// Evaluates a ProjectionLight's own emitted intensity toward a shading
// point, given `wiFromLight` (the direction the light itself travels,
// FROM the light TOWARD the shading point - the same convention
// spotLightFalloff() already uses for its own `wiFromLight`). Mirrors
// pbrt-v4 ProjectionLight::I(): reject anything behind the projector
// (`lz <= hither`, a small positive epsilon rather than exactly 0, same
// reasoning pbrt-v4's own default hither has - avoids a degenerate
// divide right at the projector's own image plane), perspective-project
// into normalized screen space, reject anything outside the frustum's
// own screen bounds, then sample the image at the resulting UV. No
// separate falloff curve the way spotLightFalloff() has - the image
// itself (all zero outside the visible frustum, real pixel values
// inside it) already IS the full directional intensity function.
inline float3 projectionLightRadiance(float3 wiFromLight, packed_float3 lightForward,
                                       packed_float3 lightRight, packed_float3 lightUp,
                                       float tanHalfFovX, float tanHalfFovY, float scale,
                                       texture2d<float, access::sample> image, sampler s) {
    float3 wi = normalize(wiFromLight);
    float lz = dot(wi, float3(lightForward));
    const float kHither = 0.001;
    if (lz <= kHither) {
        return float3(0.0); // Behind (or exactly at) the projector's own image plane.
    }
    float lx = dot(wi, float3(lightRight)) / lz;
    float ly = dot(wi, float3(lightUp)) / lz;
    float sx = lx / tanHalfFovX;
    float sy = ly / tanHalfFovY;
    if (abs(sx) > 1.0 || abs(sy) > 1.0) {
        return float3(0.0); // Outside the projector's own frustum.
    }
    // Screen space [-1,1]^2 -> image UV [0,1]^2. Y flipped (screen +Y is
    // "up," image +V is conventionally "down," the same flip every other
    // texture-sampling UV convention in this file already assumes) so
    // the projected image reads right-side-up from the projector's own
    // point of view, not mirrored top-to-bottom.
    float2 uv = float2(sx * 0.5 + 0.5, 0.5 - sy * 0.5);
    return image.sample(s, uv).rgb * scale;
}

// pbrt-v4's own equal-area octahedral sphere<->square mapping
// (src/shared/sampling_extra.h's EqualAreaSphereToSquare(), itself
// mirroring pbrt-v4 util/math.cpp) - a direct, faithful port (Clarberg
// 2008's minimax polynomial approximation of atan, not a re-derivation).
// Unlike equirectangularUV()'s own longitude/latitude mapping (which
// distorts area heavily near the poles - equal SOLID ANGLE regions map
// to very UNEQUAL image-space area there), this maps the WHOLE sphere to
// a single [0,1]^2 square with equal-AREA fidelity everywhere, the
// standard choice for a goniometric (IES-profile) light's own stored
// image: pbrt-v4 requires it specifically so a uniformly-sampled image
// pixel corresponds to a uniformly-likely direction, not a pole-biased
// one - this POC only ever uses the forward (direction-to-UV) half
// below (a goniometric light's own `eval_I` is a pure lookup, never an
// image-guided direction SAMPLE), so only EqualAreaSphereToSquare
// itself is ported, not its own inverse or WrapEqualAreaSquare.
inline float2 equalAreaSphereToSquare(float3 w) {
    float x = abs(w.x), y = abs(w.y), z = abs(w.z);
    float r = sqrt(max(0.0, 1.0 - z));
    float a = max(x, y);
    float b = min(x, y);
    b = (a == 0.0) ? 0.0 : b / a;

    const float t1 = 0.406758566246788489601959989e-5;
    const float t2 = 0.636226545274016134946890922156;
    const float t3 = 0.61572017898280213493197203466e-2;
    const float t4 = -0.247333733281268944196501420480;
    const float t5 = 0.881770664775316294736387951347e-1;
    const float t6 = 0.419038818029165735901852432784e-1;
    const float t7 = -0.251390972343483509333252996350e-1;
    float phi = t1 + b * (t2 + b * (t3 + b * (t4 + b * (t5 + b * (t6 + b * t7)))));
    if (x < y) phi = 1.0 - phi;

    float vv = phi * r;
    float uu = r - vv;
    if (w.z < 0.0) {
        float tmp = uu;
        uu = 1.0 - vv;
        vv = 1.0 - tmp;
    }
    uu = copysign(uu, w.x);
    vv = copysign(vv, w.y);
    return float2(0.5 * (uu + 1.0), 0.5 * (vv + 1.0));
}

// A "goniometric" light (pbrt-v4's own GoniometricLight, src/shared/
// goniometric_light.h) - a FIFTH delta light type, modeling a real IES
// photometric profile: unlike a spot light's own single symmetric cone,
// a genuine light fixture's intensity can vary in complex, non-radially-
// symmetric ways (real IES files often show scalloped, multi-lobed, or
// asymmetric distributions) - captured here as a 2D image indexed by
// direction (via equalAreaSphereToSquare()), the same "the image itself
// IS the directional intensity function" idea projectionLightRadiance()
// already uses, just addressed by DIRECTION FROM the light instead of
// a perspective-projected UV onto a distant plane. No real IES data file
// exists in this repo, so `image` here is a small PROCEDURALLY generated
// pattern (metal_poc.mm's own buildGoniometricProfileImage()) rather
// than a new binary asset - concentric rings of varying brightness
// around the light's own forward axis, chosen specifically because it's
// a directional pattern a plain scalar cone falloff (spotLightFalloff())
// could never produce, the same "prove this is doing something a
// simpler existing light couldn't" reasoning every earlier delta light
// increment already used.
struct GoniometricLight {
    packed_float3 position;
    packed_float3 forward;
    packed_float3 right;
    packed_float3 up;
    packed_float3 emission;
    float scale;
    // Mirrors ProjectionLight::usePbrtTexture above - selects
    // pbrtGoniometricTexture instead of the room's own shared
    // goniometricTexture.
    uint usePbrtTexture;
};

inline float3 goniometricLightRadiance(float3 wiFromLight, packed_float3 lightForward,
                                        packed_float3 lightRight, packed_float3 lightUp,
                                        packed_float3 emission, float scale,
                                        texture2d<float, access::sample> image, sampler s) {
    float3 wi = normalize(wiFromLight);
    float3 local = float3(dot(wi, float3(lightRight)), dot(wi, float3(lightUp)), dot(wi, float3(lightForward)));
    float2 uv = equalAreaSphereToSquare(local);
    float intensity = image.sample(s, uv).r;
    return float3(emission) * intensity * scale;
}

// A shadow ray toward a directional light has no real target distance
// (the light is at infinity) - this is just "farther than anything in
// this room's own [-1,1]^3 extent could be," so an unoccluded shadow ray
// reads as having genuinely exited the scene rather than being clipped
// short of a real occluder.
constant float kDirectionalLightMaxDistance = 10.0f;

// A "firefly" clamp: a rare, extremely bright single-sample outlier
// (a shadow ray that happens to graze very close to a light's own edge,
// giving it a tiny solid-angle pdf and therefore a huge NEE weight, or a
// specular chain that happens to line up with a light just so) that,
// left alone, dominates that pixel's own average out of proportion to
// its real probability - the classic "salt and pepper" bright-pixel
// noise a path tracer shows at low sample counts even where the true
// expected radiance is modest. Clamping each SAMPLE's own total
// radiance (not the final image, and not per-bounce-contribution) to
// this ceiling before folding it into the accumulator introduces a
// small, well-known, deliberately-accepted BIAS (a true outlier's own
// excess energy is discarded, not redistributed) in exchange for a much
// faster-converging, far less noisy image - the standard practical
// trade-off production renderers already make, not a free lunch. Scaled
// per-channel (preserves the sample's own hue, only caps its
// brightness) rather than a flat per-channel clamp, which would shift
// colour at the point of clamping. Tuned by actually rendering at a
// deliberately low (16) sample count and comparing, not picked from
// theory alone - 60 (comfortably above every light's own top emission
// magnitude, ~15-20) turned out too high to visibly touch this scene's
// own worst noise cluster (a fog/volumetric NEE hotspot near the spot
// light's own cone) at all; 3 visibly dimmed the ceiling lights'
// legitimate direct-view brightness, an unacceptable bias. 20 is the
// honest middle ground: still occasionally clips a LEGITIMATE bright
// sample (a direct, unlucky view of a light source's own upper range),
// not "guaranteed never to touch a real value" the way a much higher
// threshold would be, but the reduction in visible low-sample-count
// noise is real and worth that small trade, and it is invisible at this
// scene's own committed high-quality sample counts either way.
constant float kFireflyClampLuminance = 20.0f;

// Adaptive sampling's own convergence parameters, ported directly from
// this project's own CPU integrator (src/shared/adaptive_sampling.h) -
// same values, not re-tuned for this scene, since they're already a
// real, working default (Cycles' own adaptive_threshold) rather than an
// arbitrary starting guess. `kAdaptiveThreshold`: a pixel is "converged"
// once its own running mean's standard error, relative to that mean,
// drops below this. `kAdaptiveBlackFloor`: a near-black pixel (mean
// below this) is converged unconditionally rather than divided into a
// permanently-large relative error by a near-zero mean - matches
// Cycles' own behaviour of not endlessly re-sampling background/shadow
// pixels correctly converging toward zero. `kAdaptiveMinSamples`: never
// even CHECK convergence before this many samples - the CPU version's
// own `min(2 * sqrt_spp, 32)` doesn't translate directly (this POC's
// own sample loop isn't the CPU's stratified sqrt_spp x sqrt_spp grid),
// so a flat, comparably-sized minimum is used instead.
constant float kAdaptiveThreshold = 0.01f;
constant float kAdaptiveBlackFloor = 1e-4f;
constant uint kAdaptiveMinSamples = 16u;

// This scene's own room bounds (see metal_poc.mm's own floor/ceiling/
// wall addQuad() calls) - an explicit, documented scene-specific
// constant, the same category as this file's own hardcoded Suzanne
// instance_id threshold, not a general-purpose scene-bounds mechanism.
constant float3 kRoomBoundsMin = float3(-1.0, -1.0, -1.0);
constant float3 kRoomBoundsMax = float3(1.0, 1.0, 1.0);

// Distance from `origin` (assumed INSIDE the box) to where a ray leaves
// the axis-aligned box `[boxMin, boxMax]` - the standard "far" slab-test
// distance (the near one is behind the ray, since origin is inside).
// Used to give the directional light's own fog attenuation a REAL path
// length instead of skipping it (see DirectionalLight's own comment):
// valid specifically because it's only ever called on an UNOCCLUDED
// shadow ray, which therefore can only have exited through this room's
// one actual gap, not through a solid wall this box-only test doesn't
// know about.
//
// Every call site guards this behind `uniforms.fogSigmaT > 0.0` (a
// real, previously-latent bug found by section 143/B23): `origin`
// genuinely IS always inside `[kRoomBoundsMin, kRoomBoundsMax]` for
// every Cornell-family/hardcoded-room-scaled scene, but a much larger,
// far-offset bespoke scene (B23's own prism, sitting around x=8, not
// [-1,1]) violates that precondition outright. Combined with a light
// direction that has an EXACTLY-zero component on some axis (common
// for an axis-ish-aligned directional light), `1.0/dir` divides by
// zero into +-Infinity, and `boxMin/Max - origin` being the SAME sign
// on that axis (since origin sits entirely outside the box) makes both
// slab planes agree on that sign too - the overall min() then returns
// +-Infinity, not a finite number. Multiplying an unconditionally-
// computed `fogSigmaT * exitDist` by a `fogSigmaT` of EXACTLY 0.0 does
// NOT save this (`0 * Infinity` is NaN, not 0 - the finite-times-zero
// intuition doesn't apply), silently poisoning `radiance` for the rest
// of that shading call. Skipping the call entirely whenever there's no
// fog to attenuate (the common case for most scenes) is both the fix
// and a free perf win, cheaper than trying to make this function safe
// to call outside its own documented precondition.
inline float rayBoxExitDistance(float3 origin, float3 dir, float3 boxMin, float3 boxMax) {
    float3 invDir = 1.0 / dir;
    float3 tPlane1 = (boxMin - origin) * invDir;
    float3 tPlane2 = (boxMax - origin) * invDir;
    float3 tFar = max(tPlane1, tPlane2);
    return min(min(tFar.x, tFar.y), tFar.z);
}

// materialType: 0 = Lambertian diffuse, 1 = mirror (perfect specular),
// 2 = dielectric (glass), 3 = textured Lambertian (same BSDF/NEE code path
// as 0, just samples `earthTexture` at the hit's interpolated UV for
// albedo instead of reading `color` - see the shading loop below), 4 =
// rough conductor (GGX microfacet metal - `color` is the conductor's
// normal-incidence reflectance F0, not a diffuse albedo), 5 = rough
// (frosted) dielectric - materialType 2's own reflect/refract math,
// VNDF-perturbed - see that branch's own comment for exactly what's and
// isn't modeled. `color` means something different again for {2,5}: a
// per-unit-distance Beer-Lambert ABSORPTION coefficient, not a
// reflectance/tint - see applyBeerLambertAbsorption()'s own comment.
// 6 = procedural checkerboard Lambertian (same BSDF/NEE code path as 0/3
// again - `color` is tile A, tile B is a fixed fraction of it, computed
// analytically from UV with no texture/sampler involved at all, unlike
// materialType 3's image lookup - see checkerColor()'s own comment).
// 7 = procedurally bump-mapped Lambertian (same BSDF/NEE code path as
// 0/3/6 yet again - the ONLY difference is which NORMAL that shared code
// shades with: `facingNormal` is perturbed in tangent space by
// proceduralBumpNormal() before any of it runs, rather than albedo
// changing the way it does for 3/6). Only ever assigned to a primary-
// triangle-buffer primitive (never a sphere/disk/Suzanne instance) since
// tangentFor() needs that buffer's own flat vertex/uv indexing.
// 8 = clearcoat (glossy plastic/car-paint): a stochastic Fresnel-
// weighted MIX of a smooth dielectric clear coat (fixed IOR 1.5, F0 =
// 0.04, always colourless/non-metal - unlike materialType 1's own
// tinted mirror) over a Lambertian diffuse base (`color`, the same
// meaning it has for materialType 0). At each hit, a single random draw
// against the coat's own Fresnel reflectance decides which LOBE this
// bounce samples - the specular coat (probability == its own
// reflectance, so no NEE, same delta-lobe reasoning materialType 1
// already uses) or the diffuse base (probability == 1 - that
// reflectance, full NEE + cosine-sampling, same code shape as
// materialType 0's own branch) - never a blend of both in one bounce.
// This is the standard way to stochastically combine two BRDF lobes
// without bias: sampling each lobe with probability exactly equal to
// its own weight makes that weight cancel out of the estimator
// entirely, so NEITHER branch needs an extra scaling factor at the
// point of the random decision - see the shading loop's own comment on
// this material for why duplicating the diffuse NEE code here (instead
// of falling through to materialType 0's shared branch) is necessary,
// not just convenient.
// 9 = procedurally roughness-mapped GGX conductor - the SAME NEE/BSDF
// shape materialType 4 already uses, but `alphaX`/`alphaY` are computed
// from an analytic UV-space checker pattern (reusing checkerColor(),
// picking between two roughness values instead of two colours) rather
// than being one constant per primitive - patches of near-mirror-smooth
// and rough microfacet regions on the SAME surface, a "worn/scratched
// metal" look. Genuinely different from materialType 4's own anisotropy
// (varies BY DIRECTION at a single point) - this varies BY LOCATION
// (isotropic at any single point, but which isotropic alpha changes
// across the surface). Only ever assigned to a SPHERE in this scene:
// `equirectangularUV()` on the hit's own object-space normal gives a
// texture-space coordinate for free (the same technique the environment
// map already uses for direction-based sampling), rather than needing
// materialType 7's own tangentFor()/real mesh UVs.
// 10 = patterned emissive AreaLight surface - otherwise materialType
// 0's own Lambertian (the SAME light-quad-doubles-as-diffuse-reflector
// behaviour every earlier AreaLight already has for indirect bounces
// landing on it), except the "did a ray land directly on the light"
// check earlier in this loop evaluates a checkerboard pattern
// (checkerColor(), `roughness` as the tile-B fraction) at the hit's own
// UV instead of using `emission` as a flat constant - see AreaLight's
// own `patternTileB`/`patternScale` comment for the NEE-side half of
// this (a DIFFERENT point on the same light, so a different UV, hence
// two separate places evaluating the same pattern rather than one).
// 11 = thin dielectric (pbrt-v4's own ThinDielectricBxDF) - a zero-
// thickness slab (soap film, single window pane), genuinely different
// from materialType 2/5's own SOLID glass: transmission passes straight
// through with no bending at all, and reflectance uses a closed-form
// multi-bounce geometric series instead of a single-interface Fresnel
// term - see the shading loop's own comment on this materialType for
// the full formula. `ior` is meaningful here too (same slot, same
// meaning as materialType 2/5's own refraction index), `roughness` is
// unused (a thin dielectric has no rough/frosted variant modeled here).
// `ior` is meaningful for materialType == 2 and 5 (refraction index) and
// reused, differently, for materialType == 4 (perceptual roughness in
// [0,1], squared into the GGX alpha parameter below) - materialType 4
// and {2,5} never coexist on one primitive, so sharing the slot there
// avoids a second otherwise-almost-always-zero field; materialType 5
// needs both ior AND roughness at once though, hence `roughness` getting
// its own field instead of also trying to overload `ior`. materialType 9
// reuses BOTH the same way materialType 5 does - `ior` the smooth
// patch's own perceptual roughness, `roughness` the rough patch's - two
// roughness values instead of one, picked between by location rather
// than both applying at once the way 5's ior/roughness do. Both fields
// carried on every material anyway (rather than a separate per-type
// struct) since this POC values "one flat array, index by primitive_id"
// simplicity over saving a few bytes on entries that don't use every
// field. `emission` is nonzero only for a light quad's own triangles
// (see the AreaLight struct below) - same "just carry it, don't special-
// case a rare field" reasoning. Checking it unconditionally on every hit
// (regardless of material type) is what makes a light source visible at
// all when a camera/GI ray lands on it directly, on top of the light-
// SAMPLING code below which handles every other surface's direct
// illumination FROM it.
struct TriangleMaterial {
    packed_float3 color;
    uint materialType;
    // Real refraction index for materialType == 2/5/11 (see each
    // branch's own comment); reused again, differently, by materialType
    // == 4/9 (anisotropic/patchy conductor's own alphaX) and now
    // materialType == 14 (velvet's own `sigma` spread parameter, see
    // shadeVelvet's own comment) - the same "one scalar slot, per-
    // materialType meaning" pattern `roughness` already uses below.
    float ior;
    packed_float3 emission;
    // Index into the `lights` buffer for a hit ON one of a light's own
    // emissive triangles - -1 for every non-emissive material. Lets the
    // direct-hit MIS weight (see the emission check in the shading loop)
    // look up exactly which AreaLight's area/normal to weight against,
    // instead of a single global light's constants.
    int lightId;
    // Perceptual roughness for materialType == 5 (rough/frosted
    // dielectric) - unlike materialType == 4's reuse of the `ior` slot
    // for roughness, a rough dielectric genuinely needs both `ior` (real
    // refraction index) and a roughness value at once, so this gets its
    // own field rather than overloading an existing one. ALSO reused,
    // differently again, by materialType == 4 as its own alphaY
    // (anisotropic Y-axis roughness, alongside `ior`'s own alphaX) -
    // `roughness == 0.0` there falls back to the isotropic case (alphaY
    // == alphaX), so this stays a safe no-op for every scene that never
    // sets it. ALSO reused a further time by materialType == 13 (Oren-
    // Nayar rough diffuse) as its own sigma parameter - `roughness ==
    // 0.0` there reduces EXACTLY to plain Lambertian, the same "0 is a
    // safe no-op" contract every other reuse of this field already
    // follows. 0 for every other material type.
    float roughness;
    // Complex IOR (eta + i*k) per RGB channel, materialType == 4/9 only -
    // the real physically-based conductor Fresnel (frComplexRGB(), see
    // its own comment) these two materials now use in place of the flat-
    // tint Schlick approximation fresnelSchlickConductor() still is for
    // materialType == 1's mirror. 0 for every other material type (unread
    // there).
    packed_float3 conductorEta;
    packed_float3 conductorK;
    // Diffuse TRANSMITTANCE tint, materialType == 12 only (`color` is
    // this material's own diffuse REFLECTANCE tint, same convention as
    // materialType 0/3/6/7) - a two-sided translucent diffuser (paper, a
    // leaf, a thin frosted panel), pbrt-v4's own DiffuseTransmissionBxDF
    // (src/shared/bxdfs_layered.h). 0 for every other material type.
    packed_float3 transmitColor;
    // pbrt-v4's own "bool twosided" AreaLightSource parameter (section
    // 104), mirrored from AreaLight::twoSided above so the direct-hit
    // emissive check can read it without a lights[] lookup - needed
    // for a non-quad/disk emissive shape (sections 100/101), which has
    // no AreaLight entry (lightId < 0) to look up in the first place.
    // 0 for every non-emissive material.
    uint twoSided;
};

// A sphere is a custom (non-triangle) primitive - Metal has no built-in
// sphere intersection the way it does for triangles, so this needs an
// explicit bounding-box geometry + an intersection function (below) to
// tell the intersector how to test a ray against one. center/radius is
// the whole shape; material lives in a separate 1:1-indexed array (own
// buffer, not embedded here) purely so sphereIntersectionFunction - which
// only needs geometry, never material - doesn't have to carry material
// data through the intersection-function boundary at all.
struct SphereData {
    packed_float3 center;
    float radius;
};

// Bounding-box intersection functions report their result through
// attribute-tagged fields exactly like this, not a plain return value -
// [[accept_intersection]] tells the intersector whether to keep searching
// past this candidate, [[distance]] is what intersection_result::distance
// reads back on the calling side if accepted.
struct SphereIntersectionResult {
    bool accept [[accept_intersection]];
    float distance [[distance]];
};

// Mirrors MTLPackedFloat4x3's own layout (4 packed_float3 COLUMNS, no
// padding) byte-for-byte - a plain side-channel buffer of per-instance
// transforms, one entry per `instanceDescs[]` slot on the host side,
// indexed here by `intersection_result::instance_id`. Metal's own
// `intersection_result<instancing, ...>` type does NOT expose the
// instance's object-to-world transform as a queryable field (confirmed
// by probing the compiler directly - `object_to_world_transform` and
// several other plausible names all fail to compile, "no member
// named..."), unlike OptiX's `optixGetWorldToObjectTransformMatrix()`
// equivalent. This buffer is this POC's own stand-in: uploaded with
// EXACTLY the same transform values used to build each instance
// descriptor's own `transformationMatrix` (see metal_poc.mm's
// addTransformedSuzanneInstance()), so the two can't drift apart from
// each other by construction.
struct InstanceTransform {
    packed_float3 col0;
    packed_float3 col1;
    packed_float3 col2;
    packed_float3 col3;
};

// Applies only the 3x3 LINEAR part (columns 0-2; column 3 is translation,
// meaningless for a direction) of an instance's own transform to an
// object-space normal, producing the correct world-space one - needed
// for any instance whose transform isn't the identity (translation alone
// leaves a normal's direction unchanged, but Suzanne's second instance
// below also rotates, which does not). Assumes a RIGID transform
// (rotation + translation, no non-uniform scale) - the correct general
// case would need the inverse-transpose of the linear part instead of
// the linear part itself, but every instance transform this POC's own
// scene ever constructs is rigid, so that distinction is deliberately
// not implemented here (documented, not silently assumed away, same
// spirit as this POC's other explicitly-scoped simplifications).
inline float3 transformNormalByInstance(float3 objectNormal, InstanceTransform xf) {
    float3x3 linear = float3x3(float3(xf.col0), float3(xf.col1), float3(xf.col2));
    return normalize(linear * objectNormal);
}

// The tag list here (triangle_data, instancing) has to match the calling
// intersector<instancing, triangle_data>/intersection_function_table<...>'s
// own tags exactly, not just declare bounding_box - a mismatched tag set
// compiles fine but the function silently never gets dispatched at trace
// time (found by bisection: an unconditional-accept version of this
// function still produced zero sphere hits until the tags matched).
// Otherwise: this is callable FROM an intersector, not a kernel entry
// point itself, so no [[buffer(N)]] index collision with primaryRayKernel's
// own bindings to worry about - intersection functions have their own
// independent argument table, bound via MTLIntersectionFunctionTable on
// the host side, not shared with the calling kernel's buffer(0..N)
// bindings at all.
[[intersection(bounding_box, triangle_data, instancing)]]
SphereIntersectionResult sphereIntersectionFunction(
    float3 origin [[origin]],
    float3 direction [[direction]],
    float minDistance [[min_distance]],
    float maxDistance [[max_distance]],
    uint primitiveIndex [[primitive_id]],
    device const SphereData* spheres [[buffer(0)]])
{
    SphereIntersectionResult result;
    result.accept = false;

    SphereData sphere = spheres[primitiveIndex];
    float3 center = float3(sphere.center);
    float3 oc = origin - center;
    float a = dot(direction, direction);
    float bHalf = dot(oc, direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = bHalf * bHalf - a * c;
    if (discriminant < 0.0) return result;

    float sqrtDisc = sqrt(discriminant);
    float t = (-bHalf - sqrtDisc) / a;
    if (t < minDistance || t > maxDistance) {
        t = (-bHalf + sqrtDisc) / a;
        if (t < minDistance || t > maxDistance) return result;
    }

    result.accept = true;
    result.distance = t;
    return result;
}

// A second, genuinely DIFFERENT custom-primitive shape - not another
// sphere (spheres already prove "N primitives sharing ONE intersection
// function", via boundingBoxCount and a shared function/buffer, since
// step 3/PR #4). Every custom primitive in this POC before now has gone
// through function-table SLOT 0 (every geometry's own
// intersectionFunctionTableOffset was 0). This disk goes through slot 1
// instead, next to the spheres' geometry within the SAME primitive
// acceleration structure (metal_poc.mm's own sphereAccelDesc now has TWO
// geometryDescriptors, not one) - the first time this POC's function
// table actually has more than one distinct entry, and the first time
// `intersection_result::geometry_id` (not just primitive_id) matters for
// telling two hits apart.
struct DiskData {
    packed_float3 center;
    packed_float3 normal;
    float radius;
};

struct DiskIntersectionResult {
    bool accept [[accept_intersection]];
    float distance [[distance]];
};

// Plain ray-plane intersection (t = dot(center - origin, normal) /
// dot(direction, normal)) followed by a radius check on the in-plane
// distance from the disk's own centre - the textbook disk-primitive test,
// same shape this project's own CPU disk.h shape uses in spirit (plane
// test + radial bound), not an approximation of it. Declares its OWN
// `[[buffer(1)]]` (not buffer(0), which the sphere intersection function
// above already claims) - within ONE shared MTLIntersectionFunctionTable,
// every function's buffer/texture bindings share a single argument
// namespace, so two DIFFERENT functions needing different data must use
// DIFFERENT buffer indices, bound on the host side via two separate
// `setBuffer:atIndex:` calls on the same table (see metal_poc.mm's own
// comment on this at the function-table setup site) - discovered by
// reasoning through what "shared argument table" actually implies here,
// not by trial and error.
[[intersection(bounding_box, triangle_data, instancing)]]
DiskIntersectionResult diskIntersectionFunction(
    float3 origin [[origin]],
    float3 direction [[direction]],
    float minDistance [[min_distance]],
    float maxDistance [[max_distance]],
    uint primitiveIndex [[primitive_id]],
    device const DiskData* disks [[buffer(1)]])
{
    DiskIntersectionResult result;
    result.accept = false;

    DiskData disk = disks[primitiveIndex];
    float3 center = float3(disk.center);
    float3 normal = float3(disk.normal);
    float denom = dot(direction, normal);
    if (fabs(denom) < 1e-6) return result; // ray parallel to the disk's plane

    float t = dot(center - origin, normal) / denom;
    if (t < minDistance || t > maxDistance) return result;

    float3 hitPoint = origin + direction * t;
    float distSq = length_squared(hitPoint - center);
    if (distSq > disk.radius * disk.radius) return result;

    result.accept = true;
    result.distance = t;
    return result;
}

// ---------------------------------------------------------------------------
// PCG32-ish hash-based PRNG, stateless per call (no persistent generator
// object needed across bounces - each call is reseeded from a running
// state uint carried by the caller). Standard "hash the state, use the
// hash, advance the state" shape; not cryptographic, just decorrelated
// enough for path tracing. Mirrors this project's CPU rng.h in spirit
// (a seedable, bounce-advanced generator) without sharing implementation -
// the CPU renderer's actual PCG32/Sobol samplers are C++ classes with
// their own state layout that doesn't translate directly into an MSL
// per-thread scalar.
// ---------------------------------------------------------------------------
inline uint pcgHash(thread uint& state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

inline float randFloat(thread uint& state) {
    return float(pcgHash(state)) / float(0xFFFFFFFFu);
}

// Mirrors metal_poc.mm's own GpuLensElementData byte-for-byte.
struct LensElement {
    float curvatureRadius;
    float thickness;
    float eta;
    float apertureRadius;
};

// Mirrors metal_poc.mm's own GpuExitPupilBoundsData byte-for-byte.
struct ExitPupilBounds {
    float xMin, xMax, yMin, yMax;
    uint degenerate;
};

// Real multi-element-lens camera ray generation (pbrt-v4 RealisticCamera,
// src/shared/realistic_camera.h) - a direct MSL port of
// gpu/optix/optix_device_helpers.h's own already-shipped CUDA
// `sample_realistic_camera_ray()`, itself mirroring
// RealisticCamera<T>::generate_ray()/trace_lenses_from_film()/
// sample_exit_pupil() exactly (same variable names, same algorithm
// structure - a direct translation, not a re-derivation). `u`/`v` are
// the SAME raw `pixelNDC.x/y` (in [0,1]^2, row-major raster order,
// v=0 at the TOP row) every other camera mode derives `screen` from,
// used BEFORE any of their own `*2-1`/y-flip/aspect/tanHalfFov
// transforms - this mode has no screen window or FOV at all, only a
// real film-plane size (`uniforms.filmHalfX/Y`). `su`/`sv`/`sw`/
// `originWorld` are this loader's own usual `cameraRight`/`cameraUp`/
// `cameraForward`/`cameraPos` (computed the SAME `cross(forward,up)`
// way every other scene's camera already is) - NOT
// RealisticCamera<T>::world_right()/etc's own basis (which would
// inherit CPU's ALT-camera path's own differently-signed `right`,
// section 150's own D6 finding) - the host side deliberately builds
// its RealisticCamera<float> with an IDENTITY camera-to-world (see
// metal_poc_scenes_d.mm) and reads ONLY the lens/exit-pupil/film
// accessors from it, exactly mirroring gpu/optix/scene_builder.cpp's
// own "ctw doesn't matter for these fields" comment.
// Returns false (weight left at 0) if the ray is fully vignetted - a
// real, unbiased Monte Carlo outcome (this exact film position/exit-
// pupil sample genuinely can't reach the scene through this lens
// system), not an error; the caller folds the returned weight straight
// into that sample's own initial `throughput`, so a vignetted sample's
// own zero throughput naturally contributes nothing without any
// separate early-exit needed.
inline bool sampleRealisticCameraRay(constant Uniforms& uniforms,
                                      device const LensElement* lensElements,
                                      device const ExitPupilBounds* exitPupilBounds,
                                      float u, float v, thread uint& rngState,
                                      float3 su, float3 sv, float3 sw, float3 originWorld,
                                      thread float3& outOrigin, thread float3& outDirection,
                                      thread float& outWeight) {
    outWeight = 0.0;
    if (uniforms.numLensElements == 0u || uniforms.numExitPupilBounds == 0u) return false;

    // NO leading negation on pfx (unlike pbrt-v4/CUDA's own
    // `pfx = -sample.pFilm_x`) - a real sign fix found via a mirrored
    // first render, not assumed: the CUDA reference's own `su` is
    // `RealisticCamera<T>::world_right()`, built from CPU's ALT-camera
    // path (`cameras.h::make_look_at()`'s own `cross(up,forward)`,
    // section 150's own D6 finding), but THIS function is deliberately
    // passed this loader's own `cameraRight` (`cross(forward,up)`,
    // matching CPU's PRIMARY camera instead - see this function's own
    // declaration comment for why). Dropping the negation here is
    // algebraically identical to negating `su`'s own final contribution
    // below (pfx enters the whole downstream trace linearly, only ever
    // multiplied by `su` at the very end) - the same "negate the right-
    // vector term" shape D6/D7's own fixes already used, just applied
    // at the INPUT instead of the output since this trace has many
    // intermediate steps between the two.
    float pfx = (2.0 * u - 1.0) * uniforms.filmHalfX;
    float pfy = (2.0 * v - 1.0) * uniforms.filmHalfY;

    // sample_exit_pupil
    float rFilm = sqrt(pfx * pfx + pfy * pfy);
    float filmDiag = 2.0 * sqrt(uniforms.filmHalfX * uniforms.filmHalfX + uniforms.filmHalfY * uniforms.filmHalfY);
    int sz = int(uniforms.numExitPupilBounds);
    int rIndex = int(rFilm / (filmDiag * 0.5) * float(sz));
    if (rIndex >= sz) rIndex = sz - 1;
    if (rIndex < 0) rIndex = 0;
    ExitPupilBounds b = exitPupilBounds[rIndex];
    if (b.degenerate != 0u) return false;

    float area = (b.xMax - b.xMin) * (b.yMax - b.yMin);
    if (area <= 0.0) return false;
    float ppdf = 1.0 / area;

    float u0 = randFloat(rngState), u1 = randFloat(rngState);
    float lx = b.xMin + u0 * (b.xMax - b.xMin);
    float ly = b.yMin + u1 * (b.yMax - b.yMin);

    float sinTheta = (rFilm > 0.0) ? pfy / rFilm : 0.0;
    float cosTheta0 = (rFilm > 0.0) ? pfx / rFilm : 1.0;
    float ppx = cosTheta0 * lx - sinTheta * ly;
    float ppy = sinTheta * lx + cosTheta0 * ly;
    float ppz = uniforms.lensRearZ;

    float rdx = ppx - pfx, rdy = ppy - pfy, rdz = ppz;
    float rLen = sqrt(rdx * rdx + rdy * rdy + rdz * rdz);

    // trace_lenses_from_film: camera space (film z=0, +z toward scene) ->
    // lens space (z flipped): loz=-oz, ldz=-dz.
    float lox = pfx, loy = pfy, loz = 0.0;
    float ldx = rdx, ldy = rdy, ldz = -rdz;
    float elementZ = 0.0;

    for (int i = int(uniforms.numLensElements) - 1; i >= 0; --i) {
        LensElement el = lensElements[i];
        elementZ -= el.thickness;
        bool isStop = (el.curvatureRadius == 0.0);
        float t, nx = 0.0, ny = 0.0, nz = 0.0;

        if (isStop) {
            if (ldz == 0.0) return false;
            t = (elementZ - loz) / ldz;
            if (t < 0.0) return false;
        } else {
            float zCenter = elementZ + el.curvatureRadius;
            float cox = lox, coy = loy, coz = loz - zCenter;
            float A = ldx * ldx + ldy * ldy + ldz * ldz;
            float B = 2.0 * (ldx * cox + ldy * coy + ldz * coz);
            float C = cox * cox + coy * coy + coz * coz - el.curvatureRadius * el.curvatureRadius;
            float disc = B * B - 4.0 * A * C;
            if (disc < 0.0) return false;
            float sq = sqrt(disc);
            float q = (B < 0.0) ? -0.5 * (B - sq) : -0.5 * (B + sq);
            float t0 = q / A;
            float t1 = C / q;
            if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
            bool useCloserT = (ldz > 0.0) != (el.curvatureRadius < 0.0);
            t = useCloserT ? min(t0, t1) : max(t0, t1);
            if (t < 0.0) return false;
            float hx0 = lox + t * ldx, hy0 = loy + t * ldy, hz0 = loz + t * ldz;
            nx = hx0; ny = hy0; nz = hz0 - zCenter;
            float nlen = sqrt(nx * nx + ny * ny + nz * nz);
            if (nlen == 0.0) return false;
            nx /= nlen; ny /= nlen; nz /= nlen;
            if (ldx * nx + ldy * ny + ldz * nz > 0.0) { nx = -nx; ny = -ny; nz = -nz; }
        }

        float hx = lox + t * ldx, hy = loy + t * ldy, hz = loz + t * ldz;
        if (hx * hx + hy * hy > el.apertureRadius * el.apertureRadius) return false;
        lox = hx; loy = hy; loz = hz;

        if (!isStop) {
            float etaI = (el.eta == 0.0) ? 1.0 : el.eta;
            float etaT = (i > 0 && lensElements[i - 1].eta != 0.0) ? lensElements[i - 1].eta : 1.0;
            float len = sqrt(ldx * ldx + ldy * ldy + ldz * ldz);
            float dxn = ldx / len, dyn = ldy / len, dzn = ldz / len;
            float eta = etaI / etaT;
            float cosI = -(dxn * nx + dyn * ny + dzn * nz);
            float sin2T = eta * eta * max(0.0, 1.0 - cosI * cosI);
            if (sin2T >= 1.0) return false;
            float cosT = sqrt(1.0 - sin2T);
            ldx = eta * dxn + (eta * cosI - cosT) * nx;
            ldy = eta * dyn + (eta * cosI - cosT) * ny;
            ldz = eta * dzn + (eta * cosI - cosT) * nz;
        }
    }

    float lensOutOx = lox, lensOutOy = loy, lensOutOz = -loz;
    float lensOutDx = ldx, lensOutDy = ldy, lensOutDz = -ldz;

    float cosThetaW = (rLen > 0.0) ? abs(rdz / rLen) : 0.0;
    float lrz = uniforms.lensRearZ;
    if (lrz <= 0.0) return false;
    float w = (cosThetaW * cosThetaW * cosThetaW * cosThetaW) / (ppdf * lrz * lrz);

    outOrigin = originWorld + lensOutOx * su + lensOutOy * sv + lensOutOz * sw;
    outDirection = normalize(lensOutDx * su + lensOutDy * sv + lensOutDz * sw);
    outWeight = w;
    return true;
}



// Uniform sample on a unit disk (r = sqrt(u1) for area-uniform density,
// not r = u1 - the same sqrt used for the hemisphere sample's own radius
// below, same reason: linear r would bunch samples toward the centre).
// Used only by the thin-lens depth-of-field sample in the kernel below -
// a camera aperture is a flat disk, not a hemisphere, so this is its own
// small helper rather than reusing cosineSampleHemisphere's.
inline float2 sampleUnitDisk(thread uint& rngState) {
    float u1 = randFloat(rngState);
    float u2 = randFloat(rngState);
    float r = sqrt(u1);
    float theta = 2.0 * M_PI_F * u2;
    return float2(r * cos(theta), r * sin(theta));
}

// A regular-polygon aperture instead of a circular one - real camera
// lenses focus light through a finite number of physical aperture
// blades, not a perfect circle, which is exactly why out-of-focus
// highlights ("bokeh") in a real photo read as hexagons/pentagons/etc.
// rather than perfectly round discs; `sampleUnitDisk()` above is the
// idealized circular-aperture limit (infinite blades), a real
// approximation this POC's own DOF used unconditionally through step 19.
// Samples uniformly by picking one of `sides` equal triangular wedges
// (origin - vertex_k - vertex_{k+1}) uniformly at random, then a point
// within that wedge via the standard sqrt-for-uniform-triangle-area
// trick - the textbook regular-polygon sampling construction, not an
// approximation of one.
inline float2 samplePolygonAperture(uint sides, thread uint& rngState) {
    float u1 = randFloat(rngState);
    float u2 = randFloat(rngState);
    float u3 = randFloat(rngState);
    uint blade = min(uint(u1 * float(sides)), sides - 1);
    float angleStep = 2.0 * M_PI_F / float(sides);
    float2 vertexA = float2(cos(angleStep * float(blade)), sin(angleStep * float(blade)));
    float2 vertexB = float2(cos(angleStep * float(blade + 1)), sin(angleStep * float(blade + 1)));
    float s = sqrt(u2);
    return s * ((1.0 - u3) * vertexA + u3 * vertexB);
}

inline float3 cosineSampleHemisphere(float3 normal, thread uint& rngState) {
    float u1 = randFloat(rngState);
    float u2 = randFloat(rngState);
    float r = sqrt(u1);
    float theta = 2.0 * M_PI_F * u2;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0, 1.0 - u1));

    // Build an orthonormal basis around `normal` (Duff et al.'s branchless
    // construction) - same "pick any tangent frame, only the normal
    // matters" approach this project's own onb.h uses for the CPU path.
    float sign = normal.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sign + normal.z);
    float b = normal.x * normal.y * a;
    float3 tangent = float3(1.0 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    float3 bitangent = float3(b, sign + normal.y * normal.y * a, -normal.y);

    return normalize(x * tangent + y * bitangent + z * normal);
}

// UNIFORM (not cosine-weighted) hemisphere sampling - materialType 14's
// own velvet material needs this: Ashikhmin & Shirley's own velvet BRDF
// (see shadeVelvet's own comment) is sampled uniformly in the reference
// this was ported from (Blender Cycles' own bsdf_ashikhmin_velvet.h),
// the same "don't bother importance-sampling a niche lobe's own oddly-
// shaped distribution, plain uniform/cosine sampling is simpler and
// still unbiased, just higher-variance" simplification this POC's own
// Oren-Nayar material (materialType 13) already makes too. Same Duff et
// al. branchless ONB construction as cosineSampleHemisphere() above -
// only the (x,y,z) distribution differs (z = u1 directly, not sqrt(u1),
// the standard uniform-over-solid-angle construction).
inline float3 sampleUniformHemisphere(float3 normal, thread uint& rngState) {
    float u1 = randFloat(rngState);
    float u2 = randFloat(rngState);
    float z = u1;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float theta = 2.0 * M_PI_F * u2;
    float x = r * cos(theta);
    float y = r * sin(theta);

    float sign = normal.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sign + normal.z);
    float b = normal.x * normal.y * a;
    float3 tangent = float3(1.0 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    float3 bitangent = float3(b, sign + normal.y * normal.y * a, -normal.y);

    return normalize(x * tangent + y * bitangent + z * normal);
}

// Barycentric-interpolated shading normal - replaces the earlier flat
// per-triangle face normal (cross product of two edges, same value at
// every point of a triangle) with a real per-vertex-normal blend, using
// the hit's own barycentric coordinates (triangle_data tag on the
// intersector/intersection_result is what makes those available at all).
// For the hand-authored room quads, the three corner normals stored in
// `normals` are all identical (the quad's own flat face normal, written
// that way host-side) so this reduces to exactly the old flat-shading
// behaviour there - only a real mesh with genuinely different per-vertex
// normals (Suzanne's own `vn` data) sees a different, smoothly-varying
// result. Same interpolation `.obj`/pbrt-v4/this project's own CPU
// triangle.h use for a shading normal, not an approximation of it.
inline float3 shadingNormalFor(uint primId, float2 barycentric, device const packed_float3* normals) {
    float3 n0 = float3(normals[primId * 3 + 0]);
    float3 n1 = float3(normals[primId * 3 + 1]);
    float3 n2 = float3(normals[primId * 3 + 2]);
    float w0 = 1.0 - barycentric.x - barycentric.y;
    return normalize(w0 * n0 + barycentric.x * n1 + barycentric.y * n2);
}

// Same barycentric-blend idea as shadingNormalFor(), for texture
// coordinates instead of normals - a UV buffer parallel to
// vertices/normals, same per-triangle-corner indexing. Every non-textured
// primitive's three corners carry (0,0) (see addQuad()'s/loadObjMesh()'s
// host-side default), which interpolates to (0,0) too - harmless, since
// only materialType == 3 ever reads it.
inline float2 texCoordFor(uint primId, float2 barycentric, device const packed_float2* uvs) {
    float2 uv0 = uvs[primId * 3 + 0];
    float2 uv1 = uvs[primId * 3 + 1];
    float2 uv2 = uvs[primId * 3 + 2];
    float w0 = 1.0 - barycentric.x - barycentric.y;
    return w0 * uv0 + barycentric.x * uv1 + barycentric.y * uv2;
}

// A per-triangle tangent (constant across the triangle, same "flat is
// fine here" reasoning addQuad()'s own flat face normal already relies
// on - every quad this POC hand-authors is planar with a single UV
// gradient, not a smoothly-varying mesh surface): the standard position/
// UV partial-derivative construction (solve for the UV-space basis that
// maps to world-space edge1/edge2), same technique pbrt-v4's own
// triangle tangent setup uses. `primId`-indexed the same flat, 3-per-
// triangle way vertices/uvs already are - reuses those two buffers
// directly, no new per-triangle data needed host-side.
inline float3 tangentFor(uint primId, device const packed_float3* verts, device const packed_float2* uvs) {
    float3 p0 = float3(verts[primId * 3 + 0]);
    float3 p1 = float3(verts[primId * 3 + 1]);
    float3 p2 = float3(verts[primId * 3 + 2]);
    float2 uv0 = uvs[primId * 3 + 0];
    float2 uv1 = uvs[primId * 3 + 1];
    float2 uv2 = uvs[primId * 3 + 2];
    float3 edge1 = p1 - p0;
    float3 edge2 = p2 - p0;
    float2 duv1 = uv1 - uv0;
    float2 duv2 = uv2 - uv0;
    float det = duv1.x * duv2.y - duv2.x * duv1.y;
    // A degenerate UV mapping (det == 0, e.g. every corner sharing (0,0) -
    // every non-materialType-7 primitive's own default UVs) has no real
    // tangent to solve for; returning SOME unit vector rather than NaN
    // keeps this safe to call unconditionally, even though only
    // materialType == 7 ever actually uses the result.
    if (abs(det) < 1e-10) {
        return normalize(edge1);
    }
    float f = 1.0 / det;
    float3 tangent = f * (duv2.y * edge1 - duv1.y * edge2);
    return normalize(tangent);
}

// Procedural "egg carton" bump map - an analytic height field h(u,v)
// instead of a sampled normal-map texture (no new image asset needed for
// this POC to demonstrate genuine tangent-space shading-normal
// perturbation): standard bump-mapping math, perturbing the normal by
// the height field's own partial derivatives along the tangent/
// bitangent axes (`normal - dh/du * tangent - dh/dv * bitangent`,
// renormalized) rather than actually displacing geometry - the textbook
// distinction between bump mapping (shading only, what this is) and
// real displacement mapping (which this is NOT). `strength` scales the
// derivative term directly - 0.0 (every material type other than 7)
// exactly reproduces the unperturbed normal, a true no-op, not just a
// visually-close approximation of one.
inline float3 proceduralBumpNormal(float3 normal, float3 tangent, float2 uv, float strength) {
    float3 bitangent = cross(normal, tangent);
    const float freqU = 4.0;
    const float freqV = 4.0;
    float au = uv.x * 2.0 * M_PI_F * freqU;
    float av = uv.y * 2.0 * M_PI_F * freqV;
    // `strength` scales the slope DIRECTLY (capped at 1 by cos/sin, so
    // `strength` itself is the max tilt magnitude along each tangent
    // axis) rather than also carrying a 2*pi*freq amplitude factor - a
    // literal height-field derivative would include that factor, but even
    // at this modest freqU/freqV it inflates to ~25x, drowning out the
    // unit normal entirely regardless of how small `strength` is. Tuned
    // as a slope, not a physical height, the same "whatever reads well"
    // spirit checkerColor()'s own tile-B-darkening fraction already uses
    // instead of a physically-derived constant. freqU/freqV == 4 (a few
    // bumps across this panel's own 0-1 UV span) rather than something
    // higher-frequency - a bump's own spatial period needs to stay well
    // above this scene's pixel footprint per UV unit, or it aliases into
    // per-pixel noise indistinguishable from Monte Carlo grain instead of
    // a visible bump shape (a real mistake this PR's own first attempt at
    // this material made, caught by inspecting a raw shading-normal
    // visualization render, not assumed away).
    float dhdu = strength * cos(au) * sin(av);
    float dhdv = strength * sin(au) * cos(av);
    float3 bumped = normal - (dhdu * tangent + dhdv * bitangent);
    return normalize(bumped);
}

// Standard equirectangular direction-to-UV mapping (longitude from
// atan2, latitude from asin) - a genuinely different way of sampling
// `earthTexture` than texCoordFor()'s own per-vertex-UV lookup above:
// this one has no notion of a surface or a mesh at all, just a ray
// DIRECTION, the way a real environment/IBL map is sampled for a miss
// ray (or, in a fuller renderer, for image-based lighting on rough
// surfaces too - not implemented here, this POC only uses it for the
// miss/"sky" case).
inline float2 equirectangularUV(float3 dir) {
    float u = atan2(dir.z, dir.x) * (1.0 / (2.0 * M_PI_F)) + 0.5;
    float v = asin(clamp(dir.y, -1.0, 1.0)) * (1.0 / M_PI_F) + 0.5;
    return float2(u, v);
}

// --- Environment-map importance sampling (phase 2) --------------------
// Device-side counterpart to gpu/metal/metal_poc_host_math.h's own
// EnvDistribution2D/findCdfInterval/sampleEnvDistribution2D/
// pdfEnvDistribution2D (section 69, phase 1) - same CDF-slope-as-pdf
// convention, same piecewise-constant-bucket binary search, just
// reading `device const float*` buffers instead of a `std::vector`, and
// folding the equirectangular direction<->UV conversion (equirectangularUV()
// above, inverted here) and its own sin(theta) solid-angle Jacobian in
// directly, since a device-side caller wants a world DIRECTION and a
// solid-angle pdf, not an image-space (u,v) and an image-space density -
// that conversion has nowhere else to live.

inline int findCdfIntervalDevice(device const float* cdf, int n, float u) {
    int lo = 0, hi = n;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] <= u) lo = mid; else hi = mid;
    }
    return (lo < n - 1) ? lo : (n - 1);
}

// Draws a world direction from the environment map's own importance
// distribution and returns its solid-angle pdf - mirrors
// sampleEnvDistribution2D()'s own image-space sampling exactly, then
// inverts equirectangularUV() (phi = 2*pi*(u-0.5), lambda = pi*(v-0.5),
// dir = (cos(lambda)*cos(phi), sin(lambda), cos(lambda)*sin(phi))) and
// applies the equirectangular Jacobian (dOmega = 2*pi^2*cos(lambda)
// du*dv, and cos(lambda) == sin(pi*v) - see docs section 71's own
// derivation) to convert the image-space pdf into the solid-angle one
// every other light-sampling strategy in this shader already returns.
inline float3 sampleEnvironmentDirection(device const float* marginalCDF, device const float* conditionalCDF,
                                          int width, int height, float u1, float u2,
                                          thread float& pdfSolidAngle) {
    int row = findCdfIntervalDevice(marginalCDF, height, u1);
    float rowLo = marginalCDF[row], rowHi = marginalCDF[row + 1];
    float rowSpan = max(rowHi - rowLo, 1e-9);
    float dv = (u1 - rowLo) / rowSpan;
    float v = (float(row) + dv) / float(height);
    float rowPdf = rowSpan * float(height);

    device const float* condRow = conditionalCDF + row * (width + 1);
    int col = findCdfIntervalDevice(condRow, width, u2);
    float colLo = condRow[col], colHi = condRow[col + 1];
    float colSpan = max(colHi - colLo, 1e-9);
    float du = (u2 - colLo) / colSpan;
    float u = (float(col) + du) / float(width);
    float colPdf = colSpan * float(width);

    float pdfImage = rowPdf * colPdf;
    float sinTheta = max(sin(M_PI_F * v), 1e-6);
    pdfSolidAngle = pdfImage / (2.0 * M_PI_F * M_PI_F * sinTheta);

    float phi = 2.0 * M_PI_F * (u - 0.5);
    float lambda = M_PI_F * (v - 0.5);
    float cosLambda = cos(lambda);
    return float3(cosLambda * cos(phi), sin(lambda), cosLambda * sin(phi));
}

// Evaluates the SAME solid-angle pdf at an arbitrary world direction -
// what a BSDF-sampled ray that escaped toward some direction needs for
// its own MIS weight against this strategy (the miss-path's own
// contribution, see primaryRayKernel's own comment on this).
inline float pdfEnvironmentDirection(device const float* marginalCDF, device const float* conditionalCDF,
                                      int width, int height, float3 dir) {
    float2 uv = equirectangularUV(dir);
    int row = clamp(int(uv.y * float(height)), 0, height - 1);
    int col = clamp(int(uv.x * float(width)), 0, width - 1);
    float rowPdf = (marginalCDF[row + 1] - marginalCDF[row]) * float(height);
    device const float* condRow = conditionalCDF + row * (width + 1);
    float colPdf = (condRow[col + 1] - condRow[col]) * float(width);
    float pdfImage = rowPdf * colPdf;
    float sinTheta = max(sin(M_PI_F * uv.y), 1e-6);
    return pdfImage / (2.0 * M_PI_F * M_PI_F * sinTheta);
}

// A PROCEDURAL texture (materialType 6) - analytic, computed directly
// from the hit's own UV, no image/sampler involved at all, unlike
// materialType 3's earthTexture lookup or step 18's equirectangularUV()
// (both still ultimately a texture2d::sample() call). `scale` tiles are
// per UV unit; alternating tiles pick `colorA`/`colorB` based on the
// parity of floor(u*scale)+floor(v*scale) - the textbook checkerboard
// construction, same one this project's own CPU checker_texture.h uses.
inline float3 checkerColor(float2 uv, float scale, float3 colorA, float3 colorB) {
    float2 tile = floor(uv * scale);
    float parity = fmod(tile.x + tile.y, 2.0);
    return (abs(parity) < 0.5) ? colorA : colorB;
}

// materialType 16's own albedo function - the REAL 3D world-space
// checkerboard this project's own CPU renderer's checker_texture (src/
// TheRestOfYourLife/texture.h) actually implements: parity of
// floor(p.x/scale)+floor(p.y/scale)+floor(p.z/scale), keyed purely on the
// hit's own world-space POSITION, not a UV coordinate at all - unlike
// checkerColor() above (materialType 6), which needs a triangle's own
// interpolated UV and so only ever fires on a triangle (see that
// function's own comment). Needing no UV is exactly why this one CAN run
// on a sphere hit (checkerColor() above cannot - metal_poc.metal's
// sphere-intersection path computes no UV at all, the documented reason
// category-G's own mesh gallery ground uses a flat quad instead of a
// checker sphere, section 117) - this closes that gap for scenes that
// only ever needed the REAL 3D book-checker in the first place, section
// 121, docs/METAL_GPU_FEASIBILITY.md.
inline float3 checker3DColor(float3 p, float scale, float3 colorA, float3 colorB) {
    float3 cell = floor(p / scale);
    float parity = fmod(abs(cell.x) + abs(cell.y) + abs(cell.z), 2.0);
    return (parity < 0.5) ? colorA : colorB;
}

// materialType 17's own noise field - a direct port of this project's
// own CPU/GPU-shared `src/shared/noise.h` (pbrt-v4's Noise()/
// Turbulence(), CPU_GPU-tagged - already NVCC/CUDA-portable, but MSL
// itself can't #include that header directly, so this is a genuine
// re-transcription of the SAME fixed permutation table and formulas,
// not a from-scratch reimplementation). Section 124, docs/
// METAL_GPU_FEASIBILITY.md. `kNoisePerm` is pbrt-v4's own fixed table
// (noise.cpp) - identical values, do not reorder.
constant int kNoisePerm[512] = {
    151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,
    140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
    247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
     57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
     74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
     60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
     65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
    200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
     52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
    207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
    119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
    129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
    218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
     81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,
    106,157,184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,
    205, 93,222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,
    156,180,
    // second copy (identical to first 256 entries, starting at index 256)
    151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,
      7,225,140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,
      6,148,247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35,
     11, 32, 57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168,
     68,175, 74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,
    229,122, 60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,
    143, 54, 65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89,
     18,169,200,196,135,130,116,188,159, 86,164,100,109,198,173,186,
      3, 64, 52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82,
     85,212,207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,
    170,213,119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,
    172,  9,129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,
    112,104,218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,
    162,241, 81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199
};

// pbrt-v4's own Grad(): maps a lattice-point hash to one of 12 gradient
// directions - direct port of noise.h's own noise_detail::Grad<T>().
inline float noiseGrad(int x, int y, int z, float dx, float dy, float dz) {
    int h = kNoisePerm[kNoisePerm[kNoisePerm[x & 255] + (y & 255)] + (z & 255)];
    h &= 15;
    float u = (h < 8 || h == 12 || h == 13) ? dx : dy;
    float v = (h < 4 || h == 12 || h == 13) ? dy : dz;
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

// pbrt-v4's own NoiseWeight(): quintic C2 smoothstep (6t^5-15t^4+10t^3) -
// direct port, replacing Book-3's older cubic (only C1, visible seams).
inline float noiseWeight(float t) {
    float t3 = t * t * t, t4 = t3 * t, t5 = t4 * t;
    return 6.0 * t5 - 15.0 * t4 + 10.0 * t3;
}

// pbrt-v4's own Noise(x,y,z) - trilinear-interpolated gradient noise in
// [-1,1], direct port of noise.h's own perlin_noise<T>().
inline float perlinNoise3D(float3 p) {
    const float wrap = float(1 << 30);
    p = fmod(p, wrap);
    int3 i = int3(floor(p));
    float3 d = p - float3(i);
    int ix = i.x & 255, iy = i.y & 255, iz = i.z & 255;

    float w000 = noiseGrad(ix,   iy,   iz,   d.x,       d.y,       d.z);
    float w100 = noiseGrad(ix+1, iy,   iz,   d.x - 1.0, d.y,       d.z);
    float w010 = noiseGrad(ix,   iy+1, iz,   d.x,       d.y - 1.0, d.z);
    float w110 = noiseGrad(ix+1, iy+1, iz,   d.x - 1.0, d.y - 1.0, d.z);
    float w001 = noiseGrad(ix,   iy,   iz+1, d.x,       d.y,       d.z - 1.0);
    float w101 = noiseGrad(ix+1, iy,   iz+1, d.x - 1.0, d.y,       d.z - 1.0);
    float w011 = noiseGrad(ix,   iy+1, iz+1, d.x,       d.y - 1.0, d.z - 1.0);
    float w111 = noiseGrad(ix+1, iy+1, iz+1, d.x - 1.0, d.y - 1.0, d.z - 1.0);

    float wx = noiseWeight(d.x), wy = noiseWeight(d.y), wz = noiseWeight(d.z);
    float x00 = mix(w000, w100, wx);
    float x10 = mix(w010, w110, wx);
    float x01 = mix(w001, w101, wx);
    float x11 = mix(w011, w111, wx);
    float y0 = mix(x00, x10, wy);
    float y1 = mix(x01, x11, wy);
    return mix(y0, y1, wz);
}

// pbrt-v4's own Turbulence(), no-antialiasing overload (this project's
// own `turbulence_simple<T>()`, noise.h) - sum of |noise| across
// `maxOctaves`, each octave at 1.99x the previous frequency and
// `omega`x the previous amplitude. Used by materialType 17's own marble
// pattern, matching CPU's `noise_texture`/`perlin::turb()` exactly
// (depth 7, omega 0.5 - see that class's own comment).
inline float turbulenceSimple(float3 p, float omega, int maxOctaves) {
    float sum = 0.0, lambda = 1.0, o = 1.0;
    for (int i = 0; i < maxOctaves; ++i) {
        sum += o * abs(perlinNoise3D(p * lambda));
        lambda *= 1.99;
        o *= omega;
    }
    return sum;
}

// The REAL (unpolarized, real-valued-IOR) Fresnel dielectric
// reflectance - ported directly from this project's own CPU renderer
// (src/shared/fresnel.h's own FrDielectric(), mirroring pbrt-v4's
// scattering.h exactly), NOT Schlick's approximation. An earlier
// version of this comment claimed Schlick's approximation was "the same
// one... this project's own CPU dielectric material use[s]" for this
// exact reflect-vs-refract decision - checked while reviewing this
// exact code and found to be WRONG: this project's own `dielectric`
// material (src/TheRestOfYourLife/material_simple.h) uses
// `DielectricBxDF`, which itself calls FrDielectric, not Schlick - a
// documentation inaccuracy as much as a missed accuracy opportunity.
// `cosThetaI` here is ALREADY guaranteed non-negative by construction
// (computed via `facingNormal`, which always faces the incoming ray -
// see the call site), so the `< 0` flip branch below is dead code for
// how this is actually invoked here, kept anyway for a faithful,
// recognizable port rather than a call-site-specific simplification.
inline float frDielectric(float cosThetaI, float eta) {
    cosThetaI = clamp(cosThetaI, -1.0, 1.0);
    if (cosThetaI < 0.0) {
        eta = 1.0 / eta;
        cosThetaI = -cosThetaI;
    }
    float sin2ThetaI = 1.0 - cosThetaI * cosThetaI;
    float sin2ThetaT = sin2ThetaI / (eta * eta);
    if (sin2ThetaT >= 1.0) {
        return 1.0; // Total internal reflection.
    }
    float cosThetaT = sqrt(max(0.0, 1.0 - sin2ThetaT));
    float rParl = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    float rPerp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
    return (rParl * rParl + rPerp * rPerp) / 2.0;
}

// Beer-Lambert colour absorption for a dielectric (materialType 2/5) -
// real tinted glass absorbs light proportionally to how far it travels
// THROUGH the medium (a thick paperweight reads far more saturated than
// a thin windowpane of the same glass), not by a flat per-bounce
// multiply the way every earlier version of this POC's dielectric
// branches applied `albedo`. Only fires when `!frontFace` (this hit is
// on the surface's own BACKFACE, i.e. the ray is exiting, not entering) -
// for a CONVEX primitive (true of every dielectric shape this POC has,
// the sphere), the segment just travelled (the previous bounce's own
// entry point to this exit hit) was entirely inside the medium, so
// `result.distance` at the EXIT hit is exactly the in-medium path
// length Beer's law needs, no separate distance-tracking state required.
// A concave dielectric could re-enter/exit multiple times without this
// simple per-hit check catching every segment correctly - not handled,
// same "document the assumption, don't silently rely on it" approach
// this POC's other simplifications use.
//
// `mat.color` is reinterpreted here as a per-unit-distance absorption
// COEFFICIENT, not the flat reflectance/tint every other material reads
// it as - {0,0,0} means zero absorption (exp(-0*dist) == 1 exactly, a
// true no-op reproducing perfectly clear glass), not "black," the
// opposite of what {0,0,0} would mean as a reflectance colour elsewhere
// in this same struct.
inline void applyBeerLambertAbsorption(thread float3& throughput, packed_float3 absorption, bool frontFace, float distance) {
    if (!frontFace) {
        throughput *= exp(-float3(absorption) * distance);
    }
}

// ---------------------------------------------------------------------------
// GGX / Trowbridge-Reitz microfacet distribution + height-correlated Smith
// masking-shadowing - the standard model materialType == 4 (rough
// conductor) uses below, same formulation pbrt-v4's own
// TrowbridgeReitzDistribution implements. Genuinely ANISOTROPIC (alphaX,
// alphaY, not a single scalar alpha) - all four take full LOCAL-frame
// vectors (tangent/bitangent/normal components), not just a `NdotX`
// scalar, since the anisotropic case needs the vector's azimuthal
// (tangent/bitangent) components too, not only its angle to the normal.
// Passing alphaX == alphaY reduces every one of these EXACTLY to the
// isotropic formulas this POC used through step 22 (verified
// algebraically, not just assumed) - not a separate code path, the same
// formula degenerating correctly at its own isotropic boundary case,
// same spirit as the Henyey-Greenstein phase function's own g==0 case
// (step 19). Cross-checked against Blender Cycles' own anisotropic GGX
// implementation (`bsdf_aniso_D`/`bsdf_aniso_lambda` in
// intern/cycles/kernel/closure/bsdf_microfacet.h) before being committed
// here, not derived from first principles alone this time.
// ---------------------------------------------------------------------------
inline float ggxD(float3 hLocal, float alphaX, float alphaY) {
    float3 hr = float3(hLocal.x / alphaX, hLocal.y / alphaY, hLocal.z);
    float lenSq = max(dot(hr, hr), 1e-12);
    return (1.0 / M_PI_F) / max(alphaX * alphaY * lenSq * lenSq, 1e-12);
}

// Smith's Lambda function (anisotropic GGX closed form) - how much of a
// microfacet's neighbourhood is masked/shadowed as seen from direction
// `wLocal`, folded into G1/G below rather than used standalone.
inline float ggxLambda(float3 wLocal, float alphaX, float alphaY) {
    float wz2 = max(wLocal.z * wLocal.z, 1e-12);
    float sqrAlphaTanN = (alphaX * alphaX * wLocal.x * wLocal.x + alphaY * alphaY * wLocal.y * wLocal.y) / wz2;
    return 0.5 * (sqrt(1.0 + sqrAlphaTanN) - 1.0);
}

inline float ggxG1(float3 wLocal, float alphaX, float alphaY) {
    return 1.0 / (1.0 + ggxLambda(wLocal, alphaX, alphaY));
}

// Height-correlated Smith masking-shadowing for a full reflection lobe
// (both the view and light direction masked/shadowed jointly, not treated
// as independent) - the same correlated form pbrt-v4 uses, less energy
// loss at grazing angles than a naive G1(wo)*G1(wi) product.
inline float ggxG(float3 woLocal, float3 wiLocal, float alphaX, float alphaY) {
    return 1.0 / (1.0 + ggxLambda(woLocal, alphaX, alphaY) + ggxLambda(wiLocal, alphaX, alphaY));
}

// Schlick's Fresnel approximation for a CONDUCTOR: F0 (reflectance at
// normal incidence) is itself an RGB colour here, not derived from a
// scalar IOR the way frDielectric()'s own real-valued formula is - a
// metal's complex refractive index (real eta + imaginary k, wavelength-
// dependent) is what actually produces that colour, and this whole curve
// is approximated from its own F0 value rather than solving the full
// complex-Fresnel equations, unlike this POC's own dielectric material,
// which now uses the exact real-valued formula (frDielectric()) instead
// of Schlick's own approximation of it.
inline float3 fresnelSchlickConductor(float cosTheta, float3 F0) {
    float t = pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    return F0 + (float3(1.0) - F0) * t;
}

// frComplex() - the REAL complex-valued Fresnel reflectance for a
// conductor interface (pbrt-v4's own FrComplex, src/pbrt/util/
// scattering.h; ported from this POC's own reference copy at
// src/shared/fresnel.h). Unlike fresnelSchlickConductor()'s single-F0
// curve (which can only ever interpolate towards white at grazing
// angles), a genuine complex index of refraction (eta + i*k, both
// wavelength/channel-dependent) reproduces the real per-channel colour
// SHIFT actual metals show at grazing incidence - the same kind of
// upgrade frDielectric() (section 55) already made for this POC's
// dielectric materials over Schlick's own approximation of THAT curve.
// All complex arithmetic expanded manually (no complex<> type on
// Metal), identical to the reference's own GPU-compatible expansion.
inline float frComplex(float cosThetaI, float etaR, float etaK) {
    cosThetaI = clamp(cosThetaI, 0.0, 1.0);
    float sin2I = 1.0 - cosThetaI * cosThetaI;

    // Complex Snell's law: sin2T = sin2I / (etaR + i*etaK)^2
    float denomR = etaR * etaR - etaK * etaK;
    float denomI = 2.0 * etaR * etaK;
    float denomSq = denomR * denomR + denomI * denomI;
    float sin2TR = sin2I * denomR / denomSq;
    float sin2TI = -sin2I * denomI / denomSq;

    // cosT = sqrt(1 - sin2T) via the standard complex sqrt formula.
    float cR = 1.0 - sin2TR;
    float cI = -sin2TI;
    float mag = sqrt(cR * cR + cI * cI);
    float cosTR = sqrt(max(0.0, (mag + cR) * 0.5));
    float cosTI = (cI >= 0.0 ? 1.0 : -1.0) * sqrt(max(0.0, (mag - cR) * 0.5));

    // r_parl = (eta*cosI - cosT) / (eta*cosI + cosT), eta = etaR + i*etaK.
    float ecR = etaR * cosThetaI - cosTR;
    float ecI = etaK * cosThetaI - cosTI;
    float edR = etaR * cosThetaI + cosTR;
    float edI = etaK * cosThetaI + cosTI;
    float edSq = edR * edR + edI * edI;
    float rpR = (ecR * edR + ecI * edI) / edSq;
    float rpI = (ecI * edR - ecR * edI) / edSq;
    float rParlSq = rpR * rpR + rpI * rpI;

    // r_perp = (cosI - eta*cosT) / (cosI + eta*cosT).
    float etcR = etaR * cosTR - etaK * cosTI;
    float etcI = etaR * cosTI + etaK * cosTR;
    float ncR = cosThetaI - etcR;
    float ncI = -etcI;
    float ndR = cosThetaI + etcR;
    float ndI = etcI;
    float ndSq = ndR * ndR + ndI * ndI;
    float rsR = (ncR * ndR + ncI * ndI) / ndSq;
    float rsI = (ncI * ndR - ncR * ndI) / ndSq;
    float rPerpSq = rsR * rsR + rsI * rsI;

    return (rParlSq + rPerpSq) * 0.5;
}

// Evaluates frComplex() independently per RGB channel - the real
// per-channel complex Fresnel this POC's GGX conductor material
// (materialType 4/9) now uses in place of fresnelSchlickConductor().
inline float3 frComplexRGB(float cosThetaI, float3 eta, float3 k) {
    return float3(frComplex(cosThetaI, eta.x, k.x),
                  frComplex(cosThetaI, eta.y, k.y),
                  frComplex(cosThetaI, eta.z, k.z));
}

// Builds an orthonormal (tangent, bitangent) frame around `n` - same Duff
// et al. construction cosineSampleHemisphere uses inline, factored out
// here since GGX sampling needs to move both the outgoing direction and
// the sampled half-vector between world space and this local frame
// explicitly (unlike cosineSampleHemisphere, which only ever produces a
// world-space result and never needs the frame itself back).
inline void buildOnb(float3 n, thread float3& tangent, thread float3& bitangent) {
    float sign = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sign + n.z);
    float b = n.x * n.y * a;
    tangent = float3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    bitangent = float3(b, sign + n.y * n.y * a, -n.y);
}

// Unlike buildOnb() above (an ARBITRARY orthonormal frame - fine for an
// isotropic BRDF/phase function, which is rotationally symmetric around
// the normal so any tangent choice gives an identical result), an
// ANISOTROPIC material's highlight orientation depends on which
// direction the tangent actually points - an arbitrary, discontinuously-
// varying tangent (buildOnb()'s own choice depends on the normal's sign
// bit) would make the anisotropy direction jump around incoherently
// across a curved surface instead of reading as a single consistent
// "brushed" direction. This projects a FIXED world-space reference axis
// onto the tangent plane instead (Gram-Schmidt: bitangent = normalize
// (cross(normal, ref)), tangent = cross(bitangent, normal)) - the same
// construction Blender Cycles' own make_orthonormals_tangent() uses,
// given a real per-vertex tangent there; this POC's analytic sphere has
// no per-vertex tangent data to begin with, so a fixed world axis
// (world-up, falling back to world-X exactly at the poles where up is
// parallel to the normal and the projection would be degenerate) is the
// simplest thing that gives a consistent "lines of longitude" brushed-
// metal pattern instead of an arbitrary one.
// Branchless orthonormal basis from a unit normal - Duff, Burgess,
// Christensen, Hery, Kensler, Liani, Villemin, "Building an Orthonormal
// Basis, Revisited" (JCGT 2017), ported from this POC's own reference
// copy at src/shared/microfacet.h's BuildArbitraryTangentFrame(). Fixes
// a real bug this function used to have: the earlier "switch to a
// different world axis when `normal` gets too close to the reference
// direction" construction (`if (abs(dot(refDir, normal)) > 0.999)
// refDir = ...`) has a HARD DISCONTINUITY exactly at that 0.999
// threshold - as `normal` sweeps across it (e.g. anywhere on a sphere
// whose surface normal passes near world +/-Y), the chosen tangent/
// bitangent axes pop to a completely different orientation with no
// continuous transition. Invisible for an ISOTROPIC GGX lobe
// (rotationally symmetric in the tangent plane, so the frame's own
// orientation never affects the result) - a real, visible seam for a
// genuinely ANISOTROPIC one (materialType 4's own brushed-metal
// sphere, alphaX != alphaY), which is the only caller of this function.
// This formulation (using copysign rather than a manual branch) has no
// singularity anywhere on the unit sphere, unlike the one it replaces.
inline void buildAnisotropicOnb(float3 normal, thread float3& tangent, thread float3& bitangent) {
    float sign = copysign(1.0, normal.z);
    float a = -1.0 / (sign + normal.z);
    float b = normal.x * normal.y * a;
    tangent = float3(1.0 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    bitangent = float3(b, sign + normal.y * normal.y * a, -normal.y);
}

// Henyey-Greenstein phase function - the standard analytic model for
// directional (not just isotropic) volume scattering, same one pbrt-v4's
// own HGPhaseFunction implements. `cosTheta` here is dot(wo, wi) in the
// SAME `wo` convention the GGX conductor code above already uses (wo
// points back toward where the ray came from, i.e. `-rayDir`) - under
// that convention, g > 0 peaking at cosTheta == -1 (wi antiparallel to
// wo, i.e. wi roughly EQUAL to the ray's own original travel direction)
// is exactly "forward scattering," matching the physical convention;
// getting this sign backwards is the single easiest mistake to make with
// this formula, so it's called out explicitly rather than left to be
// inferred from the algebra alone.
inline float henyeyGreensteinPhase(float cosTheta, float g) {
    float denom = 1.0 + g * g + 2.0 * g * cosTheta;
    return (1.0 - g * g) / (4.0 * M_PI_F * denom * sqrt(max(denom, 1e-6)));
}

// Samples a direction from the HG phase function's own distribution
// relative to `wo` (same convention as the evaluation function above -
// the local frame's own Z axis IS wo, via buildOnb(), so the returned
// direction's dot product with wo equals the sampled `cosTheta` by
// construction, consistent with what henyeyGreensteinPhase() expects to
// be called with for MIS/NEE against this same sample). At g == 0 this
// reduces to a uniform-over-the-sphere DISTRIBUTION - a rotationally-
// invariant distribution is identical whether sampled relative to world
// axes or relative to an arbitrary local frame like `wo` - not a
// separate code path that happens to agree, the same formula
// degenerating correctly at its own
// boundary case.
inline float3 sampleHenyeyGreenstein(float3 wo, float g, thread uint& rngState) {
    float u1 = randFloat(rngState);
    float u2 = randFloat(rngState);
    float cosTheta;
    if (abs(g) < 1e-3) {
        cosTheta = 1.0 - 2.0 * u1;
    } else {
        float sqrTerm = (1.0 - g * g) / (1.0 + g - 2.0 * g * u1);
        cosTheta = -1.0 / (2.0 * g) * (1.0 + g * g - sqrTerm * sqrTerm);
    }
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = 2.0 * M_PI_F * u2;
    float3 tangent, bitangent;
    buildOnb(wo, tangent, bitangent);
    return sinTheta * cos(phi) * tangent + sinTheta * sin(phi) * bitangent + cosTheta * wo;
}

// Samples a half-vector from the GGX distribution of VISIBLE normals
// (Heitz 2018, "Sampling the GGX Distribution of Visible Normals"), given
// the outgoing direction `woLocal` already in the local (Z-up == shading
// normal) frame. Dramatically lower variance than importance-sampling
// D(h) directly, especially near grazing angles - the same algorithm
// pbrt-v4's TrowbridgeReitzDistribution::Sample_wm implements, chosen
// here for the same reason: it's what makes a rough-conductor path
// tracer converge in a reasonable sample count instead of needing far
// more samples to beat down grazing-angle noise.
// Picks one light from `lights`, power-proportionally (see AreaLight's own
// `pmf`/`aliasProb`/`aliasIndex` comment - replaces this POC's old uniform
// 1/lightCount picking), and samples a uniform point on its parallelogram -
// the "pick a light, then a point on it" step every NEE call site
// (Lambertian and GGX conductor both) shares verbatim; only what happens
// with the sampled point differs per BSDF. The picked light's own `pmf` is
// returned (not a flat 1/lightCount constant anymore, so it can no longer
// be folded in as a plain scalar the way the old comment here described) -
// every caller multiplies `ls.pmf` into its own area-to-solid-angle pdf
// expression instead.
struct LightSample {
    float3 point;
    float3 normal;
    float3 emission;
    float area;
    float pmf;
    // Mirrors AreaLight::twoSided below (section 104) - copied out of
    // the picked light here so every NEE call site's own cosLight check
    // can read it without a second lights[] lookup.
    float twoSided;
};

inline LightSample sampleAreaLight(device const AreaLight* lights, uint lightCount, thread uint& rngState,
                                    texture2d<float, access::sample> pbrtAreaLightTexture, sampler textureSampler) {
    // max(lightCount, 1u) guards the `- 1` below from underflowing (uint
    // wraps to 0xFFFFFFFF, not -1) if this were ever called on a 0-light
    // scene - not reachable with this POC's own hardcoded 2-light scene,
    // but every NEE call site calls this unconditionally with no count
    // check of its own, so this needs to be safe on its own terms.
    uint lastIdx = max(lightCount, 1u) - 1;
    // Vose alias-table sample (see PowerLightSampler::sample() in
    // src/shared/power_light_sampler_scaffold.h, ported verbatim): map u
    // into [0, lightCount), split into a slot index and its own
    // fractional remainder, then either accept that slot or fall through
    // to its precomputed alias - O(1) regardless of how skewed the
    // per-light probabilities are, unlike a running-sum/binary-search CDF
    // walk over `lights`.
    float scaled = randFloat(rngState) * float(lightCount);
    uint slot = min(uint(scaled), lastIdx);
    float frac = scaled - float(slot);
    uint idx = (frac < lights[slot].aliasProb) ? slot : lights[slot].aliasIndex;
    idx = min(idx, lastIdx);
    AreaLight light = lights[idx];
    float3 edgeU = float3(light.edgeU);
    float3 edgeV = float3(light.edgeV);
    float2 u = float2(randFloat(rngState), randFloat(rngState));
    LightSample result;
    result.point = float3(light.center) - 0.5 * edgeU - 0.5 * edgeV + u.x * edgeU + u.y * edgeV;
    result.normal = float3(light.normal);
    // Patterned emission (see AreaLight's own comment): the SAME (u.x,
    // u.y) this NEE sample point was just built from doubles as the
    // pattern's own UV coordinate, no separate UV needed. `patternScale
    // <= 0.0` (every light before this one) skips this entirely,
    // reproducing flat `light.emission` exactly.
    result.emission = (light.useTexture > 0.0)
        ? pbrtAreaLightTexture.sample(textureSampler, u).rgb * float3(light.emission)
        : (light.patternScale > 0.0)
            ? checkerColor(u, light.patternScale, float3(light.emission), float3(light.emission) * light.patternTileB)
            : float3(light.emission);
    result.area = light.area;
    result.pmf = light.pmf;
    result.twoSided = light.twoSided;
    return result;
}

// Generalized to anisotropic alphaX/alphaY (Heitz 2018's own Section
// 3.2/3.4 stretch-and-unstretch steps, using alphaX/alphaY on their
// respective axes instead of one shared alpha) - alphaX == alphaY
// reduces this exactly to the isotropic version this POC used through
// step 22, cross-checked against Blender Cycles' own
// `microfacet_ggx_sample_vndf` before being committed here.
inline float3 sampleGGXVNDF(float3 woLocal, float alphaX, float alphaY, thread uint& rngState) {
    float3 Vh = normalize(float3(alphaX * woLocal.x, alphaY * woLocal.y, woLocal.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0 ? float3(-Vh.y, Vh.x, 0.0) / sqrt(lensq) : float3(1.0, 0.0, 0.0);
    float3 T2 = cross(Vh, T1);

    float u1 = randFloat(rngState);
    float u2 = randFloat(rngState);
    float r = sqrt(u1);
    float phi = 2.0 * M_PI_F * u2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;

    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    float3 Ne = float3(alphaX * Nh.x, alphaY * Nh.y, max(0.0, Nh.z));
    return normalize(Ne);
}

// GGX multi-scatter energy compensation (phase 2 - see
// metal_poc_host_math.h's own buildGGXEnergyTable() comment for the full
// "why"). Bilinear lookup into the E(roughness, mu) table built once at
// host startup, mirroring sampleGGXEnergyTable()'s own host-side
// interpolation exactly (same index math, same clamping) rather than
// texture2d::sample() - this is plain float data with no image-file/
// sRGB concerns, and a `device const float*` buffer already matches the
// layout buildGGXEnergyTable() itself produces with no repacking needed.
inline float sampleGGXEnergyTableDevice(device const float* E, uint roughRes, uint muRes,
                                         float roughness, float mu) {
    float rf = roughness * float(roughRes) - 0.5;
    float mf = mu * float(muRes) - 0.5;
    int r0 = int(floor(rf)), m0 = int(floor(mf));
    float rt = rf - float(r0), mt = mf - float(m0);
    int r1 = r0 + 1, m1 = m0 + 1;
    r0 = clamp(r0, 0, int(roughRes) - 1);
    r1 = clamp(r1, 0, int(roughRes) - 1);
    m0 = clamp(m0, 0, int(muRes) - 1);
    m1 = clamp(m1, 0, int(muRes) - 1);
    float e00 = E[r0 * int(muRes) + m0];
    float e10 = E[r1 * int(muRes) + m0];
    float e01 = E[r0 * int(muRes) + m1];
    float e11 = E[r1 * int(muRes) + m1];
    float e0 = e00 + (e10 - e00) * rt;
    float e1 = e01 + (e11 - e01) * rt;
    return e0 + (e1 - e0) * mt;
}

// ---------------------------------------------------------------------------
// Per-material shading functions
//
// Each function below implements ONE materialType's own NEE (where
// applicable) + BSDF-sampled-continuation logic - extracted out of
// primaryRayKernel's own `mat.materialType`-dispatched if/else chain
// (which used to inline all of this directly, growing that one function
// past 1,600 lines) so the kernel itself stays a short dispatcher. Every
// function takes the read-only per-hit context it needs (material,
// albedo, hit geometry) plus whatever scene resources its own NEE
// actually reads (only the four NEE-capable materials - conductor,
// clearcoat, diffuse transmission, Lambertian - need light buffers/
// textures/the intersector at all; the four purely-specular ones do
// not), and mutates the bounce loop's own running path state (`rayDir`,
// `rayOrigin`, `throughput`, `radiance`, `bsdfPdf`, `specularBounce`,
// `rngState`) by reference - the EXACT same variables the inlined code
// used to write to directly, just threaded through as `thread &`
// parameters instead of captured implicitly.
//
// Returns `false` only when this bounce's own sampled direction is
// genuinely invalid and the whole path should terminate (materialType
// 4/9's own below-the-hemisphere VNDF sample, the one `break` inside
// this whole dispatch chain before this refactor) - every other
// material always returns `true`. The caller checks this uniformly
// (`if (!shadeXxx(...)) break;`) for every branch, not just the one
// that can actually return false, so a future material added the same
// way doesn't need special-casing at the call site to get this right.

inline bool shadeMirror(float3 albedo, float3 hitPoint, float3 facingNormal,
                         thread float3& rayDir, thread float3& rayOrigin,
                         thread float3& throughput, thread bool& specularBounce) {
    // Mirror: deterministic reflection, no light sampling (a specular
    // surface has zero probability of the shadow ray toward a delta
    // light landing exactly on the reflection vector - NEE simply
    // doesn't apply here, same reason the CPU renderer's own BSDFs skip
    // NEE for specular lobes).
    //
    // Fresnel-weighted, not a flat `albedo` multiply the way every
    // earlier version of this branch did: a real mirror's reflectance
    // rises toward white/uncolored at grazing angles regardless of its
    // base tint (the same physical effect the GGX conductor material
    // already models via this exact function - `albedo` doubles as this
    // surface's own F0 here, the same "colour IS the normal-incidence
    // reflectance" convention that material already established).
    float3 newDir = reflect(rayDir, facingNormal);
    float cosTheta = max(dot(facingNormal, -rayDir), 0.0001);
    float3 fresnel = fresnelSchlickConductor(cosTheta, albedo);
    rayDir = newDir;
    rayOrigin = hitPoint + facingNormal * 0.001f;
    throughput *= fresnel;
    specularBounce = true;
    return true;
}

inline bool shadeDielectric(TriangleMaterial mat, float3 hitPoint, float3 normal, float3 facingNormal,
                             bool frontFace, float hitDistance,
                             thread float3& rayDir, thread float3& rayOrigin,
                             thread float3& throughput, thread bool& specularBounce, thread uint& rngState) {
    // Dielectric (glass): the exact Fresnel dielectric reflectance
    // (frDielectric(), see its own comment) decides reflect vs refract
    // stochastically each bounce - the "one importance-sampled choice
    // per hit, unbiased in expectation" approach pbrt-v4 and this
    // project's own CPU dielectric material use, not a 50/50 split of
    // energy.
    float refractionRatio = frontFace ? (1.0 / mat.ior) : mat.ior;
    float3 unitDir = normalize(rayDir);
    float cosTheta = min(dot(-unitDir, facingNormal), 1.0);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    bool cannotRefract = refractionRatio * sinTheta > 1.0;

    float3 newDir;
    if (cannotRefract || frDielectric(cosTheta, 1.0 / refractionRatio) > randFloat(rngState)) {
        newDir = reflect(unitDir, facingNormal);
    } else {
        newDir = refract(unitDir, facingNormal, refractionRatio);
    }
    rayDir = newDir;
    // Offset along the GEOMETRIC (unflipped-for-facing) normal signed
    // toward the new ray direction, not always `facingNormal` -
    // reflect() and refract() can each send the continuation ray to
    // either side of the surface here, and offsetting on the wrong side
    // re-intersects the same surface immediately (self-shadowing acne).
    rayOrigin = hitPoint + (dot(newDir, normal) > 0.0 ? normal : -normal) * 0.001f;
    applyBeerLambertAbsorption(throughput, mat.color, frontFace, hitDistance);
    specularBounce = true;
    return true;
}

// Recursive-backend dispersion (B23/B24, materialType 22) - the SAME
// simplified 3-representative-wavelength RGB-channel scheme OptiX's own
// recursive (non-wavefront) backend uses, per this scene's own registry
// comment (scene_registry_data.h) - NOT the real continuous spectral
// integration CPU's own `--spectral` path or GPU's own `--wavefront`
// path use, since this loader (like OptiX-recursive) has no per-
// wavelength camera ray/hero-wavelength infrastructure at all, only
// plain RGB throughput. Ported from `gpu/optix/optix_device_helpers.h`'s
// own `MaterialType::Dielectric` dispersive branch + `optix_raygen.h`'s
// own channel-masking step (there split across two programs by OptiX's
// own payload-register architecture; here, in ONE function, since this
// loader's shading kernel is already a single self-contained loop).
//
// Mechanism: `rgbChannel` is per-SAMPLE state (declared once before the
// bounce loop, `kRgbChannelUnset` = "no dispersive hit yet"). The FIRST
// time a path hits this material, a channel (0=R/1=G/2=B) is picked
// uniformly at random and PERSISTS for the rest of that sample (every
// later dispersive hit along the same path - e.g. exiting the same
// prism, or a second dispersive object - reuses it, never re-rolls) -
// matches CPU/wavefront's own "one hero wavelength for the whole path"
// convention. `throughput` is masked to that ONE channel with a
// compensating 3x weight AT THE MOMENT the channel is first chosen (a
// standard unbiased stochastic-channel-selection estimator: each of the
// 3 equally-likely channels, averaged over many samples, reconstructs
// the full-RGB expectation) - every later `radiance +=` naturally
// inherits this through throughput's own ongoing multiply chain, and a
// SAMPLE that never reaches a dispersive hit at all pays nothing extra.
// `kRgbChannelWavelengthNm` are the sRGB primaries' own commonly-cited
// dominant wavelengths - the SAME fixed values this project's own
// `measured` material (materialType 15, `src/TheRestOfYourLife/
// material_pbrt.h`'s `kLambdaR/G/B`) already uses for the identical
// "3 fixed representative wavelengths" purpose, not a fresh/independent
// choice.
//
// `mat.ior` = eta_d (unused directly - kept for parity/debugging only);
// `mat.conductorEta.x/y` reused as the precomputed Cauchy `(A, B)`
// coefficients (`CauchyCoefficientsFromAbbe()`, computed HOST-side once
// at scene-build time - construction-time math, not per-ray, matching
// CPU's own dielectric::make_dispersive() constructor exactly) - an
// otherwise-entirely-unused field for this materialType, the same
// "reuse a field with no other meaning here" convention every earlier
// materialType's own dual-use fields already follow.
constant uint kRgbChannelUnset = 3u;
constant float kRgbChannelWavelengthNm[3] = { 612.0f, 549.0f, 465.0f };

inline float cauchyEta(float lambdaNm, float A, float B) {
    float lambdaUm = lambdaNm * 0.001f;
    return A + B / (lambdaUm * lambdaUm);
}

inline bool shadeDispersiveDielectric(TriangleMaterial mat, float3 hitPoint, float3 normal, float3 facingNormal,
                             bool frontFace, float hitDistance,
                             thread float3& rayDir, thread float3& rayOrigin,
                             thread float3& throughput, thread bool& specularBounce,
                             thread uint& rngState, thread uint& rgbChannel) {
    if (rgbChannel == kRgbChannelUnset) {
        uint newChannel = min(uint(randFloat(rngState) * 3.0), 2u);
        float3 channelMask = float3(newChannel == 0u ? 3.0 : 0.0,
                                     newChannel == 1u ? 3.0 : 0.0,
                                     newChannel == 2u ? 3.0 : 0.0);
        throughput *= channelMask;
        rgbChannel = newChannel;
    }
    float dielectricIor = cauchyEta(kRgbChannelWavelengthNm[rgbChannel],
                                     mat.conductorEta.x, mat.conductorEta.y);

    float refractionRatio = frontFace ? (1.0 / dielectricIor) : dielectricIor;
    float3 unitDir = normalize(rayDir);
    float cosTheta = min(dot(-unitDir, facingNormal), 1.0);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    bool cannotRefract = refractionRatio * sinTheta > 1.0;

    float3 newDir;
    if (cannotRefract || frDielectric(cosTheta, 1.0 / refractionRatio) > randFloat(rngState)) {
        newDir = reflect(unitDir, facingNormal);
    } else {
        newDir = refract(unitDir, facingNormal, refractionRatio);
    }
    rayDir = newDir;
    rayOrigin = hitPoint + (dot(newDir, normal) > 0.0 ? normal : -normal) * 0.001f;
    applyBeerLambertAbsorption(throughput, mat.color, frontFace, hitDistance);
    specularBounce = true;
    return true;
}

inline bool shadeRoughDielectric(TriangleMaterial mat, float3 hitPoint, float3 normal, float3 facingNormal,
                                  bool frontFace, float hitDistance,
                                  thread float3& rayDir, thread float3& rayOrigin,
                                  thread float3& throughput, thread bool& specularBounce, thread uint& rngState) {
    // Rough (frosted) dielectric: the same Schlick-Fresnel reflect-vs-
    // refract decision as materialType 2, but taken about a GGX-VNDF-
    // SAMPLED microfacet normal instead of the smooth geometric one.
    // Energy-conserving throughput correction (G/G1(wo)) derived from
    // src/shared/bxdfs_conductor.h's own validated RoughDielectricBxDF
    // f()/pdf() pair - see docs/METAL_GPU_FEASIBILITY.md section 65 for
    // the full derivation.
    float alpha = max(mat.roughness * mat.roughness, 0.0009);
    float3 tangent, bitangent;
    buildOnb(facingNormal, tangent, bitangent);
    float3 woWorld = -rayDir;
    float3 woLocal = float3(dot(woWorld, tangent), dot(woWorld, bitangent), dot(woWorld, facingNormal));
    woLocal.z = max(woLocal.z, 0.0001);
    float3 hLocal = sampleGGXVNDF(woLocal, alpha, alpha, rngState);
    float3 hWorld = normalize(hLocal.x * tangent + hLocal.y * bitangent + hLocal.z * facingNormal);

    float refractionRatio = frontFace ? (1.0 / mat.ior) : mat.ior;
    float3 unitDir = normalize(rayDir);
    float cosTheta = clamp(dot(-unitDir, hWorld), 0.0, 1.0);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    bool cannotRefract = refractionRatio * sinTheta > 1.0;

    float3 newDir;
    if (cannotRefract || frDielectric(cosTheta, 1.0 / refractionRatio) > randFloat(rngState)) {
        newDir = reflect(unitDir, hWorld);
    } else {
        newDir = refract(unitDir, hWorld, refractionRatio);
    }
    float3 newDirLocal = float3(dot(newDir, tangent), dot(newDir, bitangent), dot(newDir, facingNormal));
    float roughDielectricG = ggxG(woLocal, newDirLocal, alpha, alpha);
    float roughDielectricG1 = ggxG1(woLocal, alpha, alpha);
    throughput *= roughDielectricG / max(roughDielectricG1, 1e-6);
    rayDir = newDir;
    rayOrigin = hitPoint + (dot(newDir, normal) > 0.0 ? normal : -normal) * 0.001f;
    applyBeerLambertAbsorption(throughput, mat.color, frontFace, hitDistance);
    specularBounce = true;
    return true;
}

// B24: the frosted (rough) sibling of materialType 22's own smooth
// dispersive dielectric - same OptiX reference, same
// `MaterialType::RoughDielectric` dispersive branch (a few lines below
// its own smooth `MaterialType::Dielectric` counterpart in
// `optix_device_helpers.h`, deliberately NOT factored into a shared
// helper there either - see that code's own comment on why "exactly 2
// occurrences" isn't worth it). Structurally just
// `shadeRoughDielectric()` (materialType 5) with the SAME `rgbChannel`
// resolution `shadeDispersiveDielectric()` (materialType 22) already
// uses, substituted in place of the flat `mat.ior` - the two dispersive
// materials share the exact same channel-selection mechanism, only the
// underlying (smooth vs. GGX-rough) BSDF differs, matching how
// OptiX's own two dispersive branches are independently-but-identically
// structured. `mat.conductorEta.x/y` is the SAME dual-use Cauchy
// `(A, B)` pair materialType 22 already established (both materials
// otherwise leave that field unused).
inline bool shadeDispersiveRoughDielectric(TriangleMaterial mat, float3 hitPoint, float3 normal, float3 facingNormal,
                                  bool frontFace, float hitDistance,
                                  thread float3& rayDir, thread float3& rayOrigin,
                                  thread float3& throughput, thread bool& specularBounce,
                                  thread uint& rngState, thread uint& rgbChannel) {
    if (rgbChannel == kRgbChannelUnset) {
        uint newChannel = min(uint(randFloat(rngState) * 3.0), 2u);
        float3 channelMask = float3(newChannel == 0u ? 3.0 : 0.0,
                                     newChannel == 1u ? 3.0 : 0.0,
                                     newChannel == 2u ? 3.0 : 0.0);
        throughput *= channelMask;
        rgbChannel = newChannel;
    }
    float dispersiveIor = cauchyEta(kRgbChannelWavelengthNm[rgbChannel],
                                     mat.conductorEta.x, mat.conductorEta.y);

    float alpha = max(mat.roughness * mat.roughness, 0.0009);
    float3 tangent, bitangent;
    buildOnb(facingNormal, tangent, bitangent);
    float3 woWorld = -rayDir;
    float3 woLocal = float3(dot(woWorld, tangent), dot(woWorld, bitangent), dot(woWorld, facingNormal));
    woLocal.z = max(woLocal.z, 0.0001);
    float3 hLocal = sampleGGXVNDF(woLocal, alpha, alpha, rngState);
    float3 hWorld = normalize(hLocal.x * tangent + hLocal.y * bitangent + hLocal.z * facingNormal);

    float refractionRatio = frontFace ? (1.0 / dispersiveIor) : dispersiveIor;
    float3 unitDir = normalize(rayDir);
    float cosTheta = clamp(dot(-unitDir, hWorld), 0.0, 1.0);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    bool cannotRefract = refractionRatio * sinTheta > 1.0;

    float3 newDir;
    if (cannotRefract || frDielectric(cosTheta, 1.0 / refractionRatio) > randFloat(rngState)) {
        newDir = reflect(unitDir, hWorld);
    } else {
        newDir = refract(unitDir, hWorld, refractionRatio);
    }
    float3 newDirLocal = float3(dot(newDir, tangent), dot(newDir, bitangent), dot(newDir, facingNormal));
    float roughDielectricG = ggxG(woLocal, newDirLocal, alpha, alpha);
    float roughDielectricG1 = ggxG1(woLocal, alpha, alpha);
    throughput *= roughDielectricG / max(roughDielectricG1, 1e-6);
    rayDir = newDir;
    rayOrigin = hitPoint + (dot(newDir, normal) > 0.0 ? normal : -normal) * 0.001f;
    applyBeerLambertAbsorption(throughput, mat.color, frontFace, hitDistance);
    specularBounce = true;
    return true;
}

inline bool shadeThinDielectric(TriangleMaterial mat, float3 hitPoint, float3 normal, float3 facingNormal,
                                 thread float3& rayDir, thread float3& rayOrigin,
                                 thread float3& throughput, thread bool& specularBounce, thread uint& rngState) {
    // Thin dielectric (pbrt-v4's own ThinDielectricBxDF) - a zero-
    // thickness slab: transmission passes straight through with no
    // bending, reflectance boosted by a closed-form multi-bounce
    // geometric series. Same `ior` regardless of front/back face - see
    // docs/METAL_GPU_FEASIBILITY.md section 64.
    float cosTheta = max(abs(dot(facingNormal, -rayDir)), 0.0001);
    float thinR = frDielectric(cosTheta, mat.ior);
    if (thinR < 1.0) {
        float thinT = 1.0 - thinR;
        thinR += thinT * thinT * thinR / max(1.0 - thinR * thinR, 1e-6);
    }
    float3 newDir = (randFloat(rngState) < thinR)
        ? reflect(rayDir, facingNormal)
        : rayDir; // straight-through, no bending - zero-thickness slab
    rayDir = newDir;
    rayOrigin = hitPoint + (dot(newDir, normal) > 0.0 ? normal : -normal) * 0.001f;
    // No Beer-Lambert absorption, no material tint - pbrt-v4's own
    // ThinDielectricBxDF::Sample_f hardcodes r=g=b=1, a deliberate,
    // faithful match to the reference, not an oversight.
    specularBounce = true;
    return true;
}

inline bool shadeConductor(TriangleMaterial mat, float3 hitPoint, float3 normal, float3 facingNormal,
                            constant Uniforms& uniforms,
                            device const AreaLight* lights,
                            device const PointLight* pointLights,
                            device const DirectionalLight* directionalLights,
                            device const ProjectionLight* projectionLights,
                            device const GoniometricLight* goniometricLights,
                            device const float* envMarginalCDF,
                            device const float* envConditionalCDF,
                            uint envMapWidth, uint envMapHeight,
                            device const float* pbrtEnvMarginalCDF,
                            device const float* pbrtEnvConditionalCDF,
                            uint pbrtEnvMapWidth, uint pbrtEnvMapHeight,
                            device const float* ggxEnergyTable,
                            uint ggxEnergyRoughRes, uint ggxEnergyMuRes,
                            texture2d<float, access::sample> earthTexture,
                            texture2d<float, access::sample> pbrtEnvTexture,
                            texture2d<float, access::sample> goniometricTexture,
                            texture2d<float, access::sample> pbrtGoniometricTexture,
                            texture2d<float, access::sample> pbrtProjectionTexture,
                            texture2d<float, access::sample> pbrtAreaLightTexture,
                            sampler textureSampler,
                            intersector<instancing, triangle_data> isect,
                            instance_acceleration_structure accelStructure,
                            intersection_function_table<instancing, triangle_data> functionTable,
                            thread float3& rayDir, thread float3& rayOrigin,
                            thread float3& throughput, thread float3& radiance,
                            thread float& bsdfPdf, thread bool& specularBounce, thread uint& rngState) {
    // Rough conductor (GGX metal): structurally the same NEE + BSDF-
    // sampled-continuation + MIS shape as the Lambertian material below -
    // only the BRDF/sampling math changes, from a cosine-weighted
    // diffuse lobe to an importance-sampled microfacet one. Fresnel here
    // is `frComplexRGB(..., mat.conductorEta, mat.conductorK)` - the
    // real per-channel complex conductor Fresnel (section 66) - `albedo`
    // is unread by this material, unlike materialType 1's mirror.
    //
    // Genuinely ANISOTROPIC (materialType 4): `ior` gives alphaX,
    // `roughness` doubles as alphaY (0.0 falls back to isotropic).
    // Genuinely SPATIALLY-VARYING instead (materialType 9): alpha is
    // isotropic at any one point, but which of two roughness values
    // applies switches across the surface via a UV-space checker
    // pattern.
    float alphaX, alphaY;
    if (mat.materialType == 9u) {
        float2 roughnessUV = equirectangularUV(normal);
        float alphaSmooth = max(mat.ior * mat.ior, 0.0009);
        float alphaRough = max(mat.roughness * mat.roughness, 0.0009);
        alphaX = checkerColor(roughnessUV, 6.0, float3(alphaSmooth), float3(alphaRough)).x;
        alphaY = alphaX;
    } else {
        alphaX = max(mat.ior * mat.ior, 0.0009);
        alphaY = (mat.roughness > 0.0) ? max(mat.roughness * mat.roughness, 0.0009) : alphaX;
    }
    float3 tangent, bitangent;
    buildAnisotropicOnb(facingNormal, tangent, bitangent);
    float3 woWorld = -rayDir;
    float3 woLocal = float3(dot(woWorld, tangent), dot(woWorld, bitangent), dot(woWorld, facingNormal));
    woLocal.z = max(woLocal.z, 0.0001);

    // Multi-scatter energy compensation (section 72/73) - `energyScale`
    // recovers the energy single-scatter GGX discards to inter-
    // reflection between microfacets, applied as a flat multiplier on
    // every BRDF value below AND the continuation ray's own throughput
    // update at this function's own tail, all scaled by the SAME factor
    // since it depends only on this hit's own (alpha, view angle), not
    // on which light/direction is being evaluated. The table itself is
    // isotropic-only (built from a single alpha, section 72's own
    // buildGGXEnergyTable()); materialType 4's own genuinely anisotropic
    // alphaX/alphaY collapse to a representative isotropic
    // sqrt(alphaX*alphaY) for this lookup - an approximation, not exact,
    // but the SAME kind of "isotropic energy term applied to an
    // anisotropic lobe" approximation production renderers (including
    // Cycles itself) commonly make, since a full anisotropic energy
    // table would need a third table axis this phase doesn't build.
    float ggxEnergyIsoAlpha = sqrt(alphaX * alphaY);
    float ggxEnergyRoughness = sqrt(ggxEnergyIsoAlpha);
    float ggxE = sampleGGXEnergyTableDevice(ggxEnergyTable, ggxEnergyRoughRes, ggxEnergyMuRes,
                                             ggxEnergyRoughness, woLocal.z);
    float energyScale = 1.0 / max(ggxE, 0.05);

    if (all(mat.emission == float3(0.0))) {
        LightSample ls = sampleAreaLight(lights, uniforms.lightCount, rngState, pbrtAreaLightTexture, textureSampler);
        float3 toLight = ls.point - hitPoint;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        float3 wi = toLight / dist;
        float cosSurface = dot(facingNormal, wi);
        float cosLight = dot(ls.normal, -wi);
        if (cosSurface > 0.0 && (cosLight > 0.0 || (ls.twoSided != 0.0 && cosLight < 0.0))) {
            float3 wiLocal = float3(dot(wi, tangent), dot(wi, bitangent), dot(wi, facingNormal));
            float3 h = normalize(woLocal + wiLocal);
            float NdotO = woLocal.z;
            float NdotI = max(wiLocal.z, 0.0001);
            float Dh = ggxD(h, alphaX, alphaY);
            float G = ggxG(woLocal, wiLocal, alphaX, alphaY);
            float3 F = frComplexRGB(max(dot(woLocal, h), 0.0), mat.conductorEta, mat.conductorK);
            float3 brdf = Dh * G * F * energyScale / max(4.0 * NdotO * NdotI, 1e-6);

            ray shadowRay;
            shadowRay.origin = hitPoint + facingNormal * 0.001f;
            shadowRay.direction = wi;
            shadowRay.min_distance = 0.001f;
            shadowRay.max_distance = dist - 0.002f;
            intersection_result<instancing, triangle_data> shadowResult =
                isect.intersect(shadowRay, accelStructure, functionTable);
            if (shadowResult.type == intersection_type::none) {
                float pdfSolidAngle = (distSq / (ls.area * abs(cosLight))) * ls.pmf;
                float pdfBsdf = (Dh * ggxG1(woLocal, alphaX, alphaY)) / max(4.0 * NdotO, 1e-6);
                float weight = (pdfSolidAngle * pdfSolidAngle)
                    / (pdfSolidAngle * pdfSolidAngle + pdfBsdf * pdfBsdf);
                float transmittance = exp(-uniforms.fogSigmaT * dist);
                radiance += throughput * brdf * ls.emission * cosSurface * transmittance / pdfSolidAngle * weight;
            }
        }

        for (uint pli = 0; pli < uniforms.pointLightCount; ++pli) {
            PointLight pl = pointLights[pli];
            float3 toPointLight = float3(pl.position) - hitPoint;
            float plDistSq = dot(toPointLight, toPointLight);
            float plDist = sqrt(plDistSq);
            float3 plWi = toPointLight / plDist;
            float plCosSurface = dot(facingNormal, plWi);
            if (plCosSurface > 0.0) {
                float3 plWiLocal = float3(dot(plWi, tangent), dot(plWi, bitangent), dot(plWi, facingNormal));
                float3 plH = normalize(woLocal + plWiLocal);
                float plNdotO = woLocal.z;
                float plNdotI = max(plWiLocal.z, 0.0001);
                float plDh = ggxD(plH, alphaX, alphaY);
                float plG = ggxG(woLocal, plWiLocal, alphaX, alphaY);
                float3 plF = frComplexRGB(max(dot(woLocal, plH), 0.0), mat.conductorEta, mat.conductorK);
                float3 plBrdf = plDh * plG * plF * energyScale / max(4.0 * plNdotO * plNdotI, 1e-6);

                ray plShadowRay;
                plShadowRay.origin = hitPoint + facingNormal * 0.001f;
                plShadowRay.direction = plWi;
                plShadowRay.min_distance = 0.001f;
                plShadowRay.max_distance = plDist - 0.002f;
                intersection_result<instancing, triangle_data> plShadowResult =
                    isect.intersect(plShadowRay, accelStructure, functionTable);
                if (plShadowResult.type == intersection_type::none) {
                    float plTransmittance = exp(-uniforms.fogSigmaT * plDist);
                    float plSpot = spotLightFalloff(-plWi, float3(pl.direction), pl.cosOuterAngle, pl.cosInnerAngle);
                    radiance += throughput * plBrdf * float3(pl.emission) * plCosSurface * plSpot * plTransmittance / plDistSq;
                }
            }
        }

        for (uint dli = 0; dli < uniforms.directionalLightCount; ++dli) {
            DirectionalLight dl = directionalLights[dli];
            float3 dlWi = normalize(-float3(dl.direction));
            float dlCosSurface = dot(facingNormal, dlWi);
            if (dlCosSurface > 0.0) {
                float3 dlWiLocal = float3(dot(dlWi, tangent), dot(dlWi, bitangent), dot(dlWi, facingNormal));
                float3 dlH = normalize(woLocal + dlWiLocal);
                float dlNdotO = woLocal.z;
                float dlNdotI = max(dlWiLocal.z, 0.0001);
                float dlDh = ggxD(dlH, alphaX, alphaY);
                float dlG = ggxG(woLocal, dlWiLocal, alphaX, alphaY);
                float3 dlF = frComplexRGB(max(dot(woLocal, dlH), 0.0), mat.conductorEta, mat.conductorK);
                float3 dlBrdf = dlDh * dlG * dlF * energyScale / max(4.0 * dlNdotO * dlNdotI, 1e-6);

                ray dlShadowRay;
                dlShadowRay.origin = hitPoint + facingNormal * 0.001f;
                dlShadowRay.direction = dlWi;
                dlShadowRay.min_distance = 0.001f;
                dlShadowRay.max_distance = kDirectionalLightMaxDistance;
                intersection_result<instancing, triangle_data> dlShadowResult =
                    isect.intersect(dlShadowRay, accelStructure, functionTable);
                if (dlShadowResult.type == intersection_type::none) {
                    // Skip when there is no fog - rayBoxExitDistance()
                    // is only valid for an origin INSIDE the hardcoded
                    // room bounds (see that function.s own comment for
                    // why calling it unconditionally is unsafe).
                    float dlTransmittance = (uniforms.fogSigmaT > 0.0)
                        ? exp(-uniforms.fogSigmaT * rayBoxExitDistance(dlShadowRay.origin, dlWi, kRoomBoundsMin, kRoomBoundsMax))
                        : 1.0;
                    radiance += throughput * dlBrdf * float3(dl.emission) * dlCosSurface * dlTransmittance;
                }
            }
        }

        for (uint pji = 0; pji < uniforms.projectionLightCount; ++pji) {
            ProjectionLight pj = projectionLights[pji];
            float3 toProjLight = float3(pj.position) - hitPoint;
            float pjDistSq = dot(toProjLight, toProjLight);
            float pjDist = sqrt(pjDistSq);
            float3 pjWi = toProjLight / pjDist;
            float pjCosSurface = dot(facingNormal, pjWi);
            if (pjCosSurface > 0.0) {
                float3 pjRadiance = projectionLightRadiance(-pjWi, pj.forward, pj.right, pj.up,
                                                             pj.tanHalfFovX, pj.tanHalfFovY, pj.scale,
                                                             (pj.usePbrtTexture != 0u ? pbrtProjectionTexture : earthTexture), textureSampler);
                if (any(pjRadiance > float3(0.0))) {
                    float3 pjWiLocal = float3(dot(pjWi, tangent), dot(pjWi, bitangent), dot(pjWi, facingNormal));
                    float3 pjH = normalize(woLocal + pjWiLocal);
                    float pjNdotO = woLocal.z;
                    float pjNdotI = max(pjWiLocal.z, 0.0001);
                    float pjDh = ggxD(pjH, alphaX, alphaY);
                    float pjG = ggxG(woLocal, pjWiLocal, alphaX, alphaY);
                    float3 pjF = frComplexRGB(max(dot(woLocal, pjH), 0.0), mat.conductorEta, mat.conductorK);
                    float3 pjBrdf = pjDh * pjG * pjF * energyScale / max(4.0 * pjNdotO * pjNdotI, 1e-6);

                    ray pjShadowRay;
                    pjShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    pjShadowRay.direction = pjWi;
                    pjShadowRay.min_distance = 0.001f;
                    pjShadowRay.max_distance = pjDist - 0.002f;
                    intersection_result<instancing, triangle_data> pjShadowResult =
                        isect.intersect(pjShadowRay, accelStructure, functionTable);
                    if (pjShadowResult.type == intersection_type::none) {
                        float pjTransmittance = exp(-uniforms.fogSigmaT * pjDist);
                        radiance += throughput * pjBrdf * pjRadiance * pjCosSurface * pjTransmittance / pjDistSq;
                    }
                }
            }
        }

        for (uint gli = 0; gli < uniforms.goniometricLightCount; ++gli) {
            GoniometricLight gl = goniometricLights[gli];
            float3 toGoniLight = float3(gl.position) - hitPoint;
            float glDistSq = dot(toGoniLight, toGoniLight);
            float glDist = sqrt(glDistSq);
            float3 glWi = toGoniLight / glDist;
            float glCosSurface = dot(facingNormal, glWi);
            if (glCosSurface > 0.0) {
                float3 glRadiance = goniometricLightRadiance(-glWi, gl.forward, gl.right, gl.up,
                                                              gl.emission, gl.scale,
                                                              (gl.usePbrtTexture != 0u ? pbrtGoniometricTexture : goniometricTexture), textureSampler);
                if (any(glRadiance > float3(0.0))) {
                    float3 glWiLocal = float3(dot(glWi, tangent), dot(glWi, bitangent), dot(glWi, facingNormal));
                    float3 glH = normalize(woLocal + glWiLocal);
                    float glNdotO = woLocal.z;
                    float glNdotI = max(glWiLocal.z, 0.0001);
                    float glDh = ggxD(glH, alphaX, alphaY);
                    float glG = ggxG(woLocal, glWiLocal, alphaX, alphaY);
                    float3 glF = frComplexRGB(max(dot(woLocal, glH), 0.0), mat.conductorEta, mat.conductorK);
                    float3 glBrdf = glDh * glG * glF * energyScale / max(4.0 * glNdotO * glNdotI, 1e-6);

                    ray glShadowRay;
                    glShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    glShadowRay.direction = glWi;
                    glShadowRay.min_distance = 0.001f;
                    glShadowRay.max_distance = glDist - 0.002f;
                    intersection_result<instancing, triangle_data> glShadowResult =
                        isect.intersect(glShadowRay, accelStructure, functionTable);
                    if (glShadowResult.type == intersection_type::none) {
                        float glTransmittance = exp(-uniforms.fogSigmaT * glDist);
                        radiance += throughput * glBrdf * glRadiance * glCosSurface * glTransmittance / glDistSq;
                    }
                }
            }
        }

        // Environment map (importance-sampled NEE, section 71) - an
        // additional light-sampling strategy alongside the ones above,
        // not a replacement: see shadeLambertian's own comment for the
        // full "why." No fog-transmittance factor here (unlike the
        // delta lights above) - the miss-path's own existing
        // unconditional environment contribution (primaryRayKernel's
        // own comment) never applied one either, so this NEE addition
        // stays consistent with that pre-existing limitation rather
        // than introducing a new correctness asymmetry between the two.
        // envMapWidth>0 alone isn't enough to gate this - it's built
        // from earthPixels UNCONDITIONALLY (metal_poc.mm), independent of
        // whether the current scene's own miss path actually uses
        // earthTexture as its sky. A pbrt-loaded scene always sets
        // useEnvironmentMap=0 (its own sky, if any, replaces earthTexture
        // - see buildScene()'s own comment there), so without this check
        // every pbrt scene's materials would incorrectly importance-
        // sample and add light from earthTexture as if it were the
        // active environment, even though nothing in the miss path ever
        // shows it as sky for that render - a real bug found while
        // adding the pbrt-env NEE block below (section 96).
        if (envMapWidth > 0u && uniforms.useEnvironmentMap != 0u) {
            float envPdfSolidAngle;
            float3 envWi = sampleEnvironmentDirection(envMarginalCDF, envConditionalCDF,
                                                       int(envMapWidth), int(envMapHeight),
                                                       randFloat(rngState), randFloat(rngState), envPdfSolidAngle);
            float envCosSurface = dot(facingNormal, envWi);
            if (envCosSurface > 0.0 && envPdfSolidAngle > 1e-9) {
                float3 envWiLocal = float3(dot(envWi, tangent), dot(envWi, bitangent), dot(envWi, facingNormal));
                float3 envH = normalize(woLocal + envWiLocal);
                float envNdotO = woLocal.z;
                float envNdotI = max(envWiLocal.z, 0.0001);
                float envDh = ggxD(envH, alphaX, alphaY);
                float envG = ggxG(woLocal, envWiLocal, alphaX, alphaY);
                float3 envF = frComplexRGB(max(dot(woLocal, envH), 0.0), mat.conductorEta, mat.conductorK);
                float3 envBrdf = envDh * envG * envF * energyScale / max(4.0 * envNdotO * envNdotI, 1e-6);

                ray envShadowRay;
                envShadowRay.origin = hitPoint + facingNormal * 0.001f;
                envShadowRay.direction = envWi;
                envShadowRay.min_distance = 0.001f;
                envShadowRay.max_distance = 1e5f;
                intersection_result<instancing, triangle_data> envShadowResult =
                    isect.intersect(envShadowRay, accelStructure, functionTable);
                if (envShadowResult.type == intersection_type::none) {
                    float2 envUV = equirectangularUV(envWi);
                    float3 envRadiance = earthTexture.sample(textureSampler, envUV).rgb;
                    float envPdfBsdf = (envDh * ggxG1(woLocal, alphaX, alphaY)) / max(4.0 * envNdotO, 1e-6);
                    float envWeight = (envPdfSolidAngle * envPdfSolidAngle)
                        / (envPdfSolidAngle * envPdfSolidAngle + envPdfBsdf * envPdfBsdf);
                    radiance += throughput * envBrdf * envRadiance * envCosSurface / envPdfSolidAngle * envWeight;
                }
            }
        }

        // Same NEE/MIS strategy, for a pbrt-loaded scene's own SEPARATE
        // image-based infinite light (section 96) - a genuinely
        // different texture/CDF pair, gated on its own pbrtEnvMapWidth
        // rather than envMapWidth so it's a pure no-op for every scene
        // without one.
        if (pbrtEnvMapWidth > 0u) {
            float pbrtEnvPdfSolidAngle;
            float3 pbrtEnvWi = sampleEnvironmentDirection(pbrtEnvMarginalCDF, pbrtEnvConditionalCDF,
                                                           int(pbrtEnvMapWidth), int(pbrtEnvMapHeight),
                                                           randFloat(rngState), randFloat(rngState), pbrtEnvPdfSolidAngle);
            float pbrtEnvCosSurface = dot(facingNormal, pbrtEnvWi);
            if (pbrtEnvCosSurface > 0.0 && pbrtEnvPdfSolidAngle > 1e-9) {
                float3 pbrtEnvWiLocal = float3(dot(pbrtEnvWi, tangent), dot(pbrtEnvWi, bitangent), dot(pbrtEnvWi, facingNormal));
                float3 pbrtEnvH = normalize(woLocal + pbrtEnvWiLocal);
                float pbrtEnvNdotO = woLocal.z;
                float pbrtEnvNdotI = max(pbrtEnvWiLocal.z, 0.0001);
                float pbrtEnvDh = ggxD(pbrtEnvH, alphaX, alphaY);
                float pbrtEnvG = ggxG(woLocal, pbrtEnvWiLocal, alphaX, alphaY);
                float3 pbrtEnvF = frComplexRGB(max(dot(woLocal, pbrtEnvH), 0.0), mat.conductorEta, mat.conductorK);
                float3 pbrtEnvBrdf = pbrtEnvDh * pbrtEnvG * pbrtEnvF * energyScale / max(4.0 * pbrtEnvNdotO * pbrtEnvNdotI, 1e-6);

                ray pbrtEnvShadowRay;
                pbrtEnvShadowRay.origin = hitPoint + facingNormal * 0.001f;
                pbrtEnvShadowRay.direction = pbrtEnvWi;
                pbrtEnvShadowRay.min_distance = 0.001f;
                pbrtEnvShadowRay.max_distance = 1e5f;
                intersection_result<instancing, triangle_data> pbrtEnvShadowResult =
                    isect.intersect(pbrtEnvShadowRay, accelStructure, functionTable);
                if (pbrtEnvShadowResult.type == intersection_type::none) {
                    float2 pbrtEnvUV = equirectangularUV(pbrtEnvWi);
                    float3 pbrtEnvRadianceSample = pbrtEnvTexture.sample(textureSampler, pbrtEnvUV).rgb;
                    float pbrtEnvPdfBsdf = (pbrtEnvDh * ggxG1(woLocal, alphaX, alphaY)) / max(4.0 * pbrtEnvNdotO, 1e-6);
                    float pbrtEnvWeight = (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle)
                        / (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle + pbrtEnvPdfBsdf * pbrtEnvPdfBsdf);
                    radiance += throughput * pbrtEnvBrdf * pbrtEnvRadianceSample * pbrtEnvCosSurface / pbrtEnvPdfSolidAngle * pbrtEnvWeight;
                }
            }
        }
    }

    float3 hLocal = sampleGGXVNDF(woLocal, alphaX, alphaY, rngState);
    float3 wiLocal = reflect(-woLocal, hLocal);
    if (wiLocal.z <= 0.0) {
        // Sampled a half-vector whose reflection lands below the
        // hemisphere (possible at grazing angles/high roughness) - a
        // real BRDF value of zero, not a bug; terminate this path
        // rather than continue with an invalid direction.
        return false;
    }
    float3 wiWorld = normalize(wiLocal.x * tangent + wiLocal.y * bitangent + wiLocal.z * facingNormal);

    float NdotO = woLocal.z;
    float G = ggxG(woLocal, wiLocal, alphaX, alphaY);
    float G1 = ggxG1(woLocal, alphaX, alphaY);
    float3 F = frComplexRGB(max(dot(woLocal, hLocal), 0.0), mat.conductorEta, mat.conductorK);
    throughput *= F * (G / max(G1, 1e-6)) * energyScale;

    rayDir = wiWorld;
    rayOrigin = hitPoint + facingNormal * 0.001f;
    bsdfPdf = (ggxD(hLocal, alphaX, alphaY) * G1) / max(4.0 * NdotO, 1e-6);
    specularBounce = false;
    return true;
}

inline bool shadeClearcoat(TriangleMaterial mat, float3 albedo, float3 hitPoint, float3 facingNormal,
                            constant Uniforms& uniforms,
                            device const AreaLight* lights,
                            device const PointLight* pointLights,
                            device const DirectionalLight* directionalLights,
                            device const ProjectionLight* projectionLights,
                            device const GoniometricLight* goniometricLights,
                            device const float* envMarginalCDF,
                            device const float* envConditionalCDF,
                            uint envMapWidth, uint envMapHeight,
                            device const float* pbrtEnvMarginalCDF,
                            device const float* pbrtEnvConditionalCDF,
                            uint pbrtEnvMapWidth, uint pbrtEnvMapHeight,
                            texture2d<float, access::sample> earthTexture,
                            texture2d<float, access::sample> pbrtEnvTexture,
                            texture2d<float, access::sample> goniometricTexture,
                            texture2d<float, access::sample> pbrtGoniometricTexture,
                            texture2d<float, access::sample> pbrtProjectionTexture,
                            texture2d<float, access::sample> pbrtAreaLightTexture,
                            sampler textureSampler,
                            intersector<instancing, triangle_data> isect,
                            instance_acceleration_structure accelStructure,
                            intersection_function_table<instancing, triangle_data> functionTable,
                            thread float3& rayDir, thread float3& rayOrigin,
                            thread float3& throughput, thread float3& radiance,
                            thread float& bsdfPdf, thread bool& specularBounce, thread uint& rngState) {
    // Clearcoat (glossy plastic/car-paint) - `albedo` is the diffuse
    // base colour underneath the coat (materialType 0's own convention),
    // NOT an F0. A stochastic MIX of two lobes: a colourless specular
    // coat (fixed IOR 1.5, F0 = 0.04) with probability exactly equal to
    // its own reflectance (no extra scaling needed - see this material's
    // own section 51), or the diffuse base with the complementary
    // probability (full NEE + cosine-sampling, duplicated here rather
    // than shared with the Lambertian material below since the
    // stochastic coat-vs-base decision has to happen first).
    //
    // Entering-light coat attenuation (below): every diffuse-base
    // contribution is additionally scaled by `(1 - frDielectric(cosX,
    // kClearcoatEta))` at ITS OWN incidence angle - light reaching the
    // diffuse layer from any direction must first penetrate the SAME
    // dielectric coat, independent of which outgoing/view direction is
    // being evaluated. This is distinct from (and not already covered
    // by) the `coatFresnel` probability below: that term already
    // correctly reproduces the OUTGOING-direction attenuation in
    // expectation via unweighted stochastic lobe selection (a standard,
    // unbiased one-sample MC estimator - P(diffuse)=1-coatFresnel(wo),
    // then evaluate the chosen lobe's own BRDF unweighted), so adding an
    // extra `(1-coatFresnel)` factor here would double-count it. pbrt-v4's
    // `NormalizedFresnelBxDF` (src/shared/bxdfs_layered.h) also has a `c`
    // energy-renormalization constant (`1 - 2*FresnelMoment1(1/eta)`)
    // accounting for light trapped and re-emitted by internal reflection
    // inside the diffuse layer - deliberately DEFERRED here, same staging
    // as Oren-Nayar's/GGX's own multi-scatter compensation splits, since
    // it needs pbrt's fuller layered-BxDF stochastic-transport context to
    // get right rather than being a simple standalone factor.
    float cosThetaCoat = max(dot(facingNormal, -rayDir), 0.0001);
    const float kClearcoatF0 = 0.04;
    const float kClearcoatEta = 1.5;
    float coatFresnel = fresnelSchlickConductor(cosThetaCoat, float3(kClearcoatF0)).x;
    if (randFloat(rngState) < coatFresnel) {
        float3 newDir = reflect(rayDir, facingNormal);
        rayDir = newDir;
        rayOrigin = hitPoint + facingNormal * 0.001f;
        specularBounce = true;
    } else {
        if (all(mat.emission == float3(0.0))) {
            LightSample ls = sampleAreaLight(lights, uniforms.lightCount, rngState, pbrtAreaLightTexture, textureSampler);
            float3 toLight = ls.point - hitPoint;
            float distSq = dot(toLight, toLight);
            float dist = sqrt(distSq);
            float3 wi = toLight / dist;
            float cosSurface = dot(facingNormal, wi);
            float cosLight = dot(ls.normal, -wi);
            if (cosSurface > 0.0 && (cosLight > 0.0 || (ls.twoSided != 0.0 && cosLight < 0.0))) {
                ray shadowRay;
                shadowRay.origin = hitPoint + facingNormal * 0.001f;
                shadowRay.direction = wi;
                shadowRay.min_distance = 0.001f;
                shadowRay.max_distance = dist - 0.002f;
                intersection_result<instancing, triangle_data> shadowResult =
                    isect.intersect(shadowRay, accelStructure, functionTable);
                if (shadowResult.type == intersection_type::none) {
                    float pdfSolidAngle = (distSq / (ls.area * abs(cosLight))) * ls.pmf;
                    float pdfBsdfForThisDir = cosSurface / M_PI_F;
                    float weight = (pdfSolidAngle * pdfSolidAngle)
                        / (pdfSolidAngle * pdfSolidAngle + pdfBsdfForThisDir * pdfBsdfForThisDir);
                    float transmittance = exp(-uniforms.fogSigmaT * dist);
                    float coatTransmitIn = 1.0 - frDielectric(cosSurface, kClearcoatEta);
                    radiance += throughput * albedo * (1.0 / M_PI_F) * coatTransmitIn
                                * ls.emission * cosSurface * transmittance / pdfSolidAngle * weight;
                }
            }

            for (uint pli = 0; pli < uniforms.pointLightCount; ++pli) {
                PointLight pl = pointLights[pli];
                float3 toPointLight = float3(pl.position) - hitPoint;
                float plDistSq = dot(toPointLight, toPointLight);
                float plDist = sqrt(plDistSq);
                float3 plWi = toPointLight / plDist;
                float plCosSurface = dot(facingNormal, plWi);
                if (plCosSurface > 0.0) {
                    ray plShadowRay;
                    plShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    plShadowRay.direction = plWi;
                    plShadowRay.min_distance = 0.001f;
                    plShadowRay.max_distance = plDist - 0.002f;
                    intersection_result<instancing, triangle_data> plShadowResult =
                        isect.intersect(plShadowRay, accelStructure, functionTable);
                    if (plShadowResult.type == intersection_type::none) {
                        float plTransmittance = exp(-uniforms.fogSigmaT * plDist);
                        float plSpot = spotLightFalloff(-plWi, float3(pl.direction), pl.cosOuterAngle, pl.cosInnerAngle);
                        float plCoatTransmitIn = 1.0 - frDielectric(plCosSurface, kClearcoatEta);
                        radiance += throughput * albedo * (1.0 / M_PI_F) * plCoatTransmitIn
                                    * float3(pl.emission) * plCosSurface * plSpot * plTransmittance / plDistSq;
                    }
                }
            }

            for (uint dli = 0; dli < uniforms.directionalLightCount; ++dli) {
                DirectionalLight dl = directionalLights[dli];
                float3 dlWi = normalize(-float3(dl.direction));
                float dlCosSurface = dot(facingNormal, dlWi);
                if (dlCosSurface > 0.0) {
                    ray dlShadowRay;
                    dlShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    dlShadowRay.direction = dlWi;
                    dlShadowRay.min_distance = 0.001f;
                    dlShadowRay.max_distance = kDirectionalLightMaxDistance;
                    intersection_result<instancing, triangle_data> dlShadowResult =
                        isect.intersect(dlShadowRay, accelStructure, functionTable);
                    if (dlShadowResult.type == intersection_type::none) {
                        // Skip when there is no fog - rayBoxExitDistance()
                        // is only valid for an origin INSIDE the hardcoded
                        // room bounds (see that function's own comment for
                        // why calling it unconditionally is unsafe).
                        float dlTransmittance = (uniforms.fogSigmaT > 0.0)
                            ? exp(-uniforms.fogSigmaT * rayBoxExitDistance(dlShadowRay.origin, dlWi, kRoomBoundsMin, kRoomBoundsMax))
                            : 1.0;
                        float dlCoatTransmitIn = 1.0 - frDielectric(dlCosSurface, kClearcoatEta);
                        radiance += throughput * albedo * (1.0 / M_PI_F) * dlCoatTransmitIn
                                    * float3(dl.emission) * dlCosSurface * dlTransmittance;
                    }
                }
            }

            for (uint pji = 0; pji < uniforms.projectionLightCount; ++pji) {
                ProjectionLight pj = projectionLights[pji];
                float3 toProjLight = float3(pj.position) - hitPoint;
                float pjDistSq = dot(toProjLight, toProjLight);
                float pjDist = sqrt(pjDistSq);
                float3 pjWi = toProjLight / pjDist;
                float pjCosSurface = dot(facingNormal, pjWi);
                if (pjCosSurface > 0.0) {
                    float3 pjRadiance = projectionLightRadiance(-pjWi, pj.forward, pj.right, pj.up,
                                                                 pj.tanHalfFovX, pj.tanHalfFovY, pj.scale,
                                                                 (pj.usePbrtTexture != 0u ? pbrtProjectionTexture : earthTexture), textureSampler);
                    if (any(pjRadiance > float3(0.0))) {
                        ray pjShadowRay;
                        pjShadowRay.origin = hitPoint + facingNormal * 0.001f;
                        pjShadowRay.direction = pjWi;
                        pjShadowRay.min_distance = 0.001f;
                        pjShadowRay.max_distance = pjDist - 0.002f;
                        intersection_result<instancing, triangle_data> pjShadowResult =
                            isect.intersect(pjShadowRay, accelStructure, functionTable);
                        if (pjShadowResult.type == intersection_type::none) {
                            float pjTransmittance = exp(-uniforms.fogSigmaT * pjDist);
                            float pjCoatTransmitIn = 1.0 - frDielectric(pjCosSurface, kClearcoatEta);
                            radiance += throughput * albedo * (1.0 / M_PI_F) * pjCoatTransmitIn
                                        * pjRadiance * pjCosSurface * pjTransmittance / pjDistSq;
                        }
                    }
                }
            }

            for (uint gli = 0; gli < uniforms.goniometricLightCount; ++gli) {
                GoniometricLight gl = goniometricLights[gli];
                float3 toGoniLight = float3(gl.position) - hitPoint;
                float glDistSq = dot(toGoniLight, toGoniLight);
                float glDist = sqrt(glDistSq);
                float3 glWi = toGoniLight / glDist;
                float glCosSurface = dot(facingNormal, glWi);
                if (glCosSurface > 0.0) {
                    float3 glRadiance = goniometricLightRadiance(-glWi, gl.forward, gl.right, gl.up,
                                                                  gl.emission, gl.scale,
                                                                  (gl.usePbrtTexture != 0u ? pbrtGoniometricTexture : goniometricTexture), textureSampler);
                    if (any(glRadiance > float3(0.0))) {
                        ray glShadowRay;
                        glShadowRay.origin = hitPoint + facingNormal * 0.001f;
                        glShadowRay.direction = glWi;
                        glShadowRay.min_distance = 0.001f;
                        glShadowRay.max_distance = glDist - 0.002f;
                        intersection_result<instancing, triangle_data> glShadowResult =
                            isect.intersect(glShadowRay, accelStructure, functionTable);
                        if (glShadowResult.type == intersection_type::none) {
                            float glTransmittance = exp(-uniforms.fogSigmaT * glDist);
                            float glCoatTransmitIn = 1.0 - frDielectric(glCosSurface, kClearcoatEta);
                            radiance += throughput * albedo * (1.0 / M_PI_F) * glCoatTransmitIn
                                        * glRadiance * glCosSurface * glTransmittance / glDistSq;
                        }
                    }
                }
            }

            // Environment map (importance-sampled NEE, section 71) - see
            // shadeLambertian's own comment for the full "why." Gated on
            // useEnvironmentMap too, not just envMapWidth - see
            // shadeConductor's own comment on why (envMapWidth alone
            // stays nonzero for a pbrt-loaded scene, which always sets
            // useEnvironmentMap=0).
            if (envMapWidth > 0u && uniforms.useEnvironmentMap != 0u) {
                float envPdfSolidAngle;
                float3 envWi = sampleEnvironmentDirection(envMarginalCDF, envConditionalCDF,
                                                           int(envMapWidth), int(envMapHeight),
                                                           randFloat(rngState), randFloat(rngState), envPdfSolidAngle);
                float envCosSurface = dot(facingNormal, envWi);
                if (envCosSurface > 0.0 && envPdfSolidAngle > 1e-9) {
                    ray envShadowRay;
                    envShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    envShadowRay.direction = envWi;
                    envShadowRay.min_distance = 0.001f;
                    envShadowRay.max_distance = 1e5f;
                    intersection_result<instancing, triangle_data> envShadowResult =
                        isect.intersect(envShadowRay, accelStructure, functionTable);
                    if (envShadowResult.type == intersection_type::none) {
                        float2 envUV = equirectangularUV(envWi);
                        float3 envRadiance = earthTexture.sample(textureSampler, envUV).rgb;
                        float envPdfBsdf = envCosSurface / M_PI_F;
                        float envWeight = (envPdfSolidAngle * envPdfSolidAngle)
                            / (envPdfSolidAngle * envPdfSolidAngle + envPdfBsdf * envPdfBsdf);
                        float envCoatTransmitIn = 1.0 - frDielectric(envCosSurface, kClearcoatEta);
                        radiance += throughput * albedo * (1.0 / M_PI_F) * envCoatTransmitIn
                                    * envRadiance * envCosSurface / envPdfSolidAngle * envWeight;
                    }
                }
            }

            // Same NEE/MIS strategy, for a pbrt-loaded scene's own
            // SEPARATE image-based infinite light (section 96) - see
            // shadeConductor's own comment.
            if (pbrtEnvMapWidth > 0u) {
                float pbrtEnvPdfSolidAngle;
                float3 pbrtEnvWi = sampleEnvironmentDirection(pbrtEnvMarginalCDF, pbrtEnvConditionalCDF,
                                                               int(pbrtEnvMapWidth), int(pbrtEnvMapHeight),
                                                               randFloat(rngState), randFloat(rngState), pbrtEnvPdfSolidAngle);
                float pbrtEnvCosSurface = dot(facingNormal, pbrtEnvWi);
                if (pbrtEnvCosSurface > 0.0 && pbrtEnvPdfSolidAngle > 1e-9) {
                    ray pbrtEnvShadowRay;
                    pbrtEnvShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    pbrtEnvShadowRay.direction = pbrtEnvWi;
                    pbrtEnvShadowRay.min_distance = 0.001f;
                    pbrtEnvShadowRay.max_distance = 1e5f;
                    intersection_result<instancing, triangle_data> pbrtEnvShadowResult =
                        isect.intersect(pbrtEnvShadowRay, accelStructure, functionTable);
                    if (pbrtEnvShadowResult.type == intersection_type::none) {
                        float2 pbrtEnvUV = equirectangularUV(pbrtEnvWi);
                        float3 pbrtEnvRadianceSample = pbrtEnvTexture.sample(textureSampler, pbrtEnvUV).rgb;
                        float pbrtEnvPdfBsdf = pbrtEnvCosSurface / M_PI_F;
                        float pbrtEnvWeight = (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle)
                            / (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle + pbrtEnvPdfBsdf * pbrtEnvPdfBsdf);
                        float pbrtEnvCoatTransmitIn = 1.0 - frDielectric(pbrtEnvCosSurface, kClearcoatEta);
                        radiance += throughput * albedo * (1.0 / M_PI_F) * pbrtEnvCoatTransmitIn
                                    * pbrtEnvRadianceSample * pbrtEnvCosSurface / pbrtEnvPdfSolidAngle * pbrtEnvWeight;
                    }
                }
            }
        }

        rayDir = cosineSampleHemisphere(facingNormal, rngState);
        rayOrigin = hitPoint + facingNormal * 0.001f;
        float coatTransmitInNewDir = 1.0 - frDielectric(max(dot(facingNormal, rayDir), 0.0001), kClearcoatEta);
        throughput *= albedo * coatTransmitInNewDir;
        bsdfPdf = max(dot(facingNormal, rayDir), 0.0001) / M_PI_F;
        specularBounce = false;
    }
    return true;
}

inline bool shadeDiffuseTransmission(TriangleMaterial mat, float3 albedo, float3 hitPoint, float3 facingNormal,
                                      constant Uniforms& uniforms,
                                      device const AreaLight* lights,
                                      device const PointLight* pointLights,
                                      device const DirectionalLight* directionalLights,
                                      device const ProjectionLight* projectionLights,
                                      device const GoniometricLight* goniometricLights,
                                      device const float* envMarginalCDF,
                                      device const float* envConditionalCDF,
                                      uint envMapWidth, uint envMapHeight,
                                      device const float* pbrtEnvMarginalCDF,
                                      device const float* pbrtEnvConditionalCDF,
                                      uint pbrtEnvMapWidth, uint pbrtEnvMapHeight,
                                      texture2d<float, access::sample> earthTexture,
                                      texture2d<float, access::sample> pbrtEnvTexture,
                                      texture2d<float, access::sample> goniometricTexture,
                                      texture2d<float, access::sample> pbrtGoniometricTexture,
                                      texture2d<float, access::sample> pbrtProjectionTexture,
                                      texture2d<float, access::sample> pbrtAreaLightTexture,
                                      sampler textureSampler,
                                      intersector<instancing, triangle_data> isect,
                                      instance_acceleration_structure accelStructure,
                                      intersection_function_table<instancing, triangle_data> functionTable,
                                      thread float3& rayDir, thread float3& rayOrigin,
                                      thread float3& throughput, thread float3& radiance,
                                      thread float& bsdfPdf, thread bool& specularBounce, thread uint& rngState) {
    // Diffuse transmission (a two-sided translucent diffuser - paper, a
    // leaf, a thin frosted panel): pbrt-v4's own DiffuseTransmissionBxDF.
    // `albedo` (this material's own `color`) is the REFLECTANCE tint;
    // `mat.transmitColor` the TRANSMITTANCE tint. For any ONE hit point,
    // a given light/continuation direction falls on exactly one side of
    // `facingNormal` - which lobe applies is decided by that single
    // sign, not two separate passes over each light. See section 67.
    float pr = max(albedo.x, max(albedo.y, albedo.z));
    float pt = max(mat.transmitColor.x, max(mat.transmitColor.y, mat.transmitColor.z));
    float pSum = max(pr + pt, 1e-6);

    if (all(mat.emission == float3(0.0))) {
        LightSample ls = sampleAreaLight(lights, uniforms.lightCount, rngState, pbrtAreaLightTexture, textureSampler);
        float3 toLight = ls.point - hitPoint;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        float3 wi = toLight / dist;
        float cosSurface = dot(facingNormal, wi);
        float cosLight = dot(ls.normal, -wi);
        if (cosSurface != 0.0 && (cosLight > 0.0 || (ls.twoSided != 0.0 && cosLight < 0.0))) {
            bool reflect = cosSurface > 0.0;
            float3 lobeTint = reflect ? albedo : mat.transmitColor;
            float lobeProb = reflect ? (pr / pSum) : (pt / pSum);
            float absCos = abs(cosSurface);
            ray shadowRay;
            shadowRay.origin = hitPoint + (reflect ? facingNormal : -facingNormal) * 0.001f;
            shadowRay.direction = wi;
            shadowRay.min_distance = 0.001f;
            shadowRay.max_distance = dist - 0.002f;
            intersection_result<instancing, triangle_data> shadowResult =
                isect.intersect(shadowRay, accelStructure, functionTable);
            if (shadowResult.type == intersection_type::none) {
                float pdfSolidAngle = (distSq / (ls.area * abs(cosLight))) * ls.pmf;
                float pdfBsdfForThisDir = lobeProb * absCos / M_PI_F;
                float weight = (pdfSolidAngle * pdfSolidAngle)
                    / (pdfSolidAngle * pdfSolidAngle + pdfBsdfForThisDir * pdfBsdfForThisDir);
                float transmittance = exp(-uniforms.fogSigmaT * dist);
                radiance += throughput * lobeTint * (1.0 / M_PI_F)
                            * ls.emission * absCos * transmittance / pdfSolidAngle * weight;
            }
        }

        for (uint pli = 0; pli < uniforms.pointLightCount; ++pli) {
            PointLight pl = pointLights[pli];
            float3 toPointLight = float3(pl.position) - hitPoint;
            float plDistSq = dot(toPointLight, toPointLight);
            float plDist = sqrt(plDistSq);
            float3 plWi = toPointLight / plDist;
            float plCosSurface = dot(facingNormal, plWi);
            if (plCosSurface != 0.0) {
                bool plReflect = plCosSurface > 0.0;
                float3 plLobeTint = plReflect ? albedo : mat.transmitColor;
                float plAbsCos = abs(plCosSurface);
                ray plShadowRay;
                plShadowRay.origin = hitPoint + (plReflect ? facingNormal : -facingNormal) * 0.001f;
                plShadowRay.direction = plWi;
                plShadowRay.min_distance = 0.001f;
                plShadowRay.max_distance = plDist - 0.002f;
                intersection_result<instancing, triangle_data> plShadowResult =
                    isect.intersect(plShadowRay, accelStructure, functionTable);
                if (plShadowResult.type == intersection_type::none) {
                    float plTransmittance = exp(-uniforms.fogSigmaT * plDist);
                    float plSpot = spotLightFalloff(-plWi, float3(pl.direction), pl.cosOuterAngle, pl.cosInnerAngle);
                    radiance += throughput * plLobeTint * (1.0 / M_PI_F)
                                * float3(pl.emission) * plAbsCos * plSpot * plTransmittance / plDistSq;
                }
            }
        }

        for (uint dli = 0; dli < uniforms.directionalLightCount; ++dli) {
            DirectionalLight dl = directionalLights[dli];
            float3 dlWi = normalize(-float3(dl.direction));
            float dlCosSurface = dot(facingNormal, dlWi);
            if (dlCosSurface != 0.0) {
                bool dlReflect = dlCosSurface > 0.0;
                float3 dlLobeTint = dlReflect ? albedo : mat.transmitColor;
                float dlAbsCos = abs(dlCosSurface);
                ray dlShadowRay;
                dlShadowRay.origin = hitPoint + (dlReflect ? facingNormal : -facingNormal) * 0.001f;
                dlShadowRay.direction = dlWi;
                dlShadowRay.min_distance = 0.001f;
                dlShadowRay.max_distance = kDirectionalLightMaxDistance;
                intersection_result<instancing, triangle_data> dlShadowResult =
                    isect.intersect(dlShadowRay, accelStructure, functionTable);
                if (dlShadowResult.type == intersection_type::none) {
                    // Skip when there is no fog - rayBoxExitDistance()
                    // is only valid for an origin INSIDE the hardcoded
                    // room bounds (see that function.s own comment for
                    // why calling it unconditionally is unsafe).
                    float dlTransmittance = (uniforms.fogSigmaT > 0.0)
                        ? exp(-uniforms.fogSigmaT * rayBoxExitDistance(dlShadowRay.origin, dlWi, kRoomBoundsMin, kRoomBoundsMax))
                        : 1.0;
                    radiance += throughput * dlLobeTint * (1.0 / M_PI_F)
                                * float3(dl.emission) * dlAbsCos * dlTransmittance;
                }
            }
        }

        for (uint pji = 0; pji < uniforms.projectionLightCount; ++pji) {
            ProjectionLight pj = projectionLights[pji];
            float3 toProjLight = float3(pj.position) - hitPoint;
            float pjDistSq = dot(toProjLight, toProjLight);
            float pjDist = sqrt(pjDistSq);
            float3 pjWi = toProjLight / pjDist;
            float pjCosSurface = dot(facingNormal, pjWi);
            if (pjCosSurface != 0.0) {
                float3 pjRadiance = projectionLightRadiance(-pjWi, pj.forward, pj.right, pj.up,
                                                             pj.tanHalfFovX, pj.tanHalfFovY, pj.scale,
                                                             (pj.usePbrtTexture != 0u ? pbrtProjectionTexture : earthTexture), textureSampler);
                if (any(pjRadiance > float3(0.0))) {
                    bool pjReflect = pjCosSurface > 0.0;
                    float3 pjLobeTint = pjReflect ? albedo : mat.transmitColor;
                    float pjAbsCos = abs(pjCosSurface);
                    ray pjShadowRay;
                    pjShadowRay.origin = hitPoint + (pjReflect ? facingNormal : -facingNormal) * 0.001f;
                    pjShadowRay.direction = pjWi;
                    pjShadowRay.min_distance = 0.001f;
                    pjShadowRay.max_distance = pjDist - 0.002f;
                    intersection_result<instancing, triangle_data> pjShadowResult =
                        isect.intersect(pjShadowRay, accelStructure, functionTable);
                    if (pjShadowResult.type == intersection_type::none) {
                        float pjTransmittance = exp(-uniforms.fogSigmaT * pjDist);
                        radiance += throughput * pjLobeTint * (1.0 / M_PI_F)
                                    * pjRadiance * pjAbsCos * pjTransmittance / pjDistSq;
                    }
                }
            }
        }

        for (uint gli = 0; gli < uniforms.goniometricLightCount; ++gli) {
            GoniometricLight gl = goniometricLights[gli];
            float3 toGoniLight = float3(gl.position) - hitPoint;
            float glDistSq = dot(toGoniLight, toGoniLight);
            float glDist = sqrt(glDistSq);
            float3 glWi = toGoniLight / glDist;
            float glCosSurface = dot(facingNormal, glWi);
            if (glCosSurface != 0.0) {
                float3 glRadiance = goniometricLightRadiance(-glWi, gl.forward, gl.right, gl.up,
                                                              gl.emission, gl.scale,
                                                              (gl.usePbrtTexture != 0u ? pbrtGoniometricTexture : goniometricTexture), textureSampler);
                if (any(glRadiance > float3(0.0))) {
                    bool glReflect = glCosSurface > 0.0;
                    float3 glLobeTint = glReflect ? albedo : mat.transmitColor;
                    float glAbsCos = abs(glCosSurface);
                    ray glShadowRay;
                    glShadowRay.origin = hitPoint + (glReflect ? facingNormal : -facingNormal) * 0.001f;
                    glShadowRay.direction = glWi;
                    glShadowRay.min_distance = 0.001f;
                    glShadowRay.max_distance = glDist - 0.002f;
                    intersection_result<instancing, triangle_data> glShadowResult =
                        isect.intersect(glShadowRay, accelStructure, functionTable);
                    if (glShadowResult.type == intersection_type::none) {
                        float glTransmittance = exp(-uniforms.fogSigmaT * glDist);
                        radiance += throughput * glLobeTint * (1.0 / M_PI_F)
                                    * glRadiance * glAbsCos * glTransmittance / glDistSq;
                    }
                }
            }
        }

        // Environment map (importance-sampled NEE, section 71) - see
        // shadeLambertian's own comment for the full "why." Same signed-
        // lobe-pick shape as every other light above in this function:
        // the sampled direction can land on EITHER side of `facingNormal`
        // (unlike a fixed-position light, whose side is decided once per
        // hit point), so which lobe/tint/pdf-weight/shadow-ray-offset
        // applies is decided by that sign, exactly as above.
        // Gated on useEnvironmentMap too, not just envMapWidth - see
        // shadeConductor's own comment.
        if (envMapWidth > 0u && uniforms.useEnvironmentMap != 0u) {
            float envPdfSolidAngle;
            float3 envWi = sampleEnvironmentDirection(envMarginalCDF, envConditionalCDF,
                                                       int(envMapWidth), int(envMapHeight),
                                                       randFloat(rngState), randFloat(rngState), envPdfSolidAngle);
            float envCosSurface = dot(facingNormal, envWi);
            if (envCosSurface != 0.0 && envPdfSolidAngle > 1e-9) {
                bool envReflect = envCosSurface > 0.0;
                float3 envLobeTint = envReflect ? albedo : mat.transmitColor;
                float envLobeProb = envReflect ? (pr / pSum) : (pt / pSum);
                float envAbsCos = abs(envCosSurface);
                ray envShadowRay;
                envShadowRay.origin = hitPoint + (envReflect ? facingNormal : -facingNormal) * 0.001f;
                envShadowRay.direction = envWi;
                envShadowRay.min_distance = 0.001f;
                envShadowRay.max_distance = 1e5f;
                intersection_result<instancing, triangle_data> envShadowResult =
                    isect.intersect(envShadowRay, accelStructure, functionTable);
                if (envShadowResult.type == intersection_type::none) {
                    float2 envUV = equirectangularUV(envWi);
                    float3 envRadiance = earthTexture.sample(textureSampler, envUV).rgb;
                    float envPdfBsdf = envLobeProb * envAbsCos / M_PI_F;
                    float envWeight = (envPdfSolidAngle * envPdfSolidAngle)
                        / (envPdfSolidAngle * envPdfSolidAngle + envPdfBsdf * envPdfBsdf);
                    radiance += throughput * envLobeTint * (1.0 / M_PI_F)
                                * envRadiance * envAbsCos / envPdfSolidAngle * envWeight;
                }
            }
        }

        // Same NEE/MIS strategy, for a pbrt-loaded scene's own SEPARATE
        // image-based infinite light (section 96) - see shadeConductor's
        // own comment. Same signed-lobe-pick shape as the block above.
        if (pbrtEnvMapWidth > 0u) {
            float pbrtEnvPdfSolidAngle;
            float3 pbrtEnvWi = sampleEnvironmentDirection(pbrtEnvMarginalCDF, pbrtEnvConditionalCDF,
                                                           int(pbrtEnvMapWidth), int(pbrtEnvMapHeight),
                                                           randFloat(rngState), randFloat(rngState), pbrtEnvPdfSolidAngle);
            float pbrtEnvCosSurface = dot(facingNormal, pbrtEnvWi);
            if (pbrtEnvCosSurface != 0.0 && pbrtEnvPdfSolidAngle > 1e-9) {
                bool pbrtEnvReflect = pbrtEnvCosSurface > 0.0;
                float3 pbrtEnvLobeTint = pbrtEnvReflect ? albedo : mat.transmitColor;
                float pbrtEnvLobeProb = pbrtEnvReflect ? (pr / pSum) : (pt / pSum);
                float pbrtEnvAbsCos = abs(pbrtEnvCosSurface);
                ray pbrtEnvShadowRay;
                pbrtEnvShadowRay.origin = hitPoint + (pbrtEnvReflect ? facingNormal : -facingNormal) * 0.001f;
                pbrtEnvShadowRay.direction = pbrtEnvWi;
                pbrtEnvShadowRay.min_distance = 0.001f;
                pbrtEnvShadowRay.max_distance = 1e5f;
                intersection_result<instancing, triangle_data> pbrtEnvShadowResult =
                    isect.intersect(pbrtEnvShadowRay, accelStructure, functionTable);
                if (pbrtEnvShadowResult.type == intersection_type::none) {
                    float2 pbrtEnvUV = equirectangularUV(pbrtEnvWi);
                    float3 pbrtEnvRadianceSample = pbrtEnvTexture.sample(textureSampler, pbrtEnvUV).rgb;
                    float pbrtEnvPdfBsdf = pbrtEnvLobeProb * pbrtEnvAbsCos / M_PI_F;
                    float pbrtEnvWeight = (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle)
                        / (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle + pbrtEnvPdfBsdf * pbrtEnvPdfBsdf);
                    radiance += throughput * pbrtEnvLobeTint * (1.0 / M_PI_F)
                                * pbrtEnvRadianceSample * pbrtEnvAbsCos / pbrtEnvPdfSolidAngle * pbrtEnvWeight;
                }
            }
        }
    }

    bool reflect = randFloat(rngState) < (pr / pSum);
    float3 lobeNormal = reflect ? facingNormal : -facingNormal;
    rayDir = cosineSampleHemisphere(lobeNormal, rngState);
    rayOrigin = hitPoint + lobeNormal * 0.001f;
    throughput *= reflect ? albedo : mat.transmitColor;
    bsdfPdf = (reflect ? (pr / pSum) : (pt / pSum)) * max(dot(lobeNormal, rayDir), 0.0001) / M_PI_F;
    specularBounce = false;
    return true;
}

inline bool shadeLambertian(TriangleMaterial mat, float3 albedo, float3 hitPoint, float3 facingNormal,
                             constant Uniforms& uniforms,
                             device const AreaLight* lights,
                             device const PointLight* pointLights,
                             device const DirectionalLight* directionalLights,
                             device const ProjectionLight* projectionLights,
                             device const GoniometricLight* goniometricLights,
                             device const float* envMarginalCDF,
                             device const float* envConditionalCDF,
                             uint envMapWidth, uint envMapHeight,
                             device const float* pbrtEnvMarginalCDF,
                             device const float* pbrtEnvConditionalCDF,
                             uint pbrtEnvMapWidth, uint pbrtEnvMapHeight,
                             texture2d<float, access::sample> earthTexture,
                             texture2d<float, access::sample> pbrtEnvTexture,
                             texture2d<float, access::sample> goniometricTexture,
                             texture2d<float, access::sample> pbrtGoniometricTexture,
                             texture2d<float, access::sample> pbrtProjectionTexture,
                             texture2d<float, access::sample> pbrtAreaLightTexture,
                             sampler textureSampler,
                             intersector<instancing, triangle_data> isect,
                             instance_acceleration_structure accelStructure,
                             intersection_function_table<instancing, triangle_data> functionTable,
                             thread float3& rayDir, thread float3& rayOrigin,
                             thread float3& throughput, thread float3& radiance,
                             thread float& bsdfPdf, thread bool& specularBounce, thread uint& rngState) {
    // Lambertian (materialType 0, or 3/6/7/10 - the only difference
    // already resolved upstream into `albedo`, this BSDF/NEE math has no
    // idea where albedo came from): next-event estimation against a
    // randomly picked light, then continue the path via cosine-weighted
    // hemisphere sampling for indirect light.
    if (all(mat.emission == float3(0.0))) {
        LightSample ls = sampleAreaLight(lights, uniforms.lightCount, rngState, pbrtAreaLightTexture, textureSampler);
        float3 toLight = ls.point - hitPoint;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        float3 wi = toLight / dist;
        float cosSurface = dot(facingNormal, wi);
        float cosLight = dot(ls.normal, -wi);
        if (cosSurface > 0.0 && (cosLight > 0.0 || (ls.twoSided != 0.0 && cosLight < 0.0))) {
            ray shadowRay;
            shadowRay.origin = hitPoint + facingNormal * 0.001f;
            shadowRay.direction = wi;
            shadowRay.min_distance = 0.001f;
            shadowRay.max_distance = dist - 0.002f;
            intersection_result<instancing, triangle_data> shadowResult =
                isect.intersect(shadowRay, accelStructure, functionTable);
            if (shadowResult.type == intersection_type::none) {
                float pdfSolidAngle = (distSq / (ls.area * abs(cosLight))) * ls.pmf;
                float pdfBsdfForThisDir = cosSurface / M_PI_F;
                float weight = (pdfSolidAngle * pdfSolidAngle)
                    / (pdfSolidAngle * pdfSolidAngle + pdfBsdfForThisDir * pdfBsdfForThisDir);
                float transmittance = exp(-uniforms.fogSigmaT * dist);
                radiance += throughput * albedo * (1.0 / M_PI_F)
                            * ls.emission * cosSurface * transmittance / pdfSolidAngle * weight;
            }
        }

        for (uint pli = 0; pli < uniforms.pointLightCount; ++pli) {
            PointLight pl = pointLights[pli];
            float3 toPointLight = float3(pl.position) - hitPoint;
            float plDistSq = dot(toPointLight, toPointLight);
            float plDist = sqrt(plDistSq);
            float3 plWi = toPointLight / plDist;
            float plCosSurface = dot(facingNormal, plWi);
            if (plCosSurface > 0.0) {
                ray plShadowRay;
                plShadowRay.origin = hitPoint + facingNormal * 0.001f;
                plShadowRay.direction = plWi;
                plShadowRay.min_distance = 0.001f;
                plShadowRay.max_distance = plDist - 0.002f;
                intersection_result<instancing, triangle_data> plShadowResult =
                    isect.intersect(plShadowRay, accelStructure, functionTable);
                if (plShadowResult.type == intersection_type::none) {
                    float plTransmittance = exp(-uniforms.fogSigmaT * plDist);
                    float plSpot = spotLightFalloff(-plWi, float3(pl.direction), pl.cosOuterAngle, pl.cosInnerAngle);
                    radiance += throughput * albedo * (1.0 / M_PI_F)
                                * float3(pl.emission) * plCosSurface * plSpot * plTransmittance / plDistSq;
                }
            }
        }

        for (uint dli = 0; dli < uniforms.directionalLightCount; ++dli) {
            DirectionalLight dl = directionalLights[dli];
            float3 dlWi = normalize(-float3(dl.direction));
            float dlCosSurface = dot(facingNormal, dlWi);
            if (dlCosSurface > 0.0) {
                ray dlShadowRay;
                dlShadowRay.origin = hitPoint + facingNormal * 0.001f;
                dlShadowRay.direction = dlWi;
                dlShadowRay.min_distance = 0.001f;
                dlShadowRay.max_distance = kDirectionalLightMaxDistance;
                intersection_result<instancing, triangle_data> dlShadowResult =
                    isect.intersect(dlShadowRay, accelStructure, functionTable);
                if (dlShadowResult.type == intersection_type::none) {
                    // Skip when there is no fog - rayBoxExitDistance()
                    // is only valid for an origin INSIDE the hardcoded
                    // room bounds (see that function.s own comment for
                    // why calling it unconditionally is unsafe).
                    float dlTransmittance = (uniforms.fogSigmaT > 0.0)
                        ? exp(-uniforms.fogSigmaT * rayBoxExitDistance(dlShadowRay.origin, dlWi, kRoomBoundsMin, kRoomBoundsMax))
                        : 1.0;
                    radiance += throughput * albedo * (1.0 / M_PI_F)
                                * float3(dl.emission) * dlCosSurface * dlTransmittance;
                }
            }
        }

        for (uint pji = 0; pji < uniforms.projectionLightCount; ++pji) {
            ProjectionLight pj = projectionLights[pji];
            float3 toProjLight = float3(pj.position) - hitPoint;
            float pjDistSq = dot(toProjLight, toProjLight);
            float pjDist = sqrt(pjDistSq);
            float3 pjWi = toProjLight / pjDist;
            float pjCosSurface = dot(facingNormal, pjWi);
            if (pjCosSurface > 0.0) {
                float3 pjRadiance = projectionLightRadiance(-pjWi, pj.forward, pj.right, pj.up,
                                                             pj.tanHalfFovX, pj.tanHalfFovY, pj.scale,
                                                             (pj.usePbrtTexture != 0u ? pbrtProjectionTexture : earthTexture), textureSampler);
                if (any(pjRadiance > float3(0.0))) {
                    ray pjShadowRay;
                    pjShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    pjShadowRay.direction = pjWi;
                    pjShadowRay.min_distance = 0.001f;
                    pjShadowRay.max_distance = pjDist - 0.002f;
                    intersection_result<instancing, triangle_data> pjShadowResult =
                        isect.intersect(pjShadowRay, accelStructure, functionTable);
                    if (pjShadowResult.type == intersection_type::none) {
                        float pjTransmittance = exp(-uniforms.fogSigmaT * pjDist);
                        radiance += throughput * albedo * (1.0 / M_PI_F)
                                    * pjRadiance * pjCosSurface * pjTransmittance / pjDistSq;
                    }
                }
            }
        }

        for (uint gli = 0; gli < uniforms.goniometricLightCount; ++gli) {
            GoniometricLight gl = goniometricLights[gli];
            float3 toGoniLight = float3(gl.position) - hitPoint;
            float glDistSq = dot(toGoniLight, toGoniLight);
            float glDist = sqrt(glDistSq);
            float3 glWi = toGoniLight / glDist;
            float glCosSurface = dot(facingNormal, glWi);
            if (glCosSurface > 0.0) {
                float3 glRadiance = goniometricLightRadiance(-glWi, gl.forward, gl.right, gl.up,
                                                              gl.emission, gl.scale,
                                                              (gl.usePbrtTexture != 0u ? pbrtGoniometricTexture : goniometricTexture), textureSampler);
                if (any(glRadiance > float3(0.0))) {
                    ray glShadowRay;
                    glShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    glShadowRay.direction = glWi;
                    glShadowRay.min_distance = 0.001f;
                    glShadowRay.max_distance = glDist - 0.002f;
                    intersection_result<instancing, triangle_data> glShadowResult =
                        isect.intersect(glShadowRay, accelStructure, functionTable);
                    if (glShadowResult.type == intersection_type::none) {
                        float glTransmittance = exp(-uniforms.fogSigmaT * glDist);
                        radiance += throughput * albedo * (1.0 / M_PI_F)
                                    * glRadiance * glCosSurface * glTransmittance / glDistSq;
                    }
                }
            }
        }

        // Environment map (importance-sampled NEE, section 71): a NEW
        // light-sampling strategy for `earthTexture`'s own equirectangular
        // sample - previously reachable only via a BSDF-sampled ray that
        // happened to escape toward a bright region (high variance under
        // a small/bright environment feature, exactly the problem NEE/MIS
        // already solves for every light type above). Samples a
        // direction from the environment image's own importance
        // distribution (phase 1, section 69 - bright regions picked more
        // often), checks visibility with a shadow ray toward "infinity"
        // (`1e5f`, matching this scene's own room-scale units), and MIS-
        // weights against this material's own cosine-weighted BSDF pdf
        // for that same direction - the same NEE/MIS shape the area
        // light above already uses, just with a direction-dependent (not
        // point) light source and an importance-sampled (not uniform)
        // sampling strategy.
        // Gated on useEnvironmentMap too, not just envMapWidth - see
        // shadeConductor's own comment.
        if (envMapWidth > 0u && uniforms.useEnvironmentMap != 0u) {
            float envPdfSolidAngle;
            float3 envWi = sampleEnvironmentDirection(envMarginalCDF, envConditionalCDF,
                                                       int(envMapWidth), int(envMapHeight),
                                                       randFloat(rngState), randFloat(rngState), envPdfSolidAngle);
            float envCosSurface = dot(facingNormal, envWi);
            if (envCosSurface > 0.0 && envPdfSolidAngle > 1e-9) {
                ray envShadowRay;
                envShadowRay.origin = hitPoint + facingNormal * 0.001f;
                envShadowRay.direction = envWi;
                envShadowRay.min_distance = 0.001f;
                envShadowRay.max_distance = 1e5f;
                intersection_result<instancing, triangle_data> envShadowResult =
                    isect.intersect(envShadowRay, accelStructure, functionTable);
                if (envShadowResult.type == intersection_type::none) {
                    float2 envUV = equirectangularUV(envWi);
                    float3 envRadiance = earthTexture.sample(textureSampler, envUV).rgb;
                    float envPdfBsdf = envCosSurface / M_PI_F;
                    float envWeight = (envPdfSolidAngle * envPdfSolidAngle)
                        / (envPdfSolidAngle * envPdfSolidAngle + envPdfBsdf * envPdfBsdf);
                    radiance += throughput * albedo * (1.0 / M_PI_F)
                                * envRadiance * envCosSurface / envPdfSolidAngle * envWeight;
                }
            }
        }

        // Same NEE/MIS strategy, for a pbrt-loaded scene's own SEPARATE
        // image-based infinite light (section 96) - see shadeConductor's
        // own comment.
        if (pbrtEnvMapWidth > 0u) {
            float pbrtEnvPdfSolidAngle;
            float3 pbrtEnvWi = sampleEnvironmentDirection(pbrtEnvMarginalCDF, pbrtEnvConditionalCDF,
                                                           int(pbrtEnvMapWidth), int(pbrtEnvMapHeight),
                                                           randFloat(rngState), randFloat(rngState), pbrtEnvPdfSolidAngle);
            float pbrtEnvCosSurface = dot(facingNormal, pbrtEnvWi);
            if (pbrtEnvCosSurface > 0.0 && pbrtEnvPdfSolidAngle > 1e-9) {
                ray pbrtEnvShadowRay;
                pbrtEnvShadowRay.origin = hitPoint + facingNormal * 0.001f;
                pbrtEnvShadowRay.direction = pbrtEnvWi;
                pbrtEnvShadowRay.min_distance = 0.001f;
                pbrtEnvShadowRay.max_distance = 1e5f;
                intersection_result<instancing, triangle_data> pbrtEnvShadowResult =
                    isect.intersect(pbrtEnvShadowRay, accelStructure, functionTable);
                if (pbrtEnvShadowResult.type == intersection_type::none) {
                    float2 pbrtEnvUV = equirectangularUV(pbrtEnvWi);
                    float3 pbrtEnvRadianceSample = pbrtEnvTexture.sample(textureSampler, pbrtEnvUV).rgb;
                    float pbrtEnvPdfBsdf = pbrtEnvCosSurface / M_PI_F;
                    float pbrtEnvWeight = (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle)
                        / (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle + pbrtEnvPdfBsdf * pbrtEnvPdfBsdf);
                    radiance += throughput * albedo * (1.0 / M_PI_F)
                                * pbrtEnvRadianceSample * pbrtEnvCosSurface / pbrtEnvPdfSolidAngle * pbrtEnvWeight;
                }
            }
        }
    }

    rayDir = cosineSampleHemisphere(facingNormal, rngState);
    rayOrigin = hitPoint + facingNormal * 0.001f;
    throughput *= albedo;
    bsdfPdf = max(dot(facingNormal, rayDir), 0.0001) / M_PI_F;
    specularBounce = false;
    return true;
}

// "Improved Oren-Nayar" rough diffuse (Fujii's reformulation of Oren &
// Nayar 1994, https://mimosa-pudica.net/improved-oren-nayar.html) -
// found via Blender Cycles as reference (kernel/closure/
// bsdf_oren_nayar.h), where it's the modern replacement for the
// classic model. A rough (not perfectly Lambertian) diffuse surface -
// clay, plaster, the Moon's own regolith - reads visibly FLATTER/less
// shaded than ideal Lambertian: light and view directions both near
// grazing (and roughly aligned in azimuth) brighten noticeably (the
// classic "flat full moon" retroreflective look, real microfacet
// self-shadowing/masking within the rough surface concentrating
// reflected light back toward the source), while head-on illumination
// reads slightly DARKER than Lambertian - a real energy redistribution,
// not just a brightness knob. `mat.roughness` doubles as this
// material's own sigma parameter (materialType 5/4/7/9/10 already
// reuse this same field their own way, per TriangleMaterial's own
// comment - one more reuse, not a new struct field).
//
// Deliberately scoped to the SINGLE-scatter term only - Cycles' own
// version adds a further energy-preserving MULTI-scatter compensation
// term (OpenPBR-spec-based) on top of this, explicitly deferred here
// (the same "close the bigger, more visible gap first" staging this
// POC's own GGX energy-compensation work, sections 72/73, already
// used) - `sigma == 0` still reduces EXACTLY to plain Lambertian
// (`a == 1/pi`, `b == 0`), verified below, so this is a strict
// generalization, not a replacement with different edge-case behaviour.
inline float orenNayarF(float3 wo, float3 wi, float3 n, float sigma) {
    float nl = max(dot(n, wi), 0.0);
    float nv = max(dot(n, wo), 0.0);
    float a = 1.0 / (M_PI_F + sigma * (M_PI_F * 0.5 - 2.0 / 3.0));
    float b = sigma * a;
    if (b <= 0.0) {
        return a;
    }
    float t = dot(wi, wo) - nl * nv;
    if (t > 0.0) {
        t /= max(max(nl, nv), 1e-6);
    }
    return a + b * t;
}

inline bool shadeOrenNayar(TriangleMaterial mat, float3 albedo, float3 hitPoint, float3 facingNormal,
                            constant Uniforms& uniforms,
                            device const AreaLight* lights,
                            device const PointLight* pointLights,
                            device const DirectionalLight* directionalLights,
                            device const ProjectionLight* projectionLights,
                            device const GoniometricLight* goniometricLights,
                            device const float* envMarginalCDF,
                            device const float* envConditionalCDF,
                            uint envMapWidth, uint envMapHeight,
                            device const float* pbrtEnvMarginalCDF,
                            device const float* pbrtEnvConditionalCDF,
                            uint pbrtEnvMapWidth, uint pbrtEnvMapHeight,
                            texture2d<float, access::sample> earthTexture,
                            texture2d<float, access::sample> pbrtEnvTexture,
                            texture2d<float, access::sample> goniometricTexture,
                            texture2d<float, access::sample> pbrtGoniometricTexture,
                            texture2d<float, access::sample> pbrtProjectionTexture,
                            texture2d<float, access::sample> pbrtAreaLightTexture,
                            sampler textureSampler,
                            intersector<instancing, triangle_data> isect,
                            instance_acceleration_structure accelStructure,
                            intersection_function_table<instancing, triangle_data> functionTable,
                            thread float3& rayDir, thread float3& rayOrigin,
                            thread float3& throughput, thread float3& radiance,
                            thread float& bsdfPdf, thread bool& specularBounce, thread uint& rngState) {
    float3 woWorld = -rayDir;

    if (all(mat.emission == float3(0.0))) {
        LightSample ls = sampleAreaLight(lights, uniforms.lightCount, rngState, pbrtAreaLightTexture, textureSampler);
        float3 toLight = ls.point - hitPoint;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        float3 wi = toLight / dist;
        float cosSurface = dot(facingNormal, wi);
        float cosLight = dot(ls.normal, -wi);
        if (cosSurface > 0.0 && (cosLight > 0.0 || (ls.twoSided != 0.0 && cosLight < 0.0))) {
            ray shadowRay;
            shadowRay.origin = hitPoint + facingNormal * 0.001f;
            shadowRay.direction = wi;
            shadowRay.min_distance = 0.001f;
            shadowRay.max_distance = dist - 0.002f;
            intersection_result<instancing, triangle_data> shadowResult =
                isect.intersect(shadowRay, accelStructure, functionTable);
            if (shadowResult.type == intersection_type::none) {
                float pdfSolidAngle = (distSq / (ls.area * abs(cosLight))) * ls.pmf;
                float pdfBsdfForThisDir = cosSurface / M_PI_F;
                float weight = (pdfSolidAngle * pdfSolidAngle)
                    / (pdfSolidAngle * pdfSolidAngle + pdfBsdfForThisDir * pdfBsdfForThisDir);
                float transmittance = exp(-uniforms.fogSigmaT * dist);
                radiance += throughput * albedo * orenNayarF(woWorld, wi, facingNormal, mat.roughness)
                            * ls.emission * cosSurface * transmittance / pdfSolidAngle * weight;
            }
        }

        for (uint pli = 0; pli < uniforms.pointLightCount; ++pli) {
            PointLight pl = pointLights[pli];
            float3 toPointLight = float3(pl.position) - hitPoint;
            float plDistSq = dot(toPointLight, toPointLight);
            float plDist = sqrt(plDistSq);
            float3 plWi = toPointLight / plDist;
            float plCosSurface = dot(facingNormal, plWi);
            if (plCosSurface > 0.0) {
                ray plShadowRay;
                plShadowRay.origin = hitPoint + facingNormal * 0.001f;
                plShadowRay.direction = plWi;
                plShadowRay.min_distance = 0.001f;
                plShadowRay.max_distance = plDist - 0.002f;
                intersection_result<instancing, triangle_data> plShadowResult =
                    isect.intersect(plShadowRay, accelStructure, functionTable);
                if (plShadowResult.type == intersection_type::none) {
                    float plTransmittance = exp(-uniforms.fogSigmaT * plDist);
                    float plSpot = spotLightFalloff(-plWi, float3(pl.direction), pl.cosOuterAngle, pl.cosInnerAngle);
                    radiance += throughput * albedo * orenNayarF(woWorld, plWi, facingNormal, mat.roughness)
                                * float3(pl.emission) * plCosSurface * plSpot * plTransmittance / plDistSq;
                }
            }
        }

        for (uint dli = 0; dli < uniforms.directionalLightCount; ++dli) {
            DirectionalLight dl = directionalLights[dli];
            float3 dlWi = normalize(-float3(dl.direction));
            float dlCosSurface = dot(facingNormal, dlWi);
            if (dlCosSurface > 0.0) {
                ray dlShadowRay;
                dlShadowRay.origin = hitPoint + facingNormal * 0.001f;
                dlShadowRay.direction = dlWi;
                dlShadowRay.min_distance = 0.001f;
                dlShadowRay.max_distance = kDirectionalLightMaxDistance;
                intersection_result<instancing, triangle_data> dlShadowResult =
                    isect.intersect(dlShadowRay, accelStructure, functionTable);
                if (dlShadowResult.type == intersection_type::none) {
                    // Skip when there is no fog - rayBoxExitDistance()
                    // is only valid for an origin INSIDE the hardcoded
                    // room bounds (see that function.s own comment for
                    // why calling it unconditionally is unsafe).
                    float dlTransmittance = (uniforms.fogSigmaT > 0.0)
                        ? exp(-uniforms.fogSigmaT * rayBoxExitDistance(dlShadowRay.origin, dlWi, kRoomBoundsMin, kRoomBoundsMax))
                        : 1.0;
                    radiance += throughput * albedo * orenNayarF(woWorld, dlWi, facingNormal, mat.roughness)
                                * float3(dl.emission) * dlCosSurface * dlTransmittance;
                }
            }
        }

        for (uint pji = 0; pji < uniforms.projectionLightCount; ++pji) {
            ProjectionLight pj = projectionLights[pji];
            float3 toProjLight = float3(pj.position) - hitPoint;
            float pjDistSq = dot(toProjLight, toProjLight);
            float pjDist = sqrt(pjDistSq);
            float3 pjWi = toProjLight / pjDist;
            float pjCosSurface = dot(facingNormal, pjWi);
            if (pjCosSurface > 0.0) {
                float3 pjRadiance = projectionLightRadiance(-pjWi, pj.forward, pj.right, pj.up,
                                                             pj.tanHalfFovX, pj.tanHalfFovY, pj.scale,
                                                             (pj.usePbrtTexture != 0u ? pbrtProjectionTexture : earthTexture), textureSampler);
                if (any(pjRadiance > float3(0.0))) {
                    ray pjShadowRay;
                    pjShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    pjShadowRay.direction = pjWi;
                    pjShadowRay.min_distance = 0.001f;
                    pjShadowRay.max_distance = pjDist - 0.002f;
                    intersection_result<instancing, triangle_data> pjShadowResult =
                        isect.intersect(pjShadowRay, accelStructure, functionTable);
                    if (pjShadowResult.type == intersection_type::none) {
                        float pjTransmittance = exp(-uniforms.fogSigmaT * pjDist);
                        radiance += throughput * albedo * orenNayarF(woWorld, pjWi, facingNormal, mat.roughness)
                                    * pjRadiance * pjCosSurface * pjTransmittance / pjDistSq;
                    }
                }
            }
        }

        for (uint gli = 0; gli < uniforms.goniometricLightCount; ++gli) {
            GoniometricLight gl = goniometricLights[gli];
            float3 toGoniLight = float3(gl.position) - hitPoint;
            float glDistSq = dot(toGoniLight, toGoniLight);
            float glDist = sqrt(glDistSq);
            float3 glWi = toGoniLight / glDist;
            float glCosSurface = dot(facingNormal, glWi);
            if (glCosSurface > 0.0) {
                float3 glRadiance = goniometricLightRadiance(-glWi, gl.forward, gl.right, gl.up,
                                                              gl.emission, gl.scale,
                                                              (gl.usePbrtTexture != 0u ? pbrtGoniometricTexture : goniometricTexture), textureSampler);
                if (any(glRadiance > float3(0.0))) {
                    ray glShadowRay;
                    glShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    glShadowRay.direction = glWi;
                    glShadowRay.min_distance = 0.001f;
                    glShadowRay.max_distance = glDist - 0.002f;
                    intersection_result<instancing, triangle_data> glShadowResult =
                        isect.intersect(glShadowRay, accelStructure, functionTable);
                    if (glShadowResult.type == intersection_type::none) {
                        float glTransmittance = exp(-uniforms.fogSigmaT * glDist);
                        radiance += throughput * albedo * orenNayarF(woWorld, glWi, facingNormal, mat.roughness)
                                    * glRadiance * glCosSurface * glTransmittance / glDistSq;
                    }
                }
            }
        }

        // Gated on useEnvironmentMap too, not just envMapWidth - see
        // shadeConductor's own comment.
        if (envMapWidth > 0u && uniforms.useEnvironmentMap != 0u) {
            float envPdfSolidAngle;
            float3 envWi = sampleEnvironmentDirection(envMarginalCDF, envConditionalCDF,
                                                       int(envMapWidth), int(envMapHeight),
                                                       randFloat(rngState), randFloat(rngState), envPdfSolidAngle);
            float envCosSurface = dot(facingNormal, envWi);
            if (envCosSurface > 0.0 && envPdfSolidAngle > 1e-9) {
                ray envShadowRay;
                envShadowRay.origin = hitPoint + facingNormal * 0.001f;
                envShadowRay.direction = envWi;
                envShadowRay.min_distance = 0.001f;
                envShadowRay.max_distance = 1e5f;
                intersection_result<instancing, triangle_data> envShadowResult =
                    isect.intersect(envShadowRay, accelStructure, functionTable);
                if (envShadowResult.type == intersection_type::none) {
                    float2 envUV = equirectangularUV(envWi);
                    float3 envRadiance = earthTexture.sample(textureSampler, envUV).rgb;
                    float envPdfBsdf = envCosSurface / M_PI_F;
                    float envWeight = (envPdfSolidAngle * envPdfSolidAngle)
                        / (envPdfSolidAngle * envPdfSolidAngle + envPdfBsdf * envPdfBsdf);
                    radiance += throughput * albedo * orenNayarF(woWorld, envWi, facingNormal, mat.roughness)
                                * envRadiance * envCosSurface / envPdfSolidAngle * envWeight;
                }
            }
        }

        // Same NEE/MIS strategy, for a pbrt-loaded scene's own SEPARATE
        // image-based infinite light (section 96) - see shadeConductor's
        // own comment.
        if (pbrtEnvMapWidth > 0u) {
            float pbrtEnvPdfSolidAngle;
            float3 pbrtEnvWi = sampleEnvironmentDirection(pbrtEnvMarginalCDF, pbrtEnvConditionalCDF,
                                                           int(pbrtEnvMapWidth), int(pbrtEnvMapHeight),
                                                           randFloat(rngState), randFloat(rngState), pbrtEnvPdfSolidAngle);
            float pbrtEnvCosSurface = dot(facingNormal, pbrtEnvWi);
            if (pbrtEnvCosSurface > 0.0 && pbrtEnvPdfSolidAngle > 1e-9) {
                ray pbrtEnvShadowRay;
                pbrtEnvShadowRay.origin = hitPoint + facingNormal * 0.001f;
                pbrtEnvShadowRay.direction = pbrtEnvWi;
                pbrtEnvShadowRay.min_distance = 0.001f;
                pbrtEnvShadowRay.max_distance = 1e5f;
                intersection_result<instancing, triangle_data> pbrtEnvShadowResult =
                    isect.intersect(pbrtEnvShadowRay, accelStructure, functionTable);
                if (pbrtEnvShadowResult.type == intersection_type::none) {
                    float2 pbrtEnvUV = equirectangularUV(pbrtEnvWi);
                    float3 pbrtEnvRadianceSample = pbrtEnvTexture.sample(textureSampler, pbrtEnvUV).rgb;
                    float pbrtEnvPdfBsdf = pbrtEnvCosSurface / M_PI_F;
                    float pbrtEnvWeight = (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle)
                        / (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle + pbrtEnvPdfBsdf * pbrtEnvPdfBsdf);
                    radiance += throughput * albedo * orenNayarF(woWorld, pbrtEnvWi, facingNormal, mat.roughness)
                                * pbrtEnvRadianceSample * pbrtEnvCosSurface / pbrtEnvPdfSolidAngle * pbrtEnvWeight;
                }
            }
        }
    }

    // Continuation ray: still plain cosine-weighted hemisphere sampling
    // (NOT importance-sampled to this BRDF's own a+b*t shape - the same
    // simplification Cycles' own bsdf_oren_nayar_sample() makes too),
    // so the pdf stays cosTheta/pi exactly like Lambertian. The MC
    // weight f*cosTheta/pdf therefore collapses to
    // `albedo*orenNayarF(...)*pi` - Lambertian's own `throughput *=
    // albedo` is the special case of this at sigma == 0, where
    // orenNayarF() returns the constant `1/pi` and the two `pi`s cancel
    // back to exactly `albedo`.
    float3 newDir = cosineSampleHemisphere(facingNormal, rngState);
    rayDir = newDir;
    rayOrigin = hitPoint + facingNormal * 0.001f;
    throughput *= albedo * orenNayarF(woWorld, newDir, facingNormal, mat.roughness) * M_PI_F;
    bsdfPdf = max(dot(facingNormal, newDir), 0.0001) / M_PI_F;
    specularBounce = false;
    return true;
}

// pbrt-v4's own NormalizedFresnelBxDF (src/shared/bxdfs_layered.h) -
// Fresnel-WEIGHTED diffuse reflection: `f(wi) = (1-FrDielectric(cos_wi,
// eta)) / (c*pi)`, `c = 1 - 2*FresnelMoment1(1/eta)` an energy-
// renormalization constant accounting for light trapped and re-emitted
// by internal reflection inside a dielectric-coated diffuse layer (the
// real physical basis: a "crystal" sphere - light exits MORE at
// grazing angles, since Fresnel reflectance is LOWEST there, the
// opposite intuition from a bare specular Fresnel surface). Genuinely
// ACHROMATIC (no albedo tint at all, unlike every OTHER diffuse-family
// material here) - `eta` (materialType 2/4/5/9/11 already each reuse
// this field their own way) and the precomputed `c` constant
// (`mat.roughness`, matching Oren-Nayar's own reuse of the same field
// for an unrelated per-material scalar) are its ONLY two parameters.
// `c` is computed HOST-SIDE (FresnelMoment1's own polynomial fit is a
// fixed function of a compile-time-known `eta`, needing no per-hit
// device evaluation at all - simpler than porting FresnelMoment1()
// itself to MSL, and exactly equivalent since eta never varies per-hit
// for this material).
inline float normalizedFresnelF(float3 wi, float3 n, float eta, float c) {
    float cosWi = max(dot(n, wi), 0.0);
    if (cosWi <= 0.0) return 0.0;
    float fr = frDielectric(cosWi, eta);
    float cv = max(c, 1e-6);
    return (1.0 - fr) / (cv * M_PI_F);
}

// Structurally identical to shadeOrenNayar() just above (same cosine-
// weighted NEE+continuation shape - see that function's own comment on
// why: `albedo*orenNayarF(...)` there becomes plain
// `float3(normalizedFresnelF(...))` here, since this material has no
// separate albedo tint at all, the BRDF value IS the whole weight).
inline bool shadeNormalizedFresnel(TriangleMaterial mat, float3 hitPoint, float3 facingNormal,
                            constant Uniforms& uniforms,
                            device const AreaLight* lights,
                            device const PointLight* pointLights,
                            device const DirectionalLight* directionalLights,
                            device const ProjectionLight* projectionLights,
                            device const GoniometricLight* goniometricLights,
                            device const float* envMarginalCDF,
                            device const float* envConditionalCDF,
                            uint envMapWidth, uint envMapHeight,
                            device const float* pbrtEnvMarginalCDF,
                            device const float* pbrtEnvConditionalCDF,
                            uint pbrtEnvMapWidth, uint pbrtEnvMapHeight,
                            texture2d<float, access::sample> earthTexture,
                            texture2d<float, access::sample> pbrtEnvTexture,
                            texture2d<float, access::sample> goniometricTexture,
                            texture2d<float, access::sample> pbrtGoniometricTexture,
                            texture2d<float, access::sample> pbrtProjectionTexture,
                            texture2d<float, access::sample> pbrtAreaLightTexture,
                            sampler textureSampler,
                            intersector<instancing, triangle_data> isect,
                            instance_acceleration_structure accelStructure,
                            intersection_function_table<instancing, triangle_data> functionTable,
                            thread float3& rayDir, thread float3& rayOrigin,
                            thread float3& throughput, thread float3& radiance,
                            thread float& bsdfPdf, thread bool& specularBounce, thread uint& rngState) {
    if (all(mat.emission == float3(0.0))) {
        LightSample ls = sampleAreaLight(lights, uniforms.lightCount, rngState, pbrtAreaLightTexture, textureSampler);
        float3 toLight = ls.point - hitPoint;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        float3 wi = toLight / dist;
        float cosSurface = dot(facingNormal, wi);
        float cosLight = dot(ls.normal, -wi);
        if (cosSurface > 0.0 && (cosLight > 0.0 || (ls.twoSided != 0.0 && cosLight < 0.0))) {
            ray shadowRay;
            shadowRay.origin = hitPoint + facingNormal * 0.001f;
            shadowRay.direction = wi;
            shadowRay.min_distance = 0.001f;
            shadowRay.max_distance = dist - 0.002f;
            intersection_result<instancing, triangle_data> shadowResult =
                isect.intersect(shadowRay, accelStructure, functionTable);
            if (shadowResult.type == intersection_type::none) {
                float pdfSolidAngle = (distSq / (ls.area * abs(cosLight))) * ls.pmf;
                float pdfBsdfForThisDir = cosSurface / M_PI_F;
                float weight = (pdfSolidAngle * pdfSolidAngle)
                    / (pdfSolidAngle * pdfSolidAngle + pdfBsdfForThisDir * pdfBsdfForThisDir);
                float transmittance = exp(-uniforms.fogSigmaT * dist);
                radiance += throughput * float3(normalizedFresnelF(wi, facingNormal, mat.ior, mat.roughness))
                            * ls.emission * cosSurface * transmittance / pdfSolidAngle * weight;
            }
        }

        for (uint pli = 0; pli < uniforms.pointLightCount; ++pli) {
            PointLight pl = pointLights[pli];
            float3 toPointLight = float3(pl.position) - hitPoint;
            float plDistSq = dot(toPointLight, toPointLight);
            float plDist = sqrt(plDistSq);
            float3 plWi = toPointLight / plDist;
            float plCosSurface = dot(facingNormal, plWi);
            if (plCosSurface > 0.0) {
                ray plShadowRay;
                plShadowRay.origin = hitPoint + facingNormal * 0.001f;
                plShadowRay.direction = plWi;
                plShadowRay.min_distance = 0.001f;
                plShadowRay.max_distance = plDist - 0.002f;
                intersection_result<instancing, triangle_data> plShadowResult =
                    isect.intersect(plShadowRay, accelStructure, functionTable);
                if (plShadowResult.type == intersection_type::none) {
                    float plTransmittance = exp(-uniforms.fogSigmaT * plDist);
                    float plSpot = spotLightFalloff(-plWi, float3(pl.direction), pl.cosOuterAngle, pl.cosInnerAngle);
                    radiance += throughput * float3(normalizedFresnelF(plWi, facingNormal, mat.ior, mat.roughness))
                                * float3(pl.emission) * plCosSurface * plSpot * plTransmittance / plDistSq;
                }
            }
        }

        for (uint dli = 0; dli < uniforms.directionalLightCount; ++dli) {
            DirectionalLight dl = directionalLights[dli];
            float3 dlWi = normalize(-float3(dl.direction));
            float dlCosSurface = dot(facingNormal, dlWi);
            if (dlCosSurface > 0.0) {
                ray dlShadowRay;
                dlShadowRay.origin = hitPoint + facingNormal * 0.001f;
                dlShadowRay.direction = dlWi;
                dlShadowRay.min_distance = 0.001f;
                dlShadowRay.max_distance = kDirectionalLightMaxDistance;
                intersection_result<instancing, triangle_data> dlShadowResult =
                    isect.intersect(dlShadowRay, accelStructure, functionTable);
                if (dlShadowResult.type == intersection_type::none) {
                    // Skip when there is no fog - rayBoxExitDistance()
                    // is only valid for an origin INSIDE the hardcoded
                    // room bounds (see that function.s own comment for
                    // why calling it unconditionally is unsafe).
                    float dlTransmittance = (uniforms.fogSigmaT > 0.0)
                        ? exp(-uniforms.fogSigmaT * rayBoxExitDistance(dlShadowRay.origin, dlWi, kRoomBoundsMin, kRoomBoundsMax))
                        : 1.0;
                    radiance += throughput * float3(normalizedFresnelF(dlWi, facingNormal, mat.ior, mat.roughness))
                                * float3(dl.emission) * dlCosSurface * dlTransmittance;
                }
            }
        }

        for (uint pji = 0; pji < uniforms.projectionLightCount; ++pji) {
            ProjectionLight pj = projectionLights[pji];
            float3 toProjLight = float3(pj.position) - hitPoint;
            float pjDistSq = dot(toProjLight, toProjLight);
            float pjDist = sqrt(pjDistSq);
            float3 pjWi = toProjLight / pjDist;
            float pjCosSurface = dot(facingNormal, pjWi);
            if (pjCosSurface > 0.0) {
                float3 pjRadiance = projectionLightRadiance(-pjWi, pj.forward, pj.right, pj.up,
                                                             pj.tanHalfFovX, pj.tanHalfFovY, pj.scale,
                                                             (pj.usePbrtTexture != 0u ? pbrtProjectionTexture : earthTexture), textureSampler);
                if (any(pjRadiance > float3(0.0))) {
                    ray pjShadowRay;
                    pjShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    pjShadowRay.direction = pjWi;
                    pjShadowRay.min_distance = 0.001f;
                    pjShadowRay.max_distance = pjDist - 0.002f;
                    intersection_result<instancing, triangle_data> pjShadowResult =
                        isect.intersect(pjShadowRay, accelStructure, functionTable);
                    if (pjShadowResult.type == intersection_type::none) {
                        float pjTransmittance = exp(-uniforms.fogSigmaT * pjDist);
                        radiance += throughput * float3(normalizedFresnelF(pjWi, facingNormal, mat.ior, mat.roughness))
                                    * pjRadiance * pjCosSurface * pjTransmittance / pjDistSq;
                    }
                }
            }
        }

        for (uint gli = 0; gli < uniforms.goniometricLightCount; ++gli) {
            GoniometricLight gl = goniometricLights[gli];
            float3 toGoniLight = float3(gl.position) - hitPoint;
            float glDistSq = dot(toGoniLight, toGoniLight);
            float glDist = sqrt(glDistSq);
            float3 glWi = toGoniLight / glDist;
            float glCosSurface = dot(facingNormal, glWi);
            if (glCosSurface > 0.0) {
                float3 glRadiance = goniometricLightRadiance(-glWi, gl.forward, gl.right, gl.up,
                                                              gl.emission, gl.scale,
                                                              (gl.usePbrtTexture != 0u ? pbrtGoniometricTexture : goniometricTexture), textureSampler);
                if (any(glRadiance > float3(0.0))) {
                    ray glShadowRay;
                    glShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    glShadowRay.direction = glWi;
                    glShadowRay.min_distance = 0.001f;
                    glShadowRay.max_distance = glDist - 0.002f;
                    intersection_result<instancing, triangle_data> glShadowResult =
                        isect.intersect(glShadowRay, accelStructure, functionTable);
                    if (glShadowResult.type == intersection_type::none) {
                        float glTransmittance = exp(-uniforms.fogSigmaT * glDist);
                        radiance += throughput * float3(normalizedFresnelF(glWi, facingNormal, mat.ior, mat.roughness))
                                    * glRadiance * glCosSurface * glTransmittance / glDistSq;
                    }
                }
            }
        }

        if (envMapWidth > 0u && uniforms.useEnvironmentMap != 0u) {
            float envPdfSolidAngle;
            float3 envWi = sampleEnvironmentDirection(envMarginalCDF, envConditionalCDF,
                                                       int(envMapWidth), int(envMapHeight),
                                                       randFloat(rngState), randFloat(rngState), envPdfSolidAngle);
            float envCosSurface = dot(facingNormal, envWi);
            if (envCosSurface > 0.0 && envPdfSolidAngle > 1e-9) {
                ray envShadowRay;
                envShadowRay.origin = hitPoint + facingNormal * 0.001f;
                envShadowRay.direction = envWi;
                envShadowRay.min_distance = 0.001f;
                envShadowRay.max_distance = 1e5f;
                intersection_result<instancing, triangle_data> envShadowResult =
                    isect.intersect(envShadowRay, accelStructure, functionTable);
                if (envShadowResult.type == intersection_type::none) {
                    float2 envUV = equirectangularUV(envWi);
                    float3 envRadiance = earthTexture.sample(textureSampler, envUV).rgb;
                    float envPdfBsdf = envCosSurface / M_PI_F;
                    float envWeight = (envPdfSolidAngle * envPdfSolidAngle)
                        / (envPdfSolidAngle * envPdfSolidAngle + envPdfBsdf * envPdfBsdf);
                    radiance += throughput * float3(normalizedFresnelF(envWi, facingNormal, mat.ior, mat.roughness))
                                * envRadiance * envCosSurface / envPdfSolidAngle * envWeight;
                }
            }
        }

        if (pbrtEnvMapWidth > 0u) {
            float pbrtEnvPdfSolidAngle;
            float3 pbrtEnvWi = sampleEnvironmentDirection(pbrtEnvMarginalCDF, pbrtEnvConditionalCDF,
                                                           int(pbrtEnvMapWidth), int(pbrtEnvMapHeight),
                                                           randFloat(rngState), randFloat(rngState), pbrtEnvPdfSolidAngle);
            float pbrtEnvCosSurface = dot(facingNormal, pbrtEnvWi);
            if (pbrtEnvCosSurface > 0.0 && pbrtEnvPdfSolidAngle > 1e-9) {
                ray pbrtEnvShadowRay;
                pbrtEnvShadowRay.origin = hitPoint + facingNormal * 0.001f;
                pbrtEnvShadowRay.direction = pbrtEnvWi;
                pbrtEnvShadowRay.min_distance = 0.001f;
                pbrtEnvShadowRay.max_distance = 1e5f;
                intersection_result<instancing, triangle_data> pbrtEnvShadowResult =
                    isect.intersect(pbrtEnvShadowRay, accelStructure, functionTable);
                if (pbrtEnvShadowResult.type == intersection_type::none) {
                    float2 pbrtEnvUV = equirectangularUV(pbrtEnvWi);
                    float3 pbrtEnvRadianceSample = pbrtEnvTexture.sample(textureSampler, pbrtEnvUV).rgb;
                    float pbrtEnvPdfBsdf = pbrtEnvCosSurface / M_PI_F;
                    float pbrtEnvWeight = (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle)
                        / (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle + pbrtEnvPdfBsdf * pbrtEnvPdfBsdf);
                    radiance += throughput * float3(normalizedFresnelF(pbrtEnvWi, facingNormal, mat.ior, mat.roughness))
                                * pbrtEnvRadianceSample * pbrtEnvCosSurface / pbrtEnvPdfSolidAngle * pbrtEnvWeight;
                }
            }
        }
    }

    // Continuation ray: cosine-weighted hemisphere sampling, same
    // pdf-cancellation shape as Oren-Nayar's own (see that function's
    // own comment) - `f*cosTheta/pdf` collapses to
    // `normalizedFresnelF(...)*pi`, i.e. `(1-Fr(cosWi,eta))/c` exactly
    // (matching this BxDF's own documented closed-form `sample()`
    // weight, `bxdfs_layered.h`'s own comment - not a coincidence, the
    // same algebra pbrt-v4 itself already derived).
    float3 newDir = cosineSampleHemisphere(facingNormal, rngState);
    rayDir = newDir;
    rayOrigin = hitPoint + facingNormal * 0.001f;
    throughput *= float3(normalizedFresnelF(newDir, facingNormal, mat.ior, mat.roughness)) * M_PI_F;
    bsdfPdf = max(dot(facingNormal, newDir), 0.0001) / M_PI_F;
    specularBounce = false;
    return true;
}

// CoatedDiffuseBxDF (pbrt-v4) - a rough dielectric coat (GGX, real IOR)
// over a Lambertian base, the real physical model behind "lacquered
// wood"/"coated plastic": light either specularly reflects straight off
// the coat's own top surface, or transmits through, bounces around
// (possibly many times) between the coat's underside and the diffuse
// base below, and eventually re-exits through the top - pbrt-v4's own
// LayeredBxDF, a genuine stochastic random walk with NO closed-form
// BSDF value (unlike every earlier material in this file). This port
// deliberately transliterates the ALREADY-SHIPPED, ALREADY-VERIFIED
// OptiX GPU reference (gpu/optix/optix_device_helpers.h's own
// MaterialType::CoatedDiffuse case) rather than re-deriving from CPU's
// own src/shared/bxdfs_layered.h random walk from scratch - OptiX
// already worked out every hard design question (bounce budget, which
// Fresnel convention to use where, how to avoid the "one bounce then
// give up" darkness bug) and left comments explaining each one; this
// keeps that exact, tested shape rather than risking a fresh derivation.
//
// TWO DELIBERATELY DIFFERENT Fresnel conventions, matching OptiX's own
// real (not internally unified) behavior exactly:
//   - The CONTINUATION-ray sampler below (`shadeCoatedDiffuse`'s own
//     diffuse-escape loop) uses `frDielectric(cosOut, 1.0/eta)` (coat-
//     to-air, inverted) for its exit test - OptiX's own custom,
//     simplified per-launch bounce loop (NOT calling CPU's own
//     `layered_sample_local`), independently written and independently
//     tuned until it stopped rendering too dark.
//   - `layeredCoatedDiffuseF()` below (the stochastic NEE/MIS `f()`
//     value, called for real light sampling) uses `frDielectric(cos,
//     eta)` (uninverted) throughout, matching `src/shared/
//     bxdfs_layered.h`'s own `layered_f()` byte-for-byte - the function
//     OptiX's own shading code calls VERBATIM (it's `CPU_GPU`-tagged,
//     compiled unchanged for CUDA), not reimplemented.
// Unifying these into one "consistent" convention would be UNFAITHFUL
// to what OptiX's own already-verified code actually runs - not a bug
// to fix, two genuinely different code paths in the reference itself.
//
// No medium scattering (CoatedDiffuseBxDF never sets one - `medium_
// albedo` is always 0 in both CPU's and OptiX's own construction), so
// that whole branch of the shared CPU random walk is omitted here
// entirely, matching OptiX's own identical omission (its own comment:
// "medium scattering omitted - coateddiffuse never sets a scattering
// medium").
//
// `mat.color` = Lambertian base albedo, `mat.ior` = coat's real
// dielectric IOR, `mat.roughness` = PRECOMPUTED GGX alpha (host-side
// `RoughnessToAlpha`, not squared in-shader - see
// buildCornellCoatedDiffuse()'s own comment, metal_poc.mm).

// Stochastic BSDF value at an arbitrary queried direction (wiLocal
// toward the light, woLocal toward the camera, both already in the
// hit's own local shading frame, z-up) - a direct port of
// `layered_detail::layered_f()` (src/shared/bxdfs_layered.h), medium-
// free, `nSamples` folded to 1 (this scene's own default, matching
// every call site in this codebase). Consumes `rngState` directly
// (this file's own single-stream RNG convention) rather than a
// separate PCG32 sub-stream the way the CPU/OptiX reference does -
// Monte Carlo correctness only needs valid uniform draws, not a
// bitwise-identical sequence to CPU's own.
inline float3 layeredCoatedDiffuseF(float3 wiLocal, float3 woLocal, float eta, float alpha,
                                     float3 albedo, thread uint& rngState) {
    if (wiLocal.z <= 0.0 || woLocal.z <= 0.0) return float3(0.0);
    float3 result = float3(0.0);

    // Zero-bounce term: direct GGX reflection off the coat's own top
    // surface (no penetration at all) - matches the entrance-reflect
    // branch of the walk below exactly (standard GGX reflection shape
    // times FrDielectric at the wi/wo half-vector), deterministic, not
    // part of the stochastic average.
    {
        float3 h = wiLocal + woLocal;
        float hlen = length(h);
        if (hlen > 1e-8) {
            h /= hlen;
            float D = ggxD(h, alpha, alpha);
            float G = ggxG(woLocal, wiLocal, alpha, alpha);
            float cosWiH = dot(wiLocal, h);
            float F0 = frDielectric(cosWiH, eta);
            float val = D * G * F0 / max(4.0 * wiLocal.z * woLocal.z, 1e-8);
            result = float3(val);
        }
    }

    const int kMaxDepth = 10;
    const float kThickness = 0.01;

    float3 wm = sampleGGXVNDF(wiLocal, alpha, alpha, rngState);
    float cosI = dot(wiLocal, wm);
    float Fin = frDielectric(cosI, eta);
    float3 w = 2.0 * cosI * wm - wiLocal;
    w.z = -abs(w.z);
    if (w.z == 0.0) return result;

    float3 beta = float3(1.0 - Fin);
    float3 accum = float3(0.0);

    for (int depth = 0; depth < kMaxDepth; ++depth) {
        if (depth > 3) {
            float rrBeta = max(beta.x, max(beta.y, beta.z));
            if (rrBeta < 0.25) {
                float q = max(0.0, 1.0 - rrBeta);
                if (randFloat(rngState) < q) break;
                beta /= max(1.0 - q, 1e-6);
            }
        }

        // Beer-Lambert transmittance through the coat's own thickness
        // (no medium scattering - see this function's own header
        // comment) - advances to whichever interface `w` is heading
        // toward.
        beta *= exp(-kThickness / max(abs(w.z), 1e-6));
        bool atBottom = (w.z < 0.0);

        if (atBottom) {
            // Bottom-interface Lambertian bounce - always leaves `w`
            // pointing back upward (cosine-sampled about the local +z).
            w = cosineSampleHemisphere(float3(0.0, 0.0, 1.0), rngState);
            beta *= albedo;
        } else {
            // At the top interface from inside: connect toward wo using
            // a plain GGX reflection half-vector between the walk's
            // current direction and wo - the only exit point this
            // reflective BSDF has.
            float3 h2 = w + woLocal;
            float hlen2 = length(h2);
            if (hlen2 > 1e-8) {
                h2 /= hlen2;
                float D2 = ggxD(h2, alpha, alpha);
                float G2 = ggxG(woLocal, w, alpha, alpha);
                float cosWH = dot(w, h2);
                float Fexit = frDielectric(cosWH, eta);
                float shape = D2 * G2 / max(4.0 * w.z * woLocal.z, 1e-8);
                accum += beta * (shape * (1.0 - Fexit)) * woLocal.z;
            }

            // Continue the walk: deterministic internal reflection,
            // weighted by the actual reflectance (the transmission/exit
            // possibility was already accounted for by the connection
            // above).
            float3 wm2 = sampleGGXVNDF(w, alpha, alpha, rngState);
            float cos2 = dot(w, wm2);
            float Fout = frDielectric(cos2, eta);
            float3 r2 = 2.0 * cos2 * wm2 - w;
            r2.z = -abs(r2.z);
            w = r2;
            beta *= Fout;
        }
    }

    result += accum;
    return result;
}

// GGX VNDF reflection pdf (same `D*G1/(4*NdotO)` shape already used by
// materialType 4/9's own NEE weight, section ~104) - reused here as the
// cheap, shape-matched MIS proxy pdf for this material's own unbounded-
// depth random walk, which has no real closed-form pdf at all. Any
// valid pdf keeps MIS/NEE unbiased (only variance is affected) - the
// same choice OptiX's own `ggx_vndf_reflection_pdf` already makes for
// this exact material.
inline float coatedDiffuseProxyPdf(float3 woLocal, float3 wiLocal, float alpha) {
    float3 h = woLocal + wiLocal;
    float hlen = length(h);
    if (hlen < 1e-8) return 0.0;
    h /= hlen;
    float D = ggxD(h, alpha, alpha);
    float G1 = ggxG1(woLocal, alpha, alpha);
    return (D * G1) / max(4.0 * woLocal.z, 1e-6);
}

inline bool shadeCoatedDiffuse(TriangleMaterial mat, float3 hitPoint, float3 facingNormal,
                            constant Uniforms& uniforms,
                            device const AreaLight* lights,
                            device const PointLight* pointLights,
                            device const DirectionalLight* directionalLights,
                            device const ProjectionLight* projectionLights,
                            device const GoniometricLight* goniometricLights,
                            device const float* envMarginalCDF,
                            device const float* envConditionalCDF,
                            uint envMapWidth, uint envMapHeight,
                            device const float* pbrtEnvMarginalCDF,
                            device const float* pbrtEnvConditionalCDF,
                            uint pbrtEnvMapWidth, uint pbrtEnvMapHeight,
                            texture2d<float, access::sample> earthTexture,
                            texture2d<float, access::sample> pbrtEnvTexture,
                            texture2d<float, access::sample> goniometricTexture,
                            texture2d<float, access::sample> pbrtGoniometricTexture,
                            texture2d<float, access::sample> pbrtProjectionTexture,
                            texture2d<float, access::sample> pbrtAreaLightTexture,
                            sampler textureSampler,
                            intersector<instancing, triangle_data> isect,
                            instance_acceleration_structure accelStructure,
                            intersection_function_table<instancing, triangle_data> functionTable,
                            thread float3& rayDir, thread float3& rayOrigin,
                            thread float3& throughput, thread float3& radiance,
                            thread float& bsdfPdf, thread bool& specularBounce, thread uint& rngState) {
    float alpha = max(mat.roughness, 0.0001);
    bool effectivelySmooth = alpha < 0.001;  // TrowbridgeReitz::EffectivelySmooth() threshold

    float3 tangent, bitangent;
    buildAnisotropicOnb(facingNormal, tangent, bitangent);
    float3 woWorld = -rayDir;
    float3 woLocal = float3(dot(woWorld, tangent), dot(woWorld, bitangent), dot(woWorld, facingNormal));
    woLocal.z = max(woLocal.z, 0.0001);

    if (!effectivelySmooth && all(mat.emission == float3(0.0))) {
        LightSample ls = sampleAreaLight(lights, uniforms.lightCount, rngState, pbrtAreaLightTexture, textureSampler);
        float3 toLight = ls.point - hitPoint;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        float3 wi = toLight / dist;
        float cosSurface = dot(facingNormal, wi);
        float cosLight = dot(ls.normal, -wi);
        if (cosSurface > 0.0 && (cosLight > 0.0 || (ls.twoSided != 0.0 && cosLight < 0.0))) {
            ray shadowRay;
            shadowRay.origin = hitPoint + facingNormal * 0.001f;
            shadowRay.direction = wi;
            shadowRay.min_distance = 0.001f;
            shadowRay.max_distance = dist - 0.002f;
            intersection_result<instancing, triangle_data> shadowResult =
                isect.intersect(shadowRay, accelStructure, functionTable);
            if (shadowResult.type == intersection_type::none) {
                float3 wiLocal = float3(dot(wi, tangent), dot(wi, bitangent), dot(wi, facingNormal));
                float3 f = layeredCoatedDiffuseF(wiLocal, woLocal, mat.ior, alpha, float3(mat.color), rngState);
                float pdfSolidAngle = (distSq / (ls.area * abs(cosLight))) * ls.pmf;
                float pdfBsdf = coatedDiffuseProxyPdf(woLocal, wiLocal, alpha);
                float weight = (pdfSolidAngle * pdfSolidAngle)
                    / (pdfSolidAngle * pdfSolidAngle + pdfBsdf * pdfBsdf);
                float transmittance = exp(-uniforms.fogSigmaT * dist);
                radiance += throughput * f * ls.emission * cosSurface * transmittance / pdfSolidAngle * weight;
            }
        }

        for (uint pli = 0; pli < uniforms.pointLightCount; ++pli) {
            PointLight pl = pointLights[pli];
            float3 toPointLight = float3(pl.position) - hitPoint;
            float plDistSq = dot(toPointLight, toPointLight);
            float plDist = sqrt(plDistSq);
            float3 plWi = toPointLight / plDist;
            float plCosSurface = dot(facingNormal, plWi);
            if (plCosSurface > 0.0) {
                ray plShadowRay;
                plShadowRay.origin = hitPoint + facingNormal * 0.001f;
                plShadowRay.direction = plWi;
                plShadowRay.min_distance = 0.001f;
                plShadowRay.max_distance = plDist - 0.002f;
                intersection_result<instancing, triangle_data> plShadowResult =
                    isect.intersect(plShadowRay, accelStructure, functionTable);
                if (plShadowResult.type == intersection_type::none) {
                    float3 plWiLocal = float3(dot(plWi, tangent), dot(plWi, bitangent), dot(plWi, facingNormal));
                    float3 plF = layeredCoatedDiffuseF(plWiLocal, woLocal, mat.ior, alpha, float3(mat.color), rngState);
                    float plTransmittance = exp(-uniforms.fogSigmaT * plDist);
                    float plSpot = spotLightFalloff(-plWi, float3(pl.direction), pl.cosOuterAngle, pl.cosInnerAngle);
                    radiance += throughput * plF * float3(pl.emission) * plCosSurface * plSpot * plTransmittance / plDistSq;
                }
            }
        }

        for (uint dli = 0; dli < uniforms.directionalLightCount; ++dli) {
            DirectionalLight dl = directionalLights[dli];
            float3 dlWi = normalize(-float3(dl.direction));
            float dlCosSurface = dot(facingNormal, dlWi);
            if (dlCosSurface > 0.0) {
                ray dlShadowRay;
                dlShadowRay.origin = hitPoint + facingNormal * 0.001f;
                dlShadowRay.direction = dlWi;
                dlShadowRay.min_distance = 0.001f;
                dlShadowRay.max_distance = kDirectionalLightMaxDistance;
                intersection_result<instancing, triangle_data> dlShadowResult =
                    isect.intersect(dlShadowRay, accelStructure, functionTable);
                if (dlShadowResult.type == intersection_type::none) {
                    float3 dlWiLocal = float3(dot(dlWi, tangent), dot(dlWi, bitangent), dot(dlWi, facingNormal));
                    float3 dlF = layeredCoatedDiffuseF(dlWiLocal, woLocal, mat.ior, alpha, float3(mat.color), rngState);
                    // Skip when there is no fog - rayBoxExitDistance()
                    // is only valid for an origin INSIDE the hardcoded
                    // room bounds (see that function.s own comment for
                    // why calling it unconditionally is unsafe).
                    float dlTransmittance = (uniforms.fogSigmaT > 0.0)
                        ? exp(-uniforms.fogSigmaT * rayBoxExitDistance(dlShadowRay.origin, dlWi, kRoomBoundsMin, kRoomBoundsMax))
                        : 1.0;
                    radiance += throughput * dlF * float3(dl.emission) * dlCosSurface * dlTransmittance;
                }
            }
        }

        for (uint pji = 0; pji < uniforms.projectionLightCount; ++pji) {
            ProjectionLight pj = projectionLights[pji];
            float3 toProjLight = float3(pj.position) - hitPoint;
            float pjDistSq = dot(toProjLight, toProjLight);
            float pjDist = sqrt(pjDistSq);
            float3 pjWi = toProjLight / pjDist;
            float pjCosSurface = dot(facingNormal, pjWi);
            if (pjCosSurface > 0.0) {
                float3 pjRadiance = projectionLightRadiance(-pjWi, pj.forward, pj.right, pj.up,
                                                             pj.tanHalfFovX, pj.tanHalfFovY, pj.scale,
                                                             (pj.usePbrtTexture != 0u ? pbrtProjectionTexture : earthTexture), textureSampler);
                if (any(pjRadiance > float3(0.0))) {
                    ray pjShadowRay;
                    pjShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    pjShadowRay.direction = pjWi;
                    pjShadowRay.min_distance = 0.001f;
                    pjShadowRay.max_distance = pjDist - 0.002f;
                    intersection_result<instancing, triangle_data> pjShadowResult =
                        isect.intersect(pjShadowRay, accelStructure, functionTable);
                    if (pjShadowResult.type == intersection_type::none) {
                        float3 pjWiLocal = float3(dot(pjWi, tangent), dot(pjWi, bitangent), dot(pjWi, facingNormal));
                        float3 pjF = layeredCoatedDiffuseF(pjWiLocal, woLocal, mat.ior, alpha, float3(mat.color), rngState);
                        float pjTransmittance = exp(-uniforms.fogSigmaT * pjDist);
                        radiance += throughput * pjF * pjRadiance * pjCosSurface * pjTransmittance / pjDistSq;
                    }
                }
            }
        }

        for (uint gli = 0; gli < uniforms.goniometricLightCount; ++gli) {
            GoniometricLight gl = goniometricLights[gli];
            float3 toGoniLight = float3(gl.position) - hitPoint;
            float glDistSq = dot(toGoniLight, toGoniLight);
            float glDist = sqrt(glDistSq);
            float3 glWi = toGoniLight / glDist;
            float glCosSurface = dot(facingNormal, glWi);
            if (glCosSurface > 0.0) {
                float3 glRadiance = goniometricLightRadiance(-glWi, gl.forward, gl.right, gl.up,
                                                              gl.emission, gl.scale,
                                                              (gl.usePbrtTexture != 0u ? pbrtGoniometricTexture : goniometricTexture), textureSampler);
                if (any(glRadiance > float3(0.0))) {
                    ray glShadowRay;
                    glShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    glShadowRay.direction = glWi;
                    glShadowRay.min_distance = 0.001f;
                    glShadowRay.max_distance = glDist - 0.002f;
                    intersection_result<instancing, triangle_data> glShadowResult =
                        isect.intersect(glShadowRay, accelStructure, functionTable);
                    if (glShadowResult.type == intersection_type::none) {
                        float3 glWiLocal = float3(dot(glWi, tangent), dot(glWi, bitangent), dot(glWi, facingNormal));
                        float3 glF = layeredCoatedDiffuseF(glWiLocal, woLocal, mat.ior, alpha, float3(mat.color), rngState);
                        float glTransmittance = exp(-uniforms.fogSigmaT * glDist);
                        radiance += throughput * glF * glRadiance * glCosSurface * glTransmittance;
                    }
                }
            }
        }
    }

    // Continuation ray: entrance test at the coat's own top surface,
    // then either a specular GGX reflection (probability F_in) or a
    // transmit-and-random-walk-through-the-base escape (probability
    // 1-F_in) - transliterated directly from OptiX's own already-
    // verified `MaterialType::CoatedDiffuse` bounce loop (see this
    // function's own header comment on why NOT CPU's
    // `layered_sample_local` instead).
    float3 wm = sampleGGXVNDF(woLocal, alpha, alpha, rngState);
    float cosI = dot(woLocal, wm);
    float Fin = frDielectric(cosI, mat.ior);
    float3 newDirLocal;
    float3 beta;
    if (randFloat(rngState) < Fin) {
        float3 reflLocal = 2.0 * cosI * wm - woLocal;
        if (reflLocal.z <= 0.0) return false;
        float G1 = ggxG1(woLocal, alpha, alpha);
        float G = ggxG(reflLocal, woLocal, alpha, alpha);
        float w = (G1 > 1e-8) ? G / G1 : 0.0;
        float fw = Fin * w;
        beta = float3(fw, fw, fw);
        newDirLocal = reflLocal;
    } else {
        constexpr int kMaxCoatBounces = 8;
        float3 b = float3(1.0 - Fin);
        float3 diffDirLocal = float3(0.0, 0.0, 1.0);
        bool escaped = false;
        for (int cb = 0; cb < kMaxCoatBounces; ++cb) {
            diffDirLocal = cosineSampleHemisphere(float3(0.0, 0.0, 1.0), rngState);
            b *= float3(mat.color);
            float3 wm2 = sampleGGXVNDF(diffDirLocal, alpha, alpha, rngState);
            float cosOut = dot(diffDirLocal, wm2);
            // Coat-to-air exit test - inverted eta, matching OptiX's own
            // sample loop exactly (see this function's own header
            // comment on the two deliberately different conventions).
            float Fout = frDielectric(cosOut, 1.0 / mat.ior);
            if (randFloat(rngState) < Fout) continue;  // TIR: bounce again
            b *= (1.0 - Fout);
            escaped = true;
            break;
        }
        if (!escaped) return false;
        beta = b;
        newDirLocal = diffDirLocal;
    }

    float3 newDirWorld = normalize(newDirLocal.x * tangent + newDirLocal.y * bitangent + newDirLocal.z * facingNormal);
    rayDir = newDirWorld;
    rayOrigin = hitPoint + facingNormal * 0.001f;
    throughput *= beta;
    if (!effectivelySmooth) {
        bsdfPdf = coatedDiffuseProxyPdf(woLocal, newDirLocal, alpha);
        specularBounce = false;
    } else {
        specularBounce = true;
    }
    return true;
}

// CoatedConductorBxDF (pbrt-v4) - the SAME rough-dielectric-coat random
// walk `shadeCoatedDiffuse()` just above already implements (see that
// function's own header comment for the full derivation/rationale -
// same OptiX-reference-not-CPU-reference porting strategy, same "two
// deliberately different Fresnel conventions" note), with the bottom
// interface swapped from a Lambertian cosine bounce to a GGX-conductor
// specular bounce (real per-channel complex Fresnel, `frComplexRGB()`).
// Ported from `gpu/optix/optix_device_helpers.h`'s own
// `MaterialType::CoatedConductor` case, which is genuinely SIMPLER than
// CoatedDiffuse's own sample step for one real physical reason: a
// conductor's bottom bounce is a single specular GGX reflection (one
// direction in, one direction out), not a Lambertian bounce that can
// need several retries before finding an exit angle that clears the
// coat - so the continuation sampler below needs no retry loop at all,
// unlike `shadeCoatedDiffuse()`'s own `kMaxCoatBounces` loop.
//
// `mat.conductorEta`/`mat.conductorK` = the metal base's own real
// per-channel complex IOR (same fields materialType 4/9 already use);
// `mat.color` is UNUSED (this material has no separate diffuse albedo
// at all - the metal's own colour comes entirely from its complex
// Fresnel). `mat.ior`/`mat.roughness` are the coat's own (real IOR,
// precomputed GGX alpha), same convention as materialType 19.

// Stochastic BSDF value (see `layeredCoatedDiffuseF()`'s own header
// comment for the full derivation) with a GGX-conductor bottom bounce
// in place of the Lambertian one - direct port of
// `layered_detail::ConductorBottomBounce::bounce()` (src/shared/
// bxdfs_layered.h) folded into the SAME shared random-walk shape
// `layeredCoatedDiffuseF()` already implements.
inline float3 layeredCoatedConductorF(float3 wiLocal, float3 woLocal, float eta, float alpha,
                                       float3 conductorEta, float3 conductorK, thread uint& rngState) {
    if (wiLocal.z <= 0.0 || woLocal.z <= 0.0) return float3(0.0);
    float3 result = float3(0.0);

    {
        float3 h = wiLocal + woLocal;
        float hlen = length(h);
        if (hlen > 1e-8) {
            h /= hlen;
            float D = ggxD(h, alpha, alpha);
            float G = ggxG(woLocal, wiLocal, alpha, alpha);
            float cosWiH = dot(wiLocal, h);
            float F0 = frDielectric(cosWiH, eta);
            float val = D * G * F0 / max(4.0 * wiLocal.z * woLocal.z, 1e-8);
            result = float3(val);
        }
    }

    const int kMaxDepth = 10;
    const float kThickness = 0.01;

    float3 wm = sampleGGXVNDF(wiLocal, alpha, alpha, rngState);
    float cosI = dot(wiLocal, wm);
    float Fin = frDielectric(cosI, eta);
    float3 w = 2.0 * cosI * wm - wiLocal;
    w.z = -abs(w.z);
    if (w.z == 0.0) return result;

    float3 beta = float3(1.0 - Fin);
    float3 accum = float3(0.0);

    for (int depth = 0; depth < kMaxDepth; ++depth) {
        if (depth > 3) {
            float rrBeta = max(beta.x, max(beta.y, beta.z));
            if (rrBeta < 0.25) {
                float q = max(0.0, 1.0 - rrBeta);
                if (randFloat(rngState) < q) break;
                beta /= max(1.0 - q, 1e-6);
            }
        }

        beta *= exp(-kThickness / max(abs(w.z), 1e-6));
        bool atBottom = (w.z < 0.0);

        if (atBottom) {
            // GGX-conductor bottom bounce (ConductorBottomBounce::bounce()) -
            // flip to the conductor's own "incoming from above" frame,
            // sample a VNDF half-vector, reflect, weight by real complex
            // Fresnel times the height-correlated G/G1 ratio, and always
            // leave `w` pointing back upward.
            float3 fw = -w;
            float3 bwm = sampleGGXVNDF(fw, alpha, alpha, rngState);
            float cosC = dot(fw, bwm);
            float3 rwo = 2.0 * cosC * bwm - fw;
            float G1c = ggxG1(fw, alpha, alpha);
            float Gc = ggxG(rwo, fw, alpha, alpha);
            float wtC = (G1c > 1e-8) ? Gc / G1c : 0.0;
            beta *= frComplexRGB(cosC, conductorEta, conductorK) * wtC;
            w = float3(rwo.x, rwo.y, abs(rwo.z));
        } else {
            float3 h2 = w + woLocal;
            float hlen2 = length(h2);
            if (hlen2 > 1e-8) {
                h2 /= hlen2;
                float D2 = ggxD(h2, alpha, alpha);
                float G2 = ggxG(woLocal, w, alpha, alpha);
                float cosWH = dot(w, h2);
                float Fexit = frDielectric(cosWH, eta);
                float shape = D2 * G2 / max(4.0 * w.z * woLocal.z, 1e-8);
                accum += beta * (shape * (1.0 - Fexit)) * woLocal.z;
            }

            float3 wm2 = sampleGGXVNDF(w, alpha, alpha, rngState);
            float cos2 = dot(w, wm2);
            float Fout = frDielectric(cos2, eta);
            float3 r2 = 2.0 * cos2 * wm2 - w;
            r2.z = -abs(r2.z);
            w = r2;
            beta *= Fout;
        }
    }

    result += accum;
    return result;
}

inline bool shadeCoatedConductor(TriangleMaterial mat, float3 hitPoint, float3 facingNormal,
                            constant Uniforms& uniforms,
                            device const AreaLight* lights,
                            device const PointLight* pointLights,
                            device const DirectionalLight* directionalLights,
                            device const ProjectionLight* projectionLights,
                            device const GoniometricLight* goniometricLights,
                            device const float* envMarginalCDF,
                            device const float* envConditionalCDF,
                            uint envMapWidth, uint envMapHeight,
                            device const float* pbrtEnvMarginalCDF,
                            device const float* pbrtEnvConditionalCDF,
                            uint pbrtEnvMapWidth, uint pbrtEnvMapHeight,
                            texture2d<float, access::sample> earthTexture,
                            texture2d<float, access::sample> pbrtEnvTexture,
                            texture2d<float, access::sample> goniometricTexture,
                            texture2d<float, access::sample> pbrtGoniometricTexture,
                            texture2d<float, access::sample> pbrtProjectionTexture,
                            texture2d<float, access::sample> pbrtAreaLightTexture,
                            sampler textureSampler,
                            intersector<instancing, triangle_data> isect,
                            instance_acceleration_structure accelStructure,
                            intersection_function_table<instancing, triangle_data> functionTable,
                            thread float3& rayDir, thread float3& rayOrigin,
                            thread float3& throughput, thread float3& radiance,
                            thread float& bsdfPdf, thread bool& specularBounce, thread uint& rngState) {
    float alpha = max(mat.roughness, 0.0001);
    bool effectivelySmooth = alpha < 0.001;

    float3 tangent, bitangent;
    buildAnisotropicOnb(facingNormal, tangent, bitangent);
    float3 woWorld = -rayDir;
    float3 woLocal = float3(dot(woWorld, tangent), dot(woWorld, bitangent), dot(woWorld, facingNormal));
    woLocal.z = max(woLocal.z, 0.0001);
    float3 conductorEta = float3(mat.conductorEta);
    float3 conductorK = float3(mat.conductorK);

    if (!effectivelySmooth && all(mat.emission == float3(0.0))) {
        LightSample ls = sampleAreaLight(lights, uniforms.lightCount, rngState, pbrtAreaLightTexture, textureSampler);
        float3 toLight = ls.point - hitPoint;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        float3 wi = toLight / dist;
        float cosSurface = dot(facingNormal, wi);
        float cosLight = dot(ls.normal, -wi);
        if (cosSurface > 0.0 && (cosLight > 0.0 || (ls.twoSided != 0.0 && cosLight < 0.0))) {
            ray shadowRay;
            shadowRay.origin = hitPoint + facingNormal * 0.001f;
            shadowRay.direction = wi;
            shadowRay.min_distance = 0.001f;
            shadowRay.max_distance = dist - 0.002f;
            intersection_result<instancing, triangle_data> shadowResult =
                isect.intersect(shadowRay, accelStructure, functionTable);
            if (shadowResult.type == intersection_type::none) {
                float3 wiLocal = float3(dot(wi, tangent), dot(wi, bitangent), dot(wi, facingNormal));
                float3 f = layeredCoatedConductorF(wiLocal, woLocal, mat.ior, alpha, conductorEta, conductorK, rngState);
                float pdfSolidAngle = (distSq / (ls.area * abs(cosLight))) * ls.pmf;
                float pdfBsdf = coatedDiffuseProxyPdf(woLocal, wiLocal, alpha);
                float weight = (pdfSolidAngle * pdfSolidAngle)
                    / (pdfSolidAngle * pdfSolidAngle + pdfBsdf * pdfBsdf);
                float transmittance = exp(-uniforms.fogSigmaT * dist);
                radiance += throughput * f * ls.emission * cosSurface * transmittance / pdfSolidAngle * weight;
            }
        }

        for (uint pli = 0; pli < uniforms.pointLightCount; ++pli) {
            PointLight pl = pointLights[pli];
            float3 toPointLight = float3(pl.position) - hitPoint;
            float plDistSq = dot(toPointLight, toPointLight);
            float plDist = sqrt(plDistSq);
            float3 plWi = toPointLight / plDist;
            float plCosSurface = dot(facingNormal, plWi);
            if (plCosSurface > 0.0) {
                ray plShadowRay;
                plShadowRay.origin = hitPoint + facingNormal * 0.001f;
                plShadowRay.direction = plWi;
                plShadowRay.min_distance = 0.001f;
                plShadowRay.max_distance = plDist - 0.002f;
                intersection_result<instancing, triangle_data> plShadowResult =
                    isect.intersect(plShadowRay, accelStructure, functionTable);
                if (plShadowResult.type == intersection_type::none) {
                    float3 plWiLocal = float3(dot(plWi, tangent), dot(plWi, bitangent), dot(plWi, facingNormal));
                    float3 plF = layeredCoatedConductorF(plWiLocal, woLocal, mat.ior, alpha, conductorEta, conductorK, rngState);
                    float plTransmittance = exp(-uniforms.fogSigmaT * plDist);
                    float plSpot = spotLightFalloff(-plWi, float3(pl.direction), pl.cosOuterAngle, pl.cosInnerAngle);
                    radiance += throughput * plF * float3(pl.emission) * plCosSurface * plSpot * plTransmittance / plDistSq;
                }
            }
        }

        for (uint dli = 0; dli < uniforms.directionalLightCount; ++dli) {
            DirectionalLight dl = directionalLights[dli];
            float3 dlWi = normalize(-float3(dl.direction));
            float dlCosSurface = dot(facingNormal, dlWi);
            if (dlCosSurface > 0.0) {
                ray dlShadowRay;
                dlShadowRay.origin = hitPoint + facingNormal * 0.001f;
                dlShadowRay.direction = dlWi;
                dlShadowRay.min_distance = 0.001f;
                dlShadowRay.max_distance = kDirectionalLightMaxDistance;
                intersection_result<instancing, triangle_data> dlShadowResult =
                    isect.intersect(dlShadowRay, accelStructure, functionTable);
                if (dlShadowResult.type == intersection_type::none) {
                    float3 dlWiLocal = float3(dot(dlWi, tangent), dot(dlWi, bitangent), dot(dlWi, facingNormal));
                    float3 dlF = layeredCoatedConductorF(dlWiLocal, woLocal, mat.ior, alpha, conductorEta, conductorK, rngState);
                    // Skip when there is no fog - rayBoxExitDistance()
                    // is only valid for an origin INSIDE the hardcoded
                    // room bounds (see that function.s own comment for
                    // why calling it unconditionally is unsafe).
                    float dlTransmittance = (uniforms.fogSigmaT > 0.0)
                        ? exp(-uniforms.fogSigmaT * rayBoxExitDistance(dlShadowRay.origin, dlWi, kRoomBoundsMin, kRoomBoundsMax))
                        : 1.0;
                    radiance += throughput * dlF * float3(dl.emission) * dlCosSurface * dlTransmittance;
                }
            }
        }

        for (uint pji = 0; pji < uniforms.projectionLightCount; ++pji) {
            ProjectionLight pj = projectionLights[pji];
            float3 toProjLight = float3(pj.position) - hitPoint;
            float pjDistSq = dot(toProjLight, toProjLight);
            float pjDist = sqrt(pjDistSq);
            float3 pjWi = toProjLight / pjDist;
            float pjCosSurface = dot(facingNormal, pjWi);
            if (pjCosSurface > 0.0) {
                float3 pjRadiance = projectionLightRadiance(-pjWi, pj.forward, pj.right, pj.up,
                                                             pj.tanHalfFovX, pj.tanHalfFovY, pj.scale,
                                                             (pj.usePbrtTexture != 0u ? pbrtProjectionTexture : earthTexture), textureSampler);
                if (any(pjRadiance > float3(0.0))) {
                    ray pjShadowRay;
                    pjShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    pjShadowRay.direction = pjWi;
                    pjShadowRay.min_distance = 0.001f;
                    pjShadowRay.max_distance = pjDist - 0.002f;
                    intersection_result<instancing, triangle_data> pjShadowResult =
                        isect.intersect(pjShadowRay, accelStructure, functionTable);
                    if (pjShadowResult.type == intersection_type::none) {
                        float3 pjWiLocal = float3(dot(pjWi, tangent), dot(pjWi, bitangent), dot(pjWi, facingNormal));
                        float3 pjF = layeredCoatedConductorF(pjWiLocal, woLocal, mat.ior, alpha, conductorEta, conductorK, rngState);
                        float pjTransmittance = exp(-uniforms.fogSigmaT * pjDist);
                        radiance += throughput * pjF * pjRadiance * pjCosSurface * pjTransmittance / pjDistSq;
                    }
                }
            }
        }

        for (uint gli = 0; gli < uniforms.goniometricLightCount; ++gli) {
            GoniometricLight gl = goniometricLights[gli];
            float3 toGoniLight = float3(gl.position) - hitPoint;
            float glDistSq = dot(toGoniLight, toGoniLight);
            float glDist = sqrt(glDistSq);
            float3 glWi = toGoniLight / glDist;
            float glCosSurface = dot(facingNormal, glWi);
            if (glCosSurface > 0.0) {
                float3 glRadiance = goniometricLightRadiance(-glWi, gl.forward, gl.right, gl.up,
                                                              gl.emission, gl.scale,
                                                              (gl.usePbrtTexture != 0u ? pbrtGoniometricTexture : goniometricTexture), textureSampler);
                if (any(glRadiance > float3(0.0))) {
                    ray glShadowRay;
                    glShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    glShadowRay.direction = glWi;
                    glShadowRay.min_distance = 0.001f;
                    glShadowRay.max_distance = glDist - 0.002f;
                    intersection_result<instancing, triangle_data> glShadowResult =
                        isect.intersect(glShadowRay, accelStructure, functionTable);
                    if (glShadowResult.type == intersection_type::none) {
                        float3 glWiLocal = float3(dot(glWi, tangent), dot(glWi, bitangent), dot(glWi, facingNormal));
                        float3 glF = layeredCoatedConductorF(glWiLocal, woLocal, mat.ior, alpha, conductorEta, conductorK, rngState);
                        float glTransmittance = exp(-uniforms.fogSigmaT * glDist);
                        radiance += throughput * glF * glRadiance * glCosSurface * glTransmittance;
                    }
                }
            }
        }
    }

    // Continuation ray: entrance test at the coat's own top surface,
    // then either a specular GGX reflection (probability F_in) or a
    // transmit -> single-bounce GGX-conductor reflection -> deterministic
    // exit test (probability 1-F_in) - NO retry loop needed (unlike
    // materialType 19's own diffuse-escape loop), since a conductor's
    // own bottom bounce is a single specular direction, not a spread
    // that can miss the coat's own exit cone and need another try.
    float3 wm = sampleGGXVNDF(woLocal, alpha, alpha, rngState);
    float cosI = dot(woLocal, wm);
    float Fin = frDielectric(cosI, mat.ior);
    float3 newDirLocal;
    float3 beta;
    if (randFloat(rngState) < Fin) {
        float3 reflLocal = 2.0 * cosI * wm - woLocal;
        if (reflLocal.z <= 0.0) return false;
        float G1 = ggxG1(woLocal, alpha, alpha);
        float G = ggxG(reflLocal, woLocal, alpha, alpha);
        float w = (G1 > 1e-8) ? G / G1 : 0.0;
        float fw = Fin * w;
        beta = float3(fw, fw, fw);
        newDirLocal = reflLocal;
    } else {
        float3 wDown = 2.0 * cosI * wm - woLocal;
        if (wDown.z > 0.0) wDown.z = -wDown.z;
        if (wDown.z == 0.0) return false;

        float3 fw = -wDown;
        float3 bwm = sampleGGXVNDF(fw, alpha, alpha, rngState);
        float cosC = dot(fw, bwm);
        if (cosC <= 0.0) return false;
        float3 rwo = 2.0 * cosC * bwm - fw;
        if (rwo.z <= 0.0) return false;

        float G1c = ggxG1(fw, alpha, alpha);
        float Gc = ggxG(rwo, fw, alpha, alpha);
        float wtC = (G1c > 1e-8) ? Gc / G1c : 0.0;
        float3 Fc = frComplexRGB(cosC, conductorEta, conductorK) * wtC;

        // Coat-to-air exit test - inverted eta, matching OptiX's own
        // sample loop exactly (same convention shadeCoatedDiffuse()'s
        // own diffuse-escape loop uses, but a single deterministic test
        // here, not a retry loop - see this function's own header
        // comment).
        float Fout = frDielectric(rwo.z, 1.0 / mat.ior);
        float Tout = 1.0 - Fout;
        float Tin = 1.0 - Fin;
        beta = Fc * (Tin * Tout);
        newDirLocal = rwo;
    }

    float3 newDirWorld = normalize(newDirLocal.x * tangent + newDirLocal.y * bitangent + newDirLocal.z * facingNormal);
    rayDir = newDirWorld;
    rayOrigin = hitPoint + facingNormal * 0.001f;
    throughput *= beta;
    if (!effectivelySmooth) {
        bsdfPdf = coatedDiffuseProxyPdf(woLocal, newDirLocal, alpha);
        specularBounce = false;
    } else {
        specularBounce = true;
    }
    return true;
}

// Ashikhmin & Shirley's own velvet BRDF (2000; this exact port, though,
// mirrors Blender Cycles' own kernel/closure/bsdf_ashikhmin_velvet.h,
// itself adapted from Open Shading Language) - the classic fabric/
// cloth "fuzzy grazing-angle rim glow" look: a Blinn-Phong-shaped
// microfacet distribution `D` (peaked when the half-vector sits near
// the TANGENT plane, not near the normal the way every specular
// material in this POC so far peaks) times a heuristic geometric term
// `G`, giving a BRDF that's genuinely near-ZERO for head-on view/light
// (no highlight at all, unlike every other glossy material here) and
// rises toward a real peak somewhere around 75-80 degrees before
// falling off again approaching true grazing - not a monotonic curve,
// a real physical signature confirmed against a fresh reference
// program below, not assumed from the formula alone. `sigma` (this
// material's own roughness-like spread parameter) reuses `mat.ior`
// (materialType 2/4/5/9/11 already each reuse this same field their
// own way - one more reuse, not a new struct field); `G`'s own "TODO:
// derive G from D analytically" comment in the reference is Cycles'
// own, not this port's - a known, accepted heuristic in the original
// source, left exactly as-is here rather than silently "fixing" it.
inline float velvetF(float3 wo, float3 wi, float3 n, float sigma) {
    float invSigma2 = 1.0 / (sigma * sigma);
    float cosNO = dot(n, wo);
    float cosNI = dot(n, wi);
    if (cosNO <= 0.0 || cosNI <= 0.0) {
        return 0.0;
    }
    float3 h = normalize(wo + wi);
    float cosNH = dot(n, h);
    float cosH = abs(dot(wo, h));
    if (abs(cosNH) >= 1.0 - 1e-5 || cosH <= 1e-5) {
        return 0.0;
    }
    float cosNHdivH = max(cosNH / cosH, 1e-5);
    float fac1 = 2.0 * abs(cosNHdivH * cosNO);
    float fac2 = 2.0 * abs(cosNHdivH * cosNI);
    float sinNH2 = 1.0 - cosNH * cosNH;
    float sinNH4 = sinNH2 * sinNH2;
    float cot2 = (cosNH * cosNH) / sinNH2;
    float D = exp(-cot2 * invSigma2) * invSigma2 * M_1_PI_F / sinNH4;
    float G = min(1.0, min(fac1, fac2));
    return 0.25 * (D * G) / cosNO;
}

inline bool shadeVelvet(TriangleMaterial mat, float3 albedo, float3 hitPoint, float3 facingNormal,
                          constant Uniforms& uniforms,
                          device const AreaLight* lights,
                          device const PointLight* pointLights,
                          device const DirectionalLight* directionalLights,
                          device const ProjectionLight* projectionLights,
                          device const GoniometricLight* goniometricLights,
                          device const float* envMarginalCDF,
                          device const float* envConditionalCDF,
                          uint envMapWidth, uint envMapHeight,
                          device const float* pbrtEnvMarginalCDF,
                          device const float* pbrtEnvConditionalCDF,
                          uint pbrtEnvMapWidth, uint pbrtEnvMapHeight,
                          texture2d<float, access::sample> earthTexture,
                          texture2d<float, access::sample> pbrtEnvTexture,
                          texture2d<float, access::sample> goniometricTexture,
                          texture2d<float, access::sample> pbrtGoniometricTexture,
                          texture2d<float, access::sample> pbrtProjectionTexture,
                          texture2d<float, access::sample> pbrtAreaLightTexture,
                          sampler textureSampler,
                          intersector<instancing, triangle_data> isect,
                          instance_acceleration_structure accelStructure,
                          intersection_function_table<instancing, triangle_data> functionTable,
                          thread float3& rayDir, thread float3& rayOrigin,
                          thread float3& throughput, thread float3& radiance,
                          thread float& bsdfPdf, thread bool& specularBounce, thread uint& rngState) {
    float3 woWorld = -rayDir;
    // UNIFORM hemisphere sampling (not cosine-weighted, see
    // sampleUniformHemisphere's own comment) - the competing BSDF pdf
    // for MIS against every light below is the CONSTANT 1/(2*pi), not
    // Lambertian/Oren-Nayar's own direction-dependent cosTheta/pi.
    float uniformPdf = 1.0 / (2.0 * M_PI_F);

    if (all(mat.emission == float3(0.0))) {
        LightSample ls = sampleAreaLight(lights, uniforms.lightCount, rngState, pbrtAreaLightTexture, textureSampler);
        float3 toLight = ls.point - hitPoint;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        float3 wi = toLight / dist;
        float cosSurface = dot(facingNormal, wi);
        float cosLight = dot(ls.normal, -wi);
        if (cosSurface > 0.0 && (cosLight > 0.0 || (ls.twoSided != 0.0 && cosLight < 0.0))) {
            ray shadowRay;
            shadowRay.origin = hitPoint + facingNormal * 0.001f;
            shadowRay.direction = wi;
            shadowRay.min_distance = 0.001f;
            shadowRay.max_distance = dist - 0.002f;
            intersection_result<instancing, triangle_data> shadowResult =
                isect.intersect(shadowRay, accelStructure, functionTable);
            if (shadowResult.type == intersection_type::none) {
                float pdfSolidAngle = (distSq / (ls.area * abs(cosLight))) * ls.pmf;
                float weight = (pdfSolidAngle * pdfSolidAngle)
                    / (pdfSolidAngle * pdfSolidAngle + uniformPdf * uniformPdf);
                float transmittance = exp(-uniforms.fogSigmaT * dist);
                radiance += throughput * albedo * velvetF(woWorld, wi, facingNormal, mat.ior)
                            * ls.emission * cosSurface * transmittance / pdfSolidAngle * weight;
            }
        }

        for (uint pli = 0; pli < uniforms.pointLightCount; ++pli) {
            PointLight pl = pointLights[pli];
            float3 toPointLight = float3(pl.position) - hitPoint;
            float plDistSq = dot(toPointLight, toPointLight);
            float plDist = sqrt(plDistSq);
            float3 plWi = toPointLight / plDist;
            float plCosSurface = dot(facingNormal, plWi);
            if (plCosSurface > 0.0) {
                ray plShadowRay;
                plShadowRay.origin = hitPoint + facingNormal * 0.001f;
                plShadowRay.direction = plWi;
                plShadowRay.min_distance = 0.001f;
                plShadowRay.max_distance = plDist - 0.002f;
                intersection_result<instancing, triangle_data> plShadowResult =
                    isect.intersect(plShadowRay, accelStructure, functionTable);
                if (plShadowResult.type == intersection_type::none) {
                    float plTransmittance = exp(-uniforms.fogSigmaT * plDist);
                    float plSpot = spotLightFalloff(-plWi, float3(pl.direction), pl.cosOuterAngle, pl.cosInnerAngle);
                    radiance += throughput * albedo * velvetF(woWorld, plWi, facingNormal, mat.ior)
                                * float3(pl.emission) * plCosSurface * plSpot * plTransmittance / plDistSq;
                }
            }
        }

        for (uint dli = 0; dli < uniforms.directionalLightCount; ++dli) {
            DirectionalLight dl = directionalLights[dli];
            float3 dlWi = normalize(-float3(dl.direction));
            float dlCosSurface = dot(facingNormal, dlWi);
            if (dlCosSurface > 0.0) {
                ray dlShadowRay;
                dlShadowRay.origin = hitPoint + facingNormal * 0.001f;
                dlShadowRay.direction = dlWi;
                dlShadowRay.min_distance = 0.001f;
                dlShadowRay.max_distance = kDirectionalLightMaxDistance;
                intersection_result<instancing, triangle_data> dlShadowResult =
                    isect.intersect(dlShadowRay, accelStructure, functionTable);
                if (dlShadowResult.type == intersection_type::none) {
                    // Skip when there is no fog - rayBoxExitDistance()
                    // is only valid for an origin INSIDE the hardcoded
                    // room bounds (see that function.s own comment for
                    // why calling it unconditionally is unsafe).
                    float dlTransmittance = (uniforms.fogSigmaT > 0.0)
                        ? exp(-uniforms.fogSigmaT * rayBoxExitDistance(dlShadowRay.origin, dlWi, kRoomBoundsMin, kRoomBoundsMax))
                        : 1.0;
                    radiance += throughput * albedo * velvetF(woWorld, dlWi, facingNormal, mat.ior)
                                * float3(dl.emission) * dlCosSurface * dlTransmittance;
                }
            }
        }

        for (uint pji = 0; pji < uniforms.projectionLightCount; ++pji) {
            ProjectionLight pj = projectionLights[pji];
            float3 toProjLight = float3(pj.position) - hitPoint;
            float pjDistSq = dot(toProjLight, toProjLight);
            float pjDist = sqrt(pjDistSq);
            float3 pjWi = toProjLight / pjDist;
            float pjCosSurface = dot(facingNormal, pjWi);
            if (pjCosSurface > 0.0) {
                float3 pjRadiance = projectionLightRadiance(-pjWi, pj.forward, pj.right, pj.up,
                                                             pj.tanHalfFovX, pj.tanHalfFovY, pj.scale,
                                                             (pj.usePbrtTexture != 0u ? pbrtProjectionTexture : earthTexture), textureSampler);
                if (any(pjRadiance > float3(0.0))) {
                    ray pjShadowRay;
                    pjShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    pjShadowRay.direction = pjWi;
                    pjShadowRay.min_distance = 0.001f;
                    pjShadowRay.max_distance = pjDist - 0.002f;
                    intersection_result<instancing, triangle_data> pjShadowResult =
                        isect.intersect(pjShadowRay, accelStructure, functionTable);
                    if (pjShadowResult.type == intersection_type::none) {
                        float pjTransmittance = exp(-uniforms.fogSigmaT * pjDist);
                        radiance += throughput * albedo * velvetF(woWorld, pjWi, facingNormal, mat.ior)
                                    * pjRadiance * pjCosSurface * pjTransmittance / pjDistSq;
                    }
                }
            }
        }

        for (uint gli = 0; gli < uniforms.goniometricLightCount; ++gli) {
            GoniometricLight gl = goniometricLights[gli];
            float3 toGoniLight = float3(gl.position) - hitPoint;
            float glDistSq = dot(toGoniLight, toGoniLight);
            float glDist = sqrt(glDistSq);
            float3 glWi = toGoniLight / glDist;
            float glCosSurface = dot(facingNormal, glWi);
            if (glCosSurface > 0.0) {
                float3 glRadiance = goniometricLightRadiance(-glWi, gl.forward, gl.right, gl.up,
                                                              gl.emission, gl.scale,
                                                              (gl.usePbrtTexture != 0u ? pbrtGoniometricTexture : goniometricTexture), textureSampler);
                if (any(glRadiance > float3(0.0))) {
                    ray glShadowRay;
                    glShadowRay.origin = hitPoint + facingNormal * 0.001f;
                    glShadowRay.direction = glWi;
                    glShadowRay.min_distance = 0.001f;
                    glShadowRay.max_distance = glDist - 0.002f;
                    intersection_result<instancing, triangle_data> glShadowResult =
                        isect.intersect(glShadowRay, accelStructure, functionTable);
                    if (glShadowResult.type == intersection_type::none) {
                        float glTransmittance = exp(-uniforms.fogSigmaT * glDist);
                        radiance += throughput * albedo * velvetF(woWorld, glWi, facingNormal, mat.ior)
                                    * glRadiance * glCosSurface * glTransmittance / glDistSq;
                    }
                }
            }
        }

        // Gated on useEnvironmentMap too, not just envMapWidth - see
        // shadeConductor's own comment.
        if (envMapWidth > 0u && uniforms.useEnvironmentMap != 0u) {
            float envPdfSolidAngle;
            float3 envWi = sampleEnvironmentDirection(envMarginalCDF, envConditionalCDF,
                                                       int(envMapWidth), int(envMapHeight),
                                                       randFloat(rngState), randFloat(rngState), envPdfSolidAngle);
            float envCosSurface = dot(facingNormal, envWi);
            if (envCosSurface > 0.0 && envPdfSolidAngle > 1e-9) {
                ray envShadowRay;
                envShadowRay.origin = hitPoint + facingNormal * 0.001f;
                envShadowRay.direction = envWi;
                envShadowRay.min_distance = 0.001f;
                envShadowRay.max_distance = 1e5f;
                intersection_result<instancing, triangle_data> envShadowResult =
                    isect.intersect(envShadowRay, accelStructure, functionTable);
                if (envShadowResult.type == intersection_type::none) {
                    float2 envUV = equirectangularUV(envWi);
                    float3 envRadiance = earthTexture.sample(textureSampler, envUV).rgb;
                    float envWeight = (envPdfSolidAngle * envPdfSolidAngle)
                        / (envPdfSolidAngle * envPdfSolidAngle + uniformPdf * uniformPdf);
                    radiance += throughput * albedo * velvetF(woWorld, envWi, facingNormal, mat.ior)
                                * envRadiance * envCosSurface / envPdfSolidAngle * envWeight;
                }
            }
        }

        // Same NEE/MIS strategy, for a pbrt-loaded scene's own SEPARATE
        // image-based infinite light (section 96) - see shadeConductor's
        // own comment.
        if (pbrtEnvMapWidth > 0u) {
            float pbrtEnvPdfSolidAngle;
            float3 pbrtEnvWi = sampleEnvironmentDirection(pbrtEnvMarginalCDF, pbrtEnvConditionalCDF,
                                                           int(pbrtEnvMapWidth), int(pbrtEnvMapHeight),
                                                           randFloat(rngState), randFloat(rngState), pbrtEnvPdfSolidAngle);
            float pbrtEnvCosSurface = dot(facingNormal, pbrtEnvWi);
            if (pbrtEnvCosSurface > 0.0 && pbrtEnvPdfSolidAngle > 1e-9) {
                ray pbrtEnvShadowRay;
                pbrtEnvShadowRay.origin = hitPoint + facingNormal * 0.001f;
                pbrtEnvShadowRay.direction = pbrtEnvWi;
                pbrtEnvShadowRay.min_distance = 0.001f;
                pbrtEnvShadowRay.max_distance = 1e5f;
                intersection_result<instancing, triangle_data> pbrtEnvShadowResult =
                    isect.intersect(pbrtEnvShadowRay, accelStructure, functionTable);
                if (pbrtEnvShadowResult.type == intersection_type::none) {
                    float2 pbrtEnvUV = equirectangularUV(pbrtEnvWi);
                    float3 pbrtEnvRadianceSample = pbrtEnvTexture.sample(textureSampler, pbrtEnvUV).rgb;
                    float pbrtEnvWeight = (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle)
                        / (pbrtEnvPdfSolidAngle * pbrtEnvPdfSolidAngle + uniformPdf * uniformPdf);
                    radiance += throughput * albedo * velvetF(woWorld, pbrtEnvWi, facingNormal, mat.ior)
                                * pbrtEnvRadianceSample * pbrtEnvCosSurface / pbrtEnvPdfSolidAngle * pbrtEnvWeight;
                }
            }
        }
    }

    // Continuation ray: UNIFORM (not cosine-weighted) hemisphere
    // sampling, matching Cycles' own bsdf_ashikhmin_velvet_sample() -
    // pdf stays the constant 1/(2*pi) regardless of direction, so the MC
    // weight f*cosTheta/pdf is `albedo*velvetF(...)*cosTheta*2*pi`, NOT
    // Lambertian/Oren-Nayar's own pi-only factor (their own cosTheta/pi
    // pdf already cancels one power of cosTheta that this uniform pdf
    // does not).
    float3 newDir = sampleUniformHemisphere(facingNormal, rngState);
    rayDir = newDir;
    rayOrigin = hitPoint + facingNormal * 0.001f;
    float newDirCos = max(dot(facingNormal, newDir), 0.0001);
    throughput *= albedo * velvetF(woWorld, newDir, facingNormal, mat.ior) * newDirCos * (2.0 * M_PI_F);
    bsdfPdf = uniformPdf;
    specularBounce = false;
    return true;
}

// Schlick's Fresnel approximation, F0 + (1-F0)*(1-cosTheta)^5 - the SAME
// approximate formula src/shared/bxdfs_principled.h's own
// `schlick_fresnel()` uses (a deliberate simplification for this
// artist-friendly BSDF, unlike materialType 2/4/5's own EXACT
// frDielectric()/frComplex() - matching the reference exactly here
// means using its own approximation, not "upgrading" it).
inline float schlickFresnelPrincipled(float cosTheta, float F0) {
    float c = 1.0 - clamp(cosTheta, 0.0, 1.0);
    float c2 = c * c;
    float c5 = c2 * c2 * c;
    return F0 + (1.0 - F0) * c5;
}

// GGX specular BRDF value, local frame (z=normal) - same D*G/(4*cosO*cosI)
// shape ggxD()/ggxG() already compute for materialType 4/9's own
// shadeConductor(), just without a Fresnel term folded in (Principled's
// own 3-lobe blend applies Fresnel separately per lobe/channel).
inline float principledGgxBrdf(float3 wo, float3 wi, float alpha) {
    if (wo.z <= 0.0 || wi.z <= 0.0) return 0.0;
    float3 h = wo + wi;
    float hlen = length(h);
    if (hlen < 1e-8) return 0.0;
    h /= hlen;
    float D = ggxD(h, alpha, alpha);
    float G = ggxG(wo, wi, alpha, alpha);
    return D * G / max(4.0 * wo.z * wi.z, 1e-8);
}

// GGX VNDF sampling pdf, local frame - pbrt-v4's own
// `D(wm)*G1(wo)*AbsDot(wo,wm)/AbsCosTheta(wo) / (4*dot(wo,wm))`, matching
// `PrincipledBxDF::ggx_pdf()` (src/shared/bxdfs_principled.h) exactly.
inline float principledGgxPdf(float3 wo, float3 wi, float alpha) {
    if (wo.z <= 0.0 || wi.z <= 0.0) return 0.0;
    float3 h = wo + wi;
    float hlen = length(h);
    if (hlen < 1e-8) return 0.0;
    h /= hlen;
    float dotWoH = dot(wo, h);
    if (dotWoH <= 0.0) return 0.0;
    float D = ggxD(h, alpha, alpha);
    float G1 = ggxG1(wo, alpha, alpha);
    float pdfWm = D * G1 * dotWoH / max(wo.z, 1e-6);
    return pdfWm / max(4.0 * dotWoH, 1e-8);
}

// materialType 24 (B10, Principled Showcase) - pbrt-v4/Disney's own
// artist-friendly 3-lobe BSDF (diffuse + specular-dielectric-or-metal +
// clearcoat), a direct port of `PrincipledBxDF<T>::sample()`
// (src/shared/bxdfs_principled.h), the SAME shared header both CPU
// (`principled_material.h`) and OptiX (`sample_principled_material()`,
// `optix_device_helpers.h`) already build from directly. Deliberately
// has NO NEE/MIS at all - matches CPU's own `principled::scatter()`
// (`srec.skip_pdf = true`, no separate `scattering_pdf()`-driven light
// sampling loop) and OptiX's own identical `is_specular = true` choice
// for this exact material (`optix_intersection_sphere.h`'s own comment:
// "no NEE/MIS, res.r/g/b already divides by the sample pdf") - the
// BSDF's own `sample()` returns a complete `f*cos/pdf` weight in one
// call, the same "combined sample+eval, no separate NEE path" shape
// this loader's own `shadeDielectric()`/`shadeMirror()` already use for
// other delta-like materials, just with 3 stochastically-chosen lobes
// instead of 1. Field reuse (matching OptiX's own exact convention,
// `optix_device_helpers.h`'s `sample_principled_material()` comment):
// `mat.color`=base color, `mat.ior`=ior, `mat.roughness`=perceptual
// roughness, `mat.conductorEta.x`=metallic, `mat.conductorEta.y`=
// clearcoat, `mat.conductorEta.z`=clearcoat_roughness.
inline bool shadePrincipled(TriangleMaterial mat, float3 hitPoint, float3 facingNormal,
                            thread float3& rayDir, thread float3& rayOrigin,
                            thread float3& throughput, thread bool& specularBounce, thread uint& rngState) {
    float metallic = mat.conductorEta.x;
    float clearcoat = mat.conductorEta.y;
    float clearcoatRoughness = mat.conductorEta.z;
    // TrowbridgeReitz::RoughnessToAlpha(r) = sqrt(r) - the REAL pbrt-v4
    // formula, not materialType 4/9's own "square it in the shader"
    // convention (this is a brand-new shading function with no old
    // convention to reconcile with, same reasoning as materialType 19's
    // own comment, metal_poc.mm).
    float alpha = max(sqrt(max(mat.roughness, 0.0)), 0.0009);
    float alphaCC = max(sqrt(max(clearcoatRoughness, 0.0)), 0.0009);

    float3 tangent, bitangent;
    buildOnb(facingNormal, tangent, bitangent);
    // wi = the ray's OWN direction of travel (INTO the surface) - matches
    // CPU's own `in_dir = unit_vector(r_in.direction())` passed straight
    // into `bxdf.sample()` with NO negation; `wo = -wi` is derived
    // internally, exactly mirrored here.
    float3 wiWorld = normalize(rayDir);
    float3 wiLocal = float3(dot(wiWorld, tangent), dot(wiWorld, bitangent), dot(wiWorld, facingNormal));
    float3 woLocal = -wiLocal;
    if (woLocal.z <= 0.0) return false;

    float F0d = pow((mat.ior - 1.0) / (mat.ior + 1.0), 2.0);
    float Fspec = schlickFresnelPrincipled(woLocal.z, F0d);
    float wDiff = (1.0 - metallic) * (1.0 - Fspec);
    float wSpec = 1.0;
    float wCoat = clearcoat * 0.25;
    float wTotal = wDiff + wSpec + wCoat;
    if (wTotal < 1e-8) return false;
    float invW = 1.0 / wTotal;
    float pDiff = wDiff * invW;
    float pSpec = wSpec * invW;
    float pCoat = wCoat * invW;

    float u1 = randFloat(rngState);
    float3 woOutLocal;
    if (u1 < pDiff) {
        woOutLocal = cosineSampleHemisphere(float3(0.0, 0.0, 1.0), rngState);
    } else if (u1 < pDiff + pSpec) {
        float3 wm = sampleGGXVNDF(woLocal, alpha, alpha, rngState);
        float d = dot(woLocal, wm);
        woOutLocal = 2.0 * d * wm - woLocal;
    } else {
        float3 wm = sampleGGXVNDF(woLocal, alphaCC, alphaCC, rngState);
        float d = dot(woLocal, wm);
        woOutLocal = 2.0 * d * wm - woLocal;
    }
    if (woOutLocal.z <= 0.0) return false;

    float cosWiL = woOutLocal.z;
    float3 h = woLocal + woOutLocal;
    float hlen = length(h);
    float cosWm = (hlen > 1e-8) ? dot(woLocal, h / hlen) : woLocal.z;

    float FwiDiff = schlickFresnelPrincipled(cosWiL, F0d);
    float3 diffCol = float3(mat.color) * (1.0 / M_PI_F) * (1.0 - metallic) * (1.0 - FwiDiff);

    float specVal = principledGgxBrdf(woLocal, woOutLocal, alpha);
    float FspecWm = schlickFresnelPrincipled(cosWm, F0d);
    float3 FmetWm = float3(schlickFresnelPrincipled(cosWm, mat.color.x),
                            schlickFresnelPrincipled(cosWm, mat.color.y),
                            schlickFresnelPrincipled(cosWm, mat.color.z));
    float3 Fmix = (1.0 - metallic) * FspecWm + metallic * FmetWm;
    float3 specCol = Fmix * specVal;

    float ccF0 = 0.04;
    float Fcc = schlickFresnelPrincipled(cosWm, ccF0);
    float ccVal = principledGgxBrdf(woLocal, woOutLocal, alphaCC);
    float ccCol = clearcoat * 0.25 * Fcc * ccVal;

    float3 totalCol = diffCol + specCol + float3(ccCol);

    float pdfDiff = pDiff * cosWiL / M_PI_F;
    float pdfSpec = pSpec * principledGgxPdf(woLocal, woOutLocal, alpha);
    float pdfCoat = pCoat * principledGgxPdf(woLocal, woOutLocal, alphaCC);
    float pdf = pdfDiff + pdfSpec + pdfCoat;
    if (pdf < 1e-12) return false;

    float3 weight = totalCol * cosWiL / pdf;

    float3 newDirWorld = normalize(woOutLocal.x * tangent + woOutLocal.y * bitangent + woOutLocal.z * facingNormal);
    rayDir = newDirWorld;
    rayOrigin = hitPoint + facingNormal * 0.001f;
    throughput *= weight;
    specularBounce = true;
    return true;
}

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
    uint2 tid [[thread_position_in_grid]])
{
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
    uint actualSamples = uniforms.samplesPerPixel;

    for (uint s = 0; s < uniforms.samplesPerPixel; ++s) {
        // Jittered pixel sample - the multi-sample loop's own antialiasing,
        // not a separate feature: without the jitter every sample would
        // retrace the exact same primary ray.
        float2 jitter = float2(randFloat(rngState), randFloat(rngState));
        float2 pixelNDC = (float2(tid) + jitter) / float2(uniforms.width, uniforms.height);
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
        float shutterT = randFloat(rngState);
        float3 rayOrigin, rayDir;
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
            // Spherical (360-degree equirectangular panorama): mirrors
            // pbrt-v4 SphericalCamera::GenerateRay()'s own EquiRectangular
            // mapping exactly (Uniforms::cameraSpherical's own comment) -
            // `u`/`v` are the SAME `pixelNDC.x/y` every other mode
            // derives `screen` from, used directly here (raw [0,1],
            // BEFORE the `*2-1`/y-flip/aspect/tanHalfFov transform those
            // other modes need - this mode has no screen window or FOV
            // at all, it captures the full sphere around one point).
            float theta = M_PI_F * pixelNDC.y;
            float phi = 2.0 * M_PI_F * pixelNDC.x;
            float sinTheta = sin(theta), cosTheta = cos(theta);
            rayOrigin = float3(uniforms.cameraPos) + shutterT * float3(uniforms.cameraVelocity);
            // Leading MINUS on the `cameraRight` term - the SAME sign
            // correction `cameraOrthographic`'s own branch above needs,
            // for the identical reason (CPU's alt-camera path's own
            // `right` is this loader's `cameraRight` negated).
            rayDir = normalize(-sinTheta * cos(phi) * float3(uniforms.cameraRight)
                                + cosTheta * float3(uniforms.cameraUp)
                                + sinTheta * sin(phi) * float3(uniforms.cameraForward));
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

        float3 throughput = float3(cameraWeight);
        float3 radiance = float3(0.0);
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
        bool specularBounce = true;
        float bsdfPdf = 0.0;
        // Recursive-backend dispersion state (materialType 22, B23/B24) -
        // kRgbChannelUnset means "no dispersive hit yet, this sample
        // stays full RGB". See shadeDispersiveDielectric()'s own
        // declaration comment for the full "stochastic channel
        // selection" rationale, ported from OptiX's own identical
        // per-path convention (gpu/optix/optix_raygen.h/
        // optix_device_helpers.h).
        uint rgbChannel = kRgbChannelUnset;

        for (uint depth = 0; depth < uniforms.maxDepth; ++depth) {
            ray r;
            r.origin = rayOrigin;
            r.direction = rayDir;
            r.min_distance = 0.001f;
            r.max_distance = 1e6f;

            intersection_result<instancing, triangle_data> result =
                isect.intersect(r, accelStructure, functionTable);

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
            bool scatteredInMedium = false;
            if (uniforms.fogSigmaT > 0.0 && result.type != intersection_type::none) {
                float surfaceDist = result.distance;
                float u = randFloat(rngState);
                float t = -log(max(1.0 - u, 1e-6)) / uniforms.fogSigmaT;
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
                            isect.intersect(shadowRay, accelStructure, functionTable);
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
                            float phaseValue = henyeyGreensteinPhase(dot(wo, wi), uniforms.fogAsymmetryG);
                            float weight = (pdfSolidAngle * pdfSolidAngle)
                                / (pdfSolidAngle * pdfSolidAngle + phaseValue * phaseValue);
                            float transmittance = exp(-uniforms.fogSigmaT * dist);
                            radiance += throughput * phaseValue * ls.emission * transmittance
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
                            isect.intersect(plShadowRay, accelStructure, functionTable);
                        if (plShadowResult.type == intersection_type::none) {
                            float plPhaseValue = henyeyGreensteinPhase(dot(wo, plWi), uniforms.fogAsymmetryG);
                            float plTransmittance = exp(-uniforms.fogSigmaT * plDist);
                            float plSpot = spotLightFalloff(-plWi, float3(pl.direction), pl.cosOuterAngle, pl.cosInnerAngle);
                            radiance += throughput * plPhaseValue * float3(pl.emission) * plSpot * plTransmittance / plDistSq;
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
                            isect.intersect(dlShadowRay, accelStructure, functionTable);
                        if (dlShadowResult.type == intersection_type::none) {
                            float dlPhaseValue = henyeyGreensteinPhase(dot(wo, dlWi), uniforms.fogAsymmetryG);
                            float dlExitDist = rayBoxExitDistance(scatterPoint, dlWi, kRoomBoundsMin, kRoomBoundsMax);
                            float dlTransmittance = exp(-uniforms.fogSigmaT * dlExitDist);
                            radiance += throughput * dlPhaseValue * float3(dl.emission) * dlTransmittance;
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
                                isect.intersect(pjShadowRay, accelStructure, functionTable);
                            if (pjShadowResult.type == intersection_type::none) {
                                float pjPhaseValue = henyeyGreensteinPhase(dot(wo, pjWi), uniforms.fogAsymmetryG);
                                float pjTransmittance = exp(-uniforms.fogSigmaT * pjDist);
                                radiance += throughput * pjPhaseValue * pjRadiance * pjTransmittance / pjDistSq;
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
                                isect.intersect(glShadowRay, accelStructure, functionTable);
                            if (glShadowResult.type == intersection_type::none) {
                                float glPhaseValue = henyeyGreensteinPhase(dot(wo, glWi), uniforms.fogAsymmetryG);
                                float glTransmittance = exp(-uniforms.fogSigmaT * glDist);
                                radiance += throughput * glPhaseValue * glRadiance * glTransmittance / glDistSq;
                            }
                        }
                    }

                    float3 newDir = sampleHenyeyGreenstein(wo, uniforms.fogAsymmetryG, rngState);
                    rayDir = newDir;
                    rayOrigin = scatterPoint;
                    throughput *= float3(uniforms.fogAlbedo);
                    bsdfPdf = henyeyGreensteinPhase(dot(wo, newDir), uniforms.fogAsymmetryG);
                    specularBounce = false;
                }
            }

            if (!scatteredInMedium) {
            if (result.type == intersection_type::none) {
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
                } else {
                    float skyT = 0.5 * (rayDir.y + 1.0);
                    radiance += throughput * mix(skyBottom, skyTop, skyT);
                }
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
            // Both spheres AND the disk report intersection_type::
            // bounding_box (neither is a hardware-native triangle) - they
            // only stop being ambiguous once `geometry_id` is checked too:
            // sphereAS's own geometryDescriptors array has the spheres'
            // bounding-box geometry at index 0 and the disk's at index 1
            // (metal_poc.mm's own sphereAccelDesc.geometryDescriptors),
            // and geometry_id reports exactly that array index for a
            // bounding-box hit - the first time this POC's shading loop
            // has needed geometry_id at all (every earlier custom
            // primitive was the ONLY bounding-box geometry in its AS, so
            // "bounding_box == sphere" was unambiguous until now).
            bool isBoundingBox = (result.type == intersection_type::bounding_box);
            bool isDisk = isBoundingBox && (result.geometry_id == 1u);
            bool isSphere = isBoundingBox && !isDisk;
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
            bool isSuzanneInstance = !isSphere && !isDisk && (result.instance_id >= 2u);
            float3 normal;
            TriangleMaterial mat;
            if (isSphere) {
                SphereData sphere = spheres[primId];
                normal = normalize(hitPoint - float3(sphere.center));
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
            } else if (isSuzanneInstance) {
                // Object-space normal (Suzanne's own per-vertex data,
                // just like the non-instanced case below) transformed
                // into world space by THIS hit's own instance transform -
                // the one piece of shading math instancing actually adds
                // over the room/Spot geometry's own single-identity-
                // instance path.
                float3 objectNormal = shadingNormalFor(primId, result.triangle_barycentric_coord, suzanneNormals);
                normal = transformNormalByInstance(objectNormal, instanceTransforms[result.instance_id]);
                mat = suzanneMaterials[primId];
            } else {
                normal = shadingNormalFor(primId, result.triangle_barycentric_coord, normals);
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
                    float distSq = result.distance * result.distance;
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
                    float cosLight = max(abs(dot(float3(light.normal), -rayDir)), 0.0001);
                    float pdfLight = (distSq / (light.area * cosLight)) * light.pmf;
                    float weight = (bsdfPdf * bsdfPdf) / (bsdfPdf * bsdfPdf + pdfLight * pdfLight);
                    radiance += throughput * hitEmission * weight;
                }
            }

            if (mat.materialType == 2u) {
                if (!shadeDielectric(mat, hitPoint, normal, facingNormal, frontFace, result.distance,
                                      rayDir, rayOrigin, throughput, specularBounce, rngState)) break;
            } else if (mat.materialType == 22u) {
                if (!shadeDispersiveDielectric(mat, hitPoint, normal, facingNormal, frontFace, result.distance,
                                      rayDir, rayOrigin, throughput, specularBounce, rngState, rgbChannel)) break;
            } else if (mat.materialType == 5u) {
                if (!shadeRoughDielectric(mat, hitPoint, normal, facingNormal, frontFace, result.distance,
                                           rayDir, rayOrigin, throughput, specularBounce, rngState)) break;
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
                if (!shadeThinDielectric(mat, hitPoint, normal, facingNormal,
                                          rayDir, rayOrigin, throughput, specularBounce, rngState)) break;
            } else if (mat.materialType == 8u) {
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
            } // !scatteredInMedium

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
        }

        // Firefly clamp - see kFireflyClampLuminance's own comment.
        // Applied once per SAMPLE, here, not per NEE contribution inside
        // the bounce loop above - simpler (one clamp site, not scattered
        // across every light-sampling branch) and still catches the same
        // outliers, since an extreme single-bounce contribution dominates
        // this sample's own total `radiance` regardless of which branch
        // produced it.
        float sampleMax = max(radiance.x, max(radiance.y, radiance.z));
        if (sampleMax > kFireflyClampLuminance) {
            radiance *= kFireflyClampLuminance / sampleMax;
        }
        accumColor += radiance;

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
                    actualSamples = s + 1;
                    break;
                }
            }
        }
    }

    accumColor /= float(actualSamples);
    outTexture.write(float4(accumColor, 1.0), tid);
}

// ---------------------------------------------------------------------------
// Device-side unit-test kernels - see gpu/metal/metal_poc_shader_tests.mm's
// own comment for the full "why" (closing the device-side half of the
// testing gap docs/METAL_GPU_FEASIBILITY.md section 59 left open: every
// pure GPU-independent host function got a real CTest case there, but
// everything that only ever runs on the GPU - frDielectric(), the GGX
// microfacet math, checkerColor(), spotLightFalloff(),
// fresnelSchlickConductor(), henyeyGreensteinPhase(), the real
// projectionLightRadiance(), and sampleAreaLight()'s own alias-table
// lookup - had none at all, only the full-scene smoke test's own "not
// flat/black" check).
//
// Each kernel below calls exactly one already-existing function from this
// SAME file, with no reimplementation - metal_poc_shader_tests.mm
// dispatches each with known inputs and checks the outputs against
// independently-derived reference values, the same way
// metal_poc_math_tests.cpp already does for this file's host-side
// counterpart. Purely additive: nothing above this point is touched, and
// primaryRayKernel's own [[buffer(N)]] indices are irrelevant here - each
// test kernel is its own separate entry point with its own independent
// buffer(0..N) argument table, the same "intersection functions have
// their own independent argument table" reasoning
// sphereIntersectionFunction's own comment already established.
// ---------------------------------------------------------------------------

kernel void test_frDielectric(
    device const float2* inputs [[buffer(0)]],   // (cosThetaI, eta)
    device float* outputs [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = frDielectric(inputs[tid].x, inputs[tid].y);
}

kernel void test_ggxD(
    device const float* alphas [[buffer(0)]],
    device float* outputs [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    // Half-vector aligned exactly with the shading normal (hLocal ==
    // (0,0,1)) - the one case ggxD() has a known, exact closed form for:
    // D reduces to 1/(pi*alpha^2) regardless of alpha, since hr collapses
    // to (0,0,1) and lenSq becomes exactly 1.
    float alpha = alphas[tid];
    outputs[tid] = ggxD(float3(0.0, 0.0, 1.0), alpha, alpha);
}

kernel void test_ggxG1_smoothLimit(
    device const float3* directions [[buffer(0)]],
    device float* outputs [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    // A near-zero (not exactly zero, to avoid a literal divide-by-zero
    // in ggxLambda's own sqrAlphaTanN term) roughness must make G1
    // approach 1 for any direction with a positive z - a perfectly smooth
    // microfacet distribution has no masking/shadowing at all.
    const float alpha = 1e-4;
    outputs[tid] = ggxG1(directions[tid], alpha, alpha);
}

kernel void test_checkerColor(
    device const float2* uvs [[buffer(0)]],
    device float3* outputs [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = checkerColor(uvs[tid], /*scale=*/2.0, float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0));
}

kernel void test_spotLightFalloff(
    device const float3* wiFromLights [[buffer(0)]],
    device const float3* directions [[buffer(1)]],
    device const float2* angles [[buffer(2)]],   // (cosOuterAngle, cosInnerAngle)
    device float* outputs [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = spotLightFalloff(wiFromLights[tid], directions[tid], angles[tid].x, angles[tid].y);
}

kernel void test_fresnelSchlickConductor(
    device const float* cosThetas [[buffer(0)]],
    device const float3* f0s [[buffer(1)]],
    device float3* outputs [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = fresnelSchlickConductor(cosThetas[tid], f0s[tid]);
}

kernel void test_frComplexRGB(
    device const float* cosThetas [[buffer(0)]],
    device const float3* etas [[buffer(1)]],
    device const float3* ks [[buffer(2)]],
    device float3* outputs [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = frComplexRGB(cosThetas[tid], etas[tid], ks[tid]);
}

kernel void test_buildAnisotropicOnb(
    device const float3* normals [[buffer(0)]],
    device float3* tangentOutputs [[buffer(1)]],
    device float3* bitangentOutputs [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    float3 t, b;
    buildAnisotropicOnb(normals[tid], t, b);
    tangentOutputs[tid] = t;
    bitangentOutputs[tid] = b;
}

kernel void test_sampleEnvironmentDirection(
    device const float* marginalCDF [[buffer(0)]],
    device const float* conditionalCDF [[buffer(1)]],
    device const int2* dims [[buffer(2)]],
    device const float2* uvSamples [[buffer(3)]],
    device float3* dirOutputs [[buffer(4)]],
    device float* pdfOutputs [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    float pdf;
    float3 dir = sampleEnvironmentDirection(marginalCDF, conditionalCDF, dims[0].x, dims[0].y,
                                             uvSamples[tid].x, uvSamples[tid].y, pdf);
    dirOutputs[tid] = dir;
    pdfOutputs[tid] = pdf;
}

kernel void test_pdfEnvironmentDirection(
    device const float* marginalCDF [[buffer(0)]],
    device const float* conditionalCDF [[buffer(1)]],
    device const int2* dims [[buffer(2)]],
    device const float3* dirs [[buffer(3)]],
    device float* pdfOutputs [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    pdfOutputs[tid] = pdfEnvironmentDirection(marginalCDF, conditionalCDF, dims[0].x, dims[0].y, dirs[tid]);
}

kernel void test_sampleGGXEnergyTableDevice(
    device const float* E [[buffer(0)]],
    device const uint2* dims [[buffer(1)]],
    device const float2* roughnessMuPairs [[buffer(2)]],
    device float* outputs [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = sampleGGXEnergyTableDevice(E, dims[0].x, dims[0].y,
                                               roughnessMuPairs[tid].x, roughnessMuPairs[tid].y);
}

kernel void test_orenNayarF(
    device const float3* wos [[buffer(0)]],
    device const float3* wis [[buffer(1)]],
    device const float3* ns [[buffer(2)]],
    device const float* sigmas [[buffer(3)]],
    device float* outputs [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = orenNayarF(wos[tid], wis[tid], ns[tid], sigmas[tid]);
}

kernel void test_velvetF(
    device const float3* wos [[buffer(0)]],
    device const float3* wis [[buffer(1)]],
    device const float3* ns [[buffer(2)]],
    device const float* sigmas [[buffer(3)]],
    device float* outputs [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = velvetF(wos[tid], wis[tid], ns[tid], sigmas[tid]);
}

kernel void test_henyeyGreensteinPhase(
    device const float2* inputs [[buffer(0)]],   // (cosTheta, g)
    device float* outputs [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = henyeyGreensteinPhase(inputs[tid].x, inputs[tid].y);
}

// Mirrors ProjectionLight's own field layout exactly (see that struct's
// own comment) - a plain input-data buffer for this test kernel, not a
// re-declaration of the real struct.
kernel void test_projectionLightRadiance(
    device const float3* wiFromLights [[buffer(0)]],
    constant ProjectionLight& light [[buffer(1)]],
    device float3* outputs [[buffer(2)]],
    texture2d<float, access::sample> testImage [[texture(0)]],
    uint tid [[thread_position_in_grid]])
{
    // Nearest, clamp-to-edge sampling - deliberately DIFFERENT from
    // primaryRayKernel's own production sampler (bilinear, repeat), since
    // this test cares about projectionLightRadiance()'s own frustum/UV
    // MATH being right (an exact, predictable texel lookup), not
    // re-verifying Metal's own texture-sampling hardware, which isn't
    // this function's responsibility to get right or wrong.
    constexpr sampler nearestSampler(coord::normalized, address::clamp_to_edge, filter::nearest);
    outputs[tid] = projectionLightRadiance(wiFromLights[tid], light.forward, light.right, light.up,
                                            light.tanHalfFovX, light.tanHalfFovY, light.scale,
                                            testImage, nearestSampler);
}

// Dispatches sampleAreaLight() itself, many times, against a small
// host-supplied light list already run through buildPowerLightSampler()
// (metal_poc_host_math.h) - metal_poc_shader_tests.mm checks that the
// REAL device-side alias-table lookup this kernel calls empirically
// reproduces each light's own `pmf`, closing the exact gap
// docs/METAL_GPU_FEASIBILITY.md section 59 called out by name ("only
// this PR's own host-side test MIRROR of the alias-table lookup is
// directly regression-tested, not the actual device-side one it's
// mirroring").
kernel void test_sampleAreaLight_pmf(
    device const AreaLight* lights [[buffer(0)]],
    constant uint& lightCount [[buffer(1)]],
    constant uint& seed [[buffer(2)]],
    device float* outPmfs [[buffer(3)]],
    texture2d<float, access::sample> dummyTexture [[texture(0)]],
    uint tid [[thread_position_in_grid]])
{
    constexpr sampler nearestSampler(coord::normalized, address::clamp_to_edge, filter::nearest);
    uint rngState = tid * 9781u + seed * 26699u + 1u;
    LightSample ls = sampleAreaLight(lights, lightCount, rngState, dummyTexture, nearestSampler);
    outPmfs[tid] = ls.pmf;
}

// Added alongside PR #57's own GoniometricLight/equalAreaSphereToSquare()
// - neither had any device-side test coverage at all until now, the
// exact gap this whole "Device-side unit-test kernels" section exists to
// close for every OTHER function already here.
kernel void test_equalAreaSphereToSquare(
    device const float3* directions [[buffer(0)]],
    device float2* outputs [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    outputs[tid] = equalAreaSphereToSquare(directions[tid]);
}

kernel void test_goniometricLightRadiance(
    device const float3* wiFromLights [[buffer(0)]],
    constant GoniometricLight& light [[buffer(1)]],
    device float3* outputs [[buffer(2)]],
    texture2d<float, access::sample> testImage [[texture(0)]],
    uint tid [[thread_position_in_grid]])
{
    constexpr sampler nearestSampler(coord::normalized, address::clamp_to_edge, filter::nearest);
    outputs[tid] = goniometricLightRadiance(wiFromLights[tid], light.forward, light.right, light.up,
                                             light.emission, light.scale, testImage, nearestSampler);
}
