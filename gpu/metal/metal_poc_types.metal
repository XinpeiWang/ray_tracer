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
    // Which mapping `cameraSpherical`'s own ray generation uses - 0
    // (every scene before D11, and D7/D8's own hand-authored ports)
    // means EquiRectangular (this field's own default, matching
    // cameraSpherical's own comment above exactly, unread when
    // cameraSpherical==0). Nonzero means EqualArea instead - pbrt-v4
    // Camera "spherical" "string mapping" ["equalarea"], real pbrt
    // FILE scenes only (D11, section 159) - no hand-authored scene
    // requests it. See equalAreaSquareToSphere()'s own comment
    // (metal_poc_types.metal) for the mapping itself.
    uint sphericalMappingEqualArea;
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

// The inverse of equalAreaSphereToSquare() just above: (u,v) in [0,1]^2
// -> unit sphere direction, equal-area - needed once a caller SAMPLES a
// direction from (u,v) rather than only looking one up (the reason this
// inverse wasn't ported alongside the forward mapping originally - see
// that function's own comment). The spherical camera's own EqualArea
// mapping option (D11, pbrt-v4 Camera "spherical" "string mapping"
// ["equalarea"], section 159) is the first caller. Direct port of
// src/shared/sampling_sphere.h's EqualAreaSquareToSphere / gpu/optix/
// optix_device_helpers_lighting.h's own already-shipped CUDA mirror
// (dev_equal_area_square_to_sphere) - same algorithm, this file's own
// established `float` (not CUDA's `double`) precision, matching
// equalAreaSphereToSquare() immediately above.
inline float3 equalAreaSquareToSphere(float u, float v) {
    float uu = 2.0 * u - 1.0, vv = 2.0 * v - 1.0;
    float up = abs(uu), vp = abs(vv);
    float signedDist = 1.0 - (up + vp);
    float d = abs(signedDist);
    float r = 1.0 - d;
    float phi = (r == 0.0 ? 1.0 : (vp - up) / r + 1.0) * (M_PI_F / 4.0);
    float wz = copysign(1.0 - r * r, signedDist);
    float cosPhi = cos(phi);
    float sinPhi = sin(phi);
    float xyR = r * sqrt(max(0.0, 2.0 - r * r));
    float wx = copysign(cosPhi * xyR, uu);
    float wy = copysign(sinPhi * xyR, vv);
    return float3(wx, wy, wz);
}

// Mirrors an (u,v) outside [0,1]^2 back onto the equal-area square -
// needed wherever the caller's own (u,v) can legitimately land slightly
// outside that range (equalAreaSquareToSphere()'s own caller draws a
// per-sample jittered pixel coordinate, not an image index a plain
// clamp would already handle correctly). Direct port of
// src/shared/sampling_sphere.h's WrapEqualAreaSquare / gpu/optix/
// optix_device_helpers_lighting.h's own dev_wrap_equal_area_square.
inline void wrapEqualAreaSquare(thread float& u, thread float& v) {
    if (u < 0.0) { u = -u; v = 1.0 - v; }
    else if (u > 1.0) { u = 2.0 - u; v = 1.0 - v; }
    if (v < 0.0) { u = 1.0 - u; v = -v; }
    else if (v > 1.0) { u = 1.0 - u; v = 2.0 - v; }
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
