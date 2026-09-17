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
// Known simplification: unlike PointLight's shadow ray (a known finite
// distance, so `exp(-fogSigmaT * dist)` is a real Beer-Lambert
// attenuation), a directional light's shadow ray has no well-defined
// finite path length through the fog before it exits the room's open
// front - so this light's own NEE contribution does NOT attenuate
// through fog at all (unconditionally full contribution when
// unoccluded), rather than picking an arbitrary sentinel distance that
// would silently misrepresent the fog's real optical depth. Skipped
// deliberately, not an oversight - the same "don't fake it" judgement
// call step 24's own point light doc applied to GGX energy compensation.
struct DirectionalLight {
    packed_float3 direction;
    packed_float3 emission;
};

// A shadow ray toward a directional light has no real target distance
// (the light is at infinity) - this is just "farther than anything in
// this room's own [-1,1]^3 extent could be," so an unoccluded shadow ray
// reads as having genuinely exited the scene rather than being clipped
// short of a real occluder.
constant float kDirectionalLightMaxDistance = 10.0f;

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
// `ior` is meaningful for materialType == 2 and 5 (refraction index) and
// reused, differently, for materialType == 4 (perceptual roughness in
// [0,1], squared into the GGX alpha parameter below) - materialType 4
// and {2,5} never coexist on one primitive, so sharing the slot there
// avoids a second otherwise-almost-always-zero field; materialType 5
// needs both ior AND roughness at once though, hence `roughness` getting
// its own field instead of also trying to overload `ior`. Both fields
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
    // sets it. 0 for every other material type.
    float roughness;
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

// Schlick's approximation - the standard cheap stand-in for the full
// Fresnel dielectric reflectance formula, same one pbrt-v4 and this
// project's own CPU dielectric material use for the reflect-vs-refract
// decision.
inline float schlickReflectance(float cosine, float refractionRatio) {
    float r0 = (1.0 - refractionRatio) / (1.0 + refractionRatio);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
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
// scalar IOR the way schlickReflectance()'s dielectric version is - a
// metal's complex refractive index (real eta + imaginary k, wavelength-
// dependent) is what actually produces that colour, and approximating the
// whole curve from its F0 value is the same "Schlick, not the full
// Fresnel equations" trade this POC's dielectric material already makes.
inline float3 fresnelSchlickConductor(float cosTheta, float3 F0) {
    float t = pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    return F0 + (float3(1.0) - F0) * t;
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
inline void buildAnisotropicOnb(float3 normal, thread float3& tangent, thread float3& bitangent) {
    float3 refDir = float3(0.0, 1.0, 0.0);
    if (abs(dot(refDir, normal)) > 0.999) {
        refDir = float3(1.0, 0.0, 0.0);
    }
    bitangent = normalize(cross(normal, refDir));
    tangent = cross(bitangent, normal);
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
// Picks one light uniformly at random from `lights` and samples a
// uniform point on its parallelogram - the "pick a light, then a point on
// it" step every NEE call site (Lambertian and GGX conductor both) shares
// verbatim; only what happens with the sampled point differs per BSDF.
// The 1/lightCount light-PICKING pdf is folded in by each call site
// itself (alongside its own area-to-solid-angle conversion), not returned
// here, since it's a plain scalar constant for a given uniforms.lightCount
// and every caller already needs to multiply it into an existing pdf
// expression rather than use it standalone.
struct LightSample {
    float3 point;
    float3 normal;
    float3 emission;
    float area;
};

inline LightSample sampleAreaLight(device const AreaLight* lights, uint lightCount, thread uint& rngState) {
    // max(lightCount, 1u) guards the `- 1` below from underflowing (uint
    // wraps to 0xFFFFFFFF, not -1) if this were ever called on a 0-light
    // scene - not reachable with this POC's own hardcoded 2-light scene,
    // but every NEE call site calls this unconditionally with no count
    // check of its own, so this needs to be safe on its own terms.
    uint idx = min(uint(randFloat(rngState) * float(lightCount)), max(lightCount, 1u) - 1);
    AreaLight light = lights[idx];
    float3 edgeU = float3(light.edgeU);
    float3 edgeV = float3(light.edgeV);
    float2 u = float2(randFloat(rngState), randFloat(rngState));
    LightSample result;
    result.point = float3(light.center) - 0.5 * edgeU - 0.5 * edgeV + u.x * edgeU + u.y * edgeV;
    result.normal = float3(light.normal);
    result.emission = float3(light.emission);
    result.area = light.area;
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

kernel void primaryRayKernel(
    texture2d<float, access::write> outTexture [[texture(0)]],
    texture2d<float, access::sample> earthTexture [[texture(1)]],
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
        float3 rayOrigin = float3(uniforms.cameraPos) + shutterT * float3(uniforms.cameraVelocity);
        float3 rayDir = normalize(float3(uniforms.cameraForward)
                                   + screen.x * float3(uniforms.cameraRight)
                                   + screen.y * float3(uniforms.cameraUp));

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
        if (uniforms.lensRadius > 0.0) {
            float2 lensSample = uniforms.lensRadius * sampleUnitDisk(rngState);
            float3 focusPoint = rayOrigin + rayDir * uniforms.focusDistance;
            rayOrigin += lensSample.x * float3(uniforms.cameraRight) + lensSample.y * float3(uniforms.cameraUp);
            rayDir = normalize(focusPoint - rayOrigin);
        }

        float3 throughput = float3(1.0);
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
                    LightSample ls = sampleAreaLight(lights, uniforms.lightCount, rngState);
                    float3 toLight = ls.point - scatterPoint;
                    float distSq = dot(toLight, toLight);
                    float dist = sqrt(distSq);
                    float3 wi = toLight / dist;
                    float cosLight = dot(ls.normal, -wi);
                    if (cosLight > 0.0) {
                        ray shadowRay;
                        shadowRay.origin = scatterPoint;
                        shadowRay.direction = wi;
                        shadowRay.min_distance = 0.001f;
                        shadowRay.max_distance = dist - 0.002f;
                        intersection_result<instancing, triangle_data> shadowResult =
                            isect.intersect(shadowRay, accelStructure, functionTable);
                        if (shadowResult.type == intersection_type::none) {
                            float pdfSolidAngle = (distSq / (ls.area * cosLight)) / float(uniforms.lightCount);
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
                    // distance falloff and (deliberately) no fog
                    // attenuation, unlike the point/spot loop just above.
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
                            radiance += throughput * dlPhaseValue * float3(dl.emission);
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
                    radiance += throughput * earthTexture.sample(textureSampler, envUV).rgb;
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
                mat = diskMaterials[0];
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
            }

            // materialType == 3 (textured Lambertian) only ever occurs on
            // a triangle (the back wall - see metal_poc.mm's scene setup),
            // never the sphere, so texCoordFor()'s triangle-only inputs
            // (primId, barycentric_coord) are always valid when this
            // fires - no isSphere guard needed here the way the normal/
            // material lookup above needed one.
            float3 albedo;
            if (mat.materialType == 3u) {
                float2 uv = texCoordFor(primId, result.triangle_barycentric_coord, uvs);
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
            // `&& frontFace`: an AreaLight only emits from the side its own
            // `normal` points toward - the same one-sidedness the NEE
            // branches below already enforce via their own `cosLight > 0.0`
            // check (see e.g. this scene's own ceiling lights, which only
            // shine down into the room). Without this, a camera ray or
            // BSDF-sampled bounce landing on the BACK of a light quad would
            // still read its emission unconditionally - invisible in this
            // committed scene (every light is mounted flush against the
            // ceiling, its own back face physically inaccessible from
            // inside the room) but a genuine correctness gap: this is the
            // one place in the shader a light's own emission was reachable
            // without a facing check at all, inconsistent with every NEE
            // branch's own already-correct behaviour. `frontFace` is
            // already computed above for the dielectric branch's own eta
            // selection - reused here, not recomputed.
            if (any(float3(mat.emission) > float3(0.0)) && frontFace) {
                if (specularBounce) {
                    // No competing NEE sample could have produced this
                    // exact hit (camera ray, or a mirror/glass bounce -
                    // both skip NEE entirely, see their own branches
                    // below), so there's nothing to weight against.
                    radiance += throughput * float3(mat.emission);
                } else {
                    // Reached via a BSDF-sampled continuation ray (diffuse
                    // or conductor) - weight by the power heuristic against
                    // what the light-sampling strategy's own PDF would have
                    // been for this exact hit. mat.lightId names exactly
                    // which AreaLight this triangle belongs to, so this
                    // works for any number of lights, not just one -
                    // uniform light-picking pdf (1/lightCount) folded in
                    // alongside the same area-to-solid-angle conversion
                    // the NEE branches below use.
                    AreaLight light = lights[mat.lightId];
                    float distSq = result.distance * result.distance;
                    float cosLight = max(dot(float3(light.normal), -rayDir), 0.0001);
                    float pdfLight = (distSq / (light.area * cosLight)) / float(uniforms.lightCount);
                    float weight = (bsdfPdf * bsdfPdf) / (bsdfPdf * bsdfPdf + pdfLight * pdfLight);
                    radiance += throughput * float3(mat.emission) * weight;
                }
            }

            if (mat.materialType == 2u) {
                // Dielectric (glass): Schlick-approximated Fresnel decides
                // reflect vs refract stochastically each bounce - same
                // "one importance-sampled choice per hit, unbiased in
                // expectation" approach pbrt-v4 and this project's own CPU
                // dielectric material use, not a 50/50 split of energy.
                float refractionRatio = frontFace ? (1.0 / mat.ior) : mat.ior;
                float3 unitDir = normalize(rayDir);
                float cosTheta = min(dot(-unitDir, facingNormal), 1.0);
                float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
                bool cannotRefract = refractionRatio * sinTheta > 1.0;

                float3 newDir;
                if (cannotRefract || schlickReflectance(cosTheta, refractionRatio) > randFloat(rngState)) {
                    newDir = reflect(unitDir, facingNormal);
                } else {
                    newDir = refract(unitDir, facingNormal, refractionRatio);
                }
                rayDir = newDir;
                // Offset along the GEOMETRIC (unflipped-for-facing) normal
                // signed toward the new ray direction, not always
                // `facingNormal` - reflect() and refract() can each send
                // the continuation ray to either side of the surface here
                // (reflect always exits the front face refract() entered
                // from; total internal reflection inside the sphere does
                // not), and offsetting on the wrong side re-intersects the
                // same surface immediately (self-shadowing acne).
                rayOrigin = hitPoint + (dot(newDir, normal) > 0.0 ? normal : -normal) * 0.001f;
                // Glass is delta-transmissive/reflective, same "no NEE, a
                // shadow ray toward a point light has zero probability of
                // landing exactly on the one direction that mattered" logic
                // as the mirror branch below.
                applyBeerLambertAbsorption(throughput, mat.color, frontFace, result.distance);
                specularBounce = true;
            } else if (mat.materialType == 5u) {
                // Rough (frosted) dielectric: the same Schlick-Fresnel
                // reflect-vs-refract decision as materialType 2 above, but
                // taken about a GGX-VNDF-SAMPLED microfacet normal instead
                // of the smooth geometric one - the standard way a rough
                // interface's normal gets perturbed (same sampleGGXVNDF()
                // the conductor branch uses, called here in `facingNormal`'s
                // own local frame so alpha == 0 degenerates to hWorld ==
                // facingNormal exactly, i.e. materialType 2's own math
                // bit-for-bit, verified below).
                //
                // What this DOESN'T do, unlike the conductor branch's own
                // G/G1(wo) throughput correction: a full energy-conserving
                // rough-BTDF derivation. A correct one needs a transmission
                // Jacobian AND an eta^2 radiance-scaling term on top of the
                // reflection-side G/G1 ratio (Walter et al. 2007's rough
                // refraction model) - real, tricky-to-verify-without-a-
                // reference-implementation math, the exact "looks
                // plausible, renders something, is subtly wrong" trap
                // Section 4 warns about. Rather than ship that unverified,
                // this keeps materialType 2's existing throughput
                // accounting (`albedo`, no G/G1 correction) and treats the
                // roughness as a direction-only perturbation - a known,
                // deliberate simplification (documented here rather than
                // silently assumed away), same spirit as this POC's
                // existing Schlick-vs-full-Fresnel approximation.
                float alpha = max(mat.roughness * mat.roughness, 0.0009);
                float3 tangent, bitangent;
                buildOnb(facingNormal, tangent, bitangent);
                float3 woWorld = -rayDir;
                float3 woLocal = float3(dot(woWorld, tangent), dot(woWorld, bitangent), dot(woWorld, facingNormal));
                woLocal.z = max(woLocal.z, 0.0001);
                // Isotropic here (alphaX == alphaY == alpha) - this
                // branch's own tangent frame is buildOnb()'s arbitrary
                // one, which only gives a consistent result when the
                // distribution has no azimuthal dependence at all.
                float3 hLocal = sampleGGXVNDF(woLocal, alpha, alpha, rngState);
                float3 hWorld = normalize(hLocal.x * tangent + hLocal.y * bitangent + hLocal.z * facingNormal);

                float refractionRatio = frontFace ? (1.0 / mat.ior) : mat.ior;
                float3 unitDir = normalize(rayDir);
                // Unlike materialType 2's own dot(-unitDir, facingNormal)
                // (provably >= 0, since facingNormal is always built to
                // oppose the ray), hWorld here is a VNDF-sampled
                // microfacet normal perturbed away from facingNormal - at
                // grazing incidence combined with high roughness, the
                // angle between the view direction and THIS particular
                // sampled half-vector can exceed 90 degrees even though
                // the angle to facingNormal itself never does, so this
                // needs its own lower clamp too (schlickReflectance's
                // pow(1-cosine, 5) term overshoots past 1 on a negative
                // cosine otherwise, over-weighting the reflect branch).
                float cosTheta = clamp(dot(-unitDir, hWorld), 0.0, 1.0);
                float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
                bool cannotRefract = refractionRatio * sinTheta > 1.0;

                float3 newDir;
                if (cannotRefract || schlickReflectance(cosTheta, refractionRatio) > randFloat(rngState)) {
                    newDir = reflect(unitDir, hWorld);
                } else {
                    newDir = refract(unitDir, hWorld, refractionRatio);
                }
                rayDir = newDir;
                rayOrigin = hitPoint + (dot(newDir, normal) > 0.0 ? normal : -normal) * 0.001f;
                applyBeerLambertAbsorption(throughput, mat.color, frontFace, result.distance);
                specularBounce = true;
            } else if (mat.materialType == 4u) {
                // Rough conductor (GGX metal): structurally the same NEE +
                // BSDF-sampled-continuation + MIS shape as the Lambertian
                // branch below - only the BRDF/sampling math changes, from
                // a cosine-weighted diffuse lobe to an importance-sampled
                // microfacet one. `albedo` here is F0 (per-primitive
                // reflectance colour), not a diffuse albedo - see
                // TriangleMaterial's own comment.
                //
                // Genuinely ANISOTROPIC: `ior` gives alphaX as before,
                // and `roughness` - otherwise idle for this materialType,
                // since materialType 5 is the only other reader of that
                // field - now doubles as alphaY. `roughness == 0.0`
                // (every scene before this one) falls back to alphaY ==
                // alphaX, the exact isotropic case this material used
                // through step 22 - not a separate code path, the same
                // fallback shape this POC already uses for HG's g == 0
                // and rough dielectric's roughness == 0.
                float alphaX = max(mat.ior * mat.ior, 0.0009);
                float alphaY = (mat.roughness > 0.0) ? max(mat.roughness * mat.roughness, 0.0009) : alphaX;
                float3 tangent, bitangent;
                buildAnisotropicOnb(facingNormal, tangent, bitangent);
                float3 woWorld = -rayDir;
                float3 woLocal = float3(dot(woWorld, tangent), dot(woWorld, bitangent), dot(woWorld, facingNormal));
                woLocal.z = max(woLocal.z, 0.0001);

                if (all(mat.emission == float3(0.0))) {
                    LightSample ls = sampleAreaLight(lights, uniforms.lightCount, rngState);
                    float3 toLight = ls.point - hitPoint;
                    float distSq = dot(toLight, toLight);
                    float dist = sqrt(distSq);
                    float3 wi = toLight / dist;
                    float cosSurface = dot(facingNormal, wi);
                    float cosLight = dot(ls.normal, -wi);
                    if (cosSurface > 0.0 && cosLight > 0.0) {
                        float3 wiLocal = float3(dot(wi, tangent), dot(wi, bitangent), dot(wi, facingNormal));
                        float3 h = normalize(woLocal + wiLocal);
                        float NdotO = woLocal.z;
                        float NdotI = max(wiLocal.z, 0.0001);
                        float Dh = ggxD(h, alphaX, alphaY);
                        float G = ggxG(woLocal, wiLocal, alphaX, alphaY);
                        float3 F = fresnelSchlickConductor(max(dot(woLocal, h), 0.0), albedo);
                        float3 brdf = Dh * G * F / max(4.0 * NdotO * NdotI, 1e-6);

                        ray shadowRay;
                        shadowRay.origin = hitPoint + facingNormal * 0.001f;
                        shadowRay.direction = wi;
                        shadowRay.min_distance = 0.001f;
                        shadowRay.max_distance = dist - 0.002f;
                        intersection_result<instancing, triangle_data> shadowResult =
                            isect.intersect(shadowRay, accelStructure, functionTable);
                        if (shadowResult.type == intersection_type::none) {
                            float pdfSolidAngle = (distSq / (ls.area * cosLight)) / float(uniforms.lightCount);
                            // VNDF sampling's own pdf(wi) for this same
                            // direction - pdf(h) = D(h)*G1(wo)*max(0,dot
                            // (wo,h))/NdotO, converted to a solid-angle-of-
                            // wi pdf via the standard reflection Jacobian
                            // 1/(4*dot(wo,h)) - the counterpart the
                            // continuation-ray branch below computes for
                            // its OWN sampled direction.
                            float pdfBsdf = (Dh * ggxG1(woLocal, alphaX, alphaY)) / max(4.0 * NdotO, 1e-6);
                            float weight = (pdfSolidAngle * pdfSolidAngle)
                                / (pdfSolidAngle * pdfSolidAngle + pdfBsdf * pdfBsdf);
                            // exp(-sigmaT*dist): the fog's own attenuation
                            // along THIS shadow ray - a no-op (1.0) when
                            // fogSigmaT == 0, same purely-additive pattern
                            // as every other fog-related change here.
                            float transmittance = exp(-uniforms.fogSigmaT * dist);
                            radiance += throughput * brdf * ls.emission * cosSurface * transmittance / pdfSolidAngle * weight;
                        }
                    }

                    // Point lights: summed unconditionally, not picked,
                    // full weight, no MIS - see PointLight's own comment.
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
                            float3 plF = fresnelSchlickConductor(max(dot(woLocal, plH), 0.0), albedo);
                            float3 plBrdf = plDh * plG * plF / max(4.0 * plNdotO * plNdotI, 1e-6);

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

                    // Directional lights: summed unconditionally, not
                    // picked - see DirectionalLight's own comment. No
                    // distance falloff and (deliberately) no fog
                    // attenuation, unlike the point/spot loop just above.
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
                            float3 dlF = fresnelSchlickConductor(max(dot(woLocal, dlH), 0.0), albedo);
                            float3 dlBrdf = dlDh * dlG * dlF / max(4.0 * dlNdotO * dlNdotI, 1e-6);

                            ray dlShadowRay;
                            dlShadowRay.origin = hitPoint + facingNormal * 0.001f;
                            dlShadowRay.direction = dlWi;
                            dlShadowRay.min_distance = 0.001f;
                            dlShadowRay.max_distance = kDirectionalLightMaxDistance;
                            intersection_result<instancing, triangle_data> dlShadowResult =
                                isect.intersect(dlShadowRay, accelStructure, functionTable);
                            if (dlShadowResult.type == intersection_type::none) {
                                radiance += throughput * dlBrdf * float3(dl.emission) * dlCosSurface;
                            }
                        }
                    }
                }

                float3 hLocal = sampleGGXVNDF(woLocal, alphaX, alphaY, rngState);
                float3 wiLocal = reflect(-woLocal, hLocal);
                if (wiLocal.z <= 0.0) {
                    // Sampled a half-vector whose reflection lands below
                    // the hemisphere (possible at grazing angles/high
                    // roughness) - a real BRDF value of zero, not a bug;
                    // terminate this path rather than continue with an
                    // invalid direction.
                    break;
                }
                float3 wiWorld = normalize(wiLocal.x * tangent + wiLocal.y * bitangent + wiLocal.z * facingNormal);

                float NdotO = woLocal.z;
                float G = ggxG(woLocal, wiLocal, alphaX, alphaY);
                float G1 = ggxG1(woLocal, alphaX, alphaY);
                float3 F = fresnelSchlickConductor(max(dot(woLocal, hLocal), 0.0), albedo);
                // f(wo,wi)*cosI/pdf(wi) collapses to F*G/G1(wo) for a
                // VNDF-sampled direction - the D and 4*NdotO*NdotI terms
                // in the BRDF exactly cancel the same terms in pdf(wi)'s
                // own Jacobian-converted form, leaving only the Fresnel
                // term and the ratio of the full (both-directions) to
                // single-direction (view-only) Smith masking-shadowing
                // term. Same simplification pbrt-v4's own conductor
                // Sample_f relies on for VNDF-sampled reflection.
                throughput *= F * (G / max(G1, 1e-6));

                rayDir = wiWorld;
                rayOrigin = hitPoint + facingNormal * 0.001f;
                bsdfPdf = (ggxD(hLocal, alphaX, alphaY) * G1) / max(4.0 * NdotO, 1e-6);
                specularBounce = false;
            } else if (mat.materialType == 1u) {
                // Mirror: deterministic reflection, no light sampling (a
                // specular surface has zero probability of the shadow ray
                // toward a delta light landing exactly on the reflection
                // vector - NEE simply doesn't apply here, same reason the
                // CPU renderer's own BSDFs skip NEE for specular lobes).
                //
                // Fresnel-weighted, not a flat `albedo` multiply the way
                // every earlier version of this branch did: a real
                // mirror's reflectance rises toward white/uncolored at
                // grazing angles regardless of its base tint (the same
                // physical effect the GGX conductor material already
                // models via this exact function - `albedo` doubles as
                // this surface's own F0 here, the same "colour IS the
                // normal-incidence reflectance" convention that material
                // already established). A flat multiply is only correct
                // exactly at normal incidence (cosTheta == 1, where this
                // reduces to F0 == albedo); it silently under-brightens
                // every grazing-angle reflection otherwise.
                float3 newDir = reflect(rayDir, facingNormal);
                float cosTheta = max(dot(facingNormal, -rayDir), 0.0001);
                float3 fresnel = fresnelSchlickConductor(cosTheta, albedo);
                rayDir = newDir;
                rayOrigin = hitPoint + facingNormal * 0.001f;
                throughput *= fresnel;
                specularBounce = true;
            } else {
                // Lambertian (materialType 0, or 3 - textured, the only
                // difference already resolved above into `albedo`, the
                // BSDF/NEE math below has no idea where albedo came
                // from): next-event estimation against a RANDOMLY PICKED
                // light from `lights` (uniform-area-sampled point on it +
                // solid-angle PDF conversion, shadow ray up to just short
                // of the light rather than infinite), then continue the
                // path via cosine-weighted hemisphere sampling for
                // indirect light. Two separate rays per bounce - direct
                // (shadow) and the continuation - is the standard NEE
                // split this project's own CPU path_integrator.h also
                // uses; any light's OWN triangles skip this (mat.emission's
                // already-added contribution above is their entire direct
                // lighting - sampling a light FROM itself, including a
                // DIFFERENT light, would double count that light's own
                // emission).
                if (all(mat.emission == float3(0.0))) {
                    LightSample ls = sampleAreaLight(lights, uniforms.lightCount, rngState);
                    float3 toLight = ls.point - hitPoint;
                    float distSq = dot(toLight, toLight);
                    float dist = sqrt(distSq);
                    float3 wi = toLight / dist;
                    float cosSurface = dot(facingNormal, wi);
                    float cosLight = dot(ls.normal, -wi);
                    if (cosSurface > 0.0 && cosLight > 0.0) {
                        ray shadowRay;
                        shadowRay.origin = hitPoint + facingNormal * 0.001f;
                        shadowRay.direction = wi;
                        shadowRay.min_distance = 0.001f;
                        // Short of the light's own surface, not infinite -
                        // an infinite shadow ray would hit the light quad
                        // ITSELF and always report "occluded".
                        shadowRay.max_distance = dist - 0.002f;
                        intersection_result<instancing, triangle_data> shadowResult =
                            isect.intersect(shadowRay, accelStructure, functionTable);
                        if (shadowResult.type == intersection_type::none) {
                            // Area-to-solid-angle PDF conversion:
                            // pdf_omega = pdf_area * dist^2 / cosLight,
                            // pdf_area = 1/ls.area for uniform sampling -
                            // textbook area-light NEE, not an approximation -
                            // times 1/lightCount for the uniform light-pick
                            // probability (one-sample MIS over the light
                            // list, same approach pbrt-v4's own
                            // UniformLightSampler uses).
                            float pdfSolidAngle = (distSq / (ls.area * cosLight)) / float(uniforms.lightCount);
                            // MIS weight against what the BSDF-sampling
                            // strategy's own PDF would be for this same
                            // direction wi (cosine-weighted: cosSurface/pi) -
                            // symmetric counterpart to the weight applied
                            // to a BSDF-sampled ray landing on the light
                            // above.
                            float pdfBsdfForThisDir = cosSurface / M_PI_F;
                            float weight = (pdfSolidAngle * pdfSolidAngle)
                                / (pdfSolidAngle * pdfSolidAngle + pdfBsdfForThisDir * pdfBsdfForThisDir);
                            // exp(-sigmaT*dist): the fog's own attenuation
                            // along THIS shadow ray - a no-op (1.0) when
                            // fogSigmaT == 0.
                            float transmittance = exp(-uniforms.fogSigmaT * dist);
                            radiance += throughput * albedo * (1.0 / M_PI_F)
                                        * ls.emission * cosSurface * transmittance / pdfSolidAngle * weight;
                        }
                    }

                    // Point lights: summed unconditionally, not picked -
                    // see PointLight's own comment on why a delta light
                    // needs no MIS weight and no area-to-solid-angle pdf
                    // conversion at all, just an idealized 1/distSq
                    // falloff.
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

                    // Directional lights: summed unconditionally, not
                    // picked - see DirectionalLight's own comment. No
                    // distance falloff and (deliberately, see that same
                    // comment) no fog attenuation, unlike the point/spot
                    // loop just above.
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
                                radiance += throughput * albedo * (1.0 / M_PI_F)
                                            * float3(dl.emission) * dlCosSurface;
                            }
                        }
                    }
                }

                rayDir = cosineSampleHemisphere(facingNormal, rngState);
                rayOrigin = hitPoint + facingNormal * 0.001f;
                // Cosine-weighted sampling's pdf (cos(theta)/pi) cancels
                // the BSDF's own cos(theta)/pi exactly, leaving the flat
                // albedo below - textbook importance-sampled Lambertian,
                // not an approximation.
                throughput *= albedo;
                // Recorded for next iteration's MIS weighting of a
                // direct-light-hit encountered via THIS sampled direction -
                // cosine-weighted sampling's own PDF is cos(theta)/pi,
                // theta measured against the same facingNormal it was
                // sampled around.
                bsdfPdf = max(dot(facingNormal, rayDir), 0.0001) / M_PI_F;
                specularBounce = false;
            }
            } // !scatteredInMedium

            // Russian roulette after a few bounces, same "let cheap paths
            // terminate early, keep expensive ones unbiased" shape as
            // this project's CPU integrator - throughput's max channel is
            // the survival probability, divided back in on survival so
            // the estimator stays unbiased.
            if (depth > 3) {
                float p = max(throughput.x, max(throughput.y, throughput.z));
                if (randFloat(rngState) > p) break;
                throughput /= max(p, 0.0001);
            }
        }

        accumColor += radiance;
    }

    accumColor /= float(uniforms.samplesPerPixel);
    outTexture.write(float4(accumColor, 1.0), tid);
}
