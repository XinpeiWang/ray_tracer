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
};

// materialType: 0 = Lambertian diffuse, 1 = mirror (perfect specular),
// 2 = dielectric (glass), 3 = textured Lambertian (same BSDF/NEE code path
// as 0, just samples `earthTexture` at the hit's interpolated UV for
// albedo instead of reading `color` - see the shading loop below), 4 =
// rough conductor (GGX microfacet metal - `color` is the conductor's
// normal-incidence reflectance F0, not a diffuse albedo).
// `ior` is meaningful for materialType == 2 (refraction index) and
// reused, differently, for materialType == 4 (perceptual roughness in
// [0,1], squared into the GGX alpha parameter below) - the two never
// coexist on one primitive, so sharing the slot avoids a second
// otherwise-almost-always-zero field. Carried on every material anyway
// (rather than a separate per-type struct) since this POC values "one
// flat array, index by primitive_id" simplicity over saving a few bytes
// on entries that don't use every field. `emission` is nonzero only for
// the one light quad's two triangles (see kLight* below) - same "just
// carry it, don't special-case a rare field" reasoning. Checking it
// unconditionally on every hit (regardless of material type) is what
// makes a light source visible at all when a camera/GI ray lands on it
// directly, on top of the light-SAMPLING code below which handles every
// other surface's direct illumination FROM it.
struct TriangleMaterial {
    packed_float3 color;
    uint materialType;
    float ior;
    packed_float3 emission;
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

// Schlick's approximation - the standard cheap stand-in for the full
// Fresnel dielectric reflectance formula, same one pbrt-v4 and this
// project's own CPU dielectric material use for the reflect-vs-refract
// decision.
inline float schlickReflectance(float cosine, float refractionRatio) {
    float r0 = (1.0 - refractionRatio) / (1.0 + refractionRatio);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

// ---------------------------------------------------------------------------
// GGX / Trowbridge-Reitz microfacet distribution + height-correlated Smith
// masking-shadowing - the standard model materialType == 4 (rough
// conductor) uses below, same formulation pbrt-v4's own
// TrowbridgeReitzDistribution implements (isotropic case: alpha_x ==
// alpha_y). All three take `NdotX`/`alpha` already-computed rather than
// raw vectors, since every call site here already has the dot product on
// hand from building its own local shading frame - keeps these as pure,
// reusable scalar functions.
// ---------------------------------------------------------------------------
inline float ggxD(float NdotH, float alpha) {
    float a2 = alpha * alpha;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(M_PI_F * d * d, 1e-9);
}

// Smith's Lambda function (isotropic GGX closed form) - how much of a
// microfacet's neighbourhood is masked/shadowed as seen from direction
// `w`, folded into G1/G below rather than used standalone.
inline float ggxLambda(float NdotW, float alpha) {
    float NdotW2 = max(NdotW * NdotW, 1e-6);
    float tan2Theta = max(0.0, 1.0 - NdotW2) / NdotW2;
    return 0.5 * (sqrt(1.0 + alpha * alpha * tan2Theta) - 1.0);
}

inline float ggxG1(float NdotW, float alpha) {
    return 1.0 / (1.0 + ggxLambda(NdotW, alpha));
}

// Height-correlated Smith masking-shadowing for a full reflection lobe
// (both the view and light direction masked/shadowed jointly, not treated
// as independent) - the same correlated form pbrt-v4 uses, less energy
// loss at grazing angles than a naive G1(wo)*G1(wi) product.
inline float ggxG(float NdotO, float NdotI, float alpha) {
    return 1.0 / (1.0 + ggxLambda(NdotO, alpha) + ggxLambda(NdotI, alpha));
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

// Samples a half-vector from the GGX distribution of VISIBLE normals
// (Heitz 2018, "Sampling the GGX Distribution of Visible Normals"), given
// the outgoing direction `woLocal` already in the local (Z-up == shading
// normal) frame. Dramatically lower variance than importance-sampling
// D(h) directly, especially near grazing angles - the same algorithm
// pbrt-v4's TrowbridgeReitzDistribution::Sample_wm implements, chosen
// here for the same reason: it's what makes a rough-conductor path
// tracer converge in a reasonable sample count instead of needing far
// more samples to beat down grazing-angle noise.
inline float3 sampleGGXVNDF(float3 woLocal, float alpha, thread uint& rngState) {
    float3 Vh = normalize(float3(alpha * woLocal.x, alpha * woLocal.y, woLocal.z));
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
    float3 Ne = float3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z));
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
    uint2 tid [[thread_position_in_grid]])
{
    // Bilinear + repeat/wrap: the standard choice for a UV-mapped photo
    // texture (the earth-map's own left/right edges are meant to tile
    // seamlessly at the date line) - constexpr so it's resolved at
    // compile time, same as every Metal sample/tutorial's own pattern for
    // a sampler that never needs to change at runtime.
    constexpr sampler textureSampler(coord::normalized, address::repeat, filter::linear);
    if (tid.x >= uniforms.width || tid.y >= uniforms.height) return;

    // Area light: a small quad hanging just under the ceiling, facing
    // straight down - real geometry (added via addQuad() host-side, see
    // metal_poc.mm) with nonzero TriangleMaterial::emission on its two
    // triangles, not a separate light type. Its extent/normal/area are
    // hardcoded here to match that geometry exactly rather than derived
    // from it at trace time - a real port would carry a proper light list
    // (this project's own CPU src/TheRestOfYourLife/*light_sampler*.h is
    // exactly that abstraction) instead of one hardcoded light's shape
    // baked into the integrator.
    const float3 kLightCenter = float3(0.0, 0.98, 0.0);
    const float2 kLightHalfExtents = float2(0.3, 0.3); // x, z half-widths
    const float3 kLightNormal = float3(0.0, -1.0, 0.0);
    const float kLightArea = (2.0 * kLightHalfExtents.x) * (2.0 * kLightHalfExtents.y);
    const float3 kLightEmission = float3(15.0, 15.0, 14.0);

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

        float3 rayOrigin = float3(uniforms.cameraPos);
        float3 rayDir = normalize(float3(uniforms.cameraForward)
                                   + screen.x * float3(uniforms.cameraRight)
                                   + screen.y * float3(uniforms.cameraUp));

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

            if (result.type == intersection_type::none) {
                float skyT = 0.5 * (rayDir.y + 1.0);
                radiance += throughput * mix(skyBottom, skyTop, skyT);
                break;
            }

            uint primId = result.primitive_id;
            float3 hitPoint = rayOrigin + rayDir * result.distance;

            // Geometric normal: derived from the triangle's own vertices
            // for a triangle hit (as before), or from the sphere's centre
            // for a bounding-box hit - a sphere has no "vertices" to pull
            // a face normal from, but (hitPoint - centre) is exact for a
            // perfect sphere, no approximation.
            bool isSphere = (result.type == intersection_type::bounding_box);
            float3 normal;
            TriangleMaterial mat;
            if (isSphere) {
                SphereData sphere = spheres[primId];
                normal = normalize(hitPoint - float3(sphere.center));
                mat = sphereMaterials[primId];
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
            if (any(float3(mat.emission) > float3(0.0))) {
                if (specularBounce) {
                    // No competing NEE sample could have produced this
                    // exact hit (camera ray, or a mirror/glass bounce -
                    // both skip NEE entirely, see their own branches
                    // below), so there's nothing to weight against.
                    radiance += throughput * float3(mat.emission);
                } else {
                    // Reached via a Lambertian BSDF-sampled continuation
                    // ray - weight by the power heuristic against what
                    // the light-sampling strategy's own PDF would have
                    // been for this exact hit, using the same area-to-
                    // solid-angle conversion the NEE branch below uses.
                    float distSq = result.distance * result.distance;
                    float cosLight = max(dot(kLightNormal, -rayDir), 0.0001);
                    float pdfLight = distSq / (kLightArea * cosLight);
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
                // as the mirror branch below - throughput stays at the
                // glass's own tint (near-1.0/clear for realistic glass).
                throughput *= albedo;
                specularBounce = true;
            } else if (mat.materialType == 4u) {
                // Rough conductor (GGX metal): structurally the same NEE +
                // BSDF-sampled-continuation + MIS shape as the Lambertian
                // branch below - only the BRDF/sampling math changes, from
                // a cosine-weighted diffuse lobe to an importance-sampled
                // microfacet one. `albedo` here is F0 (per-primitive
                // reflectance colour), not a diffuse albedo - see
                // TriangleMaterial's own comment.
                float roughness = mat.ior;
                float alpha = max(roughness * roughness, 0.0009);
                float3 tangent, bitangent;
                buildOnb(facingNormal, tangent, bitangent);
                float3 woWorld = -rayDir;
                float3 woLocal = float3(dot(woWorld, tangent), dot(woWorld, bitangent), dot(woWorld, facingNormal));
                woLocal.z = max(woLocal.z, 0.0001);

                if (all(mat.emission == float3(0.0))) {
                    float2 u = float2(randFloat(rngState), randFloat(rngState));
                    float3 lightPoint = kLightCenter + float3(
                        (u.x * 2.0 - 1.0) * kLightHalfExtents.x, 0.0, (u.y * 2.0 - 1.0) * kLightHalfExtents.y);
                    float3 toLight = lightPoint - hitPoint;
                    float distSq = dot(toLight, toLight);
                    float dist = sqrt(distSq);
                    float3 wi = toLight / dist;
                    float cosSurface = dot(facingNormal, wi);
                    float cosLight = dot(kLightNormal, -wi);
                    if (cosSurface > 0.0 && cosLight > 0.0) {
                        float3 wiLocal = float3(dot(wi, tangent), dot(wi, bitangent), dot(wi, facingNormal));
                        float3 h = normalize(woLocal + wiLocal);
                        float NdotH = max(h.z, 0.0001);
                        float NdotO = woLocal.z;
                        float NdotI = max(wiLocal.z, 0.0001);
                        float Dh = ggxD(NdotH, alpha);
                        float G = ggxG(NdotO, NdotI, alpha);
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
                            float pdfSolidAngle = distSq / (kLightArea * cosLight);
                            // VNDF sampling's own pdf(wi) for this same
                            // direction - pdf(h) = D(h)*G1(wo)*max(0,dot
                            // (wo,h))/NdotO, converted to a solid-angle-of-
                            // wi pdf via the standard reflection Jacobian
                            // 1/(4*dot(wo,h)) - the counterpart the
                            // continuation-ray branch below computes for
                            // its OWN sampled direction.
                            float pdfBsdf = (Dh * ggxG1(NdotO, alpha)) / max(4.0 * NdotO, 1e-6);
                            float weight = (pdfSolidAngle * pdfSolidAngle)
                                / (pdfSolidAngle * pdfSolidAngle + pdfBsdf * pdfBsdf);
                            radiance += throughput * brdf * kLightEmission * cosSurface / pdfSolidAngle * weight;
                        }
                    }
                }

                float3 hLocal = sampleGGXVNDF(woLocal, alpha, rngState);
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
                float NdotI = max(wiLocal.z, 0.0001);
                float NdotH = max(hLocal.z, 0.0001);
                float G = ggxG(NdotO, NdotI, alpha);
                float G1 = ggxG1(NdotO, alpha);
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
                bsdfPdf = (ggxD(NdotH, alpha) * G1) / max(4.0 * NdotO, 1e-6);
                specularBounce = false;
            } else if (mat.materialType == 1u) {
                // Mirror: deterministic reflection, no light sampling (a
                // specular surface has zero probability of the shadow ray
                // toward a delta light landing exactly on the reflection
                // vector - NEE simply doesn't apply here, same reason the
                // CPU renderer's own BSDFs skip NEE for specular lobes).
                rayDir = reflect(rayDir, facingNormal);
                rayOrigin = hitPoint + facingNormal * 0.001f;
                throughput *= albedo;
                specularBounce = true;
            } else {
                // Lambertian (materialType 0, or 3 - textured, the only
                // difference already resolved above into `albedo`, the
                // BSDF/NEE math below has no idea where albedo came
                // from): next-event estimation against the area light
                // (uniform-area-sampled point + solid-angle PDF
                // conversion, shadow ray up to just short of the light
                // rather than infinite), then continue the path via
                // cosine-weighted hemisphere sampling for indirect light.
                // Two separate rays per bounce - direct (shadow) and the
                // continuation - is the standard NEE split this project's
                // own CPU path_integrator.h also uses; the light-quad's
                // OWN two triangles skip this (mat.emission's already-
                // added contribution above is their entire direct
                // lighting - sampling the light FROM itself is degenerate).
                if (all(mat.emission == float3(0.0))) {
                    float2 u = float2(randFloat(rngState), randFloat(rngState));
                    float3 lightPoint = kLightCenter + float3(
                        (u.x * 2.0 - 1.0) * kLightHalfExtents.x, 0.0, (u.y * 2.0 - 1.0) * kLightHalfExtents.y);
                    float3 toLight = lightPoint - hitPoint;
                    float distSq = dot(toLight, toLight);
                    float dist = sqrt(distSq);
                    float3 wi = toLight / dist;
                    float cosSurface = dot(facingNormal, wi);
                    float cosLight = dot(kLightNormal, -wi);
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
                            // pdf_area = 1/kLightArea for uniform sampling -
                            // textbook area-light NEE, not an approximation.
                            float pdfSolidAngle = distSq / (kLightArea * cosLight);
                            // MIS weight against what the BSDF-sampling
                            // strategy's own PDF would be for this same
                            // direction wi (cosine-weighted: cosSurface/pi) -
                            // symmetric counterpart to the weight applied
                            // to a BSDF-sampled ray landing on the light
                            // above.
                            float pdfBsdfForThisDir = cosSurface / M_PI_F;
                            float weight = (pdfSolidAngle * pdfSolidAngle)
                                / (pdfSolidAngle * pdfSolidAngle + pdfBsdfForThisDir * pdfBsdfForThisDir);
                            radiance += throughput * albedo * (1.0 / M_PI_F)
                                        * kLightEmission * cosSurface / pdfSolidAngle * weight;
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
