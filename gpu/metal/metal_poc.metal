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
// 2 = dielectric (glass). `ior` is only meaningful for materialType == 2 -
// carried on every material anyway (rather than a separate per-type
// struct) since this POC values "one flat array, index by primitive_id"
// simplicity over saving 4 bytes on non-dielectric entries. `emission` is
// nonzero only for the one light quad's two triangles (see kLight* below) -
// same "just carry it, don't special-case a rare field" reasoning as ior.
// Checking it unconditionally on every hit (regardless of material type)
// is what makes a light source visible at all when a camera/GI ray lands
// on it directly, on top of the light-SAMPLING code below which handles
// every other surface's direct illumination FROM it.
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

inline float3 faceNormalFor(uint primId, device const packed_float3* vertices) {
    float3 v0 = float3(vertices[primId * 3 + 0]);
    float3 v1 = float3(vertices[primId * 3 + 1]);
    float3 v2 = float3(vertices[primId * 3 + 2]);
    return normalize(cross(v1 - v0, v2 - v0));
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

kernel void primaryRayKernel(
    texture2d<float, access::write> outTexture [[texture(0)]],
    instance_acceleration_structure accelStructure [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]],
    device const TriangleMaterial* triMaterials [[buffer(2)]],
    device const packed_float3* vertices [[buffer(3)]],
    device const TriangleMaterial* sphereMaterials [[buffer(4)]],
    device const SphereData* spheres [[buffer(5)]],
    intersection_function_table<instancing, triangle_data> functionTable [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]])
{
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
                normal = faceNormalFor(primId, vertices);
                mat = triMaterials[primId];
            }
            // Raw (outward, unflipped) normal kept separately from here -
            // dielectric handling below needs to know which side of the
            // surface the ray is entering from (front vs back face) to
            // pick the right eta ratio, information the flipped-to-face-
            // the-ray version used by every other material below discards.
            bool frontFace = dot(normal, rayDir) < 0.0;
            float3 facingNormal = frontFace ? normal : -normal;

            float3 albedo = float3(mat.color);

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
                // Lambertian: next-event estimation against the area
                // light (uniform-area-sampled point + solid-angle PDF
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
