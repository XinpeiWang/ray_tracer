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

// materialType: 0 = Lambertian diffuse, 1 = mirror (perfect specular).
// Real materials would carry a full BxDF id + parameters (IOR, roughness,
// ...); this POC only needs enough to prove the shader can branch on
// per-primitive material data at all, the same reason step 1's per-
// primitive colour existed.
struct TriangleMaterial {
    packed_float3 color;
    uint materialType;
};

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

kernel void primaryRayKernel(
    texture2d<float, access::write> outTexture [[texture(0)]],
    instance_acceleration_structure accelStructure [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]],
    device const TriangleMaterial* triMaterials [[buffer(2)]],
    device const packed_float3* vertices [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height) return;

    const float3 lightDir = normalize(float3(0.4, 0.8, 0.3));
    const float3 lightColor = float3(1.0, 0.98, 0.92);
    const float3 skyTop = float3(0.9, 0.95, 1.0);
    const float3 skyBottom = float3(0.3, 0.5, 0.9);

    intersector<instancing, triangle_data> isect;
    isect.assume_geometry_type(geometry_type::triangle);

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

        for (uint depth = 0; depth < uniforms.maxDepth; ++depth) {
            ray r;
            r.origin = rayOrigin;
            r.direction = rayDir;
            r.min_distance = 0.001f;
            r.max_distance = 1e6f;

            intersection_result<instancing, triangle_data> result = isect.intersect(r, accelStructure);

            if (result.type == intersection_type::none) {
                float skyT = 0.5 * (rayDir.y + 1.0);
                radiance += throughput * mix(skyBottom, skyTop, skyT);
                break;
            }

            uint primId = result.primitive_id;
            float3 hitPoint = rayOrigin + rayDir * result.distance;
            float3 normal = faceNormalFor(primId, vertices);
            if (dot(normal, rayDir) > 0.0) normal = -normal;

            TriangleMaterial mat = triMaterials[primId];
            float3 albedo = float3(mat.color);

            if (mat.materialType == 1u) {
                // Mirror: deterministic reflection, no light sampling (a
                // specular surface has zero probability of the shadow ray
                // toward a delta light landing exactly on the reflection
                // vector - NEE simply doesn't apply here, same reason the
                // CPU renderer's own BSDFs skip NEE for specular lobes).
                rayDir = reflect(rayDir, normal);
                rayOrigin = hitPoint + normal * 0.001f;
                throughput *= albedo;
            } else {
                // Lambertian: next-event estimation against the one
                // directional light (shadow ray), then continue the path
                // via cosine-weighted hemisphere sampling for indirect
                // light. Two separate rays per bounce - direct (shadow)
                // and the continuation - is the standard NEE split this
                // project's own CPU path_integrator.h also uses.
                float ndotl = max(dot(normal, lightDir), 0.0);
                if (ndotl > 0.0) {
                    ray shadowRay;
                    shadowRay.origin = hitPoint + normal * 0.001f;
                    shadowRay.direction = lightDir;
                    shadowRay.min_distance = 0.001f;
                    shadowRay.max_distance = 1e6f;
                    intersection_result<instancing, triangle_data> shadowResult =
                        isect.intersect(shadowRay, accelStructure);
                    if (shadowResult.type == intersection_type::none) {
                        radiance += throughput * albedo * lightColor * ndotl * (1.0 / M_PI_F);
                    }
                }

                rayDir = cosineSampleHemisphere(normal, rngState);
                rayOrigin = hitPoint + normal * 0.001f;
                // Cosine-weighted sampling's pdf (cos(theta)/pi) cancels
                // the BSDF's own cos(theta)/pi exactly, leaving the flat
                // albedo below - textbook importance-sampled Lambertian,
                // not an approximation.
                throughput *= albedo;
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
