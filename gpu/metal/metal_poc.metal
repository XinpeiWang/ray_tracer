// metal_poc.metal
// Proof-of-concept Metal inline ray tracing kernel - see docs/METAL_GPU_FEASIBILITY.md
// section 7 ("Suggested next step"). Traces primary rays against a
// primitive acceleration structure via MSL's inline raytracing::intersector
// (no OptiX-style raygen/closest-hit pipeline - the kernel calls
// intersect() itself and branches on the result), shades with a single
// hardcoded directional light and per-triangle flat colour, and writes
// linear RGB to an output texture. Deliberately minimal: one material
// model, one light, no textures, no BVH tuning - the goal is proving the
// whole pipeline shape (accel structure -> inline intersect -> shade)
// works at all on Apple hardware, not feature coverage.

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

struct Uniforms {
    float3 cameraPos;
    float3 cameraForward;
    float3 cameraRight;
    float3 cameraUp;
    float tanHalfFov;
    float aspect;
    uint width;
    uint height;
};

// Per-triangle flat colour, indexed by primitive_id() - a real renderer
// would carry a material id + full BxDF; this POC only needs to prove a
// hit can be shaded at all.
struct TriangleColor {
    float3 color;
};

kernel void primaryRayKernel(
    texture2d<float, access::write> outTexture [[texture(0)]],
    instance_acceleration_structure accelStructure [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]],
    device const TriangleColor* triColors [[buffer(2)]],
    device const packed_float3* vertices [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height) return;

    // Pixel -> camera-space direction -> world-space ray, standard
    // perspective camera (same convention as this project's CPU
    // cameras.h: +x right, +y up, -forward down the view axis... here
    // cameraForward IS the view axis, kept positive for clarity in this
    // small POC rather than matching that sign convention exactly).
    float2 pixelNDC = (float2(tid) + 0.5) / float2(uniforms.width, uniforms.height);
    float2 screen = pixelNDC * 2.0 - 1.0;
    screen.y = -screen.y;
    screen.x *= uniforms.aspect;
    screen *= uniforms.tanHalfFov;

    float3 rayDir = normalize(uniforms.cameraForward
                               + screen.x * uniforms.cameraRight
                               + screen.y * uniforms.cameraUp);

    ray r;
    r.origin = uniforms.cameraPos;
    r.direction = rayDir;
    r.min_distance = 0.001f;
    r.max_distance = 1e6f;

    intersector<instancing, triangle_data> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    intersection_result<instancing, triangle_data> result = isect.intersect(r, accelStructure);

    if (result.type == intersection_type::none) {
        // Simple sky gradient so a miss is visually distinguishable from
        // a black/unlit hit - purely cosmetic, not part of the pipeline
        // being validated.
        float3 sky = mix(float3(0.9, 0.95, 1.0), float3(0.3, 0.5, 0.9), pixelNDC.y);
        outTexture.write(float4(sky, 1.0), tid);
        return;
    }

    // Flat face normal from the triangle's own two edges - intersection_
    // result carries no normal directly, only which primitive/vertices
    // were hit, so this POC derives it itself from the non-indexed vertex
    // buffer (3 consecutive vertices per triangle, matching how addQuad()
    // populated it host-side). A real renderer would carry interpolated
    // per-vertex normals; this POC only needs a plausible-looking
    // Lambertian term to confirm the hit geometry and shading path are
    // both wired correctly end to end.
    uint primId = result.primitive_id;
    float3 v0 = float3(vertices[primId * 3 + 0]);
    float3 v1 = float3(vertices[primId * 3 + 1]);
    float3 v2 = float3(vertices[primId * 3 + 2]);
    float3 worldNormal = normalize(cross(v1 - v0, v2 - v0));
    // Face the normal toward the ray origin so both sides of a triangle
    // shade sensibly regardless of winding.
    if (dot(worldNormal, r.direction) > 0.0) worldNormal = -worldNormal;

    float3 lightDir = normalize(float3(0.4, 0.8, 0.3));
    float ndotl = max(dot(worldNormal, lightDir), 0.0);

    float3 albedo = triColors[primId].color;
    float3 ambient = albedo * 0.15;
    float3 color = ambient + albedo * ndotl * 0.85;

    outTexture.write(float4(color, 1.0), tid);
}
