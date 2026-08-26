// OptiX type definitions and shared structures
// Shared between host and device code

#pragma once

#include <optix.h>
#include <cuda_runtime.h>

// Relative path (not a bare quoted include) so this resolves under both
// compile contexts that #include this file: MSBuild .cpp compiles (which do
// have src/shared on -I) and the recursive-path nvcc compile of
// optix_programs.cu (whose OptixFlags in build_optix.targets does NOT add
// src/shared to -I, unlike the wavefront compile flags). CPU_GPU-tagged and
// templated on T, so PointLightData<float>/SpotLightData<float>/
// DistantLightData<float> compile directly for device code - no reimplementation.
#include "../../src/shared/punctual_lights.h"
// Same relative-path reasoning as punctual_lights.h above. CloudMedium<T> is
// CPU_GPU-tagged and templated, so CloudMedium<float> compiles directly for
// device code (it already calls perlin_noise<T> from noise.h, itself
// CPU_GPU-tagged and already used on-device by optix_device_helpers.h's
// sample_texture() for TextureKind::Noise) - no device-side reimplementation
// needed, same as the punctual lights above.
#include "../../src/shared/cloud_medium.h"

#ifndef __CUDACC__
#include <stdexcept>
#include <string>
#endif

// Forward declarations
struct MaterialPOD;
struct LaunchParams;

// Ray payload for path tracing
struct PathTracingPayload {
	float3 attenuation;   // Color filter (throughput)
	float3 emission;      // Emitted light
	float3 scatterOrigin; // Next ray origin
	float3 scatterDir;    // Next ray direction
	unsigned int seed;    // Random number seed
	int depth;            // Current bounce depth
	bool scattered;       // True if ray scattered (not absorbed)
	// Denoiser guide-layer AOVs (recursive backend only) - only meaningful
	// on the primary ray (depth==0); every closest-hit/miss program packs
	// them unconditionally (p16-p21, see optix_raygen.h's own comment)
	// regardless of depth, since a hit program has no way to know which
	// bounce it's shading - raygen is the one that decides whether to
	// accumulate them, exactly mirroring how ray-differential dudx/dvdx
	// already only matters (and is only consumed) on depth==0.
	float3 albedo;
	float3 normal;
	// pbrt-v4's per-event refraction ratio (eta_i/eta_t, front_face ?
	// 1/ior : ior) for this bounce's scatter event - 1.0f (a no-op) unless
	// this hit was a genuine transmission through a dielectric interface.
	// optix_raygen.h accumulates eta_scale *= eta*eta across every bounce
	// (pbrt-v4 PathIntegrator: `if (bs->IsTransmission()) etaScale *=
	// Sqr(bs->eta);`) and folds it into the Russian Roulette throughput
	// test, matching CPU's camera.h eta_scale exactly - without it, RR
	// terminates transmission-heavy (glass) paths too aggressively. Packed
	// into payload register p22 (see optix_raygen.h): unlike p16-p21,
	// which every closest-hit/miss program packs unconditionally, only the
	// programs/branches that produce a real transmission event ever call
	// optixSetPayload_22 - everywhere else the register keeps whatever
	// value optix_raygen.h initialized it to before the trace call (1.0f),
	// the same "closest-hit only writes, unset means input survives"
	// convention p12 already relies on for __miss__ms.
	float eta;
};

// Shadow ray payload (minimal - just occlusion result)
struct ShadowPayload {
	bool occluded;  // True if ray hit anything (path is blocked)
};

// Which analytic shape a SphereData entry's near/far intersection math uses.
// Sphere (0) is the default - every pre-existing SphereData{} zero-init (the
// overwhelming majority of call sites) lands here with no changes needed, and
// is the quadratic ray/sphere test __intersection__sphere/__closesthit__sphere
// (optix_intersection_sphere.h) have always used. Box (1) is a standard
// ray/AABB slab test instead, for the few scenes whose medium boundary is
// actually an axis-aligned box rather than a sphere - e.g. scene 21's wax
// slab (see build_subsurface_slab_gpu's comment in scene_builder.cpp), which
// used to be approximated as a sphere sized to roughly match its footprint
// (bulging through the box's side walls while falling short of its floor/
// ceiling). Fixing this geometric mismatch was originally hypothesized to be
// the dominant cause of scene 21's ~32-38% CPU/GPU brightness gap, but a
// direct isolated-render A/B measurement (sphere vs box, GPU-recursive only,
// same seeds) showed the shape change moves average brightness by well under
// 1% (0.24428 -> 0.24566) - nowhere near enough to explain the ~30% gap. The
// gap's real dominant cause remains unexplained; this fix is still correct
// and worth keeping (it makes the rendered geometry actually match CPU's box
// instead of a bulging/undershooting sphere), it just isn't the fix for the
// brightness discrepancy itself. Deliberately additive: every
// field below is reused as-is regardless of shapeKind (materialIdx, the GAS/
// SBT/instancing/shadow-ray machinery, ...) - only the geometric test
// changes. center/center1/radius are unused (left zero) when shapeKind==Box;
// boxMin/boxMax are unused (left zero) when shapeKind==Sphere.
enum class GpuMediumShapeKind : int { Sphere = 0, Box = 1 };

// Sphere geometry data (custom primitive)
struct SphereData {
	float3 center;   // position at ray-time t=0 (the only position, for a static sphere)
	// Position at ray-time t=1, for motion blur (RTIOW-style linear "bounce"
	// interpolation, matching src/TheRestOfYourLife/sphere.h's moving-sphere
	// constructor: current_center = lerp(center, center1, ray.time())).
	// Left zero-initialized (garbage) for every static sphere in every scene
	// that doesn't use motion blur - always safe, because the intersection
	// program only ever multiplies it by a ray-time that is provably 0.0f
	// for those scenes (see optix_raygen.h), and lerp(a, b, 0) == a exactly
	// regardless of b. Only scenes that actually build moving spheres (with
	// center1 != center) need to also enable LaunchParams::motionBlurEnabled
	// and give their sphere GAS build 2 motion keys - see
	// OptiXRenderer::buildScene()'s sceneHasMotion_ detection. Unused (left
	// zero) for shapeKind==Box - no scene combines motion blur with a box
	// medium boundary, and center1==center==0 trivially keeps
	// sceneHasMotion_ detection from false-triggering on a box entry.
	float3 center1;
	float radius;
	int materialIdx;
	// See GpuMediumShapeKind's own comment above. Zero-init default (every
	// pre-existing `SphereData s{}` call site) is GpuMediumShapeKind::Sphere.
	GpuMediumShapeKind shapeKind;
	// Box (shapeKind==Box) world-space (equivalently object-space - every
	// scene using this is non-instanced, see GpuMediumShapeKind's comment)
	// min/max corners, mirroring the CPU box() representation exactly
	// (src/TheRestOfYourLife/quad.h) - an axis-aligned box from corner `a`
	// to corner `b`. Unused (left zero) for shapeKind==Sphere.
	float3 boxMin;
	float3 boxMax;
};

// Quad geometry data (custom primitive)
struct QuadData {
	float3 Q;  // Corner point
	float3 u;  // First edge vector
	float3 v;  // Second edge vector
	float3 normal;  // Precomputed normal
	float D;        // Plane constant
	float3 w;       // Cross product u x v
	int materialIdx;
};

// Bilinear patch geometry data (custom primitive) - a genuinely curved
// (non-planar in general) ruled surface through 4 corners, NOT a flat quad.
// p(u,v) = lerp(u, lerp(v,p00,p01), lerp(v,p10,p11)); p00=(u=0,v=0),
// p10=(u=1,v=0), p01=(u=0,v=1), p11=(u=1,v=1) - matches pbrt-v4's
// BilinearPatch / src/shared/bilinear_patch.h convention. See
// optix_intersection_bilinear_patch.h for the ray-intersection algorithm
// (Ramsey et al. 2004, ported from bilinear_patch.h's blp_intersect).
struct BilinearPatchData {
	float3 p00, p10, p01, p11;
	int materialIdx;
};

// Disk/Cylinder geometry data (custom primitives) - Shape "disk"/"cylinder"
// from a loaded .pbrt scene (see src/shared/pbrt_flatten.h's Disk/Cylinder
// and src/TheRestOfYourLife/disk_cylinder_hittable.h, the CPU backend these
// mirror). Unlike every other shape here, these are NOT baked to world
// space - a disk/cylinder is not rotation-invariant the way a sphere is, so
// baking would mean either re-deriving Sphere's own "warn under anisotropic
// scale" approximation or getting rotation silently wrong. Instead each
// carries its own object<->world affine transform (o2w/w2o, row-major 3x4 -
// same convention as SceneData::InstancePlacementGPU::transform), computed
// once host-side (pbrt_gpu_builder.h, via pbrt_scene::Matrix4::
// inverseAffine()) from the flat scene's raw double xform[16]. The device
// intersection/closest-hit programs (gpu/optix/optix_intersection_disk_
// cylinder.h) apply these manually - carrying the RAY into object space,
// exactly mirroring the CPU hittable's own technique - rather than through
// OptiX's per-GAS-instance transform, since neither shape is instanced
// (no ObjectInstance support for them yet, matching the CPU loader's own
// current scope): giving each one its own GAS+IAS-instance just to reach
// OptiX's instance-transform machinery would be heavier than applying the
// same 3x4 by hand in the two device programs that need it.
struct DiskData {
	float radius;       // outer radius
	float innerRadius;  // 0 for a solid disk
	float height;       // object-space z of the disk's plane
	float phiMax;       // azimuthal sweep, RADIANS (converted from pbrt's degrees host-side)
	int materialIdx;
	float o2w[12];  // object -> world
	float w2o[12];  // world -> object
};

struct CylinderData {
	float radius;
	float zMin, zMax;   // object-space Z extent (axis is object-space Z)
	float phiMax;       // azimuthal sweep, RADIANS
	int materialIdx;
	float o2w[12];
	float w2o[12];
};

// Triangle geometry data (native OptiX triangle, see optix_renderer.cpp's
// buildAccelerationStructure). Shading normal is per-vertex-interpolated
// (barycentric, via optixGetTriangleBarycentrics()) when the source mesh
// had "vn" data - n0/n1/n2 - and falls back to the flat geometric normal
// cross(e1,e2) when hasNormals is false (e.g. scene 37's procedural
// icosahedron, or any mesh scene whose source .obj has no vn lines - most
// of them; Suzanne, scene 45, was the first to actually exercise this
// path). Mirrors src/TheRestOfYourLife/triangle.h's CPU
// has_normals()-gated interpolation exactly.
// uv0/uv1/uv2: per-vertex texture coordinates ("vt" data), barycentric-
// interpolated the same way as n0/n1/n2 when hasUVs is set (see
// optix_intersection_triangle.h) - feeds MaterialData::textureIdx image
// sampling for meshes with a real map_Kd texture (scenes 62/63's Sponza/
// Bistro; see scene_builder.cpp's load_obj_triangles_mtl_gpu()). Meshes
// with no "vt" data (or whose material has no textureIdx) leave hasUVs
// false and uv0-2 unused, matching hasNormals' same opt-in pattern.
struct TriangleData {
	float3 p0, p1, p2;
	float3 n0, n1, n2;
	float2 uv0, uv1, uv2;
	int materialIdx;
	bool hasNormals;
	bool hasUVs;
};

// Which array a sampled area light lives in, and therefore how to sample it.
//
// Explicitly : int, and the width is load-bearing. The host uploads one of
// these per light and the device reads the same buffer back through a
// pointer; when those two widths disagreed (this was a bool* over an int
// buffer) only light 0 landed on its own entry - lights 1..3 read the upper,
// always zero, bytes of light 0's value and so all looked like the zero kind.
// A sphere light misread that way is then looked up in params.quads, which in
// a scene with no quads at all is an out-of-bounds read: an illegal memory
// access that kills the launch outright, not a subtle shading difference.
// optix_renderer.cpp static_asserts the two widths against each other at the
// upload site; wavefront_types.h and sppm_types.h carry their own copies of
// the pointer and must stay this type too.
//
// Quad is deliberately 0 so the historical false==quad / true==sphere
// encoding is preserved for anything still reasoning in those terms.
enum class GpuLightKind : int {
	Quad = 0,
	Sphere = 1,
	// A single emissive triangle. Most pbrt area lights arrive as a pair of
	// triangles that pbrt_quadify.h rejoins into one parallelogram and which
	// therefore become Quad; this kind is for the ones that will not merge -
	// an odd triangle, a fan, anything non-parallelogram - which used to be
	// emitted as geometry that glows when hit but that next-event estimation
	// could not aim at, leaving the GPU image darker and noisier than the CPU
	// one for no reason the picture explained.
	Triangle = 2,
	// Shape "bilinearmesh" carrying an AreaLightSource - real published pbrt
	// scenes use this for non-planar/non-rectangular light panels (e.g.
	// sportscar-area-lights.pbrt's studio softboxes), which pbrt_quadify.h
	// cannot fold into a Quad because they are not necessarily coplanar.
	// Same "used to glow but not be aimable" gap Triangle closed, for the one
	// shape kind that gap didn't cover.
	BilinearPatch = 3,
	// Shape "disk"/"cylinder" carrying an AreaLightSource - same "used to
	// glow but not be aimable" gap Triangle/BilinearPatch closed, for the
	// last two shape kinds that gap didn't cover. Sampling/pdf go through
	// optix_disk_cylinder_helpers.h's dc_sample_disk/dc_pdf_disk (real
	// device-safe ports of src/shared/shapes.h's DiskShape<T>/pdf_from, not
	// direct template instantiation - see that header's own comment for why).
	Disk = 4,
	Cylinder = 5,
};

// Material types
enum class MaterialType : int {
	Lambertian = 0,
	Metal = 1,
	Dielectric = 2,
	DiffuseLight = 3,
	RoughDielectric = 4,
	Conductor = 5,           // GGX VNDF + complex Fresnel (pbrt-v4 ConductorBxDF)
	CoatedDiffuse = 6,       // rough dielectric coat over Lambertian (pbrt-v4 CoatedDiffuseBxDF)
	ThinDielectric = 7,      // zero-thickness glass slab (pbrt-v4 ThinDielectricBxDF)
	CoatedConductor = 8,     // rough dielectric coat over GGX conductor (pbrt-v4 CoatedConductorBxDF)
	DiffuseTransmission = 9,  // diffuse reflection + diffuse transmission (pbrt-v4 DiffuseTransmissionBxDF)
	NormalizedFresnel   = 10, // Fresnel-weighted diffuse reflection (pbrt-v4 NormalizedFresnelBxDF)
	// Homogeneous participating medium (pbrt-v4/RTIOW constant_medium):
	// closed-form free-path (Beer-Lambert) sampling + Henyey-Greenstein
	// phase function, matching src/TheRestOfYourLife/constant_medium.h +
	// src/shared/volume_scattering.h - only usable on sphere geometry
	// (the boundary shape), reuses MaterialData.ior as sigma_t (extinction
	// coefficient) and MaterialData.fuzz as the HG asymmetry g.
	Medium = 11,
	// Marschner/Chiang fiber scattering (pbrt-v4 HairBxDF, src/shared/
	// bxdfs_hair.h) - matches src/TheRestOfYourLife/hair_material.h exactly,
	// including its "shading normal as fiber-tangent proxy" simplification
	// (no literal fiber geometry - src/shared/shapes.h's CurveShape is unused
	// dead code, never wired to any scene). Sphere-only, since that's the
	// only geometry any scene ever applies it to (scene 19's 5 hair spheres).
	// Reuses MaterialData.albedo as sigma_a (RGB absorption), .fuzz as
	// beta_m, .ior as eta, .eta_c.x as beta_n, .eta_c.y as alpha_deg.
	Hair = 12,
	// Sphere simultaneously a dielectric surface AND an internal
	// participating medium - matches src/TheRestOfYourLife/scenes_book.h's
	// build_final_scene() trick of adding the SAME boundary sphere to the
	// CPU world twice (once as a plain dielectric, once wrapped in
	// constant_medium): a ray always hits the dielectric surface first when
	// entering from outside (the medium's sampled hit distance can never be
	// closer than the entry surface), but on the very next bounce - now
	// travelling inside, hitting this sphere's exit surface - it may
	// scatter off the medium before reaching that exit. Handled inline in
	// optix_intersection_sphere.h (needs the same shape-specific near/far
	// re-intersection Medium above already does), not through
	// shade_material(). Reuses MaterialData.ior as the dielectric index of
	// refraction (Dielectric's own field), .albedo as the medium's
	// single-scatter color and .fuzz as its HG asymmetry g (Medium's own
	// fields), and .eta_c.x - otherwise unused outside Conductor/Hair - as
	// sigma_t (extinction coefficient), since this is the only material
	// type that needs a dielectric IOR and a medium's sigma_t/g/albedo at
	// the same time.
	DielectricMedium = 13,
	// Lambertian with a perturbed shading normal, sourced from a tangent-
	// space RGB normal-map texture (MaterialData.textureIdx) - matches
	// src/TheRestOfYourLife/normal_map_materials.h's normal_map_material
	// wrapper exactly, but collapsed into one material instead of CPU's
	// "wrapper delegates to inner material after perturbing rec.normal"
	// pattern: handled inline in optix_intersection_sphere.h (needs the
	// sphere's own tangent/dpdu, computed the same way CPU's sphere.h
	// does), which then calls shade_material() with the perturbed normal
	// and a temporary MaterialType::Lambertian view of this same data -
	// reusing that case's existing NEE/MIS logic verbatim rather than
	// duplicating it. Reuses MaterialData.albedo as the inner Lambertian's
	// flat color (CPU's scene never combines this with an albedo texture,
	// so .textureIdx is unambiguously "the normal map" here, not "the
	// albedo texture" the way it means for a plain Lambertian).
	// Sphere-only, matching CPU's only current use (scene 20's one
	// normal-mapped sphere) - see also this codebase's confirmed-empirically
	// finding that scene 20's OTHER normal-perturbation technique (bump-
	// mapping the back wall/box via bump_map_material + noise_texture) is
	// a no-op on CPU itself (a p-only procedural texture can't produce a
	// nonzero (u,v) finite-difference gradient), so it isn't ported here -
	// the GPU builder renders those surfaces as plain flat Lambertian,
	// which already matches CPU's actual rendered pixels exactly.
	NormalMappedLambertian = 14,
	// Disney/pbrt-v4-style multi-lobe BSDF (diffuse + GGX specular + GGX
	// clearcoat, metallic-blended Fresnel) - matches src/TheRestOfYourLife/
	// principled_material.h exactly by directly instantiating the same
	// CPU_GPU-tagged src/shared/bxdfs_principled.h::PrincipledBxDF<T> struct
	// device-side (T=float), the same pattern already used for
	// MaterialType::Hair's HairBxDF<T> - see sample_principled_material() in
	// optix_device_helpers.h. Like Hair, CPU's principled::scatter() sets
	// skip_pdf=true (its returned weight already folds in the BSDF value,
	// cosine term, and multi-lobe balance-heuristic PDF), so this is handled
	// as a specular-style bounce (no NEE/MIS) exactly like Hair/Metal/
	// Dielectric/Conductor.
	// Field reuse (no new MaterialData fields needed): albedo = base color,
	// ior = ior, fuzz = roughness, eta_c.x = metallic, eta_c.y = clearcoat,
	// eta_c.z = clearcoat_rough (k_c unused - Principled isn't a Conductor).
	// Sphere-only, matching CPU's only current use (scene 18's 7 showcase
	// spheres).
	Principled = 15,
	// Heterogeneous, procedural Perlin-FBm-density participating medium
	// (pbrt-v4 CloudMedium, src/shared/cloud_medium.h) - matches
	// src/TheRestOfYourLife/cloud_medium_hittable.h exactly (delta-tracking/
	// null-collision free-path sampling using the medium's majorant sigma_t,
	// since density varies per point unlike Medium's single fixed sigma_t).
	// Unlike every other MaterialType, this one's real parameters (the
	// world-to-medium affine transform, density/wispiness/frequency, and
	// sigma_a/sigma_s) don't fit MaterialData's spare per-material bytes, so
	// this only stores an INDEX (cloud_medium_extra.cloudMediumIdx) into
	// LaunchParams::cloudMediums, mirroring the RealisticCamera lens-table
	// pattern (GpuLensElement/GpuExitPupilBounds) rather than any other
	// MaterialType's direct field reuse. .medium_albedo/.g are still reused
	// directly (single-scatter color, HG phase asymmetry) since those DO fit
	// and every other Medium-family type already uses those same two names.
	// Sphere-only for the same reason Medium is: only usable on the boundary
	// shape the intersection code re-derives entry/exit roots for.
	CloudMedium = 16,
	// Heterogeneous participating medium with a real per-voxel R/G/B
	// scattering grid (pbrt-v4 RGBGridMedium, src/shared/rgb_grid_medium.h) -
	// matches src/TheRestOfYourLife/rgb_grid_medium_hittable.h's delta
	// tracking, except GPU uses one single GLOBAL majorant for the whole
	// medium (GpuRgbGridMedium::sigma_maj) rather than CPU's real per-voxel
	// DDA majorant grid - a deliberate simplification (see gpu_rgb_grid_
	// trilinear's comment in optix_intersection_sphere.h) to keep GPU delta
	// tracking algorithmically simple. Like CloudMedium, this only stores an
	// INDEX (rgb_grid_medium_extra.rgbGridMediumIdx) into
	// LaunchParams::rgbGridMediums, one level further than CloudMedium even:
	// RGBGridMediumData<T> can't be used directly on device at all (its
	// SampledGrid<T> members use std::vector/std::optional internally), so
	// GpuRgbGridMedium is a from-scratch flat metadata struct (below) whose
	// actual voxel data lives in a SEPARATE shared flat buffer
	// (LaunchParams::rgbGridData), indexed via GpuRgbGridMedium::dataOffset.
	// Sphere-only, same reason as Medium/CloudMedium.
	RgbGridMedium = 17,
	// Real tabulated BSSRDF (pbrt-v4 TabulatedBSSRDF / SubsurfaceMaterial),
	// on BOTH GPU backends. Mirrors src/TheRestOfYourLife/material_pbrt.h's
	// `class subsurface` + camera.h::sample_bssrdf_exit()'s 3-axis-MIS
	// probe/exit-point algorithm exactly, replacing the flat-diffuse
	// fallback pbrt_gpu_builder.h used to emit for pbrt_flatten::
	// MaterialKind::Subsurface. Landed in two phases: the recursive backend
	// first (optix_bssrdf.h / optix_probe_hit.h - see shade_material()'s own
	// comment, optix_device_helpers.h), then the wavefront backend as a real
	// probe-walk stage (BssrdfProbeWorkItem -> wavefront_probe.h's
	// wf_bssrdf_probe_walk() -> resolve_bssrdf_exit() in wavefront_kernels.cu)
	// - both now share the same tabulated BSSRDFTable data and algorithm, at
	// parity with each other and with CPU.
	//
	// Field reuse: .albedo stays `m.color` (the flat-gray fallback color,
	// 0.5/0.5/0.5 by pbrt_flatten.h's own default) - a real texture is never
	// carried by a Subsurface material in this loader, so this slot is only
	// ever read as the CPU-parity fallback color, never as a real albedo.
	// .ior is eta (index of refraction, already assigned generically before
	// this type's own switch case). sigma_a/
	// sigma_s (RGB absorption/scattering coefficients, ALREADY scale-
	// multiplied - see pbrt_flatten::Material::sigma_a/sigma_s's own comment)
	// live in the emission/transmittance union and the k_c-aliased union
	// respectively (see MaterialData's own field comments below) - both
	// otherwise unused by this material kind (no emission, no Conductor k).
	// .textureIdx is repurposed as an index into LaunchParams::bssrdfTables
	// (this material's precomputed BSSRDFTable, built once per unique (g,eta)
	// pair at scene-build time - see pbrt_gpu_builder.h) rather than a real
	// texture index; Subsurface materials never carry a real texture in this
	// loader, so this reuse is unambiguous on the recursive backend, which is
	// the only backend that ever reads it as anything but -1.
	Subsurface = 18,
	// Real tabulated measured-BRDF (pbrt-v4 Material "measured" - a real
	// gonioreflectometer-measured BRDF loaded from a binary .bsdf tensor
	// file, Dupuy & Jakob's format), BOTH GPU backends. Mirrors
	// src/TheRestOfYourLife/material_pbrt.h's `class measured` exactly: a
	// single VNDF-importance-sampled BxDF sample per hit (MeasuredBxDF::
	// sample_f - src/shared/measured_bxdf.h), no Fresnel entry/exit split,
	// no probe walk - same *shape* as MaterialType::Metal/Conductor/Hair/
	// Principled (sample once, get a direction + full skip_pdf-style
	// throughput weight, continue as a non-NEE specular-style bounce). This
	// replaces the flat-diffuse fallback pbrt_gpu_builder.h used to emit for
	// pbrt_flatten::MaterialKind::Measured before this MaterialType existed
	// (see gpu/optix/optix_measured_bxdf.h / wavefront_measured_bxdf.h for
	// the device math, and optix_device_helpers.h's shade_material() /
	// wavefront_kernels.cu's evaluate_materials() for the two backends'
	// dispatch).
	//
	// Field reuse: .textureIdx is repurposed as an index into
	// LaunchParams::measuredTables (this material's flattened 5-sub-table
	// set - see GpuMeasuredTable below), built once per unique resolved
	// .bsdf file path at scene-build time (getOrBuildMeasuredTable(),
	// pbrt_gpu_builder.h) - same "index into a shared LaunchParams array"
	// convention as MaterialType::Subsurface's own textureIdx reuse just
	// above; a measured material never carries a real texture in this
	// loader, so this reuse is unambiguous on both backends. No other
	// MaterialData field is used - the query wavelengths (612/549/465nm,
	// approximating sRGB primaries, matching `class measured`'s kLambdaR/G/B
	// exactly) are fixed constants in the device code, not per-material data.
	Measured = 19,
	// GGX microfacet metal with a flat RGB tint standing in for Fresnel, no
	// complex IOR (pbrt-v4/RTOW rough_metal, src/shared/bxdfs_conductor.h's
	// RoughMetalBxDF) - matches src/TheRestOfYourLife/material_pbrt.h's
	// `class rough_metal` exactly, and is NOT the same model as
	// MaterialType::Metal above (that one is a fuzz-perturbed mirror with no
	// real microfacet distribution at all - CPU's plain `class metal`, a
	// different class). Scenes B1 (RoughMetalSpheres) and B2
	// (CornellRoughMetal) previously called add_metal() for their "rough
	// metal" spheres/box, which silently rendered the wrong model on GPU
	// (confirmed via the CPU builder always using `rough_metal`) - fixed by
	// routing those two scenes through add_rough_metal() instead.
	// Field reuse: .albedo = flat tint, .fuzz (aliased .roughness) = GGX
	// roughness (RoughnessToAlpha'd device-side, same as Conductor/
	// RoughDielectric/CoatedDiffuse/CoatedConductor) - no eta_c/k_c, unlike
	// Conductor, since RoughMetalBxDF has no real Fresnel model.
	RoughMetal = 20,
	// Heterogeneous participating medium with a single-channel per-voxel
	// scalar density grid (pbrt-v4 GridMedium/"uniformgrid",
	// src/shared/sampled_grid.h's GridMediumData<T>) - matches
	// src/TheRestOfYourLife/grid_medium_hittable.h's delta tracking, the
	// single-channel twin of MaterialType::RgbGridMedium just above (same
	// "one global majorant, not CPU's real per-voxel DDA majorant grid"
	// deliberate GPU simplification - see that enumerator's own comment).
	// Only stores an INDEX (grid_medium_extra.gridMediumIdx) into
	// LaunchParams::gridMediums, same "flat metadata struct + separate flat
	// voxel-data buffer (LaunchParams::gridData)" shape as RgbGridMedium,
	// since GridMediumData<T> can't be used directly on device either (same
	// std::vector-backed SampledGrid<T> problem). Sphere-only, same reason
	// as Medium/CloudMedium/RgbGridMedium.
	GridMedium = 21,
	// Stochastic two-material blend (pbrt-v4 MixMaterial) - matches
	// src/TheRestOfYourLife/material_pbrt.h's `mix_material`/`branch_hash01`
	// exactly: at each shading point, deterministically (hashed from the
	// world-space hit point, NOT a fresh random draw - see branch_hash01's
	// own CPU-side comment for why: scatter()/scattering_pdf()/shadow-ray
	// classification must all agree about which sub-material a given
	// scattering event used) picks sub-material A or B and delegates
	// entirely to it - not a real blended BxDF evaluation. This MaterialType
	// itself has NO shading case anywhere; every shape's closest-hit/
	// intersection program and every shadow any-hit program resolves it
	// (mix_extra below -> LaunchParams::materials[chosen index], looping
	// while the result is itself another Mix, capped the same kMaxMixDepth
	// pbrt_gpu_builder.h's build-time resolveMixColor()/recursive
	// makeMaterial() already use) to a REAL MaterialType before any other
	// mat.type branch runs - see resolve_mix_material()/wf_resolve_mix_
	// material() in optix_device_helpers.h/wavefront_kernels.cu. Shape-
	// agnostic (unlike Hair/Subsurface/Medium-family): it never needs its
	// own entry in material_requires_sphere_only_handling()/wf_material_
	// requires_sphere_only_handling(), since resolution always happens
	// before those checks see the (by-then-real) resolved type.
	Mix = 22
};

// The single canonical list of MaterialTypes GPU SPPM's camera/photon-pass
// raygens (gpu/optix/sppm_programs.cu) actually implement BSDF sampling for
// - Lambertian/DiffuseLight handled directly, RoughDielectric/Metal/
// Dielectric/Conductor via sppm_is_delta_material()/
// sppm_sample_delta_material()'s explicit dispatch. Any MaterialType NOT in
// this list falls through sppm_programs.cu's own "treat as Lambertian"
// default, silently reading that material's (possibly unset) albedo union
// slot - which is exactly the class of bug gpu/optix/optix_interface.cpp's
// sppm_gpu_unsupported_reason() exists to reject a scene for BEFORE it can
// happen, rather than let it render silently wrong.
//
// Defined here, next to MaterialType's own definition, and used by both
// sppm_gpu_unsupported_reason() (host-side rejection check,
// optix_interface.cpp) and README/comment cross-references, SPECIFICALLY so
// there is exactly one place to update when sppm_programs.cu's own
// dispatch gains a new case - previously optix_interface.cpp carried its
// own separately-hand-maintained copy of this set with nothing forcing it
// to stay in sync with the device code it was describing.
inline bool sppm_gpu_material_supported(MaterialType t) {
	switch (t) {
	case MaterialType::Lambertian:
	case MaterialType::DiffuseLight:
	case MaterialType::RoughDielectric:
	case MaterialType::Metal:
	case MaterialType::Dielectric:
	case MaterialType::Conductor:
	case MaterialType::RoughMetal:
	case MaterialType::DiffuseTransmission:
		return true;
	default:
		return false;
	}
}

// GPU flat-array mirror of one PiecewiseLinear2D<Dimension> instance (see
// src/shared/piecewise_linear_2d.h's private members m_nx/m_ny/m_ps/m_pst/
// m_pv/m_data/m_mcdf/m_ccdf, exposed read-only via the const accessors added
// there for exactly this purpose - see that file's own comment). Used by
// MaterialType::Measured's device math (gpu/optix/optix_measured_bxdf.h /
// wavefront_measured_bxdf.h). One instance per PiecewiseLinear2D sub-table
// (ndf/sigma/vndf/luminance/spectra - see GpuMeasuredTable below); `dim`
// records which of the three Dimension values (0, 2, or 3) this particular
// sub-table was built at, since a MeasuredBRDFData mixes all three (unlike
// GpuBssrdfTable, which only ever describes one shape).
struct GpuPL2DTable {
	int nx, ny;                  // XSize()/YSize()
	int dim;                     // Dimension: 0 (ndf/sigma), 2 (vndf/luminance), 3 (spectra)
	int param_res[3];            // m_ps[i] per axis; unused axes (i >= dim) are 1
	int param_stride[3];         // m_pst[i] per axis; unused axes are 0
	// Offset into LaunchParams::measuredParamValues, param_res[i] floats
	// starting here, per axis; -1 for unused axes (i >= dim).
	int param_value_offset[3];
	int data_offset;             // offset into ::measuredData, size = slices*nx*ny
	// offset into ::measuredMcdf/::measuredCcdf, or -1 for a table with no
	// CDF (build_cdf=false - ndf/sigma/spectra; only vndf/luminance have one).
	int mcdf_offset;
	int ccdf_offset;
};

// One measured-BRDF material's full table set - mirrors MeasuredBRDFData
// (src/shared/measured_bxdf.h) exactly: ndf/sigma unconditional
// (Dimension=0, Eval()-only), vndf/luminance conditioned on (phi_i,theta_i)
// (Dimension=2, real CDFs, Sample()-only - this material samples exactly
// once per hit, see MaterialType::Measured's own comment), spectra
// conditioned on (phi_i,theta_i,lambda) (Dimension=3, Eval()-only). Built
// once per unique resolved .bsdf file path (pbrt_gpu_builder.h's
// getOrBuildMeasuredTable(), deduped the same way getOrBuildBssrdfTable()
// dedupes by (g,eta) - the sportscar scene's ilm_l3_37_metallic_spec.bsdf is
// referenced by 4 materials).
struct GpuMeasuredTable {
	GpuPL2DTable ndf, sigma, vndf, luminance, spectra;
	int isotropic;   // bool as int - see MeasuredBRDFData::isotropic
};

// Device-uploaded mirror of src/shared/bssrdf.h's BSSRDFTable, for the
// recursive GPU backend's MaterialType::Subsurface probe walk (see
// shade_material()'s own comment and gpu/optix/optix_bssrdf.h). Built ONCE
// per unique (g,eta) pair at scene-build time by calling the existing CPU
// ComputeBeamDiffusionBSSRDF() and uploading the resulting arrays - not a
// device-side computation (see pbrt_gpu_builder.h). Multiple Subsurface
// materials commonly share one table (same g/eta, different sigma_a/sigma_s
// - e.g. this codebase's own dragon_10/50/250 "scale" scenes, which all use
// eta=1.5, g=0 and differ only in scale), so tables are deduplicated and
// referenced by index (MaterialData::textureIdx, repurposed - see
// MaterialType::Subsurface's own comment) rather than embedded per-material,
// mirroring the CloudMedium/RgbGridMedium "index into a LaunchParams array"
// convention rather than GoniometricLightGPU/ProjectionLightGPU's inline-
// array convention, since a table (100 rho x 64 radius by default) is much
// larger than those lights' tiny images.
//
// rho_eff (pbrt-v4's per-row effective albedo, used by SubsurfaceFromDiffuse
// for CPU-only reflectance+mfp inversion - not needed here, see
// pbrt_flatten.h/material_pbrt.h) is NOT uploaded: it is exactly
// profile_cdf[row*n_radius + (n_radius-1)] (integrate_catmull_rom's own
// returned total equals the CDF's own last entry - see src/shared/
// sampling_2d.h), so gpu_bssrdf_pdf_sr() (optix_bssrdf.h) recovers it from
// profile_cdf directly instead of uploading a fourth per-table array.
struct GpuBssrdfTable {
	int n_rho;
	int n_radius;
	int rho_offset;      // into LaunchParams::bssrdfRhoSamples
	int radius_offset;   // into LaunchParams::bssrdfRadiusSamples
	// profile and profile_cdf share this same offset and n_rho*n_radius
	// layout (row i = rho_samples[i], column j = radius_samples[j]) into
	// their own separate flat buffers (LaunchParams::bssrdfProfile /
	// bssrdfProfileCdf) - one offset serves both since every table's rows
	// are built and stored in lockstep (see pbrt_gpu_builder.h).
	int profile_offset;
};

// Heterogeneous per-voxel R/G/B scattering grid metadata (GPU-only flat
// analog of RGBGridMediumData<T> - see MaterialType::RgbGridMedium's
// comment for why that struct can't be used directly on device). One
// instance per RgbGridMedium material, indexed via MaterialData::
// rgb_grid_medium_extra.rgbGridMediumIdx. Plain POD, no CPU_GPU tagging
// needed (its own fields are read directly by ordinary device code in
// optix_intersection_sphere.h/wavefront_kernels.cu, same as SphereData/
// QuadData already are).
struct GpuRgbGridMedium {
	float bounds_min[3], bounds_max[3];  // world-space AABB
	float mat[9];                         // row-major 3x3 world->medium transform
	float translate[3];
	int   nx, ny, nz;                     // grid resolution
	// Element offset into LaunchParams::rgbGridData - the R channel block
	// starts here, G at +nx*ny*nz, B at +2*(nx*ny*nz).
	int   dataOffset;
	float sigma_scale;  // overall density multiplier (matches CPU's sigma_scale)
	float sigma_maj;    // precomputed global majorant = sigma_scale * max
	                     // voxel value across all three channels, with a
	                     // small safety margin - see the GPU scene builder.
	float phase_g;       // Henyey-Greenstein asymmetry
};

// Single-channel twin of GpuRgbGridMedium above - see MaterialType::
// GridMedium's own comment.
struct GpuGridMedium {
	float bounds_min[3], bounds_max[3];  // world-space AABB
	float mat[9];                         // row-major 3x3 world->medium transform
	float translate[3];
	int   nx, ny, nz;                     // grid resolution
	// Element offset into LaunchParams::gridData - one flat block of
	// nx*ny*nz density values, no R/G/B split (unlike GpuRgbGridMedium's
	// dataOffset).
	int   dataOffset;
	float sigma_scale;  // sigma_a+sigma_s collapsed to one scalar (matches
	                     // GridMediumData<T>::sigma_a/sigma_s's own CPU-side
	                     // luminance() collapse - see pbrt_gpu_builder.h)
	float sigma_maj;    // precomputed global majorant = sigma_scale * max
	                     // density value, with a small safety margin - see
	                     // the GPU scene builder.
	float phase_g;       // Henyey-Greenstein asymmetry
};

// Texture kinds - see TextureData below. Matches CPU texture classes in
// src/TheRestOfYourLife/texture.h: image_texture, noise_texture,
// checker_texture, uv_checker_texture, fbm_texture, marble_texture, and
// mix_texture. mipmap_texture still has no GPU equivalent.
enum class TextureKind : int {
	Image = 0,
	Noise = 1,
	Checker = 2,
	// pbrt-v4's UV-tiled 2D "checkerboard" texture class - deliberately a
	// SEPARATE kind from Checker above rather than a variant of it: Checker
	// is this project's own original 3D world-space checker (parity of
	// floor(p.xyz * scale), keyed on the hit point), while pbrt's
	// checkerboard tiles by (u,v) * (uscale,vscale) - a materially
	// different result for the same "scale" the two shouldn't share a code
	// path over (see texture.h's uv_checker_texture, the CPU counterpart).
	UVChecker = 3,
	// pbrt-v4 FBmTexture (fractional Brownian motion) - matches CPU's
	// fbm_texture exactly (fbm_simple<T>, noise.h).
	FBm = 4,
	// pbrt-v4 MarbleTexture - matches CPU's marble_texture exactly (FBm-
	// perturbed sine wave through the same 9-knot RGB Bezier spline).
	Marble = 5,
	// pbrt-v4 SpectrumMixTexture, flat-literal tex1/tex2 only (see
	// pbrt_flatten.h's Material::hasMixReflectance comment) - matches CPU's
	// new mix_texture exactly.
	Mix = 6
};

// One entry per texture, indexed by MaterialData::textureIdx. Image
// textures point into a single shared flat 8-bit RGB pixel buffer
// (LaunchParams::texturePixels) via byte offset - same "flat device
// buffer, index in device code" convention as every other array in this
// codebase (see LaunchParams below). Matches src/TheRestOfYourLife/
// rtw_stb_image.h's own byte layout exactly (3 bytes/pixel, row-major).
struct TextureData {
	TextureKind kind;
	int pixelOffset;   // Image: byte offset into texturePixels. Unused otherwise.
	int width;         // Image: pixel width. Unused otherwise.
	int height;        // Image: pixel height. Unused otherwise.
	float noiseScale;  // Noise: scale param. Checker: 1/scale (checker_texture's own inv_scale). Unused otherwise.
	float3 color1;     // Checker/UVChecker: "even"/tex1 cell color. Mix: tex1. Unused otherwise.
	float3 color2;     // Checker/UVChecker: "odd"/tex2 cell color. Mix: tex2. Unused otherwise.
	float uScale;      // UVChecker: u-axis tile frequency (pbrt-v4 "uscale"). Unused otherwise.
	float vScale;      // UVChecker: v-axis tile frequency (pbrt-v4 "vscale"). Unused otherwise.
	float omega;       // FBm/Marble: persistence/roughness param. Unused otherwise.
	int   octaves;     // FBm/Marble: octave count. Unused otherwise.
	float marbleScale;     // Marble: spatial frequency multiplier (pbrt-v4 "scale"). Unused otherwise.
	float marbleVariation; // Marble: FBm displacement amplitude (pbrt-v4 "variation"). Unused otherwise.
	float mixAmount;   // Mix: blend weight, 0->color1 1->color2 (pbrt-v4 "amount"). Unused otherwise.
	// UVChecker/Mix only (NOT the world-space Checker kind, which has no
	// loader path that ever sets these): index into LaunchParams::textures
	// for a ONE-LEVEL-nested bare imagemap Texture bound to tex1/tex2 instead of a
	// flat literal (pbrt_flatten::Material::checkerTex1Filename/
	// checkerTex2Filename/mixTex1Filename/mixTex2Filename's own comments),
	// or -1 (the default, via `TextureData tex{};`'s aggregate-init - no
	// existing call site needed updating) to use color1/color2 directly.
	// Not itself recursive - a texture kind other than Image at this index
	// is never produced by the loader (see flatten()'s own one-level scope
	// cut), so no cycle/depth guarding is needed on GPU.
	int tex1ImageIdx = -1;
	int tex2ImageIdx = -1;
};

// Material data (packed for SBT).
//
// This struct is reused across all 16 MaterialTypes (see the field-reuse
// notes already on each MaterialType enumerator above), which used to mean
// every field's *meaning* depended entirely on which MaterialType was set
// on the same MaterialData instance - e.g. plain `.fuzz` meant Metal
// roughness, RoughDielectric/Conductor/CoatedDiffuse/CoatedConductor/
// Principled roughness, Medium/DielectricMedium's Henyey-Greenstein
// asymmetry g, or Hair's beta_m, entirely by convention/comment.
//
// Each field below is now an anonymous union of same-size alternatives,
// one name per material family that owns that storage - purely a set of
// alternate *names* for the exact same bytes (standard C++11 anonymous
// unions), not a layout change: sizeof/alignment/positional-brace-init
// order are all unchanged (the first-listed name in each union is what a
// positional aggregate-init like `{ MaterialType::Metal, albedo, fuzz,
// 0.0f, ... }` still binds to), and every pre-existing field name (.fuzz,
// .ior, .eta_c, ...) remains valid and reads/writes the identical bytes -
// this is why no reading code (shade_material(), wavefront_kernels.cu's
// parallel switch, optix_intersection_sphere.h's special-cased types) had
// to change: they still compile and run byte-for-byte identically. Only
// scene_builder.cpp's add_<type>() material factories (the sole writers,
// after the Phase 1 refactor that funneled all ~190 material-creation call
// sites through them) were updated to use the clearer names.
struct MaterialData {
	MaterialType type;

	union {
		float3 albedo;         // Lambertian/Metal/RoughDielectric(unused)/CoatedDiffuse: diffuse/base color
		float3 sigma_a;        // Hair: RGB absorption coefficient
		float3 medium_albedo;  // Medium/DielectricMedium: single-scatter color
		float3 reflectance;    // DiffuseTransmission: R (reflected diffuse color)
		float3 base_color;     // Principled: base color
		// Dielectric: transmission_filter (OBJ/.mtl "Tf") - multiplies the
		// refracted/transmitted contribution only, leaving reflection
		// clear/white. Defaults to (1,1,1) (no-op) via add_dielectric()'s
		// default param - see mesh.h's CPU dielectric class for the full
		// rationale (a flat per-surface tint, not volumetric absorption).
		float3 transmission_filter;
	};

	union {
		float fuzz;       // Legacy/generic name - still what most reading code uses
		float roughness;  // Metal/RoughDielectric/Conductor/CoatedDiffuse/CoatedConductor/Principled:
		                  //   authored roughness (RoughnessToAlpha conversion done in shading code)
		float g;          // Medium/DielectricMedium: Henyey-Greenstein phase asymmetry
		float beta_m;     // Hair: longitudinal roughness
	};

	union {
		float ior;      // Dielectric/RoughDielectric/ThinDielectric/CoatedDiffuse(coat)/
		                //   CoatedConductor(coat)/NormalizedFresnel/DielectricMedium/Principled: index of refraction
		float sigma_t;  // Medium: extinction coefficient
		float eta;      // Hair: fiber index of refraction
	};

	union {
		float3 emission;       // DiffuseLight: emitted radiance
		float3 transmittance;  // DiffuseTransmission: T (transmitted diffuse color)
		// Subsurface: RGB absorption coefficient (pbrt-v4 sigma_a, already
		// scale-multiplied - see pbrt_flatten::Material::sigma_a's comment).
		// Named bssrdf_sigma_a (not sigma_a - that name is already taken by
		// the OTHER union above, Hair's own RGB absorption field) so both can
		// coexist; shares this slot rather than `albedo` specifically so that
		// `albedo` can stay the flat-gray fallback color the wavefront
		// backend's Subsurface fallback case reads - see MaterialType::
		// Subsurface's own comment in optix_types.h.
		float3 bssrdf_sigma_a;
	};

	// Conductor complex IOR real part is this union's own, unambiguous
	// purpose; Hair/DielectricMedium/Principled each borrow the same 12
	// bytes for their own unrelated per-component data (see each nested
	// struct below - matches the field-reuse already on MaterialType's own
	// enumerator comments exactly).
	union {
		float3 eta_c;  // Conductor/CoatedConductor: real part η per R/G/B channel (pbrt-v4 ConductorBxDF)
		struct { float beta_n, alpha_deg, _hair_pad; } hair_extra;               // Hair
		struct { float sigma_t, _dielectric_medium_pad1, _dielectric_medium_pad2; } dielectric_medium_extra; // DielectricMedium
		struct { float metallic, clearcoat, clearcoat_rough; } principled_params; // Principled
		// CloudMedium: index into LaunchParams::cloudMediums (stored as a
		// float so it lives in this union without changing MaterialData's
		// layout - see MaterialType::CloudMedium's comment for why an index
		// rather than direct field reuse).
		struct { float cloudMediumIdx, _cloud_medium_pad1, _cloud_medium_pad2; } cloud_medium_extra;
		// RgbGridMedium: index into LaunchParams::rgbGridMediums - same
		// index-not-direct-field-reuse reasoning as cloud_medium_extra above.
		struct { float rgbGridMediumIdx, _rgb_grid_medium_pad1, _rgb_grid_medium_pad2; } rgb_grid_medium_extra;
		// GridMedium: index into LaunchParams::gridMediums - same
		// index-not-direct-field-reuse reasoning as cloud_medium_extra above.
		struct { float gridMediumIdx, _grid_medium_pad1, _grid_medium_pad2; } grid_medium_extra;
		// Mix: both sub-materials' indices into LaunchParams::materials
		// (stored as floats, same "index in an otherwise-unused union slot"
		// convention as cloudMediumIdx/rgbGridMediumIdx/gridMediumIdx above -
		// cast to int on read) plus the blend weight (pbrt-v4 "amount":
		// probability of B winning at any given shading point, matching CPU's
		// mix_material/pbrt_flatten::Material::mixWeight convention exactly).
		struct { float mixMaterialAIdx, mixMaterialBIdx, mixWeight; } mix_extra;
		// Dielectric/RoughDielectric: two-term Cauchy dispersion coefficients
		// (n(lambda) = cauchy_A + cauchy_B/lambda^2 - see src/shared/fresnel.h's
		// CauchyEta()/CauchyCoefficientsFromAbbe()), matching CPU's already-
		// shipped dielectric::make_dispersive()/rough_dielectric::
		// make_dispersive(). cauchy_A > 0.0f means "this material is
		// dispersive" - a real Cauchy A coefficient is always ~1.4-1.9, so 0
		// (the union's zero-initialized default) is a safe sentinel for "not
		// dispersive, use the flat .ior above", no separate bool needed. Note
		// this is a different shape than every other slot in this union
		// (eta_c/hair_extra/.../mix_extra above): those are selected purely
		// by MaterialType, with no secondary in-union value test - this is
		// the first one where MaterialType alone (Dielectric/RoughDielectric)
		// isn't enough and a runtime float comparison also has to fire. If a
		// future Dielectric/RoughDielectric variant ever needs cauchy_A <= 0
		// to be meaningful, this sentinel needs to become a real tag instead.
		struct { float cauchy_A, cauchy_B, _dispersive_pad; } dispersive_extra;
	};

	union {
		float3 k_c;         // Conductor/CoatedConductor only: imaginary part k per R/G/B channel
		// Subsurface: RGB scattering coefficient (pbrt-v4 sigma_s, already
		// scale-multiplied). k_c has no meaning for Subsurface (not a
		// Conductor), so this slot is free to reuse - same alternate-name
		// pattern as every other union in this struct.
		float3 bssrdf_sigma_s;
	};

	// Index into LaunchParams::textures, or -1 to use `albedo` directly
	// (every material before this field existed omitted it in brace-init,
	// which C++ aggregate-init already defaults to -1 via this default
	// member initializer - not 0 - so no existing call site needed
	// updating). Lambertian: albedo texture. NormalMappedLambertian reuses
	// this as the normal-map texture index instead (see that type's own
	// comment) - unambiguous since CPU never combines that material with a
	// real albedo texture too. Subsurface (recursive backend only) reuses it
	// again, as an index into LaunchParams::bssrdfTables - see
	// MaterialType::Subsurface's own comment for why this reuse is safe
	// (never a real texture for this material kind) and why the wavefront
	// backend's own Subsurface fallback case must NOT read this field as a
	// texture index.
	int textureIdx = -1;

	// Index into LaunchParams::textures for an opacity/alpha-cutout mask
	// (map_d in OBJ/.mtl), or -1 for none (the overwhelming majority of
	// materials) - see optix_intersection_triangle.h's __anyhit__triangle
	// and optix_anyhit_shadow.h's __anyhit__shadow_triangle. Independent of
	// textureIdx above (a material can have both a diffuse texture and a
	// separate alpha mask), and applies only to triangles today - no
	// sphere/quad/bilinear-patch scene sets it. Same default-member-
	// initializer trick as textureIdx: every brace-init call site gets -1
	// automatically, no existing call site needs updating.
	int alphaMaskTexIdx = -1;

	// DiffuseTransmission only: index into LaunchParams::textures for a
	// texture-bound "transmittance", or -1 to use the `transmittance` union
	// field directly - independent of textureIdx above, which this same
	// material kind reuses for its own texture-bound "reflectance"
	// (barcelona-pavilion's foliage binds both to the SAME bare imagemap in
	// practice, so they resolve to the same cached texture index there, but
	// the fields are independent for a scene that binds them differently).
	// Same default-member-initializer trick as textureIdx/alphaMaskTexIdx:
	// every brace-init call site gets -1 automatically, no existing call
	// site needs updating.
	int transmittanceTextureIdx = -1;

	// DiffuseLight only (matches CPU's diffuse_light::is_two_sided()): when
	// true, material_emission()/its wavefront equivalent emit from both
	// faces instead of gating on front_face. false (the default) preserves
	// every pre-existing light's one-sided behavior - no call site needed
	// updating.
	bool twoSided = false;

	// DiffuseLight or CoatedDiffuse, and only meaningful when textureIdx >=
	// 0: a flat scalar multiplier applied to the sampled texel at lookup
	// time (mirrors CPU's scaled_texture wrapper - see that class's own
	// comment). A flat-color DiffuseLight already bakes "scale" into
	// `emission` directly at build time and never reads this field.
	// CoatedDiffuse reuses this same field (not a dedicated one) for its own
	// "reflectance" bound to a "scale"-class Texture wrapping an imagemap
	// (barcelona-pavilion's own dominant pattern - see pbrt_flatten::
	// Material::textureScale's own comment) - same "a per-material scalar
	// multiplier over a sampled texture" concept, just read by a different
	// material kind's own shading code. Defaults to a no-op multiply so no
	// existing call site needed updating.
	float emissionScale = 1.0f;

	// pbrt-v4's "remaproughness" (default true): when true, `roughness`
	// above is a perceptually-remapped authored value that still needs
	// RoughnessToAlpha (sqrt) applied on-device before use as real GGX
	// alpha (Conductor/RoughDielectric/CoatedDiffuse/CoatedConductor/
	// RoughMetal); when false, `roughness` already IS the alpha value.
	// Defaults to true (the pre-existing unconditional-sqrt behavior), so
	// no existing call site needed updating. GPU has no anisotropic
	// roughness yet (see roughness_u/roughness_v's CPU-only status), so
	// this applies only to the isotropic `roughness` field.
	bool remapRoughness = true;
};

// Punctual (delta) light kinds - point/spot/distant. These are evaluated
// deterministically at every hit (pdf=0, no MIS, not part of lightIndices/
// aliasTable/lightKinds at all) rather than stochastically picked like
// the area lights, mirroring how src/TheRestOfYourLife/punctual_light_objects.h
// handles them on the CPU.
enum class PunctualLightKind : int {
	Point = 0,
	Spot = 1,
	Distant = 2,
	Goniometric = 3,
	Projection = 4
};

// Max image dimensions for goniometric/projection lights, stored inline in
// PunctualLightGPU (see below) rather than as a separately-allocated device
// buffer: light counts are tiny (typically 1-3 per scene) and PunctualLightGPU
// itself already lives behind a device POINTER sized to numPunctualLights at
// runtime (LaunchParams::punctualLights, not a fixed __constant__ array), so
// a fixed-size inline array per element avoids a second device-buffer-
// management path (alloc/upload/free, extra SBT plumbing) without risking any
// __constant__-memory budget. Raised from this codebase's original 32 (sized
// only for its own hand-built showcase scenes' synthetic 16x8/8x8 patterns)
// to 64 once real decoded profile/slide images could exceed that - pbrt-v4's
// own `imgtool makeequiarea` typically emits goniometric profiles in this
// range, and a real projection slide photo can be much larger still, so an
// oversized real image is downsampled (nearest-neighbor) to fit this cap at
// build time (pbrt_gpu_builder.h) rather than silently cropped or falling
// back to a flat approximation - see that file's own comment.
static constexpr int kGonioImageMaxDim = 64;
static constexpr int kProjImageMaxDim  = 64;

// GPU-side goniometric (IES-profile) point light. Mirrors the *evaluation*
// half of src/shared/goniometric_light.h's GoniometricLight<T> (sample_li +
// eval_I) - the CPU-only sample_le/pdf_le (light-tracing/BDPT) are not
// needed here since GPU rendering is NEE-only. image[] holds the same
// row-major nu*nv equal-area-square greyscale data the CPU struct owns as a
// std::vector<double>; the CPU is the source of truth for the pattern, this
// just holds a device-uploadable copy of it.
struct GoniometricLightGPU {
	float pos_x, pos_y, pos_z;
	float world_to_light[9];  // row-major 3x3 world->light rotation
	float ir, ig, ib;         // base intensity (candela)
	float scale;
	int nu, nv;                // nu*nv must be <= kGonioImageMaxDim^2
	float image[kGonioImageMaxDim * kGonioImageMaxDim];  // [v*nu+u], greyscale
};

// GPU-side projection (slide-projector) light. Mirrors the evaluation half
// of src/shared/projection_light.h's ProjectionLight<T> (sample_li +
// eval_I_rgb). `inv_tan` replaces the CPU struct's full 4x4 screenFromLight/
// lightFromScreen perspective matrices: make_perspective()'s matrix reduces
// exactly to screen_x = inv_tan*lx/lz, screen_y = inv_tan*ly/lz for a point
// (lz is already the homogeneous w after transform, per its [0,0,1,0] bottom
// row), so storing the one scalar 1/tan(fov/2) is sufficient and avoids
// needing Mat4/matrix-multiply plumbing on the device.
struct ProjectionLightGPU {
	float pos_x, pos_y, pos_z;
	float world_to_light[9];  // row-major 3x3 world->light rotation
	float scale;
	float hither;              // near-plane cutoff (pbrt-v4 default 1e-3)
	int nx, ny;                 // nx*ny must be <= kProjImageMaxDim^2
	float sb_xmin, sb_xmax, sb_ymin, sb_ymax;  // screen bounds
	float inv_tan;              // 1 / tan(fov_deg/2 in radians)
	float image_rgb[kProjImageMaxDim * kProjImageMaxDim * 3];  // [(v*nx+u)*3+c]
};

// A single punctual light, tagged by kind. Only the member matching `kind`
// is meaningful; the others are left default-constructed. Kept as inline
// structs rather than a union since light counts are tiny (typically 1-3
// per scene) and the member types are already CPU_GPU-tagged (or, for the
// two image-based kinds, trivially-copyable) PODs - no union/variant
// plumbing needed for this scale.
struct PunctualLightGPU {
	PunctualLightKind kind;
	PointLightData<float> point;
	SpotLightData<float> spot;
	DistantLightData<float> distant;
	GoniometricLightGPU gonio;
	ProjectionLightGPU proj;
};

// Alias table entry for power-weighted light sampling (pbrt-v4 PowerLightSampler pattern)
// Stored in GPU memory; sampled in O(1) by the device code.
struct GpuAliasEntry {
	float  q;      // Acceptance probability in [0,1]
	int    alias;  // Fallback index if rejected
	float  pdf;    // Probability mass for this entry (= power_i / total_power)
};

// Which ray-generation formula GpuCameraParams describes. Mirrors the CPU
// camera models this GPU camera type supports (src/shared/cameras.h's
// OrthographicCamera/PerspectiveCamera/SphericalCamera/RealisticCamera, plus
// the book-style default camera's defocus_angle/focus_dist thin-lens DOF
// extension).
enum class CameraKind : int {
	Perspective = 0,   // pinhole, optionally with thin-lens DOF (defocus_disk_u/v)
	Orthographic = 1,  // parallel projection, constant ray direction `w`
	Spherical = 2,     // 360-degree equirectangular panorama from a point
	Realistic = 3      // multi-element lens (pbrt-v4 RealisticCamera) - see GpuLensElement
};

// One spherical (or planar, for the aperture stop) lens surface - mirrors
// src/shared/cameras.h's RealisticCamera<T>::LensElement, already in metres
// and with the rear element's thickness already focus-adjusted (both done
// host-side by directly reusing RealisticCamera<float>'s own constructor -
// see scene_builder.cpp's case 36 - so device code never needs to run
// FocusThickLens/ComputeCardinalPoints itself).
struct GpuLensElement {
	float curvatureRadius;  // 0 = aperture stop
	float thickness;
	float eta;               // 0 = no interface (air on both sides don't apply eta)
	float apertureRadius;
};

// One radial exit-pupil bounding box slab - mirrors RealisticCamera<T>::Bounds2,
// precomputed host-side (RealisticCamera<float>::bound_exit_pupil, run once at
// scene-build time, same cost class as building a BVH or alias table).
struct GpuExitPupilBounds {
	float xMin, xMax, yMin, yMax;
	int   degenerate;  // 1 = no valid exit-pupil sample in this radial slab
};

// GPU flat mirror of sky_light's REAL (image + PiecewiseConstant2D)
// importance-sampling machinery (src/TheRestOfYourLife/sky_light.h +
// src/shared/piecewise_dist.h), built once per scene CPU-side (from the SAME
// decoded infinite-light image the flat-colour mean-brightness approximation
// already used - see pbrt_gpu_builder.h) and uploaded as flat device
// buffers, mirroring the measured-BRDF GPU work's "flatten CPU-built tables,
// upload flat buffers, port only the read-path math to device" pattern (see
// optix_measured_bxdf.h's own comment) - just simpler, since a scene has at
// most ONE infinite light (no per-material indexing/dedup needed the way
// GpuMeasuredTable/GpuBssrdfTable require).
//
// height <= 0 (the zero-init default) means "this scene has no real HDR
// image" (a constant-colour sky, or no infinite light at all) - every call
// site (gpu/optix/optix_sky_light.h / wavefront_sky_light.h and their
// callers) falls back to the existing flat-colour + uniform-sphere path in
// that case, unchanged from before this struct existed.
//
// Embedded directly in GpuCameraParams (not a separate LaunchParams array,
// unlike GpuMeasuredTable/GpuBssrdfTable) so it rides along on the SAME
// pass-through mechanism backgroundColor/shadowRayEpsilon already use to
// reach both GPU backends: recursive reads it via params.camera.skyDist,
// wavefront's kernels receive it as an explicit by-value parameter (same
// established pattern as GpuCameraParams itself already being passed by
// value into generate_camera_rays) extracted from the `camera` argument
// WavefrontPathTracer::render() already receives - see that function's own
// call sites. The raw device pointers below are only valid AFTER
// OptiXRenderer::buildScene() has uploaded them (mirrors CameraKind::
// Realistic's lensElements/exitPupilBounds pointers just below, which follow
// the identical "logical camera params built early, raw uploaded-table
// pointers patched in fresh inside render()" lifecycle).
//
// Deliberately NO in-class member initializers (unlike some other structs in
// this codebase): LaunchParams's own `__constant__ params` global (optix_
// device_helpers.h) requires every type it contains support trivial, non-
// dynamic initialization - nvcc rejects in-class initializers there with
// "dynamic initialization is not supported for a __constant__ variable".
// Every host-side use already zero-initializes the containing struct
// explicitly (`LaunchParams params = {};` / GpuCameraParams's own "always
// zero-initialized" convention - see its own comment), so height/scale/every
// pointer here are reliably 0/0.0f/nullptr by construction anyway.
struct GpuSkyDistribution {
	int width, height;              // image dimensions (nu, nv)
	float scale;                    // sky_light::scale (brightness multiplier)
	const float* imagePixels;       // width*height*3 floats, row-major RGB, linear
	const float* marginalCdf;       // height+1 floats
	const float* marginalFunc;      // height floats
	float marginalFuncInt;
	// Row r's slice starts at r*(width+1) (cdf) / r*width (func) - regular
	// stride since every row has the same width, unlike GpuPL2DTable's
	// per-axis offsets (which vary because different sub-tables have
	// different axis counts).
	const float* conditionalCdf;    // height*(width+1) floats
	const float* conditionalFunc;   // height*width floats
	const float* conditionalFuncInt; // height floats, one per row
};

// GPU camera parameters. `origin`/`lower_left_corner`/`horizontal`/`vertical`
// describe the perspective/orthographic viewport (same meaning as before this
// struct existed); the remaining fields are only meaningful for their
// respective CameraKind and are zeroed otherwise (scene_builder.cpp always
// zero-initializes this struct, so "zero defocus disk" reliably means
// "DOF disabled" for Perspective).
struct GpuCameraParams {
	CameraKind kind;
	float3 origin;
	float3 lower_left_corner;
	float3 horizontal;
	float3 vertical;
	float3 w;               // Orthographic: constant unit ray direction
	float3 defocus_disk_u;  // Perspective DOF: disk basis vector (zero = disabled)
	float3 defocus_disk_v;
	float3 su, sv, sw;      // Spherical: world-space camera basis (right, up, forward);
	                        // also reused by Realistic for its camera-to-world rotation
	                        // (su=right, sv=up, sw=forward) - `origin` above doubles as
	                        // its camera-to-world translation.
	int    sphericalMapping; // Spherical only: 0 = equirectangular (default, matches every
	                        // zero-initialized scene before this field existed), 1 = equal-
	                        // area (pbrt-v4's concentric-octahedral square-to-sphere map,
	                        // src/shared/sampling_sphere.h's EqualAreaSquareToSphere on CPU).

	// Realistic (CameraKind::Realistic): fixed scalars + device buffers for the
	// host-precomputed (focus-adjusted) lens table and exit-pupil bounds table -
	// see GpuLensElement/GpuExitPupilBounds. Null/zero for every other CameraKind.
	float film_half_x, film_half_y;  // physical film half-extents, metres
	float lens_rear_z;               // distance from film to rear lens element, metres
	int   numLensElements;
	int   numExitPupilBounds;
	GpuLensElement*     lensElements;
	GpuExitPupilBounds* exitPupilBounds;

	// Flat constant-color background for missed rays (default black =
	// existing behavior for every scene that doesn't set it). Piggybacked
	// onto this struct (rather than a new render()/LaunchParams field of
	// its own) since it's already threaded through the exact same call
	// chain this struct is. See optix_miss.h for why a flat color - not
	// full image-based env lighting - matches what the CPU renderer
	// actually does for every scene (scenes_advanced.h's build_hdri_sky()/
	// build_portal_sky() both return a solid-color sky_light; the
	// importance-sampled-image machinery in image_infinite_light.h is
	// unused dead code on the CPU side too, never wired to any scene).
	float3 backgroundColor;

	// Shadow-ray self-intersection offset override (world units), along the
	// shadow ray's own direction - see trace_shadow_ray()'s (optix_device_
	// helpers.h) own comment for why this offset exists at all. <= 0 (the
	// default, zero-init-safe for every scene that doesn't set it) means
	// "use the standard 0.01f". Some scenes need more: Sibenik Cathedral's
	// dense stone tracery (thin, closely-packed columns/arches) was
	// confirmed - by direct experiment, not guesswork - to false-occlude
	// most sky-NEE shadow rays at 0.01f, making the whole interior render
	// far darker than it should (GPU ~24% of CPU's brightness at matched
	// settings); bumping just this scene's epsilon to 0.5f closed the gap
	// to ~89%. A flat 0.01f isn't safe to raise for every scene (a smaller-
	// scale scene could get real light leaks from too large an offset), so
	// this is a per-scene override, not a global constant change.
	float shadowRayEpsilon;

	// Real importance-sampled HDR sky (LightSource "infinite" with an image) -
	// see GpuSkyDistribution's own comment. height<=0 (default) means "use
	// backgroundColor + uniform-sphere sampling", exactly as before this
	// field existed - every scene with a constant-colour sky, or no infinite
	// light at all, is completely unaffected.
	GpuSkyDistribution skyDist;

	// Pixel reconstruction filter (pbrt-v4 PixelFilter directive) - the GPU
	// twin of src/shared/filter.h's PixelFilterDispatch (CPU); gpu_filter_
	// evaluate() (optix_device_helpers.h) is the device-side port of its
	// evaluate(). filterKind 0 (the zero-init default, matching every scene
	// that never set one, hand-built or pbrt-loaded) = Gaussian, pbrt-v4's
	// own real default - see that enum's own values just below. NO in-class
	// member initializers here (unlike the rest of this struct's own
	// convention elsewhere) - `LaunchParams params` is declared `__constant__`
	// (optix_device_helpers.h), which nvcc requires to be static/zero-
	// initializable; a non-trivial default constructor (which member
	// initializers would introduce) breaks that with "dynamic initialization
	// is not supported for a __constant__ variable". filterSigma/filterTau
	// default to 0.0f via plain zero-init instead, which would be a
	// degenerate Gaussian - gpu_filter_evaluate() defends against exactly
	// this (sigma<=0 -> 0.5f, tau<=0 -> 3.0f) rather than relying on this
	// struct's own initialization. Radius is always 0.5 pixel on both
	// backends, matching CPU's own PixelFilterDispatch (no cross-pixel
	// splatting - see that class's own comment for why).
	int   filterKind;   // 0=gaussian 1=box 2=triangle 3=mitchell 4=sinc
	float filterB, filterC, filterSigma, filterTau;
};

// Launch parameters (passed to all OptiX programs)
struct LaunchParams {
	// Output
	float3* framebuffer;
	// Denoiser guide-layer AOVs (recursive backend only) - null unless the
	// caller is denoising this render (see OptiXRenderer::render()'s own
	// alloc site); nothing reads a null albedoBuffer/normalBuffer since
	// raygen only writes to them, never reads.
	float3* albedoBuffer;
	float3* normalBuffer;
	unsigned int width;
	unsigned int height;

	// Rendering parameters
	unsigned int samplesPerPixel;
	unsigned int maxDepth;
	unsigned int frameNumber;  // For random seed

	// Camera
	GpuCameraParams camera;

	// Scene
	OptixTraversableHandle traversable;  // Acceleration structure handle

	// Geometry arrays (device pointers)
	SphereData* spheres;
	unsigned int numSpheres;
	QuadData* quads;
	unsigned int numQuads;
	BilinearPatchData* bilinearPatches;
	unsigned int numBilinearPatches;
	// Disk/Cylinder (see DiskData/CylinderData's own comment) - supported on
	// both the recursive backend (Phase 4b) and the wavefront backend
	// (Phase 4c, WavefrontLaunchParams' own disks/cylinders fields).
	DiskData* disks;
	unsigned int numDisks;
	CylinderData* cylinders;
	unsigned int numCylinders;
	TriangleData* triangles;
	unsigned int numTriangles;

	// Where each IAS instance's primitives start in whichever flat array holds
	// them (`triangles` or `spheres`), indexed by OptixInstance::instanceId.
	//
	// Object instancing gives each instance definition its own GAS, and
	// optixGetPrimitiveIndex() restarts at 0 inside every GAS - so a primitive
	// index alone no longer identifies a primitive. Adding this base recovers
	// the global index while leaving one flat array per geometry type.
	//
	// One table serves both types because a GAS holds only ONE of them (OptiX
	// forbids mixing native triangles with custom AABB primitives), so an
	// instance id names exactly one array and the program that reads it already
	// knows which.
	//
	// The entry is SIGNED and -1 is a real sentinel, not "unused": it means
	// this instance's geometry is already in world space (the scene's own two
	// instances), which is also what gates the object-to-world normal transform
	// in the hit programs. Null means "no instancing in this scene", and every
	// lookup then uses base 0, which is exactly what a single-GAS scene has
	// always done. That is deliberate: the instancing path is inert until a
	// scene actually needs it, so it cannot change how existing scenes render.
	const int* instancePrimBase;

	// Material data
	MaterialData* materials;
	unsigned int numMaterials;

	// Heterogeneous cloud media (MaterialType::CloudMedium), indexed by
	// MaterialData::cloud_medium_extra.cloudMediumIdx. CloudMedium<float> is
	// used directly device-side (see optix_types.h's cloud_medium.h include
	// comment) rather than a separate GPU-specific mirror struct.
	CloudMedium<float>* cloudMediums;
	unsigned int numCloudMediums;

	// Heterogeneous per-voxel R/G/B media (MaterialType::RgbGridMedium),
	// indexed by MaterialData::rgb_grid_medium_extra.rgbGridMediumIdx. Each
	// GpuRgbGridMedium's actual voxel data is a separate slice of the flat
	// rgbGridData buffer below (see GpuRgbGridMedium::dataOffset).
	GpuRgbGridMedium* rgbGridMediums;
	unsigned int numRgbGridMediums;
	float* rgbGridData;
	unsigned int rgbGridDataCount;

	// Heterogeneous single-channel-density media (MaterialType::GridMedium),
	// indexed by MaterialData::grid_medium_extra.gridMediumIdx - same
	// flat-metadata-struct-plus-separate-voxel-buffer shape as
	// rgbGridMediums/rgbGridData above, just one channel instead of three
	// (see GpuGridMedium::dataOffset).
	GpuGridMedium* gridMediums;
	unsigned int numGridMediums;
	float* gridData;
	unsigned int gridDataCount;

	// Tabulated BSSRDF profile tables (MaterialType::Subsurface, recursive
	// backend only - see GpuBssrdfTable's own comment), indexed by
	// MaterialData::textureIdx (repurposed for this material kind). The four
	// flat arrays below are shared across every table; each GpuBssrdfTable
	// entry names its own slice via rho_offset/radius_offset/profile_offset.
	GpuBssrdfTable* bssrdfTables;
	unsigned int numBssrdfTables;
	float* bssrdfRhoSamples;
	float* bssrdfRadiusSamples;
	float* bssrdfProfile;
	float* bssrdfProfileCdf;

	// Real tabulated measured-BRDF tables (MaterialType::Measured, both GPU
	// backends - see GpuMeasuredTable's own comment), indexed by
	// MaterialData::textureIdx (repurposed - see MaterialType::Measured's own
	// comment). The four flat arrays below are shared across every table;
	// each GpuPL2DTable inside a GpuMeasuredTable names its own slice via
	// data_offset/mcdf_offset/ccdf_offset/param_value_offset.
	GpuMeasuredTable* measuredTables;
	unsigned int numMeasuredTables;
	float* measuredParamValues;
	float* measuredData;
	float* measuredMcdf;
	float* measuredCcdf;

	// Texture data (see TextureData above) - indexed by
	// MaterialData::textureIdx. texturePixels is one shared flat 8-bit RGB
	// buffer every Image-kind TextureData::pixelOffset points into.
	TextureData* textures;
	unsigned int numTextures;
	unsigned char* texturePixels;

	// Light sampling support (indices into sphere/quad arrays)
	int* lightIndices;          // Array of light primitive indices
	unsigned int numLights;     // Number of emissive lights in scene
	// Which array lightIndices[i] indexes - see GpuLightKind, whose comment
	// explains why the width of this pointee is load-bearing.
	const GpuLightKind* lightKinds;

	// Power-weighted alias table for light selection (pbrt-v4 PowerLightSampler)
	GpuAliasEntry* aliasTable;  // Device pointer to alias table (numLights entries)

	// Punctual (delta) lights: point/spot/distant. Separate from the
	// area-light arrays above - evaluated deterministically every hit,
	// not selected via the alias table.
	PunctualLightGPU* punctualLights;
	unsigned int numPunctualLights;

	// True only for scenes with moving spheres (SphereData::center1 != center
	// for at least one sphere - see OptiXRenderer::buildScene()). Tells the
	// raygen program to sample a random ray-time in [0,1] per pixel-sample
	// (RTIOW shutter convention) instead of always using 0.0f.
	bool motionBlurEnabled;

	// --stats device counters (see launcher/main.cpp's own "[STATS]" block
	// and src/shared/render_stats.h's CPU equivalent) - null unless --stats
	// was requested, same "null buffer means disabled, no separate bool
	// flag needed" convention albedoBuffer/normalBuffer above already use.
	// atomicAdd'd once per bounce iteration / shadow-ray optixTrace in
	// optix_raygen.h, read back to the host once after the launch completes
	// (a single 8-byte cudaMemcpy each, negligible next to the framebuffer
	// copy already happening). Russian Roulette makes these genuinely need
	// counting - a static width*height*samplesPerPixel*maxDepth formula
	// would overestimate whenever a path terminates early.
	unsigned long long* statsBounceRays;
	unsigned long long* statsShadowRays;
};

// Hit group data (per-geometry instance in SBT)
struct HitGroupData {
	// Sphere data (if sphere)
	SphereData sphere;

	// Quad data (if quad)
	QuadData quad;

	// Material index
	int materialIdx;

	// Geometry type marker
	enum class GeomType : int {
		Sphere = 0,
		Quad = 1
	} geomType;
};

// Ray types
//
// RAY_TYPE_PROBE (recursive backend only, Phase 1 BSSRDF): a small-payload
// closest-hit-only ray used by the MaterialType::Subsurface probe walk
// (bssrdf_probe_walk(), optix_device_helpers.h) to find candidate exit
// points along a probe segment, modeled on trace_shadow_ray()'s already-
// proven pattern of a sequential, non-nested optixTrace() call issued from
// within a hit program - see that function's own comment. Needs its own hit
// groups (see optix_probe_hit.h) rather than reusing RAY_TYPE_RADIANCE's,
// because it must report raw hit geometry (position/normal/material index)
// with NO shading/NEE/scattering - reusing the radiance hit groups would
// mean either running full shade_material() pointlessly on every probe
// segment step (wasteful and, worse, would recursively re-enter this same
// probe-walk machinery for a probe ray that happens to land on another
// Subsurface surface) or overloading the radiance ray's already-fully-
// packed 13-payload-register convention with an ambiguous "probe mode" flag.
// The wavefront backend has no equivalent and never traces this ray type.
enum {
	RAY_TYPE_RADIANCE = 0,
	RAY_TYPE_SHADOW = 1,
	RAY_TYPE_PROBE = 2,
	RAY_TYPE_COUNT = 3
};

// ============================================================================
// Shader Binding Table (SBT) Record Types
// Shared between OptixRenderer and all PathTracingStrategy implementations
// ============================================================================

/// @brief Generic SBT record with aligned header and user data
template<typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

using RaygenRecord   = SbtRecord<int>;           ///< Raygen program record
using MissRecord     = SbtRecord<int>;           ///< Miss program record
using HitGroupRecord = SbtRecord<HitGroupData>;  ///< Hit group record

// OptiX error checking macro (host-only)
#ifndef __CUDACC__
#define OPTIX_CHECK(call)                                                      \
	do {                                                                       \
		OptixResult res = call;                                                \
		if (res != OPTIX_SUCCESS) {                                            \
			fprintf(stderr, "OptiX call (%s) failed with code %d (line %d)\n", \
					#call, res, __LINE__);                                     \
			throw std::runtime_error(std::string("OptiX error: ") + #call);   \
		}                                                                      \
	} while (0)

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
	do {                                                                       \
		cudaError_t error = call;                                              \
		if (error != cudaSuccess) {                                            \
			fprintf(stderr, "CUDA call (%s) failed with code %d (line %d): %s\n", \
					#call, error, __LINE__, cudaGetErrorString(error));        \
			throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error)); \
		}                                                                      \
	} while (0)

// CUDA driver API error checking macro
#define CU_CHECK(call)                                                         \
	do {                                                                       \
		CUresult error = call;                                                 \
		if (error != CUDA_SUCCESS) {                                           \
			const char* errorStr;                                              \
			cuGetErrorString(error, &errorStr);                                \
			fprintf(stderr, "CUDA driver call (%s) failed with code %d (line %d): %s\n", \
					#call, error, __LINE__, errorStr);                         \
			throw std::runtime_error(std::string("CUDA driver error: ") + (errorStr ? errorStr : "unknown")); \
		}                                                                      \
	} while (0)
#endif // !__CUDACC__
