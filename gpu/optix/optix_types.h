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
};

// Shadow ray payload (minimal - just occlusion result)
struct ShadowPayload {
	bool occluded;  // True if ray hit anything (path is blocked)
};

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
	// OptiXRenderer::buildScene()'s sceneHasMotion_ detection.
	float3 center1;
	float radius;
	int materialIdx;
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
	Principled = 15
};

// Texture kinds - see TextureData below. Matches three CPU texture classes
// (src/TheRestOfYourLife/texture.h's image_texture, noise_texture, and
// checker_texture); other CPU texture classes (marble_texture,
// mipmap_texture, ...) have no GPU equivalent yet.
enum class TextureKind : int {
	Image = 0,
	Noise = 1,
	Checker = 2
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
	float noiseScale;  // Noise: scale param. Checker: 1/scale (checker_texture's own inv_scale). Unused for Image.
	float3 color1;     // Checker: "even" cell color. Unused otherwise.
	float3 color2;     // Checker: "odd" cell color. Unused otherwise.
};

// Material data (packed for SBT)
struct MaterialData {
	MaterialType type;
	float3 albedo;      // For lambertian, metal; also single-scatter color for Medium; also sigma_a (RGB absorption) for Hair
	float fuzz;         // For metal (roughness); also GGX alpha for RoughDielectric / Conductor; also HG asymmetry g for Medium; also beta_m (longitudinal roughness) for Hair
	float ior;          // For dielectric / rough_dielectric (index of refraction); also sigma_t (extinction coefficient) for Medium; also fiber eta for Hair
	float3 emission;    // For diffuse_light
	// Conductor complex IOR per RGB channel (pbrt-v4 ConductorBxDF: eta, k)
	float3 eta_c;       // Real part η per R/G/B channel (Conductor only); Hair reuses .x = beta_n (azimuthal roughness), .y = alpha_deg (scale tilt)
	float3 k_c;         // Imaginary part k per R/G/B channel (Conductor only)
	// Index into LaunchParams::textures, or -1 to use `albedo` directly
	// (every material before this field existed omitted it in brace-init,
	// which C++ aggregate-init already defaults to -1 via this default
	// member initializer - not 0 - so no existing call site needed
	// updating). Lambertian-only for now (see shade_material()'s comment).
	int textureIdx = -1;
};

// Punctual (delta) light kinds - point/spot/distant. These are evaluated
// deterministically at every hit (pdf=0, no MIS, not part of lightIndices/
// aliasTable/isLightSphere at all) rather than stochastically picked like
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
// buffer: light counts are tiny (1 per scene, matching src/TheRestOfYourLife/
// scenes_advanced.h's scene 28/29) and the images themselves are tiny (16x8
// and 8x8 respectively), so a fixed-size inline array avoids a second
// device-buffer-management path (alloc/upload/free, extra SBT plumbing) for
// data this small. Generous headroom over the actual scene data (16x8 / 8x8).
static constexpr int kGonioImageMaxDim = 32;
static constexpr int kProjImageMaxDim  = 32;

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
};

// Launch parameters (passed to all OptiX programs)
struct LaunchParams {
	// Output
	float3* framebuffer;
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
	TriangleData* triangles;
	unsigned int numTriangles;

	// Material data
	MaterialData* materials;
	unsigned int numMaterials;

	// Texture data (see TextureData above) - indexed by
	// MaterialData::textureIdx. texturePixels is one shared flat 8-bit RGB
	// buffer every Image-kind TextureData::pixelOffset points into.
	TextureData* textures;
	unsigned int numTextures;
	unsigned char* texturePixels;

	// Light sampling support (indices into sphere/quad arrays)
	int* lightIndices;          // Array of light primitive indices
	unsigned int numLights;     // Number of emissive lights in scene
	bool* isLightSphere;        // True if lightIndices[i] is sphere, false if quad

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
enum {
	RAY_TYPE_RADIANCE = 0,
	RAY_TYPE_SHADOW = 1,
	RAY_TYPE_COUNT = 2
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
