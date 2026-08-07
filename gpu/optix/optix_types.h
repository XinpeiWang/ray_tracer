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
	float3 center;
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
	NormalizedFresnel   = 10  // Fresnel-weighted diffuse reflection (pbrt-v4 NormalizedFresnelBxDF)
};

// Material data (packed for SBT)
struct MaterialData {
	MaterialType type;
	float3 albedo;      // For lambertian, metal
	float fuzz;         // For metal (roughness); also GGX alpha for RoughDielectric / Conductor
	float ior;          // For dielectric / rough_dielectric (index of refraction)
	float3 emission;    // For diffuse_light
	// Conductor complex IOR per RGB channel (pbrt-v4 ConductorBxDF: eta, k)
	float3 eta_c;       // Real part η per R/G/B channel (Conductor only)
	float3 k_c;         // Imaginary part k per R/G/B channel (Conductor only)
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
	struct {
		float3 origin;
		float3 lower_left_corner;
		float3 horizontal;
		float3 vertical;
	} camera;

	// Scene
	OptixTraversableHandle traversable;  // Acceleration structure handle

	// Geometry arrays (device pointers)
	SphereData* spheres;
	unsigned int numSpheres;
	QuadData* quads;
	unsigned int numQuads;

	// Material data
	MaterialData* materials;
	unsigned int numMaterials;

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
