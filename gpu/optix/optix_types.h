// OptiX type definitions and shared structures
// Shared between host and device code

#pragma once

#include <optix.h>
#include <cuda_runtime.h>

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
	DiffuseLight = 3
};

// Material data (packed for SBT)
struct MaterialData {
	MaterialType type;
	float3 albedo;      // For lambertian, metal
	float fuzz;         // For metal (roughness)
	float ior;          // For dielectric (index of refraction)
	float3 emission;    // For diffuse_light
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
	RAY_TYPE_COUNT
};

// OptiX error checking macro
#define OPTIX_CHECK(call)                                                      \
	do {                                                                       \
		OptixResult res = call;                                                \
		if (res != OPTIX_SUCCESS) {                                            \
			fprintf(stderr, "OptiX call (%s) failed with code %d (line %d)\n", \
					#call, res, __LINE__);                                     \
			exit(1);                                                           \
		}                                                                      \
	} while (0)

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
	do {                                                                       \
		cudaError_t error = call;                                              \
		if (error != cudaSuccess) {                                            \
			fprintf(stderr, "CUDA call (%s) failed with code %d (line %d): %s\n", \
					#call, error, __LINE__, cudaGetErrorString(error));        \
			exit(1);                                                           \
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
			exit(1);                                                           \
		}                                                                      \
	} while (0)
