#pragma once
// optix_types_launch.h -- part 5 of 5 of optix_types.h (included by it, in order; not meant to be included on its own).

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

	// pbrt-v4 bounding-cone light BVH (CPU default is bvh_light_sampler.h;
	// see docs/FEATURE_INVENTORY.md's own "no light BVH on GPU" entry) - GPU
	// recursive backend only, this round (see OptiXRenderer::d_lightBvhNodes_'s
	// own comment for the host build/upload). lightBvhNodeCount<=0 (the
	// zero-init default, same "no in-class initializer" __constant__
	// constraint as every other optional field on this struct) means "no
	// light BVH built" - gpu_light_bvh_sample_index()/gpu_light_bvh_pmf()
	// (optix_device_helpers_lighting.h) both check this first and fall
	// back to the flat alias table above, unchanged, for every scene that
	// doesn't build one
	// (currently: every scene under any backend other than GPU-recursive).
	// lightBvhBitTrail has numLights entries (0 for any light NOT present in
	// the tree - BVHLightSampler2 excludes phi<=0 lights, matching the alias
	// table's own "geometry-only target" 1e-6f floor for a truly zero-power
	// light rather than genuinely dropping it - see the light-power-loop's
	// own comment, scene_builder.cpp).
	LightBVHNode* lightBvhNodes;
	unsigned int* lightBvhBitTrail;
	int lightBvhNodeCount;
	float lightBvhAllBMinX, lightBvhAllBMinY, lightBvhAllBMinZ;
	float lightBvhAllBMaxX, lightBvhAllBMaxY, lightBvhAllBMaxZ;

	// Punctual (delta) lights: point/spot/distant. Separate from the
	// area-light arrays above - evaluated deterministically every hit,
	// not selected via the alias table.
	PunctualLightGPU* punctualLights;
	unsigned int numPunctualLights;

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
