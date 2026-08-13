// wavefront_types.h
// Queue types and work items for the wavefront GPU path tracer.
#pragma once

// When compiled by nvcc or with OptiX, pull in the real headers.
// When included from a plain C++ host-only context (e.g. unit tests without
// the CUDA toolkit on the include path), provide minimal stubs so the POD
// struct layouts can be inspected without the GPU toolchain.
#if defined(__CUDACC__) || defined(OPTIX_RENDERER_AVAILABLE)
#include <optix.h>
#include <cuda_runtime.h>
#include "optix_types.h"
#else
// Minimal float3 stub for host-only compilation
#ifndef __VECTOR_TYPES_H__
struct float3 { float x, y, z; };
#endif
#endif

// ============================================================================
// Work item types
// ============================================================================

// Number of hero wavelengths per ray (matches pbrt-v4 NSpectrumSamples = 4)
static constexpr int kWFNWavelengths = 4;

// A primary or bounce ray pending intersection.
struct RayWorkItem {
	float3       origin;
	float3       direction;
	// Spectral path weight (SampledSpectrum<4>): starts at {1,1,1,1}
	float        throughput[kWFNWavelengths];
	// Accumulated spectral radiance (SampledSpectrum<4>) for this path
	float        radiance[kWFNWavelengths];
	// Hero wavelengths (nm) sampled at camera ray generation
	float        wavelengths[kWFNWavelengths];
	// Sampling PDFs for each hero wavelength
	float        wavelength_pdfs[kWFNWavelengths];
	unsigned int seed;             // PCG state
	int          pixelIndex;       // flat pixel index (y*width + x)
	int          depth;            // current bounce (0 = primary)
	int          specular_bounce;  // 1 if this ray was spawned by a specular event
	float        tMin;             // ray t_min (normally 0.001)
	float        tMax;             // ray t_max (normally 1e30)
};

// Result of a successful intersection, produced by OptiX closesthit.
struct HitWorkItem {
	// Surface geometry
	float3 hitPoint;
	float3 normal;             // shading normal (outward)
	float  t;                  // ray parameter at hit

	// Material info
	int    materialIdx;
	int    geomType;           // 0 = sphere, 1 = quad, 2 = bilinear patch, 3 = triangle (unused beyond bookkeeping - not read anywhere)

	// MaterialType::Medium only: `t` above holds the entry (near) root of the
	// medium sphere's boundary; this holds the exit (far) root. Both are
	// recomputed in closesthit relative to the CURRENT ray origin (handles a
	// ray that already starts inside the sphere, e.g. continuing after a
	// prior in-medium scatter event) - see __closesthit__wf_sphere. Unused
	// (0) for all other material types.
	float  mediumTFar;

	// Incident ray (needed to evaluate material)
	float3 rayOrigin;
	float3 rayDir;
	// Spectral path state (SampledSpectrum<4>)
	float  throughput[kWFNWavelengths];
	float  radiance[kWFNWavelengths];
	float  wavelengths[kWFNWavelengths];
	float  wavelength_pdfs[kWFNWavelengths];
	unsigned int seed;
	int    pixelIndex;
	int    depth;
	int    specular_bounce;    // 1 if this path arrived via a specular event
};

// A shadow ray: if it reaches tMax unoccluded, Ld is added to the framebuffer.
struct ShadowRayWorkItem {
	float3 origin;
	float3 direction;
	float  tMax;               // distance to light surface (use light.t - eps)
	// Spectral direct-light contribution (throughput * Le * BSDF / pdf)
	float  Ld[kWFNWavelengths];
	// Wavelengths needed for XYZ conversion at accumulation
	float  wavelengths[kWFNWavelengths];
	float  wavelength_pdfs[kWFNWavelengths];
	int    pixelIndex;
};

// A miss result: the ray escaped, accumulate background/environment.
struct MissWorkItem {
	float  throughput[kWFNWavelengths];
	float  wavelengths[kWFNWavelengths];
	float  wavelength_pdfs[kWFNWavelengths];
	float3 rayDir;             // for environment-map lookups (currently unused)
	int    pixelIndex;
};

// ============================================================================
// WorkQueue<T> — device-side append-only queue with an atomic size counter
// ============================================================================

template<typename T>
struct WorkQueue {
	T*            items;       // device pointer to item array
	int*          counter;     // device pointer to atomic counter (1 int)
	int           capacity;    // max items (= width * height)

	#ifdef __CUDACC__
	// Append an item from device code; returns the slot index or -1 if full.
	__device__ __forceinline__ int push(const T& item) {
		int slot = atomicAdd(counter, 1);
		if (slot < capacity) {
			items[slot] = item;
			return slot;
		}
		return -1;  // overflow (should not happen if capacity >= numPixels)
	}

	// Number of items currently in the queue (only meaningful after device sync).
	__host__ __device__ __forceinline__ int size() const {
#ifdef __CUDA_ARCH__
		return atomicAdd(counter, 0);
#else
		int val = 0;
		cudaMemcpy(&val, counter, sizeof(int), cudaMemcpyDeviceToHost);
		return val;
#endif
	}
#endif // __CUDACC__
};

// ============================================================================
// WavefrontQueues — all queues for one bounce iteration, plus the framebuffer
// ============================================================================

struct WavefrontQueues {
	WorkQueue<RayWorkItem>    rayQueue;       // rays pending intersection
	WorkQueue<RayWorkItem>    nextRayQueue;   // rays for the next bounce
	WorkQueue<HitWorkItem>    hitQueue;       // intersection results
	WorkQueue<MissWorkItem>   missQueue;      // escaped rays
	WorkQueue<ShadowRayWorkItem> shadowQueue; // shadow rays pending occlusion test
};

// ============================================================================
// WavefrontLaunchParams — constant data passed to OptiX wavefront programs
// (requires optix_types.h for SphereData/QuadData/MaterialData)
// ============================================================================

#if defined(__CUDACC__) || defined(OPTIX_RENDERER_AVAILABLE)
struct WavefrontLaunchParams {
	// Queues
	WorkQueue<RayWorkItem>       rayQueue;
	WorkQueue<HitWorkItem>       hitQueue;
	WorkQueue<MissWorkItem>      missQueue;

	// Shadow ray queue is traced separately via a second optixLaunch.
	WorkQueue<ShadowRayWorkItem> shadowQueue;

	// Output
	float3*      framebuffer;
	unsigned int width;
	unsigned int height;

	// Scene
	OptixTraversableHandle traversable;

	// Geometry arrays
	SphereData*  spheres;
	unsigned int numSpheres;
	QuadData*    quads;
	unsigned int numQuads;
	BilinearPatchData* bilinearPatches;
	unsigned int numBilinearPatches;
	TriangleData* triangles;
	unsigned int numTriangles;

	// Materials
	MaterialData*  materials;
	unsigned int   numMaterials;

	// Light sampling (alias table, same layout as RecursivePathTracer)
	int*         lightIndices;
	const GpuLightKind* lightKinds;   // see optix_types.h - width is load-bearing
	GpuAliasEntry* aliasTable;
	unsigned int   numLights;

	// Rendering params
	unsigned int samplesPerPixel;
	unsigned int maxDepth;
	unsigned int frameNumber;  // seed offset
};
#endif // __CUDACC__ || OPTIX_RENDERER_AVAILABLE
