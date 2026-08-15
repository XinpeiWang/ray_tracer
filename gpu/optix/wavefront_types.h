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

	// MaterialType::Medium and MaterialType::DielectricMedium (exit surface,
	// i.e. front-facing false) only: `t` above holds the entry (near) root of
	// the medium sphere's boundary; this holds the exit (far) root. Both are
	// recomputed in closesthit relative to the CURRENT ray origin (handles a
	// ray that already starts inside the sphere, e.g. continuing after a
	// prior in-medium scatter event) - see __closesthit__wf_sphere. Unused
	// (0) for every other material type, and for DielectricMedium's entry
	// surface (frontFace true - see that field below).
	float  mediumTFar;

	// True if the ray hit the front (outward-facing) side of the surface,
	// i.e. the same thing `front_face` means in optix_intersection_sphere.h.
	// Lost once `normal` below is flipped to always oppose the incident ray
	// (see __closesthit__wf_sphere), so it has to travel separately -
	// MaterialType::DielectricMedium needs it to pick between its entry
	// (refract/reflect the dielectric surface) and exit (re-enter the
	// internal medium, or exit-refract) behaviour. Meaningless (left 0) for
	// every material that doesn't branch on it.
	int    frontFace;

	// Sphere-only, OBJECT-space (never transformed to world, even for an
	// instanced placement - mirrors optix_intersection_sphere.h's obj_normal)
	// raw outward normal, i.e. BEFORE the front-face flip that produced
	// `normal` above. MaterialType::NormalMappedLambertian's tangent (dpdu)
	// is derived from this, matching the recursive path exactly. Zero for
	// quad/triangle hits and for any sphere material that doesn't read it.
	float3 objNormal;

	// Surface texture coordinates. Sphere: standard spherical (theta,phi)
	// mapping, computed in __closesthit__wf_sphere from objNormal, matching
	// CPU's get_sphere_uv() and optix_intersection_sphere.h's sphere_uv_u/v.
	// Triangle: barycentric-interpolated from TriangleData::uv0/1/2 when
	// tri.hasUVs (matches optix_intersection_triangle.h), else 0. Quad: no
	// per-quad UV exists on this codebase's GPU side (matches the recursive
	// path, which never textures a quad either), always 0. Only meaningful
	// for MaterialType::Lambertian/NormalMappedLambertian with
	// MaterialData::textureIdx >= 0.
	float  uv_u, uv_v;

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
	// Radiance already accumulated along this path before it missed (e.g. a
	// specular-bounce chain's emission from an earlier light hit - see
	// RayWorkItem::radiance's own comment). Non-specular bounces flush their
	// radiance at every hit (evaluate_materials's NEE block), but a specular
	// chain defers the flush until the path terminates, so a miss has to
	// carry it here or it's silently lost - accumulate_miss adds this on top
	// of the flat background term, not instead of it.
	float  radiance[kWFNWavelengths];
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

	// Heterogeneous cloud media (MaterialType::CloudMedium), same shared
	// device buffer OptiXRenderer::buildScene() already uploads for the
	// recursive path - see wavefront_kernels.cu's CloudMedium case.
	CloudMedium<float>* cloudMediums;
	unsigned int         numCloudMediums;

	// Light sampling (alias table, same layout as RecursivePathTracer)
	int*         lightIndices;
	const GpuLightKind* lightKinds;   // see optix_types.h - width is load-bearing
	// Per-instance primitive base, same table and same -1 sentinel as
	// LaunchParams::instancePrimBase in optix_types.h. Null when the scene has
	// no object instances, which is every built-in scene.
	const int* instancePrimBase;
	GpuAliasEntry* aliasTable;
	unsigned int   numLights;

	// Rendering params
	unsigned int samplesPerPixel;
	unsigned int maxDepth;
	unsigned int frameNumber;  // seed offset
};
#endif // __CUDACC__ || OPTIX_RENDERER_AVAILABLE
