// sppm_types.h
// Host/device-shared types for the GPU SPPM (Stochastic Progressive Photon
// Mapping) integrator. Mirrors wavefront_types.h's role and its dual-compile
// guard (usable from a plain host TU without the CUDA toolkit, and from
// nvcc/OptiX device code) for the same reason: struct layouts need to be
// inspectable from host-only unit tests without requiring the GPU toolchain.
//
// Phase 1a only needs enough to trace a primary ray against the uploaded
// scene and confirm hit/miss -- see sppm_programs.cu's own comment. Later
// sub-phases (1b: camera-pass visible points/Ld, 1c/1d: hash grid + photon
// pass) extend this file incrementally rather than redesigning it, matching
// the plan's sub-phase-by-sub-phase verification approach (see
// C:\Users\xinpe\.claude\plans\cached-wobbling-ritchie.md).
#pragma once

#if defined(__CUDACC__) || defined(OPTIX_RENDERER_AVAILABLE)
#include <optix.h>
#include <cuda_runtime.h>
#include "optix_types.h"

// ============================================================================
// SPPMLaunchParams -- constant data passed to sppm_programs.cu's OptiX
// programs (Phase 1a: camera-pass raygen only).
// ============================================================================
struct SPPMLaunchParams {
	// Scene (device pointers, owned by OptiXRenderer -- reused as-is from
	// the existing buildScene() upload, not re-allocated here).
	OptixTraversableHandle traversable;
	SphereData*   spheres;
	unsigned int  numSpheres;
	QuadData*     quads;
	unsigned int  numQuads;
	MaterialData* materials;
	unsigned int  numMaterials;

	// Output
	float3*      framebuffer;
	unsigned int width;
	unsigned int height;

	// Camera (reused verbatim from the existing GpuCameraParams -- Phase 1a
	// only exercises the Perspective case's viewport fields).
	GpuCameraParams camera;
};

#endif // __CUDACC__ || OPTIX_RENDERER_AVAILABLE
