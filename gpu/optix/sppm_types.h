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
// SPPMPixelGPU -- per-pixel SPPM state, GPU-flattened analog of src/shared/
// sppm.h's SPPMPixel<T> (see that struct's own comment for the CPU
// reference). Phase 1b only populates/reads the visible-point and Ld
// fields (the camera pass's own output); the radius/tau/n/Phi/m fields
// SPPMUpdateRadius/the photon pass need are added in sub-phase 1d, once
// there's a photon pass to read them -- matches the plan's own
// incremental-extension approach (see this file's top comment).
//
// vp_materialIdx doubles as the CPU adapter's bsdf_id: unlike CPU's
// hittable/material (virtual, shared_ptr-based, needing sppm_adapter.h's
// whole transient_ctx_/durable_ctx_ shading-context-table machinery to
// survive from the camera pass to a later photon-pass query), GPU
// MaterialData is already a flat, permanently-addressable array -- the
// index alone is a valid handle with no extra indirection needed.
struct SPPMPixelGPU {
	float3 Ld;             // accumulated direct lighting (this iteration)
	float3 vp_p;            // visible-point world position
	float3 vp_wo;            // outgoing direction (toward camera/prior bounce)
	float3 vp_n;             // shading normal at the visible point
	float3 vp_beta;          // path throughput at the visible point
	int    vp_materialIdx;   // index into SPPMLaunchParams::materials
	bool   vp_valid;         // false if this pixel's camera ray never reached
	                         // a non-delta hit this iteration (escaped, hit a
	                         // light, or exhausted maxDepth mid-specular-chain)
};

// ============================================================================
// SPPMLaunchParams -- constant data passed to sppm_programs.cu's OptiX
// programs.
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

	// Area-light sampling (power-weighted alias table, same device data
	// OptiXRenderer::buildScene() already uploads for the regular path
	// tracers -- reused as-is, no separate SPPM copy).
	int*           lightIndices;
	bool*          isLightSphere;
	GpuAliasEntry* aliasTable;
	unsigned int   numLights;

	// SPPM per-pixel state (see SPPMPixelGPU above).
	SPPMPixelGPU* pixels;

	// Output (Phase 1b writes pixels[i].Ld here directly for visual
	// verification -- sppm_render_with_adapter()'s GPU analog in a later
	// sub-phase reconstructs the real SPPMFinalImage() formula into this
	// buffer instead once there's a photon pass contributing to it too).
	float3*      framebuffer;
	unsigned int width;
	unsigned int height;
	unsigned int maxDepth;

	// Camera (reused verbatim from the existing GpuCameraParams -- Phase 1
	// only exercises the Perspective case's viewport fields).
	GpuCameraParams camera;
};

#endif // __CUDACC__ || OPTIX_RENDERER_AVAILABLE
