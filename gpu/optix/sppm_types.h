// sppm_types.h
// Host/device-shared types for the GPU SPPM (Stochastic Progressive Photon
// Mapping) integrator. Unlike wavefront_types.h, this file does NOT have a
// CUDA-toolkit-free fallback branch: everything here is gated behind
// __CUDACC__/OPTIX_RENDERER_AVAILABLE and genuinely needs cuda_runtime.h's
// float3/etc -- every consumer of this header (sppm_programs.cu,
// sppm_kernels.cu, sppm_path_tracer.cpp, and their tests) already builds
// with OPTIX_RENDERER_AVAILABLE defined, so a stub branch isn't needed the
// way wavefront_types.h's is for its own host-only test consumers.
//
// Extended incrementally, one sub-phase at a time, rather than redesigned
// -- see C:\Users\xinpe\.claude\plans\cached-wobbling-ritchie.md.
#pragma once

#if defined(__CUDACC__) || defined(OPTIX_RENDERER_AVAILABLE)
#include <optix.h>
#include <cuda_runtime.h>
#include "optix_types.h"

// ============================================================================
// SPPMPixelGPU -- per-pixel SPPM state, GPU-flattened analog of src/shared/
// sppm.h's SPPMPixel<T> (see that struct's own comment for the CPU
// reference). radius/tau/n/Phi/m were originally going to be deferred to
// sub-phase 1d, but the hash-grid build (1c, right below) turns out to
// need `radius` to compute each visible point's search AABB -- rather than
// split that one field out awkwardly, the whole SPPMPixel-equivalent set
// is added here together, since 1d's photon pass needs the rest (tau/n/
// Phi/m) immediately after anyway.
//
// vp_materialIdx doubles as the CPU adapter's bsdf_id: unlike CPU's
// hittable/material (virtual, shared_ptr-based, needing sppm_adapter.h's
// whole transient_ctx_/durable_ctx_ shading-context-table machinery to
// survive from the camera pass to a later photon-pass query), GPU
// MaterialData is already a flat, permanently-addressable array -- the
// index alone is a valid handle with no extra indirection needed.
struct SPPMPixelGPU {
	float  radius;           // progressive search radius (SPPMUpdateRadius)
	float3 Ld;               // accumulated direct lighting (this iteration)
	float3 tau;               // long-running flux accumulator (SPPMUpdateRadius)
	float  n;                 // real photon count, after contraction (SPPMUpdateRadius)
	float3 Phi;               // per-iteration photon flux accumulator (photon pass)
	int    m;                 // per-iteration raw photon count (photon pass)
	float3 vp_p;              // visible-point world position
	float3 vp_wo;             // outgoing direction (toward camera/prior bounce)
	float3 vp_n;              // shading normal at the visible point
	float3 vp_beta;           // path throughput at the visible point
	int    vp_materialIdx;    // index into SPPMLaunchParams::materials
	bool   vp_valid;          // false if this pixel's camera ray never reached
	                          // a non-delta hit this iteration (escaped, hit a
	                          // light, or exhausted maxDepth mid-specular-chain)
};

// ============================================================================
// SPPM spatial hash grid -- backs sub-phase 1d's photon pass, which needs
// to find, for a photon landing at some world position, which pixels'
// visible points are nearby. CPU's HashGrid (src/shared/sppm.h) is a
// vector<forward_list<int>>: one heap-allocated node per grid cell a
// visible point's radius-expanded AABB overlaps. GPU has no cheap per-node
// malloc equivalent, so this uses an atomic-head-swap linked list over a
// FIXED node pool instead (see sppm_kernels.cu's sppm_hash_grid_insert for
// the actual insertion logic) -- same "many threads racing to prepend to
// one bucket's list" concurrency shape as CPU's own per-bucket-mutex
// design (Phase 7 of the CPU work, sppm_adapter.h's bucket_mutexes), just
// swapping a mutex for atomicExch and a dynamic list for a bounded pool.
//
// Grid bounds/resolution (gridMin/gridMax/gridRes/cellSize/hashSize) are
// computed HOST-side once per iteration (mirrors HashGrid::Build()'s first
// phase) rather than in a kernel -- Phase 1 simplicity, matching how the
// CPU port itself stayed single-threaded here until a much later phase.
//
// HashPoint3i/ToGrid/Bucket below are __host__ __device__ so the exact
// same implementation runs in the insertion kernel (device) and in host-
// side test/verification code that needs to predict which bucket a given
// point lands in (sub-phase 1c's isolated hash-grid test) -- duplicated
// from src/shared/sppm.h's sppm_detail::HashPoint3i/HashGrid::ToGrid
// rather than shared (this GPU port's established pattern throughout, see
// e.g. wavefront_kernels.cu's own comment on why device helpers aren't
// shared across backends), but kept numerically identical on purpose.
// ============================================================================

struct SPPMHashGridParams {
	float3 gridMin, gridMax;
	int    gridRes[3];
	float3 cellSize;
	int    hashSize;
};

// Max grid cells one visible point's radius-expanded AABB can register
// into during hash-grid insertion. Generous for Cornell-scale radii/grid
// resolution; a point whose AABB spans more cells than this silently
// registers only the first kSPPMMaxCellsPerPoint of them -- a known,
// documented Phase 1 correctness gap (see the plan), not expected to
// matter at this scale, but a real limitation worth remembering if a
// later scene needs much finer grid resolution relative to its radii.
static constexpr int kSPPMMaxCellsPerPoint = 8;

__host__ __device__ __forceinline__ unsigned int sppm_hash_point3i(int x, int y, int z) {
	unsigned int a = (unsigned int)x * 2654435769u;
	unsigned int b = (unsigned int)y * 805459861u;
	unsigned int c = (unsigned int)z * 3674653429u;
	return a ^ b ^ c;
}

__host__ __device__ __forceinline__ void sppm_to_grid(
	const SPPMHashGridParams& gp, float wx, float wy, float wz, int out[3]) {
	float pg[3] = {
		(wx - gp.gridMin.x) / (gp.gridMax.x - gp.gridMin.x),
		(wy - gp.gridMin.y) / (gp.gridMax.y - gp.gridMin.y),
		(wz - gp.gridMin.z) / (gp.gridMax.z - gp.gridMin.z)
	};
	for (int d = 0; d < 3; ++d) {
		out[d] = (int)(gp.gridRes[d] * pg[d]);
		if (out[d] < 0) out[d] = 0;
		if (out[d] >= gp.gridRes[d]) out[d] = gp.gridRes[d] - 1;
	}
}

// Next prime >= n -- host-only sizing helper for the hash table (hashSize),
// duplicated from src/shared/sppm.h's sppm_detail::NextPrime rather than
// included (that header pulls in bdpt.h/rng.h/<forward_list> etc. that
// don't belong in this GPU-facing header) -- kept numerically identical on
// purpose, same "duplicated but numerically identical" convention as
// sppm_hash_point3i/sppm_to_grid above.
inline int sppm_next_prime(int n) {
	if (n <= 2) return 2;
	if (n % 2 == 0) ++n;
	while (true) {
		bool ok = true;
		for (int i = 3; (long long)i * i <= n; i += 2) {
			if (n % i == 0) { ok = false; break; }
		}
		if (ok) return n;
		n += 2;
	}
}

__host__ __device__ __forceinline__ int sppm_hash_bucket(
	const SPPMHashGridParams& gp, float wx, float wy, float wz) {
	int gi[3];
	sppm_to_grid(gp, wx, wy, wz, gi);
	return (int)(sppm_hash_point3i(gi[0], gi[1], gi[2]) % (unsigned int)gp.hashSize);
}

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
	const GpuLightKind* lightKinds;   // see optix_types.h - width is load-bearing
	GpuAliasEntry* aliasTable;
	unsigned int   numLights;

	// SPPM per-pixel state (see SPPMPixelGPU above).
	SPPMPixelGPU* pixels;

	// Output (Phase 1b wrote pixels[i].Ld here directly for visual
	// verification, single camera-pass only. Sub-phase 1d's real
	// sppm_final_image_kernel -- see sppm_kernels.cu -- reconstructs the
	// full SPPMFinalImage() formula into this buffer instead, once the
	// photon pass is contributing to pixels[i].tau/n too).
	float3*      framebuffer;
	unsigned int width;
	unsigned int height;
	unsigned int maxDepth;

	// Camera (reused verbatim from the existing GpuCameraParams -- Phase 1
	// only exercises the Perspective case's viewport fields).
	GpuCameraParams camera;

	// Spatial hash grid (sub-phase 1d) -- bounds/resolution (hashGrid) are
	// computed HOST-side once per iteration (mirrors CPU's HashGrid::Build()
	// first phase, see sppm.h) from a readback of pixels[], then
	// cellHead/nodeNext/nodePixel are populated device-side via
	// sppm_hash_grid_insert_kernel (sppm_kernels.cu) before the photon-pass
	// raygen below reads them. Buffers are persistent (allocated once by
	// SPPMPathTracer, sized off hashGrid.hashSize/numPixels*kSPPMMaxCellsPerPoint),
	// not reallocated per iteration -- only cellHead is reset (to -1) each
	// iteration, since nodeNext/nodePixel are always fully overwritten up to
	// each point's own `count` before being linked from a fresh cellHead.
	SPPMHashGridParams hashGrid;
	int* cellHead;
	int* nodeNext;
	int* nodePixel;

	// Photon pass (sub-phase 1d) -- one thread per photon, launched as
	// (nPhotons, 1, 1). photonSeedBase varies per-iteration (set by the host
	// to the iteration index) so successive iterations' photon RNG streams
	// don't retrace identical paths.
	unsigned int nPhotons;
	unsigned int photonSeedBase;
};

#endif // __CUDACC__ || OPTIX_RENDERER_AVAILABLE
