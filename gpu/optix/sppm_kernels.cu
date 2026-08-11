// sppm_kernels.cu
// Plain CUDA compute kernels for GPU SPPM -- no OptiX, no traversal.
// Mirrors wavefront_kernels.cu's role and compile path (nvcc -c -> .obj,
// device-linked, linked into optix_renderer.lib -- see build_optix.targets'
// CompileSppmKernels/DeviceLinkSppm targets).
//
// sppm_hash_grid_insert_kernel (sub-phase 1c): one thread per pixel with a
// valid visible point. Inserts that pixel's index into every grid cell its
// radius-expanded AABB overlaps, via an atomic-head-swap linked list over
// a fixed node pool -- see sppm_types.h's own comment for the full design
// (why GPU can't just replicate CPU's forward_list-per-bucket approach,
// and why this is the chosen alternative).

#include "sppm_types.h"
#include <cuda_runtime.h>

extern "C" __global__ void sppm_hash_grid_insert_kernel(
	const SPPMPixelGPU* pixels, int numPixels,
	SPPMHashGridParams gridParams,
	int* cellHead, int* nodeNext, int* nodePixel) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numPixels) return;

	const SPPMPixelGPU& px = pixels[i];
	if (!px.vp_valid) return;

	float r = px.radius;
	int pMin[3], pMax[3];
	sppm_to_grid(gridParams, px.vp_p.x - r, px.vp_p.y - r, px.vp_p.z - r, pMin);
	sppm_to_grid(gridParams, px.vp_p.x + r, px.vp_p.y + r, px.vp_p.z + r, pMax);

	// Capped at kSPPMMaxCellsPerPoint -- see sppm_types.h's own comment on
	// this being an accepted, documented Phase 1 correctness gap rather
	// than an oversight.
	int count = 0;
	for (int z = pMin[2]; z <= pMax[2] && count < kSPPMMaxCellsPerPoint; ++z)
	for (int y = pMin[1]; y <= pMax[1] && count < kSPPMMaxCellsPerPoint; ++y)
	for (int x = pMin[0]; x <= pMax[0] && count < kSPPMMaxCellsPerPoint; ++x) {
		int h = (int)(sppm_hash_point3i(x, y, z) % (unsigned int)gridParams.hashSize);
		int slot = i * kSPPMMaxCellsPerPoint + count;
		nodePixel[slot] = i;
		// Lock-free prepend: this slot's "next" becomes whatever was
		// previously at the bucket head, and the bucket head becomes this
		// slot -- atomicExch makes the read-old/write-new pair indivisible,
		// so concurrent inserts into the same bucket from other threads
		// can never interleave and drop a node, the same guarantee CPU's
		// per-bucket mutex gives via locking instead.
		nodeNext[slot] = atomicExch(&cellHead[h], slot);
		++count;
	}
}
