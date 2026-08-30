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
//
// sppm_radius_update_kernel / sppm_final_image_kernel (sub-phase 1d): one
// thread per pixel each, direct ports of src/shared/sppm.h's
// SPPMUpdateRadius<T>()/SPPMFinalImage<T>() -- see those functions' own
// comments for the formulas being mirrored.

#include "sppm_types.h"
#include "optix_math_helpers.h"
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

// Port of src/shared/sppm.h's SPPMUpdateRadius<T>(): after a photon pass,
// contracts each pixel's search radius (gamma=2/3) and folds this
// iteration's Phi into the long-running tau accumulator, scaled by the
// radius contraction squared. Always resets vp_valid (next camera pass
// re-derives it), matching SPPMUpdateRadius's own unconditional reset.
extern "C" __global__ void sppm_radius_update_kernel(SPPMPixelGPU* pixels, int numPixels) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numPixels) return;

	SPPMPixelGPU& px = pixels[i];
	if (px.m > 0) {
		const float kGamma = 2.0f / 3.0f;
		float nNew = px.n + kGamma * (float)px.m;
		float rNew = px.radius * sqrtf(nNew / (px.n + (float)px.m));

		float scale = (rNew * rNew) / (px.radius * px.radius);
		px.tau = (px.tau + px.Phi) * scale;

		px.n      = nNew;
		px.radius = rNew;
		px.m      = 0;
		px.Phi    = make_float3(0.0f, 0.0f, 0.0f);
	}
	px.vp_valid = false;
}

// Port of src/shared/sppm.h's SPPMFinalImage<T>(): L = Ld/nIterations +
// tau/(totalPhotonPaths * pi * r^2). No `n` factor - see that function's
// own comment for why (px.n belongs only to the radius-contraction step,
// not this reconstruction; including it here darkened indirect light more
// the longer a render ran instead of converging).
extern "C" __global__ void sppm_final_image_kernel(
	const SPPMPixelGPU* pixels, int numPixels,
	int nIterations, float totalPhotonPaths, float3* framebuffer) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numPixels) return;

	const SPPMPixelGPU& px = pixels[i];
	const float kPi = 3.14159265358979323846f;
	float invIter = (nIterations > 0) ? 1.0f / (float)nIterations : 0.0f;
	float denom = totalPhotonPaths * kPi * px.radius * px.radius;
	float3 indirect = (denom > 0.0f) ? px.tau * (1.0f / denom) : make_float3(0.0f, 0.0f, 0.0f);
	framebuffer[i] = px.Ld * invIter + indirect;
}

// Test-only mirror of sppm_programs.cu's own sppm_mix_branch_hash01() (that
// file is an OptiX device-program module, not directly host-callable/
// device-linkable the way this plain-kernel file is - see
// tests/unit/sppm_gpu_mix_hash_tests.cpp's own top comment for the full
// rationale). MUST stay byte-for-byte identical to the real function - this
// is a 5th hand-duplicated copy of the same hash formula (CPU's
// branch_hash01, the recursive and wavefront backends' own copies, this
// file's own sppm_programs.cu copy, and now this test-only one), purely to
// give the `variant`/iteration-decorrelation fix (added to close a
// real-render-verified bug: a variant-less resolve froze Mix materials to a
// static per-pixel binary choice for the whole render) automated coverage
// GPU SPPM had none of before. If sppm_mix_branch_hash01() ever changes,
// update this copy too.
extern "C" __global__ void sppm_mix_branch_hash01_test_kernel(
	const float3* points, const unsigned int* variants, int n, float* outHashes) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const float3 p = points[i];
	const unsigned int variant = variants[i];
	float h = sinf(p.x * 127.1f + p.y * 311.7f + p.z * 74.7f
	               + float(variant) * 0.6180339887498949f) * 43758.5453f;
	outHashes[i] = h - floorf(h);
}
