// sppm_launch.cu
// CUDA kernel launcher wrappers called from host C++ (sppm_path_tracer.cpp,
// and directly from tests -- see tests/unit/sppm_gpu_hash_grid_tests.cpp).
// Mirrors wavefront_launch.cu's role: plain C API wrapping <<<>>> launch
// syntax, which isn't valid inside a .cpp translation unit.

#include "sppm_types.h"
#include <cuda_runtime.h>

extern "C" __global__ void sppm_hash_grid_insert_kernel(
	const SPPMPixelGPU*, int, SPPMHashGridParams, int*, int*, int*);

extern "C" void sppm_launch_hash_grid_insert(
	const SPPMPixelGPU* d_pixels, int numPixels,
	SPPMHashGridParams gridParams,
	int* d_cellHead, int* d_nodeNext, int* d_nodePixel,
	cudaStream_t stream) {
	if (numPixels == 0) return;
	dim3 block(256);
	dim3 grid((numPixels + 255) / 256);
	sppm_hash_grid_insert_kernel<<<grid, block, 0, stream>>>(
		d_pixels, numPixels, gridParams, d_cellHead, d_nodeNext, d_nodePixel);
}

extern "C" __global__ void sppm_radius_update_kernel(SPPMPixelGPU*, int);

extern "C" void sppm_launch_radius_update(
	SPPMPixelGPU* d_pixels, int numPixels, cudaStream_t stream) {
	if (numPixels == 0) return;
	dim3 block(256);
	dim3 grid((numPixels + 255) / 256);
	sppm_radius_update_kernel<<<grid, block, 0, stream>>>(d_pixels, numPixels);
}

extern "C" __global__ void sppm_final_image_kernel(
	const SPPMPixelGPU*, int, int, float, float3*);

extern "C" void sppm_launch_final_image(
	const SPPMPixelGPU* d_pixels, int numPixels,
	int nIterations, float totalPhotonPaths, float3* d_framebuffer,
	cudaStream_t stream) {
	if (numPixels == 0) return;
	dim3 block(256);
	dim3 grid((numPixels + 255) / 256);
	sppm_final_image_kernel<<<grid, block, 0, stream>>>(
		d_pixels, numPixels, nIterations, totalPhotonPaths, d_framebuffer);
}
