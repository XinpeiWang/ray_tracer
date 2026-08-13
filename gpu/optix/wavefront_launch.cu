// wavefront_launch.cu
// CUDA kernel launcher wrappers called from wavefront_path_tracer.cpp (host C++).
// These are compiled by nvcc and provide a plain C API that avoids <<<>>> in .cpp files.

#include "wavefront_types.h"
#define SPECTRAL_DEVICE_IMPL   // owns the __constant__ symbol definitions
#include "spectral_device.h"
#include "optix_types.h"
#include <cuda_runtime.h>

// ---- forward declarations of kernels from wavefront_kernels.cu ----
extern "C" __global__ void generate_camera_rays(
	WorkQueue<RayWorkItem>, unsigned int, unsigned int,
	GpuCameraParams, unsigned int, unsigned int);
extern "C" __global__ void evaluate_materials(
	WorkQueue<HitWorkItem>, int,
	WorkQueue<RayWorkItem>, WorkQueue<ShadowRayWorkItem>,
	float3*,
	const SphereData*, const QuadData*, const MaterialData*,
	const int*, const int*, const GpuAliasEntry*,
	unsigned int,
	const PunctualLightGPU*, unsigned int,
	int);
extern "C" __global__ void accumulate_miss(WorkQueue<MissWorkItem>, int, float3*, float3);
extern "C" __global__ void accumulate_shadow(WorkQueue<ShadowRayWorkItem>, int, const bool*, float3*);
extern "C" __global__ void reset_queue_counter(int*);
extern "C" __global__ void normalize_framebuffer(float3*, unsigned int, float);

// ---- plain C launcher wrappers ----

extern "C" void wf_launch_generate_camera_rays(
	WorkQueue<RayWorkItem> rq,
	int width, int height, int sampleIdx,
	GpuCameraParams camera,
	unsigned int frameNumber,
	cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid((width + 15) / 16, (height + 15) / 16);
	generate_camera_rays<<<grid, block, 0, (cudaStream_t)stream>>>(
		rq, (unsigned int)width, (unsigned int)height,
		camera,
		(unsigned int)sampleIdx, frameNumber);
}

extern "C" void wf_launch_evaluate_materials(
	WorkQueue<HitWorkItem>       hq,
	int                          numHits,
	WorkQueue<RayWorkItem>       nextRayQueue,
	WorkQueue<ShadowRayWorkItem> shadowQueue,
	float3*                      d_framebuffer,
	const SphereData*            d_spheres,   unsigned int numSpheres,
	const QuadData*              d_quads,     unsigned int numQuads,
	const MaterialData*          d_materials, unsigned int numMaterials,
	const int*                   d_lightIndices,
	const int*                   d_isLightSphere,
	const GpuAliasEntry*         d_aliasTable,
	unsigned int                 numLights,
	const PunctualLightGPU*      d_punctualLights,
	unsigned int                 numPunctualLights,
	int                          maxDepth,
	cudaStream_t                     stream)
{
	if (numHits == 0) return;
	dim3 block(256);
	dim3 grid((numHits + 255) / 256);
	evaluate_materials<<<grid, block, 0, (cudaStream_t)stream>>>(
		hq, numHits,
		nextRayQueue, shadowQueue,
		d_framebuffer,
		d_spheres, d_quads, d_materials,
		d_lightIndices, d_isLightSphere, d_aliasTable,
		numLights, d_punctualLights, numPunctualLights, maxDepth);
}

extern "C" void wf_launch_accumulate_miss(
	WorkQueue<MissWorkItem> mq, int numMiss,
	float3* d_framebuffer, float3 backgroundColor, cudaStream_t stream)
{
	if (numMiss == 0) return;
	dim3 block(256);
	dim3 grid((numMiss + 255) / 256);
	accumulate_miss<<<grid, block, 0, (cudaStream_t)stream>>>(mq, numMiss, d_framebuffer, backgroundColor);
}

extern "C" void wf_launch_accumulate_shadow(
	WorkQueue<ShadowRayWorkItem> sq, int numShadow,
	const bool* d_occluded, float3* d_framebuffer, cudaStream_t stream)
{
	if (numShadow == 0) return;
	dim3 block(256);
	dim3 grid((numShadow + 255) / 256);
	accumulate_shadow<<<grid, block, 0, (cudaStream_t)stream>>>(sq, numShadow, d_occluded, d_framebuffer);
}

extern "C" void wf_launch_normalize_framebuffer(
	unsigned int numPixels, float invSPP, float3* d_framebuffer, cudaStream_t stream)
{
	dim3 block(256);
	dim3 grid((numPixels + 255) / 256);
	normalize_framebuffer<<<grid, block, 0, (cudaStream_t)stream>>>(d_framebuffer, numPixels, invSPP);
}

extern "C" void wf_reset_queue_counter(int* d_counter, cudaStream_t stream)
{
	reset_queue_counter<<<1, 1, 0, (cudaStream_t)stream>>>(d_counter);
}

extern "C" void wf_upload_cie_tables(
	const float* h_cie_x,
	const float* h_cie_y,
	const float* h_cie_z,
	int n_samples)
{
	cudaMemcpyToSymbol(d_cie_x, h_cie_x, n_samples * sizeof(float));
	cudaMemcpyToSymbol(d_cie_y, h_cie_y, n_samples * sizeof(float));
	cudaMemcpyToSymbol(d_cie_z, h_cie_z, n_samples * sizeof(float));
}

extern "C" void wf_upload_srgb_table(
	const float* h_zNodes,    // [64]
	const float* h_coeffs,    // [3*64*64*64*3]
	int n_res)
{
	// Allocate device global memory and copy the large table
	float* d_z = nullptr;
	float* d_c = nullptr;
	size_t coeffs_bytes = (size_t)3 * n_res * n_res * n_res * 3 * sizeof(float);
	cudaMalloc(&d_z, n_res * sizeof(float));
	cudaMalloc(&d_c, coeffs_bytes);
	cudaMemcpy(d_z, h_zNodes, n_res * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_coeffs, coeffs_bytes, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_srgb_zNodes, &d_z, sizeof(float*));
	cudaMemcpyToSymbol(d_srgb_coeffs, &d_c, sizeof(float*));
}
