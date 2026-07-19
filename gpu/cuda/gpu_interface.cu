#include "gpu_interface.h"
#include "scene_serializer.h"
#include "cuda_scene.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <ctime>

// Forward declare the kernels implemented in host.cu
extern __global__ void render_kernel(unsigned char* img, int image_width, int image_height, int samples_per_pixel, unsigned int seed);
extern __global__ void render_kernel_path(
	const SpherePOD* spheres, int num_spheres,
	const QuadPOD* quads, int num_quads,
	const MaterialPOD* materials,
	const CameraPOD* camera,
	unsigned char* image,
	int width, int height, int spp, int max_depth,
	unsigned int seed
);

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		return -1; \
	} \
} while(0)
#endif

// Check if a CUDA-capable GPU is available
int gpu_is_available() {
	int device_count = 0;
	cudaError_t err = cudaGetDeviceCount(&device_count);
	if (err != cudaSuccess || device_count == 0) {
		return 0; // No GPU available
	}

	// Check if at least one device has compute capability >= 3.0
	for (int i = 0; i < device_count; i++) {
		cudaDeviceProp prop;
		err = cudaGetDeviceProperties(&prop, i);
		if (err == cudaSuccess && prop.major >= 3) {
			return 1; // GPU available
		}
	}
	return 0;
}

int gpu_render_main(int image_width, int image_height, int samples_per_pixel, int max_depth, const char* out_path,
					double cam_x, double cam_y, double cam_z) {
	std::printf("[cuda_interface] gpu_render_main start: %dx%d spp=%d camera=(%.1f,%.1f,%.1f) out=%s\n", 
				image_width, image_height, samples_per_pixel, cam_x, cam_y, cam_z, out_path);
	std::fflush(stdout);

	// Serialize the C++ scene (Cornell box) into POD structures
	SpherePOD* spheres = nullptr;
	QuadPOD* quads = nullptr;
	MaterialPOD* materials = nullptr;
	CameraPOD camera;
	int ns, nq, nm;

	serialize_scene_arrays(image_width, image_height, samples_per_pixel, max_depth,
		&spheres, &ns, &quads, &nq, &materials, &nm, &camera, cam_x, cam_y, cam_z);

	std::printf("[cuda_interface] Launching GPU render: %d spheres, %d quads, %d materials\n", ns, nq, nm);
	std::fflush(stdout);

	// Allocate device memory
	size_t pixels = size_t(image_width) * size_t(image_height);
	size_t bytes = pixels * 3;

	unsigned char* d_img = nullptr;
	CUDA_CHECK(cudaMalloc(&d_img, bytes));
	CUDA_CHECK(cudaMemset(d_img, 0, bytes));

	SpherePOD* d_spheres = nullptr;
	QuadPOD* d_quads = nullptr;
	MaterialPOD* d_materials = nullptr;
	CameraPOD* d_cam = nullptr;

	if (ns > 0) {
		CUDA_CHECK(cudaMalloc(&d_spheres, ns * sizeof(SpherePOD)));
		CUDA_CHECK(cudaMemcpy(d_spheres, spheres, ns * sizeof(SpherePOD), cudaMemcpyHostToDevice));
	}

	if (nq > 0) {
		CUDA_CHECK(cudaMalloc(&d_quads, nq * sizeof(QuadPOD)));
		CUDA_CHECK(cudaMemcpy(d_quads, quads, nq * sizeof(QuadPOD), cudaMemcpyHostToDevice));
	}

	if (nm > 0) {
		CUDA_CHECK(cudaMalloc(&d_materials, nm * sizeof(MaterialPOD)));
		CUDA_CHECK(cudaMemcpy(d_materials, materials, nm * sizeof(MaterialPOD), cudaMemcpyHostToDevice));
	}

	CUDA_CHECK(cudaMalloc(&d_cam, sizeof(CameraPOD)));
	CUDA_CHECK(cudaMemcpy(d_cam, &camera, sizeof(CameraPOD), cudaMemcpyHostToDevice));

	// Adaptive block size: Try optimal sizes with fallback for compatibility
	// Path tracing kernels are register-heavy and may fail on older/lower-end GPUs
	const int block_sizes[] = { 24, 20, 16, 12 };  // 576, 400, 256, 144 threads
	const int num_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);

	unsigned int seed = (unsigned int)time(nullptr);
	cudaError_t launch_error = cudaSuccess;
	bool kernel_launched = false;

	for (int i = 0; i < num_sizes && !kernel_launched; i++) {
		int bs = block_sizes[i];
		dim3 block(bs, bs);
		dim3 grid((image_width + block.x - 1) / block.x, (image_height + block.y - 1) / block.y);

		// Try to launch with this block size
		if (ns > 0 || nq > 0) {
			render_kernel_path<<<grid, block>>>(d_spheres, (int)ns, d_quads, (int)nq, d_materials, d_cam, d_img, image_width, image_height, samples_per_pixel, max_depth, seed);
		} else {
			render_kernel<<<grid, block>>>(d_img, image_width, image_height, samples_per_pixel, seed);
		}

		launch_error = cudaGetLastError();

		if (launch_error == cudaSuccess) {
			// Launch succeeded, wait for completion
			std::fprintf(stderr, "[cuda_interface] Using block size %dx%d (%d threads)\n", bs, bs, bs * bs);
			CUDA_CHECK(cudaDeviceSynchronize());
			kernel_launched = true;
		} else if (launch_error == cudaErrorInvalidConfiguration || 
				   launch_error == cudaErrorLaunchOutOfResources) {
			// This block size doesn't work, try smaller
			std::fprintf(stderr, "[cuda_interface] Block size %dx%d failed, trying smaller...\n", bs, bs);
			cudaGetLastError(); // Clear error
		} else {
			// Real error, not just block size issue
			CUDA_CHECK(launch_error);
		}
	}

	if (!kernel_launched) {
		std::fprintf(stderr, "[cuda_interface] All block sizes failed\n");
		cudaFree(d_img);
		if (d_spheres) cudaFree(d_spheres);
		if (d_quads) cudaFree(d_quads);
		if (d_materials) cudaFree(d_materials);
		if (d_cam) cudaFree(d_cam);
		return -1;
	}

	// Copy result back to host
	unsigned char* h_img = (unsigned char*)malloc(bytes);
	if (!h_img) {
		std::fprintf(stderr, "Out of host memory\n");
		cudaFree(d_img);
		if (d_spheres) cudaFree(d_spheres);
		if (d_quads) cudaFree(d_quads);
		if (d_materials) cudaFree(d_materials);
		if (d_cam) cudaFree(d_cam);
		return 1;
	}

	CUDA_CHECK(cudaMemcpy(h_img, d_img, bytes, cudaMemcpyDeviceToHost));

	// Write output file
	FILE* f = std::fopen(out_path, "wb");
	if (!f) {
		std::perror("fopen");
		free(h_img);
		cudaFree(d_img);
		if (d_spheres) cudaFree(d_spheres);
		if (d_quads) cudaFree(d_quads);
		if (d_materials) cudaFree(d_materials);
		if (d_cam) cudaFree(d_cam);
		return 1;
	}
	std::fprintf(f, "P6\n%d %d\n255\n", image_width, image_height);
	std::fwrite(h_img, 1, bytes, f);
	std::fclose(f);

	std::printf("[cuda_interface] Wrote %s (%zu bytes)\n", out_path, bytes);
	std::fflush(stdout);

	// Cleanup host arrays
	free_scene_arrays(spheres, quads, materials);

	// Cleanup
	free(h_img);
	cudaFree(d_img);
	if (d_spheres) cudaFree(d_spheres);
	if (d_quads) cudaFree(d_quads);
	if (d_materials) cudaFree(d_materials);
	if (d_cam) cudaFree(d_cam);

	return 0;
}
