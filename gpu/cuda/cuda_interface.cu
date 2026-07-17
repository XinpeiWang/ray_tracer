#include "cuda_interface.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>

#include "host.cu" // reuse existing utilities and kernel in this prototype repo layout

#define CUDA_CHECK(call) do { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		return -1; \
	} \
} while(0)

int gpu_render_main(int image_width, int image_height, int samples_per_pixel, int /*max_depth*/, const char* out_path) {
	size_t pixels = size_t(image_width) * size_t(image_height);
	size_t bytes = pixels * 3;

	unsigned char* d_img = nullptr;
	CUDA_CHECK(cudaMalloc(&d_img, bytes));
	CUDA_CHECK(cudaMemset(d_img, 0, bytes));

	dim3 block(16, 16);
	dim3 grid((image_width + block.x - 1) / block.x, (image_height + block.y - 1) / block.y);
	unsigned int seed = (unsigned int)time(nullptr);
	render_kernel<<<grid, block>>>(d_img, image_width, image_height, samples_per_pixel, seed);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	unsigned char* h_img = nullptr;
	try { h_img = (unsigned char*)malloc(bytes); } catch(...) { h_img = nullptr; }
	if (!h_img) {
		std::fprintf(stderr, "Out of host memory\n");
		CUDA_CHECK(cudaFree(d_img));
		return 1;
	}

	CUDA_CHECK(cudaMemcpy(h_img, d_img, bytes, cudaMemcpyDeviceToHost));

	FILE* f = std::fopen(out_path, "wb");
	if (!f) {
		std::perror("fopen");
		free(h_img);
		CUDA_CHECK(cudaFree(d_img));
		return 1;
	}
	std::fprintf(f, "P6\n%d %d\n255\n", image_width, image_height);
	std::fwrite(h_img, 1, bytes, f);
	std::fclose(f);

	std::printf("[cuda_interface] Wrote %s (%zu bytes)\n", out_path, bytes);

	free(h_img);
	CUDA_CHECK(cudaFree(d_img));
	return 0;
}
