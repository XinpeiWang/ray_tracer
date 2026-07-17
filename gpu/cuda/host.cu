// Minimal CUDA renderer prototype
// Renders a simple gradient image on the GPU and writes a PPM file
// Build: nvcc host.cu -o cuda_renderer.exe

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		std::exit(1); \
	} \
} while(0)

__global__ void render_kernel(unsigned char* img, int image_width, int image_height) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= image_width || j >= image_height) return;

	float u = float(i) / float(image_width - 1);
	float v = float(j) / float(image_height - 1);

	// simple gradient
	float r = 0.5f * (1.0f - u) + u;
	float g = 0.7f * (1.0f - v) + v * 0.4f;
	float b = 1.0f * (1.0f - u) + 0.5f * v;

	int idx = (j * image_width + i) * 3;
	img[idx + 0] = (unsigned char)(fminf(fmaxf(r, 0.0f), 1.0f) * 255.999f);
	img[idx + 1] = (unsigned char)(fminf(fmaxf(g, 0.0f), 1.0f) * 255.999f);
	img[idx + 2] = (unsigned char)(fminf(fmaxf(b, 0.0f), 1.0f) * 255.999f);
}

int main(int argc, char** argv) {
	int image_width = 800;
	int image_height = 450;
	if (argc >= 3) {
		image_width = std::atoi(argv[1]);
		image_height = std::atoi(argv[2]);
		if (image_width <= 0) image_width = 800;
		if (image_height <= 0) image_height = 450;
	}

	std::string out_path = "image_cuda.ppm";
	if (const char* od = std::getenv("OneDrive")) {
		out_path = std::string(od) + "\\Desktop\\image_cuda.ppm";
	} else if (const char* up = std::getenv("USERPROFILE")) {
		out_path = std::string(up) + "\\Desktop\\image_cuda.ppm";
	}

	size_t pixels = size_t(image_width) * size_t(image_height);
	size_t bytes = pixels * 3;

	unsigned char* d_img = nullptr;
	CUDA_CHECK(cudaMalloc(&d_img, bytes));
	CUDA_CHECK(cudaMemset(d_img, 0, bytes));

	dim3 block(16, 16);
	dim3 grid((image_width + block.x - 1) / block.x, (image_height + block.y - 1) / block.y);

	render_kernel<<<grid, block>>>(d_img, image_width, image_height);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	unsigned char* h_img = (unsigned char*)malloc(bytes);
	if (!h_img) {
		std::fprintf(stderr, "Out of host memory\n");
		CUDA_CHECK(cudaFree(d_img));
		return 1;
	}

	CUDA_CHECK(cudaMemcpy(h_img, d_img, bytes, cudaMemcpyDeviceToHost));

	// Write PPM (P6 binary for smaller files)
	FILE* f = std::fopen(out_path.c_str(), "wb");
	if (!f) {
		std::perror("fopen");
		free(h_img);
		CUDA_CHECK(cudaFree(d_img));
		return 1;
	}
	std::fprintf(f, "P6\n%d %d\n255\n", image_width, image_height);
	// PPM typically top to bottom; our j increases top->bottom, keep as-is
	std::fwrite(h_img, 1, bytes, f);
	std::fclose(f);

	std::printf("Wrote %s (%zu bytes)\n", out_path.c_str(), bytes);

	free(h_img);
	CUDA_CHECK(cudaFree(d_img));
	return 0;
}
