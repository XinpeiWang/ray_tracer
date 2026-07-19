// Minimal CUDA test to verify nvcc works
#include <cstdio>
#include <cuda_runtime.h>

__global__ void test_kernel(int* data) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	data[idx] = idx;
}

int main() {
	printf("CUDA minimal test\n");

	int* d_data;
	cudaMalloc(&d_data, 256 * sizeof(int));

	test_kernel<<<1, 256>>>(d_data);
	cudaDeviceSynchronize();

	cudaFree(d_data);
	printf("Success!\n");
	return 0;
}
