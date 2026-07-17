// Minimal CUDA renderer prototype
// Renders a simple gradient image on the GPU and writes a PPM file
// Build: nvcc host.cu -o cuda_renderer.exe

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include <ctime>
#include <ctime>

#define CUDA_CHECK(call) do { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		std::exit(1); \
	} \
} while(0)

// Minimal vector and ray helpers for device
struct Vec3 { float x, y, z; __device__ Vec3(){} __device__ Vec3(float a,float b,float c):x(a),y(b),z(c){} __device__ Vec3 operator-(const Vec3& o) const { return Vec3(x-o.x,y-o.y,z-o.z); } __device__ Vec3 operator+(const Vec3& o) const { return Vec3(x+o.x,y+o.y,z+o.z); } __device__ Vec3 operator*(float s) const { return Vec3(x*s,y*s,z*s); } __device__ Vec3 operator/(float s) const { float inv = 1.0f/s; return Vec3(x*inv,y*inv,z*inv); } };

__device__ float dot(const Vec3 &a, const Vec3 &b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ Vec3 unit_vector(const Vec3 &v) { float len = sqrtf(dot(v,v)); return v / (len>0?len:1.0f); }

struct Ray { Vec3 orig; Vec3 dir; __device__ Ray(){} __device__ Ray(const Vec3&o,const Vec3&d):orig(o),dir(d){} __device__ Vec3 at(float t) const { return orig + dir * t; } };

// Sphere intersection: returns t if hit, else -1
__device__ float hit_sphere(const Vec3& center, float radius, const Ray& r) {
	Vec3 oc = r.orig - center;
	float a = dot(r.dir, r.dir);
	float b = 2.0f * dot(oc, r.dir);
	float c = dot(oc, oc) - radius*radius;
	float disc = b*b - 4*a*c;
	if (disc < 0.0f) return -1.0f;
	float sqrtd = sqrtf(disc);
	float t = (-b - sqrtd) / (2.0f*a);
	if (t > 0.001f) return t;
	t = (-b + sqrtd) / (2.0f*a);
	if (t > 0.001f) return t;
	return -1.0f;
}

// Simple xorshift32 PRNG for device
__device__ unsigned int xor_shift32(unsigned int &state) {
	// state must be non-zero
	state ^= state << 13;
	state ^= state >> 17;
	state ^= state << 5;
	return state;
}

__device__ float rand01(unsigned int &state) {
	unsigned int v = xor_shift32(state);
	// take lower 24 bits for good fraction
	return (v & 0x00FFFFFF) / 16777216.0f;
}

__global__ void render_kernel(unsigned char* img, int image_width, int image_height, int samples_per_pixel, unsigned int seed) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= image_width || j >= image_height) return;

	// camera
	float aspect = float(image_width) / float(image_height);
	float viewport_height = 2.0f;
	float viewport_width = aspect * viewport_height;
	Vec3 origin(0.0f, 0.0f, 0.0f);
	Vec3 horizontal(viewport_width, 0.0f, 0.0f);
	Vec3 vertical(0.0f, viewport_height, 0.0f);
	Vec3 lower_left = origin - horizontal/2.0f - vertical/2.0f - Vec3(0,0,1);

	// accumulate samples in floating point
	float accum_r = 0.0f;
	float accum_g = 0.0f;
	float accum_b = 0.0f;
	unsigned int state = (unsigned int)(i * 1973u + j * 9277u + seed);
	if (state == 0) state = 1;

	for (int s = 0; s < samples_per_pixel; ++s) {
		float ru = rand01(state);
		float rv = rand01(state);
		float u = (float(i) + ru) / float(image_width - 1);
		float v = (float(j) + rv) / float(image_height - 1);
		Vec3 dir = lower_left + horizontal * u + vertical * v - origin;
		dir = unit_vector(dir);
		Ray r(origin, dir);

		// simple sphere scene
		Vec3 sphere_center(0.0f, 0.0f, -1.0f);
		float sphere_r = 0.5f;
		float t = hit_sphere(sphere_center, sphere_r, r);
		if (t > 0.0f) {
			Vec3 p = r.at(t);
			Vec3 n = unit_vector(p - sphere_center);
			float rr = 0.5f * (n.x + 1.0f);
			float gg = 0.5f * (n.y + 1.0f);
			float bb = 0.5f * (n.z + 1.0f);
			accum_r += rr;
			accum_g += gg;
			accum_b += bb;
		} else {
			// background gradient
			float tbg = 0.5f*(dir.y + 1.0f);
			float rr = (1.0f - tbg) * 1.0f + tbg * 0.5f;
			float gg = (1.0f - tbg) * 1.0f + tbg * 0.7f;
			float bb = 1.0f;
			accum_r += rr;
			accum_g += gg;
			accum_b += bb;
		}
	}

	// average and gamma-correct (gamma=2)
	float inv_samples = 1.0f / float(samples_per_pixel);
	float out_r = sqrtf(accum_r * inv_samples);
	float out_g = sqrtf(accum_g * inv_samples);
	float out_b = sqrtf(accum_b * inv_samples);

	int idx = (j * image_width + i) * 3;
	img[idx+0] = (unsigned char)(fminf(fmaxf(out_r,0.0f),1.0f)*255.999f);
	img[idx+1] = (unsigned char)(fminf(fmaxf(out_g,0.0f),1.0f)*255.999f);
	img[idx+2] = (unsigned char)(fminf(fmaxf(out_b,0.0f),1.0f)*255.999f);
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

	int samples_per_pixel = 4;
	// allow override from argv
	if (argc >= 4) samples_per_pixel = std::atoi(argv[3]);

	dim3 block(16, 16);
	dim3 grid((image_width + block.x - 1) / block.x, (image_height + block.y - 1) / block.y);

	unsigned int seed = (unsigned int)std::time(nullptr);

	// Timing: measure kernel and copy durations with cudaEvent
	cudaEvent_t kstart, kend, cstart, cend;
	CUDA_CHECK(cudaEventCreate(&kstart));
	CUDA_CHECK(cudaEventCreate(&kend));
	CUDA_CHECK(cudaEventCreate(&cstart));
	CUDA_CHECK(cudaEventCreate(&cend));

	CUDA_CHECK(cudaEventRecord(kstart));
	render_kernel<<<grid, block>>>(d_img, image_width, image_height, samples_per_pixel, seed);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaEventRecord(kend));
	CUDA_CHECK(cudaEventSynchronize(kend));

	unsigned char* h_img = (unsigned char*)malloc(bytes);
	if (!h_img) {
		std::fprintf(stderr, "Out of host memory\n");
		CUDA_CHECK(cudaFree(d_img));
		return 1;
	}

	CUDA_CHECK(cudaEventRecord(cstart));
	CUDA_CHECK(cudaMemcpy(h_img, d_img, bytes, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaEventRecord(cend));
	CUDA_CHECK(cudaEventSynchronize(cend));

	float kernel_ms = 0.0f, copy_ms = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kstart, kend));
	CUDA_CHECK(cudaEventElapsedTime(&copy_ms, cstart, cend));

	std::printf("[CUDA LOG] image=%dx%d samples=%d bytes=%zu kernel=%.3fms memcpy=%.3fms\n",
				image_width, image_height, samples_per_pixel, bytes, kernel_ms, copy_ms);

	CUDA_CHECK(cudaEventDestroy(kstart));
	CUDA_CHECK(cudaEventDestroy(kend));
	CUDA_CHECK(cudaEventDestroy(cstart));
	CUDA_CHECK(cudaEventDestroy(cend));

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
