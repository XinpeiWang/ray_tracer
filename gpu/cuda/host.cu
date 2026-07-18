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
#include "cuda_scene.h"

#define CUDA_CHECK(call) do { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		std::exit(1); \
	} \
} while(0)

// Minimal vector and ray helpers for device
struct Vec3 { 
	float x, y, z; 
	__device__ Vec3(){} 
	__device__ Vec3(float a,float b,float c):x(a),y(b),z(c){} 
	__device__ Vec3 operator-(const Vec3& o) const { return Vec3(x-o.x,y-o.y,z-o.z); } 
	__device__ Vec3 operator-() const { return Vec3(-x,-y,-z); }  // unary minus
	__device__ Vec3 operator+(const Vec3& o) const { return Vec3(x+o.x,y+o.y,z+o.z); } 
	__device__ Vec3 operator*(float s) const { return Vec3(x*s,y*s,z*s); } 
	__device__ Vec3 operator*(const Vec3& o) const { return Vec3(x*o.x, y*o.y, z*o.z); }  // component-wise multiply
	__device__ Vec3 operator/(float s) const { float inv = 1.0f/s; return Vec3(x*inv,y*inv,z*inv); }
};

__device__ float dot(const Vec3 &a, const Vec3 &b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ Vec3 cross(const Vec3 &a, const Vec3 &b) { return Vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); }
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

// Device utility math for path tracing
__device__ Vec3 reflect(const Vec3& v, const Vec3& n) {
	return v - n * (2.0f * dot(v, n));
}

__device__ bool refract(const Vec3& v, const Vec3& n, float eta_over_eta_prime, Vec3& refracted) {
	Vec3 uv = unit_vector(v);
	float dt = dot(uv, n);
	float discr = 1.0f - eta_over_eta_prime*eta_over_eta_prime*(1 - dt*dt);
	if (discr > 0) {
		refracted = (uv - n*dt) * eta_over_eta_prime - n * sqrtf(discr);
		return true;
	}
	return false;
}

__device__ float schlick(float cosine, float ref_idx) {
	float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
	r0 = r0*r0;
	return r0 + (1.0f - r0)*powf(1.0f - cosine, 5.0f);
}

// Cosine-weighted random direction in hemisphere
__device__ Vec3 random_cosine_direction(unsigned int &state) {
	float r1 = rand01(state);
	float r2 = rand01(state);
	float z = sqrtf(1.0f - r2);
	float phi = 2.0f * 3.1415926535897932385f * r1;
	float x = cosf(phi) * sqrtf(r2);
	float y = sinf(phi) * sqrtf(r2);
	return Vec3(x, y, z);
}

// Full path-tracing kernel with multiple bounces
__global__ void render_kernel_path(const SpherePOD* spheres, int nspheres, const QuadPOD* quads, int nquads, const MaterialPOD* mats, const CameraPOD* camera, unsigned char* img, int image_width, int image_height, int samples_per_pixel, int max_depth, unsigned int seed) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= image_width || j >= image_height) return;

	// camera from CameraPOD struct
	Vec3 lookfrom(camera->lookfrom_x, camera->lookfrom_y, camera->lookfrom_z);
	Vec3 lookat(camera->lookat_x, camera->lookat_y, camera->lookat_z);
	Vec3 vup(camera->vup_x, camera->vup_y, camera->vup_z);
	float vfov = camera->vfov;
	float aspect = camera->aspect_ratio;

	float theta = vfov * 3.1415926535897932385f / 180.0f;
	float h = tanf(theta/2.0f);
	float viewport_height = 2.0f * h;
	float viewport_width = aspect * viewport_height;

	Vec3 w = unit_vector(lookfrom - lookat);
	Vec3 u = unit_vector(cross(vup, w));
	Vec3 v = cross(w, u);

	Vec3 origin = lookfrom;
	Vec3 horizontal = u * viewport_width;
	Vec3 vertical = v * viewport_height;
	Vec3 lower_left = origin - horizontal/2.0f - vertical/2.0f - w;

	unsigned int state = (unsigned int)(i * 1973u + j * 9277u + seed);
	if (state == 0) state = 1;

	float accum_r = 0.0f, accum_g = 0.0f, accum_b = 0.0f;

	for (int s = 0; s < samples_per_pixel; ++s) {
		float ru = rand01(state);
		float rv = rand01(state);
		float u_samp = (float(i) + ru) / float(image_width - 1);
		float v_samp = (float(j) + rv) / float(image_height - 1);
		Vec3 dir = lower_left + horizontal * u_samp + vertical * v_samp - origin;
		Ray ray(origin, unit_vector(dir));

		// Path tracer loop
		Vec3 throughput(1.0f, 1.0f, 1.0f);
		Vec3 radiance(0.0f, 0.0f, 0.0f);
		for (int depth = 0; depth < max_depth; ++depth) {
			// find closest hit
			float closest_t = 1e30f;
			int hit_idx = -1;
			Vec3 hit_point, hit_normal;
			bool hit_is_quad = false;
			// spheres
			for (int k = 0; k < nspheres; ++k) {
				SpherePOD sp = spheres[k];
				Vec3 center(sp.cx, sp.cy, sp.cz);
				float t = hit_sphere(center, sp.radius, ray);
				if (t > 0.001f && t < closest_t) { closest_t = t; hit_idx = k; hit_point = ray.at(t); hit_normal = unit_vector(hit_point - center); hit_is_quad = false; }
			}
			// quads
			for (int qidx = 0; qidx < nquads; ++qidx) {
				QuadPOD qp = quads[qidx];
				Vec3 Q(qp.Qx, qp.Qy, qp.Qz);
				Vec3 uu(qp.ux, qp.uy, qp.uz);
				Vec3 vv(qp.vx, qp.vy, qp.vz);
				Vec3 nrm = unit_vector(cross(uu, vv));
				float denom = dot(nrm, ray.dir);
				if (fabsf(denom) < 1e-8f) continue;
				float D = dot(nrm, Q);
				float t = (D - dot(nrm, ray.orig)) / denom;
				if (!(t > 0.001f && t < closest_t)) continue;
				Vec3 intersection = ray.at(t);
				Vec3 planar = intersection - Q;
				Vec3 wtmp = cross(uu, vv);
				float alpha = dot(wtmp, cross(planar, vv)) / dot(wtmp,wtmp);
				float beta = dot(wtmp, cross(uu, planar)) / dot(wtmp,wtmp);
				if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f) continue;
				closest_t = t; hit_idx = qidx; hit_point = intersection; hit_normal = nrm; hit_is_quad = true;
			}

			if (hit_idx == -1) {
				// Cornell box has black background (closed room)
				// No environment light contribution
				break;
			}

			// fetch material
			int matIndex = hit_is_quad ? quads[hit_idx].matIndex : spheres[hit_idx].matIndex;
			MaterialPOD mm = mats[matIndex];

			// emit
			if (mm.is_emissive) {
				radiance = radiance + throughput * Vec3(mm.r, mm.g, mm.b);
				break;
			}

			// handle scattering
			// lambertian
			if (mm.type == 0) {
				// cosine-weighted hemisphere
				Vec3 w_normal = hit_normal;
				Vec3 a = fabsf(w_normal.x) > 0.9f ? Vec3(0,1,0) : Vec3(1,0,0);
				Vec3 uvec = unit_vector(cross(a, w_normal));
				Vec3 vvec = cross(w_normal, uvec);
				Vec3 d = random_cosine_direction(state);
				Vec3 scatter_dir = unit_vector(uvec * d.x + vvec * d.y + w_normal * d.z);
				ray = Ray(hit_point + scatter_dir * 0.01f, scatter_dir);
				throughput = throughput * Vec3(mm.r, mm.g, mm.b);
				continue;
			}

			// metal
			if (mm.type == 1) {
				Vec3 reflected = reflect(unit_vector(ray.dir), hit_normal);
				Vec3 scatter_dir = unit_vector(reflected + Vec3((rand01(state)-0.5f)*2.0f, (rand01(state)-0.5f)*2.0f, (rand01(state)-0.5f)*2.0f) * mm.fuzz);
				ray = Ray(hit_point + scatter_dir * 0.01f, scatter_dir);
				throughput = throughput * Vec3(mm.r, mm.g, mm.b);
				continue;
			}

			// dielectric
			if (mm.type == 2) {
				Vec3 unit_dir = unit_vector(ray.dir);
				float cos_theta = fminf(dot(-unit_dir, hit_normal), 1.0f);
				float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
				float etai_over_etat = 1.0f / mm.ref_idx;
				Vec3 refracted;
				if (etai_over_etat * sin_theta > 1.0f) {
					// total internal reflection
					Vec3 reflected = reflect(unit_dir, hit_normal);
					ray = Ray(hit_point + reflected * 0.01f, reflected);
					continue;
				}
				float reflect_prob = schlick(cos_theta, mm.ref_idx);
				if (rand01(state) < reflect_prob) {
					Vec3 reflected = reflect(unit_dir, hit_normal);
					ray = Ray(hit_point + reflected * 0.01f, reflected);
				} else {
					if (refract(unit_dir, hit_normal, etai_over_etat, refracted)) {
						ray = Ray(hit_point + refracted * 0.01f, refracted);
					} else {
						Vec3 reflected = reflect(unit_dir, hit_normal);
						ray = Ray(hit_point + reflected * 0.01f, reflected);
					}
				}
				continue;
			}
		}

		accum_r += radiance.x; accum_g += radiance.y; accum_b += radiance.z;
	}

	float inv_samples = 1.0f / float(samples_per_pixel);
	float out_r = sqrtf((accum_r * inv_samples));
	float out_g = sqrtf((accum_g * inv_samples));
	float out_b = sqrtf((accum_b * inv_samples));

	// Flip Y to write bottom-to-top (PPM expects top-to-bottom scanlines, but camera renders bottom-to-top)
	int flipped_j = image_height - 1 - j;
	int idx = (flipped_j * image_width + i) * 3;
	img[idx+0] = (unsigned char)(fminf(fmaxf(out_r,0.0f),1.0f)*255.999f);
	img[idx+1] = (unsigned char)(fminf(fmaxf(out_g,0.0f),1.0f)*255.999f);
	img[idx+2] = (unsigned char)(fminf(fmaxf(out_b,0.0f),1.0f)*255.999f);
}

// Keep the previous simpler kernel for fallback or tests
__global__ void render_kernel_serial(const SpherePOD* spheres, int nspheres, const QuadPOD* quads, int nquads, const MaterialPOD* mats, const float* cam_pod, unsigned char* img, int image_width, int image_height, int samples_per_pixel, unsigned int seed) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= image_width || j >= image_height) return;

	// camera from cam_pod array: lookfrom, lookat, vup, vfov, aspect
	Vec3 lookfrom(cam_pod[0], cam_pod[1], cam_pod[2]);
	Vec3 lookat(cam_pod[3], cam_pod[4], cam_pod[5]);
	Vec3 vup(cam_pod[6], cam_pod[7], cam_pod[8]);
	float vfov = cam_pod[9];
	float aspect = cam_pod[10];

	// build camera basis like the CPU camera
	float theta = vfov * 3.1415926535897932385f / 180.0f;
	float h = tanf(theta/2.0f);
	float viewport_height = 2.0f * h;
	float viewport_width = aspect * viewport_height;

	Vec3 w = unit_vector(lookfrom - lookat);
	Vec3 u = unit_vector(cross(vup, w));
	Vec3 v = cross(w, u);

	Vec3 origin = lookfrom;
	Vec3 horizontal = u * viewport_width * cam_pod[11]; // focus distance if provided
	Vec3 vertical = v * viewport_height * cam_pod[11];
	Vec3 lower_left = origin - horizontal/2.0f - vertical/2.0f - w;

	float accum_r = 0.0f, accum_g = 0.0f, accum_b = 0.0f;
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

		// trace against spheres and quads list (spheres first)
		float closest_t = 1e30f;
		int hit_idx = -1;
		Vec3 hit_point, hit_normal;
		for (int k = 0; k < nspheres; ++k) {
			SpherePOD sp = spheres[k];
			Vec3 center(sp.cx, sp.cy, sp.cz);
			float t = hit_sphere(center, sp.radius, r);
			if (t > 0.0f && t < closest_t) {
				closest_t = t;
				hit_idx = k;
				hit_point = r.at(t);
				hit_normal = unit_vector(hit_point - center);
			}

		// quads intersection: ray-plane + barycentric check (use u,v vectors)
		for (int qidx = 0; qidx < nquads; ++qidx) {
			QuadPOD qp = quads[qidx];
			Vec3 Q(qp.Qx, qp.Qy, qp.Qz);
			Vec3 uu(qp.ux, qp.uy, qp.uz);
			Vec3 vv(qp.vx, qp.vy, qp.vz);
			Vec3 n = unit_vector(cross(uu, vv));
			float denom = dot(n, r.dir);
			if (fabsf(denom) < 1e-8f) continue;
			float D = dot(n, Q);
			float t = (D - dot(n, r.orig)) / denom;
			if (!(t > 0.001f && t < closest_t)) continue;
			Vec3 intersection = r.at(t);
			Vec3 planar = intersection - Q;
			// compute alpha/beta via solving linear system: planar = alpha * uu + beta * vv
			// Use least squares: solve 2x2 via cross products using w = cross(uu,vv)
			Vec3 w = cross(uu, vv);
			float alpha = dot(w, cross(planar, vv)) / dot(w,w);
			float beta = dot(w, cross(uu, planar)) / dot(w,w);
			if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f) continue;
			closest_t = t;
			hit_idx = nspheres + qidx; // encode quads after spheres
			hit_point = intersection;
			hit_normal = n;
		}
		}

		if (hit_idx >= 0) {
			// Determine whether hit a sphere or a quad
			MaterialPOD mm;
			bool isQuad = false;
			int matIdx = -1;
			if (hit_idx < nspheres) {
				matIdx = spheres[hit_idx].matIndex;
			} else {
				int qidx = hit_idx - nspheres;
				matIdx = quads[qidx].matIndex;
				isQuad = true;
			}
			mm = mats[matIdx];

			// Simple emission handling
			if (mm.is_emissive) {
				accum_r += mm.r; accum_g += mm.g; accum_b += mm.b;
			} else {
				// Lambertian diffuse as cosine-weighted sample
				Vec3 target_dir = random_cosine_direction(state);
				// Build orthonormal basis
				Vec3 w = hit_normal;
				Vec3 a = fabsf(w.x) > 0.9f ? Vec3(0,1,0) : Vec3(1,0,0);
				Vec3 u = unit_vector(cross(a, w));
				Vec3 v = cross(w, u);
				Vec3 scatter_dir = unit_vector(u * target_dir.x + v * target_dir.y + w * target_dir.z);
				// Simple attenuation by albedo
				accum_r += mm.r * 0.0f; // scattering contributes via subsequent bounces handled in path tracer
				accum_g += mm.g * 0.0f;
				accum_b += mm.b * 0.0f;
				// For this per-sample kernel we only approximate by adding direct color
				accum_r += 0.2f * mm.r; accum_g += 0.2f * mm.g; accum_b += 0.2f * mm.b;
			}
		} else {
			float tbg = 0.5f*(dir.y + 1.0f);
			float rr = (1.0f - tbg) * 1.0f + tbg * 0.5f;
			float gg = (1.0f - tbg) * 1.0f + tbg * 0.7f;
			float bb = 1.0f;
			accum_r += rr; accum_g += gg; accum_b += bb;
		}
	}

	float inv_samples = 1.0f / float(samples_per_pixel);
	float out_r = sqrtf(accum_r * inv_samples);
	float out_g = sqrtf(accum_g * inv_samples);
	float out_b = sqrtf(accum_b * inv_samples);

	int idx = (j * image_width + i) * 3;
	img[idx+0] = (unsigned char)(fminf(fmaxf(out_r,0.0f),1.0f)*255.999f);
	img[idx+1] = (unsigned char)(fminf(fmaxf(out_g,0.0f),1.0f)*255.999f);
	img[idx+2] = (unsigned char)(fminf(fmaxf(out_b,0.0f),1.0f)*255.999f);
}

// Keep original simple kernel for standalone builds
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

// When building in-process (linked into the main C++ exe) we must avoid
// defining a second `main`. Guard the standalone host main with a macro
#ifndef BUILD_CUDA_STANDALONE
#define BUILD_CUDA_STANDALONE 0
#endif

#if BUILD_CUDA_STANDALONE
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
#endif // BUILD_CUDA_STANDALONE
