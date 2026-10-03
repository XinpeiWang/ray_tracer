#pragma once
// ray_hash.h -- the stateless per-ray random number pbrt-v4's stochastic alpha test uses.
//
// pbrt-v4 (cpu/primitive.cpp:57-71) does not threshold an alpha texture: for a = alpha(hit) < 1 it rejects
// the hit when HashFloat(ray.o, ray.d) > a (always when a <= 0). The hash depends only on the ray, so a
// shadow ray and the camera ray that produced it make independent choices without any RNG state. Both
// backends use this one function so a leaf edge dithers the same way on CPU and GPU.

#include <cstdint>
#include <cstring>
#include "cpu_gpu.h"

namespace ray_hash {

CPU_GPU std::uint32_t mix(std::uint32_t h) {          // murmur3 finalizer
	h ^= h >> 16; h *= 0x85ebca6bu;
	h ^= h >> 13; h *= 0xc2b2ae35u;
	h ^= h >> 16;
	return h;
}

CPU_GPU std::uint32_t bits(float f) {
#ifdef __CUDA_ARCH__
	return __float_as_uint(f);
#else
	std::uint32_t u;
	std::memcpy(&u, &f, sizeof(u));
	return u;
#endif
}

// Uniform in [0,1) from the ray's origin and direction.
CPU_GPU float hash01(float ox, float oy, float oz, float dx, float dy, float dz) {
	std::uint32_t h = 0x9e3779b9u;
	h = mix(h ^ bits(ox)); h = mix(h ^ bits(oy)); h = mix(h ^ bits(oz));
	h = mix(h ^ bits(dx)); h = mix(h ^ bits(dy)); h = mix(h ^ bits(dz));
	return static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
}

// pbrt-v4's alpha test: true when the hit is kept.
CPU_GPU bool alphaPasses(float a, float ox, float oy, float oz, float dx, float dy, float dz) {
	if (a >= 1.0f) return true;
	if (a <= 0.0f) return false;
	return hash01(ox, oy, oz, dx, dy, dz) <= a;
}

} // namespace ray_hash
