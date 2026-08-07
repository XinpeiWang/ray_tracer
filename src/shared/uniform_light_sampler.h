#pragma once
// ---------------------------------------------------------------------------
// uniform_light_sampler.h -- Uniform light sampler
//
// Mirrors pbrt-v4 UniformLightSampler (src/pbrt/lightsamplers.h).
//
// Selects one of n lights with equal probability 1/n.
// O(1) sample, O(1) PMF query.
//
// API matches power_light_sampler.h:
//   UniformLightSampler sampler(n);
//   int  idx = sampler.sample(u);   // u in [0,1)  -> light index, or -1
//   double p = sampler.pmf(i);      // always 1/n for valid i
//   int  sz  = sampler.size();      // n
//
// pbrt-v4 reference: src/pbrt/lightsamplers.h UniformLightSampler
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU inline
#   endif
#endif

class UniformLightSampler {
public:
	CPU_GPU UniformLightSampler() : n_(0) {}

	// Construct for n lights.
	CPU_GPU explicit UniformLightSampler(int n) : n_(n > 0 ? n : 0) {}

	// Sample a light index given u in [0,1).
	// Returns -1 if there are no lights.
	// Mirrors pbrt-v4: lightIndex = min((int)(u * lights.size()), size-1)
	CPU_GPU int sample(double u) const {
		if (n_ == 0) return -1;
		int idx = static_cast<int>(u * n_);
		if (idx >= n_) idx = n_ - 1;
		return idx;
	}

	// PMF of light index i: always 1/n for any valid i.
	// Returns 0 for invalid index or empty sampler.
	CPU_GPU double pmf(int i) const {
		if (n_ == 0 || i < 0 || i >= n_) return 0.0;
		return 1.0 / static_cast<double>(n_);
	}

	// Number of lights.
	CPU_GPU int size() const { return n_; }

private:
	int n_;
};
