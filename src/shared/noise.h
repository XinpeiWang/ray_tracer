#pragma once
// ---------------------------------------------------------------------------
// noise.h -- Shared CPU/GPU Perlin noise, FBm, and Turbulence
//
// Mirrors pbrt-v4 util/noise.cpp (Noise, FBm, Turbulence) and
// textures.h (FBmTexture, MarbleTexture).
//
// Key upgrades over Book-3 perlin.h:
//   1. Quintic C2 smoothstep weight: 6t^5 - 15t^4 + 10t^3  (pbrt-v4 NoiseWeight)
//      vs. Book-3 cubic: 3t^2 - 2t^3  (only C1 -- causes visible seam artifacts)
//   2. Gradient noise via pbrt-v4 fixed permutation table + Grad()
//   3. Anti-aliased FBm: clamps octave count from ray footprint dpdx/dpdy,
//      blends partial octave with SmoothStep(nPartial, 0.3, 0.7)
//      lambda *= 1.99 per octave (pbrt-v4 exact)
//   4. Turbulence: |noise| sum + partial-octave blend toward 0.2 baseline
//
// Design rules (same as bxdfs.h):
//   - Plain template structs/functions, CPU_GPU tagged
//   - No virtual functions, no heap allocation
//   - T = double on CPU, float on GPU
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU inline
#   endif
#endif

#if defined(__CUDACC__)
#   include <math_functions.h>
#else
#   include <cmath>
#   include <cstdint>
#endif

// ---------------------------------------------------------------------------
// pbrt-v4 fixed permutation table (noise.cpp, 512 entries = 256 doubled)
// Identical table to pbrt-v4 -- do not change order.
// ---------------------------------------------------------------------------
namespace noise_detail {

static constexpr int NoisePermSize = 256;
static constexpr int NoisePerm[2 * NoisePermSize] = {
	151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,
	140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
	247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
	 57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
	 74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
	 60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
	 65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
	200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
	 52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
	207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
	119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
	129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
	218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
	 81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,
	106,157,184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,
	205, 93,222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,
	156,180,
	// second copy (identical to first 256 entries, starting at index 256)
	151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,
	  7,225,140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,
	  6,148,247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35,
	 11, 32, 57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168,
	 68,175, 74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,
	229,122, 60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,
	143, 54, 65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89,
	 18,169,200,196,135,130,116,188,159, 86,164,100,109,198,173,186,
	  3, 64, 52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82,
	 85,212,207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,
	170,213,119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,
	172,  9,129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,
	112,104,218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,
	162,241, 81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199
};

// pbrt-v4 Grad: maps hash bits to one of 12 gradient directions
template<typename T>
CPU_GPU T Grad(int x, int y, int z, T dx, T dy, T dz) {
	int h = NoisePerm[NoisePerm[NoisePerm[x & (NoisePermSize-1)]
										+ (y & (NoisePermSize-1))]
										+ (z & (NoisePermSize-1))];
	h &= 15;
	T u = (h < 8 || h == 12 || h == 13) ? dx : dy;
	T v = (h < 4 || h == 12 || h == 13) ? dy : dz;
	return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

// pbrt-v4 NoiseWeight: quintic C2 smoothstep  6t^5 - 15t^4 + 10t^3
template<typename T>
CPU_GPU T NoiseWeight(T t) {
	T t3 = t * t * t;
	T t4 = t3 * t;
	T t5 = t4 * t;
	return T(6)*t5 - T(15)*t4 + T(10)*t3;
}

template<typename T>
CPU_GPU T Lerp(T t, T a, T b) { return a + t * (b - a); }

template<typename T>
CPU_GPU T SmoothStep(T x, T a, T b) {
	if (a == b) return (x < a) ? T(0) : T(1);
	T t = (x - a) / (b - a);
	t = (t < T(0)) ? T(0) : (t > T(1)) ? T(1) : t;
	return t * t * (T(3) - T(2)*t);
}

template<typename T>
CPU_GPU T safe_log2(T x) {
#if defined(__CUDACC__)
	return log2f(x > T(0) ? x : T(1e-30f));
#else
	return std::log2(x > T(0) ? x : T(1e-30));
#endif
}

template<typename T>
CPU_GPU T safe_abs(T x) {
#if defined(__CUDACC__)
	return fabsf(x);
#else
	return std::abs(x);
#endif
}

template<typename T>
CPU_GPU T clamp(T x, T lo, T hi) {
	return x < lo ? lo : (x > hi ? hi : x);
}

} // namespace noise_detail

// ---------------------------------------------------------------------------
// Noise(x,y,z) -- scalar gradient Perlin noise in [-1,1]
// Direct port of pbrt-v4 Noise(Float x, Float y, Float z)
// ---------------------------------------------------------------------------
template<typename T>
CPU_GPU T perlin_noise(T x, T y, T z) {
	using namespace noise_detail;
	// Wrap to avoid int overflow on large coordinates
	// pbrt-v4: fmod(x, Float(1 << 30))
	const T wrap = T(1 << 30);
#if defined(__CUDACC__)
	x = fmodf(x, wrap); y = fmodf(y, wrap); z = fmodf(z, wrap);
#else
	x = std::fmod(x, wrap); y = std::fmod(y, wrap); z = std::fmod(z, wrap);
#endif

	int ix = (int)std::floor((double)x);
	int iy = (int)std::floor((double)y);
	int iz = (int)std::floor((double)z);
	T dx = x - T(ix), dy = y - T(iy), dz = z - T(iz);

	ix &= NoisePermSize - 1;
	iy &= NoisePermSize - 1;
	iz &= NoisePermSize - 1;

	T w000 = Grad<T>(ix,   iy,   iz,   dx,     dy,     dz    );
	T w100 = Grad<T>(ix+1, iy,   iz,   dx-T(1),dy,     dz    );
	T w010 = Grad<T>(ix,   iy+1, iz,   dx,     dy-T(1),dz    );
	T w110 = Grad<T>(ix+1, iy+1, iz,   dx-T(1),dy-T(1),dz    );
	T w001 = Grad<T>(ix,   iy,   iz+1, dx,     dy,     dz-T(1));
	T w101 = Grad<T>(ix+1, iy,   iz+1, dx-T(1),dy,     dz-T(1));
	T w011 = Grad<T>(ix,   iy+1, iz+1, dx,     dy-T(1),dz-T(1));
	T w111 = Grad<T>(ix+1, iy+1, iz+1, dx-T(1),dy-T(1),dz-T(1));

	// Quintic C2 interpolation weights
	T wx = NoiseWeight(dx), wy = NoiseWeight(dy), wz = NoiseWeight(dz);

	T x00 = Lerp(wx, w000, w100);
	T x10 = Lerp(wx, w010, w110);
	T x01 = Lerp(wx, w001, w101);
	T x11 = Lerp(wx, w011, w111);
	T y0  = Lerp(wy, x00, x10);
	T y1  = Lerp(wy, x01, x11);
	return Lerp(wz, y0, y1);
}

// ---------------------------------------------------------------------------
// FBm -- anti-aliased fractional Brownian motion
// pbrt-v4: FBm(Point3f p, Vector3f dpdx, Vector3f dpdy, Float omega, int maxOctaves)
//
// dpdx_len2 / dpdy_len2: squared lengths of ray footprint partial derivatives
// omega: persistence (roughness), 0 < omega < 1
// max_octaves: maximum number of octaves
// ---------------------------------------------------------------------------
template<typename T>
CPU_GPU T fbm(T px, T py, T pz,
			  T dpdx_len2, T dpdy_len2,
			  T omega, int max_octaves) {
	using namespace noise_detail;

	// Compute anti-aliased octave count from footprint
	T len2 = dpdx_len2 > dpdy_len2 ? dpdx_len2 : dpdy_len2;
	T n_float = clamp(-T(1) - safe_log2(len2) / T(2), T(0), T(max_octaves));
	int n_int = (int)n_float;  // floor

	T sum = T(0), lambda = T(1), o = T(1);
	for (int i = 0; i < n_int; ++i) {
		sum    += o * perlin_noise<T>(lambda*px, lambda*py, lambda*pz);
		lambda *= T(1.99);
		o      *= omega;
	}
	// Partial octave blended with SmoothStep(nPartial, 0.3, 0.7)
	T n_partial = n_float - T(n_int);
	sum += o * SmoothStep(n_partial, T(0.3), T(0.7)) *
			   perlin_noise<T>(lambda*px, lambda*py, lambda*pz);

	return sum;
}

// Convenience overload: no antialiasing (unit footprint, uses all octaves)
template<typename T>
CPU_GPU T fbm_simple(T px, T py, T pz, T omega, int max_octaves) {
	// dpdx_len2 = dpdy_len2 = 0 => n = max_octaves
	T sum = T(0), lambda = T(1), o = T(1);
	for (int i = 0; i < max_octaves; ++i) {
		sum    += o * perlin_noise<T>(lambda*px, lambda*py, lambda*pz);
		lambda *= T(1.99);
		o      *= omega;
	}
	return sum;
}

// ---------------------------------------------------------------------------
// Turbulence -- |FBm| variant with partial-octave baseline clamp
// pbrt-v4: Turbulence(Point3f p, Vector3f dpdx, Vector3f dpdy, Float omega, int maxOctaves)
// ---------------------------------------------------------------------------
template<typename T>
CPU_GPU T turbulence(T px, T py, T pz,
					 T dpdx_len2, T dpdy_len2,
					 T omega, int max_octaves) {
	using namespace noise_detail;

	T len2 = dpdx_len2 > dpdy_len2 ? dpdx_len2 : dpdy_len2;
	T n_float = clamp(-T(1) - safe_log2(len2) / T(2), T(0), T(max_octaves));
	int n_int = (int)n_float;

	T sum = T(0), lambda = T(1), o = T(1);
	for (int i = 0; i < n_int; ++i) {
		sum    += o * safe_abs(perlin_noise<T>(lambda*px, lambda*py, lambda*pz));
		lambda *= T(1.99);
		o      *= omega;
	}

	// Partial octave: blend from 0.2 baseline toward |noise| using SmoothStep
	T n_partial = n_float - T(n_int);
	sum += o * noise_detail::Lerp(
			   SmoothStep(n_partial, T(0.3), T(0.7)),
			   T(0.2),
			   safe_abs(perlin_noise<T>(lambda*px, lambda*py, lambda*pz)));

	// Remaining octaves contribute constant 0.2 (pbrt-v4 exact)
	for (int i = n_int; i < max_octaves; ++i) {
		sum += o * T(0.2);
		o   *= omega;
	}

	return sum;
}

// Convenience overload: no antialiasing
template<typename T>
CPU_GPU T turbulence_simple(T px, T py, T pz, T omega, int max_octaves) {
	T sum = T(0), lambda = T(1), o = T(1);
	for (int i = 0; i < max_octaves; ++i) {
		sum    += o * noise_detail::safe_abs(perlin_noise<T>(lambda*px, lambda*py, lambda*pz));
		lambda *= T(1.99);
		o      *= omega;
	}
	return sum;
}
