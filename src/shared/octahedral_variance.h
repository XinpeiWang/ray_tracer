// octahedral_variance.h
// Two compact pbrt-v4 utility types:
//
//   OctahedralVector  -- Lossily encodes a unit direction into 2 x uint16_t
//                        (4 bytes) using the octahedral equal-area mapping.
//                        Ported from pbrt-v4 util/vecmath.h.
//
//   VarianceEstimator<T> -- Welford online mean + variance accumulator with
//                           parallel Merge().
//                           Ported from pbrt-v4 util/sampling.h.
//
// Dependencies:
//   scalar_math.h  -- Clamp, Sqr
//
// References:
//   pbrt-v4 src/pbrt/util/vecmath.h   (OctahedralVector)
//   pbrt-v4 src/pbrt/util/sampling.h  (VarianceEstimator)
//   Meyer et al., "Encoding Normal Vectors using Optimized Spherical Mapping", 2010

#pragma once
#include <cmath>
#include <cstdint>
#include "scalar_math.h"

#include "cpu_gpu.h"

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4141)
#endif

// ===========================================================================
// OctahedralVector
//
// Encodes a unit 3-vector as two uint16_t values using the octahedral map:
//   - Project onto the unit L1 ball (divide by |x|+|y|+|z|)
//   - For z>=0, store (x,y) directly in [-1,1]^2
//   - For z<0, "fold" the octahedron's lower hemisphere into the square
//
// Construction: OctahedralVector(float x, float y, float z)
//               or from any type with .x .y .z members
// Decoding:     ToVec3() returns a normalised {x, y, z}
//
// Storage: 4 bytes (2 x uint16_t), precision ~0.02 degrees.
// ===========================================================================

struct OctahedralVector {
	// Constructors
	OctahedralVector() = default;

	CPU_GPU explicit OctahedralVector(float vx, float vy, float vz) {
		float len1 = std::abs(vx) + std::abs(vy) + std::abs(vz);
		vx /= len1;  vy /= len1;  vz /= len1;
		if (vz >= 0.f) {
			x = Encode(vx);
			y = Encode(vy);
		} else {
			// Fold lower hemisphere
			x = Encode((1.f - std::abs(vy)) * Sign(vx));
			y = Encode((1.f - std::abs(vx)) * Sign(vy));
		}
	}

	// Decode back to a normalised float3 triple
	CPU_GPU void ToVec3(float& ox, float& oy, float& oz) const {
		ox = -1.f + 2.f * (x / 65535.f);
		oy = -1.f + 2.f * (y / 65535.f);
		oz = 1.f - (std::abs(ox) + std::abs(oy));
		if (oz < 0.f) {
			float xo = ox;
			ox = (1.f - std::abs(oy)) * Sign(xo);
			oy = (1.f - std::abs(xo)) * Sign(oy);
		}
		// Normalize
		float len = std::sqrt(ox * ox + oy * oy + oz * oz);
		ox /= len;  oy /= len;  oz /= len;
	}

	// Raw storage
	uint16_t x = 0, y = 0;

private:
	CPU_GPU static float Sign(float v) { return std::copysign(1.f, v); }

	CPU_GPU static uint16_t Encode(float f) {
		// Map [-1,1] -> [0, 65535], rounding to nearest integer
		float clamped = Clamp((f + 1.f) / 2.f, 0.f, 1.f);
		return static_cast<uint16_t>(clamped * 65535.f + 0.5f);
	}
};

// ===========================================================================
// VarianceEstimator<T>
//
// Welford's online single-pass algorithm for mean and variance.
// Numerically stable; supports parallel Merge() of two estimators.
//
// Usage:
//   VarianceEstimator<double> ve;
//   for (auto s : samples) ve.Add(s);
//   double m = ve.Mean(), v = ve.Variance();
//
// Merge (parallel reduction):
//   ve1.Merge(ve2);  // ve1 now contains combined statistics
// ===========================================================================

template<typename T = double>
struct VarianceEstimator {
	CPU_GPU void Add(T val) {
		++n;
		T delta  = val - mean;
		mean += delta / static_cast<T>(n);
		T delta2 = val - mean;
		S    += delta * delta2;
	}

	CPU_GPU T Mean()     const { return mean; }
	CPU_GPU T Variance() const { return (n > 1) ? S / static_cast<T>(n - 1) : T(0); }
	CPU_GPU int64_t Count()    const { return n; }

	CPU_GPU T RelativeVariance() const {
		return (n < 1 || mean == T(0)) ? T(0) : Variance() / Mean();
	}

	// Parallel merge (Chan et al. 1979)
	CPU_GPU void Merge(const VarianceEstimator& ve) {
		if (ve.n == 0) return;
		S    = S + ve.S + Sqr(ve.mean - mean) * static_cast<T>(n) * static_cast<T>(ve.n)
						 / static_cast<T>(n + ve.n);
		mean = (static_cast<T>(n) * mean + static_cast<T>(ve.n) * ve.mean)
			   / static_cast<T>(n + ve.n);
		n   += ve.n;
	}

private:
	T       mean = T(0);
	T       S    = T(0);
	int64_t n    = 0;
};

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
