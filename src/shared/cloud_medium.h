#pragma once
// ---------------------------------------------------------------------------
// cloud_medium.h -- Procedural cloud volume medium
//
// Mirrors pbrt-v4 CloudMedium (src/pbrt/media.h, Ã‚Â§11.4).
//
// CloudMedium<T> is a procedural heterogeneous medium whose density is
// computed from 5-octave Perlin FBm noise with optional wispiness perturbation
// (gradient noise) and an altitude-based falloff term.  The medium lives
// inside an axis-aligned bounding box in "medium space"; the caller supplies
// an affine transform (3Ãƒâ€”3 matrix + translation) mapping world-space points
// into medium space.
//
// Key design choices (same as grid_medium.h / bxdfs.h):
//   - Plain template struct, CPU_GPU tagged, header-only
//   - No virtual functions, no heap allocation in hot paths
//   - T = double on CPU, float on GPU
//   - Uses CloudMajorantIterator<T> defined here (sigma_t is taken as
//     the maximum possible, i.e. sigma_a + sigma_s, over the whole volume)
//
// Public interface
//   CloudMedium<T>::make(...)            -- factory (value semantics)
//   T    density(px, py, pz)            -- [0,1] density at medium-space point
//   void sample_point(wx,wy,wz,         -- evaluate sigma_a/s at world point
//                     &sigma_a, &sigma_s)
//   bool sample_ray(ray_o,ray_d,tmax,   -- ray/AABB overlap Ã¢â€ â€™ [tMin,tMax] with
//                   &tMin,&tMax,        --   majorant sigma_t
//                   &sigma_maj)
//
// Reference: pbrt-v4 CloudMedium, util/noise.cpp (DNoise), media.h
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU
#   endif
#endif

#include <cmath>
#include <algorithm>
#include "noise.h"
#include "grid_medium.h"

// ---------------------------------------------------------------------------
// CloudMajorantIterator<T>
// Scalar (single-channel) homogeneous majorant iterator for CloudMedium.
// Mirrors pbrt-v4 HomogeneousMajorantIterator (media.h Ã‚Â§11.4) but uses a
// single sigma_t value rather than RGB channels (ratio_tracking.h variant).
// Yields exactly one segment [tMin, tMax] with constant majorant sigma_t,
// then returns empty on subsequent calls.
// ---------------------------------------------------------------------------
template<typename T>
struct CloudMajorantIterator {
	T    tMin     = T(0);
	T    tMax     = T(0);
	T    sigma_t  = T(0);
	bool valid    = false;   // true iff the segment has not been consumed
	bool hit      = false;   // true iff ray intersected the medium bounds

	// Empty iterator (no medium overlap).
	CPU_GPU CloudMajorantIterator() = default;

	// Iterator with one valid segment.
	CPU_GPU CloudMajorantIterator(T tMin_, T tMax_, T sig_t)
		: tMin(tMin_), tMax(tMax_), sigma_t(sig_t), valid(true), hit(true) {}

	// Returns true and fills out_tMin/tMax/sigma_t for the next segment.
	// Returns false when exhausted.  Mirrors pbrt-v4 Next().
	CPU_GPU bool next(T& out_tMin, T& out_tMax, T& out_sigma_t) {
		if (!valid) return false;
		out_tMin    = tMin;
		out_tMax    = tMax;
		out_sigma_t = sigma_t;
		valid = false;
		return true;
	}
};

// ---------------------------------------------------------------------------
// cloud_detail -- DNoise (gradient perturbation) for wispiness
// Mirrors pbrt-v4 DNoise(Point3f p) in util/noise.cpp.
// Returns the finite-difference gradient of Perlin noise at (px,py,pz).
// ---------------------------------------------------------------------------
namespace cloud_detail {

template<typename T>
CPU_GPU void dnoise(T px, T py, T pz,
					T& gx, T& gy, T& gz) {
	const T delta = T(0.01);
	T n  = perlin_noise<T>(px, py, pz);
	gx = (perlin_noise<T>(px + delta, py,       pz      ) - n) / delta;
	gy = (perlin_noise<T>(px,        py + delta, pz      ) - n) / delta;
	gz = (perlin_noise<T>(px,        py,         pz + delta) - n) / delta;
}

// clamp to [lo, hi]
template<typename T>
CPU_GPU T clamp01(T x) {
	return x < T(0) ? T(0) : (x > T(1) ? T(1) : x);
}

} // namespace cloud_detail

// ---------------------------------------------------------------------------
// CloudMedium<T>
// ---------------------------------------------------------------------------
template<typename T>
struct CloudMedium {
	// AABB of the medium in *medium space* (unit cube by default).
	T bounds_min[3];
	T bounds_max[3];

	// Affine transform: world Ã¢â€ â€™ medium space.
	// Stored as column-major 3Ãƒâ€”3 matrix + translation.
	// world_to_medium(p) = mat * p + translate
	T mat[9];        // row-major 3Ãƒâ€”3
	T translate[3];

	// Scattering coefficients (spectrally flat scalars here).
	T sigma_a;
	T sigma_s;

	// Phase function asymmetry (Henyey-Greenstein g).
	T phase_g;

	// Density controls (pbrt-v4 params: density, wispiness, frequency).
	T density;     // overall density scale
	T wispiness;   // amount of gradient-noise perturbation [0 = none]
	T frequency;   // spatial frequency multiplier

	// -----------------------------------------------------------------------
	// Factory
	// -----------------------------------------------------------------------
	CPU_GPU static CloudMedium make(
			T bmin_x, T bmin_y, T bmin_z,
			T bmax_x, T bmax_y, T bmax_z,
			const T* world_to_medium_mat3x3,   // row-major
			const T* world_to_medium_translate,
			T sigma_a_, T sigma_s_, T phase_g_,
			T density_, T wispiness_, T frequency_) {
		CloudMedium c;
		c.bounds_min[0] = bmin_x; c.bounds_min[1] = bmin_y; c.bounds_min[2] = bmin_z;
		c.bounds_max[0] = bmax_x; c.bounds_max[1] = bmax_y; c.bounds_max[2] = bmax_z;
		for (int i = 0; i < 9; ++i) c.mat[i] = world_to_medium_mat3x3[i];
		for (int i = 0; i < 3; ++i) c.translate[i] = world_to_medium_translate[i];
		c.sigma_a   = sigma_a_;
		c.sigma_s   = sigma_s_;
		c.phase_g   = phase_g_;
		c.density   = density_;
		c.wispiness = wispiness_;
		c.frequency = frequency_;
		return c;
	}

	// Convenience factory: identity transform, unit cube [0,1]^3.
	CPU_GPU static CloudMedium make_unit(
			T sigma_a_, T sigma_s_, T phase_g_,
			T density_, T wispiness_, T frequency_) {
		T id[9] = { T(1),T(0),T(0),  T(0),T(1),T(0),  T(0),T(0),T(1) };
		T tr[3] = { T(0), T(0), T(0) };
		return make(T(0),T(0),T(0), T(1),T(1),T(1), id, tr,
					sigma_a_, sigma_s_, phase_g_, density_, wispiness_, frequency_);
	}

	// -----------------------------------------------------------------------
	// world_to_medium: apply affine transform
	// -----------------------------------------------------------------------
	CPU_GPU void world_to_medium_pt(T wx, T wy, T wz,
									T& mx, T& my, T& mz) const {
		mx = mat[0]*wx + mat[1]*wy + mat[2]*wz + translate[0];
		my = mat[3]*wx + mat[4]*wy + mat[5]*wz + translate[1];
		mz = mat[6]*wx + mat[7]*wy + mat[8]*wz + translate[2];
	}

	// -----------------------------------------------------------------------
	// density(px, py, pz) -- evaluate at a point in *medium space*
	// Mirrors pbrt-v4 CloudMedium::Density(Point3f p).
	// -----------------------------------------------------------------------
	CPU_GPU T compute_density(T px, T py, T pz) const {
		// Scale by frequency
		T ppx = frequency * px;
		T ppy = frequency * py;
		T ppz = frequency * pz;

		// Optional wispiness: perturb pp with 2 octaves of gradient noise
		if (wispiness > T(0)) {
			T vomega = T(0.05) * wispiness;
			T vlambda = T(10);
			for (int i = 0; i < 2; ++i) {
				T gx, gy, gz;
				cloud_detail::dnoise<T>(vlambda * ppx, vlambda * ppy, vlambda * ppz,
										gx, gy, gz);
				ppx += vomega * gx;
				ppy += vomega * gy;
				ppz += vomega * gz;
				vomega  *= T(0.5);
				vlambda *= T(1.99);
			}
		}

		// Sum 5 octaves of noise (pbrt-v4: omega=0.5, lambda starts at 1)
		T d = T(0);
		T omega  = T(0.5);
		T lambda = T(1);
		for (int i = 0; i < 5; ++i) {
			d      += omega * perlin_noise<T>(lambda * ppx, lambda * ppy, lambda * ppz);
			omega  *= T(0.5);
			lambda *= T(1.99);
		}

		// Altitude falloff: decrease density with height (py in medium space).
		// pbrt-v4: d = Clamp((1-p.y)*4.5*density*d, 0, 1)
		//          d += 2 * max(0, 0.5 - p.y)
		d = cloud_detail::clamp01<T>((T(1) - py) * T(4.5) * density * d);
		T extra = T(2) * std::max(T(0), T(0.5) - py);
		d = cloud_detail::clamp01<T>(d + extra);

		return d;
	}

	// -----------------------------------------------------------------------
	// sample_point: evaluate scattering coefficients at a world-space point.
	// Mirrors pbrt-v4 CloudMedium::SamplePoint.
	// -----------------------------------------------------------------------
	CPU_GPU void sample_point(T wx, T wy, T wz,
							  T& out_sigma_a, T& out_sigma_s) const {
		T mx, my, mz;
		world_to_medium_pt(wx, wy, wz, mx, my, mz);
		T d = compute_density(mx, my, mz);
		out_sigma_a = d * sigma_a;
		out_sigma_s = d * sigma_s;
	}

	// -----------------------------------------------------------------------
	// sample_ray: compute ray/AABB overlap and return a CloudMajorantIterator.
	// The majorant sigma_t = sigma_a + sigma_s (whole-volume upper bound).
	// Mirrors pbrt-v4 CloudMedium::SampleRay.
	// -----------------------------------------------------------------------
	CPU_GPU CloudMajorantIterator<T> sample_ray(
			const T* ray_o, const T* ray_d, T ray_tmax) const {
		// Transform ray origin and direction to medium space.
		T mo[3], md[3];
		world_to_medium_pt(ray_o[0], ray_o[1], ray_o[2], mo[0], mo[1], mo[2]);
		// Direction transforms by the matrix only (no translation).
		for (int i = 0; i < 3; ++i) {
			md[i] = mat[3*i+0]*ray_d[0] + mat[3*i+1]*ray_d[1] + mat[3*i+2]*ray_d[2];
		}

		// Ray/AABB slab test
		T tMin, tMax;
		Bounds3<T> b(bounds_min[0], bounds_min[1], bounds_min[2],
					 bounds_max[0], bounds_max[1], bounds_max[2]);
		if (!b.intersect_ray(mo, md, ray_tmax, &tMin, &tMax))
			return CloudMajorantIterator<T>{};

		T sigma_t = sigma_a + sigma_s;
		return CloudMajorantIterator<T>(tMin, tMax, sigma_t);
	}
};
