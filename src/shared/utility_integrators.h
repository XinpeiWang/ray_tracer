#pragma once
// ---------------------------------------------------------------------------
// utility_integrators.h -- RandomWalkIntegrator + AOIntegrator ports
//
// Ported from pbrt-v4 src/pbrt/cpu/integrators.h/.cpp (Apache-2.0)
// Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
//
// Provides:
//   RandomWalkLi<T, Scene>(ray_o, ray_d, scene, max_depth, rand2d, out_L[3])
//   AOLi<T, Scene>(ray_o, ray_d, scene, max_dist, cos_sample,
//                  illum_scale, illum_rgb[3], rand2d, out_L[3])
//
// ---------------------------------------------------------------------------
// RandomWalkLi -- unbiased reference path tracer (no NEE, no MIS)
//
// Mirrors pbrt-v4 RandomWalkIntegrator::LiRandomWalk().
// At each vertex the next direction is sampled uniformly over the full
// sphere (pdf = 1/(4π)).  The path weight is accumulated as:
//   beta *= f(wo, wi) * |cos(wi, shading_n)| * 4π
// which exactly corresponds to pbrt-v4's
//   Le + fcos * Li_recursive / (1/(4*Pi))
// converted to an iterative loop.
//
// Scene concept:
//   bool  Intersect(const T org[3], const T dir[3], T t_max,
//                   BDPTHit<T>& hit) const
//   void  InfiniteLightLe(const T dir[3], T out[3]) const
//   void  SurfaceLe(const BDPTHit<T>& hit, const T wo[3], T out[3]) const
//   void  BSDFf(int bsdf_id, const T wo[3], const T wi[3],
//               const T n[3], T out[3]) const
//         // NOTE: out already includes |cos(wi,n)| (unlike pbrt-v4 bsdf.f)
//   void  SpawnRay(const BDPTHit<T>& hit, const T dir[3],
//                  T new_o[3], T new_d[3]) const
//   bool  BSDFIsNull(int bsdf_id) const
//         // true for medium boundaries (no BSDF): mirrors pbrt-v4 !bsdf
//
// ---------------------------------------------------------------------------
// AOLi -- ambient-occlusion integrator
//
// Mirrors pbrt-v4 AOIntegrator::Li().
// Casts a single shadow ray from the first surface hit.  Returns
//   illum_scale * illum_rgb * cos(wi, geo_n) / (pi * pdf)
// when the direction is unoccluded within max_dist, zero otherwise.
// cos_sample==true uses cosine-hemisphere sampling via pbrt-v4's
// SampleUniformDiskConcentric mapping (pdf = cos/π → result simplifies to
// illum_scale * illum_rgb regardless of wi direction).
// cos_sample==false uses uniform hemisphere, pbrt-v4 SampleUniformHemisphere
// (z=u[0], phi=2π·u[1], pdf = 1/(2π)).
//
// Scene concept (subset of RandomWalkLi):
//   bool  Intersect(const T org[3], const T dir[3], T t_max,
//                   BDPTHit<T>& hit) const
//   bool  BSDFIsNull(int bsdf_id) const
//   bool  UnoccludedWithin(const T p[3], const T dir[3], T max_dist) const
//         // true if no hit within (0, max_dist) along dir from p
//   void  SpawnRay(const BDPTHit<T>& hit, const T dir[3],
//                  T new_o[3], T new_d[3]) const
//
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#  if defined(__CUDACC__)
#    define CPU_GPU __host__ __device__ __forceinline__
#  else
#    define CPU_GPU inline
#  endif
#endif

#include "bdpt.h"       // BDPTHit<T>
#include "scalar_math.h"

#include <cmath>
#include <algorithm>

// ===========================================================================
// RandomWalkLi<T, Scene>
// ===========================================================================
template <typename T, typename Scene, typename Rand2D>
CPU_GPU void RandomWalkLi(
	const T      ray_o[3],
	const T      ray_d[3],
	const Scene& scene,
	int          max_depth,
	Rand2D       rand2d,
	T            out_L[3])
{
	static constexpr T kTMax   = T(1e30);
	static constexpr T k4Pi    = T(4.0 * 3.14159265358979323846);

	// Path throughput beta
	T beta[3] = { T(1), T(1), T(1) };

	T org[3] = { ray_o[0], ray_o[1], ray_o[2] };
	T dir[3] = { ray_d[0], ray_d[1], ray_d[2] };

	int depth = 0;

	while (true) {
		// --- Intersect ---
		BDPTHit<T> hit{};
		bool found = scene.Intersect(org, dir, kTMax, hit);

		if (!found) {
			// Miss: accumulate infinite-light emission and stop
			// mirrors: for (light : infiniteLights) Le += light.Le(ray, lambda)
			T env[3] = {};
			scene.InfiniteLightLe(dir, env);
			out_L[0] += beta[0] * env[0];
			out_L[1] += beta[1] * env[1];
			out_L[2] += beta[2] * env[2];
			break;
		}

		// --- Surface emission (Le at hit) ---
		// mirrors: SampledSpectrum Le = isect.Le(wo, lambda)
		T wo[3] = { -dir[0], -dir[1], -dir[2] };
		T Le[3] = {};
		scene.SurfaceLe(hit, wo, Le);
		out_L[0] += beta[0] * Le[0];
		out_L[1] += beta[1] * Le[1];
		out_L[2] += beta[2] * Le[2];

		// --- Terminate at max depth ---
		// mirrors: if (depth == maxDepth) return Le
		if (depth++ == max_depth)
			break;

		// --- Skip null/medium-boundary BSDFs ---
		// mirrors: BSDF bsdf = ...; if (!bsdf) return Le
		if (scene.BSDFIsNull(hit.bsdf_id))
			break;

		// --- Sample uniform sphere ---
		// mirrors: Vector3f wp = SampleUniformSphere(u)
		//          pdf = UniformSpherePDF() = 1/(4*pi)
		auto [u1, u2] = rand2d();
		T cos_theta = T(1) - T(2) * u1;   // uniform in [-1, 1]
		T sin_theta = std::sqrt(std::max(T(0), T(1) - cos_theta * cos_theta));
		T phi       = T(2) * T(3.14159265358979323846) * u2;
		T wp[3] = {
			sin_theta * std::cos(phi),
			sin_theta * std::sin(phi),
			cos_theta
		};
		// wp is already in world space (uniform sphere has no orientation bias)

		// --- Evaluate BSDF * |cos| ---
		// mirrors: SampledSpectrum fcos = bsdf.f(wo, wp) * AbsDot(wp, isect.shading.n)
		// Our BSDFf already returns f * |cos(wi,n)|
		T fcos[3] = {};
		scene.BSDFf(hit.bsdf_id, wo, wp, hit.shading_n, fcos);

		// mirrors: if (!fcos) return Le
		if (fcos[0] == T(0) && fcos[1] == T(0) && fcos[2] == T(0))
			break;

		// --- Update throughput ---
		// mirrors: Le + fcos * LiRecursive / (1/(4*Pi))
		//        = Le + fcos * LiRecursive * 4*Pi
		// Iterative equivalent: beta *= fcos * 4*Pi  (pdf = 1/(4*Pi))
		beta[0] *= fcos[0] * k4Pi;
		beta[1] *= fcos[1] * k4Pi;
		beta[2] *= fcos[2] * k4Pi;

		// --- Advance ray ---
		T new_o[3], new_d[3];
		scene.SpawnRay(hit, wp, new_o, new_d);
		org[0] = new_o[0]; org[1] = new_o[1]; org[2] = new_o[2];
		dir[0] = new_d[0]; dir[1] = new_d[1]; dir[2] = new_d[2];
	}
}

// ===========================================================================
// AOLi<T, Scene>
// ===========================================================================
template <typename T, typename Scene, typename Rand2D>
CPU_GPU void AOLi(
	const T      ray_o[3],
	const T      ray_d[3],
	const Scene& scene,
	T            max_dist,
	bool         cos_sample,   // true: cosine-hemisphere; false: uniform
	T            illum_scale,
	const T      illum_rgb[3],
	Rand2D       rand2d,
	T            out_L[3])
{
	static constexpr T kTMax  = T(1e30);
	static constexpr T kPi    = T(3.14159265358979323846);

	out_L[0] = out_L[1] = out_L[2] = T(0);

	T org[3] = { ray_o[0], ray_o[1], ray_o[2] };
	T dir[3] = { ray_d[0], ray_d[1], ray_d[2] };

	// --- Skip medium-boundary BSDFs (mirrors pbrt-v4 goto retry loop) ---
	while (true) {
		BDPTHit<T> hit{};
		if (!scene.Intersect(org, dir, kTMax, hit))
			return;  // Miss -> L = 0

		if (!scene.BSDFIsNull(hit.bsdf_id)) {
			// Valid surface hit — proceed below
			// FaceForward geometric normal towards incoming ray
			// mirrors: n = FaceForward(isect.n, -ray.d)
			T wo[3]  = { -dir[0], -dir[1], -dir[2] };
			T gn[3]  = { hit.geo_n[0], hit.geo_n[1], hit.geo_n[2] };
			T dot_wo_n = wo[0]*gn[0] + wo[1]*gn[1] + wo[2]*gn[2];
			if (dot_wo_n < T(0)) {
				gn[0] = -gn[0]; gn[1] = -gn[1]; gn[2] = -gn[2];
			}

			// --- Sample hemisphere direction in world space ---
			auto [u1, u2] = rand2d();
			T wi_local[3], pdf;

			if (cos_sample) {
				// Mirrors pbrt-v4 SampleCosineHemisphere via SampleUniformDiskConcentric(u)
				// u = {u1, u2}, maps to concentric disk then projects up to hemisphere.
				// pdf = cos_theta / pi  (CosineHemispherePDF)
				T uo_x = T(2) * u1 - T(1);
				T uo_y = T(2) * u2 - T(1);
				T dx, dy;
				if (uo_x == T(0) && uo_y == T(0)) {
					dx = dy = T(0);
				} else if (std::fabs(uo_x) > std::fabs(uo_y)) {
					T theta = (kPi / T(4)) * (uo_y / uo_x);
					dx = uo_x * std::cos(theta);
					dy = uo_x * std::sin(theta);
				} else {
					T theta = kPi / T(2) - (kPi / T(4)) * (uo_x / uo_y);
					dx = uo_y * std::cos(theta);
					dy = uo_y * std::sin(theta);
				}
				T cos_t = std::sqrt(std::max(T(0), T(1) - dx*dx - dy*dy));
				wi_local[0] = dx;
				wi_local[1] = dy;
				wi_local[2] = cos_t;
				pdf = cos_t / kPi;   // CosineHemispherePDF(|wi.z|)
			} else {
				// Mirrors pbrt-v4 SampleUniformHemisphere(u):
				//   z = u[0]  (first),  phi = 2*pi*u[1]  (second)
				// pdf = 1/(2*pi)
				T cos_t = u1;
				T sin_t = std::sqrt(std::max(T(0), T(1) - cos_t * cos_t));
				T phi   = T(2) * kPi * u2;
				wi_local[0] = sin_t * std::cos(phi);
				wi_local[1] = sin_t * std::sin(phi);
				wi_local[2] = cos_t;
				pdf = T(1) / (T(2) * kPi);
			}

			if (pdf == T(0))
				return;

			// --- Transform from local (z=gn) to world ---
			// Frame::FromZ(n): build ONB with z = gn
			T tx, ty, tz;
			if (std::fabs(gn[0]) > T(0.9)) { tx = T(0); ty = T(1); tz = T(0); }
			else                            { tx = T(1); ty = T(0); tz = T(0); }
			T cx = ty*gn[2]-tz*gn[1], cy = tz*gn[0]-tx*gn[2], cz = tx*gn[1]-ty*gn[0];
			T cl = std::sqrt(cx*cx + cy*cy + cz*cz);
			tx = cx/cl; ty = cy/cl; tz = cz/cl;
			T bx = gn[1]*tz-gn[2]*ty, by = gn[2]*tx-gn[0]*tz, bz = gn[0]*ty-gn[1]*tx;

			T wi[3] = {
				wi_local[0]*tx + wi_local[1]*bx + wi_local[2]*gn[0],
				wi_local[0]*ty + wi_local[1]*by + wi_local[2]*gn[1],
				wi_local[0]*tz + wi_local[1]*bz + wi_local[2]*gn[2]
			};

			// --- Shadow ray within max_dist ---
			// mirrors: Ray r = isect.SpawnRay(wi); if (!IntersectP(r, maxDist))
			T shadow_o[3], shadow_d[3];
			scene.SpawnRay(hit, wi, shadow_o, shadow_d);

			if (scene.UnoccludedWithin(shadow_o, shadow_d, max_dist)) {
				// Dot(wi, n) / (Pi * pdf)
				// mirrors: illumScale * illuminant.Sample(lambda) * Dot(wi,n) / (Pi*pdf)
				T cos_wi_n = wi[0]*gn[0] + wi[1]*gn[1] + wi[2]*gn[2];
				T weight   = illum_scale * cos_wi_n / (kPi * pdf);
				out_L[0] = illum_rgb[0] * weight;
				out_L[1] = illum_rgb[1] * weight;
				out_L[2] = illum_rgb[2] * weight;
			}
			return;
		}

		// Null BSDF: skip intersection (medium boundary)
		T new_o[3], new_d[3];
		scene.SpawnRay(hit, dir, new_o, new_d);
		org[0] = new_o[0]; org[1] = new_o[1]; org[2] = new_o[2];
		dir[0] = new_d[0]; dir[1] = new_d[1]; dir[2] = new_d[2];
	}
}
