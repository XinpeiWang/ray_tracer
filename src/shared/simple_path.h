#pragma once
// ---------------------------------------------------------------------------
// simple_path.h -- SimplePathIntegrator port from pbrt-v4
//
// Ported from pbrt-v4 src/pbrt/cpu/integrators.cpp (Apache-2.0)
// Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
//
// Provides:
//   SimplePathLi<T, Scene>(ray_o, ray_d, scene, max_depth,
//                           sample_lights, sample_bsdf,
//                           rand2d, rand1d, out_L[3])
//
// This is pbrt-v4's canonical reference path tracer:
//   - Iterative loop (no recursion).
//   - Optional NEE (direct-light sampling) controlled by `sample_lights`.
//   - Optional BSDF sampling vs. uniform hemisphere via `sample_bsdf`.
//   - `specular_bounce` flag prevents double-counting of emission when NEE
//     is active (mirrors pbrt-v4 SimplePathIntegrator::Li exactly).
//   - RGB throughput vector `beta[3]`, no wavelength sampling.
//
// Design rules (same as bdpt.h / mlt.h / sppm.h / vol_path.h):
//   - Header-only, no virtual functions, no heap allocation.
//   - Template T: float or double.
//   - Template Scene: duck-typed scene concept (see below).
//
// Scene concept required API:
//   struct Scene {
//     // Intersect ray (org, dir) up to t_max; return false on miss.
//     // On hit fills BDPTHit<T>.
//     bool Intersect(const T org[3], const T dir[3], T t_max,
//                    BDPTHit<T>& hit) const;
//
//     // Shadow ray: return true if the segment p0->p1 is unoccluded.
//     bool Unoccluded(const T p0[3], const T p1[3]) const;
//
//     // Area-light emission at the intersection (zero if not a light).
//     // wo[3] is the outgoing direction (towards the camera).
//     void SurfaceLe(const BDPTHit<T>& hit, const T wo[3], T out[3]) const;
//
//     // Infinite-light background radiance for a missed ray.
//     void InfiniteLightLe(const T dir[3], T out[3]) const;
//
//     // Sample a light for direct illumination (NEE).
//     // Returns false if no lights. Fills BDPTLightSample<T>.
//     // IMPORTANT: ls.pdf must be the product of the light-selection
//     // probability and the per-light sampling PDF, i.e.
//     //   ls.pdf = p_select * p_sample
//     // This matches pbrt-v4's (sampledLight->p * ls->pdf) division.
//     bool SampleLight(T u, const T ref_p[3],
//                      BDPTLightSample<T>& ls) const;
//
//     // BSDF evaluation f(wo, wi) * |cos(wi, n)|.
//     // NOTE: Unlike pbrt-v4's bsdf.f() (which returns BSDF value only),
//     // this API pre-multiplies by |cos(wi, shading_n)| for convenience.
//     void BSDFf(int bsdf_id, const T wo[3], const T wi[3],
//                const T n[3], T out[3]) const;
//
//     // BSDF importance sampling; returns false if sampling fails.
//     bool BSDFSampleF(int bsdf_id, const T wo[3], const T n[3],
//                      T u1, T u2,
//                      T new_dir[3], T f_val[3], T& pdf,
//                      bool& is_specular) const;
//
//     // Whether the BSDF can scatter in both reflection and transmission
//     // (needed for the uniform-sphere vs hemisphere fallback).
//     bool BSDFIsReflectiveAndTransmissive(int bsdf_id) const;
//
//     // Spawn a new ray from a surface point (offset to avoid self-hit).
//     // Fills new_o[3] (origin), copies dir[3] into new_d[3].
//     void SpawnRay(const BDPTHit<T>& hit, const T dir[3],
//                   T new_o[3], T new_d[3]) const;
//   };
//
// Usage:
//   float L[3] = {0,0,0};
//   SimplePathLi<float>(org, dir, scene, /*max_depth*/8,
//                        /*sample_lights*/true, /*sample_bsdf*/true,
//                        rand2d, rand1d, L);
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#  if defined(__CUDACC__)
#    define CPU_GPU __host__ __device__ __forceinline__
#  else
#    define CPU_GPU inline
#  endif
#endif

#include "bdpt.h"       // BDPTHit<T>, BDPTLightSample<T>
#include "scalar_math.h"
#include "sampling.h"   // SampleCosineHemisphere, SampleUniformHemisphere, etc.

#include <cmath>
#include <algorithm>
#include <functional>

// ---------------------------------------------------------------------------
// SimplePathResult<T>: output of SimplePathLi -- RGB radiance estimate.
// ---------------------------------------------------------------------------
template <typename T>
struct SimplePathResult {
	T L[3] = {T(0), T(0), T(0)};
};

// ---------------------------------------------------------------------------
// SimplePathLi<T, Scene>
//
// Estimates incident radiance along the ray (ray_o, ray_d) using simple
// path tracing. Mirrors pbrt-v4 SimplePathIntegrator::Li().
//
// Parameters:
//   ray_o[3]      -- ray origin
//   ray_d[3]      -- ray direction (need not be normalised; scene may normalise)
//   scene         -- scene concept instance (see above)
//   max_depth     -- maximum path length (number of bounces)
//   sample_lights -- if true, perform NEE at each diffuse vertex
//   sample_bsdf   -- if true, use BSDF importance sampling for the next
//                    direction; if false, fall back to uniform hemisphere
//   rand2d        -- callable returning a pair of independent [0,1) values:
//                    std::pair<T,T> rand2d()
//   rand1d        -- callable returning a single [0,1) value: T rand1d()
//   out_L[3]      -- output: estimated radiance (added into, not reset)
// ---------------------------------------------------------------------------
template <typename T, typename Scene,
		  typename Rand2D, typename Rand1D>
CPU_GPU void SimplePathLi(
	const T         ray_o[3],
	const T         ray_d[3],
	const Scene&    scene,
	int             max_depth,
	bool            sample_lights,
	bool            sample_bsdf,
	Rand2D          rand2d,
	Rand1D          rand1d,
	T               out_L[3])
{
	static constexpr T kPi    = T(3.14159265358979323846);
	static constexpr T kInvPi = T(1.0 / 3.14159265358979323846);
	static constexpr T kTMax  = T(1e30);

	// beta: path throughput (RGB)
	T beta[3] = { T(1), T(1), T(1) };

	// Current ray
	T org[3] = { ray_o[0], ray_o[1], ray_o[2] };
	T dir[3] = { ray_d[0], ray_d[1], ray_d[2] };

	// specular_bounce: true at start and after a specular scatter.
	// When sample_lights is active, emission is only added on specular bounces
	// (mirrors pbrt-v4 SimplePathIntegrator: avoid double-counting from NEE).
	bool specular_bounce = true;
	int  depth           = 0;

	while (true) {
		// --- Terminate if throughput is zero ---
		if (beta[0] == T(0) && beta[1] == T(0) && beta[2] == T(0))
			break;

		// --- Intersect ray with scene ---
		BDPTHit<T> hit{};
		bool found = scene.Intersect(org, dir, kTMax, hit);

		if (!found) {
			// Account for infinite lights (background)
			if (!sample_lights || specular_bounce) {
				T env[3] = {};
				scene.InfiniteLightLe(dir, env);
				out_L[0] += beta[0] * env[0];
				out_L[1] += beta[1] * env[1];
				out_L[2] += beta[2] * env[2];
			}
			break;
		}

		// --- Account for surface emission ---
		// Only add if light was not explicitly sampled, or if this was a
		// specular bounce (cannot do NEE on the previous specular vertex).
		if (!sample_lights || specular_bounce) {
			T wo[3] = { -dir[0], -dir[1], -dir[2] };
			T Le[3] = {};
			scene.SurfaceLe(hit, wo, Le);
			out_L[0] += beta[0] * Le[0];
			out_L[1] += beta[1] * Le[1];
			out_L[2] += beta[2] * Le[2];
		}

		// --- Terminate at maximum depth ---
		if (depth++ == max_depth)
			break;

		// --- Skip medium boundaries (no BSDF) ---
		if (hit.is_medium_boundary) {
			// Advance ray past the boundary without changing direction
			specular_bounce = true;
			T new_o[3], new_d[3];
			scene.SpawnRay(hit, dir, new_o, new_d);
			org[0] = new_o[0]; org[1] = new_o[1]; org[2] = new_o[2];
			dir[0] = new_d[0]; dir[1] = new_d[1]; dir[2] = new_d[2];
			continue;
		}

		// --- NEE: sample direct illumination ---
		T wo[3] = { -dir[0], -dir[1], -dir[2] };

		if (sample_lights) {
			BDPTLightSample<T> ls{};
			T u_light = rand1d();
			if (scene.SampleLight(u_light, hit.p, ls) && ls.pdf > T(0)) {
				// Evaluate BSDF for the light direction
				T f_val[3] = {};
				scene.BSDFf(hit.bsdf_id, wo, ls.wi, hit.shading_n, f_val);
				bool nonzero = (f_val[0] > T(0) || f_val[1] > T(0) || f_val[2] > T(0));
				if (nonzero && scene.Unoccluded(hit.p, ls.p_light)) {
					T weight = T(1) / ls.pdf;
					out_L[0] += beta[0] * f_val[0] * ls.L[0] * weight;
					out_L[1] += beta[1] * f_val[1] * ls.L[1] * weight;
					out_L[2] += beta[2] * f_val[2] * ls.L[2] * weight;
				}
			}
		}

		// --- Sample outgoing direction for the next bounce ---
		if (sample_bsdf) {
			// BSDF importance sampling (mirrors pbrt-v4 bs = bsdf.Sample_f)
			auto [u1, u2] = rand2d();
			T    new_dir[3] = {};
			T    f_val[3]   = {};
			T    pdf        = T(0);
			bool is_specular = false;

			if (!scene.BSDFSampleF(hit.bsdf_id, wo, hit.shading_n,
									u1, u2, new_dir, f_val, pdf, is_specular))
				break;
			if (pdf == T(0))
				break;

			beta[0] *= f_val[0] / pdf;
			beta[1] *= f_val[1] / pdf;
			beta[2] *= f_val[2] / pdf;

			specular_bounce = is_specular;

			T new_o[3], new_d[3];
			scene.SpawnRay(hit, new_dir, new_o, new_d);
			org[0] = new_o[0]; org[1] = new_o[1]; org[2] = new_o[2];
			dir[0] = new_d[0]; dir[1] = new_d[1]; dir[2] = new_d[2];

		} else {
			// Uniform hemisphere / sphere fallback
			// (mirrors pbrt-v4: sample sphere if both refl+trans, else hemisphere)
			auto [u1, u2] = rand2d();
			T new_dir[3] = {};
			T pdf         = T(0);

			bool both_rt = scene.BSDFIsReflectiveAndTransmissive(hit.bsdf_id);
			if (both_rt) {
				// Uniform sphere
				T cos_theta = T(1) - T(2) * u2;
				T sin_theta = std::sqrt(std::max(T(0), T(1) - cos_theta * cos_theta));
				T phi       = T(2) * kPi * u1;
				T lx = sin_theta * std::cos(phi);
				T ly = sin_theta * std::sin(phi);
				T lz = cos_theta;
				pdf  = T(1) / (T(4) * kPi);

				// Transform local sphere sample to world (using geometric normal)
				const T* n = hit.shading_n;
				T tx, ty, tz;
				if (std::fabs(n[0]) > T(0.9)) { tx = T(0); ty = T(1); tz = T(0); }
				else                           { tx = T(1); ty = T(0); tz = T(0); }
				T cx = ty*n[2]-tz*n[1], cy = tz*n[0]-tx*n[2], cz = tx*n[1]-ty*n[0];
				T cl = std::sqrt(cx*cx+cy*cy+cz*cz);
				tx = cx/cl; ty = cy/cl; tz = cz/cl;
				T bx = n[1]*tz-n[2]*ty, by = n[2]*tx-n[0]*tz, bz = n[0]*ty-n[1]*tx;
				new_dir[0] = lx*tx + ly*bx + lz*n[0];
				new_dir[1] = lx*ty + ly*by + lz*n[1];
				new_dir[2] = lx*tz + ly*bz + lz*n[2];
			} else {
				// Uniform hemisphere sampling (mirrors pbrt-v4 SampleUniformHemisphere /
				// UniformHemispherePDF). pdf = 1/(2*pi) for all directions.
				// SampleUniformHemisphere: cos_theta = u[0] (maps [0,1) -> [0,1) upper hemi)
				T cos_t = u2;   // in [0,1) -> upper hemisphere relative to n
				T sin_t = std::sqrt(std::max(T(0), T(1) - cos_t * cos_t));
				T phi   = T(2) * kPi * u1;
				T lx = sin_t * std::cos(phi);
				T ly = sin_t * std::sin(phi);
				T lz = cos_t;           // z is "up" in local frame = shading n
				pdf  = T(1) / (T(2) * kPi); // UniformHemispherePDF
				// Build ONB around shading normal (same as above)
				const T* n = hit.shading_n;
				T tx, ty, tz;
				if (std::fabs(n[0]) > T(0.9)) { tx = T(0); ty = T(1); tz = T(0); }
				else                           { tx = T(1); ty = T(0); tz = T(0); }
				T cx = ty*n[2]-tz*n[1], cy = tz*n[0]-tx*n[2], cz = tx*n[1]-ty*n[0];
				T cl = std::sqrt(cx*cx+cy*cy+cz*cz);
				tx = cx/cl; ty = cy/cl; tz = cz/cl;
				T bx = n[1]*tz-n[2]*ty, by = n[2]*tx-n[0]*tz, bz = n[0]*ty-n[1]*tx;
				new_dir[0] = lx*tx + ly*bx + lz*n[0];
				new_dir[1] = lx*ty + ly*by + lz*n[1];
				new_dir[2] = lx*tz + ly*bz + lz*n[2];
				// Flip to correct hemisphere using geometric normal (mirrors pbrt-v4:
				//   if (IsReflective && Dot(wo,isect.n)*Dot(wi,isect.n) < 0) wi=-wi;
				//   if (IsTransmissive && Dot(wo,isect.n)*Dot(wi,isect.n) > 0) wi=-wi;
				// Since BSDFIsReflectiveAndTransmissive==false here, the surface is
				// either purely reflective or purely transmissive.)
				const T* gn = hit.geo_n;
				T cos_wo_gn = wo[0]*gn[0] + wo[1]*gn[1] + wo[2]*gn[2];
				T cos_wi_gn = new_dir[0]*gn[0] + new_dir[1]*gn[1] + new_dir[2]*gn[2];
				if (cos_wo_gn * cos_wi_gn < T(0)) {
					new_dir[0] = -new_dir[0];
					new_dir[1] = -new_dir[1];
					new_dir[2] = -new_dir[2];
				}
			}

			if (pdf == T(0))
				break;

			T f_val[3] = {};
			scene.BSDFf(hit.bsdf_id, wo, new_dir, hit.shading_n, f_val);

			beta[0] *= f_val[0] / pdf;
			beta[1] *= f_val[1] / pdf;
			beta[2] *= f_val[2] / pdf;

			specular_bounce = false;

			T new_o[3], new_d[3];
			scene.SpawnRay(hit, new_dir, new_o, new_d);
			org[0] = new_o[0]; org[1] = new_o[1]; org[2] = new_o[2];
			dir[0] = new_d[0]; dir[1] = new_d[1]; dir[2] = new_d[2];
		}
	}
}
