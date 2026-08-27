#pragma once
// ---------------------------------------------------------------------------
// simple_vol_path.h -- SimpleVolPathIntegrator port from pbrt-v4
//
// Ported from pbrt-v4 src/pbrt/cpu/integrators.cpp (Apache-2.0)
// Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
//
// Provides:
//   SimpleVolPathLi<T, Scene>(ray_o, ray_d, scene, max_depth, out_L[3])
//
// This is pbrt-v4's simplest volumetric path tracer:
//   - Pure delta tracking through heterogeneous media.
//   - No NEE, no MIS, no BSSRDF, no path regularization.
//   - Single scalar beta (throughput weight), RGB output via SurfaceLe /
//     InfiniteLightLe.
//   - On absorption: add beta * Le, terminate.
//   - On scatter: sample phase function, continue.
//   - On null event: resample uMode, continue delta-tracking march.
//   - On surface hit: add beta * area_Le, terminate (surface BSDFs not
//     supported -- mirrors pbrt-v4 SimpleVolPathIntegrator behaviour).
//   - On miss: add beta * InfiniteLightLe, return.
//
// pbrt-v4 alignment
// =================
//   Mirrors SimpleVolPathIntegrator::Li (cpu/integrators.cpp).
//   Key differences from the full VolPathLi:
//     * No spectral r_u / r_l MIS tracking (scalar beta only).
//     * No VolPathSampleLd (direct lighting / shadow rays).
//     * lambda.TerminateSecondary() replaced by RGB throughput convention.
//     * SampleT_maj callback uses the same three-way event dispatch
//       (absorb / scatter / null) as pbrt-v4.
//
// Design rules (same as bdpt.h / mlt.h / sppm.h / vol_path.h):
//   - Header-only, no virtual functions, no heap allocation.
//   - Template T: float or double.
//   - Template Scene: duck-typed scene concept (see below).
//
// Scene concept required API:
//   // Geometry
//   bool  Intersect(const T org[3], const T dir[3], T t_max,
//                   BDPTHit<T>& hit) const
//
//   // Light (surface and infinite only; no direct sampling needed)
//   void  InfiniteLightLe(const T dir[3], T out[3]) const
//
//   // Medium
//   bool  HasMedium(const T org[3], const T dir[3]) const
//   //   Walks the majorant segments along (org, dir) up to t_max.
//   //   For each sub-interval the callback receives:
//   //     p[3]      -- sampled interaction point
//   //     mp        -- VolPathMediumProps<T> at p
//   //     sigma_maj -- scalar majorant extinction at p
//   //     T_maj     -- transmittance to p (free-flight weight, used by VolPath;
//   //                  SimpleVolPath ignores it -- delta tracking only needs
//   //                  the event probabilities)
//   //   Return true to continue march, false to stop.
//   //   Returns the majorant transmittance to the first un-continued point.
//   T     SampleTMaj(const T org[3], const T dir[3], T t_max, T u_maj,
//                    const std::function<bool(
//                        const T p[3],
//                        const VolPathMediumProps<T>& mp,
//                        T sigma_maj, T T_maj)>& callback) const
//
//   // Phase function (Henyey-Greenstein)
//   bool  SamplePhase(const T wo[3], T g, T u1, T u2,
//                     T wi[3], T& pdf) const
//
//   // Random numbers (called const -- RNG state must be mutable)
//   T     RandFloat() const
//
//   // Spawn an offset ray from a surface hit to avoid self-intersection.
//   // Mirrors pbrt-v4 SkipIntersection -> SpawnRay(ray.d) -> OffsetRayOrigin.
//   // Used for medium-boundary surfaces that should be passed through.
//   void  SpawnRay(const BDPTHit<T>& hit, const T dir[3],
//                  T new_o[3], T new_d[3]) const
//
// Optional (zero-valued defaults acceptable):
//   // Infinite-area light background (already in the concept above)
//
// Usage:
//   float L[3] = {0, 0, 0};
//   SimpleVolPathLi<float>(org, dir, scene, /*max_depth=*/8, L);
// ---------------------------------------------------------------------------

#include "cpu_gpu.h"

#include "bdpt.h"     // BDPTHit<T>
#include "vol_path.h" // VolPathMediumProps<T>

#include <cmath>
#include <functional>
#include <algorithm>
#include <limits>

// ---------------------------------------------------------------------------
// SimpleVolPathLi<T, Scene>
//
// Direct port of pbrt-v4 SimpleVolPathIntegrator::Li.
//
// ray_o[3]   -- ray origin
// ray_d[3]   -- ray direction (need not be normalised, but should be unit for
//               physically correct results)
// scene      -- duck-typed scene satisfying the concept above
// max_depth  -- maximum number of medium scattering events before termination
// out_L[3]   -- output RGB radiance (accumulated, caller should zero first)
// ---------------------------------------------------------------------------
template <typename T, typename Scene>
CPU_GPU void SimpleVolPathLi(
		const T ray_o[3], const T ray_d[3],
		const Scene& scene,
		int max_depth,
		T out_L[3])
{
	// Current ray state
	T org[3] = { ray_o[0], ray_o[1], ray_o[2] };
	T dir[3] = { ray_d[0], ray_d[1], ray_d[2] };

	// Scalar path throughput (mirrors pbrt-v4 beta: Float)
	T beta = T(1);

	// Scattering depth counter
	int depth = 0;
	// A medium-boundary surface pass-through doesn't advance `depth` below,
	// so it needs its own bounded safety cap. kMaxMediumBoundaryCrossings
	// (cpu_gpu.h) is the one shared bound every integrator that supports
	// this uses (a degenerate/self-intersecting scene shouldn't be able to
	// hang this loop).
	int medium_boundary_crossings = 0;

	constexpr T kInfinity = std::numeric_limits<T>::max();

	while (true) {
		// ------------------------------------------------------------------
		// 1. Find nearest surface intersection
		// ------------------------------------------------------------------
		BDPTHit<T> hit;
		bool hit_surface = scene.Intersect(org, dir, kInfinity, hit);
		T t_max = hit_surface ? hit.t_hit : kInfinity;

		// ------------------------------------------------------------------
		// 2. Delta-track through medium (if any)
		// ------------------------------------------------------------------
		bool scattered   = false;
		bool terminated  = false;

		if (scene.HasMedium(org, dir)) {
			T u_maj  = scene.RandFloat();
			T u_mode = scene.RandFloat();

			scene.SampleTMaj(org, dir, t_max, u_maj,
				[&](const T p[3],
					const VolPathMediumProps<T>& mp,
					T sigma_maj, T /*T_maj*/) -> bool
				{
					if (sigma_maj <= T(0)) {
						// Degenerate majorant: skip
						u_mode = scene.RandFloat();
						return true;
					}

					// Event probabilities (pbrt-v4 uses channel 0 of sigma_maj
					// for the scalar SimpleVolPath)
					T p_absorb  = mp.sigma_a / sigma_maj;
					T p_scatter = mp.sigma_s / sigma_maj;
					T p_null    = std::max(T(0), T(1) - p_absorb - p_scatter);

					// Three-way discrete sample (SampleDiscrete equivalent)
					int mode;
					if (u_mode < p_absorb) {
						mode = 0; // absorption
					} else if (u_mode < p_absorb + p_scatter) {
						mode = 1; // scatter
					} else {
						mode = 2; // null
					}

					if (mode == 0) {
						// Absorption: add volume emission, terminate path
						// mp.Le is scalar luminance -- broadcast to RGB
						out_L[0] += beta * mp.Le;
						out_L[1] += beta * mp.Le;
						out_L[2] += beta * mp.Le;
						terminated = true;
						return false; // stop march

					} else if (mode == 1) {
						// Scatter: check depth, sample phase function
						if (depth++ >= max_depth) {
							terminated = true;
							return false;
						}

						T u1 = scene.RandFloat(), u2 = scene.RandFloat();
						T wi[3] = {};
						T phase_pdf = T(0);
						// wo = -dir (pointing back along ray)
						T wo[3] = { -dir[0], -dir[1], -dir[2] };
						if (!scene.SamplePhase(wo, mp.g, u1, u2, wi, phase_pdf)
							|| phase_pdf <= T(0))
						{
							terminated = true;
							return false;
						}

						// beta *= p / pdf  (for HG: p == pdf, so beta unchanged)
						// Still correct to apply for generality.
						T phase_val = scene.PhaseP(wo, wi, mp.g);
						beta *= phase_val / phase_pdf;

						// Advance ray to scatter point
						org[0] = p[0]; org[1] = p[1]; org[2] = p[2];
						dir[0] = wi[0]; dir[1] = wi[1]; dir[2] = wi[2];
						scattered = true;
						return false; // stop march, re-enter outer loop

					} else {
						// Null event: resample uMode and continue march
						u_mode = scene.RandFloat();
						return true;
					}
				});
		}

		// ------------------------------------------------------------------
		// 3. Handle termination and scatter results
		// ------------------------------------------------------------------
		if (terminated) return;
		if (scattered)  continue;

		// ------------------------------------------------------------------
		// 4. No medium event fired -- handle surface or miss
		// ------------------------------------------------------------------
		if (!hit_surface) {
			// Miss: add infinite-light background
			T Le[3] = {};
			scene.InfiniteLightLe(dir, Le);
			out_L[0] += beta * Le[0];
			out_L[1] += beta * Le[1];
			out_L[2] += beta * Le[2];
			return;
		}

		// Surface hit: add area emission (if any), then terminate.
		// pbrt-v4 SimpleVolPathIntegrator only handles medium scattering;
		// surface BSDFs are not supported (it asserts on a valid BSDF sample).
		// Medium-boundary surfaces (is_medium_boundary) are passed through.
		if (hit.is_medium_boundary) {
			// Skip medium-boundary primitives -- mirrors pbrt-v4
			// SkipIntersection -> SpawnRay(ray.d) -> OffsetRayOrigin.
			if (++medium_boundary_crossings > kMaxMediumBoundaryCrossings) return;
			scene.SpawnRay(hit, dir, org, dir);
			continue;
		}

		// Add surface emission and terminate
		T wo[3] = { -dir[0], -dir[1], -dir[2] };
		(void)wo; // area_Le is pre-stored in BDPTHit
		out_L[0] += beta * hit.area_Le[0];
		out_L[1] += beta * hit.area_Le[1];
		out_L[2] += beta * hit.area_Le[2];
		return;
	}
}
