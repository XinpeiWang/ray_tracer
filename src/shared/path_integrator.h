#pragma once
// ---------------------------------------------------------------------------
// path_integrator.h -- PathIntegrator port from pbrt-v4
//
// Ported from pbrt-v4 src/pbrt/cpu/integrators.cpp (Apache-2.0)
// Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
//
// Provides:
//   PathLi<T, Scene>(ray_o, ray_d, scene, max_depth,
//                    rr_threshold, rand2d, rand1d, out_L[3])
//
// This is pbrt-v4's production path tracer, extending SimplePathIntegrator
// with full MIS, Russian roulette, and etaScale correction:
//
//   Key differences from simple_path.h:
//   1. SampleLd(): NEE uses PowerHeuristic(p_l, p_b) — combines light-sample
//      and BSDF PDFs to give unbiased, low-variance direct lighting.
//   2. Emission MIS: when a surface/infinite light is hit by a BSDF-sampled
//      ray, the contribution is weighted by PowerHeuristic(p_b, p_l) using
//      the previous intersection context (prevIntrCtx).
//   3. Russian roulette with etaScale correction: RR termination probability
//      is based on beta*etaScale so transmission chains are not over-terminated.
//   4. anyNonSpecularBounces: NEE (SampleLd) is skipped on purely specular
//      paths (mirrors pbrt-v4 IsNonSpecular check).
//   5. specularBounce starts false (pbrt-v4 PathIntegrator) — emission at
//      depth 0 or after a specular bounce is always added unweighted.
//
// Design rules (same as simple_path.h / bdpt.h / mlt.h / sppm.h / vol_path.h):
//   - Header-only, no virtual functions, no heap allocation.
//   - Template T: float or double.
//   - Template Scene: duck-typed scene concept (see below).
//
// Scene concept required API (superset of simple_path.h):
//   struct Scene {
//     // Intersect ray (org, dir) up to t_max; return false on miss.
//     bool Intersect(const T org[3], const T dir[3], T t_max,
//                    BDPTHit<T>& hit) const;
//
//     // Shadow ray: true if p0->p1 is unoccluded.
//     bool Unoccluded(const T p0[3], const T p1[3]) const;
//
//     // Surface emission at hit towards wo (zero for non-emissive surfaces).
//     void SurfaceLe(const BDPTHit<T>& hit, const T wo[3], T out[3]) const;
//
//     // Infinite-light background radiance for a missed ray.
//     void InfiniteLightLe(const T dir[3], T out[3]) const;
//
//     // --- New vs simple_path.h ---
//
//     // PDF that the light sampler + light would generate ls.wi from ref_p.
//     // Used for emission MIS. Return 0 if no lights or not applicable.
//     // Corresponds to lightSampler.PMF(prevIntrCtx, light) * light.PDF_Li(...)
//     T InfiniteLightPdf(const T ref_p[3], const T ref_n[3],
//                        const T dir[3]) const;
//
//     // Solid-angle PDF that the light sampler would assign to direction wi
//     // when sampling the area light that was hit at 'emissive_hit', from the
//     // previous vertex at (prev_p, prev_n).
//     // Corresponds to: lightSampler.PMF(prevIntrCtx, areaLight)
//     //                 * areaLight.PDF_Li(prevIntrCtx, ray.d, true)
//     // Return 0 if no area lights or the hit surface is not a light.
//     T AreaLightPdf(const BDPTHit<T>& emissive_hit,
//                    const T prev_p[3], const T prev_n[3],
//                    const T wi[3]) const;
//
//     // Sample a light for direct illumination (NEE).
//     // ls.pdf must be the combined selection * sample PDF.
//     // ls.is_delta == true for delta lights (point/directional).
//     bool SampleLight(T u, const T ref_p[3], BDPTLightSample<T>& ls) const;
//
//     // BSDF evaluation f(wo, wi) * |cos(wi, shading_n)|.
//     // NOTE: pre-multiplied by |cos| unlike pbrt-v4 bsdf.f().
//     void BSDFf(int bsdf_id, const T wo[3], const T wi[3],
//                const T n[3], T out[3]) const;
//
//     // BSDF PDF for direction wi given wo (no cos factor).
//     // Used for MIS weight in SampleLd and emission MIS.
//     T BSDFPdf(int bsdf_id, const T wo[3], const T wi[3],
//               const T n[3]) const;
//
//     // BSDF importance sampling; returns false if sampling fails.
//     // f_val already includes |cos(wi, shading_n)|.
//     // pdf is the solid-angle PDF.
//     // is_specular:         mirrors pbrt-v4 BSDFSample::IsSpecular()
//     // is_transmission:     mirrors pbrt-v4 BSDFSample::IsTransmission()
//     //                      Used for etaScale: set true whenever the scatter
//     //                      crosses the surface (refraction). Independent of
//     //                      is_specular (diffuse transmission is also true).
//     // pdf_is_proportional: if true, caller queries BSDFPdf() for true PDF.
//     // eta:                 relative IOR at the boundary (1 if no refraction).
//     bool BSDFSampleF(int bsdf_id, const T wo[3], const T n[3],
//                      T u1, T u2,
//                      T new_dir[3], T f_val[3], T& pdf,
//                      bool& is_specular, bool& is_transmission,
//                      bool& pdf_is_proportional,
//                      T& eta) const;
//
//     // Regularize the BSDF at bsdf_id by bumping low roughness alphas.
//     // Mirrors pbrt-v4 BSDF::Regularize() called in PathIntegrator::Li().
//     // Only called when regularize=true and any_non_specular_bounce is set.
//     void BSDFRegularize(int bsdf_id);
//
//     // True if the BSDF can scatter in both reflection and transmission.
//     bool BSDFIsReflectiveAndTransmissive(int bsdf_id) const;
//
//     // True if the BSDF has any non-specular (diffuse/glossy) component.
//     // Mirrors pbrt-v4 IsNonSpecular(bsdf.Flags()).
//     bool BSDFIsNonSpecular(int bsdf_id) const;
//
//     // Spawn a new ray from a surface point (offset to avoid self-hit).
//     void SpawnRay(const BDPTHit<T>& hit, const T dir[3],
//                   T new_o[3], T new_d[3]) const;
//   };
//
// Usage:
//   float L[3] = {0,0,0};
//   PathLi<float>(org, dir, scene, /*max_depth*/8,
//                 /*rr_threshold*/1.f,
//                 rand2d, rand1d, L,
//                 /*regularize*/false);
// ---------------------------------------------------------------------------

#include "cpu_gpu.h"

#include "bdpt.h"        // BDPTHit<T>, BDPTLightSample<T>
#include "scalar_math.h"

#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// PowerHeuristic (local, mirrors pbrt-v4 util/sampling.h)
// w_f = (nf * fPdf)^2 / ((nf*fPdf)^2 + (ng*gPdf)^2)
// For n=1 on both sides: w = fPdf^2 / (fPdf^2 + gPdf^2)
// ---------------------------------------------------------------------------
template <typename T>
CPU_GPU T PathPowerHeuristic(T fPdf, T gPdf) {
	T f2 = fPdf * fPdf;
	T g2 = gPdf * gPdf;
	T denom = f2 + g2;
	return (denom > T(0)) ? f2 / denom : T(0);
}

// ---------------------------------------------------------------------------
// PathSampleLd<T, Scene>
//
// Evaluates direct illumination at a surface hit using one light sample,
// weighted by the MIS power heuristic for non-delta lights.
// Mirrors pbrt-v4 PathIntegrator::SampleLd().
//
// Parameters:
//   hit        -- current surface intersection
//   wo[3]      -- outgoing direction (towards previous vertex / camera)
//   scene      -- scene concept instance
//   rand1d()   -- single [0,1) sample for light selection
//   rand2d()   -- pair  [0,1) x [0,1) sample for light point sampling
//   out_Ld[3]  -- output: direct radiance contribution (zeroed before use)
// ---------------------------------------------------------------------------
template <typename T, typename Scene, typename Rand2D, typename Rand1D>
CPU_GPU void PathSampleLd(
	const BDPTHit<T>&  hit,
	const T            wo[3],
	const Scene&       scene,
	Rand2D             rand2d,
	Rand1D             rand1d,
	T                  out_Ld[3])
{
	out_Ld[0] = out_Ld[1] = out_Ld[2] = T(0);

	// Choose a light and a point on it
	BDPTLightSample<T> ls{};
	T u_light = rand1d();
	if (!scene.SampleLight(u_light, hit.p, ls) || ls.pdf <= T(0))
		return;

	// Evaluate BSDF * |cos| for the light direction
	T f_val[3] = {};
	scene.BSDFf(hit.bsdf_id, wo, ls.wi, hit.shading_n, f_val);
	bool nonzero = (f_val[0] > T(0) || f_val[1] > T(0) || f_val[2] > T(0));
	if (!nonzero)
		return;

	// Visibility
	if (!scene.Unoccluded(hit.p, ls.p_light))
		return;

	// MIS weight: delta lights need no weighting (only one sampling strategy)
	T w_l;
	if (ls.is_delta) {
		w_l = T(1);
	} else {
		// PowerHeuristic(p_l=ls.pdf, p_b=BSDFPdf)
		T p_b = scene.BSDFPdf(hit.bsdf_id, wo, ls.wi, hit.shading_n);
		w_l = PathPowerHeuristic(ls.pdf, p_b);
	}

	T contrib = w_l / ls.pdf;
	out_Ld[0] = f_val[0] * ls.L[0] * contrib;
	out_Ld[1] = f_val[1] * ls.L[1] * contrib;
	out_Ld[2] = f_val[2] * ls.L[2] * contrib;
}

// ---------------------------------------------------------------------------
// PathLi<T, Scene>
//
// Estimates incident radiance along (ray_o, ray_d) using production path
// tracing with full MIS, Russian roulette, and etaScale.
// Mirrors pbrt-v4 PathIntegrator::Li().
//
// Parameters:
//   ray_o[3]      -- ray origin
//   ray_d[3]      -- ray direction
//   scene         -- scene concept instance (see above)
//   max_depth     -- maximum path length
//   rr_threshold  -- Russian-roulette start threshold (pbrt-v4 default: 1.0)
//                    Path is terminated when beta*etaScale max-component
//                    drops below this value AND depth > 1.
//   rand2d        -- callable: std::pair<T,T> rand2d()
//   rand1d        -- callable: T rand1d()
//   out_L[3]      -- output: estimated radiance (added into, not reset)
// ---------------------------------------------------------------------------
template <typename T, typename Scene,
		  typename Rand2D, typename Rand1D>
CPU_GPU void PathLi(
	const T         ray_o[3],
	const T         ray_d[3],
	const Scene&    scene,
	int             max_depth,
	T               rr_threshold,
	Rand2D          rand2d,
	Rand1D          rand1d,
	T               out_L[3],
	bool            regularize = false)
{
	static constexpr T kTMax = T(1e30);

	// Path throughput (RGB) and etaScale for RR correction
	T beta[3]  = { T(1), T(1), T(1) };
	T eta_scale = T(1);

	// Current ray
	T org[3] = { ray_o[0], ray_o[1], ray_o[2] };
	T dir[3] = { ray_d[0], ray_d[1], ray_d[2] };

	// specularBounce starts false (unlike SimplePathIntegrator which starts true).
	// Emission at depth==0 is always added; after that, MIS weights are used
	// unless the previous scatter was specular.
	bool specular_bounce         = false;
	bool any_non_specular_bounce = false;
	int  depth                   = 0;

	// p_b: solid-angle PDF with which the last direction was sampled (for MIS)
	T p_b = T(1);

	// Previous intersection context for emission MIS
	// We store p (position) and n (geometric normal) of the previous vertex.
	T prev_p[3] = { ray_o[0], ray_o[1], ray_o[2] };
	T prev_n[3] = { T(0), T(0), T(0) };  // camera — no normal needed

	while (true) {
		// --- Terminate if beta is zero ---
		if (beta[0] == T(0) && beta[1] == T(0) && beta[2] == T(0))
			break;

		// --- Intersect ray ---
		BDPTHit<T> hit{};
		bool found = scene.Intersect(org, dir, kTMax, hit);

		// --- Miss: accumulate infinite-light emission ---
		if (!found) {
			T env[3] = {};
			scene.InfiniteLightLe(dir, env);
			bool any_env = (env[0] > T(0) || env[1] > T(0) || env[2] > T(0));
			if (any_env) {
				if (depth == 0 || specular_bounce) {
					// Direct view of background or specular bounce: add unweighted
					out_L[0] += beta[0] * env[0];
					out_L[1] += beta[1] * env[1];
					out_L[2] += beta[2] * env[2];
				} else {
					// MIS weight: PowerHeuristic(p_b, p_l)
					// p_l = scene.InfiniteLightPdf(prev_p, prev_n, dir)
					T p_l = scene.InfiniteLightPdf(prev_p, prev_n, dir);
					T w_b = PathPowerHeuristic(p_b, p_l);
					out_L[0] += beta[0] * w_b * env[0];
					out_L[1] += beta[1] * w_b * env[1];
					out_L[2] += beta[2] * w_b * env[2];
				}
			}
			break;
		}

		// --- Hit: surface emission ---
		{
			T wo[3] = { -dir[0], -dir[1], -dir[2] };
			T Le[3] = {};
			scene.SurfaceLe(hit, wo, Le);
			bool any_Le = (Le[0] > T(0) || Le[1] > T(0) || Le[2] > T(0));
			if (any_Le) {
				if (depth == 0 || specular_bounce) {
					out_L[0] += beta[0] * Le[0];
					out_L[1] += beta[1] * Le[1];
					out_L[2] += beta[2] * Le[2];
				} else {
					// Emission MIS weight: PowerHeuristic(p_b, p_l)
					// p_l = lightSampler.PMF(prevCtx, areaLight) * areaLight.PDF_Li(...)
					T p_l = scene.AreaLightPdf(hit, prev_p, prev_n, dir);
					T w_b = PathPowerHeuristic(p_b, p_l);
					out_L[0] += beta[0] * w_b * Le[0];
					out_L[1] += beta[1] * w_b * Le[1];
					out_L[2] += beta[2] * w_b * Le[2];
				}
			}
		}

		// --- Skip medium boundaries (mirrors pbrt-v4 SkipIntersection / !bsdf guard) ---
		// Must come before regularize so we don't regularize a non-surface vertex.
		if (hit.is_medium_boundary) {
			specular_bounce = true;
			T new_o[3], new_d[3];
			scene.SpawnRay(hit, dir, new_o, new_d);
			org[0] = new_o[0]; org[1] = new_o[1]; org[2] = new_o[2];
			dir[0] = new_d[0]; dir[1] = new_d[1]; dir[2] = new_d[2];
			continue;
		}

		T wo[3] = { -dir[0], -dir[1], -dir[2] };

		// --- Path regularization (pbrt-v4 PathIntegrator::Li lines 711-714) ---
		// Mirrors pbrt-v4 exactly: regularize fires BEFORE the max-depth break,
		// so the roughened BSDF is visible to the upcoming NEE / BSDF sample at
		// the maximum-depth vertex (where only NEE contributes).
		if (regularize && any_non_specular_bounce)
			scene.BSDFRegularize(hit.bsdf_id);

		// --- Terminate at max depth ---
		if (depth++ == max_depth)
			break;

		// --- Direct illumination (NEE) via MIS SampleLd ---
		// Only on non-specular (diffuse/glossy) BSDFs (mirrors pbrt-v4 IsNonSpecular check)
		if (scene.BSDFIsNonSpecular(hit.bsdf_id)) {
			T Ld[3] = {};
			PathSampleLd<T>(hit, wo, scene, rand2d, rand1d, Ld);
			out_L[0] += beta[0] * Ld[0];
			out_L[1] += beta[1] * Ld[1];
			out_L[2] += beta[2] * Ld[2];
		}

		// --- Sample BSDF for next path direction ---
		{
			auto [u1, u2] = rand2d();
			T    new_dir[3]  = {};
			T    f_val[3]    = {};
			T    pdf         = T(0);
			bool is_specular      = false;
			bool is_transmission  = false;  // mirrors pbrt-v4 BSDFSample::IsTransmission()
			bool pdf_is_prop      = false;  // pdfIsProportional
			T    eta              = T(1);

			if (!scene.BSDFSampleF(hit.bsdf_id, wo, hit.shading_n,
								   u1, u2, new_dir, f_val, pdf,
								   is_specular, is_transmission, pdf_is_prop, eta))
				break;
			if (pdf == T(0))
				break;

			// Update throughput
			beta[0] *= f_val[0] / pdf;
			beta[1] *= f_val[1] / pdf;
			beta[2] *= f_val[2] / pdf;

			// True PDF for MIS (when pdf_is_proportional, query BSDFPdf)
			p_b = pdf_is_prop
				? scene.BSDFPdf(hit.bsdf_id, wo, new_dir, hit.shading_n)
				: pdf;

			specular_bounce          = is_specular;
			any_non_specular_bounce |= !is_specular;

			// etaScale: correct RR for transmission chains.
			// Mirrors pbrt-v4: if (bs->IsTransmission()) etaScale *= Sqr(bs->eta).
			// Applied whenever the scatter crosses the surface (is_transmission),
			// regardless of whether the BSDF is specular or diffuse.
			if (is_transmission)
				eta_scale *= eta * eta;

			// Record current hit as prevIntrCtx for next-bounce MIS
			prev_p[0] = hit.p[0]; prev_p[1] = hit.p[1]; prev_p[2] = hit.p[2];
			prev_n[0] = hit.geo_n[0]; prev_n[1] = hit.geo_n[1]; prev_n[2] = hit.geo_n[2];

			T new_o[3], new_d[3];
			scene.SpawnRay(hit, new_dir, new_o, new_d);
			org[0] = new_o[0]; org[1] = new_o[1]; org[2] = new_o[2];
			dir[0] = new_d[0]; dir[1] = new_d[1]; dir[2] = new_d[2];
		}

		// --- Russian roulette (mirrors pbrt-v4) ---
		// rrBeta = beta * etaScale
		T rr_beta_max = std::max({ beta[0]*eta_scale,
								   beta[1]*eta_scale,
								   beta[2]*eta_scale });
		if (rr_beta_max < rr_threshold && depth > 1) {
			T q = std::max(T(0), T(1) - rr_beta_max);
			if (rand1d() < q)
				break;
			beta[0] /= (T(1) - q);
			beta[1] /= (T(1) - q);
			beta[2] /= (T(1) - q);
		}
	}
}
