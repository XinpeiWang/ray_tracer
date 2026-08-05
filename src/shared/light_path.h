#pragma once
// ---------------------------------------------------------------------------
// light_path.h -- LightPathIntegrator port from pbrt-v4
//
// Ported from pbrt-v4 src/pbrt/cpu/integrators.cpp (Apache-2.0)
// Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
//
// Provides:
//   LightPathTrace<T, Scene, Film>(scene, film, max_depth, rand1d, rand2d)
//
// Algorithm (mirrors pbrt-v4 LightPathIntegrator::EvaluatePixelSample):
//   1. Sample a light source to obtain an emission ray and initial throughput:
//        beta = Le * |cos(theta)| / (p_light * pdfPos * pdfDir)
//   2. If the sampled light has a surface (area light), try a direct
//      light-surface -> camera connection and splat.
//   3. Walk the path forward under importance transport:
//        a. Intersect -> break on miss.
//        b. Skip null/medium-boundary BSDFs (continue ray through surface).
//        c. Terminate if max_depth reached.
//        d. Connect to camera: splat  beta * BSDFf_imp * CameraWi / pdf.
//        e. Sample BSDF (importance mode) to get the next direction.
//        f. Update beta *= BSDFf_imp / pdf; spawn next ray.
//
// ---------------------------------------------------------------------------
// Design rules (identical to bdpt.h / mlt.h / simple_path.h):
//   - Header-only, no virtual functions, no heap allocation.
//   - Template T : float or double.
//   - Template Scene : duck-typed scene concept (see below).
//   - Template Film  : duck-typed film concept (see below).
//
// ---------------------------------------------------------------------------
// Scene concept required API:
//
//   // Ray intersection.  Returns false on miss.  Fills BDPTHit<T>.
//   bool Intersect(const T org[3], const T dir[3], T t_max,
//                  BDPTHit<T>& hit) const;
//
//   // Shadow ray: true if segment from p0 to p1 is unoccluded.
//   bool Unoccluded(const T p0[3], const T p1[3]) const;
//
//   // Spawn a new ray from a surface hit (offset to avoid self-intersection).
//   void SpawnRay(const BDPTHit<T>& hit, const T dir[3],
//                 T new_o[3], T new_d[3]) const;
//
//   // True if this BSDF is a medium boundary (no scattering) -- mirrors
//   // pbrt-v4 "if (!bsdf) { isect.SkipIntersection(...); continue; }".
//   bool BSDFIsNull(int bsdf_id) const;
//
//   // BSDF evaluation under importance transport (radiance/importance reversal):
//   //   f(wo, wi) * |cos(wi, shading_n)|
//   // (pbrt-v4: bsdf.f(isect.wo, cs.wi, TransportMode::Importance)
//   //            * AbsDot(cs.wi, isect.shading.n))
//   // NOTE: out already includes |cos(wi,n)| -- same convention as BSDFf in
//   //       simple_path.h / utility_integrators.h.
//   void BSDFfImportance(int bsdf_id, const T wo[3], const T wi[3],
//                        const T n[3], T out[3]) const;
//
//   // BSDF importance sampling under importance transport mode.
//   // Returns false if sampling fails (pdf == 0 or no valid direction).
//   // f_val[3] must include |cos(wi,n)| (same convention as BSDFfImportance).
//   bool BSDFSampleFImportance(int bsdf_id, const T wo[3], const T n[3],
//                              T u1, T u2,
//                              T new_dir[3], T f_val[3], T& pdf) const;
//
//   // Sample a light emission ray.
//   // Fills LightEmissionSample<T>.  Returns false if no valid sample.
//   bool SampleLightEmission(T u_light, T u1, T u2, T u3, T u4,
//                            LightEmissionSample<T>& les) const;
//
//   // (Optional) Direct light-surface -> camera connection for area lights.
//   // Called only when les.has_surface == true.
//   // Fills CameraConnection<T>.  Returns false if no valid connection.
//   bool SampleCameraConnection(const BDPTHit<T>& light_hit,
//                               T u1, T u2,
//                               CameraConnection<T>& cc) const;
//
//   // PDF of sampling the emission direction (wi_light) from a light surface.
//   // Used to weight the direct light-to-camera splat.
//   // Mirrors pbrt-v4 exactly: light.PDF_Li(cs->pLens, -cs->wi)
//   // IMPORTANT: the third argument must be the NEGATED connection direction,
//   //   i.e.  neg_wi = -cc.wi  (direction from camera toward the light surface).
//   T LightPdfLi(int light_id, const T p_lens[3], const T neg_wi[3]) const;
//
//   // Emitted radiance from a light surface point toward the camera.
//   // Mirrors pbrt-v4: light.L(les->intr->p(), les->intr->n, les->intr->uv,
//   //                           cs->wi, lambda)
//   // Called only when les.has_surface == true.
//   // wi_to_camera[3] is the unit direction from light surface toward camera.
//   void LightSurfaceLe(const BDPTHit<T>& light_hit, const T wi_to_camera[3],
//                        T out[3]) const;
//
// ---------------------------------------------------------------------------
// Film concept required API:
//
//   // Splat radiance L[3] at raster position (px, py).
//   // Mirrors pbrt-v4 camera.GetFilm().AddSplat(cs.pRaster, L, lambda).
//   void Splat(T px, T py, const T L[3]);
//
// ---------------------------------------------------------------------------
// Data structs:

//   LightEmissionSample<T>:
//     T  ray_o[3], ray_d[3]  -- emission ray origin and direction
//     T  Le[3]               -- emitted radiance spectrum (RGB)
//     T  pdf_pos             -- area / positional PDF
//     T  pdf_dir             -- directional PDF
//     T  p_light             -- light-selection probability
//     T  abs_cos_theta       -- |cos(theta)| at emission point
//     bool has_surface       -- true for area lights (BDPTHit valid)
//     BDPTHit<T> surface_hit -- filled when has_surface == true
//
//   CameraConnection<T>:
//     T  p_lens[3]   -- point on the camera lens
//     T  p_raster[2] -- raster pixel position (x, y)
//     T  Wi[3]       -- importance weight from camera
//     T  wi[3]       -- direction from surface towards camera
//     T  pdf         -- sampling PDF for this connection
//
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#  if defined(__CUDACC__)
#    define CPU_GPU __host__ __device__ __forceinline__
#  else
#    define CPU_GPU
#  endif
#endif

#include "bdpt.h"        // BDPTHit<T>
#include "scalar_math.h"

#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// LightEmissionSample<T>
// ---------------------------------------------------------------------------
template <typename T>
struct LightEmissionSample {
	T    ray_o[3];
	T    ray_d[3];
	T    Le[3];           // emitted RGB radiance
	T    pdf_pos;         // positional (area) PDF
	T    pdf_dir;         // directional PDF
	T    p_light;         // light-selection probability
	T    abs_cos_theta;   // |cos(theta)| at emission point (AbsCosTheta in pbrt-v4)
	bool has_surface;     // true for area lights
	BDPTHit<T> surface_hit; // valid only when has_surface == true
	int  light_id;
};

// ---------------------------------------------------------------------------
// CameraConnection<T>
// ---------------------------------------------------------------------------
template <typename T>
struct CameraConnection {
	T p_lens[3];    // point on camera lens / sensor
	T p_raster[2];  // raster pixel coordinate (x, y)
	T Wi[3];        // camera importance (cs.Wi in pbrt-v4)
	T wi[3];        // unit direction from surface hit towards camera
	T pdf;          // PDF of this camera connection sample
};

// ---------------------------------------------------------------------------
// LightPathTrace<T, Scene, Film>
// ---------------------------------------------------------------------------
// Traces one complete light path and splats all visible contributions into
// the film.  Mirrors pbrt-v4 LightPathIntegrator::EvaluatePixelSample().
//
// rand1d() -> T  in [0,1)
// rand2d() -> {T,T}  in [0,1)^2
// ---------------------------------------------------------------------------
template <typename T, typename Scene, typename Film, typename Rand1D, typename Rand2D>
CPU_GPU void LightPathTrace(
	const Scene& scene,
	Film&        film,
	int          max_depth,
	Rand1D       rand1d,
	Rand2D       rand2d)
{
	static constexpr T kTMax = T(1e30);

	// ---- Step 1: Sample a light emission ray --------------------------------
	// mirrors pbrt-v4:
	//   ul  = sampler.Get1D()                         (light selection)
	//   ul0 = sampler.Get2D(), ul1 = sampler.Get2D()  (position + direction)
	T u_light = rand1d();
	auto [u0a, u0b] = rand2d();  // positional sample
	auto [u1a, u1b] = rand2d();  // directional sample

	LightEmissionSample<T> les{};
	if (!scene.SampleLightEmission(u_light, u0a, u0b, u1a, u1b, les))
		return;

	// Reject degenerate cases (mirrors pbrt-v4's pdfPos/pdfDir/!L guards)
	if (les.pdf_pos == T(0) || les.pdf_dir == T(0) ||
		(les.Le[0] == T(0) && les.Le[1] == T(0) && les.Le[2] == T(0)))
		return;

	// ---- Initial throughput -------------------------------------------------
	// beta = Le * AbsCosTheta(ray.d) / (p_l * pdfPos * pdfDir)
	T denom = les.p_light * les.pdf_pos * les.pdf_dir;
	T beta[3] = {
		les.Le[0] * les.abs_cos_theta / denom,
		les.Le[1] * les.abs_cos_theta / denom,
		les.Le[2] * les.abs_cos_theta / denom
	};

	// ---- Step 2: Direct area-light -> camera connection ---------------------
	// mirrors pbrt-v4:
	//   if (les.intr) { cs = camera.SampleWi(*les.intr, ...); splat if visible }
	if (les.has_surface) {
		auto [uca, ucb] = rand2d();
		CameraConnection<T> cc{};
		if (scene.SampleCameraConnection(les.surface_hit, uca, ucb, cc) &&
			cc.pdf > T(0))
		{
			// PDF of sampling the connection direction from the light surface.
			// mirrors pbrt-v4: light.PDF_Li(cs->pLens, -cs->wi)
			// Pass -cc.wi (direction from camera toward light) as neg_wi.
			T neg_wi[3] = { -cc.wi[0], -cc.wi[1], -cc.wi[2] };
			T pdf_li = scene.LightPdfLi(les.light_id, cc.p_lens, neg_wi);
			if (pdf_li > T(0)) {
				// Emitted radiance from light surface toward camera.
				// mirrors pbrt-v4: Le = light.L(les->intr->p(), n, uv, cs->wi, lambda)
				T Le_conn[3] = {};
				scene.LightSurfaceLe(les.surface_hit, cc.wi, Le_conn);
				if ((Le_conn[0] > T(0) || Le_conn[1] > T(0) || Le_conn[2] > T(0)) &&
					scene.Unoccluded(les.surface_hit.p, cc.p_lens)) {
					// mirrors: L = Le * DistanceSquared(pRef, pLens) * Wi / (p_l * pdf * cs.pdf)
					T dx = les.surface_hit.p[0] - cc.p_lens[0];
					T dy = les.surface_hit.p[1] - cc.p_lens[1];
					T dz = les.surface_hit.p[2] - cc.p_lens[2];
					T dist2 = dx*dx + dy*dy + dz*dz;
					T w = dist2 / (les.p_light * pdf_li * cc.pdf);
					T splat[3] = {
						Le_conn[0] * cc.Wi[0] * w,
						Le_conn[1] * cc.Wi[1] * w,
						Le_conn[2] * cc.Wi[2] * w
					};
					film.Splat(cc.p_raster[0], cc.p_raster[1], splat);
				}
			}
		}
	}

	// ---- Step 3: Walk the light path ----------------------------------------
	T org[3] = { les.ray_o[0], les.ray_o[1], les.ray_o[2] };
	T dir[3] = { les.ray_d[0], les.ray_d[1], les.ray_d[2] };
	int depth = 0;

	while (true) {
		// Intersect current ray with scene
		BDPTHit<T> hit{};
		if (!scene.Intersect(org, dir, kTMax, hit))
			break;

		// Skip null BSDF / medium boundaries
		// mirrors: "if (!bsdf) { isect.SkipIntersection(&ray, tHit); continue; }"
		if (scene.BSDFIsNull(hit.bsdf_id)) {
			T new_o[3], new_d[3];
			scene.SpawnRay(hit, dir, new_o, new_d);
			org[0] = new_o[0]; org[1] = new_o[1]; org[2] = new_o[2];
			dir[0] = new_d[0]; dir[1] = new_d[1]; dir[2] = new_d[2];
			continue;
		}

		// Terminate if maximum depth reached
		// mirrors: "if (depth++ == maxDepth) break;"
		if (depth++ == max_depth)
			break;

		T wo[3] = { -dir[0], -dir[1], -dir[2] };

		// ---- Camera connection (splat) --------------------------------------
		// mirrors: cs = camera.SampleWi(isect, u, lambda)
		//          if (cs && cs.pdf != 0) { splat beta * bsdf.f * Wi / pdf }
		{
			auto [uca, ucb] = rand2d();
			CameraConnection<T> cc{};
			if (scene.SampleCameraConnection(hit, uca, ucb, cc) && cc.pdf > T(0)) {
				// Evaluate BSDF f(wo -> cc.wi) under importance transport
				T f_imp[3] = {};
				scene.BSDFfImportance(hit.bsdf_id, wo, cc.wi, hit.shading_n, f_imp);

				bool nonzero = (f_imp[0] > T(0) || f_imp[1] > T(0) || f_imp[2] > T(0));
				if (nonzero && scene.Unoccluded(hit.p, cc.p_lens)) {
					// L = beta * bsdf.f_imp * Wi / pdf
					T w = T(1) / cc.pdf;
					T splat[3] = {
						beta[0] * f_imp[0] * cc.Wi[0] * w,
						beta[1] * f_imp[1] * cc.Wi[1] * w,
						beta[2] * f_imp[2] * cc.Wi[2] * w
					};
					film.Splat(cc.p_raster[0], cc.p_raster[1], splat);
				}
			}
		}

		// ---- Sample BSDF for next path segment ------------------------------
		// mirrors: bs = bsdf.Sample_f(isect.wo, uc, u2, TransportMode::Importance)
		//          beta *= bs.f * AbsDot(bs.wi, shading.n) / bs.pdf
		auto [u1, u2] = rand2d();
		T    new_dir[3] = {};
		T    f_val[3]   = {};
		T    pdf        = T(0);

		if (!scene.BSDFSampleFImportance(hit.bsdf_id, wo, hit.shading_n,
										  u1, u2, new_dir, f_val, pdf))
			break;
		if (pdf == T(0))
			break;

		beta[0] *= f_val[0] / pdf;
		beta[1] *= f_val[1] / pdf;
		beta[2] *= f_val[2] / pdf;

		T new_o[3], new_d[3];
		scene.SpawnRay(hit, new_dir, new_o, new_d);
		org[0] = new_o[0]; org[1] = new_o[1]; org[2] = new_o[2];
		dir[0] = new_d[0]; dir[1] = new_d[1]; dir[2] = new_d[2];
	}
}
