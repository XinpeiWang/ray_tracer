#pragma once
// ---------------------------------------------------------------------------
// vol_path.h -- Full Volumetric Path Tracer (pbrt-v4 port)
//
// Ports pbrt-v4:
//   cpu/integrators.cpp  -- VolPathIntegrator::Li, VolPathIntegrator::SampleLd
//
// Components:
//   VolPathMediumProps<T>    -- per-point medium properties (sigma_a/s/n, Le, g)
//   VolPathLightSample<T>    -- result of sampling a light source direction
//   VolPathSampleLd<T,Sc>    -- NEE with volumetric transmittance + MIS weight
//   VolPathLi<T,Sc>          -- full path-tracing loop: Li estimate for one ray
//
// Design rules (same as bdpt.h / mlt.h / sppm.h):
//   - Header-only, no virtual functions
//   - Template T: float or double
//   - Template Scene: duck-typed scene concept (see below)
//   - Single-path reference implementation; callers loop over pixels/samples
//   - RGB output (no spectral wavelengths)
//
// Omissions vs. pbrt-v4:
//   - Sampler jitter/stratification (callers provide random numbers)
//
// Scene concept (extends sppm.h / bdpt.h concept):
//   // Basic geometry
//   bool  Intersect(const T org[3], const T dir[3], T t_max, BDPTHit<T>&) const
//   bool  Unoccluded(const T p0[3], const T p1[3]) const
//
//   // Light sampling
//   bool  SampleLight(T u, const T ref_p[3], BDPTLightSample<T>&) const
//   T     LightPMF(int id) const
//   bool  IsDeltaLight(int id) const
//   float LightPDF_Li(int id, const T ref_p[3], const T wi[3]) const
//
//   // BSDF
//   void  BSDFf(int id, const T wo[3], const T wi[3], const T n[3],
//               T out[3]) const
//   bool  BSDFSampleF(int id, const T wo[3], const T n[3], T u1, T u2,
//                     T new_dir[3], T f_val[3], T& pdf,
//                     bool& is_specular, bool& is_transmission) const
//   T     BSDFPdf(int id, const T wo[3], const T wi[3], const T n[3]) const
//   bool  BSDFIsNonSpecular(int bsdf_id) const
//   void  BSDFRegularize(int bsdf_id)   // only called when regularize=true
//
//   // BSSRDF subsurface scattering (pbrt-v4 VolPathIntegrator::Li lines 1187-1255)
//   // Return false if this surface hit has no BSSRDF (the branch is skipped).
//   bool  HasBSSRDF(const BDPTHit<T>& hit) const
//   // Sample a disc probe segment from the entry point.
//   // Fills seg.p0/p1 (world-space endpoints). Returns false on failure.
//   bool  SampleBSSRDFProbe(const BDPTHit<T>& hit,
//                            T u1, T u2,
//                            BSSRDFProbeSegment<T>& seg) const
//   // Given an exit-surface hit found along the probe segment,
//   // fill out a BSSRDFSample<T> (Sp, pdf, exit wo, exit bsdf_id).
//   // Returns false on failure.
//   bool  SampleBSSRDF(const BDPTHit<T>& entry_hit,
//                      const BDPTHit<T>& exit_hit,
//                      T sample_prob,
//                      BSSRDFSample<T>& out) const
//
//   // Medium: null-scattering (majorant) walk
//   bool  HasMedium(const T org[3], const T dir[3]) const
//   T     SampleTMaj(const T org[3], const T dir[3], T t_max,
//                    T u_maj,
//                    const std::function<bool(
//                        const T p[3],
//                        const VolPathMediumProps<T>& mp,
//                        T sigma_maj, T T_maj)>& callback) const
//   // Phase function (Henyey-Greenstein or similar)
//   T     PhaseP(const T wo[3], const T wi[3], T g) const
//   T     PhasePDF(const T wo[3], const T wi[3], T g) const
//   bool  SamplePhase(const T wo[3], T g, T u1, T u2,
//                     T wi[3], T& pdf) const
//
//   // Infinite-light background
//   void  InfiniteLightLe(const T d[3], T out[3]) const
//   float InfiniteLightPMF() const
//   float InfiniteLightPDF_Li(const T ref_p[3], const T wi[3]) const
//
//   // Random numbers
//   T     RandFloat() const
//
//   // Spawn an offset ray from a surface hit to avoid self-intersection
//   void  SpawnRay(const BDPTHit<T>& hit, const T dir[3],
//                  T new_o[3], T new_d[3]) const
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU inline
#   endif
#endif

#include "bdpt.h"   // BDPTHit, BDPTLightSample, BDPTLightLeSample
#include "reservoir_sampler.h"  // WeightedReservoirSampler

#include <cmath>
#include <functional>
#include <algorithm>
#include <limits>
#include <cstdint>

// ===========================================================================
// BSSRDFProbeSegment<T> -- world-space endpoints of the BSSRDF probe disc
//
// Mirrors pbrt-v4 BSSRDFProbeSegment (bssrdf.h).
// The scene adapter fills p0 (entry side) and p1 (opposite disc edge)
// so that the integrator can walk the segment to find the exit surface.
// ===========================================================================
template<typename T>
struct BSSRDFProbeSegment {
	T p0[3] = {}; // start of probe segment (just inside the entry surface)
	T p1[3] = {}; // end of probe segment (maximum exit point on opposite side)
};

// ===========================================================================
// BSSRDFSample<T> -- result of sampling the subsurface exit point
//
// Mirrors pbrt-v4 BSSRDFSample (bssrdf.h).
// Sp     -- profile value (RGB, i.e. one value per channel; here 3 floats)
// pdf    -- probability density of this exit point (solid-angle measure)
// wo     -- outgoing direction at exit point (pointing outward from surface)
// exit_bsdf_id -- BSDF id to use for indirect scatter at exit point
// ===========================================================================
template<typename T>
struct BSSRDFSample {
	T  Sp[3]       = {};
	T  pdf         = T(0);
	T  wo[3]       = {};
	int exit_bsdf_id = 0;
};

// ===========================================================================
// Section 1 -- VolPathMediumProps<T>
// ===========================================================================
// Per-point medium properties, returned by the scene adapter inside
// the SampleTMaj callback. Mirrors pbrt-v4 MediumProperties.

template<typename T>
struct VolPathMediumProps {
	T sigma_a = T(0);   // absorption coefficient (scalar for RGB)
	T sigma_s = T(0);   // scattering coefficient
	T Le      = T(0);   // emission (scalar luminance; RGB not needed for correctness)
	T g       = T(0);   // HG phase function asymmetry parameter
};

// ===========================================================================
// Section 2 -- VolPathSampleLd
// ===========================================================================
// Next-event estimation at a surface or medium interaction point with
// volumetric transmittance and spectral MIS weighting.
//
// Mirrors pbrt-v4 VolPathIntegrator::SampleLd.
//
// Parameters:
//   p, wo        -- interaction point and outgoing direction
//   n            -- shading normal (used for surface; zero for medium)
//   bsdf_id      -- BSDF id for surface (-1 for medium interaction)
//   g            -- HG asymmetry (used when bsdf_id < 0)
//   beta[3]      -- current path throughput (RGB)
//   r_u          -- unidirectional PDF ratio accumulator (scalar)
//   scene        -- scene query adapter
//
// Returns Ld[3] contribution to add to L.

template<typename T, typename Scene>
void VolPathSampleLd(const T p[3], const T wo[3], const T n[3],
					 int bsdf_id, T g,
					 const T beta[3], T r_u,
					 const Scene& scene,
					 T Ld_out[3]) {
	Ld_out[0] = Ld_out[1] = Ld_out[2] = T(0);

	// Sample a light source
	BDPTLightSample<T> ls{};
	T u_light = scene.RandFloat();
	if (!scene.SampleLight(u_light, p, ls)) return;
	if (ls.pdf <= T(0)) return;

	T p_l = ls.pdf;  // combined pmf * pdf_Li

	// Evaluate scatter function (BSDF or phase)
	T f_hat[3] = {T(0), T(0), T(0)};
	T scatter_pdf = T(0);
	T wi[3] = { ls.wi[0], ls.wi[1], ls.wi[2] };

	if (bsdf_id >= 0) {
		// Surface: f * |cos(theta)|
		T f[3];
		scene.BSDFf(bsdf_id, wo, wi, n, f);
		T cos_i = std::abs(wi[0]*n[0] + wi[1]*n[1] + wi[2]*n[2]);
		f_hat[0] = f[0] * cos_i;
		f_hat[1] = f[1] * cos_i;
		f_hat[2] = f[2] * cos_i;
		scatter_pdf = scene.BSDFPdf(bsdf_id, wo, wi, n);
	} else {
		// Medium: phase function
		T ph = scene.PhaseP(wo, wi, g);
		f_hat[0] = f_hat[1] = f_hat[2] = ph;
		scatter_pdf = scene.PhasePDF(wo, wi, g);
	}

	if (f_hat[0] == T(0) && f_hat[1] == T(0) && f_hat[2] == T(0)) return;

	// Trace shadow ray through media to build T_ray and PDF ratios
	T T_ray = T(1);
	T r_l = T(1), r_u_shadow = T(1);

	// Shadow ray from p toward light point
	T light_p[3] = { ls.p_light[0], ls.p_light[1], ls.p_light[2] };
	T shadow_d[3] = { light_p[0]-p[0], light_p[1]-p[1], light_p[2]-p[2] };
	T dist = std::sqrt(shadow_d[0]*shadow_d[0] +
					   shadow_d[1]*shadow_d[1] +
					   shadow_d[2]*shadow_d[2]);
	if (dist < T(1e-7)) return;
	shadow_d[0] /= dist; shadow_d[1] /= dist; shadow_d[2] /= dist;

	// Check for opaque surface occlusion first (no medium)
	if (!scene.HasMedium(p, shadow_d)) {
		// Vacuum shadow ray — just check geometry
		if (!scene.Unoccluded(p, light_p)) return;
	} else {
		// Walk majorant through media accumulating T_ray and r_u/r_l
		bool blocked = false;
		T t_max = dist * T(1 - 1e-3);
		T u_maj = scene.RandFloat();

		T T_maj_final = scene.SampleTMaj(p, shadow_d, t_max, u_maj,
			[&](const T /*pt*/[3],
				const VolPathMediumProps<T>& mp,
				T sigma_maj, T T_maj_val) -> bool {
				// sigma_n = max(0, sigma_maj - sigma_a - sigma_s)
				T sigma_n = std::max(T(0),
					sigma_maj - mp.sigma_a - mp.sigma_s);
				T pdf = T_maj_val * sigma_maj;
				if (pdf == T(0)) return true;
				T_ray     *= T_maj_val * sigma_n / pdf;
				r_l       *= T_maj_val * sigma_maj / pdf;
				r_u_shadow *= T_maj_val * sigma_n / pdf;

				// Russian roulette on transmittance estimate
				T Tr_est = (r_l + r_u_shadow) > T(0)
					? T_ray / ((r_l + r_u_shadow) * T(0.5))
					: T(0);
				if (Tr_est < T(0.05)) {
					T q = T(0.75);
					if (scene.RandFloat() < q) {
						T_ray = T(0);
						blocked = true;
						return false;
					}
					T_ray /= T(1) - q;
				}
				return T_ray > T(0);
			});

		if (blocked || T_ray == T(0)) return;
		// Apply final segment factor
		if (T_maj_final > T(0)) {
			T_ray      *= T_maj_final;
			r_l        *= T_maj_final;
			r_u_shadow *= T_maj_final;
		}
	}

	if (T_ray == T(0)) return;

	// MIS weight: delta lights use r_l only; area lights use (r_l + r_u)
	// Mirrors pbrt-v4:
	//   delta: return beta * f_hat * T_ray * L / r_l.Average()
	//   area:  return beta * f_hat * T_ray * L / (r_l + r_u).Average()
	r_l       *= r_u * p_l;
	r_u_shadow *= r_u * scatter_pdf;

	T mis_denom;
	if (scene.IsDeltaLight(ls.light_id))
		mis_denom = r_l;
	else
		mis_denom = (r_l + r_u_shadow) * T(0.5);  // Average() of 2-element sum

	if (mis_denom <= T(0)) return;

	T scale = T_ray / mis_denom;
	for (int c = 0; c < 3; ++c)
		Ld_out[c] = beta[c] * f_hat[c] * ls.L[c] * scale;
}

// ===========================================================================
// Section 3 -- VolPathLi
// ===========================================================================
// Full volumetric path tracer: compute Li for one camera ray.
// Mirrors pbrt-v4 VolPathIntegrator::Li.
//
// State variables (matching pbrt-v4 names):
//   L[3]          -- accumulated radiance estimate
//   beta[3]       -- path throughput
//   r_u           -- unidirectional PDF ratio (scalar, product of f/pdf ratios)
//   r_l           -- NEE PDF ratio (set after each scatter event for next iter)
//   specularBounce-- true if last bounce was specular (disables MIS on emission)
//   etaScale      -- accumulated index-of-refraction scale for Russian roulette
//   depth         -- current path depth
//
// Parameters:
//   org, dir     -- ray origin and direction
//   maxDepth     -- maximum path length
//   scene        -- scene query adapter
//   L_out[3]     -- output radiance (RGB)

template<typename T, typename Scene>
void VolPathLi(const T org[3], const T dir[3],
			   int maxDepth,
			   const Scene& scene,
			   T L_out[3],
			   bool regularize = false) {
	L_out[0] = L_out[1] = L_out[2] = T(0);

	// Path state (mirrors pbrt-v4 VolPathIntegrator::Li locals)
	T L[3]    = {T(0), T(0), T(0)};
	T beta[3] = {T(1), T(1), T(1)};
	T r_u     = T(1);
	T r_l     = T(1);
	bool specularBounce = true;
	bool any_non_specular_bounce = false;
	T etaScale = T(1);
	int depth  = 0;

	T ray_o[3] = { org[0], org[1], org[2] };
	T ray_d[3] = { dir[0], dir[1], dir[2] };

	while (true) {
		// ---------------------------------------------------------------
		// 1. Intersect the scene
		// ---------------------------------------------------------------
		BDPTHit<T> hit{};
		bool did_hit = scene.Intersect(ray_o, ray_d,
									   std::numeric_limits<T>::max(), hit);
		T t_max = did_hit ? hit.t_hit : std::numeric_limits<T>::max();

		// ---------------------------------------------------------------
		// 2. Handle participating medium along the ray segment
		// ---------------------------------------------------------------
		bool scattered   = false;
		bool terminated  = false;

		if (scene.HasMedium(ray_o, ray_d)) {
			T u_maj = scene.RandFloat();
			T T_maj_final = scene.SampleTMaj(ray_o, ray_d, t_max, u_maj,
				[&](const T p_med[3],
					const VolPathMediumProps<T>& mp,
					T sigma_maj, T T_maj_val) -> bool {

					if (beta[0]==T(0) && beta[1]==T(0) && beta[2]==T(0)) {
						terminated = true; return false;
					}

					// Medium emission
					if (depth < maxDepth && mp.Le > T(0)) {
						T pdf_e = sigma_maj * T_maj_val;
						if (pdf_e > T(0)) {
							T r_e = r_u * sigma_maj * T_maj_val / pdf_e;
							if (r_e > T(0)) {
								T contrib = T_maj_val * mp.sigma_a * mp.Le / pdf_e / r_e;
								for (int c = 0; c < 3; ++c) L[c] += beta[c] * contrib;
							}
						}
					}

					// Event probabilities
					T pAbsorb  = (sigma_maj > T(0)) ? mp.sigma_a / sigma_maj : T(0);
					T pScatter = (sigma_maj > T(0)) ? mp.sigma_s / sigma_maj : T(0);
					T pNull    = std::max(T(0), T(1) - pAbsorb - pScatter);

					// Sample event type
					T um = scene.RandFloat();
					if (um < pAbsorb) {
						// Absorption: terminate
						terminated = true;
						return false;
					} else if (um < pAbsorb + pScatter) {
						// Real scatter
						if (depth++ >= maxDepth) { terminated = true; return false; }

						T pdf_s = T_maj_val * mp.sigma_s;
						if (pdf_s > T(0)) {
							for (int c = 0; c < 3; ++c)
								beta[c] *= T_maj_val * mp.sigma_s / pdf_s;
							r_u *= T_maj_val * mp.sigma_s / pdf_s;
						}

						// NEE at medium scatter point
						T zero_n[3] = {T(0), T(0), T(0)};
						T Ld[3];
						VolPathSampleLd(p_med, ray_d, zero_n,
										/*bsdf_id=*/-1, mp.g,
										beta, r_u, scene, Ld);
						for (int c = 0; c < 3; ++c) L[c] += Ld[c];

						// Sample new direction (phase function)
						T new_d[3]; T phase_pdf;
						T up1 = scene.RandFloat(), up2 = scene.RandFloat();
						if (!scene.SamplePhase(ray_d, mp.g, up1, up2,
											   new_d, phase_pdf) ||
							phase_pdf == T(0)) {
							terminated = true; return false;
						}

						T ph = scene.PhaseP(ray_d, new_d, mp.g);
						for (int c = 0; c < 3; ++c)
							beta[c] *= ph / phase_pdf;
						r_l = r_u / phase_pdf;

						// Advance ray
						ray_o[0]=p_med[0]; ray_o[1]=p_med[1]; ray_o[2]=p_med[2];
						ray_d[0]=new_d[0]; ray_d[1]=new_d[1]; ray_d[2]=new_d[2];
						specularBounce = false;
								scattered = true;
								return false;  // stop the majorant walk; restart outer loop

							} else {
								// Null scatter: update beta, r_u, and r_l
								// pbrt-v4: beta *= T_maj * sigma_n / pdf
								//          r_u  *= T_maj * sigma_n / pdf
								//          r_l  *= T_maj * sigma_maj / pdf   <-- majorant weight
								T sigma_n = std::max(T(0),
									sigma_maj - mp.sigma_a - mp.sigma_s);
								T pdf_n = T_maj_val * sigma_maj;
								if (pdf_n > T(0)) {
									for (int c = 0; c < 3; ++c)
										beta[c] *= T_maj_val * sigma_n / pdf_n;
									r_u *= T_maj_val * sigma_n / pdf_n;
									r_l *= T_maj_val * sigma_maj / pdf_n;
								} else {
									// pdf==0 means sigma_maj==0: pure vacuum, no update needed
								}
								return (beta[0] != T(0) || beta[1] != T(0) || beta[2] != T(0));
							}
						});

			// Apply final residual T_maj factor from the end of the segment
			if (!scattered && !terminated && T_maj_final != T(1)) {
				for (int c = 0; c < 3; ++c) beta[c] *= T_maj_final;
				r_u *= T_maj_final;
				r_l *= T_maj_final;
			}
		}

		if (terminated) break;
		if (scattered)  continue;

		// ---------------------------------------------------------------
		// 3. Handle ray that missed the scene (background / infinite lights)
		// ---------------------------------------------------------------
		if (!did_hit) {
			T Le_inf[3];
			scene.InfiniteLightLe(ray_d, Le_inf);
			if (Le_inf[0] != T(0) || Le_inf[1] != T(0) || Le_inf[2] != T(0)) {
				if (depth == 0 || specularBounce) {
					// Direct: weight by r_u only
					if (r_u > T(0))
						for (int c = 0; c < 3; ++c)
							L[c] += beta[c] * Le_inf[c] / r_u;
				} else {
					// MIS: (r_u + r_l).Average()
					T r_l_cur = r_l * scene.InfiniteLightPMF()
							  * scene.InfiniteLightPDF_Li(ray_o, ray_d);
					T denom = (r_u + r_l_cur) * T(0.5);
					if (denom > T(0))
						for (int c = 0; c < 3; ++c)
							L[c] += beta[c] * Le_inf[c] / denom;
				}
			}
			break;
		}

		// ---------------------------------------------------------------
		// 4. Surface hit: emission
		// ---------------------------------------------------------------
		// Area light emission on the hit surface
		{
			T* Le = hit.area_Le;
			if (Le[0] != T(0) || Le[1] != T(0) || Le[2] != T(0)) {
				if (depth == 0 || specularBounce) {
					if (r_u > T(0))
						for (int c = 0; c < 3; ++c)
							L[c] += beta[c] * Le[c] / r_u;
				} else {
					// MIS with light-sampling PDF
					T p_l = scene.LightPMF(hit.light_id)
						  * scene.LightPDF_Li(hit.light_id, ray_o, ray_d);
					T r_l_cur = r_l * p_l;
					T denom = (r_u + r_l_cur) * T(0.5);
					if (denom > T(0))
						for (int c = 0; c < 3; ++c)
							L[c] += beta[c] * Le[c] / denom;
				}
			}
		}

		// ---------------------------------------------------------------
		// 5. Terminate at max depth
		// ---------------------------------------------------------------
		if (depth++ >= maxDepth) break;

		// ---------------------------------------------------------------
		// 6. Path regularization (pbrt-v4 VolPathIntegrator::Li lines 1151-1153)
		// ---------------------------------------------------------------
		if (regularize && any_non_specular_bounce)
			scene.BSDFRegularize(hit.bsdf_id);

		// ---------------------------------------------------------------
		// 7. NEE at surface (non-specular only)
		// ---------------------------------------------------------------
		if (scene.BSDFIsNonSpecular(hit.bsdf_id)) {
			T Ld[3];
			VolPathSampleLd(hit.p, hit.wo, hit.shading_n,
							hit.bsdf_id, T(0),
							beta, r_u, scene, Ld);
			for (int c = 0; c < 3; ++c) L[c] += Ld[c];
		}

		// ---------------------------------------------------------------
		// 7. Sample BSDF for indirect direction
		// ---------------------------------------------------------------
		T new_dir[3], f_val[3], bsdf_pdf;
		bool is_specular;
		bool is_transmission = false;
		T u1 = scene.RandFloat(), u2 = scene.RandFloat();
		if (!scene.BSDFSampleF(hit.bsdf_id, hit.wo, hit.shading_n,
							   u1, u2, new_dir, f_val, bsdf_pdf, is_specular, is_transmission))
			break;
		if (bsdf_pdf <= T(0)) break;

		T cos_i = std::abs(new_dir[0]*hit.shading_n[0] +
						   new_dir[1]*hit.shading_n[1] +
						   new_dir[2]*hit.shading_n[2]);

		// Update beta and r_u
		for (int c = 0; c < 3; ++c)
			beta[c] *= f_val[c] * cos_i / bsdf_pdf;

		// r_l = r_u / bsdf_pdf  (mirrors pbrt-v4: r_l = r_u / bs->pdf)
		r_l = r_u / bsdf_pdf;
		// After BSDF scatter, r_u stays unchanged until next segment
		// (it will be updated by the medium walk at the start of next iteration)

		specularBounce = is_specular;
		any_non_specular_bounce |= !is_specular;

		// Transmission: accumulate eta^2 for Russian roulette
		// (simplified: callers encode eta in the BSDF f_val weighting)

		// ---------------------------------------------------------------
		// 8. BSSRDF subsurface scattering branch
		//    (pbrt-v4 VolPathIntegrator::Li lines 1187-1255)
		//
		// When the scattered direction is a transmission (entered the
		// surface) and the hit material has a BSSRDF, we abandon the
		// refracted ray and instead find the subsurface exit point by
		// probing the geometry with a disc-shaped importance-sampled
		// segment.  The exit point replaces the next surface hit.
		// Depth is NOT incremented (mirrors pbrt-v4 "Don't increment
		// depth this time").
		// ---------------------------------------------------------------
		if (is_transmission && scene.HasBSSRDF(hit)) {
			// Sample the probe segment endpoints
			T uc = scene.RandFloat();
			T up1 = scene.RandFloat(), up2 = scene.RandFloat();
			BSSRDFProbeSegment<T> seg{};
			if (!scene.SampleBSSRDFProbe(hit, up1, up2, seg))
				break;

			// Walk the probe segment and reservoir-sample one exit hit
			// with uniform probability (all intersecting faces of the
			// same material are weighted equally, mirroring pbrt-v4's
			// WeightedReservoirSampler<SubsurfaceInteraction> with weight=1).
			uint64_t rs_seed = static_cast<uint64_t>(
				*reinterpret_cast<const uint32_t*>(&uc) ^
				static_cast<uint32_t>(depth * 2654435761u));
			WeightedReservoirSampler<BDPTHit<T>> wrs(rs_seed);
			{
				// Segment direction and length
				T seg_d[3] = {
					seg.p1[0] - seg.p0[0],
					seg.p1[1] - seg.p0[1],
					seg.p1[2] - seg.p0[2]
				};
				T seg_len = std::sqrt(seg_d[0]*seg_d[0] +
									  seg_d[1]*seg_d[1] +
									  seg_d[2]*seg_d[2]);
				if (seg_len < T(1e-8)) break;
				seg_d[0] /= seg_len;
				seg_d[1] /= seg_len;
				seg_d[2] /= seg_len;

				// Walk: intersect the probe segment repeatedly,
				// advancing the origin past each hit, collecting
				// exit-surface candidates.
				T base[3] = { seg.p0[0], seg.p0[1], seg.p0[2] };
				T remaining = seg_len;
				while (remaining > T(1e-8)) {
					BDPTHit<T> probe_hit{};
					if (!scene.Intersect(base, seg_d, remaining, probe_hit))
						break;
					// Uniform weight = 1 for each candidate hit
					wrs.add(probe_hit, 1.0);
					// Advance past this hit
					T step = probe_hit.t_hit + T(1e-4);
					base[0] += step * seg_d[0];
					base[1] += step * seg_d[1];
					base[2] += step * seg_d[2];
					remaining -= step;
				}
			}

			if (!wrs.has_sample()) break;

			// Convert the chosen exit hit to a BSSRDFSample
			BSSRDFSample<T> bss{};
			if (!scene.SampleBSSRDF(hit, wrs.sample(),
									static_cast<T>(wrs.sample_probability()),
									bss))
				break;
			if (bss.pdf <= T(0)) break;

			// Update path throughput and unidirectional PDF ratio
			// beta *= Sp / (sample_prob * pdf)  -- mirrors pbrt-v4
			// r_u  *= pdf / pdf_ch0             -- multi-channel correction
			//   (for RGB, pdf is scalar so r_u is unchanged; kept for correctness)
			T inv_pdf = T(1) / (static_cast<T>(wrs.sample_probability()) * bss.pdf);
			for (int c = 0; c < 3; ++c)
				beta[c] *= bss.Sp[c] * inv_pdf;
			// r_u unchanged for scalar pdf (bss.pdf == bss.pdf for all channels)

			// Subsurface scatter is always diffuse (pbrt-v4: anyNonSpecularBounces=true)
			any_non_specular_bounce = true;
			specularBounce = false;

			// Possibly regularize exit-point BSDF (pbrt-v4 lines 1234-1239)
			if (regularize)
				scene.BSDFRegularize(bss.exit_bsdf_id);

			// NEE at exit point
			if (scene.BSDFIsNonSpecular(bss.exit_bsdf_id)) {
				T Ld[3] = {};
				const BDPTHit<T>& exit_hit = wrs.sample();
				VolPathSampleLd(exit_hit.p, bss.wo, exit_hit.shading_n,
								bss.exit_bsdf_id, T(0),
								beta, r_u, scene, Ld);
				for (int c = 0; c < 3; ++c) L[c] += Ld[c];
			}

			// Indirect scatter at exit point (depth NOT incremented)
			{
				const BDPTHit<T>& exit_hit = wrs.sample();
				T ex_u1 = scene.RandFloat(), ex_u2 = scene.RandFloat();
				T ex_dir[3] = {}, ex_f[3] = {}, ex_pdf = T(0);
					bool ex_spec = false, ex_trans = false;
					if (!scene.BSDFSampleF(bss.exit_bsdf_id, bss.wo,
										   exit_hit.shading_n,
										   ex_u1, ex_u2,
										   ex_dir, ex_f, ex_pdf, ex_spec, ex_trans))
					break;
				if (ex_pdf <= T(0)) break;

				T ex_cos = std::abs(ex_dir[0]*exit_hit.shading_n[0] +
									ex_dir[1]*exit_hit.shading_n[1] +
									ex_dir[2]*exit_hit.shading_n[2]);
				for (int c = 0; c < 3; ++c)
					beta[c] *= ex_f[c] * ex_cos / ex_pdf;

				r_l = r_u / ex_pdf;
				specularBounce = ex_spec;

				// Spawn ray from exit point
				T ex_o[3], ex_d[3];
				scene.SpawnRay(exit_hit, ex_dir, ex_o, ex_d);
				ray_o[0] = ex_o[0]; ray_o[1] = ex_o[1]; ray_o[2] = ex_o[2];
				ray_d[0] = ex_d[0]; ray_d[1] = ex_d[1]; ray_d[2] = ex_d[2];
			}

			// ---------------------------------------------------------------
			// 9. Russian roulette (after BSSRDF scatter)
			// ---------------------------------------------------------------
			T rrBeta_ss = std::max({beta[0], beta[1], beta[2]}) * etaScale;
			if (r_u > T(0)) rrBeta_ss /= r_u;
			if (rrBeta_ss < T(1) && depth > 1) {
				T q = std::max(T(0), T(1) - rrBeta_ss);
				if (scene.RandFloat() < q) break;
				for (int c = 0; c < 3; ++c) beta[c] /= T(1) - q;
			}
			continue; // restart outer loop with new ray from exit point
		}

		// ---------------------------------------------------------------
		// 9. Russian roulette  (mirrors pbrt-v4: rrBeta = beta*etaScale/r_u.Average())
		// ---------------------------------------------------------------
		T rrBeta = std::max({beta[0], beta[1], beta[2]}) * etaScale;
		if (r_u > T(0)) rrBeta /= r_u;
		if (rrBeta < T(1) && depth > 1) {
			T q = std::max(T(0), T(1) - rrBeta);
			if (scene.RandFloat() < q) break;
			for (int c = 0; c < 3; ++c) beta[c] /= T(1) - q;
		}

		// Advance ray
		ray_o[0]=hit.p[0]; ray_o[1]=hit.p[1]; ray_o[2]=hit.p[2];
		ray_d[0]=new_dir[0]; ray_d[1]=new_dir[1]; ray_d[2]=new_dir[2];
	}

	L_out[0] = L[0];
	L_out[1] = L[1];
	L_out[2] = L[2];
}
