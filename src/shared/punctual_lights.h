#pragma once
// ---------------------------------------------------------------------------
// punctual_lights.h -- Shared CPU/GPU punctual light math structs
//
// Mirrors pbrt-v4 PointLight, SpotLight, DistantLight (lights.h / lights.cpp).
//
// Design rules (same as bxdfs.h):
//   - Plain template structs, CPU_GPU tagged
//   - No virtual functions, no heap allocation
//   - T = double on CPU, float on GPU
//
// Three types:
//   PointLightData<T>   -- isotropic point light, 1/rÃ‚Â² falloff
//   SpotLightData<T>    -- cone spotlight with smooth penumbra
//   DistantLightData<T> -- parallel directional (sun-like)
//
// Each struct exposes:
//   eval_Li(px,py,pz, ...)   -- incident radiance at shading point
//   sample_wi(px,py,pz, ...) -- direction toward light (always deterministic)
//   pdf_Li(...)              -- PDF = 0 (delta distribution)
//   power(...)               -- approximate total emitted power (for light sampler)
//
// SpotLightData additionally exposes (mirrors pbrt-v4 SpotLight::SampleLe/PDF_Le):
//   sample_le(ru,rv, wx,wy,wz, pdf_dir) -- importance-sample emitted ray direction
//   pdf_le(wx,wy,wz)                    -- PDF for a given emitted direction
//
// sample_le selects stochastically between inner-cone (uniform) and falloff-annulus
// (SmoothStep-importance-sampled) regions; mirrors pbrt-v4 exactly.
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU
#   endif
#endif

#if defined(__CUDACC__)
#   include <math_functions.h>
#else
#   include <cmath>
#endif

#include "mis_sampling.h"           // SampleUniformCone, UniformConePDF
#include "sampling_distributions.h" // SampleSmoothStep, SmoothStepPDF

// ---------------------------------------------------------------------------
// 1. PointLightData<T>
//    Isotropic point source at a fixed world-space position.
//    Li(p) = intensity / distanceÃ‚Â²
//    pbrt-v4 PointLight::SampleLi: Li = scale * I->Sample(lambda) / DistanceSquared(p, ctx.p())
// ---------------------------------------------------------------------------
template<typename T>
struct PointLightData {
	T pos_x, pos_y, pos_z;   // world-space position
	T ir, ig, ib;             // intensity (RGB, candela)
	T scale;                  // additional scale factor

	// Incident radiance at shading point p (Li = intensity / rÃ‚Â²)
	CPU_GPU void eval_Li(T px, T py, T pz,
						 T& Lr, T& Lg, T& Lb) const {
		T dx = pos_x - px, dy = pos_y - py, dz = pos_z - pz;
		T r2 = dx*dx + dy*dy + dz*dz;
		if (r2 < T(1e-12)) { Lr = Lg = Lb = T(0); return; }
		T inv_r2 = scale / r2;
		Lr = ir * inv_r2;
		Lg = ig * inv_r2;
		Lb = ib * inv_r2;
	}

	// Unit direction from shading point toward the light
	CPU_GPU void sample_wi(T px, T py, T pz,
						   T& wx, T& wy, T& wz) const {
		T dx = pos_x - px, dy = pos_y - py, dz = pos_z - pz;
		T r2 = dx*dx + dy*dy + dz*dz;
#if defined(__CUDACC__)
		T inv_r = (r2 > T(1e-12)) ? rsqrtf(r2) : T(0);
#else
		T inv_r = (r2 > T(1e-12)) ? T(1) / std::sqrt(r2) : T(0);
#endif
		wx = dx * inv_r;
		wy = dy * inv_r;
		wz = dz * inv_r;
	}

	// Delta light: PDF = 0 (handled by skip_pdf in integrator)
	CPU_GPU T pdf_Li() const { return T(0); }

	// Approximate total power = 4*pi * avg(intensity) * scale
	// pbrt-v4 PointLight::Phi = scale * I->Sample(lambda) * 4*pi
	CPU_GPU T power() const {
		const T four_pi = T(4) * T(3.14159265358979323846);
		return scale * four_pi * (ir + ig + ib) / T(3);
	}
};

// ---------------------------------------------------------------------------
// 2. SpotLightData<T>
//    Cone spotlight with smooth (SmoothStep) penumbra falloff.
//    pbrt-v4 SpotLight::I(w) = SmoothStep(CosTheta(w), cosFalloffEnd, cosFalloffStart)
//                              * scale * Iemit->Sample(lambda)
//
//    Parameters:
//      pos_x/y/z          -- world-space position
//      dir_x/y/z          -- world-space unit cone axis (toward scene)
//      ir/ig/ib           -- peak intensity at cone center (RGB)
//      scale              -- global scale
//      cos_falloff_start  -- cos(inner half-angle): full intensity inside
//      cos_falloff_end    -- cos(outer half-angle): zero intensity outside
//
//    Construction helper: from angles in degrees
//      totalWidth      maps to cos_falloff_end   = cos(totalWidth_deg)
//      falloffStart    maps to cos_falloff_start = cos(falloffStart_deg)
// ---------------------------------------------------------------------------
template<typename T>
struct SpotLightData {
	T pos_x, pos_y, pos_z;
	T dir_x, dir_y, dir_z;   // unit cone axis
	T ir, ig, ib;
	T scale;
	T cos_falloff_start;      // inner cone cos (full intensity)
	T cos_falloff_end;        // outer cone cos (zero intensity)

	// SmoothStep falloff: mirrors pbrt-v4 SmoothStep(x, a, b)
	// = 0 if x<=a, 1 if x>=b, smooth cubic in between
	CPU_GPU static T smooth_step(T x, T a, T b) {
		if (x <= a) return T(0);
		if (x >= b) return T(1);
		T t = (x - a) / (b - a);
		return t * t * (T(3) - T(2)*t);
	}

	// Cosine of angle between w and cone axis
	CPU_GPU T cos_theta(T wx, T wy, T wz) const {
		return wx*dir_x + wy*dir_y + wz*dir_z;
	}

	// Falloff factor for a direction from the light (in world space, toward scene)
	CPU_GPU T falloff(T wx, T wy, T wz) const {
		T ct = cos_theta(wx, wy, wz);
		return smooth_step(ct, cos_falloff_end, cos_falloff_start);
	}

	// Incident radiance at shading point p
	CPU_GPU void eval_Li(T px, T py, T pz,
						 T& Lr, T& Lg, T& Lb) const {
		T dx = pos_x - px, dy = pos_y - py, dz = pos_z - pz;
		T r2 = dx*dx + dy*dy + dz*dz;
		if (r2 < T(1e-12)) { Lr = Lg = Lb = T(0); return; }
		T inv_r2 = T(1) / r2;

		// Direction from light to shading point (the direction the spot shines)
#if defined(__CUDACC__)
		T inv_r = sqrtf(inv_r2);
#else
		T inv_r = std::sqrt(inv_r2);
#endif
		// from light Ã¢â€ â€™ surface: (px-pos_x)/r, (py-pos_y)/r, (pz-pos_z)/r
		T wx = -dx * inv_r, wy = -dy * inv_r, wz = -dz * inv_r;

		// Falloff evaluated with direction from light toward surface (= dx/r, ...)
		T f = falloff(wx, wy, wz);
		if (f <= T(0)) { Lr = Lg = Lb = T(0); return; }

		T att = scale * f * inv_r2;
		Lr = ir * att;
		Lg = ig * att;
		Lb = ib * att;
	}

	// Unit direction from shading point toward the light
	CPU_GPU void sample_wi(T px, T py, T pz,
						   T& wx, T& wy, T& wz) const {
		T dx = pos_x - px, dy = pos_y - py, dz = pos_z - pz;
		T r2 = dx*dx + dy*dy + dz*dz;
#if defined(__CUDACC__)
		T inv_r = (r2 > T(1e-12)) ? rsqrtf(r2) : T(0);
#else
		T inv_r = (r2 > T(1e-12)) ? T(1) / std::sqrt(r2) : T(0);
#endif
		wx = dx * inv_r;
		wy = dy * inv_r;
		wz = dz * inv_r;
	}

	CPU_GPU T pdf_Li() const { return T(0); }

	// Approximate total power (pbrt-v4 SpotLight::Phi):
	// scale * I * 2*pi * ((1 - cosFalloffStart) + (cosFalloffStart - cosFalloffEnd)/2)
	CPU_GPU T power() const {
		const T two_pi = T(2) * T(3.14159265358979323846);
		T avg_I = (ir + ig + ib) / T(3);
		return scale * avg_I * two_pi *
			   ((T(1) - cos_falloff_start) + (cos_falloff_start - cos_falloff_end) / T(2));
	}

	// -----------------------------------------------------------------------
	// cone_frame_basis: build two tangent vectors (tx,ty) perpendicular to the
	// cone axis (dir_x,dir_y,dir_z). Used to rotate light-space samples to
	// world space.  Mirrors pbrt-v4's Frame::FromZ construction.
	// -----------------------------------------------------------------------
	CPU_GPU void cone_frame_basis(T& tx, T& ty, T& tz,
								   T& bx, T& by, T& bz) const {
		// Stable Frisvad / Duff et al. orthonormal basis from dir
		double dx = (double)dir_x, dy = (double)dir_y, dz = (double)dir_z;
		double sign = (dz >= 0.0) ? 1.0 : -1.0;
		double a = -1.0 / (sign + dz);
		double b = dx * dy * a;
		tx = (T)(1.0 + sign * dx * dx * a);
		ty = (T)(sign * b);
		tz = (T)(-sign * dx);
		bx = (T)(b);
		by = (T)(sign + dy * dy * a);
		bz = (T)(-dy);
	}

	// -----------------------------------------------------------------------
	// sample_le: importance-sample an emitted ray direction.
	// Mirrors pbrt-v4 SpotLight::SampleLe.
	//
	// ru, rv0, rv1: three independent uniform random numbers in [0,1].
	// Returns world-space direction (wx,wy,wz) and pdf_dir.
	//
	// Algorithm:
	//   p[0] = 1 - cosFalloffStart  (weight of inner cone)
	//   p[1] = (cosFalloffStart - cosFalloffEnd) / 2  (weight of falloff annulus)
	//   With prob p[0]/(p[0]+p[1]): sample inner cone uniformly.
	//   Otherwise: importance-sample falloff annulus via SampleSmoothStep.
	// -----------------------------------------------------------------------
	void sample_le(double ru, double rv0, double rv1,
				   double& wx, double& wy, double& wz,
				   double& pdf_dir) const {
		double cf_start = (double)cos_falloff_start;
		double cf_end   = (double)cos_falloff_end;

		// Region weights (mirror pbrt-v4 Float p[2])
		double p0 = 1.0 - cf_start;
		double p1 = (cf_start - cf_end) / 2.0;
		double p_sum = p0 + p1;

		// Light-space direction (about +Z = cone axis)
		double lx, ly, lz;
		if (ru * p_sum < p0) {
			// Inner cone: uniform sampling
			SampleUniformCone(rv0, rv1, cf_start, lx, ly, lz);
			pdf_dir = UniformConePDF(cf_start) * (p0 / p_sum);
		} else {
			// Falloff annulus: SmoothStep importance sampling
			double cosTheta = SampleSmoothStep(rv0, cf_end, cf_start);
			double sinTheta = std::sqrt(std::max(0.0, 1.0 - cosTheta * cosTheta));
			double phi = rv1 * 2.0 * 3.14159265358979323846;
			lx = std::cos(phi) * sinTheta;
			ly = std::sin(phi) * sinTheta;
			lz = cosTheta;
			pdf_dir = SmoothStepPDF(cosTheta, cf_end, cf_start) * (p1 / p_sum)
					/ (2.0 * 3.14159265358979323846);
		}

		// Rotate from light space (+Z = dir) to world space
		T tx, ty, tz, bx, by, bz;
		cone_frame_basis(tx, ty, tz, bx, by, bz);
		wx = (double)tx * lx + (double)bx * ly + (double)dir_x * lz;
		wy = (double)ty * lx + (double)by * ly + (double)dir_y * lz;
		wz = (double)tz * lx + (double)bz * ly + (double)dir_z * lz;
	}

	// -----------------------------------------------------------------------
	// pdf_le: PDF of emitting in given world-space direction (wx,wy,wz).
	// Mirrors pbrt-v4 SpotLight::PDF_Le.
	// -----------------------------------------------------------------------
	double pdf_le(double wx, double wy, double wz) const {
		double cf_start = (double)cos_falloff_start;
		double cf_end   = (double)cos_falloff_end;
		double p0 = 1.0 - cf_start;
		double p1 = (cf_start - cf_end) / 2.0;
		double p_sum = p0 + p1;

		// cosTheta between direction and cone axis
		double ct = wx * (double)dir_x + wy * (double)dir_y + wz * (double)dir_z;

		if (ct >= cf_start) {
			// Inner cone -- mirrors pbrt-v4 PDF_Le if branch
			return UniformConePDF(cf_start) * (p0 / p_sum);
		} else {
			// Falloff annulus or outside -- SmoothStepPDF returns 0 for ct < cf_end
			// exactly as pbrt-v4's bare else branch.
			return SmoothStepPDF(ct, cf_end, cf_start) * (p1 / p_sum)
				 / (2.0 * 3.14159265358979323846);
		}
	}
};

// ---------------------------------------------------------------------------
// 3. DistantLightData<T>
//    Parallel directional light (sun-like).
//    Constant Li in the given direction; PDF = 0 (delta direction).
//    pbrt-v4 DistantLight::SampleLi: wi = normalize(renderFromLight(Vector3f(0,0,1)))
//                                    Li = scale * Lemit->Sample(lambda)
// ---------------------------------------------------------------------------
template<typename T>
struct DistantLightData {
	T dir_x, dir_y, dir_z;   // unit world-space direction TOWARD the scene (wi)
	T ir, ig, ib;             // radiance (RGB)
	T scale;
	T scene_radius;           // scene bounding sphere radius (for power estimate)

	CPU_GPU T pdf_Li() const { return T(0); }

	// Incident radiance -- constant in this direction regardless of position
	CPU_GPU void eval_Li(T& Lr, T& Lg, T& Lb) const {
		Lr = ir * scale;
		Lg = ig * scale;
		Lb = ib * scale;
	}

	// Direction toward light = dir (already world-space)
	CPU_GPU void sample_wi(T& wx, T& wy, T& wz) const {
		wx = dir_x; wy = dir_y; wz = dir_z;
	}

	// Approximate total power = scale * avg(L) * pi * rÃ‚Â²
	// pbrt-v4 DistantLight::Phi = scale * Lemit * pi * rÃ‚Â²
	CPU_GPU T power() const {
		const T pi = T(3.14159265358979323846);
		T avg_L = (ir + ig + ib) / T(3);
		return scale * avg_L * pi * scene_radius * scene_radius;
	}
};
