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
//   PointLightData<T>   -- isotropic point light, 1/r² falloff
//   SpotLightData<T>    -- cone spotlight with smooth penumbra
//   DistantLightData<T> -- parallel directional (sun-like)
//
// Each struct exposes:
//   eval_Li(px,py,pz, ...)   -- incident radiance at shading point
//   sample_wi(px,py,pz, ...) -- direction toward light (always deterministic)
//   pdf_Li(...)              -- PDF = 0 (delta distribution)
//   power(...)               -- approximate total emitted power (for light sampler)
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
#endif

// ---------------------------------------------------------------------------
// 1. PointLightData<T>
//    Isotropic point source at a fixed world-space position.
//    Li(p) = intensity / distance²
//    pbrt-v4 PointLight::SampleLi: Li = scale * I->Sample(lambda) / DistanceSquared(p, ctx.p())
// ---------------------------------------------------------------------------
template<typename T>
struct PointLightData {
	T pos_x, pos_y, pos_z;   // world-space position
	T ir, ig, ib;             // intensity (RGB, candela)
	T scale;                  // additional scale factor

	// Incident radiance at shading point p (Li = intensity / r²)
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
		// from light → surface: (px-pos_x)/r, (py-pos_y)/r, (pz-pos_z)/r
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

	// Approximate total power = scale * avg(L) * pi * r²
	// pbrt-v4 DistantLight::Phi = scale * Lemit * pi * r²
	CPU_GPU T power() const {
		const T pi = T(3.14159265358979323846);
		T avg_L = (ir + ig + ib) / T(3);
		return scale * avg_L * pi * scene_radius * scene_radius;
	}
};
