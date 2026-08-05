#pragma once
// ---------------------------------------------------------------------------
// goniometric_light.h -- Goniometric (IES-profile) point light
//
// Mirrors pbrt-v4 GoniometricLight (src/pbrt/lights.h, lights.cpp).
//
// A GoniometricLight is a point light whose emitted intensity varies with
// direction according to a 2D goniometric diagram stored as an equal-area-
// projected image.  This directly models IES photometric profiles.
//
// Algorithm (pbrt-v4 §12.4):
//   I(w_world, lambda) = scale * Iemit(lambda)
//                        * image.Lookup(EqualAreaSphereToSquare(w_light))
//   where w_light = world_to_light * w_world
//
//   SampleLi: delta point light — wi = normalize(pos - p), Li = I(wi) / r²
//   PDF_Li  : 0 (delta distribution)
//   Power   : scale * avg_image_value * 4π  (solid-angle integral over sphere)
//   SampleLe: importance-sample a ray direction from the goniometric distribution
//   PDF_Le  : distrib.pdf(uv) / (4π)
//
// Design rules (same as punctual_lights.h / cloud_medium.h):
//   - Header-only, CPU_GPU tagged, templated on scalar type T
//   - No virtual functions, no heap allocation in hot paths
//   - T = double on CPU, float on GPU
//   - World→light rotation stored as row-major 3×3 matrix
//   - Goniometric image is a flat vector<double> (nu×nv, row-major)
//     supplied at construction time; the PiecewiseConstant2D sampling
//     distribution is built from it in the constructor
//
// Public interface:
//   GoniometricLight::make(pos, world_to_light_mat, ir,ig,ib, scale,
//                          image_data, nu, nv)
//   T   eval_I(wx,wy,wz)          -- directional intensity in light space
//   void sample_li(px,py,pz,      -- Li and wi toward shading point
//                  &wr,&wg,&wb, &wi_x,&wi_y,&wi_z)
//   T   pdf_li()                  -- always 0 (delta light)
//   T   power()                   -- total emitted power estimate
//   void sample_le(ru,rv,         -- importance-sample an emitted ray direction
//                  &wx,&wy,&wz, &pdf_dir)
//   T   pdf_le(wx,wy,wz)          -- PDF of emitted ray in given direction
//
// Reference: pbrt-v4 GoniometricLight (lights.h/lights.cpp), §12.4
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU
#   endif
#endif

#include <cmath>
#include <vector>
#include <algorithm>
#include "sampling.h"       // EqualAreaSphereToSquare, EqualAreaSquareToSphere
#include "piecewise_dist.h" // PiecewiseConstant2D

// ---------------------------------------------------------------------------
// GoniometricLight<T>
// ---------------------------------------------------------------------------
template<typename T>
struct GoniometricLight {
	// World-space position of the light.
	T pos_x, pos_y, pos_z;

	// Affine world→light rotation (3×3 row-major, no translation needed
	// since we only rotate directions, not positions).
	T world_to_light[9];

	// Base spectral intensity (RGB, candela).
	T ir, ig, ib;

	// Additional scale factor (matches pbrt-v4 "scale" parameter).
	T scale;

	// Goniometric image dimensions.
	int nu, nv;

	// Goniometric image data (nu×nv, row-major).
	// image[v*nu + u] = relative intensity in direction
	//   EqualAreaSquareToSphere(u/nu, v/nv).
	std::vector<double> image;

	// Importance-sampling distribution built from the image.
	// Used for sample_le / pdf_le (bidirectional).
	PiecewiseConstant2D distrib;

	// -----------------------------------------------------------------------
	// Factory
	// -----------------------------------------------------------------------
	// NOTE: pbrt-v4 requires a square goniometric image (nu == nv) so that
	// the equal-area sphere↔square mapping is undistorted. Providing a
	// non-square image will silently stretch the directional profile.
	static GoniometricLight make(
			T px, T py, T pz,
			const T* world_to_light_mat,   // row-major 3×3
			T ir_, T ig_, T ib_,
			T scale_,
			const std::vector<double>& image_data,
			int nu_, int nv_) {
		GoniometricLight g;
		g.pos_x = px; g.pos_y = py; g.pos_z = pz;
		for (int i = 0; i < 9; ++i) g.world_to_light[i] = world_to_light_mat[i];
		g.ir = ir_; g.ig = ig_; g.ib = ib_;
		g.scale = scale_;
		g.nu = nu_; g.nv = nv_;
		g.image = image_data;
		// Build sampling distribution from image (mirrors pbrt-v4 constructor)
		g.distrib = PiecewiseConstant2D(image_data, nu_, nv_);
		return g;
	}

	// Convenience: isotropic (all-ones image), identity rotation.
	static GoniometricLight make_isotropic(
			T px, T py, T pz,
			T ir_, T ig_, T ib_, T scale_) {
		T id[9] = {T(1),T(0),T(0), T(0),T(1),T(0), T(0),T(0),T(1)};
		std::vector<double> img(4*4, 1.0);  // 4×4 uniform
		return make(px, py, pz, id, ir_, ig_, ib_, scale_, img, 4, 4);
	}

	// -----------------------------------------------------------------------
	// world_to_light_dir: rotate a world-space direction to light space.
	// Only applies the 3×3 rotation (no translation for directions).
	// -----------------------------------------------------------------------
	CPU_GPU void to_light_dir(T wx, T wy, T wz,
							  T& lx, T& ly, T& lz) const {
		lx = world_to_light[0]*wx + world_to_light[1]*wy + world_to_light[2]*wz;
		ly = world_to_light[3]*wx + world_to_light[4]*wy + world_to_light[5]*wz;
		lz = world_to_light[6]*wx + world_to_light[7]*wy + world_to_light[8]*wz;
	}

	// -----------------------------------------------------------------------
	// image_lookup: bilinear lookup into the goniometric image at (u,v) in [0,1]^2.
	// Mirrors pbrt-v4 image.LookupNearestChannel(uv, 0).
	// We use nearest-neighbour for simplicity (matches pbrt-v4 default).
	// -----------------------------------------------------------------------
	CPU_GPU double image_lookup(double u, double v) const {
		if (image.empty()) return 1.0;
		// Clamp to valid range
		int iu = (int)(u * nu);
		int iv = (int)(v * nv);
		if (iu < 0) iu = 0; if (iu >= nu) iu = nu - 1;
		if (iv < 0) iv = 0; if (iv >= nv) iv = nv - 1;
		return image[iv * nu + iu];
	}

	// -----------------------------------------------------------------------
	// eval_I: directional intensity at a world-space direction (wx,wy,wz).
	// Mirrors pbrt-v4 GoniometricLight::I(w, lambda).
	// Returns a scalar multiplier; caller multiplies by (ir,ig,ib)*scale.
	// -----------------------------------------------------------------------
	CPU_GPU double eval_I(T wx, T wy, T wz) const {
		// Transform direction to light space
		T lx, ly, lz;
		to_light_dir(wx, wy, wz, lx, ly, lz);
		// Map light-space direction to equal-area square UV
		double u, v;
		EqualAreaSphereToSquare((double)lx, (double)ly, (double)lz, u, v);
		return image_lookup(u, v);
	}

	// -----------------------------------------------------------------------
	// sample_li: sample incident radiance at shading point (px,py,pz).
	// Mirrors pbrt-v4 GoniometricLight::SampleLi.
	// Delta light: wi is always the direction from p toward the light.
	// -----------------------------------------------------------------------
	CPU_GPU void sample_li(T px, T py, T pz,
						   T& Lr, T& Lg, T& Lb,
						   T& wi_x, T& wi_y, T& wi_z) const {
		// Direction from shading point to light
		T dx = pos_x - px;
		T dy = pos_y - py;
		T dz = pos_z - pz;
		T r2 = dx*dx + dy*dy + dz*dz;
		if (r2 < T(1e-20)) { Lr = Lg = Lb = T(0); wi_x = T(0); wi_y = T(1); wi_z = T(0); return; }
		T inv_r = T(1) / std::sqrt(r2);
		wi_x = dx * inv_r;
		wi_y = dy * inv_r;
		wi_z = dz * inv_r;

		// Evaluate directional intensity in the incident direction
		// pbrt-v4: I(renderFromLight.ApplyInverse(-wi), lambda) / DistanceSquared
		// We use -wi (direction from light toward shading point) to look up intensity.
		double gonio = eval_I(-wi_x, -wi_y, -wi_z);

		T weight = (T)(scale * gonio) / r2;
		Lr = ir * weight;
		Lg = ig * weight;
		Lb = ib * weight;
	}

	// -----------------------------------------------------------------------
	// pdf_li: PDF for SampleLi. Always 0 for a delta-position light.
	// -----------------------------------------------------------------------
	CPU_GPU T pdf_li() const { return T(0); }

	// -----------------------------------------------------------------------
	// power: approximate total emitted power (solid-angle integral over sphere).
	// Mirrors pbrt-v4 GoniometricLight::Phi:
	//   Phi = scale * Iemit * 4π * sum(image) / (nu*nv)
	//       = scale * Iemit * 4π * avg(image)
	// For our scalar RGB representation we use Iavg = (ir+ig+ib)/3 as a
	// luminance proxy, matching the spirit of pbrt-v4's per-wavelength integral.
	// -----------------------------------------------------------------------
	CPU_GPU T power() const {
		if (image.empty()) return T(0);
		double sum = 0.0;
		for (double val : image) sum += val;
		double avg_image = sum / (double)(nu * nv);
		// Scalar luminance proxy for Iemit (pbrt-v4 uses per-wavelength DenselySampledSpectrum)
		double Iavg = ((double)ir + (double)ig + (double)ib) / 3.0;
		const double pi4 = 4.0 * 3.14159265358979323846;
		return (T)(scale * Iavg * pi4 * avg_image);
	}

	// -----------------------------------------------------------------------
	// sample_le: importance-sample an emitted ray direction from the
	// goniometric distribution.
	// Mirrors pbrt-v4 GoniometricLight::SampleLe.
	// ru, rv: two uniform random numbers in [0,1].
	// Returns: light-space direction (wx,wy,wz) and pdf_dir.
	// Note: direction is in *light space* — caller transforms to world if needed.
	// -----------------------------------------------------------------------
	void sample_le(double ru, double rv,
				   double& wx, double& wy, double& wz,
				   double& pdf_dir) const {
		if (distrib.empty()) {
			// Uniform sphere sampling fallback
			double u = ru, v = rv;
			EqualAreaSquareToSphere(u, v, wx, wy, wz);
			pdf_dir = 1.0 / (4.0 * 3.14159265358979323846);
			return;
		}
		double pdf_image;
		auto [u, v] = distrib.sample(ru, rv, &pdf_image);
		EqualAreaSquareToSphere(u, v, wx, wy, wz);
		// pdf_dir = pdf_image / (4π)  (Jacobian from image→sphere)
		pdf_dir = pdf_image / (4.0 * 3.14159265358979323846);
	}

	// -----------------------------------------------------------------------
	// pdf_le: PDF of emitting in a given light-space direction (wx,wy,wz).
	// Mirrors pbrt-v4 GoniometricLight::PDF_Le.
	// -----------------------------------------------------------------------
	double pdf_le(double wx, double wy, double wz) const {
		if (distrib.empty()) return 1.0 / (4.0 * 3.14159265358979323846);
		double u, v;
		EqualAreaSphereToSquare(wx, wy, wz, u, v);
		return distrib.pdf(u, v) / (4.0 * 3.14159265358979323846);
	}
};
