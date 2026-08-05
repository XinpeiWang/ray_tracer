#pragma once
// ---------------------------------------------------------------------------
// projection_light.h -- Perspective-frustum projection (slide-projector) light
//
// Mirrors pbrt-v4 ProjectionLight (lights.h / lights.cpp), section 12.5.
//
// Algorithm:
//   I(w_light): if wz < hither -> 0; perspective-project to screen; if
//               outside screenBounds -> 0; map to [0,1]^2 UV;
//               return scale * image[UV].
//   SampleLi:   delta-position  Li = I(w_light) / r^2.
//   Power:      scale * A * sum(rgb_avg * cos^3(theta)) / (nx*ny).
//   SampleLe / PDF_Le: PiecewiseConstant2D weighted by cos^3(theta).
//
// Design: header-only, CPU_GPU tagged, template<typename T>.
//   T = double on CPU, float on GPU.
//   Image stored as flat vector<double> with 3 doubles/pixel (RGB), row-major.
//
// Reference: pbrt-v4 ProjectionLight (lights.h/lights.cpp), section 12.5.
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#  if defined(__CUDACC__)
#    define CPU_GPU __host__ __device__ __forceinline__
#  else
#    define CPU_GPU
#  endif
#endif

#include <cmath>
#include <vector>
#include <algorithm>
#include "cameras.h"         // Mat4<T>, make_perspective<T>
#include "piecewise_dist.h"  // PiecewiseConstant2D

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// ProjectionLight<T>
// ---------------------------------------------------------------------------
template<typename T>
struct ProjectionLight {

	// World-space position.
	T pos_x, pos_y, pos_z;

	// 3x3 world->light rotation (row-major, directions only).
	T world_to_light[9];

	// Intensity scale factor (pbrt-v4 "scale").
	T scale;

	// Near-plane (hither) distance; directions with wz < hither are rejected.
	// pbrt-v4 default: 1e-3.
	T hither;

	// Image dimensions.
	int nx, ny;

	// Flat RGB image: image_rgb[(v*nx + u)*3 + c] for channel c in {0,1,2}.
	std::vector<double> image_rgb;

	// Screen-space bounding box (mirrors pbrt-v4 screenBounds).
	//   aspect >= 1: [-aspect, aspect] x [-1, 1]
	//   aspect <  1: [-1, 1]           x [-1/aspect, 1/aspect]
	T sb_xmin, sb_xmax, sb_ymin, sb_ymax;

	// 4x4 perspective projection matrices (cameras.h Mat4<T>).
	//   screenFromLight: maps light-space direction -> screen-space point
	//   lightFromScreen: inverse
	Mat4<T> screenFromLight;
	Mat4<T> lightFromScreen;

	// Solid-angle area of the projection frustum at unit distance.
	// A = 4 * tan^2(fov/2) * max(aspect, 1/aspect)
	// Mirrors pbrt-v4: A = 4 * Sqr(opposite) * (aspect>1 ? aspect : 1/aspect)
	T A;

	// Importance-sampling distribution over screen pixels.
	// Weight[v*nx+u] = max_channel(pixel) * cos^3(theta_pixel).
	PiecewiseConstant2D distrib;

	// -----------------------------------------------------------------------
	// Factory -- mirrors pbrt-v4 ProjectionLight constructor.
	// px,py,pz          : world-space light position
	// wtl               : row-major 3x3 world->light rotation
	// scale_            : intensity scale
	// fov_deg           : full vertical field of view in degrees
	// image_data        : nx_ * ny_ * 3 doubles (R,G,B per pixel, row-major)
	// nx_, ny_          : image width and height
	// -----------------------------------------------------------------------
	static ProjectionLight make(
			T px, T py, T pz,
			const T* wtl,
			T scale_,
			T fov_deg,
			const std::vector<double>& image_data,
			int nx_, int ny_) {

		ProjectionLight pl;
		pl.pos_x = px; pl.pos_y = py; pl.pos_z = pz;
		for (int i = 0; i < 9; ++i) pl.world_to_light[i] = wtl[i];
		pl.scale     = scale_;
		pl.hither    = T(1e-3);
		pl.nx        = nx_;
		pl.ny        = ny_;
		pl.image_rgb = image_data;

		// Screen bounds (pbrt-v4 constructor, lines 292-296)
		T aspect = T(nx_) / T(ny_);
		if (aspect >= T(1)) {
			pl.sb_xmin = -aspect; pl.sb_xmax =  aspect;
			pl.sb_ymin = T(-1);   pl.sb_ymax =  T(1);
		} else {
			pl.sb_xmin = T(-1);           pl.sb_xmax = T(1);
			pl.sb_ymin = -T(1) / aspect;  pl.sb_ymax =  T(1) / aspect;
		}

		// Projection matrices (pbrt-v4 lines 297-298)
		pl.screenFromLight = make_perspective<T>(fov_deg, T(1e-3), T(1e30));
		pl.lightFromScreen = pl.screenFromLight.inverse();

		// Solid-angle area (pbrt-v4 lines 300-302)
		T half_rad = T(M_PI / 180.0) * fov_deg / T(2);
		T opp = std::tan(half_rad);
		T af  = (aspect >= T(1)) ? aspect : T(1) / aspect;
		pl.A  = T(4) * opp * opp * af;

		// Per-pixel importance weights = max_channel * cos^3(theta)
		// Mirrors pbrt-v4 dwdA lambda (lines 304-309)
		std::vector<double> w(nx_ * ny_, 0.0);
		for (int v = 0; v < ny_; ++v) {
			for (int u = 0; u < nx_; ++u) {
				double su = (double)pl.sb_xmin
						  + ((double)pl.sb_xmax - (double)pl.sb_xmin)
						  * ((u + 0.5) / nx_);
				double sv = (double)pl.sb_ymin
						  + ((double)pl.sb_ymax - (double)pl.sb_ymin)
						  * ((v + 0.5) / ny_);
				auto ld = pl.lightFromScreen.transform_point(
							  (T)su, (T)sv, T(0));
				double len = std::sqrt(
					(double)(ld.x*ld.x + ld.y*ld.y + ld.z*ld.z));
				if (len < 1e-12) continue;
				double wz_n = (double)ld.z / len;
				double cos3 = wz_n * wz_n * wz_n;
				if (cos3 <= 0.0) continue;
				int idx = (v * nx_ + u) * 3;
				double lum = 0.0;
				if (!image_data.empty() && idx + 2 < (int)image_data.size())
					lum = std::max({image_data[idx],
									image_data[idx + 1],
									image_data[idx + 2]});
				w[v * nx_ + u] = lum * cos3;
			}
		}
		pl.distrib = PiecewiseConstant2D(w, nx_, ny_);
		return pl;
	}

	// Uniform-white image convenience factory (all pixels = 1).
	static ProjectionLight make_uniform(
			T px, T py, T pz,
			const T* wtl,
			T scale_, T fov_deg,
			int nx_ = 4, int ny_ = 4) {
		std::vector<double> img(nx_ * ny_ * 3, 1.0);
		return make(px, py, pz, wtl, scale_, fov_deg, img, nx_, ny_);
	}

	// -----------------------------------------------------------------------
	// world_to_light_dir: rotate a world-space direction into light space.
	// -----------------------------------------------------------------------
	CPU_GPU void world_to_light_dir(T wx, T wy, T wz,
									 T& lx, T& ly, T& lz) const {
		lx = world_to_light[0]*wx + world_to_light[1]*wy + world_to_light[2]*wz;
		ly = world_to_light[3]*wx + world_to_light[4]*wy + world_to_light[5]*wz;
		lz = world_to_light[6]*wx + world_to_light[7]*wy + world_to_light[8]*wz;
	}

	// -----------------------------------------------------------------------
	// image_lookup_rgb: nearest-neighbour lookup, UV in [0,1]^2.
	// Mirrors pbrt-v4 Image::LookupNearestChannel.
	// -----------------------------------------------------------------------
	CPU_GPU void image_lookup_rgb(double u, double v,
								   double& r, double& g, double& b) const {
		if (image_rgb.empty()) { r = g = b = 0.0; return; }
		int iu = (int)(u * nx); iu = std::max(0, std::min(iu, nx - 1));
		int iv = (int)(v * ny); iv = std::max(0, std::min(iv, ny - 1));
		int idx = (iv * nx + iu) * 3;
		r = std::max(0.0, image_rgb[idx + 0]);
		g = std::max(0.0, image_rgb[idx + 1]);
		b = std::max(0.0, image_rgb[idx + 2]);
	}

	// -----------------------------------------------------------------------
	// eval_I_rgb: directional intensity in light-space direction (per channel).
	// Mirrors pbrt-v4 ProjectionLight::I().
	// -----------------------------------------------------------------------
	CPU_GPU void eval_I_rgb(T wx, T wy, T wz,
							 double& Lr, double& Lg, double& Lb) const {
		Lr = Lg = Lb = 0.0;

		// Step 1: reject directions behind hither plane
		if (wz < hither) return;

		// Step 2: perspective-project to screen space
		auto ps = screenFromLight.transform_point(wx, wy, wz);

		// Step 3: clip to screen bounds
		if ((double)ps.x < (double)sb_xmin || (double)ps.x > (double)sb_xmax ||
			(double)ps.y < (double)sb_ymin || (double)ps.y > (double)sb_ymax)
			return;

		// Step 4: map to [0,1]^2 UV
		double u = ((double)ps.x - (double)sb_xmin)
				 / ((double)sb_xmax - (double)sb_xmin);
		double v = ((double)ps.y - (double)sb_ymin)
				 / ((double)sb_ymax - (double)sb_ymin);

		// Step 5: nearest-neighbour lookup and scale
		double r, g, b;
		image_lookup_rgb(u, v, r, g, b);
		Lr = (double)scale * r;
		Lg = (double)scale * g;
		Lb = (double)scale * b;
	}

	// eval_I: scalar (RGB average) intensity.
	CPU_GPU T eval_I(T wx, T wy, T wz) const {
		double r, g, b;
		eval_I_rgb(wx, wy, wz, r, g, b);
		return (T)((r + g + b) / 3.0);
	}

	// -----------------------------------------------------------------------
	// sample_li: compute incident radiance at shading point (px,py,pz).
	// Mirrors pbrt-v4 ProjectionLight::SampleLi.
	// Delta light: wi always points from p toward the light position.
	// -----------------------------------------------------------------------
	CPU_GPU void sample_li(T px, T py, T pz,
							T& Lr, T& Lg, T& Lb,
							T& wi_x, T& wi_y, T& wi_z) const {
		T dx = pos_x - px, dy = pos_y - py, dz = pos_z - pz;
		T r2 = dx*dx + dy*dy + dz*dz;
		if (r2 < T(1e-20)) {
			Lr = Lg = Lb = T(0);
			wi_x = T(0); wi_y = T(1); wi_z = T(0);
			return;
		}
		T inv_r = T(1) / std::sqrt(r2);
		wi_x = dx * inv_r; wi_y = dy * inv_r; wi_z = dz * inv_r;

		// Direction from light toward shading point in light space
		T lx, ly, lz;
		world_to_light_dir(-wi_x, -wi_y, -wi_z, lx, ly, lz);

		double r, g, b;
		eval_I_rgb(lx, ly, lz, r, g, b);

		double inv_r2 = 1.0 / (double)r2;
		Lr = (T)(r * inv_r2);
		Lg = (T)(g * inv_r2);
		Lb = (T)(b * inv_r2);
	}

	// PDF_Li: always 0 for a delta-position light.
	CPU_GPU T pdf_li() const { return T(0); }

	// -----------------------------------------------------------------------
	// power: total emitted power estimate.
	// Mirrors pbrt-v4 ProjectionLight::Phi:
	//   Phi = scale * A * sum(rgb_avg[x,y] * cos^3(theta_xy)) / (nx*ny)
	// -----------------------------------------------------------------------
	T power() const {
		if (image_rgb.empty()) return T(0);
		double sum = 0.0;
		for (int v = 0; v < ny; ++v) {
			for (int u = 0; u < nx; ++u) {
				double su = (double)sb_xmin
						  + ((double)sb_xmax - (double)sb_xmin)
						  * ((u + 0.5) / nx);
				double sv = (double)sb_ymin
						  + ((double)sb_ymax - (double)sb_ymin)
						  * ((v + 0.5) / ny);
				auto ld = lightFromScreen.transform_point(
							  (T)su, (T)sv, T(0));
				double len = std::sqrt(
					(double)(ld.x*ld.x + ld.y*ld.y + ld.z*ld.z));
				if (len < 1e-12) continue;
				double wz_n = (double)ld.z / len;
				double cos3 = wz_n * wz_n * wz_n;
				int idx = (v * nx + u) * 3;
				double avg = (image_rgb[idx] + image_rgb[idx+1]
							+ image_rgb[idx+2]) / 3.0;
				sum += avg * cos3;
			}
		}
		return (T)((double)scale * (double)A * sum / (double)(nx * ny));
	}

	// -----------------------------------------------------------------------
	// sample_le: importance-sample an emitted ray direction.
	// Mirrors pbrt-v4 ProjectionLight::SampleLe.
	// ru,rv: uniform randoms in [0,1].
	// Returns light-space direction (wx,wy,wz) normalised, and pdf_dir.
	// -----------------------------------------------------------------------
	void sample_le(double ru, double rv,
				   double& wx, double& wy, double& wz,
				   double& pdf_dir) const {
		double u, v, pdf_img;
		if (distrib.empty()) {
			u = ru; v = rv; pdf_img = 1.0;
		} else {
			auto [su, sv] = distrib.sample(ru, rv, &pdf_img);
			u = su; v = sv;
		}

		// Map [0,1]^2 UV to screen-space coordinates
		double ssx = (double)sb_xmin
				   + ((double)sb_xmax - (double)sb_xmin) * u;
		double ssy = (double)sb_ymin
				   + ((double)sb_ymax - (double)sb_ymin) * v;

		// Map screen point to light-space direction
		auto ld = lightFromScreen.transform_point((T)ssx, (T)ssy, T(0));
		double len = std::sqrt(
			(double)(ld.x*ld.x + ld.y*ld.y + ld.z*ld.z));
		if (len < 1e-12) {
			wx = wy = 0.0; wz = 1.0; pdf_dir = 0.0; return;
		}
		wx = (double)ld.x / len;
		wy = (double)ld.y / len;
		wz = (double)ld.z / len;

		// Jacobian: pdfDir = pdf_uv / (A * cos^3(theta))
		// Our PiecewiseConstant2D returns pdf_uv (per [0,1]^2 area).
		// pbrt-v4 uses pdf_screen (per screen area): pdf_screen = pdf_uv / sb_area.
		// pbrt-v4 formula: pdfDir = pdf_screen * sb_area / (A * cos3)
		//                         = pdf_uv / (A * cos3)  <-- what we compute.
		double cos3 = wz * wz * wz;
		pdf_dir = (cos3 > 1e-12)
				? pdf_img / ((double)A * cos3)
				: 0.0;
	}

	// -----------------------------------------------------------------------
	// pdf_le: PDF of emitting in given light-space direction.
	// Mirrors pbrt-v4 ProjectionLight::PDF_Le.
	// -----------------------------------------------------------------------
	double pdf_le(double wx, double wy, double wz) const {
		// Normalise input
		double len = std::sqrt(wx*wx + wy*wy + wz*wz);
		if (len < 1e-12) return 0.0;
		double wx_n = wx / len, wy_n = wy / len, wz_n = wz / len;

		if (wz_n < (double)hither) return 0.0;

		auto ps = screenFromLight.transform_point(
					  (T)wx_n, (T)wy_n, (T)wz_n);
		if ((double)ps.x < (double)sb_xmin || (double)ps.x > (double)sb_xmax ||
			(double)ps.y < (double)sb_ymin || (double)ps.y > (double)sb_ymax)
			return 0.0;

		double u = ((double)ps.x - (double)sb_xmin)
				 / ((double)sb_xmax - (double)sb_xmin);
		double v = ((double)ps.y - (double)sb_ymin)
				 / ((double)sb_ymax - (double)sb_ymin);

		// Same normalisation as sample_le: pdf_uv / (A * cos3).
		double pdf_img = distrib.empty() ? 1.0 : distrib.pdf(u, v);
		double cos3 = wz_n * wz_n * wz_n;
		return (cos3 > 1e-12)
			 ? pdf_img / ((double)A * cos3)
			 : 0.0;
	}
};
