#pragma once
// ---------------------------------------------------------------------------
// portal_image_infinite_light.h  --  PortalImageInfiniteLightData<T>
//
// Self-contained CPU port of pbrt-v4 PortalImageInfiniteLight
// (src/pbrt/lights.h / lights.cpp).
//
// The portal constrains sampling to the directions visible through a planar
// quad window, using a rectified tangent-space image + WindowedPiecewiseConstant2D.
//
// pbrt-v4 reference:
//   lights.h  PortalImageInfiniteLight (lines 630-735)
//   lights.cpp PortalImageInfiniteLight ctor / SampleLi / PDF_Li / SampleLe / PDF_Le
//
// Usage:
//   // Provide an RGB row-major image (equal-area sphere map, WxH*3 floats)
//   // and 4 corner points for the portal quad (planar rectangle).
//   PortalImageInfiniteLightData<double> light(rgb, W, H, scale, portal4);
//   // Sample a direction from shading point p:
//   Vec3d wi; double pdf;
//   light.sample_li(ru, rv, px, py, pz, wx, wy, wz, pdf);
//   // PDF for a known direction:
//   double p = light.pdf_li(px, py, pz, wx, wy, wz);
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU
#   endif
#endif

#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <optional>
#include <cassert>
#include <numeric>

#include "summed_area_table.h"   // SummedAreaTable, WindowedPiecewiseConstant2D, Array2D, SAT_Bounds2f, SAT_Point2f
#include "sampling.h"             // EqualAreaSphereToSquare, EqualAreaSquareToSphere

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace pil_detail {

// 3-component vector and helpers
template<typename T>
struct Vec3 { T x, y, z; };

template<typename T>
CPU_GPU inline T dot(Vec3<T> a, Vec3<T> b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

template<typename T>
CPU_GPU inline Vec3<T> cross(Vec3<T> a, Vec3<T> b) {
	return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
}

template<typename T>
CPU_GPU inline Vec3<T> normalize(Vec3<T> v) {
	T len = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
	if (len == T(0)) return v;
	return { v.x/len, v.y/len, v.z/len };
}

template<typename T>
CPU_GPU inline Vec3<T> sub(Vec3<T> a, Vec3<T> b) { return { a.x-b.x, a.y-b.y, a.z-b.z }; }

template<typename T>
CPU_GPU inline T length(Vec3<T> v) { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }

// Frame: orthonormal frame with x, y, z axes (matches pbrt-v4 Frame).
// portalFrame = Frame::FromXY(p03, p01) where p03 = portal[3]-portal[0], p01 = portal[1]-portal[0].
template<typename T>
struct Frame {
	Vec3<T> x, y, z;

	// Build from two basis vectors (Gram-Schmidt). Matches pbrt-v4 Frame::FromXY.
	static Frame FromXY(Vec3<T> vx, Vec3<T> vy) {
		Vec3<T> nx = normalize(vx);
		Vec3<T> nz = normalize(cross(nx, vy));
		Vec3<T> ny = cross(nz, nx);
		return { nx, ny, nz };
	}

	// Transform a render-space vector to frame-local space.
	CPU_GPU Vec3<T> ToLocal(Vec3<T> v) const {
		return { dot(v, x), dot(v, y), dot(v, z) };
	}

	// Transform a frame-local vector to render space.
	CPU_GPU Vec3<T> FromLocal(Vec3<T> v) const {
		return { x.x*v.x + y.x*v.y + z.x*v.z,
				 x.y*v.x + y.y*v.y + z.y*v.z,
				 x.z*v.x + y.z*v.y + z.z*v.z };
	}
};

// Bilinear nearest-neighbour lookup (rectified image stored row-major RGB).
// Mirrors pbrt-v4 image.LookupNearestChannel.
inline void lookup_nearest(const std::vector<float>& img, int nx, int ny,
							float u, float v,
							float& r, float& g, float& b) {
	int ix = std::min((int)(u * nx), nx - 1);
	int iy = std::min((int)(v * ny), ny - 1);
	int idx = (iy * nx + ix) * 3;
	r = img[idx]; g = img[idx+1]; b = img[idx+2];
}

CPU_GPU inline double luminance(double r, double g, double b) {
	return 0.2126*r + 0.7152*g + 0.0722*b;
}

static constexpr double PI = 3.14159265358979323846;

} // namespace pil_detail

// ---------------------------------------------------------------------------
// PortalImageInfiniteLightData<T>
// ---------------------------------------------------------------------------
template<typename T>
struct PortalImageInfiniteLightData {
	using Vec3T = pil_detail::Vec3<T>;
	using FrameT = pil_detail::Frame<T>;

	// portal[4]: four corners of the portal quad (planar rectangle in render space),
	//            ordered: p0, p1 (= p0+right), p2 (= p0+right+up), p3 (= p0+up).
	//            Matches pbrt-v4 portal[] ordering.
	// rgb:       pointer to H*W*3 row-major equal-area sphere map (R,G,B floats).
	// width,height: image dimensions.
	// scale:     radiance scale factor.
	PortalImageInfiniteLightData(const float* rgb_eq, int width, int height, T scale,
								 const std::array<Vec3T, 4>& portal_corners)
		: width_(width), height_(height), scale_(T(scale))
	{
		for (int i = 0; i < 4; ++i) portal_[i] = portal_corners[i];

		// Build portal frame: FromXY(p03, p01).
		// p01 = normalize(portal[1] - portal[0]), p03 = normalize(portal[3] - portal[0])
		Vec3T p01 = pil_detail::normalize(pil_detail::sub(portal_[1], portal_[0]));
		Vec3T p03 = pil_detail::normalize(pil_detail::sub(portal_[3], portal_[0]));
		portalFrame_ = FrameT::FromXY(p03, p01);

		// Build rectified image: for each pixel (x,y), compute uv in tangent space,
		// map to render-space direction, then to equal-area uv, bilinear sample.
		// Mirrors pbrt-v4 ctor "Resample environment map into rectified image" block.
		// Note: pbrt-v4 also applies renderFromLight.ApplyInverse(w) before the
		// equal-area lookup to account for the light's world transform. This local
		// port assumes the portal is already in render space (identity transform).
		rectified_.resize(width * height * 3);
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				float u = (x + 0.5f) / width;
				float v = (y + 0.5f) / height;

				// RenderFromImage(uv) -- tangent-space uv -> render-space direction
				Vec3T wRender = RenderFromImage_uv((T)u, (T)v);

				// Convert to equal-area uv
				double eu, ev;
				EqualAreaSphereToSquare((double)wRender.x, (double)wRender.y, (double)wRender.z, eu, ev);

				// Bilinear sample from equal-area source image
				float r, g, b;
				bilinear_sample_eq(rgb_eq, width, height, (float)eu, (float)ev, r, g, b);

				int idx = (y * width + x) * 3;
				rectified_[idx  ] = r;
				rectified_[idx+1] = g;
				rectified_[idx+2] = b;
			}
		}

		// Build sampling distribution weighted by channel average * jacobian (duv_dw).
		// Mirrors pbrt-v4 GetSamplingDistribution(duv_dw):
		//   value = GetChannels({x,y}).Average()  = (R+G+B)/3  (NOT luminance)
		//   dist(x,y) = value * duv_dw(uv)
		std::vector<float> lum(width * height);
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				float u = (x + 0.5f) / width;
				float v = (y + 0.5f) / height;
				T duv_dw;
				RenderFromImage_uv((T)u, (T)v, &duv_dw);
				int idx = (y * width + x) * 3;
				// pbrt-v4 uses channel average (R+G+B)/3, not luminance
				float avg = (rectified_[idx] + rectified_[idx+1] + rectified_[idx+2]) / 3.f;
				// Weight by Jacobian as pbrt-v4 does in GetSamplingDistribution(duv_dw)
				lum[y * width + x] = (duv_dw > T(0)) ? avg * (float)duv_dw : 0.f;
			}
		}

		Array2D<float> dist_arr(width, height, lum);
		distribution_ = WindowedPiecewiseConstant2D(dist_arr);

		// Portal area
		area_ = (T)pil_detail::length(pil_detail::sub(portal_[1], portal_[0]))
			  * (T)pil_detail::length(pil_detail::sub(portal_[3], portal_[0]));
	}

	// eval_Le: evaluate emitted radiance for a render-space ray direction.
	// Returns luminance (Y channel). Mirrors pbrt-v4 Le().
	// Pass the ray origin and direction; returns 0 if the direction is outside
	// the portal's visible angular window from that origin.
	CPU_GPU T eval_Le(T ox, T oy, T oz,
					  T dx, T dy, T dz) const {
		Vec3T wRender = pil_detail::normalize(Vec3T{dx,dy,dz});
		T u, v;
		if (!ImageFromRender(wRender, &u, &v)) return T(0);

		// Check portal bounds from origin
		T bMinU, bMinV, bMaxU, bMaxV;
		if (!ImageBounds(ox, oy, oz, bMinU, bMinV, bMaxU, bMaxV)) return T(0);
		if (u < bMinU || u > bMaxU || v < bMinV || v > bMaxV) return T(0);

		float r, g, b;
		pil_detail::lookup_nearest(rectified_, width_, height_, (float)u, (float)v, r, g, b);
		return T(scale_) * (T)pil_detail::luminance(r, g, b);
	}

	// eval_Le_rgb: RGB version of Le.
	CPU_GPU void eval_Le_rgb(T ox, T oy, T oz,
							  T dx, T dy, T dz,
							  T& out_r, T& out_g, T& out_b) const {
		Vec3T wRender = pil_detail::normalize(Vec3T{dx,dy,dz});
		T u, v;
		if (!ImageFromRender(wRender, &u, &v)) { out_r=out_g=out_b=T(0); return; }
		T bMinU, bMinV, bMaxU, bMaxV;
		if (!ImageBounds(ox, oy, oz, bMinU, bMinV, bMaxU, bMaxV)) { out_r=out_g=out_b=T(0); return; }
		if (u < bMinU || u > bMaxU || v < bMinV || v > bMaxV) { out_r=out_g=out_b=T(0); return; }
		float r, g, b;
		pil_detail::lookup_nearest(rectified_, width_, height_, (float)u, (float)v, r, g, b);
		out_r = T(scale_) * T(r);
		out_g = T(scale_) * T(g);
		out_b = T(scale_) * T(b);
	}

	// sample_li: importance-sample a render-space direction from shading point p.
	// Returns false if the portal window from p subtends zero area.
	// Mirrors pbrt-v4 PortalImageInfiniteLight::SampleLi.
	bool sample_li(double ru, double rv,
				   T px, T py, T pz,
				   T& wx, T& wy, T& wz, T& pdf) const {
		T bMinU, bMinV, bMaxU, bMaxV;
		if (!ImageBounds(px, py, pz, bMinU, bMinV, bMaxU, bMaxV)) {
			wx=wy=wz=T(0); pdf=T(0); return false;
		}
		SAT_Bounds2f b(SAT_Point2f((float)bMinU,(float)bMinV),
					   SAT_Point2f((float)bMaxU,(float)bMaxV));
		float mapPDF;
		auto sample = distribution_.Sample(SAT_Point2f((float)ru,(float)rv), b, &mapPDF);
		if (!sample) { wx=wy=wz=T(0); pdf=T(0); return false; }

		T u = (T)sample->x, v = (T)sample->y;
		T duv_dw;
		Vec3T wRender = RenderFromImage_uv(u, v, &duv_dw);
		if (duv_dw == T(0)) { wx=wy=wz=T(0); pdf=T(0); return false; }

		wx = wRender.x; wy = wRender.y; wz = wRender.z;
		pdf = T(mapPDF) / duv_dw;
		return true;
	}

	// pdf_li: solid-angle PDF for a direction. Mirrors pbrt-v4 PDF_Li.
	CPU_GPU T pdf_li(T px, T py, T pz,
					 T dx, T dy, T dz) const {
		Vec3T wRender = pil_detail::normalize(Vec3T{dx,dy,dz});
		T u, v, duv_dw;
		if (!ImageFromRender(wRender, &u, &v, &duv_dw)) return T(0);
		if (duv_dw == T(0)) return T(0);

		T bMinU, bMinV, bMaxU, bMaxV;
		if (!ImageBounds(px, py, pz, bMinU, bMinV, bMaxU, bMaxV)) return T(0);

		SAT_Bounds2f b(SAT_Point2f((float)bMinU,(float)bMinV),
					   SAT_Point2f((float)bMaxU,(float)bMaxV));
		float mapPDF = distribution_.PDF(SAT_Point2f((float)u,(float)v), b);
		return T(mapPDF) / duv_dw;
	}

	// power: approximate total power. Mirrors pbrt-v4 Phi.
	// pbrt-v4 uses luminance (via RGBIlluminantSpectrum) divided by duv_dw,
	// then scales by scale * Area / (nx * ny).
	T power() const {
		double sum = 0.0;
		for (int y = 0; y < height_; ++y) {
			for (int x = 0; x < width_; ++x) {
				float u = (x + 0.5f) / width_;
				float v = (y + 0.5f) / height_;
				T duv_dw;
				RenderFromImage_uv((T)u, (T)v, &duv_dw);
				int idx = (y * width_ + x) * 3;
				// Use luminance as pbrt-v4 uses RGBIlluminantSpectrum.Sample() -> Y
				double lv = pil_detail::luminance(rectified_[idx], rectified_[idx+1], rectified_[idx+2]);
				if (duv_dw > T(0)) sum += lv / (double)duv_dw;
			}
		}
		return T(scale_) * T(area_) * T(sum) / T(width_ * height_);
	}

	// Accessors
	int width()  const { return width_; }
	int height() const { return height_; }
	T   scale()  const { return scale_; }

private:
	// ---------------------------------------------------------------------------
	// ImageFromRender: render-space direction -> image uv in [0,1]^2.
	// Returns false if w.z <= 0 in portal frame (behind portal).
	// Optionally outputs Jacobian duv_dw = d(u,v)/dw.
	// Mirrors pbrt-v4 PortalImageInfiniteLight::ImageFromRender.
	// ---------------------------------------------------------------------------
	CPU_GPU bool ImageFromRender(Vec3T wRender,
								  T* u_out, T* v_out,
								  T* duv_dw = nullptr) const {
		Vec3T w = portalFrame_.ToLocal(wRender);
		if (w.z <= T(0)) return false;
		if (duv_dw) {
			T pi_sq = T(pil_detail::PI * pil_detail::PI);
			*duv_dw = pi_sq * (T(1) - w.x*w.x) * (T(1) - w.y*w.y) / w.z;
		}
		T alpha = std::atan2(w.x, w.z);
		T beta  = std::atan2(w.y, w.z);
		*u_out = std::max(T(0), std::min(T(1), (alpha + T(pil_detail::PI/2)) / T(pil_detail::PI)));
		*v_out = std::max(T(0), std::min(T(1), (beta  + T(pil_detail::PI/2)) / T(pil_detail::PI)));
		return true;
	}

	// ---------------------------------------------------------------------------
	// RenderFromImage: image uv -> render-space direction.
	// Mirrors pbrt-v4 PortalImageInfiniteLight::RenderFromImage.
	// ---------------------------------------------------------------------------
	CPU_GPU Vec3T RenderFromImage_uv(T u, T v, T* duv_dw = nullptr) const {
		T alpha = -T(pil_detail::PI/2) + u * T(pil_detail::PI);
		T beta  = -T(pil_detail::PI/2) + v * T(pil_detail::PI);
		T x = std::tan(alpha), y = std::tan(beta);
		Vec3T w = pil_detail::normalize(Vec3T{x, y, T(1)});
		if (duv_dw) {
			T pi_sq = T(pil_detail::PI * pil_detail::PI);
			*duv_dw = pi_sq * (T(1) - w.x*w.x) * (T(1) - w.y*w.y) / w.z;
		}
		return portalFrame_.FromLocal(w);
	}

	// ---------------------------------------------------------------------------
	// ImageBounds: compute the [uMin,uMax] x [vMin,vMax] of the portal quad
	// as seen from point p. Uses portal[0] and portal[2] (diagonally opposite).
	// Mirrors pbrt-v4 PortalImageInfiniteLight::ImageBounds.
	// ---------------------------------------------------------------------------
	CPU_GPU bool ImageBounds(T px, T py, T pz,
							  T& uMin, T& vMin, T& uMax, T& vMax) const {
		Vec3T p = {px, py, pz};
		Vec3T d0 = pil_detail::normalize(pil_detail::sub(portal_[0], p));
		Vec3T d2 = pil_detail::normalize(pil_detail::sub(portal_[2], p));
		T u0, v0, u2, v2;
		if (!ImageFromRender(d0, &u0, &v0)) return false;
		if (!ImageFromRender(d2, &u2, &v2)) return false;
		uMin = std::min(u0, u2); uMax = std::max(u0, u2);
		vMin = std::min(v0, v2); vMax = std::max(v0, v2);
		return true;
	}

	// Equal-area bilinear sample (mirrors pbrt-v4 BilerpChannel / OctahedralSphere).
	static void bilinear_sample_eq(const float* src, int nx, int ny,
									float u, float v,
									float& r, float& g, float& b) {
		float pu = u * nx - 0.5f, pv = v * ny - 0.5f;
		int x0 = (int)std::floor(pu), y0 = (int)std::floor(pv);
		float tx = pu - x0, ty = pv - y0;
		auto clampxy = [&](int xi, int yi, int c) -> float {
			xi = std::max(0, std::min(xi, nx-1));
			yi = std::max(0, std::min(yi, ny-1));
			return src[(yi * nx + xi) * 3 + c];
		};
		r = (1-tx)*(1-ty)*clampxy(x0,y0,0) + tx*(1-ty)*clampxy(x0+1,y0,0)
		  + (1-tx)*ty*clampxy(x0,y0+1,0)   + tx*ty*clampxy(x0+1,y0+1,0);
		g = (1-tx)*(1-ty)*clampxy(x0,y0,1) + tx*(1-ty)*clampxy(x0+1,y0,1)
		  + (1-tx)*ty*clampxy(x0,y0+1,1)   + tx*ty*clampxy(x0+1,y0+1,1);
		b = (1-tx)*(1-ty)*clampxy(x0,y0,2) + tx*(1-ty)*clampxy(x0+1,y0,2)
		  + (1-tx)*ty*clampxy(x0,y0+1,2)   + tx*ty*clampxy(x0+1,y0+1,2);
	}

	int      width_, height_;
	T        scale_;
	T        area_;
	std::array<Vec3T, 4> portal_;
	FrameT   portalFrame_;
	std::vector<float>   rectified_;       // rectified RGB image, row-major
	WindowedPiecewiseConstant2D distribution_;
};
