#pragma once
// ---------------------------------------------------------------------------
// film.h -- pbrt-v4 Film models (simplified, header-only, CPU, templated)
// Mirrors pbrt-v4 src/pbrt/film.h / film.cpp
//
// Ported film models:
//   1. RGBFilm      -- filter-weighted sample accumulation -> RGB image
//   2. GBufferFilm  -- RGBFilm + AOV buffers (normal, albedo, depth, uv)
//
// Design notes:
//   - Templated on scalar T (float or double) and Filter type.
//   - Receives pre-converted SensorRGB from PixelSensor::ToSensorRGB().
//   - Thread-safe splat accumulation via std::atomic<double>.
//   - No dependency on SampledSpectrum -- caller does spectrum->SensorRGB.
//   - outputRGBFromSensorRGB = colorSpace.RGBFromXYZ * sensor.XYZFromSensorRGB
//     (mirrors pbrt-v4 RGBFilm constructor).
//
// Typical usage:
//   MitchellFilter filt(2.0);
//   RGBFilm<MitchellFilter> film(800, 600, filt, sensor, &sRGB);
//   // per sample:
//   SensorRGB srgb = sensor.ToSensorRGB(L, lambda);
//   film.add_sample(px, py, weight, srgb);
//   // optional light-path splat:
//   film.add_splat(x, y, srgb);
//   // read out:
//   auto [r,g,b] = film.get_pixel_rgb(px, py);
// ---------------------------------------------------------------------------

#include "filter.h"
#include "pixel_sensor.h"
#include "square_matrix.h"
#include "rgb_colorspace.h"

#include <vector>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <array>

// ---------------------------------------------------------------------------
// Internal: atomic double accumulator (mirrors pbrt-v4 AtomicDouble)
// ---------------------------------------------------------------------------
struct AtomicDoubleFilm {
	std::atomic<double> v{0.0};

	void add(double x) {
		double old = v.load(std::memory_order_relaxed);
		double desired;
		do { desired = old + x; }
		while (!v.compare_exchange_weak(old, desired,
										std::memory_order_relaxed,
										std::memory_order_relaxed));
	}

	double load() const { return v.load(std::memory_order_relaxed); }
	void   reset()      { v.store(0.0, std::memory_order_relaxed); }
};

// ---------------------------------------------------------------------------
// FilmRGBPixel -- per-pixel accumulator (mirrors pbrt-v4 RGBFilm::Pixel)
// ---------------------------------------------------------------------------
struct FilmRGBPixel {
	double rgbSum[3]   = {0., 0., 0.};
	double weightSum   = 0.;
	AtomicDoubleFilm rgbSplat[3];  // light-path splat, atomic

	void reset() {
		rgbSum[0] = rgbSum[1] = rgbSum[2] = 0.;
		weightSum = 0.;
		rgbSplat[0].reset(); rgbSplat[1].reset(); rgbSplat[2].reset();
	}
};

// ---------------------------------------------------------------------------
// GBufferPixel -- extends FilmRGBPixel with AOV data
// (mirrors pbrt-v4 GBufferFilm::Pixel)
// ---------------------------------------------------------------------------
struct GBufferPixel : FilmRGBPixel {
	// AOV accumulators
	double gBufWeightSum = 0.;
	float  pSum[3]       = {0.f, 0.f, 0.f};   // position
	float  nSum[3]       = {0.f, 0.f, 0.f};   // geometric normal
	float  nsSum[3]      = {0.f, 0.f, 0.f};   // shading normal
	float  uvSum[2]      = {0.f, 0.f};
	float  dzdxSum       = 0.f;
	float  dzdySum       = 0.f;
	double rgbAlbedoSum[3] = {0., 0., 0.};

	void reset() {
		FilmRGBPixel::reset();
		gBufWeightSum = 0.;
		for (auto& x : pSum)  x = 0.f;
		for (auto& x : nSum)  x = 0.f;
		for (auto& x : nsSum) x = 0.f;
		for (auto& x : uvSum) x = 0.f;
		dzdxSum = dzdySum = 0.f;
		rgbAlbedoSum[0] = rgbAlbedoSum[1] = rgbAlbedoSum[2] = 0.;
	}
};

// ---------------------------------------------------------------------------
// FilmPixelRGB -- lightweight output struct
// ---------------------------------------------------------------------------
struct FilmPixelRGB {
	float r = 0.f, g = 0.f, b = 0.f;
};

// ---------------------------------------------------------------------------
// GBufferSample -- caller-supplied AOV data for GBufferFilm::add_sample
// ---------------------------------------------------------------------------
struct GBufferSample {
	float p[3]    = {0.f, 0.f, 0.f};  // world position
	float n[3]    = {0.f, 0.f, 0.f};  // geometric normal
	float ns[3]   = {0.f, 0.f, 0.f};  // shading normal
	float uv[2]   = {0.f, 0.f};
	float dzdx    = 0.f;
	float dzdy    = 0.f;
	float albedo[3] = {0.f, 0.f, 0.f};
};

// ---------------------------------------------------------------------------
// RGBFilm<Filter>
//
// Mirrors pbrt-v4 RGBFilm: filter-weighted sample accumulation,
// atomic splat, and sensor→output-RGB conversion.
//
// Template parameters:
//   Filter  -- any type with .evaluate(ox,oy)->double and .radius()->double
//              and .integral()->double.  e.g. MitchellFilter, BoxFilter.
// ---------------------------------------------------------------------------
template <typename Filter = MitchellFilter>
class RGBFilm {
public:
	// Construction mirrors pbrt-v4 RGBFilm::RGBFilm():
	//   outputRGBFromSensorRGB = colorSpace.RGBFromXYZ * sensor.XYZFromSensorRGB
	//   filterIntegral = filter.Integral()
	RGBFilm(int res_x, int res_y,
			const Filter&        filter,
			const PixelSensor&   sensor,
			const RGBColorSpace* colorSpace,
			float maxComponentValue = 1e9f)
		: res_x_(res_x), res_y_(res_y),
		  filter_(filter),
		  max_component_value_(maxComponentValue),
		  pixels_(static_cast<size_t>(res_x) * res_y)
	{
		filter_integral_ = static_cast<float>(filter_.integral());

		// outputRGBFromSensorRGB = colorSpace->RGBFromXYZ * sensor.XYZFromSensorRGB
		// Both are SquareMatrix<3>; operator* is row-major matrix multiply.
		output_rgb_from_sensor_ = colorSpace->RGBFromXYZ * sensor.XYZFromSensorRGB;
	}

	int width()  const { return res_x_; }
	int height() const { return res_y_; }

	// -----------------------------------------------------------------------
	// add_sample -- accumulate one camera-path sample at integer pixel (px,py)
	// weight is the filter weight for this sample (caller already evaluated filter).
	// srgb is the sensor-space RGB from PixelSensor::ToSensorRGB().
	// Mirrors pbrt-v4 RGBFilm::AddSample().
	// -----------------------------------------------------------------------
	void add_sample(int px, int py, float weight, const SensorRGB& srgb) {
		assert(px >= 0 && px < res_x_ && py >= 0 && py < res_y_);

		// Optionally clamp (mirrors pbrt-v4 maxComponentValue clamp)
		float r = srgb.r, g = srgb.g, b = srgb.b;
		float m = std::max({r, g, b});
		if (m > max_component_value_) {
			float s = max_component_value_ / m;
			r *= s; g *= s; b *= s;
		}

		FilmRGBPixel& pix = pixel(px, py);
		pix.rgbSum[0] += weight * r;
		pix.rgbSum[1] += weight * g;
		pix.rgbSum[2] += weight * b;
		pix.weightSum += weight;
	}

	// -----------------------------------------------------------------------
	// add_splat -- light-path contribution at continuous position (x,y).
	// Distributes across all pixels within filter radius (atomic, thread-safe).
	// Mirrors pbrt-v4 RGBFilm::AddSplat().
	// -----------------------------------------------------------------------
	void add_splat(float x, float y, const SensorRGB& srgb) {
		float r = srgb.r, g = srgb.g, b = srgb.b;
		float m = std::max({r, g, b});
		if (m > max_component_value_) {
			float s = max_component_value_ / m;
			r *= s; g *= s; b *= s;
		}

		// Compute bounds of affected pixels (mirrors pbrt-v4 splatBounds)
		float px = x + 0.5f;
		float py = y + 0.5f;
		float rad = static_cast<float>(filter_.radius());

		int x0 = std::max(0,       static_cast<int>(std::floor(px - rad)));
		int x1 = std::min(res_x_-1, static_cast<int>(std::floor(px + rad)));
		int y0 = std::max(0,       static_cast<int>(std::floor(py - rad)));
		int y1 = std::min(res_y_-1, static_cast<int>(std::floor(py + rad)));

		for (int iy = y0; iy <= y1; ++iy) {
			for (int ix = x0; ix <= x1; ++ix) {
				// Evaluate filter at offset from continuous sample position
				double wt = filter_.evaluate(
					static_cast<double>(x - ix - 0.5f),
					static_cast<double>(y - iy - 0.5f));
				if (wt == 0.0) continue;
				FilmRGBPixel& pix = pixel(ix, iy);
				pix.rgbSplat[0].add(wt * r);
				pix.rgbSplat[1].add(wt * g);
				pix.rgbSplat[2].add(wt * b);
			}
		}
	}

	// -----------------------------------------------------------------------
	// get_pixel_rgb -- resolve pixel (px,py) to output-space RGB.
	// splat_scale: scale factor applied to splat contributions (default 1).
	// Mirrors pbrt-v4 RGBFilm::GetPixelRGB().
	// -----------------------------------------------------------------------
	FilmPixelRGB get_pixel_rgb(int px, int py, float splat_scale = 1.f) const {
		assert(px >= 0 && px < res_x_ && py >= 0 && py < res_y_);
		const FilmRGBPixel& pix = pixel(px, py);

		double r = pix.rgbSum[0];
		double g = pix.rgbSum[1];
		double b = pix.rgbSum[2];

		// Normalize by filter weight sum
		double ws = pix.weightSum;
		if (ws != 0.0) { r /= ws; g /= ws; b /= ws; }

		// Add splat contribution (mirrors pbrt-v4: splatScale * splat / filterIntegral)
		float inv_fi = (filter_integral_ > 0.f) ? splat_scale / filter_integral_ : 0.f;
		r += inv_fi * pix.rgbSplat[0].load();
		g += inv_fi * pix.rgbSplat[1].load();
		b += inv_fi * pix.rgbSplat[2].load();

		// Apply outputRGBFromSensorRGB (sensor space -> output color space)
		return to_output_rgb(static_cast<float>(r),
							 static_cast<float>(g),
							 static_cast<float>(b));
	}

	// -----------------------------------------------------------------------
	// to_output_rgb -- convert sensor-space (r,g,b) to output color space.
	// Mirrors pbrt-v4 RGBFilm::ToOutputRGB().
	// -----------------------------------------------------------------------
	FilmPixelRGB to_output_rgb(float r, float g, float b) const {
		// output = outputRGBFromSensorRGB * [r, g, b]
		const auto& M = output_rgb_from_sensor_;
		FilmPixelRGB out;
		out.r = static_cast<float>(M[0][0]*r + M[0][1]*g + M[0][2]*b);
		out.g = static_cast<float>(M[1][0]*r + M[1][1]*g + M[1][2]*b);
		out.b = static_cast<float>(M[2][0]*r + M[2][1]*g + M[2][2]*b);
		return out;
	}

	// Reset a single pixel (mirrors pbrt-v4 ResetPixel)
	void reset_pixel(int px, int py) { pixel(px, py).reset(); }

	// Reset all pixels
	void clear() { for (auto& p : pixels_) p.reset(); }

private:
	FilmRGBPixel&       pixel(int x, int y)       { return pixels_[y * res_x_ + x]; }
	const FilmRGBPixel& pixel(int x, int y) const { return pixels_[y * res_x_ + x]; }

	int   res_x_, res_y_;
	Filter filter_;
	float  filter_integral_;
	float  max_component_value_;
	SquareMatrix<3> output_rgb_from_sensor_;
	std::vector<FilmRGBPixel> pixels_;
};

// ---------------------------------------------------------------------------
// GBufferFilm<Filter>
//
// Extends RGBFilm with G-buffer AOV accumulation: position, normals, uv,
// depth gradients, and albedo. Mirrors pbrt-v4 GBufferFilm.
//
// AOVs are written during add_sample() when a GBufferSample is supplied.
// Retrieve via get_pixel_rgb() (same as RGBFilm) and get_pixel_gbuffer().
// ---------------------------------------------------------------------------
template <typename Filter = MitchellFilter>
class GBufferFilm {
public:
	GBufferFilm(int res_x, int res_y,
			const Filter&        filter,
			const PixelSensor&   sensor,
			const RGBColorSpace* colorSpace,
			float maxComponentValue = 1e9f)
		: res_x_(res_x), res_y_(res_y),
		  filter_(filter),
		  max_component_value_(maxComponentValue),
		  pixels_(static_cast<size_t>(res_x) * res_y)
	{
		filter_integral_ = static_cast<float>(filter_.integral());
		output_rgb_from_sensor_ = colorSpace->RGBFromXYZ * sensor.XYZFromSensorRGB;
	}

	int width()  const { return res_x_; }
	int height() const { return res_y_; }

	// -----------------------------------------------------------------------
	// add_sample -- accumulate camera-path sample + optional AOV data.
	// Mirrors pbrt-v4 GBufferFilm::AddSample().
	// -----------------------------------------------------------------------
	void add_sample(int px, int py, float weight, const SensorRGB& srgb,
					const GBufferSample* gbuf = nullptr) {
		assert(px >= 0 && px < res_x_ && py >= 0 && py < res_y_);

		float r = srgb.r, g = srgb.g, b = srgb.b;
		float m = std::max({r, g, b});
		if (m > max_component_value_) {
			float s = max_component_value_ / m;
			r *= s; g *= s; b *= s;
		}

		GBufferPixel& pix = pixel(px, py);
		pix.rgbSum[0] += weight * r;
		pix.rgbSum[1] += weight * g;
		pix.rgbSum[2] += weight * b;
		pix.weightSum += weight;

		if (gbuf) {
			pix.gBufWeightSum += weight;
			for (int c = 0; c < 3; ++c) {
				pix.pSum[c]  += weight * gbuf->p[c];
				pix.nSum[c]  += weight * gbuf->n[c];
				pix.nsSum[c] += weight * gbuf->ns[c];
				pix.rgbAlbedoSum[c] += weight * gbuf->albedo[c];
			}
			pix.uvSum[0]  += weight * gbuf->uv[0];
			pix.uvSum[1]  += weight * gbuf->uv[1];
			pix.dzdxSum   += weight * gbuf->dzdx;
			pix.dzdySum   += weight * gbuf->dzdy;
		}
	}

	// -----------------------------------------------------------------------
	// add_splat -- identical logic to RGBFilm::add_splat.
	// -----------------------------------------------------------------------
	void add_splat(float x, float y, const SensorRGB& srgb) {
		float r = srgb.r, g = srgb.g, b = srgb.b;
		float m = std::max({r, g, b});
		if (m > max_component_value_) {
			float s = max_component_value_ / m;
			r *= s; g *= s; b *= s;
		}

		float px_ = x + 0.5f, py_ = y + 0.5f;
		float rad = static_cast<float>(filter_.radius());
		int x0 = std::max(0,       static_cast<int>(std::floor(px_ - rad)));
		int x1 = std::min(res_x_-1, static_cast<int>(std::floor(px_ + rad)));
		int y0 = std::max(0,       static_cast<int>(std::floor(py_ - rad)));
		int y1 = std::min(res_y_-1, static_cast<int>(std::floor(py_ + rad)));

		for (int iy = y0; iy <= y1; ++iy) {
			for (int ix = x0; ix <= x1; ++ix) {
				double wt = filter_.evaluate(
					static_cast<double>(x - ix - 0.5f),
					static_cast<double>(y - iy - 0.5f));
				if (wt == 0.0) continue;
				GBufferPixel& pix = pixel(ix, iy);
				pix.rgbSplat[0].add(wt * r);
				pix.rgbSplat[1].add(wt * g);
				pix.rgbSplat[2].add(wt * b);
			}
		}
	}

	// -----------------------------------------------------------------------
	// get_pixel_rgb -- matches pbrt-v4 GBufferFilm::GetPixelRGB()
	// -----------------------------------------------------------------------
	FilmPixelRGB get_pixel_rgb(int px, int py, float splat_scale = 1.f) const {
		assert(px >= 0 && px < res_x_ && py >= 0 && py < res_y_);
		const GBufferPixel& pix = pixel(px, py);

		double r = pix.rgbSum[0], g = pix.rgbSum[1], b = pix.rgbSum[2];
		double ws = pix.weightSum;
		if (ws != 0.0) { r /= ws; g /= ws; b /= ws; }

		float inv_fi = (filter_integral_ > 0.f) ? splat_scale / filter_integral_ : 0.f;
		r += inv_fi * pix.rgbSplat[0].load();
		g += inv_fi * pix.rgbSplat[1].load();
		b += inv_fi * pix.rgbSplat[2].load();

		// Apply sensor→output matrix (mirrors pbrt-v4 outputRGBFromSensorRGB * rgb)
		const auto& M = output_rgb_from_sensor_;
		FilmPixelRGB out;
		out.r = static_cast<float>(M[0][0]*r + M[0][1]*g + M[0][2]*b);
		out.g = static_cast<float>(M[1][0]*r + M[1][1]*g + M[1][2]*b);
		out.b = static_cast<float>(M[2][0]*r + M[2][1]*g + M[2][2]*b);
		return out;
	}

	// -----------------------------------------------------------------------
	// get_pixel_gbuffer -- retrieve normalized AOV values for one pixel.
	// Returns weighted averages; caller checks gBufWeightSum > 0.
	// -----------------------------------------------------------------------
	struct GBufferResult {
		float p[3]      = {};   // world position
		float n[3]      = {};   // geometric normal
		float ns[3]     = {};   // shading normal
		float uv[2]     = {};
		float dzdx      = 0.f;
		float dzdy      = 0.f;
		float albedo[3] = {};   // output-space albedo RGB
		float gBufWeight = 0.f;
	};

	GBufferResult get_pixel_gbuffer(int px, int py) const {
		assert(px >= 0 && px < res_x_ && py >= 0 && py < res_y_);
		const GBufferPixel& pix = pixel(px, py);
		GBufferResult res;
		res.gBufWeight = static_cast<float>(pix.gBufWeightSum);
		if (pix.gBufWeightSum == 0.0) return res;
		double iw = 1.0 / pix.gBufWeightSum;
		for (int c = 0; c < 3; ++c) {
			res.p[c]  = static_cast<float>(pix.pSum[c]  * iw);
			res.n[c]  = static_cast<float>(pix.nSum[c]  * iw);
			res.ns[c] = static_cast<float>(pix.nsSum[c] * iw);
		}
		res.uv[0]  = static_cast<float>(pix.uvSum[0]  * iw);
		res.uv[1]  = static_cast<float>(pix.uvSum[1]  * iw);
		res.dzdx   = static_cast<float>(pix.dzdxSum   * iw);
		res.dzdy   = static_cast<float>(pix.dzdySum   * iw);

		// Convert albedo from sensor space to output color space
		double ar = pix.rgbAlbedoSum[0] * iw;
		double ag = pix.rgbAlbedoSum[1] * iw;
		double ab = pix.rgbAlbedoSum[2] * iw;
		const auto& M = output_rgb_from_sensor_;
		res.albedo[0] = static_cast<float>(M[0][0]*ar + M[0][1]*ag + M[0][2]*ab);
		res.albedo[1] = static_cast<float>(M[1][0]*ar + M[1][1]*ag + M[1][2]*ab);
		res.albedo[2] = static_cast<float>(M[2][0]*ar + M[2][1]*ag + M[2][2]*ab);
		return res;
	}

	void reset_pixel(int px, int py) { pixel(px, py).reset(); }
	void clear()                     { for (auto& p : pixels_) p.reset(); }

private:
	GBufferPixel&       pixel(int x, int y)       { return pixels_[y * res_x_ + x]; }
	const GBufferPixel& pixel(int x, int y) const { return pixels_[y * res_x_ + x]; }

	int   res_x_, res_y_;
	Filter filter_;
	float  filter_integral_;
	float  max_component_value_;
	SquareMatrix<3> output_rgb_from_sensor_;
	std::vector<GBufferPixel> pixels_;
};
