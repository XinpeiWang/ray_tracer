#pragma once
//==============================================================================================
// filter.h -- Pixel reconstruction filters (pbrt-v4 filters.h inspired, CPU-only)
//
// Replaces the box filter (plain average) with a proper reconstruction filter.
// Each sample's contribution to a pixel is weighted by Evaluate(offset_x, offset_y)
// where (offset_x, offset_y) is the sample's sub-pixel position in [-0.5, 0.5].
//
// Usage:
//   MitchellFilter f;                    // default B=1/3, C=1/3, radius=0.5
//   double w = f.evaluate(ox, oy);       // weight for sample at sub-pixel offset (ox, oy)
//   // accumulate: weighted_sum += w * sample; weight_sum += w;
//   // final pixel: weighted_sum / weight_sum
//
// pbrt-v4 reference: src/pbrt/filters.h MitchellFilter, Mitchell1D()
//==============================================================================================

#include <cmath>
#include "scalar_math.h"

// ---------------------------------------------------------------------------
// BoxFilter -- uniform weight 1 for all samples (current default, baseline)
// ---------------------------------------------------------------------------
struct BoxFilter {
	explicit BoxFilter(double radius = 0.5) : radius_(radius) {}
	double evaluate(double /*ox*/, double /*oy*/) const { return 1.0; }
	double radius() const { return radius_; }
	// Integral of box filter over [-r,r]^2 = 4*r^2
	double integral() const { return 4.0 * radius_ * radius_; }
  private:
	double radius_;
};

// ---------------------------------------------------------------------------
// MitchellFilter
//
// pbrt-v4 defaults: B = 1/3, C = 1/3, radius = 0.5 pixel
//
// Mitchell1D(x) is evaluated on x in [-2, 2] (scaled so the filter radius
// maps to x=1). For sub-pixel offsets in [-radius, +radius]:
//   x_scaled = offset / radius  ->  in [-1, 1]
//   Mitchell1D argument = x_scaled * 2  ->  in [-2, 2] as pbrt-v4 uses.
//
// The 2D filter is separable:
//   w(ox, oy) = Mitchell1D(2*ox/radius) * Mitchell1D(2*oy/radius)
//
// Properties:
//   - B=0, C=0.5  -> Catmull-Rom (sharper, some ringing)
//   - B=1, C=0    -> cubic B-spline (blurry, no ringing)
//   - B=1/3, C=1/3 -> Mitchell-Netravali (recommended, balanced)
// ---------------------------------------------------------------------------
class MitchellFilter {
  public:
	// radius: half-width of the filter in pixel units. 0.5 = only within pixel.
	// B, C: Mitchell-Netravali shape parameters (pbrt-v4 defaults: 1/3, 1/3).
	explicit MitchellFilter(double radius = 0.5,
							double B = 1.0 / 3.0,
							double C = 1.0 / 3.0)
		: radius_(radius), B_(B), C_(C) {}

	// Evaluate the 2D filter weight for a sample at sub-pixel offset (ox, oy).
	// ox, oy are in [-radius, +radius] (typically [-0.5, 0.5] for a pixel-width filter).
	double evaluate(double ox, double oy) const {
		return mitchell1d(2.0 * ox / radius_) * mitchell1d(2.0 * oy / radius_);
	}

	double radius() const { return radius_; }
	double B()      const { return B_; }
	double C()      const { return C_; }
	// Integral of the 2D Mitchell filter -- pbrt-v4: radius.x * radius.y / 4
	double integral() const { return radius_ * radius_ / 4.0; }

  private:
	// pbrt-v4 Mitchell1D
	// Returns values in roughly [-0.25, 1] depending on B and C.
	double mitchell1d(double x) const {
		x = std::fabs(x);
		if (x <= 1.0)
			return ((12.0 - 9.0*B_ - 6.0*C_) * x*x*x
				  + (-18.0 + 12.0*B_ + 6.0*C_) * x*x
				  + (6.0 - 2.0*B_)) * (1.0 / 6.0);
		else if (x <= 2.0)
			return ((-B_ - 6.0*C_) * x*x*x
				  + (6.0*B_ + 30.0*C_) * x*x
				  + (-12.0*B_ - 48.0*C_) * x
				  + (8.0*B_ + 24.0*C_)) * (1.0 / 6.0);
		else
			return 0.0;
	}

	double radius_, B_, C_;
};

// ---------------------------------------------------------------------------
// GaussianFilter -- Gaussian with clamp-to-zero at radius (pbrt-v4 style)
//
// sigma controls the falloff; pbrt-v4 default sigma = 0.5.
// ---------------------------------------------------------------------------
class GaussianFilter {
  public:
	explicit GaussianFilter(double radius = 0.5, double sigma = 0.5)
		: radius_(radius), sigma_(sigma),
		  exp_val_(std::exp(-sigma_ * sigma_ * radius_ * radius_)) {}

	double evaluate(double ox, double oy) const {
		return gauss1d(ox) * gauss1d(oy);
	}

	double radius() const { return radius_; }
	// Analytic integral -- mirrors pbrt-v4 GaussianFilter::Integral():
	//   GaussianIntegral(a,b,mu,sigma) = 0.5*(erf((mu-a)/sqrt2sig) - erf((mu-b)/sqrt2sig))
	//   integral = (GaussianIntegral(-r,r,0,sig) - 2*r*exp_val) ^ 2
	double integral() const {
		double sqrt2sig = sigma_ * 1.41421356237309504880;
		// GaussianIntegral(-r,r,0,sig) = 0.5*(erf((0-(-r))/sqrt2sig) - erf((0-r)/sqrt2sig))
		//                               = 0.5*(erf(r/sqrt2sig) - erf(-r/sqrt2sig))
		double gi = 0.5 * (std::erf(radius_ / sqrt2sig) - std::erf(-radius_ / sqrt2sig));
		// pbrt-v4: GaussianIntegral(-r,r,0,sigma) - 2*r*expVal
		double i1d = gi - 2.0 * radius_ * exp_val_;
		return i1d * i1d;  // 2D separable
	}

  private:
	double gauss1d(double x) const {
		double v = std::exp(-sigma_ * sigma_ * x * x) - exp_val_;
		return (v > 0.0) ? v : 0.0;
	}

	double radius_, sigma_, exp_val_;
};

// ---------------------------------------------------------------------------
// LanczosSincFilter -- Windowed Lanczos sinc reconstruction filter
//
// pbrt-v4 reference: filters.h LanczosSincFilter, math.h WindowedSinc / Sinc
//
// The Lanczos sinc is sinc(x) * sinc(x/tau), windowed to zero outside
// [-radius, +radius]. tau controls how many sinc lobes fit in the window:
//   tau=3 (default) includes 3 lobes and is the pbrt-v4 default.
//
// Compared to Mitchell: sharper high-frequency reconstruction, slight ringing
// for very low tau values. Excellent choice for high-quality offline rendering.
//
// Evaluate(x, y) = WindowedSinc(x, radius, tau) * WindowedSinc(y, radius, tau)
// ---------------------------------------------------------------------------
class LanczosSincFilter {
  public:
	// radius: half-width of the filter in pixel units (pbrt-v4 default: 4)
	// tau:    number of sinc lobes inside the window (pbrt-v4 default: 3)
	explicit LanczosSincFilter(double radius = 4.0, double tau = 3.0)
		: radius_(radius), tau_(tau) {}

	double evaluate(double ox, double oy) const {
		return windowed_sinc(ox) * windowed_sinc(oy);
	}

	double radius() const { return radius_; }
	double tau()    const { return tau_; }
	// Numerical integral: sample 512 points per axis
	double integral() const {
		const int N = 512;
		double sum = 0.0;
		double step = 2.0 * radius_ / N;
		for (int i = 0; i < N; ++i) {
			double x = -radius_ + (i + 0.5) * step;
			sum += windowed_sinc(x) * step;
		}
		return sum * sum;  // 2D separable
	}

  private:
	// sinc(x) = sin(pi*x)/(pi*x)
	static double sinc(double x) { return Sinc(x); }

	// WindowedSinc(x, radius, tau) = sinc(x) * sinc(x/tau), zero outside radius
	// (pbrt-v4 WindowedSinc in math.h)
	double windowed_sinc(double x) const {
		if (std::fabs(x) > radius_) return 0.0;
		return sinc(x) * sinc(x / tau_);
	}

	double radius_, tau_;
};

// ---------------------------------------------------------------------------
// TriangleFilter -- linear (tent) falloff reconstruction filter
//
// pbrt-v4 reference: src/pbrt/filters.h TriangleFilter
//
// The simplest non-trivial filter: weight falls off linearly from 1 at the
// center to 0 at the edge of the support radius.  Separable in x and y:
//
//   w(ox, oy) = max(0, radius - |ox|) * max(0, radius - |oy|)
//
// Compared to BoxFilter: softer, avoids aliasing at the cost of slight blur.
// Compared to Mitchell/Gaussian: cheaper to evaluate, good for real-time use.
// ---------------------------------------------------------------------------
class TriangleFilter {
  public:
	explicit TriangleFilter(double radius = 2.0) : radius_(radius) {}

	// Evaluate the 2D filter weight for a sample at sub-pixel offset (ox, oy).
	double evaluate(double ox, double oy) const {
		return tent1d(ox) * tent1d(oy);
	}

	double radius() const { return radius_; }

	// Integral of the 2D filter over its support [-r,r]^2.
	// pbrt-v4 TriangleFilter::Integral() = r^2 * r^2 (each 1D integral = r^2/2, but
	// pbrt-v4 uses Sqr(radius.x)*Sqr(radius.y) which equals r^4/4 for square; however
	// the 1D integral of tent(x) = int_{-r}^{r} (r-|x|) dx = r^2, so 2D = r^4.
	// pbrt-v4: Sqr(radius.x) * Sqr(radius.y) = r^2 * r^2 = r^4. Matches.
	double integral() const { return radius_ * radius_ * radius_ * radius_; }

  private:
	// pbrt-v4 TriangleFilter::Evaluate: max(0, radius - |x|)
	double tent1d(double x) const {
		double v = radius_ - std::fabs(x);
		return (v > 0.0) ? v : 0.0;
	}

	double radius_;
};
