#pragma once
// ---------------------------------------------------------------------------
// function_integrator.h -- 2D function MC integrator for sampler verification
//
// Mirrors pbrt-v4 FunctionIntegrator (src/pbrt/cpu/integrators.h/cpp).
//
// Purpose: numerically integrates a 2D piecewise function over [0,1)^2
// using any sequence of (u,v) samples and measures Monte Carlo error.
// Used to verify sampler quality (convergence rate, variance).
//
// Algorithm (mirrors pbrt-v4 FunctionIntegrator::Render):
//   - Accumulate sum_i f(u_i, v_i) / N to estimate the integral.
//   - Compare against the known analytic integral to compute MSE.
//
// Built-in test functions (pbrt-v4 funcs namespace):
//   step              : f(x,y) = 2 if x < 0.5, else 0        (integral = 1)
//   diagonal          : f(x,y) = 2 if x+y < 1, else 0        (integral = 1)
//   disk              : uniform disk of radius 0.5 centered at (0.5,0.5)
//   checkerboard      : 10x10 checker, value 2 or 0            (integral = 1)
//   gaussian          : normalised 2D Gaussian (mu=0.5,sigma=0.25)
//
// Usage:
//   auto f  = FunctionIntegrator::step();
//   double est = FunctionIntegrator::integrate(f, samples, n);
//   double err = FunctionIntegrator::mse(f, samples, n);
//
// pbrt-v4 reference: src/pbrt/cpu/integrators.h  FunctionIntegrator
//                    src/pbrt/cpu/integrators.cpp FunctionIntegrator::Render
// ---------------------------------------------------------------------------

#include <cmath>
#include <functional>
#include <string>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class FunctionIntegrator {
public:
	// Type alias for a 2D test function f: [0,1)^2 -> double.
	using Func2D = std::function<double(double x, double y)>;

	// -----------------------------------------------------------------------
	// Built-in test functions (mirrors pbrt-v4 funcs namespace)
	// Each is normalised so its integral over [0,1)^2 equals analytic_integral().
	// -----------------------------------------------------------------------

	// f(x,y) = 2 if x < 0.5, else 0.  Integral = 1.
	static Func2D step() {
		return [](double x, double /*y*/) -> double {
			return (x < 0.5) ? 2.0 : 0.0;
		};
	}

	// f(x,y) = 2 if x+y < 1, else 0.  Integral = 1.
	static Func2D diagonal() {
		return [](double x, double y) -> double {
			return (x + y < 1.0) ? 2.0 : 0.0;
		};
	}

	// Uniform disk of radius 0.5 centered at (0.5, 0.5).
	// f = 1/(pi*r^2) inside, 0 outside.  Integral = 1.
	static Func2D disk() {
		const double r = 0.5;
		const double inv_area = 1.0 / (M_PI * r * r);
		return [r, inv_area](double x, double y) -> double {
			double dx = x - 0.5, dy = y - 0.5;
			return (dx*dx + dy*dy < r*r) ? inv_area : 0.0;
		};
	}

	// 10x10 checkerboard: value 2 on "white" cells, 0 on "black".
	// Integral = 1.
	static Func2D checkerboard() {
		return [](double x, double y) -> double {
			int freq = 10;
			int ix = static_cast<int>(x * freq);
			int iy = static_cast<int>(y * freq);
			return ((ix ^ iy) & 1) ? 2.0 : 0.0;
		};
	}

	// Rotated checkerboard (45 degrees).  Integral = 1.
	// Mirrors pbrt-v4 funcs::rotatedCheckerboard exactly:
	//   rotate point by 45 degrees, offset by 10, then apply checkerboard.
	static Func2D rotated_checkerboard() {
		const double angle = M_PI / 4.0;          // 45 degrees
		const double nrm   = 1.00006866455078125; // pbrt-v4 normalisation constant
		const double sa = std::sin(angle), ca = std::cos(angle);
		return [sa, ca, nrm](double x, double y) -> double {
			// Apply rotation + offset (matches pbrt-v4: Point2f(10+x*ca-y*sa, 10+x*sa+y*ca))
			double rx = 10.0 + x*ca - y*sa;
			double ry = 10.0 + x*sa + y*ca;
			// Checkerboard at freq=10: parity of integer cell indices
			// Mirrors pbrt-v4: Point2i pi(p * freq); ((pi.x & 1) ^ (pi.y & 1))
			int px = static_cast<int>(rx * 10);
			int py = static_cast<int>(ry * 10);
			return ((px ^ py) & 1) ? 2.0 / nrm : 0.0;
		};
	}

	// Normalised 2D Gaussian (mu=0.5, sigma=0.25) over [0,1)^2.
	// Integral over [0,1)^2 = 1 exactly: divided by GaussianIntegral([0,1])^2.
	// Mirrors pbrt-v4 funcs::gaussian: nrm = Sqr(GaussianIntegral(0, 1, mu, sigma)).
	static Func2D gaussian() {
		const double mu = 0.5, sigma = 0.25;
		// Normalisation: integral over [0,1] of each 1D Gaussian
		auto erf_val = [](double x) { return std::erf(x); };
		double sigmaRoot2 = sigma * 1.41421356237309504880;
		double g1d = 0.5 * (erf_val((mu - 0.0) / sigmaRoot2) -
							erf_val((mu - 1.0) / sigmaRoot2));
		double nrm = g1d * g1d; // 2D normalisation constant
		return [mu, sigma, nrm](double x, double y) -> double {
			auto G = [&](double t) {
				double d = t - mu;
				return std::exp(-d*d / (2.0*sigma*sigma)) /
					   std::sqrt(2.0*M_PI*sigma*sigma);
			};
			return (nrm > 0.0) ? G(x)*G(y) / nrm : 0.0;
		};
	}

	// -----------------------------------------------------------------------
	// Analytic integrals over [0,1)^2 for each built-in function.
	// (All return 1.0 by construction except gaussian which is ~1.)
	// -----------------------------------------------------------------------
	static double analytic_integral(const std::string& name) {
		if (name == "step")         return 1.0;
		if (name == "diagonal")     return 1.0;
		if (name == "disk")         return 1.0;
		if (name == "checkerboard") return 1.0;
		if (name == "rotated_checkerboard") return 1.0;
		if (name == "gaussian")     return 1.0; // exact: normalized by GaussianIntegral([0,1])^2
		throw std::invalid_argument("Unknown function: " + name);
	}

	// -----------------------------------------------------------------------
	// Factory by name (mirrors pbrt-v4 FunctionIntegrator::Create)
	// -----------------------------------------------------------------------
	static Func2D create(const std::string& name) {
		if (name == "step")                return step();
		if (name == "diagonal")            return diagonal();
		if (name == "disk")                return disk();
		if (name == "checkerboard")        return checkerboard();
		if (name == "rotated_checkerboard") return rotated_checkerboard();
		if (name == "gaussian")            return gaussian();
		throw std::invalid_argument("Unknown function: " + name);
	}

	// -----------------------------------------------------------------------
	// integrate()
	// Estimates the integral of f over [0,1)^2 using n samples.
	// samples must be an array of n*2 values: {u0,v0, u1,v1, ...}
	// -----------------------------------------------------------------------
	static double integrate(const Func2D& f,
							const double* samples, int n) {
		if (n <= 0) return 0.0;
		double sum = 0.0;
		for (int i = 0; i < n; ++i)
			sum += f(samples[2*i], samples[2*i+1]);
		return sum / n;
	}

	// -----------------------------------------------------------------------
	// mse()
	// Returns the squared error: (estimate - reference)^2.
	// reference is the analytic integral of f (caller-supplied).
	// -----------------------------------------------------------------------
	static double mse(const Func2D& f,
					  const double* samples, int n,
					  double reference) {
		double est = integrate(f, samples, n);
		double err = est - reference;
		return err * err;
	}

	// -----------------------------------------------------------------------
	// variance()
	// Estimates variance of f under the sampling distribution.
	// Uses two-pass algorithm for numerical stability.
	// -----------------------------------------------------------------------
	static double variance(const Func2D& f,
						   const double* samples, int n) {
		if (n <= 1) return 0.0;
		double mean = integrate(f, samples, n);
		double var  = 0.0;
		for (int i = 0; i < n; ++i) {
			double d = f(samples[2*i], samples[2*i+1]) - mean;
			var += d * d;
		}
		return var / (n - 1);
	}
};
