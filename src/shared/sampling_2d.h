#pragma once
#include "sampling_helpers.h"

// ===========================================================================
// Bilinear sampling -- pbrt-v4 util/sampling.h
//
// Samples a point in [0,1]^2 from a bilinear distribution defined by four
// corner weights w[0..3] ordered as:
//   w[0]=f(0,0)  w[1]=f(1,0)  w[2]=f(0,1)  w[3]=f(1,1)
//
// References:
//   pbrt-v4 LinearPDF / SampleLinear / InvertLinearSample
//   pbrt-v4 BilinearPDF / SampleBilinear / InvertBilinearSample
// ===========================================================================

// PDF of a linear distribution on [0,1] with end-points a (at x=0) and b (at x=1).
// pbrt-v4: LinearPDF(x, a, b) = 2 * Lerp(x, a, b) / (a + b)
template<typename T>
CPU_GPU T LinearPDF(T x, T a, T b) {
	if (x < T(0) || x > T(1)) return T(0);
	T sum = a + b;
	if (sum == T(0)) return T(1);
	return T(2) * (a + (b - a) * x) / sum;
}

// Invert the CDF of the linear distribution.
// pbrt-v4: InvertLinearSample(x, a, b) = x * (a*(2-x) + b*x) / (a+b)
template<typename T>
CPU_GPU T InvertLinearSample(T x, T a, T b) {
	T sum = a + b;
	if (sum == T(0)) return x;
	return x * (a * (T(2) - x) + b * x) / sum;
}

// Sample x in [0,1) from a linear distribution with f(0)=a, f(1)=b.
// pbrt-v4: SampleLinear(u, a, b)
template<typename T>
CPU_GPU T SampleLinear(T u, T a, T b) {
	if (u == T(0) && a == T(0)) return T(0);
	T sum = a + b;
	if (sum == T(0)) return u;
	// Invert CDF: u = x*(2a + (b-a)*x) / (a+b)  =>  quadratic in x
	// Solved as: x = u*(a+b) / (a + sqrt(Lerp(u, a^2, b^2)))
	T lerp_sq = a * a + (b * b - a * a) * u;
	if (lerp_sq < T(0)) lerp_sq = T(0);
	T x = u * sum / (a + std::sqrt(lerp_sq));
	// Clamp to [0, 1-epsilon)
	if (x >= T(1)) x = T(1) - T(1e-7);
	return x;
}

// PDF of a bilinear distribution at point (px, py) in [0,1]^2.
// w[0]=f(0,0), w[1]=f(1,0), w[2]=f(0,1), w[3]=f(1,1).
// pbrt-v4: BilinearPDF(p, w)
template<typename T>
CPU_GPU T BilinearPDF(T px, T py, T w0, T w1, T w2, T w3) {
	if (px < T(0) || px > T(1) || py < T(0) || py > T(1)) return T(0);
	T total = w0 + w1 + w2 + w3;
	if (total == T(0)) return T(1);
	// Bilinear interpolation of weight at (px,py):
	//   f = (1-px)*(1-py)*w0 + px*(1-py)*w1 + (1-px)*py*w2 + px*py*w3
	T f = (T(1)-px)*(T(1)-py)*w0 + px*(T(1)-py)*w1
		+ (T(1)-px)*py       *w2 + px*py       *w3;
	return T(4) * f / total;
}

// Sample (px, py) from a bilinear distribution.
// Returns sampled point in [0,1)^2.
// pbrt-v4: SampleBilinear(u, w) -- marginal-y then conditional-x
template<typename T>
CPU_GPU void SampleBilinear(T u0, T u1, T w0, T w1, T w2, T w3,
							 T& out_px, T& out_py) {
	// Marginal: f_y(y) proportional to (w0+w1)*(1-y) + (w2+w3)*y
	out_py = SampleLinear(u1, w0 + w1, w2 + w3);
	// Conditional: f_x(x|y) proportional to lerp(y,w0,w2)*(1-x) + lerp(y,w1,w3)*x
	T ax = w0 + (w2 - w0) * out_py;
	T bx = w1 + (w3 - w1) * out_py;
	out_px = SampleLinear(u0, ax, bx);
}

// Invert SampleBilinear to recover (u0, u1) from a sampled point (px, py).
// pbrt-v4: InvertBilinearSample(p, w)
template<typename T>
CPU_GPU void InvertBilinearSample(T px, T py, T w0, T w1, T w2, T w3,
								   T& out_u0, T& out_u1) {
	T ax = w0 + (w2 - w0) * py;
	T bx = w1 + (w3 - w1) * py;
	out_u0 = InvertLinearSample(px, ax, bx);
	out_u1 = InvertLinearSample(py, w0 + w1, w2 + w3);
}

// ===========================================================================
// SampleUniformTriangle -- pbrt-v4 util/sampling.h
//
// Maps a uniform 2D sample (u0, u1) in [0,1)^2 to barycentric coordinates
// (b0, b1, b2) with b0+b1+b2 = 1, uniformly over a triangle.
//
// Uses the Heitz (2019) diagonal-fold method -- same as pbrt-v4:
//   if u0 < u1:  b0 = u0/2,       b1 = u1 - u0/2
//   else:        b0 = u0 - u1/2,  b1 = u1/2
//   b2 = 1 - b0 - b1
//
// Reference: pbrt-v4 SampleUniformTriangle (util/sampling.h)
// ===========================================================================
template<typename T>
CPU_GPU void SampleUniformTriangle(T u0, T u1, T& b0, T& b1, T& b2) {
	if (u0 < u1) {
		b0 = u0 * T(0.5);
		b1 = u1 - b0;
	} else {
		b1 = u1 * T(0.5);
		b0 = u0 - b1;
	}
	b2 = T(1) - b0 - b1;
}

// ===========================================================================
// Catmull-Rom spline CDF inversion -- pbrt-v4 util/sampling.cpp
//
// These functions implement importance sampling of a 1D or 2D function
// represented as a Catmull-Rom spline on a uniform or non-uniform grid.
//
// Key helpers:
//   find_interval       -- binary search for interval containing x
//   evaluate_polynomial -- Horner's method for polynomial evaluation
//   newton_bisection    -- hybrid Newton-bisection root finding
//   catmull_rom_weights -- compute spline interpolation weights
//   integrate_catmull_rom -- build CDF from spline (trapezoidal + derivative)
//   SampleCatmullRom    -- invert 1D Catmull-Rom CDF
//   SampleCatmullRom2D  -- invert 2D Catmull-Rom CDF (marginal + conditional)
//
// References:
//   pbrt-v4 util/math.h (CatmullRomWeights, IntegrateCatmullRom,
//            FindInterval, NewtonBisection, EvaluatePolynomial)
//   pbrt-v4 util/sampling.cpp (SampleCatmullRom, SampleCatmullRom2D)
// ===========================================================================

// Binary search: find largest i in [0, n-2] s.t. pred(i) is true.
// pbrt-v4: FindInterval (util/math.h)
template<typename Predicate>
inline int catmullrom_find_interval(int n, Predicate pred) {
	int size = n - 2, first = 1;
	while (size > 0) {
		int half   = size >> 1;
		int middle = first + half;
		if (pred(middle)) { first = middle + 1; size -= half + 1; }
		else              { size   = half; }
	}
	int r = first - 1;
	if (r < 0)   r = 0;
	if (r > n-2) r = n-2;
	return r;
}

// Evaluate polynomial by Horner's method.
// pbrt-v4: EvaluatePolynomial(t, c0, c1, ..., cn) = c0 + t*(c1 + t*(...))
inline double evaluate_polynomial(double /*t*/) { return 0.0; }
inline double evaluate_polynomial(double /*t*/, double c) { return c; }
template<typename... Rest>
inline double evaluate_polynomial(double t, double c0, Rest... rest) {
	return c0 + t * evaluate_polynomial(t, static_cast<double>(rest)...);
}

// Hybrid Newton-bisection: find root of f in [x0, x1].
// f must return {f(x), f'(x)}.
// pbrt-v4: NewtonBisection (util/math.h)
template<typename Func>
inline double catmullrom_newton_bisection(double x0, double x1, Func f,
										  double x_eps = 1e-6, double f_eps = 1e-6) {
	double fx0 = f(x0).first, fx1 = f(x1).first;
	if (std::abs(fx0) < f_eps) return x0;
	if (std::abs(fx1) < f_eps) return x1;
	bool neg_start = (fx0 < 0);
	double xMid = x0 + (x1 - x0) * (-fx0) / (fx1 - fx0);
	for (int iter = 0; iter < 64; ++iter) {
		if (!(x0 < xMid && xMid < x1))
			xMid = (x0 + x1) * 0.5;
		auto [fMid, dfMid] = f(xMid);
		if (std::abs(fMid) < f_eps || x1 - x0 < x_eps) break;
		if ((fMid < 0) == neg_start) x0 = xMid;
		else                          x1 = xMid;
		xMid -= fMid / dfMid;
	}
	return xMid;
}

// Compute Catmull-Rom interpolation weights for value x in the node array.
// Returns false if x is out of bounds.
// offset is set to idx-1 (first of 4 contributing nodes).
// pbrt-v4: CatmullRomWeights (util/math.cpp)
inline bool catmull_rom_weights(const double* nodes, int n, double x,
								 int* offset, double weights[4]) {
	if (!(x >= nodes[0] && x <= nodes[n-1])) return false;
	int idx = catmullrom_find_interval(n, [&](int i){ return nodes[i] <= x; });
	*offset = idx - 1;
	double x0 = nodes[idx], x1 = nodes[idx+1];
	double t  = (x - x0) / (x1 - x0), t2 = t*t, t3 = t2*t;
	weights[1] =  2*t3 - 3*t2 + 1;
	weights[2] = -2*t3 + 3*t2;
	// w0
	if (idx > 0) {
		double w0 = (t3 - 2*t2 + t) * (x1 - x0) / (x1 - nodes[idx-1]);
		weights[0] = -w0;
		weights[2] += w0;
	} else {
		double w0 = t3 - 2*t2 + t;
		weights[0] = 0;
		weights[1] -= w0;
		weights[2] += w0;
	}
	// w3
	if (idx + 2 < n) {
		double w3 = (t3 - t2) * (x1 - x0) / (nodes[idx+2] - x0);
		weights[1] -= w3;
		weights[3]  = w3;
	} else {
		double w3 = t3 - t2;
		weights[1] -= w3;
		weights[2] += w3;
		weights[3]  = 0;
	}
	return true;
}

// Build a CDF (cumulative integral) for a Catmull-Rom spline.
// cdf must have n entries; cdf[0] = 0. Returns total integral.
// pbrt-v4: IntegrateCatmullRom (util/math.cpp)
inline double integrate_catmull_rom(const double* nodes, const double* f,
									 int n, double* cdf) {
	double sum = 0;
	cdf[0] = 0;
	for (int i = 0; i < n-1; ++i) {
		double x0 = nodes[i], x1 = nodes[i+1];
		double f0 = f[i],     f1 = f[i+1];
		double w  = x1 - x0;
		double d0 = (i > 0)   ? w * (f1 - f[i-1]) / (x1 - nodes[i-1]) : (f1 - f0);
		double d1 = (i+2 < n) ? w * (f[i+2] - f0) / (nodes[i+2] - x0) : (f1 - f0);
		sum += w * ((f0 + f1) * 0.5 + (d0 - d1) / 12.0);
		cdf[i+1] = sum;
	}
	return sum;
}

// Sample a value from a 1D Catmull-Rom spline distribution.
//
// Parameters:
//   nodes  -- n node positions (sorted ascending)
//   f      -- n function values f(nodes[i])
//   F      -- n CDF values (precomputed by integrate_catmull_rom)
//   n      -- number of nodes
//   u      -- uniform sample in [0,1)
//   fval   -- (optional) output: interpolated f at sampled position
//   pdf    -- (optional) output: pdf at sampled position
//
// Returns the sampled position.
// pbrt-v4: SampleCatmullRom (util/sampling.cpp)
inline double SampleCatmullRom(const double* nodes, const double* f,
								const double* F, int n,
								double u, double* fval = nullptr, double* pdf = nullptr) {
	u *= F[n-1];
	int i = catmullrom_find_interval(n, [&](int k){ return F[k] <= u; });
	double x0 = nodes[i], x1 = nodes[i+1];
	double f0 = f[i],     f1 = f[i+1];
	double w  = x1 - x0;
	double d0 = (i > 0)   ? w * (f1 - f[i-1]) / (x1 - nodes[i-1]) : (f1 - f0);
	double d1 = (i+2 < n) ? w * (f[i+2] - f0) / (nodes[i+2] - x0) : (f1 - f0);
	u = (u - F[i]) / w;

	double Fhat = 0, fhat = 0;
	auto eval = [&](double t) -> std::pair<double,double> {
		Fhat = evaluate_polynomial(t, 0.0, f0, 0.5*d0,
				   (1.0/3.0)*(-2*d0 - d1) + f1 - f0,
					0.25*(d0 + d1) + 0.5*(f0 - f1));
		fhat = evaluate_polynomial(t, f0, d0,
				   -2*d0 - d1 + 3*(f1 - f0),
					d0 + d1 + 2*(f0 - f1));
		return {Fhat - u, fhat};
	};
	double t = catmullrom_newton_bisection(0.0, 1.0, eval);
	// Trigger final eval to update fhat
	eval(t);

	if (fval) *fval = fhat;
	if (pdf)  *pdf  = fhat / F[n-1];
	return x0 + w * t;
}

// Sample a value from a 2D Catmull-Rom spline distribution.
//
// The 2D table is indexed as values[i * n2 + j], cdf[i * n2 + j],
// where i indexes nodes1 (the "alpha" / first dimension to condition on)
// and j indexes nodes2 (the dimension to sample).
//
// Parameters:
//   nodes1  -- n1 node positions for the first (conditioning) dimension
//   nodes2  -- n2 node positions for the second (sampled) dimension
//   values  -- n1*n2 table of function values
//   cdf     -- n1*n2 precomputed CDF values (integrate_catmull_rom per row)
//   n1, n2  -- grid dimensions
//   alpha   -- value in nodes1 range (conditioning variable)
//   u       -- uniform sample in [0,1) (for sampling the second dimension)
//   fval    -- (optional) output: interpolated function value
//   pdf     -- (optional) output: pdf at sampled position
//
// Returns the sampled position in nodes2 range.
// pbrt-v4: SampleCatmullRom2D (util/sampling.cpp)
inline double SampleCatmullRom2D(const double* nodes1, const double* nodes2,
								  const double* values, const double* cdf,
								  int n1, int n2,
								  double alpha, double u,
								  double* fval = nullptr, double* pdf = nullptr) {
	int   offset;
	double weights[4];
	if (!catmull_rom_weights(nodes1, n1, alpha, &offset, weights))
		return 0.0;

	// Interpolate row values using the 4 Catmull-Rom weights
	auto interpolate = [&](const double* arr, int col) {
		double v = 0;
		for (int k = 0; k < 4; ++k)
			if (weights[k] != 0 && offset + k >= 0 && offset + k < n1)
				v += arr[(offset + k) * n2 + col] * weights[k];
		return v;
	};

	double maximum = interpolate(cdf, n2-1);
	u *= maximum;
	int idx = catmullrom_find_interval(n2, [&](int j){ return interpolate(cdf, j) <= u; });

	double f0 = interpolate(values, idx),   f1 = interpolate(values, idx+1);
	double x0 = nodes2[idx],               x1 = nodes2[idx+1];
	double w  = x1 - x0;
	double d0 = (idx > 0)    ? w * (f1 - interpolate(values, idx-1)) / (x1 - nodes2[idx-1])
							  : f1 - f0;
	double d1 = (idx+2 < n2) ? w * (interpolate(values, idx+2) - f0) / (nodes2[idx+2] - x0)
							  : f1 - f0;
	u = (u - interpolate(cdf, idx)) / w;

	double Fhat = 0, fhat = 0;
	auto eval = [&](double t) -> std::pair<double,double> {
		Fhat = evaluate_polynomial(t, 0.0, f0, 0.5*d0,
				   (1.0/3.0)*(-2*d0 - d1) + f1 - f0,
					0.25*(d0 + d1) + 0.5*(f0 - f1));
		fhat = evaluate_polynomial(t, f0, d0,
				   -2*d0 - d1 + 3*(f1 - f0),
					d0 + d1 + 2*(f0 - f1));
		return {Fhat - u, fhat};
	};
	double t = catmullrom_newton_bisection(0.0, 1.0, eval);
	eval(t);

	if (fval) *fval = fhat;
	if (pdf)  *pdf  = fhat / maximum;
	return x0 + w * t;
}
