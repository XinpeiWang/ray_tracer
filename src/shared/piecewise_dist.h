#ifndef PIECEWISE_DIST_H
#define PIECEWISE_DIST_H
//==============================================================================
// piecewise_dist.h -- 1D and 2D piecewise-constant sampling distributions
//
// Mirrors pbrt-v4 PiecewiseConstant1D / PiecewiseConstant2D (§A.5 / sampling.h).
//
// PiecewiseConstant1D
//   Build: O(N)   sample: O(log N) via binary search on CDF
//   sample(u) -> {continuous x in [0,1], pdf value}
//   pdf(x)    -> step-function PDF for position x in [0,1]
//
// PiecewiseConstant2D
//   Per-row conditional 1D + marginal 1D (pbrt-v4 §14.2.4)
//   Build: O(N*M)   sample: O(log N + log M)
//   sample(u,v) -> {(u_out,v_out) in [0,1]^2, joint pdf}
//   pdf(u,v)    -> joint pdf at image-space (u,v)
//==============================================================================

#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>

// ---------------------------------------------------------------------------
// PiecewiseConstant1D
// ---------------------------------------------------------------------------
class PiecewiseConstant1D {
  public:
	PiecewiseConstant1D() = default;

	// Build from an array of non-negative function values f[0..n-1].
	// All values are treated as weights for equal-width bins over [0, 1].
	explicit PiecewiseConstant1D(const std::vector<double>& f) {
		int n = (int)f.size();
		assert(n > 0);
		func = f;

		// Build CDF: cdf[0]=0, cdf[i] = sum(f[0..i-1]) / n  (unnormalized)
		cdf.resize(n + 1);
		cdf[0] = 0.0;
		for (int i = 0; i < n; ++i)
			cdf[i + 1] = cdf[i] + (f[i] >= 0.0 ? f[i] : 0.0) / n;

		func_int = cdf[n]; // integral of the step function over [0,1]

		if (func_int == 0.0) {
			// Degenerate: uniform fallback -- set func values to 1 so pdf() = 1
			for (int i = 0; i < n; ++i)
				func[i] = 1.0;
			for (int i = 1; i <= n; ++i)
				cdf[i] = double(i) / double(n);
			func_int = 1.0;
		} else {
			for (int i = 1; i <= n; ++i)
				cdf[i] /= func_int;
		}
	}

	// Integral of the step function over [0,1] (normalization constant)
	double integral() const { return func_int; }
	int    size()     const { return (int)func.size(); }

	// Sample a continuous position x in [0,1] given uniform u in [0,1).
	// Returns the sampled x and sets pdf_out to the PDF value at x.
	double sample(double u, double* pdf_out = nullptr) const {
		int n = (int)func.size();
		// Binary search: find bin o such that cdf[o] <= u < cdf[o+1]
		int o = (int)(std::upper_bound(cdf.begin(), cdf.end(), u) - cdf.begin()) - 1;
		if (o < 0)  o = 0;
		if (o >= n) o = n - 1;

		// Linear interpolation within the bin
		double du = u - cdf[o];
		double dcdf = cdf[o + 1] - cdf[o];
		if (dcdf > 0.0) du /= dcdf;

		double x = (o + du) / double(n);  // continuous position in [0,1]

		if (pdf_out)
			*pdf_out = (func_int > 0.0) ? func[o] / func_int : 0.0;

		return x;
	}

	// PDF at continuous position x in [0,1]
	double pdf(double x) const {
		if (func_int == 0.0) return 0.0;
		int o = (int)(x * func.size());
		if (o < 0) o = 0;
		if (o >= (int)func.size()) o = (int)func.size() - 1;
		return func[o] / func_int;
	}

	// func[i] value for bin i (used by PiecewiseConstant2D to build marginal)
	double func_value(int i) const { return func[i]; }

  private:
	std::vector<double> func; // original function values
	std::vector<double> cdf;  // CDF[0..n], cdf[0]=0, cdf[n]=1
	double func_int = 0.0;    // integral of step function over [0,1]
};


// ---------------------------------------------------------------------------
// PiecewiseConstant2D
// ---------------------------------------------------------------------------
// Models p(u,v) = f(u,v) / integral(f) as a product:
//   p(v) = marginal,  p(u|v) = conditional row distribution
//
// Row index v corresponds to image rows (height dimension).
// Column index u corresponds to image columns (width dimension).
// ---------------------------------------------------------------------------
class PiecewiseConstant2D {
  public:
	PiecewiseConstant2D() = default;

	// Build from a flat row-major array of size nu * nv.
	// f[v * nu + u] is the luminance weight at pixel (u, v).
	PiecewiseConstant2D(const std::vector<double>& f, int nu, int nv) {
		assert((int)f.size() == nu * nv);

		// Build per-row conditional distributions
		conditionals.reserve(nv);
		std::vector<double> marginal_func(nv);
		for (int v = 0; v < nv; ++v) {
			std::vector<double> row(f.begin() + v * nu, f.begin() + (v + 1) * nu);
			conditionals.emplace_back(row);
			// Marginal weight = row integral (proportional to total row luminance)
			marginal_func[v] = conditionals.back().integral();
		}

		// Build marginal distribution over rows
		marginal = PiecewiseConstant1D(marginal_func);
	}

	// Sample a 2D position (u,v) in [0,1]^2 given two independent uniforms.
	// Returns the sampled (u,v) and sets pdf_out to the joint PDF p(u,v).
	// pdf(u,v) = p(u|v) * p(v)  (conditional × marginal)
	std::pair<double,double> sample(double ru, double rv,
									double* pdf_out = nullptr) const {
		double pdf_v, pdf_u;
		double v = marginal.sample(rv, &pdf_v);

		// Select the conditional row
		int row_idx = (int)(v * (int)conditionals.size());
		if (row_idx >= (int)conditionals.size())
			row_idx = (int)conditionals.size() - 1;

		double u = conditionals[row_idx].sample(ru, &pdf_u);

		if (pdf_out) *pdf_out = pdf_u * pdf_v;
		return { u, v };
	}

	// Joint PDF p(u,v) = p(u|v) * p(v) for image-space (u,v) in [0,1]^2
	double pdf(double u, double v) const {
		int nv = (int)conditionals.size();
		int row_idx = (int)(v * nv);
		if (row_idx < 0)   row_idx = 0;
		if (row_idx >= nv) row_idx = nv - 1;

		return conditionals[row_idx].pdf(u) * marginal.pdf(v);
	}

	bool empty() const { return conditionals.empty(); }

  private:
	std::vector<PiecewiseConstant1D> conditionals; // one per image row
	PiecewiseConstant1D              marginal;      // over rows
};

#endif
