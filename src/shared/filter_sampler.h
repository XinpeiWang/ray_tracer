#pragma once
// ---------------------------------------------------------------------------
// filter_sampler.h -- Tabulated importance sampling for reconstruction filters
//
// Mirrors pbrt-v4 FilterSampler (src/pbrt/filters.h / filters.cpp).
//
// Algorithm (pbrt-v4 Sec. 8.8.1):
//   1. Tabulate the (non-negative) filter function on an N×N grid covering
//      [-radius, +radius]² in both dimensions.
//   2. Build a 2D piecewise-constant distribution from the table:
//        - Per-row 1D CDFs for sampling the column (x) given the row.
//        - A marginal 1D CDF for sampling the row (y).
//   3. Sample(u1, u2):
//        a. Use u2 to draw row index i via marginal CDF (binary search).
//        b. Use u1 to draw column index j via row CDF (binary search).
//        c. Map (j,i) to continuous filter-domain coordinate p.
//        d. Return { p, f[i][j] / pdf } where pdf = f[i][j] / (integral * area_per_cell).
//
// Design rules (same as bxdfs.h):
//   - Header-only, no heap allocation: uses fixed-size stack arrays templated
//     on resolution N (default 32, matching pbrt-v4's "32 * radius" grid size).
//   - CPU_GPU tagged — usable on both CPU and GPU.
//   - Double precision on CPU, float on GPU (via template T).
//   - Works with any filter type that exposes:
//       double evaluate(double ox, double oy) const;
//       double radius() const;
//
// Usage:
//   MitchellFilter f;
//   FilterSampler<double, 32> fs(f);
//   auto [p, w] = fs.sample(u1, u2);   // p in [-radius,+radius]², w = f/pdf
// ---------------------------------------------------------------------------

#include <cmath>
#include <algorithm>
#include "filter.h"

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU inline
#   endif
#endif

// ---------------------------------------------------------------------------
// FilterSample: result of one importance-sampled filter draw
// ---------------------------------------------------------------------------
template <typename T>
struct FilterSample {
	T p_x, p_y;   // sub-pixel position in [-radius, +radius]
	T weight;     // unnormalized filter value / pdf  (= f(p) / pdf)
};

// ---------------------------------------------------------------------------
// FilterSampler<T, N>
//   T : scalar type (double on CPU, float on GPU)
//   N : table resolution per axis (default 32, matching pbrt-v4)
// ---------------------------------------------------------------------------
template <typename T, int N = 32>
class FilterSampler {
public:
	// Construct from any filter with evaluate(ox,oy) and radius() methods.
	// Evaluates the filter on an N×N grid and precomputes CDFs.
	template <typename Filter>
	CPU_GPU explicit FilterSampler(const Filter& filter) {
		radius_ = T(filter.radius());
		build(filter);
	}

	// Sample a filter position given two uniform [0,1) variates.
	// Returns a FilterSample with position in [-radius,+radius]² and weight = f/pdf.
	//
	// Mirrors pbrt-v4 FilterSampler::Sample + PiecewiseConstant1D::Sample:
	// after selecting the cell via CDF inversion, we linearly interpolate within
	// the cell using the fractional overshoot  du = (u - cdf_prev) / (cdf_curr - cdf_prev)
	// to produce a continuous position, exactly as pbrt-v4 does with:
	//   return Lerp((o + du) / size(), min, max)
	CPU_GPU FilterSample<T> sample(T u1, T u2) const {
		// u2 → row (y) via marginal CDF, u1 → column (x) via conditional CDF
		int row = lower_bound_cdf(marginal_cdf_, N, u2);
		int col = lower_bound_cdf(conditional_cdf_[row], N, u1);

		T cell_w = T(2) * radius_ / T(N);
		T cell_h = T(2) * radius_ / T(N);

		// Intra-cell interpolation (pbrt-v4 PiecewiseConstant1D::Sample)
		// du/dv ∈ [0,1): fractional position within the selected cell
		T cdf_col_prev = (col > 0) ? conditional_cdf_[row][col - 1] : T(0);
		T cdf_col_curr = conditional_cdf_[row][col];
		T du = (cdf_col_curr > cdf_col_prev)
				   ? (u1 - cdf_col_prev) / (cdf_col_curr - cdf_col_prev)
				   : T(0.5);

		T cdf_row_prev = (row > 0) ? marginal_cdf_[row - 1] : T(0);
		T cdf_row_curr = marginal_cdf_[row];
		T dv = (cdf_row_curr > cdf_row_prev)
				   ? (u2 - cdf_row_prev) / (cdf_row_curr - cdf_row_prev)
				   : T(0.5);

		// Continuous position: pbrt-v4 Lerp((o + du) / size(), min, max)
		T px = -radius_ + (T(col) + du) * cell_w;
		T py = -radius_ + (T(row) + dv) * cell_h;

		// weight = f(p) / pdf(p) = f_cell / (f_cell / integral) = integral
		// (constant for importance sampling from pdf ∝ f, same as pbrt-v4)
		T fval = f_[row][col];
		T pdf  = pdf_[row][col];
		T w = (pdf > T(1e-14)) ? fval / pdf : T(0);

		FilterSample<T> s;
		s.p_x    = px;
		s.p_y    = py;
		s.weight = w;
		return s;
	}

	// PDF for a continuous position p in filter domain.
	CPU_GPU T pdf(T px, T py) const {
		// Map to grid cell
		T cell_w = T(2) * radius_ / T(N);
		T cell_h = T(2) * radius_ / T(N);
		int col = int((px + radius_) / cell_w);
		int row = int((py + radius_) / cell_h);
		col = (col < 0) ? 0 : (col >= N ? N-1 : col);
		row = (row < 0) ? 0 : (row >= N ? N-1 : row);
		return pdf_[row][col];
	}

	CPU_GPU T radius() const { return radius_; }

	// Integral of the filter over its domain (useful for normalisation)
	CPU_GPU T integral() const { return integral_; }

private:
	T radius_;
	T f_[N][N];           // tabulated filter values
	T pdf_[N][N];         // per-cell PDF (= f[i][j] / integral)
	T conditional_cdf_[N][N]; // CDF over columns given row i
	T marginal_cdf_[N];   // CDF over rows
	T integral_;          // sum of all f values * cell_area

	// Build tables from filter. Called once at construction.
	template <typename Filter>
	CPU_GPU void build(const Filter& filter) {
		T cell_w = T(2) * radius_ / T(N);
		T cell_h = T(2) * radius_ / T(N);
		T cell_area = cell_w * cell_h;

		// --- Step 1: tabulate filter (clamp negative to 0) ---
		T row_sum[N] = {};
		T total = T(0);
		for (int r = 0; r < N; ++r) {
			for (int c = 0; c < N; ++c) {
				T px = -radius_ + (T(c) + T(0.5)) * cell_w;
				T py = -radius_ + (T(r) + T(0.5)) * cell_h;
				T val = T(filter.evaluate(double(px), double(py)));
				f_[r][c] = (val > T(0)) ? val : T(0);
				row_sum[r] += f_[r][c];
			}
			total += row_sum[r];
		}

		integral_ = total * cell_area;

		// --- Step 2: per-cell PDF ---
		T inv_total = (total > T(1e-30)) ? T(1) / total : T(0);
		for (int r = 0; r < N; ++r)
			for (int c = 0; c < N; ++c)
				pdf_[r][c] = f_[r][c] * inv_total / cell_area;

		// --- Step 3: conditional CDFs (over columns, per row) ---
		for (int r = 0; r < N; ++r) {
			T row_total = row_sum[r];
			T inv_row = (row_total > T(1e-30)) ? T(1) / row_total : T(0);
			T running = T(0);
			for (int c = 0; c < N; ++c) {
				running += f_[r][c] * inv_row;
				conditional_cdf_[r][c] = running;
			}
			// Ensure last entry == 1 exactly
			conditional_cdf_[r][N-1] = T(1);
		}

		// --- Step 4: marginal CDF (over rows) ---
		T inv_total2 = (total > T(1e-30)) ? T(1) / total : T(0);
		T running = T(0);
		for (int r = 0; r < N; ++r) {
			running += row_sum[r] * inv_total2;
			marginal_cdf_[r] = running;
		}
		marginal_cdf_[N-1] = T(1);
	}

	// Binary search for the first index where cdf[i] >= u.
	CPU_GPU static int lower_bound_cdf(const T* cdf, int n, T u) {
		int lo = 0, hi = n - 1;
		while (lo < hi) {
			int mid = (lo + hi) / 2;
			if (cdf[mid] < u)
				lo = mid + 1;
			else
				hi = mid;
		}
		return lo;
	}
};
