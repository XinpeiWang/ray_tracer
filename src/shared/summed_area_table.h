#ifndef SUMMED_AREA_TABLE_H
#define SUMMED_AREA_TABLE_H
//==============================================================================
// summed_area_table.h -- O(1) 2D integral queries and windowed 2D sampling
//
// Mirrors pbrt-v4 SummedAreaTable and WindowedPiecewiseConstant2D
// (util/sampling.h §A.5).
//
// SummedAreaTable
//   Build:   O(N*M)
//   Integral(Bounds2f): O(1) bilinearly-interpolated prefix-sum query
//
// WindowedPiecewiseConstant2D
//   Wraps SummedAreaTable + raw function values to importance-sample a
//   2D piecewise-constant function restricted to an arbitrary Bounds2f window.
//   Sample: O(N log N) bisection along x then y
//   PDF:    O(1)
//==============================================================================

#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <optional>

// ---------------------------------------------------------------------------
// Minimal 2D geometry helpers (self-contained, no external dependencies)
// ---------------------------------------------------------------------------

struct SAT_Point2f {
	float x = 0.f, y = 0.f;
	SAT_Point2f() = default;
	SAT_Point2f(float x, float y) : x(x), y(y) {}
};

struct SAT_Bounds2f {
	SAT_Point2f pMin, pMax;
	SAT_Bounds2f() : pMin(0.f, 0.f), pMax(1.f, 1.f) {}
	SAT_Bounds2f(SAT_Point2f pMin, SAT_Point2f pMax) : pMin(pMin), pMax(pMax) {}
};

// ---------------------------------------------------------------------------
// Simple 2D array (row-major, x = column, y = row)
// ---------------------------------------------------------------------------
template <typename T>
class Array2D {
public:
	Array2D() : nx_(0), ny_(0) {}
	Array2D(int nx, int ny, T val = T{}) : nx_(nx), ny_(ny), data_(nx * ny, val) {}
	Array2D(int nx, int ny, const std::vector<T>& data) : nx_(nx), ny_(ny), data_(data) {
		assert((int)data.size() == nx * ny);
	}

	T&       operator()(int x, int y)       { return data_[y * nx_ + x]; }
	const T& operator()(int x, int y) const { return data_[y * nx_ + x]; }
	// Point-subscript removed; use operator()(x,y) instead.

	int XSize() const { return nx_; }
	int YSize() const { return ny_; }

private:
	int nx_, ny_;
	std::vector<T> data_;
};

// ---------------------------------------------------------------------------
// SummedAreaTable
//   Stores prefix sums as double for precision.
//   Lookup(x, y) bilinearly interpolates in the prefix-sum space.
//   Integral(bounds) returns the integral of the original function over
//   the given [0,1]^2-normalised bounds, divided by (nx * ny).
// ---------------------------------------------------------------------------
class SummedAreaTable {
public:
	SummedAreaTable() = default;

	// Build from a flat row-major array of non-negative function values.
	SummedAreaTable(const Array2D<float>& values) {
		int nx = values.XSize(), ny = values.YSize();
		sum_ = Array2D<double>(nx, ny);

		sum_(0, 0) = values(0, 0);
		for (int x = 1; x < nx; ++x)
			sum_(x, 0) = values(x, 0) + sum_(x - 1, 0);
		for (int y = 1; y < ny; ++y)
			sum_(0, y) = values(0, y) + sum_(0, y - 1);
		for (int y = 1; y < ny; ++y)
			for (int x = 1; x < nx; ++x)
				sum_(x, y) = values(x, y) + sum_(x - 1, y) + sum_(x, y - 1) - sum_(x - 1, y - 1);
	}

	// Integral of the function over the given normalised [0,1]^2 bounds.
	// Returns >= 0.
	float Integral(SAT_Bounds2f extent) const {
		double s = ((Lookup(extent.pMax.x, extent.pMax.y) -
					 Lookup(extent.pMin.x, extent.pMax.y)) +
					(Lookup(extent.pMin.x, extent.pMin.y) -
					 Lookup(extent.pMax.x, extent.pMin.y)));
		int nx = sum_.XSize(), ny = sum_.YSize();
		return (float)std::max(s / (nx * ny), 0.0);
	}

	bool IsEmpty() const { return sum_.XSize() == 0; }

private:
	// Bilinearly-interpolated lookup into the prefix-sum table.
	// (x, y) are in [0,1] (normalised image coordinates).
	double Lookup(float x, float y) const {
		int nx = sum_.XSize(), ny = sum_.YSize();
		x *= (float)nx;
		y *= (float)ny;
		int x0 = (int)x, y0 = (int)y;

		double v00 = LookupInt(x0,     y0);
		double v10 = LookupInt(x0 + 1, y0);
		double v01 = LookupInt(x0,     y0 + 1);
		double v11 = LookupInt(x0 + 1, y0 + 1);
		float dx = x - (int)x, dy = y - (int)y;
		return (1 - dx) * (1 - dy) * v00 + (1 - dx) * dy * v01 +
			   dx * (1 - dy) * v10 + dx * dy * v11;
	}

	// Integer-coordinate lookup with prefix-sum boundary semantics:
	//   value at (0,*) or (*,0) is 0 (SAT lower boundary is zero).
	double LookupInt(int x, int y) const {
		if (x == 0 || y == 0) return 0.0;
		int nx = sum_.XSize(), ny = sum_.YSize();
		x = std::min(x - 1, nx - 1);
		y = std::min(y - 1, ny - 1);
		return sum_(x, y);
	}

	Array2D<double> sum_;
};

// ---------------------------------------------------------------------------
// WindowedPiecewiseConstant2D
//   Importance-samples a 2D piecewise-constant function restricted to an
//   arbitrary axis-aligned window in [0,1]^2 (pbrt-v4 §A.5).
// ---------------------------------------------------------------------------
class WindowedPiecewiseConstant2D {
public:
	WindowedPiecewiseConstant2D() = default;

	// f must be nx x ny row-major (x = column, y = row), all values >= 0.
	WindowedPiecewiseConstant2D(const Array2D<float>& f)
		: sat_(f), func_(f) {}

	// Sample a point within window b using random numbers u in [0,1]^2.
	// Returns the sampled (x,y) in [0,1]^2 and sets *pdf.
	// Returns empty optional when the function is zero everywhere in b.
	std::optional<SAT_Point2f> Sample(SAT_Point2f u, SAT_Bounds2f b, float* pdf) const {
		float bInt = sat_.Integral(b);
		if (bInt == 0.f) return {};

		// Marginal CDF in x: Px(x) = integral over [b.pMin.x, x] x [b.pMin.y, b.pMax.y]
		auto Px = [&](float x) -> float {
			SAT_Bounds2f bx = b;
			bx.pMax.x = x;
			return sat_.Integral(bx) / bInt;
		};

		SAT_Point2f p;
		p.x = SampleBisection(Px, u.x, b.pMin.x, b.pMax.x, func_.XSize());

		// Conditional bounds: one column-strip wide
		int nx = func_.XSize();
		float colMin = std::floor(p.x * nx) / (float)nx;
		float colMax = std::ceil(p.x * nx)  / (float)nx;
		SAT_Bounds2f bCond(SAT_Point2f(colMin, b.pMin.y), SAT_Point2f(colMax, b.pMax.y));
		if (bCond.pMin.x == bCond.pMax.x)
			bCond.pMax.x += 1.f / (float)nx;

		float condInt = sat_.Integral(bCond);
		if (condInt == 0.f) return {};

		// Conditional CDF in y
		auto Py = [&](float y) -> float {
			SAT_Bounds2f by = bCond;
			by.pMax.y = y;
			return sat_.Integral(by) / condInt;
		};
		p.y = SampleBisection(Py, u.y, b.pMin.y, b.pMax.y, func_.YSize());

		*pdf = Eval(p) / bInt;
		return p;
	}

	// PDF of a point p sampled uniformly within window b.
	float PDF(SAT_Point2f p, SAT_Bounds2f b) const {
		float funcInt = sat_.Integral(b);
		if (funcInt == 0.f) return 0.f;
		return Eval(p) / funcInt;
	}

private:
	// Bisection search: find x in [min,max] such that CDF P(x) = u.
	// Terminates when the bracketing interval spans <= 1 cell (width 1/n).
	// Matches pbrt-v4: Clamp(Lerp(t, min, max), min, max).
	template <typename CDF>
	static float SampleBisection(CDF P, float u, float min, float max, int n) {
		while (std::ceil(n * max) - std::floor(n * min) > 1) {
			float mid = (min + max) * 0.5f;
			if (P(mid) > u)
				max = mid;
			else
				min = mid;
		}
		// Interpolate within the final bracketing cell (pbrt-v4: Clamp(Lerp(t,min,max),min,max))
		float t = (P(max) - P(min) > 0.f) ? (u - P(min)) / (P(max) - P(min)) : 0.5f;
		return std::max(min, std::min(max, min + t * (max - min)));
	}

	// Nearest-neighbour lookup of the raw function value at point p in [0,1]^2.
	float Eval(SAT_Point2f p) const {
		int nx = func_.XSize(), ny = func_.YSize();
		int ix = std::min((int)(p.x * nx), nx - 1);
		int iy = std::min((int)(p.y * ny), ny - 1);
		return func_(ix, iy);
	}

	SummedAreaTable   sat_;
	Array2D<float>    func_;
};

#endif // SUMMED_AREA_TABLE_H
