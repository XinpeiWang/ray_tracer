#pragma once
// ---------------------------------------------------------------------------
// compensated_float.h -- High-precision floating-point arithmetic
//
// Mirrors pbrt-v4 util/math.h (Apache-2.0):
//   CompensatedFloat        -- {value, rounding-error} pair
//   TwoProd(a, b)           -- exact product split via FMA
//   TwoSum(a, b)            -- exact sum split
//   DifferenceOfProducts    -- a*b - c*d without catastrophic cancellation
//   SumOfProducts           -- a*b + c*d without catastrophic cancellation
//   CompensatedSum<T>       -- Kahan compensated accumulator
//   InnerProduct(...)       -- accurate variadic dot product (Graillat et al.)
//
// Design rules:
//   - CPU-only, double precision by default; works with float too.
//   - No pbrt Allocator, no pstd, no CUDA deps.
//   - Plain header, no .cpp needed.
//
// References:
//   pbrt-v4 src/pbrt/util/math.h  (Apache-2.0)
//   Kahan, "Further remarks on reducing truncation errors", 1965
//   Graillat & Menissier-Morain, "Accurate summation, dot product and
//     polynomial evaluation in complex floating point arithmetic", 2008
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU inline
#   endif
#endif

#include <cmath>
#include <type_traits>

// ===========================================================================
// CompensatedFloat
//
// Represents a floating-point value v together with its exact rounding error
// err, so the true value is v + err.
//
// pbrt-v4: struct CompensatedFloat
// ===========================================================================
struct CompensatedFloat {
	CPU_GPU CompensatedFloat(double v, double err = 0.0) : v(v), err(err) {}

	CPU_GPU explicit operator float()  const { return static_cast<float> (v + err); }
	CPU_GPU explicit operator double() const { return v + err; }

	double v, err;
};

// ===========================================================================
// TwoProd -- exact product split
//
// Returns {a*b, FMA(a,b,-(a*b))} so that v + err == a*b exactly.
// pbrt-v4: CompensatedFloat TwoProd(Float a, Float b)
// ===========================================================================
CPU_GPU CompensatedFloat TwoProd(double a, double b) {
	double ab = a * b;
	return {ab, std::fma(a, b, -ab)};
}

// ===========================================================================
// TwoSum -- exact sum split
//
// Returns {a+b, rounding_error} so that v + err == a+b exactly.
// pbrt-v4: CompensatedFloat TwoSum(Float a, Float b)
// ===========================================================================
CPU_GPU CompensatedFloat TwoSum(double a, double b) {
	double s = a + b, delta = s - a;
	return {s, (a - (s - delta)) + (b - delta)};
}

// ===========================================================================
// DifferenceOfProducts -- a*b - c*d without catastrophic cancellation
//
// Uses FMA to compute the exact a*b - c*d via Kahan's algorithm:
//   cd   = c*d  (rounded)
//   diff = FMA(a, b, -cd)   (exact correction for a*b part)
//   err  = FMA(-c, d, cd)   (exact correction for -c*d part)
//   result = diff + err
//
// pbrt-v4: auto DifferenceOfProducts(Ta a, Tb b, Tc c, Td d)
// ===========================================================================
template <typename Ta, typename Tb, typename Tc, typename Td>
CPU_GPU auto DifferenceOfProducts(Ta a, Tb b, Tc c, Td d) {
	auto cd   = c * d;
	auto diff = std::fma(static_cast<double>(a), static_cast<double>(b),
						 static_cast<double>(-cd));
	auto err  = std::fma(static_cast<double>(-c), static_cast<double>(d),
						 static_cast<double>(cd));
	return diff + err;
}

// ===========================================================================
// SumOfProducts -- a*b + c*d without catastrophic cancellation
//
// pbrt-v4: auto SumOfProducts(Ta a, Tb b, Tc c, Td d)
// ===========================================================================
template <typename Ta, typename Tb, typename Tc, typename Td>
CPU_GPU auto SumOfProducts(Ta a, Tb b, Tc c, Td d) {
	auto cd  = c * d;
	auto sum = std::fma(static_cast<double>(a), static_cast<double>(b),
						static_cast<double>(cd));
	auto err = std::fma(static_cast<double>(c), static_cast<double>(d),
						static_cast<double>(-cd));
	return sum + err;
}

// ===========================================================================
// CompensatedSum<T> -- Kahan compensated running accumulator
//
// Accumulates a sequence of additions while tracking the floating-point
// compensation term c, reducing accumulated rounding error from O(n*eps)
// to O(eps) regardless of sequence length.
//
// Usage:
//   CompensatedSum<double> s;
//   for (auto x : values) s += x;
//   double result = static_cast<double>(s);
//
// pbrt-v4: template <typename Float> class CompensatedSum
// ===========================================================================
template <typename T>
class CompensatedSum {
  public:
	CPU_GPU CompensatedSum() = default;
	CPU_GPU explicit CompensatedSum(T v) : sum_(v), c_(0) {}

	CPU_GPU CompensatedSum& operator=(T v) {
		sum_ = v;
		c_   = 0;
		return *this;
	}

	// Kahan summation step: pbrt-v4 operator+=
	CPU_GPU CompensatedSum& operator+=(T v) {
		T delta   = v - c_;
		T new_sum = sum_ + delta;
		c_   = (new_sum - sum_) - delta;
		sum_ = new_sum;
		return *this;
	}

	CPU_GPU explicit operator T() const { return sum_; }

  private:
	T sum_ = T(0), c_ = T(0);
};

// ===========================================================================
// InnerProduct -- accurate variadic dot product
//
// Computes sum of a_i * b_i pairs using TwoProd + TwoSum to accumulate
// without catastrophic cancellation. Returns double (the converted sum).
//
// Matches pbrt-v4: Float InnerProduct(T... terms)
// with the recursive internal::InnerProduct using CompensatedFloat.
//
// Usage:
//   double d = InnerProduct(a0, b0, a1, b1, a2, b2);  // a0*b0+a1*b1+a2*b2
// ===========================================================================
namespace detail {

CPU_GPU CompensatedFloat inner_product_impl(double a, double b) {
	return TwoProd(a, b);
}

template <typename... T>
CPU_GPU CompensatedFloat inner_product_impl(double a, double b, T... terms) {
	CompensatedFloat ab  = TwoProd(a, b);
	CompensatedFloat tp  = inner_product_impl(terms...);
	CompensatedFloat s   = TwoSum(ab.v, tp.v);
	return {s.v, ab.err + (tp.err + s.err)};
}

} // namespace detail

template <typename... T>
CPU_GPU double InnerProduct(T... terms) {
	CompensatedFloat ip = detail::inner_product_impl(static_cast<double>(terms)...);
	return double(ip);
}
