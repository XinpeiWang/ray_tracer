#pragma once
// ---------------------------------------------------------------------------
// interval.h -- Conservative floating-point interval arithmetic
//
// Mirrors pbrt-v4 Interval class (util/math.h) and its rounding primitives
// (util/float.h), adapted for double-precision CPU-only use.
//
// Components:
//   next_float_up / next_float_down  -- ULP-step helpers (pbrt-v4 float.h)
//   Rounding primitives              -- add/sub/mul/div/sqrt round-up/down
//   Interval                         -- conservative [low, high] interval
//   Free functions                   -- InRange, Sqrt, Sqr, Abs, ACos, Sin,
//                                       Cos, FMA, MulPow2, Quadratic,
//                                       SumSquares
//
// Design rules:
//   - CPU-only, double precision throughout
//   - No pbrt Allocator, no pstd, no CUDA
//   - Plain header, no .cpp needed
//
// References:
//   pbrt-v4 src/pbrt/util/float.h  (Apache-2.0)
//   pbrt-v4 src/pbrt/util/math.h   (Apache-2.0)
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU
#   endif
#endif

#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

// ===========================================================================
// ULP-step helpers (pbrt-v4: NextFloatUp / NextFloatDown for double)
// ===========================================================================

CPU_GPU double next_float_up(double v) {
	if (std::isinf(v) && v > 0.0) return v;
	if (v == -0.0) v = 0.0;
	uint64_t ui;
	std::memcpy(&ui, &v, sizeof ui);
	if (v >= 0.0) ++ui; else --ui;
	double r;
	std::memcpy(&r, &ui, sizeof r);
	return r;
}

CPU_GPU double next_float_down(double v) {
	if (std::isinf(v) && v < 0.0) return v;
	if (v == 0.0) v = -0.0;
	uint64_t ui;
	std::memcpy(&ui, &v, sizeof ui);
	if (v > 0.0) --ui; else ++ui;
	double r;
	std::memcpy(&r, &ui, sizeof r);
	return r;
}

// ===========================================================================
// Directed-rounding arithmetic primitives (pbrt-v4: float.h, CPU path)
// On CPU pbrt-v4 uses NextFloatUp/Down(a op b); we mirror that exactly.
// ===========================================================================

CPU_GPU double add_round_up  (double a, double b) { return next_float_up  (a + b); }
CPU_GPU double add_round_down(double a, double b) { return next_float_down(a + b); }
CPU_GPU double sub_round_up  (double a, double b) { return add_round_up  (a, -b); }
CPU_GPU double sub_round_down(double a, double b) { return add_round_down(a, -b); }
CPU_GPU double mul_round_up  (double a, double b) { return next_float_up  (a * b); }
CPU_GPU double mul_round_down(double a, double b) { return next_float_down(a * b); }
CPU_GPU double div_round_up  (double a, double b) { return next_float_up  (a / b); }
CPU_GPU double div_round_down(double a, double b) { return next_float_down(a / b); }
CPU_GPU double sqrt_round_up (double a)           { return next_float_up  (std::sqrt(a)); }
CPU_GPU double sqrt_round_down(double a)          { return std::max(0.0, next_float_down(std::sqrt(a))); }
CPU_GPU double fma_round_up  (double a, double b, double c) { return next_float_up  (std::fma(a, b, c)); }
CPU_GPU double fma_round_down(double a, double b, double c) { return next_float_down(std::fma(a, b, c)); }

// ===========================================================================
// Interval
//
// Conservative double-precision interval [low, high].
// All arithmetic is correctly rounded outward so the true result is always
// contained within the returned interval.
//
// Mirrors pbrt-v4 Interval (util/math.h).
// ===========================================================================
class Interval {
  public:
	// Constructors
	CPU_GPU Interval() : low_(0.0), high_(0.0) {}
	CPU_GPU explicit Interval(double v) : low_(v), high_(v) {}
	CPU_GPU constexpr Interval(double low, double high)
		: low_(low < high ? low : high), high_(low < high ? high : low) {}

	// pbrt-v4: Interval::FromValueAndError(v, err)
	CPU_GPU static Interval FromValueAndError(double v, double err) {
		Interval i;
		if (err == 0.0) {
			i.low_ = i.high_ = v;
		} else {
			i.low_  = sub_round_down(v, err);
			i.high_ = add_round_up  (v, err);
		}
		return i;
	}

	CPU_GPU Interval& operator=(double v) { low_ = high_ = v; return *this; }

	// Accessors
	CPU_GPU double LowerBound() const { return low_; }
	CPU_GPU double UpperBound() const { return high_; }
	CPU_GPU double Midpoint()   const { return (low_ + high_) * 0.5; }
	CPU_GPU double Width()      const { return high_ - low_; }

	CPU_GPU double operator[](int i) const { return (i == 0) ? low_ : high_; }
	CPU_GPU explicit operator double() const { return Midpoint(); }

	CPU_GPU bool Exactly(double v) const { return low_ == v && high_ == v; }
	CPU_GPU bool operator==(double v)    const { return Exactly(v); }
	CPU_GPU bool operator==(Interval i)  const { return low_ == i.low_ && high_ == i.high_; }
	CPU_GPU bool operator!=(double f)    const { return f < low_ || f > high_; }

	// Unary negation: [-high, -low]
	CPU_GPU Interval operator-() const { return {-high_, -low_}; }

	// Binary interval arithmetic (correctly rounded outward)
	CPU_GPU Interval operator+(Interval i) const {
		return {add_round_down(low_, i.low_), add_round_up(high_, i.high_)};
	}
	CPU_GPU Interval operator-(Interval i) const {
		return {sub_round_down(low_, i.high_), sub_round_up(high_, i.low_)};
	}
	CPU_GPU Interval operator*(Interval i) const {
		double lp[4] = {mul_round_down(low_,  i.low_),  mul_round_down(high_, i.low_),
						mul_round_down(low_,  i.high_), mul_round_down(high_, i.high_)};
		double hp[4] = {mul_round_up  (low_,  i.low_),  mul_round_up  (high_, i.low_),
						mul_round_up  (low_,  i.high_), mul_round_up  (high_, i.high_)};
		return {std::min({lp[0], lp[1], lp[2], lp[3]}),
				std::max({hp[0], hp[1], hp[2], hp[3]})};
	}
	CPU_GPU Interval operator/(Interval i) const;  // defined below (needs InRange)

	// Compound assignment
	CPU_GPU Interval& operator+=(Interval i) { *this = *this + i; return *this; }
	CPU_GPU Interval& operator-=(Interval i) { *this = *this - i; return *this; }
	CPU_GPU Interval& operator*=(Interval i) { *this = *this * i; return *this; }
	CPU_GPU Interval& operator/=(Interval i) { *this = *this / i; return *this; }

	CPU_GPU Interval& operator+=(double f) { return *this += Interval(f); }
	CPU_GPU Interval& operator-=(double f) { return *this -= Interval(f); }
	CPU_GPU Interval& operator*=(double f) {
		if (f > 0) *this = Interval(mul_round_down(f, low_), mul_round_up(f, high_));
		else       *this = Interval(mul_round_down(f, high_), mul_round_up(f, low_));
		return *this;
	}
	CPU_GPU Interval& operator/=(double f) {
		if (f > 0) *this = Interval(div_round_down(low_, f), div_round_up(high_, f));
		else       *this = Interval(div_round_down(high_, f), div_round_up(low_, f));
		return *this;
	}

	// Interval::Pi (exact double bounds around Ãâ‚¬)
	static const Interval Pi;

  private:
	double low_, high_;
};

// Initialization of static Pi (pbrt-v4: 3.1415926535897931, 3.1415926535897936)
// These are the two consecutive doubles bracketing Ãâ‚¬.
inline const Interval Interval::Pi{3.1415926535897931, 3.1415926535897936};

// ===========================================================================
// Interval free functions
// ===========================================================================

// InRange -- does value / interval overlap another interval?
CPU_GPU bool InRange(double v, Interval i) {
	return v >= i.LowerBound() && v <= i.UpperBound();
}
CPU_GPU bool InRange(Interval a, Interval b) {
	return a.LowerBound() <= b.UpperBound() && a.UpperBound() >= b.LowerBound();
}

// Division -- defined out-of-line to use InRange
CPU_GPU Interval Interval::operator/(Interval i) const {
	if (InRange(0.0, i))
		return Interval(-std::numeric_limits<double>::infinity(),
						 std::numeric_limits<double>::infinity());
	double lq[4] = {div_round_down(low_,  i.low_), div_round_down(high_, i.low_),
					div_round_down(low_,  i.high_), div_round_down(high_, i.high_)};
	double hq[4] = {div_round_up  (low_,  i.low_), div_round_up  (high_, i.low_),
					div_round_up  (low_,  i.high_), div_round_up  (high_, i.high_)};
	return {std::min({lq[0], lq[1], lq[2], lq[3]}),
			std::max({hq[0], hq[1], hq[2], hq[3]})};
}

// Scalar-on-left operators
CPU_GPU Interval operator+(double f, Interval i) { return Interval(f) + i; }
CPU_GPU Interval operator-(double f, Interval i) { return Interval(f) - i; }
CPU_GPU Interval operator*(double f, Interval i) {
	if (f > 0) return Interval(mul_round_down(f, i.LowerBound()), mul_round_up(f, i.UpperBound()));
	else       return Interval(mul_round_down(f, i.UpperBound()), mul_round_up(f, i.LowerBound()));
}
CPU_GPU Interval operator/(double f, Interval i) {
	if (InRange(0.0, i))
		return Interval(-std::numeric_limits<double>::infinity(),
						 std::numeric_limits<double>::infinity());
	if (f > 0) return Interval(div_round_down(f, i.UpperBound()), div_round_up(f, i.LowerBound()));
	else       return Interval(div_round_down(f, i.LowerBound()), div_round_up(f, i.UpperBound()));
}
CPU_GPU Interval operator+(Interval i, double f) { return i + Interval(f); }
CPU_GPU Interval operator-(Interval i, double f) { return i - Interval(f); }
CPU_GPU Interval operator*(Interval i, double f) {
	if (f > 0) return Interval(mul_round_down(f, i.LowerBound()), mul_round_up(f, i.UpperBound()));
	else       return Interval(mul_round_down(f, i.UpperBound()), mul_round_up(f, i.LowerBound()));
}
CPU_GPU Interval operator/(Interval i, double f) {
	if (f > 0) return Interval(div_round_down(i.LowerBound(), f), div_round_up(i.UpperBound(), f));
	else       return Interval(div_round_down(i.UpperBound(), f), div_round_up(i.LowerBound(), f));
}

// Sqrt -- pbrt-v4: {SqrtRoundDown(low), SqrtRoundUp(high)}
CPU_GPU Interval Sqrt(Interval i) {
	return {sqrt_round_down(i.LowerBound()), sqrt_round_up(i.UpperBound())};
}
CPU_GPU Interval sqrt(Interval i) { return Sqrt(i); }

// Sqr -- pbrt-v4: Sqr(Interval)
CPU_GPU Interval Sqr(Interval i) {
	double alow  = std::abs(i.LowerBound());
	double ahigh = std::abs(i.UpperBound());
	if (alow > ahigh) std::swap(alow, ahigh);
	if (InRange(0.0, i)) return Interval(0.0, mul_round_up(ahigh, ahigh));
	return Interval(mul_round_down(alow, alow), mul_round_up(ahigh, ahigh));
}

// Abs -- pbrt-v4: Abs(Interval)
CPU_GPU Interval Abs(Interval i) {
	if (i.LowerBound() >= 0.0) return i;
	if (i.UpperBound() <= 0.0) return Interval(-i.UpperBound(), -i.LowerBound());
	return Interval(0.0, std::max(-i.LowerBound(), i.UpperBound()));
}
CPU_GPU Interval abs(Interval i) { return Abs(i); }

// FMA -- pbrt-v4: FMA(Interval, Interval, Interval)
CPU_GPU Interval FMA(Interval a, Interval b, Interval c) {
	double low = std::min({fma_round_down(a.LowerBound(), b.LowerBound(), c.LowerBound()),
						   fma_round_down(a.UpperBound(), b.LowerBound(), c.LowerBound()),
						   fma_round_down(a.LowerBound(), b.UpperBound(), c.LowerBound()),
						   fma_round_down(a.UpperBound(), b.UpperBound(), c.LowerBound())});
	double high = std::max({fma_round_up(a.LowerBound(), b.LowerBound(), c.UpperBound()),
							fma_round_up(a.UpperBound(), b.LowerBound(), c.UpperBound()),
							fma_round_up(a.LowerBound(), b.UpperBound(), c.UpperBound()),
							fma_round_up(a.UpperBound(), b.UpperBound(), c.UpperBound())});
	return Interval(low, high);
}

// MulPow2 -- exact (power-of-2 scaling has no rounding error)
CPU_GPU Interval MulPow2(double s, Interval i) {
	return Interval(std::min(i.LowerBound() * s, i.UpperBound() * s),
					std::max(i.LowerBound() * s, i.UpperBound() * s));
}
CPU_GPU Interval MulPow2(Interval i, double s) { return MulPow2(s, i); }

// ACos -- pbrt-v4: ACos(Interval)
CPU_GPU Interval ACos(Interval i) {
	double low  = std::acos(std::min(1.0, i.UpperBound()));
	double high = std::acos(std::max(-1.0, i.LowerBound()));
	return Interval(std::max(0.0, next_float_down(low)), next_float_up(high));
}

// Sin -- restricted to [0, 2Ãâ‚¬] per pbrt-v4 contract
CPU_GPU Interval Sin(Interval i) {
	static const double kTwoPi = 2.0 * 3.14159265358979323846;
	double low  = std::sin(std::max(0.0, i.LowerBound()));
	double high = std::sin(i.UpperBound());
	if (low > high) std::swap(low, high);
	low  = std::max(-1.0, next_float_down(low));
	high = std::min( 1.0, next_float_up  (high));
	// Peak at Ãâ‚¬/2
	if (InRange(3.14159265358979323846 * 0.5, i)) high = 1.0;
	// Trough at 3Ãâ‚¬/2
	if (InRange(3.14159265358979323846 * 1.5, i)) low  = -1.0;
	return Interval(low, high);
}

// Cos -- restricted to [0, 2Ãâ‚¬] per pbrt-v4 contract
CPU_GPU Interval Cos(Interval i) {
	double low  = std::cos(std::max(0.0, i.LowerBound()));
	double high = std::cos(i.UpperBound());
	if (low > high) std::swap(low, high);
	low  = std::max(-1.0, next_float_down(low));
	high = std::min( 1.0, next_float_up  (high));
	// Trough at Ãâ‚¬
	if (InRange(3.14159265358979323846, i)) low = -1.0;
	return Interval(low, high);
}

// SumSquares -- pbrt-v4 variadic SumSquares
CPU_GPU Interval SumSquares(Interval i) { return Sqr(i); }

template <typename... Args>
CPU_GPU Interval SumSquares(Interval i, Args... args) {
	Interval ss = FMA(i, i, SumSquares(args...));
	return Interval(std::max(0.0, ss.LowerBound()), ss.UpperBound());
}

// Quadratic solver over Interval -- pbrt-v4: Quadratic(Interval a, b, c, t0*, t1*)
// Solves a*t^2 + b*t + c = 0 conservatively.
CPU_GPU bool Quadratic(Interval a, Interval b, Interval c,
							   Interval* t0, Interval* t1) {
	Interval discrim = FMA(b, b, -MulPow2(4.0, a * c));
	if (discrim.LowerBound() < 0.0) return false;
	Interval root_discrim = Sqrt(discrim);
	Interval q;
	if ((double)b < 0.0) q = MulPow2(-0.5, b - root_discrim);
	else                 q = MulPow2(-0.5, b + root_discrim);
	*t0 = q / a;
	*t1 = c / q;
	if (t0->LowerBound() > t1->LowerBound()) std::swap(*t0, *t1);
	return true;
}
