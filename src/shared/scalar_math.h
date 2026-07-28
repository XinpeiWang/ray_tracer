#pragma once
// ---------------------------------------------------------------------------
// scalar_math.h -- Scalar math utilities
//
// Mirrors pbrt-v4 util/math.h (Apache-2.0), adapted for CPU-only double/float
// use. Groups all standalone scalar functions not already in compensated_float.h
// or interval.h.
//
// Components (in order):
//   Constants          -- Pi, InvPi, Infinity, MachineEpsilon
//   Basic              -- Lerp, Clamp, Mod, Sqr, Pow<n>, Radians, Degrees
//   Trig               -- SinXOverX, Sinc, WindowedSinc, SafeASin/ACos/Sqrt
//   Polynomial         -- EvaluatePolynomial (FMA-Horner, variadic)
//   Transcendental     -- FastExp, Gaussian, GaussianIntegral, SmoothStep
//   Logistic           -- Logistic, LogisticCDF, TrimmedLogistic
//   Special functions  -- ErfInv, I0, LogI0
//   Integer math       -- Log2, Log2Int, Log4Int, IsPowerOf2/4, RoundUpPow2/4
//   Searching          -- FindInterval (binary search with predicate)
//   Root finding       -- NewtonBisection (hybrid Newton + bisection)
//   Permutation        -- PermutationElement (Feistel cipher)
//
// Dependencies:
//   <cmath>, <cstdint>, <climits>, <utility>, <type_traits>
//   No other local headers required.
//
// References:
//   pbrt-v4 src/pbrt/util/math.h  (Apache-2.0)
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU
#   endif
#endif

#include <cmath>
#include <cstdint>
#include <climits>
#include <cstring>
#include <utility>
#include <type_traits>
#include <limits>

// ===========================================================================
// Internal pi constant (not exported to global scope to avoid redefinition
// conflicts with translation units that define their own Pi).
// ===========================================================================
namespace scalar_math_detail {
    static constexpr double kPi      = 3.14159265358979323846;
    static constexpr double kInvPi   = 0.31830988618379067154;
    static constexpr double kInv2Pi  = 0.15915494309189533577;
    static constexpr double kInv4Pi  = 0.07957747154594766788;
    static constexpr double kPiOver2 = 1.57079632679489661923;
    static constexpr double kPiOver4 = 0.78539816339744830961;
    static constexpr double kSqrt2   = 1.41421356237309504880;
} // namespace scalar_math_detail


// ===========================================================================
// Lerp / Clamp / Mod / Sqr
// ===========================================================================
template <typename T>
CPU_GPU constexpr T Lerp(T t, T a, T b) {
	return (T(1) - t) * a + t * b;
}

template <typename T, typename U, typename V>
CPU_GPU constexpr T Clamp(T val, U low, V high) {
	if (val < static_cast<T>(low))  return static_cast<T>(low);
	if (val > static_cast<T>(high)) return static_cast<T>(high);
	return val;
}

// Integer Mod: result has the same sign as b (matches pbrt-v4 Mod<T>)
template <typename T>
CPU_GPU T Mod(T a, T b) {
	T result = a - (a / b) * b;
	return (T)((result < 0) ? result + b : result);
}
// Float specialization: delegate to std::fmod
CPU_GPU inline float  Mod(float  a, float  b) { return std::fmod(a, b); }
CPU_GPU inline double Mod(double a, double b) { return std::fmod(a, b); }

template <typename T>
CPU_GPU constexpr T Sqr(T v) { return v * v; }

// ===========================================================================
// Radians / Degrees
// ===========================================================================
CPU_GPU inline double Radians(double deg) { return (scalar_math_detail::kPi / 180.0) * deg; }
CPU_GPU inline double Degrees(double rad) { return (180.0 / scalar_math_detail::kPi) * rad; }
CPU_GPU inline float  Radians(float  deg) { return (float(scalar_math_detail::kPi) / 180.f) * deg; }
CPU_GPU inline float  Degrees(float  rad) { return (180.f / float(scalar_math_detail::kPi)) * rad; }

// ===========================================================================
// Pow<n> -- compile-time integer exponentiation (pbrt-v4 Pow<n>)
// ===========================================================================
template <int n>
CPU_GPU constexpr float Pow(float v) {
	if constexpr (n < 0)  return 1.f / Pow<-n>(v);
	float n2 = Pow<n / 2>(v);
	return n2 * n2 * Pow<n & 1>(v);
}
template <> CPU_GPU constexpr float Pow<1>(float v) { return v; }
template <> CPU_GPU constexpr float Pow<0>(float v) { (void)v; return 1.f; }

template <int n>
CPU_GPU constexpr double Pow(double v) {
	if constexpr (n < 0)  return 1.0 / Pow<-n>(v);
	double n2 = Pow<n / 2>(v);
	return n2 * n2 * Pow<n & 1>(v);
}
template <> CPU_GPU constexpr double Pow<1>(double v) { return v; }
template <> CPU_GPU constexpr double Pow<0>(double v) { (void)v; return 1.0; }

// ===========================================================================
// Safe trig / sqrt
// ===========================================================================
CPU_GPU inline float  SafeSqrt(float  x) { return std::sqrt(x > 0.f  ? x : 0.f); }
CPU_GPU inline double SafeSqrt(double x) { return std::sqrt(x > 0.0  ? x : 0.0); }

CPU_GPU inline float  SafeASin(float  x) { return std::asin(Clamp(x, -1.f, 1.f)); }
CPU_GPU inline double SafeASin(double x) { return std::asin(Clamp(x, -1.0, 1.0)); }
CPU_GPU inline float  SafeACos(float  x) { return std::acos(Clamp(x, -1.f, 1.f)); }
CPU_GPU inline double SafeACos(double x) { return std::acos(Clamp(x, -1.0, 1.0)); }

// ===========================================================================
// SinXOverX / Sinc / WindowedSinc
// ===========================================================================
// SinXOverX(x) = sin(x)/x, with limit 1 at x=0 (stable)
// http://www.plunk.org/~hatch/rightway.html
CPU_GPU inline double SinXOverX(double x) {
	if (1.0 - x * x == 1.0) return 1.0;
	return std::sin(x) / x;
}
CPU_GPU inline float SinXOverX(float x) {
	if (1.f - x * x == 1.f) return 1.f;
	return std::sin(x) / x;
}

// Sinc(x) = SinXOverX(Pi * x)  [normalized sinc used in filter design]
CPU_GPU inline double Sinc(double x) { return SinXOverX(scalar_math_detail::kPi * x); }
CPU_GPU inline float  Sinc(float  x) { return SinXOverX(float(scalar_math_detail::kPi) * x); }

// WindowedSinc(x, radius, tau): Lanczos-windowed sinc; zero outside radius
CPU_GPU inline double WindowedSinc(double x, double radius, double tau) {
	if (std::abs(x) > radius) return 0.0;
	return Sinc(x) * Sinc(x / tau);
}
CPU_GPU inline float WindowedSinc(float x, float radius, float tau) {
	if (std::abs(x) > radius) return 0.f;
	return Sinc(x) * Sinc(x / tau);
}

// ===========================================================================
// EvaluatePolynomial -- Horner's method via FMA (pbrt-v4 EvaluatePolynomial)
// EvaluatePolynomial(t, c0, c1, ..., cn) evaluates c0 + t*(c1 + t*(...))
// ===========================================================================
template <typename Float, typename C>
CPU_GPU constexpr Float EvaluatePolynomial(Float /*t*/, C c) {
	return static_cast<Float>(c);
}
template <typename Float, typename C, typename... Args>
CPU_GPU constexpr Float EvaluatePolynomial(Float t, C c, Args... cRemaining) {
	return std::fma(t, EvaluatePolynomial(t, cRemaining...), static_cast<Float>(c));
}

// ===========================================================================
// FastExp -- ~4x faster than std::exp for float (pbrt-v4 FastExp)
// https://stackoverflow.com/a/10792321
// ===========================================================================
CPU_GPU inline float FastExp(float x) {
	// Compute x' such that e^x = 2^x'
	float xp = x * 1.442695041f;
	// Find integer and fractional components of x'
	float fxp = std::floor(xp);
	float f   = xp - fxp;
	int   i   = static_cast<int>(fxp);
	// Evaluate polynomial approximation of 2^f
	float twoToF = EvaluatePolynomial(f, 1.f, 0.695556856f, 0.226173572f, 0.0781455737f);
	// Scale 2^f by 2^i via direct exponent manipulation
	int exponent = [&]() {
		uint32_t bits;
		std::memcpy(&bits, &twoToF, sizeof(bits));
		return static_cast<int>((bits >> 23) & 0xFF) - 127;
	}() + i;
	if (exponent < -126) return 0.f;
	if (exponent >  127) return std::numeric_limits<float>::infinity();
	uint32_t bits;
	std::memcpy(&bits, &twoToF, sizeof(bits));
	bits &= 0b10000000011111111111111111111111u;
	bits |= static_cast<uint32_t>(exponent + 127) << 23;
	float result;
	std::memcpy(&result, &bits, sizeof(result));
	return result;
}

// ===========================================================================
// SmoothStep
// ===========================================================================
CPU_GPU inline double SmoothStep(double x, double a, double b) {
	if (a == b) return (x < a) ? 0.0 : 1.0;
	double t = Clamp((x - a) / (b - a), 0.0, 1.0);
	return t * t * (3.0 - 2.0 * t);
}
CPU_GPU inline float SmoothStep(float x, float a, float b) {
	if (a == b) return (x < a) ? 0.f : 1.f;
	float t = Clamp((x - a) / (b - a), 0.f, 1.f);
	return t * t * (3.f - 2.f * t);
}

// ===========================================================================
// Gaussian / GaussianIntegral
// ===========================================================================
CPU_GPU inline double Gaussian(double x, double mu = 0.0, double sigma = 1.0) {
	return 1.0 / std::sqrt(2.0 * scalar_math_detail::kPi * sigma * sigma) *
		   FastExp(float(-Sqr(x - mu) / (2.0 * sigma * sigma)));
}
CPU_GPU inline double GaussianIntegral(double x0, double x1,
										double mu = 0.0, double sigma = 1.0) {
	double sigmaRoot2 = sigma * 1.4142135623730950488; // sqrt(2)
	return 0.5 * (std::erf((mu - x0) / sigmaRoot2) -
				  std::erf((mu - x1) / sigmaRoot2));
}

// ===========================================================================
// Logistic / LogisticCDF / TrimmedLogistic
// ===========================================================================
CPU_GPU inline double Logistic(double x, double s) {
	x = std::abs(x);
	return std::exp(-x / s) / (s * Sqr(1.0 + std::exp(-x / s)));
}
CPU_GPU inline double LogisticCDF(double x, double s) {
	return 1.0 / (1.0 + std::exp(-x / s));
}
CPU_GPU inline double TrimmedLogistic(double x, double s, double a, double b) {
	return Logistic(x, s) / (LogisticCDF(b, s) - LogisticCDF(a, s));
}

// ===========================================================================
// ErfInv -- inverse error function (pbrt-v4 ErfInv, float precision)
// https://stackoverflow.com/a/49743348
// ===========================================================================
CPU_GPU inline float ErfInv(float a) {
	float p;
	float t = std::log(std::max(std::fma(a, -a, 1.f),
								std::numeric_limits<float>::min()));
	if (std::abs(t) > 6.125f) {
		p = 3.03697567e-10f;
		p = std::fma(p, t, 2.93243101e-8f);
		p = std::fma(p, t, 1.22150334e-6f);
		p = std::fma(p, t, 2.84108955e-5f);
		p = std::fma(p, t, 3.93552968e-4f);
		p = std::fma(p, t, 3.02698812e-3f);
		p = std::fma(p, t, 4.83185798e-3f);
		p = std::fma(p, t, -2.64646143e-1f);
		p = std::fma(p, t,  8.40016484e-1f);
	} else {
		p = 5.43877832e-9f;
		p = std::fma(p, t, 1.43286059e-7f);
		p = std::fma(p, t, 1.22775396e-6f);
		p = std::fma(p, t, 1.12962631e-7f);
		p = std::fma(p, t, -5.61531961e-5f);
		p = std::fma(p, t, -1.47697705e-4f);
		p = std::fma(p, t,  2.31468701e-3f);
		p = std::fma(p, t,  1.15392562e-2f);
		p = std::fma(p, t, -2.32015476e-1f);
		p = std::fma(p, t,  8.86226892e-1f);
	}
	return a * p;
}

// ===========================================================================
// I0 / LogI0 -- modified Bessel function of the first kind, order 0
// ===========================================================================
CPU_GPU inline double I0(double x) {
	double val  = 0.0;
	double x2i  = 1.0;
	int64_t ifact = 1;
	int i4 = 1;
	// I0(x) ~= Sum_i x^(2i) / (4^i (i!)^2)
	for (int i = 0; i < 10; ++i) {
		if (i > 1) ifact *= i;
		val  += x2i / (i4 * Sqr(double(ifact)));
		x2i  *= x * x;
		i4   *= 4;
	}
	return val;
}
CPU_GPU inline double LogI0(double x) {
	if (x > 12.0)
		return x + 0.5 * (-std::log(2.0 * scalar_math_detail::kPi) + std::log(1.0 / x) + 1.0 / (8.0 * x));
	return std::log(I0(x));
}

// ===========================================================================
// Integer math: Log2, Log2Int, Log4Int, IsPowerOf2/4, RoundUpPow2/4
// ===========================================================================
CPU_GPU inline double Log2(double x) {
	constexpr double invLog2 = 1.4426950408889634074;
	return std::log(x) * invLog2;
}

// Log2Int for 32-bit unsigned: returns floor(log2(v)), or 0 for v==0
CPU_GPU inline int Log2Int(uint32_t v) {
	if (v == 0) return 0;
#if defined(_MSC_VER)
	unsigned long lz = 0;
	_BitScanReverse(&lz, v);
	return static_cast<int>(lz);
#elif defined(__GNUC__) || defined(__clang__)
	return 31 - __builtin_clz(v);
#else
	int n = 0;
	if (v >= (1u << 16)) { v >>= 16; n += 16; }
	if (v >= (1u << 8))  { v >>= 8;  n += 8;  }
	if (v >= (1u << 4))  { v >>= 4;  n += 4;  }
	if (v >= (1u << 2))  { v >>= 2;  n += 2;  }
	if (v >= (1u << 1))  { n += 1; }
	return n;
#endif
}
CPU_GPU inline int Log2Int(int32_t v)  { return Log2Int(static_cast<uint32_t>(v)); }
CPU_GPU inline int Log2Int(uint64_t v) {
	if (v == 0) return 0;
#if defined(_MSC_VER) && defined(_WIN64)
	unsigned long lz = 0;
	_BitScanReverse64(&lz, v);
	return static_cast<int>(lz);
#elif defined(__GNUC__) || defined(__clang__)
	return 63 - __builtin_clzll(v);
#else
	return (v >> 32) ? (32 + Log2Int(static_cast<uint32_t>(v >> 32)))
					 : Log2Int(static_cast<uint32_t>(v));
#endif
}
CPU_GPU inline int Log2Int(int64_t v)  { return Log2Int(static_cast<uint64_t>(v)); }

// Float/double Log2Int: round-to-nearest (pbrt-v4 style using exponent bits)
CPU_GPU inline int Log2Int(float v) {
	if (v < 1.f) return -Log2Int(1.f / v);
	uint32_t bits; std::memcpy(&bits, &v, 4);
	int exp = static_cast<int>((bits >> 23) & 0xFF) - 127;
	uint32_t midsignif = 0b00000000001101010000010011110011u;
	return exp + (((bits & 0x7FFFFFu) >= midsignif) ? 1 : 0);
}
CPU_GPU inline int Log2Int(double v) {
	if (v < 1.0) return -Log2Int(1.0 / v);
	uint64_t bits; std::memcpy(&bits, &v, 8);
	int exp = static_cast<int>((bits >> 52) & 0x7FF) - 1023;
	uint64_t midsignif = 0b110101000001001111001100110011111110011101111001101ull;
	return exp + (((bits & 0x000FFFFFFFFFFFFFull) >= midsignif) ? 1 : 0);
}

template <typename T>
CPU_GPU int Log4Int(T v) { return Log2Int(v) / 2; }

template <typename T>
CPU_GPU constexpr bool IsPowerOf2(T v) { return v && !(v & (v - 1)); }

template <typename T>
CPU_GPU bool IsPowerOf4(T v) { return v == (T(1) << (2 * Log4Int(v))); }

CPU_GPU inline constexpr int32_t RoundUpPow2(int32_t v) {
	v--;
	v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
	return v + 1;
}
CPU_GPU inline constexpr int64_t RoundUpPow2(int64_t v) {
	v--;
	v |= v >> 1; v |= v >> 2; v |= v >> 4;
	v |= v >> 8; v |= v >> 16; v |= v >> 32;
	return v + 1;
}
template <typename T>
CPU_GPU T RoundUpPow4(T v) {
	return IsPowerOf4(v) ? v : (T(1) << (2 * (1 + Log4Int(v))));
}

// ===========================================================================
// FindInterval -- sorted-array binary search with predicate
// Returns largest index i in [0, sz-2] such that pred(i) is true.
// Mirrors pbrt-v4 FindInterval (util/math.h).
// ===========================================================================
template <typename Predicate>
CPU_GPU size_t FindInterval(size_t sz, const Predicate& pred) {
	using ssize_t = std::make_signed_t<size_t>;
	ssize_t size = ssize_t(sz) - 2, first = 1;
	while (size > 0) {
		size_t half = size_t(size) >> 1, middle = size_t(first) + half;
		bool result = pred(middle);
		first = result ? ssize_t(middle) + 1 : first;
		size  = result ? size - ssize_t(half) - 1 : ssize_t(half);
	}
	return size_t(Clamp(ssize_t(first) - 1, ssize_t(0), ssize_t(sz) - 2));
}

// ===========================================================================
// NewtonBisection -- hybrid Newton + bisection root finder
// f must return std::pair<Float,Float>{f(x), f'(x)}.
// Matches pbrt-v4 NewtonBisection (util/math.h).
// ===========================================================================
template <typename Float, typename Func>
CPU_GPU Float NewtonBisection(Float x0, Float x1, Func f,
							   Float xEps = Float(1e-6),
							   Float fEps = Float(1e-6)) {
	Float fx0 = f(x0).first, fx1 = f(x1).first;
	if (std::abs(fx0) < fEps) return x0;
	if (std::abs(fx1) < fEps) return x1;
	bool startIsNegative = (fx0 < Float(0));
	Float xMid = x0 + (x1 - x0) * (-fx0) / (fx1 - fx0);
	while (true) {
		if (!(x0 < xMid && xMid < x1))
			xMid = (x0 + x1) / Float(2);
		auto [fMid, dfMid] = f(xMid);
		if (startIsNegative == (fMid < Float(0))) x0 = xMid;
		else                                       x1 = xMid;
		if ((x1 - x0) < xEps || std::abs(fMid) < fEps)
			return xMid;
		xMid -= fMid / dfMid;
	}
}

// ===========================================================================
// PermutationElement -- Feistel-cipher bijection on [0, l)
// Maps index i to a permuted index for permutation seed p.
// Matches pbrt-v4 PermutationElement (util/math.h).
// ===========================================================================
CPU_GPU inline int PermutationElement(uint32_t i, uint32_t l, uint32_t p) {
	uint32_t w = l - 1;
	w |= w >> 1; w |= w >> 2; w |= w >> 4; w |= w >> 8; w |= w >> 16;
	do {
		i ^= p;            i *= 0xe170893d;
		i ^= p >> 16;
		i ^= (i & w) >> 4;
		i ^= p >> 8;       i *= 0x0929eb3f;
		i ^= p >> 23;
		i ^= (i & w) >> 1; i *= 1 | (p >> 27);
		i *= 0x6935fa69;
		i ^= (i & w) >> 11;i *= 0x74dcb303;
		i ^= (i & w) >> 2; i *= 0x9e501cc3;
		i ^= (i & w) >> 2; i *= 0xc860a3df;
		i &= w;            i ^= i >> 5;
	} while (i >= l);
	return static_cast<int>((i + p) % l);
}
