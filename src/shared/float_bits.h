#pragma once
// ---------------------------------------------------------------------------
// float_bits.h -- IEEE 754 float/double bit-manipulation utilities
//
// Mirrors pbrt-v4 util/float.h (Apache-2.0), CPU-only adaptation.
// Provides portable, standards-compliant helpers for working directly with
// the binary representation of floating-point values.
//
// Components (in order):
//   Constants       -- OneMinusEpsilon, MachineEpsilon, Infinity
//   Classification  -- IsNaN<T>, IsInf<T>, IsFinite<T>  (float, double, integral)
//   Bit conversion  -- FloatToBits / BitsToFloat  (float <-> uint32, double <-> uint64)
//   Field access    -- Exponent, Significand, SignBit  (float and double)
//   ULP stepping    -- NextFloatUp / NextFloatDown  (float and double)
//   Sign copy       -- FlipSign(a, b)  -- return |a| with sign of b
//   Error bound     -- gamma(n)  -- interval error factor n*eps/(1 - n*eps)
//   Rounding ops    -- AddRoundUp/Down, SubRound*, MulRound*, DivRound*,
//                      SqrtRound*, FMARound*  (all via NextFloat*)
//
// Dependencies:
//   <cmath>, <cstdint>, <cstring>, <limits>, <type_traits>
//   No other local headers required.
//
// References:
//   pbrt-v4 src/pbrt/util/float.h  (Apache-2.0)
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
#include <cstring>
#include <limits>
#include <algorithm>
#include <type_traits>

// ===========================================================================
// Floating-point constants
// ===========================================================================
static constexpr float  FloatOneMinusEpsilon = 0x1.fffffep-1f;
static constexpr double DoubleOneMinusEpsilon = 0x1.fffffffffffffp-1;
// OneMinusEpsilon: largest float strictly less than 1 -- used to clamp
// sample values into [0, 1) without reaching 1.0 exactly.
static constexpr float  OneMinusEpsilon = FloatOneMinusEpsilon;

// MachineEpsilon = eps/2 = half the distance between 1.0 and the next float.
// Interval arithmetic uses this as the per-operation rounding bound.
static constexpr float  FloatMachineEpsilon  = std::numeric_limits<float>::epsilon()  * 0.5f;
static constexpr double DoubleMachineEpsilon = std::numeric_limits<double>::epsilon() * 0.5;

// ===========================================================================
// IsNaN / IsInf / IsFinite -- templated, integral overloads always false/true
// Mirrors pbrt-v4 float.h templated classification functions.
// ===========================================================================
template <typename T>
CPU_GPU typename std::enable_if_t<std::is_floating_point_v<T>, bool>
IsNaN(T v) { return std::isnan(v); }

template <typename T>
CPU_GPU typename std::enable_if_t<std::is_integral_v<T>, bool>
IsNaN(T) { return false; }

template <typename T>
CPU_GPU typename std::enable_if_t<std::is_floating_point_v<T>, bool>
IsInf(T v) { return std::isinf(v); }

template <typename T>
CPU_GPU typename std::enable_if_t<std::is_integral_v<T>, bool>
IsInf(T) { return false; }

template <typename T>
CPU_GPU typename std::enable_if_t<std::is_floating_point_v<T>, bool>
IsFinite(T v) { return std::isfinite(v); }

template <typename T>
CPU_GPU typename std::enable_if_t<std::is_integral_v<T>, bool>
IsFinite(T) { return true; }

// ===========================================================================
// FMA -- fused multiply-add, matches pbrt-v4 float.h FMA overloads
// ===========================================================================
CPU_GPU float       FMA(float a,       float b,       float c)       { return std::fma(a, b, c); }
CPU_GPU double      FMA(double a,      double b,      double c)      { return std::fma(a, b, c); }
CPU_GPU long double FMA(long double a, long double b, long double c) { return std::fma(a, b, c); }

// ===========================================================================
// FloatToBits / BitsToFloat -- type-punning via memcpy (C++17 compliant)
// Mirrors pbrt-v4 FloatToBits/BitsToFloat (pstd::bit_cast on CPU).
// ===========================================================================
CPU_GPU uint32_t FloatToBits(float f) {
	uint32_t bits;
	std::memcpy(&bits, &f, sizeof(bits));
	return bits;
}

CPU_GPU float BitsToFloat(uint32_t bits) {
	float f;
	std::memcpy(&f, &bits, sizeof(f));
	return f;
}

CPU_GPU uint64_t FloatToBits(double d) {
	uint64_t bits;
	std::memcpy(&bits, &d, sizeof(bits));
	return bits;
}

CPU_GPU double BitsToFloat(uint64_t bits) {
	double d;
	std::memcpy(&d, &bits, sizeof(d));
	return d;
}

// ===========================================================================
// Exponent / Significand / SignBit -- IEEE 754 field extractors
// float:  biased exponent = bits[30:23], significand = bits[22:0]
// double: biased exponent = bits[62:52], significand = bits[51:0]
// ===========================================================================
CPU_GPU int      Exponent(float v)    { return (int)((FloatToBits(v) >> 23) & 0xFF) - 127; }
CPU_GPU int      Significand(float v) { return (int)(FloatToBits(v) & ((1u << 23) - 1u)); }
CPU_GPU uint32_t SignBit(float v)     { return FloatToBits(v) & 0x80000000u; }

CPU_GPU int      Exponent(double v)    { return (int)((FloatToBits(v) >> 52) & 0x7FF) - 1023; }
CPU_GPU uint64_t Significand(double v) { return FloatToBits(v) & ((1ull << 52) - 1ull); }
CPU_GPU uint64_t SignBit(double v)     { return FloatToBits(v) & 0x8000000000000000ull; }

// ===========================================================================
// NextFloatUp / NextFloatDown -- ULP stepping
// Advances/retreats a float by one unit in the last place.
// Handles Â±0, Â±Inf, and NaN-preserving behaviour.
// Mirrors pbrt-v4 NextFloatUp/NextFloatDown.
// ===========================================================================
CPU_GPU float NextFloatUp(float v) {
	if (std::isinf(v) && v > 0.f) return v;
	if (v == -0.f) v = 0.f;
	uint32_t bits = FloatToBits(v);
	if (v >= 0.f) ++bits; else --bits;
	return BitsToFloat(bits);
}

CPU_GPU float NextFloatDown(float v) {
	if (std::isinf(v) && v < 0.f) return v;
	if (v == 0.f) v = -0.f;
	uint32_t bits = FloatToBits(v);
	if (v > 0.f) --bits; else ++bits;
	return BitsToFloat(bits);
}

CPU_GPU double NextFloatUp(double v) {
	if (std::isinf(v) && v > 0.) return v;
	if (v == -0.) v = 0.;
	uint64_t bits = FloatToBits(v);
	if (v >= 0.) ++bits; else --bits;
	return BitsToFloat(bits);
}

CPU_GPU double NextFloatDown(double v) {
	if (std::isinf(v) && v < 0.) return v;
	if (v == 0.) v = -0.;
	uint64_t bits = FloatToBits(v);
	if (v > 0.) --bits; else ++bits;
	return BitsToFloat(bits);
}

// ===========================================================================
// FlipSign(a, b) -- XOR a's sign bit with b's sign bit
// If b is negative (sign=1), flips a's sign. If b is positive (sign=0), a is unchanged.
// Mirrors pbrt-v4 FlipSign (util/float.h, double version).
// We also provide a float overload (pbrt-v4 only has double).
// ===========================================================================
CPU_GPU double FlipSign(double a, double b) {
	return BitsToFloat(FloatToBits(a) ^ SignBit(b));
}
CPU_GPU float FlipSign(float a, float b) {
	return BitsToFloat(FloatToBits(a) ^ SignBit(b));
}

// ===========================================================================
// gamma(n) -- interval arithmetic error bound factor
// gamma(n) = (n * eps) / (1 - n * eps) where eps = MachineEpsilon (float).
// Used by interval-arithmetic error bounds (pbrt-v4 gamma function).
// ===========================================================================
CPU_GPU constexpr float gamma(int n) {
	return (n * FloatMachineEpsilon) / (1.f - n * FloatMachineEpsilon);
}

// ===========================================================================
// Rounding-mode arithmetic (conservative interval bounds)
// On CPU, these use NextFloat* as a portable conservative rounding wrapper.
// On GPU, hardware rounding-mode intrinsics would be used instead.
// Mirrors pbrt-v4 AddRoundUp/Down, MulRound*, DivRound*, SqrtRound*,
// FMARound*, SubRound* (util/float.h).
// Template parameter Float must be float or double.
// ===========================================================================
template <typename Float>
CPU_GPU Float AddRoundUp(Float a, Float b)   { return NextFloatUp(a + b); }
template <typename Float>
CPU_GPU Float AddRoundDown(Float a, Float b) { return NextFloatDown(a + b); }

template <typename Float>
CPU_GPU Float SubRoundUp(Float a, Float b)   { return AddRoundUp(a, -b); }
template <typename Float>
CPU_GPU Float SubRoundDown(Float a, Float b) { return AddRoundDown(a, -b); }

template <typename Float>
CPU_GPU Float MulRoundUp(Float a, Float b)   { return NextFloatUp(a * b); }
template <typename Float>
CPU_GPU Float MulRoundDown(Float a, Float b) { return NextFloatDown(a * b); }

template <typename Float>
CPU_GPU Float DivRoundUp(Float a, Float b)   { return NextFloatUp(a / b); }
template <typename Float>
CPU_GPU Float DivRoundDown(Float a, Float b) { return NextFloatDown(a / b); }

template <typename Float>
CPU_GPU Float SqrtRoundUp(Float a)   { return NextFloatUp(std::sqrt(a)); }
template <typename Float>
CPU_GPU Float SqrtRoundDown(Float a) {
	return std::max(Float(0), NextFloatDown(std::sqrt(a)));
}

template <typename Float>
CPU_GPU Float FMARoundUp(Float a, Float b, Float c)   { return NextFloatUp(std::fma(a, b, c)); }
template <typename Float>
CPU_GPU Float FMARoundDown(Float a, Float b, Float c) { return NextFloatDown(std::fma(a, b, c)); }
