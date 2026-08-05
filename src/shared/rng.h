// rng.h -- Canonical PCG32 RNG, ported from pbrt-v4 util/rng.h
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// Apache License, Version 2.0.
//
// Provides a single RNG class backed by PCG32 with the full pbrt-v4 API:
//   Uniform<uint32_t/uint64_t/int32_t/int64_t/float/double>()
//   Uniform<integral T>(T bound)   -- unbiased bounded integer in [0, bound)
//   SetSequence(seqIndex [, seed])
//   Advance(int64_t delta)          -- skip forward (positive) or backward
//   operator-(const RNG&)           -- distance between two same-stream RNGs
//
// Depends on: float_bits.h (OneMinusEpsilon), pbrt_hash.h (MixBits)

#pragma once

#include "float_bits.h"
#include "pbrt_hash.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

#ifndef CPU_GPU
#  ifdef __CUDACC__
#    define CPU_GPU __host__ __device__
#  else
#    define CPU_GPU inline
#  endif
#endif

// PCG32 constants
static constexpr uint64_t PCG32_DEFAULT_STATE  = 0x853c49e6748fea9bULL;
static constexpr uint64_t PCG32_DEFAULT_STREAM = 0xda3e39cb94b95bdbULL;
static constexpr uint64_t PCG32_MULT           = 0x5851f42d4c957f2dULL;

// RNG -- PCG32 random number generator (pbrt-v4 util/rng.h)
class RNG {
  public:
	CPU_GPU RNG() : state(PCG32_DEFAULT_STATE), inc(PCG32_DEFAULT_STREAM) {}
	CPU_GPU RNG(uint64_t seqIndex, uint64_t offset) { SetSequence(seqIndex, offset); }
	CPU_GPU explicit RNG(uint64_t seqIndex)          { SetSequence(seqIndex); }

	// Seed the generator to stream seqIndex, advanced to offset.
	CPU_GPU void SetSequence(uint64_t sequenceIndex, uint64_t seed) {
		state = 0u;
		inc   = (sequenceIndex << 1u) | 1u;
		Step_();
		state += seed;
		Step_();
	}

	// Seed with seed derived from MixBits(seqIndex).
	CPU_GPU void SetSequence(uint64_t sequenceIndex) {
		SetSequence(sequenceIndex, MixBits(sequenceIndex));
	}

	// Uniform<T>() -- sample a uniformly distributed value of type T.
	// Supported: uint32_t, uint64_t, int32_t, int64_t, float, double.
	template <typename T>
	CPU_GPU T Uniform() {
		static_assert(
			std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> ||
			std::is_same_v<T, int32_t>  || std::is_same_v<T, int64_t>  ||
			std::is_same_v<T, float>    || std::is_same_v<T, double>,
			"RNG::Uniform<T>: unsupported type T");

		if constexpr (std::is_same_v<T, uint32_t>) {
			return Step_();
		} else if constexpr (std::is_same_v<T, uint64_t>) {
			uint64_t v0 = Step_(), v1 = Step_();
			return (v0 << 32) | v1;
		} else if constexpr (std::is_same_v<T, int32_t>) {
			uint32_t v = Step_();
			if (v <= (uint32_t)std::numeric_limits<int32_t>::max())
				return (int32_t)v;
			return (int32_t)(v - (uint32_t)std::numeric_limits<int32_t>::min()) +
				   std::numeric_limits<int32_t>::min();
		} else if constexpr (std::is_same_v<T, int64_t>) {
			uint64_t v0 = Step_(), v1 = Step_();
			uint64_t v  = (v0 << 32) | v1;
			if (v <= (uint64_t)std::numeric_limits<int64_t>::max())
				return (int64_t)v;
			return (int64_t)(v - (uint64_t)std::numeric_limits<int64_t>::min()) +
				   std::numeric_limits<int64_t>::min();
		} else if constexpr (std::is_same_v<T, float>) {
			// Map to [0, 1); clamp to OneMinusEpsilon (largest float < 1).
			float candidate = Step_() * 0x1p-32f;
			return candidate < (float)OneMinusEpsilon ? candidate : (float)OneMinusEpsilon;
		} else { // double
			uint64_t v0 = Step_(), v1 = Step_();
			uint64_t bits = (v0 << 32) | v1;
			double candidate = bits * 0x1p-64;
			return candidate < DoubleOneMinusEpsilon ? candidate : DoubleOneMinusEpsilon;
		}
	}

	// Uniform(T bound) -- unbiased uniform integer in [0, bound).
	// Enabled only for integral types.
	template <typename T>
	CPU_GPU typename std::enable_if_t<std::is_integral_v<T>, T> Uniform(T bound) {
		T threshold = (~bound + 1u) % bound;
		while (true) {
			T r = Uniform<T>();
			if (r >= threshold)
				return r % bound;
		}
	}

	// Advance the state forward (positive) or backward (negative) by idelta steps.
	CPU_GPU void Advance(int64_t idelta) {
		uint64_t curMult = PCG32_MULT, curPlus = inc;
		uint64_t accMult = 1u, accPlus = 0u;
		uint64_t delta = (uint64_t)idelta;
		while (delta > 0) {
			if (delta & 1u) {
				accMult *= curMult;
				accPlus = accPlus * curMult + curPlus;
			}
			curPlus = (curMult + 1u) * curPlus;
			curMult *= curMult;
			delta >>= 1;
		}
		state = accMult * state + accPlus;
	}

	// Number of steps from other to *this (streams must match).
	CPU_GPU int64_t operator-(const RNG& other) const {
		uint64_t curMult = PCG32_MULT, curPlus = inc;
		uint64_t curState = other.state;
		uint64_t theBit = 1u, distance = 0u;
		while (state != curState) {
			if ((state & theBit) != (curState & theBit)) {
				curState = curState * curMult + curPlus;
				distance |= theBit;
			}
			theBit <<= 1;
			curPlus = (curMult + 1ULL) * curPlus;
			curMult *= curMult;
		}
		return (int64_t)distance;
	}

	// -----------------------------------------------------------------------
	// Backward-compatibility API (matches the old inline PCG32 struct)
	// -----------------------------------------------------------------------
	CPU_GPU void seed(uint64_t seqIndex, uint64_t offset = 0) {
		SetSequence(seqIndex, offset);
	}
	CPU_GPU uint32_t next_uint32() { return Step_(); }
	CPU_GPU double   uniform_double() { return Uniform<double>(); }

  private:
	// Core PCG32 output step.
	CPU_GPU uint32_t Step_() {
		uint64_t oldstate = state;
		state = oldstate * PCG32_MULT + inc;
		uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
		uint32_t rot = (uint32_t)(oldstate >> 59u);
		return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
	}

	uint64_t state, inc;
};

// PCG32 is an alias for RNG, provided for backward compatibility.
using PCG32 = RNG;
