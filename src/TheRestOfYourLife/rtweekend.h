#ifndef RTWEEKEND_H
#define RTWEEKEND_H
//==============================================================================================
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <thread>


// C++ Std Usings

using std::make_shared;
using std::shared_ptr;


// Constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;


// Utility Functions

inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

// PCG32 — Permuted Congruential Generator (O'Neill 2014)
// 16-byte state, ~3-5x faster than mt19937, better statistical quality,
// supports independent per-pixel streams via (sequence, offset) seeding.
struct PCG32 {
    uint64_t state = 0x853c49e6748fea9bULL;
    uint64_t inc   = 0xda3e39cb94b95bdbULL;

    // Advance to the stream identified by seqIndex, then skip to offset.
    void seed(uint64_t seqIndex, uint64_t offset = 0) {
        state = 0u;
        inc   = (seqIndex << 1u) | 1u;
        next_uint32();
        state += offset;
        next_uint32();
    }

    uint32_t next_uint32() {
        uint64_t old = state;
        state = old * 0x5851f42d4c957f2dULL + inc;
        uint32_t xorshifted = static_cast<uint32_t>(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = static_cast<uint32_t>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
    }

    // Returns a uniform float in [0, 1).
    double uniform_double() {
        // Multiply by 2^-32 to map [0, 2^32) -> [0, 1)
        return next_uint32() * 2.3283064365386963e-10;
    }
};

// Thread-local PCG32 — one instance per render thread, initialized once.
// Declared as a macro-free inline to ensure the compiler can inline the
// state access and avoid any function-call overhead per sample.
inline PCG32& thread_rng() {
    thread_local PCG32 rng = []() {
        PCG32 r;
        uint64_t tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
        uint64_t hw  = static_cast<uint64_t>(std::random_device{}());
        r.seed(tid ^ hw, hw);
        return r;
    }();
    return rng;
}

// Random utilities — all inlined, no virtual dispatch, no heap allocation.
inline double random_double() {
    thread_local PCG32& rng = thread_rng();
    return rng.uniform_double();
}

inline double random_double(double min, double max) {
    thread_local PCG32& rng = thread_rng();
    return min + (max - min) * rng.uniform_double();
}

inline int random_int(int min, int max) {
    thread_local PCG32& rng = thread_rng();
    return static_cast<int>(min + (max - min + 1) * rng.uniform_double());
}


// Common Headers

#include "color.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"


#endif
