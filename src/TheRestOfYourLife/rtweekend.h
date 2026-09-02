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

#include "../../src/shared/rng.h"

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

// Thread-local RNG — one instance per render thread, seeded from thread id + hardware entropy.
inline RNG& thread_rng() {
    thread_local RNG rng = []() {
        uint64_t tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
        uint64_t hw  = static_cast<uint64_t>(std::random_device{}());
        return RNG(tid ^ hw, hw);
    }();
    return rng;
}

// Random utilities — all inlined, no virtual dispatch, no heap allocation.
inline double random_double() {
    thread_local RNG& rng = thread_rng();
    return rng.Uniform<double>();
}

inline double random_double(double min, double max) {
    thread_local RNG& rng = thread_rng();
    return min + (max - min) * rng.Uniform<double>();
}

inline int random_int(int min, int max) {
    thread_local RNG& rng = thread_rng();
    return static_cast<int>(min + (max - min + 1) * rng.Uniform<double>());
}


// Common Headers

#include "color.h"
#include "interval_book.h"
#include "ray.h"
#include "vec3.h"


#endif
