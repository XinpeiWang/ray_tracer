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
#include <iostream>
#include <limits>
#include <memory>
#include <random>


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

// Thread-safe random utilities using a thread_local PRNG.
inline double random_double() {
    // Returns a random real in [0,1).
    thread_local static std::mt19937 rng((std::random_device())());
    thread_local static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    thread_local static std::mt19937 rng((std::random_device())());
    return std::uniform_real_distribution<double>(min, max)(rng);
}

inline int random_int(int min, int max) {
    // Returns a random integer in [min,max].
    thread_local static std::mt19937 rng((std::random_device())());
    return std::uniform_int_distribution<int>(min, max)(rng);
}


// Common Headers

#include "color.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"


#endif
