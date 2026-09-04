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

// Thread-local RNG — one instance per render thread, seeded from thread id +
// hardware entropy (this project's own pre-existing, genuinely
// non-reproducible-across-runs default). When a --seed was requested,
// camera.h's worker loop calls reseed_render_rng() (below) once per
// scanline to overwrite this initial state with a deterministic one - this
// lazy-init lambda still runs exactly once per thread either way, since a
// per-scanline reseed always follows shortly after, on the CPU-only path
// that actually cares about determinism.
inline RNG& thread_rng() {
    thread_local RNG rng = []() {
        uint64_t tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
        uint64_t hw  = static_cast<uint64_t>(std::random_device{}());
        return RNG(tid ^ hw, hw);
    }();
    return rng;
}

// --seed support (CLI/GUI, camera_t::seed - see that field's own comment,
// camera.h).
//
// Reseeding has to happen PER SCANLINE, not once per worker thread at
// startup - camera.h's worker loop pulls scanlines from a shared atomic
// counter (next_j.fetch_sub(1)), a work-stealing scheduler, so WHICH
// thread ends up rendering scanline j is itself a scheduling race, not
// reproducible run to run. Seeding by "which worker am I" (a first
// attempt at this feature, since reverted) therefore doesn't work: the
// same scanline can land on a different thread - and see a different RNG
// state - on two runs with an identical --seed, even though each
// individual thread's OWN sequence was internally deterministic. Keying
// the reseed on the scanline index j instead - a property of the WORK
// ITEM, never the worker - fixes this: every pixel in scanline j always
// sees the exact same RNG state at the start of that scanline's render,
// regardless of which thread got there. (Within one scanline, the same
// thread renders every pixel single-threaded in the same fixed order
// every run, so one reseed per scanline is enough - no per-pixel reseed
// needed.)
inline void reseed_render_rng(int64_t seed, int64_t stream) {
    const uint64_t s = static_cast<uint64_t>(seed);
    const uint64_t line = static_cast<uint64_t>(stream);
    // 0x9E3779B97F4A7C15 = golden-ratio fractional bits (fixnum hash
    // multiplier), the same constant this project's own sobol/pmj02/
    // stratified sampler seed-mixing already uses (src/shared/
    // sobol_sampler.h and others) - reused here rather than inventing a
    // second arbitrary mixing constant.
    thread_rng().SetSequence(s ^ (line * 0x9E3779B97F4A7C15ULL), s);
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
