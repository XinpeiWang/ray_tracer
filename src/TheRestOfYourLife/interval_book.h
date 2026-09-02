#ifndef INTERVAL_BOOK_H
#define INTERVAL_BOOK_H
//==============================================================================================
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================
//
// Renamed from interval.h to interval_book.h: src/shared/interval.h (a
// separate, newer, pbrt-v4-style port with real rounding-aware interval
// arithmetic) used to share this exact filename, resolved only by C++'s
// same-directory-first / include-path-order include rules - fragile, and a
// real risk for any #include "interval.h" issued from a file in neither
// directory (see tests/unit/math_tests.cpp, which relied on this). This
// file is the original "Ray Tracing in One Weekend" book's simple
// double-only interval, kept for src/TheRestOfYourLife/'s own book-derived
// call sites (color.h, rtweekend.h) - see src/TheRestOfYourLife/README.md.


class interval {
  public:
    double min, max;

    interval() : min(+infinity), max(-infinity) {} // Default interval is empty

    interval(double min, double max) : min(min), max(max) {}

    interval(const interval& a, const interval& b) {
        // Create the interval tightly enclosing the two input intervals.
        min = a.min <= b.min ? a.min : b.min;
        max = a.max >= b.max ? a.max : b.max;
    }

    double size() const {
        return max - min;
    }

    bool contains(double x) const {
        return min <= x && x <= max;
    }

    bool surrounds(double x) const {
        return min < x && x < max;
    }

    double clamp(double x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    interval expand(double delta) const {
        auto padding = delta/2;
        return interval(min - padding, max + padding);
    }

    static const interval empty, universe;
};

inline const interval interval::empty    = interval(+infinity, -infinity);
inline const interval interval::universe = interval(-infinity, +infinity);

inline interval operator+(const interval& ival, double displacement) {
    return interval(ival.min + displacement, ival.max + displacement);
}

inline interval operator+(double displacement, const interval& ival) {
    return ival + displacement;
}


#endif
