#ifndef PERLIN_H
#define PERLIN_H
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//
// pbrt-v4 upgrades (util/noise.cpp):
//   - Quintic C2 interpolation weight: 6t^5 - 15t^4 + 10t^3  (was cubic 3t^2-2t^3)
//   - Fixed permutation table identical to pbrt-v4 (no random re-seeding)
//   - noise() delegates to shared perlin_noise<double> from noise.h
//   - turb() delegates to shared turbulence_simple<double>
//   - fbm()  delegates to shared fbm_simple<double>
//==============================================================================================

#include "../shared/noise.h"


class perlin {
  public:
    perlin() {}  // stateless: permutation table is in shared noise.h

    // Scalar gradient noise in [-1, 1].
    // Uses pbrt-v4 fixed permutation table + quintic C2 weight.
    double noise(const point3& p) const {
        return perlin_noise<double>(p.x(), p.y(), p.z());
    }

    // Turbulence: sum of |noise| octaves.
    // Delegates to shared turbulence_simple<double> (no antialiasing).
    // omega=0.5 reproduces Book-3 behaviour; depth matches old API.
    double turb(const point3& p, int depth = 7, double omega = 0.5) const {
        return turbulence_simple<double>(p.x(), p.y(), p.z(), omega, depth);
    }

    // FBm: signed sum of noise octaves.
    // New addition — not in original perlin.h.
    double fbm(const point3& p, int octaves = 7, double omega = 0.5) const {
        return fbm_simple<double>(p.x(), p.y(), p.z(), omega, octaves);
    }
};


#endif
