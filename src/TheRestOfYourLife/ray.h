#ifndef RAY_H
#define RAY_H
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "vec3.h"


class ray {
  public:
    ray() {}

    ray(const point3& origin, const vec3& direction, double time)
      : orig(origin), dir(direction), tm(time) {}

    ray(const point3& origin, const vec3& direction)
      : ray(origin, direction, 0) {}

    const point3& origin() const  { return orig; }
    const vec3& direction() const { return dir; }

    double time() const { return tm; }

    point3 at(double t) const {
        return orig + t*dir;
    }

    // Ray differentials (pbrt-v4 RayDifferential) -- the x/y-pixel-offset
    // auxiliary rays a camera casts alongside the primary ray, used to
    // estimate a texture lookup's footprint in texture space (see
    // src/shared/surface_interaction.h's compute_differentials() and
    // src/shared/mipmap.h's EWA filter). Defaulted off so every existing
    // constructor/call site is unaffected; only camera.h's primary
    // pixel-sample ray path sets these (see get_ray()).
    bool has_differentials() const { return has_diff; }
    const point3& rx_origin() const    { return rx_o; }
    const vec3&   rx_direction() const { return rx_d; }
    const point3& ry_origin() const    { return ry_o; }
    const vec3&   ry_direction() const { return ry_d; }

    void set_differentials(const point3& rx_origin_, const vec3& rx_direction_,
                            const point3& ry_origin_, const vec3& ry_direction_) {
        rx_o = rx_origin_; rx_d = rx_direction_;
        ry_o = ry_origin_; ry_d = ry_direction_;
        has_diff = true;
    }

  private:
    point3 orig;
    vec3 dir;
    double tm;

    bool has_diff = false;
    point3 rx_o, ry_o;
    vec3   rx_d, ry_d;
};


#endif
