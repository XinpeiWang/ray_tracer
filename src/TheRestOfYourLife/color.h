#ifndef COLOR_H
#define COLOR_H
//==============================================================================================
// Originally written in 2020 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "interval_book.h"
#include "vec3.h"
#include "../shared/tone_map.h"


using color = vec3;


// write_color -- apply tone mapping + sRGB encoding and emit an 8-bit PPM pixel.
//
// Tone-mapping mode (default: ACES):
//   ToneMapMode::ACES      -- Narkowicz ACES RRT+ODT approximation (filmic, pleasant highlights)
//   ToneMapMode::Reinhard  -- Simple L/(1+L) (faster, less filmic)
//   ToneMapMode::None      -- Clamp only (legacy behavior)
//
// After the chosen operator maps HDR linear values to [0,1], the standard
// sRGB OETF (piecewise, gamma ~2.2) is applied before quantising to [0,255].
// This replaces the previous sqrt / gamma-2.0 approximation.
inline void write_color(std::ostream& out, const color& pixel_color,
                        ToneMapMode tone_map = ToneMapMode::ACES) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Replace NaN/Inf components with zero (firefly guard).
    if (!std::isfinite(r)) r = 0.0;
    if (!std::isfinite(g)) g = 0.0;
    if (!std::isfinite(b)) b = 0.0;

    // Tone map from HDR linear to [0,1].
    r = apply_tone_map(r, tone_map);
    g = apply_tone_map(g, tone_map);
    b = apply_tone_map(b, tone_map);

    // Apply sRGB OETF (standard piecewise gamma ~2.2).
    r = linear_to_srgb(r);
    g = linear_to_srgb(g);
    b = linear_to_srgb(b);

    // Translate the [0,1] component values to the byte range [0,255].
    static const interval intensity(0.000, 0.999);
    int rbyte = int(256 * intensity.clamp(r));
    int gbyte = int(256 * intensity.clamp(g));
    int bbyte = int(256 * intensity.clamp(b));

    // Write out the pixel color components.
    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}


#endif
