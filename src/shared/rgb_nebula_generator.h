#pragma once
// ---------------------------------------------------------------------------
// rgb_nebula_generator.h -- procedural R/G/B density grid generator, shared
// by the CPU and GPU E4 (RGB Grid Medium) scene builders so both backends
// render the exact SAME voxel data, not just the same algorithm. This runs
// once at scene-build time on the HOST for both backends (GPU just uploads
// the resulting flat array), so it's plain host code - no CPU_GPU tagging
// needed, unlike the medium-evaluation code this data feeds into.
// ---------------------------------------------------------------------------

#include <vector>
#include "noise.h"

// Fills `out` (resized to nx*ny*nz) with a smooth, non-negative density
// field from a 3-octave Perlin FBm, remapped from noise's native [-1,1]
// range to [0,1] and clamped. `freq` controls how many noise "blobs" fit
// across the grid; `offset_*` shifts the sample point so calling this 3x
// with different offsets (once per R/G/B channel) gives each channel an
// independent, decorrelated pattern - the point of this scene over
// cloud_medium_hittable's single grayscale density is spatially-varying
// COLOR, which requires the channels to actually differ from each other
// rather than all tracking the same shape (which would just tint the whole
// volume one flat color).
template<typename T>
inline void generate_nebula_channel(int nx, int ny, int nz,
                                     T freq, T offset_x, T offset_y, T offset_z,
                                     std::vector<T>& out) {
    out.resize(static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz));
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                T px = (T(x) + T(0.5)) / T(nx) * freq + offset_x;
                T py = (T(y) + T(0.5)) / T(ny) * freq + offset_y;
                T pz = (T(z) + T(0.5)) / T(nz) * freq + offset_z;
                T d = T(0), omega = T(0.5), lambda = T(1);
                for (int oct = 0; oct < 3; ++oct) {
                    d += omega * perlin_noise<T>(lambda*px, lambda*py, lambda*pz);
                    omega *= T(0.5);
                    lambda *= T(1.99);
                }
                d = d * T(0.5) + T(0.5);  // remap [-1,1] -> ~[0,1]
                d = d < T(0) ? T(0) : (d > T(1) ? T(1) : d);
                out[x + nx * (y + ny * z)] = d;
            }
        }
    }
}
