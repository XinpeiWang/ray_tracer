#pragma once
// ---------------------------------------------------------------------------
// grid_medium.h -- Grid-based majorant infrastructure for heterogeneous media
//
// Mirrors pbrt-v4 (media.h):
//   RayMajorantSegment     -- a [tMin, tMax] interval with a majorant sigma_t
//   Bounds3<T>             -- 3D axis-aligned bounding box
//   Ray3<T>                -- 3D ray (origin + direction)
//   MajorantGrid<T>        -- uniform voxel grid of per-voxel max density
//   DDAMajorantIterator<T> -- iterates ray/grid intersections via 3D DDA
//
// Design rules:
//   - Plain template structs, CPU_GPU tagged
//   - No virtual functions, no heap allocation in hot paths
//   - T = double on CPU, float on GPU
//
// Reference: pbrt-v4 src/pbrt/media.h, MajorantGrid and DDAMajorantIterator
// ---------------------------------------------------------------------------

#include "cpu_gpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <vector>

// ---------------------------------------------------------------------------
// RayMajorantSegment<T>
// pbrt-v4: struct RayMajorantSegment (base/medium.h)
// Per-channel (RGB) majorant extinction over [tMin, tMax].
// ---------------------------------------------------------------------------
template<typename T>
struct RayMajorantSegment {
    T tMin;
    T tMax;
    T sigma_maj_r, sigma_maj_g, sigma_maj_b;  // per-channel majorant
    T sigma_maj;  // scalar representative (== all three for uniform media)

    // Convenience: scalar (uniform RGB) — used by DDAMajorantIterator
    CPU_GPU RayMajorantSegment() = default;
    CPU_GPU RayMajorantSegment(T tMin_, T tMax_, T sigma)
        : tMin(tMin_), tMax(tMax_),
          sigma_maj_r(sigma), sigma_maj_g(sigma), sigma_maj_b(sigma),
          sigma_maj(sigma) {}
};

// ---------------------------------------------------------------------------
// Bounds3<T>
// pbrt-v4: Bounds3f (util/vecmath.h)
// ---------------------------------------------------------------------------
template<typename T>
struct Bounds3 {
    T pMin[3];
    T pMax[3];

    CPU_GPU Bounds3() {
        for (int i = 0; i < 3; ++i) { pMin[i] = T(1e30); pMax[i] = T(-1e30); }
    }
    CPU_GPU Bounds3(T x0, T y0, T z0, T x1, T y1, T z1) {
        pMin[0] = x0; pMin[1] = y0; pMin[2] = z0;
        pMax[0] = x1; pMax[1] = y1; pMax[2] = z1;
    }

    CPU_GPU T diagonal(int axis) const { return pMax[axis] - pMin[axis]; }

    CPU_GPU T offset(T p, int axis) const {
        T d = diagonal(axis);
        return (d > T(0)) ? (p - pMin[axis]) / d : T(0);
    }

    // Ray/AABB slab intersection.
    // pbrt-v4: Bounds3::IntersectP (util/vecmath.h)
    CPU_GPU bool intersect_ray(
            const T* orig, const T* dir, T ray_tMax,
            T* tMin_out, T* tMax_out) const {
        T t0 = T(0), t1 = ray_tMax;
        for (int axis = 0; axis < 3; ++axis) {
            T inv_d = (dir[axis] != T(0)) ? T(1) / dir[axis] : T(1e30);
            T tNear = (pMin[axis] - orig[axis]) * inv_d;
            T tFar  = (pMax[axis] - orig[axis]) * inv_d;
            if (tNear > tFar) { T tmp = tNear; tNear = tFar; tFar = tmp; }
            t0 = tNear > t0 ? tNear : t0;
            t1 = tFar  < t1 ? tFar  : t1;
            if (t0 > t1) return false;
        }
        *tMin_out = t0;
        *tMax_out = t1;
        return true;
    }
};

// ---------------------------------------------------------------------------
// Ray3<T>
// ---------------------------------------------------------------------------
template<typename T>
struct Ray3 {
    T o[3];
    T d[3];

    CPU_GPU Ray3() {
        for (int i = 0; i < 3; ++i) { o[i] = T(0); d[i] = T(0); }
    }
    CPU_GPU Ray3(T ox, T oy, T oz, T dx, T dy, T dz) {
        o[0]=ox; o[1]=oy; o[2]=oz;
        d[0]=dx; d[1]=dy; d[2]=dz;
    }

    CPU_GPU void at(T t, T* p) const {
        p[0] = o[0] + t * d[0];
        p[1] = o[1] + t * d[1];
        p[2] = o[2] + t * d[2];
    }
};

// ---------------------------------------------------------------------------
// MajorantGrid<T>
// pbrt-v4: MajorantGrid (media.h)
// ---------------------------------------------------------------------------
template<typename T>
struct MajorantGrid {
    Bounds3<T>    bounds;
    std::vector<T> voxels;
    int res[3];

    MajorantGrid() { res[0] = res[1] = res[2] = 0; }

    MajorantGrid(Bounds3<T> b, int rx, int ry, int rz)
        : bounds(b), voxels(static_cast<size_t>(rx) * ry * rz, T(0))
    {
        res[0] = rx; res[1] = ry; res[2] = rz;
    }

    CPU_GPU T lookup(int x, int y, int z) const {
        return voxels[x + res[0] * (y + res[1] * z)];
    }

    void set(int x, int y, int z, T v) {
        voxels[x + res[0] * (y + res[1] * z)] = v;
    }

    CPU_GPU void voxel_bounds(int x, int y, int z,
                               T* pMin_out, T* pMax_out) const {
        pMin_out[0] = T(x)     / T(res[0]);
        pMin_out[1] = T(y)     / T(res[1]);
        pMin_out[2] = T(z)     / T(res[2]);
        pMax_out[0] = T(x + 1) / T(res[0]);
        pMax_out[1] = T(y + 1) / T(res[1]);
        pMax_out[2] = T(z + 1) / T(res[2]);
    }
};

// ---------------------------------------------------------------------------
// DDAMajorantIterator<T>
//
// Usage:
//   DDAMajorantIterator<T> it(ray, tMin, tMax, &grid, sigma_t);
//   while (auto seg = it.Next()) { ... use seg->tMin, seg->tMax, seg->sigma_maj ... }
//
// pbrt-v4: DDAMajorantIterator (media.h)
// ---------------------------------------------------------------------------
template<typename T>
class DDAMajorantIterator {
public:
    DDAMajorantIterator() : tMin_(T(1e30)), tMax_(T(-1e30)), grid_(nullptr), sigma_t_(T(0)) {}

    // Construct the iterator for a ray through the given grid.
    // pbrt-v4: DDAMajorantIterator constructor (media.h)
    CPU_GPU DDAMajorantIterator(const Ray3<T>& ray, T tMin, T tMax,
                                 const MajorantGrid<T>* grid, T sigma_t)
        : tMin_(tMin), tMax_(tMax), grid_(grid), sigma_t_(sigma_t)
    {
        // Transform the ray into normalized grid space [0,1]^3.
        // pbrt-v4: Ray rayGrid(bounds.Offset(ray.o), ray.d / diag)
        T diag[3];
        T ray_o_grid[3], ray_d_grid[3];
        for (int axis = 0; axis < 3; ++axis) {
            diag[axis] = grid->bounds.diagonal(axis);
            ray_o_grid[axis] = grid->bounds.offset(ray.o[axis], axis);
            ray_d_grid[axis] = (diag[axis] > T(0)) ? ray.d[axis] / diag[axis] : T(0);
        }

        // Point where the ray enters the grid in normalized grid space
        T gridIntersect[3];
        for (int axis = 0; axis < 3; ++axis)
            gridIntersect[axis] = ray_o_grid[axis] + tMin * ray_d_grid[axis];

        for (int axis = 0; axis < 3; ++axis) {
            // Current voxel index, clamped to valid range
            T vf = gridIntersect[axis] * T(grid->res[axis]);
            voxel_[axis] = clamp_int(static_cast<int>(vf), 0, grid->res[axis] - 1);

            T absD = std::abs(ray_d_grid[axis]);
            delta_t_[axis] = (absD > T(0)) ? T(1) / (absD * T(grid->res[axis])) : T(1e30);

            // Handle the degenerate -0.f direction (pbrt-v4 comment)
            T rd = ray_d_grid[axis];
            if (rd == T(-0)) rd = T(0);

            if (rd >= T(0)) {
                T nextVoxelPos = T(voxel_[axis] + 1) / T(grid->res[axis]);
                next_crossing_t_[axis] = (absD > T(0))
                    ? tMin + (nextVoxelPos - gridIntersect[axis]) / ray_d_grid[axis]
                    : T(1e30);
                step_[axis]        = 1;
                voxel_limit_[axis] = grid->res[axis];
            } else {
                T nextVoxelPos = T(voxel_[axis]) / T(grid->res[axis]);
                next_crossing_t_[axis] = tMin + (nextVoxelPos - gridIntersect[axis]) / ray_d_grid[axis];
                step_[axis]        = -1;
                voxel_limit_[axis] = -1;
            }
        }
    }

    // Return the next segment, or std::nullopt if traversal is complete.
    // Matches pbrt-v4: DDAMajorantIterator::Next (media.h)
    CPU_GPU std::optional<RayMajorantSegment<T>> Next() {
        if (tMin_ >= tMax_)
            return std::optional<RayMajorantSegment<T>>();
        // Find stepAxis for stepping to next voxel and exit point tVoxelExit
        const int cmpToAxis[8] = {2, 1, 2, 1, 2, 2, 0, 0};
        int bits = 0;
        if (next_crossing_t_[0] < next_crossing_t_[1]) bits += 4;
        if (next_crossing_t_[0] < next_crossing_t_[2]) bits += 2;
        if (next_crossing_t_[1] < next_crossing_t_[2]) bits += 1;
        int stepAxis = cmpToAxis[bits];
        T tVoxelExit = (next_crossing_t_[stepAxis] < tMax_)
            ? next_crossing_t_[stepAxis] : tMax_;
        // Build segment for current voxel
        T sigma_maj = sigma_t_ * grid_->lookup(voxel_[0], voxel_[1], voxel_[2]);
        RayMajorantSegment<T> seg{tMin_, tVoxelExit, sigma_maj};
        // Advance DDA state (pbrt-v4)
        tMin_ = tVoxelExit;
        if (next_crossing_t_[stepAxis] > tMax_) tMin_ = tMax_;
        voxel_[stepAxis] += step_[stepAxis];
        if (voxel_[stepAxis] == voxel_limit_[stepAxis]) tMin_ = tMax_;
        next_crossing_t_[stepAxis] += delta_t_[stepAxis];
        return seg;
    }

private:
    static CPU_GPU int clamp_int(int v, int lo, int hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    T    sigma_t_;
    T    tMin_, tMax_;
    const MajorantGrid<T>* grid_;
    T    next_crossing_t_[3];
    T    delta_t_[3];
    int  step_[3];
    int  voxel_limit_[3];
    int  voxel_[3];
};
