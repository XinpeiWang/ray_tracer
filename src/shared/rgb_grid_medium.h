#pragma once
// ---------------------------------------------------------------------------
// rgb_grid_medium.h -- RGB heterogeneous volume medium
//
// Mirrors pbrt-v4 RGBGridMedium (src/pbrt/media.h).
//
// RGBGridMediumData<T> extends GridMediumData<T> by replacing the single
// scalar density grid with per-channel RGB grids for absorption, scattering,
// and emission:
//
//   sigma_a_grids  [optional] -- SampledGrid<T>[3] for R, G, B absorption
//   sigma_s_grids  [optional] -- SampledGrid<T>[3] for R, G, B scattering
//   Le_grids       [optional] -- SampledGrid<T>[3] for R, G, B emission
//
// When a grid is absent the channel value defaults to 1 (for sigma_a/s) or 0
// (for Le), exactly as in pbrt-v4.
//
// The public interface uses plain T[3] arrays for RGB results so that the
// struct remains host/device friendly without any spectrum class dependency.
//
// sample_ray behavior:
//   Returns a DDAMajorantIterator with sigma_t=1; the majorant grid voxels
//   already store (sigma_scale * max_c(sigma_a[c]+sigma_s[c])) so the
//   segment sigma_maj equals the pre-scaled per-voxel bound directly.
//   Matches pbrt-v4: DDAMajorantIterator(ray,tMin,tMax,&majorantGrid,
//                                         SampledSpectrum(1))
//
// Majorant construction:
//   sigma_maj per voxel = sigma_scale * max_c( max over voxel of
//                           sigma_a_grid[c] + sigma_s_grid[c] )
//   (absent grids treated as 1 per channel)
//
// References:
//   pbrt-v4 src/pbrt/media.h -- RGBGridMedium
//   pbrt-v4 src/pbrt/util/containers.h -- SampledGrid
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU
#   endif
#endif

#include "sampled_grid.h"  // SampledGrid<T>, GridMediumData<T>
#include "grid_medium.h"   // MajorantGrid<T>, DDAMajorantIterator<T>, Bounds3<T>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <vector>

// ---------------------------------------------------------------------------
// RGBGridMediumData<T>
//
// pbrt-v4: RGBGridMedium (media.h)
//
// Stores up to three SampledGrid<T> channels each for sigma_a, sigma_s, and
// Le (emission).  All grids must share the same resolution and cover [0,1]^3.
//
// Usage:
//   // Build with all channels, unit-cube bounds, g=0
//   auto med = RGBGridMediumData<double>::build(
//       sa_r, sa_g, sa_b,   // sigma_a voxel data per channel (flat arrays)
//       ss_r, ss_g, ss_b,   // sigma_s voxel data per channel
//       {},  {},  {},        // Le (empty = no emission)
//       nx, ny, nz,
//       bounds,
//       sigma_scale, Le_scale, g);
//
//   // Sample at a normalized point
//   double sa[3], ss[3], Le[3];
//   med.sample_point(px, py, pz, sa, ss, Le);
//
//   // Iterate ray segments
//   DDAMajorantIterator<double> it = med.sample_ray(ray, tMin, tMax);
//   while (auto seg = it.Next()) { ... }
// ---------------------------------------------------------------------------
template<typename T>
struct RGBGridMediumData {

	// -----------------------------------------------------------------------
	// Data members (public for aggregate-style inspection in tests)
	// -----------------------------------------------------------------------

	// Optional per-channel absorption/scattering grids ([0]=R, [1]=G, [2]=B)
	// pbrt-v4: pstd::optional<SampledGrid<RGBUnboundedSpectrum>> sigma_aGrid, sigma_sGrid
	std::optional<std::array<SampledGrid<T>, 3>> sigma_a_grids;
	std::optional<std::array<SampledGrid<T>, 3>> sigma_s_grids;

	// Optional per-channel emission grid
	// pbrt-v4: pstd::optional<SampledGrid<RGBIlluminantSpectrum>> LeGrid
	std::optional<std::array<SampledGrid<T>, 3>> Le_grids;

	// Scalar multipliers
	// pbrt-v4: Float sigmaScale, LeScale
	T sigma_scale = T(1);
	T Le_scale    = T(0);

	// Henyey-Greenstein asymmetry parameter
	T g = T(0);

	// Majorant grid (scalar: max over channels of sigma_a+sigma_s per voxel)
	// pbrt-v4: MajorantGrid majorantGrid
	MajorantGrid<T> majorant_grid;

	// World-space AABB (used to map ray into normalized coords)
	Bounds3<T> bounds;

	// Grid resolution (shared across all present grids)
	int nx = 0, ny = 0, nz = 0;

	// -----------------------------------------------------------------------
	// Default constructor
	// -----------------------------------------------------------------------
	RGBGridMediumData() = default;

	// -----------------------------------------------------------------------
	// Factory helper: build from flat per-channel arrays.
	//
	// Each of sa_r/sa_g/sa_b etc. is a flat voxel array of length nx*ny*nz,
	// or empty (std::vector<T>{}) to omit that channel group.
	//
	// When ALL three sigma_a channels are empty → sigma_a_grids = nullopt.
	// When ALL three sigma_s channels are empty → sigma_s_grids = nullopt.
	// When ALL three Le channels are empty      → Le_grids = nullopt.
	//
	// pbrt-v4: RGBGridMedium constructor + majorantGrid initialization
	// -----------------------------------------------------------------------
	static RGBGridMediumData build(
		const std::vector<T>& sa_r, const std::vector<T>& sa_g, const std::vector<T>& sa_b,
		const std::vector<T>& ss_r, const std::vector<T>& ss_g, const std::vector<T>& ss_b,
		const std::vector<T>& le_r, const std::vector<T>& le_g, const std::vector<T>& le_b,
		int nx_, int ny_, int nz_,
		Bounds3<T> bounds_,
		T sigma_scale_ = T(1), T Le_scale_ = T(0), T g_ = T(0),
		int maj_res = 16)
	{
		RGBGridMediumData m;
		m.nx = nx_; m.ny = ny_; m.nz = nz_;
		m.bounds = bounds_;
		m.sigma_scale = sigma_scale_;
		m.Le_scale    = Le_scale_;
		m.g           = g_;

		// Absorption grids
		bool has_sa = !sa_r.empty() || !sa_g.empty() || !sa_b.empty();
		if (has_sa) {
			m.sigma_a_grids = std::array<SampledGrid<T>, 3>{
				make_grid(sa_r, nx_, ny_, nz_),
				make_grid(sa_g, nx_, ny_, nz_),
				make_grid(sa_b, nx_, ny_, nz_)
			};
		}

		// Scattering grids
		bool has_ss = !ss_r.empty() || !ss_g.empty() || !ss_b.empty();
		if (has_ss) {
			m.sigma_s_grids = std::array<SampledGrid<T>, 3>{
				make_grid(ss_r, nx_, ny_, nz_),
				make_grid(ss_g, nx_, ny_, nz_),
				make_grid(ss_b, nx_, ny_, nz_)
			};
		}

		// Emission grids
		bool has_le = !le_r.empty() || !le_g.empty() || !le_b.empty();
		if (has_le) {
			m.Le_grids = std::array<SampledGrid<T>, 3>{
				make_grid(le_r, nx_, ny_, nz_),
				make_grid(le_g, nx_, ny_, nz_),
				make_grid(le_b, nx_, ny_, nz_)
			};
		}

		// Build majorant grid
		m.majorant_grid = MajorantGrid<T>(bounds_, maj_res, maj_res, maj_res);
		m.build_majorant_grid();

		return m;
	}

	// -----------------------------------------------------------------------
	// is_emissive
	// pbrt-v4: bool IsEmissive() const { return LeGrid && LeScale > 0; }
	// -----------------------------------------------------------------------
	CPU_GPU bool is_emissive() const {
		return Le_grids.has_value() && Le_scale > T(0);
	}

	// -----------------------------------------------------------------------
	// sample_point
	//
	// Sample the medium properties at a normalized point p in [0,1]^3.
	//
	// Outputs:
	//   sigma_a_out[3]  -- absorption per RGB channel
	//   sigma_s_out[3]  -- scattering per RGB channel
	//   Le_out[3]       -- emission per RGB channel
	//
	// pbrt-v4: RGBGridMedium::SamplePoint (after converting to medium space)
	// The caller is responsible for any world->medium coordinate transform.
	// -----------------------------------------------------------------------
	CPU_GPU void sample_point(T px, T py, T pz,
							  T sigma_a_out[3],
							  T sigma_s_out[3],
							  T Le_out[3]) const
	{
		// Absorption
		// pbrt-v4: sigma_a = sigmaScale * (sigma_aGrid ? sigma_aGrid->Lookup(p,convert) : SampledSpectrum(1))
		for (int c = 0; c < 3; ++c) {
			T sa = sigma_a_grids.has_value()
				? (*sigma_a_grids)[c].lookup(px, py, pz)
				: T(1);
			sigma_a_out[c] = sigma_scale * sa;
		}

		// Scattering
		for (int c = 0; c < 3; ++c) {
			T ss = sigma_s_grids.has_value()
				? (*sigma_s_grids)[c].lookup(px, py, pz)
				: T(1);
			sigma_s_out[c] = sigma_scale * ss;
		}

		// Emission
		// pbrt-v4: Le = LeScale * LeGrid->Lookup(p,convert)  (or zero if absent)
		for (int c = 0; c < 3; ++c) {
			T le = Le_grids.has_value()
				? (*Le_grids)[c].lookup(px, py, pz)
				: T(0);
			Le_out[c] = Le_scale * le;
		}
	}

	// -----------------------------------------------------------------------
	// sigma_t_max_channel
	//
	// Return max over RGB channels of sigma_a[c] + sigma_s[c] at a voxel
	// defined by its normalized sub-bounds. Used for majorant construction.
	//
	// pbrt-v4: max_value of sigma_t over a voxel
	// -----------------------------------------------------------------------
	T sigma_t_max_channel(T bx0, T by0, T bz0, T bx1, T by1, T bz1) const {
		T result = T(0);
		for (int c = 0; c < 3; ++c) {
			T sa_max = sigma_a_grids.has_value()
				? (*sigma_a_grids)[c].max_value(bx0, by0, bz0, bx1, by1, bz1)
				: T(1);
			T ss_max = sigma_s_grids.has_value()
				? (*sigma_s_grids)[c].max_value(bx0, by0, bz0, bx1, by1, bz1)
				: T(1);
			result = std::max(result, sa_max + ss_max);
		}
		return result;
	}

	// -----------------------------------------------------------------------
	// sample_ray
	//
	// Return a DDAMajorantIterator for the given ray segment.
	// The caller must pre-clip the ray to the medium's AABB (tMin, tMax).
	// The ray origin and direction must already be in normalized [0,1]^3 space
	// (i.e. the world-to-medium transform and bounds.offset have been applied).
	//
	// pbrt-v4: RGBGridMedium::SampleRay returns DDAMajorantIterator
	// -----------------------------------------------------------------------
	CPU_GPU DDAMajorantIterator<T> sample_ray(
		const Ray3<T>& ray, T tMin, T tMax) const
	{
		// pbrt-v4: DDAMajorantIterator(ray, tMin, tMax, &majorantGrid, sigma_t)
		// where sigma_t = SampledSpectrum(1) -- the majorant grid already stores
		// sigmaScale * maxSigma_t per voxel, so sigma_t_ multiplier must be 1.
		return DDAMajorantIterator<T>(ray, tMin, tMax, &majorant_grid, T(1));
	}

	// -----------------------------------------------------------------------
	// intersect_ray
	//
	// Ray/AABB intersection using the stored world-space bounds.
	// Returns true if the ray hits, and sets tMin_out, tMax_out.
	// pbrt-v4: bounds.IntersectP inside SampleRay
	// -----------------------------------------------------------------------
	CPU_GPU bool intersect_ray(const Ray3<T>& ray, T ray_tMax,
							   T& tMin_out, T& tMax_out) const {
		return bounds.intersect_ray(ray.o, ray.d, ray_tMax, &tMin_out, &tMax_out);
	}

private:
	// Make a SampledGrid from a flat vector; if the vector is empty, produce
	// a 1x1x1 grid filled with T(0).
	static SampledGrid<T> make_grid(const std::vector<T>& v, int nx_, int ny_, int nz_) {
		if (v.empty()) {
			// Empty sentinel: 1x1x1 zero grid
			return SampledGrid<T>(std::vector<T>(1, T(0)), 1, 1, 1);
		}
		return SampledGrid<T>(v, nx_, ny_, nz_);
	}

	// Build the scalar majorant grid from per-channel grids.
	// pbrt-v4: majorantGrid initialization loop in RGBGridMedium constructor
	void build_majorant_grid() {
		int mr = majorant_grid.res[0]; // assumed cubic (maj_res x maj_res x maj_res)
		int my = majorant_grid.res[1];
		int mz = majorant_grid.res[2];
		for (int z = 0; z < mz; ++z) {
			for (int y = 0; y < my; ++y) {
				for (int x = 0; x < mr; ++x) {
					T pMin[3], pMax[3];
					majorant_grid.voxel_bounds(x, y, z, pMin, pMax);
					T maj = sigma_scale * sigma_t_max_channel(
						pMin[0], pMin[1], pMin[2],
						pMax[0], pMax[1], pMax[2]);
					majorant_grid.set(x, y, z, maj);
				}
			}
		}
	}
};
