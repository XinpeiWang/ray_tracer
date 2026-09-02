#pragma once
// ---------------------------------------------------------------------------
// sampled_grid.h -- 3D trilinear-interpolation grid + grid medium data
//
// Mirrors pbrt-v4:
//   SampledGrid<T>      -- util/containers.h: trilinear lookup over a 3D voxel
//                          grid stored in [0,1]^3 normalized coords
//   GridMediumData<T>   -- media.h GridMedium: per-point density sampling,
//                          majorant grid construction, SamplePoint logic
//
// Design rules (same as volume_scattering.h / grid_medium.h):
//   - Plain template structs, CPU_GPU tagged
//   - No virtual functions, no heap allocation in hot paths
//   - T = double on CPU, float on GPU
//
// Dependencies:
//   grid_medium.h   -- MajorantGrid<T>, Bounds3<T>, Ray3<T>
// ---------------------------------------------------------------------------

#include "cpu_gpu.h"

#include "grid_medium.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <vector>

// ---------------------------------------------------------------------------
// SampledGrid<T>
//
// pbrt-v4: SampledGrid<T> (util/containers.h)
//
// Stores a flat 3D array of T values with dimensions {nx, ny, nz}.
// The grid covers [0,1]^3 in normalized space.
//
// Key methods:
//   Lookup(p)           -- trilinear interpolation at normalized point p
//   Lookup(p, convert)  -- trilinear interpolation with a per-voxel converter
//   MaxValue(bounds)    -- maximum T value in a normalized sub-bounds
//   integer Lookup(ix,iy,iz) -- direct voxel read (clamped to boundary)
//
// Storage layout: values[(z * ny + y) * nx + x]   (pbrt-v4 convention)
// ---------------------------------------------------------------------------
template<typename T>
class SampledGrid {
public:
	SampledGrid() : nx_(0), ny_(0), nz_(0) {}

	// Construct from a flat span of values with given dimensions.
	// pbrt-v4: SampledGrid(pstd::span<const T>, int nx, int ny, int nz, Allocator)
	SampledGrid(const T* v, size_t count, int nx, int ny, int nz)
		: values_(v, v + count), nx_(nx), ny_(ny), nz_(nz) {}

	SampledGrid(std::vector<T> v, int nx, int ny, int nz)
		: values_(std::move(v)), nx_(nx), ny_(ny), nz_(nz) {}

	CPU_GPU int XSize() const { return nx_; }
	CPU_GPU int YSize() const { return ny_; }
	CPU_GPU int ZSize() const { return nz_; }
	CPU_GPU size_t size() const { return values_.size(); }

	// Direct integer voxel lookup with out-of-bounds clamping to T{}.
	// pbrt-v4: SampledGrid::Lookup(const Point3i &p)
	CPU_GPU T lookup(int ix, int iy, int iz) const {
		if (ix < 0 || ix >= nx_ || iy < 0 || iy >= ny_ || iz < 0 || iz >= nz_)
			return T{};
		return values_[(iz * ny_ + iy) * nx_ + ix];
	}

	// Trilinear interpolation at a normalized point p in [0,1]^3.
	// pbrt-v4: SampledGrid::Lookup(Point3f p)
	//
	// The fractional voxel coordinate is:
	//   pSamples = p * n - 0.5    (so that 0.5/n maps to voxel-center 0)
	// followed by floor to get the base integer voxel, then lerp in x,y,z.
	CPU_GPU T lookup(T px, T py, T pz) const {
		// Compute fractional sample coordinates (pbrt-v4: pSamples)
		T sx = px * T(nx_) - T(0.5);
		T sy = py * T(ny_) - T(0.5);
		T sz = pz * T(nz_) - T(0.5);

		int ix = static_cast<int>(std::floor(sx));
		int iy = static_cast<int>(std::floor(sy));
		int iz = static_cast<int>(std::floor(sz));

		T dx = sx - T(ix);
		T dy = sy - T(iy);
		T dz = sz - T(iz);

		// 8-corner trilinear interpolation
		// pbrt-v4 order: lerp z of (lerp y of (lerp x of corners))
		T d00 = lerp(dx, lookup(ix,   iy,   iz  ), lookup(ix+1, iy,   iz  ));
		T d10 = lerp(dx, lookup(ix,   iy+1, iz  ), lookup(ix+1, iy+1, iz  ));
		T d01 = lerp(dx, lookup(ix,   iy,   iz+1), lookup(ix+1, iy,   iz+1));
		T d11 = lerp(dx, lookup(ix,   iy+1, iz+1), lookup(ix+1, iy+1, iz+1));
		return lerp(dz, lerp(dy, d00, d10), lerp(dy, d01, d11));
	}

	// Trilinear lookup with a per-voxel converter function.
	// pbrt-v4: SampledGrid::Lookup(Point3f p, F convert)
	// Useful for grids of non-scalar types (e.g. RGB spectra).
	template<typename F>
	CPU_GPU auto lookup_convert(T px, T py, T pz, F convert) const
		-> decltype(convert(T{}))
	{
		using R = decltype(convert(T{}));
		T sx = px * T(nx_) - T(0.5);
		T sy = py * T(ny_) - T(0.5);
		T sz = pz * T(nz_) - T(0.5);

		int ix = static_cast<int>(std::floor(sx));
		int iy = static_cast<int>(std::floor(sy));
		int iz = static_cast<int>(std::floor(sz));

		T dx = sx - T(ix);
		T dy = sy - T(iy);
		T dz = sz - T(iz);

		auto c = [&](int x, int y, int z) -> R {
			if (x < 0 || x >= nx_ || y < 0 || y >= ny_ || z < 0 || z >= nz_)
				return convert(T{});
			return convert(values_[(z * ny_ + y) * nx_ + x]);
		};

		auto lerp_r = [](R t, R a, R b) { return a + t * (b - a); };

		R d00 = lerp_r(R(dx), c(ix, iy,   iz  ), c(ix+1, iy,   iz  ));
		R d10 = lerp_r(R(dx), c(ix, iy+1, iz  ), c(ix+1, iy+1, iz  ));
		R d01 = lerp_r(R(dx), c(ix, iy,   iz+1), c(ix+1, iy,   iz+1));
		R d11 = lerp_r(R(dx), c(ix, iy+1, iz+1), c(ix+1, iy+1, iz+1));
		return lerp_r(R(dz), lerp_r(R(dy), d00, d10), lerp_r(R(dy), d01, d11));
	}

	// Maximum value over all voxels whose sample-space coordinates overlap
	// the given sub-bounds (in [0,1]^3 normalized space).
	// pbrt-v4: SampledGrid::MaxValue(const Bounds3f &bounds)
	T max_value(T bx0, T by0, T bz0, T bx1, T by1, T bz1) const {
		// Map normalized bounds to sample coordinates
		T sx0 = bx0 * T(nx_) - T(0.5f);
		T sy0 = by0 * T(ny_) - T(0.5f);
		T sz0 = bz0 * T(nz_) - T(0.5f);
		T sx1 = bx1 * T(nx_) - T(0.5f);
		T sy1 = by1 * T(ny_) - T(0.5f);
		T sz1 = bz1 * T(nz_) - T(0.5f);

		// Integer range of voxels touched
		int ix0 = std::max(0,       static_cast<int>(std::floor(sx0)));
		int iy0 = std::max(0,       static_cast<int>(std::floor(sy0)));
		int iz0 = std::max(0,       static_cast<int>(std::floor(sz0)));
		int ix1 = std::min(nx_ - 1, static_cast<int>(std::floor(sx1)) + 1);
		int iy1 = std::min(ny_ - 1, static_cast<int>(std::floor(sy1)) + 1);
		int iz1 = std::min(nz_ - 1, static_cast<int>(std::floor(sz1)) + 1);

		T maxVal = values_[(iz0 * ny_ + iy0) * nx_ + ix0];
		for (int z = iz0; z <= iz1; ++z)
			for (int y = iy0; y <= iy1; ++y)
				for (int x = ix0; x <= ix1; ++x)
					maxVal = std::max(maxVal, values_[(z * ny_ + y) * nx_ + x]);
		return maxVal;
	}

	const std::vector<T>& values() const { return values_; }

private:
	static CPU_GPU T lerp(T t, T a, T b) { return a + t * (b - a); }

	std::vector<T> values_;
	int nx_, ny_, nz_;
};

// ---------------------------------------------------------------------------
// GridMediumData<T>
//
// pbrt-v4: GridMedium (media.h) — the per-point sampling and majorant-building
// parts, adapted as a plain CPU/GPU-friendly data struct.
//
// Wraps:
//   - A SampledGrid<T> density grid (normalized coords [0,1]^3)
//   - Scalar sigma_a, sigma_s, g coefficients (callers scale by wavelength)
//   - A MajorantGrid<T> built from the density grid (16×16×16 default)
//
// Key operations:
//   sample_point(px, py, pz)  -- returns density at a normalized point;
//                                caller multiplies by sigma_a/sigma_s
//   majorant_grid             -- ready-made MajorantGrid for DDAMajorantIterator
//
// Usage matches pbrt-v4 GridMedium::SamplePoint:
//   auto d = gm.sample_point(px, py, pz);   // scalar density in [0,1]
//   T sigma_a_eff = gm.sigma_a * d;
//   T sigma_s_eff = gm.sigma_s * d;
// ---------------------------------------------------------------------------
template<typename T>
struct GridMediumData {
	SampledGrid<T>  density_grid;   // per-voxel density (0..1 typical)
	MajorantGrid<T> majorant_grid;  // built from density_grid max values
	T sigma_a;                      // base absorption coefficient
	T sigma_s;                      // base scattering coefficient
	T g;                            // HG asymmetry parameter

	// Optional per-voxel RGB blackbody emission (pbrt-v4's nanovdb "string
	// temperaturename" grid, converted Kelvin->RGB by the caller - see
	// pbrt_flatten::Medium::nanovdbTemperatureGridName's own comment) -
	// mirrors RGBGridMediumData<T>::Le_grids/Le_scale (rgb_grid_medium.h)
	// exactly, just attached after construction via set_emission() below
	// rather than through a constructor parameter, since density_grid (and
	// therefore this grid's own resolution) already exists by the time a
	// caller knows whether a temperature grid was even present.
	std::optional<std::array<SampledGrid<T>, 3>> Le_grids;
	T Le_scale = T(0);

	GridMediumData() : sigma_a(T(0)), sigma_s(T(0)), g(T(0)) {}

	// Construct from a flat density array and world-space bounds.
	// Automatically builds the majorant grid (16×16×16 resolution).
	// pbrt-v4: GridMedium constructor, Initialize _majorantGrid_ block.
	GridMediumData(const T* density, size_t density_count,
				   int nx, int ny, int nz,
				   Bounds3<T> bounds,
				   T sa, T ss, T g_,
				   int maj_res = 16)
		: density_grid(density, density_count, nx, ny, nz)
		, majorant_grid(bounds, maj_res, maj_res, maj_res)
		, sigma_a(sa), sigma_s(ss), g(g_)
	{
		build_majorant_grid();
	}

	GridMediumData(std::vector<T> density,
				   int nx, int ny, int nz,
				   Bounds3<T> bounds,
				   T sa, T ss, T g_,
				   int maj_res = 16)
		: density_grid(std::move(density), nx, ny, nz)
		, majorant_grid(bounds, maj_res, maj_res, maj_res)
		, sigma_a(sa), sigma_s(ss), g(g_)
	{
		build_majorant_grid();
	}

	// Sample the density at a normalized point (px, py, pz) in [0,1]^3.
	// pbrt-v4: densityGrid.Lookup(p) inside GridMedium::SamplePoint
	CPU_GPU T sample_point(T px, T py, T pz) const {
		return density_grid.lookup(px, py, pz);
	}

	// Effective sigma_a/sigma_s at a normalized point.
	CPU_GPU void sample_sigma(T px, T py, T pz, T& sa_out, T& ss_out) const {
		T d = sample_point(px, py, pz);
		sa_out = sigma_a * d;
		ss_out = sigma_s * d;
	}

	// Attach optional per-voxel RGB emission (pbrt-v4: RGBGridMedium::
	// LeGrid/LeScale, see this struct's own Le_grids comment). le_r/g/b are
	// flat voxel arrays at THIS grid's own resolution (density_grid's own
	// XSize()/YSize()/ZSize()) or all empty to clear/omit emission -
	// mirrors RGBGridMediumData<T>::build()'s own "all three channels empty
	// -> Le_grids stays nullopt" convention.
	void set_emission(std::vector<T> le_r, std::vector<T> le_g, std::vector<T> le_b, T le_scale) {
		if (le_r.empty() && le_g.empty() && le_b.empty()) {
			Le_grids.reset();
			Le_scale = T(0);
			return;
		}
		const int gx = density_grid.XSize(), gy = density_grid.YSize(), gz = density_grid.ZSize();
		Le_grids = std::array<SampledGrid<T>, 3>{
			SampledGrid<T>(std::move(le_r), gx, gy, gz),
			SampledGrid<T>(std::move(le_g), gx, gy, gz),
			SampledGrid<T>(std::move(le_b), gx, gy, gz)
		};
		Le_scale = le_scale;
	}

	// pbrt-v4: bool IsEmissive() const { return LeGrid && LeScale > 0; }
	CPU_GPU bool is_emissive() const {
		return Le_grids.has_value() && Le_scale > T(0);
	}

	// Sample emission at a normalized point in [0,1]^3 - grid_medium_
	// hittable.h's hit() weights this by sigma_a/sigma_t at the scatter
	// event itself (mirrors rgb_grid_medium_hittable.h's own hit(), see
	// that file's comment for the full derivation); this call returns the
	// raw (Le_scale-applied) per-voxel value only.
	CPU_GPU void sample_emission(T px, T py, T pz, T Le_out[3]) const {
		for (int c = 0; c < 3; ++c) {
			Le_out[c] = Le_grids.has_value() ? Le_scale * (*Le_grids)[c].lookup(px, py, pz) : T(0);
		}
	}

	// sample_ray / intersect_ray -- same shape as RGBGridMediumData<T>'s own
	// (rgb_grid_medium.h), added so a CPU hittable wrapper (grid_medium_
	// hittable.h) can call this type exactly like that one calls
	// RGBGridMediumData. The one real difference: majorant_grid here stores
	// RAW density (build_majorant_grid() above never scales it, unlike
	// RGBGridMediumData::build_majorant_grid()'s `sigma_scale *
	// sigma_t_max_channel(...)`), so sigma_t_ isn't a no-op T(1) multiplier
	// the way rgb_grid_medium.h's sample_ray() passes - it has to be
	// sigma_a+sigma_s here to turn a raw density majorant into a real
	// sigma_t majorant. Caller must pre-clip the ray to the medium's AABB
	// (tMin, tMax) and pass origin/direction already in normalized [0,1]^3
	// medium space (matching RGBGridMediumData's own documented contract).
	CPU_GPU DDAMajorantIterator<T> sample_ray(
			const Ray3<T>& ray, T tMin, T tMax) const {
		return DDAMajorantIterator<T>(ray, tMin, tMax, &majorant_grid, sigma_a + sigma_s);
	}

	// Ray/AABB intersection using the majorant grid's own stored bounds
	// (GridMediumData has no separate top-level bounds member - unlike
	// RGBGridMediumData's `bounds`, this type only ever stored them inside
	// majorant_grid, since sample_ray/intersect_ray didn't exist before).
	CPU_GPU bool intersect_ray(const Ray3<T>& ray, T ray_tMax,
							   T& tMin_out, T& tMax_out) const {
		return majorant_grid.bounds.intersect_ray(ray.o, ray.d, ray_tMax, &tMin_out, &tMax_out);
	}

private:
	// Fill each majorant grid voxel with the maximum density in the
	// corresponding normalized sub-bounds.
	// pbrt-v4: for (z..y..x) majorantGrid.Set(x,y,z, densityGrid.MaxValue(voxelBounds))
	void build_majorant_grid() {
		for (int z = 0; z < majorant_grid.res[2]; ++z)
			for (int y = 0; y < majorant_grid.res[1]; ++y)
				for (int x = 0; x < majorant_grid.res[0]; ++x) {
					T pMin[3], pMax[3];
					majorant_grid.voxel_bounds(x, y, z, pMin, pMax);
					T maxDensity = density_grid.max_value(
						pMin[0], pMin[1], pMin[2],
						pMax[0], pMax[1], pMax[2]);
					majorant_grid.set(x, y, z, maxDensity);
				}
	}
};
