#pragma once
// nanovdb_dense.h -- reads a NanoVDB file (pbrt-v4 MakeNamedMedium "nanovdb") into a dense density grid and places it in the world, for every renderer that has a dense-grid medium.
//
// The sparse grid is baked into a flat array over its active index box (x fastest), the same shape "uniformgrid" media use, so the CPU renderer and both OptiX backends reuse
// their existing dense-grid delta-tracking code unchanged; no NanoVDB code runs on a device. Moved out of pbrt_cpu_stages.h (bakeNanovdbMedium), where only the CPU could use
// it, with the arithmetic unchanged. Split in two so a caller can cache the expensive half: readGrid() opens the file and bakes (O(voxels)); placeInWorld() only multiplies the
// scene's transform in (a few matrix operations).
//
// Scope, as before: a single named plain-float density grid (and an optional float temperature grid sampled at the same voxels), at most 512 voxels per axis. Anything else
// logs why on stderr and gives an empty grid, which renders as an invisible medium.

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "../external/nanovdb/NanoVDB.h"
#include "../external/nanovdb/io/IO.h"
#include "pbrt_flatten.h"

namespace nanovdb_dense {

// Each axis is checked before any multiplication, in int64, so a corrupt file claiming an extreme (still int32-representable) box cannot overflow the allocation size.
constexpr std::int64_t kMaxVoxelsPerAxis = 512;

struct Grid {
	int nx = 0, ny = 0, nz = 0;
	std::vector<float> density;       // nx*ny*nz, x fastest then y then z; empty when the file could not be used
	std::vector<float> temperature;   // Kelvin per voxel, same layout; empty = no emission
	// The grid's own native (pre-scene-transform) positions of the unit cube's origin and its +x, +y, +z corners. Four points fully determine the affine index->native map
	// whatever convention NanoVDB stores it in, which is why they are kept instead of a matrix.
	double corner[4][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
	bool ok() const { return !density.empty(); }
};

// Opens `file`, reads the float grid `gridName` and, when `temperatureGridName` is non-empty, the float grid of that name at the same voxel coordinates.
inline Grid readGrid(const std::string &file, const std::string &gridName, const std::string &temperatureGridName) {
	Grid out;
	try {
		auto handle = nanovdb::io::readGrid(file, gridName);
		const auto *grid = handle.grid<float>();
		if (!grid) {
			// The grid exists but is not a plain float build (Vec3f, Mask, Fp4...). readGrid() does not throw for that, so say it here or the medium silently vanishes.
			std::cerr << "[nanovdb] medium: grid \"" << gridName << "\" in \"" << file
					  << "\" is not a plain float grid (only float grids are supported); the medium will render as empty (invisible)\n";
			return out;
		}
		const auto bbox = grid->indexBBox();
		const auto bmin = bbox.min();
		const auto bmax = bbox.max();
		// int64 before subtracting: the bounds are int32 read straight from the file with no min <= max check, and a difference of two int32s overflows int32.
		const std::int64_t nx64 = static_cast<std::int64_t>(bmax[0]) - static_cast<std::int64_t>(bmin[0]) + 1;
		const std::int64_t ny64 = static_cast<std::int64_t>(bmax[1]) - static_cast<std::int64_t>(bmin[1]) + 1;
		const std::int64_t nz64 = static_cast<std::int64_t>(bmax[2]) - static_cast<std::int64_t>(bmin[2]) + 1;
		if (!(nx64 > 0 && ny64 > 0 && nz64 > 0 && nx64 <= kMaxVoxelsPerAxis && ny64 <= kMaxVoxelsPerAxis && nz64 <= kMaxVoxelsPerAxis)) {
			std::cerr << "[nanovdb] medium: grid \"" << gridName << "\" in \"" << file << "\" has an active bounding box of " << nx64 << "x" << ny64 << "x" << nz64
					  << " voxels, which is degenerate or exceeds the " << kMaxVoxelsPerAxis
					  << "-voxels-per-axis cap this loader enforces; the medium will render as empty (invisible)\n";
			return out;
		}
		const int nx = static_cast<int>(nx64), ny = static_cast<int>(ny64), nz = static_cast<int>(nz64);
		// A cached accessor: the loop is spatially coherent (x fastest), so its cached traversal state pays off.
		auto acc = grid->getAccessor();
		std::vector<float> density(static_cast<std::size_t>(nx) * ny * nz);
		for (int z = 0; z < nz; ++z)
			for (int y = 0; y < ny; ++y)
				for (int x = 0; x < nx; ++x)
					density[(static_cast<std::size_t>(z) * ny + y) * nx + x] = acc.getValue(nanovdb::Coord(bmin[0] + x, bmin[1] + y, bmin[2] + z));

		std::vector<float> temperature;
		if (!temperatureGridName.empty()) {
			// Sampled at the density grid's own voxel coordinates, not its own box: a fire or smoke asset's grids share the active region in practice, and the accessor returns
			// the grid's background value (normally 0) outside it, so a mismatch means "no emission there", not a bad read.
			try {
				auto tHandle = nanovdb::io::readGrid(file, temperatureGridName);
				const auto *tGrid = tHandle.grid<float>();
				if (!tGrid) {
					std::cerr << "[nanovdb] medium: temperature grid \"" << temperatureGridName << "\" in \"" << file
							  << "\" is not a plain float grid (only float grids are supported); blackbody emission is dropped\n";
				} else {
					auto tAcc = tGrid->getAccessor();
					temperature.resize(static_cast<std::size_t>(nx) * ny * nz);
					for (int z = 0; z < nz; ++z)
						for (int y = 0; y < ny; ++y)
							for (int x = 0; x < nx; ++x)
								temperature[(static_cast<std::size_t>(z) * ny + y) * nx + x] = tAcc.getValue(nanovdb::Coord(bmin[0] + x, bmin[1] + y, bmin[2] + z));
				}
			} catch (...) {
				std::cerr << "[nanovdb] medium: failed to read temperature grid \"" << temperatureGridName << "\" from \"" << file
						  << "\" (corrupt file or wrong gridname); blackbody emission is dropped\n";
				temperature.clear();
			}
		}

		const auto native = [&](double u, double v, double w, double *p) {
			const nanovdb::Vec3d n = grid->indexToWorld(nanovdb::Vec3d(bmin[0] + u * nx, bmin[1] + v * ny, bmin[2] + w * nz));
			p[0] = n[0]; p[1] = n[1]; p[2] = n[2];
		};
		native(0, 0, 0, out.corner[0]);
		native(1, 0, 0, out.corner[1]);
		native(0, 1, 0, out.corner[2]);
		native(0, 0, 1, out.corner[3]);
		out.nx = nx; out.ny = ny; out.nz = nz;
		out.density = std::move(density);
		out.temperature = std::move(temperature);
	} catch (...) {
		// Corrupt or truncated file, a wrong grid name, or an unsupported grid type: the only way to find out is to open the file, so this is the one place that can say so.
		std::cerr << "[nanovdb] medium: failed to read grid \"" << gridName << "\" from \"" << file
				  << "\" (corrupt file, wrong gridname, or an unsupported NanoVDB grid type - only plain float grids are read); the medium will render as empty (invisible)\n";
		return Grid{};
	}
	return out;
}

// Where a grid sits in the world: the world box of its unit index cube and the world->medium affine map ([0,1]^3 coordinates), as the dense-grid media take them.
struct Placement {
	double worldMin[3] = {0, 0, 0}, worldMax[3] = {0, 0, 0};
	double toMediumMat[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
	double toMediumTranslate[3] = {0, 0, 0};
};

// `sceneXform` is the row-major 4x4 scene transform in force where the medium was declared (Medium::nanovdbXform). False if the composed map is not invertible.
inline bool placeInWorld(const Grid &grid, const double sceneXform[16], Placement &out) {
	pbrt_scene::Matrix4 xf;
	for (int i = 0; i < 16; ++i) xf.m[i] = sceneXform[i];
	double c[4][3];
	for (int k = 0; k < 4; ++k) pbrt_flatten::flatten_detail::transformPoint(xf, grid.corner[k][0], grid.corner[k][1], grid.corner[k][2], c[k]);
	// world = worldFromMedium * (u,v,w) + corner00; the matrix columns are the three sampled basis differences, in Matrix4's row-major layout.
	pbrt_scene::Matrix4 worldFromMedium;
	for (int r = 0; r < 3; ++r) {
		worldFromMedium.m[r * 4 + 0] = c[1][r] - c[0][r];
		worldFromMedium.m[r * 4 + 1] = c[2][r] - c[0][r];
		worldFromMedium.m[r * 4 + 2] = c[3][r] - c[0][r];
		worldFromMedium.m[r * 4 + 3] = c[0][r];
	}
	pbrt_scene::Matrix4 mediumFromWorld;
	if (!worldFromMedium.inverseAffine(mediumFromWorld)) return false;
	const double unitLo[3] = {0.0, 0.0, 0.0}, unitHi[3] = {1.0, 1.0, 1.0};
	pbrt_flatten::flatten_detail::aabbOfTransformedBox(worldFromMedium, unitLo, unitHi, out.worldMin, out.worldMax);
	pbrt_flatten::flatten_detail::splitAffine(mediumFromWorld, out.toMediumMat, out.toMediumTranslate);
	return true;
}

}  // namespace nanovdb_dense
