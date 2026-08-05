#pragma once
// ---------------------------------------------------------------------------
// sppm.h -- Stochastic Progressive Photon Mapping (pbrt-v4 port)
//
// Ports pbrt-v4:
//   cpu/integrators.cpp  -- SPPMPixel, SPPMPixelListNode, ToGrid,
//                           SPPMIntegrator::Render
//
// Components:
//   SPPMPixel<T>           -- per-pixel accumulator (radius, n, tau, Phi, m,
//                             visible-point)
//   sppm_detail::HashGrid  -- spatial hash grid mapping photon positions to
//                             pixel lists (replaces atomic linked lists with
//                             a single-threaded std::vector<forward_list>)
//   SPPMCameraPass<T,Sc>   -- trace camera rays and deposit visible points
//   SPPMPhotonPass<T,Sc>   -- trace photons and accumulate Phi/m per pixel
//   SPPMUpdateRadius<T>    -- apply gamma=2/3 contraction after photon pass
//   SPPMFinalImage<T>      -- compute output Ld + tau/(n*photonPaths*pi*r^2)
//   SPPMRender<T,Sc>       -- full multi-iteration render loop
//
// Design rules (same as bdpt.h / mlt.h):
//   - Header-only, no virtual functions
//   - Template T: float or double
//   - Template Scene: duck-typed scene concept (see below)
//   - Single-threaded reference; callers wrap in parallel loops
//   - RGB output (no spectral wavelengths)
//
// Scene concept (superset of bdpt.h concept):
//   bool  Intersect(const T org[3], const T dir[3], T t_max, BDPTHit<T>&) const
//   bool  Unoccluded(const T p0[3], const T p1[3]) const
//   bool  SampleLight(T u, const T ref_p[3], BDPTLightSample<T>&) const
//   bool  SampleLightLe(T u1, const T u2a[2], const T u2b[2],
//                        BDPTLightLeSample<T>&) const
//   T     LightPMF(int id) const
//   void  LightPDFLe(int id, const T*, const T*, const T*,
//                    T& pdfPos, T& pdfDir) const
//   void  BSDFf(int id, const T wo[3], const T wi[3], const T n[3],
//               T out[3]) const
//   bool  BSDFSampleF(int id, const T wo[3], const T n[3], T u1, T u2,
//                     T new_dir[3], T f_val[3], T& pdf, bool& is_specular) const
//   T     BSDFPdf(int id, const T wo[3], const T wi[3], const T n[3]) const
//   void  CameraPDFWe(const T* o, const T* d, T& pdfPos, T& pdfDir) const
//   void  SceneBoundingSphere(T center[3], T& radius) const
//   void  InfiniteLightLe(const T d[3], T out[3]) const
//   float InfiniteLightDensity(const T d[3]) const
//   // Additional for SPPM camera pass:
//   bool  PixelToRay(T px, T py, T cam_p[3], T ray_d[3], T cam_n[3]) const
//   // Direct-lighting query (for Ld at visible points):
//   void  DirectLight(const T p[3], const T n[3], const T wo[3], int bsdf_id,
//                     T Ld[3]) const
//   // Random number source used by both passes:
//   T     RandFloat() const
// ---------------------------------------------------------------------------

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU inline
#   endif
#endif

#include "rng.h"
#include "bdpt.h"         // BDPTHit, BDPTLightSample, BDPTLightLeSample

#include <cmath>
#include <vector>
#include <forward_list>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <cassert>
#include <limits>

// ===========================================================================
// Section 1 -- SPPMPixel<T>
// ===========================================================================
// Mirrors pbrt-v4 SPPMPixel:
//   radius       -- current search radius
//   n            -- accumulated photon count (real, after contraction)
//   tau[3]       -- accumulated flux (RGB): tau = (tau + Phi) * (rNew/r)^2
//   Phi[3]       -- photon flux accumulator for this iteration
//   m            -- raw photon count for this iteration
//   Ld[3]        -- direct-lighting contribution (camera pass)
//   vp_*         -- visible-point data deposited by camera pass

template<typename T>
struct SPPMPixel {
	// Progressive radius
	T radius = T(0);

	// Accumulated radiance components
	T Ld[3]  = {T(0), T(0), T(0)};  // direct lighting (set each camera pass)
	T tau[3] = {T(0), T(0), T(0)};  // long-running flux accumulator
	T n      = T(0);                 // real photon count (after contraction)

	// Per-iteration photon accumulation (reset after each pass)
	T   Phi[3] = {T(0), T(0), T(0)};
	int m      = 0;

	// Visible-point data (set by camera pass, used by photon pass)
	bool vp_valid = false;
	T    vp_p[3]  = {T(0), T(0), T(0)};  // world-space position
	T    vp_wo[3] = {T(0), T(0), T(0)};  // outgoing direction
	T    vp_n[3]  = {T(0), T(0), T(0)};  // shading normal
	T    vp_beta[3] = {T(0), T(0), T(0)}; // path throughput at visible point
	int  vp_bsdf_id = 0;

	// Pixel coordinates (set once at construction)
	int px = 0, py = 0;
};

// ===========================================================================
// Section 2 -- sppm_detail: hash-grid helpers
// ===========================================================================
// Mirrors pbrt-v4's ToGrid + Hash + SPPMPixelListNode linked-list approach,
// but uses a plain vector<forward_list<int>> for single-threaded simplicity.

namespace sppm_detail {

// Simple integer hash (matches pbrt-v4's Hash for Point3i)
CPU_GPU uint32_t HashPoint3i(int x, int y, int z) {
	uint32_t a = (uint32_t)x * 2654435769u;
	uint32_t b = (uint32_t)y * 805459861u;
	uint32_t c = (uint32_t)z * 3674653429u;
	return a ^ b ^ c;
}

// Next prime >= n (used to size hash table like pbrt-v4 NextPrime(nPixels))
CPU_GPU int NextPrime(int n) {
	if (n <= 2) return 2;
	if (n % 2 == 0) ++n;
	while (true) {
		bool ok = true;
		for (int i = 3; (long long)i*i <= n; i += 2) {
			if (n % i == 0) { ok = false; break; }
		}
		if (ok) return n;
		n += 2;
	}
}

template<typename T>
struct HashGrid {
	int    hashSize = 0;
	int    gridRes[3] = {1,1,1};
	T      gridMin[3] = {T(0),T(0),T(0)};
	T      gridMax[3] = {T(1),T(1),T(1)};
	T      cellSize[3] = {T(1),T(1),T(1)};

	// grid[h] = list of pixel indices whose visible point maps to bucket h
	std::vector<std::forward_list<int>> grid;

	// Build from a set of visible points
	// pixels: vector of SPPMPixel<T>, nPixels entries
	template<typename PixelVec>
	void Build(const PixelVec& pixels) {
		// Find bounds and max radius over all valid visible points
		T gMin[3] = { std::numeric_limits<T>::max(),
					  std::numeric_limits<T>::max(),
					  std::numeric_limits<T>::max() };
		T gMax[3] = { std::numeric_limits<T>::lowest(),
					  std::numeric_limits<T>::lowest(),
					  std::numeric_limits<T>::lowest() };
		T maxR = T(0);
		for (int i = 0; i < (int)pixels.size(); ++i) {
			const auto& px = pixels[i];
			if (!px.vp_valid) continue;
			for (int d = 0; d < 3; ++d) {
				gMin[d] = std::min(gMin[d], px.vp_p[d] - px.radius);
				gMax[d] = std::max(gMax[d], px.vp_p[d] + px.radius);
			}
			maxR = std::max(maxR, px.radius);
		}
		// Guard against degenerate scene
		for (int d = 0; d < 3; ++d) {
			if (gMin[d] > gMax[d]) { gMin[d] = T(0); gMax[d] = T(1); }
			if (gMax[d] - gMin[d] < T(1e-6)) gMax[d] = gMin[d] + T(1e-6);
			gridMin[d] = gMin[d];
			gridMax[d] = gMax[d];
		}
		if (maxR <= T(0)) maxR = T(1);

		// Compute grid resolution (mirrors pbrt-v4)
		T diag[3] = { gridMax[0]-gridMin[0],
					  gridMax[1]-gridMin[1],
					  gridMax[2]-gridMin[2] };
		T maxDiag = std::max({diag[0], diag[1], diag[2]});
		int baseRes = std::max(1, (int)(maxDiag / maxR));
		for (int d = 0; d < 3; ++d) {
			gridRes[d] = std::max(1, (int)(baseRes * diag[d] / maxDiag));
			cellSize[d] = diag[d] / (T)gridRes[d];
		}

		// Allocate hash table
		hashSize = NextPrime((int)pixels.size());
		grid.assign(hashSize, std::forward_list<int>());

		// Insert each valid visible point into all overlapping cells
		for (int i = 0; i < (int)pixels.size(); ++i) {
			const auto& px = pixels[i];
			if (!px.vp_valid) continue;
			T r = px.radius;
			int pMin[3], pMax[3];
			ToGrid(px.vp_p[0]-r, px.vp_p[1]-r, px.vp_p[2]-r, pMin);
			ToGrid(px.vp_p[0]+r, px.vp_p[1]+r, px.vp_p[2]+r, pMax);
			for (int z = pMin[2]; z <= pMax[2]; ++z)
			for (int y = pMin[1]; y <= pMax[1]; ++y)
			for (int x = pMin[0]; x <= pMax[0]; ++x) {
				int h = (int)(HashPoint3i(x,y,z) % (uint32_t)hashSize);
				grid[h].push_front(i);
			}
		}
	}

	// Map a world-space coordinate to a grid cell (clamped)
	void ToGrid(T wx, T wy, T wz, int out[3]) const {
		T pg[3] = { (wx - gridMin[0]) / (gridMax[0] - gridMin[0]),
					(wy - gridMin[1]) / (gridMax[1] - gridMin[1]),
					(wz - gridMin[2]) / (gridMax[2] - gridMin[2]) };
		for (int d = 0; d < 3; ++d) {
			out[d] = (int)(gridRes[d] * pg[d]);
			if (out[d] < 0) out[d] = 0;
			if (out[d] >= gridRes[d]) out[d] = gridRes[d] - 1;
		}
	}

	// Query: return bucket index for a world-space photon position
	int Bucket(T wx, T wy, T wz) const {
		int gi[3];
		ToGrid(wx, wy, wz, gi);
		return (int)(HashPoint3i(gi[0], gi[1], gi[2]) % (uint32_t)hashSize);
	}

	const std::forward_list<int>& CellList(int h) const { return grid[h]; }
};

} // namespace sppm_detail

// ===========================================================================
// Section 3 -- SPPMCameraPass
// ===========================================================================
// Traces one camera ray per pixel, finds the first diffuse (non-specular)
// surface, records the visible point, and accumulates direct lighting Ld.
//
// Mirrors pbrt-v4's camera sub-pass inside SPPMIntegrator::Render:
//   - Follow specular bounces until a diffuse hit (or max depth)
//   - At each specular bounce accumulate emission * beta
//   - At diffuse hit: set pixel.vp_*, pixel.Ld = DirectLight(...)
//
// Parameters:
//   pixels    -- vector of SPPMPixel<T>, indexed by pixel index
//   width,height -- image dimensions
//   maxDepth  -- maximum specular chain length before giving up
//   scene     -- scene query object

template<typename T, typename Scene>
void SPPMCameraPass(std::vector<SPPMPixel<T>>& pixels,
					int width, int height, int maxDepth,
					const Scene& scene) {
	for (int iy = 0; iy < height; ++iy) {
		for (int ix = 0; ix < width; ++ix) {
			int idx = iy * width + ix;
			SPPMPixel<T>& pixel = pixels[idx];
			pixel.px = ix;
			pixel.py = iy;
			pixel.vp_valid = false;
			// NOTE: Ld is NOT reset here. It accumulates across all iterations
			// and is divided by nIterations in SPPMFinalImage, mirroring
			// pbrt-v4: pixel.Ld / (iter+1) + tau/(...)

			// Compute normalised pixel centre
			T px = (T(ix) + T(0.5)) / T(width);
			T py = (T(iy) + T(0.5)) / T(height);

			// Camera ray from pixel position
			T cam_p[3], ray_d[3], cam_n[3];
			if (!scene.PixelToRay(px, py, cam_p, ray_d, cam_n))
				continue;

			// Trace the camera ray; follow specular bounces
			T org[3] = { cam_p[0], cam_p[1], cam_p[2] };
			T dir[3] = { ray_d[0], ray_d[1], ray_d[2] };
			T beta[3] = { T(1), T(1), T(1) };

			for (int depth = 0; depth < maxDepth; ++depth) {
				BDPTHit<T> hit{};
				if (!scene.Intersect(org, dir, std::numeric_limits<T>::max(), hit))
					break;

				// Accumulate emission from hit surface (area light)
				pixel.Ld[0] += beta[0] * hit.area_Le[0];
				pixel.Ld[1] += beta[1] * hit.area_Le[1];
				pixel.Ld[2] += beta[2] * hit.area_Le[2];

				if (hit.is_delta_bsdf) {
					// Specular bounce: sample BSDF and continue
					T new_dir[3], f_val[3], pdf;
					bool is_spec;
					T u1 = scene.RandFloat(), u2 = scene.RandFloat();
					if (!scene.BSDFSampleF(hit.bsdf_id, hit.wo, hit.shading_n,
										   u1, u2, new_dir, f_val, pdf, is_spec))
						break;
					if (pdf <= T(0)) break;
					T cosI = std::abs(new_dir[0]*hit.shading_n[0] +
									  new_dir[1]*hit.shading_n[1] +
									  new_dir[2]*hit.shading_n[2]);
					beta[0] *= f_val[0] * cosI / pdf;
					beta[1] *= f_val[1] * cosI / pdf;
					beta[2] *= f_val[2] * cosI / pdf;
					org[0]=hit.p[0]; org[1]=hit.p[1]; org[2]=hit.p[2];
					dir[0]=new_dir[0]; dir[1]=new_dir[1]; dir[2]=new_dir[2];
				} else {
					// Diffuse hit: record visible point
					pixel.vp_valid = true;
					pixel.vp_p[0]=hit.p[0]; pixel.vp_p[1]=hit.p[1]; pixel.vp_p[2]=hit.p[2];
					pixel.vp_wo[0]=hit.wo[0]; pixel.vp_wo[1]=hit.wo[1]; pixel.vp_wo[2]=hit.wo[2];
					pixel.vp_n[0]=hit.shading_n[0]; pixel.vp_n[1]=hit.shading_n[1]; pixel.vp_n[2]=hit.shading_n[2];
					pixel.vp_beta[0]=beta[0]; pixel.vp_beta[1]=beta[1]; pixel.vp_beta[2]=beta[2];
					pixel.vp_bsdf_id = hit.bsdf_id;

					// Direct lighting at the visible point
					T ld[3];
					scene.DirectLight(hit.p, hit.shading_n, hit.wo, hit.bsdf_id, ld);
					pixel.Ld[0] += beta[0] * ld[0];
					pixel.Ld[1] += beta[1] * ld[1];
					pixel.Ld[2] += beta[2] * ld[2];
					break;
				}
			}
		}
	}
}

// ===========================================================================
// Section 4 -- SPPMPhotonPass
// ===========================================================================
// Shoots nPhotons photons from lights, traverses the scene, and for each
// diffuse surface interaction looks up nearby visible points in the hash
// grid and accumulates Phi and m.
//
// Mirrors pbrt-v4's photon sub-pass:
//   - Sample light source, emit photon ray, carry beta
//   - At each non-specular surface hit (depth > 0): lookup hash grid,
//     add contribution to pixel.Phi[i] and pixel.m
//   - Sample new photon direction via BSDF; Russian roulette termination
//
// NOTE: depth > 0 check mirrors pbrt-v4 (first bounce is left for direct
// lighting in the camera pass).

template<typename T, typename Scene>
void SPPMPhotonPass(std::vector<SPPMPixel<T>>& pixels,
					const sppm_detail::HashGrid<T>& grid,
					int nPhotons, int maxDepth,
					const Scene& scene) {
	for (int ph = 0; ph < nPhotons; ++ph) {
		// Sample a light to emit from
		BDPTLightLeSample<T> les{};
		T u1 = scene.RandFloat();
		T u2a[2] = { scene.RandFloat(), scene.RandFloat() };
		T u2b[2] = { scene.RandFloat(), scene.RandFloat() };
		if (!scene.SampleLightLe(u1, u2a, u2b, les)) continue;
		if (les.pdf_pos <= T(0) || les.pdf_dir <= T(0)) continue;

		// Initial photon beta = Le * |cos| / (p_l * pdfPos * pdfDir)
		T beta[3];
		for (int c = 0; c < 3; ++c)
			beta[c] = les.L[c] * les.abs_cos_theta / (les.pdf_pos * les.pdf_dir);

		T org[3] = { les.ray_o[0], les.ray_o[1], les.ray_o[2] };
		T dir[3] = { les.ray_d[0], les.ray_d[1], les.ray_d[2] };

		for (int depth = 0; depth < maxDepth; ++depth) {
			BDPTHit<T> hit{};
			if (!scene.Intersect(org, dir, std::numeric_limits<T>::max(), hit))
				break;

			if (depth > 0 && !hit.is_delta_bsdf) {
				// Look up the hash grid bucket for this photon position
				int h = grid.Bucket(hit.p[0], hit.p[1], hit.p[2]);
				for (int pidx : grid.CellList(h)) {
					SPPMPixel<T>& px = pixels[pidx];
					if (!px.vp_valid) continue;
					T dx = px.vp_p[0] - hit.p[0];
					T dy = px.vp_p[1] - hit.p[1];
					T dz = px.vp_p[2] - hit.p[2];
					if (dx*dx + dy*dy + dz*dz > px.radius * px.radius) continue;

					// Phi += beta * bsdf.f(vp.wo, -photon_dir)
					T wi[3] = { -dir[0], -dir[1], -dir[2] };
					T f[3];
					scene.BSDFf(px.vp_bsdf_id, px.vp_wo, wi, px.vp_n, f);
					for (int c = 0; c < 3; ++c)
						px.Phi[c] += beta[c] * px.vp_beta[c] * f[c];
					px.m++;
				}
			}

			// Sample new photon direction
			T new_dir[3], f_val[3], pdf;
			bool is_spec;
			T su1 = scene.RandFloat(), su2 = scene.RandFloat();
			if (!scene.BSDFSampleF(hit.bsdf_id, hit.wo, hit.shading_n,
								   su1, su2, new_dir, f_val, pdf, is_spec))
				break;
			if (pdf <= T(0)) break;

			T cosI = std::abs(new_dir[0]*hit.shading_n[0] +
							  new_dir[1]*hit.shading_n[1] +
							  new_dir[2]*hit.shading_n[2]);
			T beta_new[3];
			for (int c = 0; c < 3; ++c)
				beta_new[c] = beta[c] * f_val[c] * cosI / pdf;

			// Russian roulette (pbrt-v4: q = max(0, 1 - bnew.max/beta.max))
			T betaMax = std::max({beta[0], beta[1], beta[2]});
			T betaNewMax = std::max({beta_new[0], beta_new[1], beta_new[2]});
			T q = T(0);
			if (betaMax > T(0))
				q = std::max(T(0), T(1) - betaNewMax / betaMax);
			if (scene.RandFloat() < q) break;
			T invSurv = T(1) / (T(1) - q);
			for (int c = 0; c < 3; ++c) beta[c] = beta_new[c] * invSurv;

			org[0]=hit.p[0]; org[1]=hit.p[1]; org[2]=hit.p[2];
			dir[0]=new_dir[0]; dir[1]=new_dir[1]; dir[2]=new_dir[2];
		}
	}
}

// ===========================================================================
// Section 5 -- SPPMUpdateRadius
// ===========================================================================
// Apply the progressive radius contraction after each photon pass.
// Mirrors pbrt-v4:
//   gamma = 2/3
//   nNew  = n + gamma * m
//   rNew  = r * sqrt(nNew / (n + m))
//   tau   = (tau + Phi) * (rNew/r)^2
//   n     = nNew, r = rNew, m = 0, Phi = 0

template<typename T>
void SPPMUpdateRadius(std::vector<SPPMPixel<T>>& pixels) {
	static constexpr T kGamma = T(2) / T(3);
	for (auto& px : pixels) {
		// Apply contraction only when photons were accumulated this pass
		// (mirrors pbrt-v4: `if (int m = p.m.load(...); m > 0)`)
		if (px.m > 0) {
			T nNew = px.n + kGamma * (T)px.m;
			T rNew = px.radius * std::sqrt(nNew / (px.n + (T)px.m));

			T scale = (rNew * rNew) / (px.radius * px.radius);
			for (int c = 0; c < 3; ++c)
				px.tau[c] = (px.tau[c] + px.Phi[c]) * scale;

			px.n      = nNew;
			px.radius = rNew;
			px.m      = 0;
			px.Phi[0] = px.Phi[1] = px.Phi[2] = T(0);
		}
		// Always reset the visible point for the next camera pass
		// (mirrors pbrt-v4: `p.vp.beta = SampledSpectrum(0.)` unconditionally)
		px.vp_valid = false;
	}
}

// ===========================================================================
// Section 6 -- SPPMFinalImage
// ===========================================================================
// Reconstruct the final pixel radiance.
// Mirrors pbrt-v4:
//   L = Ld + tau / (n * totalPhotonPaths * pi * r^2)
//
// Parameters:
//   pixels           -- final pixel array
//   nIterations      -- total number of SPPM iterations (used to normalise Ld)
//   totalPhotonPaths -- nPhotons * nIterations
//   out_rgb          -- flat array [width*height*3], row-major, R/G/B interleaved
//
// Mirrors pbrt-v4:
//   L = pixel.Ld / (iter+1) + pixel.tau / (np * Pi * r^2)

template<typename T>
void SPPMFinalImage(const std::vector<SPPMPixel<T>>& pixels,
					int nIterations,
					int64_t totalPhotonPaths,
					std::vector<T>& out_rgb) {
	static constexpr T kPi = T(3.14159265358979323846);
	out_rgb.resize(pixels.size() * 3, T(0));
	T invIter = (nIterations > 0) ? T(1) / (T)nIterations : T(0);
	for (int i = 0; i < (int)pixels.size(); ++i) {
		const SPPMPixel<T>& px = pixels[i];
		T denom = px.n * (T)totalPhotonPaths * kPi * px.radius * px.radius;
		for (int c = 0; c < 3; ++c) {
			T indirect = (denom > T(0)) ? px.tau[c] / denom : T(0);
			out_rgb[i*3 + c] = px.Ld[c] * invIter + indirect;
		}
	}
}

// ===========================================================================
// Section 7 -- SPPMRender: full progressive render loop
// ===========================================================================
// Ties together camera pass, hash-grid build, photon pass, radius update,
// and final reconstruction over nIterations.
//
// Parameters:
//   width, height    -- image dimensions in pixels
//   nIterations      -- number of SPPM iterations (more = less noise)
//   nPhotons         -- photons shot per iteration
//   maxDepth         -- maximum ray/photon path depth
//   initialRadius    -- initial search radius (pbrt default: scene-dependent)
//   scene            -- scene query object
//   out_rgb          -- output image, size width*height*3, after completion

template<typename T, typename Scene>
void SPPMRender(int width, int height,
				int nIterations, int nPhotons, int maxDepth,
				T initialRadius,
				const Scene& scene,
				std::vector<T>& out_rgb) {
	int nPixels = width * height;

	// Initialise pixels with starting radius
	std::vector<SPPMPixel<T>> pixels(nPixels);
	for (int i = 0; i < nPixels; ++i) {
		pixels[i].radius = initialRadius;
		pixels[i].px = i % width;
		pixels[i].py = i / width;
	}

	int64_t totalPhotonPaths = 0;

	for (int iter = 0; iter < nIterations; ++iter) {
		// 1. Camera pass: deposit visible points and compute direct lighting
		SPPMCameraPass(pixels, width, height, maxDepth, scene);

		// 2. Build spatial hash grid over visible points
		sppm_detail::HashGrid<T> hashGrid;
		hashGrid.Build(pixels);

		// 3. Photon pass: shoot photons, accumulate Phi/m
		SPPMPhotonPass(pixels, hashGrid, nPhotons, maxDepth, scene);
		totalPhotonPaths += nPhotons;

		// 4. Update radius and tau
		SPPMUpdateRadius(pixels);
	}

	// 5. Reconstruct final image
	// Pass nIterations so Ld is divided correctly (mirrors pbrt-v4: Ld/(iter+1))
	SPPMFinalImage(pixels, nIterations, totalPhotonPaths, out_rgb);
}
