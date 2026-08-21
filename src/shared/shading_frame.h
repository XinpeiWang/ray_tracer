#pragma once
// ---------------------------------------------------------------------------
// shading_frame.h -- Shared CPU/GPU orthonormal basis (shading frame)
//
// Mirrors pbrt-v4's BSDF shading-frame responsibility:
//   - Build a local frame from a surface normal
//   - to_local():  world-space direction  --> local frame (z = normal)
//   - to_world():  local frame direction  --> world space
//
// Design rules (same as bxdfs.h):
//   - No virtual functions, no heap allocation
//   - Template parameter T: double on CPU, float on GPU
//   - CPU_GPU macro: __host__ __device__ under NVCC, inline otherwise
// ---------------------------------------------------------------------------

#include "cpu_gpu.h"
#if !defined(__CUDACC__)
#   include <cmath>
#endif

// No math library needed for from_normal()'s Duff et al. ONB -- only
// from_dpdu() below (Round 6 Phase 3) needs a square root.

template<typename T>
struct ShadingFrame {
	T tx, ty, tz;   // tangent   (x-axis of local frame)
	T bx, by, bz;   // bitangent (y-axis of local frame)
	T nx, ny, nz;   // normal    (z-axis of local frame = shading normal)

	// Build an orthonormal frame from an explicit tangent x and normal z.
	// Mirrors pbrt-v4 Frame::FromXZ(x, z): t=x, b=cross(z,x), n=z.
	// x and z must be unit and orthogonal (caller's responsibility).
	// Used by BSDF to align the shading frame with the surface tangent dpdu.
	CPU_GPU static ShadingFrame from_xz(T tx_, T ty_, T tz_,
										 T nx_, T ny_, T nz_) {
		ShadingFrame f;
		f.tx = tx_; f.ty = ty_; f.tz = tz_;
		// bitangent = cross(z, x)  (right-hand rule, matches pbrt-v4 FromXZ)
		f.bx = ny_*tz_ - nz_*ty_;
		f.by = nz_*tx_ - nx_*tz_;
		f.bz = nx_*ty_ - ny_*tx_;
		f.nx = nx_; f.ny = ny_; f.nz = nz_;
		return f;
	}

	// Build an orthonormal frame from a unit outward surface normal.
	// Uses the Frisvad / Duff et al. (2017) branch-free method, matching
	// pbrt-v4 CoordinateSystem() in util/vecmath.h.  Numerically stable
	// for all input directions including n.z near -1.
	CPU_GPU static ShadingFrame from_normal(T nx_, T ny_, T nz_) {
		ShadingFrame f;
		f.nx = nx_; f.ny = ny_; f.nz = nz_;

		// Duff et al. 2017 -- avoids the abs(n.x) > 0.9 discontinuity
		T sign = (nz_ >= T(0)) ? T(1) : T(-1);
		T a    = T(-1) / (sign + nz_);
		T b    = nx_ * ny_ * a;

		// tangent
		f.tx = T(1) + sign * nx_ * nx_ * a;
		f.ty = sign * b;
		f.tz = -sign * nx_;

		// bitangent
		f.bx = b;
		f.by = sign + ny_ * ny_ * a;
		f.bz = -ny_;

		return f;
	}

	// Round 6 Phase 3: build a frame aligned to the surface's real
	// parametric tangent (dpdu) instead of from_normal()'s arbitrary basis
	// -- needed so an anisotropic GGX distribution's alpha_x/alpha_y axes
	// actually point along the surface's own U/V directions (a brushed-
	// metal material looks the same regardless of shading-frame choice
	// under isotropic roughness, but rotates visibly with the frame under
	// anisotropic roughness, so from_normal()'s per-pixel-arbitrary tangent
	// would make the anisotropy direction look randomly rotated instead of
	// following the surface).
	//
	// dpdu need not be unit or exactly orthogonal to the normal (every
	// shape computes it independently, e.g. as a raw parametric derivative)
	// -- Gram-Schmidt against the normal first, matching how a real
	// surface tangent is turned into a shading tangent in pbrt-v4's own
	// SurfaceInteraction::SetShadingGeometry(). Falls back to from_normal()
	// when the projected tangent is degenerate (near-zero length, e.g.
	// exactly at a UV pole or a zero-derivative parametrization) -- an
	// arbitrary consistent frame is equally valid there since the surface
	// itself has no well-defined tangent direction to align to.
	CPU_GPU static ShadingFrame from_dpdu(T dpdu_x, T dpdu_y, T dpdu_z,
										   T nx_, T ny_, T nz_) {
		T d  = dpdu_x*nx_ + dpdu_y*ny_ + dpdu_z*nz_;
		T tx_ = dpdu_x - d*nx_;
		T ty_ = dpdu_y - d*ny_;
		T tz_ = dpdu_z - d*nz_;
		T len2 = tx_*tx_ + ty_*ty_ + tz_*tz_;
		if (len2 < T(1e-12)) return from_normal(nx_, ny_, nz_);
#if defined(__CUDACC__)
		T inv_len = T(1) / sqrtf(len2);
#else
		T inv_len = T(1) / std::sqrt(len2);
#endif
		return from_xz(tx_*inv_len, ty_*inv_len, tz_*inv_len, nx_, ny_, nz_);
	}

	// Convert a world-space direction to the local frame (z = normal).
	// Mirrors pbrt-v4 Frame::ToLocal().
	CPU_GPU void to_local(T wx, T wy, T wz,
						  T& lx, T& ly, T& lz) const {
		lx = wx*tx + wy*ty + wz*tz;
		ly = wx*bx + wy*by + wz*bz;
		lz = wx*nx + wy*ny + wz*nz;
	}

	// Convert a local-frame direction back to world space.
	// Mirrors pbrt-v4 Frame::FromLocal().
	CPU_GPU void to_world(T lx, T ly, T lz,
						  T& wx, T& wy, T& wz) const {
		wx = lx*tx + ly*bx + lz*nx;
		wy = lx*ty + ly*by + lz*ny;
		wz = lx*tz + ly*bz + lz*nz;
	}
};
