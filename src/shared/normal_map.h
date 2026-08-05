#pragma once
//==============================================================================
// normal_map.h -- Normal mapping and bump mapping math (pbrt-v4 aligned)
//
// pbrt-v4 alignment
// -----------------
// pbrt-v4 implements both techniques in materials.h:
//
//   NormalMap()  -- samples an RGB tangent-space normal map texture, decodes
//                   the normal vector, transforms it to world space using the
//                   surface shading frame (dpdu, n), then recomputes dpdu/dpdv
//                   via Gram-Schmidt so the new shading normal is consistent.
//                   Reference: pbrt-v4 materials.h:86-105
//
//   BumpMap()    -- evaluates a scalar displacement texture at the hit point
//                   and at two finite-difference offsets along dpdu and dpdv,
//                   then perturbs dpdu/dpdv using the displacement gradient.
//                   The new shading normal is cross(dpdu_new, dpdv_new).
//                   Reference: pbrt-v4 materials.h:107-145
//
// This file provides CPU/GPU-portable implementations that operate on plain
// scalar fields (no pbrt Image/TextureEvaluator dependencies).  The caller
// is responsible for sampling the texture.
//
// Inputs (shared by both functions):
//   nx,ny,nz         -- geometric surface normal (world space, unit)
//   dpdu_x/y/z       -- surface tangent along U (world space, not necessarily unit)
//                       pbrt-v4: ctx.shading.dpdu
//   dpdv_x/y/z       -- surface tangent along V (world space)
//                       pbrt-v4: ctx.shading.dpdv
//
// For NormalMap, the caller also provides the decoded tangent-space normal.
// For BumpMap, the caller provides three displacement values.
//
// Output:
//   perturbed_nx/ny/nz -- new shading normal (world space, unit)
//==============================================================================

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU
#   endif
#endif

#if defined(__CUDACC__)
#   include <math_functions.h>
#else
#   include <cmath>
#endif

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

template<typename T>
CPU_GPU T nm_dot(T ax, T ay, T az, T bx, T by, T bz) {
	return ax*bx + ay*by + az*bz;
}

template<typename T>
CPU_GPU void nm_cross(T ax, T ay, T az, T bx, T by, T bz,
					  T& cx, T& cy, T& cz) {
	cx = ay*bz - az*by;
	cy = az*bx - ax*bz;
	cz = ax*by - ay*bx;
}

template<typename T>
CPU_GPU void nm_normalize(T& x, T& y, T& z) {
#if defined(__CUDACC__)
	T len = sqrtf(x*x + y*y + z*z);
#else
	T len = std::sqrt(x*x + y*y + z*z);
#endif
	if (len > T(1e-8)) { x /= len; y /= len; z /= len; }
}

// Gram-Schmidt: project v onto the plane perpendicular to n.
// Returns v - dot(v,n)*n, does not normalise.
template<typename T>
CPU_GPU void nm_gram_schmidt(T vx, T vy, T vz,
							 T nx, T ny, T nz,
							 T& ox, T& oy, T& oz) {
	T d = nm_dot(vx, vy, vz, nx, ny, nz);
	ox = vx - d*nx;
	oy = vy - d*ny;
	oz = vz - d*nz;
}

// ---------------------------------------------------------------------------
// apply_normal_map
//
// pbrt-v4 equivalent: NormalMap() in materials.h
//
// Decode a tangent-space normal (ns_t) and transform it to world space using
// the surface shading frame (dpdu, n), then return the perturbed shading normal.
//
// Parameters:
//   ns_tx/y/z  -- tangent-space normal decoded from the map, range [-1,1]^3
//                 (caller: decode as 2*RGB - 1, then normalise)
//   nx/ny/nz   -- geometric shading normal (world space, unit)
//   dpdu_x/y/z -- surface tangent along U (world space, may be non-unit)
//
// Output:
//   out_nx/ny/nz -- perturbed shading normal (world space, unit)
//
// Steps (identical to pbrt-v4):
//   1. Build Frame from (normalize(dpdu), n)  -- tangent + normal define X/Z axes
//   2. Transform ns_t from local frame to world frame
//   3. Normalise the result
// ---------------------------------------------------------------------------
template<typename T>
CPU_GPU void apply_normal_map(
	T ns_tx, T ns_ty, T ns_tz,   // decoded tangent-space normal
	T nx,    T ny,    T nz,      // geometric shading normal (unit)
	T dpdu_x, T dpdu_y, T dpdu_z, // surface tangent along U
	T& out_nx, T& out_ny, T& out_nz)
{
	// Normalise dpdu to get the tangent axis (X axis of the shading frame)
	T tx = dpdu_x, ty = dpdu_y, tz = dpdu_z;
	nm_normalize(tx, ty, tz);

	// Gram-Schmidt: make tangent perpendicular to normal (numerical safety)
	T gx, gy, gz;
	nm_gram_schmidt(tx, ty, tz, nx, ny, nz, gx, gy, gz);
	nm_normalize(gx, gy, gz);

	// Bitangent = cross(n, tangent)  -- Z x X = Y in right-handed frame
	T bx, by, bz;
	nm_cross(nx, ny, nz, gx, gy, gz, bx, by, bz);
	nm_normalize(bx, by, bz);

	// Transform ns_t from tangent space to world space:
	//   world_normal = ns_t.x * tangent + ns_t.y * bitangent + ns_t.z * normal
	// (pbrt-v4: frame.FromLocal(ns))
	out_nx = ns_tx * gx + ns_ty * bx + ns_tz * nx;
	out_ny = ns_tx * gy + ns_ty * by + ns_tz * ny;
	out_nz = ns_tx * gz + ns_ty * bz + ns_tz * nz;
	nm_normalize(out_nx, out_ny, out_nz);

	// Ensure the perturbed normal is on the same side as the geometric normal
	// (avoids flipping when the map contains extreme values)
	if (nm_dot(out_nx, out_ny, out_nz, nx, ny, nz) < T(0)) {
		out_nx = -out_nx; out_ny = -out_ny; out_nz = -out_nz;
	}
}

// ---------------------------------------------------------------------------
// apply_bump_map
//
// pbrt-v4 equivalent: BumpMap() in materials.h
//
// Perturb the shading normal using finite differences of a displacement field.
//
// Parameters:
//   disp       -- displacement at the hit point
//   disp_u     -- displacement at hit point + du along dpdu
//   disp_v     -- displacement at hit point + dv along dpdv
//   du, dv     -- finite-difference step sizes (pbrt uses 0.5*(|dudx|+|dudy|))
//   nx/ny/nz   -- geometric shading normal (unit)
//   dpdu_x/y/z -- surface tangent along U (world space)
//   dpdv_x/y/z -- surface tangent along V (world space)
//
// Output:
//   out_nx/ny/nz -- perturbed shading normal (world space, unit)
//
// Math (identical to pbrt-v4):
//   dpdu_new = dpdu + (disp_u - disp) / du * n + disp * dndu  (dndu≈0 here)
//   dpdv_new = dpdv + (disp_v - disp) / dv * n + disp * dndv  (dndv≈0 here)
//   n_new    = normalize(cross(dpdu_new, dpdv_new))
//
// Note: we omit the dndu/dndv terms (normal curvature derivatives) as they
// are not tracked in hit_record.  This matches the simplified version used
// in many renderers and is accurate for flat/low-curvature surfaces.
// ---------------------------------------------------------------------------
template<typename T>
CPU_GPU void apply_bump_map(
	T disp, T disp_u, T disp_v,  // displacement samples
	T du,    T dv,                // finite-difference steps
	T nx,    T ny,    T nz,       // geometric shading normal (unit)
	T dpdu_x, T dpdu_y, T dpdu_z, // surface tangent along U
	T dpdv_x, T dpdv_y, T dpdv_z, // surface tangent along V
	T& out_nx, T& out_ny, T& out_nz)
{
	// Perturb dpdu and dpdv by the displacement gradient
	// pbrt-v4: *dpdu = dpdu + (uDisplace-disp)/du * n  (+  disp * dndu term omitted)
	T grad_u = (disp_u - disp) / du;
	T grad_v = (disp_v - disp) / dv;

	T new_dpdu_x = dpdu_x + grad_u * nx;
	T new_dpdu_y = dpdu_y + grad_u * ny;
	T new_dpdu_z = dpdu_z + grad_u * nz;

	T new_dpdv_x = dpdv_x + grad_v * nx;
	T new_dpdv_y = dpdv_y + grad_v * ny;
	T new_dpdv_z = dpdv_z + grad_v * nz;

	// New normal = cross(dpdu_new, dpdv_new)
	nm_cross(new_dpdu_x, new_dpdu_y, new_dpdu_z,
			 new_dpdv_x, new_dpdv_y, new_dpdv_z,
			 out_nx, out_ny, out_nz);
	nm_normalize(out_nx, out_ny, out_nz);

	// Fallback: if cross product degenerates, keep geometric normal
#if defined(__CUDACC__)
	T len2 = out_nx*out_nx + out_ny*out_ny + out_nz*out_nz;
	if (len2 < T(1e-8)) { out_nx = nx; out_ny = ny; out_nz = nz; }
#else
	if (!std::isfinite(out_nx) || !std::isfinite(out_ny) || !std::isfinite(out_nz)
		|| out_nx*out_nx + out_ny*out_ny + out_nz*out_nz < T(1e-8)) {
		out_nx = nx; out_ny = ny; out_nz = nz;
	}
#endif

	// Ensure the bumped normal faces the same side as the geometric normal
	if (nm_dot(out_nx, out_ny, out_nz, nx, ny, nz) < T(0)) {
		out_nx = -out_nx; out_ny = -out_ny; out_nz = -out_nz;
	}
}
