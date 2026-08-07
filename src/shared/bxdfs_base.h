#pragma once
// ---------------------------------------------------------------------------
// bxdfs.h -- Shared CPU/GPU BxDF math library (mirrors pbrt-v4 bxdfs.h)
#include "rng.h"
//
// Each BxDF is a plain template struct marked CPU_GPU.
// Design rules (enabling identical CPU + GPU use):
//   - No virtual functions, no heap allocation, no shared_ptr
//   - No RNG calls inside -- caller passes pre-generated random values
//   - Template parameter T: double on CPU, float on GPU
//   - All functions are CPU_GPU (compiles as __host__ __device__ under NVCC)
//
// Caller interface:
//   BxDFSampleResult<T> r = SomeBxDF<T>{...}.sample(wi_x,wi_y,wi_z,
//                                                     normal_x,...,
//                                                     u1, u2, u3, u4);
//   T pdf = SomeBxDF<T>{...}.scattering_pdf(wi_x,..., wo_x,...);
//
// BxDFSampleResult fields:
//   wo_x, wo_y, wo_z  -- sampled outgoing direction (world space)
//   r, g, b           -- attenuation / weight
//   is_specular       -- true if specular bounce (skip MIS)
//   valid             -- false if sample should be rejected
// ---------------------------------------------------------------------------

#include "cpu_gpu.h"

#include "fresnel.h"
#include "microfacet.h"
#include "sampling.h"

#if defined(__CUDACC__)
#   include <math_functions.h>
#else
#   include <cmath>
#endif

// ---------------------------------------------------------------------------
// BxDFSampleResult<T>
// Carries the result of BxDF::sample() in a plain-old-data struct.
// ---------------------------------------------------------------------------
template<typename T>
struct BxDFSampleResult {
	T wo_x, wo_y, wo_z;  // sampled outgoing direction (world/local space as caller chooses)
	T r, g, b;            // attenuation (BRDF weight)
	T eta;                // IOR ratio eta_i/eta_t (1.0 for non-transmissive; pbrt-v4 bs->eta)
	bool is_specular;     // true -> skip MIS next bounce
	bool is_transmission; // true -> refraction occurred (update etaScale in integrator)
	bool valid;           // false -> reject sample (below-surface, etc.)
};

// ---------------------------------------------------------------------------
// Shading-frame helpers shared across BxDFs
// ---------------------------------------------------------------------------
template<typename T>
CPU_GPU void make_onb(T nx, T ny, T nz,
					  T& tx, T& ty, T& tz,
					  T& bx, T& by, T& bz)
{
	// Build orthonormal basis from normal (Frisvad / Pixar variant)
#if defined(__CUDACC__)
	T ax = fabsf(nx), ay = fabsf(ny), az = fabsf(nz);
#else
	T ax = std::fabs(nx), ay = std::fabs(ny), az = std::fabs(nz);
#endif
	// Pick axis least aligned with n
	if (ax > T(0.9)) { tx = T(0); ty = T(1); tz = T(0); }
	else             { tx = T(1); ty = T(0); tz = T(0); }
	// tangent = normalize(cross(t, n))
	T cx = ty*nz - tz*ny, cy = tz*nx - tx*nz, cz = tx*ny - ty*nx;
#if defined(__CUDACC__)
	T len = sqrtf(cx*cx + cy*cy + cz*cz);
#else
	T len = std::sqrt(cx*cx + cy*cy + cz*cz);
#endif
	tx = cx/len; ty = cy/len; tz = cz/len;
	// bitangent = cross(n, tangent)
	bx = ny*tz - nz*ty; by = nz*tx - nx*tz; bz = nx*ty - ny*tx;
}

// Cosine-weighted hemisphere sample in world space via concentric-disk mapping.
// Mirrors pbrt-v4 SampleCosineHemisphere (util/sampling.h).
// u1, u2 in [0,1). Returns normalised direction in world space.
template<typename T>
CPU_GPU void cosine_hemisphere_sample(
	T nx, T ny, T nz,   // surface normal (unit)
	T tx, T ty, T tz,   // tangent (unit)
	T bx, T by, T bz,   // bitangent (unit)
	T u1, T u2,         // random [0,1)
	T& dx, T& dy, T& dz)
{
	T lx, ly, lz;
	T pdf_unused;
	SampleCosineHemisphere(u1, u2, lx, ly, lz, pdf_unused);
	dx = lx*tx + ly*bx + lz*nx;
	dy = lx*ty + ly*by + lz*ny;
	dz = lx*tz + ly*bz + lz*nz;
}

// Reflect wi about normal (all vectors in same coordinate system).
template<typename T>
CPU_GPU void reflect_dir(T wix, T wiy, T wiz,
						 T nx,  T ny,  T nz,
						 T& rox, T& roy, T& roz)
{
	T d = T(2) * (wix*nx + wiy*ny + wiz*nz);
	rox = wix - d*nx; roy = wiy - d*ny; roz = wiz - d*nz;
}

// Refract wi through interface with normal and eta = eta_i/eta_t.
// Returns false on TIR.
template<typename T>
CPU_GPU bool refract_dir(T wix, T wiy, T wiz,     // incident direction (into surface)
						  T nx,  T ny,  T nz,      // surface normal (outward)
						  T eta,                   // eta_i / eta_t
						  T& rox, T& roy, T& roz)
{
	T cos_i = -(wix*nx + wiy*ny + wiz*nz);
#if defined(__CUDACC__)
	cos_i = fabsf(cos_i);  // ensure positive (handles back-face rays)
#else
	cos_i = std::fabs(cos_i);
#endif
	T sin2t = eta*eta * (T(1) - cos_i*cos_i);
	if (sin2t >= T(1)) return false; // TIR
#if defined(__CUDACC__)
	T cos_t = sqrtf(T(1) - sin2t);
#else
	T cos_t = std::sqrt(T(1) - sin2t);
#endif
	rox = eta*wix + (eta*cos_i - cos_t)*nx;
	roy = eta*wiy + (eta*cos_i - cos_t)*ny;
	roz = eta*wiz + (eta*cos_i - cos_t)*nz;
	return true;
}

// Normalize a 3-vector in place.
template<typename T>
CPU_GPU void normalize3(T& x, T& y, T& z) {
#if defined(__CUDACC__)
	T len = sqrtf(x*x + y*y + z*z);
#else
	T len = std::sqrt(x*x + y*y + z*z);
#endif
	if (len > T(1e-8)) { x /= len; y /= len; z /= len; }
}

// ===========================================================================
// 1. DiffuseBxDF  (Lambertian diffuse reflection)
//    Mirrors pbrt-v4 DiffuseBxDF
//    f(wo,wi) = albedo / pi
//    PDF      = cos(theta_wi) / pi
// ===========================================================================
template<typename T>
