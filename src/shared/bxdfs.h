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

#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU inline
#   endif
#endif

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
struct DiffuseBxDF {
	T albedo_r, albedo_g, albedo_b;

	// sample: cosine-hemisphere in world space using pre-built ONB
	// u1, u2 in [0,1)
	CPU_GPU BxDFSampleResult<T> sample(
		T nx, T ny, T nz,   // surface normal
		T u1, T u2) const
	{
		BxDFSampleResult<T> res{};
		T tx, ty, tz, bx, by, bz;
		make_onb(nx, ny, nz, tx, ty, tz, bx, by, bz);
		cosine_hemisphere_sample(nx,ny,nz, tx,ty,tz, bx,by,bz, u1, u2,
								 res.wo_x, res.wo_y, res.wo_z);
		res.r = albedo_r; res.g = albedo_g; res.b = albedo_b;
		res.is_specular = false;
		res.valid = true;
		return res;
	}

	// scattering_pdf: cos(theta_wi) / pi, 0 if below surface
	CPU_GPU T scattering_pdf(T nx, T ny, T nz,
							  T wox, T woy, T woz) const
	{
		T cos_theta = wox*nx + woy*ny + woz*nz;
		return cos_theta > T(0) ? cos_theta / T(3.14159265358979323846) : T(0);
	}
};

// ===========================================================================
// 2. MetalBxDF  (specular metal with fuzz sphere)
//    Mirrors RTOW metal material
//    u1..u3 are uniform [-1,1] sphere samples for the fuzz direction
// ===========================================================================
template<typename T>
struct MetalBxDF {
	T albedo_r, albedo_g, albedo_b;
	T fuzz;

	// fuzz_x/y/z: pre-generated random unit-sphere sample
	CPU_GPU BxDFSampleResult<T> sample(
		T wix, T wiy, T wiz,    // incident direction (towards surface)
		T nx,  T ny,  T nz,     // surface normal
		T fuzz_x, T fuzz_y, T fuzz_z) const
	{
		BxDFSampleResult<T> res{};
		// reflected = wi - 2*dot(wi,n)*n  (wi points in, normal points out)
		T rx, ry, rz;
		reflect_dir(wix, wiy, wiz, nx, ny, nz, rx, ry, rz);
		normalize3(rx, ry, rz);
		res.wo_x = rx + fuzz * fuzz_x;
		res.wo_y = ry + fuzz * fuzz_y;
		res.wo_z = rz + fuzz * fuzz_z;
		// reject if below surface
		if (res.wo_x*nx + res.wo_y*ny + res.wo_z*nz <= T(0)) {
			res.valid = false;
			return res;
		}
		normalize3(res.wo_x, res.wo_y, res.wo_z);
		res.r = albedo_r; res.g = albedo_g; res.b = albedo_b;
		res.is_specular = true;
		res.valid = true;
		return res;
	}
};

// ===========================================================================
// 3. DielectricBxDF  (smooth glass -- reflect or refract via Fresnel)
//    Mirrors pbrt-v4 DielectricBxDF (smooth limit)
//    u1: uniform [0,1) for reflect/transmit decision
// ===========================================================================
template<typename T>
struct DielectricBxDF {
	T ior;  // index of refraction (material IOR; 1/ior when entering from outside)

	CPU_GPU BxDFSampleResult<T> sample(
		T wix, T wiy, T wiz,    // unit incident direction (into surface)
		T nx,  T ny,  T nz,     // outward surface normal
		bool front_face,        // true if ray enters from outside
		T u1) const             // random [0,1)
	{
		BxDFSampleResult<T> res{};
		res.r = T(1); res.g = T(1); res.b = T(1);
		res.eta = T(1);
		res.is_specular = true;
		res.is_transmission = false;
		res.valid = true;

		T ri = front_face ? (T(1) / ior) : ior;  // eta_i / eta_t

		T cos_theta = -(wix*nx + wiy*ny + wiz*nz);
#if defined(__CUDACC__)
		cos_theta = fminf(fabsf(cos_theta), T(1));
#else
		cos_theta = std::fmin(std::fabs(cos_theta), T(1));
#endif
		T sin_theta2 = T(1) - cos_theta*cos_theta;
#if defined(__CUDACC__)
		T sin_theta = sqrtf(sin_theta2 > T(0) ? sin_theta2 : T(0));
#else
		T sin_theta = std::sqrt(sin_theta2 > T(0) ? sin_theta2 : T(0));
#endif
		bool cannot_refract = ri * sin_theta > T(1);
		T fr = FrDielectric(cos_theta, T(1) / ri);  // FrDielectric(cosI, eta_t/eta_i)

		if (cannot_refract || fr > u1) {
			// Reflect
			reflect_dir(wix, wiy, wiz, nx, ny, nz, res.wo_x, res.wo_y, res.wo_z);
		} else {
			// Refract -- record eta for integrator etaScale (pbrt-v4: bs->eta)
			if (!refract_dir(wix, wiy, wiz, nx, ny, nz, ri,
							 res.wo_x, res.wo_y, res.wo_z)) {
				// TIR fallback
				reflect_dir(wix, wiy, wiz, nx, ny, nz, res.wo_x, res.wo_y, res.wo_z);
			} else {
				res.eta = ri;
				res.is_transmission = true;
			}
		}
		normalize3(res.wo_x, res.wo_y, res.wo_z);
		return res;
	}
};

// ===========================================================================

//    Mirrors pbrt-v4 ThinDielectricBxDF::Sample_f
//    R_eff = R + T^2*R/(1-R^2)  (multi-bounce geometric series)
// ===========================================================================
template<typename T>
struct ThinDielectricBxDF {
	T ior;

	CPU_GPU BxDFSampleResult<T> sample(
		T wix, T wiy, T wiz,    // unit incident direction (into surface)
		T nx,  T ny,  T nz,     // outward surface normal
		T u1) const             // random [0,1)
	{
		BxDFSampleResult<T> res{};
		res.r = T(1); res.g = T(1); res.b = T(1);
		res.is_specular = true;
		res.valid = true;

#if defined(__CUDACC__)
		T cos_theta = fabsf(wix*nx + wiy*ny + wiz*nz);
#else
		T cos_theta = std::fabs(wix*nx + wiy*ny + wiz*nz);
#endif
		T R = FrDielectric(cos_theta, ior);
		if (R < T(1)) {
			T Tv = T(1) - R;
			R += Tv * Tv * R / (T(1) - R * R);
		}

		if (u1 < R) {
			// Specular reflection
			reflect_dir(wix, wiy, wiz, nx, ny, nz, res.wo_x, res.wo_y, res.wo_z);
		} else {
			// Straight-through transmission (no bending -- zero thickness)
			res.wo_x = wix; res.wo_y = wiy; res.wo_z = wiz;
		}
		normalize3(res.wo_x, res.wo_y, res.wo_z);
		return res;
	}
};

// ===========================================================================
// 5. RoughMetalBxDF  (GGX microfacet + constant-color Fresnel)
//    Mirrors RTOW rough_metal: VNDF sampling, weight = G/G1, attenuation = albedo*weight
//    u1, u2 in [0,1) for VNDF microfacet sample.
//    Works in local shading frame (z = surface normal).
// ===========================================================================
template<typename T>
struct RoughMetalBxDF {
	T albedo_r, albedo_g, albedo_b;
	T alpha_x;  // GGX alpha u-direction (caller applies TrowbridgeReitz::RoughnessToAlpha)
	T alpha_y;  // GGX alpha v-direction (set equal for isotropic)

	CPU_GPU BxDFSampleResult<T> sample_local(
		T wi_x, T wi_y, T wi_z,
		T u1, T u2) const
	{
		BxDFSampleResult<T> res{};
		if (wi_z <= T(0)) { res.valid = false; return res; }

		TrowbridgeReitz<T> dist(alpha_x, alpha_y);
		T wm_x, wm_y, wm_z;
		dist.Sample_wm(wi_x, wi_y, wi_z, u1, u2, wm_x, wm_y, wm_z);

		T dot_wi_wm = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
		T wo_x = T(2)*dot_wi_wm*wm_x - wi_x;
		T wo_y = T(2)*dot_wi_wm*wm_y - wi_y;
		T wo_z = T(2)*dot_wi_wm*wm_z - wi_z;
		if (wo_z <= T(0)) { res.valid = false; return res; }

		T G1 = dist.G1(wi_x, wi_y, wi_z);
		T G  = dist.G(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
		T w  = (G1 > T(1e-8)) ? G / G1 : T(0);

		res.wo_x = wo_x; res.wo_y = wo_y; res.wo_z = wo_z;
		res.r = albedo_r * w; res.g = albedo_g * w; res.b = albedo_b * w;
		res.is_specular = true;
		res.valid = true;
		return res;
	}
};

// ===========================================================================
// 6. ConductorBxDF  (GGX microfacet + complex Fresnel per RGB channel)
//    Mirrors pbrt-v4 ConductorBxDF
// ===========================================================================
template<typename T>
struct ConductorBxDF {
	T eta_r, eta_g, eta_b;  // real IOR per channel
	T k_r,   k_g,   k_b;   // extinction coefficient per channel
	T alpha_x;               // GGX alpha (u-direction)
	T alpha_y;               // GGX alpha (v-direction, set equal for isotropic)

	CPU_GPU BxDFSampleResult<T> sample_local(
		T wi_x, T wi_y, T wi_z,
		T u1, T u2) const
	{
		BxDFSampleResult<T> res{};
		if (wi_z <= T(0)) { res.valid = false; return res; }

		TrowbridgeReitz<T> dist(alpha_x, alpha_y);
		T wm_x, wm_y, wm_z;
		dist.Sample_wm(wi_x, wi_y, wi_z, u1, u2, wm_x, wm_y, wm_z);

		T dot_wi_wm = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
		T wo_x = T(2)*dot_wi_wm*wm_x - wi_x;
		T wo_y = T(2)*dot_wi_wm*wm_y - wi_y;
		T wo_z = T(2)*dot_wi_wm*wm_z - wi_z;
		if (wo_z <= T(0)) { res.valid = false; return res; }

		T G1 = dist.G1(wi_x, wi_y, wi_z);
		T G  = dist.G(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
		T w  = (G1 > T(1e-8)) ? G / G1 : T(0);

		res.wo_x = wo_x; res.wo_y = wo_y; res.wo_z = wo_z;
		res.r = FrComplex(dot_wi_wm, eta_r, k_r) * w;
		res.g = FrComplex(dot_wi_wm, eta_g, k_g) * w;
		res.b = FrComplex(dot_wi_wm, eta_b, k_b) * w;
		res.is_specular = true;
		res.valid = true;
		return res;
	}
};

// ===========================================================================
// 7. RoughDielectricBxDF  (GGX microfacet glass)
//    Mirrors pbrt-v4 DielectricBxDF (rough path)
//    u1, u2: VNDF sample; u3: reflect/transmit decision
// ===========================================================================
template<typename T>
struct RoughDielectricBxDF {
	T ior;     // material IOR
	T alpha_x; // GGX alpha (u-direction)
	T alpha_y; // GGX alpha (v-direction, set equal for isotropic)

	// wi in local frame (z=normal), wi_z > 0.
	// eta = eta_i / eta_t  (entering: 1/ior, exiting: ior)
	// Returns wo in local frame; caller transforms back to world.
	CPU_GPU BxDFSampleResult<T> sample_local(
		T wi_x, T wi_y, T wi_z,
		T eta,
		T u1, T u2, T u3) const
	{
		BxDFSampleResult<T> res{};
		res.r = T(1); res.g = T(1); res.b = T(1);
		res.is_specular = true;

		TrowbridgeReitz<T> dist(alpha_x, alpha_y);
		T wm_x, wm_y, wm_z;
		dist.Sample_wm(wi_x, wi_y, wi_z, u1, u2, wm_x, wm_y, wm_z);

		T cos_i = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
		T F = FrDielectric(cos_i, T(1) / eta);  // FrDielectric(cosI, eta_t/eta_i)

		T wo_x, wo_y, wo_z;
		if (u3 < F) {
			// Reflect about microfacet normal
			wo_x = T(2)*cos_i*wm_x - wi_x;
			wo_y = T(2)*cos_i*wm_y - wi_y;
			wo_z = T(2)*cos_i*wm_z - wi_z;
			if (wo_z <= T(0)) { res.valid = false; return res; }
		} else {
			// Refract (Snell's law in local frame, pbrt-v4 formula)
			if (wm_z < T(0)) { wm_x = -wm_x; wm_y = -wm_y; wm_z = -wm_z; }
			T sin2t = eta*eta * (T(1) - cos_i*cos_i);
			if (sin2t >= T(1)) {
				// TIR: reflect instead
				wo_x = T(2)*cos_i*wm_x - wi_x;
				wo_y = T(2)*cos_i*wm_y - wi_y;
				wo_z = T(2)*cos_i*wm_z - wi_z;
				if (wo_z <= T(0)) { res.valid = false; return res; }
			} else {
#if defined(__CUDACC__)
				T cos_t = sqrtf(T(1) - sin2t);
#else
				T cos_t = std::sqrt(T(1) - sin2t);
#endif
				// pbrt-v4 Refract in local frame: wo = -eta*wi + (eta*cosI - cosT)*wm
				wo_x =  eta*(-wi_x) + (eta*cos_i - cos_t)*wm_x;
				wo_y =  eta*(-wi_y) + (eta*cos_i - cos_t)*wm_y;
				wo_z = -(eta*wi_z   - (eta*cos_i - cos_t)*wm_z);
			}
		}

		// VNDF sampling is self-normalizing for dielectrics: attenuation = white.
		// (G/G1 cancels against the VNDF pdf in the rendering equation.)
		res.wo_x = wo_x; res.wo_y = wo_y; res.wo_z = wo_z;
		res.r = T(1); res.g = T(1); res.b = T(1);
		res.valid = true;
		return res;
	}
};

// ===========================================================================
// ===========================================================================
// layered_detail -- internal helpers for LayeredBxDF random walk
//
// pbrt-v4 reference: LayeredBxDF (bxdfs.h), RNG (util/rng.h), Tr (bxdfs.h)
// ===========================================================================
namespace layered_detail {

// Use the shared canonical RNG (src/shared/rng.h, ported from pbrt-v4 util/rng.h).
using PCG32 = RNG;

// Tr(thickness, w) -- Beer-Lambert transmittance through slab of unit extinction
// sigma_t = 1 (normalised); pbrt-v4: Tr = exp(-sigma_t * dz / |w.z|)
template<typename T>
CPU_GPU T Tr(T thickness, T wz) {
#if defined(__CUDACC__)
	return expf(-thickness / fabsf(wz));
#else
	return std::exp(-thickness / std::fabs(wz));
#endif
}

// SampleExponential(u, rate) -- pbrt-v4 SampleExponential
template<typename T>
CPU_GPU T SampleExponential(T u, T rate) {
#if defined(__CUDACC__)
	return -logf(T(1) - u) / rate;
#else
	return -std::log(T(1) - u) / rate;
#endif
}

// safe_sqrt
template<typename T>
CPU_GPU T safe_sqrt(T x) {
#if defined(__CUDACC__)
	return sqrtf(x > T(0) ? x : T(0));
#else
	return std::sqrt(x > T(0) ? x : T(0));
#endif
}



// cosine-weighted hemisphere sample in local frame (z = normal) via concentric-disk mapping.
// Mirrors pbrt-v4 SampleCosineHemisphere.
template<typename T>
CPU_GPU void cosine_sample(T u1, T u2, T& ox, T& oy, T& oz) {
	T pdf_unused;
	SampleCosineHemisphere(u1, u2, ox, oy, oz, pdf_unused);
}

// HenyeyGreenstein phase sample and eval (inline, no world-frame)
// Returns new wz component after scattering in 1D (azimuth handled separately)
template<typename T>
CPU_GPU T hg_sample_cos(T g, T u) {
	if (g * g < T(1e-6)) return T(1) - T(2) * u;  // isotropic
	T gc  = g;
	T sq  = (T(1) - gc*gc) / (T(1) + gc - T(2)*gc*u);
	return -T(1) / (T(2)*gc) * (T(1) + gc*gc - sq*sq);
}

template<typename T>
CPU_GPU T hg_eval(T cos_theta, T g) {
	const T inv4pi = T(1) / (T(4) * T(3.14159265358979323846));
	T denom = T(1) + g*g + T(2)*g*cos_theta;
	return inv4pi * (T(1) - g*g) / (denom * safe_sqrt(denom));
}

} // namespace layered_detail

// ===========================================================================
// 8. CoatedDiffuseBxDF  (rough dielectric coat over Lambertian base)
//    Mirrors pbrt-v4 CoatedDiffuseBxDF = LayeredBxDF<DielectricBxDF, DiffuseBxDF, true>
//
//    Full random-walk evaluation (pbrt-v4 LayeredBxDF::Sample_f):
//      - Top interface: GGX dielectric (coat_ior)
//      - Bottom interface: Lambertian (albedo)
//      - Optional medium albedo and HG phase (set albedo_medium > 0)
//      - maxDepth bounces; terminates via Russian roulette after depth 3
//
//    sample_local(wi_x,wi_y,wi_z, seed0, seed1):
//      - Returns the sampled exit direction + throughput weight (no pdf division)
//      - seed0/seed1 are 64-bit values from the caller's RNG state
//    All in local frame (z = surface normal).
// ===========================================================================
template<typename T>
struct CoatedDiffuseBxDF {
	T albedo_r, albedo_g, albedo_b;   // Lambertian base color
	T coat_ior;                        // dielectric coat IOR (>1, e.g. 1.5)
	T alpha_x;                         // GGX roughness u-direction
	T alpha_y;                         // GGX roughness v-direction (equal = isotropic)
	T thickness = T(0.01);             // layer thickness (pbrt-v4 default 0.01)
	T g         = T(0);               // HG phase function asymmetry (0=none)
	T medium_albedo = T(0);           // scattering albedo of medium (0=vacuum)
	int maxDepth  = 10;               // max random-walk bounces
	int nSamples  = 1;                // samples per call (averaged)

	CPU_GPU BxDFSampleResult<T> sample_local(
		T wi_x, T wi_y, T wi_z,
		uint64_t seed0, uint64_t seed1 = 0) const
	{
		BxDFSampleResult<T> res{};
		if (wi_z <= T(0)) { res.valid = false; return res; }

		TrowbridgeReitz<T> dist(alpha_x, alpha_y);
		layered_detail::PCG32 rng(seed0, seed1 ^ 0xdeadbeefull);

		// pbrt-v4: Sample entrance (top) interface
		T u1 = (T)rng.Uniform<float>(), u2 = (T)rng.Uniform<float>(), uc = (T)rng.Uniform<float>();

		T wm_x, wm_y, wm_z;
		dist.Sample_wm(wi_x, wi_y, wi_z, u1, u2, wm_x, wm_y, wm_z);
		T cos_i = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
		T F_in  = FrDielectric(cos_i, coat_ior);

		if (uc < F_in) {
			// Reflect off the coat top -- specular (same as single-bounce)
			T wo_x = T(2)*cos_i*wm_x - wi_x;
			T wo_y = T(2)*cos_i*wm_y - wi_y;
			T wo_z = T(2)*cos_i*wm_z - wi_z;
			if (wo_z <= T(0)) { res.valid = false; return res; }

			T G1 = dist.G1(wi_x, wi_y, wi_z);
			T G  = dist.G(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
			T w  = (G1 > T(1e-8)) ? G / G1 : T(0);
			T fv = F_in * w;

			res.wo_x = wo_x; res.wo_y = wo_y; res.wo_z = wo_z;
			res.r = fv; res.g = fv; res.b = fv;
			res.is_specular = true;
			res.valid = true;
			return res;
		}

		// Transmitted into the layer -- start random walk toward bottom
		// w is the current direction (travels downward after transmission)
		T w_x = T(2)*cos_i*wm_x - wi_x;
		T w_y = T(2)*cos_i*wm_y - wi_y;
		T w_z = -(T(2)*cos_i*wm_z - wi_z);   // flip z: now going downward
		if (w_z == T(0)) { res.valid = false; return res; }
		// Ensure traveling downward (-z in layer)
		if (w_z > T(0)) w_z = -w_z;

		// pbrt-v4: beta starts with transmittance through top interface (1-F_in)
		T beta_r = T(1) - F_in, beta_g = T(1) - F_in, beta_b = T(1) - F_in;
		T z = thickness;   // started at top (z=thickness), going to bottom (z=0)

		for (int depth = 0; depth < maxDepth; ++depth) {
			// Russian roulette after depth 3 -- mirrors pbrt-v4: rrBeta = maxBeta < 0.25
			if (depth > 3) {
#if defined(__CUDACC__)
				T rrBeta = fmaxf(beta_r, fmaxf(beta_g, beta_b));
#else
				T rrBeta = std::max(beta_r, std::max(beta_g, beta_b));
#endif
				if (rrBeta < T(0.25)) {
					T q = std::max(T(0), T(1) - rrBeta);
					if ((T)rng.Uniform<float>() < q) { res.valid = false; return res; }
					beta_r /= T(1) - q; beta_g /= T(1) - q; beta_b /= T(1) - q;
				}
			}

			// Medium scattering or direct boundary hit
			if (medium_albedo > T(0)) {
				T dz = layered_detail::SampleExponential((T)rng.Uniform<float>(), T(1) / (w_z < T(0) ? -w_z : w_z));
				T zp = (w_z > T(0)) ? z + dz : z - dz;
				if (zp > T(0) && zp < thickness) {
					// pbrt-v4: beta *= albedo * ps->p / ps->pdf; for HG p/pdf=1
					// sample phase direction
					T cos_theta = layered_detail::hg_sample_cos(g, (T)rng.Uniform<float>());
					beta_r *= medium_albedo;
					beta_g *= medium_albedo;
					beta_b *= medium_albedo;
					// Update direction (simplified: only wz updated, wxy randomised)
					T phi_s = T(6.28318530717958647692) * (T)rng.Uniform<float>();
					T sin_theta = layered_detail::safe_sqrt(T(1) - cos_theta*cos_theta);
#if defined(__CUDACC__)
					w_x = sin_theta * cosf(phi_s);
					w_y = sin_theta * sinf(phi_s);
#else
					w_x = sin_theta * std::cos(phi_s);
					w_y = sin_theta * std::sin(phi_s);
#endif
					w_z = cos_theta;
					z   = zp;
					continue;
				}
#if defined(__CUDACC__)
				z = fmaxf(T(0), fminf(zp, thickness));
#else
				z = std::max(T(0), std::min(zp, thickness));
#endif
			} else {
				// No medium: advance directly to the other interface
				beta_r *= layered_detail::Tr(thickness, w_z);
				beta_g *= layered_detail::Tr(thickness, w_z);
				beta_b *= layered_detail::Tr(thickness, w_z);
				z = (w_z < T(0)) ? T(0) : thickness;
			}

			if (z <= T(0)) {
				// Hit bottom: Lambertian bounce
				layered_detail::cosine_sample((T)rng.Uniform<float>(), (T)rng.Uniform<float>(), w_x, w_y, w_z);
				// w_z is positive (going up toward top)
				beta_r *= albedo_r;
				beta_g *= albedo_g;
				beta_b *= albedo_b;
				z = T(0);
				// Next iteration will hit top interface
			} else {
				// Hit top interface from inside: try to exit
				T wm2_x, wm2_y, wm2_z;
				dist.Sample_wm(w_x, w_y, w_z, (T)rng.Uniform<float>(), (T)rng.Uniform<float>(), wm2_x, wm2_y, wm2_z);
				T cos2 = w_x*wm2_x + w_y*wm2_y + w_z*wm2_z;
				T F_out = FrDielectric(cos2, coat_ior);
				if ((T)rng.Uniform<float>() < F_out) {
					// Internal reflection: continue walk downward
					T rx = T(2)*cos2*wm2_x - w_x;
					T ry = T(2)*cos2*wm2_y - w_y;
					T rz = T(2)*cos2*wm2_z - w_z;
					w_x = rx; w_y = ry; w_z = rz;
					if (w_z > T(0)) w_z = -w_z; // keep traveling downward
					z = thickness;
					continue;
				}
				// Exit through coat top
				T out_x = T(2)*cos2*wm2_x - w_x;
				T out_y = T(2)*cos2*wm2_y - w_y;
				T out_z = T(2)*cos2*wm2_z - w_z;
				if (out_z <= T(0)) { res.valid = false; return res; }

				res.wo_x = out_x; res.wo_y = out_y; res.wo_z = out_z;
				res.r    = beta_r * (T(1) - F_out);
				res.g = beta_g * (T(1) - F_out);
				res.b    = beta_b * (T(1) - F_out);
				res.is_specular = false;
				res.valid = true;
				return res;
			}
		}
		res.valid = false;
		return res;
	}

	// Backward-compat 5-float overload: derives a seed from the 5 input values
	CPU_GPU BxDFSampleResult<T> sample_local(
		T wi_x, T wi_y, T wi_z,
		T u1, T u2, T u3, T u4, T u5) const
	{
		// Combine random inputs into a seed pair for the random walk
		uint32_t s0, s1, s2, s3, s4;
		s0 = (uint32_t)(u1 * 16777216.0f);
		s1 = (uint32_t)(u2 * 16777216.0f);
		s2 = (uint32_t)(u3 * 16777216.0f);
		s3 = (uint32_t)(u4 * 16777216.0f);
		s4 = (uint32_t)(u5 * 16777216.0f);
		uint64_t seed0 = ((uint64_t)s0 << 32) | (uint64_t)s1;
		uint64_t seed1 = ((uint64_t)s2 << 32) | ((uint64_t)s3 * 0x9e3779b97f4a7c15ULL + s4);
		return sample_local(wi_x, wi_y, wi_z, seed0, seed1);
	}
};

// ===========================================================================
// 9. CoatedConductorBxDF  (rough dielectric coat over GGX conductor)
//    Mirrors pbrt-v4 CoatedConductorBxDF = LayeredBxDF<DielectricBxDF, ConductorBxDF, true>
//
//    Full random-walk evaluation (pbrt-v4 LayeredBxDF::Sample_f):
//      - Top interface: GGX dielectric (coat_ior)
//      - Bottom interface: GGX conductor (complex Fresnel per RGB channel)
//      - Optional medium scattering (medium_albedo + HG phase g)
//      - maxDepth bounces; Russian roulette after depth 3
//
//    sample_local(wi_x,wi_y,wi_z, seed0, seed1):
//      - Returns sampled exit direction + throughput (no pdf division)
//    All in local frame (z = surface normal).
// ===========================================================================
template<typename T>
struct CoatedConductorBxDF {
	T eta_r, eta_g, eta_b;   // conductor real IOR
	T k_r,   k_g,   k_b;    // conductor extinction
	T coat_ior;              // dielectric coat IOR
	T alpha_x;               // GGX roughness u-direction
	T alpha_y;               // GGX roughness v-direction (equal = isotropic)
	T thickness = T(0.01);
	T g         = T(0);
	T medium_albedo = T(0);
	int maxDepth  = 10;
	int nSamples  = 1;

	CPU_GPU BxDFSampleResult<T> sample_local(
		T wi_x, T wi_y, T wi_z,
		uint64_t seed0, uint64_t seed1 = 0) const
	{
		BxDFSampleResult<T> res{};
		if (wi_z <= T(0)) { res.valid = false; return res; }

		TrowbridgeReitz<T> dist(alpha_x, alpha_y);
		layered_detail::PCG32 rng(seed0, seed1 ^ 0xdeadbeefull);

		T u1 = (T)rng.Uniform<float>(), u2 = (T)rng.Uniform<float>(), uc = (T)rng.Uniform<float>();

		T wm_x, wm_y, wm_z;
		dist.Sample_wm(wi_x, wi_y, wi_z, u1, u2, wm_x, wm_y, wm_z);
		T cos_i = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
		T F_in  = FrDielectric(cos_i, coat_ior);

		if (uc < F_in) {
			// Reflect off the coat top (achromatic)
			T wo_x = T(2)*cos_i*wm_x - wi_x;
			T wo_y = T(2)*cos_i*wm_y - wi_y;
			T wo_z = T(2)*cos_i*wm_z - wi_z;
			if (wo_z <= T(0)) { res.valid = false; return res; }

			T G1 = dist.G1(wi_x, wi_y, wi_z);
			T G  = dist.G(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
			T w  = (G1 > T(1e-8)) ? G / G1 : T(0);
			T fv = F_in * w;

			res.wo_x = wo_x; res.wo_y = wo_y; res.wo_z = wo_z;
			res.r = fv; res.g = fv; res.b = fv;
			res.is_specular = true;
			res.valid = true;
			return res;
		}

		// Transmitted into the layer: random walk
		T w_x = T(2)*cos_i*wm_x - wi_x;
		T w_y = T(2)*cos_i*wm_y - wi_y;
		T w_z = -(T(2)*cos_i*wm_z - wi_z);
		if (w_z == T(0)) { res.valid = false; return res; }
		if (w_z > T(0)) w_z = -w_z;

		// pbrt-v4: beta starts with transmittance through top interface (1-F_in)
		T beta_r = T(1) - F_in;
		T beta_g = T(1) - F_in;
		T beta_b = T(1) - F_in;
		T z = thickness;

		for (int depth = 0; depth < maxDepth; ++depth) {
			// Russian roulette after depth 3 -- mirrors pbrt-v4: rrBeta = maxBeta < 0.25
			if (depth > 3) {
#if defined(__CUDACC__)
				T rrBeta = fmaxf(beta_r, fmaxf(beta_g, beta_b));
#else
				T rrBeta = std::max(beta_r, std::max(beta_g, beta_b));
#endif
				if (rrBeta < T(0.25)) {
					T q = std::max(T(0), T(1) - rrBeta);
					if ((T)rng.Uniform<float>() < q) { res.valid = false; return res; }
					beta_r /= T(1) - q; beta_g /= T(1) - q; beta_b /= T(1) - q;
				}
			}

			if (medium_albedo > T(0)) {
				T dz = layered_detail::SampleExponential((T)rng.Uniform<float>(), T(1) / (w_z < T(0) ? -w_z : w_z));
				T zp = (w_z > T(0)) ? z + dz : z - dz;
				if (zp > T(0) && zp < thickness) {
					// pbrt-v4: beta *= albedo * ps->p / ps->pdf; for HG p/pdf=1
					T cos_theta = layered_detail::hg_sample_cos(g, (T)rng.Uniform<float>());
					beta_r *= medium_albedo; beta_g *= medium_albedo; beta_b *= medium_albedo;
					T phi_s = T(6.28318530717958647692) * (T)rng.Uniform<float>();
					T sin_theta = layered_detail::safe_sqrt(T(1) - cos_theta*cos_theta);
#if defined(__CUDACC__)
					w_x = sin_theta * cosf(phi_s);
					w_y = sin_theta * sinf(phi_s);
#else
					w_x = sin_theta * std::cos(phi_s);
					w_y = sin_theta * std::sin(phi_s);
#endif
					w_z = cos_theta;
					z   = zp;
					continue;
				}
#if defined(__CUDACC__)
				z = fmaxf(T(0), fminf(zp, thickness));
#else
				z = std::max(T(0), std::min(zp, thickness));
#endif
			} else {
				beta_r *= layered_detail::Tr(thickness, w_z);
				beta_g *= layered_detail::Tr(thickness, w_z);
				beta_b *= layered_detail::Tr(thickness, w_z);
				z = (w_z < T(0)) ? T(0) : thickness;
			}

			if (z <= T(0)) {
				// Hit conductor bottom: GGX VNDF + complex Fresnel
				// Flip to local frame of bottom interface (z points upward into layer)
				T fw_x = -w_x, fw_y = -w_y, fw_z = -w_z;  // now fw_z > 0 (incoming from top)
				T bwm_x, bwm_y, bwm_z;
				dist.Sample_wm(fw_x, fw_y, fw_z, (T)rng.Uniform<float>(), (T)rng.Uniform<float>(), bwm_x, bwm_y, bwm_z);
				T cos_c = fw_x*bwm_x + fw_y*bwm_y + fw_z*bwm_z;
				T rwo_x = T(2)*cos_c*bwm_x - fw_x;
				T rwo_y = T(2)*cos_c*bwm_y - fw_y;
				T rwo_z = T(2)*cos_c*bwm_z - fw_z;

				T G1_c = dist.G1(fw_x, fw_y, fw_z);
				T G_c  = dist.G(rwo_x, rwo_y, rwo_z, fw_x, fw_y, fw_z);
				T wt_c = (G1_c > T(1e-8)) ? G_c / G1_c : T(0);

				beta_r *= FrComplex(cos_c, eta_r, k_r) * wt_c;
				beta_g *= FrComplex(cos_c, eta_g, k_g) * wt_c;
				beta_b *= FrComplex(cos_c, eta_b, k_b) * wt_c;

				// Reflected direction goes upward (+z in layer frame -> -bottom frame -> +layer)
				w_x = rwo_x; w_y = rwo_y;
				w_z = (rwo_z < T(0)) ? -rwo_z : rwo_z; // going upward toward top
				z = T(0);
			} else {
				// Hit top interface from inside: attempt exit
				T wm2_x, wm2_y, wm2_z;
				dist.Sample_wm(w_x, w_y, w_z, (T)rng.Uniform<float>(), (T)rng.Uniform<float>(), wm2_x, wm2_y, wm2_z);
				T cos2  = w_x*wm2_x + w_y*wm2_y + w_z*wm2_z;
				T F_out = FrDielectric(cos2, coat_ior);
				if ((T)rng.Uniform<float>() < F_out) {
					// Internal reflection
					T rx = T(2)*cos2*wm2_x - w_x;
					T ry = T(2)*cos2*wm2_y - w_y;
					T rz = T(2)*cos2*wm2_z - w_z;
					w_x = rx; w_y = ry; w_z = rz;
					if (w_z > T(0)) w_z = -w_z;
					z = thickness;
					continue;
				}
				// Exit through coat
				T out_x = T(2)*cos2*wm2_x - w_x;
				T out_y = T(2)*cos2*wm2_y - w_y;
				T out_z = T(2)*cos2*wm2_z - w_z;
				if (out_z <= T(0)) { res.valid = false; return res; }

				res.wo_x = out_x; res.wo_y = out_y; res.wo_z = out_z;
				res.r    = beta_r * (T(1) - F_out);
				res.g = beta_g * (T(1) - F_out);
				res.b    = beta_b * (T(1) - F_out);
				res.is_specular = false;
				res.valid = true;
				return res;
			}
		}
		res.valid = false;
		return res;
	}

	// Backward-compat 5-float overload
	CPU_GPU BxDFSampleResult<T> sample_local(
		T wi_x, T wi_y, T wi_z,
		T u1, T u2, T u3, T u4, T u5) const
	{
		uint32_t s0 = (uint32_t)(u1 * 16777216.0f);
		uint32_t s1 = (uint32_t)(u2 * 16777216.0f);
		uint32_t s2 = (uint32_t)(u3 * 16777216.0f);
		uint32_t s3 = (uint32_t)(u4 * 16777216.0f);
		uint32_t s4 = (uint32_t)(u5 * 16777216.0f);
		uint64_t seed0 = ((uint64_t)s0 << 32) | (uint64_t)s1;
		uint64_t seed1 = ((uint64_t)s2 << 32) | ((uint64_t)s3 * 0x9e3779b97f4a7c15ULL + s4);
		return sample_local(wi_x, wi_y, wi_z, seed0, seed1);
	}
};

// ===========================================================================
// 10. DiffuseTransmissionBxDF  (cosine-weighted diffuse reflect + transmit)
//     Mirrors pbrt-v4 DiffuseTransmissionBxDF
//
//     Stochastically chooses:
//       reflect   (u1 < pr/(pr+pt)) -> cosine-weighted wo in upper hemisphere
//       transmit  (u1 >= pr/(pr+pt)) -> cosine-weighted wo in lower hemisphere
//
//     u1: reflect/transmit decision
//     u2, u3: cosine-hemisphere sample
//     pr, pt: max-component of R and T (caller provides)
//     All in world space; caller provides the surface normal.
// ===========================================================================
template<typename T>
struct DiffuseTransmissionBxDF {
	T R_r, R_g, R_b;  // reflectance color
	T T_r, T_g, T_b;  // transmittance color

	// nx,ny,nz: outward surface normal (world space)
	// u1: selection; u2,u3: hemisphere sample
	CPU_GPU BxDFSampleResult<T> sample(
		T nx, T ny, T nz,
		T u1, T u2, T u3) const
	{
		BxDFSampleResult<T> res{};
#if defined(__CUDACC__)
		T pr = fmaxf(R_r, fmaxf(R_g, R_b));
		T pt = fmaxf(T_r, fmaxf(T_g, T_b));
#else
		T pr = std::fmax(R_r, std::fmax(R_g, R_b));
		T pt = std::fmax(T_r, std::fmax(T_g, T_b));
#endif
		if (pr + pt <= T(0)) { res.valid = false; return res; }

		T tx, ty, tz, bx, by, bz;

		bool reflect = (u1 < pr / (pr + pt));
		if (reflect) {
			make_onb(nx, ny, nz, tx, ty, tz, bx, by, bz);
			cosine_hemisphere_sample(nx,ny,nz, tx,ty,tz, bx,by,bz, u2, u3,
									 res.wo_x, res.wo_y, res.wo_z);
			res.r = R_r; res.g = R_g; res.b = R_b;
		} else {
			// Transmit: cosine-weighted in -normal hemisphere
			make_onb(-nx, -ny, -nz, tx, ty, tz, bx, by, bz);
			cosine_hemisphere_sample(-nx,-ny,-nz, tx,ty,tz, bx,by,bz, u2, u3,
									 res.wo_x, res.wo_y, res.wo_z);
			res.r = T_r; res.g = T_g; res.b = T_b;
		}

		res.is_specular = false;
		res.valid = true;
		return res;
	}

	CPU_GPU T scattering_pdf(T nx, T ny, T nz,
							  T wox, T woy, T woz) const
	{
#if defined(__CUDACC__)
		T pr = fmaxf(R_r, fmaxf(R_g, R_b));
		T pt = fmaxf(T_r, fmaxf(T_g, T_b));
#else
		T pr = std::fmax(R_r, std::fmax(R_g, R_b));
		T pt = std::fmax(T_r, std::fmax(T_g, T_b));
#endif
		if (pr + pt <= T(0)) return T(0);
		T cos_theta = wox*nx + woy*ny + woz*nz;
		T inv_pi = T(1) / T(3.14159265358979323846);
		if (cos_theta > T(0))
			return (pr / (pr + pt)) * (cos_theta * inv_pi);   // reflection
		else
			return (pt / (pr + pt)) * (-cos_theta * inv_pi);  // transmission
	}
};

// ===========================================================================
// 11. NormalizedFresnelBxDF  (Fresnel-weighted diffuse reflection)
//     Mirrors pbrt-v4 NormalizedFresnelBxDF
//
//     f(wi) = (1 - FrDielectric(cos_wi, eta)) / (c * pi)
//     c     = 1 - 2 * FresnelMoment1(1/eta)
//     PDF   = cos_wi / pi  (cosine-weighted sampling)
//     Weight (sample / pdf) = (1 - Fr(cos_wi)) / c
//
//     u1, u2: cosine-hemisphere sample
// ===========================================================================
template<typename T>
struct NormalizedFresnelBxDF {
	T eta;  // surface IOR
	T c;    // 1 - 2 * FresnelMoment1(1/eta)  -- precomputed by caller

	CPU_GPU BxDFSampleResult<T> sample(
		T nx, T ny, T nz,
		T u1, T u2) const
	{
		BxDFSampleResult<T> res{};
		T tx, ty, tz, bx, by, bz;
		make_onb(nx, ny, nz, tx, ty, tz, bx, by, bz);
		cosine_hemisphere_sample(nx,ny,nz, tx,ty,tz, bx,by,bz, u1, u2,
								 res.wo_x, res.wo_y, res.wo_z);
		// attenuation = white; weight encoded in scattering_pdf via MIS path
		res.r = T(1); res.g = T(1); res.b = T(1);
		res.is_specular = false;
		res.valid = true;
		return res;
	}

	// scattering_pdf = BSDF * cos = (1 - Fr) * cos / (c * pi)
	CPU_GPU T scattering_pdf(T nx, T ny, T nz,
							  T wox, T woy, T woz) const
	{
		T cos_wi = wox*nx + woy*ny + woz*nz;
		if (cos_wi <= T(0)) return T(0);
		T fr = FrDielectric(cos_wi, eta);
		T cv = (c > T(1e-6)) ? c : T(1e-6);
		return (T(1) - fr) * cos_wi / (cv * T(3.14159265358979323846));
	}
};

// ===========================================================================
// 12. PrincipledBxDF  (Disney/pbrt-v4 CoatedDiffuse / CoatedConductor style)
//
// A three-lobe artist-friendly BSDF:
//   Lobe 0 (diffuse):   Lambertian, weight = base_color * (1-metallic) * (1-F_spec)
//   Lobe 1 (specular):  GGX microfacet with Schlick Fresnel (dielectric blend)
//                       or conductor Fresnel (metallic=1)
//   Lobe 2 (clearcoat): GGX with fixed IOR=1.5 and separate low roughness
//
// pbrt-v4 alignment:
//   - CoatedDiffuseBxDF  = dielectric top + diffuse bottom (metallic=0)
//   - CoatedConductorBxDF = dielectric top + conductor bottom (metallic=1)
//   Here metallic continuously blends the two.
//
// Parameters:
//   base_r/g/b       -- base (diffuse) color
//   metallic         -- 0=dielectric/plastic, 1=pure metal
//   roughness        -- perceptual roughness [0,1], mapped to GGX alpha = sqrt(r)
//   ior              -- interface IOR for dielectric Fresnel (default 1.5)
//   clearcoat        -- clearcoat weight [0,1] (0 = off)
//   clearcoat_rough  -- clearcoat roughness (default 0.1 = glossy)
//
// Caller provides 4 uniform randoms: u1 (lobe select), u2,u3 (direction), u4 (unused)
// ===========================================================================
template<typename T>
struct PrincipledBxDF {
	T base_r, base_g, base_b;
	T metallic;
	T roughness;
	T ior;
	T clearcoat;
	T clearcoat_rough;

	// Schlick Fresnel approximation: F0 + (1-F0)*(1-cos)^5
	CPU_GPU T schlick_fresnel(T cos_theta, T F0) const {
		T c = T(1) - cos_theta;
		T c2 = c * c;
		T c5 = c2 * c2 * c;
		return F0 + (T(1) - F0) * c5;
	}

	// Specular F0 from IOR: ((ior-1)/(ior+1))^2
	CPU_GPU T spec_F0() const {
		T f = (ior - T(1)) / (ior + T(1));
		return f * f;
	}

	// GGX specular BRDF value in local shading frame (z=normal)
	CPU_GPU T ggx_brdf(T wox, T woy, T woz,
					   T wix, T wiy, T wiz,
					   T alpha) const {
		if (woz <= T(0) || wiz <= T(0)) return T(0);
		TrowbridgeReitz<T> mf(alpha, alpha);
		// half-vector (local frame)
		T hmx = wox + wix, hmy = woy + wiy, hmz = woz + wiz;
#if defined(__CUDACC__)
		T hlen = sqrtf(hmx*hmx + hmy*hmy + hmz*hmz);
#else
		T hlen = std::sqrt(hmx*hmx + hmy*hmy + hmz*hmz);
#endif
		if (hlen < T(1e-8)) return T(0);
		hmx /= hlen; hmy /= hlen; hmz /= hlen;
		T D = mf.D(hmx, hmy, hmz);
		T G = mf.G(wox, woy, woz, wix, wiy, wiz);
		return D * G / (T(4) * woz * wiz);
	}

	// GGX specular PDF in local frame (visible-normal)
	CPU_GPU T ggx_pdf(T wox, T woy, T woz,
					  T wix, T wiy, T wiz,
					  T alpha) const {
		if (woz <= T(0) || wiz <= T(0)) return T(0);
		TrowbridgeReitz<T> mf(alpha, alpha);
		T hmx = wox + wix, hmy = woy + wiy, hmz = woz + wiz;
#if defined(__CUDACC__)
		T hlen = sqrtf(hmx*hmx + hmy*hmy + hmz*hmz);
#else
		T hlen = std::sqrt(hmx*hmx + hmy*hmy + hmz*hmz);
#endif
		if (hlen < T(1e-8)) return T(0);
		hmx /= hlen; hmy /= hlen; hmz /= hlen;
		T pdf_wm = mf.PDF(wox, woy, woz, hmx, hmy, hmz);
		T dot_wo_wm = wox*hmx + woy*hmy + woz*hmz;
		if (dot_wo_wm <= T(0)) return T(0);
		return pdf_wm / (T(4) * dot_wo_wm);
	}

	// sample() -- all directions in local shading frame (normal = +Z)
	// u1: lobe select, u2/u3: direction sample
	CPU_GPU BxDFSampleResult<T> sample(
		T nx, T ny, T nz,   // world-space normal (used to build ONB)
		T wix, T wiy, T wiz, // incident direction (world space, toward surface)
		T u1, T u2, T u3) const
	{
		BxDFSampleResult<T> res{};
		res.valid = false;

		// Build local shading frame (ONB): normal=Z, tangent=X, bitangent=Y
		T tx, ty, tz, bx, by, bz;
		make_onb(nx, ny, nz, tx, ty, tz, bx, by, bz);

		// Transform wi to local frame
		T wi_lx = wix*tx + wiy*ty + wiz*tz;
		T wi_ly = wix*bx + wiy*by + wiz*bz;
		T wi_lz = wix*nx + wiy*ny + wiz*nz;
		// wo = -wi (pointing away from surface in local frame)
		T wo_lx = -wi_lx, wo_ly = -wi_ly, wo_lz = -wi_lz;
		if (wo_lz <= T(0)) return res; // below surface

		T alpha = TrowbridgeReitz<T>::RoughnessToAlpha(roughness);
		T alpha_cc = TrowbridgeReitz<T>::RoughnessToAlpha(clearcoat_rough);

		// Fresnel at wo for lobe weights
		T cos_wo = wo_lz;
		T F0_d = spec_F0();
		T F_spec = schlick_fresnel(cos_wo, F0_d);
		// metallic blends toward conductor: F0 = base_color for metal
		T F0_metal_r = base_r, F0_metal_g = base_g, F0_metal_b = base_b;
		T F_met_r = schlick_fresnel(cos_wo, F0_metal_r);
		T F_met_g = schlick_fresnel(cos_wo, F0_metal_g);
		T F_met_b = schlick_fresnel(cos_wo, F0_metal_b);

		// Lobe weights (scalar, for selection)
		T w_diff  = (T(1) - metallic) * (T(1) - F_spec);
		T w_spec  = T(1); // always include specular
		T w_coat  = clearcoat * T(0.25); // pbrt-v4 uses 0.25 clearcoat weight
		T w_total = w_diff + w_spec + w_coat;
		if (w_total < T(1e-8)) return res;
		T inv_w = T(1) / w_total;
		T p_diff = w_diff * inv_w;
		T p_spec = w_spec * inv_w;
		// p_coat = w_coat * inv_w  (remainder)

		// Select lobe
		T wo_out_lx, wo_out_ly, wo_out_lz;
		TrowbridgeReitz<T> mf(alpha, alpha);

		if (u1 < p_diff) {
			// Diffuse: cosine-weighted hemisphere sample via concentric disk (pbrt-v4)
			T pdf_unused;
			SampleCosineHemisphere(u2, u3, wo_out_lx, wo_out_ly, wo_out_lz, pdf_unused);
		} else if (u1 < p_diff + p_spec) {
			// Specular: GGX VNDF sample
			T wm_lx, wm_ly, wm_lz;
			mf.Sample_wm(wo_lx, wo_ly, wo_lz, u2, u3, wm_lx, wm_ly, wm_lz);
			// reflect wo around wm
			T dot = wo_lx*wm_lx + wo_ly*wm_ly + wo_lz*wm_lz;
			wo_out_lx = T(2)*dot*wm_lx - wo_lx;
			wo_out_ly = T(2)*dot*wm_ly - wo_ly;
			wo_out_lz = T(2)*dot*wm_lz - wo_lz;
		} else {
			// Clearcoat: GGX VNDF sample with clearcoat roughness
			TrowbridgeReitz<T> mf_cc(alpha_cc, alpha_cc);
			T wm_lx, wm_ly, wm_lz;
			mf_cc.Sample_wm(wo_lx, wo_ly, wo_lz, u2, u3, wm_lx, wm_ly, wm_lz);
			T dot = wo_lx*wm_lx + wo_ly*wm_ly + wo_lz*wm_lz;
			wo_out_lx = T(2)*dot*wm_lx - wo_lx;
			wo_out_ly = T(2)*dot*wm_ly - wo_ly;
			wo_out_lz = T(2)*dot*wm_lz - wo_lz;
		}

		if (wo_out_lz <= T(0)) return res;

		// Compute multi-lobe BSDF value and PDF
		T cos_wi_l = wo_out_lz; // sampled direction cos with normal

		// Half-vector in local frame (pbrt-v4: Fresnel evaluated at AbsDot(wo, wm))
		T hx = wo_lx + wo_out_lx, hy = wo_ly + wo_out_ly, hz = wo_lz + wo_out_lz;
#if defined(__CUDACC__)
		T hlen = sqrtf(hx*hx + hy*hy + hz*hz);
#else
		T hlen = std::sqrt(hx*hx + hy*hy + hz*hz);
#endif
		T cos_wm = (hlen > T(1e-8)) ? (wo_lx*(hx/hlen) + wo_ly*(hy/hlen) + wo_lz*(hz/hlen)) : wo_lz;

		// Diffuse: Lambertian * (1-metallic) * (1-F_diffuse); F at wiÃƒâ€šÃ‚Â·n per Disney
		T F_wi_diff = schlick_fresnel(cos_wi_l, F0_d);
		const T inv_pi = T(1) / T(3.14159265358979323846);
		T diff_r = base_r * inv_pi * (T(1) - metallic) * (T(1) - F_wi_diff);
		T diff_g = base_g * inv_pi * (T(1) - metallic) * (T(1) - F_wi_diff);
		T diff_b = base_b * inv_pi * (T(1) - metallic) * (T(1) - F_wi_diff);

		// Specular: GGX BRDF with Fresnel at half-vector angle (pbrt-v4 ConductorBxDF alignment)
		T spec_val = ggx_brdf(wo_lx, wo_ly, wo_lz, wo_out_lx, wo_out_ly, wo_out_lz, alpha);
		T F_spec_wm = schlick_fresnel(cos_wm, F0_d);
		T F_met_wm_r = schlick_fresnel(cos_wm, F0_metal_r);
		T F_met_wm_g = schlick_fresnel(cos_wm, F0_metal_g);
		T F_met_wm_b = schlick_fresnel(cos_wm, F0_metal_b);
		T F_r = (T(1)-metallic)*F_spec_wm + metallic*F_met_wm_r;
		T F_g = (T(1)-metallic)*F_spec_wm + metallic*F_met_wm_g;
		T F_b = (T(1)-metallic)*F_spec_wm + metallic*F_met_wm_b;
		T spec_r = F_r * spec_val;
		T spec_g = F_g * spec_val;
		T spec_b = F_b * spec_val;

		// Clearcoat: GGX with IOR=1.5 (F0=0.04), Fresnel at half-vector angle
		T cc_F0 = T(0.04);
		T F_cc = schlick_fresnel(cos_wm, cc_F0);
		T cc_val = ggx_brdf(wo_lx, wo_ly, wo_lz, wo_out_lx, wo_out_ly, wo_out_lz, alpha_cc);
		T cc_r = clearcoat * T(0.25) * F_cc * cc_val;

		T total_r = diff_r + spec_r + cc_r;
		T total_g = diff_g + spec_g + cc_r; // cc is achromatic
		T total_b = diff_b + spec_b + cc_r;

		// Multi-lobe PDF (balance heuristic)
		T pdf_diff = p_diff * cos_wi_l * inv_pi;
		T pdf_spec = p_spec * ggx_pdf(wo_lx, wo_ly, wo_lz, wo_out_lx, wo_out_ly, wo_out_lz, alpha);
		T p_coat_w = (w_coat * inv_w);
		T pdf_coat = p_coat_w * ggx_pdf(wo_lx, wo_ly, wo_lz, wo_out_lx, wo_out_ly, wo_out_lz, alpha_cc);
		T pdf = pdf_diff + pdf_spec + pdf_coat;
		if (pdf < T(1e-12)) return res;

		// Weight = BSDF * cos / pdf
		T weight_r = total_r * cos_wi_l / pdf;
		T weight_g = total_g * cos_wi_l / pdf;
		T weight_b = total_b * cos_wi_l / pdf;

		// Transform sampled wo back to world space
		res.wo_x = wo_out_lx*tx + wo_out_ly*bx + wo_out_lz*nx;
		res.wo_y = wo_out_lx*ty + wo_out_ly*by + wo_out_lz*ny;
		res.wo_z = wo_out_lx*tz + wo_out_ly*bz + wo_out_lz*nz;

		res.r = weight_r;
		res.g = weight_g;
		res.b = weight_b;
		res.is_specular = false;
		res.is_transmission = false;
		res.eta = T(1);
		res.valid = true;
		return res;
	}

	// scattering_pdf in world space: multi-lobe balance heuristic
	CPU_GPU T scattering_pdf(
		T nx, T ny, T nz,
		T wix, T wiy, T wiz,  // incident (toward surface)
		T wox, T woy, T woz)  // scattered (away from surface)
	const {
		// Build local frame
		T tx, ty, tz, bx, by, bz;
		make_onb(nx, ny, nz, tx, ty, tz, bx, by, bz);

		// Transform to local
		T wo_lx = -wix*tx - wiy*ty - wiz*tz; // outgoing = -incident in BSDF convention
		T wo_ly = -wix*bx - wiy*by - wiz*bz;
		T wo_lz = -wix*nx - wiy*ny - wiz*nz;
		T wi_lx = wox*tx + woy*ty + woz*tz;
		T wi_ly = wox*bx + woy*by + woz*bz;
		T wi_lz = wox*nx + woy*ny + woz*nz;

		if (wo_lz <= T(0) || wi_lz <= T(0)) return T(0);

		T alpha    = TrowbridgeReitz<T>::RoughnessToAlpha(roughness);
		T alpha_cc = TrowbridgeReitz<T>::RoughnessToAlpha(clearcoat_rough);

		T F0_d = spec_F0();
		T F_spec = schlick_fresnel(wo_lz, F0_d);
		T w_diff  = (T(1) - metallic) * (T(1) - F_spec);
		T w_spec  = T(1);
		T w_coat  = clearcoat * T(0.25);
		T w_total = w_diff + w_spec + w_coat;
		if (w_total < T(1e-8)) return T(0);
		T inv_w = T(1) / w_total;

		const T inv_pi = T(1) / T(3.14159265358979323846);
		T p_diff = w_diff * inv_w;
		T p_spec = w_spec * inv_w;
		T p_coat = w_coat * inv_w;

		T pdf_diff = p_diff * wi_lz * inv_pi;
		T pdf_spec = p_spec * ggx_pdf(wo_lx, wo_ly, wo_lz, wi_lx, wi_ly, wi_lz, alpha);
		T pdf_coat = p_coat * ggx_pdf(wo_lx, wo_ly, wo_lz, wi_lx, wi_ly, wi_lz, alpha_cc);
		return pdf_diff + pdf_spec + pdf_coat;
	}
};

// ===========================================================================
// HairBxDF helpers (mirrors pbrt-v4 HairBxDF internals)
// ===========================================================================

// Modified Bessel function I0 and its log, used in the longitudinal Mp term.
template<typename T>
CPU_GPU T hair_I0(T x) {
	T val = T(0);
	T x2i = T(1);
	int64_t ifact = 1;
	int i4 = 1;
	for (int i = 0; i < 10; ++i) {
		if (i > 1) ifact *= i;
		val += x2i / T(i4 * ifact * ifact);
		x2i *= x * x;
		i4 *= 4;
	}
	return val;
}

template<typename T>
CPU_GPU T hair_LogI0(T x) {
#if defined(__CUDACC__)
	if (x > T(12))
		return x + T(0.5) * (-logf(T(2) * T(3.14159265358979323846f)) + logf(T(1)/x) + T(1)/(T(8)*x));
	return logf(hair_I0(x));
#else
	if (x > T(12))
		return x + T(0.5) * (-std::log(T(2) * T(3.14159265358979323846)) + std::log(T(1)/x) + T(1)/(T(8)*x));
	return std::log(hair_I0(x));
#endif
}

// Longitudinal scattering function Mp (Marschner 2003, Eq. 7)
template<typename T>
CPU_GPU T hair_Mp(T cosTheta_i, T cosTheta_o, T sinTheta_i, T sinTheta_o, T v) {
	T a = cosTheta_i * cosTheta_o / v;
	T b = sinTheta_i * sinTheta_o / v;
#if defined(__CUDACC__)
	T mp = (v <= T(0.1))
		? expf(hair_LogI0(a) - b - T(1)/v + T(0.6931f) + logf(T(1)/(T(2)*v)))
		: (expf(-b) * hair_I0(a)) / (sinhf(T(1)/v) * T(2) * v);
#else
	T mp = (v <= T(0.1))
		? std::exp(hair_LogI0(a) - b - T(1)/v + T(0.6931) + std::log(T(1)/(T(2)*v)))
		: (std::exp(-b) * hair_I0(a)) / (std::sinh(T(1)/v) * T(2) * v);
#endif
	return mp;
}

// Trimmed logistic for azimuthal Np
template<typename T>
CPU_GPU T hair_Logistic(T x, T s) {
#if defined(__CUDACC__)
	x = fabsf(x);
	return expf(-x/s) / (s * (T(1) + expf(-x/s)) * (T(1) + expf(-x/s)));
#else
	x = std::fabs(x);
	return std::exp(-x/s) / (s * (T(1)+std::exp(-x/s)) * (T(1)+std::exp(-x/s)));
#endif
}

template<typename T>
CPU_GPU T hair_LogisticCDF(T x, T s) {
#if defined(__CUDACC__)
	return T(1) / (T(1) + expf(-x/s));
#else
	return T(1) / (T(1) + std::exp(-x/s));
#endif
}

template<typename T>
CPU_GPU T hair_TrimmedLogistic(T x, T s, T a, T b) {
	return hair_Logistic(x, s) / (hair_LogisticCDF(b, s) - hair_LogisticCDF(a, s));
}

// Sample trimmed logistic
template<typename T>
CPU_GPU T hair_SampleTrimmedLogistic(T u, T s, T a, T b) {
	T k = hair_LogisticCDF(b, s) - hair_LogisticCDF(a, s);
#if defined(__CUDACC__)
	T x = -s * logf(T(1)/(u*k + hair_LogisticCDF(a,s)) - T(1));
	// clamp
	if (x < a) x = a; if (x > b) x = b;
#else
	T x = -s * std::log(T(1)/(u*k + hair_LogisticCDF(a,s)) - T(1));
	if (x < a) x = a; if (x > b) x = b;
#endif
	return x;
}

// Azimuthal scattering function phi and Np
template<typename T>
CPU_GPU T hair_Phi(int p, T gamma_o, T gamma_t) {
	const T pi = T(3.14159265358979323846);
	return T(2)*T(p)*gamma_t - T(2)*gamma_o + T(p)*pi;
}

template<typename T>
CPU_GPU T hair_Np(T phi, int p, T s, T gamma_o, T gamma_t) {
	const T pi = T(3.14159265358979323846);
	T dphi = phi - hair_Phi(p, gamma_o, gamma_t);
	while (dphi >  pi) dphi -= T(2)*pi;
	while (dphi < -pi) dphi += T(2)*pi;
	return hair_TrimmedLogistic(dphi, s, -pi, pi);
}

// Safe sqrt and asin helpers
template<typename T>
CPU_GPU T hair_SafeSqrt(T x) {
#if defined(__CUDACC__)
	return sqrtf(x > T(0) ? x : T(0));
#else
	return std::sqrt(x > T(0) ? x : T(0));
#endif
}

template<typename T>
CPU_GPU T hair_SafeAsin(T x) {
#if defined(__CUDACC__)
	if (x < T(-1)) x = T(-1); if (x > T(1)) x = T(1);
	return asinf(x);
#else
	if (x < T(-1)) x = T(-1); if (x > T(1)) x = T(1);
	return std::asin(x);
#endif
}

// ===========================================================================
// 13. HairBxDF  (Marschner 2003 + Chiang 2016, mirrors pbrt-v4 HairBxDF)
//
// Coordinate system: hair fiber tangent = +X in local frame (pbrt-v4 convention)
//   sinTheta = wi.x (longitudinal angle component)
//   phi      = atan2(wi.z, wi.y)  (azimuthal angle)
//
// Parameters:
//   h          : fiber offset in [-1,1] (cross-section hit position)
//   eta        : fiber IOR (typically ~1.55 for human hair)
//   sigma_a_r/g/b : absorption coefficients (RGB)
//   beta_m     : longitudinal roughness [0,1]
//   beta_n     : azimuthal roughness [0,1]
//   alpha      : scale tilt angle in degrees (typically ~2)
//
// Lobes (pMax=3):  p=0 R, p=1 TT, p=2 TRT, p=3 remainder
// ===========================================================================
template<typename T>
struct HairBxDF {
	static constexpr int pMax = 3;

	T h;            // cross-section offset [-1,1]
	T eta;          // fiber IOR
	T sigma_r, sigma_g, sigma_b;  // absorption (RGB)
	T beta_m, beta_n;

	// Precomputed from constructor
	T v[pMax + 1];          // longitudinal variance per lobe
	T s;                    // azimuthal logistic scale
	T sin2kAlpha[pMax];     // scale tilt trig values
	T cos2kAlpha[pMax];

	CPU_GPU HairBxDF() {}

	CPU_GPU HairBxDF(T h_, T eta_, T sr, T sg, T sb,
					 T beta_m_, T beta_n_, T alpha_deg)
		: h(h_), eta(eta_), sigma_r(sr), sigma_g(sg), sigma_b(sb),
		  beta_m(beta_m_), beta_n(beta_n_)
	{
		// Longitudinal variances (pbrt-v4 Eq.)
		T bm = beta_m;
		T bm2 = bm*bm;
		v[0] = (T(0.726)*bm + T(0.812)*bm2 + T(3.7)*pow20(bm)) *
			   (T(0.726)*bm + T(0.812)*bm2 + T(3.7)*pow20(bm));
		v[1] = T(0.25) * v[0];
		v[2] = T(4)    * v[0];
		for (int p = 3; p <= pMax; ++p) v[p] = v[2];

		// Azimuthal logistic scale
		const T SqrtPiOver8 = T(0.626657069);
		T bn = beta_n;
		s = SqrtPiOver8 * (T(0.265)*bn + T(1.194)*bn*bn + T(5.372)*pow22(bn));

		// Scale tilt sin/cos
		const T deg2rad = T(3.14159265358979323846) / T(180);
#if defined(__CUDACC__)
		sin2kAlpha[0] = sinf(alpha_deg * deg2rad);
		cos2kAlpha[0] = hair_SafeSqrt(T(1) - sin2kAlpha[0]*sin2kAlpha[0]);
#else
		sin2kAlpha[0] = std::sin(alpha_deg * deg2rad);
		cos2kAlpha[0] = hair_SafeSqrt(T(1) - sin2kAlpha[0]*sin2kAlpha[0]);
#endif
		for (int i = 1; i < pMax; ++i) {
			sin2kAlpha[i] = T(2) * cos2kAlpha[i-1] * sin2kAlpha[i-1];
			cos2kAlpha[i] = cos2kAlpha[i-1]*cos2kAlpha[i-1] - sin2kAlpha[i-1]*sin2kAlpha[i-1];
		}
	}

	// Raise to integer powers efficiently
	CPU_GPU static T pow20(T x) {
		T x2 = x*x, x4 = x2*x2, x8 = x4*x4, x16 = x8*x8;
		return x16*x4;
	}
	CPU_GPU static T pow22(T x) {
		T x2 = x*x, x4 = x2*x2, x8 = x4*x4, x16 = x8*x8;
		return x16*x4*x2;
	}

	// Compute Ap attenuation for each lobe (returns array of pMax+1 RGB triplets)
	// Stores results into ap_r/g/b arrays of size pMax+1
	CPU_GPU void compute_Ap(T cosTheta_o,
							T ap_r[pMax+1], T ap_g[pMax+1], T ap_b[pMax+1]) const {
		T cosGamma_o = hair_SafeSqrt(T(1) - h*h);
		T cosTheta = cosTheta_o * cosGamma_o;
		// Fresnel at fiber surface
		T f = FrDielectric(cosTheta, eta);

		// Refracted angle for transmittance
		T sinTheta_o = hair_SafeSqrt(T(1) - cosTheta_o*cosTheta_o);
		T sinTheta_t = sinTheta_o / eta;
		T cosTheta_t = hair_SafeSqrt(T(1) - sinTheta_t*sinTheta_t);
		T etap = hair_SafeSqrt(eta*eta - sinTheta_o*sinTheta_o) / cosTheta_o;
		T sinGamma_t = h / etap;
		T cosGamma_t = hair_SafeSqrt(T(1) - sinGamma_t*sinGamma_t);

		T path = T(2) * cosGamma_t / (cosTheta_t > T(1e-6) ? cosTheta_t : T(1e-6));
#if defined(__CUDACC__)
		T Tr = expf(-sigma_r * path);
		T Tg = expf(-sigma_g * path);
		T Tb = expf(-sigma_b * path);
#else
		T Tr = std::exp(-sigma_r * path);
		T Tg = std::exp(-sigma_g * path);
		T Tb = std::exp(-sigma_b * path);
#endif

		// p=0: surface reflection
		ap_r[0] = f; ap_g[0] = f; ap_b[0] = f;
		// p=1: TT transmission
		T f2 = (T(1)-f)*(T(1)-f);
		ap_r[1] = f2*Tr; ap_g[1] = f2*Tg; ap_b[1] = f2*Tb;
		// p=2..pMax-1
		for (int p = 2; p < pMax; ++p) {
			ap_r[p] = ap_r[p-1]*Tr*f; ap_g[p] = ap_g[p-1]*Tg*f; ap_b[p] = ap_b[p-1]*Tb*f;
		}
		// remainder
		T denom_r = (T(1) - Tr*f) > T(1e-6) ? (T(1) - Tr*f) : T(1e-6);
		T denom_g = (T(1) - Tg*f) > T(1e-6) ? (T(1) - Tg*f) : T(1e-6);
		T denom_b = (T(1) - Tb*f) > T(1e-6) ? (T(1) - Tb*f) : T(1e-6);
		ap_r[pMax] = ap_r[pMax-1]*f*Tr/denom_r;
		ap_g[pMax] = ap_g[pMax-1]*f*Tg/denom_g;
		ap_b[pMax] = ap_b[pMax-1]*f*Tb/denom_b;
	}

	// Compute per-lobe selection probabilities (scalar average of RGB Ap)
	CPU_GPU void compute_ApPDF(T cosTheta_o, T apPDF[pMax+1]) const {
		T ap_r[pMax+1], ap_g[pMax+1], ap_b[pMax+1];
		compute_Ap(cosTheta_o, ap_r, ap_g, ap_b);
		T sumY = T(0);
		for (int i = 0; i <= pMax; ++i)
			sumY += (ap_r[i] + ap_g[i] + ap_b[i]) / T(3);
		sumY = sumY > T(1e-8) ? sumY : T(1e-8);
		for (int i = 0; i <= pMax; ++i)
			apPDF[i] = (ap_r[i]+ap_g[i]+ap_b[i]) / (T(3)*sumY);
	}

	// Evaluate BSDF: wi/wo are in hair local frame (fiber tangent = +X)
	// wi_z = cosTheta_i * sin(phi_i)  ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the z-component of wi in local frame.
	// pbrt-v4 HairBxDF::f() divides by AbsCosTheta(wi) = |wi.z|, which equals
	// cosTheta_i * |sin(phi_i)|.  We do the same here for alignment.
	CPU_GPU void eval_local(T wo_x, T wo_y, T wo_z,
							T wi_x, T wi_y, T wi_z,
							T& fr, T& fg, T& fb) const {
		const T pi = T(3.14159265358979323846);

		T sinTheta_o = wo_x;
		T cosTheta_o = hair_SafeSqrt(T(1) - sinTheta_o*sinTheta_o);
#if defined(__CUDACC__)
		T phi_o = atan2f(wo_z, wo_y);
#else
		T phi_o = std::atan2(wo_z, wo_y);
#endif
		T gamma_o = hair_SafeAsin(h);

		T sinTheta_i = wi_x;
		T cosTheta_i = hair_SafeSqrt(T(1) - sinTheta_i*sinTheta_i);
#if defined(__CUDACC__)
		T phi_i = atan2f(wi_z, wi_y);
#else
		T phi_i = std::atan2(wi_z, wi_y);
#endif

		// gamma_t for azimuthal Np (etap uses cosTheta_o guard against zero)
		T etap = hair_SafeSqrt(eta*eta - sinTheta_o*sinTheta_o) / (cosTheta_o > T(1e-6) ? cosTheta_o : T(1e-6));
		T sinGamma_t = h / etap;
		T gamma_t = hair_SafeAsin(sinGamma_t);

		T ap_r[pMax+1], ap_g[pMax+1], ap_b[pMax+1];
		compute_Ap(cosTheta_o, ap_r, ap_g, ap_b);

		T phi = phi_i - phi_o;
		fr = T(0); fg = T(0); fb = T(0);

		for (int p = 0; p < pMax; ++p) {
			T sinThetap_o, cosThetap_o;
			if (p == 0) {
				sinThetap_o = sinTheta_o*cos2kAlpha[1] - cosTheta_o*sin2kAlpha[1];
				cosThetap_o = cosTheta_o*cos2kAlpha[1] + sinTheta_o*sin2kAlpha[1];
			} else if (p == 1) {
				sinThetap_o = sinTheta_o*cos2kAlpha[0] + cosTheta_o*sin2kAlpha[0];
				cosThetap_o = cosTheta_o*cos2kAlpha[0] - sinTheta_o*sin2kAlpha[0];
			} else if (p == 2) {
				sinThetap_o = sinTheta_o*cos2kAlpha[2] + cosTheta_o*sin2kAlpha[2];
				cosThetap_o = cosTheta_o*cos2kAlpha[2] - sinTheta_o*sin2kAlpha[2];
			} else {
				sinThetap_o = sinTheta_o;
				cosThetap_o = cosTheta_o;
			}
#if defined(__CUDACC__)
			cosThetap_o = fabsf(cosThetap_o);
#else
			cosThetap_o = std::fabs(cosThetap_o);
#endif
			T mp = hair_Mp(cosTheta_i, cosThetap_o, sinTheta_i, sinThetap_o, v[p]);
			T np = hair_Np(phi, p, s, gamma_o, gamma_t);
			fr += mp * ap_r[p] * np;
			fg += mp * ap_g[p] * np;
			fb += mp * ap_b[p] * np;
		}
		// Remainder lobe
		T mp_rem = hair_Mp(cosTheta_i, cosTheta_o, sinTheta_i, sinTheta_o, v[pMax]);
		fr += mp_rem * ap_r[pMax] / (T(2)*pi);
		fg += mp_rem * ap_g[pMax] / (T(2)*pi);
		fb += mp_rem * ap_b[pMax] / (T(2)*pi);

		// Divide by AbsCosTheta(wi) = |wi_z| ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â matches pbrt-v4 HairBxDF::f()
		// wi_z = cosTheta_i * sin(phi_i); when near zero use cosTheta_i as fallback
#if defined(__CUDACC__)
		T absCosTheta_wi = fabsf(wi_z);
#else
		T absCosTheta_wi = std::fabs(wi_z);
#endif
		if (absCosTheta_wi < T(1e-6)) absCosTheta_wi = cosTheta_i > T(1e-6) ? cosTheta_i : T(1e-6);
		fr /= absCosTheta_wi;
		fg /= absCosTheta_wi;
		fb /= absCosTheta_wi;
	}

	// Build hair local frame from world-space fiber tangent
	// In hair frame: tangent = +X, two perpendiculars = Y/Z
	CPU_GPU void build_hair_onb(T tx, T ty, T tz,
								T& bx, T& by, T& bz,
								T& cx, T& cy, T& cz) const {
		// b = any vector not parallel to t
		T ax = tx > T(0.9) ? T(0) : T(1);
		T ay = T(0), az = T(0);
		if (tx > T(0.9)) { ay = T(1); }
		// b = normalize(cross(a, t))
		T qx = ay*tz - az*ty, qy = az*tx - ax*tz, qz = ax*ty - ay*tx;
#if defined(__CUDACC__)
		T qlen = sqrtf(qx*qx + qy*qy + qz*qz);
#else
		T qlen = std::sqrt(qx*qx + qy*qy + qz*qz);
#endif
		qlen = qlen > T(1e-8) ? qlen : T(1e-8);
		bx = qx/qlen; by = qy/qlen; bz = qz/qlen;
		// c = cross(t, b)
		cx = ty*bz - tz*by; cy = tz*bx - tx*bz; cz = tx*by - ty*bx;
	}

	// Transform world-space direction to hair local frame
	CPU_GPU void to_local(T tx, T ty, T tz, T bx, T by, T bz, T cx, T cy, T cz,
						  T wx, T wy, T wz,
						  T& lx, T& ly, T& lz) const {
		lx = wx*tx + wy*ty + wz*tz;  // component along fiber tangent
		ly = wx*bx + wy*by + wz*bz;
		lz = wx*cx + wy*cy + wz*cz;
	}

	// Transform hair local direction to world space
	CPU_GPU void to_world(T tx, T ty, T tz, T bx, T by, T bz, T cx, T cy, T cz,
						  T lx, T ly, T lz,
						  T& wx, T& wy, T& wz) const {
		wx = lx*tx + ly*bx + lz*cx;
		wy = lx*ty + ly*by + lz*cy;
		wz = lx*tz + ly*bz + lz*cz;
	}

	// sample(): wi in world space, tx/ty/tz = fiber tangent, u1/u2/u3/u4 = random
	CPU_GPU BxDFSampleResult<T> sample(
		T tx, T ty, T tz,      // fiber tangent (world space)
		T wix, T wiy, T wiz,   // incident ray (toward surface, world space)
		T u1, T u2, T u3, T u4) const
	{
		const T pi = T(3.14159265358979323846);
		BxDFSampleResult<T> res{};
		res.valid = false;

		T bx, by, bz, cx, cy, cz;
		build_hair_onb(tx,ty,tz, bx,by,bz, cx,cy,cz);

		// wo in hair local frame = -wi
		T wi_lx, wi_ly, wi_lz;
		to_local(tx,ty,tz, bx,by,bz, cx,cy,cz, wix,wiy,wiz, wi_lx,wi_ly,wi_lz);
		T wo_lx = -wi_lx, wo_ly = -wi_ly, wo_lz = -wi_lz;

		T sinTheta_o = wo_lx;
		T cosTheta_o = hair_SafeSqrt(T(1) - sinTheta_o*sinTheta_o);
#if defined(__CUDACC__)
		T phi_o = atan2f(wo_lz, wo_ly);
#else
		T phi_o = std::atan2(wo_lz, wo_ly);
#endif
		T gamma_o = hair_SafeAsin(h);

		// Select lobe p from ApPDF
		T apPDF[pMax+1];
		compute_ApPDF(cosTheta_o, apPDF);
		int p = 0;
		T cumul = T(0);
		for (int i = 0; i <= pMax; ++i) {
			cumul += apPDF[i];
			if (u1 < cumul || i == pMax) { p = i; break; }
		}

		// Scale tilt for selected lobe
		T sinThetap_o, cosThetap_o;
		if (p == 0) {
			sinThetap_o = sinTheta_o*cos2kAlpha[1] - cosTheta_o*sin2kAlpha[1];
			cosThetap_o = cosTheta_o*cos2kAlpha[1] + sinTheta_o*sin2kAlpha[1];
		} else if (p == 1) {
			sinThetap_o = sinTheta_o*cos2kAlpha[0] + cosTheta_o*sin2kAlpha[0];
			cosThetap_o = cosTheta_o*cos2kAlpha[0] - sinTheta_o*sin2kAlpha[0];
		} else if (p == 2) {
			sinThetap_o = sinTheta_o*cos2kAlpha[2] + cosTheta_o*sin2kAlpha[2];
			cosThetap_o = cosTheta_o*cos2kAlpha[2] - sinTheta_o*sin2kAlpha[2];
		} else {
			sinThetap_o = sinTheta_o;
			cosThetap_o = cosTheta_o;
		}
#if defined(__CUDACC__)
		cosThetap_o = fabsf(cosThetap_o);
		// Sample Mp: cosTheta_i via inverse CDF
		T vp = v[p];
		T cosTheta_i = T(1) + vp * logf(fmaxf(u2, T(1e-5)) + (T(1)-u2)*expf(-T(2)/vp));
		T sinTheta_i = hair_SafeSqrt(T(1) - cosTheta_i*cosTheta_i);
		T cosPhi     = cosf(T(2)*pi*u3);
		sinTheta_i   = -sinThetap_o*cosTheta_i + cosThetap_o*sinTheta_i*cosPhi;
		cosTheta_i   = hair_SafeSqrt(T(1) - sinTheta_i*sinTheta_i);
		// Sample Np azimuthal
		T etap = hair_SafeSqrt(eta*eta - sinTheta_o*sinTheta_o) / (cosTheta_o > T(1e-6) ? cosTheta_o : T(1e-6));
		T sinGamma_t = h / etap;
		T gamma_t = hair_SafeAsin(sinGamma_t);
		T dphi = (p < pMax)
			? hair_SampleTrimmedLogistic(u4, s, -pi, pi) + hair_Phi(p, gamma_o, gamma_t)
			: T(2)*pi*u4;
		T phi_i = phi_o + dphi;
		T wi_out_lx = sinTheta_i;
		T wi_out_ly = cosTheta_i * cosf(phi_i);
		T wi_out_lz = cosTheta_i * sinf(phi_i);
#else
		cosThetap_o = std::fabs(cosThetap_o);
		T vp = v[p];
		T cosTheta_i = T(1) + vp * std::log(std::max(u2, T(1e-5)) + (T(1)-u2)*std::exp(-T(2)/vp));
		T sinTheta_i = hair_SafeSqrt(T(1) - cosTheta_i*cosTheta_i);
		T cosPhi     = std::cos(T(2)*pi*u3);
		sinTheta_i   = -sinThetap_o*cosTheta_i + cosThetap_o*sinTheta_i*cosPhi;
		cosTheta_i   = hair_SafeSqrt(T(1) - sinTheta_i*sinTheta_i);
		// Sample Np azimuthal
		T etap = hair_SafeSqrt(eta*eta - sinTheta_o*sinTheta_o) / (cosTheta_o > T(1e-6) ? cosTheta_o : T(1e-6));
		T sinGamma_t = h / etap;
		T gamma_t = hair_SafeAsin(sinGamma_t);
		T dphi = (p < pMax)
			? hair_SampleTrimmedLogistic(u4, s, -pi, pi) + hair_Phi(p, gamma_o, gamma_t)
			: T(2)*pi*u4;
		T phi_i = phi_o + dphi;
		T wi_out_lx = sinTheta_i;
		T wi_out_ly = cosTheta_i * std::cos(phi_i);
		T wi_out_lz = cosTheta_i * std::sin(phi_i);
#endif

		// Transform sampled wi back to world space
		T wo_wx, wo_wy, wo_wz;
		to_world(tx,ty,tz, bx,by,bz, cx,cy,cz, wi_out_lx, wi_out_ly, wi_out_lz,
				 wo_wx, wo_wy, wo_wz);

		// Evaluate BSDF and PDF
		T fr, fg, fb;
		eval_local(wo_lx, wo_ly, wo_lz, wi_out_lx, wi_out_ly, wi_out_lz, fr, fg, fb);
		T pdf = scattering_pdf_local(wo_lx, wo_ly, wo_lz, wi_out_lx, wi_out_ly, wi_out_lz);

		if (pdf < T(1e-8)) return res;

		res.wo_x = wo_wx; res.wo_y = wo_wy; res.wo_z = wo_wz;
		res.r = fr / pdf;
		res.g = fg / pdf;
		res.b = fb / pdf;
		res.is_specular = false;
		res.is_transmission = false;
		res.eta = T(1);
		res.valid = true;
		return res;
	}

	// PDF in hair local frame (shared by sample and scattering_pdf)
	CPU_GPU T scattering_pdf_local(
		T wo_lx, T wo_ly, T wo_lz,
		T wi_lx, T wi_ly, T wi_lz) const
	{
		const T pi = T(3.14159265358979323846);
		T sinTheta_o = wo_lx;
		T cosTheta_o = hair_SafeSqrt(T(1) - sinTheta_o*sinTheta_o);
#if defined(__CUDACC__)
		T phi_o = atan2f(wo_lz, wo_ly);
#else
		T phi_o = std::atan2(wo_lz, wo_ly);
#endif
		T gamma_o = hair_SafeAsin(h);

		T sinTheta_i = wi_lx;
		T cosTheta_i = hair_SafeSqrt(T(1) - sinTheta_i*sinTheta_i);
#if defined(__CUDACC__)
		T phi_i = atan2f(wi_lz, wi_ly);
#else
		T phi_i = std::atan2(wi_lz, wi_ly);
#endif

		T etap = hair_SafeSqrt(eta*eta - sinTheta_o*sinTheta_o) / (cosTheta_o > T(1e-6) ? cosTheta_o : T(1e-6));
		T sinGamma_t = h / etap;
		T gamma_t = hair_SafeAsin(sinGamma_t);

		T apPDF[pMax+1];
		compute_ApPDF(cosTheta_o, apPDF);

		T phi = phi_i - phi_o;
		T pdf = T(0);
		for (int p = 0; p < pMax; ++p) {
			T sinThetap_o, cosThetap_o;
			if (p == 0) {
				sinThetap_o = sinTheta_o*cos2kAlpha[1] - cosTheta_o*sin2kAlpha[1];
				cosThetap_o = cosTheta_o*cos2kAlpha[1] + sinTheta_o*sin2kAlpha[1];
			} else if (p == 1) {
				sinThetap_o = sinTheta_o*cos2kAlpha[0] + cosTheta_o*sin2kAlpha[0];
				cosThetap_o = cosTheta_o*cos2kAlpha[0] - sinTheta_o*sin2kAlpha[0];
			} else if (p == 2) {
				sinThetap_o = sinTheta_o*cos2kAlpha[2] + cosTheta_o*sin2kAlpha[2];
				cosThetap_o = cosTheta_o*cos2kAlpha[2] - sinTheta_o*sin2kAlpha[2];
			} else {
				sinThetap_o = sinTheta_o;
				cosThetap_o = cosTheta_o;
			}
#if defined(__CUDACC__)
			cosThetap_o = fabsf(cosThetap_o);
#else
			cosThetap_o = std::fabs(cosThetap_o);
#endif
			pdf += hair_Mp(cosTheta_i, cosThetap_o, sinTheta_i, sinThetap_o, v[p])
				 * apPDF[p]
				 * hair_Np(phi, p, s, gamma_o, gamma_t);
		}
		pdf += hair_Mp(cosTheta_i, cosTheta_o, sinTheta_i, sinTheta_o, v[pMax])
			 * apPDF[pMax] * (T(1)/(T(2)*pi));
		return pdf;
	}

	// scattering_pdf(): world-space API matching the repo convention
	// tx/ty/tz = fiber tangent; wix/wiy/wiz = incident; wox/woy/woz = outgoing
	CPU_GPU T scattering_pdf(
		T tx, T ty, T tz,
		T wix, T wiy, T wiz,
		T wox, T woy, T woz) const
	{
		T bx, by, bz, cx, cy, cz;
		build_hair_onb(tx,ty,tz, bx,by,bz, cx,cy,cz);

		T wi_lx, wi_ly, wi_lz;
		to_local(tx,ty,tz, bx,by,bz, cx,cy,cz, wix,wiy,wiz, wi_lx,wi_ly,wi_lz);
		T wo_loc_x = -wi_lx, wo_loc_y = -wi_ly, wo_loc_z = -wi_lz;

		T wo_lx, wo_ly, wo_lz;
		to_local(tx,ty,tz, bx,by,bz, cx,cy,cz, wox,woy,woz, wo_lx,wo_ly,wo_lz);

		return scattering_pdf_local(wo_loc_x, wo_loc_y, wo_loc_z, wo_lx, wo_ly, wo_lz);
	}
};

