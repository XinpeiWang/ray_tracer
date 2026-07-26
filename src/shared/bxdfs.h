#pragma once
// ---------------------------------------------------------------------------
// bxdfs.h -- Shared CPU/GPU BxDF math library (mirrors pbrt-v4 bxdfs.h)
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

// Cosine-weighted hemisphere sample in world space, given a pre-built ONB.
// u1, u2 in [0,1). Returns normalised direction.
template<typename T>
CPU_GPU void cosine_hemisphere_sample(
	T nx, T ny, T nz,        // surface normal
	T tx, T ty, T tz,        // tangent
	T bx, T by, T bz,        // bitangent
	T u1, T u2,              // random [0,1)
	T& dx, T& dy, T& dz)
{
#if defined(__CUDACC__)
	T phi    = T(2) * T(3.14159265358979323846f) * u1;
	T r_sqrt = sqrtf(u2);
	T lx     = r_sqrt * cosf(phi);
	T ly     = r_sqrt * sinf(phi);
	T lz     = sqrtf(T(1) - u2);
#else
	T phi    = T(2) * T(3.14159265358979323846) * u1;
	T r_sqrt = std::sqrt(u2);
	T lx     = r_sqrt * std::cos(phi);
	T ly     = r_sqrt * std::sin(phi);
	T lz     = std::sqrt(T(1) - u2);
#endif
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
	T alpha;  // GGX alpha (caller applies TrowbridgeReitz::RoughnessToAlpha)

	CPU_GPU BxDFSampleResult<T> sample_local(
		T wi_x, T wi_y, T wi_z,
		T u1, T u2) const
	{
		BxDFSampleResult<T> res{};
		if (wi_z <= T(0)) { res.valid = false; return res; }

		TrowbridgeReitz<T> dist(alpha, alpha);
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
	T alpha;                 // GGX alpha

	CPU_GPU BxDFSampleResult<T> sample_local(
		T wi_x, T wi_y, T wi_z,
		T u1, T u2) const
	{
		BxDFSampleResult<T> res{};
		if (wi_z <= T(0)) { res.valid = false; return res; }

		TrowbridgeReitz<T> dist(alpha, alpha);
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
	T ior;    // material IOR
	T alpha;  // GGX alpha

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

		TrowbridgeReitz<T> dist(alpha, alpha);
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
// 8. CoatedDiffuseBxDF  (rough dielectric coat over Lambertian base)
//    Mirrors pbrt-v4 CoatedDiffuseBxDF = LayeredBxDF<DielectricBxDF, DiffuseBxDF>
//
//    Path A (u3 < F_in): coat GGX specular reflection
//      attenuation = F_in * G/G1  (achromatic coat)
//    Path B: transmit into layer -> Lambertian -> exit coat
//      attenuation = albedo * (1-F_in) * (1-F_out)
//
//    u1,u2: VNDF sample; u3: coat reflect/transmit; u4,u5: cosine sample (Path B)
//    All in local frame (z = surface normal).
// ===========================================================================
template<typename T>
struct CoatedDiffuseBxDF {
	T albedo_r, albedo_g, albedo_b;
	T coat_ior;
	T alpha;

	CPU_GPU BxDFSampleResult<T> sample_local(
		T wi_x, T wi_y, T wi_z,
		T u1, T u2, T u3, T u4, T u5) const
	{
		BxDFSampleResult<T> res{};
		if (wi_z <= T(0)) { res.valid = false; return res; }

		TrowbridgeReitz<T> dist(alpha, alpha);
		T wm_x, wm_y, wm_z;
		dist.Sample_wm(wi_x, wi_y, wi_z, u1, u2, wm_x, wm_y, wm_z);

		T cos_i = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
		T F_in  = FrDielectric(cos_i, coat_ior);

		if (u3 < F_in) {
			// Path A: coat specular reflection
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

		// Path B: cosine-weighted Lambertian direction (in local frame: z=normal)
		// Cosine sample in local frame
#if defined(__CUDACC__)
		T phi    = T(2) * T(3.14159265358979323846f) * u4;
		T r_sq   = sqrtf(u5);
		T lx     = r_sq * cosf(phi);
		T ly     = r_sq * sinf(phi);
		T lz     = sqrtf(T(1) - u5);
#else
		T phi    = T(2) * T(3.14159265358979323846) * u4;
		T r_sq   = std::sqrt(u5);
		T lx     = r_sq * std::cos(phi);
		T ly     = r_sq * std::sin(phi);
		T lz     = std::sqrt(T(1) - u5);
#endif
		// wo in local frame (z = normal)
		T wo_x = lx, wo_y = ly, wo_z = lz;
		if (wo_z <= T(0)) { res.valid = false; return res; }

#if defined(__CUDACC__)
		T cos_out = fabsf(wo_z);
#else
		T cos_out = std::fabs(wo_z);
#endif
		T F_out = FrDielectric(cos_out, T(1) / coat_ior);
		T T_in  = T(1) - F_in;
		T T_out = T(1) - F_out;
		T throughput = T_in * T_out;

		res.wo_x = wo_x; res.wo_y = wo_y; res.wo_z = wo_z;
		res.r = albedo_r * throughput;
		res.g = albedo_g * throughput;
		res.b = albedo_b * throughput;
		res.is_specular = false;
		res.valid = true;
		return res;
	}
};

// ===========================================================================
// 9. CoatedConductorBxDF  (rough dielectric coat over GGX conductor)
//    Mirrors pbrt-v4 CoatedConductorBxDF = LayeredBxDF<DielectricBxDF, ConductorBxDF>
//
//    Path A (u3 < F_in): coat GGX specular reflection (achromatic)
//      attenuation = F_in * G/G1
//    Path B: conductor GGX + complex Fresnel + coat exit Fresnel
//      attenuation = FrComplex * G_c/G1_c * (1-F_in) * (1-F_out)
//
//    u1,u2: coat VNDF; u3: coat reflect/transmit; u4,u5: conductor VNDF
//    All in local frame (z = surface normal).
// ===========================================================================
template<typename T>
struct CoatedConductorBxDF {
	T eta_r, eta_g, eta_b;  // conductor real IOR
	T k_r,   k_g,   k_b;   // conductor extinction
	T coat_ior;
	T alpha;

	CPU_GPU BxDFSampleResult<T> sample_local(
		T wi_x, T wi_y, T wi_z,
		T u1, T u2, T u3, T u4, T u5) const
	{
		BxDFSampleResult<T> res{};
		if (wi_z <= T(0)) { res.valid = false; return res; }

		TrowbridgeReitz<T> dist(alpha, alpha);

		// Coat top VNDF sample
		T cwm_x, cwm_y, cwm_z;
		dist.Sample_wm(wi_x, wi_y, wi_z, u1, u2, cwm_x, cwm_y, cwm_z);
		T cos_i = wi_x*cwm_x + wi_y*cwm_y + wi_z*cwm_z;
		T F_in  = FrDielectric(cos_i, coat_ior);

		if (u3 < F_in) {
			// Path A: coat specular reflection
			T wo_x = T(2)*cos_i*cwm_x - wi_x;
			T wo_y = T(2)*cos_i*cwm_y - wi_y;
			T wo_z = T(2)*cos_i*cwm_z - wi_z;
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

		// Path B: conductor GGX bounce
		T bwm_x, bwm_y, bwm_z;
		dist.Sample_wm(wi_x, wi_y, wi_z, u4, u5, bwm_x, bwm_y, bwm_z);
		T cos_c = wi_x*bwm_x + wi_y*bwm_y + wi_z*bwm_z;

		T wo_x = T(2)*cos_c*bwm_x - wi_x;
		T wo_y = T(2)*cos_c*bwm_y - wi_y;
		T wo_z = T(2)*cos_c*bwm_z - wi_z;
		if (wo_z <= T(0)) { res.valid = false; return res; }

		T G1_c = dist.G1(wi_x, wi_y, wi_z);
		T G_c  = dist.G(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
		T wt_c = (G1_c > T(1e-8)) ? G_c / G1_c : T(0);

		T F_r = FrComplex(cos_c, eta_r, k_r) * wt_c;
		T F_g = FrComplex(cos_c, eta_g, k_g) * wt_c;
		T F_b = FrComplex(cos_c, eta_b, k_b) * wt_c;

#if defined(__CUDACC__)
		T cos_out = fabsf(wo_z);
#else
		T cos_out = std::fabs(wo_z);
#endif
		T F_out = FrDielectric(cos_out, T(1) / coat_ior);
		T T_in  = T(1) - F_in;
		T T_out = T(1) - F_out;

		res.wo_x = wo_x; res.wo_y = wo_y; res.wo_z = wo_z;
		res.r = F_r * T_in * T_out;
		res.g = F_g * T_in * T_out;
		res.b = F_b * T_in * T_out;
		res.is_specular = true;
		res.valid = true;
		return res;
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
