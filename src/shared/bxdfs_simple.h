#pragma once
#include "bxdfs_base.h"

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
