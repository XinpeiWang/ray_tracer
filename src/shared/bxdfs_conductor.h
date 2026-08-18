#pragma once
#include "bxdfs_base.h"

// ggx_reflection_shape -- pure geometric GGX reflection shape D*G/(4*cosO*cosI)
// with NO Fresnel/color factor. RoughMetalBxDF::f() is just this shape times
// a constant per-channel albedo (its Fresnel is a flat, direction-independent
// tint). ConductorBxDF's real per-channel complex Fresnel DOES vary by
// queried direction, so its CPU material wrapper (conductor::
// scattering_attenuation() in material_pbrt.h) needs this shape and the
// Fresnel term as SEPARATE pieces rather than the fused f() below -- this
// codebase's material::scattering_pdf() returns a scalar, not a color, so a
// direction-varying color has to be threaded through
// scattering_attenuation() instead (see diffuse_transmission's own R/T
// split for the established precedent).
template<typename T>
CPU_GPU T ggx_reflection_shape(T wi_x, T wi_y, T wi_z, T wo_x, T wo_y, T wo_z,
								 T alpha_x, T alpha_y) {
	if (wi_z <= T(0) || wo_z <= T(0)) return T(0);
	TrowbridgeReitz<T> dist(alpha_x, alpha_y);
	T hx = wi_x + wo_x, hy = wi_y + wo_y, hz = wi_z + wo_z;
#if defined(__CUDACC__)
	T hlen = sqrtf(hx*hx + hy*hy + hz*hz);
#else
	T hlen = std::sqrt(hx*hx + hy*hy + hz*hz);
#endif
	if (hlen < T(1e-8)) return T(0);
	hx /= hlen; hy /= hlen; hz /= hlen;
	T D = dist.D(hx, hy, hz);
	T G = dist.G(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
	return D * G / (T(4) * wi_z * wo_z);
}

// ggx_vndf_reflection_pdf -- VNDF-sampling density for a GGX reflection
// lobe, shared by RoughMetalBxDF::pdf() and ConductorBxDF::pdf() (and by
// the CPU-side ggx_reflection_pdf wrapper in pdf.h, used as srec.pdf_ptr
// for real NEE/MIS): Fresnel doesn't affect the sampling density, only
// f(), so both BxDFs share this one geometric formula rather than each
// re-deriving it.
template<typename T>
CPU_GPU T ggx_vndf_reflection_pdf(T wi_x, T wi_y, T wi_z, T wo_x, T wo_y, T wo_z,
									T alpha_x, T alpha_y) {
	if (wi_z <= T(0) || wo_z <= T(0)) return T(0);
	TrowbridgeReitz<T> dist(alpha_x, alpha_y);
	T hx = wi_x + wo_x, hy = wi_y + wo_y, hz = wi_z + wo_z;
#if defined(__CUDACC__)
	T hlen = sqrtf(hx*hx + hy*hy + hz*hz);
#else
	T hlen = std::sqrt(hx*hx + hy*hy + hz*hz);
#endif
	if (hlen < T(1e-8)) return T(0);
	hx /= hlen; hy /= hlen; hz /= hlen;
	T pdf_wm = dist.PDF(wi_x, wi_y, wi_z, hx, hy, hz);
	T dot_wi_wm = wi_x*hx + wi_y*hy + wi_z*hz;
	if (dot_wi_wm <= T(0)) return T(0);
	return pdf_wm / (T(4) * dot_wi_wm);
}

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

	// f(wi, wo) -- real closed-form BSDF value at an arbitrary QUERIED
	// direction wo (both directions local frame, z=normal). Unlike
	// sample_local(), which only ever returns a wo it importance-sampled
	// itself, this evaluates the BSDF for whatever direction the caller is
	// aiming a shadow ray toward (real NEE/MIS) -- mirrors
	// PrincipledBxDF::ggx_brdf (bxdfs_principled.h), with a constant-color
	// tint standing in for Fresnel since this struct has no real Fresnel
	// model (see the header comment above).
	CPU_GPU void f(T wi_x, T wi_y, T wi_z, T wo_x, T wo_y, T wo_z,
					T& fr, T& fg, T& fb) const {
		fr = fg = fb = T(0);
		if (wi_z <= T(0) || wo_z <= T(0)) return;
		TrowbridgeReitz<T> dist(alpha_x, alpha_y);
		T hx = wi_x + wo_x, hy = wi_y + wo_y, hz = wi_z + wo_z;
#if defined(__CUDACC__)
		T hlen = sqrtf(hx*hx + hy*hy + hz*hz);
#else
		T hlen = std::sqrt(hx*hx + hy*hy + hz*hz);
#endif
		if (hlen < T(1e-8)) return;
		hx /= hlen; hy /= hlen; hz /= hlen;
		T D = dist.D(hx, hy, hz);
		T G = dist.G(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
		T val = D * G / (T(4) * wi_z * wo_z);
		fr = albedo_r * val; fg = albedo_g * val; fb = albedo_b * val;
	}

	// pdf(wi, wo) -- VNDF sampling density for the same queried wo. Mirrors
	// PrincipledBxDF::ggx_pdf.
	CPU_GPU T pdf(T wi_x, T wi_y, T wi_z, T wo_x, T wo_y, T wo_z) const {
		return ggx_vndf_reflection_pdf(wi_x, wi_y, wi_z, wo_x, wo_y, wo_z, alpha_x, alpha_y);
	}

	// Roughness-based specular/glossy classification (pbrt-v4
	// TrowbridgeReitz::EffectivelySmooth threshold, alpha < 1e-3): callers
	// use this to decide whether the material is a perfect specular delta
	// BSDF (no finite-solid-angle f()/pdf() to evaluate, NEE impossible --
	// what this codebase's is_specular=true previously meant unconditionally)
	// or genuinely glossy (real NEE/MIS via f()/pdf() above applies).
	CPU_GPU bool effectively_smooth() const {
		return TrowbridgeReitz<T>(alpha_x, alpha_y).EffectivelySmooth();
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

	// f(wi, wo) -- real closed-form BSDF value at an arbitrary queried
	// direction wo. See RoughMetalBxDF::f()'s comment for why this exists;
	// unlike RoughMetalBxDF, the Fresnel term here is the real per-channel
	// complex conductor Fresnel, evaluated at the half-vector between the
	// two GIVEN directions (not the internally-sampled wm) -- mirrors
	// PrincipledBxDF's own Fresnel-at-half-vector convention.
	CPU_GPU void f(T wi_x, T wi_y, T wi_z, T wo_x, T wo_y, T wo_z,
					T& fr, T& fg, T& fb) const {
		fr = fg = fb = T(0);
		if (wi_z <= T(0) || wo_z <= T(0)) return;
		TrowbridgeReitz<T> dist(alpha_x, alpha_y);
		T hx = wi_x + wo_x, hy = wi_y + wo_y, hz = wi_z + wo_z;
#if defined(__CUDACC__)
		T hlen = sqrtf(hx*hx + hy*hy + hz*hz);
#else
		T hlen = std::sqrt(hx*hx + hy*hy + hz*hz);
#endif
		if (hlen < T(1e-8)) return;
		hx /= hlen; hy /= hlen; hz /= hlen;
		T D = dist.D(hx, hy, hz);
		T G = dist.G(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
		T dot_wi_wm = wi_x*hx + wi_y*hy + wi_z*hz;
		T val = D * G / (T(4) * wi_z * wo_z);
		fr = FrComplex(dot_wi_wm, eta_r, k_r) * val;
		fg = FrComplex(dot_wi_wm, eta_g, k_g) * val;
		fb = FrComplex(dot_wi_wm, eta_b, k_b) * val;
	}

	// pdf(wi, wo) -- VNDF sampling density for the same queried wo.
	// Purely geometric (no Fresnel term), same formula as
	// RoughMetalBxDF::pdf().
	CPU_GPU T pdf(T wi_x, T wi_y, T wi_z, T wo_x, T wo_y, T wo_z) const {
		return ggx_vndf_reflection_pdf(wi_x, wi_y, wi_z, wo_x, wo_y, wo_z, alpha_x, alpha_y);
	}

	// Roughness-based specular/glossy classification -- see
	// RoughMetalBxDF::effectively_smooth()'s comment.
	CPU_GPU bool effectively_smooth() const {
		return TrowbridgeReitz<T>(alpha_x, alpha_y).EffectivelySmooth();
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
		res.eta = T(1);
		res.is_specular = true;
		res.is_transmission = false;

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
				// Genuine transmission -- record eta for the integrator's
				// etaScale/Russian-roulette bookkeeping (pbrt-v4 bs->eta).
				// Previously left at the res{} default (0), which the
				// material wrapper didn't even propagate to scatter_record
				// in the first place -- see scatter_record's own comment.
				res.eta = eta;
				res.is_transmission = true;
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
#include "scalar_math.h"

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
	return SafeSqrt(x);
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
