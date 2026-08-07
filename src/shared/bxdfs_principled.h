#pragma once
#include "bxdfs_base.h"

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

		// Diffuse: Lambertian * (1-metallic) * (1-F_diffuse); F at wi·n per Disney
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
