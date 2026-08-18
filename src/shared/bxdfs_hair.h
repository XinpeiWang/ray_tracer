#pragma once
#include "bxdfs_base.h"

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
	// wi_z = cosTheta_i * sin(phi_i)  — the z-component of wi in local frame.
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

		// Divide by AbsCosTheta(wi) = |wi_z| — matches pbrt-v4 HairBxDF::f()
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

		// fr and pdf are both derived from the same Mp/Np lobe math but via
		// separate code paths (eval_local vs scattering_pdf_local); near the
		// 1e-8 floor above, small inconsistencies between the two could make
		// fr/pdf spike into extreme outliers on rare samples. Clamp the
		// ratio well above any physically-expected value (this BSDF's
		// energy-conserving lobes sum to ~1, see the WhiteFurnaceBound unit
		// test) rather than above the noise floor itself, so ordinary
		// samples are untouched and only true numerical spikes are capped.
		// (Scene B11's visible blown-white look traced to a miscalibrated
		// light intensity, not this - see build_hair_fibers()'s comment -
		// this clamp is a general defensive safety net, kept regardless.)
		const T kMaxAttenuation = T(50);
		T ratio_r = fr / pdf, ratio_g = fg / pdf, ratio_b = fb / pdf;
		res.wo_x = wo_wx; res.wo_y = wo_wy; res.wo_z = wo_wz;
		res.r = ratio_r < kMaxAttenuation ? ratio_r : kMaxAttenuation;
		res.g = ratio_g < kMaxAttenuation ? ratio_g : kMaxAttenuation;
		res.b = ratio_b < kMaxAttenuation ? ratio_b : kMaxAttenuation;
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

