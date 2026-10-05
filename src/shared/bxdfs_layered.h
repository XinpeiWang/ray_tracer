#pragma once
#include "bxdfs_base.h"
#include <cstring>

// ===========================================================================
// pbrt-v4 LayeredBxDF<Top, Bottom, twoSided=true>, ported from src/pbrt/bxdfs.h
// (LayeredBxDF::f / Sample_f / PDF) together with the three interface BxDFs it
// is instantiated with here: DielectricBxDF (coat), ConductorBxDF and
// DiffuseBxDF (base). CoatedDiffuseBxDF and CoatedConductorBxDF at the bottom
// of this section are thin wrappers that keep the field layout their brace-init
// call sites (materials.h, optix_device_helpers.h, wavefront kernels) rely on.
//
// Replaces an earlier simplified model that never refracted at the coat (it
// mirrored about the half-vector and flipped z), used one roughness for both
// interfaces and evaluated NEE with a one-connection estimator: a smooth coat
// over Cu read about a third of the analytic value on the CPU, and the GPU
// differed from the CPU again because its inline sampler was a different
// single-bounce model.
//
// Conventions (same as pbrt):
//   - every direction is in the local shading frame, z = normal, and points
//     AWAY from the surface (wo, wi both > 0 for a reflection);
//   - `radiance` is TransportMode::Radiance. The 1/eta'^2 factors it adds on
//     the coat's transmission lobes cancel exactly between entering and
//     leaving the layer, so a coated BSDF carries no net eta^2;
//   - sample_local() returns the random-walk weight f*|cos|/pdf (pbrt's
//     BSDFSample with pdfIsProportional = true). pdf() is the separate,
//     approximate PDF() estimate pbrt uses for MIS, and f() is the stochastic
//     BSDF value used for next-event estimation.
// ===========================================================================
namespace layered_detail {

constexpr int kReflection   = 1;
constexpr int kTransmission = 2;
constexpr int kAllLobes     = 3;

template<typename T> struct V3 { T x, y, z; };
template<typename T> struct S3 {
	T r, g, b;
	CPU_GPU T max_component() const {
		T m = r > g ? r : g;
		return m > b ? m : b;
	}
	CPU_GPU bool nonzero() const { return r != T(0) || g != T(0) || b != T(0); }
};
template<typename T> CPU_GPU V3<T> operator-(V3<T> a) { return {-a.x, -a.y, -a.z}; }
template<typename T> CPU_GPU S3<T> operator*(S3<T> a, T s) { return {a.r*s, a.g*s, a.b*s}; }
template<typename T> CPU_GPU S3<T> operator*(T s, S3<T> a) { return {a.r*s, a.g*s, a.b*s}; }
template<typename T> CPU_GPU S3<T> operator*(S3<T> a, S3<T> b) { return {a.r*b.r, a.g*b.g, a.b*b.b}; }
template<typename T> CPU_GPU S3<T> operator/(S3<T> a, T s) { return {a.r/s, a.g/s, a.b/s}; }
template<typename T> CPU_GPU S3<T> operator+(S3<T> a, S3<T> b) { return {a.r+b.r, a.g+b.g, a.b+b.b}; }
template<typename T> CPU_GPU V3<T> v_scale(V3<T> a, T s) { return {a.x*s, a.y*s, a.z*s}; }
template<typename T> CPU_GPU T v_dot(V3<T> a, V3<T> b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

template<typename T> CPU_GPU T l_abs(T x) { return x < T(0) ? -x : x; }
template<typename T> CPU_GPU T l_sqrt(T x) {
#if defined(__CUDACC__)
	return sqrtf(x);
#else
	return std::sqrt(x);
#endif
}
template<typename T> CPU_GPU T l_exp(T x) {
#if defined(__CUDACC__)
	return expf(x);
#else
	return std::exp(x);
#endif
}
template<typename T> CPU_GPU T l_log(T x) {
#if defined(__CUDACC__)
	return logf(x);
#else
	return std::log(x);
#endif
}
template<typename T> CPU_GPU T l_sqr(T x) { return x * x; }
template<typename T> CPU_GPU T l_safe_sqrt(T x) { return x > T(0) ? l_sqrt(x) : T(0); }
template<typename T> CPU_GPU V3<T> v_normalize(V3<T> a) {
	T inv = T(1) / l_sqrt(v_dot(a, a));
	return {a.x*inv, a.y*inv, a.z*inv};
}
template<typename T> CPU_GPU bool same_hemisphere(V3<T> a, V3<T> b) { return a.z * b.z > T(0); }

using PCG32 = RNG;

template<typename T>
CPU_GPU T next_uniform(PCG32& rng) {
	T u = (T)rng.Uniform<float>();
	return u < T(0.99999994) ? u : T(0.99999994);
}

// 64-bit finaliser (pbrt MixBits) over the bit patterns of a few floats - pbrt seeds PDF() with
// Hash(wi), Hash(wo) so the same direction pair always sees the same estimate; MIS needs the two
// strategies to agree on the pdf of a direction they both could have generated.
CPU_GPU uint64_t mix64(uint64_t v) {
	v ^= (v >> 31);
	v *= 0x7fb5d329728ea185ull;
	v ^= (v >> 27);
	v *= 0x81dadef4bc2dd44dull;
	v ^= (v >> 33);
	return v;
}
template<typename T>
CPU_GPU uint64_t hash_dir(V3<T> d) {
	float f[3] = {(float)d.x, (float)d.y, (float)d.z};
	uint32_t b[3];
#if defined(__CUDA_ARCH__)
	b[0] = __float_as_uint(f[0]); b[1] = __float_as_uint(f[1]); b[2] = __float_as_uint(f[2]);
#else
	std::memcpy(b, f, sizeof(b));
#endif
	uint64_t h = 0x9e3779b97f4a7c15ull;
	for (int i = 0; i < 3; ++i) h = mix64(h ^ ((uint64_t)b[i] + 0x9e3779b97f4a7c15ull + (h << 6)));
	return h;
}

// pbrt Tr(dz, w): unit extinction, so transmittance through dz of slab along w
template<typename T>
CPU_GPU T Tr(T dz, V3<T> w) {
	if (l_abs(dz) <= T(1.17549435e-38)) return T(1);
	return l_exp(-l_abs(dz / w.z));
}

template<typename T>
CPU_GPU T power2(T a, T b) { return (a*a) / (a*a + b*b); }

// SampleExponential() and hg_eval() come from the layered_detail helpers at the end of bxdfs_conductor.h.

// pbrt SampleHenyeyGreenstein: wi around wo (wo points away, forward scattering is wi = -wo).
template<typename T>
CPU_GPU V3<T> hg_sample(V3<T> wo, T g, T u0, T u1, T& pdf) {
	T cos_theta;
	if (l_abs(g) < T(1e-3)) cos_theta = T(1) - T(2) * u0;
	else cos_theta = -T(1) / (T(2) * g) * (T(1) + g*g - l_sqr((T(1) - g*g) / (T(1) + g - T(2) * g * u0)));
	T sin_theta = l_safe_sqrt(T(1) - cos_theta * cos_theta);
	T phi = T(2) * T(3.14159265358979323846) * u1;
	T tx, ty, tz, bx, by, bz;
	make_onb(wo.x, wo.y, wo.z, tx, ty, tz, bx, by, bz);
#if defined(__CUDACC__)
	T sp = sinf(phi), cp = cosf(phi);
#else
	T sp = std::sin(phi), cp = std::cos(phi);
#endif
	T sx = sin_theta * cp, sy = sin_theta * sp, sz = cos_theta;
	pdf = hg_eval(cos_theta, g);
	return {sx*tx + sy*bx + sz*wo.x, sx*ty + sy*by + sz*wo.y, sx*tz + sy*bz + sz*wo.z};
}

// ---- interface BxDFs -------------------------------------------------------------------------
template<typename T>
struct LayerSample {
	S3<T> f;
	V3<T> wi;
	T pdf;
	bool valid;
	bool reflection;
	bool transmission;
	bool specular;
};

template<typename T>
CPU_GPU LayerSample<T> no_sample() {
	LayerSample<T> s{};
	s.valid = false;
	return s;
}

// pbrt Refract(wi, n, eta, &etap, &wt)
template<typename T>
CPU_GPU bool refract(V3<T> wi, V3<T> n, T eta, T& etap, V3<T>& wt) {
	T cos_i = v_dot(n, wi);
	if (cos_i < T(0)) { eta = T(1) / eta; cos_i = -cos_i; n = -n; }
	T sin2_i = T(1) - cos_i * cos_i;
	if (sin2_i < T(0)) sin2_i = T(0);
	T sin2_t = sin2_i / (eta * eta);
	if (sin2_t >= T(1)) return false;
	T cos_t = l_safe_sqrt(T(1) - sin2_t);
	T k = cos_i / eta - cos_t;
	wt = {-wi.x / eta + k * n.x, -wi.y / eta + k * n.y, -wi.z / eta + k * n.z};
	etap = eta;
	return true;
}

template<typename T>
struct DielectricLayer {
	T eta;
	T alpha_x, alpha_y;

	CPU_GPU bool smooth() const { return eta == T(1) || TrowbridgeReitz<T>(alpha_x, alpha_y).EffectivelySmooth(); }

	CPU_GPU S3<T> f(V3<T> wo, V3<T> wi, bool radiance) const {
		if (smooth()) return {0, 0, 0};
		TrowbridgeReitz<T> distrib(alpha_x, alpha_y);
		T cos_o = wo.z, cos_i = wi.z;
		bool reflect = cos_i * cos_o > T(0);
		T etap = T(1);
		if (!reflect) etap = cos_o > T(0) ? eta : T(1) / eta;
		V3<T> wm = {wi.x*etap + wo.x, wi.y*etap + wo.y, wi.z*etap + wo.z};
		if (cos_i == T(0) || cos_o == T(0) || v_dot(wm, wm) == T(0)) return {0, 0, 0};
		wm = v_normalize(wm);
		if (wm.z < T(0)) wm = -wm;
		if (v_dot(wm, wi) * cos_i < T(0) || v_dot(wm, wo) * cos_o < T(0)) return {0, 0, 0};
		T F = FrDielectric(v_dot(wo, wm), eta);
		if (reflect) {
			T v = distrib.D(wm.x, wm.y, wm.z) * distrib.G(wo.x, wo.y, wo.z, wi.x, wi.y, wi.z) * F / l_abs(T(4) * cos_i * cos_o);
			return {v, v, v};
		}
		T denom = l_sqr(v_dot(wi, wm) + v_dot(wo, wm) / etap) * cos_i * cos_o;
		T ft = distrib.D(wm.x, wm.y, wm.z) * (T(1) - F) * distrib.G(wo.x, wo.y, wo.z, wi.x, wi.y, wi.z)
		     * l_abs(v_dot(wi, wm) * v_dot(wo, wm) / denom);
		if (radiance) ft /= etap * etap;
		return {ft, ft, ft};
	}

	CPU_GPU T PDF(V3<T> wo, V3<T> wi, bool /*radiance*/, int flags) const {
		if (smooth()) return T(0);
		TrowbridgeReitz<T> distrib(alpha_x, alpha_y);
		T cos_o = wo.z, cos_i = wi.z;
		bool reflect = cos_i * cos_o > T(0);
		T etap = T(1);
		if (!reflect) etap = cos_o > T(0) ? eta : T(1) / eta;
		V3<T> wm = {wi.x*etap + wo.x, wi.y*etap + wo.y, wi.z*etap + wo.z};
		if (cos_i == T(0) || cos_o == T(0) || v_dot(wm, wm) == T(0)) return T(0);
		wm = v_normalize(wm);
		if (wm.z < T(0)) wm = -wm;
		if (v_dot(wm, wi) * cos_i < T(0) || v_dot(wm, wo) * cos_o < T(0)) return T(0);
		T R = FrDielectric(v_dot(wo, wm), eta);
		T Tt = T(1) - R;
		T pr = (flags & kReflection) ? R : T(0), pt = (flags & kTransmission) ? Tt : T(0);
		if (pr == T(0) && pt == T(0)) return T(0);
		T pdf_wm = distrib.PDF(wo.x, wo.y, wo.z, wm.x, wm.y, wm.z);
		if (reflect) return pdf_wm / (T(4) * l_abs(v_dot(wo, wm))) * pr / (pr + pt);
		T denom = l_sqr(v_dot(wi, wm) + v_dot(wo, wm) / etap);
		T dwm_dwi = l_abs(v_dot(wi, wm)) / denom;
		return pdf_wm * dwm_dwi * pt / (pr + pt);
	}

	CPU_GPU LayerSample<T> Sample_f(V3<T> wo, T uc, T u0, T u1, bool radiance, int flags) const {
		LayerSample<T> s = no_sample<T>();
		if (smooth()) {
			T R = FrDielectric(wo.z, eta), Tt = T(1) - R;
			T pr = (flags & kReflection) ? R : T(0), pt = (flags & kTransmission) ? Tt : T(0);
			if (pr == T(0) && pt == T(0)) return s;
			if (uc < pr / (pr + pt)) {
				V3<T> wi = {-wo.x, -wo.y, wo.z};
				T v = R / l_abs(wi.z);
				s.f = {v, v, v}; s.wi = wi; s.pdf = pr / (pr + pt);
				s.valid = true; s.reflection = true; s.specular = true;
				return s;
			}
			V3<T> wi; T etap;
			if (!refract(wo, V3<T>{0, 0, 1}, eta, etap, wi)) return s;
			T v = Tt / l_abs(wi.z);
			if (radiance) v /= etap * etap;
			s.f = {v, v, v}; s.wi = wi; s.pdf = pt / (pr + pt);
			s.valid = true; s.transmission = true; s.specular = true;
			return s;
		}
		TrowbridgeReitz<T> distrib(alpha_x, alpha_y);
		V3<T> wm;
		distrib.Sample_wm(wo.x, wo.y, wo.z, u0, u1, wm.x, wm.y, wm.z);
		T R = FrDielectric(v_dot(wo, wm), eta), Tt = T(1) - R;
		T pr = (flags & kReflection) ? R : T(0), pt = (flags & kTransmission) ? Tt : T(0);
		if (pr == T(0) && pt == T(0)) return s;
		if (uc < pr / (pr + pt)) {
			T d = v_dot(wo, wm);
			V3<T> wi = {-wo.x + T(2)*d*wm.x, -wo.y + T(2)*d*wm.y, -wo.z + T(2)*d*wm.z};
			if (!same_hemisphere(wo, wi)) return s;
			s.pdf = distrib.PDF(wo.x, wo.y, wo.z, wm.x, wm.y, wm.z) / (T(4) * l_abs(d)) * pr / (pr + pt);
			T v = distrib.D(wm.x, wm.y, wm.z) * distrib.G(wo.x, wo.y, wo.z, wi.x, wi.y, wi.z) * R / (T(4) * wi.z * wo.z);
			s.f = {v, v, v}; s.wi = wi;
			s.valid = true; s.reflection = true;
			return s;
		}
		T etap; V3<T> wi;
		bool tir = !refract(wo, wm, eta, etap, wi);
		if (tir || same_hemisphere(wo, wi) || wi.z == T(0)) return s;
		T denom = l_sqr(v_dot(wi, wm) + v_dot(wo, wm) / etap);
		T dwm_dwi = l_abs(v_dot(wi, wm)) / denom;
		s.pdf = distrib.PDF(wo.x, wo.y, wo.z, wm.x, wm.y, wm.z) * dwm_dwi * pt / (pr + pt);
		T ft = Tt * distrib.D(wm.x, wm.y, wm.z) * distrib.G(wo.x, wo.y, wo.z, wi.x, wi.y, wi.z)
		     * l_abs(v_dot(wi, wm) * v_dot(wo, wm) / (wi.z * wo.z * denom));
		if (radiance) ft /= etap * etap;
		s.f = {ft, ft, ft}; s.wi = wi;
		s.valid = true; s.transmission = true;
		return s;
	}
};

template<typename T>
struct ConductorLayer {
	S3<T> eta, k;
	T alpha_x, alpha_y;

	CPU_GPU bool smooth() const { return TrowbridgeReitz<T>(alpha_x, alpha_y).EffectivelySmooth(); }
	CPU_GPU bool specular() const { return smooth(); }

	CPU_GPU S3<T> fresnel(T cos_theta) const {
		return {FrComplex(cos_theta, eta.r, k.r), FrComplex(cos_theta, eta.g, k.g), FrComplex(cos_theta, eta.b, k.b)};
	}

	CPU_GPU S3<T> f(V3<T> wo, V3<T> wi, bool /*radiance*/) const {
		if (!same_hemisphere(wo, wi) || smooth()) return {0, 0, 0};
		T cos_o = l_abs(wo.z), cos_i = l_abs(wi.z);
		if (cos_i == T(0) || cos_o == T(0)) return {0, 0, 0};
		V3<T> wm = {wi.x + wo.x, wi.y + wo.y, wi.z + wo.z};
		if (v_dot(wm, wm) == T(0)) return {0, 0, 0};
		wm = v_normalize(wm);
		TrowbridgeReitz<T> distrib(alpha_x, alpha_y);
		S3<T> F = fresnel(l_abs(v_dot(wo, wm)));
		T scale = distrib.D(wm.x, wm.y, wm.z) * distrib.G(wo.x, wo.y, wo.z, wi.x, wi.y, wi.z) / (T(4) * cos_i * cos_o);
		return F * scale;
	}

	CPU_GPU T PDF(V3<T> wo, V3<T> wi, bool /*radiance*/, int flags) const {
		if (!(flags & kReflection) || !same_hemisphere(wo, wi) || smooth()) return T(0);
		V3<T> wm = {wo.x + wi.x, wo.y + wi.y, wo.z + wi.z};
		if (v_dot(wm, wm) == T(0)) return T(0);
		wm = v_normalize(wm);
		if (wm.z < T(0)) wm = -wm;
		TrowbridgeReitz<T> distrib(alpha_x, alpha_y);
		return distrib.PDF(wo.x, wo.y, wo.z, wm.x, wm.y, wm.z) / (T(4) * l_abs(v_dot(wo, wm)));
	}

	CPU_GPU LayerSample<T> Sample_f(V3<T> wo, T /*uc*/, T u0, T u1, bool /*radiance*/, int flags) const {
		LayerSample<T> s = no_sample<T>();
		if (!(flags & kReflection)) return s;
		if (smooth()) {
			V3<T> wi = {-wo.x, -wo.y, wo.z};
			T c = l_abs(wi.z);
			S3<T> F = fresnel(c);
			s.f = F / c; s.wi = wi; s.pdf = T(1);
			s.valid = true; s.reflection = true; s.specular = true;
			return s;
		}
		if (wo.z == T(0)) return s;
		TrowbridgeReitz<T> distrib(alpha_x, alpha_y);
		V3<T> wm;
		distrib.Sample_wm(wo.x, wo.y, wo.z, u0, u1, wm.x, wm.y, wm.z);
		T d = v_dot(wo, wm);
		V3<T> wi = {-wo.x + T(2)*d*wm.x, -wo.y + T(2)*d*wm.y, -wo.z + T(2)*d*wm.z};
		if (!same_hemisphere(wo, wi)) return s;
		s.pdf = distrib.PDF(wo.x, wo.y, wo.z, wm.x, wm.y, wm.z) / (T(4) * l_abs(d));
		T cos_o = l_abs(wo.z), cos_i = l_abs(wi.z);
		if (cos_i == T(0) || cos_o == T(0)) return s;
		S3<T> F = fresnel(l_abs(d));
		T scale = distrib.D(wm.x, wm.y, wm.z) * distrib.G(wo.x, wo.y, wo.z, wi.x, wi.y, wi.z) / (T(4) * cos_i * cos_o);
		s.f = F * scale; s.wi = wi;
		s.valid = true; s.reflection = true;
		return s;
	}
};

template<typename T>
struct DiffuseLayer {
	S3<T> R;

	CPU_GPU bool specular() const { return false; }

	CPU_GPU S3<T> f(V3<T> wo, V3<T> wi, bool /*radiance*/) const {
		if (!same_hemisphere(wo, wi)) return {0, 0, 0};
		return R * (T(1) / T(3.14159265358979323846));
	}

	CPU_GPU T PDF(V3<T> wo, V3<T> wi, bool /*radiance*/, int flags) const {
		if (!(flags & kReflection) || !same_hemisphere(wo, wi)) return T(0);
		return l_abs(wi.z) / T(3.14159265358979323846);
	}

	CPU_GPU LayerSample<T> Sample_f(V3<T> wo, T /*uc*/, T u0, T u1, bool /*radiance*/, int flags) const {
		LayerSample<T> s = no_sample<T>();
		if (!(flags & kReflection)) return s;
		T x, y, z, pdf_unused;
		SampleCosineHemisphere(u0, u1, x, y, z, pdf_unused);
		if (wo.z < T(0)) z = -z;
		s.wi = {x, y, z};
		s.pdf = l_abs(z) / T(3.14159265358979323846);
		s.f = R * (T(1) / T(3.14159265358979323846));
		s.valid = true; s.reflection = true;
		return s;
	}
};

// ---- the random walks -------------------------------------------------------------------------
template<typename T>
struct LayerParams {
	T thickness;
	S3<T> albedo;   // medium single-scattering albedo, all zero = no medium
	T g;
	int maxDepth;
	int nSamples;
};

// LayeredBxDF::f(wo, wi) for wo.z > 0 and wi.z > 0 (the only case an opaque base produces a value in)
template<typename T, typename Bottom>
CPU_GPU S3<T> layered_f(const DielectricLayer<T>& top, const Bottom& bottom, const LayerParams<T>& P,
                        V3<T> wo, V3<T> wi, uint64_t seed0, uint64_t seed1, bool radiance = true)
{
	S3<T> f{0, 0, 0};
	if (wo.z <= T(0) || wi.z <= T(0)) return f;
	const T thickness = P.thickness;
	const bool has_medium = P.albedo.nonzero();
	const bool top_specular = top.smooth();
	const bool bottom_specular = bottom.specular();
	const T exit_z = thickness;
	const int nSamples = P.nSamples > 0 ? P.nSamples : 1;

	f = top.f(wo, wi, radiance) * (T)nSamples;

	PCG32 rng(seed0, seed1 ^ 0x9e3779b9ull);

	for (int s = 0; s < nSamples; ++s) {
		T uc = next_uniform<T>(rng);
		T a0 = next_uniform<T>(rng), a1 = next_uniform<T>(rng);
		LayerSample<T> wos = top.Sample_f(wo, uc, a0, a1, radiance, kTransmission);
		if (!wos.valid || !wos.f.nonzero() || wos.pdf == T(0) || wos.wi.z == T(0)) continue;

		uc = next_uniform<T>(rng);
		a0 = next_uniform<T>(rng); a1 = next_uniform<T>(rng);
		LayerSample<T> wis = top.Sample_f(wi, uc, a0, a1, !radiance, kTransmission);
		if (!wis.valid || !wis.f.nonzero() || wis.pdf == T(0) || wis.wi.z == T(0)) continue;

		S3<T> beta = wos.f * (l_abs(wos.wi.z) / wos.pdf);
		T z = thickness;
		V3<T> w = wos.wi;

		for (int depth = 0; depth < P.maxDepth; ++depth) {
			if (depth > 3 && beta.max_component() < T(0.25)) {
				T mx = beta.max_component();
				T q = T(1) - mx > T(0) ? T(1) - mx : T(0);
				if (next_uniform<T>(rng) < q) break;
				beta = beta / (T(1) - q);
			}

			if (!has_medium) {
				z = (z == thickness) ? T(0) : thickness;
				beta = beta * Tr(thickness, w);
			} else {
				T dz = SampleExponential<T>(next_uniform<T>(rng), T(1) / l_abs(w.z));
				T zp = w.z > T(0) ? (z + dz) : (z - dz);
				if (z == zp) continue;
				if (T(0) < zp && zp < thickness) {
					T wt = T(1);
					T phase_exit = hg_eval(v_dot(-w, -wis.wi), P.g);
					if (!top_specular) wt = power2(wis.pdf, phase_exit);
					f = f + beta * P.albedo * wis.f * (phase_exit * wt * Tr(zp - exit_z, wis.wi) / wis.pdf);

					T pdf_ps;
					V3<T> psw = hg_sample(-w, P.g, next_uniform<T>(rng), next_uniform<T>(rng), pdf_ps);
					if (pdf_ps == T(0) || psw.z == T(0)) continue;
					// p/pdf == 1 for HG sampling
					beta = beta * P.albedo;
					w = psw;
					z = zp;

					if (((z < exit_z && w.z > T(0)) || (z > exit_z && w.z < T(0))) && !top_specular) {
						S3<T> fExit = top.f(-w, wi, radiance);
						if (fExit.nonzero()) {
							T exitPDF = top.PDF(-w, wi, radiance, kTransmission);
							T wt2 = power2(pdf_ps, exitPDF);
							f = f + beta * fExit * (Tr(zp - exit_z, psw) * wt2);
						}
					}
					continue;
				}
				z = zp < T(0) ? T(0) : (zp > thickness ? thickness : zp);
			}

			if (z == exit_z) {
				// reflection at the exit (top) interface
				T uc2 = next_uniform<T>(rng);
				T b0 = next_uniform<T>(rng), b1 = next_uniform<T>(rng);
				LayerSample<T> bs = top.Sample_f(-w, uc2, b0, b1, radiance, kReflection);
				if (!bs.valid || !bs.f.nonzero() || bs.pdf == T(0) || bs.wi.z == T(0)) break;
				beta = beta * bs.f * (l_abs(bs.wi.z) / bs.pdf);
				w = bs.wi;
			} else {
				// scattering at the non-exit (bottom) interface
				if (!bottom_specular) {
					T wt = T(1);
					if (!top_specular) {
						wt = power2(wis.pdf, bottom.PDF(-w, -wis.wi, radiance, kAllLobes));
					}
					f = f + beta * bottom.f(-w, -wis.wi, radiance) * wis.f
					      * (l_abs(wis.wi.z) * wt * Tr(thickness, wis.wi) / wis.pdf);
				}
				T uc2 = next_uniform<T>(rng);
				T b0 = next_uniform<T>(rng), b1 = next_uniform<T>(rng);
				LayerSample<T> bs = bottom.Sample_f(-w, uc2, b0, b1, radiance, kReflection);
				if (!bs.valid || !bs.f.nonzero() || bs.pdf == T(0) || bs.wi.z == T(0)) break;
				beta = beta * bs.f * (l_abs(bs.wi.z) / bs.pdf);
				w = bs.wi;

				if (!top_specular) {
					S3<T> fExit = top.f(-w, wi, radiance);
					if (fExit.nonzero()) {
						T wt = T(1);
						if (!bottom_specular) {
							// Verbatim pbrt. Note exitPDF is the density of generating `wi` from inside, not of the
							// competing strategy (top.PDF(wi, -w, !radiance, kTransmission) - the direction wis would
							// have drawn), so the two MIS weights do not sum to one: with a rough coat over a rough
							// base this f() runs 5-9% above the walk in sample_local() (measured: albedo 0.594 vs
							// 0.547 at alpha 0.3; swapping in the competing density gives 0.546). pbrt-v4 renders
							// with exactly this, so it is kept for parity.
							T exitPDF = top.PDF(-w, wi, radiance, kTransmission);
							wt = power2(bs.pdf, exitPDF);
						}
						f = f + beta * fExit * (Tr(thickness, bs.wi) * wt);
					}
				}
			}
		}
	}
	return f / (T)nSamples;
}

// LayeredBxDF::Sample_f(wo) -> (f, wi, pdf, specularPath). `f`/`pdf` are the walk's running
// products (proportional to the true values, ratio f*|cos|/pdf is the unbiased path weight).
template<typename T, typename Bottom>
CPU_GPU LayerSample<T> layered_sample(const DielectricLayer<T>& top, const Bottom& bottom, const LayerParams<T>& P,
                                      V3<T> wo, uint64_t seed0, uint64_t seed1, bool radiance = true)
{
	LayerSample<T> none = no_sample<T>();
	if (wo.z <= T(0)) return none;
	const T thickness = P.thickness;
	const bool has_medium = P.albedo.nonzero();

	PCG32 rng(seed0, seed1 ^ 0xdeadbeefull);
	T uc = next_uniform<T>(rng), u0 = next_uniform<T>(rng), u1 = next_uniform<T>(rng);
	LayerSample<T> bs = top.Sample_f(wo, uc, u0, u1, radiance, kAllLobes);
	if (!bs.valid || !bs.f.nonzero() || bs.pdf == T(0) || bs.wi.z == T(0)) return none;
	if (bs.reflection) return bs;   // pbrt: pdfIsProportional

	V3<T> w = bs.wi;
	bool specular_path = bs.specular;
	S3<T> f = bs.f * l_abs(bs.wi.z);
	T pdf = bs.pdf;
	T z = thickness;

	for (int depth = 0; depth < P.maxDepth; ++depth) {
		T rr_beta = f.max_component() / pdf;
		if (depth > 3 && rr_beta < T(0.25)) {
			T q = T(1) - rr_beta > T(0) ? T(1) - rr_beta : T(0);
			if (next_uniform<T>(rng) < q) return none;
			pdf *= T(1) - q;
		}
		if (w.z == T(0)) return none;

		if (has_medium) {
			T dz = SampleExponential<T>(next_uniform<T>(rng), T(1) / l_abs(w.z));
			T zp = w.z > T(0) ? (z + dz) : (z - dz);
			if (zp == z) return none;
			if (T(0) < zp && zp < thickness) {
				T pdf_ps;
				T p0 = next_uniform<T>(rng), p1 = next_uniform<T>(rng);
				V3<T> psw = hg_sample(-w, P.g, p0, p1, pdf_ps);
				if (pdf_ps == T(0) || psw.z == T(0)) return none;
				f = f * P.albedo * pdf_ps;     // ps->p == ps->pdf for HG
				pdf *= pdf_ps;
				specular_path = false;
				w = psw;
				z = zp;
				continue;
			}
			z = zp < T(0) ? T(0) : (zp > thickness ? thickness : zp);
		} else {
			z = (z == thickness) ? T(0) : thickness;
			f = f * Tr(thickness, w);
		}

		LayerSample<T> is;
		T c0 = next_uniform<T>(rng), c1 = next_uniform<T>(rng), c2 = next_uniform<T>(rng);
		if (z == T(0)) is = bottom.Sample_f(-w, c0, c1, c2, radiance, kAllLobes);
		else           is = top.Sample_f(-w, c0, c1, c2, radiance, kAllLobes);
		if (!is.valid || !is.f.nonzero() || is.pdf == T(0) || is.wi.z == T(0)) return none;
		f = f * is.f;
		pdf *= is.pdf;
		specular_path = specular_path && is.specular;
		w = is.wi;

		if (is.transmission) {
			LayerSample<T> out{};
			out.valid = true;
			out.reflection = same_hemisphere(wo, w);
			out.transmission = !out.reflection;
			out.specular = specular_path;
			out.f = f; out.wi = w; out.pdf = pdf;
			return out;
		}
		f = f * l_abs(is.wi.z);
	}
	return none;
}

// LayeredBxDF::PDF(wo, wi) for wo.z > 0, wi.z > 0.
template<typename T, typename Bottom>
CPU_GPU T layered_pdf(const DielectricLayer<T>& top, const Bottom& bottom, const LayerParams<T>& P,
                      V3<T> wo, V3<T> wi, bool radiance = true)
{
	if (wo.z <= T(0) || wi.z <= T(0)) return T(0);
	const int nSamples = P.nSamples > 0 ? P.nSamples : 1;
	PCG32 rng(hash_dir(wi), hash_dir(wo));

	T pdf_sum = (T)nSamples * top.PDF(wo, wi, radiance, kReflection);
	const bool top_specular = top.smooth();
	const bool bottom_specular = bottom.specular();

	for (int s = 0; s < nSamples; ++s) {
		// TRT term
		T a = next_uniform<T>(rng), b = next_uniform<T>(rng), c = next_uniform<T>(rng);
		LayerSample<T> wos = top.Sample_f(wo, a, b, c, radiance, kTransmission);
		a = next_uniform<T>(rng); b = next_uniform<T>(rng); c = next_uniform<T>(rng);
		LayerSample<T> wis = top.Sample_f(wi, a, b, c, !radiance, kTransmission);

		if (wos.valid && wos.f.nonzero() && wos.pdf > T(0) && wis.valid && wis.f.nonzero() && wis.pdf > T(0)) {
			if (top_specular) {
				pdf_sum += bottom.PDF(-wos.wi, -wis.wi, radiance, kAllLobes);
			} else {
				a = next_uniform<T>(rng); b = next_uniform<T>(rng); c = next_uniform<T>(rng);
				LayerSample<T> rs = bottom.Sample_f(-wos.wi, a, b, c, radiance, kAllLobes);
				if (rs.valid && rs.f.nonzero() && rs.pdf > T(0)) {
					if (bottom_specular) {
						pdf_sum += top.PDF(-rs.wi, wi, radiance, kAllLobes);
					} else {
						T rPDF = bottom.PDF(-wos.wi, -wis.wi, radiance, kAllLobes);
						T wt = power2(wis.pdf, rPDF);
						pdf_sum += wt * rPDF;

						T tPDF = top.PDF(-rs.wi, wi, radiance, kAllLobes);
						wt = power2(rs.pdf, tPDF);
						pdf_sum += wt * tPDF;
					}
				}
			}
		}
	}
	// Lerp(0.9, 1/(4 pi), pdfSum / nSamples)
	const T inv4pi = T(1) / (T(4) * T(3.14159265358979323846));
	return (T(1) - T(0.9)) * inv4pi + T(0.9) * (pdf_sum / (T)nSamples);
}

template<typename T>
CPU_GPU BxDFSampleResult<T> to_result(const LayerSample<T>& s)
{
	BxDFSampleResult<T> res{};
	res.eta = T(1);
	res.is_transmission = false;
	if (!s.valid) { res.valid = false; return res; }
	T wgt = l_abs(s.wi.z) / s.pdf;
	res.wo_x = s.wi.x; res.wo_y = s.wi.y; res.wo_z = s.wi.z;
	res.r = s.f.r * wgt; res.g = s.f.g * wgt; res.b = s.f.b * wgt;
	res.is_specular = s.specular;
	res.valid = true;
	return res;
}

} // namespace layered_detail

// ===========================================================================
// 8. CoatedDiffuseBxDF  (dielectric coat over Lambertian base)
//    pbrt-v4 CoatedDiffuseBxDF = LayeredBxDF<DielectricBxDF, DiffuseBxDF, true>
//    All in the local frame (z = surface normal).
// ===========================================================================
template<typename T>
struct CoatedDiffuseBxDF {
	T albedo_r, albedo_g, albedo_b;   // Lambertian base reflectance
	T coat_ior;                        // dielectric coat IOR (>1, e.g. 1.5)
	T alpha_x;                         // GGX roughness u-direction of the coat
	T alpha_y;                         // GGX roughness v-direction (equal = isotropic)
	T thickness = T(0.01);             // layer thickness (pbrt-v4 default 0.01)
	T g         = T(0);                // HG phase function asymmetry
	T medium_albedo = T(0);            // scattering albedo of the layer medium (0 = clear)
	int maxDepth  = 10;                // max random-walk interactions
	int nSamples  = 1;                 // samples per f()/pdf() estimate

	CPU_GPU layered_detail::DielectricLayer<T> top_layer() const { return {coat_ior, alpha_x, alpha_y}; }
	CPU_GPU layered_detail::DiffuseLayer<T> bottom_layer() const { return {{albedo_r, albedo_g, albedo_b}}; }
	CPU_GPU layered_detail::LayerParams<T> params() const {
		return {thickness < T(1.17549435e-38) ? T(1.17549435e-38) : thickness,
		        {medium_albedo, medium_albedo, medium_albedo}, g, maxDepth, nSamples};
	}

	// Never a delta BSDF: the Lambertian base always scatters diffusely. Whether the sampled path itself
	// was specular (a smooth coat's mirror reflection) is reported per sample in is_specular.
	CPU_GPU bool is_delta() const { return false; }
	CPU_GPU bool top_smooth() const { return top_layer().smooth(); }

	// pbrt's Sample_f: res.r/g/b = f*|cos|/pdf of the random walk, res.is_specular = specularPath.
	CPU_GPU BxDFSampleResult<T> sample_local(T wi_x, T wi_y, T wi_z, uint64_t seed0, uint64_t seed1 = 0) const {
		return layered_detail::to_result(layered_detail::layered_sample(
			top_layer(), bottom_layer(), params(), layered_detail::V3<T>{wi_x, wi_y, wi_z}, seed0, seed1));
	}

	// Backward-compat 5-float overload: derives a seed from the 5 input values
	CPU_GPU BxDFSampleResult<T> sample_local(T wi_x, T wi_y, T wi_z, T u1, T u2, T u3, T u4, T u5) const {
		uint32_t s0 = (uint32_t)(u1 * 16777216.0f), s1 = (uint32_t)(u2 * 16777216.0f), s2 = (uint32_t)(u3 * 16777216.0f);
		uint32_t s3 = (uint32_t)(u4 * 16777216.0f), s4 = (uint32_t)(u5 * 16777216.0f);
		uint64_t seed0 = ((uint64_t)s0 << 32) | (uint64_t)s1;
		uint64_t seed1 = ((uint64_t)s2 << 32) | ((uint64_t)s3 * 0x9e3779b97f4a7c15ULL + s4);
		return sample_local(wi_x, wi_y, wi_z, seed0, seed1);
	}

	// BSDF value (no cosine) for NEE: pbrt's stochastic LayeredBxDF::f().
	CPU_GPU void f(T wi_x, T wi_y, T wi_z, T wo_x, T wo_y, T wo_z, uint64_t seed0, uint64_t seed1,
	               T& fr, T& fg, T& fb) const {
		auto v = layered_detail::layered_f(top_layer(), bottom_layer(), params(),
			layered_detail::V3<T>{wi_x, wi_y, wi_z}, layered_detail::V3<T>{wo_x, wo_y, wo_z}, seed0, seed1);
		fr = v.r; fg = v.g; fb = v.b;
	}

	// pbrt's LayeredBxDF::PDF(): the approximate density used for MIS (pdfIsProportional).
	CPU_GPU T pdf(T wi_x, T wi_y, T wi_z, T wo_x, T wo_y, T wo_z) const {
		return layered_detail::layered_pdf(top_layer(), bottom_layer(), params(),
			layered_detail::V3<T>{wi_x, wi_y, wi_z}, layered_detail::V3<T>{wo_x, wo_y, wo_z});
	}
};

// ===========================================================================
// 9. CoatedConductorBxDF  (dielectric coat over GGX conductor)
//    pbrt-v4 CoatedConductorBxDF = LayeredBxDF<DielectricBxDF, ConductorBxDF, true>
//    cond_alpha_x/y < 0 means "same roughness as the coat" (the old single-alpha model);
//    pbrt's own default is a smooth coat over a smooth conductor.
// ===========================================================================
template<typename T>
struct CoatedConductorBxDF {
	T eta_r, eta_g, eta_b;   // conductor real IOR
	T k_r,   k_g,   k_b;    // conductor extinction
	T coat_ior;              // dielectric coat IOR
	T alpha_x;               // coat GGX roughness u-direction
	T alpha_y;               // coat GGX roughness v-direction (equal = isotropic)
	T thickness = T(0.01);
	T g         = T(0);
	T medium_albedo = T(0);
	int maxDepth  = 10;
	int nSamples  = 1;
	T cond_alpha_x = T(-1);  // conductor GGX roughness (negative = use the coat's)
	T cond_alpha_y = T(-1);

	CPU_GPU layered_detail::DielectricLayer<T> top_layer() const { return {coat_ior, alpha_x, alpha_y}; }
	CPU_GPU layered_detail::ConductorLayer<T> bottom_layer() const {
		return {{eta_r, eta_g, eta_b}, {k_r, k_g, k_b},
		        cond_alpha_x < T(0) ? alpha_x : cond_alpha_x, cond_alpha_y < T(0) ? alpha_y : cond_alpha_y};
	}
	CPU_GPU layered_detail::LayerParams<T> params() const {
		return {thickness < T(1.17549435e-38) ? T(1.17549435e-38) : thickness,
		        {medium_albedo, medium_albedo, medium_albedo}, g, maxDepth, nSamples};
	}

	// pbrt Flags(): no Diffuse/Glossy bit when both interfaces are smooth, so no NEE and no MIS.
	CPU_GPU bool is_delta() const { return top_layer().smooth() && bottom_layer().smooth() && medium_albedo == T(0); }
	CPU_GPU bool top_smooth() const { return top_layer().smooth(); }

	CPU_GPU BxDFSampleResult<T> sample_local(T wi_x, T wi_y, T wi_z, uint64_t seed0, uint64_t seed1 = 0) const {
		return layered_detail::to_result(layered_detail::layered_sample(
			top_layer(), bottom_layer(), params(), layered_detail::V3<T>{wi_x, wi_y, wi_z}, seed0, seed1));
	}

	CPU_GPU BxDFSampleResult<T> sample_local(T wi_x, T wi_y, T wi_z, T u1, T u2, T u3, T u4, T u5) const {
		uint32_t s0 = (uint32_t)(u1 * 16777216.0f), s1 = (uint32_t)(u2 * 16777216.0f), s2 = (uint32_t)(u3 * 16777216.0f);
		uint32_t s3 = (uint32_t)(u4 * 16777216.0f), s4 = (uint32_t)(u5 * 16777216.0f);
		uint64_t seed0 = ((uint64_t)s0 << 32) | (uint64_t)s1;
		uint64_t seed1 = ((uint64_t)s2 << 32) | ((uint64_t)s3 * 0x9e3779b97f4a7c15ULL + s4);
		return sample_local(wi_x, wi_y, wi_z, seed0, seed1);
	}

	CPU_GPU void f(T wi_x, T wi_y, T wi_z, T wo_x, T wo_y, T wo_z, uint64_t seed0, uint64_t seed1,
	               T& fr, T& fg, T& fb) const {
		auto v = layered_detail::layered_f(top_layer(), bottom_layer(), params(),
			layered_detail::V3<T>{wi_x, wi_y, wi_z}, layered_detail::V3<T>{wo_x, wo_y, wo_z}, seed0, seed1);
		fr = v.r; fg = v.g; fb = v.b;
	}

	CPU_GPU T pdf(T wi_x, T wi_y, T wi_z, T wo_x, T wo_y, T wo_z) const {
		return layered_detail::layered_pdf(top_layer(), bottom_layer(), params(),
			layered_detail::V3<T>{wi_x, wi_y, wi_z}, layered_detail::V3<T>{wo_x, wo_y, wo_z});
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
