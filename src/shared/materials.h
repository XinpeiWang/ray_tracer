// materials.h — pbrt-v4 material layer (simplified, header-only, CPU, templated)
// Mirrors pbrt-v4 src/pbrt/materials.h
//
// Ported materials:
//   1. DiffuseMaterial          — Lambertian diffuse (DiffuseBxDF)
//   2. DielectricMaterial       — Smooth/rough glass (DielectricBxDF)
//   3. ThinDielectricMaterial   — Thin-plate glass (ThinDielectricBxDF)
//   4. ConductorMaterial        — GGX metal (ConductorBxDF); also accepts
//                                 reflectance shorthand matching pbrt-v4
//   5. CoatedDiffuseMaterial    — Dielectric coat over diffuse (CoatedDiffuseBxDF)
//   6. CoatedConductorMaterial  — Dielectric coat over conductor (CoatedConductorBxDF)
//   7. DiffuseTransmissionMaterial — Cosine-weighted diffuse transmit (DiffuseTransmissionBxDF)
//   8. SubsurfaceMaterial       — Rough dielectric BSDF + TabulatedBSSRDF
//   9. HairMaterial             — HairBxDF with eumelanin/pheomelanin/reflectance sigma_a resolution
//  10. MixMaterial              — Stochastic blend of two materials by float texture weight
//
// Design notes:
//   - All materials are templated on scalar type T (typically float or double).
//   - Textures are passed as values satisfying the concept:
//       T eval_float(FloatTex, ctx)        — e.g. a T scalar or callable
//       Vec3<T> eval_spectrum(SpecTex, ctx) — RGB spectrum (r,g,b in [0,1])
//     Convenience: both are unified under TexEval below.
//   - MaterialEvalContext carries the geometric state needed by texture lookups
//     (mirrors pbrt-v4 MaterialEvalContext / TextureEvalContext).
//   - GetBxDF() returns the concrete BxDF by value; no virtual dispatch.
//   - Roughness remapping (RoughnessToAlpha) is opt-in per material.

#pragma once
#include <cmath>
#include <algorithm>
#include "bxdfs.h"
#include "bssrdf.h"
#include "pbrt_hash.h"

// ---------------------------------------------------------------------------
// MaterialEvalContext
// Carries all per-hit geometric state needed by textures and BxDF constructors.
// Mirrors pbrt-v4 MaterialEvalContext (which extends TextureEvalContext).
// ---------------------------------------------------------------------------
template <typename T>
struct MaterialEvalContext {
	// Position and shading geometry
	T px = T(0), py = T(0), pz = T(0);   // hit point
	T nx = T(0), ny = T(1), nz = T(0);   // shading normal (ns)
	T dpdu_x = T(1), dpdu_y = T(0), dpdu_z = T(0); // shading dpdu (for BSDF frame)
	T wo_x = T(0), wo_y = T(0), wo_z = T(1);        // outgoing direction (world)
	T u = T(0), v = T(0);               // UV coordinates
	// Texture differentials (for mipmap level selection — optional, zero = no filtering)
	T dudx = T(0), dudy = T(0), dvdx = T(0), dvdy = T(0);
	T dpdx_x = T(0), dpdx_y = T(0), dpdx_z = T(0);
	T dpdy_x = T(0), dpdy_y = T(0), dpdy_z = T(0);
	int faceIndex = 0;
};

// ---------------------------------------------------------------------------
// Simple texture value types used by the materials below.
// In practice the integrator supplies proper Texture objects; here we use
// plain scalar / RGB structs so the materials are self-contained for tests.
// ---------------------------------------------------------------------------

// FloatTexVal: a constant float texture (or any callable T(ctx) adapter)
template <typename T>
struct FloatTexVal {
	T value;
	explicit FloatTexVal(T v = T(0)) : value(v) {}
	T operator()(const MaterialEvalContext<T>&) const { return value; }
};

// SpectrumTexVal: a constant RGB spectrum texture
template <typename T>
struct SpectrumTexVal {
	T r, g, b;
	SpectrumTexVal(T r_ = T(0), T g_ = T(0), T b_ = T(0)) : r(r_), g(g_), b(b_) {}
	// Returns {r,g,b} as a plain struct; materials unpack individually
};

// ---------------------------------------------------------------------------
// Helper: evaluate a "float texture" — works for FloatTexVal<T> or any
// callable returning T when passed a const MaterialEvalContext<T>&.
// ---------------------------------------------------------------------------
template <typename T, typename FTex>
inline T eval_float_tex(const FTex& tex, const MaterialEvalContext<T>& ctx) {
	return tex(ctx);
}

// Specialisation for plain scalar (convenient in tests)
template <typename T>
inline T eval_float_tex(T v, const MaterialEvalContext<T>&) { return v; }

// ---------------------------------------------------------------------------
// Helper: evaluate a "spectrum texture" as three RGB channels.
// Returns r,g,b via output params.  Works for SpectrumTexVal<T>.
// ---------------------------------------------------------------------------
template <typename T>
inline void eval_spectrum_tex(const SpectrumTexVal<T>& tex,
							  const MaterialEvalContext<T>&,
							  T& r, T& g, T& b) {
	r = tex.r; g = tex.g; b = tex.b;
}

// Overload for a plain Vec3-like struct with .x/.y/.z
template <typename T, typename STex>
inline void eval_spectrum_tex(const STex& tex,
							  const MaterialEvalContext<T>& ctx,
							  T& r, T& g, T& b) {
	auto v = tex(ctx);
	r = v.x; g = v.y; b = v.z;
}

namespace {
template <typename T> inline T clamp01(T x) { return std::max(T(0), std::min(T(1), x)); }
template <typename T> inline T clamp_zero(T x) { return std::max(T(0), x); }
}

// ---------------------------------------------------------------------------
// 1. DiffuseMaterial
//    Maps a reflectance spectrum texture to DiffuseBxDF<T>.
//    Mirrors pbrt-v4 DiffuseMaterial::GetBxDF().
// ---------------------------------------------------------------------------
template <typename T, typename SpecTex = SpectrumTexVal<T>>
struct DiffuseMaterial {
	SpecTex reflectance;

	explicit DiffuseMaterial(SpecTex refl = SpecTex(T(0.5), T(0.5), T(0.5)))
		: reflectance(refl) {}

	DiffuseBxDF<T> get_bxdf(const MaterialEvalContext<T>& ctx) const {
		T r, g, b;
		eval_spectrum_tex(reflectance, ctx, r, g, b);
		// clamp to [0,1] matching pbrt-v4: Clamp(texEval(reflectance,...), 0, 1)
		return DiffuseBxDF<T>{clamp01(r), clamp01(g), clamp01(b)};
	}

	static constexpr bool has_subsurface_scattering() { return false; }
};

// ---------------------------------------------------------------------------
// DielectricBxDFVariant — holds either a smooth or rough dielectric BxDF.
// Mirrors pbrt-v4 DielectricMaterial which selects at runtime.
// ---------------------------------------------------------------------------
template <typename T>
struct DielectricBxDFVariant {
	bool rough;
	DielectricBxDF<T>      smooth; // valid when !rough
	RoughDielectricBxDF<T> roughd; // valid when  rough
};

// ---------------------------------------------------------------------------
// 2. DielectricMaterial
//    Smooth or rough dielectric (glass).
//    Returns DielectricBxDFVariant<T> — smooth when alpha~0, rough otherwise.
//    Mirrors pbrt-v4 DielectricMaterial::GetBxDF().
// ---------------------------------------------------------------------------
template <typename T, typename FTex = FloatTexVal<T>>
struct DielectricMaterial {
	T eta;               // index of refraction
	FTex u_roughness;
	FTex v_roughness;
	bool remap_roughness = true;

	DielectricMaterial(T eta_ = T(1.5),
					   FTex urough = FTex(T(0)),
					   FTex vrough = FTex(T(0)),
					   bool remap = true)
		: eta(eta_), u_roughness(urough), v_roughness(vrough), remap_roughness(remap) {}

	DielectricBxDFVariant<T> get_bxdf(const MaterialEvalContext<T>& ctx) const {
		T urough = eval_float_tex(u_roughness, ctx);
		T vrough = eval_float_tex(v_roughness, ctx);
		if (remap_roughness) {
			urough = TrowbridgeReitz<T>::RoughnessToAlpha(urough);
			vrough = TrowbridgeReitz<T>::RoughnessToAlpha(vrough);
		}
		DielectricBxDFVariant<T> v{};
		if (urough == T(0) && vrough == T(0)) {
			v.rough  = false;
			v.smooth = DielectricBxDF<T>{eta};
		} else {
			v.rough  = true;
			v.roughd = RoughDielectricBxDF<T>{eta, urough, vrough};
		}
		return v;
	}

	// Convenience: smooth-only accessor (asserts roughness == 0)
	DielectricBxDF<T> get_smooth_bxdf(const MaterialEvalContext<T>& ctx) const {
		return DielectricBxDF<T>{eta};
	}

	static constexpr bool has_subsurface_scattering() { return false; }
};

// ---------------------------------------------------------------------------
// 3. ThinDielectricMaterial
//    Thin parallel-slab glass.  Mirrors pbrt-v4 ThinDielectricMaterial.
// ---------------------------------------------------------------------------
template <typename T>
struct ThinDielectricMaterial {
	T eta;

	explicit ThinDielectricMaterial(T eta_ = T(1.5)) : eta(eta_) {}

	ThinDielectricBxDF<T> get_bxdf(const MaterialEvalContext<T>&) const {
		return ThinDielectricBxDF<T>{eta};
	}

	static constexpr bool has_subsurface_scattering() { return false; }
};

// ---------------------------------------------------------------------------
// 4. ConductorMaterial
//    GGX metallic surface.  Supports either (eta, k) spectral parameters or
//    the reflectance shorthand (matching pbrt-v4 ConductorMaterial).
// ---------------------------------------------------------------------------
template <typename T,
		  typename SpecTex  = SpectrumTexVal<T>,
		  typename FTex     = FloatTexVal<T>>
struct ConductorMaterial {
	// eta and k are per-channel (RGB) complex IOR.
	// If use_reflectance == true, reflectance_tex overrides eta/k and we
	// derive ks from reflectance per pbrt-v4's shorthand path.
	SpecTex eta_tex;
	SpecTex k_tex;
	SpecTex reflectance_tex;
	FTex u_roughness;
	FTex v_roughness;
	bool remap_roughness = true;
	bool use_reflectance = false;

	// Full (eta, k) constructor
	ConductorMaterial(SpecTex eta_, SpecTex k_,
					  FTex urough = FTex(T(0)), FTex vrough = FTex(T(0)),
					  bool remap = true)
		: eta_tex(eta_), k_tex(k_), reflectance_tex(SpecTex(T(0),T(0),T(0))),
		  u_roughness(urough), v_roughness(vrough),
		  remap_roughness(remap), use_reflectance(false) {}

	// Reflectance-shorthand constructor (matches pbrt-v4 fallback path)
	ConductorMaterial(SpecTex refl,
					  FTex urough = FTex(T(0)), FTex vrough = FTex(T(0)),
					  bool remap = true)
		: eta_tex(SpecTex(T(0),T(0),T(0))), k_tex(SpecTex(T(0),T(0),T(0))),
		  reflectance_tex(refl),
		  u_roughness(urough), v_roughness(vrough),
		  remap_roughness(remap), use_reflectance(true) {}

	ConductorBxDF<T> get_bxdf(const MaterialEvalContext<T>& ctx) const {
		T urough = eval_float_tex(u_roughness, ctx);
		T vrough = eval_float_tex(v_roughness, ctx);
		if (remap_roughness) {
			urough = TrowbridgeReitz<T>::RoughnessToAlpha(urough);
			vrough = TrowbridgeReitz<T>::RoughnessToAlpha(vrough);
		}

		T er, eg, eb, kr, kg, kb;
		if (use_reflectance) {
			// pbrt-v4: etas=1, ks = 2*sqrt(r)/sqrt(1-r)  clamped to [0,0.9999]
			T rr, rg, rb;
			eval_spectrum_tex(reflectance_tex, ctx, rr, rg, rb);
			rr = std::min(rr, T(0.9999)); rg = std::min(rg, T(0.9999)); rb = std::min(rb, T(0.9999));
			er = T(1); eg = T(1); eb = T(1);
			kr = T(2) * std::sqrt(rr) / std::sqrt(clamp_zero(T(1) - rr));
			kg = T(2) * std::sqrt(rg) / std::sqrt(clamp_zero(T(1) - rg));
			kb = T(2) * std::sqrt(rb) / std::sqrt(clamp_zero(T(1) - rb));
		} else {
			eval_spectrum_tex(eta_tex, ctx, er, eg, eb);
			eval_spectrum_tex(k_tex,   ctx, kr, kg, kb);
		}
		return ConductorBxDF<T>{er, eg, eb, kr, kg, kb, urough, vrough};
	}

	static constexpr bool has_subsurface_scattering() { return false; }
};

// ---------------------------------------------------------------------------
// 5. CoatedDiffuseMaterial
//    Rough dielectric coat over Lambertian base.
//    Mirrors pbrt-v4 CoatedDiffuseMaterial.
// ---------------------------------------------------------------------------
template <typename T,
		  typename SpecTex = SpectrumTexVal<T>,
		  typename FTex    = FloatTexVal<T>>
struct CoatedDiffuseMaterial {
	SpecTex reflectance;
	FTex u_roughness;
	FTex v_roughness;
	FTex thickness;
	SpecTex albedo;
	FTex g;
	T eta;
	bool remap_roughness = true;
	int max_depth   = 10;
	int n_samples   = 1;

	CoatedDiffuseMaterial(SpecTex refl,
						  FTex urough = FTex(T(0)), FTex vrough = FTex(T(0)),
						  FTex thick  = FTex(T(0.01)),
						  SpecTex alb = SpecTex(T(0), T(0), T(0)),
						  FTex g_     = FTex(T(0)),
						  T eta_      = T(1.5),
						  bool remap  = true,
						  int maxd    = 10,
						  int ns      = 1)
		: reflectance(refl), u_roughness(urough), v_roughness(vrough),
		  thickness(thick), albedo(alb), g(g_), eta(eta_),
		  remap_roughness(remap), max_depth(maxd), n_samples(ns) {}

	CoatedDiffuseBxDF<T> get_bxdf(const MaterialEvalContext<T>& ctx) const {
		T rr, rg, rb;
		eval_spectrum_tex(reflectance, ctx, rr, rg, rb);
		T ar, ag, ab;
		eval_spectrum_tex(albedo, ctx, ar, ag, ab);

		T urough = eval_float_tex(u_roughness, ctx);
		T vrough = eval_float_tex(v_roughness, ctx);
		if (remap_roughness) {
			urough = TrowbridgeReitz<T>::RoughnessToAlpha(urough);
			vrough = TrowbridgeReitz<T>::RoughnessToAlpha(vrough);
		}
		T thick = eval_float_tex(thickness, ctx);
		// pbrt-v4: Clamp(texEval(g, ctx), -1, 1)
		T g_val = std::max(T(-1), std::min(T(1), eval_float_tex(g, ctx)));

		// CoatedDiffuseBxDF fields: albedo_r/g/b, coat_ior, alpha_x, alpha_y,
		// thickness, g, medium_albedo, maxDepth, nSamples
		T med_alb = (clamp01(ar) + clamp01(ag) + clamp01(ab)) / T(3);
		return CoatedDiffuseBxDF<T>{
			clamp01(rr), clamp01(rg), clamp01(rb),
			eta, urough, vrough, thick,
			g_val, med_alb, max_depth, n_samples};
	}

	static constexpr bool has_subsurface_scattering() { return false; }
};

// ---------------------------------------------------------------------------
// 6. CoatedConductorMaterial
//    Rough dielectric coat over GGX conductor.
//    Mirrors pbrt-v4 CoatedConductorMaterial.
// ---------------------------------------------------------------------------
template <typename T,
		  typename SpecTex = SpectrumTexVal<T>,
		  typename FTex    = FloatTexVal<T>>
struct CoatedConductorMaterial {
	SpecTex conductor_eta;
	SpecTex conductor_k;
	FTex interface_u_roughness;
	FTex interface_v_roughness;
	FTex conductor_u_roughness;
	FTex conductor_v_roughness;
	FTex thickness;
	SpecTex albedo;
	FTex g;
	T interface_eta;
	bool remap_roughness = true;
	int max_depth  = 10;
	int n_samples  = 1;

	CoatedConductorMaterial(SpecTex ceta, SpecTex ck,
							FTex iurough = FTex(T(0)), FTex ivrough = FTex(T(0)),
							FTex curough = FTex(T(0)), FTex cvrough = FTex(T(0)),
							FTex thick   = FTex(T(0.01)),
							SpecTex alb  = SpecTex(T(0),T(0),T(0)),
							FTex g_      = FTex(T(0)),
							T ieta       = T(1.5),
							bool remap   = true,
							int maxd     = 10,
							int ns       = 1)
		: conductor_eta(ceta), conductor_k(ck),
		  interface_u_roughness(iurough), interface_v_roughness(ivrough),
		  conductor_u_roughness(curough), conductor_v_roughness(cvrough),
		  thickness(thick), albedo(alb), g(g_), interface_eta(ieta),
		  remap_roughness(remap), max_depth(maxd), n_samples(ns) {}

	CoatedConductorBxDF<T> get_bxdf(const MaterialEvalContext<T>& ctx) const {
		T er, eg, eb, kr, kg, kb;
		eval_spectrum_tex(conductor_eta, ctx, er, eg, eb);
		eval_spectrum_tex(conductor_k,   ctx, kr, kg, kb);
		T ar, ag, ab;
		eval_spectrum_tex(albedo, ctx, ar, ag, ab);

		T iurough = eval_float_tex(interface_u_roughness, ctx);
		T ivrough = eval_float_tex(interface_v_roughness, ctx);
		T curough = eval_float_tex(conductor_u_roughness, ctx);
		T cvrough = eval_float_tex(conductor_v_roughness, ctx);
		if (remap_roughness) {
			iurough = TrowbridgeReitz<T>::RoughnessToAlpha(iurough);
			ivrough = TrowbridgeReitz<T>::RoughnessToAlpha(ivrough);
			curough = TrowbridgeReitz<T>::RoughnessToAlpha(curough);
			cvrough = TrowbridgeReitz<T>::RoughnessToAlpha(cvrough);
		}
		T thick = eval_float_tex(thickness, ctx);
		// pbrt-v4: Clamp(texEval(g, ctx), -1, 1)
		T g_val = std::max(T(-1), std::min(T(1), eval_float_tex(g, ctx)));

		// pbrt-v4: ce /= ieta; ck /= ieta  (normalize conductor IOR by coat IOR)
		T ieta = interface_eta;
		if (ieta != T(0)) { er /= ieta; eg /= ieta; eb /= ieta;
							 kr /= ieta; kg /= ieta; kb /= ieta; }

		// CoatedConductorBxDF fields: eta_r/g/b, k_r/g/b, coat_ior,
		// alpha_x, alpha_y, thickness, g, medium_albedo, maxDepth, nSamples
		T med_alb = (clamp01(ar) + clamp01(ag) + clamp01(ab)) / T(3);
		return CoatedConductorBxDF<T>{
			er, eg, eb, kr, kg, kb,
			ieta,
			iurough, ivrough,   // coat interface roughness (alpha_x, alpha_y)
			thick, g_val, med_alb, max_depth, n_samples};
	}

	static constexpr bool has_subsurface_scattering() { return false; }
};

// ---------------------------------------------------------------------------
// 7. DiffuseTransmissionMaterial
//    Cosine-weighted diffuse reflection + transmission.
//    Mirrors pbrt-v4 DiffuseTransmissionMaterial.
// ---------------------------------------------------------------------------
template <typename T,
		  typename SpecTex = SpectrumTexVal<T>,
		  typename FTex    = FloatTexVal<T>>
struct DiffuseTransmissionMaterial {
	SpecTex reflectance;
	SpecTex transmittance;
	FTex scale_tex;

	DiffuseTransmissionMaterial(SpecTex refl,
								 SpecTex trans,
								 FTex scale = FTex(T(1)))
		: reflectance(refl), transmittance(trans), scale_tex(scale) {}

	DiffuseTransmissionBxDF<T> get_bxdf(const MaterialEvalContext<T>& ctx) const {
		T rr, rg, rb;
		eval_spectrum_tex(reflectance, ctx, rr, rg, rb);
		T tr, tg, tb;
		eval_spectrum_tex(transmittance, ctx, tr, tg, tb);
		T s = eval_float_tex(scale_tex, ctx);

		return DiffuseTransmissionBxDF<T>{
			clamp01(rr) * s, clamp01(rg) * s, clamp01(rb) * s,
			clamp01(tr) * s, clamp01(tg) * s, clamp01(tb) * s};
	}

	static constexpr bool has_subsurface_scattering() { return false; }
};

// ---------------------------------------------------------------------------
// 8. SubsurfaceMaterial
//    Rough dielectric BSDF surface + TabulatedBSSRDF for subsurface transport.
//    Mirrors pbrt-v4 SubsurfaceMaterial.
//
//    The BxDF is DielectricBxDF<T> (same as pbrt-v4).
//    The BSSRDF uses the local bssrdf.h types (double-precision, BSSRDFTable,
//    TabulatedBSSRDF(sigma_a*, sigma_s*, table*, n_ch)).
// ---------------------------------------------------------------------------
template <typename T,
		  typename SpecTex = SpectrumTexVal<T>,
		  typename FTex    = FloatTexVal<T>>
struct SubsurfaceMaterial {
	T scale;
	SpecTex sigma_a_tex;
	SpecTex sigma_s_tex;
	bool use_sigma;          // true = sigma_a/sigma_s mode; false = reflectance+mfp
	SpecTex reflectance_tex;
	SpecTex mfp_tex;
	FTex u_roughness;
	FTex v_roughness;
	T eta;
	bool remap_roughness = true;
	BSSRDFTable table;       // pre-computed beam diffusion table (double, non-templated)

	// sigma_a / sigma_s constructor
	SubsurfaceMaterial(T scale_,
					   SpecTex sig_a, SpecTex sig_s,
					   double g, double eta_,
					   FTex urough = FTex(T(0)), FTex vrough = FTex(T(0)),
					   bool remap = true,
					   int table_rho_samples = 100,
					   int table_radius_samples = 64)
		: scale(scale_), sigma_a_tex(sig_a), sigma_s_tex(sig_s),
		  use_sigma(true),
		  reflectance_tex(SpecTex(T(0),T(0),T(0))), mfp_tex(SpecTex(T(0),T(0),T(0))),
		  u_roughness(urough), v_roughness(vrough), eta(static_cast<T>(eta_)),
		  remap_roughness(remap),
		  table(table_rho_samples, table_radius_samples) {
		ComputeBeamDiffusionBSSRDF(g, eta_, &table);
	}

	// reflectance + mfp constructor (tag overload to disambiguate)
	struct ReflMfpTag {};
	SubsurfaceMaterial(T scale_,
					   SpecTex refl, SpecTex mfp,
					   double g, double eta_,
					   FTex urough, FTex vrough,
					   bool remap,
					   ReflMfpTag,
					   int table_rho_samples = 100,
					   int table_radius_samples = 64)
		: scale(scale_),
		  sigma_a_tex(SpecTex(T(0),T(0),T(0))), sigma_s_tex(SpecTex(T(0),T(0),T(0))),
		  use_sigma(false),
		  reflectance_tex(refl), mfp_tex(mfp),
		  u_roughness(urough), v_roughness(vrough), eta(static_cast<T>(eta_)),
		  remap_roughness(remap),
		  table(table_rho_samples, table_radius_samples) {
		ComputeBeamDiffusionBSSRDF(g, eta_, &table);
	}

	DielectricBxDFVariant<T> get_bxdf(const MaterialEvalContext<T>& ctx) const {
		T urough = eval_float_tex(u_roughness, ctx);
		T vrough = eval_float_tex(v_roughness, ctx);
		if (remap_roughness) {
			urough = TrowbridgeReitz<T>::RoughnessToAlpha(urough);
			vrough = TrowbridgeReitz<T>::RoughnessToAlpha(vrough);
		}
		DielectricBxDFVariant<T> v{};
		if (urough == T(0) && vrough == T(0)) {
			v.rough  = false;
			v.smooth = DielectricBxDF<T>{eta};
		} else {
			v.rough  = true;
			v.roughd = RoughDielectricBxDF<T>{eta, urough, vrough};
		}
		return v;
	}

	// Build per-channel sigma_a/sigma_s arrays and return a TabulatedBSSRDF.
	// n_ch = 3 for RGB; the caller owns the lifetime of the returned object
	// (which holds a pointer to this->table).
	TabulatedBSSRDF get_bssrdf(const MaterialEvalContext<T>& ctx) const {
		double sig_a[3], sig_s[3];

		if (use_sigma) {
			T ar, ag, ab, sr, sg, sb;
			eval_spectrum_tex(sigma_a_tex, ctx, ar, ag, ab);
			eval_spectrum_tex(sigma_s_tex, ctx, sr, sg, sb);
			sig_a[0] = clamp_zero(static_cast<double>(scale * ar));
			sig_a[1] = clamp_zero(static_cast<double>(scale * ag));
			sig_a[2] = clamp_zero(static_cast<double>(scale * ab));
			sig_s[0] = clamp_zero(static_cast<double>(scale * sr));
			sig_s[1] = clamp_zero(static_cast<double>(scale * sg));
			sig_s[2] = clamp_zero(static_cast<double>(scale * sb));
		} else {
			T rr, rg, rb, mr, mg, mb;
			eval_spectrum_tex(reflectance_tex, ctx, rr, rg, rb);
			eval_spectrum_tex(mfp_tex,         ctx, mr, mg, mb);
			double mfp[3] = {
				clamp_zero(static_cast<double>(scale * mr)),
				clamp_zero(static_cast<double>(scale * mg)),
				clamp_zero(static_cast<double>(scale * mb))
			};
			double refl[3] = {
				static_cast<double>(clamp01(rr)),
				static_cast<double>(clamp01(rg)),
				static_cast<double>(clamp01(rb))
			};
			for (int i = 0; i < 3; ++i)
				SubsurfaceFromDiffuse(table, refl[i], mfp[i], sig_a[i], sig_s[i]);
		}

		return TabulatedBSSRDF(sig_a, sig_s, &table, 3);
	}

	static constexpr bool has_subsurface_scattering() { return true; }
};

// ---------------------------------------------------------------------------
// 9. HairMaterial
//    Resolves sigma_a (absorption) from one of three sources and builds a
//    HairBxDF<T>.  Mirrors pbrt-v4 HairMaterial::GetBxDF().
//
//    sigma_a priority (first non-null/non-zero wins):
//      1. sigma_a_tex    -- direct absorption spectrum [m^-1]
//      2. color_tex      -- reflectance color; converted via SigmaAFromReflectance
//      3. eumelanin/pheomelanin concentrations -- pigment model
//
//    beta_m and beta_n are clamped to [0.01, 1] as in pbrt-v4.
//    h (fiber cross-section offset) defaults to 0 (uniform sampling in use).
// ---------------------------------------------------------------------------
template <typename T,
		  typename SpecTex = SpectrumTexVal<T>,
		  typename FTex    = FloatTexVal<T>>
struct HairMaterial {
	// Absorption / color input (only one should be active)
	SpecTex sigma_a_tex;          // direct sigma_a (RGB, m^-1)
	SpecTex color_tex;            // reflectance color [0,1]
	FTex    eumelanin;            // brown pigment concentration [0,inf)
	FTex    pheomelanin;          // red pigment concentration [0,inf)
	bool    has_sigma_a  = false; // use sigma_a_tex
	bool    has_color    = false; // use color_tex
	bool    has_pigment  = false; // use eumelanin/pheomelanin

	// Shape parameters
	FTex eta     = FTex(T(1.55));
	FTex beta_m  = FTex(T(0.3));
	FTex beta_n  = FTex(T(0.3));
	FTex alpha   = FTex(T(2.0));  // scale tilt in degrees
	// Note: h (cross-section offset) is computed from ctx.v at evaluation
	// time: h = -1 + 2*v, mirroring pbrt-v4 HairMaterial::GetBxDF().

	// -----------------------------------------------------------------------
	// SigmaAFromConcentration (pbrt-v4 bxdfs.cpp HairBxDF::SigmaAFromConcentration)
	// Returns sigma_a RGB given eumelanin ce and pheomelanin cp concentrations.
	// -----------------------------------------------------------------------
	static void sigma_a_from_concentration(T ce, T cp,
										   T& sr, T& sg, T& sb) {
		// Measured absorption coefficients (pbrt-v4, table from d'Eon et al.)
		const T eumR = T(0.419), eumG = T(0.697), eumB = T(1.37);
		const T pheoR = T(0.187), pheoG = T(0.4),  pheoB = T(1.05);
		sr = ce * eumR + cp * pheoR;
		sg = ce * eumG + cp * pheoG;
		sb = ce * eumB + cp * pheoB;
	}

	// -----------------------------------------------------------------------
	// SigmaAFromReflectance (pbrt-v4 bxdfs.cpp HairBxDF::SigmaAFromReflectance)
	// Returns sigma_a for one channel given reflectance c and azimuthal roughness bn.
	// -----------------------------------------------------------------------
	static T sigma_a_from_reflectance_channel(T c, T bn) {
		// Clamp to avoid log(0)
		c = std::max(T(1e-4), std::min(T(1) - T(1e-4), c));
		T bn2 = bn * bn; T bn3 = bn2 * bn; T bn4 = bn3 * bn; T bn5 = bn4 * bn;
		T denom = T(5.969) - T(0.215)*bn + T(2.532)*bn2
				- T(10.73)*bn3 + T(5.574)*bn4 + T(0.245)*bn5;
		T x = std::log(c) / denom;
		return x * x;
	}

	HairBxDF<T> get_bxdf(const MaterialEvalContext<T>& ctx) const {
		T bm = clamp01(eval_float_tex(beta_m,     ctx));
		T bn = clamp01(eval_float_tex(beta_n,     ctx));
		bm = std::max(bm, T(0.01));
		bn = std::max(bn, T(0.01));
		T a   = eval_float_tex(alpha,   ctx);
		T e   = eval_float_tex(eta,     ctx);

		T sr, sg, sb;
		if (has_sigma_a) {
			eval_spectrum_tex(sigma_a_tex, ctx, sr, sg, sb);
			sr = clamp_zero(sr); sg = clamp_zero(sg); sb = clamp_zero(sb);
		} else if (has_color) {
			T cr, cg, cb;
			eval_spectrum_tex(color_tex, ctx, cr, cg, cb);
			cr = clamp01(cr); cg = clamp01(cg); cb = clamp01(cb);
			sr = sigma_a_from_reflectance_channel(cr, bn);
			sg = sigma_a_from_reflectance_channel(cg, bn);
			sb = sigma_a_from_reflectance_channel(cb, bn);
		} else if (has_pigment) {
			T ce = std::max(T(0), eval_float_tex(eumelanin,   ctx));
			T cp = std::max(T(0), eval_float_tex(pheomelanin, ctx));
			sigma_a_from_concentration(ce, cp, sr, sg, sb);
		} else {
			// Default: black hair (no absorption)
			sr = sg = sb = T(0);
		}

		// Offset along fiber width: pbrt-v4 h = -1 + 2 * ctx.uv[1] (v coord)
		T h = T(-1) + T(2) * ctx.v;
		return HairBxDF<T>(h, e, sr, sg, sb, bm, bn, a);
	}

	static constexpr bool has_subsurface_scattering() { return false; }
};

// ---------------------------------------------------------------------------
// 10. MixMaterial
//     Stochastically selects one of two child materials based on a float
//     texture weight and a deterministic hash of the shading point.
//     Mirrors pbrt-v4 MixMaterial::ChooseMaterial().
//
//     The hash approach means: at any fixed point, the same material is always
//     chosen (no noise), but the boundary between the two materials is a
//     sharp surface at weight=u rather than a smooth blend.
//
//     MaterialA is chosen when hash < amount; MaterialB otherwise.
//     amount=0 -> always B, amount=1 -> always A.
//
//     Note: MixMaterial does not expose get_bxdf() directly — the caller
//     must call choose() to get an index (0=A, 1=B) and then evaluate the
//     chosen child material.  This matches pbrt-v4's design.
// ---------------------------------------------------------------------------
template <typename T, typename FTex = FloatTexVal<T>>
struct MixMaterial {
	FTex amount;   // mixing weight in [0,1]: fraction going to material A

	// Simple deterministic hash from pbrt-v4 MixMaterial::ChooseMaterial.
	// Returns 0 to select material A, 1 to select material B.
	//
	// pbrt-v4: u = HashFloat(p, wo); return (amt < u) ? mat[0] : mat[1]
	//   -> amt=0: almost always mat[0]; amt=1: always mat[1].
	// Our mapping: index 0 = mat[0] (chosen when amount is low), 1 = mat[1].
	int choose(const MaterialEvalContext<T>& ctx) const {
		T w = clamp01(eval_float_tex(amount, ctx));
		// Hash position + outgoing direction, matching pbrt-v4 HashFloat(ctx.p, ctx.wo)
		float u = HashFloat(ctx.px, ctx.py, ctx.pz,
							ctx.wo_x, ctx.wo_y, ctx.wo_z);
		// pbrt-v4: (amt < u) ? mat[0] : mat[1]
		// w < u  -> index 0 (low weight selects A = mat[0])
		// w >= u -> index 1
		return (static_cast<float>(w) < u) ? 0 : 1;
	}

	static constexpr bool has_subsurface_scattering() { return false; }
};
