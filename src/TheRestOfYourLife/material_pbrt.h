#pragma once
#include "material_base.h"
#include "../shared/bssrdf.h"
#include "../shared/measured_bxdf_loader.h"
#include <cstring>

// coated_seed_from_dir -- deterministic 64-bit hash of a direction vector,
// used by coated_diffuse/coated_conductor's real-NEE path (below) to derive
// the CoatedDiffuseBxDF<T>::f() / CoatedConductorBxDF<T>::f() random-walk
// seed. scattering_pdf() and scattering_attenuation() are two SEPARATE
// calls for the same (rec, scattered) query (see material_base.h's own
// comment on why NEE needs both), and f()'s random walk must give the SAME
// per-channel/scalar split across both calls for a given queried direction
// -- a fresh random_double()-derived seed each call would let the two calls
// silently disagree (extra noise, not bias, but avoidable). Hashing only
// the queried direction (not rec.p) is sufficient: both calls are always
// for the same scatter() event's fixed wi/rec, only `scattered` varies
// across the different NEE/MIS strategies querying this material.
inline uint64_t coated_seed_from_dir(const vec3& d) {
	uint64_t h = 0x9E3779B97F4A7C15ull;
	auto mix = [&](double v) {
		uint64_t bits;
		std::memcpy(&bits, &v, sizeof(bits));
		h ^= bits + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
	};
	mix(d.x()); mix(d.y()); mix(d.z());
	return h;
}

// pbrt-v4's "remaproughness" (materials.cpp's GetOneBool("remaproughness",
// true), read by ConductorMaterial/DielectricMaterial/CoatedDiffuseMaterial/
// CoatedConductorMaterial alike): when true (pbrt-v4's own default), an
// authored roughness value is squeezed through RoughnessToAlpha (roughly
// perceptual, matches how older/most pbrt scenes were authored); when
// false, the value IS the GGX alpha directly, unconverted - a scene using
// low, precise alpha values (e.g. this project's own ganesha.pbrt,
// "remaproughness" false + "uroughness"/"vroughness" 0.01) needs this to
// render as intended instead of unconditionally sqrt()'d into 10x the
// roughness it asked for.
inline double roughness_or_alpha(double roughness, bool remap_roughness) {
	const double clamped = std::fmax(roughness, 1e-4);
	return remap_roughness ? TrowbridgeReitz<double>::RoughnessToAlpha(clamped) : clamped;
}

// ---------------------------------------------------------------------------
// rough_metal -- GGX microfacet BRDF (pbrt-v4 TrowbridgeReitzDistribution)
// Physically-based rough conductor; replaces simple fuzz-sphere metal for
// accurate anisotropic highlights and energy conservation.
// roughness in [0,1]: 0 = mirror, 1 = fully diffuse-like rough
// ---------------------------------------------------------------------------

class rough_metal : public material {
  public:
    using BxDF = RoughMetalBxDF<double>;

    rough_metal(const color& albedo, double roughness)
        : albedo(albedo) {
        double a = TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(roughness, 1e-4));
        alpha_x = alpha_y = a;
    }

    rough_metal(const color& albedo, double u_roughness, double v_roughness)
        : albedo(albedo),
          alpha_x(TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(u_roughness, 1e-4))),
          alpha_y(TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(v_roughness, 1e-4))) {}

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        return BxDF{ albedo.x(), albedo.y(), albedo.z(), alpha_x, alpha_y };
    }

    // Real NEE/MIS below the roughness threshold RoughMetalBxDF::
    // effectively_smooth() considers "glossy" (pbrt-v4's TrowbridgeReitz::
    // EffectivelySmooth, alpha >= 1e-3): albedo is a flat, direction-
    // independent tint (RoughMetalBxDF has no real Fresnel model), so it can
    // stay a fixed srec.attenuation exactly like lambertian's Kd -- no
    // scattering_attenuation() override needed, unlike conductor's real
    // per-direction complex Fresnel below.
    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        auto ctx  = MaterialContext<double>::from_hit(rec, r_in);
        double ex = do_regularize ? regularize_alpha(alpha_x) : alpha_x;
        double ey = do_regularize ? regularize_alpha(alpha_y) : alpha_y;
        BxDF bxdf{ albedo.x(), albedo.y(), albedo.z(), ex, ey };
        auto frame = ShadingFrame<double>::from_dpdu(ctx.dpdu_x, ctx.dpdu_y, ctx.dpdu_z, ctx.nx, ctx.ny, ctx.nz);

        double wi_x, wi_y, wi_z;
        frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);

        // Smooth/glossy branch on the TRUE (unregularized) roughness, not
        // ex/ey: scattering_pdf() below has no do_regularize parameter to
        // work with (material::scattering_pdf()'s virtual signature is
        // fixed), so it can only ever use the raw alpha_x/alpha_y members.
        // Branching on ex/ey here would let do_regularize push a truly-
        // specular material into this glossy path while scattering_pdf()
        // still evaluated the ORIGINAL near-delta alpha -- an inconsistent,
        // nearly-degenerate pdf. Branching on the true alpha instead leaves
        // do_regularize's existing effect exactly as it was: it only widens
        // the alpha actually used inside the smooth branch's sample_local()
        // call below, same as before this NEE feature existed.
        if (effectively_smooth()) {
            auto res = bxdf.sample_local(wi_x, wi_y, wi_z, random_double(), random_double());
            if (!res.valid) return false;

            double wd_x, wd_y, wd_z;
            frame.to_world(res.wo_x, res.wo_y, res.wo_z, wd_x, wd_y, wd_z);
            srec.attenuation  = color(res.r, res.g, res.b);
            srec.pdf_ptr      = nullptr;
            srec.skip_pdf     = true;
            srec.skip_pdf_ray = ray(rec.p, unit_vector(vec3(wd_x, wd_y, wd_z)), r_in.time());
            return true;
        }

        // Glossy: srec.attenuation is the material's constant color; the
        // VNDF pdf_ptr drives both real NEE (Strategy A) and the BSDF-
        // sampled continuation (Strategy B) in camera.h, with
        // scattering_pdf() below supplying the matching f*cos shape.
        srec.attenuation = albedo;
        srec.pdf_ptr      = make_shared<ggx_reflection_pdf>(
            rec.normal, vec3(ctx.wo_x, ctx.wo_y, ctx.wo_z), alpha_x, alpha_y);
        srec.skip_pdf     = false;
        return true;
    }

    // f*cos at an arbitrary queried direction (scattered), for real NEE/MIS.
    // Pure geometric GGX shape -- the constant albedo tint is supplied
    // separately via srec.attenuation (default scattering_attenuation()).
    double scattering_pdf(const ray& r_in, const hit_record& rec,
                          const ray& scattered) const override {
        auto ctx   = MaterialContext<double>::from_hit(rec, r_in);
        auto frame = ShadingFrame<double>::from_dpdu(ctx.dpdu_x, ctx.dpdu_y, ctx.dpdu_z, ctx.nx, ctx.ny, ctx.nz);
        double wi_x, wi_y, wi_z;
        frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);
        vec3 dir = unit_vector(scattered.direction());
        double wo_x, wo_y, wo_z;
        frame.to_local(dir.x(), dir.y(), dir.z(), wo_x, wo_y, wo_z);
        double shape = ggx_reflection_shape(wi_x, wi_y, wi_z, wo_x, wo_y, wo_z, alpha_x, alpha_y);
        return shape * wo_z;
    }

    double get_roughness() const { return alpha_x * alpha_x; }
    const color& get_albedo() const { return albedo; }

    // Delegates to the single effectively_smooth() helper below, the same
    // one scatter()'s own branch consults - see material::is_delta_bsdf()'s
    // own comment on why these two must never be allowed to drift apart
    // again (they did once, for this exact class - see this class's own
    // #222 comment further up).
    bool is_delta_bsdf() const override { return effectively_smooth(); }

  private:
    // The ONE place this class decides smooth-vs-glossy - both scatter()'s
    // branch and is_delta_bsdf() above consult this instead of each
    // independently re-writing TrowbridgeReitz<double>(alpha_x,
    // alpha_y).EffectivelySmooth(), so there is no longer a second copy of
    // this condition that could silently go stale if the roughness logic
    // ever changes.
    bool effectively_smooth() const { return TrowbridgeReitz<double>(alpha_x, alpha_y).EffectivelySmooth(); }

    color  albedo;
    double alpha_x, alpha_y;
};


// ---------------------------------------------------------------------------
// conductor -- GGX microfacet BRDF with complex Fresnel (pbrt-v4 ConductorBxDF)
// Uses real metal optical constants (η + i·k) per RGB channel for physically
// accurate angle-varying, wavelength-dependent reflectance.
// roughness in [0,1]: 0 = mirror, 1 = fully rough
// ---------------------------------------------------------------------------
class conductor : public material {
  public:
    using BxDF = ConductorBxDF<double>;

    conductor(double eta_r, double eta_g, double eta_b,
              double k_r,   double k_g,   double k_b,
              double roughness)
        : eta_r(eta_r), eta_g(eta_g), eta_b(eta_b),
          k_r(k_r),     k_g(k_g),     k_b(k_b) {
        double a = TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(roughness, 1e-4));
        alpha_x = alpha_y = a;
    }

    conductor(double eta_r, double eta_g, double eta_b,
              double k_r,   double k_g,   double k_b,
              double u_roughness, double v_roughness, bool remap_roughness = true)
        : eta_r(eta_r), eta_g(eta_g), eta_b(eta_b),
          k_r(k_r),     k_g(k_g),     k_b(k_b),
          alpha_x(roughness_or_alpha(u_roughness, remap_roughness)),
          alpha_y(roughness_or_alpha(v_roughness, remap_roughness)) {}

    conductor(const ConductorPreset& preset, double roughness)
        : eta_r(preset.eta_r), eta_g(preset.eta_g), eta_b(preset.eta_b),
          k_r(preset.k_r),     k_g(preset.k_g),     k_b(preset.k_b) {
        double a = TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(roughness, 1e-4));
        alpha_x = alpha_y = a;
    }

    conductor(const ConductorPreset& preset, double u_roughness, double v_roughness)
        : eta_r(preset.eta_r), eta_g(preset.eta_g), eta_b(preset.eta_b),
          k_r(preset.k_r),     k_g(preset.k_g),     k_b(preset.k_b),
          alpha_x(TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(u_roughness, 1e-4))),
          alpha_y(TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(v_roughness, 1e-4))) {}

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        return BxDF{ eta_r, eta_g, eta_b, k_r, k_g, k_b, alpha_x, alpha_y };
    }

    // Real NEE/MIS below the roughness threshold (see rough_metal's own
    // comment). Unlike rough_metal's flat albedo, real per-channel complex
    // Fresnel varies with the queried direction's half-vector angle -- but
    // camera.h's Strategy B (BSDF-sampled continuation) always multiplies
    // by the FIXED srec.attenuation set here, with no per-direction hook
    // (unlike the NEE strategies, which do call scattering_attenuation()).
    // srec.attenuation is therefore only the Fresnel at the view-normal
    // cosine; camera.h now asks scattering_attenuation() (overridden below)
    // for the half-vector Fresnel of the direction actually being evaluated,
    // in the NEE strategies and in the BSDF-sampled continuation alike.
    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        auto ctx   = MaterialContext<double>::from_hit(rec, r_in);
        double ex = do_regularize ? regularize_alpha(alpha_x) : alpha_x;
        double ey = do_regularize ? regularize_alpha(alpha_y) : alpha_y;
        BxDF bxdf{ eta_r, eta_g, eta_b, k_r, k_g, k_b, ex, ey };
        auto frame = ShadingFrame<double>::from_dpdu(ctx.dpdu_x, ctx.dpdu_y, ctx.dpdu_z, ctx.nx, ctx.ny, ctx.nz);

        double wi_x, wi_y, wi_z;
        frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);

        // Branch on the TRUE (unregularized) roughness -- see rough_metal's
        // own comment on why scattering_pdf() forces this.
        if (effectively_smooth()) {
            auto res = bxdf.sample_local(wi_x, wi_y, wi_z, random_double(), random_double());
            if (!res.valid) return false;

            double wd_x, wd_y, wd_z;
            frame.to_world(res.wo_x, res.wo_y, res.wo_z, wd_x, wd_y, wd_z);
            srec.attenuation  = color(res.r, res.g, res.b);
            srec.pdf_ptr      = nullptr;
            srec.skip_pdf     = true;
            srec.skip_pdf_ray = ray(rec.p, unit_vector(vec3(wd_x, wd_y, wd_z)), r_in.time());
            return true;
        }

        double fr_view = std::max(0.0, wi_z);
        srec.attenuation = color(FrComplex(fr_view, eta_r, k_r),
                                  FrComplex(fr_view, eta_g, k_g),
                                  FrComplex(fr_view, eta_b, k_b));
        srec.pdf_ptr      = make_shared<ggx_reflection_pdf>(
            rec.normal, vec3(ctx.wo_x, ctx.wo_y, ctx.wo_z), alpha_x, alpha_y);
        srec.skip_pdf     = false;
        return true;
    }

    // f*cos at an arbitrary queried direction (scattered), for real NEE/MIS.
    // Pure geometric GGX shape -- the (approximated, view-angle) Fresnel
    // color is supplied separately via srec.attenuation.
    double scattering_pdf(const ray& r_in, const hit_record& rec,
                          const ray& scattered) const override {
        auto ctx   = MaterialContext<double>::from_hit(rec, r_in);
        auto frame = ShadingFrame<double>::from_dpdu(ctx.dpdu_x, ctx.dpdu_y, ctx.dpdu_z, ctx.nx, ctx.ny, ctx.nz);
        double wi_x, wi_y, wi_z;
        frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);
        vec3 dir = unit_vector(scattered.direction());
        double wo_x, wo_y, wo_z;
        frame.to_local(dir.x(), dir.y(), dir.z(), wo_x, wo_y, wo_z);
        double shape = ggx_reflection_shape(wi_x, wi_y, wi_z, wo_x, wo_y, wo_z, alpha_x, alpha_y);
        return shape * wo_z;
    }

    // pbrt's ConductorBxDF::f() evaluates the Fresnel term at the half-vector, |wo . wm| with wm = normalize(wi + wo) -
    // a different value for every direction pair. scatter() can only store one colour per hit (Fresnel at the view-normal
    // cosine), which over-brightened rough metal (+4% at alpha 0.5, +8% at alpha 1 against a pbrt-v4 path-level
    // reference, more where the view is grazing), so the colour for the direction actually being evaluated - an NEE
    // direction or the resampled BSDF continuation - comes from here, paired with scattering_pdf()'s shape * cos.
    color scattering_attenuation(const ray& r_in, const hit_record& rec, const ray& scattered,
                                  const color& srec_attenuation) const override {
        if (effectively_smooth()) return srec_attenuation;
        auto ctx   = MaterialContext<double>::from_hit(rec, r_in);
        auto frame = ShadingFrame<double>::from_dpdu(ctx.dpdu_x, ctx.dpdu_y, ctx.dpdu_z, ctx.nx, ctx.ny, ctx.nz);
        double wi_x, wi_y, wi_z;
        frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);
        const vec3 dir = unit_vector(scattered.direction());
        double wo_x, wo_y, wo_z;
        frame.to_local(dir.x(), dir.y(), dir.z(), wo_x, wo_y, wo_z);
        double hx = wi_x + wo_x, hy = wi_y + wo_y, hz = wi_z + wo_z;
        const double hlen = std::sqrt(hx * hx + hy * hy + hz * hz);
        if (hlen < 1e-12) return srec_attenuation;
        const double c = std::fabs((wi_x * hx + wi_y * hy + wi_z * hz) / hlen);
        return color(FrComplex(c, eta_r, k_r), FrComplex(c, eta_g, k_g), FrComplex(c, eta_b, k_b));
    }

    double get_roughness()  const { return alpha_x * alpha_x; }
    color  get_fresnel_normal() const {
        return color(FrComplex(1.0, eta_r, k_r),
                     FrComplex(1.0, eta_g, k_g),
                     FrComplex(1.0, eta_b, k_b));
    }

    // Delegates to the single effectively_smooth() helper below, the same
    // one scatter()'s own branch consults - see material::is_delta_bsdf()'s
    // own comment on why these two must never be allowed to drift apart.
    bool is_delta_bsdf() const override { return effectively_smooth(); }

  private:
    // The ONE place this class decides smooth-vs-glossy - see
    // rough_metal::effectively_smooth()'s own comment.
    bool effectively_smooth() const { return TrowbridgeReitz<double>(alpha_x, alpha_y).EffectivelySmooth(); }

    double eta_r, eta_g, eta_b;
    double k_r,   k_g,   k_b;
    double alpha_x, alpha_y;
};


// ---------------------------------------------------------------------------
// rough_dielectric -- GGX microfacet BSDF for rough glass (pbrt-v4 RoughDielectricBxDF)
// Samples a microfacet normal from the GGX VNDF, then stochastically reflects
// or refracts based on the Fresnel weight FrDielectric(dot(wi,wm), eta).
// roughness in [0,1]: 0 = perfect smooth glass, 1 = fully diffuse-like frosted glass
// ---------------------------------------------------------------------------
// Also implements dispersive_material (material_base.h) - see dielectric's
// own comment (material_simple.h) on why, and dispersive_material's own
// comment for why this is the one shared hook every dispersive material
// kind implements rather than a hook per concrete type.
class rough_dielectric : public material, public dispersive_material {
  public:
    using BxDF = RoughDielectricBxDF<double>;

    rough_dielectric(double refraction_index, double roughness)
        : ior(refraction_index) {
        double a = TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(roughness, 1e-4));
        alpha_x = alpha_y = a;
    }

    rough_dielectric(double refraction_index, double u_roughness, double v_roughness,
                      bool remap_roughness = true)
        : ior(refraction_index),
          alpha_x(roughness_or_alpha(u_roughness, remap_roughness)),
          alpha_y(roughness_or_alpha(v_roughness, remap_roughness)) {}

    // Isotropic roughness bound to a texture (pbrt-v4 "texture roughness"
    // on a Dielectric - e.g. a scratched/frosted-glass mask), sampled fresh
    // at each hit rather than fixed at construction - see resolve_alpha()'s
    // own comment for how. alpha_x/alpha_y stay at their default 0.0 here
    // (never read when roughness_tex_ is set - effectively_smooth() and
    // resolve_alpha() both branch on roughness_tex_ first).
    rough_dielectric(double refraction_index, shared_ptr<texture> roughness_tex,
                      bool remap_roughness = true)
        : ior(refraction_index), roughness_tex_(std::move(roughness_tex)),
          remap_roughness_(remap_roughness) {}

    // Named factory for dispersive (wavelength-dependent IOR) frosted glass -
    // same artist-facing (eta_d, Abbe number) pair and Cauchy-coefficient
    // derivation as dielectric::make_dispersive() (material_simple.h).
    // Needed here too, not just for smooth dielectric: unlike smooth glass
    // (always the skip_pdf specular branch below, never reaching NEE), a
    // rough dielectric's glossy branch is real NEE/MIS - so dispersion has
    // to reach scattering_pdf() as well as scatter(), or a directional/point
    // light (reachable ONLY via NEE, never by chance through BSDF sampling)
    // would see ordinary flat-IOR glass regardless of --spectral. See
    // scatter_dispersive()/scattering_pdf_dispersive() below and camera.h's
    // ray_color_spectral() for the dispatch.
    static shared_ptr<rough_dielectric> make_dispersive(double eta_d, double abbe_number, double roughness) {
        return shared_ptr<rough_dielectric>(new rough_dielectric(eta_d, abbe_number, roughness, dispersive_tag{}));
    }

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        // Mirrors coated_diffuse::get_bxdf()'s own identical texture lookup
        // (its own comment explains why: no CPU-only tex pointer differs
        // from scatter()'s real per-point sample, and nothing in this
        // codebase actually calls rough_dielectric::get_bxdf() today -
        // grepped, only referenced in this comment - so this is a best-
        // effort mirror of true_alpha()'s real per-hit logic, not a
        // load-bearing path). Without this, a texture-bound instance would
        // silently return alpha_x/alpha_y's 0.0 default (perfectly smooth)
        // regardless of what the texture says, since those fields are only
        // ever populated for the flat-value constructors.
        if (!roughness_tex_) return BxDF{ ior, alpha_x, alpha_y };
        const color c = roughness_tex_->value(ctx.u, ctx.v, point3(ctx.px, ctx.py, ctx.pz));
        const double a = roughness_or_alpha(c.x(), remap_roughness_);
        return BxDF{ ior, a, a };
    }

    // Real NEE/MIS below the roughness threshold (see rough_metal's own
    // comment). The glossy branch spans BOTH reflection and transmission
    // lobes via ggx_dielectric_pdf (pdf.h); srec.is_transmission=true there
    // signals "this material has real refraction physics with ratio
    // srec.eta", not "this specific scatter() sample transmitted" -- see
    // camera.h's Strategy B, which re-derives whether the actual resampled
    // bounce crossed the boundary geometrically, since srec.pdf_ptr's
    // sampled direction isn't known until Strategy B draws it.
    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        return scatter_impl(r_in, rec, srec, do_regularize, ior);
    }

    // Spectral-aware variant: computes eta from the path's hero wavelength
    // via CauchyEta() instead of the flat ior, when this instance was built
    // via make_dispersive() - otherwise identical to scatter(). Mirrors
    // dielectric::scatter_dispersive() (material_simple.h); reports the
    // resolved eta back via eta_out so scattering_pdf_dispersive() below
    // doesn't have to re-derive it from lambda_nm for the same bounce - see
    // dispersive_material's own comment (material_base.h). camera.h's
    // ray_color_spectral() is the only call site.
    bool scatter_dispersive(const ray& r_in, const hit_record& rec, scatter_record& srec,
                             float lambda_nm, bool do_regularize, double& eta_out) const override {
        double eta_ior = dispersive_ ? CauchyEta((double)lambda_nm, cauchy_A_, cauchy_B_) : ior;
        eta_out = eta_ior;
        return scatter_impl(r_in, rec, srec, do_regularize, eta_ior);
    }

    // f*cos at an arbitrary queried direction (scattered), for real NEE/MIS.
    // Achromatic (RoughDielectricBxDF is colorless glass) -- attenuation
    // stays the default white srec.attenuation.
    double scattering_pdf(const ray& r_in, const hit_record& rec,
                          const ray& scattered) const override {
        return scattering_pdf_impl(r_in, rec, scattered, ior);
    }

    // Spectral-aware variant of scattering_pdf(), for the same reason
    // scatter_dispersive() exists above: NEE evaluates the BSDF at a
    // light-sampled direction through THIS function, not scatter_dispersive()
    // - so a dispersive rough dielectric's NEE contribution needs the same
    // per-wavelength eta here too. Takes that eta directly (already resolved
    // by scatter_dispersive() for this bounce) rather than lambda_nm, so NEE
    // never pays for a second CauchyEta() evaluation - see
    // dispersive_material's own comment. See camera.h's ray_color_spectral()
    // for the 4 call sites (3 NEE strategies + Strategy B) that use this
    // instead of the plain scattering_pdf().
    double scattering_pdf_dispersive(const ray& r_in, const hit_record& rec,
                                      const ray& scattered, double eta) const override {
        return scattering_pdf_impl(r_in, rec, scattered, eta);
    }

    // True when this instance was built via make_dispersive() above. Kept
    // as a public accessor (parity with dielectric::is_dispersive(), and
    // useful for tests) even though ray_color_spectral() itself no longer
    // needs to re-check it - see dispersive_material's own comment on the
    // "non-null means yes" contract as_dispersive() below relies on.
    bool is_dispersive() const { return dispersive_; }

    double get_ior()       const { return ior; }
    // Meaningless (always 0.0) for a texture-bound instance - this method
    // takes no hit_record/MaterialContext, so unlike get_bxdf() above it has
    // no per-point sample to fall back to. No live caller anywhere in this
    // codebase today (grepped), so this is a documented, not-yet-load-bearing
    // limitation rather than a fix - a future caller needing a real answer
    // should use true_alpha() (via a hit_record) or get_bxdf() instead.
    double get_roughness() const { return alpha_x * alpha_x; }

    // Non-null only when constructed via the texture-roughness overload
    // above. Test/inspection accessor, same shape as coated_diffuse's own
    // get_texture() (material_pbrt.h).
    shared_ptr<texture> get_roughness_texture() const { return roughness_tex_; }

    // Opaque to shadow rays, like every pbrt-v4 surface that has a material
    // (VolPathIntegrator::SampleLd, integrators.cpp:1335). Letting NEE pass straight through glass
    // counted a light seen through it twice (once here, once on the specular BSDF path) and ignored
    // the refraction. Matches optix_anyhit_shadow.h / wavefront_anyhit_shadow.h, and holds for a glass shape that
    // bounds a medium too: pbrt blocks NEE at that shell, so light reaches the fog only along specular chains.
    // (This is the base class's default, so there is nothing to override.)

    // See material::as_dispersive()'s comment. `this` (not nullptr) only
    // when built via make_dispersive() above - lets ray_color_spectral()
    // find this instance through a wrapper material (mix_material) the
    // same way as_subsurface() already does for BSSRDF.
    const dispersive_material* as_dispersive(const hit_record&) const override {
        return dispersive_ ? this : nullptr;
    }

    // Delegates to the single effectively_smooth() helper below, the same
    // one scatter_impl()'s own branch consults - see material::
    // is_delta_bsdf()'s own comment. This is the fix for the bug that had
    // SPPM/BDPT/MLT treating every rough_dielectric instance as delta
    // regardless of roughness - see this class's own comment above on real
    // NEE/MIS below the roughness threshold. Collapsing both call sites
    // onto one expression (rather than each independently re-deriving it)
    // is what stops that exact bug from being able to recur here.
    bool is_delta_bsdf() const override { return effectively_smooth(); }

    // Per-hit-aware override (material::is_delta_bsdf(const hit_record&)'s
    // own comment) - SPPM's and BDPT/MLT's Intersect() already have a real
    // hit_record at the point they classify a hit, so for a texture-bound
    // instance this gives them the SAME precise per-hit answer scatter_impl/
    // scattering_pdf_impl below use, instead of the conservative "never
    // delta" the no-arg override above has to fall back to. Fixes a real
    // correctness bug (not just an efficiency one): without this, SPPM's
    // camera pass classified every texture-bound hit as non-delta BEFORE
    // calling scatter(), so it would permanently stop and record a visible
    // point at a hit whose texture-sampled roughness was actually near
    // enough to zero for scatter_impl() to take the specular branch instead
    // - the ray never continued through the glass the way scatter_impl()
    // itself would have handled it, a visibly wrong image, not merely a
    // less-efficient one.
    bool is_delta_bsdf(const hit_record& rec) const override {
        if (!roughness_tex_) return effectively_smooth();
        double ax, ay;
        true_alpha(rec, ax, ay);
        return is_smooth(ax, ay);
    }

  private:
    // Single shared expression for "is this alpha pair smooth" - every
    // caller (effectively_smooth() below, is_delta_bsdf(rec) above,
    // scatter_impl()'s own per-hit branch) goes through this one static
    // helper instead of each re-deriving TrowbridgeReitz<double>(ax,ay).
    // EffectivelySmooth() independently, so a future change to the
    // smoothness threshold/wrapper type only needs one edit.
    static bool is_smooth(double ax, double ay) {
        return TrowbridgeReitz<double>(ax, ay).EffectivelySmooth();
    }

    // The ONE place this class decides smooth-vs-glossy for is_delta_bsdf()
    // (no hit_record available there) - see rough_metal::effectively_smooth()'s
    // own comment. roughness_tex_ set means the true answer varies per-hit
    // (can't be known without a hit_record), so this conservatively answers
    // "never delta" rather than guessing from an arbitrary representative
    // alpha - callers with no hit_record (the bare is_delta_bsdf() override
    // above) fall back to this; callers WITH one (is_delta_bsdf(rec) above,
    // scatter_impl/scattering_pdf_impl below) get the real per-hit answer
    // via true_alpha() instead.
    // scatter_impl/scattering_pdf_impl below make the real, per-hit
    // smooth-vs-glossy decision instead, via true_alpha()'s own per-hit
    // sample.
    bool effectively_smooth() const {
        return !roughness_tex_ && is_smooth(alpha_x, alpha_y);
    }

    // Real per-hit roughness for MaterialKind::Dielectric's own "texture
    // roughness" case (roughness_tex_ set) - resolves the SAME conversion
    // roughness_or_alpha() applies once at construction for the flat-value
    // constructors above, but freshly at every hit since a texture's
    // sampled roughness varies by point. Isotropic only (the texture's own
    // red/x channel becomes both ax and ay) - matches this codebase's
    // established single-image-texture scope for Diffuse/CoatedDiffuse/
    // DiffuseTransmission's own reflectance/transmittance texture-binding
    // (no separate uroughness/vroughness texture support). When
    // roughness_tex_ is unset, just returns the flat alpha_x/alpha_y
    // already fixed at construction - identical cost and behavior to
    // before this overload existed.
    //
    // Known, accepted inefficiency when roughness_tex_ IS set: camera.h's
    // NEE queries scattering_pdf() once per active light-sampling strategy
    // against the SAME hit (area/portal/sky/each punctual light, all with
    // identical rec.u/rec.v/rec.p), so this resamples the texture that many
    // times per hit instead of once. Not fixed here: caching the sampled
    // value on the instance would be a data race (material instances are
    // shared across render threads, not per-hit copies), and threading a
    // pre-sampled alpha through camera.h's NEE call sites the way the GPU
    // backends' glossyAlphaForNEE does would mean restructuring shared
    // integrator code well beyond this class - a materially bigger change
    // than this feature's own current scope (no bundled scene uses
    // texture-bound dielectric roughness yet).
    void true_alpha(const hit_record& rec, double& ax, double& ay) const {
        if (!roughness_tex_) { ax = alpha_x; ay = alpha_y; return; }
        const color c = roughness_tex_->value(rec.u, rec.v, rec.p);
        const double a = roughness_or_alpha(c.x(), remap_roughness_);
        ax = ay = a;
    }

    // Disambiguates the dispersive constructor below from the public
    // (double, double, double, bool=true) constructor above: a bare 3-double
    // call is otherwise genuinely ambiguous (the 4th bool param defaults),
    // and a same-arity overload distinguished only by a trailing bool tag
    // would give the type system nothing to enforce - exactly the trap
    // dielectric::make_dispersive()'s own comment (material_simple.h)
    // warns about. An empty tag type can't accidentally bind to anything
    // else.
    struct dispersive_tag {};

    // Dispersive glass: wavelength-dependent IOR via the same two-term
    // Cauchy formula dielectric's own dispersive constructor derives
    // (material_simple.h), now shared via fresnel.h's
    // CauchyCoefficientsFromAbbe() rather than a second hand-copy of the
    // derivation. Delegates to the existing (refraction_index, roughness)
    // constructor above for the alpha_x/alpha_y derivation too, instead of
    // re-deriving RoughnessToAlpha() a second time - one call site for that
    // formula stays the only one to keep in sync if it ever changes.
    // `ior` (used by the ordinary flat-IOR scatter()/scattering_pdf() path,
    // i.e. --spectral off or a non-dispersive-lookalike caller) ends up set
    // to eta_d via that delegation, so both paths agree at the reference
    // wavelength. Private - construct via make_dispersive() above.
    rough_dielectric(double eta_d, double abbe_number, double roughness, dispersive_tag)
        : rough_dielectric(eta_d, roughness) {
        dispersive_ = true;
        CauchyCoefficientsFromAbbe(eta_d, abbe_number, cauchy_A_, cauchy_B_);
    }

    bool scatter_impl(const ray& r_in, const hit_record& rec, scatter_record& srec,
                       bool do_regularize, double ior_value) const {
        auto ctx   = MaterialContext<double>::from_hit(rec, r_in);
        double ax, ay;
        true_alpha(rec, ax, ay);
        double ex = do_regularize ? regularize_alpha(ax) : ax;
        double ey = do_regularize ? regularize_alpha(ay) : ay;
        BxDF bxdf{ ior_value, ex, ey };
        auto frame = ShadingFrame<double>::from_dpdu(ctx.dpdu_x, ctx.dpdu_y, ctx.dpdu_z, ctx.nx, ctx.ny, ctx.nz);

        double eta = ctx.front_face ? (1.0 / ior_value) : ior_value;

        double wi_x, wi_y, wi_z;
        frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);
        if (wi_z < 0.0) { wi_z = -wi_z; wi_x = -wi_x; wi_y = -wi_y; }

        // Branch on the TRUE (unregularized), per-hit roughness -- see
        // rough_metal's own comment on why scattering_pdf() forces this.
        // Uses ax/ay directly (not the class-level effectively_smooth(),
        // which conservatively answers "never smooth" when roughness_tex_
        // is set, since IT has no hit_record to sample from - this call
        // site does, so it makes the real per-hit decision instead, via
        // the same shared is_smooth() helper is_delta_bsdf(rec) above uses).
        if (is_smooth(ax, ay)) {
            auto res = bxdf.sample_local(wi_x, wi_y, wi_z, eta,
                                         random_double(), random_double(), random_double());
            if (!res.valid) return false;

            double wd_x, wd_y, wd_z;
            frame.to_world(res.wo_x, res.wo_y, res.wo_z, wd_x, wd_y, wd_z);
            srec.attenuation     = color(res.r, res.g, res.b);
            srec.pdf_ptr         = nullptr;
            srec.skip_pdf        = true;
            srec.skip_pdf_ray    = ray(rec.p, unit_vector(vec3(wd_x, wd_y, wd_z)), r_in.time());
            srec.eta             = res.is_transmission ? res.eta : 1.0;
            srec.is_transmission = res.is_transmission;
            return true;
        }

        srec.attenuation     = color(1.0, 1.0, 1.0);
        srec.pdf_ptr          = make_shared<ggx_dielectric_pdf>(
            rec.normal, vec3(ctx.wo_x, ctx.wo_y, ctx.wo_z), eta, ax, ay);
        srec.skip_pdf         = false;
        srec.eta              = eta;
        srec.is_transmission  = true;
        return true;
    }

    double scattering_pdf_impl(const ray& r_in, const hit_record& rec,
                                const ray& scattered, double ior_value) const {
        auto ctx   = MaterialContext<double>::from_hit(rec, r_in);
        auto frame = ShadingFrame<double>::from_dpdu(ctx.dpdu_x, ctx.dpdu_y, ctx.dpdu_z, ctx.nx, ctx.ny, ctx.nz);
        double eta = ctx.front_face ? (1.0 / ior_value) : ior_value;

        double wi_x, wi_y, wi_z;
        frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);
        if (wi_z < 0.0) { wi_z = -wi_z; wi_x = -wi_x; wi_y = -wi_y; }

        vec3 dir = unit_vector(scattered.direction());
        double wo_x, wo_y, wo_z;
        frame.to_local(dir.x(), dir.y(), dir.z(), wo_x, wo_y, wo_z);

        double ax, ay;
        true_alpha(rec, ax, ay);
        BxDF bxdf{ ior_value, ax, ay };
        double f_val = bxdf.f(wi_x, wi_y, wi_z, eta, wo_x, wo_y, wo_z);
        return f_val * std::fabs(wo_z);
    }

    double ior;
    double alpha_x = 0.0, alpha_y = 0.0;
    // Set only by the texture constructor above; nullptr (the common case)
    // means alpha_x/alpha_y are real, fixed-at-construction values and
    // true_alpha()/effectively_smooth() take their fast, texture-free path.
    shared_ptr<texture> roughness_tex_ = nullptr;
    bool remap_roughness_ = true;
    bool dispersive_ = false;
    double cauchy_A_ = 0.0, cauchy_B_ = 0.0;
};


// ---------------------------------------------------------------------------
// Layered (coated) BSDF plumbing shared by coated_diffuse and coated_conductor.
//
// Both are pbrt-v4's LayeredBxDF (src/shared/bxdfs_layered.h): a dielectric coat over an opaque base, evaluated
// by a stochastic random walk. pbrt integrates it with three separate pieces, and so does this:
//   - Sample_f (a walk): an unbiased path weight f*cos/pdf and a direction, but no usable density
//     -> scatter_record::walk_* (camera.h continues the path along it);
//   - PDF(): an approximate density used only for MIS weights -> scatter_record::mis_pdf_ptr;
//   - f(): a stochastic BSDF value for next-event estimation -> scattering_pdf()/scattering_attenuation().
// A BSDF whose coat and base are both smooth is a delta lobe and takes the plain skip_pdf path instead.
// ---------------------------------------------------------------------------
// Roughness -> GGX alpha for the layered materials. Unlike roughness_or_alpha() this does not floor the roughness at
// 1e-4 (alpha 0.01): pbrt's default coat and conductor roughness is 0, which is an exactly smooth interface, and
// the flooring made it a glossy one that no longer reads as a delta lobe.
inline double layered_alpha(double roughness, bool remap_roughness) {
    const double r = std::fmax(roughness, 0.0);
    return remap_roughness ? TrowbridgeReitz<double>::RoughnessToAlpha(r) : r;
}

inline uint64_t layered_random_seed() {
    return (static_cast<uint64_t>(random_double() * 4294967296.0) << 32) |
            static_cast<uint64_t>(random_double() * 4294967296.0);
}

// pdf over world directions wrapping the layered BxDF's PDF() - used for MIS only (see above).
template<typename Bx>
class layered_mis_pdf : public pdf {
  public:
    layered_mis_pdf(const Bx& bx_, const ShadingFrame<double>& frame_, double wx, double wy, double wz)
        : bx(bx_), frame(frame_), wi_x(wx), wi_y(wy), wi_z(wz) {}

    double value(const vec3& direction) const override {
        const vec3 d = unit_vector(direction);
        double lx, ly, lz;
        frame.to_local(d.x(), d.y(), d.z(), lx, ly, lz);
        if (lz <= 0.0) return 0.0;
        return bx.pdf(wi_x, wi_y, wi_z, lx, ly, lz);
    }

    // The camera never samples through this object (it follows scatter_record::walk_*); a cosine lobe keeps it
    // a well-formed pdf for any caller that does.
    vec3 generate() const override {
        double x, y, z, p;
        SampleCosineHemisphere(random_double(), random_double(), x, y, z, p);
        double wx, wy, wz;
        frame.to_world(x, y, z, wx, wy, wz);
        return unit_vector(vec3(wx, wy, wz));
    }

  private:
    Bx bx;
    ShadingFrame<double> frame;
    double wi_x, wi_y, wi_z;
};

// scatter() for a layered BSDF. rep_color is the colour srec.attenuation reports for the bridges that pair it with
// scattering_pdf(); (proxy_ax, proxy_ay) shape the GGX proxy density those same bridges draw from.
template<typename Bx>
inline bool layered_scatter(const Bx& bx, const MaterialContext<double>& ctx, const hit_record& rec,
                            const ray& r_in, scatter_record& srec, const color& rep_color,
                            double proxy_ax, double proxy_ay) {
    auto frame = ShadingFrame<double>::from_dpdu(ctx.dpdu_x, ctx.dpdu_y, ctx.dpdu_z, ctx.nx, ctx.ny, ctx.nz);
    double wi_x, wi_y, wi_z;
    frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);
    if (wi_z <= 0.0) return false;

    const auto res = bx.sample_local(wi_x, wi_y, wi_z, layered_random_seed(), layered_random_seed());
    vec3 wd(0, 0, 1);
    if (res.valid) {
        double dx, dy, dz;
        frame.to_world(res.wo_x, res.wo_y, res.wo_z, dx, dy, dz);
        wd = unit_vector(vec3(dx, dy, dz));
    }

    if (bx.is_delta()) {
        if (!res.valid) return false;
        srec.attenuation  = color(res.r, res.g, res.b);
        srec.pdf_ptr      = nullptr;
        srec.skip_pdf     = true;
        srec.skip_pdf_ray = ray(rec.p, wd, r_in.time());
        return true;
    }

    srec.attenuation = rep_color;
    srec.pdf_ptr     = make_shared<ggx_reflection_pdf>(rec.normal, vec3(ctx.wo_x, ctx.wo_y, ctx.wo_z), proxy_ax, proxy_ay);
    srec.mis_pdf_ptr = make_shared<layered_mis_pdf<Bx>>(bx, frame, wi_x, wi_y, wi_z);
    srec.skip_pdf    = false;
    srec.has_walk    = true;
    srec.walk_valid  = res.valid;
    if (res.valid) {
        srec.walk_specular = res.is_specular;
        srec.walk_ray      = ray(rec.p, wd, r_in.time());
        srec.walk_weight   = color(res.r, res.g, res.b);
    }
    return true;
}

// f(wi, scattered)*cos for NEE. The seed comes from the queried direction alone, so scattering_pdf() and
// scattering_attenuation() - two separate calls for the same query - see the same stochastic estimate.
template<typename Bx>
inline bool layered_eval_f_cos(const Bx& bx, const MaterialContext<double>& ctx, const ray& scattered,
                               color& f_cos) {
    auto frame = ShadingFrame<double>::from_dpdu(ctx.dpdu_x, ctx.dpdu_y, ctx.dpdu_z, ctx.nx, ctx.ny, ctx.nz);
    double wi_x, wi_y, wi_z;
    frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);
    if (wi_z <= 0.0) return false;

    const vec3 dir = unit_vector(scattered.direction());
    double wo_x, wo_y, wo_z;
    frame.to_local(dir.x(), dir.y(), dir.z(), wo_x, wo_y, wo_z);
    if (wo_z <= 0.0) return false;

    const uint64_t seed0 = coated_seed_from_dir(dir);
    double fr, fg, fb;
    bx.f(wi_x, wi_y, wi_z, wo_x, wo_y, wo_z, seed0, seed0 ^ 0xABCDEF1234567890ull, fr, fg, fb);
    f_cos = color(fr * wo_z, fg * wo_z, fb * wo_z);
    return true;
}

// Splits f*cos into a scalar (scattering_pdf()) and a colour (scattering_attenuation()) whose product is f*cos.
inline double layered_f_cos_scalar(const color& f_cos) { return (f_cos.x() + f_cos.y() + f_cos.z()) / 3.0; }
inline color layered_f_cos_tint(const color& f_cos, const color& fallback) {
    const double s = layered_f_cos_scalar(f_cos);
    return s > 0.0 ? f_cos / s : fallback;
}


// ---------------------------------------------------------------------------
// coated_diffuse
// pbrt-v4 CoatedDiffuseBxDF = LayeredBxDF<DielectricBxDF, DiffuseBxDF, true>: a dielectric coat (IOR `ior`, GGX
// roughness `roughness`, smooth by default) over a Lambertian base (`albedo`). The coat refracts, so light
// reaching the base is bent toward the normal and a diffuse bounce has to escape the coat again - total internal
// reflection sends part of it back down for another bounce.
// ---------------------------------------------------------------------------
class coated_diffuse : public material {
  public:
    using BxDF = CoatedDiffuseBxDF<double>;

    coated_diffuse(const color& albedo, double ior, double roughness)
        : tex(make_shared<solid_color>(albedo)), ior(ior) {
        double a = layered_alpha(roughness, true);
        alpha_x = alpha_y = a;
    }

    coated_diffuse(const color& albedo, double ior, double u_roughness, double v_roughness,
                   bool remap_roughness = true)
        : tex(make_shared<solid_color>(albedo)), ior(ior),
          alpha_x(layered_alpha(u_roughness, remap_roughness)),
          alpha_y(layered_alpha(v_roughness, remap_roughness)) {}

    // "reflectance" bound to a real Texture (pbrt's own ganesha/barcelona-
    // pavilion "texture reflectance" - see pbrt_flatten::Material::
    // textureFilename's own comment) rather than a flat colour - mirrors
    // lambertian's own two-constructor shape (material_simple.h) exactly.
    coated_diffuse(shared_ptr<texture> tex, double ior, double u_roughness, double v_roughness,
                   bool remap_roughness = true)
        : tex(tex), ior(ior),
          alpha_x(layered_alpha(u_roughness, remap_roughness)),
          alpha_y(layered_alpha(v_roughness, remap_roughness)) {}

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        color albedo = tex->value(ctx.u, ctx.v, point3(ctx.px, ctx.py, ctx.pz));
        return BxDF{ albedo.x(), albedo.y(), albedo.z(), ior, alpha_x, alpha_y };
    }

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        auto ctx = MaterialContext<double>::from_hit(rec, r_in);
        const color albedo = base_albedo(rec);
        return layered_scatter(make_bxdf(albedo, do_regularize), ctx, rec, r_in, srec, albedo, alpha_x, alpha_y);
    }

    // f(wi, scattered)*cos split into a scalar here and a colour in scattering_attenuation() below - see
    // layered_f_cos_scalar().
    double scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) const override {
        auto ctx = MaterialContext<double>::from_hit(rec, r_in);
        color f_cos;
        if (!layered_eval_f_cos(make_bxdf(base_albedo(rec), false), ctx, scattered, f_cos)) return 0.0;
        return layered_f_cos_scalar(f_cos);
    }

    color scattering_attenuation(const ray& r_in, const hit_record& rec, const ray& scattered,
                                 const color& srec_attenuation) const override {
        auto ctx = MaterialContext<double>::from_hit(rec, r_in);
        color f_cos;
        if (!layered_eval_f_cos(make_bxdf(base_albedo(rec), false), ctx, scattered, f_cos)) return srec_attenuation;
        return layered_f_cos_tint(f_cos, srec_attenuation);
    }

    double get_ior()       const { return ior; }
    double get_roughness() const { return alpha_x * alpha_x; }
    // A representative colour, not the real per-point value - callers
    // wanting the latter should evaluate `tex` themselves at a real (u,v,p).
    color get_albedo() const { return tex->value(0.5, 0.5, point3(0, 0, 0)); }

    // Accessor for testing/serialization - mirrors lambertian's own
    // get_texture() (material_simple.h) exactly.
    shared_ptr<texture> get_texture() const { return tex; }

    // A Lambertian base always scatters diffusely, so this is never a delta BSDF (a smooth coat's mirror
    // reflection is one lobe of it, reported per sample through scatter_record::walk_specular).
    bool is_delta_bsdf() const override { return false; }

  private:
    color base_albedo(const hit_record& rec) const {
        return tex->value_diff(rec.u, rec.v, rec.p, rec.dudx, rec.dvdx, rec.dudy, rec.dvdy);
    }
    BxDF make_bxdf(const color& albedo, bool do_regularize) const {
        const double ex = do_regularize ? regularize_alpha(alpha_x) : alpha_x;
        const double ey = do_regularize ? regularize_alpha(alpha_y) : alpha_y;
        return BxDF{ albedo.x(), albedo.y(), albedo.z(), ior, ex, ey };
    }

    shared_ptr<texture> tex;
    double ior;
    double alpha_x, alpha_y;
};


// ---------------------------------------------------------------------------
// thin_dielectric
//
// Models a zero-thickness flat slab of glass (window pane, soap bubble).
// Because the slab has no volume, the transmitted ray exits on the same side
// it entered (wi = -wo, not refracted).
//
// Multiple internal bounces are folded analytically:
//   R_eff = R + T^2 * R / (1 - R^2)   (geometric series of internal bounces)
//   T_eff = 1 - R_eff
//
// Reference: pbrt-v4 src/pbrt/bxdfs.h ThinDielectricBxDF::Sample_f
// ---------------------------------------------------------------------------
class thin_dielectric : public material {
  public:
    using BxDF = ThinDielectricBxDF<double>;

    thin_dielectric(double ior) : ior(ior) {}

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        return BxDF{ ior };
    }

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        auto ctx = MaterialContext<double>::from_hit(rec, r_in);
        auto bxdf = get_bxdf(ctx);
        vec3 in_dir = unit_vector(r_in.direction());
        auto res = bxdf.sample(in_dir.x(), in_dir.y(), in_dir.z(),
                               ctx.nx, ctx.ny, ctx.nz,
                               random_double());
        srec.attenuation  = color(res.r, res.g, res.b);
        srec.pdf_ptr      = nullptr;
        srec.skip_pdf     = true;
        srec.skip_pdf_ray = ray(rec.p, vec3(res.wo_x, res.wo_y, res.wo_z), r_in.time());
        return true;
    }

    double get_ior() const { return ior; }

    // Opaque to shadow rays (the base default) - see rough_dielectric's comment for the pbrt-v4 reference.

    // No roughness parameter at all - scatter() above always takes the
    // skip_pdf=true path. See material::is_delta_bsdf()'s own comment.
    bool is_delta_bsdf() const override { return true; }

  private:
    double ior;
};


// ---------------------------------------------------------------------------
// coated_conductor -- dielectric coat over a GGX conductor base
// pbrt-v4 CoatedConductorBxDF = LayeredBxDF<DielectricBxDF, ConductorBxDF, true>, built the way pbrt's
// CoatedConductorMaterial::GetBxDF builds it:
//   - the conductor sits INSIDE the coat, so its complex IOR is relative to the coat: eta/k are both divided by
//     the coat's IOR before use (a copper base under a 1.5 coat is eta 0.16, k 2.25, not eta 0.246, k 3.378);
//   - the coat and the conductor each have their own roughness (pbrt: interface.roughness, conductor.roughness,
//     both 0 by default - a mirror under a glass sheet);
//   - thickness (default 0.01) only attenuates the layer's transmittance (Beer-Lambert, unit extinction).
//
// Parameters:
//   eta_r/g/b, k_r/g/b -- the conductor's complex IOR (relative to vacuum; divided by coat_ior internally)
//   coat_ior           -- the coat's index of refraction (pbrt: interface.eta, default 1.5)
//   coat roughness     -- GGX roughness of the coat (pbrt: interface.roughness)
//   conductor roughness-- GGX roughness of the conductor (pbrt: conductor.roughness); the older constructors
//                         without it use the coat's value for both interfaces
// ---------------------------------------------------------------------------
class coated_conductor : public material {
  public:
    using BxDF = CoatedConductorBxDF<double>;

    coated_conductor(double eta_r, double eta_g, double eta_b,
                     double k_r,   double k_g,   double k_b,
                     double coat_ior, double coat_roughness)
        : eta_r(eta_r), eta_g(eta_g), eta_b(eta_b),
          k_r(k_r),     k_g(k_g),     k_b(k_b),
          coat_ior(coat_ior) {
        double a = layered_alpha(coat_roughness, true);
        alpha_x = alpha_y = a;
    }

    coated_conductor(double eta_r, double eta_g, double eta_b,
                     double k_r,   double k_g,   double k_b,
                     double coat_ior, double u_roughness, double v_roughness,
                     bool remap_roughness = true)
        : eta_r(eta_r), eta_g(eta_g), eta_b(eta_b),
          k_r(k_r),     k_g(k_g),     k_b(k_b),
          coat_ior(coat_ior),
          alpha_x(layered_alpha(u_roughness, remap_roughness)),
          alpha_y(layered_alpha(v_roughness, remap_roughness)) {}

    // Independent conductor roughness (pbrt's conductor.uroughness/vroughness), and the layer thickness.
    coated_conductor(double eta_r, double eta_g, double eta_b,
                     double k_r,   double k_g,   double k_b,
                     double coat_ior, double coat_u_roughness, double coat_v_roughness,
                     double cond_u_roughness, double cond_v_roughness,
                     bool remap_roughness = true, double thickness = 0.01)
        : eta_r(eta_r), eta_g(eta_g), eta_b(eta_b),
          k_r(k_r),     k_g(k_g),     k_b(k_b),
          coat_ior(coat_ior),
          alpha_x(layered_alpha(coat_u_roughness, remap_roughness)),
          alpha_y(layered_alpha(coat_v_roughness, remap_roughness)),
          cond_alpha_x(layered_alpha(cond_u_roughness, remap_roughness)),
          cond_alpha_y(layered_alpha(cond_v_roughness, remap_roughness)),
          thickness(thickness) {}

    coated_conductor(const ConductorPreset& preset, double coat_ior, double coat_roughness)
        : eta_r(preset.eta_r), eta_g(preset.eta_g), eta_b(preset.eta_b),
          k_r(preset.k_r),     k_g(preset.k_g),     k_b(preset.k_b),
          coat_ior(coat_ior) {
        double a = layered_alpha(coat_roughness, true);
        alpha_x = alpha_y = a;
    }

    coated_conductor(const ConductorPreset& preset, double coat_ior,
                     double u_roughness, double v_roughness)
        : eta_r(preset.eta_r), eta_g(preset.eta_g), eta_b(preset.eta_b),
          k_r(preset.k_r),     k_g(preset.k_g),     k_b(preset.k_b),
          coat_ior(coat_ior),
          alpha_x(layered_alpha(u_roughness, true)),
          alpha_y(layered_alpha(v_roughness, true)) {}

    BxDF get_bxdf(const MaterialContext<double>& /*ctx*/) const { return make_bxdf(false); }

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        auto ctx = MaterialContext<double>::from_hit(rec, r_in);
        const BxDF bx = make_bxdf(do_regularize);
        // The proxy lobe the bridges (BDPT/MLT/SPPM) sample: GGX at the rougher of the two interfaces.
        const double cax = bx.cond_alpha_x < 0.0 ? bx.alpha_x : bx.cond_alpha_x;
        const double cay = bx.cond_alpha_y < 0.0 ? bx.alpha_y : bx.cond_alpha_y;
        return layered_scatter(bx, ctx, rec, r_in, srec, get_conductor_f0(),
                               std::max(bx.alpha_x, cax), std::max(bx.alpha_y, cay));
    }

    // f(wi, scattered)*cos split into a scalar here and a colour in scattering_attenuation() below - see
    // layered_f_cos_scalar().
    double scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) const override {
        auto ctx = MaterialContext<double>::from_hit(rec, r_in);
        color f_cos;
        if (!layered_eval_f_cos(make_bxdf(false), ctx, scattered, f_cos)) return 0.0;
        return layered_f_cos_scalar(f_cos);
    }

    color scattering_attenuation(const ray& r_in, const hit_record& rec, const ray& scattered,
                                 const color& srec_attenuation) const override {
        auto ctx = MaterialContext<double>::from_hit(rec, r_in);
        color f_cos;
        if (!layered_eval_f_cos(make_bxdf(false), ctx, scattered, f_cos)) return srec_attenuation;
        return layered_f_cos_tint(f_cos, srec_attenuation);
    }

    double get_coat_ior()       const { return coat_ior; }
    double get_coat_roughness() const { return alpha_x * alpha_x; }
    // Normal-incidence reflectance of the conductor as the coated BSDF sees it (eta/k relative to the coat).
    color  get_conductor_f0()   const {
        return color(FrComplex(1.0, eta_r / coat_ior, k_r / coat_ior),
                     FrComplex(1.0, eta_g / coat_ior, k_g / coat_ior),
                     FrComplex(1.0, eta_b / coat_ior, k_b / coat_ior));
    }

    // pbrt's Flags(): a smooth coat over a smooth conductor is a delta lobe, anything rougher is not.
    bool is_delta_bsdf() const override { return make_bxdf(false).is_delta(); }

  private:
    BxDF make_bxdf(bool do_regularize) const {
        const double ex = do_regularize ? regularize_alpha(alpha_x) : alpha_x;
        const double ey = do_regularize ? regularize_alpha(alpha_y) : alpha_y;
        BxDF bx{ eta_r / coat_ior, eta_g / coat_ior, eta_b / coat_ior,
                 k_r / coat_ior,   k_g / coat_ior,   k_b / coat_ior,
                 coat_ior, ex, ey, thickness };
        if (cond_alpha_x >= 0.0) {
            bx.cond_alpha_x = do_regularize ? regularize_alpha(cond_alpha_x) : cond_alpha_x;
            bx.cond_alpha_y = do_regularize ? regularize_alpha(cond_alpha_y) : cond_alpha_y;
        }
        return bx;
    }

    double eta_r, eta_g, eta_b;
    double k_r,   k_g,   k_b;
    double coat_ior;
    double alpha_x, alpha_y;
    double cond_alpha_x = -1.0, cond_alpha_y = -1.0;   // negative: use the coat's
    double thickness = 0.01;
};

// ---------------------------------------------------------------------------
// diffuse_transmission -- pbrt-v4 DiffuseTransmissionBxDF
// Models materials like wax, skin (cheap SSS), leaves, and frosted panels
// that scatter transmitted light diffusely into the opposite hemisphere.
// R = reflectance color (diffuse reflection), T = transmittance color.
// Stochastically chooses reflection (cosine-weighted same hemisphere) or
// transmission (cosine-weighted opposite hemisphere) weighted by max(R)/max(T).
// ---------------------------------------------------------------------------
class diffuse_transmission : public material {
  public:
    using BxDF = DiffuseTransmissionBxDF<double>;

    diffuse_transmission(const color& reflectance, const color& transmittance)
        : R(reflectance), T(transmittance) {}

    // "reflectance"/"transmittance" each optionally bound to a real Texture
    // (barcelona-pavilion's foliage - see pbrt_flatten::Material::
    // textureFilename/transmittanceTextureFilename's own comments) instead
    // of a flat colour - either or both may be null, in which case that
    // channel falls back to R/T above. Mirrors coated_diffuse's own
    // two-constructor shape.
    diffuse_transmission(const color& reflectance, const color& transmittance,
                          shared_ptr<texture> reflectance_tex, shared_ptr<texture> transmittance_tex)
        : R(reflectance), T(transmittance), rTex(reflectance_tex), tTex(transmittance_tex) {}

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        // No pixel-footprint differentials available from MaterialContext
        // alone - same "no CPU-only tex pointer here, not a load-bearing
        // path" limitation coated_diffuse::get_bxdf() already documents.
        return BxDF{ R.x(), R.y(), R.z(), T.x(), T.y(), T.z() };
    }

    // Per-point reflectance/transmittance when texture-bound, else the flat
    // R/T - used by scatter()/scattering_pdf()/scattering_attenuation()
    // below, each of which already has `rec` in scope.
    color reflectance_at(const hit_record& rec) const {
        return rTex ? rTex->value_diff(rec.u, rec.v, rec.p, rec.dudx, rec.dvdx, rec.dudy, rec.dvdy) : R;
    }
    color transmittance_at(const hit_record& rec) const {
        return tTex ? tTex->value_diff(rec.u, rec.v, rec.p, rec.dudx, rec.dvdx, rec.dudy, rec.dvdy) : T;
    }

    bool scatter(const ray& r_in, const hit_record& rec,
                 scatter_record& srec, bool do_regularize = false) const override {
        // Probabilities proportional to max component (pbrt-v4 pattern)
        color r = reflectance_at(rec);
        color t = transmittance_at(rec);
        double pr = std::fmax(std::fmax(r.x(), r.y()), r.z());
        double pt = std::fmax(std::fmax(t.x(), t.y()), t.z());
        if (pr + pt <= 0.0) return false;

        if (random_double() < pr / (pr + pt)) {
            // Diffuse reflection: cosine-weighted same hemisphere as normal
            srec.attenuation  = r;
            srec.pdf_ptr      = make_shared<cosine_pdf>(rec.normal);
            srec.skip_pdf     = false;
        } else {
            // Diffuse transmission: cosine-weighted opposite hemisphere
            // Use cosine_pdf around -normal so MIS and PDF evaluation are correct,
            // matching pbrt-v4 where transmission PDF = pt/(pr+pt) * cos/pi
            srec.attenuation  = t;
            srec.pdf_ptr      = make_shared<cosine_pdf>(-rec.normal);
            srec.skip_pdf     = false;
        }
        return true;
    }

    double scattering_pdf(const ray& r_in, const hit_record& rec,
                          const ray& scattered) const override {
        // pbrt-v4 DiffuseTransmissionBxDF::PDF:
        //   same hemisphere  -> pr/(pr+pt) * cos(theta)/pi
        //   opposite hemisphere -> pt/(pr+pt) * cos(theta)/pi
        color r = reflectance_at(rec);
        color t = transmittance_at(rec);
        double pr = std::fmax(std::fmax(r.x(), r.y()), r.z());
        double pt = std::fmax(std::fmax(t.x(), t.y()), t.z());
        if (pr + pt <= 0.0) return 0.0;

        double cos_theta = dot(rec.normal, unit_vector(scattered.direction()));
        if (cos_theta > 0.0)
            return (pr / (pr + pt)) * (cos_theta / pi);   // reflection
        else
            return (pt / (pr + pt)) * (-cos_theta / pi);  // transmission
    }

    // See material::scattering_attenuation()'s own comment for why this
    // exists: scatter() above fixes srec.attenuation to R or T based on a
    // ONE-TIME stochastic pick, before camera.h's NEE code even knows which
    // direction (toward a light) it will evaluate. Reusing that fixed value
    // for an NEE ray on the OPPOSITE hemisphere from what the pick landed on
    // silently applied the wrong color (R's tint on transmitted light, or
    // T's on reflected light) with probability pr/(pr+pt) or pt/(pr+pt) -
    // deterministic per scattering event, not noise that ever averaged out.
    // This mirrors scattering_pdf() above exactly: pick R or T by the
    // QUERIED direction's actual hemisphere, not by re-deriving scatter()'s
    // earlier coin flip.
    color scattering_attenuation(const hit_record& rec, const ray& scattered,
                                  const color& srec_attenuation) const override {
        (void)srec_attenuation;
        double cos_theta = dot(rec.normal, unit_vector(scattered.direction()));
        return (cos_theta > 0.0) ? reflectance_at(rec) : transmittance_at(rec);
    }

    color get_reflectance()   const { return R; }
    color get_transmittance() const { return T; }

    // Accessors for testability, mirroring coated_diffuse::get_texture()/
    // rough_dielectric::get_roughness_texture().
    shared_ptr<texture> get_reflectance_texture()   const { return rTex; }
    shared_ptr<texture> get_transmittance_texture() const { return tTex; }

    // See material::is_shadow_transmissive()'s comment - matches
    // optix_anyhit_shadow.h's MaterialType::DiffuseTransmission skip.
    // Unconditional, not weighted by R vs T, matching that same GPU
    // convention: this material is treated as non-occluding outright rather
    // than probabilistically, regardless of how much of its energy actually
    // reflects.
    bool is_shadow_transmissive(const hit_record&) const override { return true; }

  private:
    color R;  // reflectance (same-hemisphere diffuse)
    color T;  // transmittance (opposite-hemisphere diffuse)
    shared_ptr<texture> rTex;  // optional, overrides R per-point when set
    shared_ptr<texture> tTex;  // optional, overrides T per-point when set
};

// ---------------------------------------------------------------------------
// normalized_fresnel -- pbrt-v4 NormalizedFresnelBxDF
// Fresnel-weighted diffuse reflection used at BSSRDF exit boundaries.
// Models crystal, gem, or subsurface-entry surfaces.
//
// BSDF:  f(wi) = (1 - FrDielectric(cos(wi), eta)) / (c * pi)
// where: c = 1 - 2 * FresnelMoment1(1/eta)
//
// MIS path weight derivation:
//   attenuation * scattering_pdf / pdf_brdf
//   = 1 * (1-Fr)*cos/(c*pi) / (cos/pi)
//   = (1-Fr)/c                              -- matches pbrt-v4 Sample_f weight
//
// We use skip_pdf=false so the MIS integrator handles both direct and indirect
// lighting correctly, exactly as pbrt-v4's DiffuseReflection flag implies.
// ---------------------------------------------------------------------------
class normalized_fresnel : public material {
  public:
    using BxDF = NormalizedFresnelBxDF<double>;

    explicit normalized_fresnel(double ior) : eta(ior) {
        double inv_eta = 1.0 / eta;
        c = 1.0 - 2.0 * FresnelMoment1(inv_eta);
        if (c <= 0.0) c = 1e-6;
    }

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        return BxDF{ eta };
    }

    bool scatter(const ray& r_in, const hit_record& rec,
                 scatter_record& srec, bool do_regularize = false) const override {
        // attenuation=white; scattering_pdf carries the Fresnel-weighted BSDF value.
        // The MIS integrator divides by cosine_pdf internally, giving (1-Fr)/c.
        srec.attenuation = color(1.0, 1.0, 1.0);
        srec.pdf_ptr     = make_shared<cosine_pdf>(rec.normal);
        srec.skip_pdf    = false;
        return true;
    }

    double scattering_pdf(const ray& r_in, const hit_record& rec,
                          const ray& scattered) const override {
        // pbrt-v4 NormalizedFresnelBxDF::f(wo,wi) = (1 - Fr(cos_wi, eta)) / (c * pi)
        // scattering_pdf = f * cos_wi = (1 - Fr(cos_wi, eta)) * cos_wi / (c * pi)
        double cos_wi = dot(rec.normal, unit_vector(scattered.direction()));
        if (cos_wi <= 0.0) return 0.0;
        double fr = FrDielectric(cos_wi, eta);
        return (1.0 - fr) * cos_wi / (c * pi);
    }

    double get_ior() const { return eta; }
    double get_c()   const { return c; }

  private:
    double eta;
    double c;  // 1 - 2*FresnelMoment1(1/eta)
};


// Deterministic pseudo-random value in [0,1) from a world-space point - used
// by mix_material/diffuse_transmission below to pick a sub-material/lobe
// branch. scatter(), scattering_pdf() and is_shadow_transmissive() are
// separate stateless calls (the `material` interface threads no state
// between them) with no shared RNG draw - a fresh random_double() in each
// let them silently disagree about which branch a given scattering event
// actually used: scatter() might commit to mat_a (attenuation and pdf_ptr
// both from mat_a), while a later scattering_pdf() call for that exact same
// event independently re-rolled and evaluated mat_b instead, mismatching the
// term multiplied into the same estimator. Hashing rec.p pins the decision
// to the hit point so every call about the same scattering event agrees,
// matching pbrt-v4's actual MixMaterial semantics (a single per-intersection
// stochastic material choice, not a true blended BxDF evaluation).
inline double branch_hash01(const point3& p) {
    double h = std::sin(p.x() * 127.1 + p.y() * 311.7 + p.z() * 74.7) * 43758.5453;
    return h - std::floor(h);
}

// ---------------------------------------------------------------------------
// mix_material -- stochastic blend of two materials (pbrt-v4 MixMaterial)
//
// At each shading point, randomly selects material A with probability (1-w)
// or material B with probability w, where w = weight texture R channel.
// This is unbiased: expected attenuation = (1-w)*A + w*B over many samples.
//
// Reference: pbrt-v4 materials.h MixMaterial::GetBxDF
//   "Stochastically select one of the two materials based on a uniform
//    random number and the mixing weight."
//
// Parameters:
//   mat_a   -- first material (weight (1-w))
//   mat_b   -- second material (weight w)
//   weight  -- scalar or texture in [0,1]; 0 = pure A, 1 = pure B
// ---------------------------------------------------------------------------
class mix_material : public material {
  public:
    mix_material(shared_ptr<material> mat_a,
                 shared_ptr<material> mat_b,
                 double weight)
        : mat_a(mat_a), mat_b(mat_b),
          weight_tex(make_shared<solid_color>(color(weight, weight, weight))) {}

    mix_material(shared_ptr<material> mat_a,
                 shared_ptr<material> mat_b,
                 shared_ptr<texture> weight_tex)
        : mat_a(mat_a), mat_b(mat_b), weight_tex(weight_tex) {}

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        double w = weight_tex->value(rec.u, rec.v, rec.p).x();
        w = w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w);
        // Stochastic selection (pbrt-v4 MixMaterial) - deterministic given
        // rec.p, not random_double(), so scattering_pdf()/
        // is_shadow_transmissive() below agree with whichever sub-material
        // this call committed to. See branch_hash01()'s own comment.
        if (branch_hash01(rec.p) >= w)
            return mat_a->scatter(r_in, rec, srec, do_regularize);
        else
            return mat_b->scatter(r_in, rec, srec, do_regularize);
    }

    color emitted(const ray& r_in, const hit_record& rec,
                  double u, double v, const point3& p) const override {
        double w = weight_tex->value(u, v, p).x();
        w = w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w);
        color ea = mat_a->emitted(r_in, rec, u, v, p);
        color eb = mat_b->emitted(r_in, rec, u, v, p);
        return ea * (1.0 - w) + eb * w;
    }

    double scattering_pdf(const ray& r_in, const hit_record& rec,
                          const ray& scattered) const override {
        double w = weight_tex->value(rec.u, rec.v, rec.p).x();
        w = w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w);
        // Same deterministic branch as scatter() above - NOT a blend of both
        // sub-materials' scattering_pdf. Blending here while scatter() above
        // committed srec.attenuation/pdf_ptr entirely to one sub-material
        // mismatched the two terms of the same Monte Carlo estimator (see
        // branch_hash01()'s comment).
        if (branch_hash01(rec.p) >= w)
            return mat_a->scattering_pdf(r_in, rec, scattered);
        else
            return mat_b->scattering_pdf(r_in, rec, scattered);
    }

    // Same deterministic branch as scatter()/scattering_pdf() above - without
    // this override, the base class default (return srec_attenuation
    // unchanged) applied even when the committed sub-material is itself a
    // diffuse_transmission, silently reintroducing the exact NEE-hemisphere
    // color bug diffuse_transmission::scattering_attenuation() exists to fix
    // (mix(diffuse_transmission, X) would ignore it entirely).
    color scattering_attenuation(const hit_record& rec, const ray& scattered,
                                  const color& srec_attenuation) const override {
        double w = weight_tex->value(rec.u, rec.v, rec.p).x();
        w = w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w);
        if (branch_hash01(rec.p) >= w)
            return mat_a->scattering_attenuation(rec, scattered, srec_attenuation);
        else
            return mat_b->scattering_attenuation(rec, scattered, srec_attenuation);
    }

    // The incoming-direction form, forwarded the same way (a conductor inside a mix needs it).
    color scattering_attenuation(const ray& r_in, const hit_record& rec, const ray& scattered,
                                  const color& srec_attenuation) const override {
        double w = weight_tex->value(rec.u, rec.v, rec.p).x();
        w = w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w);
        if (branch_hash01(rec.p) >= w)
            return mat_a->scattering_attenuation(r_in, rec, scattered, srec_attenuation);
        else
            return mat_b->scattering_attenuation(r_in, rec, scattered, srec_attenuation);
    }

    // Same deterministic branch as scatter() above - without this override,
    // the base class default (nullptr) applied unconditionally, so
    // mix(subsurface, X) silently dropped real BSSRDF exit sampling
    // (camera.h's sample_bssrdf_exit() gates on `rec.mat->as_subsurface()`
    // being non-null right after scatter() returns a specular transmission -
    // it needs to see the SAME branch scatter() just committed to for this
    // same rec, which base_material::as_subsurface() taking `rec` now makes
    // possible).
    const subsurface* as_subsurface(const hit_record& rec) const override {
        double w = weight_tex->value(rec.u, rec.v, rec.p).x();
        w = w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w);
        if (branch_hash01(rec.p) >= w)
            return mat_a->as_subsurface(rec);
        else
            return mat_b->as_subsurface(rec);
    }

    // Same deterministic branch, same reason - without this override, the
    // base class default (nullptr) applied unconditionally, so
    // mix(dispersive_material, X) would silently drop chromatic dispersion
    // under --spectral regardless of which branch was picked (camera.h's
    // ray_color_spectral() gates dispersive scattering on
    // `rec.mat->as_dispersive(rec)` being non-null). One override covers
    // every dispersive material kind mixed in (dielectric, rough_dielectric,
    // any future one) since material::as_dispersive() is a single shared
    // hook - see dispersive_material's own comment (material_base.h).
    const dispersive_material* as_dispersive(const hit_record& rec) const override {
        double w = weight_tex->value(rec.u, rec.v, rec.p).x();
        w = w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w);
        if (branch_hash01(rec.p) >= w)
            return mat_a->as_dispersive(rec);
        else
            return mat_b->as_dispersive(rec);
    }

    shared_ptr<material> get_mat_a()   const { return mat_a; }
    shared_ptr<material> get_mat_b()   const { return mat_b; }
    shared_ptr<texture>  get_weight()  const { return weight_tex; }

    // GPU has no equivalent Mix material type, so there is no GPU convention
    // to match here (unlike every other override in this file) - this is a
    // CPU-only correctness improvement, extending the same principle
    // scatter() already uses: stochastically resolve to whichever sub-material
    // this particular shadow ray would have hit, weighted the same way a
    // scatter event would be. Deterministic given rec.p (branch_hash01()),
    // not random_double(), for the same reason scatter()/scattering_pdf()
    // above are - a shadow ray probing this exact point should agree with
    // what a scatter event at that same point would have committed to.
    bool is_shadow_transmissive(const hit_record& rec) const override {
        double w = weight_tex->value(rec.u, rec.v, rec.p).x();
        w = w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w);
        return (branch_hash01(rec.p) >= w) ? mat_a->is_shadow_transmissive(rec)
                                            : mat_b->is_shadow_transmissive(rec);
    }

  private:
    shared_ptr<material> mat_a;
    shared_ptr<material> mat_b;
    shared_ptr<texture>  weight_tex;
};


// ---------------------------------------------------------------------------
// subsurface -- pbrt-v4 SubsurfaceMaterial (real BSSRDF, CPU-only)
//
// The entry/exit interface is the exact same smooth DielectricBxDF every
// other dielectric-family material in this file uses (matches pbrt-v4's own
// SubsurfaceMaterial::GetBxDF, which returns a plain DielectricBxDF(eta,
// distrib) - none of this loader's subsurface scenes give it a roughness).
// scatter() is therefore `dielectric::scatter()` (material_simple.h) with no
// transmission_filter - the Fresnel entry/exit split is identical code.
//
// The actual subsurface LIGHT TRANSPORT (what happens between the entry and
// exit points) cannot live here: a material's scatter() only ever sees the
// local hit_record, never the scene's geometry, and finding the exit point
// means walking a probe ray through the SAME object's own geometry -
// bssrdf.h's TabulatedBSSRDF only has the radial diffusion profile
// (Sr(r)/SampleSr/PDF_Sr), not probe-ray intersection. That geometry-probing
// step lives in camera.h::sample_bssrdf_exit(), which fires whenever a
// scatter() event is a specular transmission into a material whose
// as_subsurface() is non-null - see that function's own comment for the
// full algorithm (a port of pbrt-v4 VolPathIntegrator::Li's BSSRDF branch,
// cpu/integrators.cpp ~1187-1255, and this codebase's own already-tested but
// previously unwired src/shared/vol_path.h::VolPathLi).
// ---------------------------------------------------------------------------
class subsurface : public material {
  public:
    using BxDF = DielectricBxDF<double>;

    // sigma_a/sigma_s are ALREADY resolved and scaled by the material's
    // "scale" parameter - see pbrt_flatten.h's flatten(), Subsurface branch,
    // for how a scene's name/sigma_a+sigma_s/defaults get turned into these.
    subsurface(double eta, const double sigma_a[3], const double sigma_s[3], double g)
        : eta_(eta), table_(100, 64) {
        ComputeBeamDiffusionBSSRDF(g, eta, &table_);
        for (int c = 0; c < 3; ++c) {
            sigma_a_[c] = sigma_a[c];
            sigma_s_[c] = sigma_s[c];
        }
        bssrdf_ = TabulatedBSSRDF(sigma_a_, sigma_s_, &table_, 3);
        // Sw (pbrt-v4's exit-interface BSDF) is the same NormalizedFresnelBxDF
        // at every exit point this material's BSSRDF ever samples, so one
        // instance is built once here and shared rather than allocated fresh
        // on every bounce.
        exit_bsdf_ = std::make_shared<normalized_fresnel>(eta);
    }

    // table_ owns the profile grid and bssrdf_ points INTO it (see
    // TabulatedBSSRDF's own non-owning-pointer design in bssrdf.h) - copying
    // or moving a `subsurface` would leave bssrdf_ pointing at the source
    // object's table_, not its own. Every use in this codebase goes through
    // make_shared<subsurface>(...) held by shared_ptr<material>, which never
    // copies the pointee, so this is simply closing off a foot-gun that
    // should never be exercised in practice.
    subsurface(const subsurface&) = delete;
    subsurface& operator=(const subsurface&) = delete;

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        auto ctx = MaterialContext<double>::from_hit(rec, r_in);
        BxDF bxdf{ eta_ };
        vec3 in_dir = unit_vector(r_in.direction());
        auto res = bxdf.sample(in_dir.x(), in_dir.y(), in_dir.z(),
                               ctx.nx, ctx.ny, ctx.nz,
                               ctx.front_face, random_double());
        srec.attenuation     = color(res.r, res.g, res.b);
        srec.pdf_ptr         = nullptr;
        srec.skip_pdf        = true;
        srec.skip_pdf_ray    = ray(rec.p, vec3(res.wo_x, res.wo_y, res.wo_z), r_in.time());
        srec.eta             = res.is_transmission ? (double)res.eta : 1.0;
        srec.is_transmission = res.is_transmission;
        return true;
    }

    // See material::is_shadow_transmissive()'s comment - matches
    // optix_anyhit_shadow.h's MaterialType::Dielectric skip: this material's
    // entry interface IS a dielectric, so it should not block shadow rays
    // outright any more than `class dielectric` does.
    bool is_shadow_transmissive(const hit_record&) const override { return true; }

    const subsurface* as_subsurface(const hit_record&) const override { return this; }

    double get_ior() const { return eta_; }
    const TabulatedBSSRDF& get_bssrdf() const { return bssrdf_; }
    const shared_ptr<material>& get_exit_bsdf() const { return exit_bsdf_; }

  private:
    double eta_;
    double sigma_a_[3];
    double sigma_s_[3];
    BSSRDFTable table_;              // owns the profile grid; bssrdf_ points into it
    TabulatedBSSRDF bssrdf_;
    shared_ptr<material> exit_bsdf_; // cached normalized_fresnel(eta_) -- Sw
};


// ---------------------------------------------------------------------------
// measured -- pbrt-v4 MeasuredMaterial (real gonioreflectometer-measured BRDF,
// e.g. automotive paint scans from the RGL/Dupuy&Jakob "A Practical Guide to
// Fitting BRDFs" .bsdf format)
//
// The actual warp-sampling machinery (PiecewiseLinear2D<N>, ported faithfully
// from pbrt-v4 util/sampling.h) and the BRDF evaluation built on top of it
// (MeasuredBxDF::f/sample_f/pdf, ported from pbrt-v4 bxdfs.cpp) already live
// in src/shared/measured_bxdf.h - this class only has to feed it real data
// (src/shared/measured_bxdf_loader.h's .bsdf tensor-file reader, cached by
// filename so a file referenced by several materials - e.g. this loader's
// sportscar test scene, where ilm_l3_37_metallic_spec.bsdf is shared by 4
// materials - is only ever parsed once) and drive it from scatter().
//
// A measured BRDF is a genuine glossy reflection lobe: MeasuredBxDF::
// sample_f VNDF-importance-samples the tabulated half-vector distribution,
// it is never a delta distribution the way dielectric/subsurface's specular
// interfaces are. It therefore follows THIS file's conductor/principled
// shape - skip_pdf=true, with sample_f's returned (fr,fg,fb) already being
// the full f*|cos(theta_i)|/pdf throughput ratio, baked in exactly the way
// ConductorBxDF's VNDF sampling bakes in G/G1 - not dielectric/subsurface's
// Fresnel-split-then-special-case pattern, which exists to handle a
// refractive boundary this material doesn't have. scattering_pdf() is
// implemented for API parity with conductor/principled/rough_metal (and for
// this loader's own Sample_f/PDF-consistency tests) even though this
// codebase's current MIS scheme does not call it while skip_pdf is set -
// see camera.h's "Specular bounce: no NEE" branch.
//
// R/G/B are queried from the tensor's spectral interpolant at three fixed
// wavelengths (612/549/465nm, approximating the sRGB primaries) rather than
// pbrt-v4's stochastic hero-wavelength spectral sampling - this is the same
// convention src/shared/materials.h's (currently unwired) MeasuredMaterial
// already established for this exact BxDF class, reused here rather than
// inventing a second, inconsistent one.
// ---------------------------------------------------------------------------
class measured : public material {
  public:
    using BxDF = MeasuredBxDF<double>;

    // filename must already be a resolved, load-tested path - see
    // pbrt_flatten.h's Material::measuredFilename comment. The actual load
    // happens through the same process-wide cache pbrt_load.h's scene-
    // loading pass already primed, so constructing a `measured` for a file
    // already seen (the common case: several materials naming the same
    // .bsdf) is a map lookup, not a re-parse of several megabytes of binary.
    explicit measured(const std::string& filename) {
        std::string error;
        data_ = measured_bxdf_io::GetMeasuredBRDFDataCached(filename, error);
    }

    // False when construction failed to obtain real data (missing/corrupt
    // file, or one pbrt_load.h already rejected). Callers - specifically
    // pbrt_cpu_builder.h's makeMaterial() - use this to fall back to a
    // plain diffuse material instead of holding onto a `measured` whose
    // scatter() can only ever return false (rendering the surface pure
    // black, which looks like a bug rather than the documented fallback
    // every other unsupported/failed material gets).
    bool loaded() const { return data_ != nullptr; }

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        if (!data_) return false;

        auto ctx = MaterialContext<double>::from_hit(rec, r_in);
        BxDF bxdf(data_.get(), kLambdaR, kLambdaG, kLambdaB);
        auto frame = ShadingFrame<double>::from_normal(ctx.nx, ctx.ny, ctx.nz);

        double wo_x, wo_y, wo_z;
        frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wo_x, wo_y, wo_z);

        double wi_x, wi_y, wi_z, fr, fg, fb, sampled_pdf;
        const bool ok = bxdf.sample_f(wo_x, wo_y, wo_z,
                                      static_cast<float>(random_double()),
                                      static_cast<float>(random_double()),
                                      wi_x, wi_y, wi_z, fr, fg, fb, sampled_pdf);
        if (!ok) return false;

        double wd_x, wd_y, wd_z;
        frame.to_world(wi_x, wi_y, wi_z, wd_x, wd_y, wd_z);
        // pbrt-v4 PathIntegrator: beta *= bs->f * AbsDot(wi, n) / bs->pdf. This used to be the bare f (sampled_pdf was computed and dropped), which
        // made a measured sphere in a white furnace read 7.8/8.1/3.5 (a blue table), 12 (a metallic one) or 0.34 (white paper) instead of
        // 0.1/0.2/0.4, 0.72 and 1.0 - on the CPU and both OptiX backends alike (see pbrt_scenes/measured-furnace.pbrt).
        srec.attenuation  = color(fr, fg, fb) * (std::fabs(wi_z) / sampled_pdf);
        srec.pdf_ptr      = nullptr;
        srec.skip_pdf     = true;
        srec.skip_pdf_ray = ray(rec.p, unit_vector(vec3(wd_x, wd_y, wd_z)), r_in.time());
        return true;
    }

    double scattering_pdf(const ray& r_in, const hit_record& rec,
                          const ray& scattered) const override {
        if (!data_) return 0.0;

        auto ctx = MaterialContext<double>::from_hit(rec, r_in);
        BxDF bxdf(data_.get(), kLambdaR, kLambdaG, kLambdaB);
        auto frame = ShadingFrame<double>::from_normal(ctx.nx, ctx.ny, ctx.nz);

        double wo_x, wo_y, wo_z;
        frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wo_x, wo_y, wo_z);

        vec3 out_dir = unit_vector(scattered.direction());
        double wi_x, wi_y, wi_z;
        frame.to_local(out_dir.x(), out_dir.y(), out_dir.z(), wi_x, wi_y, wi_z);

        return bxdf.pdf(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
    }

    const MeasuredBRDFData* get_data() const { return data_.get(); }

  private:
    // sRGB-primary approximations - see this class's own comment above.
    static constexpr double kLambdaR = 612.0;
    static constexpr double kLambdaG = 549.0;
    static constexpr double kLambdaB = 465.0;

    shared_ptr<const MeasuredBRDFData> data_;
};


