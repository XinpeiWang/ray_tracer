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
        if (TrowbridgeReitz<double>(alpha_x, alpha_y).EffectivelySmooth()) {
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

  private:
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
    // Rather than leaving Strategy B's continuation weight wrong, this uses
    // Fresnel evaluated at the fixed view angle (dot(wo, normal), i.e.
    // Schlick-style single-value Fresnel) as a documented approximation --
    // real per-half-vector variation is only exact within a single
    // GGX lobe's variance, and this is a large improvement over the
    // previous skip_pdf=true (zero NEE at any roughness) baseline.
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
        if (TrowbridgeReitz<double>(alpha_x, alpha_y).EffectivelySmooth()) {
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

    double get_roughness()  const { return alpha_x * alpha_x; }
    color  get_fresnel_normal() const {
        return color(FrComplex(1.0, eta_r, k_r),
                     FrComplex(1.0, eta_g, k_g),
                     FrComplex(1.0, eta_b, k_b));
    }

  private:
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
class rough_dielectric : public material {
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

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        return BxDF{ ior, alpha_x, alpha_y };
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
        auto ctx   = MaterialContext<double>::from_hit(rec, r_in);
        double ex = do_regularize ? regularize_alpha(alpha_x) : alpha_x;
        double ey = do_regularize ? regularize_alpha(alpha_y) : alpha_y;
        BxDF bxdf{ ior, ex, ey };
        auto frame = ShadingFrame<double>::from_dpdu(ctx.dpdu_x, ctx.dpdu_y, ctx.dpdu_z, ctx.nx, ctx.ny, ctx.nz);

        double eta = ctx.front_face ? (1.0 / ior) : ior;

        double wi_x, wi_y, wi_z;
        frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);
        if (wi_z < 0.0) { wi_z = -wi_z; wi_x = -wi_x; wi_y = -wi_y; }

        // Branch on the TRUE (unregularized) roughness -- see rough_metal's
        // own comment on why scattering_pdf() forces this.
        if (TrowbridgeReitz<double>(alpha_x, alpha_y).EffectivelySmooth()) {
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
            rec.normal, vec3(ctx.wo_x, ctx.wo_y, ctx.wo_z), eta, alpha_x, alpha_y);
        srec.skip_pdf         = false;
        srec.eta              = eta;
        srec.is_transmission  = true;
        return true;
    }

    // f*cos at an arbitrary queried direction (scattered), for real NEE/MIS.
    // Achromatic (RoughDielectricBxDF is colorless glass) -- attenuation
    // stays the default white srec.attenuation.
    double scattering_pdf(const ray& r_in, const hit_record& rec,
                          const ray& scattered) const override {
        auto ctx   = MaterialContext<double>::from_hit(rec, r_in);
        auto frame = ShadingFrame<double>::from_dpdu(ctx.dpdu_x, ctx.dpdu_y, ctx.dpdu_z, ctx.nx, ctx.ny, ctx.nz);
        double eta = ctx.front_face ? (1.0 / ior) : ior;

        double wi_x, wi_y, wi_z;
        frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);
        if (wi_z < 0.0) { wi_z = -wi_z; wi_x = -wi_x; wi_y = -wi_y; }

        vec3 dir = unit_vector(scattered.direction());
        double wo_x, wo_y, wo_z;
        frame.to_local(dir.x(), dir.y(), dir.z(), wo_x, wo_y, wo_z);

        BxDF bxdf{ ior, alpha_x, alpha_y };
        double f_val = bxdf.f(wi_x, wi_y, wi_z, eta, wo_x, wo_y, wo_z);
        return f_val * std::fabs(wo_z);
    }

    double get_ior()       const { return ior; }
    double get_roughness() const { return alpha_x * alpha_x; }

    // See material::is_shadow_transmissive()'s comment - matches
    // optix_anyhit_shadow.h's MaterialType::RoughDielectric skip.
    bool is_shadow_transmissive(const hit_record&) const override { return true; }

  private:
    double ior;
    double alpha_x, alpha_y;
};


// ---------------------------------------------------------------------------
// coated_diffuse
// Mirrors pbrt-v4 CoatedDiffuseBxDF = LayeredBxDF<DielectricBxDF, DiffuseBxDF>
//
// Physical model (single-bounce, no medium scattering):
//   1. Ray hits coat (top interface, GGX + FrDielectric):
//      a. Reflects with probability F_in  -> attenuation = F_in, specular bounce
//      b. Transmits into layer (1-F_in)
//   2. Lambertian bounce at diffuse base -> cosine-weighted direction
//   3. Attempts to exit through coat again:
//      - Transmits with weight (1-F_out) -> attenuation = albedo * (1-F_in) * (1-F_out)
//      - Otherwise absorbed (energy lost inside layer)
//
// Parameters:
//   albedo     -- diffuse base colour
//   ior        -- coat index of refraction (1.5 = glass-like plastic)
//   roughness  -- GGX roughness of the coat surface [0,1]
// ---------------------------------------------------------------------------
class coated_diffuse : public material {
  public:
    using BxDF = CoatedDiffuseBxDF<double>;

    coated_diffuse(const color& albedo, double ior, double roughness)
        : tex(make_shared<solid_color>(albedo)), ior(ior) {
        double a = TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(roughness, 1e-4));
        alpha_x = alpha_y = a;
    }

    coated_diffuse(const color& albedo, double ior, double u_roughness, double v_roughness,
                   bool remap_roughness = true)
        : tex(make_shared<solid_color>(albedo)), ior(ior),
          alpha_x(roughness_or_alpha(u_roughness, remap_roughness)),
          alpha_y(roughness_or_alpha(v_roughness, remap_roughness)) {}

    // "reflectance" bound to a real Texture (pbrt's own ganesha/barcelona-
    // pavilion "texture reflectance" - see pbrt_flatten::Material::
    // textureFilename's own comment) rather than a flat colour - mirrors
    // lambertian's own two-constructor shape (material_simple.h) exactly.
    coated_diffuse(shared_ptr<texture> tex, double ior, double u_roughness, double v_roughness,
                   bool remap_roughness = true)
        : tex(tex), ior(ior),
          alpha_x(roughness_or_alpha(u_roughness, remap_roughness)),
          alpha_y(roughness_or_alpha(v_roughness, remap_roughness)) {}

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        // No pixel-footprint differentials available from MaterialContext
        // alone (unlike scatter()/scattering_pdf() below, which read them
        // from the full hit_record) - same "no CPU-only tex pointer here"
        // limitation lambertian's own get_bxdf() already documents; nothing
        // in this codebase actually calls coated_diffuse::get_bxdf() today
        // (grepped - only referenced in that comment), so this is a best-
        // effort mirror of scatter()'s real per-point lookup, not a load-
        // bearing path.
        color albedo = tex->value(ctx.u, ctx.v, point3(ctx.px, ctx.py, ctx.pz));
        return BxDF{ albedo.x(), albedo.y(), albedo.z(), ior, alpha_x, alpha_y };
    }

    // Real NEE/MIS below the roughness threshold (see rough_metal's own
    // comment). CoatedDiffuseBxDF::f() is a genuinely stochastic,
    // random-walk BSDF value (see that method's own header comment in
    // bxdfs_layered.h) with real per-channel color (from the Lambertian
    // base's albedo) -- unlike RoughDielectricBxDF, it isn't colorless, so
    // scattering_pdf()'s scalar return can't carry the full RGB alone.
    // material::scattering_attenuation()'s signature has no incoming
    // direction to work with (unlike scattering_pdf(), which gets r_in),
    // so unlike diffuse_transmission's exact per-direction R/T split, this
    // uses the material's own base albedo as a fixed representative color
    // (srec.attenuation), with scattering_pdf() below returning
    // luminance(f)*cos / luminance(albedo) so that attenuation*
    // scattering_pdf() reproduces f()'s overall magnitude correctly, tinted
    // by albedo's own hue -- an approximation (the true per-direction
    // Fresnel-driven color shift is averaged away), but a large improvement
    // over the flat-white/no-color alternative, following the same
    // documented-approximation precedent as conductor's fixed-view-angle
    // Fresnel above.
    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        auto ctx   = MaterialContext<double>::from_hit(rec, r_in);
        color albedo = tex->value_diff(rec.u, rec.v, rec.p,
                                        rec.dudx, rec.dvdx, rec.dudy, rec.dvdy);
        double ex = do_regularize ? regularize_alpha(alpha_x) : alpha_x;
        double ey = do_regularize ? regularize_alpha(alpha_y) : alpha_y;
        BxDF bxdf{ albedo.x(), albedo.y(), albedo.z(), ior, ex, ey };
        auto frame = ShadingFrame<double>::from_dpdu(ctx.dpdu_x, ctx.dpdu_y, ctx.dpdu_z, ctx.nx, ctx.ny, ctx.nz);

        double wi_x, wi_y, wi_z;
        frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);
        if (wi_z <= 0.0) return false;

        if (TrowbridgeReitz<double>(alpha_x, alpha_y).EffectivelySmooth()) {
            auto res = bxdf.sample_local(wi_x, wi_y, wi_z,
                                         random_double(), random_double(),
                                         random_double(),
                                         random_double(), random_double());
            if (!res.valid) return false;

            double wd_x, wd_y, wd_z;
            frame.to_world(res.wo_x, res.wo_y, res.wo_z, wd_x, wd_y, wd_z);
            srec.attenuation  = color(res.r, res.g, res.b);
            srec.pdf_ptr      = nullptr;
            srec.skip_pdf     = true;
            srec.skip_pdf_ray = ray(rec.p, unit_vector(vec3(wd_x, wd_y, wd_z)), r_in.time());
            return true;
        }

        srec.attenuation = albedo;
        srec.pdf_ptr      = make_shared<ggx_reflection_pdf>(
            rec.normal, vec3(ctx.wo_x, ctx.wo_y, ctx.wo_z), alpha_x, alpha_y);
        srec.skip_pdf     = false;
        return true;
    }

    // luminance(f)*cos at an arbitrary queried direction, normalized by the
    // fixed attenuation's own luminance -- see scatter()'s own comment.
    double scattering_pdf(const ray& r_in, const hit_record& rec,
                          const ray& scattered) const override {
        auto ctx   = MaterialContext<double>::from_hit(rec, r_in);
        color albedo = tex->value_diff(rec.u, rec.v, rec.p,
                                        rec.dudx, rec.dvdx, rec.dudy, rec.dvdy);
        auto frame = ShadingFrame<double>::from_dpdu(ctx.dpdu_x, ctx.dpdu_y, ctx.dpdu_z, ctx.nx, ctx.ny, ctx.nz);
        double wi_x, wi_y, wi_z;
        frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);
        if (wi_z <= 0.0) return 0.0;

        vec3 dir = unit_vector(scattered.direction());
        double wo_x, wo_y, wo_z;
        frame.to_local(dir.x(), dir.y(), dir.z(), wo_x, wo_y, wo_z);
        if (wo_z <= 0.0) return 0.0;

        BxDF bxdf{ albedo.x(), albedo.y(), albedo.z(), ior, alpha_x, alpha_y };
        uint64_t seed0 = coated_seed_from_dir(dir);
        double fr, fg, fb;
        bxdf.f(wi_x, wi_y, wi_z, wo_x, wo_y, wo_z, seed0, seed0 ^ 0xABCDEF1234567890ull, fr, fg, fb);
        double f_lum = (fr + fg + fb) / 3.0;
        double albedo_lum = std::max(1e-6, (albedo.x() + albedo.y() + albedo.z()) / 3.0);
        return f_lum * wo_z / albedo_lum;
    }

    double get_ior()       const { return ior; }
    double get_roughness() const { return alpha_x * alpha_x; }
    // A representative colour, not the real per-point value - callers
    // wanting the latter should evaluate `tex` themselves at a real (u,v,p).
    // No caller currently exists for this accessor at all (grepped); kept
    // by-value (matching material_simple.h's own get_albedo() convention)
    // since a texture-backed material has no single color to return by
    // reference the way a flat one did.
    color get_albedo() const { return tex->value(0.5, 0.5, point3(0, 0, 0)); }

    // Accessor for testing/serialization - mirrors lambertian's own
    // get_texture() (material_simple.h) exactly.
    shared_ptr<texture> get_texture() const { return tex; }

  private:
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

    // See material::is_shadow_transmissive()'s comment - matches
    // optix_anyhit_shadow.h's MaterialType::ThinDielectric skip.
    bool is_shadow_transmissive(const hit_record&) const override { return true; }

  private:
    double ior;
};


// ---------------------------------------------------------------------------
// coated_conductor -- rough dielectric coat over a GGX conductor base
// Mirrors pbrt-v4 CoatedConductorBxDF = LayeredBxDF<DielectricBxDF, ConductorBxDF>
//
// Physical model (single-bounce, no medium scattering):
//   1. Ray hits coat (top interface, GGX + FrDielectric):
//      a. Reflects with probability F_in  -> attenuation = F_in (achromatic coat)
//      b. Transmits into layer (1-F_in) -- reaches conductor
//   2. Conductor micro-facet bounce (GGX VNDF + complex Fresnel FrComplex per RGB):
//      weight_c = FrComplex * G(wo,wi) / G1(wi)
//   3. Attempts to exit through coat again:
//      - Fresnel at exit angle: F_out (FrDielectric)
//      - Transmits with weight (1-F_out)
//      - Total attenuation = weight_c * (1-F_in) * (1-F_out)
//
// Parameters:
//   eta_r/g/b  -- real part of conductor IOR per RGB channel
//   k_r/g/b    -- extinction coefficient per RGB channel
//   coat_ior   -- coat index of refraction (1.5 = glass-like lacquer)
//   coat_roughness -- GGX roughness of coat AND conductor surfaces [0,1]
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
        double a = TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(coat_roughness, 1e-4));
        alpha_x = alpha_y = a;
    }

    coated_conductor(double eta_r, double eta_g, double eta_b,
                     double k_r,   double k_g,   double k_b,
                     double coat_ior, double u_roughness, double v_roughness,
                     bool remap_roughness = true)
        : eta_r(eta_r), eta_g(eta_g), eta_b(eta_b),
          k_r(k_r),     k_g(k_g),     k_b(k_b),
          coat_ior(coat_ior),
          alpha_x(roughness_or_alpha(u_roughness, remap_roughness)),
          alpha_y(roughness_or_alpha(v_roughness, remap_roughness)) {}

    coated_conductor(const ConductorPreset& preset, double coat_ior, double coat_roughness)
        : eta_r(preset.eta_r), eta_g(preset.eta_g), eta_b(preset.eta_b),
          k_r(preset.k_r),     k_g(preset.k_g),     k_b(preset.k_b),
          coat_ior(coat_ior) {
        double a = TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(coat_roughness, 1e-4));
        alpha_x = alpha_y = a;
    }

    coated_conductor(const ConductorPreset& preset, double coat_ior,
                     double u_roughness, double v_roughness)
        : eta_r(preset.eta_r), eta_g(preset.eta_g), eta_b(preset.eta_b),
          k_r(preset.k_r),     k_g(preset.k_g),     k_b(preset.k_b),
          coat_ior(coat_ior),
          alpha_x(TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(u_roughness, 1e-4))),
          alpha_y(TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(v_roughness, 1e-4))) {}

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        return BxDF{ eta_r, eta_g, eta_b, k_r, k_g, k_b, coat_ior, alpha_x, alpha_y };
    }

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        auto ctx   = MaterialContext<double>::from_hit(rec, r_in);
        double ex = do_regularize ? regularize_alpha(alpha_x) : alpha_x;
        double ey = do_regularize ? regularize_alpha(alpha_y) : alpha_y;
        BxDF bxdf{ eta_r, eta_g, eta_b, k_r, k_g, k_b, coat_ior, ex, ey };
        auto frame = ShadingFrame<double>::from_dpdu(ctx.dpdu_x, ctx.dpdu_y, ctx.dpdu_z, ctx.nx, ctx.ny, ctx.nz);

        double wi_x, wi_y, wi_z;
        frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);
        if (wi_z <= 0.0) return false;

        // See coated_diffuse::scatter()'s own comment for the roughness
        // gate + fixed-representative-color rationale; here the
        // representative color is the conductor's own normal-incidence
        // Fresnel (get_conductor_f0()), the natural analog of albedo.
        if (TrowbridgeReitz<double>(alpha_x, alpha_y).EffectivelySmooth()) {
            auto res = bxdf.sample_local(wi_x, wi_y, wi_z,
                                         random_double(), random_double(),
                                         random_double(),
                                         random_double(), random_double());
            if (!res.valid) return false;

            double wd_x, wd_y, wd_z;
            frame.to_world(res.wo_x, res.wo_y, res.wo_z, wd_x, wd_y, wd_z);
            srec.attenuation  = color(res.r, res.g, res.b);
            srec.pdf_ptr      = nullptr;
            srec.skip_pdf     = true;
            srec.skip_pdf_ray = ray(rec.p, unit_vector(vec3(wd_x, wd_y, wd_z)), r_in.time());
            return true;
        }

        srec.attenuation = get_conductor_f0();
        srec.pdf_ptr      = make_shared<ggx_reflection_pdf>(
            rec.normal, vec3(ctx.wo_x, ctx.wo_y, ctx.wo_z), alpha_x, alpha_y);
        srec.skip_pdf     = false;
        return true;
    }

    // luminance(f)*cos at an arbitrary queried direction, normalized by the
    // fixed attenuation's own luminance -- see coated_diffuse::
    // scattering_pdf()'s identical pattern.
    double scattering_pdf(const ray& r_in, const hit_record& rec,
                          const ray& scattered) const override {
        auto ctx   = MaterialContext<double>::from_hit(rec, r_in);
        auto frame = ShadingFrame<double>::from_dpdu(ctx.dpdu_x, ctx.dpdu_y, ctx.dpdu_z, ctx.nx, ctx.ny, ctx.nz);
        double wi_x, wi_y, wi_z;
        frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);
        if (wi_z <= 0.0) return 0.0;

        vec3 dir = unit_vector(scattered.direction());
        double wo_x, wo_y, wo_z;
        frame.to_local(dir.x(), dir.y(), dir.z(), wo_x, wo_y, wo_z);
        if (wo_z <= 0.0) return 0.0;

        BxDF bxdf{ eta_r, eta_g, eta_b, k_r, k_g, k_b, coat_ior, alpha_x, alpha_y };
        uint64_t seed0 = coated_seed_from_dir(dir);
        double fr, fg, fb;
        bxdf.f(wi_x, wi_y, wi_z, wo_x, wo_y, wo_z, seed0, seed0 ^ 0xABCDEF1234567890ull, fr, fg, fb);
        double f_lum = (fr + fg + fb) / 3.0;
        color f0 = get_conductor_f0();
        double f0_lum = std::max(1e-6, (f0.x() + f0.y() + f0.z()) / 3.0);
        return f_lum * wo_z / f0_lum;
    }

    double get_coat_ior()       const { return coat_ior; }
    double get_coat_roughness() const { return alpha_x * alpha_x; }
    color  get_conductor_f0()   const {
        return color(FrComplex(1.0, eta_r, k_r),
                     FrComplex(1.0, eta_g, k_g),
                     FrComplex(1.0, eta_b, k_b));
    }

  private:
    double eta_r, eta_g, eta_b;
    double k_r,   k_g,   k_b;
    double coat_ior;
    double alpha_x, alpha_y;
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
        srec.attenuation  = color(fr, fg, fb);
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


