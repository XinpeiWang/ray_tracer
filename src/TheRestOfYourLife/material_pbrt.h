#pragma once
#include "material_base.h"

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

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        auto ctx  = MaterialContext<double>::from_hit(rec, r_in);
        double ex = do_regularize ? regularize_alpha(alpha_x) : alpha_x;
        double ey = do_regularize ? regularize_alpha(alpha_y) : alpha_y;
        BxDF bxdf{ albedo.x(), albedo.y(), albedo.z(), ex, ey };
        auto frame = ShadingFrame<double>::from_normal(ctx.nx, ctx.ny, ctx.nz);

        double wi_x, wi_y, wi_z;
        frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);

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
              double u_roughness, double v_roughness)
        : eta_r(eta_r), eta_g(eta_g), eta_b(eta_b),
          k_r(k_r),     k_g(k_g),     k_b(k_b),
          alpha_x(TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(u_roughness, 1e-4))),
          alpha_y(TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(v_roughness, 1e-4))) {}

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

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        auto ctx   = MaterialContext<double>::from_hit(rec, r_in);
        double ex = do_regularize ? regularize_alpha(alpha_x) : alpha_x;
        double ey = do_regularize ? regularize_alpha(alpha_y) : alpha_y;
        BxDF bxdf{ eta_r, eta_g, eta_b, k_r, k_g, k_b, ex, ey };
        auto frame = ShadingFrame<double>::from_normal(ctx.nx, ctx.ny, ctx.nz);

        double wi_x, wi_y, wi_z;
        frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);

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

    rough_dielectric(double refraction_index, double u_roughness, double v_roughness)
        : ior(refraction_index),
          alpha_x(TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(u_roughness, 1e-4))),
          alpha_y(TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(v_roughness, 1e-4))) {}

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        return BxDF{ ior, alpha_x, alpha_y };
    }

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        auto ctx   = MaterialContext<double>::from_hit(rec, r_in);
        double ex = do_regularize ? regularize_alpha(alpha_x) : alpha_x;
        double ey = do_regularize ? regularize_alpha(alpha_y) : alpha_y;
        BxDF bxdf{ ior, ex, ey };
        auto frame = ShadingFrame<double>::from_normal(ctx.nx, ctx.ny, ctx.nz);

        double eta = ctx.front_face ? (1.0 / ior) : ior;

        double wi_x, wi_y, wi_z;
        frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);
        if (wi_z < 0.0) { wi_z = -wi_z; wi_x = -wi_x; wi_y = -wi_y; }

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

    double get_ior()       const { return ior; }
    double get_roughness() const { return alpha_x * alpha_x; }

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
        : albedo(albedo), ior(ior) {
        double a = TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(roughness, 1e-4));
        alpha_x = alpha_y = a;
    }

    coated_diffuse(const color& albedo, double ior, double u_roughness, double v_roughness)
        : albedo(albedo), ior(ior),
          alpha_x(TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(u_roughness, 1e-4))),
          alpha_y(TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(v_roughness, 1e-4))) {}

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        return BxDF{ albedo.x(), albedo.y(), albedo.z(), ior, alpha_x, alpha_y };
    }

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        auto ctx   = MaterialContext<double>::from_hit(rec, r_in);
        double ex = do_regularize ? regularize_alpha(alpha_x) : alpha_x;
        double ey = do_regularize ? regularize_alpha(alpha_y) : alpha_y;
        BxDF bxdf{ albedo.x(), albedo.y(), albedo.z(), ior, ex, ey };
        auto frame = ShadingFrame<double>::from_normal(ctx.nx, ctx.ny, ctx.nz);

        double wi_x, wi_y, wi_z;
        frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);
        if (wi_z <= 0.0) return false;

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

    double get_ior()       const { return ior; }
    double get_roughness() const { return alpha_x * alpha_x; }
    const color& get_albedo() const { return albedo; }

  private:
    color  albedo;
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
                     double coat_ior, double u_roughness, double v_roughness)
        : eta_r(eta_r), eta_g(eta_g), eta_b(eta_b),
          k_r(k_r),     k_g(k_g),     k_b(k_b),
          coat_ior(coat_ior),
          alpha_x(TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(u_roughness, 1e-4))),
          alpha_y(TrowbridgeReitz<double>::RoughnessToAlpha(std::fmax(v_roughness, 1e-4))) {}

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
        auto frame = ShadingFrame<double>::from_normal(ctx.nx, ctx.ny, ctx.nz);

        double wi_x, wi_y, wi_z;
        frame.to_local(ctx.wo_x, ctx.wo_y, ctx.wo_z, wi_x, wi_y, wi_z);
        if (wi_z <= 0.0) return false;

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

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        return BxDF{ R.x(), R.y(), R.z(), T.x(), T.y(), T.z() };
    }

    bool scatter(const ray& r_in, const hit_record& rec,
                 scatter_record& srec, bool do_regularize = false) const override {
        // Probabilities proportional to max component (pbrt-v4 pattern)
        double pr = std::fmax(std::fmax(R.x(), R.y()), R.z());
        double pt = std::fmax(std::fmax(T.x(), T.y()), T.z());
        if (pr + pt <= 0.0) return false;

        if (random_double() < pr / (pr + pt)) {
            // Diffuse reflection: cosine-weighted same hemisphere as normal
            srec.attenuation  = R;
            srec.pdf_ptr      = make_shared<cosine_pdf>(rec.normal);
            srec.skip_pdf     = false;
        } else {
            // Diffuse transmission: cosine-weighted opposite hemisphere
            // Use cosine_pdf around -normal so MIS and PDF evaluation are correct,
            // matching pbrt-v4 where transmission PDF = pt/(pr+pt) * cos/pi
            srec.attenuation  = T;
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
        double pr = std::fmax(std::fmax(R.x(), R.y()), R.z());
        double pt = std::fmax(std::fmax(T.x(), T.y()), T.z());
        if (pr + pt <= 0.0) return 0.0;

        double cos_theta = dot(rec.normal, unit_vector(scattered.direction()));
        if (cos_theta > 0.0)
            return (pr / (pr + pt)) * (cos_theta / pi);   // reflection
        else
            return (pt / (pr + pt)) * (-cos_theta / pi);  // transmission
    }

    color get_reflectance()   const { return R; }
    color get_transmittance() const { return T; }

  private:
    color R;  // reflectance (same-hemisphere diffuse)
    color T;  // transmittance (opposite-hemisphere diffuse)
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
        // Stochastic selection (pbrt-v4 MixMaterial)
        if (random_double() >= w)
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
        double pa = mat_a->scattering_pdf(r_in, rec, scattered);
        double pb = mat_b->scattering_pdf(r_in, rec, scattered);
        return (1.0 - w) * pa + w * pb;
    }

    shared_ptr<material> get_mat_a()   const { return mat_a; }
    shared_ptr<material> get_mat_b()   const { return mat_b; }
    shared_ptr<texture>  get_weight()  const { return weight_tex; }

  private:
    shared_ptr<material> mat_a;
    shared_ptr<material> mat_b;
    shared_ptr<texture>  weight_tex;
};


