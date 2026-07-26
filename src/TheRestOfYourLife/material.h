#ifndef MATERIAL_H
#define MATERIAL_H
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "hittable.h"
#include "pdf.h"
#include "texture.h"
#include "../shared/fresnel.h"
#include "../shared/conductor_data.h"
#include "../shared/microfacet.h"
#include "../shared/bxdfs.h"
#include "../shared/shading_frame.h"
#include "../shared/material_context.h"


class scatter_record {
  public:
    color attenuation;
    shared_ptr<pdf> pdf_ptr;
    bool skip_pdf;
    ray skip_pdf_ray;
    double eta;           // IOR ratio (1.0 unless refraction; pbrt-v4 bs->eta)
    bool is_transmission; // true when a refraction occurred (drives etaScale in integrator)
};


class material {
  public:
    virtual ~material() = default;

    virtual color emitted(
        const ray& r_in, const hit_record& rec, double u, double v, const point3& p
    ) const {
        return color(0,0,0);
    }

    virtual bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                         bool do_regularize = false) const {
        return false;
    }

    virtual double scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered)
    const {
        return 0;
    }
};


class lambertian : public material {
  public:
    using BxDF = DiffuseBxDF<double>;

    lambertian(const color& albedo) : tex(make_shared<solid_color>(albedo)) {}
    lambertian(shared_ptr<texture> tex) : tex(tex) {}

    // pbrt-v4 pattern: evaluate textures at hit point, return configured BxDF value.
    // On CPU, scatter() still calls tex->value() directly because it also needs
    // the color to fill srec.attenuation.  GPU callers use get_bxdf() with a
    // pre-baked albedo stored in DiffuseBxDF via the r/g/b fields (unused for
    // lambertian since sampling is done by cosine_pdf, not DiffuseBxDF::sample).
    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        // Albedo cannot be texture-evaluated here without a CPU-only tex pointer;
        // lambertian scatter() handles this via srec.attenuation = tex->value(...).
        // The BxDF is used only for scattering_pdf, which doesn't need albedo.
        return BxDF{};
    }

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        srec.attenuation = tex->value(rec.u, rec.v, rec.p);
        srec.pdf_ptr = make_shared<cosine_pdf>(rec.normal);
        srec.skip_pdf = false;
        return true;
    }

    double scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered)
    const override {
        auto cos_theta = dot(rec.normal, unit_vector(scattered.direction()));
        return cos_theta < 0 ? 0 : cos_theta/pi;
    }

    // Accessor for serialization
    shared_ptr<texture> get_texture() const { return tex; }

  private:
    shared_ptr<texture> tex;
};


class metal : public material {
  public:
    using BxDF = MetalBxDF<double>;

    metal(const color& albedo, double fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        return BxDF{ albedo.x(), albedo.y(), albedo.z(), fuzz };
    }

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        auto ctx = MaterialContext<double>::from_hit(rec, r_in);
        auto bxdf = get_bxdf(ctx);
        vec3 in_dir = unit_vector(r_in.direction());
        vec3 fv = random_unit_vector();
        auto res = bxdf.sample(in_dir.x(), in_dir.y(), in_dir.z(),
                               ctx.nx, ctx.ny, ctx.nz,
                               fv.x(), fv.y(), fv.z());
        if (!res.valid) return false;
        srec.attenuation  = color(res.r, res.g, res.b);
        srec.pdf_ptr      = nullptr;
        srec.skip_pdf     = true;
        srec.skip_pdf_ray = ray(rec.p, vec3(res.wo_x, res.wo_y, res.wo_z), r_in.time());
        return true;
    }

    color  get_albedo() const { return albedo; }
    double get_fuzz()   const { return fuzz; }

  private:
    color albedo;
    double fuzz;
};


class dielectric : public material {
  public:
    using BxDF = DielectricBxDF<double>;

    dielectric(double refraction_index) : refraction_index(refraction_index) {}

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        return BxDF{ refraction_index };
    }

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        auto ctx = MaterialContext<double>::from_hit(rec, r_in);
        auto bxdf = get_bxdf(ctx);
        vec3 in_dir = unit_vector(r_in.direction());
        auto res = bxdf.sample(in_dir.x(), in_dir.y(), in_dir.z(),
                               ctx.nx, ctx.ny, ctx.nz,
                               ctx.front_face,
                               random_double());
        srec.attenuation  = color(res.r, res.g, res.b);
        srec.pdf_ptr      = nullptr;
        srec.skip_pdf     = true;
        srec.skip_pdf_ray = ray(rec.p, vec3(res.wo_x, res.wo_y, res.wo_z), r_in.time());
        srec.eta           = res.is_transmission ? (double)res.eta : 1.0;
        srec.is_transmission = res.is_transmission;
        return true;
    }

    // Accessor for serialization
    double get_refraction_index() const { return refraction_index; }

  private:
    double refraction_index;
};


class diffuse_light : public material {
  public:
    diffuse_light(shared_ptr<texture> tex) : tex(tex) {}
    diffuse_light(const color& emit) : tex(make_shared<solid_color>(emit)) {}

    color emitted(const ray& r_in, const hit_record& rec, double u, double v, const point3& p)
    const override {
        if (!rec.front_face)
            return color(0,0,0);
        return tex->value(u, v, p);
    }

    shared_ptr<texture> get_texture() const { return tex; }

  private:
    shared_ptr<texture> tex;
};


class isotropic : public material {
  public:
    isotropic(const color& albedo) : tex(make_shared<solid_color>(albedo)) {}
    isotropic(shared_ptr<texture> tex) : tex(tex) {}

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        srec.attenuation = tex->value(rec.u, rec.v, rec.p);
        srec.pdf_ptr = make_shared<sphere_pdf>();
        srec.skip_pdf = false;
        return true;
    }

    double scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered)
    const override {
        return 1 / (4 * pi);
    }

  private:
    shared_ptr<texture> tex;
};


// Shared helper: regularize a GGX alpha value (pbrt-v4 Regularize() semantics).
// Widens the lobe for near-specular surfaces to reduce variance on subsequent bounces.
inline double regularize_alpha(double a) {
    if (a < 0.3) {
        a *= 2.0;
        if (a < 0.1) a = 0.1;
        else if (a > 0.3) a = 0.3;
    }
    return a;
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
        : albedo(albedo),
          alpha(TrowbridgeReitz<double>::RoughnessToAlpha(
              std::fmax(roughness, 1e-4))) {}

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        return BxDF{ albedo.x(), albedo.y(), albedo.z(), alpha };
    }

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        auto ctx  = MaterialContext<double>::from_hit(rec, r_in);
        double eff_alpha = do_regularize ? regularize_alpha(alpha) : alpha;
        BxDF bxdf{ albedo.x(), albedo.y(), albedo.z(), eff_alpha };
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

    double get_roughness() const { return alpha * alpha; }
    const color& get_albedo() const { return albedo; }

  private:
    color  albedo;
    double alpha;
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
          k_r(k_r),     k_g(k_g),     k_b(k_b),
          alpha(TrowbridgeReitz<double>::RoughnessToAlpha(
              std::fmax(roughness, 1e-4))) {}

    conductor(const ConductorPreset& preset, double roughness)
        : eta_r(preset.eta_r), eta_g(preset.eta_g), eta_b(preset.eta_b),
          k_r(preset.k_r),     k_g(preset.k_g),     k_b(preset.k_b),
          alpha(TrowbridgeReitz<double>::RoughnessToAlpha(
              std::fmax(roughness, 1e-4))) {}

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        return BxDF{ eta_r, eta_g, eta_b, k_r, k_g, k_b, alpha };
    }

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        auto ctx   = MaterialContext<double>::from_hit(rec, r_in);
        double eff_alpha = do_regularize ? regularize_alpha(alpha) : alpha;
        BxDF bxdf{ eta_r, eta_g, eta_b, k_r, k_g, k_b, eff_alpha };
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

    double get_roughness()  const { return alpha * alpha; }
    color  get_fresnel_normal() const {
        return color(FrComplex(1.0, eta_r, k_r),
                     FrComplex(1.0, eta_g, k_g),
                     FrComplex(1.0, eta_b, k_b));
    }

  private:
    double eta_r, eta_g, eta_b;
    double k_r,   k_g,   k_b;
    double alpha;
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
        : ior(refraction_index),
          alpha(TrowbridgeReitz<double>::RoughnessToAlpha(
              std::fmax(roughness, 1e-4))) {}

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        return BxDF{ ior, alpha };
    }

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        auto ctx   = MaterialContext<double>::from_hit(rec, r_in);
        double eff_alpha = do_regularize ? regularize_alpha(alpha) : alpha;
        BxDF bxdf{ ior, eff_alpha };
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
        srec.attenuation  = color(res.r, res.g, res.b);
        srec.pdf_ptr      = nullptr;
        srec.skip_pdf     = true;
        srec.skip_pdf_ray = ray(rec.p, unit_vector(vec3(wd_x, wd_y, wd_z)), r_in.time());
        return true;
    }

    double get_ior()       const { return ior; }
    double get_roughness() const { return alpha * alpha; }

  private:
    double ior;
    double alpha;
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
        : albedo(albedo),
          ior(ior),
          alpha(TrowbridgeReitz<double>::RoughnessToAlpha(
              std::fmax(roughness, 1e-4))) {}

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        return BxDF{ albedo.x(), albedo.y(), albedo.z(), ior, alpha };
    }

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        auto ctx   = MaterialContext<double>::from_hit(rec, r_in);
        double eff_alpha = do_regularize ? regularize_alpha(alpha) : alpha;
        BxDF bxdf{ albedo.x(), albedo.y(), albedo.z(), ior, eff_alpha };
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
    double get_roughness() const { return alpha * alpha; }
    const color& get_albedo() const { return albedo; }

  private:
    color  albedo;
    double ior;
    double alpha;
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
          coat_ior(coat_ior),
          alpha(TrowbridgeReitz<double>::RoughnessToAlpha(
              std::fmax(coat_roughness, 1e-4))) {}

    coated_conductor(const ConductorPreset& preset, double coat_ior, double coat_roughness)
        : eta_r(preset.eta_r), eta_g(preset.eta_g), eta_b(preset.eta_b),
          k_r(preset.k_r),     k_g(preset.k_g),     k_b(preset.k_b),
          coat_ior(coat_ior),
          alpha(TrowbridgeReitz<double>::RoughnessToAlpha(
              std::fmax(coat_roughness, 1e-4))) {}

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        return BxDF{ eta_r, eta_g, eta_b, k_r, k_g, k_b, coat_ior, alpha };
    }

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        auto ctx   = MaterialContext<double>::from_hit(rec, r_in);
        double eff_alpha = do_regularize ? regularize_alpha(alpha) : alpha;
        BxDF bxdf{ eta_r, eta_g, eta_b, k_r, k_g, k_b, coat_ior, eff_alpha };
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
    double get_coat_roughness() const { return alpha * alpha; }
    color  get_conductor_f0()   const {
        return color(FrComplex(1.0, eta_r, k_r),
                     FrComplex(1.0, eta_g, k_g),
                     FrComplex(1.0, eta_b, k_b));
    }

  private:
    double eta_r, eta_g, eta_b;
    double k_r,   k_g,   k_b;
    double coat_ior;
    double alpha;
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


#endif
