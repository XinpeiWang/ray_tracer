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


class scatter_record {
  public:
    color attenuation;
    shared_ptr<pdf> pdf_ptr;
    bool skip_pdf;
    ray skip_pdf_ray;
};


class material {
  public:
    virtual ~material() = default;

    virtual color emitted(
        const ray& r_in, const hit_record& rec, double u, double v, const point3& p
    ) const {
        return color(0,0,0);
    }

    virtual bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec) const {
        return false;
    }

    virtual double scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered)
    const {
        return 0;
    }
};


class lambertian : public material {
  public:
    lambertian(const color& albedo) : tex(make_shared<solid_color>(albedo)) {}
    lambertian(shared_ptr<texture> tex) : tex(tex) {}

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec) const override {
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
    metal(const color& albedo, double fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec) const override {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        reflected = unit_vector(reflected) + (fuzz * random_unit_vector());

        srec.attenuation = albedo;
        srec.pdf_ptr = nullptr;
        srec.skip_pdf = true;
        srec.skip_pdf_ray = ray(rec.p, reflected, r_in.time());

        return true;
    }

  private:
    color albedo;
    double fuzz;
};


// Exact Fresnel — shared implementation used by both CPU and GPU.
// See src/shared/fresnel.h (mirrors pbrt-v4 PBRT_CPU_GPU FrDielectric).
#include "../shared/fresnel.h"


class dielectric : public material {
  public:
    dielectric(double refraction_index) : refraction_index(refraction_index) {}

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec) const override {
        srec.attenuation = color(1.0, 1.0, 1.0);
        srec.pdf_ptr = nullptr;
        srec.skip_pdf = true;
        double ri = rec.front_face ? (1.0/refraction_index) : refraction_index;

        vec3 unit_direction = unit_vector(r_in.direction());
        double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;
        vec3 direction;

        // FrDielectric expects eta = eta_t/eta_i, but ri = eta_i/eta_t (used by refract()).
        // Pass the reciprocal so the Fresnel equations use the correct ratio.
        if (cannot_refract || reflectance(cos_theta, 1.0/ri) > random_double())
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, ri);

        srec.skip_pdf_ray = ray(rec.p, direction, r_in.time());
        return true;
    }

    // Accessor for serialization
    double get_refraction_index() const { return refraction_index; }

  private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    double refraction_index;

    static double reflectance(double cos_theta_i, double eta) {
        return FrDielectric(cos_theta_i, eta);
    }
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

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec) const override {
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


// ---------------------------------------------------------------------------
// rough_metal -- GGX microfacet BRDF (pbrt-v4 TrowbridgeReitzDistribution)
// Physically-based rough conductor; replaces simple fuzz-sphere metal for
// accurate anisotropic highlights and energy conservation.
// roughness in [0,1]: 0 = mirror, 1 = fully diffuse-like rough
// ---------------------------------------------------------------------------
#include "../shared/microfacet.h"

class rough_metal : public material {
  public:
    rough_metal(const color& albedo, double roughness)
        : albedo(albedo),
          alpha(TrowbridgeReitz<double>::RoughnessToAlpha(
              std::fmax(roughness, 1e-4))) {}

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec) const override {
        // Build local shading frame: z_axis = normal, arbitrary x/y
        vec3 normal = rec.normal;
        vec3 up = std::fabs(normal.x()) > 0.9 ? vec3(0,1,0) : vec3(1,0,0);
        vec3 tangent   = unit_vector(cross(up, normal));
        vec3 bitangent = cross(normal, tangent);

        // Transform incident direction to local frame
        vec3 wi_world = unit_vector(-r_in.direction());
        double wi_x = dot(wi_world, tangent);
        double wi_y = dot(wi_world, bitangent);
        double wi_z = dot(wi_world, normal);

        if (wi_z <= 0.0) return false;  // ray from inside -- no scatter

        // Sample microfacet normal using VNDF (pbrt-v4 Sample_wm)
        TrowbridgeReitz<double> dist(alpha, alpha);
        double wm_x, wm_y, wm_z;
        dist.Sample_wm(wi_x, wi_y, wi_z,
                       random_double(), random_double(),
                       wm_x, wm_y, wm_z);

        // Reflect wi about the sampled microfacet normal
        double dot_wi_wm = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
        double wo_x = 2.0*dot_wi_wm*wm_x - wi_x;
        double wo_y = 2.0*dot_wi_wm*wm_y - wi_y;
        double wo_z = 2.0*dot_wi_wm*wm_z - wi_z;

        if (wo_z <= 0.0) return false;  // reflection below surface

        // VNDF sampling weight: f * cosI / pdf = G(wo,wi) / G1(wi) (Fresnel=albedo)
        // (same cancellation as pbrt-v4 ConductorBxDF::Sample_f when F=constant)
        double G1_wi  = dist.G1(wi_x, wi_y, wi_z);
        double G_wo_wi = dist.G(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
        double weight = (G1_wi > 1e-8) ? G_wo_wi / G1_wi : 0.0;

        // Transform back to world frame
        vec3 scatter_dir = wo_x * tangent + wo_y * bitangent + wo_z * normal;

        srec.attenuation  = albedo * weight;
        srec.pdf_ptr      = nullptr;
        srec.skip_pdf     = true;
        srec.skip_pdf_ray = ray(rec.p, unit_vector(scatter_dir), r_in.time());
        return true;
    }

    double get_roughness() const { return alpha * alpha; }  // alpha^2 = roughness
    const color& get_albedo() const { return albedo; }

  private:
    color  albedo;
    double alpha;   // GGX alpha = sqrt(roughness)
};


// ---------------------------------------------------------------------------
// conductor -- GGX microfacet BRDF with complex Fresnel (pbrt-v4 ConductorBxDF)
// Uses real metal optical constants (η + i·k) per RGB channel for physically
// accurate angle-varying, wavelength-dependent reflectance.
// roughness in [0,1]: 0 = mirror, 1 = fully rough
// ---------------------------------------------------------------------------
class conductor : public material {
  public:
    // Construct from explicit per-channel optical constants
    conductor(double eta_r, double eta_g, double eta_b,
              double k_r,   double k_g,   double k_b,
              double roughness)
        : eta_r(eta_r), eta_g(eta_g), eta_b(eta_b),
          k_r(k_r),     k_g(k_g),     k_b(k_b),
          alpha(TrowbridgeReitz<double>::RoughnessToAlpha(
              std::fmax(roughness, 1e-4))) {}

    // Construct from a named preset (e.g. "Au", "Ag", "Al", "Cu")
    conductor(const ConductorPreset& preset, double roughness)
        : eta_r(preset.eta_r), eta_g(preset.eta_g), eta_b(preset.eta_b),
          k_r(preset.k_r),     k_g(preset.k_g),     k_b(preset.k_b),
          alpha(TrowbridgeReitz<double>::RoughnessToAlpha(
              std::fmax(roughness, 1e-4))) {}

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec) const override {
        // Build local shading frame
        vec3 normal    = rec.normal;
        vec3 up        = std::fabs(normal.x()) > 0.9 ? vec3(0,1,0) : vec3(1,0,0);
        vec3 tangent   = unit_vector(cross(up, normal));
        vec3 bitangent = cross(normal, tangent);

        // Incident direction in local frame
        vec3 wi_world = unit_vector(-r_in.direction());
        double wi_x = dot(wi_world, tangent);
        double wi_y = dot(wi_world, bitangent);
        double wi_z = dot(wi_world, normal);

        if (wi_z <= 0.0) return false;  // ray from inside

        // Sample microfacet normal (GGX VNDF, pbrt-v4 Sample_wm)
        TrowbridgeReitz<double> dist(alpha, alpha);
        double wm_x, wm_y, wm_z;
        dist.Sample_wm(wi_x, wi_y, wi_z,
                       random_double(), random_double(),
                       wm_x, wm_y, wm_z);

        // Reflect wi about wm
        double dot_wi_wm = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
        double wo_x = 2.0*dot_wi_wm*wm_x - wi_x;
        double wo_y = 2.0*dot_wi_wm*wm_y - wi_y;
        double wo_z = 2.0*dot_wi_wm*wm_z - wi_z;

        if (wo_z <= 0.0) return false;  // reflected below surface

        // VNDF sampling weight = f * cosI / pdf = F * G(wo,wi) / G1(wi)
        // (pbrt-v4 ConductorBxDF::Sample_f returns f/pdf with this cancellation)
        // G1(wi) and G(wo,wi) use the Smith height-correlated shadowing-masking.
        double G1_wi = dist.G1(wi_x, wi_y, wi_z);
        double G_wo_wi = dist.G(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
        double weight = (G1_wi > 1e-8) ? G_wo_wi / G1_wi : 0.0;

        // FrComplex per RGB channel (pbrt-v4 ConductorBxDF::Sample_f)
        // cos_theta = dot(wi, wm) = dot_wi_wm (both in upper hemisphere)
        double F_r = FrComplex(dot_wi_wm, eta_r, k_r) * weight;
        double F_g = FrComplex(dot_wi_wm, eta_g, k_g) * weight;
        double F_b = FrComplex(dot_wi_wm, eta_b, k_b) * weight;

        vec3 scatter_dir = wo_x*tangent + wo_y*bitangent + wo_z*normal;

        srec.attenuation  = color(F_r, F_g, F_b);
        srec.pdf_ptr      = nullptr;
        srec.skip_pdf     = true;
        srec.skip_pdf_ray = ray(rec.p, unit_vector(scatter_dir), r_in.time());
        return true;
    }

    double get_roughness()  const { return alpha * alpha; }
    color  get_fresnel_normal() const {
        // Approximate F0 (normal incidence reflectance) per channel
        return color(FrComplex(1.0, eta_r, k_r),
                     FrComplex(1.0, eta_g, k_g),
                     FrComplex(1.0, eta_b, k_b));
    }

  private:
    double eta_r, eta_g, eta_b;  // real IOR per RGB channel
    double k_r,   k_g,   k_b;   // extinction coefficient per RGB channel
    double alpha;                 // GGX alpha = sqrt(roughness)
};


// ---------------------------------------------------------------------------
// rough_dielectric -- GGX microfacet BSDF for rough glass (pbrt-v4 RoughDielectricBxDF)
// Samples a microfacet normal from the GGX VNDF, then stochastically reflects
// or refracts based on the Fresnel weight FrDielectric(dot(wi,wm), eta).
// roughness in [0,1]: 0 = perfect smooth glass, 1 = fully diffuse-like frosted glass
// ---------------------------------------------------------------------------
class rough_dielectric : public material {
  public:
    rough_dielectric(double refraction_index, double roughness)
        : ior(refraction_index),
          alpha(TrowbridgeReitz<double>::RoughnessToAlpha(
              std::fmax(roughness, 1e-4))) {}

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec) const override {
        // Build local shading frame: z_axis = shading normal
        vec3 normal = rec.normal;
        bool entering = rec.front_face;
        double eta = entering ? (1.0 / ior) : ior;  // eta_i / eta_t

        vec3 up        = std::fabs(normal.x()) > 0.9 ? vec3(0,1,0) : vec3(1,0,0);
        vec3 tangent   = unit_vector(cross(up, normal));
        vec3 bitangent = cross(normal, tangent);

        // Incident direction in local frame (pointing away from surface)
        vec3 wi_world = unit_vector(-r_in.direction());
        double wi_x = dot(wi_world, tangent);
        double wi_y = dot(wi_world, bitangent);
        double wi_z = dot(wi_world, normal);

        // If ray comes from inside, flip local frame so wi_z > 0
        if (wi_z < 0.0) {
            wi_z = -wi_z; wi_x = -wi_x; wi_y = -wi_y;
        }

        // Sample microfacet normal from GGX VNDF (pbrt-v4 Sample_wm)
        TrowbridgeReitz<double> dist(alpha, alpha);
        double wm_x, wm_y, wm_z;
        dist.Sample_wm(wi_x, wi_y, wi_z,
                       random_double(), random_double(),
                       wm_x, wm_y, wm_z);

        // Fresnel: F = probability of reflection (pbrt-v4 FrDielectric)
        double cos_theta_i = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
        // FrDielectric takes (cos_theta_i, eta_t/eta_i); eta = eta_i/eta_t so pass 1/eta
        double F = FrDielectric(cos_theta_i, 1.0 / eta);

        vec3 scatter_dir;
        if (random_double() < F) {
            // Reflect about microfacet normal
            double dot_wi_wm = cos_theta_i;
            double wo_x = 2.0*dot_wi_wm*wm_x - wi_x;
            double wo_y = 2.0*dot_wi_wm*wm_y - wi_y;
            double wo_z = 2.0*dot_wi_wm*wm_z - wi_z;
            if (wo_z <= 0.0) return false;  // below surface
            scatter_dir = wo_x*tangent + wo_y*bitangent + wo_z*normal;
        } else {
            // Refract about microfacet normal
            // wm must point into same hemisphere as wi
            if (wm_z < 0.0) { wm_x=-wm_x; wm_y=-wm_y; wm_z=-wm_z; }
            // Snell's law in local frame
            double sin2_theta_t = eta*eta * (1.0 - cos_theta_i*cos_theta_i);
            if (sin2_theta_t >= 1.0) {
                // TIR fallback: reflect instead
                double dot_wi_wm = cos_theta_i;
                double wo_x = 2.0*dot_wi_wm*wm_x - wi_x;
                double wo_y = 2.0*dot_wi_wm*wm_y - wi_y;
                double wo_z = 2.0*dot_wi_wm*wm_z - wi_z;
                if (wo_z <= 0.0) return false;
                scatter_dir = wo_x*tangent + wo_y*bitangent + wo_z*normal;
            } else {
                double cos_theta_t = std::sqrt(1.0 - sin2_theta_t);
                // Transmitted direction (pbrt-v4 Refract formula in local frame):
                //   wo = -eta*wi + (eta*dot(wi,wm) - cos_t)*wm
                // wo_z < 0: ray crosses through the surface boundary
                double wo_x = eta*(-wi_x) + (eta*cos_theta_i - cos_theta_t)*wm_x;
                double wo_y = eta*(-wi_y) + (eta*cos_theta_i - cos_theta_t)*wm_y;
                double wo_z = -(eta*wi_z  - (eta*cos_theta_i - cos_theta_t)*wm_z);
                // Transform back to world space (wo_z is negative for transmission)
                scatter_dir = wo_x*tangent + wo_y*bitangent + wo_z*normal;
            }
        }

        srec.attenuation  = color(1.0, 1.0, 1.0);
        srec.pdf_ptr      = nullptr;
        srec.skip_pdf     = true;
        srec.skip_pdf_ray = ray(rec.p, unit_vector(scatter_dir), r_in.time());
        return true;
    }

    double get_ior()       const { return ior; }
    double get_roughness() const { return alpha * alpha; }

  private:
    double ior;    // index of refraction
    double alpha;  // GGX alpha = RoughnessToAlpha(roughness)
};


// ---------------------------------------------------------------------------
// coated_diffuse -- rough dielectric coat over a Lambertian base
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
    coated_diffuse(const color& albedo, double ior, double roughness)
        : albedo(albedo),
          ior(ior),
          alpha(TrowbridgeReitz<double>::RoughnessToAlpha(
              std::fmax(roughness, 1e-4))) {}

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec) const override {
        // Local shading frame
        vec3 n   = rec.normal;
        vec3 up  = std::fabs(n.x()) > 0.9 ? vec3(0,1,0) : vec3(1,0,0);
        vec3 tan = unit_vector(cross(up, n));
        vec3 bit = cross(n, tan);

        // Incident direction (pointing away from surface)
        vec3 wi_w = unit_vector(-r_in.direction());
        double wi_x = dot(wi_w, tan);
        double wi_y = dot(wi_w, bit);
        double wi_z = dot(wi_w, n);
        if (wi_z <= 0.0) return false;

        // ── Coat top interface: GGX VNDF + FrDielectric ──────────────────
        TrowbridgeReitz<double> dist(alpha, alpha);
        double wm_x, wm_y, wm_z;
        dist.Sample_wm(wi_x, wi_y, wi_z,
                       random_double(), random_double(),
                       wm_x, wm_y, wm_z);

        double cos_i = wi_x*wm_x + wi_y*wm_y + wi_z*wm_z;
        // FrDielectric(cos_i, eta_t/eta_i): coat IOR=ior, air IOR=1 -> eta_t/eta_i = ior
        double F_in = FrDielectric(cos_i, ior);

        if (random_double() < F_in) {
            // ── Path A: reflect off coat (glossy specular) ────────────────
            double wo_x = 2.0*cos_i*wm_x - wi_x;
            double wo_y = 2.0*cos_i*wm_y - wi_y;
            double wo_z = 2.0*cos_i*wm_z - wi_z;
            if (wo_z <= 0.0) return false;

            // VNDF weight: G(wo,wi)/G1(wi)
            double G1 = dist.G1(wi_x, wi_y, wi_z);
            double G  = dist.G(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
            double w  = (G1 > 1e-8) ? G / G1 : 0.0;

            srec.attenuation  = color(F_in * w, F_in * w, F_in * w);
            srec.pdf_ptr      = nullptr;
            srec.skip_pdf     = true;
            srec.skip_pdf_ray = ray(rec.p,
                                    unit_vector(wo_x*tan + wo_y*bit + wo_z*n),
                                    r_in.time());
            return true;
        }

        // ── Path B: transmit into layer, diffuse bounce, exit through coat ─
        // Cosine-weighted diffuse direction at the base (Lambertian)
        vec3 diff_dir = rec.normal + random_unit_vector();
        if (diff_dir.near_zero()) diff_dir = rec.normal;
        diff_dir = unit_vector(diff_dir);

        // Fresnel at exit: angle between diff_dir and surface normal
        double cos_out = std::fabs(dot(diff_dir, n));
        // Same IOR (air→coat) -- light exits coat into air: eta_t/eta_i = 1/ior
        double F_out = FrDielectric(cos_out, 1.0 / ior);

        // Throughput: albedo * (1-F_in) * (1-F_out)
        double T_in  = 1.0 - F_in;
        double T_out = 1.0 - F_out;
        srec.attenuation  = albedo * (T_in * T_out);
        srec.pdf_ptr      = nullptr;
        srec.skip_pdf     = true;
        srec.skip_pdf_ray = ray(rec.p, diff_dir, r_in.time());
        return true;
    }

    double get_ior()       const { return ior; }
    double get_roughness() const { return alpha * alpha; }
    const color& get_albedo() const { return albedo; }

  private:
    color  albedo;  // diffuse base colour
    double ior;     // coat index of refraction
    double alpha;   // GGX alpha
};


// ---------------------------------------------------------------------------
// thin_dielectric -- pbrt-v4 ThinDielectricBxDF
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
    thin_dielectric(double ior) : ior(ior) {}

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec) const override {
        double cos_theta = std::fabs(dot(unit_vector(r_in.direction()), rec.normal));
        double R = FrDielectric(cos_theta, ior);
        // Account for internal multiple bounces: R_eff = R + T^2*R/(1-R^2)
        if (R < 1.0) {
            double T = 1.0 - R;
            R += T * T * R / (1.0 - R * R);
        }
        double T_eff = 1.0 - R;

        srec.attenuation  = color(1.0, 1.0, 1.0);
        srec.pdf_ptr      = nullptr;
        srec.skip_pdf     = true;

        if (random_double() < R) {
            // Specular reflection
            srec.skip_pdf_ray = ray(rec.p,
                                    reflect(unit_vector(r_in.direction()), rec.normal),
                                    r_in.time());
        } else {
            // Straight-through transmission (no bending — zero thickness)
            (void)T_eff;
            srec.skip_pdf_ray = ray(rec.p,
                                    unit_vector(r_in.direction()),
                                    r_in.time());
        }
        return true;
    }

    double get_ior() const { return ior; }

  private:
    double ior;
};


#endif
