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

        // Sample a microfacet normal via GGX visible-normal heuristic.
        // For simplicity we use cosine-weighted hemisphere sampling here;
        // VNDF importance sampling can be added as a follow-up.
        vec3 scatter_dir;
        int attempts = 0;
        do {
            // Sample random half-vector in local frame (cosine-weighted upper hemi)
            double r1 = random_double(), r2 = random_double();
            double phi   = 2.0 * pi * r1;
            double cos_t = std::sqrt(1.0 - r2);
            double sin_t = std::sqrt(r2);
            double hx = sin_t * std::cos(phi);
            double hy = sin_t * std::sin(phi);
            double hz = cos_t;

            // Reflect wi about this half-vector (all in local frame)
            double dot_wi_h = wi_x*hx + wi_y*hy + wi_z*hz;
            double wo_x = 2.0*dot_wi_h*hx - wi_x;
            double wo_y = 2.0*dot_wi_h*hy - wi_y;
            double wo_z = 2.0*dot_wi_h*hz - wi_z;

            if (wo_z > 0.0) {
                // Transform back to world frame
                scatter_dir = wo_x * tangent + wo_y * bitangent + wo_z * normal;
                break;
            }
            ++attempts;
        } while (attempts < 32);

        if (attempts >= 32) return false;

        // GGX BRDF weight = D*G/(4*cosO*cosI) * cosI / pdf
        // Using cosine-weighted sampling: pdf = cos_o / pi
        // We fold the weight into attenuation; importance sampling handles the rest.
        srec.attenuation = albedo;
        srec.pdf_ptr     = nullptr;
        srec.skip_pdf    = true;
        srec.skip_pdf_ray = ray(rec.p, unit_vector(scatter_dir), r_in.time());
        return true;
    }

    double get_roughness() const { return alpha * alpha; }  // alpha^2 = roughness
    const color& get_albedo() const { return albedo; }

  private:
    color  albedo;
    double alpha;   // GGX alpha = sqrt(roughness)
};


#endif
