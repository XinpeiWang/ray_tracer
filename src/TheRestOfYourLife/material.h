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


// Exact Fresnel reflectance for dielectric interfaces (pbrt-v4 FrDielectric).
// Returns the fraction of light reflected at an interface with relative IOR eta.
// cos_theta_i is the cosine of the angle of incidence (can be negative for
// rays hitting from inside — the function handles both cases correctly).
inline double FrDielectric(double cos_theta_i, double eta) {
    cos_theta_i = std::fmax(-1.0, std::fmin(1.0, cos_theta_i));

    if (cos_theta_i < 0.0) {
        eta = 1.0 / eta;
        cos_theta_i = -cos_theta_i;
    }

    double sin2_theta_i = 1.0 - cos_theta_i * cos_theta_i;
    double sin2_theta_t = sin2_theta_i / (eta * eta);

    if (sin2_theta_t >= 1.0)
        return 1.0;  // Total internal reflection

    double cos_theta_t = std::sqrt(std::fmax(0.0, 1.0 - sin2_theta_t));  // SafeSqrt: clamp to avoid NaN from float rounding

    double r_parl = (eta * cos_theta_i - cos_theta_t)
                  / (eta * cos_theta_i + cos_theta_t);
    double r_perp = (cos_theta_i - eta * cos_theta_t)
                  / (cos_theta_i + eta * cos_theta_t);

    return (r_parl * r_parl + r_perp * r_perp) / 2.0;
}


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


#endif
