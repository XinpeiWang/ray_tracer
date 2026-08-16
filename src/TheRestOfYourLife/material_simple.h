#pragma once
#include "material_base.h"


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

    // Tinted glass: a colored transmission_filter multiplies the refracted
    // (transmitted) contribution only, leaving reflection clear/white -
    // matches real colored glass (a mirror-like reflection off the surface
    // doesn't pick up the pane's tint; light that actually passes through
    // does). This is a direct, non-volumetric read of the classic OBJ/.mtl
    // spec's own "Tf" (transmission filter) field, not Beer's-law distance
    // absorption: Tf has no associated thickness/distance in the .mtl
    // format, and this codebase's existing volumetric option
    // (MaterialType::DielectricMedium / constant_medium, see that class's
    // own comment) needs a closed, watertight boundary shape with a real
    // entry+exit ray crossing to have any effect at all - true for the
    // sphere-based showcase scenes it was built for, but not for typical
    // stained-glass window geometry (a single flat layer of triangles,
    // confirmed on Sibenik Cathedral's own "staklo*" materials - zero
    // interior distance for a volumetric medium to act over). A flat
    // per-surface tint is both the practical fit for this geometry and the
    // historically accurate one: pre-path-tracing renderers this format
    // targeted applied Tf exactly this way, at the interface, not as a
    // medium.
    dielectric(double refraction_index, const color& transmission_filter)
        : refraction_index(refraction_index), transmission_filter(transmission_filter) {}

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
        srec.attenuation  = res.is_transmission
            ? color(res.r, res.g, res.b) * transmission_filter
            : color(res.r, res.g, res.b);
        srec.pdf_ptr      = nullptr;
        srec.skip_pdf     = true;
        srec.skip_pdf_ray = ray(rec.p, vec3(res.wo_x, res.wo_y, res.wo_z), r_in.time());
        srec.eta           = res.is_transmission ? (double)res.eta : 1.0;
        srec.is_transmission = res.is_transmission;
        return true;
    }

    // Accessor for serialization
    double get_refraction_index() const { return refraction_index; }

    // See material::is_shadow_transmissive()'s comment - matches
    // optix_anyhit_shadow.h's MaterialType::Dielectric skip.
    bool is_shadow_transmissive(const hit_record&) const override { return true; }

  private:
    double refraction_index;
    color transmission_filter = color(1, 1, 1);
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

