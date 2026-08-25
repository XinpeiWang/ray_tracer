#pragma once
#include "material_base.h"
#include "../shared/microfacet.h"


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
        srec.attenuation = tex->value_diff(rec.u, rec.v, rec.p,
                                            rec.dudx, rec.dvdx, rec.dudy, rec.dvdy);
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


// Also implements dispersive_material (material_base.h) - the interface
// camera.h's ray_color_spectral() dispatches through for every dispersive
// material kind via one shared hook (material::as_dispersive()), rather
// than a hook per concrete type. See dispersive_material's own comment.
class dielectric : public material, public dispersive_material {
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

    // Named factory for dispersive (wavelength-dependent IOR) glass, rather
    // than a same-arity (double, double) constructor overload disambiguated
    // only by a trailing bool tag and a call-site comment - a bare bool
    // gives the type system nothing to enforce, so a future 3-argument
    // constructor whose third parameter is itself numeric could silently
    // bind here instead via bool's implicit conversion. Mirrors
    // bvh_light_sampler.h's make_leaf()/make_interior() static factories,
    // the same pattern already established elsewhere in this codebase.
    //
    // eta_d/abbe_number are the artist-facing pair real glass catalogs
    // quote (e.g. crown glass ~=1.52/59, flint glass ~=1.62/36) - see the
    // private constructor below for the Cauchy-coefficient derivation.
    static shared_ptr<dielectric> make_dispersive(double eta_d, double abbe_number) {
        return shared_ptr<dielectric>(new dielectric(eta_d, abbe_number));
    }

    BxDF get_bxdf(const MaterialContext<double>& ctx) const {
        return BxDF{ refraction_index };
    }

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool do_regularize = false) const override {
        return scatter_impl(r_in, rec, srec, refraction_index);
    }

    // Spectral-aware variant: computes eta from the path's hero wavelength
    // via CauchyEta() instead of the flat refraction_index, when this
    // instance was constructed dispersive - otherwise identical to
    // scatter() (ignores lambda_nm). do_regularize is accepted only to
    // satisfy dispersive_material's shared interface (material_base.h) -
    // smooth dielectric has no roughness to regularize, so it's unused here.
    // eta_out reports the eta actually used back to the caller (camera.h),
    // so a later scattering_pdf_dispersive() call for the same bounce - not
    // that this class ever has one, see the comment on that method's
    // default in dispersive_material - never has to re-derive it. See
    // camera.h's ray_color_spectral() for the only call site.
    bool scatter_dispersive(const ray& r_in, const hit_record& rec,
                             scatter_record& srec, float lambda_nm,
                             bool do_regularize, double& eta_out) const override {
        (void)do_regularize;
        double eta = dispersive_ ? CauchyEta((double)lambda_nm, cauchy_A_, cauchy_B_)
                                  : refraction_index;
        eta_out = eta;
        return scatter_impl(r_in, rec, srec, eta);
    }

    // True when this instance was built via the (eta_d, abbe_number)
    // dispersive constructor - see that constructor's own comment. Kept as
    // a public accessor (used by tests) even though ray_color_spectral()
    // itself no longer needs to re-check it: as_dispersive() below only
    // ever returns non-null when this is already true - see
    // dispersive_material's own comment on that "non-null means yes"
    // contract.
    bool is_dispersive() const { return dispersive_; }

    // Accessor for serialization
    double get_refraction_index() const { return refraction_index; }

    // See material::is_shadow_transmissive()'s comment - matches
    // optix_anyhit_shadow.h's MaterialType::Dielectric skip.
    bool is_shadow_transmissive(const hit_record&) const override { return true; }

    // See material::as_dispersive()'s comment. `this` (not nullptr) only
    // when built via make_dispersive() above - lets ray_color_spectral()
    // find this instance through a wrapper material (mix_material) the
    // same way as_subsurface() already does for BSSRDF.
    const dispersive_material* as_dispersive(const hit_record&) const override {
        return dispersive_ ? this : nullptr;
    }

  private:
    // Dispersive glass: wavelength-dependent IOR via the two-term Cauchy
    // formula (fresnel.h's CauchyEta). eta_d is the index at the sodium D
    // line (589.3nm); A/B are derived once here via the standard closed
    // form (F/C/D lines at 486.1/656.3/589.3nm, lambda in micrometers):
    //   B = (eta_d - 1) / (abbe * (1/lambda_F^2 - 1/lambda_C^2))
    //   A = eta_d - B / lambda_D^2
    // `refraction_index` (used by the ordinary flat-IOR scatter() path,
    // i.e. --spectral off or a non-dispersive-lookalike caller) is set to
    // eta_d so both paths agree at the reference wavelength. Only reachable
    // via ray_color_spectral()'s as_dispersive() dispatch to
    // scatter_dispersive() below - see that function's own comment for why
    // this needed a new dispersive_material method rather than a
    // material::scatter() signature change. Private - construct via
    // make_dispersive() above.
    dielectric(double eta_d, double abbe_number)
        : refraction_index(eta_d), dispersive_(true) {
        CauchyCoefficientsFromAbbe(eta_d, abbe_number, cauchy_A_, cauchy_B_);
    }

    bool scatter_impl(const ray& r_in, const hit_record& rec, scatter_record& srec,
                       double eta) const {
        auto ctx = MaterialContext<double>::from_hit(rec, r_in);
        BxDF bxdf{ eta };
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

    double refraction_index;
    color transmission_filter = color(1, 1, 1);
    bool dispersive_ = false;
    double cauchy_A_ = 0.0, cauchy_B_ = 0.0;
};


class diffuse_light : public material {
  public:
    // point_sample: bypasses value_diff()'s EWA/mip filtering in favor of a
    // plain bilinear tap at u,v (matches pbrt-v4's own DiffuseAreaLight
    // image lookup, which point-samples too). Opt-in, not the class default:
    // scoped to pbrt AreaLightSource "diffuse" "filename" lights specifically
    // (pbrt_cpu_builder.h passes true there) because that's the one call
    // site the flag was added for; mesh.h's pre-existing map_Ke emissive
    // textures pass false (the default) to keep the real anisotropic
    // filtering they already had.
    diffuse_light(shared_ptr<texture> tex, bool two_sided = false, bool point_sample = false)
        : tex(tex), two_sided(two_sided), point_sample(point_sample) {}
    diffuse_light(const color& emit, bool two_sided = false)
        : tex(make_shared<solid_color>(emit)), two_sided(two_sided) {}

    color emitted(const ray& r_in, const hit_record& rec, double u, double v, const point3& p)
    const override {
        if (!rec.front_face && !two_sided)
            return color(0,0,0);
        if (point_sample)
            return tex->value(u, v, p);
        return tex->value_diff(u, v, p, rec.dudx, rec.dvdx, rec.dudy, rec.dvdy);
    }

    shared_ptr<texture> get_texture() const { return tex; }
    bool is_two_sided() const { return two_sided; }

  private:
    shared_ptr<texture> tex;
    bool two_sided;
    bool point_sample = false;
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
// Widens the lobe for near-specular surfaces to reduce variance on subsequent
// bounces. Thin wrapper over the CPU_GPU RegularizeAlpha<T>() in
// src/shared/microfacet.h - the single source of truth for this formula,
// also used by TrowbridgeReitz::Regularize() and the GPU wavefront backend.
inline double regularize_alpha(double a) {
    return RegularizeAlpha(a);
}

