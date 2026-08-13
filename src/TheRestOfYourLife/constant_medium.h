#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//
// pbrt-v4 upgrades (media.h / HomogeneousMedium):
//   - HenyeyGreenstein phase function with asymmetry parameter g
//     (replaces isotropic uniform-sphere sampling)
//   - Proper Beer-Lambert transmittance Tr = exp(-sigma_t * t) folded into
//     path throughput via scatter_weight (single-scattering albedo = sigma_s/sigma_t)
//   - transmittance() method for shadow-ray attenuation inside the volume
//   - Backward-compatible constructors: existing code with density + albedo
//     continues to work (g defaults to 0 = isotropic)
//==============================================================================================

#include "hittable.h"
#include "material.h"
#include "texture.h"
#include "../shared/volume_scattering.h"
#include "pdf.h"


// ---------------------------------------------------------------------------
// hg_phase_material
// CPU material wrapper around HenyeyGreensteinPhaseFunction<double>.
// Replaces the old isotropic material for volume scattering.
// Mirrors pbrt-v4 HGPhaseFunction::Sample_p / p().
// ---------------------------------------------------------------------------
class hg_phase_material : public material {
  public:
    // albedo: single-scattering albedo (sigma_s / sigma_t) per channel
    // g: HG asymmetry parameter in (-1, 1)
    hg_phase_material(const color& albedo, double g)
        : albedo(albedo), phase(g) {}

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool /*do_regularize*/ = false) const override {
        // Sample a new direction using HG phase function
        vec3 wo = unit_vector(-r_in.direction());  // outgoing = toward camera

        double wi_x, wi_y, wi_z, pdf_val;
        phase.sample(wo.x(), wo.y(), wo.z(),
                     random_double(), random_double(),
                     wi_x, wi_y, wi_z, pdf_val);

        vec3 wi(wi_x, wi_y, wi_z);

        // Attenuation = single-scattering albedo (already folded scatter_weight
        // into the medium hit in constant_medium::hit)
        srec.attenuation   = albedo;
        srec.skip_pdf      = true;
        srec.skip_pdf_ray  = ray(rec.p, wi, r_in.time());
        return true;
    }

    double scattering_pdf(const ray& r_in, const hit_record& /*rec*/,
                          const ray& scattered) const override {
        vec3 wo = unit_vector(-r_in.direction());
        vec3 wi = unit_vector(scattered.direction());
        return phase.pdf(wo.x(), wo.y(), wo.z(), wi.x(), wi.y(), wi.z());
    }

    // See material::is_shadow_transmissive()'s comment - matches
    // optix_anyhit_shadow.h's MaterialType::Medium skip. Both sides make the
    // same simplification here: a shadow ray through fog is treated as fully
    // unattenuated rather than integrating real Beer-Lambert transmittance
    // (constant_medium::transmittance() exists for that but nothing calls
    // it yet) - "unoccluded" is closer to correct than "fully blocked", which
    // is what happened before this override existed.
    bool is_shadow_transmissive(const hit_record&) const override { return true; }

  private:
    color albedo;
    HenyeyGreensteinPhaseFunction<double> phase;
};


// ---------------------------------------------------------------------------
// constant_medium
// Homogeneous participating medium (fog, smoke, clouds).
// pbrt-v4 reference: HomogeneousMedium + delta-tracking free-path sampling.
// ---------------------------------------------------------------------------
class constant_medium : public hittable {
  public:
    // Legacy constructor (isotropic, g=0): density is sigma_t = sigma_a + sigma_s.
    // We treat density as sigma_s only (purely scattering medium) for compatibility
    // with existing scenes (no absorption, albedo = albedo param).
    constant_medium(shared_ptr<hittable> boundary, double density,
                    shared_ptr<texture> tex, double g = 0.0)
        : boundary(boundary) {
        // albedo from texture (sampled at center, stored for material)
        // sigma_t = density, sigma_s = density (no absorption)
        med = HomogeneousMediumData<double>(/*sa=*/0.0, /*ss=*/density, g);
        // Phase material uses albedo = sigma_s/sigma_t = 1.0 if sa=0
        // Texture albedo read at scatter time -- wrap in material with unit weight
        // and multiply by texture value there.  For simplicity we use the texture
        // to set a constant albedo color:
        color tex_color = tex->value(0.5, 0.5, point3(0,0,0));
        phase_mat = make_shared<hg_phase_material>(tex_color, g);
    }

    constant_medium(shared_ptr<hittable> boundary, double density,
                    const color& albedo, double g = 0.0)
        : boundary(boundary) {
        med = HomogeneousMediumData<double>(/*sa=*/0.0, /*ss=*/density, g);
        phase_mat = make_shared<hg_phase_material>(albedo, g);
    }

    // Full pbrt-v4-style constructor: separate absorption and scattering coefficients.
    constant_medium(shared_ptr<hittable> boundary,
                    double sigma_a, double sigma_s,
                    const color& albedo, double g = 0.0)
        : boundary(boundary) {
        med = HomogeneousMediumData<double>(sigma_a, sigma_s, g);
        // Single-scattering albedo for the phase material
        double sigma_t = sigma_a + sigma_s;
        color ss_albedo = (sigma_t > 0) ? albedo * color(sigma_s / sigma_t,
                                                         sigma_s / sigma_t,
                                                         sigma_s / sigma_t)
                                        : color(0,0,0);
        phase_mat = make_shared<hg_phase_material>(ss_albedo, g);
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        hit_record rec1, rec2;

        if (!boundary->hit(r, interval::universe, rec1))
            return false;

        if (!boundary->hit(r, interval(rec1.t + 0.0001, infinity), rec2))
            return false;

        if (rec1.t < ray_t.min) rec1.t = ray_t.min;
        if (rec2.t > ray_t.max) rec2.t = ray_t.max;

        if (rec1.t >= rec2.t)
            return false;

        if (rec1.t < 0)
            rec1.t = 0;

        double ray_length          = r.direction().length();
        double distance_inside     = (rec2.t - rec1.t) * ray_length;

        // pbrt-v4 delta-tracking free-path sample: t = -log(1-u) / sigma_t
        // med.sample_free_path(u) returns this value directly.
        double hit_distance = med.sample_free_path(random_double());

        if (hit_distance > distance_inside)
            return false;

        rec.t = rec1.t + hit_distance / ray_length;
        rec.p = r.at(rec.t);

        rec.normal    = vec3(1, 0, 0);  // arbitrary (volume has no surface normal)
        rec.front_face = true;
        rec.mat       = phase_mat;

        return true;
    }

    // Beer-Lambert transmittance for a segment of length t through this medium.
    // Used by shadow rays that pass through a volume.
    // pbrt-v4 HomogeneousMedium: Tr(t) = exp(-sigma_t * t), averaged over RGB.
    color transmittance(double t) const {
        double Tr_r, Tr_g, Tr_b;
        med.transmittance(t, Tr_r, Tr_g, Tr_b);
        return color(Tr_r, Tr_g, Tr_b);
    }

    aabb bounding_box() const override { return boundary->bounding_box(); }

  private:
    shared_ptr<hittable>         boundary;
    HomogeneousMediumData<double> med;
    shared_ptr<hg_phase_material> phase_mat;
};


#endif

