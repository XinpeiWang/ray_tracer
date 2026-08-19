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
#include <functional>


// ---------------------------------------------------------------------------
// hg_phase_pdf
// pdf wrapper around HenyeyGreensteinPhaseFunction<double>, so medium
// scattering can go through the same NEE + MIS machinery every other
// non-specular material uses (see hg_phase_material::scatter()'s comment
// for why this replaced the old skip_pdf=true shortcut). wo (outgoing,
// toward the ray's origin) is fixed at construction, matching cosine_pdf
// fixing its normal at construction - the pdf interface's generate()/
// value() only ever deal with the incoming direction wi.
// ---------------------------------------------------------------------------
class hg_phase_pdf : public pdf {
  public:
    hg_phase_pdf(const vec3& wo, const HenyeyGreensteinPhaseFunction<double>& phase)
        : wo(unit_vector(wo)), phase(phase) {}

    double value(const vec3& direction) const override {
        vec3 wi = unit_vector(direction);
        return phase.pdf(wo.x(), wo.y(), wo.z(), wi.x(), wi.y(), wi.z());
    }

    vec3 generate() const override {
        double wi_x, wi_y, wi_z, pdf_val;
        phase.sample(wo.x(), wo.y(), wo.z(),
                     random_double(), random_double(),
                     wi_x, wi_y, wi_z, pdf_val);
        return vec3(wi_x, wi_y, wi_z);
    }

  private:
    vec3 wo;
    HenyeyGreensteinPhaseFunction<double> phase;
};


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
    // transmittance_fn: computes real shadow-ray transmittance through
    // THIS specific medium instance's geometry - a callback rather than a
    // subclass because the math needed differs per medium shape
    // (constant_medium's closed-form Beer-Lambert vs. cloud_medium_hittable/
    // rgb_grid_medium_hittable's stochastic ratio tracking) while the phase
    // function/scatter() logic below is identical for all three. Default
    // nullptr means "no real attenuation available" - falls back to
    // material::shadow_transmittance()'s own default (fully transmissive),
    // same behavior as before this feature existed.
    hg_phase_material(const color& albedo, double g,
                       std::function<color(const ray&)> transmittance_fn = nullptr)
        : albedo(albedo), phase(g), transmittance_fn_(std::move(transmittance_fn)) {}

    bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                 bool /*do_regularize*/ = false) const override {
        // skip_pdf=true (the original implementation here) tells the
        // integrator this material is a Dirac-delta/specular-style bounce
        // and to skip NEE entirely for it - correct for a mirror/glass
        // BSDF (a random NEE-sampled direction has ~zero chance of landing
        // exactly on the specular direction, so NEE would be wasted work),
        // but WRONG for a phase function: HG is smooth/continuous, exactly
        // like a diffuse BRDF, and benefits from NEE the same way. Every
        // fog/smoke/cloud medium in this codebase routes through this
        // scatter() (constant_medium, and now cloud_medium_hittable), so
        // this wasn't just a per-scene tuning issue - it silently disabled
        // direct light sampling for every volume-scattering event, forcing
        // the integrator to rely on randomly BSDF-sampling a direction that
        // happens to hit the light. That's fine for a small, localized
        // medium (most of the frame is still normal NEE-lit surfaces - see
        // A8 Cornell Smoke, where only the smoke itself reads noisier than
        // the walls), but for a medium filling most/all of the visible
        // scene (scene 30, Homogeneous Medium) it dominates the image and
        // stayed heavily noisy even at 5x the recommended sample count
        // until this fix.
        vec3 wo = unit_vector(-r_in.direction());  // outgoing = toward camera
        srec.attenuation = albedo;
        srec.pdf_ptr      = make_shared<hg_phase_pdf>(wo, phase);
        srec.skip_pdf     = false;
        return true;
    }

    double scattering_pdf(const ray& r_in, const hit_record& /*rec*/,
                          const ray& scattered) const override {
        vec3 wo = unit_vector(-r_in.direction());
        vec3 wi = unit_vector(scattered.direction());
        return phase.pdf(wo.x(), wo.y(), wo.z(), wi.x(), wi.y(), wi.z());
    }

    // See optix_anyhit_shadow.h's MaterialType::Medium skip (GPU's matching
    // treatment) - a shadow ray through fog keeps going rather than being
    // treated as a hard blocker; how much it's dimmed on the way is
    // shadow_transmittance()'s job below, not this one's.
    bool is_shadow_transmissive(const hit_record&) const override { return true; }

    color shadow_transmittance(const ray& r) const override {
        return transmittance_fn_ ? transmittance_fn_(r) : material::shadow_transmittance(r);
    }

  private:
    color albedo;
    HenyeyGreensteinPhaseFunction<double> phase;
    std::function<color(const ray&)> transmittance_fn_;
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
        phase_mat = make_shared<hg_phase_material>(tex_color, g,
            [this](const ray& r) { return shadow_transmittance_impl(r); });
    }

    constant_medium(shared_ptr<hittable> boundary, double density,
                    const color& albedo, double g = 0.0)
        : boundary(boundary) {
        med = HomogeneousMediumData<double>(/*sa=*/0.0, /*ss=*/density, g);
        phase_mat = make_shared<hg_phase_material>(albedo, g,
            [this](const ray& r) { return shadow_transmittance_impl(r); });
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
        phase_mat = make_shared<hg_phase_material>(ss_albedo, g,
            [this](const ray& r) { return shadow_transmittance_impl(r); });
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
    // Deterministic Beer-Lambert transmittance for a shadow ray's full
    // traversal of THIS medium's boundary - same rec1/rec2 entry/exit
    // computation as hit() above (lines ~186-205), just without the
    // stochastic free-path sampling: a shadow ray wants the real
    // attenuation over the whole segment it crosses, not a scatter-event
    // decision. Returns (1,1,1) (no attenuation) if the ray doesn't
    // actually cross the boundary - defensive only, shadow_ray.h only
    // calls this once it already knows this material was hit.
    color shadow_transmittance_impl(const ray& r) const {
        hit_record rec1, rec2;
        if (!boundary->hit(r, interval::universe, rec1)) return color(1, 1, 1);
        if (!boundary->hit(r, interval(rec1.t + 0.0001, infinity), rec2)) return color(1, 1, 1);

        double t0 = rec1.t < 0 ? 0 : rec1.t;
        if (t0 >= rec2.t) return color(1, 1, 1);

        double ray_length      = r.direction().length();
        double distance_inside = (rec2.t - t0) * ray_length;
        return transmittance(distance_inside);
    }

    shared_ptr<hittable>         boundary;
    HomogeneousMediumData<double> med;
    shared_ptr<hg_phase_material> phase_mat;
};


#endif

