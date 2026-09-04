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
    // emission: MakeNamedMedium's own "rgb Le"/"float Lescale" (pbrt-v4),
    // already weighted by sigma_a/sigma_t at the call site (see
    // pbrt_cpu_builder.h's addMediumIfPresent) - the per-collision emission
    // contribution added by emitted() below. (0,0,0) (the default) means
    // "no emission", matching every OTHER medium constructor in this class
    // that doesn't pass one.
    hg_phase_material(const color& albedo, double g,
                       std::function<color(const ray&, double)> transmittance_fn = nullptr,
                       const color& emission = color(0, 0, 0))
        : albedo(albedo), phase(g), transmittance_fn_(std::move(transmittance_fn)),
          emission_(emission) {}

    // MakeNamedMedium's own "rgb Le"/"float Lescale" (pbrt-v4) - see
    // pbrt_cpu_builder.h's addMediumIfPresent for the sigma_a/sigma_t
    // weighting this already carries. camera.h's ray_color()/
    // ray_color_spectral() call this unconditionally on every hit_record
    // (the same generic "Le on any material" dispatch surface-area lights
    // use), so a medium-scatter hit_record (constant_medium::hit(), below)
    // contributes emission through the EXACT SAME MIS-aware accumulation
    // surface lights do - correctly full-weight (beta*Le) when arriving via
    // a specular_bounce and correctly power-heuristic-MIS-weighted
    // otherwise, with pdf_l naturally 0 (this medium is never a member of
    // `lights`, so it can't be NEE-sampled), matching pbrt-v4's own
    // volumetric-emission treatment of never explicitly next-event-
    // estimating a medium's own Le.
    color emitted(const ray&, const hit_record&, double, double, const point3&) const override {
        return emission_;
    }

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

    // See material::is_medium_scatter()'s own comment - lets bdpt_adapter.h
    // tell this hit apart from a real emissive surface and suppress its new
    // emitted() there this round, rather than feeding it through BDPT's
    // surface-light vertex machinery incorrectly.
    bool is_medium_scatter() const override { return true; }

    color shadow_transmittance(const ray& r, double t_max) const override {
        return transmittance_fn_ ? transmittance_fn_(r, t_max)
                                  : material::shadow_transmittance(r, t_max);
    }

  private:
    color albedo;
    HenyeyGreensteinPhaseFunction<double> phase;
    std::function<color(const ray&, double)> transmittance_fn_;
    color emission_;
};


// Derives a phase material's single-scattering albedo and (already-weighted)
// emission from a HomogeneousMediumData's own scalar sigma_a/sigma_s plus a
// caller-supplied RGB albedo/Le - the exact math constant_medium's
// pbrt-v4-style constructor and ambient_medium's constructor both need,
// factored out here so it's derived once rather than by each constructor
// independently. Only meaningful for the scalar (uniform-per-channel)
// HomogeneousMediumData(sa, ss, g) constructor both callers use -
// sigma_tr()==sigma_tg()==sigma_tb() and sigma_ar==sigma_ag==sigma_ab hold
// by construction, so reading sigma_ar/sigma_tr() alone is enough. See
// hg_phase_material::emitted()'s own comment for the physical derivation of
// the emission weight (sigma_a/sigma_t).
inline void collapse_homogeneous_medium(const HomogeneousMediumData<double>& med,
                                        const color& albedo, const color& Le,
                                        color& ss_albedo, color& emission) {
    const double sigma_t = med.sigma_tr();
    double wr, wg, wb;
    med.scatter_weight(wr, wg, wb);
    ss_albedo = (sigma_t > 0) ? albedo * color(wr, wg, wb) : color(0, 0, 0);
    emission = (sigma_t > 0) ? Le * (med.sigma_ar / sigma_t) : color(0, 0, 0);
}

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
            [this](const ray& r, double t_max) { return shadow_transmittance_impl(r, t_max); });
    }

    constant_medium(shared_ptr<hittable> boundary, double density,
                    const color& albedo, double g = 0.0)
        : boundary(boundary) {
        med = HomogeneousMediumData<double>(/*sa=*/0.0, /*ss=*/density, g);
        phase_mat = make_shared<hg_phase_material>(albedo, g,
            [this](const ray& r, double t_max) { return shadow_transmittance_impl(r, t_max); });
    }

    // Full pbrt-v4-style constructor: separate absorption and scattering
    // coefficients, plus MakeNamedMedium's own raw (unweighted) "rgb Le"
    // (pbrt-v4) - see hg_phase_material::emitted()'s own comment for how
    // the WEIGHTED result reaches the render. Weighted by sigma_a/sigma_t
    // right here (not at the caller, pbrt_cpu_builder.h's
    // addMediumIfPresent) so this constructor's own pre-existing sigma_t
    // computation just below (needed for ss_albedo regardless) is computed
    // once and reused for both, rather than the caller separately deriving
    // an already-weighted emission color from its own independent copy of
    // sigma_a/sigma_t. (0,0,0) (the default) means "no emission", matching
    // every native (non-pbrt) scene that doesn't pass this argument at all.
    constant_medium(shared_ptr<hittable> boundary,
                    double sigma_a, double sigma_s,
                    const color& albedo, double g = 0.0,
                    const color& Le = color(0, 0, 0))
        : boundary(boundary) {
        med = HomogeneousMediumData<double>(sigma_a, sigma_s, g);
        // Single-scattering albedo for the phase material, and the
        // collision-probability weight (sigma_a/sigma_t) for emission -
        // see collapse_homogeneous_medium()'s own comment for the derivation.
        color ss_albedo, emission;
        collapse_homogeneous_medium(med, albedo, Le, ss_albedo, emission);
        phase_mat = make_shared<hg_phase_material>(ss_albedo, g,
            [this](const ray& r, double t_max) { return shadow_transmittance_impl(r, t_max); },
            emission);
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        double t0, t1;

        // A code-review pass found the generic "two sequential hit() calls"
        // path below is WRONG for an open (uncapped) boundary shape - see
        // hittable::volume_bounds()'s own comment for the full derivation
        // (it can silently miss real medium extent, e.g. a ray entering/
        // exiting through an open end rather than the shape's lateral
        // wall, biasing this medium systematically DIMMER than a correct
        // render - confirmed as the root cause of a real CPU/GPU brightness
        // parity test failure, tests/integration/
        // material_cpu_gpu_parity_tests.cpp's Scene112/E6 case, for
        // cylinder specifically). Every other boundary shape this class
        // wraps (sphere, disk-as-a-degenerate-case, a closed mesh) is
        // topologically closed, where hit()/hit()'s first-then-next-
        // crossing pattern IS correct - supports_volume_bounds() defaults
        // to false for those, so they fall through to the generic path
        // below unchanged. Checked separately from volume_bounds()'s own
        // return value - see supports_volume_bounds()'s own comment for
        // why: a real per-ray miss from a shape that DOES support the
        // exact technique must NOT re-try the generic (wrong-for-this-
        // shape) fallback.
        if (boundary->supports_volume_bounds()) {
            if (!boundary->volume_bounds(r, t0, t1))
                return false;
        } else {
            hit_record rec1, rec2;

            if (!boundary->hit(r, interval::universe, rec1))
                return false;

            if (!boundary->hit(r, interval(rec1.t + 0.0001, infinity), rec2))
                return false;

            t0 = rec1.t;
            t1 = rec2.t;
        }

        if (t0 < ray_t.min) t0 = ray_t.min;
        if (t1 > ray_t.max) t1 = ray_t.max;

        if (t0 >= t1)
            return false;

        if (t0 < 0)
            t0 = 0;

        double ray_length          = r.direction().length();
        double distance_inside     = (t1 - t0) * ray_length;

        // pbrt-v4 delta-tracking free-path sample: t = -log(1-u) / sigma_t
        // med.sample_free_path(u) returns this value directly.
        double hit_distance = med.sample_free_path(random_double());

        if (hit_distance > distance_inside)
            return false;

        rec.t = t0 + hit_distance / ray_length;
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
    // Deterministic Beer-Lambert transmittance for a shadow ray's
    // traversal of THIS medium's boundary, bounded by t_max (the shadow
    // ray's real target distance, e.g. a punctual light) - same rec1/rec2
    // entry/exit computation as hit() above (lines ~186-205), just without
    // the stochastic free-path sampling: a shadow ray wants the real
    // attenuation over the segment it actually needs (entry up to whichever
    // is closer, the medium's exit or the light), not a scatter-event
    // decision, and not the medium's full extent if the light is closer
    // than the far boundary. Returns (1,1,1) (no attenuation) if the ray
    // doesn't actually cross the boundary - defensive only, shadow_ray.h
    // only calls this once it already knows this material was hit.
    color shadow_transmittance_impl(const ray& r, double t_max) const {
        double raw_t0, raw_t1;

        // Same supports_volume_bounds()/volume_bounds() split as hit()
        // above - see that function's own comment.
        if (boundary->supports_volume_bounds()) {
            if (!boundary->volume_bounds(r, raw_t0, raw_t1)) return color(1, 1, 1);
        } else {
            hit_record rec1, rec2;
            if (!boundary->hit(r, interval::universe, rec1)) return color(1, 1, 1);
            if (!boundary->hit(r, interval(rec1.t + 0.0001, infinity), rec2)) return color(1, 1, 1);
            raw_t0 = rec1.t;
            raw_t1 = rec2.t;
        }

        double t0 = raw_t0 < 0 ? 0 : raw_t0;
        double t1 = std::min(raw_t1, t_max);
        if (t0 >= t1) return color(1, 1, 1);

        double ray_length      = r.direction().length();
        double distance_inside = (t1 - t0) * ray_length;
        return transmittance(distance_inside);
    }

    shared_ptr<hittable>         boundary;
    HomogeneousMediumData<double> med;
    shared_ptr<hg_phase_material> phase_mat;
};


// ---------------------------------------------------------------------------
// ambient_medium
// An UNBOUNDED homogeneous participating medium the camera itself starts
// inside - pbrt-v4's own "camera medium" (the MediumInterface active when
// the Camera directive is parsed - see pbrt_scene.h's Scene::
// cameraMediumName and pbrt_flatten.h's FlatScene::cameraMediumIndex own
// comments for how a scene requests this). No bounding shape at all: the
// medium simply fills all of space, exactly like real pbrt-v4's own
// unbounded HomogeneousMedium when nothing else declares an exit.
//
// Deliberately NOT a hittable inserted into the scene's own BVH/
// hittable_list: hittable_list::hit() already selects the true nearest hit
// regardless of child order (its shrinking closest_so_far interval means a
// farther candidate can never overwrite a nearer one), so traversal order
// itself is not the problem. The real blocker is that hittable::hit()'s
// bool-plus-hit_record signature has no side channel for a child that does
// NOT win the nearest-hit race to still attenuate the path's throughput by
// its own transmittance over the segment it WAS tested against - which an
// ambient medium spanning the whole ray needs to do even when some closer
// real surface wins. Instead called explicitly by camera.h's ray_color() as
// its own top-level step, AFTER world.hit() already ran, so its [0, t1]
// range is clipped to whatever real surface (or infinity, for an escaped
// ray) is already known to be the nearest thing in front of it, and the
// no-scatter case can multiply `beta` directly - see ray_color()'s own call
// site comment.
//
// Scope: default CPU path tracer only (ray_color(), not
// ray_color_spectral()/BDPT/MLT/SPPM); homogeneous only (matching this
// loader's own "close the homogeneous case first" precedent for other
// pbrt-v4 medium features); no true "exit" interaction with a scene that
// ALSO has real per-shape media (pbrt_flatten.h warns about this
// combination rather than modeling it - see that warning's own comment for
// why). Shadow-ray/NEE attenuation through it (a light behind the fog gets
// dimmed on its way to a shadow-ray target, not just primary/bounce-ray
// transmission) is applied by ray_color()'s own NEE strategies, each
// multiplying by transmittance_over() at the real distance to its target -
// NOT via transmittance_fn below (left null; that path is
// shadow_ray_hit()'s per-SURFACE transmittance walk, which never sees this
// untraced, unbounded medium at all - see ray_color()'s own
// camera_medium_trans comment for why this had to be a separate step).
// ---------------------------------------------------------------------------
class ambient_medium {
  public:
    ambient_medium(double sigma_a, double sigma_s, const color& albedo, double g = 0.0,
                   const color& Le = color(0, 0, 0))
        : med(sigma_a, sigma_s, g) {
        // See collapse_homogeneous_medium()'s own comment (above
        // constant_medium) for the derivation - shared with that class's
        // identical pbrt-v4-style constructor rather than re-derived here.
        color ss_albedo, emission;
        collapse_homogeneous_medium(med, albedo, Le, ss_albedo, emission);
        phase_mat = make_shared<hg_phase_material>(ss_albedo, g, /*transmittance_fn=*/nullptr, emission);
    }

    // Attempts to intercept `r` with a scatter event somewhere within
    // [0, t_max] - t_max is the ALREADY-KNOWN distance to the nearest real
    // surface, or `infinity` for an escaped ray. Mirrors constant_medium::
    // hit()'s own free-path sampling exactly, just without a boundary
    // shape's own entry/exit test (t0=0, t1=t_max directly - the whole
    // point of an unbounded medium). `out_ray_length` (optional), when
    // non-null, is always written with r.direction().length() before
    // returning (scatter or not) - the caller needs this same value to scale
    // transmittance_over() on the no-scatter path, and this way it's derived
    // once rather than a second sqrt at the call site.
    bool sample_scatter(const ray& r, double t_max, hit_record& rec, double* out_ray_length = nullptr) const {
        const double ray_length = r.direction().length();
        if (out_ray_length) *out_ray_length = ray_length;
        if (t_max <= 0) return false;

        // t_max==infinity propagates through IEEE754 arithmetic correctly
        // here (infinity * finite ray_length == infinity) - no special
        // case needed for the escaped-ray case, unlike transmittance_over()
        // below (exp(-0*infinity) would be NaN for a zero-extinction
        // channel, which this multiply doesn't have that problem).
        const double distance_inside = t_max * ray_length;

        const double hit_distance = med.sample_free_path(random_double());
        if (hit_distance > distance_inside) return false;

        rec.t = hit_distance / ray_length;
        rec.p = r.at(rec.t);
        rec.normal = vec3(1, 0, 0);   // arbitrary (volume has no surface normal)
        rec.front_face = true;
        rec.mat = phase_mat;
        // u/v have no meaning for a volume scatter (no surface parameterization)
        // but hit_record leaves them uninitialized by default - zero them so a
        // downstream differential/footprint read (see ray_color()'s primary-ray
        // path) never touches indeterminate memory.
        rec.u = 0.0;
        rec.v = 0.0;
        return true;
    }

    // Beer-Lambert transmittance for a free-flight segment of length
    // `distance` (world units, already ray_length-scaled) that did NOT
    // scatter - see constant_medium::transmittance()'s identical formula.
    // `distance` may be `infinity` (an escaped ray that never scattered
    // despite this being tested - only possible when sigma_t==0 for every
    // channel, since sample_scatter() above always eventually wins for any
    // real extinction) - guarded per-channel rather than calling exp() on
    // an `sigma_t * infinity` product, which is NaN (0*inf) for exactly the
    // zero-extinction channels this case requires.
    color transmittance_over(double distance) const {
        if (!std::isinf(distance)) {
            double Tr_r, Tr_g, Tr_b;
            med.transmittance(distance, Tr_r, Tr_g, Tr_b);
            return color(Tr_r, Tr_g, Tr_b);
        }
        return color(med.sigma_tr() > 0 ? 0.0 : 1.0,
                     med.sigma_tg() > 0 ? 0.0 : 1.0,
                     med.sigma_tb() > 0 ? 0.0 : 1.0);
    }

  private:
    HomogeneousMediumData<double> med;
    shared_ptr<hg_phase_material> phase_mat;
};


#endif

