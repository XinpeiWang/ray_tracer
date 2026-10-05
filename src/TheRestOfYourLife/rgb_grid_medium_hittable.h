#pragma once
//==============================================================================================
// rgb_grid_medium_hittable.h -- CPU hittable wrapper around RGBGridMediumData<T>
// (src/shared/rgb_grid_medium.h), a real heterogeneous medium with an
// independent per-voxel R/G/B scattering grid (mirrors pbrt-v4 RGBGridMedium,
// media.h). Unlike cloud_medium_hittable.h's CloudMedium (single scalar
// density, procedural, one global majorant), this uses RGBGridMediumData's
// own DDA majorant grid (src/shared/grid_medium.h's DDAMajorantIterator) -
// the ray is walked segment by segment, each with its own local majorant
// bound, rather than one bound for the whole ray. Each accepted scattering
// event's color comes from the local per-channel sigma_s at that point
// (normalized so the brightest channel is 1, used as the phase material's
// albedo) - the whole point of this scene over cloud_medium_hittable's
// grayscale density is spatially-varying COLOR, not just density.
//
// RGBGridMediumData's bounds member must be the unit cube [0,1]^3 (matching
// how its own unit tests use it - see rgb_grid_medium_tests.cpp) - the
// world<->medium affine transform is this hittable's own responsibility, the
// same "hittable does the transform, medium struct works in [0,1]^3" split
// cloud_medium_hittable.h/CloudMedium<T> already use, just done here
// explicitly (CloudMedium's sample_ray() does the transform internally;
// RGBGridMediumData's sample_ray()/intersect_ray() expect an already-
// transformed ray, per that struct's own doc comments).
//
// sigma_a is fixed at 0 (pure scattering) - same convention and same reason
// as cloud_medium_hittable.h and constant_medium.h: an absorption event
// needs to terminate the path with zero contribution, which none of this
// codebase's medium hittables model (their hit()->bool interface has no way
// to say "hit, but absorbed" vs "no hit").
//==============================================================================================

#include "hittable.h"
#include "constant_medium.h"  // hg_phase_material
#include "../shared/rgb_grid_medium.h"

class rgb_grid_medium_hittable : public hittable, public event_medium {
  public:
    // medium: built with bounds = unit cube (see file comment). phase_g is
    // the HG asymmetry shared by every scatter event (a single scalar, same
    // simplification cloud_medium_hittable.h makes - only the color varies
    // per point, not the phase function shape). world_to_medium_mat/
    // translate: row-major 3x3 + translate, world -> medium-space affine
    // transform (same convention as CloudMedium::world_to_medium_pt).
    rgb_grid_medium_hittable(const RGBGridMediumData<double>& medium, double phase_g,
                              const point3& world_min, const point3& world_max,
                              const double* world_to_medium_mat,
                              const double* world_to_medium_translate)
        : grid(medium), phase_g(phase_g), world_min(world_min), world_max(world_max) {
        for (int i = 0; i < 9; ++i) mat_[i] = world_to_medium_mat[i];
        for (int i = 0; i < 3; ++i) translate_[i] = world_to_medium_translate[i];
        // One phase material for every event of camera::ray_color()'s own sampling (sample_event): the collision's weights ride in the
        // hit_record (hit_record::has_medium_event), so nothing about this material depends on the point.
        event_phase_ = std::make_shared<hg_phase_material>(
            color(1, 1, 1), phase_g, [this](const ray& sr, double t_max) { return shadow_transmittance_impl(sr, t_max); }, color(0, 0, 0));
    }

    // ---- per-channel spectral tracking, sampled by camera::ray_color() (see event_medium) -------------------------------------------
    //
    // hit() below is delta tracking with ONE extinction - the brightest channel of sigma_a + sigma_s at the point - and treats every real
    // collision as a scatter whose albedo is sigma_s / max(sigma_s). That never absorbs: the collision probability counts sigma_a, the
    // scattered path does not lose it (a grid with pbrt's default sigma_a = 1 and a small sigma_s scatters at nearly full strength where
    // pbrt absorbs ~97% of the collisions), and a channel with a smaller sigma_t than the brightest is made to absorb instead of passing.
    //
    // sample_event() is spectral tracking against the same scalar majorant: at each tentative collision a real event happens with the
    // probability mean_c(sigma_t_c) / majorant and a null event otherwise, and the weights keep each channel unbiased -
    //   real: scatter weight sigma_s_c / mean(sigma_t), emission weight sigma_a_c / mean(sigma_t) (absorption ends the path, as it does
    //         for the homogeneous media: only its emission survives),
    //   null: the running weight takes (majorant - sigma_t_c) / (majorant - mean(sigma_t)), at most 3 for any channel.
    // A grey grid reduces to ordinary delta tracking with albedo sigma_s / sigma_t.
    bool chord(const ray& r, double& t0, double& t1) const override {
        const double len = r.direction().length();
        if (!(len > 0.0)) return false;
        const ray ur(r.origin(), r.direction() / len, r.time());
        double mox, moy, moz, mdx, mdy, mdz;
        world_to_medium(ur, mox, moy, moz, mdx, mdy, mdz);
        Ray3<double> mray(mox, moy, moz, mdx, mdy, mdz);
        double tMin, tMax;
        if (!grid.intersect_ray(mray, infinity, tMin, tMax)) return false;
        if (tMin < 0) tMin = 0;
        if (tMin >= tMax) return false;
        t0 = tMin / len;
        t1 = tMax / len;
        return true;
    }

    // Returns false if the ray does not cross the medium before t_surface. Otherwise `collided` says whether a real collision happened
    // (then rec is a medium-scatter hit at it, carrying that collision's path and emission weights) or not (then `weight` is the product
    // of the null-collision weights, to multiply onto the path's beta). entry_t: where the ray enters the medium (ray-parameter units).
    bool sample_event(const ray& r, double t_surface, double& entry_t, bool& collided, hit_record& rec, color& weight) const override {
        const double len = r.direction().length();
        if (!(len > 0.0)) return false;
        const ray ur(r.origin(), r.direction() / len, r.time());
        double mox, moy, moz, mdx, mdy, mdz;
        world_to_medium(ur, mox, moy, moz, mdx, mdy, mdz);
        Ray3<double> mray(mox, moy, moz, mdx, mdy, mdz);
        const double t_hi = (t_surface >= infinity) ? infinity : t_surface * len;
        double tMin, tMax;
        if (!grid.intersect_ray(mray, t_hi, tMin, tMax)) return false;
        if (tMin < 0) tMin = 0;
        if (tMin >= tMax) return false;
        entry_t = tMin / len;

        double w[3] = { 1.0, 1.0, 1.0 };
        bool real = false;
        march_segments(mray, tMin, tMax, [&](double tt, double majorant) {
            double sa[3], ss[3], le[3];
            grid.sample_point(mox + tt*mdx, moy + tt*mdy, moz + tt*mdz, sa, ss, le);
            double cw[3], ce[3];
            if (heterogeneous_tracking_step<3, double>(sa, ss, majorant, random_double(), w, cw, ce)) {
                rec.t = tt / len;
                rec.p = ur.at(tt);
                rec.normal = vec3(1, 0, 0);   // arbitrary (a volume has no surface normal)
                rec.front_face = true;
                rec.mat = event_phase_;
                rec.u = 0.0;
                rec.v = 0.0;
                rec.has_medium_event = true;
                rec.medium_weight = color(cw[0], cw[1], cw[2]);
                // The grid's own (Lescale-multiplied) emission at this point, weighted by this event's absorption share.
                rec.medium_emission = color(le[0] * ce[0], le[1] * ce[1], le[2] * ce[2]);
                real = true;
                return true;
            }
            return false;
        });
        collided = real;
        if (!real) weight = color(w[0], w[1], w[2]);
        return true;
    }

    color transmittance_along(const ray& r, double t_max) const override { return shadow_transmittance_impl(r, t_max); }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        // The default path tracer samples this medium itself (sample_event() above); it must not also collide here.
        if (g_chromatic_media_integrator_managed) return false;
        // Work in world distance: a camera ray's direction is not unit length (it is pixel_sample - origin),
        // and sigma is per world unit, so sampling along the raw parameter made the medium |d| times too thin for
        // primary rays. constant_medium converts by ray_length the same way. rec.t goes back to ray-parameter units.
        const double len = r.direction().length();
        if (!(len > 0.0)) return false;
        const ray ur(r.origin(), r.direction() / len, r.time());
        const double t_lo = ray_t.min * len, t_hi = ray_t.max * len;
        double mox, moy, moz, mdx, mdy, mdz;
        world_to_medium(ur, mox, moy, moz, mdx, mdy, mdz);

        Ray3<double> mray(mox, moy, moz, mdx, mdy, mdz);
        double tMin, tMax;
        if (!grid.intersect_ray(mray, t_hi, tMin, tMax)) return false;
        if (tMin < t_lo) tMin = t_lo;
        if (tMin >= tMax) return false;

        bool got_hit = false;
        march_segments(mray, tMin, tMax, [&](double tt, double seg_sigma_maj) {
            double px = mox + tt*mdx, py = moy + tt*mdy, pz = moz + tt*mdz;
            double sa[3], ss[3], le[3];
            grid.sample_point(px, py, pz, sa, ss, le);
            double sigma_t_local = std::max({ sa[0]+ss[0], sa[1]+ss[1], sa[2]+ss[2] });

            if (random_double() < sigma_t_local / seg_sigma_maj) {
                rec.t = tt / len;
                rec.p = ur.at(tt);
                rec.normal     = vec3(1, 0, 0);  // arbitrary (volume has no surface normal)
                rec.front_face = true;

                double maxc = std::max({ ss[0], ss[1], ss[2], 1e-6 });
                color albedo(ss[0]/maxc, ss[1]/maxc, ss[2]/maxc);
                // Real per-voxel "rgb Le"/"float Lescale" (pbrt-v4's own
                // RGBGridMedium::LeGrid/LeScale) - le[] is already the
                // fully-resolved (Le_scale-multiplied) emission at this
                // exact scatter point, from the SAME grid.sample_point()
                // call this branch already made for sa/ss above. Weighted
                // by sigma_a/sigma_t here, matching constant_medium.h's own
                // homogeneous-medium convention exactly (hg_phase_
                // material::emitted()'s own comment has the full physical
                // derivation) - a real collision here is either an
                // absorption or a scattering event with probability
                // sa[c]/(sa[c]+ss[c]) each, and only the absorption
                // fraction contributes emission (pbrt-v4's own collision-
                // probability weighting for volumetric emission).
                color emission(0, 0, 0);
                for (int c = 0; c < 3; ++c) {
                    double sigma_t_c = sa[c] + ss[c];
                    // `> 1e-9`, not constant_medium.h's plain `> 0` for the
                    // analogous homogeneous-medium divide guard (that
                    // file's own hg_phase_material-constructing
                    // constructor) - a deliberate difference, not drift: sa/
                    // ss there are exact scene-author-supplied constants
                    // (sigma_t is either exactly 0.0 or a real chosen
                    // value), while sa[c]/ss[c] here come from
                    // grid.sample_point()'s trilinear interpolation of
                    // per-voxel data, which can leave sigma_t_c a tiny
                    // nonzero float noise value near a voxel boundary where
                    // the true density is meant to be zero - dividing le[c]
                    // by that near-zero value would spike into a bright,
                    // spurious firefly rather than genuinely produce zero
                    // emission. A code-review pass flagged the two guards'
                    // inconsistency; kept apart since they guard against
                    // different failure modes (exact-zero vs. near-zero
                    // interpolation noise).
                    if (sigma_t_c > 1e-9) emission[c] = le[c] * (sa[c] / sigma_t_c);
                }
                rec.mat = make_shared<hg_phase_material>(albedo, phase_g,
                    [this](const ray& sr, double t_max) { return shadow_transmittance_impl(sr, t_max); },
                    emission);
                got_hit = true;
                return true;  // real scatter event: stop marching
            }
            return false;  // null collision: keep marching
        });
        return got_hit;
    }

    aabb bounding_box() const override {
        return aabb(world_min, world_max);
    }

  private:
    // Transforms r into medium space. Direction transforms by the matrix
    // only (no translation) - same as CloudMedium::world_to_medium_pt.
    // Applying the SAME matrix to both origin and direction means the
    // medium-space ray is still parametrized by the identical `t` as the
    // world-space ray (o+t*d transforms to (mat*o+translate)+t*(mat*d)),
    // so segment/hit `t` values read back directly with no rescaling.
    // Shared by hit() and shadow_transmittance_impl() - previously
    // duplicated inline in both.
    void world_to_medium(const ray& r, double& mox, double& moy, double& moz,
                          double& mdx, double& mdy, double& mdz) const {
        point3 o = r.origin();
        vec3   d = r.direction();
        mox = mat_[0]*o.x() + mat_[1]*o.y() + mat_[2]*o.z() + translate_[0];
        moy = mat_[3]*o.x() + mat_[4]*o.y() + mat_[5]*o.z() + translate_[1];
        moz = mat_[6]*o.x() + mat_[7]*o.y() + mat_[8]*o.z() + translate_[2];
        mdx = mat_[0]*d.x() + mat_[1]*d.y() + mat_[2]*d.z();
        mdy = mat_[3]*d.x() + mat_[4]*d.y() + mat_[5]*d.z();
        mdz = mat_[6]*d.x() + mat_[7]*d.y() + mat_[8]*d.z();
    }

    // Walks grid's majorant-DDA segments across [tMin, tMax] (medium
    // space), calling on_candidate(t, seg_sigma_maj) for each stochastic
    // candidate point along the way. Stops as soon as on_candidate returns
    // true; otherwise walks every segment to exhaustion and returns false.
    // Shared by hit() (stops at the first accepted real-scatter event) and
    // shadow_transmittance_impl() (walks every candidate, accumulating) -
    // previously each duplicated this same segment-marching loop inline.
    template <typename CandidateFn>
    bool march_segments(const Ray3<double>& mray, double tMin, double tMax,
                         CandidateFn&& on_candidate) const {
        auto it = grid.sample_ray(mray, tMin, tMax);
        for (;;) {
            auto seg = it.Next();
            if (!seg.has_value()) return false;  // ray exited the medium's AABB
            if (seg->sigma_maj <= 0.0) continue;  // empty segment, try the next one

            double tt = seg->tMin;
            for (;;) {
                double dt = -std::log(1.0 - random_double()) / seg->sigma_maj;
                tt += dt;
                if (tt >= seg->tMax) break;  // exited this segment - advance to the next
                if (on_candidate(tt, seg->sigma_maj)) return true;
            }
        }
    }

    // Per-channel ratio-tracking transmittance (same technique as
    // cloud_medium_hittable's own shadow_transmittance_impl(), see that
    // file's comment) via march_segments() above, bounded by t_max - the
    // shadow ray's real target distance (e.g. a punctual light) - rather
    // than the medium's full extent. Tracked per-RGB-channel (unlike
    // cloud_medium_hittable's single shared RatioTrackingTrHeterogeneous
    // call) since this grid's whole point is spatially-varying COLOR, not
    // just density (see file header comment) - a single random walk
    // position shared across all 3 channels' weights, not 3 independent
    // walks, which is why this stays its own loop rather than 3 calls into
    // ratio_tracking.h's single-channel helper.
    color shadow_transmittance_impl(const ray& r, double t_max) const {
        // World distance, as in hit() - the shadow ray's direction is not unit length either.
        const double len = r.direction().length();
        if (!(len > 0.0)) return color(1, 1, 1);
        const ray ur(r.origin(), r.direction() / len, r.time());
        const double t_max_w = t_max * len;
        double mox, moy, moz, mdx, mdy, mdz;
        world_to_medium(ur, mox, moy, moz, mdx, mdy, mdz);

        Ray3<double> mray(mox, moy, moz, mdx, mdy, mdz);
        double tMin, tMax;
        if (!grid.intersect_ray(mray, t_max_w, tMin, tMax)) return color(1, 1, 1);
        if (tMin < 0) tMin = 0;
        if (tMin >= tMax) return color(1, 1, 1);

        double Tr[3] = { 1.0, 1.0, 1.0 };
        // Safety cap, same spirit as shadow_ray.h's kMaxTransmissiveSkips.
        constexpr int kMaxNullCollisions = 100000;
        int collisions = 0;

        march_segments(mray, tMin, tMax, [&](double tt, double seg_sigma_maj) {
            if (++collisions > kMaxNullCollisions) return true;  // bail out, same as cloud's cap

            double px = mox + tt*mdx, py = moy + tt*mdy, pz = moz + tt*mdz;
            double sa[3], ss[3], le[3];
            grid.sample_point(px, py, pz, sa, ss, le);
            for (int c = 0; c < 3; ++c) {
                double sigma_t_c = sa[c] + ss[c];
                Tr[c] *= 1.0 - (sigma_t_c / seg_sigma_maj);
            }
            // Fully opaque already - no need to keep marching (mirrors
            // cloud_medium_hittable's single-channel Tr<=0 early exit).
            if (Tr[0] <= 0.0 && Tr[1] <= 0.0 && Tr[2] <= 0.0) return true;
            return false;
        });
        return color(Tr[0], Tr[1], Tr[2]);
    }

    RGBGridMediumData<double> grid;
    shared_ptr<hg_phase_material> event_phase_;
    double phase_g;
    double mat_[9], translate_[3];
    point3 world_min, world_max;
};
