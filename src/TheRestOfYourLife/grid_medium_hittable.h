#pragma once
//==============================================================================================
// grid_medium_hittable.h -- CPU hittable wrapper around GridMediumData<T>
// (src/shared/sampled_grid.h), a real heterogeneous medium with a single-
// channel per-voxel scalar density grid (mirrors pbrt-v4 GridMedium/
// "uniformgrid", media.h). The single-channel twin of rgb_grid_medium_
// hittable.h - same DDA-majorant-segment delta-tracking shape via
// GridMediumData<T>::sample_ray()/intersect_ray() (added alongside this
// file, since GridMediumData had no ray-facing API before - see that
// struct's own comment), just one density value per voxel instead of
// three, so there is no per-channel colour to derive - the scatter event's
// albedo is a caller-supplied constant tint (same convention
// cloud_medium_hittable.h already uses for its own single-channel density).
//
// GridMediumData's majorant grid stores RAW density (unscaled) - unlike
// RGBGridMediumData's own majorant grid, which is pre-scaled to a real
// sigma_t majorant - so this hittable's sigma_t_local at an accepted
// candidate is `density * (sigma_a+sigma_s)`, not `sa+ss` directly the way
// rgb_grid_medium_hittable reads per-channel sa/ss straight from the grid.
//
// sigma_a is fixed at 0 (pure scattering) - same convention and same reason
// as cloud_medium_hittable.h/rgb_grid_medium_hittable.h/constant_medium.h:
// an absorption event needs to terminate the path with zero contribution,
// which none of this codebase's medium hittables model (their hit()->bool
// interface has no way to say "hit, but absorbed" vs "no hit"). A scene
// that gave a real sigma_a silently loses it - flatten() already warns
// about this for cloud; pbrt_cpu_builder.h's own uniformgrid branch warns
// the same way.
//==============================================================================================

#include <utility>  // std::move

#include "hittable.h"
#include "constant_medium.h"  // hg_phase_material
#include "../shared/sampled_grid.h"

class grid_medium_hittable : public hittable {
  public:
    // medium: built with bounds = unit cube (matches rgb_grid_medium_
    // hittable's own convention - the world<->medium affine transform is
    // this hittable's own responsibility). Taken BY VALUE (not const&) and
    // moved into `grid` below - a GridMediumData<double> can carry a large
    // dense density array (up to ~1GB for a nanovdb-sourced grid near
    // pbrt_cpu_builder.h's 512-voxel-per-axis cap), so a caller passing an
    // rvalue (e.g. std::move(local_grid)) avoids a full second deep copy on
    // top of whatever copy already got the data into `medium` in the first
    // place; a caller passing an lvalue still works exactly as before
    // (one copy, same as this constructor always did). albedo: constant
    // scattering tint for every scatter event (no per-voxel colour - see
    // file comment). phase_g/world_to_medium_mat/translate: same
    // conventions as rgb_grid_medium_hittable's own constructor.
    grid_medium_hittable(GridMediumData<double> medium, const color& albedo,
                          double phase_g, const point3& world_min, const point3& world_max,
                          const double* world_to_medium_mat,
                          const double* world_to_medium_translate)
        : grid(std::move(medium)), albedo(albedo), phase_g(phase_g),
          world_min(world_min), world_max(world_max) {
        for (int i = 0; i < 9; ++i) mat_[i] = world_to_medium_mat[i];
        for (int i = 0; i < 3; ++i) translate_[i] = world_to_medium_translate[i];
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        double mox, moy, moz, mdx, mdy, mdz;
        world_to_medium(r, mox, moy, moz, mdx, mdy, mdz);

        Ray3<double> mray(mox, moy, moz, mdx, mdy, mdz);
        double tMin, tMax;
        if (!grid.intersect_ray(mray, ray_t.max, tMin, tMax)) return false;
        if (tMin < ray_t.min) tMin = ray_t.min;
        if (tMin >= tMax) return false;

        bool got_hit = false;
        march_segments(mray, tMin, tMax, [&](double tt, double seg_sigma_maj) {
            double px = mox + tt*mdx, py = moy + tt*mdy, pz = moz + tt*mdz;
            double sa, ss;
            grid.sample_sigma(px, py, pz, sa, ss);
            double sigma_t_local = sa + ss;

            if (random_double() < sigma_t_local / seg_sigma_maj) {
                rec.t = tt;
                rec.p = r.at(tt);
                rec.normal     = vec3(1, 0, 0);  // arbitrary (volume has no surface normal)
                rec.front_face = true;
                rec.mat = make_shared<hg_phase_material>(albedo, phase_g,
                    [this](const ray& sr, double t_max) { return shadow_transmittance_impl(sr, t_max); });
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
    // Same transform convention as rgb_grid_medium_hittable's own
    // world_to_medium() - see that file's comment for the full reasoning.
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

    // Same shape as rgb_grid_medium_hittable's own march_segments() - see
    // that file's comment.
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

    // Single-channel ratio tracking - same technique as cloud_medium_
    // hittable's own shadow_transmittance_impl() (a single shared Tr, not
    // rgb_grid_medium_hittable's per-channel one, since there's only one
    // channel here), bounded by t_max.
    color shadow_transmittance_impl(const ray& r, double t_max) const {
        double mox, moy, moz, mdx, mdy, mdz;
        world_to_medium(r, mox, moy, moz, mdx, mdy, mdz);

        Ray3<double> mray(mox, moy, moz, mdx, mdy, mdz);
        double tMin, tMax;
        if (!grid.intersect_ray(mray, t_max, tMin, tMax)) return color(1, 1, 1);
        if (tMin < 0) tMin = 0;
        if (tMin >= tMax) return color(1, 1, 1);

        double Tr = 1.0;
        constexpr int kMaxNullCollisions = 100000;
        int collisions = 0;

        march_segments(mray, tMin, tMax, [&](double tt, double seg_sigma_maj) {
            if (++collisions > kMaxNullCollisions) return true;  // bail out, same as cloud's cap

            double px = mox + tt*mdx, py = moy + tt*mdy, pz = moz + tt*mdz;
            double sa, ss;
            grid.sample_sigma(px, py, pz, sa, ss);
            Tr *= 1.0 - ((sa + ss) / seg_sigma_maj);
            if (Tr <= 0.0) return true;  // fully opaque already
            return false;
        });
        return color(Tr, Tr, Tr);
    }

    GridMediumData<double> grid;
    color albedo;
    double phase_g;
    double mat_[9], translate_[3];
    point3 world_min, world_max;
};
