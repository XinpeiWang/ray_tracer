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

class rgb_grid_medium_hittable : public hittable {
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
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        // Transform ray into medium space. Direction transforms by the
        // matrix only (no translation) - same as CloudMedium::world_to_medium_pt.
        // Applying the SAME matrix to both origin and direction means the
        // medium-space ray is still parametrized by the identical `t` as the
        // world-space ray (o+t*d transforms to (mat*o+translate)+t*(mat*d)),
        // so segment/hit `t` values read back directly with no rescaling.
        point3 o = r.origin();
        vec3   d = r.direction();
        double mox = mat_[0]*o.x() + mat_[1]*o.y() + mat_[2]*o.z() + translate_[0];
        double moy = mat_[3]*o.x() + mat_[4]*o.y() + mat_[5]*o.z() + translate_[1];
        double moz = mat_[6]*o.x() + mat_[7]*o.y() + mat_[8]*o.z() + translate_[2];
        double mdx = mat_[0]*d.x() + mat_[1]*d.y() + mat_[2]*d.z();
        double mdy = mat_[3]*d.x() + mat_[4]*d.y() + mat_[5]*d.z();
        double mdz = mat_[6]*d.x() + mat_[7]*d.y() + mat_[8]*d.z();

        Ray3<double> mray(mox, moy, moz, mdx, mdy, mdz);
        double tMin, tMax;
        if (!grid.intersect_ray(mray, ray_t.max, tMin, tMax)) return false;
        if (tMin < ray_t.min) tMin = ray_t.min;
        if (tMin >= tMax) return false;

        auto it = grid.sample_ray(mray, tMin, tMax);
        for (;;) {
            auto seg = it.Next();
            if (!seg.has_value()) return false;  // ray exited the medium's AABB
            if (seg->sigma_maj <= 0.0) continue; // empty segment, try the next one

            double tt = seg->tMin;
            for (;;) {
                double dt = -std::log(1.0 - random_double()) / seg->sigma_maj;
                tt += dt;
                if (tt >= seg->tMax) break;  // exited this segment - advance to the next

                double px = mox + tt*mdx, py = moy + tt*mdy, pz = moz + tt*mdz;
                double sa[3], ss[3], le[3];
                grid.sample_point(px, py, pz, sa, ss, le);
                double sigma_t_local = std::max({ sa[0]+ss[0], sa[1]+ss[1], sa[2]+ss[2] });

                if (random_double() < sigma_t_local / seg->sigma_maj) {
                    rec.t = tt;
                    rec.p = r.at(tt);
                    rec.normal     = vec3(1, 0, 0);  // arbitrary (volume has no surface normal)
                    rec.front_face = true;

                    double maxc = std::max({ ss[0], ss[1], ss[2], 1e-6 });
                    color albedo(ss[0]/maxc, ss[1]/maxc, ss[2]/maxc);
                    rec.mat = make_shared<hg_phase_material>(albedo, phase_g);
                    return true;
                }
                // Null collision: keep marching within this segment.
            }
        }
    }

    aabb bounding_box() const override {
        return aabb(world_min, world_max);
    }

  private:
    RGBGridMediumData<double> grid;
    double phase_g;
    double mat_[9], translate_[3];
    point3 world_min, world_max;
};
