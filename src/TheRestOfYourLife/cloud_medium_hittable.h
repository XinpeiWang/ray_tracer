#pragma once
//==============================================================================================
// cloud_medium_hittable.h -- CPU hittable wrapper around CloudMedium<T>
// (src/shared/cloud_medium.h), a real heterogeneous, Perlin-FBm-density
// participating medium (mirrors pbrt-v4 CloudMedium, media.h S11.4).
//
// Unlike constant_medium (single fixed density -> simple exponential
// free-path sampling), a heterogeneous medium's density varies per point, so
// a free-path sample can't be drawn from a single exponential distribution
// directly. This uses delta tracking / null-collision sampling (pbrt-v4
// SampleT_maj, media.h S11.4 / integrators.cpp VolPathIntegrator::Li):
// march using the medium's MAJORANT sigma_t (sigma_a+sigma_s, an upper
// bound valid everywhere since density is clamped to [0,1]), and at each
// candidate point stochastically accept it as a real scattering event with
// probability density(point)*sigma_s/sigma_maj, otherwise treat it as a
// "null collision" and keep marching from there - unbiased, and only needs
// point-wise density evaluation rather than an analytic integral of it.
//
// sigma_a is fixed at 0 (pure scattering, no absorption) - this matches
// constant_medium's own default convention for the same reason: an
// absorption event needs to terminate the path with zero further
// contribution rather than a "no hit", which neither hittable's simple
// hit()->bool interface represents, so both sidestep it by not modelling
// absorption at all rather than getting it subtly wrong.
//==============================================================================================

#include "hittable.h"
#include "constant_medium.h"  // hg_phase_material
#include "../shared/cloud_medium.h"
#include "../shared/ratio_tracking.h"  // RatioTrackingTrHeterogeneous

class cloud_medium_hittable : public hittable {
  public:
    // medium: sigma_a must be 0 (see file comment). world_min/world_max is
    // the medium's world-space AABB for bounding_box() - kept separate from
    // medium.bounds_min/max (which are in *medium* space, generally [0,1]^3)
    // rather than inverting medium's affine transform to recover it.
    cloud_medium_hittable(const CloudMedium<double>& medium, const color& albedo,
                          const point3& world_min, const point3& world_max)
        : cloud(medium), world_min(world_min), world_max(world_max) {
        phase_mat = make_shared<hg_phase_material>(albedo, medium.phase_g,
            [this](const ray& r, double t_max) { return shadow_transmittance_impl(r, t_max); });
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        // Work in world distance: a camera ray's direction is not unit length (it is pixel_sample - origin),
        // and sigma is per world unit, so sampling along the raw parameter made the medium |d| times too thin for
        // primary rays. constant_medium converts by ray_length the same way. rec.t goes back to ray-parameter units.
        const double len = r.direction().length();
        if (!(len > 0.0)) return false;
        const ray ur(r.origin(), r.direction() / len, r.time());
        const double t_lo = ray_t.min * len, t_hi = ray_t.max * len;
        double ray_o[3] = { ur.origin().x(), ur.origin().y(), ur.origin().z() };
        double ray_d[3] = { ur.direction().x(), ur.direction().y(), ur.direction().z() };

        auto maj_it = cloud.sample_ray(ray_o, ray_d, t_hi);
        double tMin, tMax, sigma_maj;
        if (!maj_it.next(tMin, tMax, sigma_maj)) return false;
        if (sigma_maj <= 0.0) return false;
        if (tMin < t_lo) tMin = t_lo;
        if (tMin >= tMax) return false;

        double t = tMin;
        while (true) {
            double dt = -std::log(1.0 - random_double()) / sigma_maj;
            t += dt;
            if (t >= tMax) return false;  // exited the majorant segment: no interaction

            point3 p = ur.at(t);
            double mx, my, mz;
            cloud.world_to_medium_pt(p.x(), p.y(), p.z(), mx, my, mz);
            double d = cloud.compute_density(mx, my, mz);
            double sigma_s_local = d * cloud.sigma_s;

            if (random_double() < sigma_s_local / sigma_maj) {
                // Real scattering event.
                rec.t = t / len;
                rec.p = p;
                rec.normal    = vec3(1, 0, 0);  // arbitrary (volume has no surface normal)
                rec.front_face = true;
                rec.mat       = phase_mat;
                return true;
            }
            // Else a null collision: keep marching from t (unbiased delta tracking).
        }
    }

    aabb bounding_box() const override {
        return aabb(world_min, world_max);
    }

  private:
    // Ratio-tracking transmittance estimator (Novak et al.; pbrt-v4
    // VolPathIntegrator's SampleLd uses the same technique), bounded by
    // t_max - the shadow ray's real target distance (e.g. a punctual
    // light) - rather than always integrating to the medium's true exit.
    // Routed through src/shared/ratio_tracking.h's
    // RatioTrackingTrHeterogeneous(), the generic heterogeneous-medium
    // overload that file's own RatioTrackingTr() comment anticipated
    // ("can be added when a HeterogeneousMediumData type is introduced") -
    // this is that overload, parameterized by callables instead of tied to
    // one medium's density representation, since CloudMedium's procedural
    // FBm density and rgb_grid_medium_hittable's per-voxel grid don't share
    // a lookup interface.
    //
    // Single majorant segment only, matching hit()'s own single
    // maj_it.next() call above - see this file's header comment on
    // CloudMedium's majorant iterator; a multi-segment cloud medium would
    // need both this and hit() extended together, out of scope here.
    color shadow_transmittance_impl(const ray& r, double t_max) const {
        // World distance, as in hit() - the shadow ray's direction is not unit length either.
        const double len = r.direction().length();
        if (!(len > 0.0)) return color(1, 1, 1);
        const ray ur(r.origin(), r.direction() / len, r.time());
        const double t_max_w = t_max * len;
        double ray_o[3] = { ur.origin().x(), ur.origin().y(), ur.origin().z() };
        double ray_d[3] = { ur.direction().x(), ur.direction().y(), ur.direction().z() };

        auto maj_it = cloud.sample_ray(ray_o, ray_d, t_max_w);
        double tMin, tMax, sigma_maj;
        if (!maj_it.next(tMin, tMax, sigma_maj)) return color(1, 1, 1);
        if (sigma_maj <= 0.0) return color(1, 1, 1);
        if (tMin < 0) tMin = 0;
        if (tMin >= tMax) return color(1, 1, 1);

        // Safety cap, same spirit as shadow_ray.h's kMaxTransmissiveSkips -
        // bounds worst-case cost for a pathologically dense/thick medium
        // rather than looping until Tr underflows to exactly 0.
        constexpr int kMaxNullCollisions = 100000;

        double Tr = RatioTrackingTrHeterogeneous<double>(
            tMin, tMax, sigma_maj,
            []() { return random_double(); },
            [&](double t) {
                point3 p = ur.at(t);
                double mx, my, mz;
                cloud.world_to_medium_pt(p.x(), p.y(), p.z(), mx, my, mz);
                double d = cloud.compute_density(mx, my, mz);
                return d * cloud.sigma_s;  // sigma_a=0 (file header comment)
            },
            kMaxNullCollisions);

        return color(Tr, Tr, Tr);  // grayscale density, no per-channel sigma_t here
    }

    CloudMedium<double>          cloud;
    shared_ptr<hg_phase_material> phase_mat;
    point3 world_min, world_max;
};
