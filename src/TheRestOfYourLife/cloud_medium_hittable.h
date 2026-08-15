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

class cloud_medium_hittable : public hittable {
  public:
    // medium: sigma_a must be 0 (see file comment). world_min/world_max is
    // the medium's world-space AABB for bounding_box() - kept separate from
    // medium.bounds_min/max (which are in *medium* space, generally [0,1]^3)
    // rather than inverting medium's affine transform to recover it.
    cloud_medium_hittable(const CloudMedium<double>& medium, const color& albedo,
                          const point3& world_min, const point3& world_max)
        : cloud(medium), world_min(world_min), world_max(world_max) {
        phase_mat = make_shared<hg_phase_material>(albedo, medium.phase_g);
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        double ray_o[3] = { r.origin().x(), r.origin().y(), r.origin().z() };
        double ray_d[3] = { r.direction().x(), r.direction().y(), r.direction().z() };

        auto maj_it = cloud.sample_ray(ray_o, ray_d, ray_t.max);
        double tMin, tMax, sigma_maj;
        if (!maj_it.next(tMin, tMax, sigma_maj)) return false;
        if (sigma_maj <= 0.0) return false;
        if (tMin < ray_t.min) tMin = ray_t.min;
        if (tMin >= tMax) return false;

        double t = tMin;
        while (true) {
            double dt = -std::log(1.0 - random_double()) / sigma_maj;
            t += dt;
            if (t >= tMax) return false;  // exited the majorant segment: no interaction

            point3 p = r.at(t);
            double mx, my, mz;
            cloud.world_to_medium_pt(p.x(), p.y(), p.z(), mx, my, mz);
            double d = cloud.compute_density(mx, my, mz);
            double sigma_s_local = d * cloud.sigma_s;

            if (random_double() < sigma_s_local / sigma_maj) {
                // Real scattering event.
                rec.t = t;
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
    CloudMedium<double>          cloud;
    shared_ptr<hg_phase_material> phase_mat;
    point3 world_min, world_max;
};
