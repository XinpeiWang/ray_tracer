#ifndef POWER_LIGHT_SAMPLER_H
#define POWER_LIGHT_SAMPLER_H
//==============================================================================================
// power_light_sampler.h -- Power-weighted light sampler (pbrt-v4 PowerLightSampler pattern)
//
// Mirrors pbrt-v4 PowerLightSampler design:
//   phi_i = area_i * Le_avg_i * pi   (light's emitted power, same as light.Phi() in pbrt-v4)
//   P(light_i) = phi_i / sum(phi_j)  (selection probability)
//
// Sampling uses an AliasTable (O(1)) -- direct port of pbrt-v4 AliasTable.
// Power must be supplied at add() time via add(object, power).
// See build_cornell_box_power_lights() in cornell_box_scene.h for the reference usage.
//
// Intentional differences from pbrt-v4:
//   - Spectral Phi(): we use RGB luminance (0.2126R + 0.7152G + 0.0722B); no spectral yet
//
// pdf_value() returns the correctly weighted marginal PDF over all lights:
//   p(omega) = sum_i  P(light_i) * p_i(omega | light_i)
//==============================================================================================

#include "hittable.h"
#include "hittable_list.h"
#include "rtweekend.h"

// AliasTable used to live here as its own hand-ported copy - an independent
// implementation of the identical pbrt-v4 AliasTable algorithm that
// src/shared/reservoir_sampler.h also carries, unnamespaced under the same
// global class name. Neither header could be included in the same
// translation unit as the other without an ODR collision, which is the
// entire reason src/TheRestOfYourLife/bdpt_render_bridge.h's two-.cpp-file
// split exists (see its own file comment) - scene_registry.h needs THIS
// header (for build_cornell_box_power_lights()) while bdpt_adapter.h needs
// reservoir_sampler.h's (via src/shared/mlt.h), and the two used to be
// mutually exclusive. reservoir_sampler.h's version is a strict superset
// (optional out_pmf/out_u_remapped params on sample(), an (ptr,n)
// constructor, empty()) of what this file's call sites (power_light_list's
// sample(random_double())/pmf(i)/size()) ever used, so including it here
// instead is a drop-in replacement, not a behavior change.
#include "../shared/reservoir_sampler.h"

#include <vector>
#include <numeric>
#include <algorithm>


// ---------------------------------------------------------------------------
// power_light_list -- drop-in replacement for hittable_list used as
// the `lights` parameter in ray_color().
// ---------------------------------------------------------------------------
class power_light_list : public hittable {
  public:
    power_light_list() {}

    // Build from an existing hittable_list with equal (uniform) weights
    explicit power_light_list(const hittable_list& list) {
        for (const auto& obj : list.objects)
            add(obj, 1.0);
    }

    // Register a light with its emitted-power estimate phi = area * Le * pi
    void add(shared_ptr<hittable> object, double power = 1.0) {
        objects.push_back(object);
        raw_powers.push_back(power > 1e-6 ? power : 1e-6);
        bbox = aabb(bbox, object->bounding_box());
        rebuild_alias_table();
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;
        for (const auto& obj : objects) {
            if (obj->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }

    aabb bounding_box() const override { return bbox; }

    // Power-weighted O(1) random direction toward a sampled light
    vec3 random(const point3& origin) const override {
        if (objects.empty()) return vec3(1, 0, 0);
        int idx = alias.sample(random_double());
        return objects[idx]->random(origin);
    }

    // Power-weighted PDF: sum_i P(light_i) * p_i(omega | light_i)
    double pdf_value(const point3& origin, const vec3& direction) const override {
        if (objects.empty()) return 0.0;
        double sum = 0.0;
        for (int i = 0; i < (int)objects.size(); ++i)
            sum += alias.pmf(i) * objects[i]->pdf_value(origin, direction);
        return sum;
    }

    int light_count() const { return (int)objects.size(); }

    // Expose per-light selection PMF for testing (mirrors pbrt-v4 lightSampler.PMF())
    double alias_pmf(int i) const { return alias.pmf(i); }

    std::vector<shared_ptr<hittable>> objects;

  private:
    aabb bbox;
    std::vector<double> raw_powers;
    AliasTable alias;

    void rebuild_alias_table() {
        alias = AliasTable(raw_powers);
    }
};


#endif
