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

#include <vector>
#include <numeric>
#include <algorithm>


// ---------------------------------------------------------------------------
// AliasTable -- O(1) weighted sampling, direct port of pbrt-v4 AliasTable
// ---------------------------------------------------------------------------
class AliasTable {
  public:
    AliasTable() = default;

    explicit AliasTable(const std::vector<double>& weights) {
        int n = (int)weights.size();
        bins.resize(n);

        // Normalize weights -> PMF stored in bins[i].p
        double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        if (sum <= 0.0) sum = 1.0;
        for (int i = 0; i < n; ++i)
            bins[i].p = weights[i] / sum;

        // Build alias table using Vose's algorithm (pbrt-v4 pattern)
        struct Outcome { double pHat; int index; };
        std::vector<Outcome> under, over;
        for (int i = 0; i < n; ++i) {
            double pHat = bins[i].p * n;
            if (pHat < 1.0)
                under.push_back({pHat, i});
            else
                over.push_back({pHat, i});
        }

        while (!under.empty() && !over.empty()) {
            Outcome un = under.back(); under.pop_back();
            Outcome ov = over.back();  over.pop_back();

            bins[un.index].q     = un.pHat;
            bins[un.index].alias = ov.index;

            double pExcess = un.pHat + ov.pHat - 1.0;
            if (pExcess < 1.0)
                under.push_back({pExcess, ov.index});
            else
                over.push_back({pExcess, ov.index});
        }
        // Remaining items have full probability in their own bin
        while (!over.empty())  { auto ov = over.back();  over.pop_back();  bins[ov.index].q = 1.0; bins[ov.index].alias = -1; }
        while (!under.empty()) { auto un = under.back(); under.pop_back(); bins[un.index].q = 1.0; bins[un.index].alias = -1; }
    }

    // Sample a bin index in O(1) given a uniform random u in [0,1)
    int sample(double u) const {
        int n = (int)bins.size();
        int offset = (int)(u * n);
        if (offset >= n) offset = n - 1;
        double up = u * n - offset;
        if (up < bins[offset].q)
            return offset;
        return bins[offset].alias;
    }

    // PMF for bin index i
    double pmf(int i) const { return bins[i].p; }

    int size() const { return (int)bins.size(); }

  private:
    struct Bin { double q = 1.0, p = 1.0; int alias = -1; };
    std::vector<Bin> bins;
};


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
