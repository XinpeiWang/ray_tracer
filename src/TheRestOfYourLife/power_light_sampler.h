#ifndef POWER_LIGHT_SAMPLER_H
#define POWER_LIGHT_SAMPLER_H
//==============================================================================================
// power_light_sampler.h -- Power-weighted light sampler (pbrt-v4 PowerLightSampler pattern)
//
// Instead of selecting lights uniformly (1/N), this sampler uses a CDF weighted by each
// light's emitted power.  Power must be supplied at add() time via add(object, power).
// Use build_cornell_box_lights() (in cornell_box_scene.h) as the reference implementation.
//
// pdf_value() returns the correctly weighted PDF so the estimator stays unbiased.
// Drop-in replacement for hittable_list as the `lights` parameter in ray_color().
//==============================================================================================

#include "hittable.h"
#include "hittable_list.h"
#include "rtweekend.h"

#include <vector>
#include <numeric>


class power_light_list : public hittable {
  public:
	power_light_list() {}

	// Build from an existing hittable_list with equal (uniform) weights
	explicit power_light_list(const hittable_list& list) {
		for (const auto& obj : list.objects)
			add(obj, 1.0);
	}

	// Register a light with a power estimate (area * luminance for area lights)
	void add(shared_ptr<hittable> object, double power = 1.0) {
		objects.push_back(object);
		raw_powers.push_back(power > 1e-6 ? power : 1e-6);
		bbox = aabb(bbox, object->bounding_box());
		rebuild_cdf();
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

	// Power-weighted random direction toward a sampled light
	vec3 random(const point3& origin) const override {
		if (objects.empty()) return vec3(1, 0, 0);
		return objects[sample_light_index()]->random(origin);
	}

	// Power-weighted PDF: weighted sum of each light's PDF
	double pdf_value(const point3& origin, const vec3& direction) const override {
		if (objects.empty()) return 0.0;
		double sum = 0.0;
		for (int i = 0; i < (int)objects.size(); ++i)
			sum += weights[i] * objects[i]->pdf_value(origin, direction);
		return sum;
	}

	int light_count() const { return (int)objects.size(); }

	std::vector<shared_ptr<hittable>> objects;

  private:
	aabb bbox;
	std::vector<double> raw_powers;
	std::vector<double> weights;
	std::vector<double> cdf;

	void rebuild_cdf() {
		int n = (int)raw_powers.size();
		double total = std::accumulate(raw_powers.begin(), raw_powers.end(), 0.0);
		if (total <= 0.0) total = 1.0;
		weights.resize(n);
		cdf.resize(n);
		for (int i = 0; i < n; ++i)
			weights[i] = raw_powers[i] / total;
		cdf[0] = weights[0];
		for (int i = 1; i < n; ++i)
			cdf[i] = cdf[i-1] + weights[i];
		cdf.back() = 1.0;
	}

	int sample_light_index() const {
		double xi = random_double();
		for (int i = 0; i < (int)cdf.size(); ++i)
			if (xi <= cdf[i]) return i;
		return (int)objects.size() - 1;
	}
};


#endif

