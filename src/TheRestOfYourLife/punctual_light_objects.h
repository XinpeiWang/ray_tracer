#pragma once
// ---------------------------------------------------------------------------
// punctual_light_objects.h -- CPU wrappers for pbrt-v4 punctual light types
//
// Wraps PointLightData<double>, SpotLightData<double>, DistantLightData<double>
// from src/shared/punctual_lights.h into the CPU integrator interface.
//
// Punctual lights are delta distributions (PDF = 0 from any surface sample).
// The integrator handles them via a dedicated NEE shadow ray strategy that
// calls sample_direct() rather than the area-light hittable_pdf path.
// This mirrors pbrt-v4's treatment of DeltaPosition / DeltaDirection lights.
//
// Usage (camera.h):
//   // Build lights
//   auto plist = make_shared<punctual_light_list>();
//   plist->add_point(point3(278,555,278), color(1,1,1), 100000.0);
//   plist->add_spot(...);
//   plist->add_distant(...);
//
//   // In ray_color loop, after BSDF scatter, before path continuation:
//   for each light in plist:
//       PunctualLiSample s = light.sample_direct(rec.p);
//       if shadow ray clears: L += beta * s.Li * BSDF_f / s.pdf   (pdf=1 for delta)
// ---------------------------------------------------------------------------

#include "rtweekend.h"
#include "hittable.h"
#include "../shared/punctual_lights.h"

#include <vector>
#include <memory>
#include <cmath>

// Result of one punctual-light direct sample
struct PunctualLiSample {
	color  Li;        // incident radiance from this light at the shading point
	vec3   wi;        // unit world-space direction from shading point toward light
	double pdf;       // 1.0 for delta lights (caller divides by this)
	double t_max;     // maximum shadow-ray t (distance to light, infinity for distant)
};

// ---------------------------------------------------------------------------
// 1. CPU point_light_obj  (wraps PointLightData<double>)
// ---------------------------------------------------------------------------
class point_light_obj {
  public:
	point_light_obj(const point3& pos, const color& intensity, double scale = 1.0) {
		data.pos_x = pos.x(); data.pos_y = pos.y(); data.pos_z = pos.z();
		data.ir = intensity.x(); data.ig = intensity.y(); data.ib = intensity.z();
		data.scale = scale;
	}

	PunctualLiSample sample_direct(const point3& p) const {
		double wx, wy, wz;
		data.sample_wi(p.x(), p.y(), p.z(), wx, wy, wz);

		double Lr, Lg, Lb;
		data.eval_Li(p.x(), p.y(), p.z(), Lr, Lg, Lb);

		// t_max = distance to light
		double dx = data.pos_x - p.x(), dy = data.pos_y - p.y(), dz = data.pos_z - p.z();
		double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

		return { color(Lr, Lg, Lb), vec3(wx, wy, wz), 1.0, dist };
	}

	double power() const { return data.power(); }

  private:
	PointLightData<double> data;
};

// ---------------------------------------------------------------------------
// 2. CPU spot_light_obj  (wraps SpotLightData<double>)
// ---------------------------------------------------------------------------
class spot_light_obj {
  public:
	// pos:           world-space position of the light
	// dir:           unit direction the cone points toward (toward scene)
	// intensity:     peak intensity (RGB, at cone center)
	// total_width:   outer half-angle of the cone in degrees  (pbrt-v4: totalWidth)
	// falloff_start: inner half-angle in degrees              (pbrt-v4: falloffStart)
	// scale:         global multiplier
	spot_light_obj(const point3& pos, const vec3& dir,
				   const color& intensity,
				   double total_width_deg, double falloff_start_deg,
				   double scale = 1.0) {
		double tw = degrees_to_radians(total_width_deg);
		double fs = degrees_to_radians(falloff_start_deg);
		vec3 d = unit_vector(dir);
		data.pos_x = pos.x(); data.pos_y = pos.y(); data.pos_z = pos.z();
		data.dir_x = d.x();   data.dir_y = d.y();   data.dir_z = d.z();
		data.ir = intensity.x(); data.ig = intensity.y(); data.ib = intensity.z();
		data.scale = scale;
		data.cos_falloff_start = std::cos(fs);
		data.cos_falloff_end   = std::cos(tw);
	}

	PunctualLiSample sample_direct(const point3& p) const {
		double wx, wy, wz;
		data.sample_wi(p.x(), p.y(), p.z(), wx, wy, wz);

		double Lr, Lg, Lb;
		data.eval_Li(p.x(), p.y(), p.z(), Lr, Lg, Lb);

		double dx = data.pos_x - p.x(), dy = data.pos_y - p.y(), dz = data.pos_z - p.z();
		double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

		return { color(Lr, Lg, Lb), vec3(wx, wy, wz), 1.0, dist };
	}

	double power() const { return data.power(); }

  private:
	SpotLightData<double> data;
};

// ---------------------------------------------------------------------------
// 3. CPU distant_light_obj  (wraps DistantLightData<double>)
// ---------------------------------------------------------------------------
class distant_light_obj {
  public:
	// dir:           unit direction *toward* the scene (wi direction in integrator)
	// radiance:      RGB radiance
	// scene_radius:  bounding sphere radius of the scene (for power estimate)
	// scale:         global multiplier
	distant_light_obj(const vec3& dir, const color& radiance,
					  double scene_radius = 1000.0, double scale = 1.0) {
		vec3 d = unit_vector(dir);
		data.dir_x = d.x(); data.dir_y = d.y(); data.dir_z = d.z();
		data.ir = radiance.x(); data.ig = radiance.y(); data.ib = radiance.z();
		data.scale = scale;
		data.scene_radius = scene_radius;
	}

	PunctualLiSample sample_direct(const point3& /*p*/) const {
		double Lr, Lg, Lb;
		data.eval_Li(Lr, Lg, Lb);
		double wx, wy, wz;
		data.sample_wi(wx, wy, wz);
		return { color(Lr, Lg, Lb), vec3(wx, wy, wz), 1.0, infinity };
	}

	double power() const { return data.power(); }

  private:
	DistantLightData<double> data;
};

// ---------------------------------------------------------------------------
// punctual_light_list -- heterogeneous container of all punctual lights.
// Call sample_all_direct() in the integrator NEE loop.
// ---------------------------------------------------------------------------
class punctual_light_list {
  public:
	void add_point(const point3& pos, const color& intensity, double scale = 1.0) {
		points.emplace_back(pos, intensity, scale);
	}
	void add_spot(const point3& pos, const vec3& dir, const color& intensity,
				  double total_width_deg, double falloff_start_deg, double scale = 1.0) {
		spots.emplace_back(pos, dir, intensity, total_width_deg, falloff_start_deg, scale);
	}
	void add_distant(const vec3& dir, const color& radiance,
					 double scene_radius = 1000.0, double scale = 1.0) {
		distants.emplace_back(dir, radiance, scene_radius, scale);
	}

	bool empty() const { return points.empty() && spots.empty() && distants.empty(); }

	// Iterate all lights; integrator calls this from NEE block.
	// Callback: void fn(const PunctualLiSample&)
	template<typename Fn>
	void for_each_sample(const point3& shading_p, Fn&& fn) const {
		for (const auto& l : points)   fn(l.sample_direct(shading_p));
		for (const auto& l : spots)    fn(l.sample_direct(shading_p));
		for (const auto& l : distants) fn(l.sample_direct(shading_p));
	}

	std::vector<point_light_obj>   points;
	std::vector<spot_light_obj>    spots;
	std::vector<distant_light_obj> distants;
};
