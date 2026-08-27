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
#include "../shared/goniometric_light.h"
#include "../shared/projection_light.h"

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

	// Fixed world-space position and intrinsic (already scale-applied)
	// intensity -- for a caller that needs to emit a ray FROM the light
	// (light-subpath generation) rather than evaluate incident radiance
	// AT a shading point, which sample_direct() above is scoped to.
	point3 position() const { return point3(data.pos_x, data.pos_y, data.pos_z); }
	color  intensity() const { return color(data.ir, data.ig, data.ib) * data.scale; }

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

	// Fixed world-space position -- see point_light_obj::position()'s own
	// comment.
	point3 position() const { return point3(data.pos_x, data.pos_y, data.pos_z); }

	// Forwards SpotLightData::sample_le()/pdf_le() (real cone-importance-
	// sampled emission direction, already implemented there) for light-
	// subpath generation.
	void sample_le(double ru, double rv0, double rv1,
	               vec3& dir, double& pdf_dir) const {
		double wx, wy, wz;
		data.sample_le(ru, rv0, rv1, wx, wy, wz, pdf_dir);
		dir = vec3(wx, wy, wz);
	}
	double pdf_le(const vec3& dir) const { return data.pdf_le(dir.x(), dir.y(), dir.z()); }

	// Peak (on-axis) intensity, already scale-applied -- SampleLe's emitted
	// radiance is this times the same falloff() SampleLi's eval_Li applies,
	// evaluated at the sampled direction instead of a shading-point-derived
	// one.
	color peak_intensity() const { return color(data.ir, data.ig, data.ib) * data.scale; }
	double falloff(const vec3& w) const { return data.falloff(w.x(), w.y(), w.z()); }

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

	// Fixed direction FROM any scene point TOWARD the light (the "wi"
	// convention DistantLightData::dir_x/y/z and sample_wi() already use) --
	// a light-subpath photon leaving this light travels the opposite way,
	// -direction().
	vec3  direction() const { return vec3(data.dir_x, data.dir_y, data.dir_z); }
	color radiance() const { return color(data.ir, data.ig, data.ib) * data.scale; }

  private:
	DistantLightData<double> data;
};

// ---------------------------------------------------------------------------
// 4. CPU goniometric_light_obj  (wraps GoniometricLight<double>)
// ---------------------------------------------------------------------------
class goniometric_light_obj {
  public:
	goniometric_light_obj(const point3& pos, const color& intensity, double scale,
						  const double id[9],
						  const std::vector<double>& image, int nu, int nv) {
		data = GoniometricLight<double>::make(
			pos.x(), pos.y(), pos.z(),
			id,
			intensity.x(), intensity.y(), intensity.z(),
			scale,
			image, nu, nv
		);
	}

	PunctualLiSample sample_direct(const point3& p) const {
		double Lr, Lg, Lb, wx, wy, wz;
		data.sample_li(p.x(), p.y(), p.z(), Lr, Lg, Lb, wx, wy, wz);
		double dx = data.pos_x - p.x(), dy = data.pos_y - p.y(), dz = data.pos_z - p.z();
		double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
		return { color(Lr, Lg, Lb), vec3(wx, wy, wz), 1.0, dist };
	}

	double power() const { return data.power(); }

  private:
	GoniometricLight<double> data;
};

// ---------------------------------------------------------------------------
// 5. CPU projection_light_obj  (wraps ProjectionLight<double>)
// ---------------------------------------------------------------------------
class projection_light_obj {
  public:
	projection_light_obj(const point3& pos, double scale,
						 const double wtl[9],
						 double fov_deg,
						 const std::vector<double>& image_rgb, int nx, int ny) {
		data = ProjectionLight<double>::make(
			pos.x(), pos.y(), pos.z(),
			wtl,
			scale,
			fov_deg,
			image_rgb, nx, ny
		);
	}

	PunctualLiSample sample_direct(const point3& p) const {
		double Lr, Lg, Lb, wx, wy, wz;
		data.sample_li(p.x(), p.y(), p.z(), Lr, Lg, Lb, wx, wy, wz);
		double dx = data.pos_x - p.x(), dy = data.pos_y - p.y(), dz = data.pos_z - p.z();
		double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
		return { color(Lr, Lg, Lb), vec3(wx, wy, wz), 1.0, dist };
	}

	double power() const { return data.power(); }

  private:
	ProjectionLight<double> data;
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

	// Add a goniometric (IES-profile) point light.
	// id[9]: row-major 3x3 world-to-light rotation.
	// ir,ig,ib: base emission colour; gonio pattern modulates this per-direction.
	// image: nu*nv greyscale values (row-major); nullptr or empty = uniform.
	void add_gonio(const point3& pos, const color& intensity, double scale,
				   const double id[9],
				   const std::vector<double>& image, int nu, int nv) {
		gonios.emplace_back(pos, intensity, scale, id, image, nu, nv);
	}

	// Add a projection light (slide-projector frustum).
	// wtl[9]: row-major 3x3 world-to-light rotation.
	// fov_deg: full vertical field-of-view.
	// image_rgb: nx*ny*3 doubles (R,G,B per pixel, row-major).
	void add_projection(const point3& pos, double scale,
						const double wtl[9],
						double fov_deg,
						const std::vector<double>& image_rgb, int nx, int ny) {
		projections.emplace_back(pos, scale, wtl, fov_deg, image_rgb, nx, ny);
	}

	bool empty() const {
		return points.empty() && spots.empty() && distants.empty()
			&& gonios.empty() && projections.empty();
	}

	// Iterate all lights; integrator calls this from NEE block.
	// Callback: void fn(const PunctualLiSample&)
	template<typename Fn>
	void for_each_sample(const point3& shading_p, Fn&& fn) const {
		for (const auto& l : points)      fn(l.sample_direct(shading_p));
		for (const auto& l : spots)       fn(l.sample_direct(shading_p));
		for (const auto& l : distants)    fn(l.sample_direct(shading_p));
		for (const auto& l : gonios)      fn(l.sample_direct(shading_p));
		for (const auto& l : projections) fn(l.sample_direct(shading_p));
	}

	std::vector<point_light_obj>   points;
	std::vector<spot_light_obj>    spots;
	std::vector<distant_light_obj> distants;
	std::vector<goniometric_light_obj> gonios;
	std::vector<projection_light_obj>  projections;
};
