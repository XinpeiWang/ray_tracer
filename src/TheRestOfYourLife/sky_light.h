#ifndef SKY_LIGHT_H
#define SKY_LIGHT_H
//==============================================================================
// sky_light -- HDR equirectangular environment / infinite area light
//
// Mirrors pbrt-v4 ImageInfiniteLight (uniform-sampling variant).
//
// Le(dir)          -- radiance arriving from direction dir
// pdf_Li()         -- uniform sphere PDF = 1/(4*pi)
// sample_Li()      -- returns a uniform random direction + its PDF
//
// Equirectangular mapping (same convention as pbrt-v4 §12.5):
//   theta = acos(clamp(-dir.y, -1, 1))       [0, pi]
//   phi   = atan2(-dir.z, dir.x) + pi        [0, 2*pi]
//   u     = phi  / (2*pi)
//   v     = theta / pi
//
// For scenes without an HDR file, a solid-color sky_light can be used:
//   sky_light(color(0.5, 0.7, 1.0))
//==============================================================================

#include "rtweekend.h"
#include "texture.h"

class sky_light {
  public:
	// Construct from a pre-built texture (e.g. image_texture or solid_color)
	explicit sky_light(shared_ptr<texture> tex) : env_tex(tex), scale(1.0) {}

	// Convenience: constant-color sky
	explicit sky_light(const color& c)
		: env_tex(make_shared<solid_color>(c)), scale(1.0) {}

	// Convenience: HDR file -- uses hdr_image_texture to preserve float values > 1.
	// Equivalent to pbrt-v4 ImageInfiniteLight loading an HDR latlong map.
	explicit sky_light(const char* hdr_filename, double brightness = 1.0)
		: env_tex(make_shared<hdr_image_texture>(hdr_filename)), scale(brightness) {}

	// Radiance arriving from world direction `dir` (unit vector expected)
	color Le(const vec3& dir) const {
		auto [u, v] = dir_to_uv(dir);
		return scale * env_tex->value(u, v, point3(0, 0, 0));
	}

	// Uniform sphere PDF (pbrt-v4 UniformInfiniteLight::PDF_Li = 1/(4*pi))
	double pdf_Li() const { return 1.0 / (4.0 * pi); }

	// Sample a direction toward the sky; returns (direction, pdf)
	vec3 sample_Li() const { return random_unit_vector(); }

  private:
	shared_ptr<texture> env_tex;
	double              scale;

	// Convert unit direction to equirectangular (u,v) in [0,1]^2
	static std::pair<double,double> dir_to_uv(const vec3& d) {
		// theta in [0,pi], phi in [0,2*pi]
		double cos_theta = -d.y();
		// clamp to [-1,1] to guard against floating-point drift
		if (cos_theta >  1.0) cos_theta =  1.0;
		if (cos_theta < -1.0) cos_theta = -1.0;
		double theta = std::acos(cos_theta);
		double phi   = std::atan2(-d.z(), d.x()) + pi;

		double u = phi   / (2.0 * pi);
		double v = theta / pi;
		return { u, v };
	}
};

#endif
