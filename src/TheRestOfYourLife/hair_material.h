#pragma once
// ---------------------------------------------------------------------------
// hair_material.h  --  CPU material wrapper for HairBxDF<double>
//
// Mirrors pbrt-v4's HairMaterial.
//
// Parameters exposed to the scene:
//   sigma_a_r/g/b  : absorption coefficients (default: dark brown)
//   beta_m         : longitudinal roughness [0,1] (default: 0.3)
//   beta_n         : azimuthal roughness    [0,1] (default: 0.3)
//   alpha          : scale tilt in degrees  (default: 2.0)
//   eta            : fiber IOR              (default: 1.55)
//
// The cross-section offset h is sampled uniformly in [-1,1] per ray.
//
// tangent_is_dpdu (default false, every pre-existing call site unaffected):
// the fiber tangent axis HairBxDF needs is fundamentally a "along the
// strand's length" direction, not a surface normal. By default this uses
// the shading normal as a PROXY (artistically valid for fur/hair painted on
// ordinary geometry like a sphere, where normals radiate outward in roughly
// the right pattern for a furry look - this project's own native "Hair
// Fibers" demo, scene 19/B11, relies on exactly this). Set true only when
// the hit really is genuine curve/strand geometry (curve_shape_hittable,
// whose hit_record::dpdu IS the real fiber tangent - see that class's own
// hit() for where it's set) - see pbrt_cpu_builder.h's curve-building loop,
// the one call site that passes true, for why this can't just always read
// dpdu (every OTHER shape's dpdu means something else entirely, e.g. a
// sphere's own around-the-equator tangent, which would misorient the
// Marschner highlight just as much as the normal proxy does, differently).
// ---------------------------------------------------------------------------

#include "material.h"
#include "../shared/bxdfs.h"
#include "../shared/material_context.h"

class hair_material : public material {
public:
	// Convenience constructor
	hair_material(
		double sigma_a_r = 0.06,
		double sigma_a_g = 0.10,
		double sigma_a_b = 0.20,
		double beta_m    = 0.30,
		double beta_n    = 0.30,
		double alpha_deg = 2.0,
		double eta       = 1.55,
		bool   tangent_is_dpdu = false)
		: m_sar(sigma_a_r), m_sag(sigma_a_g), m_sab(sigma_a_b),
		  m_beta_m(beta_m), m_beta_n(beta_n),
		  m_alpha(alpha_deg), m_eta(eta),
		  m_tangent_is_dpdu(tangent_is_dpdu)
	{}

	bool scatter(const ray& r_in, const hit_record& rec,
				 scatter_record& srec, bool do_regularize = false) const override
	{
		// Sample cross-section offset h uniformly in [-1, 1]
		double h = random_double() * 2.0 - 1.0;

		double beta_m = do_regularize ? std::max(m_beta_m, 0.3) : m_beta_m;
		double beta_n = do_regularize ? std::max(m_beta_n, 0.3) : m_beta_n;

		HairBxDF<double> bxdf(
			h, m_eta,
			m_sar, m_sag, m_sab,
			beta_m, beta_n,
			m_alpha);

		// See tangent_is_dpdu's own comment above. A degenerate dpdu (only
		// possible if a future non-curve caller ever passed tangent_is_dpdu
		// without a real dpdu set) falls back to the normal proxy rather than
		// sampling from a zero-length axis.
		const vec3& tangent = (m_tangent_is_dpdu && rec.dpdu.length_squared() > 1e-12)
			? rec.dpdu : rec.normal;
		double tx = tangent.x();
		double ty = tangent.y();
		double tz = tangent.z();

		vec3 in_dir = unit_vector(r_in.direction());
		double u1 = random_double();
		double u2 = random_double();
		double u3 = random_double();
		double u4 = random_double();

		auto res = bxdf.sample(
			tx, ty, tz,
			in_dir.x(), in_dir.y(), in_dir.z(),
			u1, u2, u3, u4);

		if (!res.valid) return false;

		srec.attenuation     = color(res.r, res.g, res.b);
		srec.pdf_ptr         = nullptr;
		srec.skip_pdf        = true;
		srec.skip_pdf_ray    = ray(rec.p, vec3(res.wo_x, res.wo_y, res.wo_z), r_in.time());
		srec.eta             = 1.0;
		srec.is_transmission = false;
		return true;
	}

	double scattering_pdf(const ray& r_in, const hit_record& rec,
						  const ray& scattered) const override
	{
		// h=0 for deterministic PDF evaluation
		HairBxDF<double> bxdf(
			0.0, m_eta,
			m_sar, m_sag, m_sab,
			m_beta_m, m_beta_n,
			m_alpha);

		// See scatter()'s identical tangent selection above.
		const vec3& tangent = (m_tangent_is_dpdu && rec.dpdu.length_squared() > 1e-12)
			? rec.dpdu : rec.normal;
		double tx = tangent.x(), ty = tangent.y(), tz = tangent.z();
		vec3 in_dir  = unit_vector(r_in.direction());
		vec3 out_dir = unit_vector(scattered.direction());

		return bxdf.scattering_pdf(
			tx, ty, tz,
			in_dir.x(), in_dir.y(), in_dir.z(),
			out_dir.x(), out_dir.y(), out_dir.z());
	}

	// Accessors
	double sigma_a_r() const { return m_sar; }
	double sigma_a_g() const { return m_sag; }
	double sigma_a_b() const { return m_sab; }
	double get_beta_m() const { return m_beta_m; }
	double get_beta_n() const { return m_beta_n; }
	double get_alpha()  const { return m_alpha; }
	double get_eta()    const { return m_eta; }
	bool   tangent_is_dpdu() const { return m_tangent_is_dpdu; }

private:
	double m_sar, m_sag, m_sab;
	double m_beta_m, m_beta_n;
	double m_alpha;
	double m_eta;
	bool   m_tangent_is_dpdu;
};
