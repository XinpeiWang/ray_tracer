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
// The fiber tangent uses the shading normal as a proxy (artistically valid
// for fur/hair when normals are set along the strand direction).
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
		double eta       = 1.55)
		: m_sar(sigma_a_r), m_sag(sigma_a_g), m_sab(sigma_a_b),
		  m_beta_m(beta_m), m_beta_n(beta_n),
		  m_alpha(alpha_deg), m_eta(eta)
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

		// Use the shading normal as fiber tangent proxy
		double tx = rec.normal.x();
		double ty = rec.normal.y();
		double tz = rec.normal.z();

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

		double tx = rec.normal.x(), ty = rec.normal.y(), tz = rec.normal.z();
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

private:
	double m_sar, m_sag, m_sab;
	double m_beta_m, m_beta_n;
	double m_alpha;
	double m_eta;
};
