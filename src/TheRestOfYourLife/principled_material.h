#pragma once
// principled_material -- Disney/pbrt-v4 CoatedDiffuse/CoatedConductor style
//
// Self-contained: #include "principled_material.h" is sufficient.
// Provides the `principled` CPU material class backed by PrincipledBxDF<double>.
//
// Parameters (all accept shared_ptr<texture> or solid_color):
//   base_tex          -- base / diffuse color (default white)
//   metallic_tex      -- metallic factor in R channel (0=plastic, 1=metal)
//   roughness_tex     -- perceptual roughness in R channel
//   ior               -- interface IOR (default 1.5, F0 ≈ 0.04)
//   clearcoat_tex     -- clearcoat weight in R channel (0=off)
//   clearcoat_rough   -- clearcoat roughness scalar (default 0.1)
//
// pbrt-v4 alignment:
//   metallic=0  →  CoatedDiffuseMaterial  (dielectric + diffuse substrate)
//   metallic=1  →  CoatedConductorMaterial (dielectric + conductor substrate)
//   blend in between is the full Disney model

#include "material.h"
#include "../shared/bxdfs.h"
#include "../shared/material_context.h"

class principled : public material {
  public:
	using BxDF = PrincipledBxDF<double>;

	// Convenience constructor: solid colors for quick scene setup
	principled(color base,
			   double metallic      = 0.0,
			   double roughness     = 0.5,
			   double ior           = 1.5,
			   double clearcoat     = 0.0,
			   double clearcoat_rough = 0.1)
		: base_tex(make_shared<solid_color>(base)),
		  metallic_tex(make_shared<solid_color>(color(metallic, metallic, metallic))),
		  roughness_tex(make_shared<solid_color>(color(roughness, roughness, roughness))),
		  ior(ior),
		  clearcoat_tex(make_shared<solid_color>(color(clearcoat, clearcoat, clearcoat))),
		  clearcoat_rough(clearcoat_rough) {}

	// Full texture constructor
	principled(shared_ptr<texture> base_tex,
			   shared_ptr<texture> metallic_tex,
			   shared_ptr<texture> roughness_tex,
			   double ior               = 1.5,
			   shared_ptr<texture> clearcoat_tex  = nullptr,
			   double clearcoat_rough   = 0.1)
		: base_tex(base_tex),
		  metallic_tex(metallic_tex),
		  roughness_tex(roughness_tex),
		  ior(ior),
		  clearcoat_tex(clearcoat_tex
						? clearcoat_tex
						: make_shared<solid_color>(color(0,0,0))),
		  clearcoat_rough(clearcoat_rough) {}

	bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
				 bool do_regularize = false) const override {
		auto ctx = MaterialContext<double>::from_hit(rec, r_in);

		color base      = base_tex    ->value(rec.u, rec.v, rec.p);
		double metallic = metallic_tex ->value(rec.u, rec.v, rec.p).x();
		double rough    = roughness_tex->value(rec.u, rec.v, rec.p).x();
		double cc       = clearcoat_tex->value(rec.u, rec.v, rec.p).x();

		if (do_regularize)
			rough = std::max(rough, 0.3);

		BxDF bxdf{ base.x(), base.y(), base.z(),
				   metallic, rough, ior, cc, clearcoat_rough };

		vec3 in_dir = unit_vector(r_in.direction());
		// Sample: pass 3 randoms; incident = incoming ray direction
		vec3 u = random_unit_vector(); // reuse existing helper for randoms
		double u1 = random_double();
		double u2 = random_double();
		double u3 = random_double();

		auto res = bxdf.sample(
			ctx.nx, ctx.ny, ctx.nz,
			in_dir.x(), in_dir.y(), in_dir.z(),
			u1, u2, u3);

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
						  const ray& scattered) const override {
		auto ctx = MaterialContext<double>::from_hit(rec, r_in);

		color base      = base_tex    ->value(rec.u, rec.v, rec.p);
		double metallic = metallic_tex ->value(rec.u, rec.v, rec.p).x();
		double rough    = roughness_tex->value(rec.u, rec.v, rec.p).x();
		double cc       = clearcoat_tex->value(rec.u, rec.v, rec.p).x();

		BxDF bxdf{ base.x(), base.y(), base.z(),
				   metallic, rough, ior, cc, clearcoat_rough };

		vec3 in_dir  = unit_vector(r_in.direction());
		vec3 out_dir = unit_vector(scattered.direction());

		return bxdf.scattering_pdf(
			ctx.nx, ctx.ny, ctx.nz,
			in_dir.x(), in_dir.y(), in_dir.z(),
			out_dir.x(), out_dir.y(), out_dir.z());
	}

	// Accessors for serialization
	shared_ptr<texture> get_base_tex()     const { return base_tex; }
	shared_ptr<texture> get_metallic_tex() const { return metallic_tex; }
	shared_ptr<texture> get_roughness_tex()const { return roughness_tex; }
	shared_ptr<texture> get_clearcoat_tex()const { return clearcoat_tex; }
	double get_ior()            const { return ior; }
	double get_clearcoat_rough()const { return clearcoat_rough; }

  private:
	shared_ptr<texture> base_tex;
	shared_ptr<texture> metallic_tex;
	shared_ptr<texture> roughness_tex;
	double ior;
	shared_ptr<texture> clearcoat_tex;
	double clearcoat_rough;
};
