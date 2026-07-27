#pragma once
// normal_map_material and bump_map_material -- pbrt-v4 NormalMap / BumpMap
// Included by material.h after all base types are defined.
// Guard: only compile the class bodies when material.h's types are available.
#ifdef MATERIAL_H

class normal_map_material : public material {
  public:
	normal_map_material(shared_ptr<texture> normal_tex, shared_ptr<material> inner)
		: normal_tex(normal_tex), inner(inner) {}

	// Perturb rec.normal using the tangent-space RGB normal map texture.
	hit_record apply(const hit_record& rec) const {
		color packed = normal_tex->value(rec.u, rec.v, rec.p);
		double ns_tx = 2.0*packed.x() - 1.0;
		double ns_ty = 2.0*packed.y() - 1.0;
		double ns_tz = 2.0*packed.z() - 1.0;
		double ns_len = std::sqrt(ns_tx*ns_tx + ns_ty*ns_ty + ns_tz*ns_tz);
		if (ns_len > 1e-8) { ns_tx /= ns_len; ns_ty /= ns_len; ns_tz /= ns_len; }
		else               { ns_tx = 0; ns_ty = 0; ns_tz = 1; }
		double out_nx, out_ny, out_nz;
		apply_normal_map(ns_tx, ns_ty, ns_tz,
			rec.normal.x(), rec.normal.y(), rec.normal.z(),
			rec.dpdu.x(),   rec.dpdu.y(),   rec.dpdu.z(),
			out_nx, out_ny, out_nz);
		hit_record r2 = rec;
		r2.normal = vec3(out_nx, out_ny, out_nz);
		return r2;
	}

	color emitted(const ray& r_in, const hit_record& rec,
				  double u, double v, const point3& p) const override {
		return inner->emitted(r_in, apply(rec), u, v, p);
	}
	bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
				 bool do_regularize = false) const override {
		return inner->scatter(r_in, apply(rec), srec, do_regularize);
	}
	double scattering_pdf(const ray& r_in, const hit_record& rec,
						  const ray& scattered) const override {
		return inner->scattering_pdf(r_in, apply(rec), scattered);
	}

	shared_ptr<texture>  get_normal_tex() const { return normal_tex; }
	shared_ptr<material> get_inner()      const { return inner; }

  private:
	shared_ptr<texture>  normal_tex;
	shared_ptr<material> inner;
};


class bump_map_material : public material {
  public:
	bump_map_material(shared_ptr<texture> bump_tex, shared_ptr<material> inner,
					  double scale = 1.0, double step = 0.001)
		: bump_tex(bump_tex), inner(inner), scale(scale), step(step) {}

	// Perturb rec.normal using finite differences of a scalar displacement texture.
	hit_record apply(const hit_record& rec) const {
		auto sample_disp = [&](double u, double v) -> double {
			return scale * bump_tex->value(u, v, rec.p).x();
		};
		double disp   = sample_disp(rec.u,        rec.v);
		double disp_u = sample_disp(rec.u + step,  rec.v);
		double disp_v = sample_disp(rec.u,         rec.v + step);
		vec3 n    = rec.normal;
		vec3 dpdu = rec.dpdu;
		vec3 dpdv = cross(n, dpdu);
		if (dpdv.length_squared() < 1e-12) dpdv = vec3(0, 1, 0);
		double out_nx, out_ny, out_nz;
		apply_bump_map(disp, disp_u, disp_v, step, step,
			n.x(), n.y(), n.z(),
			dpdu.x(), dpdu.y(), dpdu.z(),
			dpdv.x(), dpdv.y(), dpdv.z(),
			out_nx, out_ny, out_nz);
		hit_record r2 = rec;
		r2.normal = vec3(out_nx, out_ny, out_nz);
		return r2;
	}

	color emitted(const ray& r_in, const hit_record& rec,
				  double u, double v, const point3& p) const override {
		return inner->emitted(r_in, apply(rec), u, v, p);
	}
	bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
				 bool do_regularize = false) const override {
		return inner->scatter(r_in, apply(rec), srec, do_regularize);
	}
	double scattering_pdf(const ray& r_in, const hit_record& rec,
						  const ray& scattered) const override {
		return inner->scattering_pdf(r_in, apply(rec), scattered);
	}

	shared_ptr<texture>  get_bump_tex() const { return bump_tex; }
	shared_ptr<material> get_inner()    const { return inner; }
	double get_scale()                  const { return scale; }
	double get_step()                   const { return step; }

  private:
	shared_ptr<texture>  bump_tex;
	shared_ptr<material> inner;
	double scale;
	double step;
};

#endif // MATERIAL_H
