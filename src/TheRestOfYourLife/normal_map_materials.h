#pragma once
// normal_map_material and bump_map_material -- pbrt-v4 NormalMap / BumpMap
//
// Self-contained: including this header is sufficient — it pulls in material.h
// and normal_map.h automatically.  material.h will also include this file once
// it is saved from VS (via the in-memory #include already present there).

#include "material.h"
#include "../shared/normal_map.h"

class normal_map_material : public material {
  public:
	normal_map_material(shared_ptr<texture> normal_tex, shared_ptr<material> inner)
		: normal_tex(normal_tex), inner(inner) {}

	// Perturb rec.normal using the tangent-space RGB normal map texture.
	// Also recomputes rec.dpdu so the shading frame stays consistent with
	// the perturbed normal (pbrt-v4: NormalMap() recomputes dpdu/dpdv via
	// GramSchmidt after the normal is set).
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
		// Recompute dpdu AND dpdv so the shading frame stays orthonormal to
		// the new normal (pbrt-v4 NormalMap, materials.h:
		//   *dpdu = Normalize(GramSchmidt(shading.dpdu, ns)) * ulen;
		//   *dpdv = Normalize(Cross(ns, *dpdu)) * vlen;
		// ). Leaving dpdv at its pre-perturbation value - the bug this fixes -
		// left it non-orthogonal to the new normal and, since GGX materials
		// build their shading frame off dpdu/dpdv, skewed anisotropic-leaning
		// glossy reflections (RoughMetal/Conductor) wherever a normal map
		// combines with them.
		double ulen = rec.dpdu.length();
		double vlen = rec.dpdv.length();
		vec3 ns_w(out_nx, out_ny, out_nz);
		vec3 t = rec.dpdu - dot(rec.dpdu, ns_w) * ns_w;
		if (t.length_squared() > 1e-12)
			r2.dpdu = unit_vector(t) * ulen;
		vec3 new_dpdv = cross(ns_w, r2.dpdu);
		if (new_dpdv.length_squared() > 1e-12)
			r2.dpdv = unit_vector(new_dpdv) * vlen;
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
		// Ray-differential-adaptive step (pbrt-v4 BumpMap, materials.h:116-127):
		// du = 0.5*(|dudx|+|dudy|), matching normal_map.h's apply_bump_map()
		// doc comment, which already documented this as the intended step and
		// was never actually fed it. rec.dudx/dudy are only populated for a
		// primary camera-ray hit (hit_record's own comment) and are 0 for
		// every bounce/shadow/NEE ray, so this falls back to the constructor's
		// fixed `step` there - the same "no footprint info" fallback
		// texture.h's value_diff() already uses, and pbrt's own ".0005f only
		// if zero" fallback. Without this, a fixed step samples a screen-
		// space-constant texture footprint regardless of distance or grazing
		// angle, aliasing into moiré wherever pbrt-v4 stays smooth.
		double eff_step = 0.5 * (std::fabs(rec.dudx) + std::fabs(rec.dudy));
		if (eff_step < 1e-8) eff_step = step;
		auto sample_disp = [&](double u, double v) -> double {
			return scale * bump_tex->value(u, v, rec.p).x();
		};
		double disp   = sample_disp(rec.u,             rec.v);
		double disp_u = sample_disp(rec.u + eff_step,  rec.v);
		double disp_v = sample_disp(rec.u,             rec.v + eff_step);
		vec3 n    = rec.normal;
		vec3 dpdu = rec.dpdu;
		vec3 dpdv = cross(n, dpdu);
		if (dpdv.length_squared() < 1e-12) dpdv = vec3(0, 1, 0);
		double out_nx, out_ny, out_nz;
		apply_bump_map(disp, disp_u, disp_v, eff_step, eff_step,
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
