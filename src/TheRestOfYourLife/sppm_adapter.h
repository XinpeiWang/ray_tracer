#pragma once
//==============================================================================================
// sppm_adapter.h -- Bridges src/shared/sppm.h's duck-typed Scene concept to
// this renderer's real hittable/material/camera types.
//
// src/shared/sppm.h is a complete, unit-tested, pbrt-v4-ported SPPM engine
// (camera pass, photon pass, hash-grid nearest-neighbor lookup, progressive
// radius contraction, final reconstruction) that is header-only and
// deliberately engine-agnostic (no virtual functions, duck-typed Scene
// template parameter, GPU-portable). This file is the CPU-only glue that
// makes it usable against real scenes built from this codebase's `hittable`
// hierarchy and `material` classes -- it belongs in src/TheRestOfYourLife/
// (not src/shared/) for that reason, matching where camera.h and
// power_light_sampler.h already live.
//
// Built up in two layers:
//   1. SPPMShadingContext + free functions (sppm_is_delta_material,
//      sppm_bsdf_f, sppm_bsdf_sample_f) -- the BSDF bridge, independently
//      testable without any scene/world (see tests/unit/sppm_adapter_bsdf_tests.cpp).
//   2. SPPMSceneAdapter -- the full duck-typed Scene implementation, adding
//      Intersect/DirectLight/SampleLightLe/PixelToRay on top of layer 1.
//==============================================================================================

#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "pdf.h"
#include "onb.h"

// ===========================================================================
// Layer 1 -- BSDF bridge
// ===========================================================================

// Captured per-hit shading data, indexed by an opaque integer id.
//
// Why this exists: src/shared/sppm.h's Scene concept methods (BSDFf,
// BSDFSampleF, BSDFPdf) take only (bsdf_id, wo, wi, n) -- no position or
// UV -- yet bsdf_id must remain valid from the camera pass (which records a
// pixel's visible point) through to a LATER photon pass (which evaluates
// BSDFf against that same visible point while shooting unrelated photon
// rays). bsdf_id therefore cannot be a material-type enum; it has to be an
// index into an append-only table of captured contexts, one entry per real
// intersection. SPPMSceneAdapter (layer 2) owns that table; this struct is
// its element type.
struct SPPMShadingContext {
	point3 p;
	vec3   normal;
	double u = 0.0, v = 0.0;
	shared_ptr<material> mat;
};

// True for materials whose scatter() sets srec.skip_pdf=true
// unconditionally: metal, dielectric, rough_metal, rough_dielectric,
// conductor, coated_diffuse, thin_dielectric, coated_conductor (confirmed
// via material_simple.h/material_pbrt.h -- these are the only 8 material
// classes with an unconditional `srec.skip_pdf = true`). SPPM's own
// algorithm (sppm.h's SPPMCameraPass/SPPMPhotonPass) branches on this same
// distinction (hit.is_delta_bsdf) and only ever calls BSDFf/BSDFPdf for
// non-delta hits, treating delta hits via BSDFSampleF resampling alone --
// matching pbrt-v4's own SPPM design.
//
// diffuse_transmission/normalized_fresnel/mix_material are intentionally
// NOT delta (skip_pdf=false) but are also not given full BSDFf support in
// this v1 bridge -- see sppm_bsdf_f's comment.
inline bool sppm_is_delta_material(const material* m) {
	return dynamic_cast<const metal*>(m) != nullptr ||
	       dynamic_cast<const dielectric*>(m) != nullptr ||
	       dynamic_cast<const rough_metal*>(m) != nullptr ||
	       dynamic_cast<const rough_dielectric*>(m) != nullptr ||
	       dynamic_cast<const conductor*>(m) != nullptr ||
	       dynamic_cast<const coated_diffuse*>(m) != nullptr ||
	       dynamic_cast<const thin_dielectric*>(m) != nullptr ||
	       dynamic_cast<const coated_conductor*>(m) != nullptr;
}

// Reconstructs a synthetic hit_record from a captured shading context plus
// a freshly-supplied shading normal `n`. sppm.h always threads its own `n`
// parameter through rather than trusting a stored one, so shading math here
// honors that `n` (expected to match ctx.normal in non-degenerate cases;
// ctx.normal is kept mainly for future diagnostic use). front_face=true
// throughout: `n` is already the shading-facing normal by construction (the
// caller -- ultimately SPPMSceneAdapter::Intersect -- is responsible for
// always supplying a properly outward/ray-facing normal, the same
// convention hit_record::set_face_normal already enforces elsewhere in this
// codebase).
inline hit_record sppm_reconstruct_hit_record(const SPPMShadingContext& ctx, const double n[3]) {
	hit_record rec;
	rec.p = ctx.p;
	rec.normal = vec3(n[0], n[1], n[2]);
	rec.u = ctx.u;
	rec.v = ctx.v;
	rec.mat = ctx.mat;
	rec.front_face = true;
	rec.t = 0.0;
	return rec;
}

// f(wo,wi) for a non-specular material at a captured shading context.
// out[3] = 0 if wi is below the hemisphere, or the material has no
// evaluable BSDF. Callers must not call this for delta materials
// (sppm_is_delta_material() == true) -- SPPM's own algorithm never does,
// since it treats those via sppm_bsdf_sample_f() resampling alone.
//
// Bridges via material::scattering_pdf() (this codebase's convention: it
// returns f(wo,wi)*cos(wi), NOT a normalized pdf -- confirmed by camera.h's
// own NEE code, which computes contribution as
// `srec.attenuation * scattering_pdf(...)`) combined with
// material::scatter()'s srec.attenuation.
//
// v1 is provably exact only for `lambertian`: its attenuation
// (tex->value(u,v,p)) is deterministic and direction-independent, so
// calling scatter() here (for attenuation alone, discarding the direction
// it importance-samples) is safe. diffuse_transmission/normalized_fresnel/
// mix_material are multi-lobe materials whose scatter() stochastically
// commits to one lobe and returns THAT lobe's attenuation -- reusing it for
// an arbitrary externally-supplied wi that might belong to the other lobe
// is not sound in general. Deferred to a later phase with bespoke
// per-material closed-form formulas; not needed for the v1 scene-11
// (CornellRoughGlass) target, which uses only lambertian + a delta sphere
// material.
inline void sppm_bsdf_f(const SPPMShadingContext& ctx, const double wo[3], const double wi[3],
                         const double n[3], double out[3]) {
	out[0] = out[1] = out[2] = 0.0;
	if (!ctx.mat) return;
	double cos_wi = wi[0]*n[0] + wi[1]*n[1] + wi[2]*n[2];
	if (cos_wi <= 0.0) return;

	hit_record rec = sppm_reconstruct_hit_record(ctx, n);
	ray fake_in(ctx.p, -vec3(wo[0], wo[1], wo[2]));
	ray fake_scattered(ctx.p, vec3(wi[0], wi[1], wi[2]));

	double f_cos = ctx.mat->scattering_pdf(fake_in, rec, fake_scattered);
	if (f_cos <= 0.0) return;

	scatter_record srec;
	if (!ctx.mat->scatter(fake_in, rec, srec)) return;

	out[0] = srec.attenuation.x() * f_cos / cos_wi;
	out[1] = srec.attenuation.y() * f_cos / cos_wi;
	out[2] = srec.attenuation.z() * f_cos / cos_wi;
}

// Importance-samples a new direction at a captured shading context, shaped
// exactly as sppm.h's SPPMCameraPass/SPPMPhotonPass expect: both
// unconditionally compute `beta *= f_val*cosI/pdf` with no separate delta
// code path, so for delta materials f_val/pdf must be engineered so that
// product collapses to exactly `attenuation` (the physically-correct
// delta-BSDF limit): pdf=1, f_val=attenuation/cosI. Getting the cosI
// division direction backwards here would silently scale energy by cosI^2
// instead of leaving it untouched -- this is the one formula in the whole
// bridge most worth a dedicated regression test (see
// tests/unit/sppm_adapter_bsdf_tests.cpp).
//
// NOTE: sppm.h's BSDFSampleF signature takes external u1,u2 (implying
// caller-controlled/reproducible randomness, matching pbrt-v4's own
// convention), but material::scatter() always draws its own randomness
// internally via the global random_double() -- there is no way to thread
// caller-supplied random numbers through the existing material interface
// without changing it. u1,u2 are therefore unused here. Functionally this
// is fine (SPPM doesn't need deterministic replay to converge correctly),
// but is an intentional limitation worth calling out so it isn't later
// mistaken for an oversight.
inline bool sppm_bsdf_sample_f(const SPPMShadingContext& ctx, const double wo[3], const double n[3],
                                double /*u1*/, double /*u2*/,
                                double new_dir[3], double f_val[3], double& pdf, bool& is_specular) {
	if (!ctx.mat) return false;
	hit_record rec = sppm_reconstruct_hit_record(ctx, n);
	ray fake_in(ctx.p, -vec3(wo[0], wo[1], wo[2]));

	scatter_record srec;
	if (!ctx.mat->scatter(fake_in, rec, srec)) return false;

	if (srec.skip_pdf) {
		vec3 d = unit_vector(srec.skip_pdf_ray.direction());
		new_dir[0] = d.x(); new_dir[1] = d.y(); new_dir[2] = d.z();
		double cosI = std::fabs(d.x()*n[0] + d.y()*n[1] + d.z()*n[2]);
		cosI = std::max(cosI, 1e-6);
		f_val[0] = srec.attenuation.x() / cosI;
		f_val[1] = srec.attenuation.y() / cosI;
		f_val[2] = srec.attenuation.z() / cosI;
		pdf = 1.0;
		is_specular = true;
		return true;
	}

	// Non-specular: importance-sample via the material's own pdf, then
	// evaluate f through the SAME formula BSDFf uses (sppm_bsdf_f) so the
	// two paths can never drift into inconsistent results.
	vec3 d = srec.pdf_ptr->generate();
	double p = srec.pdf_ptr->value(d);
	if (p <= 0.0) return false;
	vec3 dn = unit_vector(d);
	double wi[3] = { dn.x(), dn.y(), dn.z() };
	double f[3];
	sppm_bsdf_f(ctx, wo, wi, n, f);
	new_dir[0] = dn.x(); new_dir[1] = dn.y(); new_dir[2] = dn.z();
	f_val[0] = f[0]; f_val[1] = f[1]; f_val[2] = f[2];
	pdf = p;
	is_specular = false;
	return true;
}
